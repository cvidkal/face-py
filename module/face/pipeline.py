"""identity_check + compare pipeline composition.

跟 face C++ face_http_server.cpp:4267-4424 (identity_check handler) 1:1 对齐.
错误路径 / clarity_score 出现规则 / response field 行为完全等价 — 详见 CLAUDE.md
"客户契约" 段.

ref_image_path 缓存: face C++ 用 RefFeatureCache 按 stat-key (path + mtime + size) 缓
存 ref embedding. face-py 实现同款 — 单 process 内 dict, mtime+size 失效, 防止
ref 图改了用旧 embedding. 主流程一个 session 8 photos 共享同 ref, 每次重 extract 浪费.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .clarity import compute_image_clarity_score
from .content_gate import compute_content_score, is_photo_unusable
from .detector import FaceAligner, FaceDetector, largest_face
from .errors import (
    COS_INCONCLUSIVE_ZONE, DETECTION_LOW_CONFIDENCE, FACE_TOO_BLURRY, FACE_TOO_SMALL,
    FEATURE_EXTRACTION_FAILED, IMAGE_READ_FAILED, MISMATCH_WITHHELD_LOW_QUALITY,
    MSG_NO_FACE, NO_FACE, PHOTO_UNUSABLE, POSE_EXCESSIVE, REF_IMAGE_READ_FAILED,
    msg_cos_inconclusive_zone, msg_detection_low_confidence, msg_face_too_blurry,
    msg_face_too_small, msg_image_read_failed, msg_mismatch_withheld,
    msg_photo_unusable, msg_pose_excessive, msg_ref_image_read_failed,
)
from .pose_gate import compute_head_pose
from .quality_gate import (
    QualityGateConfig, apply_session_match_consensus,
    session_match_consensus_enabled,
)
from .recognizer import (
    FaceRecognizer, classify_match, cosine_score, get_match_thresholds,
    is_same_person, l2_distance,
)
from .session_consistency import check_internal_consistency, session_prototype


log = logging.getLogger("face-py.pipeline")


def _verdict_with_quality(cos: float, l2: float, quality_flags: list) -> tuple[str, str]:
    """issue #1 的不对称规则: quality 不干净时**只**压制 mismatch, 不压制 match.

    实测 (154 正 / 2000 负): gate 全不拦会把正样本误判 mismatch 从 4 张涨到 19 张 ——
    误报代训正是「诚实 inconclusive」当初要防的最贵的错。只拦 mismatch 方向:
    捞回仍是 31.6%, 假接受仍是 0.80%, 误判 mismatch 回到 4 张。

    返 (match_status, 需要标的 error_code 或空串).
    """
    verdict = classify_match(cos, l2)
    if verdict == "mismatch" and quality_flags:
        return "inconclusive", MISMATCH_WITHHELD_LOW_QUALITY
    return verdict, ""


# =============================================================================
# Result types
# =============================================================================

@dataclass
class IdentityCheckResult:
    """跟 face C++ /api/v1/face/identity_check response 字段对齐 (cosine/l2/clarity
    可为 None, JSON 序列化时变 null).
    """
    face_count: int = 0
    cosine_score: Optional[float] = None
    l2_distance: Optional[float] = None
    clarity_score: Optional[float] = None
    # issue #1: 整张图有没有内容 (挖掉 OSD 后中心区 Laplacian var). 总是产出,
    # 让下游能审计"为什么判 unusable"; content gate 关闭时也照算照给。
    content_score: Optional[float] = None
    photo_unusable: bool = False
    # issue #1: quality gate 降级为审计信号 —— 没拦下判定, 但把「这个结论基于一张什么样
    # 的图」透给客户, 人工复核时能看到。gate 全过 = 空 list。
    # 注意跟 photo_unusable 的区别: 那个是**输入不可用**(整张图没内容, 仍然拦截),
    # 这个是**图能看但质量差**(不拦, 只标记)。见 content_gate.py docstring。
    quality_flags: list = field(default_factory=list)
    det_score: Optional[float] = None
    bbox_min_px: Optional[float] = None
    yaw_ratio: Optional[float] = None
    pitch_ratio: Optional[float] = None
    match_status: str = "inconclusive"  # match / mismatch / inconclusive
    error_code: str = ""
    error: str = ""
    elapsed_ms: float = 0.0

    def to_json(self) -> dict:
        return {
            "face_count": self.face_count,
            "cosine_score": self.cosine_score,
            "l2_distance": self.l2_distance,
            "clarity_score": self.clarity_score,
            "content_score": self.content_score,
            "photo_unusable": self.photo_unusable,
            "quality_flags": list(self.quality_flags),
            "det_score": self.det_score,
            "bbox_min_px": self.bbox_min_px,
            "yaw_ratio": self.yaw_ratio,
            "pitch_ratio": self.pitch_ratio,
            "match_status": self.match_status,
            "error_code": self.error_code,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


@dataclass
class SessionPhotoResult:
    """Per-photo slot inside SessionCheckResult.photo_results."""
    sequence_no: int = 0
    photo_type: str = ""
    face_count: int = 0
    passes_gate: bool = False
    cosine_score: Optional[float] = None       # vs ref, only post-gate
    l2_distance: Optional[float] = None
    clarity_score: Optional[float] = None
    content_score: Optional[float] = None       # issue #1
    photo_unusable: bool = False                # issue #1
    quality_flags: list = field(default_factory=list)   # issue #1
    match_status: str = "inconclusive"          # per-photo, A.5 tri-state
    error_code: str = ""
    error: str = ""
    is_outlier: bool = False                    # Stage 1 flag
    mean_cos_to_peers: Optional[float] = None   # Stage 1 signal

    def to_json(self) -> dict:
        return {
            "sequence_no": self.sequence_no,
            "photo_type": self.photo_type,
            "face_count": self.face_count,
            "passes_gate": self.passes_gate,
            "cosine_score": self.cosine_score,
            "l2_distance": self.l2_distance,
            "clarity_score": self.clarity_score,
            "content_score": self.content_score,
            "photo_unusable": self.photo_unusable,
            "quality_flags": list(self.quality_flags),
            "match_status": self.match_status,
            "error_code": self.error_code,
            "error": self.error,
            "is_outlier": self.is_outlier,
            "mean_cos_to_peers": self.mean_cos_to_peers,
        }


@dataclass
class SessionCheckResult:
    """Phase A.7 — two-stage session-level identity verification.

    Stage 1 (internal_consistency): pairwise cos among post-gate photos.
    Stage 2 (session_cos_to_ref / session_l2_to_ref): prototype vs ref.

    session_status = "match" / "mismatch" / "inconclusive" — the single
    authoritative session-level decision the orchestrator should consume.
    """
    session_status: str = "inconclusive"
    internal_consistency: str = "unknown"   # consistent / inconsistent / single / unknown
    outlier_sequence_nos: list[int] = field(default_factory=list)
    session_cos_to_ref: Optional[float] = None
    session_l2_to_ref: Optional[float] = None
    n_photos: int = 0
    n_post_gate: int = 0
    reason: str = ""
    photo_results: list[SessionPhotoResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_json(self) -> dict:
        return {
            "session_status": self.session_status,
            "internal_consistency": self.internal_consistency,
            "outlier_sequence_nos": self.outlier_sequence_nos,
            "session_cos_to_ref": self.session_cos_to_ref,
            "session_l2_to_ref": self.session_l2_to_ref,
            "n_photos": self.n_photos,
            "n_post_gate": self.n_post_gate,
            "reason": self.reason,
            "photo_results": [p.to_json() for p in self.photo_results],
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


@dataclass
class CompareResult:
    """跟 face C++ /api/v1/face/compare response 对齐."""
    status: str = "ok"               # "ok" | "error"
    cosine_score: Optional[float] = None
    l2_distance: Optional[float] = None
    is_same_person: bool = False
    error: str = ""
    elapsed_ms: float = 0.0

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "cosine_score": self.cosine_score,
            "l2_distance": self.l2_distance,
            "is_same_person": self.is_same_person,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


# =============================================================================
# Ref embedding cache
# =============================================================================

@dataclass(frozen=True)
class _RefCacheKey:
    path: str
    mtime_ns: int
    size: int


class RefFeatureCache:
    """跟 face C++ RefFeatureCache 同款语义: path + mtime + size 作 key, 任一变化失效."""

    def __init__(self, max_entries: int = 64):
        self._max = max_entries
        self._cache: dict[_RefCacheKey, np.ndarray] = {}
        self._lock = threading.Lock()

    @staticmethod
    def make_key(path: str) -> Optional[_RefCacheKey]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return _RefCacheKey(path=path, mtime_ns=st.st_mtime_ns, size=st.st_size)

    def get(self, key: _RefCacheKey) -> Optional[np.ndarray]:
        with self._lock:
            return self._cache.get(key)

    def put(self, key: _RefCacheKey, emb: np.ndarray) -> None:
        with self._lock:
            if len(self._cache) >= self._max:
                # 简单 FIFO 淘汰 (dict 保持插入顺序)
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = emb.copy()


# =============================================================================
# Pipeline service
# =============================================================================

@dataclass
class FacePipeline:
    detector: FaceDetector
    aligner: FaceAligner
    recognizer: FaceRecognizer
    quality: QualityGateConfig = field(default_factory=QualityGateConfig.from_env)
    ref_cache: RefFeatureCache = field(default_factory=RefFeatureCache)

    # ----- identity_check -----

    def identity_check(self, image_path: str, ref_image_path: str) -> IdentityCheckResult:
        """Phase A.5 redesign: 显式 quality gates + cos 中间区, 拒绝在不可信数据上猜.

        路径 (按 cost 升序排, gate 早返省 align/extract):
        1. imread photo → IMAGE_READ_FAILED
        2. imread ref (或 cache hit, 跳 IO) → REF_IMAGE_READ_FAILED
        3. detect photo → 空 → NO_FACE
        4. det_score < 0.88 → DETECTION_LOW_CONFIDENCE (新, Phase A.5)
        5. bbox 小于 40px → FACE_TOO_SMALL (新)
        6. pose excessive → POSE_EXCESSIVE
        7. align (~3ms)
        8. aligned crop clarity < 30 → FACE_TOO_BLURRY (新)
        9. extract embedding (~2ms GPU)
        10. extract ref embedding (cache 复用)
        11. cos / l2 三态 classify:
            cos < 0.15                → mismatch
            cos >= 0.30 AND l2 <= 1.15 → match
            else                       → COS_INCONCLUSIVE_ZONE (新)

        全部 gate / inconclusive 路径 match_status="inconclusive", error_code 区分.
        TA 的 _populate_identity 只看 mismatch 才 trigger has_identity_anomaly, 所以
        把 "不确定" 都归 inconclusive 是业务安全的.
        """
        result = IdentityCheckResult()
        t0 = time.perf_counter()
        quality = self.quality

        try:
            # 1. 读 photo
            image = cv2.imread(image_path)
            if image is None or image.size == 0:
                result.error_code = IMAGE_READ_FAILED
                result.error = msg_image_read_failed(image_path)
                return self._finish(result, t0)

            # face C++ 在 imread 成功后立刻算 image-level clarity, 但仅 error_code
            # 路径 emit. 我们改为只在 face-too-blurry 时算 aligned-crop clarity (后面),
            # image-level clarity 一般 dashcam 偏暗也大 — 没区分度. 这里不算.

            # 1.5 content gate (issue #1): 整张图有没有内容. 放在 detect **之前** —
            # 一张全黑图走到 detect 只会得到 no_face, 那个 error_code 说的是"没找到脸",
            # 掩盖了真正的原因("压根没拍到东西"). 先判 content 才能给出准确的理由.
            # 亚毫秒级 (一次 Laplacian), 不值得为省这点开销放到后面.
            result.content_score = compute_content_score(image)
            if is_photo_unusable(result.content_score, quality.photo_content_min):
                result.photo_unusable = True
                result.error_code = PHOTO_UNUSABLE
                result.error = msg_photo_unusable(
                    result.content_score or 0.0, quality.photo_content_min)
                return self._finish(result, t0)

            # 2. 读 ref (或 cache hit)
            cache_key = RefFeatureCache.make_key(ref_image_path)
            cached_ref = self.ref_cache.get(cache_key) if cache_key else None
            ref_image: Optional[np.ndarray] = None
            if cached_ref is None:
                ref_image = cv2.imread(ref_image_path)
                if ref_image is None or ref_image.size == 0:
                    result.error_code = REF_IMAGE_READ_FAILED
                    result.error = msg_ref_image_read_failed(ref_image_path)
                    return self._finish(result, t0)

            # 3. detect photo (detector 默认 score_threshold=0.3 较松, 让我们看到所有
            # 候选, gate 决策在下面)
            faces = self.detector.detect(image)
            result.face_count = len(faces)
            primary = largest_face(faces)
            if primary is None:
                result.error_code = NO_FACE
                result.error = MSG_NO_FACE
                return self._finish(result, t0)

            # 4-6. quality gates (Phase A.5) — issue #1 起默认只标记不拦截.
            # block 模式 (FACE_QUALITY_GATE_MODE=block) 恢复历史的"直接 inconclusive".
            bbox_min_side = min(primary.bbox_xywh[2], primary.bbox_xywh[3])
            pose = compute_head_pose(primary.landmarks)
            result.det_score = float(primary.score)
            result.bbox_min_px = float(bbox_min_side)
            result.yaw_ratio = float(pose.yaw_ratio)
            result.pitch_ratio = float(pose.pitch_ratio)
            for code, failed, msg in (
                (DETECTION_LOW_CONFIDENCE, primary.score < quality.det_score_min,
                 lambda: msg_detection_low_confidence(primary.score, quality.det_score_min)),
                (FACE_TOO_SMALL, bbox_min_side < quality.face_bbox_min_px,
                 lambda: msg_face_too_small(bbox_min_side, quality.face_bbox_min_px)),
                (POSE_EXCESSIVE, (abs(pose.yaw_ratio) > quality.yaw_max
                                  or abs(pose.pitch_ratio) > quality.pitch_max),
                 lambda: msg_pose_excessive(pose.yaw_ratio, pose.pitch_ratio)),
            ):
                if not failed:
                    continue
                result.quality_flags.append(code)
                if quality.blocks:
                    result.error_code = code
                    result.error = msg()
                    return self._finish(result, t0)

            # 7. align (~3ms) + 8. aligned-crop clarity gate
            try:
                aligned = self.aligner.align(image, primary)
            except Exception as exc:
                result.error_code = FEATURE_EXTRACTION_FAILED
                result.error = str(exc)
                return self._finish(result, t0)
            crop_clarity = compute_image_clarity_score(aligned)
            result.clarity_score = crop_clarity
            if crop_clarity is None or crop_clarity < quality.face_crop_clarity_min:
                result.quality_flags.append(FACE_TOO_BLURRY)
                if quality.blocks:
                    result.error_code = FACE_TOO_BLURRY
                    result.error = msg_face_too_blurry(
                        crop_clarity or 0.0, quality.face_crop_clarity_min)
                    return self._finish(result, t0)

            # 9. extract embedding
            try:
                emb_photo = self.recognizer.extract(aligned)
            except Exception as exc:
                result.error_code = FEATURE_EXTRACTION_FAILED
                result.error = str(exc)
                result.clarity_score = crop_clarity
                return self._finish(result, t0)

            # 10. extract ref embedding (或 cache 复用). Ref 也走同样 gate — 如果 ref
            # 图本身就糟, 直接 feature_extraction_failed (cache 不入)
            if cached_ref is not None:
                emb_ref = cached_ref
            else:
                assert ref_image is not None
                try:
                    ref_faces = self.detector.detect(ref_image)
                    ref_primary = largest_face(ref_faces)
                    if ref_primary is None:
                        raise RuntimeError("No face detected in reference image")
                    # ref 不走严格 det / size / clarity gate — 客户已经选了 ref 给我们,
                    # 这是"权威照片", 不该挑剔. 仅 align + extract.
                    ref_aligned = self.aligner.align(ref_image, ref_primary)
                    emb_ref = self.recognizer.extract(ref_aligned)
                except Exception as exc:
                    result.error_code = FEATURE_EXTRACTION_FAILED
                    result.error = str(exc)
                    return self._finish(result, t0)
                if cache_key:
                    self.ref_cache.put(cache_key, emb_ref)

            # 11. 算 cos / l2 + tri-state classify (Phase A.5)
            cos = cosine_score(emb_photo, emb_ref)
            l2 = l2_distance(emb_photo, emb_ref)
            result.cosine_score = cos
            result.l2_distance = l2
            verdict, withheld = _verdict_with_quality(cos, l2, result.quality_flags)
            result.match_status = verdict
            if withheld:
                result.error_code = withheld
                result.error = msg_mismatch_withheld(cos, result.quality_flags)
            elif verdict == "inconclusive":
                cos_lo, cos_hi, _ = get_match_thresholds()
                result.error_code = COS_INCONCLUSIVE_ZONE
                result.error = msg_cos_inconclusive_zone(cos, cos_lo, cos_hi)
            return self._finish(result, t0)

        except Exception as exc:
            # 兜底
            result.error_code = FEATURE_EXTRACTION_FAILED
            result.error = str(exc)
            return self._finish(result, t0)

    # ----- session_check (Phase A.7 two-stage) -----

    def session_check(self, ref_image_path: str,
                       photos: list[dict]) -> "SessionCheckResult":
        """Two-stage session-level identity verification.

        photos: list of dicts with keys 'sequence_no', 'photo_type', 'image_path'.

        Stage 1 — internal consistency: pairwise cos among post-gate photo
        embeddings. Any photo whose mean cos to peers < threshold = outlier;
        any outlier → session is 'inconsistent' → session_status='mismatch'
        (代训信号), regardless of Stage 2.

        Stage 2 — prototype vs ref: only the consistent core's mean embedding
        (L2-normalized) is compared to ref, then classified with A.5 tri-state
        thresholds. Single post-gate photo: skip Stage 1, run Stage 2 directly
        on that one embedding. Zero post-gate photos: inconclusive.

        Outlier session also reports session_cos_to_ref of the majority core
        (excluding outliers) as a diagnostic — does not change session_status.
        """
        result = SessionCheckResult()
        result.n_photos = len(photos)
        t0 = time.perf_counter()

        try:
            # ----- 1. Resolve ref embedding (via cache) -----
            cache_key = RefFeatureCache.make_key(ref_image_path)
            emb_ref = self.ref_cache.get(cache_key) if cache_key else None
            if emb_ref is None:
                ref_img = cv2.imread(ref_image_path)
                if ref_img is None or ref_img.size == 0:
                    result.session_status = "inconclusive"
                    result.reason = msg_ref_image_read_failed(ref_image_path)
                    return self._finish_session(result, t0)
                try:
                    ref_faces = self.detector.detect(ref_img)
                    ref_primary = largest_face(ref_faces)
                    if ref_primary is None:
                        raise RuntimeError("No face detected in reference image")
                    ref_aligned = self.aligner.align(ref_img, ref_primary)
                    emb_ref = self.recognizer.extract(ref_aligned)
                except Exception as exc:
                    result.session_status = "inconclusive"
                    result.reason = f"ref extraction failed: {exc}"
                    return self._finish_session(result, t0)
                if cache_key:
                    self.ref_cache.put(cache_key, emb_ref)

            # ----- 2. Per-photo: detect + gate + extract -----
            embeddings: list[np.ndarray] = []          # post-gate only
            emb_to_photo_idx: list[int] = []           # parallel to embeddings
            for p in photos:
                pr = self._process_photo_for_session(p)
                # issue #1: 只要抽出了 embedding 就给这张照片自己的 cos/判定 —— 哪怕
                # 带 quality_flags。但**只有 passes_gate 的进聚类** (embeddings),
                # 因为低质量 embedding 对 Stage 1 的影响没有实测数据支撑。
                emb = getattr(pr, "_embedding", None)
                if emb is not None:
                    cos_v = cosine_score(emb, emb_ref)
                    l2_v = float(l2_distance(emb, emb_ref))
                    pr.cosine_score = cos_v
                    pr.l2_distance = l2_v
                    pr.match_status, withheld = _verdict_with_quality(
                        cos_v, l2_v, pr.quality_flags)
                    if withheld and not pr.error_code:
                        pr.error_code = withheld
                        pr.error = msg_mismatch_withheld(cos_v, pr.quality_flags)
                    if pr.passes_gate:
                        embeddings.append(emb)
                        emb_to_photo_idx.append(len(result.photo_results))
                # Drop the private attr before serialization
                if hasattr(pr, "_embedding"):
                    delattr(pr, "_embedding")
                result.photo_results.append(pr)

            result.n_post_gate = len(embeddings)

            # ----- 3. Stage 1 + Stage 2 decision -----
            if result.n_post_gate == 0:
                result.session_status = "inconclusive"
                result.internal_consistency = "unknown"
                result.reason = "no post-gate photos"
                return self._finish_session(result, t0)

            if result.n_post_gate == 1:
                # Skip Stage 1, run Stage 2 on single photo
                emb = embeddings[0]
                cos_v = cosine_score(emb, emb_ref)
                l2_v = float(l2_distance(emb, emb_ref))
                result.session_cos_to_ref = cos_v
                result.session_l2_to_ref = l2_v
                result.session_status = classify_match(cos_v, l2_v)
                result.internal_consistency = "single"
                result.reason = (f"single post-gate photo, "
                                  f"cos={cos_v:.3f} l2={l2_v:.3f}")
                return self._finish_session(result, t0)

            # Stage 1
            consistency = check_internal_consistency(embeddings)
            for emb_idx, photo_idx in enumerate(emb_to_photo_idx):
                result.photo_results[photo_idx].mean_cos_to_peers = round(
                    consistency.mean_cos_per_index[emb_idx], 4)
                if emb_idx in consistency.outlier_indices:
                    result.photo_results[photo_idx].is_outlier = True
                    result.outlier_sequence_nos.append(
                        result.photo_results[photo_idx].sequence_no)

            if not consistency.is_consistent:
                # 内部不一致 → 代训
                result.internal_consistency = "inconsistent"
                # diagnostic: prototype of majority core vs ref
                core = [embeddings[i] for i in range(len(embeddings))
                         if i not in consistency.outlier_indices]
                if core:
                    proto = session_prototype(core)
                    result.session_cos_to_ref = cosine_score(proto, emb_ref)
                    result.session_l2_to_ref = float(l2_distance(proto, emb_ref))
                result.session_status = "mismatch"
                result.reason = (f"Stage 1: {len(consistency.outlier_indices)} outlier(s) "
                                  f"mean_cos<{consistency.threshold}")
                return self._finish_session(result, t0)

            # Stage 2 — prototype of all post-gate photos vs ref
            result.internal_consistency = "consistent"
            proto = session_prototype(embeddings)
            cos_v = cosine_score(proto, emb_ref)
            l2_v = float(l2_distance(proto, emb_ref))
            result.session_cos_to_ref = cos_v
            result.session_l2_to_ref = l2_v
            result.session_status = classify_match(cos_v, l2_v)
            result.reason = (f"Stage 2: prototype cos={cos_v:.3f} l2={l2_v:.3f}")
            if session_match_consensus_enabled():
                apply_session_match_consensus(result.photo_results)
            return self._finish_session(result, t0)

        except Exception as exc:
            result.session_status = "inconclusive"
            result.reason = f"unexpected error: {exc}"
            return self._finish_session(result, t0)

    def _process_photo_for_session(self, p: dict) -> "SessionPhotoResult":
        """Run gates + (if passing) extract embedding for one photo.

        Returns SessionPhotoResult with passes_gate / error_code set. If gate
        passes, attaches embedding as private attr _embedding for caller use.
        Caller is responsible for cos/l2 vs ref + setting match_status / cleanup.
        """
        pr = SessionPhotoResult(
            sequence_no=int(p.get("sequence_no", 0)),
            photo_type=str(p.get("photo_type", "")),
        )
        image_path = str(p.get("image_path", ""))
        if not image_path:
            pr.error_code = IMAGE_READ_FAILED
            pr.error = "image_path is empty"
            return pr
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            pr.error_code = IMAGE_READ_FAILED
            pr.error = msg_image_read_failed(image_path)
            return pr
        gate = self.quality
        # content gate 先于 detect — 理由同 identity_check (issue #1)
        pr.content_score = compute_content_score(image)
        if is_photo_unusable(pr.content_score, gate.photo_content_min):
            pr.photo_unusable = True
            pr.error_code = PHOTO_UNUSABLE
            pr.error = msg_photo_unusable(pr.content_score or 0.0, gate.photo_content_min)
            return pr
        faces = self.detector.detect(image)
        pr.face_count = len(faces)
        primary = largest_face(faces)
        if primary is None:
            pr.error_code = NO_FACE
            pr.error = MSG_NO_FACE
            return pr
        bbox_min_side = min(primary.bbox_xywh[2], primary.bbox_xywh[3])
        pose = compute_head_pose(primary.landmarks)
        # issue #1: audit 模式下 gate 只标记不拦截 —— 但 **passes_gate 的语义不变**,
        # 它继续决定哪些 embedding 进 Stage 1 聚类 / Stage 2 prototype。
        # 理由: 实测只覆盖「照片级判定」, 把低质量 embedding 放进聚类是**没测过**的,
        # 噪声 embedding 可能造出假 outlier → 课次级误报代训 (最贵的那种错).
        for code, failed, msg in (
            (DETECTION_LOW_CONFIDENCE, primary.score < gate.det_score_min,
             lambda: msg_detection_low_confidence(primary.score, gate.det_score_min)),
            (FACE_TOO_SMALL, bbox_min_side < gate.face_bbox_min_px,
             lambda: msg_face_too_small(bbox_min_side, gate.face_bbox_min_px)),
            (POSE_EXCESSIVE, (abs(pose.yaw_ratio) > gate.yaw_max
                              or abs(pose.pitch_ratio) > gate.pitch_max),
             lambda: msg_pose_excessive(pose.yaw_ratio, pose.pitch_ratio)),
        ):
            if not failed:
                continue
            pr.quality_flags.append(code)
            if gate.blocks:
                pr.error_code = code
                pr.error = msg()
                return pr
        try:
            aligned = self.aligner.align(image, primary)
        except Exception as exc:
            pr.error_code = FEATURE_EXTRACTION_FAILED
            pr.error = str(exc)
            return pr
        clarity = compute_image_clarity_score(aligned)
        pr.clarity_score = float(clarity) if clarity is not None else None
        if clarity is None or clarity < gate.face_crop_clarity_min:
            pr.quality_flags.append(FACE_TOO_BLURRY)
            if gate.blocks:
                pr.error_code = FACE_TOO_BLURRY
                pr.error = msg_face_too_blurry(clarity or 0.0, gate.face_crop_clarity_min)
                return pr
        try:
            emb = self.recognizer.extract(aligned)
        except Exception as exc:
            pr.error_code = FEATURE_EXTRACTION_FAILED
            pr.error = str(exc)
            return pr
        # 质量全过才算 post-gate (进聚类); 有 flag 的照片仍会拿到自己的 cos/判定
        pr.passes_gate = not pr.quality_flags
        # Attach embedding for caller — stripped before serialization
        pr._embedding = emb  # type: ignore[attr-defined]
        return pr

    @staticmethod
    def _finish_session(result: "SessionCheckResult", t0: float) -> "SessionCheckResult":
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result

    # ----- compare (两张图直接比, 无 ref cache, 无 pose gate) -----

    def compare(self, image_a_path: str, image_b_path: str) -> CompareResult:
        """两张图直接比. face C++ /face/compare 不做 pose gate (line ~4099 注释)."""
        result = CompareResult()
        t0 = time.perf_counter()
        try:
            img_a = cv2.imread(image_a_path)
            img_b = cv2.imread(image_b_path)
            if img_a is None or img_a.size == 0:
                raise RuntimeError(msg_image_read_failed(image_a_path))
            if img_b is None or img_b.size == 0:
                raise RuntimeError(msg_image_read_failed(image_b_path))
            emb_a = self._extract_or_raise(img_a)
            emb_b = self._extract_or_raise(img_b)
            cos = cosine_score(emb_a, emb_b)
            l2 = l2_distance(emb_a, emb_b)
            result.cosine_score = cos
            result.l2_distance = l2
            result.is_same_person = is_same_person(cos, l2)
            result.status = "ok"
        except Exception as exc:
            result.status = "error"
            result.error = str(exc)
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def _extract_or_raise(self, image: np.ndarray) -> np.ndarray:
        faces = self.detector.detect(image)
        primary = largest_face(faces)
        if primary is None:
            raise RuntimeError(MSG_NO_FACE)
        aligned = self.aligner.align(image, primary)
        return self.recognizer.extract(aligned)

    # ----- helpers -----

    @staticmethod
    def _finish(result: IdentityCheckResult, t0: float) -> IdentityCheckResult:
        result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result


# =============================================================================
# Factory
# =============================================================================

def build_pipeline_from_env() -> FacePipeline:
    detect_path = os.environ.get(
        "FACE_DETECT_MODEL_PATH", "models/face_detection_yunet_2023mar.onnx")
    recognize_path = os.environ.get(
        "FACE_RECOGNIZE_MODEL_PATH", "models/face_recognition_sface_2021dec.onnx")
    device = os.environ.get("FACE_DEVICE", "cuda")
    detector = FaceDetector(detect_path)
    aligner = FaceAligner(recognize_path)
    recognizer = FaceRecognizer(recognize_path, device=device)
    return FacePipeline(detector=detector, aligner=aligner, recognizer=recognizer)
