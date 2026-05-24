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
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .clarity import compute_image_clarity_score
from .detector import DetectedFace, FaceAligner, FaceDetector, largest_face
from .errors import (
    COS_INCONCLUSIVE_ZONE, DETECTION_LOW_CONFIDENCE, FACE_TOO_BLURRY, FACE_TOO_SMALL,
    FEATURE_EXTRACTION_FAILED, IMAGE_READ_FAILED, MSG_NO_FACE, NO_FACE,
    POSE_EXCESSIVE, REF_IMAGE_READ_FAILED,
    msg_cos_inconclusive_zone, msg_detection_low_confidence, msg_face_too_blurry,
    msg_face_too_small, msg_image_read_failed, msg_pose_excessive,
    msg_ref_image_read_failed,
)
from .pose_gate import compute_head_pose
from .quality_gate import QualityGateConfig
from .recognizer import (
    FaceRecognizer, classify_match, cosine_score, get_match_thresholds,
    is_same_person, l2_distance,
)


log = logging.getLogger("face-py.pipeline")


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
            "match_status": self.match_status,
            "error_code": self.error_code,
            "error": self.error,
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

            # 4. detection confidence gate (Phase A.5)
            if primary.score < quality.det_score_min:
                result.error_code = DETECTION_LOW_CONFIDENCE
                result.error = msg_detection_low_confidence(
                    primary.score, quality.det_score_min)
                return self._finish(result, t0)

            # 5. face size gate (Phase A.5)
            bbox_min_side = min(primary.bbox_xywh[2], primary.bbox_xywh[3])
            if bbox_min_side < quality.face_bbox_min_px:
                result.error_code = FACE_TOO_SMALL
                result.error = msg_face_too_small(
                    bbox_min_side, quality.face_bbox_min_px)
                return self._finish(result, t0)

            # 6. pose gate (cheap, 算几何不用 inference)
            pose = compute_head_pose(primary.landmarks)
            if (abs(pose.yaw_ratio) > quality.yaw_max or
                    abs(pose.pitch_ratio) > quality.pitch_max):
                result.error_code = POSE_EXCESSIVE
                result.error = msg_pose_excessive(pose.yaw_ratio, pose.pitch_ratio)
                return self._finish(result, t0)

            # 7. align (~3ms) + 8. aligned-crop clarity gate
            try:
                aligned = self.aligner.align(image, primary)
            except Exception as exc:
                result.error_code = FEATURE_EXTRACTION_FAILED
                result.error = str(exc)
                return self._finish(result, t0)
            crop_clarity = compute_image_clarity_score(aligned)
            if crop_clarity is None or crop_clarity < quality.face_crop_clarity_min:
                result.error_code = FACE_TOO_BLURRY
                result.error = msg_face_too_blurry(
                    crop_clarity or 0.0, quality.face_crop_clarity_min)
                result.clarity_score = crop_clarity
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
            verdict = classify_match(cos, l2)
            result.match_status = verdict
            if verdict == "inconclusive":
                cos_lo, cos_hi, _ = get_match_thresholds()
                result.error_code = COS_INCONCLUSIVE_ZONE
                result.error = msg_cos_inconclusive_zone(cos, cos_lo, cos_hi)
            return self._finish(result, t0)

        except Exception as exc:
            # 兜底
            result.error_code = FEATURE_EXTRACTION_FAILED
            result.error = str(exc)
            return self._finish(result, t0)

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
