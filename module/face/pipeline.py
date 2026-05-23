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
    FEATURE_EXTRACTION_FAILED, IMAGE_READ_FAILED, MSG_NO_FACE, NO_FACE,
    POSE_EXCESSIVE, REF_IMAGE_READ_FAILED,
    msg_image_read_failed, msg_pose_excessive, msg_ref_image_read_failed,
)
from .pose_gate import compute_head_pose, is_pose_excessive
from .recognizer import (
    FaceRecognizer, cosine_score, is_same_person, l2_distance,
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
    ref_cache: RefFeatureCache = field(default_factory=RefFeatureCache)

    # ----- identity_check -----

    def identity_check(self, image_path: str, ref_image_path: str) -> IdentityCheckResult:
        """跟 face C++ /face/identity_check handler 1:1 对齐.

        早返路径 (按 face C++ 顺序):
        1. cv2.imread(image_path) → None → IMAGE_READ_FAILED, clarity=None
        2. cv2.imread(ref_image_path) → None → REF_IMAGE_READ_FAILED, clarity=已算
        3. detect → 空 → NO_FACE, clarity=已算 (但 face C++ no_face 路径**不**填 clarity,
           按那个行为对齐: success / no_face / pose_excessive 三路径 clarity 不填)
        4. pose gate → 超阈值 → POSE_EXCESSIVE, clarity 不填
        5. extract ref + 算 cos/l2 + match_status

        catch-all: extract / align 抛异常 → FEATURE_EXTRACTION_FAILED, clarity 填
        """
        result = IdentityCheckResult()
        t0 = time.perf_counter()

        try:
            # 1. 读 photo
            image = cv2.imread(image_path)
            if image is None or image.size == 0:
                result.error_code = IMAGE_READ_FAILED
                result.error = msg_image_read_failed(image_path)
                # face C++ image_read_failed 路径**填** clarity (无图所以 clarity=None,
                # 但 schema 里 emit None). 这里跟 face C++ 行为对齐: clarity=None
                return self._finish(result, t0)

            # face C++ 在 imread 成功后立刻算 clarity, 但仅在错误路径 emit. 我们
            # 也把它算出来留着, success 路径 to_json 时 clarity 仍 None.
            clarity = compute_image_clarity_score(image)

            # 2. 读 ref (在 detect 前, 跟 face C++ 顺序一致, 防止 cache miss 时再 imread)
            cache_key = RefFeatureCache.make_key(ref_image_path)
            cached_ref = self.ref_cache.get(cache_key) if cache_key else None
            ref_image: Optional[np.ndarray] = None
            if cached_ref is None:
                ref_image = cv2.imread(ref_image_path)
                if ref_image is None or ref_image.size == 0:
                    result.error_code = REF_IMAGE_READ_FAILED
                    result.error = msg_ref_image_read_failed(ref_image_path)
                    # face C++ 这一路径 clarity (photo) 已算出但不 emit (它先 set 了再
                    # 直接 return). 我们保持一致: clarity 不填.
                    return self._finish(result, t0)

            # 3. detect photo
            faces = self.detector.detect(image)
            result.face_count = len(faces)
            primary = largest_face(faces)
            if primary is None:
                result.error_code = NO_FACE
                result.error = MSG_NO_FACE
                # face C++ no_face 不填 clarity (line 4359-4363 直接 return without
                # setting clarity_score). 我们对齐.
                return self._finish(result, t0)

            # 4. align + extract photo embedding, 顺便算 pose
            try:
                aligned = self.aligner.align(image, primary)
                emb_photo = self.recognizer.extract(aligned)
            except Exception as exc:
                result.error_code = FEATURE_EXTRACTION_FAILED
                result.error = str(exc)
                # face C++ catch 路径填 clarity
                result.clarity_score = clarity
                return self._finish(result, t0)

            pose = compute_head_pose(primary.landmarks)
            if is_pose_excessive(pose):
                result.error_code = POSE_EXCESSIVE
                result.error = msg_pose_excessive(pose.yaw_ratio, pose.pitch_ratio)
                # face C++ pose_excessive 路径 line 4373-4382, **不**填 clarity.
                return self._finish(result, t0)

            # 5. extract ref embedding (或用 cache)
            if cached_ref is not None:
                emb_ref = cached_ref
            else:
                assert ref_image is not None
                try:
                    ref_faces = self.detector.detect(ref_image)
                    ref_primary = largest_face(ref_faces)
                    if ref_primary is None:
                        # face C++ extract_primary 内部 no_face 会抛, 经 classify
                        # 转 feature_extraction_failed. 这里跟相同语义.
                        raise RuntimeError("No face detected in reference image")
                    ref_aligned = self.aligner.align(ref_image, ref_primary)
                    emb_ref = self.recognizer.extract(ref_aligned)
                except Exception as exc:
                    result.error_code = FEATURE_EXTRACTION_FAILED
                    result.error = str(exc)
                    result.clarity_score = clarity
                    return self._finish(result, t0)
                # 入 cache (仅成功路径才 put — face C++ ref_feature_cache.put 也是
                # 成功路径才入, line 4391)
                if cache_key:
                    self.ref_cache.put(cache_key, emb_ref)

            # 6. 算 cos / l2 / match_status
            cos = cosine_score(emb_photo, emb_ref)
            l2 = l2_distance(emb_photo, emb_ref)
            result.cosine_score = cos
            result.l2_distance = l2
            result.match_status = "match" if is_same_person(cos, l2) else "mismatch"
            return self._finish(result, t0)

        except Exception as exc:
            # 兜底 — 任何没预期的异常都按 feature_extraction_failed 处理, 跟 face C++
            # catch 路径一致. 注意此处不重复算 clarity (上面如果到这, clarity 在
            # try 块里已经算过或没算过, 我们用本地变量已不可达 — 重新算一次).
            result.error_code = FEATURE_EXTRACTION_FAILED
            result.error = str(exc)
            try:
                # 重新 imread 一次拿 clarity (face C++ classify_image_validation_failure
                # 在 catch 块里有 image 引用, 我们这里不一定有)
                fallback_image = cv2.imread(image_path)
                if fallback_image is not None:
                    result.clarity_score = compute_image_clarity_score(fallback_image)
            except Exception:
                pass
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
