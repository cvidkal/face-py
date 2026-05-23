"""YuNet face detector + alignCrop.

用 OpenCV 的 cv2.FaceDetectorYN (内部 ONNX) 做 detect, cv2.FaceRecognizerSF.alignCrop
做 5 点对齐. YuNet 输出顺序按 OpenCV 文档:
  bbox[0:4]   = x, y, w, h
  landmarks   = [(rx, ry), (lx, ly), (nx, ny), (rmx, rmy), (lmx, lmy)]
                 right_eye, left_eye, nose, right_mouth, left_mouth
  score       = float

为啥不直接走 ORT-CUDA 跑 YuNet ONNX (跟 SFace 那边一样)?
- YuNet 模型本身就 3MB, CPU 推理 ~10-15ms, 加 GPU 收益小 (而且要写 multi-stride
  anchor decode + NMS ~ 200 行)
- cv2.FaceDetectorYN 已经把这套 decode + NMS 封好, 一行调用
- 后续 Phase A.4 量速度时如果发现 detector CPU 是 bottleneck, 再换 ORT 自实现
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class DetectedFace:
    bbox_xywh: tuple[float, float, float, float]    # x, y, w, h  (像素)
    landmarks: list[tuple[float, float]]            # 5 个 (x, y), 顺序 right_eye/left_eye/nose/right_mouth/left_mouth
    score: float
    raw_record: np.ndarray = field(default_factory=lambda: np.empty(0))  # 原始 15-d record, 给 alignCrop 用


class FaceDetector:
    """thread-safe wrap around cv2.FaceDetectorYN.

    OpenCV 的 detector 实例本身**不是 thread-safe** — 多线程 detect 同一个 instance
    会乱. 这里没加锁, 由调用层 (pipeline / serve handler) 保证.
    Phase A.2 当前没并发, 单 instance 单线程; 后续如需扩 worker pool 加 RLock.
    """

    def __init__(self, model_path: str,
                  input_size: tuple[int, int] = (640, 640),
                  score_threshold: float = 0.6,
                  nms_threshold: float = 0.3,
                  top_k: int = 5000):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        self._model_path = model_path
        self._input_size = input_size
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """返回所有检到的脸. 失败 / 没脸返空 list, 不抛异常."""
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        # YuNet 用固定输入 size, setInputSize 会让它 resize, faces 坐标已 unscale 回原图
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)
        if faces is None:
            return []
        results: list[DetectedFace] = []
        for rec in faces:
            # rec layout (15 floats): bbox(4) + landmarks(10) + score(1)
            bbox = (float(rec[0]), float(rec[1]), float(rec[2]), float(rec[3]))
            lm = [(float(rec[4 + 2 * i]), float(rec[5 + 2 * i])) for i in range(5)]
            score = float(rec[14])
            results.append(DetectedFace(bbox_xywh=bbox, landmarks=lm,
                                          score=score, raw_record=rec.astype(np.float32)))
        return results


def largest_face(faces: list[DetectedFace]) -> Optional[DetectedFace]:
    """face C++ largest_face_in_collection 等价: 选 bbox 面积最大的脸 (单图主脸)."""
    if not faces:
        return None
    return max(faces, key=lambda f: f.bbox_xywh[2] * f.bbox_xywh[3])


# =============================================================================
# Aligner — cv2.FaceRecognizerSF.alignCrop 包一层
# =============================================================================

class FaceAligner:
    """5-point similarity alignment to 112×112 (SFace input). 用 cv2.FaceRecognizerSF
    内置的 alignCrop, 跟 InsightFace 标准模板一致.

    实现细节: cv2.FaceRecognizerSF 创建时要求传 SFace 模型路径 (会加载用于 feature),
    我们只用 alignCrop 不用 feature, 但 OpenCV API 不允许只加载 alignment. 所以这里
    也持有一个 model_path; 推理由 face-py 自己的 ORT session 跑, 跟 OpenCV 的 SFace
    Net 实例 (在 FaceRecognizerSF 里) 互不影响.
    """

    def __init__(self, sface_model_path: str):
        if not Path(sface_model_path).exists():
            raise FileNotFoundError(f"SFace model not found: {sface_model_path}")
        self._sface = cv2.FaceRecognizerSF.create(sface_model_path, "")

    def align(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """返 112×112 BGR aligned crop, dtype uint8.

        face.raw_record 是 alignCrop 期望的 15-d 向量 (bbox+landmarks+score),
        直接喂回去就行.
        """
        return self._sface.alignCrop(image, face.raw_record)
