"""SFace face recognizer via ONNX Runtime (CUDA EP preferred, CPU fallback).

模型: face_recognition_sface_2021dec.onnx (OpenCV Zoo)
输入: 1×3×112×112 BGR (注: SFace 用 BGR, **不**像 ArcFace 用 RGB; 验证过 cv2.FaceRecognizerSF.feature)
预处理: 像素值 0-255 转 float32, channel-first NCHW, 无 mean/std 归一化
输出: 1×128 嵌入向量, **未** L2 归一化 (我们自己 L2 归一化, 跟 face C++ FaceRecognizer line 195-200 一致)

Feasibility 实测: ORT-CUDA T4 mean 1.41ms / inference (jiapei-anticheat#20 Session 1).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import onnxruntime as ort


log = logging.getLogger("face-py.recognizer")

# SFace 输入是固定 1×3×112×112, 静态 shape, 给 ORT 优化器最多空间
_SFACE_INPUT_NAME = "data"
_SFACE_OUTPUT_NAME = "fc1"
_SFACE_INPUT_SIZE = 112


def _make_session(model_path: str, device: str) -> ort.InferenceSession:
    """优先 CUDA, fallback CPU. device='cuda'/'cpu', 别的值视作 cpu."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 1  # GPU 推理 CPU thread 1 就够
    so.log_severity_level = 3    # 抑制 SFace ONNX 的 initializer-as-input warning 噪声

    providers: list[tuple[str, dict] | str] = []
    want_cuda = device.lower() == "cuda"
    available = set(ort.get_available_providers())
    if want_cuda and "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }))
    elif want_cuda:
        log.warning("FACE_DEVICE=cuda 但 CUDAExecutionProvider 不可用, fallback CPU. "
                     "available=%s", sorted(available))
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


class FaceRecognizer:
    """SFace ONNX 包装. 一个 instance per process, thread-safe (内部 Lock)."""

    def __init__(self, model_path: str, device: str = "cuda"):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"SFace model not found: {model_path}")
        self._model_path = model_path
        self._device = device
        self._session = _make_session(model_path, device)
        self._lock = Lock()
        # 报告实际跑的 provider 给运行日志确认 CUDA 路径有没有生效
        actual = self._session.get_providers()
        log.info("FaceRecognizer ready: model=%s providers=%s", model_path, actual)
        self.providers = actual

    def extract(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        """112×112 BGR uint8 → 128-d L2-normalized embedding (float32, shape (128,))."""
        if aligned_bgr_112.shape != (_SFACE_INPUT_SIZE, _SFACE_INPUT_SIZE, 3):
            raise ValueError(
                f"expected 112×112×3 aligned BGR, got shape {aligned_bgr_112.shape}")
        # NHWC uint8 → NCHW float32, 0-255 范围 (SFace 不要 mean/std 归一)
        x = aligned_bgr_112.astype(np.float32)
        x = np.transpose(x, (2, 0, 1))[None, ...]   # 1×3×112×112
        with self._lock:
            out = self._session.run([_SFACE_OUTPUT_NAME],
                                      {_SFACE_INPUT_NAME: x})[0]
        emb = out.reshape(-1).astype(np.float32)
        # L2 归一化, 跟 face C++ face_recognizer.cpp line 195-200 一致, 让 cosine 直接 dot
        norm = float(np.linalg.norm(emb))
        if norm > 1e-6:
            emb = emb / norm
        return emb


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """L2-normalized a, b 的 cos = dot. 跟 face C++ FaceRecognizer::match line 230-244 等价."""
    return float(np.dot(a, b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# 业务阈值, 跟 face C++ 同名同默认 (但因为换了模型, 实际 v0.4.0 会重 tune).
def get_match_thresholds() -> tuple[float, float]:
    """返 (cosine_threshold, l2_threshold). 默认 0.4 / 1.0 跟 face C++ 一致.

    SFace 跟 face-reidentification-retail-0095 是不同模型, cos 分布**大概率**不一样,
    Phase A.4 重测后这俩默认值要重选. 当前先放 face C++ 默认占位.
    """
    cos = float(os.environ.get("FACE_COSINE_THRESH", "0.4"))
    l2 = float(os.environ.get("FACE_L2_THRESH", "1.0"))
    return cos, l2


def is_same_person(cos: float, l2: float,
                    cos_thresh: Optional[float] = None,
                    l2_thresh: Optional[float] = None) -> bool:
    """跟 face C++ FaceRecognizer::match line 246-247 一致:
        is_same_person = cos >= cos_thresh AND l2 <= l2_thresh
    """
    if cos_thresh is None or l2_thresh is None:
        env_cos, env_l2 = get_match_thresholds()
        if cos_thresh is None:
            cos_thresh = env_cos
        if l2_thresh is None:
            l2_thresh = env_l2
    return cos >= cos_thresh and l2 <= l2_thresh
