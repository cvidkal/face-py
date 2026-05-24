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


# 业务阈值, 用 ENV 覆盖. 默认值来自 Phase A.5 quality-gate sweep 跟 customer ground
# truth (1 真造假 + 39 clean) 联合调参 (docs/cross_validation_v0_4_0.md "Phase A.5" 段).
#
# face C++ 用 face-reidentification-retail-0095 (256-d) 默认是 cos=0.4 / l2=1.0.
# SFace (128-d, 我们 face-py 用) 跨 322 photo 测出 cos 系统性偏低 ~0.10, l2 偏高 ~0.10.
# Phase A.4 单 cos thresh 调到 0.30/1.10 → 78% 决策一致率, 但 session-level 错报飙升
# (specificity 53.8% << face C++ 92.3%). Phase A.5 引入 cos 中间区 (mismatch zone +
# match zone, 中间走 inconclusive), 同时加 detector + size + clarity quality gates:
# session-level 在 customer GT 上 recall 100% specificity 94.9% accuracy 95% (好于
# face C++ 92.3% / 92.5%), 代价是 photo-level inconclusive 率 23% (vs face C++ ~10%).
def get_match_thresholds() -> tuple[float, float, float]:
    """返 (cos_mismatch_thresh, cos_match_thresh, l2_max_thresh).

    cos < cos_mismatch_thresh           → mismatch (高置信不像)
    cos >= cos_match_thresh AND l2 <= l2_max  → match (高置信像)
    其余 → inconclusive (cos 中间区, 拿不准)

    默认 0.15 / 0.30 / 1.15 (Phase A.5 sweep 出的 pareto-optimal).
    """
    cos_mismatch = float(os.environ.get("FACE_COSINE_MISMATCH_THRESH", "0.15"))
    cos_match = float(os.environ.get("FACE_COSINE_THRESH", "0.30"))
    l2_max = float(os.environ.get("FACE_L2_THRESH", "1.15"))
    return cos_mismatch, cos_match, l2_max


def classify_match(cos: float, l2: float,
                    cos_mismatch_thresh: Optional[float] = None,
                    cos_match_thresh: Optional[float] = None,
                    l2_max_thresh: Optional[float] = None) -> str:
    """Tri-state classifier (Phase A.5).

    返 'match' | 'mismatch' | 'inconclusive'.
    """
    if cos_mismatch_thresh is None or cos_match_thresh is None or l2_max_thresh is None:
        env_lo, env_hi, env_l2 = get_match_thresholds()
        if cos_mismatch_thresh is None: cos_mismatch_thresh = env_lo
        if cos_match_thresh is None:    cos_match_thresh = env_hi
        if l2_max_thresh is None:       l2_max_thresh = env_l2
    if cos < cos_mismatch_thresh:
        return "mismatch"
    if cos >= cos_match_thresh and l2 <= l2_max_thresh:
        return "match"
    return "inconclusive"


def is_same_person(cos: float, l2: float,
                    cos_thresh: Optional[float] = None,
                    l2_thresh: Optional[float] = None) -> bool:
    """两态版本, 给 /face/compare 用 (直接比对场景, 客户预期 bool 输出).

    跟 face C++ FaceRecognizer::match line 246-247 等价 (但 cos_thresh 默认是新值 0.30,
    不是 face C++ 的 0.45).
    """
    if cos_thresh is None or l2_thresh is None:
        _, env_cos, env_l2 = get_match_thresholds()
        if cos_thresh is None: cos_thresh = env_cos
        if l2_thresh is None:  l2_thresh = env_l2
    return cos >= cos_thresh and l2 <= l2_thresh
