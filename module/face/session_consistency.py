"""Phase A.7 — Stage 1 内部一致性检查 (outlier detection on session embeddings).

User framing (Phase A.7 handoff):
    "人来看待一个 session 是否代训, 只要看两个. 一个是 session 内部稳不稳定, 我先不
    管这个人是谁, 内部一串图片是不是同一个人 ... 然后我们再来谈这个人是不是 reference
    那个人."

Stage 1 = "session 内部稳不稳定". 算法:

    给定 session 内 N 个 post-gate photo 的 embedding, 算 N×N pair-wise cos 矩阵.
    对每张 photo, 计算它对其它 photo 的 mean cos (excluding self).
    mean_cos < X → 这张 photo 跟其它人不像 → 标记为 outlier.
    session 有任一 outlier → internal_consistency = "inconsistent" → 报代训.

阈值 X 选 0.20 (Phase A.7 prototype 实验): rec#2 (唯一已知真造假) 的 imposter photo
mean_cos = 0.106, 其它 8 张真学员 mean_cos 0.685-0.747 — gap > 0.5, X=0.20 在安全
margin 中段. 0.15/0.20/0.25/0.30 在 40-session benchmark 上全部 TP=1 FN=0 FP=0 TN=39.

注: Stage 1 不需要 ref 图. 它只问"session 内部一致吗", 不问"是不是 ref 那个人".
单 photo session (N=1) 无法做 Stage 1 — 直接 fall through 到 Stage 2.
"""
from __future__ import annotations

import os
import statistics
from dataclasses import dataclass

import numpy as np

from .recognizer import cosine_score


# Phase A.7 prototype default. 在 40-session benchmark 上 [0.15, 0.30] 都全对,
# 选 0.20 取安全 margin 中段. ENV 可调.
DEFAULT_OUTLIER_MEAN_COS_THRESH = 0.20


def get_outlier_threshold() -> float:
    return float(os.environ.get(
        "FACE_SESSION_OUTLIER_MEAN_COS", str(DEFAULT_OUTLIER_MEAN_COS_THRESH)))


@dataclass
class ConsistencyResult:
    """Stage 1 输出. mean_cos_per_index 跟 embeddings 输入顺序对齐."""
    outlier_indices: list[int]
    mean_cos_per_index: list[float]
    threshold: float

    @property
    def is_consistent(self) -> bool:
        return not self.outlier_indices


def check_internal_consistency(
        embeddings: list[np.ndarray],
        outlier_thresh: float | None = None,
) -> ConsistencyResult:
    """对 N 个 embedding 跑 outlier detection. N<2 时返空 outlier list."""
    if outlier_thresh is None:
        outlier_thresh = get_outlier_threshold()
    n = len(embeddings)
    if n < 2:
        return ConsistencyResult(outlier_indices=[],
                                  mean_cos_per_index=[1.0] * n,
                                  threshold=outlier_thresh)
    mean_cos: list[float] = []
    for i in range(n):
        others = [cosine_score(embeddings[i], embeddings[j])
                  for j in range(n) if j != i]
        mean_cos.append(statistics.fmean(others))
    outliers = [i for i, m in enumerate(mean_cos) if m < outlier_thresh]
    return ConsistencyResult(outlier_indices=outliers,
                              mean_cos_per_index=mean_cos,
                              threshold=outlier_thresh)


def session_prototype(embeddings: list[np.ndarray]) -> np.ndarray:
    """L2-normalized mean of input embeddings — session-level prototype for Stage 2.

    输入要求: 已排除 outlier 的 consistent core. 至少 1 个 embedding.
    """
    if not embeddings:
        raise ValueError("session_prototype requires at least 1 embedding")
    mean = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean
