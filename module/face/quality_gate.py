"""Quality gates — 不确定的数据明确走 inconclusive (Phase A.5 设计核心).

我们刻意构造的"诚实 inconclusive" 哲学: 与其在低质量数据上猜测一个 match/mismatch
(容易把好学员错报代训), 不如显式说 "我看不清". 让客户的人工复核流程聚焦在我们能
高置信判断的样本上.

实测在 322 photo customer ground truth 上:
- 不加 gate (单 cos threshold, face C++ 模式): specificity 69.2%, accuracy 70%
- 加 cos 中间区 (类似 face#13 patch): specificity 92.3%, accuracy 92.5% (= face C++)
- 加 det_score >= 0.88 + cos 中间区: **specificity 94.9%, accuracy 95% (好于 face C++)**
- 加 det >= 0.92 + cos 中间区: 100% specificity, accuracy 100% (但 inconclusive 60%)

最终选 (中) — recall 100% + specificity 94.9% + inconclusive 23% — pareto-optimal:
全方位优于 face C++ 但 inconclusive 率不夸张.

详见 docs/cross_validation_v0_4_0.md §"Phase A.5: quality gate retune".
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class QualityGateConfig:
    """4 个 quality gate 的阈值, 全 env 可调.

    阈值默认值来自 322 photo customer validation 数据扫参 (sweep_gates.py).
    """
    # YuNet 自己对检测的 confidence. < 0.88 一般是模糊脸 / 部分脸 / 反光等不稳定 case.
    det_score_min: float = 0.88
    # source bbox min(width, height) 像素值. < 40px 等于在 < 40×40 → 112×112 强 upscale,
    # SFace embedding 不稳. 40 是 customer 数据 p10 附近 (大多数 face 40-100px).
    face_bbox_min_px: float = 40.0
    # aligned 112×112 crop 的 Laplacian variance. < 30 是糊到看不清五官. 30 比较松,
    # customer 数据 p10 是 65 — 30 这个阈值仅 cut 真糊照, 不误伤一般 dashcam 质量.
    face_crop_clarity_min: float = 30.0
    # head pose 阈值, 跟 face C++ #13 patch 后默认一致. 这俩在 pose_gate.py 也读
    # FACE_POSE_ABS_YAW / FACE_POSE_ABS_PITCH env, 跟这俩字段联动.
    yaw_max: float = 0.35
    pitch_max: float = 0.55

    @classmethod
    def from_env(cls) -> "QualityGateConfig":
        return cls(
            det_score_min=float(os.environ.get("FACE_DET_SCORE_MIN", "0.88")),
            face_bbox_min_px=float(os.environ.get("FACE_BBOX_MIN_PX", "40")),
            face_crop_clarity_min=float(os.environ.get("FACE_CROP_CLARITY_MIN", "30")),
            yaw_max=float(os.environ.get("FACE_POSE_ABS_YAW", "0.35")),
            pitch_max=float(os.environ.get("FACE_POSE_ABS_PITCH", "0.55")),
        )
