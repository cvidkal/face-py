"""Head pose 几何估计 + gating, 1:1 port from face C++ face_recognizer.cpp:202-225.

注意: face C++ 的 5 landmarks 来自 landmarks-regression-retail-0009 (单独模型),
顺序是 0=左眼, 1=右眼, 2=鼻尖, 3=左嘴角, 4=右嘴角.

YuNet (我们 face-py 用) 输出顺序按 OpenCV 文档:
  [0,1] = 右眼  (x, y)
  [2,3] = 左眼
  [4,5] = 鼻尖
  [6,7] = 右嘴角
  [8,9] = 左嘴角

注意左右**交换**了 — 这跟 face C++ 反. 但 pose 公式里:
  yaw_ratio = (nose.x - eye_mid.x) / eye_dist     ← eye_mid 是双眼平均, 跟左右无关
  pitch_ratio = (nose.y - mid_y) / vert_dist       ← 同上, 双眼 / 双嘴角中点

公式对称, 左右眼 / 左右嘴角的顺序不影响结果. 不用 swap.

Phase A.4 跨验证时如果发现 yaw 符号跟 face C++ 反, 加一个 ENV `FACE_POSE_YAW_SIGN_FLIP=1`
开关; 当前先按符号一致写, 实测再说.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class HeadPose:
    yaw_ratio: float = 0.0    # 鼻尖水平偏移 / 双眼间距. 正脸 ~0, 侧脸 ~0.3+
    pitch_ratio: float = 0.0  # 鼻尖垂直偏移 / 眼-嘴中距. 正中 ~0, 仰/俯头 ~0.4+


def compute_head_pose(landmarks: list[tuple[float, float]]) -> HeadPose:
    """5 个 landmarks (x, y) tuples → (yaw_ratio, pitch_ratio).

    landmarks 长度必须是 5, 顺序: eye_a, eye_b, nose, mouth_a, mouth_b
    (左右 a/b 顺序对结果无影响, 公式对称).
    """
    out = HeadPose()
    if len(landmarks) != 5:
        return out
    eye_a, eye_b, nose, mouth_a, mouth_b = landmarks
    eye_mid_x = (eye_a[0] + eye_b[0]) * 0.5
    eye_mid_y = (eye_a[1] + eye_b[1]) * 0.5
    mouth_mid_x = (mouth_a[0] + mouth_b[0]) * 0.5
    mouth_mid_y = (mouth_a[1] + mouth_b[1]) * 0.5
    eye_dist = math.hypot(eye_b[0] - eye_a[0], eye_b[1] - eye_a[1])
    vert_dist = math.hypot(mouth_mid_x - eye_mid_x, mouth_mid_y - eye_mid_y)
    if eye_dist > 1e-3:
        out.yaw_ratio = (nose[0] - eye_mid_x) / eye_dist
    if vert_dist > 1e-3:
        mid_y = (eye_mid_y + mouth_mid_y) * 0.5
        out.pitch_ratio = (nose[1] - mid_y) / vert_dist
    return out


# Pose 阈值 — 跟 face C++ face_pose_abs_yaw() / face_pose_abs_pitch() 同名同默认.
# face#13 patch 后这俩值是 0.35 / 0.55, 比 v0.1.0 时的 0.2 / 0.4 宽 — 减少边缘
# pose 误判 (CLAUDE.md "已知坑" 段背景).
def get_pose_thresholds() -> tuple[float, float]:
    yaw = float(os.environ.get("FACE_POSE_ABS_YAW", "0.35"))
    pitch = float(os.environ.get("FACE_POSE_ABS_PITCH", "0.55"))
    return yaw, pitch


def is_pose_excessive(pose: HeadPose,
                       yaw_threshold: Optional[float] = None,
                       pitch_threshold: Optional[float] = None) -> bool:
    """跟 face C++ check_session_pose_gate 单图分支等价 (line 2676-2691).

    单图模式 (没 session baseline) 用绝对阈值, 见 face C++ identity_check handler
    line 4371: std::abs(feat.yaw_ratio) > abs_yaw || std::abs(feat.pitch_ratio) > abs_pitch.
    """
    if yaw_threshold is None or pitch_threshold is None:
        env_yaw, env_pitch = get_pose_thresholds()
        if yaw_threshold is None:
            yaw_threshold = env_yaw
        if pitch_threshold is None:
            pitch_threshold = env_pitch
    return (abs(pose.yaw_ratio) > yaw_threshold or
             abs(pose.pitch_ratio) > pitch_threshold)
