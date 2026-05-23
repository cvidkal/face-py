"""Unit tests for module.face.pose_gate.

公式来自 face C++ face_recognizer.cpp:202-225, 测试数据手算 expect 验证.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module.face.pose_gate import (
    HeadPose, compute_head_pose, get_pose_thresholds, is_pose_excessive,
)


class ComputeHeadPoseTest(unittest.TestCase):

    def test_frontal_face_yaw_zero(self) -> None:
        # 正脸: 双眼水平距 100, 鼻尖在中线, eye_mid=(50, 0), nose=(50, 50).
        # yaw_ratio = (50 - 50) / 100 = 0.0
        landmarks = [(0.0, 0.0), (100.0, 0.0),    # 双眼 (左/右无关)
                     (50.0, 50.0),                  # 鼻尖
                     (30.0, 100.0), (70.0, 100.0)]  # 双嘴角
        pose = compute_head_pose(landmarks)
        self.assertAlmostEqual(pose.yaw_ratio, 0.0, places=6)

    def test_yaw_right_positive(self) -> None:
        # 鼻尖偏右 (x 大), face C++ 给正值 yaw_ratio
        landmarks = [(0.0, 0.0), (100.0, 0.0),
                     (70.0, 50.0),                   # 鼻尖偏右 20px
                     (30.0, 100.0), (70.0, 100.0)]
        pose = compute_head_pose(landmarks)
        # yaw = (70 - 50) / 100 = 0.2
        self.assertAlmostEqual(pose.yaw_ratio, 0.2, places=6)
        self.assertGreater(pose.yaw_ratio, 0.0)

    def test_pitch_up_negative(self) -> None:
        # 仰头: 鼻尖比 (眼-嘴中点) 更高 (y 更小). 跟 face C++ 注释 line 221-222 一致:
        # "鼻子高于中点 (y 小) → 仰头" → pitch_ratio < 0 (因 nose.y - mid_y < 0)
        landmarks = [(0.0, 0.0), (100.0, 0.0),
                     (50.0, 20.0),                   # 鼻尖偏高 (y 小)
                     (30.0, 100.0), (70.0, 100.0)]
        pose = compute_head_pose(landmarks)
        # eye_mid_y=0, mouth_mid_y=100, mid_y=50, vert_dist=100, pitch=(20-50)/100=-0.3
        self.assertAlmostEqual(pose.pitch_ratio, -0.3, places=6)
        self.assertLess(pose.pitch_ratio, 0.0)

    def test_swapped_eyes_doesnt_change_result(self) -> None:
        # 左右眼顺序对调, pose 公式对称, 应该返一样.
        lm_a = [(0.0, 0.0), (100.0, 0.0),
                 (70.0, 50.0),
                 (30.0, 100.0), (70.0, 100.0)]
        lm_b = [(100.0, 0.0), (0.0, 0.0),
                 (70.0, 50.0),
                 (70.0, 100.0), (30.0, 100.0)]
        self.assertAlmostEqual(compute_head_pose(lm_a).yaw_ratio,
                                compute_head_pose(lm_b).yaw_ratio, places=6)
        self.assertAlmostEqual(compute_head_pose(lm_a).pitch_ratio,
                                compute_head_pose(lm_b).pitch_ratio, places=6)

    def test_invalid_landmark_count_returns_zero(self) -> None:
        self.assertEqual(compute_head_pose([]), HeadPose(0.0, 0.0))
        self.assertEqual(compute_head_pose([(0.0, 0.0)] * 3), HeadPose(0.0, 0.0))


class IsPoseExcessiveTest(unittest.TestCase):

    def setUp(self) -> None:
        # 清掉 env 干扰
        for k in ("FACE_POSE_ABS_YAW", "FACE_POSE_ABS_PITCH"):
            os.environ.pop(k, None)

    def test_within_threshold_ok(self) -> None:
        pose = HeadPose(yaw_ratio=0.2, pitch_ratio=0.4)
        self.assertFalse(is_pose_excessive(pose))

    def test_yaw_over_threshold(self) -> None:
        # face#13 patch 后默认 yaw 阈值 0.35
        pose = HeadPose(yaw_ratio=0.36, pitch_ratio=0.0)
        self.assertTrue(is_pose_excessive(pose))

    def test_pitch_over_threshold(self) -> None:
        # 默认 pitch 0.55
        pose = HeadPose(yaw_ratio=0.0, pitch_ratio=-0.56)
        self.assertTrue(is_pose_excessive(pose))

    def test_either_axis_over_triggers(self) -> None:
        self.assertTrue(is_pose_excessive(HeadPose(0.4, 0.0)))
        self.assertTrue(is_pose_excessive(HeadPose(0.0, 0.6)))
        self.assertTrue(is_pose_excessive(HeadPose(0.4, 0.6)))

    def test_explicit_thresholds_override(self) -> None:
        pose = HeadPose(yaw_ratio=0.1, pitch_ratio=0.1)
        # 收紧阈值, 同 pose 变成 excessive
        self.assertTrue(is_pose_excessive(pose, yaw_threshold=0.05, pitch_threshold=0.5))

    def test_env_override(self) -> None:
        os.environ["FACE_POSE_ABS_YAW"] = "0.10"
        try:
            yaw_t, _ = get_pose_thresholds()
            self.assertAlmostEqual(yaw_t, 0.10)
            self.assertTrue(is_pose_excessive(HeadPose(0.15, 0.0)))
            self.assertFalse(is_pose_excessive(HeadPose(0.08, 0.0)))
        finally:
            os.environ.pop("FACE_POSE_ABS_YAW", None)


if __name__ == "__main__":
    unittest.main()
