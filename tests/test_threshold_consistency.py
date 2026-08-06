"""l2 与 cos 阈值自洽性 (issue #3).

归一化 embedding 下 l2 和 cos 互为函数, 所以 `cos >= X and l2 <= Y` 里永远只有更严的
那个在生效。历史默认 cos=0.30 + l2=1.15 不自洽 (l2<=1.15 实为 cos>=0.3388), 导致
FACE_COSINE_THRESH 在 0.30~0.339 区间内改了没效果。
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module.face.recognizer import (
    _warned_l2_configs, classify_match, cos_to_l2, get_match_thresholds, l2_to_cos,
)


class IdentityTests(unittest.TestCase):
    """先钉死数学关系本身 —— 这是整个 fix 的前提."""

    def test_round_trip(self) -> None:
        for cos in (-1.0, -0.3, 0.0, 0.15, 0.30, 0.5, 0.9, 1.0):
            self.assertAlmostEqual(l2_to_cos(cos_to_l2(cos)), cos, places=9)

    def test_known_values(self) -> None:
        self.assertAlmostEqual(cos_to_l2(0.30), 1.18321596, places=6)
        self.assertAlmostEqual(l2_to_cos(1.15), 0.338750, places=6)
        self.assertAlmostEqual(cos_to_l2(1.0), 0.0, places=9)

    def test_matches_actual_vectors(self) -> None:
        """跟真实归一化向量对得上, 不只是纸面公式."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = rng.normal(size=128)
            a /= np.linalg.norm(a)
            b = rng.normal(size=128)
            b /= np.linalg.norm(b)
            cos = float(np.dot(a, b))
            self.assertAlmostEqual(float(np.linalg.norm(a - b)), cos_to_l2(cos), places=6)

    def test_out_of_range_cos_clamped(self) -> None:
        # 浮点误差可能让 cos 略大于 1 → sqrt 负数, 必须夹住
        self.assertEqual(cos_to_l2(1.0000001), 0.0)


class ThresholdDerivationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = dict(os.environ)
        for k in ("FACE_COSINE_THRESH", "FACE_L2_THRESH", "FACE_COSINE_MISMATCH_THRESH"):
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old)

    def test_l2_derived_from_cos_when_unset(self) -> None:
        _, cos_match, l2_max = get_match_thresholds()
        self.assertAlmostEqual(cos_match, 0.35)   # issue #1: 默认 0.30 → 0.35
        self.assertAlmostEqual(l2_max, cos_to_l2(cos_match), places=9)

    def test_changing_cos_thresh_now_takes_effect(self) -> None:
        """回归本 bug: 以前 cos 调到 0.30~0.339 之间完全没效果 (l2 更严)."""
        os.environ["FACE_COSINE_THRESH"] = "0.32"
        _, cos_match, l2_max = get_match_thresholds()
        self.assertAlmostEqual(cos_match, 0.32)
        self.assertAlmostEqual(l2_to_cos(l2_max), 0.32, places=9)

    def test_explicit_l2_still_respected(self) -> None:
        # 运维配置里可能已经写了 FACE_L2_THRESH, 不能静默变 no-op
        os.environ["FACE_L2_THRESH"] = "1.15"
        _, _, l2_max = get_match_thresholds()
        self.assertAlmostEqual(l2_max, 1.15)

    def test_explicit_inconsistent_l2_warns(self) -> None:
        os.environ["FACE_L2_THRESH"] = "1.15"      # 隐含 cos>=0.3388
        os.environ["FACE_COSINE_THRESH"] = "0.30"
        _warned_l2_configs.clear()
        with self.assertLogs("face-py.recognizer", level="WARNING") as cm:
            get_match_thresholds()
        joined = "\n".join(cm.output)
        self.assertIn("0.3388", joined)     # 告诉运维实际生效的门槛
        self.assertIn("0.30", joined)

    def test_warns_only_once_per_config(self) -> None:
        """get_match_thresholds 是每张照片都调的 —— 无脑 warn 会淹掉生产日志."""
        os.environ["FACE_L2_THRESH"] = "1.15"
        os.environ["FACE_COSINE_THRESH"] = "0.30"
        _warned_l2_configs.clear()
        with self.assertLogs("face-py.recognizer", level="WARNING") as cm:
            for _ in range(50):
                get_match_thresholds()
        self.assertEqual(len(cm.output), 1)

    def test_consistent_explicit_l2_does_not_warn(self) -> None:
        os.environ["FACE_L2_THRESH"] = f"{cos_to_l2(0.30):.10f}"
        os.environ["FACE_COSINE_THRESH"] = "0.30"
        with self.assertNoLogs("face-py.recognizer", level="WARNING"):
            get_match_thresholds()

    def test_blank_l2_env_treated_as_unset(self) -> None:
        os.environ["FACE_L2_THRESH"] = "   "
        _, cos_match, l2_max = get_match_thresholds()
        self.assertAlmostEqual(l2_max, cos_to_l2(cos_match), places=9)


class ClassifyTests(unittest.TestCase):
    """cos 落在「旧 l2 死区」(cos_match ~ l2 隐含的 cos) 的照片现在能判 match."""

    def setUp(self) -> None:
        self._old = dict(os.environ)
        for k in ("FACE_COSINE_THRESH", "FACE_L2_THRESH", "FACE_COSINE_MISMATCH_THRESH"):
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old)

    def test_cos_in_former_dead_zone_now_matches(self) -> None:
        # 旧行为: cos>=0.30 却被 l2<=1.15 (隐含 cos>=0.3388) 挡在门外.
        # 显式传 cos_match=0.30 复现当时的配置, 验证死区没了。
        for cos in (0.300, 0.315, 0.338):
            with self.subTest(cos=cos):
                self.assertEqual(
                    classify_match(cos, cos_to_l2(cos), cos_match_thresh=0.30,
                                   l2_max_thresh=cos_to_l2(0.30)), "match")

    def test_below_cos_match_still_inconclusive(self) -> None:
        _, cos_match, l2_max = get_match_thresholds()
        self.assertEqual(
            classify_match(cos_match - 0.01, cos_to_l2(cos_match - 0.01)), "inconclusive")

    def test_mismatch_zone_unchanged(self) -> None:
        self.assertEqual(classify_match(0.10, cos_to_l2(0.10)), "mismatch")

    def test_explicit_strict_l2_still_blocks(self) -> None:
        # 显式配了严格 l2 时行为不变 (向后兼容)
        self.assertEqual(
            classify_match(0.31, cos_to_l2(0.31), l2_max_thresh=1.15), "inconclusive")


if __name__ == "__main__":
    unittest.main(verbosity=2)
