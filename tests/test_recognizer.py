"""Unit tests for module.face.recognizer (thresholds + cos/l2 math)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module.face.recognizer import (
    classify_match, cosine_score, get_match_thresholds, is_same_person,
    l2_distance, cos_to_l2, l2_to_cos,
)


class CosineL2Test(unittest.TestCase):

    def test_cosine_identical(self) -> None:
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_score(v, v), 1.0, places=6)

    def test_cosine_orthogonal(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_score(a, b), 0.0, places=6)

    def test_l2_identical(self) -> None:
        v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.assertAlmostEqual(l2_distance(v, v), 0.0, places=6)

    def test_l2_orthogonal_unit_vectors(self) -> None:
        # 两个正交单位向量 L2 距离 = sqrt(2)
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(l2_distance(a, b), np.sqrt(2.0), places=6)


class IsSamePersonTest(unittest.TestCase):
    """Phase A.5: tri-state classify_match (mismatch zone / match zone / inconclusive gap).

    Defaults: cos_mismatch=0.15, cos_match=0.30, l2_max=1.15.
    is_same_person (bool) 仍然提供给 /face/compare 用 (单 cos_match threshold).
    """

    def setUp(self) -> None:
        for k in ("FACE_COSINE_THRESH", "FACE_COSINE_MISMATCH_THRESH", "FACE_L2_THRESH"):
            os.environ.pop(k, None)

    def test_default_thresholds(self) -> None:
        cos_lo, cos_hi, l2_max = get_match_thresholds()
        self.assertAlmostEqual(cos_lo, 0.15)
        self.assertAlmostEqual(cos_hi, 0.30)
        # issue #3: l2 默认从 cos_match 推导 (归一化 embedding 下两者互为函数).
        # 旧值 1.15 隐含 cos>=0.3388, 跟 cos_hi=0.30 不自洽 —— 这里断言 1.15 等于
        # 把那个 bug 钉进测试, 已改成断言自洽关系。
        self.assertAlmostEqual(l2_max, cos_to_l2(0.30), places=9)
        self.assertAlmostEqual(l2_to_cos(l2_max), cos_hi, places=9)

    # classify_match (tri-state)

    def test_classify_high_cos_match(self) -> None:
        # cos=0.55, l2=0.9 — 远高于 0.30 + l2 OK
        self.assertEqual(classify_match(0.55, 0.9), "match")

    def test_classify_low_cos_mismatch(self) -> None:
        # cos=0.10 < 0.15
        self.assertEqual(classify_match(0.10, 0.9), "mismatch")

    def test_classify_gap_inconclusive(self) -> None:
        # cos=0.20 ∈ [0.15, 0.30), 中间区
        self.assertEqual(classify_match(0.20, 0.9), "inconclusive")
        # 边界: cos 恰等于 0.15 — 不是 mismatch (>=), 是 inconclusive
        self.assertEqual(classify_match(0.15, 0.9), "inconclusive")
        # 边界: cos 恰等于 0.30 + l2 OK — match
        self.assertEqual(classify_match(0.30, 1.0), "match")

    def test_classify_high_l2_blocks_match(self) -> None:
        # cos 够高但 l2 越线 — 不算 match. 也不算 mismatch (cos 不够低). → inconclusive
        self.assertEqual(classify_match(0.55, 1.20), "inconclusive")

    def test_classify_low_cos_overrides_l2(self) -> None:
        # cos 极低 — 即使 l2 OK 也判 mismatch (cos 优先)
        self.assertEqual(classify_match(0.05, 0.5), "mismatch")

    def test_classify_explicit_override(self) -> None:
        # 把 cos_match 调到 0.6: 原来 0.55 是 match 现在变 inconclusive
        self.assertEqual(classify_match(0.55, 0.9, cos_match_thresh=0.6), "inconclusive")

    def test_classify_env_override(self) -> None:
        os.environ["FACE_COSINE_MISMATCH_THRESH"] = "0.25"
        try:
            # cos=0.22 < 0.25 现在算 mismatch (默认 0.15 时它是 inconclusive)
            self.assertEqual(classify_match(0.22, 0.9), "mismatch")
        finally:
            os.environ.pop("FACE_COSINE_MISMATCH_THRESH", None)

    # is_same_person (legacy bool, 给 /face/compare 用)

    def test_is_same_person_default(self) -> None:
        self.assertTrue(is_same_person(0.30, 1.0))     # 恰到阈值
        self.assertFalse(is_same_person(0.29, 1.0))    # cos 差一点
        self.assertFalse(is_same_person(0.30, 1.20))   # l2 超


if __name__ == "__main__":
    unittest.main()
