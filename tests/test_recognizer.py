"""Unit tests for module.face.recognizer (thresholds + cos/l2 math)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module.face.recognizer import (
    cosine_score, get_match_thresholds, is_same_person, l2_distance,
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

    def setUp(self) -> None:
        for k in ("FACE_COSINE_THRESH", "FACE_L2_THRESH"):
            os.environ.pop(k, None)

    def test_high_cos_low_l2_match(self) -> None:
        # cos=0.55, l2=0.9 — 远高于 0.4, 远低于 1.0
        self.assertTrue(is_same_person(0.55, 0.9))

    def test_low_cos_no_match(self) -> None:
        self.assertFalse(is_same_person(0.35, 0.9))

    def test_high_l2_no_match(self) -> None:
        # 即便 cos 高, l2 越线就 mismatch (跟 face C++ AND 关系)
        self.assertFalse(is_same_person(0.55, 1.05))

    def test_boundary_at_threshold(self) -> None:
        # cos 恰等于 0.4 应该 match (>=), l2 恰等于 1.0 应该 match (<=)
        self.assertTrue(is_same_person(0.4, 1.0))

    def test_explicit_threshold_override(self) -> None:
        # 提高 cos 阈值到 0.6, 同 cos 0.55 变 mismatch
        self.assertFalse(is_same_person(0.55, 0.9, cos_thresh=0.6))

    def test_env_override(self) -> None:
        os.environ["FACE_COSINE_THRESH"] = "0.5"
        try:
            cos_t, l2_t = get_match_thresholds()
            self.assertAlmostEqual(cos_t, 0.5)
            self.assertAlmostEqual(l2_t, 1.0)
            self.assertFalse(is_same_person(0.45, 0.9))
            self.assertTrue(is_same_person(0.55, 0.9))
        finally:
            os.environ.pop("FACE_COSINE_THRESH", None)


if __name__ == "__main__":
    unittest.main()
