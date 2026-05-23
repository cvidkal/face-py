"""Unit tests for module.face.clarity (Laplacian variance)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module.face.clarity import compute_image_clarity_score


class ClarityTest(unittest.TestCase):

    def test_none_returns_none(self) -> None:
        self.assertIsNone(compute_image_clarity_score(None))

    def test_empty_returns_none(self) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertIsNone(compute_image_clarity_score(empty))

    def test_uniform_image_low_variance(self) -> None:
        # Laplacian of a constant image is all zeros, variance = 0.0
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        score = compute_image_clarity_score(img)
        self.assertIsNotNone(score)
        assert score is not None  # mypy
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_noisy_image_high_variance(self) -> None:
        # Random noise: Laplacian magnitudes spread, variance > 0
        rng = np.random.RandomState(42)
        img = (rng.rand(100, 100, 3) * 255).astype(np.uint8)
        score = compute_image_clarity_score(img)
        self.assertIsNotNone(score)
        assert score is not None
        self.assertGreater(score, 100.0)

    def test_grayscale_input(self) -> None:
        # 1-channel should also work (skip BGR2GRAY conversion)
        rng = np.random.RandomState(0)
        img = (rng.rand(50, 50) * 255).astype(np.uint8)
        score = compute_image_clarity_score(img)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
