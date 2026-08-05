"""content gate (issue #1): 「整张图没内容」判定.

用合成图覆盖三类真实失效形态 + 一类必须放行的形态 (夜间但有内容)。
阈值本身是在 2 万张真实归档照片上标的, 见 content_gate.py docstring。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module.face.content_gate import compute_content_score, is_photo_unusable
from module.face.quality_gate import QualityGateConfig

H, W = 240, 320


def _osd(img: np.ndarray) -> np.ndarray:
    """画上抓拍设备烧录的红字 OSD (上下各两行), 跟现网图一致."""
    out = img.copy()
    for y in (14, 30, H - 34, H - 14):
        cv2.putText(out, "2026-08-05 10:25:54 GPS:113.02 speed:0.0", (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
    return out


def _blank(value: int) -> np.ndarray:
    return _osd(np.full((H, W, 3), value, np.uint8))


def _with_content(base: int, amp: int) -> np.ndarray:
    """base 亮度上叠一个人脸大小的椭圆 + 纹理 — 模拟"暗但拍到人"."""
    img = np.full((H, W, 3), base, np.uint8)
    cv2.ellipse(img, (160, 120), (46, 60), 0, 0, 360, (base + amp,) * 3, -1)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, amp // 2 + 1, (H, W, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return _osd(img)


class ContentScoreTests(unittest.TestCase):
    def test_all_black_scores_near_zero(self) -> None:
        self.assertLess(compute_content_score(_blank(0)), 10.0)

    def test_blown_out_white_scores_near_zero(self) -> None:
        # 客户截图里「只拍到车窗」那种过曝图 — 跟全黑同样是"没内容"
        self.assertLess(compute_content_score(_blank(250)), 10.0)

    def test_uniform_dark_grey_scores_near_zero(self) -> None:
        self.assertLess(compute_content_score(_blank(18)), 10.0)

    def test_dark_but_has_content_scores_high(self) -> None:
        # 夜间训练照: 暗, 但拍到了人 → 必须远高于阈值, 否则会误杀合法学时
        self.assertGreater(compute_content_score(_with_content(12, 60)), 10.0)

    def test_osd_alone_does_not_create_content(self) -> None:
        """OSD 红字是判据的主要干扰源: 不挖掉它, 全黑图也会有可观的 Laplacian."""
        black = np.zeros((H, W, 3), np.uint8)
        self.assertGreater(float(np.var(cv2.Laplacian(
            cv2.cvtColor(_osd(black), cv2.COLOR_BGR2GRAY), cv2.CV_64F))), 10.0)
        # 挖掉 OSD 带 + 抹红字之后就该塌到 0 附近
        self.assertLess(compute_content_score(_osd(black)), 10.0)

    def test_none_and_empty(self) -> None:
        self.assertIsNone(compute_content_score(None))
        self.assertIsNone(compute_content_score(np.zeros((0, 0, 3), np.uint8)))

    def test_too_small_to_crop(self) -> None:
        # 裁完中心区不足 8px 高 → 不判 (返 None), 而不是瞎给一个分
        self.assertIsNone(compute_content_score(np.zeros((6, 40, 3), np.uint8)))

    def test_grayscale_input(self) -> None:
        self.assertIsNotNone(compute_content_score(np.zeros((H, W), np.uint8)))


class GateDecisionTests(unittest.TestCase):
    def test_below_threshold_is_unusable(self) -> None:
        self.assertTrue(is_photo_unusable(3.0, 10.0))

    def test_at_or_above_threshold_is_usable(self) -> None:
        self.assertFalse(is_photo_unusable(10.0, 10.0))
        self.assertFalse(is_photo_unusable(11.0, 10.0))

    def test_threshold_zero_disables_gate(self) -> None:
        self.assertFalse(is_photo_unusable(0.0, 0.0))
        self.assertFalse(is_photo_unusable(0.0, -1.0))

    def test_none_score_never_unusable(self) -> None:
        # 辅助指标算不出来时放行给 detect 去判, 不因此否掉整张照片
        self.assertFalse(is_photo_unusable(None, 10.0))

    def test_default_threshold_from_config(self) -> None:
        self.assertEqual(QualityGateConfig().photo_content_min, 10.0)


class RealisticEndToEndTests(unittest.TestCase):
    """把合成图跑一遍「算分 → 判定」, 确认默认阈值下的分档符合标定结论."""

    def test_blank_variants_all_flagged(self) -> None:
        t = QualityGateConfig().photo_content_min
        for name, img in (("black", _blank(0)), ("white", _blank(250)),
                          ("grey", _blank(18))):
            with self.subTest(name):
                self.assertTrue(is_photo_unusable(compute_content_score(img), t))

    def test_content_variants_all_passed(self) -> None:
        t = QualityGateConfig().photo_content_min
        for base, amp in ((12, 60), (40, 80), (110, 100)):
            with self.subTest(base=base):
                self.assertFalse(is_photo_unusable(
                    compute_content_score(_with_content(base, amp)), t))


if __name__ == "__main__":
    unittest.main(verbosity=2)
