"""quality gate 的两种模式 (issue #1).

`audit` (默认): gate 不拦判定, 只把结果作为 `quality_flags` 透出去。
`block` (历史): 任一 gate 不过 → 直接 inconclusive, 不做比对。

**passes_gate 的语义在两种模式下都不变** —— 它继续决定哪些 embedding 进 Stage 1
聚类。实测只覆盖照片级判定, 把低质量 embedding 放进聚类是没测过的。
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module.face.errors import (
    DETECTION_LOW_CONFIDENCE, FACE_TOO_BLURRY, FACE_TOO_SMALL,
    MISMATCH_WITHHELD_LOW_QUALITY, POSE_EXCESSIVE,
)
from module.face.pipeline import (
    IdentityCheckResult, SessionPhotoResult, _verdict_with_quality,
)
from module.face.quality_gate import (
    GATE_MODE_AUDIT, GATE_MODE_BLOCK, QualityGateConfig,
)
from module.face.recognizer import cos_to_l2


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = dict(os.environ)
        os.environ.pop("FACE_QUALITY_GATE_MODE", None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old)

    def test_default_is_audit(self) -> None:
        self.assertEqual(QualityGateConfig().mode, GATE_MODE_AUDIT)
        self.assertFalse(QualityGateConfig().blocks)

    def test_default_from_env_is_audit(self) -> None:
        self.assertFalse(QualityGateConfig.from_env().blocks)

    def test_block_mode_from_env(self) -> None:
        os.environ["FACE_QUALITY_GATE_MODE"] = "block"
        self.assertTrue(QualityGateConfig.from_env().blocks)

    def test_mode_is_case_and_space_insensitive(self) -> None:
        os.environ["FACE_QUALITY_GATE_MODE"] = "  BLOCK  "
        self.assertTrue(QualityGateConfig.from_env().blocks)

    def test_unknown_mode_does_not_block(self) -> None:
        # 打错字不该静默恢复成拦截 —— 只有明确的 "block" 才拦
        os.environ["FACE_QUALITY_GATE_MODE"] = "blcok"
        self.assertFalse(QualityGateConfig.from_env().blocks)

    def test_empty_mode_falls_back_to_audit(self) -> None:
        os.environ["FACE_QUALITY_GATE_MODE"] = ""
        self.assertEqual(QualityGateConfig.from_env().mode, GATE_MODE_AUDIT)

    def test_thresholds_unchanged_by_mode(self) -> None:
        """mode 只改「拦不拦」, 不改各 gate 的阈值本身."""
        for mode in (GATE_MODE_AUDIT, GATE_MODE_BLOCK):
            c = QualityGateConfig(mode=mode)
            self.assertEqual(c.det_score_min, 0.88)
            self.assertEqual(c.face_crop_clarity_min, 30.0)
            self.assertEqual(c.face_bbox_min_px, 40.0)


class ResultShapeTests(unittest.TestCase):
    """响应新增字段的形状 —— TA / 客户按这个解析."""

    def test_identity_result_defaults(self) -> None:
        j = IdentityCheckResult().to_json()
        self.assertEqual(j["quality_flags"], [])
        for k in ("det_score", "bbox_min_px", "yaw_ratio", "pitch_ratio"):
            self.assertIn(k, j)
            self.assertIsNone(j[k])

    def test_identity_result_carries_flags(self) -> None:
        r = IdentityCheckResult(match_status="match", cosine_score=0.42,
                                quality_flags=[DETECTION_LOW_CONFIDENCE, FACE_TOO_BLURRY],
                                det_score=0.71, bbox_min_px=33.0)
        j = r.to_json()
        # audit 模式的核心: 有 flag 但**照样出结论**, error_code 为空
        self.assertEqual(j["match_status"], "match")
        self.assertEqual(j["error_code"], "")
        self.assertEqual(j["quality_flags"], [DETECTION_LOW_CONFIDENCE, FACE_TOO_BLURRY])
        self.assertAlmostEqual(j["det_score"], 0.71)

    def test_flags_are_copied_not_aliased(self) -> None:
        r = IdentityCheckResult(quality_flags=[POSE_EXCESSIVE])
        j = r.to_json()
        j["quality_flags"].append(FACE_TOO_SMALL)
        self.assertEqual(r.quality_flags, [POSE_EXCESSIVE])

    def test_session_photo_result_shape(self) -> None:
        j = SessionPhotoResult().to_json()
        self.assertEqual(j["quality_flags"], [])
        self.assertFalse(j["passes_gate"])

    def test_session_photo_flagged_but_judged(self) -> None:
        """带 flag 的照片仍有自己的 cos/判定, 但 passes_gate=False (不进聚类)."""
        r = SessionPhotoResult(passes_gate=False, quality_flags=[POSE_EXCESSIVE],
                               cosine_score=0.51, match_status="match")
        j = r.to_json()
        self.assertEqual(j["match_status"], "match")
        self.assertFalse(j["passes_gate"])
        self.assertEqual(j["quality_flags"], [POSE_EXCESSIVE])

    def test_each_result_gets_its_own_flag_list(self) -> None:
        a, b = IdentityCheckResult(), IdentityCheckResult()
        a.quality_flags.append(FACE_TOO_SMALL)
        self.assertEqual(b.quality_flags, [])




class AsymmetricMismatchTests(unittest.TestCase):
    """quality 不干净时只压制 mismatch, 不压制 match (issue #1).

    判错方向的代价不对称: 说「是本人」错了 = 漏一个代训; 说「不是本人」错了 =
    冤枉一个正常学员被查学时。后者贵得多。
    """

    def setUp(self) -> None:
        self._old = dict(os.environ)
        for k in ("FACE_COSINE_THRESH", "FACE_COSINE_MISMATCH_THRESH", "FACE_L2_THRESH"):
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old)

    def test_mismatch_withheld_when_flagged(self) -> None:
        v, code = _verdict_with_quality(0.05, cos_to_l2(0.05), [POSE_EXCESSIVE])
        self.assertEqual(v, "inconclusive")
        self.assertEqual(code, MISMATCH_WITHHELD_LOW_QUALITY)

    def test_mismatch_kept_when_clean(self) -> None:
        v, code = _verdict_with_quality(0.05, cos_to_l2(0.05), [])
        self.assertEqual(v, "mismatch")
        self.assertEqual(code, "")

    def test_match_not_suppressed_by_flags(self) -> None:
        """这是本次改动的收益来源 —— 糊图也照样能判 match."""
        v, code = _verdict_with_quality(0.60, cos_to_l2(0.60),
                                        [DETECTION_LOW_CONFIDENCE, FACE_TOO_BLURRY])
        self.assertEqual(v, "match")
        self.assertEqual(code, "")

    def test_inconclusive_zone_unaffected(self) -> None:
        v, code = _verdict_with_quality(0.25, cos_to_l2(0.25), [FACE_TOO_SMALL])
        self.assertEqual(v, "inconclusive")
        self.assertEqual(code, "")      # 本来就是中间区, 不是被压制的




class EmptyEnvTests(unittest.TestCase):
    """compose 的 ${VAR:-} 传的是**空串**不是"不传" —— 空串必须等同未设置.

    2026-08-06 dev 演练实测: 空串让 float("") 抛 ValueError, cloth/face 双双
    CrashLoop。幸好是在 dev 抓到的。
    """

    def setUp(self) -> None:
        self._old = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old)

    def test_empty_strings_fall_back_to_defaults(self) -> None:
        for k in ("FACE_DET_SCORE_MIN", "FACE_BBOX_MIN_PX", "FACE_CROP_CLARITY_MIN",
                  "FACE_PHOTO_CONTENT_MIN", "FACE_POSE_ABS_YAW", "FACE_POSE_ABS_PITCH"):
            os.environ[k] = ""
        c = QualityGateConfig.from_env()
        self.assertEqual(c.det_score_min, 0.88)
        self.assertEqual(c.photo_content_min, 10.0)
        self.assertEqual(c.yaw_max, 0.35)

    def test_whitespace_only_also_treated_as_unset(self) -> None:
        os.environ["FACE_PHOTO_CONTENT_MIN"] = "   "
        self.assertEqual(QualityGateConfig.from_env().photo_content_min, 10.0)

    def test_real_value_still_wins(self) -> None:
        os.environ["FACE_PHOTO_CONTENT_MIN"] = "25"
        self.assertEqual(QualityGateConfig.from_env().photo_content_min, 25.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
