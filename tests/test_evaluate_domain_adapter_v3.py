from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.domain_adapter_v3_metrics import HistoricalRelativeMetrics
from tools.evaluate_domain_adapter_v3 import (
    _require_v3_candidate,
    evaluate_v3_candidate,
    v3_gate_passes,
)


def _v2_report(*, absolute: bool, relative: bool, release: bool) -> dict:
    return {
        "schema_version": 2,
        "dataset_role": "engineering",
        "cohort_status": "not_applicable",
        "cohort_sufficiency": None,
        "provenance": {
            "engineering_manifest_sha256_pinned": True,
            "engineering_dataset_role_absent": True,
        },
        "absolute_metrics": {"conditional_accuracy": 0.9},
        "relative_metrics": {"recall_lift": 0.0},
        "absolute_gate_passed": absolute,
        "relative_gate_passed": relative,
        "engineering_gate_passed": absolute and relative,
        "release_gate_passed": release,
    }


class V3RoleAuthorityTests(unittest.TestCase):
    @patch("tools.evaluate_domain_adapter_v3._require_v3_candidate")
    @patch("tools.evaluate_domain_adapter_v3.evaluate_v2_candidate")
    def test_legacy_can_pass_safety_without_gaining_release_authority(
        self,
        evaluate_v2: Mock,
        _require_candidate: Mock,
    ) -> None:
        evaluate_v2.return_value = _v2_report(
            absolute=True,
            relative=False,
            release=False,
        )
        report = evaluate_v3_candidate(
            Path("candidate"),
            Path("legacy.json"),
            Path("benchmark.json"),
            dataset_role="legacy",
        )
        self.assertTrue(report["legacy_safety_gate_passed"])
        self.assertFalse(report["release_gate_passed"])
        self.assertNotIn("engineering_gate_passed", report)
        self.assertTrue(v3_gate_passes(report))
        evaluate_v2.assert_called_once_with(
            Path("candidate"),
            Path("legacy.json"),
            Path("benchmark.json"),
            dataset_role="engineering",
            historical_manifest=None,
            cohort_registry=None,
            pipeline_factory=None,
            inference_session_factory=None,
        )

    @patch("tools.evaluate_domain_adapter_v3._require_v3_candidate")
    @patch("tools.evaluate_domain_adapter_v3.evaluate_v2_candidate")
    def test_release_authority_requires_the_underlying_release_gate(
        self,
        evaluate_v2: Mock,
        _require_candidate: Mock,
    ) -> None:
        evaluate_v2.return_value = _v2_report(
            absolute=True,
            relative=True,
            release=True,
        )
        report = evaluate_v3_candidate(
            Path("candidate"),
            Path("release.json"),
            Path("benchmark.json"),
            dataset_role="release",
            historical_manifest=Path("historical.json"),
            cohort_registry=Path("registry.json"),
        )
        self.assertFalse(report["legacy_safety_gate_passed"])
        self.assertTrue(report["release_gate_passed"])
        self.assertTrue(v3_gate_passes(report))

    def test_release_requires_historical_manifest_and_registry(self) -> None:
        with self.assertRaisesRegex(ValueError, "historical_manifest"):
            evaluate_v3_candidate(
                Path("candidate"),
                Path("release.json"),
                Path("benchmark.json"),
                dataset_role="release",
            )
        with self.assertRaisesRegex(ValueError, "cohort_registry"):
            evaluate_v3_candidate(
                Path("candidate"),
                Path("release.json"),
                Path("benchmark.json"),
                dataset_role="release",
                historical_manifest=Path("historical.json"),
            )

    def test_legacy_forbids_release_provenance_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not receive"):
            evaluate_v3_candidate(
                Path("candidate"),
                Path("legacy.json"),
                Path("benchmark.json"),
                dataset_role="legacy",
                historical_manifest=Path("historical.json"),
            )


class V3CandidateGateTests(unittest.TestCase):
    def _artifact(self, lift: float = 0.02) -> dict:
        metrics = HistoricalRelativeMetrics(
            raw_far=0.01,
            adapted_far=0.009,
            recall_lift=lift,
            true_match_delta=1,
            same_threshold_recall_delta=0.01,
            adapted_threshold=0.35,
        )
        return {
            "model_version": "identity-domain-adapter-v3",
            "training": {"historical_oof_relative": metrics.__dict__},
        }

    @patch("tools.evaluate_domain_adapter_v3._load_v2_artifact")
    def test_candidate_must_be_v3_with_passing_historical_oof(
        self,
        load_artifact: Mock,
    ) -> None:
        load_artifact.return_value = (self._artifact(), Path("candidate.onnx"))
        _require_v3_candidate(Path("candidate"))

        load_artifact.return_value = (
            {**self._artifact(), "model_version": "identity-domain-adapter-v2"},
            Path("candidate.onnx"),
        )
        with self.assertRaisesRegex(ValueError, "model_version"):
            _require_v3_candidate(Path("candidate"))

        load_artifact.return_value = (self._artifact(0.019), Path("candidate.onnx"))
        with self.assertRaisesRegex(ValueError, "historical"):
            _require_v3_candidate(Path("candidate"))

    @patch("tools.evaluate_domain_adapter_v3.evaluate_v2_candidate")
    @patch(
        "tools.evaluate_domain_adapter_v3._require_v3_candidate",
        side_effect=ValueError("historical gate failed"),
    )
    def test_failed_historical_gate_stops_before_legacy_or_release_loading(
        self,
        _require_candidate: Mock,
        evaluate_v2: Mock,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "historical"):
            evaluate_v3_candidate(
                Path("candidate"),
                Path("legacy.json"),
                Path("benchmark.json"),
                dataset_role="legacy",
            )
        evaluate_v2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
