from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from tools.evaluate_domain_adapter_v2 import (
    _absolute_gate_passes,
    _relative_gate_passes,
    evaluate_v2_candidate,
    main as evaluation_main,
    v2_gate_passes,
)


def _canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _student_hash(student_id: str) -> str:
    return hashlib.sha256(
        b"identity-domain-adapter-release-student\0" + student_id.encode("utf-8")
    ).hexdigest()


def _trusted_benchmark_patch(benchmark_manifest: Path):
    return patch(
        "tools.evaluate_domain_adapter_v2.TRUSTED_BENCHMARK_MANIFEST_FILE_SHA256",
        _file_sha256(benchmark_manifest),
    )


def _release_relative_metrics_patch():
    return patch(
        "tools.evaluate_domain_adapter_v2._relative_metrics",
        return_value={
            "raw_frozen_threshold": 0.35,
            "adapted_frozen_threshold": 0.40,
            "raw_far": 0.0,
            "adapted_far": 0.0,
            "far_delta": 0.0,
            "raw_recall_at_1pct_budget": 0.80,
            "adapted_recall_at_1pct_budget": 0.85,
            "recall_lift": 0.05,
            "raw_true_matches": 80,
            "adapted_true_matches": 85,
            "true_match_delta": 5,
            "same_threshold_0_35": {
                "threshold": 0.35,
                "raw_recall": 0.80,
                "adapted_recall": 0.85,
                "recall_delta": 0.05,
            },
            "positive_observations": 100,
            "negative_observations": 9800,
        },
    )


class AbsoluteGateTests(unittest.TestCase):
    def test_accepts_exact_absolute_boundaries(self) -> None:
        metrics = {
            "student_split_leaks": 0,
            "cross_student_far": 0.01,
            "conditional_accuracy": 0.70,
            "known_impostor_detected": True,
            "new_false_accusations": 0,
            "onnx_parity_max_abs_error": 1e-5,
        }

        self.assertTrue(_absolute_gate_passes(metrics))

    def test_rejects_each_absolute_conjunct(self) -> None:
        baseline = {
            "student_split_leaks": 0,
            "cross_student_far": 0.01,
            "conditional_accuracy": 0.70,
            "known_impostor_detected": True,
            "new_false_accusations": 0,
            "onnx_parity_max_abs_error": 1e-5,
        }

        for change in (
            {"student_split_leaks": 1},
            {"cross_student_far": 0.0101},
            {"conditional_accuracy": 0.699},
            {"known_impostor_detected": False},
            {"new_false_accusations": 1},
            {"onnx_parity_max_abs_error": 1.1e-5},
        ):
            with self.subTest(change=change):
                self.assertFalse(_absolute_gate_passes({**baseline, **change}))


class RelativeGateTests(unittest.TestCase):
    def test_requires_every_relative_conjunct(self) -> None:
        baseline = {
            "raw_far": 0.01,
            "adapted_far": 0.01,
            "recall_lift": 0.02,
            "true_match_delta": 1,
            "same_threshold_0_35": {
                "threshold": 0.35,
                "raw_recall": 0.70,
                "adapted_recall": 0.71,
                "recall_delta": 0.01,
            },
        }

        self.assertTrue(_relative_gate_passes(baseline))
        for change in (
            {"adapted_far": 0.0101},
            {"recall_lift": 0.019},
            {"true_match_delta": 0},
            {
                "same_threshold_0_35": {
                    "threshold": 0.35,
                    "raw_recall": 0.70,
                    "adapted_recall": 0.70,
                    "recall_delta": 0.0,
                }
            },
        ):
            with self.subTest(change=change):
                self.assertFalse(_relative_gate_passes({**baseline, **change}))


class EvaluateV2CandidateTests(unittest.TestCase):
    def test_engineering_success_never_becomes_release_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )

            with patch(
                "tools.evaluate_domain_adapter_v2.ENGINEERING_DATASET_FILE_SHA256",
                _file_sha256(engineering_manifest),
            ):
                with _trusted_benchmark_patch(benchmark_manifest):
                    report = evaluate_v2_candidate(
                        candidate_dir,
                        engineering_manifest,
                        benchmark_manifest,
                        dataset_role="engineering",
                        historical_manifest=None,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

            self.assertTrue(report["absolute_gate_passed"])
            self.assertTrue(report["relative_gate_passed"])
            self.assertTrue(report["engineering_gate_passed"])
            self.assertFalse(report["release_gate_passed"])
            self.assertEqual(report["cohort_status"], "not_applicable")
            self.assertEqual(report["relative_metrics"]["raw_frozen_threshold"], 0.35)
            self.assertEqual(report["relative_metrics"]["adapted_frozen_threshold"], 0.40)
            self.assertEqual(
                report["relative_metrics"]["same_threshold_0_35"]["threshold"], 0.35
            )
            self.assertTrue(v2_gate_passes(report))

    def test_release_insufficiency_is_not_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=4,
                sessions_per_student=1,
            )
            registry_path = fixture.write_registry(release_manifest)

            with _release_relative_metrics_patch():
                with _trusted_benchmark_patch(benchmark_manifest):
                    report = evaluate_v2_candidate(
                        candidate_dir,
                        release_manifest,
                        benchmark_manifest,
                        dataset_role="release",
                        historical_manifest=fixture.historical_manifest,
                        cohort_registry=registry_path,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

            self.assertEqual(report["cohort_status"], "insufficient_data")
            self.assertFalse(report["release_gate_passed"])
            self.assertFalse(v2_gate_passes(report))

    def test_release_requires_exact_registry_and_historical_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=1,
                sessions_per_student=1,
            )
            registry_path = fixture.write_registry(release_manifest)
            broken_registry = json.loads(registry_path.read_text(encoding="utf-8"))
            broken_registry["cohorts"][0]["student_hashes"] = [
                _student_hash("someone-else")
            ]
            registry_path.write_text(json.dumps(broken_registry), encoding="utf-8")

            with _trusted_benchmark_patch(benchmark_manifest):
                report = evaluate_v2_candidate(
                    candidate_dir,
                    release_manifest,
                    benchmark_manifest,
                    dataset_role="release",
                    historical_manifest=fixture.historical_manifest,
                    cohort_registry=registry_path,
                    pipeline_factory=lambda: fixture.pipeline,
                    inference_session_factory=lambda _path: fixture.inference,
                )

            self.assertFalse(report["release_gate_passed"])
            self.assertFalse(report["provenance"]["registry_entry_matches"])
            self.assertTrue(report["provenance"]["historical_digest_matches_candidate"])

    def test_release_manifest_nonzero_seen_overlap_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=50,
                sessions_per_student=2,
            )
            manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
            manifest["counts"]["excluded_by_reason"]["seen_student_overlap"] = 1
            release_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            registry_path = fixture.write_registry(release_manifest)

            with _release_relative_metrics_patch():
                with _trusted_benchmark_patch(benchmark_manifest):
                    report = evaluate_v2_candidate(
                        candidate_dir,
                        release_manifest,
                        benchmark_manifest,
                        dataset_role="release",
                        historical_manifest=fixture.historical_manifest,
                        cohort_registry=registry_path,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

            self.assertFalse(report["provenance"]["release_excluded_overlap_ok"])
            self.assertEqual(report["cohort_status"], "insufficient_data")
            self.assertFalse(report["release_gate_passed"])

    def test_release_manifest_zero_or_missing_seen_overlap_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )

            for overlap_value in ("missing", 0):
                release_manifest = fixture.build_release_manifest(
                    engineering_manifest=engineering_manifest,
                    student_count=50,
                    sessions_per_student=2,
                )
                manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
                excluded = manifest["counts"]["excluded_by_reason"]
                if overlap_value == "missing":
                    excluded.pop("seen_student_overlap", None)
                else:
                    excluded["seen_student_overlap"] = overlap_value
                release_manifest.write_text(json.dumps(manifest), encoding="utf-8")
                registry_path = fixture.write_registry(release_manifest)

                with self.subTest(overlap_value=overlap_value):
                    with _release_relative_metrics_patch():
                        with _trusted_benchmark_patch(benchmark_manifest):
                            report = evaluate_v2_candidate(
                                candidate_dir,
                                release_manifest,
                                benchmark_manifest,
                                dataset_role="release",
                                historical_manifest=fixture.historical_manifest,
                                cohort_registry=registry_path,
                                pipeline_factory=lambda: fixture.pipeline,
                                inference_session_factory=lambda _path: fixture.inference,
                            )

                    self.assertTrue(report["provenance"]["release_excluded_overlap_ok"])
                    self.assertEqual(report["cohort_status"], "sufficient")
                    self.assertTrue(report["release_gate_passed"])

    def test_release_manifest_malformed_seen_overlap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )

            for overlap_value in (True, -1, 1.5, "1"):
                release_manifest = fixture.build_release_manifest(
                    engineering_manifest=engineering_manifest,
                    student_count=50,
                    sessions_per_student=2,
                )
                manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
                manifest["counts"]["excluded_by_reason"]["seen_student_overlap"] = overlap_value
                release_manifest.write_text(json.dumps(manifest), encoding="utf-8")
                registry_path = fixture.write_registry(release_manifest)

                with self.subTest(overlap_value=overlap_value):
                    with _release_relative_metrics_patch():
                        with _trusted_benchmark_patch(benchmark_manifest):
                            report = evaluate_v2_candidate(
                                candidate_dir,
                                release_manifest,
                                benchmark_manifest,
                                dataset_role="release",
                                historical_manifest=fixture.historical_manifest,
                                cohort_registry=registry_path,
                                pipeline_factory=lambda: fixture.pipeline,
                                inference_session_factory=lambda _path: fixture.inference,
                            )

                    self.assertFalse(report["provenance"]["release_excluded_overlap_ok"])
                    self.assertEqual(report["cohort_status"], "insufficient_data")
                    self.assertFalse(report["release_gate_passed"])

    def test_rejects_self_consistent_benchmark_replacement_without_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            replacement_root = root / "replacement-benchmark"
            replacement_root.mkdir()
            replacement_manifest = fixture._build_benchmark_manifest(
                dimension=1024,
                benchmark_root=replacement_root,
            )

            with patch(
                "tools.evaluate_domain_adapter_v2.ENGINEERING_DATASET_FILE_SHA256",
                _file_sha256(engineering_manifest),
            ):
                with patch(
                    "tools.evaluate_domain_adapter_v2.TRUSTED_BENCHMARK_MANIFEST_FILE_SHA256",
                    _file_sha256(benchmark_manifest),
                ):
                    with self.assertRaisesRegex(
                        ValueError, "trusted benchmark manifest file SHA256"
                    ):
                        evaluate_v2_candidate(
                            candidate_dir,
                            engineering_manifest,
                            replacement_manifest,
                            dataset_role="engineering",
                            historical_manifest=None,
                            pipeline_factory=lambda: fixture.pipeline,
                            inference_session_factory=lambda _path: fixture.inference,
                        )

    def test_release_recomputes_actual_sufficiency_instead_of_trusting_claimed_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=1,
                sessions_per_student=1,
            )
            manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
            manifest["sufficiency"] = {
                "status": "sufficient",
                "unseen_students": 50,
                "minimum_unseen_students": 50,
                "adapter_eligible_truth_match_sessions": 100,
                "minimum_adapter_eligible_truth_match_sessions": 100,
                "ordered_cross_student_session_pairs": 2000,
                "minimum_ordered_cross_student_session_pairs": 2000,
            }
            release_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            registry_path = fixture.write_registry(release_manifest)

            with _release_relative_metrics_patch():
                with _trusted_benchmark_patch(benchmark_manifest):
                    report = evaluate_v2_candidate(
                        candidate_dir,
                        release_manifest,
                        benchmark_manifest,
                        dataset_role="release",
                        historical_manifest=fixture.historical_manifest,
                        cohort_registry=registry_path,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

            self.assertEqual(report["cohort_status"], "insufficient_data")
            self.assertEqual(report["cohort_sufficiency"]["unseen_students"], 1)
            self.assertEqual(
                report["cohort_sufficiency"]["adapter_eligible_truth_match_sessions"], 1
            )
            self.assertEqual(
                report["cohort_sufficiency"]["ordered_cross_student_session_pairs"], 0
            )
            self.assertFalse(report["release_gate_passed"])

    def test_release_duplicate_eligible_rows_count_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=50,
                sessions_per_student=1,
                duplicate_copies=2,
            )
            registry_path = fixture.write_registry(release_manifest)

            with _release_relative_metrics_patch():
                with _trusted_benchmark_patch(benchmark_manifest):
                    report = evaluate_v2_candidate(
                        candidate_dir,
                        release_manifest,
                        benchmark_manifest,
                        dataset_role="release",
                        historical_manifest=fixture.historical_manifest,
                        cohort_registry=registry_path,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

            self.assertEqual(report["cohort_sufficiency"]["unseen_students"], 50)
            self.assertEqual(
                report["cohort_sufficiency"]["adapter_eligible_truth_match_sessions"], 50
            )
            self.assertEqual(
                report["cohort_sufficiency"]["ordered_cross_student_session_pairs"], 2450
            )
            self.assertEqual(report["cohort_status"], "insufficient_data")
            self.assertFalse(report["release_gate_passed"])

    def test_mismatch_only_students_do_not_count_toward_unseen_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=50,
                sessions_per_student=1,
            )
            manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
            for row in manifest["evaluation_sessions"][:48]:
                row["label"] = "mismatch"
            release_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            registry_path = fixture.write_registry(release_manifest)

            with _release_relative_metrics_patch():
                with _trusted_benchmark_patch(benchmark_manifest):
                    report = evaluate_v2_candidate(
                        candidate_dir,
                        release_manifest,
                        benchmark_manifest,
                        dataset_role="release",
                        historical_manifest=fixture.historical_manifest,
                        cohort_registry=registry_path,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

            self.assertEqual(report["cohort_sufficiency"]["unseen_students"], 2)
            self.assertEqual(
                report["cohort_sufficiency"]["adapter_eligible_truth_match_sessions"], 2
            )
            self.assertEqual(
                report["cohort_sufficiency"]["ordered_cross_student_session_pairs"], 2
            )
            self.assertEqual(report["cohort_status"], "insufficient_data")
            self.assertFalse(report["release_gate_passed"])

    def test_release_requires_historical_manifest_only_for_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=1,
                sessions_per_student=1,
            )
            registry_path = fixture.write_registry(release_manifest)

            with _trusted_benchmark_patch(benchmark_manifest):
                with self.assertRaisesRegex(ValueError, "release evaluation requires historical_manifest"):
                    evaluate_v2_candidate(
                        candidate_dir,
                        release_manifest,
                        benchmark_manifest,
                        dataset_role="release",
                        historical_manifest=None,
                        cohort_registry=registry_path,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )
                with self.assertRaisesRegex(ValueError, "engineering evaluation must not receive historical_manifest"):
                    evaluate_v2_candidate(
                        candidate_dir,
                        engineering_manifest,
                        benchmark_manifest,
                        dataset_role="engineering",
                        historical_manifest=fixture.historical_manifest,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

    def test_release_rejects_historical_student_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=50,
                sessions_per_student=2,
            )
            manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
            manifest["evaluation_sessions"][0]["student_id"] = "eng-student-000"
            release_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            registry_path = fixture.write_registry(release_manifest)

            with _release_relative_metrics_patch():
                with _trusted_benchmark_patch(benchmark_manifest):
                    report = evaluate_v2_candidate(
                        candidate_dir,
                        release_manifest,
                        benchmark_manifest,
                        dataset_role="release",
                        historical_manifest=fixture.historical_manifest,
                        cohort_registry=registry_path,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

            self.assertFalse(report["provenance"]["historical_student_overlap_free"])
            self.assertEqual(report["cohort_status"], "insufficient_data")
            self.assertFalse(report["release_gate_passed"])

    def test_release_rejects_current_hash_reused_in_earlier_registry_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=50,
                sessions_per_student=2,
            )
            registry_path = fixture.write_registry(
                release_manifest,
                extra_prior_hashes=[_student_hash("release-student-000")],
            )

            with _release_relative_metrics_patch():
                with _trusted_benchmark_patch(benchmark_manifest):
                    report = evaluate_v2_candidate(
                        candidate_dir,
                        release_manifest,
                        benchmark_manifest,
                        dataset_role="release",
                        historical_manifest=fixture.historical_manifest,
                        cohort_registry=registry_path,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

            self.assertFalse(report["provenance"]["registry_hashes_unique_to_current"])
            self.assertEqual(report["cohort_status"], "insufficient_data")
            self.assertFalse(report["release_gate_passed"])

    def test_report_never_leaks_raw_identifiers_paths_or_group_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )

            with patch(
                "tools.evaluate_domain_adapter_v2.ENGINEERING_DATASET_FILE_SHA256",
                _file_sha256(engineering_manifest),
            ):
                with _trusted_benchmark_patch(benchmark_manifest):
                    report = evaluate_v2_candidate(
                        candidate_dir,
                        engineering_manifest,
                        benchmark_manifest,
                        dataset_role="engineering",
                        historical_manifest=None,
                        pipeline_factory=lambda: fixture.pipeline,
                        inference_session_factory=lambda _path: fixture.inference,
                    )

            encoded = json.dumps(report, sort_keys=True)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn("eng-student-00", encoded)
            self.assertNotIn("release-student-000", encoded)
            self.assertNotIn("benchmark-01", encoded)
            self.assertNotIn("group-", encoded)


class EvaluateV2CliTests(unittest.TestCase):
    def test_cli_requires_cohort_registry_for_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=1,
                sessions_per_student=1,
            )
            output = root / "release-report.json"

            with redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main(
                        [
                            "--candidate-dir",
                            str(candidate_dir),
                            "--dataset-manifest",
                            str(release_manifest),
                            "--historical-manifest",
                            str(fixture.historical_manifest),
                            "--benchmark-manifest",
                            str(benchmark_manifest),
                            "--dataset-role",
                            "release",
                            "--output",
                            str(output),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--cohort-registry is required", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_cli_requires_historical_manifest_for_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=1,
                sessions_per_student=1,
            )
            registry_path = fixture.write_registry(release_manifest)
            output = root / "release-report.json"

            with redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main(
                        [
                            "--candidate-dir",
                            str(candidate_dir),
                            "--dataset-manifest",
                            str(release_manifest),
                            "--benchmark-manifest",
                            str(benchmark_manifest),
                            "--dataset-role",
                            "release",
                            "--cohort-registry",
                            str(registry_path),
                            "--output",
                            str(output),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--historical-manifest is required", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_cli_forbids_cohort_registry_for_engineering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            release_manifest = fixture.build_release_manifest(
                engineering_manifest=engineering_manifest,
                student_count=50,
                sessions_per_student=2,
            )
            registry_path = fixture.write_registry(release_manifest)
            output = root / "engineering-report.json"

            with redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main(
                        [
                            "--candidate-dir",
                            str(candidate_dir),
                            "--dataset-manifest",
                            str(engineering_manifest),
                            "--benchmark-manifest",
                            str(benchmark_manifest),
                            "--dataset-role",
                            "engineering",
                            "--cohort-registry",
                            str(registry_path),
                            "--output",
                            str(output),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("must not be set for engineering", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_cli_forbids_historical_manifest_for_engineering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            output = root / "engineering-report.json"

            with redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main(
                        [
                            "--candidate-dir",
                            str(candidate_dir),
                            "--dataset-manifest",
                            str(engineering_manifest),
                            "--historical-manifest",
                            str(fixture.historical_manifest),
                            "--benchmark-manifest",
                            str(benchmark_manifest),
                            "--dataset-role",
                            "engineering",
                            "--output",
                            str(output),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("must not be set for engineering", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_cli_refuses_existing_output_and_writes_private_redacted_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _V2Fixture(root)
            candidate_dir, engineering_manifest, benchmark_manifest = (
                fixture.build_engineering_fixture()
            )
            output = root / "engineering-report.json"

            with patch(
                "tools.evaluate_domain_adapter_v2.ENGINEERING_DATASET_FILE_SHA256",
                _file_sha256(engineering_manifest),
            ):
                with _trusted_benchmark_patch(benchmark_manifest):
                    with patch(
                        "tools.evaluate_domain_adapter_v2.build_pipeline_from_env",
                        return_value=fixture.pipeline,
                    ):
                        with patch(
                            "tools.evaluate_domain_adapter_v2._load_onnx_session",
                            return_value=fixture.inference,
                        ):
                            exit_code = evaluation_main(
                                [
                                    "--candidate-dir",
                                    str(candidate_dir),
                                    "--dataset-manifest",
                                    str(engineering_manifest),
                                    "--benchmark-manifest",
                                    str(benchmark_manifest),
                                    "--dataset-role",
                                    "engineering",
                                    "--output",
                                    str(output),
                                ]
                            )

            report = json.loads(output.read_text(encoding="utf-8"))
            encoded = json.dumps(report, sort_keys=True)
            self.assertEqual(exit_code, 0)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn("eng-student-00", encoded)

            with redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    evaluation_main(
                        [
                            "--candidate-dir",
                            str(candidate_dir),
                            "--dataset-manifest",
                            str(engineering_manifest),
                            "--benchmark-manifest",
                            str(benchmark_manifest),
                            "--dataset-role",
                            "engineering",
                            "--output",
                            str(output),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("output already exists", stderr.getvalue())


class _V2Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pipeline = _EvaluationPipeline()
        self.inference = _BoostingAdapterSession()
        self._marker = 1
        self.historical_manifest = self.root / "historical-manifest.json"

    def build_engineering_fixture(self) -> tuple[Path, Path, Path]:
        candidate_dir = self.root / "candidate"
        candidate_dir.mkdir()
        engineering_manifest = self.root / "engineering-dataset.json"
        benchmark_manifest = self._build_benchmark_manifest(dimension=1024)
        dataset = self._engineering_manifest(dimension=1024)
        engineering_manifest.write_text(json.dumps(dataset), encoding="utf-8")
        self.historical_manifest.write_text(json.dumps(dataset), encoding="utf-8")
        self._write_candidate_manifest(
            candidate_dir,
            training_digest=_canonical_sha256(dataset),
            dimension=1024,
            adapted_threshold=0.40,
            raw_threshold=0.35,
        )
        return candidate_dir, engineering_manifest, benchmark_manifest

    def build_release_manifest(
        self,
        *,
        engineering_manifest: Path,
        student_count: int,
        sessions_per_student: int,
        duplicate_copies: int = 1,
    ) -> Path:
        manifest = json.loads(engineering_manifest.read_text(encoding="utf-8"))
        historical_digest = _canonical_sha256(manifest)
        evaluation_sessions = self._release_rows(
            dimension=1024,
            student_count=student_count,
            sessions_per_student=sessions_per_student,
            duplicate_copies=duplicate_copies,
        )
        eligible_sessions = len(evaluation_sessions)
        unseen_students = student_count
        ordered_pairs = eligible_sessions * eligible_sessions - student_count * (
            sessions_per_student * sessions_per_student
        )
        release_manifest = {
            "schema_version": 1,
            "dataset_role": "release",
            "cohort_id": "release-20260830T050122Z-20260831T000000Z",
            "after": "2026-08-30T05:01:22+00:00",
            "through": "2026-08-31T00:00:00+00:00",
            "historical_manifest_digest": historical_digest,
            "prior_cohort_registry_digest": "a" * 64,
            "prior_cohort_registry_summary": {
                "cohort_count": 0,
                "student_hash_count": 0,
                "cohort_ids": [],
            },
            "sessions": [],
            "evaluation_sessions": evaluation_sessions,
            "counts": {
                "reviewed_sessions": eligible_sessions,
                "eligible_sessions": eligible_sessions,
                "training_sessions": 0,
                "evaluation_sessions": eligible_sessions,
                "positive_sessions": eligible_sessions,
                "negative_sessions": 0,
                "excluded_sessions": 0,
                "excluded_by_reason": {},
                "feedback_load": {},
            },
            "sufficiency": {
                "status": (
                    "sufficient"
                    if unseen_students >= 50
                    and eligible_sessions >= 100
                    and ordered_pairs >= 2000
                    else "insufficient_data"
                ),
                "unseen_students": unseen_students,
                "minimum_unseen_students": 50,
                "adapter_eligible_truth_match_sessions": eligible_sessions,
                "minimum_adapter_eligible_truth_match_sessions": 100,
                "ordered_cross_student_session_pairs": ordered_pairs,
                "minimum_ordered_cross_student_session_pairs": 2000,
            },
        }
        path = self.root / f"release-{student_count}x{sessions_per_student}.json"
        path.write_text(json.dumps(release_manifest), encoding="utf-8")
        return path

    def write_registry(
        self,
        release_manifest: Path,
        *,
        extra_prior_hashes: list[str] | None = None,
    ) -> Path:
        manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
        prior_hashes = list(extra_prior_hashes or [])
        registry = {
            "schema_version": 1,
            "hash_scheme": "sha256-domain-v1",
            "cohorts": (
                [
                    {
                        "cohort_id": "release-20260829T000000Z-20260830T050122Z",
                        "after": "2026-08-29T00:00:00+00:00",
                        "through": "2026-08-30T05:01:22+00:00",
                        "student_hashes": prior_hashes,
                    }
                ]
                if prior_hashes
                else []
            )
            + [
                {
                    "cohort_id": manifest["cohort_id"],
                    "after": manifest["after"],
                    "through": manifest["through"],
                    "student_hashes": sorted(
                        {
                            _student_hash(str(row["student_id"]))
                            for row in manifest["evaluation_sessions"]
                        }
                    ),
                }
            ],
        }
        path = self.root / "registry.json"
        path.write_text(json.dumps(registry), encoding="utf-8")
        return path

    def _engineering_manifest(self, *, dimension: int) -> dict[str, object]:
        evaluation_sessions = self._release_rows(
            dimension=dimension,
            student_count=4,
            sessions_per_student=1,
            student_prefix="eng-student",
            session_prefix="eng-session",
        )
        return {
            "schema_version": 1,
            "snapshot": "2026-08-30T05:01:22+00:00",
            "split_seed": "seed-v1",
            "sessions": [
                {
                    "student_id": "train-only",
                    "session_id": "train-session",
                    "split": "train",
                    "label": "match",
                    "ref_image_path": "",
                    "photos": [],
                    "training_exclusion_reasons": [],
                }
            ],
            "evaluation_sessions": evaluation_sessions,
        }

    def _release_rows(
        self,
        *,
        dimension: int,
        student_count: int,
        sessions_per_student: int,
        student_prefix: str = "release-student",
        session_prefix: str = "release-session",
        duplicate_copies: int = 1,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        next_axis = 0
        for student_index in range(student_count):
            for session_index in range(sessions_per_student):
                student_id = f"{student_prefix}-{student_index:03d}"
                session_id = f"{session_prefix}-{student_index:03d}-{session_index:02d}"
                ref_path = self.root / f"{session_id}-ref.png"
                photo_path = self.root / f"{session_id}-photo.png"
                self._write_vector_image(ref_path, self._allocate_marker())
                self._write_vector_image(photo_path, self._allocate_marker())
                ref = np.zeros(dimension, dtype=np.float32)
                ref[next_axis] = 1.0
                positive_score = 0.34 if session_index % 2 == 0 else 0.38
                photo = self._noisy_same_axis_embedding(
                    dimension=dimension,
                    anchor_axis=next_axis,
                    start_axis=student_count * sessions_per_student + next_axis * 8,
                    cosine=positive_score,
                )
                self.pipeline.add_session(
                    ref_path=str(ref_path),
                    photo_path=str(photo_path),
                    status="match",
                    consistency="consistent",
                    ref_embedding=ref,
                    photo_embedding=photo,
                )
                row = {
                    "student_id": student_id,
                    "session_id": session_id,
                    "split": "test",
                    "label": "match",
                    "ref_image_path": str(ref_path),
                    "photos": [
                        {
                            "sequence_no": 1,
                            "photo_type": "sign_in",
                            "image_path": str(photo_path),
                        }
                    ],
                    "training_exclusion_reasons": ["test_split"],
                    "source_feedback_ids": [f"feedback-{session_id}"],
                }
                rows.extend(json.loads(json.dumps(row)) for _ in range(duplicate_copies))
                next_axis += 1
        return rows

    def _build_benchmark_manifest(
        self,
        *,
        dimension: int,
        benchmark_root: Path | None = None,
    ) -> Path:
        if benchmark_root is None:
            benchmark_root = self.root / "benchmark-view"
        benchmark_root.mkdir(exist_ok=True)
        manifest_path = benchmark_root / "benchmark-manifest.json"
        entries = []
        known_student = "S177509932310186"
        base_axis = 900
        for index in range(40):
            student_id = known_student if index == 0 else f"benchmark-{index:02d}"
            session_id = f"benchmark-session-{index:02d}"
            session_dir = benchmark_root / student_id / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            ref_path = session_dir / "ref.png"
            photo_path = session_dir / "photo.png"
            self._write_vector_image(ref_path, self._allocate_marker())
            self._write_vector_image(photo_path, self._allocate_marker())
            ref = np.zeros(dimension, dtype=np.float32)
            ref[base_axis + index] = 1.0
            photo = -ref if index == 0 else ref.copy()
            self.pipeline.add_session(
                ref_path=str(ref_path),
                photo_path=str(photo_path),
                status="mismatch" if index == 0 else "match",
                consistency="consistent",
                ref_embedding=ref,
                photo_embedding=photo,
            )
            record = {
                "student_id": student_id,
                "session_id": session_id,
                "request": {
                    "ref_image_path": str(ref_path),
                    "photos": [
                        {
                            "sequence_no": 1,
                            "photo_type": "sign_in",
                            "image_path": str(photo_path),
                        }
                    ],
                },
            }
            record_path = session_dir / "record.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            entries.append(
                {
                    "student_id": student_id,
                    "session_id": session_id,
                    "record_path": record_path.relative_to(benchmark_root).as_posix(),
                    "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                    "truth": "mismatch" if index == 0 else "match",
                }
            )
        payload = {
            "schema_version": 1,
            "benchmark_id": "synthetic-40-session-v1",
            "known_impostor": {
                "student_id": known_student,
                "session_id": "benchmark-session-00",
            },
            "sessions": entries,
        }
        manifest = {
            **payload,
            "manifest_sha256": _canonical_sha256(payload),
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return manifest_path

    def _write_candidate_manifest(
        self,
        candidate_dir: Path,
        *,
        training_digest: str,
        dimension: int,
        adapted_threshold: float,
        raw_threshold: float,
    ) -> None:
        onnx_path = candidate_dir / "identity_domain_adapter.onnx"
        onnx_path.write_bytes(b"synthetic-v2-onnx")
        artifact = {
            "schema_version": 1,
            "model_version": "identity-domain-adapter-v2",
            "embedding_dimension": dimension,
            "rank": 16,
            "match_threshold": adapted_threshold,
            "onnx_file": onnx_path.name,
            "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "onnx_parity_max_abs_error": 1e-6,
            "source_dataset_sha256": training_digest,
            "source_code_revision": "deadbeef",
            "split_seed": "seed-v2",
            "training_hyperparameters": {"seed": 17},
            "validation_metrics": {
                "threshold": adapted_threshold,
                "empirical_far": 0.0,
                "far_upper_95": 0.0,
                "student_balanced_recall": 1.0,
                "true_matches": 4,
                "feasible": True,
            },
            "test_metrics": None,
            "input_names": ["ref_embedding", "photo_embedding"],
            "output_name": "adapted_cosine",
            "training": {
                "strategy": "five_fold_student_oof_v2",
                "adapted_threshold": {
                    "threshold": adapted_threshold,
                    "empirical_far": 0.0,
                    "far_upper_95": 0.0,
                    "student_balanced_recall": 1.0,
                    "true_matches": 4,
                    "feasible": True,
                },
                "raw_comparator_threshold": {
                    "threshold": raw_threshold,
                    "empirical_far": 0.0,
                    "far_upper_95": 0.0,
                    "student_balanced_recall": 0.5,
                    "true_matches": 2,
                    "feasible": True,
                },
                "canonical_dataset_digest": training_digest,
            },
        }
        (candidate_dir / "identity_domain_adapter.manifest.json").write_text(
            json.dumps(artifact), encoding="utf-8"
        )

    def _allocate_marker(self) -> int:
        marker = self._marker
        self._marker += 1
        return marker

    @staticmethod
    def _write_vector_image(path: Path, value: int) -> None:
        encoded = np.asarray(
            [value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF],
            dtype=np.uint8,
        )
        image = np.tile(encoded, (2, 2, 1))
        if not cv2.imwrite(str(path), image):
            raise AssertionError(f"failed to write {path}")

    @staticmethod
    def _noisy_same_axis_embedding(
        *,
        dimension: int,
        anchor_axis: int,
        start_axis: int,
        cosine: float,
    ) -> np.ndarray:
        vector = np.zeros(dimension, dtype=np.float32)
        vector[anchor_axis] = cosine
        remainder = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
        extra_value = remainder / np.sqrt(8.0)
        for offset in range(8):
            vector[start_axis + offset] = extra_value
        vector /= np.linalg.norm(vector)
        return vector.astype(np.float32, copy=False)


class _EvaluationPipeline:
    def __init__(self) -> None:
        self._raw: dict[str, tuple[str, str]] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def add_session(
        self,
        *,
        ref_path: str,
        photo_path: str,
        status: str,
        consistency: str,
        ref_embedding: np.ndarray,
        photo_embedding: np.ndarray,
    ) -> None:
        self._raw[ref_path] = (status, consistency)
        self._vectors[ref_path] = ref_embedding
        self._vectors[photo_path] = photo_embedding

    def session_check(self, ref_image_path: str, _photos: list[dict[str, object]]) -> SimpleNamespace:
        status, consistency = self._raw[ref_image_path]
        return SimpleNamespace(
            session_status=status,
            internal_consistency=consistency,
        )

    def _extract_or_raise(self, image: np.ndarray) -> np.ndarray:
        marker = tuple(int(channel) for channel in image[0, 0, :3])
        for path, vector in self._vectors.items():
            loaded = cv2.imread(path)
            if loaded is not None and tuple(int(channel) for channel in loaded[0, 0, :3]) == marker:
                return vector
        raise RuntimeError(f"unknown image marker {marker}")

    def _process_photo_for_session(
        self, photo: dict[str, object]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            passes_gate=True,
            _embedding=self._vectors[str(photo["image_path"])],
        )


class _BoostingAdapterSession:
    def run(self, outputs: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if outputs != ["adapted_cosine"]:
            raise AssertionError(outputs)
        refs = np.asarray(inputs["ref_embedding"], dtype=np.float32)
        photos = np.asarray(inputs["photo_embedding"], dtype=np.float32)
        base = np.sum(refs * photos, axis=1)
        boost = (np.argmax(refs, axis=1) == np.argmax(photos, axis=1)).astype(np.float32)
        return [(base + boost * 0.10).astype(np.float32)]


if __name__ == "__main__":
    unittest.main()
