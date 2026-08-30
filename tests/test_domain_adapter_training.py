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
import onnxruntime as ort
import torch

from tools.domain_adapter_training import (
    LowRankDomainAdapter,
    PairSet,
    SessionEmbedding,
    build_pair_sets,
    contrastive_margin_loss,
    export_onnx,
    extract_session_embeddings,
    manifest_training_rows,
    select_threshold,
    train_adapter,
    write_candidate_artifacts,
)
from tools.evaluate_domain_adapter import (
    KNOWN_IMPOSTOR_STUDENT_ID,
    evaluate_release_candidate,
    main as evaluation_main,
    release_gate_passes,
)
from tools.train_domain_adapter import main as training_main


class LowRankDomainAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.refs = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.photos = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 0.6, 0.0],
                [0.0, 0.8, 0.0, 0.6],
            ]
        )
        self.labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    def test_zero_initialization_is_identity(self) -> None:
        model = LowRankDomainAdapter(dimension=128, rank=16)
        ref = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)
        photo = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)

        adapted = model(ref, photo)
        expected = (ref * photo).sum(dim=1)

        torch.testing.assert_close(adapted, expected)

    def test_one_training_step_pulls_positive_and_pushes_negative(self) -> None:
        torch.manual_seed(7)
        model = LowRankDomainAdapter(dimension=4, rank=2)
        before = model(self.refs, self.photos).detach()
        training_scores = model(self.refs, self.photos)
        loss = contrastive_margin_loss(
            training_scores,
            self.labels,
            positive_margin=0.35,
            negative_margin=0.15,
        )
        loss.backward()
        torch.optim.SGD(model.parameters(), lr=0.1).step()

        after = model(self.refs, self.photos).detach()

        self.assertGreater(
            after[self.labels == 1].mean(), before[self.labels == 1].mean()
        )
        self.assertLess(after[self.labels == 0].mean(), before[self.labels == 0].mean())


class PairConstructionTests(unittest.TestCase):
    @staticmethod
    def _session(student: str, session: str, split: str, vector: list[float]):
        embedding = np.asarray(vector, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        return SessionEmbedding(
            student_id=student,
            session_id=session,
            split=split,
            label="match",
            ref_embedding=embedding,
            session_prototype=embedding,
        )

    def test_training_mines_at_most_twenty_hardest_negatives_per_positive(self) -> None:
        anchor = self._session("anchor", "s0", "train", [1.0, 0.0])
        others = [
            self._session(
                f"student-{index:02d}",
                f"s{index:02d}",
                "train",
                [1.0, (index + 1) / 100.0],
            )
            for index in range(25)
        ]

        pairs = build_pair_sets([anchor, *others])["train"]
        anchor_negatives = [
            item
            for item in pairs.metadata
            if item.ref_session_id == "s0" and item.label == 0
        ]

        self.assertEqual(len(anchor_negatives), 20)
        self.assertEqual(
            [item.photo_session_id for item in anchor_negatives[:3]],
            ["s00", "s01", "s02"],
        )

    def test_validation_keeps_every_cross_student_pair(self) -> None:
        sessions = [
            self._session("a", "one", "validation", [1.0, 0.0]),
            self._session("b", "two", "validation", [0.0, 1.0]),
            self._session("c", "three", "validation", [-1.0, 0.0]),
        ]

        pairs = build_pair_sets(sessions)["validation"]

        self.assertEqual(int((pairs.labels == 1).sum()), 3)
        self.assertEqual(int((pairs.labels == 0).sum()), 6)

    def test_manifest_consumption_is_sorted_and_keeps_only_clean_test_rows(
        self,
    ) -> None:
        manifest = {
            "schema_version": 1,
            "sessions": [
                self._row("b", "two", "validation"),
                self._row("a", "one", "train"),
            ],
            "evaluation_sessions": [
                self._row("d", "four", "test", ["test_split", "photo_error"]),
                self._row("c", "three", "test", ["test_split"]),
            ],
        }

        rows = manifest_training_rows(manifest)

        self.assertEqual(
            [(row["split"], row["student_id"]) for row in rows],
            [("train", "a"), ("validation", "b"), ("test", "c")],
        )

    @staticmethod
    def _row(
        student: str,
        session: str,
        split: str,
        reasons: list[str] | None = None,
    ) -> dict:
        row = {
            "student_id": student,
            "session_id": session,
            "split": split,
            "label": "match",
            "ref_image_path": f"/{student}/ref.jpg",
            "photos": [{"sequence_no": 1, "image_path": f"/{student}/photo.jpg"}],
        }
        if reasons is not None:
            row["training_exclusion_reasons"] = reasons
        return row


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_lowest_threshold_with_best_recall_under_far_ceiling(self) -> None:
        scores = np.asarray([0.40, 0.30, 0.25, 0.10], dtype=np.float32)
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)

        metrics = select_threshold(scores, labels, max_false_accept_rate=0.0)

        self.assertAlmostEqual(metrics.threshold, 0.250, places=6)
        self.assertEqual(metrics.true_accept_rate, 1.0)
        self.assertEqual(metrics.false_accept_rate, 0.0)


class ReleaseGateTests(unittest.TestCase):
    @staticmethod
    def _passing_report() -> dict:
        return {
            "cross_student_far": 0.01,
            "conditional_accuracy": 0.70,
            "known_impostor_detected": True,
            "new_false_accusations": 0,
            "student_split_leaks": 0,
        }

    def test_release_gate_accepts_exact_far_and_accuracy_boundaries(self) -> None:
        self.assertTrue(release_gate_passes(self._passing_report()))

    def test_release_gate_rejects_false_accept_rate_above_one_percent(self) -> None:
        report = {**self._passing_report(), "cross_student_far": 0.0101}
        self.assertFalse(release_gate_passes(report))

    def test_release_gate_requires_seventy_percent_conditional_accuracy(self) -> None:
        report = {**self._passing_report(), "conditional_accuracy": 0.699}
        self.assertFalse(release_gate_passes(report))

    def test_release_gate_requires_impostor_and_zero_accusations_and_leaks(self) -> None:
        for change in (
            {"known_impostor_detected": False},
            {"new_false_accusations": 1},
            {"student_split_leaks": 1},
        ):
            with self.subTest(change=change):
                self.assertFalse(release_gate_passes({**self._passing_report(), **change}))


class HeldOutReleaseEvaluationTests(unittest.TestCase):
    def test_evaluates_every_test_session_and_cross_student_pair_without_adapting_stage_one_failures(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            pipeline = _EvaluationPipeline(vectors, raw_results)
            inference = _DotAdapterSession()

            report = evaluate_release_candidate(
                dataset_path,
                adapter_dir,
                benchmark,
                pipeline_factory=lambda: pipeline,
                inference_session_factory=lambda _path: inference,
            )

            self.assertEqual(pipeline.session_checks, 45)
            self.assertEqual(inference.scored_pairs, 48)
            self.assertEqual(report["held_out_test_sessions"], 5)
            self.assertEqual(report["adapter_eligible_test_sessions"], 3)
            self.assertEqual(report["cross_student_pairs"], 6)
            self.assertEqual(report["cross_student_false_accepts"], 0)
            self.assertEqual(report["cross_student_far"], 0.0)
            self.assertEqual(report["conditional_accuracy"], 0.75)
            self.assertEqual(report["raw_confusion"]["match"]["inconclusive"], 2)
            self.assertEqual(report["adapted_confusion"]["match"]["match"], 2)
            self.assertEqual(report["adapted_confusion"]["match"]["mismatch"], 1)
            self.assertEqual(report["truth_reconstruction"]["implicit_sessions"], 1)
            self.assertTrue(report["known_impostor_detected"])
            self.assertEqual(report["new_false_accusations"], 0)
            self.assertEqual(report["benchmark_sessions"], 40)
            benchmark_manifest = json.loads(
                (benchmark / "benchmark-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["benchmark_manifest_id"], "synthetic-40-session-v1"
            )
            self.assertEqual(
                report["benchmark_manifest_sha256"],
                benchmark_manifest["manifest_sha256"],
            )
            self.assertEqual(
                report["benchmark_manifest_file_sha256"],
                hashlib.sha256(
                    (benchmark / "benchmark-manifest.json").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(report["student_split_leaks"], 0)
            self.assertTrue(report["release_gate_passed"])

    def test_rejects_a_different_session_set_even_when_count_remains_forty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            source = benchmark / "benchmark-01" / "session-01"
            replacement_student = benchmark / "replacement-student"
            replacement_student.mkdir()
            source.rename(replacement_student / "replacement-session")
            source.parent.rmdir()
            pipeline = _EvaluationPipeline(vectors, raw_results)

            with self.assertRaisesRegex(ValueError, "record paths"):
                evaluate_release_candidate(
                    dataset_path,
                    adapter_dir,
                    benchmark,
                    pipeline_factory=lambda: pipeline,
                    inference_session_factory=lambda _path: _DotAdapterSession(),
                )

            self.assertEqual(pipeline.session_checks, 0)

    def test_rejects_record_content_tampering_without_count_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            record = benchmark / "benchmark-01" / "session-01" / "record.json"
            record.write_text(record.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "record SHA256"):
                evaluate_release_candidate(
                    dataset_path,
                    adapter_dir,
                    benchmark,
                    pipeline_factory=lambda: _EvaluationPipeline(vectors, raw_results),
                    inference_session_factory=lambda _path: _DotAdapterSession(),
                )

    def test_rejects_benchmark_manifest_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            manifest_path = benchmark / "benchmark-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["manifest_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                evaluate_release_candidate(
                    dataset_path,
                    adapter_dir,
                    benchmark,
                    pipeline_factory=lambda: _EvaluationPipeline(vectors, raw_results),
                    inference_session_factory=lambda _path: _DotAdapterSession(),
                )

    def test_rejects_a_manifest_entry_mismatch_even_with_a_valid_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            manifest_path = benchmark / "benchmark-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sessions"][1]["record_sha256"] = "0" * 64
            self._write_benchmark_manifest(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "record SHA256"):
                evaluate_release_candidate(
                    dataset_path,
                    adapter_dir,
                    benchmark,
                    pipeline_factory=lambda: _EvaluationPipeline(vectors, raw_results),
                    inference_session_factory=lambda _path: _DotAdapterSession(),
                )

    def test_reports_each_student_present_in_test_and_training_as_one_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            dataset["sessions"].append(
                {
                    **dataset["evaluation_sessions"][0],
                    "session_id": "leaked-train-session",
                    "split": "train",
                }
            )
            self._write_dataset_and_bind_artifact(dataset_path, adapter_dir, dataset)

            report = evaluate_release_candidate(
                dataset_path,
                adapter_dir,
                benchmark,
                pipeline_factory=lambda: _EvaluationPipeline(vectors, raw_results),
                inference_session_factory=lambda _path: _DotAdapterSession(),
            )

            self.assertEqual(report["student_split_leaks"], 1)
            self.assertEqual(report["student_split_leak_ids"], ["test-a"])
            self.assertFalse(report["release_gate_passed"])

    def test_reports_students_shared_by_train_and_validation_as_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            dataset["sessions"].append(
                {
                    **dataset["sessions"][0],
                    "session_id": "validation-session",
                    "split": "validation",
                }
            )
            self._write_dataset_and_bind_artifact(dataset_path, adapter_dir, dataset)

            report = evaluate_release_candidate(
                dataset_path,
                adapter_dir,
                benchmark,
                pipeline_factory=lambda: _EvaluationPipeline(vectors, raw_results),
                inference_session_factory=lambda _path: _DotAdapterSession(),
            )

            self.assertEqual(report["student_split_leaks"], 1)
            self.assertEqual(report["student_split_leak_ids"], ["train-only"])
            self.assertFalse(report["release_gate_passed"])

    def test_refuses_an_onnx_file_that_does_not_match_the_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            (adapter_dir / "identity_domain_adapter.onnx").write_bytes(b"corrupt")

            with self.assertRaisesRegex(ValueError, "SHA256"):
                evaluate_release_candidate(
                    dataset_path,
                    adapter_dir,
                    benchmark,
                    pipeline_factory=lambda: _EvaluationPipeline(vectors, raw_results),
                    inference_session_factory=lambda _path: _DotAdapterSession(),
                )

    def test_known_impostor_must_remain_mismatch_in_raw_and_adapted_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            known_record = next(
                (benchmark / KNOWN_IMPOSTOR_STUDENT_ID).glob("*/record.json")
            )
            request = json.loads(known_record.read_text(encoding="utf-8"))["request"]
            ref_path = request["ref_image_path"]
            photo_path = request["photos"][0]["image_path"]
            raw_results[ref_path] = ("match", "consistent")
            vectors[photo_path] = -vectors[ref_path]

            report = evaluate_release_candidate(
                dataset_path,
                adapter_dir,
                benchmark,
                pipeline_factory=lambda: _EvaluationPipeline(vectors, raw_results),
                inference_session_factory=lambda _path: _DotAdapterSession(),
            )

            self.assertEqual(report["known_impostor_raw_status"], "match")
            self.assertEqual(report["known_impostor_adapted_status"], "mismatch")
            self.assertFalse(report["known_impostor_detected"])
            self.assertFalse(report["release_gate_passed"])

    def test_runs_raw_stage_one_for_all_held_out_sessions_before_adapter_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path, adapter_dir, benchmark, vectors, raw_results = (
                self._release_fixture(root)
            )
            pipeline = _EvaluationPipeline(vectors, raw_results)

            with patch("tools.evaluate_domain_adapter.cv2.imread", return_value=None):
                with self.assertRaisesRegex(ValueError, "reference is unreadable"):
                    evaluate_release_candidate(
                        dataset_path,
                        adapter_dir,
                        benchmark,
                        pipeline_factory=lambda: pipeline,
                        inference_session_factory=lambda _path: _DotAdapterSession(),
                    )

            self.assertEqual(pipeline.session_checks, 5)

    def test_cli_writes_private_report_before_returning_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            failing = {
                "cross_student_far": 0.02,
                "conditional_accuracy": 0.8,
                "known_impostor_detected": True,
                "new_false_accusations": 0,
                "student_split_leaks": 0,
                "release_gate_passed": False,
            }
            with patch(
                "tools.evaluate_domain_adapter.evaluate_release_candidate",
                return_value=failing,
            ):
                exit_code = evaluation_main(
                    [
                        "--dataset-manifest",
                        "dataset.json",
                        "--adapter-dir",
                        "candidate",
                        "--benchmark-archive",
                        "benchmark",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), failing)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_cli_serializes_evaluation_errors_and_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch(
                "tools.evaluate_domain_adapter.evaluate_release_candidate",
                side_effect=ValueError("bad candidate"),
            ):
                exit_code = evaluation_main(
                    [
                        "--dataset-manifest",
                        "dataset.json",
                        "--adapter-dir",
                        "candidate",
                        "--benchmark-archive",
                        "benchmark",
                        "--output",
                        str(output),
                    ]
                )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 2)
            self.assertEqual(report["evaluation_error"], "bad candidate")
            self.assertFalse(report["release_gate_passed"])

    def _release_fixture(
        self, root: Path
    ) -> tuple[Path, Path, Path, dict[str, np.ndarray], dict[str, tuple[str, str]]]:
        vectors: dict[str, np.ndarray] = {}
        raw_results = {
            "test-a": ("inconclusive", "consistent"),
            "test-b": ("match", "single"),
            "test-c": ("mismatch", "consistent"),
            "test-d": ("mismatch", "inconsistent"),
            "test-e": ("inconclusive", "unknown"),
        }
        evaluation_rows = []
        basis = np.eye(128, dtype=np.float32)
        for index, student in enumerate(raw_results):
            ref = root / f"{student}-ref.png"
            photo = root / f"{student}-photo.png"
            self._write_vector_image(ref, index + 1)
            self._write_vector_image(photo, index + 31)
            ref_vector = basis[index]
            if student == "test-a":
                photo_vector = np.zeros(128, dtype=np.float32)
                photo_vector[index] = 0.8
                photo_vector[20] = 0.6
            elif student == "test-c":
                photo_vector = -basis[index]
            else:
                photo_vector = basis[index]
            vectors[str(ref)] = ref_vector
            vectors[str(photo)] = photo_vector
            reasons = ["test_split"]
            if student == "test-b":
                reasons.append("implicit_correct")
            evaluation_rows.append(
                {
                    "student_id": student,
                    "session_id": f"session-{student}",
                    "split": "test",
                    "label": "mismatch" if student == "test-c" else "match",
                    "ref_image_path": str(ref),
                    "photos": [
                        {
                            "sequence_no": 1,
                            "photo_type": "sign_in",
                            "image_path": str(photo),
                        }
                    ],
                    "source_feedback_ids": [f"feedback-{student}"],
                    "training_exclusion_reasons": reasons,
                }
            )
        dataset = {
            "schema_version": 1,
            "snapshot": "2026-08-30T12:00:00+00:00",
            "split_seed": "seed-v1",
            "sessions": [
                {
                    **evaluation_rows[0],
                    "student_id": "train-only",
                    "session_id": "train-session",
                    "split": "train",
                    "training_exclusion_reasons": [],
                }
            ],
            "evaluation_sessions": evaluation_rows,
        }
        dataset_path = root / "dataset.json"
        adapter_dir = root / "candidate"
        adapter_dir.mkdir()
        self._write_dataset_and_bind_artifact(dataset_path, adapter_dir, dataset)

        benchmark = root / "benchmark"
        for index in range(40):
            student = (
                KNOWN_IMPOSTOR_STUDENT_ID if index == 0 else f"benchmark-{index:02d}"
            )
            session_dir = benchmark / student / f"session-{index:02d}"
            session_dir.mkdir(parents=True)
            ref = session_dir / "ref.png"
            photo = session_dir / "photo.png"
            self._write_vector_image(ref, index + 61)
            self._write_vector_image(photo, index + 101)
            vectors[str(ref)] = basis[index + 40]
            vectors[str(photo)] = basis[index + 40]
            raw_results[str(ref)] = (
                ("mismatch", "inconsistent")
                if student == KNOWN_IMPOSTOR_STUDENT_ID
                else ("match", "consistent")
            )
            record = {
                "student_id": student,
                "session_id": f"session-{index:02d}",
                "request": {
                    "ref_image_path": str(ref),
                    "photos": [
                        {
                            "sequence_no": 1,
                            "photo_type": "sign_in",
                            "image_path": str(photo),
                        }
                    ],
                },
            }
            (session_dir / "record.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
        benchmark_manifest_path = benchmark / "benchmark-manifest.json"
        entries = []
        for record_path in sorted(benchmark.glob("*/*/record.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            entries.append(
                {
                    "student_id": record["student_id"],
                    "session_id": record["session_id"],
                    "record_path": record_path.relative_to(benchmark).as_posix(),
                    "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                    "truth": (
                        "mismatch"
                        if record["student_id"] == KNOWN_IMPOSTOR_STUDENT_ID
                        else "match"
                    ),
                }
            )
        self._write_benchmark_manifest(
            benchmark_manifest_path,
            {
                "schema_version": 1,
                "benchmark_id": "synthetic-40-session-v1",
                "known_impostor": {
                    "student_id": KNOWN_IMPOSTOR_STUDENT_ID,
                    "session_id": "session-00",
                },
                "sessions": entries,
            },
        )
        return dataset_path, adapter_dir, benchmark, vectors, raw_results

    @staticmethod
    def _write_vector_image(path: Path, value: int) -> None:
        image = np.full((2, 2, 3), value, dtype=np.uint8)
        if not cv2.imwrite(str(path), image):
            raise AssertionError(f"failed to write {path}")

    @staticmethod
    def _write_dataset_and_bind_artifact(
        dataset_path: Path, adapter_dir: Path, dataset: dict
    ) -> None:
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
        onnx_path = adapter_dir / "identity_domain_adapter.onnx"
        onnx_path.write_bytes(b"synthetic onnx")
        dataset_sha = hashlib.sha256(
            json.dumps(
                dataset,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        artifact = {
            "schema_version": 1,
            "model_version": "synthetic-v1",
            "embedding_dimension": 128,
            "match_threshold": 0.35,
            "onnx_file": onnx_path.name,
            "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "onnx_parity_max_abs_error": 1e-6,
            "source_dataset_sha256": dataset_sha,
            "input_names": ["ref_embedding", "photo_embedding"],
            "output_name": "adapted_cosine",
        }
        (adapter_dir / "identity_domain_adapter.manifest.json").write_text(
            json.dumps(artifact), encoding="utf-8"
        )

    @staticmethod
    def _write_benchmark_manifest(path: Path, manifest: dict) -> None:
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        manifest = {
            **payload,
            "manifest_sha256": hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


class EmbeddingExtractionTests(unittest.TestCase):
    def test_rerun_uses_private_stat_keyed_cache_and_excludes_raw_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / name
                for name in ("ref.jpg", "one.jpg", "two.jpg", "three.jpg", "bad.jpg")
            ]
            for index, path in enumerate(paths):
                image = np.full((8, 8, 3), index + 1, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(path), image))
            manifest = {
                "schema_version": 1,
                "sessions": [
                    {
                        "student_id": "student",
                        "session_id": "session",
                        "split": "train",
                        "label": "match",
                        "ref_image_path": str(paths[0]),
                        "photos": [
                            {"sequence_no": 1, "image_path": str(paths[1])},
                            {"sequence_no": 2, "image_path": str(paths[2])},
                            {"sequence_no": 3, "image_path": str(paths[3])},
                            {"sequence_no": 4, "image_path": str(paths[4])},
                        ],
                    }
                ],
                "evaluation_sessions": [],
            }
            vectors = {
                str(paths[0]): np.asarray([1.0, 0.0], dtype=np.float32),
                str(paths[1]): np.asarray([1.0, 0.0], dtype=np.float32),
                str(paths[2]): np.asarray([1.0, 0.0], dtype=np.float32),
                str(paths[3]): np.asarray([1.0, 0.0], dtype=np.float32),
                str(paths[4]): np.asarray([-1.0, 0.0], dtype=np.float32),
            }
            first_pipeline = _FakePipeline(vectors)
            cache = root / "embeddings.npz"

            first = extract_session_embeddings(
                manifest,
                cache_path=cache,
                pipeline_factory=lambda: first_pipeline,
            )

            self.assertEqual(first_pipeline.extractions, 5)
            self.assertEqual(len(first), 1)
            np.testing.assert_allclose(first[0].session_prototype, [1.0, 0.0])
            self.assertEqual(cache.stat().st_mode & 0o777, 0o600)

            second_pipeline = _FakePipeline(vectors)
            second = extract_session_embeddings(
                manifest,
                cache_path=cache,
                pipeline_factory=lambda: second_pipeline,
            )

            self.assertEqual(second_pipeline.extractions, 0)
            np.testing.assert_array_equal(
                second[0].session_prototype, first[0].session_prototype
            )


class DeterministicTrainingTests(unittest.TestCase):
    def test_identity_regularization_penalizes_residual_weights(self) -> None:
        model = LowRankDomainAdapter(dimension=4, rank=2)
        scores = torch.tensor([0.9, 0.0], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])

        loss = contrastive_margin_loss(
            scores,
            labels,
            positive_margin=0.35,
            negative_margin=0.15,
            residual_parameters=model.parameters(),
            identity_regularization_weight=1.0,
        )

        self.assertGreater(float(loss.detach()), 0.0)

    def test_same_seed_produces_identical_model_and_metrics(self) -> None:
        pairs = self._pairs()

        first = train_adapter(
            pairs,
            pairs,
            dimension=4,
            rank=2,
            epochs=4,
            seed=20260830,
            learning_rate=0.05,
            positive_margin=0.35,
            negative_margin=0.15,
            max_false_accept_rate=0.5,
        )
        second = train_adapter(
            pairs,
            pairs,
            dimension=4,
            rank=2,
            epochs=4,
            seed=20260830,
            learning_rate=0.05,
            positive_margin=0.35,
            negative_margin=0.15,
            max_false_accept_rate=0.5,
        )

        for name, value in first.model.state_dict().items():
            torch.testing.assert_close(
                value, second.model.state_dict()[name], rtol=0.0, atol=0.0
            )
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.validation_metrics, second.validation_metrics)

    def test_training_continues_until_validation_far_becomes_feasible(self) -> None:
        pairs = self._initially_infeasible_pairs()

        result = train_adapter(
            pairs,
            pairs,
            dimension=4,
            rank=2,
            epochs=15,
            seed=20260830,
            learning_rate=0.05,
            positive_margin=0.35,
            negative_margin=0.15,
            max_false_accept_rate=0.0,
        )

        self.assertGreater(result.best_epoch, 1)
        self.assertEqual(result.validation_metrics.false_accept_rate, 0.0)

    @staticmethod
    def _pairs() -> PairSet:
        refs = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        photos = np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.0, 0.43589, 0.0],
                [0.0, 0.9, 0.0, 0.43589],
            ],
            dtype=np.float32,
        )
        return PairSet(
            ref_embeddings=refs,
            photo_embeddings=photos,
            labels=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            metadata=(),
        )

    @staticmethod
    def _initially_infeasible_pairs() -> PairSet:
        refs = np.eye(4, dtype=np.float32)
        photos = np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 0.6, 0.0],
                [0.0, 0.8, 0.0, 0.6],
            ],
            dtype=np.float32,
        )
        return PairSet(
            ref_embeddings=refs,
            photo_embeddings=photos,
            labels=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            metadata=(),
        )


class OnnxExportTests(unittest.TestCase):
    def test_export_round_trip_matches_pytorch_with_dynamic_batch(self) -> None:
        torch.manual_seed(11)
        model = LowRankDomainAdapter(dimension=128, rank=4)
        with torch.no_grad():
            model.ref_tower.up.weight.normal_(std=0.01)
            model.photo_tower.up.weight.normal_(std=0.01)
        refs = torch.nn.functional.normalize(torch.randn(3, 128), dim=1)
        photos = torch.nn.functional.normalize(torch.randn(3, 128), dim=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "adapter.onnx"

            max_error = export_onnx(
                model,
                output,
                parity_inputs=(refs.numpy(), photos.numpy()),
            )

            session = ort.InferenceSession(
                str(output), providers=["CPUExecutionProvider"]
            )
            actual = session.run(
                ["adapted_cosine"],
                {"ref_embedding": refs.numpy(), "photo_embedding": photos.numpy()},
            )[0]
            expected = model(refs, photos).detach().numpy()
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
            self.assertLessEqual(max_error, 1e-5)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(session.get_inputs()[0].shape, ["N", 128])
            self.assertEqual(session.get_outputs()[0].shape, ["N"])


class ArtifactAndCliTests(unittest.TestCase):
    def test_candidate_manifest_contains_checksum_parity_and_private_report(
        self,
    ) -> None:
        pairs = self._pairs_128()
        result = train_adapter(
            pairs,
            pairs,
            pairs,
            dimension=128,
            rank=2,
            epochs=1,
            seed=17,
            max_false_accept_rate=0.5,
        )
        dataset_manifest = {
            "schema_version": 1,
            "snapshot": "2026-08-30T12:00:00+00:00",
            "split_seed": "seed-v1",
            "sessions": [
                PairConstructionTests._row("a", "train", "train"),
                PairConstructionTests._row("b", "validation", "validation"),
            ],
            "evaluation_sessions": [
                PairConstructionTests._row("c", "test", "test", ["test_split"])
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            manifest = write_candidate_artifacts(
                result,
                {"train": pairs, "validation": pairs, "test": pairs},
                dataset_manifest,
                output,
                hyperparameters={"epochs": 1, "seed": 17},
                source_revision="abc123",
            )

            onnx_path = output / "identity_domain_adapter.onnx"
            manifest_path = output / "identity_domain_adapter.manifest.json"
            report_path = output / "training-evaluation.json"
            expected_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
            self.assertEqual(manifest["onnx_sha256"], expected_sha)
            self.assertEqual(manifest["embedding_dimension"], 128)
            self.assertLessEqual(manifest["onnx_parity_max_abs_error"], 1e-5)
            self.assertEqual(manifest["split_counts"]["train"]["sessions"], 1)
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["best_epoch"], 1)
            self.assertEqual(len(report["threshold_sweeps"]["validation"]), 351)
            for path in (onnx_path, manifest_path, report_path):
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_cli_refuses_non_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            dataset.write_text("{}", encoding="utf-8")
            output = root / "candidate"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("user data", encoding="utf-8")

            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    training_main(
                        [
                            "--dataset-manifest",
                            str(dataset),
                            "--output-dir",
                            str(output),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user data")

    @staticmethod
    def _pairs_128() -> PairSet:
        refs = np.zeros((4, 128), dtype=np.float32)
        photos = np.zeros((4, 128), dtype=np.float32)
        refs[np.arange(4), np.arange(4)] = 1.0
        photos[0, 1] = 1.0
        photos[1, 0] = 1.0
        photos[2, 0] = 0.9
        photos[2, 2] = 0.43589
        photos[3, 1] = 0.9
        photos[3, 3] = 0.43589
        return PairSet(
            ref_embeddings=refs,
            photo_embeddings=photos,
            labels=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            metadata=(),
        )


class _EvaluationPipeline:
    def __init__(
        self,
        vectors: dict[str, np.ndarray],
        raw_results: dict[str, tuple[str, str]],
    ):
        self.vectors = vectors
        self.raw_results = raw_results
        self.session_checks = 0

    def session_check(self, ref_image_path: str, _photos: list[dict]) -> SimpleNamespace:
        self.session_checks += 1
        key = (
            ref_image_path
            if ref_image_path in self.raw_results
            else Path(ref_image_path).stem.removesuffix("-ref")
        )
        status, consistency = self.raw_results[key]
        return SimpleNamespace(
            session_status=status,
            internal_consistency=consistency,
        )

    def _extract_or_raise(self, image: np.ndarray) -> np.ndarray:
        marker = int(image[0, 0, 0])
        for path, vector in self.vectors.items():
            loaded = cv2.imread(path)
            if loaded is not None and int(loaded[0, 0, 0]) == marker:
                return vector
        raise RuntimeError(f"unknown image marker: {marker}")

    def _process_photo_for_session(self, photo: dict) -> SimpleNamespace:
        return SimpleNamespace(
            passes_gate=True,
            _embedding=self.vectors[photo["image_path"]],
        )


class _DotAdapterSession:
    def __init__(self) -> None:
        self.scored_pairs = 0

    def run(self, outputs: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if outputs != ["adapted_cosine"]:
            raise AssertionError(outputs)
        refs = inputs["ref_embedding"]
        photos = inputs["photo_embedding"]
        self.scored_pairs += len(refs)
        scores = (refs * photos).sum(axis=1)
        same_axis = np.argmax(np.abs(refs), axis=1) == np.argmax(
            np.abs(photos), axis=1
        )
        return [(scores + same_axis.astype(np.float32) * 0.4).astype(np.float32)]


class _FakePipeline:
    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors
        self.extractions = 0

    def _extract_or_raise(self, image: np.ndarray) -> np.ndarray:
        self.extractions += 1
        return next(iter(self.vectors.values()))

    def _process_photo_for_session(self, photo: dict) -> SimpleNamespace:
        self.extractions += 1
        return SimpleNamespace(
            passes_gate=True,
            _embedding=self.vectors[photo["image_path"]],
        )


if __name__ == "__main__":
    unittest.main()
