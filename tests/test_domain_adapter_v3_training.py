from __future__ import annotations

import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from module.face.domain_adapter import DomainAdapterRuntime
from tools.domain_adapter_training import LowRankDomainAdapter, PairMetadata, SessionEmbedding
from tools.domain_adapter_v2_metrics import V2PairSet
from tools.domain_adapter_v2_metrics import CalibratedThreshold
from tools.domain_adapter_v2_training import FoldEpochMetrics, V2Selection, select_v2_epoch
from tools.domain_adapter_v3_training import (
    HistoricalGateError,
    OofScores,
    V3TrainingResult,
    V3TrainingConfig,
    _v3_runtime_manifest,
    _group_tail_negative_loss,
    _student_balanced_positive_loss,
    _student_balanced_ranking_loss,
    continue_after_historical_oof,
    historical_metrics_from_oof,
    require_historical_gate,
    select_v3_epoch,
    train_v3_epoch,
    train_v3_candidate,
    v3_separation_loss,
    write_v3_candidate_artifacts,
)
from tools.domain_adapter_v3_metrics import HistoricalRelativeMetrics
from tools.train_domain_adapter_v3 import _parser, main


def _meta(
    ref_student: str,
    ref_session: str,
    photo_student: str,
    photo_session: str,
) -> PairMetadata:
    return PairMetadata(
        ref_student_id=ref_student,
        ref_session_id=ref_session,
        photo_student_id=photo_student,
        photo_session_id=photo_session,
        label=0,
        raw_cosine=0.0,
    )


class V3TrainingConfigTests(unittest.TestCase):
    def test_defaults_are_the_approved_frozen_configuration(self) -> None:
        config = V3TrainingConfig()
        self.assertEqual(config.hard_negative_groups_per_positive, 20)
        self.assertEqual(config.rank, 16)
        self.assertEqual(config.max_epochs, 100)
        self.assertEqual(config.learning_rate, 0.01)

    def test_configuration_rejects_unapproved_hard_negative_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 20"):
            V3TrainingConfig(hard_negative_groups_per_positive=10)


class V3LossTests(unittest.TestCase):
    def test_positive_hinge_balances_students_not_sessions(self) -> None:
        scores = torch.tensor([0.0, 0.2, 0.4])
        loss = _student_balanced_positive_loss(
            scores,
            ("A", "A", "B"),
            margin=0.35,
        )
        self.assertAlmostEqual(float(loss), 0.125, places=6)

    def test_negative_hinge_uses_only_each_groups_current_maximum(self) -> None:
        scores = torch.tensor([0.20, 0.40, 0.35], requires_grad=True)
        metadata = (
            _meta("A", "A1", "B", "B1"),
            _meta("A", "A1", "B", "B2"),
            _meta("A", "A1", "C", "C1"),
        )
        loss, selected = _group_tail_negative_loss(
            scores,
            ("A>B", "A>B", "A>C"),
            metadata,
            margin=0.30,
        )
        self.assertAlmostEqual(float(loss.detach()), 0.075, places=6)
        self.assertEqual(selected.tolist(), [1, 2])
        loss.backward()
        np.testing.assert_allclose(scores.grad.numpy(), [0.0, 0.5, 0.5])

    def test_ranking_balances_reference_students_and_distinct_photo_students(self) -> None:
        positive_scores = torch.tensor([0.4, 0.2, 0.5], requires_grad=True)
        negative_scores = torch.tensor(
            [0.6, 0.55, 0.5, 0.4, 0.3],
            requires_grad=True,
        )
        metadata = (
            _meta("A", "A1", "B", "B1"),
            _meta("A", "A1", "B", "B2"),
            _meta("A", "A1", "C", "C1"),
            _meta("A", "A2", "B", "B1"),
            _meta("B", "B1", "C", "C1"),
        )
        loss, selected = _student_balanced_ranking_loss(
            positive_scores,
            negative_scores,
            positive_student_ids=("A", "A", "B"),
            positive_session_ids=("A1", "A2", "B1"),
            negative_metadata=metadata,
            margin=0.10,
            limit=20,
        )
        self.assertAlmostEqual(float(loss.detach()), 0.1375, places=6)
        self.assertEqual(selected.tolist(), [0, 2, 3, 4])
        loss.backward()
        self.assertNotEqual(float(positive_scores.grad.abs().sum()), 0.0)
        self.assertEqual(float(negative_scores.grad[1]), 0.0)
        self.assertNotEqual(float(negative_scores.grad[[0, 2, 3]].abs().sum()), 0.0)

    def test_complete_loss_backpropagates_through_selected_tail_pairs(self) -> None:
        model = LowRankDomainAdapter(dimension=2, rank=16)
        pair_set = V2PairSet(
            positive_ref_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            positive_photo_embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
            positive_student_ids=("A",),
            positive_session_ids=("A1",),
            negative_ref_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            negative_photo_embeddings=np.asarray([[0.8, 0.6]], dtype=np.float32),
            negative_group_ids=("A>B",),
            negative_categories=("synthetic_cross_student",),
            negative_weights=np.asarray([1.0], dtype=np.float32),
            negative_metadata=(_meta("A", "A1", "B", "B1"),),
        )
        loss = v3_separation_loss(model, pair_set, V3TrainingConfig())
        loss.total.backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)
        self.assertEqual(loss.tail_negative_indices.tolist(), [0])
        self.assertEqual(loss.hard_negative_indices.tolist(), [0])

    def test_one_epoch_is_deterministic_from_the_same_state(self) -> None:
        pair_set = V2PairSet(
            positive_ref_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            positive_photo_embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
            positive_student_ids=("A",),
            positive_session_ids=("A1",),
            negative_ref_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            negative_photo_embeddings=np.asarray([[0.8, 0.6]], dtype=np.float32),
            negative_group_ids=("A>B",),
            negative_categories=("synthetic_cross_student",),
            negative_weights=np.asarray([1.0], dtype=np.float32),
            negative_metadata=(_meta("A", "A1", "B", "B1"),),
        )
        first = LowRankDomainAdapter(dimension=2, rank=16)
        second = LowRankDomainAdapter(dimension=2, rank=16)
        second.load_state_dict(first.state_dict())
        config = V3TrainingConfig()
        train_v3_epoch(first, torch.optim.Adam(first.parameters(), lr=0.01), pair_set, config)
        train_v3_epoch(
            second,
            torch.optim.Adam(second.parameters(), lr=0.01),
            pair_set,
            config,
        )
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        ):
            torch.testing.assert_close(first_parameter, second_parameter)


class V3HistoricalGateTests(unittest.TestCase):
    def _metrics(self, recall_lift: float) -> HistoricalRelativeMetrics:
        return HistoricalRelativeMetrics(
            raw_far=0.01,
            adapted_far=0.009,
            recall_lift=recall_lift,
            true_match_delta=1,
            same_threshold_recall_delta=0.01,
            adapted_threshold=0.35,
        )

    def test_gate_failure_contains_only_aggregate_metrics(self) -> None:
        metrics = self._metrics(0.019)
        with self.assertRaises(HistoricalGateError) as caught:
            require_historical_gate(metrics)
        self.assertEqual(caught.exception.metrics, metrics)
        self.assertNotIn("student", str(caught.exception).lower())

    def test_exact_two_point_lift_passes(self) -> None:
        require_historical_gate(self._metrics(0.02))

    def test_oof_metrics_compare_raw_and_adapted_at_the_same_far_budget(self) -> None:
        metrics = historical_metrics_from_oof(
            OofScores(
                raw_positive=np.asarray([0.36, 0.34]),
                adapted_positive=np.asarray([0.38, 0.36]),
                positive_student_ids=("A", "B"),
                raw_negative=np.asarray([0.10, 0.20]),
                adapted_negative=np.asarray([0.10, 0.20]),
                negative_group_ids=("A>B", "B>A"),
            ),
            dataset_digest="d" * 64,
        )
        self.assertEqual(metrics.raw_far, 0.0)
        self.assertEqual(metrics.adapted_far, 0.0)
        self.assertEqual(metrics.recall_lift, 0.5)
        self.assertEqual(metrics.true_match_delta, 1)
        self.assertEqual(metrics.same_threshold_recall_delta, 0.5)
        self.assertEqual(metrics.adapted_threshold, 0.36)

    def test_failed_oof_gate_never_calls_later_training_or_evidence(self) -> None:
        scores = OofScores(
            raw_positive=np.asarray([0.36]),
            adapted_positive=np.asarray([0.36]),
            positive_student_ids=("A",),
            raw_negative=np.asarray([0.10]),
            adapted_negative=np.asarray([0.10]),
            negative_group_ids=("A>B",),
        )
        called = False

        def later(_metrics: HistoricalRelativeMetrics) -> None:
            nonlocal called
            called = True

        with self.assertRaises(HistoricalGateError):
            continue_after_historical_oof(
                scores,
                dataset_digest="d" * 64,
                continue_fn=later,
            )
        self.assertFalse(called)


class V3EpochSelectionTests(unittest.TestCase):
    @staticmethod
    def _history() -> tuple[FoldEpochMetrics, ...]:
        return tuple(
            FoldEpochMetrics(
                fold=fold,
                epoch=epoch,
                candidate_threshold=0.35,
                empirical_far=0.0,
                student_balanced_recall=0.4 + 0.1 * epoch - 0.01 * fold,
                validation_loss=0.3 - 0.05 * epoch,
                residual_drift=0.01 * epoch,
                negative_group_ids=tuple(
                    f"G{fold}-{index // 2}" for index in range(200)
                ),
                negative_accepts=tuple(
                    epoch >= 2 and index == 0 for index in range(200)
                ),
            )
            for epoch in (1, 2, 3)
            for fold in range(5)
        )

    def test_batched_bootstrap_is_exactly_equivalent_to_v2_selection(self) -> None:
        history = self._history()
        digest = "d" * 64
        expected = select_v2_epoch(history, digest)

        from tools.domain_adapter_v2_metrics import _iter_bootstrap_sample_counts

        with patch(
            "tools.domain_adapter_v3_training._iter_bootstrap_sample_counts",
            wraps=_iter_bootstrap_sample_counts,
        ) as draws:
            actual = select_v3_epoch(history, digest)

        self.assertEqual(actual, expected)
        self.assertEqual(draws.call_count, 1)


class V3ArtifactTests(unittest.TestCase):
    def _result(self, dimension: int = 2) -> V3TrainingResult:
        threshold = CalibratedThreshold(
            threshold=0.35,
            empirical_far=0.005,
            far_upper_95=0.009,
            student_balanced_recall=0.5,
            true_matches=2,
            feasible=True,
        )
        return V3TrainingResult(
            model=LowRankDomainAdapter(dimension=dimension, rank=16),
            selection=V2Selection(
                epoch=1,
                median_recall=0.5,
                worst_fold_recall=0.4,
                pooled_far_upper_95=0.009,
                residual_drift=0.01,
                validation_loss=0.1,
            ),
            fold_history=(),
            adapted_threshold=threshold,
            raw_threshold=threshold,
            historical_metrics=HistoricalRelativeMetrics(
                raw_far=0.01,
                adapted_far=0.009,
                recall_lift=0.02,
                true_match_delta=1,
                same_threshold_recall_delta=0.01,
                adapted_threshold=0.35,
            ),
            dataset_digest="d" * 64,
            source_feedback_snapshot="private snapshot",
            split_seed="split",
            split_counts={},
            validation_negative_category_counts={},
            validation_negative_category_weight_totals={},
        )

    def test_public_manifest_records_v3_aggregation_without_private_rows(self) -> None:
        result = self._result()
        manifest = _v3_runtime_manifest(
            result=result,
            seed=7,
            onnx_file="identity_domain_adapter.onnx",
            onnx_sha256="a" * 64,
            onnx_parity_max_abs_error=1e-7,
        )
        encoded = str(manifest)
        self.assertEqual(manifest["model_version"], "identity-domain-adapter-v3")
        self.assertEqual(
            manifest["training"]["negative_weighting"],
            "ordered_student_pair_current_maximum",
        )
        self.assertEqual(
            manifest["training"]["positive_weighting"],
            "equal_total_weight_per_student",
        )
        self.assertEqual(
            manifest["training"]["historical_oof_relative"],
            asdict(result.historical_metrics),
        )
        self.assertNotIn("private snapshot", encoded)
        self.assertNotIn("student_id", encoded)
        self.assertNotIn("session_id", encoded)

    def test_real_onnx_artifact_is_private_and_supports_dynamic_batches(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidate"
            manifest = write_v3_candidate_artifacts(output, self._result(), seed=7)
            self.assertEqual(manifest["model_version"], "identity-domain-adapter-v3")
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            self.assertTrue(
                all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
            )
            self.assertLessEqual(manifest["onnx_parity_max_abs_error"], 1e-5)

    def test_v3_artifact_loads_through_the_unchanged_runtime(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidate"
            write_v3_candidate_artifacts(output, self._result(dimension=128), seed=7)
            onnx_path = output / "identity_domain_adapter.onnx"

            runtime = DomainAdapterRuntime.load("shadow", onnx_path)
            ref = np.zeros(128, dtype=np.float32)
            photo = np.zeros(128, dtype=np.float32)
            ref[0] = 1.0
            photo[0] = 1.0
            decision = runtime.compare(ref, photo)

            self.assertTrue(runtime.ready)
            self.assertEqual(runtime.version, "identity-domain-adapter-v3")
            self.assertTrue(decision.usable)

    def test_artifact_writer_never_overwrites_a_racing_destination(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidate"
            sentinel = output / "sentinel.txt"

            def create_destination() -> None:
                output.mkdir(mode=0o700)
                sentinel.write_text("keep-me", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_v3_candidate_artifacts(
                    output,
                    self._result(),
                    seed=7,
                    _before_publish=create_destination,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep-me")
            self.assertFalse(
                (output / "identity_domain_adapter.manifest.json").exists()
            )

    def test_cli_rejects_training_knob_overrides(self) -> None:
        parser = _parser()
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--manifest", "dataset.json",
                        "--embedding-cache", "cache.npz",
                        "--output-dir", "candidate",
                        "--ranking-weight", "0.9",
                    ]
                )

    def test_cli_historical_failure_writes_only_non_overwriting_aggregate_report(self) -> None:
        metrics = HistoricalRelativeMetrics(
            raw_far=0.01,
            adapted_far=0.009,
            recall_lift=0.019,
            true_match_delta=1,
            same_threshold_recall_delta=0.01,
            adapted_threshold=0.35,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "dataset.json"
            manifest_path.write_text('{"schema_version": 1}', encoding="utf-8")
            output = root / "candidate"
            failure = root / "candidate.historical-gate-failure.json"
            with patch(
                "tools.train_domain_adapter_v3.train_v3_candidate",
                side_effect=HistoricalGateError(metrics),
            ):
                exit_code = main(
                    [
                        "--manifest", str(manifest_path),
                        "--embedding-cache", str(root / "cache.npz"),
                        "--output-dir", str(output),
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertFalse(output.exists())
            self.assertEqual(failure.stat().st_mode & 0o777, 0o600)
            payload = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(payload["historical_oof_relative"], asdict(metrics))
            self.assertNotIn("student", str(payload).lower())
            self.assertNotIn("session", str(payload).lower())

            sentinel = b"keep-me\n"
            failure.write_bytes(sentinel)
            os.chmod(failure, 0o600)
            with patch(
                "tools.train_domain_adapter_v3.train_v3_candidate",
                side_effect=HistoricalGateError(metrics),
            ):
                with self.assertRaises(FileExistsError):
                    main(
                        [
                            "--manifest", str(manifest_path),
                            "--embedding-cache", str(root / "cache.npz"),
                            "--output-dir", str(output),
                        ]
                    )
            self.assertEqual(failure.read_bytes(), sentinel)


class V3CandidateTrainingTests(unittest.TestCase):
    def test_candidate_runs_v3_folds_then_returns_frozen_final_model(self) -> None:
        sessions = tuple(
            SessionEmbedding(
                student_id=f"S{index}",
                session_id=f"X{index}",
                split="train",
                label="match",
                ref_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                session_prototype=np.asarray([0.8, 0.6], dtype=np.float32),
            )
            for index in range(10)
        ) + tuple(
            SessionEmbedding(
                student_id=f"V{index}",
                session_id=f"VX{index}",
                split="validation",
                label="match",
                ref_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                session_prototype=np.asarray([0.8, 0.6], dtype=np.float32),
            )
            for index in range(2)
        )
        folds = {
            fold: (sessions[fold * 2], sessions[fold * 2 + 1])
            for fold in range(5)
        }
        selection = V2Selection(
            epoch=1,
            median_recall=0.5,
            worst_fold_recall=0.5,
            pooled_far_upper_95=0.009,
            residual_drift=0.01,
            validation_loss=0.1,
        )
        historical = HistoricalRelativeMetrics(
            raw_far=0.01,
            adapted_far=0.009,
            recall_lift=0.02,
            true_match_delta=1,
            same_threshold_recall_delta=0.01,
            adapted_threshold=0.35,
        )

        def pass_gate(_scores, *, dataset_digest, continue_fn):
            self.assertTrue(dataset_digest)
            return continue_fn(historical)

        with TemporaryDirectory() as tmp:
            with (
                patch(
                    "tools.domain_adapter_v3_training.extract_session_embeddings",
                    return_value=sessions,
                ),
                patch(
                    "tools.domain_adapter_v3_training.assign_training_folds",
                    return_value=folds,
                ),
                patch("tools.domain_adapter_v3_training._validate_fold_sufficiency"),
                patch("tools.domain_adapter_v3_training._validate_partition_size"),
                patch(
                    "tools.domain_adapter_v3_training._synthetic_group_count",
                    return_value=1000,
                ),
                patch(
                    "tools.domain_adapter_v3_training.select_v3_epoch",
                    return_value=selection,
                ),
                patch(
                    "tools.domain_adapter_v3_training.continue_after_historical_oof",
                    side_effect=pass_gate,
                ),
            ):
                result = train_v3_candidate(
                    {"schema_version": 1, "sessions": []},
                    Path(tmp) / "embedding-cache.npz",
                    V3TrainingConfig(),
                    seed=7,
                    device="cpu",
                )

        self.assertEqual(result.selection.epoch, 1)
        self.assertEqual(result.historical_metrics, historical)
        self.assertEqual(result.model.rank, 16)


if __name__ == "__main__":
    unittest.main()
