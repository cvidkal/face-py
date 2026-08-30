from __future__ import annotations

import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from tools.domain_adapter_training import (
    LowRankDomainAdapter,
    PairMetadata,
    SessionEmbedding,
)
from tools.domain_adapter_v2_metrics import (
    BootstrapFar,
    CalibratedThreshold,
    V2PairSet,
    build_v2_pair_set,
    student_fold,
)
from tools.domain_adapter_v2_training import (
    FoldEpochMetrics,
    InsufficientDataError,
    NoFeasibleEpochError,
    TrainingTrace,
    V2EpochLoss,
    V2Selection,
    V2TrainingConfig,
    write_v2_candidate_artifacts,
    score_pair_set,
    select_v2_epoch,
    train_fold_epoch,
    train_v2_candidate,
    v2_separation_loss,
)
from tools.train_domain_adapter_v2 import main as train_main


class V2TrainingConfigTests(unittest.TestCase):
    def test_v2_defaults_are_frozen(self) -> None:
        config = V2TrainingConfig()

        self.assertEqual(config.positive_margin, 0.35)
        self.assertEqual(config.negative_margin, 0.30)
        self.assertEqual(config.ranking_margin, 0.10)
        self.assertEqual(config.positive_weight, 0.25)
        self.assertEqual(config.negative_weight, 0.35)
        self.assertEqual(config.ranking_weight, 0.40)
        self.assertEqual(config.identity_regularization_weight, 0.001)
        self.assertEqual(config.hard_negatives_per_positive, 10)
        self.assertEqual(config.rank, 16)
        self.assertEqual(config.max_epochs, 100)
        self.assertEqual(config.learning_rate, 0.01)

        with self.assertRaisesRegex(Exception, "cannot assign to field"):
            config.rank = 32  # type: ignore[misc]

    def test_configuration_rejects_all_separation_terms_disabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one separation term"):
            V2TrainingConfig(
                positive_weight=0.0,
                negative_weight=0.0,
                ranking_weight=0.0,
            )


def _normalized(value: tuple[float, float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return vector


def _cosine_vector(cosine: float) -> np.ndarray:
    return _normalized((cosine, math.sqrt(1.0 - cosine * cosine)))


class V2ObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LowRankDomainAdapter(dimension=2, rank=16)
        self.config = V2TrainingConfig()
        self.pairs = self._ranking_pair_set()
        self.expected_top_ten = list(range(1, 11))

    def test_score_pair_set_keeps_positive_and_negative_order(self) -> None:
        scores = score_pair_set(self.model, self.pairs, "cpu")

        self.assertAlmostEqual(float(scores.positive[0]), 0.55, places=6)
        self.assertAlmostEqual(float(scores.negative[0]), 0.95, places=6)
        self.assertTrue(np.allclose(scores.negative[1:], 0.50, atol=1e-6))

    def test_score_pair_set_accepts_cpu_alias_with_index(self) -> None:
        scores = score_pair_set(self.model, self.pairs, "cpu:0")

        self.assertAlmostEqual(float(scores.positive[0]), 0.55, places=6)
        self.assertAlmostEqual(float(scores.negative[0]), 0.95, places=6)

    def test_score_pair_set_accepts_bare_cuda_alias_for_current_index(self) -> None:
        def fake_tensor(values, **_kwargs) -> torch.Tensor:
            return torch.from_numpy(np.asarray(values, dtype=np.float32))

        with patch("tools.domain_adapter_v2_training._model_device", return_value=torch.device("cuda:3")):
            with patch(
                "tools.domain_adapter_v2_training.torch.cuda.current_device",
                return_value=3,
            ):
                with patch(
                    "tools.domain_adapter_v2_training._embedding_tensor",
                    side_effect=fake_tensor,
                ) as embedding_tensor:
                    scores = score_pair_set(self.model, self.pairs, "cuda")

        self.assertAlmostEqual(float(scores.positive[0]), 0.55, places=6)
        self.assertAlmostEqual(float(scores.negative[0]), 0.95, places=6)
        self.assertEqual(embedding_tensor.call_count, 4)

    def test_score_pair_set_rejects_requested_device_mismatch_before_allocating_inputs(
        self,
    ) -> None:
        with patch("tools.domain_adapter_v2_training._embedding_tensor") as embedding_tensor:
            with self.assertRaisesRegex(
                ValueError,
                "requested device meta does not match model device cpu",
            ):
                score_pair_set(self.model, self.pairs, "meta")
        embedding_tensor.assert_not_called()

    def test_score_pair_set_rejects_true_device_mismatch_after_canonicalization(
        self,
    ) -> None:
        with patch("tools.domain_adapter_v2_training._model_device", return_value=torch.device("meta")):
            with patch("tools.domain_adapter_v2_training._embedding_tensor") as embedding_tensor:
                with self.assertRaisesRegex(
                    ValueError,
                    "requested device cpu:0 does not match model device meta",
                ):
                    score_pair_set(self.model, self.pairs, "cpu:0")
        embedding_tensor.assert_not_called()

    def test_ranking_uses_top_ten_current_scores_sharing_the_reference(self) -> None:
        loss = v2_separation_loss(self.model, self.pairs, self.config)

        self.assertGreater(loss.ranking.item(), 0.0)
        self.assertEqual(loss.hard_negative_indices.tolist(), self.expected_top_ten)

    def test_ranking_uses_current_adapted_scores_and_preserves_ranking_gradients(
        self,
    ) -> None:
        model = LowRankDomainAdapter(dimension=2, rank=16)
        self._configure_photo_tower_x_from_y(model, scale=1.0)
        ranking_only = V2TrainingConfig(
            positive_weight=0.0,
            negative_weight=0.0,
            ranking_weight=1.0,
            identity_regularization_weight=0.0,
        )
        pair_set = self._current_score_ranking_pair_set()

        loss = v2_separation_loss(model, pair_set, ranking_only)

        self.assertEqual(
            loss.hard_negative_indices.tolist(),
            [6, 7, 8, 9, 10, 11, 0, 1, 2, 3],
        )
        loss.total.backward()
        gradient_total = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_total, 0.0)

    def test_duplicate_sessions_inside_one_group_do_not_increase_group_weight(self) -> None:
        config = V2TrainingConfig(positive_weight=0.0, negative_weight=1.0, ranking_weight=0.0)

        before = v2_separation_loss(
            self.model,
            self._weighted_negative_pair_set(),
            config,
        ).negative.item()
        after = v2_separation_loss(
            self.model,
            self._duplicate_one_group_and_renormalize(),
            config,
        ).negative.item()

        self.assertAlmostEqual(before, after, places=6)

    def test_train_fold_epoch_rejects_non_finite_loss_components(self) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)

        with patch(
            "tools.domain_adapter_v2_training.residual_weight_regularization",
            return_value=torch.tensor(float("nan")),
        ):
            with self.assertRaisesRegex(FloatingPointError, "identity"):
                train_fold_epoch(self.model, optimizer, self.pairs, self.config)

    def test_identical_seeds_produce_identical_epoch_updates(self) -> None:
        torch.manual_seed(11)
        baseline = LowRankDomainAdapter(dimension=2, rank=16)
        pair_set = build_v2_pair_set(
            [
                self._session("student-a", "session-1", (1.0, 0.0)),
                self._session("student-b", "session-1", (0.8, 0.6)),
                self._session("student-c", "session-1", (-1.0, 0.0)),
            ]
        )

        model_a = LowRankDomainAdapter(dimension=2, rank=16)
        model_b = LowRankDomainAdapter(dimension=2, rank=16)
        model_a.load_state_dict(baseline.state_dict())
        model_b.load_state_dict(baseline.state_dict())
        optimizer_a = torch.optim.SGD(model_a.parameters(), lr=self.config.learning_rate)
        optimizer_b = torch.optim.SGD(model_b.parameters(), lr=self.config.learning_rate)

        history_a = [train_fold_epoch(model_a, optimizer_a, pair_set, self.config) for _ in range(2)]
        history_b = [train_fold_epoch(model_b, optimizer_b, pair_set, self.config) for _ in range(2)]

        self.assertEqual(history_a, history_b)
        for state_key, value in model_a.state_dict().items():
            torch.testing.assert_close(value, model_b.state_dict()[state_key])

    @staticmethod
    def _session(
        student_id: str,
        session_id: str,
        vector: tuple[float, float],
    ) -> SessionEmbedding:
        embedding = _normalized(vector)

        return SessionEmbedding(
            student_id=student_id,
            session_id=session_id,
            split="train",
            label="match",
            ref_embedding=embedding,
            session_prototype=embedding,
        )

    def _ranking_pair_set(self) -> V2PairSet:
        positive_ref = _normalized((1.0, 0.0)).reshape(1, -1)
        positive_photo = _cosine_vector(0.55).reshape(1, -1)
        negative_refs = [_normalized((1.0, 0.0))]
        negative_photos = [_cosine_vector(0.95)]
        metadata = [
            PairMetadata(
                ref_student_id="distractor-student",
                ref_session_id="other-session",
                photo_student_id="distractor-photo",
                photo_session_id="distractor-photo-session",
                label=0,
                raw_cosine=0.95,
            )
        ]
        for index in range(12):
            negative_refs.append(_normalized((1.0, 0.0)))
            negative_photos.append(_cosine_vector(0.50))
            metadata.append(
                PairMetadata(
                    ref_student_id="anchor-student",
                    ref_session_id="anchor-session",
                    photo_student_id=f"photo-student-{index:02d}",
                    photo_session_id=f"photo-session-{index:02d}",
                    label=0,
                    raw_cosine=0.50,
                )
            )
        return V2PairSet(
            positive_ref_embeddings=positive_ref.astype(np.float32, copy=False),
            positive_photo_embeddings=positive_photo.astype(np.float32, copy=False),
            positive_student_ids=("anchor-student",),
            positive_session_ids=("anchor-session",),
            negative_ref_embeddings=np.stack(negative_refs).astype(np.float32, copy=False),
            negative_photo_embeddings=np.stack(negative_photos).astype(np.float32, copy=False),
            negative_group_ids=tuple(f"group-{index:02d}" for index in range(len(metadata))),
            negative_categories=("synthetic_cross_student",) * len(metadata),
            negative_weights=np.full(len(metadata), 1.0 / len(metadata), dtype=np.float32),
            negative_metadata=tuple(metadata),
        )

    def _weighted_negative_pair_set(self) -> V2PairSet:
        metadata = (
            PairMetadata("anchor", "anchor-session", "student-a", "session-a", 0, 0.40),
            PairMetadata("anchor", "anchor-session", "student-b", "session-b", 0, 0.20),
        )
        return V2PairSet(
            positive_ref_embeddings=np.stack([_normalized((1.0, 0.0))]).astype(
                np.float32, copy=False
            ),
            positive_photo_embeddings=np.stack([_normalized((1.0, 0.0))]).astype(
                np.float32, copy=False
            ),
            positive_student_ids=("anchor",),
            positive_session_ids=("anchor-session",),
            negative_ref_embeddings=np.stack(
                [_normalized((1.0, 0.0)), _normalized((1.0, 0.0))]
            ).astype(np.float32, copy=False),
            negative_photo_embeddings=np.stack(
                [_cosine_vector(0.40), _cosine_vector(0.20)]
            ).astype(np.float32, copy=False),
            negative_group_ids=("group-a", "group-b"),
            negative_categories=("synthetic_cross_student", "synthetic_cross_student"),
            negative_weights=np.asarray([0.5, 0.5], dtype=np.float32),
            negative_metadata=metadata,
        )

    def _duplicate_one_group_and_renormalize(self) -> V2PairSet:
        metadata = (
            PairMetadata("anchor", "anchor-session", "student-a", "session-a", 0, 0.40),
            PairMetadata("anchor", "anchor-session", "student-a-dup", "session-a-dup", 0, 0.40),
            PairMetadata("anchor", "anchor-session", "student-b", "session-b", 0, 0.20),
        )
        return V2PairSet(
            positive_ref_embeddings=np.stack([_normalized((1.0, 0.0))]).astype(
                np.float32, copy=False
            ),
            positive_photo_embeddings=np.stack([_normalized((1.0, 0.0))]).astype(
                np.float32, copy=False
            ),
            positive_student_ids=("anchor",),
            positive_session_ids=("anchor-session",),
            negative_ref_embeddings=np.stack(
                [
                    _normalized((1.0, 0.0)),
                    _normalized((1.0, 0.0)),
                    _normalized((1.0, 0.0)),
                ]
            ).astype(np.float32, copy=False),
            negative_photo_embeddings=np.stack(
                [
                    _cosine_vector(0.40),
                    _cosine_vector(0.40),
                    _cosine_vector(0.20),
                ]
            ).astype(np.float32, copy=False),
            negative_group_ids=("group-a", "group-a", "group-b"),
            negative_categories=(
                "synthetic_cross_student",
                "synthetic_cross_student",
                "synthetic_cross_student",
            ),
            negative_weights=np.asarray([0.25, 0.25, 0.5], dtype=np.float32),
            negative_metadata=metadata,
        )

    def _current_score_ranking_pair_set(self) -> V2PairSet:
        positive_ref = _normalized((1.0, 0.0)).reshape(1, -1)
        positive_photo = _normalized((0.80, -0.60)).reshape(1, -1)
        negative_refs = []
        negative_photos = []
        metadata: list[PairMetadata] = []
        for index, cosine in enumerate((0.92, 0.90, 0.88, 0.86, 0.84, 0.82)):
            negative_refs.append(_normalized((1.0, 0.0)))
            negative_photos.append(_normalized((cosine, -math.sqrt(1.0 - cosine * cosine))))
            metadata.append(
                PairMetadata(
                    ref_student_id="anchor-student",
                    ref_session_id="anchor-session",
                    photo_student_id=f"negative-student-{index:02d}",
                    photo_session_id=f"negative-session-{index:02d}",
                    label=0,
                    raw_cosine=cosine,
                )
            )
        for index, cosine in enumerate((0.80, 0.78, 0.76, 0.74, 0.72, 0.70), start=6):
            negative_refs.append(_normalized((1.0, 0.0)))
            negative_photos.append(_normalized((cosine, math.sqrt(1.0 - cosine * cosine))))
            metadata.append(
                PairMetadata(
                    ref_student_id="anchor-student",
                    ref_session_id="anchor-session",
                    photo_student_id=f"negative-student-{index:02d}",
                    photo_session_id=f"negative-session-{index:02d}",
                    label=0,
                    raw_cosine=cosine,
                )
            )
        return V2PairSet(
            positive_ref_embeddings=positive_ref.astype(np.float32, copy=False),
            positive_photo_embeddings=positive_photo.astype(np.float32, copy=False),
            positive_student_ids=("anchor-student",),
            positive_session_ids=("anchor-session",),
            negative_ref_embeddings=np.stack(negative_refs).astype(np.float32, copy=False),
            negative_photo_embeddings=np.stack(negative_photos).astype(np.float32, copy=False),
            negative_group_ids=tuple(f"ranking-group-{index:02d}" for index in range(len(metadata))),
            negative_categories=("synthetic_cross_student",) * len(metadata),
            negative_weights=np.full(len(metadata), 1.0 / len(metadata), dtype=np.float32),
            negative_metadata=tuple(metadata),
        )

    @staticmethod
    def _configure_photo_tower_x_from_y(
        model: LowRankDomainAdapter,
        *,
        scale: float,
    ) -> None:
        with torch.no_grad():
            model.photo_tower.down.weight.zero_()
            model.photo_tower.up.weight.zero_()
            model.photo_tower.down.weight[0, 1] = 1.0
            model.photo_tower.up.weight[0, 0] = scale


class V2EpochSelectionTests(unittest.TestCase):
    def test_select_v2_epoch_prefers_feasible_epoch_by_recall_then_tie_breakers(self) -> None:
        history = tuple(
            self._epoch_history(2, recalls=(0.90, 0.90, 0.90, 0.90, 0.90), threshold=0.349)
            + self._epoch_history(3, recalls=(0.72, 0.72, 0.72, 0.70, 0.69))
            + self._epoch_history(4, recalls=(0.72, 0.72, 0.72, 0.70, 0.70))
            + self._epoch_history(
                6,
                recalls=(0.72, 0.72, 0.72, 0.70, 0.70),
                residual_drift=0.03,
            )
            + self._epoch_history(
                7,
                recalls=(0.72, 0.72, 0.72, 0.70, 0.70),
                validation_loss=0.08,
            )
            + self._epoch_history(8, recalls=(0.72, 0.72, 0.72, 0.70, 0.70))
        )

        selection = select_v2_epoch(history, dataset_digest="0" * 64)

        self.assertEqual(selection.epoch, 4)
        self.assertEqual(selection.median_recall, 0.72)
        self.assertEqual(selection.worst_fold_recall, 0.70)
        self.assertLessEqual(selection.pooled_far_upper_95, 0.01)

    def test_select_v2_epoch_raises_when_no_epoch_is_feasible(self) -> None:
        history = tuple(
            self._epoch_history(1, recalls=(0.80, 0.80, 0.80, 0.80, 0.80), empirical_far=0.006)
            + self._epoch_history(2, recalls=(0.90, 0.90, 0.90, 0.90, 0.90), threshold=0.349)
        )

        with self.assertRaises(NoFeasibleEpochError):
            select_v2_epoch(history, dataset_digest="1" * 64)

    def test_select_v2_epoch_bootstraps_only_epochs_passing_fold_empirical_prescreen(
        self,
    ) -> None:
        history = tuple(
            self._epoch_history(1, recalls=(0.80, 0.80, 0.80, 0.80, 0.80), empirical_far=0.006)
            + self._epoch_history(2, recalls=(0.85, 0.85, 0.85, 0.85, 0.85), threshold=0.349)
            + self._epoch_history(3, recalls=(0.82, 0.82, 0.82, 0.82, 0.82))
        )
        bootstrap_calls: list[str] = []

        def fake_bootstrap(
            negative_scores: np.ndarray,
            group_ids: tuple[str, ...],
            threshold: float,
            dataset_digest: str,
            iterations: int = 10000,
        ) -> BootstrapFar:
            del negative_scores, group_ids, threshold, iterations
            bootstrap_calls.append(dataset_digest)
            return BootstrapFar(
                estimate=0.0,
                upper_95=0.0,
                iterations=10000,
                seed=1,
                group_count=5,
            )

        with patch(
            "tools.domain_adapter_v2_training.bootstrap_far_upper_bound",
            side_effect=fake_bootstrap,
        ):
            selection = select_v2_epoch(history, dataset_digest="digest-train-only")

        self.assertEqual(selection.epoch, 3)
        self.assertEqual(bootstrap_calls, ["digest-train-only"])

    @staticmethod
    def _epoch_history(
        epoch: int,
        *,
        recalls: tuple[float, float, float, float, float],
        threshold: float = 0.35,
        empirical_far: float = 0.0,
        residual_drift: float = 0.02,
        validation_loss: float = 0.05,
    ) -> list[FoldEpochMetrics]:
        records: list[FoldEpochMetrics] = []
        for fold, recall in enumerate(recalls):
            records.append(
                FoldEpochMetrics(
                    fold=fold,
                    epoch=epoch,
                    candidate_threshold=threshold,
                    empirical_far=empirical_far,
                    student_balanced_recall=recall,
                    validation_loss=validation_loss,
                    residual_drift=residual_drift,
                    negative_group_ids=tuple(
                        f"epoch-{epoch}-fold-{fold}-group-{index:03d}" for index in range(200)
                    ),
                    negative_accepts=(False,) * 200,
                )
            )
        return records


class V2CandidateTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.cache_dir = Path(self.tempdir.name)
        self.config = V2TrainingConfig()
        self.training_students = self._students_for_all_folds(20)
        self.validation_students = tuple(f"validation-{index:03d}" for index in range(33))
        self.training_sessions = tuple(
            self._session(student_id, split="train")
            for student_id in self.training_students
        )
        self.validation_sessions = tuple(
            self._session(student_id, split="validation")
            for student_id in self.validation_students
        )
        self.all_sessions = self.training_sessions + self.validation_sessions
        self.manifest = self._manifest(
            train_students=self.training_students,
            validation_students=self.validation_students,
        )
        self.small_manifest = self._manifest(
            train_students=tuple(f"tiny-{index:03d}" for index in range(19)),
            validation_students=self.validation_students,
        )

    def test_cross_validation_rejects_small_folds_before_training(self) -> None:
        with patch(
            "tools.domain_adapter_v2_training.extract_session_embeddings",
            return_value=list(self._sessions_from_manifest(self.small_manifest)),
        ):
            with self.assertRaisesRegex(InsufficientDataError, "20 students"):
                train_v2_candidate(
                    self.small_manifest,
                    self.cache_dir,
                    self.config,
                    seed=20260830,
                    device="cpu",
                )

    def test_validation_students_never_enter_fold_training_or_selection(self) -> None:
        trace = TrainingTrace()
        loaded_manifests: list[dict[str, object]] = []

        def fake_extract(manifest: dict[str, object], *, cache_path: Path):
            loaded_manifests.append(manifest)
            self.assertEqual(cache_path, self.cache_dir / ".embedding-cache.npz")
            return list(self._sessions_from_manifest(manifest))

        with patch(
            "tools.domain_adapter_v2_training.extract_session_embeddings",
            side_effect=fake_extract,
        ):
            with patch(
                "tools.domain_adapter_v2_training.train_fold_epoch",
                side_effect=self._fake_train_fold_epoch,
            ):
                with patch(
                    "tools.domain_adapter_v2_training.score_pair_set",
                    side_effect=self._fake_score_pair_set,
                ):
                    with patch(
                        "tools.domain_adapter_v2_training._select_empirical_threshold",
                        side_effect=self._fake_empirical_threshold,
                    ):
                        with patch(
                        "tools.domain_adapter_v2_training.calibrate_threshold",
                        side_effect=self._fake_calibrate_threshold,
                        ):
                            with patch(
                                "tools.domain_adapter_v2_training.v2_separation_loss",
                                side_effect=self._fake_v2_loss,
                            ):
                                with patch(
                                    "tools.domain_adapter_v2_training.select_v2_epoch",
                                    return_value=V2Selection(
                                        epoch=3,
                                        median_recall=1.0,
                                        worst_fold_recall=1.0,
                                        pooled_far_upper_95=0.0,
                                        residual_drift=0.01,
                                        validation_loss=0.0,
                                    ),
                                ):
                                    result = train_v2_candidate(
                                        self.manifest,
                                        self.cache_dir,
                                        self.config,
                                        seed=20260830,
                                        device="cpu",
                                        trace=trace,
                                    )

        self.assertIsNotNone(result.selection)
        self.assertTrue(trace.trained_fold_students.isdisjoint(self.validation_students))
        self.assertTrue(trace.selection_students.isdisjoint(self.validation_students))
        self.assertEqual(trace.final_training_students, set(self.training_students))
        self.assertEqual(trace.final_epoch_count, result.selection.epoch)
        self.assertEqual(loaded_manifests[0]["evaluation_sessions"], [])
        loaded_students = {
            str(row["student_id"])
            for row in loaded_manifests[0]["sessions"]  # type: ignore[index]
        }
        self.assertFalse(any(student.startswith("evaluation-") for student in loaded_students))
        self.assertFalse(any(student.startswith("excluded-") for student in loaded_students))

    def test_validation_requires_at_least_one_thousand_ordered_groups(self) -> None:
        manifest = self._manifest(
            train_students=self.training_students,
            validation_students=tuple(f"small-validation-{index:03d}" for index in range(20)),
        )

        with patch(
            "tools.domain_adapter_v2_training.extract_session_embeddings",
            return_value=list(self._sessions_from_manifest(manifest)),
        ):
            with patch(
                "tools.domain_adapter_v2_training.train_fold_epoch",
                side_effect=self._fake_train_fold_epoch,
            ):
                with patch(
                    "tools.domain_adapter_v2_training.score_pair_set",
                    side_effect=self._fake_score_pair_set,
                ):
                    with patch(
                        "tools.domain_adapter_v2_training.v2_separation_loss",
                        side_effect=self._fake_v2_loss,
                    ):
                        with self.assertRaisesRegex(
                            InsufficientDataError,
                            "1000 ordered groups",
                        ):
                            train_v2_candidate(
                                manifest,
                                self.cache_dir,
                                self.config,
                                seed=20260830,
                                device="cpu",
                            )

    def test_validation_requires_one_thousand_synthetic_cross_student_groups(self) -> None:
        manifest = self._manifest(
            train_students=self.training_students,
            validation_students=tuple(f"synthetic-limit-{index:03d}" for index in range(20)),
        )
        mixed_pair_set = self._validation_pair_set_with_extra_human_groups(
            synthetic_students=20,
            extra_human_groups=700,
        )

        def fake_pair_builder(sessions: tuple[SessionEmbedding, ...]) -> V2PairSet:
            if sessions and all(session.split == "validation" for session in sessions):
                return mixed_pair_set
            return build_v2_pair_set(sessions)

        with patch(
            "tools.domain_adapter_v2_training.extract_session_embeddings",
            return_value=list(self._sessions_from_manifest(manifest)),
        ):
            with patch(
                "tools.domain_adapter_v2_training.build_v2_pair_set",
                side_effect=fake_pair_builder,
            ):
                with patch(
                    "tools.domain_adapter_v2_training.train_fold_epoch",
                    side_effect=AssertionError("validation sufficiency should fail before training"),
                ):
                    with self.assertRaisesRegex(
                        InsufficientDataError,
                        "1000 ordered groups",
                    ):
                        train_v2_candidate(
                            manifest,
                            self.cache_dir,
                            self.config,
                            seed=20260830,
                            device="cpu",
                        )

    def test_selection_uses_train_only_digest_and_final_calibration_uses_full_digest(
        self,
    ) -> None:
        first_manifest = self._manifest(
            train_students=self.training_students,
            validation_students=tuple(f"validation-a-{index:03d}" for index in range(33)),
        )
        second_manifest = self._manifest(
            train_students=self.training_students,
            validation_students=tuple(f"validation-b-{index:03d}" for index in range(33)),
        )

        first = self._run_candidate_with_instrumentation(first_manifest)
        second = self._run_candidate_with_instrumentation(second_manifest)

        self.assertEqual(first["result"].fold_history, second["result"].fold_history)
        self.assertEqual(first["result"].selection, second["result"].selection)
        self.assertEqual(len(set(first["selection_digests"])), 1)
        self.assertEqual(len(set(second["selection_digests"])), 1)
        self.assertEqual(first["selection_digests"], second["selection_digests"])
        self.assertEqual(len(first["final_calibration_digests"]), 2)
        self.assertEqual(len(second["final_calibration_digests"]), 2)
        self.assertNotEqual(
            first["final_calibration_digests"][0],
            second["final_calibration_digests"][0],
        )
        self.assertEqual(
            first["result"].dataset_digest,
            first["final_calibration_digests"][0],
        )
        self.assertEqual(
            second["result"].dataset_digest,
            second["final_calibration_digests"][0],
        )

    def test_fold_training_calls_bootstrap_once_per_epoch_and_calibrates_only_twice_at_end(
        self,
    ) -> None:
        observed = self._run_candidate_with_instrumentation(self.manifest)

        self.assertEqual(observed["bootstrap_calls"], self.config.max_epochs)
        self.assertEqual(observed["selection_bootstrap_calls"], self.config.max_epochs)
        self.assertEqual(observed["calibrate_calls"], 2)
        self.assertEqual(len(observed["final_calibration_digests"]), 2)

    @staticmethod
    def _session(student_id: str, *, split: str) -> SessionEmbedding:
        vector = _normalized(
            (
                1.0,
                (int(student_id.encode("utf-8").hex(), 16) % 7 + 1) / 10.0,
            )
        )
        return SessionEmbedding(
            student_id=student_id,
            session_id="session-1",
            split=split,
            label="match",
            ref_embedding=vector,
            session_prototype=vector,
        )

    @staticmethod
    def _students_for_all_folds(count_per_fold: int) -> tuple[str, ...]:
        buckets: dict[int, list[str]] = {fold: [] for fold in range(5)}
        index = 0
        while any(len(bucket) < count_per_fold for bucket in buckets.values()):
            student_id = f"train-{index:04d}"
            fold = student_fold(student_id)
            if len(buckets[fold]) < count_per_fold:
                buckets[fold].append(student_id)
            index += 1
        return tuple(student for fold in range(5) for student in buckets[fold])

    @staticmethod
    def _manifest(
        *,
        train_students: tuple[str, ...],
        validation_students: tuple[str, ...],
    ) -> dict[str, object]:
        def row(student_id: str, split: str, *, reasons: list[str] | None = None) -> dict[str, object]:
            item: dict[str, object] = {
                "student_id": student_id,
                "session_id": "session-1",
                "split": split,
                "label": "match",
                "ref_image_path": f"/{student_id}/ref.jpg",
                "photos": [{"sequence_no": 1, "image_path": f"/{student_id}/photo.jpg"}],
            }
            if reasons is not None:
                item["training_exclusion_reasons"] = reasons
            return item

        sessions = [row(student_id, "train") for student_id in train_students]
        sessions.extend(row(student_id, "validation") for student_id in validation_students)
        sessions.append(row("excluded-student", "train", reasons=["photo_error"]))
        return {
            "schema_version": 1,
            "snapshot": "2026-08-30T12:00:00+00:00",
            "split_seed": "seed-v2",
            "sessions": sessions,
            "evaluation_sessions": [row("evaluation-student", "test", reasons=["test_split"])],
        }

    @staticmethod
    def _sessions_from_manifest(manifest: dict[str, object]) -> tuple[SessionEmbedding, ...]:
        sessions: list[SessionEmbedding] = []
        for row in manifest["sessions"]:  # type: ignore[index]
            vector = _normalized(
                (
                    1.0,
                    (int(str(row["student_id"]).encode("utf-8").hex(), 16) % 7 + 1) / 10.0,
                )
            )
            sessions.append(
                SessionEmbedding(
                    student_id=str(row["student_id"]),
                    session_id=str(row["session_id"]),
                    split=str(row["split"]),
                    label=str(row["label"]),
                    ref_embedding=vector,
                    session_prototype=vector,
                )
            )
        return tuple(sessions)

    def _run_candidate_with_instrumentation(
        self,
        manifest: dict[str, object],
    ) -> dict[str, object]:
        bootstrap_digests: list[str] = []
        calibrate_digests: list[str] = []

        def fake_bootstrap(
            negative_scores: np.ndarray,
            group_ids: tuple[str, ...],
            threshold: float,
            dataset_digest: str,
            iterations: int = 10000,
        ) -> BootstrapFar:
            del negative_scores, group_ids, threshold, iterations
            bootstrap_digests.append(dataset_digest)
            return BootstrapFar(
                estimate=0.0,
                upper_95=0.0,
                iterations=10000,
                seed=1,
                group_count=5,
            )

        def fake_empirical_threshold(
            *,
            positive_scores: np.ndarray,
            positive_student_ids: tuple[str, ...],
            negative_scores: np.ndarray,
            negative_group_ids: tuple[str, ...],
            threshold_floor: float = 0.35,
            max_empirical_far: float = 0.005,
        ) -> CalibratedThreshold:
            del negative_scores, negative_group_ids, max_empirical_far
            return CalibratedThreshold(
                threshold=threshold_floor,
                empirical_far=0.0,
                far_upper_95=0.0,
                student_balanced_recall=(
                    1.0 if len(positive_scores) == len(positive_student_ids) else 0.0
                ),
                true_matches=len(positive_scores),
                feasible=True,
            )

        def fake_calibrate_threshold(
            *,
            positive_scores: np.ndarray,
            positive_student_ids: tuple[str, ...],
            negative_scores: np.ndarray,
            negative_group_ids: tuple[str, ...],
            dataset_digest: str,
            threshold_floor: float = 0.35,
            max_empirical_far: float = 0.005,
            max_upper_far: float = 0.01,
        ) -> CalibratedThreshold:
            del negative_scores, negative_group_ids, max_empirical_far, max_upper_far
            calibrate_digests.append(dataset_digest)
            return CalibratedThreshold(
                threshold=threshold_floor,
                empirical_far=0.0,
                far_upper_95=0.0,
                student_balanced_recall=(
                    1.0 if len(positive_scores) == len(positive_student_ids) else 0.0
                ),
                true_matches=len(positive_scores),
                feasible=True,
            )

        with patch(
            "tools.domain_adapter_v2_training.extract_session_embeddings",
            return_value=list(self._sessions_from_manifest(manifest)),
        ):
            with patch(
                "tools.domain_adapter_v2_training.train_fold_epoch",
                side_effect=self._fake_train_fold_epoch,
            ):
                with patch(
                    "tools.domain_adapter_v2_training.score_pair_set",
                    side_effect=self._fake_score_pair_set,
                ):
                    with patch(
                        "tools.domain_adapter_v2_training.v2_separation_loss",
                        side_effect=self._fake_v2_loss,
                    ):
                        with patch(
                            "tools.domain_adapter_v2_training._select_empirical_threshold",
                            side_effect=fake_empirical_threshold,
                        ):
                            with patch(
                            "tools.domain_adapter_v2_training.bootstrap_far_upper_bound",
                            side_effect=fake_bootstrap,
                            ):
                                with patch(
                                    "tools.domain_adapter_v2_training.calibrate_threshold",
                                    side_effect=fake_calibrate_threshold,
                                ):
                                    result = train_v2_candidate(
                                        manifest,
                                        self.cache_dir,
                                        self.config,
                                        seed=20260830,
                                        device="cpu",
                                    )
        return {
            "result": result,
            "bootstrap_calls": len(bootstrap_digests),
            "selection_bootstrap_calls": len(bootstrap_digests),
            "selection_digests": bootstrap_digests,
            "calibrate_calls": len(calibrate_digests),
            "final_calibration_digests": calibrate_digests,
        }

    @staticmethod
    def _validation_pair_set_with_extra_human_groups(
        *,
        synthetic_students: int,
        extra_human_groups: int,
    ) -> V2PairSet:
        sessions = tuple(
            V2CandidateTrainingTests._session(
                f"validation-mix-{index:03d}",
                split="validation",
            )
            for index in range(synthetic_students)
        )
        base = build_v2_pair_set(sessions)
        human_refs = np.repeat(base.negative_ref_embeddings[:1], extra_human_groups, axis=0)
        human_photos = np.repeat(
            base.negative_photo_embeddings[:1],
            extra_human_groups,
            axis=0,
        )
        human_group_ids = tuple(f"human-group-{index:04d}" for index in range(extra_human_groups))
        human_metadata = tuple(
            PairMetadata(
                ref_student_id="human-ref",
                ref_session_id="human-session",
                photo_student_id=f"hidden-student-{index:04d}",
                photo_session_id=f"hidden-session-{index:04d}",
                label=0,
                raw_cosine=0.0,
            )
            for index in range(extra_human_groups)
        )
        negative_ref_embeddings = np.concatenate(
            [base.negative_ref_embeddings, human_refs],
            axis=0,
        )
        negative_photo_embeddings = np.concatenate(
            [base.negative_photo_embeddings, human_photos],
            axis=0,
        )
        negative_group_ids = base.negative_group_ids + human_group_ids
        negative_categories = base.negative_categories + ("human_mismatch",) * extra_human_groups
        negative_weights = np.ones(len(negative_group_ids), dtype=np.float32)
        negative_weights /= float(len(negative_group_ids))
        negative_metadata = base.negative_metadata + human_metadata
        return V2PairSet(
            positive_ref_embeddings=base.positive_ref_embeddings,
            positive_photo_embeddings=base.positive_photo_embeddings,
            positive_student_ids=base.positive_student_ids,
            positive_session_ids=base.positive_session_ids,
            negative_ref_embeddings=negative_ref_embeddings,
            negative_photo_embeddings=negative_photo_embeddings,
            negative_group_ids=negative_group_ids,
            negative_categories=negative_categories,
            negative_weights=negative_weights,
            negative_metadata=negative_metadata,
        )

    @staticmethod
    def _fake_train_fold_epoch(
        _model: LowRankDomainAdapter,
        _optimizer: torch.optim.Optimizer,
        pair_set: V2PairSet,
        _config: V2TrainingConfig,
    ) -> V2EpochLoss:
        return V2EpochLoss(
            total=0.1,
            positive=0.02,
            negative=0.03,
            ranking=0.04,
            identity=0.0,
            residual_drift=0.01 + len(set(pair_set.positive_student_ids)) * 1e-6,
        )

    @staticmethod
    def _fake_score_pair_set(
        _model: LowRankDomainAdapter,
        pair_set: V2PairSet,
        _device: str,
    ):
        return type(
            "Scores",
            (),
            {
                "positive": np.full(len(pair_set.positive_student_ids), 0.80, dtype=np.float64),
                "negative": np.zeros(len(pair_set.negative_group_ids), dtype=np.float64),
            },
        )()

    @staticmethod
    def _fake_v2_loss(
        _model: LowRankDomainAdapter,
        _pair_set: V2PairSet,
        _config: V2TrainingConfig,
    ):
        zero = torch.tensor(0.0)
        return type(
            "Loss",
            (),
            {
                "total": zero,
                "positive": zero,
                "negative": zero,
                "ranking": zero,
                "identity": zero,
                "hard_negative_indices": torch.empty((0,), dtype=torch.int64),
            },
        )()

    @staticmethod
    def _fake_calibrate_threshold(
        *,
        positive_scores: np.ndarray,
        positive_student_ids: tuple[str, ...],
        negative_scores: np.ndarray,
        negative_group_ids: tuple[str, ...],
        dataset_digest: str,
    ) -> CalibratedThreshold:
        del negative_scores, negative_group_ids, dataset_digest
        return CalibratedThreshold(
            threshold=0.35,
            empirical_far=0.0,
            far_upper_95=0.0,
            student_balanced_recall=1.0 if len(positive_scores) == len(positive_student_ids) else 0.0,
            true_matches=len(positive_scores),
            feasible=True,
        )

    @staticmethod
    def _fake_empirical_threshold(
        *,
        positive_scores: np.ndarray,
        positive_student_ids: tuple[str, ...],
        negative_scores: np.ndarray,
        negative_group_ids: tuple[str, ...],
        threshold_floor: float = 0.35,
        max_empirical_far: float = 0.005,
    ) -> CalibratedThreshold:
        del negative_scores, negative_group_ids, max_empirical_far
        return CalibratedThreshold(
            threshold=threshold_floor,
            empirical_far=0.0,
            far_upper_95=0.0,
            student_balanced_recall=1.0 if len(positive_scores) == len(positive_student_ids) else 0.0,
            true_matches=len(positive_scores),
            feasible=True,
        )


class V2ArtifactAndCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.output_dir = self.root / "candidate-v2"
        self.embedding_cache = self.root / "embedding-cache"
        self.embedding_cache.mkdir()
        self.manifest_path = self.root / "dataset.json"
        self.manifest_path.write_text(
            json.dumps({"schema_version": 1, "sessions": [], "evaluation_sessions": []}),
            encoding="utf-8",
        )
        self.result = self._result()

    def test_write_v2_candidate_artifacts_exports_public_manifest_and_dynamic_batch_onnx(
        self,
    ) -> None:
        manifest = write_v2_candidate_artifacts(
            self.output_dir,
            self.result,
            seed=20260830,
        )

        self.assertEqual(
            sorted(path.name for path in self.output_dir.iterdir()),
            [
                "identity_domain_adapter.manifest.json",
                "identity_domain_adapter.onnx",
            ],
        )
        self.assertEqual(self.output_dir.stat().st_mode & 0o777, 0o700)
        for path in self.output_dir.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

        onnx_path = self.output_dir / "identity_domain_adapter.onnx"
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        batch_one_refs = self._normalized_rows(1, self.result.model.dimension)
        batch_one_photos = self._normalized_rows(
            1,
            self.result.model.dimension,
            offset=100,
        )
        batch_seven_refs = self._normalized_rows(
            7,
            self.result.model.dimension,
            offset=200,
        )
        batch_seven_photos = self._normalized_rows(
            7,
            self.result.model.dimension,
            offset=300,
        )
        for refs, photos in (
            (batch_one_refs, batch_one_photos),
            (batch_seven_refs, batch_seven_photos),
        ):
            actual = session.run(
                ["adapted_cosine"],
                {"ref_embedding": refs, "photo_embedding": photos},
            )[0]
            expected = (
                self.result.model(
                    torch.from_numpy(refs),
                    torch.from_numpy(photos),
                )
                .detach()
                .numpy()
            )
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
        self.assertEqual(session.get_inputs()[0].shape, ["N", self.result.model.dimension])
        self.assertEqual(session.get_outputs()[0].shape, ["N"])
        self.assertEqual(
            json.loads(
                (self.output_dir / "identity_domain_adapter.manifest.json").read_text(
                    encoding="utf-8"
                )
            ),
            manifest,
        )
        self.assertEqual(manifest["model_version"], "identity-domain-adapter-v2")
        self.assertLessEqual(manifest["onnx_parity_max_abs_error"], 1e-5)
        self.assertEqual(manifest["training"]["selected_epoch"], self.result.selection.epoch)

    def test_public_artifact_excludes_private_training_material(self) -> None:
        manifest = write_v2_candidate_artifacts(
            self.output_dir,
            self.result,
            seed=20260830,
        )

        encoded = json.dumps(manifest, sort_keys=True)
        for forbidden in (
            "student_id",
            "session_id",
            "image_path",
            "session_prototype",
            "group-secret",
            "student-secret",
            "/private/input",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(
            manifest["training"]["negative_categories"]["counts"],
            {
                "human_mismatch": 25,
                "synthetic_cross_student": 1000,
            },
        )
        self.assertEqual(
            manifest["training"]["negative_categories"]["weights"],
            {
                "human_mismatch": 0.2,
                "synthetic_cross_student": 0.8,
            },
        )
        self.assertNotIn("source_feedback_snapshot", manifest)

    def test_public_artifact_redacts_snapshot_and_omits_snapshot_key(self) -> None:
        self.result.source_feedback_snapshot = (
            "2026-08-30T12:00:00+00:00:/private/raw-feedback/session-17"
        )

        manifest = write_v2_candidate_artifacts(
            self.output_dir,
            self.result,
            seed=20260830,
        )

        encoded = json.dumps(manifest, sort_keys=True)
        self.assertNotIn("source_feedback_snapshot", manifest)
        self.assertNotIn(self.result.source_feedback_snapshot, encoded)

    def test_write_v2_candidate_artifacts_refuses_existing_output_path(self) -> None:
        self.output_dir.mkdir()

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            write_v2_candidate_artifacts(
                self.output_dir,
                self.result,
                seed=20260830,
            )

    def test_write_v2_candidate_artifacts_preserves_non_loadable_failure_evidence(
        self,
    ) -> None:
        with patch(
            "tools.domain_adapter_v2_training.export_onnx",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                write_v2_candidate_artifacts(
                    self.output_dir,
                    self.result,
                    seed=20260830,
                )

        self.assertFalse(self.output_dir.exists())
        evidence_dirs = [path for path in self.root.iterdir() if path.is_dir()]
        self.assertEqual(len(evidence_dirs), 2)
        failure_dirs = [path for path in evidence_dirs if path != self.embedding_cache]
        self.assertEqual(len(failure_dirs), 1)
        self.assertFalse(
            (failure_dirs[0] / "identity_domain_adapter.manifest.json").exists()
        )

    def test_write_v2_candidate_artifacts_never_overwrites_racing_destination(self) -> None:
        sentinel = self.output_dir / "sentinel.txt"

        def create_racing_destination() -> None:
            self.output_dir.mkdir(mode=0o700)
            sentinel.write_text("keep-me", encoding="utf-8")
            os.chmod(sentinel, 0o600)

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            write_v2_candidate_artifacts(
                self.output_dir,
                self.result,
                seed=20260830,
                _before_publish=create_racing_destination,
            )

        self.assertTrue(self.output_dir.is_dir())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep-me")
        self.assertFalse(
            (self.output_dir / "identity_domain_adapter.manifest.json").exists()
        )
        failure_dirs = [
            path
            for path in self.root.iterdir()
            if path.is_dir() and path not in {self.embedding_cache, self.output_dir}
        ]
        self.assertEqual(len(failure_dirs), 1)
        failure_dir = failure_dirs[0]
        self.assertEqual(failure_dir.stat().st_mode & 0o777, 0o700)
        self.assertFalse((failure_dir / "identity_domain_adapter.manifest.json").exists())
        evidence = failure_dir / "artifact-export-failure.json"
        self.assertTrue(evidence.exists())
        self.assertEqual(evidence.stat().st_mode & 0o777, 0o600)

    def test_cli_refuses_existing_output_directory(self) -> None:
        self.output_dir.mkdir()

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            train_main(
                [
                    "--manifest",
                    str(self.manifest_path),
                    "--embedding-cache",
                    str(self.embedding_cache),
                    "--output-dir",
                    str(self.output_dir),
                ]
            )

    def test_cli_rejects_tuning_overrides(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                train_main(
                    [
                        "--manifest",
                        str(self.manifest_path),
                        "--embedding-cache",
                        str(self.embedding_cache),
                        "--output-dir",
                        str(self.output_dir),
                        "--epochs",
                        "12",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)

    def test_cli_uses_fixed_v2_training_contract(self) -> None:
        stdout = io.StringIO()
        artifact = {"schema_version": 1, "model_version": "identity-domain-adapter-v2"}

        with patch(
            "tools.train_domain_adapter_v2.train_v2_candidate",
            return_value=self.result,
        ) as train_candidate:
            with patch(
                "tools.train_domain_adapter_v2.write_v2_candidate_artifacts",
                return_value=artifact,
            ) as write_artifacts:
                with redirect_stdout(stdout):
                    exit_code = train_main(
                        [
                            "--manifest",
                            str(self.manifest_path),
                            "--embedding-cache",
                            str(self.embedding_cache),
                            "--output-dir",
                            str(self.output_dir),
                            "--seed",
                            "20260901",
                            "--device",
                            "cpu",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        train_candidate.assert_called_once()
        self.assertEqual(train_candidate.call_args.args[1], self.embedding_cache)
        self.assertIsInstance(train_candidate.call_args.args[2], V2TrainingConfig)
        self.assertEqual(train_candidate.call_args.kwargs["seed"], 20260901)
        self.assertEqual(train_candidate.call_args.kwargs["device"], "cpu")
        write_artifacts.assert_called_once_with(
            self.output_dir,
            self.result,
            seed=20260901,
        )
        self.assertEqual(json.loads(stdout.getvalue()), artifact)

    @staticmethod
    def _normalized_rows(
        count: int,
        dimension: int,
        *,
        offset: int = 0,
    ) -> np.ndarray:
        rows = np.zeros((count, dimension), dtype=np.float32)
        for index in range(count):
            rows[index, (offset + index) % dimension] = 1.0
            rows[index, (offset + index + 1) % dimension] = 0.25
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        return rows.astype(np.float32, copy=False)

    def _result(self):
        torch.manual_seed(9)
        model = LowRankDomainAdapter(dimension=128, rank=16)
        with torch.no_grad():
            model.ref_tower.up.weight.normal_(std=0.01)
            model.photo_tower.up.weight.normal_(std=0.01)
        return type(
            "Result",
            (),
            {
                "model": model,
                "selection": V2Selection(
                    epoch=4,
                    median_recall=0.72,
                    worst_fold_recall=0.70,
                    pooled_far_upper_95=0.009,
                    residual_drift=0.02,
                    validation_loss=0.05,
                ),
                "fold_history": tuple(
                    FoldEpochMetrics(
                        fold=fold,
                        epoch=4,
                        candidate_threshold=0.35 + fold * 0.001,
                        empirical_far=0.001 * fold,
                        student_balanced_recall=0.70 + fold * 0.01,
                        validation_loss=0.05 + fold * 0.001,
                        residual_drift=0.02 + fold * 0.001,
                        negative_group_ids=(f"group-secret-{fold}",),
                        negative_accepts=(False,),
                        student_count=20,
                        session_count=20,
                        match_session_count=20,
                        negative_group_count=200 + fold,
                    )
                    for fold in range(5)
                ),
                "adapted_threshold": CalibratedThreshold(
                    threshold=0.41,
                    empirical_far=0.001,
                    far_upper_95=0.009,
                    student_balanced_recall=0.78,
                    true_matches=33,
                    feasible=True,
                ),
                "raw_threshold": CalibratedThreshold(
                    threshold=0.44,
                    empirical_far=0.003,
                    far_upper_95=0.01,
                    student_balanced_recall=0.71,
                    true_matches=30,
                    feasible=True,
                ),
                "dataset_digest": "a" * 64,
                "source_feedback_snapshot": "2026-08-30T12:00:00+00:00",
                "split_seed": "seed-v2",
                "split_counts": {
                    "train": {
                        "sessions": 100,
                        "positive_sessions": 100,
                        "negative_sessions": 0,
                    },
                    "validation": {
                        "sessions": 33,
                        "positive_sessions": 33,
                        "negative_sessions": 0,
                    },
                    "test": {
                        "sessions": 0,
                        "positive_sessions": 0,
                        "negative_sessions": 0,
                    },
                },
                "validation_negative_category_counts": {
                    "human_mismatch": 25,
                    "synthetic_cross_student": 1000,
                },
                "validation_negative_category_weight_totals": {
                    "human_mismatch": 0.2,
                    "synthetic_cross_student": 0.8,
                },
            },
        )()


if __name__ == "__main__":
    unittest.main()
