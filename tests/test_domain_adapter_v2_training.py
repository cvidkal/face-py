from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.domain_adapter_training import (
    LowRankDomainAdapter,
    PairMetadata,
    SessionEmbedding,
)
from tools.domain_adapter_v2_metrics import V2PairSet, build_v2_pair_set
from tools.domain_adapter_v2_training import (
    V2TrainingConfig,
    score_pair_set,
    train_fold_epoch,
    v2_separation_loss,
)


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


if __name__ == "__main__":
    unittest.main()
