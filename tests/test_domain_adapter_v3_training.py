from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.domain_adapter_training import LowRankDomainAdapter, PairMetadata
from tools.domain_adapter_v2_metrics import V2PairSet
from tools.domain_adapter_v3_training import (
    V3TrainingConfig,
    _group_tail_negative_loss,
    _student_balanced_positive_loss,
    _student_balanced_ranking_loss,
    v3_separation_loss,
)


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


if __name__ == "__main__":
    unittest.main()
