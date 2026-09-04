from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.domain_adapter_training import PairMetadata
from tools.domain_adapter_v3_metrics import (
    HistoricalRelativeMetrics,
    distinct_photo_student_hard_negatives,
    group_tail_indices,
    historical_relative_gate_passes,
    student_balanced_weights,
)


def _metadata(
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


class StudentBalancedWeightTests(unittest.TestCase):
    def test_each_student_has_equal_total_weight(self) -> None:
        weights = student_balanced_weights(("A", "A", "B"))
        np.testing.assert_allclose(weights, np.asarray([0.25, 0.25, 0.5]))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_empty_student_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            student_balanced_weights(())


class GroupTailTests(unittest.TestCase):
    def test_only_current_maximum_per_ordered_group_is_selected(self) -> None:
        scores = torch.tensor([0.20, 0.55, 0.40, 0.45])
        metadata = (
            _metadata("A", "A1", "B", "B1"),
            _metadata("A", "A1", "B", "B2"),
            _metadata("A", "A1", "C", "C1"),
            _metadata("A", "A1", "C", "C2"),
        )

        selected = group_tail_indices(
            scores,
            ("A>B", "A>B", "A>C", "A>C"),
            metadata,
        )

        self.assertEqual(selected.tolist(), [1, 3])

    def test_equal_scores_use_metadata_order_not_input_order(self) -> None:
        rows = (
            _metadata("A", "A1", "B", "B2"),
            _metadata("A", "A1", "B", "B1"),
        )
        selected = group_tail_indices(torch.tensor([0.5, 0.5]), ("A>B", "A>B"), rows)
        self.assertEqual(selected.tolist(), [1])

    def test_misaligned_group_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            group_tail_indices(
                torch.tensor([0.5]),
                (),
                (_metadata("A", "A1", "B", "B1"),),
            )


class DistinctHardNegativeTests(unittest.TestCase):
    def test_selects_at_most_one_maximum_per_photo_student(self) -> None:
        scores = torch.tensor([0.60, 0.55, 0.58, 0.99])
        metadata = (
            _metadata("A", "A1", "B", "B1"),
            _metadata("A", "A1", "B", "B2"),
            _metadata("A", "A1", "C", "C1"),
            _metadata("X", "X1", "Y", "Y1"),
        )

        selected = distinct_photo_student_hard_negatives(
            scores,
            metadata,
            positive_student_id="A",
            positive_session_id="A1",
            limit=20,
        )

        self.assertEqual(selected, [0, 2])

    def test_limit_applies_after_distinct_group_maxima(self) -> None:
        scores = torch.tensor([0.4, 0.7, 0.6])
        metadata = (
            _metadata("A", "A1", "B", "B1"),
            _metadata("A", "A1", "C", "C1"),
            _metadata("A", "A1", "D", "D1"),
        )
        selected = distinct_photo_student_hard_negatives(
            scores,
            metadata,
            positive_student_id="A",
            positive_session_id="A1",
            limit=2,
        )
        self.assertEqual(selected, [1, 2])


class HistoricalGateTests(unittest.TestCase):
    def _metrics(self, **overrides: float | int) -> HistoricalRelativeMetrics:
        values: dict[str, float | int] = {
            "raw_far": 0.01,
            "adapted_far": 0.01,
            "recall_lift": 0.02,
            "true_match_delta": 1,
            "same_threshold_recall_delta": 0.001,
            "adapted_threshold": 0.35,
        }
        values.update(overrides)
        return HistoricalRelativeMetrics(**values)

    def test_exact_gate_boundaries_pass(self) -> None:
        self.assertTrue(historical_relative_gate_passes(self._metrics()))

    def test_each_relative_gate_conjunct_fails_closed(self) -> None:
        failures = (
            {"adapted_far": 0.010001},
            {"recall_lift": 0.019999},
            {"true_match_delta": 0},
            {"same_threshold_recall_delta": 0.0},
            {"adapted_threshold": 0.349999},
        )
        for override in failures:
            with self.subTest(override=override):
                self.assertFalse(historical_relative_gate_passes(self._metrics(**override)))


if __name__ == "__main__":
    unittest.main()
