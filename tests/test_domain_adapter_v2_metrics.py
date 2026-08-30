from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

import numpy as np

import tools.domain_adapter_v2_metrics as domain_adapter_v2_metrics
from tools.domain_adapter_training import SessionEmbedding
from tools.domain_adapter_v2_metrics import (
    BudgetComparison,
    CalibratedThreshold,
    HumanMismatchEvidence,
    RelativeGateMetrics,
    assign_training_folds,
    bootstrap_far_upper_bound,
    build_v2_pair_set,
    calibrate_threshold,
    compare_at_far_budget,
    relative_gate_passes,
    student_balanced_match_recall,
    student_fold,
)


class DomainAdapterV2MetricTests(unittest.TestCase):
    @staticmethod
    def _session(
        student_id: str,
        session_id: str,
        *,
        split: str = "train",
        label: str = "match",
        vector: tuple[float, float] = (1.0, 0.0),
        prototype: tuple[float, float] | None = None,
    ) -> SessionEmbedding:
        ref = np.asarray(vector, dtype=np.float32)
        ref /= np.linalg.norm(ref)
        photo = np.asarray(prototype if prototype is not None else vector, dtype=np.float32)
        photo /= np.linalg.norm(photo)
        return SessionEmbedding(
            student_id=student_id,
            session_id=session_id,
            split=split,
            label=label,
            ref_embedding=ref,
            session_prototype=photo,
        )

    def setUp(self) -> None:
        self.sessions = [
            self._session("student-a", "session-1", vector=(1.0, 0.0)),
            self._session("student-a", "session-2", vector=(0.8, 0.2)),
            self._session("student-b", "session-1", vector=(0.0, 1.0)),
        ]

    def test_student_fold_uses_declared_sha256_mapping(self) -> None:
        expected = int.from_bytes(
            hashlib.sha256(
                b"identity-domain-adapter-v2-folds\0student-7"
            ).digest()[:8],
            "big",
        ) % 5
        self.assertEqual(student_fold("student-7"), expected)

    def test_assign_training_folds_groups_sessions_by_student_hash(self) -> None:
        folds = assign_training_folds(
            [
                self._session("student-b", "session-2"),
                self._session("student-a", "session-2"),
                self._session("student-a", "session-1"),
            ]
        )

        self.assertEqual(sorted(folds), [0, 1, 2, 3, 4])
        self.assertEqual(
            tuple(
                (session.student_id, session.session_id)
                for session in folds[student_fold("student-a")]
            ),
            (("student-a", "session-1"), ("student-a", "session-2")),
        )
        self.assertEqual(
            tuple(
                (session.student_id, session.session_id)
                for session in folds[student_fold("student-b")]
            ),
            (("student-b", "session-2"),),
        )

    def test_pair_builder_keeps_every_cross_student_session_pair(self) -> None:
        pairs = build_v2_pair_set(self.sessions)
        expected = sum(
            1
            for left in self.sessions
            for right in self.sessions
            if left.student_id != right.student_id
        )
        self.assertEqual(len(pairs.negative_metadata), expected)

    def test_pair_builder_assigns_equal_group_weights_and_separates_human_mismatch(
        self,
    ) -> None:
        sessions = [
            self._session("student-a", "session-2", vector=(0.8, 0.2)),
            self._session("student-z", "session-9", label="mismatch", vector=(1.0, 0.0)),
            self._session("student-b", "session-1", vector=(0.0, 1.0)),
            self._session("student-a", "session-1", vector=(1.0, 0.0)),
        ]
        mismatch_evidence = (
            HumanMismatchEvidence(
                student_id="student-z",
                session_id="session-9",
                ordered_pair_token="hashed-human-pair-z",
                photo_student_token="hidden-student-z",
                photo_session_token="hidden-session-z",
            ),
        )

        pairs = build_v2_pair_set(sessions, human_mismatch_evidence=mismatch_evidence)
        rebuilt = build_v2_pair_set(
            list(reversed(sessions)),
            human_mismatch_evidence=mismatch_evidence,
        )

        self.assertEqual(pairs.positive_student_ids, ("student-a", "student-a", "student-b"))
        self.assertEqual(pairs.positive_session_ids, ("session-1", "session-2", "session-1"))
        self.assertEqual(pairs.negative_categories.count("synthetic_cross_student"), 4)
        self.assertEqual(pairs.negative_categories.count("human_mismatch"), 1)
        self.assertEqual(len(set(pairs.negative_group_ids)), 3)
        self.assertTrue(all("student-" not in group_id for group_id in pairs.negative_group_ids))
        self.assertAlmostEqual(float(pairs.negative_weights.sum()), 1.0, places=7)
        self.assertEqual(
            pairs.negative_category_counts,
            {"synthetic_cross_student": 4, "human_mismatch": 1},
        )
        self.assertAlmostEqual(
            pairs.negative_category_weight_totals["human_mismatch"],
            1.0 / 3.0,
            places=7,
        )

        group_weight_totals: dict[str, float] = {}
        for group_id, weight in zip(
            pairs.negative_group_ids, pairs.negative_weights, strict=True
        ):
            group_weight_totals[group_id] = group_weight_totals.get(group_id, 0.0) + float(
                weight
            )
        for total in group_weight_totals.values():
            self.assertAlmostEqual(total, 1.0 / 3.0, places=7)
        self.assertAlmostEqual(float(pairs.negative_weights[0]), 1.0 / 6.0, places=7)

        np.testing.assert_array_equal(
            pairs.positive_ref_embeddings, rebuilt.positive_ref_embeddings
        )
        np.testing.assert_array_equal(
            pairs.positive_photo_embeddings, rebuilt.positive_photo_embeddings
        )
        np.testing.assert_array_equal(
            pairs.negative_ref_embeddings, rebuilt.negative_ref_embeddings
        )
        np.testing.assert_array_equal(
            pairs.negative_photo_embeddings, rebuilt.negative_photo_embeddings
        )
        np.testing.assert_array_equal(pairs.negative_weights, rebuilt.negative_weights)
        self.assertEqual(pairs.negative_group_ids, rebuilt.negative_group_ids)
        self.assertEqual(pairs.negative_categories, rebuilt.negative_categories)
        self.assertEqual(pairs.negative_metadata, rebuilt.negative_metadata)

    def test_pair_builder_groups_multiple_human_mismatch_observations_by_token(
        self,
    ) -> None:
        sessions = [
            self._session("student-a", "session-1", vector=(1.0, 0.0)),
            self._session("student-b", "session-1", vector=(0.0, 1.0)),
            self._session("student-z", "session-9", label="mismatch", vector=(1.0, 0.0)),
            self._session("student-z", "session-10", label="mismatch", vector=(0.8, 0.2)),
        ]
        evidence = (
            HumanMismatchEvidence(
                student_id="student-z",
                session_id="session-9",
                ordered_pair_token="hashed-human-pair-z",
                photo_student_token="hidden-student-z",
                photo_session_token="hidden-session-z-1",
            ),
            HumanMismatchEvidence(
                student_id="student-z",
                session_id="session-10",
                ordered_pair_token="hashed-human-pair-z",
                photo_student_token="hidden-student-z",
                photo_session_token="hidden-session-z-2",
            ),
        )

        pairs = build_v2_pair_set(sessions, human_mismatch_evidence=evidence)

        self.assertEqual(pairs.negative_categories.count("human_mismatch"), 2)
        self.assertEqual(
            pairs.negative_group_ids.count("hashed-human-pair-z"),
            2,
        )
        self.assertAlmostEqual(
            pairs.negative_category_weight_totals["human_mismatch"],
            1.0 / 3.0,
            places=7,
        )
        for group_id in set(pairs.negative_group_ids):
            total = sum(
                float(weight)
                for actual_group_id, weight in zip(
                    pairs.negative_group_ids,
                    pairs.negative_weights,
                    strict=True,
                )
                if actual_group_id == group_id
            )
            self.assertAlmostEqual(total, 1.0 / 3.0, places=7)

    def test_pair_builder_rejects_human_mismatch_token_reused_by_different_ref_students(
        self,
    ) -> None:
        sessions = [
            self._session("student-a", "session-1", vector=(1.0, 0.0)),
            self._session("student-b", "session-1", vector=(0.0, 1.0)),
            self._session("student-z", "session-9", label="mismatch", vector=(1.0, 0.0)),
            self._session("student-y", "session-8", label="mismatch", vector=(0.8, 0.2)),
        ]
        evidence = (
            HumanMismatchEvidence(
                student_id="student-z",
                session_id="session-9",
                ordered_pair_token="shared-human-token",
                photo_student_token="hidden-student-token",
                photo_session_token="hidden-session-1",
            ),
            HumanMismatchEvidence(
                student_id="student-y",
                session_id="session-8",
                ordered_pair_token="shared-human-token",
                photo_student_token="hidden-student-token",
                photo_session_token="hidden-session-2",
            ),
        )

        with self.assertRaisesRegex(ValueError, "contradictory group metadata"):
            build_v2_pair_set(sessions, human_mismatch_evidence=evidence)

    def test_pair_builder_rejects_mismatch_session_without_grouping_evidence(self) -> None:
        sessions = [
            self._session("student-a", "session-1", vector=(1.0, 0.0)),
            self._session("student-z", "session-9", label="mismatch", vector=(1.0, 0.0)),
        ]

        with self.assertRaisesRegex(ValueError, "human mismatch evidence"):
            build_v2_pair_set(sessions)

    def test_student_balanced_match_recall_weights_students_equally(self) -> None:
        recall = student_balanced_match_recall(
            np.asarray([0.90, 0.10, 0.80], dtype=np.float32),
            ("student-a", "student-a", "student-b"),
            0.35,
        )

        self.assertAlmostEqual(recall, 0.75, places=7)

    def test_bootstrap_far_upper_bound_is_deterministic(self) -> None:
        scores = np.asarray([0.90, 0.10, 0.20], dtype=np.float32)
        group_ids = ("group-a", "group-a", "group-b")

        first = bootstrap_far_upper_bound(scores, group_ids, 0.35, "digest-1", iterations=200)
        second = bootstrap_far_upper_bound(
            scores, group_ids, 0.35, "digest-1", iterations=200
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first.estimate, 1.0 / 3.0, places=7)
        self.assertGreaterEqual(first.upper_95, first.estimate)
        self.assertEqual(first.iterations, 200)
        self.assertEqual(first.group_count, 2)

    def test_bootstrap_far_upper_bound_batches_group_draws(self) -> None:
        class RecordingGenerator:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def multinomial(
                self, n: int, pvals: np.ndarray, size: int
            ) -> np.ndarray:
                self.calls.append(size)
                draws = np.zeros((size, len(pvals)), dtype=np.int64)
                draws[:, 0] = n
                return draws

        generator = RecordingGenerator()
        group_count = 9900
        scores = np.zeros(group_count, dtype=np.float32)
        group_ids = tuple(f"group-{index:05d}" for index in range(group_count))

        with patch.object(
            domain_adapter_v2_metrics.np.random,
            "default_rng",
            return_value=generator,
        ):
            result = bootstrap_far_upper_bound(
                scores,
                group_ids,
                0.35,
                "digest-many-groups",
            )

        self.assertEqual(result.iterations, 10000)
        self.assertGreater(len(generator.calls), 1)
        self.assertEqual(sum(generator.calls), 10000)
        self.assertLessEqual(
            max(generator.calls),
            domain_adapter_v2_metrics._BOOTSTRAP_DRAW_BATCH_SIZE,
        )

    def test_bootstrap_far_upper_bound_rejects_nonfinite_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            bootstrap_far_upper_bound(
                np.asarray([0.2, np.nan], dtype=np.float32),
                ("group-a", "group-b"),
                0.35,
                "digest-1",
            )

    def test_calibrate_threshold_accepts_exact_point_five_percent_far_boundary(
        self,
    ) -> None:
        positive_scores = np.asarray([0.3505, 0.40, 0.60], dtype=np.float32)
        positive_students = ("student-a", "student-b", "student-c")
        negative_scores = np.asarray([*([0.3505] * 5), *([0.0] * 995)], dtype=np.float32)
        negative_group_ids = tuple(f"group-{index:04d}" for index in range(1000))

        result = calibrate_threshold(
            positive_scores=positive_scores,
            positive_student_ids=positive_students,
            negative_scores=negative_scores,
            negative_group_ids=negative_group_ids,
            dataset_digest="digest-2",
        )

        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.threshold, 0.35, places=7)
        self.assertAlmostEqual(result.empirical_far, 0.005, places=7)
        self.assertAlmostEqual(result.student_balanced_recall, 1.0, places=7)
        self.assertEqual(result.true_matches, 3)

    def test_compare_at_far_budget_accepts_exact_one_percent_boundary(self) -> None:
        positive_scores = np.asarray([0.3505, 0.36, 0.70], dtype=np.float32)
        negative_scores = np.asarray([0.3505, *([0.0] * 99)], dtype=np.float32)
        group_ids = tuple(f"group-{index:03d}" for index in range(100))

        budget = compare_at_far_budget(
            raw_positive_scores=positive_scores,
            adapted_positive_scores=positive_scores,
            positive_student_ids=("student-a", "student-b", "student-c"),
            raw_negative_scores=negative_scores,
            adapted_negative_scores=negative_scores,
            negative_group_ids=group_ids,
            dataset_digest="digest-3",
        )

        self.assertAlmostEqual(budget.raw.threshold, 0.35, places=7)
        self.assertAlmostEqual(budget.raw.empirical_far, 0.01, places=7)
        self.assertAlmostEqual(budget.adapted.threshold, 0.35, places=7)
        self.assertAlmostEqual(budget.adapted.empirical_far, 0.01, places=7)

    def test_relative_gate_rejects_threshold_only_apparent_lift(self) -> None:
        positive_scores = np.asarray([0.3505, 0.3505, 0.34, 0.34], dtype=np.float32)
        student_ids = ("student-a", "student-b", "student-c", "student-d")
        raw_negative_scores = np.asarray([0.3505, *([0.0] * 98)], dtype=np.float32)
        adapted_negative_scores = np.zeros(99, dtype=np.float32)
        group_ids = tuple(f"group-{index:03d}" for index in range(99))

        budget = compare_at_far_budget(
            raw_positive_scores=positive_scores,
            adapted_positive_scores=positive_scores,
            positive_student_ids=student_ids,
            raw_negative_scores=raw_negative_scores,
            adapted_negative_scores=adapted_negative_scores,
            negative_group_ids=group_ids,
            dataset_digest="digest-4",
        )

        self.assertGreater(budget.recall_lift, 0.0)
        self.assertEqual(
            budget.same_threshold_adapted_recall,
            budget.same_threshold_raw_recall,
        )
        self.assertFalse(
            relative_gate_passes(
                RelativeGateMetrics(
                    raw_frozen_far=0.02,
                    adapted_frozen_far=0.01,
                    budget=budget,
                    same_threshold_recall_delta=(
                        budget.same_threshold_adapted_recall
                        - budget.same_threshold_raw_recall
                    ),
                )
            )
        )

    def test_relative_gate_accepts_exact_recall_lift_boundary(self) -> None:
        raw = CalibratedThreshold(
            threshold=0.35,
            empirical_far=0.01,
            far_upper_95=0.01,
            student_balanced_recall=0.50,
            true_matches=50,
            feasible=True,
        )
        adapted = CalibratedThreshold(
            threshold=0.35,
            empirical_far=0.01,
            far_upper_95=0.01,
            student_balanced_recall=0.52,
            true_matches=51,
            feasible=True,
        )
        budget = BudgetComparison(
            raw=raw,
            adapted=adapted,
            recall_lift=0.02,
            true_match_delta=1,
            far_delta=0.0,
            same_threshold_raw_recall=0.50,
            same_threshold_adapted_recall=0.51,
        )

        self.assertTrue(
            relative_gate_passes(
                RelativeGateMetrics(
                    raw_frozen_far=0.02,
                    adapted_frozen_far=0.02,
                    budget=budget,
                    same_threshold_recall_delta=0.01,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
