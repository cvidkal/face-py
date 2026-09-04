"""Student-balanced tail-selection primitives for identity adapter v3."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from tools.domain_adapter_training import PairMetadata


@dataclass(frozen=True)
class HistoricalRelativeMetrics:
    raw_far: float
    adapted_far: float
    recall_lift: float
    true_match_delta: int
    same_threshold_recall_delta: float
    adapted_threshold: float


def student_balanced_weights(student_ids: Sequence[str]) -> np.ndarray:
    if not student_ids:
        raise ValueError("student_ids must not be empty")
    normalized = tuple(str(value) for value in student_ids)
    counts = Counter(normalized)
    student_weight = 1.0 / len(counts)
    return np.asarray(
        [student_weight / counts[value] for value in normalized],
        dtype=np.float32,
    )


def group_tail_indices(
    scores: Tensor,
    group_ids: Sequence[str],
    metadata: Sequence[PairMetadata],
) -> Tensor:
    values = scores.reshape(-1)
    if values.numel() != len(group_ids) or values.numel() != len(metadata):
        raise ValueError("scores, group_ids, and metadata must align")
    if not torch.isfinite(values).all():
        raise ValueError("scores must be finite")

    score_values = values.detach().cpu().numpy()
    best: dict[str, tuple[float, tuple[str, str, str, str, int], int]] = {}
    for index, (group_id, row) in enumerate(zip(group_ids, metadata, strict=True)):
        candidate = (
            float(score_values[index]),
            _metadata_order(row, index),
            index,
        )
        previous = best.get(str(group_id))
        if previous is None or candidate[0] > previous[0] or (
            candidate[0] == previous[0] and candidate[1] < previous[1]
        ):
            best[str(group_id)] = candidate
    selected = [best[group_id][2] for group_id in sorted(best)]
    return torch.tensor(selected, dtype=torch.int64, device=values.device)


def distinct_photo_student_hard_negatives(
    scores: Tensor,
    metadata: Sequence[PairMetadata],
    *,
    positive_student_id: str,
    positive_session_id: str,
    limit: int,
) -> list[int]:
    values = scores.reshape(-1)
    if values.numel() != len(metadata):
        raise ValueError("scores and metadata must align")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not torch.isfinite(values).all():
        raise ValueError("scores must be finite")

    score_values = values.detach().cpu().numpy()
    best: dict[str, tuple[float, tuple[str, str, str, str, int], int]] = {}
    for index, row in enumerate(metadata):
        if (
            row.ref_student_id != positive_student_id
            or row.ref_session_id != positive_session_id
        ):
            continue
        candidate = (
            float(score_values[index]),
            _metadata_order(row, index),
            index,
        )
        previous = best.get(row.photo_student_id)
        if previous is None or candidate[0] > previous[0] or (
            candidate[0] == previous[0] and candidate[1] < previous[1]
        ):
            best[row.photo_student_id] = candidate
    ordered = sorted(best.values(), key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ordered[:limit]]


def historical_relative_gate_passes(metrics: HistoricalRelativeMetrics) -> bool:
    values = (
        metrics.raw_far,
        metrics.adapted_far,
        metrics.recall_lift,
        metrics.same_threshold_recall_delta,
        metrics.adapted_threshold,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    return (
        metrics.adapted_far <= metrics.raw_far
        and metrics.adapted_far <= 0.01
        and metrics.recall_lift >= 0.02
        and metrics.true_match_delta > 0
        and metrics.same_threshold_recall_delta > 0.0
        and metrics.adapted_threshold >= 0.35
    )


def _metadata_order(row: PairMetadata, index: int) -> tuple[str, str, str, str, int]:
    return (
        row.ref_student_id,
        row.ref_session_id,
        row.photo_student_id,
        row.photo_session_id,
        index,
    )
