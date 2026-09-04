"""Tail-focused separation loss for identity domain adapter v3."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from tools.domain_adapter_training import (
    LowRankDomainAdapter,
    PairMetadata,
    residual_weight_regularization,
)
from tools.domain_adapter_v2_metrics import V2PairSet
from tools.domain_adapter_v2_training import (
    _embedding_tensor,
    _model_device,
    _require_finite_component,
    _validate_model,
    _validate_pair_set_metadata,
    _zero_like,
)
from tools.domain_adapter_v3_metrics import (
    distinct_photo_student_hard_negatives,
    group_tail_indices,
    student_balanced_weights,
)


@dataclass(frozen=True)
class V3TrainingConfig:
    positive_margin: float = 0.35
    negative_margin: float = 0.30
    ranking_margin: float = 0.10
    positive_weight: float = 0.25
    negative_weight: float = 0.35
    ranking_weight: float = 0.40
    identity_regularization_weight: float = 0.001
    hard_negative_groups_per_positive: int = 20
    rank: int = 16
    max_epochs: int = 100
    learning_rate: float = 0.01

    def __post_init__(self) -> None:
        approved = {
            "positive_margin": 0.35,
            "negative_margin": 0.30,
            "ranking_margin": 0.10,
            "positive_weight": 0.25,
            "negative_weight": 0.35,
            "ranking_weight": 0.40,
            "identity_regularization_weight": 0.001,
            "hard_negative_groups_per_positive": 20,
            "rank": 16,
            "max_epochs": 100,
            "learning_rate": 0.01,
        }
        for name, expected in approved.items():
            value = getattr(self, name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value != expected:
                raise ValueError(f"{name} must be {expected}")


@dataclass(frozen=True)
class V3Loss:
    total: Tensor
    positive: Tensor
    negative: Tensor
    ranking: Tensor
    identity: Tensor
    tail_negative_indices: Tensor
    hard_negative_indices: Tensor


@dataclass(frozen=True)
class V3EpochLoss:
    total: float
    positive: float
    negative: float
    ranking: float
    identity: float
    residual_drift: float


def v3_separation_loss(
    model: LowRankDomainAdapter,
    pair_set: V2PairSet,
    config: V3TrainingConfig,
) -> V3Loss:
    _validate_model(model, expected_rank=config.rank)
    device = _model_device(model)
    positive_refs = _embedding_tensor(
        pair_set.positive_ref_embeddings,
        dimension=model.dimension,
        description="positive_ref_embeddings",
        device=device,
    )
    positive_photos = _embedding_tensor(
        pair_set.positive_photo_embeddings,
        dimension=model.dimension,
        description="positive_photo_embeddings",
        device=device,
    )
    negative_refs = _embedding_tensor(
        pair_set.negative_ref_embeddings,
        dimension=model.dimension,
        description="negative_ref_embeddings",
        device=device,
    )
    negative_photos = _embedding_tensor(
        pair_set.negative_photo_embeddings,
        dimension=model.dimension,
        description="negative_photo_embeddings",
        device=device,
    )
    _validate_pair_set_metadata(pair_set, positive_refs.shape[0], negative_refs.shape[0])

    positive_scores = model(positive_refs, positive_photos)
    negative_scores = model(negative_refs, negative_photos)
    zero = _zero_like(model, positive_scores, negative_scores)
    positive = _student_balanced_positive_loss(
        positive_scores,
        pair_set.positive_student_ids,
        margin=config.positive_margin,
        zero=zero,
    )
    negative, tail_indices = _group_tail_negative_loss(
        negative_scores,
        pair_set.negative_group_ids,
        pair_set.negative_metadata,
        margin=config.negative_margin,
        zero=zero,
    )
    ranking, hard_indices = _student_balanced_ranking_loss(
        positive_scores,
        negative_scores,
        positive_student_ids=pair_set.positive_student_ids,
        positive_session_ids=pair_set.positive_session_ids,
        negative_metadata=pair_set.negative_metadata,
        margin=config.ranking_margin,
        limit=config.hard_negative_groups_per_positive,
        zero=zero,
    )
    identity = residual_weight_regularization(model)
    for name, component in (
        ("positive", positive),
        ("negative", negative),
        ("ranking", ranking),
        ("identity", identity),
    ):
        _require_finite_component(component, name)
    total = (
        config.positive_weight * positive
        + config.negative_weight * negative
        + config.ranking_weight * ranking
        + config.identity_regularization_weight * identity
    )
    _require_finite_component(total, "total")
    return V3Loss(
        total=total,
        positive=positive,
        negative=negative,
        ranking=ranking,
        identity=identity,
        tail_negative_indices=tail_indices,
        hard_negative_indices=hard_indices,
    )


def train_v3_epoch(
    model: LowRankDomainAdapter,
    optimizer: torch.optim.Optimizer,
    pair_set: V2PairSet,
    config: V3TrainingConfig,
) -> V3EpochLoss:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = v3_separation_loss(model, pair_set, config)
    if not torch.isfinite(loss.total):
        raise FloatingPointError("non-finite v3 loss")
    loss.total.backward()
    optimizer.step()
    residual_drift = residual_weight_regularization(model).detach()
    _require_finite_component(residual_drift, "residual_drift")
    return V3EpochLoss(
        total=float(loss.total.detach()),
        positive=float(loss.positive.detach()),
        negative=float(loss.negative.detach()),
        ranking=float(loss.ranking.detach()),
        identity=float(loss.identity.detach()),
        residual_drift=float(residual_drift),
    )


def _student_balanced_positive_loss(
    scores: Tensor,
    student_ids: Sequence[str],
    *,
    margin: float,
    zero: Tensor | None = None,
) -> Tensor:
    values = scores.reshape(-1)
    if values.numel() != len(student_ids):
        raise ValueError("positive scores and student IDs must align")
    if not values.numel():
        return values.sum() * 0.0 if zero is None else zero
    weights = torch.from_numpy(student_balanced_weights(student_ids)).to(
        device=values.device,
        dtype=values.dtype,
    )
    return (F.relu(margin - values) * weights).sum()


def _group_tail_negative_loss(
    scores: Tensor,
    group_ids: Sequence[str],
    metadata: Sequence[PairMetadata],
    *,
    margin: float,
    zero: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    values = scores.reshape(-1)
    if not values.numel():
        empty = torch.empty((0,), dtype=torch.int64, device=values.device)
        return (values.sum() * 0.0 if zero is None else zero), empty
    selected = group_tail_indices(values, group_ids, metadata)
    selected_scores = values.index_select(0, selected)
    return F.relu(selected_scores - margin).mean(), selected


def _student_balanced_ranking_loss(
    positive_scores: Tensor,
    negative_scores: Tensor,
    *,
    positive_student_ids: Sequence[str],
    positive_session_ids: Sequence[str],
    negative_metadata: Sequence[PairMetadata],
    margin: float,
    limit: int,
    zero: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    positives = positive_scores.reshape(-1)
    negatives = negative_scores.reshape(-1)
    if positives.numel() != len(positive_student_ids):
        raise ValueError("positive scores and student IDs must align")
    if positives.numel() != len(positive_session_ids):
        raise ValueError("positive scores and session IDs must align")
    if negatives.numel() != len(negative_metadata):
        raise ValueError("negative scores and metadata must align")
    if not positives.numel() or not negatives.numel():
        empty = torch.empty((0,), dtype=torch.int64, device=positives.device)
        base = positives.sum() * 0.0 + negatives.sum() * 0.0
        return (base if zero is None else zero), empty

    by_student: dict[str, list[Tensor]] = defaultdict(list)
    flattened_indices: list[int] = []
    for positive_index, (student_id, session_id) in enumerate(
        zip(positive_student_ids, positive_session_ids, strict=True)
    ):
        selected = distinct_photo_student_hard_negatives(
            negatives,
            negative_metadata,
            positive_student_id=student_id,
            positive_session_id=session_id,
            limit=limit,
        )
        if not selected:
            continue
        index_tensor = torch.tensor(selected, dtype=torch.int64, device=negatives.device)
        selected_scores = negatives.index_select(0, index_tensor)
        per_positive = F.relu(margin - positives[positive_index] + selected_scores).mean()
        by_student[str(student_id)].append(per_positive)
        flattened_indices.extend(selected)
    if not by_student:
        empty = torch.empty((0,), dtype=torch.int64, device=positives.device)
        base = positives.sum() * 0.0 + negatives.sum() * 0.0
        return (base if zero is None else zero), empty
    student_losses = [torch.stack(by_student[key]).mean() for key in sorted(by_student)]
    return (
        torch.stack(student_losses).mean(),
        torch.tensor(flattened_indices, dtype=torch.int64, device=negatives.device),
    )
