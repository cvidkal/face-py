"""Deterministic v2 training objective for the identity-domain adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from tools.domain_adapter_training import (
    LowRankDomainAdapter,
    PairMetadata,
    residual_weight_regularization,
)
from tools.domain_adapter_v2_metrics import V2PairSet


@dataclass(frozen=True)
class V2TrainingConfig:
    positive_margin: float = 0.35
    negative_margin: float = 0.30
    ranking_margin: float = 0.10
    positive_weight: float = 0.25
    negative_weight: float = 0.35
    ranking_weight: float = 0.40
    identity_regularization_weight: float = 0.001
    hard_negatives_per_positive: int = 10
    rank: int = 16
    max_epochs: int = 100
    learning_rate: float = 0.01

    def __post_init__(self) -> None:
        _require_positive(self.positive_margin, "positive_margin")
        _require_positive(self.negative_margin, "negative_margin")
        _require_positive(self.ranking_margin, "ranking_margin")
        _require_non_negative(self.positive_weight, "positive_weight")
        _require_non_negative(self.negative_weight, "negative_weight")
        _require_non_negative(self.ranking_weight, "ranking_weight")
        _require_non_negative(
            self.identity_regularization_weight,
            "identity_regularization_weight",
        )
        _require_positive(self.learning_rate, "learning_rate")
        if (
            self.positive_weight == 0.0
            and self.negative_weight == 0.0
            and self.ranking_weight == 0.0
        ):
            raise ValueError("at least one separation term must be enabled")
        if self.hard_negatives_per_positive != 10:
            raise ValueError("hard_negatives_per_positive must be 10")
        if self.rank != 16:
            raise ValueError("rank must be 16")
        if self.max_epochs != 100:
            raise ValueError("max_epochs must be 100")


@dataclass(frozen=True)
class V2Scores:
    positive: np.ndarray
    negative: np.ndarray


@dataclass(frozen=True)
class V2Loss:
    total: Tensor
    positive: Tensor
    negative: Tensor
    ranking: Tensor
    identity: Tensor
    hard_negative_indices: Tensor


@dataclass(frozen=True)
class V2EpochLoss:
    total: float
    positive: float
    negative: float
    ranking: float
    identity: float
    residual_drift: float


def score_pair_set(
    model: LowRankDomainAdapter,
    pair_set: V2PairSet,
    device: str,
) -> V2Scores:
    _validate_model(model)
    model_device = _model_device(model)
    requested_device = _requested_device(device)
    if requested_device != model_device:
        raise ValueError(
            f"requested device {requested_device} does not match model device {model_device}"
        )
    positive_refs = _embedding_tensor(
        pair_set.positive_ref_embeddings,
        dimension=model.dimension,
        description="positive_ref_embeddings",
        device=model_device,
    )
    positive_photos = _embedding_tensor(
        pair_set.positive_photo_embeddings,
        dimension=model.dimension,
        description="positive_photo_embeddings",
        device=model_device,
    )
    negative_refs = _embedding_tensor(
        pair_set.negative_ref_embeddings,
        dimension=model.dimension,
        description="negative_ref_embeddings",
        device=model_device,
    )
    negative_photos = _embedding_tensor(
        pair_set.negative_photo_embeddings,
        dimension=model.dimension,
        description="negative_photo_embeddings",
        device=model_device,
    )
    _validate_pair_set_metadata(pair_set, positive_refs.shape[0], negative_refs.shape[0])

    previous_mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            positive_scores = model(positive_refs, positive_photos)
            negative_scores = model(negative_refs, negative_photos)
    finally:
        model.train(previous_mode)
    return V2Scores(
        positive=positive_scores.detach().cpu().numpy().astype(np.float64, copy=False),
        negative=negative_scores.detach().cpu().numpy().astype(np.float64, copy=False),
    )


def v2_separation_loss(
    model: LowRankDomainAdapter,
    pair_set: V2PairSet,
    config: V2TrainingConfig,
) -> V2Loss:
    _validate_model(model, expected_rank=config.rank)
    positive_refs = _embedding_tensor(
        pair_set.positive_ref_embeddings,
        dimension=model.dimension,
        description="positive_ref_embeddings",
        device=_model_device(model),
    )
    positive_photos = _embedding_tensor(
        pair_set.positive_photo_embeddings,
        dimension=model.dimension,
        description="positive_photo_embeddings",
        device=_model_device(model),
    )
    negative_refs = _embedding_tensor(
        pair_set.negative_ref_embeddings,
        dimension=model.dimension,
        description="negative_ref_embeddings",
        device=_model_device(model),
    )
    negative_photos = _embedding_tensor(
        pair_set.negative_photo_embeddings,
        dimension=model.dimension,
        description="negative_photo_embeddings",
        device=_model_device(model),
    )
    _validate_pair_set_metadata(pair_set, positive_refs.shape[0], negative_refs.shape[0])

    positive_scores = model(positive_refs, positive_photos)
    negative_scores = model(negative_refs, negative_photos)
    zero = _zero_like(model, positive_scores, negative_scores)

    positive = (
        F.relu(config.positive_margin - positive_scores).mean()
        if positive_scores.numel()
        else zero
    )
    negative_weights = _negative_weight_tensor(
        pair_set.negative_weights,
        expected=negative_scores.shape[0],
        device=negative_scores.device,
    )
    negative = (
        (F.relu(negative_scores - config.negative_margin) * negative_weights).sum()
        if negative_scores.numel()
        else zero
    )
    ranking, hard_negative_indices = _ranking_loss(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        pair_set=pair_set,
        config=config,
        zero=zero,
    )
    identity = residual_weight_regularization(model)

    _require_finite_component(positive, "positive")
    _require_finite_component(negative, "negative")
    _require_finite_component(ranking, "ranking")
    _require_finite_component(identity, "identity")

    total = (
        config.positive_weight * positive
        + config.negative_weight * negative
        + config.ranking_weight * ranking
        + config.identity_regularization_weight * identity
    )
    _require_finite_component(total, "total")
    return V2Loss(
        total=total,
        positive=positive,
        negative=negative,
        ranking=ranking,
        identity=identity,
        hard_negative_indices=hard_negative_indices,
    )


def train_fold_epoch(
    model: LowRankDomainAdapter,
    optimizer: torch.optim.Optimizer,
    pair_set: V2PairSet,
    config: V2TrainingConfig,
) -> V2EpochLoss:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = v2_separation_loss(model, pair_set, config)
    if not torch.isfinite(loss.total):
        raise FloatingPointError("non-finite v2 loss")
    loss.total.backward()
    optimizer.step()
    residual_drift = residual_weight_regularization(model).detach()
    _require_finite_component(residual_drift, "residual_drift")
    return V2EpochLoss(
        total=float(loss.total.detach()),
        positive=float(loss.positive.detach()),
        negative=float(loss.negative.detach()),
        ranking=float(loss.ranking.detach()),
        identity=float(loss.identity.detach()),
        residual_drift=float(residual_drift),
    )


def _require_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_model(model: LowRankDomainAdapter, expected_rank: int | None = None) -> None:
    if model.dimension <= 0 or model.rank <= 0:
        raise ValueError("model dimension and rank must be positive")
    if expected_rank is not None and model.rank != expected_rank:
        raise ValueError(f"model rank must be {expected_rank}")


def _model_device(model: LowRankDomainAdapter) -> torch.device:
    return next(model.parameters()).device


def _requested_device(device: str) -> torch.device:
    try:
        return torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device {device!r}") from exc


def _embedding_tensor(
    values: np.ndarray,
    *,
    dimension: int,
    description: str,
    device: str | torch.device,
) -> Tensor:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"{description} must have shape [N, {dimension}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{description} must be finite")
    return torch.from_numpy(array).to(device=device)


def _validate_pair_set_metadata(
    pair_set: V2PairSet,
    positive_count: int,
    negative_count: int,
) -> None:
    if positive_count != len(pair_set.positive_student_ids):
        raise ValueError("positive_student_ids must align with positive embeddings")
    if positive_count != len(pair_set.positive_session_ids):
        raise ValueError("positive_session_ids must align with positive embeddings")
    if negative_count != len(pair_set.negative_group_ids):
        raise ValueError("negative_group_ids must align with negative embeddings")
    if negative_count != len(pair_set.negative_categories):
        raise ValueError("negative_categories must align with negative embeddings")
    if negative_count != len(pair_set.negative_metadata):
        raise ValueError("negative_metadata must align with negative embeddings")


def _negative_weight_tensor(
    values: np.ndarray,
    *,
    expected: int,
    device: torch.device,
) -> Tensor:
    weights = np.asarray(values, dtype=np.float32).reshape(-1)
    if weights.shape[0] != expected:
        raise ValueError("negative_weights must align with negative embeddings")
    if not np.isfinite(weights).all():
        raise ValueError("negative_weights must be finite")
    if (weights < 0.0).any():
        raise ValueError("negative_weights must be non-negative")
    return torch.from_numpy(weights).to(device=device)


def _zero_like(model: LowRankDomainAdapter, *scores: Tensor) -> Tensor:
    for score in scores:
        if score.numel():
            return score.sum() * 0.0
    parameter = next(model.parameters())
    return parameter.sum() * 0.0


def _ranking_loss(
    *,
    positive_scores: Tensor,
    negative_scores: Tensor,
    pair_set: V2PairSet,
    config: V2TrainingConfig,
    zero: Tensor,
) -> tuple[Tensor, Tensor]:
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return zero, torch.empty((0,), dtype=torch.int64, device=zero.device)

    ranking_terms: list[Tensor] = []
    flattened_indices: list[int] = []
    for positive_index, (student_id, session_id) in enumerate(
        zip(pair_set.positive_student_ids, pair_set.positive_session_ids, strict=True)
    ):
        selected = _select_hard_negative_indices(
            negative_scores=negative_scores,
            negative_metadata=pair_set.negative_metadata,
            positive_student_id=student_id,
            positive_session_id=session_id,
            limit=config.hard_negatives_per_positive,
        )
        if not selected:
            continue
        index_tensor = torch.tensor(
            selected,
            dtype=torch.int64,
            device=negative_scores.device,
        )
        selected_scores = negative_scores.index_select(0, index_tensor)
        ranking_terms.append(
            F.relu(
                config.ranking_margin
                - positive_scores[positive_index]
                + selected_scores
            )
        )
        flattened_indices.extend(selected)
    if not ranking_terms:
        return zero, torch.empty((0,), dtype=torch.int64, device=zero.device)
    return (
        torch.cat(ranking_terms).mean(),
        torch.tensor(flattened_indices, dtype=torch.int64, device=negative_scores.device),
    )


def _select_hard_negative_indices(
    *,
    negative_scores: Tensor,
    negative_metadata: Sequence[PairMetadata],
    positive_student_id: str,
    positive_session_id: str,
    limit: int,
) -> list[int]:
    matches: list[tuple[float, tuple[str, str, str, str, int], int]] = []
    for index, metadata in enumerate(negative_metadata):
        if (
            metadata.ref_student_id != positive_student_id
            or metadata.ref_session_id != positive_session_id
        ):
            continue
        matches.append(
            (
                float(negative_scores[index].detach().cpu()),
                (
                    metadata.ref_student_id,
                    metadata.ref_session_id,
                    metadata.photo_student_id,
                    metadata.photo_session_id,
                    index,
                ),
                index,
            )
        )
    matches.sort(key=lambda item: (-item[0], item[1]))
    return [index for _score, _order, index in matches[:limit]]


def _require_finite_component(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite {name} component")
