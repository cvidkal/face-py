"""Deterministic v2 training objective for the identity-domain adapter."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from tools.domain_adapter_training import (
    LowRankDomainAdapter,
    PairMetadata,
    SessionEmbedding,
    extract_session_embeddings,
    manifest_training_rows,
    residual_weight_regularization,
    set_deterministic,
)
from tools.domain_adapter_v2_metrics import (
    CalibratedThreshold,
    V2PairSet,
    assign_training_folds,
    bootstrap_far_upper_bound,
    build_v2_pair_set,
    calibrate_threshold,
    student_balanced_match_recall,
)

_FOLD_COUNT = 5
_MIN_STUDENTS_PER_FOLD = 20
_MIN_MATCH_SESSIONS_PER_FOLD = 20
_MIN_VALIDATION_GROUPS = 1000
_MAX_FOLD_EMPIRICAL_FAR = 0.005
_MAX_POOLED_FAR_UPPER_95 = 0.01
_MIN_THRESHOLD = 0.35


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


@dataclass(frozen=True)
class FoldEpochMetrics:
    fold: int
    epoch: int
    candidate_threshold: float
    empirical_far: float
    student_balanced_recall: float
    validation_loss: float
    residual_drift: float
    negative_group_ids: tuple[str, ...] = field(repr=False)
    negative_accepts: tuple[bool, ...] = field(repr=False)


@dataclass(frozen=True)
class V2Selection:
    epoch: int
    median_recall: float
    worst_fold_recall: float
    pooled_far_upper_95: float
    residual_drift: float
    validation_loss: float


@dataclass
class TrainingTrace:
    trained_fold_students: set[str] = field(default_factory=set)
    selection_students: set[str] = field(default_factory=set)
    final_training_students: set[str] = field(default_factory=set)
    final_epoch_count: int = 0


@dataclass(frozen=True)
class V2TrainingResult:
    model: LowRankDomainAdapter
    selection: V2Selection
    fold_history: tuple[FoldEpochMetrics, ...]
    adapted_threshold: CalibratedThreshold
    raw_threshold: CalibratedThreshold
    dataset_digest: str


class InsufficientDataError(ValueError):
    """Raised when the canonical train/validation partitions are too small."""


class NoFeasibleEpochError(RuntimeError):
    """Raised when no epoch satisfies the out-of-fold safety constraints."""


def select_v2_epoch(
    history: Sequence[FoldEpochMetrics],
    dataset_digest: str,
) -> V2Selection:
    by_epoch: dict[int, dict[int, FoldEpochMetrics]] = {}
    for metric in history:
        fold_history = by_epoch.setdefault(metric.epoch, {})
        if metric.fold in fold_history:
            raise ValueError(
                f"duplicate fold history for epoch {metric.epoch}, fold {metric.fold}"
            )
        fold_history[metric.fold] = metric

    best_key: tuple[float, float, float, float, int] | None = None
    best_selection: V2Selection | None = None
    expected_folds = set(range(_FOLD_COUNT))
    for epoch in sorted(by_epoch):
        fold_history = by_epoch[epoch]
        if set(fold_history) != expected_folds:
            raise ValueError(f"epoch {epoch} is missing one or more fold metrics")
        ordered = tuple(fold_history[fold] for fold in range(_FOLD_COUNT))
        if any(metric.empirical_far > _MAX_FOLD_EMPIRICAL_FAR for metric in ordered):
            continue
        if any(metric.candidate_threshold < _MIN_THRESHOLD for metric in ordered):
            continue

        pooled_groups = tuple(
            group_id
            for metric in ordered
            for group_id in metric.negative_group_ids
        )
        pooled_accepts = tuple(
            accepted
            for metric in ordered
            for accepted in metric.negative_accepts
        )
        if not pooled_groups or len(pooled_groups) != len(pooled_accepts):
            raise ValueError(f"epoch {epoch} negative group data is incomplete")
        pooled_far = bootstrap_far_upper_bound(
            np.asarray(pooled_accepts, dtype=np.float64),
            pooled_groups,
            0.5,
            dataset_digest,
        )
        if pooled_far.upper_95 > _MAX_POOLED_FAR_UPPER_95:
            continue

        recalls = [metric.student_balanced_recall for metric in ordered]
        median_recall = float(np.median(np.asarray(recalls, dtype=np.float64)))
        worst_recall = float(min(recalls))
        mean_drift = float(
            np.mean([metric.residual_drift for metric in ordered], dtype=np.float64)
        )
        mean_validation_loss = float(
            np.mean([metric.validation_loss for metric in ordered], dtype=np.float64)
        )
        key = (
            median_recall,
            worst_recall,
            -mean_drift,
            -mean_validation_loss,
            -epoch,
        )
        if best_key is not None and key <= best_key:
            continue
        best_key = key
        best_selection = V2Selection(
            epoch=epoch,
            median_recall=median_recall,
            worst_fold_recall=worst_recall,
            pooled_far_upper_95=float(pooled_far.upper_95),
            residual_drift=mean_drift,
            validation_loss=mean_validation_loss,
        )

    if best_selection is None:
        raise NoFeasibleEpochError(
            "no feasible epoch satisfies all fold safety constraints"
        )
    return best_selection


def train_v2_candidate(
    manifest: dict[str, Any],
    cache_dir: Path,
    config: V2TrainingConfig,
    seed: int,
    device: str,
    trace: TrainingTrace | None = None,
) -> V2TrainingResult:
    canonical_manifest = _canonical_training_manifest(manifest)
    train_only_manifest = _train_only_manifest(canonical_manifest)
    selection_digest = _canonical_dataset_digest(train_only_manifest)
    dataset_digest = _canonical_dataset_digest(canonical_manifest)
    sessions = tuple(
        extract_session_embeddings(
            canonical_manifest,
            cache_path=Path(cache_dir) / ".embedding-cache.npz",
        )
    )
    train_sessions = tuple(
        session for session in sessions if session.split == "train"
    )
    validation_sessions = tuple(
        session for session in sessions if session.split == "validation"
    )
    folds = assign_training_folds(train_sessions)
    _validate_fold_sufficiency(folds)

    validation_pair_set = build_v2_pair_set(validation_sessions)
    _validate_partition_size("canonical validation", validation_sessions)
    if _synthetic_group_count(validation_pair_set) < _MIN_VALIDATION_GROUPS:
        raise InsufficientDataError(
            "canonical validation must contain at least 1000 ordered groups"
        )

    dimension = _embedding_dimension((*train_sessions, *validation_sessions))
    _reset_trace(trace)
    fold_history: list[FoldEpochMetrics] = []
    full_training_students = {session.student_id for session in train_sessions}
    for fold in range(_FOLD_COUNT):
        held_out_sessions = folds[fold]
        train_fold_sessions = tuple(
            session
            for other_fold, fold_sessions in folds.items()
            if other_fold != fold
            for session in fold_sessions
        )
        if trace is not None:
            trace.trained_fold_students.update(
                session.student_id for session in train_fold_sessions
            )
            trace.selection_students.update(
                session.student_id for session in held_out_sessions
            )
        train_pair_set = build_v2_pair_set(train_fold_sessions)
        held_out_pair_set = build_v2_pair_set(held_out_sessions)
        model = _fresh_model(dimension=dimension, rank=config.rank, seed=seed, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for epoch in range(1, config.max_epochs + 1):
            epoch_loss = train_fold_epoch(model, optimizer, train_pair_set, config)
            validation_loss = _score_validation_loss(model, held_out_pair_set, config)
            scores = score_pair_set(model, held_out_pair_set, device)
            threshold = _select_empirical_threshold(
                positive_scores=scores.positive,
                positive_student_ids=held_out_pair_set.positive_student_ids,
                negative_scores=scores.negative,
                negative_group_ids=held_out_pair_set.negative_group_ids,
            )
            fold_history.append(
                FoldEpochMetrics(
                    fold=fold,
                    epoch=epoch,
                    candidate_threshold=threshold.threshold,
                    empirical_far=threshold.empirical_far,
                    student_balanced_recall=threshold.student_balanced_recall,
                    validation_loss=validation_loss,
                    residual_drift=epoch_loss.residual_drift,
                    negative_group_ids=held_out_pair_set.negative_group_ids,
                    negative_accepts=tuple(
                        bool(score >= threshold.threshold) for score in scores.negative
                    ),
                )
            )

    selection = select_v2_epoch(fold_history, selection_digest)
    full_training_pair_set = build_v2_pair_set(train_sessions)
    final_model = _fresh_model(dimension=dimension, rank=config.rank, seed=seed, device=device)
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=config.learning_rate,
    )
    if trace is not None:
        trace.final_training_students = set(full_training_students)
        trace.final_epoch_count = selection.epoch
    for _epoch in range(1, selection.epoch + 1):
        train_fold_epoch(final_model, final_optimizer, full_training_pair_set, config)

    adapted_scores = score_pair_set(final_model, validation_pair_set, device)
    adapted_threshold = calibrate_threshold(
        positive_scores=adapted_scores.positive,
        positive_student_ids=validation_pair_set.positive_student_ids,
        negative_scores=adapted_scores.negative,
        negative_group_ids=validation_pair_set.negative_group_ids,
        dataset_digest=dataset_digest,
    )
    raw_scores = _raw_pair_scores(validation_pair_set)
    raw_threshold = calibrate_threshold(
        positive_scores=raw_scores.positive,
        positive_student_ids=validation_pair_set.positive_student_ids,
        negative_scores=raw_scores.negative,
        negative_group_ids=validation_pair_set.negative_group_ids,
        dataset_digest=dataset_digest,
    )
    return V2TrainingResult(
        model=final_model,
        selection=selection,
        fold_history=tuple(fold_history),
        adapted_threshold=adapted_threshold,
        raw_threshold=raw_threshold,
        dataset_digest=dataset_digest,
    )


def score_pair_set(
    model: LowRankDomainAdapter,
    pair_set: V2PairSet,
    device: str,
) -> V2Scores:
    _validate_model(model)
    model_device = _model_device(model)
    requested_device = _requested_device(device)
    if _canonical_device_key(requested_device) != _canonical_device_key(model_device):
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


def _canonical_training_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    canonical = {
        "schema_version": manifest.get("schema_version"),
        "snapshot": manifest.get("snapshot", ""),
        "split_seed": manifest.get("split_seed", ""),
        "sessions": manifest.get("sessions", []),
        "evaluation_sessions": [],
    }
    rows = [
        dict(row)
        for row in manifest_training_rows(canonical)
        if row["split"] in {"train", "validation"}
    ]
    return {
        "schema_version": 1,
        "snapshot": canonical["snapshot"],
        "split_seed": canonical["split_seed"],
        "sessions": rows,
        "evaluation_sessions": [],
    }


def _train_only_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "snapshot": manifest.get("snapshot", ""),
        "split_seed": manifest.get("split_seed", ""),
        "sessions": [
            dict(row)
            for row in manifest.get("sessions", [])
            if row.get("split") == "train"
        ],
        "evaluation_sessions": [],
    }


def _canonical_dataset_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_fold_sufficiency(
    folds: dict[int, tuple[SessionEmbedding, ...]],
) -> None:
    for fold in range(_FOLD_COUNT):
        _validate_partition_size(f"training fold {fold}", folds.get(fold, ()))


def _validate_partition_size(
    name: str,
    sessions: Sequence[SessionEmbedding],
) -> None:
    student_count = len({session.student_id for session in sessions})
    match_count = sum(1 for session in sessions if session.label == "match")
    if student_count < _MIN_STUDENTS_PER_FOLD or match_count < _MIN_MATCH_SESSIONS_PER_FOLD:
        raise InsufficientDataError(
            f"{name} must contain at least 20 students and 20 truth-match sessions"
        )


def _embedding_dimension(sessions: Sequence[SessionEmbedding]) -> int:
    if not sessions:
        raise InsufficientDataError("canonical data produced no train or validation sessions")
    return int(np.asarray(sessions[0].ref_embedding).reshape(-1).size)


def _synthetic_group_count(pair_set: V2PairSet) -> int:
    return len(
        {
            group_id
            for group_id, category in zip(
                pair_set.negative_group_ids,
                pair_set.negative_categories,
                strict=True,
            )
            if category == "synthetic_cross_student"
        }
    )


def _select_empirical_threshold(
    *,
    positive_scores: np.ndarray,
    positive_student_ids: Sequence[str],
    negative_scores: np.ndarray,
    negative_group_ids: Sequence[str],
    threshold_floor: float = _MIN_THRESHOLD,
    max_empirical_far: float = _MAX_FOLD_EMPIRICAL_FAR,
) -> CalibratedThreshold:
    positives = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negatives = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if positives.size != len(positive_student_ids):
        raise ValueError("positive_scores and positive_student_ids must have the same length")
    if negatives.size != len(negative_group_ids):
        raise ValueError("negative_scores and negative_group_ids must have the same length")
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("positive_scores and negative_scores must not be empty")

    best: CalibratedThreshold | None = None
    for threshold in np.round(np.arange(threshold_floor, 1.001, 0.001), 3):
        empirical_far = float(np.mean(negatives >= threshold))
        recall = student_balanced_match_recall(positives, positive_student_ids, threshold)
        candidate = CalibratedThreshold(
            threshold=float(threshold),
            empirical_far=empirical_far,
            far_upper_95=empirical_far,
            student_balanced_recall=recall,
            true_matches=int(np.count_nonzero(positives >= threshold)),
            feasible=empirical_far <= max_empirical_far,
        )
        if not candidate.feasible:
            continue
        if best is None:
            best = candidate
            continue
        key = (
            candidate.student_balanced_recall,
            -candidate.empirical_far,
            candidate.threshold,
        )
        best_key = (
            best.student_balanced_recall,
            -best.empirical_far,
            best.threshold,
        )
        if key > best_key:
            best = candidate

    if best is not None:
        return best
    return CalibratedThreshold(
        threshold=threshold_floor,
        empirical_far=float(np.mean(negatives >= threshold_floor)),
        far_upper_95=float(np.mean(negatives >= threshold_floor)),
        student_balanced_recall=student_balanced_match_recall(
            positives,
            positive_student_ids,
            threshold_floor,
        ),
        true_matches=int(np.count_nonzero(positives >= threshold_floor)),
        feasible=False,
    )


def _fresh_model(
    *,
    dimension: int,
    rank: int,
    seed: int,
    device: str,
) -> LowRankDomainAdapter:
    set_deterministic(seed)
    model = LowRankDomainAdapter(dimension=dimension, rank=rank)
    return model.to(device=_requested_device(device))


def _score_validation_loss(
    model: LowRankDomainAdapter,
    pair_set: V2PairSet,
    config: V2TrainingConfig,
) -> float:
    previous_mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            loss = v2_separation_loss(model, pair_set, config)
    finally:
        model.train(previous_mode)
    return float(loss.total.detach().cpu())


def _raw_pair_scores(pair_set: V2PairSet) -> V2Scores:
    positive = np.sum(
        np.asarray(pair_set.positive_ref_embeddings, dtype=np.float64)
        * np.asarray(pair_set.positive_photo_embeddings, dtype=np.float64),
        axis=1,
    )
    negative = np.sum(
        np.asarray(pair_set.negative_ref_embeddings, dtype=np.float64)
        * np.asarray(pair_set.negative_photo_embeddings, dtype=np.float64),
        axis=1,
    )
    return V2Scores(positive=positive, negative=negative)


def _reset_trace(trace: TrainingTrace | None) -> None:
    if trace is None:
        return
    trace.trained_fold_students.clear()
    trace.selection_students.clear()
    trace.final_training_students.clear()
    trace.final_epoch_count = 0


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


def _canonical_device_key(device: torch.device) -> tuple[str, int | None]:
    if device.type == "cpu":
        return ("cpu", None)
    if device.type == "cuda":
        return ("cuda", _resolved_cuda_index(device))
    return (device.type, device.index)


def _resolved_cuda_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    try:
        return int(torch.cuda.current_device())
    except (AssertionError, RuntimeError) as exc:
        raise ValueError("cuda device index is unavailable") from exc


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
