"""Tail-focused separation loss for identity domain adapter v3."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from tools.domain_adapter_training import (
    LowRankDomainAdapter,
    PairMetadata,
    extract_session_embeddings,
    residual_weight_regularization,
)
from tools.domain_adapter_v2_metrics import (
    CalibratedThreshold,
    V2PairSet,
    _iter_bootstrap_sample_counts,
    _top_k_higher_quantile_values,
    assign_training_folds,
    bootstrap_seed,
    build_v2_pair_set,
    calibrate_threshold,
    compare_at_far_budget,
    student_balanced_match_recall,
)
from tools.domain_adapter_v2_training import (
    FoldEpochMetrics,
    InsufficientDataError,
    NoFeasibleEpochError,
    V2Selection,
    _canonical_dataset_digest,
    _canonical_training_manifest,
    _embedding_tensor,
    _embedding_dimension,
    _export_v2_onnx_with_dynamic_batch_parity,
    _fresh_model,
    _model_device,
    _preserve_failure_evidence,
    _public_fold_summaries,
    _publish_directory_no_replace,
    _raw_pair_scores,
    _require_finite_component,
    _select_empirical_threshold,
    _source_revision,
    _split_counts,
    _synthetic_group_count,
    _train_only_manifest,
    _validate_fold_sufficiency,
    _validate_model,
    _validate_partition_size,
    _validate_pair_set_metadata,
    _zero_like,
    score_pair_set,
)
from tools.domain_adapter_v3_metrics import (
    HistoricalRelativeMetrics,
    group_tail_indices,
    historical_relative_gate_passes,
    student_balanced_weights,
)
from tools.domain_adapter_training import _atomic_write_private_json

_T = TypeVar("_T")


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


@dataclass(frozen=True)
class OofScores:
    raw_positive: np.ndarray
    adapted_positive: np.ndarray
    positive_student_ids: tuple[str, ...]
    raw_negative: np.ndarray
    adapted_negative: np.ndarray
    negative_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class V3TrainingResult:
    model: LowRankDomainAdapter
    selection: V2Selection
    fold_history: tuple[FoldEpochMetrics, ...]
    adapted_threshold: CalibratedThreshold
    raw_threshold: CalibratedThreshold
    historical_metrics: HistoricalRelativeMetrics
    dataset_digest: str
    source_feedback_snapshot: str
    split_seed: str
    split_counts: dict[str, dict[str, int]]
    validation_negative_category_counts: dict[str, int]
    validation_negative_category_weight_totals: dict[str, float]


class HistoricalGateError(RuntimeError):
    def __init__(self, metrics: HistoricalRelativeMetrics) -> None:
        super().__init__("historical out-of-fold relative gate failed")
        self.metrics = metrics


def select_v3_epoch(
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

    expected_folds = set(range(5))
    candidates: list[tuple[int, tuple[FoldEpochMetrics, ...]]] = []
    pooled_group_ids: tuple[str, ...] | None = None
    pooled_accepts: list[tuple[bool, ...]] = []
    for epoch in sorted(by_epoch):
        fold_history = by_epoch[epoch]
        if set(fold_history) != expected_folds:
            raise ValueError(f"epoch {epoch} is missing one or more fold metrics")
        ordered = tuple(fold_history[fold] for fold in range(5))
        if any(metric.empirical_far > 0.005 for metric in ordered):
            continue
        if any(metric.candidate_threshold < 0.35 for metric in ordered):
            continue
        group_ids = tuple(
            group_id for metric in ordered for group_id in metric.negative_group_ids
        )
        accepts = tuple(
            accepted for metric in ordered for accepted in metric.negative_accepts
        )
        if not group_ids or len(group_ids) != len(accepts):
            raise ValueError(f"epoch {epoch} negative group data is incomplete")
        if any(not str(group_id) for group_id in group_ids):
            raise ValueError(f"epoch {epoch} negative group IDs must be non-empty")
        if pooled_group_ids is None:
            pooled_group_ids = group_ids
        elif group_ids != pooled_group_ids:
            raise ValueError("negative group order must remain fixed across epochs")
        candidates.append((epoch, ordered))
        pooled_accepts.append(accepts)

    if not candidates or pooled_group_ids is None:
        raise NoFeasibleEpochError(
            "no feasible epoch satisfies all fold safety constraints"
        )

    group_order = tuple(dict.fromkeys(pooled_group_ids))
    group_indices = {group_id: index for index, group_id in enumerate(group_order)}
    group_sizes = np.zeros(len(group_order), dtype=np.int64)
    for group_id in pooled_group_ids:
        group_sizes[group_indices[group_id]] += 1
    accept_counts = np.zeros(
        (len(group_order), len(candidates)),
        dtype=np.int64,
    )
    for candidate_index, accepts in enumerate(pooled_accepts):
        for group_id, accepted in zip(pooled_group_ids, accepts, strict=True):
            accept_counts[group_indices[group_id], candidate_index] += int(accepted)

    iterations = 10000
    retain_count = iterations - int(math.ceil((iterations - 1) * 0.95))
    top_values = np.full(
        (retain_count, len(candidates)),
        -np.inf,
        dtype=np.float64,
    )
    active_accept_groups = np.any(accept_counts != 0, axis=1)
    active_accept_counts = accept_counts[active_accept_groups]
    generator = np.random.default_rng(bootstrap_seed(dataset_digest))
    for sample_counts in _iter_bootstrap_sample_counts(
        generator,
        group_count=len(group_order),
        iterations=iterations,
        batch_size=128,
    ):
        sampled_sizes = sample_counts @ group_sizes
        if active_accept_counts.shape[0]:
            sampled_accepts = (
                sample_counts[:, active_accept_groups] @ active_accept_counts
            )
        else:
            sampled_accepts = np.zeros(
                (sample_counts.shape[0], len(candidates)),
                dtype=np.int64,
            )
        replicates = sampled_accepts / sampled_sizes[:, None]
        top_values[...] = _top_k_higher_quantile_values(
            top_values,
            replicates,
            retain_count,
        )
    upper_95 = np.min(top_values, axis=0)

    best_key: tuple[float, float, float, float, int] | None = None
    best_selection: V2Selection | None = None
    for candidate_index, (epoch, ordered) in enumerate(candidates):
        pooled_far_upper_95 = float(upper_95[candidate_index])
        if pooled_far_upper_95 > 0.01:
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
            pooled_far_upper_95=pooled_far_upper_95,
            residual_drift=mean_drift,
            validation_loss=mean_validation_loss,
        )
    if best_selection is None:
        raise NoFeasibleEpochError(
            "no feasible epoch satisfies all fold safety constraints"
        )
    return best_selection


def train_v3_candidate(
    manifest: dict[str, Any],
    embedding_cache: Path,
    config: V3TrainingConfig,
    seed: int,
    device: str,
) -> V3TrainingResult:
    canonical_manifest = _canonical_training_manifest(manifest)
    train_only_manifest = _train_only_manifest(canonical_manifest)
    selection_digest = _canonical_dataset_digest(train_only_manifest)
    dataset_digest = _canonical_dataset_digest(canonical_manifest)
    sessions = tuple(
        extract_session_embeddings(
            canonical_manifest,
            cache_path=Path(embedding_cache),
        )
    )
    train_sessions = tuple(session for session in sessions if session.split == "train")
    validation_sessions = tuple(
        session for session in sessions if session.split == "validation"
    )
    folds = assign_training_folds(train_sessions)
    _validate_fold_sufficiency(folds)

    validation_pair_set = build_v2_pair_set(validation_sessions)
    _validate_partition_size("canonical validation", validation_sessions)
    if _synthetic_group_count(validation_pair_set) < 1000:
        raise InsufficientDataError(
            "canonical validation must contain at least 1000 ordered groups"
        )

    dimension = _embedding_dimension((*train_sessions, *validation_sessions))
    fold_history: list[FoldEpochMetrics] = []
    checkpoints: dict[tuple[int, int], dict[str, Tensor]] = {}
    held_out_pair_sets: dict[int, V2PairSet] = {}
    for fold in range(5):
        held_out_sessions = folds[fold]
        train_fold_sessions = tuple(
            session
            for other_fold, fold_sessions in folds.items()
            if other_fold != fold
            for session in fold_sessions
        )
        train_pair_set = build_v2_pair_set(train_fold_sessions)
        held_out_pair_set = build_v2_pair_set(held_out_sessions)
        held_out_pair_sets[fold] = held_out_pair_set
        model = _fresh_model(
            dimension=dimension,
            rank=config.rank,
            seed=seed,
            device=device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for epoch in range(1, config.max_epochs + 1):
            epoch_loss = train_v3_epoch(model, optimizer, train_pair_set, config)
            validation_loss = _score_v3_validation_loss(
                model,
                held_out_pair_set,
                config,
            )
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
                    student_count=len(
                        {session.student_id for session in held_out_sessions}
                    ),
                    session_count=len(held_out_sessions),
                    match_session_count=sum(
                        1 for session in held_out_sessions if session.label == "match"
                    ),
                    negative_group_count=len(set(held_out_pair_set.negative_group_ids)),
                )
            )
            checkpoints[(fold, epoch)] = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    selection = select_v3_epoch(fold_history, selection_digest)
    oof_scores = _selected_oof_scores(
        held_out_pair_sets,
        checkpoints,
        selection=selection,
        dimension=dimension,
        rank=config.rank,
        seed=seed,
        device=device,
    )

    def finish(historical_metrics: HistoricalRelativeMetrics) -> V3TrainingResult:
        full_training_pair_set = build_v2_pair_set(train_sessions)
        final_model = _fresh_model(
            dimension=dimension,
            rank=config.rank,
            seed=seed,
            device=device,
        )
        final_optimizer = torch.optim.Adam(
            final_model.parameters(),
            lr=config.learning_rate,
        )
        for _epoch in range(1, selection.epoch + 1):
            train_v3_epoch(final_model, final_optimizer, full_training_pair_set, config)

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
        return V3TrainingResult(
            model=final_model,
            selection=selection,
            fold_history=tuple(fold_history),
            adapted_threshold=adapted_threshold,
            raw_threshold=raw_threshold,
            historical_metrics=historical_metrics,
            dataset_digest=dataset_digest,
            source_feedback_snapshot=str(canonical_manifest.get("snapshot", "")),
            split_seed=str(canonical_manifest.get("split_seed", "")),
            split_counts=_split_counts(
                canonical_manifest,
                train_pairs=full_training_pair_set,
                validation_pairs=validation_pair_set,
            ),
            validation_negative_category_counts=(
                validation_pair_set.negative_category_counts
            ),
            validation_negative_category_weight_totals=(
                validation_pair_set.negative_category_weight_totals
            ),
        )

    return continue_after_historical_oof(
        oof_scores,
        dataset_digest=selection_digest,
        continue_fn=finish,
    )


def require_historical_gate(metrics: HistoricalRelativeMetrics) -> None:
    if not historical_relative_gate_passes(metrics):
        raise HistoricalGateError(metrics)


def historical_metrics_from_oof(
    scores: OofScores,
    *,
    dataset_digest: str,
) -> HistoricalRelativeMetrics:
    comparison = compare_at_far_budget(
        raw_positive_scores=scores.raw_positive,
        adapted_positive_scores=scores.adapted_positive,
        positive_student_ids=scores.positive_student_ids,
        raw_negative_scores=scores.raw_negative,
        adapted_negative_scores=scores.adapted_negative,
        negative_group_ids=scores.negative_group_ids,
        dataset_digest=dataset_digest,
    )
    adapted_threshold, adapted_far, adapted_recall, adapted_true_matches = (
        _select_adapted_at_empirical_far(
            positive_scores=scores.adapted_positive,
            positive_student_ids=scores.positive_student_ids,
            negative_scores=scores.adapted_negative,
            far_ceiling=comparison.raw.empirical_far,
        )
    )
    return HistoricalRelativeMetrics(
        raw_far=comparison.raw.empirical_far,
        adapted_far=adapted_far,
        recall_lift=adapted_recall - comparison.raw.student_balanced_recall,
        true_match_delta=adapted_true_matches - comparison.raw.true_matches,
        same_threshold_recall_delta=(
            comparison.same_threshold_adapted_recall
            - comparison.same_threshold_raw_recall
        ),
        adapted_threshold=adapted_threshold,
        raw_threshold=comparison.raw.threshold,
    )


def _select_adapted_at_empirical_far(
    *,
    positive_scores: np.ndarray,
    positive_student_ids: Sequence[str],
    negative_scores: np.ndarray,
    far_ceiling: float,
) -> tuple[float, float, float, int]:
    positives = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negatives = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if positives.size != len(positive_student_ids):
        raise ValueError("positive scores and student IDs must align")
    if not positives.size or not negatives.size:
        raise ValueError("positive and negative scores must not be empty")
    best: tuple[tuple[float, float, float], float, float, float, int] | None = None
    for threshold in np.round(np.arange(0.35, 1.001, 0.001), 3):
        far = float(np.mean(negatives >= threshold))
        recall = student_balanced_match_recall(
            positives,
            positive_student_ids,
            float(threshold),
        )
        true_matches = int(np.count_nonzero(positives >= threshold))
        key = (recall, -far, float(threshold))
        candidate = (key, float(threshold), far, recall, true_matches)
        if far <= far_ceiling and (best is None or key > best[0]):
            best = candidate
    if best is not None:
        return best[1:]

    threshold = 1.0
    far = float(np.mean(negatives >= threshold))
    recall = student_balanced_match_recall(
        positives,
        positive_student_ids,
        threshold,
    )
    return threshold, far, recall, int(np.count_nonzero(positives >= threshold))


def continue_after_historical_oof(
    scores: OofScores,
    *,
    dataset_digest: str,
    continue_fn: Callable[[HistoricalRelativeMetrics], _T],
) -> _T:
    metrics = historical_metrics_from_oof(scores, dataset_digest=dataset_digest)
    require_historical_gate(metrics)
    return continue_fn(metrics)


def _selected_oof_scores(
    pair_sets: dict[int, V2PairSet],
    checkpoints: dict[tuple[int, int], dict[str, Tensor]],
    *,
    selection: V2Selection,
    dimension: int,
    rank: int,
    seed: int,
    device: str,
) -> OofScores:
    raw_positive: list[np.ndarray] = []
    adapted_positive: list[np.ndarray] = []
    positive_student_ids: list[str] = []
    raw_negative: list[np.ndarray] = []
    adapted_negative: list[np.ndarray] = []
    negative_group_ids: list[str] = []
    for fold in range(5):
        pair_set = pair_sets[fold]
        try:
            checkpoint = checkpoints[(fold, selection.epoch)]
        except KeyError as exc:
            raise ValueError(
                f"selected epoch {selection.epoch} has no checkpoint for fold {fold}"
            ) from exc
        model = _fresh_model(
            dimension=dimension,
            rank=rank,
            seed=seed,
            device=device,
        )
        model.load_state_dict(checkpoint)
        adapted = score_pair_set(model, pair_set, device)
        raw = _raw_pair_scores(pair_set)
        raw_positive.append(raw.positive)
        adapted_positive.append(adapted.positive)
        positive_student_ids.extend(pair_set.positive_student_ids)
        raw_negative.append(raw.negative)
        adapted_negative.append(adapted.negative)
        negative_group_ids.extend(pair_set.negative_group_ids)
    return OofScores(
        raw_positive=np.concatenate(raw_positive),
        adapted_positive=np.concatenate(adapted_positive),
        positive_student_ids=tuple(positive_student_ids),
        raw_negative=np.concatenate(raw_negative),
        adapted_negative=np.concatenate(adapted_negative),
        negative_group_ids=tuple(negative_group_ids),
    )


def _score_v3_validation_loss(
    model: LowRankDomainAdapter,
    pair_set: V2PairSet,
    config: V3TrainingConfig,
) -> float:
    previous_mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            loss = v3_separation_loss(model, pair_set, config)
    finally:
        model.train(previous_mode)
    return float(loss.total.detach().cpu())


def write_v3_candidate_artifacts(
    output_dir: Path,
    result: V3TrainingResult,
    seed: int,
    *,
    _before_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    os.chmod(temporary_dir, 0o700)
    try:
        onnx_path = temporary_dir / "identity_domain_adapter.onnx"
        parity_error = _export_v2_onnx_with_dynamic_batch_parity(result.model, onnx_path)
        manifest = _v3_runtime_manifest(
            result=result,
            seed=seed,
            onnx_file=onnx_path.name,
            onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            onnx_parity_max_abs_error=parity_error,
        )
        _atomic_write_private_json(
            temporary_dir / "identity_domain_adapter.manifest.json",
            manifest,
        )
        if _before_publish is not None:
            _before_publish()
        _publish_directory_no_replace(temporary_dir, output_dir)
        os.chmod(output_dir, 0o700)
        for path in output_dir.iterdir():
            os.chmod(path, 0o600)
        return manifest
    except Exception as exc:
        _preserve_failure_evidence(temporary_dir, exc)
        raise


def _v3_runtime_manifest(
    *,
    result: V3TrainingResult,
    seed: int,
    onnx_file: str,
    onnx_sha256: str,
    onnx_parity_max_abs_error: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_version": "identity-domain-adapter-v3",
        "embedding_dimension": result.model.dimension,
        "rank": result.model.rank,
        "match_threshold": result.adapted_threshold.threshold,
        "onnx_file": onnx_file,
        "onnx_sha256": onnx_sha256,
        "onnx_parity_max_abs_error": onnx_parity_max_abs_error,
        "source_dataset_sha256": result.dataset_digest,
        "source_code_revision": _source_revision(),
        "split_seed": result.split_seed,
        "split_counts": result.split_counts,
        "training_hyperparameters": {"seed": seed},
        "validation_metrics": asdict(result.adapted_threshold),
        "test_metrics": None,
        "input_names": ["ref_embedding", "photo_embedding"],
        "output_name": "adapted_cosine",
        "training": {
            "strategy": "five_fold_student_oof_v3",
            "fold_schema_version": 1,
            "fold_seed": "identity-domain-adapter-v2-folds",
            "fold_count": 5,
            "fold_counts": _public_fold_summaries(
                result.fold_history,
                selected_epoch=result.selection.epoch,
            ),
            "selected_epoch": result.selection.epoch,
            "selection_key": [
                "median_recall",
                "worst_fold_recall",
                "residual_drift",
                "validation_loss",
                "earliest_epoch",
            ],
            "loss": asdict(V3TrainingConfig()),
            "positive_weighting": "equal_total_weight_per_student",
            "negative_construction": "exhaustive_cross_student_ordered_pairs",
            "negative_weighting": "ordered_student_pair_current_maximum",
            "ranking_weighting": "equal_reference_student_distinct_photo_students",
            "negative_categories": {
                "counts": dict(sorted(result.validation_negative_category_counts.items())),
                "weights": dict(
                    sorted(result.validation_negative_category_weight_totals.items())
                ),
            },
            "historical_oof_relative": asdict(result.historical_metrics),
            "bootstrap": {
                "iterations": 10000,
                "seed": bootstrap_seed(result.dataset_digest),
                "quantile_method": "higher",
            },
            "threshold_floor": 0.35,
            "adapted_threshold": asdict(result.adapted_threshold),
            "raw_comparator_threshold": asdict(result.raw_threshold),
            "canonical_dataset_digest": result.dataset_digest,
        },
    }


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
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not torch.isfinite(negatives).all():
        raise ValueError("negative scores must be finite")
    if not positives.numel() or not negatives.numel():
        empty = torch.empty((0,), dtype=torch.int64, device=positives.device)
        base = positives.sum() * 0.0 + negatives.sum() * 0.0
        return (base if zero is None else zero), empty

    score_values = negatives.detach().cpu().numpy()
    best_by_positive: dict[
        tuple[str, str],
        dict[str, tuple[float, tuple[str, str, str, str, int], int]],
    ] = defaultdict(dict)
    for index, row in enumerate(negative_metadata):
        key = (row.ref_student_id, row.ref_session_id)
        order = (
            row.ref_student_id,
            row.ref_session_id,
            row.photo_student_id,
            row.photo_session_id,
            index,
        )
        candidate = (float(score_values[index]), order, index)
        previous = best_by_positive[key].get(row.photo_student_id)
        if previous is None or candidate[0] > previous[0] or (
            candidate[0] == previous[0] and candidate[1] < previous[1]
        ):
            best_by_positive[key][row.photo_student_id] = candidate
    selected_by_positive = {
        key: [
            item[2]
            for item in sorted(
                candidates.values(),
                key=lambda item: (-item[0], item[1]),
            )[:limit]
        ]
        for key, candidates in best_by_positive.items()
    }

    by_student: dict[str, list[Tensor]] = defaultdict(list)
    flattened_indices: list[int] = []
    for positive_index, (student_id, session_id) in enumerate(
        zip(positive_student_ids, positive_session_ids, strict=True)
    ):
        selected = selected_by_positive.get((student_id, session_id), [])
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
