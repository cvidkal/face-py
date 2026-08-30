"""Offline v2 folds, grouped pair construction, and safety metrics."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tools.domain_adapter_training import PairMetadata, SessionEmbedding

_FOLD_COUNT = 5
_FOLD_SEED = "identity-domain-adapter-v2-folds"
_BOOTSTRAP_NAMESPACE = "identity-domain-adapter-v2-bootstrap"
_GROUP_NAMESPACE = "identity-domain-adapter-v2-group"
_BOOTSTRAP_DRAW_BATCH_SIZE = 128
_SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
_NEGATIVE_CATEGORY = Literal["synthetic_cross_student", "human_mismatch"]
_PAIR_CATEGORY_ORDER = {"synthetic_cross_student": 0, "human_mismatch": 1}


@dataclass(frozen=True)
class V2PairSet:
    positive_ref_embeddings: np.ndarray
    positive_photo_embeddings: np.ndarray
    positive_student_ids: tuple[str, ...]
    positive_session_ids: tuple[str, ...]
    negative_ref_embeddings: np.ndarray
    negative_photo_embeddings: np.ndarray
    negative_group_ids: tuple[str, ...]
    negative_categories: tuple[_NEGATIVE_CATEGORY, ...]
    negative_weights: np.ndarray
    negative_metadata: tuple[PairMetadata, ...]

    @property
    def negative_category_counts(self) -> dict[str, int]:
        return dict(Counter(self.negative_categories))

    @property
    def negative_category_weight_totals(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for category, weight in zip(
            self.negative_categories, self.negative_weights, strict=True
        ):
            totals[category] += float(weight)
        return dict(totals)


@dataclass(frozen=True)
class HumanMismatchEvidence:
    student_id: str
    session_id: str
    ordered_pair_token: str
    photo_student_token: str
    photo_session_token: str


@dataclass(frozen=True)
class BootstrapFar:
    estimate: float
    upper_95: float
    iterations: int
    seed: int
    group_count: int


@dataclass(frozen=True)
class CalibratedThreshold:
    threshold: float
    empirical_far: float
    far_upper_95: float
    student_balanced_recall: float
    true_matches: int
    feasible: bool


@dataclass(frozen=True)
class BudgetComparison:
    raw: CalibratedThreshold
    adapted: CalibratedThreshold
    recall_lift: float
    true_match_delta: int
    far_delta: float
    same_threshold_raw_recall: float
    same_threshold_adapted_recall: float


@dataclass(frozen=True)
class RelativeGateMetrics:
    raw_frozen_far: float
    adapted_frozen_far: float
    budget: BudgetComparison
    same_threshold_recall_delta: float


def student_fold(student_id: str, seed: str = _FOLD_SEED) -> int:
    digest = hashlib.sha256(f"{seed}\0{student_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % _FOLD_COUNT


def assign_training_folds(
    sessions: Sequence[SessionEmbedding],
) -> dict[int, tuple[SessionEmbedding, ...]]:
    folds: dict[int, list[SessionEmbedding]] = {index: [] for index in range(_FOLD_COUNT)}
    for session in sorted(sessions, key=_session_sort_key):
        folds[student_fold(session.student_id)].append(session)
    return {index: tuple(items) for index, items in folds.items()}


def _build_v2_pair_set(
    sessions: Sequence[SessionEmbedding],
    *,
    human_mismatch_evidence: Sequence[HumanMismatchEvidence],
) -> V2PairSet:
    ordered = tuple(sorted(sessions, key=_session_sort_key))
    dimension = _embedding_dimension(ordered)
    positives: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    negative_rows: list[
        tuple[
            tuple[object, ...],
            np.ndarray,
            np.ndarray,
            str,
            _NEGATIVE_CATEGORY,
            PairMetadata,
        ]
    ] = []
    mismatch_evidence = _human_mismatch_evidence_by_session(human_mismatch_evidence)
    used_evidence_keys: set[tuple[str, str]] = set()

    by_student: dict[str, list[SessionEmbedding]] = defaultdict(list)
    for session in ordered:
        ref = _normalized_embedding(session.ref_embedding, description="ref_embedding")
        photo = _normalized_embedding(
            session.session_prototype, description="session_prototype"
        )
        if ref.size != dimension or photo.size != dimension:
            raise ValueError("all embeddings must have one consistent dimension")
        if session.label == "match":
            positives.append((ref, photo, session.student_id, session.session_id))
            by_student[session.student_id].append(session)
        elif session.label == "mismatch":
            evidence_key = (session.student_id, session.session_id)
            evidence = mismatch_evidence.get(evidence_key)
            if evidence is None:
                raise ValueError(
                    "human mismatch evidence is required for "
                    f"{session.student_id}/{session.session_id}"
                )
            used_evidence_keys.add(evidence_key)
            metadata = PairMetadata(
                ref_student_id=session.student_id,
                ref_session_id=session.session_id,
                photo_student_id=evidence.photo_student_token,
                photo_session_id=evidence.photo_session_token,
                label=0,
                raw_cosine=float(np.dot(ref, photo)),
            )
            negative_rows.append(
                (
                    (
                        session.student_id,
                        session.session_id,
                        evidence.photo_student_token,
                        evidence.photo_session_token,
                        _PAIR_CATEGORY_ORDER["human_mismatch"],
                    ),
                    ref,
                    photo,
                    evidence.ordered_pair_token,
                    "human_mismatch",
                    metadata,
                )
            )
        else:
            raise ValueError(f"invalid session label: {session.label!r}")
    unused_keys = set(mismatch_evidence) - used_evidence_keys
    if unused_keys:
        student_id, session_id = min(unused_keys)
        raise ValueError(
            "human mismatch evidence does not match any mismatch session: "
            f"{student_id}/{session_id}"
        )

    students = sorted(by_student)
    for ref_student in students:
        for photo_student in students:
            if ref_student == photo_student:
                continue
            for ref_session in by_student[ref_student]:
                ref = _normalized_embedding(
                    ref_session.ref_embedding, description="ref_embedding"
                )
                for photo_session in by_student[photo_student]:
                    photo = _normalized_embedding(
                        photo_session.session_prototype,
                        description="session_prototype",
                    )
                    metadata = PairMetadata(
                        ref_student_id=ref_session.student_id,
                        ref_session_id=ref_session.session_id,
                        photo_student_id=photo_session.student_id,
                        photo_session_id=photo_session.session_id,
                        label=0,
                        raw_cosine=float(np.dot(ref, photo)),
                    )
                    negative_rows.append(
                        (
                            (
                                ref_session.student_id,
                                ref_session.session_id,
                                photo_session.student_id,
                                photo_session.session_id,
                                _PAIR_CATEGORY_ORDER["synthetic_cross_student"],
                            ),
                            ref,
                            photo,
                            _group_id(
                                "synthetic_cross_student",
                                ref_session.student_id,
                                photo_session.student_id,
                            ),
                            "synthetic_cross_student",
                            metadata,
                        )
                    )

    negative_rows.sort(key=lambda item: item[0])
    negative_group_ids = tuple(item[3] for item in negative_rows)
    negative_categories = tuple(item[4] for item in negative_rows)
    negative_metadata = tuple(item[5] for item in negative_rows)
    _validate_group_metadata(negative_group_ids, negative_categories, negative_metadata)

    return V2PairSet(
        positive_ref_embeddings=_stack_or_empty(
            [item[0] for item in positives], dimension=dimension
        ),
        positive_photo_embeddings=_stack_or_empty(
            [item[1] for item in positives], dimension=dimension
        ),
        positive_student_ids=tuple(item[2] for item in positives),
        positive_session_ids=tuple(item[3] for item in positives),
        negative_ref_embeddings=_stack_or_empty(
            [item[1] for item in negative_rows], dimension=dimension
        ),
        negative_photo_embeddings=_stack_or_empty(
            [item[2] for item in negative_rows], dimension=dimension
        ),
        negative_group_ids=negative_group_ids,
        negative_categories=negative_categories,
        negative_weights=_weights_for_groups(negative_group_ids),
        negative_metadata=negative_metadata,
    )


def build_v2_pair_set(
    sessions: Sequence[SessionEmbedding],
    *,
    human_mismatch_evidence: Sequence[HumanMismatchEvidence] = (),
) -> V2PairSet:
    return _build_v2_pair_set(
        sessions,
        human_mismatch_evidence=human_mismatch_evidence,
    )


def student_balanced_match_recall(
    scores: np.ndarray,
    student_ids: Sequence[str],
    threshold: float,
) -> float:
    values = _finite_scores(scores, description="scores")
    _validate_threshold(threshold)
    if values.size != len(student_ids):
        raise ValueError("scores and student_ids must have the same length")
    if values.size == 0:
        raise ValueError("scores must not be empty")
    by_student: dict[str, list[bool]] = defaultdict(list)
    for score, student_id in zip(values, student_ids, strict=True):
        by_student[str(student_id)].append(float(score) >= threshold)
    return float(np.mean([np.mean(matches) for matches in by_student.values()]))


def bootstrap_seed(dataset_digest: str) -> int:
    digest = hashlib.sha256(
        f"{dataset_digest}\0{_BOOTSTRAP_NAMESPACE}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def bootstrap_far_upper_bound(
    negative_scores: np.ndarray,
    group_ids: Sequence[str],
    threshold: float,
    dataset_digest: str,
    iterations: int = 10000,
) -> BootstrapFar:
    scores = _finite_scores(negative_scores, description="negative_scores")
    _validate_threshold(threshold)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if scores.size != len(group_ids):
        raise ValueError("negative_scores and group_ids must have the same length")
    if scores.size == 0:
        raise ValueError("negative_scores must not be empty")
    thresholds = np.asarray([threshold], dtype=np.float64)
    grid = _bootstrap_far_grid(scores, group_ids, thresholds, dataset_digest, iterations)
    return BootstrapFar(
        estimate=float(grid.empirical_far[0]),
        upper_95=float(grid.upper_95[0]),
        iterations=grid.iterations,
        seed=grid.seed,
        group_count=grid.group_count,
    )


def calibrate_threshold(
    positive_scores: np.ndarray,
    positive_student_ids: Sequence[str],
    negative_scores: np.ndarray,
    negative_group_ids: Sequence[str],
    dataset_digest: str,
    threshold_floor: float = 0.35,
    max_empirical_far: float = 0.005,
    max_upper_far: float = 0.01,
) -> CalibratedThreshold:
    positives = _finite_scores(positive_scores, description="positive_scores")
    negatives = _finite_scores(negative_scores, description="negative_scores")
    if positives.size != len(positive_student_ids):
        raise ValueError("positive_scores and positive_student_ids must have the same length")
    if negatives.size != len(negative_group_ids):
        raise ValueError("negative_scores and negative_group_ids must have the same length")
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("positive_scores and negative_scores must not be empty")
    return _select_threshold(
        positive_scores=positives,
        positive_student_ids=positive_student_ids,
        negative_scores=negatives,
        negative_group_ids=negative_group_ids,
        dataset_digest=dataset_digest,
        threshold_floor=threshold_floor,
        max_empirical_far=max_empirical_far,
        max_upper_far=max_upper_far,
    )


def compare_at_far_budget(
    raw_positive_scores: np.ndarray,
    adapted_positive_scores: np.ndarray,
    positive_student_ids: Sequence[str],
    raw_negative_scores: np.ndarray,
    adapted_negative_scores: np.ndarray,
    negative_group_ids: Sequence[str],
    dataset_digest: str,
) -> BudgetComparison:
    raw = _select_threshold(
        positive_scores=_finite_scores(raw_positive_scores, description="raw_positive_scores"),
        positive_student_ids=positive_student_ids,
        negative_scores=_finite_scores(raw_negative_scores, description="raw_negative_scores"),
        negative_group_ids=negative_group_ids,
        dataset_digest=dataset_digest,
        threshold_floor=0.35,
        max_empirical_far=0.01,
        max_upper_far=1.0,
    )
    adapted = _select_threshold(
        positive_scores=_finite_scores(
            adapted_positive_scores, description="adapted_positive_scores"
        ),
        positive_student_ids=positive_student_ids,
        negative_scores=_finite_scores(
            adapted_negative_scores, description="adapted_negative_scores"
        ),
        negative_group_ids=negative_group_ids,
        dataset_digest=dataset_digest,
        threshold_floor=0.35,
        max_empirical_far=0.01,
        max_upper_far=1.0,
    )
    same_threshold_raw_recall = student_balanced_match_recall(
        raw_positive_scores, positive_student_ids, 0.35
    )
    same_threshold_adapted_recall = student_balanced_match_recall(
        adapted_positive_scores, positive_student_ids, 0.35
    )
    return BudgetComparison(
        raw=raw,
        adapted=adapted,
        recall_lift=adapted.student_balanced_recall - raw.student_balanced_recall,
        true_match_delta=adapted.true_matches - raw.true_matches,
        far_delta=adapted.empirical_far - raw.empirical_far,
        same_threshold_raw_recall=same_threshold_raw_recall,
        same_threshold_adapted_recall=same_threshold_adapted_recall,
    )


def relative_gate_passes(metrics: RelativeGateMetrics) -> bool:
    return (
        metrics.adapted_frozen_far <= metrics.raw_frozen_far
        and metrics.budget.recall_lift >= 0.02
        and metrics.budget.true_match_delta > 0
        and metrics.same_threshold_recall_delta > 0.0
    )


def _select_threshold(
    *,
    positive_scores: np.ndarray,
    positive_student_ids: Sequence[str],
    negative_scores: np.ndarray,
    negative_group_ids: Sequence[str],
    dataset_digest: str,
    threshold_floor: float,
    max_empirical_far: float,
    max_upper_far: float,
) -> CalibratedThreshold:
    if positive_scores.size != len(positive_student_ids):
        raise ValueError("positive_scores and positive_student_ids must have the same length")
    if negative_scores.size != len(negative_group_ids):
        raise ValueError("negative_scores and negative_group_ids must have the same length")
    _validate_threshold(threshold_floor)
    if threshold_floor > 1.0:
        raise ValueError("threshold_floor must be less than or equal to 1")
    if not (0.0 <= max_empirical_far <= 1.0 and 0.0 <= max_upper_far <= 1.0):
        raise ValueError("FAR ceilings must be between 0 and 1")

    candidates = []
    thresholds = _candidate_thresholds(threshold_floor)
    bootstrap = _bootstrap_far_grid(
        negative_scores,
        negative_group_ids,
        thresholds,
        dataset_digest,
        10000,
    )
    for index, threshold in enumerate(thresholds):
        empirical_far = float(bootstrap.empirical_far[index])
        recall = student_balanced_match_recall(
            positive_scores, positive_student_ids, threshold
        )
        true_matches = int(np.count_nonzero(positive_scores >= threshold))
        candidates.append(
            CalibratedThreshold(
                threshold=float(threshold),
                empirical_far=empirical_far,
                far_upper_95=float(bootstrap.upper_95[index]),
                student_balanced_recall=recall,
                true_matches=true_matches,
                feasible=(
                    empirical_far <= max_empirical_far
                    and float(bootstrap.upper_95[index]) <= max_upper_far
                ),
            )
        )

    feasible = [candidate for candidate in candidates if candidate.feasible]
    if feasible:
        return max(
            feasible,
            key=lambda candidate: (
                candidate.student_balanced_recall,
                -candidate.far_upper_95,
                -candidate.empirical_far,
                candidate.threshold,
            ),
        )
    return min(
        candidates,
        key=lambda candidate: (
            candidate.far_upper_95,
            candidate.empirical_far,
            -candidate.student_balanced_recall,
            -candidate.threshold,
        ),
    )


def _candidate_thresholds(threshold_floor: float) -> np.ndarray:
    upper = 1.0
    count = int(math.floor((upper - threshold_floor) / 0.001 + 1e-9)) + 1
    return np.round(
        threshold_floor + np.arange(count, dtype=np.float64) * 0.001,
        3,
    )


@dataclass(frozen=True)
class _BootstrapGrid:
    empirical_far: np.ndarray
    upper_95: np.ndarray
    iterations: int
    seed: int
    group_count: int


def _embedding_dimension(sessions: Sequence[SessionEmbedding]) -> int:
    if not sessions:
        return 0
    return int(np.asarray(sessions[0].ref_embedding).reshape(-1).size)


def _finite_scores(scores: np.ndarray, *, description: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"{description} must be finite")
    return values


def _group_id(category: str, *parts: str) -> str:
    payload = "\0".join((_GROUP_NAMESPACE, category, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_embedding(value: np.ndarray, *, description: str) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(embedding).all():
        raise ValueError(f"{description} must be finite")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        raise ValueError(f"{description} norm must be positive")
    return embedding / norm


def _session_sort_key(session: SessionEmbedding) -> tuple[int, str, str]:
    return (_SPLIT_ORDER.get(session.split, 99), session.student_id, session.session_id)


def _stack_or_empty(values: Sequence[np.ndarray], *, dimension: int) -> np.ndarray:
    if not values:
        return np.empty((0, dimension), dtype=np.float32)
    return np.stack(values).astype(np.float32, copy=False)


def _validate_group_metadata(
    group_ids: Sequence[str],
    categories: Sequence[_NEGATIVE_CATEGORY],
    metadata: Sequence[PairMetadata],
) -> None:
    expected: dict[str, tuple[object, ...]] = {}
    for group_id, category, pair in zip(group_ids, categories, metadata, strict=True):
        if category == "synthetic_cross_student":
            signature = (category, pair.ref_student_id, pair.photo_student_id)
        else:
            signature = (
                category,
                pair.ref_student_id,
                pair.photo_student_id,
            )
        previous = expected.setdefault(group_id, signature)
        if previous != signature:
            raise ValueError("contradictory group metadata")


def _validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")


def _weights_for_groups(group_ids: Sequence[str]) -> np.ndarray:
    if not group_ids:
        return np.empty((0,), dtype=np.float32)
    group_sizes = Counter(group_ids)
    group_count = len(group_sizes)
    return np.asarray(
        [1.0 / group_count / group_sizes[group_id] for group_id in group_ids],
        dtype=np.float32,
    )


def _bootstrap_far_grid(
    negative_scores: np.ndarray,
    group_ids: Sequence[str],
    thresholds: np.ndarray,
    dataset_digest: str,
    iterations: int,
) -> _BootstrapGrid:
    group_scores, group_sizes = _grouped_negative_scores(negative_scores, group_ids)
    empirical_far = np.mean(negative_scores[:, None] >= thresholds[None, :], axis=0)
    seed = bootstrap_seed(dataset_digest)
    generator = np.random.default_rng(seed)
    group_count = len(group_scores)
    upper_95 = np.empty(thresholds.shape, dtype=np.float64)
    block = 128
    retain_count = iterations - int(math.ceil((iterations - 1) * 0.95))
    block_states: list[tuple[slice, np.ndarray, np.ndarray]] = []
    for start in range(0, len(thresholds), block):
        end = min(start + block, len(thresholds))
        accept_matrix = np.vstack(
            [
                scores.size - np.searchsorted(scores, thresholds[start:end], side="left")
                for scores in group_scores
            ]
        ).astype(np.int32, copy=False)
        block_states.append(
            (
                slice(start, end),
                accept_matrix,
                np.full((retain_count, end - start), -np.inf, dtype=np.float64),
            )
        )
    for sample_counts in _iter_bootstrap_sample_counts(
        generator,
        group_count=group_count,
        iterations=iterations,
        batch_size=_BOOTSTRAP_DRAW_BATCH_SIZE,
    ):
        sampled_sizes = sample_counts @ group_sizes
        for block_slice, accept_matrix, top_values in block_states:
            sampled_accepts = sample_counts @ accept_matrix
            replicates = sampled_accepts / sampled_sizes[:, None]
            top_values[...] = _top_k_higher_quantile_values(
                top_values, replicates, retain_count
            )
    for block_slice, _accept_matrix, top_values in block_states:
        upper_95[block_slice] = np.min(top_values, axis=0)
    return _BootstrapGrid(
        empirical_far=np.asarray(empirical_far, dtype=np.float64),
        upper_95=upper_95,
        iterations=iterations,
        seed=seed,
        group_count=group_count,
    )


def _human_mismatch_evidence_by_session(
    human_mismatch_evidence: Sequence[HumanMismatchEvidence],
) -> dict[tuple[str, str], HumanMismatchEvidence]:
    evidence_by_session: dict[tuple[str, str], HumanMismatchEvidence] = {}
    for item in human_mismatch_evidence:
        key = (item.student_id, item.session_id)
        if not item.ordered_pair_token:
            raise ValueError("human mismatch evidence ordered_pair_token must be non-empty")
        if not item.photo_student_token or not item.photo_session_token:
            raise ValueError(
                "human mismatch evidence photo tokens must be non-empty"
            )
        previous = evidence_by_session.setdefault(key, item)
        if previous != item:
            raise ValueError("duplicate human mismatch evidence for one session")
    return evidence_by_session


def _iter_bootstrap_sample_counts(
    generator: np.random.Generator,
    *,
    group_count: int,
    iterations: int,
    batch_size: int,
) -> Iterator[np.ndarray]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    probabilities = np.full(group_count, 1.0 / group_count, dtype=np.float64)
    remaining = iterations
    while remaining > 0:
        count = min(batch_size, remaining)
        yield generator.multinomial(group_count, probabilities, size=count)
        remaining -= count


def _top_k_higher_quantile_values(
    current: np.ndarray,
    new_values: np.ndarray,
    keep: int,
) -> np.ndarray:
    combined = np.concatenate((current, new_values), axis=0)
    partitioned = np.partition(combined, combined.shape[0] - keep, axis=0)
    return partitioned[-keep:, :]


def _grouped_negative_scores(
    negative_scores: np.ndarray,
    group_ids: Sequence[str],
) -> tuple[list[np.ndarray], np.ndarray]:
    grouped: dict[str, list[float]] = {}
    order: list[str] = []
    for score, group_id in zip(negative_scores, group_ids, strict=True):
        key = str(group_id)
        if not key:
            raise ValueError("group_ids must be non-empty")
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(float(score))
    if not order:
        raise ValueError("group_ids must define at least one group")
    arrays = [
        np.sort(np.asarray(grouped[group_id], dtype=np.float64))
        for group_id in order
    ]
    sizes = np.asarray([array.size for array in arrays], dtype=np.int64)
    if (sizes <= 0).any():
        raise ValueError("group_ids must define only non-empty groups")
    return arrays, sizes
