"""Train and export the frozen-SFace low-rank domain adapter."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from module.face.pipeline import build_pipeline_from_env
from module.face.session_consistency import (
    check_internal_consistency,
    session_prototype,
)


def set_deterministic(seed: int) -> None:
    """Seed every training RNG and reject nondeterministic PyTorch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class ResidualTower(nn.Module):
    def __init__(self, dimension: int = 128, rank: int = 16):
        super().__init__()
        if dimension <= 0 or rank <= 0:
            raise ValueError("dimension and rank must be positive")
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: Tensor) -> Tensor:
        return F.normalize(value + self.up(self.down(value)), dim=-1)


class LowRankDomainAdapter(nn.Module):
    def __init__(self, dimension: int = 128, rank: int = 16):
        super().__init__()
        self.dimension = dimension
        self.rank = rank
        self.ref_tower = ResidualTower(dimension, rank)
        self.photo_tower = ResidualTower(dimension, rank)

    def forward(self, ref_embedding: Tensor, photo_embedding: Tensor) -> Tensor:
        return (self.ref_tower(ref_embedding) * self.photo_tower(photo_embedding)).sum(
            dim=-1
        )


def residual_weight_regularization(model: LowRankDomainAdapter) -> Tensor:
    """Return the mean squared magnitude of all residual-branch weights."""
    penalties = [parameter.square().mean() for parameter in model.parameters()]
    return torch.stack(penalties).sum()


def contrastive_margin_loss(
    scores: Tensor,
    labels: Tensor,
    *,
    positive_margin: float,
    negative_margin: float,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    residual_parameters: Iterable[Tensor] = (),
    identity_regularization_weight: float = 0.0,
) -> Tensor:
    """Weighted positive/negative cosine hinges plus residual L2 regularization."""
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must contain only 0 and 1")
    if positive_weight < 0 or negative_weight < 0 or identity_regularization_weight < 0:
        raise ValueError("loss weights must be non-negative")

    zero = scores.sum() * 0.0
    positive = labels == 1
    negative = labels == 0
    positive_loss = (
        F.relu(positive_margin - scores[positive]).mean() if positive.any() else zero
    )
    negative_loss = (
        F.relu(scores[negative] - negative_margin).mean() if negative.any() else zero
    )
    parameters = tuple(residual_parameters)
    regularization = (
        torch.stack([parameter.square().mean() for parameter in parameters]).sum()
        if parameters
        else zero
    )
    return (
        positive_weight * positive_loss
        + negative_weight * negative_loss
        + identity_regularization_weight * regularization
    )


@dataclass(frozen=True)
class SessionEmbedding:
    student_id: str
    session_id: str
    split: str
    label: str
    ref_embedding: np.ndarray
    session_prototype: np.ndarray


@dataclass(frozen=True)
class PairMetadata:
    ref_student_id: str
    ref_session_id: str
    photo_student_id: str
    photo_session_id: str
    label: int
    raw_cosine: float


@dataclass(frozen=True)
class PairSet:
    ref_embeddings: np.ndarray
    photo_embeddings: np.ndarray
    labels: np.ndarray
    metadata: tuple[PairMetadata, ...]

    def __len__(self) -> int:
        return int(self.labels.size)


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    true_accept_rate: float
    false_accept_rate: float
    false_reject_rate: float
    inconclusive_rate: float
    conditional_accuracy: float
    true_accepts: int
    false_accepts: int
    false_rejects: int
    inconclusive: int
    total: int


@dataclass(frozen=True)
class TrainingResult:
    model: LowRankDomainAdapter
    history: tuple[dict[str, float | int], ...]
    validation_metrics: ThresholdMetrics
    test_metrics: ThresholdMetrics | None
    best_epoch: int


_SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def manifest_training_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic, student-disjoint rows eligible for Task 2."""
    if manifest.get("schema_version") != 1:
        raise ValueError("dataset manifest schema_version must be 1")
    primary = manifest.get("sessions", [])
    evaluation = manifest.get("evaluation_sessions", [])
    if not isinstance(primary, list) or not isinstance(evaluation, list):
        raise ValueError("dataset manifest session collections must be lists")

    rows: list[dict[str, Any]] = []
    for raw in [*primary, *evaluation]:
        if not isinstance(raw, dict):
            raise ValueError("dataset manifest session rows must be objects")
        row = dict(raw)
        split = row.get("split")
        if split not in _SPLIT_ORDER:
            raise ValueError(f"invalid dataset split: {split!r}")
        reasons = set(row.get("training_exclusion_reasons", []))
        if reasons - {"test_split"}:
            continue
        if not row.get("ref_image_path") or not row.get("photos"):
            continue
        if row.get("label") not in {"match", "mismatch"}:
            raise ValueError(f"invalid session label: {row.get('label')!r}")
        rows.append(row)

    owners: dict[str, str] = {}
    identities: set[tuple[str, str]] = set()
    for row in rows:
        student_id = str(row.get("student_id", ""))
        session_id = str(row.get("session_id", ""))
        if not student_id or not session_id:
            raise ValueError("dataset rows require student_id and session_id")
        split = str(row["split"])
        previous = owners.setdefault(student_id, split)
        if previous != split:
            raise ValueError(f"student {student_id!r} occurs in multiple splits")
        identity = (student_id, session_id)
        if identity in identities:
            raise ValueError(f"duplicate dataset session: {student_id}/{session_id}")
        identities.add(identity)

    return sorted(
        rows,
        key=lambda row: (
            _SPLIT_ORDER[str(row["split"])],
            str(row["student_id"]),
            str(row["session_id"]),
        ),
    )


def _stat_cache_key(path: str) -> str:
    stat = Path(path).stat()
    return f"{path}\0{stat.st_mtime_ns}\0{stat.st_size}"


def _normalize_embedding(value: np.ndarray) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(embedding).all():
        raise ValueError("embedding must be finite")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        raise ValueError("embedding norm must be positive")
    return embedding / norm


class _EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.values: dict[str, np.ndarray] = {}
        self.dirty = False
        if not path.exists():
            return
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = archive["keys"]
                embeddings = archive["embeddings"]
                if embeddings.ndim != 2 or len(keys) != len(embeddings):
                    return
                for key, embedding in zip(keys.tolist(), embeddings):
                    self.values[str(key)] = _normalize_embedding(embedding)
        except (OSError, ValueError, KeyError):
            self.values = {}

    def get(self, path: str) -> np.ndarray | None:
        try:
            key = _stat_cache_key(path)
        except OSError:
            return None
        value = self.values.get(key)
        return value.copy() if value is not None else None

    def put(self, path: str, embedding: np.ndarray) -> np.ndarray:
        key = _stat_cache_key(path)
        normalized = _normalize_embedding(embedding)
        self.values[key] = normalized.copy()
        self.dirty = True
        return normalized

    def save(self) -> None:
        if not self.dirty and self.path.exists():
            os.chmod(self.path, 0o600)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.values)
        if keys:
            embeddings = np.stack([self.values[key] for key in keys]).astype(np.float32)
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(
                    handle,
                    keys=np.asarray(keys, dtype=np.str_),
                    embeddings=embeddings,
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
            os.chmod(self.path, 0o600)
            self.dirty = False
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise


def extract_session_embeddings(
    manifest: dict[str, Any],
    *,
    cache_path: Path,
    pipeline_factory: Any = build_pipeline_from_env,
) -> list[SessionEmbedding]:
    """Extract raw SFace references and outlier-filtered post-gate prototypes."""
    rows = manifest_training_rows(manifest)
    pipeline = pipeline_factory()
    cache = _EmbeddingCache(cache_path)
    sessions: list[SessionEmbedding] = []
    try:
        for row in rows:
            ref_path = str(row["ref_image_path"])
            ref_embedding = cache.get(ref_path)
            if ref_embedding is None:
                image = cv2.imread(ref_path)
                if image is None or image.size == 0:
                    continue
                try:
                    ref_embedding = cache.put(
                        ref_path, pipeline._extract_or_raise(image)
                    )
                except (OSError, RuntimeError, ValueError):
                    continue

            photo_embeddings: list[np.ndarray] = []
            for photo in row["photos"]:
                photo_path = str(photo.get("image_path", ""))
                if not photo_path:
                    continue
                embedding = cache.get(photo_path)
                if embedding is None:
                    try:
                        processed = pipeline._process_photo_for_session(photo)
                        if not processed.passes_gate:
                            continue
                        embedding = cache.put(photo_path, processed._embedding)
                    except (OSError, RuntimeError, ValueError, AttributeError):
                        continue
                photo_embeddings.append(embedding)
            if not photo_embeddings:
                continue

            consistency = check_internal_consistency(photo_embeddings)
            outliers = set(consistency.outlier_indices)
            core = [
                embedding
                for index, embedding in enumerate(photo_embeddings)
                if index not in outliers
            ]
            if not core:
                continue
            prototype = _normalize_embedding(session_prototype(core))
            sessions.append(
                SessionEmbedding(
                    student_id=str(row["student_id"]),
                    session_id=str(row["session_id"]),
                    split=str(row["split"]),
                    label=str(row["label"]),
                    ref_embedding=ref_embedding,
                    session_prototype=prototype,
                )
            )
    finally:
        cache.save()
    return sessions


def _empty_pair_set(dimension: int) -> PairSet:
    return PairSet(
        ref_embeddings=np.empty((0, dimension), dtype=np.float32),
        photo_embeddings=np.empty((0, dimension), dtype=np.float32),
        labels=np.empty((0,), dtype=np.float32),
        metadata=(),
    )


def build_pair_sets(
    sessions: Iterable[SessionEmbedding],
    *,
    max_train_negatives_per_positive: int = 20,
) -> dict[str, PairSet]:
    """Build own-session pairs and deterministic raw-cosine hard negatives."""
    if max_train_negatives_per_positive < 0:
        raise ValueError("max_train_negatives_per_positive must be non-negative")
    ordered = sorted(
        sessions,
        key=lambda item: (
            _SPLIT_ORDER.get(item.split, 99),
            item.student_id,
            item.session_id,
        ),
    )
    dimension = 128
    if ordered:
        dimension = int(np.asarray(ordered[0].ref_embedding).size)
    grouped: dict[str, list[SessionEmbedding]] = {split: [] for split in _SPLIT_ORDER}
    owners: dict[str, str] = {}
    for item in ordered:
        if item.split not in grouped:
            raise ValueError(f"invalid session split: {item.split!r}")
        previous = owners.setdefault(item.student_id, item.split)
        if previous != item.split:
            raise ValueError(f"student {item.student_id!r} occurs in multiple splits")
        ref = np.asarray(item.ref_embedding, dtype=np.float32).reshape(-1)
        photo = np.asarray(item.session_prototype, dtype=np.float32).reshape(-1)
        if ref.size != dimension or photo.size != dimension:
            raise ValueError("all embeddings must have one consistent dimension")
        if not np.isfinite(ref).all() or not np.isfinite(photo).all():
            raise ValueError("embeddings must be finite")
        grouped[item.split].append(item)

    result: dict[str, PairSet] = {}
    for split, items in grouped.items():
        refs: list[np.ndarray] = []
        photos: list[np.ndarray] = []
        labels: list[float] = []
        metadata: list[PairMetadata] = []

        def append_pair(
            ref_item: SessionEmbedding,
            photo_item: SessionEmbedding,
            label: int,
            raw_cosine: float,
        ) -> None:
            refs.append(np.asarray(ref_item.ref_embedding, dtype=np.float32))
            photos.append(np.asarray(photo_item.session_prototype, dtype=np.float32))
            labels.append(float(label))
            metadata.append(
                PairMetadata(
                    ref_student_id=ref_item.student_id,
                    ref_session_id=ref_item.session_id,
                    photo_student_id=photo_item.student_id,
                    photo_session_id=photo_item.session_id,
                    label=label,
                    raw_cosine=raw_cosine,
                )
            )

        positives = [item for item in items if item.label == "match"]
        for item in items:
            raw_cosine = float(np.dot(item.ref_embedding, item.session_prototype))
            append_pair(item, item, int(item.label == "match"), raw_cosine)

        for anchor in positives:
            candidates = [
                (
                    float(np.dot(anchor.ref_embedding, other.session_prototype)),
                    other,
                )
                for other in items
                if other.student_id != anchor.student_id
            ]
            candidates.sort(
                key=lambda candidate: (
                    -candidate[0],
                    candidate[1].student_id,
                    candidate[1].session_id,
                )
            )
            if split == "train":
                candidates = candidates[:max_train_negatives_per_positive]
            for raw_cosine, other in candidates:
                append_pair(anchor, other, 0, raw_cosine)

        if not labels:
            result[split] = _empty_pair_set(dimension)
            continue
        result[split] = PairSet(
            ref_embeddings=np.stack(refs).astype(np.float32, copy=False),
            photo_embeddings=np.stack(photos).astype(np.float32, copy=False),
            labels=np.asarray(labels, dtype=np.float32),
            metadata=tuple(metadata),
        )
    return result


def evaluate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError("scores and labels must be non-empty and have the same shape")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")

    positive = labels == 1
    negative = labels == 0
    accepted = scores >= threshold
    rejected = scores < 0.15
    inconclusive_mask = ~(accepted | rejected)
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    true_accepts = int((positive & accepted).sum())
    false_accepts = int((negative & accepted).sum())
    false_rejects = int((positive & rejected).sum())
    inconclusive = int(inconclusive_mask.sum())
    decisive = int((accepted | rejected).sum())
    correct_decisive = int(((positive & accepted) | (negative & rejected)).sum())
    return ThresholdMetrics(
        threshold=float(threshold),
        true_accept_rate=true_accepts / positive_count if positive_count else 0.0,
        false_accept_rate=false_accepts / negative_count if negative_count else 0.0,
        false_reject_rate=false_rejects / positive_count if positive_count else 0.0,
        inconclusive_rate=inconclusive / int(scores.size),
        conditional_accuracy=correct_decisive / decisive if decisive else 0.0,
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        inconclusive=inconclusive,
        total=int(scores.size),
    )


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    max_false_accept_rate: float = 0.01,
) -> ThresholdMetrics:
    if not 0.0 <= max_false_accept_rate <= 1.0:
        raise ValueError("max_false_accept_rate must be between 0 and 1")
    candidates = np.arange(0.15, 0.501, 0.001)
    feasible = [evaluate_threshold(scores, labels, value) for value in candidates]
    feasible = [
        item for item in feasible if item.false_accept_rate <= max_false_accept_rate
    ]
    if not feasible:
        raise ValueError("no threshold satisfies the false-accept ceiling")
    return max(feasible, key=lambda item: (item.true_accept_rate, -item.threshold))


def _pair_tensors(pair_set: PairSet, dimension: int) -> tuple[Tensor, Tensor, Tensor]:
    refs = np.asarray(pair_set.ref_embeddings, dtype=np.float32)
    photos = np.asarray(pair_set.photo_embeddings, dtype=np.float32)
    labels = np.asarray(pair_set.labels, dtype=np.float32).reshape(-1)
    expected = (labels.size, dimension)
    if refs.shape != expected or photos.shape != expected:
        raise ValueError(f"pair embeddings must have shape {expected}")
    if labels.size == 0:
        raise ValueError("pair set must not be empty")
    if not np.isfinite(refs).all() or not np.isfinite(photos).all():
        raise ValueError("pair embeddings must be finite")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("pair labels must contain only 0 and 1")
    return torch.from_numpy(refs), torch.from_numpy(photos), torch.from_numpy(labels)


def _score_pair_set(
    model: LowRankDomainAdapter,
    tensors: tuple[Tensor, Tensor, Tensor],
) -> np.ndarray:
    refs, photos, _ = tensors
    with torch.no_grad():
        return model(refs, photos).cpu().numpy().astype(np.float64)


def train_adapter(
    train_pairs: PairSet,
    validation_pairs: PairSet,
    test_pairs: PairSet | None = None,
    *,
    dimension: int = 128,
    rank: int = 16,
    epochs: int = 100,
    seed: int = 20260830,
    learning_rate: float = 0.01,
    positive_margin: float = 0.35,
    negative_margin: float = 0.15,
    identity_regularization_weight: float = 1e-3,
    max_false_accept_rate: float = 0.01,
    early_stopping_patience: int = 10,
) -> TrainingResult:
    """Train deterministically and retain the best FAR-feasible validation epoch."""
    if epochs <= 0 or early_stopping_patience <= 0:
        raise ValueError("epochs and early_stopping_patience must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= max_false_accept_rate <= 1.0:
        raise ValueError("max_false_accept_rate must be between 0 and 1")
    set_deterministic(seed)
    train_tensors = _pair_tensors(train_pairs, dimension)
    validation_tensors = _pair_tensors(validation_pairs, dimension)
    test_tensors = (
        _pair_tensors(test_pairs, dimension) if test_pairs is not None else None
    )
    model = LowRankDomainAdapter(dimension=dimension, rank=rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    best_state: dict[str, Tensor] | None = None
    best_key: tuple[float, float, int] | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_scores = model(train_tensors[0], train_tensors[1])
        train_loss = contrastive_margin_loss(
            train_scores,
            train_tensors[2],
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            positive_weight=0.5,
            negative_weight=0.5,
            residual_parameters=model.parameters(),
            identity_regularization_weight=identity_regularization_weight,
        )
        if not torch.isfinite(train_loss):
            raise FloatingPointError("training loss became non-finite")
        train_loss.backward()
        optimizer.step()

        model.eval()
        validation_scores = model(validation_tensors[0], validation_tensors[1])
        validation_loss = contrastive_margin_loss(
            validation_scores,
            validation_tensors[2],
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            positive_weight=0.5,
            negative_weight=0.5,
        )
        if not torch.isfinite(validation_loss):
            raise FloatingPointError("validation loss became non-finite")
        validation_numpy = validation_scores.detach().cpu().numpy()
        validation_labels = validation_tensors[2].cpu().numpy()
        try:
            metrics = select_threshold(
                validation_numpy,
                validation_labels,
                max_false_accept_rate,
            )
            far_feasible = True
        except ValueError:
            metrics = evaluate_threshold(validation_numpy, validation_labels, 0.5)
            far_feasible = False
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(train_loss.detach()),
            "validation_loss": float(validation_loss.detach()),
            "validation_threshold": metrics.threshold,
            "validation_true_accept_rate": metrics.true_accept_rate,
            "validation_false_accept_rate": metrics.false_accept_rate,
            "validation_far_feasible": int(far_feasible),
        }
        history.append(epoch_record)

        key = (
            float(validation_loss.detach()),
            -metrics.true_accept_rate,
            epoch,
        )
        if far_feasible and (best_key is None or key < best_key):
            best_key = key
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            stale_epochs = 0
        elif best_state is not None:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no FAR-feasible validation epoch")
    model.load_state_dict(best_state)
    model.eval()
    validation_scores = _score_pair_set(model, validation_tensors)
    validation_metrics = select_threshold(
        validation_scores,
        validation_tensors[2].numpy(),
        max_false_accept_rate,
    )
    test_metrics = None
    if test_tensors is not None:
        test_scores = _score_pair_set(model, test_tensors)
        test_metrics = evaluate_threshold(
            test_scores,
            test_tensors[2].numpy(),
            validation_metrics.threshold,
        )
    return TrainingResult(
        model=model,
        history=tuple(history),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
    )


def export_onnx(
    model: LowRankDomainAdapter,
    output_path: Path,
    *,
    parity_inputs: tuple[np.ndarray, np.ndarray] | None = None,
    absolute_tolerance: float = 1e-5,
) -> float:
    """Atomically export a dynamic-batch ONNX graph after CPU parity succeeds."""
    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    dimension = model.dimension
    if parity_inputs is None:
        generator = torch.Generator().manual_seed(0)
        refs = F.normalize(torch.randn(4, dimension, generator=generator), dim=1)
        photos = F.normalize(torch.randn(4, dimension, generator=generator), dim=1)
    else:
        refs = torch.from_numpy(np.asarray(parity_inputs[0], dtype=np.float32))
        photos = torch.from_numpy(np.asarray(parity_inputs[1], dtype=np.float32))
    if refs.ndim != 2 or photos.shape != refs.shape or refs.shape[1] != dimension:
        raise ValueError(f"parity inputs must both have shape [N, {dimension}]")
    if (
        refs.shape[0] == 0
        or not torch.isfinite(refs).all()
        or not torch.isfinite(photos).all()
    ):
        raise ValueError("parity inputs must be non-empty and finite")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".onnx", dir=output_path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    was_training = model.training
    try:
        model.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using the legacy TorchScript-based ONNX export.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="The feature will be removed.*",
                category=DeprecationWarning,
            )
            torch.onnx.export(
                model,
                (refs, photos),
                str(temporary),
                input_names=["ref_embedding", "photo_embedding"],
                output_names=["adapted_cosine"],
                dynamic_axes={
                    "ref_embedding": {0: "N"},
                    "photo_embedding": {0: "N"},
                    "adapted_cosine": {0: "N"},
                },
                opset_version=17,
                dynamo=False,
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(temporary),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        actual = session.run(
            ["adapted_cosine"],
            {
                "ref_embedding": refs.numpy(),
                "photo_embedding": photos.numpy(),
            },
        )[0]
        with torch.no_grad():
            expected = model(refs, photos).numpy()
        max_error = float(np.max(np.abs(actual - expected)))
        if not np.isfinite(actual).all() or max_error > absolute_tolerance:
            raise RuntimeError(
                "ONNX parity failed: "
                f"max_abs_error={max_error:.12g}, tolerance={absolute_tolerance:.12g}"
            )
        os.chmod(temporary, 0o600)
        temporary.replace(output_path)
        os.chmod(output_path, 0o600)
        return max_error
    finally:
        model.train(was_training)
        temporary.unlink(missing_ok=True)


def _atomic_write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _pair_counts(
    pair_sets: dict[str, PairSet],
    dataset_manifest: dict[str, Any],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in _SPLIT_ORDER:
        pair_set = pair_sets.get(split, _empty_pair_set(128))
        labels = np.asarray(pair_set.labels)
        counts[split] = {
            "sessions": 0,
            "positive_sessions": 0,
            "negative_sessions": 0,
            "pairs": int(labels.size),
            "positive_pairs": int((labels == 1).sum()),
            "negative_pairs": int((labels == 0).sum()),
        }
    for row in manifest_training_rows(dataset_manifest):
        split_counts = counts[str(row["split"])]
        split_counts["sessions"] += 1
        key = "positive_sessions" if row["label"] == "match" else "negative_sessions"
        split_counts[key] += 1
    return counts


def _threshold_sweep(
    model: LowRankDomainAdapter,
    pair_set: PairSet,
) -> list[dict[str, Any]]:
    if len(pair_set) == 0:
        return []
    tensors = _pair_tensors(pair_set, model.dimension)
    scores = _score_pair_set(model, tensors)
    labels = tensors[2].numpy()
    return [
        asdict(evaluate_threshold(scores, labels, threshold))
        for threshold in np.arange(0.15, 0.501, 0.001)
    ]


def write_candidate_artifacts(
    result: TrainingResult,
    pair_sets: dict[str, PairSet],
    dataset_manifest: dict[str, Any],
    output_dir: Path,
    *,
    hyperparameters: dict[str, Any],
    source_revision: str,
    model_version: str = "identity-domain-adapter-v1",
) -> dict[str, Any]:
    """Write the parity-checked model, private evaluation, then runtime manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    onnx_path = output_dir / "identity_domain_adapter.onnx"
    parity_error = export_onnx(result.model, onnx_path)
    onnx_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    counts = _pair_counts(pair_sets, dataset_manifest)
    validation = asdict(result.validation_metrics)
    test = asdict(result.test_metrics) if result.test_metrics is not None else None
    evaluation = {
        "schema_version": 1,
        "model_version": model_version,
        "best_epoch": result.best_epoch,
        "history": list(result.history),
        "pair_counts": counts,
        "validation_metrics": validation,
        "test_metrics": test,
        "threshold_sweeps": {
            "validation": _threshold_sweep(
                result.model,
                pair_sets.get("validation", _empty_pair_set(result.model.dimension)),
            ),
            "test": _threshold_sweep(
                result.model,
                pair_sets.get("test", _empty_pair_set(result.model.dimension)),
            ),
        },
        "onnx_parity_max_abs_error": parity_error,
    }
    evaluation_path = output_dir / "training-evaluation.json"
    _atomic_write_private_json(evaluation_path, evaluation)

    dataset_digest = hashlib.sha256(
        json.dumps(
            dataset_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "model_version": model_version,
        "embedding_dimension": result.model.dimension,
        "rank": result.model.rank,
        "match_threshold": result.validation_metrics.threshold,
        "onnx_file": onnx_path.name,
        "onnx_sha256": onnx_sha256,
        "onnx_parity_max_abs_error": parity_error,
        "source_feedback_snapshot": dataset_manifest.get("snapshot", ""),
        "source_dataset_sha256": dataset_digest,
        "source_code_revision": source_revision,
        "split_seed": dataset_manifest.get("split_seed", ""),
        "split_counts": counts,
        "training_hyperparameters": dict(hyperparameters),
        "validation_metrics": validation,
        "test_metrics": test,
        "evaluation_file": evaluation_path.name,
        "input_names": ["ref_embedding", "photo_embedding"],
        "output_name": "adapted_cosine",
    }
    _atomic_write_private_json(
        output_dir / "identity_domain_adapter.manifest.json", manifest
    )
    return manifest
