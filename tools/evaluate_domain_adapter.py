#!/usr/bin/env python3
"""Evaluate an identity domain adapter against held-out release gates.

The benchmark archive is fail-closed around ``benchmark-manifest.json`` schema 1::

    {
      "schema_version": 1,
      "benchmark_id": "customer-identity-40-v1",
      "known_impostor": {"student_id": "...", "session_id": "..."},
      "sessions": [{
        "student_id": "...", "session_id": "...",
        "record_path": "<student_id>/<session_id>/record.json",
        "record_sha256": "<sha256>", "truth": "match|mismatch"
      }],
      "manifest_sha256": "<canonical sha256 of all preceding fields>"
    }

There must be exactly 40 entries and archive records, with no unlisted paths. The one
``mismatch`` entry must be the named known impersonation. Record bytes and record identities
must match their entries before any face replay begins.

The manifest's digest is not a trust anchor by itself. Callers must pin it independently
through ``evaluate_release_candidate(..., benchmark_manifest_sha256)`` or the CLI's
``--benchmark-manifest-sha256`` option.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module.face.pipeline import build_pipeline_from_env  # noqa: E402
from module.face.recognizer import classify_match  # noqa: E402
from module.face.session_consistency import session_prototype  # noqa: E402


ARTIFACT_MANIFEST = "identity_domain_adapter.manifest.json"
BENCHMARK_MANIFEST = "benchmark-manifest.json"
KNOWN_IMPOSTOR_STUDENT_ID = "S177509932310186"
EXPECTED_BENCHMARK_SESSIONS = 40
_PREDICTIONS = ("match", "mismatch", "inconclusive")


class BenchmarkManifestTrustError(ValueError):
    def __init__(self, expected: str, observed: str):
        self.expected = expected
        self.observed = observed
        super().__init__(
            "benchmark manifest does not match trusted expected SHA256 "
            f"(expected {expected}, observed {observed})"
        )


def release_gate_passes(report: dict) -> bool:
    return (
        report["cross_student_far"] <= 0.01
        and report["conditional_accuracy"] >= 0.70
        and report["known_impostor_detected"] is True
        and report["new_false_accusations"] == 0
        and report["student_split_leaks"] == 0
    )


@dataclass(frozen=True)
class _EvaluatedSession:
    student_id: str
    session_id: str
    truth: str
    raw_status: str
    adapted_status: str
    internal_consistency: str
    adapter_applied: bool
    ref_embedding: np.ndarray | None
    session_prototype: np.ndarray | None


class _AdapterScorer:
    def __init__(self, session: Any, manifest: dict[str, Any]):
        input_names = manifest.get("input_names", ["ref_embedding", "photo_embedding"])
        if (
            not isinstance(input_names, list)
            or len(input_names) != 2
            or not all(isinstance(name, str) and name for name in input_names)
        ):
            raise ValueError("artifact input_names must contain two names")
        output_name = manifest.get("output_name", "adapted_cosine")
        if not isinstance(output_name, str) or not output_name:
            raise ValueError("artifact output_name must be a non-empty string")
        self._session = session
        self._ref_input, self._photo_input = input_names
        self._output_name = output_name
        self.dimension = int(manifest.get("embedding_dimension", 0))
        self.match_threshold = float(manifest.get("match_threshold", math.nan))
        if self.dimension <= 0:
            raise ValueError("artifact embedding_dimension must be positive")
        if not math.isfinite(self.match_threshold):
            raise ValueError("artifact match_threshold must be finite")

    def score(self, refs: np.ndarray, photos: np.ndarray) -> np.ndarray:
        refs = np.asarray(refs, dtype=np.float32)
        photos = np.asarray(photos, dtype=np.float32)
        expected = (refs.shape[0], self.dimension) if refs.ndim == 2 else None
        if expected is None or refs.shape != expected or photos.shape != expected:
            raise ValueError(
                f"adapter inputs must both have shape [N, {self.dimension}]"
            )
        if refs.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        if not np.isfinite(refs).all() or not np.isfinite(photos).all():
            raise ValueError("adapter inputs must be finite")
        output = self._session.run(
            [self._output_name],
            {self._ref_input: refs, self._photo_input: photos},
        )
        if not isinstance(output, list) or len(output) != 1:
            raise ValueError("adapter must return exactly one output")
        scores = np.asarray(output[0], dtype=np.float32).reshape(-1)
        if scores.shape != (refs.shape[0],):
            raise ValueError("adapter output has the wrong batch shape")
        if not np.isfinite(scores).all():
            raise ValueError("adapter output must be finite")
        return scores

    def classify(self, score: float) -> str:
        if score >= self.match_threshold:
            return "match"
        if score < 0.15:
            return "mismatch"
        return "inconclusive"


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{description} does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} root must be an object")
    return payload


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _verify_artifact(
    dataset: dict[str, Any], adapter_dir: Path
) -> tuple[dict[str, Any], Path]:
    artifact = _load_json_object(adapter_dir / ARTIFACT_MANIFEST, "artifact manifest")
    if artifact.get("schema_version") != 1:
        raise ValueError("artifact manifest schema_version must be 1")
    onnx_name = artifact.get("onnx_file")
    if not isinstance(onnx_name, str) or not onnx_name or Path(onnx_name).name != onnx_name:
        raise ValueError("artifact onnx_file must be a file name")
    onnx_path = adapter_dir / onnx_name
    try:
        actual_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"ONNX artifact is unreadable: {onnx_path}: {exc}") from exc
    expected_sha = artifact.get("onnx_sha256")
    if not isinstance(expected_sha, str) or actual_sha != expected_sha:
        raise ValueError("ONNX SHA256 does not match the artifact manifest")
    dataset_sha = artifact.get("source_dataset_sha256")
    if not isinstance(dataset_sha, str) or dataset_sha != _canonical_sha256(dataset):
        raise ValueError("dataset SHA256 does not match the artifact manifest")
    return artifact, onnx_path


def _student_split_leaks(dataset: dict[str, Any]) -> list[str]:
    sessions = dataset.get("sessions", [])
    evaluation = dataset.get("evaluation_sessions", [])
    if not isinstance(sessions, list) or not isinstance(evaluation, list):
        raise ValueError("dataset session collections must be lists")
    owners: dict[str, set[str]] = {}
    for row in [*sessions, *evaluation]:
        if not isinstance(row, dict):
            raise ValueError("dataset session rows must be objects")
        split = row.get("split")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid dataset split: {split!r}")
        student_id = str(row.get("student_id", ""))
        if not student_id:
            raise ValueError("dataset sessions require student_id")
        owners.setdefault(student_id, set()).add(str(split))
    for row in evaluation:
        if row.get("split") != "test":
            raise ValueError("every evaluation session must belong to the test split")
    return sorted(student_id for student_id, splits in owners.items() if len(splits) > 1)


def _normalize(value: np.ndarray, *, description: str) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(embedding).all():
        raise ValueError(f"{description} must be finite")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        raise ValueError(f"{description} must have positive norm")
    return embedding / norm


def _extract_ref_and_prototype(
    pipeline: Any, row: dict[str, Any], dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    ref_path = str(row.get("ref_image_path", ""))
    ref_image = cv2.imread(ref_path)
    if ref_image is None or ref_image.size == 0:
        raise ValueError(f"eligible session reference is unreadable: {ref_path}")
    try:
        ref = _normalize(
            pipeline._extract_or_raise(ref_image),
            description="reference embedding",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"eligible session reference extraction failed: {ref_path}") from exc

    photos = row.get("photos")
    if not isinstance(photos, list):
        raise ValueError("evaluation session photos must be a list")
    embeddings: list[np.ndarray] = []
    for photo in photos:
        if not isinstance(photo, dict):
            raise ValueError("evaluation session photo rows must be objects")
        try:
            processed = pipeline._process_photo_for_session(photo)
            embedding = getattr(processed, "_embedding", None)
            if processed.passes_gate and embedding is not None:
                embeddings.append(
                    _normalize(embedding, description="photo embedding")
                )
        except (OSError, RuntimeError, ValueError, AttributeError):
            continue
    if not embeddings:
        raise ValueError("adapter-eligible session produced no post-gate embedding")
    prototype = _normalize(
        session_prototype(embeddings), description="session prototype"
    )
    if ref.size != dimension or prototype.size != dimension:
        raise ValueError(f"evaluation embeddings must have dimension {dimension}")
    return ref, prototype


def _truth(row: dict[str, Any]) -> str:
    truth = row.get("label")
    if truth not in {"match", "mismatch"}:
        raise ValueError("evaluation session label must be match or mismatch")
    return str(truth)


def _build_evaluation_pipeline() -> Any:
    previous_mode = os.environ.get("FACE_DOMAIN_ADAPTER_MODE")
    try:
        os.environ["FACE_DOMAIN_ADAPTER_MODE"] = "off"
        return build_pipeline_from_env()
    finally:
        if previous_mode is None:
            os.environ.pop("FACE_DOMAIN_ADAPTER_MODE", None)
        else:
            os.environ["FACE_DOMAIN_ADAPTER_MODE"] = previous_mode


def _raw_stage_two_status(raw: Any) -> str:
    raw_status = getattr(raw, "raw_session_status", "")
    if raw_status not in (None, ""):
        return str(raw_status)
    return str(getattr(raw, "session_status", ""))


def _evaluate_rows(
    rows: Iterable[dict[str, Any]], pipeline: Any, scorer: _AdapterScorer
) -> list[_EvaluatedSession]:
    raw_staged: list[tuple[dict[str, Any], Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("evaluation session rows must be objects")
        ref_path = str(row.get("ref_image_path", ""))
        photos = row.get("photos", [])
        if not isinstance(photos, list):
            raise ValueError("evaluation session photos must be a list")
        raw = pipeline.session_check(ref_path, photos)
        raw_status = _raw_stage_two_status(raw)
        if raw_status not in _PREDICTIONS:
            raise ValueError(f"raw Stage 1/2 returned invalid status: {raw_status!r}")
        raw_staged.append((row, raw))

    staged: list[tuple[dict[str, Any], Any, np.ndarray | None, np.ndarray | None]] = []
    for row, raw in raw_staged:
        consistency = getattr(raw, "internal_consistency", "")
        ref: np.ndarray | None = None
        prototype: np.ndarray | None = None
        if consistency in {"consistent", "single"}:
            ref, prototype = _extract_ref_and_prototype(
                pipeline, row, scorer.dimension
            )
        staged.append((row, raw, ref, prototype))

    eligible = [item for item in staged if item[2] is not None]
    scores = scorer.score(
        np.stack([item[2] for item in eligible]),
        np.stack([item[3] for item in eligible]),
    ) if eligible else np.empty((0,), dtype=np.float32)
    score_index = 0
    evaluated: list[_EvaluatedSession] = []
    for row, raw, ref, prototype in staged:
        raw_status = _raw_stage_two_status(raw)
        consistency = str(raw.internal_consistency)
        applied = ref is not None and prototype is not None
        adapted_status = scorer.classify(float(scores[score_index])) if applied else raw_status
        if applied:
            score_index += 1
        evaluated.append(
            _EvaluatedSession(
                student_id=str(row.get("student_id", "")),
                session_id=str(row.get("session_id", "")),
                truth=_truth(row),
                raw_status=raw_status,
                adapted_status=adapted_status,
                internal_consistency=consistency,
                adapter_applied=applied,
                ref_embedding=ref,
                session_prototype=prototype,
            )
        )
    return evaluated


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {
        truth: {prediction: 0 for prediction in _PREDICTIONS}
        for truth in ("match", "mismatch")
    }


def _confusion(
    sessions: Iterable[_EvaluatedSession], *, adapted: bool
) -> dict[str, dict[str, int]]:
    result = _empty_confusion()
    for item in sessions:
        prediction = item.adapted_status if adapted else item.raw_status
        result[item.truth][prediction] += 1
    return result


def _conditional_accuracy(confusion: dict[str, dict[str, int]]) -> float:
    correct = confusion["match"]["match"] + confusion["mismatch"]["mismatch"]
    decisive = sum(
        confusion[truth][prediction]
        for truth in ("match", "mismatch")
        for prediction in ("match", "mismatch")
    )
    return correct / decisive if decisive else 0.0


def _cross_student_metrics(
    sessions: list[_EvaluatedSession], scorer: _AdapterScorer
) -> dict[str, int | float]:
    eligible = [item for item in sessions if item.adapter_applied]
    pairs = [
        (ref_session, photo_session)
        for ref_session in eligible
        for photo_session in eligible
        if ref_session.student_id != photo_session.student_id
    ]
    if not pairs:
        return {
            "cross_student_pairs": 0,
            "cross_student_false_accepts": 0,
            "cross_student_far": 0.0,
            "raw_cross_student_false_accepts": 0,
            "raw_cross_student_far": 0.0,
        }
    refs = np.stack([ref.ref_embedding for ref, _ in pairs])
    photos = np.stack([photo.session_prototype for _, photo in pairs])
    adapted_scores = scorer.score(refs, photos)
    adapted_false_accepts = int((adapted_scores >= scorer.match_threshold).sum())
    raw_scores = (refs * photos).sum(axis=1)
    raw_false_accepts = sum(
        classify_match(float(score), math.sqrt(max(0.0, 2.0 - 2.0 * float(score))))
        == "match"
        for score in raw_scores
    )
    count = len(pairs)
    return {
        "cross_student_pairs": count,
        "cross_student_false_accepts": adapted_false_accepts,
        "cross_student_far": adapted_false_accepts / count,
        "raw_cross_student_false_accepts": raw_false_accepts,
        "raw_cross_student_far": raw_false_accepts / count,
    }


def _load_benchmark_rows(
    benchmark_archive: Path,
    expected_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, str | int]]:
    if (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in expected_manifest_sha256
        )
    ):
        raise ValueError(
            "trusted expected benchmark manifest SHA256 must be 64 hex characters"
        )
    expected_manifest_sha256 = expected_manifest_sha256.lower()
    manifest_path = benchmark_archive / BENCHMARK_MANIFEST
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except FileNotFoundError as exc:
        raise ValueError(f"benchmark manifest does not exist: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"benchmark manifest is unreadable: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("benchmark manifest root must be an object")
    if manifest.get("schema_version") != 1:
        raise ValueError("benchmark manifest schema_version must be 1")
    benchmark_id = manifest.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise ValueError("benchmark manifest benchmark_id must be non-empty")
    declared_manifest_sha = manifest.get("manifest_sha256")
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    actual_manifest_sha = _canonical_sha256(manifest_payload)
    if not hmac.compare_digest(expected_manifest_sha256, actual_manifest_sha):
        raise BenchmarkManifestTrustError(
            expected_manifest_sha256,
            actual_manifest_sha,
        )
    if (
        not isinstance(declared_manifest_sha, str)
        or declared_manifest_sha != actual_manifest_sha
    ):
        raise ValueError("benchmark manifest SHA256 does not match its payload")

    entries = manifest.get("sessions")
    if not isinstance(entries, list):
        raise ValueError("benchmark manifest sessions must be a list")
    if len(entries) != EXPECTED_BENCHMARK_SESSIONS:
        raise ValueError(
            "benchmark manifest must contain exactly "
            f"{EXPECTED_BENCHMARK_SESSIONS} sessions; found {len(entries)}"
        )

    known = manifest.get("known_impostor")
    if not isinstance(known, dict):
        raise ValueError("benchmark manifest known_impostor must be an object")
    known_identity = (
        str(known.get("student_id", "")),
        str(known.get("session_id", "")),
    )
    if known_identity[0] != KNOWN_IMPOSTOR_STUDENT_ID or not known_identity[1]:
        raise ValueError("benchmark manifest known_impostor identity is invalid")

    normalized_entries: list[dict[str, str]] = []
    expected_paths: set[str] = set()
    expected_identities: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("benchmark manifest session entries must be objects")
        student_id = str(entry.get("student_id", ""))
        session_id = str(entry.get("session_id", ""))
        record_path = entry.get("record_path")
        record_sha = entry.get("record_sha256")
        truth = entry.get("truth")
        if not student_id or not session_id:
            raise ValueError("benchmark manifest entries require student_id and session_id")
        canonical_path = f"{student_id}/{session_id}/record.json"
        if not isinstance(record_path, str) or record_path != canonical_path:
            raise ValueError("benchmark manifest record paths must match student/session")
        if not isinstance(record_sha, str) or len(record_sha) != 64:
            raise ValueError("benchmark manifest record SHA256 is invalid")
        if truth not in {"match", "mismatch"}:
            raise ValueError("benchmark manifest truth must be match or mismatch")
        identity = (student_id, session_id)
        if record_path in expected_paths or identity in expected_identities:
            raise ValueError("benchmark manifest contains duplicate session entries")
        expected_paths.add(record_path)
        expected_identities.add(identity)
        normalized_entries.append(
            {
                "student_id": student_id,
                "session_id": session_id,
                "record_path": record_path,
                "record_sha256": record_sha,
                "truth": str(truth),
            }
        )

    mismatch_identities = {
        (entry["student_id"], entry["session_id"])
        for entry in normalized_entries
        if entry["truth"] == "mismatch"
    }
    if mismatch_identities != {known_identity}:
        raise ValueError(
            "benchmark manifest must identify exactly one known impersonation"
        )

    actual_paths = {
        path.relative_to(benchmark_archive).as_posix()
        for path in benchmark_archive.glob("*/*/record.json")
    }
    if actual_paths != expected_paths:
        raise ValueError("benchmark archive record paths do not match the manifest")

    rows: list[dict[str, Any]] = []
    for entry in sorted(normalized_entries, key=lambda item: item["record_path"]):
        record_path = benchmark_archive / entry["record_path"]
        try:
            record_bytes = record_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"benchmark record is unreadable: {record_path}: {exc}") from exc
        if hashlib.sha256(record_bytes).hexdigest() != entry["record_sha256"]:
            raise ValueError(
                f"benchmark record SHA256 does not match the manifest: {entry['record_path']}"
            )
        try:
            record = json.loads(record_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"benchmark record is unreadable: {record_path}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"benchmark record root must be an object: {record_path}")
        request = record.get("request")
        if not isinstance(request, dict):
            raise ValueError(f"benchmark record has no request object: {record_path}")
        student_id = str(record.get("student_id") or request.get("student_id") or "")
        session_id = str(
            record.get("session_id")
            or request.get("training_session_id")
            or record_path.parent.name
        )
        if (student_id, session_id) != (
            entry["student_id"],
            entry["session_id"],
        ):
            raise ValueError(
                f"benchmark record identity does not match the manifest: {entry['record_path']}"
            )
        photos = request.get("photos")
        if not student_id or not isinstance(photos, list):
            raise ValueError(f"benchmark record is malformed: {record_path}")
        rows.append(
            {
                "student_id": student_id,
                "session_id": session_id,
                "label": entry["truth"],
                "ref_image_path": str(request.get("ref_image_path", "")),
                "photos": photos,
            }
        )
    provenance: dict[str, str | int] = {
        "benchmark_manifest_schema_version": 1,
        "benchmark_manifest_id": benchmark_id,
        "benchmark_manifest_sha256": actual_manifest_sha,
        "benchmark_manifest_expected_sha256": expected_manifest_sha256,
        "benchmark_manifest_observed_sha256": actual_manifest_sha,
        "benchmark_manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "known_impostor_student_id": known_identity[0],
        "known_impostor_session_id": known_identity[1],
    }
    return rows, provenance


def _truth_reconstruction(rows: list[dict[str, Any]]) -> dict[str, int]:
    implicit = sum(
        "implicit_correct" in set(row.get("training_exclusion_reasons", []))
        for row in rows
    )
    return {
        "explicit_sessions": len(rows) - implicit,
        "implicit_sessions": implicit,
        "match_sessions": sum(_truth(row) == "match" for row in rows),
        "mismatch_sessions": sum(_truth(row) == "mismatch" for row in rows),
    }


def evaluate_release_candidate(
    dataset_manifest: Path,
    adapter_dir: Path,
    benchmark_archive: Path,
    benchmark_manifest_sha256: str,
    *,
    pipeline_factory: Callable[[], Any] = _build_evaluation_pipeline,
    inference_session_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a provenance-bound candidate on unseen students and benchmark data."""
    dataset_manifest = Path(dataset_manifest)
    adapter_dir = Path(adapter_dir)
    benchmark_archive = Path(benchmark_archive)
    dataset = _load_json_object(dataset_manifest, "dataset manifest")
    if dataset.get("schema_version") != 1:
        raise ValueError("dataset manifest schema_version must be 1")
    artifact, onnx_path = _verify_artifact(dataset, adapter_dir)
    leaks = _student_split_leaks(dataset)
    benchmark_rows, benchmark_provenance = _load_benchmark_rows(
        benchmark_archive,
        benchmark_manifest_sha256,
    )
    evaluation_rows = dataset.get("evaluation_sessions", [])
    assert isinstance(evaluation_rows, list)

    if inference_session_factory is None:
        inference_session_factory = _load_onnx_session
    scorer = _AdapterScorer(inference_session_factory(onnx_path), artifact)
    pipeline = pipeline_factory()
    held_out = _evaluate_rows(evaluation_rows, pipeline, scorer)
    raw_confusion = _confusion(held_out, adapted=False)
    adapted_confusion = _confusion(held_out, adapted=True)
    cross_student = _cross_student_metrics(held_out, scorer)

    benchmark = _evaluate_rows(benchmark_rows, pipeline, scorer)
    known = next(
        item
        for item in benchmark
        if (item.student_id, item.session_id)
        == (
            benchmark_provenance["known_impostor_student_id"],
            benchmark_provenance["known_impostor_session_id"],
        )
    )
    new_false_accusations = sum(
        item.truth == "match"
        and item.adapted_status == "mismatch"
        and item.raw_status != "mismatch"
        for item in benchmark
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "model_version": artifact.get("model_version", ""),
        "artifact_sha256": artifact["onnx_sha256"],
        "onnx_sha256_verified": True,
        "dataset_sha256_verified": True,
        "onnx_parity_max_abs_error": artifact.get("onnx_parity_max_abs_error"),
        "student_split_leaks": len(leaks),
        "student_split_leak_ids": leaks,
        "held_out_test_sessions": len(held_out),
        "adapter_eligible_test_sessions": sum(item.adapter_applied for item in held_out),
        "truth_reconstruction": _truth_reconstruction(evaluation_rows),
        "raw_confusion": raw_confusion,
        "adapted_confusion": adapted_confusion,
        "raw_conditional_accuracy": _conditional_accuracy(raw_confusion),
        "conditional_accuracy": _conditional_accuracy(adapted_confusion),
        **cross_student,
        "benchmark_sessions": len(benchmark),
        **benchmark_provenance,
        "benchmark_raw_confusion": _confusion(benchmark, adapted=False),
        "benchmark_adapted_confusion": _confusion(benchmark, adapted=True),
        "known_impostor_detected": (
            known.raw_status == "mismatch" and known.adapted_status == "mismatch"
        ),
        "known_impostor_raw_status": known.raw_status,
        "known_impostor_adapted_status": known.adapted_status,
        "new_false_accusations": new_false_accusations,
    }
    report["release_gate_passed"] = release_gate_passes(report)
    return report


def _load_onnx_session(path: Path) -> Any:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--benchmark-archive", type=Path, required=True)
    parser.add_argument("--benchmark-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = evaluate_release_candidate(
            args.dataset_manifest,
            args.adapter_dir,
            args.benchmark_archive,
            args.benchmark_manifest_sha256,
        )
        report["release_gate_passed"] = release_gate_passes(report)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "evaluation_error": str(exc),
            "cross_student_far": 1.0,
            "conditional_accuracy": 0.0,
            "known_impostor_detected": False,
            "new_false_accusations": 1,
            "student_split_leaks": 1,
            "release_gate_passed": False,
        }
        if isinstance(exc, BenchmarkManifestTrustError):
            report["benchmark_manifest_expected_sha256"] = exc.expected
            report["benchmark_manifest_observed_sha256"] = exc.observed
    _atomic_write_private_json(args.output, report)
    return 0 if release_gate_passes(report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
