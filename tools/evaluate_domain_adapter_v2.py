#!/usr/bin/env python3
"""Replay-only evaluator for frozen identity-domain-adapter v2 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module.face.pipeline import build_pipeline_from_env  # noqa: E402
from tools.domain_adapter_training import (  # noqa: E402
    SessionEmbedding,
    manifest_training_rows,
)
from tools.domain_adapter_v2_metrics import (  # noqa: E402
    build_v2_pair_set,
    compare_at_far_budget,
)
from tools.evaluate_domain_adapter import (  # noqa: E402
    ARTIFACT_MANIFEST,
    _AdapterScorer,
    _canonical_sha256,
    _conditional_accuracy,
    _confusion,
    _cross_student_metrics,
    _evaluate_rows,
    _load_benchmark_rows,
    _load_json_object,
    _student_split_leaks,
    _truth_reconstruction,
)


ENGINEERING_DATASET_FILE_SHA256 = (
    "5add7c1d80e8e79f38ee1b125ad304e8f89bad6bdca146dd25d0e574870c8a3d"
)
TRUSTED_BENCHMARK_MANIFEST_FILE_SHA256 = (
    "3ab56009c957011ce77b2b71178b068b6879d64badc5b9fcf1a8f074ff13699a"
)
PRIVATE_FILE_MODE = 0o600
RELEASE_REGISTRY_SCHEMA_VERSION = 1
RELEASE_REGISTRY_HASH_SCHEME = "sha256-domain-v1"
RELEASE_STUDENT_HASH_DOMAIN = b"identity-domain-adapter-release-student\0"
SAME_THRESHOLD = 0.35
MIN_RELEASE_UNSEEN_STUDENTS = 50
MIN_RELEASE_ELIGIBLE_MATCH_SESSIONS = 100
MIN_RELEASE_ORDERED_CROSS_STUDENT_PAIRS = 2000


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


def _absolute_gate_passes(metrics: dict[str, Any]) -> bool:
    return (
        metrics["student_split_leaks"] == 0
        and metrics["cross_student_far"] <= 0.01
        and metrics["conditional_accuracy"] >= 0.70
        and metrics["known_impostor_detected"] is True
        and metrics["new_false_accusations"] == 0
        and metrics["onnx_parity_max_abs_error"] <= 0.00001
    )


def _relative_gate_passes(metrics: dict[str, Any]) -> bool:
    same_threshold = metrics["same_threshold_0_35"]
    return (
        metrics["adapted_far"] <= metrics["raw_far"]
        and metrics["recall_lift"] >= 0.02
        and metrics["true_match_delta"] > 0
        and same_threshold["recall_delta"] > 0.0
    )


def v2_gate_passes(report: dict[str, Any]) -> bool:
    if report.get("dataset_role") == "release":
        return bool(report.get("release_gate_passed"))
    return bool(report.get("engineering_gate_passed"))


def evaluate_v2_candidate(
    candidate_dir: Path,
    dataset_manifest: Path,
    benchmark_manifest: Path,
    dataset_role: Literal["engineering", "release"],
    historical_manifest: Path | None = None,
    cohort_registry: Path | None = None,
    *,
    pipeline_factory: Any | None = None,
    inference_session_factory: Any | None = None,
) -> dict[str, Any]:
    candidate_dir = Path(candidate_dir)
    dataset_manifest = Path(dataset_manifest)
    benchmark_manifest = Path(benchmark_manifest)
    if dataset_role == "release" and historical_manifest is None:
        raise ValueError("release evaluation requires historical_manifest")
    if dataset_role == "release" and cohort_registry is None:
        raise ValueError("release evaluation requires cohort_registry")
    if dataset_role == "engineering" and historical_manifest is not None:
        raise ValueError("engineering evaluation must not receive historical_manifest")
    if dataset_role == "engineering" and cohort_registry is not None:
        raise ValueError("engineering evaluation must not receive cohort_registry")

    dataset = _load_json_object(dataset_manifest, "dataset manifest")
    if dataset.get("schema_version") != 1:
        raise ValueError("dataset manifest schema_version must be 1")
    artifact, onnx_path = _load_v2_artifact(candidate_dir)
    benchmark_rows, benchmark_provenance = _load_benchmark_from_manifest(benchmark_manifest)
    evaluation_rows = dataset.get("evaluation_sessions", [])
    if not isinstance(evaluation_rows, list):
        raise ValueError("dataset manifest evaluation_sessions must be a list")

    if pipeline_factory is None:
        pipeline_factory = _build_evaluation_pipeline
    if inference_session_factory is None:
        inference_session_factory = _load_onnx_session
    scorer = _AdapterScorer(inference_session_factory(onnx_path), artifact)
    pipeline = pipeline_factory()
    leaks = _student_split_leaks(dataset)
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
    absolute_metrics = {
        "onnx_parity_max_abs_error": float(artifact["onnx_parity_max_abs_error"]),
        "student_split_leaks": len(leaks),
        "held_out_test_sessions": len(held_out),
        "adapter_eligible_test_sessions": sum(item.adapter_applied for item in held_out),
        "truth_reconstruction": _truth_reconstruction(evaluation_rows),
        "raw_confusion": raw_confusion,
        "adapted_confusion": adapted_confusion,
        "raw_conditional_accuracy": _conditional_accuracy(raw_confusion),
        "conditional_accuracy": _conditional_accuracy(adapted_confusion),
        **cross_student,
        "benchmark_sessions": len(benchmark),
        "benchmark_raw_confusion": _confusion(benchmark, adapted=False),
        "benchmark_adapted_confusion": _confusion(benchmark, adapted=True),
        "known_impostor_detected": (
            known.raw_status == "mismatch" and known.adapted_status == "mismatch"
        ),
        "known_impostor_raw_status": known.raw_status,
        "known_impostor_adapted_status": known.adapted_status,
        "new_false_accusations": new_false_accusations,
    }
    relative_metrics = _relative_metrics(
        held_out=held_out,
        scorer=scorer,
        dataset_digest=_canonical_sha256(dataset),
        raw_threshold=float(artifact["training"]["raw_comparator_threshold"]["threshold"]),
        adapted_threshold=float(artifact["training"]["adapted_threshold"]["threshold"]),
    )
    absolute_gate_passed = _absolute_gate_passes(absolute_metrics)
    relative_gate_passed = _relative_gate_passes(relative_metrics)

    provenance = _provenance_report(
        dataset=dataset,
        dataset_manifest=dataset_manifest,
        dataset_role=dataset_role,
        historical_manifest=Path(historical_manifest) if historical_manifest is not None else None,
        cohort_registry=Path(cohort_registry) if cohort_registry is not None else None,
        candidate_training_digest=str(artifact["source_dataset_sha256"]),
        held_out=held_out,
    )
    engineering_gate_passed = (
        dataset_role == "engineering"
        and provenance["engineering_manifest_sha256_pinned"]
        and provenance["engineering_dataset_role_absent"]
        and absolute_gate_passed
        and relative_gate_passed
    )
    release_gate_passed = (
        dataset_role == "release"
        and provenance["release_manifest_role_matches"]
        and provenance["historical_digest_matches_candidate"]
        and provenance["historical_manifest_matches_release"]
        and provenance["historical_student_overlap_free"]
        and provenance["registry_entry_matches"]
        and provenance["registry_hashes_unique_to_current"]
        and provenance["release_time_bounds_valid"]
        and provenance["release_excluded_overlap_ok"]
        and provenance["release_sufficiency_exact"]
        and provenance["cohort_status"] == "sufficient"
        and absolute_gate_passed
        and relative_gate_passed
    )

    return {
        "schema_version": 2,
        "dataset_role": dataset_role,
        "cohort_status": provenance["cohort_status"],
        "candidate_manifest_file_sha256": _file_sha256(candidate_dir / ARTIFACT_MANIFEST),
        "candidate_onnx_sha256": artifact["onnx_sha256"],
        "candidate_training_dataset_sha256": artifact["source_dataset_sha256"],
        "dataset_manifest_file_sha256": _file_sha256(dataset_manifest),
        "dataset_manifest_canonical_sha256": _canonical_sha256(dataset),
        "benchmark_manifest_sha256": benchmark_provenance["benchmark_manifest_sha256"],
        "benchmark_manifest_file_sha256": benchmark_provenance[
            "benchmark_manifest_file_sha256"
        ],
        "cohort_sufficiency": provenance["cohort_sufficiency"],
        "provenance": {
            "engineering_manifest_sha256_pinned": provenance[
                "engineering_manifest_sha256_pinned"
            ],
            "engineering_dataset_role_absent": provenance[
                "engineering_dataset_role_absent"
            ],
            "release_manifest_role_matches": provenance["release_manifest_role_matches"],
            "historical_digest_matches_candidate": provenance[
                "historical_digest_matches_candidate"
            ],
            "historical_manifest_matches_release": provenance[
                "historical_manifest_matches_release"
            ],
            "historical_student_overlap_free": provenance[
                "historical_student_overlap_free"
            ],
            "registry_entry_matches": provenance["registry_entry_matches"],
            "registry_hashes_unique_to_current": provenance[
                "registry_hashes_unique_to_current"
            ],
            "release_time_bounds_valid": provenance["release_time_bounds_valid"],
            "release_excluded_overlap_ok": provenance["release_excluded_overlap_ok"],
            "release_sufficiency_exact": provenance["release_sufficiency_exact"],
            "cohort_registry_file_sha256": provenance["cohort_registry_file_sha256"],
        },
        "absolute_metrics": absolute_metrics,
        "relative_metrics": relative_metrics,
        "absolute_gate_passed": absolute_gate_passed,
        "relative_gate_passed": relative_gate_passed,
        "engineering_gate_passed": engineering_gate_passed,
        "release_gate_passed": release_gate_passed,
    }


def _load_onnx_session(path: Path) -> Any:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_v2_artifact(candidate_dir: Path) -> tuple[dict[str, Any], Path]:
    artifact = _load_json_object(candidate_dir / ARTIFACT_MANIFEST, "artifact manifest")
    if artifact.get("schema_version") != 1:
        raise ValueError("artifact manifest schema_version must be 1")
    onnx_name = artifact.get("onnx_file")
    if not isinstance(onnx_name, str) or not onnx_name or Path(onnx_name).name != onnx_name:
        raise ValueError("artifact onnx_file must be a file name")
    onnx_path = candidate_dir / onnx_name
    expected_sha = artifact.get("onnx_sha256")
    if not isinstance(expected_sha, str):
        raise ValueError("artifact onnx_sha256 must be a string")
    if _file_sha256(onnx_path) != expected_sha:
        raise ValueError("ONNX SHA256 does not match the artifact manifest")
    training = artifact.get("training")
    if not isinstance(training, dict):
        raise ValueError("artifact training metadata must be an object")
    adapted = training.get("adapted_threshold")
    raw = training.get("raw_comparator_threshold")
    if not isinstance(adapted, dict) or not isinstance(raw, dict):
        raise ValueError("artifact threshold metadata is incomplete")
    adapted_threshold = float(adapted.get("threshold", math.nan))
    raw_threshold = float(raw.get("threshold", math.nan))
    match_threshold = float(artifact.get("match_threshold", math.nan))
    parity = float(artifact.get("onnx_parity_max_abs_error", math.nan))
    dataset_digest = str(
        training.get("canonical_dataset_digest") or artifact.get("source_dataset_sha256") or ""
    )
    if not math.isfinite(adapted_threshold) or not math.isfinite(raw_threshold):
        raise ValueError("artifact thresholds must be finite")
    if not math.isfinite(match_threshold) or match_threshold != adapted_threshold:
        raise ValueError("artifact match_threshold must equal frozen adapted threshold")
    if not math.isfinite(parity):
        raise ValueError("artifact onnx_parity_max_abs_error must be finite")
    if not dataset_digest:
        raise ValueError("artifact source dataset digest is missing")
    artifact["source_dataset_sha256"] = dataset_digest
    return artifact, onnx_path


def _load_benchmark_from_manifest(
    benchmark_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, str | int]]:
    observed_file_sha = _file_sha256(benchmark_manifest)
    if observed_file_sha != TRUSTED_BENCHMARK_MANIFEST_FILE_SHA256:
        raise ValueError(
            "trusted benchmark manifest file SHA256 mismatch "
            f"(expected {TRUSTED_BENCHMARK_MANIFEST_FILE_SHA256}, observed {observed_file_sha})"
        )
    manifest = _load_json_object(benchmark_manifest, "benchmark manifest")
    declared_sha = manifest.get("manifest_sha256")
    if not isinstance(declared_sha, str) or len(declared_sha) != 64:
        raise ValueError("benchmark manifest manifest_sha256 must be 64 hex characters")
    return _load_benchmark_rows(benchmark_manifest.parent, declared_sha)


def _relative_metrics(
    *,
    held_out: list[Any],
    scorer: _AdapterScorer,
    dataset_digest: str,
    raw_threshold: float,
    adapted_threshold: float,
) -> dict[str, Any]:
    positive_sessions = [
        SessionEmbedding(
            student_id=item.student_id,
            session_id=item.session_id,
            split="test",
            label="match",
            ref_embedding=np.asarray(item.ref_embedding, dtype=np.float32),
            session_prototype=np.asarray(item.session_prototype, dtype=np.float32),
        )
        for item in held_out
        if item.adapter_applied
        and item.truth == "match"
        and item.ref_embedding is not None
        and item.session_prototype is not None
    ]
    if len({item.student_id for item in positive_sessions}) < 2:
        return _empty_relative_metrics(raw_threshold, adapted_threshold)
    pair_set = build_v2_pair_set(positive_sessions)
    if pair_set.negative_ref_embeddings.shape[0] == 0:
        return _empty_relative_metrics(raw_threshold, adapted_threshold)

    raw_positive = np.sum(
        np.asarray(pair_set.positive_ref_embeddings, dtype=np.float64)
        * np.asarray(pair_set.positive_photo_embeddings, dtype=np.float64),
        axis=1,
    )
    raw_negative = np.sum(
        np.asarray(pair_set.negative_ref_embeddings, dtype=np.float64)
        * np.asarray(pair_set.negative_photo_embeddings, dtype=np.float64),
        axis=1,
    )
    adapted_positive = scorer.score(
        pair_set.positive_ref_embeddings,
        pair_set.positive_photo_embeddings,
    ).astype(np.float64, copy=False)
    adapted_negative = scorer.score(
        pair_set.negative_ref_embeddings,
        pair_set.negative_photo_embeddings,
    ).astype(np.float64, copy=False)
    budget = compare_at_far_budget(
        raw_positive_scores=raw_positive,
        adapted_positive_scores=adapted_positive,
        positive_student_ids=pair_set.positive_student_ids,
        raw_negative_scores=raw_negative,
        adapted_negative_scores=adapted_negative,
        negative_group_ids=pair_set.negative_group_ids,
        dataset_digest=dataset_digest,
    )
    same_threshold = {
        "threshold": SAME_THRESHOLD,
        "raw_recall": float(budget.same_threshold_raw_recall),
        "adapted_recall": float(budget.same_threshold_adapted_recall),
        "recall_delta": float(
            budget.same_threshold_adapted_recall - budget.same_threshold_raw_recall
        ),
    }
    return {
        "raw_frozen_threshold": raw_threshold,
        "adapted_frozen_threshold": adapted_threshold,
        "raw_far": float(np.mean(raw_negative >= raw_threshold)),
        "adapted_far": float(np.mean(adapted_negative >= adapted_threshold)),
        "far_delta": float(np.mean(adapted_negative >= adapted_threshold))
        - float(np.mean(raw_negative >= raw_threshold)),
        "raw_recall_at_1pct_budget": float(budget.raw.student_balanced_recall),
        "adapted_recall_at_1pct_budget": float(budget.adapted.student_balanced_recall),
        "recall_lift": float(budget.recall_lift),
        "raw_true_matches": int(budget.raw.true_matches),
        "adapted_true_matches": int(budget.adapted.true_matches),
        "true_match_delta": int(budget.true_match_delta),
        "same_threshold_0_35": same_threshold,
        "positive_observations": int(adapted_positive.size),
        "negative_observations": int(adapted_negative.size),
    }


def _empty_relative_metrics(raw_threshold: float, adapted_threshold: float) -> dict[str, Any]:
    return {
        "raw_frozen_threshold": raw_threshold,
        "adapted_frozen_threshold": adapted_threshold,
        "raw_far": 1.0,
        "adapted_far": 1.0,
        "far_delta": 0.0,
        "raw_recall_at_1pct_budget": 0.0,
        "adapted_recall_at_1pct_budget": 0.0,
        "recall_lift": 0.0,
        "raw_true_matches": 0,
        "adapted_true_matches": 0,
        "true_match_delta": 0,
        "same_threshold_0_35": {
            "threshold": SAME_THRESHOLD,
            "raw_recall": 0.0,
            "adapted_recall": 0.0,
            "recall_delta": 0.0,
        },
        "positive_observations": 0,
        "negative_observations": 0,
    }


def _provenance_report(
    *,
    dataset: dict[str, Any],
    dataset_manifest: Path,
    dataset_role: str,
    historical_manifest: Path | None,
    cohort_registry: Path | None,
    candidate_training_digest: str,
    held_out: list[Any],
) -> dict[str, Any]:
    engineering_dataset_role_absent = "dataset_role" not in dataset
    engineering_manifest_sha256_pinned = (
        _file_sha256(dataset_manifest) == ENGINEERING_DATASET_FILE_SHA256
    )
    release_manifest_role_matches = dataset.get("dataset_role") == "release"
    historical_digest_matches_candidate = False
    historical_manifest_matches_release = False
    historical_student_overlap_free = False
    excluded_hashes: set[str] = set()
    current_hashes = _current_cohort_student_hashes(dataset)
    if dataset_role == "release" and historical_manifest is not None:
        historical = _load_json_object(historical_manifest, "historical manifest")
        historical_digest = _canonical_sha256(historical)
        historical_digest_matches_candidate = (
            _canonical_training_dataset_sha256(historical)
            == candidate_training_digest
        )
        historical_manifest_matches_release = (
            str(dataset.get("historical_manifest_digest", "")) == historical_digest
        )
        historical_hashes = _historical_student_hashes(historical)
        historical_student_overlap_free = not bool(current_hashes & historical_hashes)
        excluded_hashes |= historical_hashes
    release_time_bounds_valid = _release_time_bounds_valid(dataset)
    registry_entry_matches = False
    registry_hashes_unique_to_current = False
    if dataset_role == "release" and cohort_registry is not None:
        registry_entry_matches, registry_hashes_unique_to_current, other_registry_hashes = (
            _registry_checks(dataset, cohort_registry)
        )
        excluded_hashes |= other_registry_hashes
    manifest_seen_overlap_ok = _release_manifest_seen_overlap_count_valid(dataset)
    release_excluded_overlap_ok = (
        manifest_seen_overlap_ok
        and historical_student_overlap_free
        and registry_hashes_unique_to_current
        if dataset_role == "release"
        else True
    )
    cohort_sufficiency = _cohort_sufficiency(
        dataset_role,
        held_out,
        excluded_hashes=excluded_hashes,
        overlap_or_anomaly=not release_excluded_overlap_ok,
    )
    return {
        "engineering_manifest_sha256_pinned": engineering_manifest_sha256_pinned,
        "engineering_dataset_role_absent": engineering_dataset_role_absent,
        "release_manifest_role_matches": release_manifest_role_matches,
        "historical_digest_matches_candidate": historical_digest_matches_candidate,
        "historical_manifest_matches_release": historical_manifest_matches_release,
        "historical_student_overlap_free": historical_student_overlap_free,
        "registry_entry_matches": registry_entry_matches,
        "registry_hashes_unique_to_current": registry_hashes_unique_to_current,
        "release_time_bounds_valid": release_time_bounds_valid,
        "release_excluded_overlap_ok": release_excluded_overlap_ok,
        "release_sufficiency_exact": cohort_sufficiency["exact_minima"],
        "cohort_status": cohort_sufficiency["status"],
        "cohort_sufficiency": cohort_sufficiency["summary"],
        "cohort_registry_file_sha256": (
            _file_sha256(cohort_registry) if cohort_registry is not None else None
        ),
    }


def _cohort_sufficiency(
    dataset_role: str,
    held_out: list[Any],
    *,
    excluded_hashes: set[str],
    overlap_or_anomaly: bool,
) -> dict[str, Any]:
    if dataset_role != "release":
        return {
            "status": "not_applicable",
            "exact_minima": False,
            "summary": None,
        }

    eligible_sessions = [
        session
        for session in _deduped_eligible_release_sessions(held_out)
        if _student_hash(session.student_id) not in excluded_hashes
    ]
    synthetic_pairs = 0
    if len({session.student_id for session in eligible_sessions}) >= 2:
        pair_set = build_v2_pair_set(eligible_sessions)
        synthetic_pairs = sum(
            category == "synthetic_cross_student"
            for category in pair_set.negative_categories
        )
    summary = {
        "unseen_students": len({session.student_id for session in eligible_sessions}),
        "minimum_unseen_students": MIN_RELEASE_UNSEEN_STUDENTS,
        "adapter_eligible_truth_match_sessions": len(eligible_sessions),
        "minimum_adapter_eligible_truth_match_sessions": MIN_RELEASE_ELIGIBLE_MATCH_SESSIONS,
        "ordered_cross_student_session_pairs": synthetic_pairs,
        "minimum_ordered_cross_student_session_pairs": MIN_RELEASE_ORDERED_CROSS_STUDENT_PAIRS,
    }
    exact_minima = (
        summary["minimum_unseen_students"] == MIN_RELEASE_UNSEEN_STUDENTS
        and summary["minimum_adapter_eligible_truth_match_sessions"]
        == MIN_RELEASE_ELIGIBLE_MATCH_SESSIONS
        and summary["minimum_ordered_cross_student_session_pairs"]
        == MIN_RELEASE_ORDERED_CROSS_STUDENT_PAIRS
    )
    return {
        "status": (
            "sufficient"
            if summary["unseen_students"] >= MIN_RELEASE_UNSEEN_STUDENTS
            and summary["adapter_eligible_truth_match_sessions"]
            >= MIN_RELEASE_ELIGIBLE_MATCH_SESSIONS
            and summary["ordered_cross_student_session_pairs"]
            >= MIN_RELEASE_ORDERED_CROSS_STUDENT_PAIRS
            and not overlap_or_anomaly
            else "insufficient_data"
        ),
        "exact_minima": exact_minima,
        "summary": summary,
    }


def _deduped_eligible_release_sessions(held_out: list[Any]) -> tuple[SessionEmbedding, ...]:
    sessions: dict[tuple[str, str], tuple[tuple[Any, ...], SessionEmbedding]] = {}
    for item in held_out:
        if (
            item.truth != "match"
            or not item.adapter_applied
            or item.ref_embedding is None
            or item.session_prototype is None
        ):
            continue
        key = (item.student_id, item.session_id)
        session = SessionEmbedding(
            student_id=item.student_id,
            session_id=item.session_id,
            split="test",
            label="match",
            ref_embedding=np.asarray(item.ref_embedding, dtype=np.float32),
            session_prototype=np.asarray(item.session_prototype, dtype=np.float32),
        )
        signature = (
            item.truth,
            item.raw_status,
            item.adapted_status,
            item.internal_consistency,
            np.asarray(item.ref_embedding, dtype=np.float32).tobytes(),
            np.asarray(item.session_prototype, dtype=np.float32).tobytes(),
        )
        previous = sessions.get(key)
        if previous is None:
            sessions[key] = (signature, session)
            continue
        if previous[0] != signature:
            raise ValueError(
                "conflicting duplicate release observation "
                f"for {item.student_id}/{item.session_id}"
            )
    return tuple(item[1] for item in sessions.values())


def _current_cohort_student_hashes(dataset: dict[str, Any]) -> set[str]:
    rows = dataset.get("evaluation_sessions", [])
    if not isinstance(rows, list):
        return set()
    return {
        _student_hash(str(row.get("student_id", "")))
        for row in rows
        if isinstance(row, dict) and str(row.get("student_id", ""))
    }


def _historical_student_hashes(historical: dict[str, Any]) -> set[str]:
    if historical.get("schema_version") != 1:
        raise ValueError("historical manifest schema_version must be 1")
    hashes: set[str] = set()
    for key in ("sessions", "evaluation_sessions"):
        rows = historical.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"historical manifest {key} must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"historical manifest {key} rows must be objects")
            student_id = str(row.get("student_id", ""))
            if not student_id:
                raise ValueError(f"historical manifest {key} rows require student_id")
            hashes.add(_student_hash(student_id))
    return hashes


def _registry_checks(
    dataset: dict[str, Any],
    cohort_registry: Path,
) -> tuple[bool, bool, set[str]]:
    registry = _load_json_object(cohort_registry, "cohort registry")
    if registry.get("schema_version") != RELEASE_REGISTRY_SCHEMA_VERSION:
        raise ValueError("cohort registry schema_version must be 1")
    if registry.get("hash_scheme") != RELEASE_REGISTRY_HASH_SCHEME:
        raise ValueError("cohort registry hash_scheme is invalid")
    cohorts = registry.get("cohorts")
    if not isinstance(cohorts, list):
        raise ValueError("cohort registry cohorts must be a list")

    expected_hashes = sorted(_current_cohort_student_hashes(dataset))
    current_exact = False
    current_match_count = 0
    other_hashes: set[str] = set()
    reused = False
    for cohort in cohorts:
        if not isinstance(cohort, dict):
            raise ValueError("cohort registry cohort entries must be objects")
        cohort_hashes_raw = cohort.get("student_hashes")
        if not isinstance(cohort_hashes_raw, list):
            raise ValueError("cohort registry cohort student_hashes must be a list")
        cohort_hashes = {str(value) for value in cohort_hashes_raw}
        is_current = (
            cohort.get("cohort_id") == dataset.get("cohort_id")
            and cohort.get("after") == dataset.get("after")
            and cohort.get("through") == dataset.get("through")
        )
        if is_current:
            current_match_count += 1
            current_exact = sorted(cohort_hashes) == expected_hashes
        else:
            if cohort_hashes & set(expected_hashes):
                reused = True
            other_hashes |= cohort_hashes
    return current_match_count == 1 and current_exact, not reused, other_hashes


def _canonical_training_dataset_sha256(manifest: dict[str, Any]) -> str:
    source = {
        "schema_version": manifest.get("schema_version"),
        "snapshot": manifest.get("snapshot", ""),
        "split_seed": manifest.get("split_seed", ""),
        "sessions": manifest.get("sessions", []),
        "evaluation_sessions": [],
    }
    rows = [
        dict(row)
        for row in manifest_training_rows(source)
        if row["split"] in {"train", "validation"}
    ]
    return _canonical_sha256(
        {
            "schema_version": 1,
            "snapshot": source["snapshot"],
            "split_seed": source["split_seed"],
            "sessions": rows,
            "evaluation_sessions": [],
        }
    )


def _release_manifest_seen_overlap_count_valid(dataset: dict[str, Any]) -> bool:
    counts = dataset.get("counts")
    if not isinstance(counts, dict):
        return False
    excluded = counts.get("excluded_by_reason", {})
    if not isinstance(excluded, dict):
        return False
    value = excluded.get("seen_student_overlap", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return False
    return True


def _release_time_bounds_valid(dataset: dict[str, Any]) -> bool:
    try:
        after = _parse_timestamp(str(dataset.get("after", "")))
        through = _parse_timestamp(str(dataset.get("through", "")))
    except ValueError:
        return False
    return after < through and dataset.get("sessions") == []


def _parse_timestamp(raw: str) -> datetime:
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed


def _registry_entry_matches(dataset: dict[str, Any], cohort_registry: Path) -> bool:
    try:
        registry = _load_json_object(cohort_registry, "cohort registry")
    except ValueError:
        return False
    if registry.get("schema_version") != RELEASE_REGISTRY_SCHEMA_VERSION:
        return False
    if registry.get("hash_scheme") != RELEASE_REGISTRY_HASH_SCHEME:
        return False
    expected_hashes = sorted(
        {
            _student_hash(str(row.get("student_id", "")))
            for row in dataset.get("evaluation_sessions", [])
            if isinstance(row, dict) and str(row.get("student_id", ""))
        }
    )
    cohorts = registry.get("cohorts")
    if not isinstance(cohorts, list):
        return False
    for cohort in cohorts:
        if not isinstance(cohort, dict):
            continue
        if (
            cohort.get("cohort_id") == dataset.get("cohort_id")
            and cohort.get("after") == dataset.get("after")
            and cohort.get("through") == dataset.get("through")
            and sorted(str(item) for item in cohort.get("student_hashes", [])) == expected_hashes
        ):
            return True
    return False


def _student_hash(student_id: str) -> str:
    return hashlib.sha256(RELEASE_STUDENT_HASH_DOMAIN + student_id.encode("utf-8")).hexdigest()


def _write_private_report_no_overwrite(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, PRIVATE_FILE_MODE)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _failure_report(dataset_role: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset_role": dataset_role,
        "cohort_status": "evaluation_error",
        "cohort_sufficiency": None,
        "provenance": {},
        "absolute_metrics": {},
        "relative_metrics": _empty_relative_metrics(SAME_THRESHOLD, SAME_THRESHOLD),
        "absolute_gate_passed": False,
        "relative_gate_passed": False,
        "engineering_gate_passed": False,
        "release_gate_passed": False,
        "evaluation_error": message,
    }


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "required evaluation input is missing"
    if isinstance(exc, ValueError):
        return "candidate evaluation failed"
    return f"{type(exc).__name__}: evaluation failed"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-role",
        choices=("engineering", "release"),
        required=True,
    )
    parser.add_argument("--cohort-registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.dataset_role == "release" and args.historical_manifest is None:
        parser.error("--historical-manifest is required for release")
    if args.dataset_role == "release" and args.cohort_registry is None:
        parser.error("--cohort-registry is required for release")
    if args.dataset_role == "engineering" and args.historical_manifest is not None:
        parser.error("--historical-manifest must not be set for engineering")
    if args.dataset_role == "engineering" and args.cohort_registry is not None:
        parser.error("--cohort-registry must not be set for engineering")

    try:
        report = evaluate_v2_candidate(
            args.candidate_dir,
            args.dataset_manifest,
            args.benchmark_manifest,
            dataset_role=args.dataset_role,
            historical_manifest=args.historical_manifest,
            cohort_registry=args.cohort_registry,
        )
    except Exception as exc:
        report = _failure_report(args.dataset_role, _safe_error_message(exc))
    _write_private_report_no_overwrite(args.output, report)
    return 0 if v2_gate_passes(report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
