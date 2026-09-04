#!/usr/bin/env python3
"""Authority-separated evaluator for identity-domain-adapter v3 candidates."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.domain_adapter_v3_metrics import (  # noqa: E402
    HistoricalRelativeMetrics,
    historical_relative_gate_passes,
)
from tools.evaluate_domain_adapter_v2 import (  # noqa: E402
    SAME_THRESHOLD,
    _empty_relative_metrics,
    _load_v2_artifact,
    _safe_error_message,
    _write_private_report_no_overwrite,
    evaluate_v2_candidate,
)


def _require_v3_candidate(candidate_dir: Path) -> HistoricalRelativeMetrics:
    artifact, _onnx_path = _load_v2_artifact(Path(candidate_dir))
    if artifact.get("model_version") != "identity-domain-adapter-v3":
        raise ValueError("candidate model_version must be identity-domain-adapter-v3")
    training = artifact.get("training")
    if not isinstance(training, dict):
        raise ValueError("candidate training metadata is missing")
    raw_metrics = training.get("historical_oof_relative")
    if not isinstance(raw_metrics, dict):
        raise ValueError("candidate historical OOF metrics are missing")
    true_match_delta = raw_metrics.get("true_match_delta")
    if isinstance(true_match_delta, bool) or not isinstance(true_match_delta, int):
        raise ValueError("candidate historical OOF true_match_delta must be an integer")
    try:
        metrics = HistoricalRelativeMetrics(
            raw_far=float(raw_metrics["raw_far"]),
            adapted_far=float(raw_metrics["adapted_far"]),
            recall_lift=float(raw_metrics["recall_lift"]),
            true_match_delta=true_match_delta,
            same_threshold_recall_delta=float(
                raw_metrics["same_threshold_recall_delta"]
            ),
            adapted_threshold=float(raw_metrics["adapted_threshold"]),
            raw_threshold=float(raw_metrics.get("raw_threshold", 0.35)),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("candidate historical OOF metrics are invalid") from exc
    numeric_values = (
        metrics.raw_far,
        metrics.adapted_far,
        metrics.recall_lift,
        metrics.same_threshold_recall_delta,
        metrics.adapted_threshold,
        metrics.raw_threshold,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("candidate historical OOF metrics must be finite")
    if not historical_relative_gate_passes(metrics):
        raise ValueError("candidate historical OOF gate did not pass")
    return metrics


def evaluate_v3_candidate(
    candidate_dir: Path,
    dataset_manifest: Path,
    benchmark_manifest: Path,
    dataset_role: Literal["legacy", "release"],
    historical_manifest: Path | None = None,
    cohort_registry: Path | None = None,
    *,
    pipeline_factory: Any | None = None,
    inference_session_factory: Any | None = None,
) -> dict[str, Any]:
    if dataset_role == "release" and historical_manifest is None:
        raise ValueError("release evaluation requires historical_manifest")
    if dataset_role == "release" and cohort_registry is None:
        raise ValueError("release evaluation requires cohort_registry")
    if dataset_role == "legacy" and (
        historical_manifest is not None or cohort_registry is not None
    ):
        raise ValueError("legacy evaluation must not receive release provenance inputs")
    if dataset_role not in {"legacy", "release"}:
        raise ValueError("dataset_role must be legacy or release")

    _require_v3_candidate(Path(candidate_dir))
    v2_role: Literal["engineering", "release"] = (
        "engineering" if dataset_role == "legacy" else "release"
    )
    report = evaluate_v2_candidate(
        Path(candidate_dir),
        Path(dataset_manifest),
        Path(benchmark_manifest),
        dataset_role=v2_role,
        historical_manifest=(
            Path(historical_manifest) if historical_manifest is not None else None
        ),
        cohort_registry=Path(cohort_registry) if cohort_registry is not None else None,
        pipeline_factory=pipeline_factory,
        inference_session_factory=inference_session_factory,
    )
    result = dict(report)
    provenance = report.get("provenance", {})
    legacy_safety_gate_passed = (
        dataset_role == "legacy"
        and isinstance(provenance, dict)
        and provenance.get("engineering_manifest_sha256_pinned") is True
        and provenance.get("engineering_dataset_role_absent") is True
        and report.get("absolute_gate_passed") is True
    )
    release_gate_passed = (
        dataset_role == "release" and report.get("release_gate_passed") is True
    )
    result.pop("engineering_gate_passed", None)
    result.update(
        {
            "schema_version": 3,
            "dataset_role": dataset_role,
            "historical_oof_gate_passed": True,
            "legacy_safety_gate_passed": legacy_safety_gate_passed,
            "release_gate_passed": release_gate_passed,
        }
    )
    return result


def v3_gate_passes(report: dict[str, Any]) -> bool:
    if report.get("dataset_role") == "release":
        return report.get("release_gate_passed") is True
    if report.get("dataset_role") == "legacy":
        return report.get("legacy_safety_gate_passed") is True
    return False


def _failure_report(dataset_role: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "dataset_role": dataset_role,
        "cohort_status": "evaluation_error",
        "cohort_sufficiency": None,
        "provenance": {},
        "absolute_metrics": {},
        "relative_metrics": _empty_relative_metrics(SAME_THRESHOLD, SAME_THRESHOLD),
        "absolute_gate_passed": False,
        "relative_gate_passed": False,
        "historical_oof_gate_passed": False,
        "legacy_safety_gate_passed": False,
        "release_gate_passed": False,
        "evaluation_error": message,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--dataset-role", choices=("legacy", "release"), required=True)
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
    if args.dataset_role == "legacy" and args.historical_manifest is not None:
        parser.error("--historical-manifest must not be set for legacy")
    if args.dataset_role == "legacy" and args.cohort_registry is not None:
        parser.error("--cohort-registry must not be set for legacy")

    try:
        report = evaluate_v3_candidate(
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
    return 0 if v3_gate_passes(report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
