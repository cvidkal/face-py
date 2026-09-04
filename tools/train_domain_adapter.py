#!/usr/bin/env python3
"""Command-line entry point for training a private identity domain adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.domain_adapter_training import (  # noqa: E402
    build_pair_sets,
    extract_session_embeddings,
    train_adapter,
    write_candidate_artifacts,
)


def _source_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"dataset manifest does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"dataset manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest root must be an object")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--positive-margin", type=float, default=0.35)
    parser.add_argument("--negative-margin", type=float, default=0.15)
    parser.add_argument("--identity-regularization-weight", type=float, default=1e-3)
    parser.add_argument("--max-false-accept-rate", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--model-version", default="identity-domain-adapter-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output_dir: Path = args.output_dir
    if output_dir.exists():
        if not output_dir.is_dir():
            parser.error(f"--output-dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            parser.error(
                f"refusing to overwrite non-empty output directory: {output_dir}"
            )

    try:
        manifest = _load_manifest(args.dataset_manifest)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    previous_umask = os.umask(0o077)
    try:
        sessions = extract_session_embeddings(
            manifest,
            cache_path=output_dir / ".embedding-cache.npz",
        )
        pair_sets = build_pair_sets(sessions)
        if len(pair_sets["train"]) == 0:
            parser.error("dataset produced no train pairs")
        if len(pair_sets["validation"]) == 0:
            parser.error("dataset produced no validation pairs")
        test_pairs = pair_sets["test"] if len(pair_sets["test"]) else None
        hyperparameters = {
            "rank": args.rank,
            "epochs": args.epochs,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "positive_margin": args.positive_margin,
            "negative_margin": args.negative_margin,
            "identity_regularization_weight": args.identity_regularization_weight,
            "max_false_accept_rate": args.max_false_accept_rate,
            "early_stopping_patience": args.early_stopping_patience,
            "max_train_negatives_per_positive": 20,
        }
        result = train_adapter(
            pair_sets["train"],
            pair_sets["validation"],
            test_pairs,
            dimension=128,
            rank=args.rank,
            epochs=args.epochs,
            seed=args.seed,
            learning_rate=args.learning_rate,
            positive_margin=args.positive_margin,
            negative_margin=args.negative_margin,
            identity_regularization_weight=args.identity_regularization_weight,
            max_false_accept_rate=args.max_false_accept_rate,
            early_stopping_patience=args.early_stopping_patience,
        )
        artifact = write_candidate_artifacts(
            result,
            pair_sets,
            manifest,
            output_dir,
            hyperparameters=hyperparameters,
            source_revision=_source_revision(),
            model_version=args.model_version,
        )
    finally:
        os.umask(previous_umask)
    print(json.dumps(artifact, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
