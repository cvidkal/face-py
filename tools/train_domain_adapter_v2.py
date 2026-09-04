#!/usr/bin/env python3
"""Command-line entry point for immutable v2 identity-domain-adapter artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.domain_adapter_v2_training import (  # noqa: E402
    V2TrainingConfig,
    train_v2_candidate,
    write_v2_candidate_artifacts,
)


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    try:
        manifest = _load_manifest(Path(args.manifest))
    except ValueError as exc:
        _parser().error(str(exc))
    previous_umask = os.umask(0o077)
    try:
        result = train_v2_candidate(
            manifest,
            Path(args.embedding_cache),
            V2TrainingConfig(),
            seed=args.seed,
            device=args.device,
        )
        artifact = write_v2_candidate_artifacts(
            output_dir,
            result,
            seed=args.seed,
        )
    finally:
        os.umask(previous_umask)
    print(json.dumps(artifact, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
