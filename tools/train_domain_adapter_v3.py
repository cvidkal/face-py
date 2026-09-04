#!/usr/bin/env python3
"""Fixed-configuration command line for identity domain adapter v3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.domain_adapter_v3_training import (  # noqa: E402
    HistoricalGateError,
    V3TrainingConfig,
    train_v3_candidate,
    write_v3_candidate_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


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


def _failure_report_path(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}.historical-gate-failure.json")


def _write_failure_report_no_overwrite(
    path: Path,
    error: HistoricalGateError,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_version": "identity-domain-adapter-v3",
        "gate": "historical_oof_relative",
        "gate_passed": False,
        "historical_oof_relative": asdict(error.metrics),
    }
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


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
        try:
            result = train_v3_candidate(
                manifest,
                Path(args.embedding_cache),
                V3TrainingConfig(),
                seed=args.seed,
                device=args.device,
            )
        except HistoricalGateError as exc:
            failure_path = _failure_report_path(output_dir)
            _write_failure_report_no_overwrite(failure_path, exc)
            print(
                json.dumps(
                    {
                        "gate_passed": False,
                        "failure_report": str(failure_path),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        artifact = write_v3_candidate_artifacts(output_dir, result, seed=args.seed)
    finally:
        os.umask(previous_umask)
    print(json.dumps(artifact, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
