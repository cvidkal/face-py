"""Safe optional runtime for the identity domain-adapter ONNX artifact."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import onnxruntime as ort


_ALLOWED_MODES = frozenset({"off", "shadow", "active"})
_SCHEMA_VERSION = 1
_EMBEDDING_DIMENSION = 128
_INPUT_NAMES = ("ref_embedding", "photo_embedding")
_OUTPUT_NAME = "adapted_cosine"


@dataclass(frozen=True)
class AdapterDecision:
    """A candidate adapter decision; unusable decisions leave raw scoring authoritative."""

    usable: bool
    cosine: float | None = None
    status: str = ""
    error: str = ""


@dataclass
class DomainAdapterRuntime:
    """Validated ONNX adapter isolated behind startup and request fail-open behavior."""

    mode: str = "off"
    ready: bool = False
    version: str = ""
    artifact_sha256: str = ""
    match_threshold: float = 0.0
    load_error: str = ""
    _session: Any = field(default=None, repr=False, compare=False)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    @classmethod
    def load(cls, mode: str, artifact_path: Path) -> "DomainAdapterRuntime":
        """Load a verified artifact, retaining raw behavior for every artifact failure.

        Invalid modes are intentionally not swallowed: they are a deployment configuration
        mistake, unlike a mounted artifact becoming unavailable or corrupt.
        """
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in _ALLOWED_MODES:
            raise ValueError(
                "FACE_DOMAIN_ADAPTER_MODE must be one of off, shadow, active"
            )
        if normalized_mode == "off":
            return cls(mode="off")

        try:
            onnx_path = Path(artifact_path)
            manifest = _read_manifest(_manifest_path_for(onnx_path))
            version, sha256, threshold = _validate_manifest(manifest, onnx_path)
            _verify_checksum(onnx_path, sha256)
            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            _validate_graph_io(session)
        except Exception as exc:  # Artifact and provider failures must not stop face-py.
            return cls(mode=normalized_mode, load_error=_safe_load_error(exc))

        return cls(
            mode=normalized_mode,
            ready=True,
            version=version,
            artifact_sha256=sha256,
            match_threshold=threshold,
            _session=session,
        )

    def compare(self, ref: np.ndarray, photo: np.ndarray) -> AdapterDecision:
        """Score a reference/session-prototype pair, returning unusable on every failure."""
        if not self.ready or self._session is None:
            return AdapterDecision(
                usable=False,
                error=self.load_error,
            )

        try:
            ref_input = _standardize_embedding(ref, "reference")
            photo_input = _standardize_embedding(photo, "photo")
            with self._lock:
                output = self._session.run(
                    [_OUTPUT_NAME],
                    {
                        _INPUT_NAMES[0]: ref_input,
                        _INPUT_NAMES[1]: photo_input,
                    },
                )
            cosine = _read_score(output)
        except Exception as exc:  # A per-request provider failure must preserve raw scoring.
            return AdapterDecision(usable=False, error=_safe_inference_error(exc))

        return AdapterDecision(
            usable=True,
            cosine=cosine,
            status="match" if cosine >= self.match_threshold else "mismatch",
        )


def load_domain_adapter_from_env() -> DomainAdapterRuntime:
    """Build the optional runtime without allowing artifact failures to abort startup."""
    mode = os.environ.get("FACE_DOMAIN_ADAPTER_MODE", "off").strip().lower() or "off"
    path = Path(os.environ.get("FACE_DOMAIN_ADAPTER_PATH", "").strip())
    return DomainAdapterRuntime.load(mode, path)


def _manifest_path_for(onnx_path: Path) -> Path:
    if onnx_path.name == "identity_domain_adapter.onnx":
        return onnx_path.with_name("identity_domain_adapter.manifest.json")
    return onnx_path.with_suffix(".manifest.json")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("adapter manifest is missing") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("adapter manifest is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("adapter manifest root must be an object")
    return payload


def _validate_manifest(
    manifest: dict[str, Any], onnx_path: Path
) -> tuple[str, str, float]:
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("adapter manifest schema version is unsupported")

    version = manifest.get("model_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("adapter manifest model version is invalid")
    if manifest.get("embedding_dimension") != _EMBEDDING_DIMENSION:
        raise ValueError("adapter manifest embedding dimension is invalid")

    rank = manifest.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= _EMBEDDING_DIMENSION:
        raise ValueError("adapter manifest rank is invalid")

    threshold = manifest.get("match_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not -1.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("adapter manifest match threshold is invalid")

    declared_file = manifest.get("onnx_file")
    if declared_file != onnx_path.name:
        raise ValueError("adapter manifest ONNX file is invalid")
    sha256 = manifest.get("onnx_sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("adapter manifest checksum is invalid")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ValueError("adapter manifest checksum is invalid") from exc

    if manifest.get("input_names") != list(_INPUT_NAMES):
        raise ValueError("adapter manifest input names are invalid")
    if manifest.get("output_name") != _OUTPUT_NAME:
        raise ValueError("adapter manifest output name is invalid")
    return version.strip(), sha256.lower(), float(threshold)


def _verify_checksum(onnx_path: Path, expected: str) -> None:
    try:
        actual = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("adapter ONNX artifact is unreadable") from exc
    if actual != expected:
        raise ValueError("adapter ONNX checksum mismatch")


def _validate_graph_io(session: Any) -> None:
    try:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
    except Exception as exc:
        raise ValueError("adapter graph metadata is unreadable") from exc

    if len(inputs) != 2 or tuple(item.name for item in inputs) != _INPUT_NAMES:
        raise ValueError("adapter graph input contract is invalid")
    if len(outputs) != 1 or outputs[0].name != _OUTPUT_NAME:
        raise ValueError("adapter graph output contract is invalid")
    for item in inputs:
        if not _is_embedding_shape(getattr(item, "shape", None)):
            raise ValueError("adapter graph input shape is invalid")
    if not _is_score_shape(getattr(outputs[0], "shape", None)):
        raise ValueError("adapter graph output shape is invalid")


def _is_embedding_shape(shape: Any) -> bool:
    return (
        isinstance(shape, (list, tuple))
        and len(shape) == 2
        and _is_dynamic_batch_dimension(shape[0])
        and shape[1] == _EMBEDDING_DIMENSION
    )


def _is_score_shape(shape: Any) -> bool:
    return (
        isinstance(shape, (list, tuple))
        and len(shape) == 1
        and _is_dynamic_batch_dimension(shape[0])
    )


def _is_dynamic_batch_dimension(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _standardize_embedding(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (_EMBEDDING_DIMENSION,):
        array = array.reshape(1, _EMBEDDING_DIMENSION)
    elif array.shape != (1, _EMBEDDING_DIMENSION):
        raise ValueError(f"adapter {name} embedding shape is invalid")
    if not np.isfinite(array).all():
        raise ValueError(f"adapter {name} embedding is non-finite")
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"adapter {name} embedding norm is invalid")
    return np.ascontiguousarray(array / norm, dtype=np.float32)


def _read_score(output: Any) -> float:
    if not isinstance(output, (list, tuple)) or len(output) != 1:
        raise ValueError("adapter inference output is invalid")
    score = np.asarray(output[0], dtype=np.float32)
    if score.shape != (1,):
        raise ValueError("adapter inference output shape is invalid")
    value = float(score[0])
    if not math.isfinite(value):
        raise ValueError("adapter inference output is non-finite")
    return value


def _safe_load_error(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return str(exc)
    return f"adapter load failed: {type(exc).__name__}"


def _safe_inference_error(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return str(exc)
    return f"adapter inference failed: {type(exc).__name__}"
