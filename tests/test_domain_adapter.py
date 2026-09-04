"""Contract tests for the optional identity domain-adapter runtime."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from module.face.domain_adapter import DomainAdapterRuntime, load_domain_adapter_from_env


class _AdapterSession:
    def __init__(
        self,
        output: object = np.asarray([0.75], dtype=np.float32),
        *,
        input_names: tuple[str, str] = ("ref_embedding", "photo_embedding"),
        input_shapes: tuple[object, object] = (["N", 128], ["N", 128]),
        input_types: tuple[str, str] = ("tensor(float)", "tensor(float)"),
        output_name: str = "adapted_cosine",
        output_shape: object = ["N"],
        output_type: str = "tensor(float)",
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self._inputs = [
            SimpleNamespace(name=name, shape=shape, type=value_type)
            for name, shape, value_type in zip(input_names, input_shapes, input_types)
        ]
        self._outputs = [
            SimpleNamespace(name=output_name, shape=output_shape, type=output_type)
        ]
        self.last_inputs: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[SimpleNamespace]:
        return self._inputs

    def get_outputs(self) -> list[SimpleNamespace]:
        return self._outputs

    def run(self, outputs: list[str], inputs: dict[str, np.ndarray]) -> list[object]:
        if self.error is not None:
            raise self.error
        self.last_inputs = inputs
        if outputs != ["adapted_cosine"]:
            raise AssertionError(outputs)
        return [self.output]


class DomainAdapterRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ref = np.zeros(128, dtype=np.float32)
        self.ref[0] = 1.0
        self.photo = np.zeros(128, dtype=np.float32)
        self.photo[1] = 1.0

    def test_off_mode_never_loads_artifact(self) -> None:
        runtime = DomainAdapterRuntime.load("off", Path("missing.onnx"))

        self.assertFalse(runtime.ready)
        self.assertEqual(runtime.mode, "off")
        self.assertEqual(runtime.load_error, "")

    def test_off_mode_reports_no_adapter_decision_without_an_error(self) -> None:
        decision = DomainAdapterRuntime.load("off", Path("missing.onnx")).compare(
            self.ref, self.photo
        )

        self.assertFalse(decision.usable)
        self.assertEqual(decision.error, "")

    def test_invalid_mode_is_a_startup_configuration_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "FACE_DOMAIN_ADAPTER_MODE"):
            DomainAdapterRuntime.load("audit", Path("ignored.onnx"))

    def test_checksum_mismatch_falls_back_without_creating_a_session(self) -> None:
        with self._artifact(onnx_bytes=b"corrupt", onnx_sha256="0" * 64) as onnx_path:
            with patch("module.face.domain_adapter.ort.InferenceSession") as factory:
                runtime = DomainAdapterRuntime.load("shadow", onnx_path)

        self.assertFalse(runtime.ready)
        self.assertEqual(runtime.load_error, "adapter_artifact_invalid")
        factory.assert_not_called()

    def test_manifest_contract_is_validated_before_session_creation(self) -> None:
        cases = {
            "schema": {"schema_version": 2},
            "version": {"model_version": ""},
            "dimension": {"embedding_dimension": 127},
            "rank": {"rank": 0},
            "threshold": {"match_threshold": float("nan")},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name), self._artifact(**overrides) as onnx_path:
                with patch("module.face.domain_adapter.ort.InferenceSession") as factory:
                    runtime = DomainAdapterRuntime.load("active", onnx_path)
            self.assertFalse(runtime.ready)
            self.assertNotEqual(runtime.load_error, "")
            factory.assert_not_called()

    def test_graph_io_contract_is_validated(self) -> None:
        session = _AdapterSession(input_names=("wrong", "photo_embedding"))
        with self._artifact() as onnx_path:
            with patch("module.face.domain_adapter.ort.InferenceSession", return_value=session):
                runtime = DomainAdapterRuntime.load("active", onnx_path)

        self.assertFalse(runtime.ready)
        self.assertEqual(runtime.load_error, "adapter_artifact_invalid")

    def test_graph_shape_contract_is_validated_before_runtime_is_ready(self) -> None:
        session = _AdapterSession(input_shapes=(["N", 127], ["N", 128]))
        with self._artifact() as onnx_path:
            with patch("module.face.domain_adapter.ort.InferenceSession", return_value=session):
                runtime = DomainAdapterRuntime.load("active", onnx_path)

        self.assertFalse(runtime.ready)
        self.assertFalse(runtime.compare(self.ref, self.photo).usable)

    def test_graph_dtype_contract_is_validated_before_runtime_is_ready(self) -> None:
        session = _AdapterSession(input_types=("tensor(int64)", "tensor(float)"))
        with self._artifact() as onnx_path:
            with patch("module.face.domain_adapter.ort.InferenceSession", return_value=session):
                runtime = DomainAdapterRuntime.load("active", onnx_path)

        self.assertFalse(runtime.ready)
        self.assertEqual(runtime.load_error, "adapter_artifact_invalid")
        self.assertFalse(runtime.compare(self.ref, self.photo).usable)

    def test_graph_output_dtype_contract_is_validated_before_runtime_is_ready(self) -> None:
        session = _AdapterSession(output_type="tensor(double)")
        with self._artifact() as onnx_path:
            with patch("module.face.domain_adapter.ort.InferenceSession", return_value=session):
                runtime = DomainAdapterRuntime.load("active", onnx_path)

        self.assertFalse(runtime.ready)
        self.assertEqual(runtime.load_error, "adapter_artifact_invalid")

    def test_load_error_redacts_provider_exception_details(self) -> None:
        sentinel = "SENTINEL_SECRET /models/private/provider"
        with self._artifact() as onnx_path:
            with patch(
                "module.face.domain_adapter.ort.InferenceSession",
                side_effect=ValueError(sentinel),
            ):
                runtime = DomainAdapterRuntime.load("active", onnx_path)

        self.assertFalse(runtime.ready)
        self.assertEqual(runtime.load_error, "adapter_load_failed")
        self.assertNotIn(sentinel, runtime.load_error)

    def test_loads_valid_artifact_and_classifies_finite_score(self) -> None:
        session = _AdapterSession(output=np.asarray([0.76], dtype=np.float32))
        with self._artifact(match_threshold=0.75) as onnx_path:
            with patch("module.face.domain_adapter.ort.InferenceSession", return_value=session):
                runtime = DomainAdapterRuntime.load("active", onnx_path)

        decision = runtime.compare(self.ref.astype(np.float64), self.photo)

        self.assertTrue(runtime.ready)
        self.assertTrue(decision.usable)
        self.assertEqual(decision.status, "match")
        self.assertAlmostEqual(decision.cosine or 0.0, 0.76, places=6)
        self.assertEqual(session.last_inputs["ref_embedding"].dtype, np.float32)
        self.assertEqual(session.last_inputs["ref_embedding"].shape, (1, 128))

    def test_wrong_shape_and_nonfinite_input_fall_open(self) -> None:
        session = _AdapterSession()
        runtime = self._loaded_runtime(session)

        wrong_shape = runtime.compare(np.zeros(127, dtype=np.float32), self.photo)
        non_finite = runtime.compare(np.full(128, np.nan, dtype=np.float32), self.photo)

        self.assertFalse(wrong_shape.usable)
        self.assertEqual(wrong_shape.error, "adapter_inference_failed")
        self.assertFalse(non_finite.usable)
        self.assertEqual(non_finite.error, "adapter_inference_failed")

    def test_inputs_are_l2_normalized_before_inference(self) -> None:
        session = _AdapterSession()
        runtime = self._loaded_runtime(session)
        ref = np.zeros(128, dtype=np.float32)
        ref[:2] = [3.0, 4.0]
        photo = np.zeros(128, dtype=np.float32)
        photo[0] = 10.0

        decision = runtime.compare(ref, photo)

        self.assertTrue(decision.usable)
        self.assertAlmostEqual(float(np.linalg.norm(session.last_inputs["ref_embedding"])), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(session.last_inputs["photo_embedding"])), 1.0)

    def test_non_finite_score_returns_fallback(self) -> None:
        runtime = self._loaded_runtime(_AdapterSession(output=np.asarray([np.nan], dtype=np.float32)))

        decision = runtime.compare(self.ref, self.photo)

        self.assertFalse(decision.usable)
        self.assertEqual(decision.error, "adapter_inference_failed")

    def test_inference_exception_and_wrong_output_shape_return_fallback(self) -> None:
        sentinel = "SENTINEL_SECRET /models/private/provider"
        failing = self._loaded_runtime(_AdapterSession(error=RuntimeError(sentinel)))
        malformed = self._loaded_runtime(_AdapterSession(output=np.asarray([[0.7]], dtype=np.float32)))

        inference = failing.compare(self.ref, self.photo)
        output = malformed.compare(self.ref, self.photo)

        self.assertFalse(inference.usable)
        self.assertEqual(inference.error, "adapter_inference_failed")
        self.assertNotIn(sentinel, inference.error)
        self.assertFalse(output.usable)
        self.assertEqual(output.error, "adapter_inference_failed")

    def test_conversion_error_redacts_input_details(self) -> None:
        sentinel = "SENTINEL_SECRET embedding contents"
        runtime = self._loaded_runtime(_AdapterSession())

        decision = runtime.compare(_SecretEmbedding(sentinel), self.photo)

        self.assertFalse(decision.usable)
        self.assertEqual(decision.error, "adapter_inference_failed")
        self.assertNotIn(sentinel, decision.error)

    def test_environment_loader_defaults_to_off_and_preserves_invalid_mode_error(self) -> None:
        old_mode = os.environ.pop("FACE_DOMAIN_ADAPTER_MODE", None)
        old_path = os.environ.pop("FACE_DOMAIN_ADAPTER_PATH", None)
        try:
            self.assertEqual(load_domain_adapter_from_env().mode, "off")
            os.environ["FACE_DOMAIN_ADAPTER_MODE"] = "invalid"
            with self.assertRaises(ValueError):
                load_domain_adapter_from_env()
        finally:
            if old_mode is not None:
                os.environ["FACE_DOMAIN_ADAPTER_MODE"] = old_mode
            else:
                os.environ.pop("FACE_DOMAIN_ADAPTER_MODE", None)
            if old_path is not None:
                os.environ["FACE_DOMAIN_ADAPTER_PATH"] = old_path
            else:
                os.environ.pop("FACE_DOMAIN_ADAPTER_PATH", None)

    def _loaded_runtime(self, session: _AdapterSession) -> DomainAdapterRuntime:
        with self._artifact() as onnx_path:
            with patch("module.face.domain_adapter.ort.InferenceSession", return_value=session):
                runtime = DomainAdapterRuntime.load("active", onnx_path)
        self.assertTrue(runtime.ready)
        return runtime

    @staticmethod
    def _manifest(onnx_bytes: bytes, **overrides: object) -> dict[str, object]:
        manifest: dict[str, object] = {
            "schema_version": 1,
            "model_version": "identity-domain-adapter-v1",
            "embedding_dimension": 128,
            "rank": 16,
            "match_threshold": 0.75,
            "onnx_file": "identity_domain_adapter.onnx",
            "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
            "input_names": ["ref_embedding", "photo_embedding"],
            "output_name": "adapted_cosine",
        }
        manifest.update(overrides)
        return manifest

    @classmethod
    def _artifact(cls, **manifest_overrides: object):
        return _Artifact(cls._manifest, manifest_overrides)


class _Artifact:
    def __init__(self, manifest_factory, manifest_overrides: dict[str, object]) -> None:
        self._manifest_factory = manifest_factory
        self._overrides = manifest_overrides
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name)
        onnx_bytes = self._overrides.pop("onnx_bytes", b"valid-onnx")
        onnx_path = root / "identity_domain_adapter.onnx"
        onnx_path.write_bytes(onnx_bytes)
        manifest = self._manifest_factory(onnx_bytes, **self._overrides)
        onnx_path.with_name("identity_domain_adapter.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return onnx_path

    def __exit__(self, *unused: object) -> None:
        assert self._temporary_directory is not None
        self._temporary_directory.cleanup()


class _SecretEmbedding:
    def __init__(self, secret: str) -> None:
        self._secret = secret

    def __array__(self, dtype: object = None) -> np.ndarray:
        raise ValueError(self._secret)


if __name__ == "__main__":
    unittest.main()
