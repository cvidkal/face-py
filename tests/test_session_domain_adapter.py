"""Domain-adapter decision-table tests for session-level Stage 2."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from module.face.domain_adapter import AdapterDecision
from module.face.pipeline import (
    FacePipeline,
    RefFeatureCache,
    SessionPhotoResult,
    build_pipeline_from_env,
)


def _unit_with_cosine(cosine: float) -> np.ndarray:
    embedding = np.zeros(128, dtype=np.float32)
    embedding[0] = cosine
    embedding[1] = math.sqrt(1.0 - cosine * cosine)
    return embedding


class _AdapterFake:
    def __init__(
        self,
        mode: str,
        *,
        ready: bool = True,
        decision: AdapterDecision | None = None,
    ) -> None:
        self.mode = mode
        self.ready = ready
        self.version = "adapter-v1" if ready else ""
        self.artifact_sha256 = "b" * 64 if ready else ""
        self.load_error = "adapter_artifact_invalid" if not ready else ""
        self.decision = decision or AdapterDecision(
            usable=True,
            cosine=0.61,
            status="match",
        )
        self.calls = 0
        self.last_ref: np.ndarray | None = None
        self.last_photo: np.ndarray | None = None

    def compare(self, ref: np.ndarray, photo: np.ndarray) -> AdapterDecision:
        self.calls += 1
        self.last_ref = ref.copy()
        self.last_photo = photo.copy()
        return self.decision


class _SyntheticSessionPipeline(FacePipeline):
    def __init__(
        self,
        ref_path: str,
        ref_embedding: np.ndarray,
        photo_embeddings: list[np.ndarray | None],
        adapter: _AdapterFake,
    ) -> None:
        ref_cache = RefFeatureCache()
        cache_key = RefFeatureCache.make_key(ref_path)
        assert cache_key is not None
        ref_cache.put(cache_key, ref_embedding)
        super().__init__(
            detector=None,  # type: ignore[arg-type]
            aligner=None,  # type: ignore[arg-type]
            recognizer=None,  # type: ignore[arg-type]
            ref_cache=ref_cache,
        )
        self.domain_adapter = adapter
        self._photo_embeddings = photo_embeddings

    def _process_photo_for_session(self, photo: dict) -> SessionPhotoResult:
        sequence_no = int(photo["sequence_no"])
        embedding = self._photo_embeddings[sequence_no - 1]
        if embedding is None:
            return SessionPhotoResult(
                sequence_no=sequence_no,
                photo_type="synthetic",
                passes_gate=False,
                error_code="synthetic_error",
                error="synthetic photo failure",
            )
        result = SessionPhotoResult(
            sequence_no=sequence_no,
            photo_type="synthetic",
            passes_gate=True,
        )
        result._embedding = embedding  # type: ignore[attr-defined]
        return result


class SessionDomainAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ref_path = Path(self._tmp.name) / "ref.jpg"
        self.ref_path.write_bytes(b"synthetic-ref")
        self.ref = _unit_with_cosine(1.0)
        self.raw_inconclusive = _unit_with_cosine(0.25)

    def pipeline(
        self,
        adapter: _AdapterFake,
        embeddings: list[np.ndarray | None] | None = None,
    ) -> _SyntheticSessionPipeline:
        return _SyntheticSessionPipeline(
            str(self.ref_path),
            self.ref,
            embeddings or [self.raw_inconclusive, self.raw_inconclusive],
            adapter,
        )

    @staticmethod
    def photos(count: int = 2) -> list[dict]:
        return [{"sequence_no": number} for number in range(1, count + 1)]

    def test_inconsistent_stage_one_never_calls_adapter_and_reports_raw_stage_one(self) -> None:
        rng = np.random.default_rng(7)
        base = _unit_with_cosine(1.0)
        embeddings = []
        for _ in range(8):
            value = base + rng.standard_normal(128).astype(np.float32) * 0.05
            embeddings.append(value / np.linalg.norm(value))
        imposter = rng.standard_normal(128).astype(np.float32)
        embeddings.append(imposter / np.linalg.norm(imposter))
        adapter = _AdapterFake("active")

        result = self.pipeline(adapter, embeddings).session_check(
            str(self.ref_path), self.photos(9)
        )

        self.assertEqual(result.session_status, "mismatch")
        self.assertEqual(result.raw_session_status, "")
        self.assertEqual(result.decision_source, "raw_stage1")
        self.assertEqual(adapter.calls, 0)

    def test_unknown_stage_one_preserves_photo_error_and_never_calls_adapter(self) -> None:
        adapter = _AdapterFake("active")

        result = self.pipeline(adapter, [None]).session_check(
            str(self.ref_path), self.photos(1)
        )

        self.assertEqual(result.session_status, "inconclusive")
        self.assertEqual(result.internal_consistency, "unknown")
        self.assertEqual(result.raw_session_status, "")
        self.assertEqual(result.decision_source, "raw_stage1")
        self.assertEqual(result.photo_results[0].error_code, "synthetic_error")
        self.assertEqual(result.photo_results[0].error, "synthetic photo failure")
        self.assertEqual(adapter.calls, 0)

    def test_off_uses_raw_stage_two_without_calling_adapter(self) -> None:
        adapter = _AdapterFake("off", ready=False)

        result = self.pipeline(adapter).session_check(
            str(self.ref_path), self.photos()
        )

        self.assertEqual(result.session_status, "inconclusive")
        self.assertEqual(result.raw_session_status, "inconclusive")
        self.assertEqual(result.session_cos_to_ref, 0.25)
        self.assertIsNotNone(result.session_l2_to_ref)
        self.assertEqual(result.adapter_mode, "off")
        self.assertEqual(result.adapted_session_status, "")
        self.assertIsNone(result.adapted_session_cos_to_ref)
        self.assertEqual(result.decision_source, "raw_adapter_off")
        self.assertEqual(adapter.calls, 0)

    def test_unready_shadow_and_active_fall_back_without_calling_adapter(self) -> None:
        for mode in ("shadow", "active"):
            with self.subTest(mode=mode):
                adapter = _AdapterFake(mode, ready=False)

                result = self.pipeline(adapter).session_check(
                    str(self.ref_path), self.photos()
                )

                self.assertEqual(result.session_status, "inconclusive")
                self.assertEqual(result.raw_session_status, "inconclusive")
                self.assertEqual(result.adapter_mode, mode)
                self.assertEqual(result.adapter_version, "")
                self.assertEqual(result.adapter_sha256, "")
                self.assertEqual(result.decision_source, "raw_adapter_unready")
                self.assertEqual(adapter.calls, 0)

    def test_shadow_records_candidate_but_keeps_raw_status(self) -> None:
        adapter = _AdapterFake("shadow")

        result = self.pipeline(adapter).session_check(
            str(self.ref_path), self.photos()
        )

        self.assertEqual(result.session_status, "inconclusive")
        self.assertEqual(result.raw_session_status, "inconclusive")
        self.assertEqual(result.adapted_session_status, "match")
        self.assertEqual(result.adapted_session_cos_to_ref, 0.61)
        self.assertEqual(result.adapter_mode, "shadow")
        self.assertEqual(result.adapter_version, "adapter-v1")
        self.assertEqual(result.adapter_sha256, "b" * 64)
        self.assertEqual(result.decision_source, "raw_shadow")
        self.assertEqual(adapter.calls, 1)
        self.assertTrue(all(
            photo.match_status == "inconclusive" for photo in result.photo_results
        ))

    def test_active_uses_adapter_only_after_consistency_passes(self) -> None:
        adapter = _AdapterFake("active")

        result = self.pipeline(adapter).session_check(
            str(self.ref_path), self.photos()
        )

        self.assertEqual(result.internal_consistency, "consistent")
        self.assertEqual(result.session_status, "match")
        self.assertEqual(result.raw_session_status, "inconclusive")
        self.assertEqual(result.adapted_session_status, "match")
        self.assertEqual(result.decision_source, "domain_adapter")
        self.assertEqual(adapter.calls, 1)
        self.assertTrue(all(
            photo.match_status == "inconclusive" for photo in result.photo_results
        ))

    def test_active_single_photo_uses_adapted_stage_two(self) -> None:
        adapter = _AdapterFake("active")

        result = self.pipeline(adapter, [self.raw_inconclusive]).session_check(
            str(self.ref_path), self.photos(1)
        )

        self.assertEqual(result.internal_consistency, "single")
        self.assertEqual(result.session_status, "match")
        self.assertEqual(result.raw_session_status, "inconclusive")
        self.assertEqual(result.decision_source, "domain_adapter")
        self.assertEqual(adapter.calls, 1)

    def test_unusable_inference_falls_back_to_raw_stage_two(self) -> None:
        adapter = _AdapterFake(
            "active",
            decision=AdapterDecision(
                usable=False,
                error="adapter_inference_failed",
            ),
        )

        result = self.pipeline(adapter).session_check(
            str(self.ref_path), self.photos()
        )

        self.assertEqual(result.session_status, "inconclusive")
        self.assertEqual(result.raw_session_status, "inconclusive")
        self.assertEqual(result.adapted_session_status, "")
        self.assertIsNone(result.adapted_session_cos_to_ref)
        self.assertEqual(result.adapter_version, "adapter-v1")
        self.assertEqual(result.adapter_sha256, "b" * 64)
        self.assertEqual(result.decision_source, "raw_adapter_fallback")
        self.assertEqual(adapter.calls, 1)

    def test_response_serializes_only_the_exact_adapter_fields(self) -> None:
        adapter = _AdapterFake("shadow")
        result = self.pipeline(adapter).session_check(
            str(self.ref_path), self.photos()
        )

        payload = result.to_json()

        expected = {
            "raw_session_status": "inconclusive",
            "adapted_session_status": "match",
            "adapted_session_cos_to_ref": 0.61,
            "adapter_mode": "shadow",
            "adapter_version": "adapter-v1",
            "adapter_sha256": "b" * 64,
            "decision_source": "raw_shadow",
        }
        self.assertEqual(
            {name: payload[name] for name in expected},
            expected,
        )

    def test_environment_factory_attaches_the_loaded_runtime(self) -> None:
        adapter = _AdapterFake("shadow")
        with (
            patch("module.face.pipeline.FaceDetector"),
            patch("module.face.pipeline.FaceAligner"),
            patch("module.face.pipeline.FaceRecognizer"),
            patch(
                "module.face.pipeline.load_domain_adapter_from_env",
                return_value=adapter,
            ),
        ):
            pipeline = build_pipeline_from_env()

        self.assertIs(pipeline.domain_adapter, adapter)


if __name__ == "__main__":
    unittest.main()
