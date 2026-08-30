"""Stage 1 outlier-detection unit tests with synthetic embeddings.

We construct embeddings as L2-normalized random vectors with controlled
"same-person" vs "imposter" cos relationships. SFace embeddings live in
roughly the same geometry, so synthetic vectors test the algorithm, not the
model.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from module.face.pipeline import (
    FacePipeline,
    RefFeatureCache,
    SessionPhotoResult,
)
from module.face.session_consistency import (
    check_internal_consistency, session_prototype,
)


def _seeded_unit(rng: np.random.Generator, dim: int = 128) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base: np.ndarray, rng: np.random.Generator, noise: float = 0.05) -> np.ndarray:
    """Returns a unit vector close to base (small perturbation).

    Note: ||rng.standard_normal(D)|| ≈ sqrt(D), so for D=128 noise needs to be
    well below 1/sqrt(128) ≈ 0.088 for the perturbed vector to remain close
    to base. Default 0.05 yields cos(base, near) ≈ 0.87, cos(near, near) ≈ 0.76.
    """
    perturb = rng.standard_normal(base.shape).astype(np.float32) * noise
    v = base + perturb
    return v / np.linalg.norm(v)


class _AdapterSpy:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.ready = True
        self.version = "test-v1"
        self.artifact_sha256 = "a" * 64
        self.calls = 0

    def compare(self, _ref: np.ndarray, _photo: np.ndarray) -> None:
        self.calls += 1
        raise AssertionError("Stage 1 must bypass the domain adapter")


class _SyntheticSessionPipeline(FacePipeline):
    def __init__(
        self,
        ref_path: str,
        ref_embedding: np.ndarray,
        photo_embeddings: list[np.ndarray],
        adapter: _AdapterSpy,
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
        result = SessionPhotoResult(
            sequence_no=int(photo["sequence_no"]),
            photo_type="synthetic",
            passes_gate=True,
        )
        result._embedding = self._photo_embeddings[result.sequence_no - 1]  # type: ignore[attr-defined]
        return result


class StageOneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_all_same_person_no_outlier(self) -> None:
        base = _seeded_unit(self.rng)
        embs = [_near(base, self.rng, noise=0.05) for _ in range(8)]
        res = check_internal_consistency(embs, outlier_thresh=0.20)
        self.assertEqual(res.outlier_indices, [])
        self.assertTrue(res.is_consistent)
        # mean_cos for "same person" cluster should be comfortably above thresh
        self.assertTrue(all(m > 0.4 for m in res.mean_cos_per_index))

    def test_one_imposter_detected(self) -> None:
        base = _seeded_unit(self.rng)
        imposter = _seeded_unit(self.rng)   # independent random direction
        embs = [_near(base, self.rng, noise=0.05) for _ in range(8)]
        embs.append(imposter)
        res = check_internal_consistency(embs, outlier_thresh=0.20)
        self.assertEqual(res.outlier_indices, [8])
        self.assertFalse(res.is_consistent)
        # imposter mean_cos should be near 0 (independent random); real cluster >> 0
        self.assertLess(res.mean_cos_per_index[8], 0.20)
        for i in range(8):
            self.assertGreater(res.mean_cos_per_index[i], 0.20)

    def test_two_clusters_minority_flagged(self) -> None:
        """Majority A (6) + minority B (2). Minority should be flagged."""
        a_base = _seeded_unit(self.rng)
        b_base = _seeded_unit(self.rng)
        embs = [_near(a_base, self.rng, 0.05) for _ in range(6)]
        embs += [_near(b_base, self.rng, 0.05) for _ in range(2)]
        res = check_internal_consistency(embs, outlier_thresh=0.20)
        # Minority members' mean_cos = mostly to A cluster (near 0) + 1 to other B
        # member (high). Average drags below threshold.
        self.assertEqual(set(res.outlier_indices), {6, 7})

    def test_single_photo_returns_empty(self) -> None:
        embs = [_seeded_unit(self.rng)]
        res = check_internal_consistency(embs)
        self.assertEqual(res.outlier_indices, [])
        self.assertTrue(res.is_consistent)

    def test_empty_returns_empty(self) -> None:
        res = check_internal_consistency([])
        self.assertEqual(res.outlier_indices, [])
        self.assertEqual(res.mean_cos_per_index, [])

    def test_threshold_overridable(self) -> None:
        base = _seeded_unit(self.rng)
        embs = [_near(base, self.rng, 0.3) for _ in range(5)]
        # Real cluster — at extreme thresh=0.99 even the real photos get flagged
        res_strict = check_internal_consistency(embs, outlier_thresh=0.99)
        self.assertEqual(len(res_strict.outlier_indices), 5)
        # At thresh=0.0 nothing is ever an outlier
        res_loose = check_internal_consistency(embs, outlier_thresh=0.0)
        self.assertEqual(res_loose.outlier_indices, [])


class SessionPrototypeTest(unittest.TestCase):
    def test_mean_is_l2_normalized(self) -> None:
        rng = np.random.default_rng(7)
        base = _seeded_unit(rng)
        embs = [_near(base, rng, 0.05) for _ in range(5)]
        proto = session_prototype(embs)
        self.assertAlmostEqual(float(np.linalg.norm(proto)), 1.0, places=5)

    def test_single_input(self) -> None:
        rng = np.random.default_rng(3)
        e = _seeded_unit(rng)
        proto = session_prototype([e])
        np.testing.assert_allclose(proto, e, atol=1e-6)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            session_prototype([])


class PipelineStageOneRegressionTest(unittest.TestCase):
    def test_known_outlier_is_unchanged_and_bypasses_every_adapter_mode(self) -> None:
        rng = np.random.default_rng(42)
        base = _seeded_unit(rng)
        embeddings = [_near(base, rng, noise=0.05) for _ in range(8)]
        embeddings.append(_seeded_unit(rng))

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "ref.jpg"
            ref_path.write_bytes(b"synthetic-ref")
            results = []
            for mode in ("off", "shadow", "active"):
                adapter = _AdapterSpy(mode)
                pipeline = _SyntheticSessionPipeline(
                    str(ref_path), base, embeddings, adapter
                )
                result = pipeline.session_check(
                    str(ref_path),
                    [{"sequence_no": number} for number in range(1, 10)],
                )
                self.assertEqual(result.session_status, "mismatch")
                self.assertEqual(result.internal_consistency, "inconsistent")
                self.assertEqual(result.outlier_sequence_nos, [9])
                self.assertEqual(adapter.calls, 0)
                results.append(result)

        self.assertEqual(
            [result.outlier_sequence_nos for result in results],
            [[9], [9], [9]],
        )


if __name__ == "__main__":
    unittest.main()
