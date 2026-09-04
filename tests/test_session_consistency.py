"""Stage 1 outlier-detection unit tests with synthetic embeddings.

We construct embeddings as L2-normalized random vectors with controlled
"same-person" vs "imposter" cos relationships. SFace embeddings live in
roughly the same geometry, so synthetic vectors test the algorithm, not the
model.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from module.face.errors import COS_INCONCLUSIVE_ZONE
from module.face.pipeline import (
    FacePipeline,
    RefFeatureCache,
    SessionPhotoResult,
)
from module.face.quality_gate import apply_session_match_consensus
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


def _photo(
        *,
        sequence_no: int,
        status: str = "inconclusive",
        cosine: float | None = 0.32,
        error_code: str = "",
        quality_flags: list[str] | None = None,
        is_outlier: bool = False,
        passes_gate: bool = True,
) -> SessionPhotoResult:
    return SessionPhotoResult(
        sequence_no=sequence_no,
        match_status=status,
        cosine_score=cosine,
        error_code=error_code,
        quality_flags=list(quality_flags or []),
        is_outlier=is_outlier,
        passes_gate=passes_gate,
    )


def _score_vec(cosine: float) -> np.ndarray:
    return np.array(
        [cosine, np.sqrt(max(0.0, 1.0 - cosine * cosine))],
        dtype=np.float32,
    )


class _CacheStub:
    def __init__(self, emb: np.ndarray) -> None:
        self._emb = emb

    def get(self, key: object) -> np.ndarray:
        return self._emb

    def put(self, key: object, emb: np.ndarray) -> None:
        self._emb = emb


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

class SessionMatchConsensusRuleTests(unittest.TestCase):
    def test_consensus_promotes_only_the_clean_midband_candidate(self) -> None:
        candidate = _photo(sequence_no=1, cosine=0.32)
        peer_a = _photo(sequence_no=2, status="match", cosine=0.40)
        peer_b = _photo(sequence_no=3, status="match", cosine=0.45)
        ineligible = _photo(sequence_no=4, cosine=0.29)

        changed = apply_session_match_consensus([candidate, peer_a, peer_b, ineligible])

        self.assertEqual(changed, 1)
        self.assertEqual(candidate.match_status, "match")
        self.assertEqual(peer_a.match_status, "match")
        self.assertEqual(peer_b.match_status, "match")
        self.assertEqual(ineligible.match_status, "inconclusive")

    def test_consensus_allows_cos_inconclusive_zone_error_code(self) -> None:
        candidate = _photo(
            sequence_no=1,
            cosine=0.32,
            error_code=COS_INCONCLUSIVE_ZONE,
        )
        peers = [
            _photo(sequence_no=2, status="match", cosine=0.40),
            _photo(sequence_no=3, status="match", cosine=0.45),
        ]

        changed = apply_session_match_consensus([candidate, *peers])

        self.assertEqual(changed, 1)
        self.assertEqual(candidate.match_status, "match")
        self.assertEqual(candidate.error_code, COS_INCONCLUSIVE_ZONE)

    def test_consensus_rejects_non_inconclusive_candidates(self) -> None:
        peers = [
            _photo(sequence_no=2, status="match", cosine=0.40),
            _photo(sequence_no=3, status="match", cosine=0.45),
        ]

        for status in ("match", "mismatch"):
            with self.subTest(status=status):
                candidate = _photo(sequence_no=1, status=status, cosine=0.32)
                changed = apply_session_match_consensus([candidate, *peers])
                self.assertEqual(changed, 0)
                self.assertEqual(candidate.match_status, status)

    def test_consensus_rejects_cosine_outside_midband(self) -> None:
        peers = [
            _photo(sequence_no=2, status="match", cosine=0.40),
            _photo(sequence_no=3, status="match", cosine=0.45),
        ]

        for cosine in (None, 0.29, 0.35):
            with self.subTest(cosine=cosine):
                candidate = _photo(sequence_no=1, cosine=cosine)
                changed = apply_session_match_consensus([candidate, *peers])
                self.assertEqual(changed, 0)
                self.assertEqual(candidate.match_status, "inconclusive")

    def test_consensus_rejects_nonempty_error_code_outside_allowed_zone(self) -> None:
        candidate = _photo(sequence_no=1, cosine=0.32, error_code="cos_borderline")
        peers = [
            _photo(sequence_no=2, status="match", cosine=0.40),
            _photo(sequence_no=3, status="match", cosine=0.45),
        ]

        changed = apply_session_match_consensus([candidate, *peers])

        self.assertEqual(changed, 0)
        self.assertEqual(candidate.match_status, "inconclusive")

    def test_consensus_rejects_quality_flags_even_without_error_code(self) -> None:
        candidate = _photo(
            sequence_no=1,
            cosine=0.32,
            quality_flags=["pose_excessive"],
        )
        peers = [
            _photo(sequence_no=2, status="match", cosine=0.40),
            _photo(sequence_no=3, status="match", cosine=0.45),
        ]

        changed = apply_session_match_consensus([candidate, *peers])

        self.assertEqual(changed, 0)
        self.assertEqual(candidate.match_status, "inconclusive")

    def test_consensus_ignores_non_clean_match_peers(self) -> None:
        candidate = _photo(sequence_no=1, cosine=0.32)
        clean_match = _photo(sequence_no=2, status="match", cosine=0.40)
        flagged_match = _photo(
            sequence_no=3,
            status="match",
            cosine=0.45,
            quality_flags=["pose_excessive"],
            passes_gate=False,
        )

        changed = apply_session_match_consensus([candidate, clean_match, flagged_match])

        self.assertEqual(changed, 0)
        self.assertEqual(candidate.match_status, "inconclusive")
        self.assertEqual(clean_match.match_status, "match")
        self.assertEqual(flagged_match.match_status, "match")

    def test_consensus_rejects_candidate_outlier(self) -> None:
        candidate = _photo(sequence_no=1, cosine=0.32, is_outlier=True)
        peers = [
            _photo(sequence_no=2, status="match", cosine=0.40),
            _photo(sequence_no=3, status="match", cosine=0.45),
        ]

        changed = apply_session_match_consensus([candidate, *peers])

        self.assertEqual(changed, 0)

    def test_consensus_rejects_fewer_than_two_match_peers(self) -> None:
        candidate = _photo(sequence_no=1, cosine=0.32)
        peers = [_photo(sequence_no=2, status="match", cosine=0.40)]

        changed = apply_session_match_consensus([candidate, *peers])

        self.assertEqual(changed, 0)

    def test_consensus_rejects_any_mismatch_peer(self) -> None:
        candidate = _photo(sequence_no=1, cosine=0.32)
        peers = [
            _photo(sequence_no=2, status="match", cosine=0.40),
            _photo(sequence_no=3, status="mismatch", cosine=0.10),
            _photo(sequence_no=4, status="match", cosine=0.45),
        ]

        changed = apply_session_match_consensus([candidate, *peers])

        self.assertEqual(changed, 0)

    def test_consensus_rejects_any_outlier_peer(self) -> None:
        candidate = _photo(sequence_no=1, cosine=0.32)
        peers = [
            _photo(sequence_no=2, status="match", cosine=0.40),
            _photo(sequence_no=3, status="match", cosine=0.45, is_outlier=True),
            _photo(sequence_no=4, status="match", cosine=0.41),
        ]

        changed = apply_session_match_consensus([candidate, *peers])

        self.assertEqual(changed, 0)


class SessionMatchConsensusIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = dict(os.environ)
        self.ref_emb = _score_vec(1.0)
        self.pipeline = FacePipeline(
            detector=object(),
            aligner=object(),
            recognizer=object(),
            ref_cache=_CacheStub(self.ref_emb),
        )

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def _passing_photo(
            self,
            *,
            sequence_no: int,
            embedding: np.ndarray,
            quality_flags: list[str] | None = None,
    ) -> SessionPhotoResult:
        photo = SessionPhotoResult(
            sequence_no=sequence_no,
            photo_type=f"p{sequence_no}",
            passes_gate=True,
            quality_flags=list(quality_flags or []),
        )
        photo._embedding = embedding  # type: ignore[attr-defined]
        return photo

    def test_consensus_is_default_off_for_consistent_multi_photo_sessions(self) -> None:
        photos = [
            self._passing_photo(sequence_no=1, embedding=_score_vec(0.32)),
            self._passing_photo(sequence_no=2, embedding=_score_vec(0.40)),
            self._passing_photo(sequence_no=3, embedding=_score_vec(0.45)),
        ]

        with patch.object(RefFeatureCache, "make_key", return_value="cached"), patch.object(
                self.pipeline, "_process_photo_for_session", side_effect=photos):
            result = self.pipeline.session_check("ref.jpg", [{}, {}, {}])

        self.assertEqual(result.internal_consistency, "consistent")
        self.assertEqual(result.photo_results[0].match_status, "inconclusive")
        self.assertFalse(any(p.is_outlier for p in result.photo_results))

    def test_consensus_runs_after_stage1_when_switch_enabled(self) -> None:
        os.environ["FACE_SESSION_MATCH_CONSENSUS"] = "true"
        photos = [
            self._passing_photo(sequence_no=1, embedding=_score_vec(0.32)),
            self._passing_photo(sequence_no=2, embedding=_score_vec(0.40)),
            self._passing_photo(sequence_no=3, embedding=_score_vec(0.45)),
        ]
        observed: dict[str, list[float | bool | None]] = {}

        def _capture_then_apply(photo_results: list[SessionPhotoResult]) -> int:
            observed["mean_cos"] = [p.mean_cos_to_peers for p in photo_results]
            observed["outliers"] = [p.is_outlier for p in photo_results]
            return apply_session_match_consensus(photo_results)

        with patch.object(RefFeatureCache, "make_key", return_value="cached"), patch.object(
                self.pipeline, "_process_photo_for_session", side_effect=photos), patch(
                "module.face.pipeline.apply_session_match_consensus",
                side_effect=_capture_then_apply) as apply_mock:
            result = self.pipeline.session_check("ref.jpg", [{}, {}, {}])

        self.assertEqual(result.internal_consistency, "consistent")
        self.assertEqual(result.photo_results[0].match_status, "match")
        self.assertEqual(observed["outliers"], [False, False, False])
        self.assertTrue(all(value is not None for value in observed["mean_cos"]))
        apply_mock.assert_called_once()

    def test_consensus_does_not_run_for_zero_post_gate_sessions(self) -> None:
        os.environ["FACE_SESSION_MATCH_CONSENSUS"] = "true"
        blocked = SessionPhotoResult(sequence_no=1, photo_type="p1", passes_gate=False)

        with patch.object(RefFeatureCache, "make_key", return_value="cached"), patch.object(
                self.pipeline, "_process_photo_for_session", return_value=blocked), patch(
                "module.face.pipeline.apply_session_match_consensus") as apply_mock:
            result = self.pipeline.session_check("ref.jpg", [{}])

        self.assertEqual(result.internal_consistency, "unknown")
        self.assertEqual(result.session_status, "inconclusive")
        apply_mock.assert_not_called()

    def test_consensus_does_not_run_for_single_post_gate_sessions(self) -> None:
        os.environ["FACE_SESSION_MATCH_CONSENSUS"] = "true"
        single = self._passing_photo(sequence_no=1, embedding=_score_vec(0.32))

        with patch.object(RefFeatureCache, "make_key", return_value="cached"), patch.object(
                self.pipeline, "_process_photo_for_session", return_value=single), patch(
                "module.face.pipeline.apply_session_match_consensus") as apply_mock:
            result = self.pipeline.session_check("ref.jpg", [{}])

        self.assertEqual(result.internal_consistency, "single")
        apply_mock.assert_not_called()

    def test_consensus_does_not_run_for_inconsistent_sessions(self) -> None:
        os.environ["FACE_SESSION_MATCH_CONSENSUS"] = "true"
        photos = [
            self._passing_photo(sequence_no=1, embedding=_score_vec(0.32)),
            self._passing_photo(sequence_no=2, embedding=_score_vec(0.40)),
            self._passing_photo(sequence_no=3, embedding=np.array([0.0, -1.0], dtype=np.float32)),
        ]

        with patch.object(RefFeatureCache, "make_key", return_value="cached"), patch.object(
                self.pipeline, "_process_photo_for_session", side_effect=photos), patch(
                "module.face.pipeline.apply_session_match_consensus") as apply_mock:
            result = self.pipeline.session_check("ref.jpg", [{}, {}, {}])

        self.assertEqual(result.internal_consistency, "inconsistent")
        apply_mock.assert_not_called()

if __name__ == "__main__":
    unittest.main()
