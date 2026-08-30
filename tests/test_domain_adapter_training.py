from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
import torch

from tools.domain_adapter_training import (
    LowRankDomainAdapter,
    PairSet,
    SessionEmbedding,
    build_pair_sets,
    contrastive_margin_loss,
    export_onnx,
    extract_session_embeddings,
    manifest_training_rows,
    select_threshold,
    train_adapter,
    write_candidate_artifacts,
)
from tools.train_domain_adapter import main as training_main


class LowRankDomainAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.refs = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.photos = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 0.6, 0.0],
                [0.0, 0.8, 0.0, 0.6],
            ]
        )
        self.labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    def test_zero_initialization_is_identity(self) -> None:
        model = LowRankDomainAdapter(dimension=128, rank=16)
        ref = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)
        photo = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)

        adapted = model(ref, photo)
        expected = (ref * photo).sum(dim=1)

        torch.testing.assert_close(adapted, expected)

    def test_one_training_step_pulls_positive_and_pushes_negative(self) -> None:
        torch.manual_seed(7)
        model = LowRankDomainAdapter(dimension=4, rank=2)
        before = model(self.refs, self.photos).detach()
        training_scores = model(self.refs, self.photos)
        loss = contrastive_margin_loss(
            training_scores,
            self.labels,
            positive_margin=0.35,
            negative_margin=0.15,
        )
        loss.backward()
        torch.optim.SGD(model.parameters(), lr=0.1).step()

        after = model(self.refs, self.photos).detach()

        self.assertGreater(
            after[self.labels == 1].mean(), before[self.labels == 1].mean()
        )
        self.assertLess(after[self.labels == 0].mean(), before[self.labels == 0].mean())


class PairConstructionTests(unittest.TestCase):
    @staticmethod
    def _session(student: str, session: str, split: str, vector: list[float]):
        embedding = np.asarray(vector, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        return SessionEmbedding(
            student_id=student,
            session_id=session,
            split=split,
            label="match",
            ref_embedding=embedding,
            session_prototype=embedding,
        )

    def test_training_mines_at_most_twenty_hardest_negatives_per_positive(self) -> None:
        anchor = self._session("anchor", "s0", "train", [1.0, 0.0])
        others = [
            self._session(
                f"student-{index:02d}",
                f"s{index:02d}",
                "train",
                [1.0, (index + 1) / 100.0],
            )
            for index in range(25)
        ]

        pairs = build_pair_sets([anchor, *others])["train"]
        anchor_negatives = [
            item
            for item in pairs.metadata
            if item.ref_session_id == "s0" and item.label == 0
        ]

        self.assertEqual(len(anchor_negatives), 20)
        self.assertEqual(
            [item.photo_session_id for item in anchor_negatives[:3]],
            ["s00", "s01", "s02"],
        )

    def test_validation_keeps_every_cross_student_pair(self) -> None:
        sessions = [
            self._session("a", "one", "validation", [1.0, 0.0]),
            self._session("b", "two", "validation", [0.0, 1.0]),
            self._session("c", "three", "validation", [-1.0, 0.0]),
        ]

        pairs = build_pair_sets(sessions)["validation"]

        self.assertEqual(int((pairs.labels == 1).sum()), 3)
        self.assertEqual(int((pairs.labels == 0).sum()), 6)

    def test_manifest_consumption_is_sorted_and_keeps_only_clean_test_rows(
        self,
    ) -> None:
        manifest = {
            "schema_version": 1,
            "sessions": [
                self._row("b", "two", "validation"),
                self._row("a", "one", "train"),
            ],
            "evaluation_sessions": [
                self._row("d", "four", "test", ["test_split", "photo_error"]),
                self._row("c", "three", "test", ["test_split"]),
            ],
        }

        rows = manifest_training_rows(manifest)

        self.assertEqual(
            [(row["split"], row["student_id"]) for row in rows],
            [("train", "a"), ("validation", "b"), ("test", "c")],
        )

    @staticmethod
    def _row(
        student: str,
        session: str,
        split: str,
        reasons: list[str] | None = None,
    ) -> dict:
        row = {
            "student_id": student,
            "session_id": session,
            "split": split,
            "label": "match",
            "ref_image_path": f"/{student}/ref.jpg",
            "photos": [{"sequence_no": 1, "image_path": f"/{student}/photo.jpg"}],
        }
        if reasons is not None:
            row["training_exclusion_reasons"] = reasons
        return row


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_lowest_threshold_with_best_recall_under_far_ceiling(self) -> None:
        scores = np.asarray([0.40, 0.30, 0.25, 0.10], dtype=np.float32)
        labels = np.asarray([1, 1, 0, 0], dtype=np.int64)

        metrics = select_threshold(scores, labels, max_false_accept_rate=0.0)

        self.assertAlmostEqual(metrics.threshold, 0.250, places=6)
        self.assertEqual(metrics.true_accept_rate, 1.0)
        self.assertEqual(metrics.false_accept_rate, 0.0)


class EmbeddingExtractionTests(unittest.TestCase):
    def test_rerun_uses_private_stat_keyed_cache_and_excludes_raw_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / name
                for name in ("ref.jpg", "one.jpg", "two.jpg", "three.jpg", "bad.jpg")
            ]
            for index, path in enumerate(paths):
                image = np.full((8, 8, 3), index + 1, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(path), image))
            manifest = {
                "schema_version": 1,
                "sessions": [
                    {
                        "student_id": "student",
                        "session_id": "session",
                        "split": "train",
                        "label": "match",
                        "ref_image_path": str(paths[0]),
                        "photos": [
                            {"sequence_no": 1, "image_path": str(paths[1])},
                            {"sequence_no": 2, "image_path": str(paths[2])},
                            {"sequence_no": 3, "image_path": str(paths[3])},
                            {"sequence_no": 4, "image_path": str(paths[4])},
                        ],
                    }
                ],
                "evaluation_sessions": [],
            }
            vectors = {
                str(paths[0]): np.asarray([1.0, 0.0], dtype=np.float32),
                str(paths[1]): np.asarray([1.0, 0.0], dtype=np.float32),
                str(paths[2]): np.asarray([1.0, 0.0], dtype=np.float32),
                str(paths[3]): np.asarray([1.0, 0.0], dtype=np.float32),
                str(paths[4]): np.asarray([-1.0, 0.0], dtype=np.float32),
            }
            first_pipeline = _FakePipeline(vectors)
            cache = root / "embeddings.npz"

            first = extract_session_embeddings(
                manifest,
                cache_path=cache,
                pipeline_factory=lambda: first_pipeline,
            )

            self.assertEqual(first_pipeline.extractions, 5)
            self.assertEqual(len(first), 1)
            np.testing.assert_allclose(first[0].session_prototype, [1.0, 0.0])
            self.assertEqual(cache.stat().st_mode & 0o777, 0o600)

            second_pipeline = _FakePipeline(vectors)
            second = extract_session_embeddings(
                manifest,
                cache_path=cache,
                pipeline_factory=lambda: second_pipeline,
            )

            self.assertEqual(second_pipeline.extractions, 0)
            np.testing.assert_array_equal(
                second[0].session_prototype, first[0].session_prototype
            )


class DeterministicTrainingTests(unittest.TestCase):
    def test_identity_regularization_penalizes_residual_weights(self) -> None:
        model = LowRankDomainAdapter(dimension=4, rank=2)
        scores = torch.tensor([0.9, 0.0], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])

        loss = contrastive_margin_loss(
            scores,
            labels,
            positive_margin=0.35,
            negative_margin=0.15,
            residual_parameters=model.parameters(),
            identity_regularization_weight=1.0,
        )

        self.assertGreater(float(loss.detach()), 0.0)

    def test_same_seed_produces_identical_model_and_metrics(self) -> None:
        pairs = self._pairs()

        first = train_adapter(
            pairs,
            pairs,
            dimension=4,
            rank=2,
            epochs=4,
            seed=20260830,
            learning_rate=0.05,
            positive_margin=0.35,
            negative_margin=0.15,
            max_false_accept_rate=0.5,
        )
        second = train_adapter(
            pairs,
            pairs,
            dimension=4,
            rank=2,
            epochs=4,
            seed=20260830,
            learning_rate=0.05,
            positive_margin=0.35,
            negative_margin=0.15,
            max_false_accept_rate=0.5,
        )

        for name, value in first.model.state_dict().items():
            torch.testing.assert_close(
                value, second.model.state_dict()[name], rtol=0.0, atol=0.0
            )
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.validation_metrics, second.validation_metrics)

    def test_training_continues_until_validation_far_becomes_feasible(self) -> None:
        pairs = self._initially_infeasible_pairs()

        result = train_adapter(
            pairs,
            pairs,
            dimension=4,
            rank=2,
            epochs=15,
            seed=20260830,
            learning_rate=0.05,
            positive_margin=0.35,
            negative_margin=0.15,
            max_false_accept_rate=0.0,
        )

        self.assertGreater(result.best_epoch, 1)
        self.assertEqual(result.validation_metrics.false_accept_rate, 0.0)

    @staticmethod
    def _pairs() -> PairSet:
        refs = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        photos = np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.0, 0.43589, 0.0],
                [0.0, 0.9, 0.0, 0.43589],
            ],
            dtype=np.float32,
        )
        return PairSet(
            ref_embeddings=refs,
            photo_embeddings=photos,
            labels=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            metadata=(),
        )

    @staticmethod
    def _initially_infeasible_pairs() -> PairSet:
        refs = np.eye(4, dtype=np.float32)
        photos = np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 0.6, 0.0],
                [0.0, 0.8, 0.0, 0.6],
            ],
            dtype=np.float32,
        )
        return PairSet(
            ref_embeddings=refs,
            photo_embeddings=photos,
            labels=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            metadata=(),
        )


class OnnxExportTests(unittest.TestCase):
    def test_export_round_trip_matches_pytorch_with_dynamic_batch(self) -> None:
        torch.manual_seed(11)
        model = LowRankDomainAdapter(dimension=128, rank=4)
        with torch.no_grad():
            model.ref_tower.up.weight.normal_(std=0.01)
            model.photo_tower.up.weight.normal_(std=0.01)
        refs = torch.nn.functional.normalize(torch.randn(3, 128), dim=1)
        photos = torch.nn.functional.normalize(torch.randn(3, 128), dim=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "adapter.onnx"

            max_error = export_onnx(
                model,
                output,
                parity_inputs=(refs.numpy(), photos.numpy()),
            )

            session = ort.InferenceSession(
                str(output), providers=["CPUExecutionProvider"]
            )
            actual = session.run(
                ["adapted_cosine"],
                {"ref_embedding": refs.numpy(), "photo_embedding": photos.numpy()},
            )[0]
            expected = model(refs, photos).detach().numpy()
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
            self.assertLessEqual(max_error, 1e-5)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(session.get_inputs()[0].shape, ["N", 128])
            self.assertEqual(session.get_outputs()[0].shape, ["N"])


class ArtifactAndCliTests(unittest.TestCase):
    def test_candidate_manifest_contains_checksum_parity_and_private_report(
        self,
    ) -> None:
        pairs = self._pairs_128()
        result = train_adapter(
            pairs,
            pairs,
            pairs,
            dimension=128,
            rank=2,
            epochs=1,
            seed=17,
            max_false_accept_rate=0.5,
        )
        dataset_manifest = {
            "schema_version": 1,
            "snapshot": "2026-08-30T12:00:00+00:00",
            "split_seed": "seed-v1",
            "sessions": [
                PairConstructionTests._row("a", "train", "train"),
                PairConstructionTests._row("b", "validation", "validation"),
            ],
            "evaluation_sessions": [
                PairConstructionTests._row("c", "test", "test", ["test_split"])
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            manifest = write_candidate_artifacts(
                result,
                {"train": pairs, "validation": pairs, "test": pairs},
                dataset_manifest,
                output,
                hyperparameters={"epochs": 1, "seed": 17},
                source_revision="abc123",
            )

            onnx_path = output / "identity_domain_adapter.onnx"
            manifest_path = output / "identity_domain_adapter.manifest.json"
            report_path = output / "training-evaluation.json"
            expected_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
            self.assertEqual(manifest["onnx_sha256"], expected_sha)
            self.assertEqual(manifest["embedding_dimension"], 128)
            self.assertLessEqual(manifest["onnx_parity_max_abs_error"], 1e-5)
            self.assertEqual(manifest["split_counts"]["train"]["sessions"], 1)
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["best_epoch"], 1)
            self.assertEqual(len(report["threshold_sweeps"]["validation"]), 351)
            for path in (onnx_path, manifest_path, report_path):
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_cli_refuses_non_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            dataset.write_text("{}", encoding="utf-8")
            output = root / "candidate"
            output.mkdir()
            marker = output / "keep"
            marker.write_text("user data", encoding="utf-8")

            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    training_main(
                        [
                            "--dataset-manifest",
                            str(dataset),
                            "--output-dir",
                            str(output),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user data")

    @staticmethod
    def _pairs_128() -> PairSet:
        refs = np.zeros((4, 128), dtype=np.float32)
        photos = np.zeros((4, 128), dtype=np.float32)
        refs[np.arange(4), np.arange(4)] = 1.0
        photos[0, 1] = 1.0
        photos[1, 0] = 1.0
        photos[2, 0] = 0.9
        photos[2, 2] = 0.43589
        photos[3, 1] = 0.9
        photos[3, 3] = 0.43589
        return PairSet(
            ref_embeddings=refs,
            photo_embeddings=photos,
            labels=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            metadata=(),
        )


class _FakePipeline:
    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors
        self.extractions = 0

    def _extract_or_raise(self, image: np.ndarray) -> np.ndarray:
        self.extractions += 1
        return next(iter(self.vectors.values()))

    def _process_photo_for_session(self, photo: dict) -> SimpleNamespace:
        self.extractions += 1
        return SimpleNamespace(
            passes_gate=True,
            _embedding=self.vectors[photo["image_path"]],
        )


if __name__ == "__main__":
    unittest.main()
