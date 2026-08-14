"""Regression tests for shared OpenCV object serialization.

The HTTP service is threaded while FaceDetectorYN and FaceRecognizerSF are native,
stateful objects. Concurrent calls used to corrupt their heap and restart production.
"""
from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from module.face.detector import DetectedFace, FaceAligner, FaceDetector


class _OverlapProbe:
    """Fake native object that records whether two calls overlap."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.active = 0
        self.overlapped = False

    def _enter(self) -> None:
        with self._guard:
            self.active += 1
            if self.active > 1:
                self.overlapped = True
        time.sleep(0.01)

    def _leave(self) -> None:
        with self._guard:
            self.active -= 1


class _FakeDetector(_OverlapProbe):
    def setInputSize(self, _size: tuple[int, int]) -> None:  # noqa: N802
        self._enter()

    def detect(self, _image: np.ndarray):
        try:
            rec = np.array([[0, 0, 10, 10, 1, 1, 2, 1, 1.5, 2,
                             1, 3, 2, 3, 0.9]], dtype=np.float32)
            return None, rec
        finally:
            self._leave()


class _FakeAligner(_OverlapProbe):
    def alignCrop(self, _image: np.ndarray, _record: np.ndarray):  # noqa: N802
        self._enter()
        try:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        finally:
            self._leave()


class OpenCvObjectSerializationTests(unittest.TestCase):
    def test_detector_calls_do_not_overlap(self) -> None:
        native = _FakeDetector()
        detector = FaceDetector.__new__(FaceDetector)
        detector._detector = native
        detector._lock = threading.Lock()
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: detector.detect(image), range(16)))

        self.assertTrue(all(len(result) == 1 for result in results))
        self.assertFalse(native.overlapped)

    def test_aligner_calls_do_not_overlap(self) -> None:
        native = _FakeAligner()
        aligner = FaceAligner.__new__(FaceAligner)
        aligner._sface = native
        aligner._lock = threading.Lock()
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        face = DetectedFace((0, 0, 10, 10), [], 0.9,
                            np.zeros(15, dtype=np.float32))

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: aligner.align(image, face), range(16)))

        self.assertTrue(all(result.shape == (112, 112, 3) for result in results))
        self.assertFalse(native.overlapped)


if __name__ == "__main__":
    unittest.main()
