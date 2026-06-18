from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np

from src.detection.gaze_estimator import GazeConfig, GazeEstimator


class GazeSmoothingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = GazeEstimator.__new__(GazeEstimator)
        self.estimator.config = GazeConfig()
        self.estimator.direction_history = deque(maxlen=self.estimator.config.smoothing_window)

    def test_left_activates_after_two_consistent_frames(self) -> None:
        self.assertEqual(self.estimator._smooth_direction("left"), "center")
        self.assertEqual(self.estimator._smooth_direction("left"), "left")

    def test_right_activates_after_two_consistent_frames(self) -> None:
        self.assertEqual(self.estimator._smooth_direction("right"), "center")
        self.assertEqual(self.estimator._smooth_direction("right"), "right")

    def test_down_requires_three_consistent_frames(self) -> None:
        self.assertEqual(self.estimator._smooth_direction("down"), "center")
        self.assertEqual(self.estimator._smooth_direction("down"), "center")
        self.assertEqual(self.estimator._smooth_direction("down"), "down")

    def test_dark_pupil_pixel_detection_reads_horizontal_position(self) -> None:
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        eye_points = {
            33: (0.20, 0.40),
            160: (0.23, 0.37),
            158: (0.29, 0.37),
            133: (0.32, 0.40),
            153: (0.29, 0.43),
            144: (0.23, 0.43),
            362: (0.68, 0.40),
            385: (0.71, 0.37),
            387: (0.77, 0.37),
            263: (0.80, 0.40),
            373: (0.77, 0.43),
            380: (0.71, 0.43),
        }
        for index, (x, y) in eye_points.items():
            landmarks[index] = SimpleNamespace(x=x, y=y)

        frame = np.full((200, 400, 3), 210, dtype=np.uint8)
        cv2.circle(frame, (90, 80), 5, (5, 5, 5), -1)
        cv2.circle(frame, (282, 80), 5, (5, 5, 5), -1)

        ratio, confidence = self.estimator._average_dark_pupil_ratio(frame, landmarks)

        self.assertIsNotNone(ratio)
        self.assertLess(ratio, 0.5)
        self.assertGreater(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
