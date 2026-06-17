from __future__ import annotations

import time

import cv2
import numpy as np

from src.detection.gaze_estimator import GazeEstimator, direction_label
from src.detection.object_detector import ObjectDetector


class WebcamCapture:
    def __init__(self, camera_index: int = 0, width: int = 960, height: int = 540) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture: cv2.VideoCapture | None = None

    def start(self) -> tuple[bool, str | None]:
        if self.capture is not None and self.capture.isOpened():
            return True, None

        capture = cv2.VideoCapture(self.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not capture.isOpened():
            capture.release()
            return False, "Unable to open webcam. Check camera permission or another app using the camera."

        self.capture = capture
        return True, None

    def read(self) -> tuple[bool, np.ndarray | None, str | None]:
        ok, error = self.start()
        if not ok:
            return False, None, error

        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok:
            return False, None, "Unable to read from webcam. Check camera permission or another app using the camera."
        return True, frame, None

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class FrameAnalyzer:
    """Combines object detection and gaze estimation for one webcam frame."""

    def __init__(self) -> None:
        self.object_detector = ObjectDetector()
        self.gaze_estimator = GazeEstimator()

    def analyze(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        display = cv2.flip(frame, 1)
        object_status = self.object_detector.detect(display)
        gaze_status = self.gaze_estimator.estimate(display)

        person_count = object_status["person_count"]
        if not self.object_detector.available and gaze_status["face_visible"]:
            person_count = max(int(gaze_status.get("face_count", 1)), 1)

        status = {
            "timestamp": time.time(),
            "person_count": person_count,
            "face_visible": gaze_status["face_visible"],
            "face_centered": gaze_status["face_centered"],
            "looking_at_camera": gaze_status["looking_at_camera"],
            "looking_left": gaze_status["looking_left"],
            "looking_right": gaze_status["looking_right"],
            "looking_down": gaze_status["looking_down"],
            "looking_away": gaze_status["looking_away"],
            "phone_detected": object_status["phone_detected"],
            "laptop_detected": object_status["laptop_detected"],
            "book_detected": object_status["book_detected"],
            "multiple_persons": person_count > 1,
            "detector_mode": "yolo" if self.object_detector.available else "opencv_fallback",
            "eyes_visible": gaze_status.get("eyes_visible", False),
        }
        _draw_status(display, status)
        return display, status


def _draw_status(frame: np.ndarray, status: dict) -> None:
    lines = [
        f"Persons: {status['person_count']}",
        f"Face centered: {'yes' if status['face_centered'] else 'no'}",
        f"Gaze approx: {direction_label(status['looking_at_camera'], status)}",
        f"Phone/Laptop/Book: {int(status['phone_detected'])}/{int(status['laptop_detected'])}/{int(status['book_detected'])}",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        y += 28
