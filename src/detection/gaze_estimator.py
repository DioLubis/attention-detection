from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GazeConfig:
    face_center_tolerance: float = 0.18
    head_left_threshold: float = -0.045
    head_right_threshold: float = 0.045
    iris_left_threshold: float = 0.42
    iris_right_threshold: float = 0.58
    iris_down_threshold: float = 0.64
    head_down_threshold: float = 0.70


class GazeEstimator:
    """Approximate face/gaze estimator using MediaPipe Face Mesh or OpenCV fallback."""

    def __init__(self, config: GazeConfig | None = None) -> None:
        self.config = config or GazeConfig()
        self.face_mesh = self._load_face_mesh()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    def estimate(self, frame: np.ndarray) -> dict:
        if self.face_mesh is not None:
            return self._estimate_with_facemesh(frame)
        return self._estimate_with_haar(frame)

    def _load_face_mesh(self):
        try:
            import mediapipe as mp

            return mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            return None

    def _empty_status(self) -> dict:
        return {
            "face_visible": False,
            "face_centered": False,
            "looking_at_camera": False,
            "looking_left": False,
            "looking_right": False,
            "looking_down": False,
            "looking_away": True,
            "face_count": 0,
            "eyes_visible": False,
        }

    def _estimate_with_facemesh(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return self._empty_status()

        height, width = frame.shape[:2]
        all_faces = results.multi_face_landmarks
        landmarks = all_faces[0].landmark
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        min_x, max_x = int(min(xs) * width), int(max(xs) * width)
        min_y, max_y = int(min(ys) * height), int(max(ys) * height)

        face_centered = self._is_face_centered(min_x, max_x, width)
        gaze = self._estimate_gaze_from_landmarks(landmarks)
        # Horizontal gaze should win over down-gaze. Head turns often move the
        # nose vertically enough to look like "down" with a simple threshold.
        if gaze["looking_left"] or gaze["looking_right"]:
            gaze["looking_down"] = False
        looking_at_camera = face_centered and not (
            gaze["looking_left"] or gaze["looking_right"] or gaze["looking_down"]
        )

        _draw_box(frame, min_x, min_y, max_x, max_y, "face", (39, 174, 96))
        cv2.putText(
            frame,
            direction_label(looking_at_camera, gaze),
            (min_x, max(min_y - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (39, 174, 96),
            2,
        )

        return {
            "face_visible": True,
            "face_centered": face_centered,
            "looking_at_camera": looking_at_camera,
            "looking_left": gaze["looking_left"],
            "looking_right": gaze["looking_right"],
            "looking_down": gaze["looking_down"],
            "looking_away": not looking_at_camera,
            "face_count": len(all_faces),
            "eyes_visible": True,
        }

    def _estimate_gaze_from_landmarks(self, landmarks) -> dict:
        nose = landmarks[1]
        left_face = landmarks[234]
        right_face = landmarks[454]
        forehead = landmarks[10]
        chin = landmarks[152]

        face_center_x = (left_face.x + right_face.x) / 2
        horizontal_offset = nose.x - face_center_x
        vertical_span = max(chin.y - forehead.y, 0.001)
        vertical_offset = (nose.y - forehead.y) / vertical_span

        iris_ratio = self._average_iris_ratio(landmarks)
        iris_vertical_ratio = self._average_iris_vertical_ratio(landmarks)
        looking_left = horizontal_offset < self.config.head_left_threshold
        looking_right = horizontal_offset > self.config.head_right_threshold

        if iris_ratio is not None:
            looking_left = looking_left or iris_ratio < self.config.iris_left_threshold
            looking_right = looking_right or iris_ratio > self.config.iris_right_threshold

        looking_down = vertical_offset >= self.config.head_down_threshold
        if iris_vertical_ratio is not None:
            looking_down = looking_down or iris_vertical_ratio >= self.config.iris_down_threshold

        return {
            "looking_left": looking_left,
            "looking_right": looking_right,
            "looking_down": looking_down,
        }

    def _average_iris_ratio(self, landmarks) -> float | None:
        try:
            left_ratio = self._iris_ratio(landmarks, 33, 133, (468, 469, 470, 471))
            right_ratio = self._iris_ratio(landmarks, 362, 263, (473, 474, 475, 476))
            return (left_ratio + right_ratio) / 2
        except (IndexError, ZeroDivisionError):
            return None

    def _average_iris_vertical_ratio(self, landmarks) -> float | None:
        try:
            left_ratio = self._iris_vertical_ratio(landmarks, eye_top=159, eye_bottom=145, iris_points=(468, 469, 470, 471))
            right_ratio = self._iris_vertical_ratio(landmarks, eye_top=386, eye_bottom=374, iris_points=(473, 474, 475, 476))
            return (left_ratio + right_ratio) / 2
        except (IndexError, ZeroDivisionError):
            return None

    def _iris_ratio(self, landmarks, eye_left: int, eye_right: int, iris_points: tuple[int, ...]) -> float:
        left_x = landmarks[eye_left].x
        right_x = landmarks[eye_right].x
        iris_x = sum(landmarks[index].x for index in iris_points) / len(iris_points)
        eye_min = min(left_x, right_x)
        eye_max = max(left_x, right_x)
        return (iris_x - eye_min) / max(eye_max - eye_min, 0.001)

    def _iris_vertical_ratio(self, landmarks, eye_top: int, eye_bottom: int, iris_points: tuple[int, ...]) -> float:
        top_y = landmarks[eye_top].y
        bottom_y = landmarks[eye_bottom].y
        iris_y = sum(landmarks[index].y for index in iris_points) / len(iris_points)
        eye_min = min(top_y, bottom_y)
        eye_max = max(top_y, bottom_y)
        return (iris_y - eye_min) / max(eye_max - eye_min, 0.001)

    def _estimate_with_haar(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return self._empty_status()

        height, width = frame.shape[:2]
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        face_centered = self._is_face_centered(x, x + w, width)
        eye_status = self._estimate_eyes_with_haar(frame, x, y, w, h)
        looking_left = not face_centered and (x + w / 2) < width / 2
        looking_right = not face_centered and (x + w / 2) >= width / 2
        looking_down = (not (looking_left or looking_right)) and (
            eye_status["looking_down"] or (y + h) / max(height, 1) > 0.90
        )
        looking_at_camera = face_centered and eye_status["eyes_visible"] and not looking_down
        _draw_box(frame, x, y, x + w, y + h, "face fallback", (39, 174, 96))
        return {
            "face_visible": True,
            "face_centered": face_centered,
            "looking_at_camera": looking_at_camera,
            "looking_left": looking_left,
            "looking_right": looking_right,
            "looking_down": looking_down,
            "looking_away": not looking_at_camera,
            "face_count": len(faces),
            "eyes_visible": eye_status["eyes_visible"],
        }

    def _estimate_eyes_with_haar(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        upper_face = gray[y : y + int(h * 0.62), x : x + w]
        eyes = self.eye_cascade.detectMultiScale(upper_face, scaleFactor=1.08, minNeighbors=4, minSize=(18, 12))

        eyes_visible = len(eyes) >= 1
        looking_down = not eyes_visible
        for ex, ey, ew, eh in eyes[:2]:
            _draw_box(frame, x + ex, y + ey, x + ex + ew, y + ey + eh, "eye", (52, 152, 219))

        return {"eyes_visible": eyes_visible, "looking_down": looking_down}

    def _is_face_centered(self, min_x: int, max_x: int, width: int) -> bool:
        center_x = (min_x + max_x) / 2
        return abs(center_x - width / 2) < width * self.config.face_center_tolerance


def direction_label(looking_at_camera: bool, gaze: dict) -> str:
    if looking_at_camera:
        return "at camera"
    if gaze["looking_left"]:
        return "looking left"
    if gaze["looking_right"]:
        return "looking right"
    if gaze["looking_down"]:
        return "looking down"
    return "looking away"


def _draw_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
