from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GazeConfig:
    face_center_tolerance: float = 0.18
    head_left_threshold: float = -0.018
    head_right_threshold: float = 0.018
    iris_left_threshold: float = 0.47
    iris_right_threshold: float = 0.53
    iris_down_threshold: float = 0.76
    head_down_threshold: float = 0.68
    pupil_left_threshold: float = 0.46
    pupil_right_threshold: float = 0.54
    pupil_min_confidence: float = 0.20
    smoothing_window: int = 5


class GazeEstimator:
    """Approximate face/gaze estimator using MediaPipe Face Mesh or OpenCV fallback."""

    def __init__(self, config: GazeConfig | None = None) -> None:
        self.config = config or GazeConfig()
        self.face_mesh = self._load_face_mesh()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
        self.direction_history: deque[str] = deque(maxlen=self.config.smoothing_window)

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
        self.direction_history.clear()
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
            "gaze_horizontal_score": 0.0,
            "gaze_vertical_score": 0.0,
            "raw_gaze_direction": "none",
            "pupil_detected": False,
            "pupil_horizontal_ratio": 0.5,
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
        gaze = self._estimate_gaze_from_landmarks(frame, landmarks)
        raw_direction = self._raw_direction(gaze)
        direction = self._smooth_direction(raw_direction)
        gaze["looking_left"] = direction == "left"
        gaze["looking_right"] = direction == "right"
        gaze["looking_down"] = direction == "down"
        looking_at_camera = direction == "center" and face_centered

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
            "gaze_horizontal_score": gaze["horizontal_score"],
            "gaze_vertical_score": gaze["vertical_score"],
            "raw_gaze_direction": raw_direction,
            "pupil_detected": gaze["pupil_detected"],
            "pupil_horizontal_ratio": gaze["pupil_horizontal_ratio"],
        }

    def _estimate_gaze_from_landmarks(self, frame: np.ndarray, landmarks) -> dict:
        nose = landmarks[1]
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        forehead = landmarks[10]
        chin = landmarks[152]

        eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
        eye_width = max(abs(right_eye_outer.x - left_eye_outer.x), 0.001)
        horizontal_offset = (nose.x - eye_center_x) / eye_width
        vertical_span = max(chin.y - forehead.y, 0.001)
        vertical_offset = (nose.y - forehead.y) / vertical_span

        iris_ratio = self._average_iris_ratio(landmarks)
        iris_vertical_ratio = self._average_iris_vertical_ratio(landmarks)
        pupil_ratio, pupil_confidence = self._average_dark_pupil_ratio(frame, landmarks)
        iris_horizontal = 0.0 if iris_ratio is None else iris_ratio - 0.5

        pupil_detected = pupil_ratio is not None and pupil_confidence >= self.config.pupil_min_confidence
        if pupil_detected:
            # Direct dark-pupil movement is the primary signal. This allows eye
            # gaze changes to register even while the head remains centered.
            pupil_horizontal = pupil_ratio - 0.5
            horizontal_score = (pupil_horizontal * 0.72) + (iris_horizontal * 0.18) + (horizontal_offset * 0.10)
            looking_left = pupil_ratio < self.config.pupil_left_threshold
            looking_right = pupil_ratio > self.config.pupil_right_threshold
        else:
            # Glasses reflections may hide the pupil. Fall back to iris
            # landmarks and head yaw instead of returning a neutral result.
            horizontal_score = (horizontal_offset * 0.65) + (iris_horizontal * 0.35)
            looking_left = horizontal_score < self.config.head_left_threshold or (
                iris_ratio is not None and iris_ratio < self.config.iris_left_threshold
            )
            looking_right = horizontal_score > self.config.head_right_threshold or (
                iris_ratio is not None and iris_ratio > self.config.iris_right_threshold
            )

        vertical_score = iris_vertical_ratio if iris_vertical_ratio is not None else 0.5
        looking_down = (
            not (looking_left or looking_right)
            and iris_vertical_ratio is not None
            and vertical_score >= self.config.iris_down_threshold
            and vertical_offset >= self.config.head_down_threshold
        )

        return {
            "looking_left": looking_left,
            "looking_right": looking_right,
            "looking_down": looking_down,
            "horizontal_score": horizontal_score,
            "vertical_score": vertical_score,
            "pupil_detected": pupil_detected,
            "pupil_horizontal_ratio": pupil_ratio if pupil_ratio is not None else 0.5,
        }

    def _average_dark_pupil_ratio(self, frame: np.ndarray, landmarks) -> tuple[float | None, float]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_specs = (
            ((33, 160, 158, 133, 153, 144), 33, 133),
            ((362, 385, 387, 263, 373, 380), 362, 263),
        )

        ratios = []
        confidences = []
        for polygon_indices, corner_a, corner_b in eye_specs:
            result = self._dark_pupil_ratio_for_eye(
                frame,
                gray,
                landmarks,
                polygon_indices,
                corner_a,
                corner_b,
                width,
                height,
            )
            if result is not None:
                ratio, confidence = result
                ratios.append(ratio)
                confidences.append(confidence)

        if not ratios:
            return None, 0.0
        return float(np.mean(ratios)), float(np.mean(confidences))

    def _dark_pupil_ratio_for_eye(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        landmarks,
        polygon_indices: tuple[int, ...],
        corner_a: int,
        corner_b: int,
        width: int,
        height: int,
    ) -> tuple[float, float] | None:
        points = np.array(
            [[int(landmarks[index].x * width), int(landmarks[index].y * height)] for index in polygon_indices],
            dtype=np.int32,
        )
        x, y, w, h = cv2.boundingRect(points)
        if w < 12 or h < 6:
            return None

        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, width), min(y + h, height)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        local_points = points - np.array([x1, y1])
        mask = np.zeros_like(roi, dtype=np.uint8)
        cv2.fillPoly(mask, [local_points], 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

        valid_pixels = roi[mask > 0]
        if valid_pixels.size < 20:
            return None

        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        dark_threshold = min(float(np.percentile(valid_pixels, 24)), float(np.mean(valid_pixels) - 8))
        dark_mask = np.zeros_like(mask)
        dark_mask[(blurred <= dark_threshold) & (mask > 0)] = 255
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        eye_area = max(cv2.countNonZero(mask), 1)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2 or area > eye_area * 0.45:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            vertical_center_score = 1.0 - min(abs(cy - roi.shape[0] / 2) / max(roi.shape[0] / 2, 1), 1.0)
            candidates.append((area * (0.5 + vertical_center_score), cx, cy, area))

        if not candidates:
            return None

        _, pupil_x, pupil_y, pupil_area = max(candidates, key=lambda item: item[0])
        global_x = x1 + pupil_x
        global_y = y1 + pupil_y
        corner_x_a = landmarks[corner_a].x * width
        corner_x_b = landmarks[corner_b].x * width
        eye_min = min(corner_x_a, corner_x_b)
        eye_max = max(corner_x_a, corner_x_b)
        ratio = float(np.clip((global_x - eye_min) / max(eye_max - eye_min, 1.0), 0.0, 1.0))

        contrast = max(float(np.mean(valid_pixels) - dark_threshold), 0.0)
        area_ratio = pupil_area / eye_area
        confidence = float(np.clip((contrast / 35.0) * 0.6 + min(area_ratio / 0.12, 1.0) * 0.4, 0.0, 1.0))
        cv2.circle(frame, (int(global_x), int(global_y)), 3, (0, 255, 255), -1)
        return ratio, confidence

    def _raw_direction(self, gaze: dict) -> str:
        if gaze["looking_left"]:
            return "left"
        if gaze["looking_right"]:
            return "right"
        if gaze["looking_down"]:
            return "down"
        return "center"

    def _smooth_direction(self, direction: str) -> str:
        self.direction_history.append(direction)
        counts = Counter(self.direction_history)

        # Two matching frames are enough for left/right responsiveness, while
        # down requires three frames to avoid accidental downward classification.
        if counts["left"] >= 2:
            return "left"
        if counts["right"] >= 2:
            return "right"
        if counts["down"] >= 3:
            return "down"
        return "center"

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
        looking_down = (not (looking_left or looking_right)) and (y + h) / max(height, 1) > 0.94
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
