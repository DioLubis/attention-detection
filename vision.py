from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


COCO_CLASS_IDS = {
    "person": 0,
    "laptop": 63,
    "cell phone": 67,
    "book": 73,
}


@dataclass(frozen=True)
class DetectionConfig:
    yolo_model_name: str = "yolov8n.pt"
    yolo_image_size: int = 416
    yolo_confidence: float = 0.35
    face_center_tolerance: float = 0.18
    head_left_threshold: float = -0.045
    head_right_threshold: float = 0.045
    iris_left_threshold: float = 0.42
    iris_right_threshold: float = 0.58
    down_threshold: float = 0.62


class FrameAnalyzer:
    """Runs object detection and approximate face/gaze estimation for one frame.

    The gaze output is a practical webcam approximation. It combines face
    landmarks, nose position, and iris position when MediaPipe refine_landmarks
    is available. It is not medical-grade or perfect eye tracking.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.yolo = self._load_yolo()
        self.face_mesh = self._load_face_mesh()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def analyze(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        display = cv2.flip(frame, 1)
        object_status = self._detect_objects(display)
        face_status = self._detect_face_and_gaze(display)

        person_count = object_status["person_count"]
        if self.yolo is None and face_status["face_visible"]:
            person_count = 1

        status = {
            "timestamp": time.time(),
            "person_count": person_count,
            "face_visible": face_status["face_visible"],
            "face_centered": face_status["face_centered"],
            "looking_at_camera": face_status["looking_at_camera"],
            "looking_left": face_status["looking_left"],
            "looking_right": face_status["looking_right"],
            "looking_down": face_status["looking_down"],
            "looking_away": face_status["looking_away"],
            "phone_detected": object_status["phone_detected"],
            "laptop_detected": object_status["laptop_detected"],
            "book_detected": object_status["book_detected"],
            "multiple_persons": person_count > 1,
        }

        self._draw_status(display, status)
        return display, status

    def _load_yolo(self):
        try:
            from ultralytics import YOLO

            return YOLO(self.config.yolo_model_name)
        except Exception:
            return None

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

    def _detect_objects(self, frame: np.ndarray) -> dict:
        empty = {
            "person_count": 0,
            "phone_detected": False,
            "laptop_detected": False,
            "book_detected": False,
        }
        if self.yolo is None:
            return empty

        try:
            results = self.yolo.predict(
                frame,
                imgsz=self.config.yolo_image_size,
                conf=self.config.yolo_confidence,
                classes=list(COCO_CLASS_IDS.values()),
                verbose=False,
            )
        except Exception:
            return empty

        status = empty.copy()
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, "")).lower()
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]

                if class_id == COCO_CLASS_IDS["person"]:
                    status["person_count"] += 1
                    self._draw_box(frame, x1, y1, x2, y2, f"person {confidence:.2f}", (41, 128, 185))
                elif class_id == COCO_CLASS_IDS["cell phone"] or class_name == "cell phone":
                    status["phone_detected"] = True
                    self._draw_box(frame, x1, y1, x2, y2, f"phone {confidence:.2f}", (192, 57, 43))
                elif class_id == COCO_CLASS_IDS["laptop"] or class_name == "laptop":
                    status["laptop_detected"] = True
                    self._draw_box(frame, x1, y1, x2, y2, f"laptop {confidence:.2f}", (142, 68, 173))
                elif class_id == COCO_CLASS_IDS["book"] or class_name == "book":
                    status["book_detected"] = True
                    self._draw_box(frame, x1, y1, x2, y2, f"book {confidence:.2f}", (211, 84, 0))

        return status

    def _detect_face_and_gaze(self, frame: np.ndarray) -> dict:
        if self.face_mesh is not None:
            return self._detect_face_mesh(frame)
        return self._detect_face_fallback(frame)

    def _empty_face_status(self) -> dict:
        return {
            "face_visible": False,
            "face_centered": False,
            "looking_at_camera": False,
            "looking_left": False,
            "looking_right": False,
            "looking_down": False,
            "looking_away": True,
        }

    def _detect_face_mesh(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return self._empty_face_status()

        height, width = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        min_x, max_x = int(min(xs) * width), int(max(xs) * width)
        min_y, max_y = int(min(ys) * height), int(max(ys) * height)

        face_centered = self._is_face_centered(min_x, max_x, width)
        gaze = self._estimate_gaze_from_landmarks(landmarks)
        looking_at_camera = face_centered and not (
            gaze["looking_left"] or gaze["looking_right"] or gaze["looking_down"]
        )
        looking_away = not looking_at_camera

        self._draw_box(frame, min_x, min_y, max_x, max_y, "face", (39, 174, 96))
        direction = self._direction_label(looking_at_camera, gaze)
        cv2.putText(frame, direction, (min_x, max(min_y - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (39, 174, 96), 2)

        return {
            "face_visible": True,
            "face_centered": face_centered,
            "looking_at_camera": looking_at_camera,
            "looking_left": gaze["looking_left"],
            "looking_right": gaze["looking_right"],
            "looking_down": gaze["looking_down"],
            "looking_away": looking_away,
        }

    def _estimate_gaze_from_landmarks(self, landmarks) -> dict:
        nose = landmarks[1]
        left_face = landmarks[234]
        right_face = landmarks[454]
        forehead = landmarks[10]
        chin = landmarks[152]

        # Head-pose proxy: if the nose shifts relative to cheek landmarks, the
        # head is likely rotated. This is robust enough for an MVP webcam UI.
        face_center_x = (left_face.x + right_face.x) / 2
        horizontal_offset = nose.x - face_center_x
        vertical_span = max(chin.y - forehead.y, 0.001)
        vertical_offset = (nose.y - forehead.y) / vertical_span

        iris_ratio = self._average_iris_ratio(landmarks)
        looking_left = horizontal_offset < self.config.head_left_threshold
        looking_right = horizontal_offset > self.config.head_right_threshold

        if iris_ratio is not None:
            looking_left = looking_left or iris_ratio < self.config.iris_left_threshold
            looking_right = looking_right or iris_ratio > self.config.iris_right_threshold

        return {
            "looking_left": looking_left,
            "looking_right": looking_right,
            "looking_down": vertical_offset >= self.config.down_threshold,
        }

    def _average_iris_ratio(self, landmarks) -> float | None:
        try:
            left_ratio = self._iris_ratio(landmarks, eye_left=33, eye_right=133, iris_points=(468, 469, 470, 471))
            right_ratio = self._iris_ratio(landmarks, eye_left=362, eye_right=263, iris_points=(473, 474, 475, 476))
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

    def _detect_face_fallback(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return self._empty_face_status()

        height, width = frame.shape[:2]
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        face_centered = self._is_face_centered(x, x + w, width)
        lower_face_ratio = (y + h) / max(height, 1)
        looking_down = lower_face_ratio > 0.86
        looking_at_camera = face_centered and not looking_down
        self._draw_box(frame, x, y, x + w, y + h, "face fallback", (39, 174, 96))
        return {
            "face_visible": True,
            "face_centered": face_centered,
            "looking_at_camera": looking_at_camera,
            "looking_left": False,
            "looking_right": False,
            "looking_down": looking_down,
            "looking_away": not looking_at_camera,
        }

    def _is_face_centered(self, min_x: int, max_x: int, width: int) -> bool:
        center_x = (min_x + max_x) / 2
        return abs(center_x - width / 2) < width * self.config.face_center_tolerance

    def _direction_label(self, looking_at_camera: bool, gaze: dict) -> str:
        if looking_at_camera:
            return "at camera"
        if gaze["looking_down"]:
            return "looking down"
        if gaze["looking_left"]:
            return "looking left"
        if gaze["looking_right"]:
            return "looking right"
        return "looking away"

    def _draw_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, color: tuple[int, int, int]) -> None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _draw_status(self, frame: np.ndarray, status: dict) -> None:
        lines = [
            f"Persons: {status['person_count']}",
            f"Face centered: {'yes' if status['face_centered'] else 'no'}",
            f"Gaze approx: {self._direction_label(status['looking_at_camera'], status)}",
            f"Phone/Laptop/Book: {int(status['phone_detected'])}/{int(status['laptop_detected'])}/{int(status['book_detected'])}",
        ]
        y = 28
        for line in lines:
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            y += 28
