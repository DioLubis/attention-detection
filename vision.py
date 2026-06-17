from __future__ import annotations

import cv2
import numpy as np


class FrameAnalyzer:
    def __init__(self) -> None:
        self.yolo = self._load_yolo()
        self.face_mesh = self._load_face_mesh()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.person_class_ids = {0}
        self.phone_names = {"cell phone", "mobile phone", "phone"}

    def analyze(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        object_status = self._detect_objects(display)
        face_status = self._detect_face_and_gaze(display)

        status = {
            "face_visible": face_status["face_visible"],
            "person_detected": object_status["person_detected"] or face_status["face_visible"],
            "looking_at_camera": face_status["looking_at_camera"],
            "looking_away": face_status["looking_away"],
            "phone_detected": object_status["phone_detected"],
            "multiple_persons": object_status["person_count"] > 1,
        }

        self._draw_status(display, status, object_status["person_count"])
        return display, status

    def _load_yolo(self):
        try:
            from ultralytics import YOLO

            return YOLO("yolov8n.pt")
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
        if self.yolo is None:
            return {"person_detected": False, "person_count": 0, "phone_detected": False}

        person_count = 0
        phone_detected = False
        try:
            results = self.yolo.predict(frame, imgsz=416, conf=0.35, verbose=False)
        except Exception:
            return {"person_detected": False, "person_count": 0, "phone_detected": False}

        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, "")).lower()
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]

                if class_id in self.person_class_ids:
                    person_count += 1
                    self._draw_box(frame, x1, y1, x2, y2, f"person {confidence:.2f}", (41, 128, 185))
                elif class_name in self.phone_names:
                    phone_detected = True
                    self._draw_box(frame, x1, y1, x2, y2, f"phone {confidence:.2f}", (192, 57, 43))

        return {
            "person_detected": person_count > 0,
            "person_count": person_count,
            "phone_detected": phone_detected,
        }

    def _detect_face_and_gaze(self, frame: np.ndarray) -> dict:
        if self.face_mesh is not None:
            return self._detect_face_mesh(frame)
        return self._detect_face_fallback(frame)

    def _detect_face_mesh(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return {"face_visible": False, "looking_at_camera": False, "looking_away": True}

        height, width = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        min_x, max_x = int(min(xs) * width), int(max(xs) * width)
        min_y, max_y = int(min(ys) * height), int(max(ys) * height)

        nose = landmarks[1]
        left_face = landmarks[234]
        right_face = landmarks[454]
        chin = landmarks[152]
        forehead = landmarks[10]

        face_center_x = (left_face.x + right_face.x) / 2
        horizontal_offset = nose.x - face_center_x
        vertical_span = max(chin.y - forehead.y, 0.001)
        vertical_offset = (nose.y - forehead.y) / vertical_span

        looking_at_camera = abs(horizontal_offset) < 0.035 and vertical_offset < 0.58
        looking_down = vertical_offset >= 0.62
        looking_away = not looking_at_camera or looking_down

        self._draw_box(frame, min_x, min_y, max_x, max_y, "face", (39, 174, 96))
        direction = "at camera" if looking_at_camera else "away"
        cv2.putText(frame, direction, (min_x, max(min_y - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (39, 174, 96), 2)

        return {
            "face_visible": True,
            "looking_at_camera": looking_at_camera,
            "looking_away": looking_away,
        }

    def _detect_face_fallback(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return {"face_visible": False, "looking_at_camera": False, "looking_away": True}

        height, width = frame.shape[:2]
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        center_x = x + w / 2
        centered = abs(center_x - width / 2) < width * 0.18
        self._draw_box(frame, x, y, x + w, y + h, "face", (39, 174, 96))
        return {
            "face_visible": True,
            "looking_at_camera": centered,
            "looking_away": not centered,
        }

    def _draw_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, color: tuple[int, int, int]) -> None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _draw_status(self, frame: np.ndarray, status: dict, person_count: int) -> None:
        lines = [
            f"Face visible: {'yes' if status['face_visible'] else 'no'}",
            f"Person count: {person_count}",
            f"Camera focus: {'yes' if status['looking_at_camera'] else 'review'}",
            f"Phone indicator: {'yes' if status['phone_detected'] else 'no'}",
        ]
        y = 28
        for line in lines:
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            y += 28
