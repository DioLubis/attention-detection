from __future__ import annotations

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
class ObjectDetectionConfig:
    model_name: str = "yolov8n.pt"
    image_size: int = 416
    confidence: float = 0.35


class ObjectDetector:
    """YOLO COCO object detector with a no-crash fallback when unavailable."""

    def __init__(self, config: ObjectDetectionConfig | None = None) -> None:
        self.config = config or ObjectDetectionConfig()
        self.model = self._load_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, frame: np.ndarray) -> dict:
        status = {
            "person_count": 0,
            "phone_detected": False,
            "laptop_detected": False,
            "book_detected": False,
        }
        if self.model is None:
            return status

        try:
            results = self.model.predict(
                frame,
                imgsz=self.config.image_size,
                conf=self.config.confidence,
                classes=list(COCO_CLASS_IDS.values()),
                verbose=False,
            )
        except Exception:
            return status

        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, "")).lower()
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]

                if class_id == COCO_CLASS_IDS["person"]:
                    status["person_count"] += 1
                    _draw_box(frame, x1, y1, x2, y2, f"person {confidence:.2f}", (41, 128, 185))
                elif class_id == COCO_CLASS_IDS["cell phone"] or class_name == "cell phone":
                    status["phone_detected"] = True
                    _draw_box(frame, x1, y1, x2, y2, f"phone {confidence:.2f}", (192, 57, 43))
                elif class_id == COCO_CLASS_IDS["laptop"] or class_name == "laptop":
                    status["laptop_detected"] = True
                    _draw_box(frame, x1, y1, x2, y2, f"laptop {confidence:.2f}", (142, 68, 173))
                elif class_id == COCO_CLASS_IDS["book"] or class_name == "book":
                    status["book_detected"] = True
                    _draw_box(frame, x1, y1, x2, y2, f"book {confidence:.2f}", (211, 84, 0))

        return status

    def _load_model(self):
        try:
            from ultralytics import YOLO

            return YOLO(self.config.model_name)
        except Exception:
            return None


def _draw_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
