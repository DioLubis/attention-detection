from __future__ import annotations

from datetime import datetime
from typing import Any


PERCENTAGE_FIELDS = (
    "face_visible",
    "person_visible",
    "looking_at_camera",
    "looking_away",
    "looking_left",
    "looking_right",
    "looking_down",
    "phone_detected",
    "laptop_detected",
    "book_detected",
    "multiple_persons",
    "unstable_frame",
)


def build_question_metrics_summary(
    question_id: int,
    question_text: str,
    frame_results: list[dict[str, Any]],
    started_at: str | None = None,
    stopped_at: str | None = None,
) -> dict[str, Any]:
    aggregate = aggregate_frame_results(frame_results, started_at=started_at, stopped_at=stopped_at)
    return {
        "question_id": question_id,
        "question_text": question_text,
        "duration_seconds": aggregate["total_duration_seconds"],
        "total_frames": aggregate["total_frames"],
        "metrics": aggregate["metrics"],
    }


def aggregate_frame_results(
    frame_results: list[dict[str, Any]],
    started_at: str | None = None,
    stopped_at: str | None = None,
) -> dict[str, Any]:
    total_frames = len(frame_results)
    if total_frames == 0:
        return {
            "total_duration_seconds": 0.0,
            "total_frames": 0,
            "metrics": {f"{field}_percentage": 0.0 for field in PERCENTAGE_FIELDS},
        }

    counts = {field: 0 for field in PERCENTAGE_FIELDS}
    for frame in frame_results:
        person_count = int(frame.get("person_count") or 0)
        face_visible = bool(frame.get("face_visible"))
        person_visible = person_count > 0
        multiple_persons = person_count > 1 or bool(frame.get("multiple_persons"))

        counts["face_visible"] += int(face_visible)
        counts["person_visible"] += int(person_visible)
        counts["looking_at_camera"] += int(bool(frame.get("looking_at_camera")))
        counts["looking_away"] += int(bool(frame.get("looking_away")))
        counts["looking_left"] += int(bool(frame.get("looking_left")))
        counts["looking_right"] += int(bool(frame.get("looking_right")))
        counts["looking_down"] += int(bool(frame.get("looking_down")))
        counts["phone_detected"] += int(bool(frame.get("phone_detected")))
        counts["laptop_detected"] += int(bool(frame.get("laptop_detected")))
        counts["book_detected"] += int(bool(frame.get("book_detected")))
        counts["multiple_persons"] += int(multiple_persons)
        counts["unstable_frame"] += int((not face_visible) or (not person_visible) or multiple_persons)

    return {
        "total_duration_seconds": calculate_duration_seconds(frame_results, started_at, stopped_at),
        "total_frames": total_frames,
        "metrics": {f"{field}_percentage": _percentage(count, total_frames) for field, count in counts.items()},
    }


def calculate_duration_seconds(
    frame_results: list[dict[str, Any]],
    started_at: str | None = None,
    stopped_at: str | None = None,
) -> float:
    timestamps = [
        float(frame["timestamp"])
        for frame in frame_results
        if isinstance(frame.get("timestamp"), (int, float))
    ]
    if len(timestamps) >= 2:
        return round(max(timestamps) - min(timestamps), 2)

    if started_at and stopped_at:
        start = _parse_datetime(started_at)
        stop = _parse_datetime(stopped_at)
        if start and stop:
            return round(max((stop - start).total_seconds(), 0.0), 2)

    return 0.0


def _percentage(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((count / total) * 100, 2)


def _parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
