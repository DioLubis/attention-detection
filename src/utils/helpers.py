from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def percentage(frame_results: list[dict], key: str) -> float:
    if not frame_results:
        return 0.0
    return sum(1 for frame in frame_results if frame.get(key)) / len(frame_results)
