from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.processing.metrics import build_question_metrics_summary
from src.processing.sentiment import build_interview_sentiment, build_question_sentiment
from src.utils.config import APP_TITLE, GLOBAL_DISCLAIMER, JOB_CONTEXT, REPORTS_DIR
from src.utils.helpers import write_json


def build_question_report(
    question_id: int,
    question_text: str,
    started_at: str | None,
    stopped_at: str,
    frame_results: list[dict],
) -> dict[str, Any]:
    question_summary = build_question_metrics_summary(
        question_id=question_id,
        question_text=question_text,
        frame_results=frame_results,
        started_at=started_at,
        stopped_at=stopped_at,
    )
    sentiment = build_question_sentiment(question_summary["metrics"])
    return question_summary | {
        "started_at": started_at,
        "stopped_at": stopped_at,
        **sentiment,
        "summary": sentiment["short_summary"],
    }


def build_internal_report(candidate_context: dict, question_reports: list[dict]) -> dict[str, Any]:
    interview_sentiment = build_interview_sentiment(question_reports)
    return {
        "report_id": f"karierly-vision-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system_name": APP_TITLE,
        "candidate_context": candidate_context,
        "privacy": {
            "raw_video_stored": False,
            "raw_images_stored": False,
            "biometric_templates_stored": False,
            "stored_data_type": "summarized_metrics_only",
            "human_in_the_loop_required": True,
        },
        "questions": question_reports,
        "overall_summary": {
            **interview_sentiment,
            "summary": interview_sentiment["short_summary"],
            "ethical_notice": [
                "Do not infer honesty, personality, intelligence, confidence, or hiring suitability from this report.",
                "A human recruiter must review all observations in context.",
                "Raw video, raw images, and biometric templates are not stored.",
            ],
        },
    }


def build_export_report(
    candidate_context: dict[str, Any],
    questions: list[str],
    question_reports: list[dict[str, Any]],
    session_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    session_id = session_id or str(uuid.uuid4())
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    final_summary = build_interview_sentiment(question_reports)

    return {
        "session_id": session_id,
        "created_at": created_at,
        "prototype_name": APP_TITLE,
        "candidate": {
            "name": candidate_context["candidate_name"],
            "position": candidate_context["position"],
        },
        "job": JOB_CONTEXT,
        "interview_questions": questions,
        "questions": [_build_question_export(report) for report in question_reports],
        "final_summary": {
            "overall_label": final_summary["observation_label"],
            "summary": final_summary["short_summary"],
            "recommended_action": final_summary["recruiter_recommendation"],
        },
        "global_disclaimer": GLOBAL_DISCLAIMER,
        "privacy": {
            "raw_video_saved": False,
            "raw_images_saved": False,
            "biometric_templates_saved": False,
            "stored_data": "summarized_metrics_and_generated_observation_text_only",
        },
    }


def save_export_report(report: dict[str, Any], reports_dir: Path = REPORTS_DIR) -> Path:
    path = reports_dir / f"interview_observation_{report['session_id']}.json"
    write_json(path, report)
    return path


def _build_question_export(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": report["question_id"],
        "question_text": report["question_text"],
        "duration_seconds": report["duration_seconds"],
        "metrics": report["metrics"],
        "observation": {
            "label": report["observation_label"],
            "summary": report["short_summary"],
            "recommendation": report["recruiter_recommendation"],
            "disclaimer": report["human_in_the_loop_disclaimer"],
        },
    }
