from __future__ import annotations

from datetime import datetime, timezone

from metrics import build_question_metrics_summary
from sentiment import build_interview_sentiment, build_question_sentiment


def build_question_summary(
    question_id: int,
    question_text: str,
    started_at: str | None,
    stopped_at: str,
    frame_results: list[dict],
) -> dict:
    question_summary = build_question_metrics_summary(
        question_id=question_id,
        question_text=question_text,
        frame_results=frame_results,
        started_at=started_at,
        stopped_at=stopped_at,
    )
    ratios = question_summary["metrics"]
    sentiment = build_question_sentiment(ratios)
    return question_summary | {
        "started_at": started_at,
        "stopped_at": stopped_at,
        **sentiment,
        "disclaimer": sentiment["human_in_the_loop_disclaimer"],
        "summary": sentiment["short_summary"],
    }


def build_final_report(candidate_context: dict, question_reports: list[dict]) -> dict:
    interview_sentiment = build_interview_sentiment(question_reports)

    return {
        "report_id": f"karierly-vision-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system_name": "Karierly Interview Vision",
        "candidate_context": candidate_context,
        "privacy": {
            "raw_video_stored": False,
            "raw_images_stored": False,
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
                "Raw video and raw images are not stored by default.",
            ],
        },
    }
