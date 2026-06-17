from __future__ import annotations

from datetime import datetime, timezone

from metrics import build_question_metrics_summary


def _label_for(ratios: dict) -> str:
    if ratios["person_visible_percentage"] < 50 or ratios["face_visible_percentage"] < 45:
        return "Unstable Interview Condition"
    if ratios["multiple_persons_percentage"] >= 15:
        return "External Assistance Indicator"
    if ratios["phone_detected_percentage"] >= 12:
        return "Possible Device Usage Indicator"
    if ratios["looking_away_percentage"] >= 45:
        return "Review Recommended"
    if ratios["looking_away_percentage"] >= 25 or ratios["face_visible_percentage"] < 75:
        return "Light Review Needed"
    if ratios["looking_at_camera_percentage"] >= 70:
        return "Stable Focus"
    return "Mostly Focused"


def _summary_for(label: str, ratios: dict) -> str:
    if label == "Stable Focus":
        return "The candidate was generally visible and maintained a stable camera-facing interview condition."
    if label == "Mostly Focused":
        return "The candidate was generally present with sufficient face visibility. Some natural gaze movement was observed."
    if label == "Light Review Needed":
        return "The interview condition was mostly usable, with some periods of reduced face visibility or looking away."
    if label == "Review Recommended":
        return "A recruiter may want to review this answer because looking-away indicators appeared for a notable portion of the response."
    if label == "External Assistance Indicator":
        return "Multiple-person indicators appeared during this answer. Recruiter review is recommended to understand the interview context."
    if label == "Possible Device Usage Indicator":
        return "A phone-like object indicator appeared during this answer. Recruiter review is recommended before drawing any conclusion."
    return "Visual conditions were not stable enough for a reliable observation summary."


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
    label = _label_for(ratios)
    return question_summary | {
        "started_at": started_at,
        "stopped_at": stopped_at,
        "observation_label": label,
        "summary": _summary_for(label, ratios),
    }


def build_final_report(candidate_context: dict, question_reports: list[dict]) -> dict:
    labels = [report["observation_label"] for report in question_reports]
    if "Unstable Interview Condition" in labels:
        final_label = "Unstable Interview Condition"
    elif "External Assistance Indicator" in labels:
        final_label = "External Assistance Indicator"
    elif "Possible Device Usage Indicator" in labels:
        final_label = "Possible Device Usage Indicator"
    elif "Review Recommended" in labels:
        final_label = "Review Recommended"
    elif "Light Review Needed" in labels:
        final_label = "Light Review Needed"
    elif labels and all(label == "Stable Focus" for label in labels):
        final_label = "Stable Focus"
    else:
        final_label = "Mostly Focused" if labels else "No Completed Questions"

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
            "observation_label": final_label,
            "summary": (
                "This report summarizes visible interview context for recruiter review. "
                "It is not a pass/fail decision and must not be used as the sole basis for hiring outcomes."
            ),
            "ethical_notice": [
                "Do not infer honesty, personality, intelligence, confidence, or hiring suitability from this report.",
                "A human recruiter must review all observations in context.",
                "Raw video and raw images are not stored by default.",
            ],
        },
    }
