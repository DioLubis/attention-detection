from __future__ import annotations

from collections import Counter
from typing import Any


DISCLAIMER = (
    "This observation is only a visual support indicator and must not be used "
    "as an automatic hiring decision. A human recruiter must review the answer "
    "and surrounding interview context."
)

LABEL_PRIORITY = (
    "Unstable Interview Condition",
    "External Assistance Indicator",
    "Possible Device Usage Indicator",
    "Review Recommended",
    "Light Review Needed",
    "Mostly Focused",
    "Stable Focus",
)


def build_question_sentiment(metrics: dict[str, float]) -> dict[str, str]:
    label = classify_observation(metrics)
    return {
        "observation_label": label,
        "short_summary": _short_summary(label),
        "recruiter_recommendation": _recruiter_recommendation(label),
        "human_in_the_loop_disclaimer": DISCLAIMER,
    }


def classify_observation(metrics: dict[str, float]) -> str:
    face_visible = _metric(metrics, "face_visible_percentage")
    person_visible = _metric(metrics, "person_visible_percentage")
    looking_at_camera = _metric(metrics, "looking_at_camera_percentage")
    looking_away = _metric(metrics, "looking_away_percentage")
    looking_down = _metric(metrics, "looking_down_percentage")
    phone_detected = _metric(metrics, "phone_detected_percentage")
    multiple_persons = _metric(metrics, "multiple_persons_percentage")

    if face_visible < 60 or person_visible < 70:
        return "Unstable Interview Condition"
    if multiple_persons >= 10:
        return "External Assistance Indicator"
    if phone_detected >= 15 or looking_down >= 40:
        return "Possible Device Usage Indicator"
    if looking_away > 50 or looking_down > 35 or face_visible < 70:
        return "Review Recommended"
    if 30 <= looking_away <= 50 and phone_detected < 15 and multiple_persons < 5:
        return "Light Review Needed"
    if looking_at_camera >= 80 and phone_detected < 5 and multiple_persons < 5 and face_visible >= 85:
        return "Stable Focus"
    if 60 <= looking_at_camera <= 79 and phone_detected < 10 and multiple_persons < 5 and face_visible >= 75:
        return "Mostly Focused"
    return "Mostly Focused"


def build_interview_sentiment(question_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [summary.get("observation_label", "Mostly Focused") for summary in question_summaries]
    final_label = _highest_priority_label(labels)
    completed_questions = len(question_summaries)
    return {
        "observation_label": final_label if labels else "No Completed Questions",
        "short_summary": _interview_summary(final_label, completed_questions),
        "recruiter_recommendation": _interview_recommendation(final_label, completed_questions),
        "human_in_the_loop_disclaimer": DISCLAIMER,
        "label_counts": dict(Counter(labels)),
    }


def _metric(metrics: dict[str, float], key: str) -> float:
    return float(metrics.get(key, 0.0) or 0.0)


def _highest_priority_label(labels: list[str]) -> str:
    if not labels:
        return "No Completed Questions"
    for label in LABEL_PRIORITY:
        if label in labels:
            return label
    return "Mostly Focused"


def _short_summary(label: str) -> str:
    summaries = {
        "Stable Focus": "The candidate was visually present and maintained a stable camera-facing interview condition for most of the answer.",
        "Mostly Focused": "The candidate was visually present for most of the answer with natural gaze movement.",
        "Light Review Needed": "The candidate was visually present for most of the answer, but looked away from the camera several times.",
        "Review Recommended": "The answer segment includes notable periods of looking away, looking down, or reduced face visibility.",
        "Unstable Interview Condition": "The visual conditions were not consistently stable because face or person visibility was limited during part of the answer.",
        "Possible Device Usage Indicator": "A phone-like object or repeated downward gaze indicator appeared during the answer.",
        "External Assistance Indicator": "More than one person appeared in the frame for a notable portion of the answer.",
    }
    return summaries.get(label, "The answer segment was processed and is available for recruiter review.")


def _recruiter_recommendation(label: str) -> str:
    recommendations = {
        "Stable Focus": "The recruiter can proceed with normal review of the candidate's answer content.",
        "Mostly Focused": "The recruiter can proceed with normal review of the candidate's answer content.",
        "Light Review Needed": "The recruiter may review this answer segment if additional context is needed.",
        "Review Recommended": "The recruiter should review this answer segment alongside notes, transcript, or interview context.",
        "Unstable Interview Condition": "The recruiter should consider whether the visual data is sufficient before relying on this segment.",
        "Possible Device Usage Indicator": "The recruiter may review the segment to determine whether the detected visual context is relevant.",
        "External Assistance Indicator": "The recruiter may review the segment to understand whether another visible person affected the interview context.",
    }
    return recommendations.get(label, "The recruiter should review the observation in context.")


def _interview_summary(final_label: str, completed_questions: int) -> str:
    if completed_questions == 0:
        return "No completed question segments are available for interview-level observation."
    if final_label == "Stable Focus":
        return "Across the completed questions, the visual interview condition was consistently stable."
    if final_label == "Mostly Focused":
        return "Across the completed questions, the candidate was generally visible with mostly stable interview conditions."
    if final_label == "Light Review Needed":
        return "Across the completed questions, most segments were usable, with some visual indicators that may benefit from light recruiter review."
    if final_label == "Review Recommended":
        return "Across the completed questions, at least one segment contains visual conditions that should be reviewed with additional context."
    if final_label == "Unstable Interview Condition":
        return "Across the completed questions, at least one segment had limited face or person visibility, so recruiter review is required."
    if final_label == "Possible Device Usage Indicator":
        return "Across the completed questions, at least one segment included a possible device or downward-gaze indicator for recruiter review."
    if final_label == "External Assistance Indicator":
        return "Across the completed questions, at least one segment included a multiple-person indicator for recruiter review."
    return f"{completed_questions} completed question segment(s) are available for recruiter review."


def _interview_recommendation(final_label: str, completed_questions: int) -> str:
    if completed_questions == 0:
        return "Complete at least one question segment before using the interview-level observation."
    if final_label in {"Stable Focus", "Mostly Focused"}:
        return "Recruiter may continue normal review and optionally replay flagged segments."
    return "Recruiter should review flagged segments with interview notes before interpreting the observation."
