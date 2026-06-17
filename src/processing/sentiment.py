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
        "short_summary": _short_summary(label, metrics),
        "recruiter_recommendation": _recruiter_recommendation(label, metrics),
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
    aggregate_notes = _aggregate_interview_notes(question_summaries)
    return {
        "observation_label": final_label if labels else "No Completed Questions",
        "short_summary": _interview_summary(final_label, completed_questions, aggregate_notes),
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


def _short_summary(label: str, metrics: dict[str, float]) -> str:
    base = {
        "Stable Focus": "The candidate was visually present and maintained a stable camera-facing interview condition for most of the answer.",
        "Mostly Focused": "The candidate was visually present for most of the answer with natural gaze movement.",
        "Light Review Needed": "The candidate was visually present for most of the answer, with some attention indicators that may benefit from light review.",
        "Review Recommended": "The answer segment includes visual conditions that should be reviewed with additional context.",
        "Unstable Interview Condition": "The visual conditions were not consistently stable because face or person visibility was limited during part of the answer.",
        "Possible Device Usage Indicator": "A possible device or repeated downward-gaze indicator appeared during the answer.",
        "External Assistance Indicator": "More than one person appeared in the frame for a notable portion of the answer.",
    }.get(label, "The answer segment was processed and is available for recruiter review.")

    details = _metric_detail_sentences(metrics)
    if not details:
        return base
    return f"{base} " + " ".join(details)


def _recruiter_recommendation(label: str, metrics: dict[str, float] | None = None) -> str:
    recommendations = {
        "Stable Focus": "The recruiter can proceed with normal review of the candidate's answer content.",
        "Mostly Focused": "The recruiter can proceed with normal review of the candidate's answer content.",
        "Light Review Needed": "The recruiter may review this answer segment if additional context is needed.",
        "Review Recommended": "The recruiter should review this answer segment alongside notes, transcript, or interview context.",
        "Unstable Interview Condition": "The recruiter should consider whether the visual data is sufficient before relying on this segment.",
        "Possible Device Usage Indicator": "The recruiter may review the segment to determine whether the detected visual context is relevant.",
        "External Assistance Indicator": "The recruiter may review the segment to understand whether another visible person affected the interview context.",
    }
    recommendation = recommendations.get(label, "The recruiter should review the observation in context.")
    if metrics:
        flagged = _flag_names(metrics)
        if len(flagged) > 1:
            recommendation += " Review all noted visual indicators together rather than focusing on a single signal."
    return recommendation


def _metric_detail_sentences(metrics: dict[str, float]) -> list[str]:
    details = []
    multiple = _metric(metrics, "multiple_persons_percentage")
    phone = _metric(metrics, "phone_detected_percentage")
    away = _metric(metrics, "looking_away_percentage")
    left = _metric(metrics, "looking_left_percentage")
    right = _metric(metrics, "looking_right_percentage")
    down = _metric(metrics, "looking_down_percentage")
    face = _metric(metrics, "face_visible_percentage")
    person = _metric(metrics, "person_visible_percentage")
    laptop = _metric(metrics, "laptop_detected_percentage")
    book = _metric(metrics, "book_detected_percentage")

    if multiple >= 5:
        details.append(f"Multiple-person indicators appeared in {multiple:.1f}% of analyzed frames.")
    if phone >= 5:
        details.append(f"Phone indicators appeared in {phone:.1f}% of analyzed frames.")
    if down >= 15:
        details.append(f"Looking-down indicators appeared in {down:.1f}% of analyzed frames.")
    if away >= 25:
        details.append(f"Looking-away indicators appeared in {away:.1f}% of analyzed frames.")
    if left >= 15 or right >= 15:
        details.append(f"Side-gaze indicators were observed, with left at {left:.1f}% and right at {right:.1f}% of analyzed frames.")
    if face < 75:
        details.append(f"Face visibility was {face:.1f}% across analyzed frames.")
    if person < 90:
        details.append(f"Person visibility was {person:.1f}% across analyzed frames.")
    if laptop >= 15:
        details.append(f"A laptop-like object was visible in {laptop:.1f}% of analyzed frames.")
    if book >= 10:
        details.append(f"Book or paper-like indicators appeared in {book:.1f}% of analyzed frames.")
    return details


def _flag_names(metrics: dict[str, float]) -> list[str]:
    names = []
    if _metric(metrics, "multiple_persons_percentage") >= 5:
        names.append("multiple-person")
    if _metric(metrics, "phone_detected_percentage") >= 5:
        names.append("phone")
    if _metric(metrics, "looking_down_percentage") >= 15:
        names.append("looking-down")
    if _metric(metrics, "looking_away_percentage") >= 25:
        names.append("looking-away")
    if _metric(metrics, "face_visible_percentage") < 75:
        names.append("face-visibility")
    return names


def _aggregate_interview_notes(question_summaries: list[dict[str, Any]]) -> list[str]:
    if not question_summaries:
        return []

    counters = Counter()
    for report in question_summaries:
        metrics = report.get("metrics", {})
        for name in _flag_names(metrics):
            counters[name] += 1

    notes = []
    for name, count in counters.items():
        notes.append(f"{name} indicator appeared in {count} question segment(s)")
    return notes


def _interview_summary(final_label: str, completed_questions: int, aggregate_notes: list[str]) -> str:
    if completed_questions == 0:
        return "No completed question segments are available for interview-level observation."
    if final_label == "Stable Focus":
        summary = "Across the completed questions, the visual interview condition was consistently stable."
    if final_label == "Mostly Focused":
        summary = "Across the completed questions, the candidate was generally visible with mostly stable interview conditions."
    if final_label == "Light Review Needed":
        summary = "Across the completed questions, most segments were usable, with some visual indicators that may benefit from light recruiter review."
    if final_label == "Review Recommended":
        summary = "Across the completed questions, at least one segment contains visual conditions that should be reviewed with additional context."
    if final_label == "Unstable Interview Condition":
        summary = "Across the completed questions, at least one segment had limited face or person visibility, so recruiter review is required."
    if final_label == "Possible Device Usage Indicator":
        summary = "Across the completed questions, at least one segment included a possible device or downward-gaze indicator for recruiter review."
    if final_label == "External Assistance Indicator":
        summary = "Across the completed questions, at least one segment included a multiple-person indicator for recruiter review."
    if final_label not in LABEL_PRIORITY:
        summary = f"{completed_questions} completed question segment(s) are available for recruiter review."
    if aggregate_notes:
        summary += " Noted indicators: " + "; ".join(aggregate_notes) + "."
    return summary


def _interview_recommendation(final_label: str, completed_questions: int) -> str:
    if completed_questions == 0:
        return "Complete at least one question segment before using the interview-level observation."
    if final_label in {"Stable Focus", "Mostly Focused"}:
        return "Recruiter may continue normal review and optionally replay flagged segments."
    return "Recruiter should review flagged segments with interview notes before interpreting the observation."
