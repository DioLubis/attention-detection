from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import streamlit as st

from config import CANDIDATE_CONTEXT, QUESTIONS
from reporting import build_final_report, build_question_summary
from vision import FrameAnalyzer


APP_TITLE = "Karierly Interview Vision"
EXPORT_DIR = Path("exports")


def init_state() -> None:
    defaults = {
        "question_index": 0,
        "running": False,
        "current_started_at": None,
        "current_metrics": None,
        "question_reports": [],
        "last_status": None,
        "capture_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_analyzer() -> FrameAnalyzer:
    return FrameAnalyzer()


@st.cache_resource(show_spinner=False)
def get_camera() -> cv2.VideoCapture:
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return camera


def new_metrics() -> dict:
    return {
        "frames": 0,
        "face_visible": 0,
        "person_detected": 0,
        "looking_at_camera": 0,
        "looking_away": 0,
        "phone_detected": 0,
        "multiple_persons": 0,
    }


def accumulate(metrics: dict, status: dict) -> None:
    metrics["frames"] += 1
    for key in (
        "face_visible",
        "person_detected",
        "looking_at_camera",
        "looking_away",
        "phone_detected",
        "multiple_persons",
    ):
        if status.get(key):
            metrics[key] += 1


def start_question() -> None:
    if st.session_state.question_index >= len(QUESTIONS):
        return
    st.session_state.running = True
    st.session_state.current_started_at = datetime.now(timezone.utc).isoformat()
    st.session_state.current_metrics = new_metrics()
    st.session_state.capture_error = None


def stop_question() -> None:
    if not st.session_state.running or st.session_state.current_metrics is None:
        return

    question = QUESTIONS[st.session_state.question_index]
    report = build_question_summary(
        question_id=st.session_state.question_index + 1,
        question_text=question,
        started_at=st.session_state.current_started_at,
        stopped_at=datetime.now(timezone.utc).isoformat(),
        metrics=st.session_state.current_metrics,
    )
    st.session_state.question_reports.append(report)
    st.session_state.running = False
    st.session_state.current_started_at = None
    st.session_state.current_metrics = None


def next_question() -> None:
    if st.session_state.running:
        stop_question()
    if st.session_state.question_index < len(QUESTIONS) - 1:
        st.session_state.question_index += 1


def reset_session() -> None:
    st.session_state.question_index = 0
    st.session_state.running = False
    st.session_state.current_started_at = None
    st.session_state.current_metrics = None
    st.session_state.question_reports = []
    st.session_state.last_status = None
    st.session_state.capture_error = None


def ratio(metrics: dict, key: str) -> float:
    frames = max(metrics.get("frames", 0), 1)
    return metrics.get(key, 0) / frames


def status_pill(label: str, active: bool, good_when_active: bool = True) -> None:
    if active == good_when_active:
        st.success(label)
    else:
        st.warning(label)


def render_context() -> None:
    st.subheader("1. Candidate & Job Context")
    cols = st.columns(4)
    cols[0].metric("Candidate", CANDIDATE_CONTEXT["candidate_name"])
    cols[1].metric("Position", CANDIDATE_CONTEXT["position"])
    cols[2].metric("Interview", CANDIDATE_CONTEXT["interview_type"])
    cols[3].metric("Recruiter", CANDIDATE_CONTEXT["recruiter"])


def render_questions() -> None:
    st.subheader("2. Hardcoded Interview Questions")
    for index, question in enumerate(QUESTIONS, start=1):
        marker = "Current" if index - 1 == st.session_state.question_index else f"Question {index}"
        st.write(f"**{marker}:** {question}")


def render_controls() -> None:
    st.subheader("3. Live Webcam Interview Session")
    current_number = st.session_state.question_index + 1
    st.info(f"Question {current_number} of {len(QUESTIONS)}: {QUESTIONS[st.session_state.question_index]}")

    col1, col2, col3, col4 = st.columns(4)
    col1.button("Start Question", on_click=start_question, disabled=st.session_state.running)
    col2.button("Stop Question", on_click=stop_question, disabled=not st.session_state.running)
    col3.button(
        "Next Question",
        on_click=next_question,
        disabled=st.session_state.running or st.session_state.question_index >= len(QUESTIONS) - 1,
    )
    col4.button("Reset Session", on_click=reset_session)


def render_live_session() -> None:
    video_slot = st.empty()
    status_slot = st.empty()

    if not st.session_state.running:
        st.caption("Webcam activates after Start Question is selected. Raw video is processed in memory only.")
        return

    camera = get_camera()
    analyzer = get_analyzer()
    ok, frame = camera.read()
    if not ok:
        st.session_state.capture_error = "Unable to read from webcam. Check camera permission or another app using the camera."
        st.error(st.session_state.capture_error)
        return

    analyzed_frame, status = analyzer.analyze(frame)
    st.session_state.last_status = status
    accumulate(st.session_state.current_metrics, status)

    video_slot.image(cv2.cvtColor(analyzed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    with status_slot.container():
        render_detection_panel(status)

    time.sleep(0.12)
    st.rerun()


def render_detection_panel(status: dict | None = None) -> None:
    st.subheader("4. Real-time Detection Panel")
    status = status or st.session_state.last_status or {}

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        status_pill("Face visible" if status.get("face_visible") else "Face not clearly visible", status.get("face_visible", False))
    with col2:
        status_pill("Person detected" if status.get("person_detected") else "Person not detected", status.get("person_detected", False))
    with col3:
        status_pill(
            "Looking at camera" if status.get("looking_at_camera") else "Looking away",
            status.get("looking_at_camera", False),
        )
    with col4:
        status_pill(
            "Phone indicator present" if status.get("phone_detected") else "No phone indicator",
            status.get("phone_detected", False),
            good_when_active=False,
        )
    with col5:
        status_pill(
            "Multiple persons indicator" if status.get("multiple_persons") else "Single-person condition",
            status.get("multiple_persons", False),
            good_when_active=False,
        )

    if st.session_state.current_metrics:
        metrics = st.session_state.current_metrics
        st.caption(
            "Current question metrics: "
            f"{metrics['frames']} frames, "
            f"face visible {ratio(metrics, 'face_visible'):.0%}, "
            f"camera focus {ratio(metrics, 'looking_at_camera'):.0%}, "
            f"phone indicator {ratio(metrics, 'phone_detected'):.0%}."
        )


def render_reports() -> dict:
    st.subheader("5. Interview Observation Sentiment Report")

    if not st.session_state.question_reports:
        st.caption("Question-level summaries will appear after each question is stopped.")
    else:
        for report in st.session_state.question_reports:
            with st.expander(f"Question {report['question_id']} - {report['observation_label']}", expanded=True):
                st.write(report["question_text"])
                st.write(report["summary"])
                st.json(report["metrics"])

    final_report = build_final_report(CANDIDATE_CONTEXT, st.session_state.question_reports)
    if len(st.session_state.question_reports) == len(QUESTIONS):
        st.success(f"Final observation label: {final_report['overall_summary']['observation_label']}")
        st.write(final_report["overall_summary"]["summary"])
    else:
        st.info("Final report will be complete after all questions have been answered and stopped.")

    return final_report


def render_export(final_report: dict) -> None:
    st.subheader("6. Export Report")
    report_json = json.dumps(final_report, indent=2)
    st.download_button(
        "Download JSON Report",
        data=report_json,
        file_name="karierly_interview_vision_report.json",
        mime="application/json",
        disabled=not st.session_state.question_reports,
    )

    if st.button("Save JSON to exports folder", disabled=not st.session_state.question_reports):
        EXPORT_DIR.mkdir(exist_ok=True)
        filename = EXPORT_DIR / f"karierly_interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename.write_text(report_json, encoding="utf-8")
        st.success(f"Saved report to {filename}")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption(
        "ATS interview observation prototype. This tool summarizes visible interview conditions only and requires human recruiter review."
    )
    st.warning(
        "Human-in-the-loop notice: this prototype does not infer honesty, personality, intelligence, confidence, or hiring suitability. "
        "It must not be used to accept or reject candidates automatically."
    )

    render_context()
    render_questions()
    render_controls()
    render_live_session()
    final_report = render_reports()
    render_export(final_report)


if __name__ == "__main__":
    main()
