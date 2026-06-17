from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import cv2
import streamlit as st

from src.detection.webcam import FrameAnalyzer, WebcamCapture
from src.processing.report_generator import (
    build_export_report,
    build_internal_report,
    build_question_report,
    save_export_report,
)
from src.utils.config import APP_TITLE, CANDIDATE_CONTEXT, QUESTIONS_PATH
from src.utils.helpers import load_json, percentage


def init_state() -> None:
    defaults = {
        "question_index": 0,
        "running": False,
        "current_started_at": None,
        "current_frame_results": None,
        "question_reports": [],
        "last_status": None,
        "capture_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "session_created_at" not in st.session_state:
        st.session_state.session_created_at = datetime.now(timezone.utc).isoformat()


@st.cache_data(show_spinner=False)
def get_questions() -> list[str]:
    return load_json(QUESTIONS_PATH)


@st.cache_resource(show_spinner=False)
def get_analyzer() -> FrameAnalyzer:
    return FrameAnalyzer()


@st.cache_resource(show_spinner=False)
def get_webcam() -> WebcamCapture:
    return WebcamCapture()


def start_question() -> None:
    if st.session_state.question_index >= len(get_questions()):
        return
    st.session_state.running = True
    st.session_state.current_started_at = datetime.now(timezone.utc).isoformat()
    st.session_state.current_frame_results = []
    st.session_state.capture_error = None


def stop_question() -> None:
    if not st.session_state.running or st.session_state.current_frame_results is None:
        return

    question = get_questions()[st.session_state.question_index]
    report = build_question_report(
        question_id=st.session_state.question_index + 1,
        question_text=question,
        started_at=st.session_state.current_started_at,
        stopped_at=datetime.now(timezone.utc).isoformat(),
        frame_results=st.session_state.current_frame_results,
    )
    st.session_state.question_reports.append(report)
    st.session_state.running = False
    st.session_state.current_started_at = None
    st.session_state.current_frame_results = None
    get_webcam().stop()


def next_question() -> None:
    if st.session_state.running:
        stop_question()
    if st.session_state.question_index < len(get_questions()) - 1:
        st.session_state.question_index += 1


def reset_session() -> None:
    get_webcam().stop()
    st.session_state.question_index = 0
    st.session_state.running = False
    st.session_state.current_started_at = None
    st.session_state.current_frame_results = None
    st.session_state.question_reports = []
    st.session_state.last_status = None
    st.session_state.capture_error = None
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_created_at = datetime.now(timezone.utc).isoformat()


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
    for index, question in enumerate(get_questions(), start=1):
        marker = "Current" if index - 1 == st.session_state.question_index else f"Question {index}"
        st.write(f"**{marker}:** {question}")


def render_controls() -> None:
    st.subheader("3. Live Webcam Interview Session")
    current_number = st.session_state.question_index + 1
    questions = get_questions()
    st.info(f"Question {current_number} of {len(questions)}: {questions[st.session_state.question_index]}")

    col1, col2, col3, col4 = st.columns(4)
    col1.button("Start Question", on_click=start_question, disabled=st.session_state.running)
    col2.button("Stop Question", on_click=stop_question, disabled=not st.session_state.running)
    col3.button(
        "Next Question",
        on_click=next_question,
        disabled=st.session_state.running or st.session_state.question_index >= len(questions) - 1,
    )
    col4.button("Reset Session", on_click=reset_session)


def render_live_session() -> None:
    video_slot = st.empty()
    status_slot = st.empty()

    if not st.session_state.running:
        st.caption("Webcam activates after Start Question is selected. Raw video is processed in memory only.")
        return

    webcam = get_webcam()
    analyzer = get_analyzer()
    ok, frame, error = webcam.read()
    if not ok:
        st.session_state.capture_error = error
        st.error(st.session_state.capture_error)
        return

    analyzed_frame, status = analyzer.analyze(frame)
    st.session_state.last_status = status
    st.session_state.current_frame_results.append(status)

    video_slot.image(cv2.cvtColor(analyzed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    with status_slot.container():
        render_detection_panel(status)

    time.sleep(0.12)
    st.rerun()


def render_detection_panel(status: dict | None = None) -> None:
    st.subheader("4. Real-time Detection Panel")
    status = status or st.session_state.last_status or {}

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        status_pill("Face visible" if status.get("face_visible") else "Face not clearly visible", status.get("face_visible", False))
    with col2:
        person_count = status.get("person_count", 0)
        status_pill(f"Person detected ({person_count})" if person_count else "Person not detected", person_count > 0)
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
    with col6:
        context_labels = []
        if status.get("laptop_detected"):
            context_labels.append("laptop")
        if status.get("book_detected"):
            context_labels.append("book")
        label = "Context: " + ", ".join(context_labels) if context_labels else "No laptop/book indicator"
        status_pill(label, bool(context_labels), good_when_active=False)

    gaze_cols = st.columns(4)
    gaze_cols[0].caption(f"Face centered: {'yes' if status.get('face_centered') else 'no'}")
    gaze_cols[1].caption(f"Eyes visible: {'yes' if status.get('eyes_visible') else 'no'}")
    gaze_cols[2].caption(f"Looking left/right: {'left' if status.get('looking_left') else 'right' if status.get('looking_right') else 'no'}")
    gaze_cols[3].caption(f"Looking down: {'yes' if status.get('looking_down') else 'no'}")
    st.caption(
        "Detector mode: "
        f"{status.get('detector_mode', 'unknown')}. "
        "YOLO/MediaPipe memberi hasil terbaik; OpenCV fallback hanya estimasi sederhana."
    )

    if st.session_state.current_frame_results:
        frame_results = st.session_state.current_frame_results
        st.caption(
            "Current question metrics: "
            f"{len(frame_results)} frames, "
            f"face visible {percentage(frame_results, 'face_visible'):.0%}, "
            f"camera focus {percentage(frame_results, 'looking_at_camera'):.0%}, "
            f"phone indicator {percentage(frame_results, 'phone_detected'):.0%}, "
            f"laptop indicator {percentage(frame_results, 'laptop_detected'):.0%}, "
            f"book indicator {percentage(frame_results, 'book_detected'):.0%}."
        )


def render_reports() -> dict:
    st.subheader("5. Interview Observation Sentiment Report")

    if not st.session_state.question_reports:
        st.caption("Question-level summaries will appear after each question is stopped.")
    else:
        for report in st.session_state.question_reports:
            with st.expander(f"Question {report['question_id']} - {report['observation_label']}", expanded=True):
                st.write(report["question_text"])
                st.write(report["short_summary"])
                st.caption(report["recruiter_recommendation"])
                st.json(report["metrics"])

    final_report = build_internal_report(CANDIDATE_CONTEXT, st.session_state.question_reports)
    if len(st.session_state.question_reports) == len(get_questions()):
        st.success(f"Final observation label: {final_report['overall_summary']['observation_label']}")
        st.write(final_report["overall_summary"]["short_summary"])
        st.caption(final_report["overall_summary"]["recruiter_recommendation"])
    else:
        st.info("Final report will be complete after all questions have been answered and stopped.")

    return final_report


def render_export() -> None:
    st.subheader("6. Export Report")
    export_report = build_export_report(
        CANDIDATE_CONTEXT,
        get_questions(),
        st.session_state.question_reports,
        session_id=st.session_state.session_id,
        created_at=st.session_state.session_created_at,
    )
    report_json = json.dumps(export_report, indent=2, ensure_ascii=False)
    st.download_button(
        "Download JSON Report",
        data=report_json,
        file_name=f"interview_observation_{export_report['session_id']}.json",
        mime="application/json",
        disabled=not st.session_state.question_reports,
    )

    if st.button("Save JSON to reports folder", disabled=not st.session_state.question_reports):
        filename = save_export_report(export_report)
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
    render_reports()
    render_export()


if __name__ == "__main__":
    main()
