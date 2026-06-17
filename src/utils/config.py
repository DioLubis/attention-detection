from __future__ import annotations

from pathlib import Path


APP_TITLE = "Karierly Interview Vision"
BASE_DIR = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = BASE_DIR / "data" / "questions.json"
REPORTS_DIR = BASE_DIR / "reports"

CANDIDATE_CONTEXT = {
    "candidate_name": "Dio Febriansyah Lubis",
    "position": "Frontend Developer",
    "interview_type": "Technical Interview",
    "recruiter": "Company Recruiter",
}

JOB_CONTEXT = {
    "title": "Frontend Developer",
    "department": "Engineering",
    "interview_type": "Technical Interview",
}

GLOBAL_DISCLAIMER = (
    "This prototype does not determine candidate suitability. It only provides "
    "visual interview observation to support human recruiter review."
)
