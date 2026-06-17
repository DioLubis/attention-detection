# Karierly Interview Vision

Standalone Streamlit prototype for ATS interview observation. The app shows hardcoded technical interview questions, activates a webcam during each answer, summarizes visual context, and exports a recruiter-friendly JSON report.

This prototype does not decide whether a candidate is accepted or rejected. It does not infer honesty, personality, intelligence, confidence, or hiring suitability.

## Features

- Candidate and job context
- One-question-at-a-time interview flow
- Webcam-based frame analysis
- Face visibility status
- Person presence status
- Approximate camera focus / looking-away status
- Phone indicator and multiple-person indicator when YOLO is available
- Question-level observation summaries
- Final interview observation report
- JSON export

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If `ultralytics` or `mediapipe` cannot run on the machine, the app still attempts basic OpenCV face-based fallback detection.

## Privacy

Raw video and raw images are not stored by default. The exported report contains summarized metrics only.
