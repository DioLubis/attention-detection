# Karierly Interview Vision

Standalone Streamlit prototype for ATS interview observation. The app shows hardcoded technical interview questions, activates a webcam during each answer, summarizes visual context, and exports a recruiter-friendly JSON report.

This prototype does not decide whether a candidate is accepted or rejected. It does not infer honesty, personality, intelligence, confidence, or hiring suitability.

## Features

- Candidate and job context
- One-question-at-a-time interview flow
- Webcam-based frame analysis
- Face visibility status
- Person count and multiple-person status
- Approximate camera focus / looking-away status
- Approximate looking-left, looking-right, and looking-down status
- Phone, laptop, and book indicators when YOLO is available
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

If `ultralytics` or `mediapipe` cannot run on the machine, the app still attempts basic OpenCV face-based fallback detection. Gaze/head direction is an approximation based on webcam-visible landmarks, not perfect eye tracking.

## Privacy

Raw video and raw images are not stored by default. The exported report contains summarized metrics only.
