# Karierly Interview Vision

Standalone Streamlit prototype for ATS interview observation. The app shows hardcoded interview questions, activates the webcam while a candidate answers, summarizes visual context, and exports a recruiter-friendly JSON report.

The prototype is an observation aid only. It does not determine candidate suitability, does not infer honesty or personality, and does not produce an automatic hiring decision.

## Folder Structure

```text
karierly-interview-vision/
  app.py
  requirements.txt
  README.md
  data/
    questions.json
  reports/
    .gitkeep
    sample_interview_observation.json
  src/
    detection/
      object_detector.py
      gaze_estimator.py
      webcam.py
    processing/
      metrics.py
      sentiment.py
      report_generator.py
    utils/
      config.py
      helpers.py
```

## Features

- Mock candidate and job context
- One-question-at-a-time interview flow
- Safe webcam start/stop controls
- YOLO object detection for person, cell phone, laptop, and book when available
- MediaPipe Face Mesh gaze/head approximation when available
- OpenCV face fallback when YOLO or MediaPipe cannot load
- Question-level metrics aggregation
- Neutral HR-friendly observation labels
- JSON report download and local save to `reports/`
- No raw video, raw images, or biometric templates are stored

## Setup From Fresh Clone

```powershell
cd karierly-interview-vision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Usage

1. Review the candidate and job context.
2. Click `Start Question` to activate the webcam for the current question.
3. Click `Stop Question` when the answer is finished.
4. Click `Next Question` and repeat.
5. Review the observation report.
6. Download the JSON report or save it locally to `reports/`.

## Report Output

Generated reports use this filename format:

```text
reports/interview_observation_<session_id>.json
```

The report contains only:

- session metadata
- mock candidate/job data
- interview questions
- summarized visual metrics
- generated observation text
- ethical disclaimers

It does not contain raw video, raw images, screenshots, embeddings, face templates, or biometric templates.

## Detection Notes

YOLO and MediaPipe are used when installed and available. If they fail to load, the app still runs with a simpler OpenCV face fallback. Gaze/head direction is an approximation based on webcam-visible landmarks and should be reviewed by a human recruiter in context.

## Tests

```powershell
python -m unittest test_metrics.py test_sentiment.py test_report_export.py
```

## Ethics

This prototype must not be used to accept or reject candidates automatically. It provides visual interview observation only and requires human-in-the-loop recruiter review.
