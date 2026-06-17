from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.processing.report_generator import build_export_report, save_export_report
from src.utils.config import CANDIDATE_CONTEXT, GLOBAL_DISCLAIMER


class ReportExportTest(unittest.TestCase):
    def test_build_export_report_schema(self) -> None:
        question_reports = [
            {
                "question_id": 1,
                "question_text": "Tell us about your experience building web applications using ReactJS.",
                "duration_seconds": 45.2,
                "metrics": {"face_visible_percentage": 92.4},
                "observation_label": "Mostly Focused",
                "short_summary": "The candidate was visually present for most of the answer.",
                "recruiter_recommendation": "The recruiter can proceed with normal review.",
                "human_in_the_loop_disclaimer": "Human recruiter review is required.",
            }
        ]

        report = build_export_report(
            CANDIDATE_CONTEXT,
            ["Tell us about your experience building web applications using ReactJS."],
            question_reports,
            session_id="test-session",
            created_at="2026-06-17T10:00:00+00:00",
        )

        self.assertEqual(report["session_id"], "test-session")
        self.assertEqual(report["prototype_name"], "Karierly Interview Vision")
        self.assertEqual(report["candidate"]["name"], "Dio Febriansyah Lubis")
        self.assertEqual(report["job"]["department"], "Engineering")
        self.assertEqual(report["global_disclaimer"], GLOBAL_DISCLAIMER)
        self.assertFalse(report["privacy"]["raw_video_saved"])
        self.assertFalse(report["privacy"]["biometric_templates_saved"])
        self.assertEqual(report["questions"][0]["observation"]["label"], "Mostly Focused")
        self.assertEqual(report["final_summary"]["overall_label"], "Mostly Focused")

    def test_save_export_report_uses_required_filename(self) -> None:
        report = build_export_report(CANDIDATE_CONTEXT, [], [], session_id="abc-123")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_export_report(report, reports_dir=Path(temp_dir))
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "interview_observation_abc-123.json")
        self.assertEqual(saved["session_id"], "abc-123")


if __name__ == "__main__":
    unittest.main()
