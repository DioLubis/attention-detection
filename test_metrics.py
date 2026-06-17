from __future__ import annotations

import unittest

from metrics import build_question_metrics_summary


class MetricsAggregationTest(unittest.TestCase):
    def test_aggregates_percentages_and_unstable_frames(self) -> None:
        frames = [
            {
                "timestamp": 10.0,
                "person_count": 1,
                "face_visible": True,
                "looking_at_camera": True,
                "looking_away": False,
                "looking_left": False,
                "looking_right": False,
                "looking_down": False,
                "phone_detected": False,
                "laptop_detected": True,
                "book_detected": False,
                "multiple_persons": False,
            },
            {
                "timestamp": 11.0,
                "person_count": 1,
                "face_visible": True,
                "looking_at_camera": False,
                "looking_away": True,
                "looking_left": True,
                "looking_right": False,
                "looking_down": False,
                "phone_detected": False,
                "laptop_detected": True,
                "book_detected": False,
                "multiple_persons": False,
            },
            {
                "timestamp": 12.0,
                "person_count": 2,
                "face_visible": False,
                "looking_at_camera": False,
                "looking_away": True,
                "looking_left": False,
                "looking_right": False,
                "looking_down": True,
                "phone_detected": True,
                "laptop_detected": False,
                "book_detected": True,
                "multiple_persons": True,
            },
        ]

        summary = build_question_metrics_summary(1, "Test question", frames)

        self.assertEqual(summary["duration_seconds"], 2.0)
        self.assertEqual(summary["total_frames"], 3)
        self.assertEqual(summary["metrics"]["face_visible_percentage"], 66.67)
        self.assertEqual(summary["metrics"]["person_visible_percentage"], 100.0)
        self.assertEqual(summary["metrics"]["looking_at_camera_percentage"], 33.33)
        self.assertEqual(summary["metrics"]["laptop_detected_percentage"], 66.67)
        self.assertEqual(summary["metrics"]["unstable_frame_percentage"], 33.33)

    def test_empty_frame_results_are_safe(self) -> None:
        summary = build_question_metrics_summary(2, "Empty question", [])

        self.assertEqual(summary["duration_seconds"], 0.0)
        self.assertEqual(summary["total_frames"], 0)
        self.assertTrue(all(value == 0.0 for value in summary["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
