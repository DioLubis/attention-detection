from __future__ import annotations

import unittest

from src.processing.sentiment import build_interview_sentiment, build_question_sentiment, classify_observation


def metrics(**overrides):
    base = {
        "face_visible_percentage": 90.0,
        "person_visible_percentage": 100.0,
        "looking_at_camera_percentage": 85.0,
        "looking_away_percentage": 10.0,
        "looking_down_percentage": 0.0,
        "phone_detected_percentage": 0.0,
        "multiple_persons_percentage": 0.0,
    }
    base.update(overrides)
    return base


class SentimentRulesTest(unittest.TestCase):
    def test_stable_focus_rule(self) -> None:
        self.assertEqual(classify_observation(metrics()), "Stable Focus")

    def test_mostly_focused_rule(self) -> None:
        self.assertEqual(
            classify_observation(metrics(looking_at_camera_percentage=70.0, face_visible_percentage=80.0)),
            "Mostly Focused",
        )

    def test_light_review_rule(self) -> None:
        self.assertEqual(
            classify_observation(metrics(looking_at_camera_percentage=55.0, looking_away_percentage=40.0)),
            "Light Review Needed",
        )

    def test_review_recommended_rule(self) -> None:
        self.assertEqual(
            classify_observation(metrics(looking_at_camera_percentage=40.0, looking_away_percentage=55.0)),
            "Review Recommended",
        )

    def test_unstable_condition_rule(self) -> None:
        self.assertEqual(classify_observation(metrics(face_visible_percentage=55.0)), "Unstable Interview Condition")

    def test_possible_device_rule(self) -> None:
        self.assertEqual(classify_observation(metrics(phone_detected_percentage=18.0)), "Possible Device Usage Indicator")

    def test_external_assistance_rule(self) -> None:
        self.assertEqual(classify_observation(metrics(multiple_persons_percentage=10.0)), "External Assistance Indicator")

    def test_question_sentiment_contains_hr_fields(self) -> None:
        result = build_question_sentiment(metrics(looking_away_percentage=40.0, looking_at_camera_percentage=55.0))

        self.assertEqual(result["observation_label"], "Light Review Needed")
        self.assertIn("short_summary", result)
        self.assertIn("recruiter_recommendation", result)
        self.assertIn("human_in_the_loop_disclaimer", result)

    def test_interview_summary_uses_highest_priority_label(self) -> None:
        result = build_interview_sentiment(
            [
                {"observation_label": "Stable Focus"},
                {"observation_label": "Possible Device Usage Indicator"},
                {"observation_label": "Light Review Needed"},
            ]
        )

        self.assertEqual(result["observation_label"], "Possible Device Usage Indicator")
        self.assertEqual(result["label_counts"]["Stable Focus"], 1)


if __name__ == "__main__":
    unittest.main()
