from datetime import datetime
from pathlib import Path
import unittest

from app.analysis.scoring import select_portrait
from app.core.models import FaceAssessment, FrameAssessment, PhotoFile


CFG = {
    "analysis": {"eye_open_threshold": 0.55},
    "portrait": {
        "expression_weight": 0.70,
        "smile_weight": 0.35,
        "eye_sharpness_weight": 1.40,
        "face_sharpness_weight": 0.90,
        "eyes_weight": 1.80,
        "technical_weight": 0.35,
        "closed_eye_penalty": 0.45,
        "prefer_open_eyes": True,
    },
}


def frame(name, eyes, sharp, smile=0.2, expression=0.6, reliable=True):
    p = PhotoFile(Path(name), datetime.now(), 1, ".jpg")
    f = FaceAssessment(
        (0, 0, 10, 10), (0.5, 0.5), 0.2,
        eyes, eyes, smile, expression, sharp, sharp, 0.7, 0.7, [],
        landmarks_reliable=reliable,
    )
    return FrameAssessment(p, [f], 0.7)


class ScoreTests(unittest.TestCase):
    def test_open_eyes_gate_beats_sharper_closed(self):
        selected = select_portrait([frame("closed.jpg", 0.1, 1.0), frame("open.jpg", 0.9, 0.55)], CFG)
        self.assertEqual(selected.photo.path.name, "open.jpg")

    def test_smile_weight_is_independent_and_can_be_disabled(self):
        cfg = {
            "analysis": {"eye_open_threshold": 0.55},
            "portrait": {
                "eyes_weight": 0.0,
                "eye_sharpness_weight": 0.0,
                "face_sharpness_weight": 0.0,
                "expression_weight": 0.0,
                "smile_weight": 3.0,
                "technical_weight": 0.0,
                "closed_eye_penalty": 0.0,
                "prefer_open_eyes": False,
            },
        }
        selected = select_portrait([
            frame("neutral.jpg", 0.9, 0.8, smile=0.1),
            frame("smile.jpg", 0.9, 0.8, smile=0.9),
        ], cfg)
        self.assertEqual(selected.photo.path.name, "smile.jpg")

    def test_unreliable_eye_landmarks_are_not_treated_as_closed(self):
        selected = select_portrait([
            frame("unknown.jpg", 0.5, 0.9, reliable=False),
            frame("closed.jpg", 0.1, 1.0, reliable=True),
        ], CFG)
        self.assertEqual(selected.photo.path.name, "unknown.jpg")


if __name__ == "__main__":
    unittest.main()
