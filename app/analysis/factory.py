from __future__ import annotations

from app.paths import ROOT

INSIGHTFACE_ROOT = ROOT / "models" / "insightface"


def create_face_analyzer(config: dict, message=None):
    from .face_insightface import InsightFaceAnalyzer

    analyzer = InsightFaceAnalyzer(INSIGHTFACE_ROOT, config, message=message)
    return analyzer, analyzer.backend_name
