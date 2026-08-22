from __future__ import annotations

from app.core.pipeline import AnalysisPipeline


def test_pipeline_progress_never_moves_backwards():
    events: list[tuple[float, str]] = []
    pipe = AnalysisPipeline(
        {"runtime": {"mode": "group"}},
        progress=lambda p, m: events.append((p, m)),
    )
    pipe.progress(10.0, "first")
    pipe.progress(8.0, "late stale event")
    pipe.progress(10.5, "next")
    assert [p for p, _m in events] == [10.0, 10.0, 10.5]
