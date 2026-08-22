from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from app.core.pipeline import AnalysisPipeline


def _pipeline(workers: int) -> AnalysisPipeline:
    return AnalysisPipeline({"runtime": {"mode": "portrait", "cpu_workers": workers}})


def test_preview_prefetch_uses_background_workers_and_preserves_order(monkeypatch):
    barrier = threading.Barrier(2)
    thread_names: list[str] = []

    def fake_load(path: Path, long_edge: int, fallback: bool):
        thread_names.append(threading.current_thread().name)
        barrier.wait(timeout=2.0)
        value = int(path.stem)
        return np.full((1, 1, 3), value, dtype=np.uint8)

    monkeypatch.setattr("app.core.pipeline.load_preview", fake_load)
    jobs = [("first", Path("1.jpg")), ("second", Path("2.jpg"))]
    result = list(_pipeline(2)._iter_loaded_previews(jobs, 3200, True))

    assert [payload for payload, _rgb, _error in result] == ["first", "second"]
    assert [int(rgb[0, 0, 0]) for _payload, rgb, _error in result] == [1, 2]
    assert all(error is None for _payload, _rgb, error in result)
    assert len(thread_names) == 2
    assert all(name.startswith("photo-preview") for name in thread_names)


def test_preview_prefetch_decode_error_does_not_stop_following_files(monkeypatch):
    def fake_load(path: Path, long_edge: int, fallback: bool):
        if path.name == "bad.jpg":
            raise OSError("broken image")
        return np.zeros((1, 1, 3), dtype=np.uint8)

    monkeypatch.setattr("app.core.pipeline.load_preview", fake_load)
    jobs = [(1, Path("good1.jpg")), (2, Path("bad.jpg")), (3, Path("good2.jpg"))]
    result = list(_pipeline(2)._iter_loaded_previews(jobs, 2048, True))

    assert [payload for payload, _rgb, _error in result] == [1, 2, 3]
    assert result[0][2] is None
    assert isinstance(result[1][2], OSError)
    assert result[2][2] is None


class _FakePhoto:
    def __init__(self, name: str):
        self.path = Path(name)


class _FakeAnalyzer:
    def __init__(self, marker: int, barrier: threading.Barrier, calls: list[tuple[int, str]]):
        self.marker = marker
        self.barrier = barrier
        self.calls = calls

    def analyze(self, photo, rgb):
        self.calls.append((self.marker, threading.current_thread().name))
        self.barrier.wait(timeout=2.0)
        return (self.marker, photo.path.name)


def test_parallel_face_worker_count_is_opt_in_and_clamped():
    disabled = AnalysisPipeline({"runtime": {"mode": "portrait", "parallel_face_analysis": False, "parallel_face_workers": 4}})
    enabled = AnalysisPipeline({"runtime": {"mode": "portrait", "parallel_face_analysis": True, "parallel_face_workers": 9}})
    assert disabled._parallel_face_workers() == 1
    assert enabled._parallel_face_workers() == 4


def test_parallel_analysis_uses_independent_analyzers(monkeypatch):
    barrier = threading.Barrier(2)
    calls: list[tuple[int, str]] = []
    analyzers = [_FakeAnalyzer(1, barrier, calls), _FakeAnalyzer(2, barrier, calls)]

    def fake_load(path: Path, long_edge: int, fallback: bool):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr("app.core.pipeline.load_preview", fake_load)
    pipeline = AnalysisPipeline({"runtime": {"mode": "portrait", "parallel_face_analysis": True, "parallel_face_workers": 2}})
    jobs = [(0, _FakePhoto("a.jpg")), (1, _FakePhoto("b.jpg"))]
    result = list(pipeline._iter_analyzed_frames(jobs, analyzers, 2048, True))

    assert {payload for payload, _assessment, _error in result} == {0, 1}
    assert all(error is None for _payload, _assessment, error in result)
    assert {marker for marker, _thread in calls} == {1, 2}
    assert all(thread.startswith("face-analysis") for _marker, thread in calls)


def test_safe_mode_caps_secondary_group_gpu_workers_without_capping_main():
    pipeline = AnalysisPipeline({"runtime": {
        "mode": "group",
        "parallel_face_analysis": True,
        "parallel_face_workers": 4,
        "gpu_memory_safe_mode": True,
        "group_secondary_face_workers": 2,
    }})
    assert pipeline._parallel_face_workers() == 4
    assert pipeline._secondary_face_workers() == 2


def test_disabling_safe_mode_keeps_all_requested_secondary_workers():
    pipeline = AnalysisPipeline({"runtime": {
        "mode": "group",
        "parallel_face_analysis": True,
        "parallel_face_workers": 4,
        "gpu_memory_safe_mode": False,
        "group_secondary_face_workers": 2,
    }})
    assert pipeline._secondary_face_workers() == 4


def test_safe_gaze_pool_recycle_interval_is_bounded():
    pipeline = AnalysisPipeline({"runtime": {
        "mode": "group",
        "gpu_memory_safe_mode": True,
        "group_gpu_recycle_every": 999,
    }})
    assert pipeline._group_gpu_recycle_every() == 50
