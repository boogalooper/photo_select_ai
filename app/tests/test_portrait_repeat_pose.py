from __future__ import annotations

import json
from pathlib import Path

from app.core.models import FaceAssessment, FrameAssessment, PhotoFile, RunStats
from app.core.pipeline import AnalysisPipeline
from app.core.series import (
    PortraitPoseSignature,
    link_repeated_portrait_series,
    portrait_pose_difference,
)


def _config() -> dict:
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / "config" / "default.json").read_text(encoding="utf-8"))
    cfg["runtime"]["mode"] = "portrait"
    cfg["portrait"]["repeat_pose_mode"] = "best_red_pose_yellow"
    cfg["portrait"]["repeat_pose_min_pose_frames"] = 2
    cfg["portrait"]["repeat_pose_max_yellows"] = 3
    return cfg


def _frame(
    number: int,
    descriptor=(1.0, 0.0),
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    center=(0.5, 0.5),
    size: float = 0.20,
    sharp: float = 0.70,
) -> FrameAssessment:
    from datetime import datetime, timedelta

    photo = PhotoFile(
        Path(f"IMG_{number:04d}.jpg"),
        datetime(2026, 1, 1, 10, 0, 0) + timedelta(seconds=number),
        number,
        ".jpg",
    )
    face = FaceAssessment(
        bbox=(0, 0, 100, 100),
        center=center,
        size_fraction=size,
        eye_open_left=0.92,
        eye_open_right=0.92,
        smile=0.45,
        expression=0.80,
        face_sharpness=sharp,
        eye_sharpness=sharp,
        technical=sharp,
        quality=sharp,
        descriptor=list(descriptor),
        descriptor_source="insightface",
        detection_confidence=0.96,
        head_yaw_deg=yaw,
        head_pitch_deg=pitch,
        head_pose_confidence=0.90,
    )
    return FrameAssessment(photo=photo, faces=[face], technical=sharp)


class _Writer:
    red = "Select"
    yellow = "Second"

    def __init__(self):
        self.writes: list[tuple[str, str]] = []

    def write(self, photo, role):
        self.writes.append((photo.path.name, role))
        return photo.path


def test_repeat_identity_linker_is_conservative_but_links_same_child():
    cfg = _config()
    groups = [
        [_frame(1), _frame(2, (0.99, 0.01))],
        [_frame(3, (0.0, 1.0)), _frame(4, (0.01, 0.99))],
        [_frame(5, (0.995, 0.005)), _frame(6, (1.0, 0.0))],
    ]
    result = link_repeated_portrait_series(groups, cfg)
    assert result.child_ids[0] == result.child_ids[2]
    assert result.child_ids[1] != result.child_ids[0]
    assert result.links == 1


def test_pose_gate_rejects_small_changes_and_accepts_strong_head_turn():
    cfg = _config()
    base = PortraitPoseSignature(0.0, 0.0, 0.50, 0.50, 0.20, 0.90, 3)
    small = PortraitPoseSignature(9.0, 5.0, 0.53, 0.50, 0.21, 0.90, 3)
    strong = PortraitPoseSignature(25.0, 3.0, 0.52, 0.50, 0.20, 0.90, 3)
    assert portrait_pose_difference(base, small, cfg)[0] is False
    assert portrait_pose_difference(base, strong, cfg)[0] is True


def test_framing_shift_alone_is_not_a_new_pose():
    cfg = _config()
    base = PortraitPoseSignature(0.0, 0.0, 0.50, 0.50, 0.20, 0.90, 3)
    moved_only = PortraitPoseSignature(2.0, 1.0, 0.64, 0.50, 0.20, 0.90, 3)
    moved_and_scaled = PortraitPoseSignature(2.0, 1.0, 0.64, 0.50, 0.30, 0.90, 3)
    moved_scaled_and_turned = PortraitPoseSignature(13.0, 1.0, 0.64, 0.50, 0.30, 0.90, 3)
    assert portrait_pose_difference(base, moved_only, cfg)[0] is False
    assert portrait_pose_difference(base, moved_and_scaled, cfg)[0] is False
    assert portrait_pose_difference(base, moved_scaled_and_turned, cfg)[0] is True


def test_one_identity_series_can_produce_global_red_and_distinct_pose_yellow():
    cfg = _config()
    # Same child and same identity series: three frontal frames, then three
    # clearly side-facing frames. The strongest frame is in the second pose.
    frames = [
        _frame(1, yaw=0, sharp=0.72),
        _frame(2, yaw=1, sharp=0.78),
        _frame(3, yaw=-1, sharp=0.74),
        _frame(4, yaw=27, sharp=0.82),
        _frame(5, yaw=26, sharp=0.99),
        _frame(6, yaw=28, sharp=0.84),
    ]
    pipeline = AnalysisPipeline(cfg)
    writer = _Writer()
    stats = RunStats(run_mode="portrait")
    selections = []
    pipeline._run_portrait_mode([frames], writer, stats, selections)

    reds = [s for s in selections if s.label_role == "red"]
    yellows = [s for s in selections if s.label_role == "yellow"]
    assert len(reds) == 1
    assert reds[0].photo.path.name == "IMG_0005.jpg"
    assert len(yellows) == 1
    assert yellows[0].photo.path.name in {"IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"}
    assert stats.portrait_selected == 1
    assert stats.portrait_repeat_yellow_selected == 1


def test_small_pose_variation_does_not_create_yellow():
    cfg = _config()
    frames = [
        _frame(1, yaw=0, sharp=0.75),
        _frame(2, yaw=3, sharp=0.80),
        _frame(3, yaw=7, sharp=0.85),
        _frame(4, yaw=9, sharp=0.90),
    ]
    pipeline = AnalysisPipeline(cfg)
    writer = _Writer()
    stats = RunStats(run_mode="portrait")
    selections = []
    pipeline._run_portrait_mode([frames], writer, stats, selections)

    assert len([s for s in selections if s.label_role == "red"]) == 1
    assert len([s for s in selections if s.label_role == "yellow"]) == 0
