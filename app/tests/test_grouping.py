from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from app.analysis.grouping import (
    build_person_tracks,
    has_group_face_count,
    looks_like_group_series,
    merge_adjacent_group_blocks,
    select_group_series,
    split_group_candidate_by_identity,
)
from app.core.models import FaceAssessment, FrameAssessment, PhotoFile


def face(idx: int, q: float, x: float) -> FaceAssessment:
    vec = [0.0] * 8
    vec[idx % 8] = 1.0
    return FaceAssessment(
        bbox=(0, 0, 100, 100), center=(x, 0.5), size_fraction=0.02,
        eye_open_left=q, eye_open_right=q, smile=q, expression=q,
        face_sharpness=q, eye_sharpness=q, technical=q, quality=q,
        descriptor=vec, landmarks_reliable=True, detection_confidence=0.95,
        descriptor_source="insightface",
    )


def frame(n: int, qualities: list[float], start: datetime) -> FrameAssessment:
    faces = [face(i, q, 0.1 + 0.18 * i) for i, q in enumerate(qualities)]
    photo = PhotoFile(Path(f"IMG_{n:04d}.jpg"), start + timedelta(seconds=n), n, ".jpg")
    return FrameAssessment(photo=photo, faces=faces, technical=0.8)


def group_frame(n: int, identity_offset: int, start: datetime, people: int = 4) -> FrameAssessment:
    faces: list[FaceAssessment] = []
    for i in range(people):
        vec = [0.0] * 256
        vec[identity_offset + i] = 1.0
        f = face(i, 0.85, 0.1 + 0.18 * i)
        f.descriptor = vec
        faces.append(f)
    photo = PhotoFile(Path(f"G_{n:04d}.jpg"), start + timedelta(seconds=n), n, ".jpg")
    return FrameAssessment(photo=photo, faces=faces, technical=0.8)


CFG = {
    "analysis": {"eye_open_threshold": 0.52, "face_min_fraction": 0.0005},
    "group": {
        "min_people": 4, "min_track_presence": 2, "track_det_thresh": 0.30,
        "min_track_face_fraction": 0.00035, "match_threshold": 0.32,
        "track_max_frame_gap": 2, "good_face_threshold": 0.62,
        "worst_percentile": 0.25, "find_headswap_candidates": True,
        "max_extra_candidates": 3, "improvement_margin": 0.12,
        "cross_block_merge_seconds": 45.0, "cross_block_min_identity_overlap": 0.75,
        "cross_block_face_distance": 0.24, "cross_block_face_count_ratio": 0.70,
        "split_face_distance": 0.24, "split_keep_identity_overlap": 0.35,
        "split_new_group_overlap": 0.35, "split_max_old_overlap": 0.15,
        "split_reference_frames": 2,
    },
}


def test_group_main_is_red_and_extras_are_yellow():
    start = datetime(2026, 1, 1, 10, 0, 0)
    frames = [
        frame(0, [0.90, 0.90, 0.40, 0.40], start),
        frame(1, [0.82, 0.82, 0.92, 0.35], start),
        frame(2, [0.80, 0.80, 0.45, 0.95], start),
    ]
    result, _diag = select_group_series(frames, CFG)
    assert result.main is not None
    assert result.main.label_role == "red"
    assert result.extras
    assert all(extra.label_role == "yellow" for extra in result.extras)
    assert len(result.extras) <= 3


def test_same_group_across_short_pause_merges_to_one_series():
    start = datetime(2026, 1, 1, 10, 0, 0)
    left = [frame(0, [0.9, 0.9, 0.9, 0.9], start), frame(1, [0.9, 0.9, 0.9, 0.9], start)]
    # Same identities, later capture times.
    right = [frame(20, [0.9, 0.9, 0.9, 0.9], start), frame(21, [0.9, 0.9, 0.9, 0.9], start)]
    merged, count = merge_adjacent_group_blocks([left, right], CFG)
    assert count == 1
    assert len(merged) == 1
    assert len(merged[0]) == 4


def test_group_selection_reports_progress_through_postprocessing():
    start = datetime(2026, 1, 1, 10, 0, 0)
    frames = [
        frame(0, [0.90, 0.90, 0.40, 0.40], start),
        frame(1, [0.82, 0.82, 0.92, 0.35], start),
        frame(2, [0.80, 0.80, 0.45, 0.95], start),
    ]
    events: list[tuple[float, str]] = []
    result, _diag = select_group_series(frames, CFG, progress=lambda p, m: events.append((p, m)))
    assert result.main is not None
    assert events
    assert events[-1][0] == 1.0
    assert any("сопоставление детей" in message for _p, message in events)
    assert any("оценка кадров" in message for _p, message in events)
    assert all(0.0 <= p <= 1.0 for p, _message in events)


def test_closed_eyes_candidate_must_be_same_child_with_open_eyes():
    start = datetime(2026, 1, 1, 11, 0, 0)
    f0 = frame(0, [0.95, 0.95, 0.95, 0.95], start)
    f1 = frame(1, [0.46, 0.46, 0.46, 0.46], start)
    f2 = frame(2, [0.44, 0.44, 0.44, 0.44], start)
    child = 2
    f0.faces[child].eye_open_left = 0.30
    f0.faces[child].eye_open_right = 0.32
    f0.faces[child].eye_sharpness = 0.90
    f1.faces[child].eye_open_left = 0.88
    f1.faces[child].eye_open_right = 0.90
    f1.faces[child].eye_sharpness = 0.72

    cfg = {**CFG, "group": {**CFG["group"],
        "eye_problem_threshold": 0.58,
        "eye_candidate_threshold": 0.64,
        "eye_improvement_margin": 0.08,
        "headswap_min_eye_sharpness": 0.35,
        "prioritize_eye_candidates": True,
    }}
    result, diag = select_group_series([f0, f1, f2], cfg)
    assert result.main is not None
    assert result.main.photo.path.name == "IMG_0000.jpg"
    assert diag.eye_problems >= 1
    assert result.extras
    assert result.extras[0].photo.path.name == "IMG_0001.jpg"
    assert result.extras[0].label_role == "yellow"
    assert "eyes" in result.extras[0].reason


def test_closed_eye_problem_rejects_same_child_if_eyes_still_closed():
    start = datetime(2026, 1, 1, 12, 0, 0)
    f0 = frame(0, [0.95, 0.95, 0.95, 0.95], start)
    f1 = frame(1, [0.80, 0.80, 0.80, 0.80], start)
    child = 1
    f0.faces[child].eye_open_left = 0.25
    f0.faces[child].eye_open_right = 0.30
    f1.faces[child].eye_open_left = 0.48
    f1.faces[child].eye_open_right = 0.50

    cfg = {**CFG, "group": {**CFG["group"],
        "eye_problem_threshold": 0.58,
        "eye_candidate_threshold": 0.64,
        "eye_improvement_margin": 0.08,
        "headswap_min_eye_sharpness": 0.35,
        "prioritize_eye_candidates": True,
        "min_extra_candidates": 0,
    }}
    result, diag = select_group_series([f0, f1], cfg)
    assert result.main is not None
    assert diag.eye_problems >= 1
    assert not result.extras
    assert diag.unresolved_problems >= 1


def test_group_forces_backup_yellow_when_strict_candidates_are_absent():
    start = datetime(2026, 1, 1, 13, 0, 0)
    frames = [
        frame(0, [0.90, 0.90, 0.90, 0.90], start),
        frame(1, [0.87, 0.88, 0.86, 0.89], start),
        frame(2, [0.82, 0.84, 0.83, 0.85], start),
    ]
    cfg = {**CFG, "group": {**CFG["group"],
        "min_extra_candidates": 1,
        "max_extra_candidates": 3,
        "backup_min_score_ratio": 0.80,
        "backup_person_improvement_margin": 0.05,
    }}
    result, diag = select_group_series(frames, cfg)
    assert result.main is not None
    assert len(result.extras) >= 1
    assert result.extras[0].label_role == "yellow"
    assert "backup" in result.extras[0].reason
    assert diag.backup_extras >= 1


def test_group_main_prioritizes_more_open_eyes_before_overall_quality():
    start = datetime(2026, 1, 1, 14, 0, 0)
    high_quality_closed = frame(0, [0.96, 0.96, 0.96, 0.96], start)
    slightly_lower_open = frame(1, [0.82, 0.82, 0.82, 0.82], start)
    # Two children blink on the otherwise very high-quality first frame.
    for child in (0, 1):
        high_quality_closed.faces[child].eye_open_left = 0.25
        high_quality_closed.faces[child].eye_open_right = 0.28
    cfg = {**CFG, "group": {**CFG["group"],
        "prioritize_open_eyes_main": True,
        "eye_problem_threshold": 0.60,
        "min_extra_candidates": 0,
    }}
    result, _diag = select_group_series([high_quality_closed, slightly_lower_open], cfg)
    assert result.main is not None
    assert result.main.photo.path.name == "IMG_0001.jpg"


def test_back_to_back_different_groups_split_inside_one_temporal_block():
    start = datetime(2026, 1, 1, 15, 0, 0)
    frames = [
        group_frame(0, 0, start), group_frame(1, 0, start), group_frame(2, 0, start),
        group_frame(3, 8, start), group_frame(4, 8, start), group_frame(5, 8, start),
    ]
    groups, splits = split_group_candidate_by_identity(frames, CFG)
    assert splits == 1
    assert len(groups) == 2
    assert [len(g) for g in groups] == [3, 3]


def test_one_bad_identity_frame_does_not_split_same_group():
    start = datetime(2026, 1, 1, 16, 0, 0)
    frames = [
        group_frame(0, 0, start), group_frame(1, 0, start),
        group_frame(2, 8, start),  # transient bad embeddings/detection
        group_frame(3, 0, start), group_frame(4, 0, start),
    ]
    groups, splits = split_group_candidate_by_identity(frames, CFG)
    assert splits == 0
    assert len(groups) == 1


def test_different_groups_with_same_positions_never_cross_block_merge():
    start = datetime(2026, 1, 1, 17, 0, 0)
    left = [group_frame(0, 0, start), group_frame(1, 0, start)]
    right = [group_frame(10, 8, start), group_frame(11, 8, start)]
    merged, count = merge_adjacent_group_blocks([left, right], CFG)
    assert count == 0
    assert len(merged) == 2


def test_back_to_back_group_count_is_not_hardcoded():
    start = datetime(2026, 1, 1, 18, 0, 0)
    for expected_groups in (3, 17, 44, 61):
        frames: list[FrameAssessment] = []
        n = 0
        for group_idx in range(expected_groups):
            identity_offset = group_idx * 4
            for _take in range(3):
                frames.append(group_frame(n, identity_offset, start))
                n += 1
        groups, splits = split_group_candidate_by_identity(frames, CFG)
        assert splits == expected_groups - 1
        assert len(groups) == expected_groups
        assert all(len(group) == 3 for group in groups)


def test_fragmented_child_track_is_merged_back_into_one_person():
    start = datetime(2026, 1, 1, 19, 0, 0)
    frames = [frame(i, [0.90, 0.90, 0.90, 0.90], start) for i in range(7)]
    # Child #2 disappears for three takes, longer than track_max_frame_gap=2.
    # The old implementation created a second confirmed person track here.
    for idx in (2, 3, 4):
        del frames[idx].faces[2]
    tracks, diag = build_person_tracks(frames, CFG)
    assert len(tracks) == 4
    assert diag.confirmed_tracks == 4
    assert diag.tracks_built > diag.confirmed_tracks


def test_single_frame_temporal_fragment_can_be_merged_before_min_frames_check():
    start = datetime(2026, 1, 1, 20, 0, 0)
    left = [group_frame(0, 0, start)]
    right = [group_frame(10, 0, start)]
    assert has_group_face_count(left, CFG)
    assert not looks_like_group_series(left, CFG)
    merged, count = merge_adjacent_group_blocks([left, right], CFG)
    assert count == 1
    assert len(merged) == 1
    assert len(merged[0]) == 2
    assert looks_like_group_series(merged[0], CFG)


def test_adaptive_eye_threshold_handles_consistently_narrow_open_eyes():
    start = datetime(2026, 1, 1, 21, 0, 0)
    frames = [frame(i, [0.90, 0.90, 0.90, 0.90], start) for i in range(3)]
    child = 1
    for f, eye in zip(frames, (0.46, 0.49, 0.51)):
        f.faces[child].eye_open_left = eye
        f.faces[child].eye_open_right = eye
    cfg = {**CFG, "group": {**CFG["group"],
        "eye_problem_threshold": 0.62,
        "eye_candidate_threshold": 0.68,
        "eye_adaptive_min_samples": 3,
        "eye_adaptive_min_reference": 0.40,
        "eye_adaptive_problem_factor": 0.76,
        "eye_adaptive_candidate_factor": 0.92,
        "eye_adaptive_floor": 0.32,
        "min_extra_candidates": 0,
    }}
    result, diag = select_group_series(frames, cfg)
    assert result.main is not None
    assert diag.eye_problems == 0


def test_group_fragments_with_scrambled_exif_are_reassembled_by_filename_sequence():
    """Regression for real IMG_4048/IMG_4049-style fragmentation.

    EXIF-derived order can be scrambled even though the camera counter is
    sequential.  Blocks from the same physical group must still meet each other
    in the merge pass and end up in deterministic filename order.
    """
    start = datetime(2026, 1, 2, 10, 0, 0)
    nums = [4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055]
    by_num = {n: group_frame(n, 0, start, people=4) for n in nums}
    # Deliberately make the metadata timestamps disagree with filename order.
    offsets = {4048: 70, 4049: 5, 4050: 65, 4051: 10, 4052: 60, 4053: 15, 4054: 55, 4055: 20}
    for n, f in by_num.items():
        f.photo.capture_time = start + timedelta(seconds=offsets[n])

    scrambled_blocks = [
        [by_num[4049]], [by_num[4055]], [by_num[4048]], [by_num[4053]],
        [by_num[4054]], [by_num[4050]], [by_num[4052]], [by_num[4051]],
    ]
    merged, count = merge_adjacent_group_blocks(scrambled_blocks, CFG)
    assert count == 7
    assert len(merged) == 1
    assert [f.photo.sequence_number for f in merged[0]] == nums


def test_sequence_recovery_does_not_merge_next_different_group():
    start = datetime(2026, 1, 2, 11, 0, 0)
    same_a = group_frame(4048, 0, start, people=4)
    same_b = group_frame(4049, 0, start, people=4)
    next_a = group_frame(4050, 8, start, people=4)
    next_b = group_frame(4051, 8, start, people=4)
    # Scramble timestamps to force the filename-sequence recovery path.
    same_a.photo.capture_time = start + timedelta(seconds=80)
    same_b.photo.capture_time = start + timedelta(seconds=5)
    next_a.photo.capture_time = start + timedelta(seconds=75)
    next_b.photo.capture_time = start + timedelta(seconds=10)
    merged, count = merge_adjacent_group_blocks([[same_b], [next_b], [same_a], [next_a]], CFG)
    assert count == 2
    assert len(merged) == 2
    assert [f.photo.sequence_number for f in merged[0]] == [4048, 4049]
    assert [f.photo.sequence_number for f in merged[1]] == [4050, 4051]


def test_adjacent_filenames_do_not_override_large_group_pause():
    """Regression for real IMG_6153..IMG_6157 PSD group boundaries."""
    start = datetime(2026, 1, 2, 12, 0, 0)
    first = [group_frame(n, 0, start, people=4) for n in (6153, 6154, 6155, 6156)]
    second = [group_frame(n, 0, start, people=4) for n in (6157, 6158)]

    # Even identical synthetic embeddings must not defeat a clear 140-second
    # boundary merely because the camera counters are adjacent.
    for offset, item in enumerate(first):
        item.photo.capture_time = start + timedelta(seconds=offset)
    for offset, item in enumerate(second):
        item.photo.capture_time = start + timedelta(seconds=143 + offset)

    merged, count = merge_adjacent_group_blocks([first, second], CFG)

    assert count == 0
    assert len(merged) == 2
    assert [f.photo.sequence_number for f in merged[0]] == [6153, 6154, 6155, 6156]
    assert [f.photo.sequence_number for f in merged[1]] == [6157, 6158]


def test_single_frame_group_member_is_promoted_into_roster():
    start = datetime(2026, 1, 2, 10, 0, 0)
    frames = [frame(i, [0.9, 0.9, 0.9, 0.9, 0.9], start) for i in range(3)]
    # Person #5 is found only in one take. Four stable people are enough to
    # establish the group envelope, so the single observation should still be
    # included in the roster and therefore influence RED selection as missing
    # on the other takes.
    del frames[0].faces[4]
    del frames[2].faces[4]
    cfg = {**CFG, "group": {**CFG["group"], "promote_singleton_tracks": True}}
    tracks, diag = build_person_tracks(frames, cfg)
    assert len(tracks) == 5
    assert diag.stable_tracks == 4
    assert diag.promoted_tracks == 1
    assert diag.confirmed_tracks == 5


def test_background_singleton_outside_group_envelope_is_not_promoted():
    start = datetime(2026, 1, 2, 11, 0, 0)
    frames = [frame(i, [0.9, 0.9, 0.9, 0.9], start) for i in range(3)]
    # A one-off face far to the right simulates a passer-by in the background.
    bg = face(7, 0.9, 0.97)
    bg.size_fraction = 0.004
    frames[1].faces.append(bg)
    cfg = {**CFG, "group": {**CFG["group"], "promote_singleton_tracks": True}}
    tracks, diag = build_person_tracks(frames, cfg)
    assert len(tracks) == 4
    assert diag.stable_tracks == 4
    assert diag.promoted_tracks == 0


def test_group_selection_does_not_promote_one_frame_person_in_v043():
    start = datetime(2026, 1, 3, 9, 0, 0)
    frames = [frame(i, [0.9, 0.9, 0.9, 0.9, 0.9], start) for i in range(3)]
    # Person #5 exists in the detector output only once. v0.4.3 must not let a
    # singleton alter the roster; recovery is reserved for repeated high-res
    # evidence from the second pass.
    del frames[0].faces[4]
    del frames[2].faces[4]
    result, diag = select_group_series(frames, CFG)
    assert result.main is not None
    assert result.track_count == 4
    assert diag.stable_tracks == 4
    assert diag.promoted_tracks == 0
    assert diag.highres_rescue_tracks == 0


def test_highres_rescue_adds_repeated_missing_group_member():
    start = datetime(2026, 1, 3, 10, 0, 0)
    # Primary detector misses person #5 on every take (the real 4б failure mode).
    primary = [frame(i, [0.9, 0.9, 0.9, 0.9, 0.9], start) for i in range(3)]
    for f in primary:
        del f.faces[4]

    # Sensitive pass finds all five on repeated takes. Existing four identities
    # must be rejected as duplicates and only the missing slot may be added.
    highres = [frame(i, [0.9, 0.9, 0.9, 0.9, 0.9], start) for i in range(3)]
    cfg = {**CFG, "group": {**CFG["group"], "highres_rescue_enabled": True}}
    result, diag = select_group_series(primary, cfg, rescue_frames=highres)
    assert result.main is not None
    assert result.track_count == 5
    assert diag.stable_tracks == 4
    assert diag.promoted_tracks == 0
    assert diag.highres_rescue_tracks == 1
    assert diag.highres_max_faces_in_frame == 5


def test_highres_rescue_rejects_repeated_background_face_outside_group():
    start = datetime(2026, 1, 3, 11, 0, 0)
    primary = [frame(i, [0.9, 0.9, 0.9, 0.9], start) for i in range(3)]
    highres = [frame(i, [0.9, 0.9, 0.9, 0.9], start) for i in range(3)]
    for idx in (0, 1):
        bg = face(7, 0.9, 0.97)
        bg.size_fraction = 0.004
        highres[idx].faces.append(bg)

    cfg = {**CFG, "group": {**CFG["group"], "highres_rescue_enabled": True}}
    result, diag = select_group_series(primary, cfg, rescue_frames=highres)
    assert result.main is not None
    assert result.track_count == 4
    assert diag.stable_tracks == 4
    assert diag.highres_rescue_tracks == 0


def test_highres_rescue_is_optional_and_ignored_when_disabled():
    start = datetime(2026, 1, 3, 12, 0, 0)
    primary = [frame(i, [0.9, 0.9, 0.9, 0.9, 0.9], start) for i in range(3)]
    for f in primary:
        del f.faces[4]
    highres = [frame(i, [0.9, 0.9, 0.9, 0.9, 0.9], start) for i in range(3)]
    cfg = {**CFG, "group": {**CFG["group"], "highres_rescue_enabled": False}}
    result, diag = select_group_series(primary, cfg, rescue_frames=highres)
    assert result.main is not None
    assert result.track_count == 4
    assert diag.stable_tracks == 4
    assert diag.highres_rescue_tracks == 0


def test_camera_attention_reranks_only_best_shortlist_candidates():
    start = datetime(2026, 1, 4, 10, 0, 0)
    # Ordinary score slightly prefers frame 0. Frame 2 is intentionally weak
    # enough to stay outside the two-frame camera-attention shortlist.
    frames = [
        frame(0, [0.92, 0.92, 0.92, 0.92], start),
        frame(1, [0.90, 0.90, 0.90, 0.90], start),
        frame(2, [0.65, 0.65, 0.65, 0.65], start),
    ]
    cfg = {**CFG, "group": {**CFG["group"],
        "camera_attention_enabled": True,
        "camera_attention_shortlist": 2,
        "camera_attention_away_penalty": 0.45,
        "camera_attention_influence": 0.12,
        "camera_attention_deficit_weight": 0.20,
    }}
    prelim, prelim_diag = select_group_series(frames, cfg)
    assert prelim.main is not None
    assert prelim.main.photo.sequence_number == 0
    assert prelim_diag.camera_attention_shortlist_indices == [0, 1]

    attention_frames = {
        0: frame(0, [0.92, 0.92, 0.92, 0.92], start),
        1: frame(1, [0.90, 0.90, 0.90, 0.90], start),
    }
    for idx, aframe in attention_frames.items():
        for person_idx, aface in enumerate(aframe.faces):
            aface.camera_attention_reliable = True
            aface.camera_attention_confidence = 0.9
            aface.camera_attention_score = 0.82
        if idx == 0:
            aframe.faces[0].camera_attention_score = 0.20
            aframe.faces[1].camera_attention_score = 0.25

    result, diag = select_group_series(frames, cfg, attention_frames=attention_frames)
    assert result.main is not None
    assert result.main.photo.sequence_number == 1
    assert diag.camera_attention_known == 4
    assert diag.camera_attention_away == 0


def test_group_identity_split_reports_smooth_progress():
    start = datetime(2026, 1, 1, 10, 0, 0)
    frames = [group_frame(i, 0 if i < 5 else 20, start) for i in range(10)]
    events: list[tuple[float, str]] = []
    groups, splits = split_group_candidate_by_identity(
        frames, CFG, progress=lambda p, m: events.append((p, m))
    )
    assert splits == 1
    assert len(groups) == 2
    assert len(events) >= 4
    values = [p for p, _m in events]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert any("проверка границ" in m for _p, m in events)


def test_group_block_merge_reports_progress_to_completion():
    start = datetime(2026, 1, 1, 10, 0, 0)
    blocks = [
        [frame(0, [0.9, 0.9, 0.9, 0.9], start)],
        [frame(1, [0.9, 0.9, 0.9, 0.9], start)],
        [group_frame(50, 20, start)],
    ]
    events: list[tuple[float, str]] = []
    _merged, _count = merge_adjacent_group_blocks(
        blocks, CFG, progress=lambda p, m: events.append((p, m))
    )
    assert events
    values = [p for p, _m in events]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert any("соседних блоков" in m for _p, m in events)
