from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from statistics import median
from typing import Callable

from app.analysis.matching import cosine_similarity
from app.core.models import FaceAssessment, FrameAssessment, Selection


_GROUP_DEFECT_KEY_LEN = 10
_GROUP_PREFERENCE_RANK_LEN = 4


@dataclass(slots=True)
class PersonTrack:
    track_id: int
    observations: list[tuple[int, FaceAssessment]] = field(default_factory=list)

    def add(self, frame_idx: int, face: FaceAssessment) -> None:
        self.observations.append((frame_idx, face))

    @property
    def last(self) -> tuple[int, FaceAssessment] | None:
        return self.observations[-1] if self.observations else None

    @property
    def centroid(self) -> list[float]:
        vecs = [f.descriptor for _i, f in self.observations if f.descriptor]
        if not vecs:
            return []
        width = len(vecs[0])
        if any(len(v) != width for v in vecs):
            return []
        mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(width)]
        norm = sum(v * v for v in mean) ** 0.5
        return [v / norm for v in mean] if norm > 1e-12 else []


@dataclass(slots=True)
class GroupSeriesSelection:
    main: Selection | None
    extras: list[Selection]


@dataclass(slots=True)
class GroupDiagnostics:
    tracks_built: int = 0
    confirmed_tracks: int = 0
    stable_tracks: int = 0
    max_faces_in_frame: int = 0
    eye_problems: int = 0
    missing_problems: int = 0
    sharpness_problems: int = 0
    quality_problems: int = 0
    pose_problems: int = 0
    covered_problems: int = 0
    unresolved_problems: int = 0
    backup_extras: int = 0
    camera_attention_shortlist_indices: list[int] = field(default_factory=list)
    camera_attention_candidate_indices: list[int] = field(default_factory=list)
    camera_attention_known: int = 0
    camera_attention_away: int = 0
    camera_attention_mean: float = 0.0
    portrait_preference_known: int = 0
    portrait_preferred_faces: int = 0
    portrait_preference_mean: float = 0.50
    portrait_preference_worst: float = 0.50


def _face_quality(face: FaceAssessment, eye_threshold: float) -> float:
    eyes = face.eyes_open_score if face.landmarks_reliable else 0.5
    open_bonus = 1.0 if eyes >= eye_threshold else 0.0
    return (
        0.28 * eyes
        + 0.22 * face.eye_sharpness
        + 0.18 * face.face_sharpness
        + 0.14 * face.expression
        + 0.08 * face.smile
        + 0.05 * face.technical
        + 0.05 * open_bonus
    )


def _position_similarity(a: FaceAssessment, b: FaceAssessment) -> float:
    dx = a.center[0] - b.center[0]
    dy = a.center[1] - b.center[1]
    dist = min(1.0, (dx * dx + dy * dy) ** 0.5 / 1.4143)
    return 1.0 - dist


def _size_similarity(a: FaceAssessment, b: FaceAssessment) -> float:
    small = min(a.size_fraction, b.size_fraction)
    big = max(a.size_fraction, b.size_fraction)
    if big <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, small / big))


def _face_track_match(face: FaceAssessment, track: PersonTrack, max_frame_gap: int) -> float:
    last = track.last
    if last is None:
        return -1.0
    _last_frame_idx, prev_face = last
    desc_score = cosine_similarity(face.descriptor, track.centroid or prev_face.descriptor)
    pos_score = _position_similarity(face, prev_face)
    size_score = _size_similarity(face, prev_face)
    # Strong bias to identity, but keep position because group members are
    # usually standing in roughly the same location across sequential takes.
    return 0.72 * max(-1.0, desc_score) + 0.20 * pos_score + 0.08 * size_score


def _track_median_center(track: PersonTrack) -> tuple[float, float]:
    if not track.observations:
        return (0.5, 0.5)
    xs = [face.center[0] for _idx, face in track.observations]
    ys = [face.center[1] for _idx, face in track.observations]
    return (float(median(xs)), float(median(ys)))


def _track_median_size(track: PersonTrack) -> float:
    values = [face.size_fraction for _idx, face in track.observations]
    return float(median(values)) if values else 0.0


def _merge_fragmented_tracks(tracks: list[PersonTrack], config: dict) -> list[PersonTrack]:
    """Merge non-overlapping fragments that are very likely the same child.

    The online tracker intentionally uses a short temporal window so it does not
    jump between neighbouring children.  The downside is fragmentation when one
    face is missed for several takes.  Without this second pass, the same child
    can be counted twice in the group roster and every frame is then penalised
    for an impossible "missing person".
    """
    if len(tracks) <= 1:
        return tracks
    cfg = config.get("group", {})
    max_identity_distance = max(0.05, min(0.60, float(cfg.get("track_merge_face_distance", 0.24))))
    max_position_distance = max(0.01, min(0.30, float(cfg.get("track_merge_position_distance", 0.08))))
    min_size_ratio = max(0.10, min(1.0, float(cfg.get("track_merge_min_size_ratio", 0.45))))

    work = [PersonTrack(track_id=t.track_id, observations=list(t.observations)) for t in tracks]
    changed = True
    while changed:
        changed = False
        best_pair: tuple[int, int] | None = None
        best_rank = -999.0
        for i in range(len(work)):
            a = work[i]
            a_frames = {idx for idx, _face in a.observations}
            a_desc = a.centroid
            if not a_desc:
                continue
            ax, ay = _track_median_center(a)
            asize = _track_median_size(a)
            for j in range(i + 1, len(work)):
                b = work[j]
                # A real child can appear only once in a frame.  Overlapping
                # observations are therefore strong evidence that these are two
                # different people, even if ArcFace happens to be similar.
                if a_frames.intersection(idx for idx, _face in b.observations):
                    continue
                b_desc = b.centroid
                if not b_desc or len(a_desc) != len(b_desc):
                    continue
                sim = cosine_similarity(a_desc, b_desc)
                identity_distance = 1.0 - sim
                if identity_distance > max_identity_distance:
                    continue
                bx, by = _track_median_center(b)
                pos_distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                if pos_distance > max_position_distance:
                    continue
                bsize = _track_median_size(b)
                size_ratio = min(asize, bsize) / max(1e-9, max(asize, bsize))
                if size_ratio < min_size_ratio:
                    continue
                rank = sim - 0.35 * pos_distance + 0.05 * size_ratio
                if rank > best_rank:
                    best_rank = rank
                    best_pair = (i, j)
        if best_pair is not None:
            i, j = best_pair
            work[i].observations.extend(work[j].observations)
            work[i].observations.sort(key=lambda item: item[0])
            del work[j]
            changed = True

    # Re-number after merging so diagnostics and matrices stay compact and
    # deterministic regardless of which temporal fragment was created first.
    work.sort(key=lambda t: (_track_median_center(t)[1], _track_median_center(t)[0], t.track_id))
    for new_id, track in enumerate(work, start=1):
        track.track_id = new_id
    return work


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _adaptive_eye_limits(tracks: list[PersonTrack], config: dict) -> tuple[dict[int, float], dict[int, float]]:
    """Per-child eye thresholds calibrated from the duplicate sequence.

    A single absolute 106-landmark ratio is a poor classifier across 10-30
    different faces.  For repeated group takes we can do something stronger:
    compare each child mainly with *their own* best-looking eye geometry.  The
    global threshold remains a safety fallback when every observation looks
    closed or there is too little evidence.
    """
    cfg = config.get("group", {})
    global_problem = float(cfg.get("eye_problem_threshold", 0.62))
    global_candidate = float(cfg.get("eye_candidate_threshold", 0.68))
    min_reference = float(cfg.get("eye_adaptive_min_reference", 0.40))
    problem_factor = float(cfg.get("eye_adaptive_problem_factor", 0.76))
    candidate_factor = float(cfg.get("eye_adaptive_candidate_factor", 0.92))
    threshold_floor = float(cfg.get("eye_adaptive_floor", 0.32))
    min_samples = max(2, int(cfg.get("eye_adaptive_min_samples", 3)))

    problem_limits: dict[int, float] = {}
    candidate_limits: dict[int, float] = {}
    for track in tracks:
        values = [
            face.eyes_open_score
            for _idx, face in track.observations
            if face.landmarks_reliable
        ]
        reference = _percentile(values, 0.80) if len(values) >= min_samples else (max(values) if values else 0.0)
        if len(values) >= min_samples and reference >= min_reference:
            problem = min(global_problem, max(threshold_floor, reference * problem_factor))
            candidate = min(global_candidate, max(problem + 0.04, reference * candidate_factor))
        else:
            problem = global_problem
            candidate = global_candidate
        problem_limits[track.track_id] = max(0.0, min(1.0, problem))
        candidate_limits[track.track_id] = max(problem_limits[track.track_id], min(1.0, candidate))
    return problem_limits, candidate_limits




def build_person_tracks(
    frames: list[FrameAssessment],
    config: dict,
    progress: Callable[[float, str], None] | None = None,
    *,
    log_roster: bool = True,
) -> tuple[list[PersonTrack], GroupDiagnostics]:
    cfg = config.get("group", {})
    min_face_fraction = float(cfg.get("min_track_face_fraction", config.get("analysis", {}).get("face_min_fraction", 0.0005)))
    min_det = float(cfg.get("track_det_thresh", 0.30))
    match_thresh = float(cfg.get("match_threshold", 0.32))
    # Convert cosine similarity-ish composite threshold to our score domain.
    score_thresh = 1.0 - match_thresh
    max_gap = int(cfg.get("track_max_frame_gap", 2))

    tracks: list[PersonTrack] = []
    next_id = 1
    total_frames = max(1, len(frames))
    for frame_idx, frame in enumerate(frames):
        if progress:
            progress(0.55 * frame_idx / total_frames, f"сопоставление детей {frame_idx + 1}/{len(frames)}")
        faces = [
            f for f in frame.faces
            if f.descriptor and f.size_fraction >= min_face_fraction and f.detection_confidence >= min_det
        ]
        if not faces:
            continue
        pairs: list[tuple[float, int, int]] = []
        for fi, face in enumerate(faces):
            for ti, track in enumerate(tracks):
                last = track.last
                if last is None:
                    continue
                if frame_idx - last[0] > max_gap:
                    continue
                score = _face_track_match(face, track, max_gap)
                pairs.append((score, fi, ti))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_faces: set[int] = set()
        used_tracks: set[int] = set()
        for score, fi, ti in pairs:
            if score < score_thresh:
                break
            if fi in used_faces or ti in used_tracks:
                continue
            tracks[ti].add(frame_idx, faces[fi])
            used_faces.add(fi)
            used_tracks.add(ti)
        for fi, face in enumerate(faces):
            if fi in used_faces:
                continue
            track = PersonTrack(track_id=next_id)
            next_id += 1
            track.add(frame_idx, face)
            tracks.append(track)
    raw_track_count = len(tracks)
    tracks = _merge_fragmented_tracks(tracks, config)
    min_presence = max(1, int(cfg.get("min_track_presence", 2)))
    min_presence_fraction = max(
        0.0, min(1.0, float(cfg.get("min_track_presence_fraction", 0.25)))
    )
    min_presence = max(min_presence, math.ceil(len(frames) * min_presence_fraction))
    stable = [t for t in tracks if len(t.observations) >= min_presence]
    roster = stable

    # Deterministic top-to-bottom/left-to-right IDs.
    roster.sort(key=lambda t: (_track_median_center(t)[1], _track_median_center(t)[0], t.track_id))
    for new_id, track in enumerate(roster, start=1):
        track.track_id = new_id

    max_faces = 0
    for frame in frames:
        accepted = sum(
            1 for f in frame.faces
            if f.descriptor and f.size_fraction >= min_face_fraction and f.detection_confidence >= min_det
        )
        max_faces = max(max_faces, accepted)

    if progress:
        merged_count = max(0, raw_track_count - len(tracks))
        details = [f"устойчивых={len(stable)}"]
        if merged_count:
            details.append(f"склеено фрагментов={merged_count}")
        progress(0.55, f"людей в составе: {len(roster)} ({'; '.join(details)})")
    if log_roster:
        log = logging.getLogger("photo_select_ai")
        log.info(
            "GROUP ROSTER | people=%d | stable=%d | max_faces_frame=%d | raw_tracks=%d",
            len(roster), len(stable), max_faces, raw_track_count,
        )
    return roster, GroupDiagnostics(
        tracks_built=raw_track_count,
        confirmed_tracks=len(roster),
        stable_tracks=len(stable),
        max_faces_in_frame=max_faces,
    )



def _frame_track_matrix(frames: list[FrameAssessment], tracks: list[PersonTrack], config: dict) -> list[dict[int, float]]:
    eye_threshold = float(config.get("group", {}).get("eye_problem_threshold", 0.62))
    matrix: list[dict[int, float]] = [dict() for _ in frames]
    for track in tracks:
        for frame_idx, face in track.observations:
            if 0 <= frame_idx < len(frames):
                matrix[frame_idx][track.track_id] = _face_quality(face, eye_threshold)
    return matrix


def _portrait_preference_matrix(
    frames: list[FrameAssessment],
    tracks: list[PersonTrack],
    config: dict,
) -> list[dict[int, float]]:
    """Build a child-relative portrait-preference matrix.

    Public facial-beauty regressors can have person-specific calibration bias.
    We therefore never compare one child's absolute score with another child's.
    Every reliable score is centred/scaled against other takes of the same
    PersonTrack. Tiny differences are damped so model noise is not exaggerated.
    """
    matrix: list[dict[int, float]] = [dict() for _ in frames]
    cfg = config.get("group", {})
    if not bool(cfg.get("portrait_preference_enabled", True)):
        return matrix

    min_span = max(0.01, min(0.50, float(cfg.get("portrait_preference_min_relative_span", 0.08))))
    for track in tracks:
        samples = [
            (frame_idx, max(0.0, min(1.0, float(face.portrait_preference_score))))
            for frame_idx, face in track.observations
            if 0 <= frame_idx < len(frames) and face.portrait_preference_reliable
        ]
        if not samples:
            continue
        values = [value for _idx, value in samples]
        if len(values) == 1:
            matrix[samples[0][0]][track.track_id] = 0.50
            continue

        centre = float(median(values))
        observed_span = max(values) - min(values)
        span = max(min_span, observed_span)
        for frame_idx, value in samples:
            relative = 0.50 + (value - centre) / span
            matrix[frame_idx][track.track_id] = max(0.0, min(1.0, relative))
    return matrix


def _portrait_preference_stats(
    frame_idx: int,
    track_ids: list[int],
    portrait_matrix: list[dict[int, float]],
    config: dict,
) -> tuple[int, int, float, float, float]:
    """Return known, preferred count, worst-tail, mean and aggregate score."""
    cfg = config.get("group", {})
    people = max(1, len(track_ids))
    threshold = max(0.50, min(0.95, float(cfg.get("portrait_preference_top_threshold", 0.68))))
    row = portrait_matrix[frame_idx] if 0 <= frame_idx < len(portrait_matrix) else {}
    known_values = [row[tid] for tid in track_ids if tid in row]
    known = len(known_values)
    preferred = sum(1 for value in known_values if value >= threshold)

    # Unknown is neutral, not bad. Missing people are handled by a harder gate.
    effective = [row.get(tid, 0.50) for tid in track_ids]
    mean_score = sum(effective) / people
    worst_fraction = max(0.05, min(1.0, float(cfg.get("portrait_preference_worst_percentile", 0.25))))
    worst_count = max(1, int(round(people * worst_fraction)))
    worst_score = sum(sorted(effective)[:worst_count]) / worst_count
    # `preferred` remains a separate primary ranking key. Keep the smooth
    # aggregate centred at 0.5 for neutral data so it can safely be blended
    # with the legacy frame score without introducing a constant hidden
    # penalty. The lower tail deliberately matters more than the mean: one or
    # two weak faces should not be washed out by several excellent ones.
    aggregate = 0.60 * worst_score + 0.40 * mean_score
    return known, preferred, worst_score, mean_score, aggregate


def _group_preference_pool_usable(
    pool_indices: list[int],
    track_ids: list[int],
    portrait_matrix: list[dict[int, float]],
    config: dict,
) -> bool:
    """Decide FBP availability once for the whole best suitability tier.

    A per-frame yes/no gate creates a mixed-FBP bias: a frame measured for just
    over half the group can automatically outrank a frame measured for just
    under half. Instead, FBP becomes active for the whole tier only when enough
    children have repeated reliable measurements across that tier. Unknown
    values then remain neutral (0.5) for every frame on the same scale.
    """
    cfg = config.get("group", {})
    if not bool(cfg.get("portrait_preference_enabled", True)) or len(pool_indices) <= 1:
        return False
    if not track_ids:
        return False
    min_fraction = max(
        0.0,
        min(1.0, float(cfg.get("portrait_preference_min_known_fraction", 0.50))),
    )
    repeated_tracks = 0
    for tid in track_ids:
        observations = sum(
            1
            for frame_idx in pool_indices
            if 0 <= frame_idx < len(portrait_matrix) and tid in portrait_matrix[frame_idx]
        )
        if observations >= 2:
            repeated_tracks += 1
    return repeated_tracks / max(1, len(track_ids)) >= min_fraction


def _group_frame_score(
    frame_idx: int,
    track_ids: list[int],
    matrix: list[dict[int, float]],
    face_matrix: list[dict[int, FaceAssessment]],
    frames: list[FrameAssessment],
    config: dict,
    eye_problem_limits: dict[int, float] | None = None,
) -> float:
    if not track_ids:
        return -1.0
    cfg = config.get("group", {})
    values = [matrix[frame_idx].get(track_id, 0.0) for track_id in track_ids]
    coverage_good_threshold = float(cfg.get("good_face_threshold", 0.62))
    good_count = sum(1 for v in values if v >= coverage_good_threshold)
    coverage = good_count / max(1, len(track_ids))
    mean_quality = sum(values) / max(1, len(values))
    worst_count = max(1, int(round(len(values) * float(cfg.get("worst_percentile", 0.25)))))
    worst_avg = sum(sorted(values)[:worst_count]) / worst_count
    missing = sum(1 for v in values if v <= 1e-9) / max(1, len(values))

    # Use a per-child eye threshold when repeated takes provide enough evidence.
    # This avoids treating natural eye shape as a blink and makes the result much
    # less sensitive to one global landmark ratio.
    global_eye_threshold = float(cfg.get("eye_problem_threshold", 0.58))
    eye_known = 0
    eye_good = 0
    eye_deficit = 0.0
    for tid in track_ids:
        face = face_matrix[frame_idx].get(tid)
        if face is None or not face.landmarks_reliable:
            continue
        eye_known += 1
        threshold = (eye_problem_limits or {}).get(tid, global_eye_threshold)
        eyes = face.eyes_open_score
        if eyes >= threshold:
            eye_good += 1
        else:
            # Continuous deficit is more stable than a pure open/closed counter:
            # a measurement 0.01 below threshold should not overturn a frame in
            # the same way as an obvious blink.
            eye_deficit += min(1.0, max(0.0, (threshold - eyes) / max(0.12, threshold * 0.45)))
    eye_coverage = eye_good / max(1, eye_known) if eye_known else 0.5
    eye_deficit_fraction = eye_deficit / max(1, len(track_ids))

    technical = frames[frame_idx].technical
    return (
        0.35 * coverage
        + 0.18 * eye_coverage
        + 0.22 * worst_avg
        + 0.13 * mean_quality
        + 0.06 * technical
        - 0.14 * missing
        - 0.20 * eye_deficit_fraction
    )


def _frame_track_faces(frames: list[FrameAssessment], tracks: list[PersonTrack]) -> list[dict[int, FaceAssessment]]:
    matrix: list[dict[int, FaceAssessment]] = [dict() for _ in frames]
    for track in tracks:
        for frame_idx, face in track.observations:
            if 0 <= frame_idx < len(frames):
                matrix[frame_idx][track.track_id] = face
    return matrix


def _eye_risk_stats(
    frame_idx: int,
    track_ids: list[int],
    face_matrix: list[dict[int, FaceAssessment]],
    config: dict,
    eye_problem_limits: dict[int, float],
) -> tuple[int, int, int, float]:
    """Return severe blinks, all below-threshold eyes, known states and deficit."""
    cfg = config.get("group", {})
    severe_margin = max(0.02, min(0.40, float(cfg.get("eye_severe_margin", 0.12))))
    severe = closed = known = 0
    deficit = 0.0
    for tid in track_ids:
        face = face_matrix[frame_idx].get(tid)
        if face is None or not face.landmarks_reliable:
            continue
        known += 1
        threshold = eye_problem_limits.get(tid, float(cfg.get("eye_problem_threshold", 0.62)))
        eyes = face.eyes_open_score
        if eyes < threshold:
            closed += 1
            deficit += min(1.0, max(0.0, (threshold - eyes) / max(0.12, threshold * 0.45)))
            if eyes < threshold - severe_margin:
                severe += 1
    return severe, closed, known, deficit


def _group_pose_is_bad(face: FaceAssessment, config: dict) -> bool:
    """Known excessive head turn is a hard Group-mode defect."""
    cfg = config.get("group", {})
    min_conf = max(0.0, min(1.0, float(cfg.get("head_pose_min_confidence", 0.30))))
    if not face.landmarks_reliable or face.head_pose_confidence < min_conf:
        return False
    max_yaw = max(0.0, float(cfg.get("max_head_yaw_deg", 20.0)))
    max_pitch = max(0.0, float(cfg.get("max_head_pitch_deg", 22.0)))
    return abs(float(face.head_yaw_deg)) > max_yaw or abs(float(face.head_pitch_deg)) > max_pitch


def _group_frame_defect_key(
    frame_idx: int,
    track_ids: list[int],
    face_matrix: list[dict[int, FaceAssessment]],
    frames: list[FrameAssessment],
    config: dict,
    eye_problem_limits: dict[int, float],
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    """Hard Group suitability plus a protected uncertain tier before FBP.

    Missing faces, clearly closed eyes, excessive head turn, definite blur and
    poor technical quality are hard defects. Borderline eyes/sharpness and
    unreliable fine-state measurements are uncertainty instead of immediate
    hard rejection. Thus a clean frame always outranks an uncertain one, but a
    potentially good borderline frame remains available when every take is
    difficult. FBP is evaluated only after this complete suitability tier.
    """
    cfg = config.get("group", {})
    min_eye_sharp = max(0.0, min(1.0, float(cfg.get("eye_sharpness_problem_threshold", 0.42))))
    eye_sharp_margin = max(0.0, min(0.25, float(cfg.get("eye_sharpness_uncertain_margin", 0.05))))
    min_face_sharp = max(0.0, min(1.0, float(cfg.get("face_sharpness_problem_threshold", 0.34))))
    face_sharp_margin = max(0.0, min(0.25, float(cfg.get("face_sharpness_uncertain_margin", 0.04))))
    eye_uncertain_margin = max(0.0, min(0.25, float(cfg.get("eye_uncertain_margin", 0.06))))
    min_face_technical = max(0.0, min(1.0, float(cfg.get("min_face_technical_quality", 0.30))))
    min_frame_technical = max(0.0, min(1.0, float(cfg.get("min_frame_technical_quality", 0.28))))

    missing = hard_closed = pose = hard_blur = poor_quality = 0
    uncertain_eyes = uncertain_blur = unknown = 0
    for tid in track_ids:
        face = face_matrix[frame_idx].get(tid)
        if face is None:
            missing += 1
            continue

        min_pose_conf = max(0.0, min(1.0, float(cfg.get("head_pose_min_confidence", 0.30))))
        if not face.landmarks_reliable:
            unknown += 1
        else:
            if face.head_pose_confidence < min_pose_conf:
                unknown += 1
            eye_limit = eye_problem_limits.get(tid, float(cfg.get("eye_problem_threshold", 0.62)))
            eyes = face.eyes_open_score
            if eyes < eye_limit - eye_uncertain_margin:
                hard_closed += 1
            elif eyes < eye_limit:
                uncertain_eyes += 1

        blur_state = 0  # 0=clean, 1=uncertain, 2=definite blur
        face_sharp = float(face.face_sharpness)
        if face_sharp < min_face_sharp - face_sharp_margin:
            blur_state = 2
        elif face_sharp < min_face_sharp:
            blur_state = 1

        if face.landmarks_reliable:
            eye_sharp = float(face.eye_sharpness)
            if eye_sharp < min_eye_sharp - eye_sharp_margin:
                blur_state = 2
            elif eye_sharp < min_eye_sharp:
                blur_state = max(blur_state, 1)

        if blur_state == 2:
            hard_blur += 1
        elif blur_state == 1:
            uncertain_blur += 1

        if _group_pose_is_bad(face, config):
            pose += 1
        if face.technical < min_face_technical:
            poor_quality += 1

    if frames[frame_idx].technical < min_frame_technical:
        poor_quality += 1

    hard_total = missing + hard_closed + pose + hard_blur + poor_quality
    uncertain_total = uncertain_eyes + uncertain_blur + unknown
    return (
        hard_total,
        missing,
        hard_closed,
        pose,
        hard_blur,
        poor_quality,
        uncertain_total,
        uncertain_eyes,
        uncertain_blur,
        unknown,
    )


def _main_problem_for_track(
    track_id: int,
    best_idx: int,
    face_matrix: list[dict[int, FaceAssessment]],
    quality_matrix: list[dict[int, float]],
    config: dict,
    eye_problem_limits: dict[int, float] | None = None,
) -> str | None:
    cfg = config.get("group", {})
    face = face_matrix[best_idx].get(track_id)
    if face is None:
        return "missing"

    # Eyes are treated as a first-class defect for group/head-swap workflow.
    # A face may have excellent sharpness/exposure and still be unusable because
    # one eye is half closed.  Do not let the average quality score hide this.
    eye_problem_threshold = (eye_problem_limits or {}).get(
        track_id, float(cfg.get("eye_problem_threshold", 0.60))
    )
    if bool(cfg.get("prioritize_eye_candidates", True)) and face.landmarks_reliable:
        if face.eyes_open_score < eye_problem_threshold:
            return "eyes"

    if _group_pose_is_bad(face, config):
        return "pose"

    eye_sharp_problem = float(cfg.get("eye_sharpness_problem_threshold", 0.42))
    face_sharp_problem = float(cfg.get("face_sharpness_problem_threshold", 0.34))
    if (face.landmarks_reliable and face.eye_sharpness < eye_sharp_problem) or face.face_sharpness < face_sharp_problem:
        return "sharpness"

    if face.technical < float(cfg.get("min_face_technical_quality", 0.30)):
        return "quality"
    return None


def _candidate_resolves_problem(
    problem: str,
    track_id: int,
    main_idx: int,
    candidate_idx: int,
    face_matrix: list[dict[int, FaceAssessment]],
    quality_matrix: list[dict[int, float]],
    config: dict,
    eye_candidate_limits: dict[int, float] | None = None,
) -> tuple[bool, float]:
    cfg = config.get("group", {})
    main_face = face_matrix[main_idx].get(track_id)
    cand_face = face_matrix[candidate_idx].get(track_id)
    if cand_face is None:
        return False, 0.0

    cand_q = quality_matrix[candidate_idx].get(track_id, 0.0)
    main_q = quality_matrix[main_idx].get(track_id, 0.0)
    min_candidate_q = float(cfg.get("headswap_candidate_min_quality", 0.52))

    # For head-swap candidates, identity is more important than aggregate
    # quality. Re-check ArcFace directly against the RED face even though the
    # observation already belongs to the same temporal track. This protects
    # against a rare track identity swap in a crowded group.
    if main_face is not None and main_face.descriptor and cand_face.descriptor:
        max_identity_distance = float(cfg.get("headswap_identity_distance", 0.30))
        identity_distance = 1.0 - cosine_similarity(main_face.descriptor, cand_face.descriptor)
        if identity_distance > max_identity_distance:
            return False, 0.0

    if problem == "missing":
        # Missing on RED: any reliably recognised, reasonably good face is
        # valuable for a manual head swap.
        ok = cand_q >= min_candidate_q
        return ok, 2.0 + cand_q if ok else 0.0

    if problem == "eyes":
        if not cand_face.landmarks_reliable:
            return False, 0.0
        eye_target = (eye_candidate_limits or {}).get(
            track_id, float(cfg.get("eye_candidate_threshold", 0.64))
        )
        eye_delta = float(cfg.get("eye_improvement_margin", 0.08))
        main_eyes = main_face.eyes_open_score if main_face and main_face.landmarks_reliable else 0.0
        cand_eyes = cand_face.eyes_open_score
        # Eyes must actually become open, not merely raise the average score.
        # Keep a modest sharpness floor so a sharp closed-eye head is not
        # replaced by an unusably blurry open-eye head.
        min_eye_sharp = float(cfg.get("headswap_min_eye_sharpness", 0.35))
        ok = (
            cand_eyes >= eye_target
            and cand_eyes >= main_eyes + eye_delta
            and cand_face.eye_sharpness >= min_eye_sharp
        )
        return ok, 3.0 + max(0.0, cand_eyes - main_eyes) + 0.25 * cand_face.eye_sharpness if ok else 0.0

    if problem == "pose":
        if _group_pose_is_bad(cand_face, config):
            return False, 0.0
        min_pose_conf = float(cfg.get("head_pose_min_confidence", 0.30))
        if not cand_face.landmarks_reliable or cand_face.head_pose_confidence < min_pose_conf:
            return False, 0.0
        eye_target = (eye_candidate_limits or {}).get(
            track_id, float(cfg.get("eye_candidate_threshold", 0.68))
        )
        if cand_face.eyes_open_score < eye_target:
            return False, 0.0
        if cand_face.eye_sharpness < float(cfg.get("headswap_min_eye_sharpness", 0.35)):
            return False, 0.0
        return True, 2.3 + 0.25 * cand_q

    if problem == "sharpness":
        sharp_delta = float(cfg.get("eye_sharpness_improvement_margin", 0.10))
        main_sharp = main_face.eye_sharpness if main_face else 0.0
        eye_floor = float(cfg.get("headswap_eye_floor", 0.48))
        eyes = cand_face.eyes_open_score if cand_face.landmarks_reliable else 0.5
        ok = cand_face.eye_sharpness >= main_sharp + sharp_delta and eyes >= eye_floor
        return ok, 1.7 + max(0.0, cand_face.eye_sharpness - main_sharp) if ok else 0.0

    quality_delta = float(cfg.get("quality_improvement_margin", 0.10))
    ok = cand_q >= max(min_candidate_q, main_q + quality_delta)
    return ok, 1.0 + max(0.0, cand_q - main_q) if ok else 0.0




def _base_main_rank_data(
    frame_idx: int,
    track_ids: list[int],
    matrix: list[dict[int, float]],
    face_matrix: list[dict[int, FaceAssessment]],
    portrait_matrix: list[dict[int, float]],
    frame_scores: list[float],
    frames: list[FrameAssessment],
    config: dict,
    eye_problem_limits: dict[int, float],
    use_preference: bool,
) -> tuple[float, tuple]:
    """Build RED rank with suitability first and one pool-level FBP mode."""
    severe, _closed_count, known_count, eye_deficit = _eye_risk_stats(
        frame_idx, track_ids, face_matrix, config, eye_problem_limits
    )
    visible = len(matrix[frame_idx])
    people = max(1, len(track_ids))
    missing_count = max(0, len(track_ids) - visible)
    severe_fraction = severe / people
    deficit_fraction = eye_deficit / people
    missing_fraction = missing_count / people
    stable_score = (
        frame_scores[frame_idx]
        - 0.55 * severe_fraction
        - 0.28 * deficit_fraction
        - 0.08 * missing_fraction
    )

    defect_key = _group_frame_defect_key(
        frame_idx, track_ids, face_matrix, frames, config, eye_problem_limits
    )
    pref_known, preferred, pref_worst, pref_mean, pref_aggregate = _portrait_preference_stats(
        frame_idx, track_ids, portrait_matrix, config
    )
    suitability_rank = tuple(-value for value in defect_key)

    if use_preference:
        priority_score = stable_score
        # No per-frame FBP usable bit: every frame in the same suitability tier
        # is compared on the same child-relative scale, with unknowns neutral.
        preference_rank = (preferred, pref_worst, pref_mean, pref_aggregate)
        rank = (
            *suitability_rank,
            *preference_rank,
            stable_score,
            visible,
            known_count,
            frame_scores[frame_idx],
            pref_known,
        )
    else:
        priority_score = stable_score
        rank = (*suitability_rank, stable_score, visible, known_count, frame_scores[frame_idx])
    return priority_score, rank


def _track_expected_center(track: PersonTrack, frame_idx: int) -> tuple[float, float]:
    for idx, face in track.observations:
        if idx == frame_idx:
            return face.center
    return _track_median_center(track)


def _camera_attention_face_matrix(
    tracks: list[PersonTrack],
    attention_frames: dict[int, FrameAssessment],
    config: dict,
) -> dict[int, dict[int, FaceAssessment]]:
    """Match high-resolution attention faces back to the stable group roster."""
    cfg = config.get("group", {})
    min_identity = float(cfg.get("camera_attention_match_identity_similarity", 0.45))
    max_position = float(cfg.get("camera_attention_match_position_distance", 0.055))
    result: dict[int, dict[int, FaceAssessment]] = {}
    for frame_idx, frame in attention_frames.items():
        if frame is None or frame.error or not frame.faces:
            result[frame_idx] = {}
            continue
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(tracks):
            expected = _track_expected_center(track, frame_idx)
            centroid = track.centroid
            for fi, face in enumerate(frame.faces):
                dx = face.center[0] - expected[0]
                dy = face.center[1] - expected[1]
                pos_dist = (dx * dx + dy * dy) ** 0.5
                if pos_dist > max_position:
                    continue
                if centroid and face.descriptor and len(centroid) == len(face.descriptor):
                    identity = cosine_similarity(centroid, face.descriptor)
                    if identity < min_identity:
                        continue
                else:
                    identity = 0.50
                pos_score = max(0.0, 1.0 - pos_dist / max(1e-6, max_position))
                pairs.append((0.78 * identity + 0.22 * pos_score, ti, fi))
        pairs.sort(reverse=True)
        used_tracks: set[int] = set()
        used_faces: set[int] = set()
        matched: dict[int, FaceAssessment] = {}
        for _score, ti, fi in pairs:
            if ti in used_tracks or fi in used_faces:
                continue
            used_tracks.add(ti)
            used_faces.add(fi)
            matched[tracks[ti].track_id] = frame.faces[fi]
        result[frame_idx] = matched
    return result


def _camera_attention_stats(
    frame_idx: int,
    track_ids: list[int],
    attention_matrix: dict[int, dict[int, FaceAssessment]],
    config: dict,
) -> tuple[int, int, float, float]:
    """Return known, clearly-away, mean score and normalized deficit."""
    cfg = config.get("group", {})
    away_threshold = float(cfg.get("camera_attention_away_threshold", 0.42))
    good_threshold = float(cfg.get("camera_attention_good_threshold", 0.62))
    faces = attention_matrix.get(frame_idx, {})
    values: list[float] = []
    away = 0
    deficit = 0.0
    for tid in track_ids:
        face = faces.get(tid)
        if face is None or not face.camera_attention_reliable:
            continue
        score = max(0.0, min(1.0, float(face.camera_attention_score)))
        values.append(score)
        if score < away_threshold:
            away += 1
        if score < good_threshold:
            deficit += min(1.0, (good_threshold - score) / max(0.18, good_threshold * 0.55))
    mean = sum(values) / len(values) if values else 0.50
    return len(values), away, mean, deficit


def _camera_attention_guard(
    frame_idx: int,
    track_ids: list[int],
    attention_matrix: dict[int, dict[int, FaceAssessment]],
    config: dict,
) -> tuple[int, int, int]:
    """Return protected gaze tier, acceptable count and reliable count.

    Tier 0 means that no reliably measured person is clearly looking away and
    enough people are positively measured as looking approximately at the
    camera. Tier 1 keeps a no-away frame available when eye detail is
    insufficient. Further tiers increase with every confirmed away gaze.
    """
    cfg = config.get("group", {})
    people = max(1, len(track_ids))
    away_threshold = float(cfg.get("camera_attention_away_threshold", 0.42))
    acceptable_threshold = float(cfg.get("camera_attention_acceptable_threshold", 0.52))
    min_known_fraction = max(
        0.0, min(1.0, float(cfg.get("camera_attention_min_known_fraction", 0.60)))
    )
    required = max(1, math.ceil(people * min_known_fraction))
    reliable = acceptable = away = 0
    faces = attention_matrix.get(frame_idx, {})
    for tid in track_ids:
        face = faces.get(tid)
        if face is None or not face.camera_attention_reliable:
            continue
        reliable += 1
        score = max(0.0, min(1.0, float(face.camera_attention_score)))
        if score < away_threshold:
            away += 1
        elif score >= acceptable_threshold:
            acceptable += 1

    if away == 0 and acceptable >= required:
        tier = 0
    elif away == 0:
        tier = 1
    else:
        tier = 1 + away
    return tier, acceptable, reliable


def _camera_attention_rank(
    frame_idx: int,
    base_score: float,
    base_rank: tuple,
    track_ids: list[int],
    attention_matrix: dict[int, dict[int, FaceAssessment]],
    config: dict,
    *,
    use_preference: bool,
) -> tuple:
    cfg = config.get("group", {})
    people = max(1, len(track_ids))
    known, away, mean, deficit = _camera_attention_stats(
        frame_idx, track_ids, attention_matrix, config
    )
    gaze_tier, acceptable, _reliable = _camera_attention_guard(
        frame_idx, track_ids, attention_matrix, config
    )
    # Unknown faces are neutral (0.5), never treated as looking away. This
    # prevents tiny/ambiguous eyes from incorrectly deciding the RED frame.
    effective_mean = (mean * known + 0.50 * (people - known)) / people
    deficit_fraction = deficit / people
    influence = float(cfg.get("camera_attention_influence", 0.12))
    deficit_weight = float(cfg.get("camera_attention_deficit_weight", 0.20))
    final_score = (
        base_score
        + influence * (effective_mean - 0.50)
        - deficit_weight * deficit_fraction
    )

    suitability_prefix = base_rank[:_GROUP_DEFECT_KEY_LEN]
    remaining = base_rank[_GROUP_DEFECT_KEY_LEN:]
    # Gaze is protected inside one technical-suitability tier. A confirmed
    # away gaze can no longer be hidden by a tiny FBP advantage. Once the
    # minimum gaze condition is satisfied, FBP remains the primary selector.
    gaze_prefix = (-gaze_tier, -away)
    if use_preference:
        preference_prefix = remaining[:_GROUP_PREFERENCE_RANK_LEN]
        trailing = remaining[_GROUP_PREFERENCE_RANK_LEN:]
        gaze_score = (
            influence * (effective_mean - 0.50)
            - deficit_weight * deficit_fraction
        )
        return (
            *suitability_prefix,
            *gaze_prefix,
            *preference_prefix,
            acceptable,
            known,
            gaze_score,
            effective_mean,
            *trailing,
            final_score,
        )

    return (
        *suitability_prefix,
        *gaze_prefix,
        *remaining,
        acceptable,
        known,
        effective_mean,
        final_score,
    )


def select_group_series(
    frames: list[FrameAssessment],
    config: dict,
    progress: Callable[[float, str], None] | None = None,
    *,
    attention_frames: dict[int, FrameAssessment] | None = None,
    diagnostic_log: bool = True,
) -> tuple[GroupSeriesSelection, GroupDiagnostics]:
    tracks, diag = build_person_tracks(
        frames, config, progress=progress, log_roster=False
    )

    log = logging.getLogger("photo_select_ai")
    if diagnostic_log:
        log.info(
            "GROUP ROSTER | people=%d | stable=%d | max_faces_frame=%d | raw_tracks=%d",
            len(tracks), diag.stable_tracks, diag.max_faces_in_frame, diag.tracks_built,
        )
    cfg = config.get("group", {})
    min_people = max(2, int(cfg.get("min_people", 4)))
    if len(tracks) < min_people:
        return GroupSeriesSelection(main=None, extras=[]), diag

    track_ids = [t.track_id for t in tracks]
    if progress:
        progress(0.62, "оценка лиц каждого ребёнка")
    matrix = _frame_track_matrix(frames, tracks, config)
    face_matrix = _frame_track_faces(frames, tracks)
    portrait_matrix = _portrait_preference_matrix(frames, tracks, config)
    eye_problem_limits, eye_candidate_limits = _adaptive_eye_limits(tracks, config)
    frame_scores: list[float] = []
    total_frames = max(1, len(frames))
    for i in range(len(frames)):
        frame_scores.append(
            _group_frame_score(i, track_ids, matrix, face_matrix, frames, config, eye_problem_limits)
        )
        if progress:
            progress(0.62 + 0.20 * (i + 1) / total_frames, f"оценка кадров {i + 1}/{len(frames)}")
    base_scores: dict[int, float] = {}
    base_ranks: dict[int, tuple] = {}

    # Determine FBP mode once for the entire best suitability tier. This avoids
    # the old mixed-FBP case where crossing a per-frame coverage threshold could
    # outweigh the actual attractiveness result.
    defect_keys = {
        i: _group_frame_defect_key(
            i, track_ids, face_matrix, frames, config, eye_problem_limits
        )
        for i in range(len(frames))
    }
    best_defect_key = min(defect_keys.values())
    best_tier_indices = [i for i, key in defect_keys.items() if key == best_defect_key]
    use_preference = _group_preference_pool_usable(
        best_tier_indices, track_ids, portrait_matrix, config
    )

    for i in range(len(frames)):
        base_scores[i], base_ranks[i] = _base_main_rank_data(
            i,
            track_ids,
            matrix,
            face_matrix,
            portrait_matrix,
            frame_scores,
            frames,
            config,
            eye_problem_limits,
            use_preference,
        )

    base_order = sorted(range(len(frames)), key=lambda i: base_ranks[i], reverse=True)
    shortlist_count = max(1, min(len(frames), int(cfg.get("camera_attention_shortlist", 5))))

    # Gaze may only reorder frames in the best complete suitability tier. Start
    # with the strongest FBP batch and ask the pipeline for another batch only
    # when the measured frames contain no protected tier-0 candidate.
    shortlist_order = [i for i in base_order if defect_keys[i] == best_defect_key]
    diag.camera_attention_candidate_indices = list(shortlist_order)
    requested_count = min(len(shortlist_order), shortlist_count)
    if attention_frames:
        analyzed = {i for i in shortlist_order if i in attention_frames}
        # Never shrink the requested prefix after a later batch produces the
        # first acceptable gaze.  Otherwise that successful frame can fall just
        # beyond the original shortlist and be discarded before final ranking.
        analyzed_positions = [
            position for position, frame_idx in enumerate(shortlist_order)
            if frame_idx in analyzed
        ]
        analyzed_prefix_count = max(analyzed_positions, default=-1) + 1
        requested_count = max(requested_count, analyzed_prefix_count)
        has_guarded_candidate = False
        if analyzed:
            provisional_matrix = _camera_attention_face_matrix(tracks, attention_frames, config)
            has_guarded_candidate = any(
                _camera_attention_guard(i, track_ids, provisional_matrix, config)[0] == 0
                for i in analyzed
            )
        if not has_guarded_candidate:
            requested_count = min(len(shortlist_order), requested_count + shortlist_count)

    diag.camera_attention_shortlist_indices = shortlist_order[:requested_count]
    best_idx = base_order[0]

    attention_matrix: dict[int, dict[int, FaceAssessment]] = {}
    if bool(cfg.get("camera_attention_enabled", False)) and attention_frames:
        attention_matrix = _camera_attention_face_matrix(tracks, attention_frames, config)
        # No reliable result means an exact fallback to the ordinary base rank.
        # This is also the complete code path when the UI option is disabled.
        eligible = [i for i in diag.camera_attention_shortlist_indices if i in attention_frames]
        if eligible and any(_camera_attention_stats(i, track_ids, attention_matrix, config)[0] > 0 for i in eligible):
            best_idx = max(
                eligible,
                key=lambda i: _camera_attention_rank(
                    i,
                    base_scores[i],
                    base_ranks[i],
                    track_ids,
                    attention_matrix,
                    config,
                    use_preference=use_preference,
                ),
            )
            known, away, mean_attention, _deficit = _camera_attention_stats(
                best_idx, track_ids, attention_matrix, config
            )
            diag.camera_attention_known = known
            diag.camera_attention_away = away
            diag.camera_attention_mean = mean_attention
    log = logging.getLogger("photo_select_ai")
    if diagnostic_log:
        for i, frame in enumerate(frames):
            severe, closed_count, known_count, eye_deficit = _eye_risk_stats(
                i, track_ids, face_matrix, config, eye_problem_limits
            )
            visible = len(matrix[i])
            att_known = att_away = 0
            att_mean = 0.50
            if attention_matrix:
                att_known, att_away, att_mean, _att_deficit = _camera_attention_stats(
                    i, track_ids, attention_matrix, config
                )
            pref_known, preferred, pref_worst, pref_mean, _pref_aggregate = _portrait_preference_stats(
                i, track_ids, portrait_matrix, config
            )
            defect_key = _group_frame_defect_key(
                i, track_ids, face_matrix, frames, config, eye_problem_limits
            )
            log.info(
                "GROUP FRAME %s | score=%.4f | suitability=%s | visible=%d/%d | eyes_known=%d | "
                "eyes_below=%d | severe_blinks=%d | eye_deficit=%.3f | "
                "portrait_known=%d | portrait_top=%d | portrait_worst=%.3f | portrait_mean=%.3f | "
                "look_known=%d | look_away=%d | look_mean=%.3f",
                frame.photo.path.name, frame_scores[i], defect_key, visible, len(track_ids), known_count,
                closed_count, severe, eye_deficit, pref_known, preferred, pref_worst, pref_mean,
                att_known, att_away, att_mean,
            )
        chosen_pref = _portrait_preference_stats(best_idx, track_ids, portrait_matrix, config)
        diag.portrait_preference_known = chosen_pref[0]
        diag.portrait_preferred_faces = chosen_pref[1]
        diag.portrait_preference_worst = chosen_pref[2]
        diag.portrait_preference_mean = chosen_pref[3]
        log.info(
            "GROUP RED CHOSEN %s | tracks=%d | portrait_known=%d | portrait_top=%d | "
            "portrait_worst=%.3f | portrait_mean=%.3f | look_known=%d | look_away=%d | look_mean=%.3f",
            frames[best_idx].photo.path.name, len(track_ids), diag.portrait_preference_known,
            diag.portrait_preferred_faces, diag.portrait_preference_worst, diag.portrait_preference_mean,
            diag.camera_attention_known, diag.camera_attention_away, diag.camera_attention_mean,
        )

    chosen_pref = _portrait_preference_stats(best_idx, track_ids, portrait_matrix, config)
    diag.portrait_preference_known = chosen_pref[0]
    diag.portrait_preferred_faces = chosen_pref[1]
    diag.portrait_preference_worst = chosen_pref[2]
    diag.portrait_preference_mean = chosen_pref[3]

    if progress:
        progress(0.84, "выбор главного RED-кадра")

    problems: dict[int, str] = {}
    for tid in track_ids:
        problem = _main_problem_for_track(
            tid, best_idx, face_matrix, matrix, config, eye_problem_limits
        )
        if problem:
            problems[tid] = problem
            if problem == "eyes":
                diag.eye_problems += 1
            elif problem == "missing":
                diag.missing_problems += 1
            elif problem == "sharpness":
                diag.sharpness_problems += 1
            elif problem == "pose":
                diag.pose_problems += 1
            else:
                diag.quality_problems += 1

    main = Selection(
        photo=frames[best_idx].photo,
        label_role="red",
        score=frame_scores[best_idx],
        reason=(
            f"group_score={frame_scores[best_idx]:.3f}; people={len(tracks)}; "
            f"portrait_top={diag.portrait_preferred_faces}/{diag.portrait_preference_known}; "
            f"portrait_worst={diag.portrait_preference_worst:.3f}; portrait_mean={diag.portrait_preference_mean:.3f}; "
            f"problems={len(problems)}; eyes={diag.eye_problems}; missing={diag.missing_problems}; "
            f"pose={diag.pose_problems}; look_away={diag.camera_attention_away}/{diag.camera_attention_known}"
        ),
    )

    extras: list[Selection] = []
    if bool(cfg.get("find_headswap_candidates", True)):
        max_extra = max(0, min(5, int(cfg.get("max_extra_candidates", 3))))
        remaining = dict(problems)
        chosen_frames = {best_idx}

        while remaining and len(extras) < max_extra:
            if progress:
                progress(0.86 + 0.12 * len(extras) / max(1, max_extra), f"поиск YELLOW-кандидата {len(extras) + 1}/{max_extra}")
            best_candidate_idx: int | None = None
            best_cover: set[int] = set()
            best_utility = -1.0
            best_types: dict[str, int] = {}

            for idx in range(len(frames)):
                if idx in chosen_frames:
                    continue
                cover: set[int] = set()
                utility = 0.0
                types: dict[str, int] = {}
                for tid, problem in remaining.items():
                    resolved, gain = _candidate_resolves_problem(
                        problem, tid, best_idx, idx, face_matrix, matrix, config, eye_candidate_limits
                    )
                    if resolved:
                        cover.add(tid)
                        utility += gain
                        types[problem] = types.get(problem, 0) + 1
                # Coverage count dominates; utility breaks ties and favours eye
                # fixes over generic score improvements.
                rank = len(cover) * 10.0 + utility
                if cover and rank > best_utility:
                    best_utility = rank
                    best_cover = cover
                    best_candidate_idx = idx
                    best_types = types

            if best_candidate_idx is None:
                break

            diag.covered_problems += len(best_cover)
            chosen_frames.add(best_candidate_idx)
            for tid in best_cover:
                remaining.pop(tid, None)

            type_text = ",".join(f"{k}:{v}" for k, v in sorted(best_types.items()))
            extras.append(
                Selection(
                    photo=frames[best_candidate_idx].photo,
                    label_role="yellow",
                    score=frame_scores[best_candidate_idx],
                    reason=f"headswap_for={len(best_cover)}; fixes={type_text}; people={len(tracks)}",
                )
            )

        diag.unresolved_problems = len(remaining)

        # Zero means automatic mode: keep only problem-oriented YELLOWs. A
        # positive minimum is explicit user intent, so fill the reserve up to
        # that number. Prefer RED's suitability tier; if it is exhausted, walk
        # through the remaining tiers from best to worst. The protected gaze
        # rank is used within a tier when high-resolution data is available.
        min_extra = max(0, min(max_extra, int(cfg.get("min_extra_candidates", 1))))
        red_suitability = defect_keys[best_idx]
        while len(extras) < min_extra:
            backup_candidates = [
                idx for idx in range(len(frames))
                if idx not in chosen_frames and defect_keys[idx] == red_suitability
            ]
            forced_degraded = False
            if not backup_candidates:
                remaining_candidates = [
                    idx for idx in range(len(frames)) if idx not in chosen_frames
                ]
                if not remaining_candidates:
                    break
                next_suitability = min(defect_keys[idx] for idx in remaining_candidates)
                backup_candidates = [
                    idx for idx in remaining_candidates
                    if defect_keys[idx] == next_suitability
                ]
                forced_degraded = True
            measured_safe = [
                idx for idx in backup_candidates
                if idx in attention_frames
                and _camera_attention_stats(
                    idx, track_ids, attention_matrix, config
                )[0] > 0
            ] if attention_matrix else []
            if measured_safe:
                best_backup_idx = max(
                    measured_safe,
                    key=lambda idx: _camera_attention_rank(
                        idx,
                        base_scores[idx],
                        base_ranks[idx],
                        track_ids,
                        attention_matrix,
                        config,
                        use_preference=use_preference,
                    ),
                )
            else:
                best_backup_idx = max(backup_candidates, key=lambda idx: base_ranks[idx])
            chosen_frames.add(best_backup_idx)
            diag.backup_extras += 1
            backup_kind = "forced_backup" if forced_degraded else "backup"
            extras.append(
                Selection(
                    photo=frames[best_backup_idx].photo,
                    label_role="yellow",
                    score=frame_scores[best_backup_idx],
                    reason=(
                        f"{backup_kind}; suitability_tier={defect_keys[best_backup_idx]}; "
                        f"requested_min={min_extra}; people={len(tracks)}"
                    ),
                )
            )
        if len(extras) < min_extra:
            log.warning(
                "GROUP YELLOW MINIMUM NOT REACHED | requested=%d | available=%d | red=%s",
                min_extra, len(extras), frames[best_idx].photo.path.name,
            )
    else:
        diag.unresolved_problems = len(problems)

    if progress:
        progress(1.0, f"группа готова: RED + {len(extras)} YELLOW")
    return GroupSeriesSelection(main=main, extras=extras), diag

def has_group_face_count(frames: list[FrameAssessment], config: dict) -> bool:
    """Whether a block looks like a group, ignoring its current frame count.

    This helper is intentionally usable on a one-frame temporal fragment so the
    pipeline can merge it with neighbouring fragments *before* enforcing the
    minimum number of takes.
    """
    if not frames:
        return False
    cfg = config.get("group", {})
    analysis_cfg = config.get("analysis", {})
    min_people = max(2, int(cfg.get("min_people", 4)))
    min_det = float(cfg.get("track_det_thresh", 0.30))
    min_fraction = float(cfg.get("min_track_face_fraction", analysis_cfg.get("face_min_fraction", 0.0005)))
    counts = [
        sum(
            1 for face in frame.faces
            if face.descriptor and face.detection_confidence >= min_det and face.size_fraction >= min_fraction
        )
        for frame in frames
        if not frame.error
    ]
    return bool(counts) and median(counts) >= min_people


def looks_like_group_series(frames: list[FrameAssessment], config: dict) -> bool:
    cfg = config.get("group", {})
    if len(frames) < max(2, int(cfg.get("min_frames", 2))):
        return False
    return has_group_face_count(frames, config)



def _strict_frame_identity_overlap(left: FrameAssessment, right: FrameAssessment, max_distance: float) -> float:
    """Identity-only overlap for deciding whether two takes show the same group.

    Position is deliberately ignored here. Different groups often stand in the
    same marked places, so geometry is useful for tracking *within* one known
    group but is unsafe evidence for deciding that two groups are identical.
    """
    lfaces = [f for f in left.faces if f.descriptor]
    rfaces = [f for f in right.faces if f.descriptor]
    if not lfaces or not rfaces:
        return 0.0
    pairs: list[tuple[float, int, int]] = []
    for li, lf in enumerate(lfaces):
        for ri, rf in enumerate(rfaces):
            sim = cosine_similarity(lf.descriptor, rf.descriptor)
            dist = 1.0 - sim
            if dist <= max_distance:
                pairs.append((sim, li, ri))
    pairs.sort(reverse=True)
    used_l: set[int] = set()
    used_r: set[int] = set()
    matched = 0
    for _sim, li, ri in pairs:
        if li in used_l or ri in used_r:
            continue
        used_l.add(li)
        used_r.add(ri)
        matched += 1
    return matched / max(1, min(len(lfaces), len(rfaces)))


def split_group_candidate_by_identity(
    frames: list[FrameAssessment], config: dict,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[list[FrameAssessment]], int]:
    """Split a temporal chunk when the photographed group composition changes.

    The hard temporal scanner cannot separate back-to-back groups when the
    photographer starts the next group within a few seconds.  In group mode we
    therefore look for an identity discontinuity. A split is accepted only when
    the first frame of the potential new group is dissimilar to recent frames of
    the old group *and* is consistent with the following frame.

    This is intentionally conservative for the user's workflow where children
    do not repeat between physical groups.
    """
    if len(frames) <= 2:
        if progress:
            progress(1.0, f"проверено кадров: {len(frames)}/{len(frames)}")
        return ([frames] if frames else []), 0
    cfg = config.get("group", {})
    max_distance = max(0.05, min(0.50, float(cfg.get("split_face_distance", 0.24))))
    keep_overlap = max(0.05, min(1.0, float(cfg.get("split_keep_identity_overlap", 0.35))))
    new_consistency = max(0.05, min(1.0, float(cfg.get("split_new_group_overlap", 0.35))))
    old_reject = max(0.0, min(1.0, float(cfg.get("split_max_old_overlap", 0.15))))
    ref_count = max(1, min(4, int(cfg.get("split_reference_frames", 2))))
    min_people = max(2, int(cfg.get("min_people", 4)))

    groups: list[list[FrameAssessment]] = []
    start = 0
    i = 1
    splits = 0
    total_boundaries = max(1, len(frames) - 1)

    def report(current_index: int, detail: str = "проверка границ") -> None:
        if progress:
            done = max(0, min(total_boundaries, current_index))
            progress(done / total_boundaries, f"{detail}: {done}/{total_boundaries}")

    report(0)
    while i < len(frames):
        report(i - 1)
        current = frames[i]
        if len(current.faces) < min_people:
            i += 1
            continue
        refs = frames[max(start, i - ref_count):i]
        refs = [f for f in refs if len(f.faces) >= min_people]
        if not refs:
            i += 1
            continue
        old_overlap = max((_strict_frame_identity_overlap(ref, current, max_distance) for ref in refs), default=0.0)
        if old_overlap >= keep_overlap:
            i += 1
            continue

        # Confirm the new composition using the immediately following usable
        # group frame.  This prevents one bad detector/embedding frame from
        # creating a false boundary.
        next_idx = next((j for j in range(i + 1, min(len(frames), i + 3)) if len(frames[j].faces) >= min_people), None)
        if next_idx is None:
            i += 1
            continue
        nxt = frames[next_idx]
        candidate_consistency = _strict_frame_identity_overlap(current, nxt, max_distance)
        next_old_overlap = max((_strict_frame_identity_overlap(ref, nxt, max_distance) for ref in refs), default=0.0)
        if candidate_consistency >= new_consistency and old_overlap <= old_reject and next_old_overlap <= old_reject:
            if i > start:
                logging.getLogger("photo_select_ai").info(
                    "GROUP SPLIT before %s | old_overlap=%.3f | next_old=%.3f | new_consistency=%.3f",
                    current.photo.path.name, old_overlap, next_old_overlap, candidate_consistency,
                )
                groups.append(frames[start:i])
                start = i
                splits += 1
                i = next_idx + 1
                continue
        i += 1

    groups.append(frames[start:])
    report(total_boundaries, f"границы готовы, найдено разделений={splits}")
    return [g for g in groups if g], splits


def _sequence_source(frame: FrameAssessment) -> tuple[str, str]:
    """Return a conservative source key for filename-sequence ordering.

    Sequence numbers are only comparable inside the same directory and filename
    prefix (for example IMG_4048, IMG_4049, ...).  This avoids mixing files from
    two cameras/subfolders whose counters happen to overlap.
    """
    path = frame.photo.path
    stem = path.stem
    cut = len(stem)
    while cut > 0 and stem[cut - 1].isdigit():
        cut -= 1
    prefix = stem[:cut].lower()
    return (str(path.parent).lower(), prefix)


def stabilize_group_frame_order(frames: list[FrameAssessment]) -> list[FrameAssessment]:
    """Repair unreliable EXIF ordering using the camera filename counter.

    Group exports/JPEG previews can carry coarse, copied or otherwise unreliable
    timestamps.  When that happens frames from one burst may arrive as e.g.
    4049, 4063, 4048, 4057.  Tracking and physical-group splitting both assume
    a local chronological order, so use the filename sequence when it is safe:
    at least 80%% of frames have a sequence number and they come from one source
    (same folder + filename prefix).
    """
    if len(frames) <= 1:
        return list(frames)
    with_seq = [f for f in frames if f.photo.sequence_number is not None]
    if len(with_seq) / len(frames) < 0.80:
        return list(frames)
    sources = {_sequence_source(f) for f in with_seq}
    if len(sources) != 1:
        return list(frames)
    numbers = [int(f.photo.sequence_number) for f in with_seq if f.photo.sequence_number is not None]
    # A very large span is likely a counter rollover or mixed material.  Keep
    # EXIF order rather than inventing a wrong numeric chronology.
    if numbers and max(numbers) - min(numbers) > 5000:
        return list(frames)
    ordered = sorted(
        enumerate(frames),
        key=lambda item: (
            item[1].photo.sequence_number if item[1].photo.sequence_number is not None else 10**15,
            item[1].photo.capture_time,
            item[1].photo.path.name.lower(),
            item[0],
        ),
    )
    result = [frame for _idx, frame in ordered]
    if [f.photo.path for f in result] != [f.photo.path for f in frames]:
        logging.getLogger("photo_select_ai").info(
            "GROUP ORDER NORMALIZED by filename sequence | %s..%s | frames=%d",
            result[0].photo.path.name,
            result[-1].photo.path.name,
            len(result),
        )
    return result


def _block_sequence_bounds(block: list[FrameAssessment]) -> tuple[tuple[str, str], int, int] | None:
    seq_frames = [f for f in block if f.photo.sequence_number is not None]
    if not seq_frames:
        return None
    sources = {_sequence_source(f) for f in seq_frames}
    if len(sources) != 1:
        return None
    nums = [int(f.photo.sequence_number) for f in seq_frames if f.photo.sequence_number is not None]
    if not nums or max(nums) - min(nums) > 5000:
        return None
    return next(iter(sources)), min(nums), max(nums)


def _order_group_blocks_for_merge(blocks: list[list[FrameAssessment]]) -> list[list[FrameAssessment]]:
    """Put fragmented group blocks back into filename-sequence order.

    The old merge pass compared *only the next block in EXIF-derived order*.
    With unreliable timestamps, two consecutive photos from the same class were
    never compared at all.  We sort sequence-compatible fragments per source
    before identity matching; different folders/prefixes retain their original
    source order.
    """
    if len(blocks) <= 1:
        return [stabilize_group_frame_order(b) for b in blocks]

    normalized = [stabilize_group_frame_order(b) for b in blocks]
    bounds = [_block_sequence_bounds(b) for b in normalized]
    usable = [b for b in bounds if b is not None]
    if len(usable) / len(normalized) < 0.80:
        return normalized

    source_order: dict[tuple[str, str], int] = {}
    for bound in bounds:
        if bound is None:
            continue
        source = bound[0]
        if source not in source_order:
            source_order[source] = len(source_order)

    decorated = []
    for idx, (block, bound) in enumerate(zip(normalized, bounds)):
        if bound is None:
            decorated.append((10**9, 10**15, idx, block))
        else:
            source, lo, _hi = bound
            decorated.append((source_order[source], lo, idx, block))
    decorated.sort(key=lambda x: (x[0], x[1], x[2]))
    result = [x[3] for x in decorated]
    if [id(b) for b in result] != [id(b) for b in normalized]:
        logging.getLogger("photo_select_ai").info(
            "GROUP BLOCK ORDER NORMALIZED by filename sequence | blocks=%d",
            len(result),
        )
    return result


def _block_sequence_gap(left: list[FrameAssessment], right: list[FrameAssessment]) -> int | None:
    lb = _block_sequence_bounds(left)
    rb = _block_sequence_bounds(right)
    if lb is None or rb is None or lb[0] != rb[0]:
        return None
    _source, lmin, lmax = lb
    _source2, rmin, rmax = rb
    if lmax < rmin:
        return rmin - lmax
    if rmax < lmin:
        return lmin - rmax
    return 0


def merge_adjacent_group_blocks(
    blocks: list[list[FrameAssessment]], config: dict,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[list[FrameAssessment]], int]:
    """Merge hard temporal blocks that are still the same physical group.

    Group takes can contain a longer pause than portrait takes. Without this
    pass one physical group may produce several independent RED selections.
    We only compare adjacent blocks and require a substantial identity overlap,
    so the next actual group is not merged just because face counts are similar.
    """
    if len(blocks) <= 1:
        if progress:
            progress(1.0, f"проверено блоков: {len(blocks)}/{len(blocks)}")
        return [stabilize_group_frame_order(b) for b in blocks], 0
    cfg = config.get("group", {})
    max_seconds = max(0.0, float(cfg.get("cross_block_merge_seconds", 45.0)))
    max_sequence_gap = max(0, int(cfg.get("cross_block_sequence_merge_gap", 40)))
    # Filename order may repair moderately scrambled/missing EXIF timestamps,
    # but it must not override an arbitrarily large real pause. Adjacent camera
    # numbers commonly straddle two different photographed groups. The old
    # unlimited override could therefore absorb the first group into the next
    # one (for example IMG_6156 -> IMG_6157 across a 140 s pause).
    sequence_recovery_seconds = max(
        max_seconds,
        float(cfg.get("cross_block_sequence_recovery_seconds", 90.0)),
    )
    min_overlap = max(0.0, min(1.0, float(cfg.get("cross_block_min_identity_overlap", 0.75))))
    max_distance = max(0.05, min(0.60, float(cfg.get("cross_block_face_distance", 0.24))))

    # Critical for exported/edited group shoots: EXIF timestamps can put a burst
    # out of filename order. Identity merging only works if related fragments
    # actually get compared, so restore a safe local sequence order first.
    blocks = _order_group_blocks_for_merge(blocks)
    merged: list[list[FrameAssessment]] = [list(blocks[0])]
    count = 0
    total_boundaries = max(1, len(blocks) - 1)
    if progress:
        progress(0.0, f"проверка соседних блоков: 0/{total_boundaries}")
    for boundary_no, current in enumerate(blocks[1:], start=1):
        if progress:
            progress((boundary_no - 1) / total_boundaries, f"проверка соседних блоков: {boundary_no - 1}/{total_boundaries}")
        previous = merged[-1]
        if not previous or not current:
            merged.append(list(current)); continue
        raw_time_delta = (current[0].photo.capture_time - previous[-1].photo.capture_time).total_seconds()
        time_gap = abs(raw_time_delta)
        sequence_gap = _block_sequence_gap(previous, current)
        sequence_close = (
            sequence_gap is not None
            and sequence_gap <= max_sequence_gap
            and time_gap <= sequence_recovery_seconds
        )
        # Never turn a negative (out-of-order) timestamp into an artificial 0 s
        # gap.  A close filename sequence may explicitly recover such a burst.
        if time_gap > max_seconds and not sequence_close:
            merged.append(list(current)); continue
        # Do not merge blocks whose detected group sizes are substantially
        # different. This protects adjacent but genuinely different groups
        # that happen to share several children.
        left_frames = previous[-min(3, len(previous)):]
        right_frames = current[:min(3, len(current))]
        left_count = median([len(f.faces) for f in left_frames]) if left_frames else 0
        right_count = median([len(f.faces) for f in right_frames]) if right_frames else 0
        count_ratio = min(left_count, right_count) / max(1.0, max(left_count, right_count))
        min_count_ratio = max(0.0, min(1.0, float(cfg.get("cross_block_face_count_ratio", 0.70))))
        if count_ratio < min_count_ratio:
            merged.append(list(current)); continue
        # Require consistent identity-only overlap across boundary frames.
        # Using the single strongest pair was unsafe: one accidental embedding
        # match could chain many distinct groups together.
        overlaps = [_strict_frame_identity_overlap(lf, rf, max_distance) for lf in left_frames for rf in right_frames]
        overlap = median(overlaps) if overlaps else 0.0
        if overlap >= min_overlap:
            logging.getLogger("photo_select_ai").info(
                "GROUP BLOCK MERGE %s -> %s | gap=%.1fs | raw_delta=%+.1fs | seq_gap=%s | identity_overlap=%.3f | count_ratio=%.3f",
                previous[-1].photo.path.name, current[0].photo.path.name, time_gap, raw_time_delta,
                "n/a" if sequence_gap is None else str(sequence_gap), overlap, count_ratio,
            )
            previous.extend(current)
            previous[:] = stabilize_group_frame_order(previous)
            count += 1
        else:
            merged.append(list(current))
        if progress:
            progress(boundary_no / total_boundaries, f"проверка соседних блоков: {boundary_no}/{total_boundaries}; объединено={count}")
    if progress:
        progress(1.0, f"объединение блоков готово; объединено={count}")
    return merged, count
