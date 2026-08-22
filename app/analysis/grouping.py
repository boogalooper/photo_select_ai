from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import logging
from statistics import median
from typing import Callable

from app.analysis.matching import cosine_similarity
from app.core.models import FaceAssessment, FrameAssessment, Selection


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
    track_count: int
    rejected_reason: str | None = None


@dataclass(slots=True)
class GroupDiagnostics:
    tracks_built: int = 0
    confirmed_tracks: int = 0
    stable_tracks: int = 0
    promoted_tracks: int = 0
    highres_rescue_tracks: int = 0
    max_faces_in_frame: int = 0
    highres_max_faces_in_frame: int = 0
    highres_tracks_built: int = 0
    candidate_extras: int = 0
    eye_problems: int = 0
    missing_problems: int = 0
    sharpness_problems: int = 0
    quality_problems: int = 0
    covered_problems: int = 0
    unresolved_problems: int = 0
    backup_extras: int = 0
    camera_attention_shortlist_indices: list[int] = field(default_factory=list)
    camera_attention_known: int = 0
    camera_attention_away: int = 0
    camera_attention_mean: float = 0.0


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




def _promote_plausible_singletons(
    stable: list[PersonTrack],
    candidates: list[PersonTrack],
    config: dict,
) -> list[PersonTrack]:
    """Recover a real group member that was detected in only one take.

    Group photographs are unusually friendly to positional reasoning: the people
    stay in roughly the same arrangement for consecutive takes.  Requiring two
    detections for every person is therefore unnecessarily strict and causes the
    displayed roster (and RED scoring) to under-count children that the detector
    misses on one take.

    We do *not* simply accept every one-frame detection.  A singleton is promoted
    only when it lies inside the spatial envelope of the already stable group and
    has a face size compatible with that group.  This keeps small/background
    bystanders from entering the roster.
    """
    cfg = config.get("group", {})
    if not bool(cfg.get("promote_singleton_tracks", False)):
        return []
    min_people = max(2, int(cfg.get("min_people", 4)))
    if len(stable) < min_people:
        return []

    centers = [_track_median_center(t) for t in stable]
    sizes = [_track_median_size(t) for t in stable if _track_median_size(t) > 0]
    if not centers or not sizes:
        return []
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_factor = max(0.0, float(cfg.get("singleton_roi_padding", 0.10)))
    min_pad = max(0.0, float(cfg.get("singleton_min_roi_padding", 0.045)))
    # Allow roughly one missing grid position beyond the stable envelope.
    # This matters when the only missed child stands at the far left/right edge.
    nn_distances: list[float] = []
    if len(centers) >= 2:
        for i, (cx, cy) in enumerate(centers):
            nearest = min(
                (((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5)
                for j, (ox, oy) in enumerate(centers) if j != i
            )
            nn_distances.append(nearest)
    neighbor_pad = (
        float(median(nn_distances))
        * max(0.0, float(cfg.get("singleton_neighbor_pad_factor", 1.05)))
        if nn_distances else 0.0
    )
    padx = max(min_pad, (xmax - xmin) * pad_factor, neighbor_pad)
    pady = max(min_pad, (ymax - ymin) * pad_factor, neighbor_pad)
    median_size = float(median(sizes))
    min_size_ratio = max(0.05, min(1.0, float(cfg.get("singleton_min_size_ratio", 0.38))))
    min_det = max(0.0, min(1.0, float(cfg.get("singleton_det_thresh", cfg.get("track_det_thresh", 0.22)))))
    dup_dist = max(0.005, min(0.10, float(cfg.get("singleton_duplicate_distance", 0.025))))

    promoted: list[PersonTrack] = []
    for track in candidates:
        # At the default min_track_presence=2 these are exactly one-frame
        # fragments.  If a user raises that setting, do not promote a more
        # ambiguous multi-frame fragment automatically.
        if len(track.observations) != 1:
            continue
        frame_idx, face = track.observations[0]
        if face.detection_confidence < min_det:
            continue
        if face.size_fraction < median_size * min_size_ratio:
            continue
        x, y = face.center
        if not (xmin - padx <= x <= xmax + padx and ymin - pady <= y <= ymax + pady):
            continue

        # Reject an accidental duplicate detection sitting almost on top of an
        # already stable face in the same take.
        duplicate = False
        for stable_track in stable:
            for stable_frame_idx, stable_face in stable_track.observations:
                if stable_frame_idx != frame_idx:
                    continue
                dx = x - stable_face.center[0]
                dy = y - stable_face.center[1]
                if (dx * dx + dy * dy) ** 0.5 <= dup_dist:
                    duplicate = True
                    break
            if duplicate:
                break
        if not duplicate:
            promoted.append(track)
    return promoted


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
    stable = [t for t in tracks if len(t.observations) >= min_presence]
    not_stable = [t for t in tracks if len(t.observations) < min_presence]
    promoted = _promote_plausible_singletons(stable, not_stable, config)
    roster = stable + promoted

    # Deterministic top-to-bottom/left-to-right IDs after recovery.
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
        if promoted:
            details.append(f"восстановлено={len(promoted)}")
        if merged_count:
            details.append(f"склеено фрагментов={merged_count}")
        progress(0.55, f"людей в составе: {len(roster)} ({'; '.join(details)})")
    if log_roster:
        log = logging.getLogger("photo_select_ai")
        log.info(
            "GROUP ROSTER | people=%d | stable=%d | promoted=%d | max_faces_frame=%d | raw_tracks=%d",
            len(roster), len(stable), len(promoted), max_faces, raw_track_count,
        )
    return roster, GroupDiagnostics(
        tracks_built=raw_track_count,
        confirmed_tracks=len(roster),
        stable_tracks=len(stable),
        promoted_tracks=len(promoted),
        max_faces_in_frame=max_faces,
    )



def _median_nearest_neighbor_distance(centers: list[tuple[float, float]]) -> float:
    if len(centers) < 2:
        return 0.08
    distances: list[float] = []
    for i, (cx, cy) in enumerate(centers):
        nearest = min(
            ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            for j, (ox, oy) in enumerate(centers)
            if j != i
        )
        distances.append(nearest)
    return float(median(distances)) if distances else 0.08


def _highres_rescue_tracking_config(config: dict, frame_count: int) -> dict:
    """Build a conservative tracking profile for the second detector pass.

    The detector itself is configured by the pipeline at a larger det_size and
    lower confidence threshold.  Here we only tell the tracker to accept those
    weaker detections, require repeated observations, and never revive a
    one-frame singleton.
    """
    result = deepcopy(config)
    cfg = result.setdefault("group", {})
    primary = config.get("group", {})
    min_presence = max(2, int(primary.get("highres_rescue_min_presence", 2)))
    fraction = max(0.0, min(1.0, float(primary.get("highres_rescue_min_presence_fraction", 0.0))))
    if fraction > 0.0:
        min_presence = max(min_presence, int(round(frame_count * fraction)))
    cfg["min_track_presence"] = min_presence
    cfg["track_det_thresh"] = float(primary.get("highres_rescue_det_thresh", 0.14))
    cfg["min_track_face_fraction"] = float(primary.get("highres_rescue_min_face_fraction", 0.00018))
    cfg["promote_singleton_tracks"] = False
    # A weak high-res detection may disappear for a few takes, so give the
    # rescue tracker a slightly wider temporal bridge than the primary pass.
    cfg["track_max_frame_gap"] = max(
        int(primary.get("track_max_frame_gap", 4)),
        int(primary.get("highres_rescue_track_max_frame_gap", 5)),
    )
    return result


def _highres_rescue_tracks(
    primary_tracks: list[PersonTrack],
    rescue_frames: list[FrameAssessment],
    config: dict,
) -> tuple[list[PersonTrack], int, int]:
    """Return repeated high-res faces that represent genuinely missing people.

    The primary pass defines the group roster and geometry.  A second, more
    sensitive detector pass is allowed to add a person only when a repeated
    track survives all of these gates:
      * it lies inside (or one normal grid step just outside) the stable group;
      * its face size is compatible with nearby group members;
      * it is not the same identity as an existing primary track;
      * it is not sitting on an already occupied spatial slot.

    This is intentionally stricter than v0.4.2 singleton promotion: one noisy
    detection can never create a new person.
    """
    if not primary_tracks or not rescue_frames:
        return [], 0, 0
    cfg = config.get("group", {})
    if not bool(cfg.get("highres_rescue_enabled", False)):
        return [], 0, 0

    rescue_config = _highres_rescue_tracking_config(config, len(rescue_frames))
    highres_tracks, highres_diag = build_person_tracks(
        rescue_frames, rescue_config, progress=None, log_roster=False
    )
    if not highres_tracks:
        return [], highres_diag.max_faces_in_frame, highres_diag.tracks_built

    primary_centers = [_track_median_center(t) for t in primary_tracks]
    primary_sizes = [_track_median_size(t) for t in primary_tracks]
    valid_sizes = [s for s in primary_sizes if s > 0.0]
    if not primary_centers or not valid_sizes:
        return [], highres_diag.max_faces_in_frame, highres_diag.tracks_built

    xs = [c[0] for c in primary_centers]
    ys = [c[1] for c in primary_centers]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    nn = _median_nearest_neighbor_distance(primary_centers)
    pad = max(
        float(cfg.get("highres_rescue_min_roi_padding", 0.035)),
        nn * float(cfg.get("highres_rescue_roi_neighbor_padding", 1.05)),
    )
    duplicate_slot_distance = max(
        0.012,
        min(
            0.050,
            nn * float(cfg.get("highres_rescue_duplicate_slot_factor", 0.32)),
        ),
    )
    min_size_ratio = max(0.05, min(1.0, float(cfg.get("highres_rescue_min_size_ratio", 0.40))))
    strong_identity_distance = max(0.05, min(0.45, float(cfg.get("highres_rescue_strong_identity_distance", 0.18))))
    normal_identity_distance = max(strong_identity_distance, min(0.55, float(cfg.get("highres_rescue_identity_distance", 0.28))))
    identity_position_gate = max(duplicate_slot_distance, min(0.20, float(cfg.get("highres_rescue_identity_position_gate", 0.10))))

    accepted: list[PersonTrack] = []
    log = logging.getLogger("photo_select_ai")
    for candidate in highres_tracks:
        cx, cy = _track_median_center(candidate)
        csize = _track_median_size(candidate)
        if not (xmin - pad <= cx <= xmax + pad and ymin - pad <= cy <= ymax + pad):
            continue

        # Compare size to the local neighbourhood rather than one global median;
        # back-row children are naturally smaller than front-row children.
        nearest_indices = sorted(
            range(len(primary_tracks)),
            key=lambda i: (cx - primary_centers[i][0]) ** 2 + (cy - primary_centers[i][1]) ** 2,
        )[: min(3, len(primary_tracks))]
        local_sizes = [primary_sizes[i] for i in nearest_indices if primary_sizes[i] > 0]
        local_size = float(median(local_sizes)) if local_sizes else float(median(valid_sizes))
        size_ratio = min(csize, local_size) / max(1e-9, max(csize, local_size))
        if size_ratio < min_size_ratio:
            continue

        candidate_desc = candidate.centroid
        duplicate = False
        nearest_primary_distance = 999.0
        for i, primary in enumerate(primary_tracks):
            px, py = primary_centers[i]
            pos_distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            nearest_primary_distance = min(nearest_primary_distance, pos_distance)
            if pos_distance <= duplicate_slot_distance:
                duplicate = True
                break
            primary_desc = primary.centroid
            if candidate_desc and primary_desc and len(candidate_desc) == len(primary_desc):
                identity_distance = 1.0 - cosine_similarity(candidate_desc, primary_desc)
                if identity_distance <= strong_identity_distance:
                    duplicate = True
                    break
                if identity_distance <= normal_identity_distance and pos_distance <= identity_position_gate:
                    duplicate = True
                    break
        if duplicate:
            continue

        accepted.append(candidate)
        log.info(
            "GROUP HIGHRES RESCUE ACCEPT | observations=%d | center=(%.4f,%.4f) | size=%.6f | nearest_slot=%.4f",
            len(candidate.observations), cx, cy, csize, nearest_primary_distance,
        )

    return accepted, highres_diag.max_faces_in_frame, highres_diag.tracks_built


def _combine_primary_and_rescue_tracks(
    primary_tracks: list[PersonTrack],
    rescue_tracks: list[PersonTrack],
) -> list[PersonTrack]:
    combined = list(primary_tracks) + list(rescue_tracks)
    combined.sort(key=lambda t: (_track_median_center(t)[1], _track_median_center(t)[0], t.track_id))
    for new_id, track in enumerate(combined, start=1):
        track.track_id = new_id
    return combined

def _frame_track_matrix(frames: list[FrameAssessment], tracks: list[PersonTrack], config: dict) -> list[dict[int, float]]:
    eye_threshold = float(config.get("analysis", {}).get("eye_open_threshold", 0.52))
    matrix: list[dict[int, float]] = [dict() for _ in frames]
    for track in tracks:
        for frame_idx, face in track.observations:
            if 0 <= frame_idx < len(frames):
                matrix[frame_idx][track.track_id] = _face_quality(face, eye_threshold)
    return matrix


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


def _open_eye_stats(
    frame_idx: int,
    track_ids: list[int],
    face_matrix: list[dict[int, FaceAssessment]],
    config: dict,
    eye_problem_limits: dict[int, float] | None = None,
) -> tuple[int, int, int]:
    """Return (open, closed, known) for reliable eye states in one group frame."""
    global_threshold = float(config.get("group", {}).get("eye_problem_threshold", 0.62))
    open_count = closed_count = known_count = 0
    for tid in track_ids:
        face = face_matrix[frame_idx].get(tid)
        if face is None or not face.landmarks_reliable:
            continue
        known_count += 1
        threshold = (eye_problem_limits or {}).get(tid, global_threshold)
        if face.eyes_open_score >= threshold:
            open_count += 1
        else:
            closed_count += 1
    return open_count, closed_count, known_count


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

    sharp_problem = float(cfg.get("eye_sharpness_problem_threshold", 0.42))
    if face.landmarks_reliable and face.eye_sharpness < sharp_problem:
        return "sharpness"

    if quality_matrix[best_idx].get(track_id, 0.0) < float(cfg.get("good_face_threshold", 0.62)):
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

    if problem == "sharpness":
        sharp_delta = float(cfg.get("eye_sharpness_improvement_margin", 0.10))
        main_sharp = main_face.eye_sharpness if main_face else 0.0
        eye_floor = float(cfg.get("headswap_eye_floor", 0.48))
        eyes = cand_face.eyes_open_score if cand_face.landmarks_reliable else 0.5
        ok = cand_face.eye_sharpness >= main_sharp + sharp_delta and eyes >= eye_floor
        return ok, 1.7 + max(0.0, cand_face.eye_sharpness - main_sharp) if ok else 0.0

    quality_delta = float(cfg.get("quality_improvement_margin", cfg.get("improvement_margin", 0.12)))
    ok = cand_q >= max(min_candidate_q, main_q + quality_delta)
    return ok, 1.0 + max(0.0, cand_q - main_q) if ok else 0.0




def _base_main_rank_data(
    frame_idx: int,
    track_ids: list[int],
    matrix: list[dict[int, float]],
    face_matrix: list[dict[int, FaceAssessment]],
    frame_scores: list[float],
    config: dict,
    eye_problem_limits: dict[int, float],
) -> tuple[float, tuple]:
    severe, closed_count, known_count, eye_deficit = _eye_risk_stats(
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
    rank = (stable_score, -severe, -closed_count, visible, known_count, frame_scores[frame_idx])
    return stable_score, rank


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


def _camera_attention_rank(
    frame_idx: int,
    base_score: float,
    base_rank: tuple,
    track_ids: list[int],
    attention_matrix: dict[int, dict[int, FaceAssessment]],
    config: dict,
) -> tuple:
    cfg = config.get("group", {})
    people = max(1, len(track_ids))
    known, away, mean, deficit = _camera_attention_stats(
        frame_idx, track_ids, attention_matrix, config
    )
    # Unknown faces are neutral (0.5), never treated as looking away. This
    # prevents tiny/ambiguous eyes from incorrectly deciding the RED frame.
    effective_mean = (mean * known + 0.50 * (people - known)) / people
    away_fraction = away / people
    deficit_fraction = deficit / people
    influence = float(cfg.get("camera_attention_influence", 0.12))
    away_penalty = float(cfg.get("camera_attention_away_penalty", 0.45))
    deficit_weight = float(cfg.get("camera_attention_deficit_weight", 0.20))
    final_score = (
        base_score
        + influence * (effective_mean - 0.50)
        - away_penalty * away_fraction
        - deficit_weight * deficit_fraction
    )
    return (final_score, -away, effective_mean, known, *base_rank)


def select_group_series(
    frames: list[FrameAssessment],
    config: dict,
    progress: Callable[[float, str], None] | None = None,
    *,
    rescue_frames: list[FrameAssessment] | None = None,
    attention_frames: dict[int, FrameAssessment] | None = None,
    diagnostic_log: bool = True,
) -> tuple[GroupSeriesSelection, GroupDiagnostics]:
    # v0.4.3: the primary roster is stable-only. One-frame promotion from
    # v0.4.2 is deliberately disabled because real validation showed it added
    # false people (4a/3e/4d). Missing people are recovered only by the repeated
    # high-resolution pass below.
    primary_config = deepcopy(config)
    primary_config.setdefault("group", {})["promote_singleton_tracks"] = False
    tracks, diag = build_person_tracks(
        frames, primary_config, progress=progress, log_roster=False
    )
    rescue_tracks: list[PersonTrack] = []
    if rescue_frames is not None:
        rescue_tracks, highres_max_faces, highres_tracks_built = _highres_rescue_tracks(
            tracks, rescue_frames, config
        )
        diag.highres_rescue_tracks = len(rescue_tracks)
        diag.highres_max_faces_in_frame = highres_max_faces
        diag.highres_tracks_built = highres_tracks_built
        if rescue_tracks:
            tracks = _combine_primary_and_rescue_tracks(tracks, rescue_tracks)
        diag.confirmed_tracks = len(tracks)
        if progress:
            progress(0.58, f"состав: устойчивых={diag.stable_tracks}; high-res восстановлено={len(rescue_tracks)}")

    log = logging.getLogger("photo_select_ai")
    if diagnostic_log:
        log.info(
            "GROUP ROSTER | people=%d | stable=%d | rescued=%d | primary_max_faces_frame=%d | "
            "highres_max_faces_frame=%d | primary_raw_tracks=%d | highres_raw_tracks=%d",
            len(tracks), diag.stable_tracks, diag.highres_rescue_tracks, diag.max_faces_in_frame,
            diag.highres_max_faces_in_frame, diag.tracks_built, diag.highres_tracks_built,
        )
    cfg = config.get("group", {})
    min_people = max(2, int(cfg.get("min_people", 4)))
    if len(tracks) < min_people:
        return GroupSeriesSelection(main=None, extras=[], track_count=len(tracks), rejected_reason="too_few_people"), diag

    track_ids = [t.track_id for t in tracks]
    if progress:
        progress(0.62, "оценка лиц каждого ребёнка")
    matrix = _frame_track_matrix(frames, tracks, config)
    face_matrix = _frame_track_faces(frames, tracks)
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
    if bool(cfg.get("prioritize_open_eyes_main", True)):
        # Eyes remain important, but do not use a brittle lexicographic
        # open-eye counter.  A marginal landmark measurement should not beat a
        # clearly sharper/better frame merely because it crossed one threshold.
        for i in range(len(frames)):
            base_scores[i], base_ranks[i] = _base_main_rank_data(
                i, track_ids, matrix, face_matrix, frame_scores, config, eye_problem_limits
            )
    else:
        for i in range(len(frames)):
            base_scores[i] = frame_scores[i]
            base_ranks[i] = (frame_scores[i],)

    base_order = sorted(range(len(frames)), key=lambda i: base_ranks[i], reverse=True)
    shortlist_count = max(1, min(len(frames), int(cfg.get("camera_attention_shortlist", 3))))
    diag.camera_attention_shortlist_indices = base_order[:shortlist_count]
    best_idx = base_order[0]

    attention_matrix: dict[int, dict[int, FaceAssessment]] = {}
    if bool(cfg.get("camera_attention_enabled", False)) and attention_frames:
        attention_matrix = _camera_attention_face_matrix(tracks, attention_frames, config)
        # Camera attention is deliberately a final-stage comparison among the
        # already strongest ordinary candidates. It cannot rescue a technically
        # poor/blinking frame from far down the list.
        eligible = [i for i in diag.camera_attention_shortlist_indices if i in attention_frames]
        if eligible and any(_camera_attention_stats(i, track_ids, attention_matrix, config)[0] > 0 for i in eligible):
            best_idx = max(
                eligible,
                key=lambda i: _camera_attention_rank(
                    i, base_scores[i], base_ranks[i], track_ids, attention_matrix, config
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
            log.info(
                "GROUP FRAME %s | score=%.4f | visible=%d/%d | eyes_known=%d | "
                "eyes_below=%d | severe_blinks=%d | eye_deficit=%.3f | "
                "look_known=%d | look_away=%d | look_mean=%.3f",
                frame.photo.path.name, frame_scores[i], visible, len(track_ids), known_count,
                closed_count, severe, eye_deficit, att_known, att_away, att_mean,
            )
        log.info(
            "GROUP RED CHOSEN %s | tracks=%d | look_known=%d | look_away=%d | look_mean=%.3f",
            frames[best_idx].photo.path.name, len(track_ids), diag.camera_attention_known,
            diag.camera_attention_away, diag.camera_attention_mean,
        )

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
            else:
                diag.quality_problems += 1

    main = Selection(
        photo=frames[best_idx].photo,
        label_role="red",
        score=frame_scores[best_idx],
        reason=(
            f"group_score={frame_scores[best_idx]:.3f}; people={len(tracks)}; "
            f"problems={len(problems)}; eyes={diag.eye_problems}; missing={diag.missing_problems}; "
            f"look_away={diag.camera_attention_away}/{diag.camera_attention_known}"
        ),
    )

    extras: list[Selection] = []
    if bool(cfg.get("find_headswap_candidates", True)):
        max_extra = max(0, min(3, int(cfg.get("max_extra_candidates", 3))))
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

            diag.candidate_extras += 1
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

        # Always keep a practical reserve when requested. Targeted YELLOWs above
        # fix concrete defects; backup YELLOWs below are simply strong alternate
        # takes for manual retouching when strict defect rules found too little.
        min_extra = max(0, min(max_extra, int(cfg.get("min_extra_candidates", 1))))
        backup_score_ratio = max(0.0, min(1.0, float(cfg.get("backup_min_score_ratio", 0.86))))
        person_margin = max(0.0, float(cfg.get("backup_person_improvement_margin", 0.05)))
        while len(extras) < min_extra:
            best_backup_idx: int | None = None
            best_backup_rank: tuple[float, float, float] | None = None
            best_improved = 0
            main_score = frame_scores[best_idx]
            for idx in range(len(frames)):
                if idx in chosen_frames:
                    continue
                improved = sum(
                    1 for tid in track_ids
                    if matrix[idx].get(tid, 0.0) >= matrix[best_idx].get(tid, 0.0) + person_margin
                )
                close_enough = main_score <= 1e-9 or frame_scores[idx] >= main_score * backup_score_ratio
                if not close_enough and improved <= 0:
                    continue
                open_count, closed_count, _known = _open_eye_stats(
                    idx, track_ids, face_matrix, config, eye_problem_limits
                )
                rank = (float(improved), float(open_count - closed_count), frame_scores[idx])
                if best_backup_rank is None or rank > best_backup_rank:
                    best_backup_rank = rank
                    best_backup_idx = idx
                    best_improved = improved
            if best_backup_idx is None:
                break
            chosen_frames.add(best_backup_idx)
            diag.candidate_extras += 1
            diag.backup_extras += 1
            extras.append(
                Selection(
                    photo=frames[best_backup_idx].photo,
                    label_role="yellow",
                    score=frame_scores[best_backup_idx],
                    reason=f"backup; children_better={best_improved}; people={len(tracks)}",
                )
            )
    else:
        diag.unresolved_problems = len(problems)

    if progress:
        progress(1.0, f"группа готова: RED + {len(extras)} YELLOW")
    return GroupSeriesSelection(main=main, extras=extras, track_count=len(tracks)), diag

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

def _frame_identity_overlap(left: FrameAssessment, right: FrameAssessment, max_distance: float) -> float:
    """Fraction of faces in the smaller frame that can be identity-matched."""
    lfaces = [f for f in left.faces if f.descriptor]
    rfaces = [f for f in right.faces if f.descriptor]
    if not lfaces or not rfaces:
        return 0.0
    pairs: list[tuple[float, int, int]] = []
    for li, lf in enumerate(lfaces):
        for ri, rf in enumerate(rfaces):
            sim = cosine_similarity(lf.descriptor, rf.descriptor)
            dist = 1.0 - sim
            # Position is only a tie-breaker. Children can shift slightly
            # between takes, but usually keep roughly the same place.
            pos = _position_similarity(lf, rf)
            score = sim + 0.05 * pos
            if dist <= max_distance:
                pairs.append((score, li, ri))
    pairs.sort(reverse=True)
    used_l: set[int] = set()
    used_r: set[int] = set()
    matched = 0
    for _score, li, ri in pairs:
        if li in used_l or ri in used_r:
            continue
        used_l.add(li); used_r.add(ri); matched += 1
    return matched / max(1, min(len(lfaces), len(rfaces)))



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
        sequence_close = sequence_gap is not None and sequence_gap <= max_sequence_gap
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
