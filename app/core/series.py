from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass

from app.analysis.matching import cosine_similarity
from .models import FaceAssessment, FrameAssessment, PhotoFile, PhotoSeries


def build_candidate_series(photos: list[PhotoFile], config: dict) -> list[PhotoSeries]:
    """Build only *hard* temporal chunks.

    Portrait subject changes are intentionally NOT decided here. The candidate
    stage merely prevents obviously unrelated parts of a shoot from being
    compared. Fine splitting is done after face embeddings are available.
    """
    if not photos:
        return []
    mode = str(config.get("runtime", {}).get("mode", "portrait")).lower()
    if mode == "group":
        group_cfg = config.get("group", {})
        max_gap = float(group_cfg.get("max_gap_seconds", config["series"]["max_gap_seconds"]))
        max_name_gap = int(group_cfg.get("max_filename_gap", config["series"]["max_filename_gap"]))
    else:
        max_gap = float(config["series"]["max_gap_seconds"])
        max_name_gap = int(config["series"]["max_filename_gap"])
    groups: list[list[PhotoFile]] = [[photos[0]]]

    for prev, cur in zip(photos, photos[1:]):
        time_gap = max(0.0, (cur.capture_time - prev.capture_time).total_seconds())
        name_gap_ok = True
        if prev.sequence_number is not None and cur.sequence_number is not None:
            delta = cur.sequence_number - prev.sequence_number
            name_gap_ok = 0 < delta <= max_name_gap
        same_source = True
        if mode == "group":
            same_source = prev.sequence_source == cur.sequence_source
        if same_source and time_gap <= max_gap and name_gap_ok:
            groups[-1].append(cur)
        else:
            groups.append([cur])

    return [PhotoSeries(index=i + 1, photos=g) for i, g in enumerate(groups)]



def _same_person_threshold(config: dict) -> float:
    return max(0.0, min(1.0, float(config.get("series", {}).get("portrait_same_person_similarity", 0.42))))


def _descriptor_similarity(a: FaceAssessment | None, b: FaceAssessment | None) -> float | None:
    if a is None or b is None or not a.descriptor or not b.descriptor:
        return None
    # Different descriptor backends have different vector lengths. Treat them
    # as unknown rather than falsely calling two people different.
    if len(a.descriptor) != len(b.descriptor):
        return None
    return max(-1.0, min(1.0, cosine_similarity(a.descriptor, b.descriptor)))


def _subject_score(face: FaceAssessment) -> float:
    dx = face.center[0] - 0.5
    dy = face.center[1] - 0.5
    dist = min(1.0, (dx * dx + dy * dy) ** 0.5 / 0.7071)
    centrality = 1.0 - dist
    return face.size_fraction * (0.72 + 0.28 * centrality)


def _initial_subject(frame: FrameAssessment) -> FaceAssessment | None:
    return max(frame.faces, key=_subject_score) if frame.faces else None


def _best_matching_face(frame: FrameAssessment, refs: list[FaceAssessment]) -> tuple[FaceAssessment | None, float | None]:
    if not frame.faces:
        return None, None
    best_face: FaceAssessment | None = None
    best_sim: float | None = None
    for face in frame.faces:
        sims = [_descriptor_similarity(face, ref) for ref in refs]
        sims = [s for s in sims if s is not None]
        if not sims:
            continue
        score = max(sims)
        if best_sim is None or score > best_sim:
            best_sim = score
            best_face = face
    if best_face is None:
        return _initial_subject(frame), None
    return best_face, best_sim


def _recent_subject_refs(frames: list[FrameAssessment], start: int, end: int, limit: int = 5) -> list[FaceAssessment]:
    refs: list[FaceAssessment] = []
    # Walk backwards and maintain identity continuity. This is more stable than
    # blindly using each frame's largest face when an assistant is in view.
    current: FaceAssessment | None = None
    for idx in range(end - 1, start - 1, -1):
        frame = frames[idx]
        if not frame.faces:
            continue
        if current is None:
            current = _initial_subject(frame)
        else:
            matched, _ = _best_matching_face(frame, [current])
            current = matched or current
        if current is not None and current.descriptor:
            refs.append(current)
        if len(refs) >= limit:
            break
    return refs


def _new_faces_consistent(faces: list[FaceAssessment], threshold: float) -> bool:
    if len(faces) <= 1:
        return True
    softer = max(0.20, threshold - 0.10)
    sims: list[float] = []
    for left, right in zip(faces, faces[1:]):
        sim = _descriptor_similarity(left, right)
        if sim is not None:
            sims.append(sim)
    return not sims or statistics.median(sims) >= softer


def _refine_portrait_v2(frames: list[FrameAssessment], config: dict) -> list[list[FrameAssessment]]:
    """Split portrait sequences using SFace identity continuity + chronology.

    The algorithm is deliberately conservative: an isolated detector miss,
    profile turn or poor embedding does not create a new child. A subject change
    must be confirmed by several consecutive usable faces that are mutually
    consistent and dissimilar to the recent subject.
    """
    if len(frames) <= 1:
        return [frames]

    threshold = _same_person_threshold(config)
    confirm = max(1, int(config["series"].get("portrait_break_confirm_frames", 2)))
    no_face_tolerance = max(0, int(config["series"].get("portrait_no_face_tolerance", 3)))

    groups: list[list[FrameAssessment]] = []
    start = 0
    i = 1

    while i < len(frames):
        current = frames[i]
        if not current.faces:
            i += 1
            continue

        old_refs = _recent_subject_refs(frames, start, i, limit=5)
        if not old_refs:
            i += 1
            continue

        current_face, current_sim = _best_matching_face(current, old_refs)
        if current_sim is None or current_sim >= threshold:
            i += 1
            continue

        # Potential new child. Confirm with several following detected faces.
        pending_faces: list[FaceAssessment] = []
        pending_indices: list[int] = []
        misses = 0
        k = i
        seed = current_face or _initial_subject(current)
        while k < len(frames) and len(pending_faces) < confirm:
            frame = frames[k]
            if not frame.faces:
                misses += 1
                if misses > no_face_tolerance:
                    break
                k += 1
                continue
            misses = 0
            if not pending_faces:
                face = seed or _initial_subject(frame)
            else:
                face, sim_to_new = _best_matching_face(frame, pending_faces[-3:])
                # If nothing matches the pending new child, choose the dominant
                # face but let consistency validation reject a bad boundary.
                if sim_to_new is not None and sim_to_new < max(0.20, threshold - 0.12):
                    face = _initial_subject(frame)
            if face is not None:
                pending_faces.append(face)
                pending_indices.append(k)
            k += 1

        if len(pending_faces) < confirm or not _new_faces_consistent(pending_faces, threshold):
            i += 1
            continue

        # Every confirmed new face should remain dissimilar to the old child.
        genuinely_new = True
        for face in pending_faces:
            sims = [_descriptor_similarity(face, ref) for ref in old_refs]
            sims = [s for s in sims if s is not None]
            if sims and max(sims) >= threshold:
                genuinely_new = False
                break

        if genuinely_new:
            groups.append(frames[start:i])
            start = i
            i = pending_indices[-1] + 1
        else:
            i += 1

    groups.append(frames[start:])
    return [g for g in groups if g]



def _normalise_vector(values: list[float]) -> list[float]:
    if not values:
        return []
    norm = sum(v * v for v in values) ** 0.5
    if norm <= 1e-12:
        return []
    return [float(v / norm) for v in values]


def _cluster_centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    if width == 0 or any(len(v) != width for v in vectors):
        return []
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]
    return _normalise_vector(mean)


def _cosine_distance_vec(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    return max(0.0, min(2.0, 1.0 - cosine_similarity(a, b)))


def _portrait_descriptor(frame: FrameAssessment) -> list[float]:
    face = frame.primary_face
    if face is None or not face.descriptor:
        return []
    # Only real recognition embeddings are clustered. Missing embeddings are
    # bridged temporally instead of being interpreted as a different child.
    if face.descriptor_source != "insightface":
        return []
    return _normalise_vector(face.descriptor)


@dataclass(slots=True)
class RefineResult:
    groups: list[list[FrameAssessment]]
    rejected_weak_series: int = 0
    merged_fragments: int = 0
    boundary_guard_splits: int = 0
    boundary_guard_proposals: int = 0


def refine_assessments_detailed(frames: list[FrameAssessment], config: dict) -> RefineResult:
    """Refine one temporal block and return diagnostics for the UI.

    The selected portrait method remains authoritative.  An optional boundary
    guard can then ask the *other* method only for suspicious internal split
    proposals.  Every proposal is independently verified against local ArcFace
    centroids before it is allowed to create an extra child series.  No image is
    re-read and InsightFace is not called again.
    """
    if not frames:
        return RefineResult([])
    algorithm = str(config.get("series", {}).get("portrait_algorithm", "dbscan")).lower()
    if algorithm == "dbscan":
        try:
            result = _refine_portrait_dbscan_v3(frames, config)
        except Exception:
            groups = _refine_portrait_v2(frames, config)
            result = RefineResult(groups)
    else:
        result = RefineResult(_refine_portrait_v2(frames, config))

    if bool(config.get("series", {}).get("portrait_boundary_guard_enabled", False)) and result.groups:
        guarded, split_count, proposal_count = _apply_portrait_boundary_guard(result.groups, config, algorithm)
        result.groups = guarded
        result.boundary_guard_splits += split_count
        result.boundary_guard_proposals += proposal_count
    return result



def _frame_descriptor(frame: FrameAssessment) -> list[float]:
    return _portrait_descriptor(frame)


def _frame_face(frame: FrameAssessment) -> FaceAssessment | None:
    return frame.primary_face


def _face_confirms_portrait(face: FaceAssessment | None, config: dict) -> bool:
    if face is None or not face.descriptor or face.descriptor_source != "insightface":
        return False
    series = config.get("series", {})
    analysis = config.get("analysis", {})
    confirm_thresh = float(series.get("portrait_confirm_det_thresh", 0.45))
    min_fraction = float(series.get("portrait_confirm_min_face_fraction", analysis.get("face_min_fraction", 0.0005)))
    return face.detection_confidence >= confirm_thresh and face.size_fraction >= min_fraction


def _segment_centroid(frames: list[FrameAssessment]) -> list[float]:
    vectors = [_frame_descriptor(frame) for frame in frames]
    vectors = [v for v in vectors if v]
    return _cluster_centroid(vectors)


def _segment_confirmed_count(frames: list[FrameAssessment], config: dict) -> int:
    return sum(1 for frame in frames if _face_confirms_portrait(_frame_face(frame), config))


def _segment_distance(left: list[FrameAssessment], right: list[FrameAssessment]) -> float:
    return _cosine_distance_vec(_segment_centroid(left), _segment_centroid(right))


def _refine_portrait_dbscan_v3(frames: list[FrameAssessment], config: dict) -> RefineResult:
    """Portrait identity clustering with temporal hysteresis.

    Detection recall is deliberately separated from *series evidence*:
    SCRFD may run with a low threshold so difficult child faces are not lost,
    but a brand-new series is accepted only after several sufficiently strong
    face detections.  Isolated false positives on walls/floors therefore do not
    become selections.

    DBSCAN remains the strict identity pass.  Adjacent fragments are then
    merged with a slightly looser centroid threshold.  This hysteresis fixes a
    common ArcFace failure mode where profile/front views of one child land in
    two neighbouring DBSCAN clusters.  A/B/A stays three chronological series
    because non-adjacent segments are never merged across a confirmed B.
    """
    if not frames:
        return RefineResult([])
    if len(frames) == 1:
        min_confirmed = max(1, int(config.get("series", {}).get("portrait_min_confirmed_frames", 2)))
        if min_confirmed <= 1 and _face_confirms_portrait(_frame_face(frames[0]), config):
            return RefineResult([frames])
        return RefineResult([], rejected_weak_series=1 if frames[0].faces else 0)

    try:
        import numpy as np
        from sklearn.cluster import DBSCAN
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("DBSCAN backend unavailable") from exc

    series_cfg = config.get("series", {})
    eps = max(0.05, min(0.80, float(series_cfg.get("portrait_dbscan_distance", 0.27))))
    min_samples = max(1, int(series_cfg.get("portrait_dbscan_min_samples", 2)))
    attach_factor = float(series_cfg.get("portrait_dbscan_noise_attach_factor", 1.12))
    attach_eps = min(0.95, eps * max(1.0, attach_factor))
    no_face_tolerance = max(0, int(series_cfg.get("portrait_no_face_tolerance", 3)))
    merge_eps = max(eps, min(0.80, float(series_cfg.get("portrait_segment_merge_distance", 0.32))))
    min_confirmed = max(1, int(series_cfg.get("portrait_min_confirmed_frames", 2)))

    descriptors: list[list[float]] = []
    descriptor_indices: list[int] = []
    expected_len: int | None = None
    for idx, frame in enumerate(frames):
        descriptor = _frame_descriptor(frame)
        if not descriptor:
            continue
        if expected_len is None:
            expected_len = len(descriptor)
        if len(descriptor) != expected_len:
            continue
        descriptors.append(descriptor)
        descriptor_indices.append(idx)

    # No reliable identity at all: do not invent a portrait series from a wall,
    # floor, blur or detector error.  A real child should have an ArcFace
    # embedding on at least one frame in the modern InsightFace backend.
    if not descriptors:
        return RefineResult([])

    x = np.asarray(descriptors, dtype=np.float32)
    labels_arr = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(x)
    desc_labels = [int(v) for v in labels_arr.tolist()]

    centroids: dict[int, list[float]] = {}
    for label in sorted(set(desc_labels)):
        if label < 0:
            continue
        members = [vec for vec, lab in zip(descriptors, desc_labels) if lab == label]
        centroids[label] = _cluster_centroid(members)

    # If a very short child series was all DBSCAN noise, allow single-point
    # identity components but validate them later with strong face evidence.
    if not centroids:
        labels_arr = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(x)
        desc_labels = [int(v) for v in labels_arr.tolist()]
        for label in sorted(set(desc_labels)):
            members = [vec for vec, lab in zip(descriptors, desc_labels) if lab == label]
            centroids[label] = _cluster_centroid(members)

    # Attach DBSCAN noise only when it is still close to a stable identity.
    # Otherwise leave it as an isolated temporary label; weak-run validation
    # below will normally discard it rather than selecting a random frame.
    next_unique = -1000
    for pos, label in enumerate(desc_labels):
        if label >= 0:
            continue
        vec = descriptors[pos]
        best_label: int | None = None
        best_dist = 99.0
        for candidate, centroid in centroids.items():
            dist = _cosine_distance_vec(vec, centroid)
            if dist < best_dist:
                best_dist = dist
                best_label = candidate
        if best_label is not None and best_dist <= attach_eps:
            desc_labels[pos] = best_label
        else:
            desc_labels[pos] = next_unique
            next_unique -= 1

    frame_labels: list[int | None] = [None] * len(frames)
    for frame_idx, label in zip(descriptor_indices, desc_labels):
        frame_labels[frame_idx] = label

    # Bridge only an *interior* detector miss where the same child is visible
    # on both sides.  Do not extend a child over leading/trailing wall/floor
    # frames and do not assign a gap between two different children to either.
    for idx, label in enumerate(frame_labels):
        if label is not None:
            continue
        prev_idx = next(
            (j for j in range(idx - 1, max(-1, idx - no_face_tolerance - 2), -1) if frame_labels[j] is not None),
            None,
        )
        next_idx = next(
            (j for j in range(idx + 1, min(len(frames), idx + no_face_tolerance + 2)) if frame_labels[j] is not None),
            None,
        )
        if prev_idx is None or next_idx is None:
            continue
        prev_label = frame_labels[prev_idx]
        next_label = frame_labels[next_idx]
        if prev_label is not None and prev_label == next_label:
            frame_labels[idx] = prev_label

    # Resolve an isolated A/B/A identity glitch only when single-frame portrait
    # series are *not* explicitly allowed. DBSCAN is intentionally stricter than
    # the sequential same-person threshold, so a strong profile view of child A
    # can occasionally receive its own one-frame label. Keep such a moderate
    # same-person view inside A, but drop a clearly different one-frame identity
    # instead of relabelling that wrong face as A (where it could compete for RED).
    if min_confirmed > 1:
        same_person_threshold = _same_person_threshold(config)
        for idx in range(1, len(frame_labels) - 1):
            if frame_labels[idx - 1] == frame_labels[idx + 1] != frame_labels[idx]:
                face = _frame_face(frames[idx])
                neighbour_faces = (_frame_face(frames[idx - 1]), _frame_face(frames[idx + 1]))
                similarities = [
                    value for value in (_descriptor_similarity(face, ref) for ref in neighbour_faces)
                    if value is not None
                ]
                if similarities and max(similarities) >= same_person_threshold:
                    frame_labels[idx] = frame_labels[idx - 1]
                else:
                    frame_labels[idx] = None

    # Build labelled chronological segments. None-only gaps are intentionally
    # omitted: they are not portrait series and therefore can never be selected.
    segments: list[list[FrameAssessment]] = []
    segment_labels: list[int] = []
    idx = 0
    while idx < len(frames):
        label = frame_labels[idx]
        if label is None:
            idx += 1
            continue
        start = idx
        idx += 1
        while idx < len(frames) and frame_labels[idx] == label:
            idx += 1
        segments.append(frames[start:idx])
        segment_labels.append(int(label))

    if not segments:
        return RefineResult([])

    merged_fragments = 0

    # First collapse adjacent DBSCAN fragments whose ArcFace centroids say they
    # are still the same child.  This is intentionally looser than DBSCAN eps.
    merged_segments: list[list[FrameAssessment]] = []
    merged_labels: list[int] = []
    for seg, label in zip(segments, segment_labels):
        if merged_segments and _segment_distance(merged_segments[-1], seg) <= merge_eps:
            merged_segments[-1].extend(seg)
            merged_fragments += 1
        else:
            merged_segments.append(list(seg))
            merged_labels.append(label)
    segments = merged_segments
    segment_labels = merged_labels

    # A short false cluster can still appear between two pieces of the same
    # child.  If the middle run lacks enough strong detections and the outer
    # centroids agree, absorb the whole A / weak / A region into one series.
    changed = True
    while changed and len(segments) >= 3:
        changed = False
        out: list[list[FrameAssessment]] = []
        i = 0
        while i < len(segments):
            if i + 2 < len(segments):
                left, middle, right = segments[i], segments[i + 1], segments[i + 2]
                middle_weak = _segment_confirmed_count(middle, config) < min_confirmed
                if middle_weak and _segment_distance(left, right) <= merge_eps:
                    # The weak middle run may be a false face (wall/mirror).
                    # Do not feed those frames to the portrait selector; join
                    # the two confirmed pieces of the child around it instead.
                    out.append(left + right)
                    merged_fragments += 2
                    i += 3
                    changed = True
                    continue
            out.append(segments[i])
            i += 1
        segments = out

    # Final evidence gate. Typical shoots contain 3–20 frames per child, so two
    # strong detections are a useful default: a one-off false face on a wall or
    # mirror no longer produces its own RED selection. Users can set this to 1
    # in the GUI for genuinely single-frame portrait sequences.
    accepted: list[list[FrameAssessment]] = []
    rejected = 0
    for seg in segments:
        if _segment_confirmed_count(seg, config) >= min_confirmed:
            accepted.append(seg)
        else:
            rejected += 1

    return RefineResult(accepted, rejected_weak_series=rejected, merged_fragments=merged_fragments)



def _boundary_guard_series_config(config: dict, algorithm: str) -> dict:
    """Shallow-copy config and switch only the portrait series algorithm."""
    result = dict(config)
    result["series"] = dict(config.get("series", {}))
    result["series"]["portrait_algorithm"] = algorithm
    # The alternate method is used only for proposals; never recursively guard.
    result["series"]["portrait_boundary_guard_enabled"] = False
    return result


def _alternate_portrait_split_proposals(
    frames: list[FrameAssessment], config: dict, main_algorithm: str
) -> list[int]:
    """Return candidate split indexes proposed by the other portrait method."""
    if len(frames) < 4:
        return []
    alternate = "sequential" if main_algorithm == "dbscan" else "dbscan"
    alt_config = _boundary_guard_series_config(config, alternate)
    try:
        if alternate == "dbscan":
            groups = _refine_portrait_dbscan_v3(frames, alt_config).groups
        else:
            groups = _refine_portrait_v2(frames, alt_config)
    except Exception:
        return []
    if len(groups) <= 1:
        return []

    index_by_id = {id(frame): idx for idx, frame in enumerate(frames)}
    proposals: list[int] = []
    for group in groups[1:]:
        if not group:
            continue
        idx = index_by_id.get(id(group[0]))
        if idx is not None and 0 < idx < len(frames):
            proposals.append(idx)
    return sorted(set(proposals))


def _nearest_descriptor_frames(
    frames: list[FrameAssessment], split_idx: int, *, left: bool, limit: int
) -> list[FrameAssessment]:
    result: list[FrameAssessment] = []
    indices = range(split_idx - 1, -1, -1) if left else range(split_idx, len(frames))
    for idx in indices:
        frame = frames[idx]
        if _frame_descriptor(frame):
            result.append(frame)
            if len(result) >= limit:
                break
    if left:
        result.reverse()
    return result


def _verify_portrait_boundary(
    frames: list[FrameAssessment], split_idx: int, config: dict
) -> tuple[bool, dict[str, float]]:
    """Verify a proposed A|B change using robust local identity evidence.

    The alternate algorithm is intentionally not trusted by itself.  A split is
    accepted only when both sides form compact local ArcFace identities and the
    frames on each side consistently prefer their own centroid over the other.
    This catches a rare merged child while rejecting an isolated bad embedding
    or a front/profile change of the same child.
    """
    cfg = config.get("series", {})
    window = max(2, int(cfg.get("portrait_boundary_guard_window", 4)))
    min_evidence = max(2, int(cfg.get("portrait_boundary_guard_min_evidence", 2)))
    min_confirmed = max(0, int(cfg.get("portrait_boundary_guard_min_confirmed_each", 1)))
    min_distance = max(0.02, min(0.80, float(cfg.get("portrait_boundary_guard_min_centroid_distance", 0.16))))
    min_margin = max(0.0, min(0.80, float(cfg.get("portrait_boundary_guard_min_identity_margin", 0.10))))
    min_vote_fraction = max(0.50, min(1.0, float(cfg.get("portrait_boundary_guard_min_vote_fraction", 0.75))))
    min_cohesion = max(0.0, min(1.0, float(cfg.get("portrait_boundary_guard_min_cohesion", 0.80))))
    vote_margin = max(0.0, min_margin * 0.40)

    left_frames = _nearest_descriptor_frames(frames, split_idx, left=True, limit=window)
    right_frames = _nearest_descriptor_frames(frames, split_idx, left=False, limit=window)
    if len(left_frames) < min_evidence or len(right_frames) < min_evidence:
        return False, {"reason": -1.0}
    if min_confirmed:
        left_confirmed = sum(_face_confirms_portrait(_frame_face(frame), config) for frame in left_frames)
        right_confirmed = sum(_face_confirms_portrait(_frame_face(frame), config) for frame in right_frames)
        if left_confirmed < min_confirmed or right_confirmed < min_confirmed:
            return False, {"reason": -2.0}

    left_vecs = [_frame_descriptor(frame) for frame in left_frames]
    right_vecs = [_frame_descriptor(frame) for frame in right_frames]
    left_centroid = _cluster_centroid(left_vecs)
    right_centroid = _cluster_centroid(right_vecs)
    if not left_centroid or not right_centroid:
        return False, {"reason": -3.0}

    cross_similarity = max(-1.0, min(1.0, cosine_similarity(left_centroid, right_centroid)))
    centroid_distance = max(0.0, min(2.0, 1.0 - cross_similarity))

    left_own = [cosine_similarity(vec, left_centroid) for vec in left_vecs]
    right_own = [cosine_similarity(vec, right_centroid) for vec in right_vecs]
    cohesion = min(statistics.median(left_own), statistics.median(right_own))

    margins: list[float] = []
    for vec in left_vecs:
        margins.append(cosine_similarity(vec, left_centroid) - cosine_similarity(vec, right_centroid))
    for vec in right_vecs:
        margins.append(cosine_similarity(vec, right_centroid) - cosine_similarity(vec, left_centroid))
    median_margin = statistics.median(margins) if margins else 0.0
    vote_fraction = sum(1 for margin in margins if margin >= vote_margin) / max(1, len(margins))

    accepted = (
        centroid_distance >= min_distance
        and cohesion >= min_cohesion
        and median_margin >= min_margin
        and vote_fraction >= min_vote_fraction
    )
    return accepted, {
        "distance": centroid_distance,
        "cohesion": cohesion,
        "margin": median_margin,
        "votes": vote_fraction,
        "left": float(len(left_frames)),
        "right": float(len(right_frames)),
    }


def _apply_portrait_boundary_guard(
    groups: list[list[FrameAssessment]], config: dict, main_algorithm: str
) -> tuple[list[list[FrameAssessment]], int, int]:
    """Split rare A+B portrait merges using the alternate method as a scout."""
    cfg = config.get("series", {})
    min_segment_frames = max(1, int(cfg.get("portrait_boundary_guard_min_segment_frames", 2)))
    result: list[list[FrameAssessment]] = []
    split_count = 0
    proposal_count = 0

    for frames in groups:
        if len(frames) < max(4, min_segment_frames * 2):
            result.append(frames)
            continue
        proposals = _alternate_portrait_split_proposals(frames, config, main_algorithm)
        proposal_count += len(proposals)
        accepted: list[int] = []
        last_boundary = 0
        for split_idx in proposals:
            # Keep enough real frames on each eventual side.  Descriptor evidence
            # is checked separately inside _verify_portrait_boundary.
            if split_idx - last_boundary < min_segment_frames:
                continue
            if len(frames) - split_idx < min_segment_frames:
                continue
            ok, metrics = _verify_portrait_boundary(frames, split_idx, config)
            if ok:
                accepted.append(split_idx)
                last_boundary = split_idx
                left_name = frames[split_idx - 1].photo.path.name
                right_name = frames[split_idx].photo.path.name
                logging.getLogger("photo_select_ai").info(
                    "PORTRAIT GUARD SPLIT %s | %s -> %s | scout=%s | distance=%.3f | cohesion=%.3f | margin=%.3f | votes=%.2f",
                    main_algorithm, left_name, right_name,
                    "sequential" if main_algorithm == "dbscan" else "dbscan",
                    metrics.get("distance", 0.0), metrics.get("cohesion", 0.0),
                    metrics.get("margin", 0.0), metrics.get("votes", 0.0),
                )

        if not accepted:
            result.append(frames)
            continue
        start = 0
        for split_idx in accepted:
            result.append(frames[start:split_idx])
            start = split_idx
            split_count += 1
        result.append(frames[start:])

    return [group for group in result if group], split_count, proposal_count


def merge_adjacent_portrait_series(
    groups: list[list[FrameAssessment]], config: dict
) -> tuple[list[list[FrameAssessment]], int]:
    """Merge adjacent same-child series across hard temporal block boundaries.

    This catches a long pause or filename gap inside one child's shoot.  It
    never jumps over another accepted series, so A / B / A remains three
    independent series as required by the workflow.
    """
    if len(groups) <= 1:
        return groups, 0
    cfg = config.get("series", {})
    max_gap = max(0.0, float(cfg.get("portrait_cross_block_merge_seconds", 45.0)))
    merge_distance = max(0.05, min(0.80, float(cfg.get("portrait_cross_block_merge_distance", 0.30))))

    merged: list[list[FrameAssessment]] = [list(groups[0])]
    count = 0
    for current in groups[1:]:
        previous = merged[-1]
        time_gap = max(
            0.0,
            (current[0].photo.capture_time - previous[-1].photo.capture_time).total_seconds(),
        )
        distance = _segment_distance(previous, current)
        guard_blocks_merge = False
        if bool(cfg.get("portrait_boundary_guard_enabled", False)):
            combined = previous + current
            guard_blocks_merge, _guard_metrics = _verify_portrait_boundary(combined, len(previous), config)
        if time_gap <= max_gap and distance <= merge_distance and not guard_blocks_merge:
            previous.extend(current)
            count += 1
        else:
            merged.append(list(current))
    return merged, count



@dataclass(slots=True)
class PortraitRepeatLinkResult:
    """Identity links between already-refined portrait series.

    ``child_ids`` is parallel to the input series list.  Equal ids mean that
    several chronological series are believed to belong to the same child.
    The original series are intentionally kept separate so selection can label
    one best RED across the child and optional YELLOW for clearly distinct poses.
    """

    child_ids: list[int]
    links: int = 0
    ambiguous_rejected: int = 0


def _repeat_pose_vectors(frames: list[FrameAssessment], config: dict, limit: int) -> list[list[float]]:
    """Return a few strong ArcFace embeddings for one portrait series."""
    ranked: list[tuple[float, list[float]]] = []
    for frame in frames:
        face = _frame_face(frame)
        vec = _frame_descriptor(frame)
        if face is None or not vec:
            continue
        # Prefer stable, large, high-quality faces without requiring the strict
        # DBSCAN confirmation gate.  A difficult side pose may be exactly the
        # repeated series that this stage is meant to recover.
        rank = (
            0.52 * max(0.0, min(1.0, float(face.detection_confidence)))
            + 0.30 * max(0.0, min(1.0, float(face.quality)))
            + 0.18 * max(0.0, min(1.0, float(face.face_sharpness)))
        )
        ranked.append((rank, vec))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [vec for _rank, vec in ranked[: max(1, limit)]]


def _repeat_pose_match_metrics(
    previous_vectors: list[list[float]],
    current_vectors: list[list[float]],
    *,
    pair_similarity: float,
) -> dict[str, float]:
    if not previous_vectors or not current_vectors:
        return {"distance": 2.0, "pair_median": -1.0, "votes": 0.0, "cohesion": 0.0}
    previous_centroid = _cluster_centroid(previous_vectors)
    current_centroid = _cluster_centroid(current_vectors)
    if not previous_centroid or not current_centroid:
        return {"distance": 2.0, "pair_median": -1.0, "votes": 0.0, "cohesion": 0.0}

    distance = _cosine_distance_vec(previous_centroid, current_centroid)
    cross = [cosine_similarity(a, b) for a in previous_vectors for b in current_vectors]
    pair_median = statistics.median(cross) if cross else -1.0
    votes = sum(1 for value in cross if value >= pair_similarity) / max(1, len(cross))
    prev_own = [cosine_similarity(v, previous_centroid) for v in previous_vectors]
    cur_own = [cosine_similarity(v, current_centroid) for v in current_vectors]
    cohesion = min(
        statistics.median(prev_own) if prev_own else 0.0,
        statistics.median(cur_own) if cur_own else 0.0,
    )
    return {
        "distance": float(distance),
        "pair_median": float(pair_median),
        "votes": float(votes),
        "cohesion": float(cohesion),
    }


def link_repeated_portrait_series(
    groups: list[list[FrameAssessment]], config: dict
) -> PortraitRepeatLinkResult:
    """Conservatively link nearby portrait series of the same child.

    This is deliberately a *post-refinement* identity stage.  DBSCAN/sequential
    series and the boundary guard stay untouched.  Only nearby chronological
    series with several mutually supporting ArcFace embeddings are linked.
    An ambiguous near-tie is rejected rather than risking two different
    children sharing one RED.
    """
    if not groups:
        return PortraitRepeatLinkResult([])

    portrait_cfg = config.get("portrait", {})
    mode = str(portrait_cfg.get("repeat_pose_mode", "off")).lower()
    if mode != "best_red_pose_yellow":
        return PortraitRepeatLinkResult(list(range(len(groups))))

    max_series_gap = max(1, int(portrait_cfg.get("repeat_pose_max_series_gap", 3)))
    max_seconds = max(0.0, float(portrait_cfg.get("repeat_pose_max_seconds", 180.0)))
    min_evidence = max(2, int(portrait_cfg.get("repeat_pose_min_evidence", 2)))
    profile_frames = max(min_evidence, min(8, int(portrait_cfg.get("repeat_pose_profile_frames", 5))))
    max_distance = max(0.05, min(0.60, float(portrait_cfg.get("repeat_pose_max_centroid_distance", 0.24))))
    min_pair_similarity = max(-1.0, min(1.0, float(portrait_cfg.get("repeat_pose_min_pair_similarity", 0.68))))
    min_vote_fraction = max(0.50, min(1.0, float(portrait_cfg.get("repeat_pose_min_vote_fraction", 0.65))))
    min_cohesion = max(0.0, min(1.0, float(portrait_cfg.get("repeat_pose_min_cohesion", 0.78))))
    min_margin = max(0.0, min(0.30, float(portrait_cfg.get("repeat_pose_min_margin", 0.035))))

    profiles = [_repeat_pose_vectors(group, config, profile_frames) for group in groups]
    child_ids: list[int] = []
    children: dict[int, list[int]] = {}
    next_child_id = 0
    links = 0
    ambiguous = 0
    log = logging.getLogger("photo_select_ai")

    for idx, frames in enumerate(groups):
        current_vectors = profiles[idx]
        if len(current_vectors) < min_evidence:
            child_id = next_child_id
            next_child_id += 1
            child_ids.append(child_id)
            children[child_id] = [idx]
            continue

        candidates: list[tuple[float, int, int, dict[str, float]]] = []
        current_start = frames[0].photo.capture_time
        for child_id, series_indices in children.items():
            last_idx = series_indices[-1]
            # ``max_series_gap=3`` means the same child may be linked across at
            # most two intervening refined series.  Typical multi-pose shoots
            # are consecutive, so this remains deliberately local.
            if idx - last_idx > max_series_gap:
                continue
            last_group = groups[last_idx]
            time_gap = max(0.0, (current_start - last_group[-1].photo.capture_time).total_seconds())
            if max_seconds > 0.0 and time_gap > max_seconds:
                continue

            # Compare against the strongest embeddings from the linked child's
            # recent poses, but keep the vector count bounded.
            previous_vectors: list[list[float]] = []
            for series_idx in reversed(series_indices):
                previous_vectors.extend(profiles[series_idx])
                if len(previous_vectors) >= profile_frames * 2:
                    break
            previous_vectors = previous_vectors[: profile_frames * 2]
            if len(previous_vectors) < min_evidence:
                continue
            metrics = _repeat_pose_match_metrics(
                previous_vectors, current_vectors, pair_similarity=min_pair_similarity
            )
            accepted = (
                metrics["distance"] <= max_distance
                and metrics["pair_median"] >= min_pair_similarity
                and metrics["votes"] >= min_vote_fraction
                and metrics["cohesion"] >= min_cohesion
            )
            if accepted:
                candidates.append((metrics["distance"], child_id, last_idx, metrics))

        candidates.sort(key=lambda item: item[0])
        chosen: tuple[float, int, int, dict[str, float]] | None = None
        if candidates:
            if len(candidates) >= 2 and candidates[1][0] - candidates[0][0] < min_margin:
                ambiguous += 1
                best = candidates[0]
                second = candidates[1]
                log.info(
                    "PORTRAIT REPEAT AMBIGUOUS | series=%d | best_child=%d distance=%.3f | second_child=%d distance=%.3f",
                    idx + 1, best[1] + 1, best[0], second[1] + 1, second[0],
                )
            else:
                chosen = candidates[0]

        if chosen is None:
            child_id = next_child_id
            next_child_id += 1
            child_ids.append(child_id)
            children[child_id] = [idx]
            continue

        distance, child_id, last_idx, metrics = chosen
        child_ids.append(child_id)
        children[child_id].append(idx)
        links += 1
        log.info(
            "PORTRAIT REPEAT LINK | series=%d -> child=%d | previous_series=%d | distance=%.3f | pair=%.3f | votes=%.2f | cohesion=%.3f",
            idx + 1, child_id + 1, last_idx + 1, distance, metrics["pair_median"],
            metrics["votes"], metrics["cohesion"],
        )

    return PortraitRepeatLinkResult(child_ids, links=links, ambiguous_rejected=ambiguous)


@dataclass(slots=True)
class PortraitPoseSignature:
    """Conservative frame/cluster-level portrait pose signature.

    It deliberately uses only cues already produced by InsightFace: head yaw /
    pitch plus face position and scale in the frame.  This is not a full-body
    pose detector; uncertain signatures are therefore allowed to collapse into
    the same pose instead of producing a false YELLOW.
    """

    yaw_deg: float
    pitch_deg: float
    center_x: float
    center_y: float
    size_fraction: float
    head_confidence: float
    samples: int


def portrait_pose_signature(frames: list[FrameAssessment]) -> PortraitPoseSignature | None:
    samples: list[tuple[float, float, float, float, float, float]] = []
    for frame in frames:
        if frame.error:
            continue
        face = _frame_face(frame)
        if face is None:
            continue
        samples.append(
            (
                float(face.head_yaw_deg),
                float(face.head_pitch_deg),
                float(face.center[0]),
                float(face.center[1]),
                max(1e-8, float(face.size_fraction)),
                max(0.0, min(1.0, float(face.head_pose_confidence))),
            )
        )
    if not samples:
        return None

    # Head angles are meaningful only when solvePnP itself is reasonably
    # supported.  Median geometry remains useful as a secondary composition
    # cue even when some angle samples are weak.
    confident = [row for row in samples if row[5] >= 0.18]
    angle_rows = confident or samples
    return PortraitPoseSignature(
        yaw_deg=float(statistics.median(row[0] for row in angle_rows)),
        pitch_deg=float(statistics.median(row[1] for row in angle_rows)),
        center_x=float(statistics.median(row[2] for row in samples)),
        center_y=float(statistics.median(row[3] for row in samples)),
        size_fraction=float(statistics.median(row[4] for row in samples)),
        head_confidence=float(statistics.median(row[5] for row in angle_rows)),
        samples=len(samples),
    )


def portrait_pose_difference(
    a: PortraitPoseSignature | None,
    b: PortraitPoseSignature | None,
    config: dict,
) -> tuple[bool, dict[str, float]]:
    """Return whether two linked series are *clearly* different portrait poses.

    A single small movement never creates a pose.  A strong head turn/tilt is
    sufficient by itself.  Pure composition changes require two strong cues,
    or one strong composition cue plus a meaningful head change.  This bias is
    intentional: missing a YELLOW is safer than labelling a nearly identical
    pose as different.
    """
    if a is None or b is None:
        return False, {
            "score": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "center": 0.0,
            "scale": 0.0,
        }

    cfg = config.get("portrait", {})
    min_head_conf = max(0.0, min(1.0, float(cfg.get("repeat_pose_min_head_confidence", 0.30))))
    yaw_threshold = max(5.0, float(cfg.get("repeat_pose_min_yaw_delta_deg", 20.0)))
    pitch_threshold = max(5.0, float(cfg.get("repeat_pose_min_pitch_delta_deg", 16.0)))
    center_threshold = max(0.02, float(cfg.get("repeat_pose_min_center_shift", 0.11)))
    scale_fraction = max(0.05, float(cfg.get("repeat_pose_min_scale_change", 0.32)))
    scale_threshold = math.log1p(scale_fraction)

    head_reliable = min(a.head_confidence, b.head_confidence) >= min_head_conf
    yaw_delta = abs(a.yaw_deg - b.yaw_deg) if head_reliable else 0.0
    pitch_delta = abs(a.pitch_deg - b.pitch_deg) if head_reliable else 0.0
    center_shift = ((a.center_x - b.center_x) ** 2 + (a.center_y - b.center_y) ** 2) ** 0.5
    scale_delta = abs(math.log(max(1e-8, b.size_fraction) / max(1e-8, a.size_fraction)))

    yaw_n = yaw_delta / yaw_threshold
    pitch_n = pitch_delta / pitch_threshold
    center_n = center_shift / center_threshold
    scale_n = scale_delta / scale_threshold
    head_n = max(yaw_n, pitch_n)
    composition_n = max(center_n, scale_n)

    strong_head = head_reliable and head_n >= 1.0
    # Framing/zoom alone is never enough: it may be the photographer moving,
    # not the child changing pose. A composition change can only support a
    # meaningful (but sub-threshold) head change.
    composition_plus_head = composition_n >= 1.0 and head_reliable and head_n >= 0.60
    distinct = bool(strong_head or composition_plus_head)

    # Diagnostic score only; the boolean gate above remains authoritative.
    score = max(
        head_n if head_reliable else 0.0,
        min(center_n, scale_n),
        min(composition_n, head_n / 0.60) if head_reliable else 0.0,
    )
    return distinct, {
        "score": float(score),
        "yaw": float(yaw_delta),
        "pitch": float(pitch_delta),
        "center": float(center_shift),
        "scale": float(math.expm1(scale_delta)),
    }
