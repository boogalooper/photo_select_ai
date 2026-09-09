from __future__ import annotations

from app.core.models import FrameAssessment, Selection


def _portrait_weights(config: dict) -> dict[str, float]:
    p = config["portrait"]
    return {
        "eyes": max(0.0, float(p.get("eyes_weight", 1.8))),
        "eye_sharpness": max(0.0, float(p.get("eye_sharpness_weight", 1.4))),
        "face_sharpness": max(0.0, float(p.get("face_sharpness_weight", 0.9))),
        "expression": max(0.0, float(p.get("expression_weight", 0.7))),
        "smile": max(0.0, float(p.get("smile_weight", 0.35))),
        "technical": max(0.0, float(p.get("technical_weight", 0.35))),
    }


def portrait_score(frame: FrameAssessment, config: dict) -> float:
    face = frame.primary_face
    if not face:
        return -1.0
    p = config["portrait"]
    weights = _portrait_weights(config)

    # Unknown 106-landmark state stays neutral rather than becoming a fake
    # closed-eye result. Detection/recognition is still valid in this case.
    eyes = face.eyes_open_score if face.landmarks_reliable else 0.50
    smile = face.smile if face.landmarks_reliable else 0.50
    expression = face.expression if face.landmarks_reliable else 0.50
    eye_sharp = face.eye_sharpness if face.landmarks_reliable else face.face_sharpness * 0.90

    components = {
        "eyes": eyes,
        "eye_sharpness": eye_sharp,
        "face_sharpness": face.face_sharpness,
        "expression": expression,
        "smile": smile,
        "technical": frame.technical,
    }
    total_weight = sum(weights.values())
    if total_weight <= 1e-9:
        score = face.quality
    else:
        score = sum(weights[name] * components[name] for name in weights) / total_weight

    eye_threshold = float(config["analysis"]["eye_open_threshold"])
    if face.landmarks_reliable and (
        face.eye_open_left < eye_threshold or face.eye_open_right < eye_threshold
    ):
        score -= max(0.0, float(p.get("closed_eye_penalty", 0.45)))
    return score


def _portrait_preference_enabled(config: dict) -> bool:
    return bool(config.get("portrait", {}).get("portrait_preference_enabled", True))


def _portrait_defect_key(frame: FrameAssessment, config: dict) -> tuple[int, int, int, int, int, int, int, int]:
    """Return a Portrait suitability tier with a protected uncertain zone.

    Portrait mode intentionally allows head rotation. Clearly closed eyes,
    definite blur and poor technical quality are hard defects. Measurements
    that miss an eye/sharpness threshold only by a small margin are marked as
    uncertain instead of immediately becoming hard rejects. This preserves a
    good borderline take as fallback without letting it outrank a clearly clean
    frame. Unknown landmark state is treated as uncertainty for the same reason.
    """
    face = frame.primary_face
    if face is None or frame.error:
        return (99, 99, 99, 99, 99, 99, 99, 99)

    p = config.get("portrait", {})
    eye_threshold = float(config.get("analysis", {}).get("eye_open_threshold", 0.52))
    eye_uncertain_margin = max(0.0, min(0.25, float(p.get("eye_uncertain_margin", 0.04))))
    min_eye_sharp = max(0.0, min(1.0, float(p.get("min_eye_sharpness", 0.40))))
    eye_sharp_margin = max(0.0, min(0.25, float(p.get("eye_sharpness_uncertain_margin", 0.05))))
    min_face_sharp = max(0.0, min(1.0, float(p.get("min_face_sharpness", 0.34))))
    face_sharp_margin = max(0.0, min(0.25, float(p.get("face_sharpness_uncertain_margin", 0.04))))
    min_face_technical = max(0.0, min(1.0, float(p.get("min_face_technical_quality", 0.30))))
    min_frame_technical = max(0.0, min(1.0, float(p.get("min_frame_technical_quality", 0.28))))

    hard_closed = hard_blur = poor_quality = 0
    uncertain_eyes = uncertain_blur = unknown = 0

    if not face.landmarks_reliable:
        unknown = 1
    else:
        eyes = face.eyes_open_score
        if eyes < eye_threshold - eye_uncertain_margin:
            hard_closed = 1
        elif eyes < eye_threshold:
            uncertain_eyes = 1

    face_sharp = float(face.face_sharpness)
    if face_sharp < min_face_sharp - face_sharp_margin:
        hard_blur = 1
    elif face_sharp < min_face_sharp:
        uncertain_blur = 1

    if face.landmarks_reliable:
        eye_sharp = float(face.eye_sharpness)
        if eye_sharp < min_eye_sharp - eye_sharp_margin:
            hard_blur = 1
        elif eye_sharp < min_eye_sharp:
            uncertain_blur = 1

    if face.technical < min_face_technical or frame.technical < min_frame_technical:
        poor_quality = 1

    hard_total = hard_closed + hard_blur + poor_quality
    uncertain_total = uncertain_eyes + uncertain_blur + unknown
    return (
        hard_total,
        hard_closed,
        hard_blur,
        poor_quality,
        uncertain_total,
        uncertain_eyes,
        uncertain_blur,
        unknown,
    )


def _candidate_pool(frames: list[FrameAssessment], config: dict) -> list[FrameAssessment]:
    """Choose the best hard-suitability tier before FBP ranking.

    Facial Beauty Prediction must never compensate for a clear closed eye,
    definite blur or poor technical quality. Borderline eye/sharpness readings
    form a separate uncertain tier: a clearly clean frame always wins first,
    while an uncertain frame remains available when no cleaner take exists.
    FBP is evaluated only after that suitability tier has been fixed. Head
    rotation is deliberately not a Portrait defect.
    """
    valid = [f for f in frames if f.primary_face is not None and not f.error]
    if not valid:
        return []
    keyed = [(f, _portrait_defect_key(f, config)) for f in valid]
    best_key = min(key for _frame, key in keyed)
    return [frame for frame, key in keyed if key == best_key]

def _portrait_preference_usable(pool: list[FrameAssessment], config: dict) -> bool:
    if not _portrait_preference_enabled(config) or len(pool) <= 1:
        return False
    known = sum(
        1 for frame in pool
        if frame.primary_face is not None and frame.primary_face.portrait_preference_reliable
    )
    min_fraction = max(
        0.0,
        min(1.0, float(config.get("portrait", {}).get("portrait_preference_min_known_fraction", 0.50))),
    )
    return known >= 2 and (known / float(len(pool))) >= min_fraction


def _selection_key(
    frame: FrameAssessment,
    config: dict,
    use_preference: bool,
) -> tuple[float, float, float, float]:
    face = frame.primary_face
    assert face is not None
    legacy = portrait_score(frame, config)
    if use_preference:
        # Do not compare an FBP score and the legacy fallback on the same
        # numeric axis.  Once FBP is usable for the current suitability tier,
        # candidates with a reliable FBP result form the preferred sub-pool.
        # Legacy scoring is only a tie-breaker inside that sub-pool; frames
        # without a reliable FBP result can win only when FBP is not usable for
        # the tier as a whole.
        if face.portrait_preference_reliable:
            return (
                1.0,
                float(face.portrait_preference_score),
                legacy,
                face.quality,
            )
        return (0.0, -1.0, legacy, face.quality)
    return (
        0.0,
        legacy,
        face.quality,
        frame.technical,
    )


def sort_portrait_selections(selections: list[Selection], config: dict) -> list[Selection]:
    """Sort already-selected portrait candidates on one consistent scale.

    This is used when several distinct pose clusters compete for a limited
    number of YELLOW labels. Per-cluster FBP availability can differ, so each
    selected pose is normalized onto one consistent comparison scale.
    """
    items = list(selections)
    if len(items) <= 1:
        return items
    known = [item for item in items if item.preference_score is not None]
    min_fraction = max(
        0.0,
        min(1.0, float(config.get("portrait", {}).get("portrait_preference_min_known_fraction", 0.50))),
    )
    use_preference = (
        _portrait_preference_enabled(config)
        and len(known) >= 2
        and (len(known) / float(len(items))) >= min_fraction
    )
    if use_preference:
        return sorted(
            items,
            key=lambda item: (
                1.0 if item.preference_score is not None else 0.0,
                float(item.preference_score) if item.preference_score is not None else -1.0,
                float(item.legacy_score) if item.legacy_score is not None else float(item.score),
            ),
            reverse=True,
        )
    return sorted(
        items,
        key=lambda item: float(item.legacy_score) if item.legacy_score is not None else float(item.score),
        reverse=True,
    )


def select_portrait(frames: list[FrameAssessment], config: dict) -> Selection | None:
    pool = _candidate_pool(frames, config)
    if not pool:
        return None

    use_preference = _portrait_preference_usable(pool, config)
    best = max(pool, key=lambda f: _selection_key(f, config, use_preference))
    face = best.primary_face
    assert face is not None
    detail = "reliable" if face.landmarks_reliable else "unknown"
    pref_detail = (
        f"pref={face.portrait_preference_score:.2f} raw={face.portrait_preference_raw:.2f}"
        if face.portrait_preference_reliable
        else "pref=unknown"
    )
    pref_mode = "FBP-primary" if use_preference else "legacy-fallback"
    defect_key = _portrait_defect_key(best, config)
    legacy_score = portrait_score(best, config)
    reliable_pref_score = (
        float(face.portrait_preference_score) if face.portrait_preference_reliable else None
    )
    return Selection(
        photo=best.photo,
        label_role="red",
        score=(
            reliable_pref_score
            if use_preference and reliable_pref_score is not None
            else legacy_score
        ),
        reason=(
            f"{pref_mode}; {pref_detail}; legacy={legacy_score:.3f}; "
            f"eyes={face.eyes_open_score:.2f}; smile={face.smile:.2f}; expression={face.expression:.2f}; "
            f"eye_sharp={face.eye_sharpness:.2f}; face_sharp={face.face_sharpness:.2f}; "
            f"technical={best.technical:.2f}; suitability={defect_key}; details={detail}"
        ),
        preference_score=reliable_pref_score,
        legacy_score=legacy_score,
    )
