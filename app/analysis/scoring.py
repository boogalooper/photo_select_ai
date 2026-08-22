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


def select_portrait(frames: list[FrameAssessment], config: dict) -> Selection | None:
    valid = [f for f in frames if f.primary_face is not None and not f.error]
    if not valid:
        return None

    threshold = float(config["analysis"]["eye_open_threshold"])
    prefer_open = bool(config["portrait"].get("prefer_open_eyes", True))
    if prefer_open:
        reliably_open = [
            f for f in valid
            if f.primary_face
            and f.primary_face.landmarks_reliable
            and f.primary_face.eye_open_left >= threshold
            and f.primary_face.eye_open_right >= threshold
        ]
        pool = reliably_open or valid
    else:
        pool = valid

    best = max(pool, key=lambda f: portrait_score(f, config))
    face = best.primary_face
    assert face is not None
    detail = "reliable" if face.landmarks_reliable else "unknown"
    return Selection(
        photo=best.photo,
        label_role="red",
        score=portrait_score(best, config),
        reason=(
            f"eyes={face.eyes_open_score:.2f}; smile={face.smile:.2f}; expression={face.expression:.2f}; "
            f"eye_sharp={face.eye_sharpness:.2f}; face_sharp={face.face_sharpness:.2f}; "
            f"technical={best.technical:.2f}; details={detail}"
        ),
    )
