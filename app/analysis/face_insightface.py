from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.core.models import FaceAssessment, FrameAssessment, PhotoFile
from app.utils.cuda_runtime import cuda_runtime_versions, prepare_windows_cuda_dlls
from .quality import clamp01, sharpness_score, technical_quality
from .portrait_preference import PortraitPreferenceScorer

# InsightFace 2d106det layout used by the official model pack.
# See InsightFace-compatible consumers: right eye 33:43, mouth 52:72,
# left eye 87:97.  All portrait face-state features come from this same model pack.
_RIGHT_EYE_106 = slice(33, 43)
_MOUTH_106 = slice(52, 72)
_LEFT_EYE_106 = slice(87, 97)


class InsightFaceAnalyzer:
    """InsightFace analyzer for portrait and group workflows.

    Full-frame face detection, identity embeddings and 106-point landmarks all
    come from the InsightFace buffalo_l pack.  Eye openness, smile proxy,
    expression proxy and eye sharpness are derived from those landmarks.

    CUDA is attempted first when requested.  If ONNX Runtime/cuDNN fails while
    executing a frame (not merely while creating the session), the analyzer can
    rebuild itself on CPU and retry that SAME frame.  This prevents a GPU plan
    error from turning into a missing child/series.
    """

    def __init__(
        self,
        model_root: Path,
        config: dict,
        message: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.log = logging.getLogger("photo_select_ai")
        self.message = message or (lambda _m: None)
        self.model_root = Path(model_root)
        self.model_pack = str(config["analysis"].get("insightface_model_pack", "buffalo_l"))
        self.mode = str(config.get("runtime", {}).get("mode", "portrait")).lower()
        det_key = "insightface_det_size_group" if self.mode == "group" else "insightface_det_size_portrait"
        faces_key = "max_faces_group" if self.mode == "group" else "max_faces_portrait"
        self.det_size = _round_det_size(int(config["analysis"].get(det_key, 1024 if self.mode == "group" else 640)))
        det_thresh_key = "insightface_det_thresh_group" if self.mode == "group" else "insightface_det_thresh"
        self.det_thresh = float(config["analysis"].get(det_thresh_key, config["analysis"].get("insightface_det_thresh", 0.25)))
        self.max_faces = int(config["analysis"].get(faces_key, 80 if self.mode == "group" else 6))
        self.portrait_preference = None
        preference_section = self.mode if self.mode in {"portrait", "group"} else "group"
        if bool(config.get(preference_section, {}).get("portrait_preference_enabled", self.mode == "group")):
            # FBP is an optional ranking enhancement at runtime. install.bat
            # still installs and validates it, but a later missing/corrupt model
            # must never block the core portrait/group workflow.
            preference_root = self.model_root.parent / "portrait_preference"
            try:
                self.portrait_preference = PortraitPreferenceScorer(
                    preference_root, config, section=preference_section
                )
            except Exception as exc:
                self.portrait_preference = None
                self.log.warning(
                    "Portrait preference model unavailable; using legacy ranking fallback: %s", exc
                )
                self.message(
                    "Facial Beauty Prediction недоступен — используется резервный рейтинг качества. "
                    "Основной анализ продолжен; при необходимости запустите install.bat для восстановления модели."
                )
        self._cpu_fallback_allowed = bool(config["analysis"].get("insightface_cuda_fallback_cpu", True))
        self._cpu_fallback_used = False
        self._requested_provider = str(config["analysis"].get("insightface_provider", "auto")).lower()
        self._ort = self._load_ort()
        self.available_providers = list(self._ort.get_available_providers())
        self.app = None
        self.providers = []
        try:
            self._activate(self._requested_provider)
        except Exception as exc:
            # CUDA failures can happen while InsightFace creates/prepares ORT
            # sessions, before the first app.get(). Recover here as well.
            if (
                self._cpu_fallback_allowed
                and self._requested_provider != "cpu"
                and "CUDAExecutionProvider" in self.available_providers
                and _looks_like_cuda_failure(exc)
            ):
                self._cpu_fallback_used = True
                self.log.error("InsightFace CUDA initialization failed; switching to CPU: %s", exc)
                self.message("CUDA/cuDNN не удалось инициализировать. InsightFace будет запущен на CPU.")
                self._activate("cpu")
            else:
                raise
        if self._cpu_fallback_allowed and _provider_has_gpu(self.providers):
            self._validate_gpu_or_fallback()

    def _load_ort(self):
        try:
            # cuDNN 9 can load NVRTC/cublasLt sublibraries lazily during the
            # first convolution. NVIDIA's pip wheels keep those DLLs in
            # separate site-packages\nvidia\*\bin directories, so expose them
            # to Windows before ONNX Runtime creates any CUDA sessions.
            dll_info = prepare_windows_cuda_dlls()
            import onnxruntime as ort
            if hasattr(ort, "preload_dlls"):
                # ORT 1.21+ preloads its known CUDA/cuDNN DLLs from NVIDIA
                # site-packages. Our helper additionally exposes NVRTC and
                # nvJitLink, which cuDNN may LoadLibrary() at graph execution.
                ort.preload_dlls(directory="")
            versions = cuda_runtime_versions()
            if versions:
                self.log.info("CUDA runtime packages: %s", ", ".join(f"{k}={v}" for k, v in versions.items()))
            if dll_info.get("configured"):
                self.log.info(
                    "Windows NVIDIA DLL search configured: %d dirs; preloaded: %s",
                    len(dll_info.get("directories", [])),
                    ", ".join(dll_info.get("loaded", [])) or "none",
                )
            return ort
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"ONNX Runtime is unavailable: {exc}") from exc

    def _activate(self, requested: str) -> None:
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("InsightFace is not installed. Run install.bat again.") from exc

        expected_dir = self.model_root / "models" / self.model_pack
        required = ("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx")
        missing = [name for name in required if not (expected_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"InsightFace model pack '{self.model_pack}' is incomplete: missing {', '.join(missing)}. "
                "Run install.bat again."
            )

        runtime_cfg = self.config.get("runtime", {})
        safe_vram = bool(runtime_cfg.get("gpu_memory_safe_mode", True))
        try:
            mem_limit_gb = float(runtime_cfg.get("gpu_session_mem_limit_gb", 6.0))
        except (TypeError, ValueError):
            mem_limit_gb = 6.0
        mem_limit_bytes = None
        if safe_vram and mem_limit_gb > 0:
            # FaceAnalysis owns three CUDA ORT sessions in this build
            # (detection, recognition, landmark_2d_106). ORT's gpu_mem_limit is
            # per *model session*, so divide the user-facing budget for one
            # InsightFace worker across those three arenas. This is not a hard
            # process-wide cap (CUDA context/model allocations also exist), but
            # it prevents any single arena from ballooning toward all VRAM.
            mem_limit_bytes = int(mem_limit_gb * (1024 ** 3) / 3.0)

        providers = _provider_order(
            self.available_providers,
            requested,
            str(self.config["analysis"].get("insightface_cuda_conv_algo", "HEURISTIC")),
            gpu_mem_limit_bytes=mem_limit_bytes,
            conservative_arena=safe_vram,
        )
        if not providers:
            raise RuntimeError(f"No usable ONNX Runtime provider. Available: {self.available_providers}")

        # 2d106det is included in buffalo_l and gives enough geometry for eyes,
        # mouth and eye-local sharpness from the same InsightFace model pack.
        app = FaceAnalysis(
            name=self.model_pack,
            root=str(self.model_root),
            allowed_modules=["detection", "recognition", "landmark_2d_106"],
            providers=providers,
        )
        gpu = _provider_has_gpu(providers)
        app.prepare(
            ctx_id=0 if gpu else -1,
            det_thresh=self.det_thresh,
            det_size=(self.det_size, self.det_size),
        )
        self.app = app
        self.providers = providers


    def _validate_gpu_or_fallback(self) -> None:
        """Execute every CUDA model once before touching user photos.

        A blank detector probe alone is not enough: if no face is detected,
        InsightFace never runs ArcFace recognition or the 106-landmark model.
        The RTX 5090 failure reported in practice occurred on the 112x112
        recognition Conv_0, so explicitly exercise every auxiliary ORT session.
        """
        probe = np.zeros((self.det_size, self.det_size, 3), dtype=np.uint8)
        try:
            try:
                self.app.get(probe, max_num=0)
            except TypeError:
                self.app.get(probe)

            for task, model in getattr(self.app, "models", {}).items():
                if task == "detection":
                    continue
                session = getattr(model, "session", None)
                if session is None:
                    continue
                input_meta = session.get_inputs()[0]
                shape = _concrete_probe_shape(input_meta.shape, getattr(model, "input_size", None))
                dtype = np.float16 if "float16" in str(getattr(input_meta, "type", "")).lower() else np.float32
                session.run(None, {input_meta.name: np.zeros(shape, dtype=dtype)})
                self.log.info("CUDA self-test passed for InsightFace model: %s shape=%s", task, shape)
        except Exception as exc:
            if self._should_fallback_cpu(exc):
                self._fallback_to_cpu(exc)
            else:
                raise

    @property
    def backend_name(self) -> str:
        first = self.providers[0] if self.providers else "unknown"
        if isinstance(first, tuple):
            first = first[0]
        suffix = " → CPU fallback" if self._cpu_fallback_used else ""
        preference = " + FBP-ResNet18" if self.portrait_preference is not None else ""
        return f"InsightFace/SCRFD+106 ({first}){suffix}{preference}"

    def _release_app(self) -> None:
        """Drop all known references to InsightFace/ORT sessions.

        FaceAnalysis keeps both ``models`` and ``det_model`` references. Merely
        assigning ``self.app = None`` can therefore leave a reference cycle alive
        until a later GC pass, which is especially visible with several CUDA
        sessions. Clearing both references makes the CUDA arenas eligible for
        release as soon as the stage/pool is closed.
        """
        app = self.app
        self.app = None
        if app is None:
            return
        try:
            if hasattr(app, "det_model"):
                app.det_model = None
        except Exception:
            pass
        try:
            models = getattr(app, "models", None)
            if isinstance(models, dict):
                models.clear()
        except Exception:
            pass
        del app

    def close(self) -> None:
        self._release_app()
        # A pool is normally collected immediately by CPython, but explicit GC
        # is cheap at stage boundaries and prevents delayed ORT-session cycles
        # from retaining many gigabytes of CUDA arena memory.
        gc.collect()

    def analyze(self, photo: PhotoFile, rgb: np.ndarray) -> FrameAssessment:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"`estimate` is deprecated.*",
                    category=FutureWarning,
                    module=r"insightface\.utils\.face_align",
                )
                return self._analyze_once(photo, rgb)
        except Exception as exc:
            if self._should_fallback_cpu(exc):
                self._fallback_to_cpu(exc)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"`estimate` is deprecated.*",
                        category=FutureWarning,
                        module=r"insightface\.utils\.face_align",
                    )
                    return self._analyze_once(photo, rgb)
            raise

    def _should_fallback_cpu(self, exc: Exception) -> bool:
        if not self._cpu_fallback_allowed or self._cpu_fallback_used:
            return False
        if not _provider_has_gpu(self.providers):
            return False
        return _looks_like_cuda_failure(exc)

    def _fallback_to_cpu(self, exc: Exception) -> None:
        self.log.error("InsightFace GPU execution failed; switching to CPU and retrying current frame: %s", exc)
        self.message(
            "Сбой CUDA/cuDNN в InsightFace. Переключаю анализ лиц на CPU и повторяю текущий кадр; "
            "кадры не будут пропущены."
        )
        self._release_app()
        gc.collect()
        self._cpu_fallback_used = True
        self._activate("cpu")

    def _analyze_once(self, photo: PhotoFile, rgb: np.ndarray) -> FrameAssessment:
        h, w = rgb.shape[:2]
        frame_technical = technical_quality(rgb)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        try:
            detected = self.app.get(bgr, max_num=0)
        except TypeError:  # API compatibility
            detected = self.app.get(bgr)

        if self.mode == "group":
            min_fraction = float(
                self.config.get("group", {}).get(
                    "min_track_face_fraction",
                    self.config["analysis"].get("face_min_fraction", 0.0005),
                )
            )
        else:
            min_fraction = float(self.config["analysis"].get("face_min_fraction", 0.0005))
        faces: list[FaceAssessment] = []

        for detected_face in detected or []:
            bbox = np.asarray(getattr(detected_face, "bbox", []), dtype=np.float32).reshape(-1)
            if bbox.size < 4:
                continue
            x1 = max(0, min(w - 1, int(np.floor(bbox[0]))))
            y1 = max(0, min(h - 1, int(np.floor(bbox[1]))))
            x2 = max(x1 + 1, min(w, int(np.ceil(bbox[2]))))
            y2 = max(y1 + 1, min(h, int(np.ceil(bbox[3]))))
            bw, bh = x2 - x1, y2 - y1
            fraction = (bw * bh) / float(max(1, w * h))
            if fraction < min_fraction:
                continue

            descriptor = _normalised_embedding(
                getattr(detected_face, "normed_embedding", None)
                if getattr(detected_face, "normed_embedding", None) is not None
                else getattr(detected_face, "embedding", None)
            )
            face_rgb = rgb[y1:y2, x1:x2]
            face_sharp = sharpness_score(face_rgb)
            tech = technical_quality(face_rgb)
            det_score = float(getattr(detected_face, "det_score", 1.0) or 0.0)
            if self.portrait_preference is not None:
                # The scorer has already passed a startup forward-pass probe.
                # If a later inference nevertheless fails, propagate it rather
                # than mixing FBP and fallback rankings inside one shoot.
                pref_score, pref_raw, pref_reliable = self.portrait_preference.score(
                    rgb, (x1, y1, x2, y2)
                )
            else:
                pref_score, pref_raw, pref_reliable = 0.50, 0.0, False

            landmarks = _landmarks_106(detected_face)
            landmarks_ok = landmarks is not None and _fine_state_landmarks_reliable(
                landmarks, (x1, y1, x2, y2), det_score, self.config, self.mode
            )
            if landmarks_ok:
                eye_left, eye_right, eye_sharp = _eye_metrics(rgb, landmarks, self.config)
                smile, expression = _mouth_expression_metrics(landmarks, (x1, y1, x2, y2))
                reliable = True
                # Head pose is a first-class suitability signal in Group mode
                # and a pose-description cue in Portrait mode. Estimate it for
                # both modes from the already available 106 landmarks.
                head_frontal, head_pose_conf, head_yaw, head_pitch = _head_pose_metrics(
                    landmarks, rgb.shape, self.config
                )
                if self.mode == "group" and bool(self.config.get("group", {}).get("camera_attention_enabled", False)):
                    attention, attention_conf, head_frontal = _camera_attention_metrics(
                        rgb, landmarks, (x1, y1, x2, y2), eye_left, eye_right, self.config,
                        head_score=head_frontal, head_conf=head_pose_conf,
                    )
                    attention_reliable = attention_conf >= float(
                        self.config.get("group", {}).get("camera_attention_min_confidence", 0.38)
                    )
                else:
                    attention, attention_conf, attention_reliable = 0.50, 0.0, False
            else:
                # Detection/identity stays usable. Fine state on a tiny or weakly
                # detected group face is unknown, never interpreted as a blink.
                eye_left = eye_right = 0.50
                eye_sharp = clamp01(face_sharp * 0.90)
                smile = 0.50
                expression = 0.50
                reliable = False
                attention, attention_conf, head_frontal, attention_reliable = 0.50, 0.0, 0.50, False
                head_pose_conf, head_yaw, head_pitch = 0.0, 0.0, 0.0

            eyes = min(eye_left, eye_right)
            quality = clamp01(
                0.25 * face_sharp
                + 0.25 * eye_sharp
                + 0.18 * expression
                + 0.17 * eyes
                + 0.08 * smile
                + 0.07 * tech
            )
            faces.append(
                FaceAssessment(
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h)),
                    size_fraction=fraction,
                    eye_open_left=eye_left,
                    eye_open_right=eye_right,
                    smile=smile,
                    expression=expression,
                    face_sharpness=face_sharp,
                    eye_sharpness=eye_sharp,
                    technical=tech,
                    quality=quality,
                    portrait_preference_score=pref_score,
                    portrait_preference_raw=pref_raw,
                    portrait_preference_reliable=pref_reliable,
                    descriptor=descriptor,
                    landmarks_reliable=reliable,
                    detection_confidence=det_score,
                    descriptor_source="insightface",
                    camera_attention_score=attention,
                    camera_attention_reliable=attention_reliable,
                    head_yaw_deg=head_yaw,
                    head_pitch_deg=head_pitch,
                    head_pose_confidence=head_pose_conf,
                )
            )

        faces.sort(key=lambda f: f.size_fraction, reverse=True)
        if self.max_faces > 0:
            faces = faces[: self.max_faces]
        return FrameAssessment(photo=photo, faces=faces, technical=frame_technical)



def _fine_state_landmarks_reliable(
    landmarks: np.ndarray,
    bbox: tuple[int, int, int, int],
    det_score: float,
    config: dict,
    mode: str,
) -> bool:
    """Conservative reliability gate for eye/expression measurements.

    2d106det can always return coordinates, including on very small faces where
    the eye state is not actually supported by enough source pixels.  Identity
    recognition remains usable in that case, but fine-state features should be
    marked unknown rather than becoming false closed-eye evidence.
    """
    if mode != "group":
        return True
    cfg = config.get("group", {})
    min_det = float(cfg.get("eye_state_min_det_confidence", 0.35))
    min_face_px = max(24.0, float(cfg.get("eye_state_min_face_px", 56.0)))
    if det_score < min_det:
        return False
    x1, y1, x2, y2 = bbox
    if min(x2 - x1, y2 - y1) < min_face_px:
        return False

    for eye_slice in (_RIGHT_EYE_106, _LEFT_EYE_106):
        pts = np.asarray(landmarks[eye_slice], dtype=np.float32)
        if pts.shape[0] < 4:
            return False
        centered = pts - pts.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            axes = centered @ vt.T
            major = float(np.ptp(axes[:, 0]))
        except np.linalg.LinAlgError:
            major = float(np.ptp(pts[:, 0]))
        if major < 3.0:
            return False
    return True

def _landmarks_106(face) -> np.ndarray | None:
    value = getattr(face, "landmark_2d_106", None)
    if value is None:
        return None
    pts = np.asarray(value, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 106 or pts.shape[1] < 2:
        return None
    if not np.all(np.isfinite(pts[:, :2])):
        return None
    return pts[:, :2]


def _eye_metrics(rgb: np.ndarray, landmarks: np.ndarray, config: dict) -> tuple[float, float, float]:
    right_pts = landmarks[_RIGHT_EYE_106]
    left_pts = landmarks[_LEFT_EYE_106]
    closed_ratio = float(config["analysis"].get("eye_closed_ratio", 0.075))
    open_ratio = float(config["analysis"].get("eye_open_ratio", 0.285))
    if open_ratio <= closed_ratio + 0.01:
        open_ratio = closed_ratio + 0.01

    right_open = _eye_open_score(right_pts, closed_ratio, open_ratio)
    left_open = _eye_open_score(left_pts, closed_ratio, open_ratio)
    sharp_scores = [_landmark_region_sharpness(rgb, right_pts, 0.45), _landmark_region_sharpness(rgb, left_pts, 0.45)]
    eye_sharp = float(sum(sharp_scores) / len(sharp_scores)) if sharp_scores else 0.0
    return left_open, right_open, eye_sharp


def _eye_open_score(points: np.ndarray, closed_ratio: float, open_ratio: float) -> float:
    """Rotation-invariant eye opening from the 106-point contour."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 4:
        return 0.50
    centered = pts - pts.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axes = centered @ vt.T
        major = float(np.ptp(axes[:, 0]))
        minor = float(np.ptp(axes[:, 1]))
    except np.linalg.LinAlgError:
        major = float(np.ptp(pts[:, 0]))
        minor = float(np.ptp(pts[:, 1]))
    if major < 1.0:
        return 0.50
    ratio = minor / major
    return clamp01((ratio - closed_ratio) / (open_ratio - closed_ratio))


def _landmark_region_sharpness(rgb: np.ndarray, points: np.ndarray, padding: float) -> float:
    h, w = rgb.shape[:2]
    xs = points[:, 0]
    ys = points[:, 1]
    span_x = max(2.0, float(xs.max() - xs.min()))
    span_y = max(2.0, float(ys.max() - ys.min()))
    pad_x = span_x * padding
    pad_y = max(span_y * (padding + 0.35), span_x * 0.12)
    x1 = max(0, int(np.floor(xs.min() - pad_x)))
    x2 = min(w, int(np.ceil(xs.max() + pad_x + 1)))
    y1 = max(0, int(np.floor(ys.min() - pad_y)))
    y2 = min(h, int(np.ceil(ys.max() + pad_y + 1)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return sharpness_score(rgb[y1:y2, x1:x2])




def _linear_preference(value: float, good: float, bad: float) -> float:
    """1 at/below ``good`` and 0 at/above ``bad`` for an absolute error."""
    value = abs(float(value))
    good = max(0.0, float(good))
    bad = max(good + 1e-6, float(bad))
    return clamp01((bad - value) / (bad - good))


def _head_pose_metrics(
    landmarks: np.ndarray,
    image_shape: tuple[int, ...],
    config: dict,
) -> tuple[float, float, float, float]:
    """Approximate head frontalness plus pitch/yaw from 106 landmarks.

    The 3-D template is generic, so angles are intentionally treated as soft
    portrait-pose cues rather than exact biometric measurements.  Confidence
    is reduced when reprojection error is large.
    """
    if landmarks.shape[0] < 106:
        return 0.50, 0.0, 0.0, 0.0
    try:
        # 106 -> conventional 68 correspondence used by InsightFace consumers:
        # chin=0, nose tip=86, eye outer corners=35/93, mouth corners=52/61.
        eye_pair = sorted((landmarks[35], landmarks[93]), key=lambda p: float(p[0]))
        mouth_pair = sorted((landmarks[52], landmarks[61]), key=lambda p: float(p[0]))
        image_points = np.asarray(
            [landmarks[86], landmarks[0], eye_pair[0], eye_pair[1], mouth_pair[0], mouth_pair[1]],
            dtype=np.float64,
        )
        model_points = np.asarray(
            [
                (0.0, 0.0, 0.0),
                (0.0, 330.0, 65.0),
                (-225.0, -170.0, 135.0),
                (225.0, -170.0, 135.0),
                (-150.0, 150.0, 125.0),
                (150.0, 150.0, 125.0),
            ],
            dtype=np.float64,
        )
        h, w = image_shape[:2]
        # Camera coordinates: Y points down, Z away from the camera.
        # The former Y-up/Z-forward template made a frontal head require
        # about 180 degrees of pitch, which was incorrectly treated as a turn.
        focal = float(max(w, h))
        camera = np.asarray(
            [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.50, 0.0, 0.0, 0.0
        rotation, _ = cv2.Rodrigues(rvec)
        depths = (rotation @ model_points.T + tvec.reshape(3, 1))[2]
        if not np.all(np.isfinite(depths)) or np.any(depths <= 0):
            return 0.50, 0.0, 0.0, 0.0
        angles = cv2.RQDecomp3x3(rotation)[0]
        pitch, yaw = float(angles[0]), float(angles[1])

        cfg = config.get("group", {})
        yaw_score = _linear_preference(
            yaw,
            float(cfg.get("camera_attention_head_yaw_good_deg", 10.0)),
            float(cfg.get("camera_attention_head_yaw_bad_deg", 30.0)),
        )
        pitch_score = _linear_preference(
            pitch,
            float(cfg.get("camera_attention_head_pitch_good_deg", 12.0)),
            float(cfg.get("camera_attention_head_pitch_bad_deg", 30.0)),
        )
        frontal = 0.62 * yaw_score + 0.38 * pitch_score

        projected, _ = cv2.projectPoints(model_points, rvec, tvec, camera, dist)
        projected = projected.reshape(-1, 2)
        reproj = float(np.mean(np.linalg.norm(projected - image_points, axis=1)))
        eye_span = max(1.0, float(np.linalg.norm(eye_pair[1] - eye_pair[0])))
        rel_error = reproj / eye_span
        confidence = clamp01((0.32 - rel_error) / 0.24)
        return clamp01(frontal), confidence, yaw, pitch
    except (cv2.error, ValueError, IndexError, np.linalg.LinAlgError):
        return 0.50, 0.0, 0.0, 0.0


def _head_frontal_metrics(
    landmarks: np.ndarray,
    image_shape: tuple[int, ...],
    config: dict,
) -> tuple[float, float]:
    """Backward-compatible frontalness wrapper used by group gaze logic."""
    frontal, confidence, _yaw, _pitch = _head_pose_metrics(landmarks, image_shape, config)
    return frontal, confidence

def _eye_dark_center_offset(
    rgb: np.ndarray,
    points: np.ndarray,
    min_eye_px: float,
) -> tuple[float, float, float] | None:
    """Estimate dark iris/pupil displacement inside one eye contour.

    Returns ``(|horizontal offset|, |vertical offset|, confidence)`` in a local
    eye coordinate system where roughly 1.0 means the edge of the eye.  It is a
    conservative image cue, not a semantic iris detector; callers must ignore
    it when confidence is low.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 4:
        return None
    center = pts.mean(axis=0)
    centered = pts - center
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    major = vt[0].astype(np.float32)
    minor = vt[1].astype(np.float32)
    major /= max(1e-6, float(np.linalg.norm(major)))
    minor /= max(1e-6, float(np.linalg.norm(minor)))
    proj_u = centered @ major
    proj_v = centered @ minor
    width = float(np.ptp(proj_u))
    height = float(np.ptp(proj_v))
    if width < min_eye_px or height < 2.5:
        return None

    pad = max(2, int(round(0.18 * width)))
    x1 = max(0, int(np.floor(float(np.min(pts[:, 0])))) - pad)
    y1 = max(0, int(np.floor(float(np.min(pts[:, 1])))) - pad)
    x2 = min(rgb.shape[1], int(np.ceil(float(np.max(pts[:, 0])))) + pad + 1)
    y2 = min(rgb.shape[0], int(np.ceil(float(np.max(pts[:, 1])))) + pad + 1)
    if x2 - x1 < 5 or y2 - y1 < 4:
        return None

    crop = rgb[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_pts = np.round(pts - np.asarray([x1, y1], dtype=np.float32)).astype(np.int32)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    hull = cv2.convexHull(local_pts)
    cv2.fillConvexPoly(mask, hull, 255)
    if min(gray.shape) >= 5:
        k = 3 if height >= 5.0 else 1
        if k > 1:
            mask = cv2.erode(mask, np.ones((k, k), dtype=np.uint8), iterations=1)

    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return None
    global_xy = np.column_stack((xs + x1, ys + y1)).astype(np.float32)
    local = global_xy - center
    uu = local @ major
    vv = local @ minor
    half_u = max(1e-6, width * 0.5)
    half_v = max(1e-6, height * 0.5)
    inner = (np.abs(uu) <= 0.78 * half_u) & (np.abs(vv) <= 0.88 * half_v)
    if int(np.count_nonzero(inner)) < 8:
        return None
    uu = uu[inner]
    vv = vv[inner]
    values = gray[ys[inner], xs[inner]]

    p10 = float(np.percentile(values, 10))
    p45 = float(np.percentile(values, 45))
    p75 = float(np.percentile(values, 75))
    contrast = max(0.0, (p75 - p10) / 255.0)
    darkness = np.maximum(0.0, p45 - values)
    # Eyelashes/lids are often dark near the vertical contour; favour the eye
    # centre without hard-coding a pupil shape.
    vertical_prior = np.exp(-0.5 * (vv / max(1e-6, 0.55 * half_v)) ** 2)
    horizontal_prior = np.exp(-0.5 * (uu / max(1e-6, 0.95 * half_u)) ** 2)
    weights = darkness * vertical_prior * (0.65 + 0.35 * horizontal_prior)
    total = float(np.sum(weights))
    if total <= 1e-4:
        return None
    pupil_u = float(np.sum(weights * uu) / total)
    pupil_v = float(np.sum(weights * vv) / total)
    offset_u = abs(pupil_u) / half_u
    offset_v = abs(pupil_v) / half_v
    size_conf = clamp01((width - min_eye_px) / max(6.0, min_eye_px * 0.75))
    contrast_conf = clamp01((contrast - 0.035) / 0.16)
    confidence = clamp01(0.45 * size_conf + 0.55 * contrast_conf)
    return float(offset_u), float(offset_v), confidence


def _camera_attention_metrics(
    rgb: np.ndarray,
    landmarks: np.ndarray,
    bbox: tuple[int, int, int, int],
    eye_left_open: float,
    eye_right_open: float,
    config: dict,
    *,
    head_score: float | None = None,
    head_conf: float | None = None,
) -> tuple[float, float, float]:
    """Soft estimate that a group subject is looking toward the camera.

    Combines a generic head-pose estimate with the dark iris/pupil centre inside
    both eye contours.  The metric is only considered reliable when source eye
    detail and local contrast support the measurement.
    """
    cfg = config.get("group", {})
    min_eye_px = float(cfg.get("camera_attention_min_eye_px", 14.0))
    min_open = float(cfg.get("camera_attention_min_eye_open", 0.38))
    if head_score is None or head_conf is None:
        head_score, head_conf = _head_frontal_metrics(landmarks, rgb.shape, config)

    eye_results: list[tuple[float, float, float]] = []
    if eye_right_open >= min_open:
        result = _eye_dark_center_offset(rgb, landmarks[_RIGHT_EYE_106], min_eye_px)
        if result is not None:
            eye_results.append(result)
    if eye_left_open >= min_open:
        result = _eye_dark_center_offset(rgb, landmarks[_LEFT_EYE_106], min_eye_px)
        if result is not None:
            eye_results.append(result)

    if not eye_results:
        return 0.50, 0.0, head_score

    horiz = float(np.mean([r[0] for r in eye_results]))
    vert = float(np.mean([r[1] for r in eye_results]))
    eye_conf = float(np.mean([r[2] for r in eye_results]))
    if len(eye_results) == 1:
        eye_conf *= 0.72

    hscore = _linear_preference(
        horiz,
        float(cfg.get("camera_attention_pupil_good", 0.10)),
        float(cfg.get("camera_attention_pupil_bad", 0.44)),
    )
    vscore = _linear_preference(
        vert,
        float(cfg.get("camera_attention_vertical_good", 0.14)),
        float(cfg.get("camera_attention_vertical_bad", 0.62)),
    )
    eye_score = 0.82 * hscore + 0.18 * vscore

    if head_conf > 0.05:
        score = 0.38 * head_score + 0.62 * eye_score
        confidence = eye_conf * (0.55 + 0.45 * head_conf)
    else:
        score = eye_score
        confidence = eye_conf * 0.72

    # Very small group faces can technically pass the landmark gate but still
    # have too little source detail for gaze.  Scale confidence by bbox size.
    x1, y1, x2, y2 = bbox
    face_px = float(min(x2 - x1, y2 - y1))
    face_scale = clamp01((face_px - 52.0) / 52.0)
    confidence *= 0.55 + 0.45 * face_scale
    return clamp01(score), clamp01(confidence), clamp01(head_score)


def _mouth_expression_metrics(landmarks: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Geometric smile/expression proxies from InsightFace mouth landmarks.

    This intentionally stays conservative. It is a ranking feature, not an
    emotion classifier: users can set its weight to zero in the GUI.
    """
    mouth = np.asarray(landmarks[_MOUTH_106], dtype=np.float32)
    if mouth.shape[0] < 8:
        return 0.50, 0.50
    x1, y1, x2, y2 = bbox
    face_w = max(1.0, float(x2 - x1))
    face_h = max(1.0, float(y2 - y1))
    width = float(np.ptp(mouth[:, 0]))
    height = float(np.ptp(mouth[:, 1]))
    if width < 1.0:
        return 0.50, 0.50

    # Extreme-x landmarks approximate mouth corners without relying on an exact
    # semantic point index. A smile tends to widen the mouth and lift corners.
    left_band = mouth[np.argsort(mouth[:, 0])[:3]]
    right_band = mouth[np.argsort(mouth[:, 0])[-3:]]
    corners_y = float((left_band[:, 1].mean() + right_band[:, 1].mean()) * 0.5)
    center_band = mouth[np.argsort(np.abs(mouth[:, 0] - mouth[:, 0].mean()))[:6]]
    center_y = float(center_band[:, 1].mean())
    curve = (center_y - corners_y) / max(1.0, height)
    width_norm = width / face_w

    width_score = clamp01((width_norm - 0.28) / 0.20)
    curve_score = clamp01((curve + 0.05) / 0.30)
    smile = clamp01(0.60 * width_score + 0.40 * curve_score)

    # Penalise very large mouth opening/asymmetry as a generic "awkward
    # expression" signal while keeping neutral and smiling faces competitive.
    mouth_open = height / face_h
    vertical_penalty = clamp01((mouth_open - 0.18) / 0.18)
    left_y = float(left_band[:, 1].mean())
    right_y = float(right_band[:, 1].mean())
    asymmetry = abs(left_y - right_y) / max(1.0, height)
    asym_penalty = clamp01((asymmetry - 0.12) / 0.45)
    expression = clamp01(0.62 + 0.25 * smile - 0.18 * vertical_penalty - 0.12 * asym_penalty)
    return smile, expression


def _normalised_embedding(value) -> list[float]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return []
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return []
    return [float(v) for v in (arr / norm)]



def _concrete_probe_shape(raw_shape, input_size=None) -> tuple[int, ...]:
    values = list(raw_shape or [])
    if not values:
        return (1, 3, 112, 112)
    width = height = None
    if isinstance(input_size, (list, tuple)) and len(input_size) >= 2:
        try:
            width, height = int(input_size[0]), int(input_size[1])
        except Exception:
            width = height = None
    result: list[int] = []
    for index, dim in enumerate(values):
        if isinstance(dim, int) and dim > 0:
            result.append(dim)
        elif index == 0:
            result.append(1)
        elif index == 1:
            result.append(3)
        elif index == 2:
            result.append(height or 112)
        elif index == 3:
            result.append(width or 112)
        else:
            result.append(1)
    return tuple(result)

def _round_det_size(value: int) -> int:
    return max(320, min(2048, int(round(value / 32.0) * 32)))



def _looks_like_cuda_failure(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".upper()
    markers = (
        "CUDNN",
        "CUDA",
        "GRAPH_EXECUTION_FAILED",
        "EP_FAIL",
        "NO KERNEL IMAGE",
        "INVALIDPTX",
        "ONNXRUNTIMEERROR",
    )
    return any(marker in text for marker in markers)

def _provider_has_gpu(providers) -> bool:
    for item in providers or []:
        name = item[0] if isinstance(item, tuple) else item
        if name in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
            return True
    return False


def _provider_order(
    available: list[str],
    requested: str,
    conv_algo: str = "HEURISTIC",
    *,
    gpu_mem_limit_bytes: int | None = None,
    conservative_arena: bool = False,
):
    requested = (requested or "auto").lower()
    conv_algo = str(conv_algo or "HEURISTIC").upper()
    if conv_algo not in {"DEFAULT", "HEURISTIC", "EXHAUSTIVE"}:
        conv_algo = "HEURISTIC"

    cpu = "CPUExecutionProvider" in available
    cuda = "CUDAExecutionProvider" in available
    cuda_options = {
        "device_id": "0",
        "cudnn_conv_algo_search": conv_algo,
        "do_copy_in_default_stream": "1",
        "cudnn_conv_use_max_workspace": "0",
    }
    if gpu_mem_limit_bytes is not None and int(gpu_mem_limit_bytes) > 0:
        cuda_options["gpu_mem_limit"] = str(int(gpu_mem_limit_bytes))
    if conservative_arena:
        # Avoid the default power-of-two arena growth, which can reserve much
        # more VRAM than the current inference actually needs.
        cuda_options["arena_extend_strategy"] = "kSameAsRequested"
    cuda_entry = ("CUDAExecutionProvider", cuda_options)

    if requested == "cpu":
        return ["CPUExecutionProvider"] if cpu else []
    if requested == "cuda":
        # Keep CUDA strict at ORT-session level. If CUDA itself fails, our
        # analyzer catches the exception once and rebuilds the entire
        # FaceAnalysis object on CPU. This avoids InsightFace creating several
        # sessions that each print the same CUDA->CPU fallback error.
        return [cuda_entry] if cuda else []
    # auto: prefer CUDA. CPU is selected only when CUDA is unavailable; runtime
    # CUDA failures are handled once by the analyzer-level fallback.
    if cuda:
        return [cuda_entry]
    return ["CPUExecutionProvider"] if cpu else []
