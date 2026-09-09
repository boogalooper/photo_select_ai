from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


class PortraitPreferenceScorer:
    """Generic portrait-preference prior backed by a public beauty model.

    The first backend is the compact ResNet-18 Caffe beauty model distributed
    with asiryan/HowCuteAmI; that upstream project references the SCUT-FBP5500
    facial-beauty benchmark. The rest of Photo Select AI intentionally sees a
    generic ``portrait preference`` interface so a future personalised ranker
    trained from Group Face Picker choices can replace or blend this backend.

    Group mode normalises the output within each ``PersonTrack`` so absolute
    model scores of different people are never compared against each other.
    Portrait mode uses the same backend inside one portrait series / linked
    child, where comparing frames of the same person is the intended use.
    """

    MODEL_NAME = "beauty_resnet.caffemodel"
    PROTO_NAME = "beauty_resnet.prototxt"
    SOURCE_NAME = "HowCuteAmI beauty ResNet-18 / SCUT-FBP5500 reference"

    def __init__(self, model_dir: Path, config: dict, section: str = "group"):
        self.section_name = str(section or "group").lower()
        self.mode_cfg = config.get(self.section_name, {})
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / self.MODEL_NAME
        self.proto_path = self.model_dir / self.PROTO_NAME

        missing = [
            path.name
            for path in (self.model_path, self.proto_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Portrait preference model is incomplete: missing "
                + ", ".join(missing)
                + ". Run install.bat to restore it."
            )

        try:
            self.net = cv2.dnn.readNetFromCaffe(str(self.proto_path), str(self.model_path))
            # Keep the small auxiliary model on CPU so it does not compete with
            # InsightFace/ONNX Runtime for CUDA memory.
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load portrait preference model from {self.model_dir}: {exc}"
            ) from exc

        self.min_face_px = max(
            24, int(self.mode_cfg.get("portrait_preference_min_face_px", 48))
        )
        self.padding = max(
            0.0,
            min(0.30, float(self.mode_cfg.get("portrait_preference_crop_padding", 0.0))),
        )
        self.raw_min = float(self.mode_cfg.get("portrait_preference_raw_min", 1.0))
        self.raw_max = float(self.mode_cfg.get("portrait_preference_raw_max", 5.0))
        if self.raw_max <= self.raw_min:
            self.raw_min, self.raw_max = 1.0, 5.0

        # readNetFromCaffe can succeed for some damaged/incompatible files and
        # fail only on the first forward pass. Probe the fixed-size network once
        # during construction so InsightFaceAnalyzer can treat a corrupt FBP
        # backend as unavailable *before* any shoot frames are ranked. This
        # avoids a mixed partial-FBP run.
        try:
            probe = np.full((224, 224, 3), 128, dtype=np.uint8)
            normalized, raw, reliable = self.score(probe, (0, 0, 224, 224))
            if not reliable or not math.isfinite(raw) or not (0.0 <= normalized <= 1.0):
                raise RuntimeError(
                    f"invalid probe output: raw={raw!r}, normalized={normalized!r}, reliable={reliable!r}"
                )
        except Exception as exc:
            raise RuntimeError(f"Portrait preference model failed startup inference: {exc}") from exc

    @property
    def backend_name(self) -> str:
        return self.SOURCE_NAME

    def score(
        self,
        rgb: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[float, float, bool]:
        """Return normalized score, raw model score and reliability flag.

        The upstream demo feeds a square BGR 224x224 face crop to the Caffe
        network with scalefactor 1/255, mean=(104,117,123), swapRB=False.
        A face that is too small is neutral/unknown instead of allowing a
        heavily upscaled crop to influence RED selection.
        """
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
            return 0.50, 0.0, False

        x1, y1, x2, y2 = bbox
        if min(x2 - x1, y2 - y1) < self.min_face_px:
            return 0.50, 0.0, False

        crop = _square_crop(rgb, bbox, self.padding)
        if crop.size == 0 or min(crop.shape[:2]) < self.min_face_px:
            return 0.50, 0.0, False

        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        interpolation = cv2.INTER_AREA if max(bgr.shape[:2]) >= 224 else cv2.INTER_CUBIC
        face = cv2.resize(bgr, (224, 224), interpolation=interpolation)
        blob = cv2.dnn.blobFromImage(
            face,
            scalefactor=1.0 / 255.0,
            size=(224, 224),
            mean=(104.0, 117.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        output = np.asarray(self.net.forward(), dtype=np.float32).reshape(-1)
        if output.size == 0:
            raise RuntimeError("Portrait preference model returned an empty output")
        raw = float(output[0])
        if not math.isfinite(raw):
            raise RuntimeError(f"Portrait preference model returned a non-finite score: {raw}")

        normalized = (raw - self.raw_min) / (self.raw_max - self.raw_min)
        normalized = max(0.0, min(1.0, normalized))
        return normalized, raw, True


def _square_crop(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: float,
) -> np.ndarray:
    """Square a detector box around its centre and clip it to the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(1.0, float(max(x2 - x1, y2 - y1))) * (1.0 + 2.0 * padding)
    sx1 = max(0, int(math.floor(cx - side * 0.5)))
    sy1 = max(0, int(math.floor(cy - side * 0.5)))
    sx2 = min(w, int(math.ceil(cx + side * 0.5)))
    sy2 = min(h, int(math.ceil(cy + side * 0.5)))
    if sx2 <= sx1 or sy2 <= sy1:
        return image[0:0, 0:0]
    return image[sy1:sy2, sx1:sx2]


def _self_test() -> None:
    """Load the installed network and execute one deterministic forward pass."""
    from app.paths import ROOT

    scorer = PortraitPreferenceScorer(
        ROOT / "models" / "portrait_preference",
        {"group": {"portrait_preference_min_face_px": 24}},
        section="group",
    )
    sample = np.full((224, 224, 3), 128, dtype=np.uint8)
    normalized, raw, reliable = scorer.score(sample, (0, 0, 224, 224))
    if not reliable or not math.isfinite(raw) or not (0.0 <= normalized <= 1.0):
        raise RuntimeError(
            f"Portrait preference self-test returned invalid values: raw={raw!r}, normalized={normalized!r}"
        )
    print(f"Portrait preference model OK: raw={raw:.4f}, normalized={normalized:.4f}")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--self-test"]:
        _self_test()
    else:
        raise SystemExit("Usage: python -m app.analysis.portrait_preference --self-test")
