from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_BOOT_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_ROOT))

from app import __version__
from app.analysis.portrait_preference import PortraitPreferenceScorer
from app.paths import ROOT

MODEL_ROOT = ROOT / "models"
INSIGHTFACE_PACK = MODEL_ROOT / "insightface" / "models" / "buffalo_l"
PREFERENCE_DIR = MODEL_ROOT / "portrait_preference"
MANIFEST_PATH = MODEL_ROOT / "model_manifest.json"

# install.bat deliberately downloads and validates the complete official pack.
INSTALL_BUFFALO_MODELS: dict[str, tuple[int, int, int, int]] = {
    "det_10g.onnx": (1, 3, 640, 640),
    "w600k_r50.onnx": (1, 3, 112, 112),
    "2d106det.onnx": (1, 3, 192, 192),
    "1k3d68.onnx": (1, 3, 192, 192),
    "genderage.onnx": (1, 3, 96, 96),
}

# Normal application runtime uses only detection, ArcFace recognition and
# 2d106 landmarks. Missing unused buffalo_l extras must not force an internet
# recovery or prevent an otherwise valid offline launch.
RUNTIME_BUFFALO_MODELS: dict[str, tuple[int, int, int, int]] = {
    name: INSTALL_BUFFALO_MODELS[name]
    for name in ("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx")
}
PREFERENCE_FILES = ("beauty_resnet.caffemodel", "beauty_resnet.prototxt")


def _paths_for(models: dict[str, tuple[int, int, int, int]], include_preference: bool) -> list[Path]:
    paths = [INSIGHTFACE_PACK / name for name in models]
    if include_preference:
        paths.extend(PREFERENCE_DIR / name for name in PREFERENCE_FILES)
    return paths


def _minimum_size(path: Path) -> int:
    name = path.name.lower()
    if name == "beauty_resnet.caffemodel":
        return 40_000_000
    if name == "beauty_resnet.prototxt":
        return 10_000
    return 100_000


def _basic_file_check(paths: list[Path]) -> tuple[bool, str]:
    problems: list[str] = []
    for path in paths:
        if not path.is_file():
            problems.append(f"missing: {path.relative_to(ROOT)}")
            continue
        size = path.stat().st_size
        if size < _minimum_size(path):
            problems.append(f"too small: {path.relative_to(ROOT)} ({size} bytes)")
    if problems:
        return False, "; ".join(problems)
    return True, "required model files are present"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_files_for_manifest() -> list[Path]:
    files = [p for p in INSIGHTFACE_PACK.glob("*.onnx") if p.is_file()]
    files += [PREFERENCE_DIR / name for name in PREFERENCE_FILES if (PREFERENCE_DIR / name).is_file()]
    source_integrity = PREFERENCE_DIR / "source_integrity.json"
    if source_integrity.is_file():
        files.append(source_integrity)
    return sorted(files, key=lambda p: str(p.relative_to(ROOT)).casefold())


def _write_manifest() -> None:
    files = []
    for path in _model_files_for_manifest():
        files.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        })
    value = {
        "schema": 2,
        "photo_select_ai_version": __version__,
        "files": files,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(MANIFEST_PATH)
    print(f"Model manifest written: {MANIFEST_PATH.relative_to(ROOT)} ({len(files)} files)")


def _verify_manifest(required_paths: list[Path]) -> tuple[bool, str]:
    if not MANIFEST_PATH.is_file():
        return False, "model manifest is missing"
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entries = data.get("files", [])
        if not isinstance(entries, list) or not entries:
            return False, "model manifest has no files"
        by_path = {
            str(item.get("path", "")): item
            for item in entries
            if isinstance(item, dict)
        }
        expected = [str(p.relative_to(ROOT)).replace("\\", "/") for p in required_paths]
        missing = [rel for rel in expected if rel not in by_path]
        if missing:
            return False, "manifest does not cover: " + ", ".join(sorted(missing))
        for rel in expected:
            item = by_path[rel]
            path = ROOT / Path(rel)
            if not path.is_file():
                return False, f"manifest file is missing: {rel}"
            if int(item.get("size", -1)) != path.stat().st_size:
                return False, f"model size changed: {rel}"
        # Startup intentionally does not hash hundreds of MB. install.bat/full
        # validation executes models and writes SHA-256 for all installed files.
    except Exception as exc:
        return False, f"could not read/verify model manifest: {exc}"
    return True, "manifest paths and sizes are valid"


def _run_onnx_cpu_models(models: dict[str, tuple[int, int, int, int]]) -> None:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    for name, fallback_shape in models.items():
        path = INSIGHTFACE_PACK / name
        print(f"Testing InsightFace model on CPU: {name} ...")
        session = ort.InferenceSession(str(path), providers=providers)
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError(f"{name}: model has no inputs")
        input_meta = inputs[0]
        shape: list[int] = []
        for idx, dim in enumerate(input_meta.shape):
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                shape.append(fallback_shape[idx] if idx < len(fallback_shape) else 1)
        dtype = np.float16 if "float16" in str(input_meta.type).lower() else np.float32
        session.run(None, {input_meta.name: np.zeros(tuple(shape), dtype=dtype)})
        print(f"  PASS {name}")


def _run_preference_cpu_model() -> None:
    print("Testing portrait-preference model on CPU ...")
    scorer = PortraitPreferenceScorer(
        PREFERENCE_DIR,
        {"group": {"portrait_preference_min_face_px": 24}},
        section="group",
    )
    sample = np.full((224, 224, 3), 128, dtype=np.uint8)
    normalized, raw, reliable = scorer.score(sample, (0, 0, 224, 224))
    if not reliable or not np.isfinite(raw) or not (0.0 <= normalized <= 1.0):
        raise RuntimeError(
            f"portrait preference returned invalid values: raw={raw}, normalized={normalized}"
        )
    print(f"  PASS portrait preference: raw={raw:.4f}, normalized={normalized:.4f}")


def _optional_preference_validation() -> None:
    paths = [PREFERENCE_DIR / name for name in PREFERENCE_FILES]
    if not all(path.is_file() and path.stat().st_size >= _minimum_size(path) for path in paths):
        print("  WARNING portrait preference model unavailable; runtime will use legacy ranking fallback.")
        return
    try:
        _run_preference_cpu_model()
    except Exception as exc:
        print(
            "  WARNING portrait preference model failed validation; runtime will use legacy ranking fallback:",
            type(exc).__name__ + ":", exc,
        )


def full_install_validation(write_manifest: bool) -> int:
    required = _paths_for(INSTALL_BUFFALO_MODELS, include_preference=True)
    ok, message = _basic_file_check(required)
    if not ok:
        print("MODEL CHECK FAILED:", message)
        return 2
    try:
        from insightface.app import FaceAnalysis  # noqa: F401
        _run_onnx_cpu_models(INSTALL_BUFFALO_MODELS)
        _run_preference_cpu_model()
    except Exception as exc:
        print("MODEL CHECK FAILED:", type(exc).__name__ + ":", exc)
        return 3
    if write_manifest:
        _write_manifest()
    print("MODEL CHECK PASSED: complete installed model set is loadable.")
    return 0


def runtime_full_validation(write_manifest: bool) -> int:
    required = _paths_for(RUNTIME_BUFFALO_MODELS, include_preference=False)
    ok, message = _basic_file_check(required)
    if not ok:
        print("RUNTIME MODEL CHECK FAILED:", message)
        return 2
    try:
        from insightface.app import FaceAnalysis  # noqa: F401
        _run_onnx_cpu_models(RUNTIME_BUFFALO_MODELS)
    except Exception as exc:
        print("RUNTIME MODEL CHECK FAILED:", type(exc).__name__ + ":", exc)
        return 3
    _optional_preference_validation()
    if write_manifest:
        _write_manifest()
    print("RUNTIME MODEL CHECK PASSED: required InsightFace models are loadable.")
    return 0


def files_only_validation(runtime: bool) -> int:
    models = RUNTIME_BUFFALO_MODELS if runtime else INSTALL_BUFFALO_MODELS
    include_preference = not runtime
    required = _paths_for(models, include_preference=include_preference)
    ok, message = _basic_file_check(required)
    if not ok:
        print("MODEL FILE CHECK FAILED:", message)
        return 2
    ok, message = _verify_manifest(required)
    if not ok:
        print("MODEL FILE CHECK NEEDS FALLBACK VALIDATION:", message)
        return 3
    print("MODEL FILE CHECK PASSED:", message)
    if runtime:
        optional_ok, optional_message = _basic_file_check(
            _paths_for({}, include_preference=True)
        )
        if not optional_ok:
            print("MODEL FILE WARNING: FBP optional at runtime:", optional_message)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--files-only", action="store_true", help="installer/full-set fast check")
    mode.add_argument("--full", action="store_true", help="installer/full-set CPU validation")
    mode.add_argument("--runtime-files", action="store_true", help="fast check of runtime-required models")
    mode.add_argument("--runtime-full", action="store_true", help="CPU check of runtime-required models")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.files_only:
        return files_only_validation(runtime=False)
    if args.runtime_files:
        return files_only_validation(runtime=True)
    if args.runtime_full:
        return runtime_full_validation(write_manifest=bool(args.write_manifest))
    return full_install_validation(write_manifest=bool(args.write_manifest))


if __name__ == "__main__":
    raise SystemExit(main())
