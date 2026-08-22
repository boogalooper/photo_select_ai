from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the project root.
_BOOT_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOT_ROOT))

from app.paths import ROOT
from app.utils.cuda_runtime import cuda_runtime_versions, prepare_windows_cuda_dlls


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return "NVIDIA GPU (nvidia-smi details unavailable)"


def _run_model(ort, model: Path, shape: tuple[int, int, int, int]) -> None:
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": "0",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
                "cudnn_conv_use_max_workspace": "0",
            },
        )
    ]
    session = ort.InferenceSession(str(model), providers=providers)
    inp = session.get_inputs()[0]
    session.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})


def main() -> int:
    print("CUDA self-test for Photo Select AI")
    print("GPU:", _gpu_name())
    info = prepare_windows_cuda_dlls()
    print("NVIDIA DLL directories:", len(info.get("directories", [])))
    print("Explicitly preloaded:", ", ".join(info.get("loaded", [])) or "none")

    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")

    versions = cuda_runtime_versions()
    for name, version in versions.items():
        print(f"{name}: {version}")
    print("ORT providers:", ort.get_available_providers())

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("CUDA self-test FAILED: CUDAExecutionProvider is unavailable.")
        return 2

    pack = ROOT / "models" / "insightface" / "models" / "buffalo_l"
    tests = (
        ("recognition", pack / "w600k_r50.onnx", (1, 3, 112, 112)),
        ("landmarks", pack / "2d106det.onnx", (1, 3, 192, 192)),
        ("detector", pack / "det_10g.onnx", (1, 3, 640, 640)),
    )
    try:
        for label, model, shape in tests:
            if not model.is_file():
                print(f"CUDA self-test FAILED: missing model {model}")
                return 3
            print(f"Testing {label} on CUDA: {model.name} {shape} ...")
            _run_model(ort, model, shape)
            print(f"  PASS {label}")
    except Exception as exc:
        print("CUDA self-test FAILED:")
        print(type(exc).__name__ + ":", exc)
        return 4

    print("CUDA self-test PASSED: detector, recognition and landmarks executed on GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
