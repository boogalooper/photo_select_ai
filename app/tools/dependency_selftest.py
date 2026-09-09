from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


# import name -> human-readable package name
REQUIRED = (
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python"),
    ("rawpy", "rawpy"),
    ("exifread", "ExifRead"),
    ("psutil", "psutil"),
    ("onnx", "onnx"),
    ("scipy", "scipy"),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("requests", "requests"),
    ("onnxruntime", "ONNX Runtime"),
    ("insightface", "InsightFace"),
)

PINNED_DISTRIBUTIONS = {
    "numpy": "1.26.4",
    "Pillow": "10.4.0",
    "opencv-python": "4.10.0.84",
    "rawpy": "0.27.0",
    "ExifRead": "3.0.0",
    "psutil": "6.0.0",
    "onnx": "1.17.0",
    "scipy": "1.13.1",
    "scikit-image": "0.24.0",
    "scikit-learn": "1.5.2",
    "tqdm": "4.66.5",
    "requests": "2.32.3",
    "protobuf": "7.36.1",
    "flatbuffers": "25.12.19",
    "networkx": "3.6.1",
    "imageio": "2.37.4",
    "tifffile": "2026.3.3",
    "packaging": "26.3",
    "lazy-loader": "0.5",
    "joblib": "1.6.0",
    "threadpoolctl": "3.6.0",
    "cloudpickle": "3.1.2",
    "charset-normalizer": "3.5.1",
    "idna": "3.19",
    "urllib3": "2.7.0",
    "certifi": "2026.7.22",
    "insightface": "1.0.1",
}

PINNED_NVIDIA_DISTRIBUTIONS = {
    "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cuda-nvrtc-cu12": "12.8.93",
    "nvidia-nvjitlink-cu12": "12.8.93",
    "nvidia-cublas-cu12": "12.8.5.5",
    "nvidia-cufft-cu12": "11.3.3.83",
    "nvidia-curand-cu12": "10.3.9.90",
    "nvidia-cudnn-cu12": "9.24.0.43",
}


def _version(module) -> str:
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else "version n/a"


def _check_windows_vc_runtime() -> list[str]:
    if os.name != "nt":
        return []
    missing: list[str] = []
    python_dir = Path(sys.executable).resolve().parent
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        loaded = False
        for candidate in (dll, str(python_dir / dll), str(system_root / "System32" / dll)):
            try:
                ctypes.WinDLL(candidate)
                loaded = True
                break
            except OSError:
                continue
        if not loaded:
            missing.append(dll)
    return missing


def _distribution_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _pip_check() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    output = (proc.stdout or "").strip()
    if proc.returncode == 0:
        return True, output

    # InsightFace declares a dependency on the *distribution* named
    # ``onnxruntime``. Photo Select AI deliberately installs exactly one ORT
    # distribution; on NVIDIA systems that is ``onnxruntime-gpu``, which
    # exports the same Python module but does not satisfy pip's distribution-
    # name metadata check. Ignore only that one known false positive when the
    # GPU distribution is actually installed. Every other pip-check finding
    # remains fatal.
    gpu_ort = _distribution_installed("onnxruntime-gpu")
    remaining: list[str] = []
    ignored: list[str] = []
    for line in output.splitlines():
        low = line.casefold()
        intentional_ort_alias = (
            gpu_ort
            and "insightface" in low
            and "requires onnxruntime" in low
        )
        if intentional_ort_alias:
            ignored.append(line.strip())
        elif line.strip():
            remaining.append(line.strip())

    if not remaining and ignored:
        return True, (
            "no broken requirements; ignored intentional InsightFace -> "
            "onnxruntime metadata alias because onnxruntime-gpu is installed"
        )
    return False, "\n".join(remaining) or output


def main() -> int:
    print("Dependency self-test for Photo Select AI")
    failures: list[str] = []

    missing_vc = _check_windows_vc_runtime()
    if missing_vc:
        message = (
            "Microsoft Visual C++ Runtime x64 is incomplete: " + ", ".join(missing_vc) +
            ". Install/repair the latest Microsoft Visual C++ Redistributable x64, then rerun install.bat."
        )
        failures.append(message)
        print("  FAIL Visual C++ Runtime:", message)
    elif os.name == "nt":
        print("  PASS Microsoft Visual C++ Runtime")

    for import_name, package_name in REQUIRED:
        try:
            module = importlib.import_module(import_name)
            print(f"  PASS {package_name}: {_version(module)}")
        except Exception as exc:
            text = f"{package_name}: {type(exc).__name__}: {exc}"
            if import_name == "onnxruntime" and os.name == "nt":
                text += " (if a DLL is missing, repair Microsoft Visual C++ Redistributable x64)"
            failures.append(text)
            print(f"  FAIL {text}")

    for distribution, expected in PINNED_DISTRIBUTIONS.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            # The import test above already emits the clearer primary error.
            continue
        if installed != expected:
            text = f"{distribution}: installed {installed}, required exactly {expected}"
            failures.append(text)
            print(f"  FAIL {text}")

    cpu_ort = _distribution_installed("onnxruntime")
    gpu_ort = _distribution_installed("onnxruntime-gpu")
    if cpu_ort and gpu_ort:
        text = "both onnxruntime and onnxruntime-gpu are installed; exactly one is required"
        failures.append(text)
        print(f"  FAIL {text}")
    ort_distribution = "onnxruntime-gpu" if gpu_ort else "onnxruntime"
    try:
        ort_version = importlib.metadata.version(ort_distribution)
        if ort_version != "1.26.0":
            text = f"{ort_distribution}: installed {ort_version}, required exactly 1.26.0"
            failures.append(text)
            print(f"  FAIL {text}")
    except importlib.metadata.PackageNotFoundError:
        pass

    if gpu_ort:
        for distribution, expected in PINNED_NVIDIA_DISTRIBUTIONS.items():
            try:
                installed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                text = f"{distribution}: required by the pinned NVIDIA runtime but not installed"
                failures.append(text)
                print(f"  FAIL {text}")
                continue
            if installed != expected:
                text = f"{distribution}: installed {installed}, required exactly {expected}"
                failures.append(text)
                print(f"  FAIL {text}")

    if not failures:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if not providers:
                failures.append("ONNX Runtime: no execution providers are available")
            else:
                print("  ONNX Runtime providers:", ", ".join(providers))
        except Exception as exc:
            failures.append(f"ONNX Runtime provider check: {type(exc).__name__}: {exc}")

    pip_ok, pip_output = _pip_check()
    if pip_ok:
        print("  PASS pip check:", pip_output or "no broken requirements")
    else:
        failures.append("pip check: " + (pip_output or "dependency conflict detected"))
        print("  FAIL pip check:", pip_output or "dependency conflict detected")

    if failures:
        print("Dependency self-test FAILED:")
        for failure in failures:
            print("  -", failure)
        return 1

    print("Dependency self-test PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
