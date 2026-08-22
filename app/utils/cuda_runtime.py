from __future__ import annotations

import ctypes
import logging
import os
import platform
import site
import sysconfig
from pathlib import Path
from typing import Iterable

_LOG = logging.getLogger("photo_select_ai")
_DLL_HANDLES: list[object] = []
_DLL_DIR_HANDLES: list[object] = []
_CONFIGURED = False


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []
    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass

    for key in ("purelib", "platlib"):
        try:
            value = sysconfig.get_path(key)
            if value:
                candidates.append(value)
        except Exception:
            pass

    seen: set[str] = set()
    for value in candidates:
        path = Path(value).resolve()
        key = str(path).lower()
        if key not in seen and path.is_dir():
            seen.add(key)
            roots.append(path)
    return roots


def nvidia_dll_directories() -> list[Path]:
    """Return all pip-installed NVIDIA DLL directories inside this Python env.

    CUDA 12 wheels install individual components under site-packages/nvidia/*/bin.
    cuDNN can dynamically LoadLibrary() cublasLt/NVRTC while executing a graph, so
    merely importing onnxruntime is not sufficient on every Windows setup.
    """
    dirs: list[Path] = []
    seen: set[str] = set()
    for root in _site_package_roots():
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for dll in nvidia.rglob("*.dll"):
            parent = dll.parent.resolve()
            key = str(parent).lower()
            if key not in seen:
                seen.add(key)
                dirs.append(parent)
    return dirs


def _prepend_process_path(directories: Iterable[Path]) -> None:
    current = os.environ.get("PATH", "")
    current_parts = [p for p in current.split(os.pathsep) if p]
    current_lower = {p.lower() for p in current_parts}
    new_parts: list[str] = []
    for directory in directories:
        value = str(directory)
        if value.lower() not in current_lower:
            new_parts.append(value)
    if new_parts:
        os.environ["PATH"] = os.pathsep.join(new_parts + current_parts)


def _register_windows_dll_dirs(directories: Iterable[Path]) -> None:
    if platform.system() != "Windows" or not hasattr(os, "add_dll_directory"):
        return
    for directory in directories:
        try:
            # Keep handles alive for the lifetime of the process. Closing the
            # handle removes that directory from the DLL search path.
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
        except OSError:
            continue


def _load_matching(directories: Iterable[Path], patterns: Iterable[str]) -> list[str]:
    loaded: list[str] = []
    if platform.system() != "Windows":
        return loaded
    for pattern in patterns:
        matches: list[Path] = []
        for directory in directories:
            matches.extend(sorted(directory.glob(pattern)))
        # Prefer the first copy in the active virtual environment.
        for path in matches:
            try:
                _DLL_HANDLES.append(ctypes.WinDLL(str(path)))
                loaded.append(path.name)
                break
            except OSError as exc:
                _LOG.debug("Could not preload %s: %s", path, exc)
    return loaded


def prepare_windows_cuda_dlls() -> dict[str, object]:
    """Make pip-installed CUDA/cuDNN DLLs visible to cuDNN on Windows.

    NVIDIA's cuDNN runtime may dynamically load cublasLt and NVRTC. The CUDA
    component wheels place these DLLs in separate site-packages directories,
    which are not automatically on PATH. Add all NVIDIA bin directories to the
    process DLL search path and explicitly preload the critical runtime pieces.
    """
    global _CONFIGURED
    if platform.system() != "Windows":
        return {"configured": False, "reason": "not-windows", "directories": [], "loaded": []}
    if _CONFIGURED:
        return {"configured": True, "reason": "already-configured", "directories": [], "loaded": []}

    directories = nvidia_dll_directories()
    _prepend_process_path(directories)
    _register_windows_dll_dirs(directories)

    # Load dependencies before cuDNN graph/runtime sublibraries. NVRTC is the
    # important missing piece for CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED on
    # pip-only Windows installs; nvJitLink can be a dependency of NVRTC.
    loaded = _load_matching(
        directories,
        (
            "nvJitLink*.dll",
            "nvrtc-builtins*.dll",
            "nvrtc*.dll",
            "cublasLt*.dll",
            "cublas64_*.dll",
            "cudart64_*.dll",
        ),
    )
    _CONFIGURED = True
    return {
        "configured": True,
        "reason": "ok",
        "directories": [str(p) for p in directories],
        "loaded": loaded,
    }


def cuda_runtime_versions() -> dict[str, str]:
    """Best-effort installed distribution versions for diagnostic logging."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return {}

    packages = (
        "onnxruntime-gpu",
        "nvidia-cudnn-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-cufft-cu12",
    )
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return result
