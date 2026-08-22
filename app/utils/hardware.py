from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

import psutil


@dataclass(slots=True)
class HardwareInfo:
    cpu: str
    logical_cpus: int
    ram_gb: float
    gpu: str | None
    onnx_providers: list[str]

    @property
    def summary(self) -> str:
        gpu = self.gpu or "GPU not detected"
        return f"CPU {self.logical_cpus} threads | RAM {self.ram_gb:.1f} GB | {gpu}"


def detect_hardware() -> HardwareInfo:
    gpu = None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        if output:
            gpu = output.splitlines()[0].strip()
    except Exception:
        pass

    providers: list[str] = []
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(directory="")
            except Exception:
                pass
        providers = list(ort.get_available_providers())
    except Exception:
        pass

    return HardwareInfo(
        cpu=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "Unknown CPU"),
        logical_cpus=os.cpu_count() or 1,
        ram_gb=psutil.virtual_memory().total / (1024**3),
        gpu=gpu,
        onnx_providers=providers,
    )
