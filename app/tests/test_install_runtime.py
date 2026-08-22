from __future__ import annotations

from unittest.mock import patch

from app.tools import install_runtime


def test_pip_install_normal_mode(monkeypatch):
    monkeypatch.delenv("PHOTOSELECT_PIP_INSECURE_PYPI", raising=False)
    with patch("app.tools.install_runtime.subprocess.call", return_value=0) as call:
        assert install_runtime.pip_install("example") == 0
    cmd = call.call_args.args[0]
    assert "--trusted-host" not in cmd
    assert cmd[-1] == "example"


def test_pip_install_kaspersky_compatibility_mode(monkeypatch):
    monkeypatch.setenv("PHOTOSELECT_PIP_INSECURE_PYPI", "1")
    with patch("app.tools.install_runtime.subprocess.call", return_value=0) as call:
        assert install_runtime.pip_install("example") == 0
    cmd = call.call_args.args[0]
    assert cmd.count("--trusted-host") == 2
    assert "https://pypi.org/simple" in cmd
    assert "pypi.org" in cmd
    assert "files.pythonhosted.org" in cmd
    assert cmd[-1] == "example"


def test_nvidia_runtime_is_pinned_to_blackwell_cuda_12_8(monkeypatch):
    monkeypatch.setattr(install_runtime, "has_nvidia", lambda: True)
    calls = []
    monkeypatch.setattr(install_runtime, "pip_install", lambda *args: calls.append(args) or 0)
    with patch("app.tools.install_runtime.subprocess.call", return_value=0):
        assert install_runtime.install_ort() == 0
    assert len(calls) == 1
    packages = calls[0]
    assert "onnxruntime-gpu==1.26.0" in packages
    assert "nvidia-cuda-runtime-cu12==12.8.90" in packages
    assert "nvidia-cuda-nvrtc-cu12==12.8.93" in packages
    assert "nvidia-nvjitlink-cu12==12.8.93" in packages
    assert "nvidia-cublas-cu12==12.8.5.5" in packages
    assert "nvidia-cudnn-cu12==9.24.0.43" in packages


def test_nvidia_runtime_can_fallback_to_cpu_package(monkeypatch):
    monkeypatch.setattr(install_runtime, "has_nvidia", lambda: True)
    calls = []
    def fake_install(*args):
        calls.append(args)
        return 1 if any("onnxruntime-gpu" in a for a in args) else 0
    monkeypatch.setattr(install_runtime, "pip_install", fake_install)
    with patch("app.tools.install_runtime.subprocess.call", return_value=0):
        assert install_runtime.install_ort() == 0
    assert calls[-1] == ("onnxruntime==1.26.0",)
