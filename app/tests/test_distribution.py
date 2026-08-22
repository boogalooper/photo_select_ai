from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_batch_files_have_windows_format_and_no_bom():
    for path in (ROOT / "install.bat", ROOT / "run.bat"):
        data = path.read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf")
        assert data.startswith(b"@echo off\r\n")
        assert b"\r\n" in data


def test_portable_functionality_is_fully_removed():
    assert not (ROOT / "app" / "tools" / "build_portable.py").exists()
    assert not (ROOT / "app" / "tools" / "build_portable.bat").exists()
    assert not (ROOT / "app" / "tools" / "portable_selftest.py").exists()
    install = (ROOT / "install.bat").read_text(encoding="utf-8")
    run_bat = (ROOT / "run.bat").read_text(encoding="utf-8")
    main = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    combined = "\n".join((install, run_bat, main)).lower()
    assert "portable" not in combined
    assert "pyinstaller" not in combined


def test_root_launcher_uses_local_venv():
    run_bat = (ROOT / "run.bat").read_text(encoding="utf-8")
    assert "runtime\\venv\\Scripts\\python.exe" in run_bat
    assert "call install.bat" in run_bat
