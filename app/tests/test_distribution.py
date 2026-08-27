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


def test_root_launcher_uses_private_local_python_without_silent_reinstall():
    run_bat = (ROOT / "run.bat").read_text(encoding="utf-8")
    install = (ROOT / "install.bat").read_text(encoding="utf-8")
    assert "runtime\\venv\\Scripts\\python.exe" in run_bat
    assert "call install.bat" not in run_bat.lower()
    assert "repair_venv.ps1" in run_bat
    assert "install_managed_python.ps1" in install
    assert "winget install" not in install.lower()
    assert "py -3.11" not in install.lower()
    assert (ROOT / "app" / "tools" / "repair_venv.ps1").is_file()
    assert (ROOT / "app" / "tools" / "install_managed_python.ps1").is_file()


def test_private_venv_is_relocatable_and_move_repair_preserves_packages():
    installer = (ROOT / "app" / "tools" / "install_managed_python.ps1").read_text(encoding="utf-8-sig")
    repair = (ROOT / "app" / "tools" / "repair_venv.ps1").read_text(encoding="utf-8-sig")
    assert "--relocatable" in installer
    assert "--relocatable" in repair
    assert "--allow-existing" in repair
    assert "--no-python-downloads" in repair
    assert "Move-Item -LiteralPath $tmp" not in repair


def test_relocatable_validation_does_not_require_absolute_home_metadata():
    installer = (ROOT / "app" / "tools" / "install_managed_python.ps1").read_text(encoding="utf-8-sig")
    repair = (ROOT / "app" / "tools" / "repair_venv.ps1").read_text(encoding="utf-8-sig")
    for script in (installer, repair):
        assert "sys.base_prefix" in script
        assert "APP_EXPECTED_BASE" in script
        assert "$HomeMoved" not in script
        assert "$venvHome" not in script
    assert "Split-Path -Parent $base" in installer


def test_fresh_venv_bootstraps_pip_with_uv_not_with_missing_pip():
    helper = (ROOT / "app" / "tools" / "install_windows.ps1").read_text(encoding="utf-8-sig")
    assert "runtime\\uv\\uv.exe" in helper
    assert "pip install --python $Python --no-index" in helper
    assert "& $Python -m pip install" not in helper
    assert "& $Python -m pip --version" in helper


def test_group_yellow_default_and_absolute_limit_are_consistent():
    import json
    from app.core.constants import GROUP_YELLOW_DEFAULT_MAX, GROUP_YELLOW_MAX_LIMIT

    config = json.loads((ROOT / "config" / "default.json").read_text(encoding="utf-8"))
    assert GROUP_YELLOW_DEFAULT_MAX == 2
    assert GROUP_YELLOW_MAX_LIMIT == 5
    assert config["group"]["max_extra_candidates"] == GROUP_YELLOW_DEFAULT_MAX
