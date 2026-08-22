@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PHOTOSELECT_PIP_INSECURE_PYPI=0"

rem Optional explicit CA bundle for unusual environments.
if exist "config\ca-bundle.pem" (
  set "PIP_CERT=%CD%\config\ca-bundle.pem"
  set "REQUESTS_CA_BUNDLE=%CD%\config\ca-bundle.pem"
  set "SSL_CERT_FILE=%CD%\config\ca-bundle.pem"
  echo Using custom CA bundle for Python tools: config\ca-bundle.pem
)

echo ==============================================
echo Photo Select AI - installation v0.5.3 Portrait + Groups
echo ==============================================
echo.
echo Connection mode for Python packages:
echo   [1] Normal secure mode ^(recommended^)
echo   [2] Kaspersky compatibility mode
echo       Bypass certificate verification ONLY for:
echo       pypi.org and files.pythonhosted.org
echo   [3] Cancel installation
echo.
choice /C 123 /N /M "Choose 1, 2 or 3: "
if errorlevel 3 goto :install_failed
if errorlevel 2 (
  set "PHOTOSELECT_PIP_INSECURE_PYPI=1"
  echo.
  echo WARNING: PyPI certificate verification bypass is enabled for this install.
  echo PowerShell downloads and model hash verification remain secure.
  echo Use this mode only on a network you trust.
  echo.
) else (
  echo.
  echo Normal TLS certificate verification selected.
  echo.
)

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo ERROR: Windows PowerShell was not found.
  echo Photo Select AI installer requires powershell.exe on Windows.
  goto :install_failed
)

if exist "runtime\venv\Scripts\python.exe" (
  echo Checking existing virtual environment...
  "runtime\venv\Scripts\python.exe" -c "import sys,pathlib; expected=pathlib.Path(r'%CD%\runtime\venv').resolve(); actual=pathlib.Path(sys.prefix).resolve(); assert sys.version_info[:2] == (3,11); assert actual == expected, (actual, expected)" >nul 2>nul
  if not errorlevel 1 goto :have_venv
  echo Existing runtime\venv is not valid at this location or on this computer.
  echo Recreating the virtual environment...
  rmdir /s /q "runtime\venv"
)

set "BASEPY="
py -3.11 -c "import sys; print(sys.executable)" >nul 2>nul
if not errorlevel 1 set "BASEPY=py -3.11"

if not defined BASEPY (
  python -c "import sys; assert sys.version_info[:2] == (3,11)" >nul 2>nul
  if not errorlevel 1 set "BASEPY=python"
)

if not defined BASEPY (
  where winget >nul 2>nul
  if errorlevel 1 (
    echo Python 3.11 was not found and winget is unavailable.
    echo Install Python 3.11 x64 and rerun install.bat.
    goto :install_failed
  )
  echo Installing Python 3.11 using winget...
  winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
  if errorlevel 1 goto :install_failed
  set "BASEPY=py -3.11"
)

echo Creating virtual environment...
%BASEPY% -m venv "runtime\venv"
if errorlevel 1 goto :install_failed

:have_venv
set "PY=runtime\venv\Scripts\python.exe"

rem Remove obsolete v0.1/v0.2.1 MediaPipe artifacts. This is intentionally
rem repeated on upgrades so an existing venv/project is cleaned too.
if exist "models\face_landmarker.task" del /q "models\face_landmarker.task" >nul 2>nul
if exist "models\face_landmarker_v2_with_blendshapes.task" del /q "models\face_landmarker_v2_with_blendshapes.task" >nul 2>nul
if exist "runtime\downloads\face_landmarker.task" del /q "runtime\downloads\face_landmarker.task" >nul 2>nul

rem Bootstrap is always downloaded by PowerShell through the Windows TLS stack.
echo.
echo Bootstrapping pip/setuptools/packaging/wheel through Windows PowerShell...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "app\tools\install_windows.ps1" -Action bootstrap-pip -Python "%CD%\%PY%"
if errorlevel 1 goto :install_failed

:install_dependencies
echo.
rem Remove distributions that are no longer part of the current InsightFace build.
"%PY%" -m pip uninstall -y mediapipe opencv-python-headless >nul 2>nul
if "%PHOTOSELECT_PIP_INSECURE_PYPI%"=="1" (
  echo Installing application dependencies in Kaspersky compatibility mode...
  "%PY%" -m pip install --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org -r "app\requirements.txt"
) else (
  echo Installing application dependencies with normal TLS verification...
  "%PY%" -m pip install -r "app\requirements.txt"
)
if errorlevel 1 goto :pip_failed

"%PY%" "app\tools\install_runtime.py"
if errorlevel 1 goto :runtime_failed
goto :download_models

:pip_failed
echo.
echo ==============================================
echo Python package installation failed.
echo ==============================================
if "%PHOTOSELECT_PIP_INSECURE_PYPI%"=="1" goto :install_failed

echo.
echo If the error above is CERTIFICATE_VERIFY_FAILED and Kaspersky is intercepting HTTPS,
echo you can retry this step with certificate verification bypassed ONLY for official PyPI hosts.
echo.
echo   [1] Retry using Kaspersky compatibility mode
echo   [2] Cancel installation
echo.
choice /C 12 /N /M "Choose 1 or 2: "
if errorlevel 2 goto :install_failed
set "PHOTOSELECT_PIP_INSECURE_PYPI=1"
echo.
echo WARNING: retrying with PyPI certificate verification bypass enabled.
goto :install_dependencies

:runtime_failed
echo.
echo ==============================================
echo ONNX Runtime installation failed.
echo ==============================================
if "%PHOTOSELECT_PIP_INSECURE_PYPI%"=="1" goto :install_failed

echo The runtime installer also uses pip and may have hit the same TLS interception.
echo.
echo   [1] Retry runtime installation using Kaspersky compatibility mode
echo   [2] Cancel installation
echo.
choice /C 12 /N /M "Choose 1 or 2: "
if errorlevel 2 goto :install_failed
set "PHOTOSELECT_PIP_INSECURE_PYPI=1"
"%PY%" "app\tools\install_runtime.py"
if errorlevel 1 goto :install_failed
goto :download_models

:download_models
if exist "models\insightface\models\buffalo_l\det_10g.onnx" if exist "models\insightface\models\buffalo_l\w600k_r50.onnx" if exist "models\insightface\models\buffalo_l\2d106det.onnx" goto :models_ready

echo.
echo Downloading/verifying InsightFace buffalo_l through Windows PowerShell...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "app\tools\install_windows.ps1" -Action download-insightface
if errorlevel 1 goto :install_failed

:models_ready
"%PY%" -m compileall -q app
if errorlevel 1 goto :install_failed

echo.
echo Running CUDA self-test ^(detector + recognition + landmarks^)...
"%PY%" "app\tools\cuda_selftest.py"
if errorlevel 1 (
  echo.
  echo WARNING: CUDA self-test failed. The application is still installed and can use CPU fallback.
  echo For RTX 5090, update the NVIDIA driver to at least 570.65 and rerun install.bat.
  echo The log above also shows the exact NVIDIA runtime package versions.
) else (
  echo CUDA self-test passed.
)

goto :install_complete


:install_complete
echo.
echo ==============================================
echo Installation complete - Portrait + Groups CUDA build.
echo ==============================================
if "%PHOTOSELECT_PIP_INSECURE_PYPI%"=="1" (
  echo Python package downloads used PyPI compatibility mode.
  echo Certificate verification was bypassed ONLY for pypi.org and files.pythonhosted.org.
) else (
  echo Python package downloads used normal TLS certificate verification.
)
echo PowerShell model downloads used the Windows certificate store.
echo InsightFace model archive is SHA-256 verified.
echo Old MediaPipe package/model remnants were removed.
echo CUDA 12.8 runtime components are pinned and Windows DLL paths are configured at runtime.
echo.
echo Use run.bat or: run.bat "D:\Photos\Shoot"
echo.
echo To move Photo Select AI to another computer, copy the project folder and run install.bat there.
echo The runtime\venv folder is machine-specific and should be recreated on the target computer.
echo.
echo Press any key to close this installer...
pause >nul
exit /b 0

:install_failed
echo.
echo ==============================================
echo Installation failed or was cancelled.
echo ==============================================
echo.
echo Review the messages above to see which step failed.
echo Press any key to close this installer...
pause >nul
exit /b 1
