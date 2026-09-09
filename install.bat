@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PHOTOSELECT_PIP_INSECURE_PYPI=0"

if exist "config\ca-bundle.pem" (
  set "PIP_CERT=%CD%\config\ca-bundle.pem"
  set "REQUESTS_CA_BUNDLE=%CD%\config\ca-bundle.pem"
  set "SSL_CERT_FILE=%CD%\config\ca-bundle.pem"
  echo Using custom CA bundle for Python tools: config\ca-bundle.pem
)

echo ==============================================
echo Photo Select AI - installation v0.7.3
echo Private Python: CPython 3.11.16 x64 via uv
echo System Python and winget are not used.
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
  echo.
) else (
  echo.
  echo Normal TLS certificate verification selected.
  echo.
)

set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "POWERSHELL_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

if not exist "app\tools\install_managed_python.ps1" (
  echo ERROR: app\tools\install_managed_python.ps1 is missing.
  goto :install_failed
)
if not exist "app\tools\repair_venv.ps1" (
  echo ERROR: app\tools\repair_venv.ps1 is missing.
  goto :install_failed
)
if not exist "app\tools\install_windows.ps1" (
  echo ERROR: app\tools\install_windows.ps1 is missing.
  goto :install_failed
)
if not exist "app\tools\install_runtime.py" (
  echo ERROR: app\tools\install_runtime.py is missing.
  goto :install_failed
)
if not exist "app\tools\model_selftest.py" (
  echo ERROR: app\tools\model_selftest.py is missing.
  goto :install_failed
)
if not exist "app\tools\dependency_selftest.py" (
  echo ERROR: app\tools\dependency_selftest.py is missing.
  goto :install_failed
)
if not exist "app\tools\cuda_selftest.py" (
  echo ERROR: app\tools\cuda_selftest.py is missing.
  goto :install_failed
)
if not exist "app\requirements.txt" (
  echo ERROR: app\requirements.txt is missing.
  goto :install_failed
)

echo.
echo Preparing private Python. No system Python is required...
"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\tools\install_managed_python.ps1"
if errorlevel 1 goto :install_failed

set "PY=runtime\venv\Scripts\python.exe"
if not exist "%PY%" goto :install_failed

rem Remove obsolete v0.1/v0.2.1 MediaPipe artifacts. This is intentionally
rem repeated on upgrades so an existing venv/project is cleaned too.
if exist "models\face_landmarker.task" del /q "models\face_landmarker.task" >nul 2>nul
if exist "models\face_landmarker_v2_with_blendshapes.task" del /q "models\face_landmarker_v2_with_blendshapes.task" >nul 2>nul
if exist "runtime\downloads\face_landmarker.task" del /q "runtime\downloads\face_landmarker.task" >nul 2>nul

rem Bootstrap is always downloaded by PowerShell through the Windows TLS stack.
echo.
echo Bootstrapping pip/setuptools/packaging/wheel through Windows PowerShell...
"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "app\tools\install_windows.ps1" -Action bootstrap-pip -Python "%CD%\%PY%"
if errorlevel 1 goto :install_failed

:install_dependencies
echo.
rem Remove distributions that are no longer part of the current InsightFace build.
"%PY%" -m pip uninstall -y mediapipe opencv-python-headless >nul 2>nul
if "%PHOTOSELECT_PIP_INSECURE_PYPI%"=="1" (
  echo Installing application dependencies in Kaspersky compatibility mode...
  "%PY%" -m pip install --no-deps --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org -r "app\requirements.txt"
) else (
  echo Installing application dependencies with normal TLS verification...
  "%PY%" -m pip install --no-deps -r "app\requirements.txt"
)
if errorlevel 1 goto :pip_failed

"%PY%" "app\tools\install_runtime.py"
if errorlevel 1 goto :runtime_failed
echo.
echo Verifying application Python dependencies and pip consistency...
"%PY%" "app\tools\dependency_selftest.py"
if errorlevel 1 goto :install_failed
"%PY%" -m pip freeze --all > "runtime\installed_packages.txt"
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
echo.
echo Verifying application Python dependencies and pip consistency...
"%PY%" "app\tools\dependency_selftest.py"
if errorlevel 1 goto :install_failed
"%PY%" -m pip freeze --all > "runtime\installed_packages.txt"
goto :download_models

:download_models
echo.
echo Installing/verifying ALL neural models through Windows PowerShell...
echo   - complete InsightFace buffalo_l pack ^(5 ONNX models^)
echo   - portrait preference ResNet-18 ^(Caffe model + prototxt^)
"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "app\tools\install_windows.ps1" -Action download-models -ForceModels
if errorlevel 1 goto :install_failed

echo.
echo Running full CPU validation of every installed model...
"%PY%" "app\tools\model_selftest.py" --full --write-manifest
if errorlevel 1 (
  echo.
  echo Model validation failed. Re-downloading ALL models once as a clean recovery...
  "%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "app\tools\install_windows.ps1" -Action download-models -ForceModels
  if errorlevel 1 goto :install_failed
  "%PY%" "app\tools\model_selftest.py" --full --write-manifest
  if errorlevel 1 goto :install_failed
)

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
echo Installation complete - private CPython 3.11.16 environment ready.
echo ==============================================
if "%PHOTOSELECT_PIP_INSECURE_PYPI%"=="1" (
  echo Python package downloads used PyPI compatibility mode.
  echo Certificate verification was bypassed ONLY for pypi.org and files.pythonhosted.org.
) else (
  echo Python package downloads used normal TLS certificate verification.
)
echo PowerShell model downloads used the Windows certificate store.
echo Complete InsightFace buffalo_l archive is SHA-256 verified and all 5 ONNX models are validated on CPU.
echo Public portrait-preference model source is pinned to an immutable upstream commit and validated with OpenCV DNN.
echo Its SHA-256 values are recorded in the model manifest; runtime can fall back safely if FBP is later unavailable.
echo Old MediaPipe package/model remnants were removed.
echo CUDA 12.8 runtime components are pinned and Windows DLL paths are configured at runtime.
echo.
echo Use run.bat or: run.bat "D:\Photos\Shoot"
echo.
echo The project folder can be renamed or moved.
echo On another computer, run install.bat once if Windows-specific drivers/runtime need validation.
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
