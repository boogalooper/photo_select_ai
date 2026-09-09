@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Photo Select AI launcher
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo [Photo Select AI] Starting...
echo [1/5] Checking private CPython 3.11.16 environment...

if not exist "runtime\venv\Scripts\python.exe" goto :not_installed
if not exist "app\tools\repair_venv.ps1" goto :not_installed
if not exist "app\tools\model_selftest.py" goto :not_installed
if not exist "app\tools\dependency_selftest.py" goto :not_installed

set "PY=runtime\venv\Scripts\python.exe"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "POWERSHELL_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\tools\repair_venv.ps1"
if errorlevel 1 goto :broken

echo [2/5] Local Python environment OK.
echo [3/5] Checking application dependencies...
"%PY%" "app\tools\dependency_selftest.py" >nul 2>&1
if errorlevel 1 goto :dependencies_broken
echo [4/5] Checking models installed by install.bat...
"%PY%" "app\tools\model_selftest.py" --runtime-files >nul 2>&1
if not errorlevel 1 goto :models_ready

echo.
echo Model state needs fallback validation.
echo Normal installations receive all models from install.bat; no download is attempted yet.
"%PY%" "app\tools\model_selftest.py" --runtime-full --write-manifest
if not errorlevel 1 goto :models_ready

echo.
echo FALLBACK: one or more runtime-required InsightFace model files are missing or invalid.
echo Re-downloading the official InsightFace model pack once...
"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\tools\install_windows.ps1" -Action download-insightface -ForceModels
if errorlevel 1 goto :models_broken

"%PY%" "app\tools\model_selftest.py" --runtime-full --write-manifest
if errorlevel 1 goto :models_broken

echo InsightFace runtime recovery completed.
echo Facial Beauty Prediction is optional at runtime; if unavailable, the app uses legacy ranking fallback.

:models_ready
echo [5/5] Opening Photo Select AI...
"%PY%" -m app.main %*
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" exit /b 0

echo.
echo Photo Select AI closed with error code %RC%.
echo See the application log for details.
pause
exit /b %RC%

:dependencies_broken
echo.
echo The private Python environment exists, but one or more required packages are missing or broken.
echo Model download fallback was NOT started because this is not a model-file problem.
echo Run install.bat to repair the application environment.
pause
exit /b 1

:models_broken
echo.
echo Model fallback recovery failed.
echo Run install.bat to perform a complete installation and model validation.
pause
exit /b 1

:not_installed
echo.
echo Private Python environment is not installed yet.
echo Run install.bat once. It installs the private Python environment, packages and ALL models.
echo System Python is NOT required.
pause
exit /b 1

:broken
echo.
echo The private Python environment could not be checked or repaired.
echo Run install.bat once to rebuild it. System Python is NOT required.
pause
exit /b 1
