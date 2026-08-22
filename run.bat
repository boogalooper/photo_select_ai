@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "APP_ARGS=%*"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if exist "runtime\venv\Scripts\python.exe" (
  "runtime\venv\Scripts\python.exe" -c "import sys,pathlib; expected=pathlib.Path(r'%CD%\runtime\venv').resolve(); actual=pathlib.Path(sys.prefix).resolve(); assert sys.version_info[:2] == (3,11); assert actual == expected" >nul 2>nul
  if not errorlevel 1 (
    "runtime\venv\Scripts\python.exe" -m app.main %APP_ARGS%
    exit /b %errorlevel%
  )
  echo Existing runtime\venv cannot be used from this location/computer.
)

echo Photo Select AI needs a local Python environment.
echo Running installer...
call install.bat
if errorlevel 1 exit /b %errorlevel%

"runtime\venv\Scripts\python.exe" -m app.main %APP_ARGS%
exit /b %errorlevel%
