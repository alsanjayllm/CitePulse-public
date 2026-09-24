@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create venv - is Python 3.11+ installed and on PATH?
        pause
        exit /b 1
    )
)

echo Installing/updating dependencies (including the ui extra)...
.venv\Scripts\python.exe -m pip install -q -e ".[ui]"
if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
)

echo.
echo Ensuring Playwright Chromium is installed (for screenshots)...
.venv\Scripts\python.exe -m playwright install chromium
if errorlevel 1 (
    echo Warning: Playwright Chromium could not be installed - screenshots and
    echo task-readiness visuals may be unavailable, but the app will still launch.
)

echo.
echo === Launching CitePulse UI (opens in your browser) ===
echo Press Ctrl+C in this window to stop the server.
echo.
.venv\Scripts\citepulse.exe ui

echo.
pause
endlocal
