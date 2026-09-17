@echo off
setlocal
cd /d "%~dp0"

rem Isolated from the real ~/.citepulse DB, so these test runs never eat
rem into your actual 5-site quota. Delete .testdata\ any time to reset.
set CITEPULSE_DATA_DIR=%~dp0.testdata

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create venv - is Python 3.11+ installed and on PATH?
        pause
        exit /b 1
    )
)

echo Installing/updating dependencies...
.venv\Scripts\python.exe -m pip install -q -e ".[dev,ui]"
if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
)

echo.
echo === Running test suite ===
.venv\Scripts\python.exe -m pytest -q
if errorlevel 1 (
    echo.
    echo Tests failed - stopping here.
    pause
    exit /b 1
)

echo.
echo === Initializing local database (isolated test copy) ===
echo (first run also downloads Playwright's Chromium browser -- may take a minute)
.venv\Scripts\citepulse.exe setup

echo.
echo === Sample audit 1/3: https://example.com (negative case) ===
echo (expect: band "critical" - no llms.txt published)
.venv\Scripts\citepulse.exe audit https://example.com

echo.
echo === Sample audit 2/3: https://www.hubspot.com (positive case) ===
echo (expect: band "best_in_class" - publishes a full llms.txt)
.venv\Scripts\citepulse.exe audit https://www.hubspot.com

echo.
echo === Sample audit 3/3: https://stripe.com (positive case) ===
echo (expect: band "best_in_class" - publishes a full llms.txt)
.venv\Scripts\citepulse.exe audit https://stripe.com

echo.
echo === Done. Run more audits yourself with: ===
echo   set CITEPULSE_DATA_DIR=%CITEPULSE_DATA_DIR%
echo   .venv\Scripts\citepulse.exe audit ^<url^>
echo   .venv\Scripts\citepulse.exe ui
echo.
pause
endlocal
