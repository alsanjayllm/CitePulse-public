@echo off
setlocal
cd /d "%~dp0"

rem Builds the standalone, no-admin-required Windows distributable (see
rem docs/PACKAGING.md for the full rationale and the manual verification
rem checklist you must run against the output before calling a build
rem release-ready -- a successful build here does NOT prove the frozen
rem exe actually works).
rem
rem Separate, optional track from the normal dev loop -- run run_test.bat
rem first to confirm tests pass; this script does not re-run pytest itself.

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create venv - is Python 3.11+ installed and on PATH?
        pause
        exit /b 1
    )
)

echo Installing/updating build dependencies...
.venv\Scripts\python.exe -m pip install -q -e ".[build,ui]"
if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
)

echo.
echo === Reading version from citepulse.__version__ ===
for /f "usebackq delims=" %%V in (`.venv\Scripts\python.exe -c "import citepulse; print(citepulse.__version__)"`) do set CITEPULSE_VERSION=%%V
if "%CITEPULSE_VERSION%"=="" (
    echo Could not read citepulse.__version__.
    pause
    exit /b 1
)
echo Version: %CITEPULSE_VERSION%

echo.
echo === Cleaning previous build output ===
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo === Running PyInstaller ===
.venv\Scripts\pyinstaller.exe citepulse.spec
if errorlevel 1 (
    echo PyInstaller build failed.
    pause
    exit /b 1
)

if not exist "dist\citepulse\citepulse.exe" (
    echo Build did not produce the expected dist\citepulse\citepulse.exe.
    pause
    exit /b 1
)

echo.
echo === Copying "Start CitePulse.bat" into dist\citepulse ===
copy /y "Start CitePulse.bat" "dist\citepulse\Start CitePulse.bat" >nul
if errorlevel 1 (
    echo Failed to copy "Start CitePulse.bat" into dist\citepulse.
    pause
    exit /b 1
)

echo.
echo === Zipping dist\citepulse ===
set ZIP_NAME=citepulse-win-x64-%CITEPULSE_VERSION%.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\citepulse\*' -DestinationPath 'dist\%ZIP_NAME%' -Force"
if errorlevel 1 (
    echo Zipping failed.
    pause
    exit /b 1
)

echo.
echo === Done: dist\%ZIP_NAME% ===
echo Before calling this release-ready, run the manual verification
echo checklist in docs\PACKAGING.md -- a successful build does not by
echo itself prove the frozen exe works with no admin rights / no system
echo Python. In particular: copy dist\citepulse\ OUTSIDE this repo before
echo testing, so it can't silently fall through to this repo's own .venv.
echo.
pause
endlocal
