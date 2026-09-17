@echo off
rem This is the SOURCE template -- build_dist.bat copies this file into
rem dist\citepulse\ alongside the built citepulse.exe. Running THIS repo-root
rem copy directly will fail with "'citepulse.exe' is not recognized", since
rem there is no exe next to it here. To test the real launcher, run
rem dist\citepulse\Start CitePulse.bat (after `build_dist.bat`), or extract
rem an actual release zip and run the copy inside it.
cd /d "%~dp0"

rem Keep this install's data (sites, audit history, logs) self-contained
rem next to the exe instead of the shared %USERPROFILE%\.citepulse default,
rem so re-extracting a new zip build never mixes with -- or appears to
rem inherit -- another install's history.
set "CITEPULSE_DATA_DIR=%~dp0data"

rem v1 core enterprise-LAN edition: restrict the UI to the core nav/KPI
rem set (Sites, Run Audit, History, Manage -- no Compare Models/OpenRouter/
rem batch mode) and disable outbound competitor-discovery web searches.
set "CORE_EDITION_MODE=true"
set "COMPETITOR_DISCOVERY_ENABLED=false"

citepulse.exe launch
pause
