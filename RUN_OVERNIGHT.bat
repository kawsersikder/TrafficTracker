@echo off
title TrafficTracker overnight run
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_overnight.ps1"
echo.
echo ================================================
echo  Run finished. Progress summary: RUN_PROGRESS.md
echo ================================================
pause
