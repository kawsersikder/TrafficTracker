@echo off
title TrafficTracker phase 2
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_phase2.ps1"
pause
