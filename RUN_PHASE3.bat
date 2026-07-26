@echo off
title TrafficTracker phase 3 - Dhaka cross-dataset
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_phase3.ps1"
pause
