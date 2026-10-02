@echo off
rem Strata for Windows: pick one of the installed models (run-<model>-vision|novision.bat) and start it.
setlocal
title Strata
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py -3"
%PY% MoreSimpleStart.py %*
if errorlevel 1 pause
