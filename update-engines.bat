@echo off
rem update-engines.bat - update-engines.ps1 실행기 (더블클릭 또는 CMD 에서 사용)
rem 인자는 그대로 전달: update-engines.bat -Pull / update-engines.bat strata poly / update-engines.bat -NoDeploy
chcp 65001 >nul
where pwsh >nul 2>&1 && (set "PS=pwsh") || (set "PS=powershell")
%PS% -NoProfile -ExecutionPolicy Bypass -File "%~dp0update-engines.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
if not defined CI pause
exit /b %RC%
