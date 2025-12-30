@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Sci-Agent-SPM.ps1" %*
endlocal
