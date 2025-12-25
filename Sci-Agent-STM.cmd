@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Sci-Agent-STM.ps1" %*
endlocal
