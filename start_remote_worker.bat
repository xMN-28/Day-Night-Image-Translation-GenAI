@echo off
setlocal
title LumiCycle Remote GPU Worker

cd /d "%~dp0"
echo Starting LumiCycle remote GPU worker...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_remote_demo_worker.ps1"
set "WORKER_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%WORKER_EXIT_CODE%"=="0" (
    echo The worker stopped with error code %WORKER_EXIT_CODE%.
    echo Review the message above, then press any key to close this window.
    pause >nul
) else (
    echo The worker has stopped.
)

exit /b %WORKER_EXIT_CODE%
