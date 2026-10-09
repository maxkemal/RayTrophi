@echo off
setlocal
REM The PowerShell driver stages each module under a unique name and publishes
REM only successful SPIR-V outputs. Keep this console open on a build failure.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0compile_shaders.ps1"
set "FAILURE_CODE=%errorlevel%"
if "%FAILURE_CODE%"=="0" exit /b 0
echo.
echo ===== Compilation FAILED =====
echo Exit code: %FAILURE_CODE%
echo Review the compiler output above. This window will remain open.
pause
endlocal & exit /b %FAILURE_CODE%
