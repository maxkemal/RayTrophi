@echo off
REM =========================================================================
REM RayTrophi - Vulkan Shader Compiler Script
REM Compiles GLSL shaders to SPIR-V using glslc from Vulkan SDK
REM =========================================================================

setlocal

set GLSLC=%VULKAN_SDK%\Bin\glslc.exe
set SHADER_DIR=%~dp0source\shaders
set OUTPUT_DIR=%~dp0source\shaders

if not exist "%GLSLC%" (
    echo ERROR: glslc not found at %GLSLC%
    echo Make sure VULKAN_SDK environment variable is set correctly.
    goto :error
)

echo ===== RayTrophi Vulkan Shader Compilation =====
echo GLSLC: %GLSLC%
echo Shader Dir: %SHADER_DIR%
echo.

REM Compile compute shaders (.comp)
for %%f in (%SHADER_DIR%\*.comp) do (
    echo Compiling: %%~nxf
    "%GLSLC%" "%%f" -o "%OUTPUT_DIR%\%%~nf.spv" --target-env=vulkan1.3
    if errorlevel 1 (
        echo FAILED: %%~nxf
        goto :error
    )
    echo   OK: %%~nf.spv
)

REM Compile ray tracing shaders (.rgen, .rmiss, .rchit, .rahit, .rint)
for %%e in (rgen rmiss rchit rahit rint) do (
    for %%f in (%SHADER_DIR%\*.%%e) do (
        REM Skip shadow_anyhit.rchit — superseded by shadow_anyhit.rahit (correct any-hit stage)
        if /I "%%~nxf"=="shadow_anyhit.rchit" (
            echo   SKIPPING: %%~nxf ^(replaced by shadow_anyhit.rahit^)
        ) else (
            echo Compiling: %%~nxf
            "%GLSLC%" "%%f" -o "%OUTPUT_DIR%\%%~nf.spv" --target-env=vulkan1.3 --target-spv=spv1.4
            if errorlevel 1 (
                echo FAILED: %%~nxf
                goto :error
            )
            echo   OK: %%~nf.spv
        )
    )
)

echo.
echo ===== All shaders compiled successfully =====

REM Deploy compiled .spv to runtime directories
echo.
echo Deploying .spv files to runtime directories...
set DEPLOY1=%~dp0x64\Release\shaders
set DEPLOY2=%~dp0..\x64\Release\shaders
set DEPLOY3=%~dp0..\build\Release\shaders

for %%d in ("%DEPLOY1%" "%DEPLOY2%" "%DEPLOY3%") do (
    if exist %%d (
        echo   Copying to %%d
        copy /Y "%OUTPUT_DIR%\*.spv" %%d >nul 2>&1
    )
)
echo Deploy complete.
pause
exit /b 0

:error
echo.
echo ===== Compilation FAILED =====
pause
exit /b 1
