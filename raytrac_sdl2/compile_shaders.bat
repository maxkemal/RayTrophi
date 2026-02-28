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
        echo Compiling: %%~nxf
        "%GLSLC%" "%%f" -o "%OUTPUT_DIR%\%%~nf.spv" --target-env=vulkan1.3 --target-spv=spv1.4
        if errorlevel 1 (
            echo FAILED: %%~nxf
            goto :error
        )
        echo   OK: %%~nf.spv
    )
)

echo.
echo ===== All shaders compiled successfully =====
pause
exit /b 0

:error
echo.
echo ===== Compilation FAILED =====
pause
exit /b 1
