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
    "%GLSLC%" "%%f" -o "%OUTPUT_DIR%\%%~nf.spv" --target-env=vulkan1.3 -O
    if errorlevel 1 (
        echo FAILED: %%~nxf
        goto :error
    )
    echo   OK: %%~nf.spv
)

REM Compile raster shaders (.vert, .frag)
for %%e in (vert frag) do (
    for %%f in (%SHADER_DIR%\*.%%e) do (
        echo Compiling: %%~nxf
        "%GLSLC%" "%%f" -o "%OUTPUT_DIR%\%%~nf.spv" --target-env=vulkan1.3 -O
        if errorlevel 1 (
            echo FAILED: %%~nxf
            goto :error
        )
        echo   OK: %%~nf.spv
    )
)

REM Compile ray tracing shaders (.rgen, .rmiss, .rchit, .rahit, .rint)
for %%e in (rgen rmiss rchit rahit rint) do (
    for %%f in (%SHADER_DIR%\*.%%e) do (
        REM Skip shadow_anyhit.rchit — superseded by shadow_anyhit.rahit (correct any-hit stage)
        if /I "%%~nxf"=="shadow_anyhit.rchit" (
            echo   SKIPPING: %%~nxf ^(replaced by shadow_anyhit.rahit^)
        ) else (
            echo Compiling: %%~nxf
            "%GLSLC%" "%%f" -o "%OUTPUT_DIR%\%%~nf.spv" --target-env=vulkan1.3 --target-spv=spv1.4 -O
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

REM Also compile any extra top-level shaders (e.g. shaders\sculpt.comp)
REM Also compile any extra top-level shaders (e.g. raytrac_sdl2\shaders\sculpt.comp)
if exist "%~dp0shaders\sculpt.comp" (
    echo Compiling extra shader: sculpt.comp
    "%GLSLC%" "%~dp0shaders\sculpt.comp" -o "%OUTPUT_DIR%\sculpt.spv" --target-env=vulkan1.3 -O
    if errorlevel 1 (
        echo FAILED: sculpt.comp
        goto :error
    )
    echo   OK: sculpt.spv
)

REM Remove stale shader artifacts that are no longer loaded by the Vulkan backend
if exist "%OUTPUT_DIR%\gradient_test.spv" del /Q "%OUTPUT_DIR%\gradient_test.spv"
if exist "%OUTPUT_DIR%\miss.rmiss.spv" del /Q "%OUTPUT_DIR%\miss.rmiss.spv"
if exist "%OUTPUT_DIR%\raygen.rgen.spv" del /Q "%OUTPUT_DIR%\raygen.rgen.spv"
if exist "%OUTPUT_DIR%\rgen.spv" del /Q "%OUTPUT_DIR%\rgen.spv"

REM Deploy compiled .spv to runtime shaders\ subfolders.
REM Auto-creates the shaders\ subfolder under any build output that exists, so no manual copy is ever needed.
echo.
echo Deploying .spv files to runtime shaders\ subfolders...
set OUT1=%~dp0..\x64\Release
set OUT2=%~dp0..\x64\Debug
set OUT3=%~dp0..\build\Release

for %%p in ("%OUT1%" "%OUT2%" "%OUT3%") do (
    if exist %%p (
        if not exist "%%~p\shaders" mkdir "%%~p\shaders"
        echo   Copying to %%~p\shaders
        copy /Y "%OUTPUT_DIR%\*.spv" "%%~p\shaders" >nul 2>&1
        if exist "%%~p\shaders\gradient_test.spv" del /Q "%%~p\shaders\gradient_test.spv"
        if exist "%%~p\shaders\miss.rmiss.spv" del /Q "%%~p\shaders\miss.rmiss.spv"
        if exist "%%~p\shaders\raygen.rgen.spv" del /Q "%%~p\shaders\raygen.rgen.spv"
        if exist "%%~p\shaders\rgen.spv" del /Q "%%~p\shaders\rgen.spv"
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
