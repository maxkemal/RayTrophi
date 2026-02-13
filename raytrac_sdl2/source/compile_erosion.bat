@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo ========================================
echo   RayTrophi Erosion PTX Compiler
echo ========================================
echo.

echo Compiling erosion_kernels.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device\erosion_kernels.cu" ^
-o "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\erosion_kernels.ptx" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\include" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device" ^
--ptxas-options=-v ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] erosion_kernels.ptx compiled successfully!
) else (
    echo [ERROR] erosion_kernels.ptx compilation failed!
    pause
    exit /b %ERRORLEVEL%
)

pause
