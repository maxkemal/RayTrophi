@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo ========================================
echo   RayTrophi PTX Compiler
echo   Supports: GTX 9xx, 10xx, 16xx, RTX
echo ========================================
echo.

echo.
echo Compiling raygen_kernels.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device\raygen.cu" ^
-o "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\raygen.ptx" ^
-I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\include" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\libs" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

echo.
echo Compiling miss_kernels.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device\miss_kernels.cu" ^
-o "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\miss_kernels.ptx" ^
-I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\include" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\libs" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

echo.
echo Compiling hitgroup_kernels.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device\hitgroup_kernels.cu" ^
-o "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\hitgroup_kernels.ptx" ^
-I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\include" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\libs" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

echo.
echo Compiling erosion_kernels.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device\erosion_kernels.cu" ^
-o "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\erosion_kernels.ptx" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\include" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\libs" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device" ^
--ptxas-options=-v ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50


echo.
echo Compiling foliage_deform.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\src\Device\foliage_deform.cu" ^
-o "e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\foliage_deform.ptx" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\include" ^
-I"e:\RayTrophi_projesi\raytracing_Proje_Moduler\raytrac_sdl2\source\libs" ^
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

echo.
echo ========================================
echo   All PTX files compiled successfully!
echo ========================================
pause
