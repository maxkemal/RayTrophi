@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo ========================================
echo   RayTrophi PTX Compiler
echo   Supports: GTX 9xx, 10xx, 16xx, RTX
echo ========================================
echo.

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "E:\visual studio proje c++\raytracing_Proje_Moduler\raytrac_sdl2\source\src\raygen.cu" ^
-o "E:\visual studio proje c++\raytracing_Proje_Moduler\raytrac_sdl2\raygen.ptx" ^
-I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include" ^
-I"E:\visual studio proje c++\raytracing_Proje_Moduler\raytrac_sdl2\source\include" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] PTX compiled successfully!
    echo Output: raygen.ptx
    echo Compatible with: SM 5.0+ (Maxwell, Pascal, Turing, Ampere, Ada Lovelace)
) else (
    echo.
    echo [ERROR] PTX compilation failed!
)

pause
