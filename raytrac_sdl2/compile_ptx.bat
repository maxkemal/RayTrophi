@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo ========================================
echo   RayTrophi PTX Compiler (Relative Paths)
echo   Supports: GTX 9xx, 10xx, 16xx, RTX
echo ========================================
echo.

:: Get the directory where this script resides (with trailing backslash)
set "ROOT_DIR=%~dp0"
:: Remove trailing backslash for consistency if needed, but paths usually handle double slashes ok.

set "SRC_ROOT=%ROOT_DIR%source\src"
set "DEVICE_DIR=%SRC_ROOT%\Device"
set "INCLUDE_ROOT=%ROOT_DIR%source\include"
set "OUTPUT_DIR=%ROOT_DIR%"

:: VCPKG and OptiX paths are usually absolute system paths, but if VCPKG is local, we can try relative too.
:: Assuming vcpkg is at ../vcpkg relative to project root? Or at E:\RayTrophi_projesi\vcpkg ?
:: Based on previous absolute path: E:\RayTrophi_projesi\raytracing_Proje_Moduler\vcpkg
:: So it is "../vcpkg" relative to "raytrac_sdl2" folder (ROOT_DIR).
set "VCPKG_INCLUDE=%ROOT_DIR%..\vcpkg\installed\x64-windows\include"

:: OptiX SDK is system-wide, so keeping it absolute is safer unless you vendor it.
set "OPTIX_INCLUDE=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include"

echo Source Dir: %SRC_ROOT%
echo Device Dir: %DEVICE_DIR%
echo Output Dir: %OUTPUT_DIR%
echo.

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "%DEVICE_DIR%\raygen.cu" ^
-o "%OUTPUT_DIR%raygen.ptx" ^
-I"%OPTIX_INCLUDE%" ^
-I"%INCLUDE_ROOT%" ^
-I"%VCPKG_INCLUDE%" ^
-I"%SRC_ROOT%" ^
-I"%DEVICE_DIR%" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "%DEVICE_DIR%\erosion_kernels.cu" ^
-o "%OUTPUT_DIR%erosion_kernels.ptx" ^
-I"%OPTIX_INCLUDE%" ^
-I"%INCLUDE_ROOT%" ^
-I"%VCPKG_INCLUDE%" ^
-I"%SRC_ROOT%" ^
-I"%DEVICE_DIR%" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "%DEVICE_DIR%\foliage_deform.cu" ^
-o "%OUTPUT_DIR%foliage_deform.ptx" ^
-I"%OPTIX_INCLUDE%" ^
-I"%INCLUDE_ROOT%" ^
-I"%VCPKG_INCLUDE%" ^
-I"%SRC_ROOT%" ^
-I"%DEVICE_DIR%" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "%DEVICE_DIR%\gas_kernels.cu" ^
-o "%OUTPUT_DIR%gas_kernels.ptx" ^
-I"%OPTIX_INCLUDE%" ^
-I"%INCLUDE_ROOT%" ^
-I"%VCPKG_INCLUDE%" ^
-I"%SRC_ROOT%" ^
-I"%DEVICE_DIR%" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "%DEVICE_DIR%\gas_fft_solver.cu" ^
-o "%OUTPUT_DIR%gas_fft_solver.ptx" ^
-I"%OPTIX_INCLUDE%" ^
-I"%INCLUDE_ROOT%" ^
-I"%VCPKG_INCLUDE%" ^
-I"%SRC_ROOT%" ^
-I"%DEVICE_DIR%" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] PTX compiled successfully!
    echo Output: raygen.ptx, gas_kernels.ptx, gas_fft_solver.ptx
) else (
    echo.
    echo [ERROR] PTX compilation failed!
)

pause