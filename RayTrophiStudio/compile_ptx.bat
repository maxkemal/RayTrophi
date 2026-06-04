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
-ptx "%~dp0source\src\Device\raygen.cu" ^
-o "%~dp0raygen.ptx" ^
-I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include" ^
-I"%~dp0source\include" ^
-I"%~dp0libs" ^
-I"%~dp0source\src\Device" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

echo.
echo Compiling miss_kernels.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "%~dp0source\src\Device\miss_kernels.cu" ^
-o "%~dp0miss_kernels.ptx" ^
-I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include" ^
-I"%~dp0source\include" ^
-I"%~dp0libs" ^
-I"%~dp0source\src\Device" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

echo.
echo Compiling hitgroup_kernels.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "%~dp0source\src\Device\hitgroup_kernels.cu" ^
-o "%~dp0hitgroup_kernels.ptx" ^
-I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0\include" ^
-I"%~dp0source\include" ^
-I"%~dp0libs" ^
-I"%~dp0source\src\Device" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50

echo.
echo Compiling erosion_kernels.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "%~dp0source\src\Device\erosion_kernels.cu" ^
-o "%~dp0erosion_kernels.ptx" ^
-I"%~dp0source\include" ^
-I"%~dp0libs" ^
-I"%~dp0source\src\Device" ^
--ptxas-options=-v ^
--use_fast_math ^
-gencode=arch=compute_50,code=compute_50


echo.
echo Compiling foliage_deform.cu...
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "%~dp0source\src\Device\foliage_deform.cu" ^
-o "%~dp0foliage_deform.ptx" ^
-I"%~dp0source\include" ^
-I"%~dp0libs" ^
-I"%~dp0source\src\Device" ^
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

echo.
echo Deploying PTX files to runtime directories...
set DEPLOY1=%~dp0..\x64\Release
set DEPLOY2=%~dp0..\x64\Debug
set DEPLOY3=%~dp0

for %%d in ("%DEPLOY1%" "%DEPLOY2%" "%DEPLOY3%") do (
    if exist %%d (
        echo   Copying to %%d
        copy /Y "%~dp0raygen.ptx" %%d >nul 2>&1
        copy /Y "%~dp0miss_kernels.ptx" %%d >nul 2>&1
        copy /Y "%~dp0hitgroup_kernels.ptx" %%d >nul 2>&1
        copy /Y "%~dp0erosion_kernels.ptx" %%d >nul 2>&1
        copy /Y "%~dp0foliage_deform.ptx" %%d >nul 2>&1
    )
)
echo Deploy complete.
pause
