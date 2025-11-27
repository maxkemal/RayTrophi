@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
-ptx "E:\visual studio proje c++\raytracing_Proje_Moduler\raytrac_sdl2\source\cpp_file\raygen.cu" ^
-o "E:\visual studio proje c++\raytracing_Proje_Moduler\raytrac_sdl2\raygen.ptx" ^
-I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0\include" ^
-I"E:\visual studio proje c++\raytracing_Proje_Moduler\raytrac_sdl2\source\header" ^
--ptxas-options=-v ^
--maxrregcount=64 ^
--use_fast_math ^
-gencode=arch=compute_86,code=compute_86 ^
-allow-unsupported-compiler

pause
