@echo off
setlocal
set "RT_WINDOW_GLSLC=%~1"
if "%RT_WINDOW_GLSLC%"=="" set "RT_WINDOW_GLSLC=glslc"

REM Compile from the same source as the full-grid shader; equations stay shared.
for %%k in (
    sim_fluid_divergence sim_fluid_divergence_var
    sim_fluid_cg_build_diag sim_fluid_cg_build_diag_var
    sim_fluid_cg_spmv sim_fluid_cg_spmv_var
    sim_fluid_cg_jacobi sim_fluid_cg_copy sim_fluid_cg_axpy sim_fluid_cg_zpby
    sim_fluid_cg_dot sim_fluid_cg_axpy_dev sim_fluid_cg_zpby_dev
    sim_fluid_cg_jacobi_dot sim_fluid_cg_spmv_dot sim_fluid_cg_spmv_dot_var
    sim_fluid_cg_axpy2_dev
) do (
    "%RT_WINDOW_GLSLC%" "%~dp0%%k.comp" -DRT_FLUID_WINDOW=1 -o "%~dp0%%k_window.spv" --target-env=vulkan1.2 -O
    if errorlevel 1 exit /b 1
)
exit /b 0
