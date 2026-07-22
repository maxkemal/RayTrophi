@echo off
:: Compiles simulation compute GLSL shaders to SPIR-V.
:: Requires glslc (from Vulkan SDK) in PATH.
:: Run once before using the GPU (Vulkan) simulation backend.

where glslc >nul 2>&1
if errorlevel 1 (
    echo ERROR: glslc not found. Install Vulkan SDK and add it to PATH.
    echo   Download: https://vulkan.lunarg.com/sdk/home
    exit /b 1
)

set SHADER_DIR=%~dp0
echo Compiling simulation compute shaders...

glslc "%SHADER_DIR%sim_fluid_clear_float.comp"        -o "%SHADER_DIR%sim_fluid_clear_float.spv"        --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_particle_forces.comp"    -o "%SHADER_DIR%sim_fluid_particle_forces.spv"    --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_p2g_scatter.comp"        -o "%SHADER_DIR%sim_fluid_p2g_scatter.spv"        --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_p2g_normalize.comp"      -o "%SHADER_DIR%sim_fluid_p2g_normalize.spv"      --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_density_splat.comp"      -o "%SHADER_DIR%sim_fluid_density_splat.spv"      --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_density_clear.comp"     -o "%SHADER_DIR%sim_fluid_density_clear.spv"     --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_g2p.comp"                -o "%SHADER_DIR%sim_fluid_g2p.spv"                --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_free_surface_sor.comp"   -o "%SHADER_DIR%sim_fluid_free_surface_sor.spv"   --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_divergence.comp"         -o "%SHADER_DIR%sim_fluid_divergence.spv"         --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_subtract_gradient.comp"  -o "%SHADER_DIR%sim_fluid_subtract_gradient.spv"  --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_cg_build_diag.comp"      -o "%SHADER_DIR%sim_fluid_cg_build_diag.spv"      --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_cg_residual_init.comp"   -o "%SHADER_DIR%sim_fluid_cg_residual_init.spv"   --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_cg_spmv.comp"            -o "%SHADER_DIR%sim_fluid_cg_spmv.spv"            --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_cg_jacobi.comp"          -o "%SHADER_DIR%sim_fluid_cg_jacobi.spv"          --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_cg_copy.comp"            -o "%SHADER_DIR%sim_fluid_cg_copy.spv"            --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_cg_axpy.comp"            -o "%SHADER_DIR%sim_fluid_cg_axpy.spv"            --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_cg_zpby.comp"            -o "%SHADER_DIR%sim_fluid_cg_zpby.spv"            --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_cg_dot.comp"             -o "%SHADER_DIR%sim_fluid_cg_dot.spv"             --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_divergence.comp"          -o "%SHADER_DIR%sim_grid_divergence.spv"          --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_sor.comp"                 -o "%SHADER_DIR%sim_grid_sor.spv"                 --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_subtract_gradient.comp"   -o "%SHADER_DIR%sim_grid_subtract_gradient.spv"   --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_advect_scalar.comp"       -o "%SHADER_DIR%sim_grid_advect_scalar.spv"       --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_advect_velocity.comp"     -o "%SHADER_DIR%sim_grid_advect_velocity.spv"     --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_velocity_dissipate.comp"  -o "%SHADER_DIR%sim_grid_velocity_dissipate.spv"  --target-env=vulkan1.2
glslc "%SHADER_DIR%terrain_snow_solver.comp"           -o "%SHADER_DIR%terrain_snow_solver.spv"           --target-env=vulkan1.2
glslc "%SHADER_DIR%terrain_hydraulic_droplet.comp"     -o "%SHADER_DIR%terrain_hydraulic_droplet.spv"     --target-env=vulkan1.2
glslc "%SHADER_DIR%terrain_edge_preservation.comp"     -o "%SHADER_DIR%terrain_edge_preservation.spv"     --target-env=vulkan1.2

if errorlevel 1 (
    echo FAILED: One or more shaders did not compile.
    exit /b 1
)

echo OK: All simulation shaders compiled successfully.
