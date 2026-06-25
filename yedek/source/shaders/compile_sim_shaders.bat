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
glslc "%SHADER_DIR%sim_grid_divergence.comp"          -o "%SHADER_DIR%sim_grid_divergence.spv"          --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_sor.comp"                 -o "%SHADER_DIR%sim_grid_sor.spv"                 --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_subtract_gradient.comp"   -o "%SHADER_DIR%sim_grid_subtract_gradient.spv"   --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_advect_scalar.comp"       -o "%SHADER_DIR%sim_grid_advect_scalar.spv"       --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_advect_velocity.comp"     -o "%SHADER_DIR%sim_grid_advect_velocity.spv"     --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_velocity_dissipate.comp"  -o "%SHADER_DIR%sim_grid_velocity_dissipate.spv"  --target-env=vulkan1.2

if errorlevel 1 (
    echo FAILED: One or more shaders did not compile.
    exit /b 1
)

echo OK: All simulation shaders compiled successfully.
