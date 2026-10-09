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

for %%k in (clear mark capture sweep) do (
    glslc "%SHADER_DIR%sim_sparse_viscosity_%%k.comp" -o "%SHADER_DIR%sim_sparse_viscosity_%%k.spv" --target-env=vulkan1.2
    if errorlevel 1 exit /b 1
)
glslc "%SHADER_DIR%sim_fluid_viscosity_rbgs.comp" -o "%SHADER_DIR%sim_fluid_viscosity_rbgs.spv" --target-env=vulkan1.2
if errorlevel 1 exit /b 1

for %%k in (clear mark init jacobi copy spmv axpy zpby scatter dense_clear) do (
    glslc "%SHADER_DIR%sim_sparse_pressure_%%k.comp" -o "%SHADER_DIR%sim_sparse_pressure_%%k.spv" --target-env=vulkan1.2
    if errorlevel 1 exit /b 1
)

for %%k in (sim_sparse_pressure_init_mac_weights sim_sparse_pressure_spmv_mac_weights) do (
    glslc "%SHADER_DIR%%%k.comp" -o "%SHADER_DIR%%%k.spv" --target-env=vulkan1.2
    if errorlevel 1 exit /b 1
)

glslc "%SHADER_DIR%sim_matter_partition.comp" -o "%SHADER_DIR%sim_matter_partition.spv" --target-env=vulkan1.2
for %%k in (sim_grain_mpm_count sim_grain_fluid_clear sim_grain_fluid_hash sim_grain_fluid_solid sim_grain_fluid_cells sim_grain_fluid_refresh sim_grain_fluid_delta sim_grain_fluid_reaction sim_grain_fluid_apply sim_grain_mpm_init sim_grain_mpm_clear sim_grain_mpm_hash sim_grain_mpm_gather sim_grain_mpm_apply sim_matter_grain_list_clear sim_matter_grain_hash sim_matter_grain_list_build sim_matter_grain_step sim_fluid_divergence_porous) do (
    glslc "%SHADER_DIR%%%k.comp" -o "%SHADER_DIR%%%k.spv" --target-env=vulkan1.2
    if errorlevel 1 exit /b 1
)
if errorlevel 1 exit /b 1

for %%k in (
    sim_sparse_grid_map_clear sim_sparse_grid_map_seed sim_sparse_grid_retire sim_sparse_grid_remap
    sim_sparse_mac_clear sim_sparse_mac_mark sim_sparse_mac_reset sim_sparse_mac_normalize sim_sparse_mac_publish sim_sparse_mac_capture sim_sparse_mac_gather sim_sparse_mac_p2g sim_sparse_mac_matter_p2g sim_sparse_mac_g2p sim_sparse_mac_matter_g2p sim_sparse_mac_zero_faces sim_sparse_mac_matter_zero_faces sim_sparse_mac_divergence sim_sparse_mac_divergence_var sim_sparse_mac_divergence_porous sim_sparse_mac_subtract_gradient sim_sparse_mac_subtract_gradient_var sim_sparse_mac_capture_compact sim_sparse_mac_advect sim_sparse_mac_matter_advect sim_sparse_mac_matter_contact sim_sparse_mac_viscosity_capture sim_sparse_mac_viscosity_sweep
    sim_matter_p2g sim_matter_g2p sim_matter_stress_update sim_matter_stress_p2g sim_matter_settle sim_matter_advect sim_matter_occupancy sim_matter_contact sim_matter_copy sim_matter_clear sim_matter_zero_faces sim_matter_pores
) do (
    glslc "%SHADER_DIR%%%k.comp" -o "%SHADER_DIR%%%k.spv" --target-env=vulkan1.2
    if errorlevel 1 exit /b 1
)

glslc "%SHADER_DIR%sim_fluid_clear_float.comp"        -o "%SHADER_DIR%sim_fluid_clear_float.spv"        --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_particle_forces.comp"    -o "%SHADER_DIR%sim_fluid_particle_forces.spv"    --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_p2g_scatter.comp"        -o "%SHADER_DIR%sim_fluid_p2g_scatter.spv"        --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_p2g_normalize.comp"      -o "%SHADER_DIR%sim_fluid_p2g_normalize.spv"      --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_normalize_window.comp"   -o "%SHADER_DIR%sim_fluid_normalize_window.spv"   --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_density_splat.comp"      -o "%SHADER_DIR%sim_fluid_density_splat.spv"      --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_density_clear.comp"     -o "%SHADER_DIR%sim_fluid_density_clear.spv"     --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_fluid_surface_combustion.comp" -o "%SHADER_DIR%sim_fluid_surface_combustion.spv" --target-env=vulkan1.2
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
glslc "%SHADER_DIR%sim_grid_maccormack_scalar.comp"   -o "%SHADER_DIR%sim_grid_maccormack_scalar.spv"   --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_grid_maccormack_velocity.comp" -o "%SHADER_DIR%sim_grid_maccormack_velocity.spv" --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_gas_inject.comp"               -o "%SHADER_DIR%sim_gas_inject.spv"               --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_gas_buoyancy.comp"             -o "%SHADER_DIR%sim_gas_buoyancy.spv"             --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_gas_combustion.comp"           -o "%SHADER_DIR%sim_gas_combustion.spv"           --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_gas_divergence.comp"           -o "%SHADER_DIR%sim_gas_divergence.spv"           --target-env=vulkan1.2
glslc "%SHADER_DIR%sim_gas_majorant.comp"             -o "%SHADER_DIR%sim_gas_majorant.spv"             --target-env=vulkan1.2
glslc "%SHADER_DIR%terrain_snow_solver.comp"           -o "%SHADER_DIR%terrain_snow_solver.spv"           --target-env=vulkan1.2
glslc "%SHADER_DIR%terrain_hydraulic_droplet.comp"     -o "%SHADER_DIR%terrain_hydraulic_droplet.spv"     --target-env=vulkan1.2
glslc "%SHADER_DIR%terrain_edge_preservation.comp"     -o "%SHADER_DIR%terrain_edge_preservation.spv"     --target-env=vulkan1.2

if errorlevel 1 (
    echo FAILED: One or more shaders did not compile.
    exit /b 1
)

call "%SHADER_DIR%compile_fluid_window_shaders.bat" glslc
if errorlevel 1 exit /b 1
echo OK: All simulation shaders compiled successfully.
