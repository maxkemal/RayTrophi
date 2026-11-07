/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiFluidSurface.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

// fluid.surface_parity: the GPU surface builder (FluidSurfaceGpu.h) against the
// CPU reference (buildLevelSet + buildMaterialCoordinateGrid) on the SAME live
// particles, cell by cell. It is the acceptance test of the GPU route
// (docs/dev/GPU_YUZEY_LEVEL_SET.md): a surface that is "a bit different but
// looks fine" is exactly the failure a picture would not catch.
//
// Builds into temporary device buffers, so the domain's live publication (the
// device proxy the Vulkan RT backend may be reading) is never touched.

#include "RtApiInternal.h"
#include "ParticleSimulation.h"
#include "Fluid/FluidLevelSet.h"
#include "Fluid/FluidSurfaceGpu.h"
#include "Fluid/FrozenSurfaceMaterial.h"
#include "Fluid/FluidViewResolver.h"
#include "Fluid/SubstanceTag.h"
#include "Fluid/MatterPhaseGrid.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace rtapi {

Result getFluidSurfaceParity(const std::string& domain_id_or_name, FluidSurfaceParity& out) {
    out = FluidSurfaceParity{};
    if (!g_ctx) return Result::fail("API is not bound to a running context");
    FluidDomainInfo info;
    Result found = getFluidDomain(domain_id_or_name, info);
    if (!found.ok) return found;
    auto& sys = g_ctx->scene.ensureParticleSimulationSystem();
    const auto& domains = sys.gridDomains();
    const auto& states = sys.gridDomainStates();
    std::size_t index = domains.size();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        if (domains[i].name == info.name) { index = i; break; }
    }
    if (index >= domains.size() || !RayTrophiSim::simulationDomainHasLiquid(domains[index])) {
        return Result::fail("liquid (fluid) domain not found: " + domain_id_or_name);
    }
    if (index >= states.size() || !states[index].valid || states[index].particles.empty()) {
        return Result::success();  // live=false: nothing stepped yet
    }
    const auto& desc = domains[index];
    const auto& state = states[index];
    const auto& grid = RayTrophiSim::Fluid::liquidGrid(state);
    out.live = true;
    out.particles = state.particles.size();

    // The same selection the render sync gives both builders.
    const auto plan = RayTrophiSim::Fluid::resolveFluidViews(
        desc, RayTrophiSim::Fluid::cachedDistinctViewKeys(state.particles, state.version));
    const RayTrophiSim::Fluid::FluidViewSelection selection{
        &plan, RayTrophiSim::Fluid::FluidView::Surface};
    auto params = desc.fluid_level_set_params;
    params.gpu_build = true;

    using Clock = std::chrono::steady_clock;
    std::vector<float> cpu_sdf;
    std::vector<float> cpu_uvw;
    RayTrophiSim::Fluid::LevelSetStats cpu_stats;
    RayTrophiSim::Fluid::SubstanceMaterialEntry entries[
        RayTrophiSim::Fluid::kSubstanceMaterialEntryCapacity];
    const std::size_t entry_count = RayTrophiSim::Fluid::collectPhaseSurfaceMaterialEntries(
        desc, state.particles, entries);
    std::vector<float> cpu_composition;
    auto begin = Clock::now();
    RayTrophiSim::Fluid::buildLevelSet(state.particles, grid, params, cpu_sdf, &cpu_stats,
                                       &selection);
    const bool cpu_uvw_ok = RayTrophiSim::Fluid::buildMaterialCoordinateGrid(
        state.particles, grid, params, cpu_uvw, &selection);
    const bool cpu_composition_ok = RayTrophiSim::Fluid::buildCompositionGrid(
        state.particles, grid, params, entries, entry_count, desc.fluid_surface_material_id,
        cpu_composition, &selection);
    out.cpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();

    RayTrophiSim::Fluid::FluidSurfaceGpuBuffers scratch;
    RayTrophiSim::Fluid::FluidSurfaceGpuRequest request;
    request.particles = &state.particles;
    request.grid = &grid;
    request.params = params;
    request.surface_band_voxels = desc.fluid_surface_band_voxels;
    request.selection = &selection;
    request.want_uvw = true;
    request.materials = entries;
    request.material_count = entry_count;
    request.fallback_material = desc.fluid_surface_material_id;
    RayTrophiSim::Fluid::FluidSurfaceGpuResult result;
    std::vector<float> gpu_sdf;
    std::vector<float> gpu_uvw;
    std::vector<float> gpu_composition;
    auto& compute = g_ctx->scene.simulation_world.compute();
    begin = Clock::now();
    out.gpu_ok = RayTrophiSim::Fluid::buildFluidSurfaceGpuSdf(
        compute, scratch, request, gpu_sdf, &gpu_uvw, &gpu_composition, result);
    out.gpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    scratch.release(compute);
    out.gpu_status = result.status;
    if (!out.gpu_ok) return Result::success();

    out.grid[0] = cpu_stats.eff_nx;
    out.grid[1] = cpu_stats.eff_ny;
    out.grid[2] = cpu_stats.eff_nz;
    out.voxel = cpu_stats.eff_voxel;
    out.cells = cpu_sdf.size();
    out.gpu_active = result.active_cells;
    if (gpu_sdf.size() != cpu_sdf.size()) {
        out.gpu_status = "grid size mismatch: cpu " + std::to_string(cpu_sdf.size()) +
                         " gpu " + std::to_string(gpu_sdf.size());
        out.gpu_ok = false;
        return Result::success();
    }
    // Compare where either side is inside the band; both far elsewhere.
    const float far_value = std::max(
        std::max(1e-4f, params.kernel_radius_voxels * grid.voxel_size),
        params.narrow_band_voxels * grid.voxel_size);
    double sum = 0.0;
    for (std::size_t c = 0; c < cpu_sdf.size(); ++c) {
        const float a = cpu_sdf[c], g = gpu_sdf[c];
        const bool a_in = a < far_value - 1e-5f, g_in = g < far_value - 1e-5f;
        if (a_in) ++out.cpu_active;
        if (!a_in && !g_in) continue;
        if (a_in != g_in) ++out.band_mismatch;
        const double d = std::fabs(static_cast<double>(a) - static_cast<double>(g));
        ++out.compared_cells;
        sum += d;
        out.max_abs_phi = std::max(out.max_abs_phi, d);
        // A surface on opposite sides: the visible failure.
        if ((a < 0.0f) != (g < 0.0f) && std::fabs(a) > 1e-4f * grid.voxel_size &&
            std::fabs(g) > 1e-4f * grid.voxel_size) {
            ++out.sign_mismatch;
        }
    }
    out.mean_abs_phi = out.compared_cells ? sum / static_cast<double>(out.compared_cells) : 0.0;
    out.max_abs_phi_over_h = out.voxel > 0.0f ? out.max_abs_phi / out.voxel : 0.0;
    out.uvw_compared = cpu_uvw_ok && result.uvw_built && cpu_uvw.size() == gpu_uvw.size();
    if (out.uvw_compared) {
        for (std::size_t i = 0; i < cpu_uvw.size(); ++i) {
            out.uvw_max_abs = std::max(out.uvw_max_abs,
                std::fabs(static_cast<double>(cpu_uvw[i]) - static_cast<double>(gpu_uvw[i])));
        }
    }
    // Composition: both must agree on whether a mixture exists at all, then on
    // the material pair per cell (the visible part) and the blend weight.
    out.composition_cpu = cpu_composition_ok;
    out.composition_gpu = result.composition_built;
    if (cpu_composition_ok && result.composition_built &&
        cpu_composition.size() == gpu_composition.size()) {
        for (std::size_t c = 0; c + 2 < cpu_composition.size(); c += 3) {
            if (cpu_composition[c] != gpu_composition[c] ||
                cpu_composition[c + 1] != gpu_composition[c + 1]) {
                ++out.composition_slot_mismatch;
            }
            out.composition_weight_max_abs = std::max(out.composition_weight_max_abs,
                std::fabs(static_cast<double>(cpu_composition[c + 2]) -
                          static_cast<double>(gpu_composition[c + 2])));
        }
    }
    out.upload_bytes = result.upload_bytes;
    out.download_bytes = result.download_bytes;
    return Result::success();
}

} // namespace rtapi
