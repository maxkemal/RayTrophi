/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiFluid.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

#include "RtApiInternal.h"
#include "Fluid/FluidObject.h"
#include "Fluid/FluidSimulationSystem.h"
#include "Fluid/APICFluidSolver.h"
#include "ParticleSimulation.h"
#include <algorithm>
#include <cctype>

namespace rtapi {

Result createFluidDomain(const std::string& name, Vec3 domain_min, Vec3 domain_max,
                         float voxel_size, const std::string& type, rtapi::FluidDomainInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    std::string domain_name = name.empty() ? "Fluid" : name;
    std::string dom_type = type;
    std::transform(dom_type.begin(), dom_type.end(), dom_type.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    const bool is_gas = (dom_type == "gas" || dom_type == "smoke" || dom_type == "fire");

    // 1. Ensure high-level Physics UI Panel domain descriptor exists
    auto& p_sys = g_ctx->scene.ensureParticleSimulationSystem();
    auto& domains = p_sys.gridDomains();
    RayTrophiSim::SimulationGridDomainDesc* grid_dom = nullptr;
    for (auto& d : domains) {
        if (d.name == domain_name) { grid_dom = &d; break; }
    }
    if (!grid_dom) {
        RayTrophiSim::SimulationGridDomainDesc desc;
        desc.name = domain_name;
        desc.type = is_gas ? RayTrophiSim::SimulationDomainType::Gas : RayTrophiSim::SimulationDomainType::Fluid;
        desc.boundary_mode = is_gas ? RayTrophiSim::SimulationGridDomainBoundaryMode::Open : RayTrophiSim::SimulationGridDomainBoundaryMode::Closed;
        desc.bounds_min = domain_min;
        desc.bounds_max = domain_max;
        if (voxel_size > 0.001f) desc.voxel_size = voxel_size;
        desc.enabled = true;
        g_ctx->scene.addSimulationGridDomain(desc);
    } else {
        grid_dom->type = is_gas ? RayTrophiSim::SimulationDomainType::Gas : RayTrophiSim::SimulationDomainType::Fluid;
        grid_dom->boundary_mode = is_gas ? RayTrophiSim::SimulationGridDomainBoundaryMode::Open : RayTrophiSim::SimulationGridDomainBoundaryMode::Closed;
        grid_dom->bounds_min = domain_min;
        grid_dom->bounds_max = domain_max;
        if (voxel_size > 0.001f) grid_dom->voxel_size = voxel_size;
    }

    // 2. Ensure low-level FluidObject exists
    auto existing = g_ctx->scene.findFluidObjectByName(domain_name);
    RayTrophiSim::Fluid::FluidObject* obj = existing;
    if (!obj) {
        obj = g_ctx->scene.addFluidObject(domain_name);
    }
    if (!obj) return Result::fail("failed to create fluid domain: " + domain_name);

    obj->params.boundary = is_gas ? RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open : RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Closed;
    obj->domain_min = domain_min;
    obj->domain_max = domain_max;
    if (voxel_size > 0.001f) obj->voxel_size = voxel_size;
    obj->grid_dirty = true;
    obj->ensureGrid();

    out_info.id = obj->id;
    out_info.name = obj->name;
    out_info.type = is_gas ? "gas" : "fluid";
    out_info.domain_min = obj->domain_min;
    out_info.domain_max = obj->domain_max;
    out_info.voxel_size = obj->voxel_size;
    out_info.particle_count = obj->particles.size();
    out_info.render_mode = (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
                           (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
    out_info.boundary = (obj->params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open) ? "open" :
                        (obj->params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Periodic) ? "periodic" : "closed";
    out_info.preset = (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Water) ? "water" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Oil) ? "oil" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Mud) ? "mud" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Honey) ? "honey" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Lava) ? "lava" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Sand) ? "sand" : "custom";
    out_info.viscosity = obj->params.viscosity;
    out_info.backend = (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
    out_info.enabled = obj->enabled;
    out_info.visible = obj->visible;
    return Result::success();
}

Result getFluidDomain(const std::string& domain_id_or_name, rtapi::FluidDomainInfo& out_info) {
    if (!g_ctx) return notBound();

    RayTrophiSim::Fluid::FluidObject* obj = nullptr;
    try {
        size_t idx = 0;
        uint32_t id = static_cast<uint32_t>(std::stoul(domain_id_or_name, &idx));
        if (idx == domain_id_or_name.size()) {
            for (auto& fo : g_ctx->scene.fluid_objects) {
                if (fo.id == id) { obj = &fo; break; }
            }
        }
    } catch (...) {}

    if (!obj) obj = g_ctx->scene.findFluidObjectByName(domain_id_or_name);
    if (!obj) return Result::fail("fluid domain not found: " + domain_id_or_name);

    RayTrophiSim::SimulationGridDomainDesc* grid_dom = nullptr;
    auto& p_sys_get = g_ctx->scene.ensureParticleSimulationSystem();
    for (auto& d : p_sys_get.gridDomains()) {
        if (d.name == obj->name) { grid_dom = &d; break; }
    }

    out_info.id = obj->id;
    out_info.name = obj->name;
    out_info.type = (grid_dom && grid_dom->type == RayTrophiSim::SimulationDomainType::Gas) ? "gas" : "fluid";
    out_info.domain_min = obj->domain_min;
    out_info.domain_max = obj->domain_max;
    out_info.voxel_size = obj->voxel_size;
    out_info.particle_count = obj->particles.size();
    out_info.render_mode = (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
                           (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
    out_info.boundary = (obj->params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open) ? "open" :
                        (obj->params.boundary == RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Periodic) ? "periodic" : "closed";
    out_info.preset = (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Water) ? "water" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Oil) ? "oil" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Mud) ? "mud" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Honey) ? "honey" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Lava) ? "lava" :
                      (obj->params.current_preset == RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Sand) ? "sand" : "custom";
    out_info.viscosity = obj->params.viscosity;
    out_info.backend = (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
    out_info.enabled = obj->enabled;
    out_info.visible = obj->visible;
    return Result::success();
}

Result removeFluidDomain(const std::string& domain_id_or_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    rtapi::FluidDomainInfo info;
    if (!getFluidDomain(domain_id_or_name, info).ok) {
        return Result::fail("fluid domain not found: " + domain_id_or_name);
    }

    g_ctx->scene.removeFluidObject(info.id);
    return Result::success();
}

Result seedFluidParticles(const std::string& domain_id_or_name, Vec3 seed_min, Vec3 seed_max,
                           int particles_per_cell, bool replace) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    rtapi::FluidDomainInfo info;
    if (!getFluidDomain(domain_id_or_name, info).ok) {
        return Result::fail("fluid domain not found: " + domain_id_or_name);
    }

    RayTrophiSim::Fluid::FluidObject* obj = nullptr;
    for (auto& fo : g_ctx->scene.fluid_objects) {
        if (fo.id == info.id) { obj = &fo; break; }
    }
    if (!obj) return Result::fail("fluid domain not found");

    obj->seed_min = seed_min;
    obj->seed_max = seed_max;
    obj->seed_particles_per_cell = std::max(1, particles_per_cell);
    obj->replace_on_seed = replace;
    obj->pending_seed = true;

    obj->ensureGrid();
    if (replace) obj->particles.clear();

    RayTrophiSim::Fluid::seedBox(obj->particles, obj->grid, seed_min, seed_max, std::max(1, particles_per_cell));

    // Also update ParticleSimulationSystem's SimulationGridDomainDesc for UI Physics sync!
    auto& p_sys = g_ctx->scene.ensureParticleSimulationSystem();
    for (auto& d : p_sys.gridDomains()) {
        if (d.name == obj->name) {
            d.fluid_seed_min = seed_min;
            d.fluid_seed_max = seed_max;
            d.fluid_seed_particles_per_cell = std::max(1, particles_per_cell);
            d.fluid_replace_on_seed = replace;
            d.fluid_pending_seed = true;
            break;
        }
    }

    return Result::success();
}

Result clearFluidParticles(const std::string& domain_id_or_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    rtapi::FluidDomainInfo info;
    if (!getFluidDomain(domain_id_or_name, info).ok) {
        return Result::fail("fluid domain not found: " + domain_id_or_name);
    }

    for (auto& fo : g_ctx->scene.fluid_objects) {
        if (fo.id == info.id) {
            fo.resetState();
            break;
        }
    }
    return Result::success();
}

Result updateFluidDomain(const std::string& domain_id_or_name,
                         const Vec3* domain_min, const Vec3* domain_max,
                         const float* voxel_size, const std::string* render_mode,
                         const std::string* backend, const std::string* boundary,
                         const std::string* preset, const float* viscosity,
                         const bool* enabled, const bool* visible) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    rtapi::FluidDomainInfo info;
    if (!getFluidDomain(domain_id_or_name, info).ok) {
        return Result::fail("fluid domain not found: " + domain_id_or_name);
    }

    RayTrophiSim::Fluid::FluidObject* obj = nullptr;
    for (auto& fo : g_ctx->scene.fluid_objects) {
        if (fo.id == info.id) { obj = &fo; break; }
    }
    if (!obj) return Result::fail("fluid domain not found");

    if (domain_min) { obj->domain_min = *domain_min; obj->grid_dirty = true; }
    if (domain_max) { obj->domain_max = *domain_max; obj->grid_dirty = true; }
    if (voxel_size && *voxel_size > 0.001f) { obj->voxel_size = *voxel_size; obj->grid_dirty = true; }

    if (render_mode) {
        std::string rm = *render_mode;
        std::transform(rm.begin(), rm.end(), rm.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (rm == "surface") obj->render_mode = RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
        else if (rm == "particles") obj->render_mode = RayTrophiSim::Fluid::FluidRenderMode::Particles;
        else obj->render_mode = RayTrophiSim::Fluid::FluidRenderMode::Volume;
    }

    if (preset) {
        std::string p = *preset;
        std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (p == "water") obj->params.applyPreset(RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Water);
        else if (p == "oil") obj->params.applyPreset(RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Oil);
        else if (p == "mud") obj->params.applyPreset(RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Mud);
        else if (p == "honey") obj->params.applyPreset(RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Honey);
        else if (p == "lava") obj->params.applyPreset(RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Lava);
        else if (p == "sand") obj->params.applyPreset(RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Sand);
    }

    if (boundary) {
        std::string b = *boundary;
        std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (b == "open") obj->params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open;
        else if (b == "periodic") obj->params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Periodic;
        else obj->params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Closed;
    }

    if (viscosity) {
        obj->params.viscosity = std::max(0.0f, *viscosity);
    }

    if (backend) {
        std::string dev = *backend;
        std::transform(dev.begin(), dev.end(), dev.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        RayTrophiSim::SimulationDomainBackend be = RayTrophiSim::SimulationDomainBackend::CPU_Dense;
        if (dev == "gpu" || dev == "gpu_compute" || dev == "cuda" || dev == "compute") {
            be = RayTrophiSim::SimulationDomainBackend::GPU_Compute;
        } else if (dev == "vulkan" || dev == "gpu_vulkan") {
            be = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
        } else if (dev == "cpu_sparse" || dev == "sparse" || dev == "vdb") {
            be = RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB;
        }

        auto& p_sys_upd = g_ctx->scene.ensureParticleSimulationSystem();
        for (auto& d : p_sys_upd.gridDomains()) {
            if (d.name == obj->name) {
                d.backend = be;
                break;
            }
        }
    }

    if (enabled) obj->enabled = *enabled;
    if (visible) obj->visible = *visible;

    obj->ensureGrid();
    return Result::success();
}

Result resetFluidSimulation() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    g_ctx->scene.ensureFluidSimulationSystem();
    for (auto& fo : g_ctx->scene.fluid_objects) {
        fo.resetState();
    }
    return Result::success();
}

Result stepFluidSimulation(float dt) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (dt <= 0.0f) dt = 0.0166667f;

    g_ctx->scene.ensureFluidSimulationSystem();
    if (g_ctx->scene.fluid_simulation_system) {
        RayTrophiSim::SimulationContext simCtx = g_ctx->scene.simulation_world.makeContext(dt, 0, 1);
        simCtx.dt = dt;
        g_ctx->scene.fluid_simulation_system->step(simCtx);
    }
    return Result::success();
}

} // namespace rtapi
