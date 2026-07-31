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
        // Production gas prefers the backend-neutral GPU compute route. Scene
        // synchronization selects CUDA first and Vulkan second, and retains
        // the deterministic CPU fallback when neither GPU backend is usable.
        if (is_gas) {
            desc.backend = RayTrophiSim::SimulationDomainBackend::GPU_Compute;
        }
        desc.boundary_mode = is_gas ? RayTrophiSim::SimulationGridDomainBoundaryMode::Open : RayTrophiSim::SimulationGridDomainBoundaryMode::Closed;
        desc.bounds_min = domain_min;
        desc.bounds_max = domain_max;
        if (voxel_size > 0.001f) desc.voxel_size = voxel_size;
        desc.enabled = true;
        g_ctx->scene.addSimulationGridDomain(desc);
        // addSimulationGridDomain may reallocate the descriptor vector; resolve
        // the pointer again so the returned API state reports the real backend.
        for (auto& d : p_sys.gridDomains()) {
            if (d.name == domain_name) { grid_dom = &d; break; }
        }
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

    bool grid_removed = false;
    for (std::size_t si = 0; si < g_ctx->scene.particle_systems.size() && !grid_removed; ++si) {
        auto& system = g_ctx->scene.particle_systems[si];
        if (!system.runtime) continue;
        const auto& domains = system.runtime->gridDomains();
        for (std::size_t di = 0; di < domains.size(); ++di) {
            if (domains[di].name == info.name) {
                grid_removed = g_ctx->scene.removeSimulationGridDomain(si, di);
                break;
            }
        }
    }
    const bool object_removed = g_ctx->scene.removeFluidObject(info.id);
    if (!grid_removed && !object_removed)
        return Result::fail("failed to remove fluid domain: " + info.name);
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
            if (replace) d.fluid_reseed_on_reset = true;
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

    auto& p_sys_upd = g_ctx->scene.ensureParticleSimulationSystem();
    RayTrophiSim::SimulationGridDomainDesc* grid_dom = nullptr;
    for (auto& d : p_sys_upd.gridDomains()) {
        if (d.name == obj->name) { grid_dom = &d; break; }
    }
    if (grid_dom) {
        if (domain_min) grid_dom->bounds_min = *domain_min;
        if (domain_max) grid_dom->bounds_max = *domain_max;
        if (voxel_size && *voxel_size > 0.001f) grid_dom->voxel_size = *voxel_size;
    }

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

        if (grid_dom) grid_dom->backend = be;
    }

    if (enabled) obj->enabled = *enabled;
    if (visible) obj->visible = *visible;
    if (grid_dom && enabled) grid_dom->enabled = *enabled;

    obj->ensureGrid();
    return Result::success();
}

Result getGasDomainSettings(const std::string& domain_id_or_name, GasDomainSettings& out) {
    if (!g_ctx) return notBound();
    FluidDomainInfo info;
    Result found = getFluidDomain(domain_id_or_name, info);
    if (!found.ok) return found;

    auto& domains = g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it = std::find_if(domains.begin(), domains.end(),
        [&info](const auto& d) { return d.name == info.name; });
    if (it == domains.end() || it->type != RayTrophiSim::SimulationDomainType::Gas)
        return Result::fail("gas domain not found: " + domain_id_or_name);

    switch (it->quality_profile) {
        case RayTrophiSim::SimulationDomainQualityProfile::Interactive: out.quality_profile = "interactive"; break;
        case RayTrophiSim::SimulationDomainQualityProfile::Preview: out.quality_profile = "preview"; break;
        case RayTrophiSim::SimulationDomainQualityProfile::Final: out.quality_profile = "final"; break;
        case RayTrophiSim::SimulationDomainQualityProfile::Cinema: out.quality_profile = "cinema"; break;
        default: out.quality_profile = "custom"; break;
    }
    out.resource_budget_mb = it->resource_budget_mb;
    out.enforce_resource_budget = it->enforce_resource_budget;
    out.use_sparse_tiles = it->use_sparse_tiles;
    out.render_to_nanovdb = it->render_to_nanovdb;
    out.fire_enabled = it->fire_enabled;
    out.ignition_temperature = it->ignition_temperature;
    out.burn_rate = it->burn_rate;
    out.heat_release = it->heat_release;
    out.smoke_generation = it->smoke_generation;
    out.flame_dissipation = it->flame_dissipation;
    out.fire_max_temperature = it->fire_max_temperature;
    out.buoyancy_heat = it->gas_buoyancy_heat;
    out.buoyancy_density = it->gas_buoyancy_density;
    out.vorticity = it->gas_vorticity;
    out.fire_expansion = it->fire_expansion;
    out.turbulence_strength = it->turbulence_strength;
    out.turbulence_scale = it->turbulence_scale;
    out.turbulence_octaves = it->turbulence_octaves;
    out.turbulence_lacunarity = it->turbulence_lacunarity;
    out.turbulence_persistence = it->turbulence_persistence;
    out.turbulence_speed = it->turbulence_speed;
    return Result::success();
}

Result updateGasDomainSettings(const std::string& domain_id_or_name, const GasDomainSettings& s) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    FluidDomainInfo info;
    Result found = getFluidDomain(domain_id_or_name, info);
    if (!found.ok) return found;

    auto& domains = g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it = std::find_if(domains.begin(), domains.end(),
        [&info](const auto& d) { return d.name == info.name; });
    if (it == domains.end() || it->type != RayTrophiSim::SimulationDomainType::Gas)
        return Result::fail("gas domain not found: " + domain_id_or_name);

    std::string profile = s.quality_profile;
    std::transform(profile.begin(), profile.end(), profile.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (profile == "interactive") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Interactive;
    else if (profile == "preview") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Preview;
    else if (profile == "final") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Final;
    else if (profile == "cinema") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Cinema;
    else if (profile == "custom") it->quality_profile = RayTrophiSim::SimulationDomainQualityProfile::Custom;
    else return Result::fail("unknown gas quality_profile: " + s.quality_profile);

    it->resource_budget_mb = s.resource_budget_mb;
    it->enforce_resource_budget = s.enforce_resource_budget;
    it->use_sparse_tiles = s.use_sparse_tiles;
    it->render_to_nanovdb = s.render_to_nanovdb;
    it->fire_enabled = s.fire_enabled;
    it->ignition_temperature = std::max(0.0f, s.ignition_temperature);
    it->burn_rate = std::max(0.0f, s.burn_rate);
    it->heat_release = std::max(0.0f, s.heat_release);
    it->smoke_generation = std::max(0.0f, s.smoke_generation);
    it->flame_dissipation = std::max(0.0f, s.flame_dissipation);
    it->fire_max_temperature = std::max(0.0f, s.fire_max_temperature);
    it->gas_buoyancy_heat = s.buoyancy_heat;
    it->gas_buoyancy_density = s.buoyancy_density;
    it->gas_vorticity = std::max(0.0f, s.vorticity);
    it->fire_expansion = std::max(0.0f, s.fire_expansion);
    it->turbulence_strength = std::max(0.0f, s.turbulence_strength);
    it->turbulence_scale = std::max(0.001f, s.turbulence_scale);
    it->turbulence_octaves = std::clamp(s.turbulence_octaves, 1, 8);
    it->turbulence_lacunarity = std::max(1.0f, s.turbulence_lacunarity);
    it->turbulence_persistence = std::clamp(s.turbulence_persistence, 0.0f, 1.0f);
    it->turbulence_speed = s.turbulence_speed;
    return Result::success();
}

Result getCombustibleFluidSettings(
    const std::string& domain_id_or_name,
    CombustibleFluidSettings& out) {
    if (!g_ctx) return notBound();
    FluidDomainInfo info;
    Result found=getFluidDomain(domain_id_or_name,info);
    if(!found.ok) return found;
    auto& domains=g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it=std::find_if(domains.begin(),domains.end(),
        [&info](const auto& d){return d.name==info.name;});
    if(it==domains.end() ||
       it->type!=RayTrophiSim::SimulationDomainType::Fluid)
        return Result::fail("fluid domain not found: "+domain_id_or_name);
    out.enabled=it->fluid_flammable;
    out.auto_ignite=it->fluid_auto_ignite;
    out.ignition_temperature=it->fluid_ignition_temperature;
    out.evaporation_rate=it->fluid_evaporation_rate;
    out.surface_fuel_capacity=it->fluid_surface_fuel_capacity;
    out.heat_release=it->fluid_combustion_heat_release;
    out.smoke_yield=it->fluid_combustion_smoke_yield;
    out.surface_cooling=it->fluid_surface_cooling;
    return Result::success();
}

Result updateCombustibleFluidSettings(
    const std::string& domain_id_or_name,
    const CombustibleFluidSettings& s) {
    if(!g_ctx) return notBound();
    if(renderJobActive())
        return Result::fail("scene is locked by the final render job");
    FluidDomainInfo info;
    Result found=getFluidDomain(domain_id_or_name,info);
    if(!found.ok) return found;
    auto& domains=g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it=std::find_if(domains.begin(),domains.end(),
        [&info](const auto& d){return d.name==info.name;});
    if(it==domains.end() ||
       it->type!=RayTrophiSim::SimulationDomainType::Fluid)
        return Result::fail("fluid domain not found: "+domain_id_or_name);
    it->fluid_flammable=s.enabled;
    it->fluid_auto_ignite=s.auto_ignite;
    it->fluid_ignition_temperature=std::max(0.0f,s.ignition_temperature);
    it->fluid_evaporation_rate=std::max(0.0f,s.evaporation_rate);
    it->fluid_surface_fuel_capacity=std::max(0.0f,s.surface_fuel_capacity);
    it->fluid_combustion_heat_release=std::max(0.0f,s.heat_release);
    it->fluid_combustion_smoke_yield=std::max(0.0f,s.smoke_yield);
    it->fluid_surface_cooling=std::max(0.0f,s.surface_cooling);
    return Result::success();
}

// Shared with RtApiParticle.cpp (declared in RtApiInternal.h): the emitters,
// colliders and grid domains all hang off the SAME runtime, so both facades
// must reach it through one accessor and invalidate it the same way.
RayTrophiSim::ParticleSimulationSystem& scriptSimulationRuntime() {
    return g_ctx->scene.ensureParticleSimulationSystem();
}

void invalidateScriptSimulation() {
    g_ctx->scene.clearSimFrameCache();
    g_ctx->scene.requestSimulationTimelineRenderResync();
}

namespace {

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

int scriptDomainIndex(RayTrophiSim::ParticleSimulationSystem& runtime,
                      const std::string& domain_name) {
    const auto& domains = runtime.gridDomains();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        if (domains[i].name == domain_name) return static_cast<int>(i);
    }
    return -1;
}

RayTrophiSim::SimulationFlowSourceMode flowModeFromString(const std::string& value,
                                                          bool& ok) {
    const std::string mode = lowerCopy(value);
    ok = true;
    if (mode == "point") return RayTrophiSim::SimulationFlowSourceMode::Point;
    if (mode == "object_bounds" || mode == "bounds")
        return RayTrophiSim::SimulationFlowSourceMode::ObjectBounds;
    if (mode == "mesh_surface" || mode == "surface")
        return RayTrophiSim::SimulationFlowSourceMode::MeshSurface;
    ok = false;
    return RayTrophiSim::SimulationFlowSourceMode::Point;
}

std::string flowModeToString(RayTrophiSim::SimulationFlowSourceMode mode) {
    switch (mode) {
        case RayTrophiSim::SimulationFlowSourceMode::ObjectBounds: return "object_bounds";
        case RayTrophiSim::SimulationFlowSourceMode::MeshSurface: return "mesh_surface";
        default: return "point";
    }
}

SimulationFlowSourceInfo flowInfoFromDesc(
    const RayTrophiSim::SimulationFlowSourceDesc& source,
    const std::vector<RayTrophiSim::SimulationGridDomainDesc>& domains) {
    SimulationFlowSourceInfo out;
    out.name = source.name;
    if (source.domain_index >= 0 &&
        source.domain_index < static_cast<int>(domains.size()))
        out.domain = domains[static_cast<std::size_t>(source.domain_index)].name;
    out.source_mode = flowModeToString(source.source_mode);
    out.source_object = source.source_name;
    out.enabled = source.enabled;
    out.position = source.position;
    out.velocity = source.velocity;
    out.radius = source.radius;
    out.velocity_coupling = source.velocity_coupling;
    out.density = source.density;
    out.temperature = source.temperature;
    out.fuel = source.fuel;
    out.falloff = source.falloff;
    out.fluid_particles_per_second = source.fluid_particles_per_second;
    out.fluid_velocity_spread = source.fluid_velocity_spread;
    out.fluid_emit_along_normal = source.fluid_emit_along_normal;
    out.use_time_limit = source.use_time_limit;
    out.start_time = source.start_time;
    out.end_time = source.end_time;
    out.use_particle_limit = source.use_particle_limit;
    out.max_emitted_particles = source.max_emitted_particles;
    return out;
}

Result flowDescFromInfo(const SimulationFlowSourceInfo& info,
                        RayTrophiSim::ParticleSimulationSystem& runtime,
                        RayTrophiSim::SimulationFlowSourceDesc& out) {
    const int domain_index = scriptDomainIndex(runtime, info.domain);
    if (domain_index < 0) return Result::fail("simulation domain not found: " + info.domain);
    bool mode_ok = false;
    out.name = info.name.empty() ? "Flow Source" : info.name;
    out.domain_index = domain_index;
    out.source_mode = flowModeFromString(info.source_mode, mode_ok);
    if (!mode_ok) return Result::fail("unknown flow source_mode: " + info.source_mode);
    out.source_name = info.source_object;
    out.enabled = info.enabled;
    out.position = info.position;
    out.velocity = info.velocity;
    out.radius = std::max(0.001f, info.radius);
    out.velocity_coupling = std::max(0.0f, info.velocity_coupling);
    out.density = std::max(0.0f, info.density);
    out.temperature = std::max(0.0f, info.temperature);
    out.fuel = std::max(0.0f, info.fuel);
    out.falloff = std::max(0.0f, info.falloff);
    out.fluid_particles_per_second = std::max(0.0f, info.fluid_particles_per_second);
    out.fluid_velocity_spread = std::max(0.0f, info.fluid_velocity_spread);
    out.fluid_emit_along_normal = info.fluid_emit_along_normal;
    out.use_time_limit = info.use_time_limit;
    out.start_time = info.start_time;
    out.end_time = std::max(info.start_time, info.end_time);
    out.use_particle_limit = info.use_particle_limit;
    out.max_emitted_particles = std::max(0, info.max_emitted_particles);
    return Result::success();
}

RayTrophiSim::ParticleColliderSourceMode colliderModeFromString(
    const std::string& value, bool& ok) {
    const std::string mode = lowerCopy(value);
    ok = true;
    if (mode == "plane" || mode == "plane_y") return RayTrophiSim::ParticleColliderSourceMode::PlaneY;
    if (mode == "sphere") return RayTrophiSim::ParticleColliderSourceMode::Sphere;
    if (mode == "capsule") return RayTrophiSim::ParticleColliderSourceMode::Capsule;
    if (mode == "aabb" || mode == "object_aabb") return RayTrophiSim::ParticleColliderSourceMode::ObjectAABB;
    if (mode == "obb" || mode == "object_obb") return RayTrophiSim::ParticleColliderSourceMode::ObjectOBB;
    if (mode == "mesh_sdf" || mode == "sdf") return RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF;
    if (mode == "convex" || mode == "convex_decomp") return RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp;
    if (mode == "mesh_bvh" || mode == "bvh") return RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH;
    ok = false;
    return RayTrophiSim::ParticleColliderSourceMode::PlaneY;
}

std::string colliderModeToString(RayTrophiSim::ParticleColliderSourceMode mode) {
    switch (mode) {
        case RayTrophiSim::ParticleColliderSourceMode::ObjectAABB: return "aabb";
        case RayTrophiSim::ParticleColliderSourceMode::ObjectOBB: return "obb";
        case RayTrophiSim::ParticleColliderSourceMode::Sphere: return "sphere";
        case RayTrophiSim::ParticleColliderSourceMode::Capsule: return "capsule";
        case RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF: return "mesh_sdf";
        case RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp: return "convex";
        case RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH: return "mesh_bvh";
        default: return "plane";
    }
}

SimulationColliderInfo colliderInfoFromDesc(const RayTrophiSim::ParticleColliderDesc& c) {
    SimulationColliderInfo out;
    out.name = c.name;
    out.source_mode = colliderModeToString(c.source_mode);
    out.source_object = c.source_name;
    out.enabled = c.enabled;
    out.plane_y = c.plane_y;
    out.sphere_center = c.sphere_center;
    out.sphere_radius = c.sphere_radius;
    out.capsule_start = c.capsule_start;
    out.capsule_end = c.capsule_end;
    out.capsule_radius = c.capsule_radius;
    out.bounds_min = c.bounds_min;
    out.bounds_max = c.bounds_max;
    out.friction = c.friction;
    out.restitution = c.restitution;
    out.thickness = c.thickness;
    out.gas_interaction_enabled = c.gas_interaction_enabled;
    out.gas_density_rate = c.gas_density_rate;
    out.gas_temperature_rate = c.gas_temperature_rate;
    out.gas_fuel_rate = c.gas_fuel_rate;
    out.gas_flame_rate = c.gas_flame_rate;
    return out;
}

Result colliderDescFromInfo(const SimulationColliderInfo& info,
                            RayTrophiSim::ParticleColliderDesc& out) {
    bool mode_ok = false;
    out.name = info.name.empty() ? "Simulation Collider" : info.name;
    out.source_mode = colliderModeFromString(info.source_mode, mode_ok);
    if (!mode_ok) return Result::fail("unknown collider source_mode: " + info.source_mode);
    out.source_name = info.source_object;
    out.enabled = info.enabled;
    out.plane_y = info.plane_y;
    out.sphere_center = info.sphere_center;
    out.sphere_radius = std::max(0.001f, info.sphere_radius);
    out.capsule_start = info.capsule_start;
    out.capsule_end = info.capsule_end;
    out.capsule_radius = std::max(0.001f, info.capsule_radius);
    out.bounds_min = Vec3::min(info.bounds_min, info.bounds_max);
    out.bounds_max = Vec3::max(info.bounds_min, info.bounds_max);
    out.friction = std::clamp(info.friction, 0.0f, 1.0f);
    out.restitution = std::clamp(info.restitution, 0.0f, 1.0f);
    out.thickness = std::max(0.0f, info.thickness);
    out.gas_interaction_enabled = info.gas_interaction_enabled;
    out.gas_density_rate = std::max(0.0f, info.gas_density_rate);
    out.gas_temperature_rate = std::max(0.0f, info.gas_temperature_rate);
    out.gas_fuel_rate = std::max(0.0f, info.gas_fuel_rate);
    out.gas_flame_rate = std::max(0.0f, info.gas_flame_rate);
    return Result::success();
}

} // namespace

Result listSimulationFlowSources(std::vector<SimulationFlowSourceInfo>& out) {
    if (!g_ctx) return notBound();
    auto& runtime = scriptSimulationRuntime();
    out.clear();
    out.reserve(runtime.flowSources().size());
    for (const auto& source : runtime.flowSources())
        out.push_back(flowInfoFromDesc(source, runtime.gridDomains()));
    return Result::success();
}

Result getSimulationFlowSource(const std::string& name, SimulationFlowSourceInfo& out) {
    if (!g_ctx) return notBound();
    auto& runtime = scriptSimulationRuntime();
    const auto it = std::find_if(runtime.flowSources().begin(), runtime.flowSources().end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == runtime.flowSources().end()) return Result::fail("flow source not found: " + name);
    out = flowInfoFromDesc(*it, runtime.gridDomains());
    return Result::success();
}

Result createSimulationFlowSource(const SimulationFlowSourceInfo& info,
                                  SimulationFlowSourceInfo& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    if (std::any_of(runtime.flowSources().begin(), runtime.flowSources().end(),
                    [&info](const auto& source) { return source.name == info.name; }))
        return Result::fail("flow source already exists: " + info.name);
    RayTrophiSim::SimulationFlowSourceDesc desc;
    Result converted = flowDescFromInfo(info, runtime, desc);
    if (!converted.ok) return converted;
    auto& created = runtime.addFlowSource(desc);
    out = flowInfoFromDesc(created, runtime.gridDomains());
    invalidateScriptSimulation();
    return Result::success();
}

Result updateSimulationFlowSource(const std::string& name,
                                  const SimulationFlowSourceInfo& info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto it = std::find_if(runtime.flowSources().begin(), runtime.flowSources().end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == runtime.flowSources().end()) return Result::fail("flow source not found: " + name);
    RayTrophiSim::SimulationFlowSourceDesc desc;
    Result converted = flowDescFromInfo(info, runtime, desc);
    if (!converted.ok) return converted;
    desc.timeline_uid = it->timeline_uid;
    *it = std::move(desc);
    invalidateScriptSimulation();
    return Result::success();
}

Result removeSimulationFlowSource(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto& sources = runtime.flowSources();
    const auto it = std::find_if(sources.begin(), sources.end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == sources.end()) return Result::fail("flow source not found: " + name);
    sources.erase(it);
    invalidateScriptSimulation();
    return Result::success();
}

Result listSimulationColliders(std::vector<SimulationColliderInfo>& out) {
    if (!g_ctx) return notBound();
    const auto& colliders = scriptSimulationRuntime().colliders();
    out.clear();
    out.reserve(colliders.size());
    for (const auto& collider : colliders) out.push_back(colliderInfoFromDesc(collider));
    return Result::success();
}

Result getSimulationCollider(const std::string& name, SimulationColliderInfo& out) {
    if (!g_ctx) return notBound();
    const auto& colliders = scriptSimulationRuntime().colliders();
    const auto it = std::find_if(colliders.begin(), colliders.end(),
        [&name](const auto& collider) { return collider.name == name; });
    if (it == colliders.end()) return Result::fail("simulation collider not found: " + name);
    out = colliderInfoFromDesc(*it);
    return Result::success();
}

Result createSimulationCollider(const SimulationColliderInfo& info,
                                SimulationColliderInfo& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    if (std::any_of(runtime.colliders().begin(), runtime.colliders().end(),
                    [&info](const auto& collider) { return collider.name == info.name; }))
        return Result::fail("simulation collider already exists: " + info.name);
    RayTrophiSim::ParticleColliderDesc desc;
    Result converted = colliderDescFromInfo(info, desc);
    if (!converted.ok) return converted;
    auto& created = runtime.addCollider(desc);
    out = colliderInfoFromDesc(created);
    invalidateScriptSimulation();
    return Result::success();
}

Result updateSimulationCollider(const std::string& name,
                                const SimulationColliderInfo& info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto& colliders = runtime.colliders();
    auto it = std::find_if(colliders.begin(), colliders.end(),
        [&name](const auto& collider) { return collider.name == name; });
    if (it == colliders.end()) return Result::fail("simulation collider not found: " + name);
    RayTrophiSim::ParticleColliderDesc desc;
    Result converted = colliderDescFromInfo(info, desc);
    if (!converted.ok) return converted;
    *it = std::move(desc);
    invalidateScriptSimulation();
    return Result::success();
}

Result removeSimulationCollider(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto& colliders = runtime.colliders();
    const auto it = std::find_if(colliders.begin(), colliders.end(),
        [&name](const auto& collider) { return collider.name == name; });
    if (it == colliders.end()) return Result::fail("simulation collider not found: " + name);
    colliders.erase(it);
    invalidateScriptSimulation();
    return Result::success();
}

Result resetFluidSimulation() {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    // Reset both the legacy FluidObject bridge and the unified particle/grid
    // runtime used by scripted Gas + APIC domains, including source counters,
    // collider history and the timeline frame cache.
    g_ctx->scene.ensureFluidSimulationSystem();
    g_ctx->scene.resetSimulation();
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
