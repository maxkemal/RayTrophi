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
#include "MaterialStateField.h"
#include "MaterialManager.h"
#include "VolumeShader.h"
#include <algorithm>
#include <cctype>

namespace rtapi {

namespace {

// Preset enum -> the string scripts use. One definition, because this ternary
// chain was copy-pasted at three call sites and adding Chocolate to two of them
// would have produced a domain that reports "custom" from fluid.list and
// "chocolate" from fluid.get.
const char* fluidPresetName(RayTrophiSim::Fluid::APICSolverParams::FluidPreset p) {
    using FluidPreset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset;
    switch (p) {
        case FluidPreset::Water:     return "water";
        case FluidPreset::Oil:       return "oil";
        case FluidPreset::Mud:       return "mud";
        case FluidPreset::Honey:     return "honey";
        case FluidPreset::Lava:      return "lava";
        case FluidPreset::Sand:      return "sand";
        case FluidPreset::Chocolate: return "chocolate";
        case FluidPreset::Custom:
        default:                     return "custom";
    }
}

// Copy the rheology block a script can read back. Kept together so a new
// rheology field cannot reach one reporting path and miss another.
void fillFluidRheology(const RayTrophiSim::Fluid::APICSolverParams& params,
                       FluidDomainInfo& info) {
    info.preset              = fluidPresetName(params.current_preset);
    info.kinematic_viscosity = params.kinematic_viscosity;
    info.viscosity_sweeps    = params.viscosity_sweeps;
    info.viscosity_wall_slip = params.viscosity_wall_slip;
}

// Report the isosurface material by NAME. Lives on the grid descriptor rather
// than in APICSolverParams (it is a look, not rheology), so it needs its own
// helper next to fillFluidRheology instead of riding along inside it.
void fillFluidSurfaceMaterial(const RayTrophiSim::SimulationGridDomainDesc& d,
                              FluidDomainInfo& info) {
    // Porosity rides along here for the same reason: it is surface LOOK, and a
    // script has to be able to read back what it set or it cannot tell "the
    // value never landed" from "the value landed and did nothing".
    info.pore_amount = d.fluid_surface_pore_amount;
    info.pore_scale  = d.fluid_surface_pore_scale;
    info.pore_detail = d.fluid_surface_pore_detail;

    info.surface_material.clear();
    if (d.fluid_surface_material_id < 0) return;
    const auto& mats = MaterialManager::getInstance().getAllMaterials();
    const std::size_t mi = static_cast<std::size_t>(d.fluid_surface_material_id);
    if (mi < mats.size() && mats[mi]) info.surface_material = mats[mi]->materialName;
}

// Locate a Gas domain descriptor by id or name. Shared by the shader accessors
// below; mirrors the lookup updateGasDomainSettings performs.
RayTrophiSim::SimulationGridDomainDesc* findGasDomainDesc(
    const std::string& domain_id_or_name, Result& out_error) {
    if (!g_ctx) { out_error = notBound(); return nullptr; }
    FluidDomainInfo info;
    Result found = getFluidDomain(domain_id_or_name, info);
    if (!found.ok) { out_error = found; return nullptr; }
    auto& domains = g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    auto it = std::find_if(domains.begin(), domains.end(),
        [&info](const auto& d) { return d.name == info.name; });
    if (it == domains.end() || it->type != RayTrophiSim::SimulationDomainType::Gas) {
        out_error = Result::fail("gas domain not found: " + domain_id_or_name);
        return nullptr;
    }
    out_error = Result::success();
    return &(*it);
}

} // namespace

Result listMaterialSubstances(std::vector<std::string>& out_names) {
    out_names.clear();
    for (const auto& profile : RayTrophiSim::substanceLibrary()) {
        out_names.push_back(profile.name);
    }
    return Result::success();
}

Result listMaterialFields(std::vector<MaterialFieldInfo>& out_fields) {
    if (!g_ctx) return notBound();
    out_fields.clear();
    auto& runtime = scriptSimulationRuntime();
    // Integrity/mass-loss live on the device during simulation. Request the
    // next scheduler step's readback before exposing the current host snapshot;
    // repeated script polling then observes a coherent completed generation
    // instead of reading the creation-time all-ones integrity forever.
    runtime.requestMaterialStateFieldReadback();
    const auto& fields = runtime.materialStateFields();
    out_fields.reserve(fields.size());
    for (const auto& entry : fields) {
        const auto& field = entry.second;
        MaterialFieldInfo info;
        info.object_key = field.object_key;
        info.substance = field.substance_name;
        info.topology_generation = field.topology_generation;
        info.content_generation = field.mask_revision;
        info.element_count = static_cast<uint32_t>(field.elementCount());
        info.mask_resolution = field.mask_resolution;
        info.centers_dirty = field.centers_dirty;
        const auto integrity =
            RayTrophiSim::MaterialStateFieldSystem::summarizeIntegrity(field);
        if (integrity.valid) {
            info.mean_integrity = integrity.mean_integrity;
            info.minimum_integrity = integrity.minimum_integrity;
            info.mass_loss = integrity.total_mass_loss;
        }
        const auto budget =
            RayTrophiSim::MaterialStateFieldSystem::summarizeMassBudget(field);
        if (budget.valid) {
            info.initial_mass = budget.initial_mass;
            info.solid_mass = budget.solid_mass;
            info.pyrolyzed_mass = budget.pyrolyzed_mass;
            info.molten_reservoir_mass = budget.molten_reservoir_mass;
            info.transferred_mass = budget.transferred_mass;
            info.mass_conservation_error = budget.conservation_error;
            info.mass_budget_overflow = budget.budget_overflow_mass;
            info.mass_negative = budget.negative_mass;
            info.mass_invalid_elements = budget.invalid_elements;
        }
        info.semantics = {
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Temperature),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Moisture),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::FuelRemaining),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Char),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Melt),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::MassLoss),
            RayTrophiSim::fieldSemanticName(RayTrophiSim::FieldSemantic::Integrity)
        };
        out_fields.push_back(std::move(info));
    }
    return Result::success();
}

Result getGasShaderSettings(const std::string& domain_id_or_name,
                            GasShaderSettings& out_settings) {
    Result error;
    auto* domain = findGasDomainDesc(domain_id_or_name, error);
    if (!domain) return error;
    if (!domain->shader) {
        // Not an error: the render bridge creates the shader lazily on the
        // first sync. Report the defaults the domain's intent will produce.
        out_settings = GasShaderSettings{};
        out_settings.preset = domain->fire_enabled ? "fire" : "smoke";
        return Result::success();
    }
    const auto& s = *domain->shader;
    out_settings.preset = domain->fire_enabled ? "fire" : "smoke";
    out_settings.density_multiplier = s.density.multiplier;
    out_settings.density_cutoff = s.density.cutoff_threshold;
    out_settings.blackbody_intensity = s.emission.blackbody_intensity;
    out_settings.temperature_min = s.emission.temperature_min;
    out_settings.temperature_max = s.emission.temperature_max;
    out_settings.scattering_coefficient = s.scattering.coefficient;
    out_settings.absorption_coefficient = s.absorption.coefficient;
    return Result::success();
}

Result updateGasShaderSettings(const std::string& domain_id_or_name,
                               const GasShaderSettings& settings) {
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    Result error;
    auto* domain = findGasDomainDesc(domain_id_or_name, error);
    if (!domain) return error;

    std::string preset = settings.preset;
    std::transform(preset.begin(), preset.end(), preset.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (preset == "fire") {
        domain->shader = VolumeShader::createFirePreset();
    } else if (preset == "smoke") {
        domain->shader = VolumeShader::createSmokePreset();
    } else if (!preset.empty()) {
        return Result::fail("unknown gas shader preset: " + settings.preset);
    } else if (!domain->shader) {
        domain->shader = domain->fire_enabled ? VolumeShader::createFirePreset()
                                              : VolumeShader::createSmokePreset();
    }
    auto& s = *domain->shader;
    s.density.multiplier = std::max(0.0f, settings.density_multiplier);
    s.density.cutoff_threshold = std::max(0.0f, settings.density_cutoff);
    s.emission.blackbody_intensity = std::max(0.0f, settings.blackbody_intensity);
    // A zero-width or inverted range maps every temperature to the same colour,
    // which reads as "emission is broken". Keep the pair ordered and separated.
    s.emission.temperature_min = std::max(0.0f, settings.temperature_min);
    s.emission.temperature_max =
        std::max(s.emission.temperature_min + 1.0f, settings.temperature_max);
    s.scattering.coefficient = std::max(0.0f, settings.scattering_coefficient);
    s.absorption.coefficient = std::max(0.0f, settings.absorption_coefficient);
    g_gas_volumes_dirty = true;
    return Result::success();
}

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
        // Both gas and liquid now start on the fastest solver the machine has —
        // scripted domains should not be slower than panel-created ones. Scene
        // synchronization selects CUDA first and Vulkan second, and retains the
        // deterministic CPU fallback when neither GPU backend is usable.
        desc.backend = RayTrophiSim::defaultSimulationDomainBackend();
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

    // ★ A GAS domain gets NO legacy FluidObject.
    //
    // FluidObject is the pre-grid-domain APIC LIQUID container (particles, seed
    // box, level set). Creating one for gas left a ghost that nothing simulated
    // but that syncFluidRenderVolumes still walked EVERY FRAME: it built a
    // density volume from the ghost's grid, found it empty, and called
    // destroyFluidRenderVolume — which unconditionally raises g_geometry_dirty +
    // g_vulkan_rebuild_pending. That is a full TLAS rebuild per frame, and it is
    // exactly why a scripted gas domain churned while a panel-created one (grid
    // domain only) did not. The grid domain is the authority; getFluidDomain
    // resolves gas domains from it directly.
    if (is_gas) {
        if (!grid_dom) return Result::fail("failed to create gas domain: " + domain_name);
        out_info.id = 0;
        out_info.name = grid_dom->name;
        out_info.type = "gas";
        out_info.domain_min = grid_dom->bounds_min;
        out_info.domain_max = grid_dom->bounds_max;
        out_info.voxel_size = grid_dom->voxel_size;
        out_info.particle_count = 0;
        out_info.render_mode = "volume";
        out_info.boundary =
            (grid_dom->boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Open) ? "open" :
            (grid_dom->boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Periodic) ? "periodic" : "closed";
        out_info.preset = "custom";
        out_info.backend =
            (grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
            (grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
            (grid_dom->backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
        out_info.enabled = grid_dom->enabled;
        out_info.visible = true;
        return Result::success();
    }

    // 2. Ensure low-level FluidObject exists (liquid domains only)
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
    fillFluidRheology(obj->params, out_info);
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
    if (!obj) {
        // Gas domains own no FluidObject (see createFluidDomain). Resolve them
        // from the grid-domain list, which is the authority for both types.
        auto& p_sys_gas = g_ctx->scene.ensureParticleSimulationSystem();
        for (const auto& d : p_sys_gas.gridDomains()) {
            if (d.name != domain_id_or_name) continue;
            out_info = FluidDomainInfo{};
            out_info.id = 0;
            out_info.name = d.name;
            out_info.type = (d.type == RayTrophiSim::SimulationDomainType::Gas) ? "gas" : "fluid";
            out_info.domain_min = d.bounds_min;
            out_info.domain_max = d.bounds_max;
            out_info.voxel_size = d.voxel_size;
            out_info.particle_count = 0;
            out_info.render_mode = "volume";
            out_info.boundary =
                (d.boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Open) ? "open" :
                (d.boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Periodic) ? "periodic" : "closed";
            out_info.preset = "custom";
                out_info.backend =
                (d.backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
                (d.backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
                (d.backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
            out_info.enabled = d.enabled;
            out_info.visible = true;
            return Result::success();
        }
        return Result::fail("fluid domain not found: " + domain_id_or_name);
    }

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
    fillFluidRheology(obj->params, out_info);
    out_info.backend = (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
                       (grid_dom && grid_dom->backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
    out_info.enabled = obj->enabled;
    out_info.visible = obj->visible;
    // The grid descriptor is the live APIC authority. Automatic molten
    // transfer may specialize an empty domain without mutating the legacy
    // editor mirror, so report those live values when available.
    if (grid_dom) {
        out_info.render_mode =
            (grid_dom->fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
            (grid_dom->fluid_render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
        fillFluidRheology(grid_dom->fluid_params, out_info);
        fillFluidSurfaceMaterial(*grid_dom, out_info);
        const auto& states = p_sys_get.gridDomainStates();
        const auto& domains = p_sys_get.gridDomains();
        const std::size_t index = static_cast<std::size_t>(grid_dom - domains.data());
        if (index < states.size() && states[index].valid)
            out_info.particle_count = states[index].particles.size();
    }
    return Result::success();
}

Result listFluidDomains(std::vector<rtapi::FluidDomainInfo>& out_domains) {
    if (!g_ctx) return notBound();
    out_domains.clear();

    // The grid-domain list is the authority for BOTH types (a gas domain owns no
    // FluidObject at all). Walk every particle system, not just the active one —
    // presets and imported scenes can leave a second system behind, and a domain
    // the caller cannot see is a domain it cannot delete.
    for (auto& system : g_ctx->scene.particle_systems) {
        if (!system.runtime) continue;
        for (const auto& d : system.runtime->gridDomains()) {
            FluidDomainInfo info;
            info.name = d.name;
            info.type = (d.type == RayTrophiSim::SimulationDomainType::Gas) ? "gas" : "fluid";
            info.domain_min = d.bounds_min;
            info.domain_max = d.bounds_max;
            info.voxel_size = d.voxel_size;
            info.boundary =
                (d.boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Open) ? "open" :
                (d.boundary_mode == RayTrophiSim::SimulationGridDomainBoundaryMode::Periodic) ? "periodic" : "closed";
            info.backend =
                (d.backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute) ? "gpu" :
                (d.backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan) ? "vulkan" :
                (d.backend == RayTrophiSim::SimulationDomainBackend::CPU_SparseVDB) ? "cpu_sparse" : "cpu";
            info.enabled = d.enabled;
            info.visible = true;
            info.render_mode = "volume";
            info.preset = "custom";
            info.particle_count = 0;
            fillFluidSurfaceMaterial(d, info);

            // Liquid domains additionally carry a legacy FluidObject; take the
            // id/preset/viscosity from it so the entry matches what fluid.get
            // reports for the same name.
            if (d.type != RayTrophiSim::SimulationDomainType::Gas) {
                if (auto* obj = g_ctx->scene.findFluidObjectByName(d.name)) {
                    info.id = obj->id;
                    info.particle_count = obj->particles.size();
                    info.visible = obj->visible;
                    info.render_mode =
                        (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) ? "surface" :
                        (obj->render_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) ? "particles" : "volume";
                    fillFluidRheology(obj->params, info);
                }
            }
            out_domains.push_back(std::move(info));
        }
    }
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
    // Gas domains report id 0 because they own no FluidObject; do not hand that
    // to removeFluidObject, which would be a lookup for an unrelated object.
    const bool object_removed =
        (info.type == "gas") ? false : g_ctx->scene.removeFluidObject(info.id);
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
                         const std::string* preset,
                         const float* kinematic_viscosity,
                         const int* viscosity_sweeps,
                         const float* viscosity_wall_slip,
                         const std::string* surface_material,
                         const float* pore_amount,
                         const float* pore_scale,
                         const float* pore_detail,
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

    auto& p_sys_upd = g_ctx->scene.ensureParticleSimulationSystem();
    RayTrophiSim::SimulationGridDomainDesc* grid_dom = nullptr;
    for (auto& d : p_sys_upd.gridDomains()) {
        if (d.name == info.name) { grid_dom = &d; break; }
    }
    if (!obj && !grid_dom) return Result::fail("fluid domain not found");

    if (obj) {
        if (domain_min) { obj->domain_min = *domain_min; obj->grid_dirty = true; }
        if (domain_max) { obj->domain_max = *domain_max; obj->grid_dirty = true; }
        if (voxel_size && *voxel_size > 0.001f) {
            obj->voxel_size = *voxel_size;
            obj->grid_dirty = true;
        }
    }
    if (grid_dom) {
        if (domain_min) grid_dom->bounds_min = *domain_min;
        if (domain_max) grid_dom->bounds_max = *domain_max;
        if (voxel_size && *voxel_size > 0.001f) grid_dom->voxel_size = *voxel_size;
    }

    if (render_mode) {
        std::string rm = *render_mode;
        std::transform(rm.begin(), rm.end(), rm.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        const auto grid_mode = rm == "surface"
            ? RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF
            : (rm == "particles" ? RayTrophiSim::Fluid::FluidRenderMode::Particles
                                  : RayTrophiSim::Fluid::FluidRenderMode::Volume);
        if (grid_dom) grid_dom->fluid_render_mode = grid_mode;
        if (obj) {
            obj->render_mode = grid_mode;
        }
    }

    // ★ Rheology writes go to BOTH representations. The grid domain descriptor is
    // what the solver actually steps; the FluidObject is the authoring-side copy.
    // Writing only the FluidObject — which is what this did — meant a script
    // asking for "honey" on a grid-domain fluid got success() and no honey. A
    // silent no-op is worse than a rejection: nothing distinguishes it from a
    // preset that simply does not look like much, which is precisely how the
    // broken viscosity dial survived this long.
    if (preset) {
        std::string p = *preset;
        std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        using FluidPreset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset;
        bool known = true;
        FluidPreset chosen = FluidPreset::Custom;
        if      (p == "water")     chosen = FluidPreset::Water;
        else if (p == "oil")       chosen = FluidPreset::Oil;
        else if (p == "mud")       chosen = FluidPreset::Mud;
        else if (p == "honey")     chosen = FluidPreset::Honey;
        else if (p == "lava")      chosen = FluidPreset::Lava;
        else if (p == "sand")      chosen = FluidPreset::Sand;
        else if (p == "chocolate") chosen = FluidPreset::Chocolate;
        else known = false;
        if (!known) {
            return Result::fail("unknown fluid preset: " + *preset +
                                " (water, oil, mud, honey, lava, sand, chocolate)");
        }
        if (obj)      obj->params.applyPreset(chosen);
        if (grid_dom) grid_dom->fluid_params.applyPreset(chosen);
    }

    if (boundary) {
        std::string b = *boundary;
        std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        const auto grid_boundary = b == "open"
            ? RayTrophiSim::SimulationGridDomainBoundaryMode::Open
            : (b == "periodic" ? RayTrophiSim::SimulationGridDomainBoundaryMode::Periodic
                                : RayTrophiSim::SimulationGridDomainBoundaryMode::Closed);
        if (grid_dom) grid_dom->boundary_mode = grid_boundary;
        if (obj) {
            if (b == "open") obj->params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Open;
            else if (b == "periodic") obj->params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Periodic;
            else obj->params.boundary = RayTrophiSim::Fluid::APICSolverParams::BoundaryMode::Closed;
        }
    }

    // Hand-setting any rheology field means the domain is no longer the named
    // material — mirror the UI's rule so a script and a slider leave the same
    // state behind.
    auto markCustom = [&]() {
        using FluidPreset = RayTrophiSim::Fluid::APICSolverParams::FluidPreset;
        if (obj)      obj->params.current_preset      = FluidPreset::Custom;
        if (grid_dom) grid_dom->fluid_params.current_preset = FluidPreset::Custom;
    };
    if (kinematic_viscosity) {
        const float v = std::max(0.0f, *kinematic_viscosity);
        if (obj)      obj->params.kinematic_viscosity = v;
        if (grid_dom) grid_dom->fluid_params.kinematic_viscosity = v;
        markCustom();
    }
    if (viscosity_sweeps) {
        const int s = std::clamp(*viscosity_sweeps, 1, 64);
        if (obj)      obj->params.viscosity_sweeps = s;
        if (grid_dom) grid_dom->fluid_params.viscosity_sweeps = s;
        markCustom();
    }
    if (viscosity_wall_slip) {
        const float w = std::clamp(*viscosity_wall_slip, 0.0f, 1.0f);
        if (obj)      obj->params.viscosity_wall_slip = w;
        if (grid_dom) grid_dom->fluid_params.viscosity_wall_slip = w;
        markCustom();
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

    if (surface_material && grid_dom) {
        if (surface_material->empty()) {
            grid_dom->fluid_surface_material_id = -1;
        } else {
            auto& mm = MaterialManager::getInstance();
            const auto& mats = mm.getAllMaterials();
            int found = -1;
            for (std::size_t mi = 0; mi < mats.size(); ++mi) {
                if (mats[mi] && mats[mi]->materialName == *surface_material) {
                    found = static_cast<int>(mi);
                    break;
                }
            }
            // Fail loudly. A missing material silently left as the built-in
            // dielectric would look like "the material had no effect", which is
            // exactly the report nobody can act on.
            if (found < 0) {
                return Result::fail("material not found: " + *surface_material);
            }
            grid_dom->fluid_surface_material_id = found;
        }
        g_ctx->scene.refreshFluidSurfaceMaterial();
    }

    // Procedural porosity. Clamped the same way the GPU packer clamps, so a
    // scripted value and a panel value cannot mean different things.
    if (grid_dom && pore_amount) {
        grid_dom->fluid_surface_pore_amount = (std::max)(0.0f, *pore_amount);
    }
    if (grid_dom && pore_scale) {
        grid_dom->fluid_surface_pore_scale = (std::max)(1e-4f, *pore_scale);
    }
    if (grid_dom && pore_detail) {
        grid_dom->fluid_surface_pore_detail =
            (std::max)(0.0f, (std::min)(1.0f, *pore_detail));
    }

    if (obj && enabled) obj->enabled = *enabled;
    if (obj && visible) obj->visible = *visible;
    if (grid_dom && enabled) grid_dom->enabled = *enabled;

    if (obj) obj->ensureGrid();
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
    out.structural_coupling_enabled = it->structural_coupling_enabled;
    out.structural_pressure_scale = it->structural_pressure_scale;
    out.structural_min_intensity = it->structural_min_intensity;
    out.structural_event_interval = it->structural_event_interval;
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
    it->structural_coupling_enabled = s.structural_coupling_enabled;
    it->structural_pressure_scale = std::max(0.0f, s.structural_pressure_scale);
    it->structural_min_intensity = std::max(0.0f, s.structural_min_intensity);
    it->structural_event_interval = std::max(1.0f / 120.0f, s.structural_event_interval);
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
    switch (it->fluid_params.chemistry_preset) {
        case RayTrophiSim::Fluid::FluidChemistryPreset::Water: out.chemistry_preset="water"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Gasoline: out.chemistry_preset="gasoline"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Alcohol: out.chemistry_preset="alcohol"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Oil: out.chemistry_preset="oil"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Plastic: out.chemistry_preset="plastic"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Wax: out.chemistry_preset="wax"; break;
        case RayTrophiSim::Fluid::FluidChemistryPreset::Custom: out.chemistry_preset="custom"; break;
        default: out.chemistry_preset="inert"; break;
    }
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
    std::string chemistry = s.chemistry_preset;
    std::transform(chemistry.begin(), chemistry.end(), chemistry.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (chemistry == "water") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Water);
    else if (chemistry == "gasoline") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Gasoline);
    else if (chemistry == "alcohol") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Alcohol);
    else if (chemistry == "oil") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Oil);
    else if (chemistry == "plastic") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Plastic);
    else if (chemistry == "wax") it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Wax);
    else if (chemistry == "custom") it->fluid_params.chemistry_preset=RayTrophiSim::Fluid::FluidChemistryPreset::Custom;
    else it->fluid_params.applyChemistryProfile(RayTrophiSim::Fluid::FluidChemistryPreset::Inert);
    // The chemistry profile owns the physical interaction mode.  In particular,
    // water must enter the extinguishing path even when the legacy `enabled`
    // flag is used by an older script.
    const auto& profile = it->fluid_params.fuel_profile;
    it->fluid_extinguishing = profile.extinguishing;
    it->fluid_flammable = s.enabled && !profile.extinguishing;
    it->fluid_cooling_power = profile.cooling_power;
    it->fluid_oxygen_dilution = profile.oxygen_dilution;
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
    out.parent_object = source.parent_object;
    out.velocity_space =
        (source.velocity_space == RayTrophiSim::SimulationEmissionVelocitySpace::World)
            ? "world" : "local";
    out.inherit_velocity = source.inherit_velocity;
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
    out.parent_object = info.parent_object;
    {
        const std::string space = lowerCopy(info.velocity_space);
        if (space == "world") {
            out.velocity_space = RayTrophiSim::SimulationEmissionVelocitySpace::World;
        } else if (space.empty() || space == "local") {
            out.velocity_space = RayTrophiSim::SimulationEmissionVelocitySpace::Local;
        } else {
            return Result::fail("unknown flow velocity_space: " + info.velocity_space);
        }
    }
    out.inherit_velocity = info.inherit_velocity;
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
    out.fluid_collision_enabled = c.fluid_collision_enabled;
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
    out.sdf_resolution_mode = c.sdf_resolution_mode;
    out.sdf_ready = c.sdf_grid_data && !c.sdf_grid_data->empty();
    out.sdf_resolution = c.sdf_nx;
    out.gas_interaction_enabled = c.gas_interaction_enabled;
    out.gas_density_rate = c.gas_density_rate;
    out.gas_temperature_rate = c.gas_temperature_rate;
    out.gas_fuel_rate = c.gas_fuel_rate;
    out.gas_flame_rate = c.gas_flame_rate;
    out.gas_ignite_on_contact = c.gas_ignite_on_contact;
    out.msf_substance = c.msf_substance;
    out.msf_override_ignition = c.msf_override.override_ignition;
    out.msf_ignition_kelvin = c.msf_override.ignition_kelvin;
    out.msf_burn_rate_scale = c.msf_override.burn_rate_scale;
    out.msf_fuel_capacity_scale = c.msf_override.fuel_capacity_scale;
    out.msf_mask_resolution = c.msf_mask_resolution;
    out.msf_generate_char_mask = c.msf_generate_char_mask;
    out.msf_auto_transfer = c.msf_auto_transfer;
    out.msf_transfer_domain = c.msf_transfer_domain;
    out.msf_transfer_rate_kg_s = c.msf_transfer_rate_kg_s;
    out.msf_transfer_min_mass_kg = c.msf_transfer_min_mass_kg;
    out.msf_transfer_particles_per_kg = c.msf_transfer_particles_per_kg;
    out.msf_transfer_max_batch_particles = c.msf_transfer_max_batch_particles;
    out.msf_transfer_velocity = c.msf_transfer_velocity;
    out.msf_melt_flow_enabled = c.msf_melt_flow_enabled;
    out.msf_melt_height_loss = c.msf_melt_height_loss;
    out.msf_melt_spread = c.msf_melt_spread;
    out.msf_melt_sdf_refresh = c.msf_melt_sdf_refresh;
    out.msf_melt_sdf_revision_interval = c.msf_melt_sdf_revision_interval;
    out.msf_melt_sdf_change_threshold = c.msf_melt_sdf_change_threshold;
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
    out.fluid_collision_enabled = info.fluid_collision_enabled;
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
    out.sdf_resolution_mode = std::clamp(info.sdf_resolution_mode, 0, 2);
    out.gas_interaction_enabled = info.gas_interaction_enabled;
    out.gas_density_rate = std::max(0.0f, info.gas_density_rate);
    out.gas_temperature_rate = std::max(0.0f, info.gas_temperature_rate);
    out.gas_fuel_rate = std::max(0.0f, info.gas_fuel_rate);
    out.gas_flame_rate = std::max(0.0f, info.gas_flame_rate);
    out.gas_ignite_on_contact = info.gas_ignite_on_contact;
    if (!info.msf_substance.empty()) {
        // Validate against the library rather than storing an unknown name that
        // would silently degrade to the "Custom" profile at simulation time.
        bool known = false;
        for (const auto& profile : RayTrophiSim::substanceLibrary()) {
            if (profile.name == info.msf_substance) { known = true; break; }
        }
        if (!known) return Result::fail("unknown msf_substance: " + info.msf_substance);
        out.msf_substance = info.msf_substance;
    }
    out.msf_override.override_ignition = info.msf_override_ignition;
    out.msf_override.ignition_kelvin = std::max(0.0f, info.msf_ignition_kelvin);
    out.msf_override.burn_rate_scale = std::max(0.0f, info.msf_burn_rate_scale);
    out.msf_override.fuel_capacity_scale = std::max(0.0f, info.msf_fuel_capacity_scale);
    out.msf_mask_resolution = std::clamp(info.msf_mask_resolution, 0, 4096);
    out.msf_generate_char_mask = info.msf_generate_char_mask;
    out.msf_auto_transfer = info.msf_auto_transfer;
    out.msf_transfer_domain = info.msf_transfer_domain;
    out.msf_transfer_rate_kg_s = std::max(0.0f, info.msf_transfer_rate_kg_s);
    out.msf_transfer_min_mass_kg = std::max(0.0f, info.msf_transfer_min_mass_kg);
    out.msf_transfer_particles_per_kg = std::clamp(
        info.msf_transfer_particles_per_kg, 1.0f, 100000.0f);
    out.msf_transfer_max_batch_particles = std::clamp(
        info.msf_transfer_max_batch_particles, 1u, 4096u);
    out.msf_transfer_velocity = info.msf_transfer_velocity;
    out.msf_melt_flow_enabled = info.msf_melt_flow_enabled;
    out.msf_melt_height_loss = std::clamp(info.msf_melt_height_loss, 0.0f, 0.92f);
    out.msf_melt_spread = std::clamp(info.msf_melt_spread, 0.0f, 2.5f);
    out.msf_melt_sdf_refresh = info.msf_melt_sdf_refresh;
    out.msf_melt_sdf_revision_interval = std::clamp<uint32_t>(
        info.msf_melt_sdf_revision_interval, 1u, 60u);
    out.msf_melt_sdf_change_threshold = std::clamp(
        info.msf_melt_sdf_change_threshold, 0.001f, 0.25f);
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
    // Keys are animation, not configuration: a property update must not wipe
    // them. This whole-desc assignment silently discarded the keyframe map
    // before flow sources on Fluid domains were keyable and nobody noticed.
    desc.keyframes = it->keyframes;
    *it = std::move(desc);
    invalidateScriptSimulation();
    return Result::success();
}

Result keySimulationFlowSource(const std::string& name,
                               const SimulationFlowSourceKey& key) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto it = std::find_if(runtime.flowSources().begin(), runtime.flowSources().end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == runtime.flowSources().end()) return Result::fail("flow source not found: " + name);

    // Merge into any key already at this frame so two calls can key different
    // channels on the same frame without erasing each other.
    auto& stored = it->keyframes[key.frame];
    if (key.has_enabled)  { stored.has_enabled = true;  stored.enabled = key.enabled; }
    if (key.has_position) { stored.has_position = true; stored.position = key.position; }
    if (key.has_velocity) { stored.has_velocity = true; stored.velocity = key.velocity; }
    if (key.has_radius)   { stored.has_radius = true;   stored.radius = std::max(0.001f, key.radius); }
    if (key.has_density)  { stored.has_density = true;  stored.density = std::max(0.0f, key.density); }
    if (key.has_temperature) { stored.has_temperature = true; stored.temperature = std::max(0.0f, key.temperature); }
    if (key.has_fuel)     { stored.has_fuel = true;     stored.fuel = std::max(0.0f, key.fuel); }
    if (key.has_falloff)  { stored.has_falloff = true;  stored.falloff = std::max(0.0f, key.falloff); }
    if (key.has_velocity_coupling) {
        stored.has_velocity_coupling = true;
        stored.velocity_coupling = std::max(0.0f, key.velocity_coupling);
    }
    if (key.has_flow_rate) { stored.has_flow_rate = true; stored.flow_rate = std::max(0.0f, key.flow_rate); }
    invalidateScriptSimulation();
    return Result::success();
}

Result clearSimulationFlowSourceKey(const std::string& name, int frame) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& runtime = scriptSimulationRuntime();
    auto it = std::find_if(runtime.flowSources().begin(), runtime.flowSources().end(),
        [&name](const auto& source) { return source.name == name; });
    if (it == runtime.flowSources().end()) return Result::fail("flow source not found: " + name);
    it->keyframes.erase(frame);
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
    // UI creation starts the asynchronous SDF cook immediately. Scripted
    // creation must use the same path; invalidating the timeline alone never
    // populates sdf_grid_data, so mesh_sdf silently behaved like no collider.
    g_ctx->scene.rebuildSDFColliderAsync(created);
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
    g_ctx->scene.rebuildSDFColliderAsync(*it);
    return Result::success();
}

Result rebuildSimulationColliderSDF(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto& colliders = scriptSimulationRuntime().colliders();
    auto it = std::find_if(colliders.begin(), colliders.end(),
        [&name](const auto& collider) { return collider.name == name; });
    if (it == colliders.end()) return Result::fail("simulation collider not found: " + name);
    if (it->source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF)
        return Result::fail("simulation collider is not mesh_sdf: " + name);
    it->sdf_grid_data.reset();
    it->sdf_nx = it->sdf_ny = it->sdf_nz = 0;
    g_ctx->scene.rebuildSDFColliderAsync(*it);
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

    // Script-authored APIC/gas domains live on ParticleSimulationSystem, while
    // legacy FluidObject instances live on FluidSimulationSystem. Both are
    // registered with SimulationWorld. Calling only the legacy system here left
    // every scripted gas domain, MSF gather and coupling stage completely
    // unstepped even though rt.fluid.step() returned success. Use the same
    // one-shot scheduler path as the main timeline so system ordering, force
    // snapshots and all registered runtimes are exercised exactly once.
    g_ctx->scene.syncSimulationWorld();
    // Scripts panel runs while the timeline is commonly paused. stepOnce()
    // intentionally honours that global mode and would otherwise return before
    // executing any system, despite this API being an explicit manual-step
    // request. Temporarily unpause only for this call and restore the user's
    // timeline state immediately afterwards.
    const auto previousMode = g_ctx->scene.simulation_world.getMode();
    if (previousMode == RayTrophiSim::SimulationMode::Paused)
        g_ctx->scene.simulation_world.setMode(RayTrophiSim::SimulationMode::Realtime);
    g_ctx->scene.simulation_world.stepOnce(dt);
    if (previousMode == RayTrophiSim::SimulationMode::Paused)
        g_ctx->scene.simulation_world.setMode(previousMode);
    return Result::success();
}

} // namespace rtapi
