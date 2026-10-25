#include "RtApiInternal.h"
#include "Api/SourceSplatGeometryApi.h"
#include "Fluid/GrainMeshNear.h"
#include "Fluid/SourceSplatRenderPlan.h"
#include "ProjectManager.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>

namespace rtapi {
namespace {
class SourceGeometryCommand final : public SceneCommand {
public:
    uint64_t uid = 0;
    RayTrophiSim::Fluid::SourceSplatGeometry before;
    RayTrophiSim::Fluid::SourceSplatGeometry after;

    void apply(UIContext& context, const RayTrophiSim::Fluid::SourceSplatGeometry& value) {
        for (auto& system : context.scene.particle_systems) {
            if (!system.runtime) {
                continue;
            }
            for (auto& source : system.runtime->flowSources()) {
                if (source.timeline_uid != uid) {
                    continue;
                }
                source.splat_geometry = value;
                context.scene.requestSimulationTimelineRenderResync();
                context.renderer.resetCPUAccumulation();
                if (context.backend_ptr) {
                    context.backend_ptr->resetAccumulation();
                }
                context.start_render = true;
                ProjectManager::getInstance().markModified();
                return;
            }
        }
    }
    void execute(UIContext& context) override {
        apply(context, after);
    }
    void undo(UIContext& context) override {
        apply(context, before);
    }
    Type getType() const override {
        return Type::Generic;
    }
    std::string getDescription() const override {
        return "Set Emitter Splat Geometry";
    }
};
} // namespace

Result validateSourceSplatGeometry(const RayTrophiSim::Fluid::SourceSplatGeometry& geometry) {
    if (geometry.mode != "sphere" && geometry.mode != "scene_object" &&
        geometry.mode != "mesh_grains" && geometry.mode != "domain") {
        return Result::fail("splat_geometry.mode must be sphere, scene_object or mesh_grains "
                            "(domain is reserved for legacy inheritance)");
    }
    if (!std::isfinite(geometry.near_distance) || geometry.near_distance <= 0.0f) {
        return Result::fail("splat_geometry.near_distance must be finite and > 0 m");
    }
    if (geometry.mode == "sphere" || geometry.mode == "domain") {
        return Result::success();
    }
    if (geometry.objects.empty()) {
        return Result::fail("scene_object and mesh_grains need "
                            "at least one exemplar object");
    }
    if (!g_ctx) {
        return notBound();
    }
    std::set<std::string> unique;
    for (const auto& name : geometry.objects) {
        if (!unique.insert(name).second) {
            return Result::fail("duplicate splat geometry object: " + name);
        }
        std::vector<RayTrophiSim::Fluid::NormalizedSceneFace> faces;
        if (!g_ctx->scene.hasLiveSimulationObject(name) ||
            !RayTrophiSim::Fluid::gatherNormalizedSceneFaces(
                g_ctx->scene.world.objects, name, faces)) {
            return Result::fail("splat geometry needs a live, nonempty flat scene mesh: " + name);
        }
    }
    return Result::success();
}

void finalizeSourceSplatProxyStats(
    const RayTrophiSim::SimulationGridDomainDesc& d, FluidDomainInfo& info) {
    const auto& runtime = g_ctx->scene.ensureParticleSimulationSystem();
    const auto& domains = runtime.gridDomains();
    const auto found = std::find_if(domains.begin(), domains.end(),
        [&](const auto& candidate) { return candidate.name == d.name; });
    const int index = found == domains.end() ? -1 :
        static_cast<int>(std::distance(domains.begin(), found));
    RayTrophiSim::Fluid::SourceSplatRenderPlan plan(d, runtime.flowSources(), index);
    const auto rendering = plan.renderDomain(d);
    const bool refines_primary = rendering.fluid_particle_geometry_mode == 0;
    const bool refines_whitewater = d.fluid_foam_params.enabled;
    info.virtual_grains_effective = refines_primary || refines_whitewater
        ? static_cast<int>(RayTrophiSim::Fluid::limitFluidRenderProxyChildren(
              static_cast<uint32_t>(info.virtual_grains_requested),
              RayTrophiSim::Fluid::fluidRenderProxyCarrierCapacity(
                  d.fluid_max_particles,
                  d.fluid_foam_params.enabled
                      ? d.fluid_foam_params.max_foam
                      : 0u),
              info.virtual_grain_budget))
        : 1;
    info.primary_visual_children = refines_primary
        ? std::min(info.virtual_grains_effective, static_cast<int>(
              plan.visualChildren(d)))
        : 1;
    info.whitewater_visual_children = refines_whitewater
        ? info.virtual_grains_effective
        : 1;
    uint64_t primary_spheres = 0;
    uint64_t rendered_spheres = refines_primary
        ? static_cast<uint64_t>(info.particle_count) *
              static_cast<uint64_t>(info.primary_visual_children)
        : static_cast<uint64_t>(info.particle_count);
    primary_spheres = rendered_spheres;
    if (info.views_measured) {
        rendered_spheres = 0;
        primary_spheres = 0;
        for (const auto& view : info.views) {
            if (view.view == "splat") {
                const uint64_t spheres = view.particles *
                    static_cast<uint64_t>(info.primary_visual_children);
                rendered_spheres += spheres;
                primary_spheres += spheres;
                rendered_spheres += view.whitewater *
                    static_cast<uint64_t>(info.whitewater_visual_children);
            }
        }
    }
    const uint64_t mesh_spheres = info.grain_mesh_near_instances *
        static_cast<uint64_t>(info.primary_visual_children);
    info.virtual_grain_count = rendered_spheres - std::min(primary_spheres, mesh_spheres);
    info.virtual_grain_rt_estimated_bytes = info.virtual_grain_count *
        RayTrophiSim::Fluid::kFluidRenderProxyRtBytesPerSphereEstimate;
}

Result makeSourceSplatGeometryCommand(const std::string& name,
    const RayTrophiSim::Fluid::SourceSplatGeometry& geometry,
    std::unique_ptr<SceneCommand>& output) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    const auto status = validateSourceSplatGeometry(geometry);
    if (!status.ok) {
        return status;
    }
    for (const auto& source : g_ctx->scene.ensureParticleSimulationSystem().flowSources()) {
        if (source.name != name) {
            continue;
        }
        auto command = std::make_unique<SourceGeometryCommand>();
        command->uid = source.timeline_uid;
        command->before = source.splat_geometry;
        command->after = geometry;
        output = std::move(command);
        return Result::success();
    }
    return Result::fail("flow source not found: " + name);
}

Result setSourceSplatGeometry(const std::string& name,
    const RayTrophiSim::Fluid::SourceSplatGeometry& geometry) {
    std::unique_ptr<SceneCommand> command;
    const auto status = makeSourceSplatGeometryCommand(name, geometry, command);
    if (!status.ok) {
        return status;
    }
    command->execute(*g_ctx);
    if (g_history) {
        g_history->record(std::move(command));
    }
    return Result::success();
}

namespace {
class SubstanceGeometryCommand final : public SceneCommand {
public:
    uint32_t system_id = 0;
    std::string domain;
    std::string substance;
    bool had_before = false;
    RayTrophiSim::Fluid::SourceSplatGeometry before;
    RayTrophiSim::Fluid::SourceSplatGeometry after;

    void apply(UIContext& context, bool restore) {
        for (auto& system : context.scene.particle_systems) {
            if (system.id != system_id || !system.runtime) {
                continue;
            }
            for (auto& value : system.runtime->gridDomains()) {
                if (value.name != domain) {
                    continue;
                }
                if (restore && !had_before) {
                    value.fluid_substance_splat_geometry.erase(substance);
                } else {
                    value.fluid_substance_splat_geometry[substance] = restore ? before : after;
                }
                context.scene.requestSimulationTimelineRenderResync();
                context.renderer.resetCPUAccumulation();
                if (context.backend_ptr) {
                    context.backend_ptr->resetAccumulation();
                }
                context.start_render = true;
                ProjectManager::getInstance().markModified();
                return;
            }
        }
    }
    void execute(UIContext& context) override {
        apply(context, false);
    }
    void undo(UIContext& context) override {
        apply(context, true);
    }
    Type getType() const override {
        return Type::Generic;
    }
    std::string getDescription() const override {
        return "Set Substance Splat Geometry";
    }
};
}

Result getSubstanceSplatGeometry(const std::string& domain, const std::string& substance,
    RayTrophiSim::Fluid::SourceSplatGeometry& geometry) {
    if (!g_ctx) {
        return notBound();
    }
    if (substance.empty()) {
        return Result::fail("substance must not be empty");
    }
    for (const auto& value : g_ctx->scene.ensureParticleSimulationSystem().gridDomains()) {
        if (value.name != domain) {
            continue;
        }
        if (!RayTrophiSim::simulationDomainHasLiquid(value.type)) {
            return Result::fail("substance Splat geometry requires a liquid/Matter domain");
        }
        const auto found = value.fluid_substance_splat_geometry.find(substance);
        geometry = found == value.fluid_substance_splat_geometry.end()
            ? RayTrophiSim::Fluid::SourceSplatGeometry{} : found->second;
        return Result::success();
    }
    return Result::fail("domain not found: " + domain);
}

Result setSubstanceSplatGeometry(const std::string& domain, const std::string& substance,
    const RayTrophiSim::Fluid::SourceSplatGeometry& geometry) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    if (geometry.mode == "domain") {
        return Result::fail("substance geometry must be sphere, scene_object or mesh_grains");
    }
    const auto status = validateSourceSplatGeometry(geometry);
    if (!status.ok) {
        return status;
    }
    RayTrophiSim::Fluid::SourceSplatGeometry current;
    const auto located = getSubstanceSplatGeometry(domain, substance, current);
    if (!located.ok) {
        return located;
    }
    const auto runtime = g_ctx->scene.getParticleSimulationSystem();
    for (const auto& system : g_ctx->scene.particle_systems) {
        if (system.runtime != runtime) {
            continue;
        }
        for (const auto& value : runtime->gridDomains()) {
            if (value.name != domain) {
                continue;
            }
            auto command = std::make_unique<SubstanceGeometryCommand>();
            command->system_id = system.id;
            command->domain = domain;
            command->substance = substance;
            command->had_before = value.fluid_substance_splat_geometry.count(substance) != 0;
            command->before = current;
            command->after = geometry;
            command->execute(*g_ctx);
            if (g_history) {
                g_history->record(std::move(command));
            }
            return Result::success();
        }
    }
    return Result::fail("particle system not found");
}

} // namespace rtapi
