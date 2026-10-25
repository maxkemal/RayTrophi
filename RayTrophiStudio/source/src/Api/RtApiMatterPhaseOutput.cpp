#include "RtApiInternal.h"
#include "Api/MatterPhaseOutputApi.h"
#include "ProjectManager.h"

namespace rtapi {
namespace {
using namespace RayTrophiSim::Fluid;

class PhaseOutputCommand final : public SceneCommand {
public:
    uint32_t system_id = 0;
    std::string domain;
    MatterPhaseOutput before;
    MatterPhaseOutput after;

    void apply(UIContext& context, bool restore) {
        for (auto& system : context.scene.particle_systems) {
            if (system.id != system_id || !system.runtime) {
                continue;
            }
            for (auto& value : system.runtime->gridDomains()) {
                if (value.name != domain) {
                    continue;
                }
                value.fluid_phase_output = restore ? before : after;
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
        return "Set Matter Phase Output";
    }
};
}

Result getPhaseOutput(const std::string& domain,
    RayTrophiSim::Fluid::MatterPhaseOutput& output) {
    if (!g_ctx) {
        return notBound();
    }
    for (const auto& value : g_ctx->scene.ensureParticleSimulationSystem().gridDomains()) {
        if (value.name != domain) {
            continue;
        }
        if (value.type != RayTrophiSim::SimulationDomainType::Matter) {
            return Result::fail("phase output requires a Matter domain");
        }
        output = value.fluid_phase_output;
        return Result::success();
    }
    return Result::fail("domain not found: " + domain);
}

Result setPhaseOutput(const std::string& domain,
    const RayTrophiSim::Fluid::MatterPhaseOutput& output) {
    using namespace RayTrophiSim::Fluid;
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    if (output.solid >= LabelRoute::Count || output.liquid >= LabelRoute::Count ||
        (output.gas != LabelRoute::Follow && output.gas != LabelRoute::Fog &&
         output.gas != LabelRoute::Hidden)) {
        return Result::fail("invalid phase output route");
    }
    MatterPhaseOutput current;
    const auto status = getPhaseOutput(domain, current);
    if (!status.ok) {
        return status;
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
            auto command = std::make_unique<PhaseOutputCommand>();
            command->system_id = system.id;
            command->domain = domain;
            command->before = current;
            command->after = output;
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
