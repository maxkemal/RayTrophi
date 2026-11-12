#include "RtApiInternal.h"
#include "Api/DomainVisualApi.h"
#include "Fluid/DomainVisualService.h"
#include "ProjectManager.h"

namespace rtapi {
namespace {
using namespace RayTrophiSim;
using namespace RayTrophiSim::Fluid;

class VisualCommand final : public SceneCommand {
public:
    std::string domain;
    DomainVisualSettings before;
    DomainVisualSettings after;
    std::shared_ptr<const DomainVisualState> checkpoint;
    uint64_t authored_version = 0;
    double authored_time = 0.0;
    bool reset = false;

    void apply(UIContext& context, bool undoing) {
        auto& runtime = context.scene.ensureParticleSimulationSystem();
        auto& domains = runtime.gridDomains();
        for (std::size_t index = 0; index < domains.size(); ++index) {
            if (domains[index].name != domain) {
                continue;
            }
            domains[index].fluid_visual_samples = undoing ? before : after;
            // A render cache is mutable through a const physical-state view.
            const auto& live = runtime.gridDomainStates();
            if (index < live.size()) {
                if (undoing && live[index].version == authored_version &&
                    live[index].render_physical_time == authored_time) {
                    live[index].visual_samples = checkpoint;
                } else if (!undoing && reset) {
                    live[index].visual_samples.reset();
                }
            }
            // Appearance-only: a timeline resync can restore an older physical
            // cache frame and discard the freshly authored visual checkpoint.
            context.scene.syncParticleRenderInstances(context.particle_display_mode != 1);
            context.renderer.resetCPUAccumulation();
            if (context.backend_ptr) {
                context.backend_ptr->resetAccumulation();
            }
            context.start_render = true;
            ProjectManager::getInstance().markModified();
            return;
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
        return "Set Domain Visual Samples";
    }
};

std::size_t findDomain(const std::string& domain) {
    const auto& domains = g_ctx->scene.ensureParticleSimulationSystem().gridDomains();
    for (std::size_t i = 0; i < domains.size(); ++i) {
        if (domains[i].name == domain) {
            return i;
        }
    }
    return domains.size();
}

Result change(const std::string& domain, const nlohmann::json& patch, bool reset) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    auto& runtime = g_ctx->scene.ensureParticleSimulationSystem();
    auto& domains = runtime.gridDomains();
    const auto index = findDomain(domain);
    if (index == domains.size()) {
        return Result::fail("domain not found: " + domain);
    }
    auto value = domains[index].fluid_visual_samples;
    try {
        from_json(patch, value);
        if (value.enabled && (domains[index].boundary_mode ==
                SimulationGridDomainBoundaryMode::Periodic)) {
            return Result::fail("field samples currently support open or closed boundaries");
        }
        if (value.enabled && domains[index].fluid_params.pore_exchange.wet_appearance_enabled) {
            return Result::fail("field samples do not yet support spatial wet appearance");
        }
        if (value.sampling.max_samples > 4000000) {
            return Result::fail("CPU field sample route supports max_samples up to 4000000");
        }
        const auto required = value.sampling.max_samples * 768ull;
        if (required >= value.working_budget_mb * 1024ull * 1024ull) {
            return Result::fail("sample staging exceeds working_budget_mb");
        }
    } catch (const std::exception& error) {
        return Result::fail(error.what());
    }
    auto command = std::make_unique<VisualCommand>();
    command->domain = domain;
    command->before = domains[index].fluid_visual_samples;
    command->after = value;
    command->reset = reset;
    const auto& states = runtime.gridDomainStates();
    if (index < states.size()) {
        command->checkpoint = states[index].visual_samples;
        command->authored_version = states[index].version;
        command->authored_time = states[index].render_physical_time;
    }
    command->execute(*g_ctx);
    if (g_history) {
        g_history->record(std::move(command));
    }
    return Result::success();
}
} // namespace

Result setDomainVisualSamples(const std::string& domain, const nlohmann::json& patch) {
    return change(domain, patch, false);
}

Result resetDomainVisualSamples(const std::string& domain) {
    return change(domain, nlohmann::json::object(), true);
}

Result getDomainVisualSamples(const std::string& domain, nlohmann::json& output,
    uint64_t offset, uint64_t limit) {
    if (!g_ctx) {
        return notBound();
    }
    if (limit > 256) {
        return Result::fail("visual sample inspection limit must be <= 256");
    }
    auto& runtime = g_ctx->scene.ensureParticleSimulationSystem();
    const auto index = findDomain(domain);
    const auto& domains = runtime.gridDomains();
    const auto& states = runtime.gridDomainStates();
    if (index == domains.size()) {
        return Result::fail("domain not found: " + domain);
    }
    const auto* state = index < states.size() ? &states[index] : nullptr;
    const auto& desc = domains[index];
    if (state && state->valid && desc.fluid_visual_samples.enabled) {
        domainSplatParticles(desc, *state);
    }
    auto result = nlohmann::json{{"settings", desc.fluid_visual_samples},
        {"stats", domainVisualStats(desc, state)}, {"offset", offset},
        {"items", nlohmann::json::array()}};
    uint64_t cursor = 0;
    if (limit && state && state->visual_samples && state->visual_samples->current) {
        for (const auto& entry : state->visual_samples->streams) {
            for (const auto& sample : entry.second.runtime.state().samples) {
                if (cursor++ < offset) {
                    continue;
                }
                if (result["items"].size() >= limit) {
                    break;
                }
                result["items"].push_back({{"uid", sample.uid}, {"stream", entry.first},
                    {"position", sample.position}, {"previous_position", sample.previous_position},
                    {"volume", sample.represented_volume}, {"orientation", sample.orientation}});
            }
            if (result["items"].size() >= limit) {
                break;
            }
        }
    }
    output = std::move(result);
    return Result::success();
}

} // namespace rtapi
