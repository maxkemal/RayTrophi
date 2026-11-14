#include "Api/MatterSeedSourcesApi.h"
#include "Fluid/MatterSeedPlanning.h"
#include "Fluid/MatterSeedSources.h"
#include "MaterialStateField.h"
#include "RtApiInternal.h"
#include "SceneSelection.h"
#include "Renderer.h"

#include <algorithm>
#include <limits>

namespace rtapi {

Result validateSingleRegionSeedRequest(const std::string &domain) {
    nlohmann::json result;
    const auto status = matterSeedSources(domain, "list", 0, nlohmann::json::object(), result);
    if (!status.ok) {
        return status;
    }
    if (result.at("enabled").get<bool>()) {
        return Result::fail("Layered initial sources are active; use fluid.apply_seed_sources "
                            "or clear the seed recipe before using a single-region seed");
    }
    return Result::success();
}

Result matterSeedSources(const std::string &domain, const std::string &operation, uint64_t id,
                         const nlohmann::json &settings, nlohmann::json &output) {
    if (!g_ctx) {
        return notBound();
    }
    if ((operation == "update" || operation == "remove") && id == 0) {
        return Result::fail("seed source ID must be a positive integer");
    }
    if (operation != "list" && operation != "select" && renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    FluidDomainInfo info;
    auto result = getFluidDomain(domain, info);
    if (!result.ok) {
        return result;
    }
    auto &runtime = g_ctx->scene.ensureParticleSimulationSystem();
    auto &domains = runtime.gridDomains();
    auto found = std::find_if(domains.begin(), domains.end(), [&](const auto &value) {
        return value.name == info.name && value.type == RayTrophiSim::SimulationDomainType::Matter;
    });
    if (found == domains.end()) {
        return Result::fail("Matter domain not found: " + domain);
    }
    const auto index = static_cast<std::size_t>(found - domains.begin());
    if (!settings.is_object()) {
        return Result::fail("seed settings must be an object");
    }
    if (operation == "select") {
        if (id != 0 &&
            std::none_of(found->matter_seed_sources.begin(), found->matter_seed_sources.end(),
                         [id](const auto &source) { return source.id == id; })) {
            return Result::fail("seed source ID not found");
        }
        found->matter_seed_selected_source = id;
        g_ctx->selection.selectSimulationDomain(g_ctx->scene.active_particle_system_index,
                                                static_cast<int>(index), found->name);
    }
    auto candidate = *found;
    auto &sources = candidate.matter_seed_sources;
    std::string error;
    if (operation == "settings") {
        for (auto item = settings.begin(); item != settings.end(); ++item) {
            if (item.key() == "auto_capacity" && item.value().is_boolean()) {
                candidate.matter_seed_auto_capacity = item.value().get<bool>();
            } else if (item.key() == "memory_guard" && item.value().is_boolean()) {
                candidate.matter_seed_memory_guard = item.value().get<bool>();
            } else if (item.key() == "particles_per_cell" && item.value().is_number_integer() &&
                       item.value().get<double>() >= 1 && item.value().get<double>() <= 64) {
                candidate.fluid_seed_particles_per_cell = item.value().get<int>();
            } else if (item.key() == "settle" && item.value().is_boolean()) {
                candidate.fluid_seed_settle = item.value().get<bool>();
            } else if (item.key() == "settle_max_steps" && item.value().is_number_integer() &&
                       item.value().get<double>() >= 1 && item.value().get<double>() <= 4000) {
                candidate.fluid_settle_max_steps = item.value().get<int>();
            } else if (item.key() == "settle_tolerance" && item.value().is_number() &&
                       item.value().get<double>() >= 0.0 && item.value().get<double>() <= 1.0) {
                candidate.fluid_settle_displacement_m = item.value().get<float>();
            } else {
                return Result::fail("invalid seed recipe setting: " + item.key());
            }
        }
    } else if (operation == "add") {
        candidate.matter_seed_sources_enabled = true;
        RayTrophiSim::Fluid::MatterSeedSource source;
        for (const auto &value : sources) {
            source.id = std::max(source.id, value.id);
        }
        if (source.id == std::numeric_limits<uint64_t>::max()) {
            return Result::fail("seed source ID space exhausted");
        }
        ++source.id;
        source.name = "Seed Layer " + std::to_string(source.id);
        source.substance = candidate.fluid_params.default_substance;
        const Vec3 extent = candidate.bounds_max - candidate.bounds_min;
        const float bottom = sources.empty() ? candidate.bounds_min.y : sources.back().bounds_max.y;
        source.bounds_min = Vec3(candidate.bounds_min.x, bottom, candidate.bounds_min.z);
        source.bounds_max =
            Vec3(candidate.bounds_max.x, bottom + extent.y * 0.2f, candidate.bounds_max.z);
        if (!RayTrophiSim::Fluid::parseMatterSeedSource(settings, source, error)) {
            return Result::fail(error);
        }
        sources.push_back(std::move(source));
    } else if (operation == "update" || operation == "remove") {
        const auto target = std::find_if(sources.begin(), sources.end(),
                                         [id](const auto &source) { return source.id == id; });
        if (target == sources.end()) {
            return Result::fail("seed source ID not found");
        }
        if (operation == "remove") {
            sources.erase(target);
        } else if (!RayTrophiSim::Fluid::parseMatterSeedSource(settings, *target, error)) {
            return Result::fail(error);
        }
    } else if (operation != "list" && operation != "apply" && operation != "select") {
        return Result::fail("unknown seed source operation: " + operation);
    }
    if (operation != "list" && operation != "select" &&
        !RayTrophiSim::Fluid::validateMatterSeedSources(candidate, error, &runtime.flowSources(),
                                                        static_cast<int>(index))) {
        return Result::fail(error);
    }
    if (operation == "apply" && !candidate.matter_seed_sources_enabled) {
        return Result::fail("add a seed source to activate the layered initial-state recipe");
    }
    if (operation == "apply" && !candidate.enabled) {
        return Result::fail("enable the Matter domain before applying seed sources");
    }
    if (operation != "list" && operation != "select") {
        found->matter_seed_sources = std::move(sources);
        if (operation == "remove" && found->matter_seed_selected_source == id) {
            found->matter_seed_selected_source = 0;
        }
        found->matter_seed_sources_enabled = candidate.matter_seed_sources_enabled;
        found->matter_seed_auto_capacity = candidate.matter_seed_auto_capacity;
        found->matter_seed_memory_guard = candidate.matter_seed_memory_guard;
        found->fluid_seed_particles_per_cell = candidate.fluid_seed_particles_per_cell;
        found->fluid_seed_settle = candidate.fluid_seed_settle;
        found->fluid_settle_max_steps = candidate.fluid_settle_max_steps;
        found->fluid_settle_displacement_m = candidate.fluid_settle_displacement_m;
        if (operation == "apply") {
            found->fluid_pending_seed = true;
            runtime.synchronizeGridDomainsNow();
            const auto &states = runtime.gridDomainStates();
            if (index >= states.size() || !states[index].valid ||
                domains[index].fluid_pending_seed) {
                domains[index].fluid_pending_seed = false;
                return Result::fail("seed domain has no valid particle grid");
            }
            if (!states[index].seed_stats.error.empty()) {
                return Result::fail(states[index].seed_stats.error);
            }
            domains[index].fluid_reseed_on_reset = true;
            invalidateScriptSimulation();
            g_ctx->scene.adoptSeedAsSimulationStart();
        }
        if (g_ctx->backend_ptr) {
            g_ctx->backend_ptr->resetAccumulation();
        }
        g_ctx->renderer.resetCPUAccumulation();
        g_ctx->start_render = true;
    }
    const auto &descriptor = domains[index];
    const float ambient =
        RayTrophiSim::fluidDomainAmbientKelvin(descriptor, runtime.worldThermal());
    nlohmann::json rows = nlohmann::json::array();
    for (const auto &source : descriptor.matter_seed_sources) {
        auto row = RayTrophiSim::Fluid::matterSeedSourceToJson(source);
        row["resolved_birth_kelvin"] =
            RayTrophiSim::Fluid::matterSeedBirthKelvin(source, descriptor, ambient);
        rows.push_back(std::move(row));
    }
    output = {
        {"domain", descriptor.name},
        {"selected_source_id", descriptor.matter_seed_selected_source},
        {"settings",
         {{"auto_capacity", descriptor.matter_seed_auto_capacity},
          {"memory_guard", descriptor.matter_seed_memory_guard},
          {"particles_per_cell", descriptor.fluid_seed_particles_per_cell},
          {"settle", descriptor.fluid_seed_settle},
          {"settle_max_steps", descriptor.fluid_settle_max_steps},
          {"settle_tolerance", descriptor.fluid_settle_displacement_m}}},
        {"sources", rows},
        {"enabled", descriptor.matter_seed_sources_enabled},
        {"environment_kelvin", ambient},
        {"particles_per_cell", descriptor.fluid_seed_particles_per_cell},
        {"max_particles", descriptor.fluid_max_particles},
        {"pending_recipe_changes", descriptor.matter_seed_sources_enabled &&
                                       descriptor.matter_seed_sources_applied_hash !=
                                           RayTrophiSim::Fluid::matterSeedRecipeHash(descriptor)},
        {"application", "replace_all_layers_then_shared_settle"}};
    try {
        output["plan"] = runtime.matterSeedPlan(descriptor);
        std::string memory_error;
        output["plan"]["memory_check_passed"] =
            RayTrophiSim::Fluid::checkMatterSeedMemory(output["plan"], memory_error);
        output["plan"]["memory_message"] = memory_error;
    } catch (const std::exception &exception) {
        output["plan"] = {{"error", exception.what()}};
    }
    return Result::success();
}

} // namespace rtapi
