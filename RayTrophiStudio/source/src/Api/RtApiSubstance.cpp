#include "Api/RtApi.h"
#include "RtApiInternal.h"
#include "SubstanceLibrary.h"
#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/MatterSubstanceState.h"

namespace rtapi {

namespace {

// Every scene entry that names a substance, as "kind 'owner'" strings.
std::vector<std::string> substanceReferences(const std::string& name) {
    std::vector<std::string> refs;
    auto& runtime = scriptSimulationRuntime();
    for (const auto& source : runtime.flowSources()) {
        if (source.fluid_substance == name) refs.push_back("flow source '" + source.name + "'");
    }
    for (const auto& collider : runtime.colliders()) {
        if (collider.msf_substance == name) refs.push_back("collider '" + collider.name + "'");
    }
    for (const auto& domain : runtime.gridDomains()) {
        if (domain.fluid_params.default_substance == name)
            refs.push_back("domain '" + domain.name + "' default substance");
        for (const auto& binding : domain.fluid_substance_materials) {
            if (binding.substance == name) refs.push_back("domain '" + domain.name + "' binding");
        }
    }
    for (const auto& domain : g_ctx->scene.fluid_objects) {
        if (domain.params.default_substance == name) {
            refs.push_back("fluid object '" + domain.name + "' default substance");
        }
    }
    return refs;
}

// Refresh physical readback immediately after a library edit, including
// descendants. Do not reapply solver hints: the user's numerical tuning stays.
void refreshDomainMaterials() {
    auto& runtime = scriptSimulationRuntime();
    const auto scale = runtime.worldThermal().scale();
    std::string error;
    for (auto& domain : runtime.gridDomains()) {
        if (RayTrophiSim::simulationDomainHasLiquid(domain.type)) {
            RayTrophiSim::Fluid::resolveGridDomainSubstancePhysics(
                domain, domain.voxel_size, scale, error);
        }
    }
    for (auto& domain : g_ctx->scene.fluid_objects) {
        RayTrophiSim::Fluid::resolveDomainSubstancePhysics(
            domain.params, domain.voxel_size, scale, error);
    }
}

} // namespace

Result listSubstances(std::vector<SubstanceSummary>& out) {
    out.clear();
    for (const RayTrophiSim::SubstanceProfile* profile : RayTrophiSim::substanceProfiles()) {
        SubstanceSummary row;
        row.name = profile->name;
        row.based_on = profile->based_on;
        row.category = RayTrophiSim::substanceCategoryName(profile->category);
        row.builtin = profile->based_on.empty();
        out.push_back(std::move(row));
    }
    return Result::success();
}

Result getSubstance(const std::string& name, std::string& out_json) {
    const RayTrophiSim::SubstanceProfile* profile = RayTrophiSim::tryFindSubstance(name);
    if (!profile) return Result::fail("unknown substance: " + name);
    nlohmann::json out = {
        {"name", profile->name},
        {"based_on", profile->based_on},
        {"builtin", profile->based_on.empty()},
        {"fields", RayTrophiSim::substanceFieldsToJson(*profile)},
        {"overridden", RayTrophiSim::substanceOverriddenFields(name)},
    };
    out_json = out.dump();
    return Result::success();
}

Result deriveSubstance(const std::string& name, const std::string& based_on) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    std::string error;
    if (!RayTrophiSim::deriveSubstance(name, based_on, error)) return Result::fail(error);
    // Nothing references a new substance yet; no simulation to invalidate.
    return Result::success();
}

Result setSubstanceFields(const std::string& name, const std::string& fields_json) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    nlohmann::json fields = nlohmann::json::parse(fields_json, nullptr, false);
    if (fields.is_discarded()) return Result::fail("substance fields: invalid JSON");
    std::string error;
    // Fixed when a grain is born: the owner (transport) and the grain mass
    // (packing). Live particles of this substance must be reset first.
    for (const char* key : {"granular_transport", "grain_packing_fraction"}) {
        if (!fields.contains(key)) continue;
        auto& runtime = scriptSimulationRuntime();
        if (!RayTrophiSim::Fluid::validateMatterTransportEdit(name, runtime.gridDomains(),
                runtime.gridDomainStates(), error, key)) {
            return Result::fail(error);
        }
    }
    if (!RayTrophiSim::patchSubstance(name, fields, error)) return Result::fail(error);
    refreshDomainMaterials();
    invalidateScriptSimulation();
    return Result::success();
}

Result removeSubstance(const std::string& name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!RayTrophiSim::isBuiltinSubstance(name) && RayTrophiSim::tryFindSubstance(name)) {
        const auto refs = substanceReferences(name);
        if (!refs.empty()) {
            std::string list;
            for (const auto& ref : refs) list += (list.empty() ? "" : ", ") + ref;
            return Result::fail("substance '" + name + "' is still used by " + list);
        }
    }
    std::string error;
    if (!RayTrophiSim::removeSubstance(name, error)) return Result::fail(error);
    invalidateScriptSimulation();
    return Result::success();
}

} // namespace rtapi
