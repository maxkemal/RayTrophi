#include "Fluid/MatterMaterialResolution.h"
#include "Fluid/MatterConstitutiveState.h"
#include "Fluid/ElasticThermalResponse.h"
#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {

const char* backendName(SimulationDomainBackend backend) {
    switch (backend) {
        case SimulationDomainBackend::CPU_Dense: return "cpu_dense";
        case SimulationDomainBackend::CPU_SparseVDB: return "cpu_sparse_vdb";
        case SimulationDomainBackend::GPU_Compute: return "cuda";
        case SimulationDomainBackend::GPU_Vulkan: return "vulkan";
        default: return "unknown";
    }
}

const char* ownerName(MatterTransportOwner owner) {
    switch (owner) {
        case MatterTransportOwner::Fluid: return "fluid";
        case MatterTransportOwner::Grain: return "dem";
        case MatterTransportOwner::Mpm: return "mpm";
        case MatterTransportOwner::Obstacle: return "obstacle";
        default: return "unknown";
    }
}

const char* requestedTransport(const SubstanceProfile& profile) {
    switch (profile.default_constitutive_model) {
        case MatterConstitutiveModel::Fluid: return "fluid";
        case MatterConstitutiveModel::Elastic: return "mpm";
        case MatterConstitutiveModel::Granular:
            return profile.granular_transport == MatterGranularTransport::Dem ? "dem" : "mpm";
        default: return "auto";
    }
}

struct CarrierGroup {
    std::size_t count = 0;
    std::size_t known_temperature = 0;
    std::size_t known_fraction = 0;
    double temperature_sum = 0.0;
    double fraction_sum = 0.0;
    std::map<std::string, std::size_t> phase_counts;
    std::map<std::string, std::size_t> phase_basis_counts;
    std::map<std::string, std::size_t> selection_basis_counts;
    std::size_t model_mismatch_particles = 0;
    std::size_t softening_known = 0;
    float softening_min = 1.0f;
    float softening_max = 0.0f;
};

} // namespace

SubstanceViscosityResolution resolveSubstanceViscosity(const SubstanceProfile& profile,
    float voxel_size, bool thermal_liquid_enabled, bool granular_enabled) {
    SubstanceViscosityResolution result;
    result.requested = std::max(profile.liquid_kinematic_viscosity, 0.0f);
    result.effective = result.requested;
    result.voxel_size = voxel_size > 0.0f ? voxel_size : 0.05f;
    // Keep the original float expression so the solver's threshold is unchanged.
    result.reference_diffusion = result.requested * kViscosityReferenceDt /
        (result.voxel_size * result.voxel_size);
    if (granular_enabled) {
        result.effective = 0.0f;
        result.reason = "granular_skeleton";
    } else if (result.requested == 0.0f) {
        result.reason = "zero_requested";
    } else if (!thermal_liquid_enabled &&
               result.reference_diffusion < kUnresolvedViscosityDiffusion) {
        result.effective = 0.0f;
        result.reason = "below_resolution_threshold";
    } else if (thermal_liquid_enabled) {
        result.reason = "thermal_curve_base";
    }
    return result;
}

bool inspectMatterMaterialResolution(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState* state, nlohmann::json& output, std::string& error) {
    const auto& params = domain.fluid_params;
    const auto* profile = domainSubstance(params);
    if (!profile) {
        error = "domain default substance '" + params.default_substance +
            "' is not in the substance library";
        return false;
    }
    if (!std::isfinite(domain.voxel_size) || domain.voxel_size <= 0.0f ||
        !std::isfinite(profile->liquid_kinematic_viscosity) ||
        profile->liquid_kinematic_viscosity < 0.0f ||
        !std::isfinite(params.kinematic_viscosity) || params.kinematic_viscosity < 0.0f) {
        error = "material resolution requires a positive finite voxel size and "
            "nonnegative finite viscosities";
        return false;
    }
    const bool measured = state && state->valid;
    const auto viscosity = resolveSubstanceViscosity(*profile, domain.voxel_size,
        params.thermal_liquid_enabled,
        profile->default_constitutive_model == MatterConstitutiveModel::Granular);
    if (!std::isfinite(viscosity.reference_diffusion)) {
        error = "material resolution: viscosity reference diffusion is not finite";
        return false;
    }
    nlohmann::json report{
        {"schema_version", 1}, {"scope", "matter_domain"},
        {"basis", "configuration_and_synchronized_carriers"},
        {"execution_measured", false}, {"carriers_measured", measured},
        {"backend_requested", backendName(domain.backend)},
        {"backend_executed", nullptr},
        {"default_substance", profile->name}, {"based_on", profile->based_on},
        {"phase_mechanics", describeMatterPhaseMechanics(*profile)},
        {"requested_model", matterConstitutiveModelName(profile->default_constitutive_model)},
        {"requested_transport", requestedTransport(*profile)},
        {"viscosity", {
            {"scope", "domain_base_not_per_particle_thermal_viscosity"},
            {"requested_m2_s", viscosity.requested},
            {"policy_effective_m2_s", viscosity.effective},
            {"configured_m2_s", params.kinematic_viscosity},
            {"configuration_matches_policy", params.kinematic_viscosity == viscosity.effective},
            {"reason", viscosity.reason}, {"voxel_size_m", viscosity.voxel_size},
            {"reference_dt_s", kViscosityReferenceDt},
            {"reference_diffusion", viscosity.reference_diffusion},
            {"skip_threshold", kUnresolvedViscosityDiffusion},
            {"thermal_curve_enabled", params.thermal_liquid_enabled}}},
        {"mpm_parameter_scope", "shared_domain_material"},
        {"carriers", nlohmann::json::array()}, {"notes", nlohmann::json::array()}};
    report["notes"].push_back("This report does not assign particle mechanics to "
        "thermal-only scene objects or measure backend execution.");
    if (!measured) {
        report["notes"].push_back("No synchronized particle state; phase and active "
            "transport are unknown.");
        output = std::move(report);
        error.clear();
        return true;
    }

    const auto& particles = state->particles;
    const auto count = particles.size();
    const auto validSize = [count](std::size_t size) { return size == 0 || size == count; };
    if (!validSize(particles.substance_tag.size()) ||
        !validSize(particles.constitutive_model.size()) ||
        !validSize(particles.temperature.size()) ||
        !validSize(particles.fusion_solid_fraction.size()) || !validSize(particles.flags.size())) {
        error = "material resolution: particle sidecar cardinality mismatch";
        return false;
    }
    using Key = std::tuple<uint32_t, MatterConstitutiveModel, MatterTransportOwner>;
    std::map<Key, CarrierGroup> groups;
    std::set<std::string> mpm_substances;
    std::vector<uint32_t> static_tags;
    collectMatterStaticTags(domain, static_tags);
    for (std::size_t index = 0; index < count; ++index) {
        const uint32_t tag = particles.substance_tag.empty()
            ? kSubstanceUntagged : particles.substance_tag[index];
        auto model = particles.constitutive_model.empty() ? MatterConstitutiveModel::Auto
            : static_cast<MatterConstitutiveModel>(particles.constitutive_model[index]);
        if (model != MatterConstitutiveModel::Auto && model != MatterConstitutiveModel::Fluid &&
            model != MatterConstitutiveModel::Granular && model != MatterConstitutiveModel::Elastic) {
            error = "material resolution: unknown particle constitutive model";
            return false;
        }
        const float kelvin = particles.temperature.empty() ? 0.0f : particles.temperature[index];
        // Zero is an unwritten legacy temperature. NaN fusion fraction is the
        // documented unresolved state; neither is silently turned into a phase.
        if (!std::isfinite(kelvin) || kelvin < 0.0f) {
            error = "material resolution: invalid particle temperature";
            return false;
        }
        if (model == MatterConstitutiveModel::Auto) {
            model = substanceBirthModel(tag, params, kelvin);
        }
        const auto owner = isMatterObstacle(particles, index,
            static_tags.empty() ? nullptr : static_tags.data(), static_tags.size())
            ? MatterTransportOwner::Obstacle : substanceTransportOwner(tag, model, params);
        auto& group = groups[{tag, model, owner}];
        ++group.count;
        if (kelvin > 0.0f) {
            ++group.known_temperature;
            group.temperature_sum += kelvin;
        }
        if (!particles.fusion_solid_fraction.empty()) {
            const float fraction = particles.fusion_solid_fraction[index];
            if (!std::isnan(fraction)) {
                if (!std::isfinite(fraction) || fraction < 0.0f || fraction > 1.0f) {
                    error = "material resolution: invalid fusion solid fraction";
                    return false;
                }
                ++group.known_fraction;
                group.fraction_sum += fraction;
            }
        }
        const float fraction = particles.fusion_solid_fraction.empty()
            ? std::numeric_limits<float>::quiet_NaN() : particles.fusion_solid_fraction[index];
        const auto resolved = resolveMatterConstitutiveState(particleSubstance(tag, params),
            params.granular_enabled, kelvin, fraction);
        ++group.phase_counts[resolved.phase];
        ++group.phase_basis_counts[resolved.phase_basis];
        ++group.selection_basis_counts[resolved.basis];
        if (resolved.model != model) {
            ++group.model_mismatch_particles;
        }
        if (model == MatterConstitutiveModel::Elastic && owner == MatterTransportOwner::Mpm &&
            index < particles.granular_softening.size()) {
            const float soft = particles.granular_softening[index];
            if (std::isfinite(soft) && soft >= 0.0f && soft <= 1.0f) {
                ++group.softening_known;
                group.softening_min = std::min(group.softening_min, soft);
                group.softening_max = std::max(group.softening_max, soft);
            }
        }
        if (owner == MatterTransportOwner::Mpm) {
            if (const auto* substance = particleSubstance(tag, params)) {
                mpm_substances.insert(substance->name);
            }
        }
    }
    auto shared_mpm_params = params;
    resolveParticleGranularPhysics(shared_mpm_params, particles);
    for (const auto& [key, group] : groups) {
        const auto [tag, model, owner] = key;
        const auto* substance = particleSubstance(tag, params);
        const auto responseAt = [&](float soft) {
            const auto& material = shared_mpm_params;
            const float yield = material.elastic_yield_strain > 0.0f
                ? std::max(1.0e-6f, material.elastic_yield_strain *
                    Granular::elasticThermalScale(soft, material.elastic_thermal_yield_power))
                : 0.0f;
            const float relaxation = material.elastic_relaxation_s > 0.0f
                ? std::max(1.0e-6f, material.elastic_relaxation_s *
                    Granular::elasticThermalScale(soft, material.elastic_thermal_relaxation_power))
                : 0.0f;
            return nlohmann::json{{"young_pa", std::max(1.0f, material.granular_young_modulus * soft)},
                {"viscosity_pa_s", material.elastic_viscosity_pa_s *
                    Granular::elasticThermalScale(soft, material.elastic_thermal_viscosity_power)},
                {"yield_strain", yield}, {"relaxation_s", relaxation}};
        };
        nlohmann::json thermal_response{{"known_particles", group.softening_known},
            {"basis", "shared_mpm_parameters_before_stability_caps_and_damage"}};
        if (group.softening_known > 0) {
            thermal_response["minimum_softening"] = group.softening_min;
            thermal_response["maximum_softening"] = group.softening_max;
            thermal_response["at_minimum_softening"] = responseAt(group.softening_min);
            thermal_response["at_maximum_softening"] = responseAt(group.softening_max);
        }
        report["carriers"].push_back({
            {"substance_tag", tag}, {"substance", substance ? substance->name : ""},
            {"identity_resolution", tag == kSubstanceUntagged ? "domain_default"
                : tryFindSubstanceByTag(tag) ? "tag" : "unknown_tag_domain_fallback"},
            {"particles", group.count}, {"model", matterConstitutiveModelName(model)},
            {"model_basis", "synchronized_model_or_birth_fallback_for_auto"},
            {"transport", ownerName(owner)},
            {"phase_mechanics", substance ? describeMatterPhaseMechanics(*substance)
                : nlohmann::json(nullptr)},
            {"phase_counts", group.phase_counts},
            {"phase_basis_counts", group.phase_basis_counts},
            {"selection_basis_counts", group.selection_basis_counts},
            {"model_mismatch_particles", group.model_mismatch_particles},
            {"elastic_thermal_response", thermal_response},
            {"temperature_known_particles", group.known_temperature},
            {"mean_temperature_k", group.known_temperature
                ? nlohmann::json(group.temperature_sum / group.known_temperature)
                : nlohmann::json(nullptr)},
            {"fusion_known_particles", group.known_fraction},
            {"mean_fusion_solid_fraction", group.known_fraction
                ? nlohmann::json(group.fraction_sum / group.known_fraction)
                : nlohmann::json(nullptr)}});
    }
    const auto owners = inspectMatterDomainOwners(particles, domain);
    report["transport_ready"] = owners.ready;
    report["transport_reason"] = owners.reason;
    report["mpm_substances"] = mpm_substances;
    if (mpm_substances.size() > 1) {
        report["notes"].push_back("Multiple MPM substances share domain material parameters; "
            "this is not independent per-substance constitutive evaluation.");
    }
    output = std::move(report);
    error.clear();
    return true;
}

} // namespace RayTrophiSim::Fluid
