#include "Fluid/MatterSubstanceState.h"

#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/MatterGrain.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace RayTrophiSim::Fluid {

void configureBuiltinMatterTransport(std::vector<SubstanceProfile>& profiles) {
    for (auto& profile : profiles) {
        profile.granular_transport = profile.name == "Sand" || profile.name == "Gravel" ||
            profile.name == "Ice" ? MatterGranularTransport::Dem : MatterGranularTransport::Mpm;
    }
}

MatterGrainOwnership matterGrainOwnership(const SimulationGridDomainDesc& domain,
                                          const std::vector<SimulationFlowSourceDesc>& sources,
                                          int domain_index, bool has_particles) {
    MatterGrainOwnership result;
    result.enabled = domain.fluid_params.grain.enabled;
    if (domain.type != SimulationDomainType::Matter) {
        result.enabled = false;
        return result;
    }
    bool wet_birth = false;
    const auto consider = [&](const std::string& name) {
        if (name.empty()) {
            return;
        }
        const auto* profile = tryFindSubstance(name);
        if (!profile) {
            return;
        }
        if (profile->category == SubstanceCategory::Liquid ||
            profile->category == SubstanceCategory::Fuel) {
            if (result.liquid.empty()) {
                result.liquid = name;
            }
            return;
        }
        if (profile->default_constitutive_model != MatterConstitutiveModel::Granular ||
            profile->granular_transport != MatterGranularTransport::Dem) {
            return;
        }
        if (result.substance.empty()) {
            result.substance = name;
        } else if (name != result.substance) {
            result.notes.push_back(name + " also asks for grains; one grain material per domain: " +
                result.substance + "'s is used.");
        }
    };
    // What is poured comes first: grains carry their source's substance tag,
    // so its material is the one that runs. Disabled sources count too:
    // `enabled` is keyable, and the owner is fixed when the first grain is
    // born. An untagged source pours the Default Substance, considered last.
    for (const auto& source : sources) {
        if (source.domain_index == domain_index &&
            source.phase != SimulationFlowSourceDesc::Phase::Gas) {
            consider(source.fluid_substance);
            wet_birth |= source.grain_birth_saturation > 0.0f;
        }
    }
    for (const auto& binding : domain.fluid_substance_materials) {
        if (binding.phase != SubstancePhase::Solid) {
            consider(binding.substance);
        }
    }
    consider(domain.fluid_params.default_substance);
    result.wanted = !result.substance.empty();
    if (result.wanted) {
        const auto* grain = tryFindSubstance(result.substance);
        const bool target_wet = grain && grain->grain_water_capacity_fraction > 0.0f &&
            (!result.liquid.empty() || wet_birth);
        // Grains born wet may hold water: they keep wet on until a reset.
        result.wet_grains = target_wet ||
            (has_particles && domain.fluid_params.grain.wet_grains);
    }
    if (result.wanted) {
        for (const auto& blocker : matterGrainBlockers(domain)) {
            result.blockers.push_back(blocker.message);
        }
    }
    const bool target = result.wanted && result.blockers.empty();
    // Carrier mass and contact ownership are fixed at birth (as for a
    // granular_transport edit): live particles keep the owner until a reset.
    if (target != result.enabled && has_particles) {
        result.reset_pending = true;
    } else {
        result.enabled = target;
    }
    return result;
}

MatterGrainOwnership matterGrainOwnership(const ParticleSimulationSystem& system,
                                          std::size_t index) {
    const auto& domains = system.gridDomains();
    const auto& states = system.gridDomainStates();
    if (index >= domains.size()) {
        return {};
    }
    return matterGrainOwnership(domains[index], system.flowSources(), static_cast<int>(index),
        index < states.size() && !states[index].particles.empty());
}

bool matterGrainsInUse(const ParticleSimulationSystem& system,
                       const SimulationGridDomainDesc& domain) {
    const auto& domains = system.gridDomains();
    for (std::size_t index = 0; index < domains.size(); ++index) {
        if (&domains[index] == &domain || domains[index].name == domain.name) {
            const auto ownership = matterGrainOwnership(system, index);
            return ownership.enabled || ownership.wanted;
        }
    }
    return domain.fluid_params.grain.enabled;
}

MatterTransportOwner substanceTransportOwner(uint32_t tag, MatterConstitutiveModel model,
                                            const APICSolverParams& params) {
    if (model == MatterConstitutiveModel::Fluid) {
        return MatterTransportOwner::Fluid;
    }
    if (model == MatterConstitutiveModel::Elastic) {
        return MatterTransportOwner::Mpm;
    }
    const auto* profile = particleSubstance(tag, params);
    if (params.grain.enabled && profile &&
        profile->granular_transport == MatterGranularTransport::Dem) {
        return MatterTransportOwner::Grain;
    }
    return MatterTransportOwner::Mpm;
}

MatterOwnerSummary inspectMatterOwners(const FluidParticles& particles,
                                      const APICSolverParams& params) {
    MatterOwnerSummary summary;
    const auto* tags = params.solid_substance_tags;
    for (std::size_t index = 0; index < particles.size(); ++index) {
        const uint32_t tag = index < particles.substance_tag.size()
            ? particles.substance_tag[index] : kSubstanceUntagged;
        auto model = index < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[index])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Auto) {
            const float kelvin = index < particles.temperature.size()
                ? particles.temperature[index] : std::numeric_limits<float>::quiet_NaN();
            model = substanceBirthModel(tag, params, kelvin);
        }
        const auto owner = isMatterObstacle(particles, index,
            tags ? tags->data() : nullptr, tags ? tags->size() : 0u)
            ? MatterTransportOwner::Obstacle : substanceTransportOwner(tag, model, params);
        ++summary.particles[static_cast<std::size_t>(owner)];
        if (model == MatterConstitutiveModel::Elastic) {
            summary.ready = false;
            summary.reason = "mixed elastic MPM transport is not implemented";
        }
    }
    if (params.grain.enabled &&
        summary.particles[static_cast<std::size_t>(MatterTransportOwner::Obstacle)] > 0) {
        summary.ready = false;
        summary.reason = "grain transport does not accept pinned or static block carriers";
    }
    return summary;
}

MatterOwnerSummary inspectMatterDomainOwners(const FluidParticles& particles,
                                            const SimulationGridDomainDesc& domain) {
    std::vector<uint32_t> tags;
    collectMatterStaticTags(domain, tags);
    auto params = domain.fluid_params;
    params.solid_substance_tags = tags.empty() ? nullptr : &tags;
    return inspectMatterOwners(particles, params);
}

bool validateMatterTransportEdit(const std::string& substance,
                                const std::vector<SimulationGridDomainDesc>& domains,
                                const std::vector<SimulationGridDomainState>& states,
                                std::string& error, const char* key) {
    const auto affected = [&](const SubstanceProfile* profile) {
        std::unordered_set<std::string> visited;
        while (profile && visited.insert(profile->name).second) {
            if (profile->name == substance) {
                return true;
            }
            profile = profile->based_on.empty() ? nullptr : tryFindSubstance(profile->based_on);
        }
        return false;
    };
    for (std::size_t domain = 0; domain < domains.size() && domain < states.size(); ++domain) {
        const auto& particles = states[domain].particles;
        std::unordered_set<uint32_t> examined;
        for (std::size_t index = 0; index < particles.size(); ++index) {
            const uint32_t tag = index < particles.substance_tag.size()
                ? particles.substance_tag[index] : kSubstanceUntagged;
            if (!examined.insert(tag).second) {
                continue;
            }
            if (affected(particleSubstance(tag, domains[domain].fluid_params))) {
                error = "reset domain '" + domains[domain].name +
                    "' before editing " + key + " of '" + substance +
                    "' (carrier mass and contact ownership are fixed at birth)";
                return false;
            }
        }
    }
    error.clear();
    return true;
}

const SubstanceProfile* particleSubstance(uint32_t tag, const APICSolverParams& params) {
    if (tag != kSubstanceUntagged) {
        if (const auto* profile = tryFindSubstanceByTag(tag)) {
            return profile;
        }
    }
    return domainSubstance(params);
}

MatterConstitutiveModel substanceBirthModel(uint32_t tag, const APICSolverParams& params,
                                          float kelvin) {
    const auto* profile = particleSubstance(tag, params);
    if (profile && profile->default_constitutive_model != MatterConstitutiveModel::Auto) {
        // A meltable mechanical skeleton becomes liquid at its own threshold.
        // The temperature is state; the cooling switch does not choose a model.
        if (profile->meltable && std::isfinite(kelvin) && kelvin >= profile->melt_kelvin &&
            (profile->default_constitutive_model == MatterConstitutiveModel::Granular ||
             profile->default_constitutive_model == MatterConstitutiveModel::Elastic)) {
            return MatterConstitutiveModel::Fluid;
        }
        return profile->default_constitutive_model;
    }
    return params.granular_enabled ? MatterConstitutiveModel::Granular
                                   : MatterConstitutiveModel::Fluid;
}

void initializeMatterBirthModels(FluidParticles& particles, std::size_t begin,
                                const APICSolverParams& params) {
    particles.constitutive_model.resize(particles.size(),
        static_cast<uint8_t>(MatterConstitutiveModel::Auto));
    for (std::size_t index = begin; index < particles.size(); ++index) {
        const uint32_t tag = index < particles.substance_tag.size()
            ? particles.substance_tag[index] : kSubstanceUntagged;
        const float kelvin = index < particles.temperature.size()
            ? particles.temperature[index] : std::numeric_limits<float>::quiet_NaN();
        const auto model = static_cast<uint8_t>(substanceBirthModel(tag, params, kelvin));
        if (particles.constitutive_model[index] == model) {
            continue;
        }
        particles.constitutive_model[index] = model;
        // A newly solidified carrier starts from its current shape. Old stress
        // from a preceding solid interval must not reappear after melting.
        if (index < particles.granular_deformation_col0.size()) {
            particles.granular_deformation_col0[index] = Vec3(1.0f, 0.0f, 0.0f);
        }
        if (index < particles.granular_deformation_col1.size()) {
            particles.granular_deformation_col1[index] = Vec3(0.0f, 1.0f, 0.0f);
        }
        if (index < particles.granular_deformation_col2.size()) {
            particles.granular_deformation_col2[index] = Vec3(0.0f, 0.0f, 1.0f);
        }
        if (index < particles.granular_plastic_volume.size()) {
            particles.granular_plastic_volume[index] = 1.0f;
        }
        if (index < particles.granular_stress_diag.size()) {
            particles.granular_stress_diag[index] = Vec3(0.0f);
        }
        if (index < particles.granular_stress_shear.size()) {
            particles.granular_stress_shear[index] = Vec3(0.0f);
        }
    }
}

void refreshMatterConstitutiveModels(FluidParticles& particles,
                                    const APICSolverParams& params) {
    initializeMatterBirthModels(particles, 0u, params);
}

bool isMatterObstacle(const FluidParticles& particles, std::size_t index,
                      const uint32_t* static_tags, std::size_t tag_count,
                      bool include_frozen) {
    if (index >= particles.size() ||
        (index < particles.mass_fraction.size() &&
         !(particles.mass_fraction[index] > 0.02f))) {
        return false;
    }
    if (include_frozen && index < particles.flags.size() &&
        (particles.flags[index] & kParticleFlagFrozen) != 0u) {
        return true;
    }
    if (!static_tags || index >= particles.substance_tag.size()) {
        return false;
    }
    const uint32_t tag = particles.substance_tag[index];
    if (tag == kSubstanceUntagged) {
        return false;
    }
    return std::find(static_tags, static_tags + tag_count, tag) != static_tags + tag_count;
}

bool buildMatterObstacleMask(const FluidParticles& particles,
                            const std::vector<uint32_t>* static_tags,
                            std::vector<uint8_t>& mask, bool& any_frozen) {
    any_frozen = false;
    const uint32_t* tags = static_tags && !static_tags->empty() ? static_tags->data() : nullptr;
    const std::size_t tag_count = static_tags ? static_tags->size() : 0;
    bool any = false;
    mask.clear();
    for (std::size_t index = 0; index < particles.size(); ++index) {
        if (!isMatterObstacle(particles, index, tags, tag_count)) {
            continue;
        }
        if (!any) {
            mask.assign(particles.size(), 0u);
            any = true;
        }
        mask[index] = 1u;
        any_frozen |= index < particles.flags.size() &&
            (particles.flags[index] & kParticleFlagFrozen) != 0u;
    }
    return any;
}

float substanceFreezeKelvin(uint32_t tag, const APICSolverParams& params) {
    const auto* profile = particleSubstance(tag, params);
    if (!profile) {
        return params.thermal_freeze_kelvin;
    }
    return profile->meltable ? profile->melt_kelvin
                             : -std::numeric_limits<float>::infinity();
}

void collectMatterStaticTags(const SimulationGridDomainDesc& domain,
                            std::vector<uint32_t>& tags) {
    tags.clear();
    if (!domain.fluid_solid_phase_enabled) {
        return;
    }
    for (const auto& binding : domain.fluid_substance_materials) {
        if (binding.substance.empty() || binding.phase != SubstancePhase::Solid) {
            continue;
        }
        const auto tag = substanceTag(binding.substance);
        if (tag != kSubstanceUntagged &&
            std::find(tags.begin(), tags.end(), tag) == tags.end()) {
            tags.push_back(tag);
        }
        if (tags.size() >= kMaxFluidSubstanceMaterials) {
            break;
        }
    }
}

float substanceThermalViscosity(uint32_t tag, float kelvin,
                               const APICSolverParams& params) {
    const auto* profile = particleSubstance(tag, params);
    const float hot = std::max(0.0f, profile ? profile->liquid_kinematic_viscosity
                                          : params.kinematic_viscosity);
    if (profile && !profile->meltable) {
        return hot;
    }
    const float freeze = substanceFreezeKelvin(tag, params);
    const float range = std::max(1.0f, profile ? profile->liquid_freeze_viscosity_range
                                             : params.thermal_viscosity_range);
    const float cold = std::max(1.0e-7f, profile ? profile->liquid_cold_viscosity
                                              : params.thermal_cold_viscosity);
    const float fraction = std::clamp((kelvin - freeze) / range, 0.0f, 1.0f);
    if (fraction >= 1.0f) {
        return hot;
    }
    return std::exp(fraction * std::log(std::max(1.0e-7f, hot)) +
                    (1.0f - fraction) * std::log(cold));
}

float substanceMeltReleaseKelvin(uint32_t tag, const APICSolverParams& params) {
    const auto* profile = particleSubstance(tag, params);
    const float range = profile ? profile->liquid_freeze_viscosity_range
                                : params.thermal_viscosity_range;
    return substanceFreezeKelvin(tag, params) + std::max(1.0f, 0.1f * range);
}

} // namespace RayTrophiSim::Fluid
