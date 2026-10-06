#include "Fluid/MatterGrain.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/SubstanceTag.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace RayTrophiSim::Fluid {

nlohmann::json matterGrainParamsToJson(const MatterGrainParams& p) {
    return {{"enabled", p.enabled}, {"radius_m", p.radius_m},
        {"stiffness_n_m", p.stiffness_n_m},
        {"normal_damping_n_s_m", p.normal_damping_n_s_m},
        {"sliding_damping_n_s_m", p.sliding_damping_n_s_m},
        {"friction", p.friction}, {"rolling_friction", p.rolling_friction},
        {"twisting_friction", p.twisting_friction},
        {"tangential_stiffness_ratio", p.tangential_stiffness_ratio},
        {"contact_resolution", p.contact_resolution},
        {"packing_fraction", p.packing_fraction},
        {"max_substeps", p.max_substeps}};
}

bool patchMatterGrainParams(const nlohmann::json& patch, MatterGrainParams& p,
                           std::string& error) {
    auto candidate = p;
    const auto fields = matterGrainParamsToJson(p);
    if (!patch.is_object()) {
        error = "grain settings must be an object";
        return false;
    }
    for (auto it = patch.begin(); it != patch.end(); ++it) {
        if (!fields.contains(it.key())) {
            error = "unknown grain setting: " + it.key();
            return false;
        }
        const bool integer = it.key() == "max_substeps" || it.key() == "contact_resolution";
        if ((it.key() == "enabled" && !it.value().is_boolean()) ||
            (integer && !it.value().is_number_integer()) ||
            (it.key() != "enabled" && !integer &&
             (!it.value().is_number() || !std::isfinite(it.value().get<double>())))) {
            error = "invalid grain setting type: " + it.key();
            return false;
        }
    }
    try {
        candidate.enabled = patch.value("enabled", p.enabled);
        candidate.radius_m = patch.value("radius_m", p.radius_m);
        candidate.stiffness_n_m = patch.value("stiffness_n_m", p.stiffness_n_m);
        candidate.normal_damping_n_s_m = patch.value("normal_damping_n_s_m", p.normal_damping_n_s_m);
        candidate.sliding_damping_n_s_m = patch.value("sliding_damping_n_s_m", p.sliding_damping_n_s_m);
        candidate.friction = patch.value("friction", p.friction);
        candidate.rolling_friction = patch.value("rolling_friction", p.rolling_friction);
        candidate.twisting_friction = patch.value("twisting_friction", p.twisting_friction);
        candidate.tangential_stiffness_ratio =
            patch.value("tangential_stiffness_ratio", p.tangential_stiffness_ratio);
        candidate.packing_fraction = patch.value("packing_fraction", p.packing_fraction);
        const auto steps = patch.value("max_substeps", double(p.max_substeps));
        if (steps < 1 || steps > 4096) {
            throw std::runtime_error("max_substeps must be 1..4096");
        }
        candidate.max_substeps = static_cast<int>(steps);
        const auto resolution = patch.value("contact_resolution", double(p.contact_resolution));
        if (resolution < 8 || resolution > 200) {
            throw std::runtime_error("contact_resolution must be 8..200 substeps per collision");
        }
        candidate.contact_resolution = static_cast<int>(resolution);
        if (candidate.radius_m < .001f || candidate.radius_m > 1.0f ||
            candidate.stiffness_n_m < 1.0f || candidate.stiffness_n_m > 1e8f ||
            candidate.normal_damping_n_s_m < 0.0f ||
            candidate.normal_damping_n_s_m > 1e5f ||
            candidate.sliding_damping_n_s_m < 0.0f ||
            candidate.sliding_damping_n_s_m > 1e5f ||
            candidate.friction < 0.0f || candidate.friction > 2.0f ||
            candidate.rolling_friction < 0.0f || candidate.rolling_friction > 1.0f ||
            candidate.twisting_friction < 0.0f || candidate.twisting_friction > 1.0f ||
            candidate.tangential_stiffness_ratio < 0.0f ||
            candidate.tangential_stiffness_ratio > 1.0f ||
            candidate.packing_fraction < .3f || candidate.packing_fraction > .74f) {
            throw std::runtime_error("grain settings outside physical/runtime limits");
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    p = candidate;
    error.clear();
    return true;
}

MatterGrainParams matterGrainParamsFromJson(const nlohmann::json& json) {
    MatterGrainParams p;
    std::string error;
    if (!patchMatterGrainParams(json, p, error)) {
        throw std::runtime_error(error);
    }
    return p;
}

float matterGrainRestMassKg(const MatterGrainParams& params, uint32_t substance_tag,
                            FluidChemistryPreset chemistry_preset,
                            MatterConstitutiveModel model, bool legacy_granular) {
    // fluidParticleRestMassKg(h, ppc=1) is bulk density * h^3, so h is the
    // cube root of the bulk volume one grain stands for.
    const float sphere = 4.18879020f * params.radius_m * params.radius_m * params.radius_m;
    const float bulk_volume = sphere / std::clamp(params.packing_fraction, .3f, .74f);
    return fluidParticleRestMassKg(substance_tag, chemistry_preset, std::cbrt(bulk_volume), 1,
        model, legacy_granular);
}

std::size_t ensureMatterGrainRestMasses(FluidParticles& particles,
                                        const MatterGrainParams& params,
                                        FluidChemistryPreset chemistry_preset,
                                        bool legacy_granular) {
    std::size_t initialized = 0;
    const auto count = particles.size();
    particles.rest_mass_kg.resize(count, 0.0f);
    for (std::size_t i = 0; i < count; ++i) {
        float& rest_mass = particles.rest_mass_kg[i];
        if (std::isfinite(rest_mass) && rest_mass > 0.0f) {
            continue;
        }
        const auto model = i < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[i])
            : MatterConstitutiveModel::Auto;
        if (model != MatterConstitutiveModel::Granular) {
            continue;
        }
        const uint32_t tag = i < particles.substance_tag.size()
            ? particles.substance_tag[i] : kSubstanceUntagged;
        rest_mass = matterGrainRestMassKg(params, tag, chemistry_preset, model, legacy_granular);
        ++initialized;
    }
    return initialized;
}

bool validateMatterGrainDomain(const SimulationGridDomainDesc& d,
                              const MatterGrainParams& p, std::string& error) {
    if (p.enabled && (d.type != SimulationDomainType::Matter ||
        d.backend != SimulationDomainBackend::GPU_Vulkan ||
        d.boundary_mode != SimulationGridDomainBoundaryMode::Closed ||
        d.fluid_params.pore_exchange.enabled ||
        d.fluid_params.pore_exchange.wet_response_enabled ||
        d.fluid_params.thermal_liquid_enabled || d.fluid_solid_phase_enabled)) {
        error = "dry grain requires Closed Vulkan Matter; disable pore/wet/thermal/solid";
        return false;
    }
    error.clear();
    return true;
}

} // namespace RayTrophiSim::Fluid
