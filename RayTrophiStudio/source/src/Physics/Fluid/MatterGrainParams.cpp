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
        {"restitution", p.restitution},
        {"sliding_damping_n_s_m", p.sliding_damping_n_s_m},
        {"friction", p.friction}, {"rolling_friction", p.rolling_friction},
        {"twisting_friction", p.twisting_friction},
        {"tangential_stiffness_ratio", p.tangential_stiffness_ratio},
        {"contact_resolution", p.contact_resolution},
        {"packing_fraction", p.packing_fraction},
        {"fluid_coupling", p.fluid_coupling},
        {"volume_exclusion", p.volume_exclusion},
        {"wet_grains", p.wet_grains},
        {"water_capacity_fraction", p.water_capacity_fraction},
        {"absorption_rate_per_s", p.absorption_rate_per_s},
        {"drying_rate_per_s", p.drying_rate_per_s},
        {"surface_tension_n_m", p.surface_tension_n_m},
        {"contact_angle_deg", p.contact_angle_deg},
        {"represented_grain_radius_m", p.represented_grain_radius_m},
        {"birth_saturation", p.birth_saturation},
        {"solver_kind", p.solver_kind},
        {"xpbd_substeps", p.xpbd_substeps},
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
        if (it.key() == "normal_damping_n_s_m") {
            error = "normal_damping_n_s_m was replaced by restitution (0.01..1)";
            return false;
        }
        if (it.key() == "drag_viscosity_pa_s") {
            error = "drag_viscosity_pa_s was removed: drag uses the liquid substance viscosity";
            return false;
        }
        if (!fields.contains(it.key())) {
            error = "unknown grain setting: " + it.key();
            return false;
        }
        const bool integer = it.key() == "max_substeps" || it.key() == "contact_resolution" ||
            it.key() == "xpbd_substeps";
        if (it.key() == "solver_kind") {
            if (!it.value().is_string() ||
                (it.value() != "dem" && it.value() != "xpbd")) {
                error = "grain solver_kind must be \"dem\" or \"xpbd\"";
                return false;
            }
            continue;
        }
        const bool boolean = it.key() == "enabled" || it.key() == "fluid_coupling" ||
            it.key() == "volume_exclusion" || it.key() == "wet_grains";
        if ((boolean && !it.value().is_boolean()) ||
            (integer && !it.value().is_number_integer()) ||
            (!boolean && !integer &&
             (!it.value().is_number() || !std::isfinite(it.value().get<double>())))) {
            error = "invalid grain setting type: " + it.key();
            return false;
        }
    }
    try {
        candidate.enabled = patch.value("enabled", p.enabled);
        candidate.radius_m = patch.value("radius_m", p.radius_m);
        candidate.stiffness_n_m = patch.value("stiffness_n_m", p.stiffness_n_m);
        candidate.restitution = patch.value("restitution", p.restitution);
        candidate.sliding_damping_n_s_m = patch.value("sliding_damping_n_s_m", p.sliding_damping_n_s_m);
        candidate.friction = patch.value("friction", p.friction);
        candidate.rolling_friction = patch.value("rolling_friction", p.rolling_friction);
        candidate.twisting_friction = patch.value("twisting_friction", p.twisting_friction);
        candidate.tangential_stiffness_ratio =
            patch.value("tangential_stiffness_ratio", p.tangential_stiffness_ratio);
        candidate.packing_fraction = patch.value("packing_fraction", p.packing_fraction);
        candidate.fluid_coupling = patch.value("fluid_coupling", p.fluid_coupling);
        candidate.volume_exclusion = patch.value("volume_exclusion", p.volume_exclusion);
        candidate.wet_grains = patch.value("wet_grains", p.wet_grains);
        candidate.water_capacity_fraction =
            patch.value("water_capacity_fraction", p.water_capacity_fraction);
        candidate.absorption_rate_per_s = patch.value("absorption_rate_per_s", p.absorption_rate_per_s);
        candidate.drying_rate_per_s = patch.value("drying_rate_per_s", p.drying_rate_per_s);
        candidate.surface_tension_n_m = patch.value("surface_tension_n_m", p.surface_tension_n_m);
        candidate.contact_angle_deg = patch.value("contact_angle_deg", p.contact_angle_deg);
        candidate.represented_grain_radius_m =
            patch.value("represented_grain_radius_m", p.represented_grain_radius_m);
        candidate.birth_saturation = patch.value("birth_saturation", p.birth_saturation);
        candidate.solver_kind = patch.value("solver_kind", p.solver_kind);
        const auto xpbd = patch.value("xpbd_substeps", double(p.xpbd_substeps));
        if (xpbd < 4 || xpbd > 512) {
            throw std::runtime_error("xpbd_substeps must be 4..512");
        }
        candidate.xpbd_substeps = static_cast<int>(xpbd);
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
            !(candidate.restitution >= .01f && candidate.restitution <= 1.0f) ||
            candidate.sliding_damping_n_s_m < 0.0f ||
            candidate.sliding_damping_n_s_m > 1e5f ||
            candidate.friction < 0.0f || candidate.friction > 2.0f ||
            candidate.rolling_friction < 0.0f || candidate.rolling_friction > 1.0f ||
            candidate.twisting_friction < 0.0f || candidate.twisting_friction > 1.0f ||
            candidate.tangential_stiffness_ratio < 0.0f ||
            candidate.tangential_stiffness_ratio > 1.0f ||
            candidate.packing_fraction < .3f || candidate.packing_fraction > .74f ||
            candidate.water_capacity_fraction < 0.0f || candidate.water_capacity_fraction > .5f ||
            candidate.absorption_rate_per_s < 0.0f || candidate.absorption_rate_per_s > 1000.0f ||
            candidate.drying_rate_per_s < 0.0f || candidate.drying_rate_per_s > 100.0f ||
            candidate.surface_tension_n_m < 0.0f || candidate.surface_tension_n_m > 1.0f ||
            candidate.contact_angle_deg < 0.0f || candidate.contact_angle_deg > 89.0f ||
            candidate.birth_saturation < 0.0f || candidate.birth_saturation > 1.0f ||
            (candidate.represented_grain_radius_m != 0.0f &&
             (candidate.represented_grain_radius_m < 1e-6f ||
              candidate.represented_grain_radius_m > candidate.radius_m))) {
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

float matterGrainDampingRatio(float restitution) {
    const double e = std::clamp(double(restitution), .01, 1.0);
    const double log_e = std::log(e);
    return static_cast<float>(-log_e / std::sqrt(9.869604401089358 + log_e * log_e));
}

MatterGrainParams matterGrainParamsFromJson(const nlohmann::json& stored) {
    // Saved scenes from before restitution: the per-domain normal damping c
    // (N s/m) is converted at the grain-grain effective mass m/2 of a
    // quartz-like grain (2667 kg/m^3: sand bulk 1600 / packing .6), the
    // reference every grain test used: e = exp(-pi zeta / sqrt(1 - zeta^2)),
    // zeta = c / (2 sqrt(k m/2)). The stored drag viscosity is dropped; drag
    // reads the liquid substance now.
    auto json = stored;
    if (json.is_object() && json.contains("normal_damping_n_s_m")) {
        const double c = json["normal_damping_n_s_m"].get<double>();
        if (!json.contains("restitution")) {
            const double radius = json.value("radius_m", 0.025);
            const double k = json.value("stiffness_n_m", 20000.0);
            const double mass = 2667.0 * 4.18879020478639 * radius * radius * radius;
            const double zeta = c / (2.0 * std::sqrt(std::max(k * .5 * mass, 1e-12)));
            json["restitution"] = zeta >= 1.0 ? .01
                : std::clamp(std::exp(-3.141592653589793 * zeta / std::sqrt(1.0 - zeta * zeta)),
                    .01, 1.0);
        }
        json.erase("normal_damping_n_s_m");
    }
    if (json.is_object()) {
        json.erase("drag_viscosity_pa_s");
    }
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

float matterGrainWaterCapacityKg(const MatterGrainParams& params) {
    return params.water_capacity_fraction * 4.18879020f * params.radius_m * params.radius_m *
        params.radius_m * 1000.0f;
}

void initMatterGrainBirthWater(FluidParticles& particles, std::size_t index,
                               const MatterGrainParams& params) {
    if (!params.wet_grains || index >= particles.size()) {
        return;
    }
    const float capacity = matterGrainWaterCapacityKg(params);
    const float water = capacity * params.birth_saturation;
    particles.pore_capacity_kg[index] = capacity;
    particles.pore_water_mass_kg[index] = water;
    particles.pore_water_energy_j[index] = water * 4186.0f *
        std::max(1.0f, particles.temperature[index]);
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

std::vector<MatterGrainBlocker> matterGrainBlockers(const SimulationGridDomainDesc& d) {
    std::vector<MatterGrainBlocker> blockers;
    const auto add = [&](const char* code, const char* message) {
        blockers.push_back({code, message});
    };
    if (d.type != SimulationDomainType::Matter) {
        add("not_matter", "Grains run in a Matter domain only.");
    }
    if (d.backend != SimulationDomainBackend::GPU_Vulkan) {
        add("backend", "Grains need the Vulkan backend.");
    }
    if (d.boundary_mode != SimulationGridDomainBoundaryMode::Closed) {
        add("boundary", "Grains need a Closed boundary.");
    }
    if (d.fluid_params.pore_exchange.enabled) {
        add("pore_exchange", "Pore water exchange cannot run with grains; use wet grains.");
    }
    if (d.fluid_params.pore_exchange.wet_response_enabled) {
        add("wet_response", "Pore wet response cannot run with grains; use wet grains.");
    }
    if (d.fluid_params.thermal_liquid_enabled) {
        add("thermal_liquid", "Thermal liquid cannot run with grains yet.");
    }
    if (d.fluid_solid_phase_enabled) {
        add("solid_phase", "Solid phase cannot run with grains yet.");
    }
    return blockers;
}

bool validateMatterGrainDomain(const SimulationGridDomainDesc& d,
                              const MatterGrainParams& p, std::string& error) {
    error.clear();
    if (!p.enabled) {
        return true;
    }
    for (const auto& blocker : matterGrainBlockers(d)) {
        error += (error.empty() ? "" : " ") + blocker.message;
    }
    return error.empty();
}

} // namespace RayTrophiSim::Fluid
