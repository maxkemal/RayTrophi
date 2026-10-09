#include "Fluid/MatterGrain.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/SubstanceTag.h"
#include "ParticleSimulation.h"
#include "MaterialStateField.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace RayTrophiSim::Fluid {

namespace {

// Keys that left the domain: where each one lives now. A script that still
// sends one gets the new home, not "unknown setting".
const std::pair<const char*, const char*> kMovedGrainKeys[] = {
    {"enabled", "derived: grains run when a substance in the domain is granular with "
        "granular_transport=dem"},
    {"stiffness_n_m", "stiffness_scale (k = scale x 8e5 N/m per m x radius_m)"},
    {"restitution", "substance grain_restitution"},
    {"friction", "substance grain_friction"},
    {"rolling_friction", "substance grain_rolling_friction"},
    {"twisting_friction", "substance grain_twisting_friction"},
    {"tangential_stiffness_ratio", "substance grain_tangential_stiffness_ratio"},
    {"packing_fraction", "substance grain_packing_fraction"},
    {"represented_grain_radius_m", "substance grain_real_radius_m"},
    {"water_capacity_fraction", "substance grain_water_capacity_fraction"},
    {"absorption_rate_per_s", "substance grain_absorption_rate_per_s"},
    {"drying_rate_per_s", "substance grain_drying_rate_per_s"},
    {"contact_angle_deg", "substance grain_contact_angle_deg"},
    {"surface_tension_n_m", "the liquid substance's liquid_surface_tension_n_m"},
    {"wet_grains", "derived: on when the domain holds liquid (or a source pours wet grains) "
        "and the grain substance has grain_water_capacity_fraction > 0"},
    {"birth_saturation", "flow source grain_birth_saturation"},
};

void deriveStiffness(MatterGrainParams& p) {
    p.stiffness_n_m = p.stiffness_scale * kMatterGrainStiffnessPerRadius * p.radius_m;
}

} // namespace

nlohmann::json matterGrainParamsToJson(const MatterGrainParams& p) {
    // Only what the domain authors (MADDE_UI_TEK_OTORITE U2/U3): the grain
    // material is the substance's, `enabled` and `wet_grains` are derived.
    return {{"radius_m", p.radius_m},
        {"stiffness_scale", p.stiffness_scale},
        {"sliding_damping_n_s_m", p.sliding_damping_n_s_m},
        {"contact_resolution", p.contact_resolution},
        {"fluid_coupling", p.fluid_coupling},
        {"volume_exclusion", p.volume_exclusion},
        {"max_substeps", p.max_substeps},
        {"sleep", p.sleep}, {"sleep_speed_m_s", p.sleep_speed_m_s},
        {"sleep_time_s", p.sleep_time_s}};
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
            error = "normal_damping_n_s_m was replaced by the substance's grain_restitution";
            return false;
        }
        if (it.key() == "drag_viscosity_pa_s") {
            error = "drag_viscosity_pa_s was removed: drag uses the liquid substance viscosity";
            return false;
        }
        if (it.key() == "solver_kind" || it.key() == "xpbd_substeps") {
            error = it.key() + " was removed: DEM is the only grain solver "
                "(XPBD failed the H1-G0 accuracy/cost gate)";
            return false;
        }
        for (const auto& [key, home] : kMovedGrainKeys) {
            if (it.key() == key) {
                error = it.key() + " moved: " + home;
                return false;
            }
        }
        if (!fields.contains(it.key())) {
            error = "unknown grain setting: " + it.key();
            return false;
        }
        const bool integer = it.key() == "max_substeps" || it.key() == "contact_resolution";
        const bool boolean = it.key() == "fluid_coupling" ||
            it.key() == "volume_exclusion" || it.key() == "sleep";
        if ((boolean && !it.value().is_boolean()) ||
            (integer && !it.value().is_number_integer()) ||
            (!boolean && !integer &&
             (!it.value().is_number() || !std::isfinite(it.value().get<double>())))) {
            error = "invalid grain setting type: " + it.key();
            return false;
        }
    }
    try {
        candidate.radius_m = patch.value("radius_m", p.radius_m);
        candidate.stiffness_scale = patch.value("stiffness_scale", p.stiffness_scale);
        candidate.sliding_damping_n_s_m = patch.value("sliding_damping_n_s_m", p.sliding_damping_n_s_m);
        candidate.fluid_coupling = patch.value("fluid_coupling", p.fluid_coupling);
        candidate.volume_exclusion = patch.value("volume_exclusion", p.volume_exclusion);
        candidate.sleep = patch.value("sleep", p.sleep);
        candidate.sleep_speed_m_s = patch.value("sleep_speed_m_s", p.sleep_speed_m_s);
        candidate.sleep_time_s = patch.value("sleep_time_s", p.sleep_time_s);
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
        deriveStiffness(candidate);
        if (candidate.radius_m < .001f || candidate.radius_m > 1.0f ||
            !(candidate.stiffness_scale >= .01f && candidate.stiffness_scale <= 1000.0f) ||
            candidate.stiffness_n_m < 1.0f || candidate.stiffness_n_m > 1e8f ||
            candidate.sliding_damping_n_s_m < 0.0f ||
            candidate.sliding_damping_n_s_m > 1e5f ||
            candidate.sleep_speed_m_s < 0.0f || candidate.sleep_speed_m_s > 1.0f ||
            candidate.sleep_time_s < .01f || candidate.sleep_time_s > 10.0f) {
            throw std::runtime_error("grain settings outside physical/runtime limits "
                "(radius .001..1 m, stiffness_scale .01..1000)");
        }
        // The real radius is the substance's, but it cannot exceed this
        // domain's simulated radius.
        if (candidate.represented_grain_radius_m > candidate.radius_m) {
            candidate.represented_grain_radius_m = candidate.radius_m;
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    p = candidate;
    error.clear();
    return true;
}

void applyMatterGrainSubstance(MatterGrainParams& p, const SubstanceProfile& grain,
                               const SubstanceProfile* liquid, bool wet) {
    p.friction = std::clamp(grain.grain_friction, 0.0f, 2.0f);
    p.rolling_friction = std::clamp(grain.grain_rolling_friction, 0.0f, 1.0f);
    p.twisting_friction = std::clamp(grain.grain_twisting_friction, 0.0f, 1.0f);
    p.restitution = std::clamp(grain.grain_restitution, .01f, 1.0f);
    p.tangential_stiffness_ratio = std::clamp(grain.grain_tangential_stiffness_ratio, 0.0f, 1.0f);
    p.packing_fraction = std::clamp(grain.grain_packing_fraction, .3f, .74f);
    const float real = std::max(grain.grain_real_radius_m, 0.0f);
    p.represented_grain_radius_m = real >= p.radius_m ? 0.0f : real;
    p.water_capacity_fraction = std::clamp(grain.grain_water_capacity_fraction, 0.0f, .5f);
    p.absorption_rate_per_s = std::clamp(grain.grain_absorption_rate_per_s, 0.0f, 1000.0f);
    p.drying_rate_per_s = std::clamp(grain.grain_drying_rate_per_s, 0.0f, 100.0f);
    p.contact_angle_deg = std::clamp(grain.grain_contact_angle_deg, 0.0f, 89.0f);
    p.surface_tension_n_m = std::clamp(liquid ? liquid->liquid_surface_tension_n_m
        : grain.liquid_surface_tension_n_m, 0.0f, 1.0f);
    p.wet_grains = wet;
    deriveStiffness(p);
}

float matterGrainDampingRatio(float restitution) {
    const double e = std::clamp(double(restitution), .01, 1.0);
    const double log_e = std::log(e);
    return static_cast<float>(-log_e / std::sqrt(9.869604401089358 + log_e * log_e));
}

MatterGrainParams matterGrainParamsFromJson(const nlohmann::json& stored, std::string* dropped_keys) {
    // Saved scenes from before the substance owned the grain material
    // (2026-10-09): the material keys are dropped and the substance's values
    // apply; `dropped_keys` names them so the loader can say so. The stiffness
    // becomes a scale.
    auto json = stored;
    if (json.is_object()) {
        if (json.contains("stiffness_n_m") && !json.contains("stiffness_scale")) {
            const double radius = std::clamp(json.value("radius_m", 0.025), .001, 1.0);
            json["stiffness_scale"] = std::clamp(json["stiffness_n_m"].get<double>() /
                (double(kMatterGrainStiffnessPerRadius) * radius), .01, 1000.0);
        }
        std::string dropped;
        for (const auto& [key, home] : kMovedGrainKeys) {
            if (json.contains(key)) {
                if (std::string(key) != "enabled" && std::string(key) != "stiffness_n_m") {
                    dropped += (dropped.empty() ? "" : ", ") + std::string(key);
                }
                json.erase(key);
            }
        }
        if (dropped_keys) {
            *dropped_keys = dropped;
        }
        for (const char* removed : {"normal_damping_n_s_m", "drag_viscosity_pa_s",
                                    "solver_kind", "xpbd_substeps"}) {
            json.erase(removed);
        }
    }
    MatterGrainParams p;
    std::string error;
    if (!patchMatterGrainParams(json, p, error)) {
        throw std::runtime_error(error);
    }
    return p;
}

float matterGrainRestMassKg(const MatterGrainParams& params, uint32_t substance_tag,
                            const SubstanceProfile* domain_substance,
                            MatterConstitutiveModel model, bool legacy_granular) {
    // fluidParticleRestMassKg(h, ppc=1) is bulk density * h^3, so h is the
    // cube root of the bulk volume one grain stands for.
    const float sphere = 4.18879020f * params.radius_m * params.radius_m * params.radius_m;
    const float bulk_volume = sphere / std::clamp(params.packing_fraction, .3f, .74f);
    return fluidParticleRestMassKg(substance_tag, domain_substance, std::cbrt(bulk_volume), 1,
        model, legacy_granular);
}

float matterGrainWaterCapacityKg(const MatterGrainParams& params) {
    return params.water_capacity_fraction * 4.18879020f * params.radius_m * params.radius_m *
        params.radius_m * 1000.0f;
}

void initMatterGrainBirthWater(FluidParticles& particles, std::size_t index,
                               float birth_saturation, const MatterGrainParams& params) {
    if (!params.wet_grains || index >= particles.size()) {
        return;
    }
    const float capacity = matterGrainWaterCapacityKg(params);
    const float water = capacity * std::clamp(birth_saturation, 0.0f, 1.0f);
    particles.pore_capacity_kg[index] = capacity;
    particles.pore_water_mass_kg[index] = water;
    particles.pore_water_energy_j[index] = water * 4186.0f *
        std::max(1.0f, particles.temperature[index]);
}

std::size_t ensureMatterGrainRestMasses(FluidParticles& particles,
                                        const MatterGrainParams& params,
                                        const SubstanceProfile* domain_substance,
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
        const auto* profile = resolveFluidSubstanceProfile(tag, domain_substance);
        if (!profile || profile->granular_transport != MatterGranularTransport::Dem) {
            continue;
        }
        rest_mass = matterGrainRestMassKg(params, tag, domain_substance, model, legacy_granular);
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
    // The switch is on by default and stamps only substances bound as Solid
    // (FluidDomainStep reads it the same way): with none authored it does
    // nothing, and blocking on it refused grains in every new Matter domain.
    if (d.fluid_solid_phase_enabled &&
        std::any_of(d.fluid_substance_materials.begin(), d.fluid_substance_materials.end(),
            [](const auto& b) { return !b.substance.empty() && b.phase == SubstancePhase::Solid; })) {
        add("solid_phase", "A substance is bound as Solid in this domain; solid phase cannot run "
            "with grains yet (Matter tab: turn off Solid Phase Blocks Flow, or unbind it).");
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
