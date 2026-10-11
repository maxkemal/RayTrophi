#include "scene_ui_fluid_thermal.hpp"

#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"

#include "ui_modern.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>

namespace ForceFieldUI {
namespace FluidThermalUI {
namespace {

constexpr ImVec4 kWarningColor(1.0f, 0.72f, 0.25f, 1.0f);
constexpr float kTemperatureEpsilon = 0.05f;

void warningText(const char* format, ...) {
    va_list args;
    va_start(args, format);
    ImGui::PushStyleColor(ImGuiCol_Text, kWarningColor);
    ImGui::TextWrappedV(format, args);
    ImGui::PopStyleColor();
    va_end(args);
}

void drawTimeConstant(float rate) {
    ImGui::SameLine();
    if (rate > 0.0f) {
        ImGui::TextDisabled("tau %.2f s", 1.0f / rate);
    } else {
        ImGui::TextDisabled("off");
    }
}

float thresholdCrossingSeconds(float birth_kelvin,
                               float ambient_kelvin,
                               float freeze_kelvin,
                               float rate) {
    if (!(rate > 0.0f) || !(birth_kelvin > freeze_kelvin) ||
        !(freeze_kelvin > ambient_kelvin)) {
        return -1.0f;
    }
    const float hot_delta = birth_kelvin - ambient_kelvin;
    const float freeze_delta = freeze_kelvin - ambient_kelvin;
    if (!(hot_delta > freeze_delta) || !(freeze_delta > 0.0f)) {
        return -1.0f;
    }
    return std::log(hot_delta / freeze_delta) / rate;
}

void drawBirthTemperatureWarning(float birth_kelvin,
                                 float ambient_kelvin,
                                 const RayTrophiSim::Fluid::APICSolverParams& params,
                                 uint32_t substance_tag) {
    if (!params.thermal_liquid_enabled) {
        ImGui::TextDisabled("Cooling & Freezing is disabled on the domain.");
        return;
    }

    // The poured substance's own melt point, as the solver reads it per parcel.
    const float freeze_kelvin =
        RayTrophiSim::Fluid::substanceFreezeKelvin(substance_tag, params);
    if (!std::isfinite(freeze_kelvin)) {
        ImGui::TextDisabled("This substance cannot melt, so it never freezes.");
        return;
    }
    if (birth_kelvin <= freeze_kelvin + kTemperatureEpsilon) {
        warningText(
            "Born at/below the %.1f K freeze point: supported parcels can set immediately.",
            freeze_kelvin);
        return;
    }
    if (std::abs(birth_kelvin - ambient_kelvin) <= kTemperatureEpsilon) {
        warningText(
            "Born at ambient (%.1f K): there is no temperature difference to cool. "
            "Use Custom for a visible flow-then-freeze interval.",
            ambient_kelvin);
        return;
    }
    if (birth_kelvin < ambient_kelvin) {
        warningText(
            "Born below ambient: the domain will warm this liquid toward %.1f K, not cool it.",
            ambient_kelvin);
        return;
    }
    if (freeze_kelvin <= ambient_kelvin + kTemperatureEpsilon) {
        warningText(
            "Freeze point is at/below ambient. Newton cooling approaches %.1f K without "
            "crossing %.1f K, so this pour will not freeze by cooling alone.",
            ambient_kelvin,
            freeze_kelvin);
        return;
    }

    const float contact_seconds = thresholdCrossingSeconds(
        birth_kelvin,
        ambient_kelvin,
        freeze_kelvin,
        params.thermal_contact_cooling_rate);
    const float air_seconds = thresholdCrossingSeconds(
        birth_kelvin,
        ambient_kelvin,
        freeze_kelvin,
        params.thermal_air_cooling_rate);
    if (contact_seconds >= 0.0f && air_seconds >= 0.0f) {
        ImGui::TextDisabled(
            "Estimated threshold crossing: contact %.2f s | air %.2f s",
            contact_seconds,
            air_seconds);
    } else if (contact_seconds >= 0.0f) {
        ImGui::TextDisabled(
            "Estimated threshold crossing at contact: %.2f s",
            contact_seconds);
    } else if (air_seconds >= 0.0f) {
        ImGui::TextDisabled(
            "Estimated threshold crossing in air: %.2f s",
            air_seconds);
    }
}

} // namespace

void drawBoundaryOverride(RayTrophiSim::SimulationGridDomainDesc& domain,
                          float world_ambient_kelvin,
                          float world_oxygen) {
    if (!UIWidgets::CollapsingHeader("Thermal Environment")) {
        return;
    }

    ImGui::Spacing();
    ImGui::Checkbox(
        "Override World Thermal Environment##DomThermal",
        &domain.thermal_override_enabled);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Off: inherit the world ambient and oxygen.\n"
            "On: use this domain's values for Material State Field surfaces\n"
            "and for thermal-liquid cooling. Gas-grid temperature remains a\n"
            "separate normalized channel.");
    }

    ImGui::BeginDisabled(!domain.thermal_override_enabled);
    ImGui::DragFloat(
        "Ambient Temperature (K)##DomThermalK",
        &domain.thermal_ambient_kelvin,
        1.0f,
        0.0f,
        3000.0f,
        "%.1f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Temperature that object surfaces and thermal liquid approach inside\n"
            "this domain. An emitter using Domain Ambient is born at this value.");
    }
    ImGui::DragFloat(
        "Oxygen##DomThermalO2",
        &domain.thermal_oxygen,
        0.01f,
        0.0f,
        1.0f,
        "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "0..1. Throttles pyrolysis burn rate inside this domain.\n"
            "It does not change liquid cooling or freezing.");
    }
    ImGui::EndDisabled();

    const float effective_ambient = domain.thermal_override_enabled
        ? domain.thermal_ambient_kelvin
        : world_ambient_kelvin;
    const float effective_oxygen = domain.thermal_override_enabled
        ? domain.thermal_oxygen
        : world_oxygen;
    ImGui::TextDisabled(
        "Effective: ambient %.1f K | oxygen %.2f",
        effective_ambient,
        effective_oxygen);
    ImGui::TextDisabled(
        "Kelvin-per-unit remains global so absolute material temperatures stay consistent.");
}

bool drawDomainControls(
    RayTrophiSim::Fluid::APICSolverParams& params,
    const RayTrophiSim::Fluid::ThermalLiquidStats* stats,
    float ambient_kelvin,
    const std::vector<RayTrophiSim::SimulationFlowSourceDesc>& sources,
    int domain_index) {
    ImGui::SeparatorText("Thermal Liquid (cools, thickens, sets)");

    bool edited = false;
    const bool grain_locked = params.grain.enabled && !params.thermal_liquid_enabled;
    ImGui::BeginDisabled(grain_locked);
    edited |= ImGui::Checkbox(
        "Enable Cooling & Freezing",
        &params.thermal_liquid_enabled);
    ImGui::EndDisabled();
    if (grain_locked) {
        ImGui::TextDisabled("Thermal liquid cannot run with grains yet.");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "The domain cools all liquid parcels toward its effective ambient.\n"
            "A source only chooses the parcel's birth temperature. Supported\n"
            "parcels below the freeze point set and remain pinned.");
    }

    ImGui::TextDisabled("Effective Ambient: %.1f K", ambient_kelvin);
    ImGui::BeginDisabled(!params.thermal_liquid_enabled);
    // The freeze point, the thickening curve and the latent heat are each
    // parcel's substance (edit them in the Substance tree above). The Default
    // Substance is shown so the cooling rates can be read against it; a bound
    // or poured substance uses its own values.
    const auto* default_profile = RayTrophiSim::Fluid::particleSubstance(
        RayTrophiSim::Fluid::kSubstanceUntagged, params);
    if (default_profile && default_profile->meltable) {
        ImGui::TextDisabled(
            "%s: melts at %.1f K, latent %.0f kJ/kg | thickens over %.1f K to %.5f m^2/s",
            default_profile->name.c_str(), default_profile->melt_kelvin,
            default_profile->latent_heat_fusion / 1000.0f,
            default_profile->liquid_freeze_viscosity_range,
            default_profile->liquid_cold_viscosity);
    } else if (default_profile) {
        ImGui::TextDisabled("%s cannot melt, so it never freezes.",
                            default_profile->name.c_str());
    }

    edited |= ImGui::DragFloat(
        "Air Cooling (1/s)",
        &params.thermal_air_cooling_rate,
        0.01f,
        0.0f,
        100.0f,
        "%.3f");
    const bool air_hovered = ImGui::IsItemHovered();
    drawTimeConstant(params.thermal_air_cooling_rate);
    if (air_hovered) {
        ImGui::SetTooltip(
            "Newton cooling rate for free-surface parcels. tau = 1/rate is the\n"
            "time to close about 63%% of the temperature difference.");
    }

    edited |= ImGui::DragFloat(
        "Contact Cooling (1/s)",
        &params.thermal_contact_cooling_rate,
        0.05f,
        0.0f,
        100.0f,
        "%.3f");
    const bool contact_hovered = ImGui::IsItemHovered();
    drawTimeConstant(params.thermal_contact_cooling_rate);
    if (contact_hovered) {
        ImGui::SetTooltip(
            "Cooling rate for parcels touching a collider or closed wall.\n"
            "This normally controls how quickly the first supported layer sets.");
    }

    // Fog water held by this domain's air (GasSaturation.h). kg/m^3 is the
    // atmosphere layer's unit, so a cloud and this fog read the same number.
    ImGui::TextDisabled("Fog water in the air (kg/m^3)");
    edited |= ImGui::DragFloat("Rain Onset (kg/m^3)", &params.fog_autoconversion_kg_m3,
                               1.0e-5f, 0.0f, 1000.0f, "%.5f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Fog water above this coalesces into rain (Kessler autoconversion).\n"
            "Real fog holds 0.0001-0.001 kg/m^3.");
    }
    edited |= ImGui::DragFloat("Rain Formation (1/s)", &params.fog_autoconversion_rate,
                               1.0e-4f, 0.0f, 100.0f, "%.4f");
    const bool rate_hovered = ImGui::IsItemHovered();
    drawTimeConstant(params.fog_autoconversion_rate);
    if (rate_hovered) {
        ImGui::SetTooltip("How fast fog water above the onset turns into rain.");
    }
    edited |= ImGui::DragFloat("Fog Ceiling (kg/m^3)", &params.fog_ceiling_kg_m3,
                               1.0e-5f, 0.0f, 1000.0f, "%.5f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "The most fog water the air can hold; any more rains out at once.\n"
            "Dense cumulus holds up to ~0.005 kg/m^3.");
    }
    ImGui::EndDisabled();

    ImGui::TextDisabled("Heat conduction %.2f 1/s (substance parcel_conduction)",
                        params.granular_thermal_conductivity);

    if (edited) {
        params.sanitizeThermalLiquid();
        params.sanitizeGranularMaterial();
    }

    if (!params.thermal_liquid_enabled) {
        return edited;
    }

    ImGui::TextDisabled(
        "A parcel at its melt point holds there while its latent heat crosses "
        "(mushy); it counts as solid once half of it has frozen.");

    // Each source against ITS substance's melt point: one domain threshold
    // would describe only the Default Substance.
    int ambient_sources = 0;
    int custom_sources = 0;
    int ambient_below = 0;
    int ambient_no_crossing = 0;
    for (const auto& source : sources) {
        if (!source.enabled || source.domain_index != domain_index) {
            continue;
        }
        if (source.fluid_temperature_override) {
            ++custom_sources;
            continue;
        }
        ++ambient_sources;
        const float freeze = RayTrophiSim::Fluid::substanceFreezeKelvin(
            RayTrophiSim::Fluid::substanceTag(source.fluid_substance), params);
        if (!std::isfinite(freeze)) {
            continue;
        }
        if (freeze > ambient_kelvin + kTemperatureEpsilon) {
            ++ambient_below;
        } else {
            ++ambient_no_crossing;
        }
    }
    ImGui::TextDisabled(
        "Enabled sources: %d domain ambient | %d custom temperature",
        ambient_sources,
        custom_sources);
    if (ambient_below > 0) {
        warningText(
            "%d source(s) are born at %.1f K, below their substance's melt point; "
            "supported parcels start freezing immediately.",
            ambient_below, ambient_kelvin);
    }
    if (ambient_no_crossing > 0) {
        warningText(
            "%d source(s) are born exactly at ambient (%.1f K). They have no "
            "cooling interval and cannot reach their melt point by cooling.",
            ambient_no_crossing, ambient_kelvin);
    }

    if (stats == nullptr || !stats->measured) {
        ImGui::TextDisabled("Not measured yet (step the timeline).");
        return edited;
    }

    ImGui::TextDisabled(
        "Temperature: min %.1f / mean %.1f / max %.1f K | mean delta %+.1f K",
        stats->min_kelvin,
        stats->mean_kelvin,
        stats->max_kelvin,
        stats->mean_kelvin - ambient_kelvin);
    ImGui::TextDisabled(
        "Frozen: %zu (+%zu set, -%zu melted this frame)",
        stats->frozen_particles,
        stats->froze_this_frame,
        stats->melted_this_frame);
    if (stats->mushy_particles > 0 || stats->fusion_released_j > 0.0 ||
        stats->fusion_absorbed_j > 0.0) {
        ImGui::TextDisabled(
            "Changing phase: %zu parcels | latent heat %.3g J released, %.3g J absorbed",
            stats->mushy_particles, stats->fusion_released_j, stats->fusion_absorbed_j);
    }
    if (stats->gas_exchange_particles > 0 || stats->evaporated_kg > 0.0 ||
        stats->condensed_kg > 0.0) {
        ImGui::TextDisabled(
            "Surface with air: %zu parcels | heat %.3g J | evaporated %.3g kg | "
            "condensed %.3g kg (%zu splits)",
            stats->gas_exchange_particles, stats->heat_to_gas_j, stats->evaporated_kg,
            stats->condensed_kg, stats->condensation_splits);
    }
    if (stats->field_heated_particles > 0) {
        ImGui::TextDisabled(
            "Heated by thermal fields: %zu parcels",
            stats->field_heated_particles);
    }
    if (stats->cold_unsupported > 0 && stats->frozen_particles == 0) {
        warningText(
            "%zu cold parcels touch no support (collider / closed wall).",
            stats->cold_unsupported);
    }
    if (stats->viscosity_field_built) {
        ImGui::TextDisabled(
            "Viscosity in liquid: %.2e .. %.2e m^2/s",
            stats->min_viscosity,
            stats->max_viscosity);
    }
    return edited;
}

void drawSourceControls(
    RayTrophiSim::SimulationFlowSourceDesc& source,
    float ambient_kelvin,
    const RayTrophiSim::Fluid::APICSolverParams& params) {
    int temperature_mode = source.fluid_temperature_override ? 1 : 0;
    const char* modes[] = {"Domain Ambient", "Custom"};
    if (ImGui::Combo(
            "Initial Temperature",
            &temperature_mode,
            modes,
            IM_ARRAYSIZE(modes))) {
        source.fluid_temperature_override = temperature_mode == 1;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Domain Ambient: born at the domain's effective ambient temperature.\n"
            "Custom: born at the Kelvin value below, then cooled by the domain solver.");
    }

    if (source.fluid_temperature_override) {
        ImGui::DragFloat(
            "Custom Temperature (K)",
            &source.fluid_temperature_kelvin,
            0.5f,
            1.0f,
            5000.0f,
            "%.1f");
        source.fluid_temperature_kelvin =
            std::max(1.0f, source.fluid_temperature_kelvin);
    }

    bool born_solid = false;
    const float effective_birth = source.fluid_temperature_override
        ? source.fluid_temperature_kelvin
        : RayTrophiSim::Fluid::substanceAmbientBirthKelvin(
              RayTrophiSim::Fluid::substanceTag(source.fluid_substance),
              params, ambient_kelvin, &born_solid);
    ImGui::TextDisabled(
        "Effective Birth Temperature: %.1f K | ambient delta %+.1f K",
        effective_birth,
        effective_birth - ambient_kelvin);
    if (born_solid) {
        ImGui::TextDisabled(
            "%s is born at %.1f K, in its own phase (ambient %.1f K); it changes phase over time.",
            source.fluid_substance.c_str(), effective_birth, ambient_kelvin);
    }
    drawBirthTemperatureWarning(effective_birth, ambient_kelvin, params,
                                RayTrophiSim::Fluid::substanceTag(source.fluid_substance));
}

std::string uniqueSourceName(
    const std::string& base_name,
    const std::vector<RayTrophiSim::SimulationFlowSourceDesc>& sources) {
    const auto nameExists = [&](const std::string& candidate) {
        return std::any_of(
            sources.begin(),
            sources.end(),
            [&](const auto& source) { return source.name == candidate; });
    };
    if (!nameExists(base_name)) {
        return base_name;
    }
    for (int suffix = 2; suffix < 1000000; ++suffix) {
        const std::string candidate = base_name + " " + std::to_string(suffix);
        if (!nameExists(candidate)) {
            return candidate;
        }
    }
    return base_name + " Unique";
}

} // namespace FluidThermalUI
} // namespace ForceFieldUI
