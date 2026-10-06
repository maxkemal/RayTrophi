#include "Fluid/MatterPoreAuthoring.h"
#include <stdexcept>

namespace RayTrophiSim::Fluid {

nlohmann::json matterPoreParamsToJson(const MatterPoreParams& params) {
    return {{"enabled", params.enabled}, {"porosity", params.porosity},
        {"permeability_m2", params.permeability_m2}, {"viscosity_pa_s", params.viscosity_pa_s},
        {"gravity_m_s2", params.gravity_m_s2}, {"drainage_scale", params.drainage_scale},
        {"wet_response_enabled", params.wet_response_enabled},
        {"wet_appearance_enabled", params.wet_appearance_enabled},
        {"wet_friction_scale", params.wet_friction_scale},
        {"wet_dilatancy_scale", params.wet_dilatancy_scale},
        {"capillary_cohesion_pa", params.capillary_cohesion_pa},
        {"pore_pressure_scale", params.pore_pressure_scale},
        {"wet_color_scale", params.wet_color_scale},
        {"wet_roughness_scale", params.wet_roughness_scale},
        {"wet_appearance_full_saturation", params.wet_appearance_full_saturation}};
}

MatterPoreParams matterPoreParamsFromJson(const nlohmann::json& settings) {
    MatterPoreParams result;
    std::string error;
    if (!patchMatterPoreParams(settings, result, error)) {
        throw std::runtime_error(error);
    }
    return result;
}

bool patchMatterPoreParams(const nlohmann::json& patch, MatterPoreParams& params,
                          std::string& error) {
    if (!patch.is_object()) {
        error = "C5 settings must be an object";
        return false;
    }
    auto candidate = params;
    for (auto it = patch.begin(); it != patch.end(); ++it) {
        const auto& key = it.key();
        if (key == "enabled" || key == "wet_response_enabled" ||
            key == "wet_appearance_enabled") {
            if (!it.value().is_boolean()) {
                error = "Pore/wet " + key + " must be boolean";
                return false;
            }
            bool* flag = key == "enabled" ? &candidate.enabled
                : (key == "wet_response_enabled" ? &candidate.wet_response_enabled
                    : &candidate.wet_appearance_enabled);
            *flag = it.value().get<bool>();
            continue;
        }
        float* value = nullptr;
        if (key == "porosity") {
            value = &candidate.porosity;
        } else if (key == "permeability_m2") {
            value = &candidate.permeability_m2;
        } else if (key == "viscosity_pa_s") {
            value = &candidate.viscosity_pa_s;
        } else if (key == "gravity_m_s2") {
            value = &candidate.gravity_m_s2;
        } else if (key == "drainage_scale") {
            value = &candidate.drainage_scale;
        } else if (key == "wet_friction_scale") {
            value = &candidate.wet_friction_scale;
        } else if (key == "wet_dilatancy_scale") {
            value = &candidate.wet_dilatancy_scale;
        } else if (key == "capillary_cohesion_pa") {
            value = &candidate.capillary_cohesion_pa;
        } else if (key == "pore_pressure_scale") {
            value = &candidate.pore_pressure_scale;
        } else if (key == "wet_color_scale") {
            value = &candidate.wet_color_scale;
        } else if (key == "wet_roughness_scale") {
            value = &candidate.wet_roughness_scale;
        } else if (key == "wet_appearance_full_saturation") {
            value = &candidate.wet_appearance_full_saturation;
        } else {
            error = "Unknown pore/wet setting: " + key;
            return false;
        }
        if (!it.value().is_number()) {
            error = "C5 " + key + " must be numeric";
            return false;
        }
        *value = it.value().get<float>();
    }
    if (!validateMatterPoreParams(candidate, error)) {
        return false;
    }
    params = candidate;
    return true;
}

} // namespace RayTrophiSim::Fluid
