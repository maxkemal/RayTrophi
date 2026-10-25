#pragma once

#include "Fluid/FluidViewResolver.h"
#include "json.hpp"
#include <stdexcept>

namespace RayTrophiSim::Fluid {

inline void to_json(nlohmann::json& json, const MatterPhaseOutput& value) {
    json = {{"solid", labelRouteName(value.solid)},
            {"liquid", labelRouteName(value.liquid)}, {"gas", labelRouteName(value.gas)}};
}

inline void from_json(const nlohmann::json& json, MatterPhaseOutput& value) {
    if (!json.is_object()) {
        throw std::invalid_argument("phase output must be an object");
    }
    auto candidate = value;
    for (const auto& field : json.items()) {
        if (field.key() != "solid" && field.key() != "liquid" && field.key() != "gas") {
            throw std::invalid_argument("unknown phase output field: " + field.key());
        }
        LabelRoute route;
        if (!field.value().is_string() ||
            !parseLabelRoute(field.value().get<std::string>(), route)) {
            throw std::invalid_argument("phase output must be follow|sdf|splat|fog|hidden");
        }
        if (field.key() == "gas") {
            if (route != LabelRoute::Follow && route != LabelRoute::Fog &&
                route != LabelRoute::Hidden) {
                throw std::invalid_argument("gas output must be follow|fog|hidden");
            }
            candidate.gas = route;
        } else {
            (field.key() == "solid" ? candidate.solid : candidate.liquid) = route;
        }
    }
    value = candidate;
}

inline MatterPhaseOutput readMatterPhaseOutput(const nlohmann::json& domain) {
    return domain.contains("fluid_phase_output")
        ? domain.at("fluid_phase_output").get<MatterPhaseOutput>()
        : MatterPhaseOutput{};
}

} // namespace RayTrophiSim::Fluid
