#pragma once

#include "json.hpp"
#include <cmath>
#include <stdexcept>

// Opt-in response: old scenes retain their original emission equation.
struct VolumeEmissionResponse {
    bool enabled = false;
    float temperature_power = 1.0f;
    float density_power = 1.0f;
    float exposure_ev = 0.0f;
    float saturation = 1.0f;

    nlohmann::json toJson() const {
        return {{"enabled", enabled}, {"temperature_power", temperature_power},
                {"density_power", density_power}, {"exposure_ev", exposure_ev},
                {"saturation", saturation}};
    }

    void fromJson(const nlohmann::json& j) {
        enabled = j.value("enabled", false);
        temperature_power = j.value("temperature_power", 1.0f);
        density_power = j.value("density_power", 1.0f);
        exposure_ev = j.value("exposure_ev", 0.0f);
        saturation = j.value("saturation", 1.0f);
        if (!std::isfinite(temperature_power) || temperature_power < 0.0f ||
            temperature_power > 8.0f || !std::isfinite(density_power) ||
            density_power < 0.0f || density_power > 4.0f ||
            !std::isfinite(exposure_ev) || exposure_ev < -16.0f || exposure_ev > 16.0f ||
            !std::isfinite(saturation) || saturation < 0.0f || saturation > 2.0f) {
            throw std::invalid_argument("invalid volume emission response");
        }
    }

    void pack(float (&out)[5]) const {
        out[0] = enabled ? 1.0f : 0.0f;
        out[1] = temperature_power;
        out[2] = density_power;
        out[3] = exposure_ev;
        out[4] = saturation;
    }
};
