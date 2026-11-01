#pragma once

#include "json.hpp"
#include "VolumeTemperatureMapping.h"
#include "VolumeEmissionMask.h"
#include "VolumeEmissionTransport.h"
#include <cmath>
#include <stdexcept>

// Opt-in response: old scenes retain their original emission equation.
struct VolumeEmissionResponse {
    bool enabled = false;
    float temperature_power = 1.0f;
    float density_power = 1.0f;
    float exposure_ev = 0.0f;
    float saturation = 1.0f;
    // 0=source default (legacy on imports), 1=Kelvin, 2=normalized authored range.
    int temperature_units = 0;
    bool mask_enabled = false;
    float mask_center = 0.2f;
    float mask_width = 0.1f;
    float mask_strength = 1.0f;
    bool independent_emission = false;
    // Extinction falls by up to soot_burnout between start and end on the
    // ramp's temperature coordinate. 0 = off. Needs independent_emission.
    float soot_burnout = 0.0f;
    float soot_burnout_start = 0.2f;
    float soot_burnout_end = 0.6f;
    // Above 1 opens the cool end of the ramp, below 1 the hot end: u^gamma before
    // the ramp, heat and burnout read it. Vulkan RT and RayFusion.
    float temperature_gamma = 1.0f;
    // Reaction field radiance added on top of the thermal term, so a burning cell
    // glows whatever its temperature and density say. 0 is off and costs nothing.
    float flame_emission = 0.0f;

    nlohmann::json toJson() const {
        return {{"enabled", enabled}, {"temperature_power", temperature_power},
                {"density_power", density_power}, {"exposure_ev", exposure_ev},
                {"saturation", saturation}, {"temperature_units", temperature_units},
                {"mask_enabled", mask_enabled}, {"mask_center", mask_center},
                {"mask_width", mask_width}, {"mask_strength", mask_strength},
                {"independent_emission", independent_emission},
                {"soot_burnout", soot_burnout},
                {"soot_burnout_start", soot_burnout_start},
                {"soot_burnout_end", soot_burnout_end},
                {"temperature_gamma", temperature_gamma},
                {"flame_emission", flame_emission}};
    }

    void fromJson(const nlohmann::json& j) {
        enabled = j.value("enabled", false);
        temperature_power = j.value("temperature_power", 1.0f);
        density_power = j.value("density_power", 1.0f);
        exposure_ev = j.value("exposure_ev", 0.0f);
        saturation = j.value("saturation", 1.0f);
        mask_enabled = j.value("mask_enabled", false);
        mask_center = j.value("mask_center", 0.2f);
        mask_width = j.value("mask_width", 0.1f);
        mask_strength = j.value("mask_strength", 1.0f);
        if (j.contains("independent_emission") && !j.at("independent_emission").is_boolean()) {
            throw std::invalid_argument("independent_emission must be boolean");
        }
        independent_emission = j.value("independent_emission", false);
        soot_burnout = j.value("soot_burnout", 0.0f);
        soot_burnout_start = j.value("soot_burnout_start", 0.2f);
        soot_burnout_end = j.value("soot_burnout_end", 0.6f);
        temperature_gamma = j.value("temperature_gamma", 1.0f);
        flame_emission = j.value("flame_emission", 0.0f);
        if (j.contains("temperature_units") &&
            (!j.at("temperature_units").is_number_integer() ||
             j.at("temperature_units") < 0 || j.at("temperature_units") > 2)) {
            throw std::invalid_argument("temperature_units must be an integer in 0..2");
        }
        temperature_units = j.value("temperature_units", 0);
        if (!std::isfinite(temperature_power) || temperature_power < 0.0f ||
            temperature_power > 8.0f || !std::isfinite(density_power) ||
            density_power < 0.0f || density_power > 4.0f ||
            !std::isfinite(exposure_ev) || exposure_ev < -16.0f || exposure_ev > 16.0f ||
            !std::isfinite(saturation) || saturation < 0.0f || saturation > 2.0f ||
            temperature_units < 0 || temperature_units > 2 ||
            !std::isfinite(mask_center) || mask_center < 0.0f || mask_center > 100000.0f ||
            !std::isfinite(mask_width) || mask_width < 1e-6f || mask_width > 100000.0f ||
            !std::isfinite(mask_strength) || mask_strength < 0.0f || mask_strength > 1.0f ||
            !std::isfinite(soot_burnout) || soot_burnout < 0.0f || soot_burnout > 1.0f ||
            !std::isfinite(soot_burnout_start) || soot_burnout_start < 0.0f ||
            !std::isfinite(soot_burnout_end) || soot_burnout_end > 1.0f ||
            soot_burnout_end <= soot_burnout_start ||
            !std::isfinite(temperature_gamma) || temperature_gamma < 0.1f ||
            temperature_gamma > 8.0f ||
            !std::isfinite(flame_emission) || flame_emission < 0.0f) {
            throw std::invalid_argument("invalid volume emission response");
        }
    }

    void pack(float (&out)[5]) const {
        out[0] = policyFlags();
        out[1] = temperature_power;
        out[2] = density_power;
        out[3] = exposure_ev;
        out[4] = saturation;
    }

    void packTemperaturePolicy(float (&curve)[12], bool sourceKelvin = false) const {
        curve[10] = static_cast<float>(
            volumeResolveTemperatureUnits(temperature_units, sourceKelvin));
        curve[11] = flame_emission;
    }

    void packMask(float (&out)[4]) const {
        out[0] = mask_enabled ? 1.0f : 0.0f;
        out[1] = mask_center;
        out[2] = mask_width;
        out[3] = mask_strength;
    }

    void packSootBurnout(float (&out)[3]) const {
        out[0] = soot_burnout;
        out[1] = soot_burnout_start;
        out[2] = soot_burnout_end;
    }

    // Burnout reads the reaction field too: soot burns where the flame is.
    bool wantsFlame() const {
        return flame_emission > 0.0f || (independent_emission && soot_burnout > 0.0f);
    }

    float maskFactor(float density, float multiplier) const {
        float packed[4];
        packMask(packed);
        return volumeEmissionMaskFactor(density,
            volumeEmissionSamplingMultiplier(multiplier, policyFlags()), packed);
    }

    float policyFlags() const {
        return (enabled ? 1.0f : 0.0f) + (independent_emission ? 2.0f : 0.0f);
    }

    template <typename T>
    T optIn(T legacy, T color, float density, float multiplier, float temperature,
        float radiance, float strength, bool thermal) const {
        if (!independent_emission) return legacy;
        float response[5];
        pack(response);
        return volumeEmissionBuild(color, radiance,
            volumeEmissionInputDensity(density, multiplier, response[0]),
            temperature, strength, response, thermal);
    }

    template <typename T>
    T evaluate(T color, float density, float multiplier, float temperature,
        float radiance, float strength, bool thermal) const {
        T legacy = color;
        legacy.x *= radiance * density * strength;
        legacy.y *= radiance * density * strength;
        legacy.z *= radiance * density * strength;
        return optIn(legacy, color, density, multiplier, temperature, radiance, strength, thermal);
    }

    template <typename T>
    T finish(T value, float density, float multiplier, bool thermal) const {
        return volumeEmissionFinish(value, maskFactor(density, multiplier),
            independent_emission && thermal);
    }
};
