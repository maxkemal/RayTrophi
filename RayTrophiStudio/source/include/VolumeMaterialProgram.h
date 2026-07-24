/*
 * RayTrophi Studio - compiled volume material output (homogeneous domain)
 *
 * This deliberately contains no renderer or node-system types.  A volume graph
 * compiler writes this POD-like result and VolumeShader folds it into the
 * existing GPU payload.  Spatial programs use a separate runtime in later
 * phases; constant graphs therefore remain completely free at ray-march time.
 */
#pragma once

#include "json.hpp"
#include <algorithm>
#include <cstdint>

enum class VolumeMaterialSlot : uint32_t {
    Density = 0,
    ScatterColor,
    ScatterStrength,
    AbsorptionColor,
    AbsorptionStrength,
    EmissionColor,
    EmissionStrength,
    Anisotropy,
    MultiScatter,
    Count
};

struct VolumeMaterialProgram {
    bool active = false;
    uint32_t written = 0;

    float density = 1.0f;
    float scatter_color[3] = {1.0f, 1.0f, 1.0f};
    float scatter_strength = 1.0f;
    float absorption_color[3] = {0.0f, 0.0f, 0.0f};
    float absorption_strength = 0.1f;
    float emission_color[3] = {1.0f, 0.5f, 0.1f};
    float emission_strength = 0.0f;
    float anisotropy = 0.0f;
    float multi_scatter = 0.0f;

    // Bounded spatial density slice (Vulkan first): source grid multiplied by
    // a 3D FBM field. General bytecode remains a later phase; these explicit
    // fields keep the first production path deterministic and allocation-free.
    bool density_noise_enabled = false;
    float density_noise_scale = 5.0f;
    float density_noise_strength = 1.0f;
    int density_noise_detail = 3;
    int density_noise_seed = 0;

    static constexpr uint32_t bit(VolumeMaterialSlot slot) {
        return 1u << static_cast<uint32_t>(slot);
    }
    bool has(VolumeMaterialSlot slot) const { return (written & bit(slot)) != 0u; }
    void set(VolumeMaterialSlot slot, bool enabled) {
        if (enabled) written |= bit(slot);
        else written &= ~bit(slot);
    }

    void sanitize() {
        density = (std::max)(0.0f, density);
        scatter_strength = (std::max)(0.0f, scatter_strength);
        absorption_strength = (std::max)(0.0f, absorption_strength);
        emission_strength = (std::max)(0.0f, emission_strength);
        anisotropy = (std::max)(-0.99f, (std::min)(0.99f, anisotropy));
        multi_scatter = (std::max)(0.0f, (std::min)(1.0f, multi_scatter));
        density_noise_scale = (std::max)(0.001f, density_noise_scale);
        density_noise_strength = (std::max)(0.0f, (std::min)(1.0f, density_noise_strength));
        density_noise_detail = (std::max)(1, (std::min)(8, density_noise_detail));
        for (float& c : scatter_color) c = (std::max)(0.0f, c);
        for (float& c : absorption_color) c = (std::max)(0.0f, c);
        for (float& c : emission_color) c = (std::max)(0.0f, c);
        const uint32_t valid = (1u << static_cast<uint32_t>(VolumeMaterialSlot::Count)) - 1u;
        written &= valid;
    }

    nlohmann::json toJson() const {
        return {
            {"version", 2}, {"active", active}, {"written", written},
            {"density", density},
            {"scatter_color", {scatter_color[0], scatter_color[1], scatter_color[2]}},
            {"scatter_strength", scatter_strength},
            {"absorption_color", {absorption_color[0], absorption_color[1], absorption_color[2]}},
            {"absorption_strength", absorption_strength},
            {"emission_color", {emission_color[0], emission_color[1], emission_color[2]}},
            {"emission_strength", emission_strength}, {"anisotropy", anisotropy},
            {"multi_scatter", multi_scatter},
            {"density_noise_enabled", density_noise_enabled},
            {"density_noise_scale", density_noise_scale},
            {"density_noise_strength", density_noise_strength},
            {"density_noise_detail", density_noise_detail},
            {"density_noise_seed", density_noise_seed}
        };
    }

    void fromJson(const nlohmann::json& j) {
        active = j.value("active", false);
        written = j.value("written", 0u);
        density = j.value("density", density);
        scatter_strength = j.value("scatter_strength", scatter_strength);
        absorption_strength = j.value("absorption_strength", absorption_strength);
        emission_strength = j.value("emission_strength", emission_strength);
        anisotropy = j.value("anisotropy", anisotropy);
        multi_scatter = j.value("multi_scatter", multi_scatter);
        density_noise_enabled = j.value("density_noise_enabled", false);
        density_noise_scale = j.value("density_noise_scale", density_noise_scale);
        density_noise_strength = j.value("density_noise_strength", density_noise_strength);
        density_noise_detail = j.value("density_noise_detail", density_noise_detail);
        density_noise_seed = j.value("density_noise_seed", density_noise_seed);
        readColor(j, "scatter_color", scatter_color);
        readColor(j, "absorption_color", absorption_color);
        readColor(j, "emission_color", emission_color);
        sanitize();
    }

private:
    static void readColor(const nlohmann::json& j, const char* key, float (&out)[3]) {
        if (!j.contains(key) || !j[key].is_array() || j[key].size() < 3) return;
        out[0] = j[key][0].get<float>();
        out[1] = j[key][1].get<float>();
        out[2] = j[key][2].get<float>();
    }
};
