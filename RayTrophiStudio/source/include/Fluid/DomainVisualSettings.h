#pragma once

#include "Fluid/DomainVisualSamples.h"
#include "json.hpp"

namespace RayTrophiSim::Fluid {

struct DomainVisualSettings {
    bool enabled = false;
    DomainFields::VisualSampleSettings sampling;
    std::uint64_t working_budget_mb = 1024;
};

inline void to_json(nlohmann::json& json, const DomainVisualSettings& value) {
    json = {{"schema", 1}, {"enabled", value.enabled},
        {"target_volume", value.sampling.target_volume},
        {"max_samples", value.sampling.max_samples}, {"seed", value.sampling.seed},
        {"cfl", value.sampling.cfl}, {"max_substeps", value.sampling.max_substeps},
        {"reseed_hysteresis", value.sampling.reseed_hysteresis},
        {"working_budget_mb", value.working_budget_mb}};
}

inline void from_json(const nlohmann::json& patch, DomainVisualSettings& value) {
    if (!patch.is_object()) {
        throw std::invalid_argument("visual sample settings must be an object");
    }
    auto next = value;
    for (auto it = patch.begin(); it != patch.end(); ++it) {
        const auto& key = it.key();
        const auto& v = it.value();
        if (key == "enabled") {
            if (!v.is_boolean()) {
                throw std::invalid_argument("enabled must be boolean");
            }
            next.enabled = v.get<bool>();
        } else if (key == "target_volume" || key == "cfl" || key == "reseed_hysteresis") {
            if (!v.is_number()) {
                throw std::invalid_argument(key + " must be a number");
            }
            const double number = v.get<double>();
            if (key == "target_volume") {
                next.sampling.target_volume = number;
            }
            if (key == "cfl") {
                next.sampling.cfl = number;
            }
            if (key == "reseed_hysteresis") {
                next.sampling.reseed_hysteresis = number;
            }
        } else if (key == "max_samples" || key == "seed" || key == "max_substeps" ||
                   key == "working_budget_mb" || key == "schema") {
            if ((!v.is_number_unsigned() && !v.is_number_integer()) ||
                (v.is_number_integer() && !v.is_number_unsigned() && v.get<int64_t>() < 0)) {
                throw std::invalid_argument(key + " must be a nonnegative integer");
            }
            const auto number = v.get<uint64_t>();
            if (key == "max_samples") {
                next.sampling.max_samples = number;
            }
            if (key == "seed") {
                next.sampling.seed = number;
            }
            if (key == "working_budget_mb") {
                next.working_budget_mb = number;
            }
            if (key == "max_substeps") {
                if (number > 65536) {
                    throw std::invalid_argument("max_substeps exceeds 65536");
                }
                next.sampling.max_substeps = static_cast<uint32_t>(number);
            }
            if (key == "schema" && number != 1) {
                throw std::invalid_argument("unsupported visual sample settings schema");
            }
        } else {
            throw std::invalid_argument("unknown visual sample setting: " + key);
        }
    }
    DomainFields::validateVisualSettings(next.sampling);
    if (next.working_budget_mb < 16 || next.working_budget_mb > 16384) {
        throw std::invalid_argument("working_budget_mb must be in [16, 16384]");
    }
    if (next.sampling.max_samples > 4000000 || next.sampling.max_samples * 768ull >=
            next.working_budget_mb * 1024ull * 1024ull) {
        throw std::invalid_argument("CPU visual sample limit or staging memory budget exceeded");
    }
    value = next;
}

inline uint64_t domainVisualSettingsKey(const DomainVisualSettings& value) {
    const auto text = nlohmann::json(value).dump();
    uint64_t result = 1469598103934665603ull;
    for (const unsigned char byte : text) {
        result = (result ^ byte) * 1099511628211ull;
    }
    return result;
}

struct DomainVisualState;

} // namespace RayTrophiSim::Fluid
