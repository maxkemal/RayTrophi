#pragma once
#include <cstdint>

namespace atmosphere {
std::uint64_t nextLensWeatherIdentity();
struct LensWeatherSettings {
    std::uint64_t identity = nextLensWeatherIdentity();
    bool enabled = false;
    bool hydrophobic = false;
    float hood_deg = 0.0f;
    float diameter_mm = 50.0f;
    std::uint32_t seed = 1;
    std::uint32_t history_revision = 1;
    std::uint32_t revision = 1; // Transient reset generation, not project data.
};

} // namespace atmosphere
