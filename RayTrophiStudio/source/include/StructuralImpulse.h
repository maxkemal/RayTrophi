#pragma once

#include "Vec3.h"

#include <cstdint>
#include <string>

namespace RayTrophiSim {
struct StructuralImpulseEvent {
    uint64_t sequence = 0;
    std::string domain;
    Vec3 center = Vec3(0.0f);
    float radius = 1.0f;
    float peak_pressure_kpa = 0.0f;
    float duration_seconds = 0.0f;
    float coupling = 1.0f;
};

struct StructuralImpulseStats {
    uint64_t queued = 0;
    uint64_t consumed = 0;
    uint64_t affected_groups = 0;
    uint64_t fractured_groups = 0;
    float last_peak_pressure_kpa = 0.0f;
    float last_max_impulse = 0.0f;
};
} // namespace RayTrophiSim
