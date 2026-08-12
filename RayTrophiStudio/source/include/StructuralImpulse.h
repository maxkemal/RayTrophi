#pragma once

#include "Vec3.h"

#include <algorithm>
#include <cstdint>
#include <string>

namespace RayTrophiSim {

// World AABB of a fracture group, accumulated from real shard GEOMETRY.
//
// ★ This lives in a shared header because the same box must answer two
// questions that used to be answered separately and disagreed: the area the
// blast front pushes on, and the `world_extent` scripts read back. When the
// reporting path built its box from shard CENTRES, a single-shard cluster
// reported extent (0,0,0) — zero projected area, zero impulse, immune to
// pressure forever — while the impulse path, working from vertices, saw a real
// box. Two answers to one question is the bug; one accumulator is the fix.
struct FractureGroupBounds {
    Vec3 min = Vec3(1e30f, 1e30f, 1e30f);
    Vec3 max = Vec3(-1e30f, -1e30f, -1e30f);
    bool any = false;

    void add(const Vec3& p) {
        min = Vec3(std::min(min.x, p.x), std::min(min.y, p.y), std::min(min.z, p.z));
        max = Vec3(std::max(max.x, p.x), std::max(max.y, p.y), std::max(max.z, p.z));
        any = true;
    }
    Vec3 center() const { return any ? (min + max) * 0.5f : Vec3(0.0f); }
    Vec3 extent() const {
        return any ? Vec3(std::max(max.x - min.x, 0.0f),
                          std::max(max.y - min.y, 0.0f),
                          std::max(max.z - min.z, 0.0f))
                   : Vec3(0.0f);
    }
};

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
    // Projected area of the most strongly affected group, m². Reported because
    // impulse is now proportional to it, so a surprising impulse is usually a
    // surprising area (an unexpectedly large group AABB) rather than a bad
    // pressure value.
    float last_projected_area_m2 = 0.0f;
};
} // namespace RayTrophiSim
