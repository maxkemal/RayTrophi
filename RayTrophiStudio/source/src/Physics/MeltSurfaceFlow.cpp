#include "MeltSurfaceFlow.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {

bool solveMeltSurfaceFlow(const std::vector<Vec3>& rest,
                          const std::vector<uint32_t>& triangles,
                          const std::vector<float>& melt,
                          const std::vector<float>& local_mass,
                          const MeltSurfaceFlowSettings& settings,
                          std::vector<Vec3>& out) {
    const std::size_t count = rest.size();
    if (count < 3 || melt.size() != count || local_mass.size() != count ||
        triangles.size() < 3) return false;

    std::vector<float> flow(count, 0.0f), scratch(count, 0.0f), weights(count, 0.0f);
    for (std::size_t i = 0; i < count; ++i)
        flow[i] = std::clamp(melt[i], 0.0f, 1.0f);
    for (uint32_t pass = 0; pass < settings.smoothing_passes; ++pass) {
        for (std::size_t i = 0; i < count; ++i) {
            scratch[i] = flow[i] * 2.0f;
            weights[i] = 2.0f;
        }
        auto add = [&](uint32_t target, uint32_t source) {
            if (target >= count || source >= count || target == source) return;
            scratch[target] += flow[source];
            weights[target] += 1.0f;
        };
        for (std::size_t i = 0; i + 2 < triangles.size(); i += 3) {
            const uint32_t a = triangles[i], b = triangles[i + 1], c = triangles[i + 2];
            add(a, b); add(b, a); add(b, c); add(c, b); add(c, a); add(a, c);
        }
        for (std::size_t i = 0; i < count; ++i) {
            // Never diffuse melt into a completely cold vertex.
            scratch[i] = melt[i] > 0.0f ? scratch[i] / weights[i] : 0.0f;
        }
        flow.swap(scratch);
    }

    float min_y = rest[0].y, max_y = rest[0].y;
    Vec3 center(0.0f);
    float center_weight = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        min_y = std::min(min_y, rest[i].y);
        max_y = std::max(max_y, rest[i].y);
        center += rest[i] * flow[i];
        center_weight += flow[i];
    }
    if (center_weight <= 1.0e-6f) { out = rest; return true; }
    center *= 1.0f / center_weight;

    const float height = std::max(max_y - min_y, 1.0e-4f);
    const float fluidity = 1.0f - std::clamp(settings.viscosity, 0.0f, 1.0f);
    out = rest;
    for (std::size_t i = 0; i < count; ++i) {
        const float amount = std::pow(flow[i], 1.35f) * fluidity;
        if (amount <= 0.0f) continue;
        const float height_loss = std::clamp(
            settings.maximum_height_loss * amount, 0.0f, 0.92f);
        out[i].y = std::max(min_y, rest[i].y - height * height_loss);

        // If height becomes sy times the rest height, sqrt(1/sy) is the X/Z
        // scale that preserves a box-like volume. The local/capped form avoids
        // explosive silhouettes on thin or sparsely tessellated meshes.
        const float sy = std::max(1.0f - height_loss, 0.08f);
        const float represented_mass = std::clamp(local_mass[i], 0.0f, 1.0f);
        const float target_scale = std::sqrt(std::max(represented_mass, 0.01f) / sy);
        const float lateral = std::clamp(target_scale - 1.0f,
            -0.75f, settings.maximum_lateral_gain) * amount;
        out[i].x += (rest[i].x - center.x) * lateral;
        out[i].z += (rest[i].z - center.z) * lateral;
    }
    return true;
}

} // namespace RayTrophiSim
