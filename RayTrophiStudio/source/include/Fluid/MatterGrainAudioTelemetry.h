#pragma once

#include "Fluid/MatterGrain.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace RayTrophiSim::Fluid {

inline constexpr std::size_t kGrainAudioWords = 4;

inline void reduceGrainAudioTelemetry(const std::vector<uint32_t>& words,
                                     std::size_t header, double duration,
                                     MatterGrainStepReport& report) {
    double energy = 0.0;
    double moment[3] = {};
    if (words.size() < header || (words.size() - header) % kGrainAudioWords != 0 ||
        !std::isfinite(duration) || duration <= 0.0) {
        return;
    }
    for (std::size_t base = header; base < words.size(); base += kGrainAudioWords) {
        float row[kGrainAudioWords];
        std::memcpy(row, words.data() + base, sizeof(row));
        if (!std::isfinite(row[0]) || row[0] < 0.0f) {
            return;
        }
        energy += row[0];
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(row[axis + 1])) {
                return;
            }
            moment[axis] += row[axis + 1];
        }
    }
    if (!std::isfinite(energy) || energy > 1e9 || energy / duration > 1e9) {
        return;
    }
    Vec3 position(0);
    if (energy > 0.0) {
        position = Vec3(static_cast<float>(moment[0] / energy),
                        static_cast<float>(moment[1] / energy),
                        static_cast<float>(moment[2] / energy));
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z)) {
            return;
        }
    }
    report.contact_damping_measured = true;
    report.contact_damping_energy_j = energy;
    report.contact_damping_duration_seconds = duration;
    report.contact_damping_position = position;
}

} // namespace RayTrophiSim::Fluid
