#pragma once

#include "Fluid/MatterGrain.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace RayTrophiSim::Fluid {

inline constexpr std::size_t kGrainAudioWords = 12;

inline void reduceGrainAudioTelemetry(const std::vector<uint32_t>& words,
                                     std::size_t header, double duration,
                                     MatterGrainStepReport& report) {
    report.contact_damping_measured = false;
    report.friction_measured = false;
    report.rolling_measured = false;
    double energy[3] = {};
    double moment[3][3] = {};
    if (words.size() < header || (words.size() - header) % kGrainAudioWords != 0 ||
        !std::isfinite(duration) || duration <= 0.0) {
        return;
    }
    for (std::size_t base = header; base < words.size(); base += kGrainAudioWords) {
        float row[kGrainAudioWords];
        std::memcpy(row, words.data() + base, sizeof(row));
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const std::size_t offset = channel * 4;
            if (!std::isfinite(row[offset]) || row[offset] < 0.0f) {
                return;
            }
            energy[channel] += row[offset];
            for (std::size_t axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(row[offset + axis + 1])) {
                    return;
                }
                moment[channel][axis] += row[offset + axis + 1];
            }
        }
    }
    Vec3 position[3] = {Vec3(0), Vec3(0), Vec3(0)};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const double value = energy[channel];
        if (!std::isfinite(value) || value > 1e9 || value / duration > 1e9) {
            return;
        }
        if (value > 0.0) {
            position[channel] = Vec3(static_cast<float>(moment[channel][0] / value),
                static_cast<float>(moment[channel][1] / value),
                static_cast<float>(moment[channel][2] / value));
            const auto& p = position[channel];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                return;
            }
        }
    }
    report.contact_damping_measured = true;
    report.contact_damping_energy_j = energy[0];
    report.contact_damping_duration_seconds = duration;
    report.contact_damping_position = position[0];
    report.friction_measured = true;
    report.friction_energy_j = energy[1];
    report.friction_position = position[1];
    report.rolling_measured = true;
    report.rolling_energy_j = energy[2];
    report.rolling_position = position[2];
}

} // namespace RayTrophiSim::Fluid
