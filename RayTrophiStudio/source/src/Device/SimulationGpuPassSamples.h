#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace RayTrophiSim {

// The report retains individual dispatches for these passes only. The existing
// aggregate counters still cover every timed kernel.
inline void recordPlayerGpuPassSample(std::string_view kernel, double ms,
                                      std::vector<double>& samples) {
    constexpr std::size_t sampleLimit = 8192;
    if (samples.size() < sampleLimit &&
        (kernel == "sim_particle_spawn" || kernel == "sim_particle_ballistic" ||
         kernel == "sim_particle_collide" || kernel == "paint_wet")) {
        samples.push_back(ms);
    }
}

inline double gpuTimestampDeltaMs(std::uint64_t start, std::uint64_t end,
                                  std::uint64_t mask, double periodNs) {
    const std::uint64_t ticks = (end - start) & mask;
    return static_cast<double>(ticks) * periodNs * 1e-6;
}

} // namespace RayTrophiSim
