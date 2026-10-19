#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace audio {

// Bounded in-memory authoring, shared by synchronous and background render paths.
inline constexpr double designMaxSeconds = 600;
inline constexpr std::uint64_t designWorkingPcmBudget = 512ULL * 1024 * 1024;

inline void validateDesignCapacity(double inputSeconds, double ratio, double tailSeconds,
                                   unsigned channels) {
    const double outputSeconds = inputSeconds * ratio + tailSeconds;
    if (!std::isfinite(inputSeconds) || !std::isfinite(outputSeconds) ||
        inputSeconds < 0.05 || inputSeconds > designMaxSeconds ||
        outputSeconds <= 0 || outputSeconds > designMaxSeconds ||
        (channels != 1 && channels != 2)) {
        throw std::invalid_argument(
            "Designer input/output including stretch and tail must fit within 600 seconds");
    }
    // Reserve four banks each for input and output (retained design, decoded PCM,
    // working buffers and analysis/stretch copies). This is a PCM estimate, not
    // an absolute process-memory bound; effect-specific limits still apply.
    const double estimatedBytes = (inputSeconds + outputSeconds) *
        48000.0 * channels * sizeof(float) * 4;
    if (estimatedBytes > static_cast<double>(designWorkingPcmBudget)) {
        throw std::invalid_argument(
            "Designer estimated PCM working set exceeds 512 MiB; select a shorter range");
    }
}

} // namespace audio
