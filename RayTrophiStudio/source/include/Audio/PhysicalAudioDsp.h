#pragma once
#include "json.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace audio {
// Validated recipe construction is outside playback. Rendering owns only fixed state.
// Used by the device data source and deterministic offline history evaluator.
class PhysicalAudioDsp {
public:
    explicit PhysicalAudioDsp(const nlohmann::json& recipe);
    float render(float* output, std::size_t count, float activity, float impulse = 0);
    bool impactMode() const {
        return model_ == 4;
    }

private:
    struct Mode {
        float first = 0;
        float second = 0;
        float cosine = 0;
        float sine = 0;
        float damping = 0;
        float weight = 0;
    };
    std::array<Mode, 6> modes_{};
    std::uint32_t random_ = 1;
    float level_ = 0;
    float low_ = 0;
    float slow_ = 0;
    float cutoff_ = 0.1f;
    int model_ = 0;
    float noise();
    float resonance(float excitation);
};
}
