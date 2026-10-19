#pragma once
#include "json.hpp"
#include "Audio/PhysicalAudioCharacterDsp.h"
#include <optional>
#include <array>
#include <cstddef>
#include <cstdint>

namespace audio {
// V2 artistic activity sonification. Fixed state shared by live, replay and bake.
// It does not invent measured solver collisions, splashes or bubble radii.
class PhysicalAudioNaturalDsp {
public:
    explicit PhysicalAudioNaturalDsp(const nlohmann::json& recipe);
    float render(float* output, std::size_t count, float activity, float impulse);

private:
    struct Mode {
        float real = 0;
        float imaginary = 0;
        float cosine = 0;
        float sine = 0;
        float damping = 0;
        float weight = 0;
        float coupling = 1;
    };
    std::optional<PhysicalAudioCharacterDsp> character_;
    std::array<Mode, 12> modes_{};
    std::uint32_t random_ = 1;
    int model_ = 0;
    float texture_ = 0.5f;
    float variation_ = 0.5f;
    float body_ = 0.5f;
    float level_ = 0;
    float low_ = 0;
    float mid_ = 0;
    float slow_ = 0;
    float gust_ = 0;
    float burst_ = 0;
    float dcInput_ = 0;
    float dcOutput_ = 0;
    float lowCoefficient_ = 0;
    float midCoefficient_ = 0;
    float burstDecay_ = 0;
    float noise();
    float resonance(float excitation);
};
}
