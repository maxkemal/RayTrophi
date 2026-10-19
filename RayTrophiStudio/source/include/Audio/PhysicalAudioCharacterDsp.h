#pragma once
#include "json.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace audio {
// Revision 2 timbres. Authored statistical textures, not measured solver events.
class PhysicalAudioCharacterDsp {
public:
    explicit PhysicalAudioCharacterDsp(const nlohmann::json& recipe);
    float render(float* output, std::size_t count, float activity, float impulse);

private:
    struct Grain {
        float real = 0;
        float imaginary = 0;
        float cosine = 0;
        float sine = 0;
        float damping = 0;
        float weight = 0;
    };
    std::array<Grain, 16> grains_{};
    std::uint32_t random_ = 1;
    int model_ = 0;
    int surface_ = 0;
    bool stone_ = false;
    float texture_ = 0.5f;
    float variation_ = 0.5f;
    float body_ = 0.5f;
    float level_ = 0;
    float low_ = 0;
    float mid_ = 0;
    float air_ = 0;
    float lowCoefficient_ = 0;
    float midCoefficient_ = 0;
    float airCoefficient_ = 0;
    float gust_ = 0;
    float gustTarget_ = 0;
    unsigned gustLeft_ = 0;
    float grainEnvelope_ = 0;
    float grainDecay_ = 0;
    float dustEnvelope_ = 0;
    float dustDecay_ = 0;
    unsigned microLeft_ = 0;
    unsigned microDelay_ = 0;
    float clusterLevel_ = 0;
    float dcInput_ = 0;
    float dcOutput_ = 0;
    float noise();
    void excite(unsigned index, float value);
    float grains();
};
}
