#pragma once
#include "Audio/AudioDesignDsp.h"
#include <array>
#include <complex>

namespace audio {
// Coefficients are prepared outside rendering. Each stereo channel owns its state.
class EqBiquad {
public:
    EqBiquad() = default;
    explicit EqBiquad(const nlohmann::json& normalizedBand);
    float process(float sample, unsigned channel);
    std::complex<double> response(double frequency) const;
private:
    double b0_ = 1, b1_ = 0, b2_ = 0, a1_ = 0, a2_ = 0;
    std::array<double, 2> z1_{}, z2_{};
};
void applyParametricEq(AudioPcm& pcm, const nlohmann::json& normalizedEffect,
                       const AudioCancelCheck& check);
nlohmann::json parametricEqVisual(const nlohmann::json& normalizedEffect);
std::array<std::complex<double>, 256> parametricEqCurve(
    const nlohmann::json& normalizedEffect);
}
