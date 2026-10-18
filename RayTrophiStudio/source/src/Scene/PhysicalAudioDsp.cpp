#include "Audio/PhysicalAudioDsp.h"
#include <algorithm>
#include <cmath>
#include <string>

namespace audio {
PhysicalAudioDsp::PhysicalAudioDsp(const nlohmann::json& recipe) {
    random_ = recipe.at("seed").get<std::uint32_t>();
    const float brightness = recipe.at("brightness").get<float>();
    const std::string name = recipe.at("model");
    model_ = name == "gas_flow" ? 1 : name == "friction" ? 2 :
        name == "rolling" ? 3 : name == "impact" ? 4 : name == "contact_activity" ? 5 :
        name == "deformation" ? 6 : name == "fracture_activity" ? 7 : 0;
    const float hz = model_ == 3 ? 90 + 260 * brightness :
        model_ == 1 ? 180 + 2400 * brightness : 700 + 6800 * brightness;
    cutoff_ = 1.0f - std::exp(-6.28318530718f * hz / 48000.0f);
    if (model_ < 4) {
        return;
    }
    const std::string material = recipe.at("resonator");
    const float base = material == "metal" ? 520 : material == "glass" ? 820 :
        material == "stone" ? 310 : 190;
    const float size = recipe.at("size_m").get<float>();
    const float decay = recipe.at("decay_seconds").get<float>();
    for (std::size_t index = 0; index < modes_.size(); ++index) {
        const float mode = static_cast<float>(index);
        const float ratio = material == "wood" ? 1 + mode * 1.73f :
            1 + mode * 1.41f + mode * mode * 0.19f;
        const float frequency = std::clamp(base * ratio * 0.1f / size, 30.0f, 10000.0f);
        auto& state = modes_[index];
        const float angle = 6.28318530718f * frequency / 48000.0f;
        state.cosine = std::cos(angle);
        state.sine = std::sin(angle);
        state.damping = std::exp(-6.907755279f * (1 + mode * (1 - brightness)) /
            (decay * 48000.0f));
        state.weight = 1.0f / (6.0f * (1 + mode * 0.4f));
    }
}

float PhysicalAudioDsp::noise() {
    random_ += 0x9e3779b9u;
    auto value = random_;
    value = (value ^ (value >> 16)) * 0x21f0aaadu;
    value = (value ^ (value >> 15)) * 0x735a2d97u;
    value ^= value >> 15;
    return static_cast<float>(value >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

float PhysicalAudioDsp::resonance(float excitation) {
    float sample = 0;
    for (auto& mode : modes_) {
        if (excitation != 0) {
            mode.first += excitation;
            const float radiusSquared = mode.first * mode.first + mode.second * mode.second;
            if (radiusSquared > 1) {
                const float inverse = 1.0f / std::sqrt(radiusSquared);
                mode.first *= inverse;
                mode.second *= inverse;
            }
        }
        const float first = mode.damping *
            (mode.cosine * mode.first - mode.sine * mode.second);
        mode.second = mode.damping *
            (mode.sine * mode.first + mode.cosine * mode.second);
        mode.first = std::abs(first) < 1e-20f ? 0 : first;
        if (std::abs(mode.second) < 1e-20f) {
            mode.second = 0;
        }
        sample += mode.second * mode.weight;
    }
    return sample;
}

float PhysicalAudioDsp::render(float* output, std::size_t count, float activity, float impulse) {
    const float target = std::isfinite(activity) ? std::clamp(activity, 0.0f, 1.0f) : 0;
    const float excitation = std::isfinite(impulse) ? std::clamp(impulse, 0.0f, 1.0f) : 0;
    float peak = 0;
    for (std::size_t index = 0; index < count; ++index) {
        float sample = 0;
        if (impactMode()) {
            sample = resonance(index == 0 ? excitation : 0);
        } else if (model_ >= 5) {
            level_ += (target - level_) * 0.001041124f;
            const float white = noise();
            float drive = white * level_ * 0.02f;
            if (model_ == 6) {
                // Smooth strain texture; this is not a recovered stress waveform.
                slow_ += (white - slow_) * 0.002f;
                drive = slow_ * level_ * 0.08f;
            } else if (model_ == 7) {
                // Seeded synthetic crackle, not discrete solver fracture events.
                const float threshold = 1.0f - level_ * 0.004f;
                drive = std::abs(white) > threshold ? white * level_ * 0.2f : 0;
            }
            sample = resonance(drive);
        } else {
            level_ += (target - level_) * 0.001041124f;
            const float white = noise();
            low_ += (white - low_) * cutoff_;
            slow_ += (white - slow_) * 0.002f;
            float texture = low_;
            if (model_ == 0) {
                texture = (low_ - slow_) * (0.65f + 0.35f * std::abs(white));
            } else if (model_ == 2) {
                texture = (white - low_) * 0.55f + low_ * 0.25f;
            }
            sample = texture * level_ * 0.35f;
        }
        if (output) {
            output[index] = sample;
        }
        peak = std::max(peak, std::abs(sample));
    }
    return peak;
}
}
