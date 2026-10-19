#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/PhysicalAudioNaturalDsp.h"
#include <algorithm>
#include <cmath>
#include <string>

namespace audio {
namespace {
constexpr float sampleRate = 48000;
constexpr float tau = 6.28318530718f;
float lowPass(float hz) {
    return 1 - std::exp(-tau * hz / sampleRate);
}
float clean(float value) {
    return std::abs(value) < 1e-20f ? 0 : value;
}
}

PhysicalAudioNaturalDsp::PhysicalAudioNaturalDsp(const nlohmann::json& recipe) {
    random_ = recipe.at("seed").get<std::uint32_t>();
    const std::string name = recipe.at("model");
    model_ = name == "gas_flow" ? 1 : name == "friction" ? 2 :
        name == "rolling" ? 3 : name == "impact" ? 4 : name == "contact_activity" ? 5 :
        name == "deformation" ? 6 : name == "fracture_activity" ? 7 : 0;
    const float brightness = recipe.at("brightness").get<float>();
    texture_ = recipe.at("texture").get<float>();
    variation_ = recipe.at("variation").get<float>();
    body_ = recipe.at("body").get<float>();
    lowCoefficient_ = lowPass(model_ == 3 ? 70 + 180 * brightness : 100 + 500 * brightness);
    midCoefficient_ = lowPass(model_ == 1 ? 600 + 4200 * brightness :
        1000 + 7000 * brightness);
    burstDecay_ = std::exp(-1 / (sampleRate * (0.001f + 0.008f * texture_)));
    const std::string material = recipe.value("resonator", std::string("wood"));
    const float size = recipe.value("size_m", 0.1f);
    const float decay = recipe.value("decay_seconds", model_ == 0 ? 0.06f : 0.15f);
    const float base = model_ == 0 ? 400 + 1800 * brightness :
        material == "metal" ? 520 : material == "glass" ? 820 :
        material == "stone" ? 310 : 190;
    float weightSum = 0;
    for (std::size_t i = 0; i < modes_.size(); ++i) {
        const float order = static_cast<float>(i);
        const float detune = 1 + noise() * variation_ * 0.025f;
        const float ratio = model_ == 0 ? 1 + order * 0.23f :
            material == "wood" ? 1 + order * 1.73f :
            1 + order * 1.41f + order * order * 0.19f;
        const float frequency = std::clamp(base * ratio * 0.1f / size * detune,
            30.0f, 12000.0f);
        auto& mode = modes_[i];
        mode.cosine = std::cos(tau * frequency / sampleRate);
        mode.sine = std::sin(tau * frequency / sampleRate);
        mode.damping = std::exp(-6.907755279f * (1 + order * (0.15f + 0.5f * (1 - brightness))) /
            (decay * sampleRate));
        mode.weight = 1 / (1 + order * (0.3f + 0.5f * (1 - body_)));
        mode.coupling = (i % 3 == 1 ? -1.0f : 1.0f) * (0.7f + 0.3f * std::abs(noise()));
        weightSum += mode.weight;
    }
    for (auto& mode : modes_) {
        mode.weight /= weightSum;
    }
    const bool tunedImpact = model_ == 4 && (material == "wood" || material == "stone");
    if (recipe.value("character_revision", 1) == 2 &&
        (model_ == 0 || model_ == 1 || model_ == 2 || model_ == 7 || tunedImpact)) {
        character_.emplace(recipe);
    }
}

float PhysicalAudioNaturalDsp::noise() {
    random_ += 0x9e3779b9u;
    auto value = random_;
    value = (value ^ (value >> 16)) * 0x21f0aaadu;
    value = (value ^ (value >> 15)) * 0x735a2d97u;
    value ^= value >> 15;
    return static_cast<float>(value >> 8) * (2.0f / 16777216.0f) - 1;
}

float PhysicalAudioNaturalDsp::resonance(float excitation) {
    float sample = 0;
    for (auto& mode : modes_) {
        mode.real += excitation * mode.coupling;
        const float radiusSquared = mode.real * mode.real + mode.imaginary * mode.imaginary;
        if (radiusSquared > 1) {
            const float inverse = 1 / std::sqrt(radiusSquared);
            mode.real *= inverse;
            mode.imaginary *= inverse;
        }
        const float real = mode.damping *
            (mode.cosine * mode.real - mode.sine * mode.imaginary);
        mode.imaginary = clean(mode.damping *
            (mode.sine * mode.real + mode.cosine * mode.imaginary));
        mode.real = clean(real);
        sample += mode.imaginary * mode.weight;
    }
    return sample;
}

float PhysicalAudioNaturalDsp::render(float* output, std::size_t count,
                                    float activity, float impulse) {
    if (character_) {
        return character_->render(output, count, activity, impulse);
    }
    const float target = std::isfinite(activity) ? std::clamp(activity, 0.0f, 1.0f) : 0;
    const float impact = std::isfinite(impulse) ? std::clamp(impulse, 0.0f, 1.0f) : 0;
    float peak = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const float white = noise();
        const float randomEvent = noise();
        level_ = clean(level_ + (target - level_) * 0.001041124f);
        low_ = clean(low_ + (white - low_) * lowCoefficient_);
        mid_ = clean(mid_ + (white - mid_) * midCoefficient_);
        slow_ = clean(slow_ + (randomEvent - slow_) * 0.0004f);
        gust_ = clean(gust_ + (slow_ - gust_) * 0.00008f);
        const float motion = 1 + variation_ * std::clamp(gust_ * 18, -0.7f, 0.7f);
        const float middle = mid_ - low_;
        const float high = white - mid_;
        const float rough = 0.6f + 0.4f * std::abs(randomEvent);
        float sample = 0;
        if (model_ == 4) {
            if (i == 0 && impact > 0) {
                burst_ = std::min(1.0f, burst_ + impact);
            }
            const float onset = i == 0 ? impact : 0;
            sample = resonance(onset) * (0.35f + 0.35f * body_) +
                high * burst_ * texture_ * 0.16f;
        } else if (model_ == 0) {
            // Statistical liquid microtexture; these are not measured bubble/splash events.
            const float probability = level_ * (20 + 160 * texture_) / sampleRate;
            const float grain = randomEvent > 1 - 2 * probability ?
                level_ * (0.08f + 0.12f * variation_) : 0;
            sample = ((middle * (0.3f + 0.5f * texture_) * rough +
                low_ * body_ * 0.8f) * level_ * motion +
                resonance(grain) * texture_) * 0.4f;
        } else if (model_ == 1) {
            sample = (middle * (0.25f + 0.55f * texture_) + low_ * (0.3f + body_) +
                high * texture_ * 0.06f) * level_ * motion * 0.5f;
        } else if (model_ == 2) {
            sample = ((high * (0.12f + 0.4f * texture_) + middle * 0.35f) * rough +
                low_ * body_ * 0.5f) * level_ * motion * 0.4f;
        } else if (model_ == 3) {
            const float contact = randomEvent > 1 - level_ * (0.0002f + texture_ * 0.002f) ?
                level_ * 0.08f : 0;
            burst_ = std::min(1.0f, burst_ + contact);
            sample = (low_ * (0.4f + body_) + middle * burst_ * texture_) *
                level_ * motion * 0.55f;
        } else {
            float drive = middle * level_ * (0.008f + 0.025f * texture_) * motion;
            if (model_ == 6) {
                drive = low_ * level_ * (0.02f + 0.04f * texture_) * motion;
            } else if (model_ == 7) {
                // Synthetic crackle controlled by activity, not solver fracture identities.
                const float probability = level_ * (10 + 150 * texture_) / sampleRate;
                drive = randomEvent > 1 - 2 * probability ?
                    white * level_ * (0.1f + 0.2f * variation_) : 0;
                burst_ = std::min(1.0f, burst_ + std::abs(drive));
            }
            sample = resonance(drive) * (0.35f + 0.45f * body_) +
                high * burst_ * texture_ * 0.08f;
        }
        burst_ = clean(burst_ * burstDecay_);
        // Reject slow DC without changing event scheduling or normalizing individual takes.
        const float filtered = clean(sample - dcInput_ + 0.9987f * dcOutput_);
        dcInput_ = sample;
        dcOutput_ = filtered;
        if (output) {
            output[i] = filtered;
        }
        peak = std::max(peak, std::abs(filtered));
    }
    return peak;
}
}
