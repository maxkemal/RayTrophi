#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/PhysicalAudioCharacterDsp.h"
#include <algorithm>
#include <cmath>
#include <string>

namespace audio {
namespace {
constexpr float rate = 48000;
constexpr float tau = 6.28318530718f;
float pole(float hz) {
    return 1 - std::exp(-tau * hz / rate);
}
float clean(float value) {
    return std::abs(value) < 1e-20f ? 0 : value;
}
}

PhysicalAudioCharacterDsp::PhysicalAudioCharacterDsp(const nlohmann::json& recipe) {
    random_ = recipe.at("seed").get<std::uint32_t>();
    const std::string name = recipe.at("model");
    model_ = name == "gas_flow" ? 1 : name == "friction" ? 2 : name == "impact" ? 4 :
        name == "fracture_activity" ? 7 : 0;
    const auto surface = recipe.value("surface", std::string("organic"));
    surface_ = surface == "mineral" ? 1 : surface == "metal" ? 2 : surface == "glass" ? 3 : 0;
    stone_ = recipe.value("resonator", std::string("wood")) == "stone";
    texture_ = recipe.at("texture").get<float>();
    variation_ = recipe.at("variation").get<float>();
    body_ = recipe.at("body").get<float>();
    const float brightness = recipe.at("brightness").get<float>();
    const float size = recipe.value("size_m", 0.1f);
    const float decay = recipe.value("decay_seconds", 0.4f);
    lowCoefficient_ = pole(model_ == 1 ? 60 + 150 * brightness :
        model_ == 4 ? 250 + 650 * brightness : 120 + 260 * brightness);
    midCoefficient_ = pole(model_ == 1 ? 350 + 1000 * brightness :
        model_ == 2 ? (surface_ == 1 ? 2400 : surface_ == 3 ? 850 : 1400) +
            2600 * brightness : 1600 + 3000 * brightness);
    airCoefficient_ = pole(model_ == 1 ? 500 + 1000 * brightness : 1700 + 3000 * brightness);
    const float grainSeconds = model_ == 4 ? (stone_ ? 0.0015f : 0.004f) :
        model_ == 7 ? 0.001f + texture_ * 0.004f : 0.002f + texture_ * 0.012f;
    grainDecay_ = std::exp(-1 / (rate * grainSeconds));
    dustDecay_ = std::exp(-1 / (rate * (0.015f + 0.06f * texture_)));
    for (std::size_t i = 0; i < grains_.size(); ++i) {
        const float order = static_cast<float>(i);
        // Liquid grains occupy distinct pitches and decay times, not one pitched chord.
        const float frequency = model_ == 0 ? (160 + 100 * brightness) * std::pow(1.19f, order) :
            (stone_ ? 280 : 130) * (1 + order * 1.37f + order * order * 0.07f) * 0.1f / size;
        const float seconds = model_ == 0 ? 0.025f + 0.045f * (1 - order / 16) :
            std::clamp(decay * (stone_ ? 0.045f : 0.10f) /
                (1 + order * 0.7f), 0.003f, 1.0f);
        auto& grain = grains_[i];
        const float angle = tau * std::clamp(frequency, 35.0f, 11000.0f) / rate;
        grain.cosine = std::cos(angle);
        grain.sine = std::sin(angle);
        grain.damping = std::exp(-6.907755279f / (seconds * rate));
        grain.weight = model_ == 0 ? 1 : 1 / (1 + order * 1.5f);
    }
}

float PhysicalAudioCharacterDsp::noise() {
    random_ += 0x9e3779b9u;
    auto value = random_;
    value = (value ^ (value >> 16)) * 0x21f0aaadu;
    value = (value ^ (value >> 15)) * 0x735a2d97u;
    value ^= value >> 15;
    return static_cast<float>(value >> 8) * (2.0f / 16777216.0f) - 1;
}

void PhysicalAudioCharacterDsp::excite(unsigned index, float value) {
    auto& grain = grains_[index % grains_.size()];
    grain.real += value;
    const float radius = grain.real * grain.real + grain.imaginary * grain.imaginary;
    if (radius > 0.0625f) {
        const float scale = 0.25f / std::sqrt(radius);
        grain.real *= scale;
        grain.imaginary *= scale;
    }
}

float PhysicalAudioCharacterDsp::grains() {
    float sum = 0;
    for (auto& grain : grains_) {
        const float real = grain.damping *
            (grain.cosine * grain.real - grain.sine * grain.imaginary);
        grain.imaginary = clean(grain.damping *
            (grain.sine * grain.real + grain.cosine * grain.imaginary));
        grain.real = clean(real);
        sum += grain.imaginary * grain.weight;
    }
    return sum;
}

float PhysicalAudioCharacterDsp::render(float* output, std::size_t count,
                                      float activity, float impulse) {
    const float target = std::isfinite(activity) ? std::clamp(activity, 0.0f, 1.0f) : 0;
    const float impact = std::isfinite(impulse) ? std::clamp(impulse, 0.0f, 1.0f) : 0;
    float peak = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const float white = noise();
        const float chance = noise();
        const float random = noise();
        level_ = clean(level_ + (target - level_) * 0.001041124f);
        low_ = clean(low_ + (white - low_) * lowCoefficient_);
        mid_ = clean(mid_ + (white - mid_) * midCoefficient_);
        air_ = clean(air_ + (mid_ - air_) * airCoefficient_);
        if (gustLeft_ == 0) {
            gustTarget_ = random;
            gustLeft_ = 4000 + static_cast<unsigned>((chance + 1) * 6000);
        }
        --gustLeft_;
        gust_ += (gustTarget_ - gust_) * 0.00025f;
        const float motion = 0.8f + variation_ * gust_ * 0.55f;
        const float band = mid_ - low_;
        float sample = 0;
        if (model_ == 0) {
            const float probability = level_ * (25 + 240 * texture_) / rate;
            if (chance > 1 - 2 * probability) {
                excite(static_cast<unsigned>((random + 1) * 7.999f),
                    level_ * (0.10f + 0.13f * std::abs(random)));
                grainEnvelope_ = std::min(1.0f, grainEnvelope_ + level_ * 0.3f);
            }
            // Foreground irregular wet grains; broadband wash is a quiet background.
            sample = grains() * (0.20f + body_ * 0.25f) +
                band * grainEnvelope_ * texture_ * 0.18f +
                low_ * level_ * body_ * motion * 0.14f;
        } else if (model_ == 1) {
            // Smooth moving pressure/noise layers, without the liquid's pitched grains.
            sample = (low_ * (0.8f + body_) + (air_ - low_) * (0.15f + texture_ * 0.3f)) *
                level_ * motion * 0.65f;
        } else if (model_ == 2) {
            const float probability = level_ * (50 + 500 * texture_) / rate;
            if (chance > 1 - 2 * probability) {
                grainEnvelope_ = std::min(1.0f, grainEnvelope_ + 0.25f + 0.5f * std::abs(random));
            }
            const float scrape = (0.15f + 0.85f * grainEnvelope_) * level_;
            if (surface_ == 0) {
                sample = (band * 0.55f + low_ * body_ * 0.6f) * scrape;
            } else if (surface_ == 1) {
                sample = ((white - mid_) * 0.45f + band * 0.5f) * scrape;
            } else if (surface_ == 2) {
                sample = ((white - air_) * 0.35f + band * 0.15f) * scrape;
            } else {
                sample = (air_ - low_) * level_ * (0.1f + 0.12f * grainEnvelope_);
            }
            sample *= 0.5f * motion;
        } else if (model_ == 4) {
            if (i == 0 && impact > 0) {
                grainEnvelope_ = std::min(1.0f, grainEnvelope_ + impact);
                for (unsigned mode = 0; mode < 6; ++mode) {
                    excite(mode, impact * 0.6f);
                }
            }
            const float transient = stone_ ? band * 0.65f + low_ * 0.35f : low_;
            sample = grains() * (0.6f + 0.4f * body_) +
                transient * grainEnvelope_ * (0.25f + texture_ * 0.35f);
        } else {
            // Clusters of short cracks plus quieter debris; no uniform ringing popcorn.
            const float probability = level_ * (0.8f + 9 * texture_) / rate;
            if (microLeft_ == 0 && chance > 1 - 2 * probability) {
                microLeft_ = 2 + static_cast<unsigned>((random + 1) * 3);
                microDelay_ = 0;
                clusterLevel_ = level_ * (0.45f + 0.4f * std::abs(random));
            }
            if (microLeft_ > 0) {
                if (microDelay_ == 0) {
                    --microLeft_;
                    grainEnvelope_ = std::min(1.0f, grainEnvelope_ + clusterLevel_);
                    dustEnvelope_ = std::min(1.0f, dustEnvelope_ + clusterLevel_ * 0.25f);
                    excite(static_cast<unsigned>((random + 1) * 2.999f), clusterLevel_ * 0.12f);
                    microDelay_ = 80 + static_cast<unsigned>((chance + 1) * 420);
                } else {
                    --microDelay_;
                }
            }
            sample = (band * 0.8f + (white - mid_) * texture_ * 0.25f) * grainEnvelope_ +
                band * dustEnvelope_ * 0.12f + grains() * body_ * 0.35f;
            sample *= 0.55f;
        }
        grainEnvelope_ = clean(grainEnvelope_ * grainDecay_);
        dustEnvelope_ = clean(dustEnvelope_ * dustDecay_);
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
