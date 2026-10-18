#include "Audio/AudioParametricEq.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace audio {
namespace {
using json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;
constexpr double rate = 48000;
void require(bool valid, const char* message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
double frequencyAt(unsigned point) {
    return 20 * std::pow(1000.0, point / 255.0);
}
double decibels(std::complex<double> value) {
    return 20 * std::log10(std::max(1e-12, std::abs(value)));
}
}

EqBiquad::EqBiquad(const json& band) {
    const double frequency = band.at("frequency_hz");
    const double q = band.at("q"), gain = band.at("gain_db");
    require(std::isfinite(frequency) && frequency >= 20 && frequency <= 20000 &&
        std::isfinite(q) && q >= 0.1 && q <= 10 &&
        std::isfinite(gain) && gain >= -24 && gain <= 24, "Invalid EQ coefficient input");
    const double w = 2 * pi * frequency / rate;
    const double c = std::cos(w), s = std::sin(w), alpha = s / (2 * q);
    const double a = std::pow(10.0, gain / 40);
    const auto mode = band.at("mode").get<std::string>();
    double a0 = 1;
    if (mode == "highpass" || mode == "lowpass") {
        const double sign = mode == "highpass" ? 1 : -1;
        b0_ = (1 + sign * c) / 2;
        b1_ = -sign * (1 + sign * c);
        b2_ = b0_;
        a0 = 1 + alpha;
        a1_ = -2 * c;
        a2_ = 1 - alpha;
    } else if (mode == "lowshelf" || mode == "highshelf") {
        const double sign = mode == "highshelf" ? 1 : -1;
        const double beta = 2 * std::sqrt(a) * alpha;
        b0_ = a * ((a + 1) + sign * (a - 1) * c + beta);
        b1_ = -2 * sign * a * ((a - 1) + sign * (a + 1) * c);
        b2_ = a * ((a + 1) + sign * (a - 1) * c - beta);
        a0 = (a + 1) - sign * (a - 1) * c + beta;
        a1_ = 2 * sign * ((a - 1) - sign * (a + 1) * c);
        a2_ = (a + 1) - sign * (a - 1) * c - beta;
    } else {
        require(mode == "peak", "Unknown EQ mode");
        b0_ = 1 + alpha * a;
        b1_ = -2 * c;
        b2_ = 1 - alpha * a;
        a0 = 1 + alpha / a;
        a1_ = -2 * c;
        a2_ = 1 - alpha / a;
    }
    require(std::isfinite(a0) && a0 > 0, "Invalid EQ denominator");
    b0_ /= a0;
    b1_ /= a0;
    b2_ /= a0;
    a1_ /= a0;
    a2_ /= a0;
    require(std::isfinite(b0_) && std::isfinite(b1_) && std::isfinite(b2_) &&
        std::isfinite(a1_) && std::isfinite(a2_) && std::abs(a2_) < 1 &&
        1 + a1_ + a2_ > 0 && 1 - a1_ + a2_ > 0, "Unstable EQ coefficients");
}

float EqBiquad::process(float sample, unsigned channel) {
    const double output = sample * b0_ + z1_[channel];
    z1_[channel] = sample * b1_ - output * a1_ + z2_[channel];
    z2_[channel] = sample * b2_ - output * a2_;
    if (std::abs(z1_[channel]) < 1e-30) {
        z1_[channel] = 0;
    }
    if (std::abs(z2_[channel]) < 1e-30) {
        z2_[channel] = 0;
    }
    return static_cast<float>(output);
}

std::complex<double> EqBiquad::response(double frequency) const {
    const auto z = std::polar(1.0, -2 * pi * frequency / rate);
    return (b0_ + b1_ * z + b2_ * z * z) / (1.0 + a1_ * z + a2_ * z * z);
}

void applyParametricEq(AudioPcm& pcm, const json& effect, const AudioCancelCheck& check) {
    require(pcm.channels == 1 || pcm.channels == 2, "EQ requires mono or stereo PCM");
    for (const auto& band : effect.at("bands")) {
        if (!band.at("enabled").get<bool>()) {
            continue;
        }
        EqBiquad filter(band);
        for (std::size_t frame = 0; frame < pcm.frames(); ++frame) {
            if (check && frame % 1024 == 0) {
                check();
            }
            for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                auto& sample = pcm.samples[frame * pcm.channels + channel];
                sample = filter.process(sample, channel);
                require(std::isfinite(sample), "Nonfinite EQ output");
            }
        }
    }
    const float output = static_cast<float>(std::pow(10.0, effect.value("output_db", 0.0) / 20));
    for (std::size_t index = 0; index < pcm.samples.size(); ++index) {
        if (check && index % 2048 == 0) {
            check();
        }
        pcm.samples[index] *= output;
        require(std::isfinite(pcm.samples[index]), "Nonfinite EQ trim output");
    }
}

std::array<std::complex<double>, 256> parametricEqCurve(const json& effect) {
    std::array<std::complex<double>, 256> result;
    result.fill({1, 0});
    if (!effect.at("enabled").get<bool>()) {
        return result;
    }
    const double mix = effect.at("mix"), output = std::pow(10.0, effect.value("output_db", 0.0) / 20);
    for (const auto& band : effect.at("bands")) {
        if (!band.at("enabled").get<bool>()) {
            continue;
        }
        const EqBiquad filter(band);
        for (unsigned point = 0; point < result.size(); ++point) {
            result[point] *= filter.response(frequencyAt(point));
        }
    }
    for (auto& value : result) {
        value = (1 - mix) + mix * output * value;
    }
    return result;
}

json parametricEqVisual(const json& effect) {
    const auto curve = parametricEqCurve(effect);
    json result = {{"sample_rate", 48000}, {"points", json::array()},
        {"band_curves", json::array()}, {"linear_response_only", true}};
    double peak = 0;
    for (unsigned point = 0; point < curve.size(); ++point) {
        const double db = decibels(curve[point]);
        peak = std::max(peak, db);
        result["points"].push_back({frequencyAt(point), db});
    }
    for (const auto& band : effect.at("bands")) {
        json entry = {{"band", band}, {"points", json::array()}};
        const EqBiquad filter(band);
        for (unsigned point = 0; point < curve.size(); ++point) {
            entry["points"].push_back({frequencyAt(point), decibels(filter.response(frequencyAt(point)))});
        }
        result["band_curves"].push_back(std::move(entry));
    }
    result["estimated_max_boost_db"] = peak;
    return result;
}
}
