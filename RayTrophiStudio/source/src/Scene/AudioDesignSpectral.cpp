#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioDesignDsp.h"
#include "../../../external/signalsmith-linear/fft.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace audio {
namespace {
constexpr unsigned size = 2048;
constexpr unsigned hop = size / 4;
constexpr double pi = 3.14159265358979323846;

double smooth(double value) {
    value = std::clamp(value, 0.0, 1.0);
    return value * value * (3 - 2 * value);
}

struct Region {
    double begin, end, low, high, feather;
    bool ellipse;

    explicit Region(const nlohmann::json& effect)
        : begin(effect.at("start_seconds")), end(effect.at("end_seconds")),
          low(effect.at("low_hz")), high(effect.at("high_hz")),
          feather(effect.at("feather")), ellipse(effect.at("shape") == "ellipse") {}

    double amount(double time, double frequency) const {
        if (time < begin || time > end || frequency < low || frequency > high) {
            return 0;
        }
        const double x = 2 * (time - begin) / (end - begin) - 1;
        const double y = 2 * std::log1p((frequency - low) / low) /
            std::log1p((high - low) / low) - 1;
        const double radius = ellipse ? std::sqrt(x * x + y * y) :
            std::max(std::abs(x), std::abs(y));
        return feather > 0 ? smooth((1 - radius) / feather) : (radius <= 1 ? 1 : 0);
    }
};
}

void applyDesignSpectralRegion(AudioPcm& pcm, const nlohmann::json& effect,
                              const AudioCancelCheck& check) {
    if (pcm.sample_rate != 48000 || (pcm.channels != 1 && pcm.channels != 2)) {
        throw std::invalid_argument("Spectral processing requires mono/stereo 48 kHz PCM");
    }
    const double gainDb = effect.at("gain_db");
    const auto selected = effect.at("channel").get<std::string>();
    if (selected == "right" && pcm.channels != 2) {
        throw std::invalid_argument("Right spectral processing requires stereo PCM");
    }
    if (gainDb == 0 || pcm.frames() == 0) {
        return;
    }
    const Region region(effect);
    signalsmith::linear::Pow2RealFFT<float> fft(size);
    std::vector<float> window(size), time(size), output(size);
    std::vector<std::complex<float>> spectrum(size / 2);
    for (unsigned index = 0; index < size; ++index) {
        window[index] = static_cast<float>(0.5 - 0.5 * std::cos(2 * pi * index / size));
    }
    // Full padded overlap-add covers the first/last sample without edge attenuation.
    // FFT inverse is unnormalised; divide by size as well as the window-square sum.
    const auto frames = static_cast<std::int64_t>(pcm.frames());
    std::vector<float> reconstructed(pcm.frames()), weights(pcm.frames());
    for (unsigned channel = 0; channel < pcm.channels; ++channel) {
        if ((selected == "left" && channel != 0) || (selected == "right" && channel != 1)) {
            continue;
        }
        std::fill(reconstructed.begin(), reconstructed.end(), 0);
        std::fill(weights.begin(), weights.end(), 0);
        for (std::int64_t start = -static_cast<std::int64_t>(size) + hop;
             start < frames; start += hop) {
            if (check) {
                check();
            }
            for (unsigned index = 0; index < size; ++index) {
                const auto frame = start + index;
                time[index] = frame >= 0 && frame < frames ?
                    pcm.samples[static_cast<std::size_t>(frame) * pcm.channels + channel] *
                        window[index] : 0;
            }
            fft.fft(time.data(), spectrum.data());
            const double center = (start + size / 2.0) / pcm.sample_rate;
            // Real FFT packs DC and Nyquist in bin zero. Both are outside 20..20000 Hz.
            for (unsigned bin = 1; bin < size / 2; ++bin) {
                const double frequency = double(bin) * pcm.sample_rate / size;
                const double amount = region.amount(center, frequency);
                spectrum[bin] *= static_cast<float>(std::pow(10.0, gainDb * amount / 20));
            }
            fft.ifft(spectrum.data(), output.data());
            for (unsigned index = 0; index < size; ++index) {
                const auto frame = start + index;
                if (frame >= 0 && frame < frames) {
                    const auto position = static_cast<std::size_t>(frame);
                    reconstructed[position] += output[index] * window[index] / size;
                    weights[position] += window[index] * window[index];
                }
            }
        }
        for (std::size_t frame = 0; frame < pcm.frames(); ++frame) {
            if (check && frame % 4096 == 0) {
                check();
            }
            const float value = reconstructed[frame] / weights[frame];
            if (!std::isfinite(value)) {
                throw std::runtime_error("Nonfinite spectral reconstruction");
            }
            pcm.samples[frame * pcm.channels + channel] = value;
        }
    }
}
} // namespace audio
