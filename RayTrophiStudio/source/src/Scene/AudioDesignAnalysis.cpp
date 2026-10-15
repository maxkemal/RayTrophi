#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioDesignDsp.h"
#include "../../../external/signalsmith-linear/fft.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace audio {
namespace {
using json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;

void require(bool valid, const char* message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
unsigned integer(const json& params, const char* key, unsigned fallback, unsigned low, unsigned high) {
    const auto value = params.value(key, json(fallback));
    require(value.is_number_integer() && value.get<double>() >= low && value.get<double>() <= high,
            "Analysis dimensions must be integers within their bounds");
    return value.get<unsigned>();
}
} // namespace

json designPcmWaveform(const AudioPcm& pcm, std::uint64_t first, std::uint64_t last,
                      unsigned points, const AudioCancelCheck& check) {
    require(first < last && last <= pcm.frames() && points >= 16 && points <= 4096,
            "Invalid waveform range or point count");
    points = static_cast<unsigned>(std::min<std::uint64_t>(points, last - first));
    json envelope = json::array();
    for (unsigned point = 0; point < points; ++point) {
        if (check) {
            check();
        }
        const auto begin = first + (last - first) * point / points;
        const auto end = first + (last - first) * (point + 1) / points;
        json minimum = json::array(), maximum = json::array(), rms = json::array();
        for (unsigned channel = 0; channel < pcm.channels; ++channel) {
            float low = std::numeric_limits<float>::max(), high = -low;
            double energy = 0;
            for (auto frame = begin; frame < end; ++frame) {
                const float sample = pcm.samples[frame * pcm.channels + channel];
                low = std::min(low, sample);
                high = std::max(high, sample);
                energy += double(sample) * sample;
            }
            minimum.push_back(low);
            maximum.push_back(high);
            rms.push_back(std::sqrt(energy / (end - begin)));
        }
        envelope.push_back({{"first_frame", begin}, {"end_frame_exclusive", end},
            {"min", minimum}, {"max", maximum}, {"rms", rms}});
    }
    return {{"sample_rate", pcm.sample_rate}, {"channels", pcm.channels}, {"frames", pcm.frames()},
        {"first_frame", first}, {"end_frame_exclusive", last}, {"points", envelope}};
}

void validateDesignAnalysis(const AudioPcm& pcm, const json& params) {
    require(params.is_object(), "Analysis params must be an object");
    for (auto entry = params.begin(); entry != params.end(); ++entry) {
        require(entry.key() == "fft_size" || entry.key() == "points" ||
                entry.key() == "time_bins" || entry.key() == "channel" ||
                entry.key() == "continuous", "Unknown analysis field");
    }
    require(!params.contains("continuous") || params.at("continuous").is_boolean(),
            "continuous must be boolean");
    const unsigned size = integer(params, "fft_size", 2048, 512, 8192);
    require((size & (size - 1)) == 0, "FFT size must be a power of two");
    integer(params, "points", 256, 16, 1024);
    integer(params, "time_bins", 64, 4, 128);
    const auto selected = params.value("channel", std::string("mix"));
    require(selected == "mix" || selected == "left" || selected == "right",
            "Analysis channel must be mix, left or right");
    require(selected != "right" || pcm.channels == 2, "Right analysis needs a stereo sound");
    require(pcm.frames() > 0 && (pcm.channels == 1 || pcm.channels == 2), "No PCM to analyze");
}

json analyzeDesignPcm(const AudioPcm& pcm, const json& params, const AudioCancelCheck& check) {
    validateDesignAnalysis(pcm, params);
    const auto checkpoint = [&] {
        if (check) {
            check();
        }
    };
    checkpoint();
    const bool continuous = params.value("continuous", true);
    const unsigned size = integer(params, "fft_size", 2048, 512, 8192);
    const unsigned points = integer(params, "points", 256, 16, 1024);
    const unsigned timeBins = integer(params, "time_bins", 64, 4, 128);
    const auto selected = params.value("channel", std::string("mix"));
    json channels = json::array();
    double totalEnergy = 0;
    float totalPeak = 0;
    std::uint64_t hash = 14695981039346656037ULL, clipped = 0;
    const auto* bytes = reinterpret_cast<const unsigned char*>(pcm.samples.data());
    for (std::size_t byte = 0; byte < pcm.samples.size() * sizeof(float); ++byte) {
        if (byte % 65536 == 0) {
            checkpoint();
        }
        hash = (hash ^ bytes[byte]) * 1099511628211ULL;
    }
    for (unsigned channel = 0; channel < pcm.channels; ++channel) {
        double mean = 0, energy = 0;
        float peak = 0;
        std::uint64_t channelClipped = 0, crossings = 0;
        float previous = pcm.samples[channel];
        for (std::size_t frame = 0; frame < pcm.frames(); ++frame) {
            const float sample = pcm.samples[frame * pcm.channels + channel];
            require(std::isfinite(sample), "Nonfinite analysis sample");
            peak = std::max(peak, std::abs(sample));
            mean += sample;
            energy += double(sample) * sample;
            channelClipped += std::abs(sample) >= 1;
            crossings += (sample >= 0) != (previous >= 0);
            previous = sample;
        }
        const double rms = std::sqrt(energy / pcm.frames());
        channels.push_back({{"peak", peak}, {"rms", rms}, {"dc_offset", mean / pcm.frames()},
            {"peak_dbfs", 20 * std::log10(std::max(1e-9, double(peak)))},
            {"rms_dbfs", 20 * std::log10(std::max(1e-9, rms))},
            {"crest_db", peak > 0 ? 20 * std::log10(peak / std::max(1e-9, rms)) : 0},
            {"clip_samples", channelClipped}, {"zero_crossings_per_second",
                crossings * double(pcm.sample_rate) / pcm.frames()}});
        totalEnergy += energy;
        totalPeak = std::max(totalPeak, peak);
        clipped += channelClipped;
    }
    const unsigned firstChannel = selected == "right" ? 1 : 0;
    const unsigned channelCount = selected == "mix" ? pcm.channels : 1;
    const double spectralScale = std::max(1.0, double(totalPeak));
    signalsmith::linear::Pow2RealFFT<float> fft(size);
    std::vector<float> window(size), time(size);
    std::vector<std::complex<float>> frequency(size / 2);
    double windowSum = 0;
    for (unsigned frame = 0; frame < size; ++frame) {
        window[frame] = static_cast<float>(0.5 - 0.5 * std::cos(2 * pi * frame / (size - 1)));
        windowSum += window[frame];
    }
    json spectrogram = json::array(), frequencies = json::array();
    std::vector<std::vector<double>> columnPower(timeBins, std::vector<double>(points, 0));
    std::vector<double> peakPower(points, 0);
    for (unsigned point = 0; point < points; ++point) {
        frequencies.push_back(20 * std::pow(1000.0, double(point) / (points - 1)));
    }
    const std::uint64_t hop = size / 4;
    const auto windows = continuous ? (pcm.frames() + hop - 1) / hop + 1 : timeBins;
    for (std::uint64_t slice = 0; slice < windows; ++slice) {
        checkpoint();
        const auto center = static_cast<std::int64_t>(continuous ? slice * hop :
                                                     pcm.frames() * slice / timeBins);
        const auto column = continuous ? std::min<std::uint64_t>(timeBins - 1,
            static_cast<std::uint64_t>(center) * timeBins / pcm.frames()) : slice;
        std::vector<double> power(points, 0);
        for (unsigned offset = 0; offset < channelCount; ++offset) {
            const unsigned channel = firstChannel + offset;
            for (unsigned frame = 0; frame < size; ++frame) {
                const auto source = center + static_cast<std::int64_t>(frame) - size / 2;
                time[frame] = source >= 0 && source < static_cast<std::int64_t>(pcm.frames()) ?
                    static_cast<float>(pcm.samples[source * pcm.channels + channel] /
                                       spectralScale * window[frame]) : 0;
            }
            fft.fft(time.data(), frequency.data());
            for (unsigned point = 0; point < points; ++point) {
                const double bin = frequencies[point].get<double>() * size / pcm.sample_rate;
                const auto low = std::min<unsigned>(size / 2 - 1, static_cast<unsigned>(bin));
                const auto high = std::min<unsigned>(size / 2 - 1, low + 1);
                const double fraction = std::clamp(bin - low, 0.0, 1.0);
                // Packed real FFT stores Nyquist in bin zero's imaginary component.
                const double a = low == 0 ? std::pow(frequency[0].real(), 2) : std::norm(frequency[low]);
                const double b = std::norm(frequency[high]);
                power[point] += ((1 - fraction) * a + fraction * b) *
                                4 * spectralScale * spectralScale /
                                (windowSum * windowSum * channelCount);
            }
        }
        for (unsigned point = 0; point < points; ++point) {
            peakPower[point] = std::max(peakPower[point], power[point]);
            columnPower[column][point] = std::max(columnPower[column][point], power[point]);
        }
    }
    for (const auto& power : columnPower) {
        json column = json::array();
        for (const auto value : power) {
            column.push_back(std::clamp(10 * std::log10(std::max(1e-12, value)), -120.0, 60.0));
        }
        spectrogram.push_back(column);
    }
    json spectrum = json::array();
    for (unsigned point = 0; point < points; ++point) {
        spectrum.push_back({frequencies[point],
            std::clamp(10 * std::log10(std::max(1e-12, peakPower[point])), -120.0, 60.0)});
    }
    const auto waveform = designPcmWaveform(pcm, 0, pcm.frames(), 256);
    json combined = json::array();
    for (const auto& point : waveform.at("points")) {
        float low = std::numeric_limits<float>::max(), high = -low;
        for (unsigned channel = 0; channel < pcm.channels; ++channel) {
            low = std::min(low, point.at("min")[channel].get<float>());
            high = std::max(high, point.at("max")[channel].get<float>());
        }
        combined.push_back({low, high});
    }
    std::ostringstream fingerprint;
    fingerprint << std::hex << std::setw(16) << std::setfill('0') << hash;
    return {{"frames", pcm.frames()}, {"sample_rate", pcm.sample_rate}, {"channels", pcm.channels},
        {"duration_seconds", double(pcm.frames()) / pcm.sample_rate}, {"peak", totalPeak},
        {"rms", std::sqrt(totalEnergy / pcm.samples.size())}, {"pcm_hash", fingerprint.str()},
        {"channel_stats", channels}, {"clip_samples", clipped}, {"waveform", combined},
        {"channel_waveform", waveform.at("points")}, {"spectrum", spectrum},
        {"spectrogram", spectrogram}, {"frequencies_hz", frequencies}, {"fft_size", size},
        {"frequency_resolution_hz", double(pcm.sample_rate) / size}, {"time_bins", timeBins},
        {"time_step_seconds", double(pcm.frames()) / pcm.sample_rate / timeBins},
        {"analysis_channel", selected}, {"continuous", continuous}, {"hop_frames", continuous ? hop : 0},
        {"spectral_method", continuous ? "Hann STFT; peak aggregation; full time coverage" :
                                          "Hann-windowed sampled STFT"}};
}
} // namespace audio
