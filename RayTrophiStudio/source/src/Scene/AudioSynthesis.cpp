#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioSynthesis.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace audio {
namespace {
using json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;
constexpr unsigned rate = 48000;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

double number(const json& params, const char* key, double fallback, double low, double high) {
    if (params.contains(key)) {
        require(params.at(key).is_number(), "Synthesis controls must be numeric");
    }
    const double value = params.value(key, fallback);
    require(std::isfinite(value) && value >= low && value <= high,
            "Synthesis control must be finite and within its documented range");
    return value;
}

void word(std::ofstream& stream, std::uint32_t value, unsigned bytes) {
    for (unsigned index = 0; index < bytes; ++index) {
        stream.put(static_cast<char>((value >> (index * 8)) & 255));
    }
}
}

json synthesisPresets() {
    return {{"version", 1}, {"sample_rate", rate}, {"channels", 1},
        {"presets", json::array({
            {{"id", "tone"}, {"name", "Tone"}, {"frequency_hz", 440}, {"decay_seconds", 1.0}},
            {{"id", "wind"}, {"name", "Wind texture"}, {"frequency_hz", 220}, {"decay_seconds", 1.0}},
            {{"id", "metal"}, {"name", "Metal impact"}, {"frequency_hz", 330}, {"decay_seconds", 0.8}},
            {{"id", "wood"}, {"name", "Wood impact"}, {"frequency_hz", 180}, {"decay_seconds", 0.15}}
        })}};
}

json generateSyntheticAudio(const json& params) {
    require(params.is_object(), "Synthesis parameters must be an object");
    const std::vector<std::string> allowed = {"path", "preset", "duration_seconds", "seed",
        "frequency_hz", "decay_seconds", "brightness", "gain"};
    for (auto entry = params.begin(); entry != params.end(); ++entry) {
        require(std::find(allowed.begin(), allowed.end(), entry.key()) != allowed.end(),
                "Unknown synthesis parameter");
    }
    const std::string preset = params.value("preset", std::string("metal"));
    require(preset == "tone" || preset == "wind" || preset == "metal" || preset == "wood",
            "Unknown synthesis preset");
    const double duration = number(params, "duration_seconds", 2, 0.05, 10);
    const double frequency = number(params, "frequency_hz", preset == "wood" ? 180 :
        preset == "tone" ? 440 : preset == "wind" ? 220 : 330, 20, 4000);
    const double decay = number(params, "decay_seconds", preset == "wood" ? 0.15 : 0.8, 0.01, 10);
    const double brightness = number(params, "brightness", 0.5, 0, 1);
    const double gain = number(params, "gain", 0.5, 0, 1);
    if (params.contains("seed")) {
        require(params.at("seed").is_number_integer(), "Synthesis seed must be an integer");
    }
    const double seedValue = number(params, "seed", 1, 0, 4294967295.0);
    const auto seed = static_cast<std::uint32_t>(seedValue);
    const auto text = params.at("path").get<std::string>();
    require(!text.empty(), "Synthesis output path is required");
    const auto path = std::filesystem::path(std::u8string(text.begin(), text.end()));
    require(path.is_absolute() && (path.extension() == ".wav" || path.extension() == ".WAV"),
            "Synthesis requires an absolute WAV path");
    require(std::filesystem::is_directory(path.parent_path()), "Output directory does not exist");
    require(!std::filesystem::exists(path), "Synthesis output already exists; choose a new path");

    const auto frames = static_cast<std::uint32_t>(std::llround(duration * rate));
    std::vector<float> samples(frames);
    std::uint32_t random = seed;
    double filtered = 0, peak = 0;
    const double cutoff = 80 + brightness * 6000;
    const double alpha = 1 - std::exp(-2 * pi * cutoff / rate);
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        const double time = static_cast<double>(frame) / rate;
        random = random * 1664525U + 1013904223U;
        const double noise = static_cast<double>(random) / 4294967295.0 * 2 - 1;
        filtered += alpha * (noise - filtered);
        double value = 0;
        if (preset == "tone") {
            value = std::sin(2 * pi * frequency * time);
        } else if (preset == "wind") {
            value = filtered * (0.7 + 0.3 * std::sin(2 * pi * 0.7 * time + seed * 0.01));
        } else {
            const bool metal = preset == "metal";
            const double ratios[3] = {1, metal ? 2.756 : 1.47, metal ? 5.404 : 2.09};
            for (int mode = 0; mode < 3; ++mode) {
                value += std::sin(2 * pi * frequency * ratios[mode] * time) *
                    std::exp(-time * (1 + mode * (1 - brightness)) / decay) /
                    (1 + mode * 1.5);
            }
            value += filtered * 0.4 * std::exp(-time / 0.015);
        }
        // Short attack/release avoids endpoint clicks; generated files are not seamless loops.
        const double envelope = std::min(1.0, time / 0.003) *
            std::min(1.0, static_cast<double>(frames - 1 - frame) / (rate * 0.02));
        samples[frame] = static_cast<float>(value * envelope);
        peak = std::max(peak, std::abs(static_cast<double>(samples[frame])));
    }
    double sum = 0;
    for (auto& sample : samples) {
        sample = static_cast<float>(sample * (peak > 0 ? gain / peak : 0));
        sum += static_cast<double>(sample) * sample;
    }

    // Exclusive temporary file beside the destination, then no-replace publication.
    std::filesystem::path temporary;
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    require(GetTempFileNameW(path.parent_path().c_str(), L"rta", 0, buffer) != 0,
            "Cannot reserve synthesis temporary file");
    temporary = buffer;
#else
    throw std::runtime_error("Synthesis publication currently requires Windows");
#endif
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        require(output.good(), "Cannot open synthesis output");
        output.write("RIFF", 4);
        word(output, 48 + frames * 4, 4);
        output.write("WAVEfmt ", 8);
        word(output, 16, 4);
        word(output, 3, 2);
        word(output, 1, 2);
        word(output, rate, 4);
        word(output, rate * 4, 4);
        word(output, 4, 2);
        word(output, 32, 2);
        output.write("fact", 4);
        word(output, 4, 4);
        word(output, frames, 4);
        output.write("data", 4);
        word(output, frames * 4, 4);
        for (float sample : samples) {
            output.write(reinterpret_cast<const char*>(&sample), sizeof(sample));
        }
        output.flush();
        require(output.good(), "Synthesis WAV write failed");
        output.close();
#ifdef _WIN32
        require(MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_WRITE_THROUGH) != 0,
                "Cannot publish synthesis output; destination may already exist");
#endif
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
    return {{"path", text}, {"preset", preset}, {"seed", seed}, {"frames", frames},
        {"sample_rate", rate}, {"duration_seconds", static_cast<double>(frames) / rate},
        {"frequency_hz", frequency}, {"decay_seconds", decay}, {"brightness", brightness},
        {"gain", gain}, {"peak", gain}, {"rms", std::sqrt(sum / frames)}, {"version", 1}};
}
}
