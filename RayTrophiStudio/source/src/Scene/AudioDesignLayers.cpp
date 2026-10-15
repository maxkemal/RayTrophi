#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioDesignDsp.h"
#include "../../../external/DaisySP/Source/Synthesis/oscillator.h"
#include "../../../external/DaisySP/Source/Synthesis/fm2.h"
#include "../../../external/DaisySP/Source/PhysicalModeling/resonator.h"
#include "../../../external/DaisySP/Source/Control/adsr.h"
#include "../../../external/DaisySP/Source/Filters/onepole.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace audio {
AudioPcm renderDesignLayers(const nlohmann::json& recipe, const nlohmann::json& assets,
                            std::vector<float>& envelope, const AudioCancelCheck& check) {
    using json = nlohmann::json;
    constexpr unsigned rate = 48000;
    constexpr double pi = 3.14159265358979323846;
    const auto require = [](bool condition, const char* message) {
        if (!condition) {
            throw std::invalid_argument(message);
        }
    };
    const auto frames = static_cast<std::size_t>(std::llround(
        recipe.at("duration_seconds").get<double>() * rate));
    AudioPcm result;
    result.channels = 2;
    result.samples.resize(frames * 2, 0);
    unsigned index = 0;
    for (const auto& layer : recipe.at("layers")) {
        if (check) {
            check();
        }
        ++index;
        if (!layer.at("enabled").get<bool>()) {
            continue;
        }
        const auto start = static_cast<std::size_t>(std::llround(
            layer.at("start_seconds").get<double>() * rate));
        require(start < frames, "Enabled layer start must precede the sound end");
        const auto type = layer.at("type").get<std::string>();
        const double pan = layer.at("pan");
        const double gain = std::pow(10.0, layer.at("gain_db").get<double>() / 20);
        const double left = std::cos((pan + 1) * pi / 4) * gain;
        const double right = std::sin((pan + 1) * pi / 4) * gain;
        daisysp::Oscillator oscillator;
        daisysp::Fm2 fm;
        daisysp::Resonator resonator;
        daisysp::OnePole noiseFilter;
        std::uint32_t random = recipe.at("seed").get<std::uint32_t>() ^ (index * 2654435761U);
        float brown = 0, pink[3] = {};
        AudioPcm sample;
        std::uint64_t first = 0, last = 0;
        if (type == "oscillator") {
            oscillator.Init(rate);
            oscillator.SetAmp(1);
            oscillator.SetFreq(static_cast<float>(layer.at("frequency_hz").get<double>() *
                std::pow(2.0, layer.at("detune_cents").get<double>() / 1200)));
            oscillator.SetPw(layer.at("pulse_width"));
            const auto wave = layer.at("wave").get<std::string>();
            oscillator.SetWaveform(wave == "sine" ? daisysp::Oscillator::WAVE_SIN :
                wave == "triangle" ? daisysp::Oscillator::WAVE_POLYBLEP_TRI :
                wave == "saw" ? daisysp::Oscillator::WAVE_POLYBLEP_SAW :
                                daisysp::Oscillator::WAVE_POLYBLEP_SQUARE);
        } else if (type == "fm") {
            fm.Init(rate);
            fm.SetFrequency(layer.at("frequency_hz"));
            fm.SetRatio(layer.at("ratio"));
            fm.SetIndex(layer.at("index"));
        } else if (type == "resonator") {
            resonator.Init(layer.at("position"), 24, rate);
            resonator.SetFreq(layer.at("frequency_hz"));
            resonator.SetStructure(layer.at("structure"));
            resonator.SetBrightness(layer.at("brightness"));
            resonator.SetDamping(layer.at("damping"));
        } else if (type == "noise") {
            noiseFilter.Init();
            noiseFilter.SetFrequency(layer.at("cutoff_hz").get<float>() / rate);
        } else {
            const auto assetId = layer.at("asset_id").get<std::string>();
            const auto found = std::find_if(assets.begin(), assets.end(), [&](const json& asset) {
                return asset.at("id").get<std::string>() == assetId;
            });
            require(found != assets.end(), "Sample layer asset not found");
            const auto text = found->at("path").get<std::string>();
            sample = decodeAudioPcm(std::filesystem::path(std::u8string(text.begin(), text.end())));
            first = static_cast<std::uint64_t>(std::llround(
                layer.at("source_in_seconds").get<double>() * rate));
            last = static_cast<std::uint64_t>(std::llround(
                layer.at("source_out_seconds").get<double>() * rate));
            require(last <= sample.frames() && last >= first + rate / 20,
                    "Sample layer must select an existing range of at least 0.05 seconds");
        }
        const auto noiseColor = layer.value("color", std::string("white"));
        const double burst = layer.value("burst_seconds", 0.005) * rate;
        for (std::size_t frame = start; frame < frames; ++frame) {
            if (check && frame % 1024 == 0) {
                check();
            }
            const auto local = frame - start;
            const double sampleEdge = type == "asset" && first + local < last ?
                std::min(1.0, (last - first - 1 - local) / 480.0) : 1;
            const double edge = std::min(1.0, local / 144.0) *
                                std::min(1.0, (frames - 1 - frame) / 480.0) * sampleEdge;
            float value = 0;
            if (type == "oscillator") {
                value = oscillator.Process();
            } else if (type == "fm") {
                value = fm.Process();
            } else if (type == "noise" || type == "resonator") {
                random = random * 1664525U + 1013904223U;
                const float noise = static_cast<float>(double(random) / 4294967295.0 * 2 - 1);
                if (type == "resonator") {
                    const float excitation = local < burst ?
                        noise * static_cast<float>(1 - local / burst) : 0;
                    value = resonator.Process(excitation);
                } else {
                    brown = std::clamp(brown * 0.995f + noise * 0.05f, -1.0f, 1.0f);
                    pink[0] = pink[0] * 0.997f + noise * 0.03f;
                    pink[1] = pink[1] * 0.985f + noise * 0.05f;
                    pink[2] = pink[2] * 0.95f + noise * 0.1f;
                    const float colored = noiseColor == "brown" ? brown : noiseColor == "pink" ?
                        (pink[0] + pink[1] + pink[2] + noise * 0.15f) * 0.6f : noise;
                    value = noiseFilter.Process(colored);
                }
            } else if (first + local < last) {
                const auto source = (first + local) * sample.channels;
                if (sample.channels == 2) {
                    result.samples[frame * 2] += static_cast<float>(sample.samples[source] * left * edge);
                    result.samples[frame * 2 + 1] += static_cast<float>(sample.samples[source + 1] * right * edge);
                    continue;
                }
                value = sample.samples[source];
            }
            // Every layer has short edges, including a sample truncated by the global duration.
            result.samples[frame * 2] += static_cast<float>(value * left * edge);
            result.samples[frame * 2 + 1] += static_cast<float>(value * right * edge);
        }
    }
    daisysp::Adsr adsr;
    adsr.Init(rate);
    adsr.SetAttackTime(recipe.at("attack_seconds"));
    adsr.SetDecayTime(recipe.at("decay_seconds"));
    adsr.SetSustainLevel(recipe.at("sustain"));
    adsr.SetReleaseTime(recipe.at("release_seconds"));
    const double release = std::min(recipe.at("release_seconds").get<double>(),
                                   double(frames) / rate * 0.5);
    const double depth = recipe.at("modulation_depth"), modulationHz = recipe.at("modulation_hz");
    const auto stride = std::max<std::size_t>(1, frames / 256);
    float peak = 0;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const double time = double(frame) / rate;
        const float shape = adsr.Process(time < double(frames) / rate - release);
        const double modulation = 1 - depth * 0.5 + depth * 0.5 * std::sin(2 * pi * modulationHz * time);
        for (unsigned channel = 0; channel < 2; ++channel) {
            auto& sample = result.samples[frame * 2 + channel];
            sample = static_cast<float>(sample * shape * modulation);
            require(std::isfinite(sample), "Nonfinite layered synthesis output");
            peak = std::max(peak, std::abs(sample));
        }
        if (frame % stride == 0 && envelope.size() < 256) {
            envelope.push_back(shape);
        }
    }
    // Preserve layer level relationships; normalize only when the composite would overload.
    if (peak > 1) {
        for (auto& sample : result.samples) {
            sample /= peak;
        }
    }
    return result;
}
} // namespace audio
