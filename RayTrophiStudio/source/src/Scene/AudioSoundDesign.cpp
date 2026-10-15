#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioSoundDesign.h"
#include "Audio/AudioDesignDsp.h"
#include "Audio/AudioDesignJobs.h"
#include "Audio/AudioService.h"
#include "../../../external/miniaudio/miniaudio.h"
#include "../../../external/DaisySP/Source/Synthesis/oscillator.h"
#include "../../../external/DaisySP/Source/Control/adsr.h"
#include "../../../external/DaisySP/Source/Filters/onepole.h"
#include "../../../external/signalsmith-stretch/signalsmith-stretch.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace audio {
namespace {
using json = nlohmann::json;
constexpr unsigned rate = 48000;
constexpr double pi = 3.14159265358979323846;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

std::filesystem::path pathOf(const std::string& text) {
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
}

double control(const json& input, const char* key, double fallback, double low, double high) {
    if (input.contains(key)) {
        require(input.at(key).is_number(), "Sound design controls must be numeric");
    }
    const double value = input.value(key, fallback);
    require(std::isfinite(value) && value >= low && value <= high,
            "Sound design control is outside its finite documented range");
    return value;
}

json validate(const json& input) {
    require(input.is_object(), "recipe must be an object");
    const std::vector<std::string> common = {"mode", "gain", "lowpass_hz", "highpass_hz",
        "drive", "fade_in_seconds", "fade_out_seconds", "pitch_semitones", "stretch_ratio",
        "loop_crossfade_seconds", "layers", "effects", "gain_db", "reverse", "tail_seconds",
        "formant_semitones", "formant_base_hz", "formant_compensate"};
    const std::vector<std::string> synth = {"preset", "duration_seconds", "seed",
        "frequency_hz", "decay_seconds", "brightness", "wave", "noise_mix",
        "attack_seconds", "sustain", "release_seconds", "detune_cents",
        "modulation_hz", "modulation_depth"};
    const std::vector<std::string> asset = {"asset_id", "source_in_seconds", "source_out_seconds"};
    const auto mode = input.value("mode", std::string("synth"));
    require(mode == "synth" || mode == "asset", "mode must be synth or asset");
    for (auto field = input.begin(); field != input.end(); ++field) {
        const auto& extra = mode == "synth" ? synth : asset;
        require(std::find(common.begin(), common.end(), field.key()) != common.end() ||
                std::find(extra.begin(), extra.end(), field.key()) != extra.end(),
                "Unknown or inapplicable sound design recipe field");
    }
    json recipe = {{"mode", mode}};
    recipe["gain"] = control(input, "gain", mode == "synth" ? 0.5 : 1, 0, 2);
    if (input.contains("gain_db")) {
        const double gainDb = control(input, "gain_db", -6, -60, 6);
        const double gain = std::pow(10.0, gainDb / 20);
        require(!input.contains("gain") || std::abs(input.at("gain").get<double>() - gain) < 1e-6,
                "Choose gain or gain_db; conflicting output levels are rejected");
        recipe["gain"] = gain;
    }
    recipe["tail_seconds"] = control(input, "tail_seconds", 0, 0, 5);
    recipe["formant_semitones"] = control(input, "formant_semitones", 0, -12, 12);
    recipe["formant_base_hz"] = control(input, "formant_base_hz", 200, 20, 2000);
    require(!input.contains("formant_compensate") || input.at("formant_compensate").is_boolean(),
            "formant_compensate must be boolean");
    require(!input.contains("reverse") || input.at("reverse").is_boolean(), "reverse must be boolean");
    recipe["formant_compensate"] = input.value("formant_compensate", false);
    recipe["reverse"] = input.value("reverse", false);
    validateDesignDsp(input, recipe);
    recipe["lowpass_hz"] = control(input, "lowpass_hz", 20000, 20, 20000);
    recipe["highpass_hz"] = control(input, "highpass_hz", 20, 20, 20000);
    require(recipe.at("highpass_hz").get<double>() < recipe.at("lowpass_hz").get<double>(),
            "Highpass must be below lowpass");
    recipe["drive"] = control(input, "drive", 0, 0, 1);
    recipe["fade_in_seconds"] = control(input, "fade_in_seconds", 0.003, 0, 10);
    recipe["fade_out_seconds"] = control(input, "fade_out_seconds", 0.02, 0, 10);
    recipe["pitch_semitones"] = control(input, "pitch_semitones", 0, -24, 24);
    recipe["stretch_ratio"] = control(input, "stretch_ratio", 1, 0.5, 2);
    recipe["loop_crossfade_seconds"] = control(input, "loop_crossfade_seconds", 0, 0, 1);
    if (mode == "asset") {
        recipe["asset_id"] = input.at("asset_id").get<std::string>();
        recipe["source_in_seconds"] = control(input, "source_in_seconds", 0, 0, 600);
        if (input.contains("source_out_seconds")) {
            recipe["source_out_seconds"] = control(input, "source_out_seconds", 0, 0, 600);
        }
        return recipe;
    }
    const auto preset = input.value("preset", std::string("metal"));
    require(preset == "tone" || preset == "wind" || preset == "metal" || preset == "wood",
            "Unknown sound design preset");
    recipe["preset"] = preset;
    recipe["duration_seconds"] = control(input, "duration_seconds", 2, 0.05, 10);
    require(!input.contains("seed") || input.at("seed").is_number_integer(),
            "seed must be an integer");
    recipe["seed"] = static_cast<std::uint32_t>(control(input, "seed", 1, 0, 4294967295.0));
    const double frequency = preset == "wood" ? 180 : preset == "tone" ? 440 :
                             preset == "wind" ? 220 : 330;
    recipe["frequency_hz"] = control(input, "frequency_hz", frequency, 20, 4000);
    recipe["decay_seconds"] = control(input, "decay_seconds", preset == "wood" ? 0.15 : 0.8,
                                      0.01, 10);
    recipe["brightness"] = control(input, "brightness", 0.5, 0, 1);
    recipe["noise_mix"] = control(input, "noise_mix", preset == "wind" ? 1 : 0.15, 0, 1);
    recipe["attack_seconds"] = control(input, "attack_seconds", 0.003, 0.001, 5);
    recipe["sustain"] = control(input, "sustain", 0.7, 0, 1);
    recipe["release_seconds"] = control(input, "release_seconds", 0.1, 0.001, 5);
    recipe["detune_cents"] = control(input, "detune_cents", 0, -100, 100);
    recipe["modulation_hz"] = control(input, "modulation_hz", 0.7, 0.05, 20);
    recipe["modulation_depth"] = control(input, "modulation_depth", 0, 0, 1);
    recipe["wave"] = input.value("wave", std::string("sine"));
    require(recipe["wave"] == "sine" || recipe["wave"] == "triangle" ||
            recipe["wave"] == "saw" || recipe["wave"] == "square", "Unknown oscillator wave");
    return recipe;
}

AudioPcm synthesize(const json& recipe, std::vector<float>& envelope,
                    const AudioCancelCheck& check) {
    AudioPcm pcm;
    pcm.channels = 1;
    const auto frames = static_cast<std::size_t>(std::llround(
        recipe.at("duration_seconds").get<double>() * rate));
    pcm.samples.resize(frames);
    daisysp::Oscillator oscillator, overtone;
    oscillator.Init(rate);
    overtone.Init(rate);
    const auto wave = recipe.at("wave").get<std::string>();
    const auto kind = wave == "sine" ? daisysp::Oscillator::WAVE_SIN :
        wave == "triangle" ? daisysp::Oscillator::WAVE_POLYBLEP_TRI :
        wave == "saw" ? daisysp::Oscillator::WAVE_POLYBLEP_SAW :
                        daisysp::Oscillator::WAVE_POLYBLEP_SQUARE;
    oscillator.SetWaveform(kind);
    oscillator.SetAmp(1);
    oscillator.SetFreq(recipe.at("frequency_hz").get<float>());
    overtone.SetAmp(1);
    overtone.SetFreq(recipe.at("frequency_hz").get<float>() *
                    std::pow(2.0f, recipe.at("detune_cents").get<float>() / 1200));
    daisysp::Adsr adsr;
    adsr.Init(rate);
    adsr.SetAttackTime(recipe.at("attack_seconds"));
    adsr.SetDecayTime(recipe.at("decay_seconds"));
    adsr.SetSustainLevel(recipe.at("sustain"));
    adsr.SetReleaseTime(recipe.at("release_seconds"));
    daisysp::OnePole noiseFilter;
    noiseFilter.Init();
    noiseFilter.SetFrequency(static_cast<float>(
        (80 + recipe.at("brightness").get<double>() * 6000) / rate));
    const auto preset = recipe.at("preset").get<std::string>();
    const double frequency = recipe.at("frequency_hz"), decay = recipe.at("decay_seconds");
    const double brightness = recipe.at("brightness"), noiseMix = recipe.at("noise_mix");
    const double depth = recipe.at("modulation_depth"), modulationHz = recipe.at("modulation_hz");
    const double release = std::min(recipe.at("release_seconds").get<double>(),
                                   frames / static_cast<double>(rate) * 0.5);
    std::uint32_t random = recipe.at("seed");
    const std::size_t stride = std::max<std::size_t>(1, frames / 256);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        if (check && frame % 1024 == 0) {
            check();
        }
        const double time = frame / static_cast<double>(rate);
        random = random * 1664525U + 1013904223U;
        const float noise = noiseFilter.Process(static_cast<float>(
            static_cast<double>(random) / 4294967295.0 * 2 - 1));
        double value = 0;
        if (preset == "tone") {
            value = (oscillator.Process() + overtone.Process()) * 0.5 * (1 - noiseMix) +
                    noise * noiseMix;
        } else if (preset == "wind") {
            value = noise * noiseMix + oscillator.Process() * (1 - noiseMix);
        } else {
            const double ratios[4] = {1, preset == "metal" ? 2.756 : 1.47,
                preset == "metal" ? 5.404 : 2.09, preset == "metal" ? 8.93 : 3.32};
            for (int mode = 0; mode < 4; ++mode) {
                value += std::sin(2 * pi * frequency * ratios[mode] * time) *
                    std::exp(-time * (1 + mode * (1 - brightness)) / decay) / (1 + mode * 1.5);
            }
            value = value * (1 - noiseMix) + noise * noiseMix * std::exp(-time / 0.025);
        }
        const float shape = adsr.Process(time < frames / static_cast<double>(rate) - release);
        const double modulation = 1 - depth * 0.5 + depth * 0.5 *
            std::sin(2 * pi * modulationHz * time);
        pcm.samples[frame] = static_cast<float>(value * shape * modulation);
        if (frame % stride == 0 && envelope.size() < 256) {
            envelope.push_back(shape);
        }
    }
    float peak = 0;
    for (const auto sample : pcm.samples) {
        peak = std::max(peak, std::abs(sample));
    }
    for (auto& sample : pcm.samples) {
        sample *= peak > 0 ? 1 / peak : 0;
    }
    return pcm;
}

AudioPcm loadRange(json& recipe, const json& assets) {
    const auto id = recipe.at("asset_id").get<std::string>();
    auto found = std::find_if(assets.begin(), assets.end(), [&](const json& asset) {
        return asset.at("id") == id;
    });
    require(found != assets.end(), "Sound design asset not found");
    const double start = recipe.at("source_in_seconds");
    const double finish = recipe.value("source_out_seconds", std::min(
        found->at("duration").get<double>(), start + 10));
    require(finish > start && finish - start <= 30 && finish <=
            found->at("duration").get<double>() + 1e-6,
            "Select an existing source range of at most 30 seconds");
    const auto decoded = decodeAudioPcm(pathOf(found->at("path")));
    const auto first = static_cast<std::uint64_t>(std::llround(start * rate));
    const auto last = std::min(decoded.frames(),
        static_cast<std::uint64_t>(std::llround(finish * rate)));
    require(last > first && last - first >= rate / 20, "Source range must be at least 0.05 s");
    AudioPcm selected;
    selected.channels = decoded.channels;
    selected.samples.assign(decoded.samples.begin() + first * decoded.channels,
                            decoded.samples.begin() + last * decoded.channels);
    recipe["source_out_seconds"] = last / static_cast<double>(rate);
    return selected;
}

AudioPcm transform(const AudioPcm& original, const json& recipe, json& processing,
                   const AudioCancelCheck& check) {
    const double ratio = recipe.at("stretch_ratio"), pitch = recipe.at("pitch_semitones");
    AudioPcm result = original;
    if (recipe.at("reverse").get<bool>()) {
        for (std::size_t frame = 0; frame < result.frames() / 2; ++frame) {
            for (unsigned channel = 0; channel < result.channels; ++channel) {
                std::swap(result.samples[frame * result.channels + channel],
                    result.samples[(result.frames() - 1 - frame) * result.channels + channel]);
            }
        }
    }
    if (ratio != 1 || pitch != 0 || recipe.at("formant_semitones").get<double>() != 0 ||
        recipe.at("formant_compensate").get<bool>()) {
        const auto frames = static_cast<std::size_t>(std::llround(original.frames() * ratio));
        require(frames <= 30 * rate, "Stretched output must not exceed 30 seconds");
        std::vector<std::vector<float>> input(original.channels), output(original.channels);
        for (unsigned channel = 0; channel < original.channels; ++channel) {
            input[channel].resize(original.frames());
            output[channel].resize(frames);
            for (std::size_t frame = 0; frame < original.frames(); ++frame) {
                input[channel][frame] = result.samples[frame * original.channels + channel];
            }
        }
        // Local fixed seed ensures repeatable phase initialization for identical recipes.
        signalsmith::stretch::SignalsmithStretch<float> stretch(1);
        stretch.presetDefault(original.channels, rate);
        stretch.setTransposeSemitones(static_cast<float>(pitch));
        stretch.setFormantBase(recipe.at("formant_base_hz").get<float>() / rate);
        stretch.setFormantFactor(static_cast<float>(std::pow(2.0,
            recipe.at("formant_semitones").get<double>() / 12)),
            recipe.at("formant_compensate").get<bool>());
        require(stretch.exact(input, static_cast<int>(original.frames()), output,
                              static_cast<int>(frames)),
                "Source range is too short for independent pitch/time processing");
        if (check) {
            check();
        }
        result.samples.resize(frames * original.channels);
        for (unsigned channel = 0; channel < original.channels; ++channel) {
            for (std::size_t frame = 0; frame < frames; ++frame) {
                if (check && frame % 1024 == 0) {
                    check();
                }
                result.samples[frame * original.channels + channel] = output[channel][frame];
            }
        }
    }
    const auto tail = static_cast<std::size_t>(std::llround(
        recipe.at("tail_seconds").get<double>() * rate));
    require(result.frames() + tail <= 30 * rate, "Sound including effect tail exceeds 30 seconds");
    result.samples.resize(result.samples.size() + tail * result.channels, 0);
    applyDesignEffects(result, recipe.at("effects"), check);
    daisysp::OnePole low[2], high[2];
    for (unsigned channel = 0; channel < result.channels; ++channel) {
        low[channel].Init();
        high[channel].Init();
        low[channel].SetFrequency(recipe.at("lowpass_hz").get<float>() / rate);
        high[channel].SetFrequency(recipe.at("highpass_hz").get<float>() / rate);
        high[channel].SetFilterMode(daisysp::OnePole::FILTER_MODE_HIGH_PASS);
    }
    const double drive = recipe.at("drive"), gain = recipe.at("gain");
    const bool useHighpass = recipe.at("highpass_hz").get<double>() > 20;
    const bool useLowpass = recipe.at("lowpass_hz").get<double>() < 20000;
    const double fadeIn = recipe.at("fade_in_seconds").get<double>() * rate;
    const double fadeOut = recipe.at("fade_out_seconds").get<double>() * rate;
    std::uint64_t clamped = 0;
    for (std::size_t frame = 0; frame < result.frames(); ++frame) {
        if (check && frame % 1024 == 0) {
            check();
        }
        const double envelope = (fadeIn > 0 ? std::min(1.0, frame / fadeIn) : 1) *
            (fadeOut > 0 ? std::min(1.0, (result.frames() - 1 - frame) / fadeOut) : 1);
        for (unsigned channel = 0; channel < result.channels; ++channel) {
            auto& sample = result.samples[frame * result.channels + channel];
            // Extreme defaults mean bypass, preserving a neutral imported-asset recipe.
            if (useHighpass) {
                sample = high[channel].Process(sample);
            }
            if (useLowpass) {
                sample = low[channel].Process(sample);
            }
            if (drive > 0) {
                sample = static_cast<float>(std::tanh(sample * (1 + drive * 15)) /
                                             std::tanh(1 + drive * 15));
            }
            const double output = sample * gain * envelope;
            clamped += std::abs(output) > 1;
            sample = static_cast<float>(std::clamp(output, -1.0, 1.0));
            require(std::isfinite(sample), "Nonfinite sound design sample");
        }
    }
    const auto overlap = static_cast<std::uint64_t>(std::llround(
        recipe.at("loop_crossfade_seconds").get<double>() * rate));
    require(overlap * 2 < result.frames(), "Loop crossfade must be shorter than half the sound");
    if (overlap > 0) {
        const auto oldFrames = result.frames();
        AudioPcm loop;
        loop.channels = result.channels;
        loop.samples.resize((oldFrames - overlap) * result.channels);
        for (std::uint64_t frame = 0; frame < oldFrames - overlap; ++frame) {
            for (unsigned channel = 0; channel < result.channels; ++channel) {
                float sample = result.samples[(frame + overlap) * result.channels + channel];
                if (frame >= oldFrames - overlap * 2) {
                    const auto join = frame - (oldFrames - overlap * 2);
                    const float mix = static_cast<float>(join + 1) / (overlap + 1);
                    sample = sample * (1 - mix) +
                             result.samples[join * result.channels + channel] * mix;
                }
                loop.samples[frame * result.channels + channel] = sample;
            }
        }
        result = std::move(loop);
    }
    processing = {{"clamped_samples_before_loop_join", clamped},
        {"effects", recipe.at("effects").size()}, {"layers", recipe.at("layers").size()}};
    return result;
}

json summarizePcm(const AudioPcm& pcm) {
    return analyzeDesignPcm(pcm, {{"points", 64}, {"time_bins", 8}, {"continuous", false}});
}

std::string safeName(const json& params) {
    auto name = params.value("name", std::string("Sound"));
    require(!name.empty() && name.size() <= 120, "Sound name must contain 1..120 UTF-8 bytes");
    if (name.size() >= 4 && (name.substr(name.size() - 4) == ".wav" ||
                            name.substr(name.size() - 4) == ".WAV")) {
        name.resize(name.size() - 4);
    }
    require(!name.empty() && name.front() != '.' && name.back() != '.' && name.back() != ' ',
            "Sound name cannot start with a dot or end with a dot/space");
    for (const unsigned char letter : name) {
        require(letter >= 32 && std::string("<>:\"/\\|?*").find(letter) == std::string::npos,
                "Sound name must be a filename, without path separators or reserved characters");
    }
    std::string stem = name.substr(0, name.find('.'));
    for (auto& letter : stem) {
        if (letter >= 'a' && letter <= 'z') {
            letter = static_cast<char>(letter - 'a' + 'A');
        }
    }
    require(stem != "CON" && stem != "PRN" && stem != "AUX" && stem != "NUL" &&
            !(stem.size() == 4 && (stem.substr(0, 3) == "COM" || stem.substr(0, 3) == "LPT") &&
              stem[3] >= '0' && stem[3] <= '9'), "Reserved Windows sound name");
    return name;
}

void word(std::ofstream& stream, std::uint32_t value, unsigned bytes) {
    for (unsigned byte = 0; byte < bytes; ++byte) {
        stream.put(static_cast<char>((value >> (byte * 8)) & 255));
    }
}

bool publish(const AudioPcm& pcm, const std::filesystem::path& path) {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    require(GetTempFileNameW(path.parent_path().c_str(), L"rtd", 0, buffer) != 0,
            "Cannot reserve sound design temporary file");
    const std::filesystem::path temporary(buffer);
    try {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        require(stream.good(), "Cannot write designed sound");
        const auto frames = static_cast<std::uint32_t>(pcm.frames());
        const auto size = static_cast<std::uint32_t>(pcm.samples.size() * sizeof(float));
        stream.write("RIFF", 4);
        word(stream, 48 + size, 4);
        stream.write("WAVEfmt ", 8);
        word(stream, 16, 4);
        word(stream, 3, 2);
        word(stream, pcm.channels, 2);
        word(stream, rate, 4);
        word(stream, rate * pcm.channels * 4, 4);
        word(stream, pcm.channels * 4, 2);
        word(stream, 32, 2);
        stream.write("fact", 4);
        word(stream, 4, 4);
        word(stream, frames, 4);
        stream.write("data", 4);
        word(stream, size, 4);
        stream.write(reinterpret_cast<const char*>(pcm.samples.data()), size);
        stream.flush();
        require(stream.good(), "Designed sound write failed");
        stream.close();
        if (MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_WRITE_THROUGH) == 0) {
            const auto error = GetLastError();
            if (error == ERROR_ALREADY_EXISTS || error == ERROR_FILE_EXISTS) {
                std::filesystem::remove(temporary);
                return false;
            }
            throw std::runtime_error("Cannot publish designed sound: " + std::to_string(error));
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
    return true;
#else
    throw std::runtime_error("Sound design publication currently requires Windows");
#endif
}
} // namespace

namespace {
json designSourceVersions(const json& recipe, const json& assets) {
    std::set<std::string> ids;
    if (recipe.at("mode") == "asset") {
        ids.insert(recipe.at("asset_id").get<std::string>());
    }
    for (const auto& layer : recipe.at("layers")) {
        if (layer.at("type") == "asset" && layer.at("enabled").get<bool>()) {
            ids.insert(layer.at("asset_id").get<std::string>());
        }
    }
    json versions = json::object();
    for (const auto& id : ids) {
        const auto asset = std::find_if(assets.begin(), assets.end(), [&](const json& entry) {
            return entry.at("id") == id;
        });
        require(asset != assets.end(), "Background design source asset is unavailable");
        const auto path = pathOf(asset->at("path").get<std::string>());
        versions[id] = {{"bytes", std::filesystem::file_size(path)},
            {"modified", std::filesystem::last_write_time(path).time_since_epoch().count()}};
    }
    return versions;
}
} // namespace

struct AudioSoundDesign::Impl {
    AudioPcm original, processed;
    AudioDesignJobs jobs;
    json report = {{"ready", false}};
    std::uint64_t revision = 0;
    std::shared_ptr<AudioPcm> sourceCache;
    AudioDesignJobs sourceJobs;
    std::string sourceRequest;
    json sourceWaveform;
    std::filesystem::path sourcePath;
    std::filesystem::file_time_type sourceModified;
    std::uintmax_t sourceBytes = 0;
    std::deque<json> undoRecipes, redoRecipes;
    bool applyingHistory = false;
    std::uint64_t historyJob = 0;
    bool historyRedo = false;
    ma_audio_buffer buffer{};
    ma_sound sound{};
    bool bufferReady = false, soundReady = false, auditionOriginal = false;

    void stop() {
        if (soundReady) {
            ma_sound_uninit(&sound);
            soundReady = false;
        }
        if (bufferReady) {
            ma_audio_buffer_uninit(&buffer);
            bufferReady = false;
        }
    }
};

AudioSoundDesign::AudioSoundDesign() : impl_(std::make_unique<Impl>()) {}
AudioSoundDesign::~AudioSoundDesign() {
    impl_->jobs.cancel();
    stop();
}

json validateSoundDesignRecipe(const json& input) {
    return validate(input);
}

AudioDesignResult prepareSoundDesign(const json& input, const json& assets,
                                    const AudioCancelCheck& check) {
    const auto checkpoint = [&] {
        if (check) {
            check();
        }
    };
    checkpoint();
    auto recipe = validate(input);
    std::vector<float> envelope;
    auto original = recipe.at("mode") == "synth" ?
        (recipe.at("layers").empty() ? synthesize(recipe, envelope, check) :
                                      renderDesignLayers(recipe, assets, envelope, check)) :
        loadRange(recipe, assets);
    checkpoint();
    json processing;
    auto processed = transform(original, recipe, processing, check);
    checkpoint();
    const auto processedReport = summarizePcm(processed);
    const auto originalReport = summarizePcm(original);
    json report = {{"ready", true}, {"version", 3}, {"recipe", recipe},
        {"processed", processedReport}, {"original", originalReport}, {"envelope", envelope},
        {"engine", "DaisySP + Signalsmith Stretch"}, {"processing", processing}};
    report["origin"] = json::object();
    if (recipe.at("mode") == "asset") {
        for (const auto& asset : assets) {
            if (asset.at("id") == recipe.at("asset_id")) {
                report["origin"] = asset;
                break;
            }
        }
    }
    json response = json::array();
    for (unsigned bin = 0; bin < 64; ++bin) {
        const double frequency = 20 * std::pow(1000.0, bin / 63.0);
        const double warped = std::tan(pi * frequency / rate);
        double magnitude = 1;
        const double low = recipe.at("lowpass_hz"), high = recipe.at("highpass_hz");
        if (low < 20000) {
            const double cutoff = std::tan(pi * low / rate);
            magnitude *= cutoff / std::sqrt(cutoff * cutoff + warped * warped);
        }
        if (high > 20) {
            const double cutoff = std::tan(pi * high / rate);
            magnitude *= warped / std::sqrt(cutoff * cutoff + warped * warped);
        }
        response.push_back({frequency, 20 * std::log10(std::max(1e-6, magnitude))});
    }
    report["filter_response"] = response;
    report["eq_response"] = designEqResponse(recipe.at("effects"));
    report["source_assets"] = json::array();
    for (const auto& layer : recipe.at("layers")) {
        if (layer.at("type") == "asset") {
            for (const auto& asset : assets) {
                if (asset.at("id").get<std::string>() == layer.at("asset_id").get<std::string>()) {
                    report["source_assets"].push_back(asset);
                    break;
                }
            }
        }
    }
    checkpoint();
    return {std::move(original), std::move(processed), std::move(report)};
}

json AudioSoundDesign::render(const json& input, const json& assets) {
    auto prepared = prepareSoundDesign(input, assets);
    impl_->jobs.cancel();
    return commit(std::move(prepared));
}

json AudioSoundDesign::commit(AudioDesignResult prepared) {
    const auto& recipe = prepared.report.at("recipe");
    // Store recipes, not PCM banks. Invalid requests do not touch history or audition.
    if (!impl_->applyingHistory && impl_->report.at("ready").get<bool>() &&
        impl_->report.at("recipe") != recipe) {
        impl_->undoRecipes.push_back(impl_->report.at("recipe"));
        if (impl_->undoRecipes.size() > 32) {
            impl_->undoRecipes.pop_front();
        }
        impl_->redoRecipes.clear();
    }
    // Validation, decode and rendering succeed before replacing the previous audition.
    stop();
    impl_->original = std::move(prepared.original);
    impl_->processed = std::move(prepared.processed);
    impl_->report = std::move(prepared.report);
    ++impl_->revision;
    impl_->auditionOriginal = false;
    return status();
}

json AudioSoundDesign::jobStart(const json& params, const json& assets) {
    const auto kind = params.at("kind").get<std::string>();
    require(kind == "render" || kind == "analysis", "Job kind must be render or analysis");
    if (kind == "render") {
        require(!params.contains("options"), "Render jobs do not accept analysis options");
        const auto recipe = validateSoundDesignRecipe(params.at("recipe"));
        const auto files = designSourceVersions(recipe, assets);
        impl_->jobs.start(kind, impl_->revision,
            [recipe, assets, files](const AudioCancelCheck& check) {
                check();
                require(designSourceVersions(recipe, assets) == files,
                        "Background design source file changed before rendering");
                AudioDesignJobs::Result result;
                result.assets = assets;
                result.files = files;
                result.design = prepareSoundDesign(recipe, assets, check);
                require(designSourceVersions(recipe, assets) == files,
                        "Background design source file changed during rendering");
                return result;
            });
    } else {
        require(!params.contains("recipe"), "Analysis jobs do not accept a recipe");
        require(impl_->report.at("ready").get<bool>(), "Render a sound before analysis");
        auto options = params.value("options", json::object());
        require(options.is_object(), "Analysis options must be an object");
        require(!options.contains("original") || options.at("original").is_boolean(),
                "original must be boolean");
        const bool original = options.value("original", false);
        options.erase("original");
        const auto& source = original ? impl_->original : impl_->processed;
        // Reject malformed requests before copying PCM or replacing an existing task.
        validateDesignAnalysis(source, options);
        auto pcm = source;
        impl_->jobs.start(kind, impl_->revision,
            [pcm = std::move(pcm), options, original, assets](const AudioCancelCheck& check) {
                AudioDesignJobs::Result result;
                result.assets = assets;
                result.analysis = analyzeDesignPcm(pcm, options, check);
                result.analysis["original"] = original;
                return result;
            });
    }
    return impl_->jobs.status();
}

json AudioSoundDesign::jobStatus() const {
    return impl_->jobs.status();
}

json AudioSoundDesign::jobCancel() {
    impl_->jobs.cancel();
    impl_->historyJob = 0;
    return impl_->jobs.status();
}

json AudioSoundDesign::jobApply(const json& assets) {
    const auto report = impl_->jobs.status();
    auto result = impl_->jobs.take();
    if (!result) {
        return report;
    }
    if (result->revision != impl_->revision || result->assets != assets) {
        return {{"id", result->id}, {"state", "stale"}};
    }
    if (!result->analysis.is_null()) {
        result->analysis["revision"] = impl_->revision;
        return {{"id", result->id}, {"state", "applied"}, {"analysis", result->analysis}};
    }
    try {
        if (designSourceVersions(result->design.report.at("recipe"), assets) != result->files) {
            return {{"id", result->id}, {"state", "stale"}};
        }
    } catch (const std::filesystem::filesystem_error&) {
        return {{"id", result->id}, {"state", "stale"}};
    }
    const bool history = result->id == impl_->historyJob;
    const bool redo = impl_->historyRedo;
    if (history) {
        const auto& from = redo ? impl_->redoRecipes : impl_->undoRecipes;
        if (from.empty() || from.back() != result->design.report.at("recipe")) {
            impl_->historyJob = 0;
            return {{"id", result->id}, {"state", "stale"}};
        }
    }
    const auto current = history ? impl_->report.at("recipe") : json();
    impl_->applyingHistory = history;
    json design;
    try {
        design = commit(std::move(result->design));
    } catch (...) {
        impl_->applyingHistory = false;
        throw;
    }
    impl_->applyingHistory = false;
    if (history) {
        auto& from = redo ? impl_->redoRecipes : impl_->undoRecipes;
        auto& to = redo ? impl_->undoRecipes : impl_->redoRecipes;
        from.pop_back();
        to.push_back(current);
        impl_->historyJob = 0;
        design = status();
    }
    return {{"id", result->id}, {"state", "applied"}, {"design", design}};
}

bool AudioSoundDesign::playing() const {
    return impl_->soundReady && ma_sound_is_playing(&impl_->sound);
}

json AudioSoundDesign::status(bool details) const {
    auto report = details ? impl_->report : json{{"ready", impl_->report.at("ready")}};
    report["revision"] = impl_->revision;
    report["undo_available"] = !impl_->undoRecipes.empty();
    report["redo_available"] = !impl_->redoRecipes.empty();
    report["playing"] = playing();
    report["audition_original"] = impl_->auditionOriginal;
    report["loop"] = impl_->soundReady && ma_sound_is_looping(&impl_->sound);
    ma_uint64 cursor = 0;
    if (impl_->soundReady) {
        ma_sound_get_cursor_in_pcm_frames(&impl_->sound, &cursor);
    }
    report["cursor_seconds"] = cursor / static_cast<double>(rate);
    return report;
}

json AudioSoundDesign::analyze(const json& params) const {
    require(impl_->report.at("ready").get<bool>(), "Render a sound before analysis");
    require(!params.contains("original") || params.at("original").is_boolean(),
            "original must be boolean");
    auto input = params;
    input.erase("original");
    const bool original = params.value("original", false);
    auto result = analyzeDesignPcm(original ? impl_->original : impl_->processed, input);
    result["revision"] = impl_->revision;
    result["original"] = original;
    return result;
}

json AudioSoundDesign::undo(const json& assets, bool redo, bool background) {
    auto& from = redo ? impl_->redoRecipes : impl_->undoRecipes;
    auto& to = redo ? impl_->undoRecipes : impl_->redoRecipes;
    require(!from.empty(), redo ? "No designer redo" : "No designer undo");
    const auto candidate = from.back(), current = impl_->report.at("recipe");
    if (background) {
        auto job = jobStart({{"kind", "render"}, {"recipe", candidate}}, assets);
        impl_->historyJob = job.at("id");
        impl_->historyRedo = redo;
        job["recipe"] = candidate;
        return job;
    }
    impl_->applyingHistory = true;
    try {
        render(candidate, assets);
    } catch (...) {
        impl_->applyingHistory = false;
        throw;
    }
    impl_->applyingHistory = false;
    from.pop_back();
    to.push_back(current);
    return status();
}

json AudioSoundDesign::waveform(const json& params, const json& assets) {
    require(!params.contains("background") || params.at("background").is_boolean(),
            "background must be boolean");
    const auto id = params.at("id").get<std::string>();
    const auto found = std::find_if(assets.begin(), assets.end(), [&](const json& asset) {
        return asset.at("id").get<std::string>() == id;
    });
    require(found != assets.end(), "Source waveform asset not found");
    const double begin = control(params, "start_seconds", 0, 0, 600);
    const double end = control(params, "end_seconds", found->at("duration"), 0, 600);
    require(end > begin && end <= found->at("duration").get<double>() + 1e-6,
            "Source waveform range is outside the asset");
    const auto pointsValue = params.value("points", json(512));
    require(pointsValue.is_number_integer() && pointsValue.get<double>() >= 16 &&
            pointsValue.get<double>() <= 4096, "Waveform points must be integer 16..4096");
    const auto path = pathOf(found->at("path"));
    const auto modified = std::filesystem::last_write_time(path);
    const auto bytes = std::filesystem::file_size(path);
    const bool cached = impl_->sourceCache && impl_->sourcePath == path &&
        impl_->sourceModified == modified && impl_->sourceBytes == bytes;
    if (params.value("background", false)) {
        const auto key = json{{"params", params}, {"asset", *found}, {"bytes", bytes},
            {"modified", modified.time_since_epoch().count()}}.dump();
        if (impl_->sourceRequest != key) {
            const auto cache = cached ? impl_->sourceCache : nullptr;
            const auto points = pointsValue.get<unsigned>();
            impl_->sourceJobs.start("waveform", impl_->revision,
                [path, modified, bytes, cache, begin, end, points, id]
                (const AudioCancelCheck& check) {
                    check();
                    const auto unchanged = [&] {
                        return std::filesystem::file_size(path) == bytes &&
                            std::filesystem::last_write_time(path) == modified;
                    };
                    require(unchanged(), "Source waveform file changed before decode");
                    AudioDesignJobs::Result result;
                    if (!cache) {
                        result.design.original = decodeAudioPcm(path);
                    }
                    check();
                    const auto& pcm = cache ? *cache : result.design.original;
                    const auto first = static_cast<std::uint64_t>(std::llround(begin * rate));
                    const auto last = std::min(pcm.frames(),
                        static_cast<std::uint64_t>(std::llround(end * rate)));
                    result.analysis = designPcmWaveform(pcm, first, last, points, check);
                    result.analysis["asset_id"] = id;
                    check();
                    require(unchanged(), "Source waveform file changed during decode");
                    return result;
                });
            impl_->sourceRequest = key;
            impl_->sourceWaveform = nullptr;
        }
        if (auto result = impl_->sourceJobs.take()) {
            // The request key includes asset identity, range and current file metadata.
            impl_->sourceWaveform = std::move(result->analysis);
            if (!cached) {
                impl_->sourceCache = std::make_shared<AudioPcm>(
                    std::move(result->design.original));
                impl_->sourcePath = path;
                impl_->sourceModified = modified;
                impl_->sourceBytes = bytes;
            }
        }
        if (!impl_->sourceWaveform.is_null()) {
            auto result = impl_->sourceWaveform;
            result["state"] = "ready";
            return result;
        }
        return impl_->sourceJobs.status();
    }
    impl_->sourceJobs.cancel();
    impl_->sourceRequest.clear();
    impl_->sourceWaveform = nullptr;
    if (!cached) {
        auto decoded = decodeAudioPcm(path);
        impl_->sourceCache = std::make_shared<AudioPcm>(std::move(decoded));
        impl_->sourcePath = path;
        impl_->sourceModified = modified;
        impl_->sourceBytes = bytes;
    }
    const auto first = static_cast<std::uint64_t>(std::llround(begin * rate));
    const auto last = std::min(impl_->sourceCache->frames(),
        static_cast<std::uint64_t>(std::llround(end * rate)));
    auto result = designPcmWaveform(*impl_->sourceCache, first, last, pointsValue.get<unsigned>());
    result["asset_id"] = id;
    return result;
}

json AudioSoundDesign::preview(ma_engine& engine, ma_sound_group* group, const json& params) {
    require(impl_->report.at("ready"), "Render a sound before preview");
    require(!params.contains("original") || params.at("original").is_boolean(),
            "original must be boolean");
    require(!params.contains("loop") || params.at("loop").is_boolean(), "loop must be boolean");
    const bool original = params.value("original", false), loop = params.value("loop", false);
    stop();
    const auto& pcm = original ? impl_->original : impl_->processed;
    auto config = ma_audio_buffer_config_init(ma_format_f32, pcm.channels, pcm.frames(),
                                              pcm.samples.data(), nullptr);
    config.sampleRate = rate;
    require(ma_audio_buffer_init(&config, &impl_->buffer) == MA_SUCCESS,
            "Cannot initialize preview PCM");
    impl_->bufferReady = true;
    if (ma_sound_init_from_data_source(&engine, &impl_->buffer, MA_SOUND_FLAG_NO_SPATIALIZATION,
                                       group, &impl_->sound) != MA_SUCCESS) {
        stop();
        throw std::runtime_error("Cannot initialize sound design voice");
    }
    impl_->soundReady = true;
    impl_->auditionOriginal = original;
    ma_sound_set_looping(&impl_->sound, loop);
    if (ma_sound_start(&impl_->sound) != MA_SUCCESS) {
        stop();
        throw std::runtime_error("Cannot start sound design preview");
    }
    return status();
}

void AudioSoundDesign::stop() {
    impl_->stop();
}

void AudioSoundDesign::clear() {
    impl_->jobs.cancel();
    impl_->historyJob = 0;
    impl_->sourceJobs.cancel();
    impl_->sourceRequest.clear();
    impl_->sourceWaveform = nullptr;
    stop();
    impl_->original = {};
    impl_->processed = {};
    impl_->report = {{"ready", false}};
    impl_->sourceCache = {};
    impl_->sourcePath.clear();
    impl_->undoRecipes.clear();
    impl_->redoRecipes.clear();
    ++impl_->revision;
}

json soundDesignDestination(const json& params) {
    const auto folder = pathOf(params.at("folder").get<std::string>()).lexically_normal();
    require(folder.is_absolute(), "Sound output folder must be absolute");
    return {{"folder", pathUtf8(folder)}, {"format", "wav"}, {"auto_suffix", true}};
}

json AudioSoundDesign::save(const json& params) const {
    require(impl_->report.at("ready"), "Render a sound before saving");
    const auto name = safeName(params);
    std::filesystem::path target;
    bool explicitPath = params.contains("path");
    if (explicitPath) {
        require(!params.contains("folder"), "Choose path or folder, not both");
        target = pathOf(params.at("path").get<std::string>());
        require(target.is_absolute() && (target.extension() == ".wav" ||
                                         target.extension() == ".WAV"),
                "Explicit output must be an absolute WAV path");
        require(std::filesystem::is_directory(target.parent_path()),
                "Explicit output parent must exist");
        require(!std::filesystem::exists(target), "Existing output is preserved");
    } else {
        const auto folder = pathOf(soundDesignDestination(params).at("folder"));
        std::filesystem::create_directories(folder);
        require(std::filesystem::is_directory(folder), "Sound output folder unavailable");
        for (unsigned suffix = 0; suffix < 10000; ++suffix) {
            const auto stem = name + (suffix ? "_" + std::to_string(suffix + 1) : "");
            target = folder / pathOf(stem + ".wav");
            if (std::filesystem::exists(target)) {
                continue;
            }
            if (publish(impl_->processed, target)) {
                return {{"path", pathUtf8(target)}, {"name", name},
                    {"recipe", impl_->report.at("recipe")},
                    {"origin", impl_->report.at("origin")},
                    {"source_assets", impl_->report.at("source_assets")},
                    {"pcm_hash", impl_->report.at("processed").at("pcm_hash")}, {"version", 3}};
            }
        }
        throw std::runtime_error("No available sound filename");
    }
    require(publish(impl_->processed, target), "Existing output is preserved");
    return {{"path", pathUtf8(target)}, {"name", name}, {"recipe", impl_->report.at("recipe")},
        {"origin", impl_->report.at("origin")},
        {"source_assets", impl_->report.at("source_assets")},
        {"pcm_hash", impl_->report.at("processed").at("pcm_hash")}, {"version", 3}};
}
} // namespace audio
