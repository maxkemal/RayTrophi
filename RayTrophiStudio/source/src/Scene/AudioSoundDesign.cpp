#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioSoundDesign.h"
#include "Audio/AudioService.h"
#include "../../../external/miniaudio/miniaudio.h"
#include "../../../external/DaisySP/Source/Synthesis/oscillator.h"
#include "../../../external/DaisySP/Source/Control/adsr.h"
#include "../../../external/DaisySP/Source/Filters/onepole.h"
#include "../../../external/signalsmith-stretch/signalsmith-stretch.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
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
        "loop_crossfade_seconds"};
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

AudioPcm synthesize(const json& recipe, std::vector<float>& envelope) {
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

AudioPcm transform(const AudioPcm& original, const json& recipe) {
    const double ratio = recipe.at("stretch_ratio"), pitch = recipe.at("pitch_semitones");
    AudioPcm result = original;
    if (ratio != 1 || pitch != 0) {
        const auto frames = static_cast<std::size_t>(std::llround(original.frames() * ratio));
        require(frames <= 30 * rate, "Stretched output must not exceed 30 seconds");
        std::vector<std::vector<float>> input(original.channels), output(original.channels);
        for (unsigned channel = 0; channel < original.channels; ++channel) {
            input[channel].resize(original.frames());
            output[channel].resize(frames);
            for (std::size_t frame = 0; frame < original.frames(); ++frame) {
                input[channel][frame] = original.samples[frame * original.channels + channel];
            }
        }
        // Local fixed seed ensures repeatable phase initialization for identical recipes.
        signalsmith::stretch::SignalsmithStretch<float> stretch(1);
        stretch.presetDefault(original.channels, rate);
        stretch.setTransposeSemitones(static_cast<float>(pitch));
        require(stretch.exact(input, static_cast<int>(original.frames()), output,
                              static_cast<int>(frames)),
                "Source range is too short for independent pitch/time processing");
        result.samples.resize(frames * original.channels);
        for (unsigned channel = 0; channel < original.channels; ++channel) {
            for (std::size_t frame = 0; frame < frames; ++frame) {
                result.samples[frame * original.channels + channel] = output[channel][frame];
            }
        }
    }
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
    for (std::size_t frame = 0; frame < result.frames(); ++frame) {
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
            sample = static_cast<float>(std::clamp(sample * gain * envelope, -1.0, 1.0));
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
    return result;
}

json analyze(const AudioPcm& pcm) {
    float peak = 0;
    double square = 0;
    std::uint64_t hash = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(pcm.samples.data());
    for (std::size_t byte = 0; byte < pcm.samples.size() * sizeof(float); ++byte) {
        hash = (hash ^ bytes[byte]) * 1099511628211ULL;
    }
    for (const float sample : pcm.samples) {
        require(std::isfinite(sample), "Nonfinite sound design PCM");
        peak = std::max(peak, std::abs(sample));
        square += static_cast<double>(sample) * sample;
    }
    json waveform = json::array();
    const auto bins = std::min<std::uint64_t>(256, pcm.frames());
    for (std::uint64_t bin = 0; bin < bins; ++bin) {
        float low = std::numeric_limits<float>::max();
        float high = -std::numeric_limits<float>::max();
        for (auto frame = pcm.frames() * bin / bins;
             frame < pcm.frames() * (bin + 1) / bins; ++frame) {
            for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                const auto value = pcm.samples[frame * pcm.channels + channel];
                low = std::min(low, value);
                high = std::max(high, value);
            }
        }
        waveform.push_back({low, high});
    }
    // Bounded log-frequency DFT over the loudest of sixteen windows (not a live analyzer).
    constexpr std::size_t window = 2048;
    const auto count = std::min<std::uint64_t>(window, pcm.frames());
    std::uint64_t offset = 0;
    double loudest = -1;
    for (unsigned section = 0; section < 16; ++section) {
        const auto start = (pcm.frames() - count) * section / 15;
        double energy = 0;
        for (std::uint64_t frame = 0; frame < count; ++frame) {
            for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                const double sample = pcm.samples[(start + frame) * pcm.channels + channel];
                energy += sample * sample;
            }
        }
        if (energy > loudest) {
            loudest = energy;
            offset = start;
        }
    }
    json spectrum = json::array();
    for (unsigned bin = 0; bin < 64; ++bin) {
        const double frequency = 20 * std::pow(1000.0, bin / 63.0);
        double magnitude2 = 0;
        for (unsigned channel = 0; channel < pcm.channels; ++channel) {
            double real = 0, imaginary = 0;
            for (std::uint64_t frame = 0; frame < count; ++frame) {
                const double sample = pcm.samples[(offset + frame) * pcm.channels + channel] *
                    (0.5 - 0.5 * std::cos(2 * pi * frame / (count - 1)));
                const double phase = 2 * pi * frequency * frame / rate;
                real += sample * std::cos(phase);
                imaginary -= sample * std::sin(phase);
            }
            magnitude2 += real * real + imaginary * imaginary;
        }
        const double magnitude = 4 * std::sqrt(magnitude2 / pcm.channels) / count;
        spectrum.push_back({frequency, 20 * std::log10(std::max(1e-6, magnitude))});
    }
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << hash;
    return {{"frames", pcm.frames()}, {"channels", pcm.channels}, {"sample_rate", rate},
        {"duration_seconds", pcm.frames() / static_cast<double>(rate)}, {"peak", peak},
        {"rms", std::sqrt(square / pcm.samples.size())}, {"pcm_hash", text.str()},
        {"waveform", waveform}, {"spectrum", spectrum},
        {"spectrum_window_start_seconds", offset / static_cast<double>(rate)}};
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

struct AudioSoundDesign::Impl {
    AudioPcm original, processed;
    json report = {{"ready", false}};
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
    stop();
}

json AudioSoundDesign::render(const json& input, const json& assets) {
    auto recipe = validate(input);
    std::vector<float> envelope;
    auto original = recipe.at("mode") == "synth" ? synthesize(recipe, envelope) :
                                                     loadRange(recipe, assets);
    auto processed = transform(original, recipe);
    const auto processedReport = analyze(processed);
    const auto originalReport = analyze(original);
    json report = {{"ready", true}, {"version", 2}, {"recipe", recipe},
        {"processed", processedReport}, {"original", originalReport}, {"envelope", envelope},
        {"engine", "DaisySP + Signalsmith Stretch"}};
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
    // Validation, decode and rendering succeed before replacing the previous audition.
    stop();
    impl_->original = std::move(original);
    impl_->processed = std::move(processed);
    impl_->report = std::move(report);
    impl_->auditionOriginal = false;
    return status();
}

bool AudioSoundDesign::playing() const {
    return impl_->soundReady && ma_sound_is_playing(&impl_->sound);
}

json AudioSoundDesign::status() const {
    auto report = impl_->report;
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
    stop();
    impl_->original = {};
    impl_->processed = {};
    impl_->report = {{"ready", false}};
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
                    {"pcm_hash", impl_->report.at("processed").at("pcm_hash")}, {"version", 2}};
            }
        }
        throw std::runtime_error("No available sound filename");
    }
    require(publish(impl_->processed, target), "Existing output is preserved");
    return {{"path", pathUtf8(target)}, {"name", name}, {"recipe", impl_->report.at("recipe")},
        {"origin", impl_->report.at("origin")},
        {"pcm_hash", impl_->report.at("processed").at("pcm_hash")}, {"version", 2}};
}
} // namespace audio
