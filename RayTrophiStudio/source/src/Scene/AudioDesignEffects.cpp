#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioDesignLimits.h"
#include "Audio/AudioDesignDsp.h"
#include "Audio/AudioParametricEq.h"
#include "../../../external/DaisySP/Source/Effects/chorus.h"
#include "../../../external/DaisySP/Source/Effects/phaser.h"
#include "../../../external/DaisySP/Source/Effects/overdrive.h"
#include "../../../external/DaisySP/Source/Filters/svf.h"
#include "../../../external/signalsmith-linear/fft.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <deque>
#include <memory>
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

json field(const char* label, double value, double low, double high,
           const char* unit = "", bool logarithmic = false) {
    return {{"label", label}, {"default", value}, {"min", low}, {"max", high},
        {"unit", unit}, {"logarithmic", logarithmic}, {"kind", "number"}};
}

json choice(const char* label, const char* value, std::initializer_list<const char*> values) {
    json options = json::array();
    for (const auto item : values) {
        options.push_back(item);
    }
    return {{"label", label}, {"default", value}, {"choices", options}, {"kind", "choice"}};
}

json layerCommon() {
    return {{"gain_db", field("Level", -6, -60, 6, "dB")},
        {"pan", field("Pan", 0, -1, 1)}, {"start_seconds", field("Start", 0, 0, 10, "s")}};
}

json normalize(const json& input, const json& fields, const std::vector<std::string>& extra) {
    require(input.is_object(), "DSP module must be an object");
    for (auto entry = input.begin(); entry != input.end(); ++entry) {
        require(fields.contains(entry.key()) || std::find(extra.begin(), extra.end(), entry.key()) !=
                extra.end(), "Unknown DSP module field");
    }
    json result;
    for (auto entry = fields.begin(); entry != fields.end(); ++entry) {
        const auto& spec = entry.value();
        const auto value = input.value(entry.key(), spec.at("default"));
        if (spec.at("kind") == "choice") {
            require(value.is_string(), "DSP choice must be a string");
            const auto& choices = spec.at("choices");
            require(std::find(choices.begin(), choices.end(), value) != choices.end(),
                    "Unknown DSP choice");
        } else {
            require(value.is_number() && std::isfinite(value.get<double>()) &&
                    value.get<double>() >= spec.at("min").get<double>() &&
                    value.get<double>() <= spec.at("max").get<double>(),
                    "DSP control is outside its finite range");
        }
        result[entry.key()] = value;
    }
    require(!input.contains("enabled") || input.at("enabled").is_boolean(),
            "DSP enabled must be boolean");
    result["enabled"] = input.value("enabled", true);
    result["type"] = input.at("type");
    return result;
}

struct Delay {
    std::vector<float> values;
    std::size_t cursor = 0;
    float dampingState = 0;
    explicit Delay(std::size_t size) : values(std::max<std::size_t>(1, size), 0) {}
    float process(float input, float feedback, float damping) {
        const float delayed = values[cursor];
        dampingState += (1 - damping) * (delayed - dampingState);
        values[cursor] = input + feedback * dampingState;
        cursor = (cursor + 1) % values.size();
        return delayed;
    }
};

double compressorReduction(double over, double knee, double ratio) {
    if (knee > 0 && std::abs(over) <= knee * 0.5) {
        return (1 / ratio - 1) * std::pow(over + knee * 0.5, 2) / (2 * knee);
    }
    return over > knee * 0.5 ? over * (1 / ratio - 1) : 0;
}
} // namespace

json designDspCatalogue() {
    json layers = json::array();
    auto oscillator = layerCommon();
    oscillator["frequency_hz"] = field("Frequency", 220, 20, 8000, "Hz", true);
    oscillator["wave"] = choice("Wave", "sine", {"sine", "triangle", "saw", "square"});
    oscillator["detune_cents"] = field("Detune", 0, -1200, 1200, "cents");
    oscillator["pulse_width"] = field("Pulse width", 0.5, 0.1, 0.9);
    layers.push_back({{"type", "oscillator"}, {"name", "Oscillator"}, {"fields", oscillator}});
    auto noise = layerCommon();
    noise["color"] = choice("Color", "white", {"white", "pink", "brown"});
    noise["cutoff_hz"] = field("Cutoff", 6000, 20, 20000, "Hz", true);
    layers.push_back({{"type", "noise"}, {"name", "Noise texture"}, {"fields", noise}});
    auto fm = layerCommon();
    fm["frequency_hz"] = field("Carrier", 220, 20, 4000, "Hz", true);
    fm["ratio"] = field("Modulator ratio", 2, 0.1, 8);
    fm["index"] = field("FM depth", 1, 0, 10);
    layers.push_back({{"type", "fm"}, {"name", "FM voice"}, {"fields", fm}});
    auto resonator = layerCommon();
    resonator["frequency_hz"] = field("Body frequency", 220, 20, 4000, "Hz", true);
    resonator["structure"] = field("Structure", 0.6, 0, 1);
    resonator["brightness"] = field("Brightness", 0.5, 0, 1);
    resonator["damping"] = field("Damping", 0.5, 0, 1);
    resonator["position"] = field("Excitation position", 0.25, 0, 1);
    resonator["burst_seconds"] = field("Excitation burst", 0.005, 0.001, 1, "s", true);
    layers.push_back({{"type", "resonator"}, {"name", "Resonant body"}, {"fields", resonator}});
    auto asset = layerCommon();
    asset["source_in_seconds"] = field("Source start", 0, 0, 600, "s");
    asset["source_out_seconds"] = field("Source end", 1, 0.05, 600, "s");
    layers.push_back({{"type", "asset"}, {"name", "Audio sample layer"}, {"fields", asset},
        {"requires_asset", true}});

    json effects = json::array();
    effects.push_back({{"type", "eq"}, {"name", "Parametric EQ"}, {"fields", {
        {"output_db", field("Output trim", 0, -24, 24, "dB")}}}});
    effects.push_back({{"type", "filter"}, {"name", "Resonant filter"}, {"fields", {
        {"mode", choice("Mode", "lowpass", {"lowpass", "highpass", "bandpass", "notch"})},
        {"cutoff_hz", field("Cutoff", 4000, 20, 16000, "Hz", true)},
        {"resonance", field("Resonance", 0.2, 0, 0.95)},
        {"drive", field("Drive", 0, 0, 1)}}}});
    effects.push_back({{"type", "compressor"}, {"name", "Compressor"}, {"fields", {
        {"threshold_db", field("Threshold", -18, -60, 0, "dB")},
        {"ratio", field("Ratio", 4, 1, 20)}, {"knee_db", field("Knee", 6, 0, 24, "dB")},
        {"attack_ms", field("Attack", 5, 0.1, 200, "ms", true)},
        {"release_ms", field("Release", 80, 5, 2000, "ms", true)},
        {"makeup_db", field("Makeup", 0, -12, 12, "dB")}}}});
    effects.push_back({{"type", "delay"}, {"name", "Delay"}, {"fields", {
        {"time_seconds", field("Delay time", 0.2, 0.001, 1.5, "s", true)},
        {"feedback", field("Feedback", 0.35, 0, 0.9)},
        {"damping", field("Damping", 0.3, 0, 0.95)}}}});
    effects.push_back({{"type", "reverb"}, {"name", "Room reverb"}, {"fields", {
        {"decay_seconds", field("Decay", 1.5, 0.1, 8, "s", true)},
        {"damping", field("Damping", 0.4, 0, 0.95)},
        {"size", field("Room size", 0.6, 0.25, 1.5)}}}});
    effects.push_back({{"type", "chorus"}, {"name", "Chorus"}, {"fields", {
        {"rate_hz", field("Rate", 0.7, 0.05, 10, "Hz", true)},
        {"depth", field("Depth", 0.3, 0, 1)}, {"delay_ms", field("Delay", 15, 1, 40, "ms")},
        {"feedback", field("Feedback", 0.1, 0, 0.8)}}}});
    effects.push_back({{"type", "phaser"}, {"name", "Phaser"}, {"fields", {
        {"rate_hz", field("Rate", 0.5, 0.05, 10, "Hz", true)},
        {"depth", field("Depth", 0.5, 0, 1)},
        {"frequency_hz", field("Allpass frequency", 700, 20, 8000, "Hz", true)},
        {"feedback", field("Feedback", 0.2, 0, 0.8)}}}});
    effects.push_back({{"type", "saturation"}, {"name", "Overdrive"}, {"fields", {
        {"drive", field("Drive", 0.3, 0, 1)}}}});
    effects.push_back({{"type", "limiter"}, {"name", "Lookahead limiter"}, {"fields", {
        {"ceiling_db", field("Ceiling", -1, -12, 0, "dB")},
        {"release_ms", field("Release", 60, 5, 1000, "ms", true)}}}});
    effects.push_back({{"type", "spectral_region"}, {"name", "Spectral region"}, {"fields", {
        {"start_seconds", field("Region start", 0, 0, 60, "s")},
        {"end_seconds", field("Region end", 1, 0.001, 60, "s")},
        {"low_hz", field("Low frequency", 200, 20, 20000, "Hz", true)},
        {"high_hz", field("High frequency", 2000, 20, 20000, "Hz", true)},
        {"gain_db", field("Region gain", -24, -96, 12, "dB")},
        {"feather", field("Edge softness", 0.25, 0, 1)},
        {"shape", choice("Mask shape", "rectangle", {"rectangle", "ellipse"})},
        {"channel", choice("Channels", "all", {"all", "left", "right"})}}}});
    for (auto& effect : effects) {
        effect["fields"]["mix"] = field("Dry / wet", effect.at("type") == "delay" ||
            effect.at("type") == "reverb" ? 0.3 : 1, 0, 1);
    }
    const json bandFields = {
        {"mode", choice("Band type", "peak", {"peak", "lowshelf", "highshelf", "lowpass", "highpass"})},
        {"frequency_hz", field("Frequency", 1000, 20, 20000, "Hz", true)},
        {"gain_db", field("Gain", 0, -24, 24, "dB")}, {"q", field("Q", 0.707, 0.1, 10)}};
    return {{"version", 3}, {"max_duration_seconds", designMaxSeconds},
        {"working_pcm_budget_bytes", designWorkingPcmBudget},
        {"processing_mode", "bounded_memory"}, {"max_layers", 8}, {"max_effects", 8}, {"max_eq_bands", 8},
        {"layers", layers}, {"effects", effects}, {"eq_band_fields", bandFields},
        {"eq_presets", json::array({{{"name", "Bass / Body / Brightness (flat)"},
            {"bands", json::array({
                {{"mode", "lowshelf"}, {"frequency_hz", 120}, {"gain_db", 0}, {"q", 0.707}, {"enabled", true}},
                {{"mode", "peak"}, {"frequency_hz", 1000}, {"gain_db", 0}, {"q", 0.707}, {"enabled", true}},
                {{"mode", "highshelf"}, {"frequency_hz", 6000}, {"gain_db", 0}, {"q", 0.707}, {"enabled", true}}
            })}}})},
        {"presets", json::array({
            {{"name", "Warm layered tone"}, {"recipe", {{"mode", "synth"}, {"preset", "tone"},
                {"layers", json::array({{{"type", "oscillator"}, {"wave", "saw"}, {"frequency_hz", 110}},
                    {{"type", "oscillator"}, {"frequency_hz", 110}, {"detune_cents", 7}}})},
                {"effects", json::array({{{"type", "filter"}, {"cutoff_hz", 2500}},
                    {{"type", "chorus"}, {"mix", 0.25}}})}}}},
            {{"name", "Resonant metal hit"}, {"recipe", {{"mode", "synth"}, {"preset", "metal"},
                {"layers", json::array({{{"type", "resonator"}, {"structure", 0.8}},
                    {{"type", "fm"}, {"gain_db", -18}, {"index", 2}}})},
                {"sustain", 0.1}, {"effects", json::array({{{"type", "reverb"}, {"mix", 0.2}}})},
                {"tail_seconds", 1}}}},
            {{"name", "Layered wind"}, {"recipe", {{"mode", "synth"}, {"preset", "wind"},
                {"layers", json::array({{{"type", "noise"}, {"color", "pink"}},
                    {{"type", "noise"}, {"color", "brown"}, {"gain_db", -12}, {"pan", 0.4}}})},
                {"modulation_depth", 0.3}, {"effects", json::array({{{"type", "eq"},
                    {"bands", json::array({{{"mode", "highpass"}, {"frequency_hz", 60}}})}}})}}}}
        })}};
}

void validateDesignDsp(const json& input, json& recipe) {
    const auto catalogue = designDspCatalogue();
    for (const char* array : {"layers", "effects"}) {
        const auto entries = input.value(array, json::array());
        require(entries.is_array() && entries.size() <= 8, "At most eight DSP modules per array");
        require(std::string(array) != "layers" || entries.empty() || recipe.at("mode") == "synth",
                "Layers require synth mode; an asset may be an explicit sample layer");
        json normalized = json::array();
        for (const auto& entry : entries) {
            require(entry.is_object() && entry.contains("type") && entry.at("type").is_string(),
                    "DSP module needs a type string");
            const auto& definitions = catalogue.at(array);
            auto definition = std::find_if(definitions.begin(), definitions.end(), [&](const json& d) {
                return d.at("type") == entry.at("type");
            });
            require(definition != definitions.end(), "Unknown DSP module type");
            auto module = normalize(entry, definition->at("fields"),
                {"type", "enabled", std::string(array) == "layers" ? "asset_id" : "bands"});
            if (entry.at("type") == "spectral_region") {
                require(module.at("end_seconds").get<double>() >
                        module.at("start_seconds").get<double>(),
                        "Spectral region end must follow start");
                require(module.at("high_hz").get<double>() > module.at("low_hz").get<double>(),
                        "Spectral high frequency must exceed low frequency");
            }
            if (std::string(array) == "layers" && entry.at("type") == "asset") {
                require(entry.contains("asset_id") && entry.at("asset_id").is_string(),
                        "Sample layer needs asset_id");
                module["asset_id"] = entry.at("asset_id");
                const double begin = module.at("source_in_seconds"), end = module.at("source_out_seconds");
                require(end > begin && end - begin <= 10, "Sample layer range must be 0..10 seconds");
            } else {
                require(!entry.contains("asset_id"), "asset_id applies only to sample layers");
            }
            if (std::string(array) == "effects" && entry.at("type") == "eq") {
                const auto bands = entry.value("bands", json::array());
                require(bands.is_array() && bands.size() <= 8, "EQ accepts at most eight bands");
                module["bands"] = json::array();
                for (const auto& band : bands) {
                    auto prepared = band;
                    require(band.is_object(), "EQ band must be an object");
                    require(!band.contains("type"), "EQ bands do not take a type field");
                    prepared["type"] = "band";
                    auto normalizedBand = normalize(prepared, catalogue.at("eq_band_fields"),
                                                    {"type", "enabled"});
                    normalizedBand.erase("type");
                    const EqBiquad coefficients(normalizedBand);
                    (void)coefficients;
                    module["bands"].push_back(normalizedBand);
                }
            } else {
                require(!entry.contains("bands"), "bands apply only to EQ");
            }
            normalized.push_back(module);
        }
        recipe[array] = normalized;
    }
}

void applyDesignEffects(AudioPcm& pcm, const json& effects, const AudioCancelCheck& check) {
    for (const auto& effect : effects) {
        if (check) {
            check();
        }
        if (!effect.at("enabled").get<bool>() || effect.at("mix").get<double>() == 0) {
            continue;
        }
        const auto type = effect.at("type").get<std::string>();
        const float mix = effect.at("mix"), dryMix = 1 - mix;
        std::vector<float> dry = pcm.samples;
        if (type == "spectral_region") {
            applyDesignSpectralRegion(pcm, effect, check);
        } else if (type == "eq") {
            applyParametricEq(pcm, effect, check);
        } else if (type == "compressor") {
            const double threshold = effect.at("threshold_db"), ratio = effect.at("ratio");
            const double knee = effect.at("knee_db");
            const double attack = std::exp(-1 / (effect.at("attack_ms").get<double>() * 0.001 * rate));
            const double release = std::exp(-1 / (effect.at("release_ms").get<double>() * 0.001 * rate));
            const double makeup = std::pow(10.0, effect.at("makeup_db").get<double>() / 20);
            double envelope = 0;
            for (std::size_t frame = 0; frame < pcm.frames(); ++frame) {
                if (check && frame % 1024 == 0) {
                    check();
                }
                double peak = 0;
                for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                    peak = std::max(peak, std::abs(double(pcm.samples[frame * pcm.channels + channel])));
                }
                const double coefficient = peak > envelope ? attack : release;
                envelope = peak + coefficient * (envelope - peak);
                const double over = 20 * std::log10(std::max(1e-9, envelope)) - threshold;
                const double reduction = compressorReduction(over, knee, ratio);
                const float gain = static_cast<float>(std::pow(10.0, reduction / 20) * makeup);
                for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                    pcm.samples[frame * pcm.channels + channel] *= gain;
                }
            }
        } else if (type == "delay") {
            const auto frames = static_cast<std::size_t>(std::llround(
                effect.at("time_seconds").get<double>() * rate));
            Delay delays[] = {Delay(frames), Delay(frames)};
            const float feedback = effect.at("feedback"), damping = effect.at("damping");
            for (std::size_t frame = 0; frame < pcm.frames(); ++frame) {
                if (check && frame % 1024 == 0) {
                    check();
                }
                for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                    auto& value = pcm.samples[frame * pcm.channels + channel];
                    value = delays[channel].process(value, feedback, damping);
                }
            }
        } else if (type == "reverb") {
            const double size = effect.at("size"), decay = effect.at("decay_seconds");
            const float damping = effect.at("damping");
            const double lengths[] = {0.0297, 0.0371, 0.0411, 0.0437};
            std::vector<Delay> combs, allpasses;
            std::vector<float> feedback;
            for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                for (const double length : lengths) {
                    const auto frames = static_cast<std::size_t>(rate * length * size) + channel * 17;
                    combs.emplace_back(frames);
                    feedback.push_back(static_cast<float>(std::exp(-3 * std::log(10.0) * frames /
                                                                  (rate * decay))));
                }
                allpasses.emplace_back(static_cast<std::size_t>(rate * 0.005 * size) + channel * 7);
                allpasses.emplace_back(static_cast<std::size_t>(rate * 0.0017 * size) + channel * 11);
            }
            for (std::size_t frame = 0; frame < pcm.frames(); ++frame) {
                if (check && frame % 1024 == 0) {
                    check();
                }
                for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                    auto& value = pcm.samples[frame * pcm.channels + channel];
                    float wet = 0;
                    for (unsigned comb = 0; comb < 4; ++comb) {
                        const auto index = channel * 4 + comb;
                        wet += combs[index].process(value, feedback[index], damping) * 0.25f;
                    }
                    for (unsigned stage = 0; stage < 2; ++stage) {
                        auto& delay = allpasses[channel * 2 + stage];
                        const float held = delay.values[delay.cursor];
                        const float output = held - wet * 0.5f;
                        delay.values[delay.cursor] = wet + output * 0.5f;
                        delay.cursor = (delay.cursor + 1) % delay.values.size();
                        wet = output;
                    }
                    value = wet;
                }
            }
        } else if (type == "limiter") {
            const double ceiling = std::pow(10.0, effect.at("ceiling_db").get<double>() / 20);
            const double release = std::exp(-1 / (effect.at("release_ms").get<double>() * 0.001 * rate));
            std::vector<double> peaks(pcm.frames());
            for (std::size_t frame = 0; frame < pcm.frames(); ++frame) {
                if (check && frame % 1024 == 0) {
                    check();
                }
                for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                    peaks[frame] = std::max(peaks[frame], std::abs(double(dry[frame * pcm.channels + channel])));
                }
            }
            std::deque<std::size_t> maximum;
            std::size_t next = 0;
            double gain = 1;
            for (std::size_t frame = 0; frame < pcm.frames(); ++frame) {
                if (check && frame % 1024 == 0) {
                    check();
                }
                while (next < std::min<std::size_t>(pcm.frames(), frame + 241)) {
                    while (!maximum.empty() && peaks[maximum.back()] <= peaks[next]) {
                        maximum.pop_back();
                    }
                    maximum.push_back(next++);
                }
                while (!maximum.empty() && maximum.front() < frame) {
                    maximum.pop_front();
                }
                const double target = std::min(1.0, ceiling / std::max(1e-9, peaks[maximum.front()]));
                gain = target < gain ? target : target + release * (gain - target);
                for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                    pcm.samples[frame * pcm.channels + channel] *= static_cast<float>(gain);
                }
            }
        } else {
            // Upstream objects contain delay arrays; allocate on the heap rather than the UI stack.
            auto chorus = std::make_unique<std::array<daisysp::ChorusEngine, 2>>();
            auto phaser = std::make_unique<std::array<daisysp::PhaserEngine, 2>>();
            daisysp::Overdrive drive[2];
            daisysp::Svf filter[2];
            for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                if (type == "chorus") {
                    (*chorus)[channel].Init(rate);
                    (*chorus)[channel].SetLfoFreq(effect.at("rate_hz"));
                    (*chorus)[channel].SetLfoDepth(effect.at("depth"));
                    (*chorus)[channel].SetDelayMs(effect.at("delay_ms"));
                    (*chorus)[channel].SetFeedback(effect.at("feedback"));
                } else if (type == "phaser") {
                    (*phaser)[channel].Init(rate);
                    (*phaser)[channel].SetLfoFreq(effect.at("rate_hz"));
                    (*phaser)[channel].SetLfoDepth(effect.at("depth"));
                    (*phaser)[channel].SetFreq(effect.at("frequency_hz"));
                    (*phaser)[channel].SetFeedback(effect.at("feedback"));
                } else if (type == "filter") {
                    filter[channel].Init(rate);
                    filter[channel].SetFreq(effect.at("cutoff_hz"));
                    filter[channel].SetRes(effect.at("resonance"));
                    filter[channel].SetDrive(effect.at("drive"));
                } else {
                    drive[channel].Init();
                    drive[channel].SetDrive(effect.at("drive"));
                }
            }
            const auto mode = effect.value("mode", std::string());
            for (std::size_t frame = 0; frame < pcm.frames(); ++frame) {
                if (check && frame % 1024 == 0) {
                    check();
                }
                for (unsigned channel = 0; channel < pcm.channels; ++channel) {
                    auto& value = pcm.samples[frame * pcm.channels + channel];
                    if (type == "chorus") {
                        value = (*chorus)[channel].Process(value);
                    } else if (type == "phaser") {
                        value = (*phaser)[channel].Process(value);
                    } else if (type == "filter") {
                        filter[channel].Process(value);
                        value = mode == "lowpass" ? filter[channel].Low() :
                            mode == "highpass" ? filter[channel].High() :
                            mode == "bandpass" ? filter[channel].Band() : filter[channel].Notch();
                    } else {
                        value = drive[channel].Process(value);
                    }
                }
            }
        }
        for (std::size_t sample = 0; sample < pcm.samples.size(); ++sample) {
            pcm.samples[sample] = dry[sample] * dryMix + pcm.samples[sample] * mix;
            require(std::isfinite(pcm.samples[sample]), "Nonfinite effect output");
        }
    }
}

json designEqResponse(const json& effects) {
    std::array<std::complex<double>, 256> combined;
    combined.fill({1, 0});
    for (const auto& effect : effects) {
        if (effect.at("type") != "eq") {
            continue;
        }
        const auto curve = parametricEqCurve(effect);
        for (unsigned point = 0; point < combined.size(); ++point) {
            combined[point] *= curve[point];
        }
    }
    json response = json::array();
    for (unsigned point = 0; point < combined.size(); ++point) {
        response.push_back({20 * std::pow(1000.0, point / 255.0),
            20 * std::log10(std::max(1e-12, std::abs(combined[point])))});
    }
    return response;
}

json designEffectVisual(const json& input) {
    json recipe = {{"mode", "synth"}};
    validateDesignDsp({{"effects", json::array({input})}}, recipe);
    const auto& effect = recipe.at("effects")[0];
    const auto type = effect.at("type").get<std::string>();
    json result = {{"effect", effect}};
    if (type == "spectral_region") {
        result["kind"] = "spectral_region";
        result["points"] = json::array();
        result["fft_size"] = 2048;
        result["hop_frames"] = 512;
    } else if (type == "eq") {
        result = parametricEqVisual(effect);
        result["kind"] = "eq";
        result["effect"] = effect;
    } else if (type == "filter") {
        result["kind"] = "filter";
        result["points"] = json::array();
        constexpr unsigned size = 4096;
        daisysp::Svf filter;
        filter.Init(rate);
        filter.SetFreq(effect.at("cutoff_hz"));
        filter.SetRes(effect.at("resonance"));
        filter.SetDrive(effect.at("drive"));
        const auto mode = effect.at("mode").get<std::string>();
        std::vector<float> time(size);
        for (unsigned frame = 0; frame < size; ++frame) {
            filter.Process(frame == 0 ? 0.001f : 0);
            time[frame] = 1000 * (mode == "lowpass" ? filter.Low() : mode == "highpass" ?
                filter.High() : mode == "bandpass" ? filter.Band() : filter.Notch());
        }
        signalsmith::linear::Pow2RealFFT<float> fft(size);
        std::vector<std::complex<float>> response(size / 2);
        fft.fft(time.data(), response.data());
        const double mix = effect.at("enabled").get<bool>() ? effect.at("mix").get<double>() : 0;
        for (unsigned point = 0; point < 256; ++point) {
            const double hz = 20 * std::pow(1000.0, double(point) / 255);
            const double bin = hz * size / rate;
            const auto low = std::min<unsigned>(size / 2 - 1, static_cast<unsigned>(bin));
            const auto high = std::min<unsigned>(size / 2 - 1, low + 1);
            const double fraction = bin - low;
            const std::complex<double> a = low == 0 ? std::complex<double>(response[0].real(), 0) :
                                                               std::complex<double>(response[low]);
            const auto wet = a * (1 - fraction) + std::complex<double>(response[high]) * fraction;
            result["points"].push_back({hz,
                20 * std::log10(std::max(1e-9, std::abs((1 - mix) + wet * mix)))});
        }
    } else if (type == "compressor" || type == "limiter" || type == "saturation") {
        result["kind"] = "transfer";
        result["points"] = json::array();
        daisysp::Overdrive drive;
        drive.Init();
        if (type == "saturation") {
            drive.SetDrive(effect.at("drive"));
        }
        for (unsigned point = 0; point < 128; ++point) {
            const double inputDb = -60 + 66 * double(point) / 127;
            double outputDb = inputDb;
            if (type == "compressor") {
                outputDb += compressorReduction(inputDb - effect.at("threshold_db").get<double>(),
                    effect.at("knee_db"), effect.at("ratio")) + effect.at("makeup_db").get<double>();
            } else if (type == "limiter") {
                outputDb = std::min(outputDb, effect.at("ceiling_db").get<double>());
            } else {
                outputDb = 20 * std::log10(std::max(1e-9,
                    double(std::abs(drive.Process(static_cast<float>(std::pow(10.0, inputDb / 20)))))));
            }
            const double dry = std::pow(10.0, inputDb / 20);
            const double wet = std::pow(10.0, outputDb / 20);
            const double mix = effect.at("enabled").get<bool>() ? effect.at("mix").get<double>() : 0;
            result["points"].push_back({inputDb,
                20 * std::log10(std::max(1e-9, dry * (1 - mix) + wet * mix))});
        }
    } else if (type == "delay" || type == "reverb") {
        result["kind"] = "tail";
        result["points"] = json::array();
        const double horizon = type == "delay" ? effect.at("time_seconds").get<double>() * 12 :
                                               effect.at("decay_seconds").get<double>();
        for (unsigned point = 0; point < 128; ++point) {
            const double time = horizon * double(point) / 127;
            const double db = type == "delay" ?
                20 * std::floor(time / effect.at("time_seconds").get<double>()) *
                    std::log10(std::max(1e-9, effect.at("feedback").get<double>())) :
                -60 * time / effect.at("decay_seconds").get<double>();
            result["points"].push_back({time, std::max(-120.0, db)});
        }
    } else {
        result["kind"] = "modulation";
        result["points"] = json::array();
        const double frequency = effect.value("rate_hz", 1.0), depth = effect.value("depth", 0.0);
        for (unsigned point = 0; point < 128; ++point) {
            const double time = 2 * double(point) / 127;
            result["points"].push_back({time, depth * std::sin(2 * pi * frequency * time)});
        }
    }
    return result;
}
} // namespace audio
