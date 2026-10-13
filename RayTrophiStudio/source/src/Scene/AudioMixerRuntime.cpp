#include "Audio/AudioMixerRuntime.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace audio {
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
namespace {
void check(ma_result result) {
    if (result != MA_SUCCESS) {
        throw std::runtime_error("audio_mixer_unavailable: " + std::to_string(result));
    }
}
float db(float level) {
    return level > 0 ? (std::max)(-120.0f, 20.0f * std::log10(level)) : -120.0f;
}
}

void AudioMixerRuntime::process(ma_node* node, const float** input, ma_uint32* countIn,
                               float** output, ma_uint32* countOut) {
    auto& meter = *reinterpret_cast<Meter*>(node);
    const bool hasInput = input && input[0] && *countIn > 0;
    const ma_uint32 frames = hasInput ? (std::min)(*countIn, *countOut) : *countOut;
    float peak = 0;
    double square = 0;
    for (ma_uint32 i = 0; i < frames * 2; ++i) {
        const float sample = hasInput && std::isfinite(input[0][i]) ? input[0][i] : 0;
        output[0][i] = sample;
        peak = (std::max)(peak, std::abs(sample));
        square += static_cast<double>(sample) * sample;
    }
    // Continuous nodes may receive a null PCM pointer with padded input frames.
    // Consume that silent padding too, so it cannot remain cached for the next block.
    *countIn = (std::min)(*countIn, frames);
    *countOut = frames;
    const float rms = frames > 0 ? static_cast<float>(std::sqrt(square / (frames * 2))) : 0;
    meter.peak.store(std::bit_cast<std::uint32_t>(peak), std::memory_order_relaxed);
    meter.rms.store(std::bit_cast<std::uint32_t>(rms), std::memory_order_relaxed);
    const auto bits = std::bit_cast<std::uint32_t>(peak);
    auto hold = meter.hold.load(std::memory_order_relaxed);
    while (hold < bits && !meter.hold.compare_exchange_weak(hold, bits,
           std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
    if (peak >= 1) {
        meter.clips.fetch_add(1, std::memory_order_relaxed);
    }
}

AudioMixerRuntime::~AudioMixerRuntime() {
    shutdown();
}

void AudioMixerRuntime::initialize(ma_engine& engine) {
    shutdown();
    static const ma_node_vtable vtable = {process, nullptr, 1, 1,
        MA_NODE_FLAG_CONTINUOUS_PROCESSING | MA_NODE_FLAG_ALLOW_NULL_INPUT};
    try {
        const ma_uint32 channels = 2;
        for (std::size_t i = 0; i < buses_.size(); ++i) {
            auto& bus = buses_[i];
            check(ma_sound_group_init(&engine, MA_SOUND_FLAG_NO_SPATIALIZATION,
                                      nullptr, &bus.group));
            bus.groupReady = true;
            auto config = ma_node_config_init();
            config.vtable = &vtable;
            config.pInputChannels = &channels;
            config.pOutputChannels = &channels;
            check(ma_node_init(ma_engine_get_node_graph(&engine), &config, nullptr, &bus.meter.node));
            bus.meterReady = true;
            check(ma_node_attach_output_bus(&bus.group, 0, &bus.meter.node, 0));
            check(ma_node_attach_output_bus(&bus.meter.node, 0,
                i == 0 ? ma_engine_get_endpoint(&engine) : static_cast<ma_node*>(&buses_[0].group), 0));
            check(ma_sound_group_start(&bus.group));
        }
        ready_ = true;
        resetPeaks();
    } catch (...) {
        shutdown();
        throw;
    }
}

void AudioMixerRuntime::shutdown() {
    for (std::size_t i = buses_.size(); i > 0; --i) {
        auto& bus = buses_[i - 1];
        if (bus.groupReady) {
            ma_sound_group_uninit(&bus.group);
            bus.groupReady = false;
        }
        if (bus.meterReady) {
            ma_node_uninit(&bus.meter.node, nullptr);
            bus.meterReady = false;
        }
    }
    ready_ = false;
}

void AudioMixerRuntime::apply(const nlohmann::json& config) {
    if (!ready_) {
        return;
    }
    for (std::size_t i = 0; i < buses_.size(); ++i) {
        ma_sound_group_set_volume(&buses_[i].group,
            AudioMixer::effectiveGain(config, AudioMixer::names()[i]));
        ma_sound_group_set_pan_mode(&buses_[i].group, ma_pan_mode_balance);
        ma_sound_group_set_pan(&buses_[i].group,
            config.at(AudioMixer::names()[i]).value("pan", 0.0f));
    }
}

ma_sound_group* AudioMixerRuntime::group(const std::string& name) {
    const auto& names = AudioMixer::names();
    const auto it = std::find(names.begin(), names.end(), name);
    if (!ready_ || it == names.end()) {
        throw std::invalid_argument("Mixer bus unavailable: " + name);
    }
    return &buses_[static_cast<std::size_t>(it - names.begin())].group;
}

void AudioMixerRuntime::route(ma_sound& sound, const std::string& bus) {
    check(ma_node_attach_output_bus(&sound, 0, group(bus), 0));
}

nlohmann::json AudioMixerRuntime::meters() const {
    nlohmann::json result = nlohmann::json::object();
    for (std::size_t i = 0; i < buses_.size(); ++i) {
        const auto& m = buses_[i].meter;
        const float peak = ready_ ? std::bit_cast<float>(m.peak.load(std::memory_order_relaxed)) : 0;
        const float rms = ready_ ? std::bit_cast<float>(m.rms.load(std::memory_order_relaxed)) : 0;
        const float hold = ready_ ? std::bit_cast<float>(m.hold.load(std::memory_order_relaxed)) : 0;
        result[AudioMixer::names()[i]] = {{"peak", peak}, {"rms", rms}, {"peak_dbfs", db(peak)},
            {"rms_dbfs", db(rms)}, {"peak_hold_dbfs", db(hold)},
            {"overload_blocks", m.clips.load(std::memory_order_relaxed)}, {"ready", ready_}};
    }
    return result;
}

void AudioMixerRuntime::resetPeaks() {
    for (auto& bus : buses_) {
        bus.meter.hold.store(0, std::memory_order_relaxed);
        bus.meter.clips.store(0, std::memory_order_relaxed);
    }
}
}
