#include "Audio/AcousticPlayback.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace audio {
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
namespace {
float peakGain(const std::array<float, 6>& gains) {
    float peak = 0;
    for (const float gain : gains) {
        peak = (std::max)(peak, std::abs(gain));
    }
    return peak;
}
}

AcousticPlayback::~AcousticPlayback() {
    if (ready_) {
        ma_node_uninit(&node_.base, nullptr);
    }
}

void AcousticPlayback::initialize(ma_engine& engine, ma_sound& sound, ma_sound_group* bus) {
    static const ma_node_vtable table = {process, nullptr, 1, 1,
        MA_NODE_FLAG_CONTINUOUS_PROCESSING | MA_NODE_FLAG_ALLOW_NULL_INPUT};
    const ma_uint32 channels = 2;
    auto config = ma_node_config_init();
    config.vtable = &table;
    config.pInputChannels = &channels;
    config.pOutputChannels = &channels;
    response(nlohmann::json::array({1, 1, 1, 1, 1, 1}));
    history_.assign(48002 * 2, 0);
    node_.owner = this;
    if (ma_node_init(ma_engine_get_node_graph(&engine), &config, nullptr, &node_.base) != MA_SUCCESS) {
        throw std::runtime_error("Could not initialize acoustic playback node");
    }
    ready_ = true;
    route(bus);
    if (ma_node_attach_output_bus(&sound, 0, &node_.base, 0) != MA_SUCCESS) {
        throw std::runtime_error("Could not attach acoustic playback node");
    }
}

void AcousticPlayback::route(ma_sound_group* bus) {
    if (ma_node_attach_output_bus(&node_.base, 0, bus, 0) != MA_SUCCESS) {
        throw std::runtime_error("Could not route acoustic playback node");
    }
}

void AcousticPlayback::response(const nlohmann::json& energy) {
    for (std::size_t band = 0; band < 6; ++band) {
        const float amplitude = std::sqrt(energy.at(band).get<float>());
        target_[band].store(std::bit_cast<std::uint32_t>(amplitude), std::memory_order_relaxed);
    }
}

void AcousticPlayback::reflections(const nlohmann::json& paths,
    const nlohmann::json& source, const nlohmann::json& listener,
    const nlohmann::json& position) {
    double squared = 0;
    for (unsigned axis = 0; axis < 3; ++axis) {
        const double delta = position.at(axis).get<double>() - listener.at(axis).get<double>();
        squared += delta * delta;
    }
    const double directUnits = std::sqrt(squared);
    const double minimum = source.at("min_distance");
    const double maximum = source.at("max_distance");
    const double rolloff = source.at("rolloff");
    const auto distanceGain = [&](double distance) {
        return minimum / (minimum + rolloff *
            (std::clamp(distance, minimum, maximum) - minimum));
    };
    for (std::size_t slot = 0; slot < taps_.size(); ++slot) {
        auto& tap = taps_[slot];
        const bool valid = slot < paths.size();
        float delay = 0;
        float relative = 0;
        if (valid) {
            const auto& path = paths.at(slot);
            delay = path.at("excess_delay_seconds").get<float>() * 48000;
            // Sound spatialization already applies the direct distance gain.
            const double directMetres = path.at("distance_meters").get<double>() -
                path.at("excess_delay_seconds").get<double>() * 343.0;
            const double reflectedUnits = directMetres > 1e-8 ?
                directUnits * path.at("distance_meters").get<double>() / directMetres : directUnits;
            relative = path.value("kind", "specular") == "diffuse" ? 1.0f :
                static_cast<float>(distanceGain(reflectedUnits) / distanceGain(directUnits));
        }
        if (valid) {
            tap.delay.store(std::bit_cast<std::uint32_t>(std::clamp(delay, 0.0f, 48000.0f)),
                            std::memory_order_relaxed);
        }
        for (std::size_t band = 0; band < 6; ++band) {
            float amplitude = valid ? relative *
                std::sqrt(paths.at(slot).at("energy").at(band).get<float>()) : 0;
            // Energetic diffuse bins carry no coherent phase. Fixed signed
            // reconstruction avoids summing every diffuse bin in positive phase;
            // it is an auralization approximation, not wave interference physics.
            if (valid && paths.at(slot).value("kind", "specular") == "diffuse") {
                const std::uint32_t hash = static_cast<std::uint32_t>(
                    (slot + 1) * 0x9e3779b9u + (band + 1) * 0x85ebca6bu);
                if (((hash ^ (hash >> 16)) & 1u) != 0) {
                    amplitude = -amplitude;
                }
            }
            tap.target[band].store(std::bit_cast<std::uint32_t>(amplitude),
                                   std::memory_order_relaxed);
        }
    }
}

void AcousticPlayback::process(ma_node* node, const float** input, ma_uint32* countIn,
                               float** output, ma_uint32* countOut) {
    auto& self = *reinterpret_cast<Node*>(node)->owner;
    const bool hasInput = input && input[0] && *countIn > 0;
    const auto frames = hasInput ? (std::min)(*countIn, *countOut) : *countOut;
    // Complementary one-pole bands: differences telescope to an exact unity bypass.
    // The end bands extend to DC/Nyquist. This is an energy-envelope approximation.
    static constexpr std::array<float, 5> alpha = {
        0.0228743383f, 0.0452254413f, 0.0884055420f, 0.1689955442f, 0.3094315944f};
    std::array<float, 6> target;
    float tapTarget[kAcousticPlaybackTaps][6];
    float delayTarget[kAcousticPlaybackTaps];
    std::array<std::size_t, kAcousticPlaybackTaps> activeSlots{};
    std::size_t activeCount = 0;
    for (std::size_t slot = 0; slot < kAcousticPlaybackTaps; ++slot) {
        delayTarget[slot] = std::bit_cast<float>(self.taps_[slot].delay.load(
            std::memory_order_relaxed));
        bool hasTarget = false;
        for (std::size_t band = 0; band < 6; ++band) {
            tapTarget[slot][band] = std::bit_cast<float>(self.taps_[slot].target[band].load(
                std::memory_order_relaxed));
            hasTarget = hasTarget || tapTarget[slot][band] != 0;
        }
        if (hasTarget || peakGain(self.taps_[slot].gain) >= 1e-7f) {
            activeSlots[activeCount++] = slot;
        }
    }
    for (std::size_t band = 0; band < 6; ++band) {
        target[band] = std::bit_cast<float>(self.target_[band].load(std::memory_order_relaxed));
    }
    bool bypass = true;
    for (std::size_t band = 0; band < 6; ++band) {
        bypass = bypass && target[band] == 1 && std::abs(self.gain_[band] - 1) < 2e-5f;
    }
    for (std::size_t slot = 0; slot < kAcousticPlaybackTaps; ++slot) {
        for (std::size_t band = 0; band < 6; ++band) {
            bypass = bypass && tapTarget[slot][band] == 0 &&
                std::abs(self.taps_[slot].gain[band]) < 1e-7f;
        }
    }
    if (bypass) {
        self.gain_.fill(1);
        for (ma_uint32 frame = 0; frame < frames; ++frame) {
            for (unsigned channel = 0; channel < 2; ++channel) {
                const auto index = frame * 2 + channel;
                const float sample = hasInput && std::isfinite(input[0][index]) ?
                    input[0][index] : 0;
                self.history_[self.cursor_ * 2 + channel] = sample;
                output[0][index] = sample;
            }
            self.cursor_ = (self.cursor_ + 1) % 48002;
        }
        *countIn = (std::min)(*countIn, frames);
        *countOut = frames;
        return;
    }
    for (ma_uint32 frame = 0; frame < frames; ++frame) {
        for (std::size_t band = 0; band < 6; ++band) {
            self.gain_[band] += 0.002f * (target[band] - self.gain_[band]);
        }
        for (std::size_t active = 0; active < activeCount; ++active) {
            const auto slot = activeSlots[active];
            auto& tap = self.taps_[slot];
            if (peakGain(tap.gain) < 1e-7f) {
                tap.frames = delayTarget[slot];
            } else {
                tap.frames += 0.002f * (delayTarget[slot] - tap.frames);
            }
            for (std::size_t band = 0; band < 6; ++band) {
                tap.gain[band] += 0.002f * (tapTarget[slot][band] - tap.gain[band]);
            }
        }
        for (unsigned channel = 0; channel < 2; ++channel) {
            const auto index = frame * 2 + channel;
            const float sample = hasInput && std::isfinite(input[0][index]) ?
                input[0][index] : 0;
            self.history_[self.cursor_ * 2 + channel] = sample;
            float previous = 0;
            float value = 0;
            for (std::size_t band = 0; band < 5; ++band) {
                auto& low = self.low_[channel][band];
                low += alpha[band] * (sample - low);
                value += self.gain_[band] * (low - previous);
                previous = low;
            }
            value += self.gain_[5] * (sample - previous);
            for (std::size_t active = 0; active < activeCount; ++active) {
                auto& tap = self.taps_[activeSlots[active]];
                if (peakGain(tap.gain) < 1e-7f) {
                    continue;
                }
                const auto whole = static_cast<std::size_t>(tap.frames);
                const float fraction = tap.frames - static_cast<float>(whole);
                const auto first = (self.cursor_ + 48002 - whole) % 48002;
                const auto second = (first + 48001) % 48002;
                const float delayed = self.history_[first * 2 + channel] * (1 - fraction) +
                    self.history_[second * 2 + channel] * fraction;
                float prior = 0;
                for (std::size_t band = 0; band < 5; ++band) {
                    auto& low = tap.low[channel][band];
                    low += alpha[band] * (delayed - low);
                    value += tap.gain[band] * (low - prior);
                    prior = low;
                }
                value += tap.gain[5] * (delayed - prior);
            }
            output[0][index] = value;
        }
        self.cursor_ = (self.cursor_ + 1) % 48002;
    }
    *countIn = (std::min)(*countIn, frames);
    *countOut = frames;
}
}
