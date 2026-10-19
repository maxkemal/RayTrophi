#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioLiveAnalysis.h"
#include "../../../external/signalsmith-linear/fft.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <utility>

namespace audio {
static_assert(std::atomic<unsigned>::is_always_lock_free);
namespace {
constexpr double pi = 3.14159265358979323846;
double powerDb(double power) {
    return 10 * std::log10(std::max(1e-12, power));
}
}

void AudioLiveAnalysis::reset(unsigned sampleRate) {
    sampleRate_ = sampleRate;
    pendingFrames_ = 0;
    writtenFrames_ = 0;
    analyzedFrame_ = 0;
    cached_ = nlohmann::json::object();
    received_ = {};
    history_.clear();
    for (auto& slot : slots_) {
        slot.state.store(0, std::memory_order_relaxed);
        slot.endFrame = 0;
    }
}

void AudioLiveAnalysis::push(const float* stereo, unsigned count) {
    for (unsigned index = 0; index < count; ++index) {
        pending_[pendingFrames_ * 2] = stereo[index * 2];
        pending_[pendingFrames_ * 2 + 1] = stereo[index * 2 + 1];
        ++writtenFrames_;
        if (++pendingFrames_ == frames) {
            publish();
            pendingFrames_ = 0;
        }
    }
}

void AudioLiveAnalysis::publish() {
    // Prefer a free slot; otherwise replace an unread snapshot. Never touch a reader.
    for (unsigned expectedState : {0u, 2u}) {
        for (auto& slot : slots_) {
            auto expected = expectedState;
            if (slot.state.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
                std::copy(pending_.begin(), pending_.end(), slot.pcm.begin());
                slot.endFrame = writtenFrames_;
                slot.state.store(2, std::memory_order_release);
                return;
            }
        }
    }
}

nlohmann::json AudioLiveAnalysis::read(bool details) {
    std::array<float, frames * 2> pcm;
    auto newest = analyzedFrame_;
    for (auto& slot : slots_) {
        unsigned expected = 2;
        if (slot.state.compare_exchange_strong(expected, 3, std::memory_order_acquire)) {
            if (slot.endFrame > newest) {
                pcm = slot.pcm;
                newest = slot.endFrame;
            }
            slot.state.store(0, std::memory_order_release);
        }
    }
    if (newest > analyzedFrame_) {
        cached_ = analyze(pcm);
        analyzedFrame_ = newest;
        received_ = std::chrono::steady_clock::now();
        // At most 15 observed windows/sec, bounded independently of polling frequency.
        if (history_.empty() || newest - history_.back().at("end_frame").get<std::uint64_t>() >=
                sampleRate_ / 15) {
            nlohmann::json db = nlohmann::json::array();
            for (const auto& point : cached_.at("spectrum")) {
                db.push_back(point[1]);
            }
            history_.push_back({{"end_frame", newest}, {"dbfs", std::move(db)}});
            if (history_.size() > 128) {
                history_.pop_front();
            }
        }
    }
    auto result = cached_;
    result["ready"] = analyzedFrame_ != 0;
    result["sample_rate"] = sampleRate_;
    result["window_frames"] = frames;
    result["end_frame"] = analyzedFrame_;
    result["snapshot_age_ms"] = analyzedFrame_ == 0 ? 0.0 :
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - received_).count();
    result["fresh"] = analyzedFrame_ != 0 && result["snapshot_age_ms"].get<double>() < 1000;
    if (details) {
        result["history"] = nlohmann::json::array();
        for (const auto& row : history_) {
            result["history"].push_back(row);
        }
        result["history_capacity"] = 128;
        result["history_kind"] = "observed_fft_windows";
    }
    return result;
}

nlohmann::json AudioLiveAnalysis::analyze(const std::array<float, frames * 2>& pcm) const {
    using json = nlohmann::json;
    signalsmith::linear::Pow2RealFFT<float> fft(frames);
    std::array<float, frames> window{}, time{};
    std::array<std::complex<float>, frames / 2> frequency{};
    std::array<double, frames / 2> power{};
    double windowSum = 0, windowSquare = 0;
    for (unsigned index = 0; index < frames; ++index) {
        window[index] = static_cast<float>(0.5 - 0.5 * std::cos(2 * pi * index / (frames - 1)));
        windowSum += window[index];
        windowSquare += window[index] * window[index];
    }
    json channels = json::array();
    double totalSquare = 0, totalPeak = 0;
    unsigned clipped = 0;
    for (unsigned channel = 0; channel < 2; ++channel) {
        double square = 0, peak = 0;
        for (unsigned index = 0; index < frames; ++index) {
            const double value = pcm[index * 2 + channel];
            peak = std::max(peak, std::abs(value));
            square += value * value;
            clipped += std::abs(value) >= 1 ? 1 : 0;
        }
        // Keep FFT arithmetic bounded even for finite but extremely overloaded PCM.
        const double spectralScale = std::max(1.0, peak);
        for (unsigned index = 0; index < frames; ++index) {
            time[index] = static_cast<float>(pcm[index * 2 + channel] /
                spectralScale * window[index]);
        }
        fft.fft(time.data(), frequency.data());
        for (unsigned bin = 0; bin < frames / 2; ++bin) {
            // Packed DC/Nyquist: the imaginary part of bin zero is not DC energy.
            power[bin] += (bin == 0 ? std::pow(frequency[0].real(), 2) :
                std::norm(frequency[bin])) * spectralScale * spectralScale / 2;
        }
        channels.push_back({{"name", channel == 0 ? "Left" : "Right"},
            {"peak_dbfs", powerDb(peak * peak)}, {"rms_dbfs", powerDb(square / frames)}});
        totalSquare += square;
        totalPeak = std::max(totalPeak, peak);
    }
    json spectrum = json::array(), bands = json::array();
    const double maximumHz = std::min(20000.0, sampleRate_ * 0.5 - sampleRate_ / double(frames));
    for (unsigned point = 0; point < 160; ++point) {
        const double hz = 20 * std::pow(maximumHz / 20, point / 159.0);
        const double position = hz * frames / sampleRate_;
        const auto low = std::min<unsigned>(frames / 2 - 2, static_cast<unsigned>(position));
        const double blend = std::clamp(position - low, 0.0, 1.0);
        const double amplitudeSquare = (power[low] * (1 - blend) + power[low + 1] * blend) *
            4 / (windowSum * windowSum);
        spectrum.push_back({hz, powerDb(amplitudeSquare)});
    }
    for (double center : {125, 250, 500, 1000, 2000, 4000}) {
        const double low = center / std::sqrt(2.0), high = center * std::sqrt(2.0);
        double energy = 0;
        for (unsigned bin = 1; bin < frames / 2; ++bin) {
            const double hz = bin * sampleRate_ / double(frames);
            if (hz >= low && hz < high) {
                energy += power[bin] * 2 / (frames * windowSquare);
            }
        }
        bands.push_back({{"frequency_hz", center}, {"low_hz", low}, {"high_hz", high},
            {"mean_square", energy}, {"dbfs", powerDb(energy)}});
    }
    return {{"spectrum", spectrum}, {"bands", bands}, {"channels", channels},
        {"peak_dbfs", powerDb(totalPeak * totalPeak)},
        {"rms_dbfs", powerDb(totalSquare / (frames * 2))},
        {"clip_samples_window", clipped}, {"silent", totalPeak < 1e-6},
        {"frequency_resolution_hz", sampleRate_ / double(frames)},
        {"spectrum_quantity", "sine_peak_amplitude_dbfs"},
        {"band_quantity", "window_corrected_mean_square_dbfs"}};
}

} // namespace audio
