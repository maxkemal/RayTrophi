#pragma once
#include "Audio/PhysicalAudioDsp.h"
#include "Audio/AudioDesignJobs.h"
#include <vector>

namespace audio {
// Construct/preroll on a worker or main thread, then transfer exclusive rendering
// ownership to the callback. Render/reset never parse JSON, allocate or lock.
class PhysicalAudioSchedule {
public:
    explicit PhysicalAudioSchedule(const nlohmann::json& validatedTimeline);
    float render(float* output, std::size_t count);
    void seek(std::uint64_t frame, const AudioCancelCheck& check = {});
    std::uint64_t cursor() const {
        return cursor_;
    }
    std::uint64_t frames() const {
        return frames_;
    }

private:
    struct Control {
        std::uint64_t frame;
        float activity;
        float gain;
        bool paused;
        bool reset;
    };
    struct Event {
        std::uint64_t frame;
        float impulse;
    };
    PhysicalAudioDsp initial_;
    PhysicalAudioDsp dsp_;
    std::vector<Control> controls_;
    std::vector<Event> events_;
    std::uint64_t frames_ = 0;
    std::uint64_t cursor_ = 0;
    std::size_t control_ = 0;
    std::size_t event_ = 0;
    float activity_ = 0;
    float gain_ = 1;
    bool paused_ = false;
};
}
