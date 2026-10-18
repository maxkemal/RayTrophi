#pragma once

#include "Audio/AudioPcm.h"
#include <functional>
#include <memory>

namespace audio {
using AudioCancelCheck = std::function<void()>;

struct AudioDesignResult {
    AudioPcm original;
    AudioPcm processed;
    nlohmann::json report;
};

nlohmann::json validateSoundDesignRecipe(const nlohmann::json& input);
AudioDesignResult prepareSoundDesign(const nlohmann::json& input,
                                    const nlohmann::json& assets,
                                    const AudioCancelCheck& check = {});

// One worker and one replaceable pending request. Tasks own immutable snapshots;
// neither a task nor its result may reference scene, UI or native playback objects.
class AudioDesignJobs {
public:
    struct Result {
        std::uint64_t id = 0;
        std::uint64_t revision = 0;
        nlohmann::json assets;
        nlohmann::json files;
        AudioDesignResult design;
        nlohmann::json analysis;
        // Optional worker-owned artifact cleanup, including cancelled/stale results.
        std::shared_ptr<void> lifetime;
    };
    using Task = std::function<Result(const AudioCancelCheck&)>;

    AudioDesignJobs();
    ~AudioDesignJobs();
    void start(const std::string& kind, std::uint64_t revision, Task task);
    void cancel();
    nlohmann::json status() const;
    std::unique_ptr<Result> take();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace audio
