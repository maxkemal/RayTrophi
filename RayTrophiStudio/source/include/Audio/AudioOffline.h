#pragma once
#include "json.hpp"
#include <memory>
#include <string>
#include <vector>

namespace audio {
class AudioOfflineJob {
public:
    AudioOfflineJob();
    ~AudioOfflineJob();
    nlohmann::json start(const nlohmann::json& scene, const nlohmann::json& params);
    void tick();
    void cancel();
    nlohmann::json status() const;
    static std::vector<std::string> parameters();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace audio
