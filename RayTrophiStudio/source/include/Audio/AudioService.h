#pragma once

#include "json.hpp"
#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace audio {
std::string pathUtf8(const std::filesystem::path& path);

// Main-thread authoring service. The device callback belongs exclusively to miniaudio.
// No scene geometry, renderer, UI or Python objects enter that callback.
class AudioService {
public:
    AudioService();
    ~AudioService();
    AudioService(const AudioService&) = delete;
    AudioService& operator=(const AudioService&) = delete;

    nlohmann::json invoke(const std::string& operation, const nlohmann::json& params);
    nlohmann::json snapshot() const;
    void restore(const nlohmann::json& data, const std::filesystem::path& base = {});
    nlohmann::json save(const std::filesystem::path& project) const;
    void clear();
    void listener(float x, float y, float z, float dx, float dy, float dz,
                  float ux, float uy, float uz);
    void resolvedPose(const std::string& id, const nlohmann::json& position,
                      const nlohmann::json& direction, bool valid);
    void tickReactions(float windMps, float rain);
    void invalidateSignal(const std::string& key);
    std::uint64_t generation() const;
    static bool isAuthoring(const std::string& operation);
    static const std::vector<std::string>& operations();
    static std::vector<std::string> parameters(const std::string& operation);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace audio
