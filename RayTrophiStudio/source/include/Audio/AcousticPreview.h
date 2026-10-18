#pragma once
#include "json.hpp"
#include <map>
#include <string>

namespace audio {
bool supportsDiffuseSource(const nlohmann::json& emitter);
// Runtime acoustic results and session A/B bypass. Authored settings live in scene data.
class AcousticPreview {
public:
    nlohmann::json configure(const nlohmann::json& emitter, const nlohmann::json& params);
    nlohmann::json batch(const nlohmann::json& emitters, const nlohmann::json& params);
    nlohmann::json status() const;
    void update(const std::string& id, const nlohmann::json& report);
    void forget(const std::string& id);
    float gain(const nlohmann::json& emitter) const;
    nlohmann::json energy(const nlohmann::json& emitter) const;
    nlohmann::json reflections(const nlohmann::json& emitter) const;
private:
    std::map<std::string, nlohmann::json> entries_;
};
}
