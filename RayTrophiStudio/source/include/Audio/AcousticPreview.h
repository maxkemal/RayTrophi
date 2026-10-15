#pragma once
#include "json.hpp"
#include <map>
#include <string>

namespace audio {
// Session-only, broadband viewport audition. Never serialized into authored audio.
class AcousticPreview {
public:
    nlohmann::json configure(const nlohmann::json& emitter, const nlohmann::json& params);
    nlohmann::json status() const;
    void update(const std::string& id, const nlohmann::json& report);
    void forget(const std::string& id);
    float gain(const nlohmann::json& emitter) const;
private:
    std::map<std::string, nlohmann::json> entries_;
};
}
