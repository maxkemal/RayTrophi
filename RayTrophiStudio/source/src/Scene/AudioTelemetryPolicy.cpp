#include "Audio/AudioTelemetryPolicy.h"
#include <stdexcept>

namespace audio {
using json = nlohmann::json;
json validateTelemetryPolicy(const json& value) {
    if (!value.is_object()) {
        throw std::invalid_argument("Audio telemetry policy must be an object");
    }
    for (auto entry = value.begin(); entry != value.end(); ++entry) {
        if (entry.key() != "enabled") {
            throw std::invalid_argument("Unknown audio telemetry policy parameter");
        }
    }
    const auto enabled = value.value("enabled", json(false));
    if (!enabled.is_boolean()) {
        throw std::invalid_argument("Audio telemetry enabled must be boolean");
    }
    return {{"enabled", enabled}};
}

json updateTelemetryPolicy(const json& current, const json& patch) {
    validateTelemetryPolicy(patch);
    auto candidate = current;
    candidate.update(patch);
    return validateTelemetryPolicy(candidate);
}
}
