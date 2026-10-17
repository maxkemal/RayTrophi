#include "Audio/AudioSourceRuntime.h"
#include <algorithm>
#include <cmath>

namespace audio {
nlohmann::json resolveAudioSourceRuntime(const nlohmann::json& emitter,
    const nlohmann::json* resolvedPose, const ReactionOutput* control) {
    using json = nlohmann::json;
    const auto& reaction = emitter.at("reaction");
    const bool reactive = reaction.at("driver") != "manual";
    const bool ready = !reactive || control != nullptr;
    const auto& pose = resolvedPose ? *resolvedPose : emitter;
    const bool follow = reactive && control && reaction.at("follow_position").get<bool>() &&
        !control->position.is_null();
    const auto& position = follow ? control->position : pose.at("position");
    const auto& direction = pose.at("direction");
    const bool parented = !emitter.at("parent").get<std::string>().empty();
    bool valid = resolvedPose ? resolvedPose->value("valid", false) : !parented;
    for (const auto* vector : {&position, &direction}) {
        valid = valid && vector->is_array() && vector->size() == 3;
        for (const auto& component : *vector) {
            valid = valid && component.is_number() &&
                std::isfinite(component.get<float>());
        }
    }
    const float gain = reactive ? (control ? control->gain : 0) : 1;
    const float pitch = reactive && control ? control->pitch : 1;
    const float activity = reactive ? (control ? control->activity : 0) : 1;
    return {{"id", emitter.at("id")}, {"position", position}, {"direction", direction},
        {"valid", valid}, {"control_ready", ready}, {"driver", reaction.at("driver")},
        {"automatic", reaction.at("automatic")}, {"activity", activity},
        {"spatial", emitter.at("spatial")}, {"reaction_gain", gain},
        {"effective_gain", valid ? emitter.at("gain").get<float>() * gain : 0},
        {"effective_pitch", std::clamp(
            emitter.at("pitch").get<float>() * pitch, 0.125f, 8.0f)},
        {"position_origin", follow ? "reaction_world" :
            (emitter.at("parent").get<std::string>().empty() ? "authored_world" : "parent_world")}};
}
}
