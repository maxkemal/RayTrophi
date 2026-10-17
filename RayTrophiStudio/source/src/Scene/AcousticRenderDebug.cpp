#include "Audio/AcousticRenderDebug.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "SceneSelection.h"
#include "scene_data.h"
#include "scene_ui.h"

#include <cmath>
#include <exception>

namespace audio {
AcousticRenderSource acousticRenderSource(UIContext& ctx, const nlohmann::json& emitter) {
    AcousticRenderSource result{};
    if (!emitter.value("spatial", false)) {
        return result;
    }
    try {
        const auto pose = ctx.scene.audio->invoke("source_runtime", {{"id", emitter.at("id")}});
        if (!pose.at("valid").get<bool>()) {
            return {};
        }
        for (unsigned axis = 0; axis < 3; ++axis) {
            result[axis] = pose.at("position").at(axis).get<float>();
            result[axis + 8] = pose.at("direction").at(axis).get<float>();
        }
        result[3] = pose.at("effective_gain").get<float>();
        result[4] = emitter.at("min_distance").get<float>();
        result[5] = emitter.at("max_distance").get<float>();
        result[6] = emitter.at("rolloff").get<float>();
        result[7] = 1.0f;
        constexpr float halfDegreesToRadians = 0.00872664626f;
        result[11] = emitter.at("cone_inner").get<float>() * halfDegreesToRadians;
        result[12] = emitter.at("cone_outer").get<float>() * halfDegreesToRadians;
        result[13] = emitter.at("cone_gain").get<float>();
        for (const float value : result) {
            if (!std::isfinite(value)) {
                return {};
            }
        }
        return result;
    } catch (const std::exception&) {
        return {};
    }
}

AcousticRenderSource acousticRenderSource(UIContext& ctx) {
    if (ctx.selection.selected.type != SelectableType::AudioSource) {
        return {};
    }
    const auto data = ctx.scene.audio->snapshot();
    for (const auto& emitter : data.at("emitters")) {
        if (emitter.at("id") == ctx.selection.selected.audio_source_id) {
            return acousticRenderSource(ctx, emitter);
        }
    }
    return {};
}
} // namespace audio
