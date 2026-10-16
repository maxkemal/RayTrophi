#include "Audio/AcousticRenderDebug.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "Backend/VulkanBackend.h"
#include "SceneSelection.h"
#include "scene_data.h"
#include "scene_ui.h"

#include <cmath>
#include <exception>

namespace audio {
AcousticRenderSource acousticRenderSource(UIContext& ctx) {
    AcousticRenderSource result{};
    if (ctx.selection.selected.type != SelectableType::AudioSource) {
        return result;
    }
    const auto data = ctx.scene.audio->snapshot();
    for (const auto& emitter : data.at("emitters")) {
        if (emitter.at("id") != ctx.selection.selected.audio_source_id ||
            !emitter.value("spatial", false) ||
            emitter.at("reaction").value("driver", "manual") != "manual") {
            continue;
        }
        try {
            const auto pose = worldPose(ctx, emitter);
            for (unsigned axis = 0; axis < 3; ++axis) {
                result[axis] = pose.at("position").at(axis).get<float>();
                result[axis + 8] = pose.at("direction").at(axis).get<float>();
            }
            result[3] = emitter.at("gain").get<float>();
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
    return result;
}

void updateAcousticRenderDebug(UIContext& ctx) {
    if (auto* backend = dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr)) {
        const auto source = ctx.render_settings.debug_view == kAcousticSurfaceDebugView ?
            acousticRenderSource(ctx) : AcousticRenderSource{};
        if (backend->setAcousticRenderSource(source) &&
            ctx.render_settings.debug_view == kAcousticSurfaceDebugView) {
            ctx.start_render = true;
        }
    }
}
} // namespace audio
