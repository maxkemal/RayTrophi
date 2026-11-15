#include "Audio/AudioRuntime.h"

#include "Audio/AcousticScene.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "Audio/AudioThunderBridge.h"
#include "Audio/AudioWeatherBridge.h"
#include "Camera.h"
#include "UIContext.h"
#include "scene_data.h"

namespace audio {

void tickRuntime(UIContext& ctx, double transport_seconds, bool playing) {
    if (!ctx.scene.audio) {
        return;
    }
    ctx.scene.audio->syncTransport(transport_seconds, playing);
    syncScene(ctx);
    if (!ctx.scene.camera) {
        tickSceneAcoustics(ctx);
        tickCloudThunder(ctx);
        return;
    }
    const auto& camera = *ctx.scene.camera;
    const Vec3 direction = camera.lookat - camera.lookfrom;
    ctx.scene.audio->listener(camera.lookfrom.x, camera.lookfrom.y, camera.lookfrom.z,
                              direction.x, direction.y, direction.z,
                              camera.vup.x, camera.vup.y, camera.vup.z);
    tickWeatherReactions(ctx);
    tickSceneAcoustics(ctx);
    tickCloudThunder(ctx);
}

} // namespace audio
