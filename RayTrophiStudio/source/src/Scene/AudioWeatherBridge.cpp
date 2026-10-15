#include "Audio/AudioWeatherBridge.h"
#include "Audio/AudioService.h"
#include "Audio/AudioDomainBridge.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "renderer.h"
#include "World.h"
#include <algorithm>

namespace audio {
void tickWeatherReactions(UIContext& ctx) {
    if (ctx.scene.audio->transportPaused()) {
        return;
    }
    const float wind = ctx.renderer.world.getClimate().wind_speed_mps;
    // Rain is the atmosphere's (the manual WeatherParams is surface cover only
    // since 2026-10-14). The rate AT the camera needs a GPU read-back:
    // ATMOSPHERE_NEAR_PRECIP.md N2.
    float rain = 0.0f;
    const auto& clouds = ctx.renderer.world.getClouds();
    // Global ground-reaching cloud precipitation only; no claim of local shelter/core sampling.
    if (clouds.layers[0].enabled && clouds.layers[0].coverage > 0 &&
        !clouds.precipitation.snow && clouds.precipitation.ground_fraction >= 0.999f) {
        rain = std::max(rain, std::clamp(clouds.precipitation.rate_mm_h / 30.0f, 0.0f, 1.0f));
    }
    tickDomainAudioInputs(ctx);
    ctx.scene.audio->tickReactions(wind, rain);
}
}
