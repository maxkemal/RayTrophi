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
    const auto weather = ctx.renderer.world.getWeatherParams();
    const float wind = ctx.renderer.world.getClimate().wind_speed_mps;
    float rain = weather.enabled && weather.type == WEATHER_RAIN ?
        std::clamp(weather.intensity, 0.0f, 1.0f) : 0.0f;
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
