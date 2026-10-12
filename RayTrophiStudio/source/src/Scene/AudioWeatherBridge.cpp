#include "Audio/AudioWeatherBridge.h"
#include "Audio/AudioService.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "renderer.h"
#include "World.h"
#include <algorithm>

namespace audio {
void tickWeatherReactions(UIContext& ctx) {
    const auto weather = ctx.renderer.world.getWeatherParams();
    const float wind = ctx.renderer.world.getClimate().wind_speed_mps;
    const float rain = weather.enabled && weather.type == WEATHER_RAIN ?
        std::clamp(weather.intensity, 0.0f, 1.0f) : 0.0f;
    ctx.scene.audio->tickReactions(wind, rain);
}
}
