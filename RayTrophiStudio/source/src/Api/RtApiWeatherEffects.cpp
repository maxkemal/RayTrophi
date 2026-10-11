#include "Api/RtApiWeatherEffects.h"
#include "RtApiInternal.h"
#include "ProjectManager.h"
#include "Renderer.h"
#include "World.h"

#include <cmath>

extern void markWorldDirty();

namespace rtapi {
namespace {

nlohmann::json weatherJson(const WeatherParams& w) {
    return {
        {"enabled", w.enabled != 0}, {"type", w.type},
        {"intensity", w.intensity}, {"density", w.density},
        {"precipitation_scale", w.precipitation_scale}, {"visibility", w.visibility},
        {"surface_wetness_output", w.surface_wetness_output},
        {"surface_accumulation_output", w.surface_accumulation_output},
        {"surface_settling_output", w.surface_settling_output},
        {"surface_height_output", w.surface_height_output},
        {"visual_mode", w.visual_mode},
        {"surface_response_enabled", w.surface_response_enabled != 0}
    };
}

bool unit(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

} // namespace

Result setWeatherEffects(const WeatherParams& w) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    if ((w.enabled != 0 && w.enabled != 1) || w.type < WEATHER_NONE || w.type > WEATHER_MIST ||
        w.visual_mode < WEATHER_VISUAL_OVERLAY || w.visual_mode > WEATHER_VISUAL_SURFACE_ONLY ||
        (w.surface_response_enabled != 0 && w.surface_response_enabled != 1)) {
        return Result::fail("weather enabled/surface flags must be 0/1, type 0..4, visual_mode 0..1");
    }
    if (!unit(w.intensity) || !unit(w.density) || !unit(w.visibility) ||
        !unit(w.surface_wetness_output) || !unit(w.surface_accumulation_output) ||
        !unit(w.surface_settling_output) || !unit(w.surface_height_output) ||
        !std::isfinite(w.precipitation_scale) || w.precipitation_scale < 0.1f ||
        w.precipitation_scale > 10.0f) {
        return Result::fail("weather strengths must be finite 0..1; precipitation_scale 0.1..10");
    }
    World& world = g_ctx->renderer.world;
    if (weatherJson(w) == weatherJson(world.getWeatherParams())) {
        return Result::success();
    }
    world.setWeatherParams(w);
    markWorldDirty();
    resetAccumulation();
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result getWeatherEffectsJson(std::string& out) {
    if (!g_ctx) {
        return notBound();
    }
    const WeatherParams w = g_ctx->renderer.world.getWeatherParams();
    auto json = weatherJson(w);
    json["wind_direction"] = {
        w.derived_wind_direction.x, w.derived_wind_direction.y, w.derived_wind_direction.z};
    json["wind_speed_mps"] = w.derived_wind_speed_mps;
    out = json.dump();
    return Result::success();
}

Result setWeatherEffectsJson(const std::string& patchText) {
    if (!g_ctx) {
        return notBound();
    }
    try {
        const auto patch = nlohmann::json::parse(patchText);
        if (!patch.is_object()) {
            return Result::fail("weather patch must be an object");
        }
        auto json = weatherJson(g_ctx->renderer.world.getWeatherParams());
        for (auto it = patch.begin(); it != patch.end(); ++it) {
            if (it.key() == "wind_direction" || it.key() == "wind_speed_mps") {
                continue; // climate-owned, accepted for get/edit/set round trips
            }
            if (!json.contains(it.key())) {
                return Result::fail("unknown weather field: " + it.key());
            }
            const bool flag = it.key() == "enabled" || it.key() == "surface_response_enabled";
            const bool integer = it.key() == "type" || it.key() == "visual_mode";
            if ((flag && !it.value().is_boolean()) ||
                (integer && !it.value().is_number_integer()) ||
                (!flag && !integer && !it.value().is_number())) {
                return Result::fail("invalid weather field type: " + it.key());
            }
            json[it.key()] = it.value();
        }
        WeatherParams w = g_ctx->renderer.world.getWeatherParams();
        w.enabled = json["enabled"].get<bool>() ? 1 : 0;
        w.type = json["type"].get<int>();
        w.intensity = json["intensity"].get<float>();
        w.density = json["density"].get<float>();
        w.precipitation_scale = json["precipitation_scale"].get<float>();
        w.visibility = json["visibility"].get<float>();
        w.surface_wetness_output = json["surface_wetness_output"].get<float>();
        w.surface_accumulation_output = json["surface_accumulation_output"].get<float>();
        w.surface_settling_output = json["surface_settling_output"].get<float>();
        w.surface_height_output = json["surface_height_output"].get<float>();
        w.visual_mode = json["visual_mode"].get<int>();
        w.surface_response_enabled = json["surface_response_enabled"].get<bool>() ? 1 : 0;
        return setWeatherEffects(w);
    } catch (const std::exception& e) {
        return Result::fail(std::string("invalid weather patch: ") + e.what());
    }
}

} // namespace rtapi
