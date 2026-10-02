/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiAtmosphere.cpp
* License:       MIT
* =========================================================================
*
* Atmosphere -> simulation coupling surface (docs/dev/ATMOSPHERE_SYSTEM.md
* Faz 2). Each consumer keeps its own value plus an `inherit_atmosphere`
* gate; what these functions report is always the EFFECTIVE value and where
* it came from ("atmosphere" | "local"), because a surface that only echoed
* the authored field would be blind to exactly the divergence the gate
* creates (CLAUDE.md: the panel-lies class).
*/

#include "RtApiInternal.h"

#include <cmath>
#include <string>

#include "Atmosphere/AtmosphereClimate.h"
#include "FoliageWindSystem.h"
#include "InstanceGroup.h"
#include "InstanceManager.h"
#include "ProjectManager.h"
#include "WaterSystem.h"

namespace rtapi {

Result getScatterWind(const std::string& group_id_or_name, ScatterWindInfo& out) {
    if (!g_ctx) return notBound();
    InstanceGroup* group = findScatterGroupHelper(group_id_or_name);
    if (!group) return Result::fail("scatter group not found: " + group_id_or_name);

    const auto& w = group->wind_settings;
    out = ScatterWindInfo{};
    out.enabled = w.enabled;
    out.inherit_atmosphere = w.inherit_atmosphere;
    out.wind_source = w.inherit_atmosphere ? "atmosphere" : "local";
    out.speed = w.speed;
    out.strength = w.strength;
    out.turbulence = w.turbulence;
    out.wave_size = w.wave_size;
    out.direction = w.direction;
    out.reference_wind_mps = InstanceGroup::kFoliageReferenceWindMps;

    const InstanceGroup::WindSettings eff = FoliageWindSystem::effectiveSettings(w);
    out.effective_speed = eff.speed;
    out.effective_strength = eff.strength;
    out.effective_direction = eff.direction;
    return Result::success();
}

Result setScatterWind(const std::string& group_id_or_name, const ScatterWindPatch& patch) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    InstanceGroup* group = findScatterGroupHelper(group_id_or_name);
    if (!group) return Result::fail("scatter group not found: " + group_id_or_name);

    // Validate everything first: a rejected call leaves the group untouched.
    const auto nonNegative = [](const std::optional<float>& v) {
        return !v || (std::isfinite(*v) && *v >= 0.0f);
    };
    if (!nonNegative(patch.speed)) return Result::fail("speed must be finite and >= 0");
    if (!nonNegative(patch.strength)) return Result::fail("strength must be finite and >= 0");
    if (!nonNegative(patch.turbulence)) return Result::fail("turbulence must be finite and >= 0");
    if (patch.wave_size && !(std::isfinite(*patch.wave_size) && *patch.wave_size >= 0.1f))
        return Result::fail("wave_size must be >= 0.1");
    Vec3 direction;
    if (patch.direction) {
        const Vec3 d = *patch.direction;
        const float h = std::sqrt(d.x * d.x + d.z * d.z);
        if (!(h > 1e-6f)) return Result::fail("direction needs a horizontal (x/z) component");
        direction = Vec3(d.x / h, 0.0f, d.z / h);
    }

    auto& w = group->wind_settings;
    const bool was_enabled = w.enabled;
    if (patch.enabled) w.enabled = *patch.enabled;
    if (patch.inherit_atmosphere) w.inherit_atmosphere = *patch.inherit_atmosphere;
    if (patch.speed) w.speed = *patch.speed;
    if (patch.strength) w.strength = *patch.strength;
    if (patch.turbulence) w.turbulence = *patch.turbulence;
    if (patch.wave_size) w.wave_size = *patch.wave_size;
    if (patch.direction) w.direction = direction;

    // Same side effects as the panel's "Enable Wind" toggle: switching off
    // must put the bent instances back to rest, or they stay frozen mid-sway.
    if (was_enabled && !w.enabled) {
        if (!group->initial_instances.empty() &&
            group->initial_instances.size() == group->instances.size()) {
            group->instances = group->initial_instances;
        }
        group->gpu_dirty = true;
        g_optix_rebuild_pending = true;
    }
    ProjectManager::getInstance().markModified();
    resetAccumulation();
    return Result::success();
}

namespace {

constexpr float kPi = 3.14159265358979f;

WaterSurface* findWaterSurface(const std::string& key) {
    auto& wm = WaterManager::getInstance();
    if (key.empty()) return nullptr;
    try {
        size_t idx = 0;
        const int id = std::stoi(key, &idx);
        if (idx == key.size()) {
            if (WaterSurface* s = wm.getWaterSurface(id)) return s;
        }
    } catch (...) {}
    for (auto& s : wm.getWaterSurfaces()) {
        if (s.name == key) return &s;
    }
    return wm.getWaterSurfaceByNodeName(key);
}

const char* waterTypeName(WaterSurface::Type t) {
    switch (t) {
        case WaterSurface::Type::Plane:  return "plane";
        case WaterSurface::Type::River:  return "river";
        case WaterSurface::Type::Custom: return "custom";
        case WaterSurface::Type::Lake:   return "lake";
    }
    return "plane";
}

} // namespace

Result getWaterWind(const std::string& surface, WaterWindInfo& out) {
    if (!g_ctx) return notBound();
    WaterSurface* s = findWaterSurface(surface);
    if (!s) return Result::fail("water surface not found: " + surface);
    const auto& p = s->params;
    out = WaterWindInfo{};
    out.surface = s->name;
    out.type = waterTypeName(s->type);
    out.inherit_atmosphere = p.inherit_atmosphere;
    out.wind_source = p.inherit_atmosphere ? "atmosphere" : "local";
    out.speed_mps = p.fft_wind_speed;
    out.direction_degrees = p.fft_wind_direction * 180.0f / kPi;
    out.effective_speed_mps = p.effectiveWindSpeed();
    out.effective_direction_degrees = p.effectiveWindDirection() * 180.0f / kPi;
    return Result::success();
}

Result setWaterWind(const std::string& surface, const WaterWindPatch& patch) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    WaterSurface* s = findWaterSurface(surface);
    if (!s) return Result::fail("water surface not found: " + surface);
    if (patch.speed_mps && !(std::isfinite(*patch.speed_mps) && *patch.speed_mps >= 0.0f))
        return Result::fail("speed_mps must be finite and >= 0");
    if (patch.direction_degrees && !std::isfinite(*patch.direction_degrees))
        return Result::fail("direction_degrees must be finite");
    if (patch.inherit_atmosphere && *patch.inherit_atmosphere &&
        s->type == WaterSurface::Type::River)
        return Result::fail("a river has no wind: its flow is authored, not inherited");

    auto& p = s->params;
    if (patch.inherit_atmosphere) p.inherit_atmosphere = *patch.inherit_atmosphere;
    if (patch.speed_mps) p.fft_wind_speed = *patch.speed_mps;
    if (patch.direction_degrees) p.fft_wind_direction = *patch.direction_degrees * kPi / 180.0f;
    // Same resync the water panel does after an edit: the wind lives in the GPU
    // material, and nothing else would push it.
    WaterManager::getInstance().syncSurfaceMaterial(s);
    g_materials_dirty = true;
    g_ctx->renderer.updateBackendMaterial(g_ctx->scene, s->material_id);
    ProjectManager::getInstance().markModified();
    resetAccumulation();
    return Result::success();
}

} // namespace rtapi
