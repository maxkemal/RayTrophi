// GPU wet paint API (Api/RtApiPaintWet.h) over Paint::WetPaintGpu.

#include "RtApiInternal.h"
#include "Api/RtApiPaintWet.h"

#include "Paint/WetPaintGpu.h"
#include "Paint/BalloonPaintDemo.h"
#include "Paint/GameDesign.h"
#include "Paint/HiddenPaintSafety.h"
#include "globals.h"
#include "scene_data.h"

#include <algorithm>
#include <cmath>

namespace rtapi {
namespace {

Paint::WetPaintSurfaceSettings toCore(const WetPaintSurfaceSettingsInfo& in) {
    Paint::WetPaintSurfaceSettings s;
    s.wetness = in.wetness;
    s.lifetime_seconds = in.lifetime_seconds;
    s.diffusion = in.diffusion;
    s.runoff = in.runoff;
    s.absorption = in.absorption;
    s.drip_head = in.drip_head;
    s.falloff = in.falloff;
    s.paint_load = in.paint_load;
    s.deposit_rate = in.deposit_rate;
    s.pickup_rate = in.pickup_rate;
    s.brush_radius = in.brush_radius;
    s.oil = in.oil;
    s.roughness_channel = in.roughness_channel;
    s.height_normal_channel = in.height_normal_channel;
    s.reveal_opacity = in.reveal_opacity;
    s.wet_roughness = in.wet_roughness;
    s.dry_roughness = in.dry_roughness;
    s.height_contribution = in.height_contribution;
    s.normal_strength = in.normal_strength;
    return s;
}

WetPaintSurfaceSettingsInfo fromCore(const Paint::WetPaintSurfaceSettings& s) {
    WetPaintSurfaceSettingsInfo o;
    o.wetness = s.wetness;
    o.lifetime_seconds = s.lifetime_seconds;
    o.diffusion = s.diffusion;
    o.runoff = s.runoff;
    o.absorption = s.absorption;
    o.drip_head = s.drip_head;
    o.falloff = s.falloff;
    o.paint_load = s.paint_load;
    o.deposit_rate = s.deposit_rate;
    o.pickup_rate = s.pickup_rate;
    o.brush_radius = s.brush_radius;
    o.oil = s.oil;
    o.roughness_channel = s.roughness_channel;
    o.height_normal_channel = s.height_normal_channel;
    o.reveal_opacity = s.reveal_opacity;
    o.wet_roughness = s.wet_roughness;
    o.dry_roughness = s.dry_roughness;
    o.height_contribution = s.height_contribution;
    o.normal_strength = s.normal_strength;
    return o;
}

WetPaintSurfaceRuntimeInfo fromCore(const Paint::WetPaintSurfaceInfo& s) {
    WetPaintSurfaceRuntimeInfo o;
    o.object = s.object;
    o.material_id = s.material_id;
    o.resolution = s.resolution;
    o.tiles = s.tiles;
    o.valid_texels = s.valid_texels;
    o.gutter_texels = s.gutter_texels;
    o.triangles = s.triangles;
    o.active_tiles = s.active_tiles;
    o.ticked_tiles = s.ticked_tiles;
    o.stamped_texels = s.stamped_texels;
    o.sources_dropped = s.sources_dropped;
    o.texture = s.texture;
    o.settings = fromCore(s.settings);
    return o;
}

bool finite01(float v) { return std::isfinite(v) && v >= 0.0f && v <= 1.0f; }

Result validateSettings(const WetPaintSurfaceSettingsInfo& s) {
    if (!finite01(s.wetness) || !finite01(s.absorption) || !finite01(s.falloff) ||
        !finite01(s.paint_load) || !finite01(s.deposit_rate) || !finite01(s.pickup_rate)) {
        return Result::fail("wetness, absorption, falloff, paint_load, deposit_rate and pickup_rate must be in [0, 1]");
    }
    if (!std::isfinite(s.lifetime_seconds) || s.lifetime_seconds < 0.05f)
        return Result::fail("lifetime_seconds must be >= 0.05");
    if (!std::isfinite(s.diffusion) || s.diffusion < 0.0f || s.diffusion > 2.0f ||
        !std::isfinite(s.runoff) || s.runoff < 0.0f || s.runoff > 2.0f)
        return Result::fail("diffusion and runoff must be in [0, 2]");
    if (!std::isfinite(s.drip_head) || s.drip_head < 0.0f || s.drip_head > 1.5f)
        return Result::fail("drip_head must be in [0, 1.5]");
    if (!std::isfinite(s.brush_radius) || s.brush_radius <= 0.0f)
        return Result::fail("brush_radius must be > 0");
    if (!finite01(s.wet_roughness) || !finite01(s.dry_roughness) ||
        !finite01(s.height_contribution)) {
        return Result::fail("wet_roughness, dry_roughness, height_contribution must be in [0, 1]");
    }
    if (!std::isfinite(s.normal_strength) || s.normal_strength < 0.0f ||
        s.normal_strength > 8.0f) {
        return Result::fail("normal_strength must be in [0, 8]");
    }
    return Result::success();
}

} // namespace

Result validateWetPaintSurfaceSettings(const WetPaintSurfaceSettingsInfo& settings) {
    return validateSettings(settings);
}

Result addWetPaintSurface(const std::string& object, int material_id, int resolution,
                          WetPaintSurfaceRuntimeInfo& out) {
    if (!g_ctx) return notBound();
    std::string error;
    Paint::WetPaintGpu& paint = Paint::wetPaintGpu();
    if (!paint.addSurface(g_ctx->scene, object, material_id, resolution, error)) {
        return Result::fail(error);
    }
    // The paint texture was just bound to the material: both executables upload
    // it from their dirty-material servicing before the first device copy.
    g_materials_dirty = true;
    for (const auto& s : paint.surfaces()) {
        if (s.object == object && (material_id < 0 || s.material_id == material_id)) out = fromCore(s);
    }
    return Result::success();
}

Result removeWetPaintSurface(const std::string& object, int material_id) {
    if (!g_ctx) return notBound();
    if (!Paint::wetPaintGpu().removeSurface(object, material_id, &g_ctx->scene))
        return Result::fail("not a wet paint surface: " + object);
    return Result::success();
}

Result resizeWetPaintSurfaces(int resolution) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("cannot resize wet paint surfaces during a render job");
    }
    if (Paint::gameDesignPending()) {
        return Result::fail("game design is still loading; retry after the next tick");
    }
    std::string error;
    if (!Paint::resizeGamePaintSurfaces(g_ctx->scene, resolution, error)) {
        return Result::fail(error);
    }
    return Result::success();
}

Result clearWetPaintSurfaces() {
    if (!g_ctx) return notBound();
    Paint::wetPaintGpu().clear();
    return Result::success();
}

Result getWetPaintStatus(WetPaintStatusInfo& out) {
    if (!g_ctx) return notBound();
    if (!renderJobActive()) Paint::reconcileGameDesign(g_ctx->scene);
    const Paint::WetPaintGpu& paint = Paint::wetPaintGpu();
    const Paint::WetPaintStats& s = paint.stats();
    out = WetPaintStatusInfo{};
    out.device_ready = s.device_ready;
    out.status = s.status;
    out.ticks = s.ticks;
    out.last_dt = s.last_dt;
    out.surfaces = static_cast<uint32_t>(paint.surfaces().size());
    out.tiles = s.tiles;
    out.dabs_applied = s.dabs_applied;
    out.device_event_dispatches = s.device_event_dispatches;
    out.host_events_applied = s.host_events_applied;
    out.active_tiles = s.active_tiles;
    out.ticked_tiles = s.ticked_tiles;
    out.stamped_texels = s.stamped_texels;
    out.sources_dropped = s.sources_dropped;
    out.pbr_dispatches = s.pbr_dispatches;
    out.pbr_device_copies = s.pbr_device_copies;
    out.pbr_buffer_bytes = s.pbr_buffer_bytes;
    out.pbr_texture_bytes = s.pbr_texture_bytes;
    out.publish_device_copies = s.publish_device_copies;
    out.publish_host_downloads = s.publish_host_downloads;
    out.publish_bytes_downloaded = s.publish_bytes_downloaded;
    out.device_rebuilds = s.device_rebuilds;
    out.geometry_uploads = s.geometry_uploads;
    out.geometry_rebakes = s.geometry_rebakes;
    out.publish_mode = s.publish_mode;
    out.compute_context = s.compute_context;
    out.last_tick_ms = s.last_tick_ms;
    out.idle_ticks = s.idle_ticks;
    out.coverage_cached = s.coverage_cached;
    out.event_sources = s.event_sources;
    out.coverage_in_tick = s.coverage_in_tick;
    out.stats_delayed = s.stats_delayed;
    out.stats_readbacks_lost = s.stats_readbacks_lost;
    for (const auto& surface : paint.surfaces()) out.surface_list.push_back(fromCore(surface));
    const Paint::WetPaintSplatSettings& sp = paint.splatSettings();
    out.splat.consume_particles = sp.consume_particles;
    out.splat.particle_system = sp.particle_system;
    out.splat.radius = sp.radius;
    out.splat.amount = sp.amount;
    out.splat.min_speed = sp.min_speed;
    out.splat.facing_min = sp.facing_min;
    out.splat.max_events_per_tick = sp.max_events_per_tick;
    return Result::success();
}

Result getWetPaintSurfaceSettings(const std::string& object, int material_id,
                                  WetPaintSurfaceSettingsInfo& out) {
    if (!g_ctx) return notBound();
    for (const auto& s : Paint::wetPaintGpu().surfaces()) {
        if (s.object == object && (material_id < 0 || s.material_id == material_id)) {
            out = fromCore(s.settings);
            return Result::success();
        }
    }
    return Result::fail("not a wet paint surface: " + object);
}

Result setWetPaintSurfaceSettings(const std::string& object, int material_id,
                                  const WetPaintSurfaceSettingsInfo& settings) {
    if (!g_ctx) return notBound();
    Result r = validateSettings(settings);
    if (!r.ok) return r;
    if (!Paint::wetPaintGpu().setSurfaceSettings(object, material_id, toCore(settings)))
        return Result::fail("not a wet paint surface: " + object);
    return Result::success();
}

Result getWetPaintSplat(WetPaintSplatInfo& out) {
    WetPaintStatusInfo status;
    Result r = getWetPaintStatus(status);
    if (!r.ok) return r;
    out = status.splat;
    return Result::success();
}

Result setWetPaintSplat(const WetPaintSplatInfo& in) {
    if (!g_ctx) return notBound();
    if (!std::isfinite(in.radius) || in.radius <= 0.0f) return Result::fail("radius must be > 0 (metres)");
    if (!finite01(in.amount)) return Result::fail("amount must be in [0, 1]");
    if (!std::isfinite(in.min_speed) || in.min_speed < 0.0f) return Result::fail("min_speed must be >= 0");
    if (!std::isfinite(in.facing_min) || in.facing_min < -1.0f || in.facing_min > 1.0f)
        return Result::fail("facing_min must be in [-1, 1]");
    if (in.max_events_per_tick == 0) return Result::fail("max_events_per_tick must be > 0");
    if (!in.particle_system.empty()) {
        bool found = false;
        for (const auto& system : g_ctx->scene.particle_systems) found = found || system.name == in.particle_system;
        if (!found) return Result::fail("particle system not found: " + in.particle_system);
    }
    Paint::WetPaintSplatSettings s;
    s.consume_particles = in.consume_particles;
    s.particle_system = in.particle_system;
    s.radius = in.radius;
    s.amount = in.amount;
    s.min_speed = in.min_speed;
    s.facing_min = in.facing_min;
    s.max_events_per_tick = in.max_events_per_tick;
    Paint::wetPaintGpu().setSplatSettings(s);
    return Result::success();
}

Result wetPaintDab(const WetPaintDabInfo& in) {
    if (!g_ctx) return notBound();
    if (Paint::wetPaintGpu().empty()) return Result::fail("no wet paint surface (paint.wet_add_surface first)");
    for (float v : in.position) if (!std::isfinite(v)) return Result::fail("position must be finite");
    if (!std::isfinite(in.radius) || in.radius <= 0.0f) return Result::fail("radius must be > 0 (metres)");
    if (!finite01(in.amount)) return Result::fail("amount must be in [0, 1]");
    for (float v : in.color) if (!finite01(v)) return Result::fail("color components must be in [0, 1]");
    Paint::WetPaintDab d;
    for (int i = 0; i < 3; ++i) {
        d.position[i] = in.position[i];
        d.color[i] = in.color[i];
        d.normal[i] = std::isfinite(in.normal[i]) ? in.normal[i] : 0.0f;
    }
    d.radius = in.radius;
    d.amount = in.amount;
    d.facing = in.facing;
    Paint::wetPaintGpu().queueDab(d);
    return Result::success();
}

Result resetWetPaint() {
    if (!g_ctx) return notBound();
    Paint::clearBalloonPaintInFlight(g_ctx->scene);
    if (!Paint::wetPaintGpu().resetPaint()) return Result::fail("no wet paint surface");
    return Result::success();
}

Result getWetPaintCoverage(WetPaintCoverageInfo& out) {
    if (!g_ctx) return notBound();
    Paint::WetPaintCoverage c;
    if (!Paint::wetPaintGpu().coverage(g_ctx->scene, c))
        return Result::fail("wet paint coverage unavailable (" + Paint::wetPaintGpu().stats().status + ")");
    out = WetPaintCoverageInfo{};
    out.covered_texels = c.covered_texels;
    out.valid_texels = c.valid_texels;
    out.ratio = c.ratio;
    for (const auto& s : c.surfaces) {
        WetPaintCoverageInfo::Surface o;
        o.object = s.object;
        o.material_id = s.material_id;
        o.covered_texels = s.covered_texels;
        o.valid_texels = s.valid_texels;
        o.wet_texels = s.wet_texels;
        o.roughness_texels = s.roughness_texels;
        o.normal_texels = s.normal_texels;
        o.ratio = s.ratio;
        for (int i = 0; i < 3; ++i) o.mean_color[i] = s.mean_color[i];
        out.surfaces.push_back(o);
    }
    return Result::success();
}

Result wetPaintDemoCommand(const std::string& operation, const std::string& config_json,
                           std::string& output_json) {
    if (!g_ctx) {
        return notBound();
    }
    try {
        nlohmann::json output;
        std::string error;
        if (!renderJobActive()) Paint::reconcileGameDesign(g_ctx->scene);
        if (!Paint::balloonPaintCommand(g_ctx->scene, operation,
                                        nlohmann::json::parse(config_json), output, error)) {
            return Result::fail(error);
        }
        output_json = output.dump();
        return Result::success();
    } catch (const std::exception& e) {
        return Result::fail(e.what());
    }
}

Result getGameDesign(GameDesignInfo& out) {
    if (!g_ctx) {
        return notBound();
    }
    if (!renderJobActive()) Paint::reconcileGameDesign(g_ctx->scene);
    out.design_json = Paint::saveGameDesign().dump();
    out.pending = Paint::gameDesignPending();
    out.status = Paint::gameDesignStatus();
    return Result::success();
}

} // namespace rtapi
