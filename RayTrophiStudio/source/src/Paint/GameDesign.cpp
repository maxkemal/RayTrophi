#include "Paint/GameDesign.h"

#include "Paint/BalloonPaintDemo.h"
#include "Paint/WetPaintGpu.h"
#include "globals.h"
#include "scene_data.h"

#include <algorithm>
#include <cmath>

namespace Paint {
namespace {

using json = nlohmann::json;

json pending_design;          // null: nothing queued
std::string apply_status;
// The balloon demo waits for the droplet colliders' SDF cooks (started by the
// project loader): its pools are GPU-required and warm up against them.
json pending_demo;            // null: nothing waiting
int pending_demo_ticks = 0;
constexpr int kDemoWaitTicks = 60 * 60;   // ~1 minute of frames, then report

json settingsToJson(const WetPaintSurfaceSettings& s) {
    return {{"wetness", s.wetness}, {"lifetime_seconds", s.lifetime_seconds},
            {"diffusion", s.diffusion}, {"runoff", s.runoff}, {"absorption", s.absorption},
            {"drip_head", s.drip_head}, {"falloff", s.falloff}, {"paint_load", s.paint_load},
            {"deposit_rate", s.deposit_rate}, {"pickup_rate", s.pickup_rate},
            {"brush_radius", s.brush_radius}, {"oil", s.oil},
            {"roughness_channel", s.roughness_channel},
            {"height_normal_channel", s.height_normal_channel},
            {"wet_roughness", s.wet_roughness}, {"dry_roughness", s.dry_roughness},
            {"height_contribution", s.height_contribution},
            {"normal_strength", s.normal_strength}};
}

float number(const json& j, const char* key, float fallback, float lo, float hi) {
    if (!j.is_object() || !j.contains(key) || !j[key].is_number()) return fallback;
    const float value = j[key].get<float>();
    return std::isfinite(value) ? std::clamp(value, lo, hi) : fallback;
}

bool flag(const json& j, const char* key, bool fallback) {
    return j.is_object() && j.contains(key) && j[key].is_boolean() ? j[key].get<bool>() : fallback;
}

// Same ranges as rtapi::setWetPaintSurfaceSettings; a hand-edited file is
// clamped, not refused.
WetPaintSurfaceSettings settingsFromJson(const json& j) {
    WetPaintSurfaceSettings s;
    s.wetness = number(j, "wetness", s.wetness, 0.0f, 1.0f);
    s.lifetime_seconds = number(j, "lifetime_seconds", s.lifetime_seconds, 0.05f, 600.0f);
    s.diffusion = number(j, "diffusion", s.diffusion, 0.0f, 2.0f);
    s.runoff = number(j, "runoff", s.runoff, 0.0f, 2.0f);
    s.absorption = number(j, "absorption", s.absorption, 0.0f, 1.0f);
    s.drip_head = number(j, "drip_head", s.drip_head, 0.0f, 1.5f);
    s.falloff = number(j, "falloff", s.falloff, 0.0f, 1.0f);
    s.paint_load = number(j, "paint_load", s.paint_load, 0.0f, 1.0f);
    s.deposit_rate = number(j, "deposit_rate", s.deposit_rate, 0.0f, 1.0f);
    s.pickup_rate = number(j, "pickup_rate", s.pickup_rate, 0.0f, 1.0f);
    s.brush_radius = number(j, "brush_radius", s.brush_radius, 0.01f, 256.0f);
    s.oil = flag(j, "oil", s.oil);
    s.roughness_channel = flag(j, "roughness_channel", s.roughness_channel);
    s.height_normal_channel = flag(j, "height_normal_channel", s.height_normal_channel);
    s.wet_roughness = number(j, "wet_roughness", s.wet_roughness, 0.0f, 1.0f);
    s.dry_roughness = number(j, "dry_roughness", s.dry_roughness, 0.0f, 1.0f);
    s.height_contribution = number(j, "height_contribution", s.height_contribution, 0.0f, 1.0f);
    s.normal_strength = number(j, "normal_strength", s.normal_strength, 0.0f, 8.0f);
    return s;
}

} // namespace

json saveGameDesign() {
    auto& paint = wetPaintGpu();
    const json demo = balloonPaintDesign();
    if (paint.empty() && demo.is_null()) {
        return nullptr;
    }
    json surfaces = json::array();
    for (const auto& s : paint.surfaces()) {
        surfaces.push_back({{"object", s.object}, {"material_id", s.material_id},
                            {"resolution", s.resolution}, {"settings", settingsToJson(s.settings)}});
    }
    const auto& splat = paint.splatSettings();
    json design = {{"version", 1},
                   {"wet_paint", {{"surfaces", surfaces},
                                  {"splat", {{"consume_particles", splat.consume_particles},
                                             {"particle_system", splat.particle_system},
                                             {"radius", splat.radius}, {"amount", splat.amount},
                                             {"min_speed", splat.min_speed},
                                             {"facing_min", splat.facing_min},
                                             {"max_events_per_tick", splat.max_events_per_tick}}}}}};
    if (!demo.is_null()) {
        design["balloon_demo"] = demo;
    }
    return design;
}

void queueGameDesign(const json& design) {
    pending_design = design.is_object() ? design : json();
    pending_demo = json();
    apply_status.clear();
}

void resetGameDesign() {
    pending_design = json();
    pending_demo = json();
    apply_status.clear();
}

bool gameDesignPending() {
    return pending_design.is_object() || pending_demo.is_object();
}

namespace {

void tickPendingDemo(SceneData& scene) {
    std::string waiting;
    if (!balloonPaintCollidersReady(scene, waiting)) {
        if (++pending_demo_ticks < kDemoWaitTicks) {
            apply_status = "waiting for collider SDF: " + waiting;
            return;
        }
        apply_status = "balloon demo not prepared: collider SDF never finished (" + waiting + ")";
        SCENE_LOG_WARN("[GameDesign] " + apply_status);
        pending_demo = json();
        return;
    }
    const json config = std::move(pending_demo);
    pending_demo = json();
    json output;
    std::string error;
    if (!balloonPaintCommand(scene, "configure", config, output, error)) {
        apply_status = "balloon demo: " + error;
        SCENE_LOG_WARN("[GameDesign] " + apply_status);
    } else {
        apply_status = "applied: " + std::to_string(wetPaintGpu().surfaces().size()) +
                       " paint surface(s), balloon demo prepared";
    }
}

} // namespace

std::string gameDesignStatus() {
    return apply_status;
}

void tickGameDesign(SceneData& scene) {
    if (pending_demo.is_object()) {
        tickPendingDemo(scene);
        return;
    }
    if (!pending_design.is_object()) {
        return;
    }
    const json design = std::move(pending_design);
    pending_design = json();
    std::string problems;
    const auto note = [&problems](const std::string& text) {
        if (!problems.empty()) problems += "; ";
        problems += text;
    };
    auto& paint = wetPaintGpu();
    size_t added = 0;
    try {
        const json wet = design.value("wet_paint", json::object());
        const json surfaces = wet.value("surfaces", json::array());
        if (surfaces.is_array() && !surfaces.empty()) {
            // The document replaces whatever a previous project left behind.
            paint.clear();
            for (const auto& entry : surfaces) {
                if (!entry.is_object() || !entry.contains("object") || !entry["object"].is_string()) {
                    note("surface entry without an object name");
                    continue;
                }
                const std::string object = entry["object"].get<std::string>();
                const int material = entry.value("material_id", -1);
                const int resolution = std::clamp(entry.value("resolution", 1024), 32, 4096);
                std::string error;
                if (!paint.addSurface(scene, object, material, resolution, error)) {
                    // A renamed or deleted object: say so, keep the rest.
                    note(object + ": " + error);
                    continue;
                }
                paint.setSurfaceSettings(object, material,
                                         settingsFromJson(entry.value("settings", json::object())));
                ++added;
            }
            g_materials_dirty = true;
        }
        if (wet.contains("splat") && wet["splat"].is_object()) {
            const json& j = wet["splat"];
            WetPaintSplatSettings splat;
            splat.consume_particles = flag(j, "consume_particles", splat.consume_particles);
            if (j.contains("particle_system") && j["particle_system"].is_string()) {
                splat.particle_system = j["particle_system"].get<std::string>();
            }
            splat.radius = number(j, "radius", splat.radius, 0.001f, 10.0f);
            splat.amount = number(j, "amount", splat.amount, 0.0f, 1.0f);
            splat.min_speed = number(j, "min_speed", splat.min_speed, 0.0f, 1000.0f);
            splat.facing_min = number(j, "facing_min", splat.facing_min, -1.0f, 1.0f);
            if (j.contains("max_events_per_tick") && j["max_events_per_tick"].is_number_unsigned()) {
                splat.max_events_per_tick =
                    std::clamp<uint32_t>(j["max_events_per_tick"].get<uint32_t>(), 1u, 1u << 20);
            }
            paint.setSplatSettings(splat);
        }
        if (design.contains("balloon_demo") && design["balloon_demo"].is_object()) {
            // Configured on a later tick, once every droplet collider is cooked.
            pending_demo = design["balloon_demo"];
            pending_demo_ticks = 0;
        }
    } catch (const std::exception& e) {
        note(e.what());
    }
    apply_status = problems.empty()
        ? "applied: " + std::to_string(added) + " paint surface(s)"
        : "applied " + std::to_string(added) + " paint surface(s), problems: " + problems;
    if (!problems.empty()) {
        SCENE_LOG_WARN("[GameDesign] " + apply_status);
    }
}

} // namespace Paint
