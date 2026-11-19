#include "RtIpcPaintWet.h"

#include "Api/RtApi.h"
#include "Api/RtApiPaintWet.h"

#include <stdexcept>
#include <vector>

using json = nlohmann::json;

namespace {

// Same names and error text as RtIpc.cpp's helpers: gen_ipc_descriptors.py
// reads parameter names and types from these call sites.
std::string requireString(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_string())
        throw std::runtime_error(std::string("missing or invalid string param: ") + key);
    return params[key].get<std::string>();
}

std::string optionalString(const json& params, const char* key, const std::string& default_val = "") {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_string()) throw std::runtime_error(std::string("invalid string param: ") + key);
    return params[key].get<std::string>();
}

int optionalInt(const json& params, const char* key, int default_val = 0) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_number_integer()) throw std::runtime_error(std::string("invalid int param: ") + key);
    return params[key].get<int>();
}

float optionalFloat(const json& params, const char* key, float default_val = 0.0f) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_number()) throw std::runtime_error(std::string("invalid number param: ") + key);
    return params[key].get<float>();
}

bool optionalBool(const json& params, const char* key, bool default_val = false) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_boolean()) throw std::runtime_error(std::string("invalid boolean param: ") + key);
    return params[key].get<bool>();
}

float numberElement(const json& value, const char* key) {
    if (!value.is_number()) throw std::runtime_error(std::string("non-numeric element in param: ") + key);
    return value.get<float>();
}

Vec3 requireVec3(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_array() || params[key].size() != 3)
        throw std::runtime_error(std::string("missing or invalid Vec3 param: ") + key);
    const auto& a = params[key];
    return Vec3(numberElement(a[0], key), numberElement(a[1], key), numberElement(a[2], key));
}

Vec3 optionalVec3(const json& params, const char* key, const Vec3& default_val = Vec3(0.0f, 0.0f, 0.0f)) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_array() || params[key].size() != 3)
        throw std::runtime_error(std::string("invalid Vec3 param: ") + key);
    const auto& a = params[key];
    return Vec3(numberElement(a[0], key), numberElement(a[1], key), numberElement(a[2], key));
}

json resultToJson(const rtapi::Result& r) {
    return r.ok ? json{{"ok", true}} : json{{"__error", r.error}};
}

json settingsJson(const rtapi::WetPaintSurfaceSettingsInfo& s) {
    return json{{"wetness", s.wetness}, {"lifetime_seconds", s.lifetime_seconds},
                {"diffusion", s.diffusion}, {"runoff", s.runoff},
                {"absorption", s.absorption}, {"drip_head", s.drip_head},
                {"falloff", s.falloff}, {"paint_load", s.paint_load},
                {"deposit_rate", s.deposit_rate}, {"pickup_rate", s.pickup_rate},
                {"brush_radius", s.brush_radius},
                {"oil", s.oil}, {"roughness_channel", s.roughness_channel}, {"height_normal_channel", s.height_normal_channel}, {"wet_roughness", s.wet_roughness}, {"dry_roughness", s.dry_roughness}, {"height_contribution", s.height_contribution}, {"normal_strength", s.normal_strength}};
}

json surfaceJson(const rtapi::WetPaintSurfaceRuntimeInfo& s) {
    return json{{"object", s.object}, {"material_id", s.material_id},
                {"resolution", s.resolution}, {"tiles", s.tiles},
                {"valid_texels", s.valid_texels}, {"gutter_texels", s.gutter_texels},
                {"triangles", s.triangles}, {"active_tiles", s.active_tiles},
                {"ticked_tiles", s.ticked_tiles}, {"stamped_texels", s.stamped_texels},
                {"sources_dropped", s.sources_dropped}, {"texture", s.texture},
                {"settings", settingsJson(s.settings)}};
}

json splatJson(const rtapi::WetPaintSplatInfo& s) {
    return json{{"consume_particles", s.consume_particles},
                {"particle_system", s.particle_system}, {"radius", s.radius},
                {"amount", s.amount}, {"min_speed", s.min_speed},
                {"facing_min", s.facing_min}, {"max_events_per_tick", s.max_events_per_tick}};
}

} // namespace

bool dispatchPaintWetIpc(const std::string& method, const json& params,
                         const RtIpcPaintWetEnqueue& enqueue, json& out_result) {
    if (method == "paint.wet_demo") {
        const std::string operation = optionalString(params, "operation", "status");
        const json config = params.value("config", json::object());
        out_result = enqueue([=](UIContext&) {
            std::string output;
            const auto result = rtapi::wetPaintDemoCommand(operation, config.dump(), output);
            return result.ok ? json::parse(output) : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "paint.wet_design") {
        out_result = enqueue([](UIContext&) {
            rtapi::GameDesignInfo info;
            const rtapi::Result r = rtapi::getGameDesign(info);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"design", json::parse(info.design_json)}, {"pending", info.pending},
                        {"status", info.status}};
        });
        return true;
    }
    if (method == "paint.wet_add_surface") {
        const std::string object = requireString(params, "object");
        const int material_id = optionalInt(params, "material_id", -1);
        const int resolution = optionalInt(params, "resolution", 1024);
        out_result = enqueue([=](UIContext&) {
            rtapi::WetPaintSurfaceRuntimeInfo info;
            const rtapi::Result r = rtapi::addWetPaintSurface(object, material_id, resolution, info);
            if (!r.ok) return json{{"__error", r.error}};
            // material_id -1 adds one surface per material slot: list them all,
            // so a caller can see every slot took part (not only the first).
            json out = surfaceJson(info);
            json slots = json::array();
            rtapi::WetPaintStatusInfo status;
            if (rtapi::getWetPaintStatus(status).ok) {
                for (const auto& surface : status.surface_list) {
                    if (surface.object == object &&
                        (material_id < 0 || surface.material_id == material_id)) {
                        slots.push_back(surfaceJson(surface));
                    }
                }
            }
            out["surfaces"] = std::move(slots);
            return out;
        });
        return true;
    }
    if (method == "paint.wet_resize_surfaces") {
        const int resolution = optionalInt(params, "resolution", 0);
        out_result = enqueue([=](UIContext&) {
            return resultToJson(rtapi::resizeWetPaintSurfaces(resolution));
        });
        return true;
    }
    if (method == "paint.wet_remove_surface") {
        const std::string object = requireString(params, "object");
        const int material_id = optionalInt(params, "material_id", -1);
        out_result = enqueue([=](UIContext&) {
            return resultToJson(rtapi::removeWetPaintSurface(object, material_id));
        });
        return true;
    }
    if (method == "paint.wet_clear_surfaces") {
        out_result = enqueue([](UIContext&) { return resultToJson(rtapi::clearWetPaintSurfaces()); });
        return true;
    }
    if (method == "paint.wet_status") {
        out_result = enqueue([](UIContext&) {
            rtapi::WetPaintStatusInfo s;
            const rtapi::Result r = rtapi::getWetPaintStatus(s);
            if (!r.ok) return json{{"__error", r.error}};
            json surfaces = json::array();
            for (const auto& surface : s.surface_list) surfaces.push_back(surfaceJson(surface));
            return json{{"device_ready", s.device_ready}, {"status", s.status},
                        {"ticks", s.ticks}, {"last_dt", s.last_dt},
                        {"surface_count", s.surfaces}, {"tiles", s.tiles},
                        {"dabs_applied", s.dabs_applied},
                        {"device_event_dispatches", s.device_event_dispatches},
                        {"host_events_applied", s.host_events_applied},
                        {"active_tiles", s.active_tiles}, {"ticked_tiles", s.ticked_tiles},
                        {"stamped_texels", s.stamped_texels},
                        {"sources_dropped", s.sources_dropped},
                        {"publish_device_copies", s.publish_device_copies},
                        {"pbr_dispatches", s.pbr_dispatches},
                        {"pbr_device_copies", s.pbr_device_copies},
                        {"pbr_buffer_bytes", s.pbr_buffer_bytes},
                        {"pbr_texture_bytes", s.pbr_texture_bytes},
                        {"publish_host_downloads", s.publish_host_downloads},
                        {"publish_bytes_downloaded", s.publish_bytes_downloaded},
                        {"geometry_rebakes", s.geometry_rebakes},
                        {"publish_mode", s.publish_mode}, {"last_tick_ms", s.last_tick_ms},
                        {"compute_context", s.compute_context},
                        {"idle_ticks", s.idle_ticks}, {"coverage_cached", s.coverage_cached},
                        {"event_sources", s.event_sources}, {"coverage_in_tick", s.coverage_in_tick},
                        {"stats_delayed", s.stats_delayed}, {"stats_readbacks_lost", s.stats_readbacks_lost},
                        {"splat", splatJson(s.splat)}, {"surfaces", surfaces}};
        });
        return true;
    }
    if (method == "paint.wet_get_settings") {
        const std::string object = requireString(params, "object");
        const int material_id = optionalInt(params, "material_id", -1);
        out_result = enqueue([=](UIContext&) {
            rtapi::WetPaintSurfaceSettingsInfo s;
            const rtapi::Result r = rtapi::getWetPaintSurfaceSettings(object, material_id, s);
            return r.ok ? settingsJson(s) : json{{"__error", r.error}};
        });
        return true;
    }
    if (method == "paint.wet_set_settings") {
        const std::string object = requireString(params, "object");
        const int material_id = optionalInt(params, "material_id", -1);
        // Every field optional: the current value is kept for a missing key.
        const bool has_wetness = params.contains("wetness");
        const float wetness = optionalFloat(params, "wetness", 0.65f);
        const bool has_lifetime = params.contains("lifetime_seconds");
        const float lifetime = optionalFloat(params, "lifetime_seconds", 12.0f);
        const bool has_diffusion = params.contains("diffusion");
        const float diffusion = optionalFloat(params, "diffusion", 0.24f);
        const bool has_runoff = params.contains("runoff");
        const float runoff = optionalFloat(params, "runoff", 0.35f);
        const bool has_absorption = params.contains("absorption");
        const float absorption = optionalFloat(params, "absorption", 0.18f);
        const bool has_drip = params.contains("drip_head");
        const float drip = optionalFloat(params, "drip_head", 0.45f);
        const bool has_falloff = params.contains("falloff");
        const float falloff = optionalFloat(params, "falloff", 0.65f);
        const bool has_load = params.contains("paint_load");
        const float load = optionalFloat(params, "paint_load", 0.85f);
        const bool has_deposit = params.contains("deposit_rate");
        const float deposit = optionalFloat(params, "deposit_rate", 0.65f);
        const bool has_pickup = params.contains("pickup_rate");
        const float pickup = optionalFloat(params, "pickup_rate", 0.35f);
        const bool has_radius = params.contains("brush_radius");
        const float radius = optionalFloat(params, "brush_radius", 5.0f);
        const bool has_oil = params.contains("oil");
        const bool oil = optionalBool(params, "oil", false);
        const bool has_roughness_channel = params.contains("roughness_channel");
        const bool roughness_channel = optionalBool(params, "roughness_channel", false);
        const bool has_height_normal_channel = params.contains("height_normal_channel");
        const bool height_normal_channel = optionalBool(params, "height_normal_channel", false);
        const bool has_wet_roughness = params.contains("wet_roughness");
        const float wet_roughness = optionalFloat(params, "wet_roughness", 0.12f);
        const bool has_dry_roughness = params.contains("dry_roughness");
        const float dry_roughness = optionalFloat(params, "dry_roughness", 0.30f);
        const bool has_height_contribution = params.contains("height_contribution");
        const float height_contribution = optionalFloat(params, "height_contribution", 0.35f);
        const bool has_normal_strength = params.contains("normal_strength");
        const float normal_strength = optionalFloat(params, "normal_strength", 0.5f);
        out_result = enqueue([=](UIContext&) {
            rtapi::WetPaintSurfaceSettingsInfo s;
            const rtapi::Result read = rtapi::getWetPaintSurfaceSettings(object, material_id, s);
            if (!read.ok) return resultToJson(read);
            if (has_wetness) s.wetness = wetness;
            if (has_lifetime) s.lifetime_seconds = lifetime;
            if (has_diffusion) s.diffusion = diffusion;
            if (has_runoff) s.runoff = runoff;
            if (has_absorption) s.absorption = absorption;
            if (has_drip) s.drip_head = drip;
            if (has_falloff) s.falloff = falloff;
            if (has_load) s.paint_load = load;
            if (has_deposit) s.deposit_rate = deposit;
            if (has_pickup) s.pickup_rate = pickup;
            if (has_radius) s.brush_radius = radius;
            if (has_oil) s.oil = oil;
            if (has_roughness_channel) s.roughness_channel = roughness_channel;
            if (has_height_normal_channel) s.height_normal_channel = height_normal_channel;
            if (has_wet_roughness) s.wet_roughness = wet_roughness;
            if (has_dry_roughness) s.dry_roughness = dry_roughness;
            if (has_height_contribution) s.height_contribution = height_contribution;
            if (has_normal_strength) s.normal_strength = normal_strength;
            return resultToJson(rtapi::setWetPaintSurfaceSettings(object, material_id, s));
        });
        return true;
    }
    if (method == "paint.wet_get_splat") {
        out_result = enqueue([](UIContext&) {
            rtapi::WetPaintSplatInfo s;
            const rtapi::Result r = rtapi::getWetPaintSplat(s);
            return r.ok ? splatJson(s) : json{{"__error", r.error}};
        });
        return true;
    }
    if (method == "paint.wet_set_splat") {
        const bool has_consume = params.contains("consume_particles");
        const bool consume = optionalBool(params, "consume_particles", true);
        const bool has_system = params.contains("particle_system");
        const std::string system = optionalString(params, "particle_system", "");
        const bool has_radius = params.contains("radius");
        const float radius = optionalFloat(params, "radius", 0.12f);
        const bool has_amount = params.contains("amount");
        const float amount = optionalFloat(params, "amount", 0.8f);
        const bool has_speed = params.contains("min_speed");
        const float min_speed = optionalFloat(params, "min_speed", 0.0f);
        const bool has_facing = params.contains("facing_min");
        const float facing = optionalFloat(params, "facing_min", 0.0f);
        const bool has_max = params.contains("max_events_per_tick");
        const int max_events = optionalInt(params, "max_events_per_tick", 4096);
        out_result = enqueue([=](UIContext&) {
            rtapi::WetPaintSplatInfo s;
            const rtapi::Result read = rtapi::getWetPaintSplat(s);
            if (!read.ok) return resultToJson(read);
            if (has_consume) s.consume_particles = consume;
            if (has_system) s.particle_system = system;
            if (has_radius) s.radius = radius;
            if (has_amount) s.amount = amount;
            if (has_speed) s.min_speed = min_speed;
            if (has_facing) s.facing_min = facing;
            if (has_max) {
                if (max_events < 1) return json{{"__error", "max_events_per_tick must be positive"}};
                s.max_events_per_tick = static_cast<uint32_t>(max_events);
            }
            return resultToJson(rtapi::setWetPaintSplat(s));
        });
        return true;
    }
    if (method == "paint.wet_dab") {
        const Vec3 position = requireVec3(params, "position");
        const float radius = optionalFloat(params, "radius", 0.1f);
        const Vec3 color = optionalVec3(params, "color", Vec3(1.0f, 0.2f, 0.2f));
        const float amount = optionalFloat(params, "amount", 0.8f);
        const bool has_normal = params.contains("normal");
        const Vec3 normal = optionalVec3(params, "normal", Vec3(0.0f, 1.0f, 0.0f));
        const bool facing = optionalBool(params, "facing", has_normal);
        out_result = enqueue([=](UIContext&) {
            rtapi::WetPaintDabInfo d;
            d.position[0] = position.x; d.position[1] = position.y; d.position[2] = position.z;
            d.radius = radius;
            d.color[0] = color.x; d.color[1] = color.y; d.color[2] = color.z;
            d.amount = amount;
            d.normal[0] = normal.x; d.normal[1] = normal.y; d.normal[2] = normal.z;
            d.facing = facing;
            return resultToJson(rtapi::wetPaintDab(d));
        });
        return true;
    }
    if (method == "paint.wet_reset") {
        out_result = enqueue([](UIContext&) { return resultToJson(rtapi::resetWetPaint()); });
        return true;
    }
    if (method == "paint.wet_coverage") {
        out_result = enqueue([](UIContext&) {
            rtapi::WetPaintCoverageInfo c;
            const rtapi::Result r = rtapi::getWetPaintCoverage(c);
            if (!r.ok) return json{{"__error", r.error}};
            json surfaces = json::array();
            for (const auto& s : c.surfaces) {
                surfaces.push_back({{"object", s.object}, {"material_id", s.material_id},
                                    {"covered_texels", s.covered_texels},
                                    {"valid_texels", s.valid_texels},
                                    {"wet_texels", s.wet_texels}, {"roughness_texels", s.roughness_texels},
                                         {"normal_texels", s.normal_texels}, {"ratio", s.ratio},
                                    {"mean_color", {s.mean_color[0], s.mean_color[1], s.mean_color[2]}}});
            }
            return json{{"covered_texels", c.covered_texels}, {"valid_texels", c.valid_texels},
                        {"ratio", c.ratio}, {"surfaces", surfaces}};
        });
        return true;
    }
    return false;
}
