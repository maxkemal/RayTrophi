/*
* rt.paint.wet_* - GPU wet paint (docs/dev/PLAYER_GAME_PAINT_ROOM.md step 3/4).
* Same names and keys as the paint.wet_* IPC methods (RtIpcPaintWet.cpp).
*/

#include "RtPythonPaintWet.h"
#include "Api/RtApiPaintWet.h"

#include "RtPyCommon.h"

#include <pybind11/stl.h>

#include <string>
#include <unordered_set>

namespace py = pybind11;

namespace {

py::dict settingsDict(const rtapi::WetPaintSurfaceSettingsInfo& s) {
    py::dict d;
    d["wetness"] = s.wetness; d["lifetime_seconds"] = s.lifetime_seconds;
    d["diffusion"] = s.diffusion; d["runoff"] = s.runoff; d["absorption"] = s.absorption;
    d["drip_head"] = s.drip_head; d["falloff"] = s.falloff; d["paint_load"] = s.paint_load;
    d["deposit_rate"] = s.deposit_rate; d["pickup_rate"] = s.pickup_rate;
    d["brush_radius"] = s.brush_radius; d["oil"] = s.oil;
    d["roughness_channel"] = s.roughness_channel;
    d["height_normal_channel"] = s.height_normal_channel;
    d["reveal_opacity"] = s.reveal_opacity;
    d["wet_roughness"] = s.wet_roughness;
    d["dry_roughness"] = s.dry_roughness;
    d["height_contribution"] = s.height_contribution;
    d["normal_strength"] = s.normal_strength;
    d["reveal_picture"] = s.reveal_picture;
    d["picture_tolerance"] = s.picture_tolerance;
    d["picture_hint"] = s.picture_hint;
    d["picture_outline"] = s.picture_outline;
    d["picture_blank"] = py::make_tuple(s.picture_blank[0], s.picture_blank[1], s.picture_blank[2]);
    return d;
}

py::dict surfaceDict(const rtapi::WetPaintSurfaceRuntimeInfo& s) {
    py::dict d;
    d["object"] = s.object; d["material_id"] = s.material_id; d["resolution"] = s.resolution;
    d["tiles"] = s.tiles; d["valid_texels"] = s.valid_texels; d["gutter_texels"] = s.gutter_texels;
    d["triangles"] = s.triangles; d["active_tiles"] = s.active_tiles;
    d["ticked_tiles"] = s.ticked_tiles; d["stamped_texels"] = s.stamped_texels;
    d["sources_dropped"] = s.sources_dropped; d["texture"] = s.texture;
    d["canvas_mode"] = s.canvas_overlay ? "overlay" : "composite";
    d["base_texture"] = s.base_texture;
    d["settings"] = settingsDict(s.settings);
    return d;
}

py::dict splatDict(const rtapi::WetPaintSplatInfo& s) {
    py::dict d;
    d["consume_particles"] = s.consume_particles; d["particle_system"] = s.particle_system;
    d["radius"] = s.radius; d["amount"] = s.amount; d["min_speed"] = s.min_speed;
    d["facing_min"] = s.facing_min; d["max_events_per_tick"] = s.max_events_per_tick;
    return d;
}

template <typename T>
void take(const py::kwargs& kw, const char* key, T& value) {
    if (kw.contains(key)) value = py::cast<T>(kw[key]);
}

} // namespace

namespace rtpy {

void registerPaintWetBindings(py::module_& paint) {
    paint.def("wet_demo", [](const std::string& operation, const py::dict& config) {
        const auto json = py::module_::import("json");
        std::string output;
        requireResult(rtapi::wetPaintDemoCommand(
            operation, py::cast<std::string>(json.attr("dumps")(config)), output));
        return json.attr("loads")(output);
    }, py::arg("operation") = "status", py::arg("config") = py::dict());
    paint.def("wet_design", [] {
        // What the next project save writes under "game_design" + apply state.
        const auto json = py::module_::import("json");
        rtapi::GameDesignInfo info;
        requireResult(rtapi::getGameDesign(info));
        py::dict d;
        d["design"] = json.attr("loads")(info.design_json);
        d["pending"] = info.pending;
        d["status"] = info.status;
        return d;
    });
    paint.def("wet_add_surface", [](const std::string& object, int material_id, int resolution) {
        rtapi::WetPaintSurfaceRuntimeInfo info;
        requireResult(rtapi::addWetPaintSurface(object, material_id, resolution, info));
        // Same shape as IPC paint.wet_add_surface: material_id -1 adds every
        // material slot, and "surfaces" lists each of them.
        py::dict d = surfaceDict(info);
        py::list slots;
        rtapi::WetPaintStatusInfo status;
        if (rtapi::getWetPaintStatus(status).ok) {
            for (const auto& surface : status.surface_list) {
                if (surface.object == object &&
                    (material_id < 0 || surface.material_id == material_id)) {
                    slots.append(surfaceDict(surface));
                }
            }
        }
        d["surfaces"] = slots;
        return d;
    }, py::arg("object"), py::arg("material_id") = -1, py::arg("resolution") = 1024);
    paint.def("wet_remove_surface", [](const std::string& object, int material_id) {
        requireResult(rtapi::removeWetPaintSurface(object, material_id));
    }, py::arg("object"), py::arg("material_id") = -1);
    paint.def("wet_resize_surfaces", [](int resolution) {
        requireResult(rtapi::resizeWetPaintSurfaces(resolution));
    }, py::arg("resolution"));
    paint.def("wet_clear_surfaces", [] { requireResult(rtapi::clearWetPaintSurfaces()); });
    paint.def("wet_status", [] {
        rtapi::WetPaintStatusInfo s;
        requireResult(rtapi::getWetPaintStatus(s));
        py::dict d;
        d["device_ready"] = s.device_ready; d["status"] = s.status; d["ticks"] = s.ticks;
        d["last_dt"] = s.last_dt; d["surface_count"] = s.surfaces; d["tiles"] = s.tiles;
        d["dabs_applied"] = s.dabs_applied; d["device_event_dispatches"] = s.device_event_dispatches;
        d["host_events_applied"] = s.host_events_applied; d["active_tiles"] = s.active_tiles;
        d["ticked_tiles"] = s.ticked_tiles; d["stamped_texels"] = s.stamped_texels;
        d["sources_dropped"] = s.sources_dropped;
        d["publish_device_copies"] = s.publish_device_copies;
        d["pbr_dispatches"] = s.pbr_dispatches;
        d["pbr_device_copies"] = s.pbr_device_copies;
        d["pbr_buffer_bytes"] = s.pbr_buffer_bytes;
        d["pbr_texture_bytes"] = s.pbr_texture_bytes;
        d["publish_host_downloads"] = s.publish_host_downloads;
        d["publish_bytes_downloaded"] = s.publish_bytes_downloaded;
        d["device_rebuilds"] = s.device_rebuilds;
        d["geometry_uploads"] = s.geometry_uploads;
        d["geometry_rebakes"] = s.geometry_rebakes;
        d["publish_mode"] = s.publish_mode; d["last_tick_ms"] = s.last_tick_ms;
        d["compute_context"] = s.compute_context;
        d["idle_ticks"] = s.idle_ticks; d["coverage_cached"] = s.coverage_cached;
        d["event_sources"] = s.event_sources; d["coverage_in_tick"] = s.coverage_in_tick;
        d["stats_delayed"] = s.stats_delayed; d["stats_readbacks_lost"] = s.stats_readbacks_lost;
        d["splat"] = splatDict(s.splat);
        py::list surfaces;
        for (const auto& surface : s.surface_list) surfaces.append(surfaceDict(surface));
        d["surfaces"] = std::move(surfaces);
        return d;
    });
    paint.def("wet_get_settings", [](const std::string& object, int material_id) {
        rtapi::WetPaintSurfaceSettingsInfo s;
        requireResult(rtapi::getWetPaintSurfaceSettings(object, material_id, s));
        return settingsDict(s);
    }, py::arg("object"), py::arg("material_id") = -1);
    paint.def("wet_set_settings", [](const std::string& object, int material_id, const py::kwargs& kw) {
        static const std::unordered_set<std::string> allowed = {
            "wetness", "lifetime_seconds", "diffusion", "runoff", "absorption",
            "drip_head", "falloff", "paint_load", "deposit_rate", "pickup_rate",
            "brush_radius", "oil", "roughness_channel", "height_normal_channel",
            "reveal_opacity", "wet_roughness", "dry_roughness", "height_contribution",
            "normal_strength", "reveal_picture", "picture_tolerance", "picture_hint",
            "picture_outline", "picture_blank"
        };
        for (const auto& item : kw) {
            const auto key = py::cast<std::string>(item.first);
            if (allowed.count(key) == 0) {
                throw py::type_error("Unknown wet paint setting: " + key);
            }
        }
        if (kw.contains("reveal_opacity") &&
            !py::isinstance<py::bool_>(kw["reveal_opacity"])) {
            throw py::type_error("reveal_opacity must be a bool");
        }
        if (kw.contains("reveal_picture") &&
            !py::isinstance<py::str>(kw["reveal_picture"])) {
            throw py::type_error("reveal_picture must be a str");
        }
        rtapi::WetPaintSurfaceSettingsInfo s;
        requireResult(rtapi::getWetPaintSurfaceSettings(object, material_id, s));
        take(kw, "wetness", s.wetness); take(kw, "lifetime_seconds", s.lifetime_seconds);
        take(kw, "diffusion", s.diffusion); take(kw, "runoff", s.runoff);
        take(kw, "absorption", s.absorption); take(kw, "drip_head", s.drip_head);
        take(kw, "falloff", s.falloff); take(kw, "paint_load", s.paint_load);
        take(kw, "deposit_rate", s.deposit_rate); take(kw, "pickup_rate", s.pickup_rate);
        take(kw, "brush_radius", s.brush_radius);
        take(kw, "oil", s.oil);
        take(kw, "roughness_channel", s.roughness_channel);
        take(kw, "height_normal_channel", s.height_normal_channel);
        take(kw, "reveal_opacity", s.reveal_opacity);
        take(kw, "wet_roughness", s.wet_roughness);
        take(kw, "dry_roughness", s.dry_roughness);
        take(kw, "height_contribution", s.height_contribution);
        take(kw, "normal_strength", s.normal_strength);
        take(kw, "reveal_picture", s.reveal_picture);
        take(kw, "picture_tolerance", s.picture_tolerance);
        take(kw, "picture_hint", s.picture_hint);
        take(kw, "picture_outline", s.picture_outline);
        if (kw.contains("picture_blank")) {
            const py::object value = kw["picture_blank"];
            if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value) ||
                py::len(value) != 3) {
                throw py::type_error("picture_blank must be a sequence of 3 floats");
            }
            const py::sequence seq = py::reinterpret_borrow<py::sequence>(value);
            for (int i = 0; i < 3; ++i) s.picture_blank[i] = py::cast<float>(seq[i]);
        }
        requireResult(rtapi::setWetPaintSurfaceSettings(object, material_id, s));
    }, py::arg("object"), py::arg("material_id") = -1);
    paint.def("wet_get_splat", [] {
        rtapi::WetPaintSplatInfo s;
        requireResult(rtapi::getWetPaintSplat(s));
        return splatDict(s);
    });
    paint.def("wet_set_splat", [](const py::kwargs& kw) {
        rtapi::WetPaintSplatInfo s;
        requireResult(rtapi::getWetPaintSplat(s));
        take(kw, "consume_particles", s.consume_particles);
        take(kw, "particle_system", s.particle_system);
        take(kw, "radius", s.radius); take(kw, "amount", s.amount);
        take(kw, "min_speed", s.min_speed); take(kw, "facing_min", s.facing_min);
        take(kw, "max_events_per_tick", s.max_events_per_tick);
        requireResult(rtapi::setWetPaintSplat(s));
    });
    paint.def("wet_dab", [](const py::handle& position, float radius, const py::handle& color,
                            float amount, const py::object& normal) {
        rtapi::WetPaintDabInfo d;
        const Vec3 p = vec3FromPython(position);
        const Vec3 c = vec3FromPython(color);
        d.position[0] = p.x; d.position[1] = p.y; d.position[2] = p.z;
        d.color[0] = c.x; d.color[1] = c.y; d.color[2] = c.z;
        d.radius = radius;
        d.amount = amount;
        if (!normal.is_none()) {
            const Vec3 n = vec3FromPython(normal);
            d.normal[0] = n.x; d.normal[1] = n.y; d.normal[2] = n.z;
            d.facing = true;
        }
        requireResult(rtapi::wetPaintDab(d));
    }, py::arg("position"), py::arg("radius") = 0.1f,
       py::arg("color") = py::make_tuple(1.0f, 0.2f, 0.2f), py::arg("amount") = 0.8f,
       py::arg("normal") = py::none());
    paint.def("wet_reset", [] { requireResult(rtapi::resetWetPaint()); });
    paint.def("wet_coverage", [] {
        rtapi::WetPaintCoverageInfo c;
        requireResult(rtapi::getWetPaintCoverage(c));
        py::dict d;
        d["covered_texels"] = c.covered_texels; d["valid_texels"] = c.valid_texels; d["ratio"] = c.ratio;
        py::list surfaces;
        for (const auto& s : c.surfaces) {
            py::dict e;
            e["object"] = s.object; e["material_id"] = s.material_id;
            e["covered_texels"] = s.covered_texels; e["valid_texels"] = s.valid_texels;
            e["wet_texels"] = s.wet_texels; e["ratio"] = s.ratio;
            e["roughness_texels"] = s.roughness_texels;
            e["normal_texels"] = s.normal_texels;
            e["picture_texels"] = s.picture_texels;
            e["picture_revealed_texels"] = s.picture_revealed_texels;
            e["picture_ratio"] = s.picture_ratio;
            e["mean_color"] = py::make_tuple(s.mean_color[0], s.mean_color[1], s.mean_color[2]);
            surfaces.append(std::move(e));
        }
        d["surfaces"] = std::move(surfaces);
        return d;
    });
}

} // namespace rtpy
