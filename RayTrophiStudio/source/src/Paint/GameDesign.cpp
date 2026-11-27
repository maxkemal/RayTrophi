#include "Paint/GameDesign.h"
#include "Paint/HiddenGameObjects.h"
#ifndef RT_PLAYER
#include "Core/StudioPlayPaint.h"
#endif

#include "Paint/BalloonPaintDemo.h"
#include "Paint/BalloonPlayerDefaults.h"
#include "Paint/GameDesignStartupHelper.h"
#include "Paint/WetPaintGpu.h"
#include "Paint/MeshPaintAdapter.h"
#include "TriangleMesh.h"
#include "Triangle.h"
#include "globals.h"
#include "scene_data.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_set>

namespace Paint {
namespace {

using json = nlohmann::json;

json pending_design;          // null: nothing queued
std::string apply_status;
// The balloon demo waits for the droplet colliders' SDF cooks (started by the
// project loader): its pools are GPU-required and warm up against them.
json pending_demo;            // null: nothing waiting
int pending_demo_ticks = 0;
bool pending_demo_timer_active = false;
std::chrono::steady_clock::time_point pending_demo_wait_start;
constexpr int kDemoWaitTicks = 60 * 60;   // ~1 minute of frames, then report

json settingsToJson(const WetPaintSurfaceSettings& s) {
    return {{"wetness", s.wetness}, {"lifetime_seconds", s.lifetime_seconds},
            {"diffusion", s.diffusion}, {"runoff", s.runoff}, {"absorption", s.absorption},
            {"drip_head", s.drip_head}, {"falloff", s.falloff}, {"paint_load", s.paint_load},
            {"deposit_rate", s.deposit_rate}, {"pickup_rate", s.pickup_rate},
            {"brush_radius", s.brush_radius}, {"oil", s.oil},
            {"roughness_channel", s.roughness_channel},
            {"height_normal_channel", s.height_normal_channel},
            {"reveal_opacity", s.reveal_opacity},
            {"wet_roughness", s.wet_roughness}, {"dry_roughness", s.dry_roughness},
            {"height_contribution", s.height_contribution},
            {"normal_strength", s.normal_strength},
            {"reveal_picture", s.reveal_picture},
            {"picture_tolerance", s.picture_tolerance},
            {"picture_hint", s.picture_hint},
            {"picture_outline", s.picture_outline},
            {"picture_blank", json::array({s.picture_blank[0], s.picture_blank[1], s.picture_blank[2]})}};
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
    s.reveal_opacity = flag(j, "reveal_opacity", s.reveal_opacity);
    s.wet_roughness = number(j, "wet_roughness", s.wet_roughness, 0.0f, 1.0f);
    s.dry_roughness = number(j, "dry_roughness", s.dry_roughness, 0.0f, 1.0f);
    s.height_contribution = number(j, "height_contribution", s.height_contribution, 0.0f, 1.0f);
    s.normal_strength = number(j, "normal_strength", s.normal_strength, 0.0f, 8.0f);
    if (j.is_object() && j.contains("reveal_picture") && j["reveal_picture"].is_string())
        s.reveal_picture = j["reveal_picture"].get<std::string>();
    s.picture_tolerance = number(j, "picture_tolerance", s.picture_tolerance, 5.0f, 90.0f);
    s.picture_hint = number(j, "picture_hint", s.picture_hint, 0.0f, 1.0f);
    s.picture_outline = number(j, "picture_outline", s.picture_outline, 0.0f, 1.0f);
    if (j.is_object() && j.contains("picture_blank") && j["picture_blank"].is_array() &&
        j["picture_blank"].size() == 3) {
        for (size_t i = 0; i < 3; ++i) {
            if (j["picture_blank"][i].is_number()) {
                s.picture_blank[i] = std::clamp(j["picture_blank"][i].get<float>(), 0.0f, 1.0f);
            }
        }
    }
    return s;
}

std::shared_ptr<MeshPaintAdapter> gamePaintAdapter(SceneData& scene,
                                                 const std::string& object_name,
                                                 int material_id,
                                                 const FlatMeshLookup* mesh_lookup = nullptr) {
    std::shared_ptr<Triangle> representative;
    auto findInMesh = [&](const std::shared_ptr<TriangleMesh>& mesh) {
        if (!mesh || !mesh->geometry) {
            return;
        }
        const auto& geometry = *mesh->geometry;
        const auto* materials = geometry.get_attribute_data<uint16_t>("materialID");
        for (size_t face = 0; face < mesh->num_triangles(); ++face) {
            const size_t offset = face * 3;
            if (offset >= geometry.indices.size()) {
                continue;
            }
            const auto vertex = geometry.indices[offset];
            if (materials && vertex < geometry.get_vertex_count() &&
                material_id >= 0 && materials[vertex] != material_id) {
                continue;
            }
            auto candidate = std::make_shared<Triangle>(mesh, static_cast<uint32_t>(face));
            if (material_id < 0 || candidate->getMaterialID() == material_id) {
                representative = candidate;
                break;
            }
        }
    };

    // Flat meshes are authoritative. One representative supplies only the
    // existing texture adapter's material key, never scene geometry.
    if (mesh_lookup) {
        auto it = mesh_lookup->find(object_name);
        if (it != mesh_lookup->end()) {
            for (const auto& mesh : it->second) {
                findInMesh(mesh);
                if (representative) {
                    break;
                }
            }
        }
    } else {
        for (const auto& object : scene.world.objects) {
            auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (!mesh || mesh->nodeName != object_name || !mesh->geometry) {
                continue;
            }
            findInMesh(mesh);
            if (representative) {
                break;
            }
        }
    }
    if (!representative) {
        return nullptr;
    }
    return std::make_shared<MeshPaintAdapter>(&scene, representative);
}

bool filesIdentical(const std::filesystem::path& a, const std::filesystem::path& b) {
    std::error_code ec;
    const auto size_a = std::filesystem::file_size(a, ec);
    if (ec) return false;
    const auto size_b = std::filesystem::file_size(b, ec);
    if (ec || size_a != size_b) return false;
    std::ifstream fa(a, std::ios::binary);
    std::ifstream fb(b, std::ios::binary);
    if (!fa || !fb) return false;
    char buf_a[8192];
    char buf_b[8192];
    while (fa && fb) {
        fa.read(buf_a, sizeof(buf_a));
        fb.read(buf_b, sizeof(buf_b));
        const auto n_a = fa.gcount();
        const auto n_b = fb.gcount();
        if (n_a != n_b || std::memcmp(buf_a, buf_b, static_cast<size_t>(n_a)) != 0) {
            return false;
        }
    }
    return fa.eof() && fb.eof();
}

// Hidden picture files are project assets, not references into the author's disk.
// On save, external files are copied into "<project stem>_pictures/" next to the
// project and stored as project-relative paths (matching "_environment/").
std::string portabilizeHiddenPicturePath(const std::string& stored,
                                         const std::string& project_filepath,
                                         const std::string& source_project_filepath) {
    if (stored.empty() || stored == "@base" || project_filepath.empty()) {
        return stored;
    }
    namespace fs = std::filesystem;
    try {
        const fs::path project_json_path(project_filepath);
        const fs::path project_dir = fs::absolute(project_json_path).parent_path();
        const fs::path pic_dir = project_dir / (project_json_path.stem().string() + "_pictures");
        fs::path source(stored);
        std::error_code ec;
        if (source.is_relative()) {
            const fs::path original = source_project_filepath.empty()
                ? project_dir / source
                : fs::absolute(fs::path(source_project_filepath)).parent_path() / source;
            const fs::path destination_candidate = project_dir / source;
            if (fs::is_regular_file(original, ec)) {
                source = original;
            } else if (fs::is_regular_file(destination_candidate, ec)) {
                source = destination_candidate;
            }
        }
        if (!fs::is_regular_file(source, ec)) {
            SCENE_LOG_WARN("[GameDesign] Hidden picture is missing, path kept as stored: " +
                           stored);
            return stored;
        }
        const fs::path relative_picture = fs::relative(source, pic_dir, ec);
        if (!ec && !relative_picture.empty() && *relative_picture.begin() != "..") {
            return (pic_dir.filename() / relative_picture).generic_string();
        }
        fs::create_directories(pic_dir, ec);
        if (ec) {
            SCENE_LOG_WARN("[GameDesign] Could not create pictures folder (" + ec.message() +
                           "): " + pic_dir.string());
            return stored;
        }
        const fs::path base_stem = source.stem();
        const fs::path ext = source.extension();
        fs::path target = pic_dir / source.filename();
        for (int suffix = 1; suffix <= 999; ++suffix) {
            if (!fs::exists(target, ec)) {
                break;
            }
            if (filesIdentical(source, target)) {
                return (pic_dir.filename() / target.filename()).generic_string();
            }
            target = pic_dir / (base_stem.string() + "_" + std::to_string(suffix) + ext.string());
        }
        fs::copy_file(source, target, fs::copy_options::none, ec);
        if (ec) {
            SCENE_LOG_WARN("[GameDesign] Could not copy hidden picture into project (" +
                           ec.message() + "): " + stored);
            return stored;
        }
        return (pic_dir.filename() / target.filename()).generic_string();
    } catch (const std::exception& e) {
        SCENE_LOG_WARN(std::string("[GameDesign] Hidden picture portability failed: ") + e.what());
        return stored;
    }
}

} // namespace

bool resizeGamePaintSurfaces(SceneData& scene, int resolution, std::string& error) {
    if (resolution < 32 || resolution > 4096) {
        error = "wet paint resolution must be between 32 and 4096";
        return false;
    }
    auto& paint = wetPaintGpu();
    const auto surfaces = paint.surfaces();
    if (surfaces.empty()) {
        error = "no wet paint surfaces registered";
        return false;
    }
    std::vector<std::shared_ptr<MeshPaintAdapter>> adapters;
    const FlatMeshLookup flat_meshes = buildFlatMeshLookup(scene);
    bool changed = false;
    for (const auto& surface : surfaces) {
        auto adapter = gamePaintAdapter(scene, surface.object, surface.material_id, &flat_meshes);
        if (!adapter || !adapter->isValid() || !adapter->getTextureSet() ||
            !adapter->getTextureSet()->initialized) {
            error = "paint texture unavailable: " + surface.object;
            return false;
        }
        adapters.push_back(adapter);
        changed |= surface.resolution != resolution;
    }
    if (!changed) {
        return true;
    }
    // Release device atlas references before texture storage is resized.
    paint.clear();
    bool ok = true;
    for (size_t i = 0; i < surfaces.size(); ++i) {
        const auto& surface = surfaces[i];
        if (surface.resolution != resolution && !adapters[i]->resizeTextureSet(resolution)) {
            error += "resize failed: " + surface.object + "; ";
            ok = false;
        }
        std::string restore_error;
        if (!paint.addSurface(scene, surface.object, surface.material_id,
                              resolution, restore_error)) {
            error += surface.object + ": " + restore_error + "; ";
            ok = false;
            continue;
        }
        paint.setSurfaceSettings(surface.object, surface.material_id, surface.settings);
    }
    g_materials_dirty = true;
    return ok;
}

json saveGameDesign(const std::string& project_filepath,
                    const std::string& source_project_filepath) {
    auto& paint = wetPaintGpu();
    const json demo = balloonPaintDesign();
    if (paint.empty() && demo.is_null()) {
        return nullptr;
    }
    json surfaces = json::array();
    for (const auto& s : paint.surfaces()) {
        WetPaintSurfaceSettings saved_settings = s.settings;
        if (!project_filepath.empty() && !saved_settings.reveal_picture.empty()) {
            saved_settings.reveal_picture =
                portabilizeHiddenPicturePath(saved_settings.reveal_picture, project_filepath,
                                             source_project_filepath);
        }
        surfaces.push_back({{"object", s.object}, {"material_id", s.material_id},
                            {"resolution", s.resolution},
                            {"settings", settingsToJson(saved_settings)}});
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
    design["hidden_objects"] = hiddenGameObjectsDesign();
    return design;
}

void adoptSavedGamePicturePaths(const json& saved_design) {
    if (saved_design.is_null()) {
        return;
    }
    auto& paint = wetPaintGpu();
    const auto surfaces = paint.surfaces();
    for (const auto& entry : saved_design.at("wet_paint").at("surfaces")) {
        const auto& object = entry.at("object").get_ref<const std::string&>();
        const int material_id = entry.at("material_id").get<int>();
        const auto& picture =
            entry.at("settings").at("reveal_picture").get_ref<const std::string&>();
        const auto surface = std::find_if(surfaces.begin(), surfaces.end(), [&](const auto& s) {
            return s.object == object && s.material_id == material_id;
        });
        if (surface != surfaces.end() && surface->settings.reveal_picture != picture) {
            auto settings = surface->settings;
            settings.reveal_picture = picture;
            paint.setSurfaceSettings(object, material_id, settings);
        }
    }
}

void queueGameDesign(const json& design) {
    pending_design = design.is_object() ? design : json();
    pending_demo = json();
    pending_demo_ticks = 0;
    pending_demo_timer_active = false;
    apply_status.clear();
}

void resetGameDesign() {
    clearHiddenGameObjects();
    pending_design = json();
    pending_demo = json();
    pending_demo_ticks = 0;
    pending_demo_timer_active = false;
    apply_status.clear();
}

bool gameDesignPending() {
    return pending_design.is_object() || pending_demo.is_object();
}

namespace {

void tickPendingDemo(SceneData& scene, GameDesignTiming* timing = nullptr) {
    const auto now = std::chrono::steady_clock::now();
    if (!pending_demo_timer_active) {
        pending_demo_wait_start = now;
        pending_demo_timer_active = true;
    }
    std::string waiting;
    if (!balloonPaintCollidersReady(scene, waiting)) {
        if (++pending_demo_ticks < kDemoWaitTicks) {
            apply_status = "waiting for collider SDF: " + waiting;
            return;
        }
        apply_status = "balloon demo not prepared: collider SDF never finished (" + waiting + ")";
        SCENE_LOG_WARN("[GameDesign] " + apply_status);
        pending_demo = json();
        pending_demo_timer_active = false;
        return;
    }
    const double sdf_wait_ms = std::chrono::duration<double, std::milli>(
        now - pending_demo_wait_start).count();
    pending_demo_timer_active = false;
    json config = std::move(pending_demo);
#ifdef RT_PLAYER
    applyBalloonPlayerDefaults(config);
#endif
    pending_demo = json();
    json output;
    std::string error;
    const auto t_demo = std::chrono::steady_clock::now();
    if (!balloonPaintCommand(scene, "configure", config, output, error)) {
        apply_status = "balloon demo: " + error +
                       " (sdf_wait_ticks=" + std::to_string(pending_demo_ticks) + ")";
        SCENE_LOG_WARN("[GameDesign] " + apply_status);
    } else {
        apply_status = "applied: " + std::to_string(wetPaintGpu().surfaces().size()) +
                       " paint surface(s), balloon demo prepared (sdf_wait_ticks=" +
                       std::to_string(pending_demo_ticks) + ")";
    }
    const double demo_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_demo).count();
    if (timing) {
        timing->sdf_wait_ms = sdf_wait_ms;
        timing->demo_ms = demo_ms;
    } else {
        const std::string demo_summary =
            "demo=" + std::to_string(static_cast<int>(demo_ms + 0.5)) +
            "ms, sdf_wait=" + std::to_string(static_cast<int>(sdf_wait_ms + 0.5)) + "ms";
        apply_status += " [" + demo_summary + "]";
        SCENE_LOG_INFO("[GameDesign] demo configure=" +
                       std::to_string(static_cast<int>(demo_ms + 0.5)) +
                       " ms (sdf_wait=" + std::to_string(static_cast<int>(sdf_wait_ms + 0.5)) +
                       " ms, sdf_wait_ticks=" + std::to_string(pending_demo_ticks) + ")");
    }
}

} // namespace

std::string gameDesignStatus() {
    return apply_status;
}

bool reconcileGameDesign(SceneData& scene) {
    // Do not prune during project replacement: its flat nodes are still loading.
    if (pending_design.is_object() || scene.isSimulationBaking()) {
        return false;
    }
    const auto missing = [&scene](const std::string& object) {
        return scene.isEditorPendingDeleteObjectName(object) ||
               !scene.getFlatNodeMesh(object);
    };
    auto& paint = wetPaintGpu();
    std::unordered_set<std::string> removed;
    for (const auto& surface : paint.surfaces()) {
        if (missing(surface.object)) {
            removed.insert(surface.object);
            paint.removeSurface(surface.object, surface.material_id, &scene);
        }
    }
    bool changed = !removed.empty();
    for (auto& entry : scene.particle_systems) {
        if (!entry.runtime) {
            continue;
        }
        auto& colliders = entry.runtime->colliders();
        for (size_t i = colliders.size(); i > 0; --i) {
            const auto& collider = colliders[i - 1];
            const auto& object = collider.source_name;
            const bool managed = collider.name == object + " PaintCollider";
            if (!object.empty() && (managed || removed.count(object) != 0) &&
                missing(object)) {
                entry.runtime->removeCollider(i - 1);
                changed = true;
            }
        }
    }
    if (changed) {
        // Preserve live pools; only invalidate obsolete scene/collider history.
        scene.clearSimFrameCache();
        scene.preserveScriptSimulationPreview();
    }
    return changed;
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
    const auto t_design_start = std::chrono::steady_clock::now();
    GameDesignTiming timing;
    try {
#ifndef RT_PLAYER
        if ((design.contains("wet_paint") || design.contains("balloon_demo")) &&
            AppRuntime::bindStudioGamePaintDevice(scene)) {
            SCENE_LOG_INFO("[StudioPlayPaint] Game compute bound to the Vulkan viewport.");
        }
#endif
        const json wet = design.value("wet_paint", json::object());
        const json surfaces = wet.value("surfaces", json::array());
        if (surfaces.is_array() && !surfaces.empty()) {
            const auto t_lookup = std::chrono::steady_clock::now();
            const FlatMeshLookup flat_meshes = buildFlatMeshLookup(scene);
            timing.adapter_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_lookup).count();
            // The document replaces whatever a previous project left behind.
            paint.clear();
            for (const auto& entry : surfaces) {
                if (!entry.is_object() || !entry.contains("object") || !entry["object"].is_string()) {
                    note("surface entry without an object name");
                    continue;
                }
                const std::string object = entry["object"].get<std::string>();
                const int material = entry.value("material_id", -1);
                int resolution = std::clamp(entry.value("resolution", 1024), 32, 4096);
#ifdef RT_PLAYER
                resolution = std::min(resolution, effectiveMaxPaintResolution());
#endif
                std::string error;
                // Sync serialized layer-stack dimensions to target resolution
                // before WetPaintGpu binds channels, avoiding repeated 512<->1024
                // CPU resamples inside MeshPaintAdapter::ensureLayerStack().
                const auto t_adapter = std::chrono::steady_clock::now();
                syncLayerStackResolution(scene, object, material, resolution);
                // Existing serialized Paint Edit textures may have a different
                // size. Honor the saved game surface size before baking its atlas.
                if (!scene.mesh_paint_texture_sets.empty()) {
                    auto adapter = gamePaintAdapter(scene, object, material, &flat_meshes);
                    timing.adapter_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_adapter).count();
                    if (adapter && adapter->getTextureSet() &&
                        adapter->getTextureSet()->initialized) {
                        const auto texture =
                            adapter->getTextureSet()->getTexture(PaintChannel::BaseColor);
                        if (texture &&
                            (texture->width != resolution || texture->height != resolution)) {
                            const auto t_resize = std::chrono::steady_clock::now();
                            const bool resized = adapter->resizeTextureSet(resolution);
                            timing.resize_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_resize).count();
                            if (!resized) {
                                note(object + ": could not restore saved paint resolution");
                                continue;
                            }
                        }
                    }
                } else {
                    timing.adapter_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_adapter).count();
                }
                const auto t_add = std::chrono::steady_clock::now();
                if (!paint.addSurface(scene, object, material, resolution, error)) {
                    timing.add_surface_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_add).count();
                    // A renamed or deleted object: say so, keep the rest.
                    note(object + ": " + error);
                    continue;
                }
                timing.add_surface_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_add).count();
                const auto t_set = std::chrono::steady_clock::now();
                WetPaintSurfaceSettings surface_settings =
                    settingsFromJson(entry.value("settings", json::object()));
#ifdef RT_PLAYER
                surface_settings.lifetime_seconds = (std::min)(
                    surface_settings.lifetime_seconds, effectiveMaxPaintLifetimeSeconds());
#endif
                paint.setSurfaceSettings(object, material, surface_settings);
                timing.settings_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_set).count();
                ++added;
            }
            g_materials_dirty = true;
        }
        if (wet.contains("splat") && wet["splat"].is_object()) {
            const auto t_splat = std::chrono::steady_clock::now();
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
            timing.settings_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_splat).count();
        }
        if (design.contains("hidden_objects")) {
            const auto t_hid = std::chrono::steady_clock::now();
            std::string error;
            if (!configureHiddenGameObjects(scene, design["hidden_objects"], error)) {
                note("Hidden objects: " + error);
            }
            timing.hidden_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_hid).count();
        }
        if (design.contains("balloon_demo") && design["balloon_demo"].is_object()) {
            // Configured on a later tick, once every droplet collider is cooked.
            pending_demo = design["balloon_demo"];
            pending_demo_ticks = 0;
            pending_demo_timer_active = false;
        }
    } catch (const std::exception& e) {
        note(e.what());
    }
    const std::string status_before_demo = apply_status;
    if (!pending_demo.is_null()) {
        // Apply immediately when droplet colliders are already ready instead of
        // waiting until Frame 1+2.
        tickPendingDemo(scene, &timing);
    }
    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_design_start).count();
    const auto roundMs = [](double ms) {
        return std::to_string(static_cast<int>(ms + 0.5));
    };
    SCENE_LOG_INFO("[GameDesign] apply total=" + roundMs(total_ms) +
                   " ms (adapter=" + roundMs(timing.adapter_ms) +
                   " ms, resize=" + roundMs(timing.resize_ms) +
                   " ms, add_surface=" + roundMs(timing.add_surface_ms) +
                   " ms, settings=" + roundMs(timing.settings_ms) +
                   " ms, hidden=" + roundMs(timing.hidden_ms) +
                   " ms, demo=" + roundMs(timing.demo_ms) +
                   " ms, sdf_wait=" + roundMs(timing.sdf_wait_ms) +
                   " ms, surfaces=" + std::to_string(added) +
                   ", sdf_wait_ticks=" + std::to_string(pending_demo_ticks) + ")");
    const std::string timing_tag =
        " [total=" + roundMs(total_ms) + " add=" + roundMs(timing.add_surface_ms) +
        " set=" + roundMs(timing.settings_ms) + " hid=" + roundMs(timing.hidden_ms) +
        " demo=" + roundMs(timing.demo_ms) + " ms]";
    if (apply_status == status_before_demo) {
        apply_status = problems.empty()
            ? "applied: " + std::to_string(added) + " paint surface(s)" + timing_tag
            : "applied " + std::to_string(added) + " paint surface(s)" + timing_tag +
                  ", problems: " + problems;
    } else {
        apply_status += timing_tag;
    }
    if (!problems.empty()) {
        SCENE_LOG_WARN("[GameDesign] " + apply_status);
    }
}

} // namespace Paint
