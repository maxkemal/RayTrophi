#include "Audio/AcousticScene.h"
#include "Audio/AcousticRenderDebug.h"
#include "Audio/AcousticSurface.h"
#include "Audio/AcousticGeometryPolicy.h"
#include "SceneSelection.h"
#include "Audio/AudioAuthoring.h"
#include "ObjectSubstance.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include "Camera.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace audio {
namespace {
using json = nlohmann::json;
void require(bool valid, const char* message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}

double number(const json& params, const char* key, double fallback, double low, double high) {
    const auto value = params.value(key, json(fallback));
    require(value.is_number() && std::isfinite(value.get<double>()) &&
            value.get<double>() >= low && value.get<double>() <= high,
            "Acoustic control must be finite and within its documented bounds");
    return value.get<double>();
}

unsigned integer(const json& params, const char* key, unsigned fallback, unsigned high) {
    const auto value = params.value(key, json(fallback));
    require(value.is_number_integer(), "Acoustic resource limit must be an integer");
    return static_cast<unsigned>(number(params, key, fallback, 1, high));
}

AcousticPoint point(const json& value) {
    require(value.is_array() && value.size() == 3, "Acoustic position needs three numbers");
    for (const auto& component : value) {
        require(component.is_number() && std::isfinite(component.get<double>()) &&
                std::abs(component.get<double>()) <= 1e9,
                "Acoustic position must be finite within +/-1e9 scene units");
    }
    return {value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
}

AcousticSceneSnapshot capture(UIContext& ctx, unsigned maximum, bool includeHidden) {
    AcousticSceneSnapshot result;
    std::map<std::string, std::size_t> objects;
    for (const auto& object : ctx.scene.world.objects) {
        if (!object) {
            continue;
        }
        if (isAcousticSimulationVisual(*object)) {
            ++result.ignoredSimulationVisuals;
            continue;
        }
        if (!includeHidden && !object->visible) {
            ++result.hidden;
            continue;
        }
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh) {
            ++result.unsupported;
            continue;
        }
        if (mesh->transient) {
            ++result.transient;
            continue;
        }
        if (ctx.scene.isEditorPendingDeleteObjectName(mesh->nodeName)) {
            continue;
        }
        if (!mesh->geometry || mesh->geometry->indices.empty()) {
            continue;
        }
        const auto& geometry = *mesh->geometry;
        require(geometry.indices.size() % 3 == 0, "Acoustic mesh has incomplete triangle indices");
        require(geometry.indices.size() / 3 <= maximum - result.faces.size(),
                "Acoustic triangle budget exceeded; no geometry was silently sampled");
        // Match the canonical CPU mesh path: skinned P is local live geometry;
        // other meshes use local P_orig and the current transform exactly once.
        const bool skinned = mesh->hasSkinWeights();
        const Vec3* positions = skinned ? geometry.get_positions() : geometry.get_positions_orig();
        auto count = skinned ? geometry.get_core_attribute_count(DNA::Attr::P) :
                             geometry.get_positions_orig_count();
        if (!positions) {
            positions = geometry.get_positions();
            count = geometry.get_core_attribute_count(DNA::Attr::P);
        }
        require(positions && count > 0, "Acoustic mesh positions are unavailable");
        const auto transform = mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
        if (!objects.contains(mesh->nodeName)) {
            require(objects.size() < 2048, "Acoustic object budget exceeded (2048)");
            objects[mesh->nodeName] = result.objects.size();
            auto material = matter::objectSubstanceState(ctx.scene, mesh->nodeName, true);
            material.erase("profile_fields");
            material.erase("colliders");
            result.objects.push_back(std::move(material));
        }
        const auto objectIndex = objects.at(mesh->nodeName);
        for (std::size_t face = 0; face < geometry.indices.size(); face += 3) {
            AcousticFace triangle;
            AcousticPoint* output[] = {&triangle.a, &triangle.b, &triangle.c};
            for (unsigned corner = 0; corner < 3; ++corner) {
                const auto index = geometry.indices[face + corner];
                require(index < count, "Acoustic index exceeds the actual position buffer");
                const auto world = transform.transform_point(positions[index]);
                require(std::isfinite(world.x) && std::isfinite(world.y) &&
                        std::isfinite(world.z), "Acoustic mesh has a nonfinite world position");
                *output[corner] = {world.x, world.y, world.z};
            }
            triangle.object = objectIndex;
            result.faces.push_back(triangle);
        }
    }
    return result;
}
}

std::vector<std::string> acousticTraceParameters() {
    return {"id", "source_position", "listener_position", "meters_per_unit", "sound_speed_mps",
        "max_triangles", "max_hits", "include_hidden", "boundary_mode"};
}

json querySceneAcoustics(UIContext& ctx, const json& params) {
    require(params.is_object(), "Acoustic trace parameters must be an object");
    const auto allowed = acousticTraceParameters();
    for (auto entry = params.begin(); entry != params.end(); ++entry) {
        require(std::find(allowed.begin(), allowed.end(), entry.key()) != allowed.end(),
                "Unknown acoustic trace parameter");
    }
    require(params.contains("id") != params.contains("source_position"),
            "Supply either a source id or source_position");
    auto sourceValue = params.value("source_position", json());
    if (params.contains("id")) {
        require(params.at("id").is_string(), "Acoustic source id must be a string");
        const auto runtime = ctx.scene.audio->invoke("source_runtime", {{"id", params.at("id")}});
        require(runtime.at("valid").get<bool>(), "Acoustic source pose is unavailable");
        sourceValue = runtime.at("position");
    }
    auto listenerValue = params.value("listener_position", json());
    if (!params.contains("listener_position")) {
        require(ctx.scene.camera != nullptr, "Acoustic trace needs a listener or active camera");
        const auto& position = ctx.scene.camera->lookfrom;
        listenerValue = {position.x, position.y, position.z};
    }
    const auto source = point(sourceValue);
    const auto listener = point(listenerValue);
    const double units = number(params, "meters_per_unit", 1, 1e-6, 1e6);
    const double speed = number(params, "sound_speed_mps", 343, 100, 2000);
    const unsigned maxTriangles = integer(params, "max_triangles", 100000, 500000);
    const unsigned maxHits = integer(params, "max_hits", 128, 256);
    require(!params.contains("include_hidden") || params.at("include_hidden").is_boolean(),
            "include_hidden must be boolean");
    const auto mode = params.value("boundary_mode", json("diagnostic"));
    require(mode.is_string() && (mode == "diagnostic" || mode == "thin_surface" ||
            mode == "object_barrier"),
            "boundary_mode must be diagnostic, thin_surface or object_barrier");
    const double dx = source.x - listener.x, dy = source.y - listener.y;
    const double dz = source.z - listener.z;
    require(std::sqrt(dx * dx + dy * dy + dz * dz) * units >= 1e-4,
            "Acoustic segment must be at least 0.1 mm long");
    const auto scene = capture(ctx, maxTriangles, params.value("include_hidden", true));
    auto result = traceAcousticSegment(scene, source, listener, units, speed, maxHits,
                                      mode == "thin_surface", mode == "object_barrier");
    result["source_pose"] = params.contains("id") ? "runtime_world" : "explicit_world";
    result["include_hidden"] = params.value("include_hidden", true);
    result["boundary_mode"] = mode;
    result["materials_revision"] = matter::objectSubstanceFingerprint(ctx.scene);
    return result;
}
json queryAcousticDebug(UIContext& ctx, const json& params) {
    require(params.is_object() && params.size() == 1 && params.contains("id") &&
        params.at("id").is_string(), "acoustics_debug requires only a string id");
    require(ctx.scene.camera != nullptr, "Acoustic debug requires a listener camera");
    const auto data = ctx.scene.audio->snapshot();
    json emitter;
    for (const auto& candidate : data.at("emitters")) {
        if (candidate.at("id") == params.at("id")) {
            emitter = candidate;
            break;
        }
    }
    require(!emitter.is_null(), "Acoustic debug source does not exist");
    const auto runtime = ctx.scene.audio->invoke("source_runtime", {{"id", emitter.at("id")}});
    const auto position = runtime.at("position");
    const auto& camera = ctx.scene.camera->lookfrom;
    const json listener = {camera.x, camera.y, camera.z};
    double squared = 0;
    for (unsigned axis = 0; axis < 3; ++axis) {
        const double delta = position.at(axis).get<double>() - listener.at(axis).get<double>();
        squared += delta * delta;
    }
    json preview = nullptr;
    for (const auto& entry : ctx.scene.audio->invoke("acoustics_preview_status", json::object())) {
        if (entry.at("id") == params.at("id")) {
            preview = entry;
            break;
        }
    }
    const auto report = preview.is_null() ? json::object() :
        preview.value("report", json::object());
    const bool matches = report.contains("source_position") &&
        report.at("source_position") == position &&
        report.value("listener_position", json()) == listener;
    const auto surface = acousticRenderSource(ctx);
    const bool selected = ctx.selection.selected.type == SelectableType::AudioSource &&
        ctx.selection.selected.audio_source_id == params.at("id").get<std::string>();
    auto visualizer = acousticSurfaceStatus(ctx);
    visualizer["source_selected"] = selected;
    visualizer["source_eligible"] = selected && surface[7] > 0.5f;
    return {{"surface_visualizer", visualizer},
        {"id", params.at("id")}, {"source_position", position},
        {"source_runtime", runtime},
        {"listener_position", listener}, {"distance_scene_units", std::sqrt(squared)},
        {"min_distance", emitter.at("min_distance")},
        {"max_distance", emitter.at("max_distance")}, {"preview", preview},
        {"result_pose_matches", matches}, {"range_is_hard_cutoff", false},
        {"representation", "direct_path_summary"},
        {"hit_positions_available", false}, {"issues_gpu_query", false}};
}
} // namespace audio
