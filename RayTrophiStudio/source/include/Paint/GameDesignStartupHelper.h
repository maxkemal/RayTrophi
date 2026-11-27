#pragma once
// Focused startup helpers for GameDesign and BalloonPaintDemo.
// Kept in a dedicated header to avoid growing >2000-line .cpp files and to
// require no build-system file-list changes.

#include "Api/RtApi.h"
#include "Material.h"
#include "MaterialManager.h"
#include "Paint/PaintLayerStack.h"
#include "TriangleMesh.h"
#include "UIContext.h"
#include "json.hpp"
#include "scene_data.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Paint {

struct GameDesignTiming {
    double adapter_ms = 0.0;
    double resize_ms = 0.0;
    double add_surface_ms = 0.0;
    double settings_ms = 0.0;
    double hidden_ms = 0.0;
    double demo_ms = 0.0;
    double sdf_wait_ms = 0.0;
};

using FlatMeshLookup = std::unordered_map<std::string, std::vector<std::shared_ptr<TriangleMesh>>>;

inline FlatMeshLookup buildFlatMeshLookup(SceneData& scene) {
    FlatMeshLookup lookup;
    lookup.reserve(scene.world.objects.size());
    for (const auto& object : scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (mesh && mesh->geometry && !mesh->nodeName.empty()) {
            lookup[mesh->nodeName].push_back(std::move(mesh));
        }
    }
    return lookup;
}

// Align serialized PaintLayerStack resolution with the target wet-paint
// surface resolution before MeshPaintAdapter::assignTextureToChannel() runs.
// On Android (defaultMaxPaintResolution = 512), projects saved at 1024 carry
// empty 1024x1024 layer stacks; without syncing first, every channel bind
// upsamples 512->1024 with single-threaded UV trig and flattens 1024->512.
inline void syncLayerStackResolution(SceneData& scene,
                                     const std::string& object,
                                     int material_id,
                                     int resolution) {
    if (resolution <= 0 || scene.mesh_paint_layer_stacks.empty()) {
        return;
    }
    if (material_id >= 0) {
        const std::string key = object + "#" + std::to_string(material_id);
        auto it = scene.mesh_paint_layer_stacks.find(key);
        if (it != scene.mesh_paint_layer_stacks.end() &&
            (it->second.width() != resolution || it->second.height() != resolution)) {
            it->second.setResolution(resolution, resolution);
        }
        return;
    }
    const std::string prefix = object + "#";
    for (auto& kv : scene.mesh_paint_layer_stacks) {
        if (kv.first.rfind(prefix, 0) == 0 &&
            (kv.second.width() != resolution || kv.second.height() != resolution)) {
            kv.second.setResolution(resolution, resolution);
        }
    }
}

// Single-pass replacement for the O(balloons * scene_vertices) nested scan in
// BalloonPaintDemo::validateMeshVisuals. Preserves all checks and error text.
inline void validateMeshVisualsFast(SceneData& scene,
                                    const nlohmann::json& config,
                                    const std::vector<std::string>& surface_names) {
    if (!config.contains("visual_nodes")) {
        return;
    }
    const auto& nodes = config["visual_nodes"];
    if (!nodes.is_array() || nodes.size() > 24 ||
        nodes.size() < config["balloons"].get<size_t>()) {
        throw std::runtime_error(
            "visual_nodes must cover every active balloon, with up to 24 meshes");
    }
    auto* context = rtapi::boundContext();
    if (!context || &context->scene != &scene) {
        throw std::runtime_error("mesh balloon visuals require the bound scene context");
    }

    const std::unordered_set<std::string> surface_set(surface_names.begin(), surface_names.end());
    std::unordered_set<std::string> seen_names;
    seen_names.reserve(nodes.size());
    std::unordered_map<uint16_t, std::string> material_owner;
    material_owner.reserve(nodes.size());
    std::unordered_map<std::string, size_t> node_entry_counts;
    node_entry_counts.reserve(nodes.size());
    std::vector<std::pair<std::string, uint16_t>> checked_nodes;
    checked_nodes.reserve(nodes.size());

    for (const auto& value : nodes) {
        if (!value.is_string() || value.get<std::string>().empty()) {
            throw std::runtime_error("visual_nodes entries must be non-empty strings");
        }
        const auto name = value.get<std::string>();
        if (!seen_names.insert(name).second) {
            throw std::runtime_error("duplicate visual node: " + name);
        }
        if (surface_set.count(name) != 0) {
            throw std::runtime_error("balloon visual cannot also be a paint surface: " + name);
        }
        auto* mesh = scene.getFlatNodeMesh(name);
        if (!mesh || !mesh->geometry || !mesh->transform || !mesh->num_triangles()) {
            throw std::runtime_error("flat balloon mesh not found: " + name);
        }
        for (const auto& weights : mesh->geometry->skin_weights) {
            if (!weights.empty()) {
                throw std::runtime_error("balloon visual must be a rigid flat mesh: " + name);
            }
        }
        const auto* ids = mesh->geometry->get_material_ids();
        const size_t material_count =
            mesh->geometry->get_core_attribute_count(DNA::Attr::MaterialID);
        if (!ids || material_count != mesh->geometry->get_vertex_count() || material_count == 0) {
            throw std::runtime_error("balloon mesh needs a material: " + name);
        }
        const uint16_t material = ids[0];
        for (size_t i = 1; i < material_count; ++i) {
            if (ids[i] != material) {
                throw std::runtime_error("balloon mesh must use one material: " + name);
            }
        }
        if (!material_owner.emplace(material, name).second) {
            throw std::runtime_error("balloon meshes must have distinct materials");
        }
        node_entry_counts.emplace(name, 0u);
        checked_nodes.emplace_back(name, material);
    }

    // One pass over scene.world.objects for all balloon nodes simultaneously.
    for (const auto& object : scene.world.objects) {
        auto other = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!other || !other->geometry) {
            continue;
        }
        auto count_it = node_entry_counts.find(other->nodeName);
        if (count_it != node_entry_counts.end()) {
            ++count_it->second;
            continue;
        }
        const auto* other_ids = other->geometry->get_material_ids();
        const size_t other_count =
            other->geometry->get_core_attribute_count(DNA::Attr::MaterialID);
        if (!other_ids || other_count == 0) {
            continue;
        }
        // Fast path: check first vertex and skip consecutive identical IDs.
        uint16_t last_checked = static_cast<uint16_t>(~other_ids[0]);
        for (size_t i = 0; i < other_count; ++i) {
            const uint16_t id = other_ids[i];
            if (id == last_checked) {
                continue;
            }
            last_checked = id;
            auto owner_it = material_owner.find(id);
            if (owner_it != material_owner.end()) {
                throw std::runtime_error(
                    "balloon material is shared with another object: " + owner_it->second);
            }
        }
    }

    for (const auto& [name, material] : checked_nodes) {
        if (node_entry_counts[name] != 1) {
            throw std::runtime_error(
                "balloon visual must resolve to exactly one flat mesh: " + name);
        }
        rtapi::MaterialParamValue color;
        const auto result = rtapi::getMaterialParam(name, "base_color", color);
        if (!result.ok || !color.is_color) {
            throw std::runtime_error("balloon mesh needs an editable color material: " + name);
        }
        if (const Material* bound = MaterialManager::getInstance().getMaterial(material);
            bound && bound->albedoProperty.texture) {
            throw std::runtime_error(
                "balloon material has a base colour texture (" +
                bound->albedoProperty.texture->name +
                "), so the balloon colour cannot show; clear it (material.clear_texture "
                "slot base_color): " + name);
        }
    }
}

} // namespace Paint
