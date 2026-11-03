// SPDX-License-Identifier: MIT
#include "UV/UvService.h"

#include "Api/RtApiInternal.h"
#include "ProjectManager.h"
#include "SceneHistory.h"
#include "TriangleMesh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace UV {
namespace {

using nlohmann::json;

std::vector<std::shared_ptr<TriangleMesh>> meshes(const std::string& object) {
    std::vector<std::shared_ptr<TriangleMesh>> result;
    std::unordered_set<TriangleMesh*> seen;
    if (!rtapi::g_ctx) {
        return result;
    }
    if (rtapi::g_ctx->scene.isEditorPendingDeleteObjectName(object)) {
        return result;
    }
    for (const auto& entry : rtapi::g_ctx->scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(entry);
        if (mesh && mesh->geometry && mesh->nodeName == object &&
            seen.insert(mesh.get()).second) {
            result.push_back(std::move(mesh));
        }
    }
    return result;
}

std::shared_ptr<TriangleMesh> resolve(const std::string& object, int index) {
    auto targets = meshes(object);
    if (index < 0 || static_cast<size_t>(index) >= targets.size()) {
        return {};
    }
    return targets[static_cast<size_t>(index)];
}

bool capture(const std::shared_ptr<TriangleMesh>& mesh, Layout& layout, std::string& error) {
    if (!mesh || !mesh->geometry) {
        error = "uv_target_not_found: choose a direct flat TriangleMesh";
        return false;
    }
    const auto& geometry = *mesh->geometry;
    const size_t count = geometry.get_vertex_count();
    if (count > (std::numeric_limits<uint32_t>::max)() ||
        geometry.indices.size() > (std::numeric_limits<uint32_t>::max)()) {
        error = "uv_mesh_too_large";
        return false;
    }
    if (geometry.get_core_attribute_count(DNA::Attr::UV) != count ||
        geometry.get_positions_orig_count() != count) {
        error = "uv_missing_buffers: a full uv and P_orig channel is required";
        return false;
    }
    const Vec2* uv = geometry.get_uvs();
    const Vec3* positions = geometry.get_positions_orig();
    if (!uv || !positions || count == 0) {
        error = "uv_missing_buffers";
        return false;
    }
    layout = {};
    layout.coordinates.assign(uv, uv + count);
    layout.positions.assign(positions, positions + count);
    layout.indices.assign(geometry.indices.begin(), geometry.indices.end());
    if (!validate(layout, error)) {
        return false;
    }
    buildIslands(layout);
    // Prevent a target ordinal from silently resolving to another mesh/geometry.
    layout.revision = std::to_string(reinterpret_cast<uintptr_t>(mesh.get())) + ":" +
        std::to_string(reinterpret_cast<uintptr_t>(mesh->geometry.get())) + ":" +
        fingerprint(layout);
    return true;
}

class EditCommand final : public SceneCommand {
public:
    EditCommand(std::shared_ptr<TriangleMesh> mesh, std::vector<Vec2> before,
                std::vector<Vec2> after, std::string label)
        : mesh_(std::move(mesh)), geometry_(mesh_->geometry),
          before_(std::move(before)), after_(std::move(after)), label_(std::move(label)) {}

    void execute(UIContext& ctx) override {
        apply(ctx, after_);
    }
    void undo(UIContext& ctx) override {
        apply(ctx, before_);
    }
    Type getType() const override {
        return Type::Generic;
    }
    std::string getDescription() const override {
        return "UV " + label_;
    }

private:
    std::shared_ptr<TriangleMesh> mesh_;
    std::shared_ptr<DNA::GeometryDetail> geometry_;
    std::vector<Vec2> before_, after_;
    std::string label_;

    void apply(UIContext& ctx, const std::vector<Vec2>& values) {
        if (mesh_->geometry != geometry_ ||
            geometry_->get_core_attribute_count(DNA::Attr::UV) != values.size()) {
            return;
        }
        Vec2* destination = geometry_->get_uvs_mut();
        if (!destination) {
            return;
        }
        std::copy(values.begin(), values.end(), destination);
        ctx.scene.requestUiMeshCacheRebuild();
        scheduleSceneMutationRebuilds(ctx, false);
        ProjectManager::getInstance().markModified();
    }
};

json failure(const std::string& error) {
    return json{{"__error", error}};
}

Vec2 pair(const json& params, const char* key, Vec2 fallback) {
    if (!params.contains(key)) {
        return fallback;
    }
    const auto& value = params.at(key);
    if (!value.is_array() || value.size() != 2 ||
        !value[0].is_number() || !value[1].is_number()) {
        throw std::invalid_argument(std::string("uv_invalid_vec2: ") + key);
    }
    return Vec2(value[0].get<float>(), value[1].get<float>());
}

} // namespace

uint64_t sceneRevision() {
    return g_scene_geometry_generation.load(std::memory_order_relaxed);
}

bool inspect(const std::string& object, int mesh, Layout& layout, std::string& error) {
    if (!rtapi::g_ctx) {
        error = "uv_not_bound";
        return false;
    }
    return capture(resolve(object, mesh), layout, error);
}

nlohmann::json request(const std::string& operation, const nlohmann::json& params) {
    try {
        if (!params.is_object()) {
            return failure("uv_invalid_params: expected object");
        }
        if (operation == "capabilities") {
            return json{{"xatlas", xatlasAvailable()}, {"uv_set", 0},
                {"island_selection", true}, {"corner_remap", false},
                {"seam_authoring", false}, {"texture_preservation", false},
                {"async_atlas", false}, {"atlas_triangle_limit", 20000}};
        }
        if (!rtapi::g_ctx) {
            return failure("uv_not_bound");
        }
        if (operation == "targets") {
            json targets = json::array();
            std::unordered_set<std::string> names;
            for (const auto& entry : rtapi::g_ctx->scene.world.objects) {
                auto mesh = std::dynamic_pointer_cast<TriangleMesh>(entry);
                if (!mesh || !mesh->geometry ||
                    rtapi::g_ctx->scene.isEditorPendingDeleteObjectName(mesh->nodeName) ||
                    !names.insert(mesh->nodeName).second) {
                    continue;
                }
                const auto siblings = meshes(mesh->nodeName);
                for (size_t i = 0; i < siblings.size(); ++i) {
                    targets.push_back({{"object", mesh->nodeName}, {"mesh", i},
                        {"triangles", siblings[i]->geometry->indices.size() / 3}});
                }
            }
            return targets;
        }
        const std::string object = params.at("object").get<std::string>();
        const int mesh_index = params.value("mesh", 0);
        Layout layout;
        std::string error;
        auto mesh = resolve(object, mesh_index);
        if (!capture(mesh, layout, error)) {
            return failure(error);
        }
        json summary{{"object", object}, {"mesh", mesh_index},
            {"revision", layout.revision}, {"vertices", layout.coordinates.size()},
            {"triangles", layout.indices.size() / 3}, {"islands", layout.island_count}};
        if (operation == "info") {
            return summary;
        }
        if (operation == "layout") {
            const int first = params.value("first_face", 0);
            const int limit = params.value("limit", 1000);
            if (first < 0 || limit < 1 || limit > 10000 ||
                static_cast<size_t>(first) > layout.islands.size()) {
                return failure("uv_invalid_page: first_face >= 0, limit 1..10000");
            }
            json faces = json::array();
            const size_t end = (std::min)(layout.islands.size(), size_t(first) + limit);
            for (size_t face = size_t(first); face < end; ++face) {
                json corners = json::array();
                for (size_t k = 0; k < 3; ++k) {
                    const Vec2& uv = layout.coordinates[layout.indices[face * 3 + k]];
                    corners.push_back({uv.u, uv.v});
                }
                faces.push_back({{"face", face}, {"island", layout.islands[face]},
                                  {"uv", std::move(corners)}});
            }
            summary["faces"] = std::move(faces);
            summary["next_face"] = end < layout.islands.size() ? json(end) : json(nullptr);
            return summary;
        }
        const bool repair = operation == "flip_u" || operation == "flip_v" || operation == "swap_uv";
        if (operation != "transform" && operation != "normalize" && !repair &&
            operation != "unwrap" && operation != "pack" && operation != "project_view") {
            return failure("uv_unknown_operation");
        }
        if (rtapi::renderJobActive()) {
            return failure("uv_render_locked");
        }
        if (!rtapi::g_history) {
            return failure("uv_history_unavailable");
        }
        for (const auto& entry : rtapi::g_ctx->scene.world.objects) {
            const auto other = std::dynamic_pointer_cast<TriangleMesh>(entry);
            if (other && other.get() != mesh.get() && other->geometry == mesh->geometry &&
                !rtapi::g_ctx->scene.isEditorPendingDeleteObjectName(other->nodeName)) {
                return failure("uv_shared_geometry: make the target geometry unique before editing");
            }
        }
        if (params.value("revision", std::string()) != layout.revision) {
            return failure("uv_stale_revision: refresh layout before editing");
        }
        // No silent texture preservation promise. This is mandatory even via IPC.
        if (!params.value("allow_texture_change", false)) {
            return failure("uv_texture_change_ack_required: texture rebake is not available yet");
        }
        std::vector<Vec2> before = layout.coordinates;
        if (operation == "transform" || operation == "normalize" || repair) {
            const auto selected = params.value("islands", std::vector<uint32_t>{});
            const Vec2 scale = repair ? Vec2(operation == "flip_u" ? -1.0f : 1.0f,
                                             operation == "flip_v" ? -1.0f : 1.0f)
                                      : pair(params, "scale", Vec2(1, 1));
            const Vec2 pivot = operation == "swap_uv" ? Vec2() :
                (repair ? Vec2(0.5f, 0.5f) : pair(params, "pivot", Vec2(0.5f, 0.5f)));
            if (!transform(layout, selected, repair ? Vec2() : pair(params, "offset", Vec2()),
                    scale, pivot, repair ? 0.0f : params.value("degrees", 0.0f),
                    operation == "normalize", error, operation == "swap_uv")) {
                return failure(error);
            }
        } else if (operation == "project_view") {
            const auto camera = rtapi::g_ctx->scene.getActiveCamera();
            const auto& geometry = *mesh->geometry;
            if (!camera) {
                return failure("uv_projection_no_camera");
            }
            if (geometry.get_core_attribute_count(DNA::Attr::P) != layout.positions.size()) {
                return failure("uv_projection_missing_positions");
            }
            const Vec3* positions = geometry.get_positions();
            if (!positions) {
                return failure("uv_projection_missing_positions");
            }
            const std::vector<Vec3> world(positions, positions + layout.positions.size());
            if (!projectFromView(layout, world, camera->lookfrom,
                    camera->lookat - camera->lookfrom, camera->vup, error, camera->orthographic)) {
                return failure(error);
            }
        } else {
            if (layout.indices.size() / 3 > 20000) {
                return failure("uv_async_required: synchronous atlas limited to 20000 triangles");
            }
            const int resolution = params.value("resolution", 2048);
            const int padding = params.value("padding", 8);
            if (resolution < 64 || resolution > 8192 || padding < 0 || padding > resolution / 8) {
                return failure("uv_invalid_atlas_options");
            }
            if (!atlas(layout, operation == "unwrap", uint32_t(resolution),
                       uint32_t(padding), error)) {
                return failure(error);
            }
        }
        const bool changed = before != layout.coordinates;
        if (changed) {
            auto command = std::make_unique<EditCommand>(mesh, std::move(before),
                std::move(layout.coordinates), operation);
            command->execute(*rtapi::g_ctx);
            rtapi::g_history->record(std::move(command));
        }
        summary["changed"] = changed;
        if (!capture(mesh, layout, error)) {
            return failure(error);
        }
        summary["revision"] = layout.revision;
        return summary;
    } catch (const std::exception& exception) {
        return failure(std::string("uv_invalid_request: ") + exception.what());
    }
}

} // namespace UV
