#include "MeshEdit/ProfileAuthoringService.h"

#include "SceneCommand.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "ProjectManager.h"

#include <algorithm>
#include <cctype>
#include <memory>

namespace MeshEdit {
namespace {

std::string uniqueName(const UIContext& ctx, const std::string& requested,
                       const std::string& operation) {
    std::string base = requested;
    if (base.empty()) base = operation == "profile.revolve" ? "Revolve" : "Sweep";
    if (base.empty()) base = "ProfileMesh";
    std::string candidate = base;
    int suffix = 1;
    auto exists = [&](const std::string& name) {
        for (const auto& object : ctx.scene.world.objects) {
            if (const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
                if (mesh->nodeName == name) return true;
            }
        }
        return false;
    };
    while (exists(candidate)) {
        candidate = base + "." + (suffix < 10 ? "00" : suffix < 100 ? "0" : "") + std::to_string(suffix++);
    }
    return candidate;
}

} // namespace

ProfilePublishResult publishGeneratedProfile(UIContext& ctx, SceneHistory& history,
                                             std::shared_ptr<DNA::GeometryDetail> geometry,
                                             const std::string& requested_name,
                                             const std::string& operation_id) {
    ProfilePublishResult result;
    result.report.operation_id = operation_id;
    if (!geometry || geometry->get_vertex_count() == 0 || geometry->indices.empty()) {
        result.report.addError("empty_profile_geometry", "Cannot publish empty profile geometry.");
        return result;
    }

    const std::string name = uniqueName(ctx, requested_name, operation_id);
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->nodeName = name;
    // TriangleMesh's constructor intentionally leaves the optional transform
    // handle null for low-level/import paths. Scene-authored objects must own a
    // handle so hierarchy selection, gizmos and transform commands can operate.
    mesh->transform = std::make_shared<Transform>();
    mesh->geometry = std::move(geometry);
    mesh->build_local_bvh();

    auto command = std::make_unique<AddObjectCommand>(mesh);
    command->execute(ctx);
    history.record(std::move(command));
    const int objectIndex = static_cast<int>(ctx.scene.world.objects.size()) - 1;
    ctx.selection.selectObject(mesh, objectIndex, name);
    result.object_name = name;
    result.report.ok = true;
    result.report.changed.vertices_changed = mesh->geometry->get_vertex_count();
    result.report.changed.triangles_changed = mesh->geometry->indices.size() / 3;
    result.report.changed.faces_changed = result.report.changed.triangles_changed;
    ProjectManager::getInstance().markModified();
    return result;
}

} // namespace MeshEdit
