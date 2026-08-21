#include "MeshEdit/FlatMeshGeometryCommand.h"

#include "TriangleMesh.h"
#include "scene_ui.h"

#include <utility>

FlatMeshGeometryCommand::FlatMeshGeometryCommand(
    std::string object_name,
    std::shared_ptr<DNA::GeometryDetail> before,
    std::shared_ptr<DNA::GeometryDetail> after,
    std::string description)
    : object_name_(std::move(object_name)), before_(std::move(before)),
      after_(std::move(after)), description_(std::move(description)) {}

void FlatMeshGeometryCommand::apply(
    UIContext& ctx, const std::shared_ptr<DNA::GeometryDetail>& geometry) {
    if (!geometry) return;
    for (const auto& object : ctx.scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || mesh->nodeName != object_name_ || !mesh->geometry) continue;
        *mesh->geometry = *geometry;
        // Geometry undo/redo must invalidate the UI/edit topology cache as well
        // as the renderer; otherwise bevel undo can keep the post-bevel cage.
        ctx.scene.requestUiMeshCacheRebuild();
        scheduleSceneMutationRebuilds(ctx, true);
        return;
    }
}

void FlatMeshGeometryCommand::execute(UIContext& ctx) {
    apply(ctx, after_);
}

void FlatMeshGeometryCommand::undo(UIContext& ctx) {
    apply(ctx, before_);
}

size_t FlatMeshGeometryCommand::getTriangleCount() const {
    const size_t before = before_ ? before_->indices.size() / 3 : 0;
    const size_t after = after_ ? after_->indices.size() / 3 : 0;
    return before + after;
}
