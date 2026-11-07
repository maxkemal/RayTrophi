// SPDX-License-Identifier: MIT
#include "UV/UvGeometryTransaction.h"
#include "SceneHistory.h"
#include "TriangleMesh.h"
#include "ProjectManager.h"
#include "scene_ui.h"

#include <utility>

namespace UV {
namespace {

class GeometryCommand final : public SceneCommand {
public:
    GeometryCommand(std::shared_ptr<TriangleMesh> mesh,
                     std::shared_ptr<DNA::GeometryDetail> after, std::string label)
        : mesh_(std::move(mesh)), before_(mesh_->geometry),
          after_(std::move(after)), label_(std::move(label)) {}

    void execute(UIContext& ctx) override {
        apply(ctx, after_);
    }
    void undo(UIContext& ctx) override {
        apply(ctx, before_);
    }
    Type getType() const override {
        return Type::Heavy;
    }
    bool isHeavyGeometry() const override {
        return true;
    }
    size_t getTriangleCount() const override {
        return (before_->indices.size() + after_->indices.size()) / 3;
    }
    std::string getDescription() const override {
        return "UV " + label_;
    }

private:
    std::shared_ptr<TriangleMesh> mesh_;
    std::shared_ptr<DNA::GeometryDetail> before_, after_;
    std::string label_;

    void apply(UIContext& ctx, const std::shared_ptr<DNA::GeometryDetail>& geometry) {
        mesh_->geometry = geometry;
        mesh_->local_bvh.reset();
        mesh_->pointiness.clear();
        mesh_->material_attribs.clear();
        ctx.scene.invalidateSurfaceMeshCache(mesh_->nodeName);
        ctx.scene.requestUiMeshCacheRebuild();
        scheduleSceneMutationRebuilds(ctx, true);
        ProjectManager::getInstance().markModified();
    }
};

} // namespace

void commitGeometry(UIContext& ctx, SceneHistory& history,
                    const std::shared_ptr<TriangleMesh>& mesh,
                    std::shared_ptr<DNA::GeometryDetail> after, const std::string& label) {
    auto command = std::make_unique<GeometryCommand>(mesh, std::move(after), label);
    command->execute(ctx);
    history.record(std::move(command));
}

} // namespace UV
