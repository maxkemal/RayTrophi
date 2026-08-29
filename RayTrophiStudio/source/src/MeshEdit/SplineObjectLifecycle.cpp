#include "MeshEdit/SplineObjectLifecycle.h"

#include "MeshEdit/SplineObject.h"
#include "GeometryNodesV2.h"
#include "ProjectManager.h"
#include "SceneCommand.h"
#include "SceneSelection.h"
#include "scene_ui.h"

#include <algorithm>
#include <iterator>

namespace MeshEdit {
namespace {

class DeleteSplineObjectCommand final : public SceneCommand {
public:
    DeleteSplineObjectCommand(
        std::shared_ptr<SplineObject> spline, size_t index,
        std::string skinHostName,
        std::shared_ptr<GeometryNodesV2::GeometryNodeGraphV2> skinGraph)
        : spline_(std::move(spline)), index_(index),
          skin_graph_(std::move(skinGraph)) {
        if (!skinHostName.empty())
            skin_delete_ = std::make_unique<DeleteObjectCommand>(
                std::move(skinHostName), std::vector<std::shared_ptr<Triangle>>{});
    }

    void execute(UIContext& ctx) override {
        if (!spline_) return;
        auto& objects = ctx.scene.world.objects;
        objects.erase(std::remove_if(objects.begin(), objects.end(), [&](const auto& object) {
            return object.get() == spline_.get();
        }), objects.end());
        if (skin_delete_) skin_delete_->execute(ctx);
        if (!spline_->skin_display.host_name.empty())
            ctx.scene.geometry_node_graphs.erase(spline_->skin_display.host_name);
        for (const auto& object : objects) {
            auto other = std::dynamic_pointer_cast<SplineObject>(object);
            if (!other || other->profile_preview_counterpart != spline_->nodeName) continue;
            other->profile_preview_geometry.reset();
            other->profile_preview_operation.clear();
            other->profile_preview_counterpart.clear();
            other->profile_preview_status.clear();
            other->profile_preview_signature = 0;
        }
        if (ctx.selection.selected.spline_object == spline_) ctx.selection.clearSelection();
        ProjectManager::getInstance().markModified();
        ctx.start_render = true;
    }

    void undo(UIContext& ctx) override {
        if (!spline_) return;
        auto& objects = ctx.scene.world.objects;
        const bool exists = std::any_of(objects.begin(), objects.end(), [&](const auto& object) {
            return object.get() == spline_.get();
        });
        if (!exists) {
            objects.insert(objects.begin() + std::min(index_, objects.size()), spline_);
        }
        if (skin_delete_) skin_delete_->undo(ctx);
        if (skin_graph_ && !spline_->skin_display.host_name.empty())
            ctx.scene.geometry_node_graphs[spline_->skin_display.host_name] = skin_graph_;
        ProjectManager::getInstance().markModified();
        ctx.start_render = true;
    }

    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override {
        return "Delete spline " + (spline_ ? spline_->nodeName : std::string{});
    }

private:
    std::shared_ptr<SplineObject> spline_;
    size_t index_ = 0;
    std::unique_ptr<DeleteObjectCommand> skin_delete_;
    std::shared_ptr<GeometryNodesV2::GeometryNodeGraphV2> skin_graph_;
};

} // namespace

bool deleteSplineObject(UIContext& ctx, SceneHistory& history,
                        const std::shared_ptr<SplineObject>& spline) {
    if (!spline) return false;
    const auto it = std::find_if(ctx.scene.world.objects.begin(),
                                 ctx.scene.world.objects.end(), [&](const auto& object) {
        return object.get() == spline.get();
    });
    if (it == ctx.scene.world.objects.end()) return false;
    const size_t index = static_cast<size_t>(std::distance(ctx.scene.world.objects.begin(), it));
    std::shared_ptr<GeometryNodesV2::GeometryNodeGraphV2> skinGraph;
    const auto graphIt = ctx.scene.geometry_node_graphs.find(spline->skin_display.host_name);
    if (graphIt != ctx.scene.geometry_node_graphs.end()) skinGraph = graphIt->second;
    auto command = std::make_unique<DeleteSplineObjectCommand>(
        spline, index, spline->skin_display.host_name, skinGraph);
    command->execute(ctx);
    history.record(std::move(command));
    return true;
}

} // namespace MeshEdit
