#include "Api/RtApiRigMotion.h"
#include "Animation/RigPoseAuthoring.h"
#include "Animation/RigClipRuntimeSync.h"
#include "Backend/IBackend.h"
#include "ProjectManager.h"
#include "RtApiInternal.h"
#include <utility>

namespace rtapi {
namespace {
using State = RigAuthoring::PoseAuthoringState;

void wake(UIContext &context) {
    context.scene.rigView.pose.invalidateEvaluation();
    context.start_render = true;
    context.renderer.resetCPUAccumulation();
    if (context.backend_ptr) {
        context.backend_ptr->resetAccumulation();
    }
    context.renderer.animation_groups_dirty = true;
}

class WalkClipCommand final : public SceneCommand {
    State pendingState;
    std::vector<std::shared_ptr<AnimationData>> pendingClips;

    void exchange(UIContext &context) {
        const bool active = context.scene.rigView.pose.active;
        std::string character;
        character.swap(context.scene.rigView.pose.character);
        std::swap(context.scene.rigView.pose, pendingState);
        context.scene.rigView.pose.active = active;
        context.scene.rigView.pose.character.swap(character);
        context.scene.rigView.pose.preview.clear();
        context.scene.rigView.pose.previewIK.clear();
        context.scene.rigView.pose.hasPreview = false;
        context.scene.rigView.pose.limitHits.clear();
        context.scene.animationDataList.swap(pendingClips);
        RigAuthoring::synchronizeClipRuntime(
            context.scene, context.scene.rigView.pose.character);
        wake(context);
        ProjectManager::getInstance().markModified();
    }

  public:
    WalkClipCommand(State state, std::vector<std::shared_ptr<AnimationData>> clips)
        : pendingState(std::move(state)), pendingClips(std::move(clips)) {}

    void execute(UIContext &context) override {
        exchange(context);
    }

    void undo(UIContext &context) override {
        exchange(context);
    }

    Type getType() const override {
        return Type::Generic;
    }

    bool isHeavyGeometry() const override {
        return false;
    }

    std::string getDescription() const override {
        return "Generate human walk clip";
    }
};

Result guard(const std::string &character) {
    if (!g_ctx) {
        return Result::fail("api_not_bound");
    }
    if (renderJobActive()) {
        return Result::fail("scene_locked");
    }
    const auto &state = g_ctx->scene.rigView.pose;
    if (!state.active || state.character != character) {
        return Result::fail("rig_pose_mode_required");
    }
    std::string error;
    if (!RigAuthoring::canAuthorPose(g_ctx->scene, character, error)) {
        return Result::fail(error);
    }
    return Result::success();
}
} // namespace

Result previewRigHumanWalk(const std::string &character,
                           const RigAuthoring::HumanWalkRecipe &recipe, nlohmann::json &output) {
    output = nullptr;
    const auto allowed = guard(character);
    if (!allowed.ok) {
        return allowed;
    }
    try {
        for (const auto &model : g_ctx->scene.importedModelContexts) {
            if (model.importName != character) {
                continue;
            }
            std::string error;
            if (!RigAuthoring::inspectHumanWalkRecipe(model.nodeHierarchy, model.rigAnatomy, recipe,
                                                      output, error)) {
                return Result::fail(error);
            }
            output["character"] = character;
            output["rig_revision"] = model.rigRevision;
            return Result::success();
        }
        return Result::fail("unknown_character");
    } catch (const std::exception &) {
        return Result::fail("rig_walk_failed");
    }
}

Result createRigHumanWalkClip(const std::string &character, const std::string &name,
                              const RigAuthoring::HumanWalkRecipe &recipe, uint64_t revision) {
    const auto allowed = guard(character);
    if (!allowed.ok) {
        return allowed;
    }
    if (!g_history) {
        return Result::fail("history_not_bound");
    }
    if (name.empty() || name.size() > 128) {
        return Result::fail("invalid_clip_name");
    }
    if (g_ctx->scene.rigView.pose.hasPreview) {
        return Result::fail("rig_pose_preview_active");
    }
    try {
        size_t replacement = g_ctx->scene.animationDataList.size();
        for (size_t i = 0; i < g_ctx->scene.animationDataList.size(); ++i) {
            const auto &clip = g_ctx->scene.animationDataList[i];
            if (clip && clip->modelName == character && clip->name == name) {
                if (!clip->rigAuthoring) {
                    return Result::fail("rig_pose_clip_name_conflict");
                }
                replacement = i;
            }
        }
        for (const auto &model : g_ctx->scene.importedModelContexts) {
            if (model.importName != character) {
                continue;
            }
            if (model.rigRevision != revision) {
                return Result::fail("rig_edit_stale_revision");
            }
            auto clip = std::make_shared<AnimationData>();
            std::string error;
            if (!RigAuthoring::buildHumanWalkClip(model.nodeHierarchy, model.rigAnatomy, recipe,
                                                  *clip, error)) {
                return Result::fail(error);
            }
            clip->name = name;
            clip->modelName = character;
            auto clips = g_ctx->scene.animationDataList;
            if (replacement < clips.size()) {
                clips[replacement] = std::move(clip);
            } else {
                clips.push_back(std::move(clip));
            }
            auto state = g_ctx->scene.rigView.pose;
            state.clips[character] = name;
            state.locals.clear();
            state.preview.clear();
            state.ik.clear();
            state.previewIK.clear();
            state.control.clear();
            state.hasPreview = false;
            state.limitHits.clear();
            ++state.serial;
            auto command = std::make_unique<WalkClipCommand>(std::move(state), std::move(clips));
            auto *raw = command.get();
            g_history->record(std::move(command));
            raw->execute(*g_ctx);
            return Result::success();
        }
        return Result::fail("unknown_character");
    } catch (const std::exception &) {
        return Result::fail("rig_walk_failed");
    }
}
} // namespace rtapi
