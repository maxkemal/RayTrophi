#include "Api/RtApiRigMotion.h"
#include "Animation/RigPoseAuthoring.h"
#include "Animation/RigClipRuntimeSync.h"
#include "Backend/IBackend.h"
#include "ProjectManager.h"
#include "RtApiInternal.h"
#include "TriangleMesh.h"
#include "scene_data.h"
#include <functional>
#include <unordered_map>
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
    std::string description;

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
    WalkClipCommand(State state, std::vector<std::shared_ptr<AnimationData>> clips,
                    std::string label)
        : pendingState(std::move(state)), pendingClips(std::move(clips)),
          description(std::move(label)) {}

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
        return description;
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

namespace {
using ClipBuilder =
    std::function<bool(const SceneData::ImportedModelContext &, AnimationData &, std::string &)>;

// Shared by every generated rig clip: validates the target, builds the clip from the current
// rig revision and publishes it (replacing a same-name authored clip) in one undo step.
Result publishAuthoredClip(const std::string &character, const std::string &name,
                           uint64_t revision, const ClipBuilder &build, const char *description,
                           const char *failure) {
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
            if (!build(model, *clip, error)) {
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
            auto command = std::make_unique<WalkClipCommand>(std::move(state), std::move(clips),
                                                             description);
            auto *raw = command.get();
            g_history->record(std::move(command));
            raw->execute(*g_ctx);
            return Result::success();
        }
        return Result::fail("unknown_character");
    } catch (const std::exception &) {
        return Result::fail(failure);
    }
}

// The ROM test reads the authored binding: every registered bound mesh with its bind
// positions, flat skin rows and the scene skin-index table restricted to this rig's bones.
bool romInputs(const SceneData &scene, const SceneData::ImportedModelContext &model,
               std::vector<RigAuthoring::RomTestMesh> &meshes,
               std::unordered_map<std::string, int> &boneIndex, std::string &error) {
    if (model.rigBoundMeshes.empty()) {
        error = "rig_rom_requires_authored_binding";
        return false;
    }
    for (const auto &meshName : model.rigBoundMeshes) {
        std::shared_ptr<TriangleMesh> mesh;
        for (const auto &object : scene.world.objects) {
            auto candidate = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (candidate && candidate->nodeName == meshName) {
                if (mesh) {
                    error = "ambiguous_mesh_name";
                    return false;
                }
                mesh = std::move(candidate);
            }
        }
        if (!mesh || !mesh->geometry) {
            error = "rig_rom_missing_bound_mesh";
            return false;
        }
        meshes.push_back({meshName, mesh->geometry.get(), mesh->num_vertices()});
    }
    for (const auto &node : model.nodeHierarchy.nodes) {
        const auto found = scene.boneData.boneNameToIndex.find(node.uniqueName);
        if (found != scene.boneData.boneNameToIndex.end()) {
            boneIndex[node.uniqueName] = static_cast<int>(found->second);
        }
    }
    return true;
}
} // namespace

Result createRigHumanWalkClip(const std::string &character, const std::string &name,
                              const RigAuthoring::HumanWalkRecipe &recipe, uint64_t revision) {
    return publishAuthoredClip(
        character, name, revision,
        [&](const SceneData::ImportedModelContext &model, AnimationData &clip,
            std::string &error) {
            return RigAuthoring::buildHumanWalkClip(model.nodeHierarchy, model.rigAnatomy, recipe,
                                                    clip, error);
        },
        "Generate human walk clip", "rig_walk_failed");
}

Result previewRigRomTest(const std::string &character, const RigAuthoring::RomTestSettings &settings,
                         nlohmann::json &output) {
    output = nullptr;
    if (!g_ctx) {
        return Result::fail("api_not_bound");
    }
    if (renderJobActive()) {
        return Result::fail("scene_locked");
    }
    try {
        for (const auto &model : g_ctx->scene.importedModelContexts) {
            if (model.importName != character) {
                continue;
            }
            std::vector<RigAuthoring::RomTestMesh> meshes;
            std::unordered_map<std::string, int> boneIndex;
            std::string error;
            if (!romInputs(g_ctx->scene, model, meshes, boneIndex, error) ||
                !RigAuthoring::runRomTest(model.nodeHierarchy, model.rigAnatomy, boneIndex, meshes,
                                          settings, output, nullptr, error)) {
                return Result::fail(error);
            }
            output["character"] = character;
            output["rig_revision"] = model.rigRevision;
            return Result::success();
        }
        return Result::fail("unknown_character");
    } catch (const std::exception &) {
        return Result::fail("rig_rom_failed");
    }
}

Result createRigRomClip(const std::string &character, const std::string &name,
                        const RigAuthoring::RomTestSettings &settings, uint64_t revision) {
    return publishAuthoredClip(
        character, name, revision,
        [&](const SceneData::ImportedModelContext &model, AnimationData &clip,
            std::string &error) {
            std::vector<RigAuthoring::RomTestMesh> meshes;
            std::unordered_map<std::string, int> boneIndex;
            nlohmann::json report;
            return romInputs(g_ctx->scene, model, meshes, boneIndex, error) &&
                   RigAuthoring::runRomTest(model.nodeHierarchy, model.rigAnatomy, boneIndex,
                                            meshes, settings, report, &clip, error);
        },
        "Generate range-of-motion clip", "rig_rom_failed");
}
} // namespace rtapi
