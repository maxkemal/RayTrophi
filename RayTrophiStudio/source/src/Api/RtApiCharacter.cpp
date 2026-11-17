#include <limits>
#include <sstream>
#include "Api/RtApiCharacter.h"
#include "RtApiInternal.h"
#include "Input/ExternalBodyRig.h"
#include "ProjectManager.h"
#include "TriangleMesh.h"
#include "../UI/scene_ui_animgraph.hpp"

#include <algorithm>
#include <unordered_set>

namespace {
template <class Container, class Predicate>
void eraseMatching(Container &values, Predicate predicate) {
    for (auto it = values.begin(); it != values.end();) {
        if (predicate(*it)) {
            it = values.erase(it);
        } else {
            ++it;
        }
    }
}

// Bone indices are deliberately not compacted: other meshes retain their slots.
struct CharacterState {
    std::vector<SceneData::ImportedModelContext> models;
    std::vector<std::shared_ptr<AnimationData>> clips;
    BoneData bones;
    RayTrophiSim::KinematicColliderRegistry colliders;
    std::vector<ImportedModelData> imports;
    RigAuthoring::ViewState view;
    decltype(g_animGraphUI.graphs) graphs;
    std::vector<SceneData::SceneGroup> groups;
    decltype(SceneData::timeline.tracks) tracks;
    std::string active;

    explicit CharacterState(UIContext &ctx)
        : models(ctx.scene.importedModelContexts), clips(ctx.scene.animationDataList),
          bones(ctx.scene.boneData), colliders(ctx.scene.kinematic_colliders),
          imports(ProjectManager::getInstance().getProjectData().imported_models),
          view(ctx.scene.rigView), graphs(g_animGraphUI.graphs), groups(ctx.scene.object_groups),
          tracks(ctx.scene.timeline.tracks), active(g_animGraphUI.activeCharacter) {}

    void exchange(UIContext &ctx) {
        models.swap(ctx.scene.importedModelContexts);
        clips.swap(ctx.scene.animationDataList);
        std::swap(bones, ctx.scene.boneData);
        std::swap(colliders, ctx.scene.kinematic_colliders);
        imports.swap(ProjectManager::getInstance().getProjectData().imported_models);
        std::swap(view, ctx.scene.rigView);
        graphs.swap(g_animGraphUI.graphs);
        groups.swap(ctx.scene.object_groups);
        tracks.swap(ctx.scene.timeline.tracks);
        active.swap(g_animGraphUI.activeCharacter);
        ctx.scene.kinematic_colliders.resetMotionHistory();
        ctx.scene.kinematic_snapshot_step_frame_ = std::numeric_limits<int>::min();
        g_animGraphUI.selectedNodeIds.clear();
        g_animGraphUI.selectedLinkIds.clear();
    }
};

class DeleteCharacterCommand final : public SceneCommand {
    std::string character_;
    CharacterState state_;
    std::vector<std::unique_ptr<DeleteObjectCommand>> meshes_;
    std::unordered_map<std::string, bool> visibility_;

    void refresh(UIContext &ctx) {
        ctx.selection.clearSelection();
        ctx.renderer.invalidateAnimationGeometry();
        scheduleSceneMutationRebuilds(ctx, true);
        ProjectManager::getInstance().markModified();
    }

public:
    DeleteCharacterCommand(UIContext &ctx, const std::string &character,
                           const SceneData::ImportedModelContext &model)
        : character_(character), state_(ctx) {
        std::unordered_set<std::string> nodes;
        for (const auto &node : model.nodeHierarchy.nodes) {
            nodes.insert(node.uniqueName);
        }
        nodes.insert(model.rigBoundMeshes.begin(), model.rigBoundMeshes.end());
        for (const auto &member : model.members) {
            if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(member)) {
                nodes.insert(mesh->nodeName);
            }
        }
        // Only canonical flat geometry supplies mesh identities.
        std::unordered_set<std::string> parts;
        for (const auto &object : ctx.scene.world.objects) {
            auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (mesh && nodes.count(mesh->nodeName) &&
                !ctx.scene.isEditorPendingDeleteObjectName(mesh->nodeName) &&
                parts.insert(mesh->nodeName).second) {
                visibility_[mesh->nodeName] = mesh->visible;
                meshes_.push_back(std::make_unique<DeleteObjectCommand>(
                    mesh->nodeName, std::vector<std::shared_ptr<Triangle>>{}));
            }
        }
        const std::string prefix = character + "_";
        auto owned = [&](const std::string &name) {
            if (nodes.count(name)) {
                return true;
            }
            if (name.compare(0, prefix.size(), prefix) != 0) {
                return false;
            }
            // A separately imported character may have a longer common prefix.
            for (const auto &other : state_.models) {
                const auto otherPrefix = other.importName + "_";
                if (other.importName != character &&
                    otherPrefix.size() > prefix.size() &&
                    name.compare(0, otherPrefix.size(), otherPrefix) == 0) {
                    return false;
                }
            }
            return true;
        };
        eraseMatching(state_.bones.boneNameToIndex, [&](const auto &v) { return owned(v.first); });
        eraseMatching(state_.bones.boneDefaultTransforms,
                      [&](const auto &v) { return owned(v.first); });
        eraseMatching(state_.bones.boneOffsetMatrices,
                      [&](const auto &v) { return owned(v.first); });
        eraseMatching(state_.bones.boneParents, [&](const auto &v) { return owned(v.first); });
        eraseMatching(state_.bones.weightedBoneNames, owned);
        state_.bones.perModelInverses.erase(character);
        state_.bones.boneIndexToName.clear();
        state_.bones.rebuildReverseLookup();
        eraseMatching(state_.clips, [&](const auto &clip) {
            return clip && clip->modelName == character;
        });
        eraseMatching(state_.colliders.sets(), [&](const auto &set) {
            return set.target_character == character;
        });
        eraseMatching(state_.imports, [&](const auto &entry) {
            return std::to_string(entry.id) == character;
        });
        for (auto &group : state_.groups) {
            eraseMatching(group.member_names, [&](const auto &name) { return parts.count(name); });
        }
        eraseMatching(state_.tracks, [&](const auto &track) {
            return track.first == character || parts.count(track.first);
        });
        const auto graphKey = model.animGraphAssetKey;
        eraseMatching(state_.models, [&](const auto &entry) {
            return entry.importName == character;
        });
        const bool sharedGraph = std::any_of(state_.models.begin(), state_.models.end(),
            [&](const auto &entry) { return entry.animGraphAssetKey == graphKey; });
        if (!sharedGraph) {
            state_.graphs.erase(graphKey);
        }
        if (state_.active == character) {
            state_.active.clear();
        }
        if (state_.view.character == character) {
            state_.view.character.clear();
            state_.view.bone.clear();
        }
        if (state_.view.edit_character == character) {
            state_.view.edit_mode = false;
            state_.view.edit_character.clear();
        }
        if (state_.view.selection_character == character) {
            state_.view.selection_character.clear();
            state_.view.selection_anchor.clear();
            state_.view.selected_bones.clear();
        }
        if (state_.view.pose.character == character) {
            state_.view.pose = RigAuthoring::PoseAuthoringState{};
        }
        state_.view.pose.clips.erase(character);
        if (state_.view.envelope_overlay_character == character) {
            state_.view.envelope_overlay_character.clear();
            state_.view.envelope_overlay_visible = false;
        }
        state_.view.pose_views.erase(character);
        state_.view.pose_view_dirty.erase(character);
    }

    void execute(UIContext &ctx) override {
        for (auto &mesh : meshes_) {
            mesh->execute(ctx);
        }
        state_.exchange(ctx);
        refresh(ctx);
    }

    void undo(UIContext &ctx) override {
        state_.exchange(ctx);
        for (auto it = meshes_.rbegin(); it != meshes_.rend(); ++it) {
            (*it)->undo(ctx);
        }
        for (const auto &[name, visible] : visibility_) {
            // The shared API updates flat flags and backend masks in both targets.
            (void)rtapi::setObjectVisible(name, visible);
        }
        refresh(ctx);
    }

    Type getType() const override { return Type::Heavy; }
    bool isHeavyGeometry() const override { return true; }
    std::string getDescription() const override { return "Delete character " + character_; }
};
} // namespace

namespace rtapi {
Result deleteCharacter(const std::string &character) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene_locked");
    }
    if (!g_history) {
        return Result::fail("history_not_bound");
    }
    auto &models = g_ctx->scene.importedModelContexts;
    const auto model = std::find_if(models.begin(), models.end(), [&](const auto &entry) {
        return entry.importName == character;
    });
    if (character.empty() || model == models.end()) {
        return Result::fail("character_not_found");
    }
    // Tracking is transient; require an explicit clear rather than losing it on undo.
    if (Input::externalBodyRigOwns(g_ctx->scene, character)) {
        return Result::fail("character_external_pose_active_clear_first");
    }
    try {
        auto command = std::make_unique<DeleteCharacterCommand>(*g_ctx, character, *model);
        command->execute(*g_ctx);
        g_history->record(std::move(command));
        return Result::success();
    } catch (const std::exception &error) {
        return Result::fail(std::string("character_delete_failed: ") + error.what());
    }
}
} // namespace rtapi
