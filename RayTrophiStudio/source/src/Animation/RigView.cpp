#include "Animation/RigView.h"
#include "Animation/RigSelection.h"
#include "Animation/RigPoseView.h"
#include "Animation/RigPoseAuthoring.h"
#include "OzzRuntime.h"
#include <algorithm>
#include "scene_data.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include <utility>

namespace RigAuthoring {
bool rigScenePlacement(const SceneData& scene, const std::string& character, Matrix4x4& placement,
                       std::string& error) {
    error.clear();
    for (const auto& ctx : scene.importedModelContexts)
        if (ctx.importName == character) {
            placement = ctx.authoringOwned && ctx.members.empty() ? ctx.rigSceneTransform
                                                                  : Matrix4x4::identity();
            for (const auto& member : ctx.members) {
                auto mesh = std::dynamic_pointer_cast<TriangleMesh>(member);
                if (mesh && mesh->hasSkinWeights() && mesh->transform) {
                    placement = mesh->transform->base;
                    break;
                }
            }
            return true;
        }
    error = "unknown_character";
    return false;
}
std::vector<std::string> listCharacters(const SceneData& scene) {
    std::vector<std::string> out;
    for (const auto& ctx : scene.importedModelContexts)
        if (!ctx.importName.empty() && !ctx.skeletonNodes.empty())
            out.push_back(ctx.importName);
    return out;
}
bool listBones(const SceneData& scene, const std::string& character, std::vector<BoneView>& out,
               std::string& error) {
    out.clear();
    error.clear();
    for (const auto& ctx : scene.importedModelContexts) {
        if (ctx.importName != character)
            continue;
        if (ctx.skeletonNodes.empty()) {
            error = "character_has_no_skeleton";
            return false;
        }
        Matrix4x4 placement;
        if (!rigScenePlacement(scene, character, placement, error))
            return false;
        // Root motion on a bound flat mesh lives in its transform, outside the
        // joint hierarchy. Never read per-face geometry to position the overlay.
        JointGlobals authorGlobals;
        const bool posing = scene.rigView.pose.active && scene.rigView.pose.character == character;
        if (posing) {
            RayTrophi::NodeHierarchy pose;
            std::vector<PreviewJoint> joints;
            if (!currentPoseHierarchy(scene, character, pose, error, true) ||
                !sampleRigPose(pose, nullptr, 0, joints, error))
                return false;
            for (const auto& j : joints)
                authorGlobals[j.name] = j.world;
        }
        const auto& evaluated = posing ? authorGlobals : ctx.rigJointGlobals;
        for (const auto& node : ctx.skeletonNodes) {
            BoneView b;
            b.character = character;
            b.name = node.name;
            b.parent = node.parentName;
            b.bone_index = node.boneIndex;
            b.weighted = node.weightedBone;
            b.authoring_owned = ctx.authoringOwned;
            b.rig_revision = ctx.rigRevision;
            b.template_id = ctx.rigTemplateId;
            b.template_version = ctx.rigTemplateVersion;
            b.scene_transform = placement;
            b.local_rest = node.localBindTransform;
            b.in_bonedata = scene.boneData.boneNameToIndex.count(node.name) != 0;
            b.in_skeleton_nodes = true;
            b.in_node_hierarchy = ctx.nodeHierarchy.find(node.name) != nullptr;
            if (ctx.ozzAnimationSet) {
                const auto& names = ctx.ozzAnimationSet->skeleton.jointNames;
                b.in_ozz_skeleton = std::find(names.begin(), names.end(), node.name) != names.end();
            }
            const auto it = evaluated.find(node.name);
            const bool rest = isRestPoseView(scene, ctx.importName);
            b.pose_source = posing ? "pose"
                            : rest ? "rest"
                                   : (it == evaluated.end() ? "bind" : ctx.rigPoseSource);
            b.world = placement *
                      ((rest || it == evaluated.end()) ? node.globalBindTransform : it->second);
            out.push_back(std::move(b));
        }
        return true;
    }
    error = "unknown_character";
    return false;
}
bool selectBone(SceneData& scene, const std::string& character, const std::string& bone,
                std::string& error) {
    return selectBones(scene, character, {bone}, bone, "replace", error);
}
void clearSelection(SceneData& scene) {
    releaseIKHandleSelection(scene.rigView);
    scene.rigView.character.clear();
    scene.rigView.bone.clear();
    scene.rigView.selected_bones.clear();
    scene.rigView.selection_character.clear();
    scene.rigView.selection_anchor.clear();
}

void clearPoseSnapshots(SceneData& scene) {
    for (auto& ctx : scene.importedModelContexts) {
        ctx.rigJointGlobals.clear();
        ctx.rigPoseSource = "bind";
    }
}
bool selectedBone(const SceneData& scene, BoneView& out) {
    std::vector<BoneView> bones;
    std::string error;
    if (!listBones(scene, scene.rigView.character, bones, error))
        return false;
    for (const auto& b : bones)
        if (b.name == scene.rigView.bone) {
            out = b;
            return true;
        }
    return false;
}
void captureGlobals(SceneData& scene, const std::string& character, const JointGlobals& globals,
                    const std::string& source) {
    for (auto& ctx : scene.importedModelContexts)
        if (ctx.importName == character) {
            ctx.rigJointGlobals.clear();
            for (const auto& node : ctx.skeletonNodes) {
                const auto it = globals.find(node.name);
                if (it != globals.end())
                    ctx.rigJointGlobals.emplace(node.name, it->second);
            }
            ctx.rigPoseSource = source;
            return;
        }
}
void captureRuntimeGlobals(SceneData& scene, const std::string& character,
                           const std::vector<Matrix4x4>& matrices,
                           const std::vector<int>& mapping) {
    JointGlobals globals;
    for (const auto& ctx : scene.importedModelContexts)
        if (ctx.importName == character) {
            for (const auto& node : ctx.skeletonNodes) {
                if (node.boneIndex < 0 || static_cast<size_t>(node.boneIndex) >= mapping.size())
                    continue;
                const int joint = mapping[node.boneIndex];
                if (joint >= 0 && static_cast<size_t>(joint) < matrices.size())
                    globals.emplace(node.name, matrices[joint]);
            }
        }
    captureGlobals(scene, character, globals, "ozz");
}
}
