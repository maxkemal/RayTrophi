#pragma once
#include "Matrix4x4.h"
#include "Animation/RigPoseAuthoringState.h"
#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct SceneData;
namespace RigAuthoring {
using JointGlobals = std::unordered_map<std::string, Matrix4x4>;
bool rigScenePlacement(const SceneData&,const std::string& character,Matrix4x4& output,std::string& error);
struct ViewState {
    PoseAuthoringState pose;
    bool weight_map_visible = false; // Transient display; never a sculpt protection mask.
    std::vector<std::string> selected_bones;
    std::string selection_character, selection_anchor;
    std::string selection_pivot = "active";
    bool visible = true;
    bool joint_limits_visible=true,joint_limits_edit=false;
    bool envelope_overlay_visible = false;
    std::string envelope_overlay_character;
    float envelope_overlay_torso_radius = .16f;
    float envelope_overlay_limb_radius = .065f;
    float envelope_overlay_extremity_radius = .05f;
    float envelope_overlay_falloff = 2.f;
    std::unordered_map<std::string,std::string> pose_views;
    std::unordered_set<std::string> pose_view_dirty;
    bool edit_mode = false;
    std::string edit_character;
    std::string character;
    std::string bone;
};
struct RigCopyBone { std::string source_bone, target_bone; };
struct BoneView {
    std::string character, name, parent, pose_source;
    int bone_index = -1;
    bool weighted = false;
    bool authoring_owned = false;
    uint64_t rig_revision = 0;
    std::string template_id;
    int template_version=1;
    bool in_bonedata=false, in_skeleton_nodes=false, in_node_hierarchy=false, in_ozz_skeleton=false;
    Matrix4x4 scene_transform = Matrix4x4::identity();
    Matrix4x4 local_rest;
    Matrix4x4 world = Matrix4x4::identity();
};
bool listBones(const SceneData&, const std::string& character, std::vector<BoneView>&, std::string& error);
std::vector<std::string> listCharacters(const SceneData&);
bool selectBone(SceneData&, const std::string& character, const std::string& bone, std::string& error);
void clearSelection(SceneData&);
void clearPoseSnapshots(SceneData&);
bool selectedBone(const SceneData&, BoneView&);
void captureGlobals(SceneData&, const std::string& character, const JointGlobals&, const std::string& source);
void captureRuntimeGlobals(SceneData&, const std::string& character, const std::vector<Matrix4x4>&,
                           const std::vector<int>& sceneToJoint);
}
