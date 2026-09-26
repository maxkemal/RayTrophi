#include "KinematicColliderScene.h"

#include "Animation/RigView.h"
#include "scene_data.h"

#include <unordered_map>
#include <utility>

namespace RayTrophiSim {
namespace {

using JointMap = std::unordered_map<std::string, Matrix4x4>;

bool buildJointMap(const SceneData& scene,
                   const std::string& character,
                   JointMap& joints,
                   std::string& error) {
    std::vector<RigAuthoring::BoneView> views;
    if (!RigAuthoring::listBones(scene, character, views, error)) {
        return false;
    }
    joints.clear();
    joints.reserve(views.size());
    for (const RigAuthoring::BoneView& view : views) {
        joints.emplace(view.name, view.world);
    }
    return true;
}

} // namespace

bool collectKinematicJointPoses(const SceneData& scene,
                                const std::string& character,
                                std::vector<KinematicJointPose>& joints,
                                std::string& error) {
    std::vector<RigAuthoring::BoneView> views;
    if (!RigAuthoring::listBones(scene, character, views, error)) {
        joints.clear();
        return false;
    }
    joints.clear();
    joints.reserve(views.size());
    for (const RigAuthoring::BoneView& view : views) {
        KinematicJointPose pose;
        pose.name = view.name;
        pose.parent = view.parent;
        pose.world = view.world;
        pose.weighted = view.weighted;
        joints.push_back(std::move(pose));
    }
    return true;
}

bool autoFitKinematicProxySet(SceneData& scene,
                              uint64_t set_id,
                              const KinematicAutoFitOptions& options,
                              uint32_t& created_count,
                              std::string& error) {
    const KinematicProxySet* set = scene.kinematic_colliders.findSet(set_id);
    if (!set) {
        error = "unknown_proxy_set";
        return false;
    }
    std::vector<KinematicJointPose> joints;
    if (!collectKinematicJointPoses(
            scene, set->target_character, joints, error)) {
        return false;
    }
    return scene.kinematic_colliders.autoFit(
        set_id, joints, options, created_count, error);
}

bool sampleKinematicProxySet(SceneData& scene,
                             uint64_t set_id,
                             float dt,
                             bool discontinuity,
                             std::vector<KinematicProxySample>& samples,
                             std::string& error) {
    const KinematicProxySet* set = scene.kinematic_colliders.findSet(set_id);
    if (!set) {
        error = "unknown_proxy_set";
        return false;
    }
    JointMap joints;
    std::string pose_error;
    const bool pose_resolved = buildJointMap(
        scene, set->target_character, joints, pose_error);
    return scene.kinematic_colliders.sampleSet(
        set_id,
        dt,
        discontinuity,
        [&](const std::string&, const std::string& bone,
            Matrix4x4& world, std::string& reason) {
            if (!pose_resolved) {
                reason = pose_error.empty()
                    ? "target_character_unresolved"
                    : pose_error;
                return false;
            }
            const auto it = joints.find(bone);
            if (it == joints.end()) {
                reason = "bone_unresolved";
                return false;
            }
            world = it->second;
            return true;
        },
        samples,
        error);
}

bool inspectKinematicProxySet(const SceneData& scene,
                              uint64_t set_id,
                              float dt,
                              std::vector<KinematicProxySample>& samples,
                              std::string& error) {
    const KinematicProxySet* set = scene.kinematic_colliders.findSet(set_id);
    if (!set) {
        error = "unknown_proxy_set";
        return false;
    }
    JointMap joints;
    std::string pose_error;
    const bool pose_resolved = buildJointMap(
        scene, set->target_character, joints, pose_error);
    KinematicColliderRegistry snapshot = scene.kinematic_colliders;
    return snapshot.sampleSet(
        set_id,
        dt,
        false,
        [&](const std::string&, const std::string& bone,
            Matrix4x4& world, std::string& reason) {
            if (!pose_resolved) {
                reason = pose_error.empty()
                    ? "target_character_unresolved"
                    : pose_error;
                return false;
            }
            const auto it = joints.find(bone);
            if (it == joints.end()) {
                reason = "bone_unresolved";
                return false;
            }
            world = it->second;
            return true;
        },
        samples,
        error);
}

bool sampleAllKinematicProxySets(SceneData& scene,
                                 float dt,
                                 bool discontinuity,
                                 std::vector<KinematicProxySample>& samples,
                                 std::string& error) {
    samples.clear();
    std::unordered_map<std::string, JointMap> pose_cache;
    std::unordered_map<std::string, std::string> pose_errors;
    const std::vector<KinematicProxySet>& sets = scene.kinematic_colliders.sets();
    for (const KinematicProxySet& set : sets) {
        if (pose_cache.find(set.target_character) == pose_cache.end() &&
            pose_errors.find(set.target_character) == pose_errors.end()) {
            JointMap joint_map;
            std::string pose_error;
            if (buildJointMap(scene, set.target_character, joint_map, pose_error)) {
                pose_cache.emplace(set.target_character, std::move(joint_map));
            } else {
                pose_errors.emplace(set.target_character, std::move(pose_error));
            }
        }

        std::vector<KinematicProxySample> set_samples;
        const auto joints_it = pose_cache.find(set.target_character);
        const auto error_it = pose_errors.find(set.target_character);
        if (!scene.kinematic_colliders.sampleSet(
                set.id,
                dt,
                discontinuity,
                [&](const std::string&, const std::string& bone,
                    Matrix4x4& world, std::string& reason) {
                    if (joints_it == pose_cache.end()) {
                        reason = error_it != pose_errors.end() &&
                                 !error_it->second.empty()
                            ? error_it->second
                            : "target_character_unresolved";
                        return false;
                    }
                    const auto bone_it = joints_it->second.find(bone);
                    if (bone_it == joints_it->second.end()) {
                        reason = "bone_unresolved";
                        return false;
                    }
                    world = bone_it->second;
                    return true;
                },
                set_samples,
                error)) {
            return false;
        }
        samples.insert(
            samples.end(), set_samples.begin(), set_samples.end());
    }
    return true;
}

} // namespace RayTrophiSim
