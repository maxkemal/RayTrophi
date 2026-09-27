#include "KinematicColliderScene.h"

#include "Animation/RigBindingScope.h"
#include "Animation/RigView.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include "scene_data.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
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

bool sampleAllWithRegistry(
    const SceneData& scene,
    KinematicColliderRegistry& registry,
    float dt,
    bool discontinuity,
    std::vector<KinematicProxySample>& samples,
    std::string& error) {
    samples.clear();
    std::unordered_map<std::string, JointMap> pose_cache;
    std::unordered_map<std::string, std::string> pose_errors;
    const std::vector<KinematicProxySet>& sets = registry.sets();
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
        if (!registry.sampleSet(
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

// Measures, per joint, the bone-local box of the rest-pose vertices it
// dominates. A foot proxy placed from joints alone sits on the ankle pivot;
// the vertices say where the heel, sole and toes actually are.
//
// Rest-pose skinning maps a bind vertex to world as
//   mesh_base * inverse_root * G_b * offset_b * p
// and the sampler places the proxy in placement * G_b, so the bone-local
// position is inverse(placement * G_b) * mesh_base * inverse_root * G_b *
// offset_b * p. Both sides use the rest globals, so the result is independent
// of whatever pose the character is currently in.
void accumulateRestMeshBounds(const SceneData& scene,
                              const std::string& character,
                              std::vector<KinematicJointPose>& joints) {
    const SceneData::ImportedModelContext* model = nullptr;
    for (const auto& context : scene.importedModelContexts) {
        if (context.importName == character) {
            model = &context;
            break;
        }
    }
    if (!model) {
        return;
    }
    Matrix4x4 placement;
    std::string placement_error;
    if (!RigAuthoring::rigScenePlacement(
            scene, character, placement, placement_error)) {
        return;
    }
    Matrix4x4 inverse_root = model->globalInverseTransform;
    const auto root_it = scene.boneData.perModelInverses.find(character);
    if (root_it != scene.boneData.perModelInverses.end()) {
        inverse_root = root_it->second;
    }

    std::unordered_map<std::string, std::size_t> joint_slot;
    for (std::size_t index = 0; index < joints.size(); ++index) {
        joint_slot.emplace(joints[index].name, index);
    }
    struct BoneMapping {
        std::size_t joint = 0;
        Matrix4x4 rest_bone_world;
        Matrix4x4 skin_rest;
    };
    std::unordered_map<int, BoneMapping> bones;
    for (const auto& node : model->skeletonNodes) {
        const auto slot_it = joint_slot.find(node.name);
        const auto offset_it = scene.boneData.boneOffsetMatrices.find(node.name);
        if (node.boneIndex < 0 || slot_it == joint_slot.end() ||
            offset_it == scene.boneData.boneOffsetMatrices.end()) {
            continue;
        }
        BoneMapping mapping;
        mapping.joint = slot_it->second;
        mapping.rest_bone_world =
            (placement * node.globalBindTransform).inverse();
        mapping.skin_rest =
            inverse_root * node.globalBindTransform * offset_it->second;
        bones.emplace(node.boneIndex, mapping);
    }
    if (bones.empty()) {
        return;
    }

    std::unordered_set<const DNA::GeometryDetail*> seen;
    for (const auto& object : scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || !mesh->geometry || !mesh->hasSkinWeights() ||
            !RigAuthoring::meshBelongsToRig(scene, character, *mesh) ||
            !seen.insert(mesh->geometry.get()).second) {
            continue;
        }
        const DNA::GeometryDetail& geometry = *mesh->geometry;
        const Vec3* rest = geometry.get_positions_orig();
        if (!rest) {
            rest = geometry.get_positions();
        }
        if (!rest) {
            continue;
        }
        const std::size_t count = std::min(
            geometry.get_positions_orig_count(), geometry.skin_weights.size());
        const Matrix4x4 mesh_base = mesh->transform
            ? mesh->transform->base
            : Matrix4x4::identity();
        for (std::size_t vertex = 0; vertex < count; ++vertex) {
            int dominant = -1;
            float dominant_weight = 0.0f;
            for (const auto& influence : geometry.skin_weights[vertex]) {
                if (std::isfinite(influence.second) &&
                    influence.second > dominant_weight) {
                    dominant = influence.first;
                    dominant_weight = influence.second;
                }
            }
            // A blend vertex at a joint crease belongs to neither bone's
            // rigid shape; only clear majorities shape the box.
            if (dominant < 0 || dominant_weight < 0.5f) {
                continue;
            }
            const auto bone_it = bones.find(dominant);
            if (bone_it == bones.end()) {
                continue;
            }
            const BoneMapping& mapping = bone_it->second;
            const Vec3 local = mapping.rest_bone_world.transform_point(
                mesh_base.transform_point(
                    mapping.skin_rest.transform_point(rest[vertex])));
            if (!std::isfinite(local.x) || !std::isfinite(local.y) ||
                !std::isfinite(local.z)) {
                continue;
            }
            KinematicJointPose& joint = joints[mapping.joint];
            if (!joint.has_mesh_bounds) {
                joint.has_mesh_bounds = true;
                joint.mesh_bounds_min = local;
                joint.mesh_bounds_max = local;
            } else {
                joint.mesh_bounds_min = Vec3::min(joint.mesh_bounds_min, local);
                joint.mesh_bounds_max = Vec3::max(joint.mesh_bounds_max, local);
            }
        }
    }
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
    accumulateRestMeshBounds(scene, character, joints);
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

bool inspectAllKinematicProxySets(
    const SceneData& scene,
    float dt,
    std::vector<KinematicProxySample>& samples,
    std::string& error) {
    KinematicColliderRegistry snapshot = scene.kinematic_colliders;
    return sampleAllWithRegistry(
        scene, snapshot, dt, false, samples, error);
}

bool sampleAllKinematicProxySets(SceneData& scene,
                                 float dt,
                                 bool discontinuity,
                                 std::vector<KinematicProxySample>& samples,
                                 std::string& error) {
    return sampleAllWithRegistry(
        scene,
        scene.kinematic_colliders,
        dt,
        discontinuity,
        samples,
        error);
}

} // namespace RayTrophiSim
