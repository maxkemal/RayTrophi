#include "Input/ExternalBodyRig.h"
#include "Input/ExternalBodyRigMath.h"
#include "Animation/RigBindingScope.h"
#include "Animation/RigBindMath.h"
#include "Animation/RigPosePreview.h"
#include "Animation/RigView.h"
#include "TriangleMesh.h"
#include "scene_data.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>

namespace Input {
namespace {
using Clock = std::chrono::steady_clock;
struct Binding {
    int person_id = 0;
    ExternalRigMapping mapping;
    nlohmann::json config;
    std::uint64_t revision = 0;
    std::uint64_t last_packet = std::numeric_limits<std::uint64_t>::max();
    bool had_person = false;
    bool evaluated = false;
    Vec3 last_anchor = Vec3(0);
    std::uint64_t motion_resets = 0;
    Matrix4x4 placement = Matrix4x4::identity();
    std::vector<std::pair<unsigned int, Matrix4x4>> skin;
    RigAuthoring::JointGlobals globals;
    std::vector<std::string> limit_hits;
    std::set<std::string> unreliable_bones;
    std::string state = "waiting_for_person";
    std::string error;
    std::uint64_t solves = 0;
    std::uint64_t skin_updates = 0;
    double solve_ms = 0.0;
    double skin_ms = 0.0;
    double solve_max_ms = 0.0;
};
struct State {
    const SceneData *owner = nullptr;
    std::map<std::string, Binding> bindings;
};
State &state() {
    static State value;
    return value;
}
SceneData::ImportedModelContext *modelFor(SceneData &scene, const std::string &name) {
    for (auto &model : scene.importedModelContexts) {
        if (model.importName == name) {
            return &model;
        }
    }
    return nullptr;
}
bool sameMatrix(const Matrix4x4 &a, const Matrix4x4 &b) {
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            if (a.m[row][column] != b.m[row][column]) {
                return false;
            }
        }
    }
    return true;
}
bool poseCache(const SceneData &scene, const SceneData::ImportedModelContext &model,
               const RayTrophi::NodeHierarchy &pose, Binding &binding, std::string &error) {
    std::vector<RigAuthoring::PreviewJoint> joints;
    if (!RigAuthoring::sampleRigPose(pose, nullptr, 0, joints, error)) {
        return false;
    }
    RigAuthoring::JointGlobals globals;
    std::vector<std::pair<unsigned int, Matrix4x4>> matrices;
    Matrix4x4 inverse = model.globalInverseTransform;
    const auto found = scene.boneData.perModelInverses.find(model.importName);
    if (found != scene.boneData.perModelInverses.end()) {
        inverse = found->second;
    }
    for (const auto &joint : joints) {
        globals.emplace(joint.name, joint.world);
        const auto slot = scene.boneData.boneNameToIndex.find(joint.name);
        const auto offset = scene.boneData.boneOffsetMatrices.find(joint.name);
        if (slot == scene.boneData.boneNameToIndex.end() ||
            offset == scene.boneData.boneOffsetMatrices.end()) {
            if (scene.boneData.weightedBoneNames.count(joint.name)) {
                error = "external_pose_incomplete_skin_binding";
                return false;
            }
            continue;
        }
        if (slot->second >= scene.boneData.getBoneIndexCapacity()) {
            error = "external_pose_invalid_skin_slot";
            return false;
        }
        const auto matrix = inverse * joint.world * offset->second;
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                if (!std::isfinite(matrix.m[row][column])) {
                    error = "external_pose_nonfinite_skin";
                    return false;
                }
            }
        }
        matrices.emplace_back(slot->second, matrix);
    }
    binding.globals = std::move(globals);
    binding.skin = std::move(matrices);
    return true;
}
bool skinFlat(SceneData &scene, const std::string &character, const Binding &binding) {
    std::vector<Matrix4x4> matrices(scene.boneData.getBoneIndexCapacity(), Matrix4x4::identity());
    for (const auto &entry : binding.skin) {
        matrices[entry.first] = entry.second;
    }
    bool changed = false;
    for (const auto &object : scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (mesh && mesh->hasSkinWeights() &&
            RigAuthoring::meshBelongsToRig(scene, character, *mesh) &&
            mesh->applySkinning(matrices)) {
            scene.markBodyGeometryDirty(mesh->nodeName);
            changed = true;
        }
    }
    return changed;
}
const Binding *findBinding(const SceneData &scene, const std::string &character) {
    if (state().owner != &scene) {
        return nullptr;
    }
    const auto found = state().bindings.find(character);
    return found == state().bindings.end() ? nullptr : &found->second;
}
bool confident(const BodyPerson &person, const ExternalRigMapping &map, const std::string &joint) {
    const auto found = person.joints.find(joint);
    return found != person.joints.end() && found->second[3] >= map.min_confidence;
}

void resetProxyMotion(SceneData &scene, const std::string &character) {
    for (const auto &set : scene.kinematic_colliders.sets()) {
        if (set.target_character == character) {
            scene.kinematic_colliders.resetMotionHistory(set.id);
        }
    }
}

void cacheConfidenceMask(const SceneData::ImportedModelContext &model, Binding &binding,
                         const BodyPerson *person) {
    binding.unreliable_bones.clear();
    if (!person) {
        return;
    }
    for (const auto &control : model.rigAnatomy.controls) {
        const auto mapped = binding.mapping.controls.find(control.name);
        if (mapped == binding.mapping.controls.end() ||
            (confident(*person, binding.mapping, mapped->second.joint) &&
             (mapped->second.pole.empty() ||
              confident(*person, binding.mapping, mapped->second.pole)))) {
            continue;
        }
        for (const auto &bone : model.nodeHierarchy.nodes) {
            const auto *node = &bone;
            for (std::size_t depth = 0; node && depth < model.nodeHierarchy.size(); ++depth) {
                if (node->uniqueName == control.root) {
                    binding.unreliable_bones.insert(bone.uniqueName);
                    break;
                }
                node = node->parent >= 0 &&
                               static_cast<std::size_t>(node->parent) < model.nodeHierarchy.size()
                           ? &model.nodeHierarchy.nodes[static_cast<std::size_t>(node->parent)]
                           : nullptr;
            }
        }
    }
}
} // namespace

bool bindExternalBodyRig(SceneData &scene, const std::string &character, int person_id,
                         const nlohmann::json &config, std::string &error) {
    auto *model = modelFor(scene, character);
    if (!model || model->nodeHierarchy.empty() || model->skeletonNodes.empty()) {
        error = "external_pose_requires_character_skeleton";
        return false;
    }
    if (person_id < 0 || (scene.rigView.pose.active && scene.rigView.pose.character == character) ||
        (scene.rigView.edit_mode && scene.rigView.edit_character == character)) {
        error = "external_pose_invalid_person_or_rig_edit_active";
        return false;
    }
    if (state().owner && state().owner != &scene) {
        error = "external_pose_different_scene_active";
        return false;
    }
    if (!state().bindings.count(character) && state().bindings.size() >= 10) {
        error = "external_pose_binding_limit_10";
        return false;
    }
    Binding staged;
    staged.person_id = person_id;
    staged.config = config;
    staged.revision = model->rigRevision;
    if (!parseExternalRigMapping(config, model->nodeHierarchy, model->rigAnatomy.controls,
                                 staged.mapping, error) ||
        !RigAuthoring::rigScenePlacement(scene, character, staged.placement, error) ||
        !poseCache(scene, *model, model->nodeHierarchy, staged, error)) {
        return false;
    }
    Matrix4x4 inverse;
    if (!RigAuthoring::bindAffineInverse(staged.placement, inverse) ||
        !RigAuthoring::validateJointRules(model->rigAnatomy.joints, model->nodeHierarchy, error)) {
        if (error.empty()) {
            error = "external_pose_invalid_placement";
        }
        return false;
    }
    state().owner = &scene;
    state().bindings[character] = std::move(staged);
    resetProxyMotion(scene, character);
    tickExternalBodyRigs(scene);
    scene.rigView.pose_view_dirty.insert(character);
    return true;
}

void tickExternalBodyRigs(SceneData &scene) {
    if (state().owner != &scene || state().bindings.empty()) {
        return;
    }
    bodyPoses().expire();
    const auto people = bodyPoses().persons();
    const auto packet = bodyPoses().packetCount();
    for (auto &entry : state().bindings) {
        auto &binding = entry.second;
        auto *model = modelFor(scene, entry.first);
        if (!model || model->rigRevision != binding.revision) {
            if (binding.state != "rig_changed_rebind_required") {
                resetProxyMotion(scene, entry.first);
                ++binding.motion_resets;
            }
            binding.state = "rig_changed_rebind_required";
            binding.error = binding.state;
            binding.evaluated = false;
            continue;
        }
        if ((scene.rigView.pose.active && scene.rigView.pose.character == entry.first) ||
            (scene.rigView.edit_mode && scene.rigView.edit_character == entry.first)) {
            if (binding.state != "rig_edit_conflict") {
                resetProxyMotion(scene, entry.first);
                ++binding.motion_resets;
            }
            binding.state = "rig_edit_conflict";
            binding.evaluated = false;
            continue;
        }
        Matrix4x4 placement;
        std::string error;
        if (!RigAuthoring::rigScenePlacement(scene, entry.first, placement, error)) {
            binding.error = error;
            binding.evaluated = false;
            continue;
        }
        const auto person = people.find(binding.person_id);
        const bool have_person =
            person != people.end() && person->second.confidence >= binding.mapping.min_confidence &&
            confident(person->second, binding.mapping, binding.mapping.anchor_joint);
        if (binding.evaluated && packet == binding.last_packet &&
            have_person == binding.had_person && sameMatrix(placement, binding.placement)) {
            continue;
        }
        const auto begin = Clock::now();
        Vec3 anchor = binding.last_anchor;
        if (have_person) {
            const auto &point = person->second.joints.at(binding.mapping.anchor_joint);
            anchor = Vec3(static_cast<float>(point[0]), static_cast<float>(point[1]),
                          static_cast<float>(point[2]));
        }
        if (have_person != binding.had_person || !sameMatrix(placement, binding.placement) ||
            (have_person &&
             (anchor - binding.last_anchor).length() > binding.mapping.teleport_metres)) {
            resetProxyMotion(scene, entry.first);
            ++binding.motion_resets;
        }
        binding.last_anchor = anchor;
        cacheConfidenceMask(*model, binding, have_person ? &person->second : nullptr);
        auto pose = model->nodeHierarchy;
        binding.limit_hits.clear();
        const bool solved =
            have_person && solveExternalBodyRig(model->nodeHierarchy, model->rigAnatomy.controls,
                                                model->rigAnatomy.joints, placement, person->second,
                                                binding.mapping, pose, binding.limit_hits, error);
        if (!solved) {
            pose = model->nodeHierarchy; // loss/error restores bind pose, not last body pose
        }
        if (!poseCache(scene, *model, pose, binding, error)) {
            binding.error = error;
            binding.state = "pose_error";
            binding.evaluated = false;
            continue;
        }
        binding.solve_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        binding.solve_max_ms = std::max(binding.solve_max_ms, binding.solve_ms);
        ++binding.solves;
        binding.placement = placement;
        binding.last_packet = packet;
        binding.had_person = have_person;
        binding.evaluated = true;
        binding.state = solved ? "tracking" : (have_person ? "pose_error" : "tracking_lost");
        binding.error = error;
        RigAuthoring::captureGlobals(scene, entry.first, binding.globals, "external_body");
        const auto skin_begin = Clock::now();
        if (skinFlat(scene, entry.first, binding)) {
            ++binding.skin_updates;
        }
        binding.skin_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - skin_begin).count();
    }
}

bool clearExternalBodyRig(SceneData &scene, const std::string &character, std::string &error) {
    const auto found = state().bindings.find(character);
    if (state().owner != &scene || found == state().bindings.end()) {
        error = "unknown_external_pose_binding";
        return false;
    }
    if (auto *model = modelFor(scene, character)) {
        Binding rest;
        std::string rest_error;
        if (!poseCache(scene, *model, model->nodeHierarchy, rest, rest_error)) {
            error = "external_pose_restore_failed: " + rest_error;
            return false;
        }
        skinFlat(scene, character, rest);
        RigAuthoring::captureGlobals(scene, character, rest.globals, "bind");
        model->restPoseApplied = false;
        model->rigPoseViewCpuRestorePending = true;
    }
    scene.rigView.pose_view_dirty.insert(character);
    resetProxyMotion(scene, character);
    state().bindings.erase(found);
    if (state().bindings.empty()) {
        state().owner = nullptr;
    }
    return true;
}

void resetExternalBodyRigs() {
    state() = State{};
}
bool hasExternalBodyRigs(const SceneData &scene) {
    return state().owner == &scene && !state().bindings.empty();
}
bool externalBodyRigOwns(const SceneData &scene, const std::string &character) {
    const auto *binding = findBinding(scene, character);
    if (!binding || !binding->evaluated) {
        return false;
    }
    for (const auto &model : scene.importedModelContexts) {
        if (model.importName == character) {
            return model.rigRevision == binding->revision;
        }
    }
    return false;
}

bool copyExternalBodyRigPose(SceneData &scene, const std::string &character,
                             std::vector<Matrix4x4> &skin, bool &changed) {
    tickExternalBodyRigs(scene);
    if (!externalBodyRigOwns(scene, character)) {
        return false;
    }
    const auto &binding = *findBinding(scene, character);
    if (skin.size() < scene.boneData.getBoneIndexCapacity()) {
        skin.resize(scene.boneData.getBoneIndexCapacity(), Matrix4x4::identity());
    }
    changed = false;
    for (const auto &entry : binding.skin) {
        changed = changed || !sameMatrix(skin[entry.first], entry.second);
        skin[entry.first] = entry.second;
    }
    return true;
}

bool externalBodyRigColliderAvailable(const SceneData &scene, const std::string &character,
                                      std::string &error) {
    const auto *binding = findBinding(scene, character);
    if (!binding) {
        return true; // ordinary authored rig: preserve the existing domain path
    }
    bodyPoses().expire();
    const auto *person = bodyPoses().findPerson(binding->person_id);
    if (!externalBodyRigOwns(scene, character) || binding->state != "tracking" || !person ||
        person->confidence < binding->mapping.min_confidence ||
        !confident(*person, binding->mapping, binding->mapping.anchor_joint)) {
        error = "external_body_person_unresolved";
        return false;
    }
    return true;
}

bool copyExternalBodyRigFrame(SceneData &scene, std::vector<Matrix4x4> &skin) {
    tickExternalBodyRigs(scene);
    if (!hasExternalBodyRigs(scene)) {
        return false;
    }
    bool changed = false;
    for (const auto &entry : state().bindings) {
        bool model_changed = false;
        if (copyExternalBodyRigPose(scene, entry.first, skin, model_changed)) {
            RigAuthoring::captureGlobals(scene, entry.first, entry.second.globals, "external_body");
            const bool skinned = skinFlat(scene, entry.first, entry.second);
            changed = model_changed || skinned || changed;
        }
    }
    return changed;
}

bool externalBodyRigColliderBoneAvailable(const SceneData &scene, const std::string &character,
                                          const std::string &bone) {
    const auto *binding = findBinding(scene, character);
    return !binding || binding->unreliable_bones.count(bone) == 0;
}

nlohmann::json externalBodyRigs(const SceneData &scene) {
    auto rows = nlohmann::json::array();
    if (state().owner == &scene) {
        for (const auto &entry : state().bindings) {
            const auto &binding = entry.second;
            std::size_t sets = 0, proxies = 0;
            for (const auto &set : scene.kinematic_colliders.sets()) {
                if (set.target_character == entry.first) {
                    ++sets;
                    proxies += set.proxies.size();
                }
            }
            std::string availability_error;
            const bool available =
                externalBodyRigColliderAvailable(scene, entry.first, availability_error);
            rows.push_back({{"character", entry.first},
                            {"person_id", binding.person_id},
                            {"collider_available", available},
                            {"collider_error", availability_error},
                            {"config", binding.config},
                            {"state", binding.state},
                            {"last_error", binding.error},
                            {"rig_revision", binding.revision},
                            {"solves", binding.solves},
                            {"skin_updates", binding.skin_updates},
                            {"solve_ms", binding.solve_ms},
                            {"motion_resets", binding.motion_resets},
                            {"solve_max_ms", binding.solve_max_ms},
                            {"skin_ms", binding.skin_ms},
                            {"limit_hits", binding.limit_hits},
                            {"unreliable_bones", binding.unreliable_bones},
                            {"proxy_sets", sets},
                            {"proxies", proxies}});
        }
    }
    return {{"bindings", rows},
            {"timing_scope", "CPU host solve and canonical flat skin only"},
            {"proxy_path", "existing KinematicColliderRegistry / domain consumers; no Jolt world"}};
}
} // namespace Input
