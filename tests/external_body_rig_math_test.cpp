// Native assertions: delivered as source; Codex does not compile/run C++ builds.
#include "Input/ExternalBodyRigMath.h"
#include "Animation/RigPosePreview.h"
#include "KinematicColliderSource.h"
#include <cassert>
#include <cmath>

namespace {
Vec3 worldJoint(const RayTrophi::NodeHierarchy &pose, const Matrix4x4 &placement,
                const std::string &bone) {
    std::vector<RigAuthoring::PreviewJoint> joints;
    std::string error;
    assert(RigAuthoring::sampleRigPose(pose, nullptr, 0, joints, error));
    for (const auto &joint : joints) {
        if (joint.name == bone) {
            return placement.transform_point(joint.world.getTranslation());
        }
    }
    assert(false);
    return Vec3(0);
}
} // namespace

int main() {
    RayTrophi::NodeHierarchy rest;
    rest.addNode("root", "Rig_root", Matrix4x4::identity(), -1);
    rest.addNode("shoulder", "Rig_shoulder", Matrix4x4::translation(0, 1, 0), 0);
    rest.addNode("elbow", "Rig_elbow", Matrix4x4::translation(0.4f, 0, 0), 1);
    rest.addNode("wrist", "Rig_wrist", Matrix4x4::translation(0.4f, 0, 0), 2);
    RigAuthoring::IKControl control;
    control.name = "hand";
    control.root = "Rig_shoulder";
    control.mid = "Rig_elbow";
    control.tip = "Rig_wrist";
    const std::vector<RigAuthoring::IKControl> controls{control};
    nlohmann::json config = {{"anchor_bone", "Rig_root"},
                             {"controls", {{"hand", {{"joint", "l_wrist"}, {"pole", "l_elbow"}}}}}};
    Input::ExternalRigMapping mapping;
    std::string error;
    assert(Input::parseExternalRigMapping(config, rest, controls, mapping, error));
    const auto original_anchor = mapping.anchor_bone;
    auto bad = config;
    bad["min_confidence"] = true;
    assert(!Input::parseExternalRigMapping(bad, rest, controls, mapping, error));
    assert(mapping.anchor_bone == original_anchor);
    bad = config;
    bad["controls"]["hand"]["joint"] = "invented_joint";
    assert(!Input::parseExternalRigMapping(bad, rest, controls, mapping, error));

    Input::BodyPerson body;
    body.id = 1;
    body.confidence = 1;
    for (const auto *name : Input::kBodyJoints) {
        body.joints[name] = {2, 0, 3, 1};
    }
    body.joints["l_wrist"] = {2.6, 1.2, 3.2, 1};
    body.joints["l_elbow"] = {2.2, 1.2, 3.4, 1};
    const auto placement = Matrix4x4::translation(2, 0, 3);
    RayTrophi::NodeHierarchy solved;
    std::vector<std::string> hits;
    assert(Input::solveExternalBodyRig(rest, controls, {}, placement, body, mapping, solved, hits,
                                       error));
    const auto tip = worldJoint(solved, placement, "Rig_wrist");
    assert((tip - Vec3(2.6f, 1.2f, 3.2f)).length() < 0.001f);
    const auto upper =
        worldJoint(solved, placement, "Rig_elbow") - worldJoint(solved, placement, "Rig_shoulder");
    const auto lower = tip - worldJoint(solved, placement, "Rig_elbow");
    assert(std::fabs(upper.length() - 0.4f) < 0.001f);
    assert(std::fabs(lower.length() - 0.4f) < 0.001f);

    RigAuthoring::JointRule root_limit;
    root_limit.bone = "Rig_root";
    root_limit.enabled = true;
    root_limit.lockTranslation = true;
    body.joints["pelvis"][0] += 1;
    body.joints["l_wrist"][0] += 1;
    body.joints["l_elbow"][0] += 1;
    assert(Input::solveExternalBodyRig(rest, controls, {root_limit}, placement, body, mapping,
                                       solved, hits, error));
    assert((worldJoint(solved, placement, "Rig_root") - Vec3(3, 0, 3)).length() < 0.001f);
    body.confidence = 0;
    assert(!Input::solveExternalBodyRig(rest, controls, {}, placement, body, mapping, solved, hits,
                                        error));
    body.confidence = 1;
    body.joints["l_wrist"][3] = 0;
    assert(Input::solveExternalBodyRig(rest, controls, {}, placement, body, mapping, solved, hits,
                                       error));
    assert((worldJoint(solved, placement, "Rig_wrist") - Vec3(3.8f, 1, 3)).length() < 0.001f);

    // Existing registry remains the producer. Sampling a diagnostic COPY cannot
    // steal the solver's motion history or make its next velocity disappear.
    RayTrophiSim::KinematicColliderRegistry registry;
    RayTrophiSim::KinematicProxySet requested, created;
    requested.name = "body";
    requested.target_character = "Rig";
    assert(registry.createSet(requested, created, error));
    RayTrophiSim::KinematicProxyDesc proxy, stored;
    proxy.bone = "Rig_wrist";
    proxy.name = "hand";
    proxy.shape = RayTrophiSim::KinematicProxyShape::Sphere;
    assert(registry.setProxy(created.id, proxy, stored, error));
    Vec3 target(0);
    const RayTrophiSim::KinematicBoneResolver resolver =
        [&](const std::string &, const std::string &, Matrix4x4 &world, std::string &) {
            world = Matrix4x4::translation(target);
            return true;
        };
    std::vector<RayTrophiSim::KinematicProxySample> samples;
    assert(registry.sampleSet(created.id, 0.1f, false, resolver, samples, error));
    assert(samples.size() == 1 && !samples[0].velocity_valid);
    target = Vec3(0.1f, 0, 0);
    auto inspection = registry;
    assert(inspection.sampleSet(created.id, 0.1f, false, resolver, samples, error));
    assert(registry.sampleSet(created.id, 0.1f, false, resolver, samples, error));
    assert(samples[0].velocity_valid && std::fabs(samples[0].linear_velocity.x - 1) < 1e-5f);
    return 0;
}
