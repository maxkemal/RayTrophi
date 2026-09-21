#include "Animation/RigWalkRecipe.h"
#include "Animation/AnimationData.h"
#include "Animation/RigAnatomy.h"
#include "Animation/RigIK.h"
#include "Animation/RigJointRules.h"
#include "Animation/RigPoseAuthoringMath.h"
#include "Animation/RigPosePreview.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace RigAuthoring {
namespace {
constexpr float kPi = 3.14159265359f;

struct WalkContext {
    std::vector<IKControl> controls;
    IKPoses matched;
    std::string pelvis;
    std::string spine;
    std::string chest;
    std::string head;
    std::string leftClavicle;
    std::string rightClavicle;
    Vec3 forward = Vec3(0, 0, 1);
    Vec3 left = Vec3(1, 0, 0);
    float height = 1.f;
    int frames = 0;
};

Vec3 origin(const Matrix4x4 &matrix) {
    return Vec3(matrix.m[0][3], matrix.m[1][3], matrix.m[2][3]);
}

size_t index(const RayTrophi::NodeHierarchy &hierarchy, const std::string &key) {
    return static_cast<size_t>(hierarchy.find(key) - hierarchy.nodes.data());
}

const std::string *roleBone(const RigAnatomy &anatomy, const std::string &role) {
    for (const auto &value : anatomy.roles) {
        if (value.role == role) {
            return &value.bone;
        }
    }
    return nullptr;
}

const IKControl *control(const WalkContext &context, const std::string &name) {
    for (const auto &value : context.controls) {
        if (value.name == name) {
            return &value;
        }
    }
    return nullptr;
}

bool validateRecipe(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                    const HumanWalkRecipe &recipe, WalkContext &context, std::string &error) {
    if (anatomy.family != "humanoid") {
        error = "rig_walk_requires_humanoid";
        return false;
    }
    if (!std::isfinite(recipe.fps) || recipe.fps < 1 || recipe.fps > 120 ||
        !std::isfinite(recipe.cadence) || recipe.cadence < 20 || recipe.cadence > 300 ||
        recipe.cycles < 1 || recipe.cycles > 8 || !std::isfinite(recipe.stride) ||
        recipe.stride < .05f || recipe.stride > .8f || !std::isfinite(recipe.stepHeight) ||
        recipe.stepHeight < 0 || recipe.stepHeight > .25f || !std::isfinite(recipe.bodyBounce) ||
        recipe.bodyBounce < 0 || recipe.bodyBounce > .15f || !std::isfinite(recipe.armSwing) ||
        recipe.armSwing < 0 || recipe.armSwing > 1 || !std::isfinite(recipe.bodyMotion) ||
        recipe.bodyMotion < 0 || recipe.bodyMotion > 1) {
        error = "rig_walk_invalid_recipe";
        return false;
    }
    const auto *pelvis = roleBone(anatomy, "pelvis");
    const auto *leftAnkle = roleBone(anatomy, "left_leg.ankle");
    const auto *rightAnkle = roleBone(anatomy, "right_leg.ankle");
    if (!pelvis || !leftAnkle || !rightAnkle) {
        error = "rig_walk_requires_humanoid_roles";
        return false;
    }
    if (!buildLimbIKControls(anatomy, hierarchy, context.controls, error)) {
        return false;
    }
    for (const auto *name : {"left_leg", "right_leg", "left_arm", "right_arm"}) {
        if (!control(context, name)) {
            error = "rig_walk_requires_humanoid_limbs";
            return false;
        }
    }
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(hierarchy, nullptr, 0, joints, error)) {
        return false;
    }
    float minimum = origin(joints.front().world).y;
    float maximum = minimum;
    for (const auto &joint : joints) {
        const float y = origin(joint.world).y;
        minimum = (std::min)(minimum, y);
        maximum = (std::max)(maximum, y);
    }
    context.height = maximum - minimum;
    if (!std::isfinite(context.height) || context.height < 1e-5f) {
        error = "rig_walk_invalid_height";
        return false;
    }
    const auto *leftToe = roleBone(anatomy, "left_leg.toe");
    if (leftToe) {
        auto direction = origin(joints[index(hierarchy, *leftToe)].world) -
                         origin(joints[index(hierarchy, *leftAnkle)].world);
        direction.y = 0;
        if (direction.length_squared() > 1e-12f) {
            context.forward = direction.normalize();
        }
    }
    context.pelvis = *pelvis;
    const double duration = double(recipe.cycles) * 120.0 / recipe.cadence;
    context.frames = static_cast<int>(std::round(duration * recipe.fps));
    if (context.frames < 2 || context.frames > 1000 ||
        static_cast<int64_t>(context.frames) * static_cast<int64_t>(hierarchy.size()) > 100000) {
        error = "rig_walk_sample_limit";
        return false;
    }
    for (const auto &definition : context.controls) {
        IKPose pose;
        if (!matchIKPose(hierarchy, definition, Matrix4x4::identity(), pose, error)) {
            return false;
        }
        pose.enabled = true;
        pose.blend = 1;
        context.matched[definition.name] = pose;
    }
    const auto pelvisPosition = origin(joints[index(hierarchy, context.pelvis)].world);
    const auto leftShoulder =
        origin(joints[index(hierarchy, control(context, "left_arm")->root)].world);
    const auto rightShoulder =
        origin(joints[index(hierarchy, control(context, "right_arm")->root)].world);
    context.left = leftShoulder - rightShoulder;
    context.left.y = 0;
    context.left -= context.forward * Vec3::dot(context.left, context.forward);
    if (context.left.length_squared() < 1e-12f) {
        context.left = Vec3(1, 0, 0);
    } else {
        context.left = context.left.normalize();
    }
    if (const auto *spine = roleBone(anatomy, "spine.lower")) {
        context.spine = *spine;
    }
    if (const auto *chest = roleBone(anatomy, "spine.upper")) {
        context.chest = *chest;
    }
    if (const auto *head = roleBone(anatomy, "head")) {
        context.head = *head;
    }
    if (const auto *clavicle = roleBone(anatomy, "left_arm.clavicle")) {
        context.leftClavicle = *clavicle;
    }
    if (const auto *clavicle = roleBone(anatomy, "right_arm.clavicle")) {
        context.rightClavicle = *clavicle;
    }
    auto placeArm = [&](const char *name, const Vec3 &shoulder, float side) {
        auto &pose = context.matched[name];
        const auto lateral = context.left * (side * context.height * .12f);
        pose.target = pelvisPosition + lateral + Vec3(0, context.height * -.02f, 0);
        pose.pole = shoulder + context.left * (side * context.height * .18f) -
                    context.forward * (context.height * .16f) + Vec3(0, context.height * -.12f, 0);
    };
    placeArm("left_arm", leftShoulder, 1);
    placeArm("right_arm", rightShoulder, -1);
    error.clear();
    return true;
}

Quaternion axisAngle(const Vec3 &axis, float radians) {
    const float sine = std::sin(radians * .5f);
    return Quaternion(std::cos(radians * .5f), axis.x * sine, axis.y * sine, axis.z * sine);
}

void moveFoot(IKPose &pose, float phase, float stride, float height, const Vec3 &forward,
              const Vec3 &left) {
    constexpr float stance = .62f;
    float travel = 0;
    float lift = 0;
    if (phase < stance) {
        travel = stride * (.5f - phase / stance);
    } else {
        const float t = (phase - stance) / (1 - stance);
        const float smooth = t * t * (3 - 2 * t);
        travel = stride * (-.5f + smooth);
        lift = height * std::sin(kPi * t);
        const float pitch = (8 * std::cos(kPi * t) - 4) * std::sin(kPi * t) * kPi / 180;
        pose.orientationWorld = axisAngle(left, pitch) * pose.orientationWorld;
        pose.orientationWorld.normalize();
    }
    pose.orientationEnabled = true;
    pose.target += forward * travel + Vec3(0, lift, 0);
}

Matrix4x4 translated(const Matrix4x4 &source, const Vec3 &offset) {
    auto result = source;
    result.m[0][3] += offset.x;
    result.m[1][3] += offset.y;
    result.m[2][3] += offset.z;
    return result;
}

Matrix4x4 rotated(const Matrix4x4 &source, const Quaternion &delta) {
    Vec3 position, scale;
    Quaternion rotation;
    RayTrophi::decomposeTRS(source, position, rotation, scale);
    rotation = delta * rotation;
    rotation.normalize();
    return Matrix4x4::translation(position) * rotation.toMatrix();
}
} // namespace

bool inspectHumanWalkRecipe(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                            const HumanWalkRecipe &recipe, nlohmann::json &output,
                            std::string &error) {
    WalkContext context;
    if (!validateRecipe(hierarchy, anatomy, recipe, context, error)) {
        return false;
    }
    const float duration = context.frames / recipe.fps;
    output = {
        {"kind", "human_walk_in_place_v1"},
        {"frames", context.frames},
        {"end_frame", context.frames - 1},
        {"duration_seconds", duration},
        {"character_height", context.height},
        {"stride_world", recipe.stride * context.height},
        {"step_height_world", recipe.stepHeight * context.height},
        {"body_bounce_world", recipe.bodyBounce * context.height},
        {"body_motion", recipe.bodyMotion},
        {"controls", {"left_leg", "right_leg", "left_arm", "right_arm"}},
        {"arms_lowered_from_rest", true},
        {"foot_orientation", true},
        {"pelvis_sway", true},
        {"torso_counter_rotation", !context.chest.empty()},
        {"head_stabilization", !context.head.empty()},
        {"shoulder_girdle_motion", !context.leftClavicle.empty() && !context.rightClavicle.empty()},
        {"root_motion", false},
        {"loopable", true}};
    return true;
}

bool buildHumanWalkClip(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                        const HumanWalkRecipe &recipe, AnimationData &output, std::string &error) {
    WalkContext context;
    if (!validateRecipe(hierarchy, anatomy, recipe, context, error)) {
        return false;
    }
    AnimationData clip;
    clip.rigAuthoring = true;
    clip.duration = 1;
    clip.ticksPerSecond = recipe.fps;
    clip.startFrame = 0;
    clip.endFrame = context.frames - 1;
    std::vector<std::string> bones;
    for (const auto &node : hierarchy.nodes) {
        bones.push_back(node.uniqueName);
    }
    const float stride = recipe.stride * context.height;
    const float stepHeight = recipe.stepHeight * context.height;
    const float bounce = recipe.bodyBounce * context.height;
    for (int frame = 0; frame < context.frames; ++frame) {
        const float cycles = float(frame) * recipe.cycles / context.frames;
        const float phase = cycles - std::floor(cycles);
        auto posed = hierarchy;
        const float bounceValue = bounce * (.5f - .5f * std::cos(4 * kPi * phase));
        const float strideWave = std::sin(2 * kPi * phase);
        const float motion = recipe.bodyMotion;
        const float sideSway = context.height * .03f * motion * strideWave;
        const float hipYaw = 5.f * kPi / 180.f * motion * strideWave;
        const float hipTilt = -3.f * kPi / 180.f * motion * strideWave;
        const float spineLean = 2.f * kPi / 180.f * motion;
        const float chestYaw = -7.f * kPi / 180.f * motion * strideWave;
        const float chestLean = 3.f * kPi / 180.f * motion * strideWave;
        const float armWave = std::cos(2 * kPi * phase);
        const Vec3 bodyOffset = context.left * sideSway + Vec3(0, bounceValue, 0);
        auto &pelvis = posed.nodes[index(posed, context.pelvis)];
        pelvis.localBind = translated(pelvis.localBind, bodyOffset);
        const auto hipRotation =
            axisAngle(Vec3(0, 1, 0), hipYaw) * axisAngle(Vec3(0, 0, 1), hipTilt);
        pelvis.localBind = rotated(pelvis.localBind, hipRotation);
        if (!context.spine.empty()) {
            auto &spine = posed.nodes[index(posed, context.spine)];
            const auto follow =
                axisAngle(Vec3(1, 0, 0), spineLean) * axisAngle(Vec3(0, 1, 0), chestYaw * .35f);
            spine.localBind = rotated(spine.localBind, follow);
        }
        if (!context.chest.empty()) {
            auto &chest = posed.nodes[index(posed, context.chest)];
            const auto counter =
                axisAngle(Vec3(0, 1, 0), chestYaw) * axisAngle(Vec3(0, 0, 1), chestLean);
            chest.localBind = rotated(chest.localBind, counter);
        }
        if (!context.head.empty()) {
            auto &head = posed.nodes[index(posed, context.head)];
            const float bodyYaw = hipYaw + chestYaw * 1.35f;
            head.localBind = rotated(head.localBind, axisAngle(Vec3(0, 1, 0), bodyYaw * -.8f));
        }
        const float shoulderYaw = 3.f * kPi / 180.f * recipe.armSwing * armWave;
        for (const auto *clavicle : {&context.leftClavicle, &context.rightClavicle}) {
            if (!clavicle->empty()) {
                auto &shoulder = posed.nodes[index(posed, *clavicle)];
                shoulder.localBind =
                    rotated(shoulder.localBind, axisAngle(Vec3(0, 1, 0), shoulderYaw));
            }
        }

        auto poses = context.matched;
        poses["left_arm"].target += bodyOffset;
        poses["left_arm"].pole += bodyOffset;
        poses["right_arm"].target += bodyOffset;
        poses["right_arm"].pole += bodyOffset;
        moveFoot(poses["left_leg"], phase, stride, stepHeight, context.forward, context.left);
        float rightPhase = phase + .5f;
        if (rightPhase >= 1) {
            rightPhase -= 1;
        }
        moveFoot(poses["right_leg"], rightPhase, stride, stepHeight, context.forward, context.left);
        const float armTravel = stride * .35f * recipe.armSwing * armWave;
        const float armArc = context.height * .025f * motion * (1 - std::fabs(armWave));
        poses["left_arm"].target -= context.forward * armTravel;
        poses["right_arm"].target += context.forward * armTravel;
        poses["left_arm"].target += Vec3(0, armArc, 0);
        poses["right_arm"].target += Vec3(0, armArc, 0);
        poses["left_arm"].pole -= context.forward * (armTravel * .45f);
        poses["right_arm"].pole += context.forward * (armTravel * .45f);
        poses["left_arm"].pole += Vec3(0, armArc * 1.5f, 0);
        poses["right_arm"].pole += Vec3(0, armArc * 1.5f, 0);

        RayTrophi::NodeHierarchy solved, limited;
        std::vector<std::string> hits;
        if (!solveIKPose(posed, context.controls, poses, Matrix4x4::identity(), solved, error) ||
            !constrainJointPose(hierarchy, anatomy.joints, solved, limited, hits, error) ||
            !insertPoseKeys(clip, limited, bones, double(frame) / recipe.fps, error)) {
            return false;
        }
    }
    output = std::move(clip);
    error.clear();
    return true;
}
} // namespace RigAuthoring
