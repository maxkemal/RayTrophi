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
// Stance fractions where the heel rocker ends and the heel starts to rise.
constexpr float kStance = .62f;
constexpr float kHeelRockerEnd = .16f;
constexpr float kHeelOffStart = .6f;
// Cycle fraction by which the chest and arms trail the pelvis.
constexpr float kUpperBodyLag = .05f;
// Pelvis drop smoothing radius as a fraction of one gait cycle.
constexpr float kDropSmoothing = .08f;

// Ground pivots of one foot, stored relative to the ankle in the ankle's rest frame.
struct FootRig {
    std::string control;
    std::string hip;
    std::string ankle;
    std::string toe; // Empty when the anatomy maps no toe role.
    float upperLength = 0, lowerLength = 0, legLength = 0;
    float ground = 0;
    Vec3 ankleRest;
    Vec3 heelRest, ballRest, tipRest;
    Vec3 heelLocal, ballLocal, tipLocal;
    Vec3 tipToeLocal; // Tip relative to the toe joint, used once the toes bend.
    Vec3 bendHint;    // Anatomy knee direction; zero falls back to the character forward.
};

// Where one foot is in its cycle, and the world pitch the roll gave it.
struct FootPhase {
    bool stance = true;
    float s = 0;     // Stance fraction 0..1.
    float t = 0;     // Swing fraction 0..1.
    float pitch = 0; // Degrees, positive lifts the toe.
};

struct WalkContext {
    std::vector<IKControl> controls;
    IKPoses matched;
    std::string pelvis;
    std::string spine;
    std::string chest;
    std::string head;
    std::string leftClavicle;
    std::string rightClavicle;
    FootRig feet[2];
    bool pivotsEstimated = false;
    bool pivotsStored = false;
    bool pivotsDiffer = false;
    Vec3 forward = Vec3(0, 0, 1);
    Vec3 left = Vec3(1, 0, 0);
    Vec3 up = Vec3(0, 1, 0);
    Vec3 pitchAxis = Vec3(-1, 0, 0); // Positive rotation lifts the toe.
    float height = 1.f;
    int frames = 0;
};

struct WalkMetrics {
    float maxReachRatio = 0;
    float footTargetErrorMax = 0;
    float groundPenetrationMax = 0;
    float pelvisDropMin = 0;
    float pelvisDropMax = 0;
    int unreachableFrames = 0;
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

Quaternion axisAngle(const Vec3 &axis, float radians) {
    const float sine = std::sin(radians * .5f);
    return Quaternion(std::cos(radians * .5f), axis.x * sine, axis.y * sine, axis.z * sine);
}

float smoothstep(float t) {
    t = (std::clamp)(t, 0.f, 1.f);
    return t * t * (3 - 2 * t);
}

// Foot pivots come from the anatomy (authored or derived once and stored); without a record
// they are derived from the chain's toe joints the same way rig.derive_limb_hints does.
bool buildFootRig(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                  const std::vector<PreviewJoint> &joints, WalkContext &context, int side,
                  std::string &error) {
    const std::string prefix = side == 0 ? "left_leg" : "right_leg";
    const auto *limb = control(context, prefix);
    auto &foot = context.feet[side];
    foot.control = prefix;
    foot.hip = limb->root;
    foot.ankle = limb->tip;
    foot.bendHint = limb->bendHint;
    const auto hip = origin(joints[index(hierarchy, limb->root)].world);
    const auto knee = origin(joints[index(hierarchy, limb->mid)].world);
    foot.ankleRest = origin(joints[index(hierarchy, limb->tip)].world);
    foot.upperLength = (knee - hip).length();
    foot.lowerLength = (foot.ankleRest - knee).length();
    foot.legLength = foot.upperLength + foot.lowerLength;
    if (!std::isfinite(foot.legLength) || foot.legLength < context.height * 1e-4f) {
        error = "rig_walk_invalid_leg";
        return false;
    }
    RigFootPivot derived;
    bool estimated = false;
    if (!deriveFootPivots(anatomy, hierarchy, prefix, derived, estimated, error)) {
        return false;
    }
    const auto *stored = findFootPivot(anatomy, prefix);
    const auto &pivots = stored ? *stored : derived;
    if (stored) {
        context.pivotsStored = true;
        // A stored record that no longer matches the toes usually means the rest pose was
        // refit after the pivots were written; report it instead of trusting either silently.
        if ((stored->ball - derived.ball).length() > context.height * .02f ||
            (stored->tip - derived.tip).length() > context.height * .02f) {
            context.pivotsDiffer = true;
        }
    } else if (estimated) {
        context.pivotsEstimated = true;
    }
    auto restRotation = Quaternion::fromMatrix(joints[index(hierarchy, foot.ankle)].world);
    restRotation.normalize();
    foot.heelLocal = pivots.heel;
    foot.ballLocal = pivots.ball;
    foot.tipLocal = pivots.tip;
    foot.heelRest = foot.ankleRest + restRotation.rotate(pivots.heel);
    foot.ballRest = foot.ankleRest + restRotation.rotate(pivots.ball);
    foot.tipRest = foot.ankleRest + restRotation.rotate(pivots.tip);
    const auto &up = context.up;
    foot.ground = (std::min)({Vec3::dot(foot.heelRest, up), Vec3::dot(foot.ballRest, up),
                              Vec3::dot(foot.tipRest, up)});
    for (const auto &chain : anatomy.chains) {
        if (chain.name == prefix && chain.bones.size() > 3) {
            foot.toe = chain.bones[3];
            const auto &toeWorld = joints[index(hierarchy, foot.toe)].world;
            auto toeRotation = Quaternion::fromMatrix(toeWorld);
            toeRotation.normalize();
            foot.tipToeLocal = toeRotation.conjugate().rotate(foot.tipRest - origin(toeWorld));
        }
    }
    return true;
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
        recipe.bodyMotion < 0 || recipe.bodyMotion > 1 || !std::isfinite(recipe.legReach) ||
        recipe.legReach < .85f || recipe.legReach > 1 || !std::isfinite(recipe.stanceKnee) ||
        recipe.stanceKnee < 0 || recipe.stanceKnee > 35 || !std::isfinite(recipe.heelStrike) ||
        recipe.heelStrike < 0 || recipe.heelStrike > 30 || !std::isfinite(recipe.toeOff) ||
        recipe.toeOff < 0 || recipe.toeOff > 60) {
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
    context.pitchAxis = Vec3::cross(context.forward, context.up).normalize();
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
    for (int side = 0; side < 2; ++side) {
        if (!buildFootRig(hierarchy, anatomy, joints, context, side, error)) {
            return false;
        }
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
        pose.target = pelvisPosition + lateral - context.up * (context.height * .02f);
        pose.pole = shoulder + context.left * (side * context.height * .18f) -
                    context.forward * (context.height * .16f) -
                    context.up * (context.height * .12f);
    };
    placeArm("left_arm", leftShoulder, 1);
    placeArm("right_arm", rightShoulder, -1);
    error.clear();
    return true;
}

// Rolls the foot about a ground pivot. flatAnkle is where the ankle sits while the sole is
// flat; the pivot keeps its ground position and the ankle swings around it.
void rollFoot(const WalkContext &context, const FootRig &foot, const Vec3 &flatAnkle,
              const Vec3 &pivotRest, float degrees, Vec3 &ankle, Quaternion &delta) {
    delta = axisAngle(context.pitchAxis, degrees * kPi / 180.f);
    const auto pivot = flatAnkle + (pivotRest - foot.ankleRest);
    ankle = pivot + delta.rotate(flatAnkle - pivot);
}

// Stance s in 0..1: heel rocker, flat foot, then heel rise about the ball.
float stanceFoot(const WalkContext &context, const FootRig &foot, const HumanWalkRecipe &recipe,
                 float stride, float s, Vec3 &ankle, Quaternion &delta) {
    const auto flat = foot.ankleRest + context.forward * (stride * (.5f - s));
    if (s < kHeelRockerEnd) {
        const float pitch = recipe.heelStrike * (1 - smoothstep(s / kHeelRockerEnd));
        rollFoot(context, foot, flat, foot.heelRest, pitch, ankle, delta);
        return pitch;
    }
    if (s < kHeelOffStart) {
        rollFoot(context, foot, flat, foot.heelRest, 0, ankle, delta);
        return 0;
    }
    const float t = (s - kHeelOffStart) / (1 - kHeelOffStart);
    const float pitch = -recipe.toeOff * t * t;
    rollFoot(context, foot, flat, foot.ballRest, pitch, ankle, delta);
    return pitch;
}

FootPhase moveFoot(const WalkContext &context, const FootRig &foot, const HumanWalkRecipe &recipe,
                   IKPose &pose, float phase, float stride, float height) {
    Vec3 ankle;
    Quaternion delta;
    FootPhase state;
    if (phase < kStance) {
        state.s = phase / kStance;
        state.pitch = stanceFoot(context, foot, recipe, stride, state.s, ankle, delta);
    } else {
        // Swing blends the toe-off pose into the next heel contact so both ends meet the
        // stance trajectory without a jump.
        const float t = (phase - kStance) / (1 - kStance);
        const float blend = smoothstep(t);
        Vec3 start, end;
        Quaternion unused;
        stanceFoot(context, foot, recipe, stride, 1, start, unused);
        stanceFoot(context, foot, recipe, stride, 0, end, unused);
        const float pitch = -recipe.toeOff + (recipe.heelStrike + recipe.toeOff) * blend;
        delta = axisAngle(context.pitchAxis, pitch * kPi / 180.f);
        ankle = start + (end - start) * blend + context.up * (height * std::sin(kPi * t));
        state.stance = false;
        state.t = t;
        state.pitch = pitch;
    }
    pose.orientationEnabled = true;
    pose.orientationWorld = delta * pose.orientationWorld;
    pose.orientationWorld.normalize();
    pose.target = ankle;
    return state;
}

// Knee flexion of the supporting leg over stance: about 5 degrees at heel contact, a loading
// flexion that absorbs the landing, then nearly straight again before the heel rises. A pelvis
// height that pins the leg at one fixed length instead turns the leg into a bent stick.
float stanceKneeFlex(float s, float loading) {
    if (s < .25f) {
        return 5.f + (loading - 5.f) * smoothstep(s / .25f);
    }
    return loading + (6.f - loading) * smoothstep((s - .25f) / .4f);
}

// How strongly a stance leg sets the pelvis height: fades in after contact and out once the
// heel is rising, when the other leg has taken the weight.
float supportWeight(float s) {
    return smoothstep(s / .08f) * (1 - smoothstep((s - .62f) / .2f));
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

Quaternion parentWorldRotation(const RayTrophi::NodeHierarchy &posed,
                               const std::vector<PreviewJoint> &joints, const std::string &key) {
    Quaternion rotation;
    const int parent = posed.nodes[index(posed, key)].parent;
    if (parent >= 0) {
        rotation = Quaternion::fromMatrix(joints[static_cast<size_t>(parent)].world);
        rotation.normalize();
    }
    return rotation;
}

// Rotates a bone about its own origin by a delta expressed in the character frame, so
// imported bones with oriented parents turn about the same axes as the template.
bool rotateInCharacterFrame(RayTrophi::NodeHierarchy &posed, const std::string &key,
                            const Quaternion &delta, std::string &error) {
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(posed, nullptr, 0, joints, error)) {
        return false;
    }
    const auto parent = parentWorldRotation(posed, joints, key);
    auto &node = posed.nodes[index(posed, key)];
    node.localBind = rotated(node.localBind, parent.conjugate() * delta * parent);
    return true;
}

bool translateInCharacterFrame(RayTrophi::NodeHierarchy &posed, const std::string &key,
                               const Vec3 &offset, std::string &error) {
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(posed, nullptr, 0, joints, error)) {
        return false;
    }
    const auto parent = parentWorldRotation(posed, joints, key);
    auto &node = posed.nodes[index(posed, key)];
    node.localBind = translated(node.localBind, parent.conjugate().rotate(offset));
    return true;
}

// Knees bend toward the pelvis-facing direction. Deriving the pole from the rest bend fails
// when a fitted leg is straight or slightly hyperextended: the solver then picks an arbitrary
// perpendicular and the knee folds sideways or backward without any error.
void placeKneePoles(const WalkContext &context, const RayTrophi::NodeHierarchy &posed,
                    const std::vector<PreviewJoint> &joints, float drop, float hipYaw,
                    IKPoses &poses) {
    const auto yaw = axisAngle(context.up, hipYaw);
    for (const auto &foot : context.feet) {
        const auto hip = origin(joints[index(posed, foot.hip)].world) - context.up * drop;
        const auto facing =
            yaw.rotate(foot.bendHint.length_squared() > 0 ? foot.bendHint : context.forward);
        auto &pose = poses[foot.control];
        pose.pole = (hip + pose.target) * .5f + facing * foot.legLength;
    }
}

// While the heel rises about the ball the toes stay on the ground, so they bend against the
// foot pitch; after toe-off they straighten during the first third of swing.
float toeBend(const HumanWalkRecipe &recipe, const FootPhase &phase) {
    if (phase.stance) {
        return phase.pitch < 0 ? -phase.pitch : 0.f;
    }
    return recipe.toeOff * (1 - smoothstep(phase.t / .3f));
}

// The supporting legs choose the pelvis height through stanceKneeFlex; legReach only caps it
// so no leg is asked to reach beyond its length (the rest leg is almost straight, so without
// the cap the IK clamps and the foot hangs short of its target). bounce is added on top.
float solvePelvisDrop(const WalkContext &context, const HumanWalkRecipe &recipe,
                      const RayTrophi::NodeHierarchy &posed, const std::vector<PreviewJoint> &joints,
                      const IKPoses &poses, const FootPhase phases[2], float bounce,
                      WalkMetrics &metrics) {
    float reachDrop = 0;
    float desired = 0;
    float weights = 0;
    bool unreachable = false;
    for (int side = 0; side < 2; ++side) {
        const auto &foot = context.feet[side];
        const auto hip = origin(joints[index(posed, foot.hip)].world);
        const auto offset = hip - poses.at(foot.control).target;
        const float vertical = Vec3::dot(offset, context.up);
        const float horizontal = (offset - context.up * vertical).length();
        const float allowed = recipe.legReach * foot.legLength;
        if (horizontal >= allowed) {
            // No pelvis height reaches this target; keep the body up and report it.
            unreachable = true;
        } else {
            reachDrop = (std::max)(reachDrop,
                                   vertical - std::sqrt(allowed * allowed - horizontal * horizontal));
        }
        if (!phases[side].stance) {
            continue;
        }
        const float weight = supportWeight(phases[side].s);
        const float flex = stanceKneeFlex(phases[side].s, recipe.stanceKnee) * kPi / 180.f;
        const float a = foot.upperLength, b = foot.lowerLength;
        const float length = std::sqrt(a * a + b * b + 2 * a * b * std::cos(flex));
        if (weight > 0 && length > horizontal) {
            desired += weight * (vertical - std::sqrt(length * length - horizontal * horizontal));
            weights += weight;
        }
    }
    float drop = reachDrop;
    if (weights > 0) {
        drop = (std::max)(desired / weights - bounce, reachDrop);
    }
    metrics.unreachableFrames += unreachable ? 1 : 0;
    return drop;
}

void recordReach(const WalkContext &context, const RayTrophi::NodeHierarchy &posed,
                 const std::vector<PreviewJoint> &joints, const IKPoses &poses, float drop,
                 WalkMetrics &metrics) {
    for (const auto &foot : context.feet) {
        const auto hip = origin(joints[index(posed, foot.hip)].world) - context.up * drop;
        const float ratio = (hip - poses.at(foot.control).target).length() / foot.legLength;
        metrics.maxReachRatio = (std::max)(metrics.maxReachRatio, ratio);
    }
}

// The per-frame drop jumps where the reaching leg hits its cap just before contact, which
// reads as the body thudding down each step. Dilating by r frames and then box-blurring by r
// gives a smooth curve that is never below the input, so every reach cap still holds.
std::vector<float> smoothDrops(const std::vector<float> &drops, int radius) {
    const int count = static_cast<int>(drops.size());
    auto at = [&](const std::vector<float> &values, int i) {
        return values[static_cast<size_t>(((i % count) + count) % count)];
    };
    std::vector<float> dilated(drops.size()), smoothed(drops.size());
    for (int i = 0; i < count; ++i) {
        float value = drops[static_cast<size_t>(i)];
        for (int k = -radius; k <= radius; ++k) {
            value = (std::max)(value, at(drops, i + k));
        }
        dilated[static_cast<size_t>(i)] = value;
    }
    for (int i = 0; i < count; ++i) {
        float sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            sum += at(dilated, i + k);
        }
        smoothed[static_cast<size_t>(i)] = sum / float(2 * radius + 1);
    }
    return smoothed;
}

bool measureFeet(const WalkContext &context, const RayTrophi::NodeHierarchy &solved,
                 const IKPoses &poses, WalkMetrics &metrics, std::string &error) {
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(solved, nullptr, 0, joints, error)) {
        return false;
    }
    for (const auto &foot : context.feet) {
        const auto &world = joints[index(solved, foot.ankle)].world;
        const auto ankle = origin(world);
        metrics.footTargetErrorMax = (std::max)(
            metrics.footTargetErrorMax, (ankle - poses.at(foot.control).target).length());
        auto rotation = Quaternion::fromMatrix(world);
        rotation.normalize();
        Vec3 tip = ankle + rotation.rotate(foot.tipLocal);
        if (!foot.toe.empty()) {
            const auto &toeWorld = joints[index(solved, foot.toe)].world;
            auto toeRotation = Quaternion::fromMatrix(toeWorld);
            toeRotation.normalize();
            tip = origin(toeWorld) + toeRotation.rotate(foot.tipToeLocal);
        }
        for (const auto &point : {ankle + rotation.rotate(foot.heelLocal),
                                  ankle + rotation.rotate(foot.ballLocal), tip}) {
            metrics.groundPenetrationMax = (std::max)(metrics.groundPenetrationMax,
                                                      foot.ground - Vec3::dot(point, context.up));
        }
    }
    return true;
}

// Body pose of one frame before the pelvis height is applied.
struct FrameBody {
    RayTrophi::NodeHierarchy posed;
    IKPoses poses;
    FootPhase phases[2];
    std::vector<PreviewJoint> joints;
    Vec3 bodyOffset;
    float bounce = 0, hipYaw = 0, armWave = 0;
};

bool buildFrameBody(const RayTrophi::NodeHierarchy &hierarchy, const HumanWalkRecipe &recipe,
                    const WalkContext &context, int frame, FrameBody &body, std::string &error) {
    const float stride = recipe.stride * context.height;
    const float stepHeight = recipe.stepHeight * context.height;
    const float cycles = float(frame) * recipe.cycles / context.frames;
    const float phase = cycles - std::floor(cycles);
    body.posed = hierarchy;
    auto &posed = body.posed;
    body.bounce =
        recipe.bodyBounce * context.height * (.5f - .5f * std::cos(4 * kPi * phase));
    const float strideWave = std::sin(2 * kPi * phase);
    const float motion = recipe.bodyMotion;
    const float sideSway = context.height * .03f * motion * strideWave;
    body.hipYaw = 5.f * kPi / 180.f * motion * strideWave;
    const float hipTilt = -3.f * kPi / 180.f * motion * strideWave;
    const float spineLean = 2.f * kPi / 180.f * motion;
    // The chest and arms trail the pelvis slightly; moving every part on one sine with
    // the same phase is a large part of what reads as mechanical.
    const float upperWave = std::sin(2 * kPi * (phase - kUpperBodyLag));
    const float chestYaw = -7.f * kPi / 180.f * motion * upperWave;
    const float chestLean = 3.f * kPi / 180.f * motion * upperWave;
    body.armWave = std::cos(2 * kPi * (phase - kUpperBodyLag));
    body.bodyOffset = context.left * sideSway + context.up * body.bounce;
    // Every body delta is expressed in the character frame (up/forward/left) and moved
    // into each bone's parent space; parents are applied before their children.
    const auto &up = context.up;
    const auto &forward = context.forward;
    const auto hipRotation = axisAngle(up, body.hipYaw) * axisAngle(forward, hipTilt);
    if (!translateInCharacterFrame(posed, context.pelvis, body.bodyOffset, error) ||
        !rotateInCharacterFrame(posed, context.pelvis, hipRotation, error)) {
        return false;
    }
    if (!context.spine.empty()) {
        const auto follow = axisAngle(context.left, spineLean) * axisAngle(up, chestYaw * .35f);
        if (!rotateInCharacterFrame(posed, context.spine, follow, error)) {
            return false;
        }
    }
    if (!context.chest.empty()) {
        const auto counter = axisAngle(up, chestYaw) * axisAngle(forward, chestLean);
        if (!rotateInCharacterFrame(posed, context.chest, counter, error)) {
            return false;
        }
    }
    if (!context.head.empty()) {
        const float bodyYaw = body.hipYaw + chestYaw * 1.35f;
        if (!rotateInCharacterFrame(posed, context.head, axisAngle(up, bodyYaw * -.8f), error)) {
            return false;
        }
    }
    const float shoulderYaw = 3.f * kPi / 180.f * recipe.armSwing * body.armWave;
    for (const auto *clavicle : {&context.leftClavicle, &context.rightClavicle}) {
        if (!clavicle->empty() &&
            !rotateInCharacterFrame(posed, *clavicle, axisAngle(up, shoulderYaw), error)) {
            return false;
        }
    }
    body.poses = context.matched;
    float rightPhase = phase + .5f;
    if (rightPhase >= 1) {
        rightPhase -= 1;
    }
    body.phases[0] = moveFoot(context, context.feet[0], recipe, body.poses["left_leg"], phase,
                              stride, stepHeight);
    body.phases[1] = moveFoot(context, context.feet[1], recipe, body.poses["right_leg"],
                              rightPhase, stride, stepHeight);
    return sampleRigPose(posed, nullptr, 0, body.joints, error);
}

// Shared by preview and bake so the reported metrics describe exactly the stored keys. The
// first pass finds each frame's pelvis drop, the second smooths it and builds the poses.
bool evaluateWalk(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                  const HumanWalkRecipe &recipe, const WalkContext &context, AnimationData *clip,
                  WalkMetrics &metrics, std::string &error) {
    std::vector<std::string> bones;
    for (const auto &node : hierarchy.nodes) {
        bones.push_back(node.uniqueName);
    }
    std::vector<float> drops(static_cast<size_t>(context.frames));
    for (int frame = 0; frame < context.frames; ++frame) {
        FrameBody body;
        if (!buildFrameBody(hierarchy, recipe, context, frame, body, error)) {
            return false;
        }
        drops[static_cast<size_t>(frame)] = solvePelvisDrop(
            context, recipe, body.posed, body.joints, body.poses, body.phases, body.bounce, metrics);
    }
    const int framesPerCycle = context.frames / recipe.cycles;
    drops = smoothDrops(drops, (std::max)(1, int(std::lround(kDropSmoothing * framesPerCycle))));

    const float stride = recipe.stride * context.height;
    const float motion = recipe.bodyMotion;
    metrics.pelvisDropMin = 1e30f;
    for (int frame = 0; frame < context.frames; ++frame) {
        FrameBody body;
        if (!buildFrameBody(hierarchy, recipe, context, frame, body, error)) {
            return false;
        }
        auto &posed = body.posed;
        auto &poses = body.poses;
        const float drop = drops[static_cast<size_t>(frame)];
        recordReach(context, posed, body.joints, poses, drop, metrics);
        metrics.pelvisDropMin = (std::min)(metrics.pelvisDropMin, drop);
        metrics.pelvisDropMax = (std::max)(metrics.pelvisDropMax, drop);
        auto &pelvis = posed.nodes[index(posed, context.pelvis)];
        const auto pelvisParent = parentWorldRotation(posed, body.joints, context.pelvis);
        pelvis.localBind =
            translated(pelvis.localBind, pelvisParent.conjugate().rotate(context.up * -drop));
        placeKneePoles(context, posed, body.joints, drop, body.hipYaw, poses);
        const Vec3 armOffset = body.bodyOffset - context.up * drop;

        poses["left_arm"].target += armOffset;
        poses["left_arm"].pole += armOffset;
        poses["right_arm"].target += armOffset;
        poses["right_arm"].pole += armOffset;
        const float armTravel = stride * .35f * recipe.armSwing * body.armWave;
        const float armArc = context.height * .025f * motion * (1 - std::fabs(body.armWave));
        poses["left_arm"].target -= context.forward * armTravel;
        poses["right_arm"].target += context.forward * armTravel;
        poses["left_arm"].target += context.up * armArc;
        poses["right_arm"].target += context.up * armArc;
        poses["left_arm"].pole -= context.forward * (armTravel * .45f);
        poses["right_arm"].pole += context.forward * (armTravel * .45f);
        poses["left_arm"].pole += context.up * (armArc * 1.5f);
        poses["right_arm"].pole += context.up * (armArc * 1.5f);

        RayTrophi::NodeHierarchy solved, limited;
        std::vector<std::string> hits;
        if (!solveIKPose(posed, context.controls, poses, Matrix4x4::identity(), solved, error) ||
            !constrainJointPose(hierarchy, anatomy.joints, solved, limited, hits, error)) {
            return false;
        }
        for (int side = 0; side < 2; ++side) {
            const float bend = toeBend(recipe, body.phases[side]);
            if (!context.feet[side].toe.empty() && bend != 0 &&
                !rotateInCharacterFrame(limited, context.feet[side].toe,
                                        axisAngle(context.pitchAxis, bend * kPi / 180.f), error)) {
                return false;
            }
        }
        if (!measureFeet(context, limited, poses, metrics, error)) {
            return false;
        }
        if (clip && !insertPoseKeys(*clip, limited, bones, double(frame) / recipe.fps, error)) {
            return false;
        }
    }
    error.clear();
    return true;
}

nlohmann::json point3(const Vec3 &value) {
    return {value.x, value.y, value.z};
}
} // namespace

bool inspectHumanWalkRecipe(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                            const HumanWalkRecipe &recipe, nlohmann::json &output,
                            std::string &error) {
    WalkContext context;
    if (!validateRecipe(hierarchy, anatomy, recipe, context, error)) {
        return false;
    }
    WalkMetrics metrics;
    if (!evaluateWalk(hierarchy, anatomy, recipe, context, nullptr, metrics, error)) {
        return false;
    }
    const float duration = context.frames / recipe.fps;
    auto pivots = nlohmann::json::object();
    for (const auto &foot : context.feet) {
        pivots[foot.control] = {{"heel", point3(foot.heelRest)},
                                {"ball", point3(foot.ballRest)},
                                {"tip", point3(foot.tipRest)},
                                {"ground", foot.ground},
                                {"leg_length", foot.legLength}};
    }
    output = {
        {"kind", "human_walk_in_place_v2"},
        {"frames", context.frames},
        {"end_frame", context.frames - 1},
        {"duration_seconds", duration},
        {"character_height", context.height},
        {"stride_world", recipe.stride * context.height},
        {"step_height_world", recipe.stepHeight * context.height},
        {"body_bounce_world", recipe.bodyBounce * context.height},
        {"body_motion", recipe.bodyMotion},
        {"leg_reach", recipe.legReach},
        {"stance_knee_deg", recipe.stanceKnee},
        {"toe_bend", !context.feet[0].toe.empty() && !context.feet[1].toe.empty()},
        {"upper_body_lag", kUpperBodyLag},
        {"heel_strike_deg", recipe.heelStrike},
        {"toe_off_deg", recipe.toeOff},
        {"controls", {"left_leg", "right_leg", "left_arm", "right_arm"}},
        {"arms_lowered_from_rest", true},
        {"foot_roll", true},
        {"foot_pivots", pivots},
        {"foot_pivots_estimated", context.pivotsEstimated},
        {"foot_pivots_source", context.pivotsStored ? "anatomy" : "derived"},
        {"foot_pivots_differ_from_toes", context.pivotsDiffer},
        {"knee_bend_hint", context.feet[0].bendHint.length_squared() > 0},
        {"pelvis_height_solve", true},
        {"body_frame", "character"},
        {"knee_pole", context.feet[0].bendHint.length_squared() > 0 ? "anatomy_bend_hint"
                                                                    : "character_forward"},
        {"pelvis_sway", true},
        {"torso_counter_rotation", !context.chest.empty()},
        {"head_stabilization", !context.head.empty()},
        {"shoulder_girdle_motion", !context.leftClavicle.empty() && !context.rightClavicle.empty()},
        {"root_motion", false},
        {"loopable", true},
        {"metrics",
         {{"max_reach_ratio", metrics.maxReachRatio},
          {"unreachable_frames", metrics.unreachableFrames},
          {"foot_target_error_max", metrics.footTargetErrorMax},
          {"foot_target_error_fraction", metrics.footTargetErrorMax / context.height},
          {"ground_penetration_max", metrics.groundPenetrationMax},
          {"pelvis_drop_min", metrics.pelvisDropMin},
          {"pelvis_drop_max", metrics.pelvisDropMax}}}};
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
    WalkMetrics metrics;
    if (!evaluateWalk(hierarchy, anatomy, recipe, context, &clip, metrics, error)) {
        return false;
    }
    output = std::move(clip);
    return true;
}
} // namespace RigAuthoring
