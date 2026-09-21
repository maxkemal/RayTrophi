#include "Animation/RigAimIK.h"
#include "Animation/RigBindMath.h"
#include "Animation/RigPosePreview.h"
#include <algorithm>
#include <cmath>

namespace RigAuthoring {
namespace {
Vec3 origin(const Matrix4x4& matrix) {
    return Vec3(matrix.m[0][3], matrix.m[1][3], matrix.m[2][3]);
}

Quaternion rotation(const Matrix4x4& matrix) {
    Vec3 translation, scale;
    Quaternion value;
    RayTrophi::decomposeTRS(matrix, translation, value, scale);
    value.normalize();
    return value;
}

Matrix4x4 localRotation(const Matrix4x4& old, Quaternion value) {
    value.normalize();
    auto result = value.toMatrix();
    for (int row = 0; row < 3; ++row) {
        result.m[row][3] = old.m[row][3];
    }
    return result;
}

size_t boneIndex(const RayTrophi::NodeHierarchy& hierarchy, const std::string& key) {
    return static_cast<size_t>(hierarchy.find(key) - hierarchy.nodes.data());
}

float handleScale(const RayTrophi::NodeHierarchy& hierarchy, size_t index) {
    for (const auto& node : hierarchy.nodes) {
        if (node.parent == static_cast<int>(index)) {
            const float length = origin(node.localBind).length();
            if (std::isfinite(length) && length > 1e-5f) {
                return length;
            }
        }
    }
    const auto& node = hierarchy.nodes[index];
    const float parentLength = origin(node.localBind).length();
    return std::isfinite(parentLength) && parentLength > 1e-5f ? parentLength : 1.f;
}

Vec3 projected(const Vec3& value, const Vec3& normal) {
    return value - normal * Vec3::dot(value, normal);
}

Quaternion axisAngle(const Vec3& axis, float radians) {
    const float sine = std::sin(radians * .5f);
    return Quaternion(std::cos(radians * .5f), axis.x * sine, axis.y * sine, axis.z * sine);
}
}

bool matchAimIKPose(const RayTrophi::NodeHierarchy& hierarchy, const IKControl& control,
                    const Matrix4x4& placement, IKPose& output, std::string& error) {
    Matrix4x4 inverse;
    if (!bindAffineInverse(placement, inverse)) {
        error = "rig_ik_invalid_placement";
        return false;
    }
    (void)inverse;
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(hierarchy, nullptr, 0, joints, error)) {
        return false;
    }
    const auto index = boneIndex(hierarchy, control.root);
    const auto actorRotation = rotation(joints[index].world);
    const auto position = origin(joints[index].world);
    const float scale = handleScale(hierarchy, index);
    auto staged = output;
    staged.target = placement.transform_point(
        position + actorRotation.rotate(control.aimAxis).normalize() * scale);
    staged.pole = placement.transform_point(
        position + actorRotation.rotate(control.upAxis).normalize() * scale);
    staged.orientationEnabled = false;
    staged.splineEnabled = false;
    staged.splineWorld.clear();
    output = staged;
    error.clear();
    return true;
}

bool solveAimIKPose(const RayTrophi::NodeHierarchy& input, const IKControl& control,
                    const IKPose& pose, const Matrix4x4& placement,
                    RayTrophi::NodeHierarchy& output, std::string& error) {
    Matrix4x4 inverse;
    if (!bindAffineInverse(placement, inverse)) {
        error = "rig_ik_invalid_placement";
        return false;
    }
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(input, nullptr, 0, joints, error)) {
        return false;
    }
    const auto index = boneIndex(input, control.root);
    const auto position = origin(joints[index].world);
    auto direction = inverse.transform_point(pose.target) - position;
    if (direction.length_squared() < 1e-12f) {
        error = "rig_ik_aim_target_at_origin";
        return false;
    }
    direction = direction.normalize();
    const auto oldLocal = input.nodes[index].localBind;
    const auto oldWorld = rotation(joints[index].world);
    auto desiredWorld =
        Quaternion::rotationBetween(oldWorld.rotate(control.aimAxis), direction) * oldWorld;
    desiredWorld.normalize();

    auto desiredUp = projected(inverse.transform_point(pose.pole) - position, direction);
    auto currentUp = projected(desiredWorld.rotate(control.upAxis), direction);
    if (desiredUp.length_squared() > 1e-12f && currentUp.length_squared() > 1e-12f) {
        desiredUp = desiredUp.normalize();
        currentUp = currentUp.normalize();
        const float cosine = (std::max)(-1.f, (std::min)(1.f, Vec3::dot(currentUp, desiredUp)));
        const float sine = Vec3::dot(direction, Vec3::cross(currentUp, desiredUp));
        const auto roll = axisAngle(direction, std::atan2(sine, cosine));
        desiredWorld = roll * desiredWorld;
        desiredWorld.normalize();
    }
    Quaternion parentWorld;
    const int parent = input.nodes[index].parent;
    if (parent >= 0) {
        parentWorld = rotation(joints[static_cast<size_t>(parent)].world);
    }
    parentWorld.normalize();
    auto desiredLocal = parentWorld.conjugate() * desiredWorld;
    desiredLocal.normalize();

    auto staged = input;
    staged.nodes[index].localBind =
        localRotation(oldLocal, Quaternion::slerp(rotation(oldLocal), desiredLocal, pose.blend));
    output = std::move(staged);
    error.clear();
    return true;
}

float aimIKErrorDegrees(const RayTrophi::NodeHierarchy& hierarchy, const IKControl& control,
                        const IKPose& pose, const Matrix4x4& placement) {
    std::vector<PreviewJoint> joints;
    std::string error;
    if (!sampleRigPose(hierarchy, nullptr, 0, joints, error)) {
        return 0.f;
    }
    const auto index = boneIndex(hierarchy, control.root);
    const auto position = placement.transform_point(origin(joints[index].world));
    const auto desired = pose.target - position;
    if (desired.length_squared() < 1e-12f) {
        return 0.f;
    }
    const auto actorAim = rotation(joints[index].world).rotate(control.aimAxis);
    const auto worldAim = placement.transform_vector(actorAim).normalize();
    const float dot = (std::max)(-1.f, (std::min)(1.f, Vec3::dot(worldAim, desired.normalize())));
    return std::acos(dot) * 180.f / 3.14159265359f;
}
}
