#include "Animation/RigChainIK.h"
#include "Animation/RigSplineIK.h"
#include "Animation/AnimationKeys.h"
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
    Vec3 position, scale;
    Quaternion result;
    RayTrophi::decomposeTRS(matrix, position, result, scale);
    result.normalize();
    return result;
}

Matrix4x4 localRotation(const Matrix4x4& original, Quaternion value) {
    value.normalize();
    auto result = value.toMatrix();
    for (int row = 0; row < 3; ++row) {
        result.m[row][3] = original.m[row][3];
    }
    return result;
}

Vec3 direction(Vec3 value, const Vec3& fallback) {
    return value.length_squared() > 1e-12f ? value.normalize() : fallback.normalize();
}
}

bool solveChainIK(const RayTrophi::NodeHierarchy& input, const IKControl& control,
                  const IKPose& pose, const Matrix4x4& placement, RayTrophi::NodeHierarchy& output,
                  std::string& error) {
    Matrix4x4 inverse;
    if (!bindAffineInverse(placement, inverse)) {
        error = "rig_ik_invalid_placement";
        return false;
    }
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(input, nullptr, 0, joints, error)) {
        return false;
    }
    std::vector<size_t> indices;
    std::vector<Vec3> points, original;
    std::vector<float> lengths;
    float total = 0;
    for (const auto& bone : control.chain) {
        const auto* node = input.find(bone);
        if (!node) {
            error = "unknown_bone";
            return false;
        }
        const auto index = static_cast<size_t>(node - input.nodes.data());
        indices.push_back(index);
        points.push_back(origin(joints[index].world));
        if (points.size() > 1) {
            const float length = (points.back() - points[points.size() - 2]).length();
            if (!std::isfinite(length) || length < 1e-6f) {
                error = "rig_ik_zero_length";
                return false;
            }
            lengths.push_back(length);
            total += length;
        }
    }
    if (points.size() < 4 || points.size() > 64) {
        error = "rig_ik_invalid_chain";
        return false;
    }
    original = points;
    const auto anchor = points.front();
    const auto target = inverse.transform_point(pose.target);
    const auto pole = inverse.transform_point(pose.pole);
    const auto offset = target - anchor;
    if (!std::isfinite(offset.length_squared()) ||
        !std::isfinite((pole - anchor).length_squared())) {
        error = "rig_ik_invalid_target";
        return false;
    }
    if (offset.length() >= total) {
        const auto axis = direction(offset, original.back() - anchor);
        for (size_t i = 1; i < points.size(); ++i) {
            points[i] = points[i - 1] + axis * lengths[i - 1];
        }
    } else if (pose.splineEnabled || (points.back() - target).length() > total * 1e-5f) {
        if (pose.splineEnabled) {
            if (!seedSplineIKChain(pose, placement, lengths, points, error)) {
                return false;
            }
            // A straight guide shorter than the chain needs a deterministic bend seed.
            bool straight = true;
            const auto axis = direction(offset, original[1] - anchor);
            for (size_t i = 1; i + 1 < points.size(); ++i) {
                const auto delta = points[i] - anchor;
                if ((delta - axis * delta.dot(axis)).length_squared() > total * total * 1e-10f) {
                    straight = false;
                }
            }
            if (straight) {
                const auto basis = std::fabs(axis.x) < .7f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
                const auto bend = (basis - axis * basis.dot(axis)).normalize();
                for (size_t i = 1; i + 1 < points.size(); ++i) {
                    points[i] += bend * (total * .01f);
                }
            }
        } else {
            // Seed a straight-chain singularity toward the authoring pole.
            const auto axis = direction(offset, original[1] - anchor);
            auto bend = pole - anchor - axis * (pole - anchor).dot(axis);
            if (bend.length_squared() < 1e-12f) {
                const auto basis = std::fabs(axis.x) < .7f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
                bend = basis - axis * basis.dot(axis);
            }
            for (size_t i = 1; i + 1 < points.size(); ++i) {
                points[i] += bend.normalize() * (total * .01f);
            }
        }
        // FABRIK fixes the root and preserves every segment length.
        for (int iteration = 0; iteration < 32; ++iteration) {
            points.back() = target;
            for (size_t i = points.size() - 1; i > 0; --i) {
                const auto axis =
                    direction(points[i - 1] - points[i], original[i - 1] - original[i]);
                points[i - 1] = points[i] + axis * lengths[i - 1];
            }
            points.front() = anchor;
            for (size_t i = 1; i < points.size(); ++i) {
                const auto axis =
                    direction(points[i] - points[i - 1], original[i] - original[i - 1]);
                points[i] = points[i - 1] + axis * lengths[i - 1];
            }
            if ((points.back() - target).length() <= total * 1e-5f) {
                break;
            }
        }
    }
    auto staged = input;
    for (size_t i = 0; i + 1 < indices.size(); ++i) {
        if (!sampleRigPose(staged, nullptr, 0, joints, error)) {
            return false;
        }
        const auto index = indices[i];
        const auto current = origin(joints[indices[i + 1]].world) - origin(joints[index].world);
        const auto desired = points[i + 1] - points[i];
        auto world = Quaternion::rotationBetween(current, desired) * rotation(joints[index].world);
        world.normalize();
        Quaternion parent;
        const int parentIndex = staged.nodes[index].parent;
        if (parentIndex >= 0) {
            parent = rotation(joints[static_cast<size_t>(parentIndex)].world);
        }
        staged.nodes[index].localBind =
            localRotation(input.nodes[index].localBind, parent.conjugate() * world);
    }
    for (size_t i = 0; i + 1 < indices.size(); ++i) {
        const auto index = indices[i];
        staged.nodes[index].localBind =
            localRotation(input.nodes[index].localBind,
                          Quaternion::slerp(rotation(input.nodes[index].localBind),
                                            rotation(staged.nodes[index].localBind), pose.blend));
    }
    if (pose.orientationEnabled) {
        if (!sampleRigPose(staged, nullptr, 0, joints, error)) {
            return false;
        }
        const auto tip = indices.back();
        const auto parent = indices[indices.size() - 2];
        const auto local = rotation(joints[parent].world).conjugate() *
                           rotation(placement).conjugate() * pose.orientationWorld;
        staged.nodes[tip].localBind = localRotation(
            input.nodes[tip].localBind,
            Quaternion::slerp(rotation(input.nodes[tip].localBind), local, pose.blend));
    }
    if (!sampleRigPose(staged, nullptr, 0, joints, error)) {
        return false;
    }
    output = std::move(staged);
    return true;
}
}
