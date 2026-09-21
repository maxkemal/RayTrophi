#include "Animation/RigSplineIK.h"
#include "Animation/RigBindMath.h"
#include <algorithm>
#include <cmath>

namespace RigAuthoring {
bool validateSplineIKPose(const IKPose& pose) {
    if (pose.splineWorld.empty()) {
        return !pose.splineEnabled;
    }
    if (pose.splineWorld.size() != 2) {
        return false;
    }
    for (const auto& point : pose.splineWorld) {
        if (!std::isfinite(point.length_squared())) {
            return false;
        }
    }
    return true;
}

std::vector<Vec3> splineIKWorldCurve(const Vec3& anchor, const IKPose& pose, int samples) {
    std::vector<Vec3> curve;
    if (!validateSplineIKPose(pose) || pose.splineWorld.size() != 2 || samples < 2 ||
        samples > 257) {
        return curve;
    }
    curve.reserve(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const float t = static_cast<float>(i) / (samples - 1);
        const float u = 1 - t;
        const auto point = anchor * (u * u * u) + pose.splineWorld[0] * (3 * u * u * t) +
                           pose.splineWorld[1] * (3 * u * t * t) + pose.target * (t * t * t);
        if (!std::isfinite(point.length_squared())) {
            return {};
        }
        curve.push_back(point);
    }
    return curve;
}

bool seedSplineIKChain(const IKPose& pose, const Matrix4x4& placement,
                       const std::vector<float>& lengths, std::vector<Vec3>& points,
                       std::string& error) {
    if (points.size() < 4 || lengths.size() + 1 != points.size()) {
        error = "rig_ik_invalid_chain";
        return false;
    }
    Matrix4x4 inverse;
    if (!bindAffineInverse(placement, inverse)) {
        error = "rig_ik_invalid_placement";
        return false;
    }
    auto curve = splineIKWorldCurve(placement.transform_point(points.front()), pose, 257);
    if (curve.empty()) {
        error = "rig_ik_invalid_spline";
        return false;
    }
    std::vector<float> distances(curve.size(), 0);
    for (size_t i = 0; i < curve.size(); ++i) {
        curve[i] = inverse.transform_point(curve[i]);
        if (!std::isfinite(curve[i].length_squared())) {
            error = "rig_ik_invalid_spline";
            return false;
        }
        if (i > 0) {
            distances[i] = distances[i - 1] + (curve[i] - curve[i - 1]).length();
        }
    }
    const float curveLength = distances.back();
    float chainLength = 0;
    for (const auto length : lengths) {
        chainLength += length;
    }
    if (!std::isfinite(curveLength) || curveLength < 1e-6f || !std::isfinite(chainLength) ||
        chainLength < 1e-6f) {
        error = "rig_ik_degenerate_spline";
        return false;
    }
    float accumulated = 0;
    for (size_t i = 1; i + 1 < points.size(); ++i) {
        accumulated += lengths[i - 1];
        const float distance = accumulated / chainLength * curveLength;
        const auto upper = std::lower_bound(distances.begin() + 1, distances.end(), distance);
        const auto index = static_cast<size_t>(upper - distances.begin());
        const float segment = distances[index] - distances[index - 1];
        const float t = segment > 1e-8f ? (distance - distances[index - 1]) / segment : 0;
        points[i] = curve[index - 1] + (curve[index] - curve[index - 1]) * t;
    }
    points.back() = curve.back();
    return true;
}
}
