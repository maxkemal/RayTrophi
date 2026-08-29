#include "MeshEdit/ProfileRevolve.h"
#include "MeshEdit/SplineEvaluationService.h"
#include "MeshEdit/SplinePrimitive.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace MeshEdit {
namespace {

Vec3 safeUnit(const Vec3& value) {
    return value.length_squared() > 1.0e-10f ? value.normalize() : Vec3(0.0f, 1.0f, 0.0f);
}

void addTriangle(DNA::GeometryDetail& geometry, uint32_t a, uint32_t b, uint32_t c) {
    geometry.indices.push_back(a);
    geometry.indices.push_back(b);
    geometry.indices.push_back(c);
}

Vec3 revolvePoint(float radius, float height, float angle, ProfileRevolveAxis axis) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    switch (axis) {
    case ProfileRevolveAxis::X: return Vec3(height, radius * c, radius * s);
    case ProfileRevolveAxis::Z: return Vec3(radius * c, radius * s, height);
    case ProfileRevolveAxis::Y:
    default: return Vec3(radius * c, height, -radius * s);
    }
}

Vec3 axisDirection(ProfileRevolveAxis axis) {
    if (axis == ProfileRevolveAxis::X) return Vec3(1.0f, 0.0f, 0.0f);
    if (axis == ProfileRevolveAxis::Z) return Vec3(0.0f, 0.0f, 1.0f);
    return Vec3(0.0f, 1.0f, 0.0f);
}

void orientRevolveOutward(DNA::GeometryDetail& geometry, const Vec3* positions,
                          const ProfileRevolveSettings& settings,
                          bool closedProfile, float sectionCenterRadius,
                          float sectionCenterHeight) {
    if (!positions) return;
    const Vec3 axis = axisDirection(settings.axis);
    double orientationScore = 0.0;
    for (size_t t = 0; t + 2 < geometry.indices.size(); t += 3) {
        const Vec3& a = positions[geometry.indices[t]];
        const Vec3& b = positions[geometry.indices[t + 1]];
        const Vec3& c = positions[geometry.indices[t + 2]];
        const Vec3 face = (b - a).cross(c - a);
        const Vec3 localCenter = (a + b + c) / 3.0f - settings.axis_pivot;
        const float height = localCenter.dot(axis);
        const Vec3 radial = localCenter - axis * height;
        const float radius = radial.length();
        if (radius <= 1.0e-6f) continue;
        const Vec3 radialDirection = radial / radius;
        const Vec3 expected = closedProfile
            ? radialDirection * (radius - sectionCenterRadius) +
              axis * (height - sectionCenterHeight)
            : radialDirection;
        orientationScore += static_cast<double>(face.dot(expected));
    }
    if (orientationScore >= 0.0) return;
    for (size_t t = 0; t + 2 < geometry.indices.size(); t += 3)
        std::swap(geometry.indices[t + 1], geometry.indices[t + 2]);
}

void addSharpPoint(BezierSpline& spline, float radius, float height) {
    spline.points.emplace_back(Vec3(radius, height, 0.0f));
    spline.points.back().autoTangent = false;
    spline.points.back().handleMode = BezierControlPoint::HandleMode::Mirrored;
}

} // namespace

ProfileRevolveResult buildProfileRevolve(const BezierSpline& profile,
                                         const ProfileRevolveSettings& settings) {
    ProfileRevolveResult result;
    result.report.operation_id = "profile.revolve";
    if (profile.points.size() < (profile.isClosed ? 3u : 2u)) {
        result.report.addError("profile_too_small", "Revolve requires an open side profile with two points or a closed profile with three points.");
        return result;
    }
    const float angleSpan = settings.end_angle - settings.start_angle;
    const bool fullRevolution = std::abs(angleSpan - 2.0f * M_PI) <= 1.0e-4f;
    if (settings.angle_segments < 3 || settings.profile_samples < 3 ||
        !std::isfinite(settings.start_angle) || !std::isfinite(settings.end_angle) ||
        angleSpan <= 0.0f || angleSpan > 2.0f * M_PI + 1.0e-4f) {
        result.report.addError("invalid_revolve_sampling", "Revolve segment counts or angle range are invalid.");
        return result;
    }
    std::string splineError;
    if (!SplineEvaluationService::validate(profile, &splineError)) {
        result.report.addError("invalid_spline", splineError);
        return result;
    }

    const int angleCount = settings.angle_segments;
    const int profileCount = settings.profile_samples;
    std::vector<Vec3> section;
    section.reserve(static_cast<size_t>(profileCount));
    float sectionCenterRadius = 0.0f;
    float sectionCenterHeight = 0.0f;
    for (int j = 0; j < profileCount; ++j) {
        const float t = profile.isClosed
            ? static_cast<float>(j) / static_cast<float>(profileCount)
            : static_cast<float>(j) / static_cast<float>(profileCount - 1);
        const Vec3 p = SplineEvaluationService::evaluate(profile, t).position;
        const float radius = p.x + settings.radius_offset;
        if (!std::isfinite(radius) || !std::isfinite(p.y) || radius < -1.0e-5f) {
            result.report.addError("invalid_profile_radius",
                "Profile radius plus Axis Radius Offset must stay non-negative. Increase the offset or move the side profile to +X.");
            return result;
        }
        section.emplace_back(std::max(0.0f, radius), p.y, 0.0f);
        sectionCenterRadius += std::max(0.0f, radius);
        sectionCenterHeight += p.y;
    }
    sectionCenterRadius /= static_cast<float>(profileCount);
    sectionCenterHeight /= static_cast<float>(profileCount);

    const int angleRingCount = fullRevolution ? angleCount : angleCount + 1;
    const uint32_t sideVertexCount = static_cast<uint32_t>(angleRingCount * profileCount);
    std::vector<int32_t> axisCenters(static_cast<size_t>(profileCount), -1);
    uint32_t axisCount = 0;
    for (int j = 0; j < profileCount; ++j) {
        if (section[static_cast<size_t>(j)].x <= 1.0e-5f)
            axisCenters[static_cast<size_t>(j)] = static_cast<int32_t>(sideVertexCount + axisCount++);
    }
    const uint32_t vertexCount = sideVertexCount + axisCount;
    result.geometry = std::make_shared<DNA::GeometryDetail>();
    result.geometry->add_attribute<Vec3>("P_orig");
    result.geometry->add_attribute<Vec3>("P");
    result.geometry->add_attribute<Vec3>("N_orig");
    result.geometry->add_attribute<Vec3>("N");
    result.geometry->add_attribute<Vec2>("uv");
    result.geometry->add_attribute<uint16_t>("materialID");
    result.geometry->resize_vertices(vertexCount);
    Vec3* positionsOrig = result.geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* positions = result.geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* normalsOrig = result.geometry->get_attribute_data_mut<Vec3>("N_orig");
    Vec3* normals = result.geometry->get_attribute_data_mut<Vec3>("N");
    Vec2* uvs = result.geometry->get_attribute_data_mut<Vec2>("uv");
    uint16_t* materials = result.geometry->get_attribute_data_mut<uint16_t>("materialID");
    std::fill(normalsOrig, normalsOrig + vertexCount, Vec3(0.0f));
    std::fill(normals, normals + vertexCount, Vec3(0.0f));

    for (int i = 0; i < angleRingCount; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(angleCount);
        const float angle = settings.start_angle + (settings.end_angle - settings.start_angle) * u;
        for (int j = 0; j < profileCount; ++j) {
            const uint32_t index = static_cast<uint32_t>(i * profileCount + j);
            const Vec3& p = section[static_cast<size_t>(j)];
            const Vec3 position = revolvePoint(p.x, p.y, angle, settings.axis) + settings.axis_pivot;
            positionsOrig[index] = position;
            positions[index] = position;
            const float v = profile.isClosed
                ? static_cast<float>(j) / static_cast<float>(profileCount)
                : static_cast<float>(j) / static_cast<float>(profileCount - 1);
            uvs[index] = Vec2(u, v);
            materials[index] = 0;
        }
    }
    for (int j = 0; j < profileCount; ++j) {
        const int32_t centerIndex = axisCenters[static_cast<size_t>(j)];
        if (centerIndex < 0) continue;
        const uint32_t index = static_cast<uint32_t>(centerIndex);
        positionsOrig[index] = revolvePoint(0.0f, section[static_cast<size_t>(j)].y,
                                             settings.start_angle, settings.axis) + settings.axis_pivot;
        positions[index] = positionsOrig[index];
        uvs[index] = Vec2(0.0f, static_cast<float>(j) / static_cast<float>(profileCount));
        materials[index] = 0;
    }

    const int profileEdgeCount = profile.isClosed ? profileCount : profileCount - 1;
    for (int i = 0; i < angleCount; ++i) {
        const int nextI = fullRevolution ? (i + 1) % angleRingCount : i + 1;
        for (int j = 0; j < profileEdgeCount; ++j) {
            const int nextJ = (j + 1) % profileCount;
            const bool axisJ = section[static_cast<size_t>(j)].x <= 1.0e-5f;
            const bool axisNextJ = section[static_cast<size_t>(nextJ)].x <= 1.0e-5f;
            const uint32_t a = static_cast<uint32_t>(i * profileCount + j);
            const uint32_t b = static_cast<uint32_t>(nextI * profileCount + j);
            const uint32_t c = static_cast<uint32_t>(nextI * profileCount + nextJ);
            const uint32_t d = static_cast<uint32_t>(i * profileCount + nextJ);
            if (!axisJ && !axisNextJ) {
                addTriangle(*result.geometry, a, b, c);
                addTriangle(*result.geometry, a, c, d);
            } else if (axisJ && !axisNextJ) {
                const uint32_t center = static_cast<uint32_t>(axisCenters[static_cast<size_t>(j)]);
                addTriangle(*result.geometry, center, c, d);
            } else if (!axisJ && axisNextJ) {
                const uint32_t center = static_cast<uint32_t>(axisCenters[static_cast<size_t>(nextJ)]);
                addTriangle(*result.geometry, center, a, b);
            }
        }
    }

    orientRevolveOutward(*result.geometry, positions, settings, profile.isClosed,
                         sectionCenterRadius, sectionCenterHeight);

    for (size_t t = 0; t + 2 < result.geometry->indices.size(); t += 3) {
        const uint32_t a = result.geometry->indices[t];
        const uint32_t b = result.geometry->indices[t + 1];
        const uint32_t c = result.geometry->indices[t + 2];
        const Vec3 face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normalsOrig[a] += face; normalsOrig[b] += face; normalsOrig[c] += face;
    }
    for (uint32_t i = 0; i < vertexCount; ++i) {
        normalsOrig[i] = safeUnit(normalsOrig[i]);
        normals[i] = normalsOrig[i];
    }

    result.angle_ring_count = static_cast<uint32_t>(angleRingCount);
    result.profile_ring_count = static_cast<uint32_t>(profileCount);
    result.report.ok = true;
    result.report.changed.vertices_changed = vertexCount;
    result.report.changed.triangles_changed = result.geometry->indices.size() / 3;
    result.report.changed.faces_changed = result.report.changed.triangles_changed;
    return result;
}

BezierSpline makeCupProfile() {
    BezierSpline profile;
    profile.isClosed = true;
    addSharpPoint(profile, 0.0f, 0.0f);
    addSharpPoint(profile, 1.8f, 0.0f);
    addSharpPoint(profile, 2.0f, 0.25f);
    addSharpPoint(profile, 1.9f, 3.0f);
    addSharpPoint(profile, 1.65f, 3.15f);
    addSharpPoint(profile, 1.55f, 0.45f);
    addSharpPoint(profile, 0.3f, 0.45f);
    addSharpPoint(profile, 0.0f, 0.55f);
    return profile;
}

BezierSpline makeBottleProfile() {
    BezierSpline profile;
    profile.isClosed = true;
    addSharpPoint(profile, 0.0f, 0.0f);
    addSharpPoint(profile, 1.55f, 0.0f);
    addSharpPoint(profile, 1.7f, 0.35f);
    addSharpPoint(profile, 1.45f, 2.2f);
    addSharpPoint(profile, 1.05f, 2.7f);
    addSharpPoint(profile, 0.65f, 3.0f);
    addSharpPoint(profile, 0.55f, 3.6f);
    addSharpPoint(profile, 0.42f, 3.75f);
    addSharpPoint(profile, 0.36f, 3.45f);
    addSharpPoint(profile, 0.38f, 2.85f);
    addSharpPoint(profile, 0.72f, 2.45f);
    addSharpPoint(profile, 1.1f, 2.0f);
    addSharpPoint(profile, 1.25f, 0.45f);
    addSharpPoint(profile, 0.3f, 0.45f);
    addSharpPoint(profile, 0.0f, 0.55f);
    return profile;
}

bool runProfileRevolveSelfTest(std::string* details) {
    ProfileRevolveSettings settings;
    settings.angle_segments = 16;
    settings.profile_samples = 12;
    const ProfileRevolveResult result = buildProfileRevolve(makeBottleProfile(), settings);
    BezierSpline openProfile;
    openProfile.isClosed = false;
    addSharpPoint(openProfile, 1.0f, -1.0f);
    addSharpPoint(openProfile, 2.0f, 0.0f);
    addSharpPoint(openProfile, 1.0f, 1.0f);
    ProfileRevolveSettings partialSettings;
    partialSettings.angle_segments = 8;
    partialSettings.profile_samples = 3;
    partialSettings.end_angle = M_PI;
    partialSettings.axis = ProfileRevolveAxis::X;
    partialSettings.axis_pivot = Vec3(2.0f, 3.0f, 4.0f);
    const ProfileRevolveResult partial = buildProfileRevolve(openProfile, partialSettings);
    BezierSpline reversedOpenProfile = openProfile;
    std::reverse(reversedOpenProfile.points.begin(), reversedOpenProfile.points.end());
    const ProfileRevolveResult reversedPartial =
        buildProfileRevolve(reversedOpenProfile, partialSettings);
    bool partialOutward = partial.geometry != nullptr;
    bool partialPivot = partial.geometry != nullptr;
    bool partialUsesAllControls = partial.geometry != nullptr;
    bool reversedOutward = reversedPartial.geometry != nullptr;
    if (partialOutward) {
        const Vec3* positions = partial.geometry->get_attribute_data<Vec3>("P");
        const Vec3* normals = partial.geometry->get_attribute_data<Vec3>("N");
        const Vec3 local = positions ? positions[0] - partialSettings.axis_pivot : Vec3(0.0f);
        partialOutward = positions && normals &&
            normals[0].dot(Vec3(0.0f, local.y, local.z)) > 0.0f;
        partialPivot = positions &&
            (positions[0] - Vec3(1.0f, 4.0f, 4.0f)).length_squared() <= 1.0e-8f;
        partialUsesAllControls = positions &&
            (positions[1] - Vec3(2.0f, 5.0f, 4.0f)).length_squared() <= 1.0e-8f;
    }
    if (reversedOutward) {
        const Vec3* positions = reversedPartial.geometry->get_attribute_data<Vec3>("P");
        const Vec3* normals = reversedPartial.geometry->get_attribute_data<Vec3>("N");
        const Vec3 local = positions ? positions[0] - partialSettings.axis_pivot : Vec3(0.0f);
        reversedOutward = positions && normals &&
            normals[0].dot(Vec3(0.0f, local.y, local.z)) > 0.0f;
    }
    ProfileRevolveSettings torusSettings;
    torusSettings.angle_segments = 12;
    torusSettings.profile_samples = 12;
    torusSettings.radius_offset = 2.0f;
    const ProfileRevolveResult torus = buildProfileRevolve(
        makeSplinePrimitive(SplinePrimitiveType::Circle), torusSettings);
    const bool pass = result.report.ok && result.geometry &&
        result.geometry->get_vertex_count() == 193 &&
        result.geometry->indices.size() == 16u * 12u * 6u &&
        partial.report.ok && partial.geometry && partial.angle_ring_count == 9 &&
        partial.geometry->indices.size() == 8u * 2u * 6u && partialOutward && partialPivot &&
        partialUsesAllControls && reversedOutward &&
        torus.report.ok && torus.geometry;
    if (details) {
        std::ostringstream out;
        out << (pass ? "PASS" : "FAIL") << " vertices="
            << (result.geometry ? result.geometry->get_vertex_count() : 0)
            << " triangles=" << (result.geometry ? result.geometry->indices.size() / 3 : 0)
            << " partial_open=" << (partial.report.ok ? "true" : "false")
            << " partial_rings=" << partial.angle_ring_count
            << " outward_normals=" << (partialOutward ? "true" : "false")
            << " axis_pivot=" << (partialPivot ? "true" : "false")
            << " all_profile_controls=" << (partialUsesAllControls ? "true" : "false")
            << " reversed_profile_outward=" << (reversedOutward ? "true" : "false")
            << " offset_circle=" << (torus.report.ok ? "true" : "false");
        *details = out.str();
    }
    return pass;
}

} // namespace MeshEdit
