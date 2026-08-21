#include "MeshEdit/ProfileRevolve.h"

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
    if (profile.points.size() < 3 || !profile.isClosed) {
        result.report.addError("profile_not_closed", "Revolve requires a closed 2D profile.");
        return result;
    }
    if (settings.angle_segments < 3 || settings.profile_samples < 3 ||
        !std::isfinite(settings.start_angle) || !std::isfinite(settings.end_angle) ||
        settings.end_angle <= settings.start_angle) {
        result.report.addError("invalid_revolve_sampling", "Revolve segment counts or angle range are invalid.");
        return result;
    }

    const int angleCount = settings.angle_segments;
    const int profileCount = settings.profile_samples;
    std::vector<Vec3> section;
    section.reserve(static_cast<size_t>(profileCount));
    for (int j = 0; j < profileCount; ++j) {
        const float t = static_cast<float>(j) / static_cast<float>(profileCount);
        const Vec3 p = profile.samplePosition(t);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < -1.0e-5f) {
            result.report.addError("invalid_profile_radius", "Revolve profile radius must be finite and non-negative.");
            return result;
        }
        section.emplace_back(std::max(0.0f, p.x), p.y, 0.0f);
    }

    const uint32_t sideVertexCount = static_cast<uint32_t>(angleCount * profileCount);
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

    for (int i = 0; i < angleCount; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(angleCount);
        const float angle = settings.start_angle + (settings.end_angle - settings.start_angle) * u;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        for (int j = 0; j < profileCount; ++j) {
            const uint32_t index = static_cast<uint32_t>(i * profileCount + j);
            const Vec3& p = section[static_cast<size_t>(j)];
            const Vec3 position(p.x * c, p.y, p.x * s);
            positionsOrig[index] = position;
            positions[index] = position;
            uvs[index] = Vec2(u, static_cast<float>(j) / static_cast<float>(profileCount));
            materials[index] = 0;
        }
    }
    for (int j = 0; j < profileCount; ++j) {
        const int32_t centerIndex = axisCenters[static_cast<size_t>(j)];
        if (centerIndex < 0) continue;
        const uint32_t index = static_cast<uint32_t>(centerIndex);
        positionsOrig[index] = Vec3(0.0f, section[static_cast<size_t>(j)].y, 0.0f);
        positions[index] = positionsOrig[index];
        uvs[index] = Vec2(0.0f, static_cast<float>(j) / static_cast<float>(profileCount));
        materials[index] = 0;
    }

    for (int i = 0; i < angleCount; ++i) {
        const int nextI = (i + 1) % angleCount;
        for (int j = 0; j < profileCount; ++j) {
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

    for (size_t t = 0; t + 2 < result.geometry->indices.size(); t += 3) {
        const uint32_t a = result.geometry->indices[t];
        const uint32_t b = result.geometry->indices[t + 1];
        const uint32_t c = result.geometry->indices[t + 2];
        const Vec3 face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normalsOrig[a] += face; normalsOrig[b] += face; normalsOrig[c] += face;
    }
    for (uint32_t i = 0; i < vertexCount; ++i) {
        // The generated profile winding is authored in radius/height order;
        // flat renderer face orientation is opposite to that local convention.
        // Publish the outward-facing normal directly so users do not need a
        // destructive manual "flip normals" step after every revolve.
        normalsOrig[i] = safeUnit(normalsOrig[i]) * -1.0f;
        normals[i] = normalsOrig[i];
    }

    result.angle_ring_count = static_cast<uint32_t>(angleCount);
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
    const bool pass = result.report.ok && result.geometry &&
        result.geometry->get_vertex_count() == 193 &&
        result.geometry->indices.size() == 16u * 12u * 6u;
    if (details) {
        std::ostringstream out;
        out << (pass ? "PASS" : "FAIL") << " vertices="
            << (result.geometry ? result.geometry->get_vertex_count() : 0)
            << " triangles=" << (result.geometry ? result.geometry->indices.size() / 3 : 0);
        *details = out.str();
    }
    return pass;
}

} // namespace MeshEdit
