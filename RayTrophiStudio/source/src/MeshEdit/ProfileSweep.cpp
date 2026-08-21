#include "MeshEdit/ProfileSweep.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace MeshEdit {
namespace {

struct PathFrame {
    Vec3 position;
    Vec3 tangent;
    Vec3 right;
    Vec3 normal;
};

Vec3 safeUnit(const Vec3& value, const Vec3& fallback) {
    return value.length_squared() > 1.0e-10f ? value.normalize() : fallback;
}

Vec3 chooseRight(const Vec3& tangent, const Vec3& up) {
    Vec3 right = tangent.cross(up);
    if (right.length_squared() <= 1.0e-10f) right = tangent.cross(Vec3(1.0f, 0.0f, 0.0f));
    if (right.length_squared() <= 1.0e-10f) right = tangent.cross(Vec3(0.0f, 0.0f, 1.0f));
    return safeUnit(right, Vec3(1.0f, 0.0f, 0.0f));
}

void addTriangle(DNA::GeometryDetail& geometry, uint32_t a, uint32_t b, uint32_t c) {
    geometry.indices.push_back(a);
    geometry.indices.push_back(b);
    geometry.indices.push_back(c);
}

bool validFinite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

} // namespace

ProfileSweepResult buildProfileSweep(const BezierSpline& profile,
                                     const BezierSpline& path,
                                     const ProfileSweepSettings& settings) {
    ProfileSweepResult result;
    result.report.operation_id = "profile.sweep";

    if (profile.points.size() < 3 || !profile.isClosed) {
        result.report.addError("profile_not_closed", "Profile sweep requires at least three closed profile points.");
        return result;
    }
    if (path.points.size() < 2 || path.isClosed) {
        result.report.addError("path_not_open", "The first sweep slice requires an open path with at least two points.");
        return result;
    }
    if (settings.profile_samples < 3 || settings.path_samples < 2 ||
        !std::isfinite(settings.profile_scale) || settings.profile_scale <= 0.0f) {
        result.report.addError("invalid_sampling", "Profile/path samples and scale are outside the supported range.");
        return result;
    }

    const int profileCount = settings.profile_samples;
    const int pathCount = settings.path_samples;
    std::vector<Vec3> profilePoints;
    profilePoints.reserve(static_cast<size_t>(profileCount));
    for (int j = 0; j < profileCount; ++j) {
        const float t = static_cast<float>(j) / static_cast<float>(profileCount);
        const Vec3 p = profile.samplePosition(t);
        if (!validFinite(p)) {
            result.report.addError("non_finite_profile", "Profile sampling produced a non-finite point.");
            return result;
        }
        profilePoints.emplace_back(p.x, p.y, 0.0f);
    }

    std::vector<PathFrame> frames;
    frames.reserve(static_cast<size_t>(pathCount));
    Vec3 previousRight;
    for (int i = 0; i < pathCount; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(pathCount - 1);
        const Vec3 position = path.samplePosition(t);
        const Vec3 tangent = safeUnit(path.sampleTangent(t), Vec3(0.0f, 0.0f, 1.0f));
        if (!validFinite(position) || !validFinite(tangent)) {
            result.report.addError("non_finite_path", "Path sampling produced a non-finite point or tangent.");
            return result;
        }

        Vec3 right;
        if (i == 0) {
            right = chooseRight(tangent, settings.up);
        } else {
            // Parallel transport avoids the visible frame flips caused by rebuilding
            // every ring from a fixed world-up vector.
            right = previousRight - tangent * previousRight.dot(tangent);
            if (right.length_squared() <= 1.0e-10f) right = chooseRight(tangent, settings.up);
            else right = right.normalize();
        }
        const Vec3 normal = safeUnit(tangent.cross(right), settings.up);
        previousRight = right;
        frames.push_back({position, tangent, right, normal});
    }

    const uint32_t sideVertexCount = static_cast<uint32_t>(pathCount * profileCount);
    const uint32_t capVertices = (settings.cap_start ? 1u : 0u) + (settings.cap_end ? 1u : 0u);
    result.geometry = std::make_shared<DNA::GeometryDetail>();
    result.geometry->add_attribute<Vec3>("P_orig");
    result.geometry->add_attribute<Vec3>("P");
    result.geometry->add_attribute<Vec3>("N_orig");
    result.geometry->add_attribute<Vec3>("N");
    result.geometry->add_attribute<Vec2>("uv");
    result.geometry->add_attribute<uint16_t>("materialID");
    result.geometry->resize_vertices(static_cast<size_t>(sideVertexCount + capVertices));

    Vec3* positionsOrig = result.geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* positions = result.geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* normalsOrig = result.geometry->get_attribute_data_mut<Vec3>("N_orig");
    Vec3* normals = result.geometry->get_attribute_data_mut<Vec3>("N");
    Vec2* uvs = result.geometry->get_attribute_data_mut<Vec2>("uv");
    uint16_t* materials = result.geometry->get_attribute_data_mut<uint16_t>("materialID");
    std::fill(normalsOrig, normalsOrig + sideVertexCount + capVertices, Vec3(0.0f));
    std::fill(normals, normals + sideVertexCount + capVertices, Vec3(0.0f));

    for (int i = 0; i < pathCount; ++i) {
        const PathFrame& frame = frames[static_cast<size_t>(i)];
        for (int j = 0; j < profileCount; ++j) {
            const uint32_t index = static_cast<uint32_t>(i * profileCount + j);
            const Vec3& p = profilePoints[static_cast<size_t>(j)];
            const Vec3 position = frame.position +
                (frame.right * p.x + frame.normal * p.y) * settings.profile_scale;
            positionsOrig[index] = position;
            positions[index] = position;
            uvs[index] = Vec2(static_cast<float>(j) / static_cast<float>(profileCount),
                              static_cast<float>(i) / static_cast<float>(pathCount - 1));
            materials[index] = 0;
        }
    }

    for (int i = 0; i + 1 < pathCount; ++i) {
        for (int j = 0; j < profileCount; ++j) {
            const int nextJ = (j + 1) % profileCount;
            const uint32_t a = static_cast<uint32_t>(i * profileCount + j);
            const uint32_t b = static_cast<uint32_t>(i * profileCount + nextJ);
            const uint32_t c = static_cast<uint32_t>((i + 1) * profileCount + nextJ);
            const uint32_t d = static_cast<uint32_t>((i + 1) * profileCount + j);
            addTriangle(*result.geometry, a, b, c);
            addTriangle(*result.geometry, a, c, d);
        }
    }

    uint32_t nextExtra = sideVertexCount;
    auto addCap = [&](bool start) {
        if (!(start ? settings.cap_start : settings.cap_end)) return;
        const int ring = start ? 0 : pathCount - 1;
        const uint32_t center = nextExtra++;
        positionsOrig[center] = frames[static_cast<size_t>(ring)].position;
        positions[center] = positionsOrig[center];
        normalsOrig[center] = start ? -frames[static_cast<size_t>(ring)].tangent : frames[static_cast<size_t>(ring)].tangent;
        normals[center] = normalsOrig[center];
        uvs[center] = Vec2(0.5f, 0.5f);
        materials[center] = 0;
        for (int j = 0; j < profileCount; ++j) {
            const int nextJ = (j + 1) % profileCount;
            const uint32_t a = static_cast<uint32_t>(ring * profileCount + j);
            const uint32_t b = static_cast<uint32_t>(ring * profileCount + nextJ);
            if (start) addTriangle(*result.geometry, center, b, a);
            else addTriangle(*result.geometry, center, a, b);
        }
    };
    addCap(true);
    addCap(false);

    // Area-weighted normals provide a stable preview and are also the exact data
    // consumed by the flat renderer until a smoothing policy is added.
    for (size_t t = 0; t + 2 < result.geometry->indices.size(); t += 3) {
        const uint32_t a = result.geometry->indices[t];
        const uint32_t b = result.geometry->indices[t + 1];
        const uint32_t c = result.geometry->indices[t + 2];
        const Vec3 face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        normalsOrig[a] += face; normalsOrig[b] += face; normalsOrig[c] += face;
    }
    for (uint32_t i = 0; i < sideVertexCount + capVertices; ++i) {
        normalsOrig[i] = safeUnit(normalsOrig[i], Vec3(0.0f, 1.0f, 0.0f)) * -1.0f;
        normals[i] = normalsOrig[i];
    }

    result.path_ring_count = static_cast<uint32_t>(pathCount);
    result.profile_ring_count = static_cast<uint32_t>(profileCount);
    result.report.ok = true;
    result.report.changed.vertices_changed = sideVertexCount + capVertices;
    result.report.changed.triangles_changed = result.geometry->indices.size() / 3;
    result.report.changed.faces_changed = result.report.changed.triangles_changed;
    return result;
}

bool runProfileSweepSelfTest(std::string* details) {
    BezierSpline profile;
    profile.isClosed = true;
    profile.addPoint(Vec3(-1.0f, -1.0f, 0.0f));
    profile.addPoint(Vec3(1.0f, -1.0f, 0.0f));
    profile.addPoint(Vec3(1.0f, 1.0f, 0.0f));
    profile.addPoint(Vec3(-1.0f, 1.0f, 0.0f));
    BezierSpline path;
    path.addPoint(Vec3(0.0f, 0.0f, 0.0f));
    path.addPoint(Vec3(0.0f, 0.0f, 2.0f));
    ProfileSweepSettings settings;
    settings.path_samples = 3;
    settings.profile_samples = 4;
    const ProfileSweepResult result = buildProfileSweep(profile, path, settings);
    const bool pass = result.report.ok && result.geometry &&
        result.geometry->get_vertex_count() == 14 &&
        result.geometry->indices.size() == 72;
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
