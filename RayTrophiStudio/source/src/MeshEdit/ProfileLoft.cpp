#include "MeshEdit/ProfileLoft.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace MeshEdit {
namespace {

void tri(DNA::GeometryDetail& g, uint32_t a, uint32_t b, uint32_t c) {
    g.indices.push_back(a); g.indices.push_back(b); g.indices.push_back(c);
}

Vec3 unit(const Vec3& v) {
    return v.length_squared() > 1.0e-10f ? v.normalize() : Vec3(0.0f, 1.0f, 0.0f);
}

} // namespace

ProfileLoftResult buildProfileLoft(const std::vector<const BezierSpline*>& sections,
                                   const ProfileLoftSettings& settings) {
    ProfileLoftResult result;
    result.report.operation_id = "profile.loft";
    if (sections.size() < 2) {
        result.report.addError("sections_required", "Loft requires at least two profile sections.");
        return result;
    }
    if (settings.samples_per_section < 3) {
        result.report.addError("invalid_sampling", "Loft ring size must be at least three.");
        return result;
    }
    for (const BezierSpline* section : sections) {
        if (!section || section->points.size() < 3 || !section->isClosed) {
            result.report.addError("section_not_closed", "Every loft section must be a closed spline with three points.");
            return result;
        }
    }

    const uint32_t sectionCount = static_cast<uint32_t>(sections.size());
    const uint32_t ringSize = static_cast<uint32_t>(settings.samples_per_section);
    const uint32_t capCount = (settings.cap_start ? 1u : 0u) + (settings.cap_end ? 1u : 0u);
    result.geometry = std::make_shared<DNA::GeometryDetail>();
    result.geometry->add_attribute<Vec3>("P_orig");
    result.geometry->add_attribute<Vec3>("P");
    result.geometry->add_attribute<Vec3>("N_orig");
    result.geometry->add_attribute<Vec3>("N");
    result.geometry->add_attribute<Vec2>("uv");
    result.geometry->add_attribute<uint16_t>("materialID");
    result.geometry->resize_vertices(static_cast<size_t>(sectionCount * ringSize + capCount));
    Vec3* p0 = result.geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* p = result.geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* n0 = result.geometry->get_attribute_data_mut<Vec3>("N_orig");
    Vec3* n = result.geometry->get_attribute_data_mut<Vec3>("N");
    Vec2* uv = result.geometry->get_attribute_data_mut<Vec2>("uv");
    uint16_t* mat = result.geometry->get_attribute_data_mut<uint16_t>("materialID");
    std::fill(n0, n0 + sectionCount * ringSize + capCount, Vec3(0.0f));
    std::fill(n, n + sectionCount * ringSize + capCount, Vec3(0.0f));

    for (uint32_t s = 0; s < sectionCount; ++s) {
        for (uint32_t j = 0; j < ringSize; ++j) {
            const float t = static_cast<float>(j) / static_cast<float>(ringSize);
            const uint32_t index = s * ringSize + j;
            const Vec3 value = sections[s]->samplePosition(t);
            if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
                result.report.addError("non_finite_section", "Loft sampling produced a non-finite point.");
                result.geometry.reset();
                return result;
            }
            p0[index] = p[index] = value;
            uv[index] = Vec2(static_cast<float>(j) / static_cast<float>(ringSize),
                             static_cast<float>(s) / static_cast<float>(sectionCount - 1));
            mat[index] = 0;
        }
    }

    for (uint32_t s = 0; s + 1 < sectionCount; ++s) {
        for (uint32_t j = 0; j < ringSize; ++j) {
            const uint32_t nj = (j + 1) % ringSize;
            const uint32_t a = s * ringSize + j;
            const uint32_t b = s * ringSize + nj;
            const uint32_t c = (s + 1) * ringSize + nj;
            const uint32_t d = (s + 1) * ringSize + j;
            tri(*result.geometry, a, b, c); tri(*result.geometry, a, c, d);
        }
    }

    uint32_t extra = sectionCount * ringSize;
    auto cap = [&](bool start) {
        if (!(start ? settings.cap_start : settings.cap_end)) return;
        const uint32_t section = start ? 0u : sectionCount - 1u;
        const uint32_t center = extra++;
        Vec3 average(0.0f);
        for (uint32_t j = 0; j < ringSize; ++j) average += p[section * ringSize + j];
        average = average / static_cast<float>(ringSize);
        p0[center] = p[center] = average;
        uv[center] = Vec2(0.5f, start ? 0.0f : 1.0f); mat[center] = 0;
        for (uint32_t j = 0; j < ringSize; ++j) {
            const uint32_t nj = (j + 1) % ringSize;
            const uint32_t a = section * ringSize + j;
            const uint32_t b = section * ringSize + nj;
            if (start) tri(*result.geometry, center, b, a);
            else tri(*result.geometry, center, a, b);
        }
    };
    cap(true); cap(false);

    for (size_t i = 0; i + 2 < result.geometry->indices.size(); i += 3) {
        const uint32_t a = result.geometry->indices[i], b = result.geometry->indices[i + 1], c = result.geometry->indices[i + 2];
        const Vec3 face = (p[b] - p[a]).cross(p[c] - p[a]);
        n0[a] += face; n0[b] += face; n0[c] += face;
    }
    for (uint32_t i = 0; i < sectionCount * ringSize + capCount; ++i) n[i] = n0[i] = unit(n0[i]);
    result.section_count = sectionCount; result.ring_size = ringSize;
    result.report.ok = true;
    result.report.changed.vertices_changed = sectionCount * ringSize + capCount;
    result.report.changed.triangles_changed = result.geometry->indices.size() / 3;
    result.report.changed.faces_changed = result.report.changed.triangles_changed;
    return result;
}

bool runProfileLoftSelfTest(std::string* details) {
    BezierSpline a, b;
    a.isClosed = b.isClosed = true;
    for (const Vec3& v : {Vec3(-1, -1, 0), Vec3(1, -1, 0), Vec3(1, 1, 0), Vec3(-1, 1, 0)}) a.addPoint(v);
    for (const Vec3& v : {Vec3(-0.5f, -0.5f, 2), Vec3(0.5f, -0.5f, 2), Vec3(0.5f, 0.5f, 2), Vec3(-0.5f, 0.5f, 2)}) b.addPoint(v);
    const std::vector<const BezierSpline*> sections{&a, &b};
    ProfileLoftSettings settings; settings.samples_per_section = 4;
    const auto result = buildProfileLoft(sections, settings);
    const bool pass = result.report.ok && result.geometry && result.geometry->get_vertex_count() == 10;
    if (details) { std::ostringstream out; out << (pass ? "PASS" : "FAIL") << " vertices=" << (result.geometry ? result.geometry->get_vertex_count() : 0); *details = out.str(); }
    return pass;
}

} // namespace MeshEdit
