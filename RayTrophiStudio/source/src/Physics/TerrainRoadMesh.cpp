#include "TerrainRoadMesh.h"

#include "DNA/GeometryDetail.h"
#include "Matrix4x4.h"
#include "Vec2.h"
#include "Vec3.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace TerrainNodesV2 {
namespace {

Vec3 unitOr(const Vec3& v, const Vec3& fallback) {
    return v.length_squared() > 1.0e-10f ? v.normalize() : fallback;
}

void triangle(DNA::GeometryDetail& geometry, uint32_t a, uint32_t b, uint32_t c) {
    geometry.indices.push_back(a);
    geometry.indices.push_back(b);
    geometry.indices.push_back(c);
}

// One cross-section of the ribbon, in metres from the centreline. The offsets
// mirror the solver's own cross-section: the crown lifts the centreline, the
// core edge sits at route height, and the shoulder continues flat. Anything else
// here would be a second, disagreeing description of the same road.
struct Section {
    float lateral[5];
    float rise[5];
    int count = 0;
};

Section sectionFor(const RoadCarveSettings& settings, float widthMultiplier,
                   bool includeShoulder) {
    const float core = settings.roadWidthMeters * 0.5f * (std::max)(widthMultiplier, 0.0f);
    const float shoulder = includeShoulder ? settings.shoulderWidthMeters : 0.0f;
    Section section;
    if (shoulder > 1.0e-3f) {
        section.count = 5;
        section.lateral[0] = -(core + shoulder); section.rise[0] = 0.0f;
        section.lateral[1] = -core;              section.rise[1] = 0.0f;
        section.lateral[2] = 0.0f;               section.rise[2] = settings.crownMeters;
        section.lateral[3] = core;               section.rise[3] = 0.0f;
        section.lateral[4] = core + shoulder;    section.rise[4] = 0.0f;
    } else {
        section.count = 3;
        section.lateral[0] = -core; section.rise[0] = 0.0f;
        section.lateral[1] = 0.0f;  section.rise[1] = settings.crownMeters;
        section.lateral[2] = core;  section.rise[2] = 0.0f;
    }
    return section;
}

} // namespace

bool buildRoadRibbonGeometry(const RoadSolvedRoute& route,
                             float terrainScaleY,
                             const Matrix4x4& terrainLocalToWorld,
                             const RoadMeshSettings& settings,
                             std::shared_ptr<DNA::GeometryDetail>& out,
                             RoadMeshStats& stats,
                             std::string* error) {
    out.reset();
    stats = RoadMeshStats{};
    if (route.samples.size() < 2) {
        if (error) *error = "solved route has fewer than two samples";
        return false;
    }
    if (!std::isfinite(terrainScaleY) || terrainScaleY <= 0.0f) {
        if (error) *error = "invalid terrain height scale";
        return false;
    }
    if (!std::isfinite(settings.uvMetersPerTile) || settings.uvMetersPerTile <= 1.0e-3f) {
        if (error) *error = "uv_meters_per_tile must be greater than zero";
        return false;
    }

    // Spans, not one strip. A tunnel carries no deck, so the ribbon has to stop
    // at the portal and start again at the far one - drawing straight through
    // would put a road inside the mountain the tunnel exists to avoid.
    std::vector<std::pair<size_t, size_t>> spans;
    size_t index = 0;
    while (index < route.samples.size()) {
        const bool skip = settings.skipTunnels &&
            route.samples[index].crossing == RoadCrossingMode::Tunnel;
        if (skip) { ++index; continue; }
        size_t end = index;
        while (end + 1 < route.samples.size() &&
               !(settings.skipTunnels &&
                 route.samples[end + 1].crossing == RoadCrossingMode::Tunnel)) {
            ++end;
        }
        if (end > index) spans.emplace_back(index, end);
        index = end + 1;
    }
    if (spans.empty()) {
        if (error) *error = "every route sample is inside a tunnel; there is no deck to build";
        return false;
    }

    auto geometry = std::make_shared<DNA::GeometryDetail>();
    geometry->add_attribute<Vec3>("P_orig");
    geometry->add_attribute<Vec3>("P");
    geometry->add_attribute<Vec3>("N_orig");
    geometry->add_attribute<Vec3>("N");
    geometry->add_attribute<Vec2>("uv");
    geometry->add_attribute<uint16_t>("materialID");

    size_t vertexTotal = 0;
    for (const auto& span : spans) {
        const size_t sections = span.second - span.first + 1;
        const Section shape = sectionFor(route.settings, 1.0f, settings.includeShoulder);
        vertexTotal += sections * static_cast<size_t>(shape.count);
    }
    geometry->resize_vertices(vertexTotal);

    Vec3* positionOrig = geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* position = geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* normalOrig = geometry->get_attribute_data_mut<Vec3>("N_orig");
    Vec3* normal = geometry->get_attribute_data_mut<Vec3>("N");
    Vec2* uv = geometry->get_attribute_data_mut<Vec2>("uv");
    uint16_t* materialId = geometry->get_attribute_data_mut<uint16_t>("materialID");
    std::fill(normalOrig, normalOrig + vertexTotal, Vec3(0.0f));
    std::fill(normal, normal + vertexTotal, Vec3(0.0f));

    uint32_t writeCursor = 0;
    for (const auto& span : spans) {
        const size_t first = span.first;
        const size_t last = span.second;
        const uint32_t spanBase = writeCursor;
        int sectionWidth = 0;

        for (size_t i = first; i <= last; ++i) {
            const RoadRouteSample& sample = route.samples[i];
            const size_t previous = i > first ? i - 1 : i;
            const size_t next = i < last ? i + 1 : i;
            const Vec3 tangent = unitOr(
                Vec3(route.samples[next].x - route.samples[previous].x, 0.0f,
                     route.samples[next].z - route.samples[previous].z),
                Vec3(1.0f, 0.0f, 0.0f));
            // Right-hand lateral in the XZ plane. Y is up in terrain-local space,
            // so the cross with up is the road's own across-direction.
            const Vec3 lateral = unitOr(Vec3(tangent.z, 0.0f, -tangent.x),
                                        Vec3(0.0f, 0.0f, 1.0f));
            const Section shape = sectionFor(route.settings, sample.widthMultiplier,
                                             settings.includeShoulder);
            sectionWidth = shape.count;
            const float baseY = sample.height * terrainScaleY + settings.surfaceOffsetMeters;
            const float v = sample.distanceMeters / settings.uvMetersPerTile;
            const float halfWidth = (std::max)(std::fabs(shape.lateral[0]),
                                               std::fabs(shape.lateral[shape.count - 1]));
            for (int k = 0; k < shape.count; ++k) {
                const Vec3 local(sample.x + lateral.x * shape.lateral[k],
                                 baseY + shape.rise[k],
                                 sample.z + lateral.z * shape.lateral[k]);
                const Vec3 world = terrainLocalToWorld.transform_point(local);
                positionOrig[writeCursor] = position[writeCursor] = world;
                const float u = halfWidth > 1.0e-4f
                    ? (shape.lateral[k] / halfWidth) * 0.5f + 0.5f : 0.5f;
                uv[writeCursor] = Vec2(u, v);
                materialId[writeCursor] = 0;
                ++writeCursor;
            }
            if (i > first) {
                const float dx = sample.x - route.samples[i - 1].x;
                const float dz = sample.z - route.samples[i - 1].z;
                stats.lengthMeters += std::sqrt(dx * dx + dz * dz);
            }
            if (sample.crossing == RoadCrossingMode::Bridge) ++stats.bridgeSamples;
        }

        const size_t sections = last - first + 1;
        for (size_t s = 0; s + 1 < sections; ++s) {
            for (int k = 0; k + 1 < sectionWidth; ++k) {
                const uint32_t a = spanBase + static_cast<uint32_t>(s * sectionWidth + k);
                const uint32_t b = a + 1;
                const uint32_t c = b + static_cast<uint32_t>(sectionWidth);
                const uint32_t d = a + static_cast<uint32_t>(sectionWidth);
                triangle(*geometry, a, b, c);
                triangle(*geometry, a, c, d);
            }
        }
        ++stats.spanCount;
    }

    for (const RoadRouteSample& sample : route.samples) {
        if (sample.crossing == RoadCrossingMode::Tunnel) ++stats.tunnelSamples;
    }

    // Winding is fixed by the surface itself: a road deck faces up, so any
    // triangle whose geometric normal points down is reversed. Trusting the
    // curve's own direction instead would flip every road drawn right to left.
    for (size_t i = 0; i + 2 < geometry->indices.size(); i += 3) {
        const uint32_t a = geometry->indices[i];
        const uint32_t b = geometry->indices[i + 1];
        const uint32_t c = geometry->indices[i + 2];
        const Vec3 face = (position[b] - position[a]).cross(position[c] - position[a]);
        if (face.y < 0.0f) std::swap(geometry->indices[i + 1], geometry->indices[i + 2]);
    }
    for (size_t i = 0; i + 2 < geometry->indices.size(); i += 3) {
        const uint32_t a = geometry->indices[i];
        const uint32_t b = geometry->indices[i + 1];
        const uint32_t c = geometry->indices[i + 2];
        const Vec3 face = (position[b] - position[a]).cross(position[c] - position[a]);
        normalOrig[a] += face; normalOrig[b] += face; normalOrig[c] += face;
    }
    for (size_t i = 0; i < vertexTotal; ++i) {
        normal[i] = normalOrig[i] = unitOr(normalOrig[i], Vec3(0.0f, 1.0f, 0.0f));
    }

    if (geometry->indices.empty()) {
        if (error) *error = "route produced no road surface triangles";
        return false;
    }
    stats.vertexCount = vertexTotal;
    stats.triangleCount = geometry->indices.size() / 3;
    out = std::move(geometry);
    return true;
}

} // namespace TerrainNodesV2
