#pragma once

#include "Triangle.h"
#include "Vec2.h"
#include "Vec3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace RayTrophiSim {

struct SurfaceMeshTriangle {
    Vec3 p0 = Vec3(0.0f);
    Vec3 p1 = Vec3(0.0f);
    Vec3 p2 = Vec3(0.0f);
    Vec3 normal = Vec3(0.0f, 1.0f, 0.0f);
    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;
    float area = 0.0f;
    int face_index = -1;
    uint16_t material_id = 0;
};

struct SurfaceMeshSample {
    Vec3 position = Vec3(0.0f);
    Vec3 normal = Vec3(0.0f, 1.0f, 0.0f);
    Vec2 uv;
    int triangle_index = -1;
};

struct SurfaceMeshCache {
    std::string node_name;
    std::vector<SurfaceMeshTriangle> triangles;
    Vec3 bounds_min = Vec3(0.0f);
    Vec3 bounds_max = Vec3(0.0f);
    Vec3 centroid = Vec3(0.0f);
    float total_area = 0.0f;
    uint64_t version = 0;

    bool empty() const { return triangles.empty(); }

    static Vec3 transformedVertex(const std::shared_ptr<Triangle>& tri, int index) {
        if (tri && tri->getTransformPtr()) {
            const Matrix4x4 m = tri->getTransformMatrix();
            const Vec3 p = tri->getOriginalVertexPosition(index);
            return m.transform_point(p);
        }
        return index == 0 ? tri->getV0() : (index == 1 ? tri->getV1() : tri->getV2());
    }

    static SurfaceMeshCache build(const std::string& node_name,
                                  const std::vector<std::shared_ptr<Triangle>>& source_triangles,
                                  uint64_t version = 0) {
        SurfaceMeshCache cache;
        cache.node_name = node_name;
        cache.version = version;
        cache.bounds_min = Vec3(std::numeric_limits<float>::max());
        cache.bounds_max = Vec3(-std::numeric_limits<float>::max());

        Vec3 centroid_accum(0.0f);
        std::size_t vertex_count = 0;
        cache.triangles.reserve(source_triangles.size());

        for (const auto& tri : source_triangles) {
            if (!tri) {
                continue;
            }

            SurfaceMeshTriangle entry;
            entry.p0 = transformedVertex(tri, 0);
            entry.p1 = transformedVertex(tri, 1);
            entry.p2 = transformedVertex(tri, 2);
            const Vec3 e0 = entry.p1 - entry.p0;
            const Vec3 e1 = entry.p2 - entry.p0;
            Vec3 normal = Vec3::cross(e0, e1);
            const float normal_len = normal.length();
            if (normal_len <= 1e-8f) {
                continue;
            }
            entry.normal = normal * (1.0f / normal_len);
            entry.area = normal_len * 0.5f;
            entry.face_index = tri->getFaceIndex();
            entry.material_id = tri->getMaterialID();
            std::tie(entry.uv0, entry.uv1, entry.uv2) = tri->getUVCoordinates();

            cache.bounds_min = Vec3::min(cache.bounds_min, entry.p0);
            cache.bounds_min = Vec3::min(cache.bounds_min, entry.p1);
            cache.bounds_min = Vec3::min(cache.bounds_min, entry.p2);
            cache.bounds_max = Vec3::max(cache.bounds_max, entry.p0);
            cache.bounds_max = Vec3::max(cache.bounds_max, entry.p1);
            cache.bounds_max = Vec3::max(cache.bounds_max, entry.p2);
            centroid_accum = centroid_accum + entry.p0 + entry.p1 + entry.p2;
            vertex_count += 3;
            cache.total_area += entry.area;
            cache.triangles.push_back(entry);
        }

        if (cache.triangles.empty()) {
            cache.bounds_min = Vec3(0.0f);
            cache.bounds_max = Vec3(0.0f);
            cache.centroid = Vec3(0.0f);
            cache.total_area = 0.0f;
        } else {
            cache.centroid = centroid_accum * (1.0f / static_cast<float>(vertex_count));
        }
        return cache;
    }

    // Flat (direct SoA) build: a flat TriangleMesh-as-Hittable has no per-face Triangle facades, so
    // build the surface cache straight from its DNA SoA. `positions` are WORLD-space (a flat mesh
    // keeps "P" world-baked), `indices` are 3-per-triangle. uvs / material_ids may be null. Mirrors
    // build()'s bounds/centroid/area accumulation, vertex-indexed instead of per-facade.
    static SurfaceMeshCache buildFromSoA(const std::string& node_name,
                                         const Vec3* positions,
                                         const Vec2* uvs,
                                         const uint16_t* material_ids,
                                         const uint32_t* indices,
                                         std::size_t index_count,
                                         uint64_t version = 0) {
        SurfaceMeshCache cache;
        cache.node_name = node_name;
        cache.version = version;
        cache.bounds_min = Vec3(std::numeric_limits<float>::max());
        cache.bounds_max = Vec3(-std::numeric_limits<float>::max());
        if (!positions || !indices || index_count < 3) {
            cache.bounds_min = Vec3(0.0f);
            cache.bounds_max = Vec3(0.0f);
            return cache;
        }
        Vec3 centroid_accum(0.0f);
        std::size_t vertex_count = 0;
        const std::size_t tri_count = index_count / 3;
        cache.triangles.reserve(tri_count);
        for (std::size_t t = 0; t < tri_count; ++t) {
            const uint32_t i0 = indices[t * 3 + 0];
            const uint32_t i1 = indices[t * 3 + 1];
            const uint32_t i2 = indices[t * 3 + 2];
            SurfaceMeshTriangle entry;
            entry.p0 = positions[i0];
            entry.p1 = positions[i1];
            entry.p2 = positions[i2];
            const Vec3 e0 = entry.p1 - entry.p0;
            const Vec3 e1 = entry.p2 - entry.p0;
            Vec3 normal = Vec3::cross(e0, e1);
            const float normal_len = normal.length();
            if (normal_len <= 1e-8f) {
                continue;
            }
            entry.normal = normal * (1.0f / normal_len);
            entry.area = normal_len * 0.5f;
            entry.face_index = static_cast<int>(t);
            entry.material_id = material_ids ? material_ids[i0] : 0;
            if (uvs) { entry.uv0 = uvs[i0]; entry.uv1 = uvs[i1]; entry.uv2 = uvs[i2]; }

            cache.bounds_min = Vec3::min(cache.bounds_min, entry.p0);
            cache.bounds_min = Vec3::min(cache.bounds_min, entry.p1);
            cache.bounds_min = Vec3::min(cache.bounds_min, entry.p2);
            cache.bounds_max = Vec3::max(cache.bounds_max, entry.p0);
            cache.bounds_max = Vec3::max(cache.bounds_max, entry.p1);
            cache.bounds_max = Vec3::max(cache.bounds_max, entry.p2);
            centroid_accum = centroid_accum + entry.p0 + entry.p1 + entry.p2;
            vertex_count += 3;
            cache.total_area += entry.area;
            cache.triangles.push_back(entry);
        }
        if (cache.triangles.empty()) {
            cache.bounds_min = Vec3(0.0f);
            cache.bounds_max = Vec3(0.0f);
            cache.centroid = Vec3(0.0f);
            cache.total_area = 0.0f;
        } else {
            cache.centroid = centroid_accum * (1.0f / static_cast<float>(vertex_count));
        }
        return cache;
    }

    bool sample(uint32_t seed, SurfaceMeshSample& out_sample) const {
        if (triangles.empty() || total_area <= 1e-8f) {
            return false;
        }

        const auto hashUInt = [](uint32_t value) {
            value ^= value >> 16u;
            value *= 0x7feb352du;
            value ^= value >> 15u;
            value *= 0x846ca68bu;
            value ^= value >> 16u;
            return value;
        };
        const auto hashUnitFloat = [&](uint32_t value) {
            return static_cast<float>(hashUInt(value) & 0x00ffffffu) / static_cast<float>(0x01000000u);
        };

        float pick = hashUnitFloat(seed ^ 0x91e10da5u) * total_area;
        const SurfaceMeshTriangle* chosen = nullptr;
        int chosen_index = -1;
        for (int i = 0; i < static_cast<int>(triangles.size()); ++i) {
            const auto& tri = triangles[static_cast<std::size_t>(i)];
            if (pick <= tri.area) {
                chosen = &tri;
                chosen_index = i;
                break;
            }
            pick -= tri.area;
        }
        if (!chosen) {
            chosen = &triangles.back();
            chosen_index = static_cast<int>(triangles.size()) - 1;
        }

        float u = hashUnitFloat(seed ^ 0x7f4a7c15u);
        float v = hashUnitFloat(seed ^ 0x123bb5u);
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        const float w = 1.0f - u - v;

        out_sample.position = chosen->p0 * w + chosen->p1 * u + chosen->p2 * v;
        out_sample.normal = chosen->normal;
        out_sample.uv = Vec2(chosen->uv0.u * w + chosen->uv1.u * u + chosen->uv2.u * v,
                             chosen->uv0.v * w + chosen->uv1.v * u + chosen->uv2.v * v);
        out_sample.triangle_index = chosen_index;
        return true;
    }
};

} // namespace RayTrophiSim
