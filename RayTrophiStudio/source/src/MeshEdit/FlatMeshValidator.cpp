#include "MeshEdit/FlatMeshValidator.h"

#include <cmath>

namespace MeshEdit {

namespace {

bool finiteVec3(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

} // namespace

FlatMeshValidation validateFlatMesh(const TriangleMesh& mesh) {
    FlatMeshValidation out;
    if (!mesh.geometry) return out;

    const auto& geom = *mesh.geometry;
    out.vertex_count = static_cast<uint64_t>(geom.get_vertex_count());
    out.triangle_count = static_cast<uint64_t>(geom.indices.size() / 3);

    const Vec3* positions = geom.get_attribute_data<Vec3>("P_orig");
    if (!positions) positions = geom.get_attribute_data<Vec3>("P");
    const Vec3* normals = geom.get_attribute_data<Vec3>("N_orig");
    if (!normals) normals = geom.get_attribute_data<Vec3>("N");

    if (!positions) {
        out.non_finite_vertices = out.vertex_count;
    } else {
        for (size_t i = 0; i < geom.get_vertex_count(); ++i) {
            if (!finiteVec3(positions[i])) ++out.non_finite_vertices;
        }
    }

    if (normals) {
        for (size_t i = 0; i < geom.get_vertex_count(); ++i) {
            if (!finiteVec3(normals[i])) ++out.non_finite_normals;
        }
    }

    for (size_t i = 0; i < geom.indices.size(); ++i) {
        if (geom.indices[i] >= geom.get_vertex_count()) ++out.out_of_range_indices;
    }

    for (size_t tri = 0; tri < out.triangle_count; ++tri) {
        const uint32_t ia = geom.indices[tri * 3 + 0];
        const uint32_t ib = geom.indices[tri * 3 + 1];
        const uint32_t ic = geom.indices[tri * 3 + 2];
        if (ia >= geom.get_vertex_count() || ib >= geom.get_vertex_count() ||
            ic >= geom.get_vertex_count() || !positions) {
            continue;
        }
        const Vec3 e1 = positions[ib] - positions[ia];
        const Vec3 e2 = positions[ic] - positions[ia];
        if (e1.cross(e2).length_squared() <= 1e-14f) ++out.degenerate_triangles;
    }

    out.valid = out.non_finite_vertices == 0 && out.non_finite_normals == 0 &&
                out.out_of_range_indices == 0 && out.degenerate_triangles == 0 &&
                positions != nullptr && geom.indices.size() % 3 == 0;
    return out;
}

} // namespace MeshEdit
