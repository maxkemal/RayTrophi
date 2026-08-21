#pragma once

#include "TriangleMesh.h"

#include <cstdint>

namespace MeshEdit {

struct FlatMeshValidation {
    bool valid = false;
    uint64_t vertex_count = 0;
    uint64_t triangle_count = 0;
    uint64_t non_finite_vertices = 0;
    uint64_t out_of_range_indices = 0;
    uint64_t degenerate_triangles = 0;
    uint64_t non_finite_normals = 0;
};

FlatMeshValidation validateFlatMesh(const TriangleMesh& mesh);

} // namespace MeshEdit
