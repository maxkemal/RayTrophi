// SPDX-License-Identifier: MIT
#include "UV/UvLayout.h"

#include <cmath>
#include <memory>
#include <limits>
#include <utility>

#if defined(RAYTROPHI_HAS_XATLAS) && RAYTROPHI_HAS_XATLAS
#include <xatlas.h>
#endif

namespace UV {

bool xatlasAvailable() {
#if defined(RAYTROPHI_HAS_XATLAS) && RAYTROPHI_HAS_XATLAS
    return true;
#else
    return false;
#endif
}

bool atlas(Layout& layout, bool unwrap, uint32_t resolution, uint32_t padding,
           std::string& error, uint32_t* preserved_faces) {
    if (preserved_faces) {
        *preserved_faces = 0;
    }
    if (!validate(layout, error)) {
        return false;
    }
    buildIslands(layout);
    if (resolution < 64 || resolution > 8192 || padding > resolution / 8) {
        error = "uv_invalid_options: resolution 64..8192; padding <= resolution/8";
        return false;
    }
#if defined(RAYTROPHI_HAS_XATLAS) && RAYTROPHI_HAS_XATLAS
    if (layout.coordinates.size() > (std::numeric_limits<uint32_t>::max)() ||
        layout.indices.size() > (std::numeric_limits<uint32_t>::max)()) {
        error = "uv_mesh_too_large";
        return false;
    }
    std::unique_ptr<xatlas::Atlas, decltype(&xatlas::Destroy)> output(
        xatlas::Create(), &xatlas::Destroy);
    if (!output) {
        error = "uv_atlas_allocation_failed";
        return false;
    }
    xatlas::AddMeshError added;
    if (unwrap) {
        xatlas::MeshDecl input;
        input.vertexPositionData = layout.positions.data();
        input.vertexPositionStride = sizeof(Vec3);
        input.vertexCount = static_cast<uint32_t>(layout.positions.size());
        input.indexData = layout.indices.data();
        input.indexCount = static_cast<uint32_t>(layout.indices.size());
        input.indexFormat = xatlas::IndexFormat::UInt32;
        added = xatlas::AddMesh(output.get(), input);
    } else {
        xatlas::UvMeshDecl input;
        input.faceMaterialData = layout.islands.data(); // Keep overlapping islands separate.
        input.vertexUvData = layout.coordinates.data();
        input.vertexStride = sizeof(Vec2);
        input.vertexCount = static_cast<uint32_t>(layout.coordinates.size());
        input.indexData = layout.indices.data();
        input.indexCount = static_cast<uint32_t>(layout.indices.size());
        input.indexFormat = xatlas::IndexFormat::UInt32;
        added = xatlas::AddUvMesh(output.get(), input);
    }
    if (added != xatlas::AddMeshError::Success) {
        error = std::string("uv_atlas_input: ") + xatlas::StringForEnum(added);
        return false;
    }
    xatlas::ComputeCharts(output.get());
    xatlas::PackOptions options;
    options.resolution = resolution;
    options.padding = padding;
    options.bilinear = true;
    xatlas::PackCharts(output.get(), options);
    if (output->meshCount != 1 || output->atlasCount != 1 ||
        output->width == 0 || output->height == 0) {
        error = "uv_atlas_failed: this slice requires one atlas";
        return false;
    }
    const xatlas::Mesh& mesh = output->meshes[0];
    if (mesh.indexCount != layout.indices.size()) {
        error = "uv_atlas_topology_changed";
        return false;
    }
    std::vector<Vec2> corners(mesh.indexCount);
    std::vector<uint8_t> degenerate(layout.indices.size() / 3, 0);
    uint32_t preserved = 0;
    for (size_t face = 0; face < degenerate.size(); ++face) {
        const uint32_t a = layout.indices[face * 3];
        const uint32_t b = layout.indices[face * 3 + 1];
        const uint32_t c = layout.indices[face * 3 + 2];
        float area;
        if (unwrap) {
            area = 0.5f * Vec3::cross(layout.positions[b] - layout.positions[a],
                                     layout.positions[c] - layout.positions[a]).length();
        } else {
            const Vec2 x = layout.coordinates[b] - layout.coordinates[a];
            const Vec2 y = layout.coordinates[c] - layout.coordinates[a];
            area = 0.5f * std::abs(x.u * y.v - y.u * x.v);
        }
        // This pinned xatlas version ignores faces at or below FLT_EPSILON area.
        // Keep their previous UVs rather than dropping them or inventing chart UVs.
        if (std::isfinite(area) && area <= (std::numeric_limits<float>::epsilon)()) {
            degenerate[face] = 1;
            ++preserved;
        }
    }
    for (uint32_t corner = 0; corner < mesh.indexCount; ++corner) {
        if (mesh.indexArray[corner] >= mesh.vertexCount) {
            error = "uv_atlas_invalid_output_index";
            return false;
        }
        const auto& vertex = mesh.vertexArray[mesh.indexArray[corner]];
        const uint32_t source = layout.indices[corner];
        if (vertex.xref != source) {
            error = "uv_atlas_remap_required: output face/corner order differs";
            return false;
        }
        if (degenerate[corner / 3]) {
            corners[corner] = layout.coordinates[source];
            continue;
        }
        if (vertex.atlasIndex != 0) {
            error = "uv_atlas_unmapped_face: non-degenerate face has no chart";
            return false;
        }
        const Vec2 uv(vertex.uv[0] / output->width, vertex.uv[1] / output->height);
        if (!std::isfinite(uv.u) || !std::isfinite(uv.v)) {
            error = "uv_atlas_non_finite";
            return false;
        }
        corners[corner] = uv;
    }
    if (!applyCornerCoordinates(layout, corners, error)) {
        return false;
    }
    if (preserved_faces) {
        *preserved_faces = preserved;
    }
    return true;
#else
    error = "uv_backend_unavailable: install MIT xatlas under external/xatlas and rebuild";
    return false;
#endif
}

} // namespace UV
