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
           std::string& error) {
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
    std::vector<Vec2> result = layout.coordinates;
    std::vector<uint8_t> seen(result.size(), 0);
    for (uint32_t corner = 0; corner < mesh.indexCount; ++corner) {
        const auto& vertex = mesh.vertexArray[mesh.indexArray[corner]];
        const uint32_t source = layout.indices[corner];
        if (vertex.xref != source || vertex.atlasIndex != 0) {
            error = "uv_atlas_remap_required: output needs a corner remap";
            return false;
        }
        const Vec2 uv(vertex.uv[0] / output->width, vertex.uv[1] / output->height);
        if (!std::isfinite(uv.u) || !std::isfinite(uv.v)) {
            error = "uv_atlas_non_finite";
            return false;
        }
        if (seen[source] && (std::abs(result[source].u - uv.u) > 1.0e-6f ||
                             std::abs(result[source].v - uv.v) > 1.0e-6f)) {
            error = "uv_vertex_split_required: no changes applied; corner remap pending";
            return false;
        }
        result[source] = uv;
        seen[source] = 1;
    }
    layout.coordinates = std::move(result);
    buildIslands(layout);
    return true;
#else
    error = "uv_backend_unavailable: install MIT xatlas under external/xatlas and rebuild";
    return false;
#endif
}

} // namespace UV
