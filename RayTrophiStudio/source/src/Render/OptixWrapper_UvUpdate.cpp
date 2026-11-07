// SPDX-License-Identifier: MIT
#include "OptixWrapper.h"
#include "OptixAccelManager.h"
#include "Backend/OptixBackend.h"
#include "TriangleMesh.h"
#include "PerfProfile.h"

#include <map>
#include <set>
#include <vector>

bool Backend::OptixBackend::updateFlatMeshUVs(const TriangleMesh* mesh) {
    return m_optix && m_optix->updateFlatMeshUVs(mesh);
}

bool OptixWrapper::updateFlatMeshUVs(const TriangleMesh* mesh) {
    RTPERF_SCOPE("uv.optix.attribute_upload");
    if (!accel_manager || !use_tlas_mode || !mesh || !mesh->geometry) {
        return false;
    }
    const auto& geometry = *mesh->geometry;
    const size_t count = geometry.get_vertex_count();
    const Vec2* uv = geometry.get_uvs();
    const uint16_t* materials = geometry.get_material_ids();
    if (!uv || geometry.get_core_attribute_count(DNA::Attr::UV) != count ||
        (materials && geometry.get_core_attribute_count(DNA::Attr::MaterialID) != count)) {
        return false;
    }
    std::set<int> targets;
    for (const auto& instance : accel_manager->getInstances()) {
        if (instance.type == InstanceType::Mesh && instance.source_hittable == mesh) {
            targets.insert(instance.blas_id);
        }
    }
    if (targets.empty()) {
        for (size_t id = 0; id < accel_manager->getMeshCount(); ++id) {
            const auto* blas = accel_manager->getBLAS(static_cast<int>(id));
            if (blas && blas->original_name == mesh->nodeName) {
                return false; // Resident legacy source with no exact flat identity.
            }
        }
    }
    // A shared BLAS may not be rewritten on behalf of only one source mesh.
    for (const auto& instance : accel_manager->getInstances()) {
        if (instance.type == InstanceType::Mesh && targets.count(instance.blas_id) &&
            instance.source_hittable && instance.source_hittable != mesh) {
            return false;
        }
    }
    std::map<int, std::vector<float2>> uploads;
    for (int id : targets) {
        const auto* blas = accel_manager->getBLAS(id);
        if (!blas || !blas->d_uvs || blas->d_tangents) {
            return false;
        }
        std::vector<float2> corners;
        for (size_t face = 0; face < geometry.indices.size() / 3; ++face) {
            const uint32_t first = geometry.indices[face * 3];
            if (first >= count) {
                return false;
            }
            const int material = materials ? materials[first] : 0;
            if (material != blas->material_id) {
                continue;
            }
            for (size_t k = 0; k < 3; ++k) {
                const uint32_t vertex = geometry.indices[face * 3 + k];
                if (vertex >= count) {
                    return false;
                }
                corners.push_back(make_float2(uv[vertex].u, uv[vertex].v));
            }
        }
        if (corners.size() != blas->index_count * 3) {
            return false;
        }
        auto& values = uploads[id];
        if (blas->weld_corner_map.empty()) {
            if (corners.size() != blas->vertex_count) {
                return false;
            }
            values = std::move(corners);
        } else {
            if (blas->weld_corner_map.size() != corners.size()) {
                return false;
            }
            values.resize(blas->vertex_count);
            std::vector<bool> written(blas->vertex_count, false);
            for (size_t corner = 0; corner < corners.size(); ++corner) {
                const uint32_t vertex = blas->weld_corner_map[corner];
                if (vertex >= values.size()) {
                    return false;
                }
                const float2 value = corners[corner];
                if (written[vertex] &&
                    (values[vertex].x != value.x || values[vertex].y != value.y)) {
                    return false; // UV edit separates a GPU-welded vertex; rebuild its layout.
                }
                values[vertex] = value;
                written[vertex] = true;
            }
        }
    }
    if (uploads.empty()) {
        return true; // No resident instance; its next upload reads current flat UVs.
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return false;
    }
    for (const auto& upload : uploads) {
        const auto* blas = accel_manager->getBLAS(upload.first);
        if (cudaMemcpyAsync(reinterpret_cast<void*>(blas->d_uvs), upload.second.data(),
                upload.second.size() * sizeof(float2), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            cudaStreamSynchronize(stream);
            return false;
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return false;
    }
    resetAccumulation();
    return true;
}
