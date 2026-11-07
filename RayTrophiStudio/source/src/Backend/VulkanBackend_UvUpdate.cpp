// SPDX-License-Identifier: MIT
#include "Backend/VulkanBackend.h"
#include "TriangleMesh.h"
#include "PerfProfile.h"

#include <set>
#include <vector>

namespace Backend {

bool VulkanBackendAdapter::updateFlatMeshUVs(const TriangleMesh* mesh) {
    RTPERF_SCOPE("uv.vulkan.attribute_upload");
    if (!m_device || !m_device->isInitialized() || !mesh || !mesh->geometry) {
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    const auto& geometry = *mesh->geometry;
    const size_t count = geometry.get_vertex_count();
    const Vec2* uv = geometry.get_uvs();
    if (!uv || geometry.get_core_attribute_count(DNA::Attr::UV) != count) {
        return false;
    }
    std::vector<float> indexed;
    indexed.reserve(count * 2);
    for (size_t vertex = 0; vertex < count; ++vertex) {
        indexed.insert(indexed.end(), {uv[vertex].u, uv[vertex].v});
    }
    // Validate all resident destinations before writing any GPU buffer.
    std::set<uint32_t> blases;
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        if (m_instanceSources[i].get() != mesh) {
            continue;
        }
        const uint32_t id = m_vkInstances[i].blasIndex;
        const auto source = m_soloBlasIndexedMesh.find(id);
        if (id >= m_device->m_blasList.size() ||
            source == m_soloBlasIndexedMesh.end() || source->second.mesh != mesh) {
            return false;
        }
        const auto& blas = m_device->m_blasList[id];
        const uint64_t bytes = indexed.size() * sizeof(float);
        if (blas.externalGeometry || blas.vertexCount != count || !blas.uvBuffer.buffer ||
            blas.uvBuffer.buffer != blas.vertexBuffer.buffer ||
            blas.uvBuffer.deviceAddress < blas.vertexBuffer.deviceAddress) {
            return false;
        }
        const uint64_t offset = blas.uvBuffer.deviceAddress - blas.vertexBuffer.deviceAddress;
        if (offset > blas.uvBuffer.size || bytes > blas.uvBuffer.size - offset) {
            return false;
        }
        blases.insert(id);
    }
    const std::string key = "[Raster-Solo]-" + mesh->nodeName + "#mesh=" +
        std::to_string(reinterpret_cast<uintptr_t>(mesh));
    auto raster = m_rasterMeshes.find(key);
    std::vector<float> expanded;
    if (raster != m_rasterMeshes.end()) {
        const auto& destination = raster->second;
        const bool is_indexed = destination.indexCount != 0;
        const size_t expected = is_indexed ? count : geometry.indices.size();
        if (destination.vertexCount != expected || !destination.uvBuffer.buffer ||
            destination.uvBuffer.size < expected * 2 * sizeof(float) ||
            (is_indexed && destination.indexCount != geometry.indices.size()) ||
            destination.isScatterProxy || destination.isScatterGroup) {
            return false;
        }
        if (!is_indexed) {
            expanded.reserve(geometry.indices.size() * 2);
            for (uint32_t vertex : geometry.indices) {
                if (vertex >= count) {
                    return false;
                }
                expanded.insert(expanded.end(), {uv[vertex].u, uv[vertex].v});
            }
        }
    } else {
        // A resident legacy/merged mesh needs a rebuild, not a guessed name match.
        for (const auto& instance : m_rasterInstances) {
            if (instance.nodeName == mesh->nodeName) {
                return false;
            }
        }
    }
    if (blases.empty() && raster == m_rasterMeshes.end()) {
        return true; // Not resident yet; the lazy upload will read the edited SoA.
    }
    drainInFlightTraces();
    drainInteractiveViewportInFlight();
    for (uint32_t id : blases) {
        const auto& blas = m_device->m_blasList[id];
        const uint64_t offset = blas.uvBuffer.deviceAddress - blas.vertexBuffer.deviceAddress;
        // UV handles alias the combined geometry allocation: upload at its UV offset.
        m_device->uploadBuffer(blas.uvBuffer, indexed.data(),
                              indexed.size() * sizeof(float), offset);
    }
    if (raster != m_rasterMeshes.end()) {
        auto& destination = raster->second;
        const auto& values = destination.indexCount != 0 ? indexed : expanded;
        m_device->uploadBuffer(destination.uvBuffer, values.data(),
                              values.size() * sizeof(float));
        destination.cpuUVs = values;
        m_interactiveViewport.dirty = true;
    }
    resetAccumulation();
    return true;
}

} // namespace Backend
