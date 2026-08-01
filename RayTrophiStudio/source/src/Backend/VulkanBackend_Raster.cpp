// ============================================================================
// VulkanBackend_Raster.cpp
//
// Rasterized viewport path of VulkanBackendAdapter: raster mesh lifetime,
// frustum extraction and culling, instance visibility upload, and the *Impl
// bodies behind the thin buildRaster*/syncRaster*/updateRasterMesh* forwarders
// that remain in VulkanBackend.cpp.
//
// Split out of VulkanBackend.cpp (which had grown past 20,000 lines). These are
// all VulkanBackendAdapter member functions, so the move changes no linkage and
// no declaration: Backend/VulkanBackend.h already declares every one of them.
// ============================================================================

#include "Backend/VulkanBackend.h"
#include "Backend/vulkan_world_data.h"
#include "VulkanBackend_Internal.h"
#include "globals.h"
#include "HittableInstance.h"
#include "InstanceManager.h"
#include "Triangle.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <thread>
#include <vector>
#include <HittableList.h>

namespace {
    // Raster frustum culling is compiled off: the CPU-side chunk test costs more
    // than it saves at current scene sizes. Kept (rather than deleted) because
    // the machinery below is still maintained behind it.
    constexpr bool kRasterFrustumCullingEnabled = false;
}

namespace Backend {

void VulkanBackendAdapter::destroyRasterMesh(RasterMeshBuffer& mesh) {
    if (!m_device) return;
    if (mesh.vertexBuffer.buffer) m_device->destroyBuffer(mesh.vertexBuffer);
    if (mesh.normalBuffer.buffer) m_device->destroyBuffer(mesh.normalBuffer);
    if (mesh.uvBuffer.buffer) m_device->destroyBuffer(mesh.uvBuffer);
    if (mesh.matIdBuffer.buffer) m_device->destroyBuffer(mesh.matIdBuffer);
    if (mesh.indexBuffer.buffer)  m_device->destroyBuffer(mesh.indexBuffer);
    if (mesh.instanceBuffer.buffer) m_device->destroyBuffer(mesh.instanceBuffer);
    if (mesh.baseVertexBuffer.buffer) m_device->destroyBuffer(mesh.baseVertexBuffer);
    if (mesh.baseNormalBuffer.buffer) m_device->destroyBuffer(mesh.baseNormalBuffer);
    if (mesh.boneIndexBuffer.buffer) m_device->destroyBuffer(mesh.boneIndexBuffer);
    if (mesh.boneWeightBuffer.buffer) m_device->destroyBuffer(mesh.boneWeightBuffer);
    if (mesh.persistentBoneMatsBuffer.buffer) m_device->destroyBuffer(mesh.persistentBoneMatsBuffer);
    mesh = RasterMeshBuffer{};
}

void VulkanBackendAdapter::destroyAllRasterMeshes() {
    for (auto& [key, mesh] : m_rasterMeshes) {
        destroyRasterMesh(mesh);
    }
    m_rasterMeshes.clear();
    m_rasterInstances.clear();
}

// ─────────────────────────────────────────────────────────────────────
// Frustum Culling for Solid Viewport
// ─────────────────────────────────────────────────────────────────────

void VulkanBackendAdapter::extractFrustumPlanes(const Matrix4x4& vp) {
    const auto* vpBytes = reinterpret_cast<const unsigned char*>(&vp);
    uint64_t frustumHash = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof(Matrix4x4); ++i) {
        frustumHash ^= static_cast<uint64_t>(vpBytes[i]);
        frustumHash *= 1099511628211ull;
    }
    if (frustumHash != m_rasterFrustumHash) {
        m_rasterFrustumHash = frustumHash;
        ++m_rasterFrustumRevision;
    }

    // Extract 6 frustum planes from the view-projection matrix (row-major).
    // Each plane: dot(normal, point) + d >= 0 means inside.
    auto row = [&](int r, int c) -> float { return vp.m[r][c]; };

    // Left:   row3 + row0
    m_frustumPlanes[0].normal = Vec3(row(3,0)+row(0,0), row(3,1)+row(0,1), row(3,2)+row(0,2));
    m_frustumPlanes[0].d      =      row(3,3)+row(0,3);
    // Right:  row3 - row0
    m_frustumPlanes[1].normal = Vec3(row(3,0)-row(0,0), row(3,1)-row(0,1), row(3,2)-row(0,2));
    m_frustumPlanes[1].d      =      row(3,3)-row(0,3);
    // Bottom: row3 + row1
    m_frustumPlanes[2].normal = Vec3(row(3,0)+row(1,0), row(3,1)+row(1,1), row(3,2)+row(1,2));
    m_frustumPlanes[2].d      =      row(3,3)+row(1,3);
    // Top:    row3 - row1
    m_frustumPlanes[3].normal = Vec3(row(3,0)-row(1,0), row(3,1)-row(1,1), row(3,2)-row(1,2));
    m_frustumPlanes[3].d      =      row(3,3)-row(1,3);
    // Near:   row3 + row2
    m_frustumPlanes[4].normal = Vec3(row(3,0)+row(2,0), row(3,1)+row(2,1), row(3,2)+row(2,2));
    m_frustumPlanes[4].d      =      row(3,3)+row(2,3);
    // Far:    row3 - row2
    m_frustumPlanes[5].normal = Vec3(row(3,0)-row(2,0), row(3,1)-row(2,1), row(3,2)-row(2,2));
    m_frustumPlanes[5].d      =      row(3,3)-row(2,3);

    // Normalize planes
    for (auto& p : m_frustumPlanes) {
        float len = p.normal.length();
        if (len > 1e-6f) {
            p.normal = p.normal / len;
            p.d /= len;
        }
    }
}

bool VulkanBackendAdapter::isAABBInFrustum(const AABB& box) const {
    const Vec3 halfExtents = (box.max - box.min) * 0.5f;
    const float conservativeSlack = std::max(halfExtents.length() * 0.75f, 0.05f);
    for (int i = 0; i < 6; ++i) {
        if (i == 4) {
            // Skip CPU-side near-plane rejection for raster viewport culling.
            // When the camera gets very close to a surface, object AABBs often
            // straddle the near plane even though a visible portion should still
            // render. Let the GPU clipper handle the near plane instead of
            // dropping the whole mesh/chunk on the CPU side.
            continue;
        }
        const auto& pl = m_frustumPlanes[i];
        // P-vertex: the corner of AABB most in the direction of the plane normal
        Vec3 pVertex(
            pl.normal.x >= 0 ? box.max.x : box.min.x,
            pl.normal.y >= 0 ? box.max.y : box.min.y,
            pl.normal.z >= 0 ? box.max.z : box.min.z
        );
        if (Vec3::dot(pl.normal, pVertex) + pl.d < -conservativeSlack)
            return false;
    }
    return true;
}

bool VulkanBackendAdapter::isAABBFullyInsideFrustum(const AABB& box) const {
    if (!box.is_valid()) return false;
    const Vec3 halfExtents = (box.max - box.min) * 0.5f;
    const float conservativeSlack = std::max(halfExtents.length() * 0.35f, 0.02f);
    for (int i = 0; i < 6; ++i) {
        if (i == 4) {
            continue;
        }
        const auto& pl = m_frustumPlanes[i];
        Vec3 nVertex(
            pl.normal.x >= 0 ? box.min.x : box.max.x,
            pl.normal.y >= 0 ? box.min.y : box.max.y,
            pl.normal.z >= 0 ? box.min.z : box.max.z
        );
        if (Vec3::dot(pl.normal, nVertex) + pl.d < -conservativeSlack)
            return false;
    }
    return true;
}

bool VulkanBackendAdapter::isRasterChunkTooSmallToDraw(const AABB& box) const {
    if (!box.is_valid() || m_rasterCullFocalLengthPixels <= 0.0f ||
        m_rasterMinChunkScreenRadiusPixels <= 0.0f) {
        return false;
    }

    const Vec3 center = (box.min + box.max) * 0.5f;
    const Vec3 halfExtents = (box.max - box.min) * 0.5f;
    const float radius = halfExtents.length();
    if (radius <= 0.0f) {
        return false;
    }

    const float distance = (center - m_rasterCullCameraPosition).length();
    if (distance <= radius) {
        return false;
    }

    const float projectedRadiusPixels = (radius * m_rasterCullFocalLengthPixels) / distance;
    return projectedRadiusPixels < m_rasterMinChunkScreenRadiusPixels;
}

void VulkanBackendAdapter::updateRasterInstanceWorldBBox(RasterInstance& ri) const {
    // Transform local AABB 8 corners to world space, compute enclosing AABB
    const AABB& lb = ri.localBBox;
    if (!lb.is_valid()) {
        ri.worldBBox = lb;
        return;
    }
    const Vec3 corners[8] = {
        Vec3(lb.min.x, lb.min.y, lb.min.z),
        Vec3(lb.max.x, lb.min.y, lb.min.z),
        Vec3(lb.min.x, lb.max.y, lb.min.z),
        Vec3(lb.max.x, lb.max.y, lb.min.z),
        Vec3(lb.min.x, lb.min.y, lb.max.z),
        Vec3(lb.max.x, lb.min.y, lb.max.z),
        Vec3(lb.min.x, lb.max.y, lb.max.z),
        Vec3(lb.max.x, lb.max.y, lb.max.z),
    };
    Vec3 wMin(1e18f, 1e18f, 1e18f), wMax(-1e18f, -1e18f, -1e18f);
    for (const auto& c : corners) {
        Vec3 wc = ri.transform.transform_point(c);
        wMin.x = std::min(wMin.x, wc.x);
        wMin.y = std::min(wMin.y, wc.y);
        wMin.z = std::min(wMin.z, wc.z);
        wMax.x = std::max(wMax.x, wc.x);
        wMax.y = std::max(wMax.y, wc.y);
        wMax.z = std::max(wMax.z, wc.z);
    }
    ri.worldBBox = AABB(wMin, wMax);
}

void VulkanBackendAdapter::rebuildRasterMeshCullingChunks(RasterMeshBuffer& mesh) {
    mesh.cullingChunks.clear();
    if (mesh.instanceIndices.size() < 128) {
        return;
    }

    float baseCellSize = 0.0f;
    for (uint32_t instanceIndex : mesh.instanceIndices) {
        if (instanceIndex >= m_rasterInstances.size()) continue;
        const AABB& localBBox = m_rasterInstances[instanceIndex].localBBox;
        if (!localBBox.is_valid()) continue;
        const float extentX = std::max(0.0f, localBBox.max.x - localBBox.min.x);
        const float extentZ = std::max(0.0f, localBBox.max.z - localBBox.min.z);
        baseCellSize = std::max(baseCellSize, std::max(extentX, extentZ));
        if (baseCellSize > 0.0f) break;
    }

    const float cellSize = std::clamp(baseCellSize > 0.0f ? baseCellSize * 8.0f : 8.0f, 4.0f, 96.0f);

    struct ChunkKey {
        int32_t x;
        int32_t z;
        bool operator==(const ChunkKey& other) const {
            return x == other.x && z == other.z;
        }
    };
    struct ChunkKeyHash {
        size_t operator()(const ChunkKey& key) const {
            return (static_cast<size_t>(static_cast<uint32_t>(key.x)) << 32) ^
                   static_cast<size_t>(static_cast<uint32_t>(key.z));
        }
    };

    std::unordered_map<ChunkKey, size_t, ChunkKeyHash> chunkLookup;
    chunkLookup.reserve(mesh.instanceIndices.size() / 8 + 1);

    for (uint32_t instanceIndex : mesh.instanceIndices) {
        if (instanceIndex >= m_rasterInstances.size()) continue;
        const auto& ri = m_rasterInstances[instanceIndex];
        if (ri.mask == 0 || !ri.worldBBox.is_valid()) continue;

        const float centerX = (ri.worldBBox.min.x + ri.worldBBox.max.x) * 0.5f;
        const float centerZ = (ri.worldBBox.min.z + ri.worldBBox.max.z) * 0.5f;
        const ChunkKey key{
            static_cast<int32_t>(std::floor(centerX / cellSize)),
            static_cast<int32_t>(std::floor(centerZ / cellSize))
        };

        auto [it, inserted] = chunkLookup.emplace(key, mesh.cullingChunks.size());
        if (inserted) {
            mesh.cullingChunks.emplace_back();
            mesh.cullingChunks.back().worldBBox = ri.worldBBox;
        } else {
            auto& bbox = mesh.cullingChunks[it->second].worldBBox;
            bbox.min.x = std::min(bbox.min.x, ri.worldBBox.min.x);
            bbox.min.y = std::min(bbox.min.y, ri.worldBBox.min.y);
            bbox.min.z = std::min(bbox.min.z, ri.worldBBox.min.z);
            bbox.max.x = std::max(bbox.max.x, ri.worldBBox.max.x);
            bbox.max.y = std::max(bbox.max.y, ri.worldBBox.max.y);
            bbox.max.z = std::max(bbox.max.z, ri.worldBBox.max.z);
        }

        mesh.cullingChunks[it->second].instanceIndices.push_back(instanceIndex);
    }

    if (mesh.cullingChunks.size() <= 1) {
        mesh.cullingChunks.clear();
    }
}

void VulkanBackendAdapter::setRasterVisibleInstances(RasterMeshBuffer& mesh,
                                                     const std::vector<uint32_t>& visibleInstanceIndices) {
    struct RasterInstanceGPU {
        float model[16];
    };

    const bool visibleSetChanged =
        visibleInstanceIndices.size() != mesh.visibleInstanceIndicesCache.size() ||
        !std::equal(visibleInstanceIndices.begin(), visibleInstanceIndices.end(),
                    mesh.visibleInstanceIndicesCache.begin(), mesh.visibleInstanceIndicesCache.end());

    mesh.instanceCount = static_cast<uint32_t>(visibleInstanceIndices.size());
    if (mesh.instanceCount == 0) {
        mesh.visibleInstanceIndicesCache.clear();
        mesh.visibleInstancesDirty = false;
        mesh.lastVisibleFrustumRevision = m_rasterFrustumRevision;
        mesh.lastScatterTriangleBudget = m_rasterScatterTriangleBudget;
        return;
    }

    if (!mesh.visibleInstancesDirty && !visibleSetChanged && mesh.instanceBuffer.buffer) {
        return;
    }

    auto matrixToGL = [](const Matrix4x4& mat, float out[16]) {
        Matrix4x4 t = mat.transpose();
        int k = 0;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[k++] = t.m[r][c];
            }
        }
    };

    std::vector<RasterInstanceGPU> gpuInstances(mesh.instanceCount);
    const size_t kParallelMatrixThreshold = 4096;
    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    auto fillGpuRange = [this, &visibleInstanceIndices, &gpuInstances, &matrixToGL](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const uint32_t instanceIndex = visibleInstanceIndices[i];
            if (instanceIndex >= m_rasterInstances.size()) continue;
            matrixToGL(m_rasterInstances[instanceIndex].transform, gpuInstances[i].model);
        }
    };
    if (visibleInstanceIndices.size() < kParallelMatrixThreshold || numThreads < 2) {
        fillGpuRange(0, visibleInstanceIndices.size());
    } else {
        const size_t chunk = (visibleInstanceIndices.size() + numThreads - 1) / numThreads;
        std::vector<std::future<void>> futures;
        futures.reserve(numThreads);
        for (unsigned t = 0; t < numThreads; ++t) {
            const size_t s = t * chunk;
            const size_t e = std::min(s + chunk, visibleInstanceIndices.size());
            if (s >= e) break;
            futures.push_back(std::async(std::launch::async, fillGpuRange, s, e));
        }
        for (auto& f : futures) f.get();
    }

    mesh.visibleInstanceIndicesCache = visibleInstanceIndices;
    mesh.visibleInstancesDirty = false;
    mesh.lastVisibleFrustumRevision = m_rasterFrustumRevision;
    mesh.lastScatterTriangleBudget = m_rasterScatterTriangleBudget;

    const VkDeviceSize requiredSize = gpuInstances.size() * sizeof(RasterInstanceGPU);
    if (mesh.instanceBuffer.buffer && mesh.instanceBuffer.size >= requiredSize) {
        m_device->uploadBuffer(mesh.instanceBuffer, gpuInstances.data(), requiredSize, 0);
        return;
    }

    if (mesh.instanceBuffer.buffer) {
        m_device->destroyBuffer(mesh.instanceBuffer);
        mesh.instanceBuffer = VulkanRT::BufferHandle{};
    }

    const VkDeviceSize allocCapacity = std::max<size_t>(
        visibleInstanceIndices.size(), mesh.instanceIndices.size()) * sizeof(RasterInstanceGPU);
    VulkanRT::BufferCreateInfo ici{};
    ici.size = std::max(requiredSize, allocCapacity);
    ici.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
    ici.location = VulkanRT::MemoryLocation::GPU_ONLY;
    ici.initialData = nullptr;
    mesh.instanceBuffer = m_device->createBuffer(ici);
    if (mesh.instanceBuffer.buffer) {
        m_device->uploadBuffer(mesh.instanceBuffer, gpuInstances.data(), requiredSize, 0);
    }
}

void VulkanBackendAdapter::uploadVisibleRasterInstances(RasterMeshBuffer& mesh) {
    if (!m_device) return;
    if (mesh.isScatterProxy) return;

    const bool proxySplitStale =
        mesh.isScatterGroup &&
        !mesh.proxyMeshKey.empty() &&
        (mesh.lastVisibleFrustumRevision != m_rasterFrustumRevision ||
         mesh.lastScatterTriangleBudget != m_rasterScatterTriangleBudget);

    if (!mesh.visibleInstancesDirty &&
        !proxySplitStale &&
        (!kRasterFrustumCullingEnabled || mesh.lastVisibleFrustumRevision == m_rasterFrustumRevision) &&
        (mesh.instanceBuffer.buffer || mesh.visibleInstanceIndicesCache.empty())) {
        mesh.instanceCount = static_cast<uint32_t>(mesh.visibleInstanceIndicesCache.size());
        if (!kRasterFrustumCullingEnabled) {
            mesh.lastVisibleFrustumRevision = m_rasterFrustumRevision;
        }
        return;
    }

    std::vector<uint32_t> visibleInstanceIndices;
    visibleInstanceIndices.reserve(mesh.instanceIndices.size());

    auto appendVisibleInstance = [&](uint32_t instanceIndex, bool skipFrustumTest) {
        if (instanceIndex >= m_rasterInstances.size()) return;
        const auto& ri = m_rasterInstances[instanceIndex];
        if (ri.mask == 0) return;
        if (kRasterFrustumCullingEnabled &&
            !skipFrustumTest &&
            !mesh.hasSkinning &&
            ri.worldBBox.is_valid() &&
            !isAABBInFrustum(ri.worldBBox)) {
            return;
        }
        visibleInstanceIndices.push_back(instanceIndex);
    };

    if (kRasterFrustumCullingEnabled && !mesh.hasSkinning && !mesh.cullingChunks.empty()) {
        for (const auto& chunk : mesh.cullingChunks) {
            if (chunk.worldBBox.is_valid() && !isAABBInFrustum(chunk.worldBBox)) {
                continue;
            }
            if (chunk.instanceIndices.size() >= 16 && isRasterChunkTooSmallToDraw(chunk.worldBBox)) {
                continue;
            }
            const bool chunkFullyInside = chunk.worldBBox.is_valid() && isAABBFullyInsideFrustum(chunk.worldBBox);
            for (uint32_t instanceIndex : chunk.instanceIndices) {
                appendVisibleInstance(instanceIndex, chunkFullyInside);
            }
        }
    } else {
        for (uint32_t instanceIndex : mesh.instanceIndices) {
            appendVisibleInstance(instanceIndex, false);
        }
    }

    std::vector<uint32_t> proxyInstanceIndices;
    if (mesh.isScatterGroup) {
        const uint64_t trianglesPerInstance = (mesh.indexBuffer.buffer && mesh.indexCount > 0)
            ? (static_cast<uint64_t>(mesh.indexCount) / 3ull)
            : (static_cast<uint64_t>(mesh.vertexCount) / 3ull);
        const uint64_t scatterTriangleBudget = std::max<uint64_t>(1ull, m_rasterScatterTriangleBudget);
        if (trianglesPerInstance > 0) {
            const uint64_t visibleTriangles = trianglesPerInstance * static_cast<uint64_t>(visibleInstanceIndices.size());
            if (visibleTriangles > scatterTriangleBudget) {
                const size_t cappedVisibleCount = static_cast<size_t>(std::max<uint64_t>(
                    1ull, scatterTriangleBudget / trianglesPerInstance));
                if (cappedVisibleCount < visibleInstanceIndices.size()) {
                    struct DistanceEntry {
                        float distanceSq;
                        uint32_t instanceIndex;
                    };
                    std::vector<DistanceEntry> nearestInstances;
                    nearestInstances.reserve(visibleInstanceIndices.size());
                    for (uint32_t instanceIndex : visibleInstanceIndices) {
                        if (instanceIndex >= m_rasterInstances.size()) continue;
                        const auto& ri = m_rasterInstances[instanceIndex];
                        Vec3 center = ri.worldBBox.is_valid()
                            ? (ri.worldBBox.min + ri.worldBBox.max) * 0.5f
                            : ri.transform.transform_point(Vec3(0.0f, 0.0f, 0.0f));
                        const Vec3 delta = center - m_rasterCullCameraPosition;
                        nearestInstances.push_back({ Vec3::dot(delta, delta), instanceIndex });
                    }

                    auto distanceCmp = [](const DistanceEntry& a, const DistanceEntry& b) {
                        if (a.distanceSq != b.distanceSq) return a.distanceSq < b.distanceSq;
                        return a.instanceIndex < b.instanceIndex;
                    };
                    std::nth_element(nearestInstances.begin(),
                                     nearestInstances.begin() + cappedVisibleCount,
                                     nearestInstances.end(),
                                     distanceCmp);
                    std::sort(nearestInstances.begin(), nearestInstances.begin() + cappedVisibleCount, distanceCmp);

                    proxyInstanceIndices.reserve(nearestInstances.size() - cappedVisibleCount);
                    for (size_t i = cappedVisibleCount; i < nearestInstances.size(); ++i) {
                        proxyInstanceIndices.push_back(nearestInstances[i].instanceIndex);
                    }

                    visibleInstanceIndices.clear();
                    visibleInstanceIndices.reserve(cappedVisibleCount);
                    for (size_t i = 0; i < cappedVisibleCount; ++i) {
                        visibleInstanceIndices.push_back(nearestInstances[i].instanceIndex);
                    }
                }
            }
        }
    }

    if (!mesh.proxyMeshKey.empty()) {
        auto proxyIt = m_rasterMeshes.find(mesh.proxyMeshKey);
        if (proxyIt != m_rasterMeshes.end()) {
            proxyIt->second.visibleInstancesDirty = mesh.visibleInstancesDirty;
            setRasterVisibleInstances(proxyIt->second, proxyInstanceIndices);
        }
    }

    setRasterVisibleInstances(mesh, visibleInstanceIndices);
}

void VulkanBackendAdapter::uploadRasterInstanceBuffer(RasterMeshBuffer& mesh) {
    if (!m_device) return;

    mesh.visibleInstancesDirty = true;
    mesh.lastVisibleFrustumRevision = 0;
    if (kRasterFrustumCullingEnabled) {
        rebuildRasterMeshCullingChunks(mesh);
    } else {
        mesh.cullingChunks.clear();
    }

    struct RasterInstanceGPU {
        float model[16];
    };

    if (mesh.instanceBuffer.buffer) {
        m_device->destroyBuffer(mesh.instanceBuffer);
        mesh.instanceBuffer = VulkanRT::BufferHandle{};
    }

    bool allInstancesVisible = true;
    for (uint32_t instanceIndex : mesh.instanceIndices) {
        if (instanceIndex >= m_rasterInstances.size() || m_rasterInstances[instanceIndex].mask == 0) {
            allInstancesVisible = false;
            break;
        }
    }

    std::vector<uint32_t> visibleInstanceIndices;
    if (allInstancesVisible) {
        visibleInstanceIndices = mesh.instanceIndices;
    } else {
        visibleInstanceIndices.reserve(mesh.instanceIndices.size());
        for (uint32_t instanceIndex : mesh.instanceIndices) {
            if (instanceIndex >= m_rasterInstances.size()) continue;
            if (m_rasterInstances[instanceIndex].mask == 0) continue;
            visibleInstanceIndices.push_back(instanceIndex);
        }
    }

    mesh.instanceCount = static_cast<uint32_t>(visibleInstanceIndices.size());
    if (mesh.instanceCount == 0) {
        mesh.visibleInstanceIndicesCache.clear();
        mesh.visibleInstancesDirty = false;
        return;
    }

    auto matrixToGL = [](const Matrix4x4& mat, float out[16]) {
        Matrix4x4 t = mat.transpose();
        int k = 0;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[k++] = t.m[r][c];
            }
        }
    };

    std::vector<RasterInstanceGPU> gpuInstances(mesh.instanceCount);
    const size_t kParallelMatrixThreshold = 4096;
    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    auto fillGpuRange = [this, &visibleInstanceIndices, &gpuInstances, &matrixToGL](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const uint32_t instanceIndex = visibleInstanceIndices[i];
            if (instanceIndex >= m_rasterInstances.size()) continue;
            matrixToGL(m_rasterInstances[instanceIndex].transform, gpuInstances[i].model);
        }
    };
    if (visibleInstanceIndices.size() < kParallelMatrixThreshold || numThreads < 2) {
        fillGpuRange(0, visibleInstanceIndices.size());
    } else {
        const size_t chunk = (visibleInstanceIndices.size() + numThreads - 1) / numThreads;
        std::vector<std::future<void>> futures;
        futures.reserve(numThreads);
        for (unsigned t = 0; t < numThreads; ++t) {
            const size_t s = t * chunk;
            const size_t e = std::min(s + chunk, visibleInstanceIndices.size());
            if (s >= e) break;
            futures.push_back(std::async(std::launch::async, fillGpuRange, s, e));
        }
        for (auto& f : futures) f.get();
    }

    VulkanRT::BufferCreateInfo ici{};
    ici.size = gpuInstances.size() * sizeof(RasterInstanceGPU);
    ici.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
    ici.location = VulkanRT::MemoryLocation::GPU_ONLY;
    ici.initialData = nullptr;
    mesh.instanceBuffer = m_device->createBuffer(ici);
    if (mesh.instanceBuffer.buffer) {
        m_device->uploadBuffer(mesh.instanceBuffer,
                               gpuInstances.data(),
                               gpuInstances.size() * sizeof(RasterInstanceGPU),
                               0);
    }

    mesh.visibleInstanceIndicesCache = std::move(visibleInstanceIndices);
    mesh.visibleInstancesDirty = false;
    mesh.lastVisibleFrustumRevision = m_rasterFrustumRevision;
    mesh.lastScatterTriangleBudget = m_rasterScatterTriangleBudget;
}

void VulkanBackendAdapter::buildRasterGeometryImpl(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!m_device || !m_device->isInitialized()) return;

    // Skip rebuild if raster cache is still valid for the current scene generation.
    {
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        const uint64_t curGen = g_scene_geometry_generation.load(std::memory_order_acquire);
        if (!m_rasterMeshes.empty() && m_rasterBuiltGeometryGeneration == curGen) {
            m_rasterGeometryDirty = false;
            return;
        }
    }

    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    // Destroy old raster buffers
    destroyAllRasterMeshes();

    auto hasInstancePrefix = [](const std::string& nodeName) -> bool {
        return nodeName.rfind("_inst_gid", 0) == 0;
    };

    size_t baseObjectCount = objects.size();
    while (baseObjectCount > 0) {
        const auto& obj = objects[baseObjectCount - 1];
        auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
        if (!inst || !hasInstancePrefix(inst->node_name)) {
            break;
        }
        --baseObjectCount;
    }

    auto ensureRasterMeshForTriangles = [&](const std::string& meshKey,
                                            const std::vector<std::shared_ptr<Triangle>>& triangles) {
        if (triangles.empty()) return;
        if (m_rasterMeshes.find(meshKey) != m_rasterMeshes.end()) return;

        // Filter nulls into a compact raw-pointer list so the parallel extraction can
        // index output slots directly (9 floats pos + 9 normals + 6 uvs + 3 matIds per
        // triangle). Mirrors VulkanViewportBackend::ensureRasterMeshForTriangles.
        std::vector<const Triangle*> valid;
        valid.reserve(triangles.size());
        for (const auto& t : triangles) {
            if (t) valid.push_back(t.get());
        }
        if (valid.empty()) return;

        const size_t validCount = valid.size();
        std::vector<float> positions(validCount * 9);
        std::vector<float> normals(validCount * 9);
        std::vector<float> uvs(validCount * 6);
        std::vector<uint32_t> matIds(validCount * 3);

        struct LocalBBox { Vec3 bMin; Vec3 bMax; };

        auto extractRange = [&valid, &positions, &normals, &uvs, &matIds]
                            (size_t start, size_t end) -> LocalBBox {
            Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);
            for (size_t i = start; i < end; ++i) {
                const Triangle* t = valid[i];
                auto [uv0, uv1, uv2] = t->getUVCoordinates();
                const uint32_t mid = static_cast<uint32_t>(t->getMaterialID());
                const size_t posBase = i * 9;
                const size_t uvBase  = i * 6;
                const size_t matBase = i * 3;
                for (int v = 0; v < 3; ++v) {
                    Vec3 p = t->getOriginalVertexPosition(v);
                    Vec3 n = t->getOriginalVertexNormal(v);
                    positions[posBase + v * 3 + 0] = p.x;
                    positions[posBase + v * 3 + 1] = p.y;
                    positions[posBase + v * 3 + 2] = p.z;
                    normals[posBase + v * 3 + 0] = n.x;
                    normals[posBase + v * 3 + 1] = n.y;
                    normals[posBase + v * 3 + 2] = n.z;
                    matIds[matBase + v] = mid;
                    bMin.x = std::min(bMin.x, p.x); bMin.y = std::min(bMin.y, p.y); bMin.z = std::min(bMin.z, p.z);
                    bMax.x = std::max(bMax.x, p.x); bMax.y = std::max(bMax.y, p.y); bMax.z = std::max(bMax.z, p.z);
                }
                uvs[uvBase + 0] = uv0.x; uvs[uvBase + 1] = uv0.y;
                uvs[uvBase + 2] = uv1.x; uvs[uvBase + 3] = uv1.y;
                uvs[uvBase + 4] = uv2.x; uvs[uvBase + 5] = uv2.y;
            }
            return { bMin, bMax };
        };

        Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);
        constexpr size_t kExtractParallelThreshold = 4096;
        unsigned extract_threads = std::thread::hardware_concurrency();
        if (extract_threads == 0) extract_threads = 4;

        if (validCount < kExtractParallelThreshold || extract_threads < 2) {
            LocalBBox lbb = extractRange(0, validCount);
            bMin = lbb.bMin;
            bMax = lbb.bMax;
        } else {
            const size_t chunk = (validCount + extract_threads - 1) / extract_threads;
            std::vector<std::future<LocalBBox>> futures;
            futures.reserve(extract_threads);
            for (unsigned t = 0; t < extract_threads; ++t) {
                const size_t s = t * chunk;
                const size_t e = std::min(s + chunk, validCount);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, extractRange, s, e));
            }
            for (auto& f : futures) {
                LocalBBox lbb = f.get();
                bMin.x = std::min(bMin.x, lbb.bMin.x); bMin.y = std::min(bMin.y, lbb.bMin.y); bMin.z = std::min(bMin.z, lbb.bMin.z);
                bMax.x = std::max(bMax.x, lbb.bMax.x); bMax.y = std::max(bMax.y, lbb.bMax.y); bMax.z = std::max(bMax.z, lbb.bMax.z);
            }
        }

        // Cache the local bounding box for this mesh key
        m_rasterMeshBBoxes[meshKey] = AABB(bMin, bMax);

        RasterMeshBuffer rmb;
        rmb.vertexCount = (uint32_t)(positions.size() / 3);
        VulkanRT::BufferCreateInfo vci{};
        vci.size = positions.size() * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        vci.initialData = nullptr;
        rmb.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = normals.size() * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        nci.initialData = nullptr;
        rmb.normalBuffer = m_device->createBuffer(nci);

        // UV buffer for MaterialPreview
        if (!uvs.empty()) {
            VulkanRT::BufferCreateInfo uci{};
            uci.size = uvs.size() * sizeof(float);
            uci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            uci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            rmb.uvBuffer = m_device->createBuffer(uci);
            if (rmb.uvBuffer.buffer) {
                m_device->uploadBuffer(rmb.uvBuffer, uvs.data(), uvs.size() * sizeof(float), 0);
            }
        }

        // MaterialID buffer for MaterialPreview
        if (!matIds.empty()) {
            VulkanRT::BufferCreateInfo mci{};
            mci.size = matIds.size() * sizeof(uint32_t);
            mci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            mci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            rmb.matIdBuffer = m_device->createBuffer(mci);
            if (rmb.matIdBuffer.buffer) {
                m_device->uploadBuffer(rmb.matIdBuffer, matIds.data(), matIds.size() * sizeof(uint32_t), 0);
            }
            rmb.cpuMatIds = std::move(matIds);
        }

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, positions.data(), positions.size() * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, normals.data(), normals.size() * sizeof(float), 0);
        }

        m_rasterMeshes[meshKey] = std::move(rmb);
    };

    struct RasterTriGroup {
        std::string meshKey;
        std::string nodeName;
        std::vector<float> positions; // interleaved x,y,z
        std::vector<float> normals;
        std::vector<float> uvs;       // interleaved u,v per vertex
        std::vector<uint32_t> matIds; // per-vertex material ID
        std::shared_ptr<TriangleMesh> mesh;
        Matrix4x4 transform;
        uint8_t mask = 0xFF;
    };

    std::vector<RasterTriGroup> groups;
    std::unordered_map<std::string, size_t> groupByKey;

    // Recursive traversal — same logic as updateGeometry but only collects vertex data
    std::function<void(const std::shared_ptr<Hittable>&)> processObj;
    processObj = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;

        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->visible || !inst->source_triangles || inst->source_triangles->empty()) return;
            if (hasInstancePrefix(inst->node_name)) return;

            const auto srcPtr = reinterpret_cast<uintptr_t>(inst->source_triangles.get());
            const std::string instanceNodeName = inst->node_name.empty()
                ? ("[RasterInst-" + std::to_string(m_rasterInstances.size()) + "]")
                : inst->node_name;
            std::unordered_map<std::string, std::vector<std::shared_ptr<Triangle>>> trianglesByNode;
            trianglesByNode.reserve(inst->source_triangles->size());
            for (const auto& tri : *inst->source_triangles) {
                if (!tri) continue;
                const std::string triNodeName = tri->getNodeName().empty() ? instanceNodeName : tri->getNodeName();
                trianglesByNode[triNodeName].push_back(tri);
            }

            for (const auto& [triNodeName, groupedTriangles] : trianglesByNode) {
                if (groupedTriangles.empty()) continue;
                // Keep raster meshes instance-local and node-local so startup preview uses
                // the same object/material grouping as the editor caches and selection code.
                std::string meshKey = "[Raster]-" + triNodeName +
                                      "-src-" + std::to_string(srcPtr) +
                                      "-tris-" + std::to_string(groupedTriangles.size());

                ensureRasterMeshForTriangles(meshKey, groupedTriangles);

                RasterInstance ri;
                ri.meshKey = meshKey;
                ri.nodeName = triNodeName;
                ri.transform = inst->transform;
                ri.mask = 0xFF;
                m_rasterInstances.push_back(ri);
            }

        } else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) processObj(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            processObj(bvh->left);
            processObj(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->visible) return;

            Transform* triTransformHandle = tri->getTransformPtr();
            const bool hasSharedTransform = (triTransformHandle != nullptr);
            std::string nodeName = tri->getNodeName();
            if (nodeName.empty()) nodeName = "[Solo-" + std::to_string(groups.size()) + "]";
            const uintptr_t transformKey = triTransformHandle
                ? reinterpret_cast<uintptr_t>(triTransformHandle)
                : reinterpret_cast<uintptr_t>(tri.get());
            const std::string groupKey = nodeName + "#th=" + std::to_string(transformKey);

            auto found = groupByKey.find(groupKey);
            if (found == groupByKey.end()) {
                RasterTriGroup g;
                g.meshKey = "[Raster-Solo]-" + nodeName;
                g.nodeName = nodeName;
                g.transform = hasSharedTransform ? tri->getTransformMatrix() : Matrix4x4::identity();
                groups.push_back(std::move(g));
                found = groupByKey.emplace(groupKey, groups.size() - 1).first;
            }

            auto& grp = groups[found->second];
            auto [uv0, uv1, uv2] = tri->getUVCoordinates();
            uint32_t mid = static_cast<uint32_t>(tri->getMaterialID());
            for (int v = 0; v < 3; ++v) {
                Vec3 p = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                Vec3 n = hasSharedTransform ? tri->getOriginalVertexNormal(v) : tri->getOriginalVertexNormal(v);
                grp.positions.push_back(p.x); grp.positions.push_back(p.y); grp.positions.push_back(p.z);
                grp.normals.push_back(n.x); grp.normals.push_back(n.y); grp.normals.push_back(n.z);
                grp.matIds.push_back(mid);
            }
            grp.uvs.push_back(uv0.x); grp.uvs.push_back(uv0.y);
            grp.uvs.push_back(uv1.x); grp.uvs.push_back(uv1.y);
            grp.uvs.push_back(uv2.x); grp.uvs.push_back(uv2.y);
        } else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (!mesh->visible || !mesh->geometry || mesh->geometry->indices.empty()) return;
            const DNA::GeometryDetail& geom = *mesh->geometry;
            const Vec3* positions = geom.get_positions_orig();
            const Vec3* normals = geom.get_normals_orig();
            if (!positions) positions = geom.get_positions();
            if (!normals) normals = geom.get_normals();
            if (!positions) return;
            const Vec2* uvs = geom.get_uvs();
            const uint16_t* materialIds = geom.get_material_ids();

            const std::string nodeName = mesh->nodeName.empty()
                ? ("[Solo-" + std::to_string(groups.size()) + "]")
                : mesh->nodeName;
            const std::string groupKey = nodeName + "#mesh=" +
                std::to_string(reinterpret_cast<uintptr_t>(mesh.get()));
            RasterTriGroup group;
            group.meshKey = "[Raster-Solo]-" + groupKey;
            group.nodeName = nodeName;
            group.transform = mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
            group.mesh = mesh;
            group.positions.reserve(geom.indices.size() * 3);
            group.normals.reserve(geom.indices.size() * 3);
            group.uvs.reserve(geom.indices.size() * 2);
            group.matIds.reserve(geom.indices.size());
            for (uint32_t vertexIndex : geom.indices) {
                const Vec3& position = positions[vertexIndex];
                const Vec3 normal = normals ? normals[vertexIndex] : Vec3(0.0f, 1.0f, 0.0f);
                const Vec2 uv = uvs ? uvs[vertexIndex] : Vec2(0.0f, 0.0f);
                group.positions.insert(group.positions.end(), {position.x, position.y, position.z});
                group.normals.insert(group.normals.end(), {normal.x, normal.y, normal.z});
                group.uvs.insert(group.uvs.end(), {uv.x, uv.y});
                group.matIds.push_back(materialIds ? static_cast<uint32_t>(materialIds[vertexIndex]) : 0u);
            }
            groups.push_back(std::move(group));
        }
        // VDB/Gas volumes are not rasterized in solid mode
    };

    for (size_t i = 0; i < baseObjectCount; ++i) {
        processObj(objects[i]);
    }

    // Upload solo triangle groups
    for (auto& grp : groups) {
        if (grp.positions.empty()) continue;

        // Compute local-space AABB from collected positions
        Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);
        for (size_t pi = 0; pi + 2 < grp.positions.size(); pi += 3) {
            float px = grp.positions[pi], py = grp.positions[pi+1], pz = grp.positions[pi+2];
            bMin.x = std::min(bMin.x, px); bMin.y = std::min(bMin.y, py); bMin.z = std::min(bMin.z, pz);
            bMax.x = std::max(bMax.x, px); bMax.y = std::max(bMax.y, py); bMax.z = std::max(bMax.z, pz);
        }
        m_rasterMeshBBoxes[grp.meshKey] = AABB(bMin, bMax);

        RasterMeshBuffer rmb;
        rmb.vertexCount = (uint32_t)(grp.positions.size() / 3);

        VulkanRT::BufferCreateInfo vci{};
        vci.size = grp.positions.size() * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        vci.initialData = nullptr;
        rmb.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = grp.normals.size() * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        nci.initialData = nullptr;
        rmb.normalBuffer = m_device->createBuffer(nci);

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, grp.positions.data(), grp.positions.size() * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, grp.normals.data(), grp.normals.size() * sizeof(float), 0);
        }

        // UV buffer for MaterialPreview
        if (!grp.uvs.empty()) {
            VulkanRT::BufferCreateInfo uci{};
            uci.size = grp.uvs.size() * sizeof(float);
            uci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            uci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            rmb.uvBuffer = m_device->createBuffer(uci);
            if (rmb.uvBuffer.buffer) {
                m_device->uploadBuffer(rmb.uvBuffer, grp.uvs.data(), grp.uvs.size() * sizeof(float), 0);
            }
        }
        // MaterialID buffer for MaterialPreview
        if (!grp.matIds.empty()) {
            VulkanRT::BufferCreateInfo mci{};
            mci.size = grp.matIds.size() * sizeof(uint32_t);
            mci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            mci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            rmb.matIdBuffer = m_device->createBuffer(mci);
            if (rmb.matIdBuffer.buffer) {
                m_device->uploadBuffer(rmb.matIdBuffer, grp.matIds.data(), grp.matIds.size() * sizeof(uint32_t), 0);
            }
            rmb.cpuMatIds = grp.matIds;
        }

        m_rasterMeshes[grp.meshKey] = rmb;

        RasterInstance ri;
        ri.meshKey = grp.meshKey;
        ri.nodeName = grp.nodeName;
        ri.transform = grp.transform;
        ri.mask = 0xFF;
        m_rasterInstances.push_back(ri);
    }

    // Append foliage/scatter instances directly from grouped instance data so
    // solid viewport does not have to traverse millions of expanded scene objects.
    //
    // Parallelized (mirrors VulkanViewportBackend::buildRasterGeometry):
    //   - Serial pre-pass resolves meshKey per (group, srcIdx) once and calls
    //     ensureRasterMeshForTriangles (mutates m_rasterMeshes / m_rasterMeshBBoxes).
    //   - Parallel per-group inner loop composes inst.toMatrix() + RasterInstance.
    //   - Parallel bbox assignment.
    const auto& instanceGroups = InstanceManager::getInstance().getGroups();

    struct GroupSrcMeta {
        std::vector<std::string> meshKeyBySrc;  // indexed by srcIdx; empty = invalid source
    };
    std::vector<GroupSrcMeta> groupMeta(instanceGroups.size());

    size_t totalValidScatterInstances = 0;
    for (size_t gi = 0; gi < instanceGroups.size(); ++gi) {
        const auto& group = instanceGroups[gi];
        if (group.instances.empty() || group.sources.empty()) continue;
        auto& meta = groupMeta[gi];
        meta.meshKeyBySrc.resize(group.sources.size());

        for (size_t si = 0; si < group.sources.size(); ++si) {
            const auto& source = group.sources[si];
            const auto* triSource = source.centered_triangles_ptr ? source.centered_triangles_ptr.get() : nullptr;
            if ((!triSource || triSource->empty()) && source.triangles.empty()) continue;

            std::string meshKey;
            if (triSource) {
                const auto srcPtr = reinterpret_cast<uintptr_t>(triSource);
                meshKey = "[Raster-Group]-" + std::to_string(group.id) + "-" + std::to_string(si) +
                          "-" + std::to_string(srcPtr) + "-" + std::to_string(triSource->size());
                ensureRasterMeshForTriangles(meshKey, *triSource);
            } else {
                const auto srcPtr = reinterpret_cast<uintptr_t>(&source.triangles);
                meshKey = "[Raster-Group]-" + std::to_string(group.id) + "-" + std::to_string(si) +
                          "-" + std::to_string(srcPtr) + "-" + std::to_string(source.triangles.size());
                ensureRasterMeshForTriangles(meshKey, source.triangles);
            }
            meta.meshKeyBySrc[si] = std::move(meshKey);
        }

        for (const auto& inst : group.instances) {
            int srcIdx = inst.source_index;
            if (srcIdx < 0 || srcIdx >= static_cast<int>(group.sources.size())) srcIdx = 0;
            if (srcIdx < static_cast<int>(meta.meshKeyBySrc.size()) &&
                !meta.meshKeyBySrc[srcIdx].empty()) {
                ++totalValidScatterInstances;
            }
        }
    }

    m_rasterInstances.reserve(m_rasterInstances.size() + totalValidScatterInstances);

    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    const size_t kParallelThreshold = 1024;

    for (size_t gi = 0; gi < instanceGroups.size(); ++gi) {
        const auto& group = instanceGroups[gi];
        if (group.instances.empty() || group.sources.empty()) continue;
        const auto& meshKeyBySrc = groupMeta[gi].meshKeyBySrc;
        if (meshKeyBySrc.empty()) continue;

        const size_t count = group.instances.size();
        std::vector<RasterInstance> localInstances(count);

        auto fillRange = [&group, &meshKeyBySrc, &localInstances](size_t start, size_t end) {
            const std::string nodePrefix = "_inst_gid" + std::to_string(group.id) + "_";
            for (size_t i = start; i < end; ++i) {
                const auto& inst = group.instances[i];
                int srcIdx = inst.source_index;
                if (srcIdx < 0 || srcIdx >= static_cast<int>(group.sources.size())) srcIdx = 0;
                if (srcIdx >= static_cast<int>(meshKeyBySrc.size()) ||
                    meshKeyBySrc[srcIdx].empty()) {
                    continue;
                }
                auto& ri = localInstances[i];
                ri.meshKey = meshKeyBySrc[srcIdx];
                ri.nodeName = nodePrefix + std::to_string(i);
                ri.transform = inst.toMatrix();
                ri.mask = 0xFF;
                ri.scatterGroupId = group.id;
                ri.scatterInstanceIndex = static_cast<uint32_t>(i);
            }
        };

        if (count < kParallelThreshold || num_threads < 2) {
            fillRange(0, count);
        } else {
            const size_t chunk = (count + num_threads - 1) / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);
            for (unsigned t = 0; t < num_threads; ++t) {
                const size_t s = t * chunk;
                const size_t e = std::min(s + chunk, count);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, fillRange, s, e));
            }
            for (auto& f : futures) f.get();
        }

        for (auto& ri : localInstances) {
            if (ri.meshKey.empty()) continue;
            m_rasterInstances.push_back(std::move(ri));
        }
    }

    // Assign localBBox from cached mesh AABB and compute worldBBox per instance.
    // Parallel: m_rasterMeshBBoxes is read-only here; writes target disjoint instances.
    {
        const size_t total = m_rasterInstances.size();
        auto bboxRange = [this](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                auto& ri = m_rasterInstances[i];
                auto bboxIt = m_rasterMeshBBoxes.find(ri.meshKey);
                if (bboxIt != m_rasterMeshBBoxes.end()) {
                    ri.localBBox = bboxIt->second;
                    updateRasterInstanceWorldBBox(ri);
                }
            }
        };

        if (total < kParallelThreshold || num_threads < 2) {
            bboxRange(0, total);
        } else {
            const size_t chunk = (total + num_threads - 1) / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);
            for (unsigned t = 0; t < num_threads; ++t) {
                const size_t s = t * chunk;
                const size_t e = std::min(s + chunk, total);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, bboxRange, s, e));
            }
            for (auto& f : futures) f.get();
        }
    }

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_rasterInstances.size()); ++i) {
        auto meshIt = m_rasterMeshes.find(m_rasterInstances[i].meshKey);
        if (meshIt == m_rasterMeshes.end()) continue;
        meshIt->second.instanceIndices.push_back(i);
    }
    for (auto& [key, mesh] : m_rasterMeshes) {
        uploadRasterInstanceBuffer(mesh);
    }

    m_rasterGeometryDirty = false;
    m_interactiveViewport.dirty = true;
    m_hasPresentedRenderedFrame = false;
    m_lastCameraHash = 0;

    // Stamp current scene generation so we can skip redundant rebuilds later.
    {
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        m_rasterBuiltGeometryGeneration = g_scene_geometry_generation.load(std::memory_order_acquire);
    }

    SCENE_LOG_INFO("[Vulkan] Raster geometry built: " + std::to_string(m_rasterMeshes.size()) +
                   " meshes, " + std::to_string(m_rasterInstances.size()) + " instances. " +
                   "(base objects scanned: " + std::to_string(baseObjectCount) + "/" + std::to_string(objects.size()) + ")");
}

void VulkanBackendAdapter::syncRasterInstanceTransformsImpl(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (m_rasterInstances.empty()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Build nodeName → transform lookup from current scene objects
    std::unordered_map<std::string, Matrix4x4> transformMap;
    transformMap.reserve(objects.size());

    std::function<void(const std::shared_ptr<Hittable>&)> collectTransforms;
    collectTransforms = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->node_name.empty()) {
                transformMap[inst->node_name] = inst->transform;
            }
        } else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) collectTransforms(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            collectTransforms(bvh->left);
            collectTransforms(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            Transform* th = tri->getTransformPtr();
            std::string name = tri->getNodeName();
            if (!name.empty() && th) {
                transformMap[name] = tri->getTransformMatrix();
            }
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            // Flat (direct SoA) mesh: drives its world transform through its own handle. Without this
            // the raster (Solid/Matcap) viewport never refreshed a keyframed/physics-driven flat
            // mesh per frame — it froze during playback, mirroring the RT-path gap.
            if (!tm->nodeName.empty() && tm->transform) {
                transformMap[tm->nodeName] = tm->transform->getFinal();
            }
        }
    };

    auto hasInstancePrefix = [](const std::string& nodeName) -> bool {
        return nodeName.rfind("_inst_gid", 0) == 0;
    };
    size_t baseObjectCount = objects.size();
    while (baseObjectCount > 0) {
        const auto& obj = objects[baseObjectCount - 1];
        auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
        if (!inst || !hasInstancePrefix(inst->node_name)) {
            break;
        }
        --baseObjectCount;
    }

    for (size_t i = 0; i < baseObjectCount; ++i) {
        collectTransforms(objects[i]);
    }

    const auto& instanceGroups = InstanceManager::getInstance().getGroups();
    std::unordered_map<int, const InstanceGroup*> scatterGroupsById;
    scatterGroupsById.reserve(instanceGroups.size());
    for (const auto& group : instanceGroups) {
        if (!group.instances.empty()) {
            scatterGroupsById.emplace(group.id, &group);
        }
    }

    // Apply to raster instances. Scatter instances can bypass nodeName hash
    // lookup and read transforms directly from InstanceManager.
    bool changed = false;
    std::unordered_set<std::string> dirtyMeshKeys;
    const size_t kParallelThreshold = 2048;
    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    auto syncRange = [this, &transformMap, &scatterGroupsById]
                     (size_t start, size_t end) {
        std::unordered_set<std::string> localDirty;
        for (size_t i = start; i < end; ++i) {
            auto& ri = m_rasterInstances[i];
            Matrix4x4 newTransform;
            bool hasTransform = false;

            if (ri.scatterGroupId >= 0 && ri.scatterInstanceIndex != UINT32_MAX) {
                auto groupIt = scatterGroupsById.find(ri.scatterGroupId);
                if (groupIt != scatterGroupsById.end()) {
                    const auto* group = groupIt->second;
                    if (ri.scatterInstanceIndex < group->instances.size()) {
                        newTransform = group->instances[ri.scatterInstanceIndex].toMatrix();
                        hasTransform = true;
                    }
                }
            } else {
                auto it = transformMap.find(ri.nodeName);
                if (it != transformMap.end()) {
                    newTransform = it->second;
                    hasTransform = true;
                }
            }

            if (hasTransform && !(ri.transform == newTransform)) {
                ri.transform = newTransform;
                updateRasterInstanceWorldBBox(ri);
                localDirty.insert(ri.meshKey);
            }
        }
        return localDirty;
    };

    if (m_rasterInstances.size() < kParallelThreshold || numThreads < 2) {
        dirtyMeshKeys = syncRange(0, m_rasterInstances.size());
    } else {
        const size_t chunk = (m_rasterInstances.size() + numThreads - 1) / numThreads;
        std::vector<std::future<std::unordered_set<std::string>>> futures;
        futures.reserve(numThreads);
        for (unsigned t = 0; t < numThreads; ++t) {
            const size_t s = t * chunk;
            const size_t e = std::min(s + chunk, m_rasterInstances.size());
            if (s >= e) break;
            futures.push_back(std::async(std::launch::async, syncRange, s, e));
        }
        for (auto& f : futures) {
            auto localDirty = f.get();
            dirtyMeshKeys.insert(localDirty.begin(), localDirty.end());
        }
    }
    changed = !dirtyMeshKeys.empty();

    if (changed) {
        for (const auto& meshKey : dirtyMeshKeys) {
            auto meshIt = m_rasterMeshes.find(meshKey);
            if (meshIt != m_rasterMeshes.end()) {
                uploadRasterInstanceBuffer(meshIt->second);
            }
        }
        m_interactiveViewport.dirty = true;
    }
}

void VulkanBackendAdapter::syncRasterSkinnedVerticesImpl(
    const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::vector<Matrix4x4>& boneMatrices)
{
    if (m_rasterInstances.empty() || m_rasterMeshes.empty() || boneMatrices.empty()) return;
    if (m_viewportMode != ViewportMode::Solid && m_viewportMode != ViewportMode::Matcap) return;

    // Collect skinned triangles grouped by raster mesh key (same grouping as buildRasterGeometry)
    // Key = transform handle pointer → meshKey
    struct SkinnedGroup {
        std::string meshKey;
        std::vector<std::shared_ptr<Triangle>> triangles;
        std::shared_ptr<TriangleMesh> mesh;
    };
    std::unordered_map<void*, SkinnedGroup> skinnedGroups;

    // Map raster instance nodeName → meshKey
    std::unordered_map<std::string, std::string> nodeToMeshKey;
    for (const auto& ri : m_rasterInstances) {
        nodeToMeshKey[ri.nodeName] = ri.meshKey;
    }

    // Traverse scene objects to find skinned triangles
    std::function<void(const std::shared_ptr<Hittable>&)> collectSkinned;
    collectSkinned = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;
        if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) collectSkinned(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            collectSkinned(bvh->left);
            collectSkinned(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->visible || !tri->hasSkinData()) return;
            Transform* th = tri->getTransformPtr();
            void* groupKey = th ? (void*)th : (void*)tri.get();
            auto& grp = skinnedGroups[groupKey];
            if (grp.meshKey.empty()) {
                std::string nodeName = tri->getNodeName();
                auto it = nodeToMeshKey.find(nodeName);
                if (it != nodeToMeshKey.end()) {
                    grp.meshKey = it->second;
                }
            }
            grp.triangles.push_back(tri);
        } else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (!mesh->visible || !mesh->hasSkinWeights()) return;
            void* groupKey = mesh->transform ? static_cast<void*>(mesh->transform.get()) : static_cast<void*>(mesh.get());
            auto& grp = skinnedGroups[groupKey];
            auto it = nodeToMeshKey.find(mesh->nodeName);
            if (it != nodeToMeshKey.end()) grp.meshKey = it->second;
            grp.mesh = mesh;
        }
    };
    for (const auto& obj : objects) collectSkinned(obj);

    if (skinnedGroups.empty()) return;

    // For each skinned group, compute skinned positions/normals and upload
    for (auto& [key, grp] : skinnedGroups) {
        if (grp.meshKey.empty() || (grp.triangles.empty() && !grp.mesh)) continue;
        auto meshIt = m_rasterMeshes.find(grp.meshKey);
        if (meshIt == m_rasterMeshes.end()) continue;

        auto& rmb = meshIt->second;
        const size_t vertCount = grp.mesh && grp.mesh->geometry
            ? grp.mesh->geometry->indices.size()
            : grp.triangles.size() * 3;
        const size_t floatCount = vertCount * 3;
        if (rmb.vertexCount != (uint32_t)vertCount) continue; // topology mismatch

        std::vector<float> newPositions(floatCount);
        std::vector<float> newNormals(floatCount);
        size_t idx = 0;

        if (grp.mesh && grp.mesh->geometry) {
            grp.mesh->applySkinning(boneMatrices);
            const Vec3* positions = grp.mesh->geometry->get_positions();
            const Vec3* normals = grp.mesh->geometry->get_normals();
            for (uint32_t vertexIndex : grp.mesh->geometry->indices) {
                const Vec3 position = positions ? positions[vertexIndex] : Vec3(0.0f);
                const Vec3 normal = normals ? normals[vertexIndex] : Vec3(0.0f, 1.0f, 0.0f);
                newPositions[idx] = position.x; newPositions[idx + 1] = position.y; newPositions[idx + 2] = position.z;
                newNormals[idx] = normal.x; newNormals[idx + 1] = normal.y; newNormals[idx + 2] = normal.z;
                idx += 3;
            }
        } else {
            for (const auto& tri : grp.triangles) {
                for (int v = 0; v < 3; ++v) {
                    Vec3 p = tri->apply_bone_to_vertex(v, boneMatrices);
                    Vec3 n = tri->apply_bone_to_normal(
                        tri->getOriginalVertexNormal(v),
                        tri->getSkinBoneWeights(v),
                        boneMatrices);
                    newPositions[idx] = p.x; newPositions[idx + 1] = p.y; newPositions[idx + 2] = p.z;
                    newNormals[idx] = n.x; newNormals[idx + 1] = n.y; newNormals[idx + 2] = n.z;
                    idx += 3;
                }
            }
        }

        // Dirty-range upload for efficiency
        if (rmb.cpuPositions.size() == floatCount) {
            size_t dirtyMin = floatCount, dirtyMax = 0;
            for (size_t i = 0; i < floatCount; ++i) {
                if (newPositions[i] != rmb.cpuPositions[i] || newNormals[i] != rmb.cpuNormals[i]) {
                    if (i < dirtyMin) dirtyMin = i;
                    if (i > dirtyMax) dirtyMax = i;
                }
            }
            if (dirtyMin <= dirtyMax) {
                dirtyMin = (dirtyMin / 3) * 3;
                dirtyMax = ((dirtyMax / 3) + 1) * 3;
                if (dirtyMax > floatCount) dirtyMax = floatCount;
                const uint64_t byteOff = dirtyMin * sizeof(float);
                const uint64_t byteLen = (dirtyMax - dirtyMin) * sizeof(float);
                m_device->uploadBuffer(rmb.vertexBuffer, &newPositions[dirtyMin], byteLen, byteOff);
                m_device->uploadBuffer(rmb.normalBuffer, &newNormals[dirtyMin],  byteLen, byteOff);
                std::memcpy(&rmb.cpuPositions[dirtyMin], &newPositions[dirtyMin], byteLen);
                std::memcpy(&rmb.cpuNormals[dirtyMin],   &newNormals[dirtyMin],  byteLen);
            }
        } else {
            m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float));
            m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(),  floatCount * sizeof(float));
            rmb.cpuPositions = std::move(newPositions);
            rmb.cpuNormals   = std::move(newNormals);
        }
    }

    m_interactiveViewport.dirty = true;
}

bool VulkanBackendAdapter::updateRasterMeshFromTrianglesImpl(const std::string& nodeName,
                                                             const std::vector<std::shared_ptr<Triangle>>& triangles) {
    if (!m_device || !m_device->isInitialized() || triangles.empty()) return false;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Find matching raster mesh by node name. For instanced meshes the meshKey does
    // not contain the scene node name, so prefer RasterInstance.nodeName matching.
    std::string targetKey;
    for (const auto& ri : m_rasterInstances) {
        if (matchesNodeNameForInstance(ri.nodeName, nodeName) ||
            matchesNodeNameForInstance(nodeName, ri.nodeName) ||
            ri.meshKey.find(nodeName) != std::string::npos) {
            targetKey = ri.meshKey;
            break;
        }
    }
    if (targetKey.empty()) return false;

    auto meshIt = m_rasterMeshes.find(targetKey);
    if (meshIt == m_rasterMeshes.end()) return false;

    auto& rmb = meshIt->second;
    const size_t vertCount = triangles.size() * 3;
    const size_t floatCount = vertCount * 3;

    // Extract new vertex/normal data
    std::vector<float> newPositions, newNormals;
    newPositions.resize(floatCount);
    newNormals.resize(floatCount);

    const size_t numTriangles = triangles.size();
    #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(static)
    for (int t = 0; t < (int)numTriangles; ++t) {
        const auto& tri = triangles[t];
        if (!tri) continue;

        const size_t local_idx = t * 9;
        const bool hasSharedTransform = (tri->getTransformPtr() != nullptr);
        
        Vec3 verts[3];
        Vec3 norms[3];
        bool resolved = false;

        if (tri->parentMesh && tri->parentMesh->geometry) {
            TriangleMesh* parentMesh = tri->parentMesh.get();
            const Vec3* cachedPositions = parentMesh->geometry->get_attribute_data<Vec3>("P");
            const Vec3* cachedNormals = parentMesh->geometry->get_attribute_data<Vec3>("N");
            const Vec3* cachedOrigPositions = parentMesh->geometry->get_attribute_data<Vec3>("P_orig");
            const Vec3* cachedOrigNormals = parentMesh->geometry->get_attribute_data<Vec3>("N_orig");
            const std::vector<uint32_t, DNA::AlignedAllocator<uint32_t, 32>>* cachedIndices = &parentMesh->geometry->indices;

            if (cachedPositions && cachedIndices && !cachedIndices->empty()) {
                uint32_t faceIdx = tri->faceIndex;
                uint32_t baseIdx = faceIdx * 3;
                if (baseIdx + 2 < cachedIndices->size()) {
                    uint32_t i0 = (*cachedIndices)[baseIdx + 0];
                    uint32_t i1 = (*cachedIndices)[baseIdx + 1];
                    uint32_t i2 = (*cachedIndices)[baseIdx + 2];

                    if (hasSharedTransform) {
                        if (tri->hasSkinData()) {
                            verts[0] = tri->getOriginalVertexPosition(0);
                            verts[1] = tri->getOriginalVertexPosition(1);
                            verts[2] = tri->getOriginalVertexPosition(2);
                        } else {
                            verts[0] = cachedOrigPositions ? cachedOrigPositions[i0] : cachedPositions[i0];
                            verts[1] = cachedOrigPositions ? cachedOrigPositions[i1] : cachedPositions[i1];
                            verts[2] = cachedOrigPositions ? cachedOrigPositions[i2] : cachedPositions[i2];
                        }
                        norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                        norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                        norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                    } else {
                        verts[0] = cachedPositions[i0];
                        verts[1] = cachedPositions[i1];
                        verts[2] = cachedPositions[i2];
                        norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                        norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                        norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                    }
                    resolved = true;
                }
            }
        }

        if (!resolved) {
            for (int v = 0; v < 3; ++v) {
                verts[v] = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                norms[v] = tri->getOriginalVertexNormal(v);
            }
        }

        for (int v = 0; v < 3; ++v) {
            newPositions[local_idx + v * 3 + 0] = verts[v].x;
            newPositions[local_idx + v * 3 + 1] = verts[v].y;
            newPositions[local_idx + v * 3 + 2] = verts[v].z;
            newNormals[local_idx + v * 3 + 0]   = norms[v].x;
            newNormals[local_idx + v * 3 + 1]   = norms[v].y;
            newNormals[local_idx + v * 3 + 2]   = norms[v].z;
        }
    }

    const uint32_t newVertCount = (uint32_t)vertCount;

    // Topology changed — full recreate
    if (newVertCount != rmb.vertexCount) {
        const std::vector<uint32_t> preservedInstanceIndices = rmb.instanceIndices;
        m_device->waitIdle();
        destroyRasterMesh(rmb);
        rmb.instanceIndices = preservedInstanceIndices;

        rmb.vertexCount = newVertCount;
        VulkanRT::BufferCreateInfo vci{};
        vci.size = floatCount * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        vci.initialData = nullptr;
        rmb.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = floatCount * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        nci.initialData = nullptr;
        rmb.normalBuffer = m_device->createBuffer(nci);

        // Upload data into device-local buffers via staging path
        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(), floatCount * sizeof(float), 0);
        }

        rmb.cpuPositions = std::move(newPositions);
        rmb.cpuNormals = std::move(newNormals);
        if (!rmb.instanceIndices.empty()) {
            uploadRasterInstanceBuffer(rmb);
        }
    } else if (rmb.cpuPositions.size() == floatCount) {
        // Same topology — find dirty range and upload only that region
        size_t dirtyMin = floatCount;
        size_t dirtyMax = 0;

        for (size_t i = 0; i < floatCount; ++i) {
            if (newPositions[i] != rmb.cpuPositions[i] || newNormals[i] != rmb.cpuNormals[i]) {
                if (i < dirtyMin) dirtyMin = i;
                if (i > dirtyMax) dirtyMax = i;
            }
        }

        if (dirtyMin <= dirtyMax) {
            // Align to vec3 boundaries (12 bytes = 3 floats)
            dirtyMin = (dirtyMin / 3) * 3;
            dirtyMax = ((dirtyMax / 3) + 1) * 3;
            if (dirtyMax > floatCount) dirtyMax = floatCount;

            const uint64_t byteOffset = dirtyMin * sizeof(float);
            const uint64_t byteSize   = (dirtyMax - dirtyMin) * sizeof(float);

            m_device->uploadBuffer(rmb.vertexBuffer, &newPositions[dirtyMin], byteSize, byteOffset);
            m_device->uploadBuffer(rmb.normalBuffer, &newNormals[dirtyMin],  byteSize, byteOffset);

            // Update CPU shadow
            std::memcpy(&rmb.cpuPositions[dirtyMin], &newPositions[dirtyMin], byteSize);
            std::memcpy(&rmb.cpuNormals[dirtyMin],   &newNormals[dirtyMin],  byteSize);
        }
    } else {
        // No CPU shadow yet — full upload and store shadow
        m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float));
        m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(),  floatCount * sizeof(float));
        rmb.cpuPositions = std::move(newPositions);
        rmb.cpuNormals = std::move(newNormals);
    }

    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanBackendAdapter::patchRasterMeshTrianglesImpl(
    const std::string& nodeName,
    const std::vector<size_t>& dirtyIndices,
    const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries) {
    if (!m_device || !m_device->isInitialized() || dirtyIndices.empty() || meshEntries.empty())
        return false;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Find matching raster mesh by node name. For instanced meshes the meshKey does
    // not contain the scene node name, so prefer RasterInstance.nodeName matching.
    std::string targetKey;
    for (const auto& ri : m_rasterInstances) {
        if (matchesNodeNameForInstance(ri.nodeName, nodeName) ||
            matchesNodeNameForInstance(nodeName, ri.nodeName) ||
            ri.meshKey.find(nodeName) != std::string::npos) {
            targetKey = ri.meshKey;
            break;
        }
    }
    if (targetKey.empty()) return false;

    auto meshIt = m_rasterMeshes.find(targetKey);
    if (meshIt == m_rasterMeshes.end()) return false;

    auto& rmb = meshIt->second;
    const size_t expectedVertCount = meshEntries.size() * 3;
    const size_t expectedFloatCount = expectedVertCount * 3;

    // Topology must match; if not, fall back to full update
    if (rmb.vertexCount != static_cast<uint32_t>(expectedVertCount) ||
        rmb.cpuPositions.size() != expectedFloatCount) {
        return false;
    }

    size_t dirtyMinFloat = expectedFloatCount;
    size_t dirtyMaxFloat = 0;

    for (const size_t triIdx : dirtyIndices) {
        if (triIdx >= meshEntries.size()) continue;
        const auto& tri = meshEntries[triIdx].second;
        if (!tri) continue;

        const size_t baseFloat = triIdx * 9; // 3 vertices * 3 floats
        if (baseFloat + 8 >= expectedFloatCount) continue;

        const bool hasSharedTransform = (tri->getTransformPtr() != nullptr);
        for (int v = 0; v < 3; ++v) {
            Vec3 p = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
            Vec3 n = hasSharedTransform ? tri->getOriginalVertexNormal(v) : tri->getOriginalVertexNormal(v);
            const size_t idx = baseFloat + static_cast<size_t>(v) * 3;
            rmb.cpuPositions[idx]     = p.x;
            rmb.cpuPositions[idx + 1] = p.y;
            rmb.cpuPositions[idx + 2] = p.z;
            rmb.cpuNormals[idx]       = n.x;
            rmb.cpuNormals[idx + 1]   = n.y;
            rmb.cpuNormals[idx + 2]   = n.z;
        }

        if (baseFloat < dirtyMinFloat) dirtyMinFloat = baseFloat;
        if (baseFloat + 8 > dirtyMaxFloat) dirtyMaxFloat = baseFloat + 8;
    }

    if (dirtyMinFloat <= dirtyMaxFloat) {
        // Align to vec3 boundaries
        dirtyMinFloat = (dirtyMinFloat / 3) * 3;
        dirtyMaxFloat = ((dirtyMaxFloat / 3) + 1) * 3;
        if (dirtyMaxFloat > expectedFloatCount) dirtyMaxFloat = expectedFloatCount;

        const uint64_t byteOffset = dirtyMinFloat * sizeof(float);
        const uint64_t byteSize   = (dirtyMaxFloat - dirtyMinFloat) * sizeof(float);

        m_device->uploadBuffer(rmb.vertexBuffer, &rmb.cpuPositions[dirtyMinFloat], byteSize, byteOffset);
        m_device->uploadBuffer(rmb.normalBuffer, &rmb.cpuNormals[dirtyMinFloat], byteSize, byteOffset);
    }

    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanBackendAdapter::cloneRasterObjectByNodeName(
    const std::string& sourceNodeName,
    const std::string& newNodeName,
    const Matrix4x4& transform) {
    if (!m_device || !m_device->isInitialized() || sourceNodeName.empty() || newNodeName.empty()) {
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    std::vector<uint32_t> sourceIndices;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_rasterInstances.size()); ++i) {
        const auto& ri = m_rasterInstances[i];
        if (matchesNodeNameForInstance(ri.nodeName, sourceNodeName) ||
            matchesNodeNameForInstance(sourceNodeName, ri.nodeName)) {
            sourceIndices.push_back(i);
        }
    }
    if (sourceIndices.empty()) {
        return false;
    }

    std::unordered_set<std::string> dirtyMeshKeys;
    for (uint32_t sourceIndex : sourceIndices) {
        if (sourceIndex >= m_rasterInstances.size()) continue;

        RasterInstance clone = m_rasterInstances[sourceIndex];
        clone.nodeName = newNodeName;
        clone.transform = transform;
        clone.scatterGroupId = -1;
        clone.scatterInstanceIndex = UINT32_MAX;
        updateRasterInstanceWorldBBox(clone);

        const uint32_t newIndex = static_cast<uint32_t>(m_rasterInstances.size());
        m_rasterInstances.push_back(std::move(clone));

        auto meshIt = m_rasterMeshes.find(m_rasterInstances.back().meshKey);
        if (meshIt != m_rasterMeshes.end()) {
            meshIt->second.instanceIndices.push_back(newIndex);
            dirtyMeshKeys.insert(meshIt->first);
        }
    }

    for (const auto& meshKey : dirtyMeshKeys) {
        auto meshIt = m_rasterMeshes.find(meshKey);
        if (meshIt != m_rasterMeshes.end()) {
            uploadRasterInstanceBuffer(meshIt->second);
        }
    }

    m_interactiveViewport.dirty = true;
    return !dirtyMeshKeys.empty();
}
} // namespace Backend
