#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace VulkanRT {

// Triangle BLAS built from buffers that ALREADY live on the device. The raster
// viewport uploaded those positions to draw them; borrowing the same allocation
// is the whole point -- the alternative (VulkanDevice::createBLAS, which takes
// CPU pointers) would upload a second copy of every vertex and the bill this
// step is meant to measure would be double what ray tracing actually costs.
//
// Geometry is EITHER flat SoA (consecutive triples) OR welded with a uint32
// index buffer -- the raster upload chooses per mesh, and this builder has to
// follow it. Building a welded mesh as if it were flat SoA does not fail: it
// silently traces triangles that were never in the model.
uint32_t VulkanDevice::createTriangleBLAS_Device(const BufferHandle& vertices,
                                                 uint32_t vertexCount,
                                                 uint32_t vertexStride,
                                                 const BufferHandle& indices,
                                                 uint32_t indexCount,
                                                 uint64_t* outAsBytes,
                                                 bool allowUpdate) {
    if (outAsBytes) *outAsBytes = 0;
    if (!hasHardwareRT() || !fpCreateAccelerationStructureKHR) return UINT32_MAX;
    if (!vertices.buffer || vertices.deviceAddress == 0 || vertexCount < 3) return UINT32_MAX;

    // An index buffer without a readable device address is NOT a reason to fall
    // back to the flat-SoA reading: that fallback is exactly the bug. Refuse the
    // mesh instead, so it is counted in meshes_skipped and the shadow path sees
    // an incomplete AS rather than a wrong one.
    const bool indexed = indexCount >= 3u;
    if (indexed && (!indices.buffer || indices.deviceAddress == 0 ||
                    indices.size < uint64_t(indexCount) * sizeof(uint32_t))) return UINT32_MAX;

    const uint32_t primitiveCount = indexed ? indexCount / 3u : vertexCount / 3u;
    if (primitiveCount == 0u) return UINT32_MAX;

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertices.deviceAddress;
    triangles.vertexStride = vertexStride;
    triangles.maxVertex = vertexCount - 1u;
    triangles.indexType = indexed ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR;
    if (indexed) triangles.indexData.deviceAddress = indices.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    // Opaque for now: probe visibility rays do not evaluate alpha yet, and a
    // non-opaque flag without an any-hit shader is a slower BVH for nothing.
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = triangles;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    // ALLOW_UPDATE only where the positions actually move. Setting it on every
    // BLAS would slow traversal across the whole scene to buy refits for the
    // static 99% that never refit.
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (allowUpdate) buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    const bool wantCompaction = !allowUpdate && blasCompactionEnabled() &&
                                m_compactionStats.supported;
    if (wantCompaction) buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    // Updatable here means skinned: recordTriangleBLASRefit rebuilds in place.
    else if (allowUpdate) ++m_compactionStats.skippedSkinned;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fpGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

    AccelStructHandle blas{};
    BufferCreateInfo asBufInfo;
    asBufInfo.size = sizeInfo.accelerationStructureSize;
    asBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    asBufInfo.location = MemoryLocation::GPU_ONLY;
    asBufInfo.category = VramCategory::AccelStruct;
    blas.buffer = createBuffer(asBufInfo);
    if (!blas.buffer.buffer) return UINT32_MAX;

    VkAccelerationStructureCreateInfoKHR asCI{};
    asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = blas.buffer.buffer;
    asCI.size = sizeInfo.accelerationStructureSize;
    asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (fpCreateAccelerationStructureKHR(m_device, &asCI, nullptr, &blas.accel) != VK_SUCCESS ||
        blas.accel == VK_NULL_HANDLE) {
        destroyBuffer(blas.buffer);
        return UINT32_MAX;
    }

    const uint64_t alignment = m_capabilities.minScratchAlignment > 0
        ? m_capabilities.minScratchAlignment : 128;
    BufferCreateInfo scratchCI;
    scratchCI.size = (sizeInfo.buildScratchSize + alignment - 1) & ~(alignment - 1);
    scratchCI.usage = BufferUsage::STORAGE;
    scratchCI.location = MemoryLocation::GPU_ONLY;
    scratchCI.category = VramCategory::Scratch;
    BufferHandle scratch = createBuffer(scratchCI);
    if (!scratch.buffer) {
        fpDestroyAccelerationStructureKHR(m_device, blas.accel, nullptr);
        destroyBuffer(blas.buffer);
        return UINT32_MAX;
    }

    buildInfo.dstAccelerationStructure = blas.accel;
    buildInfo.scratchData.deviceAddress = scratch.deviceAddress;
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        destroyBuffer(scratch);
        fpDestroyAccelerationStructureKHR(m_device, blas.accel, nullptr);
        destroyBuffer(blas.buffer);
        return UINT32_MAX;
    }
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);
    VkMemoryBarrier buildBarrier{};
    buildBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    buildBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    buildBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &buildBarrier, 0, nullptr, 0, nullptr);
    const bool queued = wantCompaction && recordBlasCompactionQuery(cmd, blas.accel);
    endSingleTimeCommands(cmd);
    // The build is complete (endSingleTimeCommands waits), so the scratch is
    // free again. Static geometry is not refit, so keeping it would be a
    // permanent allocation for work that never happens twice. An UPDATABLE
    // BLAS refits every frame, so its scratch is kept instead -- and kept at
    // BUILD size, because the periodic MODE_BUILD reset inside
    // recordTriangleBLASRefit needs the larger of the two sizes.
    if (allowUpdate) {
        blas.skinScratchBuffer = scratch;
    } else {
        destroyBuffer(scratch);
    }

    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blas.accel;
    blas.deviceAddress = fpGetAccelerationStructureDeviceAddressKHR(m_device, &addrInfo);

    // ADDRESS only. The raster path owns this buffer and will free it; an AS
    // teardown that also destroyed it would take the drawn geometry with it.
    blas.vertexBuffer = vertices;
    blas.externalGeometry = true;
    blas.allowUpdate = allowUpdate;
    blas.vertexCount = vertexCount;
    // *** The BLAS's OWN triangulation, not a copy of the raster mesh's stats.
    //   recordTriangleBLASRefit has to reproduce the primitive count of the
    //   original build exactly; recording 0 for an indexed mesh would make the
    //   refit push vertexCount/3 fabricated triangles into a tree sized for
    //   indexCount/3. (The non-updatable path never refits, so the old 0 was
    //   harmless there. It stops being harmless the moment allowUpdate can be
    //   true, and indexBuffer is BORROWED here exactly as vertexBuffer is.)
    blas.indexCount = indexed ? indexCount : 0u;
    blas.indexBuffer = indexed ? indices : BufferHandle{};
    // A later MODE_UPDATE must reproduce these flags exactly, so record them
    // rather than assume the default.
    blas.geometryFlags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    blas.buildFlags = buildInfo.flags;

    if (outAsBytes) *outAsBytes = sizeInfo.accelerationStructureSize;
    const uint32_t index = static_cast<uint32_t>(m_blasList.size());
    m_blasList.push_back(blas);
    // Compacted in groups: each copy is a submit, and a scene rebuild creates
    // a thousand of these. The caller flushes the tail before the TLAS reads
    // the (changed) addresses; outAsBytes is the PRE-compaction size.
    if (queued && m_pendingCompactions.size() >= 8u) finishPendingBlasCompactions();
    return index;
}

// Destroys a CONTIGUOUS range this caller created and shrinks the list back to
// it. Only legal when the range sits at the end -- the caller checks that,
// because indices below it are other owners and shifting them would leave every
// stored BLAS index pointing at the wrong geometry.
void VulkanDevice::destroyOwnedBLASRange(uint32_t first, uint32_t count) {
    if (count == 0u || first >= m_blasList.size()) return;
    const size_t end = std::min<size_t>(m_blasList.size(), size_t(first) + count);
    for (size_t i = first; i < end; ++i) {
        AccelStructHandle& blas = m_blasList[i];
        // A destroyed handle value can be reused by the next BLAS; a pending
        // compaction must not follow it there with this one's size.
        m_pendingCompactions.erase(
            std::remove_if(m_pendingCompactions.begin(), m_pendingCompactions.end(),
                           [&](const PendingCompaction& p) { return p.accel == blas.accel; }),
            m_pendingCompactions.end());
        if (blas.accel != VK_NULL_HANDLE && fpDestroyAccelerationStructureKHR)
            fpDestroyAccelerationStructureKHR(m_device, blas.accel, nullptr);
        blas.accel = VK_NULL_HANDLE;
        if (blas.buffer.buffer) destroyBuffer(blas.buffer);
        if (blas.skinScratchBuffer.buffer) destroyBuffer(blas.skinScratchBuffer);
        // vertexBuffer and indexBuffer are BORROWED from the raster path
        // (externalGeometry). Freeing either here would take the drawn geometry
        // with the AS. skinScratchBuffer above is ours, and only updatable
        // BLASes have one.
    }
    if (end == m_blasList.size()) m_blasList.resize(first);
}

// Refit over vertices the DEVICE rewrote. This is the whole reason the skinned
// path exists: GPU skinning writes its output into the same allocation the BLAS
// was built over, so there is nothing to upload -- only the BVH has to catch
// up. updateBLAS() cannot serve here: it takes CPU pointers and would upload a
// base pose over the skinned result.
//
// A MODE_UPDATE keeps the topology of the last full build and only moves the
// nodes. That is accurate while a character deforms around the pose the tree
// was built for, but it degenerates over a long take, so a full in-place
// MODE_BUILD is forced periodically -- same cadence and reasoning as
// refitHairAABB_BLAS.
bool VulkanDevice::recordTriangleBLASRefit(VkCommandBuffer cmd, uint32_t blasIndex) {
    if (!hasHardwareRT() || !fpCmdBuildAccelerationStructuresKHR) return false;
    if (cmd == VK_NULL_HANDLE) return false;
    if (blasIndex >= m_blasList.size()) return false;
    AccelStructHandle& blas = m_blasList[blasIndex];
    if (!blas.allowUpdate || blas.accel == VK_NULL_HANDLE) return false;
    if (!blas.vertexBuffer.buffer || blas.vertexBuffer.deviceAddress == 0 ||
        blas.vertexCount < 3u) return false;
    if (!blas.skinScratchBuffer.buffer) return false;

    constexpr uint32_t kSkinRefitBeforeRebuild = 32;
    const bool fullRebuild = (blas.refitCount >= kSkinRefitBeforeRebuild);

    // The topology is whatever the ORIGINAL build used, and a refit may not
    // change it. Reading it back off the handle rather than re-deriving it from
    // the raster mesh is what keeps the two in agreement.
    const bool indexed = blas.indexCount >= 3u && blas.indexBuffer.buffer &&
                         blas.indexBuffer.deviceAddress != 0;

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = blas.vertexBuffer.deviceAddress;
    triangles.vertexStride = sizeof(float) * 3u;
    triangles.maxVertex = blas.vertexCount - 1u;
    triangles.indexType = indexed ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR;
    if (indexed) triangles.indexData.deviceAddress = blas.indexBuffer.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    // MODE_UPDATE requires the geometry flags to match the original build
    // exactly; a mismatch is undefined behaviour inside the driver, not an
    // error return.
    geometry.flags = blas.geometryFlags;
    geometry.geometry.triangles = triangles;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                    | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = fullRebuild ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                                 : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = fullRebuild ? VK_NULL_HANDLE : blas.accel;
    buildInfo.dstAccelerationStructure = blas.accel;  // same handle, same backing buffer
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.scratchData.deviceAddress = blas.skinScratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = indexed ? blas.indexCount / 3u : blas.vertexCount / 3u;
    if (range.primitiveCount == 0u) return false;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

    // The skinning dispatch that produced these positions was submitted on its
    // own command buffer and fence-waited, so those writes are already visible.
    // The barrier that matters is the one AFTER the build: the ray query reading
    // this AS must not start before the build lands. It is recorded here rather
    // than left to the caller because a caller that batches several refits would
    // otherwise have to remember one barrier per build.
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

    blas.refitCount = fullRebuild ? 0u : (blas.refitCount + 1u);
    return true;
}

} // namespace VulkanRT

namespace Backend {

namespace {
// A scatter forest can carry hundreds of thousands of instances. The TLAS is
// built here for the first time on this device, so the count is capped and the
// remainder REPORTED rather than silently dropped -- an instance that is in the
// raster image but not in the AS is a hole that probe rays would read as sky.
constexpr uint32_t kMaxSceneASInstances = 65536u;

// ★★★★ Byte-at-a-time FNV was costing one saturated core in a foliage scene:
//   every byte is an xor plus a 64-bit multiply whose LATENCY the next byte
//   depends on, so the loop runs at multiply latency per byte, not per word.
//   This consumes eight bytes per step with the same property that matters --
//   every input byte still changes the result, so the gate cannot start missing
//   a change. Only the hash VALUE differs, and nothing persists it.
uint64_t hashMix(uint64_t seed, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    size_t offset = 0;
    for (; offset + 8 <= size; offset += 8) {
        uint64_t word;
        std::memcpy(&word, bytes + offset, sizeof(word));
        seed = (seed ^ word) * 1099511628211ull;
        seed ^= seed >> 29;   // the multiply alone leaves the low bits sticky
    }
    if (offset < size) {
        uint64_t tail = 0;
        std::memcpy(&tail, bytes + offset, size - offset);
        // The remaining LENGTH goes in too: without it "ab" and "ab\0" would
        // land on the same tail word and two different mesh keys could collide.
        seed = (seed ^ tail ^ (size - offset)) * 1099511628211ull;
        seed ^= seed >> 29;
    }
    return seed;
}
uint64_t hashMix(uint64_t seed, uint64_t value) {
    return hashMix(seed, &value, sizeof(value));
}

// Cached content hash of a mesh's flat material-ID stream. Recomputed only
// after someone writes cpuMatIds; see RasterMeshBuffer::matIdsHashValid.
// Templated because RasterMeshBuffer is declared in a protected section: this
// helper never has to NAME the type, only use an object of it.
template <typename Mesh>
uint64_t meshMatIdsHash(const Mesh& mesh) {
    if (!mesh.matIdsHashValid) {
        mesh.matIdsHash = hashMix(1469598103934665603ull, mesh.cpuMatIds.data(),
                                  mesh.cpuMatIds.size() * sizeof(uint32_t));
        mesh.matIdsHashValid = true;
    }
    return mesh.matIdsHash;
}
} // namespace

class RayFusionSceneASResources {
public:
    std::vector<uint32_t> blasIndices;
    std::unordered_map<std::string, uint32_t> meshToBlas;
    std::vector<std::string> hitMeshKeys; // exact TLAS customIndex order
    // ★★★★★ Emissive uggen listesi ICIN, ve neden ayni dongude toplandigi
    //   onemli: emissive geometri bir ISIK olarak orneklenecek, ama AS'te
    //   OLMAYAN bir ucgeni isik saymak, golgesiz -- yani hicbir seyin
    //   engelleyemedigi -- bir isik uretir. Gizli (mask==0) ve cap ile elenen
    //   instance'lar TLAS'tan cikariliyor; bu dizi ayni kesigi tasir, cunku
    //   AYNI dongude, AYNI `continue`'lardan SONRA yazilir.
    std::vector<Matrix4x4> hitTransforms;
    // ★★★ Content signatures, NOT the global geometry generation. Measured
    //   2026-09-07: scene.delete removes the object from the drawn image but
    //   does not bump g_scene_geometry_generation, so a generation-gated AS kept
    //   a DELETED object -- with no error, and no visible symptom until a ray
    //   is finally traced against it. The gate now watches what the AS actually
    //   depends on.
    uint64_t geometrySignature = 0;  // which BLASes must exist
    uint64_t instanceSignature = 0;  // what the TLAS must contain
    // Traced-scene world box, filled by the SAME loop that fills the TLAS.
    bool worldBoundsValid = false;
    float worldMin[3]{}, worldMax[3]{};
    // ★★★★ The third thing the AS depends on, and the one neither signature
    //   above can see: a GPU-skinned mesh keeps its buffer handle, its device
    //   address and its vertex count while the CONTENTS of that buffer are
    //   rewritten every frame. Both signatures hash handles and counts, so both
    //   match -- and the AS is left describing the pose of the frame it was
    //   built in. The raster image animates; the traced shadow does not, and
    //   nothing anywhere reports a problem. Watch the skin generation instead.
    std::vector<uint32_t> skinnedBlasIndices;
    uint64_t skinGeneration = UINT64_MAX;
    uint64_t skinRefits = 0;
    uint64_t skinRefitFailures = 0;
    // ★★★★★ Split, because the TOTAL cannot be acted on. Measured 2026-09-13:
    //   2.18 ms per animated frame against a 1.20 ms render frame -- but that
    //   one number covers four different things (waiting for in-flight frames,
    //   a submit plus fence wait, the BLAS refits themselves, and a full TLAS
    //   rebuild), and they have four different fixes. This repo has already
    //   paid for a frame where "no pass was measured separately" (see
    //   RT_SHADOW_HANDOFF): optimising the wrong quarter of a number is the
    //   predictable result.
    double lastSkinRefitMs = 0.0;       // all of the below
    double lastSkinDrainMs = 0.0;       // waiting for in-flight frames
    double lastSkinBlasMs = 0.0;        // record + submit + fence the refits
    double lastSkinTlasMs = 0.0;        // top-level refresh (drains again)
    double lastSignatureMs = 0.0;
    uint64_t tlasOnlyRefreshes = 0;
    uint32_t instancesHidden = 0;
    uint64_t builtGeometryGeneration = UINT64_MAX;
    uint32_t blasCount = 0;
    uint32_t blasIndexed = 0;
    uint32_t blasFlat = 0;
    uint32_t instanceCount = 0;
    uint32_t instancesSkipped = 0;
    uint32_t meshesSkipped = 0;
    uint64_t asBytes = 0;
    double lastBuildMs = 0.0;
    uint64_t builds = 0;
    uint32_t hairInstanceCount = 0;  // hair AABB BLAS instances in TLAS
    std::string inactiveReason = "not built yet";
    bool ready = false;
};

bool VulkanBackendAdapter::ensureRayFusionSceneAS() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto state = m_rayFusionSceneAS;
    if (!state) {
        state = std::make_shared<RayFusionSceneASResources>();
        m_rayFusionSceneAS = state;
    }
    if (!m_device || !m_device->isInitialized()) {
        state->inactiveReason = "no device";
        return false;
    }
    // Checked before any work: the UI warmers call this every frame on every
    // backend, and a yielded AS rebuilt by them would put the GBs straight
    // back on the GPU the render backend just ran out of.
    if (m_rayFusionSceneASYielded) {
        state->ready = false;
        state->inactiveReason =
            "yielded: the viewport shows Rendered, the render backend owns the VRAM";
        return false;
    }
    // The render backend in Rendered mode: the warmers call this on
    // ctx.backend_ptr too, and it only ever had raster meshes by mistake.
    if (!shouldUseInteractiveViewport()) {
        if (!state->blasIndices.empty()) destroyRayFusionSceneAS();
        auto fresh = m_rayFusionSceneAS;
        if (!fresh) {
            fresh = std::make_shared<RayFusionSceneASResources>();
            m_rayFusionSceneAS = fresh;
        }
        fresh->ready = false;
        fresh->inactiveReason = "backend is not serving the interactive viewport";
        return false;
    }
    if (!m_device->hasHardwareRT() ||
        !m_device->getCapabilities().supportsRayQuery) {
        state->inactiveReason = m_device->hasHardwareRT()
            ? "device reports no ray query support"
            : "device reports no hardware ray tracing";
        return false;
    }
    if (m_rasterMeshes.empty()) {
        state->ready = false;
        state->inactiveReason = "raster geometry not built yet";
        return false;
    }
    // Signatures are recomputed every frame on purpose. A gate that is cheaper
    // than the thing it guards is worth nothing if it can miss a change; the
    // cost of computing it is reported so it can be judged, not assumed.
    const auto signatureStart = std::chrono::steady_clock::now();
    std::unordered_set<std::string> rayFusionMeshKeys;
    rayFusionMeshKeys.reserve(m_rasterMeshes.size());
    for (const auto& instance : m_rasterInstances) {
        if (instance.rayFusionExcluded) continue;
        rayFusionMeshKeys.insert(instance.meshKey);
    }
    uint64_t geometrySignature = 1469598103934665603ull;
    for (const auto& [meshKey, mesh] : m_rasterMeshes) {
        if (mesh.isScatterProxy) continue;
        if (rayFusionMeshKeys.find(meshKey) == rayFusionMeshKeys.end()) continue;
        geometrySignature = hashMix(geometrySignature, meshKey.data(), meshKey.size());
        geometrySignature = hashMix(geometrySignature, mesh.vertexCount);
        geometrySignature = hashMix(geometrySignature,
                                    reinterpret_cast<uint64_t>(mesh.vertexBuffer.buffer));
        // The triangulation is as much a part of the BLAS as the positions. A
        // mesh that gains or loses its index buffer keeps the same vertex count
        // and allocation, so without these the AS would never be rebuilt.
        geometrySignature = hashMix(geometrySignature, mesh.indexCount);
        geometrySignature = hashMix(geometrySignature,
                                    reinterpret_cast<uint64_t>(mesh.indexBuffer.buffer));
    }
    uint64_t instanceSignature = 1469598103934665603ull;
    for (const auto& instance : m_rasterInstances) {
        if (instance.rayFusionExcluded) continue;
        instanceSignature = hashMix(instanceSignature, instance.meshKey.data(),
                                    instance.meshKey.size());
        instanceSignature = hashMix(instanceSignature, &instance.transform,
                                    sizeof(instance.transform));
        instanceSignature = hashMix(instanceSignature, uint64_t(instance.mask));
    }
    state->lastSignatureMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - signatureStart).count();

    if (state->ready && state->geometrySignature == geometrySignature) {
        // Deformation first: it is independent of placement, and a mesh can
        // very easily skin without its node transform moving at all (a
        // character animated purely by bones is exactly that case). Handling it
        // only inside the instance-signature branch would mean the commonest
        // skinned scene never refits.
        refreshRayFusionSkinnedBLAS();
        // It can give up: a failed top-level refresh leaves a current bottom
        // level under a stale top level, and it clears `ready` to say so. Not
        // re-reading that here would return true on exactly the structure it
        // just declared untraceable.
        if (!state->ready) return false;
        if (state->instanceSignature == instanceSignature) {
            return true; // camera motion changes neither, and must not rebuild
        }
        // Only placements moved: the BLASes are still correct, so rebuild the
        // TLAS alone. Tearing down every BLAS because an object slid sideways
        // would turn a cheap refresh into the most expensive path there is.
        const bool built = rebuildRayFusionTLAS();
        // Accept the signature either way. If every instance is now hidden there
        // is genuinely nothing to trace, and retrying that same conclusion on
        // every frame would burn the loop without ever changing the answer.
        state->instanceSignature = instanceSignature;
        if (built) {
            ++state->tlasOnlyRefreshes;
            return true;
        }
        state->ready = false;
        return false;
    }

    // ★★★ Only touch a BLAS list that is entirely ours (plus any hair BLASes
    //   appended by the shared VulkanBackendAdapter hair path). Hair BLASes
    //   are always placed AFTER m_meshBlasCount, so the owned range is
    //   [blasIndices.front() .. blasIndices.back()], which must sit at the
    //   start of m_blasList. Hair BLASes may follow.
    const size_t ownedEnd = state->blasIndices.empty()
        ? 0u : (state->blasIndices.back() + 1u);
    const size_t totalBlas = m_device->m_blasList.size();
    // Hair BLASes are the ones appended beyond the owned triangle range.
    const size_t hairBlasCount = (totalBlas > ownedEnd) ? (totalBlas - ownedEnd) : 0u;
    if (!state->blasIndices.empty() && totalBlas != ownedEnd && hairBlasCount == 0u) {
        state->inactiveReason = "another owner appended to this device BLAS list";
        return false;
    }
    if (state->blasIndices.empty() && !m_device->m_blasList.empty()
        && m_hairVkInstances.empty()) {
        state->inactiveReason = "device BLAS list already owned by the render backend";
        return false;
    }
    // Hair-only scene: hair BLASes are present but no RayFusion triangle
    // BLASes have been built yet. Allow the build to proceed — the hair
    // BLASes will be referenced by the TLAS but not owned/destroyed by us.
    const size_t hairBlasOffset = state->blasIndices.empty()
        ? m_device->m_blasList.size() : ownedEnd;

    const auto started = std::chrono::steady_clock::now();

    // Destroying an acceleration structure a queued frame may still trace is a
    // device loss, not a glitch. This codebase has already paid for that once.
    drainInteractiveViewportInFlight();
    state->ready = false;
    if (!state->blasIndices.empty()) {
        m_device->destroyOwnedBLASRange(state->blasIndices.front(),
                                        static_cast<uint32_t>(state->blasIndices.size()));
        state->blasIndices.clear();
    }
    state->asBytes = 0;
    state->meshesSkipped = 0;
    state->blasIndexed = 0;
    state->blasFlat = 0;

    // BLAS residency deliberately mirrors RASTER residency, not scene content.
    // A hidden mesh keeps its raster vertex buffer (undo must be instant), so it
    // keeps its BLAS too; tying the BLAS to visibility instead would mean a
    // placement-only refresh could reference a BLAS that no longer exists.
    // blas_count is therefore a RESIDENCY number, and does not fall on delete.
    state->meshToBlas.clear();
    state->skinnedBlasIndices.clear();
    auto& meshToBlas = state->meshToBlas;
    for (auto& [meshKey, mesh] : m_rasterMeshes) {
        // Impostor proxies are a raster LOD, not geometry: tracing them would
        // put a camera-facing billboard into the world the rays see.
        if (mesh.isScatterProxy) continue;
        if (rayFusionMeshKeys.find(meshKey) == rayFusionMeshKeys.end()) continue;
        if (!mesh.vertexBuffer.buffer || mesh.vertexCount < 3u) {
            ++state->meshesSkipped;
            continue;
        }
        uint64_t bytes = 0;
        // The index buffer is part of the geometry, not an optimisation: a
        // welded mesh built without it is a different (nonexistent) surface.
        // A skinned mesh is built UPDATABLE so the per-frame refit below is
        // legal; every other mesh keeps the faster non-updatable tree.
        const uint32_t index = m_device->createTriangleBLAS_Device(
            mesh.vertexBuffer, mesh.vertexCount, sizeof(float) * 3u,
            mesh.indexBuffer, mesh.indexCount, &bytes, mesh.hasSkinning);
        if (index == UINT32_MAX) {
            ++state->meshesSkipped;
            continue;
        }
        meshToBlas[meshKey] = index;
        state->blasIndices.push_back(index);
        if (mesh.hasSkinning) state->skinnedBlasIndices.push_back(index);
        state->asBytes += bytes;
        if (mesh.indexBuffer.buffer && mesh.indexCount >= 3u) ++state->blasIndexed;
        else ++state->blasFlat;
    }

    // ★ Before the TLAS: compaction MOVES a BLAS, and the TLAS reads the
    //   address from the device list when it writes its instances.
    m_device->finishPendingBlasCompactions();
    state->asBytes = 0;
    for (uint32_t index : state->blasIndices) {
        state->asBytes += m_device->blasResidentBytes(index);
    }

    if (meshToBlas.empty()) {
        state->inactiveReason = "no mesh produced a BLAS (missing device address or usage flag?)";
        state->ready = false;
        return false;
    }

    if (!rebuildRayFusionTLAS()) {
        state->ready = false;
        return false;
    }

    const auto finished = std::chrono::steady_clock::now();
    state->lastBuildMs =
        std::chrono::duration<double, std::milli>(finished - started).count();
    ++state->builds;
    state->blasCount = static_cast<uint32_t>(state->blasIndices.size());
    state->geometrySignature = geometrySignature;
    state->instanceSignature = instanceSignature;
    // The BLASes were just built over whatever pose is in the buffers right
    // now, so the current skin generation is already satisfied. Leaving this at
    // UINT64_MAX would spend a refit on the very first frame to reproduce the
    // tree that was just built.
    state->skinGeneration = m_rasterSkinGeneration;
    state->builtGeometryGeneration = m_rasterBuiltGeometryGeneration;
    state->inactiveReason.clear();
    state->ready = true;

    SCENE_LOG_INFO(std::string("[RayFusion] scene AS built: ") +
        std::to_string(state->blasCount) + " BLAS (" +
        std::to_string(state->blasIndexed) + " indexed, " +
        std::to_string(state->blasFlat) + " flat), " +
        std::to_string(state->instanceCount) + " instances, " +
        std::to_string(state->skinnedBlasIndices.size()) + " skinned, " +
        std::to_string(state->asBytes / 1024u) + " KB, " +
        std::to_string(state->lastBuildMs) + " ms");
    return true;
}

// ★★★★★ The gate that decides whether anything deformed at all.
//
//   MEASURED 2026-09-13: without it the refit fired ~33 times a second on a
//   timeline parked at frame 0, at 2.18 ms a batch, while the whole rendered
//   frame cost 1.20 ms GPU. The bug was not the refit -- the refit is correct
//   and necessary -- it was asking the wrong question. The skinning COMPUTE
//   DISPATCH runs every frame whether or not the pose changed, so "a dispatch
//   happened" measures the plumbing, not the subject. The pose does.
//
//   Hashing the bone matrices is cheap in the only units that matter here: a
//   200-bone rig is 12.8 KB, against a BLAS refit plus a full TLAS rebuild plus
//   an in-flight drain. And it cannot go stale the way a dirty flag can, since
//   it is derived from the very values the skinning shader consumes.
void VulkanBackendAdapter::noteSkinnedPose(const std::vector<Matrix4x4>& boneMatrices) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // An empty pose is not a pose: callers already refuse to skin without
    // matrices, and treating "no data" as a new pose would refit on nothing.
    if (boneMatrices.empty()) return;
    static_assert(sizeof(Matrix4x4) == 16 * sizeof(float),
                  "Matrix4x4 must stay a bare float[4][4] for the raw-bytes pose hash");
    const uint64_t poseHash = hashMix(1469598103934665603ull, boneMatrices.data(),
                                      boneMatrices.size() * sizeof(Matrix4x4));
    // The COUNT goes in as well: a rig swap that happens to hash the same bytes
    // over a different number of bones is a different pose.
    const uint64_t signature = hashMix(poseHash, uint64_t(boneMatrices.size()));
    if (signature == m_rasterSkinPoseHash) return;
    m_rasterSkinPoseHash = signature;
    ++m_rasterSkinGeneration;
}

// Brings the skinned BLASes up to the pose the skinning compute just wrote.
// Silent and free when the pose is unchanged: noteSkinnedPose only advances the
// generation on a real pose change, so a held pose compares two integers.
// During playback it runs once per POSE, not once per rendered frame -- on the
// measured scene that was 33 refits a second against 84 rendered frames.
void VulkanBackendAdapter::refreshRayFusionSkinnedBLAS() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto state = m_rayFusionSceneAS;
    if (!state || !m_device) return;
    if (state->skinnedBlasIndices.empty()) {
        // Nothing to refit, but the generation still has to be adopted or every
        // future frame would re-enter this and conclude the same thing.
        state->skinGeneration = m_rasterSkinGeneration;
        return;
    }
    if (state->skinGeneration == m_rasterSkinGeneration) return;

    const auto started = std::chrono::steady_clock::now();
    // A refit writes into an acceleration structure a queued frame may still be
    // tracing. This repo has already paid for skipping that drain once. It is
    // timed separately because it is the one part that is NOT our work: it is
    // how long the GPU still owed us, and no amount of refit tuning moves it.
    drainInteractiveViewportInFlight();
    const auto drained = std::chrono::steady_clock::now();
    uint32_t refitted = 0;
    // ★★★ ONE command buffer for every skinned BLAS, not one per BLAS.
    //   beginSingleTimeCommands/endSingleTimeCommands is a submit plus a fence
    //   WAIT, so a per-BLAS loop of them is a CPU stall per mesh per frame for
    //   work the GPU could have taken in a single batch. A character is rarely
    //   one mesh (body, hair cap, clothing are separate skinned meshes), so this
    //   is the difference between one stall and four, every animated frame.
    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        state->skinRefitFailures += state->skinnedBlasIndices.size();
        state->skinGeneration = m_rasterSkinGeneration;
        return;
    }
    for (const uint32_t blasIndex : state->skinnedBlasIndices) {
        if (m_device->recordTriangleBLASRefit(cmd, blasIndex)) {
            ++state->skinRefits;
            ++refitted;
        } else {
            ++state->skinRefitFailures;
        }
    }
    m_device->endSingleTimeCommands(cmd);
    const auto blasDone = std::chrono::steady_clock::now();
    // *** The TLAS is NOT independent of a BLAS refit. Each TLAS instance
    //   carries the world AABB of the BLAS as it stood when the TLAS was built,
    //   and traversal rejects a ray on that box before it ever reaches the
    //   bottom level. A skinned arm that swings outside the box it had at build
    //   time therefore stops casting a shadow ENTIRELY -- and only the part
    //   that left the box does, which reads as a shadow with a bite out of it
    //   rather than as a missing feature. Refreshing the top level is what
    //   makes the refit above visible to a ray at all.
    if (refitted > 0 && !rebuildRayFusionTLAS()) {
        // The bottom level is now current but the top level is not, and there
        // is no honest way to trace that. Fall back to a full rebuild next
        // frame rather than serving a half-updated structure.
        state->ready = false;
        state->geometrySignature = 0;
    }
    const auto finished = std::chrono::steady_clock::now();
    using Ms = std::chrono::duration<double, std::milli>;
    state->lastSkinDrainMs = Ms(drained - started).count();
    state->lastSkinBlasMs = Ms(blasDone - drained).count();
    state->lastSkinTlasMs = Ms(finished - blasDone).count();
    state->lastSkinRefitMs = Ms(finished - started).count();
    // Adopted even when a refit failed. Retrying the same failing refit on
    // every frame would burn the loop without ever changing the answer; the
    // failure count is what says it happened.
    state->skinGeneration = m_rasterSkinGeneration;
}

bool VulkanBackendAdapter::rebuildRayFusionTLAS() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto state = m_rayFusionSceneAS;
    if (!state || !m_device) return false;

    VulkanRT::TLASCreateInfo tlasInfo;
    tlasInfo.allowUpdate = false;
    state->hitMeshKeys.clear();
    state->hitTransforms.clear();
    state->instancesSkipped = 0;
    state->instancesHidden = 0;
    state->worldBoundsValid = false;
    for (int axis = 0; axis < 3; ++axis) {
        state->worldMin[axis] = std::numeric_limits<float>::max();
        state->worldMax[axis] = -std::numeric_limits<float>::max();
    }
    for (const auto& instance : m_rasterInstances) {
        if (instance.rayFusionExcluded) continue;
        // *** scene.delete does NOT erase anything: it sets mask = 0 and leaves
        //   the mesh and the instance resident so undo is instant. The raster
        //   draw loop skips mask == 0, and the traced scene MUST make the same
        //   cut -- an object the user deleted may not keep blocking light. A
        //   masked-out TLAS slot would be equivalent for tracing, but it makes
        //   instance_count report a scene that is not the one being traced, and
        //   this repo's most expensive bug class is an instrument that lies.
        auto it = state->meshToBlas.find(instance.meshKey);
        if (it == state->meshToBlas.end()) continue; // proxy or skipped mesh
        if (instance.mask == 0) {
            // Counted only for instances that would OTHERWISE be traced, so the
            // number answers "what did visibility remove", not "what is absent".
            ++state->instancesHidden;
            continue;
        }
        if (tlasInfo.instances.size() >= kMaxSceneASInstances) {
            ++state->instancesSkipped;
            continue;
        }
        VulkanRT::TLASInstance entry;
        entry.blasIndex = it->second;
        entry.transform = instance.transform;
        entry.mask = instance.mask;
        entry.customIndex = static_cast<uint32_t>(tlasInfo.instances.size());
        tlasInfo.instances.push_back(entry);
        state->hitMeshKeys.push_back(instance.meshKey);
        state->hitTransforms.push_back(instance.transform);
        // `worldBBox` zaten instance kurulurken hesaplanmis; burada yalnizca
        // indirgeniyor. Sonsuz/NaN bir kutu TEK BASINA butun sahneyi yutar ve
        // izgarayi olcusuz yapar, o yuzden eleniyor -- sessizce kabul etmek
        // "izgara neden 10^7 metre" sorusunu birakirdi.
        const Vec3 lo = instance.worldBBox.min;
        const Vec3 hi = instance.worldBBox.max;
        const float mn[3]{lo.x, lo.y, lo.z};
        const float mx[3]{hi.x, hi.y, hi.z};
        bool finite = true;
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(mn[axis]) || !std::isfinite(mx[axis]) || mx[axis] < mn[axis])
                finite = false;
        if (!finite) continue;
        for (int axis = 0; axis < 3; ++axis) {
            state->worldMin[axis] = (std::min)(state->worldMin[axis], mn[axis]);
            state->worldMax[axis] = (std::max)(state->worldMax[axis], mx[axis]);
        }
        state->worldBoundsValid = true;
    }
    state->instanceCount = static_cast<uint32_t>(tlasInfo.instances.size());

    // ── Hair AABB BLAS instances ────────────────────────────────────
    // Hair BLAS’ları RayFusion TLAS’ına opak occluder olarak eklenir.
    // customIndex, mesh instance’larından SONRA başlar: bounce shader bu
    // offset’i kullanarak hair hit’lerini triangle hit’lerden ayırır.
    // Hair’in AABB BLAS’ı zaten device’ta oluşturulmuş durumdadır
    // (VulkanBackendAdapter::uploadHairStrands / uploadHairGuidesGPU).
    state->hairInstanceCount = 0;
    for (const auto& hi : m_hairVkInstances) {
        if (tlasInfo.instances.size() >= kMaxSceneASInstances) break;
        VulkanRT::TLASInstance entry;
        entry.blasIndex   = hi.blasIndex;
        entry.transform   = hi.transform;
        entry.mask        = hi.mask;
        // customIndex, mesh instance sayacından devam eder.
        // Bounce shader’da instance >= state->instanceCount ise hair.
        entry.customIndex = static_cast<uint32_t>(tlasInfo.instances.size());
        tlasInfo.instances.push_back(entry);
        ++state->hairInstanceCount;
    }

    if (tlasInfo.instances.empty()) {
        state->instanceCount = 0;
        state->inactiveReason = state->instancesHidden > 0
            ? "every raster instance is hidden; nothing to trace"
            : "no raster instance referenced a built BLAS";
        return false;
    }

    // The TLAS a queued frame may be tracing is replaced here, so the same
    // drain the BLAS teardown needs applies.
    drainInteractiveViewportInFlight();
    m_device->createTLAS(tlasInfo, VK_NULL_HANDLE);
    if (m_device->getTLASHandle() == VK_NULL_HANDLE) {
        state->inactiveReason = "TLAS creation did not produce a handle";
        return false;
    }

    return true;
}

bool VulkanBackendAdapter::getRayFusionHitInstances(
    std::vector<RayFusion::HitInstance>& out) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    out.clear();
    const auto state = m_rayFusionSceneAS;
    if (!state || !state->ready) return false;
    // ★★★★ One entry per TLAS INSTANCE, but the payload is per MESH. Built
    //   straight off hitMeshKeys, this hashed the referenced mesh's ENTIRE
    //   material-ID stream once per entry -- so every frame it re-hashed the
    //   whole scene's ID streams, and any mesh referenced by several instances
    //   paid again for each one. On a foliage scene (22M flat triangles = 66M
    //   vertices = 264 MB of serial hashing per frame, plus repeats) that is
    //   one core at 100% while the GPU waits. Resolve each DISTINCT mesh once
    //   and expand by index; the published array and its customIndex order stay
    //   byte-identical, so nothing downstream can tell the difference.
    std::unordered_map<std::string, uint32_t> resolved;
    std::vector<RayFusion::HitInstance> unique;
    unique.reserve(64u);
    out.reserve(state->hitMeshKeys.size());
    for (const auto& key : state->hitMeshKeys) {
        const auto cached = resolved.find(key);
        if (cached != resolved.end()) {
            out.push_back(unique[cached->second]);
            continue;
        }
        const auto it = m_rasterMeshes.find(key);
        if (it == m_rasterMeshes.end()) return false;
        const auto& mesh = it->second;
        RayFusion::HitInstance hit{};
        hit.positions = mesh.vertexBuffer.deviceAddress;
        hit.materialIds = mesh.matIdBuffer.deviceAddress;
        hit.vertexCount = mesh.vertexCount;
        // Mirrors the BLAS build above EXACTLY. If these two ever disagree,
        // primitiveIndex resolves against a different triangulation than the
        // one the ray hit, and the shader reads someone else's corner.
        const bool indexed = mesh.indexBuffer.buffer && mesh.indexCount >= 3u &&
            mesh.indexBuffer.deviceAddress &&
            mesh.indexBuffer.size >= uint64_t(mesh.indexCount) * sizeof(uint32_t);
        hit.indices = indexed ? mesh.indexBuffer.deviceAddress : 0ull;
        hit.triangleCount = indexed ? mesh.indexCount / 3u : mesh.vertexCount / 3u;
        // A mesh with no UVs is a real state, not a failure: the raster path
        // only allocates uvBuffer when the geometry carries texture
        // coordinates. Publishing 0 lets the shader fall back to the scalar
        // colour instead of sampling a buffer that does not exist. The size
        // check is not paranoia -- reading two floats per vertex past the end
        // of a short buffer is a device fault, not a wrong pixel.
        const bool uvsUsable = mesh.uvBuffer.buffer && mesh.uvBuffer.deviceAddress &&
            mesh.uvBuffer.size >= uint64_t(mesh.vertexCount) * 2u * sizeof(float);
        hit.uvs = uvsUsable ? mesh.uvBuffer.deviceAddress : 0ull;
        // Material reassignment can update the existing GPU buffer in place, so
        // the ID stream has to reach the publication signature. CACHED: the
        // stream only changes where cpuMatIds is written, and every such site
        // clears matIdsHashValid.
        hit.contentHash = static_cast<uint32_t>(meshMatIdsHash(mesh));
        if (!hit.positions || !hit.materialIds || !hit.triangleCount ||
            mesh.matIdBuffer.size < uint64_t(hit.vertexCount) * sizeof(uint32_t)) return false;
        resolved.emplace(key, static_cast<uint32_t>(unique.size()));
        unique.push_back(hit);
        out.push_back(hit);
    }
    return !out.empty();
}

// ★★★★★ Emissive geometriyi ORNEKLENEBILIR bir isik listesine cevirir.
//
//   Bu, eksik olan yetenegi degil KESTIRICIYI duzeltir. `rfBounceRadiance`
//   donusu zaten `emission + diffuse * incoming`; yani emissive bir ucgene
//   CARPAN isin emission'i getiriyordu. Sorun sunu bulmakti: isik tablosu
//   yalnizca point + directional oldugu icin (Spot/Area bile
//   `unsupportedLights`) bir lamba ancak yarim kure isininin SANSINA
//   bulunuyordu, 1-4 ornek/piksel ile.
//
// ★★★ AYNI KESIK: liste `hitMeshKeys` / `hitTransforms` uzerinden kurulur,
//   yani TLAS'a giren instance kumesinin TAM AYNISI. Gizli (mask==0) ya da cap
//   ile elenen bir instance'i isik saymak, hicbir seyin ENGELLEYEMEDIGI bir
//   isik uretirdi: golge isini o ucgeni bulamaz, katki her yerde gorunur.
bool VulkanBackendAdapter::getRayFusionEmissiveTriangles(
    std::vector<RayFusion::EmissiveTriangle>& out,
    std::vector<uint8_t>& materialInNee,
    RayFusion::BounceStatus& status) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    out.clear();
    materialInNee.assign(m_cachedGpuMaterials.size(), 0u);
    status.emissiveTriangles = 0;
    status.emissiveDropped = 0;
    status.emissiveSkippedIndexed = 0;
    status.emissiveRejectedTransparent = 0;
    status.emissiveArea = 0.0f;
    const auto state = m_rayFusionSceneAS;
    if (!state || !state->ready) return false;
    if (state->hitMeshKeys.size() != state->hitTransforms.size()) return false;
    if (m_cachedGpuMaterials.empty()) return false;

    // Cap: tablo sinirsiz buyuyemez (emissive bir arazi milyonlarca ucgen
    // demek). Asilanlar SAYILIR -- sessizce kirpilmis bir isik listesi,
    // "lamba yeterince aydinlatmiyor" olarak raporlanir ve sebebi gorunmez.
    constexpr size_t kMaxEmissiveTriangles = 8192;
    // Radyans esigi: sifira yakin emission'lari isik yapmak, tablonun tamamini
    // katkisi olcum gurultusunun altinda kalan ucgenlerle doldurmak olurdu.
    constexpr float kMinEmissiveRadiance = 1e-3f;

    // --- 1. GECIS: MALZEME UYGUNLUGU ---------------------------------------
    // *** Karar MALZEME granulunde verilir, ucgen granulunde DEGIL, ve sebebi
    //   tek sayim sozlesmesidir: `rfBounceRadiance` bir malzemenin emission'ini
    //   ya NEE'ye BIRAKIR ya kendisi ekler. Bir malzemenin ucgenlerinin bir
    //   kismi listede bir kismi disinda olsa iki yol da yarim kalirdi --
    //   listedeki kisim IKI kez, disindaki HIC sayilmazdi.
    struct EmissiveCandidate {
        float radiance[3]{};
        uint32_t triangles = 0;
        bool viable = true;
    };
    std::vector<EmissiveCandidate> candidates(m_cachedGpuMaterials.size());
    for (size_t id = 0; id < m_cachedGpuMaterials.size(); ++id) {
        const auto& material = m_cachedGpuMaterials[id];
        auto& candidate = candidates[id];
        const float strength = std::isfinite(material.emission_strength)
            ? std::max(material.emission_strength, 0.0f) : 0.0f;
        const float channels[3] = {
            (std::isfinite(material.emission_r) ? std::max(material.emission_r, 0.0f) : 0.0f) * strength,
            (std::isfinite(material.emission_g) ? std::max(material.emission_g, 0.0f) : 0.0f) * strength,
            (std::isfinite(material.emission_b) ? std::max(material.emission_b, 0.0f) : 0.0f) * strength};
        candidate.radiance[0] = channels[0];
        candidate.radiance[1] = channels[1];
        candidate.radiance[2] = channels[2];
        if (std::max(channels[0], std::max(channels[1], channels[2])) < kMinEmissiveRadiance) {
            candidate.viable = false;  // emissive degil: sayilacak bir sey yok
            continue;
        }
        // ★★★ Bu deponun en pratik surprizi: LAMBA ABAJURLARI TRANSPARAN.
        //   Yarisaydam bir emissive yuzeyi alan isigi gibi ornekleyip golge
        //   isinini de ondan gecirmek, isigi iki kez saymak olurdu. Kapsam
        //   disi ve SAYILIR: "en cok beklenen emissive nesne tam olarak katki
        //   VERMEYEN nesne" surprizi gorunur olsun.
        if (material.opacity < 0.999f || material.transmission > 0.001f) {
            candidate.viable = false;
            ++status.emissiveRejectedTransparent;
        }
    }

    // Indeksli (welded) mesh bir malzemeyi KISMI birakabilir; o malzeme
    // butunuyle elenir, cunku kismi temsil tek sayim sozlesmesini bozar.
    for (size_t instance = 0; instance < state->hitMeshKeys.size(); ++instance) {
        const auto it = m_rasterMeshes.find(state->hitMeshKeys[instance]);
        if (it == m_rasterMeshes.end()) continue;
        const auto& mesh = it->second;
        if (mesh.cpuPositions.size() < size_t(mesh.vertexCount) * 3u ||
            mesh.cpuMatIds.size() < size_t(mesh.vertexCount)) continue;
        // ★ Welded mesh'in indeks tamponu yalnizca GPU'da (RasterMeshBuffer'da
        //   cpuIndices YOK), yani ucgenleri CPU'da COZULEMEZ. `triangle*3` ile
        //   dogrudan vertex dizisine girmek, ucgeni DEPOLAMA SIRASINDAN
        //   uydurmak olurdu -- bu depo o hatayi BLAS tarafinda bir kez odedi.
        if (mesh.indexBuffer.buffer && mesh.indexCount >= 3u) {
            bool removed = false;
            for (uint32_t v = 0; v < mesh.vertexCount; ++v) {
                const uint32_t id = mesh.cpuMatIds[v] & 0x7fffffffu;
                if (id < candidates.size() && candidates[id].viable) {
                    candidates[id].viable = false;
                    removed = true;
                }
            }
            if (removed) ++status.emissiveSkippedIndexed;
            continue;
        }
        const uint32_t triangleCount = mesh.vertexCount / 3u;
        for (uint32_t triangle = 0; triangle < triangleCount; ++triangle) {
            const uint32_t id = mesh.cpuMatIds[triangle * 3u] & 0x7fffffffu;
            if (id < candidates.size() && candidates[id].viable) ++candidates[id].triangles;
        }
    }

    // Cap'i MALZEME butunuyle uygula: sigmayan malzeme tamamen disarida kalir
    // ve ucgenleri `emissiveDropped`'a yazilir. Yarim alinan bir malzeme, iki
    // yolun da yarim kalmasi demek olurdu.
    size_t budget = kMaxEmissiveTriangles;
    for (size_t id = 0; id < candidates.size(); ++id) {
        auto& candidate = candidates[id];
        if (!candidate.viable || candidate.triangles == 0) {
            candidate.viable = false;
            continue;
        }
        if (candidate.triangles > budget) {
            candidate.viable = false;
            status.emissiveDropped += candidate.triangles;
            continue;
        }
        budget -= candidate.triangles;
        materialInNee[id] = 1u;
    }

    // --- 2. GECIS: UCGENLERI URET ------------------------------------------
    std::vector<float> areas;
    out.reserve(kMaxEmissiveTriangles - budget);
    areas.reserve(out.capacity());
    for (size_t instance = 0; instance < state->hitMeshKeys.size(); ++instance) {
        const auto it = m_rasterMeshes.find(state->hitMeshKeys[instance]);
        if (it == m_rasterMeshes.end()) continue;
        const auto& mesh = it->second;
        if (mesh.cpuPositions.size() < size_t(mesh.vertexCount) * 3u ||
            mesh.cpuMatIds.size() < size_t(mesh.vertexCount)) continue;
        if (mesh.indexBuffer.buffer && mesh.indexCount >= 3u) continue;
        const Matrix4x4& model = state->hitTransforms[instance];
        const uint32_t triangleCount = mesh.vertexCount / 3u;
        for (uint32_t triangle = 0; triangle < triangleCount; ++triangle) {
            const uint32_t corner = triangle * 3u;
            const uint32_t id = mesh.cpuMatIds[corner] & 0x7fffffffu;
            if (id >= candidates.size() || !candidates[id].viable) continue;

            Vec3 world[3];
            for (uint32_t v = 0; v < 3; ++v) {
                const size_t base = size_t(corner + v) * 3u;
                world[v] = model.transform_point(Vec3(mesh.cpuPositions[base],
                                                      mesh.cpuPositions[base + 1],
                                                      mesh.cpuPositions[base + 2]));
            }
            const float area = 0.5f *
                (world[1] - world[0]).cross(world[2] - world[0]).length();
            // Dejenere ucgen hicbir isik tasimaz; alan agirligi onu zaten hic
            // secmezdi, ama CDF'de 0'a bolme riskini burada kesiyoruz.
            if (!std::isfinite(area) || area <= 0.0f) continue;

            RayFusion::EmissiveTriangle entry{};
            for (uint32_t v = 0; v < 3; ++v) {
                float* target = v == 0 ? entry.v0 : (v == 1 ? entry.v1 : entry.v2);
                target[0] = world[v].x;
                target[1] = world[v].y;
                target[2] = world[v].z;
            }
            entry.v1[3] = area;
            entry.radiance[0] = candidates[id].radiance[0];
            entry.radiance[1] = candidates[id].radiance[1];
            entry.radiance[2] = candidates[id].radiance[2];
            out.push_back(entry);
            areas.push_back(area);
            status.emissiveArea += area;
        }
    }

    // ★★ Alan-agirlikli CDF. Uniform secim, buyuk bir emissive duvari ile
    //   kucuk bir filamani AYNI olasilikla secerdi ve varyans duvarin
    //   katkisinda patlardi. `v0.w` artan normalize kumulatif; GPU ikili arama
    //   yapar ve katkiyi toplam ALANLA carpar (pdf = 1 / toplamAlan).
    if (!out.empty() && status.emissiveArea > 0.0f) {
        float accumulated = 0.0f;
        for (size_t i = 0; i < out.size(); ++i) {
            accumulated += areas[i];
            out[i].v0[3] = accumulated / status.emissiveArea;
            // ★ Toplam alan HER girise yazilir, bir header elemanina DEGIL: bir
            //   header, CDF ikili aramasinin indekslemesini kaydirirdi ve
            //   belirtisi "yanlis ucgen orneklendi" olurdu. pdf = 1/toplamAlan
            //   oldugu icin shader bu sayiya her ornekte ihtiyac duyuyor.
            out[i].v2[3] = status.emissiveArea;
        }
        // Yuvarlama son kovayi 1'in ALTINDA birakirsa, u=1'e yakin bir ornek
        // hicbir ucgeni secemez ve katki sessizce duser.
        out.back().v0[3] = 1.0f;
    } else {
        // ★★★★ Hicbir ucgen uretilemediyse hicbir malzeme temsil EDILMIYOR:
        //   tek sayim bayragi TEMIZLENMELI. Yoksa emission hem NEE'den (liste
        //   bos) hem yarim kure isinindan (bayrak onu susturuyor) DUSER ve
        //   lambalar GI'da tamamen kaybolur -- eklemeye calistigimiz seyin
        //   tam tersi.
        std::fill(materialInNee.begin(), materialInNee.end(), uint8_t(0));
    }
    status.emissiveTriangles = static_cast<uint32_t>(out.size());
    return !out.empty();
}

bool VulkanBackendAdapter::getRayFusionSceneASStatus(RayFusionSceneASStatus& out) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    out = {};
    out.hardware_rt = m_device && m_device->hasHardwareRT();
    out.yielded = m_rayFusionSceneASYielded;
    out.yields = m_rayFusionSceneASYields;
    if (m_device) {
        out.vram_measured = m_device->queryDeviceLocalMemory(out.vram_usage_bytes,
                                                             out.vram_budget_bytes);
    }
    const auto state = m_rayFusionSceneAS;
    if (!state) {
        out.inactive_reason = m_rayFusionSceneASYielded
            ? "yielded: the viewport shows Rendered, the render backend owns the VRAM"
            : "scene AS has never been requested on this backend";
        return false;
    }
    out.ready = state->ready;
    out.blas_count = state->blasCount;
    out.instance_count = state->instanceCount;
    out.instances_skipped = state->instancesSkipped;
    out.instances_hidden = state->instancesHidden;
    out.blas_indexed = state->blasIndexed;
    out.blas_flat = state->blasFlat;
    out.meshes_skipped = state->meshesSkipped;
    out.as_bytes = state->asBytes;
    out.last_build_ms = state->lastBuildMs;
    out.builds = state->builds;
    out.built_geometry_generation = state->builtGeometryGeneration;
    out.geometry_signature = state->geometrySignature;
    out.instance_signature = state->instanceSignature;
    out.signature_ms = state->lastSignatureMs;
    out.tlas_only_refreshes = state->tlasOnlyRefreshes;
    out.world_bounds_valid = state->worldBoundsValid;
    for (int axis = 0; axis < 3; ++axis) {
        out.world_min[axis] = state->worldMin[axis];
        out.world_max[axis] = state->worldMax[axis];
    }
    out.blas_skinned = static_cast<uint32_t>(state->skinnedBlasIndices.size());
    out.skin_refits = state->skinRefits;
    out.skin_refit_failures = state->skinRefitFailures;
    out.last_skin_refit_ms = state->lastSkinRefitMs;
    out.last_skin_drain_ms = state->lastSkinDrainMs;
    out.last_skin_blas_ms = state->lastSkinBlasMs;
    out.last_skin_tlas_ms = state->lastSkinTlasMs;
    out.inactive_reason = state->inactiveReason;
    return true;
}

void VulkanBackendAdapter::destroyRayFusionSceneAS() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto state = m_rayFusionSceneAS;
    if (!state || !m_device) {
        m_rayFusionSceneAS.reset();
        return;
    }
    if (!state->blasIndices.empty()) {
        m_device->destroyOwnedBLASRange(state->blasIndices.front(),
                                        static_cast<uint32_t>(state->blasIndices.size()));
    }
    m_rayFusionSceneAS.reset();
}

void VulkanBackendAdapter::yieldRayFusionSceneAS() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (m_rayFusionSceneASYielded) return;
    m_rayFusionSceneASYielded = true;
    if (!m_rayFusionSceneAS) return;
    uint64_t released = 0;
    if (const auto state = m_rayFusionSceneAS) released = state->asBytes;
    // The caller drained this backend; the AS must not be referenced by an
    // in-flight shadow/probe submit when its memory goes.
    destroyRayFusionSceneAS();
    ++m_rayFusionSceneASYields;
    SCENE_LOG_INFO("[RayFusion] scene AS yielded to the render backend: " +
                   std::to_string(released / (1024ull * 1024ull)) + " MB released");
}

void VulkanBackendAdapter::reclaimRayFusionSceneAS() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Only lifts the gate; the next ensureRayFusionSceneAS rebuilds lazily
    // from the (still resident) raster geometry.
    m_rayFusionSceneASYielded = false;
}

bool VulkanBackendAdapter::isRayFusionSceneASYielded() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_rayFusionSceneASYielded;
}

} // namespace Backend
