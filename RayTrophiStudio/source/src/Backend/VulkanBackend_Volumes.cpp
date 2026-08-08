// ============================================================================
// VulkanBackend_Volumes.cpp
//
// Volume table upload for VulkanBackendAdapter: GpuVDBVolume / GpuGasVolume ->
// VkVolumeInstance conversion and the binding-9 SSBO publication the volume
// closest-hit reads.
//
// Split out of VulkanBackend.cpp under the standing rule that the file takes no
// new structure. These are VulkanBackendAdapter member functions, so the move
// changes no linkage and no declaration -- Backend/VulkanBackend.h already
// declares both.
//
// NOTE: VkVolumeInstance is the shared host/shader ABI (576 bytes,
// include/Backend/vulkan_volume_types.h). Any field added here must be appended
// at the END of that struct and mirrored into EVERY shader that declares it
// (volume_closesthit.rchit, closesthit.rchit, raygen.rgen,
// volume_intersection.rint) in the same change: the SSBO stride is
// per-declaration, so one stale copy shifts every instance after the first.
// ============================================================================

#include "Backend/VulkanBackend.h"
#include "Backend/vulkan_volume_types.h"
#include "Backend/vulkan_world_data.h"
#include "VulkanBackend_Internal.h"
#include "globals.h"
#include "VDBVolume.h"
#include "GasVolume.h"
#include "VDBVolumeManager.h"
// params.h declares OptixTraversableHandle members but does NOT include an
// OptiX header itself; it relies on the including translation unit having one
// already. VulkanBackend.cpp satisfies that by accident, through
// ParallelBVHNode.h -> OptixWrapper.h. Depend on it explicitly here rather than
// dragging in the BVH headers for a typedef.
#include <optix.h>
#include "params.h"  // GpuVDBVolume, GpuGasVolume definitions

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace Backend {

// Stable identity for one TLAS volume slot's object. See the note on
// m_volumeStableKeys in VulkanBackend.h for why the VDB id cannot be used here.
int VulkanBackendAdapter::stableVolumeKey(const std::shared_ptr<Hittable>& instance) const {
    if (!instance) return kNoVolumeKey;
    const void* address = instance.get();
    auto it = m_volumeStableKeys.find(address);
    if (it != m_volumeStableKeys.end()) {
        // Address reuse guard: only honour the cached key when the entry still
        // refers to THIS object. A recycled address gets a fresh key instead of
        // inheriting the dead volume's cached slot contents.
        if (it->second.second.lock() == instance) return it->second.first;
        m_volumeStableKeys.erase(it);
    }
    // Volumes are few, but scenes that repeatedly create and destroy domains would
    // otherwise accumulate dead entries forever. Prune expired ones when the map
    // grows past any plausible live volume count.
    if (m_volumeStableKeys.size() > 64) {
        for (auto e = m_volumeStableKeys.begin(); e != m_volumeStableKeys.end();) {
            if (e->second.second.expired()) e = m_volumeStableKeys.erase(e);
            else ++e;
        }
    }
    const int key = m_nextVolumeStableKey++;
    m_volumeStableKeys.emplace(address,
                               std::make_pair(key, std::weak_ptr<Hittable>(instance)));
    return key;
}

// Identity of every TLAS volume slot, in TLAS order. The key identifies the
// volume OBJECT occupying the slot, so it survives grid unload/re-register
// cycles — which is the whole point: the published-slot cache is consumed after
// a TLAS re-bake, long after a sim rebind may have replaced every VDB id.
std::vector<int> VulkanBackendAdapter::computeOrderedVolumeKeys() const {
    std::vector<int> keys;
    keys.reserve(m_orderedVDBInstances.size());
    for (const auto& hittable : m_orderedVDBInstances) {
        keys.push_back(stableVolumeKey(hittable));
    }
    return keys;
}

// ★★ updateGeometry rebuilds m_orderedVDBInstances and re-bakes every volume
// customIndex. Several call sites do that WITHOUT republishing the volume SSBO
// (file animations, GPU skinning, the sim worker's light path), which leaves
// slot contents laid out for the PREVIOUS TLAS: a volume then reads another
// volume's grid address and shader parameters. That is the black slab filling a
// fluid domain, and it is rare exactly because it needs the order — not just
// the count — to change, which the existing count-based tripwire cannot see.
//
// Fix it where the mapping is rebuilt instead of at every call site: the cached
// slots are keyed by volume identity, so they can be re-laid into the new order
// with no scene packet and nothing for a caller to forget. A volume with no
// cached contents gets an inactive (invisible) slot, which the not-ready retry
// then resolves — never another volume's data.
void VulkanBackendAdapter::republishVolumeSlotsForTLASOrder() {
    if (!m_device) return;
    if (m_publishedVolumeByKey.empty() && m_publishedVolumeKeyOrder.empty()) return;

    const std::vector<int> keys = computeOrderedVolumeKeys();
    if (keys == m_publishedVolumeKeyOrder) return;  // order unchanged — nothing to do

    if (keys.empty()) {
        m_device->updateVolumeBuffer(nullptr, 0, 0);
        m_publishedVolumeKeyOrder.clear();
        m_publishedVolumeByKey.clear();
        return;
    }

    std::vector<VulkanRT::VkVolumeInstance> instances(keys.size());
    std::memset(instances.data(), 0,
                instances.size() * sizeof(VulkanRT::VkVolumeInstance));
    for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == kNoVolumeKey) continue;
        auto it = m_publishedVolumeByKey.find(keys[i]);
        if (it != m_publishedVolumeByKey.end()) {
            instances[i] = it->second;
            continue;
        }
    }
    // Says that updateGeometry re-baked the mapping and this re-lay was needed.
    // If a black slab is seen WITHOUT this line, the order was not the cause.
    SCENE_LOG_ON_CHANGE("volssbo.relay", (long long)keys.size() +
        (long long)std::hash<std::string>{}(std::to_string(keys.empty() ? 0 : keys[0])),
        "[VolumeSSBO] TLAS order changed after updateGeometry; re-laid " +
        std::to_string(keys.size()) + " volume slot(s) by identity.");
    m_device->updateVolumeBuffer(instances.data(),
                                 instances.size() * sizeof(VulkanRT::VkVolumeInstance),
                                 (uint32_t)instances.size());
    m_publishedVolumeKeyOrder = keys;
    m_publishedVolumeByKey.clear();
    for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] != kNoVolumeKey && instances[i].is_active != 0)
            m_publishedVolumeByKey[keys[i]] = instances[i];
    }
}

void VulkanBackendAdapter::updateVDBVolumes(const std::vector<GpuVDBVolume>& vols) {
    if (!m_device) return;
    // Sequence playback can upload a new NanoVDB frame while the viewport
    // render thread is traversing the previous frame's device address. Match
    // material-program updates and serialize the complete VDB resource update
    // against renderProgressiveImpl.
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    bool gridUploadSynchronized = false;
    bool volumeContentChanged = false;
    auto synchronizeGridUpload = [&]() {
        if (!gridUploadSynchronized) {
            // CPU_TO_GPU NanoVDB buffers expose stable device addresses and are
            // read directly by the RT shader. Rewriting one while an earlier
            // dispatch is still traversing it can expose a partially updated
            // tree to the driver. One queue-idle covers all density/temperature
            // uploads in this synchronization batch.
            m_device->waitIdle();
            gridUploadSynchronized = true;
        }
    };
    if (vols.empty()) {
        // No active volumes: release any stale cached VDB buffers immediately.
        if (!m_vdbBuffers.empty() || !m_vdbTempBuffers.empty()) {
            m_device->waitIdle();
        }
        for (auto& [id, buf] : m_vdbBuffers) {
            (void)id;
            if (buf.buffer) m_device->destroyBuffer(buf);
        }
        m_vdbBuffers.clear();
        m_vdbUploadedVersions.clear();
        for (auto& [id, buf] : m_vdbTempBuffers) {
            (void)id;
            if (buf.buffer) m_device->destroyBuffer(buf);
        }
        m_vdbTempBuffers.clear();
        m_vdbTempUploadedVersions.clear();
        // ★INVARIANT: the SSBO length is defined by the TLAS mapping, NOT by how
        // many volumes happen to carry content this frame. m_orderedVDBInstances
        // mirrors the TLAS instance list and only updateGeometry() may rewrite it.
        //
        // This branch used to clear that mapping and publish count 0, which broke
        // the invariant in two ways:
        //  1. The TLAS still holds every volume instance, so each one's baked
        //     customIndex is now >= volCount. The closest-hit's range guard then
        //     fired for every volume box at once. (It passes the ray through today
        //     rather than terminating it, so the symptom is invisible volumes
        //     instead of black boxes — still wrong.)
        //  2. Worse and quieter: with the mapping erased, the NEXT non-empty frame
        //     took the "no geometry build yet" fallback below and built the SSBO in
        //     PACKET order instead of TLAS order. Slot i then describes some other
        //     volume than customIndex i — a coincident gas/liquid pair reads each
        //     other's grid and shader parameters.
        //
        // A frame with no content is expressed the way the rest of the system
        // expresses it: every slot present, every slot is_active = 0. The
        // intersection shader rejects those AABBs before reportIntersectionEXT, so
        // they cost nothing and reveal the scene behind them.
        if (m_orderedVDBInstances.empty()) {
            m_device->updateVolumeBuffer(nullptr, 0, 0);
        } else {
            std::vector<VulkanRT::VkVolumeInstance> inactive(m_orderedVDBInstances.size());
            std::memset(inactive.data(), 0,
                        inactive.size() * sizeof(VulkanRT::VkVolumeInstance));
            m_device->updateVolumeBuffer(
                inactive.data(),
                inactive.size() * sizeof(VulkanRT::VkVolumeInstance),
                (uint32_t)inactive.size());
        }
        m_volumeTemporal.invalidate();
        // Nothing is published, so there is nothing for a reorder to preserve.
        // Dropping the cache keeps republishVolumeSlotsForTLASOrder from
        // resurrecting a previous frame's contents into the new order.
        m_publishedVolumeByKey.clear();
        m_publishedVolumeKeyOrder.clear();
        return;
    }

    // Build id->source map for fast O(1) lookup. Procedural volumes do not have
    // stable VDB ids (sky cloud uses -1), so keep them out of the id map; a
    // shared -1 key can corrupt the TLAS customIndex -> SSBO slot mapping when
    // Nishita sky clouds coexist with live grid-domain volumes.
    std::unordered_map<int, const GpuVDBVolume*> volByID;
    std::vector<const GpuVDBVolume*> proceduralVols;
    proceduralVols.reserve(vols.size());
    for (const auto& v : vols) {
        if (v.vdb_id >= 0) {
            volByID[v.vdb_id] = &v;
        } else if (v.source_type == 3) {
            proceduralVols.push_back(&v);
        }
    }

    // Release cached buffers for volumes that no longer exist in the scene.
    bool destroyedAny = false;
    for (auto it = m_vdbBuffers.begin(); it != m_vdbBuffers.end(); ) {
        if (volByID.find(it->first) == volByID.end()) {
            if (!destroyedAny) {
                m_device->waitIdle();
                destroyedAny = true;
            }
            if (it->second.buffer) m_device->destroyBuffer(it->second);
            m_vdbUploadedVersions.erase(it->first);
            it = m_vdbBuffers.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = m_vdbTempBuffers.begin(); it != m_vdbTempBuffers.end(); ) {
        if (volByID.find(it->first) == volByID.end()) {
            if (!destroyedAny) {
                m_device->waitIdle();
                destroyedAny = true;
            }
            if (it->second.buffer) m_device->destroyBuffer(it->second);
            m_vdbTempUploadedVersions.erase(it->first);
            it = m_vdbTempBuffers.erase(it);
        } else {
            ++it;
        }
    }

    // Identity of each TLAS volume slot, in TLAS order. Kept in lockstep with
    // orderedVols below so the published contents can later be re-laid into a
    // NEW TLAS order without the scene packet — see republishVolumeSlotsForTLASOrder.
    const std::vector<int> orderedKeys = computeOrderedVolumeKeys();

    // ORDERING FIX: SSBO slot i must correspond to the unified volume with TLAS customIndex==i.
    // After updateGeometry(), m_orderedVDBInstances records VDBs in TLAS traversal order.
    // If BVH reorders them vs. scene.vdb_volumes, this ensures shader lookups are correct.
    std::vector<const GpuVDBVolume*> orderedVols;
    if (!m_orderedVDBInstances.empty()) {
        std::size_t proceduralIndex = 0;
        for (const auto& hittable : m_orderedVDBInstances) {
            auto vdb = std::dynamic_pointer_cast<VDBVolume>(hittable);
            int volume_id = -1;
            if (vdb) {
                volume_id = vdb->getVDBVolumeID();
                if (volume_id < 0 && vdb->isProceduralVolume()) {
                    orderedVols.push_back(proceduralIndex < proceduralVols.size()
                        ? proceduralVols[proceduralIndex++]
                        : nullptr);
                    continue;
                }
            } else if (auto gas = std::dynamic_pointer_cast<GasVolume>(hittable)) {
                volume_id = gas->live_vdb_id;
            }
            if (volume_id < 0 && !(vdb && vdb->isProceduralVolume())) { orderedVols.push_back(nullptr); continue; }
            auto it = volByID.find(volume_id);
            orderedVols.push_back(it != volByID.end() ? it->second : nullptr);
        }
    } else if (!m_publishedVolumeKeyOrder.empty()) {
        // ★★★ THE INVARIANT BELOW IS ACTUALLY VIOLATED — do not take the packet
        // fallback here.
        //
        // updateGeometry() clears m_orderedVDBInstances at its START and only
        // re-establishes it at the END (VulkanBackend.cpp:10233). Any publish that
        // lands inside that window sees an EMPTY mapping while the TLAS is still
        // live and still holds volume instances whose customIndex was baked
        // against the PREVIOUS order. Publishing packet order into it makes slot i
        // carry a different volume's grid address and shader parameters, and the
        // two volumes then swap contents from publish to publish.
        //
        // Measured: a gas domain and a fluid domain alternating on one slot —
        //   slot N ... source_type=4 volume_type=2  (fluid SurfaceSDF)
        //   slot N ... source_type=5 volume_type=4  (live dense gas)
        // with `keys=` flipping 2 → 0 exactly when this branch was taken. That is
        // the black band at the domain edge, its cost blow-up, and its dependence
        // on a rebuild: the liquid's AABB was marching the gas's fields.
        //
        // m_publishedVolumeKeyOrder is the right witness: it survives this window
        // (only a real TLAS teardown clears it, VulkanBackend.cpp:9848), so a
        // non-empty value means "the TLAS holds volume slots laid out in an order
        // I can no longer see". Publish that many INACTIVE slots instead. The
        // existing null-entry path below marks anyMappedVolumeMissingContent and
        // asks for one more publish, which resolves once the mapping is back —
        // the same not-ready retry the SurfaceSDF already relies on. A volume
        // invisible for a frame is recoverable; a volume reading another
        // volume's memory is not.
        orderedVols.assign(m_publishedVolumeKeyOrder.size(), nullptr);
        SCENE_LOG_ON_CHANGE("volssbo.midrebuild",
            (long long)m_publishedVolumeKeyOrder.size(),
            "[VolumeSSBO] publish arrived while the TLAS volume mapping was "
            "mid-rebuild; published " +
            std::to_string(m_publishedVolumeKeyOrder.size()) +
            " INACTIVE slot(s) instead of packet order (packet had " +
            std::to_string(vols.size()) + ").");
    } else {
        // Fallback: no TLAS volume instances exist yet (first sync of a session,
        // before updateGeometry has ever run). Packet order is arbitrary but
        // harmless HERE precisely because nothing indexes this buffer yet — there
        // is no customIndex to disagree with, and the branch above now guarantees
        // that "no mapping" really does mean "no TLAS volume slots either".
        for (const auto& v : vols) orderedVols.push_back(&v);
    }
    // Never leave the published count disagreeing with the buffer contents: if
    // there is nothing to publish, publish an empty buffer rather than only
    // zeroing the count and leaving the previous frame's data resident.
    if (orderedVols.empty()) {
        m_device->updateVolumeBuffer(nullptr, 0, 0);
        // The stale-buffer sweep above may have destroyed buffers the cached
        // slots point at, and nothing rewrites the cache on this path.
        m_publishedVolumeByKey.clear();
        m_publishedVolumeKeyOrder.clear();
        return;
    }

    // Convert GpuVDBVolume (OptiX/CUDA struct) → VkVolumeInstance (Vulkan SSBO)
    std::vector<VulkanRT::VkVolumeInstance> instances(orderedVols.size());
    auto& vdbManager = VDBVolumeManager::getInstance();
    auto hostGridLease = vdbManager.lockHostGridAccess();
    // ★★ A TLAS-mapped volume with no packet entry this frame is AMBIGUOUS: it
    // may have been deleted, or its content may merely be regenerating. Both
    // produce the same zeroed slot, and a zeroed slot is invisible.
    //
    // For the fluid SurfaceSDF the second case is routine — the grid is rebuilt
    // continuously, and a backend switch invalidates its render binding for a
    // frame or two. The surface therefore renders on the frame updateGeometry
    // gave it a TLAS slot and then VANISHES on the very next publish, which is
    // exactly the reported "it renders once and is immediately removed". Nothing
    // republishes it afterwards, because the per-frame publishes are gated on
    // volume/gas animation flags that a regenerating fluid binding never raises.
    //
    // So: notice the gap and ask for one more publish. If the volume really was
    // deleted, updateGeometry drops its slot and the request stops on its own.
    bool anyMappedVolumeMissingContent = false;
    // Snapshot the bound BEFORE the loop. The diagnostic below prints both this
    // and the live size: a loop index that exceeds a live size is impossible in
    // defined C++, so if the two ever disagree the container is being mutated
    // or overwritten while it is being iterated — a very different (and far
    // worse) bug than anything in the volume table itself.
    const size_t slotCountAtEntry = orderedVols.size();
    for (size_t i = 0; i < orderedVols.size(); i++) {
        // ★ TRIPWIRE + GUARD. The [VolumeSSBO tag] diagnostic below has been
        // observed printing `i=12 sizeNow=1 sizeAtEntry=1 instances=1`, which is
        // not a reachable state for this loop: i cannot exceed the bound it is
        // tested against. Whatever the cause, the very next statements take a
        // reference to instances[i] and memset sizeof(VkVolumeInstance) through
        // it — with i=12 and a 1-element vector that is a heap write far past the
        // allocation, i.e. silent corruption of whatever follows it. Refuse to
        // perform the write and say so, instead of publishing over other memory.
        if (i >= slotCountAtEntry || i >= instances.size() ||
            i >= orderedVols.size()) {
            SCENE_LOG_WARN("[VolumeSSBO] IMPOSSIBLE loop state: i=" +
                std::to_string(i) +
                " entryBound=" + std::to_string(slotCountAtEntry) +
                " liveBound=" + std::to_string(orderedVols.size()) +
                " instances=" + std::to_string(instances.size()) +
                " — the iterated container changed under the loop; publish aborted "
                "before writing out of bounds.");
            break;
        }
        auto& dst = instances[i];
        memset(&dst, 0, sizeof(dst));
        dst.is_active = 0;
        if (!orderedVols[i]) {
            anyMappedVolumeMissingContent = true;
            // Which mapped slot lost its content, and under which identity.
            SCENE_LOG_ON_CHANGE("volslotgap." + std::to_string(i),
                i < orderedKeys.size() ? orderedKeys[i] : 0,
                std::string("[VolumeGate 3b/4] TLAS volume slot ") + std::to_string(i) +
                " (key=" + std::to_string(i < orderedKeys.size() ? orderedKeys[i] : 0) +
                ") published INACTIVE: no packet entry this frame");
            continue; // deleted/missing → leave inactive slot
        }
        const auto& src = *orderedVols[i];
        const bool liveDenseGas =
            src.dense_fields_valid != 0 &&
            src.dense_density_address != 0 &&
            src.dense_resolution_x > 0 &&
            src.dense_resolution_y > 0 &&
            src.dense_resolution_z > 0;

        // Copy original transforms directly (preserves rotation). Keep this
        // index distinct from the outer volume-slot index: the project uses
        // MSVC legacy /Zc:forScope-, and shadowing `i` here left the outer i at
        // 12, skipping/corrupting the remaining SSBO volume slots.
        for (int transformElement = 0; transformElement < 12; ++transformElement) {
            dst.transform[transformElement]     = src.transform[transformElement];
            dst.inv_transform[transformElement] = src.inv_transform[transformElement];
        }
        
        // Pivot offset for OptiX parity
        dst.pivot_offset[0] = src.pivot_offset[0];
        dst.pivot_offset[1] = src.pivot_offset[1];
        dst.pivot_offset[2] = src.pivot_offset[2];
        dst.source_type = src.source_type;
        // Isosurface IOR (source_type==4) rides _ext_reserved[0], roughness
        // rides _ext_reserved[1] — reusing reserved tail slots rather than
        // moving any field. (The struct is 576 bytes since the majorant block
        // was appended; these offsets are unchanged.)
        dst._ext_reserved[0] = (src.ior > 1.0f) ? src.ior : 1.33f;
        dst._ext_reserved[1] = src.surface_roughness;
        dst._ext_reserved[2] = src.surface_foam;
        // Particle-foam look for the SurfaceSDF single-volume path (temperature
        // channel): tint in [3..5], extinction multiplier in [6].
        dst._ext_reserved[3] = src.foam_color.x;
        dst._ext_reserved[4] = src.foam_color.y;
        dst._ext_reserved[5] = src.foam_color.z;
        dst._ext_reserved[6] = src.foam_opacity;
        dst._ext_reserved[7] = src.density_noise_enabled ? 1.0f : 0.0f;
        dst._ext_reserved[8] = src.density_noise_scale;
        dst._ext_reserved[9] = src.density_noise_strength;
        dst._ext_reserved[10] = static_cast<float>(src.density_noise_detail);
        dst._ext_reserved[11] = static_cast<float>(src.density_noise_seed);
        if (liveDenseGas) {
            // source_type 5 reuses the volume ABI's reserved slots:
            // [0..2] dense resolution, [3..5] dense world/grid origin.
            dst.source_type = 5;
            dst._ext_reserved[0] = static_cast<float>(src.dense_resolution_x);
            dst._ext_reserved[1] = static_cast<float>(src.dense_resolution_y);
            dst._ext_reserved[2] = static_cast<float>(src.dense_resolution_z);
            // Dense simulation metadata is published in world space, whereas
            // closest-hit samples after applying inv_transform.
            dst._ext_reserved[3] =
                src.inv_transform[0] * src.dense_origin.x +
                src.inv_transform[1] * src.dense_origin.y +
                src.inv_transform[2] * src.dense_origin.z +
                src.inv_transform[3];
            dst._ext_reserved[4] =
                src.inv_transform[4] * src.dense_origin.x +
                src.inv_transform[5] * src.dense_origin.y +
                src.inv_transform[6] * src.dense_origin.z +
                src.inv_transform[7];
            dst._ext_reserved[5] =
                src.inv_transform[8] * src.dense_origin.x +
                src.inv_transform[9] * src.dense_origin.y +
                src.inv_transform[10] * src.dense_origin.z +
                src.inv_transform[11];
        }
        // Surface SDF owns slot 6 for foam opacity. Other volume types reuse
        // that otherwise-free slot for the authored minimum emission
        // temperature, avoiding a VkVolumeInstance ABI/size change.
        if (dst.source_type != 4) {
            dst._ext_reserved[6] = src.emission_pad;
        }
        dst.cloud_coverage = src.cloud_coverage;
        dst.cloud_detail = src.cloud_detail;
        dst.cloud_erosion = src.cloud_erosion;
        dst.cloud_base_scale = src.cloud_base_scale;
        dst.cloud_edge_fade = src.cloud_edge_fade;
        dst.cloud_offset_x = src.cloud_offset_x;
        dst.cloud_offset_z = src.cloud_offset_z;
        dst.cloud_seed = src.cloud_seed;

        // VDB native (original file) world-space AABB — used by the shader to remap
        // localPos [-0.5,0.5] → VDB world space before NanoVDB index lookup.
        // Must be local_bbox (not world_bbox) so gizmo moves don't corrupt the mapping.
        dst.aabb_min[0] = src.local_bbox_min.x; dst.aabb_min[1] = src.local_bbox_min.y; dst.aabb_min[2] = src.local_bbox_min.z;
        dst.aabb_max[0] = src.local_bbox_max.x; dst.aabb_max[1] = src.local_bbox_max.y; dst.aabb_max[2] = src.local_bbox_max.z;

        // Density
        dst.density_multiplier = src.density_multiplier;
        dst.density_remap_low = src.density_remap_low;
        dst.density_remap_high = src.density_remap_high;
        dst.noise_scale = 1.0f;
        // Apply the authored density cutoff consistently on dense Vulkan,
        // NanoVDB Vulkan, OptiX and CPU. A Vulkan-only 1e-5 override retained a
        // broad low-density absorption skirt which ended abruptly at the active
        // AABB/topology boundary and looked like an aerial-perspective shadow.
        dst._reserved[0] =
            (src.density_pad > 0.0f) ? src.density_pad : 0.04f;
        // 0 means no graph; otherwise material-table index + 1. Stored as an
        // exact float (material counts are tiny relative to float integer range)
        // to preserve the fixed 512-byte volume ABI.
        dst._reserved[1] = src.material_program_index >= 0
            ? static_cast<float>(src.material_program_index + 1) : 0.0f;
        dst.shadow_stride = std::max(1, std::min(src.shadow_stride, 16));
        // Sync NanoVDB Host Buffer to Vulkan Device Buffer
        dst.volume_type = liveDenseGas ? 4 : 2;
        dst.vdb_grid_address =
            liveDenseGas ? src.dense_density_address : 0;
        dst.vdb_temp_address =
            liveDenseGas ? src.dense_temperature_address : 0;
        // Empty-space acceleration. Only live dense gas produces one; anything
        // else marches as before. Published only as a complete set — a nonzero
        // address with a zero block size would make the shader divide the world
        // into blocks of size 0 and skip everything.
        const bool majorantUsable =
            liveDenseGas && src.dense_majorant_address != 0 &&
            src.dense_majorant_block > 0 &&
            src.dense_majorant_dim[0] > 0 &&
            src.dense_majorant_dim[1] > 0 &&
            src.dense_majorant_dim[2] > 0;
        dst.majorant_address = majorantUsable ? src.dense_majorant_address : 0;
        dst.majorant_dim[0] = majorantUsable ? static_cast<float>(src.dense_majorant_dim[0]) : 0.0f;
        dst.majorant_dim[1] = majorantUsable ? static_cast<float>(src.dense_majorant_dim[1]) : 0.0f;
        dst.majorant_dim[2] = majorantUsable ? static_cast<float>(src.dense_majorant_dim[2]) : 0.0f;
        dst.majorant_block = majorantUsable ? static_cast<float>(src.dense_majorant_block) : 0.0f;
        // Reaction field. Live gas only; a baked VDB has no combustion channel
        // and keeps the temperature-driven emission it always had.
        dst.flame_address = liveDenseGas ? src.dense_flame_address : 0;
        const bool emissiveUsable =
            liveDenseGas && src.dense_emissive_list_address != 0 &&
            src.dense_emissive_capacity > 0 && majorantUsable;
        dst.emissive_list_address = emissiveUsable ? src.dense_emissive_list_address : 0;
        dst.emissive_capacity =
            emissiveUsable ? static_cast<float>(src.dense_emissive_capacity) : 0.0f;
        
        int vdb_id = src.vdb_id;
        if (!liveDenseGas && vdb_id >= 0) {
            void* hostGrid = vdbManager.getHostGrid(vdb_id);
            const size_t gridSize = vdbManager.getHostGridSize(vdb_id);
            const uint32_t currentVersion =
                vdbManager.getContentVersion(vdb_id);

            if (hostGrid && gridSize > 0) {
                auto it = m_vdbBuffers.find(vdb_id);
                bool needsUpload = false;
                
                // Over-allocate by 50% to absorb frame-to-frame NanoVDB growth.
                // CPU_TO_GPU (host-visible, device-local BAR) memory lets uploadBuffer
                // use vkMapMemory + memcpy directly — no staging buffer, no command
                // buffer submit, no vkWaitForFences. This eliminates the per-frame
                // GPU stall that was blocking fluid sim playback. NanoVDB data changes
                // every frame anyway, so device-local-only has no advantage here.
                const size_t allocSize = gridSize + (gridSize / 2);
                if (it == m_vdbBuffers.end() || it->second.size < gridSize) {
                    if (it != m_vdbBuffers.end()) {
                        m_device->waitIdle();
                        m_device->destroyBuffer(it->second);
                    }
                    VulkanRT::BufferCreateInfo ci;
                    ci.size = allocSize;
                    ci.usage = (VulkanRT::BufferUsage)(
                        (uint32_t)VulkanRT::BufferUsage::STORAGE |
                        (uint32_t)VulkanRT::BufferUsage::TRANSFER_DST |
                        0x0100 /* VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT - custom */);
                    ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
                    VulkanRT::BufferHandle buf = m_device->createBuffer(ci);
                    m_vdbBuffers[vdb_id] = buf;
                    it = m_vdbBuffers.find(vdb_id);
                    needsUpload = true;
                }
                
                auto versionIt = m_vdbUploadedVersions.find(vdb_id);
                if (versionIt == m_vdbUploadedVersions.end() ||
                    versionIt->second != currentVersion) {
                    needsUpload = true;
                    volumeContentChanged = true;
                }
                if (it != m_vdbBuffers.end() && it->second.buffer) {
                    if (needsUpload) {
                        synchronizeGridUpload();
                        m_device->uploadBuffer(it->second, hostGrid, gridSize);
                        m_vdbUploadedVersions[vdb_id] = currentVersion;
                    }
                    dst.vdb_grid_address = it->second.deviceAddress;
                }
            }

            // Upload temperature NanoVDB grid for blackbody/color-ramp emission (mode 2)
            void* hostTempGrid =
                vdbManager.getHostTemperatureGrid(vdb_id);
            const size_t tempGridSize =
                vdbManager.getHostTemperatureGridSize(vdb_id);
            if (hostTempGrid && tempGridSize > 0) {
                auto it2 = m_vdbTempBuffers.find(vdb_id);
                bool needsTempUpload = false;
                
                const size_t allocTempSize = tempGridSize + (tempGridSize / 2);
                if (it2 == m_vdbTempBuffers.end() || it2->second.size < tempGridSize) {
                    if (it2 != m_vdbTempBuffers.end()) {
                        m_device->waitIdle();
                        m_device->destroyBuffer(it2->second);
                    }
                    VulkanRT::BufferCreateInfo ci2;
                    ci2.size = allocTempSize;
                    ci2.usage = (VulkanRT::BufferUsage)(
                        (uint32_t)VulkanRT::BufferUsage::STORAGE |
                        (uint32_t)VulkanRT::BufferUsage::TRANSFER_DST |
                        0x0100);
                    ci2.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
                    m_vdbTempBuffers[vdb_id] = m_device->createBuffer(ci2);
                    it2 = m_vdbTempBuffers.find(vdb_id);
                    needsTempUpload = true;
                }
                
                // Check temperature version
                auto tempVersionIt = m_vdbTempUploadedVersions.find(vdb_id);
                if (tempVersionIt == m_vdbTempUploadedVersions.end() || tempVersionIt->second != currentVersion) {
                    needsTempUpload = true;
                }
                
                if (it2 != m_vdbTempBuffers.end() && it2->second.buffer) {
                    if (needsTempUpload) {
                        synchronizeGridUpload();
                        m_device->uploadBuffer(
                            it2->second, hostTempGrid, tempGridSize);
                        m_vdbTempUploadedVersions[vdb_id] = currentVersion;
                    }
                    dst.vdb_temp_address = it2->second.deviceAddress;
                }
            } else {
                // The new sequence frame has no temperature channel. Do not
                // retain a device address from the previous frame.
                auto staleTemp = m_vdbTempBuffers.find(vdb_id);
                if (staleTemp != m_vdbTempBuffers.end()) {
                    synchronizeGridUpload();
                    if (staleTemp->second.buffer)
                        m_device->destroyBuffer(staleTemp->second);
                    m_vdbTempBuffers.erase(staleTemp);
                }
                m_vdbTempUploadedVersions.erase(vdb_id);
            }
            if (dst.vdb_temp_address == 0) {
                const auto existingTemp = m_vdbTempBuffers.find(vdb_id);
                if (existingTemp != m_vdbTempBuffers.end() &&
                    existingTemp->second.buffer) {
                    dst.vdb_temp_address = existingTemp->second.deviceAddress;
                }
            }
        }

        // Scattering
        dst.scatter_color[0] = src.scatter_color.x;
        dst.scatter_color[1] = src.scatter_color.y;
        dst.scatter_color[2] = src.scatter_color.z;
        dst.scatter_coefficient = src.scatter_coefficient;
        dst.scatter_anisotropy = src.scatter_anisotropy;
        dst.scatter_anisotropy_back = src.scatter_anisotropy_back;
        dst.scatter_lobe_mix = src.scatter_lobe_mix;
        dst.scatter_multi = src.scatter_multi;

        // Absorption
        dst.absorption_color[0] = src.absorption_color.x;
        dst.absorption_color[1] = src.absorption_color.y;
        dst.absorption_color[2] = src.absorption_color.z;
        dst.absorption_coefficient = src.absorption_coefficient;

        // Emission
        dst.emission_color[0] = src.emission_color.x;
        dst.emission_color[1] = src.emission_color.y;
        dst.emission_color[2] = src.emission_color.z;
        dst.emission_intensity = src.emission_intensity;

        // Emission mode + blackbody/color-ramp (matches shader extension block)
        dst.emission_mode       = src.emission_mode;
        dst.temperature_scale   = src.temperature_scale;
        dst.blackbody_intensity = src.blackbody_intensity;
        dst.max_temperature     = src.max_temperature;
        dst.color_ramp_enabled  = src.color_ramp_enabled;
        dst.ramp_stop_count     = std::min(src.ramp_stop_count, 8);
        for (int j = 0; j < dst.ramp_stop_count; ++j) {
            dst.ramp_positions[j] = src.ramp_positions[j];
            dst.ramp_colors_r[j]  = src.ramp_colors[j].x;
            dst.ramp_colors_g[j]  = src.ramp_colors[j].y;
            dst.ramp_colors_b[j]  = src.ramp_colors[j].z;
        }
        // OptiX parity: if temperature grid is missing in blackbody/channel mode,
        // fall back to density grid as a scalar source for ramp/blackbody mapping.
        if (dst.vdb_temp_address == 0 && dst.vdb_grid_address != 0 && dst.emission_mode >= 2) {
            dst.vdb_temp_address = dst.vdb_grid_address;
        }

        // Ray march
        dst.step_size = src.step_size;
        dst.max_steps = src.max_steps;
        dst.shadow_steps = src.shadow_steps;
        dst.shadow_strength = src.shadow_strength;

        // Flags
        // volume_type = 3 is an explicit procedural cloud source. Otherwise use
        // NanoVDB when uploaded, with the existing procedural-noise fallback.
        dst.volume_type = liveDenseGas
            ? 4
            : ((src.source_type == 3)
                ? 3
                : ((dst.vdb_grid_address != 0) ? 2 : 1));
        dst.is_active = 1;
        dst.voxel_size = src.voxel_size;

        // ── DIAGNOSTIC: the three fields the SHADER actually gates on ────────
        // volume_closesthit.rchit accepts a liquid as a SurfaceSDF candidate
        // only when source_type==4 AND volume_type==2 AND vdb_grid_address!=0
        // (nearestSurfaceSDFCrossing), and routes the gas march on source_type.
        // Every existing VolumeGate tracks INTENT instead — renderable /
        // in-packet / slot-created — so a slot can keep its "SLOT CREATED"
        // status while these three silently change underneath it, and the
        // change-gated gates stay quiet because their coarse value did not move.
        // That blind spot is why a re-entry into Vulkan RT logs nothing at all
        // even though the picture changes. Key on the TRIPLE so this fires on
        // exactly the transition we cannot currently see.
        // ★ Key on the CONTENTS, not on i. Keying on the slot index meant that if
        // `i` were ever misread, two different volumes could share one log key and
        // appear to alternate on a single slot — an artifact indistinguishable
        // from a real slot collision. Keying on the identity triple removes that
        // failure mode from the instrument itself.
        SCENE_LOG_ON_CHANGE("volssbo.tag." + std::to_string(i) + "." +
                                std::to_string((long long)dst.source_type) + "." +
                                std::to_string((long long)dst.volume_type),
            (long long)dst.source_type * 1000 +
            (long long)dst.volume_type * 100 +
            (dst.vdb_grid_address != 0 ? 10 : 0) +
            (liveDenseGas ? 1 : 0),
            std::string("[VolumeSSBO tag] i=") + std::to_string(i) +
            " sizeNow=" + std::to_string(orderedVols.size()) +
            " sizeAtEntry=" + std::to_string(slotCountAtEntry) +
            " instances=" + std::to_string(instances.size()) +
            // ★ Never print a fallback that is indistinguishable from a real
            // key: "key=0" hid whether slot 12 genuinely carries identity 0 or
            // whether orderedKeys is simply SHORTER than the slot array. Those
            // are two different bugs (identity collision vs mapping/size skew)
            // and the answer decides which one to fix.
            " keys=" + std::to_string(orderedKeys.size()) +
            " key=" + (i < orderedKeys.size()
                           ? std::to_string(orderedKeys[i])
                           : std::string("OUT-OF-RANGE")) +
            " source_type=" + std::to_string(dst.source_type) +
            " volume_type=" + std::to_string(dst.volume_type) +
            " grid=" + (dst.vdb_grid_address != 0 ? "SET" : "NULL") +
            " temp=" + (dst.vdb_temp_address != 0 ? "SET" : "NULL") +
            " liveDenseGas=" + (liveDenseGas ? "1" : "0") +
            "  [SDF-eligible=" +
            ((dst.source_type == 4 && dst.volume_type == 2 &&
              dst.vdb_grid_address != 0) ? "YES" : "NO") + "]");
    }

    // ★Contract check (see m_orderedVDBInstances in the header). The shader indexes
    // this buffer with a customIndex baked at TLAS build time, so publishing a
    // different number of slots than the TLAS was built with is always a bug — it
    // makes volumes read each other's data or vanish. Report it once per change
    // instead of letting it degrade into an intermittent, scene-dependent artifact
    // that costs days to trace back here.
    // ★★ Reported through SCENE_LOG, not VK_INFO — VK_INFO() is a NULL logger
    // (VulkanBackend.h), so this tripwire has never printed anything in its life.
    // A contract this important must be audible; the black volume slab is what it
    // looks like when it is not.
    if (m_tlasVolumeSlotCount != 0 &&
        (uint32_t)instances.size() != m_tlasVolumeSlotCount) {
        SCENE_LOG_ON_CHANGE("volssbo.count",
            (long long)m_tlasVolumeSlotCount * 65536ll + (long long)instances.size(),
            "[VolumeSSBO] customIndex contract violated (COUNT): TLAS built with " +
            std::to_string(m_tlasVolumeSlotCount) + " volume slots but " +
            std::to_string(instances.size()) + " published.");
    }
    // ★ COUNT alone does not protect the contract — the contract is count AND
    // ORDER. A publish whose slot identities differ from the order the TLAS
    // customIndex values were baked from makes each volume read another
    // volume's grid address and shader params: the opaque black slab filling a
    // domain. This is rare precisely because it needs the order, not the size,
    // to change, which the old count check could never see.
    if (!m_publishedVolumeKeyOrder.empty() &&
        orderedKeys.size() == m_publishedVolumeKeyOrder.size() &&
        orderedKeys != m_publishedVolumeKeyOrder) {
        std::string before, after;
        for (size_t i = 0; i < orderedKeys.size() && i < 8; ++i) {
            if (i) { before += ","; after += ","; }
            before += std::to_string(m_publishedVolumeKeyOrder[i]);
            after  += std::to_string(orderedKeys[i]);
        }
        SCENE_LOG_ON_CHANGE("volssbo.order", (long long)orderedKeys.size() +
            (long long)std::hash<std::string>{}(after),
            "[VolumeSSBO] volume slot ORDER changed: [" + before + "] -> [" + after +
            "]. Slots are being re-laid; if anything renders opaque black now, "
            "a consumer published against the old order.");
    }

    m_device->updateVolumeBuffer(instances.data(),
                                  instances.size() * sizeof(VulkanRT::VkVolumeInstance),
                                  (uint32_t)instances.size());

    // Remember what each volume slot HOLDS, keyed by identity. Only valid when
    // the mapping was used (the fallback branch above publishes in arbitrary
    // packet order and has no TLAS order to be keyed against).
    if (orderedKeys.size() == instances.size()) {
        m_publishedVolumeKeyOrder = orderedKeys;
        m_publishedVolumeByKey.clear();
        for (size_t i = 0; i < instances.size(); ++i) {
            if (orderedKeys[i] == kNoVolumeKey) continue;
            m_publishedVolumeByKey[orderedKeys[i]] = instances[i];
        }
    }

    // ── TLAS transform refresh ──────────────────────────────────────────────
    // When a unified volume is moved with the gizmo, setTransform() updates the C++ object
    // but the TLAS AABB instance transform remains stale.  Fix: recompute the
    // scale+translate transform from the current worldBounds for every volume
    // instance found in m_instanceSources and push an updateTLAS call.
    {
        bool tlas_changed = false;
        for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
            Vec3 worldMin;
            Vec3 worldMax;
            if (auto vdb = std::dynamic_pointer_cast<VDBVolume>(m_instanceSources[i])) {
                AABB wb = vdb->getWorldBounds();
                worldMin = wb.min;
                worldMax = wb.max;
            } else if (auto gas = std::dynamic_pointer_cast<GasVolume>(m_instanceSources[i])) {
                gas->getWorldBounds(worldMin, worldMax);
            } else {
                continue;
            }
            Vec3 center = (worldMin + worldMax) * 0.5f;
            Vec3 sz(worldMax.x - worldMin.x, worldMax.y - worldMin.y, worldMax.z - worldMin.z);
            if (sz.x < 1e-4f) sz.x = 1e-4f;
            if (sz.y < 1e-4f) sz.y = 1e-4f;
            if (sz.z < 1e-4f) sz.z = 1e-4f;
            Matrix4x4 newT = Matrix4x4::translation(center) * Matrix4x4::scaling(sz);
            if (!(newT == m_vkInstances[i].transform)) {
                m_vkInstances[i].transform = newT;
                tlas_changed = true;
            }
        }
        if (tlas_changed) {
            m_device->waitIdle();
            auto merged = m_vkInstances;
            for (const auto& h : m_hairVkInstances) merged.push_back(h);
            m_device->updateTLAS(merged);
        }
    }

    // VK_INFO() << "[VulkanBackendAdapter] Uploaded " << instances.size() << " VDB volume(s) to Vulkan SSBO." << std::endl;
    // Accumulation reset alone does not invalidate the separate volume
    // temporal ping-pong history. A sequence frame may keep the same object
    // and screen position while density/emission changes completely; accepting
    // that history leaves bright flame cores behind. Reject history exactly
    // when NanoVDB content changes, then allow it to rebuild over subsequent
    // samples of the same sequence frame.
    if (volumeContentChanged) {
        m_volumeTemporal.invalidate();
    }
    // See the note at the conversion loop. Bounded the same way the geometry-side
    // retry is: a volume whose content never returns must not keep requesting
    // publishes forever, and a publish that finds everything present clears it.
    if (!anyMappedVolumeMissingContent) {
        // Same rearm rule as the geometry-side retry: a run of clean publishes,
        // not a single one. This function runs per frame during a sim, so a
        // volume that flickers in and out of the packet would otherwise keep
        // the budget permanently full and request a rebuild every other frame.
        constexpr uint32_t kCleanRunToRearm = 8;
        if (++m_volumeContentGapCleanRuns >= kCleanRunToRearm) {
            m_volumeContentGapRetries = 0;
        }
    } else {
        m_volumeContentGapCleanRuns = 0;
        constexpr uint32_t kMaxVolumeContentGapRetries = 8;
        if (m_volumeContentGapRetries < kMaxVolumeContentGapRetries) {
            ++m_volumeContentGapRetries;
            // ★ It MUST be this flag. g_gas_volumes_dirty drives
            // updateBackendGasVolumes — the LEGACY gas path — and never reaches
            // this SSBO, so raising it here would look like a fix and change
            // nothing. g_vulkan_rebuild_pending is the one that runs
            // updateGeometry + syncVDBVolumesToGPU as a single publication.
            g_vulkan_rebuild_pending = true;
        }
    }
    resetAccumulation();
}

void VulkanBackendAdapter::updateGasVolumes(const std::vector<GpuGasVolume>& vols) {
    // Gas volumes use similar conversion — for now, handled as basic homogeneous volumes
    if (!m_device || vols.empty()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    std::vector<VulkanRT::VkVolumeInstance> instances(vols.size());
    for (size_t i = 0; i < vols.size(); i++) {
        const auto& src = vols[i];
        auto& dst = instances[i];
        memset(&dst, 0, sizeof(dst));

        // Same /Zc:forScope- rule as the VDB conversion above.
        for (int transformElement = 0; transformElement < 12; ++transformElement) {
            dst.transform[transformElement]     = src.transform[transformElement];
            dst.inv_transform[transformElement] = src.inv_transform[transformElement];
        }
        
        // GasVolume does not have pivot tracking, default to 0
        dst.pivot_offset[0] = 0.0f;
        dst.pivot_offset[1] = 0.0f;
        dst.pivot_offset[2] = 0.0f;

        // Use local bounding box for accurate containment check of localPos
        dst.aabb_min[0] = src.local_bbox_min.x; dst.aabb_min[1] = src.local_bbox_min.y; dst.aabb_min[2] = src.local_bbox_min.z;
        dst.aabb_max[0] = src.local_bbox_max.x; dst.aabb_max[1] = src.local_bbox_max.y; dst.aabb_max[2] = src.local_bbox_max.z;

        dst.density_multiplier = src.density_multiplier;
        dst.density_remap_low = src.density_remap_low;
        dst.density_remap_high = src.density_remap_high;
        dst.noise_scale = 1.0f;
        dst._reserved[0] = (src.density_pad > 0.0f) ? src.density_pad : 0.04f;
        dst._reserved[1] = src.emission_pad;

        dst.scatter_color[0] = src.scatter_color.x;
        dst.scatter_color[1] = src.scatter_color.y;
        dst.scatter_color[2] = src.scatter_color.z;
        dst.scatter_coefficient = src.scatter_coefficient;
        dst.scatter_anisotropy = src.scatter_anisotropy;

        dst.absorption_color[0] = src.absorption_color.x;
        dst.absorption_color[1] = src.absorption_color.y;
        dst.absorption_color[2] = src.absorption_color.z;
        dst.absorption_coefficient = src.absorption_coefficient;

        dst.emission_color[0] = src.emission_color.x;
        dst.emission_color[1] = src.emission_color.y;
        dst.emission_color[2] = src.emission_color.z;
        dst.emission_intensity = src.emission_intensity;

        dst.step_size = src.step_size;
        dst.max_steps = src.max_steps;
        dst.shadow_steps = src.shadow_steps;
        dst.shadow_strength = src.shadow_strength;

        dst.volume_type = 0; // Homogeneous
        dst.is_active = 1;
        dst.voxel_size = src.step_size; // GpuGasVolume has no voxel_size; approximate with step_size
    }

    // Append to existing volume buffer (after VDB volumes)
    // For now, only gas volumes if no VDB volumes exist
    if (m_device->m_volumeCount == 0) {
        m_device->updateVolumeBuffer(instances.data(),
                                      instances.size() * sizeof(VulkanRT::VkVolumeInstance),
                                      (uint32_t)instances.size());
    }
    resetAccumulation();
}

} // namespace Backend
