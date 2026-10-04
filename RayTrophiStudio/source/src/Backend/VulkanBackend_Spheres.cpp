#include "Backend/VulkanBackend.h"

#include "InstanceManager.h"
#include "MaterialManager.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "Fluid/FluidRenderProxy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace {

uint32_t visualChildrenFor(const InstanceGroup& group) {
    return std::max<uint32_t>(1u, group.point_sphere_visual_children);
}

uint32_t sourceMaterial(const ScatterSource& source) {
    for (const auto& mesh : source.flat_meshes) {
        if (!mesh || !mesh->geometry || mesh->geometry->indices.empty()) continue;
        const uint16_t* ids = mesh->geometry->get_material_ids();
        if (!ids) continue;
        const uint16_t id = ids[mesh->geometry->indices.front()];
        return id == MaterialManager::INVALID_MATERIAL_ID ? 0u : id;
    }

    const auto* triangles = source.centered_triangles_ptr
        ? source.centered_triangles_ptr.get() : &source.triangles;
    if (triangles && !triangles->empty() && triangles->front()) {
        const uint16_t id = triangles->front()->getMaterialID();
        return id == MaterialManager::INVALID_MATERIAL_ID ? 0u : id;
    }
    return 0u;
}

bool gatherSphereCloudUnchecked(
    std::vector<VkAabbPositionsKHR>& aabbs,
    std::vector<Backend::VulkanProceduralSphereRecord>& spheres,
    uint32_t& outputPoolCapacity) {
    aabbs.clear();
    spheres.clear();
    outputPoolCapacity = 0;
    const auto& groups = InstanceManager::getInstance().getGroups();
    uint64_t poolCapacity = 0;
    for (const auto& group : groups) {
        if (!group.point_sphere_mode || group.rendered_rt_excluded ||
            group.sources.empty()) {
            continue;
        }
        const uint32_t children = visualChildrenFor(group);
        const uint64_t groupCapacity = static_cast<uint64_t>(
            group.instances.size()) * children;
        if (groupCapacity > std::numeric_limits<uint64_t>::max() - poolCapacity) {
            return false;
        }
        poolCapacity += groupCapacity;
    }
    if (poolCapacity > std::numeric_limits<uint32_t>::max()) return false;
    outputPoolCapacity = static_cast<uint32_t>(poolCapacity);
    aabbs.reserve(outputPoolCapacity);
    spheres.reserve(outputPoolCapacity);

    for (const auto& group : groups) {
        if (!group.point_sphere_mode || group.rendered_rt_excluded ||
            group.sources.empty()) {
            continue;
        }
        std::vector<uint32_t> materials;
        materials.reserve(group.sources.size());
        for (const auto& source : group.sources) materials.push_back(sourceMaterial(source));

        const uint32_t children = visualChildrenFor(group);
        const float childScale = 1.0f / std::cbrt(static_cast<float>(children));
        uint32_t parentIndex = 0;
        for (const auto& instance : group.instances) {
            const uint32_t stableParentIndex = parentIndex++;
            const float diameter = std::max({
                std::abs(instance.scale.x),
                std::abs(instance.scale.y),
                std::abs(instance.scale.z)});
            const float parentRadius = diameter * 0.5f;
            if (!(parentRadius > 0.0f) || !std::isfinite(parentRadius) ||
                !std::isfinite(instance.position.x) ||
                !std::isfinite(instance.position.y) ||
                !std::isfinite(instance.position.z)) {
                continue;
            }

            int sourceIndex = instance.source_index;
            if (sourceIndex < 0 || sourceIndex >= static_cast<int>(materials.size())) {
                sourceIndex = 0;
            }
            const float baseRadius = parentRadius * childScale;
            const float spread = group.point_sphere_visual_spread_radius > 0.0f
                ? group.point_sphere_visual_spread_radius
                : std::max(0.0f, parentRadius - baseRadius);
            Vec3 neighborCandidates[
                RayTrophiSim::Fluid::kFluidRenderProxyNeighborCandidates];
            uint32_t neighborCount = 0;
            constexpr uint32_t kNeighborSteps =
                RayTrophiSim::Fluid::kFluidRenderProxyNeighborCandidates / 2u;
            for (uint32_t slot = 0; slot < kNeighborSteps; ++slot) {
                const uint32_t step = 1u << slot;
                for (uint32_t side = 0; side < 2u; ++side) {
                    if ((side == 0u && stableParentIndex < step) ||
                        (side == 1u &&
                         stableParentIndex + step >= group.instances.size())) {
                        continue;
                    }
                    const std::size_t candidateIndex = side == 0u
                        ? stableParentIndex - step
                        : stableParentIndex + step;
                    const auto& candidate = group.instances[candidateIndex];
                    const float candidateDiameter = std::max({
                        std::abs(candidate.scale.x),
                        std::abs(candidate.scale.y),
                        std::abs(candidate.scale.z)});
                    if (!(candidateDiameter > 0.0f) ||
                        !std::isfinite(candidate.position.x) ||
                        !std::isfinite(candidate.position.y) ||
                        !std::isfinite(candidate.position.z)) {
                        continue;
                    }
                    neighborCandidates[neighborCount++] = candidate.position;
                }
            }
            for (uint32_t child = 0; child < children; ++child) {
                const Vec3 offset =
                    RayTrophiSim::Fluid::fluidRenderProxyNeighborOffset(
                        stableParentIndex,
                        child,
                        children,
                        instance.position,
                        neighborCandidates,
                        neighborCount,
                        spread * RayTrophiSim::Fluid::
                            kFluidRenderProxyNeighborSupportScale,
                        spread);
                const Vec3 center = instance.position + offset;
                const float radius = parentRadius *
                    RayTrophiSim::Fluid::fluidRenderProxyChildRadiusScale(
                        stableParentIndex,
                        child,
                        children,
                        group.point_sphere_visual_size_variation);
                Backend::VulkanProceduralSphereRecord sphere{};
                sphere.centerRadius[0] = center.x;
                sphere.centerRadius[1] = center.y;
                sphere.centerRadius[2] = center.z;
                sphere.centerRadius[3] = radius;
                sphere.materialId = materials[static_cast<size_t>(sourceIndex)];
                // Object Info -> Random must follow the logical sphere, not its
                // animated centre. Hashing centre.xyz would recolour every grain
                // while it falls. Group/parent/child identity stays fixed for the
                // lifetime of this render pool and costs no additional bytes: the
                // 32-byte record already carried three padding words.
                sphere.objectRandom =
                    RayTrophiSim::Fluid::fluidRenderProxyObjectRandom(
                        group.id, stableParentIndex, child);
                spheres.push_back(sphere);

                VkAabbPositionsKHR aabb{};
                aabb.minX = center.x - radius;
                aabb.minY = center.y - radius;
                aabb.minZ = center.z - radius;
                aabb.maxX = center.x + radius;
                aabb.maxY = center.y + radius;
                aabb.maxZ = center.z + radius;
                aabbs.push_back(aabb);
            }
        }
    }
    return true;
}

bool gatherSphereCloud(
    std::vector<VkAabbPositionsKHR>& aabbs,
    std::vector<Backend::VulkanProceduralSphereRecord>& spheres,
    uint32_t& outputPoolCapacity) {
    try {
        return gatherSphereCloudUnchecked(aabbs, spheres, outputPoolCapacity);
    } catch (const std::bad_alloc&) {
        aabbs.clear();
        spheres.clear();
        outputPoolCapacity = 0;
        return false;
    }
}

} // namespace

namespace Backend {

bool VulkanBackendAdapter::appendProceduralSphereCloud(
    std::vector<VulkanRT::TLASInstance>& instances,
    std::vector<std::shared_ptr<Hittable>>& instanceSources) {
    m_foamSphereBlasIndex = UINT32_MAX;
    m_foamSpherePoolCapacity = 0;
    m_foamSphereTlasIndex = UINT32_MAX;
    if (!m_device || !m_device->hasHardwareRT() || !m_device->hasSphereShaders()) {
        return false;
    }

    uint32_t poolCapacity = 0;
    if (!gatherSphereCloud(m_foamSphereAabbs, m_foamSphereRecords, poolCapacity) ||
        poolCapacity == 0) {
        return false;
    }
    const bool hasLiveSpheres = !m_foamSphereRecords.empty();
    if (!hasLiveSpheres) {
        // Keep the procedural slot alive while a stable particle pool is empty.
        // The TLAS mask hides it and radius zero makes the shader reject it; a
        // small valid AABB lets Vulkan build the capacity-sized AS safely.
        m_foamSphereRecords.emplace_back();
        VkAabbPositionsKHR parked{};
        parked.minX = parked.minY = parked.minZ = -1.0e-6f;
        parked.maxX = parked.maxY = parked.maxZ = 1.0e-6f;
        m_foamSphereAabbs.push_back(parked);
    }
    // Headroom so ordinary pool growth refits in place instead of forcing a
    // full scene rebuild (which blanks the cloud for a frame).
    const uint64_t desiredCapacity = static_cast<uint64_t>(poolCapacity) +
        poolCapacity / 8u + 64u;
    const uint32_t capacity = static_cast<uint32_t>(std::min<uint64_t>(
        desiredCapacity,
        std::numeric_limits<uint32_t>::max()));
    if (!m_device->updateFoamSphereBuffer(
            m_foamSphereRecords.data(),
            static_cast<uint32_t>(m_foamSphereRecords.size()),
            capacity)) {
        return false;
    }
    uint32_t blas = m_device->createFoamSphereBLAS(
        m_foamSphereAabbs, capacity);
    if (blas == UINT32_MAX) return false;

    // Hair cleanup owns the contiguous BLAS tail beginning at the first hair
    // index. Lazy RT initialization can publish hair before this sphere cloud,
    // so move the newly appended sphere handle in front of that tail and keep
    // the host-side hair indices aligned. Device addresses stay unchanged.
    if (!m_hairVkInstances.empty()) {
        uint32_t firstHairBlas = UINT32_MAX;
        for (const auto& hair : m_hairVkInstances) {
            firstHairBlas = std::min(firstHairBlas, hair.blasIndex);
        }
        if (firstHairBlas < blas && blas < m_device->m_blasList.size()) {
            auto begin = m_device->m_blasList.begin();
            std::rotate(begin + firstHairBlas, begin + blas, begin + blas + 1u);
            for (auto& hair : m_hairVkInstances) {
                if (hair.blasIndex >= firstHairBlas && hair.blasIndex < blas) {
                    ++hair.blasIndex;
                }
            }
            blas = firstHairBlas;
            m_meshBlasCount = firstHairBlas + 1u;
        }
    }

    VulkanRT::TLASInstance instance;
    instance.blasIndex = blas;
    instance.transform = Matrix4x4::identity();
    instance.materialIndex = 0;
    instance.customIndex = 0;
    instance.mask = hasLiveSpheres ? 0x04 : 0u;
    instance.frontFaceCCW = true;
    instance.sbtRecordOffset = m_device->getSphereSbtOffset();

    m_foamSphereBlasIndex = blas;
    m_foamSpherePoolCapacity = capacity;
    m_foamSphereTlasIndex = static_cast<uint32_t>(instances.size());
    instances.push_back(instance);
    instanceSources.push_back(nullptr);
    return true;
}

bool VulkanBackendAdapter::finalizeDeferredProceduralSphereCloud() {
    if (!m_device || m_foamSphereBlasIndex != UINT32_MAX) return true;
    const auto& groups = InstanceManager::getInstance().getGroups();
    const bool hasDeferredGroups = std::any_of(
        groups.begin(), groups.end(), [](const InstanceGroup& group) {
            return group.point_sphere_mode && !group.rendered_rt_excluded &&
                   !group.sources.empty() && !group.instances.empty();
        });
    if (!hasDeferredGroups) return true;
    if (!m_device->hasSphereShaders()) return false;

    m_device->waitIdle();
    if (!appendProceduralSphereCloud(m_vkInstances, m_instanceSources)) return false;

    std::vector<VulkanRT::TLASInstance> merged = m_vkInstances;
    merged.insert(merged.end(), m_hairVkInstances.begin(), m_hairVkInstances.end());
    VulkanRT::TLASCreateInfo tlasInfo;
    tlasInfo.instances = std::move(merged);
    tlasInfo.allowUpdate = true;
    m_device->createTLAS(tlasInfo);
    resetAccumulation();
    return m_device->hasTLAS();
}

bool VulkanBackendAdapter::updateProceduralSphereCloud(bool& tlasMaskChanged) {
    tlasMaskChanged = false;
    if (!m_device || m_foamSphereBlasIndex == UINT32_MAX ||
        m_foamSphereTlasIndex >= m_vkInstances.size()) {
        return false;
    }

    uint32_t poolCapacity = 0;
    // Only LIVE spheres are built, so only overflowing the reserved capacity
    // needs a rebuild; pool shrink/jitter does not.
    if (!gatherSphereCloud(m_foamSphereAabbs, m_foamSphereRecords, poolCapacity) ||
        m_foamSphereRecords.size() > m_foamSpherePoolCapacity) {
        return false;
    }

    auto& tlasInstance = m_vkInstances[m_foamSphereTlasIndex];
    const uint32_t wantedMask = m_foamSphereRecords.empty() ? 0u : 0x04u;
    if (tlasInstance.mask != wantedMask) {
        tlasInstance.mask = wantedMask;
        tlasMaskChanged = true;
    }
    if (m_foamSphereRecords.empty()) return true;

    if (!m_device->updateFoamSphereBuffer(
            m_foamSphereRecords.data(),
            static_cast<uint32_t>(m_foamSphereRecords.size()),
            m_foamSpherePoolCapacity)) {
        return false;
    }
    return m_device->updateFoamSphereBLAS(
        m_foamSphereBlasIndex, m_foamSphereAabbs);
}

} // namespace Backend
