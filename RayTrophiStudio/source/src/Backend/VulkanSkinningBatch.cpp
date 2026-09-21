#include "Backend/VulkanBackend.h"

#include <cstring>
#include <vector>

namespace VulkanRT {

namespace {

constexpr uint64_t kFallbackScratchAlignment = 128u;

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

bool canBatchSkin(const AccelStructHandle& blas) {
    return blas.hasSkinning && blas.allowUpdate && blas.accel != VK_NULL_HANDLE &&
           blas.vertexCount > 0u && blas.baseVertexBuffer.buffer &&
           blas.baseNormalBuffer.buffer && blas.boneIndexBuffer.buffer &&
           blas.boneWeightBuffer.buffer && blas.vertexBuffer.buffer &&
           blas.vertexBuffer.deviceAddress != 0 && blas.normalBuffer.buffer;
}

VkBuildAccelerationStructureFlagsKHR buildFlagsFor(
    const AccelStructHandle& blas) {
    if (blas.buildFlags != 0) {
        return blas.buildFlags;
    }
    return VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
           VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
}

void describeGeometry(const AccelStructHandle& blas,
                      VkAccelerationStructureGeometryTrianglesDataKHR& triangles,
                      VkAccelerationStructureGeometryKHR& geometry) {
    const bool indexed = blas.indexCount >= 3u && blas.indexBuffer.buffer &&
                         blas.indexBuffer.deviceAddress != 0;
    triangles = {};
    triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = blas.vertexBuffer.deviceAddress;
    triangles.vertexStride = sizeof(float) * 3u;
    triangles.maxVertex = blas.vertexCount - 1u;
    triangles.indexType = indexed ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR;
    if (indexed) {
        triangles.indexData.deviceAddress = blas.indexBuffer.deviceAddress;
    }

    geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = blas.geometryFlags;
    geometry.geometry.triangles = triangles;
}

uint32_t primitiveCountFor(const AccelStructHandle& blas) {
    const bool indexed = blas.indexCount >= 3u && blas.indexBuffer.buffer &&
                         blas.indexBuffer.deviceAddress != 0;
    return indexed ? blas.indexCount / 3u : blas.vertexCount / 3u;
}

} // namespace

bool VulkanDevice::dispatchSkinningBatch(
    const std::vector<Matrix4x4>& boneMatrices) {
    if (!m_device || !m_commandPool || !m_computeQueue || boneMatrices.empty()) {
        return false;
    }
    if (m_skinningPipeline == VK_NULL_HANDLE ||
        m_skinningPipelineLayout == VK_NULL_HANDLE ||
        m_skinningDescPool == VK_NULL_HANDLE ||
        m_skinningDescLayout == VK_NULL_HANDLE ||
        !fpCmdBuildAccelerationStructuresKHR ||
        !fpGetAccelerationStructureBuildSizesKHR) {
        return false;
    }

    std::vector<uint32_t> indices;
    indices.reserve(m_blasList.size());
    for (uint32_t index = 0; index < static_cast<uint32_t>(m_blasList.size()); ++index) {
        const AccelStructHandle& blas = m_blasList[index];
        if (!blas.hasSkinning) {
            continue;
        }
        if (!canBatchSkin(blas) || primitiveCountFor(blas) == 0u) {
            return false;
        }
        indices.push_back(index);
    }
    if (indices.empty()) {
        return true;
    }

    // Bone indices across the scene address one global palette. One BLAS owns
    // the allocation for lifetime purposes; every skin descriptor binds it.
    AccelStructHandle& paletteOwner = m_blasList[indices.front()];
    const uint64_t paletteBytes = boneMatrices.size() * sizeof(Matrix4x4);
    if (!paletteOwner.persistentBoneMatsBuffer.buffer ||
        paletteOwner.persistentBoneMatsBufSize < paletteBytes) {
        if (paletteOwner.persistentBoneMatsBuffer.buffer) {
            destroyBuffer(paletteOwner.persistentBoneMatsBuffer);
        }
        BufferCreateInfo paletteInfo{};
        paletteInfo.size = paletteBytes;
        paletteInfo.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        paletteInfo.location = MemoryLocation::CPU_TO_GPU;
        paletteOwner.persistentBoneMatsBuffer = createBuffer(paletteInfo);
        if (!paletteOwner.persistentBoneMatsBuffer.buffer ||
            !paletteOwner.persistentBoneMatsBuffer.memory) {
            paletteOwner.persistentBoneMatsBuffer = {};
            paletteOwner.persistentBoneMatsBufSize = 0;
            return false;
        }
        paletteOwner.persistentBoneMatsBufSize = paletteBytes;
    }

    void* mappedPalette = nullptr;
    if (vkMapMemory(m_device, paletteOwner.persistentBoneMatsBuffer.memory, 0,
                    paletteBytes, 0, &mappedPalette) != VK_SUCCESS ||
        !mappedPalette) {
        return false;
    }
    std::memcpy(mappedPalette, boneMatrices.data(), paletteBytes);
    vkUnmapMemory(m_device, paletteOwner.persistentBoneMatsBuffer.memory);

    // Complete resource preparation before command recording. If this phase
    // fails, the caller can safely run the old path for the entire pose.
    for (const uint32_t index : indices) {
        AccelStructHandle& blas = m_blasList[index];
        if (blas.skinningDescSet == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo allocateInfo{};
            allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocateInfo.descriptorPool = m_skinningDescPool;
            allocateInfo.descriptorSetCount = 1;
            allocateInfo.pSetLayouts = &m_skinningDescLayout;
            if (vkAllocateDescriptorSets(
                    m_device, &allocateInfo, &blas.skinningDescSet) != VK_SUCCESS ||
                blas.skinningDescSet == VK_NULL_HANDLE) {
                blas.skinningDescSet = VK_NULL_HANDLE;
                return false;
            }
            ++m_skinningDescSetsLive;
        }

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
        VkAccelerationStructureGeometryKHR geometry{};
        describeGeometry(blas, triangles, geometry);
        const uint32_t primitiveCount = primitiveCountFor(blas);

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = buildFlagsFor(blas);
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        buildInfo.srcAccelerationStructure = blas.accel;
        buildInfo.dstAccelerationStructure = blas.accel;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        fpGetAccelerationStructureBuildSizesKHR(
            m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo, &primitiveCount, &sizeInfo);
        const uint64_t scratchAlignment = m_capabilities.minScratchAlignment > 0
            ? m_capabilities.minScratchAlignment
            : kFallbackScratchAlignment;
        const uint64_t scratchBytes = alignUp(sizeInfo.updateScratchSize,
                                              scratchAlignment);
        if (scratchBytes == 0u) {
            return false;
        }
        if (!blas.skinScratchBuffer.buffer ||
            blas.skinScratchBuffer.size < scratchBytes) {
            if (blas.skinScratchBuffer.buffer) {
                destroyBuffer(blas.skinScratchBuffer);
            }
            BufferCreateInfo scratchInfo{};
            scratchInfo.size = scratchBytes;
            scratchInfo.usage = BufferUsage::STORAGE;
            scratchInfo.location = MemoryLocation::GPU_ONLY;
            scratchInfo.category = VramCategory::Scratch;
            blas.skinScratchBuffer = createBuffer(scratchInfo);
            if (!blas.skinScratchBuffer.buffer ||
                blas.skinScratchBuffer.deviceAddress == 0) {
                return false;
            }
        }

        const uint64_t vertexBytes =
            static_cast<uint64_t>(blas.vertexCount) * sizeof(float) * 3u;
        const uint64_t normalOffset =
            blas.normalBuffer.buffer == blas.vertexBuffer.buffer
            ? blas.normalBuffer.deviceAddress - blas.vertexBuffer.deviceAddress
            : 0u;
        VkDescriptorBufferInfo buffers[7]{};
        buffers[0] = {blas.baseVertexBuffer.buffer, 0, VK_WHOLE_SIZE};
        buffers[1] = {blas.baseNormalBuffer.buffer, 0, VK_WHOLE_SIZE};
        buffers[2] = {blas.boneIndexBuffer.buffer, 0, VK_WHOLE_SIZE};
        buffers[3] = {blas.boneWeightBuffer.buffer, 0, VK_WHOLE_SIZE};
        buffers[4] = {
            paletteOwner.persistentBoneMatsBuffer.buffer, 0, VK_WHOLE_SIZE};
        buffers[5] = {blas.vertexBuffer.buffer, 0, vertexBytes};
        buffers[6] = {blas.normalBuffer.buffer, normalOffset, vertexBytes};

        VkWriteDescriptorSet writes[7]{};
        for (uint32_t binding = 0; binding < 7u; ++binding) {
            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = blas.skinningDescSet;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &buffers[binding];
        }
        vkUpdateDescriptorSets(m_device, 7u, writes, 0u, nullptr);
    }

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    if (commandBuffer == VK_NULL_HANDLE) {
        return false;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_skinningPipeline);
    for (const uint32_t index : indices) {
        const AccelStructHandle& blas = m_blasList[index];
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_skinningPipelineLayout, 0, 1, &blas.skinningDescSet, 0, nullptr);
        const uint32_t parameters[2] = {
            blas.vertexCount, static_cast<uint32_t>(boneMatrices.size())};
        vkCmdPushConstants(commandBuffer, m_skinningPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(parameters), parameters);
        vkCmdDispatch(commandBuffer, (blas.vertexCount + 255u) / 256u, 1u, 1u);
    }

    VkMemoryBarrier skinningBarrier{};
    skinningBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    skinningBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    skinningBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &skinningBarrier, 0, nullptr, 0, nullptr);

    for (const uint32_t index : indices) {
        AccelStructHandle& blas = m_blasList[index];
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
        VkAccelerationStructureGeometryKHR geometry{};
        describeGeometry(blas, triangles, geometry);

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = buildFlagsFor(blas);
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        buildInfo.srcAccelerationStructure = blas.accel;
        buildInfo.dstAccelerationStructure = blas.accel;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;
        buildInfo.scratchData.deviceAddress = blas.skinScratchBuffer.deviceAddress;

        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = primitiveCountFor(blas);
        const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
        fpCmdBuildAccelerationStructuresKHR(
            commandBuffer, 1u, &buildInfo, &rangePointer);
    }

    VkMemoryBarrier refitBarrier{};
    refitBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    refitBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    refitBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &refitBarrier, 0, nullptr, 0, nullptr);

    endSingleTimeCommands(commandBuffer);
    return true;
}

} // namespace VulkanRT
