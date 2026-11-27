#pragma once

#include "Backend/VulkanBackend.h"
#include "Matrix4x4.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Backend {

struct RasterSkinningBatchState {
    struct BoundEntry {
        uint64_t bufferGeneration = 0;
        VkBuffer baseVertexBuffer = VK_NULL_HANDLE;
        VkBuffer outVertexBuffer = VK_NULL_HANDLE;
        VkBuffer outNormalBuffer = VK_NULL_HANDLE;
    };

    VulkanRT::BufferHandle sharedBoneMatsBuffer;
    uint64_t sharedBoneMatsCapacity = 0;
    uint64_t bufferGeneration = 0;
    std::unordered_map<VkDescriptorSet, BoundEntry> boundSets;

    void reset(VulkanRT::VulkanDevice* device) {
        if (device && sharedBoneMatsBuffer.buffer) {
            device->destroyBuffer(sharedBoneMatsBuffer);
        }
        sharedBoneMatsBuffer = {};
        sharedBoneMatsCapacity = 0;
        ++bufferGeneration;
        boundSets.clear();
    }
};

// Dispatches GPU compute skinning for all skinned raster meshes in a single
// asynchronous command buffer with inline chunked vkCmdUpdateBuffer bone upload
// and cached descriptor sets. Avoids per-frame drainInteractiveViewportInFlight()
// and per-mesh synchronous fence waits.
//
// Returns false if the batch path cannot be used (e.g. >16384 bones exceeding
// the 1 MB inline upload budget, missing skinning pipeline, or partial mesh
// buffers), leaving the caller to fall back to the legacy path cleanly.
template <typename MeshMap>
inline bool dispatchRasterSkinningBatchAsync(
    VulkanRT::VulkanDevice& device,
    RasterSkinningBatchState& state,
    MeshMap& rasterMeshes,
    const std::vector<Matrix4x4>& boneMatrices,
    const std::function<void()>& drainInFlight,
    uint32_t& outSkinnedMeshCount) {
    using MeshEntry = typename MeshMap::mapped_type;
    outSkinnedMeshCount = 0;

    VkDevice vkDevice = device.getDevice();
    if (!vkDevice || !device.getComputeQueue()) return false;
    if (device.m_skinningPipeline == VK_NULL_HANDLE ||
        device.m_skinningPipelineLayout == VK_NULL_HANDLE ||
        device.m_skinningDescPool == VK_NULL_HANDLE ||
        device.m_skinningDescLayout == VK_NULL_HANDLE) {
        return false;
    }

    constexpr uint64_t kVkUpdateBufferMaxBytes = 65536u;
    constexpr uint64_t kMaxInlineBoneBytes = 1024u * 1024u; // 16384 bones (~150+ characters)
    const uint64_t boneMatSize =
        static_cast<uint64_t>(boneMatrices.size()) * sizeof(Matrix4x4);
    if (boneMatSize == 0u || boneMatSize > kMaxInlineBoneBytes || (boneMatSize & 3u) != 0u) {
        return false;
    }

    std::vector<MeshEntry*> skinnedMeshes;
    skinnedMeshes.reserve(rasterMeshes.size());
    for (auto& [meshKey, mesh] : rasterMeshes) {
        if (!mesh.hasSkinning) continue;
        ++outSkinnedMeshCount;
        if (!mesh.baseVertexBuffer.buffer ||
            !mesh.baseNormalBuffer.buffer ||
            !mesh.boneIndexBuffer.buffer ||
            !mesh.boneWeightBuffer.buffer ||
            !mesh.vertexBuffer.buffer ||
            !mesh.normalBuffer.buffer ||
            mesh.vertexCount == 0u) {
            return false;
        }
        skinnedMeshes.push_back(&mesh);
    }

    if (skinnedMeshes.empty()) {
        return true;
    }

    bool drained = false;
    auto ensureDrained = [&]() {
        if (!drained && drainInFlight) {
            drainInFlight();
            drained = true;
        }
    };

    if (!state.sharedBoneMatsBuffer.buffer || state.sharedBoneMatsCapacity < boneMatSize) {
        ensureDrained();
        if (state.sharedBoneMatsBuffer.buffer) {
            device.destroyBuffer(state.sharedBoneMatsBuffer);
            state.sharedBoneMatsBuffer = {};
            state.sharedBoneMatsCapacity = 0;
        }
        const uint64_t rounded = ((boneMatSize + 4095u) / 4096u) * 4096u;
        const uint64_t allocSize = (std::min)(
            kMaxInlineBoneBytes,
            (std::max)(static_cast<uint64_t>(16384u), rounded));

        VulkanRT::BufferCreateInfo bc{};
        bc.size = allocSize;
        bc.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
        bc.location = VulkanRT::MemoryLocation::GPU_ONLY;
        state.sharedBoneMatsBuffer = device.createBuffer(bc);
        if (!state.sharedBoneMatsBuffer.buffer || !state.sharedBoneMatsBuffer.memory) {
            state.sharedBoneMatsBuffer = {};
            state.sharedBoneMatsCapacity = 0;
            return false;
        }
        state.sharedBoneMatsCapacity = allocSize;
        ++state.bufferGeneration;
        state.boundSets.clear();
    }

    for (auto* meshPtr : skinnedMeshes) {
        auto& mesh = *meshPtr;
        bool freshlyAllocated = false;
        if (mesh.skinningDescSet == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = device.m_skinningDescPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &device.m_skinningDescLayout;
            if (vkAllocateDescriptorSets(vkDevice, &allocInfo, &mesh.skinningDescSet) !=
                    VK_SUCCESS ||
                mesh.skinningDescSet == VK_NULL_HANDLE) {
                mesh.skinningDescSet = VK_NULL_HANDLE;
                return false;
            }
            ++device.m_skinningDescSetsLive;
            freshlyAllocated = true;
        }

        const auto it = state.boundSets.find(mesh.skinningDescSet);
        const bool needsWrite = freshlyAllocated ||
            it == state.boundSets.end() ||
            it->second.bufferGeneration != state.bufferGeneration ||
            it->second.baseVertexBuffer != mesh.baseVertexBuffer.buffer ||
            it->second.outVertexBuffer != mesh.vertexBuffer.buffer ||
            it->second.outNormalBuffer != mesh.normalBuffer.buffer;

        if (needsWrite) {
            if (!freshlyAllocated) {
                // Mutating a descriptor set that an in-flight command buffer may
                // still reference requires draining first.
                ensureDrained();
            }

            const uint64_t outNormalByteOffset =
                (mesh.normalBuffer.buffer == mesh.vertexBuffer.buffer)
                ? (mesh.normalBuffer.deviceAddress - mesh.vertexBuffer.deviceAddress)
                : 0u;
            const uint64_t vertexByteSize =
                static_cast<uint64_t>(mesh.vertexCount) * 12u;

            VkDescriptorBufferInfo bInfo[7]{};
            bInfo[0].buffer = mesh.baseVertexBuffer.buffer;
            bInfo[0].offset = 0;
            bInfo[0].range = VK_WHOLE_SIZE;
            bInfo[1].buffer = mesh.baseNormalBuffer.buffer;
            bInfo[1].offset = 0;
            bInfo[1].range = VK_WHOLE_SIZE;
            bInfo[2].buffer = mesh.boneIndexBuffer.buffer;
            bInfo[2].offset = 0;
            bInfo[2].range = VK_WHOLE_SIZE;
            bInfo[3].buffer = mesh.boneWeightBuffer.buffer;
            bInfo[3].offset = 0;
            bInfo[3].range = VK_WHOLE_SIZE;
            bInfo[4].buffer = state.sharedBoneMatsBuffer.buffer;
            bInfo[4].offset = 0;
            bInfo[4].range = VK_WHOLE_SIZE;
            bInfo[5].buffer = mesh.vertexBuffer.buffer;
            bInfo[5].offset = 0;
            bInfo[5].range = vertexByteSize;
            bInfo[6].buffer = mesh.normalBuffer.buffer;
            bInfo[6].offset = outNormalByteOffset;
            bInfo[6].range = vertexByteSize;

            VkWriteDescriptorSet writes[7]{};
            for (int i = 0; i < 7; ++i) {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = mesh.skinningDescSet;
                writes[i].dstBinding = static_cast<uint32_t>(i);
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &bInfo[i];
            }
            vkUpdateDescriptorSets(vkDevice, 7, writes, 0, nullptr);

            state.boundSets[mesh.skinningDescSet] = {
                state.bufferGeneration,
                mesh.baseVertexBuffer.buffer,
                mesh.vertexBuffer.buffer,
                mesh.normalBuffer.buffer
            };
        }
    }

    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;

    const bool hasRayQuery = device.getCapabilities().supportsRayQuery;

    // Order bone-buffer upload and in-place vertex/normal writes strictly after
    // any previous frame's vertex-input / skinning / BLAS-build reads on this
    // same queue — no host drain required.
    VkPipelineStageFlags preSrcStage =
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkAccessFlags preSrcAccess =
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    if (hasRayQuery) {
        preSrcStage |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        preSrcAccess |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    }

    VkMemoryBarrier preBarrier{};
    preBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    preBarrier.srcAccessMask = preSrcAccess;
    preBarrier.dstAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd,
        preSrcStage,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &preBarrier, 0, nullptr, 0, nullptr);

    const auto* boneBytes = reinterpret_cast<const uint8_t*>(boneMatrices.data());
    for (uint64_t offset = 0u; offset < boneMatSize; offset += kVkUpdateBufferMaxBytes) {
        const uint64_t chunkSize = (std::min)(kVkUpdateBufferMaxBytes, boneMatSize - offset);
        vkCmdUpdateBuffer(
            cmd,
            state.sharedBoneMatsBuffer.buffer,
            offset,
            chunkSize,
            boneBytes + offset);
    }

    VkMemoryBarrier transferBarrier{};
    transferBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    transferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    transferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &transferBarrier, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, device.m_skinningPipeline);
    const uint32_t boneCount = static_cast<uint32_t>(boneMatrices.size());
    for (auto* meshPtr : skinnedMeshes) {
        auto& mesh = *meshPtr;
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            device.m_skinningPipelineLayout,
            0, 1, &mesh.skinningDescSet,
            0, nullptr);
        const uint32_t params[2] = { mesh.vertexCount, boneCount };
        vkCmdPushConstants(
            cmd,
            device.m_skinningPipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, 8, params);
        const uint32_t groupCount = (mesh.vertexCount + 255u) / 256u;
        vkCmdDispatch(cmd, groupCount, 1u, 1u);
    }

    VkPipelineStageFlags postDstStage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    VkAccessFlags postDstAccess = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    if (hasRayQuery) {
        postDstStage |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        postDstAccess |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    }

    VkMemoryBarrier postBarrier{};
    postBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    postBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    postBarrier.dstAccessMask = postDstAccess;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        postDstStage,
        0, 1, &postBarrier, 0, nullptr, 0, nullptr);

    return device.endSingleTimeCommandsAsync(cmd);
}

} // namespace Backend
