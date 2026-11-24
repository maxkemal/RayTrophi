#pragma once

#include "SimulationCompute.h"

#include <vulkan/vulkan.h>

namespace RayTrophiSim {

template <class Buffers>
bool resolveIndirectDispatchBuffer(const ComputeDispatch& command, const Buffers& buffers,
                                   VkBuffer& buffer) {
    if (!command.indirect_groups.valid()) {
        return true;
    }

    const auto allocation = buffers.find(command.indirect_groups.id);
    if (command.indirect_groups.backend != ComputeBackendType::VulkanCompute ||
        allocation == buffers.end()) {
        return false;
    }

    const auto& storage = allocation->second;
    const std::size_t offset = command.indirect_offset;
    if (!storage.buffer || !hasComputeBufferUsage(storage.usage, ComputeBufferUsage::Indirect) ||
        offset % sizeof(uint32_t) != 0 || offset > storage.size ||
        sizeof(VkDispatchIndirectCommand) > storage.size - offset) {
        return false;
    }

    buffer = storage.buffer;
    return true;
}

inline void recordIndirectDispatch(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = offset;
    barrier.size = sizeof(VkDispatchIndirectCommand);
    vkCmdPipelineBarrier(commands,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    vkCmdDispatchIndirect(commands, buffer, offset);

    // A later dispatch or upload may overwrite the counts, even when this
    // indirect command launches zero workgroups.
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 0, nullptr);
}

} // namespace RayTrophiSim
