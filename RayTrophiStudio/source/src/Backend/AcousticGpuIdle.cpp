#include "Backend/VulkanBackend.h"
#include "Backend/VulkanFailureReporting.h"

namespace VulkanRT {
bool VulkanDevice::submitAcousticOnly(uint32_t index) {
    if (!acousticPass || !acousticPass->hasPending() || !hasTLAS() ||
        index >= kFrameSlotCount || !ensureFrameSlotsCreated()) {
        return false;
    }
    auto& slot = m_frameSlots[index];
    if (slot.everSubmitted && vkGetFenceStatus(m_device, slot.fence) != VK_SUCCESS) {
        return false; // Never wait for a busy render slot on behalf of audio.
    }
    static bool firstIdle = true;
    const bool trace = firstIdle;
    if (trace) {
        SCENE_LOG_INFO("[AcousticGPU] first RT idle query: preparing ready frame slot");
    }
    acousticPass->collect(slot.fence);
    if (vkResetCommandBuffer(slot.cmd, 0) != VK_SUCCESS) {
        return false;
    }
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot.cmd, &begin) != VK_SUCCESS) {
        return false;
    }
    const bool recorded = acousticPass->record(slot.cmd, slot.fence);
    const bool ended = vkEndCommandBuffer(slot.cmd) == VK_SUCCESS;
    if (!recorded || !ended) {
        acousticPass->submitted(slot.fence, false);
        return false;
    }
    if (trace) {
        SCENE_LOG_INFO("[AcousticGPU] RT idle commands ended; resetting fence");
    }
    if (vkResetFences(m_device, 1, &slot.fence) != VK_SUCCESS) {
        acousticPass->submitted(slot.fence, false);
        return false;
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &slot.cmd;
    if (trace) {
        SCENE_LOG_INFO("[AcousticGPU] RT idle vkQueueSubmit begin");
    }
    const auto result = reportVulkanDeviceFailure(
        vkQueueSubmit(m_computeQueue, 1, &submit, slot.fence), "submitAcousticOnly");
    if (trace) {
        SCENE_LOG_INFO("[AcousticGPU] RT idle vkQueueSubmit result=" +
            std::to_string(static_cast<int>(result)));
        firstIdle = false;
    }
    acousticPass->submitted(slot.fence, result == VK_SUCCESS);
    // A failed submit owns no in-flight work (and cannot leave a dead fence wait).
    slot.everSubmitted = result == VK_SUCCESS;
    slot.acousticOnly = result == VK_SUCCESS;
    return result == VK_SUCCESS;
}
} // namespace VulkanRT

namespace Backend {
void VulkanBackendAdapter::serviceAcousticGpuIdle() {
    if (!m_device || !m_device->acousticPass || !m_device->acousticPass->hasPending()) {
        return;
    }
    const auto slot = m_tonemappedFrameSlot;
    if (m_device->submitAcousticOnly(slot)) {
        // Join the existing AS mutation/drain guards; no new reader outlives them.
        // This slot's existing image remains unchanged: no tonemap or image download.
        m_tonemappedSlotInFlight[slot] = true;
        m_tonemappedFrameSlot = (slot + 1u) % VulkanRT::VulkanDevice::kFrameSlotCount;
    }
}
} // namespace Backend
