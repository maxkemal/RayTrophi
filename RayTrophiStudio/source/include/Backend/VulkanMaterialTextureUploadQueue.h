#pragma once

#include "VulkanMaterialTextureArrayImage.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <memory>
#include <utility>

namespace Backend::MaterialTexturePages {

struct UploadQueueStatus {
    std::size_t pendingUploads = 0;
    uint64_t completedUploadSerial = 0;
    uint64_t notReadyRejections = 0;
};

struct LayerUpload {
    VkBuffer staging = VK_NULL_HANDLE;
    const ArrayImage* image = nullptr;
    uint32_t layer = 0;
    std::vector<VkDeviceSize> mipOffsets;
    VkDeviceSize texelBlockBytes = 0;
};

// Render-thread only. Use the same externally synchronized queue as the draws
// that will consume the material snapshot. Pool access is also thread-confined.
class UploadQueue {
public:
    UploadQueue() = default;
    ~UploadQueue() {
        if (m_pool) {
            std::fputs("[Vulkan] Material texture upload queue destroyed before shutdown\n",
                       stderr);
        }
        assert(!m_pool && "Destroy uploads after stopping the Vulkan queue");
    }
    UploadQueue(const UploadQueue&) = delete;
    UploadQueue& operator=(const UploadQueue&) = delete;

    VkResult initialize(VkDevice device, VkQueue queue, uint32_t family) {
        if (m_device || !device || !queue) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        VkCommandPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        info.queueFamilyIndex = family;
        VkCommandPool pool = VK_NULL_HANDLE;
        const VkResult result = vkCreateCommandPool(device, &info, nullptr, &pool);
        if (result == VK_SUCCESS) {
            m_device = device;
            m_queue = queue;
            m_pool = pool;
        }
        return result;
    }

    VkResult begin(VkCommandBuffer& output) {
        output = VK_NULL_HANDLE;
        if (!m_pool || m_recording) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        const VkResult status = poll();
        if (status != VK_SUCCESS) {
            return status;
        }
        if (m_count == m_pending.size()) {
            ++m_notReadyRejections;
            return VK_NOT_READY;
        }

        VkCommandBufferAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool = m_pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkResult result = vkAllocateCommandBuffers(m_device, &allocation, &m_recording);
        if (result != VK_SUCCESS) {
            m_recording = VK_NULL_HANDLE;
            return result;
        }
        VkCommandBufferBeginInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(m_recording, &info);
        if (result != VK_SUCCESS) {
            discardRecording();
            return result;
        }
        output = m_recording;
        return VK_SUCCESS;
    }

    // A lease owns every staging buffer and destination image used by the recorded
    // commands. Its deleter must run on the render thread. No CPU or queue wait.
    VkResult submit(std::shared_ptr<const void> lease, uint64_t& uploadSerial) {
        uploadSerial = 0;
        if (!m_recording || !lease || m_lastSubmitted == UINT64_MAX) {
            discardRecording();
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        VkResult result = vkEndCommandBuffer(m_recording);
        if (result != VK_SUCCESS) {
            discardRecording();
            return result;
        }

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        result = vkCreateFence(m_device, &fenceInfo, nullptr, &fence);
        if (result != VK_SUCCESS) {
            discardRecording();
            return result;
        }
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_recording;
        result = vkQueueSubmit(m_queue, 1, &submitInfo, fence);
        if (result != VK_SUCCESS) {
            vkDestroyFence(m_device, fence, nullptr);
            discardRecording();
            return result;
        }
        uploadSerial = ++m_lastSubmitted;
        m_pending[(m_front + m_count) % m_pending.size()] =
            {m_recording, fence, uploadSerial, std::move(lease)};
        ++m_count;
        m_recording = VK_NULL_HANDLE;
        return VK_SUCCESS;
    }

    // Record into a batch opened with begin(), after any image initialization.
    // Each copy finishes with a transfer-write to shader-read layout barrier.
    VkResult submitLayers(VkCommandBuffer command, const std::vector<LayerUpload>& layers,
                          VkPipelineStageFlags samplingStages,
                          std::shared_ptr<const void> lease, uint64_t& uploadSerial) {
        uploadSerial = 0;
        if (!command || command != m_recording) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        for (const auto& layer : layers) {
            if (!layer.image ||
                !recordArrayLayerUpload(command, layer.staging, *layer.image, layer.layer,
                                        layer.mipOffsets, layer.texelBlockBytes, samplingStages)) {
                discardRecording();
                return VK_ERROR_INITIALIZATION_FAILED;
            }
        }
        return submit(std::move(lease), uploadSerial);
    }

    VkResult poll() {
        while (m_count) {
            Pending& batch = m_pending[m_front];
            const VkResult status = vkGetFenceStatus(m_device, batch.fence);
            if (status == VK_NOT_READY) {
                break;
            }
            if (status != VK_SUCCESS) {
                return status;
            }
            m_completed = batch.serial;
            releaseFront();
        }
        return VK_SUCCESS;
    }

    uint64_t completedUploadSerial() const {
        return m_completed;
    }

    UploadQueueStatus status() const {
        return {m_count, m_completed, m_notReadyRejections};
    }

    void discardRecording() {
        if (m_recording) {
            vkFreeCommandBuffers(m_device, m_pool, 1, &m_recording);
            m_recording = VK_NULL_HANDLE;
        }
    }

    // The owner calls this after quiescing the queue, or during device-loss
    // teardown. Upload serials certify transfers only, not frame-reader fences.
    void destroyAfterQueueStopped() {
        discardRecording();
        while (m_count) {
            releaseFront();
        }
        if (m_pool) {
            vkDestroyCommandPool(m_device, m_pool, nullptr);
            m_pool = VK_NULL_HANDLE;
        }
        m_device = VK_NULL_HANDLE;
        m_queue = VK_NULL_HANDLE;
        m_front = 0;
        m_lastSubmitted = 0;
        m_completed = 0;
        m_notReadyRejections = 0;
    }

private:
    struct Pending {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        uint64_t serial = 0;
        std::shared_ptr<const void> lease;
    };

    void releaseFront() {
        Pending& batch = m_pending[m_front];
        vkDestroyFence(m_device, batch.fence, nullptr);
        vkFreeCommandBuffers(m_device, m_pool, 1, &batch.command);
        batch = {};
        m_front = (m_front + 1) % m_pending.size();
        --m_count;
    }

    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    VkCommandBuffer m_recording = VK_NULL_HANDLE;
    std::array<Pending, 8> m_pending{};
    std::size_t m_front = 0;
    std::size_t m_count = 0;
    uint64_t m_lastSubmitted = 0;
    uint64_t m_completed = 0;
    uint64_t m_notReadyRejections = 0;
};

} // namespace Backend::MaterialTexturePages
