#include "Backend/VulkanBackend.h"

#include <cstring>

namespace Backend {

bool VulkanBackendAdapter::uploadSelectionOutlineMatrices(
        const std::vector<float>& matrices) {
    if (!m_device || matrices.empty()) return false;

    auto& buffer = m_interactiveViewport.selectionInstanceBuffer;
    auto& uploaded = m_interactiveViewport.selectionUploadedMatrices;
    const uint64_t byteSize = matrices.size() * sizeof(float);
    // Draw identities and mask tiers are resolved afresh by the caller. Only
    // the matrix bytes are cached: a mesh/selection change with identical
    // transforms is safe, and camera motion does not change these bytes.
    if (buffer.buffer && buffer.size >= byteSize &&
        uploaded.size() == matrices.size() &&
        std::memcmp(uploaded.data(), matrices.data(), byteSize) == 0) {
        return true;
    }

    // Both async Realtime frames and the synchronous Rendered mask read this
    // buffer. Drain before a real host write or replacement, never merely
    // because a camera frame needs to draw the same selected object again.
    drainInteractiveViewportInFlight();
    if (!buffer.buffer || buffer.size < byteSize) {
        VulkanRT::BufferCreateInfo info{};
        info.size = byteSize;
        info.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        info.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
        auto replacement = m_device->createBuffer(info);
        if (!replacement.buffer) return false;
        if (buffer.buffer) m_device->destroyBuffer(buffer);
        buffer = replacement;
    }

    // CPU_TO_GPU requires HOST_VISIBLE | HOST_COHERENT in VulkanDevice.
    // Check mapping explicitly: a failed upload must never mark a cache valid.
    uploaded.clear();
    void* mapped = nullptr;
    if (vkMapMemory(m_device->getDevice(), buffer.memory, 0, byteSize, 0,
                    &mapped) != VK_SUCCESS) {
        return false;
    }
    std::memcpy(mapped, matrices.data(), byteSize);
    vkUnmapMemory(m_device->getDevice(), buffer.memory);
    uploaded = matrices;
    return true;
}

} // namespace Backend
