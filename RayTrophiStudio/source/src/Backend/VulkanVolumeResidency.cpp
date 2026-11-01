#include "Backend/VulkanBackend.h"

namespace VulkanRT {

int VulkanDevice::bufferDeviceLocal(const BufferHandle& buffer) const {
    if (buffer.memory == VK_NULL_HANDLE) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(m_vramTrackMutex);
    const auto found = m_vramTracked.find(buffer.memory);
    if (found == m_vramTracked.end()) {
        return -1;
    }
    return found->second.deviceLocal ? 1 : 0;
}

} // namespace VulkanRT
