#include "Backend/VulkanVolumeInstrumentation.h"

#include "Backend/VulkanBackend.h"

#include <cstring>

namespace VulkanRT {

struct VulkanVolumeInstrumentation::Impl {
    BufferHandle buffer;
};

VulkanVolumeInstrumentation::VulkanVolumeInstrumentation()
    : m_impl(std::make_unique<Impl>()) {}

VulkanVolumeInstrumentation::~VulkanVolumeInstrumentation() = default;

bool VulkanVolumeInstrumentation::ensure(VulkanDevice& device) {
    if (m_impl->buffer.buffer) {
        return true;
    }

    VolumePerformanceStats initial{};
    initial.reserved[2] = 0u; // counters are opt-in; normal rendering stays uncontaminated

    BufferCreateInfo info{};
    info.size = sizeof(VolumePerformanceStats);
    info.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
    info.location = MemoryLocation::CPU_TO_GPU;
    info.initialData = &initial;
    m_impl->buffer = device.createBuffer(info);
    return m_impl->buffer.buffer != VK_NULL_HANDLE;
}

void VulkanVolumeInstrumentation::destroy(VulkanDevice& device) {
    if (m_impl->buffer.buffer) {
        device.destroyBuffer(m_impl->buffer);
    }
    m_impl->buffer = {};
}

void VulkanVolumeInstrumentation::reset(VulkanDevice& device, bool enabled) {
    if (!ensure(device)) {
        return;
    }
    VolumePerformanceStats cleared{};
    cleared.reserved[2] = enabled ? 1u : 0u;
    if (void* mapped = device.mapBuffer(m_impl->buffer)) {
        std::memcpy(mapped, &cleared, sizeof(cleared));
        device.unmapBuffer(m_impl->buffer);
    }
}

VolumePerformanceStats VulkanVolumeInstrumentation::read(VulkanDevice& device) const {
    VolumePerformanceStats stats{};
    if (m_impl->buffer.buffer) {
        device.downloadBuffer(m_impl->buffer, &stats, sizeof(stats));
    }
    return stats;
}

const BufferHandle* VulkanVolumeInstrumentation::buffer() const {
    return &m_impl->buffer;
}

} // namespace VulkanRT
