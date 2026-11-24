#pragma once

#include "Backend/VulkanBackend.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

namespace Backend {

// Owned by one raster frame slot. Write and grow only after its reader fence
// has completed. The synchronous raster path supplies its own instance.
class ParticleBillboardFrameBuffer {
public:
    VkBuffer write(VulkanRT::VulkanDevice& device,
                   const std::vector<ParticleBillboardVertex>& alpha,
                   const std::vector<ParticleBillboardVertex>& additive) {
        constexpr auto max_count = (std::numeric_limits<uint32_t>::max)();
        if ((alpha.empty() && additive.empty()) ||
            alpha.size() > max_count || additive.size() > max_count) {
            return VK_NULL_HANDLE;
        }

        const VkDeviceSize alpha_bytes = byteSize(alpha.size());
        const VkDeviceSize bytes = alpha_bytes + byteSize(additive.size());
        if (buffer_.size < bytes) {
            const auto now = std::chrono::steady_clock::now();
            if (now < retry_after_) {
                return VK_NULL_HANDLE;
            }
            VulkanRT::BufferCreateInfo desc;
            desc.size = (std::max)(bytes, buffer_.size + buffer_.size / 2);
            desc.usage = VulkanRT::BufferUsage::VERTEX;
            // CPU_TO_GPU requires HOST_VISIBLE | HOST_COHERENT. The subsequent
            // queue submission makes these host writes available to vertex input.
            desc.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
            auto replacement = device.createBuffer(desc);
            void* mapped = device.mapBuffer(replacement);
            if (!mapped) {
                device.destroyBuffer(replacement);
                retry_after_ = now + std::chrono::seconds(1);
                return VK_NULL_HANDLE;
            }
            release(device);
            buffer_ = replacement;
            mapped_ = mapped;
        }

        if (!alpha.empty()) {
            std::memcpy(mapped_, alpha.data(), alpha_bytes);
        }
        if (!additive.empty()) {
            std::memcpy(static_cast<char*>(mapped_) + alpha_bytes,
                        additive.data(), byteSize(additive.size()));
        }
        return buffer_.buffer;
    }

    void release(VulkanRT::VulkanDevice& device) {
        if (mapped_) {
            device.unmapBuffer(buffer_);
            mapped_ = nullptr;
        }
        device.destroyBuffer(buffer_);
        retry_after_ = {};
    }

    static constexpr VkDeviceSize byteSize(std::size_t count) {
        return static_cast<VkDeviceSize>(count) * sizeof(ParticleBillboardVertex);
    }

private:
    VulkanRT::BufferHandle buffer_;
    void* mapped_ = nullptr;
    std::chrono::steady_clock::time_point retry_after_{};
};

} // namespace Backend
