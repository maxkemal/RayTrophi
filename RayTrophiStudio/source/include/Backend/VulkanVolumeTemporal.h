#pragma once

#include <cstdint>
#include <memory>

namespace VulkanRT {

class VulkanDevice;
struct ImageHandle;

// Owns the ping-pong images used by volumetric temporal reprojection.
// Descriptor binding and shader policy deliberately stay outside this class.
class VulkanVolumeTemporal {
public:
    VulkanVolumeTemporal();
    ~VulkanVolumeTemporal();

    VulkanVolumeTemporal(const VulkanVolumeTemporal&) = delete;
    VulkanVolumeTemporal& operator=(const VulkanVolumeTemporal&) = delete;

    bool ensure(VulkanDevice& device, uint32_t width, uint32_t height);
    void destroy(VulkanDevice& device);
    void invalidate();
    void advance();

    bool hasValidHistory() const;
    const ImageHandle* previousColor() const;
    const ImageHandle* previousMetadata() const;
    const ImageHandle* currentColor() const;
    const ImageHandle* currentMetadata() const;
    const ImageHandle* colorSlot(uint32_t index) const;
    const ImageHandle* metadataSlot(uint32_t index) const;
    uint32_t writeIndex() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace VulkanRT
