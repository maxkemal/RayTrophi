#include "Backend/VulkanVolumeTemporal.h"

#include "Backend/VulkanBackend.h"

namespace VulkanRT {

struct VulkanVolumeTemporal::Impl {
    ImageHandle color[2];
    ImageHandle metadata[2];
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t readIndex = 0;
    bool valid = false;
};

VulkanVolumeTemporal::VulkanVolumeTemporal()
    : m_impl(std::make_unique<Impl>()) {}

VulkanVolumeTemporal::~VulkanVolumeTemporal() = default;

bool VulkanVolumeTemporal::ensure(VulkanDevice& device, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        return false;
    }
    if (m_impl->width == width && m_impl->height == height &&
        m_impl->color[0].image && m_impl->color[1].image &&
        m_impl->metadata[0].image && m_impl->metadata[1].image) {
        return true;
    }

    destroy(device);

    constexpr VkImageUsageFlags usage =
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    for (uint32_t i = 0; i < 2; ++i) {
        m_impl->color[i] = device.createImage2D(
            width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_impl->metadata[i] = device.createImage2D(
            width, height, VK_FORMAT_R32G32B32A32_SFLOAT, usage);
    }

    const bool allocated =
        m_impl->color[0].image && m_impl->color[1].image &&
        m_impl->metadata[0].image && m_impl->metadata[1].image;
    if (!allocated) {
        destroy(device);
        return false;
    }

    m_impl->width = width;
    m_impl->height = height;
    m_impl->readIndex = 0;
    m_impl->valid = false;
    device.clearImages({
        { &m_impl->color[0], 0.0f, 0.0f, 0.0f, 0.0f },
        { &m_impl->color[1], 0.0f, 0.0f, 0.0f, 0.0f },
        { &m_impl->metadata[0], 0.0f, 0.0f, 0.0f, 0.0f },
        { &m_impl->metadata[1], 0.0f, 0.0f, 0.0f, 0.0f }
    });
    return true;
}

void VulkanVolumeTemporal::destroy(VulkanDevice& device) {
    for (uint32_t i = 0; i < 2; ++i) {
        if (m_impl->color[i].image) {
            device.destroyImage(m_impl->color[i]);
        }
        if (m_impl->metadata[i].image) {
            device.destroyImage(m_impl->metadata[i]);
        }
        m_impl->color[i] = {};
        m_impl->metadata[i] = {};
    }
    m_impl->width = 0;
    m_impl->height = 0;
    m_impl->readIndex = 0;
    m_impl->valid = false;
}

void VulkanVolumeTemporal::invalidate() {
    m_impl->valid = false;
}

void VulkanVolumeTemporal::advance() {
    m_impl->readIndex ^= 1u;
    m_impl->valid = true;
}

bool VulkanVolumeTemporal::hasValidHistory() const {
    return m_impl->valid;
}

const ImageHandle* VulkanVolumeTemporal::previousColor() const {
    return &m_impl->color[m_impl->readIndex];
}

const ImageHandle* VulkanVolumeTemporal::previousMetadata() const {
    return &m_impl->metadata[m_impl->readIndex];
}

const ImageHandle* VulkanVolumeTemporal::currentColor() const {
    return &m_impl->color[m_impl->readIndex ^ 1u];
}

const ImageHandle* VulkanVolumeTemporal::currentMetadata() const {
    return &m_impl->metadata[m_impl->readIndex ^ 1u];
}

const ImageHandle* VulkanVolumeTemporal::colorSlot(uint32_t index) const {
    return &m_impl->color[index & 1u];
}

const ImageHandle* VulkanVolumeTemporal::metadataSlot(uint32_t index) const {
    return &m_impl->metadata[index & 1u];
}

uint32_t VulkanVolumeTemporal::writeIndex() const {
    return m_impl->readIndex ^ 1u;
}

} // namespace VulkanRT
