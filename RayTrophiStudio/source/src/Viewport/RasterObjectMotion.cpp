#include "Backend/VulkanBackend.h"

namespace Backend {
bool VulkanBackendAdapter::ensureRasterObjectMotionImage(uint32_t width, uint32_t height) {
    auto& image = m_interactiveViewport.objectPreviousClipImage;
    if (image.image && image.width == width && image.height == height) {
        return true;
    }
    destroyRasterObjectMotionImage();
    image = m_device->createImage2D(width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    return image.image != VK_NULL_HANDLE;
}

void VulkanBackendAdapter::destroyRasterObjectMotionImage() {
    auto& image = m_interactiveViewport.objectPreviousClipImage;
    if (image.image) {
        m_device->destroyImage(image);
        image = {};
    }
}
} // namespace Backend
