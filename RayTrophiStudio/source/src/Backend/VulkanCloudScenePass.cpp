#include "Backend/VulkanBackend.h"

#include <algorithm>

namespace Backend {

void VulkanBackendAdapter::recordCloudScenePass(VkCommandBuffer cmd, uint32_t width,
                                               uint32_t height, float depthA,
                                               float depthB, float depthC) {
    if (!m_device || !m_cloudRtRendering || !m_cloudRasterWritten ||
        m_camera.orthographic || m_viewportMode != ViewportMode::MaterialPreview) {
        return;
    }
    m_device->recordCloudScene(cmd, m_interactiveViewport.hdrColorImage.view,
                              m_interactiveViewport.depthImage.view,
                              m_interactiveViewport.postSampler, width, height,
                              depthA, depthB, depthC,
                              static_cast<uint32_t>((std::clamp)(
                                  m_cachedClouds.quality.realtime_steps, 16, 512)));
}

} // namespace Backend
