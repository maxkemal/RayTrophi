#include "Backend/VulkanViewportBackend.h"

namespace Backend {
bool VulkanViewportBackend::serviceAcousticGpuRasterIdle() {
    if (!acousticGpuPending() || m_interactiveViewport.dirty ||
        rayFusionProbeUpdatesPending() || m_interactiveViewport.presentationPending ||
        !m_rasterFrameRing || !m_rasterFrameRing->isReady()) {
        return false;
    }
    // An idle viewport needs only the acoustic summary, not another image copy.
    // This ring remains covered by the normal raster AS/resource mutation guards.
    m_rasterFrameRing->submitAcousticOnly();
    return true;
}
} // namespace Backend
