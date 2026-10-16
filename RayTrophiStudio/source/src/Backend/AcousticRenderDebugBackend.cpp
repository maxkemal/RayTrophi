#include "Audio/AcousticRenderDebug.h"
#include "Backend/VulkanBackend.h"

#include <mutex>

namespace Backend {
bool VulkanBackendAdapter::setAcousticRenderSource(const audio::AcousticRenderSource& source) {
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock() || m_acousticRenderSource == source) {
        return false;
    }
    m_acousticRenderSource = source;
    if (m_debugView == audio::kAcousticSurfaceDebugView) {
        // Frame zero overwrites every pixel and AOV. Do not use resetAccumulation:
        // its synchronous image clears would stall the GPU for source motion.
        m_volumeTemporal.invalidate();
        m_currentSamples = 0;
        m_imagesCleared = true;
    }
    return true;
}
} // namespace Backend
