#include "Backend/VulkanBackend.h"

namespace Backend {
void VulkanBackendAdapter::restoreCurrentSkinning(
    const std::vector<std::shared_ptr<Hittable>>& objects, bool rasterOnly) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (m_currentSkinMatrices.empty()) return;
    if (rasterOnly) {
        syncRasterSkinnedVerticesImpl(objects, m_currentSkinMatrices);
        return;
    }
    // Reuse the normal pose service after the new BLAS/TLAS is ready,
    // including its trace fences and TLAS refit.
    if (!m_vkInstances.empty() && !m_topology_dirty)
        updateSceneGeometry(objects, m_currentSkinMatrices);
}
}
