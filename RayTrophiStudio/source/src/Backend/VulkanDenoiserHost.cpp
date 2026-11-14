// Player has no CUDA/OIDN interop. Native Vulkan rendering remains unchanged.
#include "Backend/VulkanBackend.h"

namespace Backend {
void VulkanBackendAdapter::destroyGpuDenoiserInterop() {
    m_gpuDenoiserInterop = nullptr;
}

bool VulkanBackendAdapter::ensureGpuDenoiserInterop(int, int, bool) {
    return false;
}

bool VulkanBackendAdapter::getDenoiserFrameGPU(DenoiserFrameDataGPU& frame, bool) {
    frame = DenoiserFrameDataGPU{};
    return false;
}
} // namespace Backend
