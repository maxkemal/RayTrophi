#pragma once

#include <vulkan/vulkan.h>
#include "globals.h"

namespace VulkanRT {

// VK_EXT_device_fault: asks every device created with the extension what the
// driver recorded (faulting address + kind, or a vendor timeout description)
// and logs it once. "Observed at" names the call that NOTICED the loss, which
// is almost never the work that caused it; this names the cause.
// Defined in VulkanBackend.cpp.
void logVulkanDeviceFaultInfo();

// Keep the first observed failure: subsequent calls on the same lost device
// cannot identify the operation which originally failed.
inline VkResult reportVulkanDeviceFailure(VkResult result, const char* operation) {
    if (result == VK_ERROR_DEVICE_LOST && !g_vulkan_device_lost) {
        g_vulkan_device_lost_msg = operation;
        g_vulkan_device_lost = true;
        SCENE_LOG_ERROR(std::string("[Vulkan] Device lost observed at ") + operation);
        logVulkanDeviceFaultInfo();
    }
    return result;
}

// ★★★ Fence wait that cannot hang the process. MEASURED 2026-11-17 (cdb on a
//   hung Studio after a GPU page fault): the NVIDIA driver spun forever inside
//   vkWaitForFences(..., UINT64_MAX) on the lost device instead of returning
//   VK_ERROR_DEVICE_LOST -- the app could neither fall back to CPU nor close.
//   A known-lost device does not wait at all; otherwise the wait is chunked so
//   a loss noticed by another thread breaks it, and a fence still unsignalled
//   after 10 s (5x the Windows TDR limit) is treated as a lost device.
inline VkResult waitFenceBounded(VkDevice device, VkFence fence, const char* operation) {
    if (g_vulkan_device_lost) return VK_ERROR_DEVICE_LOST;
    for (int chunk = 0; chunk < 100; ++chunk) {          // 100 x 100 ms
        const VkResult r = vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000ull);
        if (r != VK_TIMEOUT) return reportVulkanDeviceFailure(r, operation);
        if (g_vulkan_device_lost) return VK_ERROR_DEVICE_LOST;
    }
    SCENE_LOG_ERROR(std::string("[Vulkan] fence unsignalled after 10 s at ") + operation +
                    "; treating the device as lost");
    return reportVulkanDeviceFailure(VK_ERROR_DEVICE_LOST, operation);
}

} // namespace VulkanRT
