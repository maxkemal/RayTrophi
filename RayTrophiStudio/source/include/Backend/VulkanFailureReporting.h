#pragma once

#include <vulkan/vulkan.h>
#include "globals.h"

namespace VulkanRT {

// Keep the first observed failure: subsequent calls on the same lost device
// cannot identify the operation which originally failed.
inline VkResult reportVulkanDeviceFailure(VkResult result, const char* operation) {
    if (result == VK_ERROR_DEVICE_LOST && !g_vulkan_device_lost) {
        g_vulkan_device_lost_msg = operation;
        g_vulkan_device_lost = true;
        SCENE_LOG_ERROR(std::string("[Vulkan] Device lost observed at ") + operation);
    }
    return result;
}

} // namespace VulkanRT
