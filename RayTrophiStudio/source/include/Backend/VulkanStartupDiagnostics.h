#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <string_view>

#if defined(__ANDROID__) && defined(RT_PLAYER)
#include <sys/system_properties.h>
#endif

namespace Backend {

#if defined(__ANDROID__) && defined(RT_PLAYER)
class VulkanStartupDiagnostics;
inline thread_local VulkanStartupDiagnostics* currentVulkanStartup = nullptr;

class VulkanStartupDiagnostics {
public:
    explicit VulkanStartupDiagnostics(VkPhysicalDevice gpu = VK_NULL_HANDLE)
        : previous_(currentVulkanStartup), physicalDevice_(gpu) {
        currentVulkanStartup = this;
    }

    ~VulkanStartupDiagnostics() { currentVulkanStartup = previous_; }
    VulkanStartupDiagnostics(const VulkanStartupDiagnostics&) = delete;
    VulkanStartupDiagnostics& operator=(const VulkanStartupDiagnostics&) = delete;

    void setGpu(const VkPhysicalDeviceProperties& props) { gpu_ = props; }
    void clearFailure() { failure_.clear(); }

    void record(std::string_view step, std::string_view detail) {
        if (failure_.empty()) {
            failure_ = step;
            if (!detail.empty()) {
                failure_ += " — ";
                failure_ += detail;
            }
        }
    }

    std::string message(std::string_view fallback) const {
        std::string text = failure_.empty()
            ? std::string(fallback) + " (no VkResult returned)"
            : failure_;
        VkPhysicalDeviceProperties properties = gpu_;
        if (properties.deviceName[0] == '\0' && physicalDevice_ != VK_NULL_HANDLE) {
            vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
        }
        if (properties.deviceName[0] != '\0') {
            text += "\nGPU: ";
            text += properties.deviceName;
            // Vulkan leaves the driver version encoding to the vendor.
            text += " | Driver version: " + std::to_string(properties.driverVersion);
        }
        return text;
    }

private:
    VulkanStartupDiagnostics* previous_;
    VkPhysicalDevice physicalDevice_;
    VkPhysicalDeviceProperties gpu_{};
    std::string failure_;
};
#endif

inline void recordVulkanStartupFailure(std::string_view step, std::string_view detail = {}) {
#if defined(__ANDROID__) && defined(RT_PLAYER)
    if (currentVulkanStartup) {
        currentVulkanStartup->record(step, detail);
    }
#endif
}

inline void clearVulkanStartupFailure() {
#if defined(__ANDROID__) && defined(RT_PLAYER)
    if (currentVulkanStartup) {
        currentVulkanStartup->clearFailure();
    }
#endif
}

inline VkResult vulkanStartupResult(std::string_view step, VkResult result,
                                    std::string_view detail = {}) {
#if defined(__ANDROID__) && defined(RT_PLAYER)
    if (result != VK_SUCCESS && currentVulkanStartup) {
        std::string reason = "VkResult " + std::to_string(static_cast<int>(result));
        if (!detail.empty()) {
            reason += " — ";
            reason += detail;
        }
        currentVulkanStartup->record(step, reason);
    }
#endif
    return result;
}

inline bool failViewportStartupForDiagnostics() {
#if defined(__ANDROID__) && defined(RT_PLAYER)
    char value[PROP_VALUE_MAX]{};
    return __system_property_get("debug.raytrophi.fail_viewport_startup", value) == 1 &&
           value[0] == '1';
#else
    return false;
#endif
}

} // namespace Backend
