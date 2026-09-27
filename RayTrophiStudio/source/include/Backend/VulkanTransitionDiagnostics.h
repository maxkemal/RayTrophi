#pragma once

#include "globals.h"
#include <chrono>
#include <string>

namespace VulkanRT {

// Structural operations only; no per-frame or per-particle logging.
class TransitionDiagnostic {
public:
    explicit TransitionDiagnostic(const char* operation, bool enabled = true)
        : operation_(operation), enabled_(enabled),
          started_(std::chrono::steady_clock::now()) {
        if (enabled_) {
            SCENE_LOG_INFO(std::string("[VulkanTransition] begin ") + operation_);
        }
    }

    ~TransitionDiagnostic() {
        if (!enabled_) {
            return;
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started_).count();
        SCENE_LOG_INFO(std::string("[VulkanTransition] end ") + operation_ +
            " cpu_wall_ms=" + std::to_string(ms) +
            " device_loss_pending=" + (g_vulkan_device_lost ? "1" : "0"));
    }

private:
    const char* operation_;
    bool enabled_;
    std::chrono::steady_clock::time_point started_;
};

} // namespace VulkanRT
