#pragma once

#include "globals.h"

namespace VulkanRecoveryPolicy {

// Main-thread policy shared by the loop and the existing Python/IPC retry API.
inline unsigned failedAttempts = 0;

inline void resetAttempts() {
    failedAttempts = 0;
}

inline void recordFailedAttempt() {
    ++failedAttempts;
    SCENE_LOG_WARN("[VulkanRecovery] viewport initialization failed, attempt=" +
        std::to_string(failedAttempts));
    if (failedAttempts >= 3) {
        g_viewport_recovery_given_up.store(true);
        g_viewport_rebuild_pending_after_loss.store(false);
        SCENE_LOG_ERROR("[VulkanRecovery] stopped after three failed initializations; "
            "CPU remains available. Use viewport.retry_device_recovery to retry.");
    }
}

} // namespace VulkanRecoveryPolicy
