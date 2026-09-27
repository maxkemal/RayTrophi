#pragma once

#include "globals.h"
#include "imgui.h"

namespace VulkanRecoveryHUD {

inline void draw(bool cpuRendering, bool vulkanRendering, int samples) {
    if (g_viewport_device_lost_count.load() == 0 && !g_vulkan_device_lost) {
        return;
    }
    const ImVec4 warning(1.0f, 0.72f, 0.28f, 1.0f);
    if (g_vulkan_device_lost) {
        ImGui::TextColored(warning, "Vulkan device lost. Preparing CPU fallback...");
        return;
    }
    if (cpuRendering) {
        ImGui::TextColored(warning, "Vulkan device was lost. Rendering on CPU.");
    }
    if (g_viewport_recovery_given_up.load()) {
        ImGui::TextColored(warning,
            "Vulkan viewport recovery stopped. CPU remains available.");
    } else if (g_viewport_rebuild_pending_after_loss.load()) {
        ImGui::TextColored(warning, "Vulkan viewport recovery pending...");
    } else if (vulkanRendering) {
        ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.6f, 1.0f),
            samples > 0 ? "Vulkan RT resumed: samples received."
                        : "Vulkan RT selected. Waiting for the first sample...");
    } else if (cpuRendering) {
        ImGui::TextColored(warning,
            "To retry RT, select Vulkan in the render backend selector.");
    }
}

} // namespace VulkanRecoveryHUD
