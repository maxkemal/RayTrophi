#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace VulkanRT {

// Developer-only, process-start A/B control. Normal rendering keeps all stages.
// Read once so the shader hash, SBT layout and render guard use one policy.
inline bool omitVolumePipelineForProbe() {
    static const bool enabled = []() {
        const char* value = std::getenv("RT_VK_AB_NO_VOLUME");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool volumePipelineProbeAllowsScene(uint32_t publishedVolumes, uint32_t tlasSlots) {
    return !omitVolumePipelineForProbe() || (publishedVolumes == 0 && tlasSlots == 0);
}

} // namespace VulkanRT
