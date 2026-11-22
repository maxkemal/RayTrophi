#pragma once

#include <vulkan/vulkan.h>
#include "globals.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace RayTrophiSim {

// Close each record before entering the driver: a fatal driver exception must
// not leave the shader name in a userspace output buffer. Startup only.
inline void traceSimulationPipeline(const char* stage,
                                    const std::string& kernel,
                                    const std::string& path,
                                    const std::vector<uint32_t>& code,
                                    VkResult result = VK_SUCCESS) {
    const uint32_t version = code.size() >= 5 ? code[1] : 0;
    const std::string_view event(stage);
    const bool unsupported = event.find("_not_supported_on_this_GPU") != std::string_view::npos;
    const bool failed = result != VK_SUCCESS && event != "module.missing_or_unreadable";
    bool playerRequired = false;
#ifdef RT_PLAYER
    playerRequired = kernel == "sim_particle_spawn" || kernel == "sim_particle_ballistic" ||
                     kernel == "sim_particle_collide" || kernel == "paint_wet";
#endif
    if (failed || playerRequired) {
        const char* prefix = playerRequired ? "[Player compute] " : "[Simulation compute] ";
        const std::string message = std::string(prefix) + stage +
            " kernel=" + kernel + " shader=" + path +
            " spirv=" + std::to_string((version >> 16) & 0xff) + "." +
            std::to_string((version >> 8) & 0xff) +
            " VkResult=" + std::to_string(static_cast<int>(result));
        if (unsupported) {
            SCENE_LOG_WARN(message);
        } else if (result == VK_SUCCESS) {
            SCENE_LOG_INFO(message);
        } else {
            SCENE_LOG_ERROR(message);
        }
    }
    static const bool enabled = [] {
#ifdef _WIN32
        char* value = nullptr;
        size_t length = 0;
        const auto error = _dupenv_s(&value, &length, "RT_SIM_PIPELINE_TRACE");
        const bool requested = error == 0 && value && value[0] == '1' && value[1] == '\0';
        std::free(value);
        return requested;
#else
        const char* value = std::getenv("RT_SIM_PIPELINE_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
#endif
    }();
    if (!enabled) {
        return;
    }

    FILE* file = nullptr;
#ifdef _WIN32
    fopen_s(&file, "SimulationPipelineTrace.txt", "a");
#else
    file = std::fopen("SimulationPipelineTrace.txt", "a");
#endif
    if (!file) {
        return;
    }

    std::fprintf(file,
        "[SimPipeline] %s kernel=%s spv=%s version=%u.%u bytes=%zu result=%d\n",
        stage, kernel.c_str(), path.c_str(), (version >> 16) & 0xff,
        (version >> 8) & 0xff, code.size() * sizeof(uint32_t),
        static_cast<int>(result));
    std::fclose(file);
}

} // namespace RayTrophiSim
