#include "Viewport/ViewportPipelineCache.h"

#include "Device/SimulationPipelineCache.h"

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <vector>

namespace Backend {

namespace {

struct ViewportCacheState {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    RayTrophiSim::SimulationPipelineCache cache;
};

ViewportCacheState& state() {
    static ViewportCacheState s;
    return s;
}

std::vector<std::string> listSpvFiles(const std::string& shader_dir) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(
             fs::path(std::u8string(shader_dir.begin(), shader_dir.end())), ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const fs::path p = entry.path();
        if (p.extension() == ".spv") {
            const std::u8string name = p.filename().u8string();
            files.emplace_back(name.begin(), name.end());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace

VkPipelineCache acquireViewportPipelineCache(VkDevice device,
                                             VkPhysicalDevice physical_device,
                                             const std::string& shader_dir) {
    if (!device || !physical_device) {
        return VK_NULL_HANDLE;
    }
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.device == device && s.cache.handle() != VK_NULL_HANDLE) {
        return s.cache.handle();
    }
    if (s.device != VK_NULL_HANDLE) {
        s.cache.close();
        s.device = VK_NULL_HANDLE;
    }
    const auto spv_files = listSpvFiles(shader_dir);
    const uint64_t stamp = RayTrophiSim::simulationShaderStamp(shader_dir, spv_files);
    if (s.cache.open(device, physical_device, stamp, "viewport_raster")) {
        s.device = device;
        return s.cache.handle();
    }
    return VK_NULL_HANDLE;
}

VkPipelineCache currentViewportPipelineCache(VkDevice device) {
    if (!device) {
        return VK_NULL_HANDLE;
    }
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.device != device) {
        return VK_NULL_HANDLE;
    }
    return s.cache.handle();
}

void markViewportPipelineCacheDirty(VkDevice device) {
    if (!device) {
        return;
    }
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.device == device && !s.cache.status().loaded) {
        s.cache.markDirty();
    } else if (s.device == device) {
        // Also persist newly specialized or lazily built viewport pipelines.
        s.cache.markDirty();
    }
}

void saveViewportPipelineCache(VkDevice device) {
    if (!device) {
        return;
    }
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.device == device) {
        s.cache.save();
    }
}

void closeViewportPipelineCache(VkDevice device) {
    if (!device) {
        return;
    }
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.device == device) {
        s.cache.close();
        s.device = VK_NULL_HANDLE;
    }
}

} // namespace Backend
