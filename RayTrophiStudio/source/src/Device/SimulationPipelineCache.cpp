#include "Device/SimulationPipelineCache.h"

#include "Platform/Platform.h"
#include "globals.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <system_error>

namespace RayTrophiSim {

namespace {

struct CacheHeaderV1 {
    uint32_t headerSize;
    uint32_t headerVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint8_t uuid[VK_UUID_SIZE];
};
static_assert(sizeof(CacheHeaderV1) == 32, "pipeline cache header is 32 bytes");

bool envEquals(const char* name, const char* expected) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    const auto err = _dupenv_s(&value, &length, name);
    const bool match = (err == 0 && value && std::strcmp(value, expected) == 0);
    std::free(value);
    return match;
#else
    const char* value = std::getenv(name);
    return value && std::strcmp(value, expected) == 0;
#endif
}

std::string pathToUtf8(const std::filesystem::path& p) {
    const std::u8string u8 = p.u8string();
    return std::string(u8.begin(), u8.end());
}

std::filesystem::path utf8ToPath(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

std::filesystem::path cacheDirectory() {
    namespace fs = std::filesystem;
#ifdef __ANDROID__
    const fs::path root = Platform::appDataRoot();
    if (!root.empty()) {
        return root / "vk_pipeline_cache";
    }
#elif defined(_WIN32)
    if (const wchar_t* local = _wgetenv(L"LOCALAPPDATA"); local && *local) {
        return fs::path(local) / L"RayTrophiStudio" / L"vk_pipeline_cache";
    }
    const fs::path root = Platform::appDataRoot();
    if (!root.empty()) {
        return root / "vk_pipeline_cache";
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        return fs::path(xdg) / "RayTrophiStudio" / "vk_pipeline_cache";
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        return fs::path(home) / ".cache" / "RayTrophiStudio" / "vk_pipeline_cache";
    }
    const fs::path root = Platform::appDataRoot();
    if (!root.empty()) {
        return root / "vk_pipeline_cache";
    }
#endif
    return {};
}

std::filesystem::path stampPathFor(const std::filesystem::path& file) {
    std::filesystem::path stamp = file;
    stamp += ".shaders";
    return stamp;
}

std::string validateBlob(const std::vector<char>& blob,
                         const VkPhysicalDeviceProperties& props) {
    if (blob.size() < sizeof(CacheHeaderV1)) {
        return "file shorter than the cache header";
    }
    CacheHeaderV1 h{};
    std::memcpy(&h, blob.data(), sizeof(h));
    if (h.headerSize < sizeof(CacheHeaderV1) || h.headerSize > blob.size()) {
        return "corrupt header size " + std::to_string(h.headerSize);
    }
    if (h.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE) {
        return "unknown header version " + std::to_string(h.headerVersion);
    }
    if (h.vendorID != props.vendorID || h.deviceID != props.deviceID) {
        return "written by another GPU";
    }
    if (std::memcmp(h.uuid, props.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
        return "written by another driver version";
    }
    return {};
}

void fnvMix(uint64_t& h, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= 1099511628211ull;
    }
}

} // namespace

uint64_t simulationShaderStamp(const std::string& shader_dir,
                               const std::vector<std::string>& spv_files) {
    namespace fs = std::filesystem;
    uint64_t hash = 14695981039346656037ull;
    const fs::path base = utf8ToPath(shader_dir);
    for (const std::string& name : spv_files) {
        fnvMix(hash, name.data(), name.size());
        const uint8_t sep = 0xffu;
        fnvMix(hash, &sep, 1);
        std::error_code ec;
        const fs::path full = base / utf8ToPath(name);
        const uint64_t file_size = fs::file_size(full, ec);
        if (!ec) {
            fnvMix(hash, &file_size, sizeof(file_size));
            const auto mtime = fs::last_write_time(full, ec);
            if (!ec) {
                const int64_t ticks = mtime.time_since_epoch().count();
                fnvMix(hash, &ticks, sizeof(ticks));
            }
        }
    }
    return hash;
}

SimulationPipelineCache::~SimulationPipelineCache() {
    close();
}

bool SimulationPipelineCache::open(VkDevice device,
                                   VkPhysicalDevice physical_device,
                                   uint64_t shader_stamp,
                                   const char* cache_prefix) {
    close();
    m_device = device;
    m_stamp = shader_stamp;
    m_dirty = false;
    m_status = {};

    if (!m_device || !physical_device) {
        return false;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device, &props);

    std::vector<char> blob;
    namespace fs = std::filesystem;
    const char* prefix = (cache_prefix && *cache_prefix) ? cache_prefix : "sim_compute";

    if (envEquals("RT_VK_PIPELINE_CACHE", "0")) {
        SCENE_LOG_INFO(std::string("[PipelineCache:") + prefix +
                       "] RT_VK_PIPELINE_CACHE=0: disk cache disabled.");
    } else {
        const fs::path dir = cacheDirectory();
        if (dir.empty()) {
            SCENE_LOG_WARN(std::string("[PipelineCache:") + prefix +
                           "] No cache folder resolved; disk cache disabled.");
        } else {
            char name[96];
            std::snprintf(name, sizeof(name), "%s_%04x_%04x.bin",
                          prefix, props.vendorID, props.deviceID);
            const fs::path file = dir / name;
            m_status.enabled = true;
            m_status.path = pathToUtf8(file);

            uint64_t saved_stamp = 0;
            {
                std::ifstream stamp_in(stampPathFor(file));
                if (stamp_in) {
                    stamp_in >> std::hex >> saved_stamp;
                }
            }
            std::error_code ec;
            if (saved_stamp == shader_stamp && shader_stamp != 0) {
                std::ifstream in(file, std::ios::binary);
                if (in) {
                    blob.assign(std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>());
                }
                if (!blob.empty()) {
                    m_status.reject_reason = validateBlob(blob, props);
                }
                if (!m_status.reject_reason.empty()) {
                    SCENE_LOG_WARN("[SimComputeCache] Discarded (" + m_status.reject_reason +
                                   "): " + m_status.path);
                    blob.clear();
                }
            } else if (fs::exists(file, ec)) {
                m_status.reject_reason = "shader stamp mismatch";
                SCENE_LOG_INFO("[SimComputeCache] Shaders changed; starting empty cache.");
            }
            if (blob.empty() && fs::exists(file, ec)) {
                fs::remove(file, ec);
                fs::remove(stampPathFor(file), ec);
            }
        }
    }

    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = blob.size();
    ci.pInitialData = blob.empty() ? nullptr : blob.data();
    VkResult r = vkCreatePipelineCache(m_device, &ci, nullptr, &m_cache);
    if (r != VK_SUCCESS && !blob.empty()) {
        m_status.reject_reason = "driver rejected blob (" + std::to_string(static_cast<int>(r)) + ")";
        SCENE_LOG_WARN("[SimComputeCache] Driver rejected cached blob; starting empty.");
        blob.clear();
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        r = vkCreatePipelineCache(m_device, &ci, nullptr, &m_cache);
    }
    if (r != VK_SUCCESS) {
        m_cache = VK_NULL_HANDLE;
        SCENE_LOG_WARN("[SimComputeCache] vkCreatePipelineCache failed (" +
                       std::to_string(static_cast<int>(r)) + "); running uncached.");
        return false;
    }

    m_status.loaded = !blob.empty();
    m_status.loaded_bytes = blob.size();
    if (m_status.loaded) {
        SCENE_LOG_INFO("[SimComputeCache] Loaded (" +
                       std::to_string(blob.size() / 1024) + " KB): " + m_status.path);
    }
    return true;
}

void SimulationPipelineCache::save() {
    if (!m_device || m_cache == VK_NULL_HANDLE || !m_dirty ||
        !m_status.enabled || m_status.path.empty()) {
        return;
    }

    size_t size = 0;
    if (vkGetPipelineCacheData(m_device, m_cache, &size, nullptr) != VK_SUCCESS || size == 0) {
        return;
    }
    std::vector<char> data(size);
    if (vkGetPipelineCacheData(m_device, m_cache, &size, data.data()) != VK_SUCCESS || size == 0) {
        return;
    }
    data.resize(size);

    namespace fs = std::filesystem;
    const fs::path file = utf8ToPath(m_status.path);
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    if (ec) {
        SCENE_LOG_WARN("[SimComputeCache] Could not create cache directory: " + ec.message());
        return;
    }

    fs::path tmp = file;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            SCENE_LOG_WARN("[SimComputeCache] Could not open temp cache file for writing.");
            return;
        }
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!out.good()) {
            SCENE_LOG_WARN("[SimComputeCache] Failed writing pipeline cache blob.");
            fs::remove(tmp, ec);
            return;
        }
    }

    fs::rename(tmp, file, ec);
    if (ec) {
        // Windows rename fails if destination exists.
        std::error_code rm_ec;
        fs::remove(file, rm_ec);
        ec.clear();
        fs::rename(tmp, file, ec);
        if (ec) {
            fs::remove(tmp, rm_ec);
            SCENE_LOG_WARN("[SimComputeCache] Could not replace cache file: " + ec.message());
            return;
        }
    }

    {
        std::ofstream stamp_out(stampPathFor(file), std::ios::trunc);
        if (stamp_out) {
            stamp_out << std::hex << m_stamp << "\n";
        }
    }

    m_dirty = false;
    m_status.saved_bytes = data.size();
    SCENE_LOG_INFO("[SimComputeCache] Saved (" + std::to_string(data.size() / 1024) +
                   " KB): " + m_status.path);
}

void SimulationPipelineCache::close() {
    if (m_cache != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        save();
        vkDestroyPipelineCache(m_device, m_cache, nullptr);
    }
    m_cache = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_dirty = false;
}

} // namespace RayTrophiSim
