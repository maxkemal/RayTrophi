/*
 * VulkanDevice ray tracing pipeline build: asynchronous compile, deferred
 * host operations, and a persistent VkPipelineCache.
 *
 * ★★★ Why this file exists (reported 2026-10-05): the RT pipeline used to be
 * created with one synchronous vkCreateRayTracingPipelinesKHR call on the render
 * thread, without a cache. The first compile after a shader change takes
 * minutes (one core, volume_closesthit alone is ~4800 lines), so switching the
 * viewport to Rendered froze the app with no indication why.
 *
 *   render thread   requestRTPipelineBuild -> shader hash, layouts, start worker
 *   worker thread   modules -> create (deferred op, joined by N threads) -> cache save
 *   render thread   pollRTPipelineBuild -> install: m_rtPipeline + SBT
 *
 * ★ The worker never writes a VulkanDevice member that the render thread reads.
 * m_rtPipeline stays null until install, so every "ready = (m_rtPipeline !=
 * null)" site in VulkanBackend.cpp keeps its meaning while a build is running.
 *
 * Environment switches (for A/B timing, not for normal use):
 *   RT_VK_PIPELINE_CACHE=0      no disk cache (neither loaded nor saved)
 *   RT_VK_RT_COMPILE_THREADS=0  synchronous create, no deferred operation
 *   RT_VK_RT_COMPILE_THREADS=N  at most N threads join the deferred operation
 */
#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace VulkanRT {

struct VulkanDevice::RTPipelineBuildJob {
    RTPipelineShaderSet shaders;
    uint64_t shaderHash = 0;
    std::thread thread;
    std::atomic<bool> finished{false};
    std::chrono::steady_clock::time_point startTime;

    // Results, written by the worker before `finished` is set.
    VkResult result = VK_ERROR_UNKNOWN;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t groupCount = 0;
    uint32_t raygenGroupIdx = 0;
    uint32_t missGroupIdx = 0;
    uint32_t triHitGroupIdx = 0;
    uint32_t photonGroupIdx = VK_SHADER_UNUSED_KHR;
    bool hasVolume = false;
    bool hasHair = false;
    bool hasSphere = false;
    bool hasShadowMiss = false;
    bool hasPhoton = false;
    double seconds = 0.0;
    bool cacheHitKnown = false;
    bool cacheHit = false;
    uint32_t deferredThreads = 0;
    std::string error;
};

namespace {

uint64_t hashShaderSet(const VulkanDevice::RTPipelineShaderSet& s) {
    uint64_t h = 14695981039346656037ull;
    auto mixWord = [&h](uint32_t w) { h ^= w; h *= 1099511628211ull; };
    auto mixVec = [&](const std::vector<std::uint32_t>& v) {
        // The length separates "stage absent" from "stage present but short".
        mixWord(static_cast<uint32_t>(v.size()));
        for (uint32_t w : v) mixWord(w);
    };
    mixVec(s.raygen); mixVec(s.miss); mixVec(s.closestHit); mixVec(s.anyHit);
    mixVec(s.volumeClosestHit); mixVec(s.volumeIntersection);
    mixVec(s.hairClosestHit); mixVec(s.hairIntersection); mixVec(s.hairAnyHit);
    mixVec(s.shadowMiss);
    mixVec(s.sphereClosestHit); mixVec(s.sphereIntersection);
    mixVec(s.photonRaygen);
    return h == 0 ? 1 : h;   // 0 is reserved for "nothing installed"
}

bool envEquals(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, value) == 0;
}

// -1 = unset (automatic).
int envInt(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return -1;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || n < 0) return -1;
    return static_cast<int>(std::min<long>(n, 1024));
}

// The status carries the path as UTF-8; std::filesystem::path::u8string() is
// std::u8string under C++20.
std::string pathToUtf8(const std::filesystem::path& p) {
    const std::u8string u8 = p.u8string();
    return std::string(u8.begin(), u8.end());
}
std::filesystem::path utf8ToPath(const std::string& s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

std::filesystem::path pipelineCacheDirectory() {
    namespace fs = std::filesystem;
#ifdef _WIN32
    // Wide getenv: a user name with non-ASCII characters must not break the path.
    if (const wchar_t* local = _wgetenv(L"LOCALAPPDATA"); local && *local) {
        return fs::path(local) / L"RayTrophiStudio" / L"vk_pipeline_cache";
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        return fs::path(xdg) / "RayTrophiStudio" / "vk_pipeline_cache";
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        return fs::path(home) / ".cache" / "RayTrophiStudio" / "vk_pipeline_cache";
    }
#endif
    return {};
}

// Vulkan pipeline cache header, version one (spec: VkPipelineCacheHeaderVersionOne).
struct CacheHeaderV1 {
    uint32_t headerSize;
    uint32_t headerVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint8_t  uuid[VK_UUID_SIZE];
};
static_assert(sizeof(CacheHeaderV1) == 32, "pipeline cache header is 32 bytes");

// Empty string = the blob belongs to this device and driver.
std::string validateCacheBlob(const std::vector<char>& blob, const VkPhysicalDeviceProperties& props) {
    if (blob.size() < sizeof(CacheHeaderV1)) return "file shorter than the cache header";
    CacheHeaderV1 h{};
    std::memcpy(&h, blob.data(), sizeof(h));
    if (h.headerSize < sizeof(CacheHeaderV1) || h.headerSize > blob.size())
        return "corrupt header size " + std::to_string(h.headerSize);
    if (h.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
        return "unknown header version " + std::to_string(h.headerVersion);
    if (h.vendorID != props.vendorID || h.deviceID != props.deviceID)
        return "written by another GPU";
    if (std::memcmp(h.uuid, props.pipelineCacheUUID, VK_UUID_SIZE) != 0)
        return "written by another driver version";
    return {};
}

} // namespace

// ============================================================================
// Persistent pipeline cache
// ============================================================================

void VulkanDevice::loadRTPipelineCache() {
    if (!m_device || m_rtPipelineCache != VK_NULL_HANDLE) return;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    m_rtCreationFeedbackSupported = props.apiVersion >= VK_API_VERSION_1_3;

    Backend::RTPipelineStatus status;
    status.available = true;

    std::vector<char> blob;
    if (envEquals("RT_VK_PIPELINE_CACHE", "0")) {
        SCENE_LOG_INFO("[VulkanDevice] RT_VK_PIPELINE_CACHE=0: RT pipeline disk cache disabled.");
    } else {
        const std::filesystem::path dir = pipelineCacheDirectory();
        if (dir.empty()) {
            SCENE_LOG_WARN("[VulkanDevice] No app-data folder resolved; RT pipeline disk cache disabled.");
        } else {
            char name[64];
            std::snprintf(name, sizeof(name), "rt_pipeline_%04x_%04x.bin", props.vendorID, props.deviceID);
            const std::filesystem::path file = dir / name;
            status.cachePath = pathToUtf8(file);

            std::ifstream in(file, std::ios::binary);
            if (in) {
                blob.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
                const std::string reject = validateCacheBlob(blob, props);
                if (!reject.empty()) {
                    // A blob from another driver is valid input to the driver (it
                    // ignores it), but a corrupt one is not; drop both, rebuild fresh.
                    status.cacheRejectReason = reject;
                    SCENE_LOG_WARN("[VulkanDevice] RT pipeline cache discarded (" + reject + "): " + status.cachePath);
                    blob.clear();
                    in.close();
                    std::error_code ec;
                    std::filesystem::remove(file, ec);
                }
            }
        }
    }

    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = blob.size();
    ci.pInitialData = blob.empty() ? nullptr : blob.data();
    VkResult r = vkCreatePipelineCache(m_device, &ci, nullptr, &m_rtPipelineCache);
    if (r != VK_SUCCESS && !blob.empty()) {
        status.cacheRejectReason = "driver rejected the data (" + std::to_string(static_cast<int>(r)) + ")";
        SCENE_LOG_WARN("[VulkanDevice] RT pipeline cache rejected by the driver; starting empty.");
        blob.clear();
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        r = vkCreatePipelineCache(m_device, &ci, nullptr, &m_rtPipelineCache);
    }
    if (r != VK_SUCCESS) {
        m_rtPipelineCache = VK_NULL_HANDLE;
        VK_WARN() << "[VulkanDevice] vkCreatePipelineCache failed: " << r
                  << "; RT pipeline builds run without a cache." << std::endl;
    } else if (!blob.empty()) {
        status.cacheLoaded = true;
        status.cacheLoadedBytes = blob.size();
        SCENE_LOG_INFO("[VulkanDevice] RT pipeline cache loaded (" + std::to_string(blob.size() / 1024) +
                       " KB): " + status.cachePath);
    }

    std::lock_guard<std::mutex> lock(m_rtBuildStatusMutex);
    m_rtBuildStatus = status;
}

// Worker thread, after a successful build. vkGetPipelineCacheData is safe here:
// the cache is internally synchronized and only the RT build uses it.
void VulkanDevice::saveRTPipelineCache() {
    std::string path;
    {
        std::lock_guard<std::mutex> lock(m_rtBuildStatusMutex);
        path = m_rtBuildStatus.cachePath;
    }
    if (m_rtPipelineCache == VK_NULL_HANDLE || path.empty()) return;

    size_t size = 0;
    if (vkGetPipelineCacheData(m_device, m_rtPipelineCache, &size, nullptr) != VK_SUCCESS || size == 0) return;
    std::vector<char> data(size);
    if (vkGetPipelineCacheData(m_device, m_rtPipelineCache, &size, data.data()) != VK_SUCCESS) return;
    data.resize(size);

    namespace fs = std::filesystem;
    const fs::path file = utf8ToPath(path);
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    // Write-then-rename: a crash mid-write must not leave a truncated cache that
    // the next launch hands to the driver.
    fs::path tmp = file;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            VK_WARN() << "[VulkanDevice] Cannot write RT pipeline cache: " << path << std::endl;
            return;
        }
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!out) {
            out.close();
            fs::remove(tmp, ec);
            VK_WARN() << "[VulkanDevice] RT pipeline cache write failed: " << path << std::endl;
            return;
        }
    }
    fs::rename(tmp, file, ec);
    if (ec) {
        fs::remove(tmp, ec);
        VK_WARN() << "[VulkanDevice] RT pipeline cache rename failed: " << path << std::endl;
        return;
    }
    SCENE_LOG_INFO("[VulkanDevice] RT pipeline cache saved (" + std::to_string(data.size() / 1024) + " KB).");
}

void VulkanDevice::shutdownRTPipelineBuild() {
    if (m_rtBuildJob) {
        // A compile cannot be cancelled; shutdown waits for it. Logged so a slow
        // exit right after a shader change is not mistaken for a hang.
        if (!m_rtBuildJob->finished.load(std::memory_order_acquire)) {
            SCENE_LOG_WARN("[VulkanDevice] Shutdown is waiting for the RT pipeline compile to finish.");
        }
        if (m_rtBuildJob->thread.joinable()) m_rtBuildJob->thread.join();
        if (m_rtBuildJob->pipeline != VK_NULL_HANDLE && m_device) {
            vkDestroyPipeline(m_device, m_rtBuildJob->pipeline, nullptr);
        }
        m_rtBuildJob.reset();
    }
    m_rtQueuedShaders.reset();
    if (m_rtPipelineCache != VK_NULL_HANDLE && m_device) {
        vkDestroyPipelineCache(m_device, m_rtPipelineCache, nullptr);
    }
    m_rtPipelineCache = VK_NULL_HANDLE;
    m_rtInstalledShaderHash = 0;
    m_rtFailedShaderHash = 0;
    std::lock_guard<std::mutex> lock(m_rtBuildStatusMutex);
    m_rtBuildStatus = Backend::RTPipelineStatus{};
}

// ============================================================================
// Layouts (static; created once per device)
// ============================================================================

// ★ These were recreated by every pipeline build, which leaked the previous
// pair and left m_rtDescriptorSet allocated from a layout that no longer
// existed. Their definition never changes, so one pair per device.
bool VulkanDevice::ensureRTPipelineLayouts() {
    if (m_rtPipelineLayout != VK_NULL_HANDLE) return true;

    // Descriptor set layout
    // Binding  0: Output Image
    // Binding  1: TLAS
    // Binding  2: Materials SSBO
    // Binding  3: Lights SSBO
    // Binding  4: Geometry SSBO
    // Binding  5: Instances SSBO
    // Binding  6: Material textures (runtime array)
    // Binding  7: World data SSBO
    // Binding  8: Atmosphere LUT samplers (transmittance, skyview, multi-scatter, aerial perspective)
    // Binding  9: Volume Instances SSBO
    // Binding 10: Hair Segment SSBO
    // Binding 11: Hair Material SSBO
    // Binding 12: Terrain Layer SSBO
    // Binding 13: Denoiser Beauty AOV
    // Binding 14: Denoiser Albedo AOV
    // Binding 15: Denoiser Normal AOV
    // Binding 17: Stylize AOV
    // Binding 18: Foam sphere SSBO (intersection + closest-hit)
    // Binding 19: Photon caustic hash grid SSBO (photon raygen writes, camera
    //             raygen debug-reads, closesthit gathers in Dilim 2)
    VkDescriptorSetLayoutBinding bindings[36] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    // RAYGEN added for photon.rgen (light-side emission reads the light buffer)
    bindings[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = static_cast<uint32_t>(Backend::VULKAN_TEXTURE_CAPACITY);
    bindings[6].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 4;
    // closesthit.rchit also samples the atmosphere LUTs (binding 8); a stage missing
    // from stageFlags reads an undefined descriptor (validation: layout-07988).
    bindings[8].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[9].binding = 9;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

    // Binding 10: Hair Segment SSBO
    // ANY_HIT added: hair_shadow_anyhit.rahit reads this to map gl_PrimitiveID → materialID
    // for deep self-shadow. A shader stage touching a binding the layout doesn't expose to
    // that stage makes the NVIDIA driver crash inside vkCreateRayTracingPipelinesKHR.
    bindings[10].binding = 10;
    bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[10].descriptorCount = 1;
    bindings[10].stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    // Binding 11: Hair Material SSBO
    // ANY_HIT added: hair_shadow_anyhit.rahit reads per-groom selfShadow strength.
    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    // Binding 12: Terrain Layer SSBO
    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[12].descriptorCount = 1;
    bindings[12].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[13].binding = 13;
    bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[14].binding = 14;
    bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[14].descriptorCount = 1;
    bindings[14].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[15].binding = 15;
    bindings[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[15].descriptorCount = 1;
    bindings[15].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[16].binding = 16;
    bindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[16].descriptorCount = 1;
    bindings[16].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding 17: Stylize AOV position+depth image (raygen-written, host-read)
    bindings[17].binding = 17;
    bindings[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[17].descriptorCount = 1;
    bindings[17].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding 18: Foam point-sphere SSBO (centre+radius+matId), read by the
    // sphere intersection + closest-hit shaders.
    bindings[18].binding = 18;
    bindings[18].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[18].descriptorCount = 1;
    bindings[18].stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 19: Photon caustic hash grid (header + cells, one SSBO)
    bindings[19].binding = 19;
    bindings[19].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[19].descriptorCount = 1;
    bindings[19].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 20: VOLUME photon grid (Faz 2V — photon raygen deposits along
    // flight segments, camera raygen marches/reads it back).
    bindings[20].binding = 20;
    bindings[20].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[20].descriptorCount = 1;
    bindings[20].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 21: Path-stats AOV (Debug Visualizer) — raygen-written running
    // average of path throughput (rgb) + bounce count (a); tonemap reads it.
    bindings[21].binding = 21;
    bindings[21].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[21].descriptorCount = 1;
    bindings[21].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding 22: Photon DIRECTION grid (Debug Visualizer view 5) — parallel
    // to the volume grid; photon.rgen deposits, camera raygen reads. Declared
    // by photon_grid.glsl, which closesthit also includes.
    bindings[22].binding = 22;
    bindings[22].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[22].descriptorCount = 1;
    bindings[22].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 23: Faz 2b material-program VM stream (flattened node graphs),
    // read by closest-hit's material_program.glsl interpreter.
    bindings[23].binding = 23;
    bindings[23].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[23].descriptorCount = 1;
    bindings[23].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 24: COLD material fields (VkGpuMaterialExt) — the feature-gated
    // half of the split material record (SSS/water/bubble/resin/dust). Only
    // closesthit reads it; the shadow any-hit path stays entirely on the hot
    // core buffer (binding 2).
    bindings[24].binding = 24;
    bindings[24].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[24].descriptorCount = 1;
    // shadow_anyhit.rahit reads MaterialExt too (same validation rule as binding 8).
    bindings[24].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    // Bindings 25-28: fixed temporal ping-pong slots. Their descriptors never
    // swap while an asynchronous frame is in flight; raygen selects read/write
    // slots from the temporal frame parity.
    for (uint32_t i = 25; i <= 28; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    }
    bindings[29].binding = 29;
    bindings[29].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[29].descriptorCount = 1;
    bindings[29].stageFlags =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Bindings 30-35: cloud field (cloud_rt.glsl / cloud_common.glsl) --
    // 30 base noise 3D, 31 detail 3D, 32 curl, 33 weather map, 34 CloudParams,
    // 35 majorant map. Miss draws the clouds; closest-hit reads them for the
    // sun's cloud shadow. Always written (writeRtCloudDescriptors).
    for (uint32_t i = 30; i <= 35; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = (i == 34) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                               : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    }

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount =  36;
    dslCI.pBindings = bindings;
    // A layout left from an attempt whose pipeline-layout step failed is kept:
    // bindRTDescriptors may already have allocated m_rtDescriptorSet from it.
    if (m_rtDescriptorSetLayout == VK_NULL_HANDLE &&
        vkCreateDescriptorSetLayout(m_device, &dslCI, nullptr, &m_rtDescriptorSetLayout) != VK_SUCCESS) {
        m_rtDescriptorSetLayout = VK_NULL_HANDLE;
        VK_ERROR() << "[VulkanDevice] RT descriptor set layout creation failed" << std::endl;
        return false;
    }

    // Push constant range (camera data + rendering params)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    pushRange.offset = 0;
    pushRange.size = 256; // Matches the expanded CameraPushConstants payload.

    // Pipeline layout
    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &m_rtDescriptorSetLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(m_device, &plCI, nullptr, &m_rtPipelineLayout) != VK_SUCCESS) {
        m_rtPipelineLayout = VK_NULL_HANDLE;
        VK_ERROR() << "[VulkanDevice] RT pipeline layout creation failed" << std::endl;
        return false;
    }
    return true;
}

// ============================================================================
// Request / status (render thread; status also read by the UI and IPC)
// ============================================================================

namespace {
int64_t steadyNowTicks() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}
double ticksToSeconds(int64_t ticks) {
    using Period = std::chrono::steady_clock::period;
    return static_cast<double>(ticks) * Period::num / Period::den;
}
} // namespace

Backend::RTPipelineStatus::State VulkanDevice::requestRTPipelineBuild(RTPipelineShaderSet shaders) {
    using State = Backend::RTPipelineStatus::State;
    if (!hasHardwareRT() || !fpCreateRayTracingPipelinesKHR) {
        VK_ERROR() << "[VulkanDevice] Hardware RT not available" << std::endl;
        std::lock_guard<std::mutex> lock(m_rtBuildStatusMutex);
        m_rtBuildStatus.state = State::Failed;
        m_rtBuildStatus.error = "hardware ray tracing is not available on this device";
        return State::Failed;
    }
    const uint64_t hash = hashShaderSet(shaders);
    if (m_rtBuildJob) {
        // One compile at a time. A different set (shaders rebuilt mid-compile)
        // is queued and starts when the running one has been polled.
        if (m_rtBuildJob->shaderHash != hash) {
            m_rtQueuedShaders = std::make_unique<RTPipelineShaderSet>(std::move(shaders));
        } else {
            m_rtQueuedShaders.reset();
        }
        return State::Compiling;
    }
    // ★ Re-entry with unchanged shaders (rebuildAccelerationStructure resets the
    // adapter's init flag) used to recompile the whole pipeline and leak the old one.
    if (m_rtPipeline != VK_NULL_HANDLE && m_rtInstalledShaderHash == hash) return State::Ready;
    // A failed compile costs minutes; the same shaders fail the same way.
    if (hash == m_rtFailedShaderHash) return State::Failed;
    return startRTPipelineBuild(std::move(shaders), hash) ? State::Compiling : State::Failed;
}

bool VulkanDevice::startRTPipelineBuild(RTPipelineShaderSet shaders, uint64_t hash) {
    using State = Backend::RTPipelineStatus::State;
    // Layouts are created HERE, on the render thread, so the worker only reads them.
    if (!ensureRTPipelineLayouts()) {
        m_rtFailedShaderHash = hash;
        std::lock_guard<std::mutex> lock(m_rtBuildStatusMutex);
        m_rtBuildStatus.available = true;
        m_rtBuildStatus.state = State::Failed;
        m_rtBuildStatus.error = "RT descriptor set / pipeline layout creation failed";
        return false;
    }

    auto job = std::make_shared<RTPipelineBuildJob>();
    job->shaders = std::move(shaders);
    job->shaderHash = hash;
    job->startTime = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(m_rtBuildStatusMutex);
        m_rtBuildStatus.available = true;
        m_rtBuildStatus.state = State::Compiling;
        m_rtBuildStatus.awaitingInstall = false;
        m_rtBuildStatus.error.clear();
        m_rtBuildStatus.compileSeconds = 0.0;
        m_rtBuildStartTicks = job->startTime.time_since_epoch().count();
    }
    SCENE_LOG_INFO("[VulkanDevice] Compiling the ray tracing pipeline on a worker thread "
                   "(the viewport stays responsive meanwhile)...");

    RTPipelineBuildJob* raw = job.get();
    m_rtBuildJob = std::move(job);
    try {
        // `raw` outlives the thread: the job is only released after join().
        m_rtBuildJob->thread = std::thread([this, raw]() { runRTPipelineBuildJob(*raw); });
    } catch (const std::system_error& e) {
        SCENE_LOG_WARN(std::string("[VulkanDevice] Cannot start the RT compile thread (") + e.what() +
                       "); compiling on the render thread.");
        runRTPipelineBuildJob(*raw);
    }
    return true;
}

Backend::RTPipelineStatus VulkanDevice::getRTPipelineStatus() const {
    std::lock_guard<std::mutex> lock(m_rtBuildStatusMutex);
    Backend::RTPipelineStatus s = m_rtBuildStatus;
    if (s.state == Backend::RTPipelineStatus::State::Compiling && !s.awaitingInstall) {
        s.compileSeconds = ticksToSeconds(steadyNowTicks() - m_rtBuildStartTicks);
    }
    return s;
}

// ============================================================================
// Worker
// ============================================================================

void VulkanDevice::runRTPipelineBuildJob(RTPipelineBuildJob& job) {
    const auto t0 = std::chrono::steady_clock::now();
    const RTPipelineShaderSet& s = job.shaders;

    std::vector<VkShaderModule> modules;
    auto createModule = [&](const std::vector<std::uint32_t>& code, const char* name) -> VkShaderModule {
        if (code.empty()) return VK_NULL_HANDLE;
        // Reject truncated/stale files before handing them to the driver. The
        // SPIR-V header is five words and starts with 0x07230203.
        if (code.size() < 5 || code[0] != 0x07230203u) {
            VK_ERROR() << "[VulkanDevice] Invalid SPIR-V module " << name << " (words="
                       << code.size() << ")" << std::endl;
            return VK_NULL_HANDLE;
        }
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = code.size() * sizeof(uint32_t);
        ci.pCode = code.data();
        VkShaderModule mod = VK_NULL_HANDLE;
        const VkResult r = vkCreateShaderModule(m_device, &ci, nullptr, &mod);
        if (r != VK_SUCCESS) {
            VK_ERROR() << "[VulkanDevice] vkCreateShaderModule failed for " << name << ": "
                       << r << " (words=" << code.size() << ")" << std::endl;
            return VK_NULL_HANDLE;
        }
        modules.push_back(mod);
        return mod;
    };

    const VkShaderModule raygenModule     = createModule(s.raygen, "raygen");
    const VkShaderModule missModule       = createModule(s.miss, "miss");
    const VkShaderModule chitModule       = createModule(s.closestHit, "closesthit");
    const VkShaderModule anyhitModule     = createModule(s.anyHit, "shadow_anyhit");
    const VkShaderModule volChitModule    = createModule(s.volumeClosestHit, "volume_closesthit");
    const VkShaderModule volIntModule     = createModule(s.volumeIntersection, "volume_intersection");
    const VkShaderModule hairChitModule   = createModule(s.hairClosestHit, "hair_closesthit");
    const VkShaderModule hairIntModule    = createModule(s.hairIntersection, "hair_intersection");
    const VkShaderModule hairAnyHitModule = createModule(s.hairAnyHit, "hair_shadow_anyhit");
    const VkShaderModule shadowMissModule = createModule(s.shadowMiss, "shadow_miss");
    const VkShaderModule sphereChitModule = createModule(s.sphereClosestHit, "sphere_closesthit");
    const VkShaderModule sphereIntModule  = createModule(s.sphereIntersection, "sphere_intersection");
    const VkShaderModule photonRgenModule = createModule(s.photonRaygen, "photon");

    job.hasVolume     = (volChitModule != VK_NULL_HANDLE && volIntModule != VK_NULL_HANDLE);
    job.hasHair       = (hairChitModule != VK_NULL_HANDLE && hairIntModule != VK_NULL_HANDLE);
    job.hasSphere     = (sphereChitModule != VK_NULL_HANDLE && sphereIntModule != VK_NULL_HANDLE);
    job.hasShadowMiss = (shadowMissModule != VK_NULL_HANDLE);
    job.hasPhoton     = (photonRgenModule != VK_NULL_HANDLE);

    // One create call through a deferred operation joined by several threads.
    // `ci` and everything it points to must stay alive until the operation
    // completes, which this function guarantees by joining before it returns.
    auto createPipeline = [&](VkRayTracingPipelineCreateInfoKHR& ci) -> VkResult {
        VkPipelineCreationFeedback pipelineFeedback{};
        std::vector<VkPipelineCreationFeedback> stageFeedback(ci.stageCount);
        VkPipelineCreationFeedbackCreateInfo feedbackCI{};
        feedbackCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
        feedbackCI.pPipelineCreationFeedback = &pipelineFeedback;
        feedbackCI.pipelineStageCreationFeedbackCount = ci.stageCount;
        feedbackCI.pPipelineStageCreationFeedbacks = stageFeedback.data();
        const void* const originalNext = ci.pNext;
        if (m_rtCreationFeedbackSupported) {
            feedbackCI.pNext = originalNext;
            ci.pNext = &feedbackCI;
        }

        const int threadCap = envInt("RT_VK_RT_COMPILE_THREADS");
        const bool deferredAvailable = m_deferredHostOpsEnabled && fpCreateDeferredOperationKHR &&
            fpDestroyDeferredOperationKHR && fpGetDeferredOperationMaxConcurrencyKHR &&
            fpGetDeferredOperationResultKHR && fpDeferredOperationJoinKHR;
        VkDeferredOperationKHR op = VK_NULL_HANDLE;
        if (deferredAvailable && threadCap != 0 &&
            fpCreateDeferredOperationKHR(m_device, nullptr, &op) != VK_SUCCESS) {
            op = VK_NULL_HANDLE;
        }

        job.pipeline = VK_NULL_HANDLE;
        uint32_t joiners = 0;
        VkResult r = fpCreateRayTracingPipelinesKHR(m_device, op, m_rtPipelineCache, 1, &ci, nullptr, &job.pipeline);
        if (op != VK_NULL_HANDLE && r == VK_OPERATION_DEFERRED_KHR) {
            // UINT32_MAX = "any number". Leave one core to the UI/render thread.
            const uint32_t maxConcurrency = std::max(1u, fpGetDeferredOperationMaxConcurrencyKHR(m_device, op));
            const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
            uint32_t n = std::min(maxConcurrency, hw > 1 ? hw - 1 : 1u);
            if (threadCap > 0) n = std::min<uint32_t>(n, static_cast<uint32_t>(threadCap));
            n = std::max(1u, n);
            joiners = n;

            auto joinLoop = [this, op]() {
                for (;;) {
                    const VkResult jr = fpDeferredOperationJoinKHR(m_device, op);
                    // SUCCESS: the operation is complete. THREAD_DONE: no work
                    // left for this thread; the others finish it.
                    if (jr == VK_SUCCESS || jr == VK_THREAD_DONE_KHR) return;
                    if (jr == VK_THREAD_IDLE_KHR) {
                        // Temporarily no work to hand out; more may appear.
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        continue;
                    }
                    return;   // an error: vkGetDeferredOperationResultKHR reports it
                }
            };
            std::vector<std::thread> helpers;
            helpers.reserve(n - 1);
            for (uint32_t i = 1; i < n; ++i) {
                try {
                    helpers.emplace_back(joinLoop);
                } catch (const std::system_error&) {
                    joiners = i;   // fewer helpers; the operation still completes
                    break;
                }
            }
            joinLoop();
            for (auto& t : helpers) t.join();
            while ((r = fpGetDeferredOperationResultKHR(m_device, op)) == VK_NOT_READY) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (r == VK_OPERATION_NOT_DEFERRED_KHR) {
            // The driver completed the work inside the call (allowed by the spec).
            joiners = 1;
            r = (job.pipeline != VK_NULL_HANDLE) ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
        }
        if (op != VK_NULL_HANDLE) fpDestroyDeferredOperationKHR(m_device, op, nullptr);
        ci.pNext = originalNext;

        job.deferredThreads = joiners;
        job.cacheHitKnown = false;
        job.cacheHit = false;
        if (r == VK_SUCCESS && m_rtCreationFeedbackSupported &&
            (pipelineFeedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT)) {
            job.cacheHitKnown = true;
            job.cacheHit = (pipelineFeedback.flags &
                            VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
        }
        if (r != VK_SUCCESS && job.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, job.pipeline, nullptr);
            job.pipeline = VK_NULL_HANDLE;
        }
        return r;
    };

    auto makeStage = [](VkShaderStageFlagBits stageBit, VkShaderModule mod) {
        VkPipelineShaderStageCreateInfo st{};
        st.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st.stage  = stageBit;
        st.module = mod;
        st.pName  = "main";
        return st;
    };
    auto makeGeneralGroup = [](uint32_t stageIdx) {
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        g.generalShader      = stageIdx;
        g.closestHitShader   = VK_SHADER_UNUSED_KHR;
        g.anyHitShader       = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        return g;
    };
    auto makeHitGroup = [](VkRayTracingShaderGroupTypeKHR type, uint32_t chit, uint32_t ahit, uint32_t isect) {
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type               = type;
        g.generalShader      = VK_SHADER_UNUSED_KHR;
        g.closestHitShader   = chit;
        g.anyHitShader       = ahit;
        g.intersectionShader = isect;
        return g;
    };

    // Stage order: raygen, primary miss, [shadow miss], closesthit, [anyhit],
    // [vol chit, vol int], [hair chit, hair int, hair anyhit], [sphere chit,
    // sphere int], [photon raygen].
    // Group order (the SBT layout depends on it): raygen | miss, [shadow miss] |
    // triangle hit, [volume], [hair], [sphere] | [photon raygen]. The photon
    // group comes after all hit groups so the miss/hit regions stay contiguous.
    auto attempt = [&](bool withHairAnyHit) -> VkResult {
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
        stages.reserve(16);
        groups.reserve(10);
        auto addStage = [&](VkShaderStageFlagBits bit, VkShaderModule mod) {
            stages.push_back(makeStage(bit, mod));
            return static_cast<uint32_t>(stages.size() - 1);
        };

        const uint32_t raygenStage = addStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, raygenModule);
        const uint32_t missStage   = addStage(VK_SHADER_STAGE_MISS_BIT_KHR, missModule);
        const uint32_t shadowMissStage = job.hasShadowMiss
            ? addStage(VK_SHADER_STAGE_MISS_BIT_KHR, shadowMissModule) : VK_SHADER_UNUSED_KHR;
        const uint32_t chitStage = addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chitModule);
        const uint32_t anyhitStage = (anyhitModule != VK_NULL_HANDLE)
            ? addStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, anyhitModule) : VK_SHADER_UNUSED_KHR;
        uint32_t volChitStage = VK_SHADER_UNUSED_KHR, volIntStage = VK_SHADER_UNUSED_KHR;
        if (job.hasVolume) {
            volChitStage = addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, volChitModule);
            volIntStage  = addStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, volIntModule);
        }
        uint32_t hairChitStage = VK_SHADER_UNUSED_KHR, hairIntStage = VK_SHADER_UNUSED_KHR;
        uint32_t hairAnyHitStage = VK_SHADER_UNUSED_KHR;
        if (job.hasHair) {
            hairChitStage = addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, hairChitModule);
            hairIntStage  = addStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, hairIntModule);
            if (withHairAnyHit && hairAnyHitModule != VK_NULL_HANDLE) {
                hairAnyHitStage = addStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, hairAnyHitModule);
            }
        }
        uint32_t sphereChitStage = VK_SHADER_UNUSED_KHR, sphereIntStage = VK_SHADER_UNUSED_KHR;
        if (job.hasSphere) {
            sphereChitStage = addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, sphereChitModule);
            sphereIntStage  = addStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, sphereIntModule);
        }
        const uint32_t photonStage = job.hasPhoton
            ? addStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, photonRgenModule) : VK_SHADER_UNUSED_KHR;

        job.raygenGroupIdx = static_cast<uint32_t>(groups.size());
        groups.push_back(makeGeneralGroup(raygenStage));
        job.missGroupIdx = static_cast<uint32_t>(groups.size());
        groups.push_back(makeGeneralGroup(missStage));
        if (job.hasShadowMiss) groups.push_back(makeGeneralGroup(shadowMissStage));
        job.triHitGroupIdx = static_cast<uint32_t>(groups.size());
        groups.push_back(makeHitGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                                      chitStage, anyhitStage, VK_SHADER_UNUSED_KHR));
        if (job.hasVolume) {
            groups.push_back(makeHitGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
                                          volChitStage, VK_SHADER_UNUSED_KHR, volIntStage));
        }
        if (job.hasHair) {
            groups.push_back(makeHitGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
                                          hairChitStage, hairAnyHitStage, hairIntStage));
        }
        if (job.hasSphere) {
            groups.push_back(makeHitGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
                                          sphereChitStage, VK_SHADER_UNUSED_KHR, sphereIntStage));
        }
        job.photonGroupIdx = VK_SHADER_UNUSED_KHR;
        if (job.hasPhoton) {
            job.photonGroupIdx = static_cast<uint32_t>(groups.size());
            groups.push_back(makeGeneralGroup(photonStage));
        }
        job.groupCount = static_cast<uint32_t>(groups.size());

        VkRayTracingPipelineCreateInfoKHR rtCI{};
        rtCI.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rtCI.stageCount = static_cast<uint32_t>(stages.size());
        rtCI.pStages = stages.data();
        rtCI.groupCount = static_cast<uint32_t>(groups.size());
        rtCI.pGroups = groups.data();
        rtCI.maxPipelineRayRecursionDepth = 2; // Required for shadow rays from closesthit
        rtCI.layout = m_rtPipelineLayout;
        return createPipeline(rtCI);
    };

    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    if (raygenModule == VK_NULL_HANDLE || missModule == VK_NULL_HANDLE || chitModule == VK_NULL_HANDLE) {
        job.error = "raygen/miss/closesthit SPIR-V missing or invalid (see log)";
    } else {
        const bool hairAnyHitIncluded = job.hasHair && hairAnyHitModule != VK_NULL_HANDLE;
        result = attempt(true);
        // The NVIDIA driver once failed on the hair shadow any-hit (a stage reading a
        // binding its layout did not expose to it). Retry only when the second
        // attempt actually differs: an identical retry is another multi-minute
        // compile that fails the same way.
        if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST && hairAnyHitIncluded) {
            VK_WARN() << "[VulkanDevice] vkCreateRayTracingPipelinesKHR failed: " << result
                      << ". Retrying without the hair shadow any-hit..." << std::endl;
            result = attempt(false);
            if (result != VK_SUCCESS) {
                VK_WARN() << "[VulkanDevice] Retry also failed: " << result << std::endl;
            }
        }
        if (result != VK_SUCCESS) {
            job.error = "vkCreateRayTracingPipelinesKHR failed (VkResult " + std::to_string(static_cast<int>(result)) + ")";
        }
    }

    for (VkShaderModule m : modules) vkDestroyShaderModule(m_device, m, nullptr);

    job.result = result;
    job.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Nothing new to persist after a cache hit.
    if (result == VK_SUCCESS && !(job.cacheHitKnown && job.cacheHit)) saveRTPipelineCache();
    {
        // The install waits for the next Rendered frame; until then the status
        // must not keep counting (the user may be sitting in Solid).
        std::lock_guard<std::mutex> lock(m_rtBuildStatusMutex);
        m_rtBuildStatus.awaitingInstall = true;
        m_rtBuildStatus.compileSeconds = job.seconds;
    }
    job.finished.store(true, std::memory_order_release);
}

// ============================================================================
// Poll / install (render thread)
// ============================================================================

bool VulkanDevice::pollRTPipelineBuild(bool block) {
    if (!m_rtBuildJob) return false;
    if (!block && !m_rtBuildJob->finished.load(std::memory_order_acquire)) return false;
    if (m_rtBuildJob->thread.joinable()) m_rtBuildJob->thread.join();

    std::shared_ptr<RTPipelineBuildJob> done = std::move(m_rtBuildJob);
    const bool installed = installRTPipelineBuild(*done);

    if (m_rtQueuedShaders) {
        RTPipelineShaderSet next = std::move(*m_rtQueuedShaders);
        m_rtQueuedShaders.reset();
        const uint64_t h = hashShaderSet(next);
        if (h != m_rtInstalledShaderHash && h != m_rtFailedShaderHash) {
            startRTPipelineBuild(std::move(next), h);
        }
    }
    return installed;
}

bool VulkanDevice::installRTPipelineBuild(RTPipelineBuildJob& job) {
    using State = Backend::RTPipelineStatus::State;
    auto publish = [&](State state, const std::string& error) {
        std::lock_guard<std::mutex> lock(m_rtBuildStatusMutex);
        m_rtBuildStatus.available = true;
        m_rtBuildStatus.state = state;
        m_rtBuildStatus.awaitingInstall = false;
        m_rtBuildStatus.error = error;
        m_rtBuildStatus.compileSeconds = job.seconds;
        m_rtBuildStatus.cacheHitKnown = job.cacheHitKnown;
        m_rtBuildStatus.cacheHit = job.cacheHit;
        m_rtBuildStatus.deferredThreads = job.deferredThreads;
        ++m_rtBuildStatus.buildCount;
    };
    auto fail = [&](const std::string& error) {
        if (job.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, job.pipeline, nullptr);
            job.pipeline = VK_NULL_HANDLE;
        }
        m_rtFailedShaderHash = job.shaderHash;
        VK_ERROR() << "[VulkanDevice] RT pipeline build failed after " << job.seconds << " s: "
                   << error << std::endl;
        // An older pipeline, if one is installed, keeps rendering; the status
        // still says Failed so nobody mistakes it for the new shaders.
        publish(State::Failed, error);
        return false;
    };

    if (job.result != VK_SUCCESS || job.pipeline == VK_NULL_HANDLE) {
        return fail(job.error.empty() ? "vkCreateRayTracingPipelinesKHR failed" : job.error);
    }

    const uint32_t handleSize = m_capabilities.shaderGroupHandleSize;
    uint32_t handleAlignment = m_capabilities.shaderGroupBaseAlignment;
    if (handleAlignment == 0) handleAlignment = handleSize; // Fallback
    if (handleSize == 0) {
        return fail("shaderGroupHandleSize is 0 (RT capabilities not queried)");
    }
    const uint32_t groupCount = job.groupCount;
    const uint32_t alignedHandleSize = (handleSize + (handleAlignment - 1)) & ~(handleAlignment - 1);

    std::vector<uint8_t> handleData(static_cast<size_t>(groupCount) * handleSize);
    const VkResult sbtResult = fpGetRayTracingShaderGroupHandlesKHR(
        m_device, job.pipeline, 0, groupCount, handleData.size(), handleData.data());
    if (sbtResult != VK_SUCCESS) {
        return fail("vkGetRayTracingShaderGroupHandlesKHR failed (VkResult " + std::to_string(static_cast<int>(sbtResult)) +
                    ", groups=" + std::to_string(groupCount) + ")");
    }

    // Replacing a live pipeline (shaders changed): traces in flight still use
    // the old pipeline and its SBT.
    if (m_rtPipeline != VK_NULL_HANDLE) {
        waitIdle();
        vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
        m_rtPipeline = VK_NULL_HANDLE;
        m_rtPipelineReady = false;
    }
    if (m_sbtBuffer.buffer) destroyBuffer(m_sbtBuffer);

    // SBT layout: [raygen | miss(s) | hit(s) | photon raygen] each entry aligned
    BufferCreateInfo sbtBufInfo;
    sbtBufInfo.size = static_cast<uint64_t>(alignedHandleSize) * groupCount;
    sbtBufInfo.usage = BufferUsage::SHADER_BINDING | BufferUsage::TRANSFER_DST;
    sbtBufInfo.location = MemoryLocation::CPU_TO_GPU;
    m_sbtBuffer = createBuffer(sbtBufInfo);
    auto* mapped = m_sbtBuffer.buffer ? static_cast<uint8_t*>(mapBuffer(m_sbtBuffer)) : nullptr;
    if (!mapped) {
        if (m_sbtBuffer.buffer) destroyBuffer(m_sbtBuffer);
        return fail("shader binding table allocation failed");
    }
    for (uint32_t i = 0; i < groupCount; i++) {
        std::memcpy(mapped + static_cast<size_t>(i) * alignedHandleSize,
                    handleData.data() + static_cast<size_t>(i) * handleSize, handleSize);
    }
    unmapBuffer(m_sbtBuffer);

    const VkDeviceAddress sbtAddr = m_sbtBuffer.deviceAddress;

    // Raygen region (always 1 entry)
    m_sbtRaygenRegion.deviceAddress = sbtAddr + static_cast<VkDeviceAddress>(job.raygenGroupIdx) * alignedHandleSize;
    m_sbtRaygenRegion.stride = alignedHandleSize;
    m_sbtRaygenRegion.size   = alignedHandleSize;

    // Miss region: primary_miss + optional shadow_miss (contiguous)
    const uint32_t numMissGroups = 1u + (job.hasShadowMiss ? 1u : 0u);
    m_sbtMissRegion.deviceAddress = sbtAddr + static_cast<VkDeviceAddress>(job.missGroupIdx) * alignedHandleSize;
    m_sbtMissRegion.stride = alignedHandleSize;
    m_sbtMissRegion.size   = static_cast<VkDeviceSize>(numMissGroups) * alignedHandleSize;

    // Hit region: triangle + optional volume + optional hair + optional sphere (contiguous)
    const uint32_t numHitGroups = 1u + (job.hasVolume ? 1u : 0u) + (job.hasHair ? 1u : 0u) +
                                  (job.hasSphere ? 1u : 0u);
    m_sbtHitRegion.deviceAddress = sbtAddr + static_cast<VkDeviceAddress>(job.triHitGroupIdx) * alignedHandleSize;
    m_sbtHitRegion.stride = alignedHandleSize;
    m_sbtHitRegion.size   = static_cast<VkDeviceSize>(numHitGroups) * alignedHandleSize;

    m_sbtCallableRegion = {}; // No callable shaders

    // Photon caustic raygen region (Faz 2) — same SBT buffer, its own raygen slot.
    m_hasPhotonRaygen = job.hasPhoton && (job.photonGroupIdx != VK_SHADER_UNUSED_KHR);
    if (m_hasPhotonRaygen) {
        m_sbtPhotonRegion.deviceAddress = sbtAddr + static_cast<VkDeviceAddress>(job.photonGroupIdx) * alignedHandleSize;
        m_sbtPhotonRegion.stride = alignedHandleSize;
        m_sbtPhotonRegion.size   = alignedHandleSize;
    } else {
        m_sbtPhotonRegion = {};
    }

    m_hasVolumeShaders = job.hasVolume;
    m_hasHairShaders   = job.hasHair;
    m_hasSphereShaders = job.hasSphere;
    m_hasShadowMiss    = job.hasShadowMiss;
    m_rtPipeline = job.pipeline;
    job.pipeline = VK_NULL_HANDLE;
    m_rtInstalledShaderHash = job.shaderHash;
    m_rtPipelineReady = true;
    publish(State::Ready, {});

    char secs[32];
    std::snprintf(secs, sizeof(secs), "%.1f", job.seconds);
    std::ostringstream msg;
    msg << "[VulkanDevice] RT pipeline ready in " << secs << " s (groups=" << groupCount
        << ", threads=" << job.deferredThreads
        << ", cache=" << (job.cacheHitKnown ? (job.cacheHit ? "hit" : "miss") : "unknown")
        << ", volume=" << (m_hasVolumeShaders ? "yes" : "no")
        << ", hair=" << (m_hasHairShaders ? "yes" : "no")
        << ", sphere=" << (m_hasSphereShaders ? "yes" : "no")
        << ", shadowMiss=" << (m_hasShadowMiss ? "yes" : "no") << ")";
    SCENE_LOG_INFO(msg.str());
    return true;
}

} // namespace VulkanRT
