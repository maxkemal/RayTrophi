/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          VulkanBackend.cpp
 * Description:   Vulkan Backend Implementation
 *                Core initialization, device setup, RT extension loading
 * =========================================================================
 */
#include "Backend/VulkanBackend.h"
#include "Backend/vulkan_world_data.h"
#include "Core/RenderStateManager.h"
#include "Stylize/StylizeKernel.h"
#include "MeshModifiers.h"   // Phase 3d: CCDeviceGeometry release for device-resident meshes
#include "SimulationCompute.h" // Phase 3d: release shared mesh compute backend before device destroy
#include "globals.h"
#include <SDL_surface.h>
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <chrono>
#include <filesystem>
#include <functional>
#include <future>
#include <sstream>
#include <thread>
#include "HittableInstance.h"
#include "HittableList.h"
#include "ParallelBVHNode.h"
#include "Triangle.h"
#include "TerrainSystem.h"
#include "VDBVolume.h"
#include "GasVolume.h"
#include "VDBVolumeManager.h"
#include "params.h"  // GpuVDBVolume, GpuGasVolume definitions
#include <SDL.h>
#include "stb_image_write.h"
#include "Texture.h"
#include "TextureCompression.h"
#include "TextureCompressionCache.h"
#include "World.h"
#include "AtmosphereLUT.h"
#include "CameraPresets.h"
#include "Camera.h"
#include "InstanceManager.h"
#include "DllLoadPolicy.h"
#include "material_gpu.h"

// Delay-load handler: attempt to LoadLibrary when a delay-loaded DLL fails
#include <windows.h>
#include <delayimp.h>

// CUDA runtime — used only by the Vulkan→CUDA OIDN denoiser interop path
// (getDenoiserFrameGPU). Guarded by supportsExternalMemoryWin32 capability.
#include <cuda_runtime.h>
#include "oidn_blend_cuda.h"

namespace {
bool isWaterTriangleMaterial(const std::shared_ptr<Triangle>& tri) {
    if (!tri) return false;
    Material* mat = MaterialManager::getInstance().getMaterial(tri->getMaterialID());
    if (!mat || !mat->gpuMaterial) return false;
    const GpuMaterial& gpu = *mat->gpuMaterial;
    return (gpu.flags & GPU_MAT_FLAG_WATER) != 0 || gpu.sheen > 0.0001f;
}

bool triangleHasEffectiveSkinData(const Triangle& tri) {
    if (!tri.hasSkinData()) return false;
    for (int v = 0; v < 3; ++v) {
        for (const auto& [boneIndex, weight] : tri.getSkinBoneWeights(v)) {
            if (boneIndex >= 0 && weight > 0.0f) return true;
        }
    }
    return false;
}

bool triangleDataHasEffectiveSkinData(const Backend::TriangleData& tri) {
    const int32_t* boneIndices[3] = { tri.boneIndices_v0, tri.boneIndices_v1, tri.boneIndices_v2 };
    const float* boneWeights[3] = { tri.boneWeights_v0, tri.boneWeights_v1, tri.boneWeights_v2 };
    for (int v = 0; v < 3; ++v) {
        for (int b = 0; b < 4; ++b) {
            if (boneIndices[v][b] >= 0 && boneWeights[v][b] > 0.0f) return true;
        }
    }
    return false;
}

struct AtmosphereLUTParamsGPU {
    float sunDir_intensity[4];
    float density_intensity[4];
    float physical[4];
    float weather[4];
    float rayleigh[4];
    float mie[4];
};

static AtmosphereLUTParamsGPU makeAtmosphereLUTParamsGPU(const WorldData& world) {
    const NishitaSkyParams& n = world.nishita;
    AtmosphereLUTParamsGPU p{};
    p.sunDir_intensity[0] = n.sun_direction.x;
    p.sunDir_intensity[1] = n.sun_direction.y;
    p.sunDir_intensity[2] = n.sun_direction.z;
    p.sunDir_intensity[3] = n.sun_intensity;
    p.density_intensity[0] = n.air_density;
    p.density_intensity[1] = n.dust_density;
    p.density_intensity[2] = n.ozone_density;
    p.density_intensity[3] = n.atmosphere_intensity;
    p.physical[0] = n.planet_radius;
    p.physical[1] = n.atmosphere_height;
    p.physical[2] = n.altitude;
    p.physical[3] = n.mie_anisotropy;
    p.weather[0] = n.humidity;
    p.weather[1] = n.temperature;
    p.weather[2] = n.ozone_absorption_scale;
    p.weather[3] = 0.0f;
    p.rayleigh[0] = n.rayleigh_scattering.x;
    p.rayleigh[1] = n.rayleigh_scattering.y;
    p.rayleigh[2] = n.rayleigh_scattering.z;
    p.rayleigh[3] = n.rayleigh_density;
    p.mie[0] = n.mie_scattering.x;
    p.mie[1] = n.mie_scattering.y;
    p.mie[2] = n.mie_scattering.z;
    p.mie[3] = n.mie_density;
    return p;
}

bool materialCanUseOpaqueFastPath(uint32_t materialId) {
    if (materialId == MaterialManager::INVALID_MATERIAL_ID) return true;
    Material* mat = MaterialManager::getInstance().getMaterial(static_cast<uint16_t>(materialId));
    if (!mat) return true;
    if (mat->isTransparent()) return false;
    // Transmissive glass/water must NOT take the opaque fast path: the shadow
    // any-hit implements coloured pass-through shadows (CPU shadow-walk parity),
    // and VK_GEOMETRY_OPAQUE_BIT would silently keep their shadows solid black.
    if (mat->getTransmission(Vec2(0.5f, 0.5f)) > 0.001f) return false;
    if (mat->type() == MaterialType::Dielectric) return false;
    return true;
}

// Photon caustics: transmissive (glass/water) materials define the photon
// emission target — photons are aimed at these meshes' bounds, independent of
// the camera (aiming at the camera focus made the caustic vanish on view change).
bool materialIsCausticCaster(uint32_t materialId) {
    if (materialId == MaterialManager::INVALID_MATERIAL_ID) return false;
    Material* mat = MaterialManager::getInstance().getMaterial(static_cast<uint16_t>(materialId));
    if (!mat) return false;
    if (mat->getTransmission(Vec2(0.5f, 0.5f)) > 0.001f) return true;
    return mat->type() == MaterialType::Dielectric;
}

bool parseScatterNodeName(const std::string& nodeName, int& groupId, uint32_t& instanceIndex) {
    constexpr const char* kPrefix = "_inst_gid";
    constexpr size_t kPrefixLen = 9;
    if (nodeName.rfind(kPrefix, 0) != 0) return false;

    const size_t sep = nodeName.find('_', kPrefixLen);
    if (sep == std::string::npos || sep == kPrefixLen || sep + 1 >= nodeName.size()) {
        return false;
    }

    try {
        groupId = std::stoi(nodeName.substr(kPrefixLen, sep - kPrefixLen));
        const unsigned long long idx = std::stoull(nodeName.substr(sep + 1));
        if (idx > UINT32_MAX) return false;
        instanceIndex = static_cast<uint32_t>(idx);
        return true;
    } catch (...) {
        return false;
    }
}

bool refreshVulkanGeometryDataBinding(VulkanRT::VulkanDevice* device) {
    if (!device) return false;

    if (device->m_geometryDataBuffer.buffer) {
        device->destroyBuffer(device->m_geometryDataBuffer);
    }

    if (device->m_blasList.empty()) return true;

    std::vector<VulkanRT::VkGeometryData> geoData;
    geoData.reserve(device->m_blasList.size());
    for (const auto& blas : device->m_blasList) {
        VulkanRT::VkGeometryData d;
        d.vertexAddr = blas.vertexBuffer.deviceAddress;
        d.normalAddr = blas.normalBuffer.deviceAddress;
        d.uvAddr = blas.uvBuffer.deviceAddress;
        d.indexAddr = blas.indexBuffer.deviceAddress;
        d.materialAddr = blas.materialIndexBuffer.buffer ? blas.materialIndexBuffer.deviceAddress : 0;
        geoData.push_back(d);
    }

    VulkanRT::BufferCreateInfo ci;
    ci.size = static_cast<uint64_t>(geoData.size()) * sizeof(VulkanRT::VkGeometryData);
    ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
    ci.location = VulkanRT::MemoryLocation::GPU_ONLY;
    ci.initialData = nullptr;
    device->m_geometryDataBuffer = device->createBuffer(ci);
    if (!device->m_geometryDataBuffer.buffer) return false;
    device->uploadBuffer(device->m_geometryDataBuffer, geoData.data(), ci.size);

    if (device->m_rtDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo geoInfo{};
        geoInfo.buffer = device->m_geometryDataBuffer.buffer;
        geoInfo.offset = 0;
        geoInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet w4{};
        w4.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w4.dstSet = device->m_rtDescriptorSet;
        w4.dstBinding = 4;
        w4.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w4.descriptorCount = 1;
        w4.pBufferInfo = &geoInfo;
        vkUpdateDescriptorSets(device->m_device, 1, &w4, 0, nullptr);
    }
    return true;
}

bool refreshVulkanInstanceDataBinding(VulkanRT::VulkanDevice* device,
                                      const std::vector<VulkanRT::TLASInstance>& instances) {
    if (!device) return false;

    std::vector<VulkanRT::VkInstanceData> instData;
    instData.reserve(instances.size());
    for (const auto& vi : instances) {
        VulkanRT::VkInstanceData d;
        d.materialIndex = vi.materialIndex;
        d.blasIndex = vi.blasIndex;
        instData.push_back(d);
    }

    if (device->m_instanceDataBuffer.buffer) {
        device->destroyBuffer(device->m_instanceDataBuffer);
    }

    if (instData.empty()) return true;

    VulkanRT::BufferCreateInfo ci;
    ci.size = static_cast<uint64_t>(instData.size()) * sizeof(VulkanRT::VkInstanceData);
    ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
    ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
    ci.initialData = instData.data();
    device->m_instanceDataBuffer = device->createBuffer(ci);
    if (!device->m_instanceDataBuffer.buffer) return false;

    if (device->m_rtDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo instInfo{};
        instInfo.buffer = device->m_instanceDataBuffer.buffer;
        instInfo.offset = 0;
        instInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet w5{};
        w5.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w5.dstSet = device->m_rtDescriptorSet;
        w5.dstBinding = 5;
        w5.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w5.descriptorCount = 1;
        w5.pBufferInfo = &instInfo;
        vkUpdateDescriptorSets(device->m_device, 1, &w5, 0, nullptr);
    }
    return true;
}
}

extern "C" FARPROC WINAPI DelayLoadFailureHook(unsigned int dliNotify, PDelayLoadInfo pdli) {
    if (!pdli) return nullptr;

    if (dliNotify == dliFailLoadLib) {
        // Library failed to load; retry with secure search (no CWD/root probing).
        try {
            SCENE_LOG_WARN(std::string("[DelayLoad] Failed to load DLL: ") + pdli->szDll);
        } catch(...) { /* avoid throwing from hook */ }
        HMODULE h = Platform::Dll::loadModuleWithPolicy(pdli->szDll, Platform::Dll::DllCategory::Auto, true);
        if (h) {
            return (FARPROC)h; // return module handle per delayimp contract for dliFailLoadLib
        }
        return nullptr;
    }

    if (dliNotify == dliFailGetProc) {
        // Library loaded but GetProcAddress failed. Log missing proc and try to resolve manually.
        const bool importByName = pdli->dlp.fImportByName != 0;
        const char* procName = (importByName && pdli->dlp.szProcName) ? pdli->dlp.szProcName : "<ordinal>";
        try {
            SCENE_LOG_WARN(std::string("[DelayLoad] Failed to resolve proc: ") + procName + " in " + pdli->szDll);
        } catch(...) { }
        HMODULE h = GetModuleHandleA(pdli->szDll);
        if (!h) {
            // If module not present, try to load it first
            h = Platform::Dll::loadModuleWithPolicy(pdli->szDll, Platform::Dll::DllCategory::Auto, true);
            if (!h) return nullptr;
        }
        FARPROC proc = nullptr;
        if (importByName && pdli->dlp.szProcName) {
            proc = GetProcAddress(h, pdli->dlp.szProcName);
        } else {
            proc = GetProcAddress(h, MAKEINTRESOURCEA(pdli->dlp.dwOrdinal));
        }
        return proc;
    }

    return nullptr;
}

// Install delay-load failure hook using the supported linker-time definition.
// This avoids writing into runtime CRT hook storage from a static constructor.
ExternC const PfnDliHook __pfnDliFailureHook2 = DelayLoadFailureHook;

// ============================================================================
#include <vulkan/vulkan_win32.h>
// Debug Callback & Logging
// ============================================================================

#include <sstream>
#include <SpotLight.h>
#include <AreaLight.h>




// Structs moved to VulkanBackend.h for namespace consistency

static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    (void)type; (void)pUserData;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        VK_ERROR() << "[Vulkan] " << pCallbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}
namespace {
    constexpr bool kRasterFrustumCullingEnabled = false;
    constexpr uint32_t kMaterialPreviewTextureMaxDimension = 2048;

    inline bool matchesNodeNameForInstance(const std::string& instanceNodeName, const std::string& queryNodeName) {
        if (queryNodeName.empty() || instanceNodeName.empty()) return false;
        if (instanceNodeName == queryNodeName) return true;
        const std::string matPrefix = queryNodeName + "_mat_";
        return instanceNodeName.rfind(matPrefix, 0) == 0;
    }

    inline void signalVulkanMemoryPressure(VkResult result, const char* where) {
        if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
            result == VK_ERROR_OUT_OF_HOST_MEMORY ||
            result == VK_ERROR_MEMORY_MAP_FAILED) {
            g_vulkan_trim_recreate_requested.store(true, std::memory_order_release);
            SCENE_LOG_WARN(std::string("[Vulkan] Memory pressure signaled at ")
                           + (where ? where : "unknown")
                           + ". Safe backend recreate requested.");
        }
    }

    // IEEE 754 half -> float conversion for VK_FORMAT_R16G16B16A16_SFLOAT readback.
    inline float halfToFloat(uint16_t h) {
        const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
        const uint32_t exp = (h >> 10) & 0x1Fu;
        const uint32_t mant = h & 0x03FFu;
        uint32_t bits = 0;

        if (exp == 0) {
            if (mant == 0) {
                bits = sign; // zero
            } else {
                // subnormal
                int e = -14;
                uint32_t m = mant;
                while ((m & 0x0400u) == 0u) { m <<= 1; --e; }
                m &= 0x03FFu;
                bits = sign | (uint32_t)(e + 127) << 23 | (m << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000u | (mant << 13); // inf/nan
        } else {
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }

        float out;
        std::memcpy(&out, &bits, sizeof(float));
        return out;
    }

    inline uint8_t linearToSRGB8Fast(float c) {
        constexpr int kLutSize = 4096;
        static bool initialized = false;
        static uint8_t lut[kLutSize];
        if (!initialized) {
            for (int i = 0; i < kLutSize; ++i) {
                const float x = (float)i / (float)(kLutSize - 1);
                const float srgb = (x <= 0.0031308f)
                    ? (12.92f * x)
                    : (1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f);
                lut[i] = (uint8_t)std::clamp((int)std::lround(srgb * 255.0f), 0, 255);
            }
            initialized = true;
        }

        c = std::clamp(c, 0.0f, 1.0f);
        const int idx = std::clamp((int)std::lround(c * (float)(kLutSize - 1)), 0, kLutSize - 1);
        return lut[idx];
    }

    inline uint32_t bytesPerPixelForFormat(VkFormat format) {
        switch (format) {
            case VK_FORMAT_R8_UNORM:
            case VK_FORMAT_R8_SRGB:
                return 1;
            case VK_FORMAT_R8G8_UNORM:
            case VK_FORMAT_R8G8_SRGB:
                return 2;
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
                return 4;
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                return 8;
            case VK_FORMAT_R32_SFLOAT:
                return 4;
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                return 16;
            default:
                return 4;
        }
    }

    inline uint64_t estimateImageStorageBytes(uint32_t width, uint32_t height, VkFormat format,
                                               uint32_t mipLevels = 1) {
        const uint64_t w = std::max<uint32_t>(width, 1u);
        const uint64_t h = std::max<uint32_t>(height, 1u);
        uint64_t base = 0;
        switch (format) {
            case VK_FORMAT_BC4_UNORM_BLOCK:
                base = ((w + 3ull) / 4ull) * ((h + 3ull) / 4ull) * 8ull;
                break;
            case VK_FORMAT_BC5_UNORM_BLOCK:
            case VK_FORMAT_BC7_UNORM_BLOCK:
            case VK_FORMAT_BC7_SRGB_BLOCK:
                base = ((w + 3ull) / 4ull) * ((h + 3ull) / 4ull) * 16ull;
                break;
            default:
                base = w * h * bytesPerPixelForFormat(format);
                break;
        }
        // Full mip chain is ~4/3 × base; partial chains are proportionally less.
        if (mipLevels <= 1) return base;
        // Approximate: sum of geometric series 1 + 1/4 + 1/16 + ... capped at mipLevels
        uint64_t total = base;
        uint64_t level = base;
        for (uint32_t i = 1; i < mipLevels && level > 4; ++i) {
            level /= 4;
            total += level;
        }
        return total;
    }

    inline uint32_t calcMipLevels(uint32_t width, uint32_t height) {
        uint32_t maxDim = std::max(width, height);
        if (maxDim == 0) return 1;
        uint32_t levels = 1;
        while (maxDim > 1) { maxDim >>= 1; ++levels; }
        return levels;
    }

    inline bool isViewportTextureOwner(const char* ownerScope) {
        return ownerScope && std::strcmp(ownerScope, "VulkanViewportBackend") == 0;
    }

    inline void fitWithinMaxDimension(uint32_t srcW, uint32_t srcH, uint32_t maxDim,
                                      uint32_t& outW, uint32_t& outH) {
        outW = srcW;
        outH = srcH;
        if (srcW == 0 || srcH == 0 || maxDim == 0) return;
        const uint32_t largest = std::max(srcW, srcH);
        if (largest <= maxDim) return;

        const double scale = static_cast<double>(maxDim) / static_cast<double>(largest);
        outW = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<double>(srcW) * scale)));
        outH = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<double>(srcH) * scale)));
    }

    inline std::vector<uint8_t> resizeLdrBilinear(const std::vector<uint8_t>& src,
                                                  uint32_t srcW, uint32_t srcH,
                                                  uint32_t dstW, uint32_t dstH,
                                                  uint32_t channels) {
        if (srcW == 0 || srcH == 0 || dstW == 0 || dstH == 0 || channels == 0 ||
            (srcW == dstW && srcH == dstH)) {
            return src;
        }
        std::vector<uint8_t> dst(static_cast<size_t>(dstW) * dstH * channels);
        const float scaleX = static_cast<float>(srcW) / static_cast<float>(dstW);
        const float scaleY = static_cast<float>(srcH) / static_cast<float>(dstH);
        for (uint32_t y = 0; y < dstH; ++y) {
            const float srcY = std::clamp((static_cast<float>(y) + 0.5f) * scaleY - 0.5f,
                                          0.0f,
                                          static_cast<float>(srcH - 1u));
            const uint32_t y0 = static_cast<uint32_t>(std::floor(srcY));
            const uint32_t y1 = std::min(srcH - 1u, y0 + 1u);
            const float ty = srcY - static_cast<float>(y0);
            for (uint32_t x = 0; x < dstW; ++x) {
                const size_t dstIdx = (static_cast<size_t>(y) * dstW + x) * channels;
                const float srcX = std::clamp((static_cast<float>(x) + 0.5f) * scaleX - 0.5f,
                                              0.0f,
                                              static_cast<float>(srcW - 1u));
                const uint32_t x0 = static_cast<uint32_t>(std::floor(srcX));
                const uint32_t x1 = std::min(srcW - 1u, x0 + 1u);
                const float tx = srcX - static_cast<float>(x0);

                const size_t idx00 = (static_cast<size_t>(y0) * srcW + x0) * channels;
                const size_t idx10 = (static_cast<size_t>(y0) * srcW + x1) * channels;
                const size_t idx01 = (static_cast<size_t>(y1) * srcW + x0) * channels;
                const size_t idx11 = (static_cast<size_t>(y1) * srcW + x1) * channels;
                for (uint32_t c = 0; c < channels; ++c) {
                    const float top = static_cast<float>(src[idx00 + c]) +
                                      (static_cast<float>(src[idx10 + c]) - static_cast<float>(src[idx00 + c])) * tx;
                    const float bottom = static_cast<float>(src[idx01 + c]) +
                                         (static_cast<float>(src[idx11 + c]) - static_cast<float>(src[idx01 + c])) * tx;
                    const int value = static_cast<int>(std::lround(top + (bottom - top) * ty));
                    dst[dstIdx + c] = static_cast<uint8_t>(std::clamp(value, 0, 255));
                }
            }
        }
        return dst;
    }

    #pragma pack(push, 1)
    struct DDS_PIXELFORMAT {
        uint32_t size;
        uint32_t flags;
        uint32_t fourCC;
        uint32_t rgbBitCount;
        uint32_t rBitMask;
        uint32_t gBitMask;
        uint32_t bBitMask;
        uint32_t aBitMask;
    };

    struct DDS_HEADER {
        uint32_t size;
        uint32_t flags;
        uint32_t height;
        uint32_t width;
        uint32_t pitchOrLinearSize;
        uint32_t depth;
        uint32_t mipMapCount;
        uint32_t reserved1[11];
        DDS_PIXELFORMAT ddspf;
        uint32_t caps;
        uint32_t caps2;
        uint32_t caps3;
        uint32_t caps4;
        uint32_t reserved2;
    };

    struct DDS_HEADER_DXT10 {
        uint32_t dxgiFormat;
        uint32_t resourceDimension;
        uint32_t miscFlag;
        uint32_t arraySize;
        uint32_t miscFlags2;
    };
    #pragma pack(pop)

    constexpr uint32_t makeFourCC(char a, char b, char c, char d) {
        return (uint32_t)(uint8_t)a |
               ((uint32_t)(uint8_t)b << 8) |
               ((uint32_t)(uint8_t)c << 16) |
               ((uint32_t)(uint8_t)d << 24);
    }

    enum class DDSCompressedSemantic : uint8_t {
        Unsupported = 0,
        BC4,
        BC5,
        BC7
    };

    struct DDSCompressedPayload {
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<uint8_t> bytes;
    };

    bool loadCompressedDDSFile(
        const std::filesystem::path& ddsPath,
        TextureCompressionTarget desiredTarget,
        bool srgb,
        DDSCompressedPayload& outPayload)
    {
        namespace fs = std::filesystem;
        if (!fs::exists(ddsPath)) return false;

        std::ifstream in(ddsPath, std::ios::binary);
        if (!in) return false;

        uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (!in || magic != makeFourCC('D', 'D', 'S', ' ')) return false;

        DDS_HEADER header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in || header.size != 124 || header.ddspf.size != 32) return false;

        DDSCompressedSemantic semantic = DDSCompressedSemantic::Unsupported;
        VkFormat vkFormat = VK_FORMAT_UNDEFINED;

        if (header.ddspf.fourCC == makeFourCC('D', 'X', '1', '0')) {
            DDS_HEADER_DXT10 dx10{};
            in.read(reinterpret_cast<char*>(&dx10), sizeof(dx10));
            if (!in) return false;

            switch (dx10.dxgiFormat) {
                case 80: semantic = DDSCompressedSemantic::BC4; vkFormat = VK_FORMAT_BC4_UNORM_BLOCK; break;
                case 83: semantic = DDSCompressedSemantic::BC5; vkFormat = VK_FORMAT_BC5_UNORM_BLOCK; break;
                case 98: semantic = DDSCompressedSemantic::BC7; vkFormat = VK_FORMAT_BC7_UNORM_BLOCK; break;
                case 99: semantic = DDSCompressedSemantic::BC7; vkFormat = VK_FORMAT_BC7_SRGB_BLOCK; break;
                default: return false;
            }
        } else {
            switch (header.ddspf.fourCC) {
                case makeFourCC('A', 'T', 'I', '1'):
                case makeFourCC('B', 'C', '4', 'U'):
                    semantic = DDSCompressedSemantic::BC4;
                    vkFormat = VK_FORMAT_BC4_UNORM_BLOCK;
                    break;
                case makeFourCC('A', 'T', 'I', '2'):
                case makeFourCC('B', 'C', '5', 'U'):
                    semantic = DDSCompressedSemantic::BC5;
                    vkFormat = VK_FORMAT_BC5_UNORM_BLOCK;
                    break;
                default:
                    return false;
            }
        }

        if ((desiredTarget == TextureCompressionTarget::BC4 && semantic != DDSCompressedSemantic::BC4) ||
            (desiredTarget == TextureCompressionTarget::BC5 && semantic != DDSCompressedSemantic::BC5) ||
            (desiredTarget == TextureCompressionTarget::BC7 && semantic != DDSCompressedSemantic::BC7)) {
            return false;
        }

        if (semantic == DDSCompressedSemantic::BC7) {
            vkFormat = srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
        }

        const uint64_t expectedBytes = estimateImageStorageBytes(header.width, header.height, vkFormat);
        if (expectedBytes == 0) return false;

        outPayload.format = vkFormat;
        outPayload.width = header.width;
        outPayload.height = header.height;
        outPayload.bytes.resize((size_t)expectedBytes);
        in.read(reinterpret_cast<char*>(outPayload.bytes.data()), static_cast<std::streamsize>(expectedBytes));
        if (!in) {
            outPayload = {};
            return false;
        }

        return true;
    }
}
namespace VulkanRT {

// ============================================================================
// VulkanDevice Implementation
// ============================================================================

VulkanDevice::VulkanDevice() {
    // Initialize LUT image array to ensure no garbage data
    for (int i = 0; i < 4; i++) {
        m_lutImages[i].image = VK_NULL_HANDLE;
        m_lutImages[i].view = VK_NULL_HANDLE;
        m_lutImages[i].sampler = VK_NULL_HANDLE;
        m_lutImages[i].memory = VK_NULL_HANDLE;
    }
}

VulkanDevice::~VulkanDevice() {
    shutdown();
}

bool VulkanDevice::initialize(bool preferHardwareRT, bool validationLayers) {
    VK_INFO() << "[VulkanDevice] Initializing..." << std::endl;

    if (!createInstance(validationLayers)) return false;
    if (validationLayers) setupDebugMessenger();
    if (!selectPhysicalDevice(preferHardwareRT)) return false;
    if (!createLogicalDevice(preferHardwareRT)) return false;
    if (!createCommandPool()) return false;
    detectCapabilities();
    if (!createDescriptorPool()) return false;

    if (hasHardwareRT()) {
        loadRayTracingFunctions();
        VK_INFO() << "[VulkanDevice] Hardware RT enabled ("
                  << (m_capabilities.rtMode == RayTracingMode::HARDWARE_KHR ? "KHR" : "NV")
                  << ")" << std::endl;
    } else {
        VK_INFO() << "[VulkanDevice] No hardware RT, using compute fallback" << std::endl;
    }

    VK_INFO() << "[VulkanDevice] Ready: " << m_capabilities.deviceName
              << " | VRAM: " << (m_capabilities.dedicatedVRAM / (1024*1024)) << " MB" << std::endl;
    VK_INFO() << "[VulkanDevice] Texture compression support"
              << " | BC4: " << (m_capabilities.supportsBC4 ? "yes" : "no")
              << " | BC5: " << (m_capabilities.supportsBC5 ? "yes" : "no")
              << " | BC7: " << (m_capabilities.supportsBC7 ? "yes" : "no") << std::endl;
    return true;
}

void VulkanDevice::shutdown() {
    if (m_device) {
        vkDeviceWaitIdle(m_device);

        {
            std::lock_guard<std::mutex> lock(m_rtDescriptorMutex);
            m_pendingTextureDescriptors.clear();
            m_rtDescriptorSet = VK_NULL_HANDLE;
        }

        // Destroy atmosphere LUT images owned by VulkanDevice.
        for (int i = 0; i < 4; ++i) {
            if (m_lutImages[i].image || m_lutImages[i].view || m_lutImages[i].memory || m_lutImages[i].sampler) {
                destroyImage(m_lutImages[i]);
            }
        }

        // Destroy skinning compute resources (persistent across frames).
        if (m_skinningPipeline) {
            vkDestroyPipeline(m_device, m_skinningPipeline, nullptr);
            m_skinningPipeline = VK_NULL_HANDLE;
        }
        if (m_skinningPipelineLayout) {
            vkDestroyPipelineLayout(m_device, m_skinningPipelineLayout, nullptr);
            m_skinningPipelineLayout = VK_NULL_HANDLE;
        }
        if (m_skinningDescLayout) {
            vkDestroyDescriptorSetLayout(m_device, m_skinningDescLayout, nullptr);
            m_skinningDescLayout = VK_NULL_HANDLE;
        }
        if (m_skinningDescPool) {
            vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr);
            m_skinningDescPool = VK_NULL_HANDLE;
        }

        // Drain + destroy per-slot fences/cmd buffers before tearing down the pipeline
        // and descriptor pool (the descriptor set is referenced by in-flight submissions).
        for (uint32_t i = 0; i < kFrameSlotCount; ++i) {
            if (m_frameSlots[i].fence && m_frameSlots[i].everSubmitted) {
                vkWaitForFences(m_device, 1, &m_frameSlots[i].fence, VK_TRUE, UINT64_MAX);
            }
        }
        destroyFrameSlots();
        destroyDenoiserCopySlot();

        // Destroy tonemap compute resources (persistent across frames).
        if (m_tonemapPipeline)       { vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr); m_tonemapPipeline = VK_NULL_HANDLE; }
        if (m_tonemapPipelineLayout) { vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE; }
        if (m_tonemapDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE; }
        if (m_tonemapDescPool)       { vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE; }

        if (m_stylizePipeline)       { vkDestroyPipeline(m_device, m_stylizePipeline, nullptr); m_stylizePipeline = VK_NULL_HANDLE; }
        if (m_stylizePipelineLayout) { vkDestroyPipelineLayout(m_device, m_stylizePipelineLayout, nullptr); m_stylizePipelineLayout = VK_NULL_HANDLE; }
        if (m_stylizeDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE; }
        if (m_stylizeDescPool)       { vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE; }
        m_tonemapDescSet = VK_NULL_HANDLE;

        // Destroy atmosphere LUT compute resources.
        if (m_atmosphereLutPipeline)       { vkDestroyPipeline(m_device, m_atmosphereLutPipeline, nullptr); m_atmosphereLutPipeline = VK_NULL_HANDLE; }
        if (m_atmosphereLutPipelineLayout) { vkDestroyPipelineLayout(m_device, m_atmosphereLutPipelineLayout, nullptr); m_atmosphereLutPipelineLayout = VK_NULL_HANDLE; }
        if (m_atmosphereLutDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE; }
        if (m_atmosphereLutDescPool)       { vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE; }
        m_atmosphereLutDescSet = VK_NULL_HANDLE;
        if (m_atmosphereLutParamsBuffer.buffer) {
            destroyBuffer(m_atmosphereLutParamsBuffer);
        }

        // Release any batched command buffer/scratch that may still be alive.
        if (m_batchBLASCmd != VK_NULL_HANDLE && m_commandPool != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_batchBLASCmd);
            m_batchBLASCmd = VK_NULL_HANDLE;
        }
        if (m_batchScratchBuffer.buffer) {
            destroyBuffer(m_batchScratchBuffer);
        }
        m_inBatchedBLASBuild = false;
        m_batchBLASCount = 0;
        m_batchBLASInCurrentCmd = 0;

        // Destroy BLAS
        for (auto& blas : m_blasList) {
            if (blas.accel && fpDestroyAccelerationStructureKHR) {
                fpDestroyAccelerationStructureKHR(m_device, blas.accel, nullptr);
            }
            // Backing buffer for AS data
            destroyBuffer(blas.buffer);
            
            // Fixed memory leak: attribute buffers were being leaked.
            // Collect all unique non-null buffer/memory handles to avoid double free
            // (since multiple handles may point to the same combined geometry buffer).
            std::set<VkBuffer> destroyedBuffers;
            auto safeDestroy = [&](VulkanRT::BufferHandle& bh) {
                if (bh.buffer && destroyedBuffers.find(bh.buffer) == destroyedBuffers.end()) {
                    vkDestroyBuffer(m_device, bh.buffer, nullptr);
                    if (bh.memory) vkFreeMemory(m_device, bh.memory, nullptr);
                    destroyedBuffers.insert(bh.buffer);
                }
                bh = {};
            };
            // Device-resident geometry (Phase 3d) is owned by the CC system, not the BLAS;
            // its attribute handles carry only addresses (null VkBuffer), so skip them.
            if (!blas.externalGeometry) {
                safeDestroy(blas.vertexBuffer);
                safeDestroy(blas.normalBuffer);
                safeDestroy(blas.uvBuffer);
                safeDestroy(blas.indexBuffer);
                safeDestroy(blas.materialIndexBuffer);
            }
            safeDestroy(blas.baseVertexBuffer);
            safeDestroy(blas.baseNormalBuffer);
            safeDestroy(blas.boneIndexBuffer);
            safeDestroy(blas.boneWeightBuffer);
        }
        m_blasList.clear();

        // Destroy TLAS
        if (m_tlas.accel && fpDestroyAccelerationStructureKHR) {
            fpDestroyAccelerationStructureKHR(m_device, m_tlas.accel, nullptr);
        }
        if (m_tlas.buffer.buffer) {
            vkDestroyBuffer(m_device, m_tlas.buffer.buffer, nullptr);
            vkFreeMemory(m_device, m_tlas.buffer.memory, nullptr);
        }
        if (m_tlasInstanceBuffer.buffer) {
            destroyBuffer(m_tlasInstanceBuffer);
        }
        if (m_tlasScratchBuffer.buffer) {
            destroyBuffer(m_tlasScratchBuffer);
        }

        // Destroy RT pipeline
        if (m_rtPipeline) vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
        if (m_rtPipelineLayout) vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
        if (m_rtDescriptorSetLayout) vkDestroyDescriptorSetLayout(m_device, m_rtDescriptorSetLayout, nullptr);
        
        destroyBuffer(m_sbtBuffer);
        destroyBuffer(m_photonGridBuffer);
        destroyBuffer(m_photonVolGridBuffer);
        destroyBuffer(m_materialBuffer);
        destroyBuffer(m_lightBuffer);
        destroyBuffer(m_geometryDataBuffer);
        destroyBuffer(m_instanceDataBuffer);
        destroyBuffer(m_worldBuffer);
        destroyBuffer(m_volumeBuffer);
        destroyBuffer(m_hairMaterialBuffer);
        destroyBuffer(m_hairSegmentBuffer);
        destroyBuffer(m_terrainLayerBuffer);

        m_rtPipelineReady = false;

        // Destroy compute pipelines
        for (auto& p : m_pipelines) vkDestroyPipeline(m_device, p, nullptr);
        for (auto& pl : m_pipelineLayouts) vkDestroyPipelineLayout(m_device, pl, nullptr);
        for (auto& dsl : m_descriptorSetLayouts) vkDestroyDescriptorSetLayout(m_device, dsl, nullptr);
        m_pipelines.clear();
        m_pipelineLayouts.clear();
        m_descriptorSetLayouts.clear();
        m_activeDescriptorSets.clear();

        if (m_descriptorPool) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        if (m_commandPool) vkDestroyCommandPool(m_device, m_commandPool, nullptr);

        // The process-wide shared mesh compute backend (subdivision, CC device-resident)
        // uses THIS VkDevice. Tear it down now, while the device is still valid, and clear
        // the context — otherwise a later Vulkan re-init would leave the cached backend
        // pointing at this freed device and crash in createBuffer on the next mesh op
        // (OptiX -> Vulkan -> bake access violation).
        RayTrophiSim::releaseSharedMeshComputeBackend();
        if (g_vulkan_sim_compute_ctx.device == static_cast<void*>(m_device)) {
            g_vulkan_sim_compute_ctx = RayTrophiSim::SimulationComputeVulkanContext{};
        }

        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }

    if (m_debugMessenger && m_instance) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func) func(m_instance, m_debugMessenger, nullptr);
        m_debugMessenger = VK_NULL_HANDLE;
    }

    if (m_instance) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

// ========================================================================
// Instance Creation
// ========================================================================

bool VulkanDevice::createInstance(bool validationLayers) {
    // Ensure Vulkan loader is actually loadable.
    // GetModuleHandle only checks if the module is already loaded in this process,
    // which can cause false negatives on systems where vulkan-1.dll exists but wasn't loaded yet.
    static HMODULE s_vulkanLoader = nullptr;
    if (!s_vulkanLoader) {
        s_vulkanLoader = Platform::Dll::loadModuleWithPolicy("vulkan-1.dll", Platform::Dll::DllCategory::Driver, false);
    }
    if (!s_vulkanLoader) {
        VK_WARN() << "[VulkanDevice] Vulkan loader (vulkan-1.dll) not found or failed to load." << std::endl;
        return false;
    }
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RayTrophi Studio";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "RayTrophi Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> layers;
    if (validationLayers) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    std::vector<const char*> extensions;
    if (validationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = (uint32_t)layers.size();
    createInfo.ppEnabledLayerNames = layers.data();
    createInfo.enabledExtensionCount = (uint32_t)extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    // Resolve vkCreateInstance directly from loader to avoid sporadic delay-import thunk faults
    // during backend switch transitions.
    PFN_vkGetInstanceProcAddr pfnGetInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(s_vulkanLoader, "vkGetInstanceProcAddr"));
    if (!pfnGetInstanceProcAddr) {
        // Handle stale/invalid cached module handles after backend switching.
        s_vulkanLoader = Platform::Dll::loadModuleWithPolicy("vulkan-1.dll", Platform::Dll::DllCategory::Driver, false);
        if (s_vulkanLoader) {
            pfnGetInstanceProcAddr =
                reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(s_vulkanLoader, "vkGetInstanceProcAddr"));
        }
    }
    if (!pfnGetInstanceProcAddr) {
        // Final fallback: use linked symbol if available.
        pfnGetInstanceProcAddr = ::vkGetInstanceProcAddr;
    }
    if (!pfnGetInstanceProcAddr) {
        VK_ERROR() << "[VulkanDevice] Failed to resolve vkGetInstanceProcAddr from loader." << std::endl;
        return false;
    }

    PFN_vkCreateInstance pfnCreateInstance =
        reinterpret_cast<PFN_vkCreateInstance>(pfnGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!pfnCreateInstance) {
        pfnCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(GetProcAddress(s_vulkanLoader, "vkCreateInstance"));
    }
    if (!pfnCreateInstance) {
        VK_ERROR() << "[VulkanDevice] Failed to resolve vkCreateInstance from loader." << std::endl;
        return false;
    }

    VkResult result = pfnCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] vkCreateInstance failed: " << result << std::endl;
        return false;
    }
    return true;
}

// ========================================================================
// Physical Device Selection
// ========================================================================

bool VulkanDevice::selectPhysicalDevice(bool preferHardwareRT) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        VK_ERROR() << "[VulkanDevice] No Vulkan-capable GPU found!" << std::endl;
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    // Score each device
    int bestScore = -1;
    for (auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 100;

        // Check RT extension support
        if (preferHardwareRT) {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());

            for (auto& ext : exts) {
                if (strcmp(ext.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0)
                    score += 500;
                if (strcmp(ext.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0)
                    score += 500;
            }
        }

        // Prefer more VRAM
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(dev, &memProps);
        for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
            if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                score += (int)(memProps.memoryHeaps[i].size / (1024 * 1024 * 100)); // +1 per 100MB
            }
        }

        VK_INFO() << "[VulkanDevice] GPU: " << props.deviceName << " (score: " << score << ")" << std::endl;

        if (score > bestScore) {
            bestScore = score;
            m_physicalDevice = dev;
        }
    }

    if (!m_physicalDevice) {
        VK_ERROR() << "[VulkanDevice] No suitable GPU found!" << std::endl;
        return false;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    VK_INFO() << "[VulkanDevice] Selected: " << props.deviceName << std::endl;
    return true;
}

// ========================================================================
// Logical Device Creation
// ========================================================================

bool VulkanDevice::createLogicalDevice(bool preferHardwareRT) {
    // Find compute queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());

    m_computeQueueFamily = UINT32_MAX;
    uint32_t graphicsComputeFamily = UINT32_MAX;
    uint32_t computeOnlyFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (!(queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            if (graphicsComputeFamily == UINT32_MAX) graphicsComputeFamily = i;
        } else if (computeOnlyFamily == UINT32_MAX) {
            computeOnlyFamily = i;
        }
    }
    // Interactive viewport needs graphics commands, so prefer a queue family that
    // supports both compute and graphics whenever one exists.
    if (graphicsComputeFamily != UINT32_MAX) {
        m_computeQueueFamily = graphicsComputeFamily;
        m_queueSupportsGraphics = true;
    } else {
        m_computeQueueFamily = computeOnlyFamily;
        m_queueSupportsGraphics = false;
    }

    if (m_computeQueueFamily == UINT32_MAX) {
        VK_ERROR() << "[VulkanDevice] No compute queue family!" << std::endl;
        return false;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = m_computeQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // Check which RT extensions are available
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, availableExts.data());

    auto hasExtension = [&](const char* name) {
        return std::any_of(availableExts.begin(), availableExts.end(),
            [name](const VkExtensionProperties& ext) { return strcmp(ext.extensionName, name) == 0; });
    };

    std::vector<const char*> deviceExtensions;

    // Buffer device address (required for RT)
    bool hasBDA = hasExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    if (hasBDA) deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

    // Descriptor indexing
    bool hasDescIdx = hasExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    if (hasDescIdx) deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);

    // Shader float atomics (VK_EXT_shader_atomic_float). The fluid P2G scatter
    // and density splat compute kernels do atomicAdd() on float SSBOs; without
    // ENABLING this at device creation that atomicAdd is undefined behaviour →
    // garbage grid accumulation → wrong velocity field (the CPU-vs-Vulkan solve
    // divergence). Push the extension when the device exposes it; the matching
    // feature bit is verified below before it is actually enabled.
    bool hasAtomicFloat = hasExtension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    if (hasAtomicFloat) deviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

    // External-memory interop (Vulkan→CUDA for OIDN GPU-direct denoise).
    // Requires VK_KHR_external_memory (core in 1.1, but still named) +
    // VK_KHR_external_memory_win32 on Windows. Both are optional; capability
    // flag on m_capabilities controls whether getDenoiserFrameGPU can succeed.
#ifdef _WIN32
    const bool hasExtMem    = hasExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    const bool hasExtMemW32 = hasExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    if (hasExtMem && hasExtMemW32) {
        deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        m_capabilities.supportsExternalMemoryWin32 = true;
    }
#endif

    // Deferred host operations (required by accel struct)
    bool hasDeferredOps = hasExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    if (hasDeferredOps) deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

    // Ray tracing extensions
    bool hasAccelStruct = hasExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    bool hasRTPipeline = hasExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    bool hasRayQuery = hasExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    bool hasSPIRV14 = hasExtension(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    bool hasShaderFloatCtrl = hasExtension(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);

    if (preferHardwareRT && hasAccelStruct && hasRTPipeline && hasBDA && hasDeferredOps) {
        deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        if (hasRayQuery) deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        if (hasSPIRV14) deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
        if (hasShaderFloatCtrl) deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
        m_capabilities.rtMode = RayTracingMode::HARDWARE_KHR;
        m_capabilities.supportsRayQuery = hasRayQuery;
        m_capabilities.supportsBufferDeviceAddress = true;
    } else {
        m_capabilities.rtMode = RayTracingMode::COMPUTE;
    }

    // Query supported features first. Some drivers expose the extension string
    // but still reject vkCreateDevice if unsupported feature bits are forced on.
    VkPhysicalDeviceFeatures2 supportedFeatures{};
    supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceBufferDeviceAddressFeatures supportedBdaFeatures{};
    supportedBdaFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR supportedAccelFeatures{};
    supportedAccelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR supportedRtPipelineFeatures{};
    supportedRtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;

    VkPhysicalDeviceDescriptorIndexingFeatures supportedDescIdxFeatures{};
    supportedDescIdxFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supportedAtomicFloatFeatures{};
    supportedAtomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

    supportedFeatures.pNext = &supportedBdaFeatures;
    supportedBdaFeatures.pNext = &supportedAccelFeatures;
    supportedAccelFeatures.pNext = &supportedRtPipelineFeatures;
    supportedRtPipelineFeatures.pNext = &supportedDescIdxFeatures;
    supportedDescIdxFeatures.pNext = &supportedAtomicFloatFeatures;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &supportedFeatures);

    const bool canUseBDA = hasBDA && supportedBdaFeatures.bufferDeviceAddress == VK_TRUE;
    // All three feature bits are required by the material-preview pipeline:
    //   - runtimeDescriptorArray              → `sampler2D textures[]`
    //   - shaderSampledImageArrayNonUniformIndexing → `nonuniformEXT(...)`
    //   - descriptorBindingPartiallyBound     → sparse array without writing every slot
    // Older drivers (observed: GTX 850M / Maxwell 1 with nvoglv64.dll) expose
    // the base extension but not partiallyBound; using the PARTIALLY_BOUND
    // flag without the feature causes a null-descriptor dereference inside
    // the ICD on first draw. Require all three together.
    const bool canUseDescIdx = hasDescIdx &&
        supportedDescIdxFeatures.runtimeDescriptorArray == VK_TRUE &&
        supportedDescIdxFeatures.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
        supportedDescIdxFeatures.descriptorBindingPartiallyBound == VK_TRUE;
    const bool canUseAccelStruct = hasAccelStruct && supportedAccelFeatures.accelerationStructure == VK_TRUE;
    const bool canUseRTPipeline = hasRTPipeline && supportedRtPipelineFeatures.rayTracingPipeline == VK_TRUE;
    const bool canUseSamplerAnisotropy = supportedFeatures.features.samplerAnisotropy == VK_TRUE;
    // Only the buffer (SSBO) float-atomic-add variant is needed by the fluid
    // kernels — they do not use shared-memory float atomics.
    const bool canUseAtomicFloat = hasAtomicFloat &&
        supportedAtomicFloatFeatures.shaderBufferFloat32AtomicAdd == VK_TRUE;
    if (hasAtomicFloat && !canUseAtomicFloat) {
        VK_WARN() << "[VulkanDevice] shader_atomic_float extension present but shaderBufferFloat32AtomicAdd is unsupported; fluid P2G stays on CPU." << std::endl;
    }
    if (!canUseAtomicFloat) {
        deviceExtensions.erase(
            std::remove(deviceExtensions.begin(), deviceExtensions.end(), VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME),
            deviceExtensions.end());
    }

    if (hasBDA && !canUseBDA) {
        VK_WARN() << "[VulkanDevice] BDA extension present but bufferDeviceAddress feature is unsupported." << std::endl;
    }
    if (hasDescIdx && !canUseDescIdx) {
        VK_WARN() << "[VulkanDevice] Descriptor indexing extension present but required feature bits are unsupported." << std::endl;
    }
    if (hasAccelStruct && !canUseAccelStruct) {
        VK_WARN() << "[VulkanDevice] Acceleration structure extension present but feature is unsupported." << std::endl;
    }
    if (hasRTPipeline && !canUseRTPipeline) {
        VK_WARN() << "[VulkanDevice] RT pipeline extension present but feature is unsupported." << std::endl;
    }

    if (!canUseBDA) {
        deviceExtensions.erase(
            std::remove(deviceExtensions.begin(), deviceExtensions.end(), VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME),
            deviceExtensions.end());
    }
    if (!canUseDescIdx) {
        deviceExtensions.erase(
            std::remove(deviceExtensions.begin(), deviceExtensions.end(), VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME),
            deviceExtensions.end());
    }

    if (m_capabilities.rtMode == RayTracingMode::HARDWARE_KHR && (!canUseBDA || !canUseAccelStruct || !canUseRTPipeline)) {
        VK_WARN() << "[VulkanDevice] Downgrading to compute mode because required RT feature bits are unavailable." << std::endl;
        deviceExtensions.erase(
            std::remove(deviceExtensions.begin(), deviceExtensions.end(), VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME),
            deviceExtensions.end());
        deviceExtensions.erase(
            std::remove(deviceExtensions.begin(), deviceExtensions.end(), VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME),
            deviceExtensions.end());
        deviceExtensions.erase(
            std::remove(deviceExtensions.begin(), deviceExtensions.end(), VK_KHR_RAY_QUERY_EXTENSION_NAME),
            deviceExtensions.end());
        deviceExtensions.erase(
            std::remove(deviceExtensions.begin(), deviceExtensions.end(), VK_KHR_SPIRV_1_4_EXTENSION_NAME),
            deviceExtensions.end());
        deviceExtensions.erase(
            std::remove(deviceExtensions.begin(), deviceExtensions.end(), VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME),
            deviceExtensions.end());
        m_capabilities.rtMode = RayTracingMode::COMPUTE;
    }

    // Features chain
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.features.samplerAnisotropy = canUseSamplerAnisotropy ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures{};
    bdaFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bdaFeatures.bufferDeviceAddress = canUseBDA ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{};
    accelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelFeatures.accelerationStructure = (m_capabilities.rtMode == RayTracingMode::HARDWARE_KHR && canUseAccelStruct) ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.rayTracingPipeline = (m_capabilities.rtMode == RayTracingMode::HARDWARE_KHR && canUseRTPipeline) ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceDescriptorIndexingFeatures descIdxFeatures{};
    descIdxFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    if (canUseDescIdx) {
        descIdxFeatures.runtimeDescriptorArray = VK_TRUE;
        descIdxFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        // Enable partial binding and update-after-bind so the material-preview
        // texture array works on any Vulkan 1.2+ device without requiring RT hardware.
        descIdxFeatures.descriptorBindingPartiallyBound =
            supportedDescIdxFeatures.descriptorBindingPartiallyBound;
      
        descIdxFeatures.descriptorBindingSampledImageUpdateAfterBind =
            supportedDescIdxFeatures.descriptorBindingSampledImageUpdateAfterBind;
    }

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
    atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    atomicFloatFeatures.shaderBufferFloat32AtomicAdd = canUseAtomicFloat ? VK_TRUE : VK_FALSE;

    // Build pNext chain conservatively.
    void** nextLink = &features2.pNext;
    if (canUseBDA) {
        *nextLink = &bdaFeatures;
        nextLink = &bdaFeatures.pNext;
    }
    if (m_capabilities.rtMode == RayTracingMode::HARDWARE_KHR && canUseAccelStruct) {
        *nextLink = &accelFeatures;
        nextLink = &accelFeatures.pNext;
    }
    if (m_capabilities.rtMode == RayTracingMode::HARDWARE_KHR && canUseRTPipeline) {
        *nextLink = &rtPipelineFeatures;
        nextLink = &rtPipelineFeatures.pNext;
    }
    if (canUseDescIdx) {
        *nextLink = &descIdxFeatures;
        nextLink = &descIdxFeatures.pNext;
    }
    if (canUseAtomicFloat) {
        *nextLink = &atomicFloatFeatures;
        nextLink = &atomicFloatFeatures.pNext;
    }
    *nextLink = nullptr;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = &features2;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    VkResult result = vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device);
    // Track which features were actually enabled on the created device.
    // detectCapabilities() runs later and only queries the PHYSICAL device —
    // that returns the device's "what it could do" not "what we enabled on it",
    // so we latch the real post-create state into capabilities here.
    bool enabledDescIdx = (result == VK_SUCCESS) && canUseDescIdx;
    bool enabledSamplerAnisotropy = (result == VK_SUCCESS) && canUseSamplerAnisotropy;
    bool enabledAtomicFloat = (result == VK_SUCCESS) && canUseAtomicFloat;
    if (result != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] vkCreateDevice failed: " << result << std::endl;
        // Log requested extensions for diagnostics
        VK_INFO() << "[VulkanDevice] Requested device extensions:" << std::endl;
        for (uint32_t i = 0; i < deviceCreateInfo.enabledExtensionCount; ++i) {
            VK_INFO() << "  " << deviceCreateInfo.ppEnabledExtensionNames[i] << std::endl;
        }

        // Attempt a safe fallback: try creating device without optional RT extensions/features
        VK_INFO() << "[VulkanDevice] Retrying device creation without RT extensions/features..." << std::endl;
        // Clear requested extensions and pNext chain
        deviceCreateInfo.enabledExtensionCount = 0;
        deviceCreateInfo.ppEnabledExtensionNames = nullptr;
        deviceCreateInfo.pNext = nullptr;

        VkResult fallback = vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device);
        if (fallback != VK_SUCCESS) {
            VK_ERROR() << "[VulkanDevice] Fallback vkCreateDevice also failed: " << fallback << std::endl;
            return false;
        }
        // Fallback succeeded — mark capabilities conservatively.
        // Fallback path passes nullptr pNext, so NO extension features are
        // enabled on the device — critically, descriptor indexing is OFF.
        // Without latching this, the material-preview pipeline would later
        // use runtimeDescriptorArray/PARTIALLY_BOUND against a device that
        // never enabled them, dereferencing null descriptors inside the ICD
        // (observed crash on GTX 850M: fault_addr=0x8 in nvoglv64.dll).
        m_capabilities.rtMode = RayTracingMode::COMPUTE;
        enabledDescIdx = false;
        enabledSamplerAnisotropy = false;
        // Fallback passes nullptr pNext → no atomic-float feature enabled either.
        // Must latch false so the sim compute backend keeps fluid P2G on CPU
        // instead of invoking undefined float-atomic behaviour.
        enabledAtomicFloat = false;
        VK_INFO() << "[VulkanDevice] Device created with fallback (no HW RT, descriptor indexing disabled). Continuing in compute mode." << std::endl;
    }
    // Latch enabled-at-create descriptor indexing state into capabilities.
    // detectCapabilities() must NOT overwrite this — it now preserves the flag.
    m_capabilities.supportsDescriptorIndexing = enabledDescIdx;
    m_capabilities.supportsSamplerAnisotropy = enabledSamplerAnisotropy;

    vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_computeQueue);

    // Expose device handles for the Vulkan simulation compute backend.
    g_vulkan_sim_compute_ctx.device           = static_cast<void*>(m_device);
    g_vulkan_sim_compute_ctx.physical_device  = static_cast<void*>(m_physicalDevice);
    g_vulkan_sim_compute_ctx.compute_queue    = static_cast<void*>(m_computeQueue);
    g_vulkan_sim_compute_ctx.queue_family_index = m_computeQueueFamily;
    // Latch whether float SSBO atomics were actually enabled (not just supported)
    // so the sim compute backend can safely run fluid P2G scatter / density splat
    // on the GPU. False on the fallback path → those kernels stay on CPU.
    g_vulkan_sim_compute_ctx.shader_atomic_float_enabled = enabledAtomicFloat;
    if (enabledAtomicFloat) {
        VK_INFO() << "[VulkanDevice] VK_EXT_shader_atomic_float enabled; fluid P2G/density can run on Vulkan GPU." << std::endl;
    }

    return true;
}

// ========================================================================
// Command & Descriptor Pool
// ========================================================================

bool VulkanDevice::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_computeQueueFamily;

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] Failed to create command pool" << std::endl;
        return false;
    }
    return true;
}

bool VulkanDevice::createDescriptorPool() {
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,              256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,                32 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     Backend::VULKAN_TEXTURE_CAPACITY },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,               32 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,   8 },
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT   // mevcut
        | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;    // eklendi
    poolInfo.maxSets = 32;
    poolInfo.poolSizeCount = hasHardwareRT() ? 5 : 4;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] Failed to create descriptor pool" << std::endl;
        return false;
    }
    return true;
}

// ========================================================================
// RT Function Pointers
// ========================================================================

void VulkanDevice::loadRayTracingFunctions() {
    if (!m_device) return;

    #define LOAD_VK_FUNC(name) fp##name = (PFN_vk##name)vkGetDeviceProcAddr(m_device, "vk" #name)
    LOAD_VK_FUNC(CreateAccelerationStructureKHR);
    LOAD_VK_FUNC(DestroyAccelerationStructureKHR);
    LOAD_VK_FUNC(CmdBuildAccelerationStructuresKHR);
    LOAD_VK_FUNC(GetAccelerationStructureBuildSizesKHR);
    LOAD_VK_FUNC(GetAccelerationStructureDeviceAddressKHR);
    LOAD_VK_FUNC(CmdTraceRaysKHR);
    LOAD_VK_FUNC(CreateRayTracingPipelinesKHR);
    LOAD_VK_FUNC(GetRayTracingShaderGroupHandlesKHR);
    LOAD_VK_FUNC(GetBufferDeviceAddressKHR);
    #undef LOAD_VK_FUNC

    VK_INFO() << "[VulkanDevice] RT functions loaded" << std::endl;
}

// ========================================================================
// Capability Detection
// ========================================================================

void VulkanDevice::detectCapabilities() {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);

    m_capabilities.deviceName = props.deviceName;
    m_capabilities.apiVersion = props.apiVersion;
    m_capabilities.driverVersion = props.driverVersion;
    m_capabilities.vendor = vendorFromID(props.vendorID);
    m_capabilities.maxSamplerAnisotropy = props.limits.maxSamplerAnisotropy;

    // Device UUID — needed to match the Vulkan physical device to a CUDA device
    // ordinal during external-memory interop (OIDN GPU-direct denoise).
    {
        VkPhysicalDeviceIDProperties idProps{};
        idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 props2uuid{};
        props2uuid.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2uuid.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2uuid);
        std::memcpy(m_capabilities.deviceUUID, idProps.deviceUUID, VK_UUID_SIZE);
        m_capabilities.hasDeviceUUID = true;
    }

    // Memory
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            m_capabilities.dedicatedVRAM += memProps.memoryHeaps[i].size;
        else
            m_capabilities.sharedSystemMemory += memProps.memoryHeaps[i].size;
    }

    // Compute limits
    m_capabilities.maxComputeWorkGroupSize[0] = props.limits.maxComputeWorkGroupSize[0];
    m_capabilities.maxComputeWorkGroupSize[1] = props.limits.maxComputeWorkGroupSize[1];
    m_capabilities.maxComputeWorkGroupSize[2] = props.limits.maxComputeWorkGroupSize[2];
    m_capabilities.maxComputeWorkGroupCount[0] = props.limits.maxComputeWorkGroupCount[0];
    m_capabilities.maxComputeWorkGroupCount[1] = props.limits.maxComputeWorkGroupCount[1];
    m_capabilities.maxComputeWorkGroupCount[2] = props.limits.maxComputeWorkGroupCount[2];

    // Subgroup size
    VkPhysicalDeviceSubgroupProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);
    m_capabilities.subgroupSize = subgroupProps.subgroupSize;

    // RT properties
    if (hasHardwareRT()) {
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
        rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props2rt{};
        props2rt.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2rt.pNext = &rtProps;
        vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2rt);
        m_capabilities.maxRayRecursionDepth = rtProps.maxRayRecursionDepth;
        m_capabilities.shaderGroupHandleSize = rtProps.shaderGroupHandleSize;
        m_capabilities.shaderGroupBaseAlignment = rtProps.shaderGroupBaseAlignment;

        VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
        asProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props2as{};
        props2as.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2as.pNext = &asProps;
        vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2as);
        m_capabilities.minScratchAlignment = asProps.minAccelerationStructureScratchOffsetAlignment;
    }

    auto queryCompressedSupport = [this](VkFormat format) -> bool {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &props);
        return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
    };
    m_capabilities.supportsBC4 = queryCompressedSupport(VK_FORMAT_BC4_UNORM_BLOCK);
    m_capabilities.supportsBC5 = queryCompressedSupport(VK_FORMAT_BC5_UNORM_BLOCK);
    m_capabilities.supportsBC7 = queryCompressedSupport(VK_FORMAT_BC7_UNORM_BLOCK);

    // NOTE: supportsDescriptorIndexing is intentionally NOT set here anymore.
    // It is latched inside createLogicalDevice() based on what was actually
    // enabled on the VkDevice. Querying the physical device post-hoc would
    // report "capable but not enabled", causing material-preview pipeline
    // creation to use runtimeDescriptorArray against a device that never
    // enabled the feature — observed to crash nvoglv64.dll on GTX 850M when
    // the RT device creation path falls back to a featureless VkDevice.
}

GPUVendor VulkanDevice::vendorFromID(uint32_t vendorID) {
    switch (vendorID) {
        case 0x10DE: return GPUVendor::NVIDIA;
        case 0x1002: return GPUVendor::AMD;
        case 0x8086: return GPUVendor::INTEL;
        case 0x106B: return GPUVendor::APPLE;
        case 0x5143: return GPUVendor::QUALCOMM;
        case 0x13B5: return GPUVendor::ARM_MALI;
        default:     return GPUVendor::UNKNOWN;
    }
}

void VulkanDevice::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = vulkanDebugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (func) func(m_instance, &createInfo, nullptr, &m_debugMessenger);
}

// ========================================================================
// Buffer Operations
// ========================================================================

uint32_t VulkanDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    return UINT32_MAX;
}

VkBufferUsageFlags VulkanDevice::translateBufferUsage(BufferUsage usage) {
    VkBufferUsageFlags flags = 0;
    if ((uint32_t)usage & (uint32_t)BufferUsage::VERTEX) flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if ((uint32_t)usage & (uint32_t)BufferUsage::INDEX) flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if ((uint32_t)usage & (uint32_t)BufferUsage::UNIFORM) flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if ((uint32_t)usage & (uint32_t)BufferUsage::STORAGE) flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((uint32_t)usage & (uint32_t)BufferUsage::TRANSFER_SRC) flags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if ((uint32_t)usage & (uint32_t)BufferUsage::TRANSFER_DST) flags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if ((uint32_t)usage & (uint32_t)BufferUsage::ACCELERATION) flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if ((uint32_t)usage & (uint32_t)BufferUsage::SHADER_BINDING) flags |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
    return flags;
}

VkMemoryPropertyFlags VulkanDevice::translateMemoryLocation(MemoryLocation location) {
    switch (location) {
        case MemoryLocation::GPU_ONLY:  return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case MemoryLocation::CPU_TO_GPU: return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        // GPU_TO_CPU uses HOST_CACHED for fast CPU readback (L1/L2 cache).
        // downloadBuffer() calls vkInvalidateMappedMemoryRanges before memcpy
        // to ensure cache coherency on non-coherent memory types.
        // Using HOST_COHERENT without CACHED causes ~2-3× slower reads via
        // uncached MMIO path, halving RT pathtracing throughput.
        case MemoryLocation::GPU_TO_CPU: return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        case MemoryLocation::CPU_ONLY:  return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        default: return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
}

BufferHandle VulkanDevice::createBuffer(const BufferCreateInfo& info) {
    BufferHandle handle{};
    handle.size = info.size;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = info.size;
    bufferInfo.usage = translateBufferUsage(info.usage) | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (m_capabilities.supportsBufferDeviceAddress)
        bufferInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(m_device, &bufferInfo, nullptr, &handle.buffer);
    if (result != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] vkCreateBuffer failed (size=" << info.size << ", result=" << result << ")" << std::endl;
        handle = {};
        return handle;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(m_device, handle.buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, translateMemoryLocation(info.location));

    if (allocInfo.memoryTypeIndex == UINT32_MAX) {
        VK_ERROR() << "[VulkanDevice] No suitable memory type found (size=" << memReq.size << ")" << std::endl;
        vkDestroyBuffer(m_device, handle.buffer, nullptr);
        handle = {};
        return handle;
    }

    VkMemoryAllocateFlagsInfo flagsInfo{};
    if (m_capabilities.supportsBufferDeviceAddress) {
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        allocInfo.pNext = &flagsInfo;
    }

    result = vkAllocateMemory(m_device, &allocInfo, nullptr, &handle.memory);
    if (result != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] vkAllocateMemory failed (size=" << memReq.size << ", result=" << result << ")" << std::endl;
        // Do not aggressively recreate backend on large readback-buffer failures:
        // this can cascade into device-creation failures under transient WDDM pressure.
        const bool isLargeReadback =
            (info.location == MemoryLocation::GPU_TO_CPU) && (info.size >= (16ull * 1024ull * 1024ull));
        if (!isLargeReadback) {
            signalVulkanMemoryPressure(result, "createBuffer/vkAllocateMemory");
        } else {
            SCENE_LOG_WARN("[Vulkan] Large readback allocation failed; skipping frame without backend recreate.");
        }
        vkDestroyBuffer(m_device, handle.buffer, nullptr);
        handle = {};
        return handle;
    }
    vkBindBufferMemory(m_device, handle.buffer, handle.memory, 0);

    // Get device address
    if (m_capabilities.supportsBufferDeviceAddress && fpGetBufferDeviceAddressKHR) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = handle.buffer;
        handle.deviceAddress = fpGetBufferDeviceAddressKHR(m_device, &addrInfo);
    }

    // Upload initial data if provided
    if (info.initialData && info.location != MemoryLocation::GPU_ONLY) {
        void* mapped;
        vkMapMemory(m_device, handle.memory, 0, info.size, 0, &mapped);
        memcpy(mapped, info.initialData, info.size);
        vkUnmapMemory(m_device, handle.memory);
    }

    return handle;
}

void VulkanDevice::destroyBuffer(BufferHandle& buffer) {
    if (m_device == VK_NULL_HANDLE) {
        buffer = {};
        return;
    }
    if (buffer.buffer) vkDestroyBuffer(m_device, buffer.buffer, nullptr);
    if (buffer.memory) vkFreeMemory(m_device, buffer.memory, nullptr);
    buffer = {};
}

BufferHandle VulkanDevice::createExportableBuffer(const BufferCreateInfo& info,
                                                  void** outWin32Handle,
                                                  uint64_t* outAllocationSize) {
    if (outWin32Handle) *outWin32Handle = nullptr;
    if (outAllocationSize) *outAllocationSize = 0;
    BufferHandle handle{};

#ifdef _WIN32
    if (!m_capabilities.supportsExternalMemoryWin32) {
        return handle;
    }

    handle.size = info.size;

    // The buffer itself must advertise that its memory will be externally shared.
    VkExternalMemoryBufferCreateInfo extBufInfo{};
    extBufInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = &extBufInfo;
    bufferInfo.size = info.size;
    bufferInfo.usage = translateBufferUsage(info.usage) | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &handle.buffer) != VK_SUCCESS) {
        handle = {};
        return handle;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(m_device, handle.buffer, &memReq);

    // Exported memory for CUDA interop must be DEVICE_LOCAL. Host-visible bits
    // are not guaranteed on exportable heaps (NVIDIA Win32 commonly rejects
    // HOST_VISIBLE+EXTERNAL in the same type). Callers that need host access
    // must use the regular createBuffer path.
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
    uint32_t memType = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memType = i;
            break;
        }
    }
    if (memType == UINT32_MAX) {
        vkDestroyBuffer(m_device, handle.buffer, nullptr);
        handle = {};
        return handle;
    }

    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    // Dedicated allocation — NVIDIA's CUDA interop path requires the
    // imported memory to be a dedicated buffer allocation on Windows.
    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.buffer = handle.buffer;
    dedicatedInfo.pNext = &exportInfo;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memType;
    allocInfo.pNext = &dedicatedInfo;

    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &handle.memory) != VK_SUCCESS) {
        vkDestroyBuffer(m_device, handle.buffer, nullptr);
        handle = {};
        return handle;
    }
    vkBindBufferMemory(m_device, handle.buffer, handle.memory, 0);

    // Resolve vkGetMemoryWin32HandleKHR (device extension entry point).
    auto fpGetMemHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(m_device, "vkGetMemoryWin32HandleKHR"));
    if (!fpGetMemHandle) {
        vkFreeMemory(m_device, handle.memory, nullptr);
        vkDestroyBuffer(m_device, handle.buffer, nullptr);
        handle = {};
        return handle;
    }

    VkMemoryGetWin32HandleInfoKHR getHandleInfo{};
    getHandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    getHandleInfo.memory = handle.memory;
    getHandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE rawHandle = nullptr;
    if (fpGetMemHandle(m_device, &getHandleInfo, &rawHandle) != VK_SUCCESS || !rawHandle) {
        vkFreeMemory(m_device, handle.memory, nullptr);
        vkDestroyBuffer(m_device, handle.buffer, nullptr);
        handle = {};
        return handle;
    }

    if (outWin32Handle) *outWin32Handle = rawHandle;
    if (outAllocationSize) *outAllocationSize = memReq.size;
    return handle;
#else
    (void)info;
    return handle;
#endif
}

void* VulkanDevice::mapBuffer(const BufferHandle& buffer) {
    void* data;
    vkMapMemory(m_device, buffer.memory, 0, buffer.size, 0, &data);
    return data;
}

void VulkanDevice::unmapBuffer(const BufferHandle& buffer) {
    vkUnmapMemory(m_device, buffer.memory);
}

void VulkanDevice::uploadBuffer(const BufferHandle& dst, const void* data, uint64_t size, uint64_t offset) {
    if (!dst.buffer || !data || size == 0) return;

    void* mapped;
    if (vkMapMemory(m_device, dst.memory, offset, size, 0, &mapped) == VK_SUCCESS) {
        memcpy(mapped, data, size);
        vkUnmapMemory(m_device, dst.memory);
    } else {
        // Memory is not host-visible (e.g. GPU_ONLY / device-local).
        // Upload via a temporary host-visible staging buffer + vkCmdCopyBuffer.
        BufferCreateInfo stagingCI;
        stagingCI.size        = size;
        stagingCI.usage       = BufferUsage::TRANSFER_SRC;
        stagingCI.location    = MemoryLocation::CPU_TO_GPU;
        stagingCI.initialData = data;
        BufferHandle staging  = createBuffer(stagingCI);
        if (!staging.buffer) {
            VK_ERROR() << "[VulkanDevice] uploadBuffer fallback failed: staging allocation failed (size="
                       << size << ")" << std::endl;
            return;
        }

        VkCommandBuffer cmdBuf = beginSingleTimeCommands();
        if (cmdBuf == VK_NULL_HANDLE) {
            destroyBuffer(staging);
            return;
        }
        VkBufferCopy region{};
        region.srcOffset = 0;
        region.dstOffset = offset;
        region.size      = size;
        vkCmdCopyBuffer(cmdBuf, staging.buffer, dst.buffer, 1, &region);
        endSingleTimeCommands(cmdBuf);

        destroyBuffer(staging);
    }
}

void VulkanDevice::downloadBuffer(const BufferHandle& src, void* data, uint64_t size, uint64_t offset) {
    if (!src.buffer || !data || size == 0) return;

    void* mapped = nullptr;
    VkResult mapRes = vkMapMemory(m_device, src.memory, offset, size, 0, &mapped);
    if (mapRes == VK_SUCCESS && mapped) {
        // [FIX] Invalidate mapped range for non-coherent memory.
        // Without this, CPU cache may contain stale data on some GPU/driver
        // combinations, causing intermittent black/white/garbage output.
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = src.memory;
        range.offset = offset;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(m_device, 1, &range);
        memcpy(data, mapped, size);
        vkUnmapMemory(m_device, src.memory);
        return;
    }

    // Buffer memory is not host-visible (device-local). Perform readback via
    // a temporary host-visible staging buffer and a GPU copy.
    BufferCreateInfo stagingCI{};
    stagingCI.size = size;
    stagingCI.usage = BufferUsage::TRANSFER_DST;
    stagingCI.location = MemoryLocation::GPU_TO_CPU;
    stagingCI.initialData = nullptr;
    BufferHandle staging = createBuffer(stagingCI);
    if (!staging.buffer) {
        VK_ERROR() << "[VulkanDevice] downloadBuffer fallback failed: staging allocation failed (size=" << size << ")" << std::endl;
        return;
    }

    VkCommandBuffer cmdBuf = beginSingleTimeCommands();
    if (cmdBuf == VK_NULL_HANDLE) {
        destroyBuffer(staging);
        return;
    }

    VkBufferCopy region{};
    region.srcOffset = offset;
    region.dstOffset = 0;
    region.size = size;
    vkCmdCopyBuffer(cmdBuf, src.buffer, staging.buffer, 1, &region);
    endSingleTimeCommands(cmdBuf);

    // Map staging and copy to user pointer
    void* stagingMapped = nullptr;
    if (vkMapMemory(m_device, staging.memory, 0, size, 0, &stagingMapped) == VK_SUCCESS && stagingMapped) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = staging.memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(m_device, 1, &range);
        memcpy(data, stagingMapped, size);
        vkUnmapMemory(m_device, staging.memory);
    } else {
        VK_ERROR() << "[VulkanDevice] downloadBuffer failed to map staging buffer" << std::endl;
    }

    destroyBuffer(staging);
}

// ========================================================================
// Command Buffer Helpers
// ========================================================================

VkCommandBuffer VulkanDevice::beginSingleTimeCommands() {
    if (!m_device || !m_commandPool) return VK_NULL_HANDLE;

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
    VkResult allocRes = vkAllocateCommandBuffers(m_device, &allocInfo, &cmdBuf);
    if (allocRes != VK_SUCCESS || cmdBuf == VK_NULL_HANDLE) {
        VK_ERROR() << "[VulkanDevice] beginSingleTimeCommands: vkAllocateCommandBuffers failed (result="
                   << allocRes << ")" << std::endl;
        signalVulkanMemoryPressure(allocRes, "beginSingleTimeCommands/vkAllocateCommandBuffers");
        return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult beginRes = vkBeginCommandBuffer(cmdBuf, &beginInfo);
    if (beginRes != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] beginSingleTimeCommands: vkBeginCommandBuffer failed (result="
                   << beginRes << ")" << std::endl;
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmdBuf);
        return VK_NULL_HANDLE;
    }

    return cmdBuf;
}

void VulkanDevice::endSingleTimeCommands(VkCommandBuffer cmdBuf) {
    if (cmdBuf == VK_NULL_HANDLE || !m_device || !m_computeQueue) return;

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    // Use per-submit fence wait instead of queue-wide idle.
    // This avoids stalling unrelated queued work and significantly reduces
    // synchronization overhead during frequent BLAS/TLAS updates.
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence submitFence = VK_NULL_HANDLE;
    if (vkCreateFence(m_device, &fenceInfo, nullptr, &submitFence) != VK_SUCCESS) {
        // Fallback for robustness if fence creation fails.
        vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_computeQueue);
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmdBuf);
        return;
    }

    if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, submitFence) != VK_SUCCESS) {
        vkDestroyFence(m_device, submitFence, nullptr);
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmdBuf);
        return;
    }
    vkWaitForFences(m_device, 1, &submitFence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(m_device, submitFence, nullptr);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmdBuf);
}

// ========================================================================
// Acceleration Structures - Real Implementation
// ========================================================================

uint32_t VulkanDevice::createBLAS(const BLASCreateInfo& info) {
    if (!hasHardwareRT() || !fpCreateAccelerationStructureKHR) {
        VK_ERROR() << "[VulkanDevice] Hardware RT not available for BLAS creation" << std::endl;
        return UINT32_MAX;
    }
    // Device-resident mode (Phase 3d) supplies geometry by address, so vertexData may be
    // null; otherwise a host vertex pointer is required.
    if (info.vertexCount == 0) return UINT32_MAX;
    if (!info.useDeviceGeometry && !info.vertexData) return UINT32_MAX;
    if (info.useDeviceGeometry && info.geometryDeviceAddress == 0) return UINT32_MAX;

    // --- 1) Combine all geometry data into ONE GPU buffer to reduce allocation count ---
    // This is critical for large scenes to avoid hitting maxMemoryAllocationCount limits.
    // In device-resident mode the combined buffer already exists on the GPU (written by
    // the CC compute refine in this exact layout); we only derive the sub-offsets.
    uint64_t vertSize = (uint64_t)info.vertexCount * info.vertexStride;
    // Device-resident CC geometry always carries a normal + uv block (the compute expand
    // writes them), so derive their sizes unconditionally in that mode.
    uint64_t normSize = (info.normalData || info.useDeviceGeometry) ? (uint64_t)info.vertexCount * sizeof(float) * 3 : 0;
    uint64_t uvSize   = (info.uvData   || info.useDeviceGeometry) ? (uint64_t)info.vertexCount * sizeof(float) * 2 : 0;

    bool hasIndices = (info.indexData && info.indexCount > 0) || (info.useDeviceGeometry && info.indexCount > 0);
    uint64_t idxSize = hasIndices ? (uint64_t)info.indexCount * sizeof(uint32_t) : 0;

    bool hasMaterials = (info.materialIndexData && info.materialIndexCount > 0) || (info.useDeviceGeometry && info.materialIndexCount > 0);
    uint64_t matSize = hasMaterials ? (uint64_t)info.materialIndexCount * sizeof(uint32_t) : 0;

    uint64_t totalGeomSize = vertSize + normSize + uvSize + idxSize + matSize;

    // Base device address of the (combined) geometry: a buffer we allocate+upload in host
    // mode, or the caller's pre-written GPU buffer in device-resident mode.
    BufferHandle geometryBuffer;     // owned only in host mode
    VkDeviceAddress geomBase = 0;

    if (info.useDeviceGeometry) {
        geomBase = (VkDeviceAddress)info.geometryDeviceAddress;
    } else {
        BufferCreateInfo geomBufInfo;
        geomBufInfo.size = totalGeomSize;
        geomBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::VERTEX;
        geomBufInfo.location = MemoryLocation::GPU_ONLY;
        geomBufInfo.initialData = nullptr;

        geometryBuffer = createBuffer(geomBufInfo);
        if (!geometryBuffer.buffer) {
            VK_ERROR() << "[VulkanDevice] Failed to allocate combined geometry buffer for BLAS" << std::endl;
            return UINT32_MAX;
        }
        geomBase = geometryBuffer.deviceAddress;

        // Upload geometry via staging path (keeps persistent buffer device-local).
        uint64_t off = 0;
        uploadBuffer(geometryBuffer, info.vertexData, vertSize, off); off += vertSize;
        if (normSize && info.normalData) { uploadBuffer(geometryBuffer, info.normalData, normSize, off); off += normSize; }
        if (uvSize && info.uvData)       { uploadBuffer(geometryBuffer, info.uvData, uvSize, off); off += uvSize; }
        if (idxSize && info.indexData)   { uploadBuffer(geometryBuffer, info.indexData, idxSize, off); off += idxSize; }
        if (matSize && info.materialIndexData) { uploadBuffer(geometryBuffer, info.materialIndexData, matSize, off); }
    }
    
    // Build skinning separate buffers if required
    BufferHandle baseVertBuf, baseNormBuf, boneIdxBuf, boneWtBuf;
    if (info.hasSkinning && info.boneIndicesData && info.boneWeightsData) {
        BufferCreateInfo sInfo;
        sInfo.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        sInfo.location = MemoryLocation::GPU_ONLY;
        
        sInfo.size = vertSize; baseVertBuf = createBuffer(sInfo);
        if (baseVertBuf.buffer) uploadBuffer(baseVertBuf, info.vertexData, vertSize);
        
        if (normSize && info.normalData) {
            sInfo.size = normSize; baseNormBuf = createBuffer(sInfo);
            if (baseNormBuf.buffer) uploadBuffer(baseNormBuf, info.normalData, normSize);
        }
        
        uint64_t bIdxSz = (uint64_t)info.vertexCount * sizeof(int32_t) * 4;
        sInfo.size = bIdxSz; boneIdxBuf = createBuffer(sInfo);
        if (boneIdxBuf.buffer) uploadBuffer(boneIdxBuf, info.boneIndicesData, bIdxSz);
        
        uint64_t bWtSz = (uint64_t)info.vertexCount * sizeof(float) * 4;
        sInfo.size = bWtSz; boneWtBuf = createBuffer(sInfo);
        if (boneWtBuf.buffer) uploadBuffer(boneWtBuf, info.boneWeightsData, bWtSz);
    }

    // --- 2) Build geometry info ---
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = geomBase;
    triangles.vertexStride = info.vertexStride;
    triangles.maxVertex = info.vertexCount - 1;
    if (hasIndices) {
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = geomBase + vertSize + normSize + uvSize;
    } else {
        triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = info.opaqueGeometry ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
    geometry.geometry.triangles = triangles;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (info.allowUpdate) buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    uint32_t primitiveCount = hasIndices ? (info.indexCount / 3) : (info.vertexCount / 3);

    // --- 3) Query build sizes ---
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fpGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    // --- 4) Create AS buffer ---
    AccelStructHandle blasHandle{};

    BufferCreateInfo asBufInfo;
    asBufInfo.size = sizeInfo.accelerationStructureSize;
    asBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    asBufInfo.location = MemoryLocation::GPU_ONLY;
    blasHandle.buffer = createBuffer(asBufInfo);
    if (!blasHandle.buffer.buffer) {
        destroyBuffer(geometryBuffer);
        destroyBuffer(baseVertBuf);
        destroyBuffer(baseNormBuf);
        destroyBuffer(boneIdxBuf);
        destroyBuffer(boneWtBuf);
        return UINT32_MAX;
    }

    // --- 5) Create acceleration structure ---
    VkAccelerationStructureCreateInfoKHR asCreateInfo{};
    asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = blasHandle.buffer.buffer;
    asCreateInfo.size = sizeInfo.accelerationStructureSize;
    asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    if (fpCreateAccelerationStructureKHR(m_device, &asCreateInfo, nullptr, &blasHandle.accel) != VK_SUCCESS ||
        blasHandle.accel == VK_NULL_HANDLE) {
        destroyBuffer(blasHandle.buffer);
        destroyBuffer(geometryBuffer);
        destroyBuffer(baseVertBuf);
        destroyBuffer(baseNormBuf);
        destroyBuffer(boneIdxBuf);
        destroyBuffer(boneWtBuf);
        return UINT32_MAX;
    }

    // --- 6) Scratch buffer with proper alignment ---
    uint64_t scratchAlignment = m_capabilities.minScratchAlignment > 0 ? m_capabilities.minScratchAlignment : 128;
    uint64_t alignedScratchSize = (sizeInfo.buildScratchSize + scratchAlignment - 1) & ~(scratchAlignment - 1);

    // --- 7) Build! ---
    buildInfo.dstAccelerationStructure = blasHandle.accel;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    if (m_inBatchedBLASBuild && m_batchBLASCmd) {
        // ── Batched mode: reuse single shared scratch buffer, record into batch cmd ──
        if (alignedScratchSize > m_batchScratchBuffer.size) {
            // Scratch buffer too small — flush pending builds, then resize
            if (m_batchScratchBuffer.buffer && m_batchBLASInCurrentCmd > 0) {
                endSingleTimeCommands(m_batchBLASCmd);
                m_batchBLASCmd = beginSingleTimeCommands();
                if (m_batchBLASCmd == VK_NULL_HANDLE) {
                    if (m_batchScratchBuffer.buffer) destroyBuffer(m_batchScratchBuffer);
                    return UINT32_MAX;
                }
                m_batchBLASInCurrentCmd = 0;
            }
            if (m_batchScratchBuffer.buffer) destroyBuffer(m_batchScratchBuffer);
            BufferCreateInfo scrBuf;
            scrBuf.size = alignedScratchSize;
            scrBuf.usage = BufferUsage::STORAGE;
            scrBuf.location = MemoryLocation::GPU_ONLY;
            m_batchScratchBuffer = createBuffer(scrBuf);
            if (!m_batchScratchBuffer.buffer) {
                return UINT32_MAX;
            }
        } else if (m_batchBLASInCurrentCmd > 0) {
            // Same scratch reused — serialize via barrier
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            vkCmdPipelineBarrier(m_batchBLASCmd,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        buildInfo.scratchData.deviceAddress = m_batchScratchBuffer.deviceAddress;
        fpCmdBuildAccelerationStructuresKHR(m_batchBLASCmd, 1, &buildInfo, &pRangeInfo);
        m_batchBLASCount++;
        m_batchBLASInCurrentCmd++;
    } else {
        // ── Non-batched mode: original per-BLAS submit ──
        BufferCreateInfo scratchBufInfo;
        scratchBufInfo.size = alignedScratchSize;
        scratchBufInfo.usage = BufferUsage::STORAGE;
        scratchBufInfo.location = MemoryLocation::GPU_ONLY;
        auto scratchBuffer = createBuffer(scratchBufInfo);
        if (!scratchBuffer.buffer) return UINT32_MAX;
        buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

        VkCommandBuffer cmd = beginSingleTimeCommands();
        if (cmd == VK_NULL_HANDLE) {
            destroyBuffer(scratchBuffer);
            return UINT32_MAX;
        }
        fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
        endSingleTimeCommands(cmd);
        destroyBuffer(scratchBuffer);
    }
    // --- 8) Get device address for TLAS reference ---
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blasHandle.accel;
    blasHandle.deviceAddress = fpGetAccelerationStructureDeviceAddressKHR(m_device, &addrInfo);
    
    // Store attribute buffer segments in handle for shader access (binding 4).
    // All point to the same VkBuffer (host mode) or carry only staggered addresses into
    // the caller's device-resident buffer (Phase 3d) — in which case the BLAS does NOT
    // own them and cleanup must skip the destroy.
    blasHandle.externalGeometry = info.useDeviceGeometry;
    blasHandle.vertexBuffer = geometryBuffer;          // empty handle in device mode
    blasHandle.vertexBuffer.deviceAddress = geomBase;

    if (normSize) {
        blasHandle.normalBuffer = geometryBuffer;
        blasHandle.normalBuffer.deviceAddress = geomBase + vertSize;
    }
    if (uvSize) {
        blasHandle.uvBuffer = geometryBuffer;
        blasHandle.uvBuffer.deviceAddress = geomBase + vertSize + normSize;
    }
    if (idxSize) {
        blasHandle.indexBuffer = geometryBuffer;
        blasHandle.indexBuffer.deviceAddress = geomBase + vertSize + normSize + uvSize;
    }
    if (matSize) {
        blasHandle.materialIndexBuffer = geometryBuffer;
        blasHandle.materialIndexBuffer.deviceAddress = geomBase + vertSize + normSize + uvSize + idxSize;
    }
    
    // Store dynamic update / skinning state.
    blasHandle.hasSkinning = info.hasSkinning;
    blasHandle.allowUpdate = info.allowUpdate;
    blasHandle.vertexCount = info.vertexCount;
    blasHandle.indexCount = info.indexCount;
    if (info.hasSkinning) {
        blasHandle.baseVertexBuffer = baseVertBuf;
        blasHandle.baseNormalBuffer = baseNormBuf;
        blasHandle.boneIndexBuffer = boneIdxBuf;
        blasHandle.boneWeightBuffer = boneWtBuf;
    }

    uint32_t idx = (uint32_t)m_blasList.size();
    m_blasList.push_back(blasHandle);

    if (!m_inBatchedBLASBuild) {
        VK_INFO() << "[VulkanDevice] BLAS created (index=" << idx
                  << ", tris=" << primitiveCount << ", size=" << (sizeInfo.accelerationStructureSize / 1024) << " KB)" << std::endl;
    }
    return idx;
}

bool VulkanDevice::dispatchSkinningToBuffers(BufferHandle& baseVertexBuffer,
                                            BufferHandle& baseNormalBuffer,
                                            BufferHandle& boneIndexBuffer,
                                            BufferHandle& boneWeightBuffer,
                                            BufferHandle& persistentBoneMatsBuffer,
                                            uint64_t& persistentBoneMatsBufSize,
                                            VkDescriptorSet& skinningDescSet,
                                            const BufferHandle& outVertexBuffer,
                                            const BufferHandle& outNormalBuffer,
                                            uint32_t vertexCount,
                                            const std::vector<Matrix4x4>& boneMatrices) {
    if (!m_device || !m_commandPool || !m_computeQueue) return false;
    if (m_skinningPipeline == VK_NULL_HANDLE) return false;
    if (m_skinningPipelineLayout == VK_NULL_HANDLE || m_skinningDescPool == VK_NULL_HANDLE || m_skinningDescLayout == VK_NULL_HANDLE) return false;
    if (!baseVertexBuffer.buffer || !baseNormalBuffer.buffer || !boneIndexBuffer.buffer || !boneWeightBuffer.buffer) return false;
    if (!outVertexBuffer.buffer || !outNormalBuffer.buffer || vertexCount == 0 || boneMatrices.empty()) return false;

    const uint64_t boneMatSize = boneMatrices.size() * sizeof(Matrix4x4);
    if (!persistentBoneMatsBuffer.buffer || persistentBoneMatsBufSize < boneMatSize) {
        if (persistentBoneMatsBuffer.buffer) destroyBuffer(persistentBoneMatsBuffer);
        BufferCreateInfo bc{};
        bc.size = boneMatSize;
        bc.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        bc.location = MemoryLocation::CPU_TO_GPU;
        persistentBoneMatsBuffer = createBuffer(bc);
        if (!persistentBoneMatsBuffer.buffer || !persistentBoneMatsBuffer.memory) {
            persistentBoneMatsBuffer = {};
            persistentBoneMatsBufSize = 0;
            skinningDescSet = VK_NULL_HANDLE;
            return false;
        }
        persistentBoneMatsBufSize = boneMatSize;
        skinningDescSet = VK_NULL_HANDLE;
    }

    void* mapped = nullptr;
    if (vkMapMemory(m_device, persistentBoneMatsBuffer.memory, 0, boneMatSize, 0, &mapped) != VK_SUCCESS || !mapped) {
        return false;
    }
    memcpy(mapped, boneMatrices.data(), boneMatSize);
    vkUnmapMemory(m_device, persistentBoneMatsBuffer.memory);

    if (skinningDescSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_skinningDescPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_skinningDescLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &skinningDescSet) != VK_SUCCESS ||
            skinningDescSet == VK_NULL_HANDLE) {
            return false;
        }
    }

    const uint64_t outNormalByteOffset =
        (outNormalBuffer.buffer == outVertexBuffer.buffer)
        ? (outNormalBuffer.deviceAddress - outVertexBuffer.deviceAddress)
        : 0;
    const uint64_t vertexByteSize = (uint64_t)vertexCount * 12;

    VkDescriptorBufferInfo bInfo[7]{};
    bInfo[0].buffer = baseVertexBuffer.buffer;          bInfo[0].offset = 0;                  bInfo[0].range = VK_WHOLE_SIZE;
    bInfo[1].buffer = baseNormalBuffer.buffer;          bInfo[1].offset = 0;                  bInfo[1].range = VK_WHOLE_SIZE;
    bInfo[2].buffer = boneIndexBuffer.buffer;           bInfo[2].offset = 0;                  bInfo[2].range = VK_WHOLE_SIZE;
    bInfo[3].buffer = boneWeightBuffer.buffer;          bInfo[3].offset = 0;                  bInfo[3].range = VK_WHOLE_SIZE;
    bInfo[4].buffer = persistentBoneMatsBuffer.buffer;  bInfo[4].offset = 0;                  bInfo[4].range = VK_WHOLE_SIZE;
    bInfo[5].buffer = outVertexBuffer.buffer;           bInfo[5].offset = 0;                  bInfo[5].range = vertexByteSize;
    bInfo[6].buffer = outNormalBuffer.buffer;           bInfo[6].offset = outNormalByteOffset; bInfo[6].range = vertexByteSize;

    VkWriteDescriptorSet writes[7]{};
    for (int i = 0; i < 7; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = skinningDescSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bInfo[i];
    }
    vkUpdateDescriptorSets(m_device, 7, writes, 0, nullptr);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_skinningPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_skinningPipelineLayout, 0, 1, &skinningDescSet, 0, nullptr);
    uint32_t params[2] = { vertexCount, (uint32_t)boneMatrices.size() };
    vkCmdPushConstants(cmd, m_skinningPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, params);
    const uint32_t groupCount = (vertexCount + 255) / 256;
    vkCmdDispatch(cmd, groupCount, 1, 1);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    endSingleTimeCommands(cmd);
    return true;
}

void VulkanDevice::updateBLAS(uint32_t blasIndex, const float* newVertices, const float* newNormals) {
    if (!hasHardwareRT() || !fpCmdBuildAccelerationStructuresKHR) return;
    if (blasIndex >= m_blasList.size()) return;
    
    AccelStructHandle& blasHandle = m_blasList[blasIndex];
    if (blasHandle.accel == VK_NULL_HANDLE || !blasHandle.allowUpdate || blasHandle.vertexCount == 0) return;

    if (newVertices) {
        uploadBuffer(blasHandle.vertexBuffer, newVertices, (uint64_t)blasHandle.vertexCount * 12);
    }
    if (newNormals && blasHandle.normalBuffer.buffer) {
        const uint64_t normalByteSize = (uint64_t)blasHandle.vertexCount * 12;
        const bool normalSharesGeometryBuffer =
            blasHandle.normalBuffer.buffer == blasHandle.vertexBuffer.buffer &&
            blasHandle.normalBuffer.deviceAddress >= blasHandle.vertexBuffer.deviceAddress;
        const uint64_t normalByteOffset = normalSharesGeometryBuffer
            ? (uint64_t)(blasHandle.normalBuffer.deviceAddress - blasHandle.vertexBuffer.deviceAddress)
            : 0ull;
        uploadBuffer(
            normalSharesGeometryBuffer ? blasHandle.vertexBuffer : blasHandle.normalBuffer,
            newNormals,
            normalByteSize,
            normalByteOffset);
    }

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = blasHandle.vertexBuffer.deviceAddress;
    triangles.vertexStride = 12;
    triangles.maxVertex = blasHandle.vertexCount - 1;

    // Indexed BLAS (uploadTriangleMeshIndexed): topology is the resident index buffer, which a
    // MODE_UPDATE refit reuses unchanged — only the vertex/normal positions were re-uploaded.
    const bool hasIndices = blasHandle.indexCount > 0 && blasHandle.indexBuffer.buffer;
    if (hasIndices) {
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = blasHandle.indexBuffer.deviceAddress;
    } else {
        triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = 0;
    geometry.geometry.triangles = triangles;

    uint32_t primitiveCount = hasIndices ? (blasHandle.indexCount / 3) : (blasHandle.vertexCount / 3);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = blasHandle.accel;
    buildInfo.dstAccelerationStructure = blasHandle.accel;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fpGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

    BufferCreateInfo scratchBufInfo;
    scratchBufInfo.size = sizeInfo.buildScratchSize;
    scratchBufInfo.usage = BufferUsage::STORAGE;
    scratchBufInfo.location = MemoryLocation::GPU_ONLY;
    auto scratchBuffer = createBuffer(scratchBufInfo);
    if (!scratchBuffer.buffer) return;

    buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        destroyBuffer(scratchBuffer);
        return;
    }
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    endSingleTimeCommands(cmd);

    destroyBuffer(scratchBuffer);
}

// ========================================================================
// Batched BLAS Build Support
// ========================================================================

void VulkanDevice::beginBatchedBLASBuild() {
    if (m_inBatchedBLASBuild) return;
    m_batchBLASCmd = beginSingleTimeCommands();
    if (m_batchBLASCmd == VK_NULL_HANDLE) return;
    m_inBatchedBLASBuild = true;
    m_batchBLASCount = 0;
    m_batchBLASInCurrentCmd = 0;
}

void VulkanDevice::endBatchedBLASBuild() {
    if (!m_inBatchedBLASBuild) return;
    m_inBatchedBLASBuild = false;

    if (m_batchBLASInCurrentCmd > 0) {
        endSingleTimeCommands(m_batchBLASCmd);
    } else {
        // No builds recorded in current cmd — discard
        vkEndCommandBuffer(m_batchBLASCmd);
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_batchBLASCmd);
    }

    // Cleanup shared scratch buffer
    if (m_batchScratchBuffer.buffer) {
        destroyBuffer(m_batchScratchBuffer);
    }

    if (m_batchBLASCount > 0) {
        VK_INFO() << "[VulkanDevice] Batched BLAS build complete: "
                  << m_batchBLASCount << " structures in single submit" << std::endl;
    }

    m_batchBLASCmd = VK_NULL_HANDLE;
    m_batchBLASCount = 0;
    m_batchBLASInCurrentCmd = 0;
}

// ========================================================================
// AABB BLAS for Procedural Volumes
// ========================================================================
uint32_t VulkanDevice::createAABB_BLAS(const float aabbMin[3], const float aabbMax[3]) {
    if (!hasHardwareRT() || !fpCreateAccelerationStructureKHR) {
        VK_ERROR() << "[VulkanDevice] Hardware RT not available for AABB BLAS" << std::endl;
        return UINT32_MAX;
    }

    // AABB data: VkAabbPositionsKHR = { minX, minY, minZ, maxX, maxY, maxZ }
    VkAabbPositionsKHR aabb{};
    aabb.minX = aabbMin[0]; aabb.minY = aabbMin[1]; aabb.minZ = aabbMin[2];
    aabb.maxX = aabbMax[0]; aabb.maxY = aabbMax[1]; aabb.maxZ = aabbMax[2];

    // Upload AABB data to GPU
    BufferCreateInfo aabbBufInfo;
    aabbBufInfo.size = sizeof(VkAabbPositionsKHR);
    aabbBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    aabbBufInfo.location = MemoryLocation::CPU_TO_GPU;
    aabbBufInfo.initialData = &aabb;
    auto aabbBuffer = createBuffer(aabbBufInfo);

    // Build geometry info for AABB
    VkAccelerationStructureGeometryAabbsDataKHR aabbsData{};
    aabbsData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbsData.data.deviceAddress = aabbBuffer.deviceAddress;
    aabbsData.stride = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags = 0; // Not opaque — volume needs closest-hit processing
    geometry.geometry.aabbs = aabbsData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    uint32_t primitiveCount = 1; // One AABB

    // Query build sizes
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fpGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    // Create AS buffer
    AccelStructHandle blasHandle{};
    BufferCreateInfo asBufInfo;
    asBufInfo.size = sizeInfo.accelerationStructureSize;
    asBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    asBufInfo.location = MemoryLocation::GPU_ONLY;
    blasHandle.buffer = createBuffer(asBufInfo);

    // Create acceleration structure
    VkAccelerationStructureCreateInfoKHR asCreateInfo{};
    asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = blasHandle.buffer.buffer;
    asCreateInfo.size = sizeInfo.accelerationStructureSize;
    asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    fpCreateAccelerationStructureKHR(m_device, &asCreateInfo, nullptr, &blasHandle.accel);

    // Scratch buffer
    uint64_t scratchAlignment = m_capabilities.minScratchAlignment > 0 ? m_capabilities.minScratchAlignment : 128;
    uint64_t alignedScratchSize = (sizeInfo.buildScratchSize + scratchAlignment - 1) & ~(scratchAlignment - 1);
    BufferCreateInfo scratchBufInfo;
    scratchBufInfo.size = alignedScratchSize;
    scratchBufInfo.usage = BufferUsage::STORAGE;
    scratchBufInfo.location = MemoryLocation::GPU_ONLY;
    auto scratchBuffer = createBuffer(scratchBufInfo);
    if (!scratchBuffer.buffer) {
        if (fpDestroyAccelerationStructureKHR && blasHandle.accel) {
            fpDestroyAccelerationStructureKHR(m_device, blasHandle.accel, nullptr);
        }
        destroyBuffer(blasHandle.buffer);
        destroyBuffer(aabbBuffer);
        return UINT32_MAX;
    }

    // Build
    buildInfo.dstAccelerationStructure = blasHandle.accel;
    buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        destroyBuffer(scratchBuffer);
        if (fpDestroyAccelerationStructureKHR && blasHandle.accel) {
            fpDestroyAccelerationStructureKHR(m_device, blasHandle.accel, nullptr);
        }
        destroyBuffer(blasHandle.buffer);
        destroyBuffer(aabbBuffer);
        return UINT32_MAX;
    }
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    endSingleTimeCommands(cmd);

    // Get device address
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blasHandle.accel;
    blasHandle.deviceAddress = fpGetAccelerationStructureDeviceAddressKHR(m_device, &addrInfo);

    // Cleanup
    destroyBuffer(scratchBuffer);
    // Keep aabbBuffer alive in the BLAS handle (reuse vertexBuffer slot)   
    blasHandle.vertexBuffer = aabbBuffer;

    uint32_t idx = (uint32_t)m_blasList.size();
    m_blasList.push_back(blasHandle);

    VK_INFO() << "[VulkanDevice] AABB BLAS created (index=" << idx
              << ", aabb=[" << aabbMin[0] << "," << aabbMin[1] << "," << aabbMin[2]
              << " -> " << aabbMax[0] << "," << aabbMax[1] << "," << aabbMax[2] << "])" << std::endl;
    return idx;
}

void VulkanDevice::createTLAS(const TLASCreateInfo& info) {
    if (!hasHardwareRT() || !fpCreateAccelerationStructureKHR) return;

    uint32_t instanceCount = (uint32_t)info.instances.size();

    // Determine whether we'll perform an UPDATE (more efficient) or full rebuild.
    // We can only perform an update if:
    // 1. We already have a TLAS.
    // 2. The existing TLAS was built with ALLOW_UPDATE bit (m_tlasSupportsUpdate).
    // 3. The user requested an update in info.allowUpdate.
    // 4. The instance COUNT hasn't changed (Vulkan refit requires identical topology).
    bool performUpdate = false;
    if (m_tlas.accel && m_tlasSupportsUpdate && info.allowUpdate && instanceCount == m_tlasInstanceCount) {
        performUpdate = true;
    } else {
        if (m_tlas.accel) {
            fpDestroyAccelerationStructureKHR(m_device, m_tlas.accel, nullptr);
            destroyBuffer(m_tlas.buffer);
            m_tlas = {};
        }
    }
    
    // Safety check: if scene is empty, stop here after potentially clearing old TLAS
    if (instanceCount == 0) {
        m_tlasInstanceCount = 0;
        return;
    }

    // NOTE: the previous instance buffer is intentionally NOT freed here — it is
    // reused in place below when large enough (see PERF note). Freeing+reallocating
    // it every call was the per-frame foam-refit stall.

    // --- 1) Build VkAccelerationStructureInstanceKHR array ---
    std::vector<VkAccelerationStructureInstanceKHR> vkInstances;
    vkInstances.reserve(info.instances.size());

    for (const auto& src : info.instances) {
        if (src.blasIndex >= m_blasList.size()) continue;

        VkAccelerationStructureInstanceKHR dst{};
        // VkTransformMatrixKHR is 3x4 row-major
        const auto& m = src.transform;
        dst.transform.matrix[0][0] = m.m[0][0]; dst.transform.matrix[0][1] = m.m[0][1]; dst.transform.matrix[0][2] = m.m[0][2]; dst.transform.matrix[0][3] = m.m[0][3];
        dst.transform.matrix[1][0] = m.m[1][0]; dst.transform.matrix[1][1] = m.m[1][1]; dst.transform.matrix[1][2] = m.m[1][2]; dst.transform.matrix[1][3] = m.m[1][3];
        dst.transform.matrix[2][0] = m.m[2][0]; dst.transform.matrix[2][1] = m.m[2][1]; dst.transform.matrix[2][2] = m.m[2][2]; dst.transform.matrix[2][3] = m.m[2][3];

        dst.instanceCustomIndex = src.customIndex;
        dst.mask = src.mask;
        dst.instanceShaderBindingTableRecordOffset = src.sbtRecordOffset;
        dst.flags = src.frontFaceCCW ? VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR : 0;
        if (src.opacityOverride == 1)      dst.flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
        else if (src.opacityOverride == 2) dst.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
        dst.accelerationStructureReference = m_blasList[src.blasIndex].deviceAddress;

        vkInstances.push_back(dst);
    }

    if (vkInstances.empty()) {
        VK_WARN() << "[VulkanDevice] createTLAS: No valid instances provided." << std::endl;
        return;
    }

    // --- 2) Upload instance data to GPU ---
    // PERF: reuse the persistent instance buffer in place when it is large enough.
    // createTLAS runs EVERY frame a live instance (foam/particle) moves; the old
    // free+alloc here did a heavyweight vkAllocateMemory/vkFreeMemory per frame that
    // serialized the driver and scaled with the foam pool — the foam-on Vulkan-RT
    // stall. The buffer is CPU_TO_GPU (host-visible), so a map+memcpy refreshes it.
    const VkDeviceSize neededInstBytes =
        (VkDeviceSize)vkInstances.size() * sizeof(VkAccelerationStructureInstanceKHR);
    if (m_tlasInstanceBuffer.buffer != VK_NULL_HANDLE &&
        m_tlasInstanceBuffer.memory != VK_NULL_HANDLE &&
        m_tlasInstanceBuffer.size >= neededInstBytes) {
        uploadBuffer(m_tlasInstanceBuffer, vkInstances.data(), neededInstBytes, 0);
    } else {
        // First build, or the instance count grew past the buffer: (re)allocate.
        if (m_tlasInstanceBuffer.buffer) destroyBuffer(m_tlasInstanceBuffer);
        BufferCreateInfo instBufInfo;
        instBufInfo.size = neededInstBytes;
        instBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
        instBufInfo.location = MemoryLocation::CPU_TO_GPU;
        instBufInfo.initialData = vkInstances.data();
        m_tlasInstanceBuffer = createBuffer(instBufInfo);
        if (!m_tlasInstanceBuffer.buffer) return;
    }
    const BufferHandle& instanceBuffer = m_tlasInstanceBuffer;

    // --- 3) Build geometry info ---
    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = instanceBuffer.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (info.allowUpdate) buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

   

    // --- 4) Setup Build Mode and Query Sizes ---
    if (performUpdate) {
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        buildInfo.srcAccelerationStructure = m_tlas.accel;
    } else {
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    }

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fpGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &instanceCount, &sizeInfo);

    // --- 5) Create TLAS IF NOT UPDATING ---
    if (!performUpdate) {
        BufferCreateInfo asBufInfo;
        asBufInfo.size = sizeInfo.accelerationStructureSize;
        asBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
        asBufInfo.location = MemoryLocation::GPU_ONLY;
        m_tlas.buffer = createBuffer(asBufInfo);

        VkAccelerationStructureCreateInfoKHR asCreateInfo{};
        asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        asCreateInfo.buffer = m_tlas.buffer.buffer;
        asCreateInfo.size = sizeInfo.accelerationStructureSize;
        asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        fpCreateAccelerationStructureKHR(m_device, &asCreateInfo, nullptr, &m_tlas.accel);

        // Get device address
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = m_tlas.accel;
        m_tlas.deviceAddress = fpGetAccelerationStructureDeviceAddressKHR(m_device, &addrInfo);
    }

    // --- 6) Scratch buffer (persistent; reused across frames, grows on demand) ---
    // The build completes synchronously (endSingleTimeCommands waits), so the same
    // scratch is free to reuse next frame — avoids another per-frame GPU alloc.
    uint64_t scratchAlignment = m_capabilities.minScratchAlignment > 0 ? m_capabilities.minScratchAlignment : 128;
    uint64_t scratchSize = performUpdate ? sizeInfo.updateScratchSize : sizeInfo.buildScratchSize;
    uint64_t alignedScratchSize = (scratchSize + scratchAlignment - 1) & ~(scratchAlignment - 1);

    if (m_tlasScratchBuffer.buffer == VK_NULL_HANDLE || m_tlasScratchBuffer.size < alignedScratchSize) {
        if (m_tlasScratchBuffer.buffer) destroyBuffer(m_tlasScratchBuffer);
        BufferCreateInfo scratchBufInfo;
        scratchBufInfo.size = alignedScratchSize;
        scratchBufInfo.usage = BufferUsage::STORAGE;
        scratchBufInfo.location = MemoryLocation::GPU_ONLY;
        m_tlasScratchBuffer = createBuffer(scratchBufInfo);
        if (!m_tlasScratchBuffer.buffer) {
            return;
        }
    }

    // --- 7) Build TLAS ---
    buildInfo.dstAccelerationStructure = m_tlas.accel;
    buildInfo.scratchData.deviceAddress = m_tlasScratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        return;
    }
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    endSingleTimeCommands(cmd);

    // Instance + scratch buffers are persistent device members now — nothing to free.

    // Update RT descriptor for TLAS (binding 1) if not performing an update (since it's in-place)
    if (!performUpdate && m_rtDescriptorSet != VK_NULL_HANDLE && m_tlas.accel) {
        VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
        asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        asWrite.accelerationStructureCount = 1;
        asWrite.pAccelerationStructures = &m_tlas.accel;

        VkWriteDescriptorSet w1{};
        w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1.dstSet = m_rtDescriptorSet;
        w1.dstBinding = 1;
        w1.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        w1.descriptorCount = 1;
        w1.pNext = &asWrite;
        vkUpdateDescriptorSets(m_device, 1, &w1, 0, nullptr);
    }

    m_tlasInstanceCount = instanceCount;
    m_tlasSupportsUpdate = info.allowUpdate;

    //VK_INFO() << "[VulkanDevice] TLAS " << (performUpdate ? "updated" : "created") << " (" << instanceCount << " instances)" << std::endl;
}


void VulkanDevice::updateTLAS(const std::vector<TLASInstance>& instances) {
    // Rebuild TLAS with updated transforms
    TLASCreateInfo info;
    info.instances = instances;
    info.allowUpdate = true;
    createTLAS(info);
}

void VulkanDevice::traceRays(uint32_t w, uint32_t h, uint32_t d) {
    // [CRASH GUARD] Reject dispatch if pipeline is unready OR if TLAS was destroyed mid-rebuild.
    // rebuildAccelerationStructure() sets m_tlas.accel = VK_NULL_HANDLE but does NOT yet
    // recreate the RT pipeline flag, so we must check TLAS validity independently.
    if (!m_rtPipelineReady || !fpCmdTraceRaysKHR || !m_tlas.accel) {
        if (!m_tlas.accel && m_rtPipelineReady) {
            VK_ERROR() << "[VulkanDevice] traceRays skipped: TLAS not yet built (rebuild in progress)." ;
        } else {
            VK_ERROR() << "[VulkanDevice] RT pipeline not ready for traceRays!" ;
        }
        return;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;

    // Bind RT pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);

    // Bind RT descriptor set
    if (m_rtDescriptorSet) {
       // VK_INFO() << "[VulkanDevice] traceRays - binding RT descriptor set: " << (void*)m_rtDescriptorSet << std::endl;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            m_rtPipelineLayout, 0, 1, &m_rtDescriptorSet, 0, nullptr);
    }

    // Push constants
    if (!m_pushConstantData.empty()) {
        vkCmdPushConstants(cmd, m_rtPipelineLayout,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
            0, (uint32_t)m_pushConstantData.size(), m_pushConstantData.data());
    }

    // Trace rays!
    fpCmdTraceRaysKHR(cmd,
        &m_sbtRaygenRegion,
        &m_sbtMissRegion,
        &m_sbtHitRegion,
        &m_sbtCallableRegion,
        w, h, d);

    endSingleTimeCommands(cmd);
}

// ========================================================================
// Photon caustic pass (Faz 2 / Dilim 1)
// ========================================================================
// Two-step: schedulePhotonPass() writes the header + arms a pending flag on
// the CPU; recordPhotonPass() then records clear + photon trace + barriers at
// the START of the camera trace command buffer. Recording into the SAME
// command buffer is essential: a separate synchronous submit raced the async
// ping-pong camera reads — the next frame's grid clear executed while the
// previous frame's camera rays were still reading, so the grid appeared
// mostly empty with occasional one-frame flashes.
void VulkanDevice::schedulePhotonPass(const PhotonGridHeader& header, const PhotonGridHeader& volHeader, bool clearGrid) {
    m_photonPassPending = false;
    if (!m_rtPipelineReady || !m_hasPhotonRaygen || !fpCmdTraceRaysKHR || !m_tlas.accel) return;
    if (!m_photonGridBuffer.buffer) return;
    if (header.photonCount == 0 || header.lightCountReal == 0) return;

    // Headers are host-written (CPU_TO_GPU buffers); the cell regions are
    // cleared on-GPU inside the camera command buffer (only on accum reset).
    if (void* mapped = mapBuffer(m_photonGridBuffer)) {
        PhotonGridHeader h = header;
        h.tableSize = m_photonTableSize;   // single authority: the allocated table
        memcpy(mapped, &h, sizeof(PhotonGridHeader));
        unmapBuffer(m_photonGridBuffer);
    } else {
        return;
    }
    if (m_photonVolGridBuffer.buffer) {
        if (void* mapped = mapBuffer(m_photonVolGridBuffer)) {
            PhotonGridHeader h = volHeader;
            h.tableSize = m_photonTableSize;
            memcpy(mapped, &h, sizeof(PhotonGridHeader));
            unmapBuffer(m_photonVolGridBuffer);
        }
    }

    m_pendingPhotonCount = header.photonCount;
    m_pendingPhotonClear = clearGrid;
    m_photonPassPending = true;
}

void VulkanDevice::disablePhotonGrid() {
    m_photonPassPending = false;
    if (m_photonGridBuffer.buffer) {
        if (void* mapped = mapBuffer(m_photonGridBuffer)) {
            static_cast<PhotonGridHeader*>(mapped)->debugMode = 0u;
            unmapBuffer(m_photonGridBuffer);
        }
    }
    if (m_photonVolGridBuffer.buffer) {
        if (void* mapped = mapBuffer(m_photonVolGridBuffer)) {
            static_cast<PhotonGridHeader*>(mapped)->debugMode = 0u;
            unmapBuffer(m_photonVolGridBuffer);
        }
    }
}

void VulkanDevice::recordPhotonPass(VkCommandBuffer cmd) {
    if (!m_photonPassPending) return;
    m_photonPassPending = false;
    if (!m_hasPhotonRaygen || !m_photonGridBuffer.buffer || !m_rtDescriptorSet) return;

    if (m_pendingPhotonClear) {
        // First frame after an accumulation reset: clear the cell region
        // (header preserved). Subsequent frames ACCUMULATE photons on top —
        // progressive photon mapping; readers divide by frameSeed+1.
        const VkDeviceSize cellBytes = (VkDeviceSize)m_photonTableSize * 20ull; // 5 uints/cell: R,G,B,count,key
        vkCmdFillBuffer(cmd, m_photonGridBuffer.buffer, sizeof(PhotonGridHeader), cellBytes, 0u);
        if (m_photonVolGridBuffer.buffer) {
            vkCmdFillBuffer(cmd, m_photonVolGridBuffer.buffer, sizeof(PhotonGridHeader), cellBytes, 0u);
        }

        VkMemoryBarrier clearBarrier{};
        clearBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        clearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        clearBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             0, 1, &clearBarrier, 0, nullptr, 0, nullptr);
    } else {
        // Make the previous frame's photon atomics visible before adding more
        // (the two ping-pong slots may overlap on the GPU).
        VkMemoryBarrier prevBarrier{};
        prevBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        prevBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        prevBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             0, 1, &prevBarrier, 0, nullptr, 0, nullptr);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            m_rtPipelineLayout, 0, 1, &m_rtDescriptorSet, 0, nullptr);

    // Push a COPY of the camera push constants with lightCount zeroed so
    // closesthit skips its per-bounce NEE work during photon tracing —
    // photon.rgen reads the real light data from the LightBuffer via the grid
    // header instead. lightCount sits at byte offset 72 (4x vec4 camera block +
    // frameCount + minSamples); see CameraPushConstants in renderProgressive.
    // The caller re-pushes the original constants for the camera trace after this.
    if (!m_pushConstantData.empty()) {
        std::vector<uint8_t> pc = m_pushConstantData;
        if (pc.size() >= 76) std::memset(pc.data() + 72, 0, 4);
        vkCmdPushConstants(cmd, m_rtPipelineLayout,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
            0, (uint32_t)pc.size(), pc.data());
    }

    // 1D photon budget folded into a 2D launch.
    const uint32_t launchW = 8192u;
    const uint32_t launchH = std::max(1u, (m_pendingPhotonCount + launchW - 1u) / launchW);
    fpCmdTraceRaysKHR(cmd, &m_sbtPhotonRegion, &m_sbtMissRegion,
                      &m_sbtHitRegion, &m_sbtCallableRegion,
                      launchW, launchH, 1);

    // Photon grid writes → camera-trace reads (same command buffer).
    VkMemoryBarrier gatherBarrier{};
    gatherBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    gatherBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    gatherBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 1, &gatherBarrier, 0, nullptr, 0, nullptr);
}

// ========================================================================
// RT Pipeline Creation
// ========================================================================

bool VulkanDevice::createRTPipeline(const std::vector<std::uint32_t>& raygenSPV,
                                     const std::vector<std::uint32_t>& missSPV,
                                     const std::vector<std::uint32_t>& closestHitSPV,
                                     const std::vector<std::uint32_t>& anyHitSPV,
                                     const std::vector<std::uint32_t>& volumeClosestHitSPV,
                                     const std::vector<std::uint32_t>& volumeIntersectionSPV,
                                     const std::vector<std::uint32_t>& hairClosestHitSPV,
                                     const std::vector<std::uint32_t>& hairIntersectionSPV,
                                     const std::vector<std::uint32_t>& shadowMissSPV,
                                     const std::vector<std::uint32_t>& hairAnyHitSPV,
                                     const std::vector<std::uint32_t>& sphereClosestHitSPV,
                                     const std::vector<std::uint32_t>& sphereIntersectionSPV,
                                     const std::vector<std::uint32_t>& photonRaygenSPV) {
    if (!hasHardwareRT() || !fpCreateRayTracingPipelinesKHR) {
        VK_ERROR() << "[VulkanDevice] Hardware RT not available" << std::endl;
        return false;
    }

    VK_INFO() << "[VulkanDevice] Creating RT pipeline..." << std::endl;

    // --- 1) Create shader modules ---
    auto createModule = [&](const std::vector<std::uint32_t>& code) -> VkShaderModule {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = code.size() * sizeof(uint32_t);
        ci.pCode = code.data();
        VkShaderModule mod;
        vkCreateShaderModule(m_device, &ci, nullptr, &mod);
        return mod;
    };

    VkShaderModule raygenModule  = raygenSPV.empty()              ? VK_NULL_HANDLE : createModule(raygenSPV);
    VkShaderModule missModule    = missSPV.empty()                ? VK_NULL_HANDLE : createModule(missSPV);
    VkShaderModule chitModule    = closestHitSPV.empty()          ? VK_NULL_HANDLE : createModule(closestHitSPV);
    VkShaderModule anyhitModule  = anyHitSPV.empty()              ? VK_NULL_HANDLE : createModule(anyHitSPV);
    VkShaderModule volChitModule = volumeClosestHitSPV.empty()    ? VK_NULL_HANDLE : createModule(volumeClosestHitSPV);
    VkShaderModule volIntModule  = volumeIntersectionSPV.empty()  ? VK_NULL_HANDLE : createModule(volumeIntersectionSPV);
    VkShaderModule hairChitModule= hairClosestHitSPV.empty()      ? VK_NULL_HANDLE : createModule(hairClosestHitSPV);
    VkShaderModule hairIntModule = hairIntersectionSPV.empty()    ? VK_NULL_HANDLE : createModule(hairIntersectionSPV);
    VkShaderModule shadowMissModule = shadowMissSPV.empty()       ? VK_NULL_HANDLE : createModule(shadowMissSPV);
    VkShaderModule hairAnyHitModule = hairAnyHitSPV.empty()       ? VK_NULL_HANDLE : createModule(hairAnyHitSPV);
    VkShaderModule sphereChitModule = sphereClosestHitSPV.empty()   ? VK_NULL_HANDLE : createModule(sphereClosestHitSPV);
    VkShaderModule sphereIntModule  = sphereIntersectionSPV.empty() ? VK_NULL_HANDLE : createModule(sphereIntersectionSPV);
    VkShaderModule photonRgenModule = photonRaygenSPV.empty()       ? VK_NULL_HANDLE : createModule(photonRaygenSPV);

    bool hasVolume     = (volChitModule  != VK_NULL_HANDLE && volIntModule  != VK_NULL_HANDLE);
    bool hasHair       = (hairChitModule != VK_NULL_HANDLE && hairIntModule != VK_NULL_HANDLE);
    bool hasSphere     = (sphereChitModule != VK_NULL_HANDLE && sphereIntModule != VK_NULL_HANDLE);
    bool hasShadowMiss = (shadowMissModule != VK_NULL_HANDLE);
    bool hasPhoton     = (photonRgenModule != VK_NULL_HANDLE);

    if (raygenModule == VK_NULL_HANDLE || missModule == VK_NULL_HANDLE || chitModule == VK_NULL_HANDLE) {
        if (raygenModule) vkDestroyShaderModule(m_device, raygenModule, nullptr);
        if (missModule)   vkDestroyShaderModule(m_device, missModule, nullptr);
        if (chitModule)   vkDestroyShaderModule(m_device, chitModule, nullptr);
        if (volChitModule)    vkDestroyShaderModule(m_device, volChitModule, nullptr);
        if (volIntModule)     vkDestroyShaderModule(m_device, volIntModule, nullptr);
        if (hairChitModule)   vkDestroyShaderModule(m_device, hairChitModule, nullptr);
        if (hairIntModule)    vkDestroyShaderModule(m_device, hairIntModule, nullptr);
        if (shadowMissModule) vkDestroyShaderModule(m_device, shadowMissModule, nullptr);
        if (hairAnyHitModule) vkDestroyShaderModule(m_device, hairAnyHitModule, nullptr);
        if (photonRgenModule) vkDestroyShaderModule(m_device, photonRgenModule, nullptr);
        VK_ERROR() << "[VulkanDevice] Failed to load RT shader modules!" << std::endl;
        return false;
    }

    // --- 2) Pipeline shader stages ---
    // Stage order: raygen(0), primary_miss(1), [shadow_miss(2)?], closesthit(2or3),
    //              [anyhit?], [vol_chit?], [vol_int?], [hair_chit?], [hair_int?]
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    stages.reserve(11);

    auto makeStage = [](VkShaderStageFlagBits stageBit, VkShaderModule mod) {
        VkPipelineShaderStageCreateInfo s{};
        s.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        s.stage  = stageBit;
        s.module = mod;
        s.pName  = "main";
        return s;
    };

    // Raygen (stage 0)
    uint32_t raygenStageIdx = (uint32_t)stages.size();
    stages.push_back(makeStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, raygenModule));

    // Primary miss (stage 1)
    uint32_t primaryMissStageIdx = (uint32_t)stages.size();
    stages.push_back(makeStage(VK_SHADER_STAGE_MISS_BIT_KHR, missModule));

    // Shadow miss (optional, stage 2 when present)
    uint32_t shadowMissStageIdx = VK_SHADER_UNUSED_KHR;
    if (hasShadowMiss) {
        shadowMissStageIdx = (uint32_t)stages.size();
        stages.push_back(makeStage(VK_SHADER_STAGE_MISS_BIT_KHR, shadowMissModule));
        VK_INFO() << "[VulkanDevice] Shadow miss shader loaded (stage=" << shadowMissStageIdx << ")" << std::endl;
    }

    // Triangle closest hit
    uint32_t chitStageIdx = (uint32_t)stages.size();
    stages.push_back(makeStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chitModule));

    // Triangle any hit (optional)
    uint32_t anyhitStageIdx = VK_SHADER_UNUSED_KHR;
    if (anyhitModule != VK_NULL_HANDLE) {
        anyhitStageIdx = (uint32_t)stages.size();
        stages.push_back(makeStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, anyhitModule));
    }

    // Volume shader stages (appended after triangle stages)
    uint32_t volChitStageIdx = VK_SHADER_UNUSED_KHR;
    uint32_t volIntStageIdx  = VK_SHADER_UNUSED_KHR;
    if (hasVolume) {
        volChitStageIdx = (uint32_t)stages.size();
        stages.push_back(makeStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, volChitModule));
        volIntStageIdx = (uint32_t)stages.size();
        stages.push_back(makeStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, volIntModule));
        VK_INFO() << "[VulkanDevice] Volume shaders loaded (closesthit stage=" << volChitStageIdx << ", intersection stage=" << volIntStageIdx << ")" << std::endl;
    }

    // Hair shader stages (appended last)
    uint32_t hairChitStageIdx = VK_SHADER_UNUSED_KHR;
    uint32_t hairIntStageIdx  = VK_SHADER_UNUSED_KHR;
    if (hasHair) {
        hairChitStageIdx = (uint32_t)stages.size();
        stages.push_back(makeStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, hairChitModule));
        hairIntStageIdx = (uint32_t)stages.size();
        stages.push_back(makeStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, hairIntModule));

        uint32_t hairAnyHitStageIdx = VK_SHADER_UNUSED_KHR;
        if (hairAnyHitModule != VK_NULL_HANDLE) {
            hairAnyHitStageIdx = (uint32_t)stages.size();
            stages.push_back(makeStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, hairAnyHitModule));
        }

        VK_INFO() << "[VulkanDevice] Hair shaders loaded (closesthit stage=" << hairChitStageIdx
                  << ", intersection stage=" << hairIntStageIdx
                  << (hairAnyHitModule != VK_NULL_HANDLE ? ", anyhit stage=" + std::to_string(hairAnyHitStageIdx) : "") << ")" << std::endl;
    }

    // Foam point-sphere shader stages (appended last)
    uint32_t sphereChitStageIdx = VK_SHADER_UNUSED_KHR;
    uint32_t sphereIntStageIdx  = VK_SHADER_UNUSED_KHR;
    if (hasSphere) {
        sphereChitStageIdx = (uint32_t)stages.size();
        stages.push_back(makeStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, sphereChitModule));
        sphereIntStageIdx = (uint32_t)stages.size();
        stages.push_back(makeStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, sphereIntModule));
        VK_INFO() << "[VulkanDevice] Foam sphere shaders loaded (closesthit stage=" << sphereChitStageIdx
                  << ", intersection stage=" << sphereIntStageIdx << ")" << std::endl;
    }

    // Photon caustic raygen (Faz 2) — SECOND raygen group in the same pipeline.
    // Photons reuse the existing hit/miss shaders; tracePhotons() dispatches with
    // its own SBT raygen region pointing at this group's handle.
    uint32_t photonStageIdx = VK_SHADER_UNUSED_KHR;
    if (hasPhoton) {
        photonStageIdx = (uint32_t)stages.size();
        stages.push_back(makeStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, photonRgenModule));
        VK_INFO() << "[VulkanDevice] Photon caustic raygen loaded (stage=" << photonStageIdx << ")" << std::endl;
    }

    // --- 3) Shader groups ---
    // Group layout:
    //   [raygenGroupIdx]     General  — raygen
    //   [missGroupIdx]       General  — primary miss (miss index 0)
    //   [shadowMissGroupIdx] General  — shadow miss  (miss index 1, optional)
    //   [triHitGroupIdx]     Triangles hit group (hit index 0)
    //   [volHitGroupIdx]     Procedural hit group (hit index 1, optional)
    //   [hairHitGroupIdx]    Procedural hit group (hit index 1 or 2, optional)
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

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

    // Raygen group
    uint32_t raygenGroupIdx = (uint32_t)groups.size();
    groups.push_back(makeGeneralGroup(raygenStageIdx));

    // Primary miss group
    uint32_t missGroupIdx = (uint32_t)groups.size();
    groups.push_back(makeGeneralGroup(primaryMissStageIdx));

    // Shadow miss group (optional)
    uint32_t shadowMissGroupIdx = missGroupIdx; // falls back if not present
    if (hasShadowMiss) {
        shadowMissGroupIdx = (uint32_t)groups.size();
        groups.push_back(makeGeneralGroup(shadowMissStageIdx));
    }

    // Triangle hit group
    uint32_t triHitGroupIdx = (uint32_t)groups.size();
    {
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        g.generalShader      = VK_SHADER_UNUSED_KHR;
        g.closestHitShader   = chitStageIdx;
        g.anyHitShader       = anyhitStageIdx; // VK_SHADER_UNUSED_KHR when absent
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        groups.push_back(g);
    }

    // Volume procedural hit group (optional)
    uint32_t volHitGroupIdx = triHitGroupIdx; // fallback
    if (hasVolume) {
        volHitGroupIdx = (uint32_t)groups.size();
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        g.generalShader      = VK_SHADER_UNUSED_KHR;
        g.closestHitShader   = volChitStageIdx;
        g.anyHitShader       = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = volIntStageIdx;
        groups.push_back(g);
        VK_INFO() << "[VulkanDevice] Volume procedural hit group added (group index " << volHitGroupIdx << ")" << std::endl;
    }

    // Hair procedural hit group (optional)
    uint32_t hairHitGroupIdx = triHitGroupIdx; // fallback
    if (hasHair) {
        hairHitGroupIdx = (uint32_t)groups.size();
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        g.generalShader      = VK_SHADER_UNUSED_KHR;
        g.closestHitShader   = hairChitStageIdx;
        
        // Use the newly added hair any-hit stage if available
        uint32_t hairAhStage = VK_SHADER_UNUSED_KHR;
        if (hairAnyHitModule != VK_NULL_HANDLE) {
            // Find the stage index for hairAnyHitModule
            for (size_t s = 0; s < stages.size(); ++s) {
                if (stages[s].module == hairAnyHitModule) {
                    hairAhStage = (uint32_t)s;
                    break;
                }
            }
        }
        g.anyHitShader       = hairAhStage;
        g.intersectionShader = hairIntStageIdx;
        groups.push_back(g);
        VK_INFO() << "[VulkanDevice] Hair procedural hit group added (group index " << hairHitGroupIdx << ")" << std::endl;
    }

    // Foam point-sphere procedural hit group (optional, always last)
    uint32_t sphereHitGroupIdx = triHitGroupIdx; // fallback
    if (hasSphere) {
        sphereHitGroupIdx = (uint32_t)groups.size();
        VkRayTracingShaderGroupCreateInfoKHR g{};
        g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        g.generalShader      = VK_SHADER_UNUSED_KHR;
        g.closestHitShader   = sphereChitStageIdx;
        g.anyHitShader       = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = sphereIntStageIdx;
        groups.push_back(g);
        VK_INFO() << "[VulkanDevice] Foam sphere procedural hit group added (group index " << sphereHitGroupIdx << ")" << std::endl;
    }

    // Photon raygen group — appended AFTER all hit groups so the contiguous
    // miss/hit SBT regions keep their existing group indices.
    uint32_t photonGroupIdx = VK_SHADER_UNUSED_KHR;
    if (hasPhoton) {
        photonGroupIdx = (uint32_t)groups.size();
        groups.push_back(makeGeneralGroup(photonStageIdx));
        VK_INFO() << "[VulkanDevice] Photon caustic raygen group added (group index " << photonGroupIdx << ")" << std::endl;
    }

    // --- 4) Descriptor set layout ---
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
    VkDescriptorSetLayoutBinding bindings[21] = {};
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
    bindings[8].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[9].binding = 9;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

    // Binding 10: Hair Segment SSBO
    bindings[10].binding = 10;
    bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[10].descriptorCount = 1;
    bindings[10].stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 11: Hair Material SSBO
    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

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

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = 21;
    dslCI.pBindings = bindings;
    vkCreateDescriptorSetLayout(m_device, &dslCI, nullptr, &m_rtDescriptorSetLayout);

    // --- 5) Push constant range (camera data + rendering params) ---
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    pushRange.offset = 0;
    pushRange.size = 256; // Matches the expanded CameraPushConstants payload.

    // --- 6) Pipeline layout ---
    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &m_rtDescriptorSetLayout;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(m_device, &plCI, nullptr, &m_rtPipelineLayout);

    // --- 7) Create RT pipeline ---
    VkRayTracingPipelineCreateInfoKHR rtCI{};
    rtCI.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtCI.stageCount = (uint32_t)stages.size();
    rtCI.pStages = stages.data();
    rtCI.groupCount = (uint32_t)groups.size();
    rtCI.pGroups = groups.data();
    rtCI.maxPipelineRayRecursionDepth = 2; // Required for shadow rays from closesthit
    rtCI.layout = m_rtPipelineLayout;

    VkResult result = fpCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, VK_NULL_HANDLE,
        1, &rtCI, nullptr, &m_rtPipeline);
    // If pipeline creation fails and an any-hit module was provided, retry without any-hit
    if (result != VK_SUCCESS && anyhitModule != VK_NULL_HANDLE) {
        VK_WARN() << "[VulkanDevice] vkCreateRayTracingPipelinesKHR failed (with any-hit): " << result 
                  << ". anyhitModule=" << (anyhitModule != VK_NULL_HANDLE ? "VALID" : "NULL")
                  << " stages=" << stages.size() << " groups=" << groups.size()
                  << ". Retrying without any-hit..." << std::endl;

        // Rebuild stages without anyhit: raygen(0), primary_miss(1), [shadow_miss(2)?],
        //   closesthit(2or3), [vol_chit?], [vol_int?], [hair_chit?], [hair_int?]
        std::vector<VkPipelineShaderStageCreateInfo> stages2;
        stages2.reserve(11);
        uint32_t s2RaygenIdx = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, raygenModule));
        uint32_t s2PrimaryMissIdx = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_MISS_BIT_KHR, missModule));
        uint32_t s2ShadowMissIdx = VK_SHADER_UNUSED_KHR;
        if (hasShadowMiss) { s2ShadowMissIdx = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_MISS_BIT_KHR, shadowMissModule)); }
        uint32_t s2ChitIdx = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chitModule));
        // Re-add anyhit (opacity test for shadow rays) — this was the original failure point
        uint32_t s2AnyHitIdx = VK_SHADER_UNUSED_KHR;
        if (anyhitModule != VK_NULL_HANDLE) {
            s2AnyHitIdx = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, anyhitModule));
        }

        // Re-add volume stages with corrected indices
        uint32_t s2VolChitIdx = VK_SHADER_UNUSED_KHR, s2VolIntIdx = VK_SHADER_UNUSED_KHR;
        if (hasVolume && volChitModule != VK_NULL_HANDLE && volIntModule != VK_NULL_HANDLE) {
            s2VolChitIdx = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, volChitModule));
            s2VolIntIdx  = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, volIntModule));
        } else if (hasVolume) {
            hasVolume = false;
        }

        // Re-add hair stages with corrected indices
        uint32_t s2HairChitIdx = VK_SHADER_UNUSED_KHR, s2HairIntIdx = VK_SHADER_UNUSED_KHR;
        if (hasHair && hairChitModule != VK_NULL_HANDLE && hairIntModule != VK_NULL_HANDLE) {
            s2HairChitIdx = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, hairChitModule));
            s2HairIntIdx  = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, hairIntModule));
        } else if (hasHair) {
            hasHair = false;
        }

        // Re-add foam sphere stages with corrected indices
        uint32_t s2SphereChitIdx = VK_SHADER_UNUSED_KHR, s2SphereIntIdx = VK_SHADER_UNUSED_KHR;
        if (hasSphere && sphereChitModule != VK_NULL_HANDLE && sphereIntModule != VK_NULL_HANDLE) {
            s2SphereChitIdx = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, sphereChitModule));
            s2SphereIntIdx  = (uint32_t)stages2.size(); stages2.push_back(makeStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, sphereIntModule));
        } else if (hasSphere) {
            hasSphere = false;
        }

        // Rebuild groups with corrected stage indices
        groups.clear();
        uint32_t r2RaygenGroup = (uint32_t)groups.size(); groups.push_back(makeGeneralGroup(s2RaygenIdx));
        uint32_t r2MissGroup   = (uint32_t)groups.size(); groups.push_back(makeGeneralGroup(s2PrimaryMissIdx));
        if (hasShadowMiss) { groups.push_back(makeGeneralGroup(s2ShadowMissIdx)); }
        triHitGroupIdx = (uint32_t)groups.size();
        { VkRayTracingShaderGroupCreateInfoKHR g{}; g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR; g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR; g.generalShader = VK_SHADER_UNUSED_KHR; g.closestHitShader = s2ChitIdx; g.anyHitShader = s2AnyHitIdx; g.intersectionShader = VK_SHADER_UNUSED_KHR; groups.push_back(g); }
        if (hasVolume) { volHitGroupIdx = (uint32_t)groups.size(); VkRayTracingShaderGroupCreateInfoKHR g{}; g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR; g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR; g.generalShader = VK_SHADER_UNUSED_KHR; g.closestHitShader = s2VolChitIdx; g.anyHitShader = VK_SHADER_UNUSED_KHR; g.intersectionShader = s2VolIntIdx; groups.push_back(g); }
        if (hasHair)   { hairHitGroupIdx = (uint32_t)groups.size(); VkRayTracingShaderGroupCreateInfoKHR g{}; g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR; g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR; g.generalShader = VK_SHADER_UNUSED_KHR; g.closestHitShader = s2HairChitIdx; g.anyHitShader = VK_SHADER_UNUSED_KHR; g.intersectionShader = s2HairIntIdx; groups.push_back(g); }
        if (hasSphere) { sphereHitGroupIdx = (uint32_t)groups.size(); VkRayTracingShaderGroupCreateInfoKHR g{}; g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR; g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR; g.generalShader = VK_SHADER_UNUSED_KHR; g.closestHitShader = s2SphereChitIdx; g.anyHitShader = VK_SHADER_UNUSED_KHR; g.intersectionShader = s2SphereIntIdx; groups.push_back(g); }
        // Photon raygen re-added last (same ordering contract as the primary path)
        if (hasPhoton) {
            uint32_t s2PhotonIdx = (uint32_t)stages2.size();
            stages2.push_back(makeStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, photonRgenModule));
            photonGroupIdx = (uint32_t)groups.size();
            groups.push_back(makeGeneralGroup(s2PhotonIdx));
        }
        (void)r2RaygenGroup; (void)r2MissGroup;

        rtCI.stageCount = (uint32_t)stages2.size();
        rtCI.pStages = stages2.data();
        rtCI.groupCount = (uint32_t)groups.size();
        rtCI.pGroups = groups.data();

        result = fpCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtCI, nullptr, &m_rtPipeline);
        if (result != VK_SUCCESS) {
            VK_WARN() << "[VulkanDevice] Retry also failed: " << result << std::endl;
        }
    }

    // Cleanup shader modules (safe after pipeline creation attempt)
    if (raygenModule)     vkDestroyShaderModule(m_device, raygenModule, nullptr);
    if (missModule)       vkDestroyShaderModule(m_device, missModule, nullptr);
    if (chitModule)       vkDestroyShaderModule(m_device, chitModule, nullptr);
    if (anyhitModule)     vkDestroyShaderModule(m_device, anyhitModule, nullptr);
    if (volChitModule)    vkDestroyShaderModule(m_device, volChitModule, nullptr);
    if (volIntModule)     vkDestroyShaderModule(m_device, volIntModule, nullptr);
    if (hairChitModule)   vkDestroyShaderModule(m_device, hairChitModule, nullptr);
    if (hairIntModule)    vkDestroyShaderModule(m_device, hairIntModule, nullptr);
    if (shadowMissModule) vkDestroyShaderModule(m_device, shadowMissModule, nullptr);
    if (hairAnyHitModule) vkDestroyShaderModule(m_device, hairAnyHitModule, nullptr);
    if (sphereChitModule) vkDestroyShaderModule(m_device, sphereChitModule, nullptr);
    if (sphereIntModule)  vkDestroyShaderModule(m_device, sphereIntModule, nullptr);
    if (photonRgenModule) vkDestroyShaderModule(m_device, photonRgenModule, nullptr);

    if (result != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] vkCreateRayTracingPipelinesKHR failed: " << result << std::endl;
        return false;
    }

    // --- 8) Build Shader Binding Table (SBT) ---
    // Store hair/shadow/volume state as members for consistent use during rendering
    m_hasVolumeShaders = hasVolume;
    m_hasHairShaders   = hasHair;
    m_hasSphereShaders = hasSphere;
    m_hasShadowMiss    = hasShadowMiss;

    uint32_t handleSize = m_capabilities.shaderGroupHandleSize;
    uint32_t handleAlignment = m_capabilities.shaderGroupBaseAlignment;
    if (handleAlignment == 0) handleAlignment = handleSize; // Fallback
    if (handleSize == 0) {
        VK_ERROR() << "[VulkanDevice] shaderGroupHandleSize is 0 — RT capabilities not queried!" << std::endl;
        return false;
    }
    // groupCount must match the actual pipeline group count
    uint32_t groupCount = (uint32_t)groups.size();

    // Aligned handle size (each entry must be aligned)
    uint32_t alignedHandleSize = (handleSize + (handleAlignment - 1)) & ~(handleAlignment - 1);

    // Get shader group handles
    uint32_t handleStorageSize = groupCount * handleSize;
    std::vector<uint8_t> handleData(handleStorageSize);
    VkResult sbtResult = fpGetRayTracingShaderGroupHandlesKHR(m_device, m_rtPipeline, 0, groupCount, handleStorageSize, handleData.data());
    if (sbtResult != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] fpGetRayTracingShaderGroupHandlesKHR failed: " << sbtResult << " (groupCount=" << groupCount << ", handleSize=" << handleSize << ")" << std::endl;
        return false;
    }

    // SBT layout: [raygen | miss(s) | hit(s)] each entry aligned
    uint64_t sbtSize = (uint64_t)alignedHandleSize * groupCount;

    BufferCreateInfo sbtBufInfo;
    sbtBufInfo.size = sbtSize;
    sbtBufInfo.usage = BufferUsage::SHADER_BINDING | BufferUsage::TRANSFER_DST;
    sbtBufInfo.location = MemoryLocation::CPU_TO_GPU;
    m_sbtBuffer = createBuffer(sbtBufInfo);

    // Write handles into SBT buffer with proper alignment
    auto* mapped = (uint8_t*)mapBuffer(m_sbtBuffer);
    for (uint32_t i = 0; i < groupCount; i++) {
        memcpy(mapped + i * alignedHandleSize, handleData.data() + i * handleSize, handleSize);
    }
    unmapBuffer(m_sbtBuffer);

    // Set SBT regions using explicit group indices
    VkDeviceAddress sbtAddr = m_sbtBuffer.deviceAddress;

    // Raygen region (always 1 entry)
    m_sbtRaygenRegion.deviceAddress = sbtAddr + (VkDeviceAddress)raygenGroupIdx * alignedHandleSize;
    m_sbtRaygenRegion.stride = alignedHandleSize;
    m_sbtRaygenRegion.size   = alignedHandleSize;

    // Miss region: primary_miss + optional shadow_miss (contiguous)
    uint32_t numMissGroups = 1u + (hasShadowMiss ? 1u : 0u);
    m_sbtMissRegion.deviceAddress = sbtAddr + (VkDeviceAddress)missGroupIdx * alignedHandleSize;
    m_sbtMissRegion.stride = alignedHandleSize;
    m_sbtMissRegion.size   = (VkDeviceSize)numMissGroups * alignedHandleSize;

    // Hit region: triangle + optional volume + optional hair + optional sphere (contiguous)
    uint32_t numHitGroups = 1u + (hasVolume ? 1u : 0u) + (hasHair ? 1u : 0u) + (hasSphere ? 1u : 0u);
    m_sbtHitRegion.deviceAddress = sbtAddr + (VkDeviceAddress)triHitGroupIdx * alignedHandleSize;
    m_sbtHitRegion.stride = alignedHandleSize;
    m_sbtHitRegion.size   = (VkDeviceSize)numHitGroups * alignedHandleSize;

    m_sbtCallableRegion = {}; // No callable shaders

    // Photon caustic raygen region (Faz 2) — same SBT buffer, its own raygen slot.
    m_hasPhotonRaygen = hasPhoton && (photonGroupIdx != VK_SHADER_UNUSED_KHR);
    if (m_hasPhotonRaygen) {
        m_sbtPhotonRegion.deviceAddress = sbtAddr + (VkDeviceAddress)photonGroupIdx * alignedHandleSize;
        m_sbtPhotonRegion.stride = alignedHandleSize;
        m_sbtPhotonRegion.size   = alignedHandleSize;
    } else {
        m_sbtPhotonRegion = {};
    }

    m_rtPipelineReady = true;
    VK_INFO() << "[VulkanDevice] RT pipeline + SBT created successfully! (groups=" << groupCount
              << ", volume=" << (m_hasVolumeShaders ? "YES" : "NO")
              << ", hair="   << (m_hasHairShaders   ? "YES" : "NO")
              << ", shadowMiss=" << (m_hasShadowMiss ? "YES" : "NO") << ")" << std::endl;
    return true;
}

void VulkanDevice::bindRTDescriptors(const ImageHandle& outputImage,
                                     const ImageHandle* denoiserColorImage,
                                     const ImageHandle* denoiserAlbedoImage,
                                     const ImageHandle* denoiserNormalImage,
                                     const ImageHandle* varianceImage,
                                     const ImageHandle* denoiserPositionImage) {
    if (!m_rtDescriptorSetLayout || !m_tlas.accel) {
        VK_ERROR() << "[VulkanDevice] Cannot bind RT descriptors: missing layout or TLAS" << std::endl;
        return;
    }

    // Ensure material and light buffers exist (create dummy if not uploaded yet)
    if (!m_materialBuffer.buffer) {
        VulkanRT::VkGpuMaterial defaultMat{};
        defaultMat.albedo_r = 0.8f; defaultMat.albedo_g = 0.8f; defaultMat.albedo_b = 0.8f; defaultMat.opacity = 1.0f;
        defaultMat.roughness = 0.5f; // roughness
        defaultMat.metallic = 0.0f;
        defaultMat.specular = 0.5f;
        defaultMat.ior = 1.45f;
        defaultMat.transmission = 0.0f;
        updateMaterialBuffer(&defaultMat, sizeof(VulkanRT::VkGpuMaterial), 1);
    }
    if (!m_lightBuffer.buffer) {
        VulkanRT::VkGpuLight defaultLight{};
        defaultLight.position[0] = 5.0f; defaultLight.position[1] = 10.0f; defaultLight.position[2] = 5.0f; defaultLight.position[3] = 0.0f; // Point
        defaultLight.color[0] = 1.0f; defaultLight.color[1] = 1.0f; defaultLight.color[2] = 1.0f; defaultLight.color[3] = 100.0f; // White, intensity 100
        updateLightBuffer(&defaultLight, sizeof(::VulkanRT::VkGpuLight), 1);
    }
    if (!m_hairMaterialBuffer.buffer) {
        VulkanRT::HairGpuMaterial defaultHair{};
        defaultHair.baseColor[0] = 0.8f; defaultHair.baseColor[1] = 0.5f; defaultHair.baseColor[2] = 0.3f;
        defaultHair.roughness = 0.2f;
        defaultHair.melanin = 0.5f;
        defaultHair.melaninRedness = 0.2f;
        defaultHair.ior = 1.55f;
        defaultHair.cuticleAngle = 0.05f;
        defaultHair.colorMode = 1; // Melanin
        defaultHair.radialRoughness = 0.3f;
        std::vector<VulkanRT::HairGpuMaterial> dummy(64, defaultHair);
        updateHairMaterialBuffer(dummy);
    }
    if (!m_hairSegmentBuffer.buffer) {
        VulkanRT::HairSegmentGPU dummySeg{};
        std::vector<VulkanRT::HairSegmentGPU> dummy(1, dummySeg);
        updateHairSegmentBuffer(dummy);
    }

    // Allocate geometry data if needed
    if (!m_geometryDataBuffer.buffer && !m_blasList.empty()) {
        std::vector<::VulkanRT::VkGeometryData> geoData;
        for (const auto& blas : m_blasList) {
            ::VulkanRT::VkGeometryData d;
            d.vertexAddr = blas.vertexBuffer.deviceAddress;
            d.normalAddr = blas.normalBuffer.deviceAddress;
            d.uvAddr = blas.uvBuffer.deviceAddress;
            d.indexAddr = blas.indexBuffer.deviceAddress;
            d.materialAddr = blas.materialIndexBuffer.buffer ? blas.materialIndexBuffer.deviceAddress : 0;
            geoData.push_back(d);
        }
        BufferCreateInfo ci;
        ci.size = geoData.size() * sizeof(VkGeometryData);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        ci.initialData = nullptr;
        m_geometryDataBuffer = createBuffer(ci);
        if (m_geometryDataBuffer.buffer && !geoData.empty()) {
            uploadBuffer(m_geometryDataBuffer, geoData.data(), ci.size);
        }
    }

    // Allocate descriptor set ONLY IF NOT ALREADY ALLOCATED
    if (m_rtDescriptorSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_rtDescriptorSetLayout;
           if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_rtDescriptorSet) != VK_SUCCESS) {
               VK_ERROR() << "[VulkanDevice] Failed to allocate RT descriptor set" << std::endl;
               return;
           }
           VK_INFO() << "[VulkanDevice] Allocated RT descriptor set: " << (void*)m_rtDescriptorSet << std::endl;
    }

    // binding 0: output storage image
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = outputImage.view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // binding 1: TLAS
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &m_tlas.accel;

    // binding 2: Materials
    VkDescriptorBufferInfo matInfo{};
    matInfo.buffer = m_materialBuffer.buffer;
    matInfo.offset = 0;
    matInfo.range = VK_WHOLE_SIZE;

    // binding 3: Lights
    VkDescriptorBufferInfo lightInfo{};
    lightInfo.buffer = m_lightBuffer.buffer;
    lightInfo.offset = 0;
    lightInfo.range = VK_WHOLE_SIZE;

    // binding 4: Geometry Data
    VkDescriptorBufferInfo geoInfo{};
    geoInfo.buffer = m_geometryDataBuffer.buffer;
    geoInfo.offset = 0;
    geoInfo.range = VK_WHOLE_SIZE;

    // binding 5: Instance Data
    VkDescriptorBufferInfo instInfo{};
    if (m_instanceDataBuffer.buffer) {
        instInfo.buffer = m_instanceDataBuffer.buffer;
        instInfo.offset = 0;
        instInfo.range = VK_WHOLE_SIZE;
    } else {
        // Fallback to material buffer if instance data is missing (to avoid null binding)
        instInfo.buffer = m_materialBuffer.buffer;
        instInfo.offset = 0;
        instInfo.range = 0;
    }

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(12);

    // Binding 0
    VkWriteDescriptorSet w0{};
    w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w0.dstSet = m_rtDescriptorSet;
    w0.dstBinding = 0;
    w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w0.descriptorCount = 1;
    w0.pImageInfo = &imageInfo;
    writes.push_back(w0);

    // Binding 1 (TLAS)
    VkWriteDescriptorSet w1{};
    w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w1.dstSet = m_rtDescriptorSet;
    w1.dstBinding = 1;
    w1.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w1.descriptorCount = 1;
    w1.pNext = &asWrite;
    writes.push_back(w1);

    // Binding 2 (Materials)
    VkWriteDescriptorSet w2{};
    w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w2.dstSet = m_rtDescriptorSet;
    w2.dstBinding = 2;
    w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w2.descriptorCount = 1;
    w2.pBufferInfo = &matInfo;
    writes.push_back(w2);

    // Binding 3 (Lights)
    VkWriteDescriptorSet w3{};
    w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w3.dstSet = m_rtDescriptorSet;
    w3.dstBinding = 3;
    w3.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w3.descriptorCount = 1;
    w3.pBufferInfo = &lightInfo;
    writes.push_back(w3);

    // Binding 4 (Geometry)
    VkWriteDescriptorSet w4{};
    w4.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w4.dstSet = m_rtDescriptorSet;
    w4.dstBinding = 4;
    w4.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w4.descriptorCount = 1;
    w4.pBufferInfo = &geoInfo;
    writes.push_back(w4);

    // Binding 5 (Instances)
    VkWriteDescriptorSet w5{};
    w5.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w5.dstSet = m_rtDescriptorSet;
    w5.dstBinding = 5;
    w5.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w5.descriptorCount = 1;
    w5.pBufferInfo = &instInfo;
    writes.push_back(w5);

    // Binding 7 (WorldData)
    VkDescriptorBufferInfo worldInfo{};
    if (m_worldBuffer.buffer) {
        worldInfo.buffer = m_worldBuffer.buffer;
        worldInfo.offset = 0;
        worldInfo.range = VK_WHOLE_SIZE;
    }
    VkWriteDescriptorSet w7{};
    w7.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w7.dstSet = m_rtDescriptorSet;
    w7.dstBinding = 7;
    w7.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w7.descriptorCount = 1;
    w7.pBufferInfo = &worldInfo;
    writes.push_back(w7);

    // Binding 9 (Volume Instances)
    VkDescriptorBufferInfo volInfo{};
    if (m_volumeBuffer.buffer) {
        volInfo.buffer = m_volumeBuffer.buffer;
        volInfo.offset = 0;
        volInfo.range = VK_WHOLE_SIZE;
    } else {
        // Fallback to material buffer if volume data is missing (to avoid null binding)
        // range must be > 0 per Vulkan spec (VUID-VkDescriptorBufferInfo-range-00341)
        volInfo.buffer = m_materialBuffer.buffer;
        volInfo.offset = 0;
        volInfo.range = VK_WHOLE_SIZE;
    }
    VkWriteDescriptorSet w9{};
    w9.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w9.dstSet = m_rtDescriptorSet;
    w9.dstBinding = 9;
    w9.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w9.descriptorCount = 1;
    w9.pBufferInfo = &volInfo;
    writes.push_back(w9);

    // Binding 10: Hair Segment SSBO (fallback to materialBuffer if empty)
    VkDescriptorBufferInfo hairSegInfo{};
    hairSegInfo.buffer = m_hairSegmentBuffer.buffer ? m_hairSegmentBuffer.buffer : m_materialBuffer.buffer;
    hairSegInfo.offset = 0;
    hairSegInfo.range  = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w10{};
    w10.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w10.dstSet = m_rtDescriptorSet;
    w10.dstBinding = 10;
    w10.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w10.descriptorCount = 1;
    w10.pBufferInfo = &hairSegInfo;
    writes.push_back(w10);

    // Binding 11: Hair Material SSBO (fallback to materialBuffer if empty)
    VkDescriptorBufferInfo hairMatInfo{};
    hairMatInfo.buffer = m_hairMaterialBuffer.buffer ? m_hairMaterialBuffer.buffer : m_materialBuffer.buffer;
    hairMatInfo.offset = 0;
    hairMatInfo.range  = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w11{};
    w11.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w11.dstSet = m_rtDescriptorSet;
    w11.dstBinding = 11;
    w11.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w11.descriptorCount = 1;
    w11.pBufferInfo = &hairMatInfo;
    writes.push_back(w11);

    // Binding 12: Terrain Layer SSBO (fallback to materialBuffer if empty)
    VkDescriptorBufferInfo terrainLayerInfo{};
    terrainLayerInfo.buffer = m_terrainLayerBuffer.buffer ? m_terrainLayerBuffer.buffer : m_materialBuffer.buffer;
    terrainLayerInfo.offset = 0;
    terrainLayerInfo.range  = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w12{};
    w12.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w12.dstSet = m_rtDescriptorSet;
    w12.dstBinding = 12;
    w12.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w12.descriptorCount = 1;
    w12.pBufferInfo = &terrainLayerInfo;
    writes.push_back(w12);

    VkDescriptorImageInfo denoiserColorInfo{};
    denoiserColorInfo.imageView = (denoiserColorImage && denoiserColorImage->view) ? denoiserColorImage->view : outputImage.view;
    denoiserColorInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w13{};
    w13.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w13.dstSet = m_rtDescriptorSet;
    w13.dstBinding = 13;
    w13.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w13.descriptorCount = 1;
    w13.pImageInfo = &denoiserColorInfo;
    writes.push_back(w13);

    VkDescriptorImageInfo denoiserAlbedoInfo{};
    denoiserAlbedoInfo.imageView = (denoiserAlbedoImage && denoiserAlbedoImage->view) ? denoiserAlbedoImage->view : outputImage.view;
    denoiserAlbedoInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w14{};
    w14.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w14.dstSet = m_rtDescriptorSet;
    w14.dstBinding = 14;
    w14.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w14.descriptorCount = 1;
    w14.pImageInfo = &denoiserAlbedoInfo;
    writes.push_back(w14);

    VkDescriptorImageInfo denoiserNormalInfo{};
    denoiserNormalInfo.imageView = (denoiserNormalImage && denoiserNormalImage->view) ? denoiserNormalImage->view : outputImage.view;
    denoiserNormalInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w15{};
    w15.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w15.dstSet = m_rtDescriptorSet;
    w15.dstBinding = 15;
    w15.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w15.descriptorCount = 1;
    w15.pImageInfo = &denoiserNormalInfo;
    writes.push_back(w15);

    VkDescriptorImageInfo varianceInfo{};
    varianceInfo.imageView = (varianceImage && varianceImage->view) ? varianceImage->view : outputImage.view;
    varianceInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w16{};
    w16.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w16.dstSet = m_rtDescriptorSet;
    w16.dstBinding = 16;
    w16.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w16.descriptorCount = 1;
    w16.pImageInfo = &varianceInfo;
    writes.push_back(w16);

    VkDescriptorImageInfo denoiserPositionInfo{};
    denoiserPositionInfo.imageView = (denoiserPositionImage && denoiserPositionImage->view) ? denoiserPositionImage->view : outputImage.view;
    denoiserPositionInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w17{};
    w17.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w17.dstSet = m_rtDescriptorSet;
    w17.dstBinding = 17;
    w17.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w17.descriptorCount = 1;
    w17.pImageInfo = &denoiserPositionInfo;
    writes.push_back(w17);

    // Binding 18: Foam point-sphere SSBO (fallback to materialBuffer when no foam
    // exists yet, so the binding is always valid; the foam consumer re-uploads it).
    VkDescriptorBufferInfo foamSphereInfo{};
    foamSphereInfo.buffer = m_foamSphereBuffer.buffer ? m_foamSphereBuffer.buffer : m_materialBuffer.buffer;
    foamSphereInfo.offset = 0;
    foamSphereInfo.range  = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w18{};
    w18.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w18.dstSet = m_rtDescriptorSet;
    w18.dstBinding = 18;
    w18.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w18.descriptorCount = 1;
    w18.pBufferInfo = &foamSphereInfo;
    writes.push_back(w18);

    // Binding 19: Photon caustic hash grid (Faz 2). Created lazily; CPU_TO_GPU so
    // tracePhotons() writes the 64-byte header directly before each pass while the
    // cell region is cleared on-GPU with vkCmdFillBuffer.
    if (!m_photonGridBuffer.buffer) {
        BufferCreateInfo pgci;
        pgci.size = sizeof(PhotonGridHeader) + (uint64_t)m_photonTableSize * 20ull; // 5 uints/cell: R,G,B,count,key
        pgci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        pgci.location = MemoryLocation::CPU_TO_GPU;
        pgci.initialData = nullptr;
        m_photonGridBuffer = createBuffer(pgci);
        if (m_photonGridBuffer.buffer) {
            // Zero the whole buffer once — the header (incl. debugMode) must not
            // start as garbage; raygen reads it every frame even when caustics
            // are off. tracePhotons re-clears the cell region per pass on-GPU.
            if (void* mapped = mapBuffer(m_photonGridBuffer)) {
                memset(mapped, 0, (size_t)pgci.size);
                unmapBuffer(m_photonGridBuffer);
            }
            VK_INFO() << "[VulkanDevice] Photon caustic grid allocated ("
                      << (pgci.size >> 20) << " MB, " << m_photonTableSize << " cells)" << std::endl;
        }
    }
    VkDescriptorBufferInfo photonGridInfo{};
    photonGridInfo.buffer = m_photonGridBuffer.buffer ? m_photonGridBuffer.buffer : m_materialBuffer.buffer;
    photonGridInfo.offset = 0;
    photonGridInfo.range  = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w19{};
    w19.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w19.dstSet = m_rtDescriptorSet;
    w19.dstBinding = 19;
    w19.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w19.descriptorCount = 1;
    w19.pBufferInfo = &photonGridInfo;
    writes.push_back(w19);

    // Binding 20: VOLUME photon grid (Faz 2V) — same layout/size as binding 19.
    if (!m_photonVolGridBuffer.buffer) {
        BufferCreateInfo pvci;
        pvci.size = sizeof(PhotonGridHeader) + (uint64_t)m_photonTableSize * 20ull;
        pvci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        pvci.location = MemoryLocation::CPU_TO_GPU;
        pvci.initialData = nullptr;
        m_photonVolGridBuffer = createBuffer(pvci);
        if (m_photonVolGridBuffer.buffer) {
            if (void* mapped = mapBuffer(m_photonVolGridBuffer)) {
                memset(mapped, 0, (size_t)pvci.size);   // header must not start as garbage
                unmapBuffer(m_photonVolGridBuffer);
            }
        }
    }
    VkDescriptorBufferInfo photonVolInfo{};
    photonVolInfo.buffer = m_photonVolGridBuffer.buffer ? m_photonVolGridBuffer.buffer : m_materialBuffer.buffer;
    photonVolInfo.offset = 0;
    photonVolInfo.range  = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w20{};
    w20.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w20.dstSet = m_rtDescriptorSet;
    w20.dstBinding = 20;
    w20.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w20.descriptorCount = 1;
    w20.pBufferInfo = &photonVolInfo;
    writes.push_back(w20);

    // Update bindings immediately (safe local buffers)
    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }

    // Binding 6: Material textures (runtime-sized array)
    // Update immediately to avoid dangling pointers from local vectors
    VkDescriptorSet descriptorSetSnapshot = VK_NULL_HANDLE;
    std::vector<std::pair<uint32_t, ImageHandle>> pendingTextureDescriptors;
    {
        std::lock_guard<std::mutex> lock(m_rtDescriptorMutex);
        descriptorSetSnapshot = m_rtDescriptorSet;
        if (!m_pendingTextureDescriptors.empty()) {
            pendingTextureDescriptors.swap(m_pendingTextureDescriptors);
        }
    }

    if (descriptorSetSnapshot != VK_NULL_HANDLE && !pendingTextureDescriptors.empty()) {
        std::vector<VkDescriptorImageInfo> extraImageInfos;
        std::vector<VkWriteDescriptorSet> extraWrites;

        extraImageInfos.reserve(pendingTextureDescriptors.size());
        extraWrites.reserve(pendingTextureDescriptors.size());

        // Build writes only for valid, still-live image handles.
        for (const auto& p : pendingTextureDescriptors) {
            const uint32_t slot = p.first;
            const ImageHandle& img = p.second;
            if (slot >= static_cast<uint32_t>(Backend::VULKAN_TEXTURE_CAPACITY)) continue;
            if (img.sampler == VK_NULL_HANDLE || img.view == VK_NULL_HANDLE || img.image == VK_NULL_HANDLE) continue;

            VkDescriptorImageInfo ii{};
            ii.sampler = img.sampler;
            ii.imageView = img.view;
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            extraImageInfos.push_back(ii);

            const size_t infoIdx = extraImageInfos.size() - 1;
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = descriptorSetSnapshot;
            w.dstBinding = 6;
            w.dstArrayElement = slot;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = 1;
            w.pImageInfo = &extraImageInfos[infoIdx];
            extraWrites.push_back(w);
        }

        if (!extraWrites.empty()) {
            vkUpdateDescriptorSets(m_device, (uint32_t)extraWrites.size(), extraWrites.data(), 0, nullptr);
        }
    }

    // Binding 8: Atmosphere LUT Samplers (4 textures)
    // Only update if at least one LUT is valid (avoid null descriptor updates)
    bool hasValidLUT = false;
    for (int i = 0; i < 4; i++) {
        if (m_lutImages[i].view != VK_NULL_HANDLE) {
            hasValidLUT = true;
            break;
        }
    }
    
    if (hasValidLUT) {
        std::vector<VkDescriptorImageInfo> lutInfos;
        std::vector<VkWriteDescriptorSet> lutWrites;
        lutInfos.reserve(4);
        lutWrites.reserve(4);

        for (uint32_t i = 0; i < 4; ++i) {
            if (m_lutImages[i].sampler == VK_NULL_HANDLE || m_lutImages[i].view == VK_NULL_HANDLE) {
                continue;
            }

            VkDescriptorImageInfo ii{};
            ii.sampler = m_lutImages[i].sampler;
            ii.imageView = m_lutImages[i].view;
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            lutInfos.push_back(ii);

            VkWriteDescriptorSet w8{};
            w8.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w8.dstSet = m_rtDescriptorSet;
            w8.dstBinding = 8;
            w8.dstArrayElement = i;
            w8.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w8.descriptorCount = 1;
            w8.pImageInfo = &lutInfos.back();
            lutWrites.push_back(w8);
        }

        if (!lutWrites.empty()) {
            vkUpdateDescriptorSets(m_device, (uint32_t)lutWrites.size(), lutWrites.data(), 0, nullptr);
        }
    }
}

// Update a single combined image sampler entry in the RT descriptor set (binding 6)
void VulkanDevice::updateRTTextureDescriptor(uint32_t slot, const ImageHandle& image) {
    if (slot >= static_cast<uint32_t>(Backend::VULKAN_TEXTURE_CAPACITY)) {
        VK_WARN() << "[VulkanDevice] Texture slot " << slot << " out of range for materialTextures array" << std::endl;
        return;
    }

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    // If descriptor set isn't allocated yet, queue the update to be applied later
    {
        std::lock_guard<std::mutex> lock(m_rtDescriptorMutex);
        descriptorSet = m_rtDescriptorSet;
        if (descriptorSet == VK_NULL_HANDLE) {
            m_pendingTextureDescriptors.emplace_back(slot, image);
            return;
        }
    }

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler = image.sampler;
    imgInfo.imageView = image.view;
    // Assume shader expects read-only optimal layout for sampled textures
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = 6;
    write.dstArrayElement = slot;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imgInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void VulkanDevice::clearPendingRTTextureDescriptors() {
    std::lock_guard<std::mutex> lock(m_rtDescriptorMutex);
    m_pendingTextureDescriptors.clear();
}

void VulkanDevice::removePendingRTTextureDescriptor(const ImageHandle& image) {
    std::lock_guard<std::mutex> lock(m_rtDescriptorMutex);
    m_pendingTextureDescriptors.erase(
        std::remove_if(
            m_pendingTextureDescriptors.begin(),
            m_pendingTextureDescriptors.end(),
            [&](const std::pair<uint32_t, ImageHandle>& p) {
                const auto& h = p.second;
                return h.image == image.image || h.view == image.view || h.sampler == image.sampler;
            }),
        m_pendingTextureDescriptors.end());
}


void VulkanDevice::updateMaterialBuffer(const void* data, uint64_t size, uint32_t count) {
    if (m_materialBuffer.size < size) {
        if (m_materialBuffer.buffer) destroyBuffer(m_materialBuffer);
        
        BufferCreateInfo ci;
        ci.size = size > 1024 ? size : 1024; // Min size
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_materialBuffer = createBuffer(ci);

        // Update descriptor if set already exists
        if (m_rtDescriptorSet != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo matInfo{};
            matInfo.buffer = m_materialBuffer.buffer;
            matInfo.offset = 0;
            matInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet w2{};
            w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w2.dstSet = m_rtDescriptorSet;
            w2.dstBinding = 2;
            w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w2.descriptorCount = 1;
            w2.pBufferInfo = &matInfo;
            vkUpdateDescriptorSets(m_device, 1, &w2, 0, nullptr);
        }
    }
    uploadBuffer(m_materialBuffer, data, size);
    m_materialCount = count;
}

void VulkanDevice::updateLightBuffer(const void* data, uint64_t size, uint32_t count) {
    if (m_lightBuffer.size < size) {
        if (m_lightBuffer.buffer) destroyBuffer(m_lightBuffer);
        
        BufferCreateInfo ci;
        ci.size = size > 1024 ? size : 1024;
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_lightBuffer = createBuffer(ci);

        // Update descriptor if set already exists
        if (m_rtDescriptorSet != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo lightInfo{};
            lightInfo.buffer = m_lightBuffer.buffer;
            lightInfo.offset = 0;
            lightInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet w3{};
            w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w3.dstSet = m_rtDescriptorSet;
            w3.dstBinding = 3;
            w3.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w3.descriptorCount = 1;
            w3.pBufferInfo = &lightInfo;
            vkUpdateDescriptorSets(m_device, 1, &w3, 0, nullptr);
        }
    }
    uploadBuffer(m_lightBuffer, data, size);
    m_lightCount = count;
}

// World buffer upload
void VulkanDevice::updateWorldBuffer(const void* data, uint64_t size, uint32_t count) {
    if (m_worldBuffer.size < size) {
        if (m_worldBuffer.buffer) destroyBuffer(m_worldBuffer);
        BufferCreateInfo ci;
        ci.size = size > 1024 ? size : 1024;
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_worldBuffer = createBuffer(ci);
    }
    uploadBuffer(m_worldBuffer, data, size);
    // Debug: if this appears to be a VkWorldDataExtended, dump LUT fields
    if (size >= sizeof(VulkanRT::VkWorldDataExtended)) {
        const VulkanRT::VkWorldDataExtended* gw = reinterpret_cast<const VulkanRT::VkWorldDataExtended*>(data);
        uint64_t sky = gw->skyviewLUT;
        uint64_t trans = gw->transmittanceLUT;
       
    }
    // If RT descriptor set already exists, update binding 7 so shaders read the current world buffer.
    if (m_rtDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo worldInfo{};
        worldInfo.buffer = m_worldBuffer.buffer;
        worldInfo.offset = 0;
        worldInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet w7{};
        w7.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w7.dstSet = m_rtDescriptorSet;
        w7.dstBinding = 7;
        w7.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w7.descriptorCount = 1;
        w7.pBufferInfo = &worldInfo;

        vkUpdateDescriptorSets(m_device, 1, &w7, 0, nullptr);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Volume Buffer Upload — OptiX-compatible VkVolumeInstance SSBO (binding 9)
// ════════════════════════════════════════════════════════════════════════════════
void VulkanDevice::updateVolumeBuffer(const void* data, uint64_t size, uint32_t count) {
    if (size == 0 || data == nullptr) {
        m_volumeCount = 0;
        return;
    }
    
    if (m_volumeBuffer.size < size) {
        if (m_volumeBuffer.buffer) destroyBuffer(m_volumeBuffer);
        BufferCreateInfo ci;
        ci.size = size > 1024 ? size : 1024;
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_volumeBuffer = createBuffer(ci);

        // Update descriptor if set already exists
        if (m_rtDescriptorSet != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo volInfo{};
            volInfo.buffer = m_volumeBuffer.buffer;
            volInfo.offset = 0;
            volInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet w9{};
            w9.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w9.dstSet = m_rtDescriptorSet;
            w9.dstBinding = 9;
            w9.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w9.descriptorCount = 1;
            w9.pBufferInfo = &volInfo;
            vkUpdateDescriptorSets(m_device, 1, &w9, 0, nullptr);
        }
    }
    uploadBuffer(m_volumeBuffer, data, size);
    m_volumeCount = count;
    // VK_INFO() << "[VulkanDevice] updateVolumeBuffer - " << count << " volume instances uploaded (" << size << " bytes)" << std::endl;
}

void VulkanDevice::updateTerrainLayerBuffer(const void* data, uint64_t size, uint32_t count) {
    if (size == 0 || data == nullptr) {
        m_terrainLayerCount = 0;
        return;
    }

    if (m_terrainLayerBuffer.size < size) {
        if (m_terrainLayerBuffer.buffer) destroyBuffer(m_terrainLayerBuffer);
        BufferCreateInfo ci;
        ci.size = size > 256 ? size : 256; // Min 256 bytes
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_terrainLayerBuffer = createBuffer(ci);

        // Update descriptor if set already exists
        if (m_rtDescriptorSet != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo terrainInfo{};
            terrainInfo.buffer = m_terrainLayerBuffer.buffer;
            terrainInfo.offset = 0;
            terrainInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet w12{};
            w12.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w12.dstSet = m_rtDescriptorSet;
            w12.dstBinding = 12;
            w12.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w12.descriptorCount = 1;
            w12.pBufferInfo = &terrainInfo;
            vkUpdateDescriptorSets(m_device, 1, &w12, 0, nullptr);
        }
    }
    uploadBuffer(m_terrainLayerBuffer, data, size);
    m_terrainLayerCount = count;
   // VK_INFO() << "[VulkanDevice] updateTerrainLayerBuffer - " << count << " terrain layers uploaded (" << size << " bytes)" << std::endl;
}

// ════════════════════════════════════════════════════════════════════════════════
// Hair AABB BLAS — one AABB per hair segment (procedural geometry)
// ════════════════════════════════════════════════════════════════════════════════
uint32_t VulkanDevice::createHairAABB_BLAS(const std::vector<VkAabbPositionsKHR>& aabbs) {
    if (!hasHardwareRT() || !fpCreateAccelerationStructureKHR) {
        VK_ERROR() << "[VulkanDevice] Hardware RT not available for hair AABB BLAS" << std::endl;
        return UINT32_MAX;
    }
    if (aabbs.empty()) return UINT32_MAX;

    // Upload AABB array to GPU
    const uint64_t aabbDataSize = aabbs.size() * sizeof(VkAabbPositionsKHR);
    BufferCreateInfo aabbBufInfo;
    aabbBufInfo.size = aabbDataSize;
    aabbBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    aabbBufInfo.location = MemoryLocation::CPU_TO_GPU;
    aabbBufInfo.initialData = const_cast<VkAabbPositionsKHR*>(aabbs.data());
    auto aabbBuffer = createBuffer(aabbBufInfo);
    if (!aabbBuffer.buffer) return UINT32_MAX;

    VkAccelerationStructureGeometryAabbsDataKHR aabbsData{};
    aabbsData.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbsData.data.deviceAddress = aabbBuffer.deviceAddress;
    aabbsData.stride = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags        = 0; // Not opaque — intersection shader decides
    geometry.geometry.aabbs = aabbsData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    uint32_t primitiveCount = (uint32_t)aabbs.size();

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fpGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    AccelStructHandle blasHandle{};
    BufferCreateInfo asBufInfo;
    asBufInfo.size     = sizeInfo.accelerationStructureSize;
    asBufInfo.usage    = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    asBufInfo.location = MemoryLocation::GPU_ONLY;
    blasHandle.buffer  = createBuffer(asBufInfo);
    if (!blasHandle.buffer.buffer) {
        destroyBuffer(aabbBuffer);
        return UINT32_MAX;
    }

    VkAccelerationStructureCreateInfoKHR asCI{};
    asCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = blasHandle.buffer.buffer;
    asCI.size   = sizeInfo.accelerationStructureSize;
    asCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (fpCreateAccelerationStructureKHR(m_device, &asCI, nullptr, &blasHandle.accel) != VK_SUCCESS ||
        blasHandle.accel == VK_NULL_HANDLE) {
        destroyBuffer(blasHandle.buffer);
        destroyBuffer(aabbBuffer);
        return UINT32_MAX;
    }

    uint64_t scratchAlignment = m_capabilities.minScratchAlignment > 0 ? m_capabilities.minScratchAlignment : 128;
    uint64_t alignedScratchSize = (sizeInfo.buildScratchSize + scratchAlignment - 1) & ~(scratchAlignment - 1);
    BufferCreateInfo scratchCI;
    scratchCI.size     = alignedScratchSize;
    scratchCI.usage    = BufferUsage::STORAGE;
    scratchCI.location = MemoryLocation::GPU_ONLY;
    auto scratchBuffer = createBuffer(scratchCI);
    if (!scratchBuffer.buffer) {
        if (fpDestroyAccelerationStructureKHR) fpDestroyAccelerationStructureKHR(m_device, blasHandle.accel, nullptr);
        destroyBuffer(blasHandle.buffer);
        destroyBuffer(aabbBuffer);
        return UINT32_MAX;
    }

    buildInfo.dstAccelerationStructure  = blasHandle.accel;
    buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        destroyBuffer(scratchBuffer);
        if (fpDestroyAccelerationStructureKHR) fpDestroyAccelerationStructureKHR(m_device, blasHandle.accel, nullptr);
        destroyBuffer(blasHandle.buffer);
        destroyBuffer(aabbBuffer);
        return UINT32_MAX;
    }
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);
    endSingleTimeCommands(cmd);

    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blasHandle.accel;
    blasHandle.deviceAddress = fpGetAccelerationStructureDeviceAddressKHR(m_device, &addrInfo);

    destroyBuffer(scratchBuffer);
    blasHandle.vertexBuffer = aabbBuffer; // keep AABB buffer alive in BLAS

    uint32_t idx = (uint32_t)m_blasList.size();
    m_blasList.push_back(blasHandle);

    VK_INFO() << "[VulkanDevice] Hair AABB BLAS created (index=" << idx
              << ", segments=" << primitiveCount
              << ", size=" << (sizeInfo.accelerationStructureSize / 1024) << " KB)" << std::endl;
    return idx;
}

// ════════════════════════════════════════════════════════════════════════════════
// Foam point-sphere AABB BLAS (binding-18 buffer holds centre/radius/matId)
// ════════════════════════════════════════════════════════════════════════════════
// One combined AABB BLAS for the whole foam cloud, with a HOST-visible AABB buffer
// so updateFoamSphereBLAS() can MODE_BUILD it in place every frame. The AS + buffer
// are sized for the pool capacity but only the COMPACT live spheres are built each
// frame (dead pool slots are dropped, not padded with degenerate AABBs — that
// bloated the BVH and made the build slower than OptiX). This is the N→1-instance
// Vulkan analogue of the OptiX compact sphere GAS.
uint32_t VulkanDevice::createFoamSphereBLAS(const std::vector<VkAabbPositionsKHR>& aabbs, uint32_t poolCapacity) {
    if (!hasHardwareRT() || !fpCreateAccelerationStructureKHR) return UINT32_MAX;
    if (aabbs.empty()) return UINT32_MAX;

    // Size the AABB buffer + AS for the POOL capacity (>= current live count) so the
    // per-frame rebuild can grow the live count up to it without reallocating. Only
    // the live `aabbs` are uploaded + built now.
    const uint32_t capPrims = (std::max)(poolCapacity, (uint32_t)aabbs.size());
    const uint64_t aabbBufBytes = (uint64_t)capPrims * sizeof(VkAabbPositionsKHR);
    BufferCreateInfo aabbBufInfo;
    aabbBufInfo.size = aabbBufBytes;
    aabbBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    aabbBufInfo.location = MemoryLocation::CPU_TO_GPU;   // host-writable for per-frame rebuild
    aabbBufInfo.initialData = nullptr;
    auto aabbBuffer = createBuffer(aabbBufInfo);
    if (!aabbBuffer.buffer) return UINT32_MAX;
    uploadBuffer(aabbBuffer, aabbs.data(), aabbs.size() * sizeof(VkAabbPositionsKHR)); // live spheres

    VkAccelerationStructureGeometryAabbsDataKHR aabbsData{};
    aabbsData.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbsData.data.deviceAddress = aabbBuffer.deviceAddress;
    aabbsData.stride = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags        = 0; // intersection shader decides
    geometry.geometry.aabbs = aabbsData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                            | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    // Size the AS + scratch for the POOL capacity (worst case); build only the live
    // primitives below. A later per-frame rebuild with up to capPrims spheres fits.
    uint32_t sizePrims = capPrims;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fpGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &sizePrims, &sizeInfo);

    AccelStructHandle blasHandle{};
    BufferCreateInfo asBufInfo;
    asBufInfo.size     = sizeInfo.accelerationStructureSize;
    asBufInfo.usage    = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    asBufInfo.location = MemoryLocation::GPU_ONLY;
    blasHandle.buffer  = createBuffer(asBufInfo);
    if (!blasHandle.buffer.buffer) { destroyBuffer(aabbBuffer); return UINT32_MAX; }

    VkAccelerationStructureCreateInfoKHR asCI{};
    asCI.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCI.buffer = blasHandle.buffer.buffer;
    asCI.size   = sizeInfo.accelerationStructureSize;
    asCI.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (fpCreateAccelerationStructureKHR(m_device, &asCI, nullptr, &blasHandle.accel) != VK_SUCCESS ||
        blasHandle.accel == VK_NULL_HANDLE) {
        destroyBuffer(blasHandle.buffer); destroyBuffer(aabbBuffer); return UINT32_MAX;
    }

    uint64_t scratchAlignment = m_capabilities.minScratchAlignment > 0 ? m_capabilities.minScratchAlignment : 128;
    uint64_t alignedScratchSize = (sizeInfo.buildScratchSize + scratchAlignment - 1) & ~(scratchAlignment - 1);
    BufferCreateInfo scratchCI;
    scratchCI.size = alignedScratchSize;
    scratchCI.usage = BufferUsage::STORAGE;
    scratchCI.location = MemoryLocation::GPU_ONLY;
    auto scratchBuffer = createBuffer(scratchCI);
    if (!scratchBuffer.buffer) {
        if (fpDestroyAccelerationStructureKHR) fpDestroyAccelerationStructureKHR(m_device, blasHandle.accel, nullptr);
        destroyBuffer(blasHandle.buffer); destroyBuffer(aabbBuffer); return UINT32_MAX;
    }

    buildInfo.dstAccelerationStructure  = blasHandle.accel;
    buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;
    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = (uint32_t)aabbs.size();   // build only the LIVE spheres
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        destroyBuffer(scratchBuffer);
        if (fpDestroyAccelerationStructureKHR) fpDestroyAccelerationStructureKHR(m_device, blasHandle.accel, nullptr);
        destroyBuffer(blasHandle.buffer); destroyBuffer(aabbBuffer); return UINT32_MAX;
    }
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);
    endSingleTimeCommands(cmd);

    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blasHandle.accel;
    blasHandle.deviceAddress = fpGetAccelerationStructureDeviceAddressKHR(m_device, &addrInfo);

    destroyBuffer(scratchBuffer);
    blasHandle.vertexBuffer = aabbBuffer;   // keep AABB buffer alive (host-writable)
    blasHandle.allowUpdate  = true;

    uint32_t idx = (uint32_t)m_blasList.size();
    m_blasList.push_back(blasHandle);
    return idx;
}

// In-place foam BLAS REBUILD (MODE_BUILD on the existing AS handle/buffer — NOT a
// MODE_UPDATE refit). Foam is highly mobile: a refit keeps the frame-0 BVH topology
// so the tree degenerates as particles disperse and path tracing chokes (the exact
// "degenerate BVH" symptom seen on OptiX, which we likewise fixed by per-frame
// rebuild). Rebuilding into the SAME accel + backing buffer gives optimal BVH every
// frame with NO destroy/recreate (lifecycle-safe). `aabbs` is the COMPACT live set
// (variable count, ≤ the pool capacity the AS + AABB buffer were sized for in
// createFoamSphereBLAS), so it always fits; the caller forces a full rebuild only
// when the POOL itself grows past that capacity.
bool VulkanDevice::updateFoamSphereBLAS(uint32_t blasIndex, const std::vector<VkAabbPositionsKHR>& aabbs) {
    if (blasIndex >= m_blasList.size() || aabbs.empty()) return false;
    AccelStructHandle& blas = m_blasList[blasIndex];
    if (!blas.allowUpdate || blas.accel == VK_NULL_HANDLE || !blas.vertexBuffer.buffer) return false;

    // Re-upload the AABBs to the existing host-visible buffer.
    uploadBuffer(blas.vertexBuffer, aabbs.data(), aabbs.size() * sizeof(VkAabbPositionsKHR));

    VkAccelerationStructureGeometryAabbsDataKHR aabbsData{};
    aabbsData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbsData.data.deviceAddress = blas.vertexBuffer.deviceAddress;
    aabbsData.stride = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags        = 0;
    geometry.geometry.aabbs = aabbsData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    // Keep ALLOW_UPDATE so the build size matches the pre-sized AS from createFoamSphereBLAS.
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                            | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR; // full rebuild, optimal BVH
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;                      // build (not update)
    buildInfo.dstAccelerationStructure = blas.accel;                         // same handle/buffer
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &geometry;

    uint32_t primitiveCount = (uint32_t)aabbs.size();
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fpGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

    uint64_t scratchAlignment = m_capabilities.minScratchAlignment > 0 ? m_capabilities.minScratchAlignment : 128;
    uint64_t bldScratch = (sizeInfo.buildScratchSize + scratchAlignment - 1) & ~(scratchAlignment - 1);
    BufferCreateInfo scratchCI;
    scratchCI.size = bldScratch;
    scratchCI.usage = BufferUsage::STORAGE;
    scratchCI.location = MemoryLocation::GPU_ONLY;
    auto scratchBuffer = createBuffer(scratchCI);
    if (!scratchBuffer.buffer) return false;
    buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) { destroyBuffer(scratchBuffer); return false; }
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);
    endSingleTimeCommands(cmd);
    destroyBuffer(scratchBuffer);
    return true;
}

// Upload the combined foam sphere buffer (centre/radius/matId per sphere) to
// binding 18 and (re)point the descriptor at it. `data` is FoamSphereGPU[count]
// (32 bytes each, see sphere_intersection.rint).
void VulkanDevice::updateFoamSphereBuffer(const void* data, uint32_t count) {
    if (count == 0) return;
    const uint64_t bytes = (uint64_t)count * 32ull;   // sizeof(FoamSphereGPU)
    bool recreated = false;
    if (!m_foamSphereBuffer.buffer || m_foamSphereBuffer.size < bytes) {
        if (m_foamSphereBuffer.buffer) destroyBuffer(m_foamSphereBuffer);
        BufferCreateInfo ci;
        ci.size = bytes;
        ci.usage = BufferUsage::STORAGE;
        ci.location = MemoryLocation::CPU_TO_GPU;
        ci.initialData = const_cast<void*>(data);
        m_foamSphereBuffer = createBuffer(ci);
        recreated = true;
    } else {
        uploadBuffer(m_foamSphereBuffer, data, bytes);
    }
    m_foamSphereCount = count;

    // Re-point binding 18 at the (possibly reallocated) buffer.
    if (recreated && m_rtDescriptorSet != VK_NULL_HANDLE && m_foamSphereBuffer.buffer) {
        VkDescriptorBufferInfo info{};
        info.buffer = m_foamSphereBuffer.buffer;
        info.offset = 0;
        info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = m_rtDescriptorSet;
        w.dstBinding = 18;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo = &info;
        vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Hair segment / material SSBO upload (bindings 10 and 11)
// ════════════════════════════════════════════════════════════════════════════════
void VulkanDevice::updateHairSegmentBuffer(const std::vector<VulkanRT::HairSegmentGPU>& segments) {
    if (segments.empty()) return;
    const uint64_t dataSize = segments.size() * sizeof(VulkanRT::HairSegmentGPU);

    if (m_hairSegmentBuffer.size < dataSize) {
        if (m_hairSegmentBuffer.buffer) destroyBuffer(m_hairSegmentBuffer);
        BufferCreateInfo ci;
        ci.size     = dataSize;
        ci.usage    = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_hairSegmentBuffer = createBuffer(ci);
    }
    uploadBuffer(m_hairSegmentBuffer, segments.data(), dataSize);

    // Live-update descriptor binding 10 if set exists
    if (m_rtDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo info{};
        info.buffer = m_hairSegmentBuffer.buffer;
        info.offset = 0;
        info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet         = m_rtDescriptorSet;
        w.dstBinding     = 10;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo    = &info;
        vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
    }
    VK_INFO() << "[VulkanDevice] updateHairSegmentBuffer - " << segments.size() << " segments (" << dataSize << " bytes)" << std::endl;
}

void VulkanDevice::updateHairMaterialBuffer(const std::vector<VulkanRT::HairGpuMaterial>& materials) {
    if (materials.empty()) return;
    const uint64_t dataSize = materials.size() * sizeof(VulkanRT::HairGpuMaterial);

    if (m_hairMaterialBuffer.size < dataSize) {
        if (m_hairMaterialBuffer.buffer) destroyBuffer(m_hairMaterialBuffer);
        BufferCreateInfo ci;
        ci.size     = dataSize;
        ci.usage    = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_hairMaterialBuffer = createBuffer(ci);
    }
    uploadBuffer(m_hairMaterialBuffer, materials.data(), dataSize);

    // Live-update descriptor binding 11 if set exists
    if (m_rtDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo info{};
        info.buffer = m_hairMaterialBuffer.buffer;
        info.offset = 0;
        info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet         = m_rtDescriptorSet;
        w.dstBinding     = 11;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo    = &info;
        vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
    }
    VK_INFO() << "[VulkanDevice] updateHairMaterialBuffer - " << materials.size() << " materials (" << dataSize << " bytes)" << std::endl;
}

void VulkanDevice::updateAtmosphereLUTs(const ImageHandle* lutImages) {
    if (!lutImages) return;
    
    // Store the LUT image handles
    for (int i = 0; i < 4; i++) {
        m_lutImages[i] = lutImages[i];
    }
    // Stored LUT handles updated
    
    // If RT descriptor set already exists, update binding 8 with LUT samplers
    if (m_rtDescriptorSet != VK_NULL_HANDLE) {
        // Check if at least one LUT is valid
        bool hasValidLUT = false;
        for (int i = 0; i < 4; i++) {
            if (m_lutImages[i].view != VK_NULL_HANDLE) {
                hasValidLUT = true;
                break;
            }
        }
        
        if (hasValidLUT) {
            // Use stack allocation (fixed-size array) to ensure lifetime safety
            VkDescriptorImageInfo lutImageInfos[4] = {};
            for (int i = 0; i < 4; i++) {
                if (m_lutImages[i].view != VK_NULL_HANDLE) {
                    lutImageInfos[i].sampler = m_lutImages[i].sampler;
                    lutImageInfos[i].imageView = m_lutImages[i].view;
                    lutImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                } else {
                    lutImageInfos[i].sampler = VK_NULL_HANDLE;
                    lutImageInfos[i].imageView = VK_NULL_HANDLE;
                    lutImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                }
            }
            
            uint32_t validCount = 0;
            for (int i = 0; i < 4; ++i) if (lutImageInfos[i].imageView != VK_NULL_HANDLE) ++validCount;
            // validCount LUT(s) will be updated
            if (validCount > 0) {
                VkWriteDescriptorSet w8{};
                w8.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w8.dstSet = m_rtDescriptorSet;
                w8.dstBinding = 8;
                w8.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w8.descriptorCount = validCount;
                w8.pImageInfo = lutImageInfos;
                vkUpdateDescriptorSets(m_device, 1, &w8, 0, nullptr);
            }
        }
    }
}
void VulkanDevice::clearImage(const ImageHandle& image, float r, float g, float b, float a) {
    if (!image.image) return;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;

    // Transition to TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL; 
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.image = image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkClearColorValue clearColor;
    clearColor.float32[0] = r;
    clearColor.float32[1] = g;
    clearColor.float32[2] = b;
    clearColor.float32[3] = a;

    vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &barrier.subresourceRange);

    // Transition back to GENERAL for shader use
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(cmd);
}

void VulkanDevice::clearImages(const std::vector<ImageClearRequest>& requests) {
    if (requests.empty()) return;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;

    for (const auto& req : requests) {
        if (!req.image || !req.image->image) continue;

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.image = req.image->image;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkClearColorValue clearColor;
        clearColor.float32[0] = req.r;
        clearColor.float32[1] = req.g;
        clearColor.float32[2] = req.b;
        clearColor.float32[3] = req.a;
        vkCmdClearColorImage(cmd, req.image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clearColor, 1, &barrier.subresourceRange);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    endSingleTimeCommands(cmd);
}

// ========================================================================
// Compute Pipeline - Real Implementation
// ========================================================================

uint32_t VulkanDevice::createPipeline(const PipelineCreateInfo& info) {
    if (info.shaders.empty()) return UINT32_MAX;

    // --- 1) Create shader module from SPIR-V ---
    const auto& shaderInfo = info.shaders[0]; // Compute = single shader
    VkShaderModuleCreateInfo moduleCreateInfo{};
    moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleCreateInfo.codeSize = shaderInfo.spirvCode.size() * sizeof(uint32_t);
    moduleCreateInfo.pCode = shaderInfo.spirvCode.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_device, &moduleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] Failed to create shader module" << std::endl;
        return UINT32_MAX;
    }

    // --- 2) Descriptor set layout: binding 0 = storage image ---
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    VkDescriptorSetLayout descriptorSetLayout;
    vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &descriptorSetLayout);
    m_descriptorSetLayouts.push_back(descriptorSetLayout);

    // --- 3) Push constant range ---
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = info.pushConstantSize > 0 ? info.pushConstantSize : 12; // default: width, height, time

    // --- 4) Pipeline layout ---
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
    m_pipelineLayouts.push_back(pipelineLayout);

    // --- 5) Compute pipeline ---
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = shaderInfo.entryPoint.c_str();

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] Failed to create compute pipeline" << std::endl;
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
        return UINT32_MAX;
    }

    m_pipelines.push_back(pipeline);
    vkDestroyShaderModule(m_device, shaderModule, nullptr); // Safe to destroy after pipeline creation

    uint32_t index = (uint32_t)(m_pipelines.size() - 1);
    VK_INFO() << "[VulkanDevice] Compute pipeline created (index=" << index << ")" << std::endl;
    return index;
}

void VulkanDevice::bindPipeline(uint32_t pipelineIndex) {
    m_activePipeline = pipelineIndex;
}

void VulkanDevice::dispatchCompute(uint32_t gx, uint32_t gy, uint32_t gz) {
    if (m_activePipeline >= m_pipelines.size()) {
        VK_ERROR() << "[VulkanDevice] No active pipeline for dispatch!" << std::endl;
        return;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[m_activePipeline]);

    // Bind descriptor set if available
    if (!m_activeDescriptorSets.empty()) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayouts[m_activePipeline], 0,
            (uint32_t)m_activeDescriptorSets.size(), m_activeDescriptorSets.data(),
            0, nullptr);
    }

    // Push constants
    if (m_pushConstantData.size() > 0) {
        vkCmdPushConstants(cmd, m_pipelineLayouts[m_activePipeline],
            VK_SHADER_STAGE_COMPUTE_BIT, 0,
            (uint32_t)m_pushConstantData.size(), m_pushConstantData.data());
    }

    vkCmdDispatch(cmd, gx, gy, gz);

    endSingleTimeCommands(cmd);
}

void VulkanDevice::setPushConstants(const void* data, uint32_t size) {
    m_pushConstantData.resize(size);
    memcpy(m_pushConstantData.data(), data, size);
}

// ========================================================================
// Image Operations - Real Implementation
// ========================================================================

ImageHandle VulkanDevice::createImage2D(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspectMask) {
    ImageHandle handle{};
    if (width == 0 || height == 0) {
        VK_ERROR() << "[VulkanDevice] createImage2D called with invalid extent: "
                   << width << "x" << height << std::endl;
        return {};
    }
    handle.width = width;
    handle.height = height;
    handle.format = format;

    // --- 1) Create VkImage ---
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = { width, height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // Always allow readback
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(m_device, &imageInfo, nullptr, &handle.image) != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] Failed to create image" << std::endl;
        return {};
    }

    // --- 2) Allocate memory ---
    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(m_device, handle.image, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (allocInfo.memoryTypeIndex == UINT32_MAX) {
        VK_ERROR() << "[VulkanDevice] Failed to find device-local memory type for image." << std::endl;
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }

    VkResult allocRes = vkAllocateMemory(m_device, &allocInfo, nullptr, &handle.memory);
    if (allocRes != VK_SUCCESS || !handle.memory) {
        VK_ERROR() << "[VulkanDevice] Failed to allocate image memory ("
                   << width << "x" << height << ", result=" << allocRes << ")" << std::endl;
        signalVulkanMemoryPressure(allocRes, "createImage2D/vkAllocateMemory");
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }

    VkResult bindRes = vkBindImageMemory(m_device, handle.image, handle.memory, 0);
    if (bindRes != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] Failed to bind image memory (result=" << bindRes << ")" << std::endl;
        vkFreeMemory(m_device, handle.memory, nullptr);
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }

    // --- 3) Create image view ---
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = handle.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectMask;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkResult viewRes = vkCreateImageView(m_device, &viewInfo, nullptr, &handle.view);
    if (viewRes != VK_SUCCESS || !handle.view) {
        VK_ERROR() << "[VulkanDevice] Failed to create image view (result=" << viewRes << ")" << std::endl;
        vkDestroyImage(m_device, handle.image, nullptr);
        vkFreeMemory(m_device, handle.memory, nullptr);
        return {};
    }

    // --- 4) Transition to GENERAL layout (for storage image access) ---
    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd != VK_NULL_HANDLE) {
        transitionImageLayout(cmd, handle.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        endSingleTimeCommands(cmd);
    } else {
        VK_ERROR() << "[VulkanDevice] Failed to allocate command buffer for image transition." << std::endl;
        destroyImage(handle);
        return {};
    }

   // VK_INFO() << "[VulkanDevice] Image created: " << width << "x" << height << std::endl;
    return handle;
}

ImageHandle VulkanDevice::createImage2DWithMips(uint32_t width, uint32_t height, uint32_t mipLevels,
                                                VkFormat format, VkImageUsageFlags usage) {
    ImageHandle handle{};
    if (width == 0 || height == 0 || mipLevels == 0) return {};
    handle.width = width;
    handle.height = height;
    handle.mipLevels = mipLevels;
    handle.format = format;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(m_device, &imageInfo, nullptr, &handle.image) != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] createImage2DWithMips: vkCreateImage failed" << std::endl;
        return {};
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(m_device, handle.image, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (allocInfo.memoryTypeIndex == UINT32_MAX) {
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }

    VkResult allocRes = vkAllocateMemory(m_device, &allocInfo, nullptr, &handle.memory);
    if (allocRes != VK_SUCCESS || !handle.memory) {
        signalVulkanMemoryPressure(allocRes, "createImage2DWithMips/vkAllocateMemory");
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }

    if (vkBindImageMemory(m_device, handle.image, handle.memory, 0) != VK_SUCCESS) {
        vkFreeMemory(m_device, handle.memory, nullptr);
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }

    // Image view covers all mip levels so the sampler can access the full chain.
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = handle.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_device, &viewInfo, nullptr, &handle.view) != VK_SUCCESS) {
        vkFreeMemory(m_device, handle.memory, nullptr);
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }

    // Pre-transition all mip levels to TRANSFER_DST_OPTIMAL so the caller can copy into mip 0.
    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        destroyImage(handle);
        return {};
    }
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = handle.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(cmd);

    return handle;
}

void VulkanDevice::generateMipmaps(VkCommandBuffer cmd, VkImage image,
                                    uint32_t width, uint32_t height, uint32_t mipLevels) {
    if (mipLevels <= 1 || !image || cmd == VK_NULL_HANDLE) {
        // Just finalize mip 0 if it was in TRANSFER_DST_OPTIMAL
        if (mipLevels >= 1 && image && cmd != VK_NULL_HANDLE) {
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.image = image;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                0, 0, nullptr, 0, nullptr, 1, &b);
        }
        return;
    }

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipW = (int32_t)width;
    int32_t mipH = (int32_t)height;

    for (uint32_t i = 1; i < mipLevels; ++i) {
        // Transition mip i-1: TRANSFER_DST -> TRANSFER_SRC (ready as blit source)
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        const int32_t nextW = std::max(1, mipW / 2);
        const int32_t nextH = std::max(1, mipH / 2);

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipW, mipH, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {nextW, nextH, 1};
        vkCmdBlitImage(cmd,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        // Transition mip i-1 to final SHADER_READ_ONLY_OPTIMAL
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        mipW = nextW;
        mipH = nextH;
    }

    // Finalize the last mip level
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VulkanDevice::destroyImage(ImageHandle& image) {
    if (image.sampler) vkDestroySampler(m_device, image.sampler, nullptr);
    if (image.view) vkDestroyImageView(m_device, image.view, nullptr);
    if (image.image) vkDestroyImage(m_device, image.image, nullptr);
    if (image.memory) vkFreeMemory(m_device, image.memory, nullptr);
    image = {};
}

void VulkanDevice::transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                                          VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage, dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcAccessMask = 0;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
        0, nullptr, 0, nullptr, 1, &barrier);
}

bool VulkanDevice::traceRaysAndReadback(uint32_t w, uint32_t h,
    const ImageHandle& outputImage, const BufferHandle& stagingBuffer) {
    if (!m_rtPipelineReady || !fpCmdTraceRaysKHR || !m_tlas.accel) return false;
    if (!outputImage.image || !stagingBuffer.buffer) return false;
    if (outputImage.width == 0 || outputImage.height == 0) return false;

    const uint64_t bytesPerPixel = (outputImage.format == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8ull : 16ull;
    const uint64_t requiredBytes = (uint64_t)outputImage.width * (uint64_t)outputImage.height * bytesPerPixel;
    if (stagingBuffer.size < requiredBytes) {
        SCENE_LOG_WARN("[Vulkan] traceRaysAndReadback skipped: staging buffer too small for output image.");
        return false;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;

    // ── 0. Photon caustic pass (if scheduled) — same-buffer recording, see
    //       recordPhotonPass for the race rationale.
    recordPhotonPass(cmd);

    // ── 1. Bind pipeline + descriptors + push constants ───────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    if (m_rtDescriptorSet)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            m_rtPipelineLayout, 0, 1, &m_rtDescriptorSet, 0, nullptr);
    if (!m_pushConstantData.empty())
        vkCmdPushConstants(cmd, m_rtPipelineLayout,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_MISS_BIT_KHR  | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
            0, (uint32_t)m_pushConstantData.size(), m_pushConstantData.data());

    // ── 2. Trace ──────────────────────────────────────────────────────────────
    fpCmdTraceRaysKHR(cmd, &m_sbtRaygenRegion, &m_sbtMissRegion,
                      &m_sbtHitRegion, &m_sbtCallableRegion, w, h, 1);

    // ── 3. Barrier: shader write → transfer read ─────────────────────────────
    VkImageMemoryBarrier imgBarrier{};
    imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imgBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imgBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBarrier.image = outputImage.image;
    imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &imgBarrier);

    // ── 4. Copy image → staging buffer ───────────────────────────────────────
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {outputImage.width, outputImage.height, 1};
    vkCmdCopyImageToBuffer(cmd, outputImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuffer.buffer, 1, &region);

    // ── 5. Transition image back to GENERAL ──────────────────────────────────
    // No further shader access happens in this same command buffer, so keep this
    // transition conservative (destination access = 0, destination stage = BOTTOM).
    // This is more robust across drivers under memory pressure/device stress.
    imgBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    imgBarrier.dstAccessMask = 0;
    imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &imgBarrier);

    endSingleTimeCommands(cmd); // single vkQueueWaitIdle
    return true;
}

// Fused trace + GPU tonemap + readback. Replaces traceRaysAndReadback when the
// tonemap compute pipeline is available. Reduces per-frame CPU work from a full
// scalar tonemap pass (Reinhard + sRGB encode at full resolution, single-thread)
// to a plain memcpy. Also cuts staging readback bytes by 4x for FP32 (and 2x for
// FP16) since the LDR image is RGBA8.
bool VulkanDevice::traceRaysTonemapAndReadback(uint32_t w, uint32_t h,
    const ImageHandle& hdrImage, const ImageHandle& ldrImage, const BufferHandle& ldrStaging) {
    if (!m_rtPipelineReady || !fpCmdTraceRaysKHR || !m_tlas.accel) return false;
    if (m_tonemapPipeline == VK_NULL_HANDLE || m_tonemapPipelineLayout == VK_NULL_HANDLE) return false;
    if (!hdrImage.image || !ldrImage.image || !ldrStaging.buffer) return false;
    if (hdrImage.width == 0 || hdrImage.height == 0) return false;

    const uint64_t requiredBytes = (uint64_t)ldrImage.width * (uint64_t)ldrImage.height * 4ull;
    if (ldrStaging.size < requiredBytes) {
        SCENE_LOG_WARN("[Vulkan] traceRaysTonemapAndReadback skipped: LDR staging too small.");
        return false;
    }

    // Allocate a fresh tonemap descriptor set (same pattern as dispatchSculpt — pool sized
    // for many sets, fence-driven recycling will be added in stage 2).
    VkDescriptorSet tmDescSet = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_tonemapDescPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_tonemapDescLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &tmDescSet) != VK_SUCCESS || tmDescSet == VK_NULL_HANDLE) {
            // Pool exhausted — reset it and retry once. Per-frame allocations make this
            // a normal recovery path, not an error.
            vkResetDescriptorPool(m_device, m_tonemapDescPool, 0);
            if (vkAllocateDescriptorSets(m_device, &allocInfo, &tmDescSet) != VK_SUCCESS || tmDescSet == VK_NULL_HANDLE) {
                SCENE_LOG_WARN("[Vulkan] tonemap descriptor allocation failed; falling back to legacy readback.");
                return false;
            }
        }
    }

    VkDescriptorImageInfo inInfo{};
    inInfo.imageView = hdrImage.view;
    inInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo outInfo{};
    outInfo.imageView = ldrImage.view;
    outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = tmDescSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &inInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = tmDescSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &outInfo;
    vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;

    // ── 1. Trace rays into HDR image ─────────────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    if (m_rtDescriptorSet)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            m_rtPipelineLayout, 0, 1, &m_rtDescriptorSet, 0, nullptr);
    if (!m_pushConstantData.empty())
        vkCmdPushConstants(cmd, m_rtPipelineLayout,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_MISS_BIT_KHR  | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
            0, (uint32_t)m_pushConstantData.size(), m_pushConstantData.data());
    fpCmdTraceRaysKHR(cmd, &m_sbtRaygenRegion, &m_sbtMissRegion,
                      &m_sbtHitRegion, &m_sbtCallableRegion, w, h, 1);

    // ── 2. Barrier: RT shader write → compute shader read on HDR image ───────
    VkImageMemoryBarrier hdrBarrier{};
    hdrBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    hdrBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hdrBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    hdrBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    hdrBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    hdrBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hdrBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hdrBarrier.image = hdrImage.image;
    hdrBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &hdrBarrier);

    // ── 3. Dispatch tonemap compute ──────────────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_tonemapPipelineLayout, 0, 1, &tmDescSet, 0, nullptr);
    struct TonemapPush { uint32_t width; uint32_t height; } tmPush{ w, h };
    vkCmdPushConstants(cmd, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(TonemapPush), &tmPush);
    const uint32_t gx = (w + 7) / 8;
    const uint32_t gy = (h + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    // ── 4. Barrier: compute write → transfer read on LDR image ───────────────
    VkImageMemoryBarrier ldrBarrier{};
    ldrBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ldrBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ldrBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ldrBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    ldrBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ldrBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ldrBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ldrBarrier.image = ldrImage.image;
    ldrBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &ldrBarrier);

    // ── 5. Copy LDR image → staging buffer (1/4 the bytes vs FP32 HDR) ───────
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {ldrImage.width, ldrImage.height, 1};
    vkCmdCopyImageToBuffer(cmd, ldrImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           ldrStaging.buffer, 1, &region);

    // ── 6. Transition LDR back to GENERAL for next frame's compute write ─────
    ldrBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ldrBarrier.dstAccessMask = 0;
    ldrBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ldrBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &ldrBarrier);

    endSingleTimeCommands(cmd); // single vkQueueWaitIdle; stage 2 will replace with fences
    return true;
}

void VulkanDevice::copyImageToBuffer(const ImageHandle& src, const BufferHandle& dst) {
    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;

    // Transition image to TRANSFER_SRC
    transitionImageLayout(cmd, src.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Copy
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;    // Tightly packed
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {src.width, src.height, 1};

    vkCmdCopyImageToBuffer(cmd, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.buffer, 1, &region);

    // Transition back to GENERAL
    transitionImageLayout(cmd, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

    endSingleTimeCommands(cmd);
}

void VulkanDevice::copyImagesToBuffersBatched(const ImageHandle* srcs, const BufferHandle* dsts, size_t count) {
    if (!srcs || !dsts || count == 0) return;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;

    // First barrier batch: GENERAL → TRANSFER_SRC_OPTIMAL on all sources at once.
    std::vector<VkImageMemoryBarrier> toSrc(count);
    for (size_t i = 0; i < count; ++i) {
        VkImageMemoryBarrier& b = toSrc[i];
        b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = srcs[i].image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr,
        static_cast<uint32_t>(count), toSrc.data());

    for (size_t i = 0; i < count; ++i) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {srcs[i].width, srcs[i].height, 1};
        vkCmdCopyImageToBuffer(cmd, srcs[i].image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               dsts[i].buffer, 1, &region);
    }

    // Second barrier batch: TRANSFER_SRC_OPTIMAL → GENERAL.
    std::vector<VkImageMemoryBarrier> toGen(count);
    for (size_t i = 0; i < count; ++i) {
        VkImageMemoryBarrier& b = toGen[i];
        b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = srcs[i].image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr,
        static_cast<uint32_t>(count), toGen.data());

    endSingleTimeCommands(cmd);
}

void VulkanDevice::copyBufferToImage(const BufferHandle& src, const ImageHandle& dst) {
    if (!src.buffer || !dst.image || dst.width == 0 || dst.height == 0) {
        VK_ERROR() << "[VulkanDevice] copyBufferToImage skipped: invalid src/dst handle." << std::endl;
        return;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        VK_ERROR() << "[VulkanDevice] copyBufferToImage skipped: failed to begin command buffer." << std::endl;
        return;
    }

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {dst.width, dst.height, 1};

    vkCmdCopyBufferToImage(cmd, src.buffer, dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);

    endSingleTimeCommands(cmd);
}

void VulkanDevice::copyBufferToImageRegion(const BufferHandle& src, const ImageHandle& dst,
                                            int32_t offsetX, int32_t offsetY,
                                            uint32_t regionW, uint32_t regionH) {
    if (!src.buffer || !dst.image || dst.width == 0 || dst.height == 0 || regionW == 0 || regionH == 0) {
        VK_ERROR() << "[VulkanDevice] copyBufferToImageRegion skipped: invalid src/dst handle." << std::endl;
        return;
    }
    if (offsetX < 0 || offsetY < 0 ||
        static_cast<uint32_t>(offsetX) + regionW > dst.width ||
        static_cast<uint32_t>(offsetY) + regionH > dst.height) {
        VK_ERROR() << "[VulkanDevice] copyBufferToImageRegion skipped: region out of bounds." << std::endl;
        return;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        VK_ERROR() << "[VulkanDevice] copyBufferToImageRegion skipped: failed to begin command buffer." << std::endl;
        return;
    }

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { offsetX, offsetY, 0 };
    region.imageExtent = { regionW, regionH, 1 };
    vkCmdCopyBufferToImage(cmd, src.buffer, dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);

    endSingleTimeCommands(cmd);
}

void VulkanDevice::recordCopyBufferToImage(VkCommandBuffer cmd, const BufferHandle& src, const ImageHandle& dst) {
    if (cmd == VK_NULL_HANDLE || !src.buffer || !dst.image || dst.width == 0 || dst.height == 0) return;
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {dst.width, dst.height, 1};
    vkCmdCopyBufferToImage(cmd, src.buffer, dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

void VulkanDevice::recordCopyBufferToImageDst(VkCommandBuffer cmd, const BufferHandle& src, const ImageHandle& dst) {
    if (cmd == VK_NULL_HANDLE || !src.buffer || !dst.image || dst.width == 0 || dst.height == 0) return;
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {dst.width, dst.height, 1};
    vkCmdCopyBufferToImage(cmd, src.buffer, dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

// ========================================================================
// Descriptor Set Helper
// ========================================================================

void VulkanDevice::bindStorageImage(uint32_t pipelineIndex, uint32_t bindingIndex, const ImageHandle& image) {
    if (pipelineIndex >= m_descriptorSetLayouts.size()) return;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayouts[pipelineIndex];

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &descriptorSet) != VK_SUCCESS ||
        descriptorSet == VK_NULL_HANDLE) {
        return;
    }

    // Write descriptor
    VkDescriptorImageInfo imageDescInfo{};
    imageDescInfo.imageView = image.view;
    imageDescInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = bindingIndex;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageDescInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    m_activeDescriptorSets = { descriptorSet };
}

void VulkanDevice::waitIdle() { if (m_device) vkDeviceWaitIdle(m_device); }
void VulkanDevice::submitAndWait() { if (m_computeQueue) vkQueueWaitIdle(m_computeQueue); }

// Factory
std::unique_ptr<VulkanDevice> createVulkanDevice(bool preferHardwareRT, bool validation) {
    auto device = std::make_unique<VulkanDevice>();
    if (device->initialize(preferHardwareRT, validation)) return device;
    return nullptr;
}



// ============================================================================
// Backend::VulkanBackendAdapter Implementation
// ============================================================================



bool VulkanDevice::createSkinningPipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    // Recreate-safe: free previous skinning resources first.
    // Existing per-BLAS descriptor sets are allocated from the old pool/layout.
    // Invalidate all cached handles so dispatchSkinning reallocates safely.
    for (auto& blas : m_blasList) {
        blas.skinningDescSet = VK_NULL_HANDLE;
    }
    if (m_skinningPipeline) { vkDestroyPipeline(m_device, m_skinningPipeline, nullptr); m_skinningPipeline = VK_NULL_HANDLE; }
    if (m_skinningPipelineLayout) { vkDestroyPipelineLayout(m_device, m_skinningPipelineLayout, nullptr); m_skinningPipelineLayout = VK_NULL_HANDLE; }
    if (m_skinningDescLayout) { vkDestroyDescriptorSetLayout(m_device, m_skinningDescLayout, nullptr); m_skinningDescLayout = VK_NULL_HANDLE; }
    if (m_skinningDescPool) { vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr); m_skinningDescPool = VK_NULL_HANDLE; }
    
    // Create descriptor pool — one persistent set per skinned BLAS, no upper bound known at
    // pipeline-creation time so we use a generous cap.  Pool is never reset; sets are
    // allocated once per BLAS and reused every frame (FREE_DESCRIPTOR_SET_BIT not needed).
    // 64 skinned meshes × 7 bindings = 448 descriptors max.
    const uint32_t kMaxSkinnedMeshes = 64;
    VkDescriptorPoolSize poolSizes[] = { {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSkinnedMeshes * 7} };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = kMaxSkinnedMeshes;
    poolInfo.flags = 0; // no free needed — persistent sets
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_skinningDescPool) != VK_SUCCESS) return false;

    // Create descriptor set layout (7 storage bindings)
    std::vector<VkDescriptorSetLayoutBinding> bindings(7);
    for(int i=0; i<7; ++i){
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 7;
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_skinningDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr);
        m_skinningDescPool = VK_NULL_HANDLE;
        return false;
    }

    // Create Pipeline Layout
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 8; // 2 uints: vertexCount + boneCount

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_skinningDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_skinningPipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_skinningDescLayout, nullptr);
        m_skinningDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr);
        m_skinningDescPool = VK_NULL_HANDLE;
        return false;
    }

    // Create shader module
    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_skinningPipelineLayout, nullptr);
        m_skinningPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_skinningDescLayout, nullptr);
        m_skinningDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr);
        m_skinningDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_skinningPipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";
    
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_skinningPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_skinningPipelineLayout, nullptr);
        m_skinningPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_skinningDescLayout, nullptr);
        m_skinningDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr);
        m_skinningDescPool = VK_NULL_HANDLE;
        return false;
    }

    vkDestroyShaderModule(m_device, compModule, nullptr);
    return true;
}

bool VulkanDevice::createSculptPipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    if (m_sculptPipeline) { vkDestroyPipeline(m_device, m_sculptPipeline, nullptr); m_sculptPipeline = VK_NULL_HANDLE; }
    if (m_sculptPipelineLayout) { vkDestroyPipelineLayout(m_device, m_sculptPipelineLayout, nullptr); m_sculptPipelineLayout = VK_NULL_HANDLE; }
    if (m_sculptDescLayout) { vkDestroyDescriptorSetLayout(m_device, m_sculptDescLayout, nullptr); m_sculptDescLayout = VK_NULL_HANDLE; }
    if (m_sculptDescPool) { vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr); m_sculptDescPool = VK_NULL_HANDLE; }

    // Descriptor pool: allow up to 64 sets, 3 storage buffers each
    const uint32_t kMaxSculptSets = 64;
    VkDescriptorPoolSize poolSizes[] = { {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSculptSets * 3} };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = kMaxSculptSets;
    poolInfo.flags = 0;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_sculptDescPool) != VK_SUCCESS) return false;

    // Descriptor layout: 3 storage buffers (positions, normals, weights)
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_sculptDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr);
        m_sculptDescPool = VK_NULL_HANDLE;
        return false;
    }

    // Pipeline layout with optional push constants (up to 64 bytes)
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 64;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_sculptDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_sculptPipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_sculptDescLayout, nullptr);
        m_sculptDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr);
        m_sculptDescPool = VK_NULL_HANDLE;
        return false;
    }

    // Create shader module
    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_sculptPipelineLayout, nullptr);
        m_sculptPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_sculptDescLayout, nullptr);
        m_sculptDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr);
        m_sculptDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_sculptPipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_sculptPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_sculptPipelineLayout, nullptr);
        m_sculptPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_sculptDescLayout, nullptr);
        m_sculptDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr);
        m_sculptDescPool = VK_NULL_HANDLE;
        return false;
    }

    vkDestroyShaderModule(m_device, compModule, nullptr);
    return true;
}

bool VulkanDevice::createTonemapPipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    if (m_tonemapPipeline)       { vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr); m_tonemapPipeline = VK_NULL_HANDLE; }
    if (m_tonemapPipelineLayout) { vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE; }
    if (m_tonemapDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE; }
    if (m_tonemapDescPool)       { vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE; }
    m_tonemapDescSet = VK_NULL_HANDLE;

    // Pool sized for a single persistent set. Aşama 2 binds the same set from both
    // frame slots — images don't change frame-to-frame, only on resize (at which point
    // fences are drained before updateTonemapDescriptors rewrites it).
    const uint32_t kMaxSets = 1;
    VkDescriptorPoolSize poolSizes[] = { {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxSets * 2} };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = kMaxSets;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_tonemapDescPool) != VK_SUCCESS) return false;

    // Layout: binding 0 = HDR input (storage image, read), binding 1 = LDR output (storage image, write).
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_tonemapDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr);
        m_tonemapDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 8; // uint width, uint height

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_tonemapDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_tonemapPipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_tonemapPipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_tonemapPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE;
        return false;
    }

    vkDestroyShaderModule(m_device, compModule, nullptr);

    // Pre-allocate the persistent descriptor set. Image views aren't known yet;
    // updateTonemapDescriptors() will populate them once the adapter has its targets.
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_tonemapDescPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_tonemapDescLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_tonemapDescSet) != VK_SUCCESS) {
            VK_ERROR() << "[VulkanDevice] Failed to allocate persistent tonemap descriptor set." << std::endl;
            // Tear down the pipeline objects so hasTonemapPipeline() reports false.
            vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr); m_tonemapPipeline = VK_NULL_HANDLE;
            vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE;
            vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE;
            vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE;
            return false;
        }
    }
    return true;
}

bool VulkanDevice::updateTonemapDescriptors(const VulkanRT::ImageHandle& hdrImage, const VulkanRT::ImageHandle& ldrImage) {
    if (m_tonemapDescSet == VK_NULL_HANDLE) return false;
    if (!hdrImage.view || !ldrImage.view) return false;

    VkDescriptorImageInfo inInfo{};
    inInfo.imageView = hdrImage.view;
    inInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo outInfo{};
    outInfo.imageView = ldrImage.view;
    outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_tonemapDescSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &inInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_tonemapDescSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &outInfo;
    vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);
    return true;
}

bool VulkanDevice::createStylizePipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    if (m_stylizePipeline)       { vkDestroyPipeline(m_device, m_stylizePipeline, nullptr); m_stylizePipeline = VK_NULL_HANDLE; }
    if (m_stylizePipelineLayout) { vkDestroyPipelineLayout(m_device, m_stylizePipelineLayout, nullptr); m_stylizePipelineLayout = VK_NULL_HANDLE; }
    if (m_stylizeDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE; }
    if (m_stylizeDescPool)       { vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE; }
    m_stylizeDescSet = VK_NULL_HANDLE;

    // Pool: one persistent set with 2 storage buffers (color + params) and 3 storage images (AOVs).
    const uint32_t kMaxSets = 1;
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSets * 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  kMaxSets * 3 },
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = kMaxSets;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_stylizeDescPool) != VK_SUCCESS) return false;

    // Layout: 0=color SSBO, 1=position img, 2=albedo img, 3=normal img, 4=params SSBO.
    VkDescriptorSetLayoutBinding bindings[5]{};
    auto setBinding = [](VkDescriptorSetLayoutBinding& b, uint32_t idx, VkDescriptorType type) {
        b.binding = idx; b.descriptorType = type; b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    };
    setBinding(bindings[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    setBinding(bindings[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    setBinding(bindings[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    setBinding(bindings[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    setBinding(bindings[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 5;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_stylizeDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_stylizeDescLayout;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_stylizePipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_stylizePipelineLayout, nullptr); m_stylizePipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_stylizePipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_stylizePipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_stylizePipelineLayout, nullptr); m_stylizePipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }
    vkDestroyShaderModule(m_device, compModule, nullptr);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_stylizeDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_stylizeDescLayout;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_stylizeDescSet) != VK_SUCCESS) {
        vkDestroyPipeline(m_device, m_stylizePipeline, nullptr); m_stylizePipeline = VK_NULL_HANDLE;
        vkDestroyPipelineLayout(m_device, m_stylizePipelineLayout, nullptr); m_stylizePipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool VulkanDevice::updateStylizeDescriptors(const VulkanRT::BufferHandle& colorBuf,
                                            const VulkanRT::BufferHandle& paramsBuf,
                                            VkImageView posView, VkImageView albView, VkImageView nrmView) {
    if (m_stylizeDescSet == VK_NULL_HANDLE) return false;
    if (!colorBuf.buffer || !paramsBuf.buffer || !posView || !albView || !nrmView) return false;

    VkDescriptorBufferInfo colorInfo{ colorBuf.buffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo paramsInfo{ paramsBuf.buffer, 0, VK_WHOLE_SIZE };
    VkDescriptorImageInfo posInfo{}; posInfo.imageView = posView; posInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo albInfo{}; albInfo.imageView = albView; albInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo nrmInfo{}; nrmInfo.imageView = nrmView; nrmInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w[5]{};
    for (int i = 0; i < 5; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = m_stylizeDescSet;
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &colorInfo;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  w[1].pImageInfo  = &posInfo;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  w[2].pImageInfo  = &albInfo;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  w[3].pImageInfo  = &nrmInfo;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[4].pBufferInfo = &paramsInfo;
    vkUpdateDescriptorSets(m_device, 5, w, 0, nullptr);
    return true;
}

bool VulkanDevice::dispatchStylizeCompute(uint32_t w, uint32_t h, VkImage posImg, VkImage albImg, VkImage nrmImg) {
    if (m_stylizePipeline == VK_NULL_HANDLE || m_stylizeDescSet == VK_NULL_HANDLE) return false;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;

    // Make the RT-frame AOV writes available to the compute read. Images stay in
    // GENERAL (they are storage images written by raygen). Coarse but safe.
    VkImageMemoryBarrier barriers[3]{};
    VkImage imgs[3] = { posImg, albImg, nrmImg };
    uint32_t bcount = 0;
    for (int i = 0; i < 3; ++i) {
        if (!imgs[i]) continue;
        VkImageMemoryBarrier& b = barriers[bcount++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = imgs[i];
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    if (bcount > 0) {
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, bcount, barriers);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stylizePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_stylizePipelineLayout, 0, 1, &m_stylizeDescSet, 0, nullptr);
    const uint32_t gx = (w + 7) / 8;
    const uint32_t gy = (h + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    endSingleTimeCommands(cmd);   // submits + waits
    return true;
}

bool VulkanDevice::createAtmosphereLUTPipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    if (m_atmosphereLutPipeline)       { vkDestroyPipeline(m_device, m_atmosphereLutPipeline, nullptr); m_atmosphereLutPipeline = VK_NULL_HANDLE; }
    if (m_atmosphereLutPipelineLayout) { vkDestroyPipelineLayout(m_device, m_atmosphereLutPipelineLayout, nullptr); m_atmosphereLutPipelineLayout = VK_NULL_HANDLE; }
    if (m_atmosphereLutDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE; }
    if (m_atmosphereLutDescPool)       { vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE; }
    m_atmosphereLutDescSet = VK_NULL_HANDLE;

    VkDescriptorPoolSize poolSizes[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 1;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_atmosphereLutDescPool) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding bindings[4]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 4;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_atmosphereLutDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr);
        m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 16; // phase, width, height, pad

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_atmosphereLutDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_atmosphereLutPipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_atmosphereLutPipelineLayout, nullptr); m_atmosphereLutPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_atmosphereLutPipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_atmosphereLutPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_atmosphereLutPipelineLayout, nullptr); m_atmosphereLutPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }
    vkDestroyShaderModule(m_device, compModule, nullptr);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_atmosphereLutDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_atmosphereLutDescLayout;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_atmosphereLutDescSet) != VK_SUCCESS) {
        vkDestroyPipeline(m_device, m_atmosphereLutPipeline, nullptr); m_atmosphereLutPipeline = VK_NULL_HANDLE;
        vkDestroyPipelineLayout(m_device, m_atmosphereLutPipelineLayout, nullptr); m_atmosphereLutPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool VulkanDevice::updateAtmosphereLUTComputeDescriptors(const ImageHandle* lutImages) {
    if (m_atmosphereLutDescSet == VK_NULL_HANDLE || !lutImages) return false;
    if (!lutImages[0].view || !lutImages[1].view || !lutImages[2].view || !m_atmosphereLutParamsBuffer.buffer) return false;

    VkDescriptorImageInfo imageInfos[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        imageInfos[i].imageView = lutImages[i].view;
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkDescriptorBufferInfo paramInfo{};
    paramInfo.buffer = m_atmosphereLutParamsBuffer.buffer;
    paramInfo.offset = 0;
    paramInfo.range = sizeof(AtmosphereLUTParamsGPU);

    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_atmosphereLutDescSet;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &imageInfos[i];
    }
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = m_atmosphereLutDescSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].descriptorCount = 1;
    writes[3].pBufferInfo = &paramInfo;
    vkUpdateDescriptorSets(m_device, 4, writes, 0, nullptr);
    return true;
}

bool VulkanDevice::generateAtmosphereLUTGPU(const WorldData& world) {
    if (m_atmosphereLutPipeline == VK_NULL_HANDLE || m_atmosphereLutDescSet == VK_NULL_HANDLE) return false;

    waitIdle();

    const AtmosphereLUTParamsGPU params = makeAtmosphereLUTParamsGPU(world);
    if (m_atmosphereLutParamsBuffer.size < sizeof(params)) {
        if (m_atmosphereLutParamsBuffer.buffer) destroyBuffer(m_atmosphereLutParamsBuffer);
        BufferCreateInfo ci{};
        ci.size = sizeof(params);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::CPU_TO_GPU;
        m_atmosphereLutParamsBuffer = createBuffer(ci);
    }
    if (!m_atmosphereLutParamsBuffer.buffer) return false;
    uploadBuffer(m_atmosphereLutParamsBuffer, &params, sizeof(params));

    for (int i = 0; i < 4; ++i) {
        if (m_lutImages[i].image) {
            destroyImage(m_lutImages[i]);
            m_lutImages[i] = {};
        }
    }

    constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ImageHandle lutImgs[4]{};
    lutImgs[0] = createImage2D(TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H, VK_FORMAT_R32G32B32A32_SFLOAT, usage);
    lutImgs[1] = createImage2D(SKYVIEW_LUT_W, SKYVIEW_LUT_H, VK_FORMAT_R32G32B32A32_SFLOAT, usage);
    lutImgs[2] = createImage2D(MULTI_SCATTER_LUT_RES, MULTI_SCATTER_LUT_RES, VK_FORMAT_R32G32B32A32_SFLOAT, usage);
    if (!lutImgs[0].image || !lutImgs[1].image || !lutImgs[2].image) {
        for (auto& img : lutImgs) if (img.image) destroyImage(img);
        return false;
    }

    auto createSampler = [&](ImageHandle& img, bool wrapU) {
        VkSamplerCreateInfo sInfo{};
        sInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sInfo.magFilter = VK_FILTER_LINEAR;
        sInfo.minFilter = VK_FILTER_LINEAR;
        sInfo.addressModeU = wrapU ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.anisotropyEnable = VK_FALSE;
        sInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sInfo.unnormalizedCoordinates = VK_FALSE;
        vkCreateSampler(m_device, &sInfo, nullptr, &img.sampler);
    };
    createSampler(lutImgs[0], false);
    createSampler(lutImgs[1], true);
    createSampler(lutImgs[2], false);

    if (!updateAtmosphereLUTComputeDescriptors(lutImgs)) {
        for (auto& img : lutImgs) if (img.image) destroyImage(img);
        return false;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        for (auto& img : lutImgs) if (img.image) destroyImage(img);
        return false;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atmosphereLutPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atmosphereLutPipelineLayout, 0, 1, &m_atmosphereLutDescSet, 0, nullptr);

    struct PushConstants { uint32_t phase; uint32_t width; uint32_t height; uint32_t pad; } pc{};
    auto dispatchPhase = [&](uint32_t phase, uint32_t w, uint32_t h) {
        pc.phase = phase;
        pc.width = w;
        pc.height = h;
        vkCmdPushConstants(cmd, m_atmosphereLutPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (w + 7u) / 8u, (h + 7u) / 8u, 1);
    };
    dispatchPhase(0, TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H);
    dispatchPhase(1, SKYVIEW_LUT_W, SKYVIEW_LUT_H);
    dispatchPhase(2, MULTI_SCATTER_LUT_RES, MULTI_SCATTER_LUT_RES);

    for (int i = 0; i < 3; ++i) {
        transitionImageLayout(cmd, lutImgs[i].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    endSingleTimeCommands(cmd);

    updateAtmosphereLUTs(lutImgs);
    return true;
}

bool VulkanDevice::ensureFrameSlotsCreated() {
    if (m_frameSlots[0].cmd != VK_NULL_HANDLE) return true;
    if (!m_device || !m_computeQueue) return false;

    // Dedicated command pool with RESET_COMMAND_BUFFER_BIT — we re-record the same
    // cmd buffer every frame slot reuse, and that flag is required to call
    // vkResetCommandBuffer (or to rely on vkBeginCommandBuffer's implicit reset).
    if (m_frameSlotCommandPool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = m_computeQueueFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_frameSlotCommandPool) != VK_SUCCESS) {
            VK_ERROR() << "[VulkanDevice] ensureFrameSlotsCreated: vkCreateCommandPool failed." << std::endl;
            return false;
        }
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_frameSlotCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // First wait should be a no-op; signal up front so submit-wait order is symmetric.
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < kFrameSlotCount; ++i) {
        if (vkAllocateCommandBuffers(m_device, &allocInfo, &m_frameSlots[i].cmd) != VK_SUCCESS) {
            VK_ERROR() << "[VulkanDevice] ensureFrameSlotsCreated: cmd alloc failed (slot " << i << ")." << std::endl;
            destroyFrameSlots();
            return false;
        }
        if (vkCreateFence(m_device, &fenceInfo, nullptr, &m_frameSlots[i].fence) != VK_SUCCESS) {
            VK_ERROR() << "[VulkanDevice] ensureFrameSlotsCreated: fence create failed (slot " << i << ")." << std::endl;
            destroyFrameSlots();
            return false;
        }
        m_frameSlots[i].everSubmitted = false;
    }
    return true;
}

void VulkanDevice::destroyFrameSlots() {
    for (auto& slot : m_frameSlots) {
        if (slot.fence) {
            vkDestroyFence(m_device, slot.fence, nullptr);
            slot.fence = VK_NULL_HANDLE;
        }
        if (slot.cmd && m_frameSlotCommandPool) {
            vkFreeCommandBuffers(m_device, m_frameSlotCommandPool, 1, &slot.cmd);
            slot.cmd = VK_NULL_HANDLE;
        }
        slot.everSubmitted = false;
    }
    if (m_frameSlotCommandPool) {
        vkDestroyCommandPool(m_device, m_frameSlotCommandPool, nullptr);
        m_frameSlotCommandPool = VK_NULL_HANDLE;
    }
}

bool VulkanDevice::waitFrameSlot(uint32_t slot, uint64_t timeoutNs) {
    if (slot >= kFrameSlotCount) return false;
    if (m_frameSlots[slot].fence == VK_NULL_HANDLE) return false;
    if (!m_frameSlots[slot].everSubmitted) return true; // nothing in flight
    VkResult res = vkWaitForFences(m_device, 1, &m_frameSlots[slot].fence, VK_TRUE, timeoutNs);
    return res == VK_SUCCESS;
}

bool VulkanDevice::ensureDenoiserCopySlotCreated() {
    if (m_denoiserCopySlot.cmd != VK_NULL_HANDLE) return true;
    if (!m_device || !m_computeQueue) return false;

    if (m_denoiserCopyCommandPool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = m_computeQueueFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_denoiserCopyCommandPool) != VK_SUCCESS) {
            VK_ERROR() << "[VulkanDevice] ensureDenoiserCopySlotCreated: vkCreateCommandPool failed." << std::endl;
            return false;
        }
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_denoiserCopyCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(m_device, &allocInfo, &m_denoiserCopySlot.cmd) != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] ensureDenoiserCopySlotCreated: cmd alloc failed." << std::endl;
        destroyDenoiserCopySlot();
        return false;
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // pre-signaled — first wait is a no-op
    if (vkCreateFence(m_device, &fenceInfo, nullptr, &m_denoiserCopySlot.fence) != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] ensureDenoiserCopySlotCreated: fence create failed." << std::endl;
        destroyDenoiserCopySlot();
        return false;
    }
    m_denoiserCopySlot.everSubmitted = false;
    return true;
}

void VulkanDevice::destroyDenoiserCopySlot() {
    if (m_denoiserCopySlot.fence) {
        vkDestroyFence(m_device, m_denoiserCopySlot.fence, nullptr);
        m_denoiserCopySlot.fence = VK_NULL_HANDLE;
    }
    if (m_denoiserCopySlot.cmd && m_denoiserCopyCommandPool) {
        vkFreeCommandBuffers(m_device, m_denoiserCopyCommandPool, 1, &m_denoiserCopySlot.cmd);
        m_denoiserCopySlot.cmd = VK_NULL_HANDLE;
    }
    m_denoiserCopySlot.everSubmitted = false;
    if (m_denoiserCopyCommandPool) {
        vkDestroyCommandPool(m_device, m_denoiserCopyCommandPool, nullptr);
        m_denoiserCopyCommandPool = VK_NULL_HANDLE;
    }
}

bool VulkanDevice::waitDenoiserCopy(uint64_t timeoutNs) {
    if (m_denoiserCopySlot.fence == VK_NULL_HANDLE) return false;
    if (!m_denoiserCopySlot.everSubmitted) return false; // nothing to wait on
    VkResult res = vkWaitForFences(m_device, 1, &m_denoiserCopySlot.fence, VK_TRUE, timeoutNs);
    return res == VK_SUCCESS;
}

void VulkanDevice::drainDenoiserCopy() {
    if (m_denoiserCopySlot.fence == VK_NULL_HANDLE) return;
    if (!m_denoiserCopySlot.everSubmitted) return;
    vkWaitForFences(m_device, 1, &m_denoiserCopySlot.fence, VK_TRUE, UINT64_MAX);
    // Reset everSubmitted so the next getDenoiserFrameGPU call treats the
    // pipeline as empty and re-seeds. This matters after resize/teardown of
    // the staging buffers — the old fence is signaled but the staging it
    // referred to has been destroyed, so reading without a fresh copy would
    // hit garbage.
    m_denoiserCopySlot.everSubmitted = false;
}

bool VulkanDevice::submitDenoiserCopyAsync(const ImageHandle* srcs, const BufferHandle* dsts, size_t count) {
    if (!srcs || !dsts || count == 0) return false;
    if (!ensureDenoiserCopySlotCreated()) return false;

    auto& slot = m_denoiserCopySlot;

    // If a previous submit is in flight, wait for it before reusing the cmd
    // buffer/fence. At steady state the fence is already signaled because the
    // CUDA prep + OIDN + D2H pipeline took longer than the GPU copy.
    if (slot.everSubmitted) {
        if (vkWaitForFences(m_device, 1, &slot.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return false;
    }
    vkResetFences(m_device, 1, &slot.fence);

    if (vkResetCommandBuffer(slot.cmd, 0) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(slot.cmd, &bi) != VK_SUCCESS) return false;

    VkCommandBuffer cmd = slot.cmd;

    // Barrier batch: GENERAL → TRANSFER_SRC_OPTIMAL across all sources.
    std::vector<VkImageMemoryBarrier> toSrc(count);
    for (size_t i = 0; i < count; ++i) {
        VkImageMemoryBarrier& b = toSrc[i];
        b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = srcs[i].image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr,
        static_cast<uint32_t>(count), toSrc.data());

    for (size_t i = 0; i < count; ++i) {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {srcs[i].width, srcs[i].height, 1};
        vkCmdCopyImageToBuffer(cmd, srcs[i].image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               dsts[i].buffer, 1, &region);
    }

    std::vector<VkImageMemoryBarrier> toGen(count);
    for (size_t i = 0; i < count; ++i) {
        VkImageMemoryBarrier& b = toGen[i];
        b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = srcs[i].image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr,
        static_cast<uint32_t>(count), toGen.data());

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (vkQueueSubmit(m_computeQueue, 1, &submit, slot.fence) != VK_SUCCESS) {
        SCENE_LOG_WARN("[Vulkan] submitDenoiserCopyAsync: vkQueueSubmit failed.");
        return false;
    }
    slot.everSubmitted = true;
    return true;
}

bool VulkanDevice::submitTraceTonemapAsync(uint32_t slot, uint32_t w, uint32_t h,
    const VulkanRT::ImageHandle& hdrImage, const VulkanRT::ImageHandle& ldrImage,
    const VulkanRT::BufferHandle& ldrStaging) {
    if (slot >= kFrameSlotCount) return false;
    if (!m_rtPipelineReady || !fpCmdTraceRaysKHR || !m_tlas.accel) return false;
    if (m_tonemapPipeline == VK_NULL_HANDLE || m_tonemapDescSet == VK_NULL_HANDLE) return false;
    if (!hdrImage.image || !ldrImage.image || !ldrStaging.buffer) return false;
    if (!ensureFrameSlotsCreated()) return false;

    const uint64_t requiredBytes = (uint64_t)ldrImage.width * (uint64_t)ldrImage.height * 4ull;
    if (ldrStaging.size < requiredBytes) {
        SCENE_LOG_WARN("[Vulkan] submitTraceTonemapAsync: LDR staging too small.");
        return false;
    }

    FrameSlot& fs = m_frameSlots[slot];

    // Wait for any previous use of this slot's cmd buffer + staging. At steady state
    // this fence is already signaled (we consumed the staging two frames ago).
    if (fs.everSubmitted) {
        VkResult wr = vkWaitForFences(m_device, 1, &fs.fence, VK_TRUE, UINT64_MAX);
        if (wr != VK_SUCCESS) return false;
    }
    vkResetFences(m_device, 1, &fs.fence);

    if (vkResetCommandBuffer(fs.cmd, 0) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(fs.cmd, &bi) != VK_SUCCESS) return false;

    VkCommandBuffer cmd = fs.cmd;

    // ── 0. Photon caustic pass (if scheduled) — must precede the camera trace
    //       in the SAME command buffer so the grid clear/fill cannot race the
    //       previous frame's in-flight camera reads.
    recordPhotonPass(cmd);

    // ── 1. Trace ─────────────────────────────────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    if (m_rtDescriptorSet)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            m_rtPipelineLayout, 0, 1, &m_rtDescriptorSet, 0, nullptr);
    if (!m_pushConstantData.empty())
        vkCmdPushConstants(cmd, m_rtPipelineLayout,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_MISS_BIT_KHR  | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
            0, (uint32_t)m_pushConstantData.size(), m_pushConstantData.data());
    fpCmdTraceRaysKHR(cmd, &m_sbtRaygenRegion, &m_sbtMissRegion,
                      &m_sbtHitRegion, &m_sbtCallableRegion, w, h, 1);

    // ── 2. RT write → compute read on HDR image ──────────────────────────────
    VkImageMemoryBarrier hdrBarrier{};
    hdrBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    hdrBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hdrBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    hdrBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    hdrBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    hdrBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hdrBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hdrBarrier.image = hdrImage.image;
    hdrBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &hdrBarrier);

    // ── 3. Tonemap compute ───────────────────────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_tonemapPipelineLayout, 0, 1, &m_tonemapDescSet, 0, nullptr);
    struct TonemapPush { uint32_t width; uint32_t height; } tmPush{ w, h };
    vkCmdPushConstants(cmd, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(TonemapPush), &tmPush);
    const uint32_t gx = (w + 7) / 8;
    const uint32_t gy = (h + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    // ── 4. Compute write → transfer read on LDR image ────────────────────────
    VkImageMemoryBarrier ldrBarrier{};
    ldrBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ldrBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ldrBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ldrBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    ldrBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ldrBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ldrBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ldrBarrier.image = ldrImage.image;
    ldrBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &ldrBarrier);

    // ── 5. Copy LDR → this slot's staging ────────────────────────────────────
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {ldrImage.width, ldrImage.height, 1};
    vkCmdCopyImageToBuffer(cmd, ldrImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           ldrStaging.buffer, 1, &region);

    // ── 6. Transition LDR back to GENERAL ────────────────────────────────────
    ldrBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ldrBarrier.dstAccessMask = 0;
    ldrBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ldrBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &ldrBarrier);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (vkQueueSubmit(m_computeQueue, 1, &submit, fs.fence) != VK_SUCCESS) {
        SCENE_LOG_WARN("[Vulkan] submitTraceTonemapAsync: vkQueueSubmit failed.");
        return false;
    }
    fs.everSubmitted = true;
    return true;
}

bool VulkanDevice::hasSkinningPipeline() const {
    return m_skinningPipeline != VK_NULL_HANDLE &&
           m_skinningPipelineLayout != VK_NULL_HANDLE &&
           m_skinningDescLayout != VK_NULL_HANDLE &&
           m_skinningDescPool != VK_NULL_HANDLE;
}

void VulkanDevice::dispatchSculpt(const BufferHandle& positions, const BufferHandle& normals, const BufferHandle& weights,
                                  uint32_t vertexCount, const void* pushConstants, uint32_t pushSize) {
    if (!m_device || !m_commandPool || !m_computeQueue) return;
    if (m_sculptPipeline == VK_NULL_HANDLE || m_sculptPipelineLayout == VK_NULL_HANDLE || m_sculptDescPool == VK_NULL_HANDLE || m_sculptDescLayout == VK_NULL_HANDLE) return;
    if (!positions.buffer || vertexCount == 0) return;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_sculptDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_sculptDescLayout;

    VkDescriptorSet descSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &descSet) != VK_SUCCESS || descSet == VK_NULL_HANDLE) return;

    VkDescriptorBufferInfo bInfo[3]{};
    bInfo[0].buffer = positions.buffer; bInfo[0].offset = 0; bInfo[0].range = VK_WHOLE_SIZE;
    bInfo[1].buffer = normals.buffer ? normals.buffer : positions.buffer; bInfo[1].offset = 0; bInfo[1].range = VK_WHOLE_SIZE;
    bInfo[2].buffer = weights.buffer; bInfo[2].offset = 0; bInfo[2].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bInfo[i];
    }
    vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sculptPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sculptPipelineLayout, 0, 1, &descSet, 0, nullptr);
    if (pushConstants && pushSize > 0) {
        vkCmdPushConstants(cmd, m_sculptPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushConstants);
    }

    const uint32_t WG = 128;
    uint32_t groups = (vertexCount + WG - 1) / WG;
    vkCmdDispatch(cmd, groups, 1, 1);

    // Ensure writes are visible to subsequent graphics or BLAS updates
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    endSingleTimeCommands(cmd);
}

void VulkanDevice::dispatchSkinning(uint32_t blasIndex, const std::vector<Matrix4x4>& boneMatrices) {
    if (!m_device || !m_commandPool || !m_computeQueue) return;
    if (blasIndex >= m_blasList.size() || m_skinningPipeline == VK_NULL_HANDLE) return;
    if (m_skinningPipelineLayout == VK_NULL_HANDLE || m_skinningDescPool == VK_NULL_HANDLE || m_skinningDescLayout == VK_NULL_HANDLE) return;
    auto& blas = m_blasList[blasIndex];
    if (!blas.hasSkinning) return;
    if (!blas.baseVertexBuffer.buffer || !blas.boneIndexBuffer.buffer || !blas.boneWeightBuffer.buffer || !blas.vertexBuffer.buffer) return;
    // Skinning compute bindings require both base and destination normal buffers.
    if (!blas.baseNormalBuffer.buffer || !blas.normalBuffer.buffer) return;
    if (blas.vertexCount == 0) return;

    uint64_t boneMatSize = boneMatrices.size() * sizeof(Matrix4x4);
    if (boneMatSize == 0) return;

    // ── 1. Persist bone matrix buffer (realloc only when it needs to grow) ─────────
    if (!blas.persistentBoneMatsBuffer.buffer || blas.persistentBoneMatsBufSize < boneMatSize) {
        if (blas.persistentBoneMatsBuffer.buffer) destroyBuffer(blas.persistentBoneMatsBuffer);
        BufferCreateInfo bc{};
        bc.size = boneMatSize;
        bc.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        bc.location = MemoryLocation::CPU_TO_GPU;
        blas.persistentBoneMatsBuffer = createBuffer(bc);
        if (!blas.persistentBoneMatsBuffer.buffer || !blas.persistentBoneMatsBuffer.memory) {
            blas.persistentBoneMatsBuffer = {};
            blas.persistentBoneMatsBufSize = 0;
            return;
        }
        blas.persistentBoneMatsBufSize = boneMatSize;
    }
    {
        void* m = nullptr;
        if (vkMapMemory(m_device, blas.persistentBoneMatsBuffer.memory, 0, boneMatSize, 0, &m) != VK_SUCCESS || !m) {
            return;
        }
        memcpy(m, boneMatrices.data(), boneMatSize);
        vkUnmapMemory(m_device, blas.persistentBoneMatsBuffer.memory);
    }

    // ── 2. Cached descriptor set — allocate once, update in-place every frame ─────
    if (blas.skinningDescSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_skinningDescPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_skinningDescLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &blas.skinningDescSet) != VK_SUCCESS ||
            blas.skinningDescSet == VK_NULL_HANDLE) {
            return;
        }
    }

    // normalBuffer is an aliased view of the same VkBuffer as vertexBuffer but at byte
    // offset = vertexCount*12 (positions come first, then normals in the combined buffer).
    // We MUST pass this byte offset to the descriptor or the compute shader will overwrite
    // position data with normal data (causing the "all triangles collapse to sphere" bug).
    const uint64_t normalByteOffset =
        (blas.normalBuffer.buffer && blas.normalBuffer.buffer == blas.vertexBuffer.buffer)
        ? (blas.normalBuffer.deviceAddress - blas.vertexBuffer.deviceAddress)
        : 0;
    const uint64_t normalRange = blas.normalBuffer.buffer ? (uint64_t)blas.vertexCount * 12 : 0;

    VkDescriptorBufferInfo bInfo[7]{};
    bInfo[0].buffer = blas.baseVertexBuffer.buffer;        bInfo[0].offset = 0; bInfo[0].range = VK_WHOLE_SIZE;
    bInfo[1].buffer = blas.baseNormalBuffer.buffer;        bInfo[1].offset = 0; bInfo[1].range = VK_WHOLE_SIZE;
    bInfo[2].buffer = blas.boneIndexBuffer.buffer;         bInfo[2].offset = 0; bInfo[2].range = VK_WHOLE_SIZE;
    bInfo[3].buffer = blas.boneWeightBuffer.buffer;        bInfo[3].offset = 0; bInfo[3].range = VK_WHOLE_SIZE;
    bInfo[4].buffer = blas.persistentBoneMatsBuffer.buffer; bInfo[4].offset = 0; bInfo[4].range = VK_WHOLE_SIZE;
    bInfo[5].buffer = blas.vertexBuffer.buffer;            bInfo[5].offset = 0;                bInfo[5].range = (uint64_t)blas.vertexCount * 12;
    bInfo[6].buffer = blas.normalBuffer.buffer             ? blas.normalBuffer.buffer : blas.vertexBuffer.buffer;
    bInfo[6].offset = normalByteOffset;
    bInfo[6].range  = normalRange ? normalRange : (uint64_t)blas.vertexCount * 12;

    VkWriteDescriptorSet writes[7]{};
    for (int i = 0; i < 7; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = blas.skinningDescSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bInfo[i];
    }
    vkUpdateDescriptorSets(m_device, 7, writes, 0, nullptr);

    // ── 3. Single command buffer: compute dispatch + BLAS update ──────────────────
    //    Only ONE vkQueueWaitIdle per call instead of two separate submits.
    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;

    // ── Compute: skin vertices ───────────────────────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_skinningPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_skinningPipelineLayout, 0, 1, &blas.skinningDescSet, 0, nullptr);
    uint32_t params[2] = { blas.vertexCount, (uint32_t)boneMatrices.size() };
    vkCmdPushConstants(cmd, m_skinningPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, params);
    uint32_t groupCount = (blas.vertexCount + 255) / 256;
    vkCmdDispatch(cmd, groupCount, 1, 1);

    // ── Barrier: vertex writes must be visible to AS build ───────────────────────
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    // ── BLAS update: refit in the same command buffer ────────────────────────────
    if (blas.accel != VK_NULL_HANDLE && fpCmdBuildAccelerationStructuresKHR) {
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = blas.vertexBuffer.deviceAddress;
        triangles.vertexStride = 12;
        triangles.maxVertex = blas.vertexCount - 1;
        triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = 0;
        geometry.geometry.triangles = triangles;

        uint32_t primitiveCount = blas.vertexCount / 3;

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        buildInfo.srcAccelerationStructure = blas.accel;
        buildInfo.dstAccelerationStructure = blas.accel;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        fpGetAccelerationStructureBuildSizesKHR(m_device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

        BufferCreateInfo scratchBufInfo;
        scratchBufInfo.size = sizeInfo.buildScratchSize;
        scratchBufInfo.usage = BufferUsage::STORAGE;
        scratchBufInfo.location = MemoryLocation::GPU_ONLY;
        auto scratchBuffer = createBuffer(scratchBufInfo);
        if (!scratchBuffer.buffer) {
            endSingleTimeCommands(cmd);
            return;
        }
        buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = primitiveCount;
        const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;
        fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);

        endSingleTimeCommands(cmd); // single GPU sync
        destroyBuffer(scratchBuffer);
    } else {
        endSingleTimeCommands(cmd);
    }
}
} // namespace VulkanRT

// Helper: load SPIR-V from file
static std::vector<uint32_t> loadSPV(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) return {};
    size_t size = (size_t)file.tellg();
    std::vector<uint32_t> code(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);
    return code;
}

static uint64_t estimateSceneTextureBytes(const Texture* tex, uint32_t uploadChannels) {
    if (!tex || tex->width <= 0 || tex->height <= 0) return 0;
    const uint64_t pixelCount = static_cast<uint64_t>(tex->width) * static_cast<uint64_t>(tex->height);
    if (tex->is_hdr) {
        return pixelCount * (uploadChannels <= 1 ? 4ull : 16ull);
    }
    return pixelCount * static_cast<uint64_t>((std::max)(1u, uploadChannels));
}

static std::string buildSceneTextureKey(const Texture* tex,
                                        TextureType textureType,
                                        bool forceLinear,
                                        bool preferSingleChannel) {
    if (!tex) return {};

    std::ostringstream oss;
    if (!tex->name.empty()) {
        oss << "tex:" << tex->name;
    } else {
        oss << "tex_ptr:" << reinterpret_cast<uintptr_t>(tex);
    }
    oss << "|type=" << static_cast<uint32_t>(textureType)
        << "|forceLinear=" << (forceLinear ? 1 : 0)
        << "|preferSingle=" << (preferSingleChannel ? 1 : 0)
        << "|hdr=" << (tex->is_hdr ? 1 : 0);
    return oss.str();
}

static Backend::TextureHandle registerSceneTexture(Backend::SceneTextureManager* manager,
                                                   const Texture* tex,
                                                   TextureType textureType,
                                                   bool forceLinear,
                                                   bool preferSingleChannel,
                                                   Backend::TextureConsumer consumers) {
    if (!manager || !tex || !tex->is_loaded()) {
        return {};
    }

    const bool useSrgb = forceLinear ? false : tex->is_srgb;
    const TextureCompressionPlan plan = buildTextureCompressionPlan(tex, textureType);
    const bool canUseSingleChannel =
        (preferSingleChannel || plan.preferSingleChannelFallback) &&
        !useSrgb && tex->is_gray_scale && !tex->has_alpha;

    return manager->registerTextureKey(
        buildSceneTextureKey(tex, textureType, forceLinear, preferSingleChannel),
        consumers,
        static_cast<uint32_t>((std::max)(0, tex->width)),
        static_cast<uint32_t>((std::max)(0, tex->height)),
        estimateSceneTextureBytes(tex, canUseSingleChannel ? 1u : 4u));
}

static Backend::TextureHandle registerTerrainSplatTexture(Backend::SceneTextureManager* manager,
                                                          const Texture* tex,
                                                          Backend::TextureConsumer consumers) {
    if (!manager || !tex || !tex->is_loaded()) {
        return {};
    }

    return manager->registerTextureKey(
        buildSceneTextureKey(tex, TextureType::Unknown, true, false),
        consumers,
        static_cast<uint32_t>((std::max)(0, tex->width)),
        static_cast<uint32_t>((std::max)(0, tex->height)),
        estimateSceneTextureBytes(tex, 4u));
}

namespace Backend {

VulkanBackendAdapter::VulkanBackendAdapter()
    : m_device(std::make_unique<VulkanRT::VulkanDevice>()),
      m_sceneTextureManager(getSharedSceneTextureManager()) {
    m_targetSamples = 1000; // Default
}

VulkanBackendAdapter::~VulkanBackendAdapter() {
    shutdown();
}

bool VulkanBackendAdapter::initialize() {
    if (!m_device) {
        m_device = std::make_unique<VulkanRT::VulkanDevice>();
    }
    if (!m_sceneTextureManager) {
        m_sceneTextureManager = getSharedSceneTextureManager();
    }
    if (m_sceneTextureManager) {
        // Shared manager outlives Vulkan device instances. When we recreate the
        // backend after OptiX/device switches, any previously published raw
        // VkImage/VkImageView/VkSampler handles for this owner scope are stale.
        // Clear them before the new device comes online so reuse only happens
        // against backings created by the current Vulkan device lifetime.
        m_sceneTextureManager->clearAllVulkanBackingForOwner(sceneTextureOwnerScope());
    }

#ifdef _DEBUG
    bool validation = true;
#else
    bool validation = false;
#endif
    bool ok = m_device->initialize(true, validation);
    if (ok) {
        m_sceneTextureManager->initialize(captureRuntimeRenderCapabilities(), "VulkanBackendAdapter");
    }

    if (ok && !m_device->hasHardwareRT()) {
        if (m_device->supportsGraphicsQueue()) {
            VK_INFO() << "[VulkanBackendAdapter] No hardware RT — path tracing disabled, "
                         "but raster viewport (Solid/Matcap) is available." << std::endl;
        } else {
            VK_ERROR() << "[VulkanBackendAdapter] No hardware RT and no graphics queue, "
                          "Vulkan backend disabled." << std::endl;
            m_device->shutdown();
            return false;
        }
    }

    if (ok && !m_cachedLights.empty()) {
        VK_INFO() << "[VulkanBackendAdapter] Uploading cached lights after device init (" << m_cachedLights.size() << ")" << std::endl;
        setLights(m_cachedLights);
        m_cachedLights.clear();
    }

    if (ok) {
        m_forceClearOnNextPresent = true;
        m_hasPresentedRenderedFrame = false;
        m_currentSamples = 0;
        m_testInitialized = false;
        m_device->m_rtPipelineReady = false;
    }
    return ok;
}

void VulkanBackendAdapter::purgeUploadedTextureCacheLocked() {
    if (!m_device) return;
    m_device->waitIdle();
    m_device->clearPendingRTTextureDescriptors();
    // Let manager call destroyFns for all manager-tracked textures (device still alive here).
    // The callbacks erase their entries from m_uploadedImages and call destroyImage, so the
    // loop below only has to clean up non-manager-tracked (legacy) entries.
    if (m_sceneTextureManager) {
        m_sceneTextureManager->destroyAllVulkanBackingForOwner(sceneTextureOwnerScope());
    }
    // Signal lambdas still in SceneTextureManager that containers are about to be cleared.
    // Any lambda that runs after this point must not touch m_uploadedImages etc.
    *m_containerAlive = false;
    m_containerAlive = std::make_shared<bool>(true); // re-arm for next upload cycle
    for (auto& [id, img] : m_uploadedImages) {
        if (img.image || img.view || img.memory || img.sampler) {
            m_device->destroyImage(img);
        }
    }
    m_uploadedImages.clear();
    m_uploadedImageIDs.clear();
    m_textureIdToCacheKey.clear();
    m_nextTextureID = 1;
    ++m_textureCacheGeneration;
    if (m_sceneTextureManager) {
        m_sceneTextureManager->logBudgetSummary("after-purge");
    }
}

void VulkanBackendAdapter::releaseInactiveViewportTextureCache() {
    if (!m_device || !m_device->isInitialized()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // This is intended for the dedicated viewport backend after leaving
    // Material Preview or switching to Rendered mode. Tear down interactive
    // descriptor sets before destroying texture views, so no inactive Material
    // Preview descriptor keeps stale VkImageView handles around for the next
    // mode switch.
    m_device->waitIdle();
    destroyInteractiveViewportResourcesImpl(false);
    // Also clear non-Vk handles such as material-preview env texture ids.
    // They point into m_uploadedImages, which is purged just below; keeping
    // stale integer ids makes the next Material Preview descriptor rebuild
    // believe those textures still exist.
    m_interactiveViewport = {};
    m_interactiveViewport.dirty = true;
    purgeUploadedTextureCacheLocked();
    SCENE_LOG_INFO(std::string("[Vulkan] Released inactive viewport texture cache for owner=") +
                   sceneTextureOwnerScope());
}

void VulkanBackendAdapter::setInteractiveViewportMatcap(int64_t textureID) {
    setInteractiveViewportMatcapImpl(textureID);
}

void VulkanBackendAdapter::setInteractiveViewportMatcapImpl(int64_t textureID) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;

    // textureID <= 0 means "clear matcap" → revert to procedural
    if (textureID <= 0) {
        m_interactiveViewport.matcapUserLoaded = false;
        m_interactiveViewport.dirty = true;
        SCENE_LOG_INFO("[Matcap] Cleared — reverting to procedural matcap");
        return;
    }

    auto it = m_uploadedImages.find(textureID);
    if (it == m_uploadedImages.end()) { SCENE_LOG_INFO("[Matcap] textureID not found in uploaded images"); return; }

    VkDevice vkDevice = m_device->getDevice();

    // Assign the uploaded image to interactive viewport matcap reference.
    m_interactiveViewport.matcapImage = it->second;
    m_interactiveViewport.matcapUserLoaded = true;

    // If descriptor set doesn't exist yet, allocate it now
    if (m_interactiveViewport.matcapDescSet == VK_NULL_HANDLE &&
        m_interactiveViewport.matcapDescPool != VK_NULL_HANDLE &&
        m_interactiveViewport.matcapDescLayout != VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = m_interactiveViewport.matcapDescPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &m_interactiveViewport.matcapDescLayout;
        vkAllocateDescriptorSets(vkDevice, &dsai, &m_interactiveViewport.matcapDescSet);
    }

    // Update descriptor set with the user's texture
    if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE && m_interactiveViewport.matcapImage.image) {
        VkDescriptorImageInfo di{};
        di.sampler = m_interactiveViewport.matcapImage.sampler;
        di.imageView = m_interactiveViewport.matcapImage.view;
        di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet wds{};
        wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet = m_interactiveViewport.matcapDescSet;
        wds.dstBinding = 0;
        wds.dstArrayElement = 0;
        wds.descriptorCount = 1;
        wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wds.pImageInfo = &di;

        vkUpdateDescriptorSets(vkDevice, 1, &wds, 0, nullptr);
        SCENE_LOG_INFO("[Matcap] Texture bound: " + std::to_string(it->second.width) + "x" + std::to_string(it->second.height));
    } else {
        SCENE_LOG_INFO("[Matcap] WARNING: Could not bind texture — descSet="
                       + std::to_string((uintptr_t)m_interactiveViewport.matcapDescSet)
                       + " pool=" + std::to_string((uintptr_t)m_interactiveViewport.matcapDescPool)
                       + " layout=" + std::to_string((uintptr_t)m_interactiveViewport.matcapDescLayout));
    }

    m_interactiveViewport.dirty = true;
}

void VulkanBackendAdapter::setInteractiveViewportMatcapPreset(int preset) {
    setInteractiveViewportMatcapPresetImpl(preset);
}

void VulkanBackendAdapter::setInteractiveViewportMatcapPresetImpl(int preset) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;
    // clamp to allowed range 0..9
    if (preset < 0) preset = 0;
    if (preset > 9) preset = 9;
    // Selecting a preset clears any user-loaded texture
    m_interactiveViewport.matcapUserLoaded = false;
    m_interactiveViewport.matcapPreset = preset;
    m_interactiveViewport.dirty = true;
    SCENE_LOG_INFO("[Matcap] Preset selected: " + std::to_string(preset));
}

/* custom matcap support removed */

void VulkanBackendAdapter::shutdown() {
    if (m_sceneTextureManager) {
        m_sceneTextureManager->clearAllVulkanBackingForOwner(sceneTextureOwnerScope());
    }
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    if (!m_device) {
        return;
    }

    if (m_device->isInitialized()) {
        m_device->waitIdle();
    }

    destroyInteractiveViewportResources(false);
    destroyAllRasterMeshes();

    // Adapter-owned output/readback resources.
    if (m_outputImage.image) {
        m_device->destroyImage(m_outputImage);
    }
    if (m_varianceImage.image) {
        m_device->destroyImage(m_varianceImage);
    }
    if (m_stagingBuffer.buffer) {
        m_device->destroyBuffer(m_stagingBuffer);
    }
    if (m_tonemappedImage.image) {
        m_device->destroyImage(m_tonemappedImage);
    }
    for (auto& s : m_tonemappedStagings) {
        if (s.buffer) m_device->destroyBuffer(s);
    }
    if (m_denoiserColorImage.image) {
        m_device->destroyImage(m_denoiserColorImage);
    }
    if (m_denoiserAlbedoImage.image) {
        m_device->destroyImage(m_denoiserAlbedoImage);
    }
    if (m_denoiserNormalImage.image) {
        m_device->destroyImage(m_denoiserNormalImage);
    }
    if (m_denoiserPositionImage.image) {
        m_device->destroyImage(m_denoiserPositionImage);
    }
    if (m_denoiserColorStagingBuffer.buffer) {
        m_device->destroyBuffer(m_denoiserColorStagingBuffer);
    }
    if (m_denoiserAlbedoStagingBuffer.buffer) {
        m_device->destroyBuffer(m_denoiserAlbedoStagingBuffer);
    }
    if (m_denoiserNormalStagingBuffer.buffer) {
        m_device->destroyBuffer(m_denoiserNormalStagingBuffer);
    }
    if (m_denoiserPositionStagingBuffer.buffer) {
        m_device->destroyBuffer(m_denoiserPositionStagingBuffer);
    }

    // GPU-direct OIDN interop: destroy CUDA imports + prep buffers BEFORE
    // freeing the underlying exportable VkDeviceMemory.
    // Drain the async denoiser copy fence first; same rationale as the resize
    // path — in-flight copy would target soon-to-be-freed staging.
    m_device->drainDenoiserCopy();
    destroyGpuDenoiserInterop();
    if (m_denoiserColorSharedStaging.buffer)  m_device->destroyBuffer(m_denoiserColorSharedStaging);
    if (m_denoiserAlbedoSharedStaging.buffer) m_device->destroyBuffer(m_denoiserAlbedoSharedStaging);
    if (m_denoiserNormalSharedStaging.buffer) m_device->destroyBuffer(m_denoiserNormalSharedStaging);
    m_denoiserColorSharedStaging = {};
    m_denoiserAlbedoSharedStaging = {};
    m_denoiserNormalSharedStaging = {};
    m_denoiserColorSharedHandle = nullptr;
    m_denoiserAlbedoSharedHandle = nullptr;
    m_denoiserNormalSharedHandle = nullptr;
    m_denoiserColorSharedAllocSize = 0;
    m_denoiserAlbedoSharedAllocSize = 0;
    m_denoiserNormalSharedAllocSize = 0;

    // Adapter-owned uploaded texture/image cache.
    purgeUploadedTextureCacheLocked();

    // Adapter-owned NanoVDB buffers.
    for (auto& [id, buf] : m_vdbBuffers) {
        (void)id;
        if (buf.buffer) {
            m_device->destroyBuffer(buf);
        }
    }
    m_vdbBuffers.clear();

    for (auto& [id, buf] : m_vdbTempBuffers) {
        (void)id;
        if (buf.buffer) {
            m_device->destroyBuffer(buf);
        }
    }
    m_vdbTempBuffers.clear();

    // Reset adapter caches/state.
    // Phase 3d: free device-resident CC geometry buffers (BLASes are being torn down in
    // this shutdown; the shared compute backend still owns the same VkDevice here).
    for (auto& e : m_deviceResidentMeshes) {
        MeshModifiers::CCDeviceGeometry g; g.bufferId = e.bufferId;
        MeshModifiers::releaseCCDeviceGeometry(g);
    }
    m_deviceResidentMeshes.clear();
    for (uint64_t bufId : m_deviceResidentReleaseQueue) {
        MeshModifiers::CCDeviceGeometry g; g.bufferId = bufId;
        MeshModifiers::releaseCCDeviceGeometry(g);
    }
    m_deviceResidentReleaseQueue.clear();

    m_orderedVDBInstances.clear();
    m_meshRegistry.clear();
    m_blasMaterialBounds.clear();
    m_blasBuiltNonOpaque.clear();
    m_blasMaterialIds.clear();
    m_vkInstances.clear();
    m_lastObjects.clear();
    m_instanceSources.clear();
    m_instance_sync_cache.clear();
    m_hairVkInstances.clear();
    m_hairGroomRegistry.clear();
    m_meshBlasCount = 0;
    m_volumeBlasIndex = UINT32_MAX;
    m_topology_dirty = true;
    m_cachedLights.clear();
    m_cachedWorld = WorldData{};
    m_envTexID = 0;
    m_atmosphereLutReady = false;
    m_lastCameraHash = 0;
    m_prevViewDir = Vec3(0.0f);
    m_hasPrevView = false;
    m_forceClearOnNextPresent = false;
    m_hasPresentedRenderedFrame = false;
    m_currentSamples = 0;
    m_testInitialized = false;
    m_hdrPixels.clear();
    m_denoiserColorPixels.clear();
    m_denoiserAlbedoPixels.clear();
    m_denoiserNormalPixels.clear();
    m_denoiserPositionPixels.clear();
    m_outputImage = {};
    m_varianceImage = {};
    m_stagingBuffer = {};
    m_tonemappedImage = {};
    for (auto& s : m_tonemappedStagings) s = {};
        m_tonemappedSlotInFlight[0] = m_tonemappedSlotInFlight[1] = false;
        m_tonemappedFrameSlot = 0;
    m_denoiserColorImage = {};
    m_denoiserAlbedoImage = {};
    m_denoiserNormalImage = {};
    m_denoiserPositionImage = {};
    m_denoiserColorStagingBuffer = {};
    m_denoiserAlbedoStagingBuffer = {};
    m_denoiserNormalStagingBuffer = {};
    m_denoiserPositionStagingBuffer = {};

    // Destroy the device object once. VulkanDevice destructor calls shutdown(),
    // so reset the unique_ptr to release all remaining VulkanDevice-owned resources.
    m_device.reset();
}

void VulkanBackendAdapter::loadShaders(const ShaderProgramData& data) {
    // TODO: Load SPIR-V shaders and create pipeline
    (void)data;
}

BackendInfo VulkanBackendAdapter::getInfo() const {
    BackendInfo info;
    info.type = BackendType::VULKAN_RT;
    info.name = "Vulkan RT";
    if (!m_device) {
        info.deviceName = "Uninitialized";
        info.hasHardwareRT = false;
        info.vramBytes = 0;
        info.driverVersion = "0";
        return info;
    }
    const auto& caps = m_device->getCapabilities();
    info.deviceName = caps.deviceName;
    info.hasHardwareRT = m_device->hasHardwareRT();
    info.vramBytes = caps.dedicatedVRAM;
    info.driverVersion = std::to_string(caps.driverVersion);
    return info;
}

GpuMemoryStats VulkanBackendAdapter::getMemoryStats() const {
    GpuMemoryStats stats;
    if (!m_device) {
        return stats;
    }

    const auto& caps = m_device->getCapabilities();
    stats.totalBytes = caps.dedicatedVRAM;

    if (m_sceneTextureManager) {
        stats.trackedTextureBytes = m_sceneTextureManager->totalResidentTextureBytes();
        stats.trackedTextureBytesThisBackend =
            m_sceneTextureManager->estimatedTextureBytesForOwner(sceneTextureOwnerScope());
        stats.usedBytes = stats.trackedTextureBytesThisBackend;
        stats.hasTrackedTextures = true;
    }

    return stats;
}

// Geometry implementation
uint32_t VulkanBackendAdapter::uploadTriangles(const std::vector<TriangleData>& triangles, const std::string& meshName) {
    if (!m_device || !m_device->isInitialized() || triangles.empty()) return UINT32_MAX;
    if (!m_device->hasHardwareRT()) return UINT32_MAX;

    // Fast path: check if mesh already uploaded
    auto it = m_meshRegistry.find(meshName);
    if (it != m_meshRegistry.end()) return it->second;

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<int32_t> boneIndices;
    std::vector<float> boneWeights;
    bool hasSkinning = false;
    for (const auto& t : triangles) {
        if (t.hasSkinData && triangleDataHasEffectiveSkinData(t)) {
            hasSkinning = true;
            break;
        }
    }
    
    positions.resize(triangles.size() * 9);
    normals.resize(triangles.size() * 9);
    uvs.resize(triangles.size() * 6);
    if (hasSkinning) {
        boneIndices.assign(triangles.size() * 12, -1);
        boneWeights.assign(triangles.size() * 12, 0.0f);
    }
    std::vector<uint32_t> materialIndices(triangles.size());

    auto fillRange = [&triangles, &positions, &normals, &uvs,
                      &materialIndices, &boneIndices, &boneWeights,
                      hasSkinning](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const auto& t = triangles[i];
            const size_t posBase = i * 9;
            const size_t uvBase = i * 6;
            const size_t skinBase = i * 12;

            positions[posBase + 0] = t.v0.x; positions[posBase + 1] = t.v0.y; positions[posBase + 2] = t.v0.z;
            positions[posBase + 3] = t.v1.x; positions[posBase + 4] = t.v1.y; positions[posBase + 5] = t.v1.z;
            positions[posBase + 6] = t.v2.x; positions[posBase + 7] = t.v2.y; positions[posBase + 8] = t.v2.z;

            normals[posBase + 0] = t.n0.x; normals[posBase + 1] = t.n0.y; normals[posBase + 2] = t.n0.z;
            normals[posBase + 3] = t.n1.x; normals[posBase + 4] = t.n1.y; normals[posBase + 5] = t.n1.z;
            normals[posBase + 6] = t.n2.x; normals[posBase + 7] = t.n2.y; normals[posBase + 8] = t.n2.z;

            uvs[uvBase + 0] = t.uv0.x; uvs[uvBase + 1] = t.uv0.y;
            uvs[uvBase + 2] = t.uv1.x; uvs[uvBase + 3] = t.uv1.y;
            uvs[uvBase + 4] = t.uv2.x; uvs[uvBase + 5] = t.uv2.y;

            uint16_t mId = t.materialID;
            if (mId == MaterialManager::INVALID_MATERIAL_ID) mId = 0;
            materialIndices[i] = static_cast<uint32_t>(mId);

            if (!hasSkinning) continue;
            for (int b = 0; b < 4; ++b) {
                boneIndices[skinBase + static_cast<size_t>(b)] = t.hasSkinData ? t.boneIndices_v0[b] : -1;
                boneWeights[skinBase + static_cast<size_t>(b)] = t.hasSkinData ? t.boneWeights_v0[b] : 0.0f;
                boneIndices[skinBase + 4 + static_cast<size_t>(b)] = t.hasSkinData ? t.boneIndices_v1[b] : -1;
                boneWeights[skinBase + 4 + static_cast<size_t>(b)] = t.hasSkinData ? t.boneWeights_v1[b] : 0.0f;
                boneIndices[skinBase + 8 + static_cast<size_t>(b)] = t.hasSkinData ? t.boneIndices_v2[b] : -1;
                boneWeights[skinBase + 8 + static_cast<size_t>(b)] = t.hasSkinData ? t.boneWeights_v2[b] : 0.0f;
            }
        }
    };

    constexpr size_t kUploadPackParallelThreshold = 4096;
    unsigned packThreads = std::thread::hardware_concurrency();
    if (packThreads == 0) packThreads = 4;
    if (triangles.size() < kUploadPackParallelThreshold || packThreads < 2) {
        fillRange(0, triangles.size());
    } else {
        const size_t chunk = (triangles.size() + packThreads - 1) / packThreads;
        std::vector<std::future<void>> futures;
        futures.reserve(packThreads);
        for (unsigned t = 0; t < packThreads; ++t) {
            const size_t s = t * chunk;
            const size_t e = (std::min)(s + chunk, triangles.size());
            if (s >= e) break;
            futures.push_back(std::async(std::launch::async, fillRange, s, e));
        }
        for (auto& f : futures) f.get();
    }

    if (!hasSkinning) {
        boneIndices.clear();
        boneWeights.clear();
    }

    bool opaqueGeometry = true;
    for (uint32_t materialIndex : materialIndices) {
        if (!materialCanUseOpaqueFastPath(materialIndex)) {
            opaqueGeometry = false;
            break;
        }
    }

    // Photon caustics: BLAS-local AABB per MATERIAL (which materials are caustic
    // casters is decided per frame against the live MaterialManager — deciding
    // here at upload time missed project loads where materials resolve later).
    // Per-material so a flat mesh's one glass object doesn't inflate the photon
    // target to the whole scene. Vertices may be LOCAL (P_orig) for
    // solo/project-loaded meshes — the photon target applies the TLAS instance
    // transform.
    std::unordered_map<uint32_t, CausticBounds> causticMatBounds;
    {
        const size_t triN = (std::min)(materialIndices.size(), positions.size() / 9);
        for (size_t t = 0; t < triN; ++t) {
            auto ins = causticMatBounds.try_emplace(materialIndices[t],
                CausticBounds{ 1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f });
            CausticBounds& mb = ins.first->second;
            for (size_t c = 0; c < 3; ++c) {
                const size_t i = t * 9 + c * 3;
                mb.minX = std::min(mb.minX, positions[i + 0]); mb.maxX = std::max(mb.maxX, positions[i + 0]);
                mb.minY = std::min(mb.minY, positions[i + 1]); mb.maxY = std::max(mb.maxY, positions[i + 1]);
                mb.minZ = std::min(mb.minZ, positions[i + 2]); mb.maxZ = std::max(mb.maxZ, positions[i + 2]);
            }
        }
    }

    VulkanRT::BLASCreateInfo blasInfo;
    blasInfo.vertexData = positions.data();
    blasInfo.normalData = normals.data();
    blasInfo.uvData = uvs.data();
    blasInfo.vertexCount = (uint32_t)triangles.size() * 3;
    blasInfo.vertexStride = 12; // 3 * float
    blasInfo.materialIndexData = materialIndices.data();
    blasInfo.materialIndexCount = (uint32_t)materialIndices.size();
    blasInfo.opaqueGeometry = opaqueGeometry;
    
    blasInfo.hasSkinning = hasSkinning;
    const bool allowDynamicUpdate =
        hasSkinning ||
        meshName.rfind("[World-Solo]-", 0) == 0 ||
        meshName.rfind("[InstSource]-", 0) == 0 ||
        meshName.rfind("[InstGroup]-", 0) == 0;
    blasInfo.allowUpdate = allowDynamicUpdate;
    blasInfo.boneIndicesData = hasSkinning ? boneIndices.data() : nullptr;
    blasInfo.boneWeightsData = hasSkinning ? boneWeights.data() : nullptr;
    
    uint32_t blasIndex = m_device->createBLAS(blasInfo);
    if (blasIndex == UINT32_MAX) {
        SCENE_LOG_ERROR("[Vulkan] Failed to create BLAS for mesh: " + meshName);
        return UINT32_MAX;
    }

    m_meshRegistry[meshName] = blasIndex;
    m_blasMaterialIds[blasIndex] = std::move(materialIndices);
    m_blasBuiltNonOpaque[blasIndex] = !opaqueGeometry;
    {
        auto& entries = m_blasMaterialBounds[blasIndex];
        entries.assign(causticMatBounds.begin(), causticMatBounds.end());
    }

    // Reset geometry data buffer because a new BLAS was added
    if (m_device->m_geometryDataBuffer.buffer) {
        m_device->destroyBuffer(m_device->m_geometryDataBuffer);
    }

   // SCENE_LOG_INFO("[Vulkan] Uploaded mesh: " + meshName + " (" + std::to_string(triangles.size()) + " tris)");
    return blasIndex;
}

uint32_t VulkanBackendAdapter::uploadTriangleMeshIndexed(const TriangleMesh* mesh,
                                                         const std::string& meshName,
                                                         bool useOriginalSpace) {
    if (!m_device || !m_device->isInitialized() || !m_device->hasHardwareRT()) return UINT32_MAX;
    if (!mesh || !mesh->geometry) return UINT32_MAX;

    // Fast path: mesh already uploaded.
    auto it = m_meshRegistry.find(meshName);
    if (it != m_meshRegistry.end()) return it->second;

    const DNA::GeometryDetail& geom = *mesh->geometry;
    const size_t vCount   = geom.get_vertex_count();
    const size_t idxCount = geom.indices.size();
    const size_t triCount = idxCount / 3;
    if (vCount == 0 || triCount == 0) return UINT32_MAX;

    // Vertex space must match what the non-indexed solo path uploaded (Triangle facade reads
    // geom P_orig for local/static and geom P for live), so the result is bit-identical, just
    // deduplicated. Fall back to live if the bind-pose channel is absent.
    const Vec3* srcP = useOriginalSpace ? geom.get_positions_orig() : geom.get_positions();
    const Vec3* srcN = useOriginalSpace ? geom.get_normals_orig()   : geom.get_normals();
    if (!srcP) srcP = geom.get_positions();
    if (!srcN) srcN = geom.get_normals();
    if (!srcP) return UINT32_MAX;
    const Vec2* srcUV = geom.get_uvs();
    const uint16_t* srcMat = geom.get_material_ids();

    std::vector<float> positions(vCount * 3);
    std::vector<float> normals(vCount * 3);
    std::vector<float> uvs(vCount * 2, 0.0f);
    for (size_t v = 0; v < vCount; ++v) {
        positions[v * 3 + 0] = srcP[v].x;
        positions[v * 3 + 1] = srcP[v].y;
        positions[v * 3 + 2] = srcP[v].z;
        if (srcN) {
            normals[v * 3 + 0] = srcN[v].x;
            normals[v * 3 + 1] = srcN[v].y;
            normals[v * 3 + 2] = srcN[v].z;
        } else {
            normals[v * 3 + 0] = 0.0f; normals[v * 3 + 1] = 1.0f; normals[v * 3 + 2] = 0.0f;
        }
        if (srcUV) {
            uvs[v * 2 + 0] = srcUV[v].x;
            uvs[v * 2 + 1] = srcUV[v].y;
        }
    }

    // Index buffer is the mesh's own flat index buffer (corner -> unique vertex).
    std::vector<uint32_t> indices(geom.indices.begin(), geom.indices.end());

    // Per-primitive material id (the shader looks this up by primID). All 3 corners of a face
    // share a material id (the weld key includes materialID), so corner 0 is representative.
    std::vector<uint32_t> materialIndices(triCount);
    for (size_t t = 0; t < triCount; ++t) {
        uint16_t mId = srcMat ? srcMat[geom.indices[t * 3 + 0]] : 0;
        if (mId == MaterialManager::INVALID_MATERIAL_ID) mId = 0;
        materialIndices[t] = static_cast<uint32_t>(mId);
    }

    bool opaqueGeometry = true;
    for (uint32_t materialIndex : materialIndices) {
        if (!materialCanUseOpaqueFastPath(materialIndex)) { opaqueGeometry = false; break; }
    }

    // Photon caustics: BLAS-local AABB per MATERIAL (caster decision is per frame
    // — see the non-indexed path). These vertices are LOCAL P_orig when
    // useOriginalSpace, so the photon target must apply the TLAS instance transform.
    std::unordered_map<uint32_t, CausticBounds> causticMatBounds;
    for (size_t t = 0; t < triCount; ++t) {
        auto ins = causticMatBounds.try_emplace(materialIndices[t],
            CausticBounds{ 1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f });
        CausticBounds& mb = ins.first->second;
        for (size_t c = 0; c < 3; ++c) {
            const size_t vi = (size_t)indices[t * 3 + c] * 3;
            if (vi + 2 >= positions.size()) continue;
            mb.minX = std::min(mb.minX, positions[vi + 0]); mb.maxX = std::max(mb.maxX, positions[vi + 0]);
            mb.minY = std::min(mb.minY, positions[vi + 1]); mb.maxY = std::max(mb.maxY, positions[vi + 1]);
            mb.minZ = std::min(mb.minZ, positions[vi + 2]); mb.maxZ = std::max(mb.maxZ, positions[vi + 2]);
        }
    }

    VulkanRT::BLASCreateInfo blasInfo;
    blasInfo.vertexData = positions.data();
    blasInfo.normalData = normals.data();
    blasInfo.uvData = uvs.data();
    blasInfo.vertexCount = (uint32_t)vCount;
    blasInfo.vertexStride = 12; // 3 * float
    blasInfo.indexData = indices.data();
    blasInfo.indexCount = (uint32_t)idxCount;
    blasInfo.materialIndexData = materialIndices.data();
    blasInfo.materialIndexCount = (uint32_t)triCount;
    blasInfo.opaqueGeometry = opaqueGeometry;
    blasInfo.hasSkinning = false; // skinned meshes are routed to uploadTriangles()
    blasInfo.allowUpdate = true;  // solo meshes are editable (sculpt/deform refit)

    uint32_t blasIndex = m_device->createBLAS(blasInfo);
    if (blasIndex == UINT32_MAX) {
        SCENE_LOG_ERROR("[Vulkan] Failed to create indexed BLAS for mesh: " + meshName);
        return UINT32_MAX;
    }

    m_meshRegistry[meshName] = blasIndex;
    m_blasMaterialIds[blasIndex] = std::move(materialIndices);
    m_soloBlasIndexedMesh[blasIndex] = SoloIndexedMeshInfo{ const_cast<TriangleMesh*>(mesh), useOriginalSpace };
    m_blasBuiltNonOpaque[blasIndex] = !opaqueGeometry;
    {
        auto& entries = m_blasMaterialBounds[blasIndex];
        entries.assign(causticMatBounds.begin(), causticMatBounds.end());
    }

    // Reset geometry data buffer because a new BLAS was added.
    if (m_device->m_geometryDataBuffer.buffer) {
        m_device->destroyBuffer(m_device->m_geometryDataBuffer);
    }
    return blasIndex;
}

// Interactive refit for a solo INDEXED BLAS: re-read the mesh SoA (vCount unique verts) and
// MODE_UPDATE the acceleration structure. Cheaper than the 3N gather (no per-corner expand)
// and topology-stable. Phase 2 will add a dirty-region partial path; for now any edit re-pushes
// the whole vertex array (still a refit, not a full rebuild).
bool VulkanBackendAdapter::refitIndexedSoloBLAS(uint32_t blasIndex) {
    auto idxIt = m_soloBlasIndexedMesh.find(blasIndex);
    if (idxIt == m_soloBlasIndexedMesh.end()) {
        SCENE_LOG_WARN("[refitIndexedSoloBLAS] Not found in m_soloBlasIndexedMesh");
        return false;
    }
    const SoloIndexedMeshInfo info = idxIt->second;
    if (!info.mesh || !info.mesh->geometry) {
        SCENE_LOG_WARN("[refitIndexedSoloBLAS] Mesh or geometry is null");
        return false;
    }
    if (blasIndex >= m_device->m_blasList.size()) {
        SCENE_LOG_WARN("[refitIndexedSoloBLAS] blasIndex out of bounds");
        return false;
    }
    const auto& blasHandle = m_device->m_blasList[blasIndex];
    if (!blasHandle.allowUpdate || blasHandle.vertexCount == 0) {
        SCENE_LOG_WARN("[refitIndexedSoloBLAS] BLAS update not allowed or vertex count is 0");
        return false;
    }

    const DNA::GeometryDetail& geom = *info.mesh->geometry;
    const size_t vCount = geom.get_vertex_count();
    if (vCount != blasHandle.vertexCount) {
        SCENE_LOG_WARN("[refitIndexedSoloBLAS] Vertex count mismatch: geom=" + std::to_string(vCount) + ", blas=" + std::to_string(blasHandle.vertexCount));
        return false;
    }

    const Vec3* srcP = info.localSpace ? geom.get_positions_orig() : geom.get_positions();
    const Vec3* srcN = info.localSpace ? geom.get_normals_orig()   : geom.get_normals();
    if (!srcP) srcP = geom.get_positions();
    if (!srcN) srcN = geom.get_normals();
    if (!srcP) {
        SCENE_LOG_WARN("[refitIndexedSoloBLAS] Position attribute not found");
        return false;
    }

    SCENE_LOG_INFO("[refitIndexedSoloBLAS] Uploading " + std::to_string(vCount) + " vertices for BLAS " + std::to_string(blasIndex));
    std::vector<float> positions(vCount * 3);
    std::vector<float> normals(vCount * 3);
    for (size_t v = 0; v < vCount; ++v) {
        positions[v * 3 + 0] = srcP[v].x;
        positions[v * 3 + 1] = srcP[v].y;
        positions[v * 3 + 2] = srcP[v].z;
        if (srcN) {
            normals[v * 3 + 0] = srcN[v].x;
            normals[v * 3 + 1] = srcN[v].y;
            normals[v * 3 + 2] = srcN[v].z;
        } else {
            normals[v * 3 + 0] = 0.0f; normals[v * 3 + 1] = 1.0f; normals[v * 3 + 2] = 0.0f;
        }
    }

    m_device->updateBLAS(blasIndex, positions.data(), normals.data());

    // Refresh the TLAS instance transform from the live handle (mirrors the non-indexed refit),
    // so a scale/rotate since the last full rebuild renders at the correct world-space pose.
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        if (m_vkInstances[i].blasIndex != blasIndex) continue;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            m_vkInstances[i].transform = inst->transform;
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            if (tri->getTransformPtr()) m_vkInstances[i].transform = tri->getTransformPtr()->getFinal();
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            if (tm->transform) m_vkInstances[i].transform = tm->transform->getFinal(); // flat mesh
        }
    }

    auto merged = m_vkInstances;
    for (const auto& h : m_hairVkInstances) merged.push_back(h);
    if (!merged.empty()) m_device->updateTLAS(merged);

    resetAccumulation();
    return true;
}

uint32_t VulkanBackendAdapter::uploadDeviceResidentMesh(const std::string& meshName,
                                                        uint64_t geometryDeviceAddress,
                                                        uint32_t vertexCount,
                                                        uint32_t triCount,
                                                        const std::vector<uint32_t>& materialIds,
                                                        bool opaqueGeometry) {
    if (!m_device || !m_device->isInitialized()) return UINT32_MAX;
    if (!m_device->hasHardwareRT()) return UINT32_MAX;
    if (geometryDeviceAddress == 0 || vertexCount == 0 || triCount == 0) return UINT32_MAX;

    // Cache like uploadTriangles — a rebuild reuses the existing BLAS for this mesh.
    auto it = m_meshRegistry.find(meshName);
    if (it != m_meshRegistry.end()) return it->second;

    // Build straight from the device buffer (non-indexed expanded soup; the mat block
    // lives in the buffer, declared via materialIndexCount so createBLAS sizes it).
    VulkanRT::BLASCreateInfo blasInfo;
    blasInfo.useDeviceGeometry = true;
    blasInfo.geometryDeviceAddress = geometryDeviceAddress;
    blasInfo.vertexCount = vertexCount;       // triCount * 3
    blasInfo.vertexStride = 12;               // 3 * float, non-indexed
    blasInfo.indexCount = 0;
    blasInfo.materialIndexCount = triCount;
    blasInfo.opaqueGeometry = opaqueGeometry;
    blasInfo.allowUpdate = true;              // Slice 4: position-only cage edits MODE_UPDATE-refit this BLAS

    uint32_t blasIndex = m_device->createBLAS(blasInfo);
    if (blasIndex == UINT32_MAX) {
        SCENE_LOG_ERROR("[Vulkan] Failed to create device-resident BLAS for mesh: " + meshName);
        return UINT32_MAX;
    }

    m_meshRegistry[meshName] = blasIndex;
    // Host material-id copy for shader lookup (the CC plan's per-triangle materials).
    m_blasMaterialIds[blasIndex] = materialIds.empty()
        ? std::vector<uint32_t>(triCount, 0u)
        : materialIds;
    m_blasBuiltNonOpaque[blasIndex] = !opaqueGeometry;
    {
        // Unique material list with INVERTED (empty) bounds — device-resident
        // meshes have no host positions. The caustic target skips inverted
        // boxes; the uploadMaterials opacity audit only needs the ids.
        std::unordered_map<uint32_t, char> seen;
        auto& entries = m_blasMaterialBounds[blasIndex];
        entries.clear();
        for (uint32_t mId : m_blasMaterialIds[blasIndex]) {
            if (seen.emplace(mId, 0).second) {
                entries.emplace_back(mId, CausticBounds{ 1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f });
            }
        }
    }

    if (m_device->m_geometryDataBuffer.buffer) {
        m_device->destroyBuffer(m_device->m_geometryDataBuffer);
    }
    return blasIndex;
}

void VulkanBackendAdapter::registerDeviceResidentCCMesh(const DeviceResidentCCMesh& mesh) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    for (auto& e : m_deviceResidentMeshes) {
        if (e.meshKey == mesh.meshKey) {
            // Replacing this key — the old GPU buffer may still be referenced by a live
            // BLAS, so defer its release until the next BLAS teardown (GPU idle).
            if (e.bufferId != 0 && e.bufferId != mesh.bufferId) {
                m_deviceResidentReleaseQueue.push_back(e.bufferId);
            }
            e = mesh;
            return;
        }
    }
    m_deviceResidentMeshes.push_back(mesh);
}

void VulkanBackendAdapter::removeDeviceResidentCCMesh(const std::string& meshKey) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    for (size_t i = 0; i < m_deviceResidentMeshes.size(); ++i) {
        if (m_deviceResidentMeshes[i].meshKey == meshKey) {
            if (m_deviceResidentMeshes[i].bufferId != 0) {
                m_deviceResidentReleaseQueue.push_back(m_deviceResidentMeshes[i].bufferId);
            }
            m_deviceResidentMeshes.erase(m_deviceResidentMeshes.begin() + i);
            return;
        }
    }
}

bool VulkanBackendAdapter::hasDeviceResidentCCNode(const std::string& cageNodeName) const {
    if (cageNodeName.empty()) return false;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    for (const auto& e : m_deviceResidentMeshes) {
        if (e.cageNodeName == cageNodeName) return true;
    }
    return false;
}

void VulkanBackendAdapter::queueDeviceResidentCCRefit(const std::string& meshKey, uint64_t newGeomBase,
                                                      uint64_t newBufferId, uint32_t vertexCount) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Coalesce: if a refit for this key is already pending (rapid sculpt before the render
    // thread drains), the earlier buffer was never bound to a BLAS — release it and keep
    // only the latest. The buffer the caller passes is the registered entry's current one.
    for (auto& req : m_deviceResidentRefitQueue) {
        if (req.meshKey == meshKey) {
            if (req.newBufferId != 0 && req.newBufferId != newBufferId)
                m_deviceResidentReleaseQueue.push_back(req.newBufferId);
            req.newGeomBase = newGeomBase;
            req.newBufferId = newBufferId;
            req.vertexCount = vertexCount;
            return;
        }
    }
    m_deviceResidentRefitQueue.push_back({ meshKey, newGeomBase, newBufferId, vertexCount });
}

bool VulkanBackendAdapter::drainDeviceResidentCCRefits() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (m_deviceResidentRefitQueue.empty()) return true;
    if (!m_device || !m_device->isInitialized() || !m_device->hasHardwareRT()) {
        m_deviceResidentRefitQueue.clear();
        return false;   // can't refit -> let the caller full-rebuild
    }

    bool allOk = true;
    bool anyRefit = false;
    for (const auto& req : m_deviceResidentRefitQueue) {
        // Locate the registered entry + its current BLAS slot.
        auto regIt = m_meshRegistry.find(req.meshKey);
        DeviceResidentCCMesh* entry = nullptr;
        for (auto& e : m_deviceResidentMeshes) if (e.meshKey == req.meshKey) { entry = &e; break; }
        if (regIt == m_meshRegistry.end() || !entry) {
            // Entry vanished (e.g. modifier removed the same frame) — don't leak the new buffer.
            if (req.newBufferId != 0) m_deviceResidentReleaseQueue.push_back(req.newBufferId);
            allOk = false; continue;
        }

        const uint32_t blasIndex = regIt->second;
        if (blasIndex >= m_device->m_blasList.size()) { allOk = false; continue; }
        auto& h = m_device->m_blasList[blasIndex];
        // Topology must match the built BLAS (vertexCount) and it must be updatable; else the
        // caller's fallback full rebuild re-creates it from the entry (updated just below).
        if (!h.allowUpdate || h.vertexCount != req.vertexCount) { allOk = false;
            // still adopt the new buffer into the entry so the full rebuild uses fresh geometry
            if (entry->bufferId != 0 && entry->bufferId != req.newBufferId)
                m_deviceResidentReleaseQueue.push_back(entry->bufferId);
            entry->bufferId = req.newBufferId; entry->deviceAddress = req.newGeomBase;
            continue;
        }

        // Repoint the BLAS geometry-buffer addresses at the freshly-evaluated combined buffer
        // (same non-indexed layout: pos | norm | uv | mat). The hit shader reads norm/uv via
        // these addresses; materials come from m_blasMaterialIds (unchanged: same topology).
        const uint64_t vertSize = (uint64_t)req.vertexCount * 12;
        const uint64_t normSize = (uint64_t)req.vertexCount * 12;
        const uint64_t uvSize   = (uint64_t)req.vertexCount * 8;
        h.vertexBuffer.deviceAddress        = req.newGeomBase;
        h.normalBuffer.deviceAddress        = req.newGeomBase + vertSize;
        h.uvBuffer.deviceAddress            = req.newGeomBase + vertSize + normSize;
        h.materialIndexBuffer.deviceAddress = req.newGeomBase + vertSize + normSize + uvSize;

        // MODE_UPDATE refit reads positions from the (now-updated) vertexBuffer.deviceAddress.
        m_device->updateBLAS(blasIndex, nullptr, nullptr);

        // Adopt the new buffer into the entry; queue the old one for release after GPU idle.
        if (entry->bufferId != 0 && entry->bufferId != req.newBufferId)
            m_deviceResidentReleaseQueue.push_back(entry->bufferId);
        entry->bufferId = req.newBufferId;
        entry->deviceAddress = req.newGeomBase;
        anyRefit = true;
    }
    m_deviceResidentRefitQueue.clear();

    if (anyRefit) {
        // Propagate the new norm/uv addresses to the shader geometry-data SSBO and refit the
        // TLAS over the updated BLAS (transforms unchanged — geometry-only edit).
        refreshVulkanGeometryDataBinding(m_device.get());
        std::vector<VulkanRT::TLASInstance> merged = m_vkInstances;
        for (const auto& hi : m_hairVkInstances) merged.push_back(hi);
        m_device->updateTLAS(merged);
        resetAccumulation();
    }

    // Free superseded / replaced buffers now that the GPU is done with the refit submits and
    // no in-flight frame still references them (we are between frames on the render thread).
    if (!m_deviceResidentReleaseQueue.empty()) {
        m_device->waitIdle();
        for (uint64_t bufId : m_deviceResidentReleaseQueue) {
            MeshModifiers::CCDeviceGeometry g; g.bufferId = bufId;
            MeshModifiers::releaseCCDeviceGeometry(g);
        }
        m_deviceResidentReleaseQueue.clear();
    }
    return allOk;
}

bool VulkanBackendAdapter::updateTerrainBLASPartial(const std::string& nodeName, const TerrainObject* terrain) {
    // Terrain is now a single flat (SoA) TriangleMesh, not a list of facade
    // Triangle objects — refit straight from its DNA SoA via the generic
    // flat-mesh refit path (registered as an indexed solo BLAS like any other
    // flat mesh). nodeName here is historically the bare terrain name; the
    // flat mesh itself is registered under "<name>_Chunk", so prefer that.
    if (!terrain) return false;
    const std::string& meshNodeName = terrain->flatMesh ? terrain->flatMesh->nodeName : nodeName;
    return refitFlatMeshBLAS(meshNodeName);
}

bool VulkanBackendAdapter::updateMeshBLASPartial(const std::string& nodeName, const std::vector<std::shared_ptr<Triangle>>& triangles) {
    if (!m_device || !m_device->isInitialized() || !m_device->hasHardwareRT()) return false;
    if (triangles.empty() || m_vkInstances.empty() || m_instanceSources.empty()) return false;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // NOTE:
    // This path is currently not used by edit/sculpt runtime sync because
    // transformed editable meshes can still desynchronize here. Keep the logic
    // intact for future debugging/rework, but the UI layer intentionally gates
    // Vulkan RT mesh edits to the full rebuild path for correctness.

    uint32_t targetBlasIndex = UINT32_MAX;
    bool targetUsesLocalSpace = false;
    const Triangle* firstTriangle = triangles.front().get();
    const size_t expectedTriangleCount = triangles.size();
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        if (!m_instanceSources[i]) continue;

        std::string instName;
        bool strongMatch = false;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            instName = inst->node_name;
            if (inst->source_triangles &&
                !inst->source_triangles->empty() &&
                inst->source_triangles->size() == expectedTriangleCount &&
                inst->source_triangles->front().get() == firstTriangle) {
                strongMatch = true;
                targetUsesLocalSpace = true;
            }
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            instName = tri->getNodeName();
            if (tri.get() == firstTriangle) {
                strongMatch = true;
                targetUsesLocalSpace = (tri->getTransformPtr() != nullptr) && !isWaterTriangleMaterial(tri);
            }
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            // Flat (direct) SoA mesh: matched by node name below. Its solo BLAS is indexed, so
            // the refit re-reads the mesh SoA directly (refitIndexedSoloBLAS) — no per-triangle
            // strong match / build-order gather is needed (and the rep-facade caller order won't
            // match). This routes sculpt on an apply-flat / import-flat mesh to an incremental
            // refit instead of a full Vulkan RT rebuild every dab.
            instName = tm->nodeName;
        }

        if (strongMatch) {
            targetBlasIndex = m_vkInstances[i].blasIndex;
            break;
        }

        if (matchesNodeNameForInstance(instName, nodeName)) {
            targetBlasIndex = m_vkInstances[i].blasIndex;
            if (std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                targetUsesLocalSpace = true;
            } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
                targetUsesLocalSpace = (tri->getTransformPtr() != nullptr) && !isWaterTriangleMaterial(tri);
            }
            break;
        }
    }

    if (targetBlasIndex == UINT32_MAX || targetBlasIndex >= m_device->m_blasList.size()) {
        return false;
    }

    // Indexed solo BLAS: refit straight from the mesh SoA (vCount unique verts), not the 3N
    // build-order gather below (whose vertexCount==triangles*3 assumption would fail anyway).
    if (m_soloBlasIndexedMesh.find(targetBlasIndex) != m_soloBlasIndexedMesh.end()) {
        return refitIndexedSoloBLAS(targetBlasIndex);
    }

    const auto& blasHandle = m_device->m_blasList[targetBlasIndex];
    const size_t expectedVertexCount = triangles.size() * 3ull;
    if (expectedVertexCount == 0 || expectedVertexCount != blasHandle.vertexCount || !blasHandle.allowUpdate) {
        return false;
    }

    // CRITICAL: upload positions in the EXACT order the BLAS was built. The caller's
    // `triangles` (from the editable mesh_cache) is not guaranteed to be in the same
    // order as the scene-graph traversal that built the solo BLAS, so we prefer the
    // recorded build-order list when available. Without this, positions land in the
    // wrong BLAS slots and the mesh corrupts. If no build-order list exists (e.g. an
    // instance-source BLAS), fall back to the caller order only when counts match.
    const std::vector<std::shared_ptr<Triangle>>* orderedTris = &triangles;
    auto buildOrderIt = m_soloBlasBuildTriangles.find(targetBlasIndex);
    if (buildOrderIt != m_soloBlasBuildTriangles.end()) {
        if (buildOrderIt->second.size() != triangles.size()) {
            // Identity mismatch (grouped/changed) — refuse, let the full rebuild run.
            return false;
        }
        orderedTris = &buildOrderIt->second;
    }

    std::vector<float> positions;
    std::vector<float> normals;
    positions.reserve(expectedVertexCount * 3ull);
    normals.reserve(expectedVertexCount * 3ull);

    TriangleMesh* lastParentMesh = nullptr;
    const Vec3* cachedPositions = nullptr;
    const Vec3* cachedNormals = nullptr;
    const Vec3* cachedOrigPositions = nullptr;
    const Vec3* cachedOrigNormals = nullptr;
    const std::vector<uint32_t, DNA::AlignedAllocator<uint32_t, 32>>* cachedIndices = nullptr;
    bool hasGeometry = false;

    for (const auto& tri : *orderedTris) {
        if (!tri) continue;
        
        Vec3 verts[3];
        Vec3 norms[3];
        bool resolved = false;

        if (tri->parentMesh) {
            if (tri->parentMesh.get() != lastParentMesh) {
                lastParentMesh = tri->parentMesh.get();
                if (lastParentMesh->geometry) {
                    cachedPositions = lastParentMesh->geometry->get_attribute_data<Vec3>("P");
                    cachedNormals = lastParentMesh->geometry->get_attribute_data<Vec3>("N");
                    cachedOrigPositions = lastParentMesh->geometry->get_attribute_data<Vec3>("P_orig");
                    cachedOrigNormals = lastParentMesh->geometry->get_attribute_data<Vec3>("N_orig");
                    cachedIndices = &lastParentMesh->geometry->indices;
                    hasGeometry = (cachedPositions != nullptr) && (cachedIndices != nullptr) && (!cachedIndices->empty());
                } else {
                    hasGeometry = false;
                }
            }

            if (hasGeometry) {
                uint32_t faceIdx = tri->faceIndex;
                uint32_t baseIdx = faceIdx * 3;
                uint32_t i0 = (*cachedIndices)[baseIdx + 0];
                uint32_t i1 = (*cachedIndices)[baseIdx + 1];
                uint32_t i2 = (*cachedIndices)[baseIdx + 2];

                if (targetUsesLocalSpace) {
                    if (tri->hasSkinData()) {
                        verts[0] = tri->getOriginalVertexPosition(0);
                        verts[1] = tri->getOriginalVertexPosition(1);
                        verts[2] = tri->getOriginalVertexPosition(2);
                    } else {
                        verts[0] = cachedOrigPositions ? cachedOrigPositions[i0] : cachedPositions[i0];
                        verts[1] = cachedOrigPositions ? cachedOrigPositions[i1] : cachedPositions[i1];
                        verts[2] = cachedOrigPositions ? cachedOrigPositions[i2] : cachedPositions[i2];
                    }
                    norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                    norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                    norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                } else {
                    verts[0] = cachedPositions[i0];
                    verts[1] = cachedPositions[i1];
                    verts[2] = cachedPositions[i2];
                    norms[0] = cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0);
                    norms[1] = cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0);
                    norms[2] = cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0);
                }
                resolved = true;
            }
        }

        if (!resolved) {
            for (int v = 0; v < 3; ++v) {
                verts[v] = targetUsesLocalSpace ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                norms[v] = targetUsesLocalSpace ? tri->getOriginalVertexNormal(v) : tri->getVertexNormal(v);
            }
        }

        for (int v = 0; v < 3; ++v) {
            positions.push_back(verts[v].x);
            positions.push_back(verts[v].y);
            positions.push_back(verts[v].z);
            normals.push_back(norms[v].x);
            normals.push_back(norms[v].y);
            normals.push_back(norms[v].z);
        }
    }

    if (positions.size() != expectedVertexCount * 3ull || normals.size() != expectedVertexCount * 3ull) {
        return false;
    }

    m_device->updateBLAS(targetBlasIndex, positions.data(), normals.data());

    // Refresh the TLAS instance transform from the live transform handle so that
    // objects scaled/rotated since the last full rebuild are rendered at the
    // correct world-space position.  Without this, editing vertices of a
    // recently-scaled object would show the BLAS data through a stale TLAS
    // transform, producing oversized / misplaced geometry.
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        if (m_vkInstances[i].blasIndex != targetBlasIndex) continue;

        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            m_vkInstances[i].transform = inst->transform;
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            if (tri->getTransformPtr()) {
                m_vkInstances[i].transform = tri->getTransformPtr()->getFinal();
            }
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            if (tm->transform) m_vkInstances[i].transform = tm->transform->getFinal(); // flat mesh
        }
    }

    auto merged = m_vkInstances;
    for (const auto& h : m_hairVkInstances) merged.push_back(h);
    if (!merged.empty()) {
        m_device->updateTLAS(merged);
    }

    resetAccumulation();
    return true;
}

int64_t VulkanBackendAdapter::updateMeshBLASPartial(
    const std::string& nodeName,
    const std::vector<size_t>& dirtyIndices,
    const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries) {

    if (!m_device || !m_device->isInitialized() || !m_device->hasHardwareRT()) return -1;
    if (dirtyIndices.empty() || meshEntries.empty() || m_vkInstances.empty() || m_instanceSources.empty()) return -1;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    uint32_t targetBlasIndex = UINT32_MAX;
    bool targetUsesLocalSpace = false;
    const Triangle* firstTriangle = nullptr;
    for (const auto& entry : meshEntries) {
        if (entry.second) {
            firstTriangle = entry.second.get();
            break;
        }
    }
    const size_t expectedTriangleCount = meshEntries.size();
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        if (!m_instanceSources[i]) continue;

        std::string instName;
        bool strongMatch = false;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            instName = inst->node_name;
            if (inst->source_triangles &&
                !inst->source_triangles->empty() &&
                inst->source_triangles->size() == expectedTriangleCount &&
                inst->source_triangles->front().get() == firstTriangle) {
                strongMatch = true;
                targetUsesLocalSpace = true;
            }
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            instName = tri->getNodeName();
            if (tri.get() == firstTriangle) {
                strongMatch = true;
                targetUsesLocalSpace = (tri->getTransformPtr() != nullptr) && !isWaterTriangleMaterial(tri);
            }
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            // Flat (direct) SoA mesh: matched by node name below. Its solo BLAS is indexed, so
            // the refit re-reads the mesh SoA directly (refitIndexedSoloBLAS) — no per-triangle
            // strong match / build-order gather is needed (and the rep-facade caller order won't
            // match). This routes sculpt on an apply-flat / import-flat mesh to an incremental
            // refit instead of a full Vulkan RT rebuild every dab.
            instName = tm->nodeName;
        }

        if (strongMatch) {
            targetBlasIndex = m_vkInstances[i].blasIndex;
            break;
        }

        if (matchesNodeNameForInstance(instName, nodeName)) {
            targetBlasIndex = m_vkInstances[i].blasIndex;
            if (std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                targetUsesLocalSpace = true;
            } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
                targetUsesLocalSpace = (tri->getTransformPtr() != nullptr) && !isWaterTriangleMaterial(tri);
            }
            break;
        }
    }

    if (targetBlasIndex == UINT32_MAX || targetBlasIndex >= m_device->m_blasList.size()) {
        return -1;
    }

    // Indexed solo BLAS has no 3N build-order slot map. Phase 1: decline the dirty-region
    // partial so the caller falls back to the (indexed-aware) full SoA refit. A dirty-region
    // indexed partial is Phase 2.
    if (m_soloBlasIndexedMesh.find(targetBlasIndex) != m_soloBlasIndexedMesh.end()) {
        return -1;
    }

    const auto& blasHandle = m_device->m_blasList[targetBlasIndex];
    if (blasHandle.vertexCount == 0 || !blasHandle.allowUpdate) {
        return -1;
    }

    auto buildOrderIt = m_soloBlasBuildTriangles.find(targetBlasIndex);
    if (buildOrderIt == m_soloBlasBuildTriangles.end()) {
        return -1;
    }
    const auto& orderedTris = buildOrderIt->second;

    auto slotIt = m_soloBlasBuildTriangleSlots.find(targetBlasIndex);
    if (slotIt == m_soloBlasBuildTriangleSlots.end()) {
        return -1;
    }
    const auto& slots = slotIt->second;

    if (orderedTris.size() != meshEntries.size()) {
        return -1;
    }

    uint32_t minSlot = UINT32_MAX;
    uint32_t maxSlot = 0;

    for (const size_t triIdx : dirtyIndices) {
        if (triIdx >= meshEntries.size()) continue;
        const auto& tri = meshEntries[triIdx].second;
        if (!tri) continue;

        auto it = slots.find(tri.get());
        if (it != slots.end()) {
            uint32_t slot = it->second;
            if (slot < minSlot) minSlot = slot;
            if (slot > maxSlot) maxSlot = slot;
        }
    }

    if (minSlot > maxSlot) {
        return 0; // No valid dirty slots found, 0 bytes uploaded
    }

    uint32_t numSlots = maxSlot - minSlot + 1;
    uint32_t numVertices = numSlots * 3;

    std::vector<float> rangePositions;
    std::vector<float> rangeNormals;
    rangePositions.reserve(numVertices * 3);
    rangeNormals.reserve(numVertices * 3);

    TriangleMesh* lastParentMesh = nullptr;
    const Vec3* cachedPositions = nullptr;
    const Vec3* cachedNormals = nullptr;
    const Vec3* cachedOrigPositions = nullptr;
    const Vec3* cachedOrigNormals = nullptr;
    const std::vector<uint32_t, DNA::AlignedAllocator<uint32_t, 32>>* cachedIndices = nullptr;
    bool hasGeometry = false;

    for (uint32_t slot = minSlot; slot <= maxSlot; ++slot) {
        if (slot >= orderedTris.size()) break;
        const auto& tri = orderedTris[slot];
        if (!tri) {
            for (int v = 0; v < 9; ++v) {
                rangePositions.push_back(0.0f);
                rangeNormals.push_back(0.0f);
            }
            continue;
        }

        Vec3 verts[3];
        Vec3 norms[3];
        bool resolved = false;

        if (tri->parentMesh) {
            if (tri->parentMesh.get() != lastParentMesh) {
                lastParentMesh = tri->parentMesh.get();
                if (lastParentMesh->geometry) {
                    cachedPositions = lastParentMesh->geometry->get_attribute_data<Vec3>("P");
                    cachedNormals = lastParentMesh->geometry->get_attribute_data<Vec3>("N");
                    cachedOrigPositions = lastParentMesh->geometry->get_attribute_data<Vec3>("P_orig");
                    cachedOrigNormals = lastParentMesh->geometry->get_attribute_data<Vec3>("N_orig");
                    cachedIndices = &lastParentMesh->geometry->indices;
                    hasGeometry = (cachedPositions != nullptr) && (cachedIndices != nullptr) && (!cachedIndices->empty());
                } else {
                    hasGeometry = false;
                }
            }

            if (hasGeometry) {
                uint32_t faceIdx = tri->faceIndex;
                uint32_t baseIdx = faceIdx * 3;
                uint32_t i0 = (*cachedIndices)[baseIdx + 0];
                uint32_t i1 = (*cachedIndices)[baseIdx + 1];
                uint32_t i2 = (*cachedIndices)[baseIdx + 2];

                if (targetUsesLocalSpace) {
                    if (tri->hasSkinData()) {
                        verts[0] = tri->getOriginalVertexPosition(0);
                        verts[1] = tri->getOriginalVertexPosition(1);
                        verts[2] = tri->getOriginalVertexPosition(2);
                    } else {
                        verts[0] = cachedOrigPositions ? cachedOrigPositions[i0] : cachedPositions[i0];
                        verts[1] = cachedOrigPositions ? cachedOrigPositions[i1] : cachedPositions[i1];
                        verts[2] = cachedOrigPositions ? cachedOrigPositions[i2] : cachedPositions[i2];
                    }
                    norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                    norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                    norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                } else {
                    verts[0] = cachedPositions[i0];
                    verts[1] = cachedPositions[i1];
                    verts[2] = cachedPositions[i2];
                    norms[0] = cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0);
                    norms[1] = cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0);
                    norms[2] = cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0);
                }
                resolved = true;
            }
        }

        if (!resolved) {
            for (int v = 0; v < 3; ++v) {
                verts[v] = targetUsesLocalSpace ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                norms[v] = targetUsesLocalSpace ? tri->getOriginalVertexNormal(v) : tri->getVertexNormal(v);
            }
        }

        for (int v = 0; v < 3; ++v) {
            rangePositions.push_back(verts[v].x);
            rangePositions.push_back(verts[v].y);
            rangePositions.push_back(verts[v].z);
            rangeNormals.push_back(norms[v].x);
            rangeNormals.push_back(norms[v].y);
            rangeNormals.push_back(norms[v].z);
        }
    }

    const uint64_t positionByteOffset = (uint64_t)minSlot * 3 * 12;
    const uint64_t positionByteSize = (uint64_t)numVertices * 12;

    m_device->uploadBuffer(blasHandle.vertexBuffer, rangePositions.data(), positionByteSize, positionByteOffset);

    uint64_t totalUploadedBytes = positionByteSize;

    if (blasHandle.normalBuffer.buffer) {
        const bool normalSharesGeometryBuffer =
            blasHandle.normalBuffer.buffer == blasHandle.vertexBuffer.buffer &&
            blasHandle.normalBuffer.deviceAddress >= blasHandle.vertexBuffer.deviceAddress;
        
        const uint64_t normalBaseOffset = normalSharesGeometryBuffer
            ? (uint64_t)(blasHandle.normalBuffer.deviceAddress - blasHandle.vertexBuffer.deviceAddress)
            : 0ull;
        
        const uint64_t normalByteOffset = normalBaseOffset + (uint64_t)minSlot * 3 * 12;
        const uint64_t normalByteSize = (uint64_t)numVertices * 12;

        m_device->uploadBuffer(
            normalSharesGeometryBuffer ? blasHandle.vertexBuffer : blasHandle.normalBuffer,
            rangeNormals.data(),
            normalByteSize,
            normalByteOffset);

        totalUploadedBytes += normalByteSize;
    }

    // Trigger local refit/update of the bottom-level acceleration structure (BLAS)
    // with vertices/normals arrays set to nullptr (to skip full buffers upload)
    m_device->updateBLAS(targetBlasIndex, nullptr, nullptr);

    // Refresh the TLAS instance transforms
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        if (m_vkInstances[i].blasIndex != targetBlasIndex) continue;

        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            m_vkInstances[i].transform = inst->transform;
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            if (tri->getTransformPtr()) {
                m_vkInstances[i].transform = tri->getTransformPtr()->getFinal();
            }
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            if (tm->transform) m_vkInstances[i].transform = tm->transform->getFinal(); // flat mesh
        }
    }

    auto merged = m_vkInstances;
    for (const auto& h : m_hairVkInstances) merged.push_back(h);
    if (!merged.empty()) {
        m_device->updateTLAS(merged);
    }

    resetAccumulation();
    return (int64_t)totalUploadedBytes;
}

bool VulkanBackendAdapter::updateInteractiveMesh(const std::string& nodeName,
                                                 const std::vector<std::shared_ptr<Triangle>>& triangles) {
    return updateMeshBLASPartial(nodeName, triangles);
}

bool VulkanBackendAdapter::refitFlatMeshBLAS(const std::string& nodeName) {
    if (!m_device || !m_device->isInitialized() || !m_device->hasHardwareRT()) {
        SCENE_LOG_WARN("[refitFlatMeshBLAS] Hardware RT or device not initialized");
        return false;
    }
    if (m_vkInstances.empty() || m_instanceSources.empty()) {
        SCENE_LOG_WARN("[refitFlatMeshBLAS] m_vkInstances or m_instanceSources empty");
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i]);
        if (!tm || !matchesNodeNameForInstance(tm->nodeName, nodeName)) continue;
        const uint32_t blasIndex = m_vkInstances[i].blasIndex;
        if (blasIndex >= m_device->m_blasList.size()) {
            SCENE_LOG_WARN("[refitFlatMeshBLAS] blasIndex out of bounds: " + std::to_string(blasIndex));
            return false;
        }
        if (m_soloBlasIndexedMesh.find(blasIndex) == m_soloBlasIndexedMesh.end()) {
            SCENE_LOG_WARN("[refitFlatMeshBLAS] blasIndex not found in m_soloBlasIndexedMesh: " + std::to_string(blasIndex));
            return false;
        }
        SCENE_LOG_INFO("[refitFlatMeshBLAS] Routing to refitIndexedSoloBLAS for node: " + nodeName);
        return refitIndexedSoloBLAS(blasIndex);
    }
    SCENE_LOG_WARN("[refitFlatMeshBLAS] No matching TriangleMesh instance found for name: " + nodeName);
    return false;
}

void VulkanBackendAdapter::clearHairGeometry(bool rebuild_tlas) {
    // Destroy all hair BLASes that were appended after the mesh BLASes.
    // Hair BLASes always live at m_blasList[m_meshBlasCount .. end).
    if (m_device && m_device->isInitialized()) {
        // Wait for the GPU to finish any in-flight work that might be using these BLASes.
        vkDeviceWaitIdle(m_device->m_device);

        const uint32_t blasCount = (uint32_t)m_device->m_blasList.size();
        const uint32_t firstHairBlas = (m_meshBlasCount <= blasCount) ? m_meshBlasCount : blasCount;

        for (uint32_t i = firstHairBlas; i < blasCount; ++i) {
            auto& blas = m_device->m_blasList[i];
            if (blas.accel && m_device->fpDestroyAccelerationStructureKHR) {
                m_device->fpDestroyAccelerationStructureKHR(m_device->m_device, blas.accel, nullptr);
            }
            m_device->destroyBuffer(blas.buffer);
            m_device->destroyBuffer(blas.vertexBuffer); // stores the AABB buffer for hair
        }
        if (blasCount > firstHairBlas) {
            m_device->m_blasList.resize(firstHairBlas);
        }

        // Always invalidate geometry data after a hair clear attempt. Hair can
        // transition 0->N or N->0 between frames, and leaving a mesh-only buffer
        // alive across a later hair re-upload can make BLAS indexing stale.
        if (m_device->m_geometryDataBuffer.buffer) {
            m_device->destroyBuffer(m_device->m_geometryDataBuffer);
        }
    }
    m_hairVkInstances.clear();
    m_hairGroomRegistry.clear();

    // Rebuild TLAS immediately only when requested. During full hair re-upload
    // we skip this intermediate rebuild to avoid doing TLAS twice.
    if (rebuild_tlas && m_device && m_device->isInitialized()) {
        // Re-calculate combining MESHES ONLY (since m_hairVkInstances was cleared above).
        std::vector<VulkanRT::TLASInstance> allInstances = m_vkInstances;
        
        if (!allInstances.empty()) {
            VulkanRT::TLASCreateInfo tlasInfo;
            tlasInfo.instances   = allInstances;
            tlasInfo.allowUpdate = false; // full rebuild when topology changes
            m_device->createTLAS(tlasInfo);
        } else {
            // If the scene is completely empty, potentially clear TLAS and disable RT.
            VulkanRT::TLASCreateInfo tlasInfo; // empty
            m_device->createTLAS(tlasInfo);
            m_device->m_rtPipelineReady = false; 
        }
        resetAccumulation();
    }
}

uint32_t VulkanBackendAdapter::uploadHairStrands(const std::vector<HairStrandData>& strands, const std::string& groomName) {
    if (!m_device || !m_device->isInitialized() || strands.empty()) return UINT32_MAX;
    if (!m_device->hasHardwareRT()) return UINT32_MAX;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Vulkan hair currently uses a single combined upload. If an older code path
    // accidentally tries to append another groom without a prior clear, recover
    // by clearing stale hair state instead of keeping multiple incompatible SSBO/BLAS layouts alive.
    if (!m_hairVkInstances.empty() || !m_hairGroomRegistry.empty()) {
        SCENE_LOG_WARN("[Vulkan] uploadHairStrands called while hair state already exists. Forcing hair clear before re-upload.");
        clearHairGeometry(false);
    }
    if (groomName != "combined_hair") {
        SCENE_LOG_WARN("[Vulkan] uploadHairStrands expected combined_hair but got \"" + groomName + "\". Proceeding in combined mode.");
    }

    // Convert all strands into B-spline segments: N points → N-3 cubic segments
    std::vector<VulkanRT::HairSegmentGPU> segments;
    std::vector<VkAabbPositionsKHR>       aabbs;
    uint32_t strandIdx = 0;
    uint32_t maxMaterialID = 0;

    for (const auto& strand : strands) {
        const auto& pts = strand.points;
        const int N = (int)pts.size();
        if (N < 4) { ++strandIdx; continue; }

        // Default radius when per-point radii are absent
        const float defaultR = 0.002f;

        for (int i = 0; i < N - 3; i++) {
            VulkanRT::HairSegmentGPU seg{};
            // Control points (vec4: xyz=position, w=radius)
            auto getR = [&](int idx) {
                if (idx < (int)strand.radii.size()) return strand.radii[idx];
                return defaultR;
            };
            seg.cp0[0] = pts[i    ].x; seg.cp0[1] = pts[i    ].y; seg.cp0[2] = pts[i    ].z; seg.cp0[3] = getR(i);
            seg.cp1[0] = pts[i + 1].x; seg.cp1[1] = pts[i + 1].y; seg.cp1[2] = pts[i + 1].z; seg.cp1[3] = getR(i + 1);
            seg.cp2[0] = pts[i + 2].x; seg.cp2[1] = pts[i + 2].y; seg.cp2[2] = pts[i + 2].z; seg.cp2[3] = getR(i + 2);
            seg.cp3[0] = pts[i + 3].x; seg.cp3[1] = pts[i + 3].y; seg.cp3[2] = pts[i + 3].z; seg.cp3[3] = getR(i + 3);
            seg.strandID   = strandIdx;
            seg.groomID    = strand.materialID;
            seg.materialID = strand.materialID;
            seg.padding    = 0;
            maxMaterialID  = (std::max)(maxMaterialID, static_cast<uint32_t>(strand.materialID));

            // LSS-tight AABB: hair_intersection.rint now treats each segment as a
            // linear-swept-sphere between the cubic B-spline endpoints curve(u=0) and
            // curve(u=1). The body lies entirely inside the union of the two endpoint
            // spheres' bounding boxes — much tighter than the old "convex hull of 4
            // control points ± max radius" approach, which over-estimated by ~3× in
            // the curve direction because cp0 and cp3 sit far from the actual segment.
            // Tighter AABBs reduce false-positive .rint invocations on close-up hair.
            //
            // Uniform cubic B-spline values at endpoints:
            //   curve(0) = (cp0 + 4*cp1 + cp2) / 6,  curve(1) = (cp1 + 4*cp2 + cp3) / 6
            // Same formulas applied to radii.
            const float inv6 = 1.0f / 6.0f;
            const float p0x = (seg.cp0[0] + 4.0f * seg.cp1[0] + seg.cp2[0]) * inv6;
            const float p0y = (seg.cp0[1] + 4.0f * seg.cp1[1] + seg.cp2[1]) * inv6;
            const float p0z = (seg.cp0[2] + 4.0f * seg.cp1[2] + seg.cp2[2]) * inv6;
            const float r0  = (seg.cp0[3] + 4.0f * seg.cp1[3] + seg.cp2[3]) * inv6;
            const float p1x = (seg.cp1[0] + 4.0f * seg.cp2[0] + seg.cp3[0]) * inv6;
            const float p1y = (seg.cp1[1] + 4.0f * seg.cp2[1] + seg.cp3[1]) * inv6;
            const float p1z = (seg.cp1[2] + 4.0f * seg.cp2[2] + seg.cp3[2]) * inv6;
            const float r1  = (seg.cp1[3] + 4.0f * seg.cp2[3] + seg.cp3[3]) * inv6;

            const float minX = std::min(p0x - r0, p1x - r1);
            const float minY = std::min(p0y - r0, p1y - r1);
            const float minZ = std::min(p0z - r0, p1z - r1);
            const float maxX = std::max(p0x + r0, p1x + r1);
            const float maxY = std::max(p0y + r0, p1y + r1);
            const float maxZ = std::max(p0z + r0, p1z + r1);

            VkAabbPositionsKHR aabb{};
            aabb.minX = minX; aabb.minY = minY; aabb.minZ = minZ;
            aabb.maxX = maxX; aabb.maxY = maxY; aabb.maxZ = maxZ;

            segments.push_back(seg);
            aabbs.push_back(aabb);
        }
        ++strandIdx;
    }

    if (segments.empty()) return UINT32_MAX;

    // Upload segment SSBO (append to existing)
    // TODO: support multi-groom by appending; for now replace entire buffer
    m_device->updateHairSegmentBuffer(segments);

    // Build AABB BLAS
    uint32_t blasIdx = m_device->createHairAABB_BLAS(aabbs);
    if (blasIdx == UINT32_MAX) return UINT32_MAX;

    // Register as a TLAS instance with the hair SBT offset
    VulkanRT::TLASInstance vi;
    vi.blasIndex       = blasIdx;
    vi.transform       = Matrix4x4(); // identity — hair is in world space
    vi.materialIndex   = 0; // materialID is stored per-segment
    vi.customIndex     = (uint32_t)m_hairVkInstances.size();
    vi.mask            = 0xFF;
    vi.frontFaceCCW    = false;
    vi.sbtRecordOffset = m_device->getHairSbtOffset();
    m_hairVkInstances.push_back(vi);

    // Issue TLAS rebuild (merge mesh + hair instances)
    std::vector<VulkanRT::TLASInstance> allInstances = m_vkInstances;
    for (const auto& h : m_hairVkInstances) allInstances.push_back(h);

    if (!allInstances.empty()) {
        VulkanRT::TLASCreateInfo tlasInfo;
        tlasInfo.instances   = allInstances;
        tlasInfo.allowUpdate = false; // full rebuild when topology changes
        m_device->createTLAS(tlasInfo);
    } else {
        // [VULKAN FIX] Even if empty, we MUST clear the TLAS and stop dispatcher
        VulkanRT::TLASCreateInfo tlasInfo; 
        m_device->createTLAS(tlasInfo);
        m_device->m_rtPipelineReady = false;
    }
    resetAccumulation();

    uint32_t groomHandle = (uint32_t)(m_hairVkInstances.size() - 1);
    m_hairGroomRegistry.clear();
    m_hairGroomRegistry["combined_hair"] = groomHandle;

    SCENE_LOG_INFO("[Vulkan] Hair upload material range: 0.." + std::to_string(maxMaterialID));

    SCENE_LOG_INFO("[Vulkan] Hair groom \"" + groomName + "\" uploaded: "
        + std::to_string(strands.size()) + " strands, "
        + std::to_string(segments.size()) + " segments (BLAS=" + std::to_string(blasIdx) + ")");
    return groomHandle;
}
void VulkanBackendAdapter::updateMeshTransform(uint32_t h, const Matrix4x4& t) { (void)h; (void)t; }

void VulkanBackendAdapter::rebuildAccelerationStructure() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    SCENE_LOG_INFO("[Vulkan] Full scene/project rebuild triggered.");
    m_meshRegistry.clear();
    m_blasMaterialBounds.clear();
    m_blasBuiltNonOpaque.clear();
    m_blasMaterialIds.clear();
    m_vkInstances.clear();
    m_instanceSources.clear();
    m_instance_sync_cache.clear();
    m_hairVkInstances.clear();
    m_hairGroomRegistry.clear();
    m_meshBlasCount = 0; // Reset: all existing BLASes will be destroyed below
    m_topology_dirty = true;
    m_rasterGeometryDirty = true;
    destroyAllRasterMeshes();
    m_rasterBuiltGeometryGeneration = 0; // Invalidate raster cache
    m_envTexID = 0;
    m_atmosphereLutReady = false;
    
    if (m_device) {
        m_device->waitIdle();
        
        // Destroy all existing BLAS (Geometry)
        for (auto& blas : m_device->m_blasList) {
            if (blas.accel && m_device->fpDestroyAccelerationStructureKHR) {
                m_device->fpDestroyAccelerationStructureKHR(m_device->m_device, blas.accel, nullptr);
            }
            // [FIX] In createBLAS all geometry data is packed into one combined buffer that is
            // stored in vertexBuffer.  normalBuffer / uvBuffer / indexBuffer / materialIndexBuffer
            // are NOT separate allocations — they are aliased views of the same VkBuffer with
            // different deviceAddress offsets.  Calling destroyBuffer on each of them would
            // double-free the same VkBuffer and VkDeviceMemory, causing a read-access-violation
            // inside vkDestroyBuffer.  Solution: zero-out the aliases first so that destroyBuffer
            // sees VK_NULL_HANDLE and skips the Vulkan call, then destroy vertexBuffer once.
            blas.normalBuffer        = {};
            blas.uvBuffer            = {};
            blas.indexBuffer         = {};
            blas.materialIndexBuffer = {};
            m_device->destroyBuffer(blas.buffer);       // dedicated AS backing buffer
            m_device->destroyBuffer(blas.vertexBuffer); // single combined geometry buffer
            m_device->destroyBuffer(blas.baseVertexBuffer);
            m_device->destroyBuffer(blas.baseNormalBuffer);
            m_device->destroyBuffer(blas.boneIndexBuffer);
            m_device->destroyBuffer(blas.boneWeightBuffer);
        }
        m_device->m_blasList.clear();

        // Phase 3d: every BLAS is now destroyed and the GPU is idle (waitIdle above), so
        // it is finally safe to free device-resident CC buffers queued for release — the
        // closest-hit shader read their normal/uv/mat blocks through the (now-gone) BLAS.
        for (uint64_t bufId : m_deviceResidentReleaseQueue) {
            MeshModifiers::CCDeviceGeometry g; g.bufferId = bufId;
            MeshModifiers::releaseCCDeviceGeometry(g);
        }
        m_deviceResidentReleaseQueue.clear();

        if (m_device->m_geometryDataBuffer.buffer) m_device->destroyBuffer(m_device->m_geometryDataBuffer);

        // Destroy existing TLAS
        if (m_device->m_tlas.accel && m_device->fpDestroyAccelerationStructureKHR) {
             m_device->fpDestroyAccelerationStructureKHR(m_device->m_device, m_device->m_tlas.accel, nullptr);
             m_device->m_tlas.accel = VK_NULL_HANDLE;
        }
        m_device->destroyBuffer(m_device->m_tlas.buffer);

        // purgeUploadedTextureCacheLocked tears down SceneTextureManager destroyFns
        // BEFORE freeing VkImages, then invalidates the alive sentinel, then cleans
        // any remaining non-manager-tracked images via the fallback loop.
        // The old manual destroy loop here skipped all of that: SceneTextureManager
        // still held destroyFns capturing the (now-freed) handles, so the next
        // purgeUploadedTextureCacheLocked call (e.g. on material-preview → Vulkan RT
        // transition) would call destroyImage on stale pointers → access violation.
        purgeUploadedTextureCacheLocked();

        // Clear cached NanoVDB device buffers from previous scene/project.
        for (auto& [id, buf] : m_vdbBuffers) {
            (void)id;
            if (buf.buffer) m_device->destroyBuffer(buf);
        }
        m_vdbBuffers.clear();
        for (auto& [id, buf] : m_vdbTempBuffers) {
            (void)id;
            if (buf.buffer) m_device->destroyBuffer(buf);
        }
        m_vdbTempBuffers.clear();
        m_orderedVDBInstances.clear();
        m_volumeBlasIndex = UINT32_MAX;
        // Foam sphere BLAS lived in m_blasList (just cleared above) — drop the stale
        // index so the next updateGeometry() rebuilds it fresh and the motion hook
        // doesn't refit a freed slot.
        m_foamSphereBlasIndex   = UINT32_MAX;
        m_foamSphereGroupId     = -1;
        m_foamSpherePoolCapacity = 0;

        // [CRASH GUARD] Disable ray tracing until updateGeometry() rebuilds a valid TLAS.
        // traceRays() checks m_tlas.accel (Fix 1) but also gate m_rtPipelineReady so legacy
        // callers that skip the TLAS check cannot accidentally dispatch against a dead TLAS.
        m_device->m_rtPipelineReady = false;
    }
    
    m_testInitialized = false; 
    resetAccumulation();
    
    // NOTE: We DO NOT call updateGeometry(m_lastObjects) here anymore.
    // The caller is expected to call updateGeometry() with the LATEST scene objects
    // to avoid race conditions with deleted items.
}

void VulkanBackendAdapter::showAllInstances() {}

bool VulkanBackendAdapter::updateFlatMeshBLAS(const std::string& nodeName, const TriangleMesh* mesh) {
    (void)mesh;
    return refitFlatMeshBLAS(nodeName);
}

void VulkanBackendAdapter::updateSceneGeometry(const std::vector<std::shared_ptr<Hittable>>& o, const std::vector<Matrix4x4>& b) { 
    if (!m_device || !m_device->isInitialized()) return;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // ── Fast path: skinned BLAS refit + lightweight TLAS update ──────────────────
    // Only run the full updateGeometry (scene rebuild + waitIdle) on the very first
    // frame or when topology changes (m_vkInstances empty / topology_dirty).
    // All other frames: compute-dispatch skinning and refit TLAS in-place.
    if (!m_vkInstances.empty() && !m_topology_dirty) {
        // 1. Refit each skinned BLAS (compute + AS update in one command buffer)
        if (!b.empty()) {
            for (uint32_t i = 0; i < (uint32_t)m_device->m_blasList.size(); ++i) {
                if (m_device->m_blasList[i].hasSkinning) {
                    m_device->dispatchSkinning(i, b);
                }
            }
        }

        // 2. Refit TLAS with current instance list
        // [HANG DIAGNOSIS] When the skin path runs, dispatchSkinning above
        // submits compute work that internally drains via fence — so by the
        // time we reach here the GPU is idle and updateTLAS is safe. On the
        // rigid (non-skinned) keyframe path b is empty, dispatchSkinning is
        // skipped, and updateTLAS proceeds while the previous frame's
        // traceRays may still be in flight. Modifying an acceleration
        // structure while it is being traced is undefined behavior in
        // Vulkan — the typical NVIDIA driver symptom is a silent hang
        // (no exception, no AV) inside subsequent traceRays, exactly
        // matching the sequence-render-stops-at-frame-N report.
        // waitIdle here is a heavy hammer for verification; once confirmed,
        // a per-frame fence on the RT submission would be the targeted fix.
        if (b.empty() && m_device) {
            m_device->waitIdle();
        }
        auto merged = m_vkInstances;
        for (const auto& h : m_hairVkInstances) merged.push_back(h);
        m_device->updateTLAS(merged);

        // Sync raster viewport instances so Solid/Matcap mode reflects animation
        if (!m_rasterInstances.empty() && shouldUseInteractiveViewport()) {
            syncRasterInstanceTransforms(o);
            // CPU-skin raster vertex buffers so skinned meshes don't stay in T-pose
            if (!b.empty()) {
                syncRasterSkinnedVertices(o, b);
            }
        }

        resetAccumulation();
        return;
    }

    // ── Slow path: first frame or topology changed — full scene rebuild ───────────
    // Dispatch skinning first so refitted BLASes are ready for the new TLAS
    if (!b.empty()) {
        for (uint32_t i = 0; i < (uint32_t)m_device->m_blasList.size(); ++i) {
            if (m_device->m_blasList[i].hasSkinning) {
                m_device->dispatchSkinning(i, b);
            }
        }
    }
    updateGeometry(o);
    m_topology_dirty = false;

    // Sync raster viewport instances so Solid/Matcap mode reflects animation
    if (!m_rasterInstances.empty() && shouldUseInteractiveViewport()) {
        syncRasterInstanceTransforms(o);
        // CPU-skin raster vertex buffers so skinned meshes don't stay in T-pose
        if (!b.empty()) {
            syncRasterSkinnedVertices(o, b);
        }
    }
}

void VulkanBackendAdapter::updateInstanceMaterialBinding(const std::string& nodeName, int oldMatID, int newMatID) {
    if (!m_device || !m_device->isInitialized()) return;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    bool changed = false;
    std::unordered_set<std::string> dirtyMeshKeys;

    for (const auto& ri : m_rasterInstances) {
        if (matchesNodeNameForInstance(ri.nodeName, nodeName) ||
            matchesNodeNameForInstance(nodeName, ri.nodeName)) {
            dirtyMeshKeys.insert(ri.meshKey);
        }
    }

    for (const auto& meshKey : dirtyMeshKeys) {
        auto meshIt = m_rasterMeshes.find(meshKey);
        if (meshIt == m_rasterMeshes.end()) continue;

        auto& mesh = meshIt->second;
        if (!mesh.matIdBuffer.buffer || mesh.cpuMatIds.empty()) continue;

        bool meshChanged = false;
        for (auto& matId : mesh.cpuMatIds) {
            if (oldMatID < 0 || static_cast<int>(matId) == oldMatID) {
                matId = static_cast<uint32_t>(newMatID);
                meshChanged = true;
            }
        }

        if (!meshChanged) continue;

        m_device->uploadBuffer(mesh.matIdBuffer,
                               mesh.cpuMatIds.data(),
                               mesh.cpuMatIds.size() * sizeof(uint32_t),
                               0);
        changed = true;
    }

    // ---- RT path -------------------------------------------------------
    // The path tracer reads materials from two places the raster matId update
    // above never touches:
    //   1) per-triangle material indices baked into each BLAS geometry buffer
    //      (geo.materialAddr — preferred by closesthit/anyhit when non-zero)
    //   2) per-instance materialIndex in the instance SSBO (binding 5)
    // Without rewriting these, a material *assignment* change only appears
    // after a full backend rebuild (parameter edits worked all along because
    // they rewrite the material SSBO at the unchanged index).
    bool instanceDataChanged = false;
    std::unordered_set<uint32_t> candidateBlasIndices;
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        if (!m_instanceSources[i]) continue;

        std::string instName;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            instName = inst->node_name;
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            instName = tri->getNodeName();
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            // Flat (direct SoA) mesh: no facade, the node name lives on the
            // TriangleMesh itself (same fix as setVisibilityByNodeName below).
            // Without this branch a flat object's RT instance never matched, so
            // ASSIGNING a different material only took effect after a full
            // backend rebuild — while parameter edits worked, since those
            // rewrite the material SSBO at the unchanged index.
            instName = tm->nodeName;
        }
        if (instName.empty()) continue;
        if (!matchesNodeNameForInstance(instName, nodeName) &&
            !matchesNodeNameForInstance(nodeName, instName)) continue;

        auto& vi = m_vkInstances[i];
        if ((oldMatID < 0 || static_cast<int>(vi.materialIndex) == oldMatID) &&
            static_cast<int>(vi.materialIndex) != newMatID) {
            vi.materialIndex = static_cast<uint32_t>(newMatID);
            instanceDataChanged = true;
        }
        candidateBlasIndices.insert(vi.blasIndex);
    }

    std::vector<uint32_t> blasToUpload;
    for (uint32_t blasIndex : candidateBlasIndices) {
        if (blasIndex >= m_device->m_blasList.size()) continue;
        auto mirrorIt = m_blasMaterialIds.find(blasIndex);
        if (mirrorIt == m_blasMaterialIds.end() || mirrorIt->second.empty()) continue;

        bool blasChanged = false;
        for (auto& matId : mirrorIt->second) {
            if ((oldMatID < 0 || static_cast<int>(matId) == oldMatID) &&
                static_cast<int>(matId) != newMatID) {
                matId = static_cast<uint32_t>(newMatID);
                blasChanged = true;
            }
        }
        if (blasChanged) blasToUpload.push_back(blasIndex);
    }

    if (instanceDataChanged || !blasToUpload.empty()) {
        // Serialize with any in-flight trace before rewriting buffers it reads.
        m_device->waitIdle();

        for (uint32_t blasIndex : blasToUpload) {
            const auto& blas = m_device->m_blasList[blasIndex];
            if (!blas.materialIndexBuffer.buffer || !blas.vertexBuffer.buffer) continue;
            const auto& ids = m_blasMaterialIds[blasIndex];
            // materialIndexBuffer aliases the combined geometry buffer; its
            // byte offset is the device-address delta from the buffer start.
            const uint64_t byteOffset =
                blas.materialIndexBuffer.deviceAddress - blas.vertexBuffer.deviceAddress;
            m_device->uploadBuffer(blas.materialIndexBuffer,
                                   ids.data(),
                                   ids.size() * sizeof(uint32_t),
                                   byteOffset);
        }

        // Keep the per-material caustic bounds keyed to the NEW id, and flip
        // the per-instance opacity override if the assignment changed the
        // mesh's opaque<->transmissive class (assigning a glass material to an
        // opaque-built mesh must wake the shadow any-hit without a BLAS rebuild).
        bool opacityFlipped = false;
        for (uint32_t blasIndex : blasToUpload) {
            auto bIt = m_blasMaterialBounds.find(blasIndex);
            if (bIt != m_blasMaterialBounds.end()) {
                for (auto& e : bIt->second) {
                    if ((oldMatID < 0 || static_cast<int>(e.first) == oldMatID) &&
                        static_cast<int>(e.first) != newMatID) {
                        e.first = static_cast<uint32_t>(newMatID);
                    }
                }
            }
            auto builtIt = m_blasBuiltNonOpaque.find(blasIndex);
            if (builtIt == m_blasBuiltNonOpaque.end() || bIt == m_blasMaterialBounds.end()) continue;
            bool wantNonOpaque = false;
            for (const auto& e : bIt->second) {
                if (!materialCanUseOpaqueFastPath(e.first)) { wantNonOpaque = true; break; }
            }
            const uint8_t want = (wantNonOpaque == builtIt->second)
                ? (uint8_t)0 : (wantNonOpaque ? (uint8_t)2 : (uint8_t)1);
            for (auto& vi : m_vkInstances) {
                if (vi.blasIndex != blasIndex || vi.opacityOverride == want) continue;
                vi.opacityOverride = want;
                opacityFlipped = true;
            }
        }
        if (opacityFlipped) {
            auto merged = m_vkInstances;
            for (const auto& h : m_hairVkInstances) merged.push_back(h);
            if (!merged.empty()) m_device->updateTLAS(merged);
        }

        if (instanceDataChanged) {
            // Refresh instance SSBO (binding 5). The TLAS itself carries no
            // material data, so no AS update is needed for a binding change.
            std::vector<VulkanRT::VkInstanceData> instData;
            instData.reserve(m_vkInstances.size());
            for (const auto& vi : m_vkInstances) { VulkanRT::VkInstanceData d; d.materialIndex = vi.materialIndex; d.blasIndex = vi.blasIndex; instData.push_back(d); }
            if (!instData.empty()) {
                if (m_device->m_instanceDataBuffer.buffer) m_device->destroyBuffer(m_device->m_instanceDataBuffer);
                ::VulkanRT::BufferCreateInfo ci; ci.size = (uint64_t)instData.size() * sizeof(::VulkanRT::VkInstanceData); ci.usage = (::VulkanRT::BufferUsage)((uint32_t)::VulkanRT::BufferUsage::STORAGE | (uint32_t)::VulkanRT::BufferUsage::TRANSFER_DST); ci.location = ::VulkanRT::MemoryLocation::CPU_TO_GPU; ci.initialData = instData.data(); m_device->m_instanceDataBuffer = m_device->createBuffer(ci);
                if (m_device->m_rtDescriptorSet != VK_NULL_HANDLE) { VkDescriptorBufferInfo instInfo{}; instInfo.buffer = m_device->m_instanceDataBuffer.buffer; instInfo.offset = 0; instInfo.range = VK_WHOLE_SIZE; VkWriteDescriptorSet w5{}; w5.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w5.dstSet = m_device->m_rtDescriptorSet; w5.dstBinding = 5; w5.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w5.descriptorCount = 1; w5.pBufferInfo = &instInfo; vkUpdateDescriptorSets(m_device->m_device, 1, &w5, 0, nullptr); }
            }
        }
        changed = true;
    }

    if (changed) {
        m_interactiveViewport.dirty = true;
        resetAccumulation();
    }
}
void VulkanBackendAdapter::setVisibilityByNodeName(const std::string& nodeName, bool visible) {
    if (!m_device || !m_device->isInitialized()) return;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    bool changed = false;
    std::unordered_set<std::string> dirtyRasterMeshes;
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        if (!m_instanceSources[i]) continue;

        std::string instName;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            instName = inst->node_name;
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            instName = tri->getNodeName();
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            // Flat (direct SoA) mesh: no facade, so the node name lives on the TriangleMesh
            // itself. Without this branch instName stayed empty for a flat object's RT
            // instance, the node-name match below never fired, and its TLAS mask never got
            // cleared on delete — the deleted object kept rendering in Vulkan RT even after
            // it vanished from CPU/OptiX/Solid (whose visibility paths don't need this cast).
            instName = tm->nodeName;
        }

        if (matchesNodeNameForInstance(instName, nodeName)) {
            uint8_t newMask = visible ? 0xFF : 0x00;
            if (m_vkInstances[i].mask != newMask) {
                m_vkInstances[i].mask = newMask;
                changed = true;
            }
        }
    }

    for (auto& ri : m_rasterInstances) {
        if (matchesNodeNameForInstance(ri.nodeName, nodeName)) {
            uint8_t newMask = visible ? 0xFF : 0x00;
            if (ri.mask != newMask) {
                ri.mask = newMask;
                dirtyRasterMeshes.insert(ri.meshKey);
                changed = true;
            }
        }
    }

    if (changed) {
        // Visibility edits rebuild/refit TLAS instance data. Serialize them with
        // any in-flight trace or foam-driven topology rebuild before touching AS.
        m_device->waitIdle();

        for (const auto& meshKey : dirtyRasterMeshes) {
            auto meshIt = m_rasterMeshes.find(meshKey);
            if (meshIt != m_rasterMeshes.end()) {
                uploadRasterInstanceBuffer(meshIt->second);
            }
        }
        if (!m_topology_dirty && !m_vkInstances.empty() && m_device->hasTLAS()) {
            // [VULKAN FIX] Update the TLAS with new visibility masks (include hair)
            auto mergedVis = m_vkInstances;
            for (const auto& h : m_hairVkInstances) mergedVis.push_back(h);
            m_device->updateTLAS(mergedVis);
        }
        resetAccumulation();
    }
}

// NOTE: foam (point_sphere_mode) is rendered on Vulkan through the per-particle
// InstanceTransform scatter path (real material/closest-hit shader). The native
// sphere-AABB BLAS path (createFoamSphereBLAS / sphere_*.rint|rchit / binding 18)
// is OptiX-only on this backend; the Vulkan device methods + shaders remain built
// but unused so the approach can be revisited without re-plumbing the pipeline.

void VulkanBackendAdapter::updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!m_device || !m_device->isInitialized()) return;

    // [VULKAN THREAD SAFETY] Prevent background import thread from crashing main render thread.
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // [VULKAN STABILITY] Wait for GPU to finish current frame before destroying/rebuilding resources.
    // This is critical during "Import" where the renderer is already active.
    m_device->waitIdle();

    // Reset ordered VDB instance list — rebuilt below during TLAS construction.
    // SSBO index 0..N must match TLAS customIndex 0..N so shaders look up correct volumes.
    m_orderedVDBInstances.clear();

    // Enable batched BLAS build — all createBLAS calls below will be recorded
    // into a single command buffer instead of N separate GPU submissions.
    if (m_device->hasHardwareRT()) {
        m_device->beginBatchedBLASBuild();
    }

    std::vector<VulkanRT::TLASInstance> vkInstances;
    std::vector<std::shared_ptr<Hittable>> instanceSources;
    // Rebuilt below; the interactive refit consults this to upload positions in
    // BLAS-slot order. Stale across a full rebuild, so clear it up front.
    m_soloBlasBuildTriangles.clear();
    m_soloBlasBuildTriangleSlots.clear();
    m_soloBlasIndexedMesh.clear();

    struct SoloTriangleGroup {
        std::string nodeName;
        std::vector<TriangleData> triangles;
        // Parallel to `triangles` (1:1, same order) — kept so the interactive refit
        // can re-upload vertex positions in the exact BLAS slot order.
        std::vector<std::shared_ptr<Triangle>> trianglePtrs;
        Matrix4x4 transform;
        uint16_t materialID = 0;
        std::shared_ptr<Hittable> representative;
        // Indexed-BLAS eligibility: the whole group is one facade mesh (shared SoA), non-skinned,
        // single space. Set on first triangle, cleared if any later triangle breaks the invariant.
        TriangleMesh* idxMesh = nullptr;
        bool idxLocalSpace = false;
        bool idxEligible = false;
    };
    std::vector<SoloTriangleGroup> soloGroups;
    std::unordered_map<void*, size_t> soloGroupByTransform;

    // Phase 3d: nodes that render via a device-resident CC BLAS keep only their low-poly
    // CAGE in world.objects (for Solid raster + picking + CPU). Their host triangles must
    // be EXCLUDED from the RT geometry gather, else the cage would render on top of the
    // GPU CC. Empty set => no exclusions (zero behaviour change).
    std::unordered_set<std::string> deviceResidentCageNodes;
    for (const auto& drm : m_deviceResidentMeshes) {
        if (!drm.cageNodeName.empty()) deviceResidentCageNodes.insert(drm.cageNodeName);
    }

    // Helper to find and upload all unique meshes recursively
    std::function<void(const std::shared_ptr<Hittable>&)> processObj;
    processObj = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;
        
        // 1. Handle Instances (The primary way geometry is organized)
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->visible) return;
            
            // If we have source triangles, make sure they are uploaded as a BLAS
            if (inst->source_triangles && !inst->source_triangles->empty()) {
                // IMPORTANT: Do NOT key by instance node name (usually unique per instance).
                // Key by shared source geometry identity so thousands of instances reuse one BLAS.
                const auto srcPtrValue = reinterpret_cast<uintptr_t>(inst->source_triangles.get());
                std::string meshKey = "[InstSource]-" + std::to_string(srcPtrValue) +
                                      "-tris-" + std::to_string(inst->source_triangles->size());
                
                // If this mesh hasn't been uploaded to Vulkan yet, do it now
                if (m_meshRegistry.find(meshKey) == m_meshRegistry.end()) {
                    std::vector<TriangleData> triData;
                    triData.reserve(inst->source_triangles->size());
                    
                    for (const auto& t : *inst->source_triangles) {
                        TriangleData d;
                        // IMPORTANT:
                        // Keep instanced source geometry in LOCAL/object space here.
                        // World transforms belong only in TLAS (`inst->transform` below).
                        // If world-space vertices are uploaded here as well, Vulkan applies
                        // the same transform twice (once in BLAS data, once in TLAS),
                        // which shows up as imported/project-loaded meshes being offset
                        // while CPU/OptiX still look correct.
                        d.v0 = t->getOriginalVertexPosition(0);
                        d.v1 = t->getOriginalVertexPosition(1);
                        d.v2 = t->getOriginalVertexPosition(2);
                        d.n0 = t->getOriginalVertexNormal(0);
                        d.n1 = t->getOriginalVertexNormal(1);
                        d.n2 = t->getOriginalVertexNormal(2);
                        auto uv = t->getUVCoordinates();
                        d.uv0 = std::get<0>(uv); d.uv1 = std::get<1>(uv); d.uv2 = std::get<2>(uv);
                        d.materialID = t->getMaterialID();
                        if (d.materialID == MaterialManager::INVALID_MATERIAL_ID) d.materialID = 0;
                        
                        d.hasSkinData = triangleHasEffectiveSkinData(*t);
                        if (d.hasSkinData) {
                            for (int v = 0; v < 3; ++v) {
                                const auto& weights = t->getSkinBoneWeights(v);
                                for (size_t b = 0; b < 4; ++b) {
                                    int bid = -1; float bw = 0.0f;
                                    if (b < weights.size()) { bid = weights[b].first; bw = weights[b].second; }
                                    if (v == 0)      { d.boneIndices_v0[b] = bid; d.boneWeights_v0[b] = bw; }
                                    else if (v == 1) { d.boneIndices_v1[b] = bid; d.boneWeights_v1[b] = bw; }
                                    else if (v == 2) { d.boneIndices_v2[b] = bid; d.boneWeights_v2[b] = bw; }
                                }
                            }
                        }
                        
                        triData.push_back(d);
                    }
                    uploadTriangles(triData, meshKey);
                }
                
                auto it = m_meshRegistry.find(meshKey);
                if (it != m_meshRegistry.end()) {
                    VulkanRT::TLASInstance vi;
                    vi.blasIndex = it->second;
                    vi.transform = inst->transform;
                    uint16_t mId = inst->source_triangles->at(0)->getMaterialID();
                    if (mId == MaterialManager::INVALID_MATERIAL_ID) mId = 0;
                    vi.materialIndex = mId;
                    vi.customIndex = 0; 
                    vi.mask = 0xFF;
                    vi.frontFaceCCW = true;
                    vkInstances.push_back(vi);
                    instanceSources.push_back(inst);
                }
            }
        } 
        // 2. Handle Lists/Collections
        else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) processObj(child);
        }
        // 3. Handle BVH Nodes
        else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            processObj(bvh->left);
            processObj(bvh->right);
        }
        // 4. Handle Solo Triangles (if any are directly in the world)
        else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->visible) return;
            // Phase 3d: this node's dense CC lives in a device-resident BLAS; skip its cage
            // triangles here so RT renders only the GPU CC (cage stays for Solid/picking).
            if (!deviceResidentCageNodes.empty() &&
                deviceResidentCageNodes.count(tri->getNodeName())) return;
            TriangleData d;
            // Use the live vertex state here so deformed standalone meshes such as
            // animated water surfaces reach Vulkan BLAS rebuilds correctly.
            const bool hasSharedTransform = (tri->getTransformPtr() != nullptr);
            const bool useLiveVertexState = isWaterTriangleMaterial(tri);
            if (hasSharedTransform && !useLiveVertexState) {
                // Static imported meshes keep BLAS geometry in object-local space.
                // Their world transform is carried by the TLAS instance transform.
                d.v0 = tri->getOriginalVertexPosition(0);
                d.v1 = tri->getOriginalVertexPosition(1);
                d.v2 = tri->getOriginalVertexPosition(2);
                d.n0 = tri->getOriginalVertexNormal(0);
                d.n1 = tri->getOriginalVertexNormal(1);
                d.n2 = tri->getOriginalVertexNormal(2);
            } else {
                // Procedural/deformed triangles, including animated water with a
                // shared transform, must upload the current local vertex state.
                d.v0 = tri->getV0();
                d.v1 = tri->getV1();
                d.v2 = tri->getV2();
                d.n0 = tri->getN0();
                d.n1 = tri->getN1();
                d.n2 = tri->getN2();
            }
            auto uv = tri->getUVCoordinates();
            d.uv0 = std::get<0>(uv); d.uv1 = std::get<1>(uv); d.uv2 = std::get<2>(uv);
            d.materialID = tri->getMaterialID();

            // [SKINNING FIX] Copy bone weights into TriangleData so the BLAS is created
            // with hasSkinning=true and dispatchSkinning() can deform it each frame.
            d.hasSkinData = triangleHasEffectiveSkinData(*tri);
            if (d.hasSkinData) {
                for (int v = 0; v < 3; ++v) {
                    const auto& weights = tri->getSkinBoneWeights(v);
                    for (size_t b = 0; b < 4; ++b) {
                        int bid = -1; float bw = 0.0f;
                        if (b < weights.size()) { bid = weights[b].first; bw = weights[b].second; }
                        if (v == 0)      { d.boneIndices_v0[b] = bid; d.boneWeights_v0[b] = bw; }
                        else if (v == 1) { d.boneIndices_v1[b] = bid; d.boneWeights_v1[b] = bw; }
                        else if (v == 2) { d.boneIndices_v2[b] = bid; d.boneWeights_v2[b] = bw; }
                    }
                }
            }

            // Group by transform handle so each object keeps independent TLAS transform updates.
            void* groupKey = nullptr;
            Transform* triTransformHandle = tri->getTransformPtr();
            if (triTransformHandle) {
                groupKey = triTransformHandle;
            } else {
                groupKey = tri.get();
            }

            auto found = soloGroupByTransform.find(groupKey);
            if (found == soloGroupByTransform.end()) {
                SoloTriangleGroup group;
                group.nodeName = tri->getNodeName();
                if (group.nodeName.empty()) {
                    group.nodeName = "[World-Solo-Node-" + std::to_string(soloGroups.size()) + "]";
                }
                group.transform = hasSharedTransform ? tri->getTransformMatrix() : Matrix4x4::identity();
                group.materialID = tri->getMaterialID();
                group.representative = tri;
                soloGroups.push_back(std::move(group));
                found = soloGroupByTransform.emplace(groupKey, soloGroups.size() - 1).first;
            }

            auto& targetGroup = soloGroups[found->second];
            targetGroup.triangles.push_back(d);
            targetGroup.trianglePtrs.push_back(tri);

            // Indexed-BLAS eligibility (whole group == one non-skinned facade mesh, single space).
            // The same (hasSharedTransform && !useLiveVertexState) decision the d.v* fill above used
            // determines the upload space (local P_orig vs live P). Mixed mesh / skin / space → off.
            {
                const bool triLocalSpace = (hasSharedTransform && !useLiveVertexState);
                TriangleMesh* triMesh = tri->parentMesh.get();
                if (targetGroup.triangles.size() == 1) {
                    targetGroup.idxMesh = triMesh;
                    targetGroup.idxLocalSpace = triLocalSpace;
                    targetGroup.idxEligible = (triMesh && triMesh->geometry &&
                                               !triMesh->geometry->indices.empty() && !d.hasSkinData);
                } else if (targetGroup.idxEligible) {
                    if (!triMesh || triMesh != targetGroup.idxMesh || d.hasSkinData ||
                        triLocalSpace != targetGroup.idxLocalSpace) {
                        targetGroup.idxEligible = false;
                    }
                }
            }
        }
        // 4b. Dense mesh placed directly in world.objects as a single TriangleMesh (flat/proxy
        // migration) — no per-face Triangle facades. Build an indexed BLAS straight from the mesh
        // SoA (uploadTriangleMeshIndexed) + one TLAS instance; geometry is object-local, the world
        // transform rides the TLAS. DORMANT until the CC materialize flip emits TriangleMesh;
        // harmless when none are present.
        else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (mesh->visible && mesh->geometry && !mesh->geometry->indices.empty() &&
                (deviceResidentCageNodes.empty() || !deviceResidentCageNodes.count(mesh->nodeName))) {
                const auto meshPtrValue = reinterpret_cast<uintptr_t>(mesh.get());
                std::string meshKey = "[DirectMesh]-" + mesh->nodeName + "-" + std::to_string(meshPtrValue);
                // Geometry P is WORLD-baked (convertFromRawArraysToMesh bakes P = getFinal*P_orig),
                // so upload the LOCAL bind pose (P_orig) and let the TLAS instance carry getFinal()
                // — mirrors the facade solo path (local BLAS + world TLAS). Uploading P here would
                // double-apply the transform.
                uint32_t blasIndex = uploadTriangleMeshIndexed(mesh.get(), meshKey, /*useOriginalSpace=*/true);
                if (blasIndex != UINT32_MAX) {
                    uint16_t mId = 0;
                    if (const uint16_t* gMat = mesh->geometry->get_material_ids()) {
                        mId = gMat[mesh->geometry->indices[0]];
                        if (mId == MaterialManager::INVALID_MATERIAL_ID) mId = 0;
                    }
                    VulkanRT::TLASInstance vi;
                    vi.blasIndex = blasIndex;
                    vi.transform = mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
                    vi.materialIndex = mId;
                    vi.customIndex = 0;
                    vi.mask = 0xFF;
                    vi.frontFaceCCW = true;
                    vkInstances.push_back(vi);
                    instanceSources.push_back(mesh);
                }
            }
        }
        // 5. Handle VDB Volumes — create AABB BLAS + TLAS instance for procedural hit group
        else if (auto vdb = std::dynamic_pointer_cast<VDBVolume>(obj)) {
            if (!vdb->isLoaded() || !vdb->visible) return;

            // We create a shared AABB BLAS that covers the unit cube [-0.5, 0.5]^3.
            // The actual world-space bounds and scaling are applied via the TLAS instance transform.
            float aabbMin[3] = { -0.5f, -0.5f, -0.5f };
            float aabbMax[3] = {  0.5f,  0.5f,  0.5f };

            uint32_t aabbBlasIdx = m_device->createAABB_BLAS(aabbMin, aabbMax);
            if (aabbBlasIdx != UINT32_MAX) {
                VulkanRT::TLASInstance vi;
                vi.blasIndex = aabbBlasIdx;
                
                // Construct a transform that maps [-0.5, 0.5]^3 to worldBounds
                AABB worldBounds = vdb->getWorldBounds();
                Vec3 center = (worldBounds.min + worldBounds.max) * 0.5f;
                Vec3 size = worldBounds.max - worldBounds.min;
                // Avoid zero scaling
                if (size.x < 1e-4f) size.x = 1e-4f;
                if (size.y < 1e-4f) size.y = 1e-4f;
                if (size.z < 1e-4f) size.z = 1e-4f;
                
                Matrix4x4 scale = Matrix4x4::scaling(size);
                Matrix4x4 trans = Matrix4x4::translation(center);
                vi.transform = trans * scale;

                vi.customIndex = (uint32_t)m_orderedVDBInstances.size(); // stable SSBO index
                m_orderedVDBInstances.push_back(vdb); // record TLAS order for SSBO build
                // 0x02 = volume-only mask. Shadow rays use mask 0x01 (triangles only),
                // so they never intersect volume AABBs and cannot cast hard shadows.
                vi.mask = 0x02;
                vi.frontFaceCCW = false;
                // SBT offset = 1 → routes to hit group index (raygen=0, miss=1, triangle_hit=2, volume_hit=3)
                // In the SBT hit region, triangle is at offset 0, volume is at offset 1
                vi.sbtRecordOffset = 1;
                vkInstances.push_back(vi);
                instanceSources.push_back(vdb);
                VK_INFO() << "[Vulkan] VDB volume added to TLAS: " << vdb->name
                          << " worldBounds=[" << worldBounds.min.x << "," << worldBounds.min.y << "," << worldBounds.min.z
                          << " -> " << worldBounds.max.x << "," << worldBounds.max.y << "," << worldBounds.max.z << "]" << std::endl;
            }
        }
        // 6. Handle Gas Volumes in unified VDB mode — same procedural volume path as VDBs
        else if (auto gas = std::dynamic_pointer_cast<GasVolume>(obj)) {
            if (!gas->visible || gas->render_path != GasVolume::VolumeRenderPath::VDBUnified || gas->live_vdb_id < 0) return;

            float aabbMin[3] = { -0.5f, -0.5f, -0.5f };
            float aabbMax[3] = {  0.5f,  0.5f,  0.5f };

            uint32_t aabbBlasIdx = m_device->createAABB_BLAS(aabbMin, aabbMax);
            if (aabbBlasIdx != UINT32_MAX) {
                VulkanRT::TLASInstance vi;
                vi.blasIndex = aabbBlasIdx;

                Vec3 worldMin, worldMax;
                gas->getWorldBounds(worldMin, worldMax);
                Vec3 center = (worldMin + worldMax) * 0.5f;
                Vec3 size = worldMax - worldMin;
                if (size.x < 1e-4f) size.x = 1e-4f;
                if (size.y < 1e-4f) size.y = 1e-4f;
                if (size.z < 1e-4f) size.z = 1e-4f;

                Matrix4x4 scale = Matrix4x4::scaling(size);
                Matrix4x4 trans = Matrix4x4::translation(center);
                vi.transform = trans * scale;

                vi.customIndex = (uint32_t)m_orderedVDBInstances.size();
                m_orderedVDBInstances.push_back(gas);
                vi.mask = 0x02;
                vi.frontFaceCCW = false;
                vi.sbtRecordOffset = 1;
                vkInstances.push_back(vi);
                instanceSources.push_back(gas);
                VK_INFO() << "[Vulkan] Unified gas volume added to TLAS: " << gas->name
                          << " worldBounds=[" << worldMin.x << "," << worldMin.y << "," << worldMin.z
                          << " -> " << worldMax.x << "," << worldMax.y << "," << worldMax.z << "]" << std::endl;
            }
        }
    };

    auto hasInstancePrefix = [](const std::string& nodeName) -> bool {
        return nodeName.rfind("_inst_gid", 0) == 0;
    };
    size_t baseObjectCount = objects.size();
    while (baseObjectCount > 0) {
        const auto& obj = objects[baseObjectCount - 1];
        auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
        if (!inst || !hasInstancePrefix(inst->node_name)) {
            break;
        }
        --baseObjectCount;
    }

    for (size_t i = 0; i < baseObjectCount; ++i) {
        processObj(objects[i]);
    }
    
    // Store for updateInstanceTransforms
    m_lastObjects = objects;

    // Handle Solo Triangles: one BLAS/TLAS instance per object-transform group.
    if (!soloGroups.empty()) {
        for (size_t groupIndex = 0; groupIndex < soloGroups.size(); ++groupIndex) {
            const auto& group = soloGroups[groupIndex];
            if (group.triangles.empty()) continue;

            std::string meshKey = "[World-Solo]-" + group.nodeName + "-" + std::to_string(groupIndex);

            // Indexed path only when the group is the ENTIRE mesh (count match guards against
            // partial visibility / culled faces, which the indexed BLAS — built from the whole
            // SoA — cannot represent). Otherwise the proven non-indexed 3N upload runs.
            const bool indexedOk = group.idxEligible && group.idxMesh && group.idxMesh->geometry &&
                                   group.trianglePtrs.size() == group.idxMesh->num_triangles();

            uint32_t soloBlasIndex;
            if (indexedOk) {
                soloBlasIndex = uploadTriangleMeshIndexed(group.idxMesh, meshKey, group.idxLocalSpace);
            } else {
                soloBlasIndex = uploadTriangles(group.triangles, meshKey);

                // Record build-order triangle pointers so the interactive refit can
                // re-upload positions into the exact BLAS slots (mesh_cache order may
                // differ from this scene-graph traversal order). Indexed BLASes refit
                // straight from the mesh SoA instead, so they skip this.
                if (soloBlasIndex != UINT32_MAX) {
                    m_soloBlasBuildTriangles[soloBlasIndex] = group.trianglePtrs;
                    auto& slots = m_soloBlasBuildTriangleSlots[soloBlasIndex];
                    slots.clear();
                    slots.reserve(group.trianglePtrs.size());
                    for (uint32_t slotIdx = 0; slotIdx < group.trianglePtrs.size(); ++slotIdx) {
                        if (group.trianglePtrs[slotIdx]) {
                            slots[group.trianglePtrs[slotIdx].get()] = slotIdx;
                        }
                    }
                }
            }

            VulkanRT::TLASInstance vi;
            vi.blasIndex = soloBlasIndex;
            vi.transform = group.transform;
            vi.materialIndex = group.materialID;
            vi.customIndex = 0;
            vi.mask = 0xFF;
            vi.frontFaceCCW = true;
            vkInstances.push_back(vi);

            if (group.representative) {
                instanceSources.push_back(group.representative);
            } else {
                instanceSources.push_back(nullptr);
            }
        }
    }

    auto makeTriangleData = [](const Triangle& tri) {
        TriangleData d;
        d.v0 = tri.getOriginalVertexPosition(0);
        d.v1 = tri.getOriginalVertexPosition(1);
        d.v2 = tri.getOriginalVertexPosition(2);
        d.n0 = tri.getOriginalVertexNormal(0);
        d.n1 = tri.getOriginalVertexNormal(1);
        d.n2 = tri.getOriginalVertexNormal(2);
        auto uv = tri.getUVCoordinates();
        d.uv0 = std::get<0>(uv);
        d.uv1 = std::get<1>(uv);
        d.uv2 = std::get<2>(uv);
        d.materialID = tri.getMaterialID();
        if (d.materialID == MaterialManager::INVALID_MATERIAL_ID) d.materialID = 0;
        d.hasSkinData = triangleHasEffectiveSkinData(tri);
        if (d.hasSkinData) {
            for (int v = 0; v < 3; ++v) {
                const auto& weights = tri.getSkinBoneWeights(v);
                for (size_t b = 0; b < 4; ++b) {
                    int bid = -1;
                    float bw = 0.0f;
                    if (b < weights.size()) {
                        bid = weights[b].first;
                        bw = weights[b].second;
                    }
                    if (v == 0) {
                        d.boneIndices_v0[b] = bid;
                        d.boneWeights_v0[b] = bw;
                    } else if (v == 1) {
                        d.boneIndices_v1[b] = bid;
                        d.boneWeights_v1[b] = bw;
                    } else {
                        d.boneIndices_v2[b] = bid;
                        d.boneWeights_v2[b] = bw;
                    }
                }
            }
        }
        return d;
    };

    const auto& instanceGroups = InstanceManager::getInstance().getGroups();
    struct ScatterSourceMeta {
        std::vector<uint32_t> blasBySource;
        std::vector<uint32_t> materialBySource;
    };
    std::vector<ScatterSourceMeta> scatterMeta(instanceGroups.size());
    size_t totalScatterInstances = 0;

    for (size_t gi = 0; gi < instanceGroups.size(); ++gi) {
        const auto& group = instanceGroups[gi];
        if (group.instances.empty() || group.sources.empty()) continue;

        auto& meta = scatterMeta[gi];
        meta.blasBySource.assign(group.sources.size(), UINT32_MAX);
        meta.materialBySource.assign(group.sources.size(), 0);

        for (size_t si = 0; si < group.sources.size(); ++si) {
            const auto& source = group.sources[si];
            const auto* triSource = source.centered_triangles_ptr ? source.centered_triangles_ptr.get() : nullptr;
            const auto& fallbackTriangles = source.triangles;
            const auto* triangles = (triSource && !triSource->empty()) ? triSource : &fallbackTriangles;
            if (!triangles || triangles->empty()) continue;

            const uintptr_t srcPtr = reinterpret_cast<uintptr_t>(triangles);
            const std::string meshKey = "[InstGroup]-" + std::to_string(group.id) + "-" +
                                        std::to_string(si) + "-" + std::to_string(srcPtr) +
                                        "-tris-" + std::to_string(triangles->size());

            uint32_t blasIndex = UINT32_MAX;
            auto registryIt = m_meshRegistry.find(meshKey);
            if (registryIt != m_meshRegistry.end()) {
                blasIndex = registryIt->second;
            } else {
                std::vector<TriangleData> triData;
                triData.reserve(triangles->size());
                for (const auto& tri : *triangles) {
                    if (tri) triData.push_back(makeTriangleData(*tri));
                }
                if (!triData.empty()) {
                    blasIndex = uploadTriangles(triData, meshKey);
                }
            }
            if (blasIndex == UINT32_MAX) continue;

            meta.blasBySource[si] = blasIndex;
            if (!triangles->empty() && (*triangles)[0]) {
                uint16_t matId = (*triangles)[0]->getMaterialID();
                meta.materialBySource[si] =
                    (matId == MaterialManager::INVALID_MATERIAL_ID) ? 0u : static_cast<uint32_t>(matId);
            }
        }

        for (const auto& inst : group.instances) {
            int srcIdx = inst.source_index;
            if (srcIdx < 0 || srcIdx >= static_cast<int>(group.sources.size())) srcIdx = 0;
            if (srcIdx < static_cast<int>(meta.blasBySource.size()) &&
                meta.blasBySource[srcIdx] != UINT32_MAX) {
                ++totalScatterInstances;
            }
        }
    }

    vkInstances.reserve(vkInstances.size() + totalScatterInstances);
    instanceSources.reserve(instanceSources.size() + totalScatterInstances);

    unsigned scatterThreads = std::thread::hardware_concurrency();
    if (scatterThreads == 0) scatterThreads = 4;
    constexpr size_t kScatterParallelThreshold = 1024;

    for (size_t gi = 0; gi < instanceGroups.size(); ++gi) {
        const auto& group = instanceGroups[gi];
        if (group.instances.empty() || group.sources.empty()) continue;

        // NOTE: point_sphere_mode (foam) groups are rendered on Vulkan via the SAME
        // per-particle InstanceTransform scatter path as foliage — one TLAS instance
        // per particle using the real material/closest-hit shader (correct foam
        // materials + NEE). The flag is honoured only by the OptiX backend (native
        // sphere GAS). BVH degradation from foam motion is handled by a per-frame
        // full TLAS rebuild in updateInstanceTransforms() (see foamPresent there).
        const auto& meta = scatterMeta[gi];
        if (meta.blasBySource.empty()) continue;

        std::vector<VulkanRT::TLASInstance> local(group.instances.size());
        for (auto& vi : local) vi.blasIndex = UINT32_MAX;
        auto fillRange = [&group, &meta, &local](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                const auto& inst = group.instances[i];
                int srcIdx = inst.source_index;
                if (srcIdx < 0 || srcIdx >= static_cast<int>(group.sources.size())) srcIdx = 0;
                if (srcIdx >= static_cast<int>(meta.blasBySource.size()) ||
                    meta.blasBySource[srcIdx] == UINT32_MAX) {
                    continue;
                }

                auto& vi = local[i];
                vi.blasIndex = meta.blasBySource[srcIdx];
                vi.transform = inst.toMatrix();
                vi.materialIndex = meta.materialBySource[srcIdx];
                vi.customIndex = 0;
                vi.mask = 0xFF;
                vi.frontFaceCCW = true;
                vi.scatterGroupId = group.id;
                vi.scatterInstanceIndex = static_cast<uint32_t>(i);
            }
        };

        if (group.instances.size() < kScatterParallelThreshold || scatterThreads < 2) {
            fillRange(0, group.instances.size());
        } else {
            const size_t chunk = (group.instances.size() + scatterThreads - 1) / scatterThreads;
            std::vector<std::future<void>> futures;
            futures.reserve(scatterThreads);
            for (unsigned t = 0; t < scatterThreads; ++t) {
                const size_t s = t * chunk;
                const size_t e = (std::min)(s + chunk, group.instances.size());
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, fillRange, s, e));
            }
            for (auto& f : futures) f.get();
        }

        for (auto& vi : local) {
            if (vi.blasIndex == UINT32_MAX) continue;
            vkInstances.push_back(std::move(vi));
            instanceSources.push_back(nullptr);
        }
    }
    
    // Phase 3d: device-resident CC meshes (static / render-only). Geometry is already on
    // the GPU, so register a BLAS straight from its address and append a TLAS instance —
    // no host triangles, no upload. This only APPENDS; existing instance logic is
    // untouched, and an empty list is a complete no-op. Runs inside the batched BLAS
    // region (flushed just below). instanceSources push nullptr: these are not pickable
    // as triangles — picking / Solid raster use the editable cage instead.
    for (const auto& drm : m_deviceResidentMeshes) {
        if (drm.deviceAddress == 0 || drm.triCount == 0) continue;
        uint32_t blasIndex = uploadDeviceResidentMesh(drm.meshKey, drm.deviceAddress,
                                                      drm.vertexCount, drm.triCount,
                                                      drm.materialIds, drm.opaque);
        if (blasIndex == UINT32_MAX) continue;

        VulkanRT::TLASInstance vi;
        vi.blasIndex = blasIndex;
        vi.transform = drm.transform;
        vi.materialIndex = drm.materialID;
        vi.customIndex = 0;
        vi.mask = 0xFF;
        vi.frontFaceCCW = true;
        vkInstances.push_back(vi);
        instanceSources.push_back(nullptr);
    }

    // Flush all pending BLAS builds in one GPU submit before TLAS creation
    if (m_device->hasHardwareRT()) {
        m_device->endBatchedBLASBuild();
    }

    m_vkInstances = vkInstances; // Store for updates
    m_instanceSources = instanceSources;
    m_topology_dirty = true;

    // Snapshot the number of mesh BLASes so clearHairGeometry() can safely remove
    // only hair BLASes later.  IMPORTANT: uploadHairToGPU() may have been called
    // BEFORE this function (e.g. from updateAnimationState's force_bind_pose path),
    // meaning hair BLASes are already appended to m_blasList.  We must subtract them
    // so that m_meshBlasCount only covers true mesh BLASes; otherwise clearHairGeometry
    // starts at the wrong index, never removes the stale hair BLAS, and orphaned BLASes
    // accumulate each time the Hair panel is opened — eventually crashing Vulkan.
    uint32_t hairBlasCount = m_device ? (uint32_t)m_hairVkInstances.size() : 0;
    m_meshBlasCount = m_device ? (uint32_t)(m_device->m_blasList.size() - hairBlasCount) : 0;

    // Merge mesh + hair instances for TLAS
    std::vector<VulkanRT::TLASInstance> allInstances = m_vkInstances;
    for (const auto& hi : m_hairVkInstances) allInstances.push_back(hi);

    if (!allInstances.empty()) {
        VulkanRT::TLASCreateInfo tlasInfo;
        tlasInfo.instances = allInstances;
        // [VULKAN FIX] Use allowUpdate = true so that subsequent updateObjectTransform calls can refit.
        tlasInfo.allowUpdate = true; 
        m_device->createTLAS(tlasInfo);

        // [CRASH GUARD] Re-enable ray tracing now that a valid TLAS exists.
        if (m_device->m_rtPipeline != VK_NULL_HANDLE) {
            m_device->m_rtPipelineReady = true;
        }

        // Upload shader-side BLAS/instance lookup buffers. These descriptors may
        // already exist after a previous render, so refresh bindings immediately.
        refreshVulkanGeometryDataBinding(m_device.get());
        refreshVulkanInstanceDataBinding(m_device.get(), m_vkInstances);
        
        resetAccumulation();

        // [OPTIMIZATION] Pre-populate the sync cache now while we have the mapping fresh.
        m_instance_sync_cache.clear();
        for (size_t i = 0; i < m_instanceSources.size(); ++i) {
            if (m_instanceSources[i]) {
                VulkanBackendAdapter::InstanceTransformCache item;
                item.instance_id = (int)i;
                item.representative_hittable = m_instanceSources[i];
                m_instance_sync_cache.push_back(item);
            }
        }
        m_topology_dirty = false;

        SCENE_LOG_INFO("[Vulkan] TLAS rebuilt with " + std::to_string(allInstances.size()) + " instances ("
            + std::to_string(vkInstances.size()) + " mesh, "
            + std::to_string(m_hairVkInstances.size()) + " hair).");
    } else {
        SCENE_LOG_WARN("[Vulkan] updateGeometry: No valid geometry found in the scene.");
        // [VULKAN FIX] Also clear TLAS and disable RT when empty to prevent crash
        VulkanRT::TLASCreateInfo emptyTlas;
        m_device->createTLAS(emptyTlas);
        m_device->m_rtPipelineReady = false;
        resetAccumulation();
    }

    // Keep raster geometry only when this backend is actively serving the
    // interactive viewport itself. In the refactored architecture a dedicated
    // viewport backend owns Solid/Matcap raster state, so letting the Vulkan RT
    // render backend also retain/rebuild raster meshes causes duplicate uploads,
    // slower backend switches, and extra instability during Rendered mode.
    if (shouldUseInteractiveViewport()) {
        buildRasterGeometry(objects);
    } else if (!m_rasterMeshes.empty() || !m_rasterInstances.empty()) {
        destroyAllRasterMeshes();
        m_interactiveViewport.dirty = true;
    }
}

bool VulkanBackendAdapter::tryAppendGeometryIncremental(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!m_device || !m_device->isInitialized()) return false;
    if (!m_device->hasHardwareRT()) return false;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Pre-conditions: must already have a built TLAS / mesh state to extend.
    if (m_topology_dirty) return false;
    if (m_vkInstances.empty()) return false;
    if (m_instanceSources.size() != m_vkInstances.size()) return false;

    // Snapshot of every existing source pointer (HittableInstance, Triangle, VDB, ...).
    // The walk below must visit each of these — if anything is missing, geometry was
    // removed and the incremental path is unsafe (would leave stale BLAS instances live).
    std::unordered_set<const Hittable*> existing_sources;
    existing_sources.reserve(m_instanceSources.size());
    std::unordered_set<void*> existing_solo_group_keys;
    existing_solo_group_keys.reserve(m_instanceSources.size());
    for (const auto& src : m_instanceSources) {
        if (src) {
            existing_sources.insert(src.get());
            if (auto tri = std::dynamic_pointer_cast<Triangle>(src)) {
                existing_solo_group_keys.insert(
                    tri->getTransformPtr() ? static_cast<void*>(tri->getTransformPtr()) : static_cast<void*>(tri.get()));
            }
        }
    }

    std::unordered_set<uint64_t> existing_scatter_instances;
    existing_scatter_instances.reserve(m_vkInstances.size());
    for (const auto& vi : m_vkInstances) {
        if (vi.scatterGroupId < 0 || vi.scatterInstanceIndex == UINT32_MAX) continue;
        const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(vi.scatterGroupId)) << 32) |
                             static_cast<uint64_t>(vi.scatterInstanceIndex);
        existing_scatter_instances.insert(key);
    }

    std::vector<std::shared_ptr<HittableInstance>> new_instances;
    struct NewSoloTriangleGroup {
        std::string nodeName;
        std::vector<TriangleData> triangles;
        Matrix4x4 transform;
        uint16_t materialID = 0;
        std::shared_ptr<Hittable> representative;
    };
    std::vector<NewSoloTriangleGroup> new_solo_groups;
    std::unordered_map<void*, size_t> new_solo_group_by_transform;
    std::unordered_set<const Hittable*> walk_seen;
    walk_seen.reserve(existing_sources.size() + 16);
    bool unsupported_topology = false;

    std::function<void(const std::shared_ptr<Hittable>&)> walk;
    walk = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj || unsupported_topology) return;
        if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) walk(child);
            return;
        }
        if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            walk(bvh->left);
            walk(bvh->right);
            return;
        }
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->visible) return;
            if (!inst->source_triangles || inst->source_triangles->empty()) return;
            int scatterGroupId = -1;
            uint32_t scatterInstanceIndex = UINT32_MAX;
            if (parseScatterNodeName(inst->node_name, scatterGroupId, scatterInstanceIndex)) {
                const uint64_t scatterKey =
                    (static_cast<uint64_t>(static_cast<uint32_t>(scatterGroupId)) << 32) |
                    static_cast<uint64_t>(scatterInstanceIndex);
                if (existing_scatter_instances.count(scatterKey) != 0) {
                    return;
                }
            }
            walk_seen.insert(inst.get());
            if (existing_sources.count(inst.get()) == 0) {
                new_instances.push_back(inst);
            }
            return;
        }
        // Solo Triangle / VDBVolume / anything else: incremental can only verify they
        // are still represented in m_instanceSources. They must NOT be new — the slow
        // path groups solo triangles by transform handle and rebuilds the VDB ordered
        // list, which the incremental path does not replicate.
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->visible) return;
            walk_seen.insert(tri.get());
            if (existing_sources.count(tri.get()) == 0) {
                void* groupKey = tri->getTransformPtr() ? static_cast<void*>(tri->getTransformPtr()) : static_cast<void*>(tri.get());
                if (existing_solo_group_keys.count(groupKey) != 0) {
                    return;
                }

                TriangleData d;
                const bool hasSharedTransform = (tri->getTransformPtr() != nullptr);
                const bool useLiveVertexState = isWaterTriangleMaterial(tri);
                if (hasSharedTransform && !useLiveVertexState) {
                    d.v0 = tri->getOriginalVertexPosition(0);
                    d.v1 = tri->getOriginalVertexPosition(1);
                    d.v2 = tri->getOriginalVertexPosition(2);
                    d.n0 = tri->getOriginalVertexNormal(0);
                    d.n1 = tri->getOriginalVertexNormal(1);
                    d.n2 = tri->getOriginalVertexNormal(2);
                } else {
                    d.v0 = tri->getV0();
                    d.v1 = tri->getV1();
                    d.v2 = tri->getV2();
                    d.n0 = tri->getN0();
                    d.n1 = tri->getN1();
                    d.n2 = tri->getN2();
                }
                auto uv = tri->getUVCoordinates();
                d.uv0 = std::get<0>(uv);
                d.uv1 = std::get<1>(uv);
                d.uv2 = std::get<2>(uv);
                d.materialID = tri->getMaterialID();
                if (d.materialID == MaterialManager::INVALID_MATERIAL_ID) d.materialID = 0;
                d.hasSkinData = triangleHasEffectiveSkinData(*tri);
                if (d.hasSkinData) {
                    for (int v = 0; v < 3; ++v) {
                        const auto& weights = tri->getSkinBoneWeights(v);
                        for (size_t b = 0; b < 4; ++b) {
                            int bid = -1;
                            float bw = 0.0f;
                            if (b < weights.size()) {
                                bid = weights[b].first;
                                bw = weights[b].second;
                            }
                            if (v == 0) {
                                d.boneIndices_v0[b] = bid;
                                d.boneWeights_v0[b] = bw;
                            } else if (v == 1) {
                                d.boneIndices_v1[b] = bid;
                                d.boneWeights_v1[b] = bw;
                            } else {
                                d.boneIndices_v2[b] = bid;
                                d.boneWeights_v2[b] = bw;
                            }
                        }
                    }
                }

                auto found = new_solo_group_by_transform.find(groupKey);
                if (found == new_solo_group_by_transform.end()) {
                    NewSoloTriangleGroup group;
                    group.nodeName = tri->getNodeName();
                    if (group.nodeName.empty()) {
                        group.nodeName = "[Append-Solo-Node-" + std::to_string(new_solo_groups.size()) + "]";
                    }
                    group.transform = hasSharedTransform ? tri->getTransformMatrix() : Matrix4x4::identity();
                    group.materialID = static_cast<uint16_t>(d.materialID);
                    group.representative = tri;
                    new_solo_groups.push_back(std::move(group));
                    found = new_solo_group_by_transform.emplace(groupKey, new_solo_groups.size() - 1).first;
                }
                new_solo_groups[found->second].triangles.push_back(d);
            }
            return;
        }
        if (auto vdb = std::dynamic_pointer_cast<VDBVolume>(obj)) {
            if (!vdb->isLoaded() || !vdb->visible) return;
            walk_seen.insert(vdb.get());
            if (existing_sources.count(vdb.get()) == 0) {
                unsupported_topology = true;
            }
            return;
        }
        // Unknown leaf — bail.
        unsupported_topology = true;
    };
    for (const auto& obj : objects) walk(obj);

    if (unsupported_topology) return false;

    // Every existing source must still be present in the new walk. Missing → removal,
    // fall back to full rebuild so dead BLAS instances are pruned.
    for (const Hittable* src : existing_sources) {
        if (walk_seen.count(src) == 0) return false;
    }

    if (new_instances.empty() && new_solo_groups.empty()) {
        // Nothing actually changed (or only material/transform updates that other paths cover).
        return true;
    }

    // Upload any unseen source meshes as BLAS; reuse the registry entry otherwise.
    if (m_device->hasHardwareRT()) {
        m_device->beginBatchedBLASBuild();
    }

    std::vector<VulkanRT::TLASInstance> appended;
    appended.reserve(new_instances.size());
    std::vector<std::shared_ptr<Hittable>> appendedSources;
    appendedSources.reserve(new_instances.size() + new_solo_groups.size());

    for (auto& inst : new_instances) {
        const auto srcPtrValue = reinterpret_cast<uintptr_t>(inst->source_triangles.get());
        std::string meshKey = "[InstSource]-" + std::to_string(srcPtrValue) +
                              "-tris-" + std::to_string(inst->source_triangles->size());

        if (m_meshRegistry.find(meshKey) == m_meshRegistry.end()) {
            std::vector<TriangleData> triData;
            triData.reserve(inst->source_triangles->size());
            for (const auto& t : *inst->source_triangles) {
                TriangleData d;
                d.v0 = t->getOriginalVertexPosition(0);
                d.v1 = t->getOriginalVertexPosition(1);
                d.v2 = t->getOriginalVertexPosition(2);
                d.n0 = t->getOriginalVertexNormal(0);
                d.n1 = t->getOriginalVertexNormal(1);
                d.n2 = t->getOriginalVertexNormal(2);
                auto uv = t->getUVCoordinates();
                d.uv0 = std::get<0>(uv); d.uv1 = std::get<1>(uv); d.uv2 = std::get<2>(uv);
                d.materialID = t->getMaterialID();
                if (d.materialID == MaterialManager::INVALID_MATERIAL_ID) d.materialID = 0;
                d.hasSkinData = triangleHasEffectiveSkinData(*t);
                if (d.hasSkinData) {
                    for (int v = 0; v < 3; ++v) {
                        const auto& weights = t->getSkinBoneWeights(v);
                        for (size_t b = 0; b < 4; ++b) {
                            int bid = -1; float bw = 0.0f;
                            if (b < weights.size()) { bid = weights[b].first; bw = weights[b].second; }
                            if (v == 0)      { d.boneIndices_v0[b] = bid; d.boneWeights_v0[b] = bw; }
                            else if (v == 1) { d.boneIndices_v1[b] = bid; d.boneWeights_v1[b] = bw; }
                            else if (v == 2) { d.boneIndices_v2[b] = bid; d.boneWeights_v2[b] = bw; }
                        }
                    }
                }
                triData.push_back(d);
            }
            uploadTriangles(triData, meshKey);
        }

        auto it = m_meshRegistry.find(meshKey);
        if (it == m_meshRegistry.end()) continue; // upload failed; skip silently

        VulkanRT::TLASInstance vi;
        vi.blasIndex = it->second;
        vi.transform = inst->transform;
        uint16_t mId = inst->source_triangles->at(0)->getMaterialID();
        if (mId == MaterialManager::INVALID_MATERIAL_ID) mId = 0;
        vi.materialIndex = mId;
        vi.customIndex = 0;
        vi.mask = 0xFF;
        vi.frontFaceCCW = true;
        int scatterGroupId = -1;
        uint32_t scatterInstanceIndex = UINT32_MAX;
        if (parseScatterNodeName(inst->node_name, scatterGroupId, scatterInstanceIndex)) {
            vi.scatterGroupId = scatterGroupId;
            vi.scatterInstanceIndex = scatterInstanceIndex;
        }
        appended.push_back(vi);
        appendedSources.push_back(inst);
    }

    for (size_t groupIndex = 0; groupIndex < new_solo_groups.size(); ++groupIndex) {
        const auto& group = new_solo_groups[groupIndex];
        if (group.triangles.empty()) continue;

        std::string meshKey = "[World-Solo-Append]-" + group.nodeName + "-" + std::to_string(groupIndex) +
                              "-tris-" + std::to_string(group.triangles.size());
        uint32_t soloBlasIndex = uploadTriangles(group.triangles, meshKey);
        if (soloBlasIndex == UINT32_MAX) continue;

        VulkanRT::TLASInstance vi;
        vi.blasIndex = soloBlasIndex;
        vi.transform = group.transform;
        vi.materialIndex = group.materialID;
        vi.customIndex = 0;
        vi.mask = 0xFF;
        vi.frontFaceCCW = true;
        appended.push_back(vi);
        appendedSources.push_back(group.representative);
    }

    if (m_device->hasHardwareRT()) {
        m_device->endBatchedBLASBuild();
    }

    if (appended.empty()) return false;

    m_vkInstances.insert(m_vkInstances.end(), appended.begin(), appended.end());
    m_instanceSources.insert(m_instanceSources.end(), appendedSources.begin(), appendedSources.end());

    // Refit TLAS with the merged instance list (incl. hair).
    std::vector<VulkanRT::TLASInstance> merged = m_vkInstances;
    for (const auto& h : m_hairVkInstances) merged.push_back(h);
    m_device->updateTLAS(merged);

    // New source meshes can append BLASes, so both geometry lookup (binding 4)
    // and per-instance lookup (binding 5) must grow together.
    refreshVulkanGeometryDataBinding(m_device.get());
    refreshVulkanInstanceDataBinding(m_device.get(), m_vkInstances);

    // Refresh meshBlasCount snapshot — needed so clearHairGeometry() never walks into mesh BLASes.
    uint32_t hairBlasCount = (uint32_t)m_hairVkInstances.size();
    m_meshBlasCount = (uint32_t)(m_device->m_blasList.size() - hairBlasCount);

    // Keep instance-sync cache aligned with m_instanceSources (used by transform updates).
    for (size_t i = m_instance_sync_cache.size(); i < m_instanceSources.size(); ++i) {
        if (m_instanceSources[i]) {
            InstanceTransformCache item;
            item.instance_id = (int)i;
            item.representative_hittable = m_instanceSources[i];
            m_instance_sync_cache.push_back(item);
        }
    }

    resetAccumulation();

    SCENE_LOG_INFO("[Vulkan] Incremental geometry append: +" + std::to_string(appended.size()) +
                   " instances, total " + std::to_string(m_vkInstances.size()) + " (hair " +
                   std::to_string(m_hairVkInstances.size()) + ").");
    return true;
}

// Materials & Textures
void VulkanBackendAdapter::uploadMaterials(const std::vector<MaterialData>& materials) {
    if (!m_device || !m_device->isInitialized()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    // Bake hot-reload eviction: the UI sets g_texture_pool_dirty instead of calling
    // destroyAllVulkanBackingForOwner directly, because that would destroy VkImageViews
    // while the GPU still references them in the previous frame's command buffer.
    // We call purgeUploadedTextureCacheLocked here (not bare destroyAllVulkanBackingForOwner)
    // because it also clears RT descriptor bindings before destroying — skipping that step
    // causes a Vulkan validation __debugbreak when the sampler is still bound to a slot.
    // DDS-backed textures registered with createdByPool=true have no destroyFn and are
    // only cleaned up by purge's fallback loop, so the full purge path is required.
    // g_texture_pool_dirty is cleared by Main.cpp after BOTH backends have purged.
    if (g_texture_pool_dirty) {
        purgeUploadedTextureCacheLocked(); // waitIdle + clearDescriptors + destroy + clear containers
    }

    // Descriptor slot exhaustion guard.
    if (m_nextTextureID >= VULKAN_TEXTURE_PURGE_THRESHOLD ||
        static_cast<int32_t>(m_uploadedImages.size()) >= VULKAN_TEXTURE_PURGE_THRESHOLD) {
        SCENE_LOG_WARN("[Vulkan] Texture cache near descriptor capacity (" +
                       std::to_string(VULKAN_TEXTURE_CAPACITY) + " slots); purging and re-uploading active textures.");
        purgeUploadedTextureCacheLocked();
    }

    // VRAM byte-pressure check: warn at 70%, trigger LRU trim at 85%.
    if (m_sceneTextureManager) {
        const uint64_t dedicatedVRAM = m_device->getCapabilities().dedicatedVRAM;
        if (dedicatedVRAM > 0) {
            // Use GPU-resident bytes only — totalEstimatedTextureBytes() also
            // counts CPU-side ghost records (DDS cache, paint history, records
            // whose backings have been torn down), which used to inflate the
            // estimate ~4x and triggered false-positive LRU eviction during
          // mesh paint, evicting freshly uploaded paint textures before
            // raster (and the active RT backend) could pick them up.
            checkAndTrimVRAMThreshold();
            const uint64_t textureBytes = m_sceneTextureManager->totalResidentTextureBytes();
            const uint64_t warnThreshold = dedicatedVRAM * 7 / 10;
            const uint64_t usedMB  = textureBytes / (1024 * 1024);
            const uint64_t totalMB = dedicatedVRAM / (1024 * 1024);
            if (textureBytes > warnThreshold && textureBytes <= dedicatedVRAM * 85 / 100) {
                const std::string msg = "VRAM pressure: ~" + std::to_string(usedMB) +
                                        " MB texture / " + std::to_string(totalMB) + " MB.";
                SCENE_LOG_WARN("[Vulkan] " + msg);
                if (m_statusCallback) m_statusCallback(msg, 1); // sari
            }
        }
    }

    m_textureUploadBytes = 0;
    m_textureUploadCount = 0;
    m_textureUploadBC4Count = 0;
    m_textureUploadBC5Count = 0;
    m_textureUploadBC7Count = 0;
    m_textureUploadR8Count = 0;
    m_textureUploadRGBA8Count = 0;
    m_textureUploadFloatCount = 0;
    m_textureUploadRG8Count = 0;
    m_textureUploadSummaryDirty = false;

    std::vector<VulkanRT::VkGpuMaterial> gpuMats;
    gpuMats.reserve(materials.size());

    // ── Parallel pixel staging + batched GPU upload ───────────────────────────
    // Phase 1: collect unique textures not yet uploaded and not dirty
    const bool limitMaterialPreviewTextures = isViewportTextureOwner(sceneTextureOwnerScope());
    struct TexPrepReq {
        uint64_t    cacheKey;
        Texture*    tex;
        TextureType texType;
        bool        forceLinear;
        bool        preferSingleChannel;
        TextureHandle sceneHandle;
        std::string sceneKey;
    };
    std::vector<TexPrepReq> freshUploads;
    {
        std::unordered_set<uint64_t> seen;
        auto maybeQueue = [&](int64_t key, TextureType tt, bool fl, bool psc) {
            if (!key) return;
            if (key > 0 && static_cast<uint64_t>(key) < (1ull << 32)) return;
            const uint64_t ck = (static_cast<uint64_t>(key) << 2) |
                                (fl  ? 1ull : 0ull) |
                                (psc ? 2ull : 0ull);
            if (m_uploadedImageIDs.count(ck) || seen.count(ck)) return;
            Texture* t = reinterpret_cast<Texture*>(key);
            if (!t || !t->is_loaded() || t->vulkan_dirty) return;
            TextureHandle sceneHandle{};
            std::string sceneKey;
            if (m_sceneTextureManager) {
                sceneKey = buildSceneTextureKey(t, tt, fl, psc);
                sceneHandle = registerSceneTexture(
                    m_sceneTextureManager.get(),
                    t,
                    tt,
                    fl,
                    psc,
                    TextureConsumer::RasterPreview | TextureConsumer::VulkanRT);
                int64_t existingSceneTextureId = 0;
                VulkanRT::ImageHandle existingSceneImage{};
                if (sceneHandle.isValid() &&
                    m_sceneTextureManager->tryGetVulkanTextureId(sceneHandle, sceneTextureOwnerScope(), existingSceneTextureId) &&
                    existingSceneTextureId != 0 &&
                    tryGetUploadedImageHandle(existingSceneTextureId, existingSceneImage)) {
                    uint32_t desiredW = static_cast<uint32_t>(t->width);
                    uint32_t desiredH = static_cast<uint32_t>(t->height);
                    if (limitMaterialPreviewTextures) {
                        fitWithinMaxDimension(desiredW, desiredH, kMaterialPreviewTextureMaxDimension, desiredW, desiredH);
                    }
                    if (existingSceneImage.width == desiredW && existingSceneImage.height == desiredH) {
                        m_uploadedImageIDs[ck] = existingSceneTextureId;
                        m_textureIdToCacheKey[existingSceneTextureId] = ck;
                        return;
                    }
                }
            }
            seen.insert(ck);
            freshUploads.push_back({ck, t, tt, fl, psc, sceneHandle, sceneKey});
        };
        for (const auto& m : materials) {
            maybeQueue(m.albedoTexture,       TextureType::Albedo,       false, false);
            maybeQueue(m.normalTexture,       TextureType::Normal,       true,  false);
            maybeQueue(m.roughnessTexture,    TextureType::Roughness,    true,  false);
            maybeQueue(m.metallicTexture,     TextureType::Metallic,     true,  false);
            maybeQueue(m.specularTexture,     TextureType::Specular,     true,  true);
            maybeQueue(m.emissionTexture,     TextureType::Emission,     false, false);
            maybeQueue(m.transmissionTexture, TextureType::Transmission, true,  true);
            maybeQueue(m.opacityTexture,      TextureType::Opacity,      true,  true);
            maybeQueue(m.heightTexture,       TextureType::Unknown,      true,  false);
        }
    }

    // Phase 2: parallel CPU decode — pure pixel packing, no Vulkan calls
    struct StagedTexData {
        uint64_t         cacheKey       = 0;
        std::vector<uint8_t> bytes;        // raw bytes to upload (LDR or HDR cast)
        uint32_t         width          = 0;
        uint32_t         height         = 0;
        uint32_t         uploadChannels = 4;
        bool             useSrgb        = false;
        bool             isHdr          = false;
        TextureType      texType        = TextureType::Unknown;
        bool             preferSingle   = false;
    };
    std::vector<StagedTexData> staged(freshUploads.size());
    if (!freshUploads.empty()) {
        std::vector<std::future<void>> futs;
        futs.reserve(freshUploads.size());
        for (size_t i = 0; i < freshUploads.size(); ++i) {
            futs.push_back(std::async(std::launch::async, [&, i]() {
                const auto& req  = freshUploads[i];
                auto&       s    = staged[i];
                const Texture*   tex = req.tex;
                s.cacheKey       = req.cacheKey;
                s.width          = (uint32_t)tex->width;
                s.height         = (uint32_t)tex->height;
                s.isHdr          = tex->is_hdr;
                s.useSrgb        = req.forceLinear ? false : tex->is_srgb;
                s.texType        = req.texType;
                s.preferSingle   = req.preferSingleChannel;

                if (tex->is_hdr) {
                    const auto& fp = tex->float_pixels;
                    if (fp.empty()) return;
                    const size_t byteCount = fp.size() * sizeof(fp[0]);
                    s.bytes.resize(byteCount);
                    memcpy(s.bytes.data(), fp.data(), byteCount);
                    s.uploadChannels = 4;
                    return;
                }

                const TextureCompressionPlan plan = buildTextureCompressionPlan(tex, req.texType);
                const bool canSingle = (req.preferSingleChannel || plan.preferSingleChannelFallback)
                    && !s.useSrgb && tex->is_gray_scale && !tex->has_alpha;
                s.uploadChannels = canSingle ? 1u : 4u;
                const auto& px   = tex->pixels;
                s.bytes.resize((size_t)tex->width * tex->height * s.uploadChannels);
                if (canSingle) {
                    for (size_t j = 0; j < px.size(); ++j) s.bytes[j] = px[j].r;
                } else {
                    for (size_t j = 0; j < px.size(); ++j) {
                        s.bytes[j*4+0] = px[j].r; s.bytes[j*4+1] = px[j].g;
                        s.bytes[j*4+2] = px[j].b; s.bytes[j*4+3] = px[j].a;
                    }
                }
                if (limitMaterialPreviewTextures) {
                    uint32_t dstW = s.width;
                    uint32_t dstH = s.height;
                    fitWithinMaxDimension(s.width, s.height, kMaterialPreviewTextureMaxDimension, dstW, dstH);
                    if (dstW != s.width || dstH != s.height) {
                        s.bytes = resizeLdrBilinear(s.bytes, s.width, s.height, dstW, dstH, s.uploadChannels);
                        s.width = dstW;
                        s.height = dstH;
                    }
                }
            }));
        }
        for (auto& f : futs) f.get();
    }

    // Phase 3: batched GPU upload — single command buffer per chunk (caps staging RAM)
    if (!staged.empty()) {
        const auto& caps             = m_device->getCapabilities();
        // Flush the batch every ~256 MB of accumulated staging to cap host memory usage.
        const uint64_t FLUSH_BYTES   = 256ull << 20;
        uint64_t batchStagingBytes   = 0;

        beginBatchedTextureUpload();
        for (size_t i = 0; i < staged.size(); ++i) {
            auto& s = staged[i];
            if (s.bytes.empty()) continue;

            auto uploadStagedTexture = [&]() -> int64_t {
                if (s.isHdr) {
                    return uploadTexture2D(s.bytes.data(), s.width, s.height, 4, false, true);
                }

                const Texture* texPtr = freshUploads[i].tex;
                const TextureCompressionPlan plan = buildTextureCompressionPlan(texPtr, s.texType);
                const bool supportsComp =
                    (plan.preferredTarget == TextureCompressionTarget::BC4 && caps.supportsBC4) ||
                    (plan.preferredTarget == TextureCompressionTarget::BC5 && caps.supportsBC5) ||
                    (plan.preferredTarget == TextureCompressionTarget::BC7 && caps.supportsBC7);
                if (supportsComp && !texPtr->name.empty()) {
                    if (auto cand = findCompressedTextureCacheCandidate(*texPtr, s.texType, s.useSrgb)) {
                        DDSCompressedPayload payload{};
                        if (loadCompressedDDSFile(cand->ddsPath, cand->target, s.useSrgb, payload) &&
                            payload.width == s.width && payload.height == s.height) {
                            int64_t compressedId = uploadCompressedTexture2D(payload.bytes.data(), payload.bytes.size(),
                                                                              s.width, s.height, payload.format);
                            if (compressedId) {
                                return compressedId;
                            }
                        }
                    }
                }
                return uploadTexture2D(s.bytes.data(), s.width, s.height,
                                       s.uploadChannels, s.useSrgb, false);
            };

            int64_t id = 0;
            VulkanBackingRecord resolvedBacking{};
            bool resolvedThroughPool = false;
            bool createdByPool = false;
            const auto& req = freshUploads[i];
            if (m_sceneTextureManager && !req.sceneKey.empty()) {
                resolvedThroughPool = m_sceneTextureManager->resolveOrCreateVulkanBacking(
                    req.sceneKey,
                    sceneTextureOwnerScope(),
                    TextureConsumer::RasterPreview | TextureConsumer::VulkanRT,
                    s.width,
                    s.height,
                    estimateSceneTextureBytes(req.tex, s.uploadChannels),
                    [&](TextureHandle, VulkanBackingRecord& outBacking) -> bool {
                        const int64_t uploadedId = uploadStagedTexture();
                        if (!uploadedId) {
                            return false;
                        }
                        return buildVulkanBackingRecord(uploadedId, outBacking);
                    },
                    resolvedBacking,
                    &createdByPool);

                if (resolvedThroughPool && resolvedBacking.textureId != 0) {
                    VulkanRT::ImageHandle localImage{};
                    if (tryGetUploadedImageHandle(resolvedBacking.textureId, localImage)) {
                        if (localImage.width == s.width && localImage.height == s.height) {
                            id = resolvedBacking.textureId;
                        } else {
                            m_sceneTextureManager->clearVulkanBacking(sceneTextureOwnerScope(), resolvedBacking.textureId);
                            resolvedThroughPool = false;
                            createdByPool = false;
                        }
                    } else {
                        m_sceneTextureManager->clearVulkanBacking(sceneTextureOwnerScope(), resolvedBacking.textureId);
                        resolvedThroughPool = false;
                    }
                }
            }

            if (!id) {
                id = uploadStagedTexture();
            }

            if (id) {
                m_uploadedImageIDs[s.cacheKey] = id;
                m_textureIdToCacheKey[id] = s.cacheKey;
                if (!resolvedThroughPool || !createdByPool) {
                    registerSceneTextureUpload(req.sceneHandle, id);
                }
            }

            batchStagingBytes += (uint64_t)s.width * s.height * s.uploadChannels;
            if (batchStagingBytes >= FLUSH_BYTES && i + 1 < staged.size()) {
                endBatchedTextureUpload();
                beginBatchedTextureUpload();
                batchStagingBytes = 0;
            }
        }
        endBatchedTextureUpload();
    }
    // ─────────────────────────────────────────────────────────────────────────

    for (const auto& m : materials) {
        VulkanRT::VkGpuMaterial gm{};
        gm.albedo_r = m.albedo.x; gm.albedo_g = m.albedo.y; gm.albedo_b = m.albedo.z; gm.opacity = m.opacity;
        // ... (remaining fields)
        gm.roughness = m.roughness;
        gm.metallic = m.metallic;
        gm.specular = m.specular;
        gm.ior = m.ior;
        gm.transmission = m.transmission;
        gm.emission_r = m.emission.x; gm.emission_g = m.emission.y; gm.emission_b = m.emission.z;
        gm.emission_strength = m.emissionStrength;
        gm.subsurface_r = m.subsurfaceColor.x; gm.subsurface_g = m.subsurfaceColor.y; gm.subsurface_b = m.subsurfaceColor.z;
        gm.subsurface_amount = m.subsurface;
        gm.subsurface_radius_r = m.subsurfaceRadius.x; gm.subsurface_radius_g = m.subsurfaceRadius.y; gm.subsurface_radius_b = m.subsurfaceRadius.z;
        gm.subsurface_scale = m.subsurfaceScale;
        gm.subsurface_ior = m.subsurfaceIOR;
        gm.normal_strength = m.normalStrength;
        // SSS control flags (match VkGpuMaterial layout Block 12)
       
        gm.clearcoat = m.clearcoat;
        gm.clearcoat_roughness = m.clearcoatRoughness;
        gm.translucent = m.translucent;
        gm.subsurface_anisotropy = m.subsurfaceAnisotropy;
        gm.anisotropic = m.anisotropic;
        gm.sheen = m.sheen;
        gm.sheen_tint = m.sheenTint;
        gm.flags = (uint32_t)m.flags;
        gm.bubble_ior = m.bubble_ior;
        gm.bubble_film = m.bubble_film;
        gm.clearcoat_iridescence = m.clearcoat_iridescence;
        gm.clearcoat_film_thickness = m.clearcoat_film_thickness;
        if (m.is_bubble) gm.flags |= VulkanRT::VK_MAT_FLAG_BUBBLE;
        gm.resin_color_r = static_cast<float>(m.resin_color.x);
        gm.resin_color_g = static_cast<float>(m.resin_color.y);
        gm.resin_color_b = static_cast<float>(m.resin_color.z);
        gm.transmission_density = m.transmission_density;
        gm.resin_roughness = m.resin_roughness;
        gm.dispersion = m.dispersion;
        gm.resin_inclusion = m.resin_inclusion;
        gm.resin_dirt = m.resin_dirt;
        gm.resin_inclusion_scale = m.resin_inclusion_scale;
        gm.resin_dirt_color_r = static_cast<float>(m.resin_dirt_color.x);
        gm.resin_dirt_color_g = static_cast<float>(m.resin_dirt_color.y);
        gm.resin_dirt_color_b = static_cast<float>(m.resin_dirt_color.z);
        gm.resin_shard = m.resin_shard;
        gm.resin_shard_hue = m.resin_shard_hue;
        gm.dust_color_a_r = static_cast<float>(m.dust_color_a.x);
        gm.dust_color_a_g = static_cast<float>(m.dust_color_a.y);
        gm.dust_color_a_b = static_cast<float>(m.dust_color_a.z);
        gm.dust_style = static_cast<float>(m.dust_style);
        gm.dust_color_b_r = static_cast<float>(m.dust_color_b.x);
        gm.dust_color_b_g = static_cast<float>(m.dust_color_b.y);
        gm.dust_color_b_b = static_cast<float>(m.dust_color_b.z);
        gm.shard_shape = static_cast<float>(m.shard_shape);
        if (m.resin_object_space) gm.flags |= VulkanRT::VK_MAT_FLAG_RESIN_OBJ_SPACE;
        if (m.glass_marble_volume) gm.flags |= VulkanRT::VK_MAT_FLAG_MARBLE_VOLUME;
        gm.uv_scale_x = static_cast<float>(m.uvScale.x);
        gm.uv_scale_y = static_cast<float>(m.uvScale.y);
        gm.uv_offset_x = static_cast<float>(m.uvOffset.x);
        gm.uv_offset_y = static_cast<float>(m.uvOffset.y);
        gm.uv_rotation_degrees = m.uvRotationDegrees;
        gm.uv_tiling_x = static_cast<float>(m.uvTiling.x);
        gm.uv_tiling_y = static_cast<float>(m.uvTiling.y);
        gm.uv_wrap_mode = m.uvWrapMode;
        // If this is a terrain material, embed the terrain layer buffer index
        if (m.flags & Backend::IBackend::MAT_FLAG_TERRAIN) {
            gm._terrain_layer_idx = m.terrainLayerIdx;
        }
        // Procedural detail params
        gm.micro_detail_strength = m.micro_detail_strength;
        gm.micro_detail_scale    = m.micro_detail_scale;
        gm.tile_break_strength   = m.tile_break_strength;
        // Water-specific params → VkGpuMaterial Block 8 & Block 9
        gm.fft_amplitude         = m.fft_amplitude;
        gm.fft_time_scale        = m.fft_time_scale;
        gm.foam_threshold        = m.foam_threshold;
        gm.fft_ocean_size        = m.fft_ocean_size;
        gm.fft_choppiness        = m.fft_choppiness;
        gm.fft_wind_speed        = m.fft_wind_speed;
        gm.fft_wind_direction    = m.fft_wind_direction;
        gm.micro_anim_speed      = m.micro_anim_speed;
        gm.micro_morph_speed     = m.micro_morph_speed;
        gm.foam_noise_scale      = m.foam_noise_scale;
        // Backend-owned texture handles (e.g. FFT ocean height/normal returned by
        // uploadTexture2D) are small integer ids, NOT host Texture* pointers. The
        // flag blocks below need a host Texture* to inspect cache/format state, so
        // filter handles out instead of dereferencing them as pointers.
        auto resolveHostTexture = [](int64_t key) -> Texture* {
            if (!key) return nullptr;
            if (key > 0 && static_cast<uint64_t>(key) < (1ull << 32)) return nullptr;
            return reinterpret_cast<Texture*>(key);
        };

        // ... getTexID mapping (cast to uint32_t for GLSL compatibility)
        auto getTexID = [this](int64_t key, TextureType textureType, bool forceLinear = false, bool preferSingleChannel = false) -> uint32_t {
            if (!key) return 0;
            VulkanRT::ImageHandle existingUploaded{};
            if (tryGetUploadedImageHandle(key, existingUploaded)) {
                return static_cast<uint32_t>(key);
            }
            // Runtime backend-owned texture handles are small integer ids, not real Texture* pointers.
            // If such a handle is stale/missing from the uploaded-image table, fail safely instead of
            // treating it as a host pointer and dereferencing invalid memory.
            if (key > 0 && static_cast<uint64_t>(key) < (1ull << 32)) {
                return 0;
            }
            uint64_t cacheKey = (static_cast<uint64_t>(key) << 2) |
                                (forceLinear ? 1ull : 0ull) |
                                (preferSingleChannel ? 2ull : 0ull);
            Texture* texCheck = reinterpret_cast<Texture*>(key);
            const bool needsDirtyRefresh = (texCheck && texCheck->vulkan_dirty);
            TextureHandle sceneHandle{};
            if (texCheck && texCheck->is_loaded() && m_sceneTextureManager && !needsDirtyRefresh) {
                sceneHandle = registerSceneTexture(
                    m_sceneTextureManager.get(),
                    texCheck,
                    textureType,
                    forceLinear,
                    preferSingleChannel,
                    TextureConsumer::RasterPreview | TextureConsumer::VulkanRT);
                int64_t existingSceneTextureId = 0;
                if (sceneHandle.isValid() &&
                    m_sceneTextureManager->tryGetVulkanTextureId(sceneHandle, sceneTextureOwnerScope(), existingSceneTextureId) &&
                    existingSceneTextureId != 0 &&
                    tryGetUploadedImageHandle(existingSceneTextureId, existingUploaded)) {
                    uint32_t desiredW = static_cast<uint32_t>(std::max(0, texCheck->width));
                    uint32_t desiredH = static_cast<uint32_t>(std::max(0, texCheck->height));
                    if (isViewportTextureOwner(sceneTextureOwnerScope())) {
                        fitWithinMaxDimension(desiredW, desiredH,
                                              kMaterialPreviewTextureMaxDimension,
                                              desiredW, desiredH);
                    }
                    if (existingUploaded.width == desiredW && existingUploaded.height == desiredH) {
                        m_uploadedImageIDs[cacheKey] = existingSceneTextureId;
                        m_textureIdToCacheKey[existingSceneTextureId] = cacheKey;
                        return static_cast<uint32_t>(existingSceneTextureId);
                    }
                }
            }
            // [FIX] vulkan_dirty: texture content was updated in-place (paint stroke,
            // autoMask, paintSplatMap, etc.). Prefer updateTexture2DRegion when the
            // texture tracked exactly which pixels changed (brush dab) — only the
            // dirty rect is staged + copied, avoiding the fullWidth*fullHeight*bpp
            // temp buffer that dominated paint cost at 2K/4K. Fall back to
            // updateTexture2DInPlace (full image, same VkImage slot) when no rect
            // is tracked or the partial path can't be used. Only fall back to
            // destroy+upload when in-place fails too.
            if (texCheck && texCheck->is_loaded() && !texCheck->is_hdr && needsDirtyRefresh) {
                auto oldIt = m_uploadedImageIDs.find(cacheKey);
                if (oldIt != m_uploadedImageIDs.end()) {
                    const int64_t existingId = oldIt->second;
                    const std::vector<CompactVec4>& dpx = texCheck->pixels;
                    if (existingId > 0 && !dpx.empty()) {
                        const bool dUseSrgb = forceLinear ? false : texCheck->is_srgb;
                        const TextureCompressionPlan dPlan = buildTextureCompressionPlan(texCheck, textureType);
                        const bool dCanUseSingleChannel =
                            (preferSingleChannel || dPlan.preferSingleChannelFallback) &&
                            !dUseSrgb && texCheck->is_gray_scale && !texCheck->has_alpha;
                        const bool dCanUseTwoChannels = (textureType == TextureType::Normal) && !texCheck->is_hdr;
                        const uint32_t dUploadChannels = dCanUseSingleChannel ? 1u : (dCanUseTwoChannels ? 2u : 4u);

                        // Try the partial-region fast path when a rect is known.
                        int rx = 0, ry = 0, rw = 0, rh = 0;
                        bool rfull = true;
                        const bool haveRect = texCheck->getVulkanDirtyRegion(rx, ry, rw, rh, rfull);
                        if (haveRect && !rfull && rw > 0 && rh > 0) {
                            std::vector<uint8_t> region(static_cast<size_t>(rw) *
                                                        static_cast<size_t>(rh) *
                                                        dUploadChannels);
                            const int srcW = texCheck->width;
                            if (dCanUseSingleChannel) {
                                for (int j = 0; j < rh; ++j) {
                                    const size_t srcRow = static_cast<size_t>(ry + j) * static_cast<size_t>(srcW);
                                    const size_t dstRow = static_cast<size_t>(j) * static_cast<size_t>(rw);
                                    for (int i = 0; i < rw; ++i) {
                                        region[dstRow + i] = dpx[srcRow + (rx + i)].r;
                                    }
                                }
                            } else if (dCanUseTwoChannels) {
                                for (int j = 0; j < rh; ++j) {
                                    const size_t srcRow = static_cast<size_t>(ry + j) * static_cast<size_t>(srcW);
                                    const size_t dstRow = static_cast<size_t>(j) * static_cast<size_t>(rw) * 2;
                                    for (int i = 0; i < rw; ++i) {
                                        const auto& p = dpx[srcRow + (rx + i)];
                                        const size_t o = dstRow + static_cast<size_t>(i) * 2;
                                        region[o + 0] = p.r;
                                        region[o + 1] = p.g;
                                    }
                                }
                            } else {
                                for (int j = 0; j < rh; ++j) {
                                    const size_t srcRow = static_cast<size_t>(ry + j) * static_cast<size_t>(srcW);
                                    const size_t dstRow = static_cast<size_t>(j) * static_cast<size_t>(rw) * 4;
                                    for (int i = 0; i < rw; ++i) {
                                        const auto& p = dpx[srcRow + (rx + i)];
                                        const size_t o = dstRow + static_cast<size_t>(i) * 4;
                                        region[o + 0] = p.r;
                                        region[o + 1] = p.g;
                                        region[o + 2] = p.b;
                                        region[o + 3] = p.a;
                                    }
                                }
                            }
                            if (this->updateTexture2DRegion(existingId, region.data(),
                                                            texCheck->width, texCheck->height,
                                                            dUploadChannels, dUseSrgb,
                                                            rx, ry,
                                                            static_cast<uint32_t>(rw),
                                                            static_cast<uint32_t>(rh))) {
                                texCheck->clearVulkanDirty();
                                return (uint32_t)existingId;
                            }
                            // Region path failed (slot mismatch, format diverged).
                            // Fall through to the full in-place path below.
                        }

                        std::vector<uint8_t> dtmp((size_t)texCheck->width * texCheck->height * dUploadChannels);
                        if (dCanUseSingleChannel) {
                            for (size_t i = 0; i < dpx.size(); ++i) dtmp[i] = dpx[i].r;
                        } else if (dCanUseTwoChannels) {
                            for (size_t i = 0; i < dpx.size(); ++i) {
                                dtmp[i*2+0] = dpx[i].r; dtmp[i*2+1] = dpx[i].g;
                            }
                        } else {
                            for (size_t i = 0; i < dpx.size(); ++i) {
                                dtmp[i*4+0] = dpx[i].r; dtmp[i*4+1] = dpx[i].g;
                                dtmp[i*4+2] = dpx[i].b; dtmp[i*4+3] = dpx[i].a;
                            }
                        }
                        if (this->updateTexture2DInPlace(existingId, dtmp.data(),
                                                         texCheck->width, texCheck->height,
                                                         dUploadChannels, dUseSrgb, false)) {
                            texCheck->clearVulkanDirty();
                            return (uint32_t)existingId;
                        }
                    }
                }
                // In-place didn't apply (missing slot, dim change, compressed-only
                // cache entry…). Evict and continue to the destroy+upload path.
                auto evictIt = m_uploadedImageIDs.find(cacheKey);
                if (evictIt != m_uploadedImageIDs.end()) {
                    int64_t oldId = evictIt->second;
                    if (oldId) this->destroyTexture(oldId);
                }
                texCheck->clearVulkanDirty();
            }
            if (texCheck && texCheck->is_loaded() && m_sceneTextureManager && !sceneHandle.isValid()) {
                sceneHandle = registerSceneTexture(
                    m_sceneTextureManager.get(),
                    texCheck,
                    textureType,
                    forceLinear,
                    preferSingleChannel,
                    TextureConsumer::RasterPreview | TextureConsumer::VulkanRT);
            }
            auto it = m_uploadedImageIDs.find(cacheKey);
            if (it != m_uploadedImageIDs.end()) return (uint32_t)it->second;
            Texture* tex = texCheck;
            if (!tex || !tex->is_loaded()) return 0;
            const bool useSrgb = forceLinear ? false : tex->is_srgb;
            const TextureCompressionPlan compressionPlan = buildTextureCompressionPlan(tex, textureType);
            const bool limitThisTextureForPreview = isViewportTextureOwner(sceneTextureOwnerScope());
            uint32_t uploadWidth = static_cast<uint32_t>(std::max(0, tex->width));
            uint32_t uploadHeight = static_cast<uint32_t>(std::max(0, tex->height));
            if (limitThisTextureForPreview) {
                fitWithinMaxDimension(uploadWidth, uploadHeight,
                                      kMaterialPreviewTextureMaxDimension,
                                      uploadWidth, uploadHeight);
            }
            auto uploadTextureNow = [&]() -> int64_t {
                if (tex->is_hdr) {
                    // HDR path: no in-place fast path yet (rarely painted in practice).
                    // vulkan_dirty HDR textures simply re-upload here after the eviction
                    // above cleared the stale cache entry.
                    const std::vector<float4>& fp = tex->float_pixels;
                    if (fp.empty()) return 0;
                    return this->uploadTexture2D(fp.data(), tex->width, tex->height, 4, false, true);
                }

                // Use the dirty snapshot captured at function entry
                // (`needsDirtyRefresh`, line 7113) — by the time we reach this
                // lambda the in-place fast path above has already called
                // clearVulkanDirty() at line 7222 when it failed (which is
                // exactly the BC/DDS-backed case we care about), so reading
                // tex->vulkan_dirty here would always see false.
                // If the user just painted on this texture, `pixels` holds the
                // new content and any on-disk DDS cache is stale (it was
                // written before the stroke). Skipping the cache forces
                // re-encoding from `pixels`, which is what makes paint visible
                // in raster + Vulkan-RT. CPU/OptiX worked already because
                // they never consult the cache.
                const bool hasPendingPaintEdits = needsDirtyRefresh;
                if (tex->vulkan_dirty) tex->clearVulkanDirty();

                // Wipe the stale managed DDS cache for this texture on the
                // first dirty transition. Without this, closing and reopening
                // the project would replay the pre-paint compressed bytes,
                // silently reverting the stroke. Idempotent: subsequent
                // strokes hit the in-place fast path and never reach this
                // lambda, so file I/O stays one-shot per painted texture.
                // Adjacent user-supplied DDS files are not touched.
                if (hasPendingPaintEdits && !tex->name.empty()) {
                    const size_t removed = invalidateManagedTextureCacheForTexture(*tex);
                    if (removed > 0) {
                        SCENE_LOG_INFO("[Vulkan] Invalidated " + std::to_string(removed) +
                                       " stale DDS cache file(s) for painted texture: " + tex->name);
                    }
                }
                const std::vector<CompactVec4>& px = tex->pixels;
                if (px.empty()) return 0;

                const bool supportsPreferredCompression =
                    (compressionPlan.preferredTarget == TextureCompressionTarget::BC4 && m_device->getCapabilities().supportsBC4) ||
                    (compressionPlan.preferredTarget == TextureCompressionTarget::BC5 && m_device->getCapabilities().supportsBC5) ||
                    (compressionPlan.preferredTarget == TextureCompressionTarget::BC7 && m_device->getCapabilities().supportsBC7);
                if (supportsPreferredCompression && !tex->name.empty() && !hasPendingPaintEdits &&
                    !limitThisTextureForPreview) {
                    DDSCompressedPayload payload{};
                    if (auto cacheCandidate = findCompressedTextureCacheCandidate(*tex, textureType, useSrgb)) {
                        if (loadCompressedDDSFile(cacheCandidate->ddsPath, cacheCandidate->target, useSrgb, payload) &&
                            payload.width == static_cast<uint32_t>(tex->width) &&
                            payload.height == static_cast<uint32_t>(tex->height)) {
                            int64_t compressedId = this->uploadCompressedTexture2D(
                                payload.bytes.data(),
                                payload.bytes.size(),
                                payload.width,
                                payload.height,
                                payload.format);
                            if (compressedId) {
                                return compressedId;
                            }
                        }
                    }
                }

                const bool canUseSingleChannel =
                    (preferSingleChannel || compressionPlan.preferSingleChannelFallback) &&
                    !useSrgb && tex->is_gray_scale && !tex->has_alpha;
                const bool canUseTwoChannels = (textureType == TextureType::Normal) && !tex->is_hdr;
                std::vector<uint8_t> tmp;
                const uint32_t uploadChannels = canUseSingleChannel ? 1u : (canUseTwoChannels ? 2u : 4u);
                tmp.resize((size_t)tex->width * tex->height * uploadChannels);
                if (canUseSingleChannel) {
                    for (size_t i = 0; i < px.size(); ++i) {
                        tmp[i] = px[i].r;
                    }
                } else if (canUseTwoChannels) {
                    for (size_t i = 0; i < px.size(); ++i) {
                        tmp[i*2 + 0] = px[i].r; tmp[i*2 + 1] = px[i].g;
                    }
                } else {
                    for (size_t i = 0; i < px.size(); ++i) {
                        tmp[i*4 + 0] = px[i].r; tmp[i*4 + 1] = px[i].g; tmp[i*4 + 2] = px[i].b; tmp[i*4 + 3] = px[i].a;
                    }
                }
                if (uploadWidth != static_cast<uint32_t>(tex->width) ||
                    uploadHeight != static_cast<uint32_t>(tex->height)) {
                    tmp = resizeLdrBilinear(tmp,
                                           static_cast<uint32_t>(tex->width),
                                           static_cast<uint32_t>(tex->height),
                                           uploadWidth,
                                           uploadHeight,
                                           uploadChannels);
                }
                return this->uploadTexture2D(tmp.data(), uploadWidth, uploadHeight, uploadChannels, useSrgb, false);
            };

            const bool canUseSingleChannel =
                (preferSingleChannel || compressionPlan.preferSingleChannelFallback) &&
                !useSrgb && tex->is_gray_scale && !tex->has_alpha;
            const uint32_t estimatedUploadChannels = tex->is_hdr ? 4u : (canUseSingleChannel ? 1u : 4u);
            int64_t id = 0;
            bool resolvedThroughPool = false;
            bool createdByPool = false;
            if (m_sceneTextureManager && sceneHandle.isValid()) {
                VulkanBackingRecord resolvedBacking{};
                resolvedThroughPool = m_sceneTextureManager->resolveOrCreateVulkanBacking(
                    buildSceneTextureKey(tex, textureType, forceLinear, preferSingleChannel),
                    sceneTextureOwnerScope(),
                    TextureConsumer::RasterPreview | TextureConsumer::VulkanRT,
                    uploadWidth,
                    uploadHeight,
                    tex->is_hdr
                        ? estimateSceneTextureBytes(tex, estimatedUploadChannels)
                        : static_cast<uint64_t>(uploadWidth) * uploadHeight * estimatedUploadChannels,
                    [&](TextureHandle, VulkanBackingRecord& outBacking) -> bool {
                        const int64_t uploadedId = uploadTextureNow();
                        if (!uploadedId) {
                            return false;
                        }
                        return buildVulkanBackingRecord(uploadedId, outBacking);
                    },
                    resolvedBacking,
                    &createdByPool);

                if (resolvedThroughPool && resolvedBacking.textureId != 0) {
                    VulkanRT::ImageHandle localImage{};
                    if (tryGetUploadedImageHandle(resolvedBacking.textureId, localImage)) {
                        if (localImage.width == uploadWidth && localImage.height == uploadHeight) {
                            id = resolvedBacking.textureId;
                        } else {
                            m_sceneTextureManager->clearVulkanBacking(sceneTextureOwnerScope(), resolvedBacking.textureId);
                            resolvedThroughPool = false;
                            createdByPool = false;
                        }
                    } else {
                        m_sceneTextureManager->clearVulkanBacking(sceneTextureOwnerScope(), resolvedBacking.textureId);
                        resolvedThroughPool = false;
                    }
                }
            }

            if (!id) {
                id = uploadTextureNow();
            }
            if (id) {
                m_uploadedImageIDs[cacheKey] = id;
                m_textureIdToCacheKey[id] = cacheKey;
                if (!resolvedThroughPool || !createdByPool) {
                    registerSceneTextureUpload(sceneHandle, id);
                }
                return (uint32_t)id;
            }
            return 0;
        };

        gm.albedo_tex = getTexID(m.albedoTexture, TextureType::Albedo, false);
        gm.normal_tex = getTexID(m.normalTexture, TextureType::Normal, true);
        gm.roughness_tex = getTexID(m.roughnessTexture, TextureType::Roughness, true, false);
        gm.metallic_tex = getTexID(m.metallicTexture, TextureType::Metallic, true, false);
        gm.specular_tex = getTexID(m.specularTexture, TextureType::Specular, true, true);
        gm.emission_tex = getTexID(m.emissionTexture, TextureType::Emission, false);
        gm.transmission_tex = getTexID(m.transmissionTexture, TextureType::Transmission, true, true);
        gm.opacity_tex = getTexID(m.opacityTexture, TextureType::Opacity, true, true);
        // Bit 8: shader reads .a (RGBA source) vs .r (grayscale or BC4 DDS).
        // After baking, BC4 stores the original .a data in its single R channel,
        // so bit 8 must be cleared — otherwise the shader reads .a from a 1-channel
        // texture and always gets 1.0 (fully opaque).
        if (m.opacityTexture) {
            Texture* opTex = resolveHostTexture(m.opacityTexture);
            if (opTex && opTex->is_loaded() && opTex->has_alpha) {
                const bool hasBc4Dds = [&]() -> bool {
                    auto cand = findCompressedTextureCacheCandidate(*opTex, TextureType::Opacity, false);
                    return cand && cand->target == TextureCompressionTarget::BC4;
                }();
                if (!hasBc4Dds) {
                    gm.flags |= (1u << 8); // Bit 8: opacity is in .a channel
                }
            }
        }

        // Bits 9/10: shader reads .r (BC4 cache or single-channel R8 upload) vs
        // the default ORM packed channels (.g for roughness, .b for metallic).
        // Roughness/Metallic plan now ALWAYS targets BC4 (see TextureCompression.h),
        // so as soon as a BC4 cache exists the shader must read .r — regardless of
        // whether the source was grayscale or ORM-packed (the BC4 file holds the
        // single channel that the plan chose to extract).
        // Without a cache we fall back to R8 (grayscale, no alpha) or RGBA8 (ORM):
        //   R8   → data is in .r → flag set
        //   RGBA8 → data is in .g/.b (original ORM) → flag clear
        auto needsRedChannelFlag = [](Texture* tex, TextureType type) -> bool {
            if (!tex || !tex->is_loaded()) return false;
            // Cache hit on BC4 always wins — the file contains a single R channel
            // built from the planned sourceChannel.
            if (auto cand = findCompressedTextureCacheCandidate(*tex, type, false)) {
                if (cand->target == TextureCompressionTarget::BC4) return true;
            }
            // No BC4 cache: shader sees whatever the live upload path produces.
            // R8 single-channel upload requires grayscale + no alpha (matches the
            // canSingle gate in the upload pipeline).
            return tex->is_gray_scale && !tex->has_alpha;
        };
        if (m.roughnessTexture) {
            Texture* rTex = resolveHostTexture(m.roughnessTexture);
            if (needsRedChannelFlag(rTex, TextureType::Roughness)) {
                gm.flags |= (1u << 9);
            }
        }
        if (m.metallicTexture) {
            Texture* mTex = resolveHostTexture(m.metallicTexture);
            if (needsRedChannelFlag(mTex, TextureType::Metallic)) {
                gm.flags |= (1u << 10);
            }
        }
        // Bits 12-13 / 14-15: explicit user channel override for the metallic /
        // roughness packed textures (0 = Auto, 1 = R, 2 = G, 3 = B). Overrides
        // the auto policy above in the shader — non-ORM packings (RMA/MRA,
        // metal-in-R maps) otherwise read the wrong channel and shade as metal.
        gm.flags |= ((uint32_t)std::clamp(m.metallicTexChannel,  0, 3)) << 12;
        gm.flags |= ((uint32_t)std::clamp(m.roughnessTexChannel, 0, 3)) << 14;

        // Bit 11: normal map is BC5-encoded. BC5 only stores RG (XY); the shader
        // must reconstruct Z = sqrt(1 - X² - Y²) instead of reading .b (which is
        // always 0 in BC5). Without this flag, large 4K/8K normal maps that go
        // through the BC5 cache decode to (X, Y, -1) → broken/black surface.
        if (m.normalTexture) {
            Texture* nTex = resolveHostTexture(m.normalTexture);
            if (nTex && nTex->is_loaded()) {
                bool isBC5 = false;
                if (auto cand = findCompressedTextureCacheCandidate(*nTex, TextureType::Normal, false)) {
                    if (cand->target == TextureCompressionTarget::BC5) {
                        isBC5 = true;
                    }
                }
                const bool isRG8 = !nTex->is_hdr; // uncompressed LDR normal maps uploaded as RG8 (2 channels)
                if (isBC5 || isRG8) {
                    gm.flags |= (1u << 11);
                }
            }
        }
        gm.height_tex = getTexID(m.heightTexture, TextureType::Unknown, true);

        gpuMats.push_back(gm);
    }

    if (gpuMats.empty()) {
        VulkanRT::VkGpuMaterial defaultMat{};
        defaultMat.albedo_r = 0.8f; defaultMat.albedo_g = 0.8f; defaultMat.albedo_b = 0.8f;
        defaultMat.opacity = 1.0f;
        defaultMat.roughness = 0.5f;
        defaultMat.specular = 0.5f;
        gpuMats.push_back(defaultMat);
    }

    m_cachedGpuMaterials = gpuMats;
    m_device->updateMaterialBuffer(gpuMats.data(), gpuMats.size() * sizeof(::VulkanRT::VkGpuMaterial), (uint32_t)gpuMats.size());

    // Update material preview descriptor set when material buffer changes
    if (m_interactiveViewport.materialPreviewDescSet != VK_NULL_HANDLE &&
        m_device->m_materialBuffer.buffer != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo matBufInfo{};
        matBufInfo.buffer = m_device->m_materialBuffer.buffer;
        matBufInfo.offset = 0;
        matBufInfo.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet mpWds{};
        mpWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        mpWds.dstSet = m_interactiveViewport.materialPreviewDescSet;
        mpWds.dstBinding = 0;
        mpWds.descriptorCount = 1;
        mpWds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        mpWds.pBufferInfo = &matBufInfo;
        vkUpdateDescriptorSets(m_device->getDevice(), 1, &mpWds, 0, nullptr);
    }

    // Re-upload per-vertex matId buffers from the CPU shadow so that any material
    // reassignment that happened between buildRasterGeometry and uploadMaterials is
    // reflected immediately (fixes startup material mismatch: wrong object → material).
    for (auto& [key, rmb] : m_rasterMeshes) {
        if (!rmb.cpuMatIds.empty() && rmb.matIdBuffer.buffer) {
            m_device->uploadBuffer(rmb.matIdBuffer,
                                   rmb.cpuMatIds.data(),
                                   rmb.cpuMatIds.size() * sizeof(uint32_t), 0);
        }
    }

    if (m_textureUploadSummaryDirty) {
        SCENE_LOG_INFO("[Vulkan] Texture upload summary | new=" + std::to_string(m_textureUploadCount) +
                       " | bc4=" + std::to_string(m_textureUploadBC4Count) +
                       " | bc5=" + std::to_string(m_textureUploadBC5Count) +
                       " | bc7=" + std::to_string(m_textureUploadBC7Count) +
                       " | r8=" + std::to_string(m_textureUploadR8Count) +
                       " | rg8=" + std::to_string(m_textureUploadRG8Count) +
                       " | rgba8=" + std::to_string(m_textureUploadRGBA8Count) +
                       " | float=" + std::to_string(m_textureUploadFloatCount));
        if (m_sceneTextureManager) {
            m_sceneTextureManager->logBudgetSummary("after-upload-materials");
        }
        m_textureUploadSummaryDirty = false;
    }

    // ── BLAS opacity audit ──────────────────────────────────────────────────
    // VK_GEOMETRY_OPAQUE_BIT is baked into each BLAS from the materials as they
    // were AT BUILD TIME. A material edit that turns transmission on (glass) or
    // off left the BLAS stale — colored/transmissive shadows only woke up after
    // a backend switch forced a full re-upload. Instead of rebuilding the BLAS,
    // flip the per-instance FORCE_(NO_)OPAQUE override and refresh the TLAS —
    // the same cheap operation as moving an object.
    {
        std::unordered_map<uint32_t, uint8_t> desired; // blasIndex -> override
        for (const auto& bm : m_blasMaterialBounds) {
            auto builtIt = m_blasBuiltNonOpaque.find(bm.first);
            if (builtIt == m_blasBuiltNonOpaque.end()) continue;
            bool wantNonOpaque = false;
            for (const auto& e : bm.second) {
                if (!materialCanUseOpaqueFastPath(e.first)) { wantNonOpaque = true; break; }
            }
            desired[bm.first] = (wantNonOpaque == builtIt->second)
                ? (uint8_t)0 : (wantNonOpaque ? (uint8_t)2 : (uint8_t)1);
        }
        bool anyFlip = false;
        for (auto& vi : m_vkInstances) {
            auto dIt = desired.find(vi.blasIndex);
            if (dIt == desired.end()) continue;
            if (vi.opacityOverride != dIt->second) {
                vi.opacityOverride = dIt->second;
                anyFlip = true;
            }
        }
        if (anyFlip) {
            auto merged = m_vkInstances;
            for (const auto& h : m_hairVkInstances) merged.push_back(h);
            if (!merged.empty()) m_device->updateTLAS(merged);
            SCENE_LOG_INFO("[Vulkan] Material sync flipped BLAS opacity override(s) — TLAS refreshed");
        }
    }
    resetAccumulation();
    // Material SSBO and/or texture descriptors have just been rewritten. The
    // interactive raster viewport caches its last framebuffer and only re-renders
    // on dirty — without this flag, paint strokes refresh the texture memory but
    // Material Preview keeps presenting the stale cached frame (visible only when
    // render device is Vulkan RT, since the CPU/OptiX path uses a dedicated
    // VulkanViewportBackend that gets its own sync).
    m_interactiveViewport.dirty = true;
}

bool VulkanBackendAdapter::updateMaterial(uint32_t materialIndex, const MaterialData& m) {
    if (!m_device || !m_device->isInitialized()) return false;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    if (!m_device->m_materialBuffer.buffer ||
        materialIndex >= m_cachedGpuMaterials.size() ||
        (uint64_t)(materialIndex + 1) * sizeof(VulkanRT::VkGpuMaterial) > m_device->m_materialBuffer.size) {
        return false;
    }

    auto resolveHostTexture = [](int64_t key) -> Texture* {
        if (!key) return nullptr;
        if (key > 0 && static_cast<uint64_t>(key) < (1ull << 32)) return nullptr;
        return reinterpret_cast<Texture*>(key);
    };

    auto syncExistingTexture = [&](int64_t hostHandle,
                                   uint32_t& gpuTextureId,
                                   TextureType textureType,
                                   bool forceLinear,
                                   bool preferSingleChannel = false) -> bool {
        if (!hostHandle) {
            gpuTextureId = 0;
            return true;
        }

        Texture* tex = resolveHostTexture(hostHandle);
        if (!tex || !tex->is_loaded()) {
            gpuTextureId = 0;
            return true;
        }

        const uint64_t cacheKey = (static_cast<uint64_t>(hostHandle) << 2) |
                                  (forceLinear ? 1ull : 0ull) |
                                  (preferSingleChannel ? 2ull : 0ull);
        auto cacheIt = m_uploadedImageIDs.find(cacheKey);
        if (cacheIt != m_uploadedImageIDs.end() && cacheIt->second != 0) {
            gpuTextureId = static_cast<uint32_t>(cacheIt->second);
        } else if (tex->vulkan_dirty) {
            // The material now references a host Texture* that this Vulkan
            // backend has not uploaded yet. This happens on the first mesh-paint
            // dab after bindTextureSetToMaterial() swaps the material from its
            // source texture to the generated paint texture. Falling through to
            // the old cached GPU slot would update the wrong image; force the
            // caller to rebuild the material table so getTexID() uploads and
            // binds the new pointer.
            return false;
        }

        if (gpuTextureId == 0) {
            return false;
        }

        if (!tex->vulkan_dirty) {
            return true;
        }

        if (tex->is_hdr) {
            return false;
        }

        const auto& pixels = tex->pixels;
        if (pixels.empty()) {
            return false;
        }

        const bool useSrgb = forceLinear ? false : tex->is_srgb;
        const TextureCompressionPlan compressionPlan = buildTextureCompressionPlan(tex, textureType);
        const bool canUseSingleChannel =
            (preferSingleChannel || compressionPlan.preferSingleChannelFallback) &&
            !useSrgb && tex->is_gray_scale && !tex->has_alpha;
        const bool canUseTwoChannels = (textureType == TextureType::Normal) && !tex->is_hdr;
        const uint32_t uploadChannels = canUseSingleChannel ? 1u : (canUseTwoChannels ? 2u : 4u);

        int rx = 0, ry = 0, rw = 0, rh = 0;
        bool fullRegion = true;
        const bool haveDirtyRect = tex->getVulkanDirtyRegion(rx, ry, rw, rh, fullRegion);
        if (haveDirtyRect && !fullRegion && rw > 0 && rh > 0) {
            std::vector<uint8_t> region(static_cast<size_t>(rw) *
                                        static_cast<size_t>(rh) *
                                        static_cast<size_t>(uploadChannels));
            const int srcW = tex->width;
            if (canUseSingleChannel) {
                for (int j = 0; j < rh; ++j) {
                    const size_t srcRow = static_cast<size_t>(ry + j) * static_cast<size_t>(srcW);
                    const size_t dstRow = static_cast<size_t>(j) * static_cast<size_t>(rw);
                    for (int i = 0; i < rw; ++i) {
                        region[dstRow + static_cast<size_t>(i)] = pixels[srcRow + static_cast<size_t>(rx + i)].r;
                    }
                }
            } else if (canUseTwoChannels) {
                for (int j = 0; j < rh; ++j) {
                    const size_t srcRow = static_cast<size_t>(ry + j) * static_cast<size_t>(srcW);
                    const size_t dstRow = static_cast<size_t>(j) * static_cast<size_t>(rw) * 2u;
                    for (int i = 0; i < rw; ++i) {
                        const auto& pixel = pixels[srcRow + static_cast<size_t>(rx + i)];
                        const size_t dstOffset = dstRow + static_cast<size_t>(i) * 2u;
                        region[dstOffset + 0] = pixel.r;
                        region[dstOffset + 1] = pixel.g;
                    }
                }
            } else {
                for (int j = 0; j < rh; ++j) {
                    const size_t srcRow = static_cast<size_t>(ry + j) * static_cast<size_t>(srcW);
                    const size_t dstRow = static_cast<size_t>(j) * static_cast<size_t>(rw) * 4u;
                    for (int i = 0; i < rw; ++i) {
                        const auto& pixel = pixels[srcRow + static_cast<size_t>(rx + i)];
                        const size_t dstOffset = dstRow + static_cast<size_t>(i) * 4u;
                        region[dstOffset + 0] = pixel.r;
                        region[dstOffset + 1] = pixel.g;
                        region[dstOffset + 2] = pixel.b;
                        region[dstOffset + 3] = pixel.a;
                    }
                }
            }

            if (updateTexture2DRegion(gpuTextureId,
                                      region.data(),
                                      static_cast<uint32_t>(tex->width),
                                      static_cast<uint32_t>(tex->height),
                                      uploadChannels,
                                      useSrgb,
                                      rx,
                                      ry,
                                      static_cast<uint32_t>(rw),
                                      static_cast<uint32_t>(rh))) {
                tex->clearVulkanDirty();
                return true;
            }
        }

        std::vector<uint8_t> packedPixels(static_cast<size_t>(tex->width) *
                                          static_cast<size_t>(tex->height) *
                                          static_cast<size_t>(uploadChannels));
        if (canUseSingleChannel) {
            for (size_t i = 0; i < pixels.size(); ++i) {
                packedPixels[i] = pixels[i].r;
            }
        } else if (canUseTwoChannels) {
            for (size_t i = 0; i < pixels.size(); ++i) {
                const auto& pixel = pixels[i];
                const size_t dstOffset = i * 2u;
                packedPixels[dstOffset + 0] = pixel.r;
                packedPixels[dstOffset + 1] = pixel.g;
            }
        } else {
            for (size_t i = 0; i < pixels.size(); ++i) {
                const auto& pixel = pixels[i];
                const size_t dstOffset = i * 4u;
                packedPixels[dstOffset + 0] = pixel.r;
                packedPixels[dstOffset + 1] = pixel.g;
                packedPixels[dstOffset + 2] = pixel.b;
                packedPixels[dstOffset + 3] = pixel.a;
            }
        }

        if (!updateTexture2DInPlace(gpuTextureId,
                                    packedPixels.data(),
                                    static_cast<uint32_t>(tex->width),
                                    static_cast<uint32_t>(tex->height),
                                    uploadChannels,
                                    useSrgb,
                                    false)) {
            return false;
        }

        tex->clearVulkanDirty();
        return true;
    };

    VulkanRT::VkGpuMaterial gm = m_cachedGpuMaterials[materialIndex];
    gm.albedo_r = m.albedo.x;
    gm.albedo_g = m.albedo.y;
    gm.albedo_b = m.albedo.z;
    gm.opacity = m.opacity;
    gm.roughness = m.roughness;
    gm.metallic = m.metallic;
    gm.specular = m.specular;
    gm.ior = m.ior;
    gm.transmission = m.transmission;
    gm.emission_r = m.emission.x;
    gm.emission_g = m.emission.y;
    gm.emission_b = m.emission.z;
    gm.emission_strength = m.emissionStrength;
    gm.subsurface_r = m.subsurfaceColor.x;
    gm.subsurface_g = m.subsurfaceColor.y;
    gm.subsurface_b = m.subsurfaceColor.z;
    gm.subsurface_amount = m.subsurface;
    gm.subsurface_radius_r = m.subsurfaceRadius.x;
    gm.subsurface_radius_g = m.subsurfaceRadius.y;
    gm.subsurface_radius_b = m.subsurfaceRadius.z;
    gm.subsurface_scale = m.subsurfaceScale;
    gm.subsurface_ior = m.subsurfaceIOR;
    gm.normal_strength = m.normalStrength;
    gm.clearcoat = m.clearcoat;
    gm.clearcoat_roughness = m.clearcoatRoughness;
    gm.translucent = m.translucent;
    gm.subsurface_anisotropy = m.subsurfaceAnisotropy;
    gm.anisotropic = m.anisotropic;
    gm.sheen = m.sheen;
    gm.sheen_tint = m.sheenTint;
    gm.flags = (uint32_t)m.flags;
    gm.bubble_ior = m.bubble_ior;
    gm.bubble_film = m.bubble_film;
    gm.clearcoat_iridescence = m.clearcoat_iridescence;
    gm.clearcoat_film_thickness = m.clearcoat_film_thickness;
    if (m.is_bubble) gm.flags |= VulkanRT::VK_MAT_FLAG_BUBBLE;
    gm.resin_color_r = static_cast<float>(m.resin_color.x);
    gm.resin_color_g = static_cast<float>(m.resin_color.y);
    gm.resin_color_b = static_cast<float>(m.resin_color.z);
    gm.transmission_density = m.transmission_density;
    gm.resin_roughness = m.resin_roughness;
    gm.dispersion = m.dispersion;
    gm.resin_inclusion = m.resin_inclusion;
    gm.resin_dirt = m.resin_dirt;
    gm.resin_inclusion_scale = m.resin_inclusion_scale;
    gm.resin_dirt_color_r = static_cast<float>(m.resin_dirt_color.x);
    gm.resin_dirt_color_g = static_cast<float>(m.resin_dirt_color.y);
    gm.resin_dirt_color_b = static_cast<float>(m.resin_dirt_color.z);
    gm.resin_shard = m.resin_shard;
    gm.resin_shard_hue = m.resin_shard_hue;
    gm.dust_color_a_r = static_cast<float>(m.dust_color_a.x);
    gm.dust_color_a_g = static_cast<float>(m.dust_color_a.y);
    gm.dust_color_a_b = static_cast<float>(m.dust_color_a.z);
    gm.dust_style = static_cast<float>(m.dust_style);
    gm.dust_color_b_r = static_cast<float>(m.dust_color_b.x);
    gm.dust_color_b_g = static_cast<float>(m.dust_color_b.y);
    gm.dust_color_b_b = static_cast<float>(m.dust_color_b.z);
    gm.shard_shape = static_cast<float>(m.shard_shape);
    if (m.resin_object_space) gm.flags |= VulkanRT::VK_MAT_FLAG_RESIN_OBJ_SPACE;
    if (m.glass_marble_volume) gm.flags |= VulkanRT::VK_MAT_FLAG_MARBLE_VOLUME;
    gm.uv_scale_x = static_cast<float>(m.uvScale.x);
    gm.uv_scale_y = static_cast<float>(m.uvScale.y);
    gm.uv_offset_x = static_cast<float>(m.uvOffset.x);
    gm.uv_offset_y = static_cast<float>(m.uvOffset.y);
    gm.uv_rotation_degrees = m.uvRotationDegrees;
    gm.uv_tiling_x = static_cast<float>(m.uvTiling.x);
    gm.uv_tiling_y = static_cast<float>(m.uvTiling.y);
    gm.uv_wrap_mode = m.uvWrapMode;
    gm.micro_detail_strength = m.micro_detail_strength;
    gm.micro_detail_scale = m.micro_detail_scale;
    gm.tile_break_strength = m.tile_break_strength;
    gm.fft_amplitude = m.fft_amplitude;
    gm.fft_time_scale = m.fft_time_scale;
    gm.foam_threshold = m.foam_threshold;
    gm.fft_ocean_size = m.fft_ocean_size;
    gm.fft_choppiness = m.fft_choppiness;
    gm.fft_wind_speed = m.fft_wind_speed;
    gm.fft_wind_direction = m.fft_wind_direction;
    gm.micro_anim_speed = m.micro_anim_speed;
    gm.micro_morph_speed = m.micro_morph_speed;
    gm.foam_noise_scale = m.foam_noise_scale;
    if (m.flags & Backend::IBackend::MAT_FLAG_TERRAIN) {
        gm._terrain_layer_idx = m.terrainLayerIdx;
    }

    if (!syncExistingTexture(m.albedoTexture, gm.albedo_tex, TextureType::Albedo, false) ||
        !syncExistingTexture(m.normalTexture, gm.normal_tex, TextureType::Normal, true) ||
        !syncExistingTexture(m.roughnessTexture, gm.roughness_tex, TextureType::Roughness, true, false) ||
        !syncExistingTexture(m.metallicTexture, gm.metallic_tex, TextureType::Metallic, true, false) ||
        !syncExistingTexture(m.specularTexture, gm.specular_tex, TextureType::Specular, true, true) ||
        !syncExistingTexture(m.emissionTexture, gm.emission_tex, TextureType::Emission, false) ||
        !syncExistingTexture(m.transmissionTexture, gm.transmission_tex, TextureType::Transmission, true, true) ||
        !syncExistingTexture(m.opacityTexture, gm.opacity_tex, TextureType::Opacity, true, true) ||
        !syncExistingTexture(m.heightTexture, gm.height_tex, TextureType::Unknown, true)) {
        return false;
    }

    if (m.opacityTexture) {
        Texture* opTex = resolveHostTexture(m.opacityTexture);
        if (opTex && opTex->is_loaded() && opTex->has_alpha) {
            const bool hasBc4Dds = [&]() -> bool {
                auto cand = findCompressedTextureCacheCandidate(*opTex, TextureType::Opacity, false);
                return cand && cand->target == TextureCompressionTarget::BC4;
            }();
            if (!hasBc4Dds) {
                gm.flags |= (1u << 8);
            }
        }
    }

    auto needsRedChannelFlag = [](Texture* tex, TextureType type) -> bool {
        if (!tex || !tex->is_loaded()) return false;
        if (auto cand = findCompressedTextureCacheCandidate(*tex, type, false)) {
            if (cand->target == TextureCompressionTarget::BC4) return true;
        }
        return tex->is_gray_scale && !tex->has_alpha;
    };
    if (m.roughnessTexture) {
        Texture* roughnessTex = resolveHostTexture(m.roughnessTexture);
        if (needsRedChannelFlag(roughnessTex, TextureType::Roughness)) {
            gm.flags |= (1u << 9);
        }
    }
    if (m.metallicTexture) {
        Texture* metallicTex = resolveHostTexture(m.metallicTexture);
        if (needsRedChannelFlag(metallicTex, TextureType::Metallic)) {
            gm.flags |= (1u << 10);
        }
    }
    // Bits 12-13 / 14-15: explicit user channel override (see uploadMaterials).
    gm.flags |= ((uint32_t)std::clamp(m.metallicTexChannel,  0, 3)) << 12;
    gm.flags |= ((uint32_t)std::clamp(m.roughnessTexChannel, 0, 3)) << 14;
    if (m.normalTexture) {
        Texture* normalTex = resolveHostTexture(m.normalTexture);
        if (normalTex && normalTex->is_loaded()) {
            bool isBC5 = false;
            if (auto cand = findCompressedTextureCacheCandidate(*normalTex, TextureType::Normal, false)) {
                if (cand->target == TextureCompressionTarget::BC5) {
                    isBC5 = true;
                }
            }
            const bool isRG8 = !normalTex->is_hdr; // uncompressed LDR normal maps uploaded as RG8 (2 channels)
            if (isBC5 || isRG8) {
                gm.flags |= (1u << 11);
            }
        }
    }

    const uint64_t offset = (uint64_t)materialIndex * sizeof(VulkanRT::VkGpuMaterial);
    m_device->uploadBuffer(m_device->m_materialBuffer, &gm, sizeof(VulkanRT::VkGpuMaterial), offset);
    m_cachedGpuMaterials[materialIndex] = gm;
    m_interactiveViewport.dirty = true;
    resetAccumulation();
    return true;
}

void VulkanBackendAdapter::uploadHairMaterials(const std::vector<HairMaterialData>& materials) {
    if (!m_device || !m_device->isInitialized() || materials.empty()) return;

    std::vector<VulkanRT::HairGpuMaterial> gpuMats;
    gpuMats.reserve(materials.size());

    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& m = materials[i];
        VulkanRT::HairGpuMaterial gm{};
        // Block 1: Color & Roughness
        gm.baseColor[0]   = m.color.x;
        gm.baseColor[1]   = m.color.y;
        gm.baseColor[2]   = m.color.z;
        gm.roughness      = m.roughness;
        // Block 2: Physical
        gm.melanin        = m.melanin;
        gm.melaninRedness = m.melaninRedness;
        gm.ior            = m.ior;
        gm.cuticleAngle   = m.cuticleAngle;
        // Block 3: Mode & Surface
        gm.colorMode      = (uint32_t)m.colorMode;
        gm.radialRoughness = m.radialRoughness;
        gm.specularTint   = m.specularTint;
        gm.diffuseSoftness = m.diffuseSoftness;
        // Block 4: Tint
        gm.tintColor[0]   = m.tintColor.x;
        gm.tintColor[1]   = m.tintColor.y;
        gm.tintColor[2]   = m.tintColor.z;
        gm.tint           = m.tint;
        // Block 5: Coat
        gm.coatTint[0]    = m.coatTint.x;
        gm.coatTint[1]    = m.coatTint.y;
        gm.coatTint[2]    = m.coatTint.z;
        gm.coat           = m.coat;
        // Block 6: Emission
        gm.emission[0]    = m.emission.x;
        gm.emission[1]    = m.emission.y;
        gm.emission[2]    = m.emission.z;
        gm.emissionStrength = m.emissionStrength;
        // Block 7: Root-Tip Gradient
        gm.tipColor[0]    = m.tipColor.x;
        gm.tipColor[1]    = m.tipColor.y;
        gm.tipColor[2]    = m.tipColor.z;
        gm.rootTipBalance = m.rootTipBalance;
        // Block 8: Absorption & Gradient Flag
        gm.absorption[0]  = m.absorption.x;
        gm.absorption[1]  = m.absorption.y;
        gm.absorption[2]  = m.absorption.z;
        gm.enableGradient = m.enableRootTipGradient ? 1u : 0u;
        // Block 9: Random & ID
        gm.randomHue      = m.randomHue;
        gm.randomValue    = m.randomValue;
        gm.groomID        = (uint32_t)i;
        gm.pad            = 0.0f;
        gpuMats.push_back(gm);
    }

    m_device->updateHairMaterialBuffer(gpuMats);
    SCENE_LOG_INFO("[Vulkan] Hair materials uploaded: " + std::to_string(gpuMats.size()));
}

void VulkanBackendAdapter::uploadTerrainLayerMaterials(const std::vector<TerrainLayerData>& layers) {
    if (!m_device || !m_device->isInitialized()) return;
    if (layers.empty()) return;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // [FIX] Wait for in-flight GPU work before potentially destroying/reallocating
    // the terrain layer SSBO (updateTerrainLayerBuffer frees it when resizing).
    m_device->waitIdle();

    if (m_nextTextureID >= VULKAN_TEXTURE_PURGE_THRESHOLD ||
        static_cast<int32_t>(m_uploadedImages.size()) >= VULKAN_TEXTURE_PURGE_THRESHOLD) {
        SCENE_LOG_WARN("[Vulkan] Texture cache near descriptor capacity (terrain upload); purging.");
        purgeUploadedTextureCacheLocked();
    }

    // Convert to GPU structs
    std::vector<VulkanRT::VkTerrainLayerData> gpuLayers;
    gpuLayers.reserve(layers.size());

    for (const auto& ld : layers) {
        VulkanRT::VkTerrainLayerData gld{};
        for (int k = 0; k < 4; ++k) {
            gld.layer_mat_id[k]   = ld.layer_mat_id[k];
            gld.layer_uv_scale[k] = ld.layer_uv_scale[k];
        }
        // Resolve splat map texture to a Vulkan sampler slot
        if (ld.splatMapTexture) {
            Texture* splatTex = reinterpret_cast<Texture*>(ld.splatMapTexture);
            if (splatTex && splatTex->is_loaded()) {
                uint64_t cacheKey = static_cast<uint64_t>(ld.splatMapTexture) << 1; // linear
                TextureHandle sceneHandle{};
                if (m_sceneTextureManager) {
                    sceneHandle = registerTerrainSplatTexture(
                        m_sceneTextureManager.get(),
                        splatTex,
                        TextureConsumer::RasterPreview | TextureConsumer::VulkanRT);
                }
                // [FIX] Evict stale Vulkan sampler entry when splat map pixels were modified
                // in-place (autoMask / importSplatMap / paintSplatMap → updateGPU sets vulkan_dirty).
                if (splatTex->vulkan_dirty) {
                    if (sceneHandle.isValid() && m_sceneTextureManager) {
                        int64_t pooledId = 0;
                        if (m_sceneTextureManager->tryGetVulkanTextureId(
                                sceneHandle, sceneTextureOwnerScope(), pooledId) &&
                            pooledId != 0) {
                            m_sceneTextureManager->clearVulkanBacking(sceneTextureOwnerScope(), pooledId);
                            this->destroyTexture(pooledId);
                        }
                    }
                    auto oldIt = m_uploadedImageIDs.find(cacheKey);
                    if (oldIt != m_uploadedImageIDs.end()) {
                        int64_t oldId = oldIt->second;
                        if (oldId) this->destroyTexture(oldId);
                    }
                    splatTex->clearVulkanDirty();
                }
                auto it = m_uploadedImageIDs.find(cacheKey);
                if (it != m_uploadedImageIDs.end()) {
                    gld.splat_map_tex = (uint32_t)it->second;
                } else {
                    uint32_t splatUploadW = static_cast<uint32_t>(std::max(0, splatTex->width));
                    uint32_t splatUploadH = static_cast<uint32_t>(std::max(0, splatTex->height));
                    if (isViewportTextureOwner(sceneTextureOwnerScope())) {
                        fitWithinMaxDimension(splatUploadW, splatUploadH,
                                              kMaterialPreviewTextureMaxDimension,
                                              splatUploadW, splatUploadH);
                    }
                    auto uploadSplatTextureNow = [&]() -> int64_t {
                        const std::vector<CompactVec4>& px = splatTex->pixels;
                        if (px.empty()) return 0;

                        std::vector<uint8_t> tmp(static_cast<size_t>(splatTex->width) * splatTex->height * 4);
                        for (size_t i = 0; i < px.size(); ++i) {
                            tmp[i*4+0] = px[i].r; tmp[i*4+1] = px[i].g;
                            tmp[i*4+2] = px[i].b; tmp[i*4+3] = px[i].a;
                        }
                        if (splatUploadW != static_cast<uint32_t>(splatTex->width) ||
                            splatUploadH != static_cast<uint32_t>(splatTex->height)) {
                            tmp = resizeLdrBilinear(tmp,
                                                   static_cast<uint32_t>(splatTex->width),
                                                   static_cast<uint32_t>(splatTex->height),
                                                   splatUploadW,
                                                   splatUploadH,
                                                   4u);
                        }
                        return this->uploadTexture2D(tmp.data(), splatUploadW, splatUploadH, 4, false, false);
                    };

                    int64_t id = 0;
                    bool resolvedThroughPool = false;
                    bool createdByPool = false;
                    if (sceneHandle.isValid() && m_sceneTextureManager) {
                        VulkanBackingRecord resolvedBacking{};
                        resolvedThroughPool = m_sceneTextureManager->resolveOrCreateVulkanBacking(
                            buildSceneTextureKey(splatTex, TextureType::Unknown, true, false),
                            sceneTextureOwnerScope(),
                            TextureConsumer::RasterPreview | TextureConsumer::VulkanRT,
                            splatUploadW,
                            splatUploadH,
                            static_cast<uint64_t>(splatUploadW) * splatUploadH * 4ull,
                            [&](TextureHandle, VulkanBackingRecord& outBacking) -> bool {
                                const int64_t uploadedId = uploadSplatTextureNow();
                                if (!uploadedId) {
                                    return false;
                                }
                                return buildVulkanBackingRecord(uploadedId, outBacking);
                            },
                            resolvedBacking,
                            &createdByPool);

                        if (resolvedThroughPool && resolvedBacking.textureId != 0) {
                            VulkanRT::ImageHandle localImage{};
                            if (tryGetUploadedImageHandle(resolvedBacking.textureId, localImage)) {
                                id = resolvedBacking.textureId;
                            } else {
                                m_sceneTextureManager->clearVulkanBacking(sceneTextureOwnerScope(), resolvedBacking.textureId);
                                resolvedThroughPool = false;
                            }
                        }
                    }

                    if (!id) {
                        id = uploadSplatTextureNow();
                    }
                    if (id) {
                        m_uploadedImageIDs[cacheKey] = id;
                        m_textureIdToCacheKey[id] = cacheKey;
                        if (!resolvedThroughPool || !createdByPool) {
                            registerSceneTextureUpload(sceneHandle, id);
                        }
                        gld.splat_map_tex = (uint32_t)id;
                    }
                }
            }
        }
        gld.layer_count = ld.layer_count;
        gpuLayers.push_back(gld);
    }

    m_device->updateTerrainLayerBuffer(
        gpuLayers.data(),
        gpuLayers.size() * sizeof(VulkanRT::VkTerrainLayerData),
        (uint32_t)gpuLayers.size());

    SCENE_LOG_INFO("[Vulkan] Terrain layer materials uploaded: " + std::to_string(gpuLayers.size()));
}

void VulkanBackendAdapter::beginBatchedTextureUpload() {
    if (!m_device || m_inBatchedTextureUpload) return;
    m_batchTextureCmd = m_device->beginSingleTimeCommands();
    if (m_batchTextureCmd == VK_NULL_HANDLE) return;
    m_inBatchedTextureUpload = true;
}

void VulkanBackendAdapter::endBatchedTextureUpload() {
    if (!m_inBatchedTextureUpload) return;
    m_inBatchedTextureUpload = false;
    if (m_batchTextureCmd != VK_NULL_HANDLE) {
        m_device->endSingleTimeCommands(m_batchTextureCmd);
        m_batchTextureCmd = VK_NULL_HANDLE;
    }
    for (auto& buf : m_batchTextureStagingBuffers)
        m_device->destroyBuffer(buf);
    m_batchTextureStagingBuffers.clear();
}

int64_t VulkanBackendAdapter::uploadTexture2D(const void* data, uint32_t width, uint32_t height, uint32_t channels, bool srgb, bool isFloat) {
    if (!m_device || !m_device->isInitialized() || !data) return 0;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    // Call VRAM check and eviction before allocating staging/image resources
    checkAndTrimVRAMThreshold();
    
    if (!m_inBatchedTextureUpload) m_device->waitIdle();

    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t bpp = 4; // bytes per pixel

    if (isFloat) {
        if (channels == 1) {
            fmt = VK_FORMAT_R32_SFLOAT;
            bpp = 4;
        } else {
            fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
            bpp = 16;
        }
    } else if (channels == 1) {
        fmt = srgb ? VK_FORMAT_R8_SRGB : VK_FORMAT_R8_UNORM;
        bpp = 1;
    } else if (channels == 2) {
        fmt = srgb ? VK_FORMAT_R8G8_SRGB : VK_FORMAT_R8G8_UNORM;
        bpp = 2;
    } else if (srgb) {
        fmt = VK_FORMAT_R8G8B8A8_SRGB;
    }

    // Budget-aware mip level calculation.
    // vkCmdBlitImage requires a graphics-capable queue; float textures skip mips to
    // avoid linear blit precision issues on some drivers.
    const bool canGenerateMips = m_device->supportsGraphicsQueue() && !isFloat;
    uint32_t mipLevels = 1;
    if (canGenerateMips) {
        const uint32_t fullMips = calcMipLevels(width, height);
        const uint64_t dedicatedVRAM = m_device->getCapabilities().dedicatedVRAM;
        if (isViewportTextureOwner(sceneTextureOwnerScope())) {
            // Material Preview must prioritize stable paint/opacity masks. Camera
            // movement changes implicit LOD; for layered masks that can expose
            // lower mip levels that do not match the freshly composited base level.
            // Keep the viewport preview on mip 0 and let render backends use mips.
            mipLevels = 1;
        } else if (m_sceneTextureManager && dedicatedVRAM > 0) {
            const float pressure = static_cast<float>(m_sceneTextureManager->totalResidentTextureBytes())
                                 / static_cast<float>(dedicatedVRAM);
            if (pressure < 0.50f) {
                mipLevels = fullMips;
            } else if (pressure < 0.75f) {
                mipLevels = std::min(fullMips, 4u);  // cap at ~1/8 minimum resolution
            } else {
                mipLevels = std::min(fullMips, 2u);  // cap at ~1/2 minimum resolution
            }
        } else {
            mipLevels = fullMips;
        }
    }

    // Create staging buffer
    VulkanRT::BufferCreateInfo ci;
    ci.size = (uint64_t)width * height * bpp;
    ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_SRC;
    ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;

    VulkanRT::BufferHandle staging = m_device->createBuffer(ci);
    if (!staging.buffer) return 0;

    m_device->uploadBuffer(staging, data, ci.size);

    VulkanRT::ImageHandle img;
    if (mipLevels > 1) {
        // Mip path: create image pre-transitioned to TRANSFER_DST_OPTIMAL for all levels.
        img = m_device->createImage2DWithMips(width, height, mipLevels, fmt);
        if (!img.image) {
            m_device->destroyBuffer(staging);
            return 0;
        }
        if (m_inBatchedTextureUpload) {
            m_device->recordCopyBufferToImageDst(m_batchTextureCmd, staging, img);
            m_device->generateMipmaps(m_batchTextureCmd, img.image, width, height, mipLevels);
        } else {
            VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
            if (cmd == VK_NULL_HANDLE) {
                m_device->destroyImage(img);
                m_device->destroyBuffer(staging);
                return 0;
            }
            m_device->recordCopyBufferToImageDst(cmd, staging, img);
            m_device->generateMipmaps(cmd, img.image, width, height, mipLevels);
            m_device->endSingleTimeCommands(cmd);
        }
    } else {
        // Single-mip path: existing GENERAL layout path.
        img = m_device->createImage2D(width, height, fmt,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        if (!img.image) {
            m_device->destroyBuffer(staging);
            return 0;
        }
        if (m_inBatchedTextureUpload) {
            m_device->recordCopyBufferToImage(m_batchTextureCmd, staging, img);
            m_device->transitionImageLayout(m_batchTextureCmd, img.image,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        } else {
            m_device->copyBufferToImage(staging, img);
            VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
            if (cmd == VK_NULL_HANDLE) {
                m_device->destroyImage(img);
                m_device->destroyBuffer(staging);
                return 0;
            }
            m_device->transitionImageLayout(cmd, img.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            m_device->endSingleTimeCommands(cmd);
        }
    }

    // Sampler: maxLod drives which mip levels the hardware will sample.
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    const auto& caps = m_device->getCapabilities();
    if (caps.supportsSamplerAnisotropy && mipLevels > 1) {
        sci.anisotropyEnable = VK_TRUE;
        sci.maxAnisotropy = std::clamp(caps.maxSamplerAnisotropy, 1.0f, 8.0f);
    } else {
        sci.anisotropyEnable = VK_FALSE;
        sci.maxAnisotropy = 1.0f;
    }
    sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sci.unnormalizedCoordinates = VK_FALSE;
    sci.compareEnable = VK_FALSE;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.minLod = 0.0f;
    sci.maxLod = static_cast<float>(mipLevels - 1);

    VkSampler sampler = VK_NULL_HANDLE;
    vkCreateSampler(m_device->getDevice(), &sci, nullptr, &sampler);
    if (sampler) img.sampler = sampler;

    // Register image with an ID
    int64_t id = m_nextTextureID++;
    m_uploadedImages[id] = img;
    const uint64_t estimated_vram_bytes = estimateImageStorageBytes(width, height, fmt, mipLevels);
    m_textureUploadBytes += estimated_vram_bytes;
    ++m_textureUploadCount;
    if (fmt == VK_FORMAT_R8_UNORM || fmt == VK_FORMAT_R8_SRGB) {
        ++m_textureUploadR8Count;
    } else if (fmt == VK_FORMAT_R8G8_UNORM || fmt == VK_FORMAT_R8G8_SRGB) {
        ++m_textureUploadRG8Count;
    } else if (fmt == VK_FORMAT_R32G32B32A32_SFLOAT || fmt == VK_FORMAT_R32_SFLOAT) {
        ++m_textureUploadFloatCount;
    } else {
        ++m_textureUploadRGBA8Count;
    }
    m_textureUploadSummaryDirty = true;
    if (m_device) {
        uint32_t slot = (uint32_t)id;
        if (m_device->hasHardwareRT()) {
            m_device->updateRTTextureDescriptor(slot, img);
        }
        updateMaterialPreviewTextureDescriptor(slot, img);
    }
    if (m_inBatchedTextureUpload) {
        m_batchTextureStagingBuffers.push_back(staging);
    } else {
        // CRITICAL: release temporary upload staging buffer.
        // Missing destroy here caused per-texture memory growth across Vulkan sessions.
        m_device->destroyBuffer(staging);
    }
    return id;
}

bool VulkanBackendAdapter::tryGetUploadedImageHandle(int64_t textureHandle, VulkanRT::ImageHandle& outImage) const {
    auto it = m_uploadedImages.find(textureHandle);
    if (it != m_uploadedImages.end()) {
        outImage = it->second;
        return outImage.image != VK_NULL_HANDLE && outImage.view != VK_NULL_HANDLE;
    }
    return false;
}

bool VulkanBackendAdapter::buildVulkanBackingRecord(int64_t textureHandle, VulkanBackingRecord& outBacking) const {
    outBacking = VulkanBackingRecord{};
    VulkanRT::ImageHandle image{};
    if (!tryGetUploadedImageHandle(textureHandle, image)) {
        return false;
    }

    outBacking.textureId = textureHandle;
    outBacking.image = reinterpret_cast<uint64_t>(image.image);
    outBacking.view = reinterpret_cast<uint64_t>(image.view);
    outBacking.memory = reinterpret_cast<uint64_t>(image.memory);
    outBacking.sampler = reinterpret_cast<uint64_t>(image.sampler);
    outBacking.width = image.width;
    outBacking.height = image.height;
    outBacking.format = static_cast<uint32_t>(image.format);
    outBacking.allocatedBytes = estimateImageStorageBytes(image.width, image.height, image.format, image.mipLevels);
    return outBacking.isValid();
}

void VulkanBackendAdapter::checkAndTrimVRAMThreshold() {
    if (!m_sceneTextureManager) return;
    const uint64_t dedicatedVRAM = m_device->getCapabilities().dedicatedVRAM;
    if (dedicatedVRAM <= 0) return;

    const uint64_t textureBytes = m_sceneTextureManager->totalResidentTextureBytes();
    const uint64_t trimThreshold = dedicatedVRAM * 85 / 100;
    const uint64_t trimTarget    = dedicatedVRAM * 6 / 10;
    const uint64_t totalMB = dedicatedVRAM / (1024 * 1024);

    if (textureBytes > trimThreshold) {
        const std::string currentOwner = sceneTextureOwnerScope();
        const std::string otherOwner = (currentOwner == "VulkanBackendAdapter") ? "VulkanViewportBackend" : "VulkanBackendAdapter";
        
        // First, try evicting all inactive textures from the other backend
        const size_t otherEvicted = m_sceneTextureManager->trimVulkanBackingLRU(otherOwner, 0);
        const uint64_t postOtherTextureBytes = m_sceneTextureManager->totalResidentTextureBytes();
        const uint64_t postOtherUsedMB = postOtherTextureBytes / (1024 * 1024);
        
        if (otherEvicted > 0) {
            SCENE_LOG_INFO("[Vulkan] Evicted " + std::to_string(otherEvicted) + 
                           " textures from inactive backend (" + otherOwner + 
                           ") to free VRAM. Post-evict VRAM: ~" + std::to_string(postOtherUsedMB) + " MB.");
        }
        
        if (postOtherTextureBytes > trimThreshold) {
            const std::string msg = "VRAM critical after inactive backend eviction: ~" + std::to_string(postOtherUsedMB) +
                                    " MB texture / " + std::to_string(totalMB) +
                                    " MB — LRU eviction of active backend triggered.";
            SCENE_LOG_WARN("[Vulkan] " + msg);
            if (m_statusCallback) m_statusCallback(msg, 2); // kirmizi
            const size_t evicted = m_sceneTextureManager->trimVulkanBackingLRU(currentOwner, trimTarget);
            if (evicted > 0 && m_statusCallback) {
                m_statusCallback("Evicted " + std::to_string(evicted) + " active textures to free VRAM.", 1);
            }
        } else if (otherEvicted > 0 && m_statusCallback) {
            m_statusCallback("Evicted " + std::to_string(otherEvicted) + " inactive textures to free VRAM.", 1);
        }
    }
}

void VulkanBackendAdapter::registerSceneTextureUpload(TextureHandle sceneHandle, int64_t textureHandle) {
    if (!m_sceneTextureManager || !sceneHandle.isValid() || textureHandle == 0) {
        return;
    }

    VulkanBackingRecord backing{};
    if (buildVulkanBackingRecord(textureHandle, backing)) {
        // Attach a destroy callback so the manager can drive physical lifecycle.
        // Use shared_ptr for device and alive-sentinel to survive backend rebuilds:
        // if this backend's containers are torn down before the lambda runs, the
        // sentinel is false and container access is skipped (GPU-only cleanup).
        VulkanRT::VulkanDevice* devPtr = m_device.get();
        std::shared_ptr<bool> alive = m_containerAlive;
        auto* images = &m_uploadedImages;
        auto* imageIDs = &m_uploadedImageIDs;
        auto* idToCacheKey = &m_textureIdToCacheKey;
        const int64_t capturedId = textureHandle;
        VulkanRT::ImageHandle capturedImg{};
        {
            auto it = m_uploadedImages.find(capturedId);
            if (it != m_uploadedImages.end()) capturedImg = it->second;
        }
        backing.destroyFn = [devPtr, alive, images, imageIDs, idToCacheKey, capturedId, capturedImg]() mutable {
            // alive is false when the backend was torn down without going through
            // purgeUploadedTextureCacheLocked (e.g. fast mode switch). In that case
            // both the device and the containers are gone — skip everything.
            if (!*alive) return;
            if (devPtr && (capturedImg.image || capturedImg.view || capturedImg.memory || capturedImg.sampler)) {
                devPtr->destroyImage(capturedImg);
            }
            images->erase(capturedId);
            auto ckIt = idToCacheKey->find(capturedId);
            if (ckIt != idToCacheKey->end()) {
                imageIDs->erase(ckIt->second);
                idToCacheKey->erase(ckIt);
            }
        };
        m_sceneTextureManager->setVulkanBacking(sceneHandle, sceneTextureOwnerScope(), backing);
        return;
    }

    m_sceneTextureManager->setVulkanTextureId(sceneHandle, sceneTextureOwnerScope(), textureHandle);
}

int64_t VulkanBackendAdapter::uploadCompressedTexture2D(
    const void* data,
    uint64_t dataSize,
    uint32_t width,
    uint32_t height,
    VkFormat format)
{
    if (!m_device || !m_device->isInitialized() || !data || dataSize == 0) return 0;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    // Call VRAM check and eviction before allocating staging/image resources
    checkAndTrimVRAMThreshold();
    
    if (!m_inBatchedTextureUpload) m_device->waitIdle();

    const uint64_t expectedBytes = estimateImageStorageBytes(width, height, format);
    if (expectedBytes == 0 || expectedBytes != dataSize) return 0;

    VulkanRT::BufferCreateInfo ci;
    ci.size = dataSize;
    ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_SRC;
    ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;

    VulkanRT::BufferHandle staging = m_device->createBuffer(ci);
    if (!staging.buffer) return 0;

    m_device->uploadBuffer(staging, data, dataSize);

    VulkanRT::ImageHandle img = m_device->createImage2D(
        width, height, format,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!img.image) {
        m_device->destroyBuffer(staging);
        return 0;
    }

    if (m_inBatchedTextureUpload) {
        m_device->recordCopyBufferToImage(m_batchTextureCmd, staging, img);
        m_device->transitionImageLayout(m_batchTextureCmd, img.image,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        m_device->copyBufferToImage(staging, img);
        VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
        if (cmd == VK_NULL_HANDLE) {
            m_device->destroyImage(img);
            m_device->destroyBuffer(staging);
            return 0;
        }
        m_device->transitionImageLayout(cmd, img.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_device->endSingleTimeCommands(cmd);
    }

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    const auto& caps = m_device->getCapabilities();
    if (caps.supportsSamplerAnisotropy && !isViewportTextureOwner(sceneTextureOwnerScope())) {
        sci.anisotropyEnable = VK_TRUE;
        sci.maxAnisotropy = std::clamp(caps.maxSamplerAnisotropy, 1.0f, 8.0f);
    } else {
        sci.anisotropyEnable = VK_FALSE;
        sci.maxAnisotropy = 1.0f;
    }
    sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sci.unnormalizedCoordinates = VK_FALSE;
    sci.compareEnable = VK_FALSE;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.minLod = 0.0f;
    sci.maxLod = 0.0f;

    VkSampler sampler = VK_NULL_HANDLE;
    vkCreateSampler(m_device->getDevice(), &sci, nullptr, &sampler);
    if (sampler) img.sampler = sampler;

    int64_t id = m_nextTextureID++;
    m_uploadedImages[id] = img;
    m_textureUploadBytes += expectedBytes;
    ++m_textureUploadCount;
    switch (format) {
        case VK_FORMAT_BC4_UNORM_BLOCK: ++m_textureUploadBC4Count; break;
        case VK_FORMAT_BC5_UNORM_BLOCK: ++m_textureUploadBC5Count; break;
        case VK_FORMAT_BC7_UNORM_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK: ++m_textureUploadBC7Count; break;
        default: break;
    }
    m_textureUploadSummaryDirty = true;
    if (m_device) {
        const uint32_t slot = (uint32_t)id;
        if (m_device->hasHardwareRT()) {
            m_device->updateRTTextureDescriptor(slot, img);
        }
        updateMaterialPreviewTextureDescriptor(slot, img);
    }

    if (m_inBatchedTextureUpload) {
        m_batchTextureStagingBuffers.push_back(staging);
    } else {
        m_device->destroyBuffer(staging);
    }
    return id;
}

int64_t VulkanBackendAdapter::uploadTexture3D(const void* data, uint32_t width, uint32_t height, uint32_t depth,
                                               uint32_t channels, bool isFloat) {
    if (!m_device || !m_device->isInitialized() || !data) return 0;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    // --- Format selection ---
    VkFormat fmt = VK_FORMAT_R8_UNORM;
    uint32_t bpp = 1;
    if (isFloat) {
        if (channels == 1)      { fmt = VK_FORMAT_R32_SFLOAT;            bpp = 4; }
        else if (channels == 4) { fmt = VK_FORMAT_R32G32B32A32_SFLOAT;   bpp = 16; }
        else                    { fmt = VK_FORMAT_R32_SFLOAT;            bpp = 4; }
    } else {
        if (channels == 1)      { fmt = VK_FORMAT_R8_UNORM;              bpp = 1; }
        else if (channels == 4) { fmt = VK_FORMAT_R8G8B8A8_UNORM;        bpp = 4; }
        else                    { fmt = VK_FORMAT_R8_UNORM;              bpp = 1; }
    }

    VkDevice dev   = m_device->getDevice();
    uint64_t bytes = (uint64_t)width * height * depth * bpp;

    // --- Staging buffer ---
    VulkanRT::BufferCreateInfo sci;
    sci.size     = bytes;
    sci.usage    = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_SRC;
    sci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
    VulkanRT::BufferHandle staging = m_device->createBuffer(sci);
    if (!staging.buffer) return 0;
    m_device->uploadBuffer(staging, data, bytes);

    // --- VkImage (3D) ---
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_3D;
    ici.format        = fmt;
    ici.extent        = { width, height, depth };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VulkanRT::ImageHandle img{};
    img.width  = width;
    img.height = height;
    img.format = fmt;

    if (vkCreateImage(dev, &ici, nullptr, &img.image) != VK_SUCCESS) {
        m_device->destroyBuffer(staging);
        return 0;
    }

    // --- Memory ---
    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(dev, img.image, &memReq);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = m_device->findMemoryType(memReq.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) {
        signalVulkanMemoryPressure(VK_ERROR_OUT_OF_DEVICE_MEMORY, "uploadTexture3D/findMemoryType");
        m_device->destroyImage(img);
        m_device->destroyBuffer(staging);
        return 0;
    }
    VkResult imgAllocRes = vkAllocateMemory(dev, &mai, nullptr, &img.memory);
    if (imgAllocRes != VK_SUCCESS || !img.memory) {
        signalVulkanMemoryPressure(imgAllocRes, "uploadTexture3D/vkAllocateMemory");
        m_device->destroyImage(img);
        m_device->destroyBuffer(staging);
        return 0;
    }
    VkResult imgBindRes = vkBindImageMemory(dev, img.image, img.memory, 0);
    if (imgBindRes != VK_SUCCESS) {
        m_device->destroyImage(img);
        m_device->destroyBuffer(staging);
        return 0;
    }

    // --- ImageView (3D) ---
    VkImageViewCreateInfo vci{};
    vci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image                           = img.image;
    vci.viewType                        = VK_IMAGE_VIEW_TYPE_3D;
    vci.format                          = fmt;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount     = 1;
    if (vkCreateImageView(dev, &vci, nullptr, &img.view) != VK_SUCCESS || !img.view) {
        m_device->destroyImage(img);
        m_device->destroyBuffer(staging);
        return 0;
    }

    // --- Upload: UNDEFINED → TRANSFER_DST_OPTIMAL → copy → SHADER_READ_ONLY_OPTIMAL ---
    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        m_device->destroyImage(img);
        m_device->destroyBuffer(staging);
        return 0;
    }

    m_device->transitionImageLayout(cmd, img.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent                 = { width, height, depth };
    vkCmdCopyBufferToImage(cmd, staging.buffer, img.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    m_device->transitionImageLayout(cmd, img.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    m_device->endSingleTimeCommands(cmd);
    m_device->destroyBuffer(staging);

    // --- Sampler ---
    VkSamplerCreateInfo smpl{};
    smpl.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    smpl.magFilter    = VK_FILTER_LINEAR;
    smpl.minFilter    = VK_FILTER_LINEAR;
    smpl.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    smpl.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    smpl.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    smpl.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    smpl.maxAnisotropy = 1.0f;
    smpl.borderColor  = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    vkCreateSampler(dev, &smpl, nullptr, &img.sampler);

    // --- Register ---
    int64_t id = m_nextTextureID++;
    m_uploadedImages[id] = img;
    if (m_device) {
        if (m_device->hasHardwareRT()) {
            m_device->updateRTTextureDescriptor((uint32_t)id, img);
        }
        updateMaterialPreviewTextureDescriptor((uint32_t)id, img);
    }
    return id;
}

void VulkanBackendAdapter::destroyTexture(int64_t texID) {
    VulkanRT::ImageHandle img{};
    if (!tryGetUploadedImageHandle(texID, img)) return;
    if (m_device->hasHardwareRT()) {
        m_device->removePendingRTTextureDescriptor(img);
    }
    // If the manager owns a destroy callback for this backing, let it drive lifecycle.
    // destroyAndClearVulkanBacking will invoke the callback (which erases from m_uploadedImages
    // and calls destroyImage), then clears the manager record. Fall back to direct destroy only
    // when the backing has no registered callback (e.g., legacy non-scene textures).
    bool destroyedByManager = false;
    if (m_sceneTextureManager) {
        destroyedByManager = m_sceneTextureManager->destroyAndClearVulkanBacking(
            sceneTextureOwnerScope(), texID);
        if (!destroyedByManager) {
            m_sceneTextureManager->clearVulkanTextureId(sceneTextureOwnerScope(), texID);
        }
    }
    if (!destroyedByManager) {
        m_device->destroyImage(img);
        m_uploadedImages.erase(texID);
        auto ckIt = m_textureIdToCacheKey.find(texID);
        if (ckIt != m_textureIdToCacheKey.end()) {
            m_uploadedImageIDs.erase(ckIt->second);
            m_textureIdToCacheKey.erase(ckIt);
        }
    }
}

void VulkanBackendAdapter::regenerateMipChainAfterPartialUpdate(VkCommandBuffer cmd,
                                                                 const VulkanRT::ImageHandle& img) {
    if (cmd == VK_NULL_HANDLE || !img.image || img.mipLevels <= 1) return;

    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.image = img.image;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = 1;
    b.subresourceRange.levelCount = 1;

    // Step 1: mip 0 GENERAL -> TRANSFER_SRC_OPTIMAL (it just received a
    // copyBufferToImage in GENERAL layout; we now use it as a blit source).
    b.subresourceRange.baseMipLevel = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b);

    int32_t mipW = static_cast<int32_t>(img.width);
    int32_t mipH = static_cast<int32_t>(img.height);

    for (uint32_t i = 1; i < img.mipLevels; ++i) {
        // Step 2a: mip i SHADER_READ_ONLY -> TRANSFER_DST_OPTIMAL (blit dst).
        b.subresourceRange.baseMipLevel = i;
        b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        const int32_t nextW = std::max(1, mipW / 2);
        const int32_t nextH = std::max(1, mipH / 2);

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipW, mipH, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {nextW, nextH, 1};
        vkCmdBlitImage(cmd,
            img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        // Step 2b: mip i TRANSFER_DST -> TRANSFER_SRC (next iteration's
        // source) for non-final levels; final level goes straight to
        // SHADER_READ_ONLY in the closing barrier below.
        if (i + 1 < img.mipLevels) {
            b.subresourceRange.baseMipLevel = i;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
        }

        mipW = nextW;
        mipH = nextH;
    }

    // Step 3a: mips 0 .. N-2 are in TRANSFER_SRC_OPTIMAL — flip them all to
    // SHADER_READ_ONLY_OPTIMAL in a single barrier.
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = img.mipLevels - 1;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &b);

    // Step 3b: final mip is in TRANSFER_DST_OPTIMAL.
    b.subresourceRange.baseMipLevel = img.mipLevels - 1;
    b.subresourceRange.levelCount = 1;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &b);
}

bool VulkanBackendAdapter::updateTexture2DInPlace(int64_t textureID, const void* data,
                                                  uint32_t width, uint32_t height,
                                                  uint32_t channels, bool srgb, bool isFloat) {
    if (!m_device || !m_device->isInitialized() || !data || textureID <= 0) return false;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    VulkanRT::ImageHandle img{};
    if (!tryGetUploadedImageHandle(textureID, img)) return false;
    if (!img.image || img.width != width || img.height != height) return false;

    // Derive the expected format from (channels, srgb, isFloat) and bail if it
    // doesn't match what the existing VkImage was created with — caller falls
    // back to destroy + re-create in that case.
    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t bpp = 4;
    if (isFloat) {
        if (channels == 1) { fmt = VK_FORMAT_R32_SFLOAT;           bpp = 4; }
        else               { fmt = VK_FORMAT_R32G32B32A32_SFLOAT;  bpp = 16; }
    } else if (channels == 1) {
        fmt = srgb ? VK_FORMAT_R8_SRGB : VK_FORMAT_R8_UNORM;
        bpp = 1;
    } else if (channels == 2) {
        fmt = srgb ? VK_FORMAT_R8G8_SRGB : VK_FORMAT_R8G8_UNORM;
        bpp = 2;
    } else if (srgb) {
        fmt = VK_FORMAT_R8G8B8A8_SRGB;
    }
    if (img.format != fmt) return false;

    m_device->waitIdle();

    // Staging buffer -> existing VkImage. Same pattern as uploadTexture2D but
    // without creating a new ImageHandle, sampler, descriptor slot, or id.
    VulkanRT::BufferCreateInfo ci;
    ci.size = (uint64_t)width * height * bpp;
    ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_SRC;
    ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;

    VulkanRT::BufferHandle staging = m_device->createBuffer(ci);
    if (!staging.buffer) return false;
    m_device->uploadBuffer(staging, data, ci.size);

    // Transition SHADER_READ_ONLY_OPTIMAL -> GENERAL for the copy, then back.
    // copyBufferToImage uses GENERAL layout (see VulkanDevice::copyBufferToImage),
    // so we match that instead of TRANSFER_DST_OPTIMAL.
    VkCommandBuffer cmdPre = m_device->beginSingleTimeCommands();
    if (cmdPre == VK_NULL_HANDLE) {
        m_device->destroyBuffer(staging);
        return false;
    }
    m_device->transitionImageLayout(cmdPre, img.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    m_device->endSingleTimeCommands(cmdPre);

    m_device->copyBufferToImage(staging, img);

    VkCommandBuffer cmdPost = m_device->beginSingleTimeCommands();
    if (cmdPost == VK_NULL_HANDLE) {
        m_device->destroyBuffer(staging);
        return false;
    }
    if (img.mipLevels > 1) {
        // Image was created with a mip chain (asset-style upload). Mip 0 just
        // received the new pixels; mips 1..N still hold whatever they were
        // last blitted from — typically the canvas's *initial* white fill
        // when this is a paint texture. Regenerate the chain from mip 0 so
        // the raster Material Preview viewport doesn't surface that stale
        // content as soon as the camera moves a step away. Single-mip
        // images skip this entirely (no behaviour change for them).
        regenerateMipChainAfterPartialUpdate(cmdPost, img);
    } else {
        m_device->transitionImageLayout(cmdPost, img.image,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    m_device->endSingleTimeCommands(cmdPost);

    m_device->destroyBuffer(staging);
    // The VkImage / view / sampler are unchanged, so no descriptor rewrite is
    // needed on either the RT pipeline or the material-preview raster pipeline.
    // But the cached raster framebuffer in m_interactiveViewport is now stale —
    // without this flag, renderInteractiveViewportImpl short-circuits and paint
    // strokes stay invisible in Material Preview mode.
    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanBackendAdapter::updateTexture2DRegion(int64_t textureID, const void* data,
                                                 uint32_t fullWidth, uint32_t fullHeight,
                                                 uint32_t channels, bool srgb,
                                                 int32_t offsetX, int32_t offsetY,
                                                 uint32_t regionW, uint32_t regionH) {
    if (!m_device || !m_device->isInitialized() || !data || textureID <= 0) return false;
    if (regionW == 0 || regionH == 0) return false;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    VulkanRT::ImageHandle img{};
    if (!tryGetUploadedImageHandle(textureID, img)) return false;
    if (!img.image || img.width != fullWidth || img.height != fullHeight) return false;

    // Only the non-HDR formats produced by uploadTexture2D's paint path are
    // valid here. Caller is expected to have ruled out the HDR path already.
    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t bpp = 4;
    if (channels == 1) {
        fmt = srgb ? VK_FORMAT_R8_SRGB : VK_FORMAT_R8_UNORM;
        bpp = 1;
    } else if (channels == 2) {
        fmt = srgb ? VK_FORMAT_R8G8_SRGB : VK_FORMAT_R8G8_UNORM;
        bpp = 2;
    } else if (srgb) {
        fmt = VK_FORMAT_R8G8B8A8_SRGB;
    }
    if (img.format != fmt) return false;

    if (offsetX < 0 || offsetY < 0 ||
        static_cast<uint32_t>(offsetX) + regionW > fullWidth ||
        static_cast<uint32_t>(offsetY) + regionH > fullHeight) {
        return false;
    }

    m_device->waitIdle();

    VulkanRT::BufferCreateInfo ci;
    ci.size = static_cast<uint64_t>(regionW) * static_cast<uint64_t>(regionH) * bpp;
    ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_SRC;
    ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;

    VulkanRT::BufferHandle staging = m_device->createBuffer(ci);
    if (!staging.buffer) return false;
    m_device->uploadBuffer(staging, data, ci.size);

    VkCommandBuffer cmdPre = m_device->beginSingleTimeCommands();
    if (cmdPre == VK_NULL_HANDLE) {
        m_device->destroyBuffer(staging);
        return false;
    }
    m_device->transitionImageLayout(cmdPre, img.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    m_device->endSingleTimeCommands(cmdPre);

    m_device->copyBufferToImageRegion(staging, img, offsetX, offsetY, regionW, regionH);

    VkCommandBuffer cmdPost = m_device->beginSingleTimeCommands();
    if (cmdPost == VK_NULL_HANDLE) {
        m_device->destroyBuffer(staging);
        return false;
    }
    if (img.mipLevels > 1) {
        // Same mip-chain refresh as updateTexture2DInPlace. The partial-region
        // copy only touched mip 0 inside the dirty rect, so without this the
        // upper mip levels keep returning the canvas's initial fill (white
        // for fresh paint canvases) every time the raster sampler picks a
        // higher mip — which is exactly what happens once the camera moves
        // a little further from a high-resolution paint surface.
        regenerateMipChainAfterPartialUpdate(cmdPost, img);
    } else {
        m_device->transitionImageLayout(cmdPost, img.image,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    m_device->endSingleTimeCommands(cmdPost);

    m_device->destroyBuffer(staging);
    m_interactiveViewport.dirty = true;
    return true;
}

void VulkanBackendAdapter::setLights(const std::vector<std::shared_ptr<Light>>& lights) {
    if (!m_device || !m_device->isInitialized()) {
        m_cachedLights = lights;
        return;
    }
    
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    if (lights.empty()) {
        // Upload zero-sized light buffer / set light count to 0 to clear GPU lights
        VK_INFO() << "[VulkanBackendAdapter] Clearing GPU lights (no lights in scene)" << std::endl;
        m_device->updateLightBuffer(nullptr, 0, 0);
        resetAccumulation();
        m_cachedLights = lights;
        return;
    }

    std::vector<::VulkanRT::VkGpuLight> gpuLights;
    gpuLights.reserve(lights.size());

    for (size_t li = 0; li < lights.size(); ++li) {
        const auto& l = lights[li];
        if (!l || !l->visible) continue;
        (void)0;
        ::VulkanRT::VkGpuLight gl{};
        gl.position[0] = l->position.x; gl.position[1] = l->position.y; gl.position[2] = l->position.z;
        // NOTE: GLSL shader expects types as: 0=Point, 1=Directional, 2=Area, 3=Spot
        // Our C++ enum is: 0=Point, 1=Directional, 2=Spot, 3=Area — map accordingly when uploading.
        float gpuType = 0.0f;
        switch (l->type()) {
            case LightType::Point: gpuType = 0.0f; break;
            case LightType::Directional: gpuType = 1.0f; break;
            case LightType::Spot: gpuType = 3.0f; break; // map Spot -> 3 for GLSL
            case LightType::Area: gpuType = 2.0f; break; // map Area -> 2 for GLSL
            default: gpuType = 0.0f; break;
        }
        gl.position[3] = gpuType;
        // Clamp intensity to sane range
        float intensity = isnan(l->intensity) ? 0.0f : l->intensity;
        if (intensity < 0.0f) intensity = 0.0f;
        if (intensity > 1e6f) intensity = 1e6f;

        // Always upload color and intensity distinctly (no premultiplied heuristic)
        gl.color[0] = l->color.x; gl.color[1] = l->color.y; gl.color[2] = l->color.z; gl.color[3] = intensity;
        // Direction vector: for directional lights negate stored direction (matches OptiX/CUDA convention)
        if (l->type() == LightType::Directional) {
            gl.direction[0] = -l->direction.x; gl.direction[1] = -l->direction.y; gl.direction[2] = -l->direction.z;
        } else {
            gl.direction[0] = l->direction.x; gl.direction[1] = l->direction.y; gl.direction[2] = l->direction.z;
        }
        // Default tail values
        gl.direction[3] = 0.0f; // outer cone (for spot) - filled below for Spot

        // Common params: radius, width, height. Some light types reinterpret these fields
        const float MIN_LIGHT_RADIUS = 1e-3f; // Avoid too-small radii that lead to sampling/precision issues in shaders
        const float MIN_AREA_DIM = 1e-4f;
        gl.params[0] = (std::max)(l->getRadius(), MIN_LIGHT_RADIUS);
        // AreaLight has its own width/height that shadow Light:: base members — must access via derived
        float w = 1.0f;
        float h = 1.0f;
        // Default u/v axes (overwritten for Area lights)
        gl.area_u[0] = 1.0f; gl.area_u[1] = 0.0f; gl.area_u[2] = 0.0f; gl.area_u[3] = 0.0f;
        gl.area_v[0] = 0.0f; gl.area_v[1] = 1.0f; gl.area_v[2] = 0.0f; gl.area_v[3] = 0.0f;
        if (l->type() == LightType::Area) {
            if (auto al = std::dynamic_pointer_cast<AreaLight>(l)) {
                w = al->getWidth();
                h = al->getHeight();
                Vec3 au = al->getU();
                Vec3 av = al->getV();
                gl.area_u[0] = au.x; gl.area_u[1] = au.y; gl.area_u[2] = au.z;
                gl.area_v[0] = av.x; gl.area_v[1] = av.y; gl.area_v[2] = av.z;
            }
        }
        gl.params[1] = (std::max)(w, MIN_AREA_DIM);
        gl.params[2] = (std::max)(h, MIN_AREA_DIM); // For area lights this is height, for spot lights this will be used for inner cone (overwritten below)

        // Spot lights require inner/outer cone cosines packed into params/direction.
        if (l->type() == LightType::Spot) {
            // Try to get SpotLight-specific angle information
            auto spot = std::dynamic_pointer_cast<SpotLight>(l);
            float outerCos = 0.0f;
            float innerCos = 1.0f;
            if (spot) {
                float angleDeg = spot->getAngleDegrees();
                float angleRad = angleDeg * (3.14159265358979323846f / 180.0f);
                outerCos = cosf(angleRad);
                innerCos = cosf(angleRad * 0.8f); // inner cone narrower (80% of outer)
            }
            // Shader expects inner cone in params.z and outer cone in direction.w
            gl.params[2] = innerCos;
            gl.direction[3] = outerCos;
        }
        gpuLights.push_back(gl);
    }

    m_cachedLights = lights;

    if (!gpuLights.empty()) {
        // Upload packed lights to GPU
        m_device->updateLightBuffer(gpuLights.data(), gpuLights.size() * sizeof(::VulkanRT::VkGpuLight), (uint32_t)gpuLights.size());
        resetAccumulation();
    }
}
void VulkanBackendAdapter::setRenderParams(const RenderParams& p) { 
    // Do NOT update m_imageWidth/Height here, otherwise renderProgressive detects no change
    const float clampedThreshold = std::clamp(p.adaptiveThreshold, 0.0f, 1.0f);
    const int clampedMinSamples = std::clamp(p.minSamples, 1, 4096);
    const int nextMaxBounces = (p.maxBounces > 0) ? p.maxBounces : m_maxBounces;
    const int nextDiffuseBounces = std::clamp(p.diffuseBounces, 1, nextMaxBounces);
    const int nextTransmissionBounces = std::clamp(p.transmissionBounces, 1, nextMaxBounces);
    if (m_targetSamples != p.samplesPerPixel ||
        m_minSamples != clampedMinSamples ||
        m_useAdaptiveSampling != p.useAdaptiveSampling ||
        m_maxBounces != nextMaxBounces ||
        m_diffuseBounces != nextDiffuseBounces ||
        m_transmissionBounces != nextTransmissionBounces ||
        m_causticsEnabled != p.causticsEnabled ||
        m_causticsDebug != p.causticsDebug ||
        m_causticsPhotons != p.causticsPhotons ||
        m_causticsCellSize != p.causticsCellSize ||
        m_causticsEnergy != p.causticsEnergy ||
        m_causticsVolumetric != p.causticsVolumetric ||
        m_causticsVolDebug != p.causticsVolDebug ||
        m_causticsVolStrength != p.causticsVolStrength ||
        m_causticsVolDirect != p.causticsVolDirect ||
        m_causticsVolNoise != p.causticsVolNoise) {
        resetAccumulation();
    }
    m_targetSamples = p.samplesPerPixel;
    m_minSamples = clampedMinSamples;
    m_useAdaptiveSampling = p.useAdaptiveSampling;
    m_varianceThreshold = clampedThreshold;
    m_maxBounces = nextMaxBounces; // 0 = UI henüz set etmedi, mevcut değeri koru
    m_diffuseBounces = nextDiffuseBounces;
    m_transmissionBounces = nextTransmissionBounces;
    m_causticsEnabled = p.causticsEnabled;
    m_causticsDebug = p.causticsDebug;
    m_causticsPhotons = p.causticsPhotons;
    m_causticsCellSize = p.causticsCellSize;
    m_causticsEnergy = p.causticsEnergy;
    m_causticsVolumetric = p.causticsVolumetric;
    m_causticsVolDebug = p.causticsVolDebug;
    m_causticsVolStrength = p.causticsVolStrength;
    m_causticsVolDirect = p.causticsVolDirect;
    m_causticsVolNoise = p.causticsVolNoise;
}

void VulkanBackendAdapter::setCamera(const CameraParams& c) { 
    m_camera = c;

    // Calculate Physical Exposure for Vulkan
    float factor = 1.0f;
    float ev_comp = std::pow(2.0f, c.ev_compensation);

    if (c.exposureFactor > 0.0f) {
        factor = c.exposureFactor;
    } else if (c.autoAE) {
        factor = ev_comp; 
    } else if (c.usePhysicalExposure) {
        float iso_mult = (c.isoPresetIndex >= 0 && c.isoPresetIndex < (int)CameraPresets::ISO_PRESET_COUNT) ? 
                         CameraPresets::ISO_PRESETS[c.isoPresetIndex].exposure_multiplier : 1.0f;
        float shutter_time = (c.shutterPresetIndex >= 0 && c.shutterPresetIndex < (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT) ? 
                             CameraPresets::SHUTTER_SPEED_PRESETS[c.shutterPresetIndex].speed_seconds : 0.004f;
        
        float f_number = 16.0f;
        if (c.fstopPresetIndex > 0 && c.fstopPresetIndex < (int)CameraPresets::FSTOP_PRESET_COUNT) {
             f_number = CameraPresets::FSTOP_PRESETS[c.fstopPresetIndex].f_number;
        }
        
        float aperture_sq = f_number * f_number;
        float current_val = (iso_mult * shutter_time) / (aperture_sq + 1e-6f);
        
        // Calibration: Boosted baseline to avoid black viewport
        float baseline_val = 0.00003125f; 
        factor = (current_val / baseline_val) * ev_comp * 2.0f;
    }
    
    m_camera.exposureFactor = factor;
    resetAccumulation(); 
}

void VulkanBackendAdapter::syncCamera(const Camera& cam) {
    // Convert CPU Camera to CameraParams and call setCamera()
    // This ensures all camera properties (including advanced cinema settings) are synchronized
    Backend::CameraParams cp{};
    cp.origin = cam.lookfrom;
    cp.lookAt = cam.lookat;
    cp.up = cam.vup;
    cp.fov = cam.vfov;
    cp.aperture = cam.aperture;
    cp.focusDistance = cam.focus_dist;
    cp.aspectRatio = cam.aspect;

    // Orthographic / standard-view state -> grid auto-orients to the active plane.
    cp.orthographic = cam.orthographic;
    cp.orthoHeight = cam.ortho_height;
    switch (cam.standard_view) {
        case Camera::StandardView::Front:
        case Camera::StandardView::Back:  cp.gridPlane = 1; break; // XY plane
        case Camera::StandardView::Left:
        case Camera::StandardView::Right: cp.gridPlane = 2; break; // YZ plane
        default:                          cp.gridPlane = 0; break; // XZ floor (Top/Bottom/Persp)
    }

    cp.isoPresetIndex = cam.iso_preset_index;
    cp.shutterPresetIndex = cam.shutter_preset_index;
    cp.fstopPresetIndex = cam.fstop_preset_index;
    cp.ev_compensation = cam.ev_compensation;
    cp.autoAE = cam.auto_exposure;
    cp.usePhysicalExposure = cam.use_physical_exposure;
    cp.exposureFactor = cam.getPhysicalExposureMultiplier();

    // Pro camera features
    cp.distortion = cam.distortion;
    cp.vignettingEnabled = cam.enable_vignetting;
    cp.vignetting_amount = cam.vignetting_amount;
    cp.vignetting_falloff = cam.vignetting_falloff;
    cp.chromaticAberrationEnabled = cam.enable_chromatic_aberration;
    cp.chromatic_aberration = cam.chromatic_aberration;
    cp.chromatic_aberration_r = cam.chromatic_aberration_r;
    cp.chromatic_aberration_b = cam.chromatic_aberration_b;
    cp.blade_count = cam.blade_count;
    cp.lens_quality = cam.lens_quality;
    cp.camera_mode = (int)cam.camera_mode;
    cp.motionBlurEnabled = cam.enable_motion_blur;

    cp.shake_enabled = cam.enable_camera_shake;
    cp.shake_intensity = cam.shake_intensity;
    cp.shake_frequency = cam.shake_frequency;
    cp.handheld_sway_amplitude = cam.handheld_sway_amplitude;
    cp.handheld_sway_frequency = cam.handheld_sway_frequency;
    cp.breathing_amplitude = cam.breathing_amplitude;
    cp.breathing_frequency = cam.breathing_frequency;
    cp.enable_focus_drift = cam.enable_focus_drift;
    cp.focus_drift_amount = cam.focus_drift_amount;
    cp.operator_skill = (int)cam.operator_skill;
    cp.ibis_enabled = cam.ibis_enabled;
    cp.ibis_effectiveness = cam.ibis_effectiveness;
    cp.rig_mode = (int)cam.rig_mode;
    
    setCamera(cp);
}

void VulkanBackendAdapter::setTime(float t, float dt) { m_currentTime = t; (void)dt; }



void VulkanBackendAdapter::updateInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) { 
    if (!m_device || !m_device->isInitialized()) return;
    if (objects.empty()) return;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Wind and other runtime transform edits can replace/refresh instance objects.
    // Rebuild the mapping here to avoid syncing TLAS against stale representatives.
    syncInstanceTransforms(objects, true);

    // ── Foam BVH-degradation guard ───────────────────────────────────────────────
    // Foam (point_sphere_mode) groups scatter as MANY fast-moving per-particle TLAS
    // instances. A per-frame TLAS REFIT keeps the frame-0 tree topology, so as the
    // foam disperses the BVH degenerates and path tracing chokes (the 10x slowdown
    // that a Solid→Rendered toggle — i.e. a full rebuild — cured). When foam is
    // present we therefore REBUILD the TLAS from scratch each frame instead of
    // refitting it: optimal BVH every frame, which is what made the toggle fast.
    bool foamPresent = false;
    for (const auto& g : InstanceManager::getInstance().getGroups()) {
        if (g.point_sphere_mode && !g.instances.empty()) { foamPresent = true; break; }
    }

    // Commit the merged instance list to the TLAS: a full rebuild when foam is
    // present (optimal BVH every frame, no degradation), else a cheap refit.
    // NOTE: createTLAS() does an in-place REFIT when allowUpdate==true AND the
    // instance count is unchanged — which is exactly the degradation we must avoid.
    // Passing allowUpdate=false forces its full-rebuild branch (old TLAS destroyed,
    // built fresh; the instance buffer is still reused in place, so no per-frame
    // alloc churn). This is what the Solid→Rendered toggle did to cure the slowdown.
    auto commitTLAS = [this](std::vector<VulkanRT::TLASInstance>& merged, bool rebuild) {
        if (rebuild) {
            VulkanRT::TLASCreateInfo ci;
            ci.instances   = merged;
            ci.allowUpdate = false;       // force a FULL BUILD → optimal BVH under foam motion
            m_device->createTLAS(ci);
        } else {
            m_device->updateTLAS(merged); // refit (cheap) for static / slow-moving scenes
        }
    };

    const auto& scatterGroups = InstanceManager::getInstance().getGroups();
    std::unordered_map<int, const InstanceGroup*> scatterGroupsById;
    scatterGroupsById.reserve(scatterGroups.size());
    for (const auto& group : scatterGroups) {
        if (!group.instances.empty()) {
            scatterGroupsById.emplace(group.id, &group);
        }
    }

    auto applyScatterTransforms = [&scatterGroupsById](std::vector<VulkanRT::TLASInstance>& instances) {
        constexpr size_t kParallelThreshold = 2048;
        unsigned threads = std::thread::hardware_concurrency();
        if (threads == 0) threads = 4;

        auto updateRange = [&instances, &scatterGroupsById](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                auto& vi = instances[i];
                if (vi.scatterGroupId < 0 || vi.scatterInstanceIndex == UINT32_MAX) continue;
                auto groupIt = scatterGroupsById.find(vi.scatterGroupId);
                if (groupIt == scatterGroupsById.end()) continue;
                const auto* group = groupIt->second;
                if (vi.scatterInstanceIndex >= group->instances.size()) continue;
                vi.transform = group->instances[vi.scatterInstanceIndex].toMatrix();
            }
        };

        if (instances.size() < kParallelThreshold || threads < 2) {
            updateRange(0, instances.size());
            return;
        }

        const size_t chunk = (instances.size() + threads - 1) / threads;
        std::vector<std::future<void>> futures;
        futures.reserve(threads);
        for (unsigned t = 0; t < threads; ++t) {
            const size_t s = t * chunk;
            const size_t e = std::min(s + chunk, instances.size());
            if (s >= e) break;
            futures.push_back(std::async(std::launch::async, updateRange, s, e));
        }
        for (auto& f : futures) f.get();
    };

    if (m_instance_sync_cache.empty() || m_instanceSources.size() != m_vkInstances.size()) {
        // Fallback to a conservative rebuild of full TLAS if mapping missing
        std::vector<VulkanRT::TLASInstance> updatedInstances = m_vkInstances;
        applyScatterTransforms(updatedInstances);
        // Try to rebuild order by scanning objects (fallback behavior)
        for (const auto& obj : objects) {
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                for (size_t i = 0; i < updatedInstances.size(); ++i) {
                    // match by node_name
                    if (m_instanceSources.size() > i && m_instanceSources[i]) {
                        if (auto srcInst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                            if (matchesNodeNameForInstance(srcInst->node_name, inst->node_name) ||
                                matchesNodeNameForInstance(inst->node_name, srcInst->node_name)) {
                                updatedInstances[i].transform = inst->transform;
                                uint16_t mId = inst->source_triangles && !inst->source_triangles->empty() ? inst->source_triangles->at(0)->getMaterialID() : MaterialManager::INVALID_MATERIAL_ID;
                                if (mId == MaterialManager::INVALID_MATERIAL_ID) mId = 0;
                                updatedInstances[i].materialIndex = mId;
                            }
                        }
                    }
                }
            }
        }
        bool transform_changed = false;
        bool data_changed = false;
        
        if (updatedInstances.size() != m_vkInstances.size()) {
            data_changed = true;
            transform_changed = true;
        } else {
            for (size_t i = 0; i < updatedInstances.size(); ++i) {
                if (!(updatedInstances[i].transform == m_vkInstances[i].transform)) {
                    transform_changed = true;
                }
                if (updatedInstances[i].blasIndex != m_vkInstances[i].blasIndex || 
                    updatedInstances[i].materialIndex != m_vkInstances[i].materialIndex) {
                    data_changed = true;
                }
            }
        }

        if (transform_changed || data_changed) {
            m_vkInstances = updatedInstances;

            // [VULKAN] Wait for device to finish any pending ray tracing before modifying AS
            m_device->waitIdle();

            { auto merged = m_vkInstances; for (const auto& h : m_hairVkInstances) merged.push_back(h); commitTLAS(merged, foamPresent); }

            if (data_changed) {
                // update instance SSBO (Binding 5)
                std::vector<VulkanRT::VkInstanceData> instData;
                for (const auto& vi : m_vkInstances) { VulkanRT::VkInstanceData d; d.materialIndex = vi.materialIndex; d.blasIndex = vi.blasIndex; instData.push_back(d); }
                if (m_device->m_instanceDataBuffer.buffer) m_device->destroyBuffer(m_device->m_instanceDataBuffer);
                ::VulkanRT::BufferCreateInfo ci; ci.size = (uint64_t)instData.size() * sizeof(::VulkanRT::VkInstanceData); ci.usage = (::VulkanRT::BufferUsage)((uint32_t)::VulkanRT::BufferUsage::STORAGE | (uint32_t)::VulkanRT::BufferUsage::TRANSFER_DST); ci.location = ::VulkanRT::MemoryLocation::CPU_TO_GPU; ci.initialData = instData.data(); m_device->m_instanceDataBuffer = m_device->createBuffer(ci);
                if (m_device->m_rtDescriptorSet != VK_NULL_HANDLE) { VkDescriptorBufferInfo instInfo{}; instInfo.buffer = m_device->m_instanceDataBuffer.buffer; instInfo.offset = 0; instInfo.range = VK_WHOLE_SIZE; VkWriteDescriptorSet w5{}; w5.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w5.dstSet = m_device->m_rtDescriptorSet; w5.dstBinding = 5; w5.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w5.descriptorCount = 1; w5.pBufferInfo = &instInfo; vkUpdateDescriptorSets(m_device->m_device, 1, &w5, 0, nullptr); }
            }
            resetAccumulation();
        }
        // Always sync raster instances for Solid/Matcap - even when TLAS is empty/unchanged
        if (!m_rasterInstances.empty() && shouldUseInteractiveViewport()) {
            syncRasterInstanceTransforms(objects);
        }
        return;
    }

    // Use cached mapping for efficient per-instance transform update
    std::vector<VulkanRT::TLASInstance> updated = m_vkInstances;
    applyScatterTransforms(updated);
    for (const auto& item : m_instance_sync_cache) {
        if (!item.representative_hittable) continue;
        Matrix4x4 m;
        if (auto tri = std::dynamic_pointer_cast<Triangle>(item.representative_hittable)) m = tri->getTransformMatrix();
        else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(item.representative_hittable)) m = inst->transform;
        // Flat (direct SoA) TriangleMesh-as-Hittable: drives its world transform through its own
        // Transform handle. Without this branch a keyframed / physics-driven flat mesh fell through
        // to `continue` and its TLAS instance transform was never refreshed per frame — it froze in
        // place during playback and only snapped to the right pose on the next full rebuild (stop).
        else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(item.representative_hittable)) {
            if (!tm->transform) continue;
            m = tm->transform->getFinal();
        }
        else continue;
        if (item.instance_id >= 0 && item.instance_id < (int)updated.size()) {
            updated[item.instance_id].transform = m;
            
            // Also sync material index in case it changed
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(item.representative_hittable)) {
                uint16_t mId = inst->source_triangles && !inst->source_triangles->empty() ? inst->source_triangles->at(0)->getMaterialID() : MaterialManager::INVALID_MATERIAL_ID;
                if (mId == MaterialManager::INVALID_MATERIAL_ID) mId = 0;
                updated[item.instance_id].materialIndex = mId;
            }
        }
    }

    bool transform_changed = false;
    bool data_changed = false;

    if (updated.size() != m_vkInstances.size()) {
        data_changed = true;
        transform_changed = true;
    } else {
        for (size_t i = 0; i < updated.size(); ++i) {
            if (!(updated[i].transform == m_vkInstances[i].transform)) {
                transform_changed = true;
            }
            if (updated[i].blasIndex != m_vkInstances[i].blasIndex || 
                updated[i].materialIndex != m_vkInstances[i].materialIndex) {
                data_changed = true;
            }
        }
    }

    if (transform_changed || data_changed) {
        m_vkInstances = updated;

        // [VULKAN] Must wait idle before modifying AS that is actively being traced
        m_device->waitIdle();

        {
            auto merged = m_vkInstances;
            for (const auto& h : m_hairVkInstances) merged.push_back(h);
            commitTLAS(merged, foamPresent);
        }

        if (data_changed) {
            // update instance SSBO (Binding 5)
            std::vector<VulkanRT::VkInstanceData> instData;
            for (const auto& vi : m_vkInstances) { VulkanRT::VkInstanceData d; d.materialIndex = vi.materialIndex; d.blasIndex = vi.blasIndex; instData.push_back(d); }
            if (m_device->m_instanceDataBuffer.buffer) m_device->destroyBuffer(m_device->m_instanceDataBuffer);
            ::VulkanRT::BufferCreateInfo ci; ci.size = (uint64_t)instData.size() * sizeof(::VulkanRT::VkInstanceData); ci.usage = (::VulkanRT::BufferUsage)((uint32_t)::VulkanRT::BufferUsage::STORAGE | (uint32_t)::VulkanRT::BufferUsage::TRANSFER_DST); ci.location = ::VulkanRT::MemoryLocation::CPU_TO_GPU; ci.initialData = instData.data(); m_device->m_instanceDataBuffer = m_device->createBuffer(ci);
            if (m_device->m_rtDescriptorSet != VK_NULL_HANDLE) { VkDescriptorBufferInfo instInfo{}; instInfo.buffer = m_device->m_instanceDataBuffer.buffer; instInfo.offset = 0; instInfo.range = VK_WHOLE_SIZE; VkWriteDescriptorSet w5{}; w5.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w5.dstSet = m_device->m_rtDescriptorSet; w5.dstBinding = 5; w5.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w5.descriptorCount = 1; w5.pBufferInfo = &instInfo; vkUpdateDescriptorSets(m_device->m_device, 1, &w5, 0, nullptr); }
        }
        resetAccumulation();
    }
    // Always sync raster instances for Solid/Matcap - even when TLAS is empty/unchanged
    if (!m_rasterInstances.empty() && shouldUseInteractiveViewport()) {
        syncRasterInstanceTransforms(objects);
    }
}

void VulkanBackendAdapter::syncInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects, bool force_rebuild_cache) {
    if (objects.empty()) return;

    if (m_topology_dirty || m_instance_sync_cache.empty() || force_rebuild_cache) {
        m_instance_sync_cache.clear();

        // 1. Build a Quick Lookup Map for Scene Objects by Pointer and Name
        // This makes the matching O(N) instead of O(N^2).
        std::unordered_map<void*, std::shared_ptr<Hittable>> ptr_to_obj;
        std::unordered_map<std::string, std::shared_ptr<HittableInstance>> name_to_inst;
        
        for (const auto& obj : objects) {
            if (!obj) continue;
            ptr_to_obj[obj.get()] = obj;
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                if (!inst->node_name.empty()) name_to_inst[inst->node_name] = inst;
            }
        }

        // 2. Iterate TLAS sources and find current matches
        for (size_t i = 0; i < m_instanceSources.size(); ++i) {
            auto src = m_instanceSources[i];
            if (!src) continue;

            VulkanBackendAdapter::InstanceTransformCache item;
            item.instance_id = (int)i;
            item.representative_hittable = nullptr;

            // Try direct pointer match first (fastest)
            auto itPtr = ptr_to_obj.find(src.get());
            if (itPtr != ptr_to_obj.end()) {
                item.representative_hittable = itPtr->second;
            } else {
                // Fallback for instances with same node name (e.g. after re-import or logic change)
                if (auto inst = std::dynamic_pointer_cast<HittableInstance>(src)) {
                    auto itName = name_to_inst.find(inst->node_name);
                    if (itName != name_to_inst.end()) {
                        item.representative_hittable = itName->second;
                    }
                }
            }

            if (item.representative_hittable) {
                m_instance_sync_cache.push_back(item);
            }
        }

        m_topology_dirty = false;
    }
}

bool VulkanBackendAdapter::isUsingTLAS() const {
    return m_device && m_device->isInitialized() && m_device->hasHardwareRT();
}

std::vector<int> VulkanBackendAdapter::getInstancesByNodeName(const std::string& nodeName) const {
    std::vector<int> ids;
    for (size_t i = 0; i < m_instanceSources.size(); ++i) {
        if (m_instanceSources[i]) {
            std::string instName;
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                instName = inst->node_name;
            } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
                instName = tri->getNodeName();
            } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
                // Flat (direct SoA) mesh: without this branch a keyframed flat mesh wasn't found
                // here, so the cheap per-instance TLAS transform update (updateInstanceTransform)
                // was skipped and playback fell back to a full updateSceneGeometry rebuild every
                // frame — the "flat keyframe is much heavier than facade" cost.
                instName = tm->nodeName;
            }

            if (matchesNodeNameForInstance(instName, nodeName)) {
                ids.push_back(static_cast<int>(i));
            }
        }
    }
    return ids;
}

void VulkanBackendAdapter::updateInstanceTransform(int instance_id, const float transform[12]) {
    if (instance_id < 0 || instance_id >= static_cast<int>(m_vkInstances.size())) return;

    m_vkInstances[instance_id].transform.m[0][0] = transform[0]; m_vkInstances[instance_id].transform.m[0][1] = transform[1]; m_vkInstances[instance_id].transform.m[0][2] = transform[2]; m_vkInstances[instance_id].transform.m[0][3] = transform[3];
    m_vkInstances[instance_id].transform.m[1][0] = transform[4]; m_vkInstances[instance_id].transform.m[1][1] = transform[5]; m_vkInstances[instance_id].transform.m[1][2] = transform[6]; m_vkInstances[instance_id].transform.m[1][3] = transform[7];
    m_vkInstances[instance_id].transform.m[2][0] = transform[8]; m_vkInstances[instance_id].transform.m[2][1] = transform[9]; m_vkInstances[instance_id].transform.m[2][2] = transform[10]; m_vkInstances[instance_id].transform.m[2][3] = transform[11];
    m_vkInstances[instance_id].transform.m[3][0] = 0.0f; m_vkInstances[instance_id].transform.m[3][1] = 0.0f; m_vkInstances[instance_id].transform.m[3][2] = 0.0f; m_vkInstances[instance_id].transform.m[3][3] = 1.0f;

    if (m_instanceSources.size() > instance_id && m_instanceSources[instance_id]) {
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[instance_id])) {
            inst->setTransform(m_vkInstances[instance_id].transform);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[instance_id])) {
            tri->setBaseTransform(m_vkInstances[instance_id].transform);
        }
    }
}

void VulkanBackendAdapter::invalidateTargetedTransformIndex() {
    m_rasterNodeIndex.clear();
    m_rtNodeIndex.clear();
    m_rasterNodeIndexInstanceCount = 0;
    m_rtNodeIndexInstanceCount = 0;
}

void VulkanBackendAdapter::rebuildTargetedTransformIndex() {
    if (m_rasterNodeIndexInstanceCount == m_rasterInstances.size() &&
        m_rtNodeIndexInstanceCount == m_instanceSources.size()) {
        return;
    }

    m_rasterNodeIndex.clear();
    m_rtNodeIndex.clear();

    auto addNodeIndex = [](std::unordered_map<std::string, std::vector<uint32_t>>& index,
                           const std::string& nodeName,
                           uint32_t instanceIndex) {
        if (nodeName.empty()) return;
        index[nodeName].push_back(instanceIndex);

        const std::string matToken = "_mat_";
        const size_t matPos = nodeName.find(matToken);
        if (matPos != std::string::npos && matPos > 0) {
            const std::string baseName = nodeName.substr(0, matPos);
            if (!baseName.empty() && baseName != nodeName) {
                index[baseName].push_back(instanceIndex);
            }
        }
    };

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_instanceSources.size()); ++i) {
        if (!m_instanceSources[i]) continue;

        std::string instName;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            instName = inst->node_name;
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            instName = tri->getNodeName();
        } else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            // Flat/proxy: a dense direct TriangleMesh instance — index it by node name so
            // updateObjectTransform() can find and refresh its TLAS transform on a gizmo move.
            instName = mesh->nodeName;
        }
        addNodeIndex(m_rtNodeIndex, instName, i);
    }

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_rasterInstances.size()); ++i) {
        addNodeIndex(m_rasterNodeIndex, m_rasterInstances[i].nodeName, i);
    }

    m_rasterNodeIndexInstanceCount = m_rasterInstances.size();
    m_rtNodeIndexInstanceCount = m_instanceSources.size();
}

void VulkanBackendAdapter::updateObjectTransform(const std::string& nodeName, const Matrix4x4& transform) {
    if (!m_device || !m_device->isInitialized()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    rebuildTargetedTransformIndex();

    bool changed = false;
    std::unordered_set<std::string> dirtyRasterMeshes;

    // Update RT instances (when hardware RT is available)
    auto updateRtInstance = [&](size_t i) {
        if (i >= m_vkInstances.size()) return;
        if (m_instanceSources.size() > i && m_instanceSources[i]) {
            m_vkInstances[i].transform = transform;
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                inst->setTransform(transform);
            } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
                tri->setBaseTransform(transform);
            }
            changed = true;
        }
    };

    bool usedRtIndex = false;
    auto rtIt = m_rtNodeIndex.find(nodeName);
    if (rtIt != m_rtNodeIndex.end()) {
        usedRtIndex = true;
        for (uint32_t i : rtIt->second) {
            updateRtInstance(i);
        }
    }
    if (!usedRtIndex) {
        for (size_t i = 0; i < m_vkInstances.size(); ++i) {
            if (m_instanceSources.size() > i && m_instanceSources[i]) {
                std::string instName;
                if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                    instName = inst->node_name;
                } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
                    instName = tri->getNodeName();
                } else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
                    instName = mesh->nodeName; // Flat/proxy: direct TriangleMesh instance
                }

                if (matchesNodeNameForInstance(instName, nodeName)) {
                    updateRtInstance(i);
                }
            }
        }
    }

    // Always sync raster instance transforms (for Solid/Matcap viewport)
    auto updateRasterInstance = [&](uint32_t i) {
        if (i >= m_rasterInstances.size()) return;
        auto& ri = m_rasterInstances[i];
        if (!(ri.transform == transform)) {
            ri.transform = transform;
            updateRasterInstanceWorldBBox(ri);
            dirtyRasterMeshes.insert(ri.meshKey);
            changed = true;
        }
    };

    bool usedRasterIndex = false;
    auto rasterIt = m_rasterNodeIndex.find(nodeName);
    if (rasterIt != m_rasterNodeIndex.end()) {
        usedRasterIndex = true;
        for (uint32_t i : rasterIt->second) {
            updateRasterInstance(i);
        }
    }
    if (!usedRasterIndex) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_rasterInstances.size()); ++i) {
            if (matchesNodeNameForInstance(m_rasterInstances[i].nodeName, nodeName)) {
                updateRasterInstance(i);
            }
        }
    }

    if (changed) {
        for (const auto& meshKey : dirtyRasterMeshes) {
            auto meshIt = m_rasterMeshes.find(meshKey);
            if (meshIt != m_rasterMeshes.end()) {
                uploadRasterInstanceBuffer(meshIt->second);
            }
        }
        if (m_viewportMode == ViewportMode::Rendered && m_device->hasHardwareRT()) {
            // Only update TLAS in rendered mode (expensive, not needed for solid rasterization)
            m_device->waitIdle();
            auto merged = m_vkInstances;
            for (const auto& h : m_hairVkInstances) merged.push_back(h);
            m_device->updateTLAS(merged);
        }
        resetAccumulation();
    }
}
void VulkanBackendAdapter::setStatusCallback(std::function<void(const std::string&, int)> cb) { m_statusCallback = cb; }
void* VulkanBackendAdapter::getNativeCommandQueue() { return (void*)m_device->getComputeQueue(); }

void VulkanBackendAdapter::renderPass(bool accumulate) { (void)accumulate; /* TODO */ }

// ============================================================================
// Raster mesh buffers — lightweight vertex/normal/index for solid mode
// ============================================================================

void VulkanBackendAdapter::buildRasterGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) {
    buildRasterGeometryImpl(objects);
}

void VulkanBackendAdapter::syncRasterInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) {
    syncRasterInstanceTransformsImpl(objects);
}

void VulkanBackendAdapter::syncRasterSkinnedVertices(
    const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::vector<Matrix4x4>& boneMatrices) {
    syncRasterSkinnedVerticesImpl(objects, boneMatrices);
}

bool VulkanBackendAdapter::updateRasterMeshFromTriangles(
    const std::string& nodeName,
    const std::vector<std::shared_ptr<Triangle>>& triangles) {
    return updateRasterMeshFromTrianglesImpl(nodeName, triangles);
}

bool VulkanBackendAdapter::patchRasterMeshTriangles(
    const std::string& nodeName,
    const std::vector<size_t>& dirtyIndices,
    const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries) {
    return patchRasterMeshTrianglesImpl(nodeName, dirtyIndices, meshEntries);
}

bool VulkanBackendAdapter::shouldUseInteractiveViewport() const {
    return shouldUseInteractiveViewportImpl();
}

bool VulkanBackendAdapter::ensureInteractiveViewportResources(const std::string& shaderDir, int width, int height) {
    return ensureInteractiveViewportResourcesImpl(shaderDir, width, height);
}

void VulkanBackendAdapter::destroyInteractiveViewportResources(bool keepPipeline) {
    destroyInteractiveViewportResourcesImpl(keepPipeline);
}

void VulkanBackendAdapter::resetForProjectReload() {
    // Block the OIDN viewport denoiser from touching backend CUDA buffers while
    // we tear down/rebuild. Without this, the render thread keeps calling
    // applyOIDNDenoisingGPU against the now-freed device pointers from
    // getDenoiserFrameGPU, hits cudaErrorIllegalAddress (700), and poisons the
    // CUDA context — which then takes down the OptiX rebuild that runs after.
    g_viewport_rebuild_in_progress.store(true, std::memory_order_release);
    // Order matters: the material preview descriptor set (binding 1) references
    // VkImageViews from m_uploadedImages. rebuildAccelerationStructure() destroys
    // those images. If we didn't tear down the interactive viewport resources
    // first, the next material-preview draw would sample dead views and crash the
    // driver. Destroying them here forces ensureInteractiveViewportResourcesImpl
    // to rebuild and backfill on the next frame from the freshly uploaded textures.
    if (m_device) {
        try { m_device->waitIdle(); } catch (...) {}
    }
    destroyInteractiveViewportResourcesImpl(false);
    // Zero out the interactive viewport state struct so cached handles
    // (matcapImage, matcapUserLoaded, dirty flags, etc.) don't survive into
    // the next ensure pass. rebuildAccelerationStructure() is about to destroy
    // every VkImage in m_uploadedImages; if matcapImage still referenced one
    // of them, the next ensure would write a dead VkImageView into the
    // matcapDescSet via vkUpdateDescriptorSets and crash the driver. This
    // mirrors the pattern used in the viewport-mode-switch path (see
    // ViewportMode transition handling).
    m_interactiveViewport = {};
    m_interactiveViewport.dirty = true;
    rebuildAccelerationStructure();
    g_viewport_rebuild_in_progress.store(false, std::memory_order_release);
}

void VulkanBackendAdapter::renderInteractiveViewport(void* outSurface, int width, int height,
                                                     void* outFramebuffer, void* outTexture) {
    renderInteractiveViewportImpl(outSurface, width, height, outFramebuffer, outTexture);
}

void VulkanBackendAdapter::renderProgressive(void* s, void* w, void* r, int width, int height, void* fb, void* tex) {
    renderProgressiveImpl(s, w, r, width, height, fb, tex);
}

void VulkanBackendAdapter::destroyRasterMesh(RasterMeshBuffer& mesh) {
    if (!m_device) return;
    if (mesh.vertexBuffer.buffer) m_device->destroyBuffer(mesh.vertexBuffer);
    if (mesh.normalBuffer.buffer) m_device->destroyBuffer(mesh.normalBuffer);
    if (mesh.uvBuffer.buffer) m_device->destroyBuffer(mesh.uvBuffer);
    if (mesh.matIdBuffer.buffer) m_device->destroyBuffer(mesh.matIdBuffer);
    if (mesh.indexBuffer.buffer)  m_device->destroyBuffer(mesh.indexBuffer);
    if (mesh.instanceBuffer.buffer) m_device->destroyBuffer(mesh.instanceBuffer);
    if (mesh.baseVertexBuffer.buffer) m_device->destroyBuffer(mesh.baseVertexBuffer);
    if (mesh.baseNormalBuffer.buffer) m_device->destroyBuffer(mesh.baseNormalBuffer);
    if (mesh.boneIndexBuffer.buffer) m_device->destroyBuffer(mesh.boneIndexBuffer);
    if (mesh.boneWeightBuffer.buffer) m_device->destroyBuffer(mesh.boneWeightBuffer);
    if (mesh.persistentBoneMatsBuffer.buffer) m_device->destroyBuffer(mesh.persistentBoneMatsBuffer);
    mesh = RasterMeshBuffer{};
}

void VulkanBackendAdapter::destroyAllRasterMeshes() {
    for (auto& [key, mesh] : m_rasterMeshes) {
        destroyRasterMesh(mesh);
    }
    m_rasterMeshes.clear();
    m_rasterInstances.clear();
}

// ─────────────────────────────────────────────────────────────────────
// Frustum Culling for Solid Viewport
// ─────────────────────────────────────────────────────────────────────

void VulkanBackendAdapter::extractFrustumPlanes(const Matrix4x4& vp) {
    const auto* vpBytes = reinterpret_cast<const unsigned char*>(&vp);
    uint64_t frustumHash = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof(Matrix4x4); ++i) {
        frustumHash ^= static_cast<uint64_t>(vpBytes[i]);
        frustumHash *= 1099511628211ull;
    }
    if (frustumHash != m_rasterFrustumHash) {
        m_rasterFrustumHash = frustumHash;
        ++m_rasterFrustumRevision;
    }

    // Extract 6 frustum planes from the view-projection matrix (row-major).
    // Each plane: dot(normal, point) + d >= 0 means inside.
    auto row = [&](int r, int c) -> float { return vp.m[r][c]; };

    // Left:   row3 + row0
    m_frustumPlanes[0].normal = Vec3(row(3,0)+row(0,0), row(3,1)+row(0,1), row(3,2)+row(0,2));
    m_frustumPlanes[0].d      =      row(3,3)+row(0,3);
    // Right:  row3 - row0
    m_frustumPlanes[1].normal = Vec3(row(3,0)-row(0,0), row(3,1)-row(0,1), row(3,2)-row(0,2));
    m_frustumPlanes[1].d      =      row(3,3)-row(0,3);
    // Bottom: row3 + row1
    m_frustumPlanes[2].normal = Vec3(row(3,0)+row(1,0), row(3,1)+row(1,1), row(3,2)+row(1,2));
    m_frustumPlanes[2].d      =      row(3,3)+row(1,3);
    // Top:    row3 - row1
    m_frustumPlanes[3].normal = Vec3(row(3,0)-row(1,0), row(3,1)-row(1,1), row(3,2)-row(1,2));
    m_frustumPlanes[3].d      =      row(3,3)-row(1,3);
    // Near:   row3 + row2
    m_frustumPlanes[4].normal = Vec3(row(3,0)+row(2,0), row(3,1)+row(2,1), row(3,2)+row(2,2));
    m_frustumPlanes[4].d      =      row(3,3)+row(2,3);
    // Far:    row3 - row2
    m_frustumPlanes[5].normal = Vec3(row(3,0)-row(2,0), row(3,1)-row(2,1), row(3,2)-row(2,2));
    m_frustumPlanes[5].d      =      row(3,3)-row(2,3);

    // Normalize planes
    for (auto& p : m_frustumPlanes) {
        float len = p.normal.length();
        if (len > 1e-6f) {
            p.normal = p.normal / len;
            p.d /= len;
        }
    }
}

bool VulkanBackendAdapter::isAABBInFrustum(const AABB& box) const {
    const Vec3 halfExtents = (box.max - box.min) * 0.5f;
    const float conservativeSlack = std::max(halfExtents.length() * 0.75f, 0.05f);
    for (int i = 0; i < 6; ++i) {
        if (i == 4) {
            // Skip CPU-side near-plane rejection for raster viewport culling.
            // When the camera gets very close to a surface, object AABBs often
            // straddle the near plane even though a visible portion should still
            // render. Let the GPU clipper handle the near plane instead of
            // dropping the whole mesh/chunk on the CPU side.
            continue;
        }
        const auto& pl = m_frustumPlanes[i];
        // P-vertex: the corner of AABB most in the direction of the plane normal
        Vec3 pVertex(
            pl.normal.x >= 0 ? box.max.x : box.min.x,
            pl.normal.y >= 0 ? box.max.y : box.min.y,
            pl.normal.z >= 0 ? box.max.z : box.min.z
        );
        if (Vec3::dot(pl.normal, pVertex) + pl.d < -conservativeSlack)
            return false;
    }
    return true;
}

bool VulkanBackendAdapter::isAABBFullyInsideFrustum(const AABB& box) const {
    if (!box.is_valid()) return false;
    const Vec3 halfExtents = (box.max - box.min) * 0.5f;
    const float conservativeSlack = std::max(halfExtents.length() * 0.35f, 0.02f);
    for (int i = 0; i < 6; ++i) {
        if (i == 4) {
            continue;
        }
        const auto& pl = m_frustumPlanes[i];
        Vec3 nVertex(
            pl.normal.x >= 0 ? box.min.x : box.max.x,
            pl.normal.y >= 0 ? box.min.y : box.max.y,
            pl.normal.z >= 0 ? box.min.z : box.max.z
        );
        if (Vec3::dot(pl.normal, nVertex) + pl.d < -conservativeSlack)
            return false;
    }
    return true;
}

bool VulkanBackendAdapter::isRasterChunkTooSmallToDraw(const AABB& box) const {
    if (!box.is_valid() || m_rasterCullFocalLengthPixels <= 0.0f ||
        m_rasterMinChunkScreenRadiusPixels <= 0.0f) {
        return false;
    }

    const Vec3 center = (box.min + box.max) * 0.5f;
    const Vec3 halfExtents = (box.max - box.min) * 0.5f;
    const float radius = halfExtents.length();
    if (radius <= 0.0f) {
        return false;
    }

    const float distance = (center - m_rasterCullCameraPosition).length();
    if (distance <= radius) {
        return false;
    }

    const float projectedRadiusPixels = (radius * m_rasterCullFocalLengthPixels) / distance;
    return projectedRadiusPixels < m_rasterMinChunkScreenRadiusPixels;
}

void VulkanBackendAdapter::updateRasterInstanceWorldBBox(RasterInstance& ri) const {
    // Transform local AABB 8 corners to world space, compute enclosing AABB
    const AABB& lb = ri.localBBox;
    if (!lb.is_valid()) {
        ri.worldBBox = lb;
        return;
    }
    const Vec3 corners[8] = {
        Vec3(lb.min.x, lb.min.y, lb.min.z),
        Vec3(lb.max.x, lb.min.y, lb.min.z),
        Vec3(lb.min.x, lb.max.y, lb.min.z),
        Vec3(lb.max.x, lb.max.y, lb.min.z),
        Vec3(lb.min.x, lb.min.y, lb.max.z),
        Vec3(lb.max.x, lb.min.y, lb.max.z),
        Vec3(lb.min.x, lb.max.y, lb.max.z),
        Vec3(lb.max.x, lb.max.y, lb.max.z),
    };
    Vec3 wMin(1e18f, 1e18f, 1e18f), wMax(-1e18f, -1e18f, -1e18f);
    for (const auto& c : corners) {
        Vec3 wc = ri.transform.transform_point(c);
        wMin.x = std::min(wMin.x, wc.x);
        wMin.y = std::min(wMin.y, wc.y);
        wMin.z = std::min(wMin.z, wc.z);
        wMax.x = std::max(wMax.x, wc.x);
        wMax.y = std::max(wMax.y, wc.y);
        wMax.z = std::max(wMax.z, wc.z);
    }
    ri.worldBBox = AABB(wMin, wMax);
}

void VulkanBackendAdapter::rebuildRasterMeshCullingChunks(RasterMeshBuffer& mesh) {
    mesh.cullingChunks.clear();
    if (mesh.instanceIndices.size() < 128) {
        return;
    }

    float baseCellSize = 0.0f;
    for (uint32_t instanceIndex : mesh.instanceIndices) {
        if (instanceIndex >= m_rasterInstances.size()) continue;
        const AABB& localBBox = m_rasterInstances[instanceIndex].localBBox;
        if (!localBBox.is_valid()) continue;
        const float extentX = std::max(0.0f, localBBox.max.x - localBBox.min.x);
        const float extentZ = std::max(0.0f, localBBox.max.z - localBBox.min.z);
        baseCellSize = std::max(baseCellSize, std::max(extentX, extentZ));
        if (baseCellSize > 0.0f) break;
    }

    const float cellSize = std::clamp(baseCellSize > 0.0f ? baseCellSize * 8.0f : 8.0f, 4.0f, 96.0f);

    struct ChunkKey {
        int32_t x;
        int32_t z;
        bool operator==(const ChunkKey& other) const {
            return x == other.x && z == other.z;
        }
    };
    struct ChunkKeyHash {
        size_t operator()(const ChunkKey& key) const {
            return (static_cast<size_t>(static_cast<uint32_t>(key.x)) << 32) ^
                   static_cast<size_t>(static_cast<uint32_t>(key.z));
        }
    };

    std::unordered_map<ChunkKey, size_t, ChunkKeyHash> chunkLookup;
    chunkLookup.reserve(mesh.instanceIndices.size() / 8 + 1);

    for (uint32_t instanceIndex : mesh.instanceIndices) {
        if (instanceIndex >= m_rasterInstances.size()) continue;
        const auto& ri = m_rasterInstances[instanceIndex];
        if (ri.mask == 0 || !ri.worldBBox.is_valid()) continue;

        const float centerX = (ri.worldBBox.min.x + ri.worldBBox.max.x) * 0.5f;
        const float centerZ = (ri.worldBBox.min.z + ri.worldBBox.max.z) * 0.5f;
        const ChunkKey key{
            static_cast<int32_t>(std::floor(centerX / cellSize)),
            static_cast<int32_t>(std::floor(centerZ / cellSize))
        };

        auto [it, inserted] = chunkLookup.emplace(key, mesh.cullingChunks.size());
        if (inserted) {
            mesh.cullingChunks.emplace_back();
            mesh.cullingChunks.back().worldBBox = ri.worldBBox;
        } else {
            auto& bbox = mesh.cullingChunks[it->second].worldBBox;
            bbox.min.x = std::min(bbox.min.x, ri.worldBBox.min.x);
            bbox.min.y = std::min(bbox.min.y, ri.worldBBox.min.y);
            bbox.min.z = std::min(bbox.min.z, ri.worldBBox.min.z);
            bbox.max.x = std::max(bbox.max.x, ri.worldBBox.max.x);
            bbox.max.y = std::max(bbox.max.y, ri.worldBBox.max.y);
            bbox.max.z = std::max(bbox.max.z, ri.worldBBox.max.z);
        }

        mesh.cullingChunks[it->second].instanceIndices.push_back(instanceIndex);
    }

    if (mesh.cullingChunks.size() <= 1) {
        mesh.cullingChunks.clear();
    }
}

void VulkanBackendAdapter::setRasterVisibleInstances(RasterMeshBuffer& mesh,
                                                     const std::vector<uint32_t>& visibleInstanceIndices) {
    struct RasterInstanceGPU {
        float model[16];
    };

    const bool visibleSetChanged =
        visibleInstanceIndices.size() != mesh.visibleInstanceIndicesCache.size() ||
        !std::equal(visibleInstanceIndices.begin(), visibleInstanceIndices.end(),
                    mesh.visibleInstanceIndicesCache.begin(), mesh.visibleInstanceIndicesCache.end());

    mesh.instanceCount = static_cast<uint32_t>(visibleInstanceIndices.size());
    if (mesh.instanceCount == 0) {
        mesh.visibleInstanceIndicesCache.clear();
        mesh.visibleInstancesDirty = false;
        mesh.lastVisibleFrustumRevision = m_rasterFrustumRevision;
        mesh.lastScatterTriangleBudget = m_rasterScatterTriangleBudget;
        return;
    }

    if (!mesh.visibleInstancesDirty && !visibleSetChanged && mesh.instanceBuffer.buffer) {
        return;
    }

    auto matrixToGL = [](const Matrix4x4& mat, float out[16]) {
        Matrix4x4 t = mat.transpose();
        int k = 0;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[k++] = t.m[r][c];
            }
        }
    };

    std::vector<RasterInstanceGPU> gpuInstances(mesh.instanceCount);
    const size_t kParallelMatrixThreshold = 4096;
    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    auto fillGpuRange = [this, &visibleInstanceIndices, &gpuInstances, &matrixToGL](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const uint32_t instanceIndex = visibleInstanceIndices[i];
            if (instanceIndex >= m_rasterInstances.size()) continue;
            matrixToGL(m_rasterInstances[instanceIndex].transform, gpuInstances[i].model);
        }
    };
    if (visibleInstanceIndices.size() < kParallelMatrixThreshold || numThreads < 2) {
        fillGpuRange(0, visibleInstanceIndices.size());
    } else {
        const size_t chunk = (visibleInstanceIndices.size() + numThreads - 1) / numThreads;
        std::vector<std::future<void>> futures;
        futures.reserve(numThreads);
        for (unsigned t = 0; t < numThreads; ++t) {
            const size_t s = t * chunk;
            const size_t e = std::min(s + chunk, visibleInstanceIndices.size());
            if (s >= e) break;
            futures.push_back(std::async(std::launch::async, fillGpuRange, s, e));
        }
        for (auto& f : futures) f.get();
    }

    mesh.visibleInstanceIndicesCache = visibleInstanceIndices;
    mesh.visibleInstancesDirty = false;
    mesh.lastVisibleFrustumRevision = m_rasterFrustumRevision;
    mesh.lastScatterTriangleBudget = m_rasterScatterTriangleBudget;

    const VkDeviceSize requiredSize = gpuInstances.size() * sizeof(RasterInstanceGPU);
    if (mesh.instanceBuffer.buffer && mesh.instanceBuffer.size >= requiredSize) {
        m_device->uploadBuffer(mesh.instanceBuffer, gpuInstances.data(), requiredSize, 0);
        return;
    }

    if (mesh.instanceBuffer.buffer) {
        m_device->destroyBuffer(mesh.instanceBuffer);
        mesh.instanceBuffer = VulkanRT::BufferHandle{};
    }

    const VkDeviceSize allocCapacity = std::max<size_t>(
        visibleInstanceIndices.size(), mesh.instanceIndices.size()) * sizeof(RasterInstanceGPU);
    VulkanRT::BufferCreateInfo ici{};
    ici.size = std::max(requiredSize, allocCapacity);
    ici.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
    ici.location = VulkanRT::MemoryLocation::GPU_ONLY;
    ici.initialData = nullptr;
    mesh.instanceBuffer = m_device->createBuffer(ici);
    if (mesh.instanceBuffer.buffer) {
        m_device->uploadBuffer(mesh.instanceBuffer, gpuInstances.data(), requiredSize, 0);
    }
}

void VulkanBackendAdapter::uploadVisibleRasterInstances(RasterMeshBuffer& mesh) {
    if (!m_device) return;
    if (mesh.isScatterProxy) return;

    const bool proxySplitStale =
        mesh.isScatterGroup &&
        !mesh.proxyMeshKey.empty() &&
        (mesh.lastVisibleFrustumRevision != m_rasterFrustumRevision ||
         mesh.lastScatterTriangleBudget != m_rasterScatterTriangleBudget);

    if (!mesh.visibleInstancesDirty &&
        !proxySplitStale &&
        (!kRasterFrustumCullingEnabled || mesh.lastVisibleFrustumRevision == m_rasterFrustumRevision) &&
        (mesh.instanceBuffer.buffer || mesh.visibleInstanceIndicesCache.empty())) {
        mesh.instanceCount = static_cast<uint32_t>(mesh.visibleInstanceIndicesCache.size());
        if (!kRasterFrustumCullingEnabled) {
            mesh.lastVisibleFrustumRevision = m_rasterFrustumRevision;
        }
        return;
    }

    std::vector<uint32_t> visibleInstanceIndices;
    visibleInstanceIndices.reserve(mesh.instanceIndices.size());

    auto appendVisibleInstance = [&](uint32_t instanceIndex, bool skipFrustumTest) {
        if (instanceIndex >= m_rasterInstances.size()) return;
        const auto& ri = m_rasterInstances[instanceIndex];
        if (ri.mask == 0) return;
        if (kRasterFrustumCullingEnabled &&
            !skipFrustumTest &&
            !mesh.hasSkinning &&
            ri.worldBBox.is_valid() &&
            !isAABBInFrustum(ri.worldBBox)) {
            return;
        }
        visibleInstanceIndices.push_back(instanceIndex);
    };

    if (kRasterFrustumCullingEnabled && !mesh.hasSkinning && !mesh.cullingChunks.empty()) {
        for (const auto& chunk : mesh.cullingChunks) {
            if (chunk.worldBBox.is_valid() && !isAABBInFrustum(chunk.worldBBox)) {
                continue;
            }
            if (chunk.instanceIndices.size() >= 16 && isRasterChunkTooSmallToDraw(chunk.worldBBox)) {
                continue;
            }
            const bool chunkFullyInside = chunk.worldBBox.is_valid() && isAABBFullyInsideFrustum(chunk.worldBBox);
            for (uint32_t instanceIndex : chunk.instanceIndices) {
                appendVisibleInstance(instanceIndex, chunkFullyInside);
            }
        }
    } else {
        for (uint32_t instanceIndex : mesh.instanceIndices) {
            appendVisibleInstance(instanceIndex, false);
        }
    }

    std::vector<uint32_t> proxyInstanceIndices;
    if (mesh.isScatterGroup) {
        const uint64_t trianglesPerInstance = (mesh.indexBuffer.buffer && mesh.indexCount > 0)
            ? (static_cast<uint64_t>(mesh.indexCount) / 3ull)
            : (static_cast<uint64_t>(mesh.vertexCount) / 3ull);
        const uint64_t scatterTriangleBudget = std::max<uint64_t>(1ull, m_rasterScatterTriangleBudget);
        if (trianglesPerInstance > 0) {
            const uint64_t visibleTriangles = trianglesPerInstance * static_cast<uint64_t>(visibleInstanceIndices.size());
            if (visibleTriangles > scatterTriangleBudget) {
                const size_t cappedVisibleCount = static_cast<size_t>(std::max<uint64_t>(
                    1ull, scatterTriangleBudget / trianglesPerInstance));
                if (cappedVisibleCount < visibleInstanceIndices.size()) {
                    struct DistanceEntry {
                        float distanceSq;
                        uint32_t instanceIndex;
                    };
                    std::vector<DistanceEntry> nearestInstances;
                    nearestInstances.reserve(visibleInstanceIndices.size());
                    for (uint32_t instanceIndex : visibleInstanceIndices) {
                        if (instanceIndex >= m_rasterInstances.size()) continue;
                        const auto& ri = m_rasterInstances[instanceIndex];
                        Vec3 center = ri.worldBBox.is_valid()
                            ? (ri.worldBBox.min + ri.worldBBox.max) * 0.5f
                            : ri.transform.transform_point(Vec3(0.0f, 0.0f, 0.0f));
                        const Vec3 delta = center - m_rasterCullCameraPosition;
                        nearestInstances.push_back({ Vec3::dot(delta, delta), instanceIndex });
                    }

                    auto distanceCmp = [](const DistanceEntry& a, const DistanceEntry& b) {
                        if (a.distanceSq != b.distanceSq) return a.distanceSq < b.distanceSq;
                        return a.instanceIndex < b.instanceIndex;
                    };
                    std::nth_element(nearestInstances.begin(),
                                     nearestInstances.begin() + cappedVisibleCount,
                                     nearestInstances.end(),
                                     distanceCmp);
                    std::sort(nearestInstances.begin(), nearestInstances.begin() + cappedVisibleCount, distanceCmp);

                    proxyInstanceIndices.reserve(nearestInstances.size() - cappedVisibleCount);
                    for (size_t i = cappedVisibleCount; i < nearestInstances.size(); ++i) {
                        proxyInstanceIndices.push_back(nearestInstances[i].instanceIndex);
                    }

                    visibleInstanceIndices.clear();
                    visibleInstanceIndices.reserve(cappedVisibleCount);
                    for (size_t i = 0; i < cappedVisibleCount; ++i) {
                        visibleInstanceIndices.push_back(nearestInstances[i].instanceIndex);
                    }
                }
            }
        }
    }

    if (!mesh.proxyMeshKey.empty()) {
        auto proxyIt = m_rasterMeshes.find(mesh.proxyMeshKey);
        if (proxyIt != m_rasterMeshes.end()) {
            proxyIt->second.visibleInstancesDirty = mesh.visibleInstancesDirty;
            setRasterVisibleInstances(proxyIt->second, proxyInstanceIndices);
        }
    }

    setRasterVisibleInstances(mesh, visibleInstanceIndices);
}

void VulkanBackendAdapter::uploadRasterInstanceBuffer(RasterMeshBuffer& mesh) {
    if (!m_device) return;

    mesh.visibleInstancesDirty = true;
    mesh.lastVisibleFrustumRevision = 0;
    if (kRasterFrustumCullingEnabled) {
        rebuildRasterMeshCullingChunks(mesh);
    } else {
        mesh.cullingChunks.clear();
    }

    struct RasterInstanceGPU {
        float model[16];
    };

    if (mesh.instanceBuffer.buffer) {
        m_device->destroyBuffer(mesh.instanceBuffer);
        mesh.instanceBuffer = VulkanRT::BufferHandle{};
    }

    bool allInstancesVisible = true;
    for (uint32_t instanceIndex : mesh.instanceIndices) {
        if (instanceIndex >= m_rasterInstances.size() || m_rasterInstances[instanceIndex].mask == 0) {
            allInstancesVisible = false;
            break;
        }
    }

    std::vector<uint32_t> visibleInstanceIndices;
    if (allInstancesVisible) {
        visibleInstanceIndices = mesh.instanceIndices;
    } else {
        visibleInstanceIndices.reserve(mesh.instanceIndices.size());
        for (uint32_t instanceIndex : mesh.instanceIndices) {
            if (instanceIndex >= m_rasterInstances.size()) continue;
            if (m_rasterInstances[instanceIndex].mask == 0) continue;
            visibleInstanceIndices.push_back(instanceIndex);
        }
    }

    mesh.instanceCount = static_cast<uint32_t>(visibleInstanceIndices.size());
    if (mesh.instanceCount == 0) {
        mesh.visibleInstanceIndicesCache.clear();
        mesh.visibleInstancesDirty = false;
        return;
    }

    auto matrixToGL = [](const Matrix4x4& mat, float out[16]) {
        Matrix4x4 t = mat.transpose();
        int k = 0;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[k++] = t.m[r][c];
            }
        }
    };

    std::vector<RasterInstanceGPU> gpuInstances(mesh.instanceCount);
    const size_t kParallelMatrixThreshold = 4096;
    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    auto fillGpuRange = [this, &visibleInstanceIndices, &gpuInstances, &matrixToGL](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const uint32_t instanceIndex = visibleInstanceIndices[i];
            if (instanceIndex >= m_rasterInstances.size()) continue;
            matrixToGL(m_rasterInstances[instanceIndex].transform, gpuInstances[i].model);
        }
    };
    if (visibleInstanceIndices.size() < kParallelMatrixThreshold || numThreads < 2) {
        fillGpuRange(0, visibleInstanceIndices.size());
    } else {
        const size_t chunk = (visibleInstanceIndices.size() + numThreads - 1) / numThreads;
        std::vector<std::future<void>> futures;
        futures.reserve(numThreads);
        for (unsigned t = 0; t < numThreads; ++t) {
            const size_t s = t * chunk;
            const size_t e = std::min(s + chunk, visibleInstanceIndices.size());
            if (s >= e) break;
            futures.push_back(std::async(std::launch::async, fillGpuRange, s, e));
        }
        for (auto& f : futures) f.get();
    }

    VulkanRT::BufferCreateInfo ici{};
    ici.size = gpuInstances.size() * sizeof(RasterInstanceGPU);
    ici.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
    ici.location = VulkanRT::MemoryLocation::GPU_ONLY;
    ici.initialData = nullptr;
    mesh.instanceBuffer = m_device->createBuffer(ici);
    if (mesh.instanceBuffer.buffer) {
        m_device->uploadBuffer(mesh.instanceBuffer,
                               gpuInstances.data(),
                               gpuInstances.size() * sizeof(RasterInstanceGPU),
                               0);
    }

    mesh.visibleInstanceIndicesCache = std::move(visibleInstanceIndices);
    mesh.visibleInstancesDirty = false;
    mesh.lastVisibleFrustumRevision = m_rasterFrustumRevision;
    mesh.lastScatterTriangleBudget = m_rasterScatterTriangleBudget;
}

void VulkanBackendAdapter::buildRasterGeometryImpl(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!m_device || !m_device->isInitialized()) return;

    // Skip rebuild if raster cache is still valid for the current scene generation.
    {
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        const uint64_t curGen = g_scene_geometry_generation.load(std::memory_order_acquire);
        if (!m_rasterMeshes.empty() && m_rasterBuiltGeometryGeneration == curGen) {
            m_rasterGeometryDirty = false;
            return;
        }
    }

    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    // Destroy old raster buffers
    destroyAllRasterMeshes();

    auto hasInstancePrefix = [](const std::string& nodeName) -> bool {
        return nodeName.rfind("_inst_gid", 0) == 0;
    };

    size_t baseObjectCount = objects.size();
    while (baseObjectCount > 0) {
        const auto& obj = objects[baseObjectCount - 1];
        auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
        if (!inst || !hasInstancePrefix(inst->node_name)) {
            break;
        }
        --baseObjectCount;
    }

    auto ensureRasterMeshForTriangles = [&](const std::string& meshKey,
                                            const std::vector<std::shared_ptr<Triangle>>& triangles) {
        if (triangles.empty()) return;
        if (m_rasterMeshes.find(meshKey) != m_rasterMeshes.end()) return;

        // Filter nulls into a compact raw-pointer list so the parallel extraction can
        // index output slots directly (9 floats pos + 9 normals + 6 uvs + 3 matIds per
        // triangle). Mirrors VulkanViewportBackend::ensureRasterMeshForTriangles.
        std::vector<const Triangle*> valid;
        valid.reserve(triangles.size());
        for (const auto& t : triangles) {
            if (t) valid.push_back(t.get());
        }
        if (valid.empty()) return;

        const size_t validCount = valid.size();
        std::vector<float> positions(validCount * 9);
        std::vector<float> normals(validCount * 9);
        std::vector<float> uvs(validCount * 6);
        std::vector<uint32_t> matIds(validCount * 3);

        struct LocalBBox { Vec3 bMin; Vec3 bMax; };

        auto extractRange = [&valid, &positions, &normals, &uvs, &matIds]
                            (size_t start, size_t end) -> LocalBBox {
            Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);
            for (size_t i = start; i < end; ++i) {
                const Triangle* t = valid[i];
                auto [uv0, uv1, uv2] = t->getUVCoordinates();
                const uint32_t mid = static_cast<uint32_t>(t->getMaterialID());
                const size_t posBase = i * 9;
                const size_t uvBase  = i * 6;
                const size_t matBase = i * 3;
                for (int v = 0; v < 3; ++v) {
                    Vec3 p = t->getOriginalVertexPosition(v);
                    Vec3 n = t->getOriginalVertexNormal(v);
                    positions[posBase + v * 3 + 0] = p.x;
                    positions[posBase + v * 3 + 1] = p.y;
                    positions[posBase + v * 3 + 2] = p.z;
                    normals[posBase + v * 3 + 0] = n.x;
                    normals[posBase + v * 3 + 1] = n.y;
                    normals[posBase + v * 3 + 2] = n.z;
                    matIds[matBase + v] = mid;
                    bMin.x = std::min(bMin.x, p.x); bMin.y = std::min(bMin.y, p.y); bMin.z = std::min(bMin.z, p.z);
                    bMax.x = std::max(bMax.x, p.x); bMax.y = std::max(bMax.y, p.y); bMax.z = std::max(bMax.z, p.z);
                }
                uvs[uvBase + 0] = uv0.x; uvs[uvBase + 1] = uv0.y;
                uvs[uvBase + 2] = uv1.x; uvs[uvBase + 3] = uv1.y;
                uvs[uvBase + 4] = uv2.x; uvs[uvBase + 5] = uv2.y;
            }
            return { bMin, bMax };
        };

        Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);
        constexpr size_t kExtractParallelThreshold = 4096;
        unsigned extract_threads = std::thread::hardware_concurrency();
        if (extract_threads == 0) extract_threads = 4;

        if (validCount < kExtractParallelThreshold || extract_threads < 2) {
            LocalBBox lbb = extractRange(0, validCount);
            bMin = lbb.bMin;
            bMax = lbb.bMax;
        } else {
            const size_t chunk = (validCount + extract_threads - 1) / extract_threads;
            std::vector<std::future<LocalBBox>> futures;
            futures.reserve(extract_threads);
            for (unsigned t = 0; t < extract_threads; ++t) {
                const size_t s = t * chunk;
                const size_t e = std::min(s + chunk, validCount);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, extractRange, s, e));
            }
            for (auto& f : futures) {
                LocalBBox lbb = f.get();
                bMin.x = std::min(bMin.x, lbb.bMin.x); bMin.y = std::min(bMin.y, lbb.bMin.y); bMin.z = std::min(bMin.z, lbb.bMin.z);
                bMax.x = std::max(bMax.x, lbb.bMax.x); bMax.y = std::max(bMax.y, lbb.bMax.y); bMax.z = std::max(bMax.z, lbb.bMax.z);
            }
        }

        // Cache the local bounding box for this mesh key
        m_rasterMeshBBoxes[meshKey] = AABB(bMin, bMax);

        RasterMeshBuffer rmb;
        rmb.vertexCount = (uint32_t)(positions.size() / 3);
        VulkanRT::BufferCreateInfo vci{};
        vci.size = positions.size() * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        vci.initialData = nullptr;
        rmb.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = normals.size() * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        nci.initialData = nullptr;
        rmb.normalBuffer = m_device->createBuffer(nci);

        // UV buffer for MaterialPreview
        if (!uvs.empty()) {
            VulkanRT::BufferCreateInfo uci{};
            uci.size = uvs.size() * sizeof(float);
            uci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            uci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            rmb.uvBuffer = m_device->createBuffer(uci);
            if (rmb.uvBuffer.buffer) {
                m_device->uploadBuffer(rmb.uvBuffer, uvs.data(), uvs.size() * sizeof(float), 0);
            }
        }

        // MaterialID buffer for MaterialPreview
        if (!matIds.empty()) {
            VulkanRT::BufferCreateInfo mci{};
            mci.size = matIds.size() * sizeof(uint32_t);
            mci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            mci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            rmb.matIdBuffer = m_device->createBuffer(mci);
            if (rmb.matIdBuffer.buffer) {
                m_device->uploadBuffer(rmb.matIdBuffer, matIds.data(), matIds.size() * sizeof(uint32_t), 0);
            }
            rmb.cpuMatIds = std::move(matIds);
        }

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, positions.data(), positions.size() * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, normals.data(), normals.size() * sizeof(float), 0);
        }

        m_rasterMeshes[meshKey] = std::move(rmb);
    };

    struct RasterTriGroup {
        std::string meshKey;
        std::string nodeName;
        std::vector<float> positions; // interleaved x,y,z
        std::vector<float> normals;
        std::vector<float> uvs;       // interleaved u,v per vertex
        std::vector<uint32_t> matIds; // per-vertex material ID
        Matrix4x4 transform;
        uint8_t mask = 0xFF;
    };

    std::vector<RasterTriGroup> groups;
    std::unordered_map<std::string, size_t> groupByKey;

    // Recursive traversal — same logic as updateGeometry but only collects vertex data
    std::function<void(const std::shared_ptr<Hittable>&)> processObj;
    processObj = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;

        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->visible || !inst->source_triangles || inst->source_triangles->empty()) return;
            if (hasInstancePrefix(inst->node_name)) return;

            const auto srcPtr = reinterpret_cast<uintptr_t>(inst->source_triangles.get());
            const std::string instanceNodeName = inst->node_name.empty()
                ? ("[RasterInst-" + std::to_string(m_rasterInstances.size()) + "]")
                : inst->node_name;
            std::unordered_map<std::string, std::vector<std::shared_ptr<Triangle>>> trianglesByNode;
            trianglesByNode.reserve(inst->source_triangles->size());
            for (const auto& tri : *inst->source_triangles) {
                if (!tri) continue;
                const std::string triNodeName = tri->getNodeName().empty() ? instanceNodeName : tri->getNodeName();
                trianglesByNode[triNodeName].push_back(tri);
            }

            for (const auto& [triNodeName, groupedTriangles] : trianglesByNode) {
                if (groupedTriangles.empty()) continue;
                // Keep raster meshes instance-local and node-local so startup preview uses
                // the same object/material grouping as the editor caches and selection code.
                std::string meshKey = "[Raster]-" + triNodeName +
                                      "-src-" + std::to_string(srcPtr) +
                                      "-tris-" + std::to_string(groupedTriangles.size());

                ensureRasterMeshForTriangles(meshKey, groupedTriangles);

                RasterInstance ri;
                ri.meshKey = meshKey;
                ri.nodeName = triNodeName;
                ri.transform = inst->transform;
                ri.mask = 0xFF;
                m_rasterInstances.push_back(ri);
            }

        } else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) processObj(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            processObj(bvh->left);
            processObj(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->visible) return;

            Transform* triTransformHandle = tri->getTransformPtr();
            const bool hasSharedTransform = (triTransformHandle != nullptr);
            std::string nodeName = tri->getNodeName();
            if (nodeName.empty()) nodeName = "[Solo-" + std::to_string(groups.size()) + "]";
            const uintptr_t transformKey = triTransformHandle
                ? reinterpret_cast<uintptr_t>(triTransformHandle)
                : reinterpret_cast<uintptr_t>(tri.get());
            const std::string groupKey = nodeName + "#th=" + std::to_string(transformKey);

            auto found = groupByKey.find(groupKey);
            if (found == groupByKey.end()) {
                RasterTriGroup g;
                g.meshKey = "[Raster-Solo]-" + nodeName;
                g.nodeName = nodeName;
                g.transform = hasSharedTransform ? tri->getTransformMatrix() : Matrix4x4::identity();
                groups.push_back(std::move(g));
                found = groupByKey.emplace(groupKey, groups.size() - 1).first;
            }

            auto& grp = groups[found->second];
            auto [uv0, uv1, uv2] = tri->getUVCoordinates();
            uint32_t mid = static_cast<uint32_t>(tri->getMaterialID());
            for (int v = 0; v < 3; ++v) {
                Vec3 p = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                Vec3 n = hasSharedTransform ? tri->getOriginalVertexNormal(v) : tri->getOriginalVertexNormal(v);
                grp.positions.push_back(p.x); grp.positions.push_back(p.y); grp.positions.push_back(p.z);
                grp.normals.push_back(n.x); grp.normals.push_back(n.y); grp.normals.push_back(n.z);
                grp.matIds.push_back(mid);
            }
            grp.uvs.push_back(uv0.x); grp.uvs.push_back(uv0.y);
            grp.uvs.push_back(uv1.x); grp.uvs.push_back(uv1.y);
            grp.uvs.push_back(uv2.x); grp.uvs.push_back(uv2.y);
        }
        // VDB/Gas volumes are not rasterized in solid mode
    };

    for (size_t i = 0; i < baseObjectCount; ++i) {
        processObj(objects[i]);
    }

    // Upload solo triangle groups
    for (auto& grp : groups) {
        if (grp.positions.empty()) continue;

        // Compute local-space AABB from collected positions
        Vec3 bMin(1e18f, 1e18f, 1e18f), bMax(-1e18f, -1e18f, -1e18f);
        for (size_t pi = 0; pi + 2 < grp.positions.size(); pi += 3) {
            float px = grp.positions[pi], py = grp.positions[pi+1], pz = grp.positions[pi+2];
            bMin.x = std::min(bMin.x, px); bMin.y = std::min(bMin.y, py); bMin.z = std::min(bMin.z, pz);
            bMax.x = std::max(bMax.x, px); bMax.y = std::max(bMax.y, py); bMax.z = std::max(bMax.z, pz);
        }
        m_rasterMeshBBoxes[grp.meshKey] = AABB(bMin, bMax);

        RasterMeshBuffer rmb;
        rmb.vertexCount = (uint32_t)(grp.positions.size() / 3);

        VulkanRT::BufferCreateInfo vci{};
        vci.size = grp.positions.size() * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        vci.initialData = nullptr;
        rmb.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = grp.normals.size() * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        nci.initialData = nullptr;
        rmb.normalBuffer = m_device->createBuffer(nci);

        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, grp.positions.data(), grp.positions.size() * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, grp.normals.data(), grp.normals.size() * sizeof(float), 0);
        }

        // UV buffer for MaterialPreview
        if (!grp.uvs.empty()) {
            VulkanRT::BufferCreateInfo uci{};
            uci.size = grp.uvs.size() * sizeof(float);
            uci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            uci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            rmb.uvBuffer = m_device->createBuffer(uci);
            if (rmb.uvBuffer.buffer) {
                m_device->uploadBuffer(rmb.uvBuffer, grp.uvs.data(), grp.uvs.size() * sizeof(float), 0);
            }
        }
        // MaterialID buffer for MaterialPreview
        if (!grp.matIds.empty()) {
            VulkanRT::BufferCreateInfo mci{};
            mci.size = grp.matIds.size() * sizeof(uint32_t);
            mci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            mci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            rmb.matIdBuffer = m_device->createBuffer(mci);
            if (rmb.matIdBuffer.buffer) {
                m_device->uploadBuffer(rmb.matIdBuffer, grp.matIds.data(), grp.matIds.size() * sizeof(uint32_t), 0);
            }
            rmb.cpuMatIds = grp.matIds;
        }

        m_rasterMeshes[grp.meshKey] = rmb;

        RasterInstance ri;
        ri.meshKey = grp.meshKey;
        ri.nodeName = grp.nodeName;
        ri.transform = grp.transform;
        ri.mask = 0xFF;
        m_rasterInstances.push_back(ri);
    }

    // Append foliage/scatter instances directly from grouped instance data so
    // solid viewport does not have to traverse millions of expanded scene objects.
    //
    // Parallelized (mirrors VulkanViewportBackend::buildRasterGeometry):
    //   - Serial pre-pass resolves meshKey per (group, srcIdx) once and calls
    //     ensureRasterMeshForTriangles (mutates m_rasterMeshes / m_rasterMeshBBoxes).
    //   - Parallel per-group inner loop composes inst.toMatrix() + RasterInstance.
    //   - Parallel bbox assignment.
    const auto& instanceGroups = InstanceManager::getInstance().getGroups();

    struct GroupSrcMeta {
        std::vector<std::string> meshKeyBySrc;  // indexed by srcIdx; empty = invalid source
    };
    std::vector<GroupSrcMeta> groupMeta(instanceGroups.size());

    size_t totalValidScatterInstances = 0;
    for (size_t gi = 0; gi < instanceGroups.size(); ++gi) {
        const auto& group = instanceGroups[gi];
        if (group.instances.empty() || group.sources.empty()) continue;
        auto& meta = groupMeta[gi];
        meta.meshKeyBySrc.resize(group.sources.size());

        for (size_t si = 0; si < group.sources.size(); ++si) {
            const auto& source = group.sources[si];
            const auto* triSource = source.centered_triangles_ptr ? source.centered_triangles_ptr.get() : nullptr;
            if ((!triSource || triSource->empty()) && source.triangles.empty()) continue;

            std::string meshKey;
            if (triSource) {
                const auto srcPtr = reinterpret_cast<uintptr_t>(triSource);
                meshKey = "[Raster-Group]-" + std::to_string(group.id) + "-" + std::to_string(si) +
                          "-" + std::to_string(srcPtr) + "-" + std::to_string(triSource->size());
                ensureRasterMeshForTriangles(meshKey, *triSource);
            } else {
                const auto srcPtr = reinterpret_cast<uintptr_t>(&source.triangles);
                meshKey = "[Raster-Group]-" + std::to_string(group.id) + "-" + std::to_string(si) +
                          "-" + std::to_string(srcPtr) + "-" + std::to_string(source.triangles.size());
                ensureRasterMeshForTriangles(meshKey, source.triangles);
            }
            meta.meshKeyBySrc[si] = std::move(meshKey);
        }

        for (const auto& inst : group.instances) {
            int srcIdx = inst.source_index;
            if (srcIdx < 0 || srcIdx >= static_cast<int>(group.sources.size())) srcIdx = 0;
            if (srcIdx < static_cast<int>(meta.meshKeyBySrc.size()) &&
                !meta.meshKeyBySrc[srcIdx].empty()) {
                ++totalValidScatterInstances;
            }
        }
    }

    m_rasterInstances.reserve(m_rasterInstances.size() + totalValidScatterInstances);

    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    const size_t kParallelThreshold = 1024;

    for (size_t gi = 0; gi < instanceGroups.size(); ++gi) {
        const auto& group = instanceGroups[gi];
        if (group.instances.empty() || group.sources.empty()) continue;
        const auto& meshKeyBySrc = groupMeta[gi].meshKeyBySrc;
        if (meshKeyBySrc.empty()) continue;

        const size_t count = group.instances.size();
        std::vector<RasterInstance> localInstances(count);

        auto fillRange = [&group, &meshKeyBySrc, &localInstances](size_t start, size_t end) {
            const std::string nodePrefix = "_inst_gid" + std::to_string(group.id) + "_";
            for (size_t i = start; i < end; ++i) {
                const auto& inst = group.instances[i];
                int srcIdx = inst.source_index;
                if (srcIdx < 0 || srcIdx >= static_cast<int>(group.sources.size())) srcIdx = 0;
                if (srcIdx >= static_cast<int>(meshKeyBySrc.size()) ||
                    meshKeyBySrc[srcIdx].empty()) {
                    continue;
                }
                auto& ri = localInstances[i];
                ri.meshKey = meshKeyBySrc[srcIdx];
                ri.nodeName = nodePrefix + std::to_string(i);
                ri.transform = inst.toMatrix();
                ri.mask = 0xFF;
                ri.scatterGroupId = group.id;
                ri.scatterInstanceIndex = static_cast<uint32_t>(i);
            }
        };

        if (count < kParallelThreshold || num_threads < 2) {
            fillRange(0, count);
        } else {
            const size_t chunk = (count + num_threads - 1) / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);
            for (unsigned t = 0; t < num_threads; ++t) {
                const size_t s = t * chunk;
                const size_t e = std::min(s + chunk, count);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, fillRange, s, e));
            }
            for (auto& f : futures) f.get();
        }

        for (auto& ri : localInstances) {
            if (ri.meshKey.empty()) continue;
            m_rasterInstances.push_back(std::move(ri));
        }
    }

    // Assign localBBox from cached mesh AABB and compute worldBBox per instance.
    // Parallel: m_rasterMeshBBoxes is read-only here; writes target disjoint instances.
    {
        const size_t total = m_rasterInstances.size();
        auto bboxRange = [this](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                auto& ri = m_rasterInstances[i];
                auto bboxIt = m_rasterMeshBBoxes.find(ri.meshKey);
                if (bboxIt != m_rasterMeshBBoxes.end()) {
                    ri.localBBox = bboxIt->second;
                    updateRasterInstanceWorldBBox(ri);
                }
            }
        };

        if (total < kParallelThreshold || num_threads < 2) {
            bboxRange(0, total);
        } else {
            const size_t chunk = (total + num_threads - 1) / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);
            for (unsigned t = 0; t < num_threads; ++t) {
                const size_t s = t * chunk;
                const size_t e = std::min(s + chunk, total);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, bboxRange, s, e));
            }
            for (auto& f : futures) f.get();
        }
    }

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_rasterInstances.size()); ++i) {
        auto meshIt = m_rasterMeshes.find(m_rasterInstances[i].meshKey);
        if (meshIt == m_rasterMeshes.end()) continue;
        meshIt->second.instanceIndices.push_back(i);
    }
    for (auto& [key, mesh] : m_rasterMeshes) {
        uploadRasterInstanceBuffer(mesh);
    }

    m_rasterGeometryDirty = false;
    m_interactiveViewport.dirty = true;
    m_hasPresentedRenderedFrame = false;
    m_lastCameraHash = 0;

    // Stamp current scene generation so we can skip redundant rebuilds later.
    {
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        m_rasterBuiltGeometryGeneration = g_scene_geometry_generation.load(std::memory_order_acquire);
    }

    SCENE_LOG_INFO("[Vulkan] Raster geometry built: " + std::to_string(m_rasterMeshes.size()) +
                   " meshes, " + std::to_string(m_rasterInstances.size()) + " instances. " +
                   "(base objects scanned: " + std::to_string(baseObjectCount) + "/" + std::to_string(objects.size()) + ")");
}

void VulkanBackendAdapter::syncRasterInstanceTransformsImpl(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (m_rasterInstances.empty()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Build nodeName → transform lookup from current scene objects
    std::unordered_map<std::string, Matrix4x4> transformMap;
    transformMap.reserve(objects.size());

    std::function<void(const std::shared_ptr<Hittable>&)> collectTransforms;
    collectTransforms = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->node_name.empty()) {
                transformMap[inst->node_name] = inst->transform;
            }
        } else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) collectTransforms(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            collectTransforms(bvh->left);
            collectTransforms(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            Transform* th = tri->getTransformPtr();
            std::string name = tri->getNodeName();
            if (!name.empty() && th) {
                transformMap[name] = tri->getTransformMatrix();
            }
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            // Flat (direct SoA) mesh: drives its world transform through its own handle. Without this
            // the raster (Solid/Matcap) viewport never refreshed a keyframed/physics-driven flat
            // mesh per frame — it froze during playback, mirroring the RT-path gap.
            if (!tm->nodeName.empty() && tm->transform) {
                transformMap[tm->nodeName] = tm->transform->getFinal();
            }
        }
    };

    auto hasInstancePrefix = [](const std::string& nodeName) -> bool {
        return nodeName.rfind("_inst_gid", 0) == 0;
    };
    size_t baseObjectCount = objects.size();
    while (baseObjectCount > 0) {
        const auto& obj = objects[baseObjectCount - 1];
        auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
        if (!inst || !hasInstancePrefix(inst->node_name)) {
            break;
        }
        --baseObjectCount;
    }

    for (size_t i = 0; i < baseObjectCount; ++i) {
        collectTransforms(objects[i]);
    }

    const auto& instanceGroups = InstanceManager::getInstance().getGroups();
    std::unordered_map<int, const InstanceGroup*> scatterGroupsById;
    scatterGroupsById.reserve(instanceGroups.size());
    for (const auto& group : instanceGroups) {
        if (!group.instances.empty()) {
            scatterGroupsById.emplace(group.id, &group);
        }
    }

    // Apply to raster instances. Scatter instances can bypass nodeName hash
    // lookup and read transforms directly from InstanceManager.
    bool changed = false;
    std::unordered_set<std::string> dirtyMeshKeys;
    const size_t kParallelThreshold = 2048;
    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    auto syncRange = [this, &transformMap, &scatterGroupsById]
                     (size_t start, size_t end) {
        std::unordered_set<std::string> localDirty;
        for (size_t i = start; i < end; ++i) {
            auto& ri = m_rasterInstances[i];
            Matrix4x4 newTransform;
            bool hasTransform = false;

            if (ri.scatterGroupId >= 0 && ri.scatterInstanceIndex != UINT32_MAX) {
                auto groupIt = scatterGroupsById.find(ri.scatterGroupId);
                if (groupIt != scatterGroupsById.end()) {
                    const auto* group = groupIt->second;
                    if (ri.scatterInstanceIndex < group->instances.size()) {
                        newTransform = group->instances[ri.scatterInstanceIndex].toMatrix();
                        hasTransform = true;
                    }
                }
            } else {
                auto it = transformMap.find(ri.nodeName);
                if (it != transformMap.end()) {
                    newTransform = it->second;
                    hasTransform = true;
                }
            }

            if (hasTransform && !(ri.transform == newTransform)) {
                ri.transform = newTransform;
                updateRasterInstanceWorldBBox(ri);
                localDirty.insert(ri.meshKey);
            }
        }
        return localDirty;
    };

    if (m_rasterInstances.size() < kParallelThreshold || numThreads < 2) {
        dirtyMeshKeys = syncRange(0, m_rasterInstances.size());
    } else {
        const size_t chunk = (m_rasterInstances.size() + numThreads - 1) / numThreads;
        std::vector<std::future<std::unordered_set<std::string>>> futures;
        futures.reserve(numThreads);
        for (unsigned t = 0; t < numThreads; ++t) {
            const size_t s = t * chunk;
            const size_t e = std::min(s + chunk, m_rasterInstances.size());
            if (s >= e) break;
            futures.push_back(std::async(std::launch::async, syncRange, s, e));
        }
        for (auto& f : futures) {
            auto localDirty = f.get();
            dirtyMeshKeys.insert(localDirty.begin(), localDirty.end());
        }
    }
    changed = !dirtyMeshKeys.empty();

    if (changed) {
        for (const auto& meshKey : dirtyMeshKeys) {
            auto meshIt = m_rasterMeshes.find(meshKey);
            if (meshIt != m_rasterMeshes.end()) {
                uploadRasterInstanceBuffer(meshIt->second);
            }
        }
        m_interactiveViewport.dirty = true;
    }
}

void VulkanBackendAdapter::syncRasterSkinnedVerticesImpl(
    const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::vector<Matrix4x4>& boneMatrices)
{
    if (m_rasterInstances.empty() || m_rasterMeshes.empty() || boneMatrices.empty()) return;
    if (m_viewportMode != ViewportMode::Solid && m_viewportMode != ViewportMode::Matcap) return;

    // Collect skinned triangles grouped by raster mesh key (same grouping as buildRasterGeometry)
    // Key = transform handle pointer → meshKey
    struct SkinnedGroup {
        std::string meshKey;
        std::vector<std::shared_ptr<Triangle>> triangles;
    };
    std::unordered_map<void*, SkinnedGroup> skinnedGroups;

    // Map raster instance nodeName → meshKey
    std::unordered_map<std::string, std::string> nodeToMeshKey;
    for (const auto& ri : m_rasterInstances) {
        nodeToMeshKey[ri.nodeName] = ri.meshKey;
    }

    // Traverse scene objects to find skinned triangles
    std::function<void(const std::shared_ptr<Hittable>&)> collectSkinned;
    collectSkinned = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;
        if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
            for (auto& child : list->objects) collectSkinned(child);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            collectSkinned(bvh->left);
            collectSkinned(bvh->right);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (!tri->visible || !tri->hasSkinData()) return;
            Transform* th = tri->getTransformPtr();
            void* groupKey = th ? (void*)th : (void*)tri.get();
            auto& grp = skinnedGroups[groupKey];
            if (grp.meshKey.empty()) {
                std::string nodeName = tri->getNodeName();
                auto it = nodeToMeshKey.find(nodeName);
                if (it != nodeToMeshKey.end()) {
                    grp.meshKey = it->second;
                }
            }
            grp.triangles.push_back(tri);
        }
    };
    for (const auto& obj : objects) collectSkinned(obj);

    if (skinnedGroups.empty()) return;

    // For each skinned group, compute skinned positions/normals and upload
    for (auto& [key, grp] : skinnedGroups) {
        if (grp.meshKey.empty() || grp.triangles.empty()) continue;
        auto meshIt = m_rasterMeshes.find(grp.meshKey);
        if (meshIt == m_rasterMeshes.end()) continue;

        auto& rmb = meshIt->second;
        const size_t vertCount = grp.triangles.size() * 3;
        const size_t floatCount = vertCount * 3;
        if (rmb.vertexCount != (uint32_t)vertCount) continue; // topology mismatch

        std::vector<float> newPositions(floatCount);
        std::vector<float> newNormals(floatCount);
        size_t idx = 0;

        for (const auto& tri : grp.triangles) {
            for (int v = 0; v < 3; ++v) {
                // Compute bone-skinned position (local space, no root transform)
                Vec3 p = tri->apply_bone_to_vertex(v, boneMatrices);
                Vec3 n = tri->apply_bone_to_normal(
                    tri->getOriginalVertexNormal(v),
                    tri->getSkinBoneWeights(v),
                    boneMatrices);
                newPositions[idx]   = p.x; newPositions[idx+1] = p.y; newPositions[idx+2] = p.z;
                newNormals[idx]     = n.x; newNormals[idx+1]   = n.y; newNormals[idx+2]   = n.z;
                idx += 3;
            }
        }

        // Dirty-range upload for efficiency
        if (rmb.cpuPositions.size() == floatCount) {
            size_t dirtyMin = floatCount, dirtyMax = 0;
            for (size_t i = 0; i < floatCount; ++i) {
                if (newPositions[i] != rmb.cpuPositions[i] || newNormals[i] != rmb.cpuNormals[i]) {
                    if (i < dirtyMin) dirtyMin = i;
                    if (i > dirtyMax) dirtyMax = i;
                }
            }
            if (dirtyMin <= dirtyMax) {
                dirtyMin = (dirtyMin / 3) * 3;
                dirtyMax = ((dirtyMax / 3) + 1) * 3;
                if (dirtyMax > floatCount) dirtyMax = floatCount;
                const uint64_t byteOff = dirtyMin * sizeof(float);
                const uint64_t byteLen = (dirtyMax - dirtyMin) * sizeof(float);
                m_device->uploadBuffer(rmb.vertexBuffer, &newPositions[dirtyMin], byteLen, byteOff);
                m_device->uploadBuffer(rmb.normalBuffer, &newNormals[dirtyMin],  byteLen, byteOff);
                std::memcpy(&rmb.cpuPositions[dirtyMin], &newPositions[dirtyMin], byteLen);
                std::memcpy(&rmb.cpuNormals[dirtyMin],   &newNormals[dirtyMin],  byteLen);
            }
        } else {
            m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float));
            m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(),  floatCount * sizeof(float));
            rmb.cpuPositions = std::move(newPositions);
            rmb.cpuNormals   = std::move(newNormals);
        }
    }

    m_interactiveViewport.dirty = true;
}

bool VulkanBackendAdapter::updateRasterMeshFromTrianglesImpl(const std::string& nodeName,
                                                             const std::vector<std::shared_ptr<Triangle>>& triangles) {
    if (!m_device || !m_device->isInitialized() || triangles.empty()) return false;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Find matching raster mesh by node name. For instanced meshes the meshKey does
    // not contain the scene node name, so prefer RasterInstance.nodeName matching.
    std::string targetKey;
    for (const auto& ri : m_rasterInstances) {
        if (matchesNodeNameForInstance(ri.nodeName, nodeName) ||
            matchesNodeNameForInstance(nodeName, ri.nodeName) ||
            ri.meshKey.find(nodeName) != std::string::npos) {
            targetKey = ri.meshKey;
            break;
        }
    }
    if (targetKey.empty()) return false;

    auto meshIt = m_rasterMeshes.find(targetKey);
    if (meshIt == m_rasterMeshes.end()) return false;

    auto& rmb = meshIt->second;
    const size_t vertCount = triangles.size() * 3;
    const size_t floatCount = vertCount * 3;

    // Extract new vertex/normal data
    std::vector<float> newPositions, newNormals;
    newPositions.resize(floatCount);
    newNormals.resize(floatCount);

    const size_t numTriangles = triangles.size();
    #pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(static)
    for (int t = 0; t < (int)numTriangles; ++t) {
        const auto& tri = triangles[t];
        if (!tri) continue;

        const size_t local_idx = t * 9;
        const bool hasSharedTransform = (tri->getTransformPtr() != nullptr);
        
        Vec3 verts[3];
        Vec3 norms[3];
        bool resolved = false;

        if (tri->parentMesh && tri->parentMesh->geometry) {
            TriangleMesh* parentMesh = tri->parentMesh.get();
            const Vec3* cachedPositions = parentMesh->geometry->get_attribute_data<Vec3>("P");
            const Vec3* cachedNormals = parentMesh->geometry->get_attribute_data<Vec3>("N");
            const Vec3* cachedOrigPositions = parentMesh->geometry->get_attribute_data<Vec3>("P_orig");
            const Vec3* cachedOrigNormals = parentMesh->geometry->get_attribute_data<Vec3>("N_orig");
            const std::vector<uint32_t, DNA::AlignedAllocator<uint32_t, 32>>* cachedIndices = &parentMesh->geometry->indices;

            if (cachedPositions && cachedIndices && !cachedIndices->empty()) {
                uint32_t faceIdx = tri->faceIndex;
                uint32_t baseIdx = faceIdx * 3;
                if (baseIdx + 2 < cachedIndices->size()) {
                    uint32_t i0 = (*cachedIndices)[baseIdx + 0];
                    uint32_t i1 = (*cachedIndices)[baseIdx + 1];
                    uint32_t i2 = (*cachedIndices)[baseIdx + 2];

                    if (hasSharedTransform) {
                        if (tri->hasSkinData()) {
                            verts[0] = tri->getOriginalVertexPosition(0);
                            verts[1] = tri->getOriginalVertexPosition(1);
                            verts[2] = tri->getOriginalVertexPosition(2);
                        } else {
                            verts[0] = cachedOrigPositions ? cachedOrigPositions[i0] : cachedPositions[i0];
                            verts[1] = cachedOrigPositions ? cachedOrigPositions[i1] : cachedPositions[i1];
                            verts[2] = cachedOrigPositions ? cachedOrigPositions[i2] : cachedPositions[i2];
                        }
                        norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                        norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                        norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                    } else {
                        verts[0] = cachedPositions[i0];
                        verts[1] = cachedPositions[i1];
                        verts[2] = cachedPositions[i2];
                        norms[0] = cachedOrigNormals ? cachedOrigNormals[i0] : (cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0));
                        norms[1] = cachedOrigNormals ? cachedOrigNormals[i1] : (cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0));
                        norms[2] = cachedOrigNormals ? cachedOrigNormals[i2] : (cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0));
                    }
                    resolved = true;
                }
            }
        }

        if (!resolved) {
            for (int v = 0; v < 3; ++v) {
                verts[v] = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
                norms[v] = tri->getOriginalVertexNormal(v);
            }
        }

        for (int v = 0; v < 3; ++v) {
            newPositions[local_idx + v * 3 + 0] = verts[v].x;
            newPositions[local_idx + v * 3 + 1] = verts[v].y;
            newPositions[local_idx + v * 3 + 2] = verts[v].z;
            newNormals[local_idx + v * 3 + 0]   = norms[v].x;
            newNormals[local_idx + v * 3 + 1]   = norms[v].y;
            newNormals[local_idx + v * 3 + 2]   = norms[v].z;
        }
    }

    const uint32_t newVertCount = (uint32_t)vertCount;

    // Topology changed — full recreate
    if (newVertCount != rmb.vertexCount) {
        const std::vector<uint32_t> preservedInstanceIndices = rmb.instanceIndices;
        m_device->waitIdle();
        destroyRasterMesh(rmb);
        rmb.instanceIndices = preservedInstanceIndices;

        rmb.vertexCount = newVertCount;
        VulkanRT::BufferCreateInfo vci{};
        vci.size = floatCount * sizeof(float);
        vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        vci.initialData = nullptr;
        rmb.vertexBuffer = m_device->createBuffer(vci);

        VulkanRT::BufferCreateInfo nci{};
        nci.size = floatCount * sizeof(float);
        nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        nci.initialData = nullptr;
        rmb.normalBuffer = m_device->createBuffer(nci);

        // Upload data into device-local buffers via staging path
        if (rmb.vertexBuffer.buffer) {
            m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float), 0);
        }
        if (rmb.normalBuffer.buffer) {
            m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(), floatCount * sizeof(float), 0);
        }

        rmb.cpuPositions = std::move(newPositions);
        rmb.cpuNormals = std::move(newNormals);
        if (!rmb.instanceIndices.empty()) {
            uploadRasterInstanceBuffer(rmb);
        }
    } else if (rmb.cpuPositions.size() == floatCount) {
        // Same topology — find dirty range and upload only that region
        size_t dirtyMin = floatCount;
        size_t dirtyMax = 0;

        for (size_t i = 0; i < floatCount; ++i) {
            if (newPositions[i] != rmb.cpuPositions[i] || newNormals[i] != rmb.cpuNormals[i]) {
                if (i < dirtyMin) dirtyMin = i;
                if (i > dirtyMax) dirtyMax = i;
            }
        }

        if (dirtyMin <= dirtyMax) {
            // Align to vec3 boundaries (12 bytes = 3 floats)
            dirtyMin = (dirtyMin / 3) * 3;
            dirtyMax = ((dirtyMax / 3) + 1) * 3;
            if (dirtyMax > floatCount) dirtyMax = floatCount;

            const uint64_t byteOffset = dirtyMin * sizeof(float);
            const uint64_t byteSize   = (dirtyMax - dirtyMin) * sizeof(float);

            m_device->uploadBuffer(rmb.vertexBuffer, &newPositions[dirtyMin], byteSize, byteOffset);
            m_device->uploadBuffer(rmb.normalBuffer, &newNormals[dirtyMin],  byteSize, byteOffset);

            // Update CPU shadow
            std::memcpy(&rmb.cpuPositions[dirtyMin], &newPositions[dirtyMin], byteSize);
            std::memcpy(&rmb.cpuNormals[dirtyMin],   &newNormals[dirtyMin],  byteSize);
        }
    } else {
        // No CPU shadow yet — full upload and store shadow
        m_device->uploadBuffer(rmb.vertexBuffer, newPositions.data(), floatCount * sizeof(float));
        m_device->uploadBuffer(rmb.normalBuffer, newNormals.data(),  floatCount * sizeof(float));
        rmb.cpuPositions = std::move(newPositions);
        rmb.cpuNormals = std::move(newNormals);
    }

    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanBackendAdapter::patchRasterMeshTrianglesImpl(
    const std::string& nodeName,
    const std::vector<size_t>& dirtyIndices,
    const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries) {
    if (!m_device || !m_device->isInitialized() || dirtyIndices.empty() || meshEntries.empty())
        return false;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // Find matching raster mesh by node name. For instanced meshes the meshKey does
    // not contain the scene node name, so prefer RasterInstance.nodeName matching.
    std::string targetKey;
    for (const auto& ri : m_rasterInstances) {
        if (matchesNodeNameForInstance(ri.nodeName, nodeName) ||
            matchesNodeNameForInstance(nodeName, ri.nodeName) ||
            ri.meshKey.find(nodeName) != std::string::npos) {
            targetKey = ri.meshKey;
            break;
        }
    }
    if (targetKey.empty()) return false;

    auto meshIt = m_rasterMeshes.find(targetKey);
    if (meshIt == m_rasterMeshes.end()) return false;

    auto& rmb = meshIt->second;
    const size_t expectedVertCount = meshEntries.size() * 3;
    const size_t expectedFloatCount = expectedVertCount * 3;

    // Topology must match; if not, fall back to full update
    if (rmb.vertexCount != static_cast<uint32_t>(expectedVertCount) ||
        rmb.cpuPositions.size() != expectedFloatCount) {
        return false;
    }

    size_t dirtyMinFloat = expectedFloatCount;
    size_t dirtyMaxFloat = 0;

    for (const size_t triIdx : dirtyIndices) {
        if (triIdx >= meshEntries.size()) continue;
        const auto& tri = meshEntries[triIdx].second;
        if (!tri) continue;

        const size_t baseFloat = triIdx * 9; // 3 vertices * 3 floats
        if (baseFloat + 8 >= expectedFloatCount) continue;

        const bool hasSharedTransform = (tri->getTransformPtr() != nullptr);
        for (int v = 0; v < 3; ++v) {
            Vec3 p = hasSharedTransform ? tri->getOriginalVertexPosition(v) : tri->getVertexPosition(v);
            Vec3 n = hasSharedTransform ? tri->getOriginalVertexNormal(v) : tri->getOriginalVertexNormal(v);
            const size_t idx = baseFloat + static_cast<size_t>(v) * 3;
            rmb.cpuPositions[idx]     = p.x;
            rmb.cpuPositions[idx + 1] = p.y;
            rmb.cpuPositions[idx + 2] = p.z;
            rmb.cpuNormals[idx]       = n.x;
            rmb.cpuNormals[idx + 1]   = n.y;
            rmb.cpuNormals[idx + 2]   = n.z;
        }

        if (baseFloat < dirtyMinFloat) dirtyMinFloat = baseFloat;
        if (baseFloat + 8 > dirtyMaxFloat) dirtyMaxFloat = baseFloat + 8;
    }

    if (dirtyMinFloat <= dirtyMaxFloat) {
        // Align to vec3 boundaries
        dirtyMinFloat = (dirtyMinFloat / 3) * 3;
        dirtyMaxFloat = ((dirtyMaxFloat / 3) + 1) * 3;
        if (dirtyMaxFloat > expectedFloatCount) dirtyMaxFloat = expectedFloatCount;

        const uint64_t byteOffset = dirtyMinFloat * sizeof(float);
        const uint64_t byteSize   = (dirtyMaxFloat - dirtyMinFloat) * sizeof(float);

        m_device->uploadBuffer(rmb.vertexBuffer, &rmb.cpuPositions[dirtyMinFloat], byteSize, byteOffset);
        m_device->uploadBuffer(rmb.normalBuffer, &rmb.cpuNormals[dirtyMinFloat], byteSize, byteOffset);
    }

    m_interactiveViewport.dirty = true;
    return true;
}

bool VulkanBackendAdapter::cloneRasterObjectByNodeName(
    const std::string& sourceNodeName,
    const std::string& newNodeName,
    const Matrix4x4& transform) {
    if (!m_device || !m_device->isInitialized() || sourceNodeName.empty() || newNodeName.empty()) {
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    std::vector<uint32_t> sourceIndices;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_rasterInstances.size()); ++i) {
        const auto& ri = m_rasterInstances[i];
        if (matchesNodeNameForInstance(ri.nodeName, sourceNodeName) ||
            matchesNodeNameForInstance(sourceNodeName, ri.nodeName)) {
            sourceIndices.push_back(i);
        }
    }
    if (sourceIndices.empty()) {
        return false;
    }

    std::unordered_set<std::string> dirtyMeshKeys;
    for (uint32_t sourceIndex : sourceIndices) {
        if (sourceIndex >= m_rasterInstances.size()) continue;

        RasterInstance clone = m_rasterInstances[sourceIndex];
        clone.nodeName = newNodeName;
        clone.transform = transform;
        clone.scatterGroupId = -1;
        clone.scatterInstanceIndex = UINT32_MAX;
        updateRasterInstanceWorldBBox(clone);

        const uint32_t newIndex = static_cast<uint32_t>(m_rasterInstances.size());
        m_rasterInstances.push_back(std::move(clone));

        auto meshIt = m_rasterMeshes.find(m_rasterInstances.back().meshKey);
        if (meshIt != m_rasterMeshes.end()) {
            meshIt->second.instanceIndices.push_back(newIndex);
            dirtyMeshKeys.insert(meshIt->first);
        }
    }

    for (const auto& meshKey : dirtyMeshKeys) {
        auto meshIt = m_rasterMeshes.find(meshKey);
        if (meshIt != m_rasterMeshes.end()) {
            uploadRasterInstanceBuffer(meshIt->second);
        }
    }

    m_interactiveViewport.dirty = true;
    return !dirtyMeshKeys.empty();
}

bool VulkanBackendAdapter::cloneRtObjectByNodeName(
    const std::string& sourceNodeName,
    const std::string& newNodeName,
    const std::shared_ptr<Hittable>& representativeSource,
    const Matrix4x4& transform) {
    if (!m_device || !m_device->isInitialized() || !m_device->hasHardwareRT() ||
        sourceNodeName.empty() || newNodeName.empty() || !representativeSource) {
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    std::vector<size_t> sourceIndices;
    for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
        if (!m_instanceSources[i]) continue;
        std::string instName;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            instName = inst->node_name;
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            instName = tri->getNodeName();
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            // Flat (SoA) mesh instance source — without this branch flat duplicates miss every
            // source instance and the clone falls back to a full geometry rebuild.
            instName = tm->nodeName;
        }
        if (matchesNodeNameForInstance(instName, sourceNodeName) ||
            matchesNodeNameForInstance(sourceNodeName, instName)) {
            sourceIndices.push_back(i);
        }
    }
    if (sourceIndices.empty()) {
        return false;
    }

    for (size_t sourceIndex : sourceIndices) {
        VulkanRT::TLASInstance clone = m_vkInstances[sourceIndex];
        clone.transform = transform;
        clone.scatterGroupId = -1;
        clone.scatterInstanceIndex = UINT32_MAX;
        m_vkInstances.push_back(clone);
        m_instanceSources.push_back(representativeSource);

        InstanceTransformCache item;
        item.instance_id = static_cast<int>(m_instanceSources.size() - 1);
        item.representative_hittable = representativeSource;
        m_instance_sync_cache.push_back(item);
    }

    auto merged = m_vkInstances;
    for (const auto& h : m_hairVkInstances) merged.push_back(h);
    m_device->updateTLAS(merged);

    std::vector<VulkanRT::VkInstanceData> instData;
    instData.reserve(m_vkInstances.size());
    for (const auto& vi : m_vkInstances) {
        VulkanRT::VkInstanceData d;
        d.materialIndex = vi.materialIndex;
        d.blasIndex = vi.blasIndex;
        instData.push_back(d);
    }
    if (m_device->m_instanceDataBuffer.buffer) {
        m_device->destroyBuffer(m_device->m_instanceDataBuffer);
    }
    ::VulkanRT::BufferCreateInfo ci;
    ci.size = static_cast<uint64_t>(instData.size()) * sizeof(::VulkanRT::VkInstanceData);
    ci.usage = (::VulkanRT::BufferUsage)((uint32_t)::VulkanRT::BufferUsage::STORAGE | (uint32_t)::VulkanRT::BufferUsage::TRANSFER_DST);
    ci.location = ::VulkanRT::MemoryLocation::CPU_TO_GPU;
    ci.initialData = instData.data();
    m_device->m_instanceDataBuffer = m_device->createBuffer(ci);
    if (m_device->m_rtDescriptorSet != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo instInfo{};
        instInfo.buffer = m_device->m_instanceDataBuffer.buffer;
        instInfo.offset = 0;
        instInfo.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w5{};
        w5.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w5.dstSet = m_device->m_rtDescriptorSet;
        w5.dstBinding = 5;
        w5.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w5.descriptorCount = 1;
        w5.pBufferInfo = &instInfo;
        vkUpdateDescriptorSets(m_device->m_device, 1, &w5, 0, nullptr);
    }

    m_topology_dirty = false;
    resetAccumulation();
    return true;
}

void VulkanBackendAdapter::setViewportMode(ViewportMode mode) {
    if (m_viewportMode == mode) return;
    const ViewportMode oldMode = m_viewportMode;
    m_viewportMode = mode;
    m_loggedInteractiveViewportFallback = false;

    // Switching from solid→rendered: BLAS/TLAS may be stale (solid mode defers RT rebuilds).
    // Mark topology dirty so the next render triggers a full rebuild.
    if (oldMode != ViewportMode::Rendered && mode == ViewportMode::Rendered) {
        m_topology_dirty = true;
        // Interactive solid/matcap resources must not leak into the RT rendered path.
        if (m_device && m_device->isInitialized()) {
            m_device->waitIdle();
        }
        destroyInteractiveViewportResourcesImpl(false);
        m_interactiveViewport = {};
    } else if (oldMode == ViewportMode::Rendered && mode != ViewportMode::Rendered) {
        // Re-entering interactive modes should rebuild cleanly from scratch.
        if (m_device && m_device->isInitialized()) {
            m_device->waitIdle();
        }
        destroyInteractiveViewportResourcesImpl(false);
        m_interactiveViewport = {};
        m_interactiveViewport.dirty = true;
    }

    resetAccumulation();

    // Publish to the passive authoritative store for any external reader.
    Core::RenderStateManager::instance().setViewportMode(mode);
}

ViewportMode VulkanBackendAdapter::getViewportMode() const {
    return m_viewportMode;
}

bool VulkanBackendAdapter::supportsViewportMode(ViewportMode mode) const {
    switch (mode) {
        case ViewportMode::Rendered:
            return true;
        case ViewportMode::Solid:
            return true;
        case ViewportMode::MaterialPreview:
            return true;
        case ViewportMode::Matcap:
            // Matcap is supported via the interactive/raster solid pipeline
            return true;
        default:
            return false;
    }
}

bool VulkanBackendAdapter::shouldUseInteractiveViewportImpl() const {
    // Use the interactive viewport only for modes the Vulkan backend supports
    // and that are non-raytraced (Rendered uses the pathtracer).
    return (m_viewportMode != ViewportMode::Rendered) && supportsViewportMode(m_viewportMode);
}

void VulkanBackendAdapter::destroyInteractiveViewportResourcesImpl(bool keepPipeline) {
    if (!m_device) return;
    VkDevice vkDevice = m_device->getDevice();

    if (m_interactiveViewport.framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(vkDevice, m_interactiveViewport.framebuffer, nullptr);
        m_interactiveViewport.framebuffer = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.renderPass != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyRenderPass(vkDevice, m_interactiveViewport.renderPass, nullptr);
        m_interactiveViewport.renderPass = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.solidPipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.solidPipeline, nullptr);
        m_interactiveViewport.solidPipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.pipelineLayout != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.pipelineLayout, nullptr);
        m_interactiveViewport.pipelineLayout = VK_NULL_HANDLE;
    }
    // Material Preview pipeline cleanup
    if (m_interactiveViewport.materialPreviewPipeline != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipeline(vkDevice, m_interactiveViewport.materialPreviewPipeline, nullptr);
        m_interactiveViewport.materialPreviewPipeline = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.materialPreviewPipelineLayout != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.materialPreviewPipelineLayout, nullptr);
        m_interactiveViewport.materialPreviewPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.materialPreviewDescPool != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyDescriptorPool(vkDevice, m_interactiveViewport.materialPreviewDescPool, nullptr);
        m_interactiveViewport.materialPreviewDescPool = VK_NULL_HANDLE;
        m_interactiveViewport.materialPreviewDescSet = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.materialPreviewDescLayout != VK_NULL_HANDLE && !keepPipeline) {
        vkDestroyDescriptorSetLayout(vkDevice, m_interactiveViewport.materialPreviewDescLayout, nullptr);
        m_interactiveViewport.materialPreviewDescLayout = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.colorImage.image) {
        m_device->destroyImage(m_interactiveViewport.colorImage);
    }
    if (m_interactiveViewport.depthImage.image) {
        m_device->destroyImage(m_interactiveViewport.depthImage);
    }
    if (m_interactiveViewport.stagingBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.stagingBuffer);
    }
    if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
        // descriptor sets are freed when pool is destroyed; just reset handle
        m_interactiveViewport.matcapDescSet = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.matcapDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vkDevice, m_interactiveViewport.matcapDescPool, nullptr);
        m_interactiveViewport.matcapDescPool = VK_NULL_HANDLE;
    }
    if (m_interactiveViewport.matcapDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vkDevice, m_interactiveViewport.matcapDescLayout, nullptr);
        m_interactiveViewport.matcapDescLayout = VK_NULL_HANDLE;
    }
    // Note: matcap images uploaded via uploadTexture2D are tracked in m_uploadedImages
    // and should not be destroyed here to avoid double-free. We keep the ImageHandle
    // reference around but do not free the underlying VkImage.
    if (m_interactiveViewport.gridVertexBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.gridVertexBuffer);
        m_interactiveViewport.gridVertexCount = 0;
    }
    if (m_interactiveViewport.gridNormalBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.gridNormalBuffer);
    }
    if (m_interactiveViewport.identityInstanceBuffer.buffer) {
        m_device->destroyBuffer(m_interactiveViewport.identityInstanceBuffer);
    }
    m_interactiveViewport.width = 0;
    m_interactiveViewport.height = 0;
    if (!keepPipeline) {
        m_interactiveViewport.initialized = false;
    }
}

bool VulkanBackendAdapter::ensureInteractiveViewportResourcesImpl(const std::string& shaderDir, int width, int height) {
    if (!m_device || !m_device->isInitialized() || width <= 0 || height <= 0) return false;
    if (!m_device->supportsGraphicsQueue()) return false;

    VkDevice vkDevice = m_device->getDevice();

    if (m_interactiveViewport.solidPipeline == VK_NULL_HANDLE) {
        const std::string vertPath = shaderDir + "/solid.spv";
        const std::string fragPath = shaderDir + "/solid_frag.spv";
        if (!std::filesystem::exists(vertPath) || !std::filesystem::exists(fragPath)) {
            return false;
        }

        std::vector<uint32_t> vertSPV = loadSPV(vertPath);
        std::vector<uint32_t> fragSPV = loadSPV(fragPath);
        if (vertSPV.empty() || fragSPV.empty()) {
            return false;
        }

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

        shaderInfo.codeSize = vertSPV.size() * sizeof(uint32_t);
        shaderInfo.pCode = vertSPV.data();
        VkShaderModule vertModule = VK_NULL_HANDLE;
        if (vkCreateShaderModule(vkDevice, &shaderInfo, nullptr, &vertModule) != VK_SUCCESS) {
            return false;
        }

        shaderInfo.codeSize = fragSPV.size() * sizeof(uint32_t);
        shaderInfo.pCode = fragSPV.data();
        VkShaderModule fragModule = VK_NULL_HANDLE;
        if (vkCreateShaderModule(vkDevice, &shaderInfo, nullptr, &fragModule) != VK_SUCCESS) {
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        VkAttachmentDescription attachments[2]{};
        attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_GENERAL;

        attachments[1].format = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 1;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 2;
        renderPassInfo.pAttachments = attachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
        if (vkCreateRenderPass(vkDevice, &renderPassInfo, nullptr, &m_interactiveViewport.renderPass) != VK_SUCCESS) {
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        // Two mat4 (32 floats -> 128 bytes) plus int + custom params
        pushRange.size = sizeof(float) * 32 + sizeof(int) + sizeof(float) * (3 + 3 + 3 + 2); // custom colors (3*3) + 2 floats
        // Create descriptor set layout for matcap (binding = 0)
        VkDescriptorSetLayoutBinding matcapBinding{};
        matcapBinding.binding = 0;
        matcapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        matcapBinding.descriptorCount = 1;
        matcapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        matcapBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 1;
        dslci.pBindings = &matcapBinding;
        if (vkCreateDescriptorSetLayout(vkDevice, &dslci, nullptr, &m_interactiveViewport.matcapDescLayout) != VK_SUCCESS) {
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        // Descriptor pool for a single combined image sampler
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &poolSize;
        dpci.maxSets = 1;
        if (vkCreateDescriptorPool(vkDevice, &dpci, nullptr, &m_interactiveViewport.matcapDescPool) != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(vkDevice, m_interactiveViewport.matcapDescLayout, nullptr);
            m_interactiveViewport.matcapDescLayout = VK_NULL_HANDLE;
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_interactiveViewport.matcapDescLayout;
        if (vkCreatePipelineLayout(vkDevice, &pipelineLayoutInfo, nullptr, &m_interactiveViewport.pipelineLayout) != VK_SUCCESS) {
            vkDestroyRenderPass(vkDevice, m_interactiveViewport.renderPass, nullptr);
            m_interactiveViewport.renderPass = VK_NULL_HANDLE;
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo shaderStages[2]{};
        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = vertModule;
        shaderStages[0].pName = "main";
        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = fragModule;
        shaderStages[1].pName = "main";

        VkVertexInputBindingDescription bindings[3]{};
        bindings[0].binding = 0;
        bindings[0].stride = sizeof(float) * 3;
        bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindings[1].binding = 1;
        bindings[1].stride = sizeof(float) * 3;
        bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindings[2].binding = 2;
        bindings[2].stride = sizeof(float) * 16;
        bindings[2].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

        VkVertexInputAttributeDescription attributes[6]{};
        attributes[0].location = 0;
        attributes[0].binding = 0;
        attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[0].offset = 0;
        attributes[1].location = 1;
        attributes[1].binding = 1;
        attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[1].offset = 0;
        attributes[2].location = 2;
        attributes[2].binding = 2;
        attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[2].offset = sizeof(float) * 0;
        attributes[3].location = 3;
        attributes[3].binding = 2;
        attributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[3].offset = sizeof(float) * 4;
        attributes[4].location = 4;
        attributes[4].binding = 2;
        attributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[4].offset = sizeof(float) * 8;
        attributes[5].location = 5;
        attributes[5].binding = 2;
        attributes[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributes[5].offset = sizeof(float) * 12;

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 3;
        vertexInput.pVertexBindingDescriptions = bindings;
        vertexInput.vertexAttributeDescriptionCount = 6;
        vertexInput.pVertexAttributeDescriptions = attributes;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        // Alpha blend so the grid distance fade can dissolve lines; mesh draws
        // output alpha 1.0 so their result is unchanged.
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_interactiveViewport.pipelineLayout;
        pipelineInfo.renderPass = m_interactiveViewport.renderPass;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_interactiveViewport.solidPipeline) != VK_SUCCESS) {
            vkDestroyPipelineLayout(vkDevice, m_interactiveViewport.pipelineLayout, nullptr);
            m_interactiveViewport.pipelineLayout = VK_NULL_HANDLE;
            vkDestroyRenderPass(vkDevice, m_interactiveViewport.renderPass, nullptr);
            m_interactiveViewport.renderPass = VK_NULL_HANDLE;
            vkDestroyShaderModule(vkDevice, fragModule, nullptr);
            vkDestroyShaderModule(vkDevice, vertModule, nullptr);
            return false;
        }

        vkDestroyShaderModule(vkDevice, fragModule, nullptr);
        vkDestroyShaderModule(vkDevice, vertModule, nullptr);
        // Allocate and update matcap descriptor set (one combined image sampler)
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = m_interactiveViewport.matcapDescPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &m_interactiveViewport.matcapDescLayout;
        VkDescriptorSet descSet = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(vkDevice, &dsai, &descSet) == VK_SUCCESS) {
            // Always keep the descriptor set handle — texture will be bound later
            // when user loads a matcap via setInteractiveViewportMatcap().
            m_interactiveViewport.matcapDescSet = descSet;

            // Bind a dummy texture so the descriptor set is valid for Vulkan
            // (even though solid/procedural modes don't sample it).
            if (!m_interactiveViewport.matcapImage.image) {
                std::vector<uint8_t> white(4 * 2 * 2, 255);
                int64_t id = this->uploadTexture2D(white.data(), 2, 2, 4, false, false);
                if (id && m_uploadedImages.find(id) != m_uploadedImages.end()) {
                    m_interactiveViewport.matcapImage = m_uploadedImages[id];
                }
            }

            if (m_interactiveViewport.matcapImage.image) {
                VkDescriptorImageInfo di{};
                di.sampler = m_interactiveViewport.matcapImage.sampler;
                di.imageView = m_interactiveViewport.matcapImage.view;
                di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet wds{};
                wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wds.dstSet = descSet;
                wds.dstBinding = 0;
                wds.dstArrayElement = 0;
                wds.descriptorCount = 1;
                wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                wds.pImageInfo = &di;

                vkUpdateDescriptorSets(vkDevice, 1, &wds, 0, nullptr);
            }
        }
        m_interactiveViewport.initialized = true;
    }

    // ── Material Preview Pipeline (created once, reuses solid render pass) ──
    // Capability gate: this pipeline requires descriptor indexing (runtimeDescriptorArray +
    // partially-bound) plus 208 bytes of push-constant space. The shader material_preview_frag.spv
    // uses `sampler2D textures[]` with nonuniformEXT. On GPUs that lack these (e.g. GTX 850M),
    // pipeline creation or first draw hits a null-deref inside the vendor driver.
    // When unsupported, skip the pipeline entirely and fall back to the matcap path, and notify
    // the user via the HUD. Covers the base VulkanBackendAdapter path (g_backend).
    const bool mpHasDescIdx = m_device && m_device->getCapabilities().supportsDescriptorIndexing;
    constexpr uint32_t kMpPushBytes_Base = sizeof(float) * 48 + sizeof(uint32_t) * 4;
    bool mpPushOkGate = false;
    if (m_device && m_device->getPhysicalDevice()) {
        VkPhysicalDeviceProperties _p{};
        vkGetPhysicalDeviceProperties(m_device->getPhysicalDevice(), &_p);
        mpPushOkGate = _p.limits.maxPushConstantsSize >= kMpPushBytes_Base;
    }
    const bool mpCapsOk = mpHasDescIdx && mpPushOkGate;
    if (!mpCapsOk) {
        if (!m_materialPreviewUnsupportedNotified) {
            m_materialPreviewUnsupportedNotified = true;
            SCENE_LOG_WARN(std::string("[Vulkan] Material preview unsupported on this GPU ") +
                "(descriptorIndexing=" + (mpHasDescIdx ? "yes" : "no") +
                ", pushConstants>=208=" + (mpPushOkGate ? "yes" : "no") +
                "). Falling back to matcap.");
            if (m_statusCallback) {
                m_statusCallback("Material preview not supported on this GPU - using matcap fallback.", 1);
            }
        }
    } else if (m_interactiveViewport.materialPreviewPipeline == VK_NULL_HANDLE &&
        m_interactiveViewport.renderPass != VK_NULL_HANDLE) {
        const std::string mpVertPath = shaderDir + "/material_preview.spv";
        const std::string mpFragPath = shaderDir + "/material_preview_frag.spv";
        if (std::filesystem::exists(mpVertPath) && std::filesystem::exists(mpFragPath)) {
            std::vector<uint32_t> mpVertSPV = loadSPV(mpVertPath);
            std::vector<uint32_t> mpFragSPV = loadSPV(mpFragPath);
            if (!mpVertSPV.empty() && !mpFragSPV.empty()) {
                VkShaderModuleCreateInfo smci{};
                smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

                smci.codeSize = mpVertSPV.size() * sizeof(uint32_t);
                smci.pCode = mpVertSPV.data();
                VkShaderModule mpVertModule = VK_NULL_HANDLE;
                vkCreateShaderModule(vkDevice, &smci, nullptr, &mpVertModule);

                smci.codeSize = mpFragSPV.size() * sizeof(uint32_t);
                smci.pCode = mpFragSPV.data();
                VkShaderModule mpFragModule = VK_NULL_HANDLE;
                vkCreateShaderModule(vkDevice, &smci, nullptr, &mpFragModule);

                if (mpVertModule && mpFragModule) {
                    VkPhysicalDeviceProperties mpDevProps{};
                    vkGetPhysicalDeviceProperties(m_device->getPhysicalDevice(), &mpDevProps);
                    const uint32_t mpTextureArrayLen = (mpDevProps.limits.maxPerStageDescriptorSampledImages > 2u)
                        ? (std::min)(static_cast<uint32_t>(Backend::VULKAN_TEXTURE_CAPACITY),
                                     mpDevProps.limits.maxPerStageDescriptorSampledImages - 2u)
                        : 1u;
                    m_interactiveViewport.materialPreviewTextureArrayLen = mpTextureArrayLen;

                    VkDescriptorSetLayoutBinding mpDslBindings[4]{};
                    mpDslBindings[0].binding = 0;
                    mpDslBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    mpDslBindings[0].descriptorCount = 1;
                    mpDslBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpDslBindings[1].binding = 1;
                    mpDslBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    mpDslBindings[1].descriptorCount = mpTextureArrayLen;
                    mpDslBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpDslBindings[2].binding = 2;
                    mpDslBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    mpDslBindings[2].descriptorCount = 2;
                    mpDslBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpDslBindings[3].binding = 3;
                    mpDslBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    mpDslBindings[3].descriptorCount = 1;
                    mpDslBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                    // Guard: check device push-constant limit before touching descriptor/pipeline layout.
                    // Some older drivers crash inside vkCreateDescriptorSetLayout or vkCreatePipelineLayout
                    // when push constants exceed maxPushConstantsSize, rather than returning an error.
                    constexpr uint32_t kMpPushBytes = sizeof(float) * 48 + sizeof(uint32_t) * 4; // 208 bytes
                    const bool mpPushOk = mpDevProps.limits.maxPushConstantsSize >= kMpPushBytes;
                    if (!mpPushOk) {
                        SCENE_LOG_WARN("[Vulkan] Material preview pipeline skipped: "
                            "maxPushConstantsSize=" +
                            std::to_string(mpDevProps.limits.maxPushConstantsSize) +
                            " < required " + std::to_string(kMpPushBytes) + " bytes.");
                    }

                    VkDescriptorSetLayoutCreateInfo mpDslci{};
                    mpDslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                    mpDslci.bindingCount = 4;
                    mpDslci.pBindings = mpDslBindings;
                    VkDescriptorBindingFlags mpBindingFlags[4] = {
                        0,
                        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
                        0,
                        0
                    };
                    VkDescriptorSetLayoutBindingFlagsCreateInfo mpBindingFlagsCI{};
                    mpBindingFlagsCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
                    mpBindingFlagsCI.bindingCount = 4;
                    mpBindingFlagsCI.pBindingFlags = mpBindingFlags;
                    mpDslci.pNext = &mpBindingFlagsCI;
                    if (mpPushOk) {
                        vkCreateDescriptorSetLayout(vkDevice, &mpDslci, nullptr,
                                                    &m_interactiveViewport.materialPreviewDescLayout);
                    }

                    // Descriptor pool
                    VkDescriptorPoolSize mpPoolSizes[2]{};
                    mpPoolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    mpPoolSizes[0].descriptorCount = 2;
                    mpPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    mpPoolSizes[1].descriptorCount = mpTextureArrayLen + 2u;
                    VkDescriptorPoolCreateInfo mpDpci{};
                    mpDpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                    mpDpci.poolSizeCount = 2;
                    mpDpci.pPoolSizes = mpPoolSizes;
                    mpDpci.maxSets = 1;
                    vkCreateDescriptorPool(vkDevice, &mpDpci, nullptr,
                                           &m_interactiveViewport.materialPreviewDescPool);

                    // Push constant range: 2x mat4 + 4x vec4 + uvec4 = 208 bytes
                    VkPushConstantRange mpPushRange{};
                    mpPushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpPushRange.offset = 0;
                    mpPushRange.size = kMpPushBytes;

                    VkPipelineLayoutCreateInfo mpPlci{};
                    mpPlci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                    mpPlci.pushConstantRangeCount = 1;
                    mpPlci.pPushConstantRanges = &mpPushRange;
                    mpPlci.setLayoutCount = 1;
                    mpPlci.pSetLayouts = &m_interactiveViewport.materialPreviewDescLayout;
                    if (m_interactiveViewport.materialPreviewDescLayout != VK_NULL_HANDLE) {
                        vkCreatePipelineLayout(vkDevice, &mpPlci, nullptr,
                                               &m_interactiveViewport.materialPreviewPipelineLayout);
                    }

                    // Shader stages
                    VkPipelineShaderStageCreateInfo mpStages[2]{};
                    mpStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    mpStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
                    mpStages[0].module = mpVertModule;
                    mpStages[0].pName = "main";
                    mpStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    mpStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
                    mpStages[1].module = mpFragModule;
                    mpStages[1].pName = "main";

                    // Vertex input: binding 0=pos, 1=normal, 2=matId, 3=instance matrix
                    VkVertexInputBindingDescription mpBindings[4]{};
                    mpBindings[0].binding = 0; mpBindings[0].stride = sizeof(float) * 3; mpBindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                    mpBindings[1].binding = 1; mpBindings[1].stride = sizeof(float) * 3; mpBindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                    mpBindings[2].binding = 2; mpBindings[2].stride = sizeof(uint32_t);  mpBindings[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                    mpBindings[3].binding = 3; mpBindings[3].stride = sizeof(float) * 16; mpBindings[3].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

                    VkVertexInputAttributeDescription mpAttribs[7]{};
                    // location 0: position
                    mpAttribs[0].location = 0; mpAttribs[0].binding = 0; mpAttribs[0].format = VK_FORMAT_R32G32B32_SFLOAT; mpAttribs[0].offset = 0;
                    // location 1: normal
                    mpAttribs[1].location = 1; mpAttribs[1].binding = 1; mpAttribs[1].format = VK_FORMAT_R32G32B32_SFLOAT; mpAttribs[1].offset = 0;
                    // location 3: materialID
                    mpAttribs[2].location = 2; mpAttribs[2].binding = 2; mpAttribs[2].format = VK_FORMAT_R32_UINT; mpAttribs[2].offset = 0;
                    // location 4-7: instance model matrix (4 columns)
                    mpAttribs[3].location = 3; mpAttribs[3].binding = 3; mpAttribs[3].format = VK_FORMAT_R32G32B32A32_SFLOAT; mpAttribs[3].offset = sizeof(float) * 0;
                    mpAttribs[4].location = 4; mpAttribs[4].binding = 3; mpAttribs[4].format = VK_FORMAT_R32G32B32A32_SFLOAT; mpAttribs[4].offset = sizeof(float) * 4;
                    mpAttribs[5].location = 5; mpAttribs[5].binding = 3; mpAttribs[5].format = VK_FORMAT_R32G32B32A32_SFLOAT; mpAttribs[5].offset = sizeof(float) * 8;
                    mpAttribs[6].location = 6; mpAttribs[6].binding = 3; mpAttribs[6].format = VK_FORMAT_R32G32B32A32_SFLOAT; mpAttribs[6].offset = sizeof(float) * 12;

                    VkPipelineVertexInputStateCreateInfo mpVertInput{};
                    mpVertInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                    mpVertInput.vertexBindingDescriptionCount = 4;
                    mpVertInput.pVertexBindingDescriptions = mpBindings;
                    mpVertInput.vertexAttributeDescriptionCount = 7;
                    mpVertInput.pVertexAttributeDescriptions = mpAttribs;

                    VkPipelineInputAssemblyStateCreateInfo mpIA{};
                    mpIA.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                    mpIA.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

                    VkPipelineViewportStateCreateInfo mpVpState{};
                    mpVpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                    mpVpState.viewportCount = 1;
                    mpVpState.scissorCount = 1;

                    VkPipelineRasterizationStateCreateInfo mpRast{};
                    mpRast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                    mpRast.polygonMode = VK_POLYGON_MODE_FILL;
                    mpRast.lineWidth = 1.0f;
                    mpRast.cullMode = VK_CULL_MODE_NONE;
                    mpRast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

                    VkPipelineMultisampleStateCreateInfo mpMS{};
                    mpMS.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                    mpMS.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

                    VkPipelineDepthStencilStateCreateInfo mpDS{};
                    mpDS.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                    mpDS.depthTestEnable = VK_TRUE;
                    mpDS.depthWriteEnable = VK_TRUE;
                    mpDS.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

                    VkPipelineColorBlendAttachmentState mpCBA{};
                    mpCBA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

                    VkPipelineColorBlendStateCreateInfo mpCB{};
                    mpCB.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                    mpCB.attachmentCount = 1;
                    mpCB.pAttachments = &mpCBA;

                    VkDynamicState mpDynStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
                    VkPipelineDynamicStateCreateInfo mpDyn{};
                    mpDyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                    mpDyn.dynamicStateCount = 2;
                    mpDyn.pDynamicStates = mpDynStates;

                    VkGraphicsPipelineCreateInfo mpPCI{};
                    mpPCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                    mpPCI.stageCount = 2;
                    mpPCI.pStages = mpStages;
                    mpPCI.pVertexInputState = &mpVertInput;
                    mpPCI.pInputAssemblyState = &mpIA;
                    mpPCI.pViewportState = &mpVpState;
                    mpPCI.pRasterizationState = &mpRast;
                    mpPCI.pMultisampleState = &mpMS;
                    mpPCI.pDepthStencilState = &mpDS;
                    mpPCI.pColorBlendState = &mpCB;
                    mpPCI.pDynamicState = &mpDyn;
                    mpPCI.layout = m_interactiveViewport.materialPreviewPipelineLayout;
                    mpPCI.renderPass = m_interactiveViewport.renderPass;
                    mpPCI.subpass = 0;

                    // Guard: if layout creation failed (e.g. push constant size > maxPushConstantsSize
                    // or descriptor layout null on older GPU), skip to avoid crash with null handle.
                    VkResult mpResult = VK_ERROR_INITIALIZATION_FAILED;
                    if (m_interactiveViewport.materialPreviewPipelineLayout != VK_NULL_HANDLE &&
                        m_interactiveViewport.materialPreviewDescPool != VK_NULL_HANDLE) {
                        mpResult = vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &mpPCI, nullptr,
                                                             &m_interactiveViewport.materialPreviewPipeline);
                    } else {
                        SCENE_LOG_WARN("[Vulkan] Material preview pipeline skipped: layout or pool null (device limit exceeded).");
                    }
                    if (mpResult != VK_SUCCESS) {
                        SCENE_LOG_WARN("[Vulkan] Material preview pipeline creation failed.");
                        m_interactiveViewport.materialPreviewPipeline = VK_NULL_HANDLE;
                    } else {
                        // Allocate descriptor set and bind material buffer
                        VkDescriptorSetAllocateInfo mpDsai{};
                        mpDsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                        mpDsai.descriptorPool = m_interactiveViewport.materialPreviewDescPool;
                        mpDsai.descriptorSetCount = 1;
                        mpDsai.pSetLayouts = &m_interactiveViewport.materialPreviewDescLayout;
                        vkAllocateDescriptorSets(vkDevice, &mpDsai, &m_interactiveViewport.materialPreviewDescSet);

                        if (m_device->m_materialBuffer.buffer && m_interactiveViewport.materialPreviewDescSet) {
                            VkDescriptorBufferInfo matBufInfo{};
                            matBufInfo.buffer = m_device->m_materialBuffer.buffer;
                            matBufInfo.offset = 0;
                            matBufInfo.range = VK_WHOLE_SIZE;

                            VkWriteDescriptorSet mpWds{};
                            mpWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                            mpWds.dstSet = m_interactiveViewport.materialPreviewDescSet;
                            mpWds.dstBinding = 0;
                            mpWds.descriptorCount = 1;
                            mpWds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                            mpWds.pBufferInfo = &matBufInfo;
                            vkUpdateDescriptorSets(vkDevice, 1, &mpWds, 0, nullptr);

                            const VkBuffer terrainOrDummyBuffer = m_device->m_terrainLayerBuffer.buffer
                                ? m_device->m_terrainLayerBuffer.buffer
                                : m_device->m_materialBuffer.buffer;
                            if (terrainOrDummyBuffer) {
                                VkDescriptorBufferInfo terrainBufInfo{};
                                terrainBufInfo.buffer = terrainOrDummyBuffer;
                                terrainBufInfo.offset = 0;
                                terrainBufInfo.range = VK_WHOLE_SIZE;
                                VkWriteDescriptorSet terrainWds{};
                                terrainWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                terrainWds.dstSet = m_interactiveViewport.materialPreviewDescSet;
                                terrainWds.dstBinding = 3;
                                terrainWds.descriptorCount = 1;
                                terrainWds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                                terrainWds.pBufferInfo = &terrainBufInfo;
                                vkUpdateDescriptorSets(vkDevice, 1, &terrainWds, 0, nullptr);
                            }

                            if (!m_interactiveViewport.matcapImage.image) {
                                std::vector<uint8_t> white(4 * 2 * 2, 255);
                                const int64_t id = this->uploadTexture2D(white.data(), 2, 2, 4, false, false);
                                auto it = m_uploadedImages.find(id);
                                if (it != m_uploadedImages.end()) {
                                    m_interactiveViewport.matcapImage = it->second;
                                }
                            }
                            if (m_interactiveViewport.matcapImage.view && m_interactiveViewport.matcapImage.sampler) {
                                VkDescriptorImageInfo dummyInfo{};
                                dummyInfo.sampler = m_interactiveViewport.matcapImage.sampler;
                                dummyInfo.imageView = m_interactiveViewport.matcapImage.view;
                                dummyInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                std::vector<VkDescriptorImageInfo> dummyInfos(
                                    m_interactiveViewport.materialPreviewTextureArrayLen,
                                    dummyInfo);
                                VkWriteDescriptorSet dummyWds{};
                                dummyWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                dummyWds.dstSet = m_interactiveViewport.materialPreviewDescSet;
                                dummyWds.dstBinding = 1;
                                dummyWds.descriptorCount = m_interactiveViewport.materialPreviewTextureArrayLen;
                                dummyWds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                                dummyWds.pImageInfo = dummyInfos.data();
                                vkUpdateDescriptorSets(vkDevice, 1, &dummyWds, 0, nullptr);
                            }

                            // Backfill binding 1 with all textures already uploaded.
                            // Without this, when the Vulkan RT backend serves its own
                            // interactive viewport, Material Preview sees an empty texture
                            // array (paint strokes look "missing") because textures loaded
                            // before the descSet existed never reached binding 1.
                            for (auto& kv : m_uploadedImages) {
                                const int64_t texID = kv.first;
                                if (texID <= 0 || static_cast<uint32_t>(texID) >= m_interactiveViewport.materialPreviewTextureArrayLen) continue;
                                const VulkanRT::ImageHandle& texImg = kv.second;
                                if (!texImg.view || !texImg.sampler) continue;
                                VkDescriptorImageInfo tii{};
                                tii.sampler     = texImg.sampler;
                                tii.imageView   = texImg.view;
                                tii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                VkWriteDescriptorSet twds{};
                                twds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                twds.dstSet          = m_interactiveViewport.materialPreviewDescSet;
                                twds.dstBinding      = 1;
                                twds.dstArrayElement = static_cast<uint32_t>(texID);
                                twds.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                                twds.descriptorCount = 1;
                                twds.pImageInfo      = &tii;
                                vkUpdateDescriptorSets(vkDevice, 1, &twds, 0, nullptr);
                            }

                            if (m_interactiveViewport.matcapImage.view && m_interactiveViewport.matcapImage.sampler) {
                                VkDescriptorImageInfo envInfos[2]{};
                                for (auto& envInfo : envInfos) {
                                    envInfo.sampler = m_interactiveViewport.matcapImage.sampler;
                                    envInfo.imageView = m_interactiveViewport.matcapImage.view;
                                    envInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                }
                                VkWriteDescriptorSet envWds{};
                                envWds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                                envWds.dstSet = m_interactiveViewport.materialPreviewDescSet;
                                envWds.dstBinding = 2;
                                envWds.descriptorCount = 2;
                                envWds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                                envWds.pImageInfo = envInfos;
                                vkUpdateDescriptorSets(vkDevice, 1, &envWds, 0, nullptr);
                            }
                        }
                        SCENE_LOG_INFO("[Vulkan] Material preview pipeline created successfully.");
                    }
                }
                if (mpVertModule) vkDestroyShaderModule(vkDevice, mpVertModule, nullptr);
                if (mpFragModule) vkDestroyShaderModule(vkDevice, mpFragModule, nullptr);
            }
        }
    }

    if (m_interactiveViewport.width == width &&
        m_interactiveViewport.height == height &&
        m_interactiveViewport.framebuffer != VK_NULL_HANDLE &&
        m_interactiveViewport.colorImage.image != VK_NULL_HANDLE &&
        m_interactiveViewport.depthImage.image != VK_NULL_HANDLE &&
        m_interactiveViewport.stagingBuffer.buffer != VK_NULL_HANDLE) {
        return true;
    }

    destroyInteractiveViewportResources(true);

    m_interactiveViewport.colorImage = m_device->createImage2D(
        (uint32_t)width, (uint32_t)height, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    m_interactiveViewport.depthImage = m_device->createImage2D(
        (uint32_t)width, (uint32_t)height, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    VulkanRT::BufferCreateInfo stagingInfo{};
    stagingInfo.size = (uint64_t)width * (uint64_t)height * 4ull;
    stagingInfo.usage = VulkanRT::BufferUsage::TRANSFER_DST;
    stagingInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
    m_interactiveViewport.stagingBuffer = m_device->createBuffer(stagingInfo);

    if (!m_interactiveViewport.colorImage.image ||
        !m_interactiveViewport.depthImage.image ||
        !m_interactiveViewport.stagingBuffer.buffer) {
        destroyInteractiveViewportResources(true);
        return false;
    }

    VkImageView attachments[2] = {
        m_interactiveViewport.colorImage.view,
        m_interactiveViewport.depthImage.view
    };
    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = m_interactiveViewport.renderPass;
    framebufferInfo.attachmentCount = 2;
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = (uint32_t)width;
    framebufferInfo.height = (uint32_t)height;
    framebufferInfo.layers = 1;
    if (vkCreateFramebuffer(vkDevice, &framebufferInfo, nullptr, &m_interactiveViewport.framebuffer) != VK_SUCCESS) {
        destroyInteractiveViewportResources(true);
        return false;
    }

    m_interactiveViewport.width = width;
    m_interactiveViewport.height = height;
    return true;
}

void VulkanBackendAdapter::renderInteractiveViewportImpl(void* s, int width, int height, void* fb, void* tex) {
    m_interactiveViewport.initialized = true;
    const ViewportMode requestedMode = m_viewportMode;

    if (!m_loggedInteractiveViewportFallback) {
        const char* mode_name = "Interactive";
        switch (requestedMode) {
            case ViewportMode::Solid: mode_name = "Solid"; break;
            case ViewportMode::MaterialPreview: mode_name = "MaterialPreview"; break;
            case ViewportMode::Matcap: mode_name = "Matcap"; break;
            case ViewportMode::Rendered: mode_name = "Rendered"; break;
        }
        SCENE_LOG_INFO(std::string("[Vulkan] Interactive viewport mode selected: '")
                       + mode_name + "'.");
        m_loggedInteractiveViewportFallback = true;
    }

    auto resolveShaderDir = [&]() -> std::string {
        std::string shaderDir = "shaders";
        if (!std::filesystem::exists(shaderDir + "/solid.spv")) shaderDir = "source/shaders";
        if (!std::filesystem::exists(shaderDir + "/solid.spv")) shaderDir = "../shaders";
        if (!std::filesystem::exists(shaderDir + "/solid.spv")) {
            char exePath[MAX_PATH] = {};
            GetModuleFileNameA(nullptr, exePath, MAX_PATH);
            std::string exeDir = std::filesystem::path(exePath).parent_path().string();
            shaderDir = exeDir + "/shaders";
        }
        return shaderDir;
    };

    const bool rasterModeRequested = (requestedMode == ViewportMode::Solid ||
                                      requestedMode == ViewportMode::Matcap ||
                                      requestedMode == ViewportMode::MaterialPreview);
    const std::string shaderDir = resolveShaderDir();
    if (!rasterModeRequested ||
        !m_device ||
        !m_device->supportsGraphicsQueue() ||
        !ensureInteractiveViewportResources(shaderDir, width, height)) {
        m_viewportMode = ViewportMode::Rendered;
        renderProgressive(s, nullptr, nullptr, width, height, fb, tex);
        m_viewportMode = requestedMode;
        return;
    }

    // Camera change detection — only re-render when something changed
    auto hashCamera = [](const CameraParams& c) -> uint64_t {
        uint64_t h = 14695981039346656037ull;
        auto mix = [&](float v) { uint32_t bits; std::memcpy(&bits, &v, 4); h ^= bits; h *= 1099511628211ull; };
        mix(c.origin.x); mix(c.origin.y); mix(c.origin.z);
        mix(c.lookAt.x); mix(c.lookAt.y); mix(c.lookAt.z);
        mix(c.up.x); mix(c.up.y); mix(c.up.z);
        mix(c.fov);
        mix(c.orthographic ? 1.0f : 0.0f);
        mix(c.orthoHeight);
        mix((float)c.gridPlane);
        return h;
    };
    uint64_t camHash = hashCamera(m_camera);
    {
        // Grid settings join the change hash so slider edits invalidate the cached frame.
        auto mixGrid = [&](float v) { uint32_t bits; std::memcpy(&bits, &v, 4); camHash ^= bits; camHash *= 1099511628211ull; };
        mixGrid(::render_settings.grid_fade_distance);
        mixGrid(::render_settings.grid_opacity);
    }
    if (!m_interactiveViewport.dirty && camHash == m_lastCameraHash &&
        m_interactiveViewport.width == width && m_interactiveViewport.height == height) {
        // Nothing changed — just re-present the cached framebuffer
        std::vector<uint32_t>* framebuffer = static_cast<std::vector<uint32_t>*>(fb);
        if (framebuffer && !framebuffer->empty()) {
            if (s) {
                SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
                if (outSurf->pixels && outSurf->w == width && outSurf->h == height) {
                    std::memcpy(outSurf->pixels, framebuffer->data(), framebuffer->size() * sizeof(uint32_t));
                }
            }
            if (tex) {
                SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, framebuffer->data(), width * 4);
            }
            return;
        }
    }
    m_lastCameraHash = camHash;
    m_interactiveViewport.dirty = false;

    struct SolidPushConstants {
        float viewProj[16];
        float view[16];
        int useMatcap; // -1 = flat color, 0 = off, 1 = matcap texture, 2..9 procedural
        float overrideR, overrideG, overrideB; // flat color (useMatcap == -1)
        // Grid distance fade (world units, around fadeCenter). fadeEnd <= fadeStart disables.
        float fadeCenterX, fadeCenterY, fadeCenterZ;
        float fadeStart, fadeEnd;
        float overrideA; // base opacity for flat-color draws (grid)
    };

    auto matrixToGL = [](const Matrix4x4& mat, float out[16]) {
        Matrix4x4 t = mat.transpose();
        int k = 0;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[k++] = t.m[r][c];
            }
        }
    };

    auto makeViewMatrix = [](const Vec3& eye, const Vec3& center, const Vec3& up) {
        Vec3 f = (center - eye).normalize();
        Vec3 sAxis = Vec3::cross(f, up).normalize();
        if (sAxis.length() < 0.0001f) sAxis = Vec3(1.0f, 0.0f, 0.0f);
        Vec3 uAxis = Vec3::cross(sAxis, f);
        return Matrix4x4(
            sAxis.x, sAxis.y, sAxis.z, -Vec3::dot(sAxis, eye),
            uAxis.x, uAxis.y, uAxis.z, -Vec3::dot(uAxis, eye),
            -f.x,   -f.y,   -f.z,    Vec3::dot(f, eye),
            0.0f,   0.0f,   0.0f,    1.0f
        );
    };

    auto makePerspectiveMatrix = [&](float fovDeg, float aspect, float zNear, float zFar) {
        const float f = 1.0f / std::tan(fovDeg * 0.5f * 3.14159265358979f / 180.0f);
        Matrix4x4 proj = Matrix4x4::zero();
        proj.m[0][0] = f / aspect;
        proj.m[1][1] = -f; // Vulkan clip Y flip
        proj.m[2][2] = zFar / (zNear - zFar);
        proj.m[2][3] = (zFar * zNear) / (zNear - zFar);
        proj.m[3][2] = -1.0f;
        return proj;
    };

    auto makeOrthoMatrix = [&](float orthoHeight, float aspect, float zNear, float zFar) {
        const float oh = (orthoHeight > 1e-4f) ? orthoHeight : 10.0f;
        const float ow = oh * aspect;
        Matrix4x4 proj = Matrix4x4::zero();
        proj.m[0][0] = 2.0f / ow;
        proj.m[1][1] = -2.0f / oh;              // Vulkan clip Y flip
        proj.m[2][2] = 1.0f / (zNear - zFar);   // z_eye[-near,-far] -> NDC z[0,1]
        proj.m[2][3] = zNear / (zNear - zFar);
        proj.m[3][3] = 1.0f;
        return proj;
    };

    Matrix4x4 view = makeViewMatrix(m_camera.origin, m_camera.lookAt, m_camera.up);
    const float aspect = (height > 0) ? ((float)width / (float)height) : 1.0f;
    const float fovDeg = m_camera.fov > 1.0f ? m_camera.fov : 60.0f;
    Matrix4x4 proj = m_camera.orthographic
        ? makeOrthoMatrix(m_camera.orthoHeight, aspect, 0.01f, 1000000.0f)
        : makePerspectiveMatrix(fovDeg, aspect, 0.01f, 1000000.0f);
    Matrix4x4 viewProj = proj * view;

    const float tanHalfFov = std::tan(fovDeg * 0.5f * 3.14159265358979f / 180.0f);
    m_rasterCullCameraPosition = m_camera.origin;
    m_rasterCullFocalLengthPixels = (height > 0 && tanHalfFov > 1e-4f)
        ? ((0.5f * static_cast<float>(height)) / tanHalfFov)
        : 0.0f;
    m_rasterMinChunkScreenRadiusPixels = 3.0f;
    {
        double rasterQualityScale = 1.0;
        double baseBudgetMillions = 72.0;
        uint64_t minBudget = 24ull * 1000ull * 1000ull;
        uint64_t maxBudget = 96ull * 1000ull * 1000ull;
        bool allowAdaptiveRasterBudget = false;
        if (m_viewportMode == ViewportMode::Solid || m_viewportMode == ViewportMode::Matcap) {
            switch (::render_settings.raster_viewport_quality_preset) {
                case ::RasterViewportQualityPreset::Performance: rasterQualityScale = 0.58; break;
                case ::RasterViewportQualityPreset::Balanced: rasterQualityScale = 0.78; break;
                case ::RasterViewportQualityPreset::Quality: rasterQualityScale = 1.0; break;
                case ::RasterViewportQualityPreset::Auto:
                default: allowAdaptiveRasterBudget = true; break;
            }
        } else if (m_viewportMode == ViewportMode::MaterialPreview) {
            baseBudgetMillions = 36.0;
            minBudget = 8ull * 1000ull * 1000ull;
            maxBudget = 48ull * 1000ull * 1000ull;
            switch (::render_settings.raster_viewport_quality_preset) {
                case ::RasterViewportQualityPreset::Performance: rasterQualityScale = 0.45; break;
                case ::RasterViewportQualityPreset::Balanced: rasterQualityScale = 0.65; break;
                case ::RasterViewportQualityPreset::Quality: rasterQualityScale = 1.0; break;
                case ::RasterViewportQualityPreset::Auto:
                default:
                    rasterQualityScale = 0.72;
                    allowAdaptiveRasterBudget = true;
                    break;
            }
        }
        const double pixelCount = (std::max)(1.0, static_cast<double>(width) * static_cast<double>(height));
        const double referencePixels = 1920.0 * 1080.0;
        const double resolutionScale = std::sqrt(referencePixels / pixelCount);
        const double feedbackScale = allowAdaptiveRasterBudget
            ? (std::clamp)(static_cast<double>(m_rasterScatterBudgetScale), 0.35, 1.0)
            : 1.0;
        const uint64_t adaptiveBudget = static_cast<uint64_t>(
            baseBudgetMillions * 1000.0 * 1000.0 * resolutionScale * rasterQualityScale * feedbackScale);
        m_rasterScatterTriangleBudget = std::clamp<uint64_t>(adaptiveBudget, minBudget, maxBudget);
        if (!allowAdaptiveRasterBudget) {
            m_rasterScatterBudgetScale = 1.0f;
        }
    }

    if (!m_interactiveViewport.identityInstanceBuffer.buffer) {
        struct RasterInstanceGPU {
            float model[16];
        } identityGpu{};
        matrixToGL(Matrix4x4::identity(), identityGpu.model);
        VulkanRT::BufferCreateInfo ici{};
        ici.size = sizeof(identityGpu);
        ici.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
        ici.location = VulkanRT::MemoryLocation::GPU_ONLY;
        ici.initialData = nullptr;
        m_interactiveViewport.identityInstanceBuffer = m_device->createBuffer(ici);
        if (m_interactiveViewport.identityInstanceBuffer.buffer) {
            m_device->uploadBuffer(m_interactiveViewport.identityInstanceBuffer,
                                   &identityGpu,
                                   sizeof(identityGpu),
                                   0);
        }
    }

    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        m_interactiveViewport.dirty = true;
        return;
    }

    VkClearValue clearValues[2]{};
    // Use World's solid background color for the raster viewport clear
    const bool hasWorldBg = (m_cachedWorld.color.x != 0.0f || m_cachedWorld.color.y != 0.0f || m_cachedWorld.color.z != 0.0f);
    clearValues[0].color = { {
        hasWorldBg ? m_cachedWorld.color.x : 0.13f,
        hasWorldBg ? m_cachedWorld.color.y : 0.14f,
        hasWorldBg ? m_cachedWorld.color.z : 0.16f,
        1.0f } };
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_interactiveViewport.renderPass;
    renderPassInfo.framebuffer = m_interactiveViewport.framebuffer;
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = { (uint32_t)width, (uint32_t)height };
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Decide which pipeline to use based on viewport mode
    const bool useMaterialPreview = (m_viewportMode == ViewportMode::MaterialPreview) &&
                                     m_interactiveViewport.materialPreviewPipeline != VK_NULL_HANDLE &&
                                     m_interactiveViewport.materialPreviewDescSet != VK_NULL_HANDLE &&
                                     m_device->m_materialBuffer.buffer != VK_NULL_HANDLE;

    if (useMaterialPreview) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.materialPreviewPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_interactiveViewport.materialPreviewPipelineLayout,
                                0, 1, &m_interactiveViewport.materialPreviewDescSet, 0, nullptr);
    } else {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.solidPipeline);
        if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.pipelineLayout, 0, 1, &m_interactiveViewport.matcapDescSet, 0, nullptr);
        }
    }

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)width;
    viewport.height = (float)height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = { (uint32_t)width, (uint32_t)height };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Draw from raster buffers (lightweight, BLAS-independent)
    if (!m_rasterInstances.empty()) {
        for (const auto& [meshKey, rmb] : m_rasterMeshes) {
            if (!rmb.vertexBuffer.buffer || !rmb.instanceBuffer.buffer || rmb.vertexCount == 0 || rmb.instanceCount == 0) continue;

            const bool drawMaterialPreview =
                useMaterialPreview &&
                !rmb.isScatterProxy &&
                rmb.matIdBuffer.buffer &&
                rmb.uvBuffer.buffer;

            if (drawMaterialPreview) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.materialPreviewPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_interactiveViewport.materialPreviewPipelineLayout,
                                        0, 1, &m_interactiveViewport.materialPreviewDescSet, 0, nullptr);
                // Material Preview push constants
                struct MaterialPreviewPushConstants {
                    float viewProj[16];
                    float view[16];
                    float cameraPos[4];   // xyz + pad
                    float lightDir0[4];   // xyz = dir, w = intensity
                    float lightDir1[4];   // fill
                    float lightDir2[4];   // rim
                    uint32_t materialMeta[4];
                };
                MaterialPreviewPushConstants mpPush{};
                matrixToGL(viewProj, mpPush.viewProj);
                matrixToGL(view, mpPush.view);
                mpPush.cameraPos[0] = m_camera.origin.x;
                mpPush.cameraPos[1] = m_camera.origin.y;
                mpPush.cameraPos[2] = m_camera.origin.z;
                mpPush.cameraPos[3] = 0.0f;
                // Key light: from upper-right-front
                mpPush.lightDir0[0] = 0.45f; mpPush.lightDir0[1] = 0.8f; mpPush.lightDir0[2] = 0.35f; mpPush.lightDir0[3] = 2.5f;
                // Fill light: from left side, softer
                mpPush.lightDir1[0] = -0.6f; mpPush.lightDir1[1] = 0.3f; mpPush.lightDir1[2] = 0.4f; mpPush.lightDir1[3] = 0.8f;
                // Rim light: from behind
                mpPush.lightDir2[0] = -0.2f; mpPush.lightDir2[1] = 0.4f; mpPush.lightDir2[2] = -0.8f; mpPush.lightDir2[3] = 1.2f;
                uint32_t previewQuality = 2u;
                switch (::render_settings.raster_viewport_quality_preset) {
                    case ::RasterViewportQualityPreset::Performance: previewQuality = 1u; break;
                    case ::RasterViewportQualityPreset::Balanced: previewQuality = 2u; break;
                    case ::RasterViewportQualityPreset::Quality: previewQuality = 3u; break;
                    case ::RasterViewportQualityPreset::Auto:
                    default: previewQuality = 2u; break;
                }
                mpPush.materialMeta[0] = m_device ? m_device->m_materialCount : 0u;
                mpPush.materialMeta[1] = previewQuality;
                mpPush.materialMeta[2] = static_cast<uint32_t>(::render_settings.material_preview_lighting_preset);
                mpPush.materialMeta[3] = m_interactiveViewport.materialPreviewTextureArrayLen;

                vkCmdPushConstants(cmd, m_interactiveViewport.materialPreviewPipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(MaterialPreviewPushConstants), &mpPush);

                // Bind 4 vertex buffers: pos, normal, matId, instance
                VkBuffer mpVertBuffers[4] = {
                    rmb.vertexBuffer.buffer,
                    rmb.normalBuffer.buffer ? rmb.normalBuffer.buffer : rmb.vertexBuffer.buffer,
                    rmb.matIdBuffer.buffer ? rmb.matIdBuffer.buffer : rmb.vertexBuffer.buffer,
                    rmb.instanceBuffer.buffer
                };
                VkDeviceSize mpOffsets[4] = { 0, 0, 0, 0 };
                vkCmdBindVertexBuffers(cmd, 0, 4, mpVertBuffers, mpOffsets);
            } else {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_interactiveViewport.solidPipeline);
                if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            m_interactiveViewport.pipelineLayout,
                                            0, 1, &m_interactiveViewport.matcapDescSet, 0, nullptr);
                }
                // Solid / Matcap push constants
                SolidPushConstants push{};
                matrixToGL(viewProj, push.viewProj);
                matrixToGL(view, push.view);
                const bool isMatcap = (m_viewportMode == ViewportMode::Matcap);
                if (isMatcap) {
                    if (m_interactiveViewport.matcapUserLoaded) {
                        push.useMatcap = 1;
                    } else {
                        push.useMatcap = m_interactiveViewport.matcapPreset;
                    }
                } else {
                    push.useMatcap = 0;
                }
                vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SolidPushConstants), &push);

                VkBuffer vertexBuffers[3] = {
                    rmb.vertexBuffer.buffer,
                    rmb.normalBuffer.buffer ? rmb.normalBuffer.buffer : rmb.vertexBuffer.buffer,
                    rmb.instanceBuffer.buffer
                };
                VkDeviceSize offsets[3] = { 0, 0, 0 };
                vkCmdBindVertexBuffers(cmd, 0, 3, vertexBuffers, offsets);
            }

            if (rmb.indexBuffer.buffer && rmb.indexCount > 0) {
                vkCmdBindIndexBuffer(cmd, rmb.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, rmb.indexCount, rmb.instanceCount, 0, 0, 0);
            } else {
                vkCmdDraw(cmd, rmb.vertexCount, rmb.instanceCount, 0, 0);
            }
        }
    } else if (m_rasterGeometryDirty) {
        // Fallback: draw from BLAS buffers only while raster geometry is still waiting
        // for its first build/sync. If raster geometry was already rebuilt and ended up
        // empty, drawing stale RT BLAS data here causes ghost objects in Solid mode.
        for (const auto& instance : m_vkInstances) {
            if (instance.mask == 0) continue;
            if (instance.blasIndex >= m_device->m_blasList.size()) continue;

            const auto& blas = m_device->m_blasList[instance.blasIndex];
            if (!blas.vertexBuffer.buffer || blas.vertexCount == 0) continue;
            if (!m_interactiveViewport.identityInstanceBuffer.buffer) continue;

            SolidPushConstants push{};
            matrixToGL(viewProj, push.viewProj);
            matrixToGL(view, push.view);
            const bool isMatcap = (m_viewportMode == ViewportMode::Matcap);
            if (isMatcap) {
                push.useMatcap = m_interactiveViewport.matcapUserLoaded ? 1 : m_interactiveViewport.matcapPreset;
            } else {
                push.useMatcap = 0;
            }
            vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SolidPushConstants), &push);

            VkBuffer vertexBuffers[3] = {
                blas.vertexBuffer.buffer,
                blas.normalBuffer.buffer ? blas.normalBuffer.buffer : blas.vertexBuffer.buffer,
                m_interactiveViewport.identityInstanceBuffer.buffer
            };
            VkDeviceSize offsets[3] = { 0, 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 3, vertexBuffers, offsets);

            if (blas.indexBuffer.buffer && blas.indexCount > 0) {
                vkCmdBindIndexBuffer(cmd, blas.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, blas.indexCount, 1, 0, 0, 0);
            } else {
                vkCmdDraw(cmd, blas.vertexCount, 1, 0, 0);
            }
        }
    }

    // ── Reference Grid (auto-orients to the active plane, adaptive spacing) ──
    {
        const int activeGridPlane = m_camera.gridPlane;
        // Use the camera's perpendicular distance to the active grid plane
        // instead of orbit distance (origin-lookAt), which is unreliable because
        // lookAt can jump when the user clicks a new focus point or when orbit
        // center resets. Perpendicular distance:
        //  - Stays constant during pan (camera slides parallel to the grid)
        //  - Changes smoothly during orbit/zoom
        //  - Directly reflects how much grid the user actually sees
        float viewScale;
        if (m_camera.orthographic) {
            viewScale = (m_camera.orthoHeight > 1e-4f) ? m_camera.orthoHeight : 10.0f;
        } else {
            float planeDist;
            switch (activeGridPlane) {
                case 1:  planeDist = std::abs(m_camera.origin.z); break; // XY plane
                case 2:  planeDist = std::abs(m_camera.origin.x); break; // YZ plane
                default: planeDist = std::abs(m_camera.origin.y); break; // XZ floor
            }
            // Scale by FOV so wider lenses get coarser grid (more visible area)
            const float fovRad = (m_camera.fov > 1.0f ? m_camera.fov : 60.0f) * 3.14159265f / 180.0f;
            const float fovScale = 2.0f * std::tan(fovRad * 0.5f);
            viewScale = planeDist * fovScale;
            // Fallback: when camera is ON the grid plane (planeDist ~ 0), use orbit distance
            if (viewScale < 0.5f) {
                const float orbitDist = (m_camera.origin - m_camera.lookAt).length();
                viewScale = std::max(orbitDist * fovScale, 1.0f);
            }
        }
        // Finer 1-2-5 sub-steps to reduce visible grid "pop" (max ratio ≈ 1.5×).
        auto niceStep = [](float x) -> float {
            if (x <= 1e-6f) return 1.0f;
            const float p = std::pow(10.0f, std::floor(std::log10(x)));
            const float n = x / p; // 1..10
            float m;
            if      (n < 1.25f) m = 1.0f;
            else if (n < 1.75f) m = 1.5f;
            else if (n < 2.5f)  m = 2.0f;
            else if (n < 4.0f)  m = 3.0f;
            else if (n < 6.0f)  m = 5.0f;
            else if (n < 8.5f)  m = 7.0f;
            else                m = 10.0f;
            return m * p;
        };
        // Hysteresis: only switch spacing when the new value differs by >15%
        // from the currently built spacing. This prevents rapid flip-flopping
        // at the boundary of two nice-step buckets during camera pan.
        float candidateSpacing = niceStep(viewScale * 0.1f);
        const float builtSpacing = m_interactiveViewport.gridBuiltSpacing;
        if (builtSpacing > 0.0f) {
            const float ratio = candidateSpacing / builtSpacing;
            if (ratio > 0.87f && ratio < 1.15f) {
                candidateSpacing = builtSpacing; // keep current
            }
        }
        const float spacing = candidateSpacing;

        // The grid stays ORIGIN-centred (the world axes are the visual anchor), but its extent
        // must not be cropped to the zoom alone: gridHalf ~ viewScale meant a floor close-up far
        // from the origin shrank the patch away from under the camera and the grid seemed to
        // vanish. The camera's planar distance from the origin now sets a lower bound instead.
        float camU, camV;
        switch (activeGridPlane) {
            case 1:  camU = m_camera.origin.x; camV = m_camera.origin.y; break; // XY (Front/Back)
            case 2:  camU = m_camera.origin.z; camV = m_camera.origin.y; break; // YZ (Left/Right)
            default: camU = m_camera.origin.x; camV = m_camera.origin.z; break; // XZ floor
        }
        const float camPlanarDist = std::max(std::abs(camU), std::abs(camV));
        // User knob: scales the fog horizon (and with it the built extents below). Growth
        // re-triggers a rebuild via extentStale; shrink may keep oversized geometry, which the
        // draw-side fade bands simply dissolve earlier.
        const float gridFadeScale = std::clamp(::render_settings.grid_fade_distance, 0.25f, 4.0f);
        // 16x viewScale: the major lattice must reach past the fade horizon (~19x viewScale, see
        // drawSeg fade bands below) so lines dissolve into the fog instead of ending at a visible
        // geometric edge. With the 1.25x build margin the edge sits at >=20x viewScale.
        const float requiredHalf = viewScale * 16.0f * gridFadeScale + camPlanarDist;
        // Minor (fine) lattice is camera-centred (see rebuild below); snap its centre to the
        // major (10x) lattice so minor lines always land on the global grid as it follows the camera.
        const float coarseSpacing = spacing * 10.0f;
        const float fineCenterU = std::round(camU / coarseSpacing) * coarseSpacing;
        const float fineCenterV = std::round(camV / coarseSpacing) * coarseSpacing;
        // Hysteresis: rebuild when the camera outgrows the built extent, or when the built
        // extent is so oversized (>2.5x) that the grid should shrink back. The 1.25x build
        // margin below keeps small camera moves from re-triggering this every frame.
        const bool extentStale =
            requiredHalf > m_interactiveViewport.gridBuiltHalf ||
            requiredHalf < m_interactiveViewport.gridBuiltHalf * 0.4f;

        if (!m_interactiveViewport.gridVertexBuffer.buffer ||
            m_interactiveViewport.gridBuiltPlane != activeGridPlane ||
            m_interactiveViewport.gridBuiltSpacing != spacing ||
            m_interactiveViewport.gridBuiltCenterU != fineCenterU ||
            m_interactiveViewport.gridBuiltCenterV != fineCenterV ||
            extentStale) {
            if (m_interactiveViewport.gridVertexBuffer.buffer) {
                m_device->destroyBuffer(m_interactiveViewport.gridVertexBuffer);
                m_interactiveViewport.gridVertexBuffer = {};
            }
            if (m_interactiveViewport.gridNormalBuffer.buffer) {
                m_device->destroyBuffer(m_interactiveViewport.gridNormalBuffer);
                m_interactiveViewport.gridNormalBuffer = {};
            }

            // Two-tier lattice. MINOR lines (current spacing) only span the visible area around
            // the camera footprint — building them across the whole extended extent made distant
            // lines collapse below a pixel and moiré badly in perspective (especially at 720p,
            // no MSAA). MAJOR lines (10x spacing, 10x thickness = every 10th minor) span the full
            // extent from the origin out past the camera, so the far field stays referenced with
            // far fewer, fatter lines. Axes are unchanged (origin-anchored, full extent). 25%
            // growth margin so small camera moves don't immediately re-trigger a rebuild.
            const float desiredHalf = requiredHalf * 1.25f;
            const int   coarseEachSide = std::clamp((int)std::ceil(desiredHalf / coarseSpacing), 4, 2048);
            const float gridHalf = coarseSpacing * (float)coarseEachSide;
            // 18x viewScale: minor lines reach well past the working area (10x still read as an
            // early cutoff); the distance fade below dissolves them before the geometric edge and
            // before they collapse sub-pixel and moiré (~40x viewScale at 720p).
            const int   fineEachSide = std::clamp((int)std::ceil((viewScale * 18.0f * gridFadeScale) / spacing), 10, 1024);
            const float fineHalf = spacing * (float)fineEachSide;
            const float step     = spacing;
            const float thin     = spacing * 0.008f; // thicker so 720p (no MSAA) doesn't break lines up
            const float coarseThin = coarseSpacing * 0.008f;
            const float axisThin = spacing * 0.022f;

            // Plane basis: ax = horizontal axis, ay = vertical axis, nrm = plane normal.
            Vec3 ax, ay, nrm;
            switch (activeGridPlane) {
                case 1: ax = Vec3(1,0,0); ay = Vec3(0,1,0); nrm = Vec3(0,0,1); break; // XY (Front/Back)
                case 2: ax = Vec3(0,0,1); ay = Vec3(0,1,0); nrm = Vec3(1,0,0); break; // YZ (Left/Right)
                default: ax = Vec3(1,0,0); ay = Vec3(0,0,1); nrm = Vec3(0,1,0); break; // XZ (Top/Persp floor)
            }
            const Vec3 nOff = nrm * (spacing * 0.0002f);

            std::vector<float> positions, normals;
            auto addLineQuad = [&](Vec3 a, Vec3 b, Vec3 widthDir) {
                Vec3 w = widthDir;
                Vec3 p0 = a - w, p1 = a + w, p2 = b + w, p3 = b - w;
                Vec3 n = nrm;
                // Emit triangles (front-facing) and reversed (back-facing) so grid is double-sided
                for (const Vec3& p : {p0, p1, p2, p0, p2, p3, p2, p1, p0, p3, p2, p0}) {
                    positions.push_back(p.x); positions.push_back(p.y); positions.push_back(p.z);
                    normals.push_back(n.x);   normals.push_back(n.y);   normals.push_back(n.z);
                }
            };

            // Major lattice (full extent, origin-centred; i==0 is covered by the axis quads).
            // Kept in its own segment so it can fade at the far horizon while the minor
            // lattice fades earlier (separate fade bands per drawSeg call).
            uint32_t segMajorStart = 0;
            for (int i = -coarseEachSide; i <= coarseEachSide; ++i) {
                if (i == 0) continue;
                const float u = coarseSpacing * (float)i;
                addLineQuad(ax * u - ay * gridHalf, ax * u + ay * gridHalf, ax * coarseThin);
            }
            for (int i = -coarseEachSide; i <= coarseEachSide; ++i) {
                if (i == 0) continue;
                const float vv = coarseSpacing * (float)i;
                addLineQuad(ay * vv - ax * gridHalf, ay * vv + ax * gridHalf, ay * coarseThin);
            }
            uint32_t segMajorCount = (uint32_t)(positions.size() / 3) - segMajorStart;

            // Minor lattice (visible area, centred on the camera footprint snapped to the major
            // lattice — every 10th index lands on a major line / the axes and is skipped).
            uint32_t segMinorStart = (uint32_t)(positions.size() / 3);
            for (int i = -fineEachSide; i <= fineEachSide; ++i) {
                if (((int)std::lround(fineCenterU / step) + i) % 10 == 0) continue;
                const float u = fineCenterU + step * (float)i;
                addLineQuad(ax * u + ay * (fineCenterV - fineHalf), ax * u + ay * (fineCenterV + fineHalf), ax * thin);
            }
            for (int i = -fineEachSide; i <= fineEachSide; ++i) {
                if (((int)std::lround(fineCenterV / step) + i) % 10 == 0) continue;
                const float vv = fineCenterV + step * (float)i;
                addLineQuad(ay * vv + ax * (fineCenterU - fineHalf), ay * vv + ax * (fineCenterU + fineHalf), ay * thin);
            }
            uint32_t segMinorCount = (uint32_t)(positions.size() / 3) - segMinorStart;

            // +U axis (red)
            uint32_t segAxisUStart = (uint32_t)(positions.size() / 3);
            addLineQuad(nOff, ax * gridHalf + nOff, ay * axisThin);
            uint32_t segAxisUCount = (uint32_t)(positions.size() / 3) - segAxisUStart;

            // +V axis (blue)
            uint32_t segAxisVStart = (uint32_t)(positions.size() / 3);
            addLineQuad(nOff, ay * gridHalf + nOff, ax * axisThin);
            uint32_t segAxisVCount = (uint32_t)(positions.size() / 3) - segAxisVStart;

            // -U/-V axis halves (dim grey)
            uint32_t segNegStart = (uint32_t)(positions.size() / 3);
            addLineQuad(ax * -gridHalf + nOff, nOff, ay * thin);
            addLineQuad(ay * -gridHalf + nOff, nOff, ax * thin);
            uint32_t segNegCount = (uint32_t)(positions.size() / 3) - segNegStart;

            m_interactiveViewport.gridBuiltPlane = activeGridPlane;
            m_interactiveViewport.gridBuiltSpacing = spacing;
            // Store the UNCLAMPED target so a hit line-cap doesn't re-trigger a rebuild every frame.
            m_interactiveViewport.gridBuiltHalf = std::max(gridHalf, desiredHalf);
            m_interactiveViewport.gridBuiltCenterU = fineCenterU;
            m_interactiveViewport.gridBuiltCenterV = fineCenterV;
            m_interactiveViewport.gridBuiltFineHalf = fineHalf;

            m_interactiveViewport.gridVertexCount = (uint32_t)(positions.size() / 3);
            m_interactiveViewport.gridSegments[0] = segMajorStart;  m_interactiveViewport.gridSegments[1] = segMajorCount;
            m_interactiveViewport.gridSegments[2] = segMinorStart;  m_interactiveViewport.gridSegments[3] = segMinorCount;
            m_interactiveViewport.gridSegments[4] = segAxisUStart;  m_interactiveViewport.gridSegments[5] = segAxisUCount;
            m_interactiveViewport.gridSegments[6] = segAxisVStart;  m_interactiveViewport.gridSegments[7] = segAxisVCount;
            m_interactiveViewport.gridSegments[8] = segNegStart;    m_interactiveViewport.gridSegments[9] = segNegCount;

            VulkanRT::BufferCreateInfo vci{};
            vci.size = positions.size() * sizeof(float);
            vci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            vci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            m_interactiveViewport.gridVertexBuffer = m_device->createBuffer(vci);
            if (m_interactiveViewport.gridVertexBuffer.buffer)
                m_device->uploadBuffer(m_interactiveViewport.gridVertexBuffer, positions.data(), vci.size, 0);

            VulkanRT::BufferCreateInfo nci{};
            nci.size = normals.size() * sizeof(float);
            nci.usage = VulkanRT::BufferUsage::VERTEX | VulkanRT::BufferUsage::TRANSFER_DST;
            nci.location = VulkanRT::MemoryLocation::GPU_ONLY;
            m_interactiveViewport.gridNormalBuffer = m_device->createBuffer(nci);
            if (m_interactiveViewport.gridNormalBuffer.buffer)
                m_device->uploadBuffer(m_interactiveViewport.gridNormalBuffer, normals.data(), nci.size, 0);
        }

        if (m_interactiveViewport.gridVertexBuffer.buffer && m_interactiveViewport.gridVertexCount > 0) {
            Matrix4x4 identity = Matrix4x4::identity();
            Matrix4x4 gridMvp = proj * view;

            VkBuffer gridBufs[3] = {
                m_interactiveViewport.gridVertexBuffer.buffer,
                m_interactiveViewport.gridNormalBuffer.buffer,
                m_interactiveViewport.identityInstanceBuffer.buffer
            };
            VkDeviceSize gridOff[3] = { 0, 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 3, gridBufs, gridOff);

            // Distance fade is measured IN-PLANE from the built minor-patch centre (not 3D camera
            // distance — that would wipe the whole grid in ortho, where the camera sits far away).
            Vec3 fadeAx, fadeAy;
            switch (m_interactiveViewport.gridBuiltPlane) {
                case 1:  fadeAx = Vec3(1, 0, 0); fadeAy = Vec3(0, 1, 0); break; // XY
                case 2:  fadeAx = Vec3(0, 0, 1); fadeAy = Vec3(0, 1, 0); break; // YZ
                default: fadeAx = Vec3(1, 0, 0); fadeAy = Vec3(0, 0, 1); break; // XZ
            }
            const Vec3 fadeCenter = fadeAx * m_interactiveViewport.gridBuiltCenterU +
                                    fadeAy * m_interactiveViewport.gridBuiltCenterV;
            // Minor lines dissolve just inside their built extent so the geometric edge is never
            // visible; majors/axes fog out at the far horizon (kept below the >=20x viewScale
            // major-lattice edge guaranteed by requiredHalf above). The built extent caps the
            // minor band because shrinking grid_fade_distance doesn't shrink built geometry.
            const float gridOpacity = std::clamp(::render_settings.grid_opacity, 0.0f, 1.0f);
            const float builtCoarse = m_interactiveViewport.gridBuiltSpacing * 10.0f;
            const float minorGeomLimit = std::max(m_interactiveViewport.gridBuiltFineHalf - 2.0f * builtCoarse, builtCoarse);
            const float minorFadeEnd = std::min(minorGeomLimit, viewScale * 16.0f * gridFadeScale);
            const float minorFadeStart = minorFadeEnd * 0.45f;
            const float majorFadeStart = viewScale * 12.0f * gridFadeScale;
            const float majorFadeEnd = viewScale * 19.0f * gridFadeScale;

            auto drawSeg = [&](uint32_t first, uint32_t count, float r, float g, float b,
                               float fadeStart, float fadeEnd) {
                if (!count) return;
                SolidPushConstants gp{};
                matrixToGL(gridMvp, gp.viewProj);
                matrixToGL(identity, gp.view);
                gp.useMatcap = -1;
                gp.overrideR = r; gp.overrideG = g; gp.overrideB = b;
                gp.fadeCenterX = fadeCenter.x; gp.fadeCenterY = fadeCenter.y; gp.fadeCenterZ = fadeCenter.z;
                gp.fadeStart = fadeStart; gp.fadeEnd = fadeEnd;
                gp.overrideA = gridOpacity;
                vkCmdPushConstants(cmd, m_interactiveViewport.pipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SolidPushConstants), &gp);
                vkCmdDraw(cmd, count, 1, first, 0);
            };

            if (gridOpacity > 0.01f) {
                const auto* s = m_interactiveViewport.gridSegments;
                // Draw raster grid (depth-tested) so it is occluded by scene geometry.
                drawSeg(s[2], s[3], 0.38f, 0.38f, 0.38f, minorFadeStart, minorFadeEnd); // minor lattice: grey
                drawSeg(s[0], s[1], 0.38f, 0.38f, 0.38f, majorFadeStart, majorFadeEnd); // major lattice: grey
                drawSeg(s[4], s[5], 0.75f, 0.15f, 0.15f, majorFadeStart, majorFadeEnd); // +X axis: red
                drawSeg(s[6], s[7], 0.20f, 0.45f, 0.95f, majorFadeStart, majorFadeEnd); // +Z axis: bright blue
                drawSeg(s[8], s[9], 0.30f, 0.30f, 0.30f, majorFadeStart, majorFadeEnd); // negative halves: dim grey
            }
        }
    }

    vkCmdEndRenderPass(cmd);
    m_device->endSingleTimeCommands(cmd);

    m_device->copyImageToBuffer(m_interactiveViewport.colorImage, m_interactiveViewport.stagingBuffer);

    std::vector<uint32_t>* framebuffer = static_cast<std::vector<uint32_t>*>(fb);
    const size_t pixelCount = (size_t)width * (size_t)height;
    if (framebuffer->size() != pixelCount) {
        framebuffer->resize(pixelCount);
    }
    m_device->downloadBuffer(m_interactiveViewport.stagingBuffer, framebuffer->data(), pixelCount * sizeof(uint32_t));

    if (s) {
        SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
        if (outSurf->pixels && outSurf->w == width && outSurf->h == height) {
            std::memcpy(outSurf->pixels, framebuffer->data(), pixelCount * sizeof(uint32_t));
        }
    }
    if (tex) {
        SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, framebuffer->data(), width * 4);
    }
}

void VulkanBackendAdapter::renderProgressiveImpl(void* s, void* w, void* r, int width, int height, void* fb, void* tex) {
    (void)w; (void)r; 
    
    // [VULKAN THREAD SAFETY] Mutex prevents background updateGeometry from destroying resources mid-frame.
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // [DIAG] Track render path — separate counter for Rendered mode
    static int s_diagFrameCounter = 0;
    static int s_diagRenderedCounter = 0;
    ++s_diagFrameCounter;
    const bool isRenderedMode = (m_viewportMode == ViewportMode::Rendered);   
    if (isRenderedMode) ++s_diagRenderedCounter;   

    // [FIX] Helper lambda: re-present last valid framebuffer to SDL surface/texture.
    // When the RT pipeline or TLAS isn't ready yet, or accumulation is already
    // complete, we must still write valid pixel data to the output surface so
    // Main.cpp's display blit doesn't show uninitialized memory (black/white/garbage).
    auto rePresentCachedFrame = [&]() {
        if (!fb || !s) return;
        std::vector<uint32_t>* framebuffer = static_cast<std::vector<uint32_t>*>(fb);
        const size_t pixelCount = (size_t)width * (size_t)height;
        // If framebuffer hasn't been filled yet, allocate and zero-fill so the
        // surface gets deterministic black instead of uninitialized garbage.
        if (framebuffer->size() != pixelCount) {
            framebuffer->resize(pixelCount, 0u);
        }
        SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
        if (outSurf && outSurf->pixels && outSurf->w == width && outSurf->h == height) {
            std::memcpy(outSurf->pixels, framebuffer->data(), pixelCount * sizeof(uint32_t));
        }
        if (tex) {
            SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, framebuffer->data(), width * 4);
        }
    };

    // [STABILITY] Robust null checks to prevent startup/switch crashes
    if (!m_device || !m_device->isInitialized()) {
        rePresentCachedFrame();
        return;
    }

    if (shouldUseInteractiveViewport()) {
        const ViewportMode requestedMode = m_viewportMode;
        renderInteractiveViewport(s, width, height, fb, tex);
        m_viewportMode = requestedMode;
        return;
    }

    // tex==nullptr is allowed (animation render path passes null so the worker
    // thread never touches the shared SDL_Texture concurrently with the main
    // thread's SDL_RenderCopy). All SDL_UpdateTexture sites below guard with
    // `if (tex)`. Animation still relies on `fb` (host framebuffer vector) for
    // pixel readback.
    if (!m_device->hasHardwareRT() || !fb) {
        // Still write valid (black) pixels so original_surface isn't left uninitialized
        rePresentCachedFrame();
        return;
    }
    
    if (isAccumulationComplete()) {
        // [FIX] Even when done accumulating, re-present the last valid frame
        // so original_surface always has valid pixel data for display blit.
        rePresentCachedFrame();
        return;
    }

    auto presentBackgroundOnly = [&]() {
        std::vector<uint32_t>* framebuffer = static_cast<std::vector<uint32_t>*>(fb);
        const size_t count = (size_t)width * (size_t)height;
        if (framebuffer->size() != count) {
            framebuffer->resize(count);
        }

        static SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA32);
        static std::unique_ptr<World> fallbackWorld;
        static uint64_t fallbackWorldHash = 0;
        auto hashWorld = [](const WorldData& wd) -> uint64_t {
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&wd);
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < sizeof(WorldData); ++i) {
                h ^= bytes[i];
                h *= 1099511628211ull;
            }
            return h;
        };
        uint64_t currentHash = hashWorld(m_cachedWorld);
        if (!fallbackWorld || fallbackWorldHash != currentHash) {
            fallbackWorld = std::make_unique<World>();
            fallbackWorld->data = m_cachedWorld;
            if (fallbackWorld->data.mode == WORLD_MODE_NISHITA) {
                fallbackWorld->initializeLUT();
            }
            fallbackWorldHash = currentHash;
        }

        float aspect = (height > 0) ? ((float)width / (float)height) : 1.0f;
        float fov = this->m_camera.fov > 1.0f ? this->m_camera.fov : 60.0f;
        float h_half = tanf(fov * 0.5f * 3.14159f / 180.0f);
        float viewport_h = 2.0f * h_half;
        float viewport_w = aspect * viewport_h;

        Vec3 lookFrom = this->m_camera.origin;
        Vec3 lookAt = this->m_camera.lookAt;
        if ((lookFrom - lookAt).length() < 0.0001f) {
            lookAt = lookFrom + Vec3(0.0f, 0.0f, -1.0f);
        }
        Vec3 camW = (lookFrom - lookAt).normalize();
        Vec3 camU = Vec3::cross(this->m_camera.up, camW).normalize();
        if (camU.length() < 0.0001f) camU = Vec3(1.0f, 0.0f, 0.0f);
        Vec3 camV = Vec3::cross(camW, camU);
        float focus_dist = this->m_camera.focusDistance > 0.001f ? this->m_camera.focusDistance : 1.0f;
        Vec3 horizontal = focus_dist * viewport_w * camU;
        Vec3 vertical = focus_dist * viewport_h * camV;
        Vec3 lowerLeft = lookFrom - horizontal * 0.5f - vertical * 0.5f - focus_dist * camW;

        auto sanitize = [](float v) -> float {
            if (std::isnan(v)) return 0.0f;
            if (std::isinf(v)) return (v > 0.0f) ? 65504.0f : 0.0f;
            return std::max(v, 0.0f);
        };

        uint32_t* fbData = framebuffer->data();
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                float u = ((float)i + 0.5f) / (float)width;
                float v = 1.0f - (((float)j + 0.5f) / (float)height);
                Vec3 dir = (lowerLeft + u * horizontal + v * vertical - lookFrom).normalize();
                Vec3 c = fallbackWorld->evaluate(dir, lookFrom) * m_cachedWorld.color_intensity;
                float rr = sanitize(c.x);
                float gg = sanitize(c.y);
                float bb = sanitize(c.z);
                rr = rr / (rr + 1.0f);
                gg = gg / (gg + 1.0f);
                bb = bb / (bb + 1.0f);
                const uint8_t ri = linearToSRGB8Fast(rr);
                const uint8_t gi = linearToSRGB8Fast(gg);
                const uint8_t bi = linearToSRGB8Fast(bb);
                size_t idx = (size_t)j * (size_t)width + (size_t)i;
                fbData[idx] = SDL_MapRGB(fmt, ri, gi, bi);
            }
        }

        // Bulk copy framebuffer → SDL surface (single memcpy instead of per-pixel write)
        if (s) {
            SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
            if (outSurf && outSurf->pixels && outSurf->w == width && outSurf->h == height) {
                std::memcpy(outSurf->pixels, fbData, count * sizeof(uint32_t));
            }
        }
        if (tex) {
            SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, fbData, width * 4);
        }
    };

    // If a reset requested immediate UI clear, wipe the provided framebuffer/texture now.
    // In RT Rendered mode this can cause a white/blank viewport if the next frame
    // temporarily skips tracing (for example while TLAS/RT readiness catches up after
    // a camera or scene change), so keep the last valid host-side image in that mode.
    const bool allowImmediateHostClear = shouldUseInteractiveViewport();
    if (m_forceClearOnNextPresent && allowImmediateHostClear) {
        // Clear CPU-side framebuffer vector if provided
        if (fb) {
            auto vec = static_cast<std::vector<uint32_t>*>(fb);
            if (vec) {
                size_t count = (size_t)width * (size_t)height;
                if (vec->size() != count) vec->resize(count);
                std::fill(vec->begin(), vec->end(), 0u);

                // Update SDL texture immediately so UI shows cleared image
                if (tex) {
                    SDL_Texture* sdlTex = static_cast<SDL_Texture*>(tex);
                    SDL_UpdateTexture(sdlTex, nullptr, vec->data(), width * 4);
                }
            }
        }
        if (s) {
            SDL_Surface* surf = static_cast<SDL_Surface*>(s);
            if (surf && surf->pixels) {
                memset(surf->pixels, 0, (size_t)width * (size_t)height * 4);
            }
        }
        m_forceClearOnNextPresent = false;
    }

    // Use robust shader dir detection — try several common locations
    std::string shaderDir = "shaders";
    if (!std::filesystem::exists(shaderDir + "/raygen.spv"))
        shaderDir = "source/shaders"; // VS dev layout: run from raytrac_sdl2/
    if (!std::filesystem::exists(shaderDir + "/raygen.spv"))
        shaderDir = "../shaders"; // Exe inside x64/Release, shaders at project/shaders
    if (!std::filesystem::exists(shaderDir + "/raygen.spv")) {
        // Try exe-relative path
        char exePath[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, exePath, MAX_PATH);
        std::string exeDir = std::filesystem::path(exePath).parent_path().string();
        shaderDir = exeDir + "/shaders";
    }

    // 1. Recreate output image if size changed
    if (m_imageWidth != width || m_imageHeight != height) {
        m_hasPresentedRenderedFrame = false;
        if (m_outputImage.image) m_device->destroyImage(m_outputImage);
        if (m_varianceImage.image) m_device->destroyImage(m_varianceImage);
        if (m_stagingBuffer.buffer) m_device->destroyBuffer(m_stagingBuffer);
        // Drain any in-flight frame slots before destroying images/stagings they may
        // still be writing. Aşama 2 ping-pong submits without waiting, so we MUST
        // synchronize before reallocation or the GPU writes into freed memory.
        for (uint32_t i = 0; i < VulkanRT::VulkanDevice::kFrameSlotCount; ++i) {
            if (m_tonemappedSlotInFlight[i]) {
                m_device->waitFrameSlot(i);
                m_tonemappedSlotInFlight[i] = false;
            }
        }
        if (m_tonemappedImage.image) m_device->destroyImage(m_tonemappedImage);
        for (auto& s : m_tonemappedStagings) {
            if (s.buffer) m_device->destroyBuffer(s);
            s = {};
        }
        m_tonemappedImage = {};
        m_tonemappedFrameSlot = 0;
        if (m_denoiserColorImage.image) m_device->destroyImage(m_denoiserColorImage);
        if (m_denoiserAlbedoImage.image) m_device->destroyImage(m_denoiserAlbedoImage);
        if (m_denoiserNormalImage.image) m_device->destroyImage(m_denoiserNormalImage);
        if (m_denoiserPositionImage.image) m_device->destroyImage(m_denoiserPositionImage);
        if (m_denoiserColorStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserColorStagingBuffer);
        if (m_denoiserAlbedoStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserAlbedoStagingBuffer);
        if (m_denoiserNormalStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserNormalStagingBuffer);
        if (m_denoiserPositionStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserPositionStagingBuffer);

        // Tear down CUDA interop state + shared staging before reallocating.
        // Interop destroy is a no-op on first call (null ptr guards inside).
        // CRITICAL: drain the async copy fence first — an in-flight copy would
        // write into the about-to-be-freed staging buffer.
        m_device->drainDenoiserCopy();
        destroyGpuDenoiserInterop();
        if (m_denoiserColorSharedStaging.buffer)  m_device->destroyBuffer(m_denoiserColorSharedStaging);
        if (m_denoiserAlbedoSharedStaging.buffer) m_device->destroyBuffer(m_denoiserAlbedoSharedStaging);
        if (m_denoiserNormalSharedStaging.buffer) m_device->destroyBuffer(m_denoiserNormalSharedStaging);
        m_denoiserColorSharedStaging = {};
        m_denoiserAlbedoSharedStaging = {};
        m_denoiserNormalSharedStaging = {};
        m_denoiserColorSharedAllocSize = 0;
        m_denoiserAlbedoSharedAllocSize = 0;
        m_denoiserNormalSharedAllocSize = 0;
        // The Win32 handles were either consumed by CUDA (ownership transferred
        // in importGpuDenoiserInterop) or destroyed along with the VkDeviceMemory.
        m_denoiserColorSharedHandle = nullptr;
        m_denoiserAlbedoSharedHandle = nullptr;
        m_denoiserNormalSharedHandle = nullptr;

        // Prefer half-float output to cut readback memory pressure by 50%.
        VkFormat outFmt = VK_FORMAT_R32G32B32A32_SFLOAT;
        VkFormatProperties fmtProps{};
        vkGetPhysicalDeviceFormatProperties(m_device->getPhysicalDevice(), VK_FORMAT_R16G16B16A16_SFLOAT, &fmtProps);
        const bool supports16fStorage =
            (fmtProps.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) &&
            (fmtProps.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT);
        if (supports16fStorage) {
            outFmt = VK_FORMAT_R16G16B16A16_SFLOAT;
        }
        m_outputImage = m_device->createImage2D(
            width, height, outFmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_varianceImage = m_device->createImage2D(
            width, height, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_denoiserColorImage = m_device->createImage2D(
            width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_denoiserAlbedoImage = m_device->createImage2D(
            width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_denoiserNormalImage = m_device->createImage2D(
            width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_denoiserPositionImage = m_device->createImage2D(
            width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        VulkanRT::BufferCreateInfo stagingInfo;
        const uint64_t bytesPerPixel = (outFmt == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8ull : 16ull;
        stagingInfo.size = (uint64_t)width * height * bytesPerPixel;
        stagingInfo.usage = VulkanRT::BufferUsage::TRANSFER_DST;
        stagingInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
        m_stagingBuffer = m_device->createBuffer(stagingInfo);

        // GPU-tonemap LDR target + two stagings (RGBA8 = 4 bytes/pixel; 1/4 of FP32, 1/2 of FP16).
        // Single shared LDR image is safe across the 2-slot ping-pong because both
        // submissions go to the same queue in order. Stagings are doubled so CPU can
        // memcpy slot N-1 while GPU writes slot N+1 next frame.
        m_tonemappedImage = m_device->createImage2D(
            width, height, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        VulkanRT::BufferCreateInfo tmStagingInfo;
        tmStagingInfo.size = (uint64_t)width * height * 4ull;
        tmStagingInfo.usage = VulkanRT::BufferUsage::TRANSFER_DST;
        tmStagingInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
        m_tonemappedStagings[0] = m_device->createBuffer(tmStagingInfo);
        m_tonemappedStagings[1] = m_device->createBuffer(tmStagingInfo);
        if (!m_tonemappedImage.image || !m_tonemappedStagings[0].buffer || !m_tonemappedStagings[1].buffer) {
            SCENE_LOG_WARN("[Vulkan] Tonemap LDR target allocation failed — using CPU tonemap fallback.");
            if (m_tonemappedImage.image) m_device->destroyImage(m_tonemappedImage);
            for (auto& s : m_tonemappedStagings) {
                if (s.buffer) m_device->destroyBuffer(s);
                s = {};
            }
            m_tonemappedImage = {};
        }
        // Refresh persistent tonemap descriptor set to point at the new image views.
        // updateTonemapDescriptors is a no-op when the pipeline isn't ready yet
        // (pipeline init runs later in the same call); it will be retried below.
        if (m_tonemappedImage.image) {
            m_device->updateTonemapDescriptors(m_outputImage, m_tonemappedImage);
        }
        m_tonemappedSlotInFlight[0] = false;
        m_tonemappedSlotInFlight[1] = false;
        m_tonemappedFrameSlot = 0;
        stagingInfo.size = (uint64_t)width * height * 4ull * sizeof(float);
        m_denoiserColorStagingBuffer = m_device->createBuffer(stagingInfo);
        m_denoiserAlbedoStagingBuffer = m_device->createBuffer(stagingInfo);
        m_denoiserNormalStagingBuffer = m_device->createBuffer(stagingInfo);
        m_denoiserPositionStagingBuffer = m_device->createBuffer(stagingInfo);

        // Exportable parallel stagings for the GPU-direct path. Failure is
        // non-fatal — getDenoiserFrameGPU will simply return false and the
        // host path continues to work.
        if (m_device->getCapabilities().supportsExternalMemoryWin32 && !m_gpuDenoiserDisabled) {
            VulkanRT::BufferCreateInfo sharedInfo;
            sharedInfo.size = stagingInfo.size;
            sharedInfo.usage = VulkanRT::BufferUsage::TRANSFER_DST;
            sharedInfo.location = VulkanRT::MemoryLocation::GPU_ONLY; // exported memory must be DEVICE_LOCAL
            m_denoiserColorSharedStaging  = m_device->createExportableBuffer(
                sharedInfo, &m_denoiserColorSharedHandle, &m_denoiserColorSharedAllocSize);
            m_denoiserAlbedoSharedStaging = m_device->createExportableBuffer(
                sharedInfo, &m_denoiserAlbedoSharedHandle, &m_denoiserAlbedoSharedAllocSize);
            m_denoiserNormalSharedStaging = m_device->createExportableBuffer(
                sharedInfo, &m_denoiserNormalSharedHandle, &m_denoiserNormalSharedAllocSize);
            const bool allOK = m_denoiserColorSharedStaging.buffer
                             && m_denoiserAlbedoSharedStaging.buffer
                             && m_denoiserNormalSharedStaging.buffer;
            if (!allOK) {
                SCENE_LOG_WARN("[Vulkan] Exportable denoiser staging allocation failed — GPU-direct OIDN disabled.");
                if (m_denoiserColorSharedStaging.buffer)  m_device->destroyBuffer(m_denoiserColorSharedStaging);
                if (m_denoiserAlbedoSharedStaging.buffer) m_device->destroyBuffer(m_denoiserAlbedoSharedStaging);
                if (m_denoiserNormalSharedStaging.buffer) m_device->destroyBuffer(m_denoiserNormalSharedStaging);
                m_denoiserColorSharedStaging = {};
                m_denoiserAlbedoSharedStaging = {};
                m_denoiserNormalSharedStaging = {};
                m_denoiserColorSharedHandle = nullptr;
                m_denoiserAlbedoSharedHandle = nullptr;
                m_denoiserNormalSharedHandle = nullptr;
                m_denoiserColorSharedAllocSize = 0;
                m_denoiserAlbedoSharedAllocSize = 0;
                m_denoiserNormalSharedAllocSize = 0;
                m_gpuDenoiserDisabled = true;
            }
        }

        const uint64_t output_bytes = (uint64_t)width * height * bytesPerPixelForFormat(outFmt);
        const uint64_t variance_bytes = (uint64_t)width * height * bytesPerPixelForFormat(VK_FORMAT_R32_SFLOAT);
        const uint64_t denoiser_bytes = (uint64_t)width * height * bytesPerPixelForFormat(VK_FORMAT_R32G32B32A32_SFLOAT) * 3ull;
        SCENE_LOG_INFO("[Perf] [Vulkan] render targets | output=" + std::to_string(output_bytes) +
                       " bytes | variance=" + std::to_string(variance_bytes) +
                       " bytes | denoiser=" + std::to_string(denoiser_bytes) + " bytes");

        if (!m_outputImage.image || !m_varianceImage.image || !m_stagingBuffer.buffer ||
            !m_denoiserColorImage.image || !m_denoiserAlbedoImage.image || !m_denoiserNormalImage.image ||
            !m_denoiserPositionImage.image ||
            !m_denoiserColorStagingBuffer.buffer || !m_denoiserAlbedoStagingBuffer.buffer || !m_denoiserNormalStagingBuffer.buffer ||
            !m_denoiserPositionStagingBuffer.buffer) {
            SCENE_LOG_ERROR("[Vulkan] Failed to allocate output/readback buffers for current resolution.");
            if (m_outputImage.image) m_device->destroyImage(m_outputImage);
            if (m_varianceImage.image) m_device->destroyImage(m_varianceImage);
            if (m_stagingBuffer.buffer) m_device->destroyBuffer(m_stagingBuffer);
            if (m_denoiserColorImage.image) m_device->destroyImage(m_denoiserColorImage);
            if (m_denoiserAlbedoImage.image) m_device->destroyImage(m_denoiserAlbedoImage);
            if (m_denoiserNormalImage.image) m_device->destroyImage(m_denoiserNormalImage);
            if (m_denoiserPositionImage.image) m_device->destroyImage(m_denoiserPositionImage);
            if (m_denoiserColorStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserColorStagingBuffer);
            if (m_denoiserAlbedoStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserAlbedoStagingBuffer);
            if (m_denoiserNormalStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserNormalStagingBuffer);
            if (m_denoiserPositionStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserPositionStagingBuffer);
            if (m_tonemappedImage.image) m_device->destroyImage(m_tonemappedImage);
            for (auto& s : m_tonemappedStagings) {
                if (s.buffer) m_device->destroyBuffer(s);
                s = {};
            }
            m_outputImage = {};
            m_varianceImage = {};
            m_stagingBuffer = {};
            m_denoiserColorImage = {};
            m_denoiserAlbedoImage = {};
            m_denoiserNormalImage = {};
            m_denoiserPositionImage = {};
            m_denoiserColorStagingBuffer = {};
            m_denoiserAlbedoStagingBuffer = {};
            m_denoiserNormalStagingBuffer = {};
            m_denoiserPositionStagingBuffer = {};
            m_tonemappedImage = {};
            return;
        }
        
        m_imageWidth = width;
        m_imageHeight = height;
        resetAccumulation();
    }

    // 2. Build Pipeline/Resources lazy
    if (!this->m_testInitialized) {
        this->m_testInitialized = true;

        using namespace VulkanRT;
        
        // Load RT Shaders
        std::vector<std::uint32_t> raygenSPV = loadSPV(shaderDir + "/raygen.spv");
        std::vector<std::uint32_t> missSPV = loadSPV(shaderDir + "/miss.spv");
        std::vector<std::uint32_t> chitSPV = loadSPV(shaderDir + "/closesthit.spv");
        std::vector<std::uint32_t> ahitSPV;
        if (std::filesystem::exists(shaderDir + "/shadow_anyhit.spv")) ahitSPV = loadSPV(shaderDir + "/shadow_anyhit.spv");
        
        // Load Volume Shaders (optional — gracefully skipped if not compiled)
        std::vector<std::uint32_t> volChitSPV;
        std::vector<std::uint32_t> volIntSPV;
        if (std::filesystem::exists(shaderDir + "/volume_closesthit.spv") &&
            std::filesystem::exists(shaderDir + "/volume_intersection.spv")) {
            volChitSPV = loadSPV(shaderDir + "/volume_closesthit.spv");
            volIntSPV  = loadSPV(shaderDir + "/volume_intersection.spv");
            SCENE_LOG_INFO("[Vulkan] Volume shaders loaded successfully.");
        } else {
            SCENE_LOG_INFO("[Vulkan] Volume shaders not found — volume rendering disabled.");
        }

        // Load Hair Shaders
        std::vector<std::uint32_t> hairChitSPV;
        std::vector<std::uint32_t> hairIntSPV;
        std::vector<std::uint32_t> hairAhitSPV;
        if (std::filesystem::exists(shaderDir + "/hair_closesthit.spv") &&
            std::filesystem::exists(shaderDir + "/hair_intersection.spv")) {
            hairChitSPV = loadSPV(shaderDir + "/hair_closesthit.spv");
            hairIntSPV  = loadSPV(shaderDir + "/hair_intersection.spv");
            if (std::filesystem::exists(shaderDir + "/hair_shadow_anyhit.spv")) {
                hairAhitSPV = loadSPV(shaderDir + "/hair_shadow_anyhit.spv");
            }
            SCENE_LOG_INFO("[Vulkan] Hair shaders loaded successfully.");
        } else {
            SCENE_LOG_INFO("[Vulkan] Hair shaders not found — hair rendering disabled.");
        }

        // Load Foam point-sphere shaders (optional — N→1 instance foam path)
        std::vector<std::uint32_t> sphereChitSPV;
        std::vector<std::uint32_t> sphereIntSPV;
        if (std::filesystem::exists(shaderDir + "/sphere_closesthit.spv") &&
            std::filesystem::exists(shaderDir + "/sphere_intersection.spv")) {
            sphereChitSPV = loadSPV(shaderDir + "/sphere_closesthit.spv");
            sphereIntSPV  = loadSPV(shaderDir + "/sphere_intersection.spv");
            SCENE_LOG_INFO("[Vulkan] Foam sphere shaders loaded successfully.");
        } else {
            SCENE_LOG_INFO("[Vulkan] Foam sphere shaders not found — foam falls back to instanced spheres.");
        }

        // Load Shadow Miss Shader (optional — enables shadow rays from hit shaders)
        std::vector<std::uint32_t> shadowMissSPV;
        if (std::filesystem::exists(shaderDir + "/shadow_miss.spv")) {
            shadowMissSPV = loadSPV(shaderDir + "/shadow_miss.spv");
            SCENE_LOG_INFO("[Vulkan] Shadow miss shader loaded successfully.");
        } else {
            SCENE_LOG_INFO("[Vulkan] Shadow miss shader not found — shadow rays will use primary miss.");
        }

        // Load Photon Caustic Raygen (optional — Faz 2 photon caustic pass)
        std::vector<std::uint32_t> photonRgenSPV;
        if (std::filesystem::exists(shaderDir + "/photon.spv")) {
            photonRgenSPV = loadSPV(shaderDir + "/photon.spv");
            SCENE_LOG_INFO("[Vulkan] Photon caustic raygen loaded successfully.");
        } else {
            SCENE_LOG_INFO("[Vulkan] photon.spv not found — photon caustics disabled.");
        }

        // Load Skinning Compute Shader
        if (std::filesystem::exists(shaderDir + "/skinning.spv")) {
            std::vector<std::uint32_t> skinningSPV = loadSPV(shaderDir + "/skinning.spv");
            if (m_device->createSkinningPipeline(skinningSPV)) {
                SCENE_LOG_INFO("[Vulkan] Skinning compute pipeline created successfully.");
            } else {
                SCENE_LOG_ERROR("[Vulkan] Failed to create Skinning compute pipeline.");
            }
        }

        // Load Sculpt Compute Shader (optional)
        if (std::filesystem::exists(shaderDir + "/sculpt.spv")) {
            std::vector<std::uint32_t> sculptSPV = loadSPV(shaderDir + "/sculpt.spv");
            if (m_device->createSculptPipeline(sculptSPV)) {
                SCENE_LOG_INFO("[Vulkan] Sculpt compute pipeline created successfully.");
            } else {
                SCENE_LOG_ERROR("[Vulkan] Failed to create Sculpt compute pipeline.");
            }
        }

        // Load Tonemap Compute Shader (optional — when present, render path skips the
        // per-frame CPU Reinhard+sRGB loop and reads back 1/4 the bytes).
        if (std::filesystem::exists(shaderDir + "/tonemap.spv")) {
            std::vector<std::uint32_t> tonemapSPV = loadSPV(shaderDir + "/tonemap.spv");
            if (m_device->createTonemapPipeline(tonemapSPV)) {
                SCENE_LOG_INFO("[Vulkan] Tonemap compute pipeline created — GPU tonemap path enabled.");
                // Bind the persistent tonemap descriptor set to the current HDR+LDR
                // images. The resize block already created them; we couldn't write the
                // set there because the pipeline wasn't ready yet.
                if (m_tonemappedImage.image && m_outputImage.image) {
                    m_device->updateTonemapDescriptors(m_outputImage, m_tonemappedImage);
                }
            } else {
                SCENE_LOG_ERROR("[Vulkan] Failed to create Tonemap compute pipeline; falling back to CPU tonemap.");
            }
        } else {
            SCENE_LOG_INFO("[Vulkan] tonemap.spv not found — using CPU tonemap fallback.");
        }

        // Load Stylize Compute Shader (optional; CPU stylize remains fallback). Descriptors
        // are bound lazily in VulkanBackendAdapter::applyStylizeGPU once the per-frame color
        // SSBO + params buffer + AOV image views exist.
        if (std::filesystem::exists(shaderDir + "/stylize.spv")) {
            std::vector<std::uint32_t> stylizeSPV = loadSPV(shaderDir + "/stylize.spv");
            if (m_device->createStylizePipeline(stylizeSPV)) {
                SCENE_LOG_INFO("[Vulkan] Stylize compute pipeline created — GPU stylize path enabled.");
            } else {
                SCENE_LOG_ERROR("[Vulkan] Failed to create Stylize compute pipeline; falling back to CPU stylize.");
            }
        } else {
            SCENE_LOG_INFO("[Vulkan] stylize.spv not found — using CPU stylize fallback.");
        }

        // Load Atmosphere LUT Compute Shader (optional; CPU LUT path remains fallback).
        if (std::filesystem::exists(shaderDir + "/atmosphere_lut.spv")) {
            std::vector<std::uint32_t> atmosphereSPV = loadSPV(shaderDir + "/atmosphere_lut.spv");
            if (m_device->createAtmosphereLUTPipeline(atmosphereSPV)) {
                SCENE_LOG_INFO("[Vulkan] Atmosphere LUT compute pipeline created — GPU Nishita LUT path enabled.");
            } else {
                SCENE_LOG_ERROR("[Vulkan] Failed to create Atmosphere LUT compute pipeline; falling back to CPU LUT upload.");
            }
        } else {
            SCENE_LOG_INFO("[Vulkan] atmosphere_lut.spv not found — using CPU LUT upload fallback.");
        }

        // Only create the heavy RT pipeline when the viewport mode requires it.
        if (m_viewportMode == ViewportMode::Rendered) {
            if (!m_device->createRTPipeline(raygenSPV, missSPV, chitSPV, ahitSPV,
                    volChitSPV, volIntSPV, hairChitSPV, hairIntSPV, shadowMissSPV, hairAhitSPV,
                    sphereChitSPV, sphereIntSPV, photonRgenSPV)) {
                SCENE_LOG_ERROR("[Vulkan] Failed to create RT Pipeline.");
                return;
            }
        } else {
            SCENE_LOG_INFO("[Vulkan] Skipping RT pipeline creation (not in Rendered mode)");
        }

        // [FIX] Hair SBT offset correction after pipeline creation.
        // When hair was uploaded during backend switch, the pipeline was not yet
        // created so getHairSbtOffset() returned a stale value (default
        // m_hasVolumeShaders=false). If the pipeline now reports volumes are
        // present, the hair TLAS instances have the wrong sbtRecordOffset.
        // Rebuild the TLAS with the corrected offset to ensure the correct hit
        // shader is dispatched for hair geometry.
        if (m_device->m_hasHairShaders && !m_hairVkInstances.empty()) {
            uint32_t correctOffset = m_device->getHairSbtOffset();
            bool needsTlasRebuild = false;
            for (auto& hvi : m_hairVkInstances) {
                if (hvi.sbtRecordOffset != correctOffset) {
                    hvi.sbtRecordOffset = correctOffset;
                    needsTlasRebuild = true;
                }
            }
            if (needsTlasRebuild) {
                std::vector<VulkanRT::TLASInstance> allInstances = m_vkInstances;
                for (const auto& h : m_hairVkInstances) allInstances.push_back(h);
                if (!allInstances.empty()) {
                    VulkanRT::TLASCreateInfo tlasInfo;
                    tlasInfo.instances   = allInstances;
                    tlasInfo.allowUpdate = false;
                    m_device->createTLAS(tlasInfo);
                    SCENE_LOG_INFO("[Vulkan] Hair SBT offset corrected after pipeline init (offset="
                        + std::to_string(correctOffset) + ")");
                }
            }
        }
    }

    // Safety: ensure pipeline is actually built and TLAS exist before proceeding to trace.
    // Preserve the last valid host-side frame while RT resources catch up after
    // camera/backend changes.
    if (!m_device->isRTReady() || !m_device->hasTLAS()) {
        // [FIX] Render world/sky background so the user sees the environment
        // instead of stale Solid-mode grey pixels while the RT pipeline builds.
        if (m_hasPresentedRenderedFrame) {
            rePresentCachedFrame();
        } else {
            presentBackgroundOnly();
        }
        return;
    }
    if (m_forceClearOnNextPresent && !allowImmediateHostClear) {
        m_forceClearOnNextPresent = false;
    }

    // Push Constants
    // Push Constants - MUST MATCH raygen.rgen. The GLSL shader will have:
    /*
    layout(push_constant) uniform PushConstants {
        vec4 origin;        // Camera position
        vec4 lowerLeft;     // Lower-left corner of the image plane
        vec4 horizontal;    // Horizontal extent of image plane
        vec4 vertical;      // Vertical extent of image plane
        uint frameCount;    // For accumulation
    } camera;

    // New bindings for Materials and Lights (though not used in raygen, layout must match)
    layout(set = 0, binding = 2) buffer MaterialBuffer { vec4 m[]; } materials;
    layout(set = 0, binding = 3) buffer LightBuffer { vec4 l[]; } lights;
    */
    struct CameraPushConstants {
        float origin[4];         // [x,y,z, dummy]
        float lowerLeft[4];      // Lower-left corner of image plane
        float horizontal[4];     // Horizontal extent
        float vertical[4];       // Vertical extent
        uint32_t frameCount;     // Accumulation counter
        uint32_t minSamples;     // Minimum samples before adaptive skipping
        uint32_t lightCount;     // Number of active lights
        float varianceThreshold; // Adaptive sampling threshold
        uint32_t maxSamples;     // Target sample limit
        float exposureFactor;    // Physical exposure multiplier

        // Extended Pro Features
        float aperture;
        float focusDistance;
        float distortion;
        uint32_t bladeCount;

        uint32_t caEnabled;
        float caAmount;
        float caRScale;
        float caBScale;

        uint32_t vignetteEnabled;
        float vignetteAmount;
        float vignetteFalloff;
        float pad0;

        uint32_t shakeEnabled;
        float shakeOffsetX;
        float shakeOffsetY;
        float shakeOffsetZ;

        float shakeRotX;
        float shakeRotY;
        float shakeRotZ;
        float waterTime;   // Real wall-clock time in seconds for water animation
        uint32_t maxBounces; // UI'dan gelen toplam bounce limiti
        uint32_t diffuseBounces;
        uint32_t transmissionBounces;
    };

    CameraPushConstants pushConst{};
    
    // Calculate camera vectors from m_camera
    float fov = this->m_camera.fov > 1.0f ? this->m_camera.fov : 60.0f;
    float aspect = (float)width / (float)height;
    float h_half = tanf(fov * 0.5f * 3.14159f / 180.0f);
    float viewport_height = 2.0f * h_half;
    float viewport_width = aspect * viewport_height;

    // Use stored camera vectors or defaults
    Vec3 lookFrom = this->m_camera.origin;
    Vec3 lookAt = this->m_camera.lookAt;
    Vec3 vup = this->m_camera.up;

    // Safety fallback for empty/default camera
    if ((lookFrom - lookAt).length() < 0.0001f) {
        lookFrom = Vec3(0, 0, 5);
        lookAt = Vec3(0, 0, 0);
        vup = Vec3(0, 1, 0);
    }

    Vec3 camW = (lookFrom - lookAt).normalize();
    Vec3 camU = vup.cross(camW).normalize();
    Vec3 camV = camW.cross(camU);

    // [FOCUS FIX] Vectors must be scaled by focusDistance for correct DOF projection
    float focus_dist = this->m_camera.focusDistance > 0.001f ? this->m_camera.focusDistance : 1.0f;
    Vec3 horizontal = camU * viewport_width * focus_dist;
    Vec3 vertical = camV * viewport_height * focus_dist;
    Vec3 lower_left_corner = lookFrom - horizontal * 0.5f - vertical * 0.5f - camW * focus_dist;

    pushConst.origin[0] = lookFrom.x; pushConst.origin[1] = lookFrom.y; pushConst.origin[2] = lookFrom.z; pushConst.origin[3] = 1.0f;
    pushConst.horizontal[0] = horizontal.x; pushConst.horizontal[1] = horizontal.y; pushConst.horizontal[2] = horizontal.z; pushConst.horizontal[3] = 0.0f;
    pushConst.vertical[0] = vertical.x; pushConst.vertical[1] = vertical.y; pushConst.vertical[2] = vertical.z; pushConst.vertical[3] = 0.0f;
    pushConst.lowerLeft[0] = lower_left_corner.x; pushConst.lowerLeft[1] = lower_left_corner.y; pushConst.lowerLeft[2] = lower_left_corner.z; pushConst.lowerLeft[3] = 0.0f;
    pushConst.frameCount = this->m_currentSamples;
    pushConst.minSamples = m_useAdaptiveSampling ? static_cast<uint32_t>(std::max(1, m_minSamples)) : 0u;
    pushConst.lightCount = (uint32_t)m_cachedLights.size();
    pushConst.varianceThreshold = m_useAdaptiveSampling ? m_varianceThreshold : 0.0f;
    pushConst.maxSamples = m_targetSamples;
    pushConst.exposureFactor = this->m_camera.exposureFactor;

    // Population of extended features
    pushConst.aperture = this->m_camera.aperture;
    pushConst.focusDistance = this->m_camera.focusDistance;
    pushConst.distortion = this->m_camera.distortion;
    pushConst.bladeCount = (uint32_t)this->m_camera.blade_count;

    pushConst.caEnabled = this->m_camera.chromaticAberrationEnabled ? 1 : 0;
    pushConst.caAmount = this->m_camera.chromatic_aberration;
    pushConst.caRScale = this->m_camera.chromatic_aberration_r;
    pushConst.caBScale = this->m_camera.chromatic_aberration_b;

    pushConst.vignetteEnabled = this->m_camera.vignettingEnabled ? 1 : 0;
    pushConst.vignetteAmount = this->m_camera.vignetting_amount;
    pushConst.vignetteFalloff = this->m_camera.vignetting_falloff;
    // pad0 repurposed as active volume count — closesthit.rchit reads int(cam.pad0) to loop over
    // the VolumeBuffer SSBO (binding 9) for volumetric shadow transmittance computation.
    pushConst.pad0 = float(m_device->m_volumeCount);

    // Keep water time on the same deterministic timeline contract as CPU/OptiX.
    // Falling back to wall-clock time makes Vulkan RT water drift out of phase
    // when the timeline is paused or when switching devices on a static frame.
    pushConst.waterTime = m_currentTime;
    pushConst.maxBounces = (uint32_t)std::max(1, m_maxBounces); // m_maxBounces her zaman UI'dan gelir
    pushConst.diffuseBounces = (uint32_t)std::clamp(m_diffuseBounces, 1, m_maxBounces);
    pushConst.transmissionBounces = (uint32_t)std::clamp(m_transmissionBounces, 1, m_maxBounces);

    pushConst.shakeEnabled = this->m_camera.shake_enabled ? 1 : 0;
    if (pushConst.shakeEnabled) {
        float time = (float)SDL_GetTicks() / 1000.0f;
        float freq = this->m_camera.shake_frequency;
        float skill_mult = 1.0f;
        switch (this->m_camera.operator_skill) {
            case 0: skill_mult = 1.0f; break;
            case 1: skill_mult = 0.6f; break;
            case 2: skill_mult = 0.25f; break;
            case 3: skill_mult = 0.1f; break;
            default: break;
        }

        float intensity = this->m_camera.shake_intensity * skill_mult;
        if (this->m_camera.ibis_enabled) {
            intensity /= powf(2.0f, this->m_camera.ibis_effectiveness);
        }

        pushConst.shakeOffsetX = sinf(time * freq * 1.0f) * this->m_camera.handheld_sway_amplitude * intensity;
        pushConst.shakeOffsetY =
            sinf(time * freq * 1.3f + 1.5f) * this->m_camera.handheld_sway_amplitude * intensity +
            sinf(time * this->m_camera.breathing_frequency * 6.28f) * this->m_camera.breathing_amplitude * intensity;
        pushConst.shakeOffsetZ = sinf(time * freq * 0.7f + 3.0f) * this->m_camera.handheld_sway_amplitude * intensity * 0.3f;
        
        pushConst.shakeRotX = sinf(time * freq * 1.1f) * 0.003f * intensity;
        pushConst.shakeRotY = sinf(time * freq * 0.9f + 1.0f) * 0.003f * intensity;
        pushConst.shakeRotZ = sinf(time * freq * 0.5f + 2.0f) * 0.001f * intensity;

        if (this->m_camera.enable_focus_drift && this->m_camera.focus_drift_amount > 0.0f) {
            float base_intensity = this->m_camera.shake_intensity * skill_mult;
            float focus_wave = sinf(time * freq * 0.4f + 2.5f);
            float distance_scale = 1.0f / (1.0f + this->m_camera.focusDistance * 0.1f);
            float aperture_scale = this->m_camera.aperture * 10.0f;
            float focus_variation =
                focus_wave *
                this->m_camera.focus_drift_amount *
                base_intensity *
                distance_scale *
                aperture_scale;
            pushConst.focusDistance = this->m_camera.focusDistance + focus_variation;
        }
    } else {
        pushConst.shakeOffsetX = 0.0f;
        pushConst.shakeOffsetY = 0.0f;
        pushConst.shakeOffsetZ = 0.0f;
        pushConst.shakeRotX = 0.0f;
        pushConst.shakeRotY = 0.0f;
        pushConst.shakeRotZ = 0.0f;
    }

    // Detect camera movement/rotation by hashing camera push-constant vectors.
    uint64_t camHash = 1469598103934665603ULL; // FNV-1a 64-bit offset
    auto mix32 = [&](uint32_t v){ camHash ^= v; camHash *= 1099511628211ULL; };
    auto mixFloat4 = [&](const float f[4]){
        for (int i = 0; i < 4; ++i) { uint32_t bits; memcpy(&bits, &f[i], sizeof(uint32_t)); mix32(bits); }
    };
    mixFloat4(pushConst.origin);
    mixFloat4(pushConst.lowerLeft);
    mixFloat4(pushConst.horizontal);
    mixFloat4(pushConst.vertical);
    // [HASH FIX] Include exposure and lens parameters to reset accumulation when they change
    mix32(*(uint32_t*)&pushConst.exposureFactor);
    mix32(*(uint32_t*)&pushConst.aperture);
    mix32(*(uint32_t*)&pushConst.focusDistance);
    mix32(*(uint32_t*)&pushConst.distortion);
    mix32(pushConst.caEnabled);
    mix32(pushConst.vignetteEnabled);
    mix32(pushConst.shakeEnabled);
    mix32(*(uint32_t*)&pushConst.shakeOffsetX);
    mix32(*(uint32_t*)&pushConst.shakeOffsetY);
    mix32(*(uint32_t*)&pushConst.shakeOffsetZ);
    mix32(*(uint32_t*)&pushConst.shakeRotX);
    mix32(*(uint32_t*)&pushConst.shakeRotY);
    mix32(*(uint32_t*)&pushConst.shakeRotZ);
    mix32(*(uint32_t*)&pushConst.focusDistance);

    if (camHash != this->m_lastCameraHash) {
        this->m_lastCameraHash = camHash;
        // [PERF] Only reset if not already reset — setCamera() from Main.cpp
        // may have already called resetAccumulation() this frame.
        if (m_currentSamples > 0) {
            resetAccumulation();
        }
    }

    // Additionally detect significant view direction changes using previous view dir
    Vec3 viewDir = (lookAt - lookFrom).normalize();
    if (this->m_hasPrevView) {
        float dotv = std::clamp(this->m_prevViewDir.dot(viewDir), -1.0f, 1.0f);
        float ang = acos(dotv);
        if (ang > 0.01f && m_currentSamples > 0) { // ~0.57 degrees
            resetAccumulation();
        }
    }
    // If camera is looking almost straight down, force a full reset to avoid
    // horizon/undersampling ghosting artifacts when pitch approaches -Y.
    if (viewDir.y < -0.999f && m_currentSamples > 0) {
        resetAccumulation();
    }
    this->m_prevViewDir = viewDir;
    this->m_hasPrevView = true;

    m_device->setPushConstants(&pushConst, sizeof(CameraPushConstants));

    // 3. Trace Rays
        if (m_device->isRTReady() && m_device->hasTLAS()) {
        // Explicitly clear image on frame 0 to prevent ghosting or stale adaptive data
        // [PERF] Skip if resetAccumulation() already cleared them this cycle
        if (m_currentSamples == 0 && !m_imagesCleared) {
            std::vector<VulkanRT::VulkanDevice::ImageClearRequest> clears;
            if (m_outputImage.image)        clears.push_back({&m_outputImage, 0,0,0,0});
            if (m_varianceImage.image)      clears.push_back({&m_varianceImage, 0,0,0,0});
            if (m_denoiserColorImage.image) clears.push_back({&m_denoiserColorImage, 0,0,0,0});
            if (m_denoiserAlbedoImage.image) clears.push_back({&m_denoiserAlbedoImage, 0,0,0,0});
            if (m_denoiserNormalImage.image) clears.push_back({&m_denoiserNormalImage, 0.5f,0.5f,0.5f,0});
            if (m_denoiserPositionImage.image) clears.push_back({&m_denoiserPositionImage, 0,0,0,0});
            m_device->clearImages(clears);
        }
        m_imagesCleared = false; // consumed — next reset can clear again

        m_device->bindRTDescriptors(
            m_outputImage,
            &m_denoiserColorImage,
            &m_denoiserAlbedoImage,
            &m_denoiserNormalImage,
            &m_varianceImage,
            &m_denoiserPositionImage);
        m_device->setPushConstants(&pushConst, sizeof(CameraPushConstants));

        // ── Photon caustic pass (Faz 2 / Dilim 1) ────────────────────────────
        // Writes the grid header + arms the pass; the clear + photon trace is
        // recorded INSIDE this frame's camera command buffer (recordPhotonPass)
        // so it cannot race the async ping-pong camera reads.
        // Volumetric shafts are INDEPENDENT of surface caustics: either feature
        // arms the photon pass; ph.debugMode below decides whether the surface
        // grid is gathered.
        if ((m_causticsEnabled || m_causticsVolumetric || m_causticsVolDebug) &&
            m_device->hasPhotonPipeline() && !m_cachedLights.empty()) {
            VulkanRT::VulkanDevice::PhotonGridHeader ph{};
            ph.originCell[0] = 0.0f; ph.originCell[1] = 0.0f; ph.originCell[2] = 0.0f;
            // Emission target: the union AABB of transmissive meshes — camera-
            // INDEPENDENT, so the caustic converges to the same distribution
            // after every reset. (Aiming at the camera focus made the pattern
            // vanish whenever the view changed.) Fallback: camera focus region.
            Vec3 tgt = this->m_camera.lookAt;
            float rad = std::max(3.0f, this->m_camera.focusDistance * 0.5f);
            if (!m_blasMaterialBounds.empty() && !m_vkInstances.empty()) {
                // Per-BLAS, per-material LOCAL bounds. Caster status is decided
                // HERE, per frame, against the live MaterialManager — project
                // loads can upload BLASes before their materials resolve, and
                // material edits after upload must still move the target.
                // The 8 box corners go through the live TLAS instance transform
                // (solo/loaded meshes upload local P_orig verts), then union in
                // WORLD space.
                float mn[3] = { 1e30f, 1e30f, 1e30f };
                float mx[3] = { -1e30f, -1e30f, -1e30f };
                bool any = false;
                for (const auto& vi : m_vkInstances) {
                    auto bit = m_blasMaterialBounds.find(vi.blasIndex);
                    if (bit == m_blasMaterialBounds.end()) continue;
                    for (const auto& entry : bit->second) {
                        if (!materialIsCausticCaster(entry.first)) continue;
                        const CausticBounds& b = entry.second;
                        if (b.minX > b.maxX) continue; // inverted = id-only entry (device-resident mesh)
                        for (int corner = 0; corner < 8; ++corner) {
                            Vec3 p((corner & 1) ? b.maxX : b.minX,
                                   (corner & 2) ? b.maxY : b.minY,
                                   (corner & 4) ? b.maxZ : b.minZ);
                            Vec3 w = vi.transform.transform_point(p);
                            mn[0] = std::min(mn[0], w.x); mx[0] = std::max(mx[0], w.x);
                            mn[1] = std::min(mn[1], w.y); mx[1] = std::max(mx[1], w.y);
                            mn[2] = std::min(mn[2], w.z); mx[2] = std::max(mx[2], w.z);
                        }
                        any = true;
                    }
                }
                if (any) {
                    tgt = Vec3((mn[0] + mx[0]) * 0.5f, (mn[1] + mx[1]) * 0.5f, (mn[2] + mx[2]) * 0.5f);
                    const float dx = mx[0] - mn[0], dy = mx[1] - mn[1], dz = mx[2] - mn[2];
                    // Enclosing sphere + PROPORTIONAL margin. A fixed +0.25
                    // margin dominated tiny scenes (a 5 cm object got a ~0.3
                    // target radius), which also drove the auto cell size —
                    // the grid stayed far too coarse for zoomed-in objects.
                    const float halfDiag = 0.5f * std::sqrt(dx * dx + dy * dy + dz * dz);
                    rad = halfDiag * 1.15f + 0.005f;
                }
            }
            ph.emitCenter[0] = tgt.x; ph.emitCenter[1] = tgt.y; ph.emitCenter[2] = tgt.z;
            ph.emitCenter[3] = rad;
            // Scale-aware cell size: the UI value is an UPPER BOUND. A small
            // (zoomed-in) glass object with a fixed 0.05-unit cell quantizes its
            // whole caustic into a few giant blocks and the gathered energy
            // smears into nothing — tie the cell to the target radius so the
            // pattern always spans a useful number of cells. rad only changes on
            // caster edits, which also reset accumulation (grid is cleared), so
            // splat/read never disagree on the cell size mid-accumulation.
            ph.originCell[3] = std::min(std::max(0.001f, m_causticsCellSize),
                                        std::max(rad * (1.0f / 48.0f), 1e-4f));
            ph.photonCount   = (uint32_t)std::max(1024, m_causticsPhotons);
            ph.frameSeed     = m_currentSamples;
            // mode 1 = surface debug viz, 2 = gather into shading (Dilim 2),
            // 3 = VOLUME grid debug march (Faz 2V / Dilim V1),
            // 0 = surface gather off (volumetric-only mode).
            ph.debugMode     = m_causticsVolDebug ? 3u
                             : (m_causticsEnabled ? (m_causticsDebug ? 1u : 2u) : 0u);
            ph.lightIndex    = 0u;
            ph.lightCountReal = (uint32_t)m_cachedLights.size();
            ph.energyScale   = m_causticsEnergy;
            ph.debugExposure = 1.0f;

            // Volume grid header (Faz 2V): coarser cells (3x the surface grid),
            // debugMode 1 arms the photon-side deposit, energyScale carries σs.
            VulkanRT::VulkanDevice::PhotonGridHeader pv{};
            pv.originCell[0] = 0.0f; pv.originCell[1] = 0.0f; pv.originCell[2] = 0.0f;
            pv.originCell[3] = ph.originCell[3] * 3.0f;
            pv.emitCenter[0] = ph.emitCenter[0]; pv.emitCenter[1] = ph.emitCenter[1];
            pv.emitCenter[2] = ph.emitCenter[2]; pv.emitCenter[3] = ph.emitCenter[3];
            pv.photonCount   = ph.photonCount;
            pv.frameSeed     = ph.frameSeed;
            pv.debugMode     = (m_causticsVolumetric || m_causticsVolDebug) ? 1u : 0u;
            // lightIndex is unused by the volume grid — bit 0 repurposed as the
            // direct-shaft flag (photon.rgen deposits the light->glass leg too).
            pv.lightIndex    = m_causticsVolDirect ? 1u : 0u;
            pv.lightCountReal = ph.lightCountReal;
            pv.energyScale   = m_causticsVolStrength;   // σs knob
            // debugExposure doubles as the dust-noise amount OUTSIDE vol-debug:
            // raygen decodes noise = clamp(debugExposure - 1, 0, 1).
            pv.debugExposure = m_causticsVolDebug ? 1.0f : (1.0f + m_causticsVolNoise);

            // Grids accumulate across frames; clear only on accumulation reset.
            m_device->schedulePhotonPass(ph, pv, m_currentSamples == 0);
        } else if (m_device->hasPhotonPipeline()) {
            // Caustics just turned off (or no lights): zero the header mode so
            // raygen stops gathering from the stale grid.
            m_device->disablePhotonGrid();
        }

        // Prefer fused GPU path: trace + GPU tonemap + small LDR readback in one cmd buffer.
        // Falls back to legacy HDR readback + CPU tonemap loop when the tonemap pipeline or
        // LDR target are unavailable.
        const bool useGpuTonemap = m_device->hasTonemapPipeline()
                                && m_tonemappedImage.image
                                && m_tonemappedStagings[0].buffer
                                && m_tonemappedStagings[1].buffer;

        // Aşama 2 ping-pong:
        //   submitSlot = slot we submit to this frame (GPU writes here).
        //   consumeSlot = slot whose previously-submitted work we consume now (CPU reads).
        // On the first frame after a reset, no slot is in-flight to consume — we submit
        // and skip the consume path (display falls back to cached/background).
        const uint32_t submitSlot = m_tonemappedFrameSlot;
        const uint32_t consumeSlot = (submitSlot + 1u) % VulkanRT::VulkanDevice::kFrameSlotCount;

        std::vector<uint32_t>* framebuffer = static_cast<std::vector<uint32_t>*>(fb);
        if (framebuffer->size() != (size_t)(width * height)) {
            framebuffer->resize(width * height);
        }

        if (useGpuTonemap) {
            // 1. Submit current frame's work asynchronously (no host wait).
            const bool submitted = m_device->submitTraceTonemapAsync(submitSlot, width, height,
                m_outputImage, m_tonemappedImage, m_tonemappedStagings[submitSlot]);
            if (submitted) {
                m_tonemappedSlotInFlight[submitSlot] = true;
            }

            // 2. Consume the previously-submitted slot's staging if any. This is where
            //    overlap happens: while CPU memcpy + SDL update run, GPU is already
            //    chewing on submitSlot's command buffer.
            const bool canConsume = m_tonemappedSlotInFlight[consumeSlot];
            bool consumed = false;
            auto consumeTonemappedSlot = [&](uint32_t slot) {
                if (!m_device->waitFrameSlot(slot)) return;
                const size_t totalBytes = (size_t)width * (size_t)height * 4;
                m_device->downloadBuffer(m_tonemappedStagings[slot], framebuffer->data(), totalBytes);
                if (s) {
                    SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
                    if (outSurf->pixels && outSurf->w == (int)width && outSurf->h == (int)height) {
                        std::memcpy(outSurf->pixels, framebuffer->data(), totalBytes);
                    }
                }
                if (tex) SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, framebuffer->data(), width * 4);
                m_tonemappedSlotInFlight[slot] = false;
                consumed = true;
            };
            if (canConsume) {
                consumeTonemappedSlot(consumeSlot);
            }

            // 3. Advance ping-pong index for next call.
            m_tonemappedFrameSlot = consumeSlot;

            if (!submitted && !consumed) {
                // First-call seed failed or both slots empty — present background.
                if (m_hasPresentedRenderedFrame) {
                    rePresentCachedFrame();
                } else {
                    presentBackgroundOnly();
                }
                return;
            }
            if (!consumed) {
                // First successful submit but nothing to display yet — keep last frame.
                if (m_hasPresentedRenderedFrame) {
                    rePresentCachedFrame();
                } else {
                    presentBackgroundOnly();
                }
                this->m_currentSamples++;
                m_hasPresentedRenderedFrame = true;
                return;
            }
            // Fast path complete — staging consumed, surface/texture updated. Skip the
            // legacy HDR processing chain entirely.
            this->m_currentSamples++;
            m_hasPresentedRenderedFrame = true;
            return;
        }

        // Legacy synchronous path — kept for the rare case the tonemap pipeline
        // is unavailable (shader load failed, alloc failed, etc).
        bool traceOK = m_device->traceRaysAndReadback(width, height, m_outputImage, m_stagingBuffer);
        if (!traceOK) {
            if (m_hasPresentedRenderedFrame) {
                rePresentCachedFrame();
            } else {
                presentBackgroundOnly();
            }
            return;
        }

        // If the output image is float/half-float RGBA, download HDR and tonemap on CPU
        if (m_outputImage.format == VK_FORMAT_R32G32B32A32_SFLOAT ||
            m_outputImage.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
            m_hdrPixels.resize((size_t)width * (size_t)height * 4);
            if (m_outputImage.format == VK_FORMAT_R32G32B32A32_SFLOAT) {
                m_device->downloadBuffer(m_stagingBuffer, m_hdrPixels.data(), (uint64_t)width * height * 4 * sizeof(float));
            } else {
                m_halfPixels.resize((size_t)width * (size_t)height * 4);
                m_device->downloadBuffer(m_stagingBuffer, m_halfPixels.data(), (uint64_t)width * height * 4 * sizeof(uint16_t));
                for (size_t i = 0; i < m_halfPixels.size(); ++i) {
                    m_hdrPixels[i] = halfToFloat(m_halfPixels[i]);
                }
            }

            // Convert HDR floats -> 8-bit sRGB packed pixels
            // [PERF] Single-pass: tonemap into framebuffer, then bulk-copy to surface.
            // Eliminates per-pixel SDL_MapRGB on surface (was double-write per pixel).
            static SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA32);
            const size_t totalPixels = (size_t)width * (size_t)height;
            uint32_t* fbData = framebuffer->data();
            const float* hdrData = m_hdrPixels.data();

            for (size_t idx = 0; idx < totalPixels; ++idx) {
                float r = hdrData[idx * 4 + 0];
                float g = hdrData[idx * 4 + 1];
                float b = hdrData[idx * 4 + 2];

                // sanitize NaN/Inf, clamp negatives
                auto sanitize = [](float v) -> float {
                    if (std::isnan(v)) return 0.0f;
                    if (std::isinf(v)) return (v > 0.0f) ? 65504.0f : 0.0f;
                    return std::max(v, 0.0f);
                };
                float rr = sanitize(r);
                float gg = sanitize(g);
                float bb = sanitize(b);

                // Reinhard tonemap
                rr = rr / (rr + 1.0f);
                gg = gg / (gg + 1.0f);
                bb = bb / (bb + 1.0f);

                uint8_t ri = linearToSRGB8Fast(rr);
                uint8_t gi = linearToSRGB8Fast(gg);
                uint8_t bi = linearToSRGB8Fast(bb);

                fbData[idx] = SDL_MapRGB(fmt, ri, gi, bi);
            }

            // Bulk copy framebuffer → SDL surface (single memcpy instead of per-pixel write)
            if (s) {
                SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
                if (outSurf->pixels && outSurf->w == width && outSurf->h == height) {
                    std::memcpy(outSurf->pixels, fbData, totalPixels * sizeof(uint32_t));
                }
            }

            if (tex) SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, fbData, width * 4);
        }
        else {
            m_device->downloadBuffer(m_stagingBuffer, framebuffer->data(), (uint64_t)width * height * 4);

            // Update Surface (Critical for Main.cpp display and blitting)
            if (s) {
                SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
                if (outSurf->pixels && outSurf->w == width && outSurf->h == height) {
                    std::memcpy(outSurf->pixels, framebuffer->data(), width * height * 4);
                }
            }

            if (tex) {
                SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, framebuffer->data(), width * 4);
            }
        }

        this->m_currentSamples++;
        m_hasPresentedRenderedFrame = true;

       
        if (m_statusCallback) {
           // m_statusCallback("Vulkan Progressive Rendering (" + std::to_string(m_currentSamples) + " samples)", m_currentSamples);
        }
    } else {
        // [FIX] RT pipeline or TLAS not ready yet — show background (sky) instead
        // of returning without writing anything (which caused black screen).
        if (m_hasPresentedRenderedFrame) {
            rePresentCachedFrame();
        } else {
            presentBackgroundOnly();
        }
    }
}

void VulkanBackendAdapter::downloadImage(void* out) { (void)out; }
bool VulkanBackendAdapter::getDenoiserFrame(DenoiserFrameData& frame, bool useAuxiliary, bool includeColor) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device || !m_device->isInitialized() || m_imageWidth <= 0 || m_imageHeight <= 0) return false;
    if (includeColor && (!m_denoiserColorImage.image || !m_denoiserColorStagingBuffer.buffer)) return false;
    if (useAuxiliary && (!m_denoiserAlbedoImage.image || !m_denoiserNormalImage.image)) return false;
    if (useAuxiliary && (!m_denoiserAlbedoStagingBuffer.buffer || !m_denoiserNormalStagingBuffer.buffer)) return false;
    if (!includeColor && !useAuxiliary) return false;   // nothing requested

    const size_t pixelCount = (size_t)m_imageWidth * (size_t)m_imageHeight;
    const bool isHalfFloat = (m_denoiserColorImage.format == VK_FORMAT_R16G16B16A16_SFLOAT);

    // Position AOV (stylize): copied alongside the aux set when its image exists.
    const bool wantPosition = useAuxiliary && m_denoiserPositionImage.image && m_denoiserPositionStagingBuffer.buffer;

    // Build the copy list dynamically. Callers that don't need color (the Stylize AOV
    // pull reads color from the display surface, not this buffer) skip a full-res image
    // copy + download — meaningful during camera orbit where this fires every frame.
    VulkanRT::ImageHandle  copySrcs[4];
    VulkanRT::BufferHandle copyDsts[4];
    size_t copyCount = 0;
    if (includeColor) { copySrcs[copyCount] = m_denoiserColorImage;    copyDsts[copyCount] = m_denoiserColorStagingBuffer;    ++copyCount; }
    if (useAuxiliary) {
        copySrcs[copyCount] = m_denoiserAlbedoImage; copyDsts[copyCount] = m_denoiserAlbedoStagingBuffer; ++copyCount;
        copySrcs[copyCount] = m_denoiserNormalImage; copyDsts[copyCount] = m_denoiserNormalStagingBuffer; ++copyCount;
    }
    if (wantPosition) { copySrcs[copyCount] = m_denoiserPositionImage; copyDsts[copyCount] = m_denoiserPositionStagingBuffer; ++copyCount; }
    if (copyCount == 0) return false;
    if (copyCount == 1) m_device->copyImageToBuffer(copySrcs[0], copyDsts[0]);
    else                m_device->copyImagesToBuffersBatched(copySrcs, copyDsts, copyCount);

    if (includeColor) m_denoiserColorPixels.resize(pixelCount * 3);
    else              m_denoiserColorPixels.clear();
    if (useAuxiliary) {
        m_denoiserAlbedoPixels.resize(pixelCount * 3);
        m_denoiserNormalPixels.resize(pixelCount * 3);
    } else {
        m_denoiserAlbedoPixels.clear();
        m_denoiserNormalPixels.clear();
    }
    if (wantPosition) {
        m_denoiserPositionPixels.resize(pixelCount * 4);
    } else {
        m_denoiserPositionPixels.clear();
    }

    auto downloadFloat3 = [&](const VulkanRT::BufferHandle& staging, std::vector<float>& dst, bool decodeNormal) {
        if (isHalfFloat) {
            std::vector<uint16_t> packed(pixelCount * 4);
            m_device->downloadBuffer(staging, packed.data(), (uint64_t)packed.size() * sizeof(uint16_t));
            for (size_t i = 0; i < pixelCount; ++i) {
                float x = halfToFloat(packed[i * 4 + 0]);
                float y = halfToFloat(packed[i * 4 + 1]);
                float z = halfToFloat(packed[i * 4 + 2]);
                if (decodeNormal) {
                    x = x * 2.0f - 1.0f;
                    y = y * 2.0f - 1.0f;
                    z = z * 2.0f - 1.0f;
                }
                size_t px = i % (size_t)m_imageWidth;
                size_t py = i / (size_t)m_imageWidth;
                size_t flipped = ((size_t)m_imageHeight - 1 - py) * (size_t)m_imageWidth + px;
                dst[flipped * 3 + 0] = x;
                dst[flipped * 3 + 1] = y;
                dst[flipped * 3 + 2] = z;
            }
        } else {
            std::vector<float> packed(pixelCount * 4);
            m_device->downloadBuffer(staging, packed.data(), (uint64_t)packed.size() * sizeof(float));
            for (size_t i = 0; i < pixelCount; ++i) {
                float x = packed[i * 4 + 0];
                float y = packed[i * 4 + 1];
                float z = packed[i * 4 + 2];
                if (decodeNormal) {
                    x = x * 2.0f - 1.0f;
                    y = y * 2.0f - 1.0f;
                    z = z * 2.0f - 1.0f;
                }
                size_t px = i % (size_t)m_imageWidth;
                size_t py = i / (size_t)m_imageWidth;
                size_t flipped = ((size_t)m_imageHeight - 1 - py) * (size_t)m_imageWidth + px;
                dst[flipped * 3 + 0] = x;
                dst[flipped * 3 + 1] = y;
                dst[flipped * 3 + 2] = z;
            }
        }
    };

    if (includeColor) downloadFloat3(m_denoiserColorStagingBuffer, m_denoiserColorPixels, false);
    if (useAuxiliary) {
        downloadFloat3(m_denoiserAlbedoStagingBuffer, m_denoiserAlbedoPixels, false);
        downloadFloat3(m_denoiserNormalStagingBuffer, m_denoiserNormalPixels, true);
    }

    if (wantPosition) {
        // Position AOV is always rgba32f (x,y,z,depth). Keep all 4 channels and apply the
        // same vertical flip as downloadFloat3 so it matches the bottom-up CPU AOV layout.
        std::vector<float> packed(pixelCount * 4);
        m_device->downloadBuffer(m_denoiserPositionStagingBuffer, packed.data(), (uint64_t)packed.size() * sizeof(float));
        for (size_t i = 0; i < pixelCount; ++i) {
            size_t px = i % (size_t)m_imageWidth;
            size_t py = i / (size_t)m_imageWidth;
            size_t flipped = ((size_t)m_imageHeight - 1 - py) * (size_t)m_imageWidth + px;
            m_denoiserPositionPixels[flipped * 4 + 0] = packed[i * 4 + 0];
            m_denoiserPositionPixels[flipped * 4 + 1] = packed[i * 4 + 1];
            m_denoiserPositionPixels[flipped * 4 + 2] = packed[i * 4 + 2];
            m_denoiserPositionPixels[flipped * 4 + 3] = packed[i * 4 + 3];
        }
    }

    frame.width = m_imageWidth;
    frame.height = m_imageHeight;
    frame.color = includeColor ? m_denoiserColorPixels.data() : nullptr;
    frame.albedo = useAuxiliary ? m_denoiserAlbedoPixels.data() : nullptr;
    frame.normal = useAuxiliary ? m_denoiserNormalPixels.data() : nullptr;
    frame.position = wantPosition ? m_denoiserPositionPixels.data() : nullptr;
    return true;
}

// ============================================================================
// GPU-native stylize (no CUDA) — std430 params block matching stylize.comp.
// All members are 16 bytes so std430 == std140 (no padding); keep in lockstep
// with StylizeParams in shaders/stylize.comp.
// ============================================================================
namespace {
struct StylizeParamsStd430 {
    float    cam_lower_left[4];
    float    cam_horizontal[4];
    float    cam_vertical[4];
    float    cam_origin[4];
    float    ray_origin[4];
    float    sun_direction[4];
    float    misc0[4];   // sun_size, sun_elevation, cloud_coverage, cloud_density
    float    misc1[4];   // cloud_scale, cloud_offset_x, cloud_offset_z, global_strength
    float    misc2[4];   // temporal_coherence
    int32_t  idims[4];   // width, height, frame_index
    int32_t  iflags[4];  // clouds_enabled, cloud_seed
    uint32_t color_mask[4];  // R, G, B, A
    int32_t  color_shift[4]; // R, G, B
    float    palette_shadow[4];
    float    palette_mid[4];
    float    palette_highlight[4];
    float    sky_horizon[4];
    float    sky_zenith[4];
    float    sky_sunglow[4];
    float    sky0[4];    // gradient_strength, cloud_brush_scale, cloud_brush_strength, wind_smear
    float    sky1[4];    // horizon_haze, sun_disc_scale, cloud_roundness
    int32_t  sky_flags[4];   // enabled, style
    float    mtl0[4];    // brush_strength, brush_scale, pigment_thickness, dry_brush
    float    mtl1[4];    // oil_body, paint_load, pickup_rate, deposit_rate
    float    mtl2[4];    // bristle_buildup, surface_adherence, depth_scale_response, edge_respect
    float    mtl3[4];    // palette_influence, material_color_preservation, color_simplification
    int32_t  mat_flags[4];   // enabled, stroke_direction, wet_oil_model
    float    out_custom[4];  // custom_color
    float    out0[4];    // strength, width, taper, break_up
    float    out1[4];    // color_bleed, distance_thinning, detail_protection
    int32_t  out_flags[4];   // enabled, line_type, color_mode
};
static_assert(sizeof(StylizeParamsStd430) == 31 * 16, "StylizeParamsStd430 layout mismatch with stylize.comp");
}  // namespace

bool VulkanBackendAdapter::applyStylizeGPU(void* surfacePtr,
                                           const StylizeGPU::KernelParams& params,
                                           const StylizeCore::StyleProfileCore& profile) {
    SDL_Surface* surf = static_cast<SDL_Surface*>(surfacePtr);
    if (!surf || !surf->pixels) return false;

    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device || !m_device->isInitialized() || !m_device->hasStylizePipeline()) return false;

    // Skip while viewport tears down/rebuilds (same hazard class as the denoiser).
    extern std::atomic<bool> g_viewport_rebuild_in_progress;
    if (g_viewport_rebuild_in_progress.load(std::memory_order_acquire)) return false;

    const int w = m_imageWidth, h = m_imageHeight;
    if (w <= 0 || h <= 0) return false;
    if (surf->w != w || surf->h != h) return false;
    if (!surf->format || surf->format->BytesPerPixel != 4) return false;
    // Resident AOV images + views are required (surface-locked stylize parity with CPU).
    if (!m_denoiserPositionImage.image || !m_denoiserPositionImage.view) return false;
    if (!m_denoiserAlbedoImage.image   || !m_denoiserAlbedoImage.view)   return false;
    if (!m_denoiserNormalImage.image   || !m_denoiserNormalImage.view)   return false;

    const size_t colorBytes = (size_t)w * (size_t)h * sizeof(uint32_t);

    // Persistent host-visible color SSBO (reallocated on resize) + params SSBO.
    if (!m_stylizeColorBuf.buffer || m_stylizeColorW != w || m_stylizeColorH != h) {
        if (m_stylizeColorBuf.buffer) m_device->destroyBuffer(m_stylizeColorBuf);
        VulkanRT::BufferCreateInfo ci;
        ci.size = colorBytes;
        ci.usage = VulkanRT::BufferUsage::STORAGE;
        ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
        m_stylizeColorBuf = m_device->createBuffer(ci);
        if (!m_stylizeColorBuf.buffer) return false;
        m_stylizeColorW = w; m_stylizeColorH = h;
    }
    if (!m_stylizeParamsBuf.buffer) {
        VulkanRT::BufferCreateInfo ci;
        ci.size = sizeof(StylizeParamsStd430);
        ci.usage = VulkanRT::BufferUsage::STORAGE;
        ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
        m_stylizeParamsBuf = m_device->createBuffer(ci);
        if (!m_stylizeParamsBuf.buffer) return false;
    }

    // Upload the graded surface into the color SSBO (handle row pitch).
    const bool packed = ((size_t)surf->pitch == (size_t)w * sizeof(uint32_t));
    std::vector<uint32_t> rowtmp;
    if (packed) {
        m_device->uploadBuffer(m_stylizeColorBuf, surf->pixels, colorBytes);
    } else {
        rowtmp.resize((size_t)w * (size_t)h);
        for (int y = 0; y < h; ++y)
            std::memcpy(&rowtmp[(size_t)y * w],
                        (const uint8_t*)surf->pixels + (size_t)y * surf->pitch,
                        (size_t)w * sizeof(uint32_t));
        m_device->uploadBuffer(m_stylizeColorBuf, rowtmp.data(), colorBytes);
    }

    // Build the std430 params block from KernelParams + StyleProfileCore + surface format.
    StylizeParamsStd430 p{};
    auto set4 = [](float d[4], float a, float b, float c, float e) { d[0]=a; d[1]=b; d[2]=c; d[3]=e; };
    auto set4i = [](int32_t d[4], int a, int b, int c, int e) { d[0]=a; d[1]=b; d[2]=c; d[3]=e; };
    set4(p.cam_lower_left, params.cam_lower_left.x, params.cam_lower_left.y, params.cam_lower_left.z, 0.0f);
    set4(p.cam_horizontal, params.cam_horizontal.x, params.cam_horizontal.y, params.cam_horizontal.z, 0.0f);
    set4(p.cam_vertical,   params.cam_vertical.x,   params.cam_vertical.y,   params.cam_vertical.z,   0.0f);
    set4(p.cam_origin,     params.cam_origin.x,     params.cam_origin.y,     params.cam_origin.z,     0.0f);
    set4(p.ray_origin,     params.ray_origin.x,     params.ray_origin.y,     params.ray_origin.z,     0.0f);
    set4(p.sun_direction,  params.sun_direction.x,  params.sun_direction.y,  params.sun_direction.z,  0.0f);
    set4(p.misc0, params.sun_size, params.sun_elevation, params.cloud_coverage, params.cloud_density);
    set4(p.misc1, params.cloud_scale, params.cloud_offset_x, params.cloud_offset_z, profile.global_strength);
    set4(p.misc2, profile.temporal_coherence, 0.0f, 0.0f, 0.0f);
    set4i(p.idims, w, h, params.frame_index, 0);
    set4i(p.iflags, params.clouds_enabled, params.cloud_seed, 0, 0);
    p.color_mask[0] = surf->format->Rmask; p.color_mask[1] = surf->format->Gmask;
    p.color_mask[2] = surf->format->Bmask; p.color_mask[3] = surf->format->Amask;
    set4i(p.color_shift, surf->format->Rshift, surf->format->Gshift, surf->format->Bshift, 0);
    set4(p.palette_shadow,    profile.palette_shadow.x,    profile.palette_shadow.y,    profile.palette_shadow.z,    0.0f);
    set4(p.palette_mid,       profile.palette_mid.x,       profile.palette_mid.y,       profile.palette_mid.z,       0.0f);
    set4(p.palette_highlight, profile.palette_highlight.x, profile.palette_highlight.y, profile.palette_highlight.z, 0.0f);
    set4(p.sky_horizon,  profile.sky.horizon_color.x,  profile.sky.horizon_color.y,  profile.sky.horizon_color.z,  0.0f);
    set4(p.sky_zenith,   profile.sky.zenith_color.x,   profile.sky.zenith_color.y,   profile.sky.zenith_color.z,   0.0f);
    set4(p.sky_sunglow,  profile.sky.sun_glow_color.x, profile.sky.sun_glow_color.y, profile.sky.sun_glow_color.z, 0.0f);
    set4(p.sky0, profile.sky.gradient_strength, profile.sky.cloud_brush_scale, profile.sky.cloud_brush_strength, profile.sky.wind_smear);
    set4(p.sky1, profile.sky.horizon_haze, profile.sky.sun_disc_scale, profile.sky.cloud_roundness, 0.0f);
    set4i(p.sky_flags, profile.sky.enabled, profile.sky.style, 0, 0);
    set4(p.mtl0, profile.material.brush_strength, profile.material.brush_scale, profile.material.pigment_thickness, profile.material.dry_brush);
    set4(p.mtl1, profile.material.oil_body, profile.material.paint_load, profile.material.pickup_rate, profile.material.deposit_rate);
    set4(p.mtl2, profile.material.bristle_buildup, profile.material.surface_adherence, profile.material.depth_scale_response, profile.material.edge_respect);
    set4(p.mtl3, profile.material.palette_influence, profile.material.material_color_preservation, profile.material.color_simplification, 0.0f);
    set4i(p.mat_flags, profile.material.enabled, profile.material.stroke_direction, profile.material.wet_oil_model, 0);
    set4(p.out_custom, profile.outline.custom_color.x, profile.outline.custom_color.y, profile.outline.custom_color.z, 0.0f);
    set4(p.out0, profile.outline.strength, profile.outline.width, profile.outline.taper, profile.outline.break_up);
    set4(p.out1, profile.outline.color_bleed, profile.outline.distance_thinning, profile.outline.detail_protection, 0.0f);
    set4i(p.out_flags, profile.outline.enabled, profile.outline.line_type, profile.outline.color_mode, 0);
    m_device->uploadBuffer(m_stylizeParamsBuf, &p, sizeof(p));

    if (!m_device->updateStylizeDescriptors(m_stylizeColorBuf, m_stylizeParamsBuf,
            m_denoiserPositionImage.view, m_denoiserAlbedoImage.view, m_denoiserNormalImage.view)) {
        return false;
    }
    if (!m_device->dispatchStylizeCompute((uint32_t)w, (uint32_t)h,
            m_denoiserPositionImage.image, m_denoiserAlbedoImage.image, m_denoiserNormalImage.image)) {
        return false;
    }

    // Download the stylized color back into the surface.
    if (packed) {
        m_device->downloadBuffer(m_stylizeColorBuf, surf->pixels, colorBytes);
    } else {
        if (rowtmp.size() != (size_t)w * (size_t)h) rowtmp.resize((size_t)w * (size_t)h);
        m_device->downloadBuffer(m_stylizeColorBuf, rowtmp.data(), colorBytes);
        for (int y = 0; y < h; ++y)
            std::memcpy((uint8_t*)surf->pixels + (size_t)y * surf->pitch,
                        &rowtmp[(size_t)y * w],
                        (size_t)w * sizeof(uint32_t));
    }
    return true;
}

// ============================================================================
// GPU-direct denoiser interop (Vulkan → CUDA → OIDN)
// ============================================================================

struct VulkanBackendAdapter::VulkanCudaDenoiserInterop {
    // Imported Vulkan memory (one per AOV). CUDA owns the Win32 handle once
    // cudaImportExternalMemory succeeds; the memory itself stays owned by
    // Vulkan, so we must destroy this import BEFORE freeing VkDeviceMemory.
    cudaExternalMemory_t colorExt  = nullptr;
    cudaExternalMemory_t albedoExt = nullptr;
    cudaExternalMemory_t normalExt = nullptr;
    // Mapped device pointers into the imported memory (read-only source).
    void* colorSrcDev  = nullptr;
    void* albedoSrcDev = nullptr;
    void* normalSrcDev = nullptr;
    // CUDA-owned destination buffers the prep kernel writes into. OIDN reads
    // from these (float4, 16 B stride, Y-flipped + normal-decoded).
    void* colorDstDev  = nullptr;
    void* albedoDstDev = nullptr;
    void* normalDstDev = nullptr;
    // Cached shape — invalidates on mismatch.
    int width  = 0;
    int height = 0;
    bool hasAux = false;
    cudaStream_t stream = nullptr;
    int cudaOrdinal = -1;
};

void VulkanBackendAdapter::destroyGpuDenoiserInterop() {
    if (!m_gpuDenoiserInterop) return;
    VulkanCudaDenoiserInterop* s = m_gpuDenoiserInterop;

    // Order matters: free CUDA-owned dst buffers, drop mapped ptrs via destroyExternalMemory
    // (which also closes the Win32 HANDLE CUDA took ownership of), then the VkDeviceMemory
    // gets freed later when the caller destroys the VkBuffer handles.
    auto freeDst = [](void*& p) {
        if (p) { cudaFree(p); p = nullptr; }
    };
    freeDst(s->colorDstDev);
    freeDst(s->albedoDstDev);
    freeDst(s->normalDstDev);

    auto dropExt = [](cudaExternalMemory_t& e, void*& mapped) {
        mapped = nullptr; // was a child of `e`; destroyed with it
        if (e) {
            cudaDestroyExternalMemory(e);
            e = nullptr;
        }
    };
    dropExt(s->colorExt,  s->colorSrcDev);
    dropExt(s->albedoExt, s->albedoSrcDev);
    dropExt(s->normalExt, s->normalSrcDev);

    s->width = 0;
    s->height = 0;
    s->hasAux = false;

    delete s;
    m_gpuDenoiserInterop = nullptr;
}

bool VulkanBackendAdapter::ensureGpuDenoiserInterop(int width, int height, bool needAux) {
    if (m_gpuDenoiserDisabled) return false;
    if (!m_device || !m_device->isInitialized()) return false;
    if (!m_device->getCapabilities().supportsExternalMemoryWin32) return false;
    if (width <= 0 || height <= 0) return false;

    // Fast path FIRST — after the initial import the Win32 handles have been
    // nulled (CUDA took ownership) so the handle-presence checks below would
    // permanently fail on every subsequent frame. The interop object itself
    // remains valid and reusable. NOTE: hasAux is a runtime usage flag, not a
    // structural property — we don't invalidate on aux toggle, because re-import
    // is impossible once handles are consumed (would permanently kill GPU OIDN).
    if (m_gpuDenoiserInterop &&
        m_gpuDenoiserInterop->width == width &&
        m_gpuDenoiserInterop->height == height) {
        m_gpuDenoiserInterop->hasAux = needAux; // keep flag synced for callers
        return true;
    }

    // Initial-import path: handles must still be present. We always import ALL 3
    // AOVs regardless of needAux at this moment, so that subsequent calls with a
    // different needAux value can reuse the interop. Allocation already produced
    // all three exportable stagings up front.
    if (!m_denoiserColorSharedStaging.buffer || !m_denoiserColorSharedHandle) return false;
    if (!m_denoiserAlbedoSharedStaging.buffer || !m_denoiserAlbedoSharedHandle ||
        !m_denoiserNormalSharedStaging.buffer || !m_denoiserNormalSharedHandle) {
        return false;
    }

    // Shape changed (resize) — tear down and rebuild.
    destroyGpuDenoiserInterop();

    // Match Vulkan physical device to a CUDA device ordinal via UUID. Without
    // UUID match, the imported memory would be read from the wrong device and
    // OIDN results would be garbage.
    int cudaOrdinal = -1;
    if (m_device->getCapabilities().hasDeviceUUID) {
        int cudaDeviceCount = 0;
        if (cudaGetDeviceCount(&cudaDeviceCount) == cudaSuccess) {
            for (int i = 0; i < cudaDeviceCount; ++i) {
                cudaDeviceProp props{};
                if (cudaGetDeviceProperties(&props, i) != cudaSuccess) continue;
                if (std::memcmp(props.uuid.bytes, m_device->getCapabilities().deviceUUID, 16) == 0) {
                    cudaOrdinal = i;
                    break;
                }
            }
        }
        cudaGetLastError(); // clear any sticky error from enumeration
    }
    if (cudaOrdinal < 0) {
        SCENE_LOG_WARN("[Vulkan] No matching CUDA device for Vulkan physical device — GPU-direct OIDN disabled.");
        m_gpuDenoiserDisabled = true;
        return false;
    }

    int prevDevice = 0;
    cudaGetDevice(&prevDevice);
    cudaSetDevice(cudaOrdinal);

    auto* s = new VulkanCudaDenoiserInterop();
    s->width = width;
    s->height = height;
    s->hasAux = needAux;
    s->cudaOrdinal = cudaOrdinal;
    s->stream = nullptr; // default stream; CPU already fence-synced the Vulkan copy

    auto importOne = [&](void* win32Handle, uint64_t allocSize, uint64_t mappedBytes,
                         cudaExternalMemory_t& outExt, void*& outDev) -> bool {
        cudaExternalMemoryHandleDesc desc{};
        desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        desc.handle.win32.handle = win32Handle;
        desc.handle.win32.name = nullptr;
        desc.size = allocSize;
        desc.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&outExt, &desc) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cudaExternalMemoryBufferDesc bufDesc{};
        bufDesc.offset = 0;
        bufDesc.size = mappedBytes;
        bufDesc.flags = 0;
        if (cudaExternalMemoryGetMappedBuffer(&outDev, outExt, &bufDesc) != cudaSuccess) {
            cudaGetLastError();
            cudaDestroyExternalMemory(outExt);
            outExt = nullptr;
            return false;
        }
        return true;
    };

    const uint64_t mappedBytes = static_cast<uint64_t>(width) * height * 4ull * sizeof(float);

    // Always import all 3 AOVs so runtime aux toggle is just a flag (handles
    // can be consumed exactly once per process; importing on-demand later is
    // impossible). Memory cost is ~30 MB extra at 720p — acceptable.
    bool ok = importOne(m_denoiserColorSharedHandle, m_denoiserColorSharedAllocSize,
                        mappedBytes, s->colorExt, s->colorSrcDev);
    m_denoiserColorSharedHandle = nullptr;

    if (ok) {
        ok = importOne(m_denoiserAlbedoSharedHandle, m_denoiserAlbedoSharedAllocSize,
                       mappedBytes, s->albedoExt, s->albedoSrcDev);
        m_denoiserAlbedoSharedHandle = nullptr;
    }
    if (ok) {
        ok = importOne(m_denoiserNormalSharedHandle, m_denoiserNormalSharedAllocSize,
                       mappedBytes, s->normalExt, s->normalSrcDev);
        m_denoiserNormalSharedHandle = nullptr;
    }

    if (ok) {
        // Allocate CUDA-owned destination buffers (Y-flipped / decoded AOVs) for
        // all AOVs unconditionally — same rationale as the imports above.
        auto allocDst = [&](void*& p) -> bool {
            return cudaMalloc(&p, mappedBytes) == cudaSuccess;
        };
        ok = allocDst(s->colorDstDev) && allocDst(s->albedoDstDev) && allocDst(s->normalDstDev);
        if (!ok) {
            cudaGetLastError();
        }
    }

    if (!ok) {
        SCENE_LOG_WARN("[Vulkan] CUDA external-memory import failed — disabling GPU-direct OIDN.");
        // Swap it in so destroyGpuDenoiserInterop can clean up what did succeed.
        m_gpuDenoiserInterop = s;
        destroyGpuDenoiserInterop();
        m_gpuDenoiserDisabled = true;
        cudaSetDevice(prevDevice);
        return false;
    }

    m_gpuDenoiserInterop = s;
    cudaSetDevice(prevDevice);
    return true;
}

bool VulkanBackendAdapter::getDenoiserFrameGPU(DenoiserFrameDataGPU& frame, bool useAuxiliary) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (m_gpuDenoiserDisabled) {
        static bool loggedDisable = false;
        if (!loggedDisable) {
            loggedDisable = true;
            SCENE_LOG_WARN("[Vulkan][OIDN-GPU] Disabled — falling back to CPU-visible OIDN path.");
        }
        return false;
    }
    if (!m_device || !m_device->isInitialized() || m_imageWidth <= 0 || m_imageHeight <= 0) return false;
    if (!m_denoiserColorImage.image) return false;
    if (useAuxiliary && (!m_denoiserAlbedoImage.image || !m_denoiserNormalImage.image)) return false;
    if (!m_denoiserColorSharedStaging.buffer) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            SCENE_LOG_WARN(std::string("[Vulkan][OIDN-GPU] Skipped: exportable staging not allocated (supportsExternalMemoryWin32=")
                           + std::to_string((int)m_device->getCapabilities().supportsExternalMemoryWin32)
                           + ").");
        }
        return false;
    }
    if (useAuxiliary && (!m_denoiserAlbedoSharedStaging.buffer || !m_denoiserNormalSharedStaging.buffer)) return false;

    if (!ensureGpuDenoiserInterop(m_imageWidth, m_imageHeight, useAuxiliary)) {
        return false;
    }
    VulkanCudaDenoiserInterop* s = m_gpuDenoiserInterop;
    static bool firstSuccessLogged = false;
    if (!firstSuccessLogged) {
        firstSuccessLogged = true;
        SCENE_LOG_INFO(std::string("[Vulkan][OIDN-GPU] Active: Vulkan-CUDA external-memory path engaged (cudaOrdinal=")
                       + std::to_string(s->cudaOrdinal)
                       + ", " + std::to_string(m_imageWidth) + "x" + std::to_string(m_imageHeight)
                       + ", aux=" + std::to_string((int)useAuxiliary) + ").");
    }

    // ── Fence-deferred ping-pong ──────────────────────────────────────────────
    // Steady-state ordering each call:
    //   1. Wait the previously-submitted copy fence (near-zero at steady state).
    //   2. CUDA prep kernel reads staging → writes dst.
    //   3. cudaStreamSynchronize so staging is fully consumed.
    //   4. Submit NEW Vulkan copy of m_denoiserColorImage → staging, async (no
    //      wait). The next call's wait covers this.
    //   5. Return dst pointers for OIDN. While Renderer runs OIDN+tonemap+D2H,
    //      the GPU is concurrently executing RT(N+1) + copy(N+1).
    //
    // First call: there is no prior submit, so we have nothing prep'd to return.
    // Submit the seed copy and bail with `false`; the caller falls back to last
    // frame's display. From the second call on, the pipeline is filled.

#if RT_OIDN_PROFILING
    const auto profStartTotal = std::chrono::high_resolution_clock::now();
#endif

    const bool firstCall = !m_device->hasDenoiserCopyEverSubmitted();

    // Submit list — same triple in aux mode, single color in performance mode.
    const VulkanRT::ImageHandle  srcsAux[3] = { m_denoiserColorImage, m_denoiserAlbedoImage, m_denoiserNormalImage };
    const VulkanRT::BufferHandle dstsAux[3] = { m_denoiserColorSharedStaging, m_denoiserAlbedoSharedStaging, m_denoiserNormalSharedStaging };
    const VulkanRT::ImageHandle  srcsPerf[1] = { m_denoiserColorImage };
    const VulkanRT::BufferHandle dstsPerf[1] = { m_denoiserColorSharedStaging };
    const VulkanRT::ImageHandle*  srcsPtr = useAuxiliary ? srcsAux  : srcsPerf;
    const VulkanRT::BufferHandle* dstsPtr = useAuxiliary ? dstsAux  : dstsPerf;
    const size_t copyCount             = useAuxiliary ? 3 : 1;

    if (firstCall) {
        // Seed the pipeline. We block on this one submit so the CPU/GPU state
        // settles before steady-state ping-pong begins — small one-time stall,
        // not part of the hot path.
        m_device->submitDenoiserCopyAsync(srcsPtr, dstsPtr, copyCount);
        m_device->waitDenoiserCopy();
        // Resubmit isn't needed — the data is already in staging. But we have
        // no prep'd dst yet, so we'd still have to do prep + return. To keep
        // first-call logic uniform with the steady state, fall through to the
        // wait+prep block below.
    }

    // Wait on the previously-submitted copy (signaled at steady state).
#if RT_OIDN_PROFILING
    const auto profWaitStart = std::chrono::high_resolution_clock::now();
#endif
    if (!m_device->waitDenoiserCopy()) {
        // First-call path with no prior submit must have happened above; if we
        // still see no submit, something is wrong. Bail.
        return false;
    }
#if RT_OIDN_PROFILING
    const auto profWaitEnd = std::chrono::high_resolution_clock::now();
    const float profWaitMs = std::chrono::duration<float, std::milli>(profWaitEnd - profWaitStart).count();
#endif

    int prevDevice = 0;
    cudaGetDevice(&prevDevice);
    cudaSetDevice(s->cudaOrdinal);

    // CUDA prep kernel: staging → dst buffers (Y-flip + normal-decode).
#if RT_OIDN_PROFILING
    const auto profPrepStart = std::chrono::high_resolution_clock::now();
#endif
    bool kernelOK = launchVulkanDenoiserPrepKernel(
        s->colorDstDev, s->colorSrcDev, m_imageWidth, m_imageHeight,
        /*decodeNormal=*/false, s->stream);
    if (kernelOK && useAuxiliary) {
        kernelOK = kernelOK && launchVulkanDenoiserPrepKernel(
            s->albedoDstDev, s->albedoSrcDev, m_imageWidth, m_imageHeight,
            /*decodeNormal=*/false, s->stream);
        kernelOK = kernelOK && launchVulkanDenoiserPrepKernel(
            s->normalDstDev, s->normalSrcDev, m_imageWidth, m_imageHeight,
            /*decodeNormal=*/true, s->stream);
    }
    // Sync so staging is fully consumed before we submit the next overwrite.
    cudaStreamSynchronize(s->stream);
#if RT_OIDN_PROFILING
    const auto profPrepEnd = std::chrono::high_resolution_clock::now();
    const float profPrepMs = std::chrono::duration<float, std::milli>(profPrepEnd - profPrepStart).count();
#endif

    if (!kernelOK) {
        cudaGetLastError();
        cudaSetDevice(prevDevice);
        return false;
    }

    // Submit the NEXT copy asynchronously. Vulkan queues this after RT(N) is
    // already in flight; the fence will be checked next call.
#if RT_OIDN_PROFILING
    const auto profSubmitStart = std::chrono::high_resolution_clock::now();
#endif
    m_device->submitDenoiserCopyAsync(srcsPtr, dstsPtr, copyCount);
#if RT_OIDN_PROFILING
    const auto profSubmitEnd = std::chrono::high_resolution_clock::now();
    const float profSubmitMs = std::chrono::duration<float, std::milli>(profSubmitEnd - profSubmitStart).count();

    const auto profEndTotal = std::chrono::high_resolution_clock::now();
    const float profTotalMs = std::chrono::duration<float, std::milli>(profEndTotal - profStartTotal).count();

    // Light telemetry: every 300 frames so the log stays quiet but regressions
    // (sudden wait spike from a re-introduced sync wait, prep ballooning, etc.)
    // are still visible during long sessions.
    {
        static thread_local int   profCounter = 0;
        static thread_local float profWaitAvg = 0.0f;
        static thread_local float profPrepAvg = 0.0f;
        static thread_local float profSubmitAvg = 0.0f;
        static thread_local float profTotalAvg = 0.0f;
        const float a = 0.95f, b = 0.05f;
        profWaitAvg   = profWaitAvg   * a + profWaitMs   * b;
        profPrepAvg   = profPrepAvg   * a + profPrepMs   * b;
        profSubmitAvg = profSubmitAvg * a + profSubmitMs * b;
        profTotalAvg  = profTotalAvg  * a + profTotalMs  * b;
        if (++profCounter % 300 == 0) {
            SCENE_LOG_INFO(std::string("[OIDN][Vulkan] wait=")
                           + std::to_string(profWaitAvg)
                           + "ms prep=" + std::to_string(profPrepAvg)
                           + "ms submit=" + std::to_string(profSubmitAvg)
                           + "ms total=" + std::to_string(profTotalAvg)
                           + "ms (aux=" + std::to_string((int)useAuxiliary) + ")");
        }
    }
#endif

    frame.width = m_imageWidth;
    frame.height = m_imageHeight;
    frame.colorDevPtr  = s->colorDstDev;
    frame.albedoDevPtr = useAuxiliary ? s->albedoDstDev : nullptr;
    frame.normalDevPtr = useAuxiliary ? s->normalDstDev : nullptr;
    frame.pixelByteStride = sizeof(float) * 4;
    frame.rowByteStride   = static_cast<size_t>(m_imageWidth) * sizeof(float) * 4;
    frame.cudaStream = static_cast<void*>(s->stream);
    frame.cudaDeviceOrdinal = s->cudaOrdinal;

    cudaSetDevice(prevDevice);
    return true;
}

int VulkanBackendAdapter::getCurrentSampleCount() const { return this->m_currentSamples; }
bool VulkanBackendAdapter::isAccumulationComplete() const {
    // Interactive viewport modes manage their own dirty/cached-frame lifecycle inside
    // renderInteractiveViewportImpl(). The main loop should keep calling renderProgressive()
    // so material/light/world edits can re-present immediately without waiting for camera motion.
    if (shouldUseInteractiveViewport()) {
        return false;
    }
    return this->m_currentSamples >= this->m_targetSamples;
}

// Environment stubs
void VulkanBackendAdapter::setEnvironmentMap(int64_t h) {
    if (!m_device || !m_device->isInitialized()) {
        VK_INFO() << "[VulkanBackendAdapter] Device not ready — caching env texture id" << std::endl;
        m_envTexID = h;
        return;
    }

    // The uploadTexture2D already registered the ImageHandle and updated binding 6.
    // We just record the env tex id and update the world buffer so shaders can read the slot.
    m_envTexID = h;

    // Update the full GPU-world struct with complete atmosphere parameters
    setWorldData(&m_cachedWorld);
}
void VulkanBackendAdapter::setSkyParams() {}

void VulkanBackendAdapter::uploadAtmosphereLUT(const AtmosphereLUT* lut) {
    if (!m_device || !m_device->isInitialized()) return;
    // Destroy previous LUT images held by device
    for (int i = 0; i < 4; ++i) {
        if (m_device->m_lutImages[i].image) {
            m_device->destroyImage(m_device->m_lutImages[i]);
            m_device->m_lutImages[i] = {};
        }
    }

    if (!lut) {
        VulkanRT::ImageHandle empty[4] = {};
        m_device->updateAtmosphereLUTs(empty);
        m_atmosphereLutReady = false;
        setWorldData(&m_cachedWorld);
        return;
    }

    auto upload2D = [&](const std::vector<float4>& src, uint32_t w, uint32_t h, bool wrapU) -> VulkanRT::ImageHandle {
        VulkanRT::ImageHandle img{};
        if (src.empty()) return img;
        uint64_t bytes = (uint64_t)w * h * sizeof(float4);

        VulkanRT::BufferCreateInfo stagingInfo;
        stagingInfo.size = bytes;
        stagingInfo.usage = VulkanRT::BufferUsage::TRANSFER_SRC;
        stagingInfo.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
        auto staging = m_device->createBuffer(stagingInfo);
        if (!staging.buffer) {
            VK_ERROR() << "[VulkanBackendAdapter] LUT staging buffer allocation failed (" << w << "x" << h
                       << ", bytes=" << bytes << ")." << std::endl;
            return {};
        }
        m_device->uploadBuffer(staging, src.data(), bytes);

        VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        img = m_device->createImage2D(w, h, VK_FORMAT_R32G32B32A32_SFLOAT, usage);
        if (!img.image) {
            m_device->destroyBuffer(staging);
            return {};
        }

        m_device->copyBufferToImage(staging, img);

        // Transition to SHADER_READ_ONLY_OPTIMAL
        VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
        if (cmd == VK_NULL_HANDLE) {
            VK_ERROR() << "[VulkanBackendAdapter] Failed to transition LUT image layout." << std::endl;
            m_device->destroyBuffer(staging);
            m_device->destroyImage(img);
            return {};
        }
        m_device->transitionImageLayout(cmd, img.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_device->endSingleTimeCommands(cmd);

        VkSamplerCreateInfo sInfo{};
        sInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sInfo.magFilter = VK_FILTER_LINEAR;
        sInfo.minFilter = VK_FILTER_LINEAR;
        sInfo.addressModeU = wrapU ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.anisotropyEnable = VK_FALSE;
        sInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sInfo.unnormalizedCoordinates = VK_FALSE;

        vkCreateSampler(m_device->getDevice(), &sInfo, nullptr, &img.sampler);

        m_device->destroyBuffer(staging);
        return img;
    };

    VulkanRT::ImageHandle lutImgs[4] = {};
    lutImgs[0] = upload2D(lut->getHostTransmittance(), TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H, false);
    lutImgs[1] = upload2D(lut->getHostSkyView(), SKYVIEW_LUT_W, SKYVIEW_LUT_H, true);
    lutImgs[2] = upload2D(lut->getHostMultiScatter(), MULTI_SCATTER_LUT_RES, MULTI_SCATTER_LUT_RES, false);

    // Currently skipping 3D aerial perspective LUT upload

    m_device->updateAtmosphereLUTs(lutImgs);
    // Updated device with new LUT images

    // Optional debug readback (disabled by default to avoid extra shared-memory pressure).
    constexpr bool kEnableLutReadbackDebug = false;
    if (kEnableLutReadbackDebug && lutImgs[1].image && !lut->getHostSkyView().empty()) {
        uint32_t w = SKYVIEW_LUT_W;
        uint32_t h = SKYVIEW_LUT_H;
        uint64_t bytes = (uint64_t)w * h * sizeof(float4);
        VulkanRT::BufferCreateInfo stagingInfo;
        stagingInfo.size = bytes;
        stagingInfo.usage = VulkanRT::BufferUsage::TRANSFER_DST;
        stagingInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
        auto staging = m_device->createBuffer(stagingInfo);
        if (staging.buffer) {
            m_device->copyImageToBuffer(lutImgs[1], staging);
            std::vector<float4> pixels(w * h);
            m_device->downloadBuffer(staging, pixels.data(), bytes);
            if (staging.buffer) m_device->destroyBuffer(staging);
        }
    }

    // Mark LUT as ready only when the GLSL samplers used by sky/aerial paths are valid.
    m_atmosphereLutReady = (lutImgs[0].view != VK_NULL_HANDLE && lutImgs[1].view != VK_NULL_HANDLE);
    // Push updated world buffer immediately so GPU sees _pad5 = 1 without waiting for next frame
    setWorldData(&m_cachedWorld);
}

bool VulkanBackendAdapter::generateAtmosphereLUTGPU(const WorldData* worldData) {
    if (!m_device || !m_device->isInitialized() || !worldData) return false;
    if (!m_device->hasAtmosphereLUTPipeline()) return false;

    if (!m_device->generateAtmosphereLUTGPU(*worldData)) {
        m_atmosphereLutReady = false;
        return false;
    }

    m_atmosphereLutReady = true;
    m_cachedWorld = *worldData;
    setWorldData(&m_cachedWorld);
    return true;
}

void VulkanBackendAdapter::setWorldData(const void* w) {
    if (!w) return;
    
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    const WorldData* wd = static_cast<const WorldData*>(w);
    m_cachedWorld = *wd; // Always update cache

    if (!m_device || !m_device->isInitialized()) {
        VK_INFO() << "[VulkanBackendAdapter] Device not ready — cached WorldData for later upload" << std::endl;
        return;
    }

    // Safety net for backend/viewport transitions: if a plain setWorldData()
    // reaches Vulkan before the Nishita LUT has been generated, build it here so
    // the first frame does not render through the analytic/fallback atmosphere
    // until a UI slider dirties the world again.
    if (wd->mode == WORLD_MODE_NISHITA &&
        !m_atmosphereLutReady &&
        !m_atmosphereLutGenerationInProgress &&
        m_device->hasAtmosphereLUTPipeline()) {
        m_atmosphereLutGenerationInProgress = true;
        WorldData worldCopy = *wd;
        const bool generated = generateAtmosphereLUTGPU(&worldCopy);
        m_atmosphereLutGenerationInProgress = false;
        if (generated) {
            return;
        }
    }

    // Pack a complete GPU-friendly world struct with full Nishita support
    VulkanRT::VkWorldDataExtended gw{};

    // ═════════════════════════════════════════════════════════════════
    // CORE MODE & SUN TINT
    // ═════════════════════════════════════════════════════════════════
    gw.sunDir[0] = wd->nishita.sun_direction.x;
    gw.sunDir[1] = wd->nishita.sun_direction.y;
    gw.sunDir[2] = wd->nishita.sun_direction.z;
    gw.mode = wd->mode;
    
    // Prefer top-level world color for tint (Color mode), otherwise use a warm default
    if (wd->color.x != 0.0f || wd->color.y != 0.0f || wd->color.z != 0.0f) {
        gw.sunColor[0] = wd->color.x;
        gw.sunColor[1] = wd->color.y;
        gw.sunColor[2] = wd->color.z;
    } else {
        gw.sunColor[0] = 1.0f;
        gw.sunColor[1] = 0.95f;
        gw.sunColor[2] = 0.9f;
    }
    gw.sunIntensity = wd->nishita.sun_intensity;
    gw.atmosphereIntensity = wd->nishita.atmosphere_intensity;

    // ═════════════════════════════════════════════════════════════════
    // NISHITA SUN PARAMETERS
    // ═════════════════════════════════════════════════════════════════
    gw.sunSize = wd->nishita.sun_size;
    gw.mieAnisotropy = wd->nishita.mie_anisotropy;
    gw.rayleighDensity = wd->nishita.rayleigh_density;
    gw.mieDensity = wd->nishita.mie_density;
    
    gw.humidity = wd->nishita.humidity;
    gw.temperature = wd->nishita.temperature;
    gw.ozoneAbsorptionScale = wd->nishita.ozone_absorption_scale;
    gw.atmosphereIntensity = wd->nishita.atmosphere_intensity;

    // ═════════════════════════════════════════════════════════════════
    // ATMOSPHERE DENSITY PARAMETERS
    // ═════════════════════════════════════════════════════════════════
    gw.airDensity = wd->nishita.air_density;
    gw.dustDensity = wd->nishita.dust_density;
    gw.ozoneDensity = wd->nishita.ozone_density;
    gw.altitude = wd->nishita.altitude;
    
    gw.planetRadius = wd->nishita.planet_radius;
    gw.atmosphereHeight = wd->nishita.atmosphere_height;
    // multiScatterEnabled / multiScatterFactor doldurması aşağıda
    // (AERIAL PERSPECTIVE bloğunun altında) yapılıyor.

    // ═════════════════════════════════════════════════════════════════
    // CLOUD LAYER 1 PARAMETERS
    // ═════════════════════════════════════════════════════════════════
    // Legacy sky-shader clouds are disabled. Sky cloud UI now drives an internal
    // procedural VDBVolume, so Vulkan sees clouds through the volume pipeline.
    gw.cloudsEnabled = 0;
    gw.cloudCoverage = wd->nishita.cloud_coverage;
    gw.cloudDensity = wd->nishita.cloud_density;
    gw.cloudScale = wd->nishita.cloud_scale;
    
    gw.cloudHeightMin = wd->nishita.cloud_height_min;
    gw.cloudHeightMax = wd->nishita.cloud_height_max;
    gw.cloudOffsetX = wd->nishita.cloud_offset_x;
    gw.cloudOffsetZ = wd->nishita.cloud_offset_z;
    
    gw.cloudQuality = wd->nishita.cloud_quality;
    gw.cloudDetail = wd->nishita.cloud_detail;
    gw.cloudBaseSteps = wd->nishita.cloud_base_steps;
    gw.cloudLightSteps = wd->nishita.cloud_light_steps;
    
    gw.cloudShadowStrength = wd->nishita.cloud_shadow_strength;
    gw.cloudAmbientStrength = wd->nishita.cloud_ambient_strength;
    gw.cloudSilverIntensity = wd->nishita.cloud_silver_intensity;
    gw.cloudAbsorption = wd->nishita.cloud_absorption;

    // ═════════════════════════════════════════════════════════════════
    // ADVANCED CLOUD SCATTERING
    // ═════════════════════════════════════════════════════════════════
    gw.cloudAnisotropy = wd->nishita.cloud_anisotropy;
    gw.cloudAnisotropyBack = wd->nishita.cloud_anisotropy_back;
    gw.cloudLobeMix = wd->nishita.cloud_lobe_mix;
    gw.cloudEmissiveIntensity = wd->nishita.cloud_emissive_intensity;
    
    gw.cloudEmissiveColor[0] = wd->nishita.cloud_emissive_color.x;
    gw.cloudEmissiveColor[1] = wd->nishita.cloud_emissive_color.y;
    gw.cloudEmissiveColor[2] = wd->nishita.cloud_emissive_color.z;
    gw._pad3 = 0.0f;

    // ═════════════════════════════════════════════════════════════════
    // FOG PARAMETERS
    // ═════════════════════════════════════════════════════════════════
    gw.fogEnabled = wd->nishita.fog_enabled ? 1 : 0;
    gw.fogDensity = wd->nishita.fog_density;
    gw.fogHeight = wd->nishita.fog_height;
    gw.fogFalloff = wd->nishita.fog_falloff;
    
    gw.fogDistance = wd->nishita.fog_distance;
    gw.fogSunScatter = wd->nishita.fog_sun_scatter;
    gw.fogColor[0] = wd->nishita.fog_color.x;
    gw.fogColor[1] = wd->nishita.fog_color.y;
    gw.fogColor[2] = wd->nishita.fog_color.z;
    gw._pad4 = 0.0f;

    // ═════════════════════════════════════════════════════════════════
    // VOLUMETRIC GOD RAYS
    // ═════════════════════════════════════════════════════════════════
    gw.godRaysEnabled = wd->nishita.godrays_enabled ? 1 : 0;
    gw.godRaysIntensity = wd->nishita.godrays_intensity;
    gw.godRaysDensity = wd->nishita.godrays_density;
    gw.godRaysSamples = wd->nishita.godrays_samples;

    // ═════════════════════════════════════════════════════════════════
    // AERIAL PERSPECTIVE (matches OptiX world.advanced)
    // ═════════════════════════════════════════════════════════════════
    gw.aerialEnabled     = wd->advanced.aerial_perspective ? 1 : 0;
    gw.aerialMinDistance = wd->advanced.aerial_min_distance;
    gw.aerialMaxDistance = wd->advanced.aerial_max_distance;
    gw.aerialDensity     = wd->advanced.aerial_density;

    // Multi-scatter (analytic, matches World::calculateNishitaSky)
    gw.multiScatterEnabled = wd->advanced.multi_scatter_enabled ? 1 : 0;
    gw.multiScatterFactor  = wd->advanced.multi_scatter_factor;

    // ═════════════════════════════════════════════════════════════════
    // WEATHER PAYLOAD (passive transport; render paths opt in later)
    // ═════════════════════════════════════════════════════════════════
    gw.weatherEnabled = wd->weather.enabled ? 1 : 0;
    gw.weatherType = wd->weather.type;
    gw.weatherIntensity = wd->weather.intensity;
    gw.weatherDensity = wd->weather.density;
    gw.weatherWindDirection[0] = wd->weather.wind_direction.x;
    gw.weatherWindDirection[1] = wd->weather.wind_direction.y;
    gw.weatherWindDirection[2] = wd->weather.wind_direction.z;
    gw.weatherWindSpeed = wd->weather.wind_speed;
    gw.weatherPrecipitationScale = wd->weather.precipitation_scale;
    gw.weatherVisibility = wd->weather.visibility;
    gw.weatherSurfaceWetness = wd->weather.surface_wetness_output;
    gw.weatherSurfaceAccumulation = wd->weather.surface_accumulation_output;
    gw.weatherSurfaceSettling = wd->weather.surface_settling_output;
    gw.weatherSurfaceHeight = wd->weather.surface_height_output;
    gw.weatherVisualMode = wd->weather.visual_mode;
    gw.weatherSurfaceResponseEnabled = wd->weather.surface_response_enabled;

    // ═════════════════════════════════════════════════════════════════
    // ENVIRONMENT & LUT REFERENCES
    // ═════════════════════════════════════════════════════════════════
    gw.envTexSlot = (int)m_envTexID;
    gw.envIntensity = wd->env_intensity;
    gw.envRotation = wd->env_rotation;
    // _pad5 repurposed as nishitaLutReady: 1 = Vulkan binding 8 has valid LUT textures.
    // The uint64 LUT fields below are CUDA texture handles in OptiX/CUDA mode and are
    // not meaningful as availability checks in GLSL.
    gw._pad5 = m_atmosphereLutReady ? 1 : 0;
    gw.envOverlayEnabled = wd->advanced.env_overlay_enabled;
    gw.envOverlayBlendMode = wd->advanced.env_overlay_blend_mode;
    gw.envOverlayIntensity = wd->advanced.env_overlay_intensity;
    gw.envOverlayRotation = wd->advanced.env_overlay_rotation * (3.14159265358979323846f / 180.0f);
    
    // LUT handles - if AtmosphereLUT was precomputed, these will be valid GPU texture objects
    // Otherwise, shaders will fall back to on-the-fly computation
    gw.transmittanceLUT = wd->lut.transmittance_lut;
    gw.skyviewLUT = wd->lut.skyview_lut;
    gw.multiScatterLUT = wd->lut.multi_scattering_lut;
    gw.aerialPerspectiveLUT = wd->lut.aerial_perspective_lut;
   

    m_device->updateWorldBuffer(&gw, sizeof(gw), 1);
    resetAccumulation();
}

void VulkanBackendAdapter::updateVDBVolumes(const std::vector<GpuVDBVolume>& vols) {
    if (!m_device) return;
    if (vols.empty()) {
        // No active volumes: release any stale cached VDB buffers immediately.
        if (!m_vdbBuffers.empty() || !m_vdbTempBuffers.empty()) {
            m_device->waitIdle();
        }
        for (auto& [id, buf] : m_vdbBuffers) {
            (void)id;
            if (buf.buffer) m_device->destroyBuffer(buf);
        }
        m_vdbBuffers.clear();
        m_vdbUploadedVersions.clear();
        for (auto& [id, buf] : m_vdbTempBuffers) {
            (void)id;
            if (buf.buffer) m_device->destroyBuffer(buf);
        }
        m_vdbTempBuffers.clear();
        m_vdbTempUploadedVersions.clear();
        m_orderedVDBInstances.clear();
        m_device->updateVolumeBuffer(nullptr, 0, 0);
        return;
    }

    // Build id->source map for fast O(1) lookup. Procedural volumes do not have
    // stable VDB ids (sky cloud uses -1), so keep them out of the id map; a
    // shared -1 key can corrupt the TLAS customIndex -> SSBO slot mapping when
    // Nishita sky clouds coexist with live grid-domain volumes.
    std::unordered_map<int, const GpuVDBVolume*> volByID;
    std::vector<const GpuVDBVolume*> proceduralVols;
    proceduralVols.reserve(vols.size());
    for (const auto& v : vols) {
        if (v.vdb_id >= 0) {
            volByID[v.vdb_id] = &v;
        } else if (v.source_type == 3) {
            proceduralVols.push_back(&v);
        }
    }

    // Release cached buffers for volumes that no longer exist in the scene.
    bool destroyedAny = false;
    for (auto it = m_vdbBuffers.begin(); it != m_vdbBuffers.end(); ) {
        if (volByID.find(it->first) == volByID.end()) {
            if (!destroyedAny) {
                m_device->waitIdle();
                destroyedAny = true;
            }
            if (it->second.buffer) m_device->destroyBuffer(it->second);
            m_vdbUploadedVersions.erase(it->first);
            it = m_vdbBuffers.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = m_vdbTempBuffers.begin(); it != m_vdbTempBuffers.end(); ) {
        if (volByID.find(it->first) == volByID.end()) {
            if (!destroyedAny) {
                m_device->waitIdle();
                destroyedAny = true;
            }
            if (it->second.buffer) m_device->destroyBuffer(it->second);
            m_vdbTempUploadedVersions.erase(it->first);
            it = m_vdbTempBuffers.erase(it);
        } else {
            ++it;
        }
    }

    // ORDERING FIX: SSBO slot i must correspond to the unified volume with TLAS customIndex==i.
    // After updateGeometry(), m_orderedVDBInstances records VDBs in TLAS traversal order.
    // If BVH reorders them vs. scene.vdb_volumes, this ensures shader lookups are correct.
    std::vector<const GpuVDBVolume*> orderedVols;
    if (!m_orderedVDBInstances.empty()) {
        std::size_t proceduralIndex = 0;
        for (const auto& hittable : m_orderedVDBInstances) {
            auto vdb = std::dynamic_pointer_cast<VDBVolume>(hittable);
            int volume_id = -1;
            if (vdb) {
                volume_id = vdb->getVDBVolumeID();
                if (volume_id < 0 && vdb->isProceduralVolume()) {
                    orderedVols.push_back(proceduralIndex < proceduralVols.size()
                        ? proceduralVols[proceduralIndex++]
                        : nullptr);
                    continue;
                }
            } else if (auto gas = std::dynamic_pointer_cast<GasVolume>(hittable)) {
                volume_id = gas->live_vdb_id;
            }
            if (volume_id < 0 && !(vdb && vdb->isProceduralVolume())) { orderedVols.push_back(nullptr); continue; }
            auto it = volByID.find(volume_id);
            orderedVols.push_back(it != volByID.end() ? it->second : nullptr);
        }
    } else {
        // Fallback: no geometry build yet, use input order
        for (const auto& v : vols) orderedVols.push_back(&v);
    }
    if (orderedVols.empty()) { m_device->m_volumeCount = 0; return; }

    // Convert GpuVDBVolume (OptiX/CUDA struct) → VkVolumeInstance (Vulkan SSBO)
    std::vector<VulkanRT::VkVolumeInstance> instances(orderedVols.size());
    for (size_t i = 0; i < orderedVols.size(); i++) {
        auto& dst = instances[i];
        memset(&dst, 0, sizeof(dst));
        dst.is_active = 0;
        if (!orderedVols[i]) continue; // deleted/missing → leave inactive slot
        const auto& src = *orderedVols[i];

        // Copy original transforms directly (preserves rotation)
        for (int i = 0; i < 12; ++i) {
            dst.transform[i]     = src.transform[i];
            dst.inv_transform[i] = src.inv_transform[i];
        }
        
        // Pivot offset for OptiX parity
        dst.pivot_offset[0] = src.pivot_offset[0];
        dst.pivot_offset[1] = src.pivot_offset[1];
        dst.pivot_offset[2] = src.pivot_offset[2];
        dst.source_type = src.source_type;
        // Isosurface IOR (source_type==4) rides _ext_reserved[0], roughness
        // rides _ext_reserved[1] — keeps the 512-byte VkVolumeInstance layout
        // unchanged (reserved tail slots).
        dst._ext_reserved[0] = (src.ior > 1.0f) ? src.ior : 1.33f;
        dst._ext_reserved[1] = src.surface_roughness;
        dst._ext_reserved[2] = src.surface_foam;
        // Particle-foam look for the SurfaceSDF single-volume path (temperature
        // channel): tint in [3..5], extinction multiplier in [6].
        dst._ext_reserved[3] = src.foam_color.x;
        dst._ext_reserved[4] = src.foam_color.y;
        dst._ext_reserved[5] = src.foam_color.z;
        dst._ext_reserved[6] = src.foam_opacity;
        dst.cloud_coverage = src.cloud_coverage;
        dst.cloud_detail = src.cloud_detail;
        dst.cloud_erosion = src.cloud_erosion;
        dst.cloud_base_scale = src.cloud_base_scale;
        dst.cloud_edge_fade = src.cloud_edge_fade;
        dst.cloud_offset_x = src.cloud_offset_x;
        dst.cloud_offset_z = src.cloud_offset_z;
        dst.cloud_seed = src.cloud_seed;

        // VDB native (original file) world-space AABB — used by the shader to remap
        // localPos [-0.5,0.5] → VDB world space before NanoVDB index lookup.
        // Must be local_bbox (not world_bbox) so gizmo moves don't corrupt the mapping.
        dst.aabb_min[0] = src.local_bbox_min.x; dst.aabb_min[1] = src.local_bbox_min.y; dst.aabb_min[2] = src.local_bbox_min.z;
        dst.aabb_max[0] = src.local_bbox_max.x; dst.aabb_max[1] = src.local_bbox_max.y; dst.aabb_max[2] = src.local_bbox_max.z;

        // Density
        dst.density_multiplier = src.density_multiplier;
        dst.density_remap_low = src.density_remap_low;
        dst.density_remap_high = src.density_remap_high;
        dst.noise_scale = 1.0f;
        dst._reserved[0] = (src.density_pad > 0.0f) ? src.density_pad : 0.04f;
        dst._reserved[1] = src.emission_pad;
        
        // Sync NanoVDB Host Buffer to Vulkan Device Buffer
        dst.volume_type = 2; // 2 = NanoVDB
        dst.vdb_grid_address = 0;
        dst.vdb_temp_address = 0;
        
        int vdb_id = src.vdb_id;
        if (vdb_id >= 0) {
            auto& mgr = VDBVolumeManager::getInstance();
            void* hostGrid = mgr.getHostGrid(vdb_id);
            size_t gridSize = mgr.getHostGridSize(vdb_id);
            uint32_t currentVersion = mgr.getContentVersion(vdb_id);
            
            if (hostGrid && gridSize > 0) {
                auto it = m_vdbBuffers.find(vdb_id);
                bool needsUpload = false;
                
                // Over-allocate by 50% to absorb frame-to-frame NanoVDB growth.
                // CPU_TO_GPU (host-visible, device-local BAR) memory lets uploadBuffer
                // use vkMapMemory + memcpy directly — no staging buffer, no command
                // buffer submit, no vkWaitForFences. This eliminates the per-frame
                // GPU stall that was blocking fluid sim playback. NanoVDB data changes
                // every frame anyway, so device-local-only has no advantage here.
                const size_t allocSize = gridSize + (gridSize / 2);
                if (it == m_vdbBuffers.end() || it->second.size < gridSize) {
                    if (it != m_vdbBuffers.end()) {
                        m_device->waitIdle();
                        m_device->destroyBuffer(it->second);
                    }
                    VulkanRT::BufferCreateInfo ci;
                    ci.size = allocSize;
                    ci.usage = (VulkanRT::BufferUsage)(
                        (uint32_t)VulkanRT::BufferUsage::STORAGE |
                        (uint32_t)VulkanRT::BufferUsage::TRANSFER_DST |
                        0x0100 /* VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT - custom */);
                    ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
                    VulkanRT::BufferHandle buf = m_device->createBuffer(ci);
                    m_vdbBuffers[vdb_id] = buf;
                    it = m_vdbBuffers.find(vdb_id);
                    needsUpload = true;
                }
                
                // Check version
                auto versionIt = m_vdbUploadedVersions.find(vdb_id);
                if (versionIt == m_vdbUploadedVersions.end() || versionIt->second != currentVersion) {
                    needsUpload = true;
                }
                
                if (it != m_vdbBuffers.end() && it->second.buffer) {
                    if (needsUpload) {
                        m_device->uploadBuffer(it->second, hostGrid, gridSize);
                        m_vdbUploadedVersions[vdb_id] = currentVersion;
                    }
                    dst.vdb_grid_address = it->second.deviceAddress;
                }
            }

            // Upload temperature NanoVDB grid for blackbody/color-ramp emission (mode 2)
            void* hostTempGrid = mgr.getHostTemperatureGrid(vdb_id);
            size_t tempGridSize = mgr.getHostTemperatureGridSize(vdb_id);
            if (hostTempGrid && tempGridSize > 0) {
                auto it2 = m_vdbTempBuffers.find(vdb_id);
                bool needsTempUpload = false;
                
                const size_t allocTempSize = tempGridSize + (tempGridSize / 2);
                if (it2 == m_vdbTempBuffers.end() || it2->second.size < tempGridSize) {
                    if (it2 != m_vdbTempBuffers.end()) {
                        m_device->waitIdle();
                        m_device->destroyBuffer(it2->second);
                    }
                    VulkanRT::BufferCreateInfo ci2;
                    ci2.size = allocTempSize;
                    ci2.usage = (VulkanRT::BufferUsage)(
                        (uint32_t)VulkanRT::BufferUsage::STORAGE |
                        (uint32_t)VulkanRT::BufferUsage::TRANSFER_DST |
                        0x0100);
                    ci2.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
                    m_vdbTempBuffers[vdb_id] = m_device->createBuffer(ci2);
                    it2 = m_vdbTempBuffers.find(vdb_id);
                    needsTempUpload = true;
                }
                
                // Check temperature version
                auto tempVersionIt = m_vdbTempUploadedVersions.find(vdb_id);
                if (tempVersionIt == m_vdbTempUploadedVersions.end() || tempVersionIt->second != currentVersion) {
                    needsTempUpload = true;
                }
                
                if (it2 != m_vdbTempBuffers.end() && it2->second.buffer) {
                    if (needsTempUpload) {
                        m_device->uploadBuffer(it2->second, hostTempGrid, tempGridSize);
                        m_vdbTempUploadedVersions[vdb_id] = currentVersion;
                    }
                    dst.vdb_temp_address = it2->second.deviceAddress;
                }
            }
        }

        // Scattering
        dst.scatter_color[0] = src.scatter_color.x;
        dst.scatter_color[1] = src.scatter_color.y;
        dst.scatter_color[2] = src.scatter_color.z;
        dst.scatter_coefficient = src.scatter_coefficient;
        dst.scatter_anisotropy = src.scatter_anisotropy;
        dst.scatter_anisotropy_back = src.scatter_anisotropy_back;
        dst.scatter_lobe_mix = src.scatter_lobe_mix;
        dst.scatter_multi = src.scatter_multi;

        // Absorption
        dst.absorption_color[0] = src.absorption_color.x;
        dst.absorption_color[1] = src.absorption_color.y;
        dst.absorption_color[2] = src.absorption_color.z;
        dst.absorption_coefficient = src.absorption_coefficient;

        // Emission
        dst.emission_color[0] = src.emission_color.x;
        dst.emission_color[1] = src.emission_color.y;
        dst.emission_color[2] = src.emission_color.z;
        dst.emission_intensity = src.emission_intensity;

        // Emission mode + blackbody/color-ramp (matches shader extension block)
        dst.emission_mode       = src.emission_mode;
        dst.temperature_scale   = src.temperature_scale;
        dst.blackbody_intensity = src.blackbody_intensity;
        dst.max_temperature     = src.max_temperature;
        dst.color_ramp_enabled  = src.color_ramp_enabled;
        dst.ramp_stop_count     = std::min(src.ramp_stop_count, 8);
        for (int j = 0; j < dst.ramp_stop_count; ++j) {
            dst.ramp_positions[j] = src.ramp_positions[j];
            dst.ramp_colors_r[j]  = src.ramp_colors[j].x;
            dst.ramp_colors_g[j]  = src.ramp_colors[j].y;
            dst.ramp_colors_b[j]  = src.ramp_colors[j].z;
        }
        // OptiX parity: if temperature grid is missing in blackbody/channel mode,
        // fall back to density grid as a scalar source for ramp/blackbody mapping.
        if (dst.vdb_temp_address == 0 && dst.vdb_grid_address != 0 && dst.emission_mode >= 2) {
            dst.vdb_temp_address = dst.vdb_grid_address;
        }

        // Ray march
        dst.step_size = src.step_size;
        dst.max_steps = src.max_steps;
        dst.shadow_steps = src.shadow_steps;
        dst.shadow_strength = src.shadow_strength;

        // Flags
        // volume_type = 3 is an explicit procedural cloud source. Otherwise use
        // NanoVDB when uploaded, with the existing procedural-noise fallback.
        dst.volume_type = (src.source_type == 3) ? 3 : ((dst.vdb_grid_address != 0) ? 2 : 1);
        dst.is_active = 1;
        dst.voxel_size = src.voxel_size;
    }

    m_device->updateVolumeBuffer(instances.data(),
                                  instances.size() * sizeof(VulkanRT::VkVolumeInstance),
                                  (uint32_t)instances.size());

    // ── TLAS transform refresh ──────────────────────────────────────────────
    // When a unified volume is moved with the gizmo, setTransform() updates the C++ object
    // but the TLAS AABB instance transform remains stale.  Fix: recompute the
    // scale+translate transform from the current worldBounds for every volume
    // instance found in m_instanceSources and push an updateTLAS call.
    {
        bool tlas_changed = false;
        for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
            Vec3 worldMin;
            Vec3 worldMax;
            if (auto vdb = std::dynamic_pointer_cast<VDBVolume>(m_instanceSources[i])) {
                AABB wb = vdb->getWorldBounds();
                worldMin = wb.min;
                worldMax = wb.max;
            } else if (auto gas = std::dynamic_pointer_cast<GasVolume>(m_instanceSources[i])) {
                gas->getWorldBounds(worldMin, worldMax);
            } else {
                continue;
            }
            Vec3 center = (worldMin + worldMax) * 0.5f;
            Vec3 sz(worldMax.x - worldMin.x, worldMax.y - worldMin.y, worldMax.z - worldMin.z);
            if (sz.x < 1e-4f) sz.x = 1e-4f;
            if (sz.y < 1e-4f) sz.y = 1e-4f;
            if (sz.z < 1e-4f) sz.z = 1e-4f;
            Matrix4x4 newT = Matrix4x4::translation(center) * Matrix4x4::scaling(sz);
            if (!(newT == m_vkInstances[i].transform)) {
                m_vkInstances[i].transform = newT;
                tlas_changed = true;
            }
        }
        if (tlas_changed) {
            m_device->waitIdle();
            auto merged = m_vkInstances;
            for (const auto& h : m_hairVkInstances) merged.push_back(h);
            m_device->updateTLAS(merged);
        }
    }

    // VK_INFO() << "[VulkanBackendAdapter] Uploaded " << instances.size() << " VDB volume(s) to Vulkan SSBO." << std::endl;
    resetAccumulation();
}

void VulkanBackendAdapter::updateGasVolumes(const std::vector<GpuGasVolume>& vols) {
    // Gas volumes use similar conversion — for now, handled as basic homogeneous volumes
    if (!m_device || vols.empty()) return;

    std::vector<VulkanRT::VkVolumeInstance> instances(vols.size());
    for (size_t i = 0; i < vols.size(); i++) {
        const auto& src = vols[i];
        auto& dst = instances[i];
        memset(&dst, 0, sizeof(dst));

        for (int i = 0; i < 12; ++i) {
            dst.transform[i]     = src.transform[i];
            dst.inv_transform[i] = src.inv_transform[i];
        }
        
        // GasVolume does not have pivot tracking, default to 0
        dst.pivot_offset[0] = 0.0f;
        dst.pivot_offset[1] = 0.0f;
        dst.pivot_offset[2] = 0.0f;

        // Use local bounding box for accurate containment check of localPos
        dst.aabb_min[0] = src.local_bbox_min.x; dst.aabb_min[1] = src.local_bbox_min.y; dst.aabb_min[2] = src.local_bbox_min.z;
        dst.aabb_max[0] = src.local_bbox_max.x; dst.aabb_max[1] = src.local_bbox_max.y; dst.aabb_max[2] = src.local_bbox_max.z;

        dst.density_multiplier = src.density_multiplier;
        dst.density_remap_low = src.density_remap_low;
        dst.density_remap_high = src.density_remap_high;
        dst.noise_scale = 1.0f;
        dst._reserved[0] = (src.density_pad > 0.0f) ? src.density_pad : 0.04f;
        dst._reserved[1] = src.emission_pad;

        dst.scatter_color[0] = src.scatter_color.x;
        dst.scatter_color[1] = src.scatter_color.y;
        dst.scatter_color[2] = src.scatter_color.z;
        dst.scatter_coefficient = src.scatter_coefficient;
        dst.scatter_anisotropy = src.scatter_anisotropy;

        dst.absorption_color[0] = src.absorption_color.x;
        dst.absorption_color[1] = src.absorption_color.y;
        dst.absorption_color[2] = src.absorption_color.z;
        dst.absorption_coefficient = src.absorption_coefficient;

        dst.emission_color[0] = src.emission_color.x;
        dst.emission_color[1] = src.emission_color.y;
        dst.emission_color[2] = src.emission_color.z;
        dst.emission_intensity = src.emission_intensity;

        dst.step_size = src.step_size;
        dst.max_steps = src.max_steps;
        dst.shadow_steps = src.shadow_steps;
        dst.shadow_strength = src.shadow_strength;

        dst.volume_type = 0; // Homogeneous
        dst.is_active = 1;
        dst.voxel_size = src.step_size; // GpuGasVolume has no voxel_size; approximate with step_size
    }

    // Append to existing volume buffer (after VDB volumes)
    // For now, only gas volumes if no VDB volumes exist
    if (m_device->m_volumeCount == 0) {
        m_device->updateVolumeBuffer(instances.data(),
                                      instances.size() * sizeof(VulkanRT::VkVolumeInstance),
                                      (uint32_t)instances.size());
    }
    resetAccumulation();
}

// Utility
void VulkanBackendAdapter::waitForCompletion() { m_device->waitIdle(); }
void VulkanBackendAdapter::resetAccumulation() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // [PERF] Skip GPU image clears if already at sample 0 — images were already
    // cleared by a prior resetAccumulation() this frame.  This eliminates 10-25
    // redundant synchronous GPU round-trips during camera movement.
    const bool needsImageClear = (m_currentSamples > 0);
    m_currentSamples = 0;
    m_hasPresentedRenderedFrame = false;
    m_interactiveViewport.dirty = true;
    if (needsImageClear && m_device) {
        std::vector<VulkanRT::VulkanDevice::ImageClearRequest> clears;
        if (m_outputImage.image)        clears.push_back({&m_outputImage, 0,0,0,0});
        if (m_varianceImage.image)      clears.push_back({&m_varianceImage, 0,0,0,0});
        if (m_denoiserColorImage.image) clears.push_back({&m_denoiserColorImage, 0,0,0,0});
        if (m_denoiserAlbedoImage.image) clears.push_back({&m_denoiserAlbedoImage, 0,0,0,0});
        if (m_denoiserNormalImage.image) clears.push_back({&m_denoiserNormalImage, 0.5f,0.5f,0.5f,0});
        if (m_denoiserPositionImage.image) clears.push_back({&m_denoiserPositionImage, 0,0,0,0});
        m_device->clearImages(clears);
        m_imagesCleared = true;
    }
    m_forceClearOnNextPresent = true;
}
float VulkanBackendAdapter::getMillisecondsPerSample() const { return 0.0f; }

} // namespace Backend