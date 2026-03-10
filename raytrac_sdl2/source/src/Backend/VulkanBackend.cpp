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
#include "globals.h"
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <functional>
#include "HittableInstance.h"
#include "HittableList.h"
#include "ParallelBVHNode.h"
#include "Triangle.h"
#include "VDBVolume.h"
#include "GasVolume.h"
#include "VDBVolumeManager.h"
#include "params.h"  // GpuVDBVolume, GpuGasVolume definitions
#include <SDL.h>
#include "stb_image_write.h"
#include "Texture.h"
#include "World.h"
#include "AtmosphereLUT.h"
#include "CameraPresets.h"
#include "Camera.h"
#include "DllLoadPolicy.h"

// Delay-load handler: attempt to LoadLibrary when a delay-loaded DLL fails
#include <windows.h>
#include <delayimp.h>

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
// Debug Callback & Logging
// ============================================================================

#include <sstream>
#include <SpotLight.h>

namespace {
    class VulkanLogHelper {
        std::ostringstream ss;
        LogLevel level;
    public:
        VulkanLogHelper(LogLevel l) : level(l) {}
        ~VulkanLogHelper() { g_sceneLog.add(ss.str(), level); }
        
        template<typename T>
        VulkanLogHelper& operator<<(const T& t) { ss << t; return *this; }
        
        // Ignore std::endl
        VulkanLogHelper& operator<<(std::ostream& (*)(std::ostream&)) { return *this; }
    };
}

#define VK_INFO() VulkanLogHelper(LogLevel::Info)
#define VK_ERROR() VulkanLogHelper(LogLevel::Error)
#define VK_WARN() VulkanLogHelper(LogLevel::Warning)

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
            safeDestroy(blas.vertexBuffer);
            safeDestroy(blas.normalBuffer);
            safeDestroy(blas.uvBuffer);
            safeDestroy(blas.indexBuffer);
            safeDestroy(blas.materialIndexBuffer);
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

        // Destroy RT pipeline
        if (m_rtPipeline) vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
        if (m_rtPipelineLayout) vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
        if (m_rtDescriptorSetLayout) vkDestroyDescriptorSetLayout(m_device, m_rtDescriptorSetLayout, nullptr);
        
        destroyBuffer(m_sbtBuffer);
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
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            m_computeQueueFamily = i;
            // Prefer dedicated compute if available
            if (!(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
        }
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

    // Features chain
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures{};
    bdaFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bdaFeatures.bufferDeviceAddress = hasBDA ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{};
    accelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceDescriptorIndexingFeatures descIdxFeatures{};
    descIdxFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    descIdxFeatures.runtimeDescriptorArray = VK_TRUE;
    descIdxFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

    // Build pNext chain
    features2.pNext = &bdaFeatures;
    if (m_capabilities.rtMode == RayTracingMode::HARDWARE_KHR) {
        bdaFeatures.pNext = &accelFeatures;
        accelFeatures.pNext = &rtPipelineFeatures;
        rtPipelineFeatures.pNext = hasDescIdx ? (void*)&descIdxFeatures : nullptr;
    } else {
        bdaFeatures.pNext = hasDescIdx ? (void*)&descIdxFeatures : nullptr;
    }

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = &features2;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    VkResult result = vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device);
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
        // Fallback succeeded — mark capabilities conservatively
        m_capabilities.rtMode = RayTracingMode::COMPUTE;
        VK_INFO() << "[VulkanDevice] Device created with fallback (no HW RT). Continuing in compute mode." << std::endl;
    }

    vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_computeQueue);
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
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     1024 },
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
    if (buffer.buffer) vkDestroyBuffer(m_device, buffer.buffer, nullptr);
    if (buffer.memory) vkFreeMemory(m_device, buffer.memory, nullptr);
    buffer = {};
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
    void* mapped;
    vkMapMemory(m_device, src.memory, offset, size, 0, &mapped);
    memcpy(data, mapped, size);
    vkUnmapMemory(m_device, src.memory);
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
    if (!info.vertexData || info.vertexCount == 0) return UINT32_MAX;

    // --- 1) Combine all geometry data into ONE GPU buffer to reduce allocation count ---
    // This is critical for large scenes to avoid hitting maxMemoryAllocationCount limits.
    uint64_t vertSize = (uint64_t)info.vertexCount * info.vertexStride;
    uint64_t normSize = info.normalData ? (uint64_t)info.vertexCount * sizeof(float) * 3 : 0;
    uint64_t uvSize   = info.uvData ? (uint64_t)info.vertexCount * sizeof(float) * 2 : 0;
    
    bool hasIndices = (info.indexData && info.indexCount > 0);
    uint64_t idxSize = hasIndices ? (uint64_t)info.indexCount * sizeof(uint32_t) : 0;
    
    bool hasMaterials = (info.materialIndexData && info.materialIndexCount > 0);
    uint64_t matSize = hasMaterials ? (uint64_t)info.materialIndexCount * sizeof(uint32_t) : 0;

    uint64_t totalGeomSize = vertSize + normSize + uvSize + idxSize + matSize;

    BufferCreateInfo geomBufInfo;
    geomBufInfo.size = totalGeomSize;
    geomBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::VERTEX;
    geomBufInfo.location = MemoryLocation::GPU_ONLY;
    geomBufInfo.initialData = nullptr; 

    auto geometryBuffer = createBuffer(geomBufInfo);
    if (!geometryBuffer.buffer) {
        VK_ERROR() << "[VulkanDevice] Failed to allocate combined geometry buffer for BLAS" << std::endl;
        return UINT32_MAX;
    }

    // Upload geometry via staging path (keeps persistent buffer device-local).
    uint64_t off = 0;
    uploadBuffer(geometryBuffer, info.vertexData, vertSize, off); off += vertSize;
    if (normSize && info.normalData) { uploadBuffer(geometryBuffer, info.normalData, normSize, off); off += normSize; }
    if (uvSize && info.uvData)       { uploadBuffer(geometryBuffer, info.uvData, uvSize, off); off += uvSize; }
    if (idxSize && info.indexData)   { uploadBuffer(geometryBuffer, info.indexData, idxSize, off); off += idxSize; }
    if (matSize && info.materialIndexData) { uploadBuffer(geometryBuffer, info.materialIndexData, matSize, off); }
    
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
    triangles.vertexData.deviceAddress = geometryBuffer.deviceAddress;
    triangles.vertexStride = info.vertexStride;
    triangles.maxVertex = info.vertexCount - 1;
    if (hasIndices) {
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = geometryBuffer.deviceAddress + vertSize + normSize + uvSize;
    } else {
        triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = 0; // Not opaque — shadow_anyhit tests opacity per pixel
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
    
    // Store attribute buffer segments in handle for shader access (binding 4)
    // All point to the same VkBuffer but with staggered deviceAddresses.
    blasHandle.vertexBuffer = geometryBuffer;
    blasHandle.vertexBuffer.deviceAddress = geometryBuffer.deviceAddress;

    if (normSize) {
        blasHandle.normalBuffer = geometryBuffer;
        blasHandle.normalBuffer.deviceAddress = geometryBuffer.deviceAddress + vertSize;
    }
    if (uvSize) {
        blasHandle.uvBuffer = geometryBuffer;
        blasHandle.uvBuffer.deviceAddress = geometryBuffer.deviceAddress + vertSize + normSize;
    }
    if (idxSize) {
        blasHandle.indexBuffer = geometryBuffer;
        blasHandle.indexBuffer.deviceAddress = geometryBuffer.deviceAddress + vertSize + normSize + uvSize;
    }
    if (matSize) {
        blasHandle.materialIndexBuffer = geometryBuffer;
        blasHandle.materialIndexBuffer.deviceAddress = geometryBuffer.deviceAddress + vertSize + normSize + uvSize + idxSize;
    }
    
    // Store skinning buffers
    blasHandle.hasSkinning = info.hasSkinning;
    blasHandle.vertexCount = info.vertexCount;
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

void VulkanDevice::updateBLAS(uint32_t blasIndex, const float* newVertices) {
    if (!hasHardwareRT() || !fpCmdBuildAccelerationStructuresKHR) return;
    if (blasIndex >= m_blasList.size()) return;
    
    AccelStructHandle& blasHandle = m_blasList[blasIndex];
    if (blasHandle.accel == VK_NULL_HANDLE || !blasHandle.hasSkinning) return;

    if (newVertices) {
        uploadBuffer(blasHandle.vertexBuffer, newVertices, (uint64_t)blasHandle.vertexCount * 12);
    }

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = blasHandle.vertexBuffer.deviceAddress;
    triangles.vertexStride = 12;
    triangles.maxVertex = blasHandle.vertexCount - 1;
    triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = 0;
    geometry.geometry.triangles = triangles;

    uint32_t primitiveCount = blasHandle.vertexCount / 3;

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

    // Free previous instance data buffer now (we will replace it with the new one)
    if (m_tlasInstanceBuffer.buffer) {
        destroyBuffer(m_tlasInstanceBuffer);
    }

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
        dst.accelerationStructureReference = m_blasList[src.blasIndex].deviceAddress;

        vkInstances.push_back(dst);
    }

    if (vkInstances.empty()) {
        VK_WARN() << "[VulkanDevice] createTLAS: No valid instances provided." << std::endl;
        return;
    }

    // --- 2) Upload instance data to GPU ---
    BufferCreateInfo instBufInfo;
    instBufInfo.size = vkInstances.size() * sizeof(VkAccelerationStructureInstanceKHR);
    instBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    instBufInfo.location = MemoryLocation::CPU_TO_GPU;
    instBufInfo.initialData = vkInstances.data();
    auto instanceBuffer = createBuffer(instBufInfo);

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

    // --- 6) Scratch buffer ---
    uint64_t scratchAlignment = m_capabilities.minScratchAlignment > 0 ? m_capabilities.minScratchAlignment : 128;
    uint64_t scratchSize = performUpdate ? sizeInfo.updateScratchSize : sizeInfo.buildScratchSize;
    uint64_t alignedScratchSize = (scratchSize + scratchAlignment - 1) & ~(scratchAlignment - 1);

    BufferCreateInfo scratchBufInfo;
    scratchBufInfo.size = alignedScratchSize;
    scratchBufInfo.usage = BufferUsage::STORAGE;
    scratchBufInfo.location = MemoryLocation::GPU_ONLY;
    auto scratchBuffer = createBuffer(scratchBufInfo);
    if (!scratchBuffer.buffer) {
        destroyBuffer(instanceBuffer);
        return;
    }

    // --- 7) Build TLAS ---
    buildInfo.dstAccelerationStructure = m_tlas.accel;
    buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        destroyBuffer(scratchBuffer);
        destroyBuffer(instanceBuffer);
        return;
    }
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    endSingleTimeCommands(cmd);

    destroyBuffer(scratchBuffer);
    
    // Store it in the device member `m_tlasInstanceBuffer` (ownership moved here).
    m_tlasInstanceBuffer = instanceBuffer;

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

    VK_INFO() << "[VulkanDevice] TLAS " << (performUpdate ? "updated" : "created") << " (" << instanceCount << " instances)" << std::endl;
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
                                     const std::vector<std::uint32_t>& hairAnyHitSPV) {
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

    bool hasVolume     = (volChitModule  != VK_NULL_HANDLE && volIntModule  != VK_NULL_HANDLE);
    bool hasHair       = (hairChitModule != VK_NULL_HANDLE && hairIntModule != VK_NULL_HANDLE);
    bool hasShadowMiss = (shadowMissModule != VK_NULL_HANDLE);

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
    VkDescriptorSetLayoutBinding bindings[16] = {};
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
    bindings[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

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
    bindings[6].descriptorCount = 1024; // runtime array capacity
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

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = 16;
    dslCI.pBindings = bindings;
    vkCreateDescriptorSetLayout(m_device, &dslCI, nullptr, &m_rtDescriptorSetLayout);

    // --- 5) Push constant range (camera data + rendering params) ---
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    pushRange.offset = 0;
    pushRange.size = 128; // Headroom for camera data and rendering params

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

        // Rebuild groups with corrected stage indices
        groups.clear();
        uint32_t r2RaygenGroup = (uint32_t)groups.size(); groups.push_back(makeGeneralGroup(s2RaygenIdx));
        uint32_t r2MissGroup   = (uint32_t)groups.size(); groups.push_back(makeGeneralGroup(s2PrimaryMissIdx));
        if (hasShadowMiss) { groups.push_back(makeGeneralGroup(s2ShadowMissIdx)); }
        triHitGroupIdx = (uint32_t)groups.size();
        { VkRayTracingShaderGroupCreateInfoKHR g{}; g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR; g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR; g.generalShader = VK_SHADER_UNUSED_KHR; g.closestHitShader = s2ChitIdx; g.anyHitShader = s2AnyHitIdx; g.intersectionShader = VK_SHADER_UNUSED_KHR; groups.push_back(g); }
        if (hasVolume) { volHitGroupIdx = (uint32_t)groups.size(); VkRayTracingShaderGroupCreateInfoKHR g{}; g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR; g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR; g.generalShader = VK_SHADER_UNUSED_KHR; g.closestHitShader = s2VolChitIdx; g.anyHitShader = VK_SHADER_UNUSED_KHR; g.intersectionShader = s2VolIntIdx; groups.push_back(g); }
        if (hasHair)   { hairHitGroupIdx = (uint32_t)groups.size(); VkRayTracingShaderGroupCreateInfoKHR g{}; g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR; g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR; g.generalShader = VK_SHADER_UNUSED_KHR; g.closestHitShader = s2HairChitIdx; g.anyHitShader = VK_SHADER_UNUSED_KHR; g.intersectionShader = s2HairIntIdx; groups.push_back(g); }
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

    if (result != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] vkCreateRayTracingPipelinesKHR failed: " << result << std::endl;
        return false;
    }

    // --- 8) Build Shader Binding Table (SBT) ---
    // Store hair/shadow/volume state as members for consistent use during rendering
    m_hasVolumeShaders = hasVolume;
    m_hasHairShaders   = hasHair;
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

    // Hit region: triangle + optional volume + optional hair (contiguous)
    uint32_t numHitGroups = 1u + (hasVolume ? 1u : 0u) + (hasHair ? 1u : 0u);
    m_sbtHitRegion.deviceAddress = sbtAddr + (VkDeviceAddress)triHitGroupIdx * alignedHandleSize;
    m_sbtHitRegion.stride = alignedHandleSize;
    m_sbtHitRegion.size   = (VkDeviceSize)numHitGroups * alignedHandleSize;

    m_sbtCallableRegion = {}; // No callable shaders

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
                                     const ImageHandle* denoiserNormalImage) {
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
    writes.reserve(11);

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
            if (slot >= 1024) continue;
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
    if (slot >= 1024) {
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
    VK_INFO() << "[VulkanDevice] updateVolumeBuffer - " << count << " volume instances uploaded (" << size << " bytes)" << std::endl;
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
    VK_INFO() << "[VulkanDevice] updateTerrainLayerBuffer - " << count << " terrain layers uploaded (" << size << " bytes)" << std::endl;
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

ImageHandle VulkanDevice::createImage2D(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage) {
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
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
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

void VulkanDevice::traceRaysAndReadback(uint32_t w, uint32_t h,
    const ImageHandle& outputImage, const BufferHandle& stagingBuffer) {
    if (!m_rtPipelineReady || !fpCmdTraceRaysKHR || !m_tlas.accel) return;
    if (!outputImage.image || !stagingBuffer.buffer) return;
    if (outputImage.width == 0 || outputImage.height == 0) return;

    const uint64_t bytesPerPixel = (outputImage.format == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8ull : 16ull;
    const uint64_t requiredBytes = (uint64_t)outputImage.width * (uint64_t)outputImage.height * bytesPerPixel;
    if (stagingBuffer.size < requiredBytes) {
        SCENE_LOG_WARN("[Vulkan] traceRaysAndReadback skipped: staging buffer too small for output image.");
        return;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return;

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

namespace Backend {

VulkanBackendAdapter::VulkanBackendAdapter()
    : m_device(std::make_unique<VulkanRT::VulkanDevice>()) {
    m_targetSamples = 1000; // Default
}

VulkanBackendAdapter::~VulkanBackendAdapter() {
    shutdown();
}

bool VulkanBackendAdapter::initialize() {
#ifdef _DEBUG
    bool validation = true;
#else
    bool validation = false;
#endif
    bool ok = m_device->initialize(true, validation);

    if (ok && !m_device->hasHardwareRT()) {
        VK_ERROR() << "[VulkanBackendAdapter] Hardware RT not supported by this GPU, Vulkan backend disabled." << std::endl;
        m_device->shutdown();
        return false;
    }

    if (ok && !m_cachedLights.empty()) {
        VK_INFO() << "[VulkanBackendAdapter] Uploading cached lights after device init (" << m_cachedLights.size() << ")" << std::endl;
        setLights(m_cachedLights);
        m_cachedLights.clear();
    }
    return ok;
}

void VulkanBackendAdapter::purgeUploadedTextureCacheLocked() {
    if (!m_device) return;
    m_device->waitIdle();
    m_device->clearPendingRTTextureDescriptors();
    for (auto& [id, img] : m_uploadedImages) {
        (void)id;
        if (img.image || img.view || img.memory || img.sampler) {
            m_device->destroyImage(img);
        }
    }
    m_uploadedImages.clear();
    m_uploadedImageIDs.clear();
    m_nextTextureID = 1;
}

void VulkanBackendAdapter::shutdown() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    if (!m_device) {
        return;
    }

    if (m_device->isInitialized()) {
        m_device->waitIdle();
    }

    // Adapter-owned output/readback resources.
    if (m_outputImage.image) {
        m_device->destroyImage(m_outputImage);
    }
    if (m_stagingBuffer.buffer) {
        m_device->destroyBuffer(m_stagingBuffer);
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
    if (m_denoiserColorStagingBuffer.buffer) {
        m_device->destroyBuffer(m_denoiserColorStagingBuffer);
    }
    if (m_denoiserAlbedoStagingBuffer.buffer) {
        m_device->destroyBuffer(m_denoiserAlbedoStagingBuffer);
    }
    if (m_denoiserNormalStagingBuffer.buffer) {
        m_device->destroyBuffer(m_denoiserNormalStagingBuffer);
    }

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
    m_orderedVDBInstances.clear();
    m_meshRegistry.clear();
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
    m_currentSamples = 0;
    m_testInitialized = false;
    m_hdrPixels.clear();
    m_denoiserColorPixels.clear();
    m_denoiserAlbedoPixels.clear();
    m_denoiserNormalPixels.clear();

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
    const auto& caps = m_device->getCapabilities();
    info.deviceName = caps.deviceName;
    info.hasHardwareRT = m_device->hasHardwareRT();
    info.vramBytes = caps.dedicatedVRAM;
    info.driverVersion = std::to_string(caps.driverVersion);
    return info;
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
        if (t.hasSkinData) { hasSkinning = true; break; }
    }
    
    positions.reserve(triangles.size() * 9);
    normals.reserve(triangles.size() * 9);
    uvs.reserve(triangles.size() * 6);
    if (hasSkinning) {
        boneIndices.reserve(triangles.size() * 12);
        boneWeights.reserve(triangles.size() * 12);
    }

    for (const auto& t : triangles) {
        positions.push_back(t.v0.x); positions.push_back(t.v0.y); positions.push_back(t.v0.z);
        positions.push_back(t.v1.x); positions.push_back(t.v1.y); positions.push_back(t.v1.z);
        positions.push_back(t.v2.x); positions.push_back(t.v2.y); positions.push_back(t.v2.z);

        normals.push_back(t.n0.x); normals.push_back(t.n0.y); normals.push_back(t.n0.z);
        normals.push_back(t.n1.x); normals.push_back(t.n1.y); normals.push_back(t.n1.z);
        normals.push_back(t.n2.x); normals.push_back(t.n2.y); normals.push_back(t.n2.z);

        uvs.push_back(t.uv0.x); uvs.push_back(t.uv0.y);
        uvs.push_back(t.uv1.x); uvs.push_back(t.uv1.y);
        uvs.push_back(t.uv2.x); uvs.push_back(t.uv2.y);
        
        if (hasSkinning) {
            for(int i=0; i<4; ++i) { boneIndices.push_back(t.hasSkinData ? t.boneIndices_v0[i] : -1); boneWeights.push_back(t.hasSkinData ? t.boneWeights_v0[i] : 0.0f); }
            for(int i=0; i<4; ++i) { boneIndices.push_back(t.hasSkinData ? t.boneIndices_v1[i] : -1); boneWeights.push_back(t.hasSkinData ? t.boneWeights_v1[i] : 0.0f); }
            for(int i=0; i<4; ++i) { boneIndices.push_back(t.hasSkinData ? t.boneIndices_v2[i] : -1); boneWeights.push_back(t.hasSkinData ? t.boneWeights_v2[i] : 0.0f); }
        }
    }

    // Per-primitive material indices (one per triangle)
    std::vector<uint32_t> materialIndices;
    materialIndices.reserve(triangles.size());
    for (const auto& t : triangles) {
        uint16_t mId = t.materialID;
        if (mId == MaterialManager::INVALID_MATERIAL_ID) mId = 0; 
        materialIndices.push_back((uint32_t)mId);
    }

    VulkanRT::BLASCreateInfo blasInfo;
    blasInfo.vertexData = positions.data();
    blasInfo.normalData = normals.data();
    blasInfo.uvData = uvs.data();
    blasInfo.vertexCount = (uint32_t)triangles.size() * 3;
    blasInfo.vertexStride = 12; // 3 * float
    blasInfo.materialIndexData = materialIndices.data();
    blasInfo.materialIndexCount = (uint32_t)materialIndices.size();
    
    blasInfo.hasSkinning = hasSkinning;
    blasInfo.allowUpdate = hasSkinning;
    blasInfo.boneIndicesData = hasSkinning ? boneIndices.data() : nullptr;
    blasInfo.boneWeightsData = hasSkinning ? boneWeights.data() : nullptr;
    
    uint32_t blasIndex = m_device->createBLAS(blasInfo);
    if (blasIndex == UINT32_MAX) {
        SCENE_LOG_ERROR("[Vulkan] Failed to create BLAS for mesh: " + meshName);
        return UINT32_MAX;
    }

    m_meshRegistry[meshName] = blasIndex;

    // Reset geometry data buffer because a new BLAS was added
    if (m_device->m_geometryDataBuffer.buffer) {
        m_device->destroyBuffer(m_device->m_geometryDataBuffer);
    }
    
   // SCENE_LOG_INFO("[Vulkan] Uploaded mesh: " + meshName + " (" + std::to_string(triangles.size()) + " tris)");
    return blasIndex;
}

void VulkanBackendAdapter::clearHairGeometry(bool rebuild_tlas) {
    // Destroy all hair BLASes that were appended after the mesh BLASes.
    // Hair BLASes always live at m_blasList[m_meshBlasCount .. end).
    if (m_device && m_device->isInitialized() && !m_hairVkInstances.empty()) {
        // Wait for the GPU to finish any in-flight work that might be using these BLASes.
        vkDeviceWaitIdle(m_device->m_device);

        for (uint32_t i = m_meshBlasCount; i < (uint32_t)m_device->m_blasList.size(); ++i) {
            auto& blas = m_device->m_blasList[i];
            if (blas.accel && m_device->fpDestroyAccelerationStructureKHR) {
                m_device->fpDestroyAccelerationStructureKHR(m_device->m_device, blas.accel, nullptr);
            }
            m_device->destroyBuffer(blas.buffer);
            m_device->destroyBuffer(blas.vertexBuffer); // stores the AABB buffer for hair
        }
        if ((uint32_t)m_device->m_blasList.size() > m_meshBlasCount) {
            m_device->m_blasList.resize(m_meshBlasCount);
        }

        // Invalidate geometry data buffer so it gets rebuilt on the next
        // bindRTDescriptors call with the updated (mesh-only + fresh hair) BLAS list.
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

    // Deduplicate: if this groom was already uploaded, return cached handle
    auto it = m_hairGroomRegistry.find(groomName);
    if (it != m_hairGroomRegistry.end()) return it->second;

    // Convert all strands into B-spline segments: N points → N-3 cubic segments
    std::vector<VulkanRT::HairSegmentGPU> segments;
    std::vector<VkAabbPositionsKHR>       aabbs;
    uint32_t strandIdx = 0;

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
            seg.materialID = strand.materialID;
            seg.padding    = 0;

            // Conservative AABB: bounding box of 4 control points ± max radius
            float maxR = std::max({seg.cp0[3], seg.cp1[3], seg.cp2[3], seg.cp3[3]});
            float minX = std::min({seg.cp0[0], seg.cp1[0], seg.cp2[0], seg.cp3[0]}) - maxR;
            float minY = std::min({seg.cp0[1], seg.cp1[1], seg.cp2[1], seg.cp3[1]}) - maxR;
            float minZ = std::min({seg.cp0[2], seg.cp1[2], seg.cp2[2], seg.cp3[2]}) - maxR;
            float maxX = std::max({seg.cp0[0], seg.cp1[0], seg.cp2[0], seg.cp3[0]}) + maxR;
            float maxY = std::max({seg.cp0[1], seg.cp1[1], seg.cp2[1], seg.cp3[1]}) + maxR;
            float maxZ = std::max({seg.cp0[2], seg.cp1[2], seg.cp2[2], seg.cp3[2]}) + maxR;

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
    m_hairGroomRegistry[groomName] = groomHandle;

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
    m_vkInstances.clear();
    m_instanceSources.clear();
    m_instance_sync_cache.clear();
    m_hairVkInstances.clear();
    m_hairGroomRegistry.clear();
    m_meshBlasCount = 0; // Reset: all existing BLASes will be destroyed below
    m_topology_dirty = true;
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
        
        if (m_device->m_geometryDataBuffer.buffer) m_device->destroyBuffer(m_device->m_geometryDataBuffer);

        // Destroy existing TLAS
        if (m_device->m_tlas.accel && m_device->fpDestroyAccelerationStructureKHR) {
             m_device->fpDestroyAccelerationStructureKHR(m_device->m_device, m_device->m_tlas.accel, nullptr);
             m_device->m_tlas.accel = VK_NULL_HANDLE;
        }
        m_device->destroyBuffer(m_device->m_tlas.buffer);

        // Clear uploaded image/texture cache from previous scene.
        // This is important for New/Open project workflows so old scene textures
        // do not keep consuming VRAM until backend switch/shutdown.
        m_device->clearPendingRTTextureDescriptors();
        for (auto& [id, img] : m_uploadedImages) {
            (void)id;
            if (img.image || img.view || img.memory || img.sampler) {
                m_device->destroyImage(img);
            }
        }
        m_uploadedImages.clear();
        m_uploadedImageIDs.clear();
        m_nextTextureID = 1;

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

        // 2. Refit TLAS with current instance list (no full rebuild, no waitIdle)
        auto merged = m_vkInstances;
        for (const auto& h : m_hairVkInstances) merged.push_back(h);
        m_device->updateTLAS(merged);

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
}

void VulkanBackendAdapter::updateInstanceMaterialBinding(const std::string& n, int o, int nw) { (void)n; (void)o; (void)nw; }
void VulkanBackendAdapter::setVisibilityByNodeName(const std::string& nodeName, bool visible) {
    if (!m_device || !m_device->isInitialized() || m_instanceSources.empty()) return;

    bool changed = false;
    for (size_t i = 0; i < m_instanceSources.size(); ++i) {
        if (!m_instanceSources[i]) continue;

        std::string instName;
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            instName = inst->node_name;
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            instName = tri->getNodeName();
        }

        if (matchesNodeNameForInstance(instName, nodeName)) {
            uint8_t newMask = visible ? 0xFF : 0x00;
            if (m_vkInstances[i].mask != newMask) {
                m_vkInstances[i].mask = newMask;
                changed = true;
            }
        }
    }

    if (changed) {
        // [VULKAN FIX] Update the TLAS with new visibility masks (include hair)
        auto mergedVis = m_vkInstances;
        for (const auto& h : m_hairVkInstances) mergedVis.push_back(h);
        m_device->updateTLAS(mergedVis);
        resetAccumulation();
    }
}

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

    struct SoloTriangleGroup {
        std::string nodeName;
        std::vector<TriangleData> triangles;
        Matrix4x4 transform;
        uint16_t materialID = 0;
        std::shared_ptr<Hittable> representative;
    };
    std::vector<SoloTriangleGroup> soloGroups;
    std::unordered_map<void*, size_t> soloGroupByTransform;
    
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
                        d.v0 = t->getV0(); d.v1 = t->getV1(); d.v2 = t->getV2();
                        d.n0 = t->getN0(); d.n1 = t->getN1(); d.n2 = t->getN2();
                        auto uv = t->getUVCoordinates();
                        d.uv0 = std::get<0>(uv); d.uv1 = std::get<1>(uv); d.uv2 = std::get<2>(uv);
                        d.materialID = t->getMaterialID();
                        if (d.materialID == MaterialManager::INVALID_MATERIAL_ID) d.materialID = 0;
                        
                        d.hasSkinData = t->hasSkinData();
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
            TriangleData d;
            // IMPORTANT: BLAS geometry must stay in local space.
            // World/object transform is carried by TLAS instance transform.
            d.v0 = tri->getOriginalVertexPosition(0);
            d.v1 = tri->getOriginalVertexPosition(1);
            d.v2 = tri->getOriginalVertexPosition(2);
            d.n0 = tri->getOriginalVertexNormal(0);
            d.n1 = tri->getOriginalVertexNormal(1);
            d.n2 = tri->getOriginalVertexNormal(2);
            auto uv = tri->getUVCoordinates();
            d.uv0 = std::get<0>(uv); d.uv1 = std::get<1>(uv); d.uv2 = std::get<2>(uv);
            d.materialID = tri->getMaterialID();

            // [SKINNING FIX] Copy bone weights into TriangleData so the BLAS is created
            // with hasSkinning=true and dispatchSkinning() can deform it each frame.
            d.hasSkinData = tri->hasSkinData();
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
            auto triTransformHandle = tri->getTransformHandle();
            if (triTransformHandle) {
                groupKey = triTransformHandle.get();
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
                group.transform = tri->getTransformMatrix();
                group.materialID = tri->getMaterialID();
                group.representative = tri;
                soloGroups.push_back(std::move(group));
                found = soloGroupByTransform.emplace(groupKey, soloGroups.size() - 1).first;
            }

            auto& targetGroup = soloGroups[found->second];
            targetGroup.triangles.push_back(d);
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
    };

    for (const auto& obj : objects) {
        processObj(obj);
    }
    
    // Store for updateInstanceTransforms
    m_lastObjects = objects;

    // Handle Solo Triangles: one BLAS/TLAS instance per object-transform group.
    if (!soloGroups.empty()) {
        for (size_t groupIndex = 0; groupIndex < soloGroups.size(); ++groupIndex) {
            const auto& group = soloGroups[groupIndex];
            if (group.triangles.empty()) continue;

            std::string meshKey = "[World-Solo]-" + group.nodeName + "-" + std::to_string(groupIndex);
            uint32_t soloBlasIndex = uploadTriangles(group.triangles, meshKey);

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

        // Upload instance data SSBO (Binding 5)
        std::vector<VulkanRT::VkInstanceData> instData;
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
        ci.size = (uint64_t)instData.size() * sizeof(::VulkanRT::VkInstanceData);
        ci.usage = (::VulkanRT::BufferUsage)((uint32_t)::VulkanRT::BufferUsage::STORAGE | (uint32_t)::VulkanRT::BufferUsage::TRANSFER_DST);
        ci.location = ::VulkanRT::MemoryLocation::CPU_TO_GPU;
        ci.initialData = instData.data();
        m_device->m_instanceDataBuffer = m_device->createBuffer(ci);
        
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
}

// Materials & Textures
void VulkanBackendAdapter::uploadMaterials(const std::vector<MaterialData>& materials) {
    if (!m_device || !m_device->isInitialized()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    // Avoid descriptor slot exhaustion/stale texture behavior after many scene/backend toggles.
    // 1024 is binding-6 runtime array capacity; keep headroom.
    if (m_nextTextureID >= 980 || m_uploadedImages.size() >= 980) {
        SCENE_LOG_WARN("[Vulkan] Texture cache near descriptor capacity; purging and re-uploading active textures.");
        purgeUploadedTextureCacheLocked();
    }

    std::vector<VulkanRT::VkGpuMaterial> gpuMats;
    gpuMats.reserve(materials.size());

    for (const auto& m : materials) {
        VulkanRT::VkGpuMaterial gm{};
        gm.albedo_r = m.albedo.x; gm.albedo_g = m.albedo.y; gm.albedo_b = m.albedo.z; gm.opacity = m.opacity;
        // ... (remaining fields)
        gm.roughness = m.roughness;
        gm.metallic = m.metallic;
        gm.ior = m.ior;
        gm.transmission = m.transmission;
        gm.emission_r = m.emission.x; gm.emission_g = m.emission.y; gm.emission_b = m.emission.z;
        gm.emission_strength = m.emissionStrength;
        gm.subsurface_r = m.subsurfaceColor.x; gm.subsurface_g = m.subsurfaceColor.y; gm.subsurface_b = m.subsurfaceColor.z;
        gm.subsurface_amount = m.subsurface;
        gm.subsurface_radius_r = m.subsurfaceRadius.x; gm.subsurface_radius_g = m.subsurfaceRadius.y; gm.subsurface_radius_b = m.subsurfaceRadius.z;
        gm.subsurface_scale = m.subsurfaceScale;
        gm.subsurface_ior = m.subsurfaceIOR;
        // SSS control flags (match VkGpuMaterial layout Block 12)
       
        gm.clearcoat = m.clearcoat;
        gm.clearcoat_roughness = m.clearcoatRoughness;
        gm.translucent = m.translucent;
        gm.subsurface_anisotropy = m.subsurfaceAnisotropy;
        gm.anisotropic = m.anisotropic;
        gm.sheen = m.sheen;
        gm.sheen_tint = m.sheenTint;
        gm.flags = (uint32_t)m.flags;
        // If this is a terrain material, embed the terrain layer buffer index
        if (m.flags & Backend::IBackend::MAT_FLAG_TERRAIN) {
            gm._terrain_layer_idx = m.terrainLayerIdx;
        }
        // Water-specific params → VkGpuMaterial Block 8 & Block 9
        gm.micro_detail_strength = m.micro_detail_strength;
        gm.micro_detail_scale    = m.micro_detail_scale;
        gm.fft_amplitude         = m.fft_amplitude;
        gm.fft_time_scale        = m.fft_time_scale;
        gm.foam_threshold        = m.foam_threshold;
        gm.fft_ocean_size        = m.fft_ocean_size;
        gm.fft_choppiness        = m.fft_choppiness;
        gm.fft_wind_speed        = m.fft_wind_speed;
        // ... getTexID mapping (cast to uint32_t for GLSL compatibility)
        auto getTexID = [this](int64_t key, bool forceLinear = false) -> uint32_t {
            if (!key) return 0;
            uint64_t cacheKey = (static_cast<uint64_t>(key) << 1) | (forceLinear ? 1ull : 0ull);
            // [FIX] If the texture content was updated in-place (e.g. autoMask / paintSplatMap),
            // updateGPU() sets vulkan_dirty=true. Evict the stale cache entry so we re-upload.
            Texture* texCheck = reinterpret_cast<Texture*>(key);
            if (texCheck && texCheck->vulkan_dirty) {
                m_uploadedImageIDs.erase(cacheKey);
                texCheck->vulkan_dirty = false;
            }
            auto it = m_uploadedImageIDs.find(cacheKey);
            if (it != m_uploadedImageIDs.end()) return (uint32_t)it->second;
            Texture* tex = reinterpret_cast<Texture*>(key);
            if (!tex || !tex->is_loaded()) return 0;
            if (tex->is_hdr) {
                const std::vector<float4>& fp = tex->float_pixels;
                if (fp.empty()) return 0;
                int64_t id = this->uploadTexture2D(fp.data(), tex->width, tex->height, 4, false, true);
                if (id) { m_uploadedImageIDs[cacheKey] = id; return (uint32_t)id; }
                return 0;
            }
            const std::vector<CompactVec4>& px = tex->pixels;
            if (px.empty()) return 0;
            std::vector<uint8_t> tmp;
            tmp.resize(tex->width * tex->height * 4);
            for (size_t i = 0; i < px.size(); ++i) {
                tmp[i*4 + 0] = px[i].r; tmp[i*4 + 1] = px[i].g; tmp[i*4 + 2] = px[i].b; tmp[i*4 + 3] = px[i].a;
            }
            const bool useSrgb = forceLinear ? false : tex->is_srgb;
            int64_t id = this->uploadTexture2D(tmp.data(), tex->width, tex->height, 4, useSrgb, false);
            if (id) { m_uploadedImageIDs[cacheKey] = id; return (uint32_t)id; }
            return 0;
        };

        gm.albedo_tex = getTexID(m.albedoTexture, false);
        gm.normal_tex = getTexID(m.normalTexture, true);
        gm.roughness_tex = getTexID(m.roughnessTexture, true);
        gm.metallic_tex = getTexID(m.metallicTexture, true);
        gm.emission_tex = getTexID(m.emissionTexture, false);
        gm.transmission_tex = getTexID(m.transmissionTexture, true);
        gm.opacity_tex = getTexID(m.opacityTexture, true);
        // Set bit 8 in flags if opacity texture uses the alpha channel (RGBA texture)
        // vs. bit 8 clear = grayscale mask (R channel)
        if (m.opacityTexture) {
            Texture* opTex = reinterpret_cast<Texture*>(m.opacityTexture);
            if (opTex && opTex->is_loaded() && opTex->has_alpha) {
                gm.flags |= (1u << 8); // Bit 8: opacity is in .a channel
            }
        }
        gm.height_tex = getTexID(m.heightTexture, true);

        gpuMats.push_back(gm);
    }

    if (gpuMats.empty()) {
        VulkanRT::VkGpuMaterial defaultMat{};
        defaultMat.albedo_r = 0.8f; defaultMat.albedo_g = 0.8f; defaultMat.albedo_b = 0.8f;
        defaultMat.opacity = 1.0f;
        defaultMat.roughness = 0.5f;
        gpuMats.push_back(defaultMat);
    }

    m_device->updateMaterialBuffer(gpuMats.data(), gpuMats.size() * sizeof(::VulkanRT::VkGpuMaterial), (uint32_t)gpuMats.size());
    resetAccumulation();
}

void VulkanBackendAdapter::uploadHairMaterials(const std::vector<HairMaterialData>& materials) {
    if (!m_device || materials.empty()) return;

    std::vector<VulkanRT::HairGpuMaterial> gpuMats;
    gpuMats.reserve(materials.size());

    for (const auto& m : materials) {
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
        gm.groomID        = 0; // single-groom for now
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

    if (m_nextTextureID >= 980 || m_uploadedImages.size() >= 980) {
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
                // [FIX] Evict stale Vulkan sampler entry when splat map pixels were modified
                // in-place (autoMask / importSplatMap / paintSplatMap → updateGPU sets vulkan_dirty).
                if (splatTex->vulkan_dirty) {
                    m_uploadedImageIDs.erase(cacheKey);
                    splatTex->vulkan_dirty = false;
                }
                auto it = m_uploadedImageIDs.find(cacheKey);
                if (it != m_uploadedImageIDs.end()) {
                    gld.splat_map_tex = (uint32_t)it->second;
                } else {
                    // Upload the splat map
                    const std::vector<CompactVec4>& px = splatTex->pixels;
                    if (!px.empty()) {
                        std::vector<uint8_t> tmp(splatTex->width * splatTex->height * 4);
                        for (size_t i = 0; i < px.size(); ++i) {
                            tmp[i*4+0] = px[i].r; tmp[i*4+1] = px[i].g;
                            tmp[i*4+2] = px[i].b; tmp[i*4+3] = px[i].a;
                        }
                        int64_t id = this->uploadTexture2D(tmp.data(), splatTex->width, splatTex->height, 4, false, false);
                        if (id) { m_uploadedImageIDs[cacheKey] = id; gld.splat_map_tex = (uint32_t)id; }
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

int64_t VulkanBackendAdapter::uploadTexture2D(const void* data, uint32_t width, uint32_t height, uint32_t channels, bool srgb, bool isFloat) {
    if (!m_device || !m_device->isInitialized() || !data) return 0;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_device->waitIdle();

    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t bpp = 4; // bytes per pixel

    if (isFloat) {
        fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
        bpp = 16;
    } else if (srgb) {
        fmt = VK_FORMAT_R8G8B8A8_SRGB;
    }

    // Create staging buffer
    VulkanRT::BufferCreateInfo ci;
    ci.size = (uint64_t)width * height * bpp;
    ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_SRC;
    ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;

    VulkanRT::BufferHandle staging = m_device->createBuffer(ci);
    if (!staging.buffer) return 0;

    m_device->uploadBuffer(staging, data, ci.size);

    // Create image as a sampled texture (allow transfer dst for upload)
    VulkanRT::ImageHandle img = m_device->createImage2D(width, height, fmt,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!img.image) {
        m_device->destroyBuffer(staging);
        return 0;
    }

    m_device->copyBufferToImage(staging, img);

    // Transition uploaded image to SHADER_READ_ONLY_OPTIMAL for sampling in shaders
    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        m_device->destroyImage(img);
        m_device->destroyBuffer(staging);
        return 0;
    }
    m_device->transitionImageLayout(cmd, img.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_device->endSingleTimeCommands(cmd);

    // Create simple sampler
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.anisotropyEnable = VK_FALSE;
    sci.maxAnisotropy = 1.0f;
    sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sci.unnormalizedCoordinates = VK_FALSE;
    sci.compareEnable = VK_FALSE;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    VkSampler sampler = VK_NULL_HANDLE;
    vkCreateSampler(m_device->getDevice(), &sci, nullptr, &sampler);
    if (sampler) img.sampler = sampler;

    // Register image with an ID
    int64_t id = m_nextTextureID++;
    m_uploadedImages[id] = img;
    // Immediately update the RT descriptor array (binding 6) so shaders can sample this texture.
    if (m_device) {
        uint32_t slot = (uint32_t)id;
        m_device->updateRTTextureDescriptor(slot, img);
    }
    // CRITICAL: release temporary upload staging buffer.
    // Missing destroy here caused per-texture memory growth across Vulkan sessions.
    m_device->destroyBuffer(staging);
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
        m_device->updateRTTextureDescriptor((uint32_t)id, img);
    }
    return id;
}

void VulkanBackendAdapter::destroyTexture(int64_t texID) {
    auto it = m_uploadedImages.find(texID);
    if (it == m_uploadedImages.end()) return;
    VulkanRT::ImageHandle& img = it->second;
    m_device->removePendingRTTextureDescriptor(img);
    m_device->destroyImage(img);
    m_uploadedImages.erase(it);
    // Also remove any pointer -> id mappings that reference this id
    for (auto it2 = m_uploadedImageIDs.begin(); it2 != m_uploadedImageIDs.end(); ) {
        if (it2->second == texID) it2 = m_uploadedImageIDs.erase(it2);
        else ++it2;
    }
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
        gl.params[0] = (std::max)(l->radius, MIN_LIGHT_RADIUS);
        gl.params[1] = (std::max)(l->width, MIN_AREA_DIM);
        gl.params[2] = (std::max)(l->height, MIN_AREA_DIM); // For area lights this is height, for spot lights this will be used for inner cone (overwritten below)

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
    if (m_targetSamples != p.samplesPerPixel || m_useAdaptiveSampling != p.useAdaptiveSampling) {
        resetAccumulation();
    }
    m_targetSamples = p.samplesPerPixel; 
    m_useAdaptiveSampling = p.useAdaptiveSampling;
    m_varianceThreshold = p.adaptiveThreshold;
    m_maxBounces = (p.maxBounces > 0) ? p.maxBounces : m_maxBounces; // 0 = UI henüz set etmedi, mevcut değeri koru
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
    Backend::CameraParams cp;
    cp.origin = cam.lookfrom;
    cp.lookAt = cam.lookat;
    cp.up = cam.vup;
    cp.fov = cam.vfov;
    cp.aperture = cam.aperture;
    cp.focusDistance = cam.focus_dist;
    cp.aspectRatio = cam.aspect;
    cp.isoPresetIndex = cam.iso_preset_index;
    cp.shutterPresetIndex = cam.shutter_preset_index;
    cp.fstopPresetIndex = cam.fstop_preset_index;
    cp.ev_compensation = cam.ev_compensation;
    cp.autoAE = cam.auto_exposure;
    cp.usePhysicalExposure = cam.use_physical_exposure;
    cp.exposureFactor = cam.getPhysicalExposureMultiplier();

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

    // Fast path: use instance->source mapping + cache to only update transforms
    syncInstanceTransforms(objects, false);

    if (m_instance_sync_cache.empty() || m_instanceSources.size() != m_vkInstances.size()) {
        // Fallback to a conservative rebuild of full TLAS if mapping missing
        std::vector<VulkanRT::TLASInstance> updatedInstances = m_vkInstances;
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
            
            { auto merged = m_vkInstances; for (const auto& h : m_hairVkInstances) merged.push_back(h); m_device->updateTLAS(merged); }
            
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
        return;
    }

    // Use cached mapping for efficient per-instance transform update
    std::vector<VulkanRT::TLASInstance> updated = m_vkInstances;
    for (const auto& item : m_instance_sync_cache) {
        if (!item.representative_hittable) continue;
        Matrix4x4 m;
        if (auto tri = std::dynamic_pointer_cast<Triangle>(item.representative_hittable)) m = tri->getTransformMatrix();
        else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(item.representative_hittable)) m = inst->transform;
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
            m_device->updateTLAS(merged); 
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
            inst->transform = m_vkInstances[instance_id].transform; 
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[instance_id])) {
            tri->setBaseTransform(m_vkInstances[instance_id].transform);
        }
    }
}

void VulkanBackendAdapter::updateObjectTransform(const std::string& nodeName, const Matrix4x4& transform) {
    if (!m_device || !m_device->isInitialized()) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    
    bool changed = false;
    for (size_t i = 0; i < m_vkInstances.size(); ++i) {
        if (m_instanceSources.size() > i && m_instanceSources[i]) {
            std::string instName;
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                instName = inst->node_name;
            } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
                instName = tri->getNodeName();
            }

            if (matchesNodeNameForInstance(instName, nodeName)) {
                m_vkInstances[i].transform = transform;
                // Also update the sync cache so updateInstanceTransforms doesn't revert it
                if (m_instanceSources.size() > i) {
                     if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                          inst->transform = transform; 
                     } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
                         tri->setBaseTransform(transform);
                     }
                }
                changed = true;
            }
        }
    }
    
    if (changed) {
        // [VULKAN] Must wait idle before modifying AS that is actively being traced
        m_device->waitIdle();
        { 
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
void VulkanBackendAdapter::renderProgressive(void* s, void* w, void* r, int width, int height, void* fb, void* tex) {
    (void)s; (void)w; (void)r; 
    
    // [VULKAN THREAD SAFETY] Mutex prevents background updateGeometry from destroying resources mid-frame.
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    // [STABILITY] Robust null checks to prevent startup/switch crashes
    if (!m_device || !m_device->isInitialized()) return;
    if (!m_device->hasHardwareRT() || !fb || !tex) return;
    
    if (isAccumulationComplete()) return;

    // If a reset requested immediate UI clear, wipe the provided framebuffer/texture now
    if (m_forceClearOnNextPresent) {
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
        if (m_outputImage.image) m_device->destroyImage(m_outputImage);
        if (m_stagingBuffer.buffer) m_device->destroyBuffer(m_stagingBuffer);
        if (m_denoiserColorImage.image) m_device->destroyImage(m_denoiserColorImage);
        if (m_denoiserAlbedoImage.image) m_device->destroyImage(m_denoiserAlbedoImage);
        if (m_denoiserNormalImage.image) m_device->destroyImage(m_denoiserNormalImage);
        if (m_denoiserColorStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserColorStagingBuffer);
        if (m_denoiserAlbedoStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserAlbedoStagingBuffer);
        if (m_denoiserNormalStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserNormalStagingBuffer);

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
        m_denoiserColorImage = m_device->createImage2D(
            width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_denoiserAlbedoImage = m_device->createImage2D(
            width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_denoiserNormalImage = m_device->createImage2D(
            width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        VulkanRT::BufferCreateInfo stagingInfo;
        const uint64_t bytesPerPixel = (outFmt == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8ull : 16ull;
        stagingInfo.size = (uint64_t)width * height * bytesPerPixel;
        stagingInfo.usage = VulkanRT::BufferUsage::TRANSFER_DST;
        stagingInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
        m_stagingBuffer = m_device->createBuffer(stagingInfo);
        stagingInfo.size = (uint64_t)width * height * 4ull * sizeof(float);
        m_denoiserColorStagingBuffer = m_device->createBuffer(stagingInfo);
        m_denoiserAlbedoStagingBuffer = m_device->createBuffer(stagingInfo);
        m_denoiserNormalStagingBuffer = m_device->createBuffer(stagingInfo);

        if (!m_outputImage.image || !m_stagingBuffer.buffer ||
            !m_denoiserColorImage.image || !m_denoiserAlbedoImage.image || !m_denoiserNormalImage.image ||
            !m_denoiserColorStagingBuffer.buffer || !m_denoiserAlbedoStagingBuffer.buffer || !m_denoiserNormalStagingBuffer.buffer) {
            SCENE_LOG_ERROR("[Vulkan] Failed to allocate output/readback buffers for current resolution.");
            if (m_outputImage.image) m_device->destroyImage(m_outputImage);
            if (m_stagingBuffer.buffer) m_device->destroyBuffer(m_stagingBuffer);
            if (m_denoiserColorImage.image) m_device->destroyImage(m_denoiserColorImage);
            if (m_denoiserAlbedoImage.image) m_device->destroyImage(m_denoiserAlbedoImage);
            if (m_denoiserNormalImage.image) m_device->destroyImage(m_denoiserNormalImage);
            if (m_denoiserColorStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserColorStagingBuffer);
            if (m_denoiserAlbedoStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserAlbedoStagingBuffer);
            if (m_denoiserNormalStagingBuffer.buffer) m_device->destroyBuffer(m_denoiserNormalStagingBuffer);
            m_outputImage = {};
            m_stagingBuffer = {};
            m_denoiserColorImage = {};
            m_denoiserAlbedoImage = {};
            m_denoiserNormalImage = {};
            m_denoiserColorStagingBuffer = {};
            m_denoiserAlbedoStagingBuffer = {};
            m_denoiserNormalStagingBuffer = {};
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

        // Load Shadow Miss Shader (optional — enables shadow rays from hit shaders)
        std::vector<std::uint32_t> shadowMissSPV;
        if (std::filesystem::exists(shaderDir + "/shadow_miss.spv")) {
            shadowMissSPV = loadSPV(shaderDir + "/shadow_miss.spv");
            SCENE_LOG_INFO("[Vulkan] Shadow miss shader loaded successfully.");
        } else {
            SCENE_LOG_INFO("[Vulkan] Shadow miss shader not found — shadow rays will use primary miss.");
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

        if (!m_device->createRTPipeline(raygenSPV, missSPV, chitSPV, ahitSPV,
                volChitSPV, volIntSPV, hairChitSPV, hairIntSPV, shadowMissSPV, hairAhitSPV)) {
            SCENE_LOG_ERROR("[Vulkan] Failed to create RT Pipeline.");
            return;
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
    if (!m_device->isRTReady() || !m_device->hasTLAS()) return;

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
    pushConst.minSamples = 8; // safe default; UI-exposed minSamples not yet wired
    pushConst.lightCount = (uint32_t)m_cachedLights.size();
    pushConst.varianceThreshold = m_varianceThreshold;
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

    pushConst.waterTime = (float)SDL_GetTicks() / 1000.0f;
    pushConst.maxBounces = (uint32_t)std::max(1, m_maxBounces); // m_maxBounces her zaman UI'dan gelir

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
        resetAccumulation();
    }

    // Additionally detect significant view direction changes using previous view dir
    Vec3 viewDir = (lookAt - lookFrom).normalize();
    if (this->m_hasPrevView) {
        float dotv = std::clamp(this->m_prevViewDir.dot(viewDir), -1.0f, 1.0f);
        float ang = acos(dotv);
        if (ang > 0.01f) { // ~0.57 degrees
            resetAccumulation();
        }
    }
    // If camera is looking almost straight down, force a full reset to avoid
    // horizon/undersampling ghosting artifacts when pitch approaches -Y.
    if (viewDir.y < -0.999f) {
        resetAccumulation();
    }
    this->m_prevViewDir = viewDir;
    this->m_hasPrevView = true;

    m_device->setPushConstants(&pushConst, sizeof(CameraPushConstants));

    // 3. Trace Rays
        if (m_device->isRTReady() && m_device->hasTLAS()) {
        // Explicitly clear image on frame 0 to prevent ghosting or stale adaptive data
        if (m_currentSamples == 0) {
            m_device->clearImage(m_outputImage, 0.0f, 0.0f, 0.0f, 0.0f);
            m_device->clearImage(m_denoiserColorImage, 0.0f, 0.0f, 0.0f, 0.0f);
            m_device->clearImage(m_denoiserAlbedoImage, 0.0f, 0.0f, 0.0f, 0.0f);
            m_device->clearImage(m_denoiserNormalImage, 0.5f, 0.5f, 0.5f, 0.0f);
        }

        m_device->bindRTDescriptors(
            m_outputImage,
            &m_denoiserColorImage,
            &m_denoiserAlbedoImage,
            &m_denoiserNormalImage);
        m_device->setPushConstants(&pushConst, sizeof(CameraPushConstants));

        // Single command buffer: trace + layout transition + buffer copy + transition back
        // This reduces the per-frame GPU stalls from 4× vkQueueWaitIdle down to 1×.
        m_device->traceRaysAndReadback(width, height, m_outputImage, m_stagingBuffer);

        std::vector<uint32_t>* framebuffer = static_cast<std::vector<uint32_t>*>(fb);
        if (framebuffer->size() != (size_t)(width * height)) {
            framebuffer->resize(width * height);
        }

        // If the output image is float/half-float RGBA, download HDR and tonemap on CPU
        if (m_outputImage.format == VK_FORMAT_R32G32B32A32_SFLOAT ||
            m_outputImage.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
            m_hdrPixels.resize((size_t)width * (size_t)height * 4);
            if (m_outputImage.format == VK_FORMAT_R32G32B32A32_SFLOAT) {
                m_device->downloadBuffer(m_stagingBuffer, m_hdrPixels.data(), (uint64_t)width * height * 4 * sizeof(float));
            } else {
                std::vector<uint16_t> halfPixels((size_t)width * (size_t)height * 4);
                m_device->downloadBuffer(m_stagingBuffer, halfPixels.data(), (uint64_t)width * height * 4 * sizeof(uint16_t));
                for (size_t i = 0; i < halfPixels.size(); ++i) {
                    m_hdrPixels[i] = halfToFloat(halfPixels[i]);
                }
            }

            // Convert HDR floats -> 8-bit sRGB packed pixels
            SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);
            for (int j = 0; j < height; ++j) {
                for (int i = 0; i < width; ++i) {
                    size_t idx = (size_t)j * (size_t)width + (size_t)i;
                    // Exposure is already applied in raygen shader via cam.exposure_factor.
                    // Do not apply exposure again here (prevents double-exposure/gamma-like artifacts).
                    float r = m_hdrPixels[idx * 4 + 0];
                    float g = m_hdrPixels[idx * 4 + 1];
                    float b = m_hdrPixels[idx * 4 + 2];

                    // Match CPU renderer pipeline (Renderer.cpp):
                    // 1) sanitize NaN/Inf, clamp negatives to 0
                    // 2) Reinhard tone mapping: x / (x + 1) — compresses HDR range without clipping
                    // 3) Piecewise IEC 61966-2-1 sRGB transfer function
                    // This matches what the CPU path writes to original_surface for all backends.
                    auto sanitize = [](float v) -> float {
                        if (std::isnan(v)) return 0.0f;
                        if (std::isinf(v)) return (v > 0.0f) ? 65504.0f : 0.0f;
                        return std::max(v, 0.0f); // clamp negatives only, preserve HDR range
                    };
                    // Piecewise IEC 61966-2-1 sRGB transfer function
                    auto toSRGB = [](float c) -> float {
                        c = std::clamp(c, 0.0f, 1.0f);
                        return (c <= 0.0031308f)
                            ? 12.92f * c
                            : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
                    };

                    float rr = sanitize(r);
                    float gg = sanitize(g);
                    float bb = sanitize(b);

                    // Reinhard operator — same as CPU renderer (exposed_color / (exposed_color + 1))
                    rr = rr / (rr + 1.0f);
                    gg = gg / (gg + 1.0f);
                    bb = bb / (bb + 1.0f);

                    int ri = static_cast<int>(255.0f * toSRGB(rr));
                    int gi = static_cast<int>(255.0f * toSRGB(gg));
                    int bi = static_cast<int>(255.0f * toSRGB(bb));

                    uint32_t packed = SDL_MapRGB(fmt, ri, gi, bi);
                    (*framebuffer)[idx] = packed;

                    // Write to SDL surface if provided
                    if (s) {
                        SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
                        if (outSurf->pixels && outSurf->w == width && outSurf->h == height) {
                            Uint32* pixels_ptr = static_cast<Uint32*>(outSurf->pixels);
                            size_t screen_idx = idx;
                            pixels_ptr[screen_idx] = SDL_MapRGB(outSurf->format, ri, gi, bi);
                        }
                    }
                }
            }
            SDL_FreeFormat(fmt);

            if (tex) SDL_UpdateTexture(static_cast<SDL_Texture*>(tex), nullptr, framebuffer->data(), width * 4);
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

        if (m_statusCallback) {
            m_statusCallback("Vulkan Progressive Rendering (" + std::to_string(m_currentSamples) + " samples)", m_currentSamples);
        }
    }
}

void VulkanBackendAdapter::downloadImage(void* out) { (void)out; }
bool VulkanBackendAdapter::getDenoiserFrame(DenoiserFrameData& frame) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device || !m_device->isInitialized() || m_imageWidth <= 0 || m_imageHeight <= 0) return false;
    if (!m_denoiserColorImage.image || !m_denoiserAlbedoImage.image || !m_denoiserNormalImage.image) return false;
    if (!m_denoiserColorStagingBuffer.buffer || !m_denoiserAlbedoStagingBuffer.buffer || !m_denoiserNormalStagingBuffer.buffer) return false;

    const size_t pixelCount = (size_t)m_imageWidth * (size_t)m_imageHeight;
    const bool isHalfFloat = (m_denoiserColorImage.format == VK_FORMAT_R16G16B16A16_SFLOAT);

    m_device->copyImageToBuffer(m_denoiserColorImage, m_denoiserColorStagingBuffer);
    m_device->copyImageToBuffer(m_denoiserAlbedoImage, m_denoiserAlbedoStagingBuffer);
    m_device->copyImageToBuffer(m_denoiserNormalImage, m_denoiserNormalStagingBuffer);

    m_denoiserColorPixels.resize(pixelCount * 3);
    m_denoiserAlbedoPixels.resize(pixelCount * 3);
    m_denoiserNormalPixels.resize(pixelCount * 3);

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

    downloadFloat3(m_denoiserColorStagingBuffer, m_denoiserColorPixels, false);
    downloadFloat3(m_denoiserAlbedoStagingBuffer, m_denoiserAlbedoPixels, false);
    downloadFloat3(m_denoiserNormalStagingBuffer, m_denoiserNormalPixels, true);

    frame.width = m_imageWidth;
    frame.height = m_imageHeight;
    frame.color = m_denoiserColorPixels.data();
    frame.albedo = m_denoiserAlbedoPixels.data();
    frame.normal = m_denoiserNormalPixels.data();
    return true;
}
int VulkanBackendAdapter::getCurrentSampleCount() const { return this->m_currentSamples; }
bool VulkanBackendAdapter::isAccumulationComplete() const { return this->m_currentSamples >= this->m_targetSamples; }

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

    // Mark LUT as ready so the GLSL shader's _pad5 check succeeds on next setWorldData call
    m_atmosphereLutReady = true;
    // Push updated world buffer immediately so GPU sees _pad5 = 1 without waiting for next frame
    setWorldData(&m_cachedWorld);
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
    gw._pad0 = 0.0f;

    // ═════════════════════════════════════════════════════════════════
    // ATMOSPHERE DENSITY PARAMETERS
    // ═════════════════════════════════════════════════════════════════
    gw.airDensity = wd->nishita.air_density;
    gw.dustDensity = wd->nishita.dust_density;
    gw.ozoneDensity = wd->nishita.ozone_density;
    gw.altitude = wd->nishita.altitude;
    
    gw.planetRadius = wd->nishita.planet_radius;
    gw.atmosphereHeight = wd->nishita.atmosphere_height;
    gw._pad1 = 0.0f;
    gw._pad2 = 0.0f;

    // ═════════════════════════════════════════════════════════════════
    // CLOUD LAYER 1 PARAMETERS
    // ═════════════════════════════════════════════════════════════════
    gw.cloudsEnabled = wd->nishita.clouds_enabled ? 1 : 0;
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
    gw._pad5_aerial      = 0.0f;

    // ═════════════════════════════════════════════════════════════════
    // ENVIRONMENT & LUT REFERENCES
    // ═════════════════════════════════════════════════════════════════
    gw.envTexSlot = (int)m_envTexID;
    gw.envIntensity = wd->env_intensity;
    gw.envRotation = wd->env_rotation;
    // _pad5 repurposed as nishitaLutReady: 1 = atmosphereLUTs[4] binding has valid textures
    gw._pad5 = m_atmosphereLutReady ? 1 : 0;
    
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
    if (vols.empty() && m_orderedVDBInstances.empty()) {
        // No active volumes: release any stale cached VDB buffers immediately.
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
        m_device->m_volumeCount = 0;
        return;
    }

    // Build id→source map for fast O(1) lookup
    std::unordered_map<int, const GpuVDBVolume*> volByID;
    for (const auto& v : vols) volByID[v.vdb_id] = &v;

    // Release cached buffers for volumes that no longer exist in the scene.
    for (auto it = m_vdbBuffers.begin(); it != m_vdbBuffers.end(); ) {
        if (volByID.find(it->first) == volByID.end()) {
            if (it->second.buffer) m_device->destroyBuffer(it->second);
            it = m_vdbBuffers.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = m_vdbTempBuffers.begin(); it != m_vdbTempBuffers.end(); ) {
        if (volByID.find(it->first) == volByID.end()) {
            if (it->second.buffer) m_device->destroyBuffer(it->second);
            it = m_vdbTempBuffers.erase(it);
        } else {
            ++it;
        }
    }

    // ORDERING FIX: SSBO slot i must correspond to the VDB with TLAS customIndex==i.
    // After updateGeometry(), m_orderedVDBInstances records VDBs in TLAS traversal order.
    // If BVH reorders them vs. scene.vdb_volumes, this ensures shader lookups are correct.
    std::vector<const GpuVDBVolume*> orderedVols;
    if (!m_orderedVDBInstances.empty()) {
        for (const auto& hittable : m_orderedVDBInstances) {
            auto vdb = std::dynamic_pointer_cast<VDBVolume>(hittable);
            if (!vdb) { orderedVols.push_back(nullptr); continue; }
            auto it = volByID.find(vdb->getVDBVolumeID());
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

        // VDB native (original file) world-space AABB — used by the shader to remap
        // localPos [-0.5,0.5] → VDB world space before NanoVDB index lookup.
        // Must be local_bbox (not world_bbox) so gizmo moves don't corrupt the mapping.
        dst.aabb_min[0] = src.local_bbox_min.x; dst.aabb_min[1] = src.local_bbox_min.y; dst.aabb_min[2] = src.local_bbox_min.z;
        dst.aabb_max[0] = src.local_bbox_max.x; dst.aabb_max[1] = src.local_bbox_max.y; dst.aabb_max[2] = src.local_bbox_max.z;

        // Density
        dst.density_multiplier = src.density_multiplier;
        dst.density_remap_low = src.density_remap_low;
        dst.density_remap_high = src.density_remap_high;
        dst.noise_scale = 10.0f; // Give a non-zero scale so the procedural cloud actually renders nicely
        
        // Sync NanoVDB Host Buffer to Vulkan Device Buffer
        dst.volume_type = 2; // 2 = NanoVDB
        dst.vdb_grid_address = 0;
        dst.vdb_temp_address = 0;
        
        int vdb_id = src.vdb_id;
        if (vdb_id >= 0) {
            auto& mgr = VDBVolumeManager::getInstance();
            void* hostGrid = mgr.getHostGrid(vdb_id);
            size_t gridSize = mgr.getHostGridSize(vdb_id);
            if (hostGrid && gridSize > 0) {
                auto it = m_vdbBuffers.find(vdb_id);
                // Simple reallocation if size differs (could optimize to only check once)
                if (it == m_vdbBuffers.end() || it->second.size < gridSize) {
                    if (it != m_vdbBuffers.end()) {
                        m_device->destroyBuffer(it->second);
                    }
                    VulkanRT::BufferCreateInfo ci;
                    ci.size = gridSize;
                    ci.usage = (VulkanRT::BufferUsage)(
                        (uint32_t)VulkanRT::BufferUsage::STORAGE | 
                        (uint32_t)VulkanRT::BufferUsage::TRANSFER_DST |
                        0x0100 /* VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT - custom */);
                    ci.location = VulkanRT::MemoryLocation::GPU_ONLY;
                    VulkanRT::BufferHandle buf = m_device->createBuffer(ci);
                    m_vdbBuffers[vdb_id] = buf;
                    it = m_vdbBuffers.find(vdb_id);
                }
                if (it != m_vdbBuffers.end() && it->second.buffer) {
                    m_device->uploadBuffer(it->second, hostGrid, gridSize);
                    dst.vdb_grid_address = it->second.deviceAddress;
                }
            }

            // Upload temperature NanoVDB grid for blackbody/color-ramp emission (mode 2)
            void* hostTempGrid = mgr.getHostTemperatureGrid(vdb_id);
            size_t tempGridSize = mgr.getHostTemperatureGridSize(vdb_id);
            if (hostTempGrid && tempGridSize > 0) {
                auto it2 = m_vdbTempBuffers.find(vdb_id);
                if (it2 == m_vdbTempBuffers.end() || it2->second.size < tempGridSize) {
                    if (it2 != m_vdbTempBuffers.end()) m_device->destroyBuffer(it2->second);
                    VulkanRT::BufferCreateInfo ci2;
                    ci2.size = tempGridSize;
                    ci2.usage = (VulkanRT::BufferUsage)(
                        (uint32_t)VulkanRT::BufferUsage::STORAGE |
                        (uint32_t)VulkanRT::BufferUsage::TRANSFER_DST |
                        0x0100);
                    ci2.location = VulkanRT::MemoryLocation::GPU_ONLY;
                    m_vdbTempBuffers[vdb_id] = m_device->createBuffer(ci2);
                    it2 = m_vdbTempBuffers.find(vdb_id);
                }
                if (it2 != m_vdbTempBuffers.end() && it2->second.buffer) {
                    m_device->uploadBuffer(it2->second, hostTempGrid, tempGridSize);
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
        // volume_type = 2 (NanoVDB) when grid address was successfully uploaded,
        // fall back to 1 (procedural noise) only if no NanoVDB buffer is available.
        dst.volume_type = (dst.vdb_grid_address != 0) ? 2 : 1;
        dst.is_active = 1;
        dst.voxel_size = src.voxel_size;
    }

    m_device->updateVolumeBuffer(instances.data(),
                                  instances.size() * sizeof(VulkanRT::VkVolumeInstance),
                                  (uint32_t)instances.size());

    // ── TLAS transform refresh ──────────────────────────────────────────────
    // When a VDB is moved with the gizmo, setTransform() updates the C++ object
    // but the TLAS AABB instance transform remains stale.  Fix: recompute the
    // scale+translate transform from the current worldBounds for every volume
    // instance found in m_instanceSources and push an updateTLAS call.
    {
        bool tlas_changed = false;
        for (size_t i = 0; i < m_instanceSources.size() && i < m_vkInstances.size(); ++i) {
            auto vdb = std::dynamic_pointer_cast<VDBVolume>(m_instanceSources[i]);
            if (!vdb) continue;
            AABB wb = vdb->getWorldBounds();
            Vec3 center = (wb.min + wb.max) * 0.5f;
            Vec3 sz(wb.max.x - wb.min.x, wb.max.y - wb.min.y, wb.max.z - wb.min.z);
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

    VK_INFO() << "[VulkanBackendAdapter] Uploaded " << instances.size() << " VDB volume(s) to Vulkan SSBO." << std::endl;
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
    m_currentSamples = 0;
    // Also clear the output image to avoid ghosting when accumulation restarts
    if (m_outputImage.image && m_device) {
        m_device->clearImage(m_outputImage, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (m_denoiserColorImage.image && m_device) {
        m_device->clearImage(m_denoiserColorImage, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (m_denoiserAlbedoImage.image && m_device) {
        m_device->clearImage(m_denoiserAlbedoImage, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (m_denoiserNormalImage.image && m_device) {
        m_device->clearImage(m_denoiserNormalImage, 0.5f, 0.5f, 0.5f, 0.0f);
    }
    // Request UI-level clear on next present so host-side view is immediately cleared
    m_forceClearOnNextPresent = true;
}
float VulkanBackendAdapter::getMillisecondsPerSample() const { return 0.0f; }

} // namespace Backend
