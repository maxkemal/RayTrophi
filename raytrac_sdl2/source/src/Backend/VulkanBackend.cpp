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
#include <filesystem>
#include <functional>
#include "HittableInstance.h"
#include "HittableList.h"
#include "ParallelBVHNode.h"
#include "Triangle.h"
#include <SDL.h>
#include "stb_image_write.h"
#include "Texture.h"
#include "World.h"
#include "AtmosphereLUT.h"
#include "CameraPresets.h"
#include "Camera.h"

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

        // Destroy BLAS
        for (auto& blas : m_blasList) {
            if (blas.accel && fpDestroyAccelerationStructureKHR) {
                fpDestroyAccelerationStructureKHR(m_device, blas.accel, nullptr);
            }
            if (blas.buffer.buffer) {
                vkDestroyBuffer(m_device, blas.buffer.buffer, nullptr);
                vkFreeMemory(m_device, blas.buffer.memory, nullptr);
            }
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
//        m_device = VK_NULL_HANDLE;
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

    VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);
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

    vkCreateBuffer(m_device, &bufferInfo, nullptr, &handle.buffer);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(m_device, handle.buffer, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, translateMemoryLocation(info.location));

    VkMemoryAllocateFlagsInfo flagsInfo{};
    if (m_capabilities.supportsBufferDeviceAddress) {
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        allocInfo.pNext = &flagsInfo;
    }

    vkAllocateMemory(m_device, &allocInfo, nullptr, &handle.memory);
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

    // Check if memory is host visible
    VkMemoryPropertyFlags flags = translateMemoryLocation(MemoryLocation::CPU_TO_GPU); // host visible
    // This is a bit simplified, ideally we check actual memory type flags
    
    void* mapped;
    if (vkMapMemory(m_device, dst.memory, offset, size, 0, &mapped) == VK_SUCCESS) {
        memcpy(mapped, data, size);
        vkUnmapMemory(m_device, dst.memory);
    } else {
        VK_ERROR() << "[VulkanDevice] uploadBuffer failed: Memory not mappable" << std::endl;
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
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(m_device, &allocInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    return cmdBuf;
}

void VulkanDevice::endSingleTimeCommands(VkCommandBuffer cmdBuf) {
    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_computeQueue);
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

    // --- 1) Upload vertex data to GPU buffer ---
    BufferCreateInfo vertBufInfo;
    vertBufInfo.size = (uint64_t)info.vertexCount * info.vertexStride;
    vertBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    vertBufInfo.location = MemoryLocation::CPU_TO_GPU;
    vertBufInfo.initialData = info.vertexData;
    auto vertexBuffer = createBuffer(vertBufInfo);

    // --- 1b) Upload normal data if provided ---
    BufferHandle normalBuffer{};
    if (info.normalData) {
        BufferCreateInfo normBufInfo;
        normBufInfo.size = (uint64_t)info.vertexCount * sizeof(float) * 3;
        normBufInfo.usage = BufferUsage::STORAGE;
        normBufInfo.location = MemoryLocation::CPU_TO_GPU;
        normBufInfo.initialData = info.normalData;
        normalBuffer = createBuffer(normBufInfo);
    }

    // --- 1c) Upload UV data if provided ---
    BufferHandle uvBuffer{};
    if (info.uvData) {
        BufferCreateInfo uvBufInfo;
        uvBufInfo.size = (uint64_t)info.vertexCount * sizeof(float) * 2;
        uvBufInfo.usage = BufferUsage::STORAGE;
        uvBufInfo.location = MemoryLocation::CPU_TO_GPU;
        uvBufInfo.initialData = info.uvData;
        uvBuffer = createBuffer(uvBufInfo);
    }

    // --- 2) Upload index data if provided ---
    BufferHandle indexBuffer{};
    bool hasIndices = (info.indexData && info.indexCount > 0);
    if (hasIndices) {
        BufferCreateInfo idxBufInfo;
        idxBufInfo.size = (uint64_t)info.indexCount * sizeof(uint32_t);
        idxBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
        idxBufInfo.location = MemoryLocation::CPU_TO_GPU;
        idxBufInfo.initialData = info.indexData;
        indexBuffer = createBuffer(idxBufInfo);
    }

    // --- 3) Build geometry info ---
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexBuffer.deviceAddress;
    triangles.vertexStride = info.vertexStride;
    triangles.maxVertex = info.vertexCount - 1;
    if (hasIndices) {
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = indexBuffer.deviceAddress;
    } else {
        triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = triangles;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (info.allowUpdate) buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    uint32_t primitiveCount = hasIndices ? (info.indexCount / 3) : (info.vertexCount / 3);

    // --- 4) Query build sizes ---
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fpGetAccelerationStructureBuildSizesKHR(m_device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo);

    // --- 5) Create AS buffer ---
    AccelStructHandle blasHandle{};

    BufferCreateInfo asBufInfo;
    asBufInfo.size = sizeInfo.accelerationStructureSize;
    asBufInfo.usage = BufferUsage::ACCELERATION | BufferUsage::STORAGE;
    asBufInfo.location = MemoryLocation::GPU_ONLY;
    blasHandle.buffer = createBuffer(asBufInfo);

    // --- 6) Create acceleration structure ---
    VkAccelerationStructureCreateInfoKHR asCreateInfo{};
    asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = blasHandle.buffer.buffer;
    asCreateInfo.size = sizeInfo.accelerationStructureSize;
    asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    fpCreateAccelerationStructureKHR(m_device, &asCreateInfo, nullptr, &blasHandle.accel);

    // --- 7) Scratch buffer with proper alignment ---
    uint64_t scratchAlignment = m_capabilities.minScratchAlignment > 0 ? m_capabilities.minScratchAlignment : 128;
    uint64_t alignedScratchSize = (sizeInfo.buildScratchSize + scratchAlignment - 1) & ~(scratchAlignment - 1);

    BufferCreateInfo scratchBufInfo;
    scratchBufInfo.size = alignedScratchSize;
    scratchBufInfo.usage = BufferUsage::STORAGE;
    scratchBufInfo.location = MemoryLocation::GPU_ONLY;
    auto scratchBuffer = createBuffer(scratchBufInfo);

    // --- 8) Build! ---
    buildInfo.dstAccelerationStructure = blasHandle.accel;
    buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    fpCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    endSingleTimeCommands(cmd);

    // --- 9) Get device address for TLAS reference ---
    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = blasHandle.accel;
    blasHandle.deviceAddress = fpGetAccelerationStructureDeviceAddressKHR(m_device, &addrInfo);

    // Cleanup scratch buffer
    destroyBuffer(scratchBuffer);
    
    // Store attribute buffers in handle for shader access
    blasHandle.vertexBuffer = vertexBuffer;
    blasHandle.normalBuffer = normalBuffer;
    blasHandle.uvBuffer = uvBuffer;
    blasHandle.indexBuffer = indexBuffer;

    // Upload per-primitive material indices if provided
    if (info.materialIndexData && info.materialIndexCount > 0) {
        BufferCreateInfo matBufInfo;
        matBufInfo.size = (uint64_t)info.materialIndexCount * sizeof(uint32_t);
        matBufInfo.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        matBufInfo.location = MemoryLocation::CPU_TO_GPU;
        matBufInfo.initialData = info.materialIndexData;
        blasHandle.materialIndexBuffer = createBuffer(matBufInfo);
    }

    uint32_t idx = (uint32_t)m_blasList.size();
    m_blasList.push_back(blasHandle);

    VK_INFO() << "[VulkanDevice] BLAS created (index=" << idx
              << ", tris=" << primitiveCount << ", size=" << (sizeInfo.accelerationStructureSize / 1024) << " KB)" << std::endl;
    return idx;
}

void VulkanDevice::createTLAS(const TLASCreateInfo& info) {
    if (!hasHardwareRT() || !fpCreateAccelerationStructureKHR) return;
    if (info.instances.empty()) return;

    // Determine whether we'll perform an UPDATE (more efficient) or full rebuild
    bool performUpdate = false;
    if (m_tlas.accel) {
        if (info.allowUpdate) {
            performUpdate = true;
        } else {
            fpDestroyAccelerationStructureKHR(m_device, m_tlas.accel, nullptr);
            destroyBuffer(m_tlas.buffer);
            m_tlas = {};
        }
    }
    // Free previous instance data buffer now (we will replace it with the new one)
    if (m_tlasInstanceBuffer.buffer) {
        destroyBuffer(m_tlasInstanceBuffer);
    }

    // --- 1) Build VkAccelerationStructureInstanceKHR array ---
    std::vector<VkAccelerationStructureInstanceKHR> vkInstances(info.instances.size());
    for (size_t i = 0; i < info.instances.size(); i++) {
        const auto& src = info.instances[i];
        auto& dst = vkInstances[i];

        if (src.blasIndex >= m_blasList.size()) continue;

        // VkTransformMatrixKHR is 3x4 row-major
        const auto& m = src.transform;
        dst.transform.matrix[0][0] = m.m[0][0]; dst.transform.matrix[0][1] = m.m[0][1]; dst.transform.matrix[0][2] = m.m[0][2]; dst.transform.matrix[0][3] = m.m[0][3];
        dst.transform.matrix[1][0] = m.m[1][0]; dst.transform.matrix[1][1] = m.m[1][1]; dst.transform.matrix[1][2] = m.m[1][2]; dst.transform.matrix[1][3] = m.m[1][3];
        dst.transform.matrix[2][0] = m.m[2][0]; dst.transform.matrix[2][1] = m.m[2][1]; dst.transform.matrix[2][2] = m.m[2][2]; dst.transform.matrix[2][3] = m.m[2][3];

        dst.instanceCustomIndex = src.customIndex;
        dst.mask = src.mask;
        dst.instanceShaderBindingTableRecordOffset = 0;
        dst.flags = src.frontFaceCCW ? VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR : 0;
        dst.accelerationStructureReference = m_blasList[src.blasIndex].deviceAddress;
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

    uint32_t instanceCount = (uint32_t)vkInstances.size();

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

    // --- 7) Build TLAS ---
    buildInfo.dstAccelerationStructure = m_tlas.accel;
    buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = beginSingleTimeCommands();
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

    VK_INFO() << "[VulkanDevice] TLAS " << (performUpdate ? "updated" : "created") << " (" << instanceCount << " instances)" << std::endl;
}

void VulkanDevice::updateBLAS(uint32_t blasIndex, const float* newVertices) {
    // TODO: Refit BLAS with new vertex positions (for deforming geometry)
    (void)blasIndex; (void)newVertices;
}

void VulkanDevice::updateTLAS(const std::vector<TLASInstance>& instances) {
    // Rebuild TLAS with updated transforms
    TLASCreateInfo info;
    info.instances = instances;
    info.allowUpdate = true;
    createTLAS(info);
}

void VulkanDevice::traceRays(uint32_t w, uint32_t h, uint32_t d) {
    if (!m_rtPipelineReady || !fpCmdTraceRaysKHR) {
        VK_ERROR() << "[VulkanDevice] RT pipeline not ready for traceRays!" << std::endl;
        return;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();

    // Bind RT pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);

    // Bind RT descriptor set
    if (m_rtDescriptorSet) {
        VK_INFO() << "[VulkanDevice] traceRays - binding RT descriptor set: " << (void*)m_rtDescriptorSet << std::endl;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            m_rtPipelineLayout, 0, 1, &m_rtDescriptorSet, 0, nullptr);
    }

    // Push constants
    if (!m_pushConstantData.empty()) {
        vkCmdPushConstants(cmd, m_rtPipelineLayout,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
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

bool VulkanDevice::createRTPipeline(const std::vector<uint32_t>& raygenSPV,
                                     const std::vector<uint32_t>& missSPV,
                                     const std::vector<uint32_t>& closestHitSPV,
                                     const std::vector<uint32_t>& anyHitSPV) {
    if (!hasHardwareRT() || !fpCreateRayTracingPipelinesKHR) {
        VK_ERROR() << "[VulkanDevice] Hardware RT not available" << std::endl;
        return false;
    }

    VK_INFO() << "[VulkanDevice] Creating RT pipeline..." << std::endl;

    // --- 1) Create shader modules ---
    auto createModule = [&](const std::vector<uint32_t>& code) -> VkShaderModule {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = code.size() * sizeof(uint32_t);
        ci.pCode = code.data();
        VkShaderModule mod;
        vkCreateShaderModule(m_device, &ci, nullptr, &mod);
        return mod;
    };

    VkShaderModule raygenModule = raygenSPV.empty() ? VK_NULL_HANDLE : createModule(raygenSPV);
    VkShaderModule missModule = missSPV.empty() ? VK_NULL_HANDLE : createModule(missSPV);
    VkShaderModule chitModule = closestHitSPV.empty() ? VK_NULL_HANDLE : createModule(closestHitSPV);
    VkShaderModule anyhitModule = anyHitSPV.empty() ? VK_NULL_HANDLE : createModule(anyHitSPV);

    if (raygenModule == VK_NULL_HANDLE || missModule == VK_NULL_HANDLE || chitModule == VK_NULL_HANDLE) {
        if (raygenModule) vkDestroyShaderModule(m_device, raygenModule, nullptr);
        if (missModule) vkDestroyShaderModule(m_device, missModule, nullptr);
        if (chitModule) vkDestroyShaderModule(m_device, chitModule, nullptr);
        VK_ERROR() << "[VulkanDevice] Failed to load RT shader modules!" << std::endl;
        return false;
    }

    // --- 2) Pipeline shader stages ---
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    stages.reserve(4);
    VkPipelineShaderStageCreateInfo sRay{}; sRay.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; sRay.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR; sRay.module = raygenModule; sRay.pName = "main"; stages.push_back(sRay);
    VkPipelineShaderStageCreateInfo sMiss{}; sMiss.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; sMiss.stage = VK_SHADER_STAGE_MISS_BIT_KHR; sMiss.module = missModule; sMiss.pName = "main"; stages.push_back(sMiss);
    VkPipelineShaderStageCreateInfo sChit{}; sChit.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; sChit.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; sChit.module = chitModule; sChit.pName = "main"; stages.push_back(sChit);
    if (anyhitModule != VK_NULL_HANDLE) { VkPipelineShaderStageCreateInfo sAhit{}; sAhit.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; sAhit.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR; sAhit.module = anyhitModule; sAhit.pName = "main"; stages.push_back(sAhit); }

    // --- 3) Shader groups ---
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(3);

    // Raygen group
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0; // stage index
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Miss group
    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Hit group (closest hit)
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader = 2;
    groups[2].anyHitShader = (anyhitModule != VK_NULL_HANDLE) ? uint32_t(3) : VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // --- 4) Descriptor set layout ---
    // Binding 0: Output Image
    // Binding 1: TLAS
    // Binding 2: Materials SSBO
    // Binding 3: Lights SSBO
    // Binding 4: Geometry SSBO
    // Binding 5: Instances SSBO
    // Binding 6: Material textures (runtime array)
    // Binding 7: World data SSBO
    // Binding 8: Atmosphere LUT samplers (transmittance, skyview, multi-scatter, aerial perspective)
    VkDescriptorSetLayoutBinding bindings[9] = {};
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
    bindings[2].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 64; // runtime array capacity
    bindings[6].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

    // Binding 7: WorldData SSBO (extended with complete Nishita parameters)
    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding 8: Atmosphere LUT Samplers (4 textures: transmittance, skyview, multi-scatter, aerial perspective)
    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 4;
    bindings[8].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = 9;
    dslCI.pBindings = bindings;
    vkCreateDescriptorSetLayout(m_device, &dslCI, nullptr, &m_rtDescriptorSetLayout);

    // --- 5) Push constant range (camera data + rendering params) ---
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
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
        VK_WARN() << "[VulkanDevice] vkCreateRayTracingPipelinesKHR failed (with any-hit): " << result << ". Retrying without any-hit..." << std::endl;

        // Rebuild stages without any-hit
        std::vector<VkPipelineShaderStageCreateInfo> stages2;
        stages2.reserve(3);
        VkPipelineShaderStageCreateInfo sRay2{}; sRay2.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; sRay2.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR; sRay2.module = raygenModule; sRay2.pName = "main"; stages2.push_back(sRay2);
        VkPipelineShaderStageCreateInfo sMiss2{}; sMiss2.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; sMiss2.stage = VK_SHADER_STAGE_MISS_BIT_KHR; sMiss2.module = missModule; sMiss2.pName = "main"; stages2.push_back(sMiss2);
        VkPipelineShaderStageCreateInfo sChit2{}; sChit2.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; sChit2.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; sChit2.module = chitModule; sChit2.pName = "main"; stages2.push_back(sChit2);

        // Update group anyHit to unused
        groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;

        rtCI.stageCount = (uint32_t)stages2.size();
        rtCI.pStages = stages2.data();

        result = fpCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtCI, nullptr, &m_rtPipeline);
    }

    // Cleanup shader modules (safe after pipeline creation attempt)
    if (raygenModule) vkDestroyShaderModule(m_device, raygenModule, nullptr);
    if (missModule)   vkDestroyShaderModule(m_device, missModule, nullptr);
    if (chitModule)   vkDestroyShaderModule(m_device, chitModule, nullptr);
    if (anyhitModule) vkDestroyShaderModule(m_device, anyhitModule, nullptr);

    if (result != VK_SUCCESS) {
        VK_ERROR() << "[VulkanDevice] vkCreateRayTracingPipelinesKHR failed: " << result << std::endl;
        return false;
    }

    // --- 8) Build Shader Binding Table (SBT) ---
    uint32_t handleSize = m_capabilities.shaderGroupHandleSize;
    uint32_t handleAlignment = m_capabilities.shaderGroupBaseAlignment;
    if (handleAlignment == 0) handleAlignment = handleSize; // Fallback
    uint32_t groupCount = 3;

    // Aligned handle size (each entry must be aligned)
    uint32_t alignedHandleSize = (handleSize + (handleAlignment - 1)) & ~(handleAlignment - 1);

    // Get shader group handles
    uint32_t handleStorageSize = groupCount * handleSize;
    std::vector<uint8_t> handleData(handleStorageSize);
    fpGetRayTracingShaderGroupHandlesKHR(m_device, m_rtPipeline, 0, groupCount, handleStorageSize, handleData.data());

    // SBT layout: [raygen | miss | hit] each aligned
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

    // Set SBT regions
    VkDeviceAddress sbtAddr = m_sbtBuffer.deviceAddress;
    m_sbtRaygenRegion.deviceAddress = sbtAddr;
    m_sbtRaygenRegion.stride = alignedHandleSize;
    m_sbtRaygenRegion.size = alignedHandleSize;

    m_sbtMissRegion.deviceAddress = sbtAddr + alignedHandleSize;
    m_sbtMissRegion.stride = alignedHandleSize;
    m_sbtMissRegion.size = alignedHandleSize;

    m_sbtHitRegion.deviceAddress = sbtAddr + 2 * alignedHandleSize;
    m_sbtHitRegion.stride = alignedHandleSize;
    m_sbtHitRegion.size = alignedHandleSize;

    m_sbtCallableRegion = {}; // No callable shaders

    m_rtPipelineReady = true;
    VK_INFO() << "[VulkanDevice] RT pipeline + SBT created successfully!" << std::endl;
    return true;
}

void VulkanDevice::bindRTDescriptors(const ImageHandle& outputImage) {
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
        ci.location = MemoryLocation::CPU_TO_GPU;
        ci.initialData = geoData.data();
        m_geometryDataBuffer = createBuffer(ci);
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
    writes.reserve(8);

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

    // Update bindings 0-7 immediately (safe local buffers)
    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }

    // Binding 6: Material textures (runtime-sized array)
    // Update immediately to avoid dangling pointers from local vectors
    if (!m_pendingTextureDescriptors.empty()) {
        std::vector<VkDescriptorImageInfo> extraImageInfos;
        std::vector<VkWriteDescriptorSet> extraWrites;
        
        extraImageInfos.reserve(m_pendingTextureDescriptors.size());
        extraWrites.reserve(m_pendingTextureDescriptors.size());

        // First pass: create all image infos
        size_t infoIdx = 0;
        for (const auto& p : m_pendingTextureDescriptors) {
            const ImageHandle& img = p.second;
            VkDescriptorImageInfo ii{};
            ii.sampler = img.sampler;
            ii.imageView = img.view;
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            extraImageInfos.push_back(ii);
            infoIdx++;
        }

        // Second pass: create writes pointing to stable image info storage
        infoIdx = 0;
        for (const auto& p : m_pendingTextureDescriptors) {
            uint32_t slot = p.first;
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = m_rtDescriptorSet;
            w.dstBinding = 6;
            w.dstArrayElement = slot;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = 1;
            w.pImageInfo = &extraImageInfos[infoIdx];
            extraWrites.push_back(w);
            infoIdx++;
        }

        // Update immediately while extraImageInfos is in scope
        vkUpdateDescriptorSets(m_device, (uint32_t)extraWrites.size(), extraWrites.data(), 0, nullptr);
        m_pendingTextureDescriptors.clear();
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
        // Use stack allocation (fixed-size array) to ensure lifetime safety
        VkDescriptorImageInfo lutImageInfos[4] = {};
        for (int i = 0; i < 4; i++) {
            if (m_lutImages[i].view != VK_NULL_HANDLE) {
                lutImageInfos[i].sampler = m_lutImages[i].sampler;
                lutImageInfos[i].imageView = m_lutImages[i].view;
                lutImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            } else {
                // Leave as zeroed (will be ignored if sampler/view are null)
                lutImageInfos[i].sampler = VK_NULL_HANDLE;
                lutImageInfos[i].imageView = VK_NULL_HANDLE;
                lutImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            }
        }
        
        // Count valid entries and write only those to avoid null descriptors
        uint32_t validCount = 0;
        for (int i = 0; i < 4; ++i) if (lutImageInfos[i].imageView != VK_NULL_HANDLE) ++validCount;
        if (validCount > 0) {
            VkWriteDescriptorSet w8{};
            w8.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w8.dstSet = m_rtDescriptorSet;
            w8.dstBinding = 8;
            w8.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w8.descriptorCount = validCount;
            w8.pImageInfo = lutImageInfos; // first validCount entries correspond to LUT indices 0..validCount-1
            VK_INFO() << "[VulkanDevice] bindRTDescriptors - writing " << validCount << " LUT descriptors to set " << (void*)m_rtDescriptorSet << std::endl;
            for (uint32_t ii = 0; ii < validCount; ++ii) {
                VK_INFO() << "  lutImageInfos[" << ii << "] view=" << (void*)lutImageInfos[ii].imageView << " sampler=" << (void*)lutImageInfos[ii].sampler << std::endl;
            }
            vkUpdateDescriptorSets(m_device, 1, &w8, 0, nullptr);
            VK_INFO() << "[VulkanDevice] bindRTDescriptors - LUT descriptors written" << std::endl;
        }
    }
}

// Update a single combined image sampler entry in the RT descriptor set (binding 6)
void VulkanDevice::updateRTTextureDescriptor(uint32_t slot, const ImageHandle& image) {
    if (slot >= 1024) {
        VK_WARN() << "[VulkanDevice] Texture slot " << slot << " out of range for materialTextures array" << std::endl;
        return;
    }

    // If descriptor set isn't allocated yet, queue the update to be applied later
    if (m_rtDescriptorSet == VK_NULL_HANDLE) {
        m_pendingTextureDescriptors.emplace_back(slot, image);
        return;
    }

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler = image.sampler;
    imgInfo.imageView = image.view;
    // Assume shader expects read-only optimal layout for sampled textures
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_rtDescriptorSet;
    write.dstBinding = 6;
    write.dstArrayElement = slot;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imgInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}


void VulkanDevice::updateMaterialBuffer(const void* data, uint64_t size, uint32_t count) {
    if (m_materialBuffer.size < size) {
        if (m_materialBuffer.buffer) destroyBuffer(m_materialBuffer);
        
        BufferCreateInfo ci;
        ci.size = size > 1024 ? size : 1024; // Min size
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::CPU_TO_GPU;
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
        ci.location = MemoryLocation::CPU_TO_GPU;
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
        ci.location = MemoryLocation::CPU_TO_GPU;
        m_worldBuffer = createBuffer(ci);
    }
    uploadBuffer(m_worldBuffer, data, size);
    // Debug: if this appears to be a VkWorldDataExtended, dump LUT fields
    if (size >= sizeof(VulkanRT::VkWorldDataExtended)) {
        const VulkanRT::VkWorldDataExtended* gw = reinterpret_cast<const VulkanRT::VkWorldDataExtended*>(data);
        uint64_t sky = gw->skyviewLUT;
        uint64_t trans = gw->transmittanceLUT;
        VK_INFO() << "[VulkanDevice] updateWorldBuffer - gw.transmittanceLUT=" << trans << " skyviewLUT=" << sky << std::endl;
        VK_INFO() << "[VulkanDevice] updateWorldBuffer - skyviewLUT low32=" << (uint32_t)(sky & 0xFFFFFFFFULL)
                  << " high32=" << (uint32_t)((sky >> 32) & 0xFFFFFFFFULL) << std::endl;
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

    vkAllocateMemory(m_device, &allocInfo, nullptr, &handle.memory);
    vkBindImageMemory(m_device, handle.image, handle.memory, 0);

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

    vkCreateImageView(m_device, &viewInfo, nullptr, &handle.view);

    // --- 4) Transition to GENERAL layout (for storage image access) ---
    VkCommandBuffer cmd = beginSingleTimeCommands();
    transitionImageLayout(cmd, handle.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    endSingleTimeCommands(cmd);

    VK_INFO() << "[VulkanDevice] Image created: " << width << "x" << height << std::endl;
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

void VulkanDevice::copyImageToBuffer(const ImageHandle& src, const BufferHandle& dst) {
    VkCommandBuffer cmd = beginSingleTimeCommands();

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
    VkCommandBuffer cmd = beginSingleTimeCommands();

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

    VkDescriptorSet descriptorSet;
    vkAllocateDescriptorSets(m_device, &allocInfo, &descriptorSet);

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

} // namespace VulkanRT


// ============================================================================
// Backend::VulkanBackendAdapter Implementation
// ============================================================================

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

VulkanBackendAdapter::~VulkanBackendAdapter() = default;

bool VulkanBackendAdapter::initialize() {
#ifdef _DEBUG
    bool validation = true;
#else
    bool validation = false;
#endif
    bool ok = m_device->initialize(true, validation);
    if (ok && !m_cachedLights.empty()) {
        VK_INFO() << "[VulkanBackendAdapter] Uploading cached lights after device init (" << m_cachedLights.size() << ")" << std::endl;
        setLights(m_cachedLights);
        m_cachedLights.clear();
    }
    return ok;
}

void VulkanBackendAdapter::shutdown() { 
    // Destroy the device object once. VulkanDevice destructor calls shutdown(),
    // so reset the unique_ptr to avoid calling shutdown() twice on the same object
    if (m_device) {
        m_device.reset();
    }
    m_meshRegistry.clear();
    m_vkInstances.clear();
    m_lastObjects.clear();
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
    if (triangles.empty()) return UINT32_MAX;

    // Check registry for existing BLAS
    auto it = m_meshRegistry.find(meshName);
    if (it != m_meshRegistry.end()) return it->second;

    // Flatten vertices, normals, and UVs for Vulkan
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    positions.reserve(triangles.size() * 9);
    normals.reserve(triangles.size() * 9);
    uvs.reserve(triangles.size() * 6);

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
    }

    // Per-primitive material indices (one per triangle)
    std::vector<uint32_t> materialIndices;
    materialIndices.reserve(triangles.size());
    for (const auto& t : triangles) materialIndices.push_back(t.materialID);

    VulkanRT::BLASCreateInfo blasInfo;
    blasInfo.vertexData = positions.data();
    blasInfo.normalData = normals.data();
    blasInfo.uvData = uvs.data();
    blasInfo.vertexCount = (uint32_t)triangles.size() * 3;
    blasInfo.vertexStride = 12; // 3 * float
    blasInfo.materialIndexData = materialIndices.data();
    blasInfo.materialIndexCount = (uint32_t)materialIndices.size();
    
    uint32_t blasIndex = m_device->createBLAS(blasInfo);
    m_meshRegistry[meshName] = blasIndex;

    // Reset geometry data buffer because a new BLAS was added
    if (m_device->m_geometryDataBuffer.buffer) {
        m_device->destroyBuffer(m_device->m_geometryDataBuffer);
    }
    
    SCENE_LOG_INFO("[Vulkan] Uploaded mesh: " + meshName + " (" + std::to_string(triangles.size()) + " tris)");
    return blasIndex;
}

uint32_t VulkanBackendAdapter::uploadHairStrands(const std::vector<HairStrandData>& s, const std::string& n) { (void)s; (void)n; return 0; }
void VulkanBackendAdapter::updateMeshTransform(uint32_t h, const Matrix4x4& t) { (void)h; (void)t; }

void VulkanBackendAdapter::rebuildAccelerationStructure() {
    SCENE_LOG_INFO("[Vulkan] Full scene/project rebuild triggered.");
    m_meshRegistry.clear();
    m_vkInstances.clear();
    
    if (m_device) {
        m_device->waitIdle();
        // Clear BLAS list on device side too
        for (auto& blas : m_device->m_blasList) {
            if (blas.accel && m_device->fpDestroyAccelerationStructureKHR) {
                m_device->fpDestroyAccelerationStructureKHR(m_device->m_device, blas.accel, nullptr);
            }
            m_device->destroyBuffer(blas.buffer);
            m_device->destroyBuffer(blas.vertexBuffer);
            m_device->destroyBuffer(blas.normalBuffer);
            m_device->destroyBuffer(blas.uvBuffer);
            m_device->destroyBuffer(blas.indexBuffer);
        }
        m_device->m_blasList.clear();
        
        if (m_device->m_geometryDataBuffer.buffer) m_device->destroyBuffer(m_device->m_geometryDataBuffer);
        if (m_device->m_instanceDataBuffer.buffer) m_device->destroyBuffer(m_device->m_instanceDataBuffer);
    }
    
    m_testInitialized = false; 
    resetAccumulation();
    
    if (!m_lastObjects.empty()) {
        updateGeometry(m_lastObjects);
    }
}

void VulkanBackendAdapter::showAllInstances() {}

void VulkanBackendAdapter::updateSceneGeometry(const std::vector<std::shared_ptr<Hittable>>& o, const std::vector<Matrix4x4>& b) { 
    (void)b;
    updateGeometry(o); 
}

void VulkanBackendAdapter::updateInstanceMaterialBinding(const std::string& n, int o, int nw) { (void)n; (void)o; (void)nw; }
void VulkanBackendAdapter::setVisibilityByNodeName(const std::string& n, bool v) { (void)n; (void)v; }

void VulkanBackendAdapter::updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!m_device || !m_device->isInitialized()) return;

    std::vector<VulkanRT::TLASInstance> vkInstances;
    std::vector<TriangleData> soloTriDatas;
    std::vector<std::shared_ptr<Hittable>> instanceSources;
    
    // Helper to find and upload all unique meshes recursively
    std::function<void(const std::shared_ptr<Hittable>&)> processObj;
    processObj = [&](const std::shared_ptr<Hittable>& obj) {
        if (!obj) return;
        
        // 1. Handle Instances (The primary way geometry is organized)
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->visible) return;
            
            // If we have source triangles, make sure they are uploaded as a BLAS
            if (inst->source_triangles && !inst->source_triangles->empty()) {
                std::string meshKey = inst->node_name;
                
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
                        triData.push_back(d);
                    }
                    uploadTriangles(triData, meshKey);
                }
                
                auto it = m_meshRegistry.find(meshKey);
                if (it != m_meshRegistry.end()) {
                    VulkanRT::TLASInstance vi;
                    vi.blasIndex = it->second;
                    vi.transform = inst->transform;
                    vi.materialIndex = inst->source_triangles->at(0)->getMaterialID();
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
            d.v0 = tri->getV0(); d.v1 = tri->getV1(); d.v2 = tri->getV2();
            d.n0 = tri->getN0(); d.n1 = tri->getN1(); d.n2 = tri->getN2();
            auto uv = tri->getUVCoordinates();
            d.uv0 = std::get<0>(uv); d.uv1 = std::get<1>(uv); d.uv2 = std::get<2>(uv);
            d.materialID = tri->getMaterialID();
            soloTriDatas.push_back(d);
            instanceSources.push_back(tri);
        }
    };

    for (const auto& obj : objects) {
        processObj(obj);
    }
    
    // Store for updateInstanceTransforms
    m_lastObjects = objects;

    // Handle Solo Triangles (group them into one BLAS)
    if (!soloTriDatas.empty()) {
        std::string meshKey = "[World-Solo]-" + std::to_string(soloTriDatas.size());
        uint32_t soloBlasIndex = uploadTriangles(soloTriDatas, meshKey);
        
        VulkanRT::TLASInstance vi;
        vi.blasIndex = soloBlasIndex;
        vi.transform = Matrix4x4::identity();
        vi.customIndex = 0;
        vi.mask = 0xFF;
        vi.frontFaceCCW = true;
        vkInstances.push_back(vi);
    }
    
    m_vkInstances = vkInstances; // Store for updates
    m_instanceSources = instanceSources;
    m_topology_dirty = true;
    
    if (!vkInstances.empty()) {
        VulkanRT::TLASCreateInfo tlasInfo;
        tlasInfo.instances = m_vkInstances;
        tlasInfo.allowUpdate = true; // Enable fast updates for TLAS
        m_device->createTLAS(tlasInfo);

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
        SCENE_LOG_INFO("[Vulkan] TLAS rebuilt with " + std::to_string(vkInstances.size()) + " instances.");
    } else {
        SCENE_LOG_WARN("[Vulkan] updateGeometry: No valid geometry found in the scene.");
    }
}

// Materials & Textures
void VulkanBackendAdapter::uploadMaterials(const std::vector<MaterialData>& materials) {
    if (materials.empty()) return;

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
        gm.clearcoat = m.clearcoat;
        gm.clearcoat_roughness = m.clearcoatRoughness;
        gm.translucent = m.translucent;
        gm.subsurface_anisotropy = m.subsurfaceAnisotropy;
        gm.anisotropic = m.anisotropic;
        gm.sheen = m.sheen;
        gm.sheen_tint = m.sheenTint;
        gm.flags = (uint32_t)m.flags;
        // ... getTexID mapping (cast to uint32_t for GLSL compatibility)
        auto getTexID = [this](int64_t key) -> uint32_t {
            if (!key) return 0;
            auto it = m_uploadedImageIDs.find(key);
            if (it != m_uploadedImageIDs.end()) return (uint32_t)it->second;
            Texture* tex = reinterpret_cast<Texture*>(key);
            if (!tex || !tex->is_loaded()) return 0;
            if (tex->is_hdr) {
                const std::vector<float4>& fp = tex->float_pixels;
                if (fp.empty()) return 0;
                int64_t id = this->uploadTexture2D(fp.data(), tex->width, tex->height, 4, false, true);
                if (id) { m_uploadedImageIDs[key] = id; return (uint32_t)id; }
                return 0;
            }
            const std::vector<CompactVec4>& px = tex->pixels;
            if (px.empty()) return 0;
            std::vector<uint8_t> tmp;
            tmp.resize(tex->width * tex->height * 4);
            for (size_t i = 0; i < px.size(); ++i) {
                tmp[i*4 + 0] = px[i].r; tmp[i*4 + 1] = px[i].g; tmp[i*4 + 2] = px[i].b; tmp[i*4 + 3] = px[i].a;
            }
            int64_t id = this->uploadTexture2D(tmp.data(), tex->width, tex->height, 4, tex->is_srgb, false);
            if (id) { m_uploadedImageIDs[key] = id; return (uint32_t)id; }
            return 0;
        };

        gm.albedo_tex = getTexID(m.albedoTexture);
        gm.normal_tex = getTexID(m.normalTexture);
        gm.roughness_tex = getTexID(m.roughnessTexture);
        gm.metallic_tex = getTexID(m.metallicTexture);
        gm.emission_tex = getTexID(m.emissionTexture);
        gm.transmission_tex = getTexID(m.transmissionTexture);
        gm.opacity_tex = getTexID(m.opacityTexture);
        gm.height_tex = getTexID(m.heightTexture);

        gpuMats.push_back(gm);
    }

    m_device->updateMaterialBuffer(gpuMats.data(), gpuMats.size() * sizeof(::VulkanRT::VkGpuMaterial), (uint32_t)gpuMats.size());
    resetAccumulation();
}

void VulkanBackendAdapter::uploadHairMaterials(const std::vector<HairMaterialData>& m) { (void)m; }

int64_t VulkanBackendAdapter::uploadTexture2D(const void* d, uint32_t w, uint32_t h, uint32_t c, bool s, bool f) {
    if (!d || w == 0 || h == 0) return 0;

    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t bpp = 4; // bytes per pixel

    if (f) {
        fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
        bpp = 16;
    } else if (s) {
        fmt = VK_FORMAT_R8G8B8A8_SRGB;
    }

    // Create staging buffer
    VulkanRT::BufferCreateInfo ci;
    ci.size = (uint64_t)w * h * bpp;
    ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_SRC;
    ci.location = VulkanRT::MemoryLocation::CPU_TO_GPU;

    VulkanRT::BufferHandle staging = m_device->createBuffer(ci);
    if (!staging.buffer) return 0;

    m_device->uploadBuffer(staging, d, ci.size);

    // Create image as a sampled texture (allow transfer dst for upload)
    VulkanRT::ImageHandle img = m_device->createImage2D(w, h, fmt,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!img.image) {
        m_device->destroyBuffer(staging);
        return 0;
    }

    m_device->copyBufferToImage(staging, img);

    // Transition uploaded image to SHADER_READ_ONLY_OPTIMAL for sampling in shaders
    VkCommandBuffer cmd = m_device->beginSingleTimeCommands();
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
    return id;
}

void VulkanBackendAdapter::destroyTexture(int64_t texID) {
    auto it = m_uploadedImages.find(texID);
    if (it == m_uploadedImages.end()) return;
    VulkanRT::ImageHandle& img = it->second;
    if (img.sampler) vkDestroySampler(m_device->getDevice(), img.sampler, nullptr);
    m_device->destroyImage(img);
    m_uploadedImages.erase(it);
    // Also remove any pointer -> id mappings that reference this id
    for (auto it2 = m_uploadedImageIDs.begin(); it2 != m_uploadedImageIDs.end(); ) {
        if (it2->second == texID) it2 = m_uploadedImageIDs.erase(it2);
        else ++it2;
    }
}

void VulkanBackendAdapter::setLights(const std::vector<std::shared_ptr<Light>>& lights) {
    // If the lights list is empty we still need to update the GPU so any
    // previously uploaded lights are cleared. If the device isn't ready
    // we cache the empty list for later upload when initialization completes.
    if (lights.empty()) {
        if (!m_device || !m_device->isInitialized()) {
            VK_INFO() << "[VulkanBackendAdapter] Device not ready — caching lights for later upload (0)" << std::endl;
            m_cachedLights = lights;
            return;
        }
        // Upload zero-sized light buffer / set light count to 0 to clear GPU lights
        VK_INFO() << "[VulkanBackendAdapter] Clearing GPU lights (no lights in scene)" << std::endl;
        m_device->updateLightBuffer(nullptr, 0, 0);
        resetAccumulation();
        m_cachedLights = lights;
        return;
    }

    // Safety: if device isn't ready, stash lights for later to avoid calling Vulkan with uninitialized device
    if (!m_device || !m_device->isInitialized()) {
        VK_INFO() << "[VulkanBackendAdapter] Device not ready — caching lights for later upload (" << lights.size() << ")" << std::endl;
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
        gl.params[0] = std::max(l->radius, MIN_LIGHT_RADIUS);
        gl.params[1] = std::max(l->width, MIN_AREA_DIM);
        gl.params[2] = std::max(l->height, MIN_AREA_DIM); // For area lights this is height, for spot lights this will be used for inner cone (overwritten below)

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
    
    setCamera(cp);
}

void VulkanBackendAdapter::setTime(float t, float dt) { m_currentTime = t; (void)dt; }
void VulkanBackendAdapter::updateInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) { 
    if (!m_device || !m_device->isInitialized()) return;
    if (objects.empty()) return;
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
                            if (srcInst->node_name == inst->node_name) {
                                updatedInstances[i].transform = inst->transform;
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
            
            // Wait for device to finish any pending ray tracing before modifying AS
            vkDeviceWaitIdle(m_device->getDevice());
            
            m_device->updateTLAS(m_vkInstances);
            
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
        
        // Wait for device to finish any pending ray tracing before modifying AS
        vkDeviceWaitIdle(m_device->getDevice());
        
        m_device->updateTLAS(m_vkInstances);
        
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

        // Build pointer map for O(1) matching
        std::unordered_map<void*, std::shared_ptr<Hittable>> ptr_to_obj;
        for (const auto& obj : objects) ptr_to_obj[obj.get()] = obj;

        for (size_t i = 0; i < m_instanceSources.size(); ++i) {
            VulkanBackendAdapter::InstanceTransformCache item;
            item.instance_id = (int)i;
            item.representative_hittable = nullptr;

            auto src = m_instanceSources[i];
            if (src) {
                // Direct pointer match
                if (ptr_to_obj.count(src.get())) {
                    item.representative_hittable = ptr_to_obj[src.get()];
                } else {
                    // Name-based fallback for instances
                    if (auto inst = std::dynamic_pointer_cast<HittableInstance>(src)) {
                        std::string name = inst->node_name;
                        size_t mat_pos = name.find("_mat_");
                        if (mat_pos != std::string::npos) name = name.substr(0, mat_pos);
                        for (const auto& obj : objects) {
                            if (auto oinst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                                if (oinst->node_name == name) { item.representative_hittable = oinst; break; }
                            }
                        }
                    }
                }
            }

            if (item.representative_hittable) m_instance_sync_cache.push_back(item);
        }

        m_topology_dirty = false;
    }
}

bool VulkanBackendAdapter::isUsingTLAS() const {
    return m_device && m_device->isInitialized() && m_device->hasTLAS();
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
            
            size_t mat_pos = instName.find("_mat_");
            if (mat_pos != std::string::npos) instName = instName.substr(0, mat_pos);
            
            if (instName == nodeName) {
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
        }
    }
}

void VulkanBackendAdapter::updateObjectTransform(const std::string& nodeName, const Matrix4x4& transform) {
    if (!m_device || !m_device->isInitialized()) return;
    
    bool changed = false;
    for (size_t i = 0; i < m_vkInstances.size(); ++i) {
        if (m_instanceSources.size() > i && m_instanceSources[i]) {
            std::string instName;
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                instName = inst->node_name;
            } else if (auto tri = std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
                instName = tri->getNodeName();
            }
            
            size_t mat_pos = instName.find("_mat_");
            if (mat_pos != std::string::npos) instName = instName.substr(0, mat_pos);
            
            if (instName == nodeName) {
                m_vkInstances[i].transform = transform;
                // Also update the sync cache so updateInstanceTransforms doesn't revert it
                if (m_instanceSources.size() > i) {
                     if (auto inst = std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
                          inst->transform = transform; 
                     }
                }
                changed = true;
            }
        }
    }
    
    if (changed) {
        vkDeviceWaitIdle(m_device->getDevice());
        m_device->updateTLAS(m_vkInstances);
        resetAccumulation();
    }
}
void VulkanBackendAdapter::setStatusCallback(std::function<void(const std::string&, int)> cb) { m_statusCallback = cb; }
void* VulkanBackendAdapter::getNativeCommandQueue() { return (void*)m_device->getComputeQueue(); }

void VulkanBackendAdapter::renderPass(bool accumulate) { (void)accumulate; /* TODO */ }
void VulkanBackendAdapter::renderProgressive(void* s, void* w, void* r, int width, int height, void* fb, void* tex) {
    (void)s; (void)w; (void)r; 
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

    // Use robust shader dir detection
    std::string shaderDir = "shaders";
    if (!std::filesystem::exists(shaderDir + "/raygen.spv"))
        shaderDir = "/shaders"; // Fallback if running directly from project root
    if (!std::filesystem::exists(shaderDir + "/raygen.spv"))
        shaderDir = "/shaders"; 

    // 1. Recreate output image if size changed
    if (m_imageWidth != width || m_imageHeight != height) {
        if (m_outputImage.image) m_device->destroyImage(m_outputImage);
        if (m_stagingBuffer.buffer) m_device->destroyBuffer(m_stagingBuffer);

        // Use 32-bit float RGBA accumulation image so shaders can store HDR + variance
        m_outputImage = m_device->createImage2D(width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        VulkanRT::BufferCreateInfo stagingInfo;
        stagingInfo.size = (uint64_t)width * height * 4 * sizeof(float);
        stagingInfo.usage = VulkanRT::BufferUsage::TRANSFER_DST;
        stagingInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
        m_stagingBuffer = m_device->createBuffer(stagingInfo);
        
        m_imageWidth = width;
        m_imageHeight = height;
        resetAccumulation();
    }

    // 2. Build Pipeline/Resources lazy
    if (!this->m_testInitialized) {
        this->m_testInitialized = true;

        using namespace VulkanRT;
        
        // Load RT Shaders
        std::vector<uint32_t> raygenSPV = loadSPV(shaderDir + "/raygen.spv");
        std::vector<uint32_t> missSPV = loadSPV(shaderDir + "/miss.spv");
        std::vector<uint32_t> chitSPV = loadSPV(shaderDir + "/closesthit.spv");
        std::vector<uint32_t> ahitSPV;
        if (std::filesystem::exists(shaderDir + "/shadow_anyhit.spv")) ahitSPV = loadSPV(shaderDir + "/shadow_anyhit.spv");

        if (!m_device->createRTPipeline(raygenSPV, missSPV, chitSPV, ahitSPV)) {
            SCENE_LOG_ERROR("[Vulkan] Failed to create RT Pipeline.");
            return;
        }
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
        float pad1;
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

    pushConst.shakeEnabled = this->m_camera.shake_enabled ? 1 : 0;
    if (pushConst.shakeEnabled) {
        float time = (float)SDL_GetTicks() / 1000.0f;
        float freq = this->m_camera.shake_frequency;
        float intensity = this->m_camera.shake_intensity;
        
        // Simplified shake for Vulkan (mirroring OptixWrapper logic)
        pushConst.shakeOffsetX = sinf(time * freq * 1.0f) * this->m_camera.handheld_sway_amplitude * intensity;
        pushConst.shakeOffsetY = sinf(time * freq * 1.3f + 1.5f) * this->m_camera.handheld_sway_amplitude * intensity;
        pushConst.shakeOffsetZ = sinf(time * freq * 0.7f + 3.0f) * this->m_camera.handheld_sway_amplitude * intensity * 0.3f;
        
        pushConst.shakeRotX = sinf(time * freq * 1.1f) * 0.003f * intensity;
        pushConst.shakeRotY = sinf(time * freq * 0.9f + 1.0f) * 0.003f * intensity;
        pushConst.shakeRotZ = sinf(time * freq * 0.5f + 2.0f) * 0.001f * intensity;
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
        }

        m_device->bindRTDescriptors(m_outputImage);
        m_device->setPushConstants(&pushConst, sizeof(CameraPushConstants));
        
        m_device->traceRays(width, height, 1);
        m_device->waitIdle();

        // 4. Download and Update SDL
        m_device->copyImageToBuffer(m_outputImage, m_stagingBuffer);
        m_device->waitIdle();

        std::vector<uint32_t>* framebuffer = static_cast<std::vector<uint32_t>*>(fb);
        if (framebuffer->size() != (size_t)(width * height)) {
            framebuffer->resize(width * height);
        }

        // If the output image is float RGBA, download HDR floats and tonemap on CPU
        if (m_outputImage.format == VK_FORMAT_R32G32B32A32_SFLOAT) {
            // Resize hdr buffer
            m_hdrPixels.resize((size_t)width * (size_t)height * 4);
            m_device->downloadBuffer(m_stagingBuffer, m_hdrPixels.data(), (uint64_t)width * height * 4 * sizeof(float));

            // Convert HDR floats -> 8-bit sRGB packed pixels
            SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);
            for (int j = 0; j < height; ++j) {
                for (int i = 0; i < width; ++i) {
                    size_t idx = (size_t)j * (size_t)width + (size_t)i;
                    float exposure = this->m_camera.exposureFactor > 0.0f ? this->m_camera.exposureFactor : 1.0f;
                    float r = m_hdrPixels[idx * 4 + 0] * exposure;
                    float g = m_hdrPixels[idx * 4 + 1] * exposure;
                    float b = m_hdrPixels[idx * 4 + 2] * exposure;

                    // Simple Reinhard tonemap + sRGB gamma
                    auto tonemap = [](float v){ return v / (v + 1.0f); };
                    // Use same gamma approximation as CUDA/OptiX path for parity (pow 1/2.2)
                    auto toSRGB_fast = [](float c) {
                        return std::pow(c, 1.0f / 2.2f);
                    };

                    float rr = tonemap(r);
                    float gg = tonemap(g);
                    float bb = tonemap(b);

                    int ri = static_cast<int>(255.0f * std::clamp(toSRGB_fast(std::max(0.0f, rr)), 0.0f, 1.0f));
                    int gi = static_cast<int>(255.0f * std::clamp(toSRGB_fast(std::max(0.0f, gg)), 0.0f, 1.0f));
                    int bi = static_cast<int>(255.0f * std::clamp(toSRGB_fast(std::max(0.0f, bb)), 0.0f, 1.0f));

                    uint32_t packed = SDL_MapRGB(fmt, ri, gi, bi);
                    (*framebuffer)[idx] = packed;

                    // Write to SDL surface if provided (flip Y as expected by UI)
                    if (s) {
                        SDL_Surface* outSurf = static_cast<SDL_Surface*>(s);
                        if (outSurf->pixels && outSurf->w == width && outSurf->h == height) {
                            Uint32* pixels_ptr = static_cast<Uint32*>(outSurf->pixels);
                            size_t screen_idx = (size_t)j * (size_t)width + (size_t)i;
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

    // Debug readback: verify skyview image contents match host LUT (sample first texel)
    if (lutImgs[1].image && !lut->getHostSkyView().empty()) {
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
}

void VulkanBackendAdapter::setWorldData(const void* w) {
    if (!w) return;
    
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

    // ═════════════════════════════════════════════════════════════════
    // VOLUMETRIC GOD RAYS
    // ═════════════════════════════════════════════════════════════════
    gw.godRaysEnabled = wd->nishita.godrays_enabled ? 1 : 0;
    gw.godRaysIntensity = wd->nishita.godrays_intensity;
    gw.godRaysDensity = wd->nishita.godrays_density;
    gw.godRaysSamples = wd->nishita.godrays_samples;

    // ═════════════════════════════════════════════════════════════════
    // ENVIRONMENT & LUT REFERENCES
    // ═════════════════════════════════════════════════════════════════
    gw.envTexSlot = (int)m_envTexID;
    gw.envIntensity = wd->env_intensity;
    gw.envRotation = wd->env_rotation;
    gw._pad5 = 0;
    
    // LUT handles - if AtmosphereLUT was precomputed, these will be valid GPU texture objects
    // Otherwise, shaders will fall back to on-the-fly computation
    gw.transmittanceLUT = wd->lut.transmittance_lut;
    gw.skyviewLUT = wd->lut.skyview_lut;
    gw.multiScatterLUT = wd->lut.multi_scattering_lut;
    gw.aerialPerspectiveLUT = wd->lut.aerial_perspective_lut;

    // Debug: print incoming LUT handles and packed GW values
    VK_INFO() << "[VulkanBackendAdapter] setWorldData - wd->lut trans=" << wd->lut.transmittance_lut
              << " sky=" << wd->lut.skyview_lut << " multi=" << wd->lut.multi_scattering_lut
              << " aerial=" << wd->lut.aerial_perspective_lut << std::endl;
    VK_INFO() << "[VulkanBackendAdapter] setWorldData - gw.skyviewLUT low32=" << (uint32_t)(gw.skyviewLUT & 0xFFFFFFFFULL)
              << " high32=" << (uint32_t)((gw.skyviewLUT >> 32) & 0xFFFFFFFFULL) << std::endl;
    VK_INFO() << "[VulkanBackendAdapter] setWorldData - mode=" << gw.mode << " sunIntensity=" << gw.sunIntensity
              << " sunColor=(" << gw.sunColor[0] << "," << gw.sunColor[1] << "," << gw.sunColor[2] << ")"
              << " airDensity=" << gw.airDensity << " dustDensity=" << gw.dustDensity << std::endl;

    m_device->updateWorldBuffer(&gw, sizeof(gw), 1);
    resetAccumulation();
}
void VulkanBackendAdapter::updateVDBVolumes(const std::vector<GpuVDBVolume>& v) { (void)v; }
void VulkanBackendAdapter::updateGasVolumes(const std::vector<GpuGasVolume>& v) { (void)v; }

// Utility
void VulkanBackendAdapter::waitForCompletion() { m_device->waitIdle(); }
void VulkanBackendAdapter::resetAccumulation() {
    m_currentSamples = 0;
    // Also clear the output image to avoid ghosting when accumulation restarts
    if (m_outputImage.image && m_device) {
        m_device->clearImage(m_outputImage, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    // Request UI-level clear on next present so host-side view is immediately cleared
    m_forceClearOnNextPresent = true;
}
float VulkanBackendAdapter::getMillisecondsPerSample() const { return 0.0f; }

} // namespace Backend
