/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          VulkanBackend.h
 * Description:   Cross-platform Vulkan Compute/RT Backend Interface
 *                Abstracts GPU operations for AMD/Intel/NVIDIA/Apple(MoltenVK)
 * 
 * Why Vulkan over OpenCL:
 *   1. Ray tracing extensions (VK_KHR_ray_tracing_pipeline)
 *   2. macOS support via MoltenVK (OpenCL deprecated on Apple)
 *   3. Unified graphics + compute in single API
 *   4. Active development by Khronos
 *   5. Better driver support across vendors
 * =========================================================================
 */
#ifndef VULKAN_BACKEND_H
#define VULKAN_BACKEND_H

// Prevent Vulkan from loading its own headers if we include this first
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "Vec3.h"
#include "Matrix4x4.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace VulkanRT {

// ============================================================================
// Capability Detection
// ============================================================================

enum class GPUVendor : uint8_t {
    UNKNOWN,
    NVIDIA,
    AMD,
    INTEL,
    APPLE,      // MoltenVK on macOS
    QUALCOMM,   // Mobile
    ARM         // Mali
};

enum class RayTracingMode : uint8_t {
    NONE,           // No RT support (fallback to compute)
    COMPUTE,        // Software RT via compute shaders
    HARDWARE_KHR,   // VK_KHR_ray_tracing_pipeline (AMD RDNA2+, Intel Arc, NVIDIA)
    HARDWARE_NV     // VK_NV_ray_tracing (older NVIDIA extension)
};

struct GPUCapabilities {
    GPUVendor vendor;
    RayTracingMode rtMode;
    
    std::string deviceName;
    uint32_t apiVersion;
    uint32_t driverVersion;
    
    // Memory
    uint64_t dedicatedVRAM;         // Bytes
    uint64_t sharedSystemMemory;
    uint32_t maxBufferSize;
    
    // Compute
    uint32_t maxComputeWorkGroupSize[3];
    uint32_t maxComputeWorkGroupCount[3];
    uint32_t subgroupSize;          // Warp/wavefront size
    
    // Ray tracing (if supported)
    uint32_t maxRayRecursionDepth;
    uint32_t shaderGroupHandleSize;
    bool supportsRayQuery;          // Inline RT in any shader stage
    bool supportsMotionBlur;
    
    // Features
    bool supports16BitFloat;
    bool supportsInt64Atomics;
    bool supportsBufferDeviceAddress;
    bool supportsDescriptorIndexing;
};

// ============================================================================
// Buffer Management
// ============================================================================

enum class BufferUsage : uint32_t {
    VERTEX          = 0x0001,
    INDEX           = 0x0002,
    UNIFORM         = 0x0004,
    STORAGE         = 0x0008,
    TRANSFER_SRC    = 0x0010,
    TRANSFER_DST    = 0x0020,
    ACCELERATION    = 0x0040,   // For RT acceleration structures
    SHADER_BINDING  = 0x0080    // For shader binding table
};

inline BufferUsage operator|(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

enum class MemoryLocation : uint8_t {
    GPU_ONLY,       // Device local (fastest)
    CPU_TO_GPU,     // Host visible, device local (staging)
    GPU_TO_CPU,     // For readback
    CPU_ONLY        // Host only (debugging)
};

struct BufferCreateInfo {
    uint64_t size;
    BufferUsage usage;
    MemoryLocation location;
    const void* initialData;    // Optional
};

struct BufferHandle {
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint64_t size;
    VkDeviceAddress deviceAddress;  // For buffer device address
};

// ============================================================================
// Acceleration Structure
// ============================================================================

struct BLASCreateInfo {
    // Triangle geometry
    const float* vertexData;        // [x,y,z, x,y,z, ...]
    uint32_t vertexCount;
    uint32_t vertexStride;          // Bytes between vertices
    
    const uint32_t* indexData;      // Optional, null for non-indexed
    uint32_t indexCount;
    
    // Or curve geometry (for hair)
    bool isCurve;
    const float* curveControlPoints;
    const float* curveRadii;
    const uint32_t* curveSegmentOffsets;
    uint32_t curveCount;
    
    bool allowUpdate;               // For dynamic geometry
};

struct TLASInstance {
    uint32_t blasIndex;
    Matrix4x4 transform;
    uint32_t customIndex;           // User data (e.g., material ID)
    uint8_t mask;                   // Visibility mask
    bool frontFaceCCW;
};

struct TLASCreateInfo {
    std::vector<TLASInstance> instances;
    bool allowUpdate;
};

// ============================================================================
// Shader Management
// ============================================================================

enum class ShaderStage : uint8_t {
    COMPUTE,
    RAYGEN,
    MISS,
    CLOSEST_HIT,
    ANY_HIT,
    INTERSECTION
};

struct ShaderModuleInfo {
    ShaderStage stage;
    std::vector<uint32_t> spirvCode;    // SPIR-V bytecode
    std::string entryPoint;
};

struct PipelineCreateInfo {
    std::vector<ShaderModuleInfo> shaders;
    uint32_t maxRayRecursion;
    
    // Push constants
    uint32_t pushConstantSize;
    
    // Descriptor set layout
    // (simplified - real implementation would have more detail)
};

// ============================================================================
// Main Backend Interface
// ============================================================================

class VulkanBackend {
public:
    VulkanBackend();
    ~VulkanBackend();
    
    // ========================================================================
    // Initialization
    // ========================================================================
    
    /**
     * @brief Initialize Vulkan with optional ray tracing
     * @param preferHardwareRT Try to use hardware RT if available
     * @param validationLayers Enable Vulkan validation (debug builds)
     * @return true if successful
     */
    bool initialize(bool preferHardwareRT = true, bool validationLayers = false);
    
    /**
     * @brief Get detected GPU capabilities
     */
    const GPUCapabilities& getCapabilities() const { return m_capabilities; }
    
    /**
     * @brief Check if hardware ray tracing is available
     */
    bool hasHardwareRT() const { 
        return m_capabilities.rtMode == RayTracingMode::HARDWARE_KHR ||
               m_capabilities.rtMode == RayTracingMode::HARDWARE_NV;
    }
    
    // ========================================================================
    // Buffer Operations
    // ========================================================================
    
    BufferHandle createBuffer(const BufferCreateInfo& info);
    void destroyBuffer(BufferHandle& buffer);
    
    void* mapBuffer(const BufferHandle& buffer);
    void unmapBuffer(const BufferHandle& buffer);
    
    void uploadBuffer(const BufferHandle& dst, const void* data, uint64_t size, uint64_t offset = 0);
    void downloadBuffer(const BufferHandle& src, void* data, uint64_t size, uint64_t offset = 0);
    
    // ========================================================================
    // Acceleration Structures (Ray Tracing)
    // ========================================================================
    
    /**
     * @brief Build bottom-level acceleration structure (geometry)
     */
    uint32_t createBLAS(const BLASCreateInfo& info);
    
    /**
     * @brief Build top-level acceleration structure (instances)
     */
    void createTLAS(const TLASCreateInfo& info);
    
    /**
     * @brief Update existing BLAS (for animation)
     */
    void updateBLAS(uint32_t blasIndex, const float* newVertices);
    
    /**
     * @brief Rebuild TLAS with new transforms
     */
    void updateTLAS(const std::vector<TLASInstance>& instances);
    
    // ========================================================================
    // Pipeline Management
    // ========================================================================
    
    /**
     * @brief Create ray tracing or compute pipeline
     */
    uint32_t createPipeline(const PipelineCreateInfo& info);
    
    /**
     * @brief Bind pipeline for execution
     */
    void bindPipeline(uint32_t pipelineIndex);
    
    // ========================================================================
    // Dispatch / Trace
    // ========================================================================
    
    /**
     * @brief Dispatch compute shader
     */
    void dispatchCompute(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ);
    
    /**
     * @brief Trace rays (ray tracing pipeline)
     */
    void traceRays(uint32_t width, uint32_t height, uint32_t depth = 1);
    
    /**
     * @brief Set push constants
     */
    void setPushConstants(const void* data, uint32_t size);
    
    // ========================================================================
    // Synchronization
    // ========================================================================
    
    void waitIdle();
    void submitAndWait();
    
    // ========================================================================
    // Image/Texture Support
    // ========================================================================
    
    struct ImageHandle {
        VkImage image;
        VkImageView view;
        VkDeviceMemory memory;
        VkSampler sampler;
        uint32_t width, height;
    };
    
    ImageHandle createImage2D(uint32_t width, uint32_t height, VkFormat format);
    void destroyImage(ImageHandle& image);
    
    void copyImageToBuffer(const ImageHandle& src, const BufferHandle& dst);
    void copyBufferToImage(const BufferHandle& src, const ImageHandle& dst);
    
private:
    // Vulkan handles
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    
    // Ray tracing handles
    VkAccelerationStructureKHR m_tlas = VK_NULL_HANDLE;
    std::vector<VkAccelerationStructureKHR> m_blasList;
    
    // Pipelines
    std::vector<VkPipeline> m_pipelines;
    std::vector<VkPipelineLayout> m_pipelineLayouts;
    
    // Detected capabilities
    GPUCapabilities m_capabilities;
    
    // Extension function pointers (loaded dynamically)
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR = nullptr;
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR = nullptr;
    
    // Internal helpers
    bool selectPhysicalDevice();
    bool createLogicalDevice();
    void loadRayTracingFunctions();
    void detectCapabilities();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

// ============================================================================
// Factory Function
// ============================================================================

/**
 * @brief Create the best available backend
 * Tries Vulkan first, falls back to compute-only if no RT
 */
std::unique_ptr<VulkanBackend> createBestBackend();

} // namespace VulkanRT

#endif // VULKAN_BACKEND_H
