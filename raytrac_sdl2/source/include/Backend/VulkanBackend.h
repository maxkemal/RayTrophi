/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          VulkanBackend.h
 * Description:   Cross-platform Vulkan Compute/RT Backend
 *                Implements IBackend for AMD/Intel/NVIDIA/Apple(MoltenVK)
 * 
 * Architecture:
 *   VulkanRT::VulkanDevice  - Low-level Vulkan device management
 *   Backend::VulkanBackendAdapter - IBackend implementation using VulkanDevice
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

#include <vulkan/vulkan.h>

#include "Backend/IBackend.h"
#include "vulkan_material_types.h"
#include "Vec3.h"
#include "Matrix4x4.h"
#include "World.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include "AtmosphereLUT.h"
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
    ARM_MALI    // Mali
};

enum class RayTracingMode : uint8_t {
    NONE,           // No RT support (fallback to compute)
    COMPUTE,        // Software RT via compute shaders
    HARDWARE_KHR,   // VK_KHR_ray_tracing_pipeline (AMD RDNA2+, Intel Arc, NVIDIA)
    HARDWARE_NV     // VK_NV_ray_tracing (older NVIDIA extension)
};

struct GPUCapabilities {
    GPUVendor vendor = GPUVendor::UNKNOWN;
    RayTracingMode rtMode = RayTracingMode::NONE;
    
    std::string deviceName;
    uint32_t apiVersion = 0;
    uint32_t driverVersion = 0;
    
    // Memory
    uint64_t dedicatedVRAM = 0;         // Bytes
    uint64_t sharedSystemMemory = 0;
    uint32_t maxBufferSize = 0;
    
    // Compute
    uint32_t maxComputeWorkGroupSize[3] = {0, 0, 0};
    uint32_t maxComputeWorkGroupCount[3] = {0, 0, 0};
    uint32_t subgroupSize = 0;          // Warp/wavefront size
    
    // Ray tracing (if supported)
    uint32_t maxRayRecursionDepth = 0;
    uint32_t shaderGroupHandleSize = 0;
    uint32_t shaderGroupBaseAlignment = 0;
    uint32_t minScratchAlignment = 128;
    bool supportsRayQuery = false;          // Inline RT in any shader stage
    bool supportsMotionBlur = false;
    
    // Features
    bool supports16BitFloat = false;
    bool supportsInt64Atomics = false;
    bool supportsBufferDeviceAddress = false;
    bool supportsDescriptorIndexing = false;
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

inline BufferUsage operator&(BufferUsage a, BufferUsage b) {
    return static_cast<BufferUsage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

enum class MemoryLocation : uint8_t {
    GPU_ONLY,       // Device local (fastest)
    CPU_TO_GPU,     // Host visible, device local (staging)
    GPU_TO_CPU,     // For readback
    CPU_ONLY        // Host only (debugging)
};

struct BufferCreateInfo {
    uint64_t size = 0;
    BufferUsage usage = BufferUsage::STORAGE;
    MemoryLocation location = MemoryLocation::GPU_ONLY;
    const void* initialData = nullptr;    // Optional
};

struct BufferHandle {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t size = 0;
    VkDeviceAddress deviceAddress = 0;  // For buffer device address
    void* mappedPtr = nullptr;          // Non-null if persistently mapped
};

// ============================================================================
// Acceleration Structure
// ============================================================================

struct BLASCreateInfo {
    // Triangle geometry
    const float* vertexData = nullptr;        // [x,y,z, x,y,z, ...]
    const float* normalData = nullptr;        // [nx,ny,nz, ...]
    const float* uvData = nullptr;            // [u,v, ...]
    uint32_t vertexCount = 0;
    uint32_t vertexStride = 12;               // Bytes between vertices (default: 3 floats)
    
    const uint32_t* indexData = nullptr;      // Optional, null for non-indexed
    uint32_t indexCount = 0;
    // Optional per-primitive material indices (one entry per triangle)
    const uint32_t* materialIndexData = nullptr;
    uint32_t materialIndexCount = 0;
    
    // Or curve geometry (for hair)
    bool isCurve = false;
    const float* curveControlPoints = nullptr;
    const float* curveRadii = nullptr;
    const uint32_t* curveSegmentOffsets = nullptr;
    uint32_t curveCount = 0;
    
    bool allowUpdate = false;               // For dynamic geometry
};

struct VkGeometryData {
    uint64_t vertexAddr;
    uint64_t normalAddr;
    uint64_t uvAddr;
    uint64_t indexAddr;
    uint64_t materialAddr; // optional device address of per-primitive material index array
};

struct TLASInstance {
    uint32_t blasIndex = 0;
    Matrix4x4 transform;
    uint32_t materialIndex = 0;
    uint32_t customIndex = 0;           // Extra user data
    uint8_t mask = 0xFF;                // Visibility mask
    bool frontFaceCCW = true;
};

struct VkInstanceData {
    uint32_t materialIndex;
    uint32_t blasIndex;
};

struct TLASCreateInfo {
    std::vector<TLASInstance> instances;
    bool allowUpdate = false;
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
    std::string entryPoint = "main";
};

struct PipelineCreateInfo {
    std::vector<ShaderModuleInfo> shaders;
    uint32_t maxRayRecursion = 1;
    
    // Push constants
    uint32_t pushConstantSize = 0;
    
    // Descriptor layouts will be managed internally
};

// ============================================================================
// Acceleration Structure Handle
// ============================================================================

struct AccelStructHandle {
    VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
    BufferHandle buffer;    // Backing buffer (AS data)
    BufferHandle vertexBuffer;
    BufferHandle normalBuffer;
    BufferHandle uvBuffer;
    BufferHandle indexBuffer;
    BufferHandle materialIndexBuffer;
    VkDeviceAddress deviceAddress = 0;
};

// ============================================================================
// Image Handle
// ============================================================================

struct ImageHandle {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

// ============================================================================
// Main Low-Level Vulkan Device
// ============================================================================

class VulkanDevice {
public:
    VulkanDevice();
    ~VulkanDevice();
    
    // Non-copyable
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

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
     * @brief Clean up all Vulkan resources
     */
    void shutdown();
    
    /**
     * @brief Check if device is initialized
     */
    bool isInitialized() const { return m_device != VK_NULL_HANDLE; }
    
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

    bool isRTReady() const { return m_rtPipelineReady; }
    bool hasTLAS() const { return m_tlas.accel != VK_NULL_HANDLE; }
    
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
     * @brief Create ray tracing pipeline from raygen/miss/closesthit SPIR-V
     * @return true on success
     */
    bool createRTPipeline(const std::vector<uint32_t>& raygenSPV,
                          const std::vector<uint32_t>& missSPV,
                          const std::vector<uint32_t>& closestHitSPV,
                          const std::vector<uint32_t>& anyHitSPV = std::vector<uint32_t>());
    
    /**
     * @brief Bind RT descriptors (output image + TLAS)
     */
    void bindRTDescriptors(const ImageHandle& outputImage);
    void updateRTTextureDescriptor(uint32_t slot, const ImageHandle& image);
    void clearImage(const ImageHandle& image, float r, float g, float b, float a);
    
    /**
     * @brief Bind pipeline for execution
     */
    void bindPipeline(uint32_t pipelineIndex);
    
    // ========================================================================
    // Command Execution
    // ========================================================================
    
    /**
     * @brief Begin recording commands
     */
    VkCommandBuffer beginSingleTimeCommands();
    
    /**
     * @brief End recording and submit commands  
     */
    void endSingleTimeCommands(VkCommandBuffer cmdBuf);
    
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
    
    /**
     * @brief Bind a storage image to a pipeline's descriptor set
     */
    void bindStorageImage(uint32_t pipelineIndex, uint32_t bindingIndex, const ImageHandle& image);

    // Update a single combined image sampler entry in the RT descriptor set (binding 6)
  
    
    // ========================================================================
    // Image/Texture Support
    // ========================================================================
    
    ImageHandle createImage2D(uint32_t width, uint32_t height, VkFormat format,
                              VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    void destroyImage(ImageHandle& image);
    
    void copyImageToBuffer(const ImageHandle& src, const BufferHandle& dst);
    void copyBufferToImage(const BufferHandle& src, const ImageHandle& dst);
    
    void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                               VkImageLayout oldLayout, VkImageLayout newLayout);
    
    // ========================================================================
    // Synchronization
    // ========================================================================
    
    void waitIdle();
    void submitAndWait();
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    VkDevice getDevice() const { return m_device; }
    VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
    VkInstance getInstance() const { return m_instance; }
    VkQueue getComputeQueue() const { return m_computeQueue; }
    uint32_t getComputeQueueFamily() const { return m_computeQueueFamily; }
    // SSBO Update helpers
    void updateMaterialBuffer(const void* data, uint64_t size, uint32_t count);
    void updateLightBuffer(const void* data, uint64_t size, uint32_t count);
    void updateWorldBuffer(const void* data, uint64_t size, uint32_t count);
    void updateAtmosphereLUTs(const ImageHandle* lutImages);  // uint256_t array of 4 LUT textures
    
    // RT Resources (SSBOs)
    VulkanRT::BufferHandle m_materialBuffer;
    VulkanRT::BufferHandle m_lightBuffer;
    VulkanRT::BufferHandle m_geometryDataBuffer; // SSBO containing VkGeometryData for each BLAS
    VulkanRT::BufferHandle m_instanceDataBuffer; // SSBO containing VkInstanceData for each TLAS instance
    VulkanRT::BufferHandle m_tlasInstanceBuffer; // Buffer containing VkAccelerationStructureInstanceKHR for TLAS building
    VulkanRT::BufferHandle m_worldBuffer; // SSBO containing complete Nishita parameters
    
    // Atmosphere LUT Textures (for raygen/miss shaders)
    // [0] = transmittance_lut, [1] = skyview_lut, [2] = multi_scatter_lut, [3] = aerial_perspective_lut
    ImageHandle m_lutImages[4];
    
    uint32_t m_materialCount = 0;
    uint32_t m_lightCount = 0;
    std::vector<AccelStructHandle> m_blasList;
    // Pending texture descriptor updates (slot -> ImageHandle) queued until RT descriptor set exists
    std::vector<std::pair<uint32_t, ImageHandle>> m_pendingTextureDescriptors;
    // Extension function pointers (loaded dynamically per-device)
    PFN_vkCreateAccelerationStructureKHR fpCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR fpDestroyAccelerationStructureKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR fpCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR fpGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR fpGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCmdTraceRaysKHR fpCmdTraceRaysKHR = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR fpCreateRayTracingPipelinesKHR = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR fpGetRayTracingShaderGroupHandlesKHR = nullptr;
    PFN_vkGetBufferDeviceAddressKHR fpGetBufferDeviceAddressKHR = nullptr;
    VkDevice m_device = VK_NULL_HANDLE;
    VkDescriptorSet m_rtDescriptorSet = VK_NULL_HANDLE;
private:
    // Core Vulkan handles
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
   
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    uint32_t m_computeQueueFamily = 0;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    
    // Debug
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    
    // Acceleration structures
    AccelStructHandle m_tlas;
    
    
    // Compute Pipelines
    std::vector<VkPipeline> m_pipelines;
    std::vector<VkPipelineLayout> m_pipelineLayouts;
    uint32_t m_activePipeline = UINT32_MAX;
    
    // RT Pipeline (separate from compute)
    VkPipeline m_rtPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_rtPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_rtDescriptorSetLayout = VK_NULL_HANDLE;
   
    bool m_rtPipelineReady = false;
    
    // Shader Binding Table
    BufferHandle m_sbtBuffer;
    VkStridedDeviceAddressRegionKHR m_sbtRaygenRegion{};
    VkStridedDeviceAddressRegionKHR m_sbtMissRegion{};
    VkStridedDeviceAddressRegionKHR m_sbtHitRegion{};
    VkStridedDeviceAddressRegionKHR m_sbtCallableRegion{};
    

    
    // Active command buffer (for recording)
    VkCommandBuffer m_activeCommandBuffer = VK_NULL_HANDLE;
    
    // Descriptor set layouts (one per compute pipeline)
    std::vector<VkDescriptorSetLayout> m_descriptorSetLayouts;
    
    // Active descriptor sets for current dispatch
    std::vector<VkDescriptorSet> m_activeDescriptorSets;
    
    // Push constant staging buffer
    std::vector<uint8_t> m_pushConstantData;
    
    // Detected capabilities
    GPUCapabilities m_capabilities;
    
    
    
  
    
    // Internal helpers
    bool createInstance(bool validationLayers);
    bool selectPhysicalDevice(bool preferHardwareRT);
    bool createLogicalDevice(bool preferHardwareRT);
    bool createCommandPool();
    bool createDescriptorPool();
    void loadRayTracingFunctions();
    void detectCapabilities();
    void setupDebugMessenger();
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    
    GPUVendor vendorFromID(uint32_t vendorID);
    
    VkBufferUsageFlags translateBufferUsage(BufferUsage usage);
    VkMemoryPropertyFlags translateMemoryLocation(MemoryLocation location);
};

// ============================================================================
// Factory Function
// ============================================================================

/**
 * @brief Create a VulkanDevice with best available capabilities
 */
std::unique_ptr<VulkanDevice> createVulkanDevice(bool preferHardwareRT = true, bool validation = false);

} // namespace VulkanRT


// ============================================================================
// IBackend Adapter
// ============================================================================

namespace Backend {




/**
 * @brief Vulkan implementation of IBackend
 * 
 * Bridges the IBackend interface to VulkanRT::VulkanDevice.
 * Similar to how OptixBackend wraps OptixWrapper.
 */
class VulkanBackendAdapter : public IBackend {
public:
    VulkanBackendAdapter();
    virtual ~VulkanBackendAdapter();

    // ========================================================================
    // IBackend - Initialization
    // ========================================================================
    bool initialize() override;
    void shutdown() override;
    void loadShaders(const ShaderProgramData& data) override;
    BackendInfo getInfo() const override;

    // ========================================================================
    // IBackend - Geometry Upload
    // ========================================================================
    uint32_t uploadTriangles(const std::vector<TriangleData>& triangles, const std::string& meshName) override;
    uint32_t uploadHairStrands(const std::vector<HairStrandData>& strands, const std::string& groomName) override;
    void updateMeshTransform(uint32_t meshHandle, const Matrix4x4& transform) override;
    void rebuildAccelerationStructure() override;
    void showAllInstances() override;
    void updateSceneGeometry(const std::vector<std::shared_ptr<Hittable>>& objects, const std::vector<Matrix4x4>& boneMatrices) override;
    void updateInstanceMaterialBinding(const std::string& nodeName, int oldMatID, int newMatID) override;
    void setVisibilityByNodeName(const std::string& nodeName, bool visible) override;
    void updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) override;

    // ========================================================================
    // IBackend - Materials & Textures
    // ========================================================================
    void uploadMaterials(const std::vector<MaterialData>& materials) override;
    void uploadHairMaterials(const std::vector<HairMaterialData>& materials) override;
    int64_t uploadTexture2D(const void* data, uint32_t width, uint32_t height, uint32_t channels, bool sRGB, bool isFloat = false) override;
    void destroyTexture(int64_t textureHandle) override;

    // ========================================================================
    // IBackend - Rendering
    // ========================================================================
    void setRenderParams(const RenderParams& params) override;
    void setCamera(const CameraParams& camera) override;
    void syncCamera(const Camera& cam) override;
    void setTime(float time, float deltaTime) override;
    void updateInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) override;
    bool isUsingTLAS() const override;
    std::vector<int> getInstancesByNodeName(const std::string& nodeName) const override;
    void updateInstanceTransform(int instance_id, const float transform[12]) override;
    void updateObjectTransform(const std::string& nodeName, const Matrix4x4& transform) override;
    void setStatusCallback(std::function<void(const std::string&, int)> callback) override;
    void* getNativeCommandQueue() override;
    void renderPass(bool accumulate = true) override;
    void renderProgressive(void* outSurface, void* outWindow, void* outRenderer,
                           int width, int height, void* outFramebuffer, void* outTexture) override;
    void downloadImage(void* outPixels) override;
    int getCurrentSampleCount() const override;
    bool isAccumulationComplete() const override;

    // ========================================================================
    // IBackend - Environment
    // ========================================================================
    void setEnvironmentMap(int64_t hdrTextureHandle) override;
    void setSkyParams() override;
    void setLights(const std::vector<std::shared_ptr<Light>>& lights) override;
    void setWorldData(const void* worldData) override;
    void updateVDBVolumes(const std::vector<GpuVDBVolume>& volumes) override;
    void updateGasVolumes(const std::vector<GpuGasVolume>& volumes) override;

    // Internal helper to sync instance transforms efficiently
    void syncInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects, bool force_rebuild_cache = false);

    // ========================================================================
    // IBackend - Utility
    // ========================================================================
    void waitForCompletion() override;
    void resetAccumulation() override;
    float getMillisecondsPerSample() const override;

    // ========================================================================
    // Vulkan-specific access
    // ========================================================================
    VulkanRT::VulkanDevice* getVulkanDevice() { return m_device.get(); }

    /**
     * @brief Upload Atmosphere LUT host arrays into Vulkan images and bind to descriptor slot 8
     * @param lut Pointer to AtmosphereLUT (may be nullptr)
     */
    void uploadAtmosphereLUT(const AtmosphereLUT* lut);

private:
    std::unique_ptr<VulkanRT::VulkanDevice> m_device;
    
    // Render state
    int m_imageWidth = 0;
    int m_imageHeight = 0;
    int m_currentSamples = 0;
    int m_targetSamples = 0;
    float m_currentTime = 0.0f;
    bool m_testInitialized = false;
    CameraParams m_camera;
    bool m_useAdaptiveSampling = false;
    float m_varianceThreshold = 0.05f;
    std::vector<float> m_hdrPixels;   // Float32 HDR readback buffer — her frame realloc önler
    // Output resources
    VulkanRT::ImageHandle m_outputImage;
    VulkanRT::BufferHandle m_stagingBuffer;
    
    // Status callback
    std::function<void(const std::string&, int)> m_statusCallback;
    
    // Mesh registry (meshName -> BLAS index)
    std::unordered_map<std::string, uint32_t> m_meshRegistry;

    // TLAS storage for updates
    std::vector<VulkanRT::TLASInstance> m_vkInstances;
    std::vector<std::shared_ptr<Hittable>> m_lastObjects;
    // Instance source pointers for fast sync (parallel to m_vkInstances)
    std::vector<std::shared_ptr<Hittable>> m_instanceSources;
    struct InstanceTransformCache { int instance_id; std::shared_ptr<Hittable> representative_hittable; };
    std::vector<InstanceTransformCache> m_instance_sync_cache;
    bool m_topology_dirty = false; // set when instances/BLAS list changes
    // Uploaded textures (map from opaque handle/key -> ImageHandle)
    std::unordered_map<int64_t, VulkanRT::ImageHandle> m_uploadedImages;
    std::unordered_map<int64_t, int64_t> m_uploadedImageIDs; // maps pointer/key -> texture ID
    int64_t m_nextTextureID = 1;
    // Cached lights when device not yet ready
    std::vector<std::shared_ptr<Light>> m_cachedLights;
    // Cached world data when device not yet ready
    WorldData m_cachedWorld;
    int64_t m_envTexID = 0;
    uint64_t m_lastCameraHash = 0;
    Vec3 m_prevViewDir;
    bool  m_hasPrevView = false;
    // When true, clear the UI framebuffer/texture on next renderProgressive call
    bool m_forceClearOnNextPresent = false;
};

} // namespace Backend

#endif // VULKAN_BACKEND_H
