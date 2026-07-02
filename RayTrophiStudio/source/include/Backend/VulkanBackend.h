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
#include <vector>
#include <cstdint>

#include "vulkan_material_types.h"
#include "vulkan_volume_types.h"
#include "Vec3.h"
#include "Matrix4x4.h"
#include "World.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>
#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "Backend/SceneTextureManager.h"
#include "AtmosphereLUT.h"

// Facade-mesh SoA database (DNA::GeometryDetail) — used by the indexed-BLAS upload path.
class TriangleMesh;

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
    bool supportsSamplerAnisotropy = false;
    float maxSamplerAnisotropy = 1.0f;
    bool supportsBC4 = false;
    bool supportsBC5 = false;
    bool supportsBC7 = false;

    // External-memory interop (Vulkan ↔ CUDA for OIDN GPU-direct denoise)
    bool supportsExternalMemoryWin32 = false;   // VK_KHR_external_memory_win32
    uint8_t deviceUUID[16] = {};                // VK_UUID_SIZE — for CUDA device matching
    bool hasDeviceUUID = false;
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

// ============================================================================
// Hair Rendering GPU Data Structures
// ============================================================================

// Per-B-spline-segment GPU layout — matches GLSL HairSegmentGPU (80 bytes, scalar)
struct HairSegmentGPU {
    float cp0[4];        // xyz = world-space position, w = radius
    float cp1[4];
    float cp2[4];
    float cp3[4];
    uint32_t strandID;
    uint32_t groomID;
    uint32_t materialID;
    uint32_t padding;
};
static_assert(sizeof(HairSegmentGPU) == 80, "HairSegmentGPU size mismatch");

// Per-groom hair material — matches GLSL HairGpuMaterial (144 bytes, scalar)
// Synced with OptiX GpuHairMaterial feature set (Marschner BSDF)
struct HairGpuMaterial {
    // Block 1: Color & Roughness (16 bytes)
    float baseColor[3];       // offset  0  — Direct color / fallback
    float roughness;           // offset 12  — Longitudinal roughness

    // Block 2: Physical Properties (16 bytes)
    float melanin;             // offset 16
    float melaninRedness;      // offset 20
    float ior;                 // offset 24  — Index of refraction
    float cuticleAngle;        // offset 28  — In radians

    // Block 3: Mode & Surface (16 bytes)
    uint32_t colorMode;        // offset 32  — 0=Direct, 1=Melanin, 2=Absorption, 3=RootUV
    float radialRoughness;     // offset 36  — Azimuthal roughness
    float specularTint;        // offset 40  — 0=white highlight, 1=tinted
    float diffuseSoftness;     // offset 44  — Multiple scattering weight

    // Block 4: Artistic Tint (16 bytes)
    float tintColor[3];        // offset 48
    float tint;                // offset 60  — Tint strength

    // Block 5: Coat / Fur (16 bytes)
    float coatTint[3];         // offset 64
    float coat;                // offset 76  — Coat strength

    // Block 6: Emission (16 bytes)
    float emission[3];         // offset 80
    float emissionStrength;    // offset 92

    // Block 7: Root-Tip Gradient (16 bytes)
    float tipColor[3];         // offset 96
    float rootTipBalance;      // offset 108

    // Block 8: Absorption & Gradient Flag (16 bytes)
    float absorption[3];       // offset 112 — Explicit sigma_a for mode 2
    uint32_t enableGradient;   // offset 124 — 0 or 1

    // Block 9: Random Variation & ID (16 bytes)
    float randomHue;           // offset 128
    float randomValue;         // offset 132
    uint32_t groomID;          // offset 136
    float pad;                 // offset 140
};
static_assert(sizeof(HairGpuMaterial) == 144, "HairGpuMaterial size mismatch");

// ============================================================================

struct BLASCreateInfo {
    // Triangle geometry
    const float* vertexData = nullptr;        // [x,y,z, x,y,z, ...]
    const float* normalData = nullptr;        // [nx,ny,nz, ...]
    const float* uvData = nullptr;            // [u,v, ...]
    uint32_t vertexCount = 0;
    uint32_t vertexStride = 12;               // Bytes between vertices (default: 3 floats)
    
    // Skinning / Bone animation
    bool hasSkinning = false;
    const int32_t* boneIndicesData = nullptr; // [b0,b1,b2,b3, ...] (16 bytes per vertex)
    const float* boneWeightsData = nullptr;   // [w0,w1,w2,w3, ...] (16 bytes per vertex)
    
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
    bool opaqueGeometry = false;            // True when every primitive can skip opacity any-hit

    // Phase 3d zero-copy device-resident geometry: when set, the geometry already lives
    // in a GPU buffer (e.g. written by the CC compute refine) laid out exactly as
    // createBLAS would combine host data — vert | norm | uv | idx | mat, contiguous,
    // with the same offsets derived from the *Count fields. The BLAS builds straight
    // from geometryDeviceAddress with no host upload; the buffer is NOT owned by the
    // BLAS (see AccelStructHandle::externalGeometry). vertexData/normalData/etc. are
    // ignored in this mode; the *Count fields must still be set for offset math.
    bool useDeviceGeometry = false;
    uint64_t geometryDeviceAddress = 0;
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
    uint32_t sbtRecordOffset = 0;       // SBT hit group offset (0=triangle, 1=volume procedural)
    int scatterGroupId = -1;            // Direct InstanceManager lookup for expanded scatter groups
    uint32_t scatterInstanceIndex = UINT32_MAX;
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

    // Phase 3d: geometry buffer is owned elsewhere (device-resident CC output); the
    // attribute BufferHandles only carry addresses, so cleanup must NOT destroy them.
    bool externalGeometry = false;

    // Skinning
    bool hasSkinning = false;
    bool allowUpdate = false;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    BufferHandle baseVertexBuffer;
    BufferHandle baseNormalBuffer;
    BufferHandle boneIndexBuffer;
    BufferHandle boneWeightBuffer;

    // Persistent skinning resources — reused every frame to avoid per-frame alloc/free
    BufferHandle persistentBoneMatsBuffer; // GPU bone matrix buffer, grown as needed
    uint64_t    persistentBoneMatsBufSize = 0; // current allocated size in bytes
    VkDescriptorSet skinningDescSet = VK_NULL_HANDLE; // cached desc set (updated in-place)
    BufferHandle skinScratchBuffer;  // Cached BLAS update scratch buffer (avoids alloc/free per frame)
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
    uint32_t mipLevels = 1;
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

    /**
     * @brief Create a buffer whose backing VkDeviceMemory is exportable to CUDA.
     *
     * Uses VK_KHR_external_memory_win32 (Opaque Win32 handle) on Windows.
     * The returned HANDLE is owned by the caller when useKmt=false; when passed
     * to cudaImportExternalMemory with cudaExternalMemoryHandleTypeOpaqueWin32,
     * CUDA takes ownership and the caller must NOT CloseHandle it.
     *
     * outAllocationSize returns the VkDeviceMemory size needed for CUDA import.
     * Returns a zero BufferHandle on failure (extension missing, alloc failure).
     */
    BufferHandle createExportableBuffer(const BufferCreateInfo& info,
                                        void** outWin32Handle,
                                        uint64_t* outAllocationSize);
    
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
     * @brief Begin batched BLAS build mode.
     * All subsequent createBLAS calls record into a single command buffer.
     * Call endBatchedBLASBuild() to submit all builds at once.
     */
    void beginBatchedBLASBuild();

    /**
     * @brief End batched mode and submit all pending BLAS builds in one GPU submission.
     */
    void endBatchedBLASBuild();

    /**
     * @brief Build AABB bottom-level acceleration structure (procedural volumes)
     * @param aabbMin World-space AABB minimum [x,y,z]
     * @param aabbMax World-space AABB maximum [x,y,z]
     * @return BLAS index, or UINT32_MAX on failure
     */
    uint32_t createAABB_BLAS(const float aabbMin[3], const float aabbMax[3]);

    /**
     * @brief Build multi-AABB BLAS for hair geometry (one AABB per segment).
     * @param aabbs Array of per-segment bounding boxes (world space)
     * @return BLAS index, or UINT32_MAX on failure
     */
    uint32_t createHairAABB_BLAS(const std::vector<VkAabbPositionsKHR>& aabbs);

    // Foam point-sphere AABB BLAS (combined, one per foam cloud). The AS + AABB
    // buffer are sized for `poolCapacity` (the group's full instance pool) but only
    // the supplied `aabbs` (LIVE spheres, compact) are built — so each frame rebuilds
    // just the live foam, matching the OptiX compact sphere GAS, while leaving room
    // for the live count to grow up to poolCapacity without a structural rebuild.
    uint32_t createFoamSphereBLAS(const std::vector<VkAabbPositionsKHR>& aabbs, uint32_t poolCapacity);
    bool     updateFoamSphereBLAS(uint32_t blasIndex, const std::vector<VkAabbPositionsKHR>& aabbs);
    // Upload the combined foam sphere SSBO (FoamSphereGPU[count], 32 bytes each)
    // to binding 18.
    void     updateFoamSphereBuffer(const void* data, uint32_t count);

    /**
     * @brief Upload hair segment and material GPU buffers.
     * Called by uploadHairStrands / uploadHairMaterials, also updates descriptors.
     */
    void updateHairSegmentBuffer(const std::vector<HairSegmentGPU>& segments);
    void updateHairMaterialBuffer(const std::vector<HairGpuMaterial>& materials);

    /// Returns the SBT hit-region offset for hair instances.
    /// 1 if no volume shaders loaded, 2 if volume shaders present.
    uint32_t getHairSbtOffset() const { return m_hasVolumeShaders ? 2u : 1u; }

    /// Returns the SBT hit-region offset for foam point-sphere instances.
    /// Hit region order: triangle(0), [volume], [hair], [sphere] — sphere is
    /// always last, so its offset is 1 + the optional groups before it.
    uint32_t getSphereSbtOffset() const {
        return 1u + (m_hasVolumeShaders ? 1u : 0u) + (m_hasHairShaders ? 1u : 0u);
    }
    
    /**
     * @brief Build top-level acceleration structure (instances)
     */
    void createTLAS(const TLASCreateInfo& info);
    
    /**
     * @brief Update existing BLAS (for animation)
     */
    void updateBLAS(uint32_t blasIndex, const float* newVertices, const float* newNormals = nullptr);
    
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
    bool createRTPipeline(const std::vector<std::uint32_t>& raygenSPV,
                          const std::vector<std::uint32_t>& missSPV,
                          const std::vector<std::uint32_t>& closestHitSPV,
                          const std::vector<std::uint32_t>& anyHitSPV               = std::vector<std::uint32_t>(),
                          const std::vector<std::uint32_t>& volumeClosestHitSPV     = std::vector<std::uint32_t>(),
                          const std::vector<std::uint32_t>& volumeIntersectionSPV   = std::vector<std::uint32_t>(),
                          const std::vector<std::uint32_t>& hairClosestHitSPV       = std::vector<std::uint32_t>(),
                          const std::vector<std::uint32_t>& hairIntersectionSPV     = std::vector<std::uint32_t>(),
                          const std::vector<std::uint32_t>& shadowMissSPV           = std::vector<std::uint32_t>(),
                          const std::vector<std::uint32_t>& hairAnyHitSPV           = std::vector<std::uint32_t>(),
                          const std::vector<std::uint32_t>& sphereClosestHitSPV     = std::vector<std::uint32_t>(),
                          const std::vector<std::uint32_t>& sphereIntersectionSPV   = std::vector<std::uint32_t>());
    
    /**
     * @brief Bind RT descriptors (output image + TLAS)
     */
    void bindRTDescriptors(const ImageHandle& outputImage,
                           const ImageHandle* denoiserColorImage = nullptr,
                           const ImageHandle* denoiserAlbedoImage = nullptr,
                           const ImageHandle* denoiserNormalImage = nullptr,
                           const ImageHandle* varianceImage = nullptr,
                           const ImageHandle* denoiserPositionImage = nullptr);
    void updateRTTextureDescriptor(uint32_t slot, const ImageHandle& image);
    void clearImage(const ImageHandle& image, float r, float g, float b, float a);

    /** @brief Clear multiple images in a single command buffer submission (batched). */
    struct ImageClearRequest { const ImageHandle* image; float r, g, b, a; };
    void clearImages(const std::vector<ImageClearRequest>& requests);
    
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
    // Trace rays + image→buffer copy in a single command buffer (1 GPU sync instead of 4)
    bool traceRaysAndReadback(uint32_t width, uint32_t height, const ImageHandle& outputImage, const BufferHandle& stagingBuffer);
    
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
                              VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);
    // Creates image with full mip chain; pre-transitions all levels to TRANSFER_DST_OPTIMAL.
    // Caller must fill mip 0 then call generateMipmaps to produce and finalize the chain.
    ImageHandle createImage2DWithMips(uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format,
                                      VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                                                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    // Records blit-based mip generation into cmd. Assumes mip 0 is in TRANSFER_DST_OPTIMAL.
    // All mip levels are transitioned to SHADER_READ_ONLY_OPTIMAL on completion.
    void generateMipmaps(VkCommandBuffer cmd, VkImage image, uint32_t width, uint32_t height, uint32_t mipLevels);
    void destroyImage(ImageHandle& image);

    void copyImageToBuffer(const ImageHandle& src, const BufferHandle& dst);
    // Batched variant — issues N image-to-buffer copies in a single command
    // buffer with one fence wait. Replaces the N×submit+wait pattern (3× for
    // the denoiser AOV path), removing per-copy submit/wait overhead which is
    // the dominant cost on the OIDN frame path.
    void copyImagesToBuffersBatched(const ImageHandle* srcs, const BufferHandle* dsts, size_t count);

    // Async batched variant for the denoiser path. Records into a persistent
    // command buffer + signals a persistent fence. Does NOT wait — the caller
    // calls waitDenoiserCopy() the *next* frame, by which time the fence is
    // already signaled at steady state. This is the fence-deferred copy that
    // replaces the per-frame submit+wait stall (~18 ms at 720p aux=1) with a
    // near-zero wait, while CUDA OIDN work overlaps with the next frame's RT.
    bool submitDenoiserCopyAsync(const ImageHandle* srcs, const BufferHandle* dsts, size_t count);
    // Wait for the previously-submitted async denoiser copy. Returns false if
    // there has been no submit yet (first call) or if the wait failed.
    bool waitDenoiserCopy(uint64_t timeoutNs = UINT64_MAX);
    bool hasDenoiserCopyEverSubmitted() const { return m_denoiserCopySlot.everSubmitted; }
    // Force a drain — used by resize/destroy paths before tearing down staging
    // buffers or the cmd pool. Safe to call when nothing is in flight.
    void drainDenoiserCopy();

    void copyBufferToImage(const BufferHandle& src, const ImageHandle& dst);
    // Copies a sub-rectangle from `src` into the existing `dst` VkImage. Used
    // by the paint pipeline so that a brush-sized dab does not need a full
    // image stage+copy. Image must already be in VK_IMAGE_LAYOUT_GENERAL.
    void copyBufferToImageRegion(const BufferHandle& src, const ImageHandle& dst,
                                 int32_t offsetX, int32_t offsetY,
                                 uint32_t regionW, uint32_t regionH);
    void recordCopyBufferToImage(VkCommandBuffer cmd, const BufferHandle& src, const ImageHandle& dst);
    // Like recordCopyBufferToImage but assumes the image is in TRANSFER_DST_OPTIMAL (used for mip uploads).
    void recordCopyBufferToImageDst(VkCommandBuffer cmd, const BufferHandle& src, const ImageHandle& dst);

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
    bool supportsGraphicsQueue() const { return m_queueSupportsGraphics; }
    // SSBO Update helpers
    void updateMaterialBuffer(const void* data, uint64_t size, uint32_t count);
    void updateLightBuffer(const void* data, uint64_t size, uint32_t count);
    void updateWorldBuffer(const void* data, uint64_t size, uint32_t count);
    void updateVolumeBuffer(const void* data, uint64_t size, uint32_t count);
    void updateTerrainLayerBuffer(const void* data, uint64_t size, uint32_t count);
    void updateAtmosphereLUTs(const ImageHandle* lutImages);  // uint256_t array of 4 LUT textures
    bool generateAtmosphereLUTGPU(const WorldData& world);
    void clearPendingRTTextureDescriptors();
    void removePendingRTTextureDescriptor(const ImageHandle& image);
    
    // RT Resources (SSBOs)
    VulkanRT::BufferHandle m_materialBuffer;
    VulkanRT::BufferHandle m_lightBuffer;
    VulkanRT::BufferHandle m_geometryDataBuffer; // SSBO containing VkGeometryData for each BLAS
    VulkanRT::BufferHandle m_instanceDataBuffer; // SSBO containing VkInstanceData for each TLAS instance
    VulkanRT::BufferHandle m_tlasInstanceBuffer; // Buffer containing VkAccelerationStructureInstanceKHR for TLAS building
    VulkanRT::BufferHandle m_tlasScratchBuffer;  // Persistent TLAS build/update scratch (reused across frames; grows on demand)
    VulkanRT::BufferHandle m_worldBuffer; // SSBO containing complete Nishita parameters
    VulkanRT::BufferHandle m_volumeBuffer; // SSBO containing VkVolumeInstance array (binding 9)
    VulkanRT::BufferHandle m_terrainLayerBuffer; // SSBO containing VkTerrainLayerData array (binding 12)
    uint32_t m_volumeCount = 0;
    uint32_t m_terrainLayerCount = 0;
    
    // Atmosphere LUT Textures (for raygen/miss shaders)
    // [0] = transmittance_lut, [1] = skyview_lut, [2] = multi_scatter_lut, [3] = aerial_perspective_lut
    ImageHandle m_lutImages[4];
    
    uint32_t m_materialCount = 0;
    uint32_t m_lightCount = 0;
    std::vector<AccelStructHandle> m_blasList;
    // Pending texture descriptor updates (slot -> ImageHandle) queued until RT descriptor set exists
    std::vector<std::pair<uint32_t, ImageHandle>> m_pendingTextureDescriptors;
    // Guards m_pendingTextureDescriptors and m_rtDescriptorSet access during async uploads/switch.
    mutable std::mutex m_rtDescriptorMutex;
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
    // Acceleration structures
    AccelStructHandle m_tlas;
    uint32_t m_tlasInstanceCount = 0;
    bool m_tlasSupportsUpdate = false;
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    bool m_rtPipelineReady = false;
    bool m_hasVolumeShaders = false; // Whether pipeline includes volume procedural hit group
    bool m_hasHairShaders = false; // Whether pipeline includes hair procedural hit group
    bool m_hasSphereShaders = false; // Whether pipeline includes foam point-sphere procedural hit group
    bool m_hasShadowMiss = false; // Whether a shadow miss shader is at miss index 1
    VkPipeline m_rtPipeline = VK_NULL_HANDLE;
    
    // Skinning Compute Pipeline
    bool createSkinningPipeline(const std::vector<uint32_t>& computeSPV);
    bool hasSkinningPipeline() const;
    void dispatchSkinning(uint32_t blasIndex, const std::vector<Matrix4x4>& boneMatrices); // legacy shim
    void dispatchSkinningAll(const std::vector<Matrix4x4>& boneMatrices); // batch: 1 submit for all BLASes
    bool dispatchSkinningToBuffers(BufferHandle& baseVertexBuffer,
                                   BufferHandle& baseNormalBuffer,
                                   BufferHandle& boneIndexBuffer,
                                   BufferHandle& boneWeightBuffer,
                                   BufferHandle& persistentBoneMatsBuffer,
                                   uint64_t& persistentBoneMatsBufSize,
                                   VkDescriptorSet& skinningDescSet,
                                   const BufferHandle& outVertexBuffer,
                                   const BufferHandle& outNormalBuffer,
                                   uint32_t vertexCount,
                                   const std::vector<Matrix4x4>& boneMatrices);
    
    VkPipeline m_skinningPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_skinningPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_skinningDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_skinningDescPool = VK_NULL_HANDLE;

    // Sculpt Compute Pipeline (simple GPU sculpt pass)
    bool createSculptPipeline(const std::vector<uint32_t>& computeSPV);
    void dispatchSculpt(const BufferHandle& positions, const BufferHandle& normals, const BufferHandle& weights,
                        uint32_t vertexCount, const void* pushConstants = nullptr, uint32_t pushSize = 0);

    VkPipeline m_sculptPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_sculptPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_sculptDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_sculptDescPool = VK_NULL_HANDLE;

    // Tonemap Compute Pipeline (GPU HDR -> LDR sRGB pack; removes per-frame CPU tonemap loop)
    bool createTonemapPipeline(const std::vector<uint32_t>& computeSPV);
    // Legacy synchronous path — kept as fallback. Submits + waits inside the call.
    bool traceRaysTonemapAndReadback(uint32_t w, uint32_t h,
                                     const ImageHandle& hdrImage,
                                     const ImageHandle& ldrImage,
                                     const BufferHandle& ldrStaging);
    // Async path — records into persistent cmd buffer for the given slot and submits
    // signaling that slot's fence. Does NOT wait. Adapter must call waitFrameSlot()
    // before reading the staging buffer or reusing the same slot.
    bool submitTraceTonemapAsync(uint32_t slot, uint32_t w, uint32_t h,
                                 const ImageHandle& hdrImage,
                                 const ImageHandle& ldrImage,
                                 const BufferHandle& ldrStaging);
    bool waitFrameSlot(uint32_t slot, uint64_t timeoutNs = UINT64_MAX);
    // Updates the persistent tonemap descriptor set with new image views. Caller MUST
    // ensure no in-flight submission references the previous binding (drain fences first).
    bool updateTonemapDescriptors(const ImageHandle& hdrImage, const ImageHandle& ldrImage);
    bool hasTonemapPipeline() const { return m_tonemapPipeline != VK_NULL_HANDLE; }
    static constexpr uint32_t kFrameSlotCount = 2;

    VkPipeline m_tonemapPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_tonemapPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_tonemapDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_tonemapDescPool = VK_NULL_HANDLE;
    VkDescriptorSet m_tonemapDescSet = VK_NULL_HANDLE; // persistent, single set, rewritten on image change

    // Stylize compute pipeline (Vulkan-native GPU stylize, no CUDA). bindings:
    // 0=color SSBO (in/out), 1/2/3=position/albedo/normal AOV storage images,
    // 4=params SSBO. Descriptors rewritten per frame by the adapter.
    bool createStylizePipeline(const std::vector<uint32_t>& computeSPV);
    bool hasStylizePipeline() const { return m_stylizePipeline != VK_NULL_HANDLE; }
    bool updateStylizeDescriptors(const BufferHandle& colorBuf, const BufferHandle& paramsBuf,
                                  VkImageView posView, VkImageView albView, VkImageView nrmView);
    bool dispatchStylizeCompute(uint32_t w, uint32_t h, VkImage posImg, VkImage albImg, VkImage nrmImg);
    VkPipeline m_stylizePipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_stylizePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_stylizeDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_stylizeDescPool = VK_NULL_HANDLE;
    VkDescriptorSet m_stylizeDescSet = VK_NULL_HANDLE;

    // Atmosphere LUT Compute Pipeline (Nishita LUT generation on Vulkan GPU)
    bool createAtmosphereLUTPipeline(const std::vector<uint32_t>& computeSPV);
    bool hasAtmosphereLUTPipeline() const { return m_atmosphereLutPipeline != VK_NULL_HANDLE; }
    bool updateAtmosphereLUTComputeDescriptors(const ImageHandle* lutImages);

    VkPipeline m_atmosphereLutPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_atmosphereLutPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_atmosphereLutDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_atmosphereLutDescPool = VK_NULL_HANDLE;
    VkDescriptorSet m_atmosphereLutDescSet = VK_NULL_HANDLE;
    BufferHandle m_atmosphereLutParamsBuffer;

    // Per-slot persistent command buffer + fence for async submit. Lazy-init on first
    // submitTraceTonemapAsync. Cmd buffers are allocated from m_commandPool with
    // RESET_COMMAND_BUFFER semantics (vkBeginCommandBuffer implicitly resets a primary
    // cmd buf when the pool is not RESET_COMMAND_BUFFER_BIT capable, but here we
    // call vkResetCommandBuffer explicitly which requires the pool flag — see
    // createFrameSlots()).
    struct FrameSlot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool everSubmitted = false;
    };
    FrameSlot m_frameSlots[kFrameSlotCount];
    VkCommandPool m_frameSlotCommandPool = VK_NULL_HANDLE; // RESET_COMMAND_BUFFER_BIT capable
    bool ensureFrameSlotsCreated();
    void destroyFrameSlots();

    // Single-slot persistent cmd buffer + fence for the fence-deferred denoiser
    // copy. One slot is sufficient: CUDA fully consumes staging before we submit
    // the next copy (see submitDenoiserCopyAsync notes). Uses the same RESET_-
    // COMMAND_BUFFER pool as the RT path.
    struct DenoiserCopySlot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool everSubmitted = false;
    };
    DenoiserCopySlot m_denoiserCopySlot;
    VkCommandPool    m_denoiserCopyCommandPool = VK_NULL_HANDLE; // RESET_COMMAND_BUFFER_BIT
    bool ensureDenoiserCopySlotCreated();
    void destroyDenoiserCopySlot();

private:
    // Core Vulkan handles
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
   
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    uint32_t m_computeQueueFamily = 0;
    bool m_queueSupportsGraphics = false;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    
    // Debug
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    

    
    
    // Compute Pipelines
    std::vector<VkPipeline> m_pipelines;
    std::vector<VkPipelineLayout> m_pipelineLayouts;
    uint32_t m_activePipeline = UINT32_MAX;
    
    // RT Pipeline (separate from compute)
   
    VkPipelineLayout m_rtPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_rtDescriptorSetLayout = VK_NULL_HANDLE;
   
   

    // Hair GPU buffers (bindings 10 and 11)
    BufferHandle m_hairSegmentBuffer;    // HairSegmentGPU[] — binding 10
    BufferHandle m_hairMaterialBuffer;   // HairGpuMaterial[] — binding 11

    // Foam point-sphere buffer (binding 18). One combined buffer for the whole
    // foam cloud: { vec4 centreRadius, uint matId, uint pad[3] } per sphere,
    // indexed by gl_PrimitiveID in sphere_intersection.rint / sphere_closesthit.
    // Kept persistently bound (dummy 1-element until foam exists) so binding 18
    // is always valid; the foam consumer re-uploads it each frame.
    BufferHandle m_foamSphereBuffer;     // FoamSphereGPU[] — binding 18
    uint32_t m_foamSphereCount = 0;
    
    // Shader Binding Table
    BufferHandle m_sbtBuffer;
    VkStridedDeviceAddressRegionKHR m_sbtRaygenRegion{};
    VkStridedDeviceAddressRegionKHR m_sbtMissRegion{};
    VkStridedDeviceAddressRegionKHR m_sbtHitRegion{};
    VkStridedDeviceAddressRegionKHR m_sbtCallableRegion{};
    

    
    // Active command buffer (for recording)
    VkCommandBuffer m_activeCommandBuffer = VK_NULL_HANDLE;

    // Batched BLAS build support — reduces N GPU submits to ~1 for large scene loads
    VkCommandBuffer m_batchBLASCmd = VK_NULL_HANDLE;
    bool m_inBatchedBLASBuild = false;
    BufferHandle m_batchScratchBuffer;       // Single reusable scratch buffer for batch
    uint32_t m_batchBLASCount = 0;           // Total BLASes in current batch (for logging)
    uint32_t m_batchBLASInCurrentCmd = 0;    // BLASes in current cmd buffer fragment (barrier logic)
    
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

struct TerrainObject;
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

// ============================================================================
// IBackend Adapter
// ============================================================================
#define VK_INFO() VulkanLogHelper(LogLevel::Info)
#define VK_ERROR() VulkanLogHelper(LogLevel::Error)
#define VK_WARN() VulkanLogHelper(LogLevel::Warning)

namespace Backend {

// Maximum number of texture descriptor slots available to shaders.
// Both the RT (binding 6) and raster-preview (binding 1) descriptor arrays
// are sized to this value. Shaders declare runtime arrays ([]) so no shader
// recompilation is needed when this constant changes.
// Purge is triggered at (VULKAN_TEXTURE_CAPACITY - 64) to keep headroom.
static constexpr int32_t VULKAN_TEXTURE_CAPACITY = 2048;
static constexpr int32_t VULKAN_TEXTURE_PURGE_THRESHOLD = VULKAN_TEXTURE_CAPACITY - 64; // 1984

// Draw parameters for the raster-viewport edit-mesh overlay (GPU wireframe /
// vertex-marker / face-fill passes). Geometry lives in dedicated buffers set
// via uploadEditMeshOverlay*; this struct only carries per-frame draw state.
// Positions are uploaded object-local; `model` maps them to world so object
// transform changes never force a vertex re-upload.
struct EditMeshOverlayParams {
    bool enabled = false;
    bool drawEdges = false;
    bool drawPoints = false;
    bool drawFaces = false;        // dim fill of every face (Face select mode)
    bool softHighlight = false;    // tint unselected elements by soft-selection weight (flag bits 8..15)
    bool xray = false;             // draw through the mesh (depth test effectively disabled in-shader)
    Matrix4x4 model;               // object -> world for the uploaded local-space positions
    float pointRadiusPx = 4.0f;
    // Relative (distance-proportional) depth pull toward the camera so the
    // overlay sits on its own surface. Must stay tiny: anything near the
    // mesh's own NDC depth range acts as accidental x-ray.
    float depthBias = 0.00002f;
    float edgeColor[4]       = { 0.61f, 0.65f, 0.69f, 0.89f };
    float pointColor[4]      = { 0.62f, 0.65f, 0.68f, 0.95f };
    float faceColor[4]       = { 0.47f, 0.52f, 0.56f, 0.13f };
    float selectColor[4]     = { 0.28f, 0.73f, 0.84f, 1.0f };
    float selectFaceColor[4] = { 0.28f, 0.73f, 0.84f, 0.48f };
};

// Selection-outline overlay (raster viewport GPU silhouette). The backend
// resolves node names against its own raster instances each frame, re-draws
// just those instances into an R8G8 mask (G = full silhouette, R =
// depth-tested visible) and edge-detects the mask over the finished frame —
// no geometry ever crosses the CPU/GPU boundary for this. The LAST name in
// nodeNames is the primary selection and gets primaryColor; the rest get
// secondaryColor; portions hidden behind other geometry use occludedColor.
struct SelectionOutlineParams {
    bool enabled = false;
    std::vector<std::string> nodeNames;
    // Muted green: the original neon (0,255,128) at 2px read as a thick,
    // attention-grabbing band — keep the outline informative but quiet.
    float primaryColor[4]   = { 0.28f, 0.75f, 0.49f, 0.85f };
    float secondaryColor[4] = { 0.20f, 0.55f, 0.37f, 0.62f };
    float occludedColor[4]  = { 0.63f, 0.63f, 0.63f, 0.47f };
    float thicknessPx = 1.0f;
};

/**
 * @brief Vulkan implementation of IBackend
 *
 * Bridges the IBackend interface to VulkanRT::VulkanDevice.
 * Similar to how OptixBackend wraps OptixWrapper.
 */
class VulkanBackendAdapter : public IViewportBackend {
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
    GpuMemoryStats getMemoryStats() const override;

    // ========================================================================
    // IBackend - Geometry Upload
    // ========================================================================
    uint32_t uploadTriangles(const std::vector<TriangleData>& triangles, const std::string& meshName) override;

    // INDEXED upload for a single facade mesh: reads the shared SoA (DNA::GeometryDetail)
    // directly and builds an INDEXED BLAS (unique verts + index buffer) instead of the 3N
    // expanded soup uploadTriangles() produces — a large VRAM win for dense meshes (e.g.
    // Catmull-Clark subdivide). `useOriginalSpace` picks bind/local P_orig (static imported
    // meshes whose world transform lives in the TLAS instance) vs live P (transformless /
    // procedural). Skinned / water / live-deformed meshes stay on uploadTriangles(). The
    // shader already supports indexAddr, so no shader change is needed. Returns the BLAS
    // index (cached by meshName) or UINT32_MAX on failure.
    uint32_t uploadTriangleMeshIndexed(const TriangleMesh* mesh,
                                       const std::string& meshName,
                                       bool useOriginalSpace);

    // Phase 3d: register a BLAS for a DEVICE-RESIDENT CC mesh whose geometry already
    // lives in a GPU buffer (combined non-indexed layout written by the CC compute
    // expand; same VkDevice). No host triangles, no upload — createBLAS builds straight
    // from geometryDeviceAddress. materialIds is the per-triangle host copy (from the CC
    // plan's triMat) used for shader material lookup; the buffer's mat block carries the
    // device copy. Returns the BLAS index (cached by meshName) or UINT32_MAX on failure.
    uint32_t uploadDeviceResidentMesh(const std::string& meshName,
                                      uint64_t geometryDeviceAddress,
                                      uint32_t vertexCount,
                                      uint32_t triCount,
                                      const std::vector<uint32_t>& materialIds,
                                      bool opaqueGeometry);

    // Phase 3d: a static / render-only device-resident CC mesh. Its geometry already lives
    // in a GPU buffer (combined non-indexed layout written by the CC compute expand); there
    // are NO host triangles. Backend-owned; the heavy rebuild registers a BLAS straight from
    // deviceAddress and appends a TLAS instance. Picking / Solid raster fall back to the cage.
    struct DeviceResidentCCMesh {
        std::string meshKey;
        uint64_t bufferId = 0;             // shared mesh-compute buffer handle id (for release)
        uint64_t deviceAddress = 0;
        uint32_t vertexCount = 0;          // triCount * 3
        uint32_t triCount = 0;
        Matrix4x4 transform;               // world transform of the instance
        uint32_t materialID = 0;           // representative material id for the instance
        std::vector<uint32_t> materialIds; // per-triangle material ids (shader lookup)
        bool opaque = false;
        std::string cageNodeName;          // editable cage (picking / raster fallback)
    };

    // Register or replace a device-resident CC mesh (matched by meshKey). The caller must
    // trigger a geometry rebuild afterwards. Replacing queues the old GPU buffer for release.
    void registerDeviceResidentCCMesh(const DeviceResidentCCMesh& mesh);
    // Remove a device-resident CC mesh by key; queues its GPU buffer for deferred release.
    void removeDeviceResidentCCMesh(const std::string& meshKey);
    // True if a live device-resident CC mesh is currently registered for this cage node.
    // The host-display machine uses this to skip the per-frame full CPU BVH rebuild on cage
    // edits (the dense lives on the GPU + is RT-excluded; CPU BVH is deferred to stroke end).
    bool hasDeviceResidentCCNode(const std::string& cageNodeName) const;

    // Slice 4 fast path: queue a position-only refit (UI thread). The entry must already be
    // registered with the same vertexCount; the caller also registerDeviceResidentCCMesh()'s
    // the new buffer so a fallback full rebuild stays correct. Coalesces per meshKey (the
    // superseded buffer is queued for release).
    void queueDeviceResidentCCRefit(const std::string& meshKey, uint64_t newGeomBase,
                                    uint64_t newBufferId, uint32_t vertexCount);
    // Drain queued refits (render thread): MODE_UPDATE-refit each BLAS to its new device
    // buffer + refresh the shader geometry-address binding + refit the TLAS. Returns false if
    // any request could not be refit (missing / non-updatable / topology changed) so the
    // caller can fall back to a full rebuild. No-op (returns true) when the queue is empty.
    bool drainDeviceResidentCCRefits();

    uint32_t uploadHairStrands(const std::vector<HairStrandData>& strands, const std::string& groomName) override;
    void clearHairGeometry(bool rebuild_tlas = true);
    // Upload flat LINE_LIST vertex data {x,y,z,v_coord} x vertexCount for the raster viewport hair overlay
    void uploadHairViewportLines(const std::vector<float>& vertexData, uint32_t vertexCount);
    // Upload camera-facing particle billboards for the raster viewport overlay.
    // Vertex layout: {pos.xyz, uv.xy, rgba} = 9 floats; TRIANGLE_LIST (6 verts/quad).
    // Two groups by blend mode: additive and alpha.
    void uploadParticleBillboards(const std::vector<float>& addData, uint32_t addVertexCount,
                                  const std::vector<float>& alphaData, uint32_t alphaVertexCount);
    // Edit-mesh overlay (raster viewport GPU wireframe/vertex/face passes).
    // Buffers are split by update frequency so each piece re-uploads only
    // when its source actually changes:
    //  - geometry: object-local {x,y,z} per editable vertex (vertex edits / sculpt)
    //  - flags:    one uint per vertex — bit0 selected, bits 8..15 soft-selection weight (selection changes)
    //  - topology: uint32 index pairs for every unique edge + triangulated face fan (topology rebuilds)
    //  - selection indices: index buffers for the currently selected edges/faces
    void uploadEditMeshOverlayGeometry(const std::vector<float>& positions, uint32_t vertexCount);
    void uploadEditMeshOverlayFlags(const std::vector<uint32_t>& flags);
    void uploadEditMeshOverlayTopology(const std::vector<uint32_t>& edgeIndices,
                                       const std::vector<uint32_t>& faceIndices);
    void uploadEditMeshOverlaySelectionIndices(const std::vector<uint32_t>& selEdgeIndices,
                                               const std::vector<uint32_t>& selFaceIndices);
    void setEditMeshOverlayParams(const EditMeshOverlayParams& params);
    void clearEditMeshOverlay();
    // Selection outline (raster viewport GPU silhouette pass). Cheap to call
    // every frame: only re-renders when something actually changed.
    void setSelectionOutlineParams(const SelectionOutlineParams& params);
    void clearSelectionOutline();
    // True once the outline pipelines + targets are live. The UI gates on
    // this so a missing SPIR-V / unsupported format degrades to the ImGui
    // CPU outline instead of silently losing the outline.
    bool hasGpuSelectionOutline() const;
    // Rendered-mode support: renders the selection mask standalone (scene
    // depth prepass + selected-instance mask) and reads the R8G8 mask back
    // (interleaved R,G per pixel; R = depth-tested visible, G = full
    // silhouette). The caller draws the boundary as an ImGui overlay on top
    // of the path-traced image, so this works over any render backend.
    // The camera comes from the CALLER: in Rendered mode this backend's own
    // m_camera is not updated (the render backend owns the camera), so using
    // it would freeze the outline at a stale view. Always renders a
    // PERSPECTIVE projection — the path tracer ignores orthographic cameras.
    // width/height may be smaller than the screen (the caller scales the
    // boundary runs back up). Returns false when raster geometry or the
    // outline pipelines are unavailable; caller then uses the CPU outline.
    // camAspect: the projection aspect of the DISPLAYED image (render
    // width/height, same convention as the gizmo overlay projection) — not
    // the readback target's aspect.
    // fullWidth/fullHeight size the GPU targets (keep them constant — e.g.
    // the screen size — so scale changes don't recreate resources);
    // maskWidth/maskHeight select the sub-region actually rendered and read
    // back (the caller scales the boundary runs back up).
    bool renderSelectionOutlineMaskReadback(const std::vector<std::string>& nodeNames,
                                            const Vec3& camEye, const Vec3& camLookAt,
                                            const Vec3& camUp, float camFovDeg,
                                            float camAspect,
                                            int fullWidth, int fullHeight,
                                            int maskWidth, int maskHeight,
                                            std::vector<uint8_t>& outMaskRG);
    // Targeted raster-instance transform refresh for the Rendered-mode
    // outline: updates ONLY the given nodes' raster instances (composed
    // world matrices, getTransformMatrix convention). Unlike
    // updateObjectTransform it never writes back to the scene objects —
    // that one calls setBaseTransform and would corrupt pivot-composed
    // transforms when fed a composed matrix.
    void setRasterInstanceTransformsForNodes(
        const std::vector<std::pair<std::string, Matrix4x4>>& nodes);
    void updateMeshTransform(uint32_t meshHandle, const Matrix4x4& transform) override;
    void rebuildAccelerationStructure() override;
    void showAllInstances() override;
    void updateSceneGeometry(const std::vector<std::shared_ptr<Hittable>>& objects, const std::vector<Matrix4x4>& boneMatrices) override;
    void updateInstanceMaterialBinding(const std::string& nodeName, int oldMatID, int newMatID) override;
    void setVisibilityByNodeName(const std::string& nodeName, bool visible) override;
    void updateGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) override;
    bool updateTerrainBLASPartial(const std::string& nodeName, const TerrainObject* terrain);
    bool updateMeshBLASPartial(const std::string& nodeName, const std::vector<std::shared_ptr<Triangle>>& triangles);
    int64_t updateMeshBLASPartial(const std::string& nodeName,
                                  const std::vector<size_t>& dirtyIndices,
                                  const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries);
    // Append-only fast path for scatter/instance-add mutations: reuses existing BLASes and
    // only refits the TLAS + appends new HittableInstance TLAS records. Falls back to false
    // when topology shape changes in a way the incremental path cannot safely diff (removal,
    // solo-triangle / VDB structure churn, hair-only state, etc.) — caller must then run the
    // full updateGeometry path.
    bool tryAppendGeometryIncremental(const std::vector<std::shared_ptr<Hittable>>& objects);

    // ========================================================================
    // IBackend - Materials & Textures
    // ========================================================================
    void uploadMaterials(const std::vector<MaterialData>& materials) override;
    bool updateMaterial(uint32_t materialIndex, const MaterialData& material) override;
    void uploadHairMaterials(const std::vector<HairMaterialData>& materials) override;
    void uploadTerrainLayerMaterials(const std::vector<TerrainLayerData>& layers) override;
    int64_t uploadTexture2D(const void* data, uint32_t width, uint32_t height, uint32_t channels, bool sRGB, bool isFloat = false) override;
    int64_t uploadTexture3D(const void* data, uint32_t width, uint32_t height, uint32_t depth, uint32_t channels, bool isFloat = false) override;
    void destroyTexture(int64_t textureHandle) override;
    void releaseInactiveViewportTextureCache();
    virtual const char* sceneTextureOwnerScope() const { return "VulkanBackendAdapter"; }
    bool tryGetUploadedImageHandle(int64_t textureHandle, VulkanRT::ImageHandle& outImage) const;
    uint64_t textureCacheGeneration() const { return m_textureCacheGeneration; }
    bool buildVulkanBackingRecord(int64_t textureHandle, VulkanBackingRecord& outBacking) const;
    void registerSceneTextureUpload(TextureHandle sceneHandle, int64_t textureHandle);

    // In-place re-upload: write `data` into the existing VkImage at `textureID`,
    // provided width/height/format match. Used by the paint pipeline so that
    // vulkan_dirty textures don't destroy/create their VkImage every dab — which
    // both thrashes the allocator and forces the RT descriptor path even when
    // the active backend is raster-only. Returns false when the slot is missing
    // or dims/format diverge; caller should fall back to destroy + uploadTexture2D.
    bool updateTexture2DInPlace(int64_t textureID, const void* data,
                                uint32_t width, uint32_t height,
                                uint32_t channels, bool sRGB, bool isFloat = false);

    // Partial-rectangle re-upload. `data` is a tightly-packed regionW × regionH
    // buffer (no row padding) and is copied into the existing VkImage at the
    // specified offset. Returns false on slot/format mismatch — caller falls
    // back to updateTexture2DInPlace or full re-upload. Used by the paint
    // sync to avoid full-texture transfers when only a brush-sized dab
    // changed in CPU pixels.
    bool updateTexture2DRegion(int64_t textureID, const void* data,
                               uint32_t fullWidth, uint32_t fullHeight,
                               uint32_t channels, bool sRGB,
                               int32_t offsetX, int32_t offsetY,
                               uint32_t regionW, uint32_t regionH);

private:
    // Records mip 1..N regeneration into `cmd` for paint-style partial
    // updates: mip 0 is assumed to be in VK_IMAGE_LAYOUT_GENERAL (the layout
    // updateTexture2DInPlace/Region use for the buffer→image copy) and mip
    // 1..N in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (the steady state
    // outside paint). On exit, every mip is left in SHADER_READ_ONLY_OPTIMAL
    // so the very next sample sees the freshly-blitted chain. Only intended
    // for color-aspect 2D images created with TRANSFER_SRC + TRANSFER_DST +
    // SAMPLED usage flags (uploadTexture2D's mip-chain path).
    void regenerateMipChainAfterPartialUpdate(VkCommandBuffer cmd, const VulkanRT::ImageHandle& img);

public:

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
    void setViewportMode(ViewportMode mode) override;
    ViewportMode getViewportMode() const override;
    bool supportsViewportMode(ViewportMode mode) const override;
    bool updateInteractiveMesh(const std::string& nodeName,
                               const std::vector<std::shared_ptr<Triangle>>& triangles) override;
    // Refit a flat (direct SoA) mesh's RT BLAS straight from its DNA SoA, found by node name — no
    // per-face Triangle facades (a flat mesh has none, so updateInteractiveMesh's facade-list path
    // can't reach it). Used by the physics per-mesh deform refit for rigid-on-flat. Returns false
    // (mesh/BLAS not found, or not an indexed solo BLAS) so the caller can fall back to a rebuild.
    bool refitFlatMeshBLAS(const std::string& nodeName);
    bool updateRasterMeshFromTriangles(const std::string& nodeName,
                                       const std::vector<std::shared_ptr<Triangle>>& triangles);
    // Refit a flat (direct SoA) mesh's raster vertices straight from its DNA SoA — no per-face
    // Triangle facades (a flat mesh has none). Uses the same efficient dirty-range upload as
    // updateRasterMeshFromTriangles so a sculpt dab stays realtime in Solid without a full
    // buildRasterGeometry (which early-outs on an unchanged scene generation anyway).
   // bool updateRasterMeshFromMeshSoA(const std::string& nodeName, const TriangleMesh* mesh) override;
    bool patchRasterMeshTriangles(const std::string& nodeName,
                                  const std::vector<size_t>& dirtyIndices,
                                  const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries);
    bool cloneRasterObjectByNodeName(const std::string& sourceNodeName,
                                     const std::string& newNodeName,
                                     const Matrix4x4& transform) override;
    bool cloneRtObjectByNodeName(const std::string& sourceNodeName,
                                 const std::string& newNodeName,
                                 const std::shared_ptr<Hittable>& representativeSource,
                                 const Matrix4x4& transform);
    void buildRasterGeometry(const std::vector<std::shared_ptr<Hittable>>& objects);
    void syncRasterInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects);
    void syncRasterSkinnedVertices(const std::vector<std::shared_ptr<Hittable>>& objects,
                                   const std::vector<Matrix4x4>& boneMatrices);
    bool hasValidRasterCache(uint64_t sceneGeometryGeneration) const override {
        return !m_rasterMeshes.empty() && m_rasterBuiltGeometryGeneration == sceneGeometryGeneration;
    }
    void renderProgressive(void* outSurface, void* outWindow, void* outRenderer,
                           int width, int height, void* outFramebuffer, void* outTexture) override;
    void downloadImage(void* outPixels) override;
    bool getDenoiserFrame(DenoiserFrameData& frame, bool useAuxiliary = true, bool includeColor = true) override;
    bool getDenoiserFrameGPU(DenoiserFrameDataGPU& frame, bool useAuxiliary = true) override;
    // GPU-native stylize (no CUDA): upload the graded surface to an SSBO, run the
    // stylize compute over the resident AOV images, download. Returns false → CPU fallback.
    bool applyStylizeGPU(void* surface,
                         const StylizeGPU::KernelParams& params,
                         const StylizeCore::StyleProfileCore& profile) override;
    int getCurrentSampleCount() const override;
    bool isAccumulationComplete() const override;
    bool needsViewportRender() const override {
        if (shouldUseInteractiveViewport()) {
            return m_interactiveViewport.dirty;
        }
        return m_currentSamples < m_targetSamples;
    }

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
    bool generateAtmosphereLUTGPU(const WorldData* worldData);
    void setInteractiveViewportMatcap(int64_t textureID);
    void setInteractiveViewportMatcapPreset(int preset);

    // Full teardown for New/Open project workflows. Destroys the interactive
    // viewport descriptor set (so its binding-1 texture array no longer holds
    // soon-to-be-dead VkImageViews) and then runs rebuildAccelerationStructure
    // to purge m_uploadedImages / BLAS / TLAS / raster meshes. Safe to call on
    // any VulkanBackendAdapter; the next frame will recreate interactive
    // viewport resources and backfill textures from the fresh project.
    void resetForProjectReload();
protected:
    bool updateRasterMeshFromTrianglesImpl(const std::string& nodeName,
                                           const std::vector<std::shared_ptr<Triangle>>& triangles);
    bool patchRasterMeshTrianglesImpl(const std::string& nodeName,
                                      const std::vector<size_t>& dirtyIndices,
                                      const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries);
    void buildRasterGeometryImpl(const std::vector<std::shared_ptr<Hittable>>& objects);
    void syncRasterInstanceTransformsImpl(const std::vector<std::shared_ptr<Hittable>>& objects);
    void syncRasterSkinnedVerticesImpl(const std::vector<std::shared_ptr<Hittable>>& objects,
                                       const std::vector<Matrix4x4>& boneMatrices);
    virtual bool shouldUseInteractiveViewportImpl() const;
    virtual bool ensureInteractiveViewportResourcesImpl(const std::string& shaderDir, int width, int height);
    virtual void destroyInteractiveViewportResourcesImpl(bool keepPipeline = false);
    virtual void renderInteractiveViewportImpl(void* outSurface, int width, int height, void* outFramebuffer, void* outTexture);
    void renderProgressiveImpl(void* outSurface, void* outWindow, void* outRenderer,
                               int width, int height, void* outFramebuffer, void* outTexture);
    virtual void setInteractiveViewportMatcapImpl(int64_t textureID);
    virtual void setInteractiveViewportMatcapPresetImpl(int preset);
    // Write a single texture slot into the material-preview descriptor set (binding 1).
    // Called from texture-upload paths so the viewport sees textures as they arrive.
    void updateMaterialPreviewTextureDescriptor(uint32_t slot, const VulkanRT::ImageHandle& img) {
        const uint32_t textureArrayLen = (m_interactiveViewport.materialPreviewTextureArrayLen > 0u)
            ? m_interactiveViewport.materialPreviewTextureArrayLen
            : static_cast<uint32_t>(VULKAN_TEXTURE_CAPACITY);
        if (slot == 0 || slot >= textureArrayLen) return;
        const bool hasDescIdx = m_device && m_device->getCapabilities().supportsDescriptorIndexing;
        if (!hasDescIdx) return;

        const VkDescriptorSet ds = m_interactiveViewport.materialPreviewDescSet;
        if (ds == VK_NULL_HANDLE || !img.view || !img.sampler) return;
        VkDevice vkDev = m_device ? m_device->getDevice() : VK_NULL_HANDLE;
        if (!vkDev) return;
        VkDescriptorImageInfo imgInfo{};
        imgInfo.sampler     = img.sampler;
        imgInfo.imageView   = img.view;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet wds{};
        wds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet          = ds;
        wds.dstBinding      = 1;
        wds.dstArrayElement = slot;
        wds.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wds.descriptorCount = 1;
        wds.pImageInfo      = &imgInfo;
        vkUpdateDescriptorSets(vkDev, 1, &wds, 0, nullptr);
        // Descriptor updates can happen while the viewport is otherwise idle
        // (first texture assign, live paint, project load restore). Force the
        // interactive viewport to redraw immediately instead of waiting for
        // camera motion or a material re-selection.
        m_currentSamples = 0;
        m_hasPresentedRenderedFrame = false;
        m_interactiveViewport.dirty = true;
        m_forceClearOnNextPresent = true;
    }

    struct InteractiveViewportState {
        bool initialized = false;
        bool dirty = true;          // needs re-render (cleared after draw, set on scene change)
        int width = 0;
        int height = 0;
        VulkanRT::ImageHandle colorImage;
        VulkanRT::ImageHandle depthImage;
        VulkanRT::BufferHandle stagingBuffer;
        VkRenderPass renderPass = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline solidPipeline = VK_NULL_HANDLE;
        // Material Preview pipeline (PBR shading with material data)
        VkPipeline materialPreviewPipeline = VK_NULL_HANDLE;
        VkPipelineLayout materialPreviewPipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout materialPreviewDescLayout = VK_NULL_HANDLE;
        VkDescriptorPool materialPreviewDescPool = VK_NULL_HANDLE;
        VkDescriptorSet materialPreviewDescSet = VK_NULL_HANDLE;
        uint32_t materialPreviewTextureArrayLen = 0;
        // Baked environment maps for specular reflection (binding 2): [0]=studio, [1]=outdoor
        int64_t envMapStudioID  = 0;
        int64_t envMapOutdoorID = 0;
        // Matcap resources for solid viewport
        VulkanRT::ImageHandle matcapImage; // optional matcap texture
        bool matcapUserLoaded = false;     // true only when user explicitly loaded a matcap texture
        int matcapPreset = 2;              // default procedural preset when no user texture (2..9)
        // (no custom matcap storage)
        VkDescriptorSetLayout matcapDescLayout = VK_NULL_HANDLE;
        VkDescriptorPool matcapDescPool = VK_NULL_HANDLE;
        VkDescriptorSet matcapDescSet = VK_NULL_HANDLE;

        // Reference grid (Solid/Matcap viewport overlay)
        VulkanRT::BufferHandle gridVertexBuffer;
        VulkanRT::BufferHandle gridNormalBuffer;
        VulkanRT::BufferHandle identityInstanceBuffer;
        uint32_t gridVertexCount = 0;
        uint32_t gridSegments[10] = {}; // [start,count] pairs: major, minor, axisU, axisV, negAxis
        int gridBuiltPlane = -1;       // plane the cached grid was built for: 0=XZ,1=XY,2=YZ (-1=none)
        float gridBuiltSpacing = 0.0f; // adaptive line spacing the cached grid was built for (0=none)
        float gridBuiltHalf = 0.0f;    // half-extent the cached grid was built with (world units)
        float gridBuiltCenterU = 0.0f; // minor-lattice patch centre (camera footprint, major-snapped)
        float gridBuiltCenterV = 0.0f;
        float gridBuiltFineHalf = 0.0f; // minor-lattice half-extent (drives the minor fade band)

        // Hair polyline overlay (Solid/Matcap viewport)
        VkPipeline hairLinePipeline = VK_NULL_HANDLE;
        VulkanRT::BufferHandle hairLineVertexBuffer;
        uint32_t hairLineVertexCount = 0;

        // Particle billboard overlay (Solid/Matcap viewport). Two pipelines share
        // the same shaders/layout and differ only in blend state; particles are
        // grouped into the matching vertex buffer by their system's blend mode.
        VkPipeline particleAddPipeline = VK_NULL_HANDLE;    // additive (fire/spark)
        VkPipeline particleAlphaPipeline = VK_NULL_HANDLE;  // alpha (smoke/dust)
        VulkanRT::BufferHandle particleAddVertexBuffer;
        uint32_t particleAddVertexCount = 0;
        VulkanRT::BufferHandle particleAlphaVertexBuffer;
        uint32_t particleAlphaVertexCount = 0;

        // Edit-mesh overlay (Solid/Matcap viewport): GPU wireframe, vertex
        // markers and face fills for mesh edit mode. Own pipeline layout
        // (push-constant only, 128 bytes) shared by all three pipelines.
        VkPipelineLayout editOverlayPipelineLayout = VK_NULL_HANDLE;
        VkPipeline editLinePipeline = VK_NULL_HANDLE;   // LINE_LIST, indexed
        VkPipeline editFacePipeline = VK_NULL_HANDLE;   // TRIANGLE_LIST, indexed, alpha blend
        VkPipeline editPointPipeline = VK_NULL_HANDLE;  // instanced billboard discs
        VulkanRT::BufferHandle editPositionBuffer;      // vec3 object-local per editable vertex
        VulkanRT::BufferHandle editFlagBuffer;          // uint per vertex: bit0 selected, bits 8..15 soft weight
        uint32_t editVertexCount = 0;
        VulkanRT::BufferHandle editEdgeIndexBuffer;     // all unique edges (uint32 pairs)
        uint32_t editEdgeIndexCount = 0;
        VulkanRT::BufferHandle editFaceIndexBuffer;     // triangulated fan of every face
        uint32_t editFaceIndexCount = 0;
        VulkanRT::BufferHandle editSelEdgeIndexBuffer;  // selected edges
        uint32_t editSelEdgeIndexCount = 0;
        VulkanRT::BufferHandle editSelFaceIndexBuffer;  // triangulated selected faces
        uint32_t editSelFaceIndexCount = 0;
        EditMeshOverlayParams editOverlayParams;

        // Selection outline (Solid/Matcap/MaterialPreview viewport): GPU
        // silhouette mask + screen-space edge composite. Render passes and
        // pipelines are created once with the other viewport pipelines; the
        // mask image + framebuffers share the colorImage/depthImage resize
        // lifecycle. selectionInstanceBuffer holds one mat4 per selected
        // instance, re-uploaded only on selection/transform change frames
        // (the viewport only re-renders when dirty anyway).
        VkRenderPass selectionMaskRenderPass = VK_NULL_HANDLE;
        VkRenderPass selectionCompositeRenderPass = VK_NULL_HANDLE;
        VkPipelineLayout selectionMaskPipelineLayout = VK_NULL_HANDLE;
        VkPipeline selectionMaskFullPipeline = VK_NULL_HANDLE;    // depth test OFF -> writes G
        VkPipeline selectionMaskVisiblePipeline = VK_NULL_HANDLE; // depth test ON  -> writes R
        VkPipelineLayout selectionCompositePipelineLayout = VK_NULL_HANDLE;
        VkPipeline selectionCompositePipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayout selectionCompositeDescLayout = VK_NULL_HANDLE;
        VkDescriptorPool selectionCompositeDescPool = VK_NULL_HANDLE;
        VkDescriptorSet selectionCompositeDescSet = VK_NULL_HANDLE;
        VkSampler selectionMaskSampler = VK_NULL_HANDLE;
        VulkanRT::ImageHandle selectionMaskImage; // R8G8_UNORM, viewport-sized
        VkFramebuffer selectionMaskFramebuffer = VK_NULL_HANDLE;
        VkFramebuffer selectionCompositeFramebuffer = VK_NULL_HANDLE;
        VulkanRT::BufferHandle selectionInstanceBuffer;
        SelectionOutlineParams selectionOutlineParams;
    };

    bool shouldUseInteractiveViewport() const;
    bool ensureInteractiveViewportResources(const std::string& shaderDir, int width, int height);
    void destroyInteractiveViewportResources(bool keepPipeline = false);
    void renderInteractiveViewport(void* outSurface, int width, int height, void* outFramebuffer, void* outTexture);
    void purgeUploadedTextureCacheLocked();
    int64_t uploadCompressedTexture2D(const void* data, uint64_t dataSize, uint32_t width, uint32_t height, VkFormat format);

    std::unique_ptr<VulkanRT::VulkanDevice> m_device;
    
    // Render state
    int m_imageWidth = 0;
    int m_imageHeight = 0;
    int m_currentSamples = 0;
    int m_targetSamples = 0;
    int m_minSamples = 4;
    float m_currentTime = 0.0f;
    bool m_testInitialized = false;
    CameraParams m_camera;
    bool m_useAdaptiveSampling = false;
    float m_varianceThreshold = 0.05f;
    int m_maxBounces = 12;
    int m_diffuseBounces = 4;
    int m_transmissionBounces = 8;
    ViewportMode m_viewportMode = ViewportMode::Rendered;
    InteractiveViewportState m_interactiveViewport;
    bool m_loggedInteractiveViewportFallback = false;
    bool m_materialPreviewUnsupportedNotified = false;
    bool m_materialPreviewPipelineGaveUp = false;
    std::vector<float> m_hdrPixels;   // Float32 HDR readback buffer — her frame realloc önler
    std::vector<uint16_t> m_halfPixels;
    std::vector<float> m_denoiserColorPixels;
    std::vector<float> m_denoiserAlbedoPixels;
    std::vector<float> m_denoiserNormalPixels;
    std::vector<float> m_denoiserPositionPixels;   // Stylize AOV: stride 4 (x,y,z,encoded matid), bottom-up
    // Output resources
    VulkanRT::ImageHandle m_outputImage;
    VulkanRT::BufferHandle m_stagingBuffer;
    // GPU-tonemapped LDR target: RGBA8 storage image + small staging.
    // When valid, render path skips the per-frame CPU Reinhard+sRGB loop.
    // Two stagings + the device's two frame slots implement frames-in-flight pipelining:
    // CPU consumes slot[(N-1)%2] while GPU writes slot[N%2]. Single shared LDR image
    // is safe because both submissions go to the same queue in order.
    VulkanRT::ImageHandle m_tonemappedImage;
    VulkanRT::BufferHandle m_tonemappedStagings[2];
    bool m_tonemappedSlotInFlight[2] = {false, false};
    uint32_t m_tonemappedFrameSlot = 0;
    VulkanRT::ImageHandle m_denoiserColorImage;
    VulkanRT::ImageHandle m_denoiserAlbedoImage;
    VulkanRT::ImageHandle m_denoiserNormalImage;
    VulkanRT::ImageHandle m_varianceImage;
    VulkanRT::BufferHandle m_denoiserColorStagingBuffer;
    VulkanRT::BufferHandle m_denoiserAlbedoStagingBuffer;
    VulkanRT::BufferHandle m_denoiserNormalStagingBuffer;
    // Stylize AOV: primary-hit world position + depth (rgba32f). Host-readback only;
    // no CUDA interop needed since stylize is a CPU post pass.
    VulkanRT::ImageHandle m_denoiserPositionImage;
    VulkanRT::BufferHandle m_denoiserPositionStagingBuffer;

    // GPU stylize: host-visible color SSBO (graded surface round-trip) + params SSBO.
    // Persistent; color buffer reallocated on resolution change.
    VulkanRT::BufferHandle m_stylizeColorBuf;
    VulkanRT::BufferHandle m_stylizeParamsBuf;
    int m_stylizeColorW = 0;
    int m_stylizeColorH = 0;

    // ---- GPU-direct denoiser interop (Vulkan→CUDA for OIDN) ----
    // Parallel set of staging buffers whose VkDeviceMemory is exportable to
    // CUDA as external memory. The CPU host-path uses the non-shared buffers
    // above; the GPU-direct path uses these shared ones. Interop state
    // (CUDA-imported memory + device ptrs + CUDA-owned prep output buffers)
    // lives behind an opaque pointer so the public header stays CUDA-free.
    VulkanRT::BufferHandle m_denoiserColorSharedStaging;
    VulkanRT::BufferHandle m_denoiserAlbedoSharedStaging;
    VulkanRT::BufferHandle m_denoiserNormalSharedStaging;
    void* m_denoiserColorSharedHandle = nullptr;   // Win32 HANDLE; owned by CUDA once imported
    void* m_denoiserAlbedoSharedHandle = nullptr;
    void* m_denoiserNormalSharedHandle = nullptr;
    uint64_t m_denoiserColorSharedAllocSize = 0;
    uint64_t m_denoiserAlbedoSharedAllocSize = 0;
    uint64_t m_denoiserNormalSharedAllocSize = 0;
    struct VulkanCudaDenoiserInterop;
    VulkanCudaDenoiserInterop* m_gpuDenoiserInterop = nullptr;
    // Clamp false→true transitions: once the GPU path is known to be
    // unsupported on this device (e.g., UUID mismatch), stop retrying.
    bool m_gpuDenoiserDisabled = false;

    // Lazy: imports exportable staging memory into CUDA + allocates prep buffers.
    // Returns true if the interop is ready for this frame's dimensions.
    bool ensureGpuDenoiserInterop(int width, int height, bool needAux);
    void destroyGpuDenoiserInterop();
    
    // Status callback
    std::function<void(const std::string&, int)> m_statusCallback;
    
    // Mesh registry (meshName -> BLAS index)
    std::unordered_map<std::string, uint32_t> m_meshRegistry;

    // CPU mirror of each BLAS's per-triangle material indices (blasIndex -> ids).
    // The hit shaders read materials through geo.materialAddr (baked into the
    // BLAS geometry buffer at build); material *assignment* changes must rewrite
    // that region in place, which requires knowing the current per-triangle ids.
    std::unordered_map<uint32_t, std::vector<uint32_t>> m_blasMaterialIds;

    // Phase 3d: backend-owned static device-resident CC meshes (geometry on the GPU; no
    // host triangles). The heavy rebuild appends a BLAS + TLAS instance for each. Buffers
    // queued in m_deviceResidentReleaseQueue are freed only after the BLAS teardown +
    // GPU idle inside rebuildAccelerationStructure (the hit shader reads them while live).
    std::vector<DeviceResidentCCMesh> m_deviceResidentMeshes;
    std::vector<uint64_t> m_deviceResidentReleaseQueue;

    // Phase 3d Slice 4: a position-only cage edit (same triangle count) re-evaluates the CC
    // stencils into a NEW device buffer, then REFITS the existing BLAS to it instead of a full
    // scene rebuild. queueDeviceResidentCCRefit (UI thread) coalesces per meshKey;
    // drainDeviceResidentCCRefits (render thread) MODE_UPDATE-refits each BLAS + TLAS.
    struct DeviceCCRefitRequest {
        std::string meshKey;
        uint64_t newGeomBase = 0;   // device address of the freshly-evaluated combined buffer
        uint64_t newBufferId = 0;   // compute-backend handle id (replaces the entry's buffer)
        uint32_t vertexCount = 0;   // triCount * 3 (must match the registered BLAS to refit)
    };
    std::vector<DeviceCCRefitRequest> m_deviceResidentRefitQueue;

    // Volume instance tracking for Vulkan volume rendering
    uint32_t m_volumeBlasIndex = UINT32_MAX; // Shared AABB BLAS for all volumes

    // Foam / fluid-splat point-sphere tracking. A point_sphere_mode InstanceGroup
    // becomes ONE AABB BLAS (binding-18 sphere SSBO) + ONE TLAS instance instead of
    // N per-particle instances. The BLAS is refit in place each frame via
    // updateFoamSphereBLAS(); a pool-capacity change forces a full rebuild.
    uint32_t m_foamSphereBlasIndex   = UINT32_MAX; // index into m_device->m_blasList
    int      m_foamSphereGroupId     = -1;         // owning InstanceGroup::id
    uint32_t m_foamSpherePoolCapacity = 0;         // stable primitive count for refit

    // TLAS storage for updates
    std::vector<VulkanRT::TLASInstance> m_vkInstances;
    std::vector<std::shared_ptr<Hittable>> m_lastObjects;
    // Instance source pointers for fast sync (parallel to m_vkInstances)
    std::vector<std::shared_ptr<Hittable>> m_instanceSources;
    // Solo-mesh BLAS build-order triangle pointers (blasIndex -> ordered tris).
    // Captured during full geometry build so the interactive sculpt/edit refit
    // (updateMeshBLASPartial) can upload vertex positions in the EXACT slot order
    // the BLAS was built with — the editable mesh_cache iteration order is not
    // guaranteed to match the scene-graph traversal order used at build time, and
    // a mismatch would scatter positions into the wrong BLAS slots (corruption).
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<Triangle>>> m_soloBlasBuildTriangles;
    std::unordered_map<uint32_t, std::unordered_map<const Triangle*, uint32_t>> m_soloBlasBuildTriangleSlots;

    // Solo BLASes built via uploadTriangleMeshIndexed() (indexed geometry). These do NOT use
    // m_soloBlasBuildTriangles for refit — their interactive refit re-reads the mesh SoA
    // directly (vCount unique verts), so we record the source mesh + space here instead.
    struct SoloIndexedMeshInfo {
        TriangleMesh* mesh = nullptr;
        bool localSpace = false; // true => refit reads bind/local P_orig, false => live P
    };
    std::unordered_map<uint32_t, SoloIndexedMeshInfo> m_soloBlasIndexedMesh;
    // Re-reads a solo indexed BLAS's mesh SoA and MODE_UPDATE-refits it (no 3N gather).
    bool refitIndexedSoloBLAS(uint32_t blasIndex);
    struct InstanceTransformCache { int instance_id; std::shared_ptr<Hittable> representative_hittable; };
    std::vector<InstanceTransformCache> m_instance_sync_cache;
    bool m_topology_dirty = false; // set when instances/BLAS list changes
    // Lifetime sentinel shared with destroyFn lambdas stored in SceneTextureManager.
    // Set to false before container teardown so lambdas that outlive this backend
    // skip the container-erase step and only destroy GPU handles.
    std::shared_ptr<bool> m_containerAlive = std::make_shared<bool>(true);
    // Uploaded textures (map from opaque handle/key -> ImageHandle)
    std::unordered_map<int64_t, VulkanRT::ImageHandle> m_uploadedImages;
    std::unordered_map<uint64_t, int64_t> m_uploadedImageIDs;     // cacheKey (pointer+flags) -> textureId
    std::unordered_map<int64_t, uint64_t> m_textureIdToCacheKey;  // reverse: textureId -> cacheKey (O(1) eviction cleanup)
    int64_t m_nextTextureID = 1;
    uint64_t m_textureCacheGeneration = 1;
    std::shared_ptr<SceneTextureManager> m_sceneTextureManager;
    uint64_t m_textureUploadBytes = 0;
    uint32_t m_textureUploadCount = 0;
    uint32_t m_textureUploadBC4Count = 0;
    uint32_t m_textureUploadBC5Count = 0;
    uint32_t m_textureUploadBC7Count = 0;
    uint32_t m_textureUploadR8Count = 0;
    uint32_t m_textureUploadRGBA8Count = 0;
    uint32_t m_textureUploadFloatCount = 0;
    uint32_t m_textureUploadRG8Count = 0;
    bool m_textureUploadSummaryDirty = false;
    
    // NanoVDB grid device buffers mapped by vdb_id
    std::unordered_map<int, VulkanRT::BufferHandle> m_vdbBuffers;
    // NanoVDB temperature grid device buffers mapped by vdb_id (fire/blackbody)
    std::unordered_map<int, VulkanRT::BufferHandle> m_vdbTempBuffers;
    // Mapped content versions for NanoVDB grids to prevent redundant CPU-to-GPU kopyalama
    std::unordered_map<int, uint32_t> m_vdbUploadedVersions;
    std::unordered_map<int, uint32_t> m_vdbTempUploadedVersions;
    // VDB volumes in the ORDER they were added to the TLAS (customIndex 0,1,2...)
    // Guarantees SSBO layout matches gl_InstanceCustomIndexEXT lookups in the shader.
    std::vector<std::shared_ptr<Hittable>> m_orderedVDBInstances;
    // Cached lights when device not yet ready
    std::vector<std::shared_ptr<Light>> m_cachedLights;
    // Cached world data when device not yet ready
    WorldData m_cachedWorld;
    int64_t m_envTexID = 0;
    // Set to true after uploadAtmosphereLUT() succeeds — used to set _pad5 (nishitaLutReady) in world buffer
    bool m_atmosphereLutReady = false;
    bool m_atmosphereLutGenerationInProgress = false;
    uint64_t m_lastCameraHash = 0;
    Vec3 m_prevViewDir;
    bool  m_hasPrevView = false;
    // When true, clear the UI framebuffer/texture on next renderProgressive call
    bool m_forceClearOnNextPresent = false;
    // Tracks whether Rendered mode has produced at least one valid host-visible frame.
    bool m_hasPresentedRenderedFrame = false;
    // [PERF] True after resetAccumulation() already cleared GPU images — skip redundant
    // frame-0 clears in renderProgressiveImpl to avoid double work.
    bool m_imagesCleared = false;

    // ── Hair geometry ────────────────────────────────────────────────────────
    // Hair TLAS instances (kept separate from mesh instances so they survive
    // partial scene reloads and are merged in at TLAS build time).
    std::vector<VulkanRT::TLASInstance> m_hairVkInstances;
    // groomName -> BLAS index (so we can re-upload strands when they change)
    std::unordered_map<std::string, uint32_t> m_hairGroomRegistry;

    // Number of mesh BLASes stored at the start of m_device->m_blasList.
    // Hair BLASes are always appended after this index so clearHairGeometry()
    // can destroy and remove only the hair BLASes without touching mesh BLASes.
    uint32_t m_meshBlasCount = 0;

    // ── Raster mesh buffers (lightweight, BLAS/TLAS-independent) ────────────
    struct RasterMeshBuffer {
        struct CullingChunk {
            AABB worldBBox;
            std::vector<uint32_t> instanceIndices;
        };

        VulkanRT::BufferHandle vertexBuffer;   // float3 positions (GPU_ONLY)
        VulkanRT::BufferHandle normalBuffer;   // float3 normals  (GPU_ONLY)
        VulkanRT::BufferHandle uvBuffer;       // float2 UVs (GPU_ONLY) — for MaterialPreview
        VulkanRT::BufferHandle matIdBuffer;    // uint32 per-vertex materialID — for MaterialPreview
        VulkanRT::BufferHandle indexBuffer;     // uint32 indices (optional)
        VulkanRT::BufferHandle instanceBuffer;  // per-instance model matrices
        VulkanRT::BufferHandle baseVertexBuffer;
        VulkanRT::BufferHandle baseNormalBuffer;
        VulkanRT::BufferHandle boneIndexBuffer;
        VulkanRT::BufferHandle boneWeightBuffer;
        VulkanRT::BufferHandle persistentBoneMatsBuffer;
        VkDescriptorSet skinningDescSet = VK_NULL_HANDLE;
        uint64_t persistentBoneMatsBufSize = 0;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        uint32_t instanceCount = 0;
        bool hasSkinning = false;
        bool isScatterGroup = false;
        bool isScatterProxy = false;
        std::string proxyMeshKey;
        std::vector<uint32_t> instanceIndices;  // indices into m_rasterInstances
        std::vector<CullingChunk> cullingChunks;
        std::vector<uint32_t> visibleInstanceIndicesCache;
        bool visibleInstancesDirty = true;
        uint64_t lastVisibleFrustumRevision = 0;
        uint64_t lastScatterTriangleBudget = 0;
        // CPU shadow copy for dirty-range detection during sculpt
        std::vector<float> cpuPositions;
        std::vector<float> cpuNormals;
        std::vector<uint32_t> cpuMatIds;
    };

    struct RasterInstance {
        std::string meshKey;       // key into m_rasterMeshes
        std::string nodeName;      // for transform sync lookup
        Matrix4x4 transform;
        AABB localBBox;            // local-space bounding box (from source mesh)
        AABB worldBBox;            // world-space AABB (localBBox transformed)
        uint8_t mask = 0xFF;       // visibility
        int scatterGroupId = -1;   // direct InstanceManager lookup for large scatter groups
        uint32_t scatterInstanceIndex = UINT32_MAX;
    };

    std::unordered_map<std::string, RasterMeshBuffer> m_rasterMeshes;
    std::vector<RasterInstance> m_rasterInstances;
    std::vector<VulkanRT::VkGpuMaterial> m_cachedGpuMaterials;
    std::unordered_map<std::string, std::vector<uint32_t>> m_rasterNodeIndex;
    std::unordered_map<std::string, std::vector<uint32_t>> m_rtNodeIndex;
    size_t m_rasterNodeIndexInstanceCount = 0;
    size_t m_rtNodeIndexInstanceCount = 0;
    bool m_rasterGeometryDirty = true;  // force initial build

    // Generation counter: last g_scene_geometry_generation value seen when
    // raster geometry was built.  Allows skipping redundant rebuilds on
    // Rendered→Solid transitions when scene geometry hasn't changed.
    uint64_t m_rasterBuiltGeometryGeneration = 0;

    // Frustum culling for solid viewport
    struct FrustumPlane {
        Vec3 normal;
        float d;
        float distanceTo(const Vec3& p) const { return Vec3::dot(normal, p) + d; }
    };
    FrustumPlane m_frustumPlanes[6];
    uint64_t m_rasterFrustumRevision = 0;
    uint64_t m_rasterFrustumHash = 0;
    Vec3 m_rasterCullCameraPosition;
    float m_rasterCullFocalLengthPixels = 0.0f;
    float m_rasterMinChunkScreenRadiusPixels = 2.0f;
    uint64_t m_rasterScatterTriangleBudget = 96ull * 1000ull * 1000ull;
    float m_rasterScatterBudgetScale = 1.0f;
    void extractFrustumPlanes(const Matrix4x4& viewProj);
    bool isAABBInFrustum(const AABB& box) const;
    bool isAABBFullyInsideFrustum(const AABB& box) const;
    bool isRasterChunkTooSmallToDraw(const AABB& box) const;
    void updateRasterInstanceWorldBBox(RasterInstance& ri) const;
    void rebuildRasterMeshCullingChunks(RasterMeshBuffer& mesh);
    void setRasterVisibleInstances(RasterMeshBuffer& mesh, const std::vector<uint32_t>& visibleInstanceIndices);
    void uploadVisibleRasterInstances(RasterMeshBuffer& mesh);

    // Selection outline internals shared by the raster-mode composite path
    // and the Rendered-mode mask readback. Caller must hold m_mutex.
    struct SelectionOutlineDrawItem {
        const RasterMeshBuffer* mesh = nullptr;
        VkDeviceSize instanceByteOffset = 0;
        float maskValue = 1.0f; // 1.0 = primary selection tier, 0.5 = secondary
    };
    // Resolves node names against m_rasterInstances and uploads one matrix
    // per matched instance into selectionInstanceBuffer (must run before
    // command recording starts — the upload submits its own transfer).
    void resolveSelectionOutlineDraws(const std::vector<std::string>& nodeNames,
                                      std::vector<SelectionOutlineDrawItem>& outDraws);
    // Records the R8G8 mask render pass (G = depth-test-off silhouette,
    // R = visible vs the depth already in depthImage) into cmd.
    void recordSelectionOutlineMaskPass(VkCommandBuffer cmd,
                                        const std::vector<SelectionOutlineDrawItem>& draws,
                                        const Matrix4x4& viewProj, int width, int height);
    void invalidateTargetedTransformIndex();
    void rebuildTargetedTransformIndex();

    // Per-mesh local AABB cache (computed once during buildRasterGeometry)
    std::unordered_map<std::string, AABB> m_rasterMeshBBoxes;

    void destroyRasterMesh(RasterMeshBuffer& mesh);
    void destroyAllRasterMeshes();
    void uploadRasterInstanceBuffer(RasterMeshBuffer& mesh);

    // Batched texture upload — reduces N GPU submits to 1 during uploadMaterials
    bool m_inBatchedTextureUpload = false;
    VkCommandBuffer m_batchTextureCmd = VK_NULL_HANDLE;
    std::vector<VulkanRT::BufferHandle> m_batchTextureStagingBuffers;
    void beginBatchedTextureUpload();
    void endBatchedTextureUpload();

    mutable std::recursive_mutex m_mutex;
};

} // namespace Backend

#endif // VULKAN_BACKEND_H
