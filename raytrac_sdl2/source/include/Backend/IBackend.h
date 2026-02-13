/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          IBackend.h
 * Description:   Abstract Render Backend Interface
 *                Allows switching between OptiX (CUDA), Vulkan, Metal
 * =========================================================================
 */
#ifndef I_BACKEND_H
#define I_BACKEND_H

#include "Vec3.h"
#include "Vec2.h"
#include "Matrix4x4.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace Backend {

// ============================================================================
// Backend Types
// ============================================================================

enum class BackendType : uint8_t {
    AUTO,           // Auto-detect best available
    OPTIX,          // NVIDIA OptiX (CUDA)
    VULKAN_RT,      // Vulkan with ray tracing
    VULKAN_COMPUTE, // Vulkan compute-only fallback
    METAL,          // Apple Metal (future)
    CPU_EMBREE,     // CPU fallback with Embree
    CPU_CUSTOM      // Custom CPU BVH
};

struct BackendInfo {
    BackendType type;
    std::string name;
    std::string deviceName;
    std::string driverVersion;
    bool hasHardwareRT;
    uint64_t vramBytes;
};

// ============================================================================
// Common Data Structures
// ============================================================================

struct RenderParams {
    int imageWidth;
    int imageHeight;
    int samplesPerPixel;
    int maxBounces;
    int currentPass;
    int frameNumber;
    bool useAdaptiveSampling;
    float adaptiveThreshold;
};

struct CameraParams {
    Vec3 origin;
    Vec3 lookAt;
    Vec3 up;
    float fov;
    float aperture;
    float focusDistance;
    float aspectRatio;
};

// Triangle data for upload
struct TriangleData {
    Vec3 v0, v1, v2;        // Positions
    Vec3 n0, n1, n2;        // Normals
    Vec2 uv0, uv1, uv2;     // Texture coords
    uint16_t materialID;
};

// Hair strand data for upload
struct HairStrandData {
    std::vector<Vec3> points;
    std::vector<float> radii;
    uint16_t materialID;
    Vec2 rootUV;
};

// ============================================================================
// Abstract Backend Interface
// ============================================================================

class IBackend {
public:
    virtual ~IBackend() = default;
    
    // ========================================================================
    // Initialization
    // ========================================================================
    
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual BackendInfo getInfo() const = 0;
    
    // ========================================================================
    // Geometry Upload
    // ========================================================================
    
    /**
     * @brief Upload triangle mesh geometry
     * @param triangles Vector of triangle data
     * @param meshName Unique identifier for this mesh
     * @return Mesh handle/index
     */
    virtual uint32_t uploadTriangles(
        const std::vector<TriangleData>& triangles,
        const std::string& meshName
    ) = 0;
    
    /**
     * @brief Upload hair strand geometry
     * @param strands Vector of hair strands
     * @param groomName Unique identifier
     * @return Groom handle/index
     */
    virtual uint32_t uploadHairStrands(
        const std::vector<HairStrandData>& strands,
        const std::string& groomName
    ) = 0;
    
    /**
     * @brief Update mesh transform (for instances)
     */
    virtual void updateMeshTransform(uint32_t meshHandle, const Matrix4x4& transform) = 0;
    
    /**
     * @brief Rebuild acceleration structure after geometry changes
     */
    virtual void rebuildAccelerationStructure() = 0;
    
    // ========================================================================
    // Material Upload
    // ========================================================================
    
    struct MaterialData {
        Vec3 albedo;
        float roughness;
        float metallic;
        Vec3 emission;
        float emissionStrength;
        float ior;
        float transmission;
        // Texture handles
        int64_t albedoTexture;
        int64_t normalTexture;
        int64_t roughnessTexture;
        // ... more as needed
    };
    
    virtual void uploadMaterials(const std::vector<MaterialData>& materials) = 0;
    
    // Hair-specific material
    struct HairMaterialData {
        Vec3 color;
        float melanin;
        float melaninRedness;
        float roughness;
        float radialRoughness;
        float ior;
    };
    
    virtual void uploadHairMaterials(const std::vector<HairMaterialData>& materials) = 0;
    
    // ========================================================================
    // Texture Upload
    // ========================================================================
    
    virtual int64_t uploadTexture2D(
        const void* data,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        bool sRGB
    ) = 0;
    
    virtual void destroyTexture(int64_t textureHandle) = 0;
    
    // ========================================================================
    // Rendering
    // ========================================================================
    
    /**
     * @brief Set render parameters
     */
    virtual void setRenderParams(const RenderParams& params) = 0;
    
    /**
     * @brief Set camera for rendering
     */
    virtual void setCamera(const CameraParams& camera) = 0;
    
    /**
     * @brief Execute one render pass
     * @param accumulate If true, blend with previous passes
     */
    virtual void renderPass(bool accumulate = true) = 0;
    
    /**
     * @brief Download rendered image to CPU
     * @param outPixels Output buffer (RGBA float or uint8 depending on format)
     */
    virtual void downloadImage(void* outPixels) = 0;
    
    /**
     * @brief Get current sample count (for progressive rendering)
     */
    virtual int getCurrentSampleCount() const = 0;
    
    // ========================================================================
    // Environment
    // ========================================================================
    
    virtual void setEnvironmentMap(int64_t hdrTextureHandle) = 0;
    virtual void setSkyParams(/* sky parameters */) = 0;
    
    // ========================================================================
    // Utility
    // ========================================================================
    
    virtual void waitForCompletion() = 0;
    virtual void resetAccumulation() = 0;
    
    /**
     * @brief Get estimated render time per sample (for UI feedback)
     */
    virtual float getMillisecondsPerSample() const = 0;
};

// ============================================================================
// Backend Factory
// ============================================================================

/**
 * @brief Create render backend
 * @param preferredType Preferred backend (AUTO for best available)
 * @return Unique pointer to backend instance
 */
std::unique_ptr<IBackend> createBackend(BackendType preferredType = BackendType::AUTO);

/**
 * @brief Get list of available backends
 */
std::vector<BackendInfo> enumerateBackends();

/**
 * @brief Check if specific backend is available
 */
bool isBackendAvailable(BackendType type);

} // namespace Backend

#endif // I_BACKEND_H
