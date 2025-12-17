/**
 * @file unified_converters.h
 * @brief Conversion functions between legacy CPU types and unified types
 * 
 * This header provides functions to convert between:
 * - CPU classes (PrincipledBSDF, Light subclasses) -> UnifiedMaterial/UnifiedLight
 * - UnifiedMaterial/UnifiedLight -> GPU structs (GpuMaterial, LightGPU)
 * 
 * This allows gradual migration while maintaining backward compatibility.
 */
#pragma once

#include "unified_types.h"
#include "Vec3.h"  // CPU Vec3
#include <memory>

// Forward declarations of legacy CPU types
class PrincipledBSDF;
class Light;
class PointLight;
class DirectionalLight;
class AreaLight;
class SpotLight;

// Forward declaration of GPU types (only available when CUDA is present)
#ifdef __CUDACC__
#include "material_gpu.h"
#include "params.h"
#endif

// =============================================================================
// VEC3 <-> VEC3F CONVERSION
// =============================================================================

/**
 * @brief Convert CPU Vec3 to unified Vec3f
 */
inline Vec3f toVec3f(const Vec3& v) {
    return Vec3f(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}

/**
 * @brief Convert unified Vec3f to CPU Vec3
 */
inline Vec3 toVec3(const Vec3f& v) {
    return Vec3(v.x, v.y, v.z);
}

// =============================================================================
// PRINCIPLEDBSDF -> UNIFIEDMATERIAL
// =============================================================================

/**
 * @brief Convert PrincipledBSDF to UnifiedMaterial
 * 
 * This extracts the core material properties from the CPU class
 * into a simple POD struct that can be used on both CPU and GPU.
 * 
 * @param bsdf Source PrincipledBSDF material
 * @param material_id Material ID for texture lookup
 * @return Unified material structure
 */
UnifiedMaterial toUnifiedMaterial(const PrincipledBSDF& bsdf, int material_id);

/**
 * @brief Convert Material base class pointer to UnifiedMaterial
 * 
 * Handles polymorphic materials by checking type and casting.
 * Currently only PrincipledBSDF is fully supported.
 * 
 * @param material Source material pointer
 * @param material_id Material ID for texture lookup
 * @return Unified material structure
 */
UnifiedMaterial toUnifiedMaterial(const std::shared_ptr<class Material>& material, int material_id);

// =============================================================================
// LIGHT CLASSES -> UNIFIEDLIGHT
// =============================================================================

/**
 * @brief Convert any Light subclass to UnifiedLight
 * 
 * Detects light type and extracts appropriate parameters.
 * 
 * @param light Source light pointer
 * @return Unified light structure
 */
UnifiedLight toUnifiedLight(const std::shared_ptr<Light>& light);

/**
 * @brief Convert PointLight to UnifiedLight
 */
UnifiedLight pointLightToUnified(const PointLight& light);

/**
 * @brief Convert DirectionalLight to UnifiedLight
 */
UnifiedLight directionalLightToUnified(const DirectionalLight& light);

/**
 * @brief Convert AreaLight to UnifiedLight
 */
UnifiedLight areaLightToUnified(const AreaLight& light);

/**
 * @brief Convert SpotLight to UnifiedLight
 */
UnifiedLight spotLightToUnified(const SpotLight& light);

// =============================================================================
// UNIFIEDMATERIAL -> GPUMATERIAL (CUDA ONLY)
// =============================================================================

#ifdef __CUDACC__

/**
 * @brief Convert UnifiedMaterial to GPU's GpuMaterial
 * 
 * This is used when uploading materials to the GPU.
 * 
 * @param unified Source unified material
 * @return GPU material structure
 */
inline GpuMaterial toGpuMaterial(const UnifiedMaterial& unified) {
    GpuMaterial gpu;
    
    gpu.albedo = make_float3(unified.albedo.x, unified.albedo.y, unified.albedo.z);
    gpu.opacity = unified.opacity;
    gpu.roughness = unified.roughness;
    gpu.metallic = unified.metallic;
    gpu.clearcoat = unified.clearcoat;
    gpu.transmission = unified.transmission;
    gpu.emission = make_float3(unified.emission.x, unified.emission.y, unified.emission.z);
    gpu.ior = unified.ior;
    gpu.subsurface_color = make_float3(
        unified.subsurface_color.x, 
        unified.subsurface_color.y, 
        unified.subsurface_color.z
    );
    gpu.subsurface = unified.subsurface;
    gpu.artistic_albedo_response = unified.artistic_albedo_response;
    gpu.anisotropic = unified.anisotropic;
    gpu.sheen = unified.sheen;
    gpu.sheen_tint = unified.sheen_tint;
    
    return gpu;
}

/**
 * @brief Convert GpuMaterial back to UnifiedMaterial
 * 
 * Useful for debugging or when GPU data needs to be read back.
 * 
 * @param gpu Source GPU material
 * @return Unified material structure
 */
inline UnifiedMaterial fromGpuMaterial(const GpuMaterial& gpu) {
    UnifiedMaterial unified;
    
    unified.albedo = Vec3f(gpu.albedo.x, gpu.albedo.y, gpu.albedo.z);
    unified.opacity = gpu.opacity;
    unified.roughness = gpu.roughness;
    unified.metallic = gpu.metallic;
    unified.clearcoat = gpu.clearcoat;
    unified.transmission = gpu.transmission;
    unified.emission = Vec3f(gpu.emission.x, gpu.emission.y, gpu.emission.z);
    unified.ior = gpu.ior;
    unified.subsurface_color = Vec3f(
        gpu.subsurface_color.x, 
        gpu.subsurface_color.y, 
        gpu.subsurface_color.z
    );
    unified.subsurface = gpu.subsurface;
    unified.artistic_albedo_response = gpu.artistic_albedo_response;
    unified.anisotropic = gpu.anisotropic;
    unified.sheen = gpu.sheen;
    unified.sheen_tint = gpu.sheen_tint;
    
    // Texture IDs are not stored in GpuMaterial, leave as -1
    
    return unified;
}

/**
 * @brief Convert UnifiedLight to GPU's LightGPU
 * 
 * @param unified Source unified light
 * @return GPU light structure
 */
inline LightGPU toGpuLight(const UnifiedLight& unified) {
    LightGPU gpu;
    
    gpu.position = make_float3(unified.position.x, unified.position.y, unified.position.z);
    gpu.direction = make_float3(unified.direction.x, unified.direction.y, unified.direction.z);
    gpu.color = make_float3(unified.color.x, unified.color.y, unified.color.z);
    gpu.intensity = unified.intensity;
    gpu.radius = unified.radius;
    gpu.type = unified.type;
    gpu.inner_cone_cos = unified.inner_cone_cos;
    gpu.outer_cone_cos = unified.outer_cone_cos;
    gpu.area_width = unified.area_width;
    gpu.area_height = unified.area_height;
    gpu.area_u = make_float3(unified.area_u.x, unified.area_u.y, unified.area_u.z);
    gpu.area_v = make_float3(unified.area_v.x, unified.area_v.y, unified.area_v.z);
    
    return gpu;
}

/**
 * @brief Convert LightGPU back to UnifiedLight
 * 
 * @param gpu Source GPU light
 * @return Unified light structure
 */
inline UnifiedLight fromGpuLight(const LightGPU& gpu) {
    UnifiedLight unified;
    
    unified.position = Vec3f(gpu.position.x, gpu.position.y, gpu.position.z);
    unified.direction = Vec3f(gpu.direction.x, gpu.direction.y, gpu.direction.z);
    unified.color = Vec3f(gpu.color.x, gpu.color.y, gpu.color.z);
    unified.intensity = gpu.intensity;
    unified.radius = gpu.radius;
    unified.type = gpu.type;
    unified.inner_cone_cos = gpu.inner_cone_cos;
    unified.outer_cone_cos = gpu.outer_cone_cos;
    unified.area_width = gpu.area_width;
    unified.area_height = gpu.area_height;
    unified.area_u = Vec3f(gpu.area_u.x, gpu.area_u.y, gpu.area_u.z);
    unified.area_v = Vec3f(gpu.area_v.x, gpu.area_v.y, gpu.area_v.z);
    
    return unified;
}

#endif // __CUDACC__
