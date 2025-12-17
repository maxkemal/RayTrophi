/**
 * @file unified_converters.cpp
 * @brief Implementation of conversion functions for unified types
 */

#include "unified_converters.h"
#include "PrincipledBSDF.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "AreaLight.h"
#include "SpotLight.h"
#include "Material.h"

// =============================================================================
// PRINCIPLEDBSDF -> UNIFIEDMATERIAL
// =============================================================================

UnifiedMaterial toUnifiedMaterial(const PrincipledBSDF& bsdf, int material_id) {
    UnifiedMaterial unified;
    
    // Sample properties at center UV (0.5, 0.5) for base values
    Vec2 uv(0.5f, 0.5f);
    
    // Core albedo
    Vec3 albedo = bsdf.getPropertyValue(bsdf.albedoProperty, uv);
    unified.albedo = toVec3f(albedo);
    
    // Roughness (stored in Y channel)
    Vec3 roughness_vec = bsdf.getPropertyValue(bsdf.roughnessProperty, uv);
    unified.roughness = static_cast<float>(roughness_vec.y);
    
    // Metallic (stored in Z channel)  
    Vec3 metallic_vec = bsdf.getPropertyValue(bsdf.metallicProperty, uv);
    unified.metallic = static_cast<float>(metallic_vec.z);
    
    // Opacity
    unified.opacity = bsdf.get_opacity(uv);
    
    // Transmission
    unified.transmission = bsdf.getTransmission(uv);
    
    // IOR
    unified.ior = bsdf.getIOR();
    
    // Emission
    Vec3 emission = bsdf.getEmission(uv, Vec3(0.0f));
    unified.emission = toVec3f(emission);
    
    // Texture IDs - use material_id to create unique texture slots
    // The actual texture object binding is done separately
    unified.albedo_tex_id = bsdf.albedoProperty.texture ? material_id * 10 + 0 : -1;
    unified.normal_tex_id = bsdf.normalProperty.texture ? material_id * 10 + 1 : -1;
    unified.roughness_tex_id = bsdf.roughnessProperty.texture ? material_id * 10 + 2 : -1;
    unified.metallic_tex_id = bsdf.metallicProperty.texture ? material_id * 10 + 3 : -1;
    unified.emission_tex_id = bsdf.emissionProperty.texture ? material_id * 10 + 4 : -1;
    unified.opacity_tex_id = bsdf.opacityProperty.texture ? material_id * 10 + 5 : -1;
    unified.transmission_tex_id = bsdf.transmissionProperty.texture ? material_id * 10 + 6 : -1;
    
    // Additional properties (from private members - these may need getter functions)
    // For now, use defaults
    unified.subsurface = 0.0f;
    unified.subsurface_color = Vec3f(0.0f);
    unified.clearcoat = 0.0f;
    unified.anisotropic = 0.0f;
    unified.sheen = 0.0f;
    unified.sheen_tint = 0.0f;
    unified.artistic_albedo_response = 0.0f;
    
    return unified;
}

UnifiedMaterial toUnifiedMaterial(const std::shared_ptr<Material>& material, int material_id) {
    if (!material) {
        // Return default material
        return UnifiedMaterial();
    }
    
    // Check if it's a PrincipledBSDF
    auto pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(material);
    if (pbsdf) {
        return toUnifiedMaterial(*pbsdf, material_id);
    }
    
    // For other material types, create a basic unified material
    UnifiedMaterial unified;
    unified.albedo = Vec3f(0.8f, 0.8f, 0.8f);
    unified.roughness = 0.5f;
    unified.metallic = 0.0f;
    unified.opacity = 1.0f;
    
    return unified;
}

// =============================================================================
// LIGHT CLASSES -> UNIFIEDLIGHT
// =============================================================================

UnifiedLight pointLightToUnified(const PointLight& light) {
    UnifiedLight unified;
    
    // Match GPU: OptixWrapper::setLightParams
    unified.position = toVec3f(light.position);
    unified.color = toVec3f(light.color);
    unified.intensity = light.intensity;
    unified.radius = light.getRadius();
    unified.type = static_cast<int>(UnifiedLightType::Point);  // type = 0
    
    // Direction not used for point lights (GPU doesn't set it)
    unified.direction = Vec3f(0.0f);
    
    // GPU defaults for unused fields
    unified.inner_cone_cos = 1.0f;
    unified.outer_cone_cos = 0.0f;  // GPU: l.outer_cone_cos = 0.0f
    unified.area_width = 0.0f;
    unified.area_height = 0.0f;
    unified.area_u = Vec3f(1.0f, 0.0f, 0.0f);
    unified.area_v = Vec3f(0.0f, 1.0f, 0.0f);  // GPU: make_float3(0, 1, 0)
    
    return unified;
}

UnifiedLight directionalLightToUnified(const DirectionalLight& light) {
    UnifiedLight unified;
    
    // Match GPU: OptixWrapper::setLightParams
    unified.position = Vec3f(0.0f);  // Not meaningful for directional (GPU doesn't set)
    // CRITICAL: Negate direction (GPU does: -dirLight->direction.normalize())
    Vec3 negDir = -light.direction.normalize();
    unified.direction = toVec3f(negDir);
    unified.color = toVec3f(light.color);
    unified.intensity = light.intensity;
    unified.radius = light.getDiskRadius();  // GPU: dirLight->getDiskRadius()
    unified.type = static_cast<int>(UnifiedLightType::Directional);  // type = 1
    
    // GPU defaults for unused fields
    unified.inner_cone_cos = 1.0f;
    unified.outer_cone_cos = 0.0f;  // GPU default
    unified.area_width = 0.0f;
    unified.area_height = 0.0f;
    unified.area_u = Vec3f(1.0f, 0.0f, 0.0f);
    unified.area_v = Vec3f(0.0f, 1.0f, 0.0f);  // GPU: make_float3(0, 1, 0)
    
    return unified;
}

UnifiedLight areaLightToUnified(const AreaLight& light) {
    UnifiedLight unified;
    
    // Match GPU: OptixWrapper::setLightParams
    unified.position = toVec3f(light.position);
    // GPU: areaLight->direction.normalize()
    Vec3 normalizedDir = light.direction.normalize();
    unified.direction = toVec3f(normalizedDir);
    unified.color = toVec3f(light.color);
    unified.intensity = light.intensity;
    unified.radius = 0.0f;  // GPU: l.radius = 0.0f
    unified.type = static_cast<int>(UnifiedLightType::Area);  // type = 2
    
    // AreaLight specific parameters
    unified.area_width = light.getWidth();
    unified.area_height = light.getHeight();
    unified.area_u = toVec3f(light.getU());
    unified.area_v = toVec3f(light.getV());
    
    // GPU doesn't set cone params for area lights (uses defaults)
    unified.inner_cone_cos = 1.0f;
    unified.outer_cone_cos = 0.0f;
    
    return unified;
}

UnifiedLight spotLightToUnified(const SpotLight& light) {
    UnifiedLight unified;
    
    // Match GPU: OptixWrapper::setLightParams
    unified.position = toVec3f(light.position);
    // GPU: spotLight->direction.normalize()
    Vec3 normalizedDir = light.direction.normalize();
    unified.direction = toVec3f(normalizedDir);
    unified.color = toVec3f(light.color);
    unified.intensity = light.intensity;
    unified.radius = 0.0f;  // GPU: l.radius = 0.0f (not 0.01f!)
    unified.type = static_cast<int>(UnifiedLightType::Spot);  // type = 3
    
    // SpotLight cone angle parameters (exactly as GPU)
    float angleDeg = light.getAngleDegrees();
    float angleRad = angleDeg * (3.14159265358979323846f / 180.0f);
    unified.inner_cone_cos = cosf(angleRad * 0.8f);  // Inner cone (80% of outer)
    unified.outer_cone_cos = cosf(angleRad);          // Outer cone (full angle)
    
    // Area params not used
    unified.area_width = 0.0f;
    unified.area_height = 0.0f;
    unified.area_u = Vec3f(1.0f, 0.0f, 0.0f);
    unified.area_v = Vec3f(0.0f, 1.0f, 0.0f);
    
    return unified;
}

UnifiedLight toUnifiedLight(const std::shared_ptr<Light>& light) {
    if (!light) {
        return UnifiedLight();
    }
    
    // Check each light type
    if (auto point = std::dynamic_pointer_cast<PointLight>(light)) {
        return pointLightToUnified(*point);
    }
    
    if (auto directional = std::dynamic_pointer_cast<DirectionalLight>(light)) {
        return directionalLightToUnified(*directional);
    }
    
    if (auto area = std::dynamic_pointer_cast<AreaLight>(light)) {
        return areaLightToUnified(*area);
    }
    
    if (auto spot = std::dynamic_pointer_cast<SpotLight>(light)) {
        return spotLightToUnified(*spot);
    }
    
    // Unknown light type - return default
    return UnifiedLight();
}
