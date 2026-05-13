#pragma once

#include <cstdint>
#include "PrincipledBSDF.h"
#include "material_gpu.h"
#include "Backend/IBackend.h"

struct PBRMaterialSnapshot {
    Vec3 albedo = Vec3(0.8f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float specular = 0.5f;
    Vec3 emission = Vec3(0.0f);
    float emissionStrength = 0.0f;
    float ior = 1.5f;
    float transmission = 0.0f;
    float opacity = 1.0f;

    float subsurface = 0.0f;
    Vec3 subsurfaceColor = Vec3(1.0f);
    Vec3 subsurfaceRadius = Vec3(1.0f);
    float subsurfaceScale = 1.0f;
    float subsurfaceAnisotropy = 0.0f;
    float subsurfaceIOR = 1.33f;
    bool useRandomWalkSSS = true;
    int sssMaxSteps = 6;

    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.0f;
    float translucent = 0.0f;
    float anisotropic = 0.0f;
    float normalStrength = 1.0f;
    float sheen = 0.0f;
    float sheenTint = 0.0f;

    float microDetailStrength = 0.0f;
    float microDetailScale = 0.0f;
    float tileBreakStrength = 0.0f;

    Vec2 uvScale = Vec2(1.0f, 1.0f);
    Vec2 uvOffset = Vec2(0.0f, 0.0f);
    Vec2 uvTiling = Vec2(1.0f, 1.0f);
    float uvRotationDegrees = 0.0f;
    uint32_t uvWrapMode = 0;
};

inline PBRMaterialSnapshot capturePBRMaterialSnapshot(const PrincipledBSDF& pbsdf) {
    PBRMaterialSnapshot s;
    s.albedo = pbsdf.albedoProperty.color;
    s.roughness = pbsdf.getScalarRoughness();
    s.metallic = pbsdf.getScalarMetallic();
    s.specular = pbsdf.specularProperty.intensity;
    s.emission = pbsdf.emissionProperty.color;
    s.emissionStrength = pbsdf.emissionProperty.intensity;
    s.ior = pbsdf.ior;
    s.transmission = pbsdf.transmission;
    s.opacity = pbsdf.opacityProperty.alpha;

    s.subsurface = pbsdf.subsurface;
    s.subsurfaceColor = pbsdf.subsurfaceColor;
    s.subsurfaceRadius = pbsdf.subsurfaceRadius;
    s.subsurfaceScale = pbsdf.subsurfaceScale;
    s.subsurfaceAnisotropy = pbsdf.subsurfaceAnisotropy;
    s.subsurfaceIOR = pbsdf.subsurfaceIOR;
    s.useRandomWalkSSS = pbsdf.useRandomWalkSSS;
    s.sssMaxSteps = pbsdf.sssMaxSteps;

    s.clearcoat = pbsdf.clearcoat;
    s.clearcoatRoughness = pbsdf.clearcoatRoughness;
    s.translucent = pbsdf.translucent;
    s.anisotropic = pbsdf.anisotropic;
    s.normalStrength = pbsdf.get_normal_strength();
    s.sheen = pbsdf.sheen;
    s.sheenTint = pbsdf.sheen_tint;

    s.microDetailStrength = pbsdf.micro_detail_strength;
    s.microDetailScale = pbsdf.micro_detail_scale;
    s.tileBreakStrength = pbsdf.tile_break_strength;

    s.uvScale = pbsdf.textureTransform.scale;
    s.uvOffset = pbsdf.textureTransform.translation;
    s.uvTiling = pbsdf.textureTransform.tilingFactor;
    s.uvRotationDegrees = pbsdf.textureTransform.rotation_degrees;
    s.uvWrapMode = static_cast<uint32_t>(pbsdf.textureTransform.wrapMode);
    return s;
}

inline void applyPBRMaterialSnapshotToGpuMaterial(const PBRMaterialSnapshot& s, GpuMaterial& gpu) {
    gpu.albedo = make_float3((float)s.albedo.x, (float)s.albedo.y, (float)s.albedo.z);
    gpu.roughness = s.roughness;
    gpu.metallic = s.metallic;
    gpu.specular = s.specular;
    gpu.emission = make_float3(
        (float)(s.emission.x * s.emissionStrength),
        (float)(s.emission.y * s.emissionStrength),
        (float)(s.emission.z * s.emissionStrength));
    gpu.ior = s.ior;
    gpu.transmission = s.transmission;
    gpu.opacity = s.opacity;

    gpu.subsurface = s.subsurface;
    gpu.subsurface_color = make_float3(
        (float)s.subsurfaceColor.x,
        (float)s.subsurfaceColor.y,
        (float)s.subsurfaceColor.z);
    gpu.subsurface_radius = make_float3(
        (float)s.subsurfaceRadius.x,
        (float)s.subsurfaceRadius.y,
        (float)s.subsurfaceRadius.z);
    gpu.subsurface_scale = s.subsurfaceScale;
    gpu.subsurface_anisotropy = s.subsurfaceAnisotropy;
    gpu.subsurface_ior = s.subsurfaceIOR;
    gpu.sss_use_random_walk = s.useRandomWalkSSS ? 1 : 0;
    gpu.sss_max_steps = s.sssMaxSteps;

    gpu.clearcoat = s.clearcoat;
    gpu.clearcoat_roughness = s.clearcoatRoughness;
    gpu.translucent = s.translucent;
    gpu.anisotropic = s.anisotropic;
    gpu.normal_strength = s.normalStrength;
    gpu.sheen = s.sheen;
    gpu.sheen_tint = s.sheenTint;

    gpu.micro_detail_strength = s.microDetailStrength;
    gpu.micro_detail_scale = s.microDetailScale;
    gpu.tile_break_strength = s.tileBreakStrength;

    gpu.uv_scale_x = (float)s.uvScale.u;
    gpu.uv_scale_y = (float)s.uvScale.v;
    gpu.uv_offset_x = (float)s.uvOffset.u;
    gpu.uv_offset_y = (float)s.uvOffset.v;
    gpu.uv_rotation_degrees = s.uvRotationDegrees;
    gpu.uv_tiling_x = (float)s.uvTiling.u;
    gpu.uv_tiling_y = (float)s.uvTiling.v;
    gpu.uv_wrap_mode = (int)s.uvWrapMode;
}

inline Backend::IBackend::MaterialData makeBackendMaterialDataFromSnapshot(const PBRMaterialSnapshot& s) {
    Backend::IBackend::MaterialData data;
    data.albedo = s.albedo;
    data.roughness = s.roughness;
    data.metallic = s.metallic;
    data.specular = s.specular;
    data.emission = s.emission;
    data.emissionStrength = s.emissionStrength;
    data.ior = s.ior;
    data.transmission = s.transmission;
    data.opacity = s.opacity;

    data.subsurface = s.subsurface;
    data.subsurfaceColor = s.subsurfaceColor;
    data.subsurfaceRadius = s.subsurfaceRadius;
    data.subsurfaceScale = s.subsurfaceScale;
    data.subsurfaceAnisotropy = s.subsurfaceAnisotropy;
    data.subsurfaceIOR = s.subsurfaceIOR;
    data.useRandomWalkSSS = s.useRandomWalkSSS;
    data.sssMaxSteps = s.sssMaxSteps;

    data.clearcoat = s.clearcoat;
    data.clearcoatRoughness = s.clearcoatRoughness;
    data.translucent = s.translucent;
    data.anisotropic = s.anisotropic;
    data.normalStrength = s.normalStrength;
    data.sheen = s.sheen;
    data.sheenTint = s.sheenTint;

    data.micro_detail_strength = s.microDetailStrength;
    data.micro_detail_scale = s.microDetailScale;
    data.tile_break_strength = s.tileBreakStrength;

    data.uvScale = s.uvScale;
    data.uvOffset = s.uvOffset;
    data.uvTiling = s.uvTiling;
    data.uvRotationDegrees = s.uvRotationDegrees;
    data.uvWrapMode = s.uvWrapMode;
    return data;
}
