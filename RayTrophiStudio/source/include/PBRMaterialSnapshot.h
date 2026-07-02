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

    // Thin-shell bubble (champagne / soda / soap close-up)
    bool  isBubble   = false;
    float bubbleIor  = 1.33f;
    float bubbleFilm = 0.0f;

    // Iridescent clearcoat (thin-film tint on the clearcoat lobe)
    float clearcoatIridescence = 0.0f;
    float clearcoatFilmThickness = 0.55f;

    // Transmission interior absorption density (thick resin / glass-marble depth)
    float transmissionDensity = 0.0f;
    Vec3  resinColor = Vec3(1.0f, 1.0f, 1.0f); // resin absorption tint (separate from albedo)
    float resinRoughness = 0.1f;               // resin coat gloss (reflect lobe), independent of base
    float resinInclusion = 0.0f;               // dust cloudiness (heterogeneous absorption)
    float resinDirt = 0.0f;                    // opaque dirt-speck amount (early-return)
    float resinInclusionScale = 8.0f;          // procedural feature size
    Vec3  resinDirtColor = Vec3(0.18f, 0.14f, 0.10f);
    bool  glassMarbleVolume = false;           // full-volume marble medium march (vs front-shell)
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

    s.subsurface = pbsdf.getSubsurface();
    s.subsurfaceColor = pbsdf.getSubsurfaceColor();
    s.subsurfaceRadius = pbsdf.getSubsurfaceRadius();
    s.subsurfaceScale = pbsdf.getSubsurfaceScale();
    s.subsurfaceAnisotropy = pbsdf.getSubsurfaceAnisotropy();
    s.subsurfaceIOR = pbsdf.getSubsurfaceIOR();
    s.useRandomWalkSSS = pbsdf.getUseRandomWalkSSS();
    s.sssMaxSteps = pbsdf.getSssMaxSteps();

    s.clearcoat = pbsdf.getClearcoat();
    s.clearcoatRoughness = pbsdf.getClearcoatRoughness();
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

    s.isBubble   = pbsdf.getIsBubble();
    s.bubbleIor  = pbsdf.getBubbleIor();
    s.bubbleFilm = pbsdf.getBubbleFilm();
    s.clearcoatIridescence = pbsdf.getClearcoatIridescence();
    s.clearcoatFilmThickness = pbsdf.getClearcoatFilmThickness();
    s.transmissionDensity = pbsdf.getTransmissionDensity();
    s.resinColor = pbsdf.getResinColor();
    s.resinRoughness = pbsdf.getResinRoughness();
    s.resinInclusion = pbsdf.getResinInclusion();
    s.resinDirt = pbsdf.getResinDirt();
    s.resinInclusionScale = pbsdf.getResinInclusionScale();
    s.resinDirtColor = pbsdf.getResinDirtColor();
    s.glassMarbleVolume = pbsdf.getGlassMarbleVolume();
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

    // Thin-shell bubble: toggle ONLY the bubble bit (leave water/terrain intact).
    if (s.isBubble) gpu.flags |= GPU_MAT_FLAG_BUBBLE;
    else            gpu.flags &= ~GPU_MAT_FLAG_BUBBLE;
    gpu.bubble_ior  = s.bubbleIor;
    gpu.bubble_film = s.bubbleFilm;
    gpu.clearcoat_iridescence = s.clearcoatIridescence;
    gpu.clearcoat_film_thickness = s.clearcoatFilmThickness;
    gpu.transmission_density = s.transmissionDensity;
    gpu.resin_color = make_float3((float)s.resinColor.x, (float)s.resinColor.y, (float)s.resinColor.z);
    gpu.resin_roughness = s.resinRoughness;
    gpu.resin_inclusion = s.resinInclusion;
    gpu.resin_dirt = s.resinDirt;
    gpu.resin_inclusion_scale = s.resinInclusionScale;
    gpu.resin_dirt_color = make_float3((float)s.resinDirtColor.x, (float)s.resinDirtColor.y, (float)s.resinDirtColor.z);
    if (s.glassMarbleVolume) gpu.flags |= GPU_MAT_FLAG_MARBLE_VOLUME;
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

    data.is_bubble   = s.isBubble;
    data.bubble_ior  = s.bubbleIor;
    data.bubble_film = s.bubbleFilm;
    data.clearcoat_iridescence = s.clearcoatIridescence;
    data.clearcoat_film_thickness = s.clearcoatFilmThickness;
    data.transmission_density = s.transmissionDensity;
    data.resin_color = s.resinColor;
    data.resin_roughness = s.resinRoughness;
    data.resin_inclusion = s.resinInclusion;
    data.resin_dirt = s.resinDirt;
    data.resin_inclusion_scale = s.resinInclusionScale;
    data.resin_dirt_color = s.resinDirtColor;
    data.glass_marble_volume = s.glassMarbleVolume;
    return data;
}
