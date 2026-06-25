/*
 * RayTrophi Studio — Vulkan Closest Hit Shader
 * Principled BSDF Material Scatter
 *
 * Desteklenen materyaller:
 *   - Lambertian Diffuse (cosine-weighted hemisphere sampling)
 *   - GGX Metallic Reflection (importance-sampled)
 *   - Dielectric Glass (Fresnel + TIR)
 *   - Principled Blend (diffuse ↔ metal geçiş)
 *   - Emissive
 *
 * Değişiklikler (v2):
 *   - randomInUnitSphere() → cosine-weighted hemisphere (daha hızlı, doğru PDF)
 *   - Emission payload'dan ayrıldı (scatter ile çakışma giderildi)
 *   - Metallic blend attenuation PDF düzeltildi
 *   - Glass offset: yüzey normaline göre (direction değil)
 *   - GGX NDF ile metallic roughness importance sampling eklendi
 *   - ONB (Orthonormal Basis) yardımcı fonksiyonları
 */

#version 460
#extension GL_EXT_ray_tracing                          : require
#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_EXT_nonuniform_qualifier                 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#include "pbr_texture_policy.glsl"

// Push Constants — must match C++ CameraPushConstants
layout(push_constant) uniform CameraPC {
    vec4  origin;
    vec4  lowerLeft;
    vec4  horizontal;
    vec4  vertical;
    uint  frameCount;
    uint  minSamples;
    uint  lightCount;
    float varianceThreshold;
    uint  maxSamples;
    float exposure_factor;

    // Extended Pro Features (must match CameraPushConstants in VulkanBackend.cpp)
    float aperture;
    float focusDistance;
    float distortion;
    uint  bladeCount;

    uint  caEnabled;
    float caAmount;
    float caRScale;
    float caBScale;

    uint  vignetteEnabled;
    float vignetteAmount;
    float vignetteFalloff;
    float pad0;

    uint  shakeEnabled;
    float shakeOffsetX;
    float shakeOffsetY;
    float shakeOffsetZ;

    float shakeRotX;
    float shakeRotY;
    float shakeRotZ;
    float waterTime;   // Wall-clock seconds for water animation
} cam;

// ============================================================
// Sabitler
// ============================================================
const float PI          = 3.14159265358979323846;
const float TWO_PI      = 6.28318530717958647692;
const float INV_PI      = 0.31830988618379067154;
const float EPSILON     = 1e-4;
const float RAY_OFFSET  = 1e-3;   // Yüzey offset (self-intersection önleme)
const float SHADOW_TMIN = 1e-3;   // Shadow rays: avoid near-field self/adjacent contact acne
const float OPACITY_THRESHOLD = 0.5;  // Alpha cutout threshold
const uint MAT_FLAG_WATER = (1u << 17);
const uint MAT_FLAG_WATER_FFT_READY = (1u << 18);
const uint MAT_FLAG_BUBBLE = (1u << 19);
const uint MAT_FLAG_MARBLE_VOLUME = (1u << 20); // glass marble full-volume medium march (raygen integrates interior)

// ============================================================
// Payload — raygen shader ile eşleşmeli
// ============================================================
struct RayPayload {
    vec3     radiance;
    vec3     attenuation;
    vec3     scatterOrigin;
    vec3     scatterDir;
    uint     seed;
    bool     scattered;
    bool     hitEmissive;
    uint     occluded;
    bool     skipAABBs;    // set by volume_closesthit when a solid surface is found inside
    vec3     primaryAlbedo;
    vec3     primaryNormal;
    uint     primaryHit;
    float    primaryTransmission;
    float    primaryMetallic;
    uint     bounceType;
    uint     primaryMaterialId;   // Stylize AOV: real material index of the primary hit
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
// Separate shadow payload storage to avoid corrupting the main payload during shadow tracing
// Use rayPayloadInEXT here to match any-hit/miss declarations (avoid ABI mismatch)
layout(location = 1) rayPayloadEXT bool shadowOccluded;

// ============================================================
// Descriptor Bindings
// ============================================================
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

// Material struct — single source of truth shared by every material-reading shader.
#include "material_struct.glsl"

float wrapRepeat(float x) {
    float r = mod(x, 1.0);
    return (r < 0.0) ? (r + 1.0) : r;
}

float wrapMirror(float x) {
    float r = mod(x, 2.0);
    if (r < 0.0) r += 2.0;
    return (r > 1.0) ? (2.0 - r) : r;
}

vec2 applyMaterialUVTransform(Material mat, vec2 originalUV) {
    vec2 uv = originalUV - vec2(0.5);
    uv.x *= mat.uv_scale_x;
    uv.y *= mat.uv_scale_y;

    float angleRad = radians(mat.uv_rotation_degrees);
    float c = cos(angleRad);
    float s = sin(angleRad);
    uv = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);

    uv += vec2(0.5);
    uv += vec2(mat.uv_offset_x, mat.uv_offset_y);
    uv *= vec2(mat.uv_tiling_x, mat.uv_tiling_y);

    switch (mat.uv_wrap_mode) {
    case 0u:
        uv = vec2(wrapRepeat(uv.x), wrapRepeat(uv.y));
        break;
    case 1u:
        uv = vec2(wrapMirror(uv.x), wrapMirror(uv.y));
        break;
    case 2u:
        uv = clamp(uv, vec2(0.0), vec2(1.0));
        break;
    case 3u:
        uv = originalUV;
        break;
    case 4u: {
        vec2 scaled = uv * 3.0;
        int face = int(scaled.x) + 3 * int(scaled.y);
        vec2 local = mod(scaled, 1.0);
        if (local.x < 0.0) local.x += 1.0;
        if (local.y < 0.0) local.y += 1.0;
        switch (face % 6) {
        case 0: uv = local; break;
        case 1: uv = vec2(local.y, 1.0 - local.x); break;
        case 2: uv = vec2(1.0 - local.x, local.y); break;
        case 3: uv = vec2(1.0 - local.y, 1.0 - local.x); break;
        case 4: uv = vec2(local.x, 1.0 - local.y); break;
        default: uv = local.yx; break;
        }
        break;
    }
    default:
        break;
    }

    return uv;
}

struct LightData {
    vec4 position;    // xyz + type (0=point, 1=dir, 2=area, 3=spot)
    vec4 color;       // rgb + intensity
    vec4 params;      // radius, width, height, inner_angle
    vec4 direction;   // xyz + outer_angle
    vec4 area_u;      // xyz: AreaLight u-axis (unit)
    vec4 area_v;      // xyz: AreaLight v-axis (unit)
};

struct VkGeometryData {
    uint64_t vertexAddr;
    uint64_t normalAddr;
    uint64_t uvAddr;
    uint64_t indexAddr;
    uint64_t materialAddr;
};

struct VkInstanceData {
    uint materialIndex;
    uint blasIndex;
};

layout(set = 0, binding = 2, scalar) readonly buffer MaterialBuffer  { Material     m[]; } materials;
layout(set = 0, binding = 3, scalar) readonly buffer LightBuffer     { LightData    l[]; } lights;
layout(set = 0, binding = 4, scalar) readonly buffer GeometryBuffer  { VkGeometryData g[]; } geometries;
layout(set = 0, binding = 5, scalar) readonly buffer InstanceBuffer  { VkInstanceData  i[]; } instances;

// Array of combined image samplers for uploaded textures
layout(set = 0, binding = 6) uniform sampler2D materialTextures[];

// ════════════════════════════════════════════════════════════════════════════════
// EXTENDED WORLD DATA — Full Nishita Sky Model + Atmosphere LUT
// ════════════════════════════════════════════════════════════════════════════════
struct VkWorldDataExtended {
    // ════════════════════════════ CORE MODE & SUN TINT (32 bytes)
    vec3  sunDir;
    int   mode;
    vec3  sunColor;
    float sunIntensity;
    
    // ════════════════════════════ NISHITA SUN PARAMETERS (32 bytes)
    float sunSize;
    float mieAnisotropy;
    float rayleighDensity;
    float mieDensity;
    float humidity;
    float temperature;
    float ozoneAbsorptionScale;
    float atmosphereIntensity;
    
    // ════════════════════════════ ATMOSPHERE DENSITY (32 bytes)
    float airDensity;
    float dustDensity;
    float ozoneDensity;
    float altitude;
    float planetRadius;
    float atmosphereHeight;
    int   multiScatterEnabled;
    float multiScatterFactor;
    
    // ════════════════════════════ CLOUD LAYER 1 PARAMETERS (64 bytes)
    int   cloudsEnabled;
    float cloudCoverage;
    float cloudDensity;
    float cloudScale;
    float cloudHeightMin;
    float cloudHeightMax;
    float cloudOffsetX;
    float cloudOffsetZ;
    float cloudQuality;
    float cloudDetail;
    int   cloudBaseSteps;
    int   cloudLightSteps;
    float cloudShadowStrength;
    float cloudAmbientStrength;
    float cloudSilverIntensity;
    float cloudAbsorption;
    
    // ════════════════════════════ ADVANCED CLOUD SCATTERING (32 bytes)
    float cloudAnisotropy;
    float cloudAnisotropyBack;
    float cloudLobeMix;
    float cloudEmissiveIntensity;
    vec3  cloudEmissiveColor;
    float _pad3;
    
    // ════════════════════════════ FOG PARAMETERS (32 bytes)
    int   fogEnabled;
    float fogDensity;
    float fogHeight;
    float fogFalloff;
    float fogDistance;
    float fogSunScatter;
    vec3  fogColor;
    float _pad4;
    
    // ════════════════════════════ GOD RAYS (16 bytes)
    int   godRaysEnabled;
    float godRaysIntensity;
    float godRaysDensity;
    int   godRaysSamples;
    
    // ════════════════════════════ ENVIRONMENT & LUT REFS (32 bytes)
    int   aerialEnabled;
    float aerialMinDistance;
    float aerialMaxDistance;
    float aerialDensity;

    int   weatherEnabled;
    int   weatherType;
    float weatherIntensity;
    float weatherDensity;
    vec3  weatherWindDirection;
    float weatherWindSpeed;
    float weatherPrecipitationScale;
    float weatherVisibility;
    float weatherSurfaceWetness;
    float weatherSurfaceAccumulation;
    float weatherSurfaceSettling;
    float weatherSurfaceHeight;
    int   weatherVisualMode;
    int   weatherSurfaceResponseEnabled;

    int   envTexSlot;
    float envIntensity;
    float envRotation;
    int   _pad5;                 // nishitaLutReady: Vulkan binding 8 has valid LUT samplers
    int   envOverlayEnabled;
    int   envOverlayBlendMode;
    float envOverlayIntensity;
    float envOverlayRotation;
    uvec2 transmittanceLUT;      // 64-bit handle as uvec2
    uvec2 skyviewLUT;            // 64-bit handle as uvec2
    uvec2 multiScatterLUT;       // 64-bit handle as uvec2
    uvec2 aerialPerspectiveLUT;  // 64-bit handle as uvec2
};

layout(set = 0, binding = 7, scalar) readonly buffer WorldBuffer     { VkWorldDataExtended w; } worldData;
// Atmosphere LUT samplers: [0]=transmittance, [1]=skyview, [2]=multi_scatter, [3]=aerial_perspective
layout(set = 0, binding = 8) uniform sampler2D atmosphereLUTs[4];

// ════════════════════════════════════════════════════════════════════════════════
// Binding 9: Volume Instances SSBO (OptiX-compatible volumetric data)
// ════════════════════════════════════════════════════════════════════════════════
struct VkVolumeInstance {
    float transform[12];
    vec3  aabb_min;
    vec3  aabb_max;
    float density_multiplier;
    float density_remap_low;
    float density_remap_high;
    float noise_scale;
    vec3  scatter_color;
    float scatter_coefficient;
    float scatter_anisotropy;
    float scatter_anisotropy_back;
    float scatter_lobe_mix;
    float scatter_multi;
    vec3  absorption_color;
    float absorption_coefficient;
    vec3  emission_color;
    float emission_intensity;
    float step_size;
    int   max_steps;
    int   shadow_steps;
    float shadow_strength;
    int   volume_type;
    int   is_active;
    float voxel_size;
    int   _pad0;
    float    inv_transform[12];
    uint64_t vdb_grid_address;   // NanoVDB grid device address (or 0)
    uint64_t vdb_temp_address;   // secondary grid (temperature etc.)
    float    _reserved[2];       // padding to complete 24 bytes
    int   emission_mode;
    float temperature_scale;
    float blackbody_intensity;
    float max_temperature;
    int   color_ramp_enabled;
    int   ramp_stop_count;
    int   _ramp_pad[2];
    float ramp_positions[8];
    float ramp_colors_r[8];
    float ramp_colors_g[8];
    float ramp_colors_b[8];
    float pivot_offset[3];
    int   source_type;
    float cloud_coverage;
    float cloud_detail;
    float cloud_erosion;
    float cloud_base_scale;
    float cloud_edge_fade;
    float cloud_offset_x;
    float cloud_offset_z;
    float cloud_seed;
    float _ext_reserved[12];
};

layout(set = 0, binding = 9, scalar) readonly buffer VolumeBuffer { VkVolumeInstance v[]; } volumes;

// ════════════════════════════════════════════════════════════════════════════════
// Volumetric shadow transmittance — surface shader side
// Computes soft light attenuation through VDB/gas volume AABBs.
// cam.pad0 stores float(volumeCount) set each frame by C++ renderProgressive.
// ════════════════════════════════════════════════════════════════════════════════

// ── NanoVDB sampler (PNanoVDB GLSL port) ─────────────────────────────────────
#define PNANOVDB_GLSL
#define PNANOVDB_BUF_CUSTOM
struct pnanovdb_buf_t { uint64_t address; };
layout(buffer_reference, std430, buffer_reference_align=4) buffer NanoVDBBlockSurf { uint data[]; };
uint  pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byte_offset) {
    NanoVDBBlockSurf blk = NanoVDBBlockSurf(buf.address);
    return blk.data[byte_offset >> 2];
}
uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byte_offset) {
    NanoVDBBlockSurf blk = NanoVDBBlockSurf(buf.address);
    uint idx = byte_offset >> 2;
    return uvec2(blk.data[idx], blk.data[idx + 1]);
}
void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint byte_offset, uint value)  {}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint byte_offset, uvec2 value) {}
#include "PNanoVDB.h"
#include "procedural_detail.glsl"

// Trilinear NanoVDB float grid sampler — inputs in VDB native world-space
float ch_sampleNanoVDB(uint64_t gridAddr, vec3 worldPos) {
    if (gridAddr == 0u) return 0.0;
    pnanovdb_buf_t buf; buf.address = gridAddr;
    pnanovdb_grid_handle_t gridH; gridH.address.byte_offset = 0u;
    pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(buf, gridH);
    pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(buf, treeH);
    pnanovdb_map_handle_t   mapH = pnanovdb_grid_get_map(buf, gridH);
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, rootH);
    pnanovdb_vec3_t wPos; wPos.x = worldPos.x; wPos.y = worldPos.y; wPos.z = worldPos.z;
    pnanovdb_vec3_t iPos = pnanovdb_map_apply_inverse(buf, mapH, wPos);
    vec3 p0 = floor(vec3(iPos.x, iPos.y, iPos.z) - 0.5);
    vec3 fr = fract(vec3(iPos.x, iPos.y, iPos.z) - 0.5);
    float d[8];
    for (int i = 0; i < 8; ++i) {
        pnanovdb_coord_t c;
        c.x = int(p0.x) + ((i & 1) != 0 ? 1 : 0);
        c.y = int(p0.y) + ((i & 2) != 0 ? 1 : 0);
        c.z = int(p0.z) + ((i & 4) != 0 ? 1 : 0);
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, acc, c);
        d[i] = pnanovdb_read_float(buf, addr);
    }
    float dx00 = mix(d[0],d[1],fr.x); float dx10 = mix(d[2],d[3],fr.x);
    float dx01 = mix(d[4],d[5],fr.x); float dx11 = mix(d[6],d[7],fr.x);
    return mix(mix(dx00,dx10,fr.y), mix(dx01,dx11,fr.y), fr.z);
}
// ─────────────────────────────────────────────────────────────────────────────

// Procedural noise (type-1 volumes)
float ch_hash3D(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

vec3 ch_hash3Gradient(vec3 p) {
    p = vec3(
        dot(p, vec3(127.1, 311.7, 74.7)),
        dot(p, vec3(269.5, 183.3, 246.1)),
        dot(p, vec3(113.5, 271.9, 124.6))
    );
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453);
}

float ch_noise3D(vec3 p) {
    vec3 i = floor(p); vec3 f = fract(p);
    vec3 u = f*f*f*(f*(f*6.0-vec3(15.0))+vec3(10.0));
    float n000=dot(ch_hash3Gradient(i+vec3(0,0,0)), f-vec3(0,0,0)); float n100=dot(ch_hash3Gradient(i+vec3(1,0,0)), f-vec3(1,0,0));
    float n010=dot(ch_hash3Gradient(i+vec3(0,1,0)), f-vec3(0,1,0)); float n110=dot(ch_hash3Gradient(i+vec3(1,1,0)), f-vec3(1,1,0));
    float n001=dot(ch_hash3Gradient(i+vec3(0,0,1)), f-vec3(0,0,1)); float n101=dot(ch_hash3Gradient(i+vec3(1,0,1)), f-vec3(1,0,1));
    float n011=dot(ch_hash3Gradient(i+vec3(0,1,1)), f-vec3(0,1,1)); float n111=dot(ch_hash3Gradient(i+vec3(1,1,1)), f-vec3(1,1,1));
    return mix(mix(mix(n000,n100,u.x),mix(n010,n110,u.x),u.y),
               mix(mix(n001,n101,u.x),mix(n011,n111,u.x),u.y),u.z) * 0.5 + 0.5;
}
float ch_fbmNoise(vec3 p, int oct) {
    float v=0.0,a=0.5,fr=1.0;
    for(int i=0;i<oct;i++){v+=a*ch_noise3D(p*fr);fr*=2.0;a*=0.5;}
    return v;
}

float ch_proceduralCloudDensity(VkVolumeInstance vol, vec3 lp, vec3 bmin, vec3 bmax) {
    vec3 span = max(bmax - bmin, vec3(1e-5));
    vec3 norm = clamp((lp - bmin) / span, vec3(0.0), vec3(1.0));
    float baseScale = max(vol.cloud_base_scale, 1.0);
    vec3 cloudCoord = vec3(
        norm.x * baseScale + vol.cloud_offset_x,
        norm.y * 1.35,
        norm.z * baseScale + vol.cloud_offset_z);
    cloudCoord += vec3(vol.cloud_seed * 0.137, vol.cloud_seed * 0.317, vol.cloud_seed * 0.719);

    float coverage = clamp(vol.cloud_coverage, 0.0, 1.0);
    float detail = clamp(vol.cloud_detail, 0.0, 1.0);
    float erosion = clamp(vol.cloud_erosion, 0.0, 1.0);
    float warpX = ch_fbmNoise(vec3(cloudCoord.x * 0.38, cloudCoord.y * 0.16, cloudCoord.z * 0.38) + vec3(11.0, 0.0, 7.0), 2) - 0.5;
    float warpZ = ch_fbmNoise(vec3(cloudCoord.x * 0.38, cloudCoord.y * 0.16, cloudCoord.z * 0.38) + vec3(41.0, 3.0, 23.0), 2) - 0.5;
    vec3 warped = cloudCoord + vec3(warpX * 1.35, 0.0, warpZ * 1.35);

    float base = ch_fbmNoise(vec3(warped.x * 0.52, warped.y * 0.28, warped.z * 0.52), 4);
    float billow = 1.0 - abs(ch_fbmNoise(vec3(warped.x * 1.15, warped.y * 0.5, warped.z * 1.15) + vec3(17.0, 3.0, 11.0), 4) * 2.0 - 1.0);
    float detailNoise = ch_fbmNoise(warped * mix(2.8, 7.0, detail) + vec3(31.0, 7.0, 19.0), 2);

    float puffy = smoothstep(0.32, 0.88, billow);
    float shape = mix(base, base * 0.45 + puffy * 0.75, 0.72);
    shape -= detailNoise * mix(0.06, 0.28, erosion);

    float threshold = mix(0.78, 0.30, coverage);
    float density = max((shape - threshold) / max(1.0 - threshold, 1e-4), 0.0);

    float bottom = smoothstep(0.12, 0.42, norm.y);
    float top = 1.0 - smoothstep(0.72, 1.02, norm.y);
    vec3 ed = vec3(0.5) - abs(norm - vec3(0.5));
    float edge = smoothstep(0.0, max(vol.cloud_edge_fade, 0.02), min(ed.x, ed.z));
    return density * density * bottom * top * edge * 4.6;
}

// World-pos → object-space density for shadow ray march.
// type 0 (homogeneous): density=1.0,  type 1 (noise): fbm density,
// type 2 (NanoVDB): real trilinear grid sample via ch_sampleNanoVDB.
float ch_volDensity(VkVolumeInstance vol, vec3 wp) {
    vec3 lp;
    lp.x = vol.inv_transform[0]*wp.x + vol.inv_transform[1]*wp.y + vol.inv_transform[2]*wp.z + vol.inv_transform[3];
    lp.y = vol.inv_transform[4]*wp.x + vol.inv_transform[5]*wp.y + vol.inv_transform[6]*wp.z + vol.inv_transform[7];
    lp.z = vol.inv_transform[8]*wp.x + vol.inv_transform[9]*wp.y + vol.inv_transform[10]*wp.z + vol.inv_transform[11];
    vec3 bmin = vec3(vol.aabb_min[0], vol.aabb_min[1], vol.aabb_min[2]);
    vec3 bmax = vec3(vol.aabb_max[0], vol.aabb_max[1], vol.aabb_max[2]);
    if (any(lessThan(lp, bmin)) || any(greaterThan(lp, bmax))) return 0.0;
    float density = 1.0;
    if (vol.volume_type == 1) {
        vec3 span = max(bmax - bmin, vec3(1e-5));
        vec3 norm = (lp - bmin) / span;
        vec3 nc = norm * vol.noise_scale;
        density = ch_fbmNoise(nc, 4);
        vec3 ed = vec3(0.5) - abs(norm - vec3(0.5));
        density *= smoothstep(0.0, 0.1, min(min(ed.x, ed.y), ed.z));
    } else if (vol.volume_type == 2) {
        // NanoVDB: sample the actual grid data.
        // Guard: vdb_grid_address may be 0 when the source VDB file is missing or
        // the Vulkan buffer has not yet been uploaded (e.g. project opened without
        // the original .vdb file present).  Dereferencing address 0 = GPU crash.
        if (vol.vdb_grid_address != 0) {
            vec3 vdbWorldPos = lp;
            vdbWorldPos.x -= vol.pivot_offset[0];
            vdbWorldPos.y -= vol.pivot_offset[1];
            vdbWorldPos.z -= vol.pivot_offset[2];
            density = ch_sampleNanoVDB(vol.vdb_grid_address, vdbWorldPos);
        } else {
            // Fallback: procedural noise so the volume still renders visibly
            vec3 span = max(bmax - bmin, vec3(1e-5));
            vec3 norm = (lp - bmin) / span;
            vec3 nc = norm * max(vol.noise_scale, 1.0);
            density = ch_fbmNoise(nc, 4);
        }
    } else if (vol.volume_type == 3 || vol.source_type == 3) {
        density = ch_proceduralCloudDensity(vol, lp, bmin, bmax);
    }
    density = max((density - vol.density_remap_low) /
                  max(vol.density_remap_high - vol.density_remap_low, 1e-6), 0.0);
    return max(density * vol.density_multiplier, 0.0);
}

// Ray-march all active volumes between shadowOrigin and light (maxDist).
// Returns transmittance in [0,1]: 1.0 = fully lit, 0.0 = fully shadowed.
float computeVolumeShadowTransmittance(vec3 shadowOrigin, vec3 lightDir, float maxDist) {
    int volCount = int(cam.pad0);
    if (volCount <= 0) return 1.0;
    if (maxDist <= 1e-4) return 1.0;
    const float EPS = 1e-6;
    float transmittance = 1.0;
    for (int vi = 0; vi < min(volCount, 16); vi++) {
        VkVolumeInstance vol = volumes.v[vi];
        if (vol.is_active == 0) continue;
        if (vol.volume_type == 3 || vol.source_type == 3) continue;
        float sigma_t = vol.scatter_coefficient + vol.absorption_coefficient;
        if (sigma_t < EPS || vol.density_multiplier < EPS) continue;

        vec3 lo, ld;
        lo.x = vol.inv_transform[0] * shadowOrigin.x + vol.inv_transform[1] * shadowOrigin.y + vol.inv_transform[2] * shadowOrigin.z + vol.inv_transform[3];
        lo.y = vol.inv_transform[4] * shadowOrigin.x + vol.inv_transform[5] * shadowOrigin.y + vol.inv_transform[6] * shadowOrigin.z + vol.inv_transform[7];
        lo.z = vol.inv_transform[8] * shadowOrigin.x + vol.inv_transform[9] * shadowOrigin.y + vol.inv_transform[10] * shadowOrigin.z + vol.inv_transform[11];
        ld.x = vol.inv_transform[0] * lightDir.x + vol.inv_transform[1] * lightDir.y + vol.inv_transform[2] * lightDir.z;
        ld.y = vol.inv_transform[4] * lightDir.x + vol.inv_transform[5] * lightDir.y + vol.inv_transform[6] * lightDir.z;
        ld.z = vol.inv_transform[8] * lightDir.x + vol.inv_transform[9] * lightDir.y + vol.inv_transform[10] * lightDir.z;

        vec3 inv;
        inv.x = abs(ld.x) > EPS ? 1.0 / ld.x : (ld.x >= 0.0 ? 1e7 : -1e7);
        inv.y = abs(ld.y) > EPS ? 1.0 / ld.y : (ld.y >= 0.0 ? 1e7 : -1e7);
        inv.z = abs(ld.z) > EPS ? 1.0 / ld.z : (ld.z >= 0.0 ? 1e7 : -1e7);

        vec3 bmin = vec3(vol.aabb_min[0], vol.aabb_min[1], vol.aabb_min[2]);
        vec3 bmax = vec3(vol.aabb_max[0], vol.aabb_max[1], vol.aabb_max[2]);
        vec3 t0 = (bmin - lo) * inv;
        vec3 t1 = (bmax - lo) * inv;
        vec3 tS = min(t0, t1);
        vec3 tL = max(t0, t1);
        float tNL = max(max(tS.x, tS.y), max(tS.z, 0.0));
        float tFL = min(min(tL.x, tL.y), tL.z);
        if (tNL >= tFL) continue;

        float tNW = max(tNL, 0.001);
        float tFW = min(tFL, maxDist);
        if (tFW <= tNW) continue;

        int reqSteps = clamp(vol.shadow_steps, 1, 64);
        float segLen = tFW - tNW;
        if (segLen <= 1e-5) continue;
        float dMid = ch_volDensity(vol, shadowOrigin + lightDir * (tNW + 0.5 * segLen));
        float tauHint = max(0.0, dMid) * sigma_t * segLen;
        if (tauHint <= 0.02) continue;
        float stepScale = clamp(sqrt(tauHint), 0.25, 1.0);
        int steps = int(ceil(float(reqSteps) * stepScale));
        steps = clamp(steps, 3, min(reqSteps, 16));
        float stepW = segLen / float(steps) ;
        stepW = max(stepW, 1e-5);
        float jitter = fract(sin(dot(shadowOrigin + lightDir * float(vi + 1), vec3(12.9898, 78.233, 37.719))) * 43758.5453);
        float opticalDepth = 0.0;
        for (int s = 0; s < steps; s++) {
            vec3 sp = shadowOrigin + lightDir * (tNW + (float(s) + jitter + 0.5) * stepW);
            float d = ch_volDensity(vol, sp);
            opticalDepth += d * sigma_t * stepW;
            if (opticalDepth > 10.0) break;
        }

        float beers = exp(-opticalDepth);
        float physTrans = beers;
        if (vol.scatter_multi > 0.0 && vol.scatter_coefficient > 0.0) {
            float albedoLum = dot(vol.scatter_color, vec3(0.2126, 0.7152, 0.0722));
            float beersSoft = exp(-opticalDepth * 0.25);
            physTrans = beers * (1.0 - vol.scatter_multi * albedoLum) + beersSoft * (vol.scatter_multi * albedoLum);
        }
        float strength = clamp(vol.shadow_strength, 0.0, 1.0);
        transmittance *= (1.0 - strength * (1.0 - physTrans));
        if (transmittance <= 0.0) break;
    }
    return transmittance;
}
// ════════════════════════════════════════════════════════════════════════════════
struct VkTerrainLayerData {
    uint  layer_mat_id[4];    // Material indices for layers 0-3
    float layer_uv_scale[4];  // UV tiling for each layer
    uint  splat_map_tex;       // Combined-image-sampler slot for RGBA splat map
    uint  layer_count;         // Active layer count (1-4)
    uint  _pad[2];
};
layout(set = 0, binding = 12, scalar) readonly buffer TerrainLayerBuffer { VkTerrainLayerData d[]; } terrainLayers;

// Buffer Device Address referansları
layout(buffer_reference, scalar) readonly buffer VertexBuffer { vec3 v[]; };
layout(buffer_reference, scalar) readonly buffer NormalBuffer { vec3 n[]; };
layout(buffer_reference, scalar) readonly buffer UVBuffer     { vec2 u[]; };
layout(buffer_reference, scalar) readonly buffer IndexBuffer  { uint i[]; };
layout(buffer_reference, scalar) readonly buffer MaterialIndexBuffer { uint m[]; };

// Hit attributes (barycentrics)
hitAttributeEXT vec2 baryCoord;

// ============================================================
// PCG Hash — hızlı, düşük korelasyonlu RNG
// ============================================================
uint pcgNext(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;

}


// [0, 1) aralığında float
float rnd(inout uint seed) {
    return float(pcgNext(seed)) * (1.0 / 4294967296.0);
}

// ============================================================
// ONB — Orthonormal Basis (Frisvad yöntemi, branch-free)
// Normal'e dik tangent/bitangent üret
// ============================================================
void buildONB(in vec3 n, out vec3 tangent, out vec3 bitangent) {
    float sign_ = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (sign_ + n.z);
    float b = n.x * n.y * a;
    tangent   = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
    bitangent = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

vec3 safeNormalize(vec3 v, vec3 fallback);

bool buildSurfaceTBN(
    vec3 objV0, vec3 objV1, vec3 objV2,
    vec2 uv0, vec2 uv1, vec2 uv2,
    vec3 shadingNormal,
    out vec3 tangent,
    out vec3 bitangent
) {
    vec2 dUV1 = uv1 - uv0;
    vec2 dUV2 = uv2 - uv0;
    float detUV = dUV1.x * dUV2.y - dUV2.x * dUV1.y;

    if (abs(detUV) <= 1e-8) {
        buildONB(shadingNormal, tangent, bitangent);
        return false;
    }

    vec3 worldV0 = vec3(gl_ObjectToWorldEXT * vec4(objV0, 1.0));
    vec3 worldV1 = vec3(gl_ObjectToWorldEXT * vec4(objV1, 1.0));
    vec3 worldV2 = vec3(gl_ObjectToWorldEXT * vec4(objV2, 1.0));
    vec3 worldEdge1 = worldV1 - worldV0;
    vec3 worldEdge2 = worldV2 - worldV0;

    float invDet = 1.0 / detUV;
    tangent = vec3(
        invDet * (dUV2.y * worldEdge1.x - dUV1.y * worldEdge2.x),
        invDet * (dUV2.y * worldEdge1.y - dUV1.y * worldEdge2.y),
        invDet * (dUV2.y * worldEdge1.z - dUV1.y * worldEdge2.z)
    );

    tangent = safeNormalize(tangent - shadingNormal * dot(shadingNormal, tangent), vec3(0.0));
    if (dot(tangent, tangent) <= 1e-8) {
        buildONB(shadingNormal, tangent, bitangent);
        return false;
    }

    float sigmaInst = (determinant(mat3(gl_ObjectToWorldEXT)) < 0.0) ? -1.0 : 1.0;
    float sigmaUV = (detUV < 0.0) ? -1.0 : 1.0;
    bitangent = safeNormalize(cross(shadingNormal, tangent), vec3(0.0));
    bitangent *= (sigmaUV * sigmaInst);
    if (dot(bitangent, bitangent) <= 1e-8) {
        buildONB(shadingNormal, tangent, bitangent);
        return false;
    }

    return true;
}

// Robust ray origin offset — Wächter & Binder, Ray Tracing Gems Ch. 6.
// Uses ULP-based integer offsetting: scales with the magnitude of p so it
// works correctly at any distance from the world origin. Unlike a fixed
// world-space epsilon this never under-offsets on thin/distant geometry
// or over-offsets on nearby geometry.
vec3 offset_ray(vec3 p, vec3 n) {
    const float origin      = 1.0 / 32.0;
    const float float_scale = 1.0 / 65536.0;
    const float int_scale   = 256.0;
    ivec3 of_i = ivec3(int_scale * n.x, int_scale * n.y, int_scale * n.z);
    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x) + (p.x < 0.0 ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y) + (p.y < 0.0 ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z) + (p.z < 0.0 ? -of_i.z : of_i.z)));
    return vec3(
        abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

vec3 safeNormalize(vec3 v, vec3 fallback) {
    float len2 = dot(v, v);
    bool invalid = isnan(v.x) || isnan(v.y) || isnan(v.z)
                || isinf(v.x) || isinf(v.y) || isinf(v.z);
    if (len2 <= 1e-20 || invalid) return fallback;
    return v * inversesqrt(len2);
}

// -----------------------------
// Unified light sampling (GLSL port - simplified parity)
// -----------------------------

float gc_luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float power_heuristic(float a, float b) {
    float a2 = a * a;
    float b2 = b * b;
    return a2 / (a2 + b2 + 1e-4);
}

// Spot falloff
float spot_light_falloff_gl(const LightData light, vec3 wi) {
    float cos_theta = dot(-wi, normalize(light.direction.xyz));
    float inner = light.params.z; // inner angle stored in params[2] in some paths
    float outer = light.direction.w; // outer angle in direction.w
    if (cos_theta < outer) return 0.0;
    if (cos_theta > inner) return 1.0;
    float t = (cos_theta - outer) / (inner - outer + 1e-6);
    return t * t;
}

// Compute simple light PDF (approx)
float compute_light_pdf_gl(const LightData light, float distance, float pdf_select) {
    int type = int(light.position.w + 0.5);
    if (type == 0) {
        // Point Light: Treat as delta for MIS purposes
        return 1.0 * pdf_select;
    } else if (type == 1) {
        // Directional Light: Treat as delta for MIS purposes
        return 1.0 * pdf_select;
    } else if (type == 2) {
        float area = light.params.y * light.params.z;
        return (1.0 / max(area, 1e-4)) * pdf_select;
    } else if (type == 3) {
        float solid = 2.0 * 3.14159265 * (1.0 - light.direction.w);
        return (1.0 / max(solid, 1e-4)) * pdf_select;
    }
    return 0.0;
}

// Sample direction toward light (approximation matching CPU logic)
bool sample_light_direction_gl(const LightData light, vec3 hit_pos, float rand_u, float rand_v, out vec3 wi, out float distance, out float attenuation) {
    int type = int(light.position.w + 0.5);
    attenuation = 1.0;
    if (type == 0) {
        vec3 L = light.position.xyz - hit_pos;
        distance = length(L);
        if (distance < 1e-3) return false;
        vec3 dir = L / distance;
        vec3 jitter = normalize(vec3((rand_u - 0.5) * 2.0, (rand_v - 0.5) * 2.0, (rand_u * rand_v - 0.5) * 2.0)) * light.params.x;
        wi = normalize(dir * distance + jitter);
        attenuation = 1.0 / (distance * distance);
        return true;
    } else if (type == 1) {
        vec3 L = normalize(light.direction.xyz);
        // Build tangent frame: check raw cross product BEFORE normalize.
        // normalize(zero) is undefined (often NaN) and NaN<threshold is false → fallback would never fire.
        vec3 tangent_raw = cross(L, vec3(0.0, 1.0, 0.0));
        if (dot(tangent_raw, tangent_raw) < 1e-6) {
            tangent_raw = cross(L, vec3(1.0, 0.0, 0.0));
        }
        vec3 tangent = normalize(tangent_raw);
        vec3 bitangent = normalize(cross(L, tangent));
        float r = sqrt(rand_u) * light.params.x;
        float phi = 2.0 * 3.14159265 * rand_v;
        vec2 disk = vec2(cos(phi) * r, sin(phi) * r);
        vec3 light_pos = L * 1000.0 + tangent * disk.x + bitangent * disk.y;
        wi = normalize(light_pos);
        attenuation = 1.0;
        distance = 1e8;
        return true;
    } else if (type == 2) {
        // Area: random point on rectangle using AreaLight's true u/v axes (parity with OptiX/CPU)
        float u_off = (rand_u - 0.5) * light.params.y;
        float v_off = (rand_v - 0.5) * light.params.z;
        vec3 light_sample = light.position.xyz + light.area_u.xyz * u_off + light.area_v.xyz * v_off;
        vec3 L = light_sample - hit_pos;
        distance = length(L);
        if (distance < 1e-3) return false;
        wi = L / distance;
        vec3 light_normal = normalize(cross(light.area_u.xyz, light.area_v.xyz));
        float cos_light = max(dot(-wi, light_normal), 0.0);
        attenuation = cos_light / (distance * distance);
        return true;
    } else if (type == 3) {
        vec3 L = light.position.xyz - hit_pos;
        distance = length(L);
        if (distance < 1e-3) return false;
        wi = normalize(L);
        float falloff = spot_light_falloff_gl(light, wi);
        if (falloff < 1e-4) return false;
        attenuation = falloff / (distance * distance);
        return true;
    }
    return false;
}

vec3 fresnel_schlick_roughness_gl(float cosTheta, vec3 F0, float roughness) {
    vec3 F90 = max(vec3(1.0 - roughness), F0);
    float f = pow(1.0 - cosTheta, 5.0);
    return F0 + (F90 - F0) * f;
}

// BRDF evaluation (Cook-Torrance simplified port)
vec3 evaluate_brdf_gl(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic, float specular, float transmission) {
    vec3 VpL = V + L;
    float VpL_len2 = dot(VpL, VpL);
    vec3 H = (VpL_len2 > 1e-12) ? (VpL * inversesqrt(VpL_len2)) : N;
    float NdotV = max(dot(N, V), 1e-4);
    float NdotL = max(dot(N, L), 1e-4);
    float NdotH = max(dot(N, H), 1e-4);
    float VdotH = max(dot(V, H), 1e-4);
    float dielectricF0 = clamp(0.08 * specular, 0.0, 0.08);
    vec3 F0 = mix(vec3(dielectricF0), albedo, metallic);
    vec3 F = fresnel_schlick_roughness_gl(VdotH, F0, roughness);
    vec3 F_avg = F0 + (vec3(1.0) - F0) / 21.0;
    // D (GGX)
    float safeRoughness = clamp(roughness, 0.02, 1.0);
    float alpha = max(safeRoughness * safeRoughness, 1e-4);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (3.14159265 * denom * denom + 1e-8);
    // G (Smith)
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float G = (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));
    vec3 spec = (F * D * G) / (4.0 * NdotV * NdotL + 1e-6);
    // Diffuse — F'yi gerçek açıyla kullan (energy conservation)
    vec3 k_d = (vec3(1.0) - F_avg) * (1.0 - metallic) * max(0.0, 1.0 - transmission);
    vec3 diff = (k_d * albedo) * INV_PI;
    return diff + spec;
}

// BRDF PDF approx (GGX-based)
float pdf_brdf_gl(vec3 N, vec3 V, vec3 L, float roughness) {
    vec3 VpL = V + L;
    float VpL_len2 = dot(VpL, VpL);
    vec3 H = (VpL_len2 > 1e-12) ? (VpL * inversesqrt(VpL_len2)) : N;
    float NdotH = max(dot(N, H), 1e-4);
    float VdotH = max(dot(V, H), 1e-4);
    float safeRoughness = clamp(roughness, 0.02, 1.0);
    float alpha = max(safeRoughness * safeRoughness, 1e-4);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (3.14159265 * denom * denom + 1e-8);
    return D * NdotH / (4.0 * VdotH + 1e-6);
}

// Pick smart light (importance-based) - simplified GPU-parity using rnd
int pick_smart_light_gl(uvec2 dummySize, vec3 hit_pos, out float pdf_out) {
    int light_count = int(cam.lightCount);
    if (light_count == 0) { pdf_out = 0.0; return -1; }
    float rng = rnd(payload.seed);

    // Directional/güneş ışıklarını önce say — sabit 0.33 yerine uniform prob ver
    // böylece PDF değeri her zaman gerçek seçim olasılığıyla eşleşir
    int dir_count = 0;
    for (int i = 0; i < light_count; ++i) {
        if (int(lights.l[i].position.w + 0.5) == 1) dir_count++;
    }
    float prob_to_reach = 1.0;
    if (dir_count > 0) {
        float dir_prob = float(dir_count) / float(light_count);
        if (rng < dir_prob) {
            // Seçilen directional ışığa ulaş
            float step = dir_prob / float(dir_count);
            int sel = int(rng / step);
            int found = 0;
            for (int i = 0; i < light_count; ++i) {
                if (int(lights.l[i].position.w + 0.5) == 1) {
                    if (found == sel) { pdf_out = dir_prob / float(dir_count); return i; }
                    found++;
                }
            }
        }
        rng = (rng - dir_prob) / max(1.0 - dir_prob, 1e-6);
        prob_to_reach = 1.0 - dir_prob;
    }
    // Weighted selection
    float weights[128];
    float total = 0.0;
    int max_l = (light_count < 128) ? light_count : 128;
    for (int i = 0; i < max_l; ++i) {
        float w = 0.0;
        if (int(lights.l[i].position.w + 0.5) != 1) {
            vec3 delta = lights.l[i].position.xyz - hit_pos;
            float dist = max(length(delta), 1.0);
            float intensity = gc_luminance(lights.l[i].color.rgb) * lights.l[i].color.a;
            int t = int(lights.l[i].position.w + 0.5);
            if (t == 0) {
                // Point light: account for spherical sampling area (4*pi*r^2) so selection pdf
                // and per-light sampling pdf are consistent (avoids intensity scaling with radius).
                float area = 4.0 * PI * lights.l[i].params.x * lights.l[i].params.x;
                w = (1.0 / (dist * dist)) * intensity * area;
            } else if (t == 2) {
                w = (1.0 / (dist * dist)) * intensity * min(lights.l[i].params.y * lights.l[i].params.z, 10.0);
            } else if (t == 3) {
                w = (1.0 / (dist * dist)) * intensity * 0.8;
            }
        }
        weights[i] = w; total += w;
    }
    int sel = max_l - 1;
    if (total < 1e-6) {
        sel = int(rng * float(light_count)) % light_count;
        pdf_out = prob_to_reach * (1.0 / float(light_count));
        return sel;
    }
    float r = rng * total;
    float acc = 0.0;
    for (int i = 0; i < max_l; ++i) { acc += weights[i]; if (r <= acc) { sel = i; break; } }
    pdf_out = prob_to_reach * (weights[sel] / total);
    return sel;
}


// ============================================================
// Hemisphere Sampling
// ============================================================

// Cosine-weighted hemisphere — Lambert diffuse için ideal PDF
// PDF = cos(theta) / PI
vec3 cosineSampleHemisphere(vec3 normal, inout uint seed) {
    float r1  = rnd(seed);
    float r2  = rnd(seed);
    float phi = TWO_PI * r1;

    // Shirley disk mapping
    float sqrtR2 = sqrt(r2);
    float x = cos(phi) * sqrtR2;
    float y = sin(phi) * sqrtR2;
    float z = sqrt(max(0.0, 1.0 - r2));

    vec3 tangent, bitangent;
    buildONB(normal, tangent, bitangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}

// GGX NDF hemisphere sampling — sadece scatterGlass tarafından kullanılıyor
// PDF = D(h) * cos(theta_h) / (4 * dot(v, h))
vec3 ggxSampleHemisphere(vec3 normal, vec3 viewDir, float roughness, inout uint seed) {
    float r1    = rnd(seed);
    float r2    = rnd(seed);
    float safeRoughness = clamp(roughness, 0.02, 1.0);
    float alpha = safeRoughness * safeRoughness;

    float phi       = TWO_PI * r1;
    float cosTheta  = sqrt((1.0 - r2) / max(1.0 + (alpha * alpha - 1.0) * r2, 1e-7));
    float sinTheta  = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 halfVecLocal = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    vec3 tangent, bitangent;
    buildONB(normal, tangent, bitangent);
    vec3 halfVec = normalize(tangent * halfVecLocal.x + bitangent * halfVecLocal.y + normal * halfVecLocal.z);

    return reflect(-viewDir, halfVec);
}

// GGX VNDF sampling (Heitz 2018) — scatterMetal için
// Weight = F * G1(L), her zaman [0,1] aralığında → blow-up yok
vec3 ggxSampleVNDF(vec3 normal, vec3 viewDir, float alpha, float r1, float r2) {
    // ONB kur
    vec3 tangent, bitangent;
    buildONB(normal, tangent, bitangent);

    // V'yi tangent uzayına al
    vec3 Ve = vec3(dot(viewDir, tangent), dot(viewDir, bitangent), dot(viewDir, normal));

    // Alpha ile gerer
    vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    // Vh'ye dik ONB
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = (lensq > 1e-7) ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq)
                              : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    // Birim küre üzerinde örnek
    float r   = sqrt(r1);
    float phi = TWO_PI * r2;
    float t1  = r * cos(phi);
    float t2  = r * sin(phi);
    float s   = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;

    // Mikrofaset normali (lokal)
    vec3 Nh = T1 * t1 + T2 * t2
            + Vh * sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2));

    // Geriye doğru uzat → dünya normali
    vec3 Ne = normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
    vec3 H  = normalize(tangent * Ne.x + bitangent * Ne.y + normal * Ne.z);

    return reflect(-viewDir, H);
}

// ============================================================
// Fresnel
// ============================================================

// Schlick approximation
float schlickFresnel(float cosTheta, float ior) {
    float r0 = (1.0 - ior) / (1.0 + ior);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

bool refractLikeOptix(vec3 incident, vec3 normal, float eta, out vec3 refractedDir) {
    vec3 unitDir = normalize(incident);
    float cosTheta = clamp(dot(-unitDir, normal), -1.0, 1.0);
    vec3 rOutPerp = eta * (unitDir + cosTheta * normal);
    float k = 1.0 - dot(rOutPerp, rOutPerp);
    if (k < 0.0) {
        refractedDir = vec3(0.0);
        return false;
    }
    vec3 rOutParallel = -sqrt(k) * normal;
    refractedDir = normalize(rOutPerp + rOutParallel);
    return true;
}

// Metal için renkli Fresnel (F0 = albedo)
vec3 schlickFresnelVec(float cosTheta, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ============================================================
// Resin inclusion procedural fields (self-contained 3D noise)
// Used by the resin internal march: dust = fbm cloudiness (heterogeneous
// absorption), dirt = worley specks (opaque early-return). No scene rays.
// ============================================================
float rh_hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}
vec3 rh_hash33(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453);
}
// Value noise (trilinear, quintic-smoothed) — soft, non-blocky gradients.
float rh_vnoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);  // quintic — smoother than cubic (less coarse)
    float n000 = rh_hash13(i + vec3(0,0,0));
    float n100 = rh_hash13(i + vec3(1,0,0));
    float n010 = rh_hash13(i + vec3(0,1,0));
    float n110 = rh_hash13(i + vec3(1,1,0));
    float n001 = rh_hash13(i + vec3(0,0,1));
    float n101 = rh_hash13(i + vec3(1,0,1));
    float n011 = rh_hash13(i + vec3(0,1,1));
    float n111 = rh_hash13(i + vec3(1,1,1));
    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    return mix(nxy0, nxy1, f.z);
}
// Billowy turbulence FBM — sum of |signed noise| gives cloud-puff structure
// (wispy dense cores, clear gaps) instead of a flat smooth haze. 5 octaves with a
// per-octave offset to avoid axis-aligned repetition. Normalised to ~0..1.
float rh_fbm(vec3 p) {
    float v = 0.0, a = 0.5, tot = 0.0;
    for (int i = 0; i < 5; ++i) {
        v   += a * abs(2.0 * rh_vnoise(p) - 1.0);
        tot += a;
        p = p * 2.03 + vec3(7.1, 3.7, 11.3);
        a *= 0.5;
    }
    return v / max(tot, 1e-4);
}
// Worley/cellular: returns F1 distance (0 at cell centres) — small values =
// near a seed point → opaque dirt speck.
float rh_worley(vec3 p) {
    vec3 ip = floor(p);
    vec3 fp = fract(p);
    float d = 1.0;
    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x) {
        vec3 g = vec3(x, y, z);
        vec3 o = rh_hash33(ip + g);
        d = min(d, length(g + o - fp));
    }
    return d;
}

// ============================================================
// Material Scatter Fonksiyonları
// ============================================================

const uint BOUNCE_SPECULAR = 0u;
const uint BOUNCE_DIFFUSE = 1u;
const uint BOUNCE_TRANSMISSION = 2u;
const uint BOUNCE_TRANSPARENT = 3u;
// Resin interactions (glossy coat reflect + absorbing diffuse base) are tagged
// separately so raygen can cap them at a small dedicated budget — an
// energy-preserving resin would otherwise run full-depth GI paths (TDR risk).
const uint BOUNCE_RESIN = 4u;
// Glass marble full-volume entry: tagged on the FRONT-face transmit so raygen
// integrates the real interior segment (dust/dirt) before the next surface.
const uint BOUNCE_MARBLE = 5u;

// --- Lambertian Diffuse ---
void scatterDiffuse(vec3 hitPos, vec3 normal, vec3 albedo, inout uint seed) {
    vec3 dir = cosineSampleHemisphere(normal, seed);

    payload.scatterOrigin = hitPos + normal * RAY_OFFSET;
    payload.scatterDir    = dir;
    // Cosine-weighted sampling ile PDF = cos/PI, BRDF = albedo/PI
    // Throughput = BRDF * cos / PDF = albedo → direkt albedo
    payload.attenuation  *= albedo;
    payload.scattered     = true;
    payload.bounceType     = BOUNCE_DIFFUSE;
}

// --- GGX Metallic Reflection ---
void scatterMetal(vec3 hitPos, vec3 normal, vec3 rayDir, vec3 albedo, float roughness, inout uint seed) {
    vec3 viewDir = -rayDir;

    // Pürüzsüz ayna: tam reflection
    if (roughness < 0.01) {
        vec3 mirrorDir = reflect(rayDir, normal);
        float cosTheta = max(dot(viewDir, normal), 0.0);
        vec3 fresnel = schlickFresnelVec(cosTheta, albedo);
        payload.scatterOrigin = hitPos + normal * RAY_OFFSET;
        payload.scatterDir    = mirrorDir;
        payload.attenuation  *= fresnel;
        payload.scattered     = true;
        payload.bounceType     = BOUNCE_SPECULAR;
        return;
    }

    float alpha = max(roughness * roughness, 1e-4);

    // VNDF örnekleme: görünür faset dağılımından örnek al
    // Bu sayede weight = F * G1(L) — her zaman [0,1] aralığında, blow-up yok
    vec3 scatterDir = ggxSampleVNDF(normal, viewDir, alpha, rnd(seed), rnd(seed));

    // Yüzeyin altına düştüyse fallback
    if (dot(scatterDir, normal) <= 0.0) {
        scatterDir = reflect(rayDir, normal);
        if (dot(scatterDir, normal) <= 0.0) {
            scatterDir = normal;
        }
    }

    // Half-vector ve açılar
    vec3  H      = normalize(viewDir + scatterDir);
    float NdotL  = max(dot(normal, scatterDir), 1e-4);
    float VdotH  = max(dot(viewDir, H),         1e-4);

    // Fresnel: VdotH ile
    vec3 fresnel = schlickFresnelVec(VdotH, albedo);

    // VNDF weight = F * G1(L)
    // Türetme: VNDF PDF = G1(V)*D(H)*VdotH/NdotV → weight sadeleşince F*G1(L) kalır
    // G1(L) her zaman [0,1] → weight ≤ F ≤ 1 (metal için F0=albedo≤1)
    float k   = alpha * 0.5;  // IBL remapping
    float G1L = NdotL / (NdotL * (1.0 - k) + k);

    vec3 weight = fresnel * G1L;

    payload.scatterOrigin = hitPos + normal * RAY_OFFSET;
    payload.scatterDir    = scatterDir;
    payload.attenuation  *= weight;
    payload.scattered     = true;
    payload.bounceType     = BOUNCE_SPECULAR;
}

// --- Dielectric Glass (Fresnel + TIR + Roughness) ---
void scatterGlass(vec3 hitPos, vec3 macroNormalIn, vec3 shadingNormalIn, bool frontFace, vec3 rayDir, vec3 albedo, float ior, float roughness, float transmissionDensity, vec3 resinColor, inout uint seed) {
    // Işığın hangi taraftan geldiğini belirle
    vec3  macroNormal  = safeNormalize(macroNormalIn, vec3(0.0, 1.0, 0.0));
    vec3  shadingNormal = safeNormalize(shadingNormalIn, macroNormal);
    if (dot(shadingNormal, macroNormal) < 0.0) {
        shadingNormal = -shadingNormal;
    }
    // Resin needs real refraction to read as a solid volume (OptiX parity): with
    // IOR≈1 the ray passes straight through and only darkens (no lensing / depth cue).
    if (transmissionDensity > 1e-4) ior = max(ior, 1.45);
    float etaRatio     = frontFace ? (1.0 / ior) : ior;
    
    vec3 outNormal = shadingNormal;

    // Fade in GGX microfacet normals instead of switching abruptly at 0.01.
    if (roughness > 0.0005) {
        vec3 V = -rayDir;
        float sampleRoughness = max(roughness, 0.02);
        float roughBlend = smoothstep(0.0, 0.02, roughness);
        vec3 sampledNormal = ggxSampleHemisphere(shadingNormal, V, sampleRoughness, seed);
        outNormal = normalize(mix(shadingNormal, sampledNormal, roughBlend));
    }

    // Fresnel ve TIR kararı için makro normal kullan (OptiX ile aynı).
    // Mikrofaset normali grazing angle'a yakın örneklenirse fresnelProb
    // yapay biçimde ~1.0'a çıkıyor ve aşırı reflection üretiyordu.
    float cosTheta     = min(dot(-rayDir, macroNormal), 1.0);
    float sinTheta     = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool  totalIntRefl = (etaRatio * sinTheta) > 1.0;

    float fresnelProb  = schlickFresnel(cosTheta, ior);
    bool  doReflect    = totalIntRefl || (rnd(seed) < fresnelProb);

    bool realDepth = (transmissionDensity > 1e-4);

    // ── RESIN terminate-on-base: the refraction lobe travels the resin THICKNESS,
    // hits the actual base material albedo at that depth, and scatters back out
    // through the resin (absorb in + out). The object is OPAQUE under a refractive
    // absorbing resin layer (no see-through). The reflection lobe stays the glossy
    // resin top. resinColor = the colour that builds over the thickness. ──
    if (realDepth && !doReflect) {
        vec3  ct      = clamp(resinColor, vec3(0.0), vec3(1.0));
        float cosV    = max(abs(dot(-rayDir, macroNormal)), 0.25);
        float pathLen = 2.0 * transmissionDensity / cosV;   // in + out through the thickness
        // Beer-Lambert extinction. A small BASE extinction (0.25) makes Resin Depth
        // darken even for a white/clear resin (artist expectation); resinColor then
        // tints which channels survive (lower channel = absorbed faster = that hue stays).
        vec3  ext     = (vec3(1.0) - ct) + vec3(0.25);
        vec3  absorb  = exp(-pathLen * ext);
        vec3  baseDir = cosineSampleHemisphere(macroNormal, seed); // diffuse off the base albedo
        payload.scatterOrigin = offset_ray(hitPos, macroNormal);
        payload.scatterDir    = normalize(baseDir);
        payload.attenuation  *= clamp(albedo, vec3(0.0), vec3(1.0)) * absorb;
        payload.scattered     = true;
        payload.bounceType     = BOUNCE_DIFFUSE;
        return;
    }

    vec3 dir;
    vec3 offsetDir;
    if (doReflect) {
        dir       = reflect(rayDir, outNormal);
        offsetDir = macroNormal;           // Yüzeyin dışına offset
    } else {
        bool refractedSuccess = refractLikeOptix(rayDir, outNormal, etaRatio, dir);
        offsetDir = -macroNormal;          // Yüzeyin içine offset (refract için)
        if (!refractedSuccess) {
            dir = reflect(rayDir, macroNormal);
            offsetDir = macroNormal;
        }
    }

    // OptiX parity: only refraction is guarded against escaping to the wrong side.
    if (!doReflect && dot(dir, macroNormal) >= 0.0) {
        dir = reflect(rayDir, macroNormal);
        offsetDir = macroNormal;
    }

    payload.scatterOrigin = offset_ray(hitPos, offsetDir);
    payload.scatterDir    = normalize(dir);
    if (doReflect) {
        payload.attenuation *= vec3(1.0);
    } else {
        vec3 glassTint = clamp(albedo, vec3(0.0), vec3(1.0));
        float cosInside = max(abs(dot(normalize(dir), -macroNormal)), 0.05);
        float opticalThickness = 0.65 / cosInside;
        vec3 absorption = (vec3(1.0) - glassTint) * opticalThickness;
        vec3 transmissionColor = vec3(
            exp(-absorption.x),
            exp(-absorption.y),
            exp(-absorption.z)
        );
        payload.attenuation *= transmissionColor;
    }
    payload.scattered     = true;
    payload.bounceType     = BOUNCE_TRANSMISSION;
}

// ============================================================
// Henyey-Greenstein phase function direction sampling
// ============================================================
vec3 sampleHG(vec3 forward, float g, inout uint seed) {
    float cosTheta;
    if (abs(g) < 0.001) {
        cosTheta = 1.0 - 2.0 * rnd(seed);  // Isotropic
    } else {
        float sqrTerm = (1.0 - g * g) / (1.0 - g + 2.0 * g * rnd(seed));
        cosTheta = (1.0 + g * g - sqrTerm * sqrTerm) / (2.0 * g);
    }
    cosTheta = clamp(cosTheta, -1.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = TWO_PI * rnd(seed);

    vec3 up = (abs(forward.z) < 0.999) ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 T = normalize(cross(up, forward));
    vec3 B = cross(forward, T);

    return normalize(T * (sinTheta * cos(phi)) + B * (sinTheta * sin(phi)) + forward * cosTheta);
}

// ============================================================
// Subsurface Scattering — Random Walk (OptiX parity)
// ============================================================
void scatterSSS(vec3 hitPos, vec3 normal, vec3 albedo,
                vec3 sssColor, float sssAmount, float sssScale,
                vec3 sssRadius, float sssAnisotropy,
                inout uint seed) {
    // Multiscatter random-walk SSS (bounded)
    float safeScale = max(sssScale, 0.001);
    vec3 scaledRadius = sssRadius * safeScale;
    vec3 sigma_t = vec3(
        scaledRadius.x > 0.0001 ? 1.0 / scaledRadius.x : 10000.0,
        scaledRadius.y > 0.0001 ? 1.0 / scaledRadius.y : 10000.0,
        scaledRadius.z > 0.0001 ? 1.0 / scaledRadius.z : 10000.0
    );

    const int maxSteps = 6;
    vec3 throughput = vec3(1.0);
    vec3 pos = hitPos - normal * 0.001; // start slightly inside
    vec3 dir = sampleHG(-normal, sssAnisotropy, seed);

    for (int step = 0; step < maxSteps; ++step) {
        float randCh = rnd(seed);
        float sigmaSample = (randCh < 0.333) ? sigma_t.x : (randCh < 0.666) ? sigma_t.y : sigma_t.z;
        float scatterDist = -log(max(rnd(seed), 1e-6)) / max(sigmaSample, 1e-6);
        float maxRadius = max(max(scaledRadius.x, scaledRadius.y), scaledRadius.z);
        scatterDist = min(scatterDist, maxRadius * 3.0);

        pos += dir * scatterDist;

        vec3 absorb = vec3(
            exp(-sigma_t.x * scatterDist),
            exp(-sigma_t.y * scatterDist),
            exp(-sigma_t.z * scatterDist)
        );
        throughput *= absorb;

        float survive = clamp((throughput.x + throughput.y + throughput.z) / 3.0, 0.01, 0.99);
        if (rnd(seed) > survive) break;

        if (dot(dir, normal) > 0.0) {
            // Exiting to surface: apply accumulated SSS tint and exit
            payload.attenuation *= sssColor * throughput;
            payload.scatterOrigin = pos + normal * RAY_OFFSET;
            payload.scatterDir = normalize(dir);
            payload.scattered = true;
            payload.bounceType = BOUNCE_DIFFUSE;
            return;
        }

        // Scatter internally
        dir = sampleHG(dir, sssAnisotropy, seed);
    }

    // Fallback exit: cosine hemisphere outward
    vec3 outDir = cosineSampleHemisphere(normal, seed);
    payload.attenuation *= sssColor * throughput;
    payload.scatterOrigin = pos + normal * RAY_OFFSET;
    payload.scatterDir = outDir;
    payload.scattered = true;
    payload.bounceType = BOUNCE_DIFFUSE;
}

// ============================================================
// Clearcoat — Second GGX Specular Lobe (IOR=1.5, lacquer layer)
// ============================================================
void scatterClearcoat(vec3 hitPos, vec3 normal, vec3 rayDir,
                      float ccRoughness, float iridescence, float filmThickness,
                      inout uint seed) {
    // IOR=1.5 → F0 = ((1.5-1)/(1.5+1))^2 ≈ 0.04
    const float CC_F0 = 0.04;

    vec3 viewDir = -rayDir;
    float alpha = max(ccRoughness * ccRoughness, 1e-4);

    // GGX VNDF sample — reuse same ggxSampleVNDF function
    vec3 L = ggxSampleVNDF(normal, viewDir, alpha, rnd(seed), rnd(seed));
    if (dot(L, normal) <= 0.0) {
        L = reflect(rayDir, normal);
        if (dot(L, normal) <= 0.0) {
            payload.scattered = false;
            return;
        }
    }

    vec3  H      = normalize(viewDir + L);
    float VdotH  = max(dot(viewDir, H), 0.001);
    float NdotL  = max(dot(normal, L), 1e-4);
    float NdotV  = max(dot(normal, viewDir), 1e-4);

    // Schlick Fresnel for clearcoat F0=0.04
    float fresnel = CC_F0 + (1.0 - CC_F0) * pow(1.0 - VdotH, 5.0);

    // G1(L) geometry term (same as scatterMetal)
    float k   = alpha * 0.5;
    float G1L = NdotL / (NdotL * (1.0 - k) + k);

    // Iridescent thin-film tint (same OPD/cos model as the bubble path). The clearcoat
    // is a thin dielectric layer; at grazing the optical path difference grows, cycling
    // the interference hue. iridescence=0 → white (plain clearcoat, no change).
    vec3 ccTint = vec3(1.0);
    if (iridescence > 1e-3) {
        float opd = filmThickness * (1.0 / max(VdotH, 0.15));
        vec3 filmCol = vec3(0.55 + 0.45 * cos(opd * 6.2831853),
                            0.55 + 0.45 * cos(opd * 6.2831853 + 2.0944),
                            0.55 + 0.45 * cos(opd * 6.2831853 + 4.18879));
        ccTint = mix(vec3(1.0), filmCol, clamp(iridescence, 0.0, 1.0));
    }

    payload.attenuation  *= vec3(fresnel * G1L) * ccTint;
    payload.scatterOrigin = hitPos + normal * RAY_OFFSET;
    payload.scatterDir    = L;
    payload.scattered     = true;
    payload.bounceType     = BOUNCE_SPECULAR;
}

// ============================================================
// Translucent — Thin-surface diffuse transmission (leaves, cloth, paper)
// ============================================================
void scatterTranslucent(vec3 hitPos, vec3 normal, vec3 albedo, inout uint seed) {
    // Cosine-weighted hemisphere on the opposite (transmission) side
    vec3 transDir = cosineSampleHemisphere(-normal, seed);

    // Slight absorption on pass-through
    payload.attenuation  *= albedo * 0.8;
    payload.scatterOrigin = hitPos - normal * 0.001;
    payload.scatterDir    = transDir;
    payload.scattered     = true;
    payload.bounceType     = BOUNCE_DIFFUSE;
}

// ============================================================
// WATER — Gerstner Waves + Micro-Detail Ripples (IS_WATER path)
//
// Parameter packing from WaterManager.cpp:
//   mat.anisotropic          = wave_speed
//   mat.sheen                = wave_strength  (IS_WATER flag when > 0)
//   mat.sheen_tint           = wave_frequency
//   mat.emission_r/g/b       = shallow_color
//   mat.albedo_r/g/b         = deep_color
//   mat.translucent          = foam_level
//   mat.subsurface_amount    = depth_max / 100
//   mat.fft_time_scale       = animation speed multiplier
//   mat.micro_detail_strength= micro ripple strength
//   mat.micro_detail_scale   = micro ripple scale
//   mat.foam_threshold       = foam appearance threshold
//   mat.fft_wind_speed       = wind speed for micro ripples
// ============================================================

// --- Hash / Noise helpers (mirrors water_shaders_cpu.h) ---
float water_hash12(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float water_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(water_hash12(i + vec2(0.0,0.0)), water_hash12(i + vec2(1.0,0.0)), u.x),
               mix(water_hash12(i + vec2(0.0,1.0)), water_hash12(i + vec2(1.0,1.0)), u.x), u.y) * 2.0 - 1.0;
}

float water_fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    const float c = 0.866025, s = 0.5; // cos/sin(30)
    for (int i = 0; i < 4; ++i) {
        v += a * water_noise(p);
        p  = vec2(p.x * c - p.y * s, p.x * s + p.y * c) * 2.0 + 100.0;
        a *= 0.5;
    }
    return v;
}

vec3 calculateDepthColorGL(float depth, float depth_max, vec3 shallow_color, vec3 deep_color) {
    float t = min(depth / max(depth_max, 0.1), 1.0);
    t = t * t * (3.0 - 2.0 * t);
    return mix(shallow_color, deep_color, t);
}

float calculateWaterCausticsGL(vec3 floor_position, float time, float caustic_scale, float caustic_speed) {
    vec2 uv = vec2(floor_position.x, floor_position.z) * caustic_scale;
    float t = time * caustic_speed;
    float v1 = abs(water_fbm(uv + vec2(t * 0.4, -t * 0.2)));
    float v2 = abs(water_fbm(uv * 1.5 + vec2(50.0 + t * 0.3, 50.0 - t * 0.25)));
    float caustic = 1.0 - v1 * v2;
    return pow(max(caustic, 0.0), 2.0);
}

float calculateShoreFoamGL(float depth, float shore_distance, float shore_intensity, vec3 position, float time, float foam_scale) {
    if (depth > shore_distance || shore_distance < 0.001) return 0.0;
    float shore_t = 1.0 - (depth / shore_distance);
    shore_t = smoothstep(0.0, 1.0, shore_t * shore_t);
    float scale = max(foam_scale, 0.001);
    float foam_noise = water_fbm(vec2(position.x * scale + time * 0.5, position.z * scale - time * 0.3)) * 0.5 + 0.5;
    float edge_pattern = sin(depth * (10.0 / shore_distance) - time * 3.0) * 0.5 + 0.5;
    float shore_foam = shore_t * shore_intensity * ((foam_noise * 0.7) + (edge_pattern * 0.3));
    return min(shore_foam * 1.5, 1.0);
}

bool estimateWaterDepthGL(vec3 hitPos, float maxDepth, out float waterDepth, out vec3 floorPosition) {
    waterDepth = max(maxDepth, 0.1);
    floorPosition = hitPos - vec3(0.0, waterDepth, 0.0);
    if (waterDepth <= SHADOW_TMIN) return false;

    float low = SHADOW_TMIN;
    float high = waterDepth;
    bool found = false;
    vec3 probeOrigin = hitPos - vec3(0.0, 0.05, 0.0);
    vec3 probeDir = vec3(0.0, -1.0, 0.0);
    uint probeFlags = gl_RayFlagsTerminateOnFirstHitEXT
                    | gl_RayFlagsSkipClosestHitShaderEXT;

    for (int i = 0; i < 7; ++i) {
        float mid = mix(low, high, 0.5);
        shadowOccluded = true;
        traceRayEXT(topLevelAS, probeFlags, 0x01, 0, 1, 1, probeOrigin, SHADOW_TMIN, probeDir, mid, 1);
        if (shadowOccluded) {
            found = true;
            high = mid;
        } else {
            low = mid;
        }
    }

    if (found) {
        waterDepth = high;
        floorPosition = hitPos - vec3(0.0, waterDepth, 0.0);
    }
    return found;
}

bool weatherActive() {
    return worldData.w.weatherEnabled != 0 && worldData.w.weatherType != 0 &&
           worldData.w.weatherIntensity > 0.0 && worldData.w.weatherDensity > 0.0;
}

bool weatherSurfaceActive() {
    if (worldData.w.weatherEnabled == 0 || worldData.w.weatherType == 0 ||
        worldData.w.weatherSurfaceResponseEnabled == 0) {
        return false;
    }

    float surfaceSignal = 0.0;
    if (worldData.w.weatherType == 1) {
        surfaceSignal = worldData.w.weatherSurfaceWetness;
    } else if (worldData.w.weatherType == 2 || worldData.w.weatherType == 3) {
        surfaceSignal = worldData.w.weatherSurfaceAccumulation;
    }

    return surfaceSignal > 0.001 || (worldData.w.weatherIntensity > 0.0 && worldData.w.weatherDensity > 0.0);
}

float weatherSurfaceGeometricSupport(vec3 supportNormal) {
    vec3 macroNormal = safeNormalize(supportNormal, vec3(0.0, 1.0, 0.0));
    float support = 0.0;
    if (worldData.w.weatherType == 2) {
        support = clamp((macroNormal.y - 0.02) / 0.72, 0.0, 1.0);
        support *= support;
    } else {
        support = clamp((macroNormal.y - 0.02) / 0.78, 0.0, 1.0);
    }
    return support * support * (3.0 - 2.0 * support);
}

float weatherSurfaceExposure(vec3 hitPos, vec3 normal) {
    float upMask = smoothstep(0.12, 0.90, normal.y);
    float windAmount = clamp(worldData.w.weatherWindSpeed / 35.0, 0.0, 1.0);
    vec3 windRaw = worldData.w.weatherWindDirection;
    vec3 windDir = (dot(windRaw, windRaw) > 1e-8) ? normalize(windRaw) : vec3(1.0, 0.0, 0.0);
    vec3 incoming = normalize(vec3(0.0, 1.0, 0.0) - windDir * windAmount);
    float windFacing = clamp(dot(safeNormalize(normal, vec3(0.0, 1.0, 0.0)), incoming), 0.0, 1.0);
    float exposure = clamp(upMask * (1.0 - windAmount * 0.78) + windFacing * (0.12 + windAmount * 1.22), 0.0, 1.0);
    float scale = max(worldData.w.weatherPrecipitationScale, 0.1);
    float n = water_fbm(hitPos.xz * scale * 0.18 + vec2(13.1, 47.2)) * 0.5 + 0.5;
    float breakup = clamp(n * 1.35 - 0.18, 0.0, 1.0);
    return exposure * mix(0.45, 1.0, breakup);
}

float weatherSurfaceSettling(vec3 hitPos, vec3 normal, vec3 supportNormal);

float weatherSurfaceAccumulation(vec3 hitPos, vec3 normal, vec3 supportNormal) {
    if (worldData.w.weatherType != 2 && worldData.w.weatherType != 3) {
        return 0.0;
    }

    float baseAccum = clamp(worldData.w.weatherSurfaceAccumulation, 0.0, 1.0);
    float intensity = clamp(worldData.w.weatherIntensity, 0.0, 1.0);
    float density = clamp(worldData.w.weatherDensity, 0.0, 1.0);
    float geomSupport = weatherSurfaceGeometricSupport(supportNormal);
    float intensityResponse = 0.80 + intensity * 0.70;
    float densityResponse = 0.35 + density * 1.15;
    float typeBoost = (worldData.w.weatherType == 2) ? 1.10 : 0.90;
    float directAccum = baseAccum * intensityResponse * weatherSurfaceExposure(hitPos, normal) * densityResponse * typeBoost * geomSupport;
    float settling = weatherSurfaceSettling(hitPos, normal, supportNormal);
    return clamp(directAccum + (1.0 - clamp(directAccum, 0.0, 1.0)) * settling, 0.0, 1.0);
}

float weatherSurfaceSettling(vec3 hitPos, vec3 normal, vec3 supportNormal) {
    if (worldData.w.weatherType != 2 && worldData.w.weatherType != 3) {
        return 0.0;
    }

    float settlingAmount = clamp(worldData.w.weatherSurfaceSettling, 0.0, 1.0);
    if (settlingAmount <= 1e-4) {
        return 0.0;
    }

    vec3 shadingNormal = safeNormalize(normal, vec3(0.0, 1.0, 0.0));
    vec3 macroNormal = safeNormalize(supportNormal, shadingNormal);
    float support = weatherSurfaceGeometricSupport(supportNormal);
    float supportGate = clamp((support - 0.02) / 0.58, 0.0, 1.0);
    if (supportGate <= 1e-4) {
        return 0.0;
    }
    float exposure = weatherSurfaceExposure(hitPos, shadingNormal);
    float cavity = clamp((1.0 - dot(shadingNormal, macroNormal)) * 3.8 + (1.0 - support) * 0.10, 0.0, 1.0);
    vec3 windFlat = vec3(worldData.w.weatherWindDirection.x, 0.0, worldData.w.weatherWindDirection.z);
    vec3 leeDir = dot(windFlat, windFlat) > 1e-8 ? safeNormalize(vec3(-windFlat.x, 0.28, -windFlat.z), vec3(0.0, 1.0, 0.0)) : vec3(0.0, 1.0, 0.0);
    float lee = clamp(dot(macroNormal, leeDir) * 0.85 + cavity * 0.35, 0.0, 1.0);
    float shelter = clamp((1.0 - exposure) * 0.52 + cavity * 0.26 + (1.0 - support) * 0.22 + lee * 0.42, 0.0, 1.0);
    float pocketNoise = pd_vnoise3(hitPos * 0.085 + vec3(31.4, 9.7, 54.2));
    float pocketMask = clamp(cavity * 0.92 + pocketNoise * 0.26, 0.0, 1.0);
    float slopeBase = clamp((support - 0.16) / 0.54, 0.0, 1.0);
    float density = clamp(worldData.w.weatherDensity, 0.0, 1.0);
    float typeBoost = (worldData.w.weatherType == 2) ? 1.34 : 1.04;
    float anchor = max(pocketMask, slopeBase * 0.30 + cavity * 0.40 + lee * 0.30);
    return clamp(settlingAmount * supportGate * shelter * anchor * (0.76 + density * 1.10) * typeBoost, 0.0, 1.0);
}

float weatherSurfaceHeight(vec3 hitPos) {
    float scale = max(worldData.w.weatherPrecipitationScale, 0.1);
    float heightBoost = 0.25 + clamp(worldData.w.weatherSurfaceHeight, 0.0, 1.0) * 3.75;
    if (worldData.w.weatherType == 2) {
        vec2 windXZ = worldData.w.weatherWindDirection.xz;
        float windLen = length(windXZ);
        vec2 along = windLen > 1e-4 ? windXZ / windLen : vec2(1.0, 0.0);
        vec2 across = vec2(-along.y, along.x);
        vec2 uv = vec2(dot(hitPos.xz, along), dot(hitPos.xz, across));
        vec3 p = vec3(uv.x * scale * 0.12, hitPos.y * scale * 0.03, uv.y * scale * 0.12);
        float broad = pd_vnoise3(p * 0.55 + vec3(17.3, 9.1, 41.7));
        float drift = 1.0 - abs(pd_vnoise3(vec3(p.x * 1.45, p.y * 0.8, p.z * 0.58) + vec3(3.7, 29.4, 11.8)) * 2.0 - 1.0);
        drift *= drift;
        float clumps = 1.0 - abs(pd_vnoise3(p * 2.90 + vec3(61.2, 7.5, 18.9)) * 2.0 - 1.0);
        float micro = pd_vnoise3(p * 7.40 + vec3(8.3, 51.7, 27.4));
        return (broad * 0.22 + drift * 0.36 + clumps * 0.27 + micro * 0.15) * heightBoost;
    }

    vec3 p = hitPos * (scale * 0.18);
    float wisps = pd_vnoise3(p + vec3(19.7, 5.3, 27.1));
    float grain = pd_vnoise3(p * 2.85 + vec3(4.1, 37.8, 12.4));
    float streak = pd_vnoise3(p * 1.65 + vec3(44.5, 14.2, 7.6));
    return (wisps * 0.30 + grain * 0.45 + streak * 0.25) * heightBoost;
}

vec3 weatherSurfaceNormal(vec3 hitPos, vec3 normal, vec3 supportNormal) {
    vec3 baseNormal = safeNormalize(normal, vec3(0.0, 1.0, 0.0));
    if (!weatherSurfaceActive()) return baseNormal;
    if (worldData.w.weatherType != 2 && worldData.w.weatherType != 3) return baseNormal;

    float accumulation = weatherSurfaceAccumulation(hitPos, baseNormal, supportNormal);
    if (accumulation <= 1e-4) return baseNormal;

    float geomSupport = weatherSurfaceGeometricSupport(supportNormal);
    if (geomSupport <= 1e-4) return baseNormal;

    float settling = weatherSurfaceSettling(hitPos, baseNormal, supportNormal);
    float detailCapture = 0.45 + 0.55 * clamp((baseNormal.y - 0.04) / 0.82, 0.0, 1.0);
    float heightResponse = 0.12 + clamp(worldData.w.weatherSurfaceHeight, 0.0, 1.0) * 0.95;
    float buildup = clamp(accumulation + settling * 0.85, 0.0, 1.0);
    float normalStrength = buildup * detailCapture * heightResponse * (worldData.w.weatherType == 2 ? 0.42 : 0.15);
    if (normalStrength <= 1e-4) return baseNormal;

    vec3 wind = worldData.w.weatherWindDirection;
    vec3 tangent = wind - baseNormal * dot(wind, baseNormal);
    if (dot(tangent, tangent) <= 1e-8) {
        vec3 helper = abs(baseNormal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        tangent = cross(helper, baseNormal);
    }
    tangent = safeNormalize(tangent, vec3(1.0, 0.0, 0.0));
    vec3 bitangent = safeNormalize(cross(baseNormal, tangent), vec3(0.0, 0.0, 1.0));
    tangent = safeNormalize(cross(bitangent, baseNormal), tangent);

    float sampleStep = (worldData.w.weatherType == 2 ? 0.62 : 0.90) / max(worldData.w.weatherPrecipitationScale, 0.35);
    float heightCenter = weatherSurfaceHeight(hitPos);
    float heightT = weatherSurfaceHeight(hitPos + tangent * sampleStep);
    float heightB = weatherSurfaceHeight(hitPos + bitangent * sampleStep);
    float gradT = clamp((heightT - heightCenter) / sampleStep, -0.28, 0.28);
    float gradB = clamp((heightB - heightCenter) / sampleStep, -0.28, 0.28);

    vec3 perturbed = safeNormalize(baseNormal - tangent * (gradT * normalStrength) - bitangent * (gradB * normalStrength), baseNormal);
    if (dot(perturbed, baseNormal) < 0.05) {
        perturbed = safeNormalize(mix(baseNormal, perturbed, 0.35), baseNormal);
    }
    if (dot(perturbed, supportNormal) < 0.55) {
        perturbed = safeNormalize(mix(supportNormal, perturbed, 0.05), supportNormal);
    }
    return perturbed;
}

void applyWeatherSurface(vec3 hitPos, vec3 normal, vec3 supportNormal, inout vec3 albedo, inout float roughness, inout float metallic) {
    if (!weatherSurfaceActive()) return;

    float exposed = weatherSurfaceExposure(hitPos, normal);

    if (worldData.w.weatherType == 1) {
        float wet = clamp(worldData.w.weatherSurfaceWetness, 0.0, 1.0) *
                    mix(0.35, 1.0, exposed);
        albedo = mix(albedo, albedo * 0.50, wet * 0.62);
        roughness = max(0.012, roughness * (1.0 - wet * 0.78));
        metallic = max(0.0, metallic - wet * 0.05);
    } else if (worldData.w.weatherType == 2) {
        float acc = weatherSurfaceAccumulation(hitPos, normal, supportNormal);
        float settling = weatherSurfaceSettling(hitPos, normal, supportNormal);
        float heightLift = clamp(worldData.w.weatherSurfaceHeight, 0.0, 1.0);
        float cover = clamp(acc + settling * 0.84 + heightLift * (acc * 0.08 + settling * 0.30), 0.0, 1.0);
        albedo = mix(albedo, vec3(0.88, 0.91, 0.96), cover * 0.72);
        roughness = min(1.0, roughness + cover * (0.45 + heightLift * 0.10));
        metallic *= (1.0 - cover * 0.8);
    } else if (worldData.w.weatherType == 3) {
        float acc = weatherSurfaceAccumulation(hitPos, normal, supportNormal);
        float settling = weatherSurfaceSettling(hitPos, normal, supportNormal);
        float heightLift = clamp(worldData.w.weatherSurfaceHeight, 0.0, 1.0);
        float cover = clamp(acc + settling * 0.90 + heightLift * settling * 0.22, 0.0, 1.0);
        albedo = mix(albedo, vec3(0.58, 0.46, 0.30), cover * 0.55);
        roughness = min(1.0, roughness + cover * (0.35 + heightLift * 0.08));
        metallic *= (1.0 - cover * 0.55);
    }
}

// --- Multi-octave Gerstner waves (8 waves, matches CPU/CUDA impl) ---
void evaluateWaterGerstner(vec3 pos, float time,
                           float speed_mult, float strength_mult, float freq_mult,
                           out vec3 waveNormal, out float foam)
{
    const vec2 dirs[8] = vec2[8](
        normalize(vec2( 1.0,  0.2)), normalize(vec2( 0.7,  0.7)),
        normalize(vec2(-0.2,  1.0)), normalize(vec2(-0.6,  0.5)),
        normalize(vec2(-0.8, -0.3)), normalize(vec2( 0.0, -1.0)),
        normalize(vec2( 0.5, -0.8)), normalize(vec2( 0.9, -0.4))
    );

    float dHx = 0.0, dHz = 0.0, jacobian = 1.0, height = 0.0;
    float frequency = 0.2 * freq_mult;
    float amplitude = 0.5 * strength_mult;
    float speed     = 0.5 * speed_mult;

    for (int i = 0; i < 8; ++i) {
        vec2  d     = dirs[i];
        float x     = pos.x * d.x + pos.z * d.y;
        float phase = x * frequency + time * speed;
        float cp = cos(phase), sp = sin(phase);
        float wa = frequency * amplitude;
        dHx      += d.x * wa * cp;
        dHz      += d.y * wa * cp;
        jacobian -= 0.5 * wa * sp;   // steepness = 0.5
        height   += amplitude * sp;
        frequency *= 1.8;
        amplitude *= 0.55;
        speed     *= 1.1;
    }

    waveNormal = normalize(vec3(-dHx, 1.0, -dHz));

    float j_foam = 0.5 - jacobian;
    float h_foam = height - 0.5 * strength_mult;
    foam = clamp(max(0.0, j_foam * 2.0) + max(0.0, h_foam), 0.0, 1.0);
    foam = smoothstep(0.3, 0.7, foam);
}

// --- Main water scatter entry ---
void scatterWater(vec3 hitPos, vec3 geoNormal, vec3 rayDir,
                  float wave_speed, float wave_strength, float wave_freq,
                  float foam_level, float foam_threshold,
                  float micro_strength, float micro_scale,
                  float micro_anim_speed, float micro_morph_speed,
                  float foam_noise_scale, float wind_direction, float wind_speed,
                  float fft_time_scale, float fft_ocean_size,
                  uint fft_height_tex, uint fft_normal_tex,
                  float depth_max, float absorption_density,
                  float shore_foam_distance, float shore_foam_intensity,
                  float caustic_intensity, float caustic_scale, float caustic_speed,
                  vec3 shallow_color, vec3 deep_color,
                  float ior, float roughness,
                  inout uint seed)
{
    // OptiX parity: FFT simulation is already time-scaled before textures are generated.
    // The shading pass uses the resolved water time directly for both FFT sampling and
    // micro-ripple drift so Vulkan does not double-accelerate the surface.
    float time = cam.waterTime;

    // ── Gerstner wave normal + foam ─────────────────────────────
    bool useFFTOcean = fft_height_tex > 0u && fft_normal_tex > 0u && fft_ocean_size > 0.001;
    vec3  waveNormal;
    float foam;
    if (useFFTOcean) {
        vec2 fftUV = fract(vec2(hitPos.x / fft_ocean_size, hitPos.z / fft_ocean_size));
        float fftHeight = texture(materialTextures[nonuniformEXT(int(fft_height_tex))], fftUV).r;
        vec2 fftSlopeXZ = texture(materialTextures[nonuniformEXT(int(fft_normal_tex))], fftUV).xy;
        fftSlopeXZ *= 1.35;
        float slopeLen = length(fftSlopeXZ);
        if (slopeLen > 0.999) {
            fftSlopeXZ *= 0.999 / slopeLen;
        }
        float fftNy = sqrt(max(0.0, 1.0 - dot(fftSlopeXZ, fftSlopeXZ)));
        vec3 fftNormal = normalize(vec3(fftSlopeXZ.x, max(fftNy, 0.001), fftSlopeXZ.y));
        waveNormal = fftNormal;
        float fftSlope = clamp(1.0 - fftNormal.y, 0.0, 1.0);
        foam = smoothstep(max(foam_threshold, 0.05), 1.0, fftSlope * 2.0 + abs(fftHeight) * 0.25);
    } else {
        evaluateWaterGerstner(hitPos, time, wave_speed, wave_strength, wave_freq,
                              waveNormal, foam);
    }
    float foamSignal = foam;

    // ── Micro-detail capillary ripples ──────────────────────────
    if (micro_strength > 0.001) {
        // micro_scale is authored in world space. Do not scale it by the FFT
        // ocean tile size, or large water surfaces lose their capillary detail.
        float sc = max(micro_scale, 0.001);
        float wind_dx = cos(wind_direction);
        float wind_dz = sin(wind_direction);
        float cross_dx = -wind_dz;
        float cross_dz = wind_dx;
        float base_speed = sqrt(max(1.0, wind_speed)) * max(micro_anim_speed, 0.001);
        float morph = max(micro_morph_speed, 0.001);

        float off1_x = wind_dx * time * base_speed + sin(time * 0.3 * morph) * 0.5;
        float off1_z = wind_dz * time * base_speed + cos(time * 0.2 * morph) * 0.5;
        vec2 p1 = vec2(hitPos.x * sc + off1_x, hitPos.z * sc + off1_z);

        float off2_x = (wind_dx * 0.7 + cross_dx * 0.3) * time * base_speed * 0.6 + cos(time * 0.15 * morph + 1.5) * 0.8;
        float off2_z = (wind_dz * 0.7 + cross_dz * 0.3) * time * base_speed * 0.6 + sin(time * 0.25 * morph + 2.0) * 0.8;
        vec2 p2 = vec2(hitPos.x * sc * 0.5 + off2_x, hitPos.z * sc * 0.5 + off2_z);

        float off3_x = cross_dx * time * base_speed * 0.4 + sin(time * 0.5 * morph + 3.0) * 0.3;
        float off3_z = cross_dz * time * base_speed * 0.4 + cos(time * 0.4 * morph + 1.0) * 0.3;
        vec2 p3 = vec2(hitPos.x * sc * 2.0 + off3_x, hitPos.z * sc * 2.0 + off3_z);

        const float dx = 0.01;
        float h1_c = water_fbm(p1);
        float h1_x = water_fbm(p1 + vec2(dx,0.0));
        float h1_z = water_fbm(p1 + vec2(0.0,dx));
        float h2_c = water_fbm(p2);
        float h2_x = water_fbm(p2 + vec2(dx,0.0));
        float h2_z = water_fbm(p2 + vec2(0.0,dx));
        float h3_c = water_noise(p3);
        float h3_x = water_noise(p3 + vec2(dx,0.0));
        float h3_z = water_noise(p3 + vec2(0.0,dx));

        float hc = h1_c * 0.5 + h2_c * 0.35 + h3_c * 0.15;
        float hx = h1_x * 0.5 + h2_x * 0.35 + h3_x * 0.15;
        float hz = h1_z * 0.5 + h2_z * 0.35 + h3_z * 0.15;

        float dsdx = (hx - hc) / dx;
        float dsdz = (hz - hc) / dx;
        float microGain = 1.0;
        vec3 microN = normalize(vec3(-dsdx * micro_strength * microGain, 1.0, -dsdz * micro_strength * microGain));
        waveNormal  = normalize(waveNormal + microN);

        // Micro-peak foam (replaces/supplements Gerstner foam for FFT-style look)
        float microSlope = clamp(hc * 0.5 + 0.5, 0.0, 1.0);
        float scaledFoamNoise = max(foam_noise_scale, 0.001);
        float foamBreakup = water_fbm(vec2(hitPos.x * scaledFoamNoise + off1_x * 0.5,
                                           hitPos.z * scaledFoamNoise + off1_z * 0.5)) * 0.5 + 0.5;
        float microFoam  = clamp((microSlope + (foamBreakup - 0.5) * 0.35 - foam_threshold) * 5.0, 0.0, 1.0);
        foamSignal = max(foamSignal, microFoam);
    }

    // ── Build shading normal from wave perturbation ──────────────
    // waveNormal lives in a y-up tangent frame; project onto geoNormal's ONB
    vec3 tgt, btgt;
    buildONB(geoNormal, tgt, btgt);
    vec3 shadingNormal = normalize(tgt * waveNormal.x + geoNormal * waveNormal.y + btgt * waveNormal.z);
    if (dot(shadingNormal, -rayDir) < 0.0) shadingNormal = geoNormal; // sanity

    float maxProbeDepth = max(depth_max, 0.1);
    float waterDepth = maxProbeDepth;
    vec3 floorPosition = hitPos - vec3(0.0, waterDepth, 0.0);
    bool foundFloor = false;
    if (shore_foam_intensity > 0.01 || absorption_density > 0.01 || caustic_intensity > 0.01) {
        foundFloor = estimateWaterDepthGL(hitPos, maxProbeDepth, waterDepth, floorPosition);
    }
    vec3 baseWaterColor = calculateDepthColorGL(waterDepth, depth_max, shallow_color, deep_color);
    if (caustic_intensity > 0.01 && foundFloor) {
        float causticVal = calculateWaterCausticsGL(floorPosition, time, caustic_scale, caustic_speed);
        float causticFade = exp(-waterDepth * absorption_density * 0.5);
        baseWaterColor += shallow_color * causticVal * caustic_intensity * causticFade;
    }
    float shoreFoam = 0.0;
    if (shore_foam_intensity > 0.01 && foundFloor) {
        shoreFoam = calculateShoreFoamGL(waterDepth, shore_foam_distance, shore_foam_intensity, hitPos, time, max(foam_noise_scale, 0.001));
    }
    float totalFoam = min(foamSignal * foam_level + shoreFoam, 1.0);
    float cosNV = 0.0;

    // ── Depth-based color blend ──────────────────────────────────
    float tDepth   = 1.0 - cosNV;  // grazing angle → deeper look

    // ── Blend foam (white crest) ─────────────────────────────────
    // CPU parity: foam/depth color is for visible BRDF/direct albedo only.
    // PrincipledBSDF::scatter sends the constant water material albedo
    // (deep_color) into Dielectric, so refraction tint must not vary with foam.
    vec3 transmissionTint = deep_color;
    bool waterFrontFace = dot(rayDir, shadingNormal) < 0.0;
    scatterGlass(hitPos, shadingNormal, shadingNormal, waterFrontFace, rayDir, transmissionTint, ior, mix(roughness, 0.8, totalFoam), 0.0, vec3(1.0), seed);
}

// ============================================================
// Main — Closest Hit Entry Point
// ============================================================
void main() {
    // ----------------------------------------------------------
    // 1. Instance & materyal verisi
    // ----------------------------------------------------------
    VkInstanceData   inst = instances.i[gl_InstanceID];
    VkGeometryData   geo  = geometries.g[inst.blasIndex];
    uint matIndex = inst.materialIndex;
    if (geo.materialAddr != 0ul) {
        MaterialIndexBuffer mi = MaterialIndexBuffer(geo.materialAddr);
        matIndex = mi.m[uint(gl_PrimitiveID)];
    }
    Material         mat  = materials.m[matIndex];

    // ----------------------------------------------------------
    // 2. Vertex & Index Verilerini Çekip Gerçek Yüzey Normalini Bul
    // ----------------------------------------------------------
    uint i0, i1, i2;
    if (geo.indexAddr != 0) {
        IndexBuffer iBuf = IndexBuffer(geo.indexAddr);
        i0 = iBuf.i[gl_PrimitiveID * 3 + 0];
        i1 = iBuf.i[gl_PrimitiveID * 3 + 1];
        i2 = iBuf.i[gl_PrimitiveID * 3 + 2];
    } else {
        i0 = uint(gl_PrimitiveID) * 3 + 0;
        i1 = uint(gl_PrimitiveID) * 3 + 1;
        i2 = uint(gl_PrimitiveID) * 3 + 2;
    }

    vec3 worldNormal;
    vec3 geomNormalRaw;
    vec3 geomNormal;
    vec3 objV0 = vec3(0.0);
    vec3 objV1 = vec3(0.0);
    vec3 objV2 = vec3(0.0);
    vec2 uv0 = vec2(0.0);
    vec2 uv1 = vec2(0.0);
    vec2 uv2 = vec2(0.0);

    if (geo.vertexAddr != 0) {
        VertexBuffer vBuf = VertexBuffer(geo.vertexAddr);
        objV0 = vBuf.v[i0];
        objV1 = vBuf.v[i1];
        objV2 = vBuf.v[i2];
        vec3 localFaceNormal = normalize(cross(objV1 - objV0, objV2 - objV0));
        geomNormalRaw = normalize(vec3(localFaceNormal * mat3(gl_WorldToObjectEXT)));
    } else {
        geomNormalRaw = normalize(vec3(0, 1, 0));  // Fallback
    }

    vec3 bary = vec3(1.0 - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);

    if (geo.normalAddr != 0) {
        NormalBuffer nBuf = NormalBuffer(geo.normalAddr);
        vec3 localNormal = nBuf.n[i0] * bary.x
                         + nBuf.n[i1] * bary.y
                         + nBuf.n[i2] * bary.z;

        // Object → world dönüşümü (ölçeği yok saymak için: inverse transpose)
        worldNormal = normalize(vec3(localNormal * mat3(gl_WorldToObjectEXT)));
    } else {
        // Normal buffer yoksa ham üçgen normalini kullan
        worldNormal = geomNormalRaw;
    }

    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 rayDir = normalize(gl_WorldRayDirectionEXT);

    // Compute UV coordinates if available
    vec2 hitUV = vec2(0.0);
    if (geo.uvAddr != 0) {
        UVBuffer uvBuf = UVBuffer(geo.uvAddr);
        uv0 = uvBuf.u[i0];
        uv1 = uvBuf.u[i1];
        uv2 = uvBuf.u[i2];
        hitUV = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    }

    // Vulkan shader coordinate origin differs; flip V to match OptiX (and texture upload)
    hitUV.y = 1.0 - hitUV.y;

    bool surfaceFrontFace = dot(geomNormalRaw, rayDir) < 0.0;
    geomNormal = geomNormalRaw;

    if (!surfaceFrontFace) {
        worldNormal = -worldNormal;
    }
    if (!surfaceFrontFace) {
        geomNormal = -geomNormalRaw;
    }

    // TBN must use ORIGINAL mesh UVs (not texture-sampling-flipped UVs).
    // Normal maps are authored in the original UV parameterisation; flipping V
    // here would negate the bitangent and invert the Green channel → bumps↔dents.
    vec3 surfaceTangent;
    vec3 surfaceBitangent;
    buildSurfaceTBN(objV0, objV1, objV2, uv0, uv1, uv2, worldNormal, surfaceTangent, surfaceBitangent);

    // ----------------------------------------------------------
    // 2b. Terrain Splat-Layer Blending (FLAG_TERRAIN = bit 16)
    // Blends up to 4 material layers using an RGBA splat map.
    // Replaces albedo, roughness and metallic with weighted blend.
    // ----------------------------------------------------------
    const uint FLAG_TERRAIN = (1u << 16);
    if ((mat.flags & FLAG_TERRAIN) != 0u) {
        uint layerIdx = mat._terrain_layer_idx;
        VkTerrainLayerData tl = terrainLayers.d[layerIdx];
        if (tl.splat_map_tex > 0u && tl.layer_count > 0u) {
            // Sample RGBA splat map — R=layer0, G=layer1, B=layer2, A=layer3
            vec4 splatW = texture(materialTextures[nonuniformEXT(int(tl.splat_map_tex))], hitUV);
            float weights[4];
            weights[0] = splatW.r;
            weights[1] = splatW.g;
            weights[2] = splatW.b;
            weights[3] = splatW.a;

            // Normalize so weights sum to 1
            float totalW = weights[0] + weights[1] + weights[2] + weights[3];
            if (totalW < 0.001) totalW = 1.0;
            for (int k = 0; k < 4; k++) weights[k] /= totalW;

            vec3  blendAlbedo    = vec3(0.0);
            float blendRoughness = 0.0;
            float blendMetallic  = 0.0;
            float blendTransmission = 0.0;
            float blendIor = 0.0;
            // [FIX] Accumulate per-layer normal maps in tangent space
            vec3  blendNormal_ts = vec3(0.0);  // weighted sum of tangent-space normals
            bool  anyNormalTex   = false;

            uint activeCount = min(tl.layer_count, 4u);
            for (uint k = 0u; k < activeCount; k++) {
                if (weights[k] < 0.001) continue;
                Material lm = materials.m[tl.layer_mat_id[k]];
                vec2 layerUV = hitUV * tl.layer_uv_scale[k];
                layerUV = applyMaterialUVTransform(lm, layerUV);

                // Layer albedo
                vec3 lAlbedo = max(vec3(lm.albedo_r, lm.albedo_g, lm.albedo_b), vec3(0.0));
                if (int(lm.albedo_tex) > 0) {
                    lAlbedo = texture(materialTextures[nonuniformEXT(int(lm.albedo_tex))], layerUV).rgb;
                }
                blendAlbedo += weights[k] * lAlbedo;

                // Layer roughness — terrain layers use the same per-material flag bits
                // as the parent terrain material; channel selection follows pbr_texture_policy.
                float lRough = clamp(lm.roughness, 0.0, 1.0);
                if (int(lm.roughness_tex) > 0) {
                    lRough = samplePackedRoughness(
                        texture(materialTextures[nonuniformEXT(int(lm.roughness_tex))], layerUV),
                        0.0, lm.flags);
                }
                blendRoughness += weights[k] * lRough;

                // Layer metallic
                float lMetal = clamp(lm.metallic, 0.0, 1.0);
                if (int(lm.metallic_tex) > 0) {
                    lMetal = samplePackedMetallic(
                        texture(materialTextures[nonuniformEXT(int(lm.metallic_tex))], layerUV),
                        lm.flags);
                }
                blendMetallic += weights[k] * lMetal;

                float lTransmission = clamp(lm.transmission, 0.0, 1.0);
                if (int(lm.transmission_tex) > 0) {
                    lTransmission = texture(materialTextures[nonuniformEXT(int(lm.transmission_tex))], layerUV).r;
                }
                blendTransmission += weights[k] * lTransmission;
                blendIor += weights[k] * max(lm.ior, 1.0);

                // Layer normal map — blend tangent-space normals by weight.
                // Keep channel orientation aligned with OptiX path (no ad-hoc X/Y flips).
                // Layers without a normal map contribute a flat (0,0,1) tangent-space vector.
                if (int(lm.normal_tex) > 0) {
                    vec3 ns = decodeNormalMapSample(
                        texture(materialTextures[nonuniformEXT(int(lm.normal_tex))], layerUV).rgb,
                        lm.flags);
                    ns.x *= lm.normal_strength;
                    ns.y *= lm.normal_strength;
                    blendNormal_ts += weights[k] * ns;
                    anyNormalTex = true;
                } else {
                    blendNormal_ts += weights[k] * vec3(0.0, 0.0, 1.0); // flat contribution
                }
            }

            // Override local mat copy with blended values
            mat.albedo_r   = blendAlbedo.r;
            mat.albedo_g   = blendAlbedo.g;
            mat.albedo_b   = blendAlbedo.b;
            mat.roughness  = blendRoughness;
            mat.metallic   = blendMetallic;
            mat.transmission = clamp(blendTransmission, 0.0, 1.0);
            mat.ior = (blendIor > 0.01) ? blendIor : mat.ior;
            // Clear per-material texture slots — blending already resolved them above
            mat.albedo_tex    = 0u;
            mat.roughness_tex = 0u;
            mat.metallic_tex  = 0u;
            mat.transmission_tex = 0u;

            // [FIX] Apply blended normal map to world-space normal immediately.
            // Set mat.normal_tex = 0 so the standard normal-map section below does nothing.
            if (anyNormalTex) {
                vec3 nts = normalize(blendNormal_ts);
                vec3 perturbed = normalize(
                    surfaceTangent * nts.x +
                    surfaceBitangent * nts.y +
                    worldNormal * nts.z
                );
                // Only use perturbed normal if it faces the ray (sanity check)
                if (dot(perturbed, -rayDir) > 0.0) worldNormal = perturbed;
                mat.normal_tex = 0u; // prevent double-application in section 4
            }
        }
    }

    // ----------------------------------------------------------
    // 3. Materyal parametreleri
    // ----------------------------------------------------------
    vec3  albedo      = max(vec3(mat.albedo_r, mat.albedo_g, mat.albedo_b), vec3(0.0));
    vec3  emColor     = vec3(mat.emission_r, mat.emission_g, mat.emission_b);
    float emStrength  = max(mat.emission_strength, 0.0);
    float roughness   = clamp(mat.roughness, 0.0, 1.0);
    float metallic    = clamp(mat.metallic, 0.0, 1.0);
    float specular    = clamp(mat.specular, 0.0, 1.0);
    float ior         = (mat.ior > 0.01) ? mat.ior : 1.5;
    float transmission = clamp(mat.transmission, 0.0, 1.0);
    vec2 materialUV = applyMaterialUVTransform(mat, hitUV);

    // Procedural tile-break: perturb UV before any texture sampling.
    // Independent slider — set to 0 to keep albedo maps clean.
    if (mat.tile_break_strength > 0.0 &&
        (mat.albedo_tex > 0u || mat.roughness_tex > 0u || mat.normal_tex > 0u)) {
        materialUV = pd_tileBreak(materialUV, hitPos, mat.tile_break_strength);
    }

    // ----------------------------------------------------------
    // 4. Emission — ayrı field, scatter ile karışmaz
    //    payload.radiance ve hitEmissive, emission texture sampling
    //    SONRASI atanır (aşağıda) — texture'ın rengi override edebilmesi için
    // ----------------------------------------------------------

    // Sample albedo texture
    int albedoTexID = int(mat.albedo_tex);
    if (albedoTexID > 0) {
        albedo = texture(materialTextures[nonuniformEXT(albedoTexID)], materialUV).rgb;
    }
    
    // ----------------------------------------------------------------
    // OPACITY DECISION
    // ----------------------------------------------------------------
    // Two clean modes:
    //
    // Mode 1: opacity_tex is explicitly connected
    //   1a: opacity_tex == albedo_tex  → RGBA texture, opacity is in .a channel
    //   1b: opacity_tex != albedo_tex  → standalone grayscale mask, opacity is in .r channel
    //
    // Mode 2: No opacity_tex → mat.opacity for glass/transmission only
    // ----------------------------------------------------------------
    int opacityTexID = int(mat.opacity_tex);
    float finalAlpha = mat.opacity;
    
    if (opacityTexID > 0) {
        // --- MODE 1: Explicit opacity texture connected ---
        // Bit 8 of mat.flags is set by C++ if the opacity texture has RGBA alpha data.
        // Bit 8 clear = standalone grayscale mask → use R channel (luminance)
        // Bit 8 set   = RGBA texture with opacity in alpha → use A channel
        float maskValue;
        if ((mat.flags & 256u) != 0u) {
            maskValue = texture(materialTextures[nonuniformEXT(opacityTexID)], materialUV).a;
        } else {
            maskValue = texture(materialTextures[nonuniformEXT(opacityTexID)], materialUV).r;
        }
        finalAlpha *= maskValue;
    }
    
    if (finalAlpha < 0.1) {
        finalAlpha = 0.0;
    }
    
    if (finalAlpha < 0.99) {
        // Stochastic transparency matching OptiX `anyhit` implementation
        float randVal = rnd(payload.seed);
        if (randVal > finalAlpha) {
            // Transparent pixel — ray continues forward cleanly.
            // Push ray safely to the BACK side of the triangle using offset_ray.
            // geomNormal already faces the incoming ray (due to flip earlier).
            // So -geomNormal points to the other side of the surface.
            payload.radiance      = vec3(0.0);
            payload.hitEmissive   = false;
            payload.scatterOrigin = offset_ray(hitPos, -geomNormal);
            payload.scatterDir    = rayDir;
            payload.scattered     = true;
            payload.bounceType     = BOUNCE_TRANSPARENT;
            return;
        }
    }
    
    // ── Thin-shell BUBBLE (champagne / soda / soap-foam close-up) ──────────────
    // A bubble is a THIN dielectric film: light either Fresnel-reflects off the
    // shell (bright silver rim, strong at grazing) or passes STRAIGHT through (a
    // thin shell enters/exits parallel, no net refraction). Reads as a bright-rimmed
    // transparent sphere independent of the surrounding medium. bubble_ior drives the
    // rim Fresnel; bubble_film adds thin-film iridescence. Mirrors material_scatter.cuh.
    // Returns before emission so an emissive bubble material won't wash out the look.
    if ((mat.flags & MAT_FLAG_BUBBLE) != 0u) {
        // Use the SMOOTH (interpolated) shading normal, not the faceted geometric
        // normal, so the rim curves smoothly across a sphere (parity with OptiX N).
        vec3  Nb   = normalize(worldNormal);
        float cosT = min(abs(dot(rayDir, Nb)), 1.0);
        float bio  = (mat.bubble_ior > 1.0001) ? mat.bubble_ior : 1.33;
        float r0   = (1.0 - bio) / (1.0 + bio); r0 = r0 * r0;
        float fres = r0 + (1.0 - r0) * pow(1.0 - cosT, 5.0);
        vec3 dir, att;
        if (rnd(payload.seed) < fres) {
            dir = reflect(rayDir, Nb);                  // bright Fresnel rim
            if (mat.bubble_film > 1e-3) {
                float opd = mat.bubble_film * (1.0 / max(cosT, 0.15));
                att = vec3(0.55 + 0.45 * cos(opd * 6.2831853),
                           0.55 + 0.45 * cos(opd * 6.2831853 + 2.0944),
                           0.55 + 0.45 * cos(opd * 6.2831853 + 4.1888));
            } else {
                att = vec3(1.0);
            }
            payload.scatterOrigin = hitPos + Nb * RAY_OFFSET;
        } else {
            dir = rayDir;                               // straight pass-through (thin shell)
            att = vec3(0.85) + 0.15 * vec3(mat.albedo_r, mat.albedo_g, mat.albedo_b);
            payload.scatterOrigin = offset_ray(hitPos, -geomNormal);
        }
        payload.radiance            = vec3(0.0);
        payload.hitEmissive         = false;
        payload.scatterDir          = dir;
        payload.attenuation        *= att;
        payload.scattered           = true;
        payload.bounceType          = BOUNCE_SPECULAR;
        payload.primaryTransmission = 1.0;              // aerial parity (don't wash bubble)
        return;
    }

    // Opaque pixel — continue to shading normally

    // --- MODE 2: glass/transmission adjustment ---
    if (mat.opacity < 0.99 && metallic < 0.1 && transmission < 0.01) {
        transmission = 1.0 - mat.opacity;
    }


   // Sample emission texture
int emissionTexID = int(mat.emission_tex);
if (emissionTexID > 0) {
    vec3 emTex = texture(materialTextures[nonuniformEXT(emissionTexID)], materialUV).rgb;
    // Emission texture is authoritative; intensity remains controlled by emission strength.
    emColor = emTex;
} else if (emStrength > 0.001) {
    // Emission texture yok ama strength > 0 → albedo rengini kullan (Blender default)
    float matEmLum = dot(emColor, vec3(0.2126, 0.7152, 0.0722));
   // if (matEmLum < 0.01) {
   //     emColor = albedo; // albedo texture zaten yukarıda uygulandı
   // }
}
    // Texture sampling bitti — artık kesin emColor belli, radiance'ı şimdi ata
    payload.radiance = emColor * emStrength;
    payload.hitEmissive = (length(payload.radiance) > 0.001);
    
    // Sample transmission texture (for glass/transparent materials)
    int transmissionTexID = int(mat.transmission_tex);
    if (transmissionTexID > 0) {
        float trans = texture(materialTextures[nonuniformEXT(transmissionTexID)], materialUV).r;
        transmission = clamp(trans, 0.0, 1.0);
    }
    
    // Sample roughness texture
    int roughTexID = int(mat.roughness_tex);
    if (roughTexID > 0) {
        float r = samplePackedRoughness(
            texture(materialTextures[nonuniformEXT(roughTexID)], materialUV), 0.0, mat.flags);
        roughness = clamp(r, 0.0, 1.0);
    }

    // Sample metallic texture
    int metallicTexID = int(mat.metallic_tex);
    if (metallicTexID > 0) {
        float m = samplePackedMetallic(
            texture(materialTextures[nonuniformEXT(metallicTexID)], materialUV), mat.flags);
        metallic = clamp(m, 0.0, 1.0);
    }

    int specularTexID = int(mat.specular_tex);
    if (specularTexID > 0) {
        specular = clamp(texture(materialTextures[nonuniformEXT(specularTexID)], materialUV).r * specular, 0.0, 1.0);
    }

    // ── Procedural detail: subtle color variation + dirt + roughness ──────────
    // micro_detail_strength drives all world-space effects without touching UVs.
    // tile_break_strength (above) is the separate UV-warp control.
    if (mat.micro_detail_strength > 0.0) {
        float sc  = max(mat.micro_detail_scale, 0.5);
        float str = mat.micro_detail_strength;

        // Subtle world-space luminance variation — ±8% max, independent seed
        float colorVar   = pd_vnoise3(hitPos * sc * 0.7 + vec3(31.4, 17.2, 42.9));
        float colorDelta = (colorVar - 0.5) * 0.16 * str;
        albedo = clamp(albedo * (1.0 + colorDelta), vec3(0.0), vec3(1.0));

        // Dirt: fBm-based darkening (dust, grime, worn patches)
        float dirtFactor = pd_dirt(hitPos, sc) * str;
        vec3  dirtColor  = vec3(0.14, 0.10, 0.08);
        albedo = mix(albedo, albedo * dirtColor, dirtFactor);

        // Roughness micro-variation: breaks uniform-gloss appearance
        roughness = clamp(roughness + pd_roughnessVar(hitPos, sc) * str * 0.5,
                          0.0, 1.0);
    }

    vec3 weatherMacroNormal = worldNormal;

    // Apply normal map if present (perturb surface normal)
    int normalTexID = int(mat.normal_tex);
    bool isWaterMaterial = ((mat.flags & MAT_FLAG_WATER) != 0u) || mat.sheen > 0.001;
    bool waterUsesFFT = isWaterMaterial &&
                        ((mat.flags & MAT_FLAG_WATER_FFT_READY) != 0u) &&
                        mat.height_tex > 0u &&
                        mat.normal_tex > 0u &&
                        mat.fft_ocean_size > 0.001 &&
                        abs(mat.anisotropic) < 1e-5 &&
                        abs(mat.sheen_tint) < 1e-5;
    vec3 tangentNormal = worldNormal;  // Default to geometry normal
    if (normalTexID > 0 && !waterUsesFFT) {
        // Sample normal map (OpenGL format: RGB = normal direction).
        // BC5 cache only stores RG — decodeNormalMapSample reconstructs Z when
        // bit 11 is set; otherwise the .b channel from the source RGB is used.
        vec3 normalMapSample = texture(materialTextures[nonuniformEXT(normalTexID)], materialUV).rgb;

        // Validate against pure-black sample (RGB normals encode the rest-pose at
        // ~0.5,0.5,1.0 → length ≈ 1.22; BC5 with reconstructed Z is unit length
        // so length ≈ 1.0; both safely above the 0.1 floor).
        float mapLength = length(normalMapSample);
        if (mapLength > 0.1) {
            vec3 normalMapDir = decodeNormalMapSample(normalMapSample, mat.flags);
            normalMapDir.x *= mat.normal_strength;
            normalMapDir.y *= mat.normal_strength;
            
            // Normalize to ensure unit vector
            vec3 tangentSpaceNormal = normalize(normalMapDir);
            
            // Build orthonormal basis from geometry normal
            // Transform from tangent space to world space
            vec3 worldNormalPerturbed = normalize(
                surfaceTangent * tangentSpaceNormal.x +
                surfaceBitangent * tangentSpaceNormal.y +
                worldNormal * tangentSpaceNormal.z
            );
            
            // Ensure the perturbed normal points outward (away from ray origin)
            // rayDir is ray.direction (pointing away from origin)
            // Normal should point toward viewer (opposite of ray direction inside object)
            if (dot(worldNormalPerturbed, -rayDir) > 0.0) {
                tangentNormal = worldNormalPerturbed;
            }
            // else: keep geometry normal if perturbed normal points wrong way
        }
    }

    vec3 weatherSupportNormal = safeNormalize(mix(weatherMacroNormal, tangentNormal, 0.85), weatherMacroNormal);
    applyWeatherSurface(hitPos, tangentNormal, weatherSupportNormal, albedo, roughness, metallic);
    roughness = clamp(roughness, 0.0, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);

    if (payload.primaryHit == 0u) {
        payload.primaryAlbedo = albedo;
        payload.primaryNormal = worldNormal;
        payload.primaryHit = 1u;
        payload.primaryTransmission = transmission;
        payload.primaryMetallic = metallic;
        payload.primaryMaterialId = matIndex;   // Stylize AOV: real material boundary for outlines
    }
    worldNormal = tangentNormal;
    worldNormal = weatherSurfaceNormal(hitPos, worldNormal, weatherSupportNormal);

    // ----------------------------------------------------------
    // IS_WATER fast path. Prefer the explicit material flag, keep sheen as legacy fallback.
    // Water has its own scatter: Gerstner waves + glass refraction.
    // Must run BEFORE transmission/direct-lighting/diffuse paths.
    // ----------------------------------------------------------
    if (isWaterMaterial) {
        scatterWater(
            hitPos, worldNormal, rayDir,
            /*wave_speed*/     mat.anisotropic,
            /*wave_strength*/  mat.sheen,
            /*wave_freq*/      mat.sheen_tint,
            /*foam_level*/     mat.translucent,
            /*foam_threshold*/ mat.foam_threshold,
            /*micro_strength*/ mat.micro_detail_strength,
            /*micro_scale*/    mat.micro_detail_scale,
            /*micro_anim*/     mat.micro_anim_speed,
            /*micro_morph*/    mat.micro_morph_speed,
            /*foam_noise*/     mat.foam_noise_scale,
            /*wind_dir*/       mat.fft_wind_direction,
            /*wind_speed*/     mat.fft_wind_speed,
            /*fft_time_scale*/ mat.fft_time_scale,
            /*fft_ocean_size*/ mat.fft_ocean_size,
            /*fft_height_tex*/ 0u,
            /*fft_normal_tex*/ 0u,
            /*depth_max*/      mat.subsurface_amount * 100.0,
            /*absorption*/     mat.subsurface_scale,
            /*shore_dist*/     mat.subsurface_radius_r,
            /*shore_int*/      mat.clearcoat,
            /*caustic_int*/    mat.clearcoat_roughness,
            /*caustic_scale*/  mat.subsurface_radius_g,
            /*caustic_speed*/  mat.subsurface_anisotropy,
            /*shallow_color*/  vec3(mat.emission_r, mat.emission_g, mat.emission_b),
            /*deep_color*/     vec3(mat.albedo_r,   mat.albedo_g,   mat.albedo_b),
            /*ior*/            (mat.ior > 0.01) ? mat.ior : 1.333,
            /*roughness*/      clamp(mat.roughness, 0.0, 0.15),
            payload.seed
        );
        return;
    }

    // ----------------------------------------------------------
    // Stochastic Principled Transmission (Glass)
    // Evaluated before Direct Lighting to prevent mismatched diffuse/GGX specular highlights.
    // OptiX-like probablilistic branching based on transmission weight.
    // ----------------------------------------------------------
    vec3 directAttenuation = payload.attenuation;
    vec3  resinColor = vec3(mat.resin_color_r, mat.resin_color_g, mat.resin_color_b);

    // Carried into the NEE block below so direct light reaching the base also gets
    // absorbed on its ENTRY path through the resin (at the light's angle).
    bool  resinActive  = false;
    vec3  resinExt     = vec3(0.0);
    float resinDensity = 0.0;

    if (mat.transmission_density > 1e-4) {
        // RESIN: a refractive ABSORBING layer over an OPAQUE base. Fresnel-split the
        // surface — the reflection lobe is the glossy resin top (specular, skips NEE);
        // light that enters reaches the base, which we tint by the coat absorption over
        // the thickness and shade as a normal diffuse surface, so the base gets full
        // direct lighting (NEE) + indirect (deeper, cleaner). A small base extinction
        // (0.25) makes Resin Depth darken even for a white resin; resinColor tints.
        float effIor = max(ior, 1.45);
        float cosT   = clamp(dot(-rayDir, worldNormal), 0.0, 1.0);
        float fres   = schlickFresnel(cosT, effIor);
        // Coat gloss is the resin LAYER's own roughness, independent of the base.
        float resinRough = clamp(mat.resin_roughness, 0.0, 1.0);
        if (rnd(payload.seed) < fres) {
            vec3 V = -rayDir;
            vec3 refl;
            if (resinRough < 0.02) {
                refl = reflect(rayDir, worldNormal);
            } else {
                // ggxSampleVNDF returns the REFLECTED direction directly (matches
                // OptiX + scatterMetal). The old ggxSampleHemisphere path double-
                // reflected (it already returns L, not a half-vector) → tiny/garbage
                // highlight that ignored roughness on the resin surface.
                float alpha = max(resinRough * resinRough, 1e-4);
                refl = ggxSampleVNDF(worldNormal, V, alpha, rnd(payload.seed), rnd(payload.seed));
                if (dot(refl, worldNormal) <= 0.0) refl = reflect(rayDir, worldNormal);
            }
            payload.scatterOrigin = offset_ray(hitPos, worldNormal);
            payload.scatterDir    = normalize(refl);
            payload.attenuation  *= vec3(1.0);
            payload.scattered     = true;
            payload.bounceType     = BOUNCE_RESIN; // capped by raygen resin budget
            return;
        }
        // Base under the resin: absorb over the thickness (in + out), then shade as an
        // opaque diffuse surface → falls through to direct lighting (NEE) + BRDF below.
        vec3  ct      = clamp(resinColor, vec3(0.0), vec3(1.0));
        float cosV    = max(abs(cosT), 0.25);
        vec3  ext     = (vec3(1.0) - ct) + vec3(0.25);

        // --- Resin INTERNAL inclusions (Phase 1) -----------------------------------
        // March the refracted ray through the resin thickness (no scene rays — pure
        // procedural sampling): dust = heterogeneous absorption at depth, dirt = opaque
        // worley specks that terminate early (their colour shows through the resin
        // already crossed), and the refracted lateral travel offsets the base lookup
        // (parallax). Falls back to the cheap analytic path when inclusions are off.
        bool resinHasInclusions = (mat.resin_inclusion > 0.001 || mat.resin_dirt > 0.001);
        if (resinHasInclusions) {
            vec3 Tdir = refract(rayDir, worldNormal, 1.0 / effIor);
            if (dot(Tdir, Tdir) < 1e-6) Tdir = rayDir;        // total internal reflection fallback
            Tdir = normalize(Tdir);
            const int RESIN_STEPS = 6;
            float dt  = max(mat.transmission_density, 1e-3) / float(RESIN_STEPS);
            float scl = max(mat.resin_inclusion_scale, 0.01);
            vec3  P      = hitPos;
            vec3  absorb = vec3(1.0);
            bool  dirtHit = false;
            for (int i = 0; i < RESIN_STEPS; ++i) {
                P += Tdir * dt;
                // Dust cloudiness → extra extinction at this depth. Power curve makes
                // it sparse/wispy (mostly clear, occasional dense cores) not flat haze.
                float dust     = rh_fbm(P * scl);
                dust           = pow(dust, 2.0);
                float localExt = 1.0 + mat.resin_inclusion * dust * 6.0;
                absorb *= exp(-dt * ext * localExt);
                // Dirt specks → opaque early return (ray stops inside the resin).
                if (mat.resin_dirt > 0.001) {
                    float cell  = rh_worley(P * scl * 3.0);            // 0 at speck centres
                    float speck = 1.0 - smoothstep(0.0, 0.18, cell);  // 1 inside a speck
                    if (speck * mat.resin_dirt > 0.5) { dirtHit = true; break; }
                }
            }
            if (dirtHit) {
                // Terminate on the dirt speck: its colour, dimmed by the resin crossed.
                albedo = vec3(mat.resin_dirt_color_r, mat.resin_dirt_color_g, mat.resin_dirt_color_b) * absorb;
            } else {
                // Reached the base: parallax-offset the base lookup along the refracted
                // lateral travel, then apply the accumulated (heterogeneous) absorption.
                vec3 inPlane = Tdir - worldNormal * dot(Tdir, worldNormal);
                vec2 parUV = materialUV
                           + vec2(dot(inPlane, surfaceTangent), dot(inPlane, surfaceBitangent))
                             * (mat.transmission_density * 0.05);
                if (albedoTexID > 0) albedo = texture(materialTextures[nonuniformEXT(albedoTexID)], parUV).rgb;
                albedo *= absorb;
            }
        } else {
            float pathLen = 2.0 * mat.transmission_density / cosV;
            albedo       *= exp(-pathLen * ext);
        }
        roughness     = 1.0;
        metallic      = 0.0;
        transmission  = 0.0;
        // Hand the absorption to the NEE block so direct light entering the resin is
        // also attenuated by its own (light-angle) path length, not just the albedo tint.
        resinActive   = true;
        resinExt      = ext;
        resinDensity  = mat.transmission_density;
        // (no return — direct lighting + diffuse BRDF below shade the tinted base)
    }
    else if (transmission > 0.01) {
        if (rnd(payload.seed) < transmission) {
            // NOTE: glass-marble FULL VOLUME (real-interior medium march, MAT_FLAG_MARBLE_VOLUME)
            // was disabled — it was too camera-angle dependent and the interior dust/dirt never
            // read as intended. The flag + serialize fields are kept dormant (saved scenes load
            // fine); inclusion-bearing glass now always uses the shell march below.
            bool inclusionsOn = (mat.resin_inclusion > 0.001 || mat.resin_dirt > 0.001);
            // GLASS MARBLE (shell): when inclusions are enabled on a GLASS base, march the
            // refracted ray through the interior — dust (haze) + dirt specks (opaque
            // early-return) — BEFORE refracting through, so light still passes through
            // (real see-through glass) but picks up volumetric internal structure.
            // Independent of the resin coat (that path forces an opaque base). No extra
            // scene rays: the march is procedural; scatterGlass does the real refraction.
            if (inclusionsOn) {
                vec3 Tg = refract(rayDir, worldNormal, surfaceFrontFace ? (1.0 / ior) : ior);
                if (dot(Tg, Tg) < 1e-6) Tg = rayDir;            // total internal reflection fallback
                Tg = normalize(Tg);
                float cosIn = max(abs(dot(Tg, -worldNormal)), 0.05);
                const int GMS = 6;
                float gdt  = (0.65 / cosIn) / float(GMS);       // matches scatterGlass thickness model
                float gscl = max(mat.resin_inclusion_scale, 0.01);
                vec3  Pg = hitPos;
                vec3  gabsorb = vec3(1.0);
                bool  gdirt = false;
                for (int i = 0; i < GMS; ++i) {
                    Pg += Tg * gdt;
                    float dust = rh_fbm(Pg * gscl);
                    dust = pow(dust, 2.0);                                      // sparse wispy cores
                    gabsorb *= exp(-gdt * (mat.resin_inclusion * dust * 6.0));  // grey dust haze
                    if (mat.resin_dirt > 0.001) {
                        float cell  = rh_worley(Pg * gscl * 3.0);
                        float speck = 1.0 - smoothstep(0.0, 0.18, cell);
                        if (speck * mat.resin_dirt > 0.5) { gdirt = true; break; }
                    }
                }
                if (gdirt) {
                    // Opaque dirt speck suspended in the glass: shade as a lit diffuse
                    // point (colour dimmed by the glass crossed) → fall through to NEE.
                    albedo = vec3(mat.resin_dirt_color_r, mat.resin_dirt_color_g, mat.resin_dirt_color_b) * gabsorb;
                    roughness = 1.0; metallic = 0.0; transmission = 0.0;
                    // (no return — direct lighting + diffuse BRDF below shade the speck)
                } else {
                    // Hazy clear glass: refract through carrying the dust absorption.
                    scatterGlass(hitPos, worldNormal, worldNormal, surfaceFrontFace, rayDir, albedo * gabsorb, ior, roughness, 0.0, vec3(1.0), payload.seed);
                    return;
                }
            } else {
                // Chosen transmission path - act as Glass
                scatterGlass(hitPos, worldNormal, worldNormal, surfaceFrontFace, rayDir, albedo, ior, roughness, 0.0, vec3(1.0), payload.seed);
                return; // Immediately return, skipping direct lighting (Next Event Estimation)
            }
        } else {
            // Chosen base path (diffuse/metal), compensate probability weight
            payload.attenuation *= (1.0 / max(1.0 - transmission, 0.01));
            transmission = 0.0;
        }
    }

    // ----------------------------------------------------------
    // Direct lighting (one light sample, MIS with BRDF pdf)
    // ----------------------------------------------------------
   // Direct lighting scope
    {
        float pdf_select = 0.0;
        int lightIdx = pick_smart_light_gl(uvec2(0), hitPos, pdf_select);
        if (lightIdx >= 0) {
            float ru = rnd(payload.seed);
            float rv = rnd(payload.seed);
            vec3 wi; float dist; float lightAtten;
            bool ok = sample_light_direction_gl(lights.l[lightIdx], hitPos, ru, rv, wi, dist, lightAtten);
            if (ok) {
                if (length(wi) <= 1e-6) {
                    // Degenerate sample, skip
                } else {
                    wi = normalize(wi);
                    float NdotL = max(dot(worldNormal, wi), 0.0);
                    if (NdotL > 1e-6) {
                        // Use a dedicated shadow payload so the main path payload isn't overwritten by shadow traversal
                        // Conservative init: assume blocked. shadow_miss.rmiss (missIndex=1) sets false on escape.
                        // any-hit with ignoreIntersectionEXT lets ray continue → eventually misses → shadow_miss → false
                        // any-hit with terminateRayEXT → stays true (opaque shadow)
                        // SkipClosestHit: geometry hit without opacity test → stays true (solid shadow)
                        shadowOccluded = true;
                        // ULP-based offset: self-intersection-safe on thin/distant geometry
                        const uint FLAG_TERRAIN = (1u << 16);
                        bool useTerrainShadowNormal = (mat.flags & FLAG_TERRAIN) != 0u;
                        vec3 shadowNormal = useTerrainShadowNormal
                            ? safeNormalize(geomNormal, vec3(0.0, 1.0, 0.0))
                            : safeNormalize(worldNormal, vec3(0.0, 1.0, 0.0));
                        vec3 shadowOrigin = hitPos + shadowNormal * SHADOW_TMIN;
                        float tmin = SHADOW_TMIN;
                        float tmax = min(max(0.0, dist - 1e-3), 10000.0);
                        if (tmax > tmin) {
                            // No OpaqueEXT → any-hit shader tests transparency per pixel
                            // SkipClosestHit → no closest-hit overhead; shadow value set by any-hit/miss only
                            // missIndex=1 → shadow_miss.rmiss sets shadowOccluded=false when ray escapes
                            uint shadowFlags = gl_RayFlagsTerminateOnFirstHitEXT
                                             | gl_RayFlagsSkipClosestHitShaderEXT;
                            // mask 0x01 = triangles only — volume AABBs have mask 0x02 so
                            // they are invisible to shadow rays and cannot cast hard shadows.
                            traceRayEXT(topLevelAS, shadowFlags, 0x01, 0, 1, 1, shadowOrigin, tmin, wi, tmax, 1);
                        }
                        float shadowVisibility = shadowOccluded ? 0.0 : 1.0;
                        if (shadowVisibility > 1e-4) {
                            // Volumetric soft shadow: march through any volume AABB between surface and light.
                            // cam.pad0 carries float(volumeCount) from C++ renderProgressive each frame.
                            float volShadowTr = computeVolumeShadowTransmittance(shadowOrigin, wi, tmax);
                            vec3 V = normalize(-rayDir);
                            vec3 brdf = evaluate_brdf_gl(worldNormal, V, wi, albedo, roughness, metallic, specular, transmission);
                            vec3 Li = lights.l[lightIdx].color.rgb * lights.l[lightIdx].color.a * lightAtten;

                            int ltype = int(lights.l[lightIdx].position.w + 0.5);
                            bool isDelta = (ltype == 0 || ltype == 1); // point or directional

                            vec3 contrib;
                            if (isDelta) {
                                // Delta ışıklar (point, directional/güneş) için MIS uygulanmaz.
                                // Tek örnekleme yolu light side olduğundan w = 1.
                                // Estimator: brdf * Li * NdotL / pdf_select
                                float invPdf = 1.0 / max(pdf_select, 1e-6);
                                contrib = brdf * Li * NdotL * invPdf;
                            } else {
                                // Alan/spot ışıklar için tam MIS
                                float pdf_light_area = compute_light_pdf_gl(lights.l[lightIdx], dist, 1.0);
                                float pdf_light_total = pdf_light_area * pdf_select;
                                float pdf_brdf = pdf_brdf_gl(worldNormal, V, wi, roughness);
                                float w = power_heuristic(pdf_light_total, pdf_brdf);
                                float invPdf = 1.0 / max(pdf_light_total, 1e-6);
                                contrib = brdf * Li * NdotL * w * invPdf;
                            }
                            contrib = max(contrib, vec3(0.0));
                            contrib.x = isnan(contrib.x) ? 0.0 : (isinf(contrib.x) ? (contrib.x > 0.0 ? 1e4 : 0.0) : contrib.x);
                            contrib.y = isnan(contrib.y) ? 0.0 : (isinf(contrib.y) ? (contrib.y > 0.0 ? 1e4 : 0.0) : contrib.y);
                            contrib.z = isnan(contrib.z) ? 0.0 : (isinf(contrib.z) ? (contrib.z > 0.0 ? 1e4 : 0.0) : contrib.z);
                            contrib = min(contrib, vec3(1e4));
                            // Resin: the direct light also travels through the coat to reach
                            // the base. Absorb it over its ENTRY path (light-angle slant),
                            // so thick/tinted resin visibly dims direct lighting too.
                            if (resinActive) {
                                float cosL = max(NdotL, 0.05);
                                contrib *= exp(-(resinDensity / cosL) * resinExt);
                            }
                            // Apply volumetric transmittance (soft shadow from volumes)
                            contrib *= volShadowTr * shadowVisibility;

                            vec3 att = max(directAttenuation, vec3(0.0));
                            att.x = isnan(att.x) ? 0.0 : (isinf(att.x) ? (att.x > 0.0 ? 1e2 : 0.0) : att.x);
                            att.y = isnan(att.y) ? 0.0 : (isinf(att.y) ? (att.y > 0.0 ? 1e2 : 0.0) : att.y);
                            att.z = isnan(att.z) ? 0.0 : (isinf(att.z) ? (att.z > 0.0 ? 1e2 : 0.0) : att.z);
                            payload.radiance += att * contrib;
                        }
                    }       // ← NdotL if
                }           // ← else (length check)
            }               // ← ok if
        }                   // ← lightIdx if
    }                       // ← direct lighting scope

    // ----------------------------------------------------------
    // Nishita direct sun lighting is intentionally disabled here.
    // Direct sun contribution must come from scene Directional lights only;
    // sky sun intensity remains handled by the miss/sky path.
    // ----------------------------------------------------------
    if (false && worldData.w.mode == 2 && worldData.w.sunIntensity > 1e-4) {
        vec3 sunDir = normalize(worldData.w.sunDir);
        float NdotSun = max(dot(worldNormal, sunDir), 0.0);
        if (NdotSun > 1e-6) {
            shadowOccluded = true;
            // ULP-based offset: self-intersection-safe on thin/distant geometry
            const uint FLAG_TERRAIN = (1u << 16);
            bool useTerrainShadowNormal = (mat.flags & FLAG_TERRAIN) != 0u;
            vec3 sunShadowNormal = useTerrainShadowNormal
                ? safeNormalize(geomNormal, vec3(0.0, 1.0, 0.0))
                : safeNormalize(worldNormal, vec3(0.0, 1.0, 0.0));
            vec3 sunShadowOrigin = hitPos + sunShadowNormal * SHADOW_TMIN;
            float sunTmin = SHADOW_TMIN;
            float sunTmax = 1e8;
            uint sunShadowFlags = gl_RayFlagsTerminateOnFirstHitEXT
                                | gl_RayFlagsSkipClosestHitShaderEXT;
            // mask 0x01 = triangles only — volume AABBs skipped (handled by volumetric transmittance)
            traceRayEXT(topLevelAS, sunShadowFlags, 0x01, 0, 1, 1,
                        sunShadowOrigin, sunTmin, sunDir, sunTmax, 1);
            float sunShadowVisibility = shadowOccluded ? 0.0 : 1.0;
            if (sunShadowVisibility > 1e-4) {
                float sunVolTr = computeVolumeShadowTransmittance(sunShadowOrigin, sunDir, sunTmax);
                vec3 V        = normalize(-rayDir);
                vec3 sunBRDF  = evaluate_brdf_gl(worldNormal, V, sunDir,
                                                 albedo, roughness, metallic, specular, transmission);
                vec3 sunLi    = worldData.w.sunColor * worldData.w.sunIntensity;
                vec3 sunContrib = sunBRDF * sunLi * NdotSun * sunVolTr * sunShadowVisibility;
                sunContrib = clamp(sunContrib, vec3(0.0), vec3(1e4));
                vec3 att = clamp(directAttenuation, vec3(0.0), vec3(1e2));
                payload.radiance += att * sunContrib;
            }
        }
    }

    // ----------------------------------------------------------
    // 5. Scatter kararı — Principled BSDF
    // ----------------------------------------------------------


    // ----------------------------------------------------------
    // Read Principled BSDF extended parameters
    // ----------------------------------------------------------
    float clearcoat         = clamp(mat.clearcoat, 0.0, 1.0);
    float clearcoatRoughness = clamp(mat.clearcoat_roughness, 0.001, 1.0);
    if (weatherSurfaceActive() && worldData.w.weatherType == 1) {
        float wet = clamp(worldData.w.weatherSurfaceWetness, 0.0, 1.0);
        clearcoat = max(clearcoat, wet * 0.72);
        clearcoatRoughness = min(clearcoatRoughness, max(0.006, 0.045 - wet * 0.030));
    }
    float translucent        = clamp(mat.translucent, 0.0, 1.0);
    float subsurfaceAmount   = clamp(mat.subsurface_amount, 0.0, 1.0);
    vec3  subsurfaceColor    = max(vec3(mat.subsurface_r, mat.subsurface_g, mat.subsurface_b), vec3(0.001));
    vec3  subsurfaceRadius   = max(vec3(mat.subsurface_radius_r, mat.subsurface_radius_g, mat.subsurface_radius_b), vec3(0.001));
    float subsurfaceScale    = max(mat.subsurface_scale, 0.001);
    float subsurfaceAniso    = clamp(mat.subsurface_anisotropy, -0.99, 0.99);

    // ----------------------------------------------------------
    // 5. Scatter kararı — Full Principled BSDF
    // ----------------------------------------------------------
    // Layer order (top to bottom), stochastic selection each bounce:
    //   1. Clearcoat (IOR=1.5 GGX specular, weight = clearcoat * fresnel)
    //   2. Metallic (if metallic > 0)
    //   3. Dielectric specular (Fresnel) or diffuse sub-layer:
    //        a. Translucent (thin-surface transmission)
    //        b. SSS random walk
    //        c. Lambertian diffuse

    // --- Clearcoat layer (stochastic, IOR=1.5, F0=0.04) ---
    if (clearcoat > 0.01) {
        vec3  viewDirCC  = -rayDir;
        float cosNV_CC   = max(dot(viewDirCC, worldNormal), 0.0);
        const float CC_F0 = 0.04;
        float ccFresnel  = CC_F0 + (1.0 - CC_F0) * pow(1.0 - cosNV_CC, 5.0);
        // Probability of choosing clearcoat lobe = clearcoat weight * fresnel
        float ccProb = clearcoat * ccFresnel;
        if (rnd(payload.seed) < ccProb) {
            scatterClearcoat(hitPos, worldNormal, rayDir, clearcoatRoughness,
                             mat.clearcoat_iridescence, mat.clearcoat_film_thickness, payload.seed);
            // Compensate selection probability
            payload.attenuation *= (1.0 / max(ccProb, 0.01));
            return;
        }
        // Base layer continues; compensate probability of NOT picking clearcoat
        payload.attenuation *= (1.0 / max(1.0 - ccProb, 0.01));
    }

    // --- Metallic / Diffuse blend ---
    float diffuseWeight = 1.0 - metallic;
    float metalWeight   = metallic;

    if (metallic >= 0.999) {
        // Pure metal
        scatterMetal(hitPos, worldNormal, rayDir, albedo, roughness, payload.seed);
    }
    else if (metallic <= 0.001) {
        // Dielectric Fresnel — F0 = 0.04 (non-metal standart)
        float F0_DIELECTRIC = clamp(0.08 * specular, 0.0, 0.08);
        vec3  viewDir  = -rayDir;
        float cosTheta = max(dot(viewDir, worldNormal), 0.0);

        // Schlick Fresnel, attenuated by roughness (rough surfaces scatter less)
        float fresnelBase   = F0_DIELECTRIC + (1.0 - F0_DIELECTRIC)
                              * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);       

        if (rnd(payload.seed) < fresnelBase) {
            // Specular lob: GGX reflection (roughness=0 → mirror)
            scatterMetal(hitPos, worldNormal, rayDir, vec3(1.0), roughness, payload.seed);
        } else {
            // Diffuse sub-layer: choose translucency / SSS / diffuse

            // Compute sub-layer probabilities (sum must be <= 1.0)
            float pTrans = translucent;
            float pSSS   = (1.0 - pTrans) * subsurfaceAmount;
            float pDiff  = 1.0 - pTrans - pSSS;

            float r = rnd(payload.seed);

            if (pTrans > 0.01 && r < pTrans) {
                // --- Translucent: thin-surface diffuse transmission ---
                scatterTranslucent(hitPos, worldNormal, albedo, payload.seed);
                payload.attenuation *= (1.0 / max(pTrans, 0.01));
            }
            else if (pSSS > 0.01 && r < pTrans + pSSS) {
                // --- Subsurface Scattering: random walk ---
                scatterSSS(hitPos, worldNormal, albedo,
                           subsurfaceColor, subsurfaceAmount, subsurfaceScale,
                           subsurfaceRadius, subsurfaceAniso, payload.seed);
                payload.attenuation *= (1.0 / max(pSSS, 0.01));
            }
            else {
                // --- Lambertian Diffuse ---
                scatterDiffuse(hitPos, worldNormal, albedo, payload.seed);
                payload.attenuation *= (1.0 / max(pDiff, 0.01));
            }
        }
    }
    else {
        // Metallic blend: stochastic selection.
        //
        // The lobe selection probability is the material weight. Do not apply
        // 1/p compensation here unless the sampled lobe is also multiplied by
        // its material weight first; otherwise intermediate metallic values
        // estimate full diffuse + full specular energy and create fireflies.
        if (rnd(payload.seed) < metalWeight) {
            scatterMetal(hitPos, worldNormal, rayDir, albedo, roughness, payload.seed);
        } else {
            scatterDiffuse(hitPos, worldNormal, albedo, payload.seed);
        }
    }

    // Resin base scattered as a normal diffuse lobe above (which set BOUNCE_DIFFUSE).
    // Re-tag it BOUNCE_RESIN so raygen counts it against the small dedicated resin
    // budget instead of the global diffuse budget — bounds resin GI cost (TDR fix).
    if (resinActive && payload.scattered) {
        payload.bounceType = BOUNCE_RESIN;
    }
}

