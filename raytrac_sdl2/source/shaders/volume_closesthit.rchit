/*
 * RayTrophi Studio — Vulkan Volume Closest-Hit Shader
 * Volumetric Ray Marching with Multi-Lobe Henyey-Greenstein Scattering
 *
 * OptiX uyumluluk:
 *   - HitGroupData volumetric fields ile tam eşleşme
 *   - GpuVDBVolume / GpuGasVolume density/scatter/absorption modeline uyumlu  
 *   - Henyey-Greenstein dual-lobe phase function
 *   - Delta tracking (ratio tracking) ile uyumlu woodcock stepping
 *   - Light march ile self-shadowing
 *
 * SBT offset = 1 (volume hit group index), sbtStride = 2 (triangle + volume)
 * Bu shader VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR ile çalışır.
 */

#version 460
#extension GL_EXT_ray_tracing                          : require
#extension GL_EXT_buffer_reference                     : require
#extension GL_EXT_scalar_block_layout                  : require
#extension GL_EXT_nonuniform_qualifier                 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// ============================================================
// Constants
// ============================================================
const float PI      = 3.14159265358979323846;
const float TWO_PI  = 6.28318530717958647692;
const float INV_4PI = 0.07957747154594766788;
const float EPSILON = 1e-6;

// ============================================================
// Push Constants — must match raygen/closesthit CameraPushConstants
// ============================================================
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
    float pad1;
} cam;

// ============================================================
// Payload — raygen/closesthit ile tam eşleşme
// ============================================================
struct RayPayload {
    vec3  radiance;
    vec3  attenuation;
    vec3  scatterOrigin;
    vec3  scatterDir;
    uint  seed;
    bool  scattered;
    bool  hitEmissive;
    uint  occluded;
    bool  skipAABBs;    // set when solid surface detected inside volume
    vec3  primaryAlbedo;
    vec3  primaryNormal;
    uint  primaryHit;
    float primaryTransmission;
    float primaryMetallic;
    uint  bounceType;
    uint  primaryMaterialId;   // Stylize AOV: real material index of the primary hit
};

const uint BOUNCE_TRANSPARENT = 3u;

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT bool shadowOccluded;

// ============================================================
// Descriptor Bindings
// ============================================================
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

struct LightData {
    vec4 position;
    vec4 color;
    vec4 params;
    vec4 direction;
    vec4 area_u;
    vec4 area_v;
};

layout(set = 0, binding = 3, scalar) readonly buffer LightBuffer { LightData l[]; } lights;

// ═══════════════════════════════════════════════════════════════════════════════
// Binding 9: Volume Instances SSBO
// Matches VulkanRT::VkVolumeInstance (256 bytes per instance)
// ═══════════════════════════════════════════════════════════════════════════════
struct VkVolumeInstance {
    // Transform (48 bytes = 12 floats)
    float transform[12];
    
    // Bounds (24 bytes)
    vec3  aabb_min;
    vec3  aabb_max;
    
    // Density (16 bytes)
    float density_multiplier;
    float density_remap_low;
    float density_remap_high;
    float noise_scale;
    
    // Scattering (32 bytes)
    vec3  scatter_color;
    float scatter_coefficient;
    float scatter_anisotropy;
    float scatter_anisotropy_back;
    float scatter_lobe_mix;
    float scatter_multi;
    
    // Absorption (16 bytes)
    vec3  absorption_color;
    float absorption_coefficient;
    
    // Emission (16 bytes)
    vec3  emission_color;
    float emission_intensity;
    
    // Ray march params (16 bytes)
    float step_size;
    int   max_steps;
    int   shadow_steps;
    float shadow_strength;
    
    // Flags (16 bytes)
    int   volume_type;
    int   is_active;
    float voxel_size;
    int   _pad0;
    
    // Inverse transform (48 bytes = 12 floats)
    float inv_transform[12];
    
    // Reserved (24 bytes) — matches VkVolumeInstance C++ layout
    uint64_t vdb_grid_address;   // NanoVDB grid device address (or 0)
    uint64_t vdb_temp_address;   // secondary grid (temperature etc.)
    float    _reserved[2];       // [0] density cutoff, [1] reserved

    // Emission extension (256 bytes) — blackbody / color-ramp
    int   emission_mode;         // 0=off, 1=plain color, 2=blackbody/color-ramp
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
// EXTENDED WORLD DATA — for fog/atmosphere access
// ════════════════════════════════════════════════════════════════════════════════
struct VkWorldDataExtended {
    vec3  sunDir;       int   mode;
    vec3  sunColor;     float sunIntensity;
    float sunSize;      float mieAnisotropy;
    float rayleighDensity; float mieDensity;
    float humidity;     float temperature;
    float ozoneAbsorptionScale; float atmosphereIntensity;
    float airDensity;   float dustDensity;
    float ozoneDensity; float altitude;
    float planetRadius; float atmosphereHeight;
    int   multiScatterEnabled; float multiScatterFactor;
    int   cloudsEnabled; float cloudCoverage; float cloudDensity; float cloudScale;
    float cloudHeightMin; float cloudHeightMax; float cloudOffsetX; float cloudOffsetZ;
    float cloudQuality; float cloudDetail; int cloudBaseSteps; int cloudLightSteps;
    float cloudShadowStrength; float cloudAmbientStrength; float cloudSilverIntensity; float cloudAbsorption;
    float cloudAnisotropy; float cloudAnisotropyBack; float cloudLobeMix; float cloudEmissiveIntensity;
    vec3  cloudEmissiveColor; float _pad3;
    int   fogEnabled;   float fogDensity; float fogHeight; float fogFalloff;
    float fogDistance;  float fogSunScatter; vec3 fogColor; float _pad4;
    int   godRaysEnabled; float godRaysIntensity; float godRaysDensity; int godRaysSamples;
    int   aerialEnabled; float aerialMinDistance; float aerialMaxDistance; float aerialDensity;
    int   weatherEnabled; int weatherType; float weatherIntensity; float weatherDensity;
    vec3  weatherWindDirection; float weatherWindSpeed;
    float weatherPrecipitationScale; float weatherVisibility; float weatherSurfaceWetness; float weatherSurfaceAccumulation;
    float weatherSurfaceSettling; float weatherSurfaceHeight;
    int   weatherVisualMode; int weatherSurfaceResponseEnabled;
    int   envTexSlot;   float envIntensity; float envRotation; int _pad5; // nishitaLutReady
    int   envOverlayEnabled; int envOverlayBlendMode; float envOverlayIntensity; float envOverlayRotation;
    uvec2 transmittanceLUT; uvec2 skyviewLUT; uvec2 multiScatterLUT; uvec2 aerialPerspectiveLUT;
};

layout(set = 0, binding = 7, scalar) readonly buffer WorldBuffer { VkWorldDataExtended w; } worldData;
layout(set = 0, binding = 8) uniform sampler2D atmosphereLUTs[4];

// ============================================================
// Hit Attributes from intersection shader
// ============================================================
hitAttributeEXT vec2 volumeHitAttrib; // .x = tNear, .y = tFar

// ============================================================
// PCG RNG
// ============================================================
uint pcgNext(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint seed) {
    return float(pcgNext(seed)) * (1.0 / 4294967296.0);
}

vec3 sampleTransmittanceLUT(vec3 worldPos, vec3 sunDir) {
    if (worldData.w.mode != 2) return vec3(1.0);
    if (worldData.w.atmosphereHeight <= 0.0) return vec3(1.0);
    if (worldData.w._pad5 == 0) return vec3(1.0);
    float Rg = max(worldData.w.planetRadius, 1.0);
    vec3 p = worldPos + vec3(0.0, Rg, 0.0);
    float altitude = max(length(p) - Rg, 0.0);
    vec3 up = normalize(p);
    float cosTheta = dot(up, normalize(sunDir));
    float u = clamp((cosTheta + 0.2) / 1.2, 0.0, 1.0);
    float v = clamp(altitude / worldData.w.atmosphereHeight, 0.0, 1.0);
    return textureLod(atmosphereLUTs[0], vec2(u, v), 0.0).rgb;
}

vec3 sampleSkyAmbient(vec3 viewDir) {
    // Directional sky ambient: blend up-direction with view direction
    // so horizon/sun tint mixes into the medium more naturally.
    vec3 ambDir = normalize(mix(vec3(0.0, 1.0, 0.0), normalize(viewDir), 0.45));
    ambDir = normalize(ambDir + normalize(worldData.w.sunDir) * 0.15);

    if (worldData.w.mode == 2 && worldData.w._pad5 != 0) {
        float az = atan(ambDir.z, ambDir.x);
        if (az < 0.0) az += TWO_PI;
        float u = az / TWO_PI;
        float v = (1.0 - clamp(ambDir.y, -1.0, 1.0)) * 0.5;
        return textureLod(atmosphereLUTs[1], vec2(u, v), 0.0).rgb
             * (0.15 * worldData.w.atmosphereIntensity);
    }
    return worldData.w.sunColor * (0.15 * worldData.w.atmosphereIntensity);
}

// ============================================================
// Powder Effect for volumetric clouds (OptiX Parity)
// ============================================================
float gpu_powder_effect(float density, float cos_theta) {
    float powder = 1.0 - exp(-density * 2.0);
    float forward_bias = 0.5 + 0.5 * max(0.0, cos_theta);
    return powder * forward_bias;
}

// ============================================================
// Henyey-Greenstein Phase Function
// ============================================================
float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return INV_4PI * (1.0 - g2) / (denom * sqrt(denom) + EPSILON);
}

// Dual-lobe HG phase function (matches OptiX implementation)
float dualLobeHG(float cosTheta, float g_forward, float g_back, float lobeMix) {
    float phaseForward = henyeyGreenstein(cosTheta, g_forward);
    float phaseBack    = henyeyGreenstein(cosTheta, g_back);
    return mix(phaseBack, phaseForward, lobeMix);
}

// ============================================================
// Sample HG Phase Function Direction
// ============================================================
vec3 sampleHG(vec3 inDir, float g, inout uint seed) {
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    
    float cosTheta;
    if (abs(g) < 1e-3) {
        // Isotropic
        cosTheta = 1.0 - 2.0 * r1;
    } else {
        float s = (1.0 - g * g) / (1.0 - g + 2.0 * g * r1);
        cosTheta = (1.0 + g * g - s * s) / (2.0 * g);
    }
    
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = TWO_PI * r2;
    
    // Build local frame from inDir
    vec3 w = normalize(inDir);
    vec3 u, v;
    if (abs(w.x) > 0.9) {
        u = normalize(cross(vec3(0, 1, 0), w));
    } else {
        u = normalize(cross(vec3(1, 0, 0), w));
    }
    v = cross(w, u);
    
    return normalize(u * (sinTheta * cos(phi)) + v * (sinTheta * sin(phi)) + w * cosTheta);
}

// ============================================================
// 3D Noise (procedural density for volume_type=1)
// ============================================================
float hash3D(vec3 p) {
    p = fract(p * vec3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

vec3 hash3Gradient(vec3 p) {
    p = vec3(
        dot(p, vec3(127.1, 311.7, 74.7)),
        dot(p, vec3(269.5, 183.3, 246.1)),
        dot(p, vec3(113.5, 271.9, 124.6))
    );
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453);
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * f * (f * (f * 6.0 - vec3(15.0)) + vec3(10.0));

    float n000 = dot(hash3Gradient(i + vec3(0, 0, 0)), f - vec3(0, 0, 0));
    float n100 = dot(hash3Gradient(i + vec3(1, 0, 0)), f - vec3(1, 0, 0));
    float n010 = dot(hash3Gradient(i + vec3(0, 1, 0)), f - vec3(0, 1, 0));
    float n110 = dot(hash3Gradient(i + vec3(1, 1, 0)), f - vec3(1, 1, 0));
    float n001 = dot(hash3Gradient(i + vec3(0, 0, 1)), f - vec3(0, 0, 1));
    float n101 = dot(hash3Gradient(i + vec3(1, 0, 1)), f - vec3(1, 0, 1));
    float n011 = dot(hash3Gradient(i + vec3(0, 1, 1)), f - vec3(0, 1, 1));
    float n111 = dot(hash3Gradient(i + vec3(1, 1, 1)), f - vec3(1, 1, 1));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z) * 0.5 + 0.5;
}

float fbmNoise(vec3 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise3D(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

float proceduralCloudDensity(VkVolumeInstance vol, vec3 localPos) {
    vec3 span = max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
    vec3 normPos = clamp((localPos - vol.aabb_min) / span, vec3(0.0), vec3(1.0));
    float baseScale = max(vol.cloud_base_scale, 1.0);
    vec3 cloudCoord = vec3(
        normPos.x * baseScale + vol.cloud_offset_x,
        normPos.y * 1.35,
        normPos.z * baseScale + vol.cloud_offset_z);
    vec3 seedOffset = vec3(vol.cloud_seed * 0.137, vol.cloud_seed * 0.317, vol.cloud_seed * 0.719);
    cloudCoord += seedOffset;

    float coverage = clamp(vol.cloud_coverage, 0.0, 1.0);
    float detail = clamp(vol.cloud_detail, 0.0, 1.0);
    float erosion = clamp(vol.cloud_erosion, 0.0, 1.0);

    float warpX = fbmNoise(vec3(cloudCoord.x * 0.38, cloudCoord.y * 0.16, cloudCoord.z * 0.38) + vec3(11.0, 0.0, 7.0), 2) - 0.5;
    float warpZ = fbmNoise(vec3(cloudCoord.x * 0.38, cloudCoord.y * 0.16, cloudCoord.z * 0.38) + vec3(41.0, 3.0, 23.0), 2) - 0.5;
    vec3 warped = cloudCoord + vec3(warpX * 1.35, 0.0, warpZ * 1.35);

    float base = fbmNoise(vec3(warped.x * 0.52, warped.y * 0.28, warped.z * 0.52), 4);
    float billow = 1.0 - abs(fbmNoise(vec3(warped.x * 1.15, warped.y * 0.5, warped.z * 1.15) + vec3(17.0, 3.0, 11.0), 4) * 2.0 - 1.0);
    float detailNoise = fbmNoise(warped * mix(2.8, 7.0, detail) + vec3(31.0, 7.0, 19.0), 2);

    float puffy = smoothstep(0.32, 0.88, billow);
    float cumulus = clamp((vol.density_multiplier - 0.38) / 0.85, 0.0, 1.0);
    float shape = mix(base, base * 0.45 + puffy * 0.75, mix(0.55, 0.9, cumulus));
    shape -= detailNoise * mix(0.06, 0.28, erosion);

    float threshold = mix(0.80, 0.26, coverage) - cumulus * 0.08;
    float density = max((shape - threshold) / max(1.0 - threshold, 1e-4), 0.0);

    float bottom = smoothstep(mix(0.12, 0.04, cumulus), mix(0.42, 0.24, cumulus), normPos.y);
    float top = 1.0 - smoothstep(mix(0.72, 0.82, cumulus), 1.02, normPos.y);
    float dome = mix(1.0, smoothstep(0.08, 0.58, normPos.y) * (1.0 - smoothstep(0.88, 1.04, normPos.y)) + 0.25, cumulus);
    float heightProfile = bottom * top * dome;

    vec3 edge = vec3(0.5) - abs(normPos - vec3(0.5));
    float edgeFalloff = smoothstep(0.0, max(vol.cloud_edge_fade, 0.02), min(edge.x, edge.z));
    return density * mix(density, sqrt(max(density, 0.0)), cumulus * 0.55) * heightProfile * edgeFalloff * mix(4.6, 3.4, cumulus);
}

// ============================================================
// NanoVDB GLSL Setup
// ============================================================
#define PNANOVDB_GLSL
#define PNANOVDB_BUF_CUSTOM

struct pnanovdb_buf_t {
    uint64_t address;
};

// We define a buffer block matching NanoVDB scalar layout
layout(buffer_reference, std430, buffer_reference_align=4) buffer NanoVDBBlock {
    uint data[];
};

uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byte_offset) {
    NanoVDBBlock blk = NanoVDBBlock(buf.address);
    return blk.data[byte_offset >> 2];
}

uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byte_offset) {
    NanoVDBBlock blk = NanoVDBBlock(buf.address);
    uint idx = byte_offset >> 2;
    return uvec2(blk.data[idx], blk.data[idx + 1]);
}

void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint byte_offset, uint value) {}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint byte_offset, uvec2 value) {}

#include "PNanoVDB.h"

// ── Persistent-accessor trilinear sampler ─────────────────────────────────
// Caller must have pre-fetched buf/mapH/rootH and called pnanovdb_readaccessor_init
// once. The accessor caches the last walked root→internal→leaf path; reusing it
// across spatially-coherent march steps (typical step ≪ leaf size = 8 voxels)
// skips the tree walk on cache hits. This collapses NanoVDB sampling cost in the
// hot loop from O(tree-depth) to O(1) for the common case.
float sampleNanoVDBFloatTrilinearAcc(
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    vec3 worldPos)
{
    pnanovdb_vec3_t wPos = pnanovdb_vec3_uniform(0.0);
    wPos.x = worldPos.x; wPos.y = worldPos.y; wPos.z = worldPos.z;
    pnanovdb_vec3_t iPos = pnanovdb_map_apply_inverse(buf, mapH, wPos);

    vec3 idxPos = vec3(iPos.x, iPos.y, iPos.z);
    vec3 p0 = floor(idxPos);
    vec3 frac = fract(idxPos);

    float d[8];
    for (int i = 0; i < 8; ++i) {
        pnanovdb_coord_t coord;
        coord.x = int(p0.x) + ((i & 1) != 0 ? 1 : 0);
        coord.y = int(p0.y) + ((i & 2) != 0 ? 1 : 0);
        coord.z = int(p0.z) + ((i & 4) != 0 ? 1 : 0);
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, coord);
        d[i] = pnanovdb_read_float(buf, addr);
    }

    float dx00 = mix(d[0], d[1], frac.x);
    float dx10 = mix(d[2], d[3], frac.x);
    float dx01 = mix(d[4], d[5], frac.x);
    float dx11 = mix(d[6], d[7], frac.x);
    float dxy0 = mix(dx00, dx10, frac.y);
    float dxy1 = mix(dx01, dx11, frac.y);
    return mix(dxy0, dxy1, frac.z);
}

// Trilinear interpolation of a FloatGrid
float sampleNanoVDBFloatTrilinear(uint64_t gridAddr, vec3 worldPos) {
    if (gridAddr == 0) return 0.0;
    
    pnanovdb_buf_t buf;
    buf.address = gridAddr;
    
    // Get Handles
    pnanovdb_grid_handle_t gridH; gridH.address.byte_offset = 0u;
    pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(buf, gridH);
    pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(buf, treeH);
    pnanovdb_map_handle_t  mapH  = pnanovdb_grid_get_map(buf, gridH);
    
    // Init Accessor
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, rootH);
    
    // World to Index mapped
    pnanovdb_vec3_t wPos = pnanovdb_vec3_uniform(0.0);
    wPos.x = worldPos.x; wPos.y = worldPos.y; wPos.z = worldPos.z;
    pnanovdb_vec3_t iPos = pnanovdb_map_apply_inverse(buf, mapH, wPos);
    
    vec3 idxPos = vec3(iPos.x, iPos.y, iPos.z);
    
    vec3 p0 = floor(idxPos);
    vec3 frac = fract(idxPos);
    
    float d[8];
    for (int i = 0; i < 8; ++i) {
        pnanovdb_coord_t coord;
        coord.x = int(p0.x) + ((i & 1) != 0 ? 1 : 0);
        coord.y = int(p0.y) + ((i & 2) != 0 ? 1 : 0);
        coord.z = int(p0.z) + ((i & 4) != 0 ? 1 : 0);
        
        // Fast leaf-level read
        pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, coord);
        d[i] = pnanovdb_read_float(buf, addr);
    }
    
    float dx00 = mix(d[0], d[1], frac.x);
    float dx10 = mix(d[2], d[3], frac.x);
    float dx01 = mix(d[4], d[5], frac.x);
    float dx11 = mix(d[6], d[7], frac.x);
    
    float dxy0 = mix(dx00, dx10, frac.y);
    float dxy1 = mix(dx01, dx11, frac.y);
    
    return mix(dxy0, dxy1, frac.z);
}

// ============================================================
// Blackbody RGB — Kim et al. approximation (matches OptiX blackbody_to_rgb)
// ============================================================
vec3 blackbodyToRGB(float kelvin) {
    kelvin = clamp(kelvin, 1000.0, 40000.0);
    float temp = kelvin / 100.0;
    float r, g, b;
    // Red
    if (temp <= 66.0) { r = 1.0; }
    else { r = clamp(329.698727446 * pow(temp - 60.0, -0.1332047592) / 255.0, 0.0, 1.0); }
    // Green
    if (temp <= 66.0) { g = clamp((99.4708025861 * log(temp) - 161.1195681661) / 255.0, 0.0, 1.0); }
    else              { g = clamp(288.1221695283 * pow(temp - 60.0, -0.0755148492) / 255.0, 0.0, 1.0); }
    // Blue
    if (temp >= 66.0)      { b = 1.0; }
    else if (temp <= 19.0) { b = 0.0; }
    else { b = clamp((138.5177312231 * log(temp - 10.0) - 305.0447927307) / 255.0, 0.0, 1.0); }
    return vec3(r, g, b);
}

// ============================================================
// Color Ramp — linear interpolation over stop list
// ============================================================
vec3 sampleColorRamp(VkVolumeInstance vol, float t) {
    if (vol.ramp_stop_count == 0) return vec3(1.0);
    vec3 c0 = vec3(vol.ramp_colors_r[0], vol.ramp_colors_g[0], vol.ramp_colors_b[0]);
    if (t <= vol.ramp_positions[0]) return c0;
    int last = vol.ramp_stop_count - 1;
    vec3 cN = vec3(vol.ramp_colors_r[last], vol.ramp_colors_g[last], vol.ramp_colors_b[last]);
    if (t >= vol.ramp_positions[last]) return cN;
    for (int i = 1; i < vol.ramp_stop_count; ++i) {
        if (t < vol.ramp_positions[i]) {
            float f = (t - vol.ramp_positions[i-1]) / max(vol.ramp_positions[i] - vol.ramp_positions[i-1], 1e-6);
            vec3 a = vec3(vol.ramp_colors_r[i-1], vol.ramp_colors_g[i-1], vol.ramp_colors_b[i-1]);
            vec3 b = vec3(vol.ramp_colors_r[i],   vol.ramp_colors_g[i],   vol.ramp_colors_b[i]);
            return mix(a, b, f);
        }
    }
    return cN;
}

// ============================================================
// Temperature Sampling — secondary NanoVDB grid via vdb_temp_address
// ============================================================
float sampleTemperature(VkVolumeInstance vol, vec3 worldPos) {
    if (vol.vdb_temp_address == 0) return 0.0;
    vec3 localPos;
    localPos.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y
               + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y
               + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y
               + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11];
    
    // Safety check bound box instead of 0.5 cube
    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) return 0.0;
    
    // Pivot offset correction (OptiX parity)
    localPos.x -= vol.pivot_offset[0];
    localPos.y -= vol.pivot_offset[1];
    localPos.z -= vol.pivot_offset[2];
    
    return sampleNanoVDBFloatTrilinear(vol.vdb_temp_address, localPos);
}

// ── Persistent-accessor density sampler ───────────────────────────────────
// Same dispatch as sampleDensity but routes volume_type==2 NanoVDB reads through
// the caller's pre-initialized accessor. Other volume types (homogeneous, procedural,
// cloud) ignore the accessor params. Caller must guarantee buf/mapH are valid when
// vol.volume_type == 2 && vol.vdb_grid_address != 0; otherwise the accessor args are
// untouched.
float sampleDensityAcc(
    VkVolumeInstance vol,
    vec3 worldPos,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc)
{
    vec3 localPos;
    localPos.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y
               + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y
               + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y
               + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];

    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) {
        return 0.0;
    }

    float density = 1.0;

    if (vol.volume_type == 0) {
        density = 1.0;
    } else if (vol.volume_type == 1) {
        vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
        vec3 noiseCoord = normPos * vol.noise_scale;
        density = fbmNoise(noiseCoord, 4);
        vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
        float edgeFalloff = min(min(edgeDist.x, edgeDist.y), edgeDist.z);
        density *= smoothstep(0.0, 0.1, edgeFalloff);
    } else if (vol.volume_type == 2) {
        if (vol.vdb_grid_address != 0) {
            vec3 vdbWorldPos = localPos;
            vdbWorldPos.x -= vol.pivot_offset[0];
            vdbWorldPos.y -= vol.pivot_offset[1];
            vdbWorldPos.z -= vol.pivot_offset[2];
            density = sampleNanoVDBFloatTrilinearAcc(buf, mapH, acc, vdbWorldPos);
        } else {
            vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
            vec3 noiseCoord = normPos * max(vol.noise_scale, 1.0);
            density = fbmNoise(noiseCoord, 4);
            vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
            density *= smoothstep(0.0, 0.1, min(min(edgeDist.x, edgeDist.y), edgeDist.z));
        }
    } else if (vol.volume_type == 3 || vol.source_type == 3) {
        density = proceduralCloudDensity(vol, localPos);
    }

    float remappedDensity = max((density - vol.density_remap_low) / max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0);
    float densityCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    if (remappedDensity <= densityCutoff) {
        return 0.0;
    }
    return remappedDensity * vol.density_multiplier;
}

// ============================================================
// Density Sampling — supports homogeneous and procedural noise
// ============================================================
float sampleDensity(VkVolumeInstance vol, vec3 worldPos) {
    // Transform world pos → object space
    vec3 localPos;
    localPos.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y 
               + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
    localPos.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y 
               + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
    localPos.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y 
               + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];
    
    // Check against real bounding box instead of [-0.5, 0.5]^3
    if (any(lessThan(localPos, vol.aabb_min)) || any(greaterThan(localPos, vol.aabb_max))) {
        return 0.0;
    }
    
    float density = 1.0;
    
    if (vol.volume_type == 0) {
        // Homogeneous: constant density
        density = 1.0;
    } else if (vol.volume_type == 1) {
        // Procedural noise: convert localPos from world scale back to standard normalized coords
        // The procedural noise historically mapped [-0.5, 0.5] to fit bounds.
        // We remap the precise boundary.
        vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
        vec3 noiseCoord = normPos * vol.noise_scale;
        density = fbmNoise(noiseCoord, 4);
        
        // Smooth falloff near edges
        vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
        float edgeFalloff = min(min(edgeDist.x, edgeDist.y), edgeDist.z);
        density *= smoothstep(0.0, 0.1, edgeFalloff);
        
    } else if (vol.volume_type == 2) {
        // NanoVDB grid sampling.
        if (vol.vdb_grid_address != 0) {
            // Apply OptiX pivot parity since NanoVDB indexing assumes raw bounding spatial coordinates
            vec3 vdbWorldPos = localPos;
            vdbWorldPos.x -= vol.pivot_offset[0];
            vdbWorldPos.y -= vol.pivot_offset[1];
            vdbWorldPos.z -= vol.pivot_offset[2];
            
            density = sampleNanoVDBFloatTrilinear(vol.vdb_grid_address, vdbWorldPos);
        } else {
            // Fallback: procedural noise
            vec3 normPos = (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
            vec3 noiseCoord = normPos * max(vol.noise_scale, 1.0);
            density = fbmNoise(noiseCoord, 4);
            vec3 edgeDist = vec3(0.5) - abs(normPos - vec3(0.5));
            density *= smoothstep(0.0, 0.1, min(min(edgeDist.x, edgeDist.y), edgeDist.z));
        }
    } else if (vol.volume_type == 3 || vol.source_type == 3) {
        density = proceduralCloudDensity(vol, localPos);
    }
    
    // Apply density remap (No upper clamp, matches OptiX fmaxf)
    float remappedDensity = max((density - vol.density_remap_low) / max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0);
    float densityCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    if (remappedDensity <= densityCutoff) {
        return 0.0;
    }
    
    return remappedDensity * vol.density_multiplier;
}

// Accessor-aware lightMarch. Reuses the caller's density-grid accessor so the
// shadow-march inner loop also benefits from leaf-level cache hits.
float lightMarchAcc(
    VkVolumeInstance vol,
    vec3 pos,
    vec3 lightDir,
    float maxDist,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc)
{
    if (vol.shadow_steps <= 0) return 1.0;
    if (maxDist <= 1e-4) return 1.0;

    float sigma_t = vol.scatter_coefficient + vol.absorption_coefficient;
    if (sigma_t <= EPSILON) return 1.0;

    int reqSteps = clamp(vol.shadow_steps, 1, 64);
    float dMid = sampleDensityAcc(vol, pos + lightDir * (0.5 * maxDist), buf, mapH, acc);
    float tauHint = max(0.0, dMid) * sigma_t * maxDist;
    if (tauHint <= 0.02) return 1.0;
    float stepScale = clamp(sqrt(tauHint), 0.25, 1.0);
    int steps = int(ceil(float(reqSteps) * stepScale));
    steps = clamp(steps, 3, min(reqSteps, 16));

    float stepSize = maxDist / (float(steps) * 2.0);
    stepSize = max(stepSize, 1e-5);
    float jitter = fract(sin(dot(pos, vec3(12.9898, 78.233, 37.719)) +
                             dot(lightDir, vec3(39.346, 11.135, 83.155))) * 43758.5453);

    float s_trans = 0.0;
    for (int i = 0; i < steps; i++) {
        vec3 samplePos = pos + lightDir * (float(i) + jitter + 0.5) * stepSize;
        float d = sampleDensityAcc(vol, samplePos, buf, mapH, acc);
        s_trans += d * sigma_t * stepSize;
        if (s_trans > 10.0) break;
    }

    float beers = exp(-s_trans);
    float phys_trans = beers;
    if (vol.scatter_multi > 0.0) {
        float albedo_lum = dot(vol.scatter_color, vec3(0.2126, 0.7152, 0.0722));
        float beers_soft = exp(-s_trans * 0.25);
        phys_trans = beers * (1.0 - vol.scatter_multi * albedo_lum)
                   + beers_soft * (vol.scatter_multi * albedo_lum);
    }

    float shadowStrength = clamp(vol.shadow_strength * 0.92, 0.0, 1.0);
    return 1.0 - shadowStrength * (1.0 - phys_trans);
}

// ============================================================
// Light March — estimate transmittance from scatter point toward light
// (OptiX vol_light_steps eşdeğeri)
// ============================================================
float lightMarch(VkVolumeInstance vol, vec3 pos, vec3 lightDir, float maxDist) {
    if (vol.shadow_steps <= 0) return 1.0;
    if (maxDist <= 1e-4) return 1.0;
    
    // [SHADOW FIX] Match OptiX: s_step_world = world_vol_extent / (shadow_steps * 2)
    // This covers half of maxDist with shadow_steps samples.
    // OLD code used  min(maxDist, step_size*2) / shadow_steps  which for a large cloud
    // (maxDist=10, step_size=0.1) would only march 0.2 world units — missing 98% of the
    // volume. That left dense cores fully lit → solid white.
    float sigma_t = vol.scatter_coefficient + vol.absorption_coefficient;
    if (sigma_t <= EPSILON) return 1.0;

    int reqSteps = clamp(vol.shadow_steps, 1, 64);
    float dMid = sampleDensity(vol, pos + lightDir * (0.5 * maxDist));
    float tauHint = max(0.0, dMid) * sigma_t * maxDist;
    if (tauHint <= 0.02) return 1.0;
    float stepScale = clamp(sqrt(tauHint), 0.25, 1.0);
    int steps = int(ceil(float(reqSteps) * stepScale));
    steps = clamp(steps, 3, min(reqSteps, 16));

    float stepSize = maxDist / (float(steps) * 2.0);
    // Clamp to avoid NaN/zero-step if shadow_steps was set very high
    stepSize = max(stepSize, 1e-5);
    float jitter = fract(sin(dot(pos, vec3(12.9898, 78.233, 37.719)) +
                             dot(lightDir, vec3(39.346, 11.135, 83.155))) * 43758.5453);
    
    // Accumulate optical depth (matches OptiX: density_sum += sigma_t * step)
    float s_trans = 0.0;
    for (int i = 0; i < steps; i++) {
        vec3 samplePos = pos + lightDir * (float(i) + jitter + 0.5) * stepSize;
        float d = sampleDensity(vol, samplePos);
        s_trans += d * sigma_t * stepSize;
        if (s_trans > 10.0) break; // fully occluded
    }
    
    // Multi-scatter shadow blend — matches OptiX:
    // phys_trans = beers*(1-ms*alb) + beers_soft*ms*alb
    // shadow = 1 - shadow_strength*(1 - phys_trans)
    float beers = exp(-s_trans);
    float phys_trans = beers;
    if (vol.scatter_multi > 0.0) {
        float albedo_lum = dot(vol.scatter_color, vec3(0.2126, 0.7152, 0.0722));
        float beers_soft = exp(-s_trans * 0.25);
        phys_trans = beers * (1.0 - vol.scatter_multi * albedo_lum)
                   + beers_soft * (vol.scatter_multi * albedo_lum);
    }
    
    float shadowStrength = clamp(vol.shadow_strength * 0.92, 0.0, 1.0);
    return 1.0 - shadowStrength * (1.0 - phys_trans);
}

// ============================================================
// Main — Volume Ray March Entry Point
// ============================================================
void main() {
    // Volume instance index from gl_InstanceCustomIndexEXT
    // (Set via TLASInstance::customIndex when building TLAS for volume objects)
    uint volIdx = gl_InstanceCustomIndexEXT;
    uint volCount = uint(max(int(cam.pad0), 0));
    if (volIdx >= volCount) {
        payload.scattered = false;
        return;
    }
    VkVolumeInstance vol = volumes.v[volIdx];
    
    // Skip inactive volumes
    if (vol.is_active == 0) {
        // Ignore hit — let ray continue
        payload.scattered = false;
        return;
    }
    
    vec3 rayOrigin = gl_WorldRayOriginEXT;
    vec3 rayDir    = normalize(gl_WorldRayDirectionEXT);
    
    // Get intersection range from the intersection shader
    float rawTNear = volumeHitAttrib.x;
    float tNear = rawTNear;
    float tFar  = volumeHitAttrib.y;
    bool cameraInsideVolume = (rawTNear <= 0.0);
    
    // Ensure valid march range
    tNear = max(tNear, 0.001);
    if (tFar <= tNear) {
        payload.scattered = false;
        return;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Persistent NanoVDB accessor — initialized ONCE per ray, reused across all
    // density samples in the main march, solid probe, and lightMarch inner loops.
    // For typical step_size ≪ 8 voxels (leaf size), most consecutive samples land
    // in the same leaf and the accessor's cached path skips the tree walk entirely.
    // Non-NanoVDB volumes (homogeneous, procedural, cloud) ignore these handles.
    // ══════════════════════════════════════════════════════════════════════════
    pnanovdb_buf_t        vdbBuf;
    pnanovdb_map_handle_t vdbMapH;
    pnanovdb_readaccessor_t vdbAcc;
    {
        vdbBuf.address = (vol.volume_type == 2 && vol.vdb_grid_address != 0) ? vol.vdb_grid_address : uint64_t(0);
        if (vdbBuf.address != 0) {
            pnanovdb_grid_handle_t gridH; gridH.address.byte_offset = 0u;
            pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(vdbBuf, gridH);
            pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(vdbBuf, treeH);
            vdbMapH = pnanovdb_grid_get_map(vdbBuf, gridH);
            pnanovdb_readaccessor_init(vdbAcc, rootH);
        } else {
            // Dummy zero-init so unrelated reads through Acc variants are well-defined.
            vdbMapH.address.byte_offset = 0u;
            pnanovdb_root_handle_t dummyRoot; dummyRoot.address.byte_offset = 0u;
            pnanovdb_readaccessor_init(vdbAcc, dummyRoot);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // FLUID SURFACE (source_type == 4): isosurface raymarch + Snell refraction
    // ══════════════════════════════════════════════════════════════════════════
    // The volume's density channel is a SDF-derived proxy band: 0 outside the
    // fluid, 1 inside, with a smooth ~surface_band_voxels-wide transition at
    // the surface. We walk the ray, find the first iso=0.5 crossing, compute
    // the surface normal from the gradient, and refract through with the
    // water IOR. No volumetric scatter — the surface is treated as a
    // dielectric boundary. Subsequent entries (second hit = exit) attenuate
    // by Beer-Lambert through the absorption channel for distance traversed.
    if (vol.source_type == 4) {
        const float ISO_THRESH = 0.5;
        // IOR comes from the bound fluid material (_ext_reserved[0]); fall back
        // to water if unset. Drives both refraction bending and Fresnel.
        float IOR_WATER = (vol._ext_reserved[0] > 1.0) ? vol._ext_reserved[0] : 1.33;

        // Walk quality. The step must satisfy TWO constraints:
        //   1. fine enough for surface accuracy (voxel-relative, step_size),
        //   2. coarse enough that `max_steps` actually reach tFar — otherwise
        //      the walk stops partway and the far side of the fluid is skipped
        //      (back cells "pass"). We take the MAX of the fine step and the
        //      cover step (marchLen / cap), so a large domain coarsens
        //      gracefully instead of being truncated. Raise max_steps for both
        //      fine AND complete on big domains.
        float marchLen = max(0.0, tFar - tNear);
        int   isoCap = clamp(vol.max_steps, 32, 2048);
        float fineStep = clamp(vol.step_size, vol.voxel_size * 0.1, vol.voxel_size * 0.5);
        float coverStep = marchLen / float(isoCap);
        float step = max(0.001, max(fineStep, coverStep));
        int   maxSteps = min(int(marchLen / step) + 2, isoCap + 2);

        float t      = tNear;
        float startD = sampleDensityAcc(vol, rayOrigin + rayDir * t, vdbBuf, vdbMapH, vdbAcc);
        bool  startInside = startD > ISO_THRESH;
        float prevD  = startD;
        float hitT   = -1.0;

        for (int s = 0; s < maxSteps; ++s) {
            float nextT = min(t + step, tFar);
            float curD = sampleDensityAcc(vol, rayOrigin + rayDir * nextT, vdbBuf, vdbMapH, vdbAcc);
            bool crossed = startInside
                ? (prevD >= ISO_THRESH && curD < ISO_THRESH)   // exit
                : (prevD <  ISO_THRESH && curD >= ISO_THRESH); // enter
            if (crossed) {
                // Precise binary search (bisection) refinement to find the exact isosurface intersection.
                // Eliminates sawtooth / staircasing / wood-grain aliasing artifacts completely!
                float t0 = t;
                float t1 = nextT;
                float d0 = prevD;
                float d1 = curD;
                for (int it = 0; it < 4; ++it) {
                    float tMid = 0.5 * (t0 + t1);
                    float dMid = sampleDensityAcc(vol, rayOrigin + rayDir * tMid, vdbBuf, vdbMapH, vdbAcc);
                    bool midInside = dMid > ISO_THRESH;
                    if (startInside == midInside) {
                        t0 = tMid;
                        d0 = dMid;
                    } else {
                        t1 = tMid;
                        d1 = dMid;
                    }
                }
                float denom = d1 - d0;
                float frac = (ISO_THRESH - d0) / ((abs(denom) > EPSILON) ? denom : EPSILON);
                frac = clamp(frac, 0.0, 1.0);
                hitT = t0 + frac * (t1 - t0);
                break;
            }
            prevD = curD;
            t = nextT;
            if (t >= tFar) break;
        }

        // ── Whitewater foam composited in FRONT of the surface ──────────────
        // Foam rides this same volume's TEMPERATURE channel (scaled by
        // FOAM_TEMP_SCALE = 10000 on upload; see
        // SceneData::syncSimulationRenderVolumes). One volume on the domain AABB
        // means no coincident-volume case for the integrator to drop — this is
        // the fix for the SDF+foam black cube. March it as bright white single-
        // scatter in front of the iso surface; everything behind is dimmed by
        // the foam transmittance. NO self-shadow: self-shadowing is what blacked
        // out the old separate fog volume.
        vec3  foam_inscatter = vec3(0.0);
        float foam_T = 1.0;
        if (vol.vdb_temp_address != 0) {
            const float FOAM_TEMP_SCALE = 10000.0;
            // Opacity rides _ext_reserved[6] (foam_shader density × scatter,
            // packed at sync); fall back to a sane default if unset.
            float foamOptical = vol._ext_reserved[6] > 1e-3 ? vol._ext_reserved[6] : 8.0;
            // Foam tint rides _ext_reserved[3..5] (foam_shader scattering colour);
            // default to a faint cool white.
            vec3 foamAlbedo = vec3(vol._ext_reserved[3], vol._ext_reserved[4], vol._ext_reserved[5]);
            if (dot(foamAlbedo, vec3(1.0)) < 1e-3) foamAlbedo = vec3(0.95, 0.97, 1.0);

            float foamEnd = (hitT < 0.0) ? tFar : hitT;
            float foamLen = foamEnd - tNear;
            if (foamLen > 1e-5) {
                int fsteps = int(foamLen / max(vol.voxel_size, foamLen / 48.0)) + 1;
                fsteps = min(fsteps, 64);
                float fstep = foamLen / float(fsteps);
                vec3  skyAmb  = sampleSkyAmbient(rayDir);
                vec3  sunDirF = normalize(worldData.w.sunDir);
                float jitf    = rnd(payload.seed);
                for (int fs = 0; fs < fsteps && foam_T > 0.01; ++fs) {
                    float ft   = tNear + (float(fs) + jitf) * fstep;
                    vec3  fpos = rayOrigin + rayDir * ft;
                    float fd   = sampleTemperature(vol, fpos) * (1.0 / FOAM_TEMP_SCALE);
                    if (fd <= 1e-4) continue;
                    float a    = 1.0 - exp(-fd * foamOptical * fstep);

                    // Light the foam the SAME way the fog raymarch does — scene
                    // lights[] + nishita sun + sky ambient. Lighting it with ONLY
                    // the sun made foam BLACK in scene-light-lit setups (the sun
                    // term is ~0 there), and a dark in-scatter under a dimming
                    // foam_T is exactly the black-on-water the user reported.
                    // Foam is diffuse white → isotropic, NO self-shadow (bright).
                    vec3 Lf = vec3(0.0);
                    if (cam.lightCount > 0u) {
                        int li = clamp(int(floor(rnd(payload.seed) * float(cam.lightCount))),
                                       0, int(cam.lightCount) - 1);
                        LightData lt = lights.l[li];
                        int   lty = int(lt.position.w + 0.5);
                        vec3  ldir; float latten = 1.0;
                        if (lty == 1) {
                            ldir = normalize(lt.direction.xyz);
                        } else {
                            vec3 toL = lt.position.xyz - fpos;
                            float dL = length(toL);
                            ldir = (dL > EPSILON) ? (toL / dL) : vec3(0.0, 1.0, 0.0);
                            if (dL > EPSILON) latten = 1.0 / (dL * dL);
                        }
                        Lf += float(cam.lightCount) * lt.color.rgb * lt.color.a * latten;
                    }
                    if (worldData.w.mode == 2) {
                        Lf += sampleTransmittanceLUT(fpos, sunDirF)
                            * worldData.w.sunColor * worldData.w.sunIntensity;
                    }
                    Lf += skyAmb;                       // ambient fill (never fully dark)

                    foam_inscatter += foam_T * a * (foamAlbedo * Lf);
                    foam_T *= (1.0 - a);
                }
            }
        }

        // ── Solid geometry inside the domain ────────────────────────────────
        // The volume AABB is hit immediately when the ray is inside it, so the
        // iso branch would refract straight to the water surface and SKIP any
        // solid triangle sitting between the ray origin and that surface (or,
        // on a miss, anywhere in the domain). Probe for a solid in the relevant
        // span; if one is closer, stop and let it render (skipAABBs) instead of
        // refracting past it. Tints by the water depth in front of it.
        {
            float probeFar = (hitT < 0.0) ? tFar : hitT;
            const uint SOLID_FLAGS = gl_RayFlagsTerminateOnFirstHitEXT
                                   | gl_RayFlagsSkipClosestHitShaderEXT
                                   | gl_RayFlagsOpaqueEXT;
            const uint SOLID_MASK  = 0xFD;  // all solids, ignore other volume AABBs
            shadowOccluded = true;
            traceRayEXT(topLevelAS, SOLID_FLAGS, SOLID_MASK, 0, 1, 1,
                        rayOrigin, max(1e-4, tNear - 0.002), rayDir, probeFar + 0.002, 1);
            if (shadowOccluded) {
                float lo = max(tNear, 1e-4), hi = probeFar;
                for (int it = 0; it < 6; ++it) {
                    float mid = (lo + hi) * 0.5;
                    shadowOccluded = true;
                    traceRayEXT(topLevelAS, SOLID_FLAGS, SOLID_MASK, 0, 1, 1,
                                rayOrigin, max(1e-4, lo - 0.002), rayDir, mid + 0.002, 1);
                    if (shadowOccluded) hi = mid; else lo = mid;
                }
                float solidT = lo;
                // Absorption for the water the ray crossed before the solid.
                if (startInside) {
                    float depth = max(0.0, solidT - tNear);
                    payload.attenuation *= exp(-vol.absorption_color * vol.absorption_coefficient * depth);
                    payload.attenuation *= vol.scatter_color;
                }
                // Foam floating in front of the solid still shows; dim the solid
                // behind it by the foam transmittance.
                payload.radiance    += foam_inscatter;
                payload.attenuation *= foam_T;
                // Continue straight to the solid; skip volume AABBs so the
                // triangle closesthit fires on the next trace.
                payload.scatterOrigin = rayOrigin + rayDir * max(solidT - 0.01, tNear);
                payload.scatterDir    = rayDir;
                payload.skipAABBs     = true;
                payload.scattered     = true;
                payload.hitEmissive   = false;
                payload.bounceType    = 0u;
                return;
            }
        }

        if (hitT < 0.0) {
            // Ray missed the iso-surface AND no solid — pass through, but any
            // airborne foam/spray in this segment still adds light + occlusion.
            payload.radiance    += foam_inscatter;
            payload.attenuation *= foam_T;
            payload.scatterOrigin = rayOrigin + rayDir * (tFar + 0.002);
            payload.scatterDir    = rayDir;
            payload.scattered     = true;
            payload.hitEmissive   = false;
            payload.bounceType    = BOUNCE_TRANSPARENT;
            return;
        }

        vec3 hitPos = rayOrigin + rayDir * hitT;

        // Central-difference gradient on the density field for the surface
        // normal. h = voxel_size keeps the stencil aligned with the proxy
        // band thickness (band ≈ 0.5*voxel by default), so the gradient is
        // numerically well-conditioned.
        float h = max(0.001, vol.voxel_size);
        float sxp = sampleDensityAcc(vol, hitPos + vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float sxm = sampleDensityAcc(vol, hitPos - vec3(h, 0.0, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float syp = sampleDensityAcc(vol, hitPos + vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float sym = sampleDensityAcc(vol, hitPos - vec3(0.0, h, 0.0), vdbBuf, vdbMapH, vdbAcc);
        float szp = sampleDensityAcc(vol, hitPos + vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
        float szm = sampleDensityAcc(vol, hitPos - vec3(0.0, 0.0, h), vdbBuf, vdbMapH, vdbAcc);
        vec3 grad = vec3(sxp - sxm, syp - sym, szp - szm);
        float gradLen = length(grad);

        // Foam / whitewater: SDF Laplacian = surface curvature. High |curvature|
        // = wave crest / breaking edge / splash -> whiten. Reuses the 6 gradient
        // samples + the centre (≈iso), so it's nearly free.
        float foam_strength = clamp(vol._ext_reserved[2], 0.0, 1.0);
        if (foam_strength > 1e-3) {
            float dc = sampleDensityAcc(vol, hitPos, vdbBuf, vdbMapH, vdbAcc);
            float lap = abs((sxp + sxm + syp + sym + szp + szm) - 6.0 * dc);
            float foam = foam_strength * smoothstep(0.15, 0.7, lap);
            // Bright white whitewater, lit by the current throughput.
            payload.radiance += payload.attenuation * foam * vec3(0.9);
        }
        // Density increases TOWARD fluid interior, so -gradient points OUT of
        // the surface (toward the less-dense / air side).
        vec3 N = (gradLen > 1e-6) ? normalize(-grad) : -rayDir;

        // ── Rough dielectric event (Fresnel importance-sampled). ────────────
        // Orient the geometric normal against the incoming ray.
        if (dot(rayDir, N) > 0.0) N = -N;

        // GGX roughness: jitter the normal inside a microfacet lobe so both the
        // reflection AND the refraction blur with surface_roughness
        // (_ext_reserved[1]). 0 = mirror-smooth still water.
        float roughness = clamp(vol._ext_reserved[1], 0.0, 1.0);
        if (roughness > 1e-3) {
            float a = roughness * roughness;
            float u1 = rnd(payload.seed);
            float u2 = rnd(payload.seed);
            float phi = TWO_PI * u1;
            float cosT = sqrt(max(0.0, (1.0 - u2) / (1.0 + (a * a - 1.0) * u2)));
            float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
            vec3 hT = vec3(sinT * cos(phi), sinT * sin(phi), cosT);
            vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            vec3 T = normalize(cross(up, N));
            vec3 B = cross(N, T);
            vec3 Np = normalize(hT.x * T + hT.y * B + hT.z * N);
            if (dot(rayDir, Np) < 0.0) N = Np;  // keep facing the ray
        }

        // Beer-Lambert over the segment the ray just traversed INSIDE the fluid
        // (only when it started inside — i.e. this is an exit / internal event).
        // Pure water is clear; the blue comes from this depth absorption.
        if (startInside) {
            float depth = max(0.0, hitT - tNear);
            vec3 sigmaA = vol.absorption_color * vol.absorption_coefficient;
            payload.attenuation *= exp(-sigmaA * depth);
        }

        // Fresnel, importance-sampled: reflect (scene/sky) vs refract (through).
        // Picking the branch by Fresnel probability means the per-branch weight
        // is 1 — no (1-fresnel)/fresnel multiply, far less noise than splitting.
        float eta = startInside ? IOR_WATER : (1.0 / IOR_WATER);
        float cosTheta = clamp(abs(dot(rayDir, N)), 0.0, 1.0);
        float r0 = (1.0 - IOR_WATER) / (1.0 + IOR_WATER);
        r0 = r0 * r0;
        float fresnel = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);

        vec3 refrDir = refract(rayDir, N, eta);
        bool tir = dot(refrDir, refrDir) < 1e-6;
        vec3 outDir;
        if (tir || rnd(payload.seed) < fresnel) {
            // Reflection — traces the scene/sky next bounce. No water cast.
            outDir = reflect(rayDir, N);
        } else {
            // Refraction — through the water; mild surface cast.
            outDir = normalize(refrDir);
            payload.attenuation *= vol.scatter_color;
        }

        // Whitewater foam in front of the surface: add its in-scatter now and
        // dim the redirected lobe (water/sky behind) by the foam transmittance.
        payload.radiance    += foam_inscatter;
        payload.attenuation *= foam_T;

        payload.scatterOrigin       = hitPos + outDir * 0.003;
        payload.scatterDir          = outDir;
        payload.scattered           = true;
        payload.hitEmissive         = false;
        payload.primaryAlbedo       = vol.scatter_color;
        payload.primaryNormal       = N;
        payload.primaryTransmission = 1.0;
        payload.bounceType          = 0u;
        return;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // SOLID SURFACE DETECTION inside the volume AABB
    // If a solid triangle exists between tNear and tFar, we must stop the march
    // just before it and signal raygen to fire the next bounce with
    // gl_RayFlagsSkipAABBEXT so the triangle closesthit fires correctly.
    //
    // Strategy: 1 initial probe + 6 binary-search probes (7 shadow rays total)
    // to locate the solid's t value to ~(tFar-tNear)/64 precision.
    // Only fires when a solid is actually inside the volume (rare case).
    // ══════════════════════════════════════════════════════════════════════════
    float solidT = -1.0;  // -1 = no solid found
    {
        // Performance gate:
        // In optically thick segments, inner solid surfaces are effectively invisible.
        // Skip costly triangle probes in those cases.
        bool needSolidProbe = true;
        float marchDistProbe = max(tFar - tNear, 0.0);
        float sigmaTCoeff = max(vol.scatter_coefficient + vol.absorption_coefficient, 0.0);
        if (sigmaTCoeff > EPSILON && marchDistProbe > 1e-4) {
            float tA = tNear + min(0.05 * marchDistProbe, 0.25);
            float tB = tNear + 0.5 * marchDistProbe;
            float dA = sampleDensityAcc(vol, rayOrigin + rayDir * tA, vdbBuf, vdbMapH, vdbAcc);
            float dB = sampleDensityAcc(vol, rayOrigin + rayDir * tB, vdbBuf, vdbMapH, vdbAcc);
            float dAvg = max(0.0, 0.4 * dA + 0.6 * dB);
            float tauEst = dAvg * sigmaTCoeff * marchDistProbe;
            float transEst = exp(-tauEst);
            float probeThreshold = cameraInsideVolume ? 0.08 : 0.03;
            needSolidProbe = (transEst > probeThreshold);
        }

        if (needSolidProbe) {
            const uint PROBE_FLAGS = gl_RayFlagsTerminateOnFirstHitEXT
                                   | gl_RayFlagsSkipClosestHitShaderEXT
                                   | gl_RayFlagsOpaqueEXT;
            const uint PROBE_MASK  = 0xFD; // ~0x02: check all solid geometry, ignore other volume AABBs

            // Initial check: any solid in [tNear, tFar]?
            shadowOccluded = true;
            traceRayEXT(topLevelAS, PROBE_FLAGS, PROBE_MASK, 0, 1, 1,
                        rayOrigin, max(1e-4, tNear - 0.002), rayDir, tFar + 0.002, 1);
            if (shadowOccluded) {
                // Binary search to find approximate t of first solid hit
                float lo = tNear, hi = tFar;
                int probeIters = cameraInsideVolume ? 2 : 5;
                for (int it = 0; it < probeIters; it++) {
                    float mid = (lo + hi) * 0.5;
                    shadowOccluded = true;
                    // Provide a small overlap in the search window to prevent near-misses
                    traceRayEXT(topLevelAS, PROBE_FLAGS, PROBE_MASK, 0, 1, 1,
                                rayOrigin, max(1e-4, lo - 0.002), rayDir, mid + 0.002, 1);
                    if (shadowOccluded) hi = mid;  // solid is in [lo, mid]
                    else                lo = mid;  // solid is in (mid, hi]
                }
                solidT = lo;
                // Clamp march to just before the solid
                tFar = solidT - 0.01;
                // If the solid is essentially at or before the entry point, skip AABBs on next bounce
                if (tFar <= tNear) {
                    payload.radiance = vec3(0.0);
                    payload.scatterOrigin = rayOrigin + rayDir * max(solidT - 0.01, tNear);
                    payload.scatterDir = rayDir;
                    payload.scattered = true;
                    payload.hitEmissive = false;
                    payload.skipAABBs = true;
                    return;
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // RAY MARCH through volume (Regular stepping with jitter)
    // Matches OptiX volumetric ray march approach
    // ══════════════════════════════════════════════════════════════════════════
    float stepSize = max(vol.step_size, 1e-4);
    float marchDist = tFar - tNear;
    int maxSteps = max(vol.max_steps, 1);
    float minStepToCover = marchDist / float(maxSteps);
    float baseStep = max(stepSize, minStepToCover);
    float minStep = max(baseStep * 0.25, 1e-4);
    const float tauMax = 0.2;
    
    float sigma_s_coeff = vol.scatter_coefficient;
    float sigma_a_coeff = vol.absorption_coefficient;
    
    vec3  accumulated_radiance = vec3(0.0);
    float transmittance = 1.0;
    bool  didScatter = false;
    float scatterT = tFar; // will be set if scatter event happens
    vec3 ambientSky = sampleSkyAmbient(rayDir);

    // Jitter first sample to reduce banding. Mix in ray state so horizon rays do
    // not share coherent step planes across neighboring pixels.
    float rayJitter = fract(sin(dot(rayOrigin + rayDir * 17.0, vec3(12.9898, 78.233, 37.719)) + float(payload.seed) * 0.000173) * 43758.5453);
    float t = tNear + rayJitter * baseStep;
    int step = 0;
    while (t < tFar && step < maxSteps) {
        vec3  samplePos = rayOrigin + rayDir * t;
        
        float density = sampleDensityAcc(vol, samplePos, vdbBuf, vdbMapH, vdbAcc);
        float sigma_s_local = density * sigma_s_coeff;
        float sigma_a_local = density * sigma_a_coeff;
        float sigma_t_local = sigma_s_local + sigma_a_local;
        // Let low-density regions survive based on scattering optical depth, not
        // raw density alone. This means increasing scatter can actually keep thin
        // edges visible instead of them being discarded by a fixed density gate.
        float sparseCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.04;
        float scatter_keep = clamp((sigma_s_local * baseStep) / sparseCutoff, 0.0, 1.0);
        if (rnd(payload.seed) > scatter_keep) {
            t += baseStep;
            step++;
            continue;
        }
        if (sigma_t_local <= EPSILON) {
            t += baseStep;
            step++;
            continue;
        }
        
        // Optical-depth-limited adaptive step (industry-standard robustness in dense regions)
        float dt = min(baseStep, tauMax / sigma_t_local);
        dt = max(dt, minStep);
        dt = min(dt, tFar - t);
        if (dt <= 1e-6) break;

        // Current extinction
        float extinction = sigma_t_local * dt;
        float sampleTransmittance = exp(-extinction);
        
        // ── Multi-scatter transmittance blend (matches OptiX) ──
        // Blends single-scatter (Beer's law) with a softer 0.25x extinction approximation
        // to model multiple scattering. When scatter_multi > 0, volumes appear brighter
        // and more translucent — matching the OptiX renderer output.
        if (vol.scatter_multi > 0.0 && sigma_s_local > 0.0) {
            float albedo_avg = dot(vol.scatter_color, vec3(0.2126, 0.7152, 0.0722));
            float T_multi = exp(-extinction * 0.25);
            sampleTransmittance = sampleTransmittance * (1.0 - vol.scatter_multi * albedo_avg)
                                 + T_multi * (vol.scatter_multi * albedo_avg);
        }
        float one_minus_sampleT = 1.0 - sampleTransmittance;
        
        // ── Volume Emission ──
        // Mode 0 = none, 1 = plain color, 2 = blackbody/color-ramp via temperature grid.
        // Energy-stable integration: multiply by one_minus_sampleT (bounded by 1) instead of dt.
        vec3 emis = vec3(0.0);
        if (vol.emission_mode >= 1) {
            if (vol.emission_mode == 1) {
                // Plain constant-color emission
                emis = vol.emission_color * vol.emission_intensity * density;
            } else if (vol.emission_mode >= 2) {
                // Blackbody / color-ramp — temperature grid first, density fallback for parity.
                float temperature = sampleTemperature(vol, samplePos);
                if (temperature <= 0.0) temperature = density;

                vec3 e_color;
                if (vol.color_ramp_enabled != 0 && vol.ramp_stop_count > 0) {
                    float t_ramp = (temperature > 20.0)
                        ? ((vol.max_temperature > 20.0) ? (temperature / vol.max_temperature) : (temperature / 6000.0))
                        : temperature;
                    e_color = sampleColorRamp(vol, clamp(t_ramp * vol.temperature_scale, 0.0, 1.0));
                } else {
                    // Matches OptiX: raw kelvin if temp > 20 (already in K), else normalize [0..1]→[1000..4000K]
                    float kelvin = (temperature > 20.0)
                        ? temperature * vol.temperature_scale
                        : (temperature * 3000.0 + 1000.0) * vol.temperature_scale;
                    e_color = blackbodyToRGB(kelvin);
                }
                emis = e_color * density * vol.blackbody_intensity;
            }
        }
        
        // ── In-Scattering (Direct Lighting) ──
        if (sigma_s_local > 0.0) {
            vec3 inscatter = vec3(0.0);
            
            // Sample scene lights for object volumes. For the procedural sky cloud,
            // Nishita already provides the sun/sky block below; sampling the default
            // directional light here doubles the light march cost and shifts parity
            // away from the OptiX sky-cloud path.
            bool useSceneLights = !(worldData.w.mode == 2 && (vol.volume_type == 3 || vol.source_type == 3));
            if (useSceneLights && cam.lightCount > 0u) {
                int maxLightsToSample = min(int(cam.lightCount), 2);
                float lightWeight = float(cam.lightCount) / float(maxLightsToSample);
                for (int ls = 0; ls < maxLightsToSample; ls++) {
                    int li = int(floor(rnd(payload.seed) * float(cam.lightCount)));
                    li = clamp(li, 0, int(cam.lightCount) - 1);
                    LightData light = lights.l[li];
                    int lightType = int(light.position.w + 0.5);
                    
                    vec3 lightDir;
                    float lightDist;
                    float lightAtten = 1.0;
                    
                    if (lightType == 1) {
                        // Directional light
                        lightDir = normalize(light.direction.xyz);
                        lightDist = 1e6;
                    } else {
                        // Point/spot/area light
                        vec3 toLight = light.position.xyz - samplePos;
                        lightDist = length(toLight);
                        if (lightDist < EPSILON) continue;
                        lightDir = toLight / lightDist;
                        lightAtten = 1.0 / (lightDist * lightDist);
                    }
                    
                    // Phase function evaluation
                    float cosTheta = dot(rayDir, lightDir);
                    float phase = dualLobeHG(cosTheta, vol.scatter_anisotropy, 
                                             vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                    float powder = gpu_powder_effect(density, cosTheta);
                    phase *= (1.0 + powder * 0.5);
                    
                    // Light march transmittance through volume toward light.
                    // Geometry occlusion is NOT checked per-step (matching OptiX raymarch_volumetric_object
                    // behavior): solid objects are handled by TLAS traversal order, not per-sample rays.
                    // This prevents hard black shadows from solids inside the volume.
                    float shadowTr = 1.0;
                    if (sigma_t_local * dt > 0.02) {
                        float shadowMaxDist = min(lightDist, max(8.0 * baseStep, marchDist * 0.35));
                        shadowTr = lightMarchAcc(vol, samplePos, lightDir, shadowMaxDist, vdbBuf, vdbMapH, vdbAcc);
                    }
                    
                    vec3 lightColor = light.color.rgb * light.color.a;
                    inscatter += lightWeight * lightColor * lightAtten * phase * shadowTr
                               * vol.scatter_color * sigma_s_local;
                }
            }
            
            // Sun/sky light contribution (if Nishita sky active)
            if (worldData.w.mode == 2) {
                vec3 sunDir = normalize(worldData.w.sunDir);
                float cosSun = dot(rayDir, sunDir);
                float sunPhase = dualLobeHG(cosSun, vol.scatter_anisotropy, 
                                            vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                float sunPowder = gpu_powder_effect(density, cosSun);
                sunPhase *= (1.0 + sunPowder * 0.5);
                
                float sunShadowTr = 1.0;
                if (sigma_t_local * dt > 0.02) {
                    float sunShadowMaxDist = max(12.0 * baseStep, marchDist * 0.45);
                    sunShadowTr = lightMarchAcc(vol, samplePos, sunDir, sunShadowMaxDist, vdbBuf, vdbMapH, vdbAcc);
                }
                
                vec3 sunLi = sampleTransmittanceLUT(samplePos, sunDir) * worldData.w.sunColor * worldData.w.sunIntensity;
                inscatter += sunLi * sunPhase * sunShadowTr * vol.scatter_color * sigma_s_local;
            }
            
            // 3. Sky/Ambient lighting (closer to CPU world.evaluate(up) behavior)
            float thin_scatter = scatter_keep * scatter_keep;
            inscatter += ambientSky * vol.scatter_color * sigma_s_local * thin_scatter;
            
            // Multi-scatter ambient + source boost (matches OptiX ms_boost)
            vec3 ms_boost = vec3(1.0) + vol.scatter_color * vol.scatter_multi * (2.0 * thin_scatter);
            inscatter *= ms_boost;
            
            // CPU parity integration:
            // step_color = source * (1 - step_transmittance)
            // accumulated += step_color * current_transparency
            accumulated_radiance += transmittance * (inscatter + emis) * one_minus_sampleT;
        } else if (any(greaterThan(emis, vec3(0.0)))) {
            // Emission-only medium segment
            accumulated_radiance += transmittance * emis * one_minus_sampleT;
        }
        
        // Update transmittance
        transmittance *= sampleTransmittance;
        
        // ── Stochastic Scatter Event (Delta Tracking) ──
        // Probability of scattering at this step
        float scatterProb = 1.0 - exp(-sigma_s_local * dt);
        scatterProb = clamp(scatterProb, 0.0, 1.0);
        if (rnd(payload.seed) < scatterProb && !didScatter) {
            didScatter = true;
            scatterT = t;
        }
        
        // Early termination if transmittance is negligible
        if (transmittance < 0.01) {
            transmittance = 0.0;
            break;
        }
        t += dt;
        step++;
    }
    
    // ══════════════════════════════════════════════════════════════════════════
    // OUTPUT — Set payload for path tracer integration
    // ══════════════════════════════════════════════════════════════════════════
    
    // Sanitize against NaN/Inf. A single non-finite sample (e.g. a degenerate
    // ray when two coincident volumes — foam fog + fluid iso surface — share the
    // domain AABB) otherwise poisons the accumulation buffer permanently → the
    // whole domain goes black until accumulation resets. Drop the bad sample.
    if (any(isnan(accumulated_radiance)) || any(isinf(accumulated_radiance)))
        accumulated_radiance = vec3(0.0);
    if (isnan(transmittance) || isinf(transmittance) || transmittance < 0.0)
        transmittance = 1.0;

    // Accumulated in-scattered radiance
    payload.radiance  = accumulated_radiance;
    payload.skipAABBs = false; // default; overridden below if solid found
    // ── Solid surface found inside the volume: hand off to closesthit ──
    // March was already clamped to solidT. Now position the scatter ray
    // just before the solid surface and tell raygen to skip AABBs next
    // bounce (gl_RayFlagsSkipAABBEXT) so the triangle closesthit fires.
    if (solidT >= 0.0) {
        payload.attenuation  *= vec3(transmittance);
        payload.scatterOrigin = rayOrigin + rayDir * (solidT - 0.01);
        payload.scatterDir    = rayDir;
        payload.scattered     = true;
        payload.hitEmissive   = false;
        payload.skipAABBs     = true;
        return;
    }

    float volumeContribution = max(max(accumulated_radiance.r, accumulated_radiance.g), accumulated_radiance.b);
    float volumeOpacity = 1.0 - transmittance;
    // Keep ultra-thin fog tails out of the primary auxiliary buffers. They were
    // being treated as first-hit geometry, which overemphasized weak density
    // regions in Vulkan RT denoiser output compared to OptiX.
    bool primaryVolumeInteraction = didScatter || volumeOpacity > 0.04 || volumeContribution > 5e-4;
    if (payload.primaryHit == 0u && primaryVolumeInteraction) {
        payload.primaryAlbedo = vol.scatter_color;
        payload.primaryNormal = -rayDir;
        payload.primaryHit = 1u;
        payload.primaryTransmission = transmittance;
        payload.primaryMetallic = 0.0;
    }

    if (didScatter && transmittance < 0.99) {
        // Scatter event — continue path with new direction
        vec3 scatterPos = rayOrigin + rayDir * scatterT;
        
        // Choose lobe for direction sampling
        float g = (rnd(payload.seed) < vol.scatter_lobe_mix) 
                  ? vol.scatter_anisotropy 
                  : vol.scatter_anisotropy_back;
        vec3 newDir = sampleHG(rayDir, g, payload.seed);
        
        payload.scatterOrigin = scatterPos;
        payload.scatterDir    = newDir;
        payload.attenuation  *= vol.scatter_color * transmittance;
        payload.scattered     = true;
        payload.hitEmissive   = false;
    } else {
        // No scatter — attenuate throughput by volume transmittance
        payload.attenuation *= vec3(transmittance);
        
        // Set scattered = true with original direction to let the ray continue through
        // Ensure forward progress to avoid re-hitting the same boundary when camera is inside.
        payload.scatterOrigin = rayOrigin + rayDir * (tFar + 0.002);
        payload.scatterDir    = rayDir;
        payload.scattered     = (transmittance > 0.01); // Stop if fully absorbed
        payload.hitEmissive   = (vol.emission_mode >= 1 && transmittance < 0.99);
        if (volumeOpacity <= 0.04 && volumeContribution <= 5e-4) {
            payload.bounceType = BOUNCE_TRANSPARENT;
        }
    }
}
