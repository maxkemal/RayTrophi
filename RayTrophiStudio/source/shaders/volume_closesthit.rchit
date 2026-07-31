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
    float waterTime;
} cam;

// ============================================================
// Payload — raygen/closesthit ile tam eşleşme
// ============================================================
// Payload — shared ABI, single source of truth
#include "rt_payload.glsl"
#include "volume_instrumentation.glsl"

const uint BOUNCE_TRANSPARENT = 3u;

layout(location = 0) rayPayloadInEXT RayPayload payload;
// Shadow payload: rgb = transmissive tint, w = reached-light flag (0 = hit/occluded).
// This shader only uses it as a binary probe (w), with OpaqueEXT so any-hits never run.
layout(location = 1) rayPayloadEXT vec4 shadowPayload;

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
layout(set = 0, binding = 6) uniform sampler2D materialTextures[];
// Volume programs are live inside a long ray-march loop. Keeping the surface
// VM's 32 vec3 register file here cuts GPU occupancy even for legacy VDBs whose
// runtime branch never evaluates a graph. The host only binds programs whose
// compacted peak-live count fits this volume-specific budget.
#define MP_REGISTER_COUNT 12
#include "material_program.glsl"

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
    int   shadow_stride;
    
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
    // [6] is Surface-SDF foam opacity for source_type 4, otherwise authored
    // minimum emission temperature. [7..11] carry density-noise parameters.
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

// Live gas domains expose their Vulkan compute fields directly to RT. The
// dense grid stores cell-centred floats in x-major order.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer DenseGasFloatGrid {
    float values[];
};

float sampleDenseGasFloat(uint64_t address, VkVolumeInstance vol, vec3 localPos) {
    if (address == 0) return 0.0;

    ivec3 resolution = ivec3(
        int(vol._ext_reserved[0] + 0.5),
        int(vol._ext_reserved[1] + 0.5),
        int(vol._ext_reserved[2] + 0.5));
    if (any(lessThanEqual(resolution, ivec3(0)))) return 0.0;

    float voxelSize = max(vol.voxel_size, 1e-6);
    vec3 gridOrigin = vec3(
        vol._ext_reserved[3],
        vol._ext_reserved[4],
        vol._ext_reserved[5]);
    vec3 gridPos = (localPos - gridOrigin) / voxelSize - vec3(0.5);
    if (any(lessThan(gridPos, vec3(-0.5))) ||
        any(greaterThan(gridPos, vec3(resolution) - vec3(0.5)))) {
        return 0.0;
    }

    ivec3 p0 = clamp(ivec3(floor(gridPos)), ivec3(0), resolution - ivec3(1));
    ivec3 p1 = min(p0 + ivec3(1), resolution - ivec3(1));
    vec3 f = clamp(gridPos - vec3(p0), vec3(0.0), vec3(1.0));
    DenseGasFloatGrid grid = DenseGasFloatGrid(address);
    int xyStride = resolution.x * resolution.y;

    int i000 = p0.x + p0.y * resolution.x + p0.z * xyStride;
    int i100 = p1.x + p0.y * resolution.x + p0.z * xyStride;
    int i010 = p0.x + p1.y * resolution.x + p0.z * xyStride;
    int i110 = p1.x + p1.y * resolution.x + p0.z * xyStride;
    int i001 = p0.x + p0.y * resolution.x + p1.z * xyStride;
    int i101 = p1.x + p0.y * resolution.x + p1.z * xyStride;
    int i011 = p0.x + p1.y * resolution.x + p1.z * xyStride;
    int i111 = p1.x + p1.y * resolution.x + p1.z * xyStride;

    float z0 = mix(
        mix(grid.values[i000], grid.values[i100], f.x),
        mix(grid.values[i010], grid.values[i110], f.x),
        f.y);
    float z1 = mix(
        mix(grid.values[i001], grid.values[i101], f.x),
        mix(grid.values[i011], grid.values[i111], f.x),
        f.y);
    return mix(z0, z1, f.z);
}

// Conservative NanoVDB hierarchy skip. Leaf voxels cannot be skipped at this
// hierarchy level, so avoid the second is_active lookup when dim <= 1.
// Inactive coarse tiles advance to one voxel before their exit so trilinear
// reconstruction still sees density across the active boundary.
float nanoEmptyTileStep(
    VkVolumeInstance vol,
    vec3 worldPos,
    vec3 worldDir,
    float baseStep,
    pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    out uint skipKind)
{
    skipKind = 0u;
    if (buf.address == 0) return baseStep;

    vec3 localP;
    localP.x = vol.inv_transform[0]*worldPos.x + vol.inv_transform[1]*worldPos.y
             + vol.inv_transform[2]*worldPos.z + vol.inv_transform[3] - vol.pivot_offset[0];
    localP.y = vol.inv_transform[4]*worldPos.x + vol.inv_transform[5]*worldPos.y
             + vol.inv_transform[6]*worldPos.z + vol.inv_transform[7] - vol.pivot_offset[1];
    localP.z = vol.inv_transform[8]*worldPos.x + vol.inv_transform[9]*worldPos.y
             + vol.inv_transform[10]*worldPos.z + vol.inv_transform[11] - vol.pivot_offset[2];

    vec3 worldP1 = worldPos + worldDir;
    vec3 localP1;
    localP1.x = vol.inv_transform[0]*worldP1.x + vol.inv_transform[1]*worldP1.y
              + vol.inv_transform[2]*worldP1.z + vol.inv_transform[3] - vol.pivot_offset[0];
    localP1.y = vol.inv_transform[4]*worldP1.x + vol.inv_transform[5]*worldP1.y
              + vol.inv_transform[6]*worldP1.z + vol.inv_transform[7] - vol.pivot_offset[1];
    localP1.z = vol.inv_transform[8]*worldP1.x + vol.inv_transform[9]*worldP1.y
              + vol.inv_transform[10]*worldP1.z + vol.inv_transform[11] - vol.pivot_offset[2];

    pnanovdb_vec3_t p0 = pnanovdb_vec3_uniform(0.0);
    p0.x = localP.x; p0.y = localP.y; p0.z = localP.z;
    pnanovdb_vec3_t p1 = pnanovdb_vec3_uniform(0.0);
    p1.x = localP1.x; p1.y = localP1.y; p1.z = localP1.z;
    pnanovdb_vec3_t ip0 = pnanovdb_map_apply_inverse(buf, mapH, p0);
    pnanovdb_vec3_t ip1 = pnanovdb_map_apply_inverse(buf, mapH, p1);
    vec3 idx = vec3(ip0.x, ip0.y, ip0.z);
    vec3 idxDir = vec3(ip1.x - ip0.x, ip1.y - ip0.y, ip1.z - ip0.z);

    pnanovdb_coord_t ijk;
    ijk.x = int(floor(idx.x)); ijk.y = int(floor(idx.y)); ijk.z = int(floor(idx.z));
    uint dim = pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk);
    if (dim <= 1u) return baseStep;
    bool regionIsActive = pnanovdb_readaccessor_is_active(
        PNANOVDB_GRID_TYPE_FLOAT, buf, acc, ijk);
    if (regionIsActive) return baseStep;

    vec3 tileMin = floor(idx / float(dim)) * float(dim);
    vec3 tileMax = tileMin + vec3(float(dim));
    vec3 boundary = mix(tileMin, tileMax, greaterThan(idxDir, vec3(0.0)));
    vec3 axisT = vec3(1e30);
    if (abs(idxDir.x) > 1e-8) axisT.x = (boundary.x - idx.x) / idxDir.x;
    if (abs(idxDir.y) > 1e-8) axisT.y = (boundary.y - idx.y) / idxDir.y;
    if (abs(idxDir.z) > 1e-8) axisT.z = (boundary.z - idx.z) / idxDir.z;
    float tileExit = min(axisT.x, min(axisT.y, axisT.z));
    if (!(tileExit > baseStep)) return baseStep;
    skipKind = 1u;
    return max(baseStep, tileExit - max(vol.voxel_size, baseStep));
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

// OptiX parity: preserve hue while limiting tiny, extremely hot cells that
// otherwise dominate rare indirect paths and show up as volume fireflies.
vec3 clampVolumeRadiance(vec3 c, float maxLuma) {
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return (luma > maxLuma && luma > 1e-6) ? c * (maxLuma / luma) : c;
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

    if (vol.volume_type == 4 && vol.source_type == 5) {
        // Simulation heat is normalized; retain the old live-VDB Kelvin scale.
        return sampleDenseGasFloat(vol.vdb_temp_address, vol, localPos) * 3000.0;
    }

    return sampleNanoVDBFloatTrilinear(vol.vdb_temp_address, localPos);
}

float applyMaterialDensityNoise(VkVolumeInstance vol, vec3 localPos, float density) {
    if (vol._ext_reserved[7] < 0.5 || density <= 0.0) return density;
    vec3 extent = max(vol.aabb_max - vol.aabb_min, vec3(1e-5));
    vec3 p = (localPos - vol.aabb_min) / extent;
    float scale = max(vol._ext_reserved[8], 0.001);
    float strength = clamp(vol._ext_reserved[9], 0.0, 1.0);
    int detail = clamp(int(vol._ext_reserved[10] + 0.5), 1, 8);
    float seed = vol._ext_reserved[11];
    vec3 seedOffset = vec3(seed * 0.1031, seed * 0.11369, seed * 0.13787);
    float field = fbmNoise(p * scale + seedOffset, detail);
    return density * mix(1.0, field, strength);
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
    } else if (vol.volume_type == 4 && vol.source_type == 5) {
        density = sampleDenseGasFloat(vol.vdb_grid_address, vol, localPos);
    }

    density = applyMaterialDensityNoise(vol, localPos, density);
    float remappedDensity = max((density - vol.density_remap_low) / max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0);
    float densityCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    if (remappedDensity <= densityCutoff) {
        return 0.0;
    }
    // A binary cutoff changes from empty to full extinction in one sample.
    // Against aerial perspective that discontinuity reads as a dark contour at
    // the sparse topology boundary. Preserve the authored rejection threshold,
    // then feather only the narrow band immediately above it.
    float cutoffFade = densityCutoff > 0.0
        ? smoothstep(densityCutoff, densityCutoff * 2.0, remappedDensity)
        : 1.0;
    return remappedDensity * vol.density_multiplier * cutoffFade;
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
    } else if (vol.volume_type == 4 && vol.source_type == 5) {
        density = sampleDenseGasFloat(vol.vdb_grid_address, vol, localPos);
    }
    
    density = applyMaterialDensityNoise(vol, localPos, density);
    // Apply density remap (No upper clamp, matches OptiX fmaxf)
    float remappedDensity = max((density - vol.density_remap_low) / max(vol.density_remap_high - vol.density_remap_low, EPSILON), 0.0);
    float densityCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.0;
    if (remappedDensity <= densityCutoff) {
        return 0.0;
    }

    float cutoffFade = densityCutoff > 0.0
        ? smoothstep(densityCutoff, densityCutoff * 2.0, remappedDensity)
        : 1.0;
    return remappedDensity * vol.density_multiplier * cutoffFade;
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
    inout pnanovdb_readaccessor_t acc,
    float shadowStrengthOverride)
{
    if (vol.shadow_steps <= 0) return 1.0;
    if (maxDist <= 1e-4) return 1.0;

    vec3 shadowAbsColor = max(vol.absorption_color, vec3(0.0));
    float shadowAbsPeak = max(
        shadowAbsColor.r, max(shadowAbsColor.g, shadowAbsColor.b));
    vec3 shadowAbsWeights = shadowAbsPeak > 1e-5
        ? shadowAbsColor / shadowAbsPeak : vec3(1.0);
    float sigma_t = vol.scatter_coefficient +
        vol.absorption_coefficient *
        dot(shadowAbsWeights, vec3(0.2126, 0.7152, 0.0722));
    if (sigma_t <= EPSILON) return 1.0;

    int reqSteps = clamp(vol.shadow_steps, 1, 64);
    uint shadowMatIndex = vol._reserved[1] > 0.5 ? uint(vol._reserved[1] - 1.0) : 0u;
    bool allowSparseTraversal = buf.address != 0 &&
        (vol._reserved[1] <= 0.5 || matProgramOffset(shadowMatIndex) == MATPROG_NONE);
    // maxDist is the chord from this scatter point to the volume boundary.
    // Integrate the whole chord; half-length marching over-lights the far side
    // of dense gas and erases the rolling self-shadow detail.
    float marchLength = maxDist;
    float midpointDensity = sampleDensityAcc(
        vol, pos + lightDir * (0.5 * marchLength), buf, mapH, acc);
    float tauHint = max(midpointDensity, 0.0) * sigma_t * marchLength;
    float stepScale = clamp(sqrt(max(tauHint, 0.0)), 0.20, 1.0);
    // Dense gas needs eight buffer loads per trilinear density lookup. Twelve
    // jittered samples over the exact exit chord retain soft self-shadow detail
    // while avoiding the 16-sample worst case on every cached shadow update.
    int shadowSampleCap = (vol.volume_type == 4 && vol.source_type == 5) ? 12 : 16;
    int steps = clamp(
        int(ceil(float(min(reqSteps, shadowSampleCap)) * stepScale)),
        min(3, reqSteps),
        min(reqSteps, shadowSampleCap));
    float stepSize = marchLength / float(max(steps, 1));
    stepSize = max(stepSize, 1e-5);
    float jitter = fract(sin(dot(pos, vec3(12.9898, 78.233, 37.719)) +
                             dot(lightDir, vec3(39.346, 11.135, 83.155))) * 43758.5453);

    float s_trans = 0.0;
    float distanceAlongRay = min((jitter + 0.5) * stepSize, marchLength);
    int densitySamples = 1; // includes midpoint density used for adaptive step count
    int traversalIters = 0;
    int maxTraversalIters = steps * 4 + 8;
    while (distanceAlongRay < marchLength &&
           densitySamples < steps &&
           traversalIters < maxTraversalIters) {
        vec3 samplePos = pos + lightDir * distanceAlongRay;
        if (allowSparseTraversal) {
            uint shadowSkipKind = 0u;
            float sparseStep = nanoEmptyTileStep(
                vol, samplePos, lightDir, stepSize, buf, mapH, acc,
                shadowSkipKind);
            if (sparseStep > stepSize * 1.01) {
                distanceAlongRay += min(sparseStep, marchLength - distanceAlongRay);
                traversalIters++;
                continue;
            }
        }

        float d = sampleDensityAcc(vol, samplePos, buf, mapH, acc);
        s_trans += d * sigma_t * stepSize;
        if (s_trans > 10.0) break;
        distanceAlongRay += stepSize;
        densitySamples++;
        traversalIters++;
    }
    volumeRecordShadowSamples(uint(densitySamples));

    float phys_trans = exp(-s_trans);

    float shadowStrength = (shadowStrengthOverride >= 0.0)
        ? clamp(shadowStrengthOverride, 0.0, 1.0)
        : clamp(vol.shadow_strength * 0.92, 0.0, 1.0);
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
    vec3 shadowAbsColor = max(vol.absorption_color, vec3(0.0));
    float shadowAbsPeak = max(
        shadowAbsColor.r, max(shadowAbsColor.g, shadowAbsColor.b));
    vec3 shadowAbsWeights = shadowAbsPeak > 1e-5
        ? shadowAbsColor / shadowAbsPeak : vec3(1.0);
    float sigma_t = vol.scatter_coefficient +
        vol.absorption_coefficient *
        dot(shadowAbsWeights, vec3(0.2126, 0.7152, 0.0722));
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
    uint measuredShadowSamples = 1u; // includes dMid used for adaptive step count
    for (int i = 0; i < steps; i++) {
        vec3 samplePos = pos + lightDir * (float(i) + jitter + 0.5) * stepSize;
        float d = sampleDensity(vol, samplePos);
        measuredShadowSamples++;
        s_trans += d * sigma_t * stepSize;
        if (s_trans > 10.0) break; // fully occluded
    }
    volumeRecordShadowSamples(measuredShadowSamples);
    
    // Multi-scatter shadow blend — matches OptiX:
    // Beer-Lambert shadow transmission. Shadow strength remains an artistic
    // blend between the physical result and an unshadowed result.
    float phys_trans = exp(-s_trans);
    
    float shadowStrength = clamp(vol.shadow_strength * 0.92, 0.0, 1.0);
    return 1.0 - shadowStrength * (1.0 - phys_trans);
}

// ============================================================
// Main — Volume Ray March Entry Point
// ============================================================
// Distance, in world-ray parameter units, from a point in the volume to the
// local AABB exit along the light direction.
float volumeExitDistance(VkVolumeInstance vol, vec3 worldPos, vec3 worldDir) {
    vec3 lp;
    lp.x = vol.inv_transform[0] * worldPos.x + vol.inv_transform[1] * worldPos.y
         + vol.inv_transform[2] * worldPos.z + vol.inv_transform[3];
    lp.y = vol.inv_transform[4] * worldPos.x + vol.inv_transform[5] * worldPos.y
         + vol.inv_transform[6] * worldPos.z + vol.inv_transform[7];
    lp.z = vol.inv_transform[8] * worldPos.x + vol.inv_transform[9] * worldPos.y
         + vol.inv_transform[10] * worldPos.z + vol.inv_transform[11];
    vec3 ld;
    ld.x = vol.inv_transform[0] * worldDir.x + vol.inv_transform[1] * worldDir.y
         + vol.inv_transform[2] * worldDir.z;
    ld.y = vol.inv_transform[4] * worldDir.x + vol.inv_transform[5] * worldDir.y
         + vol.inv_transform[6] * worldDir.z;
    ld.z = vol.inv_transform[8] * worldDir.x + vol.inv_transform[9] * worldDir.y
         + vol.inv_transform[10] * worldDir.z;
    const float DIR_EPS = 1e-8;
    vec3 invDir = vec3(
        abs(ld.x) > DIR_EPS ? 1.0 / ld.x : (ld.x >= 0.0 ? 1e20 : -1e20),
        abs(ld.y) > DIR_EPS ? 1.0 / ld.y : (ld.y >= 0.0 ? 1e20 : -1e20),
        abs(ld.z) > DIR_EPS ? 1.0 / ld.z : (ld.z >= 0.0 ? 1e20 : -1e20));
    vec3 t0 = (vol.aabb_min - lp) * invDir;
    vec3 t1 = (vol.aabb_max - lp) * invDir;
    vec3 tFar = max(t0, t1);
    return max(min(min(tFar.x, tFar.y), tFar.z), 0.0);
}

bool volumeRayInterval(VkVolumeInstance vol,
                       vec3 worldOrigin,
                       vec3 worldDir,
                       out float intervalNear,
                       out float intervalFar) {
    vec3 lp;
    lp.x = vol.inv_transform[0] * worldOrigin.x + vol.inv_transform[1] * worldOrigin.y
         + vol.inv_transform[2] * worldOrigin.z + vol.inv_transform[3];
    lp.y = vol.inv_transform[4] * worldOrigin.x + vol.inv_transform[5] * worldOrigin.y
         + vol.inv_transform[6] * worldOrigin.z + vol.inv_transform[7];
    lp.z = vol.inv_transform[8] * worldOrigin.x + vol.inv_transform[9] * worldOrigin.y
         + vol.inv_transform[10] * worldOrigin.z + vol.inv_transform[11];
    vec3 ld;
    ld.x = vol.inv_transform[0] * worldDir.x + vol.inv_transform[1] * worldDir.y
         + vol.inv_transform[2] * worldDir.z;
    ld.y = vol.inv_transform[4] * worldDir.x + vol.inv_transform[5] * worldDir.y
         + vol.inv_transform[6] * worldDir.z;
    ld.z = vol.inv_transform[8] * worldDir.x + vol.inv_transform[9] * worldDir.y
         + vol.inv_transform[10] * worldDir.z;
    const float DIR_EPS = 1e-8;
    vec3 invDir = vec3(
        abs(ld.x) > DIR_EPS ? 1.0 / ld.x : (ld.x >= 0.0 ? 1e20 : -1e20),
        abs(ld.y) > DIR_EPS ? 1.0 / ld.y : (ld.y >= 0.0 ? 1e20 : -1e20),
        abs(ld.z) > DIR_EPS ? 1.0 / ld.z : (ld.z >= 0.0 ? 1e20 : -1e20));
    vec3 t0 = (vol.aabb_min - lp) * invDir;
    vec3 t1 = (vol.aabb_max - lp) * invDir;
    vec3 lo = min(t0, t1);
    vec3 hi = max(t0, t1);
    intervalNear = max(max(lo.x, lo.y), lo.z);
    intervalFar = min(min(hi.x, hi.y), hi.z);
    return intervalFar > max(intervalNear, 0.0);
}

// Find the nearest real liquid iso crossing, not merely its procedural AABB.
// This is the Vulkan equivalent of OptiX's per-ray ordered VDB loop: gas is
// integrated only up to the liquid boundary, then the path is handed to the
// SurfaceSDF closest-hit instead of marching two complete overlapping boxes.
float nearestSurfaceSDFCrossing(vec3 rayOrigin,
                                vec3 rayDir,
                                float rangeNear,
                                float rangeFar,
                                uint currentVolume,
                                uint volumeCount) {
    float nearestHit = rangeFar + 1.0;
    uint count = min(volumeCount, 16u);
    for (uint candidateIndex = 0u; candidateIndex < count; ++candidateIndex) {
        if (candidateIndex == currentVolume) continue;
        VkVolumeInstance surface = volumes.v[candidateIndex];
        if (surface.is_active == 0 || surface.source_type != 4 ||
            surface.volume_type != 2 || surface.vdb_grid_address == 0) {
            continue;
        }

        float boxNear, boxFar;
        if (!volumeRayInterval(surface, rayOrigin, rayDir, boxNear, boxFar)) continue;
        float beginT = max(rangeNear, max(boxNear, 0.001));
        float endT = min(rangeFar, boxFar);
        if (endT <= beginT || beginT >= nearestHit) continue;

        pnanovdb_buf_t surfaceBuf;
        surfaceBuf.address = surface.vdb_grid_address;
        pnanovdb_grid_handle_t gridH;
        gridH.address.byte_offset = 0u;
        pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(surfaceBuf, gridH);
        pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(surfaceBuf, treeH);
        pnanovdb_map_handle_t mapH = pnanovdb_grid_get_map(surfaceBuf, gridH);
        pnanovdb_readaccessor_t acc;
        pnanovdb_readaccessor_init(acc, rootH);

        const float ISO = 0.5;
        float span = endT - beginT;
        int cap = clamp(surface.max_steps, 32, 512);
        float fineStep = clamp(surface.step_size,
                               surface.voxel_size * 0.1,
                               surface.voxel_size * 0.5);
        float step = max(0.001, max(fineStep, span / float(cap)));
        int steps = min(int(ceil(span / step)) + 1, cap + 1);
        float t0s = beginT;
        float d0 = sampleDensityAcc(
            surface, rayOrigin + rayDir * t0s, surfaceBuf, mapH, acc);
        bool startedInside = d0 > ISO;
        for (int s = 0; s < steps; ++s) {
            float t1s = min(t0s + step, endT);
            float d1 = sampleDensityAcc(
                surface, rayOrigin + rayDir * t1s, surfaceBuf, mapH, acc);
            bool crossed = startedInside
                ? (d0 >= ISO && d1 < ISO)
                : (d0 < ISO && d1 >= ISO);
            if (crossed) {
                float a = t0s;
                float b = t1s;
                for (int refine = 0; refine < 4; ++refine) {
                    float mid = 0.5 * (a + b);
                    float dm = sampleDensityAcc(
                        surface, rayOrigin + rayDir * mid, surfaceBuf, mapH, acc);
                    if ((dm > ISO) == startedInside) a = mid;
                    else b = mid;
                }
                nearestHit = min(nearestHit, 0.5 * (a + b));
                break;
            }
            t0s = t1s;
            d0 = d1;
            if (t0s >= endT) break;
        }
    }
    return nearestHit <= rangeFar ? nearestHit : -1.0;
}

void main() {
    // Volume instance index from gl_InstanceCustomIndexEXT
    // (Set via TLASInstance::customIndex when building TLAS for volume objects)
    uint volIdx = gl_InstanceCustomIndexEXT;
    payload.skipGasVolumes = false;
    uint volCount = uint(max(int(cam.pad0), 0));
    if (volIdx >= volCount) {
        payload.scattered = false;
        return;
    }
    VkVolumeInstance vol = volumes.v[volIdx];
    
    // Defensive fallback: inactive AABBs are rejected by the intersection
    // shader and should never reach closest-hit. `scattered=false` is terminal
    // in raygen, so it must not be used as a transparent-skip mechanism.
    if (vol.is_active == 0) {
        vec3 inactiveDir = normalize(gl_WorldRayDirectionEXT);
        payload.radiance = vec3(0.0);
        payload.attenuation = vec3(1.0);
        payload.scatterOrigin =
            gl_WorldRayOriginEXT + inactiveDir * (max(volumeHitAttrib.y, gl_HitTEXT) + 0.002);
        payload.scatterDir = inactiveDir;
        payload.scattered = true;
        payload.skipAABBs = false;
        payload.bounceType = BOUNCE_TRANSPARENT;
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
                // ── Ray-constant shading terms (HOISTED out of the per-step loop) ──
                // view·sun, the HG phase and the thin-film cos() are identical for
                // every foam sample on this ray → evaluate them ONCE, not per step.
                float cosVS      = dot(rayDir, sunDirF);                      // view·sun
                float foamPhase  = henyeyGreenstein(cosVS, 0.35);            // mild forward lobe
                vec3  silver     = mix(vec3(1.0), 0.5 + 0.5 * cos(cosVS * 9.0 + vec3(0.0, 2.0, 4.0)), 0.15);
                // Foam shares the volume's Edge Cutoff (_reserved[0]) so the faint
                // low-density foam fringe is clipped like the water density — otherwise
                // it leaves a grey haze the density cutoff can't reach.
                float foamCutoff = max(vol._reserved[0], 1e-4);
                for (int fs = 0; fs < fsteps && foam_T > 0.01; ++fs) {
                    float ft   = tNear + (float(fs) + jitf) * fstep;
                    vec3  fpos = rayOrigin + rayDir * ft;
                    float fd   = sampleTemperature(vol, fpos) * (1.0 / FOAM_TEMP_SCALE);
                    if (fd <= foamCutoff) continue;
                    float a    = 1.0 - exp(-fd * foamOptical * fstep);
                    // Soft edge falloff over [cutoff, 2·cutoff] → clean fade, not a ring.
                    a *= smoothstep(foamCutoff, foamCutoff * 2.0, fd);

                    // ── Production whitewater shading ──────────────────────────
                    // Foam is a bright, high-albedo single-scatter medium. Two cues
                    // sell the look: a FORWARD-SCATTER silver lining (HG phase toward
                    // the light, so backlit crests glow) and a POWDER term (gentle
                    // edge-dark / core-bright self-occlusion). Still scene-lit (lights
                    // + nishita sun + sky ambient) and NO self-shadow, so it never
                    // goes black in scene-light-lit setups.
                    float powder    = mix(0.75, 1.0, gpu_powder_effect(fd, cosVS)); // gentle edge-dark (cosVS/foamPhase hoisted)
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
                        float lph = henyeyGreenstein(dot(rayDir, ldir), 0.35);  // per-light silver lining
                        Lf += float(cam.lightCount) * lt.color.rgb * lt.color.a * latten
                            * (1.0 + lph * 3.0) * powder;
                    }
                    // Bubble-aggregate upgrade: thin-film silver tint (hoisted above).
                    if (worldData.w.mode == 2) {
                        Lf += sampleTransmittanceLUT(fpos, sunDirF)
                            * worldData.w.sunColor * worldData.w.sunIntensity
                            * (1.0 + foamPhase * 4.0) * powder * silver;        // backlit crest glow
                    } else {
                        Lf += worldData.w.sunColor * worldData.w.sunIntensity
                            * (foamPhase * 2.0) * powder * silver;              // sun glow w/o atmosphere LUT
                    }
                    Lf += skyAmb;                       // ambient fill (never fully dark)
                    // MULTIPLE SCATTERING: foam is a packed mass of air bubbles; its
                    // white comes from light diffusing across many interfaces, so thick
                    // foam reads as a bright FILLED white, not a thin wisp. Near-
                    // isotropic boost that saturates with local density.
                    float ms = 1.0 - exp(-fd * 6.0);
                    Lf += 1.2 * ms * (skyAmb + 0.4 * worldData.w.sunColor * worldData.w.sunIntensity);

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
                                   | gl_RayFlagsNoOpaqueEXT;
            // Exclude gas/fog AABBs (0x02), transient simulation particles
            // (0x04), and SurfaceSDF AABBs (0x08). Particles remain visible
            // to primary rays, but must not
            // trigger the 1+6 solid-location probes for every gas ray.
            const uint SOLID_MASK  = 0xF1;
            const uint VOLUME_SOLID_PROBE = 0xC17D5EEDu;
            shadowPayload = vec4(0.0, 0.0, 0.0, uintBitsToFloat(VOLUME_SOLID_PROBE));
            traceRayEXT(topLevelAS, SOLID_FLAGS, SOLID_MASK, 0, 1, 1,
                        rayOrigin, max(1e-4, tNear - 0.002), rayDir, probeFar + 0.002, 1);
            if (shadowPayload.w < 0.5) {
                float solidT = shadowPayload.x;
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
        payload.primaryARG          = packHalf2x16(vol.scatter_color.rg);
        payload.primaryABT          = packHalf2x16(vec2(vol.scatter_color.b, 1.0));
        payload.primaryNrm          = plPackNormal(N);
        payload.bounceType          = 0u;
        return;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Ordered gas -> liquid handoff for overlapping burning-fluid domains.
    // Ordinary VDB/cloud volumes keep the original fast path.
    float layeredSurfaceT = -1.0;
    // ★Gate on "there is something in front of me", NOT on how the gas happens to be
    // stored. This used to require source_type == 5 (live dense GPU gas), which silently
    // excluded the SAME gas domain whenever it was replayed from a cache: restoring a
    // frame calls setGridDomainStates(), which clears gpu_resident_fields_valid, so the
    // domain publishes as a host NanoVDB volume (source_type 2) instead. With the handoff
    // gated off, the gas marched its whole AABB and terminated the segment, and the
    // SurfaceSDF volume overlapping it was never reached — "the gas renders, the water
    // surface is gone", reproducible on cached playback only.
    // Widening to NanoVDB volumes is safe: nearestSurfaceSDFCrossing skips every
    // candidate that is not itself a SurfaceSDF (source_type 4), so with no liquid in
    // the scene this is a ≤16-slot early-out loop and nothing changes. An ordinary VDB
    // that genuinely overlaps a water surface WANTS this ordering too — that is the
    // OptiX per-ray ordered-volume behaviour this path exists to match.
    if ((vol.source_type == 5 || vol.source_type == 2) && volCount > 1u) {
        layeredSurfaceT = nearestSurfaceSDFCrossing(
            rayOrigin, rayDir, tNear, tFar, volIdx, volCount);
        if (layeredSurfaceT > 0.0) {
            if (layeredSurfaceT <= tNear + 0.003) {
                payload.radiance = vec3(0.0);
                payload.attenuation = vec3(1.0);
                payload.scatterOrigin =
                    rayOrigin + rayDir * max(tNear, layeredSurfaceT - 0.001);
                payload.scatterDir = rayDir;
                payload.scattered = true;
                payload.skipGasVolumes = true;
                payload.bounceType = BOUNCE_TRANSPARENT;
                return;
            }
            tFar = min(tFar, layeredSurfaceT - 0.001);
        }
    }

    // SOLID SURFACE DETECTION inside the volume AABB
    // If a solid triangle exists between tNear and tFar, we must stop the march
    // just before it and signal raygen to fire the next bounce with
    // gl_RayFlagsSkipAABBEXT so the triangle closesthit fires correctly.
    //
    // Strategy: one any-hit distance probe. The dedicated payload sentinel makes
    // shadow_anyhit return gl_HitTEXT directly; the former 1 + 5/6 binary-search
    // traces multiplied catastrophically when a gas domain touched the ground.
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
                                   | gl_RayFlagsNoOpaqueEXT;
            // Real scene solids use bit 0x01. Exclude gas/fog AABBs (0x02),
            // transient simulation particles (0x04), and SurfaceSDF AABBs
            // (0x08); otherwise a particle-rich
            // gas preset turns each volume hit into up to six nested RT probes.
            const uint PROBE_MASK  = 0xF1;

            // Initial check: any solid in [tNear, tFar]?
            const uint VOLUME_SOLID_PROBE = 0xC17D5EEDu;
            shadowPayload = vec4(0.0, 0.0, 0.0, uintBitsToFloat(VOLUME_SOLID_PROBE));
            traceRayEXT(topLevelAS, PROBE_FLAGS, PROBE_MASK, 0, 1, 1,
                        rayOrigin, max(1e-4, tNear - 0.002), rayDir, tFar + 0.002, 1);
            if (shadowPayload.w < 0.5) {
                solidT = shadowPayload.x;
                // Clamp march to just before the solid
                tFar = solidT - 0.01;
                // If the solid is essentially at or before the entry point, skip AABBs on next bounce
                if (tFar <= tNear) {
                    payload.radiance = vec3(0.0);
                    payload.scatterOrigin = rayOrigin + rayDir * max(solidT - 0.01, tNear);
                    payload.scatterDir = rayDir;
                    payload.scattered = true;
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
    const int hardStepCeiling = 2048;
    int authoredStepBudget = max(vol.max_steps, 1);
    int effectiveStepBudget = min(hardStepCeiling, authoredStepBudget);
    int requiredSteps = max(1, int(ceil(marchDist / stepSize)));
    int maxSteps = min(requiredSteps, effectiveStepBudget);
    // Canonical Vulkan baseline: divide the complete interval into a fixed,
    // authored number of segments. Optical integration must never shorten the
    // geometric advance and strand the ray before tFar.
    float baseStep = marchDist / float(max(maxSteps, 1));
    
    float sigma_s_coeff = vol.scatter_coefficient;
    float sigma_a_coeff = vol.absorption_coefficient;
    uint volumeMatIndex = vol._reserved[1] > 0.5 ? uint(vol._reserved[1] - 1.0) : 0u;
    uint volumeProgram = vol._reserved[1] > 0.5 ? matProgramOffset(volumeMatIndex) : MATPROG_NONE;
    // Preserve the legacy fast path: a temperature NanoVDB lookup is eight
    // trilinear tree samples and must not run for an ordinary non-emissive cloud.
    bool needsTemperature = (vol.emission_mode >= 2) || (volumeProgram != MATPROG_NONE);
    
    vec3  accumulated_radiance = vec3(0.0);
    vec3  transmittance = vec3(1.0);
    float opticalDepthWeightedT = 0.0;
    float opticalDepthWeight = 0.0;
    // Deterministic analytical single-scattering is accumulated below. Keep
    // the legacy continuation branch structurally present but disabled: a
    // second stochastic HG estimator would double-count the same event.
    bool didScatter = false;
    float scatterT = tFar;
    vec3 ambientSky = sampleSkyAmbient(rayDir);
    int shadowStride = clamp(vol.shadow_stride, 1, 16);
    // Keep this scalarized. Some NVIDIA Vulkan RT compiler versions crash
    // during pipeline creation on dynamically indexed local bool arrays in a
    // closest-hit shader.
    float cachedSceneShadow0 = 1.0;
    float cachedSceneShadow1 = 1.0;
    bool cachedSceneShadow0Valid = false;
    bool cachedSceneShadow1Valid = false;
    float cachedSunShadow = 1.0;
    bool cachedSunShadowValid = false;
    float cachedAmbientShadow = 1.0;
    bool cachedAmbientShadowValid = false;
    // Must match sampleSkyAmbient(): this is the representative direction of
    // the sky hemisphere whose radiance is used as the ambient source.
    vec3 ambientLightDir = normalize(vec3(0.0, 1.0, 0.0) * 0.55 + rayDir * 0.45);

    // Jitter first sample to reduce banding. Mix in ray state so horizon rays do
    // not share coherent step planes across neighboring pixels.
    float rayJitter = fract(sin(dot(rayOrigin + rayDir * 17.0, vec3(12.9898, 78.233, 37.719)) + float(payload.seed) * 0.000173) * 43758.5453);
    float t = tNear + rayJitter * baseStep;
    int step = 0;
    uint measuredDensitySamples = 0u;
    uint measuredEmptySegments = 0u;
    uint measuredTopologySegments = 0u;
    uint measuredDensityLeafSegments = 0u;
    while (t < tFar && step < maxSteps) {
        vec3  samplePos = rayOrigin + rayDir * t;

        // A Volume Graph may synthesize density in inactive source voxels.
        // Hierarchy skipping is therefore safe only for legacy/baked density.
        if (vdbBuf.address != 0 && volumeProgram == MATPROG_NONE) {
            uint skipKind = 0u;
            float sparseStep = nanoEmptyTileStep(
                vol, samplePos, rayDir, baseStep, vdbBuf, vdbMapH, vdbAcc,
                skipKind);
            if (sparseStep > baseStep * 1.01) {
                int skippedSegments = max(1, int(floor(sparseStep / baseStep)));
                measuredEmptySegments += uint(skippedSegments);
                if (skipKind == 2u)
                    measuredDensityLeafSegments += uint(skippedSegments);
                else
                    measuredTopologySegments += uint(skippedSegments);
                step += skippedSegments;
                t = tNear + (float(step) + rayJitter) * baseStep;
                continue;
            }
        }
        
        float density = sampleDensityAcc(vol, samplePos, vdbBuf, vdbMapH, vdbAcc);
        measuredDensitySamples++;
        float temperature = needsTemperature ? sampleTemperature(vol, samplePos) : 0.0;
        vec3 stepScatterColor = vol.scatter_color;
        vec3 stepAbsorptionColor = vol.absorption_color;
        vec3 stepEmissionColor = vol.emission_color;
        float stepEmissionStrength = vol.emission_intensity;
        float stepAnisotropy = vol.scatter_anisotropy;
        float stepMultiScatter = vol.scatter_multi;
        if (volumeProgram != MATPROG_NONE) {
            float emptyAttribs[MP_ATTRIB_SLOTS];
            for (int ai = 0; ai < MP_ATTRIB_SLOTS; ++ai) emptyAttribs[ai] = 0.0;
            vec3 localPos;
            localPos.x = vol.inv_transform[0]*samplePos.x + vol.inv_transform[1]*samplePos.y + vol.inv_transform[2]*samplePos.z + vol.inv_transform[3];
            localPos.y = vol.inv_transform[4]*samplePos.x + vol.inv_transform[5]*samplePos.y + vol.inv_transform[6]*samplePos.z + vol.inv_transform[7];
            localPos.z = vol.inv_transform[8]*samplePos.x + vol.inv_transform[9]*samplePos.y + vol.inv_transform[10]*samplePos.z + vol.inv_transform[11];
            vec3 normalizedLocalPos = clamp(
                (localPos - vol.aabb_min) / max(vol.aabb_max - vol.aabb_min, vec3(1e-6)),
                vec3(0.0), vec3(1.0));
            MatProgOut vp = evalMaterialProgram(
                volumeProgram, vec2(0.0), samplePos, -rayDir, 0.5, vec3(0.0),
                emptyAttribs, localPos, -rayDir,
                density, temperature,
                (temperature > 0.0) ? clamp(temperature / max(vol.max_temperature, 1.0), 0.0, 1.0) : 0.0,
                0.0, vec3(0.0), samplePos,
                vol.scatter_color, vol.emission_color, max(vol.voxel_size, 1e-6), normalizedLocalPos,
                cam.waterTime);
            if ((vp.volumeWritten & (1u << 0)) != 0u) density = max(vp.volumeDensity, 0.0);
            if ((vp.volumeWritten & (1u << 1)) != 0u) stepScatterColor = max(vp.volumeScatterColor, vec3(0.0));
            if ((vp.volumeWritten & (1u << 2)) != 0u) sigma_s_coeff = max(vp.volumeScatterStrength, 0.0);
            if ((vp.volumeWritten & (1u << 3)) != 0u) stepAbsorptionColor = max(vp.volumeAbsorptionColor, vec3(0.0));
            if ((vp.volumeWritten & (1u << 4)) != 0u) sigma_a_coeff = max(vp.volumeAbsorptionStrength, 0.0);
            if ((vp.volumeWritten & (1u << 5)) != 0u) stepEmissionColor = max(vp.volumeEmissionColor, vec3(0.0));
            if ((vp.volumeWritten & (1u << 6)) != 0u) stepEmissionStrength = max(vp.volumeEmissionStrength, 0.0);
            if ((vp.volumeWritten & (1u << 7)) != 0u) stepAnisotropy = clamp(vp.volumeAnisotropy, -0.99, 0.99);
            if ((vp.volumeWritten & (1u << 8)) != 0u) stepMultiScatter = clamp(vp.volumeMultiScatter, 0.0, 1.0);
        }
        float sigma_s_local = density * sigma_s_coeff;
        // RGB absorption. Black historically meant "neutral absorption" in
        // existing assets, so preserve it as an achromatic fallback. Otherwise
        // Absorption Color identifies the wavelengths removed by the medium.
        vec3 authoredAbsorption = max(stepAbsorptionColor, vec3(0.0));
        float absorptionPeak = max(
            authoredAbsorption.r,
            max(authoredAbsorption.g, authoredAbsorption.b));
        vec3 absorptionWeights = absorptionPeak > 1e-5
            ? authoredAbsorption / absorptionPeak
            : vec3(1.0);
        vec3 sigma_a_local = density * sigma_a_coeff * absorptionWeights;
        vec3 sigma_t_rgb = vec3(sigma_s_local) + sigma_a_local;
        float sigma_t_local = dot(
            sigma_t_rgb, vec3(0.2126, 0.7152, 0.0722));
        // Let low-density regions survive based on scattering optical depth, not
        // raw density alone. This means increasing scatter can actually keep thin
        // edges visible instead of them being discarded by a fixed density gate.
        float sparseCutoff = (vol._reserved[0] > 0.0) ? vol._reserved[0] : 0.04;
        float scatter_keep = clamp((sigma_s_local * baseStep) / sparseCutoff, 0.0, 1.0);
        if (sigma_t_local <= EPSILON) {
            step++;
            t = tNear + (float(step) + rayJitter) * baseStep;
            continue;
        }
        
        // Fixed coverage segment. Beer-Lambert remains analytic over the full
        // segment; adaptive optical substeps will return only with a separate
        // substep budget.
        float dt = min(baseStep, tFar - t);
        if (dt <= 1e-6) break;

        // Current extinction
        vec3 extinction = sigma_t_rgb * dt;
        vec3 sampleTransmittance = exp(-extinction);
        
        // ── Multi-scatter transmittance blend (matches OptiX) ──
        // Blends single-scatter (Beer's law) with a softer 0.25x extinction approximation
        // to model multiple scattering. When scatter_multi > 0, volumes appear brighter
        // and more translucent — matching the OptiX renderer output.
        // Extinction remains Beer-Lambert; multiple scattering redistributes
        // direction instead of making the same medium transmit extra energy.
        vec3 one_minus_sampleT = vec3(1.0) - sampleTransmittance;
        // Representative volume distance for aerial perspective. Weight each
        // sample by the camera-path opacity it actually contributes, not by the
        // procedural AABB entry/exit. Sparse fringe therefore cannot move the
        // apparent atmospheric endpoint to the domain wall.
        float stepOpacityWeight = dot(
            transmittance * one_minus_sampleT,
            vec3(0.2126, 0.7152, 0.0722));
        opticalDepthWeightedT +=
            (t + 0.5 * dt) * max(stepOpacityWeight, 0.0);
        opticalDepthWeight += max(stepOpacityWeight, 0.0);
        
        // ── Volume Emission ──
        // Mode 0 = none, 1 = plain color, 2 = blackbody/color-ramp via temperature grid.
        // Energy-stable integration: multiply by one_minus_sampleT (bounded by 1) instead of dt.
        vec3 emis = vec3(0.0);
        if (vol.emission_mode >= 1) {
            if (vol.emission_mode == 1) {
                // Plain constant-color emission
                emis = stepEmissionColor * stepEmissionStrength * density;
            } else if (vol.emission_mode >= 2) {
                // Blackbody / color-ramp — temperature grid first, density fallback for parity.
                if (temperature <= 0.0) temperature = density;

                // _ext_reserved[6] is foam opacity for Surface SDF (type 4);
                // all other volumes receive authored minimum temperature here.
                float rangeMin = (vol.source_type == 4)
                    ? 0.0 : max(vol._ext_reserved[6], 0.0);
                float rangeMax = (vol.max_temperature > rangeMin + 1.0)
                    ? vol.max_temperature : (rangeMin + 1500.0);
                // Simulation grids arrive as Kelvin-scaled heat. Normalized
                // fallbacks map into the same authored temperature interval.
                float authoredKelvin = (temperature > 20.0)
                    ? clamp(temperature, rangeMin, rangeMax)
                    : mix(rangeMin, rangeMax, clamp(temperature, 0.0, 1.0));
                float t_ramp = clamp(
                    (authoredKelvin - rangeMin) / max(rangeMax - rangeMin, 1.0),
                    0.0, 1.0);

                vec3 e_color;
                if (vol.color_ramp_enabled != 0 && vol.ramp_stop_count > 0) {
                    e_color = sampleColorRamp(vol, clamp(t_ramp * vol.temperature_scale, 0.0, 1.0));
                } else {
                    // The authored interval constrains blackbody color; Temp
                    // Scale then provides the intentional artistic offset.
                    e_color = blackbodyToRGB(authoredKelvin * vol.temperature_scale);
                }
                emis = clampVolumeRadiance(
                    e_color * density * vol.blackbody_intensity, 64.0);
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
                    int li = min(ls, int(cam.lightCount) - 1);
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
                    float phase = dualLobeHG(cosTheta, stepAnisotropy,
                                             vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                    vec3 scatteringAlbedoRGB =
                        vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
                    float scatteringAlbedo = dot(
                        scatteringAlbedoRGB, vec3(0.2126, 0.7152, 0.0722));
                    phase = mix(
                        phase, 1.0 / (4.0 * PI),
                        clamp(stepMultiScatter * scatteringAlbedo, 0.0, 1.0));
                    
                    // Light march transmittance through volume toward light.
                    // Geometry occlusion is NOT checked per-step (matching OptiX raymarch_volumetric_object
                    // behavior): solid objects are handled by TLAS traversal order, not per-sample rays.
                    // This prevents hard black shadows from solids inside the volume.
                    float shadowTr = 1.0;
                    if (sigma_t_local * dt > 0.02) {
                        bool shadowValid = (ls == 0)
                            ? cachedSceneShadow0Valid : cachedSceneShadow1Valid;
                        bool updateShadow = !shadowValid || (step % shadowStride) == 0;
                        if (updateShadow) {
                            float shadowMaxDist = min(
                                lightDist,
                                max(baseStep, volumeExitDistance(vol, samplePos, lightDir)));
                            float evaluatedShadow = lightMarchAcc(
                                vol, samplePos, lightDir, shadowMaxDist,
                                vdbBuf, vdbMapH, vdbAcc, -1.0);
                            if (ls == 0) {
                                cachedSceneShadow0 = evaluatedShadow;
                                cachedSceneShadow0Valid = true;
                            } else {
                                cachedSceneShadow1 = evaluatedShadow;
                                cachedSceneShadow1Valid = true;
                            }
                        }
                        shadowTr = (ls == 0) ? cachedSceneShadow0 : cachedSceneShadow1;
                    }
                    
                    vec3 lightColor = light.color.rgb * light.color.a;
                    inscatter += lightWeight * lightColor * lightAtten * phase * shadowTr
                               * stepScatterColor * scatteringAlbedoRGB;
                }
            }
            
            // Sun/sky light contribution (if Nishita sky active)
            if (worldData.w.mode == 2) {
                vec3 sunDir = normalize(worldData.w.sunDir);
                float cosSun = dot(rayDir, sunDir);
                float sunPhase = dualLobeHG(cosSun, stepAnisotropy,
                                            vol.scatter_anisotropy_back, vol.scatter_lobe_mix);
                vec3 scatteringAlbedoRGB =
                    vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
                float scatteringAlbedo = dot(
                    scatteringAlbedoRGB, vec3(0.2126, 0.7152, 0.0722));
                sunPhase = mix(
                    sunPhase, 1.0 / (4.0 * PI),
                    clamp(stepMultiScatter * scatteringAlbedo, 0.0, 1.0));
                
                float sunShadowTr = 1.0;
                if (sigma_t_local * dt > 0.02) {
                    bool updateSunShadow = !cachedSunShadowValid || (step % shadowStride) == 0;
                    if (updateSunShadow) {
                        float sunShadowMaxDist = max(
                            baseStep, volumeExitDistance(vol, samplePos, sunDir));
                        cachedSunShadow = lightMarchAcc(
                            vol, samplePos, sunDir, sunShadowMaxDist,
                            vdbBuf, vdbMapH, vdbAcc, -1.0);
                        cachedSunShadowValid = true;
                    }
                    sunShadowTr = cachedSunShadow;
                }
                
                vec3 sunLi = sampleTransmittanceLUT(samplePos, sunDir) * worldData.w.sunColor * worldData.w.sunIntensity;
                inscatter += sunLi * sunPhase * sunShadowTr
                           * stepScatterColor * scatteringAlbedoRGB;
            }
            
            // 3. Sky/Ambient lighting (closer to CPU world.evaluate(up) behavior)
            float thin_scatter = scatter_keep * scatter_keep;
            vec3 ambientAlbedo =
                vec3(sigma_s_local) / max(sigma_t_rgb, vec3(EPSILON));
            float ambientAlbedoLuma = dot(
                ambientAlbedo, vec3(0.2126, 0.7152, 0.0722));
            // Ambient is incident radiance too: letting it reach every dense
            // sample unattenuated fills the entire cloud with a flat white
            // source and erases density/depth cues. Use a cached representative
            // sky-hemisphere march so cores darken while sky-facing lobes remain
            // lit. A minimum stride of four bounds the Vulkan RT cost.
            // Hemispherical ambient changes slowly along a primary ray. Eight
            // samples of reuse removes a major nested NanoVDB-march cost without
            // altering its Beer-Lambert visibility.
            int ambientShadowStride = max(shadowStride, 8);
            bool updateAmbientShadow =
                !cachedAmbientShadowValid || (step % ambientShadowStride) == 0;
            if (updateAmbientShadow && sigma_t_local * dt > 0.01) {
                float ambientShadowMaxDist = max(
                    baseStep,
                    volumeExitDistance(vol, samplePos, ambientLightDir));
                cachedAmbientShadow = lightMarchAcc(
                    vol, samplePos, ambientLightDir, ambientShadowMaxDist,
                    vdbBuf, vdbMapH, vdbAcc, 1.0);
                cachedAmbientShadowValid = true;
            }
            inscatter += ambientSky * stepScatterColor * ambientAlbedo
                       * thin_scatter * cachedAmbientShadow;

            // Bounded higher-order sky scattering approximation. Recycle only
            // a fraction of the ambient energy blocked on the representative
            // sky path; unlike emission this vanishes when external sky light
            // vanishes, remains density/depth dependent, and never exceeds the
            // original unoccluded ambient budget:
            //   visibility = T_sky + multi * albedo * P(2+) * (1-T_sky) <= 1.
            float trappedSky = 1.0 - cachedAmbientShadow;
            float localOpticalDepth = sigma_t_local * dt;
            float higherOrderProbability = 1.0 - exp(-localOpticalDepth);
            float multiSkyVisibility =
                stepMultiScatter * ambientAlbedoLuma * trappedSky
                * higherOrderProbability;
            inscatter += ambientSky * stepScatterColor * ambientAlbedo
                       * thin_scatter * multiSkyVisibility;
            
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
        // No stochastic volume continuation: the direct single-scattering
        // estimator above already accounts for redirected radiance.
        
        // Early termination if transmittance is negligible
        if (max(transmittance.r, max(transmittance.g, transmittance.b)) < 0.001) {
            break;
        }
        step++;
        t = tNear + (float(step) + rayJitter) * baseStep;
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
    if (any(isnan(transmittance)) || any(isinf(transmittance)) ||
        any(lessThan(transmittance, vec3(0.0))))
        transmittance = vec3(1.0);
    float transmittanceLuma = dot(
        transmittance, vec3(0.2126, 0.7152, 0.0722));

    // Accumulated in-scattered radiance
    payload.radiance  = accumulated_radiance;
    payload.skipAABBs = false; // default; overridden below if solid found
    // ── Solid surface found inside the volume: hand off to closesthit ──
    // March was already clamped to solidT. Now position the scatter ray
    // just before the solid surface and tell raygen to skip AABBs next
    // bounce (gl_RayFlagsSkipAABBEXT) so the triangle closesthit fires.
    if (solidT >= 0.0) {
        volumeRecordRay(
            measuredDensitySamples,
            measuredEmptySegments,
            measuredTopologySegments,
            measuredDensityLeafSegments,
            VOLUME_MARCH_COMPLETED);
        payload.attenuation  *= transmittance;
        payload.scatterOrigin = rayOrigin + rayDir * (solidT - 0.01);
        payload.scatterDir    = rayDir;
        payload.scattered     = true;
        payload.skipAABBs     = true;
        return;
    }

    float volumeContribution = max(max(accumulated_radiance.r, accumulated_radiance.g), accumulated_radiance.b);
    float volumeOpacity = 1.0 - transmittanceLuma;
    // Keep ultra-thin fog tails out of the primary auxiliary buffers. They were
    // being treated as first-hit geometry, which overemphasized weak density
    // regions in Vulkan RT denoiser output compared to OptiX.
    bool primaryVolumeInteraction =
        volumeOpacity > 0.04 || volumeContribution > 5e-4;
    if ((payload.primaryMeta & PL_PRIMARY_DONE) == 0u && primaryVolumeInteraction) {
        float representativeVolumeT = opticalDepthWeight > 1e-8
            ? opticalDepthWeightedT / opticalDepthWeight
            : 0.5 * (tNear + tFar);
        payload.primaryARG  = packHalf2x16(vol.scatter_color.rg);
        payload.primaryABT  = packHalf2x16(vec2(vol.scatter_color.b, transmittanceLuma));
        // PL_PRIMARY_VOLUME reinterprets primaryNrm as float bits containing
        // the optical-depth centroid distance. Raygen reconstructs the volume
        // AOV normal as -primaryRayDir without increasing payload size.
        payload.primaryNrm  = floatBitsToUint(representativeVolumeT);
        // Material id stays 0xFFFF: volumes have no scene material index.
        payload.primaryMeta = (payload.primaryMeta & PL_DISP_MASK)
                            | PL_PRIMARY_DONE | PL_PRIMARY_VOLUME
                            | PL_PRIMARY_VOLUME_DEPTH | PL_MATID_MASK;
    }

    if (didScatter && transmittanceLuma < 0.99) {
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
    } else {
        // No scatter — attenuate throughput by volume transmittance
        payload.attenuation *= transmittance;

        // Set scattered = true with original direction to let the ray continue through
        // Ensure forward progress to avoid re-hitting the same boundary when camera is inside.
        payload.scatterOrigin = layeredSurfaceT > 0.0
            ? rayOrigin + rayDir * max(tNear, layeredSurfaceT - 0.0005)
            : rayOrigin + rayDir * (tFar + 0.002);
        payload.scatterDir    = rayDir;
        payload.scattered     = (transmittanceLuma > 0.01); // Stop if fully absorbed
        payload.skipGasVolumes =
            layeredSurfaceT > 0.0 && payload.scattered;
        if (volumeOpacity <= 0.04 && volumeContribution <= 5e-4) {
            payload.bounceType = BOUNCE_TRANSPARENT;
        }
    }
    uint marchOutcome = transmittanceLuma <= 0.01
        ? VOLUME_MARCH_EXTINCTION
        : ((step >= maxSteps && t < tFar)
            ? VOLUME_MARCH_STEP_BUDGET
            : VOLUME_MARCH_COMPLETED);
    volumeRecordRay(
        measuredDensitySamples,
        measuredEmptySegments,
        measuredTopologySegments,
        measuredDensityLeafSegments,
        marchOutcome);
}
