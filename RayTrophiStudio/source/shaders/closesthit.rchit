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
    uint  maxBounces;
    uint  diffuseBounces;
    uint  transmissionBounces;

    // Debug Visualizer (must stay offset-identical to raygen's CameraPC)
    uint  debugView;      // 9 = Medium Density: closesthit terminates at the
                          // first hit and returns the dust-coverage integral
    float debugExposure;
    uint  debugFlags;
    float debugParam;
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
const uint MAT_FLAG_WATER_LAKE = (1u << 22);
const uint MAT_FLAG_WATER_RIVER = (1u << 23);
const uint MAT_FLAG_BUBBLE = (1u << 19);
const uint MAT_FLAG_MARBLE_VOLUME = (1u << 20); // glass marble full-volume medium march (raygen integrates interior)

// ============================================================
// Payload — shared ABI, single source of truth
// ============================================================
#include "rt_payload.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
// Separate shadow payload storage to avoid corrupting the main payload during shadow tracing.
// rgb = transmissive tint accumulated by shadow any-hits (coloured glass shadows),
// w   = reached-light flag. Init to (1,1,1,0) before each shadow trace; shadow_miss sets w=1.
layout(location = 1) rayPayloadEXT vec4 shadowPayload;

// ============================================================
// Descriptor Bindings
// ============================================================
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

// Material struct — single source of truth shared by every material-reading shader.
#include "material_struct.glsl"
#include "water_v3.glsl"

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
    uint64_t pointinessAddr;  // per-vertex pointiness (Geometry node); 0 = not uploaded
    uint64_t attribAddr;      // per-vertex named attributes (Attribute node), INTERLEAVED
                              // MP_ATTRIB_SLOTS floats per vertex; 0 = not uploaded
    uint64_t waterAddr;       // per-vertex hydrology: three vec4 records; 0 = absent
};

struct VkInstanceData {
    uint materialIndex;
    uint blasIndex;
};

layout(set = 0, binding = 2, scalar) readonly buffer MaterialBuffer  { Material     m[]; } materials;
// COLD material fields (split record, see material_struct.glsl). Accessed via
// the `matx` macro below so every read is an independent SSBO load AT ITS USE
// SITE — the loads sink into the feature-gated branches (SSS/water/bubble/
// resin/dust) instead of joining a monolithic per-hit struct fetch.
layout(set = 0, binding = 24, scalar) readonly buffer MaterialExtBuffer { MaterialExt m[]; } materialsExt;
#define matx materialsExt.m[matIndex]
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

// Ambient Occlusion node: the material VM does not trace rays itself — it calls out to
// the stage it runs in. Closest-hit is the ONE stage that may trace (the NEE shadow ray
// below already does, so the recursion depth this needs is depth the pipeline is already
// paying for), so it defines MP_HAS_AO and supplies the tracer. Prototype here, body
// after the RNG (rnd) it needs; ray-gen and the shadow any-hit do not include this VM at
// all, so nothing else can accidentally trace from a stage where tracing is illegal.
#define MP_HAS_AO 1
float mp_traceAO(vec3 p, vec3 n, float dist, int samples, bool inside);

// Bevel node: same contract. The probe rays reuse the shadow payload with a MARKER in .w
// (see mp_traceBevel below and the probe branch at the top of shadow_anyhit.rahit).
#define MP_HAS_BEVEL 1
vec3 mp_traceBevel(vec3 p, vec3 n, float radius, int samples);

#include "material_program.glsl"   // Faz 2b: per-pixel material-graph VM (binding 23)

// Trilinear NanoVDB float grid sampler with a CALLER-OWNED accessor — the
// pnanovdb_readaccessor caches the root→leaf tree path, so a march loop must
// init it once per volume and reuse it across steps (per-sample re-init walks
// the whole tree for all 8 taps again — the exact per-sample-reinit trap the
// volume closesthit already avoids).
float ch_sampleNanoVDBAcc(pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
                          inout pnanovdb_readaccessor_t acc, vec3 worldPos) {
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
// type 2 (NanoVDB): real trilinear grid sample via the caller's accessor.
// vdbReady + buf/map/acc come from the caller, initialized ONCE per volume —
// see computeVolumeShadowTransmittance.
float ch_volDensity(VkVolumeInstance vol, vec3 wp,
                    pnanovdb_buf_t vdbBuf, pnanovdb_map_handle_t vdbMapH,
                    inout pnanovdb_readaccessor_t vdbAcc, bool vdbReady) {
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
        // vdbReady is false when vdb_grid_address == 0 (source VDB missing /
        // buffer not yet uploaded) — dereferencing address 0 = GPU crash.
        if (vdbReady) {
            vec3 vdbWorldPos = lp;
            vdbWorldPos.x -= vol.pivot_offset[0];
            vdbWorldPos.y -= vol.pivot_offset[1];
            vdbWorldPos.z -= vol.pivot_offset[2];
            density = ch_sampleNanoVDBAcc(vdbBuf, vdbMapH, vdbAcc, vdbWorldPos);
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

        // NanoVDB persistent accessor — initialized ONCE per volume and reused
        // by every density sample of this march (tauHint + all steps). The
        // accessor caches the root→leaf path; the old per-sample init re-walked
        // the whole tree for each of the 8 trilinear taps, every step.
        pnanovdb_buf_t          vdbBuf;
        pnanovdb_map_handle_t   vdbMapH;
        pnanovdb_readaccessor_t vdbAcc;
        bool vdbReady = (vol.volume_type == 2) && (vol.vdb_grid_address != 0);
        if (vdbReady) {
            vdbBuf.address = vol.vdb_grid_address;
            pnanovdb_grid_handle_t gridH; gridH.address.byte_offset = 0u;
            pnanovdb_tree_handle_t treeH = pnanovdb_grid_get_tree(vdbBuf, gridH);
            pnanovdb_root_handle_t rootH = pnanovdb_tree_get_root(vdbBuf, treeH);
            vdbMapH = pnanovdb_grid_get_map(vdbBuf, gridH);
            pnanovdb_readaccessor_init(vdbAcc, rootH);
        }

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
        float dMid = ch_volDensity(vol, shadowOrigin + lightDir * (tNW + 0.5 * segLen),
                                   vdbBuf, vdbMapH, vdbAcc, vdbReady);
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
            float d = ch_volDensity(vol, sp, vdbBuf, vdbMapH, vdbAcc, vdbReady);
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
layout(buffer_reference, scalar) readonly buffer PointinessBuffer    { float p[]; };
layout(buffer_reference, scalar) readonly buffer AttribBuffer        { float a[]; };
layout(buffer_reference, scalar) readonly buffer WaterVertexBuffer   { vec4 w[]; };

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

// ============================================================
// Ambient Occlusion tracer (material VM's AO op — see MP_HAS_AO above)
// ============================================================
// Cosine-weighted hemisphere around the shading normal, `samples` shadow rays, each
// capped at `dist`. Reuses the EXACT machinery the NEE shadow ray uses: same payload
// (location 1), same miss shader (index 1, sets w = 1 on escape), same triangles-only
// mask 0x01 — so an alpha-cut leaf occludes AO exactly as it occludes a light, and a
// volume AABB (mask 0x02) does not occlude at all.
//
// COST: this multiplies the ray count of every shading call that runs an AO chain by
// `samples`. It is the only node in the graph that does that. Nothing here is cached.
float mp_traceAO(vec3 p, vec3 n, float dist, int samples, bool inside) {
    vec3 nrm = safeNormalize(n, vec3(0.0, 1.0, 0.0));
    if (inside) nrm = -nrm;                      // occlusion of the cavity BEHIND the surface
    samples = clamp(samples, 1, 64);

    vec3 t, b;
    buildONB(nrm, t, b);

    // Seed from the SHADING POINT, not the pixel: a camera-seeded AO crawls over a static
    // surface as the camera moves. payload.seed adds the per-sample decorrelation that
    // lets the estimate average out across accumulation.
    uint seed = payload.seed
              ^ (floatBitsToUint(p.x) * 73856093u)
              ^ (floatBitsToUint(p.y) * 19349663u)
              ^ (floatBitsToUint(p.z) * 83492791u);

    vec3 origin = p + nrm * SHADOW_TMIN;
    int hits = 0;
    for (int s = 0; s < samples; ++s) {
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        float phi = 6.2831853 * r1;
        float sq = sqrt(r2);                     // cosine-weighted
        vec3 dir = t * (cos(phi) * sq) + b * (sin(phi) * sq) + nrm * sqrt(max(0.0, 1.0 - r2));

        shadowPayload = vec4(1.0, 1.0, 1.0, 0.0);   // conservative: blocked until the miss says otherwise
        uint aoFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
        traceRayEXT(topLevelAS, aoFlags, 0x01, 0, 1, 1, origin, SHADOW_TMIN, dir, max(dist, SHADOW_TMIN * 2.0), 1);
        if (shadowPayload.w <= 0.5) ++hits;
    }
    return 1.0 - float(hits) / float(samples);
}

// ============================================================
// Bevel tracer (material VM's Bevel op — see MP_HAS_BEVEL above)
// ============================================================
// Rounded-edge shading normal = the AREA-AVERAGE of the surface normal over the part
// of the scene inside a sphere of `radius` around the shading point, distance-weighted.
// Estimated with `samples` random CHORDS through that sphere: a uniform direction, a
// uniform disk offset perpendicular to it, and a segment spanning the sphere; the
// any-hit adds EVERY intersection's stored normal, weighted by (1 - dist/R), into the
// payload. A hard edge then SHADES like a fillet while the silhouette stays as modeled.
//
// Why chords and not rays cast from the shading point (three shipped attempts document
// the difference): from-P sampling sees the face it stands on in half of all directions
// but an edge's neighbor face in only a quarter, so the blend over-rotates past the mid
// normal on BOTH sides of the edge and the normal JUMPS at the exact crest line — a
// sharp seam right where the rounding should be smoothest. The area estimator is
// continuous in P and lands on the mid normal at the crest by symmetry.
//
// The probes cannot invoke this closest-hit recursively (payload clobber + recursion
// budget), and rayQuery's feature bit is not enabled on the device — so they reuse the
// SHADOW pipeline in a probe mode flagged by the payload w's SIGN BIT (the shadow path
// only ever writes +0.0 / +1.0 there, so bit 31 is unambiguous):
//   w   = packHalf2x16(vec2(h, diskR)) | 0x80000000  (h = chord half-length, diskR =
//         the chord's offset from P; sphere radius = sqrt(h^2 + diskR^2)),
//   xyz = running sum of weighted stored normals (the miss shader only touches .w,
//         so the sum survives traversal officially "missing").
// MIRROR: MatOp::Bevel in MaterialProgram.h — same estimator, only the RNG differs.
vec3 mp_traceBevel(vec3 p, vec3 nIn, float radius, int samples) {
    vec3 n = safeNormalize(nIn, vec3(0.0, 1.0, 0.0));
    radius = max(radius, 1e-5);
    samples = clamp(samples, 1, 16);

    // Shading-point seed (camera-stable, like AO) with a different salt so a graph
    // using both ops does not correlate their sample patterns.
    uint seed = payload.seed
              ^ (floatBitsToUint(p.x) * 0xB5297A4Du)
              ^ (floatBitsToUint(p.y) * 0x68E31DA4u)
              ^ (floatBitsToUint(p.z) * 0x1B56C4E9u);

    // Tiny N seed only breaks the tie when every chord misses (a needle tip);
    // anywhere normal it is noise-floor against the accumulated real weights.
    vec3 accum = n * 0.05;
    for (int s = 0; s < samples; ++s) {
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        float z = 1.0 - 2.0 * r1;                // chord axis: uniform sphere
        float phi = 6.2831853 * r2;
        float sxy = sqrt(max(0.0, 1.0 - z * z));
        vec3 D = vec3(cos(phi) * sxy, sin(phi) * sxy, z);

        vec3 e1, e2;
        buildONB(D, e1, e2);
        float r3 = rnd(seed);
        float r4 = rnd(seed);
        float diskR = radius * sqrt(r3) * 0.999; // uniform disk; keep h > 0
        float ph2 = 6.2831853 * r4;
        float h = sqrt(max(radius * radius - diskR * diskR, 0.0));
        vec3 origin = p + e1 * (cos(ph2) * diskR) + e2 * (sin(ph2) * diskR) - D * h;

        // NoOpaque forces the any-hit to run on opaque-flagged BLASes too; no
        // TerminateOnFirstHit — the any-hit must see EVERY crossing to accumulate.
        shadowPayload = vec4(0.0, 0.0, 0.0,
            uintBitsToFloat(packHalf2x16(vec2(h, diskR)) | 0x80000000u));
        uint probeFlags = gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsNoOpaqueEXT;
        traceRayEXT(topLevelAS, probeFlags, 0x01, 0, 1, 1, origin, 0.0, D, 2.0 * h, 1);
        accum += shadowPayload.xyz;
    }
    return safeNormalize(accum, n);
}

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
// Selection weight of one light — kept in a helper so the two-pass walk below
// (sum, then CDF re-walk) computes identical values WITHOUT materializing a
// per-light array. The old float weights[128] was a dynamically indexed local
// array: 512 bytes of scratch memory per invocation on every closesthit,
// a pure occupancy tax. Recomputing the weight is a handful of ALU ops against
// an SSBO read that is warm in cache on the second pass.
float smart_light_weight_gl(int i, vec3 hit_pos) {
    int t = int(lights.l[i].position.w + 0.5);
    if (t == 1) return 0.0; // directional — handled by the uniform branch above
    vec3 delta = lights.l[i].position.xyz - hit_pos;
    float dist = max(length(delta), 1.0);
    float intensity = gc_luminance(lights.l[i].color.rgb) * lights.l[i].color.a;
    if (t == 0) {
        // Point light: account for spherical sampling area (4*pi*r^2) so selection pdf
        // and per-light sampling pdf are consistent (avoids intensity scaling with radius).
        float area = 4.0 * PI * lights.l[i].params.x * lights.l[i].params.x;
        return (1.0 / (dist * dist)) * intensity * area;
    } else if (t == 2) {
        return (1.0 / (dist * dist)) * intensity * min(lights.l[i].params.y * lights.l[i].params.z, 10.0);
    } else if (t == 3) {
        return (1.0 / (dist * dist)) * intensity * 0.8;
    }
    return 0.0;
}

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
    // Weighted selection — two passes over the light list (sum, then CDF walk),
    // no per-light array. Also lifts the old 128-light cap: every light gets a
    // selection weight now, not just the first 128.
    float total = 0.0;
    for (int i = 0; i < light_count; ++i) total += smart_light_weight_gl(i, hit_pos);
    if (total < 1e-6) {
        int sel = int(rng * float(light_count)) % light_count;
        pdf_out = prob_to_reach * (1.0 / float(light_count));
        return sel;
    }
    float r = rng * total;
    float acc = 0.0;
    int sel = light_count - 1;
    float selW = 0.0;
    for (int i = 0; i < light_count; ++i) {
        float w = smart_light_weight_gl(i, hit_pos);
        acc += w;
        selW = w; // if the walk never breaks (numeric edge), fall back to the last light
        if (r <= acc) { sel = i; break; }
    }
    pdf_out = prob_to_reach * (selW / total);
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

// GGX NDF half-vector sampling — only scatterGlass uses this helper.
// Returning the reflected direction here caused scatterGlass to treat that
// direction as a normal and reflect/refract a second time. Roughness == 0
// bypassed the bug, while any small positive value flattened water detail.
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

    return halfVec;
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
// Worley F1 + the nearest seed point and its cell id. The seed point lets a
// speck build a pseudo-normal (P - seed) so it shades as a tiny lit sphere
// instead of a flat colour stamp; the cell id hashes per-speck size/colour/type.
float rh_worley_pt(vec3 p, out vec3 seedPt, out vec3 cellId) {
    vec3 ip = floor(p);
    vec3 fp = fract(p);
    float d = 1e9;
    seedPt = ip; cellId = ip;
    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x) {
        vec3 g = vec3(x, y, z);
        vec3 o = rh_hash33(ip + g);
        float dd = length(g + o - fp);
        if (dd < d) { d = dd; seedPt = ip + g + o; cellId = ip + g; }
    }
    return d;
}

// ── Resin interior march, quality pass ──────────────────────────────────────
// Shared by the resin-coat base and the glass-marble shell. Same recipe as the
// volumetric caustic march: JITTERED stochastic stepping + progressive
// accumulation replace brute-force step count, and SOFT densities replace the
// old hard thresholds (fixed 6 steps + binary speck test = the coarse look).
//   dust   — two-scale billowy fbm: extinction PLUS a milky in-scatter
//            coverage, so wisps are VISIBLE, not just darkening
//   specks — worley cells with per-speck hashed radius/colour/type: ~30% are
//            BUBBLES (bright shell rim, no occlusion), the rest dirt that
//            terminates stochastically with a soft edge (antialiases across
//            accumulation) and shades as a top-lit micro-sphere via its
//            pseudo-normal
// Self-contained: no scene rays; cost only on resin/marble materials.
struct ResinMarch {
    vec3  absorb;      // transmittance through the crossed interior
    float dustCover;   // milky wisp coverage, 0..1 (caller mixes toward dustTint)
    vec3  dustTint;    // coverage-weighted NEBULA colour of the dust
    bool  dirtHit;     // ray stopped on a dirt speck
    vec3  dirtAlbedo;  // light-direction-shaded speck colour (valid when dirtHit)
    float sparkle;     // bubble/shard rim highlight, transmittance-weighted
    vec3  shardGlow;   // shards' own visible colour body (additive, T-weighted)
    vec3  dustGlow;    // forward-scatter excess past the coverage clamp —
                       // the backlit silver lining (additive, like shardGlow)
};
// hue → vivid rgb (saturation baked at 0.85) for the glass-shard palette.
vec3 rh_hue(float h) {
    vec3 rgb = clamp(abs(fract(vec3(h) + vec3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(1.0), rgb, 0.85);
}
// Dust density field only, no colour — MUST mirror the density half of the
// style branches in resinMarchInterior's Phase A (the colour half stays
// fused there because it shares intermediates like the swirl warp). Used by
// the light march below so the self-shadow sees exactly the field it shadows.
float rh_dustDensity(vec3 P, float scl, uint dustStyle) {
    float dust;
    if (dustStyle == 2u) {
        float n = rh_fbm(P * scl * vec3(2.4, 0.55, 2.4));
        dust = pow(1.0 - abs(2.0 * n - 1.0), 3.0);
    } else if (dustStyle == 3u) {
        vec3 wp = P * scl;
        vec3 warp = vec3(rh_fbm(wp * 0.5),
                         rh_fbm(wp * 0.5 + vec3(19.7)),
                         rh_fbm(wp * 0.5 + vec3(47.3))) - 0.5;
        dust = rh_fbm(wp + warp * 2.6);
    } else {   // styles 0/1 share the billow field (they differ in colour only)
        dust = rh_fbm(P * scl) * (0.6 + 0.8 * rh_vnoise(P * scl * 3.1));
    }
    return pow(dust, 2.0);
}
// Dust transmittance TOWARD THE LIGHT: 3 jittered steps through the dust
// field only (no scene rays, no lattice). One mechanism buys two behaviours:
// dense cores shadow themselves (lit side bright, core dark) and shadow the
// specks suspended below them. σt matches the camera march (dust * 6).
float rh_dustLightTr(vec3 P, vec3 ldir, float scl, uint dustStyle,
                     float inclusion, float span, float jit) {
    float ldt = span / 3.0;
    float tau = 0.0;
    for (int j = 0; j < 3; ++j)
        tau += rh_dustDensity(P + ldir * ((float(j) + jit) * ldt), scl, dustStyle);
    return exp(-tau * inclusion * ldt * 6.0);
}
// Dual-lobe phase, 4π-normalized (isotropic = 1): 65% forward HG g=0.55 —
// the backlit "silver lining" — plus 35% isotropic so side/back lighting
// never goes fully dark. NOT a parameter: one curated look, zero ABI growth.
float rh_dustPhase(float cosT) {
    const float g = 0.55, g2 = g * g;
    float hgN = (1.0 - g2) * pow(max(1.0 + g2 - 2.0 * g * cosT, 1e-3), -1.5);
    return mix(1.0, hgN, 0.65);
}
// Self-shadow inside the speck lattice: a short DDA from a lit speck TOWARD
// the light, testing only the lattice (no scene rays — from inside an
// opaque-based resin those always self-occlude on the enclosing surface).
// Dirt spheres block the light; shard chips TINT the shadow like stained
// glass; bubbles and dust are ignored. 8 cells ≈ a few speck diameters —
// enough for the neighbour-shadowing depth cue, cheap enough per lit speck.
vec3 resinSpeckShadow(vec3 qFrom, vec3 ldir, float totalAmt, float shardCut,
                      float shardHue) {
    vec3 shadow = vec3(1.0);
    vec3 cell  = floor(qFrom);
    vec3 cell0 = cell;
    vec3 sgn = vec3(ldir.x >= 0.0 ? 1.0 : -1.0,
                    ldir.y >= 0.0 ? 1.0 : -1.0,
                    ldir.z >= 0.0 ? 1.0 : -1.0);
    vec3 ad = max(abs(ldir), vec3(1e-6));
    vec3 tDelta = 1.0 / ad;
    vec3 fr = qFrom - cell;
    vec3 tMax = vec3((ldir.x >= 0.0 ? 1.0 - fr.x : fr.x) / ad.x,
                     (ldir.y >= 0.0 ? 1.0 - fr.y : fr.y) / ad.y,
                     (ldir.z >= 0.0 ? 1.0 - fr.z : fr.z) / ad.z);
    for (int it = 0; it < 8; ++it) {
        if (!all(equal(cell, cell0)) &&
            rh_hash13(cell + vec3(5.77)) < totalAmt) {
            vec3  h      = rh_hash33(cell + 17.31);
            vec3  seedPt = cell + rh_hash33(cell);
            float rad    = mix(0.10, 0.26, h.x);
            vec3  oc     = qFrom - seedPt;
            float bq     = dot(oc, ldir);
            float perp2  = dot(oc, oc) - bq * bq;
            if (bq < 0.0) {
                if (h.y < shardCut) {
                    float r = rad * 1.05;
                    if (perp2 < r * r) {
                        float hue = (shardHue >= 0.0)
                                  ? fract(shardHue + (h.z - 0.5) * 0.16) : h.z;
                        shadow *= mix(vec3(1.0), rh_hue(hue), 0.6);
                    }
                } else if (h.y >= shardCut + 0.25 * (1.0 - shardCut)) {
                    float r = rad * 0.95;
                    if (perp2 < r * r) { shadow *= 0.18; break; }
                }
            }
        }
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            tMax.x += tDelta.x; cell.x += sgn.x;
        } else if (tMax.y < tMax.z) {
            tMax.y += tDelta.y; cell.y += sgn.y;
        } else {
            tMax.z += tDelta.z; cell.z += sgn.z;
        }
    }
    return shadow;
}

ResinMarch resinMarchInterior(vec3 origin, vec3 Tdir, float thickness,
                              vec3 extBase, float inclusion, float dirtAmt,
                              vec3 dirtColor, float shardAmt, float shardHue,
                              vec3 dustBaseTint, vec3 lightDir,
                              float scl, uint dustStyle, vec3 dustA, vec3 dustB,
                              uint shardShape, inout uint seed) {
    ResinMarch rm;
    rm.absorb = vec3(1.0); rm.dustCover = 0.0; rm.dustTint = dustBaseTint;
    rm.dirtHit = false; rm.dirtAlbedo = vec3(0.0); rm.sparkle = 0.0;
    rm.shardGlow = vec3(0.0); rm.dustGlow = vec3(0.0);
    vec3  dustAcc  = vec3(0.0);
    float coverRaw = 0.0;
    vec3  glowAcc  = vec3(0.0);
    // Directional single scatter: the phase angle is fixed per march (light
    // and view directions are constant), the light transmittance is marched
    // lazily and cached for two steps (halves the light-march cost; the
    // field is low-frequency at that scale).
    float lSpan  = max(thickness, 1e-3) * 0.6;
    float phMix  = rh_dustPhase(dot(lightDir, Tdir));
    float Tlight = 1.0;
    int   tlAge  = 99;

    // ── Phase A: dust — jittered stochastic march (unchanged recipe) ────────
    const int STEPS = 12;
    float dt  = max(thickness, 1e-3) / float(STEPS);
    float jit = rnd(seed);
    for (int i = 0; i < STEPS; ++i) {
        vec3 P = origin + Tdir * ((float(i) + jit) * dt);
        // Dust field + colour, by STYLE:
        //   0 Nebula (auto)  — billow turbulence; colour drifts between the
        //     derived base tint and its .gbr hue-rotation (legacy default).
        //   1 Billow 2-colour — same field, colour mixes between the user's
        //     A/B poles on a low frequency.
        //   2 Wispy streaks  — anisotropically stretched ridged filaments
        //     (long horizontal wisps), A/B coloured.
        //   3 Paint swirl    — DOMAIN-WARPED fbm: ink-in-water curls; the
        //     colour field is warped by the same flow so the A/B pigments
        //     fold into each other like stirred paint.
        float dust;
        vec3  nb;
        if (dustStyle == 1u) {
            dust = rh_fbm(P * scl) * (0.6 + 0.8 * rh_vnoise(P * scl * 3.1));
            nb   = mix(dustA, dustB,
                       smoothstep(0.30, 0.70, rh_vnoise(P * scl * 0.5 + vec3(11.3))));
        } else if (dustStyle == 2u) {
            vec3 Ps = P * scl * vec3(2.4, 0.55, 2.4);
            float n = rh_fbm(Ps);
            dust = pow(1.0 - abs(2.0 * n - 1.0), 3.0);
            nb   = mix(dustA, dustB,
                       smoothstep(0.35, 0.65, rh_vnoise(Ps * 0.4 + vec3(5.9))));
        } else if (dustStyle == 3u) {
            vec3 wp = P * scl;
            vec3 warp = vec3(rh_fbm(wp * 0.5),
                             rh_fbm(wp * 0.5 + vec3(19.7)),
                             rh_fbm(wp * 0.5 + vec3(47.3))) - 0.5;
            dust = rh_fbm(wp + warp * 2.6);
            nb   = mix(dustA, dustB,
                       smoothstep(0.35, 0.65, rh_fbm(wp * 0.7 + warp * 1.8)));
        } else {
            dust = rh_fbm(P * scl) * (0.6 + 0.8 * rh_vnoise(P * scl * 3.1));
            float hueT = smoothstep(0.25, 0.75, rh_vnoise(P * scl * 0.6 + vec3(31.7)));
            nb   = mix(dustBaseTint, dustBaseTint.gbr, hueT);
        }
        dust = pow(dust, 2.0) * inclusion;                 // sparse wispy cores
        float trAvg = dot(rm.absorb, vec3(0.3333));
        float w    = dust * dt * 2.5 * trAvg;
        if (w > 1e-5) {
            // LIT VOLUME MODEL (single scatter + multi-scatter floor):
            //   lit = ambient + Tlight·phase (directional single scatter)
            //       + (1-Tlight)·floor (light absorbed on the way partially
            //         re-emerging as diffuse glow — the cheap stand-in for
            //         multiple scattering; energy-limited, never > absorbed).
            // Calibrated so an unshadowed side-lit step ≈ 1.0 (the confirmed
            // pre-TUR-9 look); backlit forward peak reaches ~4 and the excess
            // past the coverage clamp goes out as ADDITIVE dustGlow.
            if (tlAge >= 2) {
                Tlight = rh_dustLightTr(P, lightDir, scl, dustStyle,
                                        inclusion, lSpan, jit);
                tlAge = 0;
            }
            float lit = min(0.25 + 1.15 * Tlight * phMix
                                 + 0.45 * (1.0 - Tlight), 4.0);
            float cw = w * min(lit, 1.2);
            coverRaw += cw;
            dustAcc  += nb * cw;
            glowAcc  += nb * w * (max(lit, 1.2) - 1.2) * 0.35;
        }
        tlAge++;
        rm.absorb *= exp(-dt * (extBase + vec3(dust * 6.0)));
    }
    rm.dustCover = clamp(coverRaw, 0.0, 0.7);
    if (coverRaw > 1e-4) rm.dustTint = clamp(dustAcc / coverRaw, 0.0, 1.0);
    rm.dustGlow = min(glowAcc, vec3(1.5));

    // ── Phase B: specks — 3D DDA cell walk, fully DETERMINISTIC ─────────────
    // The speck field walks EVERY noise cell the ray crosses, in order (voxel
    // DDA), independent of the dust march's sample points. The previous
    // version only saw a speck when a dust sample happened to land in ITS
    // cell — with a step spanning several cells most passes missed it, and
    // accumulation averaged hit/miss into translucent, blurred blobs (the
    // "flu lekeler" report). Population: one candidate per cell, existence
    // hash < (dirt+shard) so the knobs control density; type split honours
    // the dirt/shard ratio, bubbles take a fixed slice of the dirt share.
    // Transmittance at a speck's depth is approximated channelwise as
    // absorb^(t/tEnd) — consistent with Phase A without re-marching the dust.
    float totalAmt = clamp(dirtAmt + shardAmt, 0.0, 1.0);
    if (totalAmt > 0.001) {
        float shardCut = shardAmt / max(dirtAmt + shardAmt, 1e-4);
        float s3   = scl * 3.0;                 // world → speck noise space
        vec3  qo   = origin * s3;
        float tEnd = max(thickness, 1e-3) * s3;
        vec3  cell = floor(qo);
        vec3  sgn  = vec3(Tdir.x >= 0.0 ? 1.0 : -1.0,
                          Tdir.y >= 0.0 ? 1.0 : -1.0,
                          Tdir.z >= 0.0 ? 1.0 : -1.0);
        vec3  ad   = max(abs(Tdir), vec3(1e-6));
        vec3  tDelta = 1.0 / ad;
        vec3  frac0  = qo - cell;
        vec3  tMax = vec3(
            (Tdir.x >= 0.0 ? 1.0 - frac0.x : frac0.x) / ad.x,
            (Tdir.y >= 0.0 ? 1.0 - frac0.y : frac0.y) / ad.y,
            (Tdir.z >= 0.0 ? 1.0 - frac0.z : frac0.z) / ad.z);
        float tCur = 0.0;
        for (int it = 0; it < 48 && tCur < tEnd; ++it) {
            if (rh_hash13(cell + vec3(5.77)) < totalAmt) {
                vec3  h      = rh_hash33(cell + 17.31);
                vec3  seedPt = cell + rh_hash33(cell);  // same layout worley used
                float rad    = mix(0.10, 0.26, h.x);    // per-speck size
                vec3  oc     = qo - seedPt;
                float bq     = dot(oc, Tdir);
                float perp2  = dot(oc, oc) - bq * bq;
                if (h.y < shardCut) {
                    // GLASS SHARD: translucent colour chip. Tints what lies
                    // behind it (stained glass) AND carries its own visible
                    // colour body (shardGlow) so it reads on an opaque resin
                    // base too, plus a bright rim glint. Palette: material
                    // base hue ± hashed spread, or full rainbow when the hue
                    // knob is negative. Shape 1 = CRYSTAL: an elongated
                    // ellipsoid (random per-shard axis, ~2.6x) intersected in
                    // squashed space, with the normal QUANTIZED to flat facets
                    // — sharp lighting breaks read as cut crystal faces.
                    float r = rad * 1.05;
                    vec3  ocs = oc, ds = Tdir;
                    if (shardShape == 1u) {
                        vec3 axis = normalize(rh_hash33(cell + 3.3) - 0.5 + vec3(1e-4));
                        const float k = 0.38;                 // 1/k ≈ 2.6x elongation
                        ocs = oc  - axis * (dot(oc,  axis) * (1.0 - k));
                        ds  = Tdir - axis * (dot(Tdir, axis) * (1.0 - k));
                    }
                    float A  = dot(ds, ds);
                    float Bq = dot(ocs, ds);
                    float Cq = dot(ocs, ocs) - r * r;
                    float disc = Bq * Bq - A * Cq;
                    if (disc > 0.0 && Bq < 0.0) {
                        float tn = (-Bq - sqrt(disc)) / max(A, 1e-6);
                        if (tn > 0.0 && tn < tEnd) {
                            float hue  = (shardHue >= 0.0)
                                       ? fract(shardHue + (h.z - 0.5) * 0.16) : h.z;
                            vec3  sc   = rh_hue(hue);
                            // closest-approach distance in (possibly squashed) space
                            float dmin2 = max(Cq + r * r - Bq * Bq / max(A, 1e-6), 0.0);
                            float grz  = sqrt(dmin2) / r;             // 0 centre → 1 graze
                            float body = 1.0 - smoothstep(0.65, 1.0, grz);
                            vec3  T    = pow(max(rm.absorb, vec3(1e-4)),
                                             vec3(clamp(tn / tEnd, 0.0, 1.0)));
                            vec3 lit = vec3(1.0);
                            if (shardShape == 1u) {
                                // Faceted crystal shading: quantize the surface
                                // normal into a per-shard rotated lattice and
                                // light it — flat faces that FLASH as the light
                                // or the object turns. Neighbouring specks
                                // self-shadow the directional term (stained-
                                // glass tinted when the occluder is a shard).
                                vec3 pn = normalize(ocs + ds * tn);
                                vec3 h2 = rh_hash33(cell + 91.3);
                                vec3 fn = normalize(round(pn * 1.4 + (h2 - 0.5) * 0.8) + vec3(1e-3));
                                pn = normalize(mix(pn, fn, 0.85));
                                vec3 sshadow = (body > 0.2)
                                    ? resinSpeckShadow(qo + Tdir * tn, lightDir,
                                                       totalAmt, shardCut, shardHue)
                                    : vec3(1.0);
                                if (body > 0.2 && inclusion > 1e-3)
                                    sshadow *= rh_dustLightTr(origin + Tdir * (tn / s3),
                                                              lightDir, scl, dustStyle,
                                                              inclusion, lSpan, 0.5);
                                lit = vec3(0.45)
                                    + (0.85 * max(dot(pn, lightDir), 0.0)) * sshadow
                                    + vec3(pow(max(dot(pn, -Tdir), 0.0), 6.0) * 0.6);
                            }
                            rm.absorb    *= mix(vec3(1.0), sc, body * 0.85);
                            rm.shardGlow += sc * T * (0.20 + 0.30 * body) * lit;
                            rm.sparkle   += smoothstep(0.55, 0.95, grz) * 0.15
                                          * dot(T, vec3(0.3333));
                        }
                    }
                } else if (h.y < shardCut + 0.25 * (1.0 - shardCut)) {
                    // BUBBLE: bright rim where the ray grazes the shell.
                    float r = rad;
                    if (perp2 < r * r && bq < 0.0) {
                        float tn = -bq - sqrt(r * r - perp2);
                        if (tn > 0.0 && tn < tEnd) {
                            float grz = sqrt(perp2) / r;
                            vec3  T   = pow(max(rm.absorb, vec3(1e-4)),
                                            vec3(clamp(tn / tEnd, 0.0, 1.0)));
                            rm.sparkle += smoothstep(0.45, 0.95, grz) * 0.30
                                        * dot(T, vec3(0.3333));
                        }
                    }
                } else {
                    // DIRT: analytic ray-sphere — exact, identical every pass
                    // (sharp silhouette). First hit terminates; cells are
                    // visited in order so shard tints beyond it never apply.
                    float r = rad * 0.95;
                    if (perp2 < r * r && bq < 0.0) {
                        float tn = -bq - sqrt(r * r - perp2);
                        if (tn > 0.0 && tn < tEnd) {
                            vec3 pn = normalize((qo + Tdir * tn) - seedPt);
                            // REAL sampled light direction (surface NEE pick):
                            // specks brighten on the light side, fall dark
                            // opposite. No scene shadow ray (would always
                            // self-occlude inside an opaque-based resin) —
                            // instead the LATTICE self-shadows: neighbouring
                            // dirt blocks the directional term, shards tint it.
                            vec3  sshadow = resinSpeckShadow(qo + Tdir * tn, lightDir,
                                                             totalAmt, shardCut, shardHue);
                            // Dense dust above also shadows the speck (same
                            // short light march the dust shades itself with).
                            if (inclusion > 1e-3)
                                sshadow *= rh_dustLightTr(origin + Tdir * (tn / s3),
                                                          lightDir, scl, dustStyle,
                                                          inclusion, lSpan, 0.5);
                            float ndl = max(dot(pn, lightDir), 0.0);
                            vec3  lit = vec3(0.28) + (0.72 * ndl) * sshadow;
                            float rim = pow(clamp(1.0 - abs(dot(pn, Tdir)), 0.0, 1.0), 3.0) * 0.25;
                            vec3  col = dirtColor * (0.70 + 0.60 * h.z);
                            float depthN = clamp(tn / tEnd, 0.0, 1.0);
                            vec3  T   = pow(max(rm.absorb, vec3(1e-4)), vec3(depthN));
                            rm.dirtAlbedo = clamp(col * lit + vec3(rim), 0.0, 1.0) * T;
                            rm.dirtHit = true;
                            // Trim the dust of the UNREACHED depth off the result.
                            rm.absorb    = T;
                            rm.dustCover *= depthN;
                            rm.dustGlow  *= depthN;
                            break;
                        }
                    }
                }
            }
            // advance to the next crossed cell
            if (tMax.x < tMax.y && tMax.x < tMax.z) {
                tCur = tMax.x; tMax.x += tDelta.x; cell.x += sgn.x;
            } else if (tMax.y < tMax.z) {
                tCur = tMax.y; tMax.y += tDelta.y; cell.y += sgn.y;
            } else {
                tCur = tMax.z; tMax.z += tDelta.z; cell.z += sgn.z;
            }
        }
    }
    return rm;
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
// Interior-volume anchor: bit 21 of mat.flags (VK_MAT_FLAG_RESIN_OBJ_SPACE).
// Set = the dust/speck fields are evaluated in OBJECT space (the interior
// moves/rotates with the mesh); clear = legacy world anchor (fixed in space).
const uint MAT_FLAG_RESIN_OBJ_SPACE = (1u << 21);
// Glass marble full-volume entry: tagged on the FRONT-face transmit so raygen
// integrates the real interior segment (dust/dirt) before the next surface.
const uint BOUNCE_MARBLE = 5u;
// Glass mirror lobe (Fresnel reflect or TIR at an interface): the ray did NOT
// cross the surface. Kept distinct from BOUNCE_TRANSMISSION so the photon pass
// only counts real refractions as "crossed glass" — tagging reflections as
// transmission made photons bounced off a sphere's OUTER surface splat a
// mirrored ghost caustic on the floor. Camera-side raygen spends the same
// transmission budget on it, so camera behavior is unchanged.
const uint BOUNCE_GLASS_REFLECT = 6u;

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
void scatterGlass(vec3 hitPos, vec3 macroNormalIn, vec3 shadingNormalIn, bool frontFace, vec3 rayDir, vec3 albedo, float ior, float roughness, float transmissionDensity, vec3 resinColor, float dispersion, inout uint seed) {
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

    // ── Spectral dispersion: ONLY the refracted lobe disperses. The mirror lobe is
    // wavelength-independent — selecting the hero channel before the lobe decision
    // splashed ×3 mono-channel noise onto reflection-lit surfaces. Channel is chosen
    // ONCE per path (payload.dispersionChannel persists, reset per path in raygen)
    // so the exit interface refracts with the same channel IOR. Selection collapses
    // attenuation to one channel ×3; blue bends more than red. Resin path skipped.
    if (!doReflect && !realDepth && dispersion > 1e-3) {
        int dispCh = int((payload.primaryMeta & PL_DISP_MASK) >> PL_DISP_SHIFT) - 1;   // -1 = unset, 0/1/2 = R/G/B
        if (dispCh < 0) {
            dispCh = min(int(rnd(seed) * 3.0), 2);
            vec3 sel = vec3(0.0);
            sel[dispCh] = 3.0;
            payload.attenuation *= sel;
            payload.primaryMeta = (payload.primaryMeta & ~PL_DISP_MASK)
                                | (uint(dispCh + 1) << PL_DISP_SHIFT);
        }
        float spread = (ior - 1.0) * dispersion * 0.06;    // half of the total F–C spread
        ior += (dispCh == 0) ? -spread : ((dispCh == 2) ? spread : 0.0);
        etaRatio = frontFace ? (1.0 / ior) : ior;          // refraction uses channel IOR
    }

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
    bool didRefract = false;               // true only when the ray actually crossed the interface
    if (doReflect) {
        dir       = reflect(rayDir, outNormal);
        offsetDir = macroNormal;           // Yüzeyin dışına offset
    } else {
        bool refractedSuccess = refractLikeOptix(rayDir, outNormal, etaRatio, dir);
        didRefract = refractedSuccess;
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
        didRefract = false;
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
    payload.bounceType     = didRefract ? BOUNCE_TRANSMISSION : BOUNCE_GLASS_REFLECT;
}

// Explicit-light response for Water V3. The generic material NEE block is
// intentionally bypassed by the water fast path, so water needs its own
// dielectric GGX estimator or scene lights only appear through secondary rays.
void addWaterV3DirectLighting(vec3 hitPos,
                              vec3 carrierNormalIn,
                              vec3 macroNormalIn,
                              vec3 shadingNormalIn,
                              vec3 rayDir,
                              float ior,
                              float roughness,
                              float foamCoverage,
                              inout uint seed) {
    if (cam.lightCount == 0u) return;

    vec3 carrierNormal = safeNormalize(carrierNormalIn, vec3(0.0, 1.0, 0.0));
    vec3 macroNormal = safeNormalize(macroNormalIn, carrierNormal);
    vec3 N = safeNormalize(shadingNormalIn, macroNormal);
    vec3 V = safeNormalize(-rayDir, macroNormal);
    float NdotV = max(dot(N, V), 0.0);
    if (NdotV <= 1e-5 || dot(macroNormal, V) <= 1e-5) return;

    float pdfSelect = 0.0;
    int lightIndex = pick_smart_light_gl(uvec2(0), hitPos, pdfSelect);
    if (lightIndex < 0 || pdfSelect <= 0.0) return;

    vec3 L;
    float distanceToLight;
    float lightAttenuation;
    if (!sample_light_direction_gl(lights.l[lightIndex], hitPos,
                                   rnd(seed), rnd(seed),
                                   L, distanceToLight, lightAttenuation)) return;
    L = safeNormalize(L, macroNormal);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 1e-5 || dot(macroNormal, L) <= 1e-5) return;

    // Offsetting with the interpolated normal can leave the origin below an
    // adjacent triangle. Always use the true oriented carrier face here.
    vec3 shadowOrigin = offset_ray(hitPos, carrierNormal);
    float tMax = min(max(distanceToLight - 1e-3, SHADOW_TMIN * 2.0), 10000.0);
    shadowPayload = vec4(1.0, 1.0, 1.0, 0.0);
    uint shadowFlags = gl_RayFlagsTerminateOnFirstHitEXT
                     | gl_RayFlagsSkipClosestHitShaderEXT;
    traceRayEXT(topLevelAS, shadowFlags, 0x01, 0, 1, 1,
                shadowOrigin, SHADOW_TMIN, L, tMax, 1);
    vec3 visibility = shadowPayload.w > 0.5 ? shadowPayload.rgb : vec3(0.0);
    if (!any(greaterThan(visibility, vec3(1e-4)))) return;

    vec3 H = safeNormalize(V + L, N);
    float NdotH = max(dot(N, H), 1e-5);
    float VdotH = max(dot(V, H), 1e-5);
    float safeRoughness = clamp(roughness, 0.02, 1.0);
    float alpha = max(safeRoughness * safeRoughness, 1e-4);
    float alpha2 = alpha * alpha;
    float dDenom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / max(PI * dDenom * dDenom, 1e-8);
    float k = (safeRoughness + 1.0);
    k = (k * k) * 0.125;
    float Gv = NdotV / max(NdotV * (1.0 - k) + k, 1e-5);
    float Gl = NdotL / max(NdotL * (1.0 - k) + k, 1e-5);
    float f0Scalar = pow((max(ior, 1.0001) - 1.0) / (max(ior, 1.0001) + 1.0), 2.0);
    vec3 F = vec3(f0Scalar) + (vec3(1.0) - vec3(f0Scalar)) * pow(1.0 - VdotH, 5.0);
    vec3 dielectricSpecular = F * D * (Gv * Gl) / max(4.0 * NdotV * NdotL, 1e-6);

    float foam = clamp(foamCoverage, 0.0, 1.0);
    vec3 foamDiffuse = mix(vec3(0.72, 0.76, 0.75), vec3(0.98), foam) * INV_PI;
    vec3 brdf = dielectricSpecular * (1.0 - foam) + foamDiffuse * foam;
    vec3 Li = lights.l[lightIndex].color.rgb * lights.l[lightIndex].color.a * lightAttenuation;

    int lightType = int(lights.l[lightIndex].position.w + 0.5);
    bool deltaLight = lightType == 0 || lightType == 1;
    float estimatorWeight;
    if (deltaLight) {
        estimatorWeight = 1.0 / max(pdfSelect, 1e-6);
    } else {
        float lightPdf = compute_light_pdf_gl(lights.l[lightIndex], distanceToLight, 1.0) * pdfSelect;
        float bsdfPdf = pdf_brdf_gl(N, V, L, safeRoughness);
        estimatorWeight = power_heuristic(lightPdf, bsdfPdf) / max(lightPdf, 1e-6);
    }

    float volumeTransmittance = computeVolumeShadowTransmittance(shadowOrigin, L, tMax);
    vec3 contribution = brdf * Li * NdotL * estimatorWeight
                      * visibility * volumeTransmittance;
    contribution = clamp(max(contribution, vec3(0.0)), vec3(0.0), vec3(1e4));
    payload.radiance += clamp(payload.attenuation, vec3(0.0), vec3(1e2)) * contribution;
}

// Water V3 dielectric sampler. The true carrier face owns interface crossing
// and ray offsets, the smooth macro normal owns Fresnel, and the resolved
// surface normal owns reflection/refraction detail.
void scatterWaterV3Dielectric(vec3 hitPos,
                              vec3 carrierNormalIn,
                              vec3 macroNormalIn,
                              vec3 shadingNormalIn,
                              bool frontFace,
                              vec3 rayDir,
                              vec3 bodyTint,
                              float waterDepth,
                              float absorptionDensity,
                              float ior,
                              float roughness,
                              inout uint seed) {
    vec3 carrierNormal = safeNormalize(carrierNormalIn, vec3(0.0, 1.0, 0.0));
    vec3 macroNormal = safeNormalize(macroNormalIn, carrierNormal);
    vec3 shadingNormal = safeNormalize(shadingNormalIn, macroNormal);
    if (dot(shadingNormal, macroNormal) < 0.0) shadingNormal = -shadingNormal;

    float safeIor = max(ior, 1.0001);
    float etaRatio = frontFace ? (1.0 / safeIor) : safeIor;
    vec3 facetNormal = shadingNormal;

    // Resolved waves stay present at every roughness. GGX only represents the
    // unresolved distribution around that already-resolved normal.
    if (roughness > 0.0005) {
        float sampleRoughness = max(roughness, 0.004);
        vec3 sampledFacet = ggxSampleHemisphere(shadingNormal, -rayDir,
                                                sampleRoughness, seed);
        float blend = smoothstep(0.0, 0.035, roughness);
        facetNormal = safeNormalize(mix(shadingNormal, sampledFacet, blend), shadingNormal);
        if (dot(facetNormal, macroNormal) < 0.02) facetNormal = shadingNormal;
    }

    float cosTheta = clamp(dot(-rayDir, macroNormal), 0.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool totalInternalReflection = etaRatio * sinTheta > 1.0;
    bool reflectLobe = totalInternalReflection || rnd(seed) < schlickFresnel(cosTheta, safeIor);

    vec3 direction;
    vec3 offsetDirection;
    bool crossedInterface = false;
    if (reflectLobe) {
        direction = reflect(rayDir, facetNormal);
        offsetDirection = carrierNormal;
        if (dot(direction, carrierNormal) <= 0.0) direction = reflect(rayDir, carrierNormal);
    } else {
        crossedInterface = refractLikeOptix(rayDir, facetNormal, etaRatio, direction);
        offsetDirection = -carrierNormal;
        if (!crossedInterface || dot(direction, carrierNormal) >= 0.0) {
            direction = reflect(rayDir, carrierNormal);
            offsetDirection = carrierNormal;
            crossedInterface = false;
        }
    }

    payload.scatterOrigin = offset_ray(hitPos, offsetDirection);
    payload.scatterDir = safeNormalize(direction, reflect(rayDir, macroNormal));
    if (crossedInterface) {
        float cosineThroughSurface = max(abs(dot(payload.scatterDir, -macroNormal)), 0.12);
        float opticalDistance = min(max(waterDepth, 0.02) / cosineThroughSurface, 40.0);
        vec3 extinction = (vec3(1.0) - clamp(bodyTint, vec3(0.0), vec3(0.999))) *
                          (0.12 + max(absorptionDensity, 0.0) * 0.35);
        payload.attenuation *= exp(-extinction * opticalDistance);
    }
    payload.scattered = true;
    payload.bounceType = crossedInterface ? BOUNCE_TRANSMISSION : BOUNCE_GLASS_REFLECT;
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
//   matx.anisotropic          = wave_speed
//   matx.sheen                = wave_strength  (IS_WATER flag when > 0)
//   matx.sheen_tint           = wave_frequency
//   mat.emission_r/g/b       = shallow_color
//   mat.albedo_r/g/b         = deep_color
//   mat.translucent          = foam_level
//   mat.subsurface_amount    = depth_max / 100
//   matx.fft_time_scale       = animation speed multiplier
//   matx.micro_detail_strength= micro ripple strength
//   matx.micro_detail_scale   = micro ripple scale
//   matx.foam_threshold       = foam appearance threshold
//   matx.fft_wind_speed       = wind speed for micro ripples
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

// Foam micro-structure: anisotropic cell field, [0,1]. Filaments stretch along
// the first axis (downstream for rivers), a second rotated octave pops bubble
// holes into them. Advection happens in the caller's coordinate, so the same
// field serves rivers (ribbon metres) and open water (world XZ).
float waterFoamCells(vec2 q) {
    float filaments = water_fbm(q * vec2(0.55, 1.45));
    float bubbles   = water_fbm(q * 2.6 + vec2(37.2, 11.7));
    return clamp((filaments * 0.62 + bubbles * 0.38) * 0.5 + 0.5, 0.0, 1.0);
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
    vec3 probeOrigin = hitPos - vec3(0.0, 0.05, 0.0);
    vec3 probeDir = vec3(0.0, -1.0, 0.0);
    // OpaqueEXT: this is a DEPTH HEURISTIC, not lighting — running the shadow
    // any-hit (full material fetch + transmissive tint math per candidate) for
    // every probe bought nothing. Cutout/glass under the water now count as
    // floor, which is fine for a depth-tint estimate.
    uint probeFlags = gl_RayFlagsTerminateOnFirstHitEXT
                    | gl_RayFlagsSkipClosestHitShaderEXT
                    | gl_RayFlagsOpaqueEXT;

    // Existence probe over the whole window first: open water (no floor within
    // maxDepth — the common deep-ocean case) exits after ONE trace instead of
    // seven blind bisections.
    shadowPayload = vec4(1.0, 1.0, 1.0, 0.0);
    traceRayEXT(topLevelAS, probeFlags, 0x01, 0, 1, 1, probeOrigin, SHADOW_TMIN, probeDir, high, 1);
    if (shadowPayload.w >= 0.5) return false;

    for (int i = 0; i < 6; ++i) {
        float mid = mix(low, high, 0.5);
        shadowPayload = vec4(1.0, 1.0, 1.0, 0.0);
        traceRayEXT(topLevelAS, probeFlags, 0x01, 0, 1, 1, probeOrigin, SHADOW_TMIN, probeDir, mid, 1);
        if (shadowPayload.w < 0.5) high = mid;   // floor is within [low, mid]
        else                       low = mid;
    }

    waterDepth = high;
    floorPosition = hitPos - vec3(0.0, waterDepth, 0.0);
    return true;
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
                           float travel_direction,
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

    float cd = cos(travel_direction);
    float sd = sin(travel_direction);
    for (int i = 0; i < 8; ++i) {
        vec2 baseDir = dirs[i];
        vec2 d = vec2(baseDir.x * cd - baseDir.y * sd,
                      baseDir.x * sd + baseDir.y * cd);
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
void scatterWater(vec3 hitPos, vec3 geoNormal, vec3 carrierNormal, vec3 rayDir,
                  uint waterProfile, vec2 surfaceUV,
                  vec3 flowTangent, vec3 crossTangent,
                  vec4 hydrologyA, vec4 hydrologyB, vec4 hydrologyC,
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
                  bool writePrimaryNormal,
                  inout uint seed)
{
    // OptiX parity: FFT simulation is already time-scaled before textures are generated.
    // The shading pass uses the resolved water time directly for both FFT sampling and
    // micro-ripple drift so Vulkan does not double-accelerate the surface.
    float time = cam.waterTime;

    // ── Gerstner wave normal + foam ─────────────────────────────
    bool isRiver = waterProfile == 2u;
    bool isLake = waterProfile == 1u;
    WaterV3Hydrology hydrology = waterV3DecodeHydrology(hydrologyA, hydrologyB, hydrologyC);
    // Hydrology stores physical metres independently from texture UVs. The
    // legacy U convention was 0.5 units per metre, so retain that artistic
    // scale while making the cross-channel axis use the same unit.
    vec2 riverMetricUV = hydrology.width > 0.001
        ? vec2(hydrology.alongDistance, hydrology.crossDistance) * 0.5
        : surfaceUV;
    float riverSpeed = (hydrology.speed > 0.001 ? hydrology.speed : 1.0) * max(wave_speed, 0.01);
    float rapidResponse = waterV3RapidResponse(hydrology.froude);
    float dischargeResponse = 1.0 + min(log2(1.0 + hydrology.discharge) * 0.06, 0.35);
    float riverStrength = wave_strength * dischargeResponse * mix(0.65, 1.65, rapidResponse);

    // ── Shore treatment (rivers) ─────────────────────────────────
    // hydrology.depth carries the true water column above the carved bed and
    // reaches zero at the terrain waterline. Fade the interface out over the
    // last few centimetres: rays continue to the bank below, so the visible
    // waterline is the terrain intersection curve instead of the mesh edge.
    float shorePresence = 1.0;
    if (isRiver && hydrology.width > 0.001) {
        shorePresence = smoothstep(0.004, 0.05, hydrology.depth);
        if (shorePresence < 0.999 && rnd(seed) >= shorePresence) {
            payload.scatterOrigin = offset_ray(hitPos,
                dot(rayDir, carrierNormal) < 0.0 ? -carrierNormal : carrierNormal);
            payload.scatterDir = rayDir;
            payload.scattered = true;
            payload.bounceType = BOUNCE_TRANSMISSION;
            return;
        }
        // The surviving shallow band stays calm so the waterline lies flat
        // against the bank instead of carrying full channel waves.
        riverStrength *= mix(0.45, 1.0, shorePresence);
    }
    bool useFFTOcean = !isRiver && fft_height_tex > 0u && fft_normal_tex > 0u && fft_ocean_size > 0.001;
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
        waveNormal = isLake ? normalize(mix(vec3(0.0, 1.0, 0.0), fftNormal, 0.42)) : fftNormal;
        float fftSlope = clamp(1.0 - fftNormal.y, 0.0, 1.0);
        foam = smoothstep(max(foam_threshold, 0.05), 1.0, fftSlope * 2.0 + abs(fftHeight) * 0.25);
    } else if (isRiver) {
        waterV3EvaluateRiverSpectrum(riverMetricUV, time, hydrology,
                                     riverSpeed, riverStrength, wave_freq,
                                     waveNormal, foam);
    } else {
        evaluateWaterGerstner(hitPos, time, wave_speed, wave_strength, wave_freq, wind_direction,
                              waveNormal, foam);
    }
    float foamSignal = foam;
    // Preserve the continuous analytic/flow-scale normal separately. The
    // capillary FBM below is intentionally shading-only; exposing its finite-
    // difference cell boundaries to the denoiser normal AOV makes them look
    // like persistent glass cracks.
    vec3 macroWaveNormal = waveNormal;
    float riverFoamBreakup = 0.5;

    // ── Micro-detail capillary ripples ──────────────────────────
    if (micro_strength > 0.001) {
        if (isRiver) {
            // Foam coverage remains analytic, while the established FBM
            // micro-normal character below is restored for river rendering.
            vec3 unusedRiverMicroNormal;
            waterV3EvaluateRiverCapillary(riverMetricUV, time,
                                           riverSpeed * max(micro_anim_speed, 0.001),
                                           micro_strength, micro_scale,
                                           unusedRiverMicroNormal, riverFoamBreakup);
        }
        // micro_scale is authored in world space. Do not scale it by the FFT
        // ocean tile size, or large water surfaces lose their capillary detail.
        float sc = max(micro_scale, 0.001);
        float wind_dx = isRiver ? 1.0 : cos(wind_direction);
        float wind_dz = isRiver ? 0.0 : sin(wind_direction);
        float cross_dx = -wind_dz;
        float cross_dz = wind_dx;
        float base_speed = isRiver
            ? riverSpeed * max(micro_anim_speed, 0.001)
            : sqrt(max(1.0, wind_speed)) * max(micro_anim_speed, 0.001);
        float morph = max(micro_morph_speed, 0.001);
        vec2 surfaceCoord = isRiver ? riverMetricUV : hitPos.xz;

        float off1_x = wind_dx * time * base_speed + sin(time * 0.3 * morph) * 0.5;
        float off1_z = wind_dz * time * base_speed + cos(time * 0.2 * morph) * 0.5;
        vec2 p1 = surfaceCoord * sc + vec2(off1_x, off1_z);

        float off2_x = (wind_dx * 0.7 + cross_dx * 0.3) * time * base_speed * 0.6 + cos(time * 0.15 * morph + 1.5) * 0.8;
        float off2_z = (wind_dz * 0.7 + cross_dz * 0.3) * time * base_speed * 0.6 + sin(time * 0.25 * morph + 2.0) * 0.8;
        vec2 p2 = surfaceCoord * sc * 0.5 + vec2(off2_x, off2_z);

        float off3_x = cross_dx * time * base_speed * 0.4 + sin(time * 0.5 * morph + 3.0) * 0.3;
        float off3_z = cross_dz * time * base_speed * 0.4 + cos(time * 0.4 * morph + 1.0) * 0.3;
        vec2 p3 = surfaceCoord * sc * 2.0 + vec2(off3_x, off3_z);

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

        // Compose resolved wave and capillary detail in slope space. Adding two
        // unit normals directly halves both slopes near (0,1,0), which made the
        // Vulkan surface look flat precisely when the dielectric roughness was
        // low enough to expose the normal-field quality.
        waveNormal = waterV3ComposeSlopeNormals(waveNormal, microN);

        // Micro-peak foam (replaces/supplements Gerstner foam for FFT-style look)
        float microSlope = clamp(hc * 0.5 + 0.5, 0.0, 1.0);
        float scaledFoamNoise = max(foam_noise_scale, 0.001);
        float foamBreakup = water_fbm(surfaceCoord * scaledFoamNoise + vec2(off1_x, off1_z) * 0.5) * 0.5 + 0.5;
        float stableFoamBreakup = isRiver ? riverFoamBreakup : foamBreakup;
        float microFoam  = clamp((microSlope + (stableFoamBreakup - 0.5) * 0.35 - foam_threshold) * 5.0, 0.0, 1.0);
        foamSignal = max(foamSignal, microFoam);
    }
    WaterV3SurfaceSample waterSurface;
    waterSurface.macroNormalTS = macroWaveNormal;
    waterSurface.shadingNormalTS = waveNormal;
    waterSurface.foamProduction = clamp(foamSignal, 0.0, 1.0);
    waterSurface.depth = hydrology.depth;
    waterSurface.bankProximity = hydrology.bankProximity;
    waterSurface.speed = hydrology.speed;
    waterSurface.froude = hydrology.froude;

    // ── Build shading normal from wave perturbation ──────────────
    // waveNormal lives in a y-up tangent frame; project onto geoNormal's ONB
    vec3 tgt, btgt;
    if (isRiver && dot(flowTangent, flowTangent) > 1e-6) {
        tgt = normalize(flowTangent - geoNormal * dot(flowTangent, geoNormal));
        btgt = normalize(crossTangent - geoNormal * dot(crossTangent, geoNormal));
        if (dot(cross(tgt, btgt), geoNormal) < 0.0) btgt = -btgt;
    } else if (abs(geoNormal.y) > 0.999) {
        tgt = vec3(1.0, 0.0, 0.0);
        btgt = vec3(0.0, 0.0, 1.0);
    } else {
        tgt = normalize(cross(geoNormal, vec3(0.0, 0.0, 1.0)));
        btgt = cross(tgt, geoNormal);
    }
    vec3 shadingNormal = waterV3TangentToWorld(waterSurface.shadingNormalTS, tgt, geoNormal, btgt);
    if (dot(shadingNormal, -rayDir) < 0.0) shadingNormal = geoNormal; // sanity

    vec3 macroShadingNormal = waterV3TangentToWorld(waterSurface.macroNormalTS, tgt, geoNormal, btgt);
    if (dot(macroShadingNormal, -rayDir) < 0.0) macroShadingNormal = geoNormal;

    // The generic primary AOV is recorded before this water fast path and only
    // knows the static carrier normal. Replace it for the camera hit after the
    // resolved wave field exists, otherwise the Vulkan denoiser interprets the
    // water as a flat plane and removes macro/capillary reflection detail.
    if (writePrimaryNormal) {
        payload.primaryNrm = plPackNormal(macroShadingNormal);
    }

    float maxProbeDepth = max(depth_max, 0.1);
    float waterDepth = maxProbeDepth;
    vec3 floorPosition = hitPos - vec3(0.0, waterDepth, 0.0);
    bool foundFloor = false;
    if (isRiver && waterSurface.depth > 0.001) {
        waterDepth = min(waterSurface.depth, maxProbeDepth);
        floorPosition = hitPos - vec3(0.0, waterDepth, 0.0);
        foundFloor = true;
    } else if (shore_foam_intensity > 0.01 || absorption_density > 0.01 || caustic_intensity > 0.01) {
        foundFloor = estimateWaterDepthGL(hitPos, maxProbeDepth, waterDepth, floorPosition);
    }
    vec3 baseWaterColor = calculateDepthColorGL(waterDepth, depth_max, shallow_color, deep_color);
    if (caustic_intensity > 0.01 && foundFloor) {
        float causticVal = calculateWaterCausticsGL(floorPosition, time, caustic_scale, caustic_speed);
        float causticFade = exp(-waterDepth * absorption_density * 0.5);
        baseWaterColor += shallow_color * causticVal * caustic_intensity * causticFade;
    }
    float shoreFoam = 0.0;
    if (isRiver && shore_foam_intensity > 0.01) {
        float bankFoam = waterSurface.bankProximity * waterSurface.bankProximity
                       * shore_foam_intensity * mix(0.35, 1.0, riverFoamBreakup);
        // Waterline foam rides the true depth field: a thin band that peaks at
        // the terrain intersection and dies off toward the channel. The
        // presence factor keeps foam off the already-faded skirt fringe.
        float band = max(shore_foam_distance, 0.04);
        float waterline = 1.0 - smoothstep(0.0, band, hydrology.depth);
        float waterlineFoam = waterline * waterline * shore_foam_intensity
                            * mix(0.45, 1.0, riverFoamBreakup) * shorePresence;
        shoreFoam = max(bankFoam, waterlineFoam);
    } else if (shore_foam_intensity > 0.01 && foundFloor) {
        shoreFoam = calculateShoreFoamGL(waterDepth, shore_foam_distance, shore_foam_intensity, hitPos, time, max(foam_noise_scale, 0.001));
    }
    vec2 foamCoord = isRiver ? riverMetricUV : hitPos.xz;
    vec2 foamDrift = isRiver ? vec2(time * riverSpeed * 0.12, 0.0)
                             : vec2(cos(wind_direction), sin(wind_direction)) * time * 0.08;
    // One shared, advected cell field drives both the ragged coverage border
    // and the lit structure inside the patch, so borders and interior move as
    // a single coherent whitewater body.
    vec2 foamCellCoord = foamCoord * max(foam_noise_scale, 0.001) + foamDrift;
    float foamCell = waterFoamCells(foamCellCoord);
    float producedFoam = min(waterSurface.foamProduction * foam_level + shoreFoam, 1.0);
    float totalFoam = waterV3FoamCoverageStructured(producedFoam, foamCell, foam_threshold);

    // Authored roughness controls only unresolved gloss. Direct explicit-light
    // response and indirect dielectric scattering share this exact value.
    float capillaryRoughness = clamp(micro_strength * 0.18, 0.004, 0.035);
    float waterRoughness = max(roughness, capillaryRoughness);
    bool waterFrontFace = dot(rayDir, carrierNormal) < 0.0;
    if (waterFrontFace) {
        addWaterV3DirectLighting(hitPos, carrierNormal, macroShadingNormal,
                                 shadingNormal, rayDir,
                                 ior, mix(waterRoughness, 0.8, totalFoam),
                                 totalFoam, seed);
    }

    // Foam is a shading lobe, not animated geometry. This avoids the legacy
    // foam-sphere BLAS/TLAS rebuild path while still producing whitewater.
    if (totalFoam > 0.001 && rnd(seed) < totalFoam) {
        // Bubble holes: thin spots in the cell field let some rays reach the
        // dielectric beneath, so the patch sparkles and breaks up instead of
        // reading as a solid matte decal. Dense coverage closes the holes.
        float holeChance = smoothstep(0.62, 0.18, foamCell) * (0.55 - 0.35 * totalFoam);
        if (rnd(seed) >= holeChance) {
            // Relief: tilt the shading normal by the cell-field gradient in the
            // flow frame so clumps actually catch light. The two extra taps run
            // only on this lobe, and the detail stays out of the denoiser's
            // normal guide just like the capillary FBM above.
            const float tapDistance = 0.35;
            float cellU = waterFoamCells(foamCellCoord + vec2(tapDistance, 0.0));
            float cellV = waterFoamCells(foamCellCoord + vec2(0.0, tapDistance));
            vec3 foamNormalTS = normalize(vec3(-(cellU - foamCell) / tapDistance * 0.85, 1.0,
                                               -(cellV - foamCell) / tapDistance * 0.85));
            vec3 foamNormal = waterV3TangentToWorld(foamNormalTS, tgt, shadingNormal, btgt);
            if (dot(foamNormal, -rayDir) < 0.0) foamNormal = shadingNormal;
            // Crevices are waterlogged grey-cyan, crests dry bright white; a
            // grazing-angle rim mimics the forward scatter of the bubble mass.
            vec3 foamAlbedo = mix(vec3(0.52, 0.63, 0.66), vec3(0.96, 0.98, 0.99),
                                  smoothstep(0.25, 0.85, foamCell));
            float rim = pow(1.0 - clamp(dot(-rayDir, foamNormal), 0.0, 1.0), 3.0);
            foamAlbedo = min(foamAlbedo * (1.0 + 0.35 * rim), vec3(1.0));
            scatterDiffuse(hitPos, foamNormal, foamAlbedo, seed);
            return;
        }
        // This sample fell through a bubble hole: continue into the dielectric
        // below so broken foam shows glints of the water it rides on.
    }

    // The downward RT probe provides local optical depth. Use its shallow/deep
    // result for dielectric attenuation instead of one constant tint.
    vec3 transmissionTint = clamp(baseWaterColor, vec3(0.001), vec3(1.0));

    // Keep interface classification/Fresnel on the real carrier geometry while
    // using the resolved wave normal only for reflection/refraction direction.
    // This mirrors the CPU path's rec.normal vs rec.interpolated_normal split.
    // Authored roughness controls unresolved gloss; it must not switch off the
    // resolved normal field. A small capillary floor avoids a singular delta
    // lobe while preserving visibly glassy water at low authored roughness.
    scatterWaterV3Dielectric(hitPos, carrierNormal, macroShadingNormal,
                             shadingNormal, waterFrontFace,
                             rayDir, transmissionTint, waterDepth,
                             absorption_density, ior,
                             mix(waterRoughness, 0.8, totalFoam), seed);
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

    // Geometry-node Pointiness: same barycentric blend of the same per-vertex attribute the
    // CPU render interpolates (MeshPointiness.h). 0.5 (flat) whenever the block is absent —
    // no graph reads it, or this is a device-resident BLAS.
    float hitPointiness = 0.5;
    if (geo.pointinessAddr != 0) {
        PointinessBuffer ptBuf = PointinessBuffer(geo.pointinessAddr);
        hitPointiness = ptBuf.p[i0] * bary.x + ptBuf.p[i1] * bary.y + ptBuf.p[i2] * bary.z;
    }

    // Attribute node: the named per-vertex channels (sculpt/Geo-DAG masks, paint layers),
    // interleaved MP_ATTRIB_SLOTS floats per vertex, blended with the same barycentrics the
    // CPU uses (MeshAttr::sampleMaterialAttributes) so both backends read one value. Absent
    // block => all zeros = "unpainted", which is what the CPU returns too.
    float hitAttribs[MP_ATTRIB_SLOTS];
    for (int ai = 0; ai < MP_ATTRIB_SLOTS; ++ai) hitAttribs[ai] = 0.0;
    if (geo.attribAddr != 0) {
        AttribBuffer atBuf = AttribBuffer(geo.attribAddr);
        for (int ai = 0; ai < MP_ATTRIB_SLOTS; ++ai) {
            hitAttribs[ai] = atBuf.a[i0 * MP_ATTRIB_SLOTS + ai] * bary.x
                           + atBuf.a[i1 * MP_ATTRIB_SLOTS + ai] * bary.y
                           + atBuf.a[i2 * MP_ATTRIB_SLOTS + ai] * bary.z;
        }
    }

    // Hydrology is a dedicated geometry stream, not a material-node attribute.
    // A = flow direction XZ, water depth, bank proximity.
    // B = flow speed, discharge, Froude number, authored foam potential.
    // C = along-channel metres, normalized cross coordinate, local width, reserved.
    vec4 hitWaterA = vec4(0.0);
    vec4 hitWaterB = vec4(0.0);
    vec4 hitWaterC = vec4(0.0);
    if (geo.waterAddr != 0) {
        WaterVertexBuffer waterBuf = WaterVertexBuffer(geo.waterAddr);
        hitWaterA = waterBuf.w[i0 * 3 + 0] * bary.x
                  + waterBuf.w[i1 * 3 + 0] * bary.y
                  + waterBuf.w[i2 * 3 + 0] * bary.z;
        hitWaterB = waterBuf.w[i0 * 3 + 1] * bary.x
                  + waterBuf.w[i1 * 3 + 1] * bary.y
                  + waterBuf.w[i2 * 3 + 1] * bary.z;
        hitWaterC = waterBuf.w[i0 * 3 + 2] * bary.x
                  + waterBuf.w[i1 * 3 + 2] * bary.y
                  + waterBuf.w[i2 * 3 + 2] * bary.z;
    }

    // Compute UV coordinates if available
    vec2 hitUV = vec2(0.0);
    if (geo.uvAddr != 0) {
        UVBuffer uvBuf = UVBuffer(geo.uvAddr);
        uv0 = uvBuf.u[i0];
        uv1 = uvBuf.u[i1];
        uv2 = uvBuf.u[i2];
        hitUV = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    }

    // The RAW mesh UV, before the Vulkan V-flip. The material program runs in THIS
    // space, because that is the space the CPU VM runs in (it is handed rec.u/rec.v
    // straight off the hit record). Any program op that does explicit V math — the
    // Mapping node, MatOp::MatMapping — would otherwise compute on (u, 1-v) here and
    // on (u, v) there, and the two renders would quietly disagree. The GPU's TexColor
    // does the V-flip at sample time instead (material_program.glsl), which is exactly
    // what the CPU's get_color_bilinear already does internally.
    vec2 rawUV = hitUV;

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

    // Closed triangle-mesh volume boundary.  Do not shade the shell as a
    // surface: the entry hit simply continues into the mesh.  The following
    // back-face hit receives gl_HitTEXT equal to the actual segment travelled
    // inside, so it can apply the medium integral without a recursive exit-ray
    // query (and without confusing another object with this mesh's exit).
    if ((mat.flags & (1u << 24)) != 0u) {
        vec3 scatterColor = max(vec3(mat.albedo_r, mat.albedo_g, mat.albedo_b), vec3(0.0));
        if ((payload.primaryMeta & PL_PRIMARY_DONE) == 0u) {
            payload.primaryARG = packHalf2x16(scatterColor.rg);
            payload.primaryABT = packHalf2x16(vec2(scatterColor.b, 1.0));
            payload.primaryNrm = plPackNormal(-rayDir);
            payload.primaryMeta = (payload.primaryMeta & PL_DISP_MASK)
                                | (matIndex & PL_MATID_MASK) | PL_PRIMARY_DONE;
        }

        if (!surfaceFrontFace) {
            float segmentLength = max(gl_HitTEXT, 0.0);
            uint volumeProgram = matProgramOffset(matIndex);
            int maxSteps = clamp(int(matx.volume_max_steps + 0.5), 1, 256);
            float stepLength = max(matx.volume_step_size, segmentLength / float(maxSteps));
            int stepCount = min(maxSteps, max(1, int(ceil(segmentLength / max(stepLength, 1e-4)))));
            stepLength = segmentLength / float(stepCount);
            float transmittance = 1.0;
            vec3 integratedRadiance = vec3(0.0);

            vec3 lightDir = vec3(0.0, 1.0, 0.0);
            vec3 lightValue = vec3(0.0);
            float pdfSelect = 1.0;
            bool hasVolumeLight = false;
            float volumeLightDistance = segmentLength;
            if (cam.lightCount > 0u) {
                vec3 samplePoint = gl_WorldRayOriginEXT + rayDir * (0.5 * segmentLength);
                pdfSelect = 0.0;
                int lightIndex = pick_smart_light_gl(uvec2(0), samplePoint, pdfSelect);
                if (lightIndex >= 0 && pdfSelect > 1e-6) {
                    float lightAttenuation;
                    if (sample_light_direction_gl(
                            lights.l[lightIndex], samplePoint,
                            rnd(payload.seed), rnd(payload.seed),
                            lightDir, volumeLightDistance, lightAttenuation)) {
                        lightValue = lights.l[lightIndex].color.rgb *
                                     lights.l[lightIndex].color.a * lightAttenuation;
                        hasVolumeLight = true;
                    }
                }
            }

            for (int si = 0; si < stepCount && transmittance > 0.002; ++si) {
                float t = (float(si) + 0.5) * stepLength;
                vec3 p = gl_WorldRayOriginEXT + rayDir * t;
                float density = max(matx.volume_density, 0.0);
                float scatterStrength = max(matx.volume_scattering, 0.0);
                float absorptionStrength = max(matx.volume_absorption, 0.0);
                vec3 stepScatterColor = scatterColor;
                vec3 absorptionColor = vec3(1.0);
                vec3 emissionColor = max(vec3(mat.emission_r, mat.emission_g, mat.emission_b), vec3(0.0));
                float emissionStrength = 1.0;
                float anisotropy = matx.volume_anisotropy;
                float multiScatter = clamp(matx.volume_multi_scatter, 0.0, 1.0);

                if (volumeProgram != MATPROG_NONE) {
                    vec3 objP = gl_WorldToObjectEXT * vec4(p, 1.0);
                    MatProgOut vp = evalMaterialProgram(
                        volumeProgram, vec2(0.0), p, -rayDir, 0.5,
                        gl_ObjectToWorldEXT[3], hitAttribs, objP, -rayDir,
                        1.0, 0.0, 0.0, 0.0, vec3(0.0), p,
                        scatterColor, vec3(mat.emission_r, mat.emission_g, mat.emission_b),
                        stepLength, objP, cam.waterTime);
                    if ((vp.volumeWritten & (1u << 0)) != 0u) density = max(vp.volumeDensity, 0.0);
                    if ((vp.volumeWritten & (1u << 1)) != 0u) stepScatterColor = max(vp.volumeScatterColor, vec3(0.0));
                    if ((vp.volumeWritten & (1u << 2)) != 0u) scatterStrength = max(vp.volumeScatterStrength, 0.0);
                    if ((vp.volumeWritten & (1u << 3)) != 0u) absorptionColor = max(vp.volumeAbsorptionColor, vec3(0.0));
                    if ((vp.volumeWritten & (1u << 4)) != 0u) absorptionStrength = max(vp.volumeAbsorptionStrength, 0.0);
                    if ((vp.volumeWritten & (1u << 5)) != 0u) emissionColor = max(vp.volumeEmissionColor, vec3(0.0));
                    if ((vp.volumeWritten & (1u << 6)) != 0u) emissionStrength = max(vp.volumeEmissionStrength, 0.0);
                    if ((vp.volumeWritten & (1u << 7)) != 0u) anisotropy = clamp(vp.volumeAnisotropy, -0.95, 0.95);
                    if ((vp.volumeWritten & (1u << 8)) != 0u) multiScatter = clamp(vp.volumeMultiScatter, 0.0, 1.0);
                } else if (matx.volume_noise_scale > 0.0) {
                    density *= max(ch_fbmNoise(p * matx.volume_noise_scale, 4), 0.0);
                }

                float sigmaS = density * scatterStrength;
                vec3 sigmaAColor = density * absorptionStrength * max(absorptionColor, vec3(1e-4));
                float sigmaA = dot(sigmaAColor, vec3(0.2126, 0.7152, 0.0722));
                float sigmaT = sigmaS + sigmaA;
                float stepT = exp(-sigmaT * stepLength);
                if (multiScatter > 0.0 && sigmaS > 0.0) {
                    float albedoLum = dot(stepScatterColor, vec3(0.2126, 0.7152, 0.0722));
                    stepT = mix(stepT, exp(-sigmaT * stepLength * 0.25),
                                multiScatter * clamp(albedoLum, 0.0, 1.0));
                }
                vec3 source = emissionColor * emissionStrength * density;
                if (hasVolumeLight && sigmaS > 1e-6) {
                    int shadowSteps = clamp(int(matx.volume_light_steps + 0.5), 0, 48);
                    float shadowTrans = 1.0;
                    if (shadowSteps > 0 && matx.volume_shadow_strength > 0.0) {
                        float shadowLength = min(max(volumeLightDistance, 0.0), segmentLength);
                        float shadowStep = shadowLength / float(shadowSteps);
                        float shadowTau = 0.0;
                        for (int sj = 1; sj <= shadowSteps; ++sj) {
                            vec3 sp = p + normalize(lightDir) * ((float(sj) - 0.5) * shadowStep);
                            float sd = max(matx.volume_density, 0.0);
                            if (volumeProgram != MATPROG_NONE) {
                                vec3 sobjP = gl_WorldToObjectEXT * vec4(sp, 1.0);
                                MatProgOut svp = evalMaterialProgram(
                                    volumeProgram, vec2(0.0), sp, -rayDir, 0.5,
                                    gl_ObjectToWorldEXT[3], hitAttribs, sobjP, -rayDir,
                                    1.0, 0.0, 0.0, 0.0, vec3(0.0), sp,
                                    scatterColor, vec3(mat.emission_r, mat.emission_g, mat.emission_b),
                                    shadowStep, sobjP, cam.waterTime);
                                if ((svp.volumeWritten & (1u << 0)) != 0u) {
                                    sd = max(svp.volumeDensity, 0.0);
                                }
                            } else if (matx.volume_noise_scale > 0.0) {
                                sd *= max(ch_fbmNoise(sp * matx.volume_noise_scale, 4), 0.0);
                            }
                            shadowTau += sd * (scatterStrength + absorptionStrength) * shadowStep;
                            if (shadowTau > 12.0) break;
                        }
                        float physicalShadow = exp(-shadowTau);
                        shadowTrans = mix(1.0, physicalShadow,
                                          clamp(matx.volume_shadow_strength, 0.0, 1.0));
                    }
                    float g = clamp(anisotropy, -0.95, 0.95);
                    float g2 = g * g;
                    float cosTheta = dot(rayDir, normalize(lightDir));
                    float phase = (1.0 - g2) *
                        pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), -1.5);
                    source += stepScatterColor * lightValue * shadowTrans * phase *
                              sigmaS * (1.0 + multiScatter) / max(pdfSelect, 1e-6);
                }
                float integral = (sigmaT > 1e-6) ? (1.0 - stepT) / sigmaT : stepLength;
                integratedRadiance += transmittance * source * integral;
                transmittance *= stepT;
            }
            payload.radiance += integratedRadiance;
            payload.attenuation *= vec3(transmittance);
        }

        payload.scatterOrigin = hitPos + rayDir * 0.002;
        payload.scatterDir = rayDir;
        payload.scattered = true;
        payload.skipAABBs = false;
        payload.bounceType = BOUNCE_TRANSPARENT;
        return;
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

    // ── Faz 2b: per-pixel material program (mirrors the CPU MaterialProgram VM).
    // Overrides the driven slots point-by-point (Noise/Voronoi/Checker/Ramp/Mix
    // chains) using the RAW mesh UV — the same space the CPU VM runs in.
    // Procedural bump: the program can drive a tangent-space normal (Bump node).
    // Captured here, applied at the normal-map section below (needs the TBN).
    bool mpHasNormal = false;
    bool mpNormalWorld = false;   // Bevel: mpTangentNormal is a WORLD normal, skip the TBN
    vec3 mpTangentNormal = vec3(0.0, 0.0, 1.0);
    uint mpWritten = 0u;   // slots the program owns — bound textures must NOT overwrite them
    uint mp_procOff = matProgramOffset(matIndex);
    if (mp_procOff != MATPROG_NONE) {
        // Object Info: the instance's world origin, free from the TLAS transform. The CPU
        // fills HitRecord::object_origin from the translation of the very same matrix, so
        // both backends hash identical bits and a scattered rock keeps its color.
        vec3 hitObjOrigin = gl_ObjectToWorldEXT[3];
        // Object-space shading point for the procedural "Object Space" toggle: the BLAS
        // vertices ARE object space, so this is a barycentric blend of values already loaded
        // — no inverse transform. The CPU builds it the same way out of the mesh's P_orig.
        // Falls back to the world point when there is no vertex buffer, so a missing value
        // can never silently swap one space for the other.
        vec3 hitObjPos = (geo.vertexAddr != 0)
            ? (objV0 * bary.x + objV1 * bary.y + objV2 * bary.z)
            : hitPos;
        // gview: toward the viewer, i.e. the direction this ray CAME from. Fresnel /
        // Layer Weight are the only consumers; on a secondary bounce it is that bounce's
        // incoming direction, which is exactly what a path tracer should ask them about.
        vec3 hitView = normalize(-gl_WorldRayDirectionEXT);
        MatProgOut mp = evalMaterialProgram(mp_procOff, rawUV, hitPos, worldNormal, hitPointiness, hitObjOrigin,
                                            hitAttribs, hitObjPos, hitView,
                                            0.0, 0.0, 0.0, 0.0, vec3(0.0), hitPos,
                                            vec3(0.0), vec3(0.0), 0.0, hitObjPos, cam.waterTime);
        mpWritten = mp.written;
        if ((mp.written & MP_SLOT_BASECOLOR)        != 0u) albedo       = max(mp.baseColor, vec3(0.0));
        if ((mp.written & MP_SLOT_ROUGHNESS)        != 0u) roughness    = clamp(mp.roughness, 0.0, 1.0);
        if ((mp.written & MP_SLOT_METALLIC)         != 0u) metallic     = clamp(mp.metallic, 0.0, 1.0);
        if ((mp.written & MP_SLOT_SPECULAR)         != 0u) specular     = clamp(mp.specular, 0.0, 1.0);
        if ((mp.written & MP_SLOT_TRANSMISSION)     != 0u) transmission = clamp(mp.transmission, 0.0, 1.0);
        if ((mp.written & MP_SLOT_EMISSIONCOLOR)    != 0u) emColor      = mp.emissionColor;
        if ((mp.written & MP_SLOT_EMISSIONSTRENGTH) != 0u) emStrength   = max(mp.emissionStrength, 0.0);
        if ((mp.written & MP_SLOT_IOR)              != 0u) ior          = (mp.ior > 0.01) ? mp.ior : ior;
        if ((mp.written & MP_SLOT_NORMAL)           != 0u) {
            mpHasNormal = true;
            mpTangentNormal = mp.normal;
            mpNormalWorld = mp.normalWorld;
        }
    }

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

    // Sample albedo texture.
    // Skipped when the program owns the slot: a Mix Material blends BOTH sides'
    // textures per-pixel, and the fold still binds one of them to the slot (it has
    // only one to give). Letting that single bound texture land here would overwrite
    // the blend with A's or B's texture and hard-switch at Fac 0.5 — the exact
    // behaviour the per-pixel mix exists to replace. The CPU path already resolves
    // this the same way: applyProgramSurface runs AFTER the texture fetch and wins.
    int albedoTexID = int(mat.albedo_tex);
    if (albedoTexID > 0 && (mpWritten & MP_SLOT_BASECOLOR) == 0u) {
        albedo = texture(materialTextures[nonuniformEXT(albedoTexID)], materialUV).rgb;
    }
    
    // ----------------------------------------------------------------
    // OPACITY: resolved in the ANY-HIT now (shadow_anyhit.rahit camera-mode
    // branch). Camera/bounce/photon rays are traced WITHOUT OpaqueEXT, so
    // alpha-cutout candidates are stochastically ignored during traversal —
    // a hit that reaches this shader has already passed the alpha test and
    // must shade as opaque. The old in-closesthit stochastic pass-through
    // (emit BOUNCE_TRANSPARENT, re-trace from raygen) cost a full payload
    // round trip per foliage layer and is gone with it.
    // ----------------------------------------------------------------

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
        float bio  = (matx.bubble_ior > 1.0001) ? matx.bubble_ior : 1.33;
        float r0   = (1.0 - bio) / (1.0 + bio); r0 = r0 * r0;
        float fres = r0 + (1.0 - r0) * pow(1.0 - cosT, 5.0);
        vec3 dir, att;
        if (rnd(payload.seed) < fres) {
            dir = reflect(rayDir, Nb);                  // bright Fresnel rim
            if (matx.bubble_film > 1e-3) {
                float opd = matx.bubble_film * (1.0 / max(cosT, 0.15));
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
        payload.scatterDir          = dir;
        payload.attenuation        *= att;
        payload.scattered           = true;
        payload.bounceType          = BOUNCE_SPECULAR;
        // aerial parity (don't wash bubble): transmission=1 in the AOV pack (albedo.b stays 0)
        payload.primaryABT          = packHalf2x16(vec2(0.0, 1.0));
        return;
    }

    // Opaque pixel — continue to shading normally

    // --- MODE 2: glass/transmission adjustment ---
    if (mat.opacity < 0.99 && metallic < 0.1 && transmission < 0.01) {
        transmission = 1.0 - mat.opacity;
    }


   // Sample emission texture (skipped when the program drives Emission Color — see albedo)
int emissionTexID = int(mat.emission_tex);
if (emissionTexID > 0 && (mpWritten & MP_SLOT_EMISSIONCOLOR) == 0u) {
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
    
    // Sample transmission texture (for glass/transparent materials).
    // Skipped when the program drives Transmission — the CPU's applyProgramSurface
    // overwrites transmission after its own fetch, so the GPU has to yield too.
    int transmissionTexID = int(mat.transmission_tex);
    if (transmissionTexID > 0 && (mpWritten & MP_SLOT_TRANSMISSION) == 0u) {
        float trans = texture(materialTextures[nonuniformEXT(transmissionTexID)], materialUV).r;
        transmission = clamp(trans, 0.0, 1.0);
    }
    
    // Sample roughness texture (skipped when the program drives Roughness — see albedo)
    int roughTexID = int(mat.roughness_tex);
    if (roughTexID > 0 && (mpWritten & MP_SLOT_ROUGHNESS) == 0u) {
        float r = samplePackedRoughness(
            texture(materialTextures[nonuniformEXT(roughTexID)], materialUV), 0.0, mat.flags);
        roughness = clamp(r, 0.0, 1.0);
    }

    // Sample metallic texture (skipped when the program drives Metallic — see albedo)
    int metallicTexID = int(mat.metallic_tex);
    if (metallicTexID > 0 && (mpWritten & MP_SLOT_METALLIC) == 0u) {
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
    if (matx.micro_detail_strength > 0.0) {
        float sc  = max(matx.micro_detail_scale, 0.5);
        float str = matx.micro_detail_strength;

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
    bool isWaterMaterial = ((mat.flags & MAT_FLAG_WATER) != 0u) || matx.sheen > 0.001;
    bool waterUsesFFT = isWaterMaterial &&
                        ((mat.flags & MAT_FLAG_WATER_FFT_READY) != 0u) &&
                        mat.height_tex > 0u &&
                        mat.normal_tex > 0u &&
                        matx.fft_ocean_size > 0.001 &&
                        abs(matx.anisotropic) < 1e-5 &&
                        abs(matx.sheen_tint) < 1e-5;
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

    // Procedural bump (Bump node -> program Normal slot). mpTangentNormal is ALREADY
    // a tangent-space normal (-dh/du, -dh/dv, 1)*k with strength baked — no decode,
    // no re-scale. Same TBN transform as the texture path; mirrors CPU apply_normal_map.
    // EXCEPT when the program flagged it WORLD-space (Bevel): that normal is final as-is,
    // and pushing it through the UV tangent frame would twist it by the UV layout.
    if (mpHasNormal && !waterUsesFFT) {
        vec3 perturbed;
        if (mpNormalWorld) {
            perturbed = safeNormalize(mpTangentNormal, worldNormal);
        } else {
            vec3 tsN = normalize(mpTangentNormal);
            perturbed = normalize(
                surfaceTangent * tsN.x +
                surfaceBitangent * tsN.y +
                worldNormal * tsN.z
            );
        }
        if (dot(perturbed, -rayDir) > 0.0) tangentNormal = perturbed;
    }

    vec3 weatherSupportNormal = safeNormalize(mix(weatherMacroNormal, tangentNormal, 0.85), weatherMacroNormal);
    applyWeatherSurface(hitPos, tangentNormal, weatherSupportNormal, albedo, roughness, metallic);
    roughness = clamp(roughness, 0.0, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);

    const bool primarySurfacePending = (payload.primaryMeta & PL_PRIMARY_DONE) == 0u;
    if (primarySurfacePending) {
        payload.primaryARG  = packHalf2x16(albedo.rg);
        payload.primaryABT  = packHalf2x16(vec2(albedo.b, transmission));
        payload.primaryNrm  = plPackNormal(worldNormal);
        // Stylize AOV: real material boundary for outlines (16-bit id space)
        payload.primaryMeta = (payload.primaryMeta & PL_DISP_MASK)
                            | PL_PRIMARY_DONE | (matIndex & PL_MATID_MASK);
    }
    worldNormal = tangentNormal;
    worldNormal = weatherSurfaceNormal(hitPos, worldNormal, weatherSupportNormal);

    // ----------------------------------------------------------
    // IS_WATER fast path. Prefer the explicit material flag, keep sheen as legacy fallback.
    // Water has its own scatter: Gerstner waves + glass refraction.
    // Must run BEFORE transmission/direct-lighting/diffuse paths.
    // ----------------------------------------------------------
    if (isWaterMaterial) {
        vec3 waterFlowTangent = surfaceTangent;
        vec3 waterCrossTangent = surfaceBitangent;
        if ((mat.flags & MAT_FLAG_WATER_RIVER) != 0u && dot(hitWaterA.xy, hitWaterA.xy) > 1e-6) {
            vec3 objectFlow = normalize(vec3(hitWaterA.x, 0.0, hitWaterA.y));
            waterFlowTangent = normalize(mat3(gl_ObjectToWorldEXT) * objectFlow);
            waterFlowTangent = normalize(waterFlowTangent - worldNormal * dot(waterFlowTangent, worldNormal));
            waterCrossTangent = normalize(cross(worldNormal, waterFlowTangent));
        }
        scatterWater(
            hitPos, worldNormal, geomNormal, rayDir,
            ((mat.flags & MAT_FLAG_WATER_RIVER) != 0u) ? 2u :
            (((mat.flags & MAT_FLAG_WATER_LAKE) != 0u) ? 1u : 0u),
            rawUV, waterFlowTangent, waterCrossTangent,
            hitWaterA, hitWaterB, hitWaterC,
            /*wave_speed*/     matx.anisotropic,
            /*wave_strength*/  matx.sheen,
            /*wave_freq*/      matx.sheen_tint,
            /*foam_level*/     mat.translucent,
            /*foam_threshold*/ matx.foam_threshold,
            /*micro_strength*/ matx.micro_detail_strength,
            /*micro_scale*/    matx.micro_detail_scale,
            /*micro_anim*/     matx.micro_anim_speed,
            /*micro_morph*/    matx.micro_morph_speed,
            /*foam_noise*/     matx.foam_noise_scale,
            /*wind_dir*/       matx.fft_wind_direction,
            /*wind_speed*/     matx.fft_wind_speed,
            /*fft_time_scale*/ matx.fft_time_scale,
            /*fft_ocean_size*/ matx.fft_ocean_size,
            /*fft_height_tex*/ waterUsesFFT ? mat.height_tex : 0u,
            /*fft_normal_tex*/ waterUsesFFT ? mat.normal_tex : 0u,
            /*depth_max*/      mat.subsurface_amount * 100.0,
            /*absorption*/     matx.subsurface_scale,
            /*shore_dist*/     matx.subsurface_radius_r,
            /*shore_int*/      mat.clearcoat,
            /*caustic_int*/    mat.clearcoat_roughness,
            /*caustic_scale*/  matx.subsurface_radius_g,
            /*caustic_speed*/  matx.subsurface_anisotropy,
            /*shallow_color*/  vec3(mat.emission_r, mat.emission_g, mat.emission_b),
            /*deep_color*/     vec3(mat.albedo_r,   mat.albedo_g,   mat.albedo_b),
            /*ior*/            (mat.ior > 0.01) ? mat.ior : 1.333,
            /*roughness*/      clamp(mat.roughness, 0.0, 1.0),
            /*primary AOV*/    primarySurfacePending,
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
    vec3  resinColor = vec3(matx.resin_color_r, matx.resin_color_g, matx.resin_color_b);

    // Carried into the NEE block below so direct light reaching the base also gets
    // absorbed on its ENTRY path through the resin (at the light's angle).
    bool  resinActive  = false;
    vec3  resinExt     = vec3(0.0);
    float resinDensity = 0.0;

    // ── DEBUG VIEW 9: MEDIUM DENSITY ────────────────────────────────────────
    // Visualize the Interior Volume dust field: terminate every camera path at
    // the first hit and return the dust-coverage integral along the (refracted)
    // view ray — lobe gates and Fresnel are bypassed so the field itself is
    // shown, not its lit appearance. Materials without an interior return 0
    // (dark purple in the viridis ramp); opaque specks flash 1.0. Photon paths
    // die here too while the view is active — the grids reset on view exit.
    if (cam.debugView == 9u) {
        vec3 mdOut = vec3(0.0);
        bool hasInterior = (matx.transmission_density > 1e-4 ||
                            matx.resin_inclusion > 0.001 || matx.resin_dirt > 0.001 ||
                            matx.resin_shard > 0.001);
        if (hasInterior) {
            float effIor = max(ior, 1.45);
            vec3 Tdir = refract(rayDir, worldNormal, surfaceFrontFace ? (1.0 / effIor) : effIor);
            if (dot(Tdir, Tdir) < 1e-6) Tdir = rayDir;
            Tdir = normalize(Tdir);
            float thick = (matx.transmission_density > 1e-4) ? matx.transmission_density : 0.65;
            vec3 mOrg = hitPos, mDir = Tdir, mLit = worldNormal;
            if ((mat.flags & MAT_FLAG_RESIN_OBJ_SPACE) != 0u) {
                mOrg = gl_WorldToObjectEXT * vec4(hitPos, 1.0);
                mDir = normalize(mat3(gl_WorldToObjectEXT) * Tdir);
                mLit = normalize(mat3(gl_WorldToObjectEXT) * worldNormal);
            }
            ResinMarch rm = resinMarchInterior(
                mOrg, mDir, thick, vec3(0.0),
                matx.resin_inclusion, matx.resin_dirt,
                vec3(matx.resin_dirt_color_r, matx.resin_dirt_color_g, matx.resin_dirt_color_b),
                matx.resin_shard, matx.resin_shard_hue,
                vec3(0.85), mLit,
                max(matx.resin_inclusion_scale, 0.01),
                uint(matx.dust_style + 0.5),
                vec3(matx.dust_color_a_r, matx.dust_color_a_g, matx.dust_color_a_b),
                vec3(matx.dust_color_b_r, matx.dust_color_b_g, matx.dust_color_b_b),
                uint(matx.shard_shape + 0.5), payload.seed);
            mdOut = rm.dirtHit ? vec3(1.0) : vec3(clamp(rm.dustCover, 0.0, 1.0));
        }
        payload.radiance      = mdOut;
        payload.attenuation   = vec3(0.0);
        payload.scatterOrigin = hitPos;   // firstHitValid gate in raygen
        payload.scattered     = false;
        return;
    }

    // MODE BLEND: with Interior Depth active, Transmission is a CONTINUOUS
    // mix between the opaque resin coat and the see-through translucent
    // stone — the lobe is picked stochastically with probability =
    // transmission, so accumulation converges to t·stone + (1-t)·coat.
    // (The old hard threshold at 0.5 flipped the whole material between two
    // looks in one slider tick.) t=0 is the classic coat, t=1 pure stone
    // (amber/jade: real-distance Beer-Lambert on interior segments, photons
    // keep crossing → amber caustics); in between the stone body picks up an
    // increasingly opaque milky skin. Plain glass (no depth) is untouched.
    bool takeGlassLobe = (transmission > 0.01) && (rnd(payload.seed) < transmission);
    if (matx.transmission_density > 1e-4 && !takeGlassLobe) {
        // RESIN: a refractive ABSORBING layer over an OPAQUE base. Fresnel-split the
        // surface — the reflection lobe is the glossy resin top (specular, skips NEE);
        // light that enters reaches the base, which we tint by the coat absorption over
        // the thickness and shade as a normal diffuse surface, so the base gets full
        // direct lighting (NEE) + indirect (deeper, cleaner).
        float effIor = max(ior, 1.45);
        float cosT   = clamp(dot(-rayDir, worldNormal), 0.0, 1.0);
        float fres   = schlickFresnel(cosT, effIor);
        // Coat gloss is the resin LAYER's own roughness, independent of the base.
        float resinRough = clamp(matx.resin_roughness, 0.0, 1.0);
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
        // Physical absorption: per-channel coefficient ∝ (1 - tint), so a warm
        // tint passes its own hue and swallows the complement (amber glows red,
        // kills blue) instead of darkening everything. The old flat +0.25 base
        // extinction blackened even a WHITE interior at depth — the base term
        // now scales with tint darkness only (clear tint = pure lensing).
        float ctMax   = max(ct.r, max(ct.g, ct.b));
        vec3  ext     = (vec3(1.0) - ct) * 1.35 + vec3(0.22 * (1.0 - ctMax));

        // --- Resin INTERNAL inclusions (Phase 1) -----------------------------------
        // March the refracted ray through the resin thickness (no scene rays — pure
        // procedural sampling): dust = heterogeneous absorption at depth, dirt = opaque
        // worley specks that terminate early (their colour shows through the resin
        // already crossed), and the refracted lateral travel offsets the base lookup
        // (parallax).
        vec3 Tdir = refract(rayDir, worldNormal, 1.0 / effIor);
        if (dot(Tdir, Tdir) < 1e-6) Tdir = rayDir;        // total internal reflection fallback
        Tdir = normalize(Tdir);

        // Reached the base: parallax-offset the base lookup along the refracted
        // lateral travel. Always applied when resin layer is active.
        vec3 inPlane = Tdir - worldNormal * dot(Tdir, worldNormal);
        vec2 parUV = materialUV
                   + vec2(dot(inPlane, surfaceTangent), dot(inPlane, surfaceBitangent))
                     * (matx.transmission_density * 0.05);
        if (albedoTexID > 0) {
            albedo = texture(materialTextures[nonuniformEXT(albedoTexID)], parUV).rgb;
        }

        bool resinHasInclusions = (matx.resin_inclusion > 0.001 || matx.resin_dirt > 0.001 ||
                                   matx.resin_shard > 0.001);
        if (resinHasInclusions) {
            // Sample one light direction at the surface for the interior march
            // (cheap NEE-direction shading of the specks; no shadow rays).
            vec3 resinLightDir = worldNormal;
            {
                float plsel; int li = pick_smart_light_gl(uvec2(0), hitPos, plsel);
                if (li >= 0) {
                    vec3 wi_; float d_; float a_;
                    if (sample_light_direction_gl(lights.l[li], hitPos,
                                                  rnd(payload.seed), rnd(payload.seed),
                                                  wi_, d_, a_) && dot(wi_, wi_) > 1e-8) {
                        resinLightDir = normalize(wi_);
                    }
                }
            }
            // Anchor: object space marches the fields in the mesh's local frame
            // (interior travels with the object); world space leaves the pattern
            // fixed in space (a deliberate effect sometimes — e.g. moving through
            // a "frozen" medium). The light direction rotates into the same frame
            // so speck shading stays consistent.
            vec3 mOrg = hitPos, mDir = Tdir, mLit = resinLightDir;
            if ((mat.flags & MAT_FLAG_RESIN_OBJ_SPACE) != 0u) {
                mOrg = gl_WorldToObjectEXT * vec4(hitPos, 1.0);
                mDir = normalize(mat3(gl_WorldToObjectEXT) * Tdir);
                mLit = normalize(mat3(gl_WorldToObjectEXT) * resinLightDir);
            }
            ResinMarch rm = resinMarchInterior(
                mOrg, mDir, matx.transmission_density, ext,
                matx.resin_inclusion, matx.resin_dirt,
                vec3(matx.resin_dirt_color_r, matx.resin_dirt_color_g, matx.resin_dirt_color_b),
                matx.resin_shard, matx.resin_shard_hue,
                clamp(ct * 0.5 + vec3(0.45), 0.0, 1.0),   // dust base tint from resin colour
                mLit,
                max(matx.resin_inclusion_scale, 0.01),
                uint(matx.dust_style + 0.5),
                vec3(matx.dust_color_a_r, matx.dust_color_a_g, matx.dust_color_a_b),
                vec3(matx.dust_color_b_r, matx.dust_color_b_g, matx.dust_color_b_b),
                uint(matx.shard_shape + 0.5), payload.seed);
            if (rm.dirtHit) {
                // Terminate on the speck: light-direction-shaded colour, dimmed
                // by the resin crossed.
                albedo = rm.dirtAlbedo;
            } else {
                // Milky nebula wisps are VISIBLE (mixed toward the marched
                // colour), not just extra darkening; shards contribute their
                // own colour body (visible even over a dark base) and
                // bubble/shard rims sparkle.
                albedo = mix(albedo * rm.absorb, rm.dustTint * rm.absorb, rm.dustCover);
                albedo = clamp(albedo + rm.shardGlow + rm.dustGlow + vec3(rm.sparkle), 0.0, 1.0);
            }
        } else {
            float pathLen = 2.0 * matx.transmission_density / cosV;
            albedo       *= exp(-pathLen * ext);
        }
        roughness     = 1.0;
        metallic      = 0.0;
        transmission  = 0.0;
        // Hand the absorption to the NEE block so direct light entering the resin is
        // also attenuated by its own (light-angle) path length, not just the albedo tint.
        resinActive   = true;
        resinExt      = ext;
        resinDensity  = matx.transmission_density;
        // (no return — direct lighting + diffuse BRDF below shade the tinted base)
    }
    else if (takeGlassLobe) {
        {
            // NOTE: glass-marble FULL VOLUME (real-interior medium march, MAT_FLAG_MARBLE_VOLUME)
            // was disabled — it was too camera-angle dependent and the interior dust/dirt never
            // read as intended. The flag + serialize fields are kept dormant (saved scenes load
            // fine); inclusion-bearing glass now always uses the shell march below.
            bool inclusionsOn = (matx.resin_inclusion > 0.001 || matx.resin_dirt > 0.001 ||
                                 matx.resin_shard > 0.001);
            // TRANSLUCENT STONE (amber/jade): Interior Depth on a transmissive
            // body = REAL-DISTANCE Beer-Lambert. gl_HitTEXT is the actual
            // length of the segment that just arrived at this hit; it only ran
            // through the interior when the hit is a back face (ray inside), so
            // entry hits absorb nothing. Each internal segment (exit legs, TIR
            // bounces) absorbs its own true length — thick centres deepen,
            // thin edges stay clear, photons keep crossing (amber caustics).
            // Applied DIRECTLY to the path throughput, not via scatterGlass's
            // albedo parameter: scatterGlass remaps its albedo through a FIXED
            // optical thickness (exp(-(1-tint)*0.65/cos)) — folding the real
            // Beer-Lambert factor in there crushed it to a faint constant tint
            // (even a fully absorbed channel survived at ~0.52 per interface)
            // and the reflect/TIR lobe dropped it entirely. The segment behind
            // this hit was already traversed — its absorption is unconditional,
            // whatever lobe the ray takes next.
            if (matx.transmission_density > 1e-4 && !surfaceFrontFace) {
                vec3  sct   = clamp(vec3(matx.resin_color_r, matx.resin_color_g, matx.resin_color_b),
                                    vec3(0.0), vec3(1.0));
                float sMax  = max(sct.r, max(sct.g, sct.b));
                vec3  sExt  = (vec3(1.0) - sct) * 1.35 + vec3(0.22 * (1.0 - sMax));
                payload.attenuation *= exp(-gl_HitTEXT * sExt * matx.transmission_density);
            }
            // STONE COLOUR MODEL: with Interior Depth active, the transmitted
            // colour comes from the REAL-DISTANCE absorption alone — the
            // surface albedo tint (the hack for depthless coloured glass)
            // fades out. Otherwise a saturated Base Color pre-kills exactly
            // the channels the depth gradient needs: a pure green albedo made
            // Depth look like it did nothing (same image at 0 and 8).
            vec3 glassBase = mix(albedo, vec3(1.0),
                                 clamp(matx.transmission_density * 10.0, 0.0, 1.0));
            // GLASS MARBLE (shell): when inclusions are enabled on a GLASS base, march the
            // refracted ray through the interior — dust (haze) + dirt specks (opaque
            // early-return) — BEFORE refracting through, so light still passes through
            // (real see-through glass) but picks up volumetric internal structure.
            // Independent of the resin coat (that path forces an opaque base). No extra
            // scene rays: the march is procedural; scatterGlass does the real refraction.
            if (inclusionsOn) {
                vec3 Tg = refract(rayDir, worldNormal, surfaceFrontFace ? (1.0 / ior) : ior);
                // TIR'da interior: refract() returns the zero vector when the
                // inside->outside angle is past critical — the continuation is
                // then CERTAIN to be the internal reflection, so march the
                // interior along it. The structure used to vanish exactly in
                // the marble's mirror zones because the march followed an exit
                // refraction that does not exist there. (At non-critical exit
                // angles the Fresnel lobe choice is probabilistic and the
                // legacy exit-side march approximation is kept.)
                bool tirCertain = dot(Tg, Tg) < 1e-6;
                Tg = tirCertain ? reflect(rayDir, worldNormal) : normalize(Tg);
                float cosIn = max(abs(dot(Tg, -worldNormal)), 0.05);
                vec3 marbleLightDir = worldNormal;
                {
                    float plsel; int li = pick_smart_light_gl(uvec2(0), hitPos, plsel);
                    if (li >= 0) {
                        vec3 wi_; float d_; float a_;
                        if (sample_light_direction_gl(lights.l[li], hitPos,
                                                      rnd(payload.seed), rnd(payload.seed),
                                                      wi_, d_, a_) && dot(wi_, wi_) > 1e-8) {
                            marbleLightDir = normalize(wi_);
                        }
                    }
                }
                vec3 gOrg = hitPos, gDir = Tg, gLit = marbleLightDir;
                if ((mat.flags & MAT_FLAG_RESIN_OBJ_SPACE) != 0u) {
                    gOrg = gl_WorldToObjectEXT * vec4(hitPos, 1.0);
                    gDir = normalize(mat3(gl_WorldToObjectEXT) * Tg);
                    gLit = normalize(mat3(gl_WorldToObjectEXT) * marbleLightDir);
                }
                ResinMarch rm = resinMarchInterior(
                    gOrg, gDir, 0.65 / cosIn,                   // matches scatterGlass thickness model
                    vec3(0.0),                                  // clear glass: dust is the only extinction
                    matx.resin_inclusion, matx.resin_dirt,
                    vec3(matx.resin_dirt_color_r, matx.resin_dirt_color_g, matx.resin_dirt_color_b),
                    matx.resin_shard, matx.resin_shard_hue,
                    vec3(0.85),                                 // neutral milky dust in clear glass
                    gLit,
                    max(matx.resin_inclusion_scale, 0.01),
                    uint(matx.dust_style + 0.5),
                    vec3(matx.dust_color_a_r, matx.dust_color_a_g, matx.dust_color_a_b),
                    vec3(matx.dust_color_b_r, matx.dust_color_b_g, matx.dust_color_b_b),
                    uint(matx.shard_shape + 0.5), payload.seed);
                if (rm.dirtHit) {
                    // Opaque speck suspended in the glass: baked-lit micro-sphere
                    // colour (stone depth already in the throughput) →
                    // fall through to NEE.
                    albedo = rm.dirtAlbedo;
                    roughness = 1.0; metallic = 0.0; transmission = 0.0;
                    // (no return — direct lighting + diffuse BRDF below shade the speck)
                } else {
                    // Hazy glass: nebula dust whitens/tints (milky scatter
                    // approximation), shards carry their own colour body,
                    // bubble/shard rims sparkle; refract through.
                    vec3 gal = mix(glassBase * rm.absorb, rm.dustTint * rm.absorb, rm.dustCover * 0.8);
                    gal = clamp(gal + rm.shardGlow + rm.dustGlow + vec3(rm.sparkle), 0.0, 1.0);
                    scatterGlass(hitPos, worldNormal, worldNormal, surfaceFrontFace, rayDir, gal, ior, roughness, 0.0, vec3(1.0), mat.dispersion, payload.seed);
                    return;
                }
            } else {
                // Chosen transmission path - act as Glass (with stone depth
                // absorption when Interior Depth is set on a transmissive body)
                scatterGlass(hitPos, worldNormal, worldNormal, surfaceFrontFace, rayDir, glassBase, ior, roughness, 0.0, vec3(1.0), mat.dispersion, payload.seed);
                return; // Immediately return, skipping direct lighting (Next Event Estimation)
            }
        }
    }
    else if (transmission > 0.01) {
        // Chosen base path (diffuse/metal), compensate probability weight
        payload.attenuation *= (1.0 / max(1.0 - transmission, 0.01));
        transmission = 0.0;
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
                        // Conservative init: w=0 (blocked). shadow_miss.rmiss (missIndex=1) sets w=1 on escape.
                        // any-hit transmissive → rgb *= tint + ignoreIntersection (coloured glass shadow)
                        // any-hit with terminateRayEXT → w stays 0 (opaque shadow)
                        // SkipClosestHit: geometry hit without opacity test → w stays 0 (solid shadow)
                        shadowPayload = vec4(1.0, 1.0, 1.0, 0.0);
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
                            // missIndex=1 → shadow_miss.rmiss sets shadowPayload.w=1 when ray escapes
                            uint shadowFlags = gl_RayFlagsTerminateOnFirstHitEXT
                                             | gl_RayFlagsSkipClosestHitShaderEXT;
                            // mask 0x01 = triangles only — volume AABBs have mask 0x02 so
                            // they are invisible to shadow rays and cannot cast hard shadows.
                            traceRayEXT(topLevelAS, shadowFlags, 0x01, 0, 1, 1, shadowOrigin, tmin, wi, tmax, 1);
                        }
                        vec3 shadowVisibility = (shadowPayload.w > 0.5) ? shadowPayload.rgb : vec3(0.0);
                        if (any(greaterThan(shadowVisibility, vec3(1e-4)))) {
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
            shadowPayload = vec4(1.0, 1.0, 1.0, 0.0);
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
            float sunShadowVisibility = (shadowPayload.w > 0.5) ? 1.0 : 0.0;
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
    vec3  subsurfaceColor    = max(vec3(matx.subsurface_r, matx.subsurface_g, matx.subsurface_b), vec3(0.001));
    vec3  subsurfaceRadius   = max(vec3(matx.subsurface_radius_r, matx.subsurface_radius_g, matx.subsurface_radius_b), vec3(0.001));
    float subsurfaceScale    = max(matx.subsurface_scale, 0.001);
    float subsurfaceAniso    = clamp(matx.subsurface_anisotropy, -0.99, 0.99);

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
                             matx.clearcoat_iridescence, matx.clearcoat_film_thickness, payload.seed);
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

