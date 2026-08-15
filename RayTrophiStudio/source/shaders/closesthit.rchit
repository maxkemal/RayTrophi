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
const float EPSILON     = 1e-4;
const float OPACITY_THRESHOLD = 0.5;  // Alpha cutout threshold
const uint MAT_FLAG_WATER = (1u << 17);
const uint MAT_FLAG_WATER_FFT_READY = (1u << 18);
const uint MAT_FLAG_WATER_LAKE = (1u << 22);
const uint MAT_FLAG_WATER_RIVER = (1u << 23);
// MAT_FLAG_BUBBLE moved to rt_payload.glsl (included above): the fluid
// isosurface shader needs the same bit, and two copies is how they drift.
const uint MAT_FLAG_MARBLE_VOLUME = (1u << 20); // glass marble full-volume medium march (raygen integrates interior)

// ============================================================
// Payload — shared ABI, single source of truth
// ============================================================
#include "rt_payload.glsl"
#include "volume_instrumentation.glsl"

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
    // Material State Field mask (burn/heat) for this instance; 0 = none.
    // KEEP BYTE-IDENTICAL with VulkanRT::VkInstanceData and with the copies in
    // the other shaders reading binding 5 — a divergence reads as garbage.
    uint msfCharTex;
    uint msfCharPacked;   // char_color RGB (low 24), molten emission (high 8)
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
    // Appended acceleration block — MUST match VkVolumeInstance in
    // include/Backend/vulkan_volume_types.h (576 bytes). Every shader that
    // declares this struct carries the same tail: the SSBO stride is
    // per-declaration, so one stale copy shifts every instance after the first.
    uint64_t majorant_address;   // per-block density max for live dense gas (0 = none)
    float    majorant_dim[3];    // block-grid resolution
    float    majorant_block;     // cells per block edge
    uint64_t flame_address;      // combustion reaction field for live dense gas (0 = none)
    uint64_t emissive_list_address; // [0]=count, [1..]=emitting block indices
    float    emissive_capacity;
    float    iso_material_index;  // SDF isosurface material, 1-based (0 = none)
    // FULLY CLAIMED: [0]=pore amount, [1]=world units per pore cell,
    // [2]=pore size variation, [3]=coordinate space (0=Material, 1=Domain,
    // 2=World — see materialAnchor below / in volume_closesthit.rchit).
    float    _accel_reserved[4];
    // Material coordinate (UVW) RESIDUAL grid: dense xyz triples at sim-grid
    // resolution holding (uvw - cell centre). Consumed only in
    // volume_closesthit.rchit; mirrored here because the SSBO stride is
    // per-declaration. 0 = not published.
    uint64_t uvw_residual_address;
    // Same grid/origin/voxel as the residual field; 0 = not published.
    // Slotted beside the other address so the struct stays 624 bytes.
    uint64_t composition_address;
    float    uvw_dim[3];
    float    uvw_origin[3];
    float    uvw_voxel;
    // Explicit tail padding, mirroring vulkan_volume_types.h exactly. C++ pads
    // the alignas(16) struct to 608; scalar layout would stop at 600. Implicit
    // padding is where the two are free to disagree, and a stride mismatch does
    // not fail loudly — it reads the NEXT volume's fields as this one's.
    float    _uvw_pad[1];
};

layout(set = 0, binding = 9, scalar) readonly buffer VolumeBuffer { VkVolumeInstance v[]; } volumes;

// Live Vulkan gas density is a device-addressable dense float grid. Surface
// shadows must sample it too; otherwise source type 5 falls through to the
// default homogeneous density and the complete domain becomes a shadow box.
layout(buffer_reference, std430, buffer_reference_align = 4)
readonly buffer ChDenseGasFloatGrid {
    float values[];
};

float ch_sampleDenseGasFloat(uint64_t address, VkVolumeInstance vol, vec3 localPos) {
    if (address == 0) return 0.0;
    ivec3 resolution = ivec3(
        int(vol._ext_reserved[0] + 0.5),
        int(vol._ext_reserved[1] + 0.5),
        int(vol._ext_reserved[2] + 0.5));
    if (any(lessThanEqual(resolution, ivec3(0)))) return 0.0;

    float voxelSize = max(vol.voxel_size, 1e-6);
    vec3 gridOrigin = vec3(vol._ext_reserved[3], vol._ext_reserved[4], vol._ext_reserved[5]);
    vec3 gridPos = (localPos - gridOrigin) / voxelSize - vec3(0.5);
    if (any(lessThan(gridPos, vec3(-0.5))) ||
        any(greaterThan(gridPos, vec3(resolution) - vec3(0.5)))) return 0.0;

    ivec3 p0 = clamp(ivec3(floor(gridPos)), ivec3(0), resolution - ivec3(1));
    ivec3 p1 = min(p0 + ivec3(1), resolution - ivec3(1));
    vec3 f = clamp(gridPos - vec3(p0), vec3(0.0), vec3(1.0));
    ChDenseGasFloatGrid grid = ChDenseGasFloatGrid(address);
    int xy = resolution.x * resolution.y;
    int i000 = p0.x + p0.y * resolution.x + p0.z * xy;
    int i100 = p1.x + p0.y * resolution.x + p0.z * xy;
    int i010 = p0.x + p1.y * resolution.x + p0.z * xy;
    int i110 = p1.x + p1.y * resolution.x + p0.z * xy;
    int i001 = p0.x + p0.y * resolution.x + p1.z * xy;
    int i101 = p1.x + p0.y * resolution.x + p1.z * xy;
    int i011 = p0.x + p1.y * resolution.x + p1.z * xy;
    int i111 = p1.x + p1.y * resolution.x + p1.z * xy;
    float z0 = mix(mix(grid.values[i000], grid.values[i100], f.x),
                   mix(grid.values[i010], grid.values[i110], f.x), f.y);
    float z1 = mix(mix(grid.values[i001], grid.values[i101], f.x),
                   mix(grid.values[i011], grid.values[i111], f.x), f.y);
    return mix(z0, z1, f.z);
}

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
    } else if (vol.volume_type == 4 && vol.source_type == 5) {
        density = ch_sampleDenseGasFloat(vol.vdb_grid_address, vol, lp);
    }
    density = max((density - vol.density_remap_low) /
                  max(vol.density_remap_high - vol.density_remap_low, 1e-6), 0.0);
    float densityCutoff = max(vol._reserved[0], 0.0);
    if (density <= densityCutoff) return 0.0;
    float cutoffFade = densityCutoff > 0.0
        ? smoothstep(densityCutoff, densityCutoff * 2.0, density)
        : 1.0;
    return max(density * vol.density_multiplier * cutoffFade, 0.0);
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

        float physTrans = exp(-opticalDepth);
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

// Principled scatter lobes, light sampling and the water/resin helpers now
// live in one shared module so this shader and volume_closesthit.rchit
// evaluate the SAME BSDF. Two copies of sampleHG already existed here and
// in the volume shader; that drift is exactly what this removes.
#include "bsdf_scatter.glsl"


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


// ============================================================
// Material Scatter Fonksiyonları
// ============================================================



















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



// ============================================================
// Main — Closest Hit Entry Point
// ============================================================
void main() {
    payload.skipGasVolumes = false;
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
        bool primaryVolumeInteraction = false;

        if (!surfaceFrontFace) {
            float segmentLength = max(gl_HitTEXT, 0.0);
            uint volumeProgram = matProgramOffset(matIndex);
            int maxSteps = clamp(int(matx.volume_max_steps + 0.5), 1, 256);
            float stepLength = max(matx.volume_step_size, segmentLength / float(maxSteps));
            int stepCount = min(maxSteps, max(1, int(ceil(segmentLength / max(stepLength, 1e-4)))));
            stepLength = segmentLength / float(stepCount);
            float transmittance = 1.0;
            vec3 integratedRadiance = vec3(0.0);
            uint measuredDensitySamples = 0u;
            uint measuredShadowSamples = 0u;

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
                measuredDensitySamples++;
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
                        float shadowTauHint =
                            density * (scatterStrength + absorptionStrength) * shadowLength;
                        int effectiveShadowSteps = clamp(
                            int(ceil(float(shadowSteps) *
                                     clamp(sqrt(max(shadowTauHint, 0.0)), 0.20, 1.0))),
                            min(2, shadowSteps), shadowSteps);
                        float shadowStep = shadowLength / float(effectiveShadowSteps);
                        float shadowTau = 0.0;
                        for (int sj = 1; sj <= effectiveShadowSteps; ++sj) {
                            measuredShadowSamples++;
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
            volumeRecordShadowSamples(measuredShadowSamples);
            uint marchOutcome = transmittance <= 0.002
                ? VOLUME_MARCH_EXTINCTION
                : VOLUME_MARCH_COMPLETED;
            volumeRecordRay(
                measuredDensitySamples, 0u, 0u, 0u, marchOutcome);

            // Match the VDB primary-hit contract: an empty/near-empty medium is
            // a transparent pass, not geometry at the AABB/mesh boundary.
            // Recording the front face with transmission=1 made raygen multiply
            // aerial distance by zero, so background seen through an empty
            // volume skipped atmosphere and exposed the enclosing mesh as a
            // ghost box. Publish only a substantive optical interaction and
            // carry the measured segment transmittance.
            float volumeContribution = max(max(integratedRadiance.r,
                                               integratedRadiance.g),
                                           integratedRadiance.b);
            float volumeOpacity = 1.0 - transmittance;
            primaryVolumeInteraction =
                volumeOpacity > 0.04 || volumeContribution > 5e-4;
            if ((payload.primaryMeta & PL_PRIMARY_DONE) == 0u &&
                primaryVolumeInteraction) {
                payload.primaryARG = packHalf2x16(scatterColor.rg);
                payload.primaryABT = packHalf2x16(vec2(scatterColor.b, transmittance));
                payload.primaryNrm = plPackNormal(-rayDir);
                payload.primaryMeta = (payload.primaryMeta & PL_DISP_MASK)
                                    | (matIndex & PL_MATID_MASK)
                                    | PL_PRIMARY_DONE | PL_PRIMARY_VOLUME;
            }

            payload.radiance += integratedRadiance;
            payload.attenuation *= vec3(transmittance);
        }

        payload.scatterOrigin = hitPos + rayDir * 0.002;
        payload.scatterDir = rayDir;
        payload.scattered = true;
        payload.skipAABBs = false;
        // Only an optically empty interval is a free transparent pass. A real
        // medium interaction must be visible to raygen's first-hit/aerial logic.
        if (!primaryVolumeInteraction) {
            payload.bounceType = BOUNCE_TRANSPARENT;
        }
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

    // ── Material State Field: burn / heat ─────────────────────────────────────
    // Per-INSTANCE, so two objects sharing a material char independently. The
    // mask is R=char, G=absolute temperature, B=mass loss, A=integrity.
    // Applied AFTER the base-colour texture and after the
    // material program, because burning is a state of the surface, not another
    // material input — a charred plank is black whatever its texture says.
    //
    // ★ Sampled with rawUV, NOT hitUV and NOT materialUV.
    //   - materialUV would apply the material's scale/offset and slide the burn
    //     mark off the part that actually burned.
    //   - hitUV is already V-FLIPPED for Vulkan (see rawUV above). The mask is
    //     rasterized on the host in the mesh's own, unflipped UV space, so
    //     sampling with hitUV mirrors the mark vertically — it lands on the
    //     opposite side of every UV island from where the surface really burned.
    if (inst.msfCharTex > 0u) {
        vec4 msf = texture(materialTextures[nonuniformEXT(int(inst.msfCharTex))], rawUV);
        float charAmt = clamp(msf.x, 0.0, 1.0);
        float heat    = clamp(msf.y, 0.0, 1.0);
        float massLoss = clamp(msf.z, 0.0, 1.0);
        uint msfFlags = (inst.msfCharPacked >> 24) & 0xC0u;

        if (charAmt > 0.0) {
            vec3 charColor = vec3(float((inst.msfCharPacked      ) & 0xFFu),
                                  float((inst.msfCharPacked >>  8) & 0xFFu),
                                  float((inst.msfCharPacked >> 16) & 0xFFu)) * (1.0 / 255.0);
            albedo = mix(albedo, charColor, charAmt);
            // Char is porous soot: it scatters wide and kills specular. Without
            // this a "burnt" surface keeps a clean highlight and reads as black
            // paint rather than as charcoal.
            roughness = mix(roughness, 1.0, charAmt * 0.85);
            metallic  = mix(metallic, 0.0, charAmt);
        }

        // Wood keeps its topology until a later fracture phase, but its damaged
        // surface can still read as split fibres: two deterministic UV ridges
        // expose a pale/black crack edge as integrity falls. Paper holes are
        // handled in any-hit; shade their surviving rim here.
        if ((msfFlags & 0x40u) != 0u && massLoss > 0.0) {
            float grain = abs(sin(rawUV.x * 173.0 + sin(rawUV.y * 31.0) * 2.4));
            float crack = smoothstep(0.985 - massLoss * 0.18, 1.0, grain) * massLoss;
            albedo = mix(albedo, vec3(0.018, 0.012, 0.008), crack);
            roughness = mix(roughness, 1.0, crack);
        } else if ((msfFlags & 0x80u) != 0u && massLoss > 0.0) {
            float rim = smoothstep(0.25, 0.75, massLoss) * (1.0 - smoothstep(0.82, 1.0, massLoss));
            albedo = mix(albedo, vec3(0.025, 0.012, 0.006), rim);
            roughness = mix(roughness, 1.0, rim);
        }

        // Blackbody glow, thresholded in ABSOLUTE KELVIN.
        //
        // msf.y carries temperature quantized against a fixed 3000 K range
        // (MaterialStateField::kMaskKelvinRange) — NOT against the domain's
        // max_temperature. Keep the two constants in sync.
        //
        // The threshold is the Draper point (~798 K), the temperature at which a
        // heated surface first becomes visibly red. It is a real physical
        // constant, which is the point: a normalized threshold made glow depend
        // on a solver slider, and at the default ceiling a genuinely hot 717 K
        // surface quantized below it and never lit up at all.
        const float kMaskKelvinRange = 3000.0;
        const float kDraperKelvin    = 798.0;
        float kelvin = heat * kMaskKelvinRange;
        float moltenScale = float((inst.msfCharPacked >> 24) & 0x3Fu) * (1.0 / 8.0);
        if (moltenScale > 0.0 && kelvin > kDraperKelvin) {
            // Normalize the visible incandescent band: Draper -> 2400 K covers
            // dull red through white-hot, which is the whole range that reads.
            float g = clamp((kelvin - kDraperKelvin) / (2400.0 - kDraperKelvin),
                            0.0, 1.0);
            g = g * g * g;
            // Red -> orange -> white-hot, matching the volume blackbody ramp.
            vec3 glow = mix(vec3(1.0, 0.16, 0.02),
                            mix(vec3(1.0, 0.55, 0.10), vec3(1.0, 0.95, 0.85),
                                clamp((g - 0.5) * 2.0, 0.0, 1.0)),
                            clamp(g * 2.0, 0.0, 1.0));
            emColor = mix(emColor, glow, clamp(g, 0.0, 1.0));
            emStrength = max(emStrength, g * moltenScale);
        }
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
        bool passThrough = false;
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
            passThrough = true;
        }
        payload.radiance            = vec3(0.0);
        payload.scatterDir          = dir;
        payload.attenuation        *= att;
        payload.scattered           = true;
        // ★ A straight-through crossing is not a scattering event and must not
        // spend a GI bounce. raygen's isTransparentPass frees a pass only when
        // the direction is unchanged AND (attenuation is exactly 1 OR the tag is
        // BOUNCE_TRANSPARENT); the film always tints slightly, so tagging both
        // lobes SPECULAR meant every crossing cost a bounce.
        //
        // Found on the fluid isosurface, where a foam shell is crossed many
        // times over and the budget ran out before the ray reached what was
        // behind it (black patches). A mesh bubble only has two crossings, so
        // here it was merely wasteful rather than visibly broken — but it is the
        // same defect, and stacked bubbles (champagne, soap foam) pay it too.
        payload.bounceType          = passThrough ? BOUNCE_TRANSPARENT
                                                  : BOUNCE_SPECULAR;
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
                            // Authored triangles and transient splats cast hard
                            // shadows; volume AABBs remain on soft transmittance.
                            traceRayEXT(topLevelAS, shadowFlags, RT_MASK_DIRECT_SHADOW,
                                        0, 1, 1, shadowOrigin, tmin, wi, tmax, 1);
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
            // Authored triangles + transient splats; volume AABBs are handled
            // separately by volumetric transmittance.
            traceRayEXT(topLevelAS, sunShadowFlags, RT_MASK_DIRECT_SHADOW, 0, 1, 1,
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
    // Read Principled BSDF extended parameters, then hand the RESOLVED surface
    // to the shared lobe selection. The selection itself moved to
    // bsdf_scatter.glsl so volume_closesthit.rchit runs the identical rules on
    // the fluid isosurface - one BSDF, not two that drift.
    // ----------------------------------------------------------
    SurfaceSample ss = defaultSurfaceSample();
    ss.P = hitPos; ss.N = worldNormal; ss.rayDir = rayDir;
    ss.albedo = albedo; ss.roughness = roughness;
    ss.metallic = metallic; ss.specular = specular;
    ss.clearcoat          = clamp(mat.clearcoat, 0.0, 1.0);
    ss.clearcoatRoughness = clamp(mat.clearcoat_roughness, 0.001, 1.0);
    if (weatherSurfaceActive() && worldData.w.weatherType == 1) {
        float wet = clamp(worldData.w.weatherSurfaceWetness, 0.0, 1.0);
        ss.clearcoat = max(ss.clearcoat, wet * 0.72);
        ss.clearcoatRoughness = min(ss.clearcoatRoughness, max(0.006, 0.045 - wet * 0.030));
    }
    ss.clearcoatIridescence   = matx.clearcoat_iridescence;
    ss.clearcoatFilmThickness = matx.clearcoat_film_thickness;
    ss.translucent        = clamp(mat.translucent, 0.0, 1.0);
    ss.subsurfaceAmount   = clamp(mat.subsurface_amount, 0.0, 1.0);
    ss.subsurfaceColor    = max(vec3(matx.subsurface_r, matx.subsurface_g, matx.subsurface_b), vec3(0.001));
    ss.subsurfaceRadius   = max(vec3(matx.subsurface_radius_r, matx.subsurface_radius_g, matx.subsurface_radius_b), vec3(0.001));
    ss.subsurfaceScale    = max(matx.subsurface_scale, 0.001);
    ss.subsurfaceAnisotropy = clamp(matx.subsurface_anisotropy, -0.99, 0.99);
    scatterPrincipled(ss, payload.seed);

    // Resin base scattered as a normal diffuse lobe above (which set BOUNCE_DIFFUSE).
    // Re-tag it BOUNCE_RESIN so raygen counts it against the small dedicated resin
    // budget instead of the global diffuse budget — bounds resin GI cost (TDR fix).
    if (resinActive && payload.scattered) {
        payload.bounceType = BOUNCE_RESIN;
    }
}

