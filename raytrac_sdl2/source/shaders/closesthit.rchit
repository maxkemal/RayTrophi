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
const float OPACITY_THRESHOLD = 0.5;  // Alpha cutout threshold

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
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
// Separate shadow payload storage to avoid corrupting the main payload during shadow tracing
// Use rayPayloadInEXT here to match any-hit/miss declarations (avoid ABI mismatch)
layout(location = 1) rayPayloadEXT bool shadowOccluded;

// ============================================================
// Descriptor Bindings
// ============================================================
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

struct Material {
    // Block 1: Albedo + opacity
    float albedo_r, albedo_g, albedo_b, opacity;
    // Block 2: Emission + strength
    float emission_r, emission_g, emission_b, emission_strength;
    // Block 3: PBR properties
    float roughness, metallic, ior, transmission;
    // Block 4: Subsurface color + amount
    float subsurface_r, subsurface_g, subsurface_b, subsurface_amount;
    // Block 5: Subsurface radius + scale
    float subsurface_radius_r, subsurface_radius_g, subsurface_radius_b, subsurface_scale;
    // Block 6: Coatings & Translucency
    float clearcoat, clearcoat_roughness, translucent, subsurface_anisotropy;
    // Block 7: Additional properties
    float anisotropic, sheen, sheen_tint;
    uint flags;
    // Block 8: Water/Extra params
    float fft_amplitude, fft_time_scale, micro_detail_strength, micro_detail_scale;
    // Block 9: Extra water params
    float foam_threshold, fft_ocean_size, fft_choppiness, fft_wind_speed;
    // Block 10: Standard Textures (first 4)
    uint albedo_tex;
    uint normal_tex;
    uint roughness_tex;
    uint metallic_tex;
    // Block 11: Standard Textures (second 4)
    uint emission_tex;
    uint height_tex;
    uint opacity_tex;
    uint transmission_tex;
    // Block 12: Reserved
    float subsurface_ior;
    uint _terrain_layer_idx; // terrain layer buffer index (valid when FLAG_TERRAIN set)
    uint _reserved_2, _reserved_3;
};

struct LightData {
    vec4 position;    // xyz + type (0=point, 1=dir)
    vec4 color;       // rgb + intensity
    vec4 params;      // radius, width, height, inner_angle
    vec4 direction;   // xyz + outer_angle
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
    float _pad0;
    
    // ════════════════════════════ ATMOSPHERE DENSITY (32 bytes)
    float airDensity;
    float dustDensity;
    float ozoneDensity;
    float altitude;
    float planetRadius;
    float atmosphereHeight;
    float _pad1;
    float _pad2;
    
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
    
    // ════════════════════════════ GOD RAYS (16 bytes)
    int   godRaysEnabled;
    float godRaysIntensity;
    float godRaysDensity;
    int   godRaysSamples;
    
    // ════════════════════════════ ENVIRONMENT & LUT REFS (32 bytes)
    int   envTexSlot;
    float envIntensity;
    float envRotation;
    int   _pad5;
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
    float    _reserved[2];       // padding to 256-byte total
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
float ch_noise3D(vec3 p) {
    vec3 i = floor(p); vec3 f = fract(p); f = f*f*(3.0-2.0*f);
    float n000=ch_hash3D(i+vec3(0,0,0)); float n100=ch_hash3D(i+vec3(1,0,0));
    float n010=ch_hash3D(i+vec3(0,1,0)); float n110=ch_hash3D(i+vec3(1,1,0));
    float n001=ch_hash3D(i+vec3(0,0,1)); float n101=ch_hash3D(i+vec3(1,0,1));
    float n011=ch_hash3D(i+vec3(0,1,1)); float n111=ch_hash3D(i+vec3(1,1,1));
    return mix(mix(mix(n000,n100,f.x),mix(n010,n110,f.x),f.y),
               mix(mix(n001,n101,f.x),mix(n011,n111,f.x),f.y),f.z);
}
float ch_fbmNoise(vec3 p, int oct) {
    float v=0.0,a=0.5,fr=1.0;
    for(int i=0;i<oct;i++){v+=a*ch_noise3D(p*fr);fr*=2.0;a*=0.5;}
    return v;
}

// World-pos → object-space density for shadow ray march.
// type 0 (homogeneous): density=1.0,  type 1 (noise): fbm density,
// type 2 (NanoVDB): real trilinear grid sample via ch_sampleNanoVDB.
float ch_volDensity(VkVolumeInstance vol, vec3 wp) {
    vec3 lp;
    lp.x = vol.inv_transform[0]*wp.x + vol.inv_transform[1]*wp.y + vol.inv_transform[2]*wp.z + vol.inv_transform[3];
    lp.y = vol.inv_transform[4]*wp.x + vol.inv_transform[5]*wp.y + vol.inv_transform[6]*wp.z + vol.inv_transform[7];
    lp.z = vol.inv_transform[8]*wp.x + vol.inv_transform[9]*wp.y + vol.inv_transform[10]*wp.z + vol.inv_transform[11];
    if (any(lessThan(lp,vec3(-0.5))) || any(greaterThan(lp,vec3(0.5)))) return 0.0;
    float density = 1.0;
    if (vol.volume_type == 1) {
        vec3 nc = (lp + vec3(0.5)) * vol.noise_scale;
        density = ch_fbmNoise(nc, 4);
        vec3 ed = vec3(0.5) - abs(lp);
        density *= smoothstep(0.0, 0.1, min(min(ed.x,ed.y),ed.z));
    } else if (vol.volume_type == 2) {
        // NanoVDB: sample the actual grid data.
        // Guard: vdb_grid_address may be 0 when the source VDB file is missing or
        // the Vulkan buffer has not yet been uploaded (e.g. project opened without
        // the original .vdb file present).  Dereferencing address 0 = GPU crash.
        if (vol.vdb_grid_address != 0) {
            vec3 vdbWorldPos = vol.aabb_min + (lp + vec3(0.5)) * (vol.aabb_max - vol.aabb_min);
            density = ch_sampleNanoVDB(vol.vdb_grid_address, vdbWorldPos);
        } else {
            // Fallback: procedural noise so the volume still renders visibly
            vec3 nc = (lp + vec3(0.5)) * max(vol.noise_scale, 1.0);
            density = ch_fbmNoise(nc, 4);
        }
    }
    density = clamp((density - vol.density_remap_low) /
                    max(vol.density_remap_high - vol.density_remap_low, 1e-6), 0.0, 1.0);
    return max(density * vol.density_multiplier, 0.0);
}

// Ray-march all active volumes between shadowOrigin and light (maxDist).
// Returns transmittance in [0,1]: 1.0 = fully lit, 0.0 = fully shadowed.
float computeVolumeShadowTransmittance(vec3 shadowOrigin, vec3 lightDir, float maxDist) {
    int volCount = int(cam.pad0);   // float(volumeCount) packed by C++ each frame
    if (volCount <= 0) return 1.0;
    const float EPS = 1e-6;
    float transmittance = 1.0;
    for (int vi = 0; vi < min(volCount, 16); vi++) {
        VkVolumeInstance vol = volumes.v[vi];
        if (vol.is_active == 0) continue;
        float sigma_t = vol.scatter_coefficient + vol.absorption_coefficient;
        if (sigma_t < EPS || vol.density_multiplier < EPS) continue;
        // Transform ray to volume local space for unit-cube slab test
        vec3 lo, ld;
        lo.x=vol.inv_transform[0]*shadowOrigin.x+vol.inv_transform[1]*shadowOrigin.y+vol.inv_transform[2]*shadowOrigin.z+vol.inv_transform[3];
        lo.y=vol.inv_transform[4]*shadowOrigin.x+vol.inv_transform[5]*shadowOrigin.y+vol.inv_transform[6]*shadowOrigin.z+vol.inv_transform[7];
        lo.z=vol.inv_transform[8]*shadowOrigin.x+vol.inv_transform[9]*shadowOrigin.y+vol.inv_transform[10]*shadowOrigin.z+vol.inv_transform[11];
        ld.x=vol.inv_transform[0]*lightDir.x+vol.inv_transform[1]*lightDir.y+vol.inv_transform[2]*lightDir.z;
        ld.y=vol.inv_transform[4]*lightDir.x+vol.inv_transform[5]*lightDir.y+vol.inv_transform[6]*lightDir.z;
        ld.z=vol.inv_transform[8]*lightDir.x+vol.inv_transform[9]*lightDir.y+vol.inv_transform[10]*lightDir.z;
        // Safe inverse direction for slab test
        vec3 inv;
        inv.x = abs(ld.x)>EPS ? 1.0/ld.x : (ld.x>=0.0 ?  1e7:-1e7);
        inv.y = abs(ld.y)>EPS ? 1.0/ld.y : (ld.y>=0.0 ?  1e7:-1e7);
        inv.z = abs(ld.z)>EPS ? 1.0/ld.z : (ld.z>=0.0 ?  1e7:-1e7);
        vec3 t0 = (vec3(-0.5)-lo)*inv;  vec3 t1 = (vec3(0.5)-lo)*inv;
        vec3 tS = min(t0,t1);           vec3 tL = max(t0,t1);
        float tNL = max(max(tS.x,tS.y),max(tS.z,0.0));
        float tFL = min(min(tL.x,tL.y),tL.z);
        if (tNL >= tFL) continue;
        // tNL/tFL ARE already world-space t values.
        // The affine inv_transform shares the same scalar t as the world ray:
        //   P_local(t) = lo + t*ld  ↔  P_world(t) = origin + t*lightDir
        // No scale conversion is needed — dividing by length(ld) was wrong.
        float tNW = max(tNL, 0.001);
        float tFW = min(tFL, maxDist);
        if (tFW <= tNW) continue;
        // Ray march in world space (ch_volDensity does world→local internally)
        int steps = clamp(vol.shadow_steps, 4, 16);
        float stepW = (tFW - tNW) / float(steps);
        for (int s = 0; s < steps; s++) {
            vec3 sp = shadowOrigin + lightDir*(tNW + (float(s)+0.5)*stepW);
            float d = ch_volDensity(vol, sp);
            transmittance *= exp(-d * sigma_t * stepW * vol.shadow_strength);
            if (transmittance < 0.01) { transmittance = 0.0; break; }
        }
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
        vec3 tangent = normalize(cross(L, vec3(0.0,1.0,0.0)));
        if (length(tangent) < 1e-6) tangent = normalize(cross(L, vec3(1.0,0.0,0.0)));
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
        float u_off = (rand_u - 0.5) * light.params.y;
        float v_off = (rand_v - 0.5) * light.params.z;
        vec3 light_sample = light.position.xyz + light.direction.xyz * u_off + vec3(0.0); // area_u/area_v not stored; approximate
        vec3 L = light_sample - hit_pos;
        distance = length(L);
        if (distance < 1e-3) return false;
        wi = L / distance;
        vec3 light_normal = normalize(cross(light.direction.xyz, vec3(0.0,1.0,0.0)));
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

// BRDF evaluation (Cook-Torrance simplified port)
vec3 evaluate_brdf_gl(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic, float transmission) {
    vec3 VpL = V + L;
    float VpL_len2 = dot(VpL, VpL);
    vec3 H = (VpL_len2 > 1e-12) ? (VpL * inversesqrt(VpL_len2)) : N;
    float NdotV = max(dot(N, V), 1e-4);
    float NdotL = max(dot(N, L), 1e-4);
    float NdotH = max(dot(N, H), 1e-4);
    float VdotH = max(dot(V, H), 1e-4);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    // Fresnel
    float f = pow(1.0 - VdotH, 5.0);
    vec3 F = F0 + (vec3(1.0) - F0) * f;
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
    vec3 k_d = (vec3(1.0) - F) * (1.0 - metallic) * max(0.0, 1.0 - transmission);
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

// Metal için renkli Fresnel (F0 = albedo)
vec3 schlickFresnelVec(float cosTheta, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ============================================================
// Material Scatter Fonksiyonları
// ============================================================

// --- Lambertian Diffuse ---
void scatterDiffuse(vec3 hitPos, vec3 normal, vec3 albedo, inout uint seed) {
    vec3 dir = cosineSampleHemisphere(normal, seed);

    payload.scatterOrigin = hitPos + normal * RAY_OFFSET;
    payload.scatterDir    = dir;
    // Cosine-weighted sampling ile PDF = cos/PI, BRDF = albedo/PI
    // Throughput = BRDF * cos / PDF = albedo → direkt albedo
    payload.attenuation  *= albedo;
    payload.scattered     = true;
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
}

// --- Dielectric Glass (Fresnel + TIR + Roughness) ---
void scatterGlass(vec3 hitPos, vec3 normal, vec3 rayDir, vec3 albedo, float ior, float roughness, inout uint seed) {
    // Işığın hangi taraftan geldiğini belirle
    bool  frontFace    = dot(rayDir, normal) < 0.0;
    vec3  macroNormal  = frontFace ? normal : -normal;
    float etaRatio     = frontFace ? (1.0 / ior) : ior;
    
    vec3 outNormal = macroNormal;

    // Use GGX microfacet normal if surface is rough
    if (roughness >= 0.01) {
        vec3 V = -rayDir;
        outNormal = ggxSampleHemisphere(macroNormal, V, roughness, seed);
    }

    // Fresnel ve TIR kararı için makro normal kullan (OptiX ile aynı).
    // Mikrofaset normali grazing angle'a yakın örneklenirse fresnelProb
    // yapay biçimde ~1.0'a çıkıyor ve aşırı reflection üretiyordu.
    float cosTheta     = min(dot(-rayDir, macroNormal), 1.0);
    float sinTheta     = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool  totalIntRefl = (etaRatio * sinTheta) > 1.0;

    float fresnelProb  = schlickFresnel(cosTheta, ior);
    bool  doReflect    = totalIntRefl || (rnd(seed) < fresnelProb);

    vec3 dir;
    vec3 offsetDir;
    if (doReflect) {
        dir       = reflect(rayDir, outNormal);
        offsetDir = macroNormal;           // Yüzeyin dışına offset
    } else {
        dir       = refract(rayDir, outNormal, etaRatio);
        offsetDir = -macroNormal;          // Yüzeyin içine offset (refract için)
    }

    // Prevent rays from going through the wrong side of the macroscopic surface due to high roughness.
    // Use safe fallback instead of hard absorb to avoid black pixel artifacts.
    if (doReflect && dot(dir, macroNormal) <= 0.0) {
        dir = reflect(rayDir, macroNormal);
        offsetDir = macroNormal;
    } else if (!doReflect && dot(dir, macroNormal) >= 0.0) {
        dir = reflect(rayDir, macroNormal);
        offsetDir = macroNormal;
    }

    payload.scatterOrigin = hitPos + offsetDir * RAY_OFFSET;
    payload.scatterDir    = normalize(dir);
    payload.attenuation  *= albedo;      // Transmission tint
    payload.scattered     = true;
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
        scatterDist = min(scatterDist, safeScale * 10.0);

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
}

// ============================================================
// Clearcoat — Second GGX Specular Lobe (IOR=1.5, lacquer layer)
// ============================================================
void scatterClearcoat(vec3 hitPos, vec3 normal, vec3 rayDir,
                      float ccRoughness, inout uint seed) {
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

    payload.attenuation  *= vec3(fresnel * G1L);
    payload.scatterOrigin = hitPos + normal * RAY_OFFSET;
    payload.scatterDir    = L;
    payload.scattered     = true;
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
                  float micro_strength, float micro_scale, float wind_speed,
                  float fft_time_scale,
                  vec3 shallow_color, vec3 deep_color,
                  float ior, float roughness,
                  inout uint seed)
{
    // Animation time: frame count converted to seconds, user-scaled
    float time = cam.waterTime * max(fft_time_scale, 0.1);

    // ── Gerstner wave normal + foam ─────────────────────────────
    vec3  waveNormal;
    float foam;
    evaluateWaterGerstner(hitPos, time, wave_speed, wave_strength, wave_freq,
                          waveNormal, foam);

    // ── Micro-detail capillary ripples ──────────────────────────
    if (micro_strength > 0.001) {
        float sc = max(micro_scale, 1.0);
        float morph = time * 0.5;
        vec2 p1 = vec2(hitPos.x, hitPos.z) * sc       + vec2(sin(morph), cos(morph * 0.8)) * 0.3;
        vec2 p2 = vec2(hitPos.x, hitPos.z) * sc * 0.5 + vec2(sin(time * 0.25), sin(time * 0.175)) * 0.4;

        const float dx = 0.01;
        float hc = water_fbm(p1) * 0.5 + water_fbm(p2) * 0.5;
        float hx = water_fbm(p1 + vec2(dx,0.0)) * 0.5 + water_fbm(p2 + vec2(dx,0.0)) * 0.5;
        float hz = water_fbm(p1 + vec2(0.0,dx)) * 0.5 + water_fbm(p2 + vec2(0.0,dx)) * 0.5;

        float dsdx = (hx - hc) / dx;
        float dsdz = (hz - hc) / dx;
        vec3 microN = normalize(vec3(-dsdx * micro_strength, 1.0, -dsdz * micro_strength));
        waveNormal  = normalize(waveNormal + microN);

        // Micro-peak foam (replaces/supplements Gerstner foam for FFT-style look)
        float microSlope = clamp(hc * 0.5 + 0.5, 0.0, 1.0);
        float microFoam  = clamp((microSlope - foam_threshold) * foam_level * 5.0, 0.0, 1.0);
        foam = max(foam, microFoam);
    }

    // ── Build shading normal from wave perturbation ──────────────
    // waveNormal lives in a y-up tangent frame; project onto geoNormal's ONB
    vec3 tgt, btgt;
    buildONB(geoNormal, tgt, btgt);
    vec3 shadingNormal = normalize(tgt * waveNormal.x + geoNormal * waveNormal.y + btgt * waveNormal.z);
    if (dot(shadingNormal, -rayDir) < 0.0) shadingNormal = geoNormal; // sanity

    // ── Depth-based color blend ──────────────────────────────────
    float cosNV    = max(dot(-rayDir, shadingNormal), 0.0);
    float tDepth   = 1.0 - cosNV;  // grazing angle → deeper look
    tDepth = tDepth * tDepth * (3.0 - 2.0 * tDepth); // smoothstep
    vec3 waterColor = mix(shallow_color, deep_color, tDepth);

    // ── Blend foam (white crest) ─────────────────────────────────
    vec3 finalAlbedo = mix(waterColor, vec3(0.92, 0.95, 1.0), clamp(foam, 0.0, 0.85));

    // ── Tint attenuation with water color, then scatter as glass ─
    payload.attenuation *= finalAlbedo;
    scatterGlass(hitPos, shadingNormal, rayDir, vec3(1.0), ior, roughness, seed);
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
    vec3 geomNormal; // The physical flat non-interpolated triangle normal

    if (geo.vertexAddr != 0) {
        VertexBuffer vBuf = VertexBuffer(geo.vertexAddr);
        vec3 v0 = vBuf.v[i0];
        vec3 v1 = vBuf.v[i1];
        vec3 v2 = vBuf.v[i2];
        vec3 localFaceNormal = normalize(cross(v1 - v0, v2 - v0));
        geomNormal = normalize(vec3(localFaceNormal * mat3(gl_WorldToObjectEXT)));
    } else {
        geomNormal = normalize(vec3(0, 1, 0));  // Fallback
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
        worldNormal = geomNormal;
    }

    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 rayDir = normalize(gl_WorldRayDirectionEXT);

    // Compute UV coordinates if available
    vec2 hitUV = vec2(0.0);
    if (geo.uvAddr != 0) {
        UVBuffer uvBuf = UVBuffer(geo.uvAddr);
        hitUV = uvBuf.u[i0] * bary.x + uvBuf.u[i1] * bary.y + uvBuf.u[i2] * bary.z;
    }

    // Vulkan shader coordinate origin differs; flip V to match OptiX (and texture upload)
    hitUV.y = 1.0 - hitUV.y;

    // Double-sided: Normaller her zaman ray'e karşı baksın
    if (dot(worldNormal, rayDir) > 0.0) {
        worldNormal = -worldNormal;
    }
    // Geometrik normali de kamera/ışın yönüne çeviriyoruz (kendi kendine çarpışmayı önlemek için)
    if (dot(geomNormal, rayDir) > 0.0) {
        geomNormal = -geomNormal;
    }

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
            // [FIX] Accumulate per-layer normal maps in tangent space
            vec3  blendNormal_ts = vec3(0.0);  // weighted sum of tangent-space normals
            bool  anyNormalTex   = false;

            uint activeCount = min(tl.layer_count, 4u);
            for (uint k = 0u; k < activeCount; k++) {
                if (weights[k] < 0.001) continue;
                Material lm = materials.m[tl.layer_mat_id[k]];
                vec2 layerUV = hitUV * tl.layer_uv_scale[k];

                // Layer albedo
                vec3 lAlbedo = max(vec3(lm.albedo_r, lm.albedo_g, lm.albedo_b), vec3(0.0));
                if (int(lm.albedo_tex) > 0) {
                    lAlbedo = texture(materialTextures[nonuniformEXT(int(lm.albedo_tex))], layerUV).rgb;
                }
                blendAlbedo += weights[k] * lAlbedo;

                // Layer roughness
                float lRough = clamp(lm.roughness, 0.0, 1.0);
                if (int(lm.roughness_tex) > 0) {
                    lRough = texture(materialTextures[nonuniformEXT(int(lm.roughness_tex))], layerUV).r;
                }
                blendRoughness += weights[k] * lRough;

                // Layer metallic
                float lMetal = clamp(lm.metallic, 0.0, 1.0);
                if (int(lm.metallic_tex) > 0) {
                    lMetal = texture(materialTextures[nonuniformEXT(int(lm.metallic_tex))], layerUV).r;
                }
                blendMetallic += weights[k] * lMetal;

                // [FIX] Layer normal map — blend tangent-space normals by weight.
                // Layers without a normal map contribute a flat (0,0,1) tangent-space vector.
                if (int(lm.normal_tex) > 0) {
                    vec3 ns = texture(materialTextures[nonuniformEXT(int(lm.normal_tex))], layerUV).rgb;
                    ns = ns * 2.0 - vec3(1.0); // [0,1] -> [-1,1]
                    ns.x = -ns.x;               // [FIX] Flip X for basis consistency
                    ns.y = -ns.y;               // [FIX] Flip Y for DirectX/OpenGL normal map convention
                    ns.z = abs(ns.z);           // ensure outward-pointing Z
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
            // Clear per-material texture slots — blending already resolved them above
            mat.albedo_tex    = 0u;
            mat.roughness_tex = 0u;
            mat.metallic_tex  = 0u;

            // [FIX] Apply blended normal map to world-space normal immediately.
            // Set mat.normal_tex = 0 so the standard normal-map section below does nothing.
            if (anyNormalTex) {
                vec3 nts = normalize(blendNormal_ts);
                vec3 tgt, btgt;
                buildONB(worldNormal, tgt, btgt);
                vec3 perturbed = normalize(tgt * nts.x + btgt * nts.y + worldNormal * nts.z);
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
    float ior         = (mat.ior > 0.01) ? mat.ior : 1.5;
    float transmission = clamp(mat.transmission, 0.0, 1.0);

    // ----------------------------------------------------------
    // 4. Emission — ayrı field, scatter ile karışmaz
    //    payload.radiance ve hitEmissive, emission texture sampling
    //    SONRASI atanır (aşağıda) — texture'ın rengi override edebilmesi için
    // ----------------------------------------------------------

    // Sample albedo texture
    int albedoTexID = int(mat.albedo_tex);
    if (albedoTexID > 0) {
        albedo = texture(materialTextures[nonuniformEXT(albedoTexID)], hitUV).rgb;
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
            maskValue = texture(materialTextures[nonuniformEXT(opacityTexID)], hitUV).a;
        } else {
            maskValue = texture(materialTextures[nonuniformEXT(opacityTexID)], hitUV).r;
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
            payload.scatterOrigin = offset_ray(hitPos, -geomNormal);
            payload.scatterDir    = rayDir;
            payload.scattered     = true;
            return;
        }
    }
    
    // Opaque pixel — continue to shading normally
    
    // --- MODE 2: glass/transmission adjustment ---
    if (mat.opacity < 0.99 && metallic < 0.1 && transmission < 0.01) {
        transmission = 1.0 - mat.opacity;
    }


   // Sample emission texture
int emissionTexID = int(mat.emission_tex);
if (emissionTexID > 0) {
    vec3 emTex = texture(materialTextures[nonuniformEXT(emissionTexID)], hitUV).rgb;
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
        float trans = texture(materialTextures[nonuniformEXT(transmissionTexID)], hitUV).r;
        transmission = clamp(trans, 0.0, 1.0);
    }
    
    // Sample roughness texture
    int roughTexID = int(mat.roughness_tex);
    if (roughTexID > 0) {
        float r = texture(materialTextures[nonuniformEXT(roughTexID)], hitUV).g;
        roughness = clamp(r, 0.0, 1.0);
    }
    
    // Sample metallic texture
    int metallicTexID = int(mat.metallic_tex);
    if (metallicTexID > 0) {
        float m = texture(materialTextures[nonuniformEXT(metallicTexID)], hitUV).b;
        metallic = clamp(m, 0.0, 1.0);
    }
    
    // Apply normal map if present (perturb surface normal)
    int normalTexID = int(mat.normal_tex);
    vec3 tangentNormal = worldNormal;  // Default to geometry normal
    if (normalTexID > 0) {
        // Sample normal map (OpenGL format: RGB = normal direction)
        vec3 normalMapSample = texture(materialTextures[nonuniformEXT(normalTexID)], hitUV).rgb;
        
        // Validate: ensure we don't have pure black or NaN
        float mapLength = length(normalMapSample);
        if (mapLength > 0.1) {  // Non-zero check
            // Convert from [0, 1] to [-1, 1] range
            vec3 normalMapDir = normalMapSample * 2.0 - vec3(1.0);
            
            // Ensure Z is positive (pointing outward in tangent space)
            normalMapDir.z = abs(normalMapDir.z);
            
            // Normalize to ensure unit vector
            vec3 tangentSpaceNormal = normalize(normalMapDir);
            
            // Build orthonormal basis from geometry normal
            vec3 tangent, bitangent;
            buildONB(worldNormal, tangent, bitangent);
            
            // Transform from tangent space to world space
            vec3 worldNormalPerturbed = normalize(
                tangent * tangentSpaceNormal.x +
                bitangent * tangentSpaceNormal.y +
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
    worldNormal = tangentNormal;

    // ----------------------------------------------------------
    // IS_WATER fast path  (mat.sheen > 0 = IS_WATER flag)
    // Water has its own scatter: Gerstner waves + glass refraction.
    // Must run BEFORE transmission/direct-lighting/diffuse paths.
    // ----------------------------------------------------------
    if (mat.sheen > 0.001) {
        scatterWater(
            hitPos, worldNormal, rayDir,
            /*wave_speed*/     mat.anisotropic,
            /*wave_strength*/  mat.sheen,
            /*wave_freq*/      mat.sheen_tint,
            /*foam_level*/     mat.translucent,
            /*foam_threshold*/ mat.foam_threshold,
            /*micro_strength*/ mat.micro_detail_strength,
            /*micro_scale*/    mat.micro_detail_scale,
            /*wind_speed*/     mat.fft_wind_speed,
            /*fft_time_scale*/ mat.fft_time_scale,
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
    if (transmission > 0.01) {
        if (rnd(payload.seed) < transmission) {
            // 1) Chosen transmission path - act as Glass
            scatterGlass(hitPos, worldNormal, rayDir, albedo, ior, roughness, payload.seed);
            return; // Immediately return, skipping direct lighting (Next Event Estimation)
        } else {
            // 2) Chosen base path (diffuse/metal), compensate probability weight
            payload.attenuation *= (1.0 / max(1.0 - transmission, 0.01));
            // Ensure transmission doesn't accidentally affect BRDF down the line anymore
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
                        vec3 shadowOrigin = offset_ray(hitPos, geomNormal);
                        float tmin = 0.0;
                        float tmax = min(max(0.0, dist - 1e-3), 10000.0);
                        if (tmax > tmin) {
                            // No OpaqueEXT → any-hit shader tests transparency per pixel
                            // SkipClosestHit → no closest-hit overhead; shadow value set by any-hit/miss only
                            // missIndex=1 → shadow_miss.rmiss sets shadowOccluded=false when ray escapes
                            uint shadowFlags = gl_RayFlagsTerminateOnFirstHitEXT
                                             | gl_RayFlagsSkipClosestHitShaderEXT
                                             | gl_RayFlagsNoOpaqueEXT; // force anyhit even on OPAQUE geometry
                            // mask 0x01 = triangles only — volume AABBs have mask 0x02 so
                            // they are invisible to shadow rays and cannot cast hard shadows.
                            traceRayEXT(topLevelAS, shadowFlags, 0x01, 0, 1, 1, shadowOrigin, tmin, wi, tmax, 1);
                        }
                        if (!shadowOccluded) {
                            // Volumetric soft shadow: march through any volume AABB between surface and light.
                            // cam.pad0 carries float(volumeCount) from C++ renderProgressive each frame.
                            float volShadowTr = computeVolumeShadowTransmittance(shadowOrigin, wi, tmax);
                            vec3 V = normalize(-rayDir);
                            vec3 brdf = evaluate_brdf_gl(worldNormal, V, wi, albedo, roughness, metallic, transmission);
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
                            // Apply volumetric transmittance (soft shadow from volumes)
                            contrib *= volShadowTr;

                            vec3 att = max(payload.attenuation, vec3(0.0));
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
    // Nishita Sun Direct Lighting (atmosphere mode == 2)
    // The sun is NOT in the lights[] buffer, so it is sampled here.
    // Uses the same shadow + volumetric-shadow pipeline as the light loop.
    // ----------------------------------------------------------
    if (worldData.w.mode == 2 && worldData.w.sunIntensity > 1e-4) {
        vec3 sunDir = normalize(worldData.w.sunDir);
        float NdotSun = max(dot(worldNormal, sunDir), 0.0);
        if (NdotSun > 1e-6) {
            shadowOccluded = true;
            // ULP-based offset: self-intersection-safe on thin/distant geometry
            vec3 sunShadowOrigin = offset_ray(hitPos, geomNormal);
            float sunTmin = 0.0;
            float sunTmax = 1e8;
            uint sunShadowFlags = gl_RayFlagsTerminateOnFirstHitEXT
                                | gl_RayFlagsSkipClosestHitShaderEXT
                                | gl_RayFlagsNoOpaqueEXT; // force anyhit even on OPAQUE geometry
            // mask 0x01 = triangles only — volume AABBs skipped (handled by volumetric transmittance)
            traceRayEXT(topLevelAS, sunShadowFlags, 0x01, 0, 1, 1,
                        sunShadowOrigin, sunTmin, sunDir, sunTmax, 1);
            if (!shadowOccluded) {
                float sunVolTr = computeVolumeShadowTransmittance(sunShadowOrigin, sunDir, sunTmax);
                vec3 V        = normalize(-rayDir);
                vec3 sunBRDF  = evaluate_brdf_gl(worldNormal, V, sunDir,
                                                 albedo, roughness, metallic, transmission);
                vec3 sunLi    = worldData.w.sunColor * worldData.w.sunIntensity;
                vec3 sunContrib = sunBRDF * sunLi * NdotSun * sunVolTr;
                sunContrib = clamp(sunContrib, vec3(0.0), vec3(1e4));
                vec3 att = clamp(payload.attenuation, vec3(0.0), vec3(1e2));
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
            scatterClearcoat(hitPos, worldNormal, rayDir, clearcoatRoughness, payload.seed);
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
        const float F0_DIELECTRIC = 0.04;
        vec3  viewDir  = -rayDir;
        float cosTheta = max(dot(viewDir, worldNormal), 0.0);

        // Schlick Fresnel, attenuated by roughness (rough surfaces scatter less)
        float fresnelBase   = F0_DIELECTRIC + (1.0 - F0_DIELECTRIC)
                              * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
        float fresnelWeight = fresnelBase * (1.0 - roughness * roughness);
        fresnelWeight = clamp(fresnelWeight, 0.001, 0.999);

        if (rnd(payload.seed) < fresnelWeight) {
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
        // Metallic blend: stochastic selection
        if (rnd(payload.seed) < metalWeight) {
            scatterMetal(hitPos, worldNormal, rayDir, albedo, roughness, payload.seed);
            payload.attenuation *= (1.0 / max(metalWeight, 0.1));
        } else {
            scatterDiffuse(hitPos, worldNormal, albedo, payload.seed);
            payload.attenuation *= (1.0 / max(diffuseWeight, 0.1));
        }
    }
}
