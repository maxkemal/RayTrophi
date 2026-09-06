#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : enable

// Realtime SurfaceSDF compositor. The producer and Vulkan RT keep ownership of
// the field; this pass reads the exact same VkVolumeInstance bytes through a raw
// uint view. Raw access is intentional: there is no fifth private copy of the
// 624-byte volume struct whose stride could silently drift from the RT ABI.

layout(location = 0) in vec2 vNdc;
layout(location = 0) out vec4 outColor;

#define GpuMaterial Material
#include "material_struct.glsl"
#include "pbr_texture_policy.glsl"
#include "material_preview_transmission.glsl"
#include "post_chain.glsl"

layout(set = 0, binding = 0, std430) readonly buffer MaterialBuffer {
    GpuMaterial materials[];
};
layout(set = 0, binding = 4, std430) readonly buffer MaterialExtBuffer {
    MaterialExt materialsExt[];
};
layout(set = 0, binding = 1) uniform sampler2D textures[];
layout(set = 0, binding = 2) uniform sampler2D envMaps[2];
layout(set = 0, binding = 9) uniform sampler2D worldEnvironment;
layout(set = 0, binding = 13) uniform sampler2D worldIrradiance;
layout(set = 0, binding = 14) uniform sampler2D worldPrefiltered;
layout(set = 0, binding = 15) uniform sampler2D worldBrdfLut;
layout(set = 0, binding = 17) uniform sampler2D previewOpaqueColor;
layout(set = 0, binding = 18) uniform sampler2D previewOpaqueDepth;

struct LightData {
    vec4 position;
    vec4 color;
    vec4 params;
    vec4 direction;
    vec4 area_u;
    vec4 area_v;
};
layout(set = 0, binding = 5, std430) readonly buffer PreviewLightBuffer {
    LightData sceneLights[];
};

layout(set = 0, binding = 6, std430) readonly buffer PreviewSceneGlobalsBuffer {
    uint sceneLightCount;
    uint sceneFlags;
    uint shadowedLightCount;
    uint worldMode;
    vec4 worldColor;
    vec4 worldParams;
    vec4 worldSun;
    vec4 atmosphereA;
    vec4 atmosphereB;
    vec4 postA;
    vec4 postB;
    vec4 postC;
};

const uint PREVIEW_MAX_LIGHTS = 32u;
struct PreviewShadowRecord {
    mat4 viewProj[6];
    vec4 atlasRect[6];
    uvec4 meta;
    vec4 params;
};
layout(set = 0, binding = 7, std430) readonly buffer PreviewShadowBuffer {
    PreviewShadowRecord shadowRecords[PREVIEW_MAX_LIGHTS + 1u];
};
layout(set = 0, binding = 8) uniform sampler2D previewShadowAtlas;

// binding 20 aliases VulkanDevice::m_volumeBuffer. One record is 624 bytes =
// 156 uints. Offsets below are byte offsets from vulkan_volume_types.h and are
// statically audited by scripts/audit_realtime_sdf_surface.py.
layout(set = 0, binding = 20, std430) readonly buffer PreviewVolumeRawBuffer {
    uint volumeWords[];
};

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;
    vec4 lightDir0;
    vec4 lightDir1;
    vec4 lightDir2;
    // x=material count, y=quality, z=lighting preset, w=volume count
    uvec4 materialMeta;
} pc;

const float PI = 3.14159265358979323846;
const uint VOLUME_STRIDE_WORDS = 156u;

uint volWord(uint volumeIndex, uint byteOffset) {
    return volumeWords[volumeIndex * VOLUME_STRIDE_WORDS + (byteOffset >> 2u)];
}
float volFloat(uint volumeIndex, uint byteOffset) {
    return uintBitsToFloat(volWord(volumeIndex, byteOffset));
}
uint64_t volAddress(uint volumeIndex, uint byteOffset) {
    uvec2 pair = uvec2(volWord(volumeIndex, byteOffset),
                       volWord(volumeIndex, byteOffset + 4u));
    return uint64_t(pair.x) | (uint64_t(pair.y) << 32);
}
vec3 volVec3(uint volumeIndex, uint byteOffset) {
    return vec3(volFloat(volumeIndex, byteOffset),
                volFloat(volumeIndex, byteOffset + 4u),
                volFloat(volumeIndex, byteOffset + 8u));
}

vec3 transformPoint(uint vi, uint base, vec3 p) {
    return vec3(
        volFloat(vi, base +  0u) * p.x + volFloat(vi, base +  4u) * p.y +
        volFloat(vi, base +  8u) * p.z + volFloat(vi, base + 12u),
        volFloat(vi, base + 16u) * p.x + volFloat(vi, base + 20u) * p.y +
        volFloat(vi, base + 24u) * p.z + volFloat(vi, base + 28u),
        volFloat(vi, base + 32u) * p.x + volFloat(vi, base + 36u) * p.y +
        volFloat(vi, base + 40u) * p.z + volFloat(vi, base + 44u));
}
vec3 transformVector(uint vi, uint base, vec3 p) {
    return vec3(
        volFloat(vi, base +  0u) * p.x + volFloat(vi, base +  4u) * p.y + volFloat(vi, base +  8u) * p.z,
        volFloat(vi, base + 16u) * p.x + volFloat(vi, base + 20u) * p.y + volFloat(vi, base + 24u) * p.z,
        volFloat(vi, base + 32u) * p.x + volFloat(vi, base + 36u) * p.y + volFloat(vi, base + 40u) * p.z);
}

bool rayVolumeInterval(uint vi, vec3 ro, vec3 rd, out float tNear, out float tFar) {
    vec3 localOrigin = transformPoint(vi, 184u, ro);
    vec3 localDir = transformVector(vi, 184u, rd);
    // Keep the sign of near-parallel rays. Replacing a tiny negative component
    // with +epsilon flips the slab ordering at axis-aligned camera angles and
    // can make the whole volume disappear until the camera moves again.
    vec3 safeDir = vec3(
        abs(localDir.x) > 1e-8 ? localDir.x : (localDir.x < 0.0 ? -1e-8 : 1e-8),
        abs(localDir.y) > 1e-8 ? localDir.y : (localDir.y < 0.0 ? -1e-8 : 1e-8),
        abs(localDir.z) > 1e-8 ? localDir.z : (localDir.z < 0.0 ? -1e-8 : 1e-8));
    vec3 t0 = (volVec3(vi, 48u) - localOrigin) / safeDir;
    vec3 t1 = (volVec3(vi, 60u) - localOrigin) / safeDir;
    vec3 lo = min(t0, t1);
    vec3 hi = max(t0, t1);
    tNear = max(max(lo.x, lo.y), lo.z);
    tFar = min(min(hi.x, hi.y), hi.z);
    return tFar > max(tNear, 0.0);
}

// NanoVDB reader: identical buffer-reference contract to volume_closesthit.
#define PNANOVDB_GLSL
#define PNANOVDB_BUF_CUSTOM
struct pnanovdb_buf_t { uint64_t address; };
layout(buffer_reference, std430, buffer_reference_align=4) buffer NanoVDBBlock {
    uint data[];
};
uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byteOffset) {
    NanoVDBBlock block = NanoVDBBlock(buf.address);
    return block.data[byteOffset >> 2u];
}
uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byteOffset) {
    NanoVDBBlock block = NanoVDBBlock(buf.address);
    uint i = byteOffset >> 2u;
    return uvec2(block.data[i], block.data[i + 1u]);
}
void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint byteOffset, uint value) {}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint byteOffset, uvec2 value) {}
#include "PNanoVDB.h"

float sampleGrid(pnanovdb_buf_t buf, pnanovdb_map_handle_t mapHandle,
                 inout pnanovdb_readaccessor_t accessor, vec3 p) {
    pnanovdb_vec3_t wp = pnanovdb_vec3_uniform(0.0);
    wp.x = p.x; wp.y = p.y; wp.z = p.z;
    pnanovdb_vec3_t ip = pnanovdb_map_apply_inverse(buf, mapHandle, wp);
    vec3 q = vec3(ip.x, ip.y, ip.z);
    vec3 base = floor(q);
    vec3 f = fract(q);
    float d[8];
    for (int corner = 0; corner < 8; ++corner) {
        pnanovdb_coord_t c;
        c.x = int(base.x) + ((corner & 1) != 0 ? 1 : 0);
        c.y = int(base.y) + ((corner & 2) != 0 ? 1 : 0);
        c.z = int(base.z) + ((corner & 4) != 0 ? 1 : 0);
        pnanovdb_address_t a = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, c);
        d[corner] = pnanovdb_read_float(buf, a);
    }
    return mix(mix(mix(d[0], d[1], f.x), mix(d[2], d[3], f.x), f.y),
               mix(mix(d[4], d[5], f.x), mix(d[6], d[7], f.x), f.y), f.z);
}

float sampleIso(uint vi, pnanovdb_buf_t buf, pnanovdb_map_handle_t mapHandle,
                inout pnanovdb_readaccessor_t accessor, vec3 worldPos) {
    vec3 local = transformPoint(vi, 184u, worldPos);
    if (any(lessThan(local, volVec3(vi, 48u))) ||
        any(greaterThan(local, volVec3(vi, 60u)))) return 0.0;
    return sampleGrid(buf, mapHandle, accessor,
                      local - volVec3(vi, 416u));
}

bool marchSurface(uint vi, vec3 ro, vec3 rd, float maxVisibleT,
                  out float hitT, out vec3 hitNormal) {
    float boxNear, boxFar;
    if (!rayVolumeInterval(vi, ro, rd, boxNear, boxFar)) return false;
    float beginT = max(boxNear, 0.001);
    float endT = min(boxFar, maxVisibleT);
    if (endT <= beginT) return false;

    pnanovdb_buf_t buf;
    buf.address = volAddress(vi, 232u);
    pnanovdb_grid_handle_t gridHandle;
    gridHandle.address.byte_offset = 0u;
    pnanovdb_tree_handle_t treeHandle = pnanovdb_grid_get_tree(buf, gridHandle);
    pnanovdb_root_handle_t rootHandle = pnanovdb_tree_get_root(buf, treeHandle);
    pnanovdb_map_handle_t mapHandle = pnanovdb_grid_get_map(buf, gridHandle);
    pnanovdb_readaccessor_t accessor;
    pnanovdb_readaccessor_init(accessor, rootHandle);

    uint quality = pc.materialMeta.y & 0xffu;
    // A SurfaceSDF is a thin iso band, not a participating medium. The volume
    // shader's authored max_steps may legitimately be very low for smoke but
    // must not make this boundary camera-phase dependent. These caps cover the
    // diagonal of common 64/128/256 grids at sub-voxel spacing while retaining
    // an explicit Performance escape hatch.
    int cap = quality <= 1u ? 192 : (quality == 2u ? 512 : 1024);
    float voxel = max(volFloat(vi, 176u), 1e-4);
    float authoredStep = max(volFloat(vi, 152u), voxel * 0.1);
    float targetStep = min(authoredStep, voxel * 0.45);
    float stepSize = max(targetStep, (endT - beginT) / float(cap));
    int steps = min(int(ceil((endT - beginT) / stepSize)) + 1, cap + 1);

    const float iso = 0.5;
    float t0 = beginT;
    float d0 = sampleIso(vi, buf, mapHandle, accessor, ro + rd * t0);
    // Match Vulkan RT's boundary-side contract. A ray begins on the volume
    // AABB and can land within rounding distance of the 0.5 surface; treating
    // that sample as randomly inside/outside makes entry hits angle-dependent.
    const float isoHysteresis = 0.05;
    bool startsInside = d0 > iso + isoHysteresis;
    if (!startsInside) d0 = min(d0, iso - 1e-4);
    for (int s = 0; s < steps; ++s) {
        float t1 = min(t0 + stepSize, endT);
        float d1 = sampleIso(vi, buf, mapHandle, accessor, ro + rd * t1);
        bool crossed = startsInside ? (d0 >= iso && d1 < iso) : (d0 < iso && d1 >= iso);
        if (crossed) {
            float a = t0, b = t1;
            for (int refine = 0; refine < 4; ++refine) {
                float mid = 0.5 * (a + b);
                float dm = sampleIso(vi, buf, mapHandle, accessor, ro + rd * mid);
                if ((dm > iso) == startsInside) a = mid; else b = mid;
            }
            float da = sampleIso(vi, buf, mapHandle, accessor, ro + rd * a);
            float db = sampleIso(vi, buf, mapHandle, accessor, ro + rd * b);
            float denom = db - da;
            float fraction = abs(denom) > 1e-7
                ? clamp((iso - da) / denom, 0.0, 1.0) : 0.5;
            hitT = mix(a, b, fraction);
            vec3 hp = ro + rd * hitT;
            float h = voxel;
            vec3 grad = vec3(
                sampleIso(vi, buf, mapHandle, accessor, hp + vec3(h,0,0)) - sampleIso(vi, buf, mapHandle, accessor, hp - vec3(h,0,0)),
                sampleIso(vi, buf, mapHandle, accessor, hp + vec3(0,h,0)) - sampleIso(vi, buf, mapHandle, accessor, hp - vec3(0,h,0)),
                sampleIso(vi, buf, mapHandle, accessor, hp + vec3(0,0,h)) - sampleIso(vi, buf, mapHandle, accessor, hp - vec3(0,0,h)));
            hitNormal = dot(grad, grad) > 1e-8 ? normalize(-grad) : (startsInside ? rd : -rd);
            if (dot(hitNormal, rd) > 0.0) hitNormal = -hitNormal;
            return true;
        }
        t0 = t1;
        d0 = d1;
        if (t0 >= endT) break;
    }
    return false;
}

vec2 directionUv(vec3 d) {
    d = normalize(d);
    return vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5,
                acos(clamp(d.y, -1.0, 1.0)) / PI);
}
vec3 previewEnvironment(vec3 d, float roughness) {
    uint preset = pc.materialMeta.z;
    if (preset == 3u) {
        if ((sceneFlags & (1u << 5u)) != 0u)
            return textureLod(worldPrefiltered, directionUv(d), roughness * 7.0).rgb * worldParams.y;
        return textureLod(worldEnvironment, directionUv(d), roughness * 5.0).rgb * worldParams.y;
    }
    return textureLod(envMaps[preset == 2u ? 1 : 0], directionUv(d), roughness * 5.0).rgb;
}

vec3 sampleOpaqueSceneRough(vec2 uv, float roughness) {
    vec2 texel = 1.0 / max(postC.xy, vec2(1.0));
    float radius = roughness * roughness * 8.0;
    vec3 sum = texture(previewOpaqueColor, uv).rgb * 2.0;
    float weight = 2.0;
    int taps = (pc.materialMeta.y & 0xffu) >= 3u ? 8 : 4;
    const float goldenAngle = 2.39996323;
    for (int i = 0; i < 8; ++i) {
        if (i >= taps) break;
        float angle = float(i) * goldenAngle;
        float ring = sqrt((float(i) + 0.5) / float(taps)) * radius;
        vec2 q = clamp(uv + vec2(cos(angle), sin(angle)) * texel * ring,
                       vec2(0.002), vec2(0.998));
        sum += texture(previewOpaqueColor, q).rgb;
        weight += 1.0;
    }
    return sum / weight;
}

bool projectOpaqueRefraction(vec3 origin, vec3 direction, float distance,
                             float surfaceDepth, out vec2 uv) {
    vec4 projected = pc.viewProj * vec4(origin + direction * distance, 1.0);
    if (projected.w <= 1e-5) return false;
    vec3 ndc = projected.xyz / projected.w;
    uv = ndc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.002))) ||
        any(greaterThan(uv, vec2(0.998)))) return false;
    // Reject foreground objects, but accept opaque geometry or sky behind the
    // liquid boundary. The snapshot was captured before this SDF draw.
    return texture(previewOpaqueDepth, uv).r + 1e-5 >= surfaceDepth;
}

vec4 sampleTriplanar(uint textureIndex, vec3 p, vec3 n, vec2 scale, vec2 offset) {
    vec3 w = pow(abs(n), vec3(4.0));
    w /= max(w.x + w.y + w.z, 1e-5);
    vec4 x = textureLod(textures[nonuniformEXT(textureIndex)], p.zy * scale + offset, 0.0);
    vec4 y = textureLod(textures[nonuniformEXT(textureIndex)], p.xz * scale + offset, 0.0);
    vec4 z = textureLod(textures[nonuniformEXT(textureIndex)], p.xy * scale + offset, 0.0);
    return x * w.x + y * w.y + z * w.z;
}

float distributionGGX(float ndh, float roughness) {
    float a = max(roughness * roughness, 0.0025);
    float a2 = a * a;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}
float geometrySmith(float ndv, float ndl, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    float gv = ndv / mix(ndv, 1.0, k);
    float gl = ndl / mix(ndl, 1.0, k);
    return gv * gl;
}
vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (vec3(1.0) - f0) * pow(1.0 - clamp(cosTheta, 0.0, 1.0), 5.0);
}

bool evaluateLight(uint index, vec3 p, out vec3 l, out vec3 radiance) {
    LightData light = sceneLights[index];
    int type = int(light.position.w + 0.5);
    radiance = light.color.rgb * light.color.w;
    if (type == 1) { l = normalize(light.direction.xyz); return true; }
    vec3 toLight = light.position.xyz - p;
    float d2 = dot(toLight, toLight);
    if (d2 < 1e-6) return false;
    l = toLight * inversesqrt(d2);
    float attenuation = 1.0 / d2;
    if (type == 2) {
        vec3 ln = normalize(cross(light.area_u.xyz, light.area_v.xyz));
        attenuation *= max(dot(-l, ln), 0.0) * max(light.params.y * light.params.z, 0.0);
    } else if (type == 3) {
        float c = dot(-l, normalize(light.direction.xyz));
        attenuation *= smoothstep(light.direction.w, light.params.z, c);
    }
    radiance *= attenuation;
    return attenuation > 0.0;
}

uint pointShadowFace(vec3 d) {
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.x && a.y >= a.z) return d.y >= 0.0 ? 2u : 3u;
    return d.z >= 0.0 ? 4u : 5u;
}

float evaluateShadow(uint recordIndex, int lightType, vec3 lightPosition,
                     vec3 p, vec3 n, vec3 l) {
    if ((sceneFlags & 1u) == 0u || recordIndex > PREVIEW_MAX_LIGHTS ||
        shadowRecords[recordIndex].meta.x == 0u) return 1.0;
    uint faceCount = min(shadowRecords[recordIndex].meta.z, 6u);
    uint face = lightType == 0 ? pointShadowFace(p - lightPosition) : 0u;
    vec3 biased = p + n * shadowRecords[recordIndex].params.y;
    vec3 ndc = vec3(0.0);
    vec2 localUv = vec2(0.0);
    bool found = false;
    if (lightType == 1 && faceCount > 1u) {
        for (uint cascade = 0u; cascade < 6u; ++cascade) {
            if (cascade >= faceCount) break;
            vec4 c = shadowRecords[recordIndex].viewProj[cascade] * vec4(biased, 1.0);
            if (c.w <= 0.0) continue;
            vec3 q = c.xyz / c.w;
            vec2 uv = q.xy * 0.5 + 0.5;
            if (q.z > 0.0 && q.z < 1.0 && all(greaterThanEqual(uv, vec2(0.0))) &&
                all(lessThanEqual(uv, vec2(1.0)))) {
                face = cascade; ndc = q; localUv = uv; found = true; break;
            }
        }
    } else if (face < faceCount) {
        vec4 c = shadowRecords[recordIndex].viewProj[face] * vec4(biased, 1.0);
        if (c.w > 0.0) {
            ndc = c.xyz / c.w;
            localUv = ndc.xy * 0.5 + 0.5;
            found = ndc.z > 0.0 && ndc.z < 1.0 &&
                all(greaterThanEqual(localUv, vec2(0.0))) &&
                all(lessThanEqual(localUv, vec2(1.0)));
        }
    }
    if (!found) return 1.0;
    vec4 rect = shadowRecords[recordIndex].atlasRect[face];
    vec2 atlasUv = rect.xy + localUv * rect.zw;
    float bias = shadowRecords[recordIndex].params.x *
                 max(0.25, 1.0 - max(dot(n, l), 0.0));
    float texel = shadowRecords[recordIndex].params.z;
    int radius = (pc.materialMeta.y & 0xffu) <= 1u ? 1 : 2;
    float lit = 0.0;
    float samples = 0.0;
    vec2 lo = rect.xy + vec2(texel * 0.5);
    vec2 hi = rect.xy + rect.zw - vec2(texel * 0.5);
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            if (abs(x) > radius || abs(y) > radius) continue;
            float stored = texture(previewShadowAtlas,
                clamp(atlasUv + vec2(x, y) * texel, lo, hi)).r;
            lit += ndc.z - bias <= stored ? 1.0 : 0.0;
            samples += 1.0;
        }
    }
    return lit / max(samples, 1.0);
}

RtPostParams sdfPostParams() {
    RtPostParams p;
    p.exposure = postA.x;
    p.gamma = postA.y;
    p.saturation = postA.z;
    p.colorTemperature = postA.w;
    p.vignetteStrength = postB.x;
    p.toneMapping = uint(postB.y + 0.5);
    p.vignetteEnabled = uint(postB.z + 0.5);
    p.cameraExposure = postB.w;
    return p;
}

void main() {
    mat4 invViewProj = inverse(pc.viewProj);
    vec4 nearH = invViewProj * vec4(vNdc, 0.0, 1.0);
    vec4 farH  = invViewProj * vec4(vNdc, 1.0, 1.0);
    vec3 nearP = nearH.xyz / nearH.w;
    vec3 farP  = farH.xyz / farH.w;
    vec3 ro = nearP;
    vec3 rd = normalize(farP - nearP);

    float nearestT = 1e30;
    vec3 nearestN = vec3(0.0, 1.0, 0.0);
    uint nearestVolume = ~0u;
    uint count = min(pc.materialMeta.w, 16u);
    for (uint vi = 0u; vi < count; ++vi) {
        // is_active @172, volume_type @168, source_type @428, grid @232.
        if (volWord(vi, 172u) == 0u || volWord(vi, 168u) != 2u ||
            int(volWord(vi, 428u)) != 4 || volAddress(vi, 232u) == uint64_t(0)) continue;
        float t;
        vec3 n;
        if (marchSurface(vi, ro, rd, nearestT, t, n) && t < nearestT) {
            nearestT = t;
            nearestN = n;
            nearestVolume = vi;
        }
    }
    if (nearestVolume == ~0u) discard;

    vec3 p = ro + rd * nearestT;
    vec4 clip = pc.viewProj * vec4(p, 1.0);
    float depth = clip.z / clip.w;
    if (clip.w <= 1e-6 || depth < -1e-5 || depth > 1.00001) discard;
    gl_FragDepth = clamp(depth, 0.0, 1.0);

    float materialSlot = volFloat(nearestVolume, 556u);
    bool hasMaterial = materialSlot >= 1.0 && uint(materialSlot - 1.0) < pc.materialMeta.x;
    uint mi = hasMaterial ? uint(materialSlot - 1.0) : 0u;
    GpuMaterial mat;
    MaterialExt matx;
    if (hasMaterial) mat = materials[mi];
    if (hasMaterial) matx = materialsExt[mi];
    vec3 albedo = hasMaterial ? vec3(mat.albedo_r, mat.albedo_g, mat.albedo_b)
                              : volVec3(nearestVolume, 88u);
    if (hasMaterial && mat.albedo_tex > 0u) {
        albedo = sampleTriplanar(mat.albedo_tex, p, nearestN,
            vec2(mat.uv_scale_x, mat.uv_scale_y),
            vec2(mat.uv_offset_x, mat.uv_offset_y)).rgb;
    }
    float roughness = hasMaterial ? clamp(mat.roughness, 0.0, 1.0)
                                  : clamp(volFloat(nearestVolume, 468u), 0.0, 1.0);
    float metallic = hasMaterial ? clamp(mat.metallic, 0.0, 1.0) : 0.0;
    float transmission = hasMaterial ? clamp(mat.transmission, 0.0, 1.0) : 1.0;
    float ior = hasMaterial ? max(mat.ior, 1.0001) : max(volFloat(nearestVolume, 464u), 1.0001);
    float specular = hasMaterial ? clamp(mat.specular, 0.0, 1.0) : 1.0;
    if (hasMaterial) {
        vec2 uvScale = vec2(mat.uv_scale_x, mat.uv_scale_y);
        vec2 uvOffset = vec2(mat.uv_offset_x, mat.uv_offset_y);
        if (mat.roughness_tex > 0u) {
            roughness = samplePackedRoughness(
                sampleTriplanar(mat.roughness_tex, p, nearestN, uvScale, uvOffset),
                roughness, mat.flags);
        }
        if (mat.metallic_tex > 0u) {
            metallic = samplePackedMetallic(
                sampleTriplanar(mat.metallic_tex, p, nearestN, uvScale, uvOffset),
                mat.flags);
        }
        if (mat.transmission_tex > 0u) {
            transmission *= sampleTriplanar(
                mat.transmission_tex, p, nearestN, uvScale, uvOffset).r;
        }
        if (mat.specular_tex > 0u) {
            specular *= sampleTriplanar(
                mat.specular_tex, p, nearestN, uvScale, uvOffset).r;
        }
    }

    vec3 n = normalize(nearestN);
    vec3 v = normalize(-rd);
    float ndv = max(dot(n, v), 1e-4);
    float dielectricF0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    vec3 f0 = mix(vec3(dielectricF0 * specular), albedo, metallic);
    vec3 direct = vec3(0.0);
    uint lightCount = min(sceneLightCount, 32u);
    for (uint li = 0u; li < lightCount; ++li) {
        vec3 l, radiance;
        if (!evaluateLight(li, p, l, radiance)) continue;
        float ndl = max(dot(n, l), 0.0);
        if (ndl <= 0.0) continue;
        vec3 h = normalize(v + l);
        float ndh = max(dot(n, h), 0.0);
        float vdh = max(dot(v, h), 0.0);
        vec3 f = fresnelSchlick(vdh, f0);
        vec3 spec = distributionGGX(ndh, max(roughness, 0.035)) *
                    geometrySmith(ndv, ndl, max(roughness, 0.035)) * f /
                    max(4.0 * ndv * ndl, 1e-5);
        vec3 kd = (vec3(1.0) - f) * (1.0 - metallic) * (1.0 - transmission);
        LightData sourceLight = sceneLights[li];
        int lightType = int(sourceLight.position.w + 0.5);
        float shadow = evaluateShadow(li, lightType, sourceLight.position.xyz,
                                      p, n, l);
        direct += (kd * albedo / PI + spec) * radiance * ndl * shadow;
    }

    // Physical Sky's sun is not duplicated in the scene-light SSBO. It owns
    // the reserved shadow record after the 32 scene-light slots.
    if (pc.materialMeta.z == 3u && worldMode == 2u && worldSun.w > 0.0) {
        vec3 l = normalize(worldSun.xyz);
        float ndl = max(dot(n, l), 0.0);
        if (ndl > 0.0) {
            vec3 h = normalize(v + l);
            vec3 f = fresnelSchlick(max(dot(v, h), 0.0), f0);
            vec3 spec = distributionGGX(max(dot(n, h), 0.0), max(roughness, 0.035)) *
                        geometrySmith(ndv, ndl, max(roughness, 0.035)) * f /
                        max(4.0 * ndv * ndl, 1e-5);
            vec3 kd = (vec3(1.0) - f) * (1.0 - metallic) * (1.0 - transmission);
            float shadow = evaluateShadow(PREVIEW_MAX_LIGHTS, 1, vec3(0.0), p, n, l);
            direct += (kd * albedo / PI + spec) *
                      vec3(1.0, 0.95, 0.86) * worldSun.w * ndl * shadow;
        }
    }

    vec3 reflected = reflect(rd, n);
    vec3 refracted = refract(rd, n, 1.0 / ior);
    bool tir = dot(refracted, refracted) < 1e-8;
    vec3 fresnel = fresnelSchlick(ndv, f0);
    vec3 envReflection = previewEnvironment(reflected, roughness);
    vec3 envTransmission = tir ? vec3(0.0) : previewEnvironment(normalize(refracted), roughness);
    float voxelDepth = max(volFloat(nearestVolume, 176u) * 2.0, 0.02);
    vec3 beer;
    if (hasMaterial && matx.transmission_density > 1e-4) {
        // Authored Interior Depth follows the same resin extinction model as
        // mesh transmission and Vulkan RT. Interior colour is the canonical
        // tint once depth is explicitly authored.
        vec3 interiorTint = vec3(
            matx.resin_color_r, matx.resin_color_g, matx.resin_color_b);
        beer = exp(-previewBeerExtinction(interiorTint) *
                   matx.transmission_density * voxelDepth);
    } else if (hasMaterial) {
        // RT depthless dielectric applies Base Color at both interfaces. This
        // bounded chord proxy makes the default dielectric useful for water
        // without inventing an opaque diffuse body.
        vec3 interiorDirection = tir ? rd : normalize(refracted);
        float interfaceDistance = max(voxelDepth, 1.30) /
            max(abs(dot(interiorDirection, -n)), 0.05);
        beer = pow(clamp(albedo, vec3(0.001), vec3(1.0)),
                   vec3(interfaceDistance));
    } else {
        vec3 absorption = max(volVec3(nearestVolume, 120u), vec3(0.0)) *
                          volFloat(nearestVolume, 132u);
        beer = exp(-absorption * voxelDepth);
    }
    bool sceneRefractionValid = false;
    vec3 sceneThrough = vec3(0.0);
    if (pc.lightDir0.w > 0.5 && !tir && transmission > 0.001 && metallic < 0.999) {
        float authoredDepth = hasMaterial ? max(matx.transmission_density, 0.0) : 0.0;
        float screenTravel = authoredDepth > 1e-4
            ? authoredDepth : max(voxelDepth * 6.0, 0.65);
        vec2 sceneUv;
        sceneRefractionValid = projectOpaqueRefraction(
            p, normalize(refracted), screenTravel, gl_FragDepth, sceneUv);
        if (sceneRefractionValid)
            sceneThrough = sampleOpaqueSceneRough(sceneUv, roughness);
    }
    vec3 transmissionWeight = beer * transmission *
                              (vec3(1.0) - fresnel) * (1.0 - metallic);
    vec3 env = envReflection * fresnel;
    if (!sceneRefractionValid) env += envTransmission * transmissionWeight;
    vec3 diffuseAmbient = previewEnvironment(n, 1.0) * albedo *
                          (1.0 - metallic) * (1.0 - transmission) * 0.35;
    vec3 emission = hasMaterial
        ? vec3(mat.emission_r, mat.emission_g, mat.emission_b) * mat.emission_strength
        : volVec3(nearestVolume, 136u) * volFloat(nearestVolume, 148u);
    if (hasMaterial && mat.emission_tex > 0u) {
        emission = sampleTriplanar(mat.emission_tex, p, nearestN,
            vec2(mat.uv_scale_x, mat.uv_scale_y),
            vec2(mat.uv_offset_x, mat.uv_offset_y)).rgb * mat.emission_strength;
    }
    vec3 color = direct + env + diffuseAmbient + emission;
    color = rtApplyPost(color, sdfPostParams(),
                        gl_FragCoord.xy / max(postC.xy, vec2(1.0)));
    // Opaque snapshot pixels already passed through the same display transform;
    // composite them afterwards to avoid applying tone mapping twice.
    if (sceneRefractionValid) color += sceneThrough * transmissionWeight;
    outColor = vec4(color, 1.0);
}
