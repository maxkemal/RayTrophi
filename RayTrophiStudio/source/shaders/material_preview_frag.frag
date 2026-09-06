#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "procedural_detail.glsl"
#include "pbr_texture_policy.glsl"
// ★★★ Goruntuleme donusumu artik BURADA TANIMLI DEGIL: tek tanim
//   post_chain.glsl icinde ve Rendered yolu (tonemap.comp) ayni dosyayi
//   kullaniyor. Buraya ikinci bir tonemap yazma.
#include "post_chain.glsl"

layout(location = 0) in vec3 vWorldNormal;
layout(location = 1) flat in uint vMaterialID;
layout(location = 2) in vec2 vTexCoord;
layout(location = 3) in vec3 vWorldPos;
layout(location = 4) in vec3 vObjectPos;
layout(location = 5) flat in vec3 vObjectOrigin;
layout(location = 6) flat in vec3 vWorldToObject0;
layout(location = 7) flat in vec3 vWorldToObject1;
layout(location = 8) flat in vec3 vWorldToObject2;

layout(location = 0) out vec4 outColor;

// Material buffer struct — single source of truth shared by every material-reading
// shader. This shader historically called the type `GpuMaterial`; alias it so the
// existing code (materials[i], GpuMaterial mat, function params) is unchanged.
#define GpuMaterial Material
#include "material_struct.glsl"

layout(set = 0, binding = 0, std430) readonly buffer MaterialBuffer {
    GpuMaterial materials[];
};
// COLD half of the split material record (see material_struct.glsl). The
// preview only touches it for micro-detail / sheen / SSS tint, loaded at the
// use sites via the macro below.
layout(set = 0, binding = 4, std430) readonly buffer MaterialExtBuffer {
    MaterialExt materialsExt[];
};
#define matx materialsExt[materialIndex]

// Texture array — same slots as RT pipeline binding 6
// Access guarded by albedo_tex/normal_tex > 0 checks.
layout(set = 0, binding = 1) uniform sampler2D textures[];

// The compiled Material Graph VM is shared with Vulkan RT. Only the binding
// number differs in this smaller descriptor set; the bytecode and evaluator
// are the same source. Texture ops use the preview's same bindless array.
#define MATERIAL_PROGRAM_BINDING 16
#define materialTextures textures
#include "material_program.glsl"
#include "material_preview_transmission.glsl"

// Baked equirectangular environment maps for specular reflection lookup.
// [0] = studio  [1] = outdoor  (128×64 RGBA32F, uploaded at pipeline init)
layout(set = 0, binding = 2) uniform sampler2D envMaps[2];

// Terrain layer SSBO — mirrors RT pipeline binding 12.
// Layout must match VkTerrainLayerData (112 bytes, 16-byte aligned).
// Every member is a 4-byte scalar or an array of them, so std430 and scalar
// layout agree here; do not add a vec3 without re-checking that.
struct TerrainLayerData {
    uint  layer_mat_id[8];   // 0–3 splat-weighted, 4–7 semantic overlays
    float layer_uv_scale[8]; // UV tiling per slot
    float overlay_strength[4]; // Artist dial for slots 4–7
    uint  splat_map_tex;     // RGBA splat map texture slot
    uint  layer_count;       // Active splat layers (1–4)
    uint  macro_color_tex;   // Macro color map slot
    float macro_color_strength; // Macro color blend strength
    uint  semantic_map_tex;  // R=Flow, G=Wetness, B=Ice, A=Hardness
    float semantic_wet_darkening;
    float semantic_wet_roughness;
    uint  overlay_mask;      // bit s = slot 4+s bound; bit 8+s = exempt from snow burial
};
layout(set = 0, binding = 3, std430) readonly buffer TerrainLayerBuffer {
    TerrainLayerData terrainLayers[];
};

// ---------------------------------------------------------------------------
// Sahne isiklari — RT hattinin BINDING 3'undeki BUFFER'IN AYNISI.
//
// ★★★ Ayri bir onizleme isik listesi CIKARILMADI, bilerek. Iki liste iki
//   dogruluk kaynagi demektir ve bu depoda o hata sinifinin adi var: panelin
//   yalan soylemesi. Ayni buffer okundugu icin onizleme ile Rendered arasinda
//   "hangi isiklar var" sorusunda AYRISMA MUMKUN DEGIL; ayrisabilecek tek sey
//   isigin nasil degerlendirildigidir, ve o da asagida acikca yaklasikliktir.
//
// Duzen VkGpuLight ile birebir (96 bayt); closesthit.rchit'teki LightData'nin
// ayni kopyasi.
struct LightData {
    vec4 position;    // xyz + type (0=point, 1=dir, 2=area, 3=spot)
    vec4 color;       // rgb + intensity
    vec4 params;      // radius, width, height/inner_cos, -
    vec4 direction;   // xyz + outer_cos
    vec4 area_u;      // xyz: AreaLight u ekseni (birim)
    vec4 area_v;      // xyz: AreaLight v ekseni (birim)
};
layout(set = 0, binding = 5, std430) readonly buffer PreviewLightBuffer {
    LightData sceneLights[];
};

// Onizleme sahne globalleri. ★★ Push constant'a EKLENMEDI: o blogun duzeni
// alti yerde yasiyor (iki GLSL, iki C++ struct, uc boyut ifadesi) ve bugun 208
// bayt, yani cihaz esigi zaten yuksek. Golge/world durumu descriptor-backed
// kayitlarda tutulur; ana material push ABI'si bu nedenle degismez.
layout(set = 0, binding = 6, std430) readonly buffer PreviewSceneGlobalsBuffer {
    uint sceneLightCount;
    uint sceneFlags;       // bit0=shadow, bit1=env, bit2=overlay, bit3=atmos LUT, bit4=sun shadow, bit5=HDRI IBL
    uint shadowedLightCount;
    uint worldMode;        // 0=color, 1=HDRI, 2=Nishita
    vec4 worldColor;       // rgb + color intensity
    vec4 worldParams;      // env/overlay rotation, intensity, atmosphere intensity, blend mode
    vec4 worldSun;         // xyz direction toward sun, w intensity
    vec4 atmosphereA;      // multi enabled, factor, mie anisotropy, mie density
    vec4 atmosphereB;      // planet radius, atmosphere height, reserved
    // ★★ post.* ayarlari. Atmosfer alanlariyla toplam ABI 144 bayt;
    //   C++ tarafindaki SceneGlobals
    //   static_assert'i, buffer boyutu ve memset ayni partide buyutuldu.
    //   Ayni buffer'i material_preview_sky.frag da okuyor -- gokyuzu ile
    //   nesnenin AYNI pozlamayi gormesi bu paylasimla garanti.
    vec4 postA;            // x=exposure y=gamma z=saturation w=colorTemperature
    vec4 postB;            // x=vignetteStrength y=toneMappingType z=vignetteEnabled w=(bos)
    vec4 postC;            // x=viewportWidth y=viewportHeight (vignette icin)
};

RtPostParams previewPostParams() {
    RtPostParams p;
    p.exposure         = postA.x;
    p.gamma            = postA.y;
    p.saturation       = postA.z;
    p.colorTemperature = postA.w;
    p.vignetteStrength = postB.x;
    p.toneMapping      = uint(postB.y + 0.5);
    p.vignetteEnabled  = uint(postB.z + 0.5);
    // ★ postB.w eskiden bostu; kamera pozlamasi oraya girdi, yani globals
    //   ABI'si (112 bayt) DEGISMEDI.
    p.cameraExposure   = postB.w;
    return p;
}

const uint kPreviewMaxSceneLights = 32u;
struct PreviewShadowRecord {
    mat4 viewProj[6];
    vec4 atlasRect[6];     // xy offset, zw scale
    uvec4 meta;            // x=valid, y=light type, z=face count, w=light index
    vec4 params;           // depth bias, normal bias, atlas texel, PCF radius
};
layout(set = 0, binding = 7, std430) readonly buffer PreviewShadowBuffer {
    PreviewShadowRecord shadowRecords[kPreviewMaxSceneLights + 1u];
};
layout(set = 0, binding = 8) uniform sampler2D previewShadowAtlas;
layout(set = 0, binding = 9) uniform sampler2D worldEnvironment;
layout(set = 0, binding = 10) uniform sampler2D atmosphereTransmittance;
layout(set = 0, binding = 11) uniform sampler2D atmosphereSkyView;
layout(set = 0, binding = 12) uniform sampler2D atmosphereMultiScatter;
layout(set = 0, binding = 13) uniform sampler2D worldIrradiance;
layout(set = 0, binding = 14) uniform sampler2D worldPrefiltered;
layout(set = 0, binding = 15) uniform sampler2D worldBrdfLut;
// Frozen after the opaque pass. These never alias the live color/depth
// attachments while sampled by the dedicated transmission replay.
layout(set = 0, binding = 17) uniform sampler2D previewOpaqueColor;
layout(set = 0, binding = 18) uniform sampler2D previewOpaqueDepth;
layout(set = 0, binding = 19, r32ui) uniform uimage2D previewTransmissionBackDepth;

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;
    vec4 lightDir0;   // key light
    vec4 lightDir1;   // fill light
    vec4 lightDir2;   // rim light
    // y low 8 bits = quality; bits 8..9 = draw phase
    // (0 combined/backward-compatible, 1 opaque, 2 transmission).
    uvec4 materialMeta; // x = material count, z = lighting preset
} pc;

const uint PREVIEW_PHASE_COMBINED     = 0u;
const uint PREVIEW_PHASE_OPAQUE       = 1u;
const uint PREVIEW_PHASE_TRANSMISSION = 2u;

uint previewQualityMode() { return pc.materialMeta.y & 0xffu; }
uint previewDrawPhase()   { return (pc.materialMeta.y >> 8u) & 0x3u; }

// Onizleme icin sahne isigi degerlendirmesi.
//
// ★★★ Bu bir YAKLASIKLIKTIR ve nerede yaklastigi burada yazili: alan isigi
//   MERKEZINDEN ornekleniyor (RT rastgele bir nokta secip MIS uyguluyor), ve
//   paylasilan atlas golgesi TEK MERKEZ ORNEGINI izler. Yon, dusum (falloff),
//   koni ve siddet semantigi ise
//   bsdf_scatter.glsl'deki sample_light_direction_gl ile BIREBIR ayni --
//   cunku yanlis olsa gozle gorunur bicimde yanlis olur (ters yonden
//   aydinlatma), oysa golgesizlik "eksik" olarak gorunur.
//
// ★ Yonlu isikte direction.xyz ISIGA DOGRU bakar: yukleme tarafi
//   (VulkanBackend setLights) yonu zaten negatifliyor. Burada bir kez daha
//   negatiflemek sahneyi ters taraftan aydinlatirdi.
bool evalSceneLight(uint idx, vec3 P, out vec3 L, out vec3 radiance) {
    LightData lt = sceneLights[idx];
    int type = int(lt.position.w + 0.5);
    vec3 base = lt.color.rgb * lt.color.w;

    if (type == 1) {                       // Directional
        L = normalize(lt.direction.xyz);
        radiance = base;
        return true;
    }

    vec3 toLight = lt.position.xyz - P;
    float d2 = dot(toLight, toLight);
    if (d2 < 1e-6) return false;
    L = toLight * inversesqrt(d2);
    float atten = 1.0 / d2;

    if (type == 2) {                       // Area: merkez ornegi, tek yuzlu
        vec3 ln = normalize(cross(lt.area_u.xyz, lt.area_v.xyz));
        // RT tarafinda cos/d^2 attenuation'i pdf = 1/alan ile bolunur, yani
        // etkin carpan cos * alan / d^2 olur. Ayni carpani burada dogrudan
        // uyguluyoruz ki iki hat AYNI parlaklik mertebesinde olsun.
        atten *= max(dot(-L, ln), 0.0) * max(lt.params.y * lt.params.z, 0.0);
    } else if (type == 3) {                // Spot
        float cosTheta = dot(-L, normalize(lt.direction.xyz));
        float inner = lt.params.z;
        float outer = lt.direction.w;
        float f;
        if (cosTheta < outer)      f = 0.0;
        else if (cosTheta > inner) f = 1.0;
        else { float t = (cosTheta - outer) / (inner - outer + 1e-6); f = t * t; }
        atten *= f;
    }

    if (atten <= 0.0) return false;
    radiance = base * atten;
    return true;
}

uint pointShadowFace(vec3 d) {
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.x && a.y >= a.z) return d.y >= 0.0 ? 2u : 3u;
    return d.z >= 0.0 ? 4u : 5u;
}

float evaluatePreviewShadow(uint lightIndex, vec3 P, vec3 N, vec3 L) {
    if ((sceneFlags & 1u) == 0u || lightIndex > kPreviewMaxSceneLights) return 1.0;
    if (shadowRecords[lightIndex].meta.x == 0u) return 1.0;

    uint lightType = shadowRecords[lightIndex].meta.y;
    uint faceCount = min(shadowRecords[lightIndex].meta.z, 6u);
    uint face = (lightIndex < kPreviewMaxSceneLights && lightType == 0u)
        ? pointShadowFace(P - sceneLights[lightIndex].position.xyz) : 0u;
    if (face >= faceCount) return 1.0;

    vec3 biasedP = P + N * shadowRecords[lightIndex].params.y;
    vec4 clip = vec4(0.0);
    vec3 ndc = vec3(0.0);
    vec2 localUV = vec2(0.0);
    bool projectionFound = false;
    // Directional lights and Physical Sky sun store near-to-far cascades in
    // the existing six-face record. Select the smallest projection containing
    // the receiver; no shadow ABI growth or split-distance buffer is needed.
    if (lightType == 1u && faceCount > 1u) {
        for (uint cascade = 0u; cascade < 6u; ++cascade) {
            if (cascade >= faceCount) break;
            vec4 candidateClip = shadowRecords[lightIndex].viewProj[cascade] *
                                 vec4(biasedP, 1.0);
            if (candidateClip.w <= 0.0) continue;
            vec3 candidateNdc = candidateClip.xyz / candidateClip.w;
            vec2 candidateUV = candidateNdc.xy * 0.5 + 0.5;
            bool inside = candidateNdc.z > 0.0 && candidateNdc.z < 1.0 &&
                all(greaterThanEqual(candidateUV, vec2(0.0))) &&
                all(lessThanEqual(candidateUV, vec2(1.0)));
            if (inside) {
                face = cascade;
                clip = candidateClip;
                ndc = candidateNdc;
                localUV = candidateUV;
                projectionFound = true;
                break;
            }
        }
    } else {
        clip = shadowRecords[lightIndex].viewProj[face] * vec4(biasedP, 1.0);
        if (clip.w > 0.0) {
            ndc = clip.xyz / clip.w;
            localUV = ndc.xy * 0.5 + 0.5;
            projectionFound = ndc.z > 0.0 && ndc.z < 1.0 &&
                all(greaterThanEqual(localUV, vec2(0.0))) &&
                all(lessThanEqual(localUV, vec2(1.0)));
        }
    }
    if (!projectionFound) return 1.0;

    vec4 rect = shadowRecords[lightIndex].atlasRect[face];
    vec2 atlasUV = rect.xy + localUV * rect.zw;
    float bias = shadowRecords[lightIndex].params.x *
                 max(0.25, 1.0 - max(dot(N, L), 0.0));
    float lit = 0.0;
    float radius = max(shadowRecords[lightIndex].params.w, 1.0);
    vec2 halfTexel = vec2(shadowRecords[lightIndex].params.z * 0.5);
    vec2 rectMin = rect.xy + halfTexel;
    vec2 rectMax = rect.xy + rect.zw - halfTexel;
    // Quality changes receiver filtering only. Atlas allocation and caster
    // submissions remain bounded by the CPU-side preset budget.
    int kernelHalfWidth = previewQualityMode() >= 3u ? 2 : 1;
    float sampleCount = float((kernelHalfWidth * 2 + 1) *
                              (kernelHalfWidth * 2 + 1));
    for (int y = -kernelHalfWidth; y <= kernelHalfWidth; ++y) {
        for (int x = -kernelHalfWidth; x <= kernelHalfWidth; ++x) {
            vec2 uv = clamp(atlasUV + vec2(x, y) *
                            shadowRecords[lightIndex].params.z * radius,
                            rectMin, rectMax);
            float storedDepth = texture(previewShadowAtlas, uv).r;
            lit += (ndc.z - bias <= storedDepth) ? 1.0 : 0.0;
        }
    }
    return lit / sampleCount;
}

vec2 worldDirToUV(vec3 d) {
    d = normalize(d);
    float phi = atan(d.z, d.x) - worldParams.x;
    return vec2(fract(phi / (2.0 * 3.14159265359) + 0.5),
                acos(clamp(d.y, -1.0, 1.0)) / 3.14159265359);
}

vec3 sampleCanonicalWorld(vec3 d) {
    d = normalize(d);
    if (worldMode == 1u) {
        if ((sceneFlags & 2u) != 0u)
            return texture(worldEnvironment, worldDirToUV(d)).rgb * max(worldParams.y, 0.0);
        return vec3(0.0);
    }

    if (worldMode == 2u) {
        vec3 result;
        if ((sceneFlags & 8u) != 0u) {
            float azimuth = atan(d.z, d.x) / (2.0 * 3.14159265359);
            if (azimuth < 0.0) azimuth += 1.0;
            result = texture(atmosphereSkyView,
                             vec2(azimuth, (1.0 - clamp(d.y, -1.0, 1.0)) * 0.5)).rgb;
            // Match miss.rmiss: the LUT is single scatter; RT adds bounded
            // second/third order scatter before exposing it as sky radiance.
            if (atmosphereA.x > 0.5) {
                vec3 scatteringAlbedo = vec3(0.8, 0.85, 0.9);
                vec3 secondOrder = result * scatteringAlbedo * 0.5 * exp(-0.5 * 0.3);
                vec3 thirdOrder = secondOrder * scatteringAlbedo * 0.25 * exp(-0.5 * 0.1);
                result += secondOrder * atmosphereA.y +
                          thirdOrder * (atmosphereA.y * 0.5);
            }
        } else {
            float up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
            vec3 horizon = vec3(0.42, 0.53, 0.68);
            vec3 zenith = vec3(0.09, 0.24, 0.52);
            vec3 ground = max(worldColor.rgb, vec3(0.025));
            vec3 sky = mix(horizon, zenith, pow(up, 0.65));
            result = mix(ground, sky, smoothstep(0.0, 0.12, d.y)) *
                     max(worldParams.z / 10.0, 0.0);
        }
        vec3 sunDir = dot(worldSun.xyz, worldSun.xyz) > 1e-8
            ? normalize(worldSun.xyz) : vec3(0.0, 1.0, 0.0);
        float sunSize = max(worldColor.w, 0.05);
        float elevation = degrees(asin(clamp(sunDir.y, -1.0, 1.0)));
        if (elevation < 15.0)
            sunSize *= 1.0 + (15.0 - max(elevation, -10.0)) * 0.04;
        float sunRadius = radians(sunSize * 0.5);
        float mu = dot(d, sunDir);
        if (mu > cos(sunRadius) && worldSun.w > 0.0) {
            float radial = acos(clamp(mu, -1.0, 1.0)) / max(sunRadius, 1e-6);
            float limb = 1.0 - 0.6 * (1.0 - sqrt(max(0.0, 1.0 - radial * radial)));
            float edge = 1.0 - smoothstep(0.85, 1.0, radial);
            vec3 transSun = vec3(1.0);
            if ((sceneFlags & 8u) != 0u) {
                float u = clamp((max(0.01, sunDir.y) + 0.2) / 1.2, 0.0, 1.0);
                float radius = max(atmosphereB.x, 1.0);
                float altitude = max(0.0, length(pc.cameraPos.xyz + vec3(0.0, radius, 0.0)) - radius);
                float v = clamp(altitude / max(atmosphereB.y, 1.0), 0.0, 1.0);
                transSun = texture(atmosphereTransmittance, vec2(u, v)).rgb;
            }
            result += transSun * worldSun.w * 80000.0 * limb * edge;
        }
        if ((sceneFlags & 4u) != 0u) {
            vec3 sampled = texture(worldEnvironment, worldDirToUV(d)).rgb;
            float strength = max(worldParams.y, 0.0);
            float amount = min(strength, 1.0);
            vec3 overlay = sampled * strength;
            int blendMode = int(worldParams.w + 0.5);
            if (blendMode == 1) result *= mix(vec3(1.0), sampled, amount);
            else if (blendMode == 2) result += overlay;
            else if (blendMode == 3) result = overlay;
            else result = mix(result, overlay, amount);
        }
        return result;
    }
    return max(worldColor.rgb * worldColor.w, vec3(0.0));
}

// RT integrates the environment over the GGX/cosine lobe. A single bent
// lookup (mix(R,N,roughness^2)) over-selects the blue zenith of Physical Sky,
// especially on broad metallic highlights. Use a small deterministic cone
// integration for Nishita only; HDRI has its prefiltered maps below.
vec3 samplePhysicalSkyFiltered(vec3 axis, float roughness) {
    axis = normalize(axis);
    if (worldMode != 2u || roughness <= 0.025)
        return sampleCanonicalWorld(axis);

    vec3 helper = abs(axis.y) < 0.95 ? vec3(0.0, 1.0, 0.0)
                                      : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, axis));
    vec3 bitangent = cross(axis, tangent);
    int tapCount = previewQualityMode() >= 2u ? 8 : 4;
    float cone = clamp(roughness * roughness * 1.35, 0.035, 1.25);
    vec3 sum = sampleCanonicalWorld(axis) * 2.0;
    float weight = 2.0;
    const float goldenAngle = 2.39996323;
    for (int i = 0; i < 8; ++i) {
        if (i >= tapCount) break;
        float radius = sqrt((float(i) + 0.5) / float(tapCount)) * cone;
        float phi = float(i) * goldenAngle;
        vec3 ring = tangent * cos(phi) + bitangent * sin(phi);
        vec3 direction = normalize(axis + ring * radius);
        float tapWeight = max(dot(axis, direction), 0.05);
        sum += sampleCanonicalWorld(direction) * tapWeight;
        weight += tapWeight;
    }
    return sum / max(weight, 1e-5);
}

vec3 canonicalWorldSunRadiance(vec3 sunDir) {
    vec3 tint = vec3(1.0, 0.95, 0.86);
    if ((sceneFlags & 8u) != 0u) {
        float u = clamp((max(0.01, sunDir.y) + 0.2) / 1.2, 0.0, 1.0);
        tint = texture(atmosphereTransmittance, vec2(u, 0.0)).rgb;
    }
    return tint * max(worldSun.w, 0.0);
}

bool validTexture(uint textureId) {
    uint textureCount = max(pc.materialMeta.w, 1u);
    return textureId > 0u && textureId < textureCount;
}

// ── Helpers ──

const float PI = 3.14159265359;

// ★ `acesTonemap` ve `linearToSRGB` BURADAN SILINDI. Ikisi de post_chain.glsl'e
//   tasindi ve artik Rendered yolu ile paylasiliyor. Not: buradaki eski
//   `linearToSRGB` aslinda sRGB DEGIL duz pow(1/2.2) idi -- yani adi da
//   yaptigi isi yanlis soyluyordu.

float D_GGX(float NdotH, float roughness) {
    float a = max(roughness * roughness, 0.0025);
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265359 * d * d, 1e-5);
}

float G_SchlickGGX(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotX / max(NdotX * (1.0 - k) + k, 1e-5);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return F0 + (vec3(1.0) - F0) * x5;
}

// Same bounded optical-path-difference model used by Vulkan RT clearcoat.
// Direct lights receive the same V.H input; environment lighting uses N.V as
// its split-sum approximation because no single half-vector exists there.
vec3 clearcoatFilmTint(float angularCosine, float iridescence,
                       float filmThickness) {
    float amount = clamp(iridescence, 0.0, 1.0);
    if (amount <= 0.001) return vec3(1.0);
    float opd = filmThickness / max(angularCosine, 0.15);
    vec3 filmColor = vec3(
        0.55 + 0.45 * cos(opd * 6.2831853),
        0.55 + 0.45 * cos(opd * 6.2831853 + 2.0944),
        0.55 + 0.45 * cos(opd * 6.2831853 + 4.18879));
    return mix(vec3(1.0), filmColor, amount);
}

// Ambient Fresnel with roughness correction (Lagarde 2012).
// Rough surfaces saturate toward F0 — the sharp Fresnel rim fades out.
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    vec3 envelope = max(vec3(1.0 - roughness), F0);
    return F0 + (envelope - F0) * x5;
}

// Sheen lobe — Charlie distribution (fabric/velvet rim highlight).
// Matches Blender Principled BSDF sheen behaviour.
float D_Charlie(float NdotH, float roughness) {
    float invR  = 1.0 / max(roughness * roughness, 1e-4);
    float sin2h = max(1.0 - NdotH * NdotH, 1e-4);
    return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * 3.14159265359);
}

vec3 evaluateSheen(vec3 N, vec3 V, vec3 L, vec3 sheenColor, float sheenRoughness) {
    vec3  H     = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);
    float D  = D_Charlie(NdotH, max(sheenRoughness, 0.07));
    // Neubelt visibility term
    float Vs = 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV));
    return sheenColor * D * Vs * NdotL;
}

vec3 evaluateSpecularGGX(vec3 N, vec3 V, vec3 L, vec3 F0, float roughness) {
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);
    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    vec3  F = fresnelSchlick(VdotH, F0);
    return (D * G * F) / max(4.0 * NdotV * NdotL, 1e-5);
}

vec3 sampleStudioEnvironment(vec3 dir) {
    float up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 base = mix(vec3(0.07, 0.07, 0.08), vec3(0.42, 0.44, 0.46), smoothstep(0.05, 1.0, up));
    float softboxKey = pow(max(dot(dir, normalize(vec3(0.62, 0.54, 0.56))), 0.0), 28.0);
    float softboxFill = pow(max(dot(dir, normalize(vec3(-0.52, 0.38, 0.76))), 0.0), 20.0);
    float rimStrip = pow(max(dot(dir, normalize(vec3(-0.18, 0.26, -0.95))), 0.0), 56.0);
    base += vec3(1.00, 0.98, 0.95) * softboxKey * 1.8;
    base += vec3(0.72, 0.80, 0.96) * softboxFill * 0.9;
    base += vec3(0.95, 0.96, 1.00) * rimStrip * 0.7;
    return base;
}

vec3 sampleOutdoorEnvironment(vec3 dir) {
    float up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky = mix(vec3(0.22, 0.26, 0.32), vec3(0.55, 0.70, 0.96), smoothstep(0.1, 1.0, up));
    vec3 ground = vec3(0.12, 0.10, 0.08);
    vec3 env = mix(ground, sky, up);
    float sun = pow(max(dot(dir, normalize(vec3(0.32, 0.82, 0.46))), 0.0), 220.0);
    env += vec3(1.0, 0.96, 0.82) * sun * 2.2;
    return env;
}

vec3 samplePreviewEnvironment(vec3 dir, uint lightingPreset) {
    if (lightingPreset == 2u) {
        return sampleOutdoorEnvironment(dir);
    }
    return sampleStudioEnvironment(dir);
}

// Raster equivalent of the RT glass continuation lobe. RT samples one GGX
// microfacet direction per path; preview integrates a small deterministic cone
// so rough glass converges immediately and quality changes sample count, not
// lobe energy. Physical Sky/HDRI use the canonical world reader while the two
// studio presets keep their analytical environments.
vec3 sampleTransmissionEnvironment(vec3 axis, float roughness,
                                   uint lightingPreset) {
    axis = normalize(axis);
    bool canonical = lightingPreset == 3u;
    if (roughness <= 0.001)
        return canonical ? sampleCanonicalWorld(axis)
                         : samplePreviewEnvironment(axis, lightingPreset);

    vec3 helper = abs(axis.y) < 0.95 ? vec3(0.0, 1.0, 0.0)
                                      : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, axis));
    vec3 bitangent = cross(axis, tangent);
    int tapCount = previewQualityMode() >= 3u ? 8
                 : (previewQualityMode() >= 2u ? 6 : 4);
    float cone = clamp(roughness * roughness * 1.15, 0.015, 1.10);
    vec3 sum = (canonical ? sampleCanonicalWorld(axis)
                          : samplePreviewEnvironment(axis, lightingPreset)) * 2.0;
    float weight = 2.0;
    const float goldenAngle = 2.39996323;
    for (int i = 0; i < 8; ++i) {
        if (i >= tapCount) break;
        float radius = sqrt((float(i) + 0.5) / float(tapCount)) * cone;
        float phi = float(i) * goldenAngle;
        vec3 ring = tangent * cos(phi) + bitangent * sin(phi);
        vec3 direction = normalize(axis + ring * radius);
        float tapWeight = max(dot(axis, direction), 0.05);
        sum += (canonical ? sampleCanonicalWorld(direction)
                          : samplePreviewEnvironment(direction, lightingPreset)) * tapWeight;
        weight += tapWeight;
    }
    return sum / max(weight, 1e-5);
}

// Same F-C spread used by scatterGlass(): red bends less, blue bends more.
// RT chooses one persistent hero wavelength; raster evaluates all three
// channels deterministically, which is the converged spectral result.
vec3 sampleDispersedTransmission(vec3 incoming, vec3 normal, float ior,
                                float dispersion, float roughness,
                                uint lightingPreset, out bool allTir) {
    float spread = (ior - 1.0) * dispersion * 0.06;
    vec3 channelIor = max(vec3(ior - spread, ior, ior + spread), vec3(1.0001));
    vec3 spectral = vec3(0.0);
    allTir = true;
    for (int channel = 0; channel < 3; ++channel) {
        vec3 direction = refract(incoming, normal, 1.0 / channelIor[channel]);
        bool tir = dot(direction, direction) < 1e-8;
        allTir = allTir && tir;
        if (tir)
            direction = reflect(incoming, normal);
        vec3 sampleValue = sampleTransmissionEnvironment(
            normalize(direction), roughness, lightingPreset);
        spectral[channel] = sampleValue[channel];
    }
    return spectral;
}

bool previewProjectRefraction(vec3 origin, vec3 direction, float distance,
                              out vec2 uv) {
    vec4 clip = pc.viewProj * vec4(origin + direction * distance, 1.0);
    if (clip.w <= 1e-5) { uv = vec2(0.5); return false; }
    vec2 ndc = clip.xy / clip.w;
    uv = ndc * 0.5 + 0.5;
    return all(greaterThanEqual(uv, vec2(0.002))) &&
           all(lessThanEqual(uv, vec2(0.998)));
}

// Trace the frozen opaque depth buffer along a reflected world-space ray.
// This is deliberately bounded: it supplies scene detail that an environment
// lookup cannot contain, while off-screen/missed rays retain the canonical
// world fallback. Quadratic spacing keeps useful precision close to the glass
// without turning a full-screen transparent object into a TDR-sized workload.
bool tracePreviewOpaqueReflection(vec3 origin, vec3 direction,
                                  float maxDistance, out vec2 hitUv) {
    int steps = previewQualityMode() >= 3u ? 16
              : (previewQualityMode() >= 2u ? 12 : 8);
    float previousDelta = -1.0;
    bool havePrevious = false;
    for (int step = 0; step < 16; ++step) {
        if (step >= steps) break;
        float u = (float(step) + 1.0) / float(steps);
        float distance = maxDistance * u * u;
        vec4 clip = pc.viewProj * vec4(origin + direction * distance, 1.0);
        if (clip.w <= 1e-5) break;
        vec3 ndc = clip.xyz / clip.w;
        vec2 uv = ndc.xy * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.002))) ||
            any(greaterThan(uv, vec2(0.998)))) break;

        float opaqueDepth = texture(previewOpaqueDepth, uv).r;
        if (opaqueDepth >= 0.999999) {
            havePrevious = false;
            continue;
        }
        float delta = ndc.z - opaqueDepth;
        float depthTolerance = mix(0.0008, 0.006, u);
        bool crossedSurface = havePrevious && previousDelta < 0.0 && delta >= 0.0;
        if ((delta >= -depthTolerance && delta <= depthTolerance) || crossedSurface) {
            hitUv = uv;
            return true;
        }
        previousDelta = delta;
        havePrevious = true;
    }
    hitUv = vec2(0.5);
    return false;
}

vec3 sampleOpaqueSceneRough(vec2 uv, float roughness) {
    vec2 texel = 1.0 / max(postC.xy, vec2(1.0));
    float radius = roughness * roughness * 10.0;
    vec3 sum = texture(previewOpaqueColor, uv).rgb * 2.0;
    float weight = 2.0;
    int taps = previewQualityMode() >= 3u ? 8 :
               (previewQualityMode() >= 2u ? 6 : 4);
    const float goldenAngle = 2.39996323;
    for (int i = 0; i < 8; ++i) {
        if (i >= taps) break;
        float a = float(i) * goldenAngle;
        float r = sqrt((float(i) + 0.5) / float(taps)) * radius;
        vec2 q = clamp(uv + vec2(cos(a), sin(a)) * texel * r,
                       vec2(0.002), vec2(0.998));
        sum += texture(previewOpaqueColor, q).rgb;
        weight += 1.0;
    }
    return sum / weight;
}

// Convert a direction vector to equirectangular UV (matches CPU bake convention).
// u: longitude −π…π → 0…1   v: latitude 0…π (top→bottom) → 0…1
vec2 dirToEquirect(vec3 d) {
    return vec2(
        atan(d.z, d.x) / (2.0 * 3.14159265359) + 0.5,
        acos(clamp(d.y, -1.0, 1.0)) / 3.14159265359
    );
}

// Sample the baked env map for specular reflections.
// Uses the prefiltered 128×64 texture instead of the analytical approximation,
// giving much better results on metallic / glossy / clearcoat surfaces.
vec3 sampleEnvSpecular(vec3 reflectDir, uint lightingPreset) {
    uint envIdx = (lightingPreset == 2u) ? 1u : 0u;
    return texture(envMaps[envIdx], dirToEquirect(reflectDir)).rgb;
}

// Apply material UV transform: scale → rotate → offset → tiling
vec2 applyUVTransform(vec2 originalUV, const GpuMaterial mat) {
    vec2 uv = originalUV - vec2(0.5);

    // Scale
    float sx = (mat.uv_scale_x != 0.0) ? mat.uv_scale_x : 1.0;
    float sy = (mat.uv_scale_y != 0.0) ? mat.uv_scale_y : 1.0;
    uv *= vec2(sx, sy);

    // Rotation
    if (mat.uv_rotation_degrees != 0.0) {
        float angle = mat.uv_rotation_degrees * (3.14159265359 / 180.0);
        float c = cos(angle), s = sin(angle);
        uv = vec2(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
    }

    // Offset and Pivot
    uv += vec2(0.5);
    uv += vec2(mat.uv_offset_x, mat.uv_offset_y);

    // Tiling
    float tx = (mat.uv_tiling_x != 0.0) ? mat.uv_tiling_x : 1.0;
    float ty = (mat.uv_tiling_y != 0.0) ? mat.uv_tiling_y : 1.0;
    uv *= vec2(tx, ty);

    return uv;
}

// Derivative-based TBN (tangent from screen-space partial derivatives)
// Returns a new perturbed normal using the normal map sample.
vec3 applyNormalMap(vec3 N, vec3 worldPos, vec2 uv, vec3 nmSample, float strength, uint matFlags) {
    vec3 dp1  = dFdx(worldPos);
    vec3 dp2  = dFdy(worldPos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);

    float det = duv1.x * duv2.y - duv2.x * duv1.y;
    // Only reject truly degenerate geometry (UV singularity / zero-area triangle).
    // 1e-7 was too coarse — at close range dFdx/dFdy shrink legitimately
    // and det drops below that threshold even though the normal map is valid.
    if (abs(det) < 1e-20) return N;

    float invDet = 1.0 / det;
    vec3 T = normalize((duv2.y * dp1 - duv1.y * dp2) * invDet);
    vec3 B = normalize((-duv2.x * dp1 + duv1.x * dp2) * invDet);

    // The vertex shader flips V for texture sampling (1-V), so dFdy(uv)
    // has its V component negated. This inverts B relative to the UV space
    // the normal map was authored in. Negate B to compensate.
    B = -B;

    // Decode tangent-space normal — handles BC5 Z reconstruction when bit 11 set.
    vec3 tNormal = decodeNormalMapSample(nmSample, matFlags);
    tNormal.xy  *= max(strength, 0.0);
    tNormal       = normalize(tNormal);

    return normalize(mat3(T, B, N) * tNormal);
}

void main() {
    vec3 N = normalize(vWorldNormal);
    uint qualityMode = previewQualityMode();
    uint lightingPreset = pc.materialMeta.z;

    uint materialCount = max(pc.materialMeta.x, 1u);
    // ★★★ Materyal ID'sinin 31. biti = IMPOSTOR (scatter proxy seridi).
    //   Push constant'a yeni bir alan EKLENMEDI, bilerek: bu bloğun duzeni
    //   ALTI yerde yasiyor (iki GLSL, iki C++ struct, uc boyut ifadesi) ve
    //   buyutmek ayrica cihaz esigini 208'den 224 bayta cikarirdi. Impostor
    //   olmak cizimin degil GEOMETRININ ozelligi oldugu icin vertex basina
    //   tasinmasi zaten daha dogru.
    //   ★★ Maskelemeyi unutan bir tuketici materialCount-1'e KIRPILIR, yani
    //   sessizce YANLIS materyal okur. Bu attribute'u okuyan baska yer yok;
    //   eklenirse maskeyi de eklemek zorunda.
    bool isImpostor = (vMaterialID & 0x80000000u) != 0u;
    uint materialIndex = min(vMaterialID & 0x7FFFFFFFu, materialCount - 1u);
    GpuMaterial mat = materials[materialIndex];

    vec2 uv = applyUVTransform(vTexCoord, mat);

    // ── Procedural tile-break (independent slider, applied before texture sampling) ──
    // Breaks visible UV tiling seams. Separate from dirt/roughness so albedo maps
    // that shouldn't be warped can leave this at 0.
    if (mat.tile_break_strength > 0.0 &&
        (validTexture(mat.albedo_tex) || validTexture(mat.roughness_tex) || validTexture(mat.normal_tex))) {
        uv = pd_tileBreak(uv, vWorldPos, mat.tile_break_strength);
    }

    // ── Terrain Splat-Layer Blending (FLAG_TERRAIN = bit 16) ──
    // Mirrors closesthit.rchit logic: blends up to 4 material layers using an RGBA
    // splat map, then overrides albedo / roughness / metallic / normal before the
    // standard per-material texture sampling below.
    const uint FLAG_TERRAIN = (1u << 16);
    if ((mat.flags & FLAG_TERRAIN) != 0u) {
        uint layerIdx = mat._terrain_layer_idx;
        TerrainLayerData tl = terrainLayers[layerIdx];
        if (validTexture(tl.splat_map_tex) && tl.layer_count > 0u) {
            // R=layer0, G=layer1, B=layer2, A=layer3
            vec4 splatW = texture(textures[nonuniformEXT(tl.splat_map_tex)], uv);
            float weights[4];
            weights[0] = splatW.r;
            weights[1] = splatW.g;
            weights[2] = splatW.b;
            weights[3] = splatW.a;
            float totalW = weights[0]+weights[1]+weights[2]+weights[3];
            if (totalW < 0.001) totalW = 1.0;
            for (int k = 0; k < 4; k++) weights[k] /= totalW;

            vec3  blendAlbedo    = vec3(0.0);
            float blendRoughness = 0.0;
            float blendMetallic  = 0.0;
            vec3  blendNormal_ts = vec3(0.0);
            bool  anyNormalTex   = false;

            uint activeCount = min(tl.layer_count, 4u);
            for (uint k = 0u; k < activeCount; k++) {
                if (weights[k] < 0.001) continue;
                GpuMaterial lm = materials[min(tl.layer_mat_id[k], materialCount - 1u)];
                vec2 layerUV = uv * tl.layer_uv_scale[k];
                // apply per-layer UV transform
                float lsx = (lm.uv_scale_x != 0.0) ? lm.uv_scale_x : 1.0;
                float lsy = (lm.uv_scale_y != 0.0) ? lm.uv_scale_y : 1.0;
                layerUV *= vec2(lsx, lsy);
                layerUV += vec2(lm.uv_offset_x, lm.uv_offset_y);

                vec3 lAlbedo = max(vec3(lm.albedo_r, lm.albedo_g, lm.albedo_b), vec3(0.0));
                if (validTexture(lm.albedo_tex))
                    lAlbedo = texture(textures[nonuniformEXT(lm.albedo_tex)], layerUV).rgb;
                blendAlbedo += weights[k] * lAlbedo;

                float lRough = clamp(lm.roughness, 0.0, 1.0);
                if (validTexture(lm.roughness_tex))
                    lRough = texture(textures[nonuniformEXT(lm.roughness_tex)], layerUV).g;
                blendRoughness += weights[k] * lRough;

                float lMetal = clamp(lm.metallic, 0.0, 1.0);
                if (validTexture(lm.metallic_tex))
                    lMetal = texture(textures[nonuniformEXT(lm.metallic_tex)], layerUV).b;
                blendMetallic += weights[k] * lMetal;

                if (validTexture(lm.normal_tex)) {
                    vec3 ns = decodeNormalMapSample(
                        texture(textures[nonuniformEXT(lm.normal_tex)], layerUV).rgb,
                        lm.flags);
                    ns.xy *= max(lm.normal_strength, 0.0);
                    blendNormal_ts += weights[k] * ns;
                    anyNormalTex = true;
                } else {
                    blendNormal_ts += weights[k] * vec3(0.0, 0.0, 1.0);
                }
            }

            // Macro Color Map (SatMap) blending
            if (validTexture(tl.macro_color_tex) && tl.macro_color_strength > 0.0) {
                vec4 mColor = texture(textures[nonuniformEXT(tl.macro_color_tex)], uv);
                blendAlbedo = mix(blendAlbedo, mColor.rgb, clamp(tl.macro_color_strength, 0.0, 1.0));
            }

            if (validTexture(tl.semantic_map_tex)) {
                vec4 semantic = texture(textures[nonuniformEXT(tl.semantic_map_tex)], uv);
                // Snow burial, matching the ray-traced paths: a semantic value
                // is a measurement, its coverage is a visibility decision.
                // Flow reads 0.9 under two metres of snow and must not be
                // painted there.
                float exposed = 1.0 - clamp(weights[2], 0.0, 1.0);
                float wet = clamp(max(semantic.r, semantic.g), 0.0, 1.0) * exposed;
                float ice = clamp(semantic.b, 0.0, 1.0);
                // Hardness is a substrate property, so it is gated by rock
                // EXPOSURE (splat G) rather than merely buried by snow.
                float hard = clamp(semantic.a, 0.0, 1.0) * clamp(weights[1], 0.0, 1.0);
                // A bound overlay replaces the built-in tweak for its channel.
                // The viewport preview does not composite overlay MATERIALS
                // yet, so a bound channel simply keeps the plain splat blend
                // here rather than showing the built-in effect on top of a
                // material the ray-traced view is already drawing.
                if ((tl.overlay_mask & 3u) == 0u) {
                    blendAlbedo *= 1.0 - wet * clamp(tl.semantic_wet_darkening, 0.0, 0.8);
                    blendRoughness = mix(blendRoughness, 0.16,
                        wet * clamp(tl.semantic_wet_roughness, 0.0, 1.0));
                }
                if ((tl.overlay_mask & (1u << 2u)) == 0u) {
                    float iceLuma = dot(blendAlbedo, vec3(0.2126, 0.7152, 0.0722));
                    blendAlbedo = mix(blendAlbedo,
                        vec3(0.70, 0.82, 0.88) * max(iceLuma, 0.35), ice * 0.55);
                    blendRoughness = mix(blendRoughness, 0.12, ice * 0.65);
                }
                if ((tl.overlay_mask & (1u << 3u)) == 0u) {
                    blendRoughness = blendRoughness + hard * 0.035;
                }
                blendRoughness = clamp(blendRoughness, 0.0, 1.0);
            }

            // Apply blended normal first (derivative TBN — no surfaceTBN available in raster)
            if (anyNormalTex) {
                vec3 nts = normalize(blendNormal_ts);
                // Already in tangent space, fully decoded — re-encode to [0,1] and
                // pass flags=0 so applyNormalMap takes the plain decode path
                // (we already handled BC5 reconstruction inside the per-layer loop).
                N = applyNormalMap(N, vWorldPos, uv, nts * 0.5 + 0.5, 1.0, 0u);
            }

            // Override albedo/roughness/metallic — skip per-material texture sections below
            mat.albedo_r = blendAlbedo.r; mat.albedo_g = blendAlbedo.g; mat.albedo_b = blendAlbedo.b;
            mat.roughness = blendRoughness; mat.metallic = blendMetallic;
            mat.albedo_tex = 0u; mat.roughness_tex = 0u; mat.metallic_tex = 0u; mat.normal_tex = 0u;
        }
    }

    // ── Albedo ──
    vec3 albedo = vec3(mat.albedo_r, mat.albedo_g, mat.albedo_b);
    vec4 albedoTexel = vec4(1.0);
    if (validTexture(mat.albedo_tex)) {
        vec4 texAlbedo = texture(textures[nonuniformEXT(mat.albedo_tex)], uv);
        albedoTexel = texAlbedo;
        // Vulkan RT contract: a bound Base Color texture is authoritative.
        // Its sampled colour also becomes the depthless glass Beer tint.
        albedo = texAlbedo.rgb;
    }
    // Only fall back to neutral gray when no albedo source is bound.
    // Previously this fired whenever base color × texture went to ~0, which
    // turned black-paint strokes into gray in the raster material preview.

    // ── Normal map ──
    if (validTexture(mat.normal_tex)) {
        vec3 nmSample = texture(textures[nonuniformEXT(mat.normal_tex)], uv).rgb;
        float strength = (mat.normal_strength > 0.0) ? mat.normal_strength : 1.0;
        N = applyNormalMap(N, vWorldPos, uv, nmSample, strength, mat.flags);
    }

    // ── Roughness / Metallic ──
    float roughness = clamp(mat.roughness, 0.04, 1.0);
    float metallic  = clamp(mat.metallic,  0.0,  1.0);
    float specular  = clamp(mat.specular,  0.0,  1.0);
    float transmission = clamp(mat.transmission, 0.0, 1.0);
    float materialIor = max(mat.ior, 1.0001);
    if (validTexture(mat.roughness_tex)) {
        roughness = samplePackedRoughness(
            texture(textures[nonuniformEXT(mat.roughness_tex)], uv), 0.04, mat.flags);
    }
    if (validTexture(mat.metallic_tex)) {
        metallic = samplePackedMetallic(
            texture(textures[nonuniformEXT(mat.metallic_tex)], uv), mat.flags);
    }
    if (validTexture(mat.specular_tex)) {
        specular = clamp(texture(textures[nonuniformEXT(mat.specular_tex)], uv).r * specular, 0.0, 1.0);
    }
    if (validTexture(mat.transmission_tex)) {
        // RT contract: the map is authoritative, not multiplied by the scalar.
        // A graph Transmission output below intentionally overrides this fetch.
        transmission = clamp(
            texture(textures[nonuniformEXT(mat.transmission_tex)], uv).r, 0.0, 1.0);
    }

    // Per-pixel Material Graph program. Vulkan RT and Realtime consume the
    // same flattened bytecode and evaluator. Raster has no barycentric
    // pointiness/named-attribute stream yet, so those inputs use neutral
    // values; UV/object/world procedural chains, textures, bump and Bevel run.
    vec3 V = normalize(pc.cameraPos.xyz - vWorldPos);
    vec2 rawUV = vec2(vTexCoord.x, 1.0 - vTexCoord.y);
    float graphAttrs[MP_ATTRIB_SLOTS];
    for (int graphAttr = 0; graphAttr < MP_ATTRIB_SLOTS; ++graphAttr)
        graphAttrs[graphAttr] = 0.0;
    MatProgOut graphOut = mp_defaultOut();
    uint graphWritten = 0u;
    uint graphOffset = matProgramOffset(materialIndex);
    if (graphOffset != MATPROG_NONE) {
        graphOut = evalMaterialProgram(
            graphOffset, rawUV, vWorldPos, N, 0.5, vObjectOrigin,
            graphAttrs, vObjectPos, V,
            0.0, 0.0, 0.0, 0.0, vec3(0.0), vWorldPos,
            vec3(0.0), vec3(0.0), 0.0, vObjectPos, 0.0);
        graphWritten = graphOut.written;
        if ((graphWritten & MP_SLOT_BASECOLOR) != 0u)
            albedo = max(graphOut.baseColor, vec3(0.0));
        if ((graphWritten & MP_SLOT_ROUGHNESS) != 0u)
            roughness = clamp(graphOut.roughness, 0.04, 1.0);
        if ((graphWritten & MP_SLOT_METALLIC) != 0u)
            metallic = clamp(graphOut.metallic, 0.0, 1.0);
        if ((graphWritten & MP_SLOT_SPECULAR) != 0u)
            specular = clamp(graphOut.specular, 0.0, 1.0);
        if ((graphWritten & MP_SLOT_TRANSMISSION) != 0u)
            transmission = clamp(graphOut.transmission, 0.0, 1.0);
        if ((graphWritten & MP_SLOT_IOR) != 0u)
            materialIor = max(graphOut.ior, 1.0001);
        if ((graphWritten & MP_SLOT_NORMAL) != 0u) {
            vec3 graphNormal = N;
            if (graphOut.normalWorld) {
                if (dot(graphOut.normal, graphOut.normal) > 1e-8)
                    graphNormal = normalize(graphOut.normal);
            } else if (dot(graphOut.normal, graphOut.normal) > 1e-8) {
                vec3 encodedNormal = normalize(graphOut.normal) * 0.5 + 0.5;
                graphNormal = applyNormalMap(
                    N, vWorldPos, uv, encodedNormal, 1.0, 0u);
            }
            if (dot(graphNormal, V) > 0.0) N = graphNormal;
        }
    }

    // ★★★★ Impostor OPAKTIR ve bu bir kestirme degil, DOGRULUK sartidir.
    //   Proxy seridi dilim basina TEK bir UV tasir. Alfa-test edilen bir
    //   yaprakta o UV atlasin seffaf bir yerine denk gelirse asagidaki
    //   discard seridin TAMAMINI siler -- yani sahnenin uzak yarisi
    //   "kabalasmaz", BOSALIR. Ustelik tek bir texel'in maskesini butun
    //   impostor'a uygulamanin fiziksel bir karsiligi da yok: proxy zaten
    //   yaprak deseninin degil SILUETIN yaklasikligidir.
    float opacity = clamp(mat.opacity, 0.0, 1.0);
    if (isImpostor) {
        opacity = 1.0;
    } else if (validTexture(mat.opacity_tex)) {
        vec4 opacityTexel = texture(textures[nonuniformEXT(mat.opacity_tex)], uv);
        // flags bit 8: RGBA texture (opacity in .a); clear: grayscale mask (opacity in .r)
        // If opacity_tex == albedo_tex the user wired the same RGBA texture to both slots:
        // always read .a in that case — reading .r would bleed colour into the mask.
        bool useAlpha = ((mat.flags & 256u) != 0u) || (mat.opacity_tex == mat.albedo_tex);
        float maskValue = useAlpha ? opacityTexel.a : opacityTexel.r;
        opacity *= maskValue;
        // Hard floor matching RT pipeline (closesthit line ~1824):
        // values < 0.1 are treated as fully transparent to kill texture-compression ghosts.
        if (opacity < 0.1) opacity = 0.0;
    }
    if (!isImpostor && (graphWritten & MP_SLOT_OPACITY) != 0u) {
        opacity = clamp(graphOut.opacity, 0.0, 1.0);
        if (opacity < 0.1) opacity = 0.0;
    }
    if (opacity == 0.0) {
        discard;
    }
    // Match closesthit's legacy glass contract: partial material opacity on a
    // non-metal becomes transmission when no explicit transmission is authored.
    if (!isImpostor && opacity < 0.99 && metallic < 0.1 && transmission < 0.01)
        transmission = 1.0 - opacity;

    // The backend records the same canonical draw list twice. Classification
    // stays here because transmission texture and Material Graph outputs are
    // per-fragment; CPU material-level sorting would misclassify mapped glass.
    // Phase 0 preserves combined rendering until snapshot descriptors are
    // valid, avoiding a half-wired frame during resize/reload.
    uint drawPhase = previewDrawPhase();
    bool transmissiveFragment = !isImpostor &&
        (transmission > 0.001 || (mat.flags & PREVIEW_MAT_FLAG_BUBBLE) != 0u);
    if (drawPhase == PREVIEW_PHASE_OPAQUE && transmissiveFragment) {
        // Capture the farthest back-facing surface in view-distance units.
        // Positive IEEE float bits preserve ordering for atomicMax. This image
        // is private to transmission and never replaces selection depth.
        if (!gl_FrontFacing) {
            float backDistance = max(-(pc.view * vec4(vWorldPos, 1.0)).z, 0.0);
            imageAtomicMax(previewTransmissionBackDepth,
                           ivec2(gl_FragCoord.xy), floatBitsToUint(backDistance));
        }
        discard;
    }
    if (drawPhase == PREVIEW_PHASE_TRANSMISSION && !transmissiveFragment) {
        discard;
    }

    // scatterGlass() derives a transmitting interface from IOR, not from the
    // ordinary opaque Spec control. Blend the two F0 values by Transmission so
    // partial materials remain continuous while full glass loses the broad
    // white opaque-specular coat.
    bool bubbleSurface = (mat.flags & PREVIEW_MAT_FLAG_BUBBLE) != 0u;
    float effectiveIor = bubbleSurface && matx.bubble_ior > 1.0001
        ? matx.bubble_ior : materialIor;
    float interiorDepth = max(matx.transmission_density, 0.0);
    bool resinSurface = interiorDepth > 1e-4;

    // Specular anti-aliasing: at authored Roughness=0 a curved highlight can
    // become narrower than one pixel and disappear as the camera recedes.
    // Screen-space normal variation estimates that footprint and widens only
    // the reflection lobe, preserving a sharp close-up without adding samples.
    float authoredReflectionRoughness = resinSurface
        ? clamp(matx.resin_roughness, 0.0, 1.0) : roughness;
    float normalFootprint = max(length(dFdx(N)), length(dFdy(N)));
    float specularAa = clamp(normalFootprint * 0.65, 0.0, 0.18);
    float reflectionRoughness = clamp(sqrt(
        authoredReflectionRoughness * authoredReflectionRoughness +
        specularAa * specularAa), 0.04, 1.0);
    // An ideal mirror is a delta distribution. RT traces that direction
    // explicitly, whereas a rasterized punctual-light GGX peak can fall
    // entirely between pixels. Widen only the analytic direct-light lobe;
    // environment and SSR keep reflectionRoughness and therefore remain sharp.
    float directSpecularRoughness = max(
        reflectionRoughness,
        (transmissiveFragment || resinSurface || bubbleSurface) ? 0.075 : 0.04);

    // ── Procedural detail: subtle color variation + dirt + roughness ──
    // micro_detail_strength drives all world-space effects without touching UVs.
    // tile_break_strength (above) is separate — warps UV only when needed.
    if (matx.micro_detail_strength > 0.0) {
        float sc  = max(matx.micro_detail_scale, 0.5);
        float str = matx.micro_detail_strength;

        // Subtle world-space luminance variation — preserves texture detail,
        // breaks the "too clean" uniform look. ±8% max, independent seed.
        float colorVar   = pd_vnoise3(vWorldPos * sc * 0.7 + vec3(31.4, 17.2, 42.9));
        float colorDelta = (colorVar - 0.5) * 0.16 * str;
        albedo = clamp(albedo * (1.0 + colorDelta), vec3(0.0), vec3(1.0));

        // Dirt: fBm darkening in world-space valleys (dust / grime)
        float dirtFactor = pd_dirt(vWorldPos, sc) * str;
        vec3  dirtColor  = vec3(0.14, 0.10, 0.08);
        albedo = mix(albedo, albedo * dirtColor, dirtFactor);

        // Roughness micro-variation: breaks uniform-gloss appearance
        roughness = clamp(roughness + pd_roughnessVar(vWorldPos, sc) * str * 0.5,
                          0.04, 1.0);
    }

    // ── Emission ──
    vec3 emissionColor = vec3(mat.emission_r, mat.emission_g, mat.emission_b);
    float emissionStrength = max(mat.emission_strength, 0.0);
    if ((graphWritten & MP_SLOT_EMISSIONCOLOR) != 0u)
        emissionColor = max(graphOut.emissionColor, vec3(0.0));
    if ((graphWritten & MP_SLOT_EMISSIONSTRENGTH) != 0u)
        emissionStrength = max(graphOut.emissionStrength, 0.0);
    if (validTexture(mat.emission_tex) &&
        (graphWritten & MP_SLOT_EMISSIONCOLOR) == 0u) {
        emissionColor = texture(textures[nonuniformEXT(mat.emission_tex)], uv).rgb;
    }
    vec3 emission = emissionColor * emissionStrength;

    // ── Principled direct lighting ──
    // Opaque dielectric follows Spec; transmitting dielectric follows IOR.
    float authoredDielectricF0 = clamp(0.08 * specular, 0.0, 0.08);
    float interfaceDielectricF0 = previewDielectricF0(effectiveIor);
    float dielectricF0 = mix(authoredDielectricF0, interfaceDielectricF0,
                             (bubbleSurface || resinSurface) ? 1.0 : transmission);
    vec3 F0           = mix(vec3(dielectricF0), albedo, metallic);
    vec3 diffuseColor = albedo * (1.0 - metallic) * (1.0 - transmission);

    // Specular exponent from roughness
    float specExp = max(2.0, (1.0 - directSpecularRoughness) *
                                  (1.0 - directSpecularRoughness) * 128.0);

    vec3 lightDirs[3];
    float lightIntensities[3];
    vec3 lightColors[3];

    if (lightingPreset == 0u) {
        lightDirs[0] = normalize(pc.lightDir0.xyz);
        lightDirs[1] = normalize(pc.lightDir1.xyz);
        lightDirs[2] = normalize(pc.lightDir2.xyz);
        lightIntensities[0] = pc.lightDir0.w;
        lightIntensities[1] = pc.lightDir1.w;
        lightIntensities[2] = pc.lightDir2.w;
        lightColors[0] = vec3(1.0, 0.98, 0.95);
        lightColors[1] = vec3(0.75, 0.82, 0.95);
        lightColors[2] = vec3(0.90, 0.90, 0.95);
    } else if (lightingPreset == 2u) {
        lightDirs[0] = normalize(vec3(0.35, 0.84, 0.42));
        lightDirs[1] = normalize(vec3(-0.46, 0.58, 0.67));
        lightDirs[2] = normalize(vec3(-0.10, 0.22, -0.97));
        lightIntensities[0] = 1.20;
        lightIntensities[1] = 0.30;
        lightIntensities[2] = 0.14;
        lightColors[0] = vec3(1.0, 0.96, 0.88);
        lightColors[1] = vec3(0.64, 0.76, 0.95);
        lightColors[2] = vec3(0.88, 0.92, 1.0);
    } else {
        lightDirs[0] = normalize(vec3(0.60, 0.52, 0.61));
        lightDirs[1] = normalize(vec3(-0.54, 0.34, 0.77));
        lightDirs[2] = normalize(vec3(-0.16, 0.28, -0.95));
        lightIntensities[0] = 0.95;
        lightIntensities[1] = 0.48;
        lightIntensities[2] = 0.32;
        lightColors[0] = vec3(1.0, 0.98, 0.95);
        lightColors[1] = vec3(0.76, 0.82, 0.94);
        lightColors[2] = vec3(0.95, 0.96, 1.0);
    }

    // ── Sheen / SSS material params ──
    float sheenWeight    = clamp(matx.sheen, 0.0, 1.0);
    vec3  sheenColor     = mix(vec3(1.0), albedo, clamp(matx.sheen_tint, 0.0, 1.0)) * sheenWeight;
    float sssAmount      = clamp(mat.subsurface_amount, 0.0, 1.0);
    vec3  sssColor       = vec3(matx.subsurface_r, matx.subsurface_g, matx.subsurface_b);
    vec3  sssRadius      = max(vec3(matx.subsurface_radius_r,
                                    matx.subsurface_radius_g,
                                    matx.subsurface_radius_b), vec3(0.001));
    float sssScale       = max(matx.subsurface_scale, 0.001);
    float translucency   = clamp(mat.translucent, 0.0, 1.0);
    vec3  sssProfile     = vec3(1.0) - exp(-sssRadius * sssScale);
    vec3  profileTint    = mix(vec3(1.0), max(sssColor, vec3(0.0)),
                               sssProfile);
    vec3  boundedScatterTint = mix(vec3(1.0), profileTint, sssAmount);
    float profileTravel  = dot(sssProfile, vec3(0.2126, 0.7152, 0.0722));
    float sssWrap        = sssAmount * mix(0.25, 1.0, profileTravel);

    vec3 diffuseLit  = vec3(0.0);
    vec3 specularLit = vec3(0.0);
    vec3 sheenLit    = vec3(0.0);

    // ★★ Ust sinir var ve SESSIZ DEGIL: asildiginda C++ tarafi bir kez log
    //   basar. Scene golgeleri atlas tabanlidir; yine de sinirsiz bir isik
    //   dongusu 500 isikli bir sahnede viewport'u
    //   oldururdu ve bunun belirtisi "yavas", yani teshis edilmesi en zor
    //   belirti olurdu.
    // ★★ AYNI SAYI globals.h'deki kMaterialPreviewMaxSceneLights'ta da yazili
    //   (shader o basligi include edemez). Birini degistirip otekini birakmak,
    //   panelin/IPC'nin "N isik kullaniliyor" deyip shader'in baskasini
    //   kullanmasi demektir.
    int sceneLoopCount = (lightingPreset == 3u)
        ? int(min(sceneLightCount, kPreviewMaxSceneLights)) : 0;
    bool useWorldSun = lightingPreset == 3u && worldMode == 2u && worldSun.w > 0.0;
    int lightLoopCount = (lightingPreset == 3u)
        ? sceneLoopCount + (useWorldSun ? 1 : 0) : 3;

    for (int i = 0; i < lightLoopCount; ++i) {
        vec3 L;
        vec3 radiance;
        if (lightingPreset == 3u) {
            // Koninin disinda / arkasinda kalan isik KATKI VERMEZ, sifir
            // katkiyla toplanmaz -- aksi halde alan isiginin arkasi da
            // NdotL uzerinden hafifce aydinlanirdi.
            if (i < sceneLoopCount) {
                if (!evalSceneLight(uint(i), vWorldPos, L, radiance)) continue;
                radiance *= evaluatePreviewShadow(uint(i), vWorldPos, N, L);
            } else {
                L = dot(worldSun.xyz, worldSun.xyz) > 1e-8
                    ? normalize(worldSun.xyz) : vec3(0.0, 1.0, 0.0);
                radiance = canonicalWorldSunRadiance(L);
                radiance *= evaluatePreviewShadow(kPreviewMaxSceneLights,
                                                  vWorldPos, N, L);
            }
        } else {
            L        = lightDirs[i];
            radiance = lightColors[i] * lightIntensities[i];
        }

        // ── Subsurface scattering: wrapped diffuse (Jensen 2001 approximation) ──
        // Shifts the NdotL threshold so light bleeds around the terminator.
        // sssAmount=0 → standard Lambertian; sssAmount=1 → full wrap.
        float wrapNdotL = (dot(N, L) + sssWrap) / (1.0 + sssWrap);
        float NdotL = max(wrapNdotL, 0.0);
        // SSS tints the sub-surface contribution toward the sssColor.
        vec3 diffuseAlbedo = diffuseColor * boundedScatterTint;
        // ★★★★ 1/PI Lambert BRDF'in PARCASI, bir kadran degil. Yillarca eksikti
        //   ve yerine asagida `diffuseWeight = 0.35` sihirli katsayisi vardi --
        //   ama YALNIZCA dusuk kalite dalinda. Yani kalite preset'i degistirmek
        //   POZLAMAYI degistiriyordu (olculdu: 0.4055 vs 0.4893). Kalite
        //   preset'i BRDF kademesini degistirir, pozlamayi DEGIL.
        // Same bounded directional-average Fresnel term as the Vulkan RT
        // reference (bsdf_scatter.glsl::evaluate_brdf_gl). Diffuse only uses
        // energy not assigned to the reflection lobe.
        vec3 Favg = F0 + (vec3(1.0) - F0) * (1.0 / 21.0);
        vec3 diffuseBrdf = diffuseAlbedo * (vec3(1.0) - Favg) * (1.0 / PI);

        // Clearcoat is a top layer, not an additive glow: coat Fresnel removes
        // energy from the base lobes before its own GGX reflection is added.
        vec3 H = normalize(V + L);
        float VdotH = max(dot(V, H), 0.0);
        float coatFresnel = mat.clearcoat > 0.001
            ? fresnelSchlick(VdotH, vec3(0.04)).r * clamp(mat.clearcoat, 0.0, 1.0)
            : 0.0;
        vec3 coatTint = clearcoatFilmTint(
            VdotH, matx.clearcoat_iridescence,
            matx.clearcoat_film_thickness);
        float baseLayerWeight = 1.0 - coatFresnel;
        diffuseLit += diffuseBrdf * radiance * NdotL * baseLayerWeight *
                      (1.0 - translucency);
        // Thin-surface transmission is a bounded back-lighting lobe. It does
        // not claim refraction/thickness parity; those belong to the forward
        // transmission pass. The probability split mirrors RT's principled
        // diffuse sub-layer so translucency does not add energy to the front.
        float backNdotL = max(dot(-N, L), 0.0);
        diffuseLit += diffuseColor * boundedScatterTint * radiance * backNdotL *
                      translucency * (1.0 / PI) * baseLayerWeight;

        // Scene is the path-traced comparison surface and always uses GGX.
        // The cheap lobe remains only in the independent inspection rig.
        bool useFullPbr = lightingPreset == 3u || qualityMode > 1u;
        if (!useFullPbr) {
            float NdotH  = max(dot(N, H), 0.0);
            float pureNdotL = max(dot(N, L), 0.0);
            // ★★ Blinn-Phong lobu NORMALIZE edilir: (n+2)/(8*PI). Eskiden
            //   normalizasyon yoktu ve yerine `specularWeight = 0.15` vardi.
            //   Katsayi lobun uslu terimine bagli degildi, yani parlaklik
            //   puruzlulukle birlikte yanlis yone kayiyordu.
            float spec   = pow(NdotH, specExp) * pureNdotL;
            specularLit += F0 * radiance * spec *
                           ((specExp + 2.0) / (8.0 * PI)) * baseLayerWeight;
            if (mat.clearcoat > 0.001) {
                float ccExp = max(2.0, (1.0 - mat.clearcoat_roughness) * (1.0 - mat.clearcoat_roughness) * 128.0);
                float ccSpec = pow(NdotH, ccExp) * pureNdotL;
                specularLit += vec3(0.04) * radiance * ccSpec * mat.clearcoat
                             * ((ccExp + 2.0) / (8.0 * PI)) * coatTint;
            }
        } else {
            float pureNdotL = max(dot(N, L), 0.0);
            specularLit += evaluateSpecularGGX(
                               N, V, L, F0, directSpecularRoughness) *
                           radiance * pureNdotL * baseLayerWeight;
            if (mat.clearcoat > 0.001) {
                specularLit += evaluateSpecularGGX(
                    N, V, L, vec3(0.04), clamp(mat.clearcoat_roughness, 0.02, 1.0)) *
                    radiance * pureNdotL * mat.clearcoat * coatTint;
            }
        }

        // ── Sheen lobe (fabric / velvet) ──
        if (sheenWeight > 0.001) {
            sheenLit += evaluateSheen(N, V, L, sheenColor, roughness) *
                        radiance * baseLayerWeight;
        }
    }

    float NdotV_main = max(dot(N, V), 0.0);
    float coatViewFresnel = mat.clearcoat > 0.001
        ? fresnelSchlick(NdotV_main, vec3(0.04)).r *
          clamp(mat.clearcoat, 0.0, 1.0)
        : 0.0;
    float ambientBaseWeight = 1.0 - coatViewFresnel;
    vec3 coatEnvTint = clearcoatFilmTint(
        NdotV_main, matx.clearcoat_iridescence,
        matx.clearcoat_film_thickness);

    vec3 ambient = vec3(0.0);
    vec3 envSpecular = vec3(0.0);
    if (lightingPreset == 3u) {
        vec3 R = reflect(-V, N);
        if (worldMode == 1u && (sceneFlags & 32u) != 0u) {
            float intensity = max(worldParams.y, 0.0);
            vec3 irradiance = texture(worldIrradiance, worldDirToUV(N)).rgb * intensity;
            vec3 prefiltered = textureLod(worldPrefiltered, worldDirToUV(R),
                                          reflectionRoughness * 8.0).rgb * intensity;
            vec2 brdf = texture(worldBrdfLut,
                                vec2(clamp(NdotV_main, 0.001, 0.999), reflectionRoughness)).rg;
            vec3 fresnelAmb = fresnelSchlickRoughness(
                NdotV_main, F0, reflectionRoughness);
            ambient = irradiance * diffuseColor * (vec3(1.0) - fresnelAmb) *
                      ambientBaseWeight;
            envSpecular = prefiltered * (F0 * brdf.x + brdf.y) *
                          ambientBaseWeight;
            if (mat.clearcoat > 0.001) {
                float ccRoughness = clamp(mat.clearcoat_roughness, 0.02, 1.0);
                vec3 ccEnv = textureLod(worldPrefiltered, worldDirToUV(R),
                                        ccRoughness * 8.0).rgb * intensity;
                vec2 ccBrdf = texture(worldBrdfLut,
                                      vec2(clamp(NdotV_main, 0.001, 0.999),
                                           ccRoughness)).rg;
                envSpecular += ccEnv * (vec3(0.04) * ccBrdf.x + ccBrdf.y) *
                               mat.clearcoat * coatEnvTint;
            }
        } else {
            vec3 envDiffuse = worldMode == 2u
                ? samplePhysicalSkyFiltered(N, 1.0)
                : sampleCanonicalWorld(N);
            // Physical Sky gets a bounded deterministic approximation of RT's
            // rough GGX environment integral instead of a blue-biased bent ray.
            vec3 envReflection = worldMode == 2u
                ? samplePhysicalSkyFiltered(R, reflectionRoughness)
                : sampleCanonicalWorld(normalize(mix(
                    R, N, reflectionRoughness * reflectionRoughness)));
            vec3 fresnelAmb = fresnelSchlickRoughness(
                NdotV_main, F0, reflectionRoughness);
            ambient = envDiffuse * diffuseColor * (vec3(1.0) - fresnelAmb) *
                      ambientBaseWeight;
            envSpecular = envReflection * fresnelAmb *
                          mix(1.0, 0.18, reflectionRoughness * reflectionRoughness) *
                          ambientBaseWeight;
            if (mat.clearcoat > 0.001) {
                vec3 ccFresnel = fresnelSchlickRoughness(
                    NdotV_main, vec3(0.04), mat.clearcoat_roughness);
                envSpecular += envReflection * ccFresnel * mat.clearcoat *
                               mix(0.8, 0.05, mat.clearcoat_roughness) *
                               coatEnvTint;
            }
        }
    } else if (lightingPreset == 0u) {
        vec3 ambientUp   = vec3(0.15, 0.17, 0.22);
        vec3 ambientDown = vec3(0.08, 0.06, 0.05);
        float ambientBlend = N.y * 0.5 + 0.5;
        // Energy conservation: metallic surfaces have no diffuse ambient
        vec3 kD = diffuseColor * (vec3(1.0) - fresnelSchlickRoughness(
            NdotV_main, F0, reflectionRoughness));
        ambient = mix(ambientDown, ambientUp, ambientBlend) * kD *
                  ambientBaseWeight;
    } else {
        vec3 envDiffuse    = samplePreviewEnvironment(N, lightingPreset);
        vec3 R             = reflect(-V, N);
        vec3 envReflection = sampleEnvSpecular(R, lightingPreset);

        // Roughness-corrected Fresnel for ambient (Lagarde 2012)
        vec3 fresnelAmb = fresnelSchlickRoughness(
            NdotV_main, F0, reflectionRoughness);
        // kD: diffuse only gets energy not taken by specular, and none for metals
        vec3 kD = (vec3(1.0) - fresnelAmb) * (1.0 - metallic);
        ambient = envDiffuse * diffuseColor * kD *
                  (lightingPreset == 2u ? 0.42 : 0.36) * ambientBaseWeight;

        // roughness² gives physically correct falloff: smooth metals get strong env
        // reflection, rough metals stay dim. Multiplier compensates for the low-res
        // baked env map (128×64 blurs softbox peaks from 1.8 → ~0.3).
        envSpecular = envReflection * fresnelAmb *
                      mix(3.0, 0.20, reflectionRoughness * reflectionRoughness) *
                      ambientBaseWeight;

        // Clearcoat env reflection
        if (mat.clearcoat > 0.001) {
            vec3 ccFresnel = fresnelSchlickRoughness(NdotV_main, vec3(0.04), mat.clearcoat_roughness);
            envSpecular += envReflection * ccFresnel
                         * mix(0.9, 0.0, mat.clearcoat_roughness)
                         * mat.clearcoat * coatEnvTint;
        }
    }

    // ★★★ Kalite dallarina gore agirlik YOK. Iki dal da normalize edilmis
    //   BRDF kullandigi icin `qualityMode` artik yalnizca lobun BICIMINI
    //   secer (ucuz Blinn-Phong vs GGX), siddetini degil. Kabul kriteri:
    //   preset degistirmek goruntunun parlakligini DEGISTIRMEMELI.
    // Keep the diffuse sub-layer energy split consistent outside direct
    // lights too. Environment back-lighting is deliberately a bounded lookup,
    // not screen refraction or a claim of thick-transmission parity.
    vec3 backEnvironment = lightingPreset == 3u
        ? sampleCanonicalWorld(-N)
        : samplePreviewEnvironment(-N, lightingPreset);
    ambient = ambient * boundedScatterTint *
              (1.0 - translucency);
    ambient += backEnvironment * diffuseColor * boundedScatterTint * translucency *
               (1.0 / PI) * ambientBaseWeight;

    // ── Glass / resin / thin-film surface ────────────────────────────────
    // The material contract mirrors Vulkan RT here (texture/graph
    // transmission, IOR Fresnel and Beer extinction). Until the dedicated
    // scene-color/depth refraction pass lands, the continuation ray resolves
    // against the canonical world; capability reporting calls this bounded
    // environment refraction rather than traversal parity.
    float viewCos = clamp(abs(dot(N, V)), 0.0, 1.0);
    float interfaceFresnel = previewDielectricFresnel(viewCos, effectiveIor);
    vec3 incoming = -V;
    vec3 refractedDir = bubbleSurface
        ? incoming
        : refract(incoming, N, 1.0 / effectiveIor);
    bool totalInternalReflection = dot(refractedDir, refractedDir) < 1e-8;
    if (totalInternalReflection)
        refractedDir = reflect(incoming, N);
    refractedDir = normalize(refractedDir);

    vec3 resinTint = clamp(
        vec3(matx.resin_color_r, matx.resin_color_g, matx.resin_color_b),
        vec3(0.0), vec3(1.0));
    // Match scatterGlass: spectral dispersion belongs only to thin/ordinary
    // glass. Authored resin depth is a coat/base path and deliberately keeps a
    // single IOR so inclusions and the base do not split into false RGB layers.
    vec3 transmittedWorld;
    if (!bubbleSurface && interiorDepth <= 1e-4 && mat.dispersion > 1e-3) {
        bool spectralTir = false;
        transmittedWorld = sampleDispersedTransmission(
            incoming, N, effectiveIor, mat.dispersion, roughness,
            lightingPreset, spectralTir);
        totalInternalReflection = spectralTir;
    } else {
        transmittedWorld = sampleTransmissionEnvironment(
            refractedDir, roughness, lightingPreset);
    }
    vec3 extinction = previewBeerExtinction(resinTint);
    float frontViewDistance = max(-(pc.view * vec4(vWorldPos, 1.0)).z, 0.0);
    uint backBits = drawPhase != PREVIEW_PHASE_COMBINED
        ? imageLoad(previewTransmissionBackDepth, ivec2(gl_FragCoord.xy)).r
        : 0u;
    float backViewDistance = uintBitsToFloat(backBits);
    float geometricThickness = max(backViewDistance - frontViewDistance, 0.0);
    bool hasProceduralInterior = matx.resin_inclusion > 0.001 ||
        matx.resin_dirt > 0.001 || matx.resin_shard > 0.001;
    float shellDistance = geometricThickness > 1e-4 ? geometricThickness : 0.65;
    float pathLength = interiorDepth > 1e-4
        ? shellDistance * interiorDepth / max(abs(dot(refractedDir, -N)), 0.10)
        : (hasProceduralInterior
            ? shellDistance / max(abs(dot(refractedDir, -N)), 0.10) : 0.0);
    bool resinObjectSpace = (mat.flags & (1u << 21)) != 0u;
    mat3 worldToObject = mat3(vWorldToObject0, vWorldToObject1, vWorldToObject2);
    vec3 interiorOrigin = resinObjectSpace ? vObjectPos : vWorldPos;
    vec3 interiorDirection = resinObjectSpace
        ? normalize(worldToObject * refractedDir)
        : refractedDir;
    PreviewInteriorSample interior = previewResinInterior(
        interiorOrigin, interiorDirection, pathLength, extinction,
        matx.resin_inclusion, matx.resin_dirt, matx.resin_shard,
        matx.resin_inclusion_scale,
        vec3(matx.dust_color_a_r, matx.dust_color_a_g, matx.dust_color_a_b),
        vec3(matx.dust_color_b_r, matx.dust_color_b_g, matx.dust_color_b_b),
        vec3(matx.resin_dirt_color_r, matx.resin_dirt_color_g, matx.resin_dirt_color_b),
        matx.resin_shard_hue,
        uint(matx.dust_style + 0.5),
        uint(matx.shard_shape + 0.5));

    // Depthless coloured glass consumes Base Color exactly as Vulkan RT's
    // scatterGlass tint. The raster pass composites entry and exit together,
    // so a closed shell can use its measured front/back thickness; open/thin
    // geometry retains the RT fallback distance. Once Interior Depth is
    // authored, Interior Color remains the sole absorption tint by contract.
    if (interiorDepth <= 1e-4 && !bubbleSurface) {
        vec3 glassTint = clamp(albedo, vec3(0.0), vec3(1.0));
        // scatterGlass attenuates at entry and exit using a 0.65 fallback per
        // interface. This replay composites both at once, hence the 1.30 floor.
        float glassDistance = max(geometricThickness, 1.30) /
            max(abs(dot(refractedDir, -N)), 0.05);
        interior.transmittance *= exp(
            -(vec3(1.0) - glassTint) * glassDistance);
    }

    vec3 sceneThrough = vec3(0.0);
    bool sceneRefractionValid = false;
    if (drawPhase == PREVIEW_PHASE_TRANSMISSION && !bubbleSurface) {
        // RT's depthless glass shell uses 0.65 for its bounded interior march.
        // Authored Interior Depth takes precedence. Front/back geometric depth
        // will replace only this distance source; Snell/dispersion/composite
        // semantics stay unchanged.
        float screenTravel = geometricThickness > 1e-4
            ? geometricThickness
            : (interiorDepth > 1e-4 ? interiorDepth : 0.65);
        if (mat.dispersion > 1e-3 && interiorDepth <= 1e-4) {
            float spread = (effectiveIor - 1.0) * mat.dispersion * 0.06;
            vec3 channelIor = max(vec3(effectiveIor - spread, effectiveIor,
                                       effectiveIor + spread), vec3(1.0001));
            sceneRefractionValid = true;
            for (int channel = 0; channel < 3; ++channel) {
                vec3 d = refract(incoming, N, 1.0 / channelIor[channel]);
                if (dot(d, d) < 1e-8) {
                    sceneRefractionValid = false;
                    break;
                }
                vec2 channelUv;
                if (!previewProjectRefraction(vWorldPos, normalize(d), screenTravel,
                                              channelUv) ||
                    texture(previewOpaqueDepth, channelUv).r + 1e-5 < gl_FragCoord.z) {
                    sceneRefractionValid = false;
                    break;
                }
                sceneThrough[channel] = sampleOpaqueSceneRough(
                    channelUv, roughness)[channel];
            }
        } else {
            vec2 sceneUv;
            sceneRefractionValid = previewProjectRefraction(
                vWorldPos, refractedDir, screenTravel, sceneUv) &&
                texture(previewOpaqueDepth, sceneUv).r + 1e-5 >= gl_FragCoord.z;
            if (sceneRefractionValid)
                sceneThrough = sampleOpaqueSceneRough(sceneUv, roughness);
        }
        // A projected Snell ray can miss a thin/concave shell or cross an
        // unrelated foreground depth discontinuity. If opaque geometry is
        // visibly present behind this glass pixel, keep transmission on that
        // scene continuation instead of replacing the whole fragment with a
        // homogeneous Physical Sky lookup. True sky/off-screen pixels still
        // use the refracted environment direction below.
        if (!sceneRefractionValid) {
            vec2 localUv = gl_FragCoord.xy / max(postC.xy, vec2(1.0));
            float localOpaqueDepth = texture(previewOpaqueDepth, localUv).r;
            if (localOpaqueDepth < 0.999999 &&
                localOpaqueDepth + 1e-5 >= gl_FragCoord.z) {
                sceneThrough = sampleOpaqueSceneRough(localUv, roughness);
                sceneRefractionValid = true;
            }
        }
        if (sceneRefractionValid)
            sceneThrough *= interior.transmittance;
    }

    vec3 sceneReflection = vec3(0.0);
    bool sceneReflectionValid = false;
    if (drawPhase == PREVIEW_PHASE_TRANSMISSION) {
        vec2 reflectionUv;
        vec3 reflectionDir = normalize(reflect(incoming, N));
        float reflectionReach = max(frontViewDistance * 2.0, 4.0);
        sceneReflectionValid = tracePreviewOpaqueReflection(
            vWorldPos, reflectionDir, reflectionReach, reflectionUv);
        if (sceneReflectionValid)
            sceneReflection = sampleOpaqueSceneRough(
                reflectionUv, reflectionRoughness);
    }

    vec3 transmissionLit = vec3(0.0);
    if (bubbleSurface) {
        vec3 filmTint = matx.bubble_film > 0.001
            ? previewBubbleFilm(matx.bubble_film / max(viewCos, 0.15))
            : vec3(1.0);
        vec3 reflectedWorld = lightingPreset == 3u
            ? sampleCanonicalWorld(reflect(incoming, N))
            : samplePreviewEnvironment(reflect(incoming, N), lightingPreset);
        transmissionLit = reflectedWorld * filmTint * interfaceFresnel +
                          transmittedWorld * (vec3(0.85) + albedo * 0.15) *
                          (1.0 - interfaceFresnel);
    } else if (transmission > 0.001 && !sceneRefractionValid) {
        vec3 body = transmittedWorld * interior.transmittance + interior.scatter;
        float passWeight = totalInternalReflection ? 0.0 :
            transmission * (1.0 - metallic) * (1.0 - interfaceFresnel);
        transmissionLit = body * passWeight;
    }

    // Resin depth with a remaining opaque lobe is a coat over a base, not a
    // global tint over reflections. Attenuate only diffuse/ambient/sheens;
    // direct and environment specular remain on the top interface.
    float resinBaseWeight = interiorDepth > 1e-4
        ? clamp(1.0 - transmission, 0.0, 1.0) : 0.0;
    vec3 resinBaseAttenuation = mix(vec3(1.0), interior.transmittance,
                                    resinBaseWeight);
    // A valid screen-space hit replaces the environment reflection. It is
    // composited after post because the opaque snapshot is already display
    // referred. Direct-light GGX remains, so local lights still make highlights.
    // Interior Depth already selected IOR + Coat Glossy as the canonical
    // surface lobe above; a second resin environment term would double it.
    if (sceneReflectionValid) {
        envSpecular = vec3(0.0);
    }

    vec3 color = (ambient + diffuseLit + sheenLit) * resinBaseAttenuation
               + specularLit + envSpecular
               + transmissionLit
               + interior.scatter * resinBaseWeight
               + (bubbleSurface ? vec3(0.0) : emission);

    // ★★★ Buradaki ACES + sRGB SOKULDU. Onizleme artik projenin `post.*`
    //   zincirinden geciyor, yani Rendered ile AYNI operatorden. Eskiden
    //   shader ACES uygularken projenin tone_mapping ayari `none` idi ve
    //   ustune CPU post gecisi zaten sRGB'ye kodlanmis 8-bit degerlere
    //   exposure uyguluyordu -- CIFT KODLAMA.
    color = rtApplyPost(color, previewPostParams(),
                        gl_FragCoord.xy / max(postC.xy, vec2(1.0)));

    if (sceneReflectionValid) {
        vec3 reflectionFresnel = fresnelSchlickRoughness(
            NdotV_main, F0, reflectionRoughness);
        color += sceneReflection * reflectionFresnel * ambientBaseWeight;
    }

    // The framebuffer already contains sky and any scene geometry submitted
    // before this surface. Transmission must therefore open the raster alpha
    // gate as well as evaluate a refracted environment direction. Keeping
    // alpha at material opacity (usually 1) was the reason correct slider/IOR
    // reactions still looked like a flat opaque grey body.
    float interiorPass = clamp(dot(interior.transmittance,
                                   vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    float dielectricPass = bubbleSurface
        ? (1.0 - interfaceFresnel)
        : transmission * (1.0 - metallic) * (1.0 - interfaceFresnel);
    // Beer is already RGB-multiplied into sceneThrough/transmissionLit. Keep
    // its luminance only for legacy alpha fallback; multiplying it into the
    // refraction weight as well applied absorption twice and washed out tint.
    float outputOpacity = clamp(
        opacity * (1.0 - dielectricPass * interiorPass), 0.015, 1.0);
    if (drawPhase == PREVIEW_PHASE_TRANSMISSION) {
        if (sceneRefractionValid) {
            // previewOpaqueColor is already display-referred by the same post
            // chain. Add it after rtApplyPost so it is never tone-mapped twice.
            color += sceneThrough * dielectricPass;
        }
        // Both scene continuation and refracted-environment fallback already
        // contain the complete reflection/transmission split. Alpha-blending
        // the live framebuffer again added an un-refracted sky term and was the
        // source of the homogeneous atmosphere-coloured glass body.
        outputOpacity = 1.0;
    }
    outColor = vec4(color, outputOpacity);
}
