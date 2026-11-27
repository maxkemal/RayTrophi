#version 450
precision highp float;
precision highp int;
#ifdef PREVIEW_COVERED_SHADING
layout(early_fragment_tests) in;
#endif
// Scene feature specialization (PLAYER_ANDROID_PERF_PLAN O3). The host sets
// each one to false only when NO material / scene state can reach that
// branch, so the driver drops it at pipeline creation. Default true: a
// pipeline created without specialization info is the full shader.
// Measured with malioc on Mali-G52 2026-11-23: these three branches hold
// most of the per-thread stack (1600 B -> ~740 B when all three are off).
// Host mirror: Viewport/MaterialPreviewCoverage.cpp (kMpSpec* ids).
layout(constant_id = 0) const bool RT_SPEC_TERRAIN = true;
layout(constant_id = 1) const bool RT_SPEC_GRAPH   = true;
layout(constant_id = 2) const bool RT_SPEC_WEATHER = true;
// Glass / resin / bubble surface (T2 remaining-spill item, 56 B on Mali-G52).
// Off only when no bound material can end up with transmission > 0, a bubble
// flag or an Interior Depth: then those three are exactly 0 / false here and
// forcing them is bit-identical, while the compiler drops the refraction,
// dispersion, resin-interior and screen-refraction code.
layout(constant_id = 3) const bool RT_SPEC_GLASS        = true;
// Extended material lobes (clearcoat, sheen, SSS, translucency, micro-detail,
// tile-break). Off only when no referenced material uses any of these lobes
// and no Material Graph is active.
layout(constant_id = 4) const bool RT_SPEC_EXT_LOBES    = true;
// Non-scene preview lighting rigs (three-point, studio, outdoor + Blinn-Phong).
// Off when lightingPreset == 3 (Scene lighting), which is always true in Player.
layout(constant_id = 5) const bool RT_SPEC_ALT_LIGHTING = true;
// Unfiltered sky fallback when prefiltered IBL is not ready (sceneFlags bit 32).
layout(constant_id = 6) const bool RT_SPEC_FALLBACK_SKY = true;
// RayFusion GI / probe visibility / RT screen shadow / volume deep shadow.
// Off only on non-ray-query GPUs with no active volumes.
layout(constant_id = 7) const bool RT_SPEC_RAYFUSION_GI = true;
#define RT_PREVIEW_HAS_RAYFUSION_GI_SPEC 1
// Optional PBR textures (metallic, specular, emission). Off when no referenced
// material binds any of them and no Material Graph is active.
layout(constant_id = 8) const bool RT_SPEC_OPT_TEX      = true;
// Clearcoat lobe + film iridescence tint. Separated from RT_SPEC_EXT_LOBES so
// scenes with clearcoated surfaces (e.g. glossy balloons/toys) do not pay for
// sheen, subsurface, translucency, or micro/tile-break branches on Mali-G52.
layout(constant_id = 9) const bool RT_SPEC_CLEARCOAT      = true;
// Active material texture array page count (1..32). Specializing to the scene's
// actual page count lets the driver fold unused textureGrad/Lod switch cases.
layout(constant_id = 10) const uint RT_SPEC_TEX_PAGES     = 32u;
#define RT_SPEC_HAS_TEX_PAGES 1
// Wet paint canvas overlay (mat.paint_base_tex != 0). Off when no bound
// material uses a paint base texture.
layout(constant_id = 11) const bool RT_SPEC_PAINT_CANVAS  = true;
// Clearcoat thin-film iridescence tint. Off when no clearcoated material sets
// clearcoat_iridescence > 0.001.
layout(constant_id = 12) const bool RT_SPEC_CLEARCOAT_IRID = true;
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
layout(location = 9) flat in float vObjectRandom;
layout(location = 10) in vec4 vPreviousClip;
layout(location = 11) in vec4 vCurrentClip;

layout(location = 0) out mediump vec4 outColor;
// ★★★ RayFusion reflection G-buffer. Bu iki cikis, yansima compute gecisinin
//   OKUDUGU seydir ve onlari yazan shader, ayni degerlerle golgelendiren
//   shader'in KENDISIDIR. Normali derinlikten geri kurmak bir secenek degildi:
//   yansimada yon hatasi normal hatasinin 2 KATI ve roughness 0,05'te lob ~3
//   derece -- depth turevinden gelen geometrik normal o lobun icine sigmaz, ve
//   normal map orada hic yok. Kanal sozlesmesi:
//   rayfusion_reflection_gbuffer.glsl icinde, TEK yerde.
layout(location = 1) out mediump vec2 outRfNormal;
layout(location = 2) out mediump vec4 outRfSpecular;
layout(location = 3) out mediump vec2 outPreviousClip;

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

// Each viewport material texture reference selects a page and layer.
#define MATERIAL_TEXTURE_FRAGMENT
#define MATERIAL_TEXTURE_BINDING 1
#define MATERIAL_TEXTURE_REF_BINDING 29
#include "material_texture_arrays.glsl"
#undef MATERIAL_TEXTURE_BINDING
#undef MATERIAL_TEXTURE_REF_BINDING

// The compiled Material Graph VM is shared with Vulkan RT. Only the binding
// number differs in this smaller descriptor set; the bytecode and evaluator
// are shared. Texture ops read the preview's material texture pages.
#define MATERIAL_PROGRAM_BINDING 16
#include "material_program.glsl"
#include "material_preview_transmission.glsl"
#include "material_preview_screen_gi.glsl"

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
    uint sceneFlags;       // bit0=shadow, bit1=env, bit2=overlay, bit3=atmos LUT, bit4=unused (was sky-sun shadow), bit5=HDRI IBL, bit6=cloud shadow (26/27)
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
#include "material_preview_rt_shadow.glsl"
#define RT_PREVIEW_SCREEN_SHADOW
#include "material_preview_shadow_data.glsl"
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

layout(std140, set = 0, binding = 28) uniform RayFusionFrameUniform {
    mat4 view;
    vec4 lightDir2;
    mat4 curViewProj;
    mat4 prevViewProj;
    vec4 motionMeta;
} rayFusionFrame;

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;

    vec4 cameraPos;
    vec4 lightDir0;   // key light
    vec4 lightDir1;   // fill light

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
// Cloud shadow (Faz 3c, cloud_shadow.comp): transmittance toward the sun on
// the y = 0 plane around the camera. Read only with sceneFlags bit 64 (the
// adapter bound bindings 26/27). Directional light = the sun (sun rule).
layout(set = 0, binding = 26) uniform sampler2D cloudShadowMap;
#include "weather_render_types.glsl"
#include "cloud_lightning_light.glsl"
layout(set = 0, binding = 27, std430) readonly buffer CloudShadowFrame {
    vec3  sunDir;
    float sunIntensity;
    float atmosphereHeight;
    int   lutReady;
    float centerX;
    float centerZ;
    vec4 camPos;
    vec4 camRight;
    vec4 camUp;
    vec4 camFwd;
    vec4 prevPos;
    vec4 prevRight;
    vec4 prevUp;
    vec4 prevFwd;
    uvec4 info;
    WeatherRenderData weatherEffects;
    CloudFlashLightG flash[4];   // lit cloud bases (Backend::CloudRasterFrameGPU)
} cloudShadowFrame;
#define WEATHER_DATA cloudShadowFrame.weatherEffects
#define WEATHER_SURFACE_AVAILABLE (RT_SPEC_WEATHER && (sceneFlags & 64u) != 0u)
#include "weather_surface.glsl"
const float CLOUD_SHADOW_EXTENT_M = 32000.0;   // cloud_shadow.comp

// Lightning flash: the same CPU-built disc lights as the RT surface NEE
// (closesthit.rchit), unshadowed here (no flash shadow map), and without the
// RT's cloud-coverage gate (no cloud field bound): a flash the CPU placed in a
// clear patch of the storm footprint still lights the ground in RayFusion.
vec3 previewLightningIrradiance(vec3 P, vec3 N) {
    if (!RT_SPEC_WEATHER || (sceneFlags & 64u) == 0u) return vec3(0.0);
    vec3 E = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        vec3 wi;
        E += cloudFlashIrradiance(cloudShadowFrame.flash[i], P, N, wi);
    }
    return E;
}

float previewCloudShadow(vec3 P) {
    if (!RT_SPEC_WEATHER || (sceneFlags & 64u) == 0u) return 1.0;
    vec3 s = normalize(cloudShadowFrame.sunDir);
    if (s.y <= 0.02) return 1.0;
    // Where this point's sun ray crosses y = 0 (the map's plane).
    vec2 g = P.xz - s.xz * (P.y / s.y);
    vec2 uv = (g - vec2(cloudShadowFrame.centerX, cloudShadowFrame.centerZ)) / CLOUD_SHADOW_EXTENT_M + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    return texture(cloudShadowMap, uv).r;
}

bool evalSceneLight(uint idx, vec3 P, out mediump vec3 L, out mediump vec3 radiance, out vec3 lightPos) {
    vec4 posType = sceneLights[idx].position;
    lightPos = posType.xyz;
    int type = int(posType.w + 0.5);
    mediump vec4 colInt = sceneLights[idx].color;
    mediump vec3 base = colInt.rgb * colInt.w;

    if (type == 1) {                       // Directional
        L = normalize(sceneLights[idx].direction.xyz);
        radiance = base * previewCloudShadow(P);
        return true;
    }

    vec3 toLight = lightPos - P;
    float d2 = dot(toLight, toLight);
    if (d2 < 1e-6) return false;
    vec3 L_hp = toLight * inversesqrt(d2);
    L = L_hp;
    float falloffSoft = sceneLights[idx].area_v.w;
    float atten;
    if (falloffSoft <= 1e-4) {
        atten = 1.0 / d2;
    } else {
        float falloffExp = max(2.0 - falloffSoft, 0.0);
        atten = (falloffExp <= 1e-4) ? 1.0 : pow(d2, -0.5 * falloffExp);
    }

    if (type == 2) {                       // Area: merkez ornegi, tek yuzlu
        vec3 ln = normalize(cross(sceneLights[idx].area_u.xyz, sceneLights[idx].area_v.xyz));
        vec2 wh = sceneLights[idx].params.yz;
        // RT tarafinda cos/d^2 attenuation'i pdf = 1/alan ile bolunur, yani
        // etkin carpan cos * alan / d^2 olur. Ayni carpani burada dogrudan
        // uyguluyoruz ki iki hat AYNI parlaklik mertebesinde olsun.
        atten *= max(dot(-L_hp, ln), 0.0) * max(wh.x * wh.y, 0.0);
    } else if (type == 3) {                // Spot
        vec4 dirOuter = sceneLights[idx].direction;
        float cosTheta = dot(-L_hp, normalize(dirOuter.xyz));
        float inner = sceneLights[idx].params.z;
        float outer = dirOuter.w;
        float f;
        if (cosTheta < outer)      f = 0.0;
        else if (cosTheta > inner) f = 1.0;
        else { float t = (cosTheta - outer) / (inner - outer + 1e-6); f = t * t; }
        atten *= f;
    }

    if (atten <= 0.0) return false;
    mediump float attenM = atten;
    radiance = base * attenM;
    return true;
}

uint pointShadowFace(vec3 d) {
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.x && a.y >= a.z) return d.y >= 0.0 ? 2u : 3u;
    return d.z >= 0.0 ? 4u : 5u;
}

mediump float evaluatePreviewShadow(uint lightIndex, vec3 P, mediump vec3 N, mediump vec3 L, vec3 lightPosition) {
    return rtPreviewShadow(lightIndex, P, N, L, lightPosition, previewQualityMode());
}

#include "canonical_world.glsl"
#include "probe_field.glsl"
#include "rayfusion_specular_visibility.glsl"
#include "rayfusion_reflection_gbuffer.glsl"

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

bool validTexture(uint textureId) {
    return materialTextureValid(textureId);
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

mediump vec3 fresnelSchlick(mediump float cosTheta, mediump vec3 F0) {
    mediump float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    mediump float x2 = x * x;
    mediump float x5 = x2 * x2 * x;
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
mediump vec3 fresnelSchlickRoughness(mediump float cosTheta, mediump vec3 F0, mediump float roughness) {
    mediump float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    mediump float x2 = x * x;
    mediump float x5 = x2 * x2 * x;
    mediump vec3 envelope = max(vec3(1.0 - roughness), F0);
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

mediump vec3 evaluateSpecularGGXPrecomputed(float NdotV, float NdotL, float NdotH,
                                            mediump float VdotH, mediump vec3 F0, float roughness) {
    if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);
    float D = D_GGX(NdotH, roughness);
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    float oneMinusK = 1.0 - k;
    float visDenom = 4.0 * max(NdotV * oneMinusK + k, 1e-5) * max(NdotL * oneMinusK + k, 1e-5);
    mediump vec3 F = fresnelSchlick(VdotH, F0);
    mediump float specTerm = D / visDenom;
    return specTerm * F;
}

vec3 evaluateSpecularGGX(vec3 N, vec3 V, vec3 L, vec3 F0, float roughness) {
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    return evaluateSpecularGGXPrecomputed(NdotV, NdotL, NdotH, VdotH, F0, roughness);
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

// Approximate the rough glass continuation with the HDRI's GGX prefilter.
// The deterministic cone covers unfiltered environments and Physical Sky,
// whose IBL bake omits the sun disc present in transmission.
vec3 sampleTransmissionEnvironment(vec3 axis, float roughness,
                                   uint lightingPreset) {
    axis = normalize(axis);
    bool canonical = lightingPreset == 3u;
    if (roughness <= 0.001)
        return canonical ? sampleCanonicalWorld(axis)
                         : samplePreviewEnvironment(axis, lightingPreset);

    const uint hdriAndIblReady = 2u | 32u;
    if (canonical && worldMode == 1u &&
        (sceneFlags & hdriAndIblReady) == hdriAndIblReady) {
        return textureLod(worldPrefiltered, worldDirToUV(axis), roughness * 8.0).rgb *
               max(worldParams.y, 0.0);
    }

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
        // Constant-select instead of channelIor[channel] / spectral[channel]:
        // dynamic vector indexing forces the vectors into stack memory on Mali.
        float chIor = channel == 0 ? channelIor.x : (channel == 1 ? channelIor.y : channelIor.z);
        vec3 direction = refract(incoming, normal, 1.0 / chIor);
        bool tir = dot(direction, direction) < 1e-8;
        allTir = allTir && tir;
        if (tir)
            direction = reflect(incoming, normal);
        vec3 sampleValue = sampleTransmissionEnvironment(
            normalize(direction), roughness, lightingPreset);
        float sv = channel == 0 ? sampleValue.x : (channel == 1 ? sampleValue.y : sampleValue.z);
        spectral = vec3(channel == 0 ? sv : spectral.x,
                        channel == 1 ? sv : spectral.y,
                        channel == 2 ? sv : spectral.z);
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
#include "material_preview_uv.glsl"

// Derivative-based TBN (tangent from screen-space partial derivatives)
// Returns a new perturbed normal using the normal map sample.
mediump vec3 applyNormalMapGrad(mediump vec3 N, vec3 worldPos, vec2 duv1, vec2 duv2,
                                mediump vec3 nmSample, float strength, uint matFlags) {
    vec3 dp1  = dFdx(worldPos);
    vec3 dp2  = dFdy(worldPos);

    float det = duv1.x * duv2.y - duv2.x * duv1.y;
    if (abs(det) < 1e-20) return N;

    float invDet = 1.0 / det;
    mediump vec3 T = normalize((duv2.y * dp1 - duv1.y * dp2) * invDet);
    mediump vec3 B = -normalize((-duv2.x * dp1 + duv1.x * dp2) * invDet);

    mediump vec3 tNormal = decodeNormalMapSample(nmSample, matFlags);
    tNormal.xy *= max(strength, 0.0);
    tNormal     = normalize(tNormal);

    return normalize(mat3(T, B, N) * tNormal);
}

vec3 applyNormalMap(vec3 N, vec3 worldPos, vec2 uv, vec3 nmSample, float strength, uint matFlags) {
    return applyNormalMapGrad(N, worldPos, dFdx(uv), dFdy(uv), nmSample, strength, matFlags);
}

#include "raster_material_policy.h"
#include "material_preview_opacity.glsl"


// Studio/matcap rig lights, evaluated on demand (no local arrays: a local
// array indexed by the loop counter becomes a Mali stack alloca).
vec4 previewRigRim() { return previewDrawPhase() == 2u ? vec4(0.0) : rayFusionFrame.lightDir2; }
vec3 previewRigDir(uint preset, int i) {
    if (preset == 0u) {
        if (i == 0) return normalize(pc.lightDir0.xyz);
        if (i == 1) return normalize(pc.lightDir1.xyz);
        return normalize(previewRigRim().xyz);
    } else if (preset == 2u) {
        if (i == 0) return normalize(vec3(0.35, 0.84, 0.42));
        if (i == 1) return normalize(vec3(-0.46, 0.58, 0.67));
        return normalize(vec3(-0.10, 0.22, -0.97));
    }
    if (i == 0) return normalize(vec3(0.60, 0.52, 0.61));
    if (i == 1) return normalize(vec3(-0.54, 0.34, 0.77));
    return normalize(vec3(-0.16, 0.28, -0.95));
}
float previewRigIntensity(uint preset, int i) {
    if (preset == 0u) return i == 0 ? pc.lightDir0.w : (i == 1 ? pc.lightDir1.w : previewRigRim().w);
    if (preset == 2u) return i == 0 ? 1.20 : (i == 1 ? 0.30 : 0.14);
    return i == 0 ? 0.95 : (i == 1 ? 0.48 : 0.32);
}
vec3 previewRigColor(uint preset, int i) {
    if (preset == 0u) return i == 0 ? vec3(1.0, 0.98, 0.95) : (i == 1 ? vec3(0.75, 0.82, 0.95) : vec3(0.90, 0.90, 0.95));
    if (preset == 2u) return i == 0 ? vec3(1.0, 0.96, 0.88) : (i == 1 ? vec3(0.64, 0.76, 0.95) : vec3(0.88, 0.92, 1.0));
    return i == 0 ? vec3(1.0, 0.98, 0.95) : (i == 1 ? vec3(0.76, 0.82, 0.94) : vec3(0.95, 0.96, 1.0));
}

void main() {
    mediump vec3 N = normalize(vWorldNormal);
    uint qualityMode = previewQualityMode();
    uint lightingPreset = RT_SPEC_ALT_LIGHTING ? pc.materialMeta.z : 3u;

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
    uint matFlags = mat.flags;

    uint drawPhase = previewDrawPhase();
    uint graphOffset = RT_SPEC_GRAPH ? matProgramOffset(materialIndex) : MATPROG_NONE;
    // Wet paint canvas over a tiled base (docs/dev/WET_PAINT_CANVAS_UV.md): the
    // slots are canvas pages at the raw UV, the transform tiles only the base.
    bool paintCanvas = RT_SPEC_PAINT_CANVAS && mat.paint_base_tex != 0u;
    vec2 paintBaseUV = applyUVTransform(vTexCoord, mat);
    vec2 uv = paintCanvas ? vTexCoord : paintBaseUV;

    // ── Procedural tile-break (independent slider, applied before texture sampling) ──
    // Breaks visible UV tiling seams. Separate from dirt/roughness so albedo maps
    // that shouldn't be warped can leave this at 0.
    if (RT_SPEC_EXT_LOBES && mat.tile_break_strength > 0.0 &&
        (validTexture(mat.albedo_tex) || validTexture(mat.roughness_tex) || validTexture(mat.normal_tex))) {
        // A paint canvas has no tiling seam; its base does.
        if (paintCanvas) paintBaseUV = pd_tileBreak(paintBaseUV, vWorldPos, mat.tile_break_strength);
        else uv = pd_tileBreak(uv, vWorldPos, mat.tile_break_strength);
    }

    vec2 uvDx = dFdx(uv);
    vec2 uvDy = dFdy(uv);
    vec2 paintBaseDx = RT_SPEC_PAINT_CANVAS ? dFdx(paintBaseUV) : uvDx;
    vec2 paintBaseDy = RT_SPEC_PAINT_CANVAS ? dFdy(paintBaseUV) : uvDy;

    // Reject provably opaque replay fragments before terrain, textures and
    // graph evaluation. Maps are authoritative and graphs may write transmission
    // or opacity; partial opacity also becomes legacy glass below. All of those
    // cases must retain the per-fragment classification path.
    if (RT_SPEC_GLASS && drawPhase == PREVIEW_PHASE_TRANSMISSION &&
        (isImpostor ||
         !rasterMaterialMayTransmit(mat.transmission, mat.opacity,
             validTexture(mat.transmission_tex), validTexture(mat.opacity_tex),
             graphOffset != MATPROG_NONE, (matFlags & PREVIEW_MAT_FLAG_BUBBLE) != 0u,
             (matFlags & MATERIAL_FLAGS_PREVIEW_CUTOUT) != 0u))) {
        discard;
    }

    // Coverage first: transparent needle/card pixels must not fetch normal,
    // roughness, terrain layers, or evaluate lighting before being discarded.
    // A graph may REPLACE opacity, so only its final output may reject it.
    mediump float opacity = previewSurfaceOpacity(mat, uv, uvDx, uvDy, isImpostor);
    if (graphOffset == MATPROG_NONE) {
        if (opacity == 0.0) discard;
        // Once opacity is sampled, opaque texels of a mixed card can also
        // leave replay before the rest of the material is evaluated.
        if (RT_SPEC_GLASS && drawPhase == PREVIEW_PHASE_TRANSMISSION &&
            !rasterMaterialMayTransmit(mat.transmission, opacity,
                validTexture(mat.transmission_tex), false, false,
                (matFlags & PREVIEW_MAT_FLAG_BUBBLE) != 0u,
                (matFlags & MATERIAL_FLAGS_PREVIEW_CUTOUT) != 0u)) discard;
    }

    // ── Terrain Splat-Layer Blending (FLAG_TERRAIN = bit 16) ──
    // Mirrors closesthit.rchit logic: blends up to 4 material layers using an RGBA
    // splat map, then overrides albedo / roughness / metallic / normal before the
    // standard per-material texture sampling below.
    const uint FLAG_TERRAIN = (1u << 16);
    bool terrainOverridden = false;
    vec3  terrainAlbedo    = vec3(0.0);
    float terrainRoughness = 0.0;
    float terrainMetallic  = 0.0;
    if (RT_SPEC_TERRAIN && (matFlags & FLAG_TERRAIN) != 0u) {
        uint layerIdx = mat._terrain_layer_idx;
        TerrainLayerData tl = terrainLayers[layerIdx];
        if (validTexture(tl.splat_map_tex) && tl.layer_count > 0u) {
            // R=layer0, G=layer1, B=layer2, A=layer3
            vec4 splatW = sampleMaterialTextureGrad(tl.splat_map_tex, uv, uvDx, uvDy);
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
                vec2 layerScale = tl.layer_uv_scale[k] * vec2(lsx, lsy);
                vec2 layerDx = uvDx * layerScale;
                vec2 layerDy = uvDy * layerScale;

                vec3 lAlbedo = max(vec3(lm.albedo_r, lm.albedo_g, lm.albedo_b), vec3(0.0));
                if (validTexture(lm.albedo_tex))
                    lAlbedo = sampleMaterialTextureGrad(lm.albedo_tex, layerUV, layerDx, layerDy).rgb;
                blendAlbedo += weights[k] * lAlbedo;

                float lRough = clamp(lm.roughness, 0.0, 1.0);
                if (validTexture(lm.roughness_tex))
                    lRough = sampleMaterialTextureGrad(lm.roughness_tex, layerUV, layerDx, layerDy).g;
                blendRoughness += weights[k] * lRough;

                float lMetal = clamp(lm.metallic, 0.0, 1.0);
                if (validTexture(lm.metallic_tex))
                    lMetal = sampleMaterialTextureGrad(lm.metallic_tex, layerUV, layerDx, layerDy).b;
                blendMetallic += weights[k] * lMetal;

                if (validTexture(lm.normal_tex)) {
                    vec3 ns = decodeNormalMapSample(
                        sampleMaterialTextureGrad(lm.normal_tex, layerUV, layerDx, layerDy).rgb,
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
                vec4 mColor = sampleMaterialTextureGrad(tl.macro_color_tex, uv, uvDx, uvDy);
                blendAlbedo = mix(blendAlbedo, mColor.rgb, clamp(tl.macro_color_strength, 0.0, 1.0));
            }

            if (validTexture(tl.semantic_map_tex)) {
                vec4 semantic = sampleMaterialTextureGrad(tl.semantic_map_tex, uv, uvDx, uvDy);
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
            terrainOverridden = true;
            terrainAlbedo = blendAlbedo;
            terrainRoughness = blendRoughness;
            terrainMetallic = blendMetallic;
        }
    }

    // ── Albedo ──
    mediump vec3 albedo;
    if (RT_SPEC_TERRAIN && terrainOverridden) {
        albedo = terrainAlbedo;
    } else {
        uint albedoTex = mat.albedo_tex;
        if (validTexture(albedoTex)) {
            mediump vec4 texAlbedo = sampleMaterialTextureGrad(albedoTex, uv, uvDx, uvDy);
            if (RT_SPEC_PAINT_CANVAS && paintCanvas) {
                uint paintBaseTex = mat.paint_base_tex;
                if (validTexture(paintBaseTex)) {
                    // Premultiplied canvas over the tiled base; the canvas alpha is
                    // paint coverage, never surface opacity (the base keeps that).
                    mediump vec4 baseTexel = sampleMaterialTextureGrad(paintBaseTex, paintBaseUV, paintBaseDx, paintBaseDy);
                    texAlbedo = vec4(baseTexel.rgb * (1.0 - texAlbedo.a) + texAlbedo.rgb, baseTexel.a);
                }
            }
            // Vulkan RT contract: a bound Base Color texture is authoritative.
            // Its sampled colour also becomes the depthless glass Beer tint.
            albedo = texAlbedo.rgb;
        } else {
            albedo = vec3(mat.albedo_r, mat.albedo_g, mat.albedo_b);
        }
    }
    // Only fall back to neutral gray when no albedo source is bound.
    // Previously this fired whenever base color × texture went to ~0, which
    // turned black-paint strokes into gray in the raster material preview.

    // ── Normal map ──
    if (!(RT_SPEC_TERRAIN && terrainOverridden)) {
        uint normalTex = mat.normal_tex;
        if (validTexture(normalTex)) {
            mediump vec3 nmSample = sampleMaterialTextureGrad(normalTex, uv, uvDx, uvDy).rgb;
            float nStr = mat.normal_strength;
            float strength = (nStr > 0.0) ? nStr : 1.0;
            N = applyNormalMapGrad(N, vWorldPos, uvDx, uvDy, nmSample, strength, matFlags);
        }
    }

    // ── Roughness / Metallic ──
    mediump float roughness;
    if (RT_SPEC_TERRAIN && terrainOverridden) {
        roughness = clamp(terrainRoughness, 0.04, 1.0);
    } else {
        uint roughTex = mat.roughness_tex;
        if (validTexture(roughTex)) {
            roughness = samplePackedRoughness(
                sampleMaterialTextureGrad(roughTex, uv, uvDx, uvDy), 0.04, matFlags);
        } else {
            roughness = clamp(mat.roughness, 0.04, 1.0);
        }
    }
    mediump float metallic;
    if (RT_SPEC_TERRAIN && terrainOverridden) {
        metallic = clamp(terrainMetallic, 0.0, 1.0);
    } else if (RT_SPEC_OPT_TEX && validTexture(mat.metallic_tex)) {
        metallic = samplePackedMetallic(
            sampleMaterialTextureGrad(mat.metallic_tex, uv, uvDx, uvDy), matFlags);
    } else {
        metallic = clamp(mat.metallic, 0.0, 1.0);
    }
    mediump float specular  = clamp(mat.specular,  0.0,  1.0);
    if (RT_SPEC_OPT_TEX && validTexture(mat.specular_tex)) {
        specular = clamp(sampleMaterialTextureGrad(mat.specular_tex, uv, uvDx, uvDy).r * specular, 0.0, 1.0);
    }
    mediump float transmission = RT_SPEC_GLASS ? clamp(mat.transmission, 0.0, 1.0) : 0.0;
    float materialIor  = RT_SPEC_GLASS ? max(mat.ior, 1.0001) : 1.5;
    if (RT_SPEC_GLASS && validTexture(mat.transmission_tex)) {
        // RT contract: the map is authoritative, not multiplied by the scalar.
        // A graph Transmission output below intentionally overrides this fetch.
        transmission = clamp(
            sampleMaterialTextureGrad(mat.transmission_tex, uv, uvDx, uvDy).r, 0.0, 1.0);
    }

    // Per-pixel Material Graph program. Vulkan RT and Realtime consume the
    // same flattened bytecode and evaluator. Raster has no barycentric
    // pointiness/named-attribute stream yet, so those inputs use neutral
    // values; UV/object/world procedural chains, textures, bump and Bevel run.
    mediump vec3 V = normalize(pc.cameraPos.xyz - vWorldPos);
    MatProgOut graphOut = mp_defaultOut();
    uint graphWritten = 0u;
    if (RT_SPEC_GRAPH && graphOffset != MATPROG_NONE) {
        vec2 rawUV = vec2(vTexCoord.x, 1.0 - vTexCoord.y);
        float graphAttrs[MP_ATTRIB_SLOTS];
        for (int graphAttr = 0; graphAttr < MP_ATTRIB_SLOTS; ++graphAttr)
            graphAttrs[graphAttr] = 0.0;
        graphOut = evalMaterialProgramWithObjectRandom(
            graphOffset, rawUV, vWorldPos, N, 0.5, vObjectOrigin,
            vObjectRandom,
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
    if (RT_SPEC_GRAPH && !isImpostor && (graphWritten & MP_SLOT_OPACITY) != 0u) {
        opacity = clamp(graphOut.opacity, 0.0, 1.0);
        if (opacity < 0.1) opacity = 0.0;
        opacity = materialCoverageOpacity(opacity, (matFlags & MATERIAL_FLAGS_PREVIEW_CUTOUT) != 0u);
        if (opacity == 0.0) {
            discard;
        }
    }
    // Match closesthit's legacy glass contract: partial material opacity on a
    // non-metal becomes transmission when no explicit transmission is authored.
    if (RT_SPEC_GLASS && !isImpostor && opacity < 0.99 && metallic < 0.1 && transmission < 0.01)
        transmission = 1.0 - opacity;
    if (!RT_SPEC_GLASS) transmission = 0.0;

    // Replay retains only meshes that MAY transmit. Final classification stays
    // per-fragment for opacity/transmission maps and Material Graph outputs.
    // Phase 0 preserves combined rendering until snapshot descriptors are
    // valid, avoiding a half-wired frame during resize/reload.
    bool transmissiveFragment = RT_SPEC_GLASS && !isImpostor &&
        (transmission > 0.001 || (matFlags & PREVIEW_MAT_FLAG_BUBBLE) != 0u);
    if (RT_SPEC_GLASS && drawPhase == PREVIEW_PHASE_OPAQUE && transmissiveFragment) {
        // Capture the farthest back-facing surface in view-distance units.
        // Positive IEEE float bits preserve ordering for atomicMax. This image
        // is private to transmission and never replaces selection depth.
        if (!gl_FrontFacing) {
            float backDistance = max(-(rayFusionFrame.view * vec4(vWorldPos, 1.0)).z, 0.0);
            imageAtomicMax(previewTransmissionBackDepth,
                           ivec2(gl_FragCoord.xy), floatBitsToUint(backDistance));
        }
        discard;
    }
    if (RT_SPEC_GLASS && drawPhase == PREVIEW_PHASE_TRANSMISSION && !transmissiveFragment) {
        discard;
    }
    // With glass specialized off no fragment is transmissive, so the line
    // above already discarded the whole transmission replay. Spelling that out
    // lets the compiler fold the screen-space refraction/reflection below.
    bool transmissionPhase = RT_SPEC_GLASS && drawPhase == PREVIEW_PHASE_TRANSMISSION;

    // scatterGlass() derives a transmitting interface from IOR, not from the
    // ordinary opaque Spec control. Blend the two F0 values by Transmission so
    // partial materials remain continuous while full glass loses the broad
    // white opaque-specular coat.
    bool bubbleSurface = RT_SPEC_GLASS && (matFlags & PREVIEW_MAT_FLAG_BUBBLE) != 0u;
    float effectiveIor = bubbleSurface && matx.bubble_ior > 1.0001
        ? matx.bubble_ior : materialIor;
    float interiorDepth = RT_SPEC_GLASS ? max(matx.transmission_density, 0.0) : 0.0;
    bool resinSurface = interiorDepth > 1e-4;

    if (WEATHER_SURFACE_AVAILABLE) {
        vec3 support = normalize(mix(normalize(vWorldNormal), N, 0.85));
        applyWeatherSurface(vWorldPos, N, support, albedo, roughness, metallic);
        N = weatherSurfaceNormal(vWorldPos, N, support);
    }

    // Specular anti-aliasing: at authored Roughness=0 a curved highlight can
    // become narrower than one pixel and disappear as the camera recedes.
    // Screen-space normal variation estimates that footprint and widens only
    // the reflection lobe, preserving a sharp close-up without adding samples.
    mediump float authoredReflectionRoughness = resinSurface
        ? clamp(matx.resin_roughness, 0.0, 1.0) : roughness;
    mediump float normalFootprint = max(length(dFdx(N)), length(dFdy(N)));
    mediump float specularAa = clamp(normalFootprint * 0.65, 0.0, 0.18);
    mediump float reflectionRoughness = clamp(sqrt(
        authoredReflectionRoughness * authoredReflectionRoughness +
        specularAa * specularAa), 0.04, 1.0);
    // An ideal mirror is a delta distribution. RT traces that direction
    // explicitly, whereas a rasterized punctual-light GGX peak can fall
    // entirely between pixels. Widen only the analytic direct-light lobe;
    // environment and SSR keep reflectionRoughness and therefore remain sharp.
    mediump float directSpecularRoughness = max(
        reflectionRoughness,
        (transmissiveFragment || resinSurface || bubbleSurface) ? 0.075 : 0.04);

    // ── Procedural detail: subtle color variation + dirt + roughness ──
    // micro_detail_strength drives all world-space effects without touching UVs.
    // tile_break_strength (above) is separate — warps UV only when needed.
    if (RT_SPEC_EXT_LOBES && matx.micro_detail_strength > 0.0) {
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
    // When emission texture and Material Graph are both specialized off, defer
    // the SSBO read until final compositing so emission does not occupy 3 live
    // registers across the entire direct-light and IBL blocks on Mali-G52.
    mediump vec3 emission = vec3(0.0);
    if (RT_SPEC_OPT_TEX || RT_SPEC_GRAPH) {
        vec3 emissionColor = vec3(mat.emission_r, mat.emission_g, mat.emission_b);
        float emissionStrength = max(mat.emission_strength, 0.0);
        if ((graphWritten & MP_SLOT_EMISSIONCOLOR) != 0u)
            emissionColor = max(graphOut.emissionColor, vec3(0.0));
        if ((graphWritten & MP_SLOT_EMISSIONSTRENGTH) != 0u)
            emissionStrength = max(graphOut.emissionStrength, 0.0);
        if (RT_SPEC_OPT_TEX && validTexture(mat.emission_tex) &&
            (graphWritten & MP_SLOT_EMISSIONCOLOR) == 0u) {
            emissionColor = sampleMaterialTextureGrad(mat.emission_tex, uv, uvDx, uvDy).rgb;
        }
        emission = emissionColor * emissionStrength;
    }

    // ── Principled direct lighting ──
    // Opaque dielectric follows Spec; transmitting dielectric follows IOR.
    mediump float authoredDielectricF0 = clamp(0.08 * specular, 0.0, 0.08);
    mediump float dielectricF0 = authoredDielectricF0;
    if (RT_SPEC_GLASS) {
        float interfaceDielectricF0 = previewDielectricF0(effectiveIor);
        dielectricF0 = mix(authoredDielectricF0, interfaceDielectricF0,
                           (bubbleSurface || resinSurface) ? 1.0 : transmission);
    }
    mediump vec3 F0           = mix(vec3(dielectricF0), albedo, metallic);
    mediump vec3 diffuseColor = albedo * (1.0 - metallic) * (1.0 - transmission);

    // Specular exponent from roughness
    float specExp = max(2.0, (1.0 - directSpecularRoughness) *
                                  (1.0 - directSpecularRoughness) * 128.0);

    // ── Sheen / SSS material params ──
    float sheenWeight    = RT_SPEC_EXT_LOBES ? clamp(matx.sheen, 0.0, 1.0) : 0.0;
    vec3  sheenColor     = RT_SPEC_EXT_LOBES
        ? mix(vec3(1.0), albedo, clamp(matx.sheen_tint, 0.0, 1.0)) * sheenWeight
        : vec3(0.0);
    float sssAmount      = RT_SPEC_EXT_LOBES ? clamp(mat.subsurface_amount, 0.0, 1.0) : 0.0;
    vec3  sssColor       = RT_SPEC_EXT_LOBES
        ? vec3(matx.subsurface_r, matx.subsurface_g, matx.subsurface_b)
        : vec3(0.0);
    vec3  sssRadius      = RT_SPEC_EXT_LOBES
        ? max(vec3(matx.subsurface_radius_r,
                   matx.subsurface_radius_g,
                   matx.subsurface_radius_b), vec3(0.001))
        : vec3(0.001);
    float sssScale       = RT_SPEC_EXT_LOBES ? max(matx.subsurface_scale, 0.001) : 0.001;
    float translucency   = RT_SPEC_EXT_LOBES ? clamp(mat.translucent, 0.0, 1.0) : 0.0;
    // ★★★ RT parity (bsdf_scatter.glsl::scatterSSS). In RT the SSS lobe takes
    //   a share `sssAmount` of the diffuse sub-layer and its random walk
    //   converges to subsurface_color — the MULTI-scatter albedo — not to base
    //   colour. The old raster path tinted base colour by 1-exp(-radius*scale),
    //   which at the presets' scale (0.05) is ~5%: the swatch the artist picked
    //   was nearly invisible here while RT showed it in full.
    mediump vec3 diffuseSurface    = diffuseColor * (1.0 - sssAmount);
    mediump vec3 diffuseSubsurface = clamp(sssColor, vec3(0.0), vec3(1.0)) *
                              (1.0 - metallic) * (1.0 - transmission) * sssAmount;
    diffuseColor = diffuseSurface + diffuseSubsurface;
    // Light travel vs local curvature (pre-integrated-skin style). Mean free
    // path is in WORLD units, like RT's radius*scale, so the same dial means
    // the same thing on both paths: a 1 cm radius bleeds on a nose, not on a
    // wall. Curvature from screen derivatives of the geometric normal.
    vec3  sssMfp       = sssRadius * sssScale;
    float sssCurvature = RT_SPEC_EXT_LOBES
        ? (length(fwidth(normalize(vWorldNormal))) /
           max(length(fwidth(vWorldPos)), 1e-6))
        : 0.0;
    mediump vec3 sssWrap      = RT_SPEC_EXT_LOBES
        ? clamp(sssMfp * sssCurvature, vec3(0.0), vec3(1.0))
        : vec3(0.0);
    // Thin-feature back-lighting: the osculating diameter 2/curvature is the
    // thickness proxy. Per channel, so a long red radius glows red (ears).
    mediump vec3 sssBackTransmit = (RT_SPEC_EXT_LOBES && sssCurvature > 1e-4)
        ? exp(-(2.0 / sssCurvature) / max(sssMfp, vec3(1e-5)))
        : vec3(0.0);

    mediump vec3 diffuseLit  = vec3(0.0);
    mediump vec3 specularLit = vec3(0.0);
    mediump vec3 sheenLit    = vec3(0.0);

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
    // ★★★ The Physical Sky sun is NOT a light here, exactly as in Vulkan RT
    //   (closesthit.rchit): direct sun comes only from a directional scene
    //   light. This used to add worldSun as an extra light on top of the scene
    //   lights -- with sun sync on, that directional IS the sun, so RayFusion
    //   lit it twice. Switching to Nishita now adds a "Sun" directional when
    //   the scene has none (rtapi::ensureWorldSunLight).
    int lightLoopCount = (lightingPreset == 3u) ? sceneLoopCount : 3;

    // Extract clearcoat scalars into mediump (1 register packed) so the
    // 160-byte GpuMaterial `mat` dies before the light loop and IBL block.
    mediump float matClearcoat = RT_SPEC_CLEARCOAT ? mat.clearcoat : 0.0;
    mediump float matClearcoatRoughness = RT_SPEC_CLEARCOAT ? mat.clearcoat_roughness : 0.0;
    bool hasClearcoat = RT_SPEC_CLEARCOAT && matClearcoat > 0.001;

    for (int i = 0; i < lightLoopCount; ++i) {
        mediump vec3 L;
        mediump vec3 radiance;
        if (lightingPreset == 3u) {
            // Koninin disinda / arkasinda kalan isik KATKI VERMEZ, sifir
            // katkiyla toplanmaz -- aksi halde alan isiginin arkasi da
            // NdotL uzerinden hafifce aydinlanirdi.
            vec3 lightPos;
            if (!evalSceneLight(uint(i), vWorldPos, L, radiance, lightPos)) continue;
            radiance *= evaluatePreviewShadow(uint(i), vWorldPos, N, L, lightPos);
        } else {
            L        = previewRigDir(lightingPreset, i);
            radiance = previewRigColor(lightingPreset, i) * previewRigIntensity(lightingPreset, i);
        }

        // ── Subsurface scattering: wrapped diffuse (Jensen 2001 approximation) ──
        // Shifts the NdotL threshold so light bleeds around the terminator.
        // Per-channel wrap on the SSS share only; the surface share and the
        // specular lobe keep the true cosine.
        mediump float rawNdotL = dot(N, L);
        mediump float NdotL = max(rawNdotL, 0.0);
        mediump vec3 sssNdotL = RT_SPEC_EXT_LOBES
            ? max((vec3(rawNdotL) + sssWrap) / (vec3(1.0) + sssWrap), vec3(0.0))
            : vec3(0.0);
        mediump vec3 Favg = F0 + (vec3(1.0) - F0) * (1.0 / 21.0);
        mediump vec3 diffuseBrdf = (vec3(1.0) - Favg) * (1.0 / PI);

        // Clearcoat is a top layer, not an additive glow: coat Fresnel removes
        // energy from the base lobes before its own GGX reflection is added.
        mediump vec3 H = normalize(V + L);
        mediump float VdotH = max(dot(V, H), 0.0);
        mediump float NdotH = max(dot(N, H), 0.0);
        mediump float coatFresnel = hasClearcoat
            ? fresnelSchlick(VdotH, vec3(0.04)).r * clamp(matClearcoat, 0.0, 1.0)
            : 0.0;
        mediump vec3 coatTint = (hasClearcoat && RT_SPEC_CLEARCOAT_IRID)
            ? clearcoatFilmTint(
                VdotH, matx.clearcoat_iridescence,
                matx.clearcoat_film_thickness)
            : vec3(1.0);
        mediump float baseLayerWeight = 1.0 - coatFresnel;
        mediump vec3 lightDiffuse = diffuseBrdf * radiance * baseLayerWeight *
                      (diffuseSurface * NdotL + diffuseSubsurface *
                       (sssNdotL + sssBackTransmit * max(-rawNdotL, 0.0))) *
                      (1.0 - translucency);
        // Thin-surface transmission is a bounded back-lighting lobe. It does
        // not claim refraction/thickness parity; those belong to the forward
        // transmission pass. The probability split mirrors RT's principled
        // diffuse sub-layer so translucency does not add energy to the front.
        if (RT_SPEC_EXT_LOBES && translucency > 0.0) {
            mediump float backNdotL = max(dot(-N, L), 0.0);
            lightDiffuse += diffuseColor * radiance * backNdotL *
                            translucency * (1.0 / PI) * baseLayerWeight;
        }
        diffuseLit += lightDiffuse;

        // Scene is the path-traced comparison surface and always uses GGX.
        // The cheap lobe remains only in the independent inspection rig.
        mediump vec3 lightSpec = vec3(0.0);
        bool useFullPbr = lightingPreset == 3u || qualityMode > 1u;
        if (!useFullPbr) {
            mediump float pureNdotL = max(dot(N, L), 0.0);
            float spec   = pow(NdotH, specExp) * pureNdotL;
            lightSpec += F0 * radiance * spec *
                         ((specExp + 2.0) / (8.0 * PI)) * baseLayerWeight;
            if (hasClearcoat) {
                float ccR = matClearcoatRoughness;
                float ccExp = max(2.0, (1.0 - ccR) * (1.0 - ccR) * 128.0);
                float ccSpec = pow(NdotH, ccExp) * pureNdotL;
                lightSpec += vec3(0.04) * radiance * ccSpec * matClearcoat
                           * ((ccExp + 2.0) / (8.0 * PI)) * coatTint;
            }
        } else {
            mediump float pureNdotL = NdotL;
            mediump float pureNdotV = max(dot(N, V), 0.0);
            lightSpec += evaluateSpecularGGXPrecomputed(
                             pureNdotV, pureNdotL, NdotH, VdotH,
                             F0, directSpecularRoughness) *
                         radiance * pureNdotL * baseLayerWeight;
            if (hasClearcoat) {
                lightSpec += evaluateSpecularGGXPrecomputed(
                    pureNdotV, pureNdotL, NdotH, VdotH,
                    vec3(0.04), clamp(matClearcoatRoughness, 0.02, 1.0)) *
                    radiance * pureNdotL * matClearcoat * coatTint;
            }
        }
        // Without glass/resin attenuation, diffuse and specular sum directly;
        // folding into diffuseLit saves 3 live registers across the light loop.
        if (RT_SPEC_GLASS) specularLit += lightSpec;
        else diffuseLit += lightSpec;

        // ── Sheen lobe (fabric / velvet) ──
        if (sheenWeight > 0.001) {
            sheenLit += evaluateSheen(N, V, L, sheenColor, roughness) *
                        radiance * baseLayerWeight;
        }
    }

    // Lightning flash from the lit cloud base: diffuse lobe only (the disc is
    // kilometres wide; its specular highlight is the sky reflection's job).
    if (RT_SPEC_WEATHER && lightingPreset == 3u) {
        vec3 flashE = previewLightningIrradiance(vWorldPos, N);
        if (any(greaterThan(flashE, vec3(0.0)))) {
            mediump vec3 Favg = F0 + (vec3(1.0) - F0) * (1.0 / 21.0);
            diffuseLit += (vec3(1.0) - Favg) * (1.0 / PI) * flashE * diffuseSurface *
                          (1.0 - translucency);
        }
    }

    mediump float NdotV_main = max(dot(N, V), 0.0);
    bool hasEnvClearcoat = hasClearcoat;
    mediump float coatViewFresnel = hasEnvClearcoat
        ? fresnelSchlick(NdotV_main, vec3(0.04)).r * clamp(matClearcoat, 0.0, 1.0)
        : 0.0;
    mediump float ambientBaseWeight = 1.0 - coatViewFresnel;

    mediump vec3 ambient = vec3(0.0);
    mediump vec3 envSpecular = vec3(0.0);
    mediump vec3 rfEnvSpecularWeight = vec3(0.0);
    if (lightingPreset == 3u) {
        mediump vec3 R = reflect(-V, N);
        if (!RT_SPEC_FALLBACK_SKY || (sceneFlags & 32u) != 0u) {
            mediump float intensity = worldMode == 1u ? max(worldParams.y, 0.0) : 1.0;
            mediump vec3 irradiance = vec3(0.0);
            float screenGiConfidence = 0.0;
            bool screenGiValid = RT_SPEC_RAYFUSION_GI
                ? sampleScreenGi(N, irradiance, screenGiConfidence)
                : false;
            vec3 probeIrradiance;
            if (!screenGiValid || screenGiConfidence < 1.0) {
                mediump float skyVisibility = 1.0;
                if (RT_SPEC_RAYFUSION_GI && !sampleScreenGiSkyVisibility(N, skyVisibility))
                    skyVisibility = rfSpecularSkyVisibility(vWorldPos, N, N);

                mediump vec3 fallbackIrradiance = texture(worldIrradiance, worldDirToUV(N)).rgb * intensity;
                if (RT_SPEC_RAYFUSION_GI && rfSampleProbeField(vWorldPos, N, probeIrradiance))
                    fallbackIrradiance = probeIrradiance * intensity;
                fallbackIrradiance *= skyVisibility;
                irradiance = screenGiValid ? mix(fallbackIrradiance, irradiance, screenGiConfidence) : fallbackIrradiance;
            }
            mediump vec3 prefiltered = textureLod(worldPrefiltered, worldDirToUV(R),
                                          reflectionRoughness * 8.0).rgb * intensity;
            mediump vec2 brdf = texture(worldBrdfLut,
                                vec2(clamp(NdotV_main, 0.001, 0.999), reflectionRoughness)).rg;
            mediump vec3 fresnelAmb = fresnelSchlickRoughness(
                NdotV_main, F0, reflectionRoughness);
            ambient = irradiance * diffuseColor * (vec3(1.0) - fresnelAmb) *
                      ambientBaseWeight;
            envSpecular = prefiltered * (F0 * brdf.x + brdf.y) *
                          ambientBaseWeight;
            rfEnvSpecularWeight = (F0 * brdf.x + brdf.y) * ambientBaseWeight;
            // Note: reflectionRoughness is clamped to [0.04, 1.0] above, so the
            // legacy `reflectionRoughness <= 0.025` direct sun-disc branch was
            // mathematically unreachable yet forced sampleCanonicalWorld() and
            // all atmosphere/world LUT samplers into every prefiltered IBL shader.
            if (hasEnvClearcoat) {
                mediump vec3 coatEnvTint = RT_SPEC_CLEARCOAT_IRID
                    ? clearcoatFilmTint(
                        NdotV_main, matx.clearcoat_iridescence,
                        matx.clearcoat_film_thickness)
                    : vec3(1.0);
                mediump float ccRoughness = clamp(matClearcoatRoughness, 0.02, 1.0);
                mediump vec3 ccEnv = textureLod(worldPrefiltered, worldDirToUV(R),
                                        ccRoughness * 8.0).rgb * intensity;
                mediump vec2 ccBrdf = texture(worldBrdfLut,
                                      vec2(clamp(NdotV_main, 0.001, 0.999),
                                           ccRoughness)).rg;
                envSpecular += ccEnv * (vec3(0.04) * ccBrdf.x + ccBrdf.y) *
                               matClearcoat * coatEnvTint;
            }
        } else {
            // Fully metallic/transmitting surfaces have no diffuse term. Avoid
            // the 5/9-tap sky integral when its contribution is exactly zero;
            // partial materials retain the complete existing filter.
            vec3 envDiffuse = vec3(0.0);
            if (any(notEqual(diffuseColor, vec3(0.0)))) {
                envDiffuse = worldMode == 2u
                    ? samplePhysicalSkyFiltered(N, 1.0)
                    : sampleCanonicalWorld(N);
            }
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
            rfEnvSpecularWeight = fresnelAmb *
                          mix(1.0, 0.18, reflectionRoughness * reflectionRoughness) *
                          ambientBaseWeight;
            if (hasEnvClearcoat) {
                vec3 coatEnvTint = RT_SPEC_CLEARCOAT_IRID
                    ? clearcoatFilmTint(
                        NdotV_main, matx.clearcoat_iridescence,
                        matx.clearcoat_film_thickness)
                    : vec3(1.0);
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
        if (hasEnvClearcoat) {
            vec3 coatEnvTint = RT_SPEC_CLEARCOAT_IRID
                ? clearcoatFilmTint(
                    NdotV_main, matx.clearcoat_iridescence,
                    matx.clearcoat_film_thickness)
                : vec3(1.0);
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
    ambient = ambient * (1.0 - translucency);
    if (translucency > 0.0 && any(notEqual(diffuseColor, vec3(0.0)))) {
        vec3 backEnvironment = lightingPreset == 3u
            ? sampleCanonicalWorld(-N)
            : samplePreviewEnvironment(-N, lightingPreset);
        ambient += backEnvironment * diffuseColor * translucency *
                   (1.0 / PI) * ambientBaseWeight;
    }

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
    // Only fragments that can transmit pay for the refracted environment,
    // the resin interior and the glass tint. Outside this gate every consumer
    // below is multiplied by zero or sits behind `transmission > 0.001`
    // (transmissionLit, resinBaseWeight, dielectricPass), so the result is
    // bit-identical -- but an opaque wall used to evaluate 5-9 environment
    // taps and an FBM per pixel and then throw them away (measured with malioc,
    // PLAYER_ANDROID_PERF_PLAN B8). The transmission phase is unaffected: it
    // discards non-transmissive fragments before this point.
    bool glassPath = RT_SPEC_GLASS &&
        (bubbleSurface || transmission > 0.0 || interiorDepth > 1e-4);
    vec3 transmittedWorld = vec3(0.0);
    if (!glassPath) {
        // Opaque surface: nothing below reads the refracted radiance.
    } else if (!bubbleSurface && interiorDepth <= 1e-4 && mat.dispersion > 1e-3) {
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
    float frontViewDistance = max(-(rayFusionFrame.view * vec4(vWorldPos, 1.0)).z, 0.0);
    uint backBits = RT_SPEC_GLASS && drawPhase != PREVIEW_PHASE_COMBINED
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
    PreviewInteriorSample interior;
    interior.transmittance = vec3(1.0);
    interior.scatter = vec3(0.0);
    if (glassPath) interior = previewResinInterior(
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
    if (glassPath && interiorDepth <= 1e-4 && !bubbleSurface) {
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
    if (transmissionPhase && !bubbleSurface) {
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
                float chIor = channel == 0 ? channelIor.x : (channel == 1 ? channelIor.y : channelIor.z);
                vec3 d = refract(incoming, N, 1.0 / chIor);
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
                vec3 rough = sampleOpaqueSceneRough(channelUv, roughness);
                float sv = channel == 0 ? rough.x : (channel == 1 ? rough.y : rough.z);
                sceneThrough = vec3(channel == 0 ? sv : sceneThrough.x,
                                    channel == 1 ? sv : sceneThrough.y,
                                    channel == 2 ? sv : sceneThrough.z);
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
    if (transmissionPhase) {
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

    if (RT_SPEC_RAYFUSION_GI && !sceneReflectionValid && lightingPreset == 3u)
        envSpecular *= rfSpecularSkyVisibility(vWorldPos, N, reflect(-V, N));

    if (!RT_SPEC_OPT_TEX && !RT_SPEC_GRAPH) {
        mediump float emStr = max(materials[materialIndex].emission_strength, 0.0);
        if (emStr > 0.0) {
            emission = vec3(materials[materialIndex].emission_r,
                            materials[materialIndex].emission_g,
                            materials[materialIndex].emission_b) * emStr;
        }
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
    // *** GORUNTULEME DONUSUMU BURADAN SOKULDU (2026-09-06).
    //   Bu shader artik SCENE-LINEAR yaziyor; zincir (exposure -> operator ->
    //   grade -> vignette -> sRGB) tek yerde, `raster_post.comp` icinde kosuyor.
    //   Zorunluydu: alan derinligi bu shader'larin ciktisini BULANISTIRIR ve
    //   bokeh, parlak noktanin daire olarak acilmasidir. Tonemap'ten gecmis bir
    //   deger o noktayi zaten kirpmistir; onu bulanistirmak gri leke uretir.

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
    if (transmissionPhase) {
        if (sceneRefractionValid) {
            // * Anlik goruntu ARTIK O DA scene-linear: ayni uzayda
            //   topluyoruz, "iki kez tonemap" tehlikesi yok -- zincir bu
            //   shader'dan sonra, tek sefer kosuyor.
            color += sceneThrough * dielectricPass;
        }
        // Both scene continuation and refracted-environment fallback already
        // contain the complete reflection/transmission split. Alpha-blending
        // the live framebuffer again added an un-refracted sky term and was the
        // source of the homogeneous atmosphere-coloured glass body.
        outputOpacity = 1.0;
    }
    if (any(isnan(color)) || any(isinf(color))) {
        color = vec3(0.0);
    }
    color = clamp(color, vec3(0.0), vec3(65000.0));
    outColor = vec4(color, outputOpacity);

    // ─── RayFusion reflection G-buffer ───────────────────────────────────────
    // ★★★ Speküler agirligi UC durumda sifirliyoruz, ve ucu de "yansima yok"
    //   degil "bu yuzey bu karede yansima BEYAN ETMIYOR" demek:
    //
    //   1. Transmission fazi. Cam/resin yuzeyinin yansimasini ekran uzayi
    //      yolu (tracePreviewOpaqueReflection) zaten uretiyor; buraya
    //      yazsaydik, ARKASINDAKI opak yuzeyin normalini silerdik -- derinlik
    //      testi bir yarisaydam fragment'i elemez. Belirtisi "camin arkasindaki
    //      metal yansimayi kaybetti" olurdu ve sebebi camda gorunmezdi.
    //   2. Ekran uzayi yansimasi GECERLI. `envSpecular` yukarida sifirlandi;
    //      agirligi burada birakmak ayni yansimayi IKI kez eklerdi.
    //   3. Scene lighting disinda `rfEnvSpecularWeight` hic doldurulmadi.
    //
    // ★ `reflectionRoughness` her zaman yazilir: kapi agirlikta, roughness'ta
    //   degil. Kapiyi roughness'a kurmak temizlenmis her texel'i (gokyuzu
    //   dahil) "mukemmel ayna" ilan etmek olurdu.
    vec3 rfWeight = rfEnvSpecularWeight;
    if (transmissionPhase || sceneReflectionValid)
        rfWeight = vec3(0.0);
    // ★★★★ DORDUNCU durum ve en sinsisi: YARISAYDAM fragment. Yansima gecisi
    //   `agirlik * (izlenen - env)` ekliyor, yani fragment shader'in env
    //   terimini EKLEDIGINI varsayiyor. Blend acikken eklenen miktar alfa ile
    //   olceklenir (`src * alpha`), ama G-buffer'a yazilan agirlik
    //   olceklenmemistir -- geri cikarma FAZLA cikarir ve belirtisi "yari
    //   saydam yuzeyler yansimada kararmis" olur. Kimse bunu yansima hatasi
    //   diye raporlamaz. Bu yuzden tam opak olmayan fragment env yolunda kalir.
    if (outputOpacity < 0.999) rfWeight = vec3(0.0);
    outRfNormal = rfOctEncodeNormal(N);
    outRfSpecular = vec4(max(rfWeight, vec3(0.0)), reflectionRoughness);
    if (vPreviousClip.w > 1e-9 && vCurrentClip.w > 1e-9) {
        vec2 deltaUv = (vPreviousClip.xy / vPreviousClip.w -
                        vCurrentClip.xy / vCurrentClip.w) * 0.5;
        outPreviousClip = clamp(deltaUv, vec2(-100.0), vec2(100.0));
    } else {
        outPreviousClip = vec2(-65504.0, 0.0);
    }
}
