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
    vec4 origin;
    vec4 lowerLeft;
    vec4 horizontal;
    vec4 vertical;
    uint frameCount;
    uint minSamples;
    uint lightCount;
    float varianceThreshold;
    uint maxSamples;
    float exposure_factor;
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
    vec3     radiance;       // Bu bounce'un emit ettiği ışık
    vec3     attenuation;    // Birikimli throughput (path tracing weight)
    vec3     scatterOrigin;  // Bir sonraki ray'in başlangıcı
    vec3     scatterDir;     // Bir sonraki ray'in yönü
    uint     seed;           // PCG RNG state
    bool     scattered;      // false → ray absorbe edildi, döngü dur
    bool     hitEmissive;    // raygen'de emission'ı ayrı handle etmek için
    uint     occluded;       // shadow test flag (0 = clear, 1 = occluded)
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
    uint _reserved_0, _reserved_1, _reserved_2, _reserved_3;
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
vec3 evaluate_brdf_gl(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic) {
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 1e-4);
    float NdotL = max(dot(N, L), 1e-4);
    float NdotH = max(dot(N, H), 1e-4);
    float VdotH = max(dot(V, H), 1e-4);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    // Fresnel
    float f = pow(1.0 - VdotH, 5.0);
    vec3 F = F0 + (vec3(1.0) - F0) * f;
    // D (GGX)
    float alpha = max(roughness * roughness, 1e-4);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (3.14159265 * denom * denom + 1e-8);
    // G (Smith)
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float G = (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));
    vec3 spec = (F * D * G) / (4.0 * NdotV * NdotL + 1e-6);
    // Diffuse
    vec3 F_avg = F0 + (vec3(1.0) - F0) / 21.0;
    vec3 k_d = (vec3(1.0) - F_avg) * (1.0 - metallic);
    vec3 diff = (k_d * albedo) / 3.14159265;
    return diff + spec;
}

// BRDF PDF approx (GGX-based)
float pdf_brdf_gl(vec3 N, vec3 V, vec3 L, float roughness) {
    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 1e-4);
    float VdotH = max(dot(V, H), 1e-4);
    float alpha = max(roughness * roughness, 1e-4);
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
    float prob_to_reach = 1.0;
    for (int i = 0; i < light_count; ++i) {
        if (lights.l[i].position.w == 1.0) {
            if (rng < 0.33) { pdf_out = 0.33 * prob_to_reach; return i; }
            rng = (rng - 0.33) / 0.67;
            prob_to_reach *= 0.67;
        }
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

// GGX NDF — importance sampling (metallic/rough yüzeyler için)
// PDF = D(h) * cos(theta_h) / (4 * dot(v, h))
vec3 ggxSampleHemisphere(vec3 normal, vec3 viewDir, float roughness, inout uint seed) {
    float r1    = rnd(seed);
    float r2    = rnd(seed);
    float alpha = roughness * roughness;     // perceptual → linear roughness

    // GGX mikrofacet normal örnekle
    float phi       = TWO_PI * r1;
    float cosTheta  = sqrt((1.0 - r2) / max(1.0 + (alpha * alpha - 1.0) * r2, 1e-7));
    float sinTheta  = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    // Tangent uzayında half vector
    vec3 halfVecLocal = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    // Dünya uzayına dönüştür
    vec3 tangent, bitangent;
    buildONB(normal, tangent, bitangent);
    vec3 halfVec = normalize(tangent * halfVecLocal.x + bitangent * halfVecLocal.y + normal * halfVecLocal.z);

    // Half vector'den reflect yönünü hesapla
    return reflect(-viewDir, halfVec);
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

    vec3 scatterDir;
    if (roughness < 0.01) {
        // Pürüzsüz ayna: tam reflection
        scatterDir = reflect(rayDir, normal);
    } else {
        scatterDir = ggxSampleHemisphere(normal, viewDir, roughness, seed);
    }

    // Yüzeyin altına düştüyse absorbe et
    if (dot(scatterDir, normal) <= 0.0) {
        payload.scattered = false;
        return;
    }

    // Metal Fresnel: F0 = albedo (metalik yüzeyler için)
    float cosTheta = max(dot(viewDir, normal), 0.0);
    vec3  fresnel  = schlickFresnelVec(cosTheta, albedo);

    payload.scatterOrigin = hitPos + normal * RAY_OFFSET;
    payload.scatterDir    = scatterDir;
    payload.attenuation  *= fresnel;
    payload.scattered     = true;
}

// --- Dielectric Glass (Fresnel + TIR) ---
void scatterGlass(vec3 hitPos, vec3 normal, vec3 rayDir, vec3 albedo, float ior, inout uint seed) {
    // Işığın hangi taraftan geldiğini belirle
    bool  frontFace    = dot(rayDir, normal) < 0.0;
    vec3  outNormal    = frontFace ? normal : -normal;
    float etaRatio     = frontFace ? (1.0 / ior) : ior;

    float cosTheta     = min(dot(-rayDir, outNormal), 1.0);
    float sinTheta     = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    bool  totalIntRefl = (etaRatio * sinTheta) > 1.0;

    float fresnelProb  = schlickFresnel(cosTheta, ior);
    bool  doReflect    = totalIntRefl || (rnd(seed) < fresnelProb);

    vec3 dir;
    vec3 offsetDir;
    if (doReflect) {
        dir       = reflect(rayDir, outNormal);
        offsetDir = outNormal;           // Yüzeyin dışına offset
    } else {
        dir       = refract(rayDir, outNormal, etaRatio);
        offsetDir = -outNormal;          // Yüzeyin içine offset (refract için)
    }

    payload.scatterOrigin = hitPos + offsetDir * RAY_OFFSET;
    payload.scatterDir    = normalize(dir);
    payload.attenuation  *= albedo;      // Berrak cam için albedo = vec3(1)
    payload.scattered     = true;
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
    // 2. Barycentric koordinatlarla smooth normal hesapla
    // ----------------------------------------------------------
    vec3 bary = vec3(1.0 - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);
    vec3 worldNormal;

    if (geo.normalAddr != 0) {
        NormalBuffer nBuf = NormalBuffer(geo.normalAddr);

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

        vec3 localNormal = nBuf.n[i0] * bary.x
                         + nBuf.n[i1] * bary.y
                         + nBuf.n[i2] * bary.z;

        // Object → world dönüşümü (ölçeği yok saymak için: inverse transpose)
        worldNormal = normalize(vec3(localNormal * mat3(gl_WorldToObjectEXT)));
    } else {
        // Normal buffer yoksa ham üçgen normalini hesapla
        if (geo.vertexAddr != 0) {
            VertexBuffer vBuf = VertexBuffer(geo.vertexAddr);
            uint base = uint(gl_PrimitiveID) * 3;
            vec3 v0 = vBuf.v[base + 0];
            vec3 v1 = vBuf.v[base + 1];
            vec3 v2 = vBuf.v[base + 2];
            vec3 localFaceNormal = normalize(cross(v1 - v0, v2 - v0));
            worldNormal = normalize(vec3(localFaceNormal * mat3(gl_WorldToObjectEXT)));
        } else {
            worldNormal = normalize(vec3(0, 1, 0));  // Fallback
        }
    }

    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 rayDir = normalize(gl_WorldRayDirectionEXT);

    // Compute UV coordinates if available
    vec2 hitUV = vec2(0.0);
    if (geo.uvAddr != 0) {
        UVBuffer uvBuf = UVBuffer(geo.uvAddr);
        uint ui0, ui1, ui2;
        if (geo.indexAddr != 0) {
            IndexBuffer iBuf = IndexBuffer(geo.indexAddr);
            ui0 = iBuf.i[gl_PrimitiveID * 3 + 0];
            ui1 = iBuf.i[gl_PrimitiveID * 3 + 1];
            ui2 = iBuf.i[gl_PrimitiveID * 3 + 2];
        } else {
            ui0 = uint(gl_PrimitiveID) * 3 + 0;
            ui1 = uint(gl_PrimitiveID) * 3 + 1;
            ui2 = uint(gl_PrimitiveID) * 3 + 2;
        }
        hitUV = uvBuf.u[ui0] * bary.x + uvBuf.u[ui1] * bary.y + uvBuf.u[ui2] * bary.z;
    }

    // Vulkan shader coordinate origin differs; flip V to match OptiX (and texture upload)
    hitUV.y = 1.0 - hitUV.y;

    // Double-sided: normal her zaman ray'e karşı baksın
    if (dot(worldNormal, rayDir) > 0.0) {
        worldNormal = -worldNormal;
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
    //    raygen shader payload.radiance'ı throughput ile çarpar
    // ----------------------------------------------------------
    payload.radiance  = emColor * emStrength;
    payload.hitEmissive = (emStrength > 0.001);

    // Sample albedo texture (if present)
    int albedoTexID = int(mat.albedo_tex);
    if (albedoTexID > 0) {
        vec3 texCol = texture(materialTextures[nonuniformEXT(albedoTexID)], hitUV).rgb;
        albedo *= texCol;
    }
    
    // Sample opacity texture (if present)
    float textureOpacity = 1.0;
    int opacityTexID = int(mat.opacity_tex);
    if (opacityTexID > 0) {
        textureOpacity = texture(materialTextures[nonuniformEXT(opacityTexID)], hitUV).r;
    }
    // Combine material opacity with texture opacity
    float finalOpacity = mat.opacity * textureOpacity;
    
    // Sample roughness texture
    int roughTexID = int(mat.roughness_tex);
    if (roughTexID > 0) {
        float r = texture(materialTextures[nonuniformEXT(roughTexID)], hitUV).r;
        roughness = clamp(roughness * r, 0.0, 1.0);
    }
    
    // Sample metallic texture
    int metallicTexID = int(mat.metallic_tex);
    if (metallicTexID > 0) {
        float m = texture(materialTextures[nonuniformEXT(metallicTexID)], hitUV).r;
        metallic = clamp(metallic * m, 0.0, 1.0);
    }
    
   // Sample emission texture
int emissionTexID = int(mat.emission_tex);
if (emissionTexID > 0) {
    vec3 emTex = texture(materialTextures[nonuniformEXT(emissionTexID)], hitUV).rgb;

    float matEmLum = dot(emColor, vec3(0.2126, 0.7152, 0.0722));
    if (matEmLum < 0.01) {
        // mat.emission_rgb set edilmemiş → texture kendi rengiyle parlasın
        emColor = emTex;
    } else {
        // mat.emission_rgb bir tint olarak davransın
        emColor = emColor * emTex;
    }
} else if (emStrength > 0.001) {
    // Emission texture yok ama strength > 0 → albedo rengini kullan (Blender default)
    float matEmLum = dot(emColor, vec3(0.2126, 0.7152, 0.0722));
    if (matEmLum < 0.01) {
        emColor = albedo; // albedo texture zaten yukarıda uygulandı
    }
}
    
    // Sample transmission texture (for glass/transparent materials)
    int transmissionTexID = int(mat.transmission_tex);
    if (transmissionTexID > 0) {
        float trans = texture(materialTextures[nonuniformEXT(transmissionTexID)], hitUV).r;
        transmission = clamp(transmission * trans, 0.0, 1.0);
    }
    
    // Alpha cutout threshold (universal for all materials)
    const float OPACITY_THRESHOLD = 0.5;
    
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
    // Stochastic Transparency — Probabilistic transmission ray
    // ----------------------------------------------------------
    // Match CPU/OptiX behavior: if opacity < 1.0, stochastically continue or absorb
    // Probability of continuing = finalOpacity (pass-through)
    // Probability of absorbing = 1 - finalOpacity
    if (finalOpacity < 0.999) {
        float rnd_val = rnd(payload.seed);
        if (rnd_val > finalOpacity) {
            // TRANSMIT: Create new ray path through geometry
            // Move ray origin forward slightly to avoid re-hitting same geometry
            vec3 transmitOrigin = hitPos + rayDir * 1e-3;
            vec3 transmitDir = rayDir;  // Continue in same direction (refraction without bending)
            
            // Spawn transmission ray recursively
            payload.scatterOrigin = transmitOrigin;
            payload.scatterDir = transmitDir;
            payload.scattered = true;
            // Note: attenuation is NOT multiplied by transmission color here
            // (could be enhanced with Beer's law transmission color later)
            return;
        }
        // Else: ray absorbed (probabilistic opacity cutout)
        // Continue to direct lighting with reduced probability
        // Optional: uncomment line below to make absorption more frequent
        // payload.attenuation *= finalOpacity;
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
                        shadowOccluded = true; // INITIALIZE TO TRUE
                        vec3 shadowOrigin = hitPos + worldNormal * RAY_OFFSET;
                        float tmin = 1e-3;
                        float tmax = min(max(0.0, dist - 1e-3), 10000.0);
                        if (tmax > tmin) {
                            uint shadowFlags = gl_RayFlagsTerminateOnFirstHitEXT
                                             | gl_RayFlagsSkipClosestHitShaderEXT
                                             | gl_RayFlagsOpaqueEXT;
                            // sbtOffset=0, sbtStride=0, missIndex=0 (use main miss), payloadLocation=1
                            traceRayEXT(topLevelAS, shadowFlags, 0xFF, 0, 0, 0, shadowOrigin, tmin, wi, tmax, 1);
                        }
                        if (!shadowOccluded) {
                            vec3 V = normalize(-rayDir);
                            vec3 brdf = evaluate_brdf_gl(worldNormal, V, wi, albedo, roughness, metallic);
                            vec3 Li = lights.l[lightIdx].color.rgb * lights.l[lightIdx].color.a * lightAtten;
                            
                            // MIS logic: pdf_light is the probability of picking this direction
                            float pdf_light_area = compute_light_pdf_gl(lights.l[lightIdx], dist, 1.0);
                            float pdf_light_total = pdf_light_area * pdf_select;
                            float pdf_brdf = pdf_brdf_gl(worldNormal, V, wi, roughness);
                            
                            float w = power_heuristic(pdf_light_total, pdf_brdf);
                            
                            // Standard Monte Carlo estimator: (Integral of f*L) ≈ f * L / p
                            // Result = (brdf * Li * NdotL * w) / pdf_light_total
                            vec3 contrib = (brdf * Li * NdotL * w) / max(pdf_light_total, 1e-6);
                            payload.radiance += payload.attenuation * contrib;
                        }
                    }       // ← NdotL if
                }           // ← else (length check)
            }               // ← ok if
        }                   // ← lightIdx if
    }                       // ← direct lighting scope
    // ----------------------------------------------------------
    // 5. Scatter kararı — Principled BSDF
    // ----------------------------------------------------------

    // Glass (transmission öncelikli) — with proper opacity handling
    if (transmission > 0.5) {
        // Opacity already checked above (line ~688), no need to check again
        scatterGlass(hitPos, worldNormal, rayDir, albedo, ior, payload.seed);
        return;
    }

    // Metallic / Diffuse blend
    // Enerji korunumu: diffuse + metal = 1
    // Stochastic seçim → attenuation PDF'i doğru kompanse eder
    float diffuseWeight = 1.0 - metallic;
    float metalWeight   = metallic;

    // Apply clearcoat layer if present (top surface)
    if (mat.clearcoat > 0.01) {
        // Clearcoat creates a smooth reflective layer
        float clearcoatWeight = clamp(mat.clearcoat, 0.0, 1.0);
        float clearcoatRoughness = clamp(mat.clearcoat_roughness, 0.0, 1.0);
        // Blend roughness values: coated surfaces appear smoother overall
        roughness = mix(roughness, clearcoatRoughness, clearcoatWeight * 0.5);
    }

    if (metallic >= 0.999) {
        // Tam metal
        scatterMetal(hitPos, worldNormal, rayDir, albedo, roughness, payload.seed);
    }
    else if (metallic <= 0.001) {
    // Dielektrik Fresnel — F0 = 0.04 (non-metal standart)
    // Grazing açılarda ve düşük roughness'ta yansıma göster
    const float F0_DIELECTRIC = 0.04;
    vec3 viewDir = -rayDir;
    float cosTheta = max(dot(viewDir, worldNormal), 0.0);

    // Schlick Fresnel: grazing açıda artar, roughness düştükçe etkisi belirginleşir
    float fresnelBase = F0_DIELECTRIC + (1.0 - F0_DIELECTRIC)
                        * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);

    // Roughness yüksekse Fresnel yansıması körleşir — mat yüzeyler daha az yansır
    // roughness=0 → tam fresnel, roughness=1 → neredeyse saf diffuse
    float fresnelWeight = fresnelBase * (1.0 - roughness * roughness);

    if (rnd(payload.seed) < fresnelWeight) {
        // Specular lob: GGX reflection (roughness=0 → ayna gibi)
        scatterMetal(hitPos, worldNormal, rayDir, vec3(1.0), roughness, payload.seed);
        // Fresnel seçim PDF'ini kompanse et; attenuation tint yok (dielektrik = renksiz spec)
        float safeW = max(fresnelWeight, 0.01);
        payload.attenuation *= (F0_DIELECTRIC / safeW);
    } else {
        // Diffuse lob
        scatterDiffuse(hitPos, worldNormal, albedo, payload.seed);
        
        // Diffuse enerji: Fresnel'in yansımadığı kısım
        float safeW = max(1.0 - fresnelWeight, 0.01);
        payload.attenuation *= ((1.0 - F0_DIELECTRIC) / safeW);

        // Subsurface katkısı (değişmedi)
        if (mat.subsurface_amount > 0.01) {
            float ssStrength = clamp(mat.subsurface_amount, 0.0, 1.0);
            vec3 subsurfaceColor = vec3(mat.subsurface_r, mat.subsurface_g, mat.subsurface_b);
            payload.attenuation *= mix(vec3(1.0), subsurfaceColor, ssStrength * 0.3);
        }
    }
}
    else {
        // Blend: stochastic seçim
        // Seçilen lob'un PDF'ini kompanse etmek için weight ile böl
        if (rnd(payload.seed) < metalWeight) {
            scatterMetal(hitPos, worldNormal, rayDir, albedo, roughness, payload.seed);
            // Metal lobu seçildi; diffuse lobu da katkı yapar → weight ile ölçekle
            float safeWeight = max(metalWeight, 0.1);
            payload.attenuation *= (1.0 / safeWeight);
        } else {
            scatterDiffuse(hitPos, worldNormal, albedo, payload.seed);
            float safeWeight = max(diffuseWeight, 0.1);
            payload.attenuation *= (1.0 / safeWeight);
        }
    }
}
