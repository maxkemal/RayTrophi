// bsdf_scatter.glsl
// The Principled BSDF scatter lobes, direct-light sampling and the
// water/resin helpers they need — shared by closesthit.rchit (real geometry)
// and volume_closesthit.rchit (the fluid SDF isosurface).
//
// WHY THIS FILE EXISTS
// Shader modules in a ray-tracing pipeline are compiled independently: the
// volume closest-hit cannot call into the surface closest-hit, and it cannot
// include it either (that file owns main() and its own bindings). So a liquid
// surface could only ever be shaded by a SECOND, hand-written model — which is
// how the fluid isosurface ended up as a bare Fresnel + Beer-Lambert dielectric
// while the material system already carried random-walk SSS, clearcoat and
// transmission. Lifting the lobes here is what lets a fluid surface take an
// ORDINARY scene material: molten glass, lava, mud and chocolate stop being
// special cases and become material settings.
//
// It also collapses a duplication that had already started: sampleHG was
// defined separately in both shaders. Two copies agree until the day one of
// them is tuned, and then the liquid and the solid scatter differently with
// nothing to indicate it.
//
// ENVIRONMENT INTERFACE
// The includer must provide these BEFORE including this file. They are left out
// deliberately rather than moved: each shader already has its own, and the
// volume shader's volume-density sampler is the better one for its own grids.
//   uint  pcgNext(inout uint state);
//   float rnd(inout uint seed);
//   float computeVolumeShadowTransmittance(vec3 origin, vec3 dir, float maxDist);
// plus the bindings the lobes read: payload, cam, lights, worldData, topLevelAS,
// and the Material/MaterialExt structs (material_struct.glsl) and water_v3.glsl.

#ifndef BSDF_SCATTER_GLSL
#define BSDF_SCATTER_GLSL

// Provided by the including shader — see ENVIRONMENT INTERFACE above.
float rnd(inout uint seed);
float computeVolumeShadowTransmittance(vec3 shadowOrigin, vec3 lightDir, float maxDist);

// Result of one interior resin march. Moved with the code that fills it;
// LightData deliberately did NOT move — each shader declares its own
// alongside its light binding.
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

// -- Principled layer selection ----------------------------------------------
// The stochastic lobe pick that used to live inline in closesthit.rchit's
// main(). It is here so the fluid SDF isosurface runs the SAME selection: a
// liquid with a scene material assigned gets clearcoat, metal, transmission and
// SSS by exactly the rules a triangle does, instead of the bespoke Fresnel +
// Beer-Lambert dielectric that shader used to hard-code.
//
// Everything it needs arrives RESOLVED in SurfaceSample - albedo after textures
// and the material graph, normal after normal mapping. That is what lets one
// function serve a rasterised triangle and a raymarched isosurface: by this
// point neither is geometry any more, just a shaded point.
struct SurfaceSample {
    vec3  P;              // hit position, world
    vec3  N;              // shading normal, world (faces the incoming ray)
    vec3  rayDir;         // incoming direction
    vec3  albedo;
    float roughness;
    float metallic;
    float specular;
    float clearcoat;
    float clearcoatRoughness;
    float clearcoatIridescence;
    float clearcoatFilmThickness;
    float translucent;
    float subsurfaceAmount;
    vec3  subsurfaceColor;
    vec3  subsurfaceRadius;
    float subsurfaceScale;
    float subsurfaceAnisotropy;
};

// Zero state, so a caller only fills what it actually has.
SurfaceSample defaultSurfaceSample() {
    SurfaceSample s;
    s.P = vec3(0.0); s.N = vec3(0.0, 1.0, 0.0); s.rayDir = vec3(0.0, -1.0, 0.0);
    s.albedo = vec3(0.5); s.roughness = 0.5; s.metallic = 0.0; s.specular = 0.5;
    s.clearcoat = 0.0; s.clearcoatRoughness = 0.03;
    s.clearcoatIridescence = 0.0; s.clearcoatFilmThickness = 0.0;
    s.translucent = 0.0;
    s.subsurfaceAmount = 0.0; s.subsurfaceColor = vec3(1.0);
    s.subsurfaceRadius = vec3(1.0); s.subsurfaceScale = 1.0;
    s.subsurfaceAnisotropy = 0.0;
    return s;
}

// Layer order (top to bottom), stochastic selection each bounce:
//   1. Clearcoat (IOR=1.5 GGX specular, weight = clearcoat * fresnel)
//   2. Metallic (if metallic > 0)
//   3. Dielectric specular (Fresnel) or diffuse sub-layer:
//        a. Translucent (thin-surface transmission)
//        b. SSS random walk
//        c. Lambertian diffuse
void scatterPrincipled(SurfaceSample s, inout uint seed) {
    // --- Clearcoat layer (stochastic, IOR=1.5, F0=0.04) ---
    if (s.clearcoat > 0.01) {
        vec3  viewDirCC  = -s.rayDir;
        float cosNV_CC   = max(dot(viewDirCC, s.N), 0.0);
        const float CC_F0 = 0.04;
        float ccFresnel  = CC_F0 + (1.0 - CC_F0) * pow(1.0 - cosNV_CC, 5.0);
        // Probability of choosing clearcoat lobe = clearcoat weight * fresnel.
        //
        // ★ Capped at 0.9. The compensation below divides by (1 - ccProb), so a
        // grazing hit on a high-clearcoat material (ccFresnel -> 1) used to
        // weight the base layer by up to 1/0.01 = 100x on the draws where the
        // lottery skipped the coat. That is unbiased but wildly high-variance:
        // it makes fireflies, and it makes them at GRAZING ANGLES specifically.
        // A mesh has few of those. A liquid is almost nothing but — curved
        // droplets, thin sheets, edges — so the fluid isosurface turned a
        // long-standing variance problem into visible glare piling up along
        // exactly those boundaries. Bounding the probability bounds the weight
        // at 10x. Slightly biased, vastly quieter, and the bias sits where the
        // coat is already nearly opaque.
        float ccProb = min(s.clearcoat * ccFresnel, 0.9);
        if (rnd(seed) < ccProb) {
            scatterClearcoat(s.P, s.N, s.rayDir, s.clearcoatRoughness,
                             s.clearcoatIridescence, s.clearcoatFilmThickness, seed);
            // Compensate selection probability
            payload.attenuation *= (1.0 / max(ccProb, 0.01));
            return;
        }
        // Base layer continues; compensate probability of NOT picking clearcoat
        payload.attenuation *= (1.0 / max(1.0 - ccProb, 0.01));
    }

    // --- Metallic / Diffuse blend ---
    float metalWeight = s.metallic;

    if (s.metallic >= 0.999) {
        // Pure metal
        scatterMetal(s.P, s.N, s.rayDir, s.albedo, s.roughness, seed);
    }
    else if (s.metallic <= 0.001) {
        // Dielectric Fresnel - F0 = 0.04 (non-metal standard)
        float F0_DIELECTRIC = clamp(0.08 * s.specular, 0.0, 0.08);
        vec3  viewDir  = -s.rayDir;
        float cosTheta = max(dot(viewDir, s.N), 0.0);

        // Schlick Fresnel, attenuated by roughness (rough surfaces scatter less)
        float fresnelBase   = F0_DIELECTRIC + (1.0 - F0_DIELECTRIC)
                              * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);

        if (rnd(seed) < fresnelBase) {
            // Specular lobe: GGX reflection (roughness=0 -> mirror)
            scatterMetal(s.P, s.N, s.rayDir, vec3(1.0), s.roughness, seed);
        } else {
            // Diffuse sub-layer: choose translucency / SSS / diffuse
            float pTrans = s.translucent;
            float pSSS   = (1.0 - pTrans) * s.subsurfaceAmount;
            float pDiff  = 1.0 - pTrans - pSSS;

            float r = rnd(seed);

            if (pTrans > 0.01 && r < pTrans) {
                scatterTranslucent(s.P, s.N, s.albedo, seed);
                payload.attenuation *= (1.0 / max(pTrans, 0.01));
            }
            else if (pSSS > 0.01 && r < pTrans + pSSS) {
                scatterSSS(s.P, s.N, s.albedo,
                           s.subsurfaceColor, s.subsurfaceAmount, s.subsurfaceScale,
                           s.subsurfaceRadius, s.subsurfaceAnisotropy, seed);
                payload.attenuation *= (1.0 / max(pSSS, 0.01));
            }
            else {
                scatterDiffuse(s.P, s.N, s.albedo, seed);
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
        if (rnd(seed) < metalWeight) {
            scatterMetal(s.P, s.N, s.rayDir, s.albedo, s.roughness, seed);
        } else {
            scatterDiffuse(s.P, s.N, s.albedo, seed);
        }
    }
}

#endif // BSDF_SCATTER_GLSL
