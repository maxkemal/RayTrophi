#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "procedural_detail.glsl"
#include "pbr_texture_policy.glsl"

layout(location = 0) in vec3 vWorldNormal;
layout(location = 1) flat in uint vMaterialID;
layout(location = 2) in vec2 vTexCoord;
layout(location = 3) in vec3 vWorldPos;

layout(location = 0) out vec4 outColor;

// Material buffer (matches VkGpuMaterial layout — 15 blocks x 16 bytes = 240 bytes)
struct GpuMaterial {
    // Block 1
    float albedo_r, albedo_g, albedo_b, opacity;
    // Block 2
    float emission_r, emission_g, emission_b, emission_strength;
    // Block 3
    float roughness, metallic, ior, transmission;
    // Block 4
    float subsurface_r, subsurface_g, subsurface_b, subsurface_amount;
    // Block 5
    float subsurface_radius_r, subsurface_radius_g, subsurface_radius_b, subsurface_scale;
    // Block 6
    float clearcoat, clearcoat_roughness, translucent, subsurface_anisotropy;
    // Block 7
    float anisotropic, sheen, sheen_tint;
    uint flags;
    // Block 8
    float fft_amplitude, fft_time_scale, micro_detail_strength, micro_detail_scale;
    // Block 9
    float foam_threshold, fft_ocean_size, fft_choppiness, fft_wind_speed;
    // Block 10
    float micro_anim_speed, micro_morph_speed, foam_noise_scale, fft_wind_direction;
    // Block 11
    float uv_scale_x, uv_scale_y, uv_offset_x, uv_offset_y;
    // Block 12
    float uv_rotation_degrees, uv_tiling_x, uv_tiling_y;
    uint uv_wrap_mode;
    // Block 13
    uint albedo_tex, normal_tex, roughness_tex, metallic_tex;
    // Block 14
    uint emission_tex, height_tex, opacity_tex, transmission_tex;
    // Block 15
    float subsurface_ior;
    uint _terrain_layer_idx;
    float normal_strength;
    float tile_break_strength;  // UV tile-break (independent from dirt/roughness)
};

layout(set = 0, binding = 0, std430) readonly buffer MaterialBuffer {
    GpuMaterial materials[];
};

// Texture array — same slots as RT pipeline binding 6
// Access guarded by albedo_tex/normal_tex > 0 checks.
layout(set = 0, binding = 1) uniform sampler2D textures[];

// Baked equirectangular environment maps for specular reflection lookup.
// [0] = studio  [1] = outdoor  (128×64 RGBA32F, uploaded at pipeline init)
layout(set = 0, binding = 2) uniform sampler2D envMaps[2];

// Terrain layer SSBO — mirrors RT pipeline binding 12.
// Layout must match VkTerrainLayerData (48 bytes, 16-byte aligned).
struct TerrainLayerData {
    uint  layer_mat_id[4];   // Material buffer indices for layers 0–3
    float layer_uv_scale[4]; // UV tiling per layer
    uint  splat_map_tex;     // RGBA splat map texture slot
    uint  layer_count;       // Active layers (1–4)
    uint  _pad[2];
};
layout(set = 0, binding = 3, std430) readonly buffer TerrainLayerBuffer {
    TerrainLayerData terrainLayers[];
};

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;
    vec4 lightDir0;   // key light
    vec4 lightDir1;   // fill light
    vec4 lightDir2;   // rim light
    uvec4 materialMeta; // x = material count, y = quality, z = lighting preset
} pc;

// ── Helpers ──

vec3 acesTonemap(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 linearToSRGB(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2));
}

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

// Apply material UV transform: scale → rotate → offset
vec2 applyUVTransform(vec2 uv, const GpuMaterial mat) {
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

    // Offset
    uv += vec2(mat.uv_offset_x, mat.uv_offset_y);
    return uv;
}

// Derivative-based TBN (tangent from screen-space partial derivatives)
// Returns a new perturbed normal using the normal map sample.
vec3 applyNormalMap(vec3 N, vec3 worldPos, vec2 uv, vec3 nmSample, float strength) {
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

    // Decode tangent-space normal
    vec3 tNormal = nmSample * 2.0 - 1.0;
    tNormal.xy  *= max(strength, 0.0);
    tNormal       = normalize(tNormal);

    return normalize(mat3(T, B, N) * tNormal);
}

void main() {
    vec3 N = normalize(vWorldNormal);
    uint qualityMode = pc.materialMeta.y;
    uint lightingPreset = pc.materialMeta.z;

    uint materialCount = max(pc.materialMeta.x, 1u);
    uint materialIndex = min(vMaterialID, materialCount - 1u);
    GpuMaterial mat = materials[materialIndex];

    vec2 uv = applyUVTransform(vTexCoord, mat);

    // ── Procedural tile-break (independent slider, applied before texture sampling) ──
    // Breaks visible UV tiling seams. Separate from dirt/roughness so albedo maps
    // that shouldn't be warped can leave this at 0.
    if (mat.tile_break_strength > 0.0 &&
        (mat.albedo_tex > 0u || mat.roughness_tex > 0u || mat.normal_tex > 0u)) {
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
        if (tl.splat_map_tex > 0u && tl.layer_count > 0u) {
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
                GpuMaterial lm = materials[tl.layer_mat_id[k]];
                vec2 layerUV = uv * tl.layer_uv_scale[k];
                // apply per-layer UV transform
                float lsx = (lm.uv_scale_x != 0.0) ? lm.uv_scale_x : 1.0;
                float lsy = (lm.uv_scale_y != 0.0) ? lm.uv_scale_y : 1.0;
                layerUV *= vec2(lsx, lsy);
                layerUV += vec2(lm.uv_offset_x, lm.uv_offset_y);

                vec3 lAlbedo = max(vec3(lm.albedo_r, lm.albedo_g, lm.albedo_b), vec3(0.0));
                if (lm.albedo_tex > 0u)
                    lAlbedo = texture(textures[nonuniformEXT(lm.albedo_tex)], layerUV).rgb;
                blendAlbedo += weights[k] * lAlbedo;

                float lRough = clamp(lm.roughness, 0.0, 1.0);
                if (lm.roughness_tex > 0u)
                    lRough = texture(textures[nonuniformEXT(lm.roughness_tex)], layerUV).g;
                blendRoughness += weights[k] * lRough;

                float lMetal = clamp(lm.metallic, 0.0, 1.0);
                if (lm.metallic_tex > 0u)
                    lMetal = texture(textures[nonuniformEXT(lm.metallic_tex)], layerUV).b;
                blendMetallic += weights[k] * lMetal;

                if (lm.normal_tex > 0u) {
                    vec3 ns = texture(textures[nonuniformEXT(lm.normal_tex)], layerUV).rgb;
                    ns = ns * 2.0 - 1.0;
                    ns.xy *= max(lm.normal_strength, 0.0);
                    blendNormal_ts += weights[k] * ns;
                    anyNormalTex = true;
                } else {
                    blendNormal_ts += weights[k] * vec3(0.0, 0.0, 1.0);
                }
            }

            // Apply blended normal first (derivative TBN — no surfaceTBN available in raster)
            if (anyNormalTex) {
                vec3 nts = normalize(blendNormal_ts);
                // Re-encode to [0,1] range so applyNormalMap can decode it back
                N = applyNormalMap(N, vWorldPos, uv, nts * 0.5 + 0.5, 1.0);
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
    if (mat.albedo_tex > 0u) {
        vec4 texAlbedo = texture(textures[nonuniformEXT(mat.albedo_tex)], uv);
        albedoTexel = texAlbedo;
        // Modulate with base color (matches Blender/PBR convention)
        albedo *= texAlbedo.rgb;
    }
    // Only fall back to neutral gray when no albedo source is bound.
    // Previously this fired whenever base color × texture went to ~0, which
    // turned black-paint strokes into gray in the raster material preview.
    if (mat.albedo_tex == 0u &&
        max(albedo.r, max(albedo.g, albedo.b)) < 0.001) {
        albedo = vec3(0.8);
    }

    // ── Normal map ──
    if (mat.normal_tex > 0u) {
        vec3 nmSample = texture(textures[nonuniformEXT(mat.normal_tex)], uv).rgb;
        float strength = (mat.normal_strength > 0.0) ? mat.normal_strength : 1.0;
        N = applyNormalMap(N, vWorldPos, uv, nmSample, strength);
    }

    // ── Roughness / Metallic ──
    float roughness = clamp(mat.roughness, 0.04, 1.0);
    float metallic  = clamp(mat.metallic,  0.0,  1.0);
    if (mat.roughness_tex > 0u) {
        roughness = samplePackedRoughness(
            texture(textures[nonuniformEXT(mat.roughness_tex)], uv), 0.04);
    }
    if (mat.metallic_tex > 0u) {
        metallic = samplePackedMetallic(
            texture(textures[nonuniformEXT(mat.metallic_tex)], uv));
    }

    float opacity = clamp(mat.opacity, 0.0, 1.0);
    if (mat.opacity_tex > 0u) {
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
    if (opacity == 0.0) {
        discard;
    }

    // ── Procedural detail: subtle color variation + dirt + roughness ──
    // micro_detail_strength drives all world-space effects without touching UVs.
    // tile_break_strength (above) is separate — warps UV only when needed.
    if (mat.micro_detail_strength > 0.0) {
        float sc  = max(mat.micro_detail_scale, 0.5);
        float str = mat.micro_detail_strength;

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
    vec3 emission = vec3(mat.emission_r, mat.emission_g, mat.emission_b) * mat.emission_strength;
    if (mat.emission_tex > 0u) {
        vec3 emitTex = texture(textures[nonuniformEXT(mat.emission_tex)], uv).rgb;
        emission += emitTex * mat.emission_strength;
    }

    // ── PBR-lite lighting (diffuse + Blinn-Phong specular) ──
    vec3 V = normalize(pc.cameraPos.xyz - vWorldPos);

    // F0: dielectric = 0.04, metallic = albedo
    vec3 F0           = mix(vec3(0.04), albedo, metallic);
    vec3 diffuseColor = albedo * (1.0 - metallic);

    // Specular exponent from roughness
    float specExp = max(2.0, (1.0 - roughness) * (1.0 - roughness) * 128.0);

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
    float sheenWeight    = clamp(mat.sheen, 0.0, 1.0);
    vec3  sheenColor     = mix(vec3(1.0), albedo, clamp(mat.sheen_tint, 0.0, 1.0)) * sheenWeight;
    float sssAmount      = clamp(mat.subsurface_amount, 0.0, 1.0);
    vec3  sssColor       = vec3(mat.subsurface_r, mat.subsurface_g, mat.subsurface_b);

    vec3 diffuseLit  = vec3(0.0);
    vec3 specularLit = vec3(0.0);
    vec3 sheenLit    = vec3(0.0);

    for (int i = 0; i < 3; ++i) {
        vec3  L        = lightDirs[i];
        vec3  radiance = lightColors[i] * lightIntensities[i];

        // ── Subsurface scattering: wrapped diffuse (Jensen 2001 approximation) ──
        // Shifts the NdotL threshold so light bleeds around the terminator.
        // sssAmount=0 → standard Lambertian; sssAmount=1 → full wrap.
        float wrapNdotL = (dot(N, L) + sssAmount) / (1.0 + sssAmount);
        float NdotL = max(wrapNdotL, 0.0);
        // SSS tints the sub-surface contribution toward the sssColor.
        vec3 diffuseAlbedo = mix(diffuseColor, diffuseColor * sssColor, sssAmount * step(0.001, sssAmount));
        diffuseLit += diffuseAlbedo * radiance * NdotL;

        if (qualityMode <= 1u) {
            vec3  H      = normalize(V + L);
            float NdotH  = max(dot(N, H), 0.0);
            float pureNdotL = max(dot(N, L), 0.0);
            float spec   = pow(NdotH, specExp) * pureNdotL;
            specularLit += F0 * radiance * spec;
            if (mat.clearcoat > 0.001) {
                float ccExp = max(2.0, (1.0 - mat.clearcoat_roughness) * (1.0 - mat.clearcoat_roughness) * 128.0);
                float ccSpec = pow(NdotH, ccExp) * pureNdotL;
                specularLit += vec3(0.04) * radiance * ccSpec * mat.clearcoat;
            }
        } else {
            float pureNdotL = max(dot(N, L), 0.0);
            specularLit += evaluateSpecularGGX(N, V, L, F0, roughness) * radiance * pureNdotL;
            if (mat.clearcoat > 0.001) {
                specularLit += evaluateSpecularGGX(
                    N, V, L, vec3(0.04), clamp(mat.clearcoat_roughness, 0.02, 1.0)) *
                    radiance * pureNdotL * mat.clearcoat;
            }
        }

        // ── Sheen lobe (fabric / velvet) ──
        if (sheenWeight > 0.001) {
            sheenLit += evaluateSheen(N, V, L, sheenColor, roughness) * radiance;
        }
    }

    float NdotV_main = max(dot(N, V), 0.0);

    vec3 ambient = vec3(0.0);
    vec3 envSpecular = vec3(0.0);
    if (lightingPreset == 0u) {
        vec3 ambientUp   = vec3(0.15, 0.17, 0.22);
        vec3 ambientDown = vec3(0.08, 0.06, 0.05);
        float ambientBlend = N.y * 0.5 + 0.5;
        // Energy conservation: metallic surfaces have no diffuse ambient
        vec3 kD = diffuseColor * (vec3(1.0) - fresnelSchlickRoughness(NdotV_main, F0, roughness));
        ambient = mix(ambientDown, ambientUp, ambientBlend) * kD;
    } else {
        vec3 envDiffuse    = samplePreviewEnvironment(N, lightingPreset);
        vec3 R             = reflect(-V, N);
        vec3 envReflection = sampleEnvSpecular(R, lightingPreset);

        // Roughness-corrected Fresnel for ambient (Lagarde 2012)
        vec3 fresnelAmb = fresnelSchlickRoughness(NdotV_main, F0, roughness);
        // kD: diffuse only gets energy not taken by specular, and none for metals
        vec3 kD = (vec3(1.0) - fresnelAmb) * (1.0 - metallic);
        ambient = envDiffuse * diffuseColor * kD * (lightingPreset == 2u ? 0.42 : 0.36);

        // roughness² gives physically correct falloff: smooth metals get strong env
        // reflection, rough metals stay dim. Multiplier compensates for the low-res
        // baked env map (128×64 blurs softbox peaks from 1.8 → ~0.3).
        envSpecular = envReflection * fresnelAmb * mix(3.0, 0.20, roughness * roughness);

        // Clearcoat env reflection
        if (mat.clearcoat > 0.001) {
            vec3 ccFresnel = fresnelSchlickRoughness(NdotV_main, vec3(0.04), mat.clearcoat_roughness);
            envSpecular += envReflection * ccFresnel
                         * mix(0.9, 0.0, mat.clearcoat_roughness)
                         * mat.clearcoat;
        }
    }

    float diffuseWeight  = (qualityMode <= 1u) ? 0.35 : 1.0;
    float specularWeight = (qualityMode <= 1u) ? 0.15 : 1.0;
    vec3 color = ambient
               + diffuseLit  * diffuseWeight
               + (specularLit + envSpecular) * specularWeight
               + sheenLit
               + emission;

    color = acesTonemap(color);
    color = linearToSRGB(color);

    outColor = vec4(color, opacity);
}
