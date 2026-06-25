/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          material_gpu.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once  
#include <cuda_runtime.h>  
#include <functional>
#include <cmath>
#include <type_traits>
#include <string>

// Float comparison epsilon
constexpr float FLOAT_COMPARE_EPSILON = 1e-4f;
static constexpr int GPU_MAT_FLAG_WATER = (1 << 17);
// Thin-shell BUBBLE material (champagne / soda / soap-foam close-up). The BSDF
// treats the surface as a thin dielectric shell: Fresnel rim reflection vs a
// STRAIGHT pass-through (no net refraction bending), so it reads as a bright-
// rimmed transparent bubble regardless of the surrounding medium (no nested-
// dielectric needed). bubble_ior drives the rim Fresnel; bubble_film is reserved
// for the thin-film interference (soap iridescence) phase.
static constexpr int GPU_MAT_FLAG_BUBBLE = (1 << 18);
static constexpr int GPU_MAT_FLAG_MARBLE_VOLUME = (1 << 19); // glass marble full-volume medium march

/**
 * @brief Optimized GPU Material structure
 * 
 * Memory optimization: Removed unnecessary padding by reorganizing fields.
 * Old size: ~92 bytes (with padding)
 * New size: ~64 bytes (30% reduction)
 * 
 * Layout is designed for 16-byte alignment and optimal GPU cache access.
 */
struct alignas(16) GpuMaterial {  
    // Block 1: Albedo + opacity (16 bytes)
    float3 albedo;                    // 12 bytes - base color
    float opacity;                    // 4 bytes  - alpha/opacity
    
    // Block 2: PBR core properties (16 bytes)
    float roughness;                  // 4 bytes
    float metallic;                   // 4 bytes
    float clearcoat;                  // 4 bytes
    float transmission;               // 4 bytes
    
    // Block 3: Emission + IOR (16 bytes)
    float3 emission;                  // 12 bytes - emission color
    float ior;                        // 4 bytes  - index of refraction
    
    // Block 4: Subsurface color + amount (16 bytes)
    float3 subsurface_color;          // 12 bytes - SSS tint color
    float subsurface;                 // 4 bytes  - SSS amount (0-1)
    
    // Block 5: Subsurface Radius RGB + scale (16 bytes)
    float3 subsurface_radius;         // 12 bytes - Per-channel scatter distance (e.g., skin: R=1.0, G=0.2, B=0.1)
    float subsurface_scale;           // 4 bytes  - Global radius multiplier
    
    // Block 6: Clear Coat + Translucent (16 bytes)
    float clearcoat_roughness;        // 4 bytes  - Clear coat layer roughness
    float translucent;                // 4 bytes  - Thin surface light pass-through
    float subsurface_anisotropy;      // 4 bytes  - SSS scatter direction bias (-1 to 1)
    float subsurface_ior;             // 4 bytes  - SSS internal IOR (typically 1.3-1.5)
    
    // Block 7: Additional properties (16 bytes)   
    float anisotropic;                // 4 bytes - Surface anisotropy
    float sheen;                      // 4 bytes - Sheen amount  
    float sheen_tint;                 // 4 bytes - Sheen color tint
    float normal_strength;            // 4 bytes - tangent-space normal XY multiplier

    // Subsurface control flags + tile-break (fills the implicit 4-byte gap before 8-byte-aligned fft_height_tex)
    int flags;                        // 4 bytes - bitfield flags
    int sss_use_random_walk = 1;      // 4 bytes
    int sss_max_steps = 6;            // 4 bytes - bounded random-walk depth
    float tile_break_strength = 0.0f; // 4 bytes - UV tile-break strength (0=off, 0.1–0.3 typical)

    // Block 8: Water FFT Textures (16 bytes)
    cudaTextureObject_t fft_height_tex; // 8 bytes 
    cudaTextureObject_t fft_normal_tex; // 8 bytes

    // Block 9: Advanced Water Details (16 bytes)
    float micro_detail_strength;      // 4 bytes
    float micro_detail_scale;         // 4 bytes
    float micro_anim_speed;           // 4 bytes - Animation speed multiplier
    float micro_morph_speed;          // 4 bytes - Shape morphing speed

    // Block 10: Water Foam (16 bytes)
    float foam_noise_scale;           // 4 bytes
    float foam_threshold;             // 4 bytes
    float fft_ocean_size;             // 4 bytes
    float fft_choppiness;             // 4 bytes
    
    // Block 11: FFT Wind Settings (16 bytes)
    float fft_wind_speed;             // 4 bytes
    float fft_wind_direction;         // 4 bytes
    float fft_amplitude;              // 4 bytes
    float fft_time_scale;             // 4 bytes

    // Block 12: UV transform core (16 bytes)
    float uv_scale_x = 1.0f;
    float uv_scale_y = 1.0f;
    float uv_offset_x = 0.0f;
    float uv_offset_y = 0.0f;

    // Block 13: UV transform extra (16 bytes)
    float uv_rotation_degrees = 0.0f;
    float uv_tiling_x = 1.0f;
    float uv_tiling_y = 1.0f;
    int uv_wrap_mode = 0;

    // Block 14: Standard Textures (Bindless Support)
    cudaTextureObject_t albedo_tex = 0;
    cudaTextureObject_t normal_tex = 0;
    cudaTextureObject_t roughness_tex = 0;
    cudaTextureObject_t metallic_tex = 0;
    cudaTextureObject_t emission_tex = 0;
    cudaTextureObject_t height_tex = 0; // Displacement
    cudaTextureObject_t opacity_tex = 0;
    cudaTextureObject_t transmission_tex = 0;
    cudaTextureObject_t specular_tex = 0;

    // Scalar specular amount. Blender-style dielectric F0 = 0.08 * specular.
    float specular = 0.5f;
    // Thin-shell BUBBLE params (active when flags & GPU_MAT_FLAG_BUBBLE). Repurposed
    // from the trailing specular padding so the struct size/layout is UNCHANGED
    // (both backends share this struct byte-for-byte).
    float bubble_ior  = 1.33f;  // rim Fresnel IOR (air/liquid interface)
    float bubble_film = 0.0f;   // thin-film interference strength (Faz 2; 0 = off)
    // Transmission interior absorption density (thick resin / glass-marble depth).
    // 0 = legacy constant-thickness glass tint; >0 = real Beer-Lambert over the
    // actual in-medium ray distance (deep centre, clear edges). Repurposed pad.
    float transmission_density = 0.0f;
    // Resin absorption colour (the depth tint), separate from albedo. White = clear.
    float3 resin_color = {1.0f, 1.0f, 1.0f};
    // Resin coat gloss — reflection-lobe roughness of the resin layer, independent of base.
    float resin_roughness = 0.1f;
    // Resin internal inclusions (procedural dust/dirt march through the resin thickness).
    float resin_inclusion = 0.0f;        // dust cloudiness amount (0 = off)
    float resin_dirt = 0.0f;             // opaque dirt-speck amount (early-return)
    float resin_inclusion_scale = 8.0f;  // procedural feature size
    float3 resin_dirt_color = {0.18f, 0.14f, 0.10f};
    // Iridescent clearcoat: thin-film tint on the clearcoat lobe (oil-slick / beetle-shell /
    // candy paint). Repurposed from the trailing implicit padding so the struct size/layout
    // is UNCHANGED (matches GLSL _resin2_pad0/_resin2_pad1, shared byte-for-byte).
    // 0 = plain white clearcoat (no behaviour change). Reuses the bubble thin-film formula.
    float clearcoat_iridescence    = 0.0f;  // tint strength (0 = off)
    float clearcoat_film_thickness = 0.55f; // hue cycle / film thickness (OPD scale)
};

/**
 * @brief GPU-friendly hair material
 */
struct alignas(16) GpuHairMaterial {
    // Block 1: Color & Appearance (16 bytes)
    float3 sigma_a;               // Absorption coefficient (12 bytes)
    int colorMode;                // Color mode (4 bytes)

    // Block 2: Physically Based Model (16 bytes)
    float3 color;                 // Direct color (12 bytes)
    float roughness;              // Longitudinal roughness (4 bytes)

    // Block 3: Physical Properties (16 bytes)
    float melanin;                // Melanin amount (4 bytes)
    float melaninRedness;         // Melanin redness (4 bytes)
    float ior;                    // Index of refraction (4 bytes)
    float cuticleAngle;           // In radians (4 bytes)

    // Block 4: Styling & Tint (16 bytes)
    float3 tintColor;             // Tint color (12 bytes)
    float tint;                   // Tint strength (4 bytes)

    // Block 5: Azimuthal & Random (16 bytes)
    float radialRoughness;        // Azimuthal roughness (4 bytes)
    float randomHue;              // Variation (4 bytes)
    float randomValue;            // Variation (4 bytes)
    float emissionStrength;       // (4 bytes)

    // Block 6: Emission & Variances (16 bytes)
    float3 emission;              // (12 bytes)
    float v_R;                    // Variance for R lobe (4 bytes)

    // Block 7: More Variances & Misc (16 bytes)
    float v_TT;                   // Variance for TT lobe (4 bytes)
    float v_TRT;                  // Variance for TRT lobe (4 bytes)
    float s_R;                    // Logistic scale for R lobe (4 bytes)
    float s_TT;                   // Logistic scale for TT lobe (4 bytes)

    // Block 8: More Azimuthal & Textures (16 bytes)
    float s_TRT;                  // Logistic scale for TRT lobe (4 bytes)
    float s_MS;                   // Logistic scale for MS lobe (4 bytes)
    cudaTextureObject_t albedo_tex;    // 8 bytes
    
    // Block 9: Remaining Textures (16 bytes)
    cudaTextureObject_t roughness_tex; // 8 bytes
    float coat;                   // Coat strength for fur (4 bytes)
    float specularTint;           // Tint primary highlight by hair color (4 bytes)

    // Block 10: Coat & Gradient (16 bytes)
    float3 coatTint;              // Coat reflection tint (12 bytes)
    float diffuseSoftness;        // MS weight: 0=hard specular, 1=soft diffuse (4 bytes)

    // Block 11: Root-Tip Gradient (16 bytes)
    float3 tipSigma;              // Absorption at tip (12 bytes)
    float rootTipBalance;         // 0=root color, 1=tip color at tip (4 bytes)

    // Block 12: Flags & Padding (16 bytes)
    int enableRootTipGradient;    // 0 or 1 (4 bytes)
    float pad1;                   // (4 bytes)
    float pad2;                   // (4 bytes)
    float pad3;                   // (4 bytes)
};
// Total: 192 bytes - well aligned (12 x 16-byte blocks)

inline bool float3_equal(float3 a, float3 b, float epsilon = FLOAT_COMPARE_EPSILON) {  
    return fabsf(a.x - b.x) < epsilon && 
           fabsf(a.y - b.y) < epsilon && 
           fabsf(a.z - b.z) < epsilon;  
}  

inline bool operator==(const GpuMaterial& a, const GpuMaterial& b) {  
    return float3_equal(a.albedo, b.albedo) &&
        fabsf(a.opacity - b.opacity) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.roughness - b.roughness) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.metallic - b.metallic) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.specular - b.specular) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.clearcoat - b.clearcoat) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.transmission - b.transmission) < FLOAT_COMPARE_EPSILON &&
        float3_equal(a.emission, b.emission) &&
        fabsf(a.ior - b.ior) < FLOAT_COMPARE_EPSILON &&
        // SSS fields
        float3_equal(a.subsurface_color, b.subsurface_color) &&
        fabsf(a.subsurface - b.subsurface) < FLOAT_COMPARE_EPSILON &&
        float3_equal(a.subsurface_radius, b.subsurface_radius) &&
        fabsf(a.subsurface_scale - b.subsurface_scale) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.subsurface_anisotropy - b.subsurface_anisotropy) < FLOAT_COMPARE_EPSILON &&
        // Clear Coat & Translucent
        fabsf(a.clearcoat_roughness - b.clearcoat_roughness) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.translucent - b.translucent) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.normal_strength - b.normal_strength) < FLOAT_COMPARE_EPSILON &&
        // Procedural Detail
        fabsf(a.micro_detail_strength - b.micro_detail_strength) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.micro_detail_scale - b.micro_detail_scale) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.tile_break_strength - b.tile_break_strength) < FLOAT_COMPARE_EPSILON &&
        // Water Details
        fabsf(a.foam_noise_scale - b.foam_noise_scale) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.foam_threshold - b.foam_threshold) < FLOAT_COMPARE_EPSILON &&
        // FFT Settings
        fabsf(a.fft_ocean_size - b.fft_ocean_size) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_choppiness - b.fft_choppiness) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_wind_speed - b.fft_wind_speed) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_wind_direction - b.fft_wind_direction) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_amplitude - b.fft_amplitude) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_time_scale - b.fft_time_scale) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.uv_scale_x - b.uv_scale_x) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.uv_scale_y - b.uv_scale_y) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.uv_offset_x - b.uv_offset_x) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.uv_offset_y - b.uv_offset_y) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.uv_rotation_degrees - b.uv_rotation_degrees) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.uv_tiling_x - b.uv_tiling_x) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.uv_tiling_y - b.uv_tiling_y) < FLOAT_COMPARE_EPSILON &&
        a.uv_wrap_mode == b.uv_wrap_mode &&
        // Bubble (thin-shell) params + material flags
        a.flags == b.flags &&
        fabsf(a.bubble_ior - b.bubble_ior) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.bubble_film - b.bubble_film) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.transmission_density - b.transmission_density) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_color.x - b.resin_color.x) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_color.y - b.resin_color.y) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_color.z - b.resin_color.z) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_roughness - b.resin_roughness) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_inclusion - b.resin_inclusion) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_dirt - b.resin_dirt) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_inclusion_scale - b.resin_inclusion_scale) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_dirt_color.x - b.resin_dirt_color.x) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_dirt_color.y - b.resin_dirt_color.y) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.resin_dirt_color.z - b.resin_dirt_color.z) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.clearcoat_iridescence - b.clearcoat_iridescence) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.clearcoat_film_thickness - b.clearcoat_film_thickness) < FLOAT_COMPARE_EPSILON;
}

namespace std {  
    template <>  
    struct hash<GpuMaterial> {  
        size_t operator()(const GpuMaterial& m) const {  
            size_t h = 0;

            auto hash_combine_f = [&](size_t& seed, float v) {  
                seed ^= std::hash<float>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  
            };  

            // Albedo block
            hash_combine_f(h, m.albedo.x);  
            hash_combine_f(h, m.albedo.y);  
            hash_combine_f(h, m.albedo.z);  
            hash_combine_f(h, m.opacity);  

            // PBR block
            hash_combine_f(h, m.roughness);  
            hash_combine_f(h, m.metallic);  
            hash_combine_f(h, m.specular);
            hash_combine_f(h, m.clearcoat);  
            hash_combine_f(h, m.transmission);  

            // Emission block
            hash_combine_f(h, m.emission.x);  
            hash_combine_f(h, m.emission.y);  
            hash_combine_f(h, m.emission.z);  
            hash_combine_f(h, m.ior);  

            // SSS block
            hash_combine_f(h, m.subsurface_color.x);
            hash_combine_f(h, m.subsurface_color.y);
            hash_combine_f(h, m.subsurface_color.z);
            hash_combine_f(h, m.subsurface);
            hash_combine_f(h, m.subsurface_radius.x);
            hash_combine_f(h, m.subsurface_radius.y);
            hash_combine_f(h, m.subsurface_radius.z);
            hash_combine_f(h, m.subsurface_scale);
            hash_combine_f(h, m.subsurface_anisotropy);
            // SSS control fields
            hash_combine_f(h, (float)m.sss_use_random_walk);
            hash_combine_f(h, (float)m.sss_max_steps);

            // Clear Coat & Translucent
            hash_combine_f(h, m.clearcoat_roughness);
            hash_combine_f(h, m.translucent);
            hash_combine_f(h, m.normal_strength);

            // Procedural Detail
            hash_combine_f(h, m.micro_detail_strength);
            hash_combine_f(h, m.micro_detail_scale);
            hash_combine_f(h, m.tile_break_strength);
            // Water Details
            hash_combine_f(h, m.foam_noise_scale);
            hash_combine_f(h, m.foam_threshold);

            // FFT Settings
            hash_combine_f(h, m.fft_ocean_size);
            hash_combine_f(h, m.fft_choppiness);
            hash_combine_f(h, m.fft_wind_speed);
            hash_combine_f(h, m.fft_wind_direction);
            hash_combine_f(h, m.fft_amplitude);
            hash_combine_f(h, m.fft_time_scale);
            hash_combine_f(h, m.uv_scale_x);
            hash_combine_f(h, m.uv_scale_y);
            hash_combine_f(h, m.uv_offset_x);
            hash_combine_f(h, m.uv_offset_y);
            hash_combine_f(h, m.uv_rotation_degrees);
            hash_combine_f(h, m.uv_tiling_x);
            hash_combine_f(h, m.uv_tiling_y);
            hash_combine_f(h, (float)m.uv_wrap_mode);

                // include sss fields at end for hash stability
                hash_combine_f(h, (float)m.sss_use_random_walk);
                hash_combine_f(h, (float)m.sss_max_steps);

            // Bubble (thin-shell) params + flags
            hash_combine_f(h, (float)m.flags);
            hash_combine_f(h, m.bubble_ior);
            hash_combine_f(h, m.bubble_film);
            hash_combine_f(h, m.transmission_density);
            hash_combine_f(h, m.resin_color.x);
            hash_combine_f(h, m.resin_color.y);
            hash_combine_f(h, m.resin_color.z);
            hash_combine_f(h, m.resin_roughness);
            hash_combine_f(h, m.resin_inclusion);
            hash_combine_f(h, m.resin_dirt);
            hash_combine_f(h, m.resin_inclusion_scale);
            hash_combine_f(h, m.resin_dirt_color.x);
            hash_combine_f(h, m.resin_dirt_color.y);
            hash_combine_f(h, m.resin_dirt_color.z);
            hash_combine_f(h, m.clearcoat_iridescence);
            hash_combine_f(h, m.clearcoat_film_thickness);

            return h;
        }  
    };  
}

struct GpuMaterialWithTextures {  
    GpuMaterial material;  
    size_t albedoTexID = 0;  
    size_t normalTexID = 0;  
    size_t roughnessTexID = 0;  
    size_t metallicTexID = 0;  
    size_t specularTexID = 0;
    size_t opacityTexID = 0;  
    size_t emissionTexID = 0;  
    size_t subsurfaceTexID = 0;  
    
    // CRITICAL: Unique material identifier to prevent hash collisions between imports
    // Without this, materials from different imports with same texture pointers and 
    // similar scalar values (roughness, metallic) would incorrectly match.
    // This is especially common with PolyHaven ORM textures where metallic and roughness
    // share the same texture file (different channels).
    size_t materialNameHash = 0;

    bool operator==(const GpuMaterialWithTextures& other) const {  
        return material == other.material &&
            albedoTexID == other.albedoTexID &&
            normalTexID == other.normalTexID &&
            roughnessTexID == other.roughnessTexID &&
            metallicTexID == other.metallicTexID &&
            specularTexID == other.specularTexID &&
            opacityTexID == other.opacityTexID &&
            emissionTexID == other.emissionTexID &&
            materialNameHash == other.materialNameHash;
    }  
};  

namespace std {  
    template <>  
    struct hash<GpuMaterialWithTextures> {  
        size_t operator()(const GpuMaterialWithTextures& x) const {  
            size_t h = std::hash<GpuMaterial>{}(x.material);  

            auto hash_combine_s = [&](size_t& seed, size_t v) {  
                seed ^= std::hash<size_t>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  
            };  

            hash_combine_s(h, x.albedoTexID);  
            hash_combine_s(h, x.normalTexID);  
            hash_combine_s(h, x.roughnessTexID);  
            hash_combine_s(h, x.metallicTexID);  
            hash_combine_s(h, x.specularTexID);
            hash_combine_s(h, x.opacityTexID);  
            hash_combine_s(h, x.emissionTexID);  
            hash_combine_s(h, x.materialNameHash);

            return h;  
        }  
    };  
}


