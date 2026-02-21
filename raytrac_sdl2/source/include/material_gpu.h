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
    int flags;                        // 4 bytes - bitfield flags (replaced padding)

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

    // Block 12: Standard Textures (Bindless Support)
    cudaTextureObject_t albedo_tex = 0;
    cudaTextureObject_t normal_tex = 0;
    cudaTextureObject_t roughness_tex = 0;
    cudaTextureObject_t metallic_tex = 0;
    cudaTextureObject_t emission_tex = 0;
    cudaTextureObject_t height_tex = 0; // Displacement
    cudaTextureObject_t opacity_tex = 0;
    cudaTextureObject_t transmission_tex = 0;
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
        // Water Details
        fabsf(a.micro_detail_strength - b.micro_detail_strength) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.micro_detail_scale - b.micro_detail_scale) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.foam_noise_scale - b.foam_noise_scale) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.foam_threshold - b.foam_threshold) < FLOAT_COMPARE_EPSILON &&
        // FFT Settings
        fabsf(a.fft_ocean_size - b.fft_ocean_size) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_choppiness - b.fft_choppiness) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_wind_speed - b.fft_wind_speed) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_wind_direction - b.fft_wind_direction) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_amplitude - b.fft_amplitude) < FLOAT_COMPARE_EPSILON &&
        fabsf(a.fft_time_scale - b.fft_time_scale) < FLOAT_COMPARE_EPSILON;
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

            // Clear Coat & Translucent
            hash_combine_f(h, m.clearcoat_roughness);
            hash_combine_f(h, m.translucent);

            // Water Details
            hash_combine_f(h, m.micro_detail_strength);
            hash_combine_f(h, m.micro_detail_scale);
            hash_combine_f(h, m.foam_noise_scale);
            hash_combine_f(h, m.foam_threshold);

            // FFT Settings
            hash_combine_f(h, m.fft_ocean_size);
            hash_combine_f(h, m.fft_choppiness);
            hash_combine_f(h, m.fft_wind_speed);
            hash_combine_f(h, m.fft_wind_direction);
            hash_combine_f(h, m.fft_amplitude);
            hash_combine_f(h, m.fft_time_scale);

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
            hash_combine_s(h, x.opacityTexID);  
            hash_combine_s(h, x.emissionTexID);  
            hash_combine_s(h, x.materialNameHash);

            return h;  
        }  
    };  
}


