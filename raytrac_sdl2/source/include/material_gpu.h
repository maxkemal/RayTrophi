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
    float foam_noise_scale;           // 4 bytes
    float foam_threshold;             // 4 bytes

    // Block 10: FFT Settings (16 bytes)
    float fft_ocean_size;             // 4 bytes
    float fft_choppiness;             // 4 bytes
    float padding1;                   // 4 bytes
    float padding2;                   // 4 bytes
};  
// Total: 160 bytes - well aligned (10 x 16-byte blocks)

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
        fabsf(a.fft_choppiness - b.fft_choppiness) < FLOAT_COMPARE_EPSILON;
}    

namespace std {  
    template <>  
    struct hash<GpuMaterial> {  
        size_t operator()(const GpuMaterial& m) const {  
            size_t h = 0;

            auto hash_combine = [&](size_t& seed, const auto& v) {  
                seed ^= std::hash<std::decay_t<decltype(v)>>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  
            };  

            // Albedo block
            hash_combine(h, m.albedo.x);  
            hash_combine(h, m.albedo.y);  
            hash_combine(h, m.albedo.z);  
            hash_combine(h, m.opacity);  

            // PBR block
            hash_combine(h, m.roughness);  
            hash_combine(h, m.metallic);  
            hash_combine(h, m.clearcoat);  
            hash_combine(h, m.transmission);  

            // Emission block
            hash_combine(h, m.emission.x);  
            hash_combine(h, m.emission.y);  
            hash_combine(h, m.emission.z);  
            hash_combine(h, m.ior);  

            // SSS block
            hash_combine(h, m.subsurface_color.x);
            hash_combine(h, m.subsurface_color.y);
            hash_combine(h, m.subsurface_color.z);
            hash_combine(h, m.subsurface);
            hash_combine(h, m.subsurface_radius.x);
            hash_combine(h, m.subsurface_radius.y);
            hash_combine(h, m.subsurface_radius.z);
            hash_combine(h, m.subsurface_scale);
            hash_combine(h, m.subsurface_anisotropy);

            // Clear Coat & Translucent
            hash_combine(h, m.clearcoat_roughness);
            hash_combine(h, m.translucent);

            // Water Details
            hash_combine(h, m.micro_detail_strength);
            hash_combine(h, m.micro_detail_scale);
            hash_combine(h, m.foam_noise_scale);
            hash_combine(h, m.foam_threshold);

            // FFT Settings
            hash_combine(h, m.fft_ocean_size);
            hash_combine(h, m.fft_choppiness);

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

            auto hash_combine = [&](size_t& seed, const auto& v) {  
                seed ^= std::hash<std::decay_t<decltype(v)>>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  
            };  

            hash_combine(h, x.albedoTexID);  
            hash_combine(h, x.normalTexID);  
            hash_combine(h, x.roughnessTexID);  
            hash_combine(h, x.metallicTexID);  
            hash_combine(h, x.opacityTexID);  
            hash_combine(h, x.emissionTexID);  
            hash_combine(h, x.materialNameHash);

            return h;  
        }  
    };  
}

