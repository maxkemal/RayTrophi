#pragma once
#include <cuda_runtime.h>
#include <functional> // hash için

struct alignas(16) GpuMaterial {
    float3 albedo;
    float padding0;
    float opacity;
    float roughness;
    float metallic;
    float clearcoat;
    float2 padding1;
    float3 emission;
    float padding2;
    float transmission;
    float ior;
    float2 padding3;
    float artistic_albedo_response;
    float subsurface=0.0f;                // 0.0 - 1.0 arası
    float3 subsurface_color;         // genelde albedoya yakın ama daha yumuşak

};

// Eşitlik operatörü - tüm önemli alanları karşılaştırır
inline bool float3_esit(float3 a, float3 b, float epsilon = 1e-6f) {
    return abs(a.x - b.x) < epsilon && abs(a.y - b.y) < epsilon && abs(a.z - b.z) < epsilon;
}

inline bool operator==(const GpuMaterial& a, const GpuMaterial& b) {
    return float3_esit(a.albedo, b.albedo) &&
        abs(a.roughness - b.roughness) < 1e-6f &&
        abs(a.metallic - b.metallic) < 1e-6f &&
        float3_esit(a.emission, b.emission) &&
        abs(a.transmission - b.transmission) < 1e-6f &&
        abs(a.ior - b.ior) < 1e-6f &&
        abs(a.opacity - b.opacity) < 1e-6f &&
        abs(a.artistic_albedo_response - b.artistic_albedo_response) < 1e-6f;
}

// Hash fonksiyonu (unordered_map için) - geliştirilmiş sürüm
namespace std {
    template <>
    struct hash<GpuMaterial> {
        size_t operator()(const GpuMaterial& m) const {
            // Daha ayırt edici hash için daha iyi bir başlangıç değeri kullanıyoruz
            const size_t prime = 31;
            size_t h = 17; // Rastgele bir asal sayı ile başla

            // Her alan için hash'i güncelle
            auto combine = [&](auto val) {
                h = h * prime + std::hash<decltype(val)>{}(val);
                };

            combine(m.albedo.x); combine(m.albedo.y); combine(m.albedo.z);
            combine(m.roughness);
            combine(m.metallic);
            combine(m.emission.x); combine(m.emission.y); combine(m.emission.z);
            combine(m.transmission);
            combine(m.ior);
            combine(m.opacity);
            combine(m.artistic_albedo_response); // Bu alanı da dahil et

            return h;
        }
    };
}
// CPU tarafı karşılaştırma yapısı
struct GpuMaterialWithTextures {
    GpuMaterial material;
    size_t albedoTexID = 0;
    size_t normalTexID = 0;
    size_t roughnessTexID = 0;
    size_t metallicTexID = 0;
    size_t opacityTexID = 0;
    size_t emissionTexID = 0;

    bool operator==(const GpuMaterialWithTextures& other) const {
        return material == other.material &&
            albedoTexID == other.albedoTexID &&
            normalTexID == other.normalTexID &&
            roughnessTexID == other.roughnessTexID &&
            metallicTexID == other.metallicTexID &&
            opacityTexID == other.opacityTexID &&
            emissionTexID == other.emissionTexID;
    }
};

namespace std {
    template <>
    struct hash<GpuMaterialWithTextures> {
        size_t operator()(const GpuMaterialWithTextures& x) const {
            size_t h = std::hash<GpuMaterial>{}(x.material);
            h ^= std::hash<size_t>{}(x.albedoTexID) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<size_t>{}(x.normalTexID);
            h ^= std::hash<size_t>{}(x.roughnessTexID);
            h ^= std::hash<size_t>{}(x.metallicTexID);
            h ^= std::hash<size_t>{}(x.opacityTexID);
            h ^= std::hash<size_t>{}(x.emissionTexID);
            return h;
        }
    };
}
