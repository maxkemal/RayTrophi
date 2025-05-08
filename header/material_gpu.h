#pragma once
#include <cuda_runtime.h>
#include <functional> // hash için

struct __align__(16) GpuMaterial {
    float3 albedo;
    float padding0;
	float opacity;
    float roughness;
    float metallic;
    float2 padding1;
    float3 emission;
    float padding2;
    float transmission;
    float ior;
    float2 padding3;
    float artistic_albedo_response ;

};


// Eþitlik karþýlaþtýrmasý
inline bool operator==(const GpuMaterial& a, const GpuMaterial& b) {
    return a.albedo.x == b.albedo.x && a.albedo.y == b.albedo.y && a.albedo.z == b.albedo.z &&
        a.roughness == b.roughness &&
        a.metallic == b.metallic &&
        a.emission.x == b.emission.x && a.emission.y == b.emission.y && a.emission.z == b.emission.z &&
        a.transmission == b.transmission &&
        a.ior == b.ior;
}

// Hash fonksiyonu (unordered_map için)
namespace std {
    template <>
    struct hash<GpuMaterial> {
        size_t operator()(const GpuMaterial& m) const {
            size_t h1 = hash<float>{}(m.albedo.x) ^ hash<float>{}(m.albedo.y) ^ hash<float>{}(m.albedo.z);
            size_t h2 = hash<float>{}(m.roughness);
            size_t h3 = hash<float>{}(m.metallic);
            size_t h4 = hash<float>{}(m.emission.x) ^ hash<float>{}(m.emission.y) ^ hash<float>{}(m.emission.z);
            size_t h5 = hash<float>{}(m.transmission);
            size_t h6 = hash<float>{}(m.ior);

            return (((((h1 ^ (h2 << 1)) ^ (h3 << 2)) ^ (h4 << 3)) ^ (h5 << 4)) ^ (h6 << 5));
        }
    };
}
