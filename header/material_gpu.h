#pragma once  
#include <cuda_runtime.h>  
#include <functional> // std::hash için  
#include <cmath>      // fabsf için  
#include <type_traits> // std::remove_reference_t için  

// float değerlerini karşılaştırmak için global bir epsilon  
constexpr float FLOAT_COMPARE_EPSILON = 1e-4f;  

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
   float subsurface = 0.0f;  
   float3 subsurface_color;  
};  

inline bool float3_esit(float3 a, float3 b, float epsilon = FLOAT_COMPARE_EPSILON) {  
   return fabsf(a.x - b.x) < epsilon && fabsf(a.y - b.y) < epsilon && fabsf(a.z - b.z) < epsilon;  
}  

inline bool operator==(const GpuMaterial& a, const GpuMaterial& b) {  
   return float3_esit(a.albedo, b.albedo) &&  
       fabsf(a.roughness - b.roughness) < FLOAT_COMPARE_EPSILON &&  
       fabsf(a.metallic - b.metallic) < FLOAT_COMPARE_EPSILON &&  
       float3_esit(a.emission, b.emission) &&  
       fabsf(a.transmission - b.transmission) < FLOAT_COMPARE_EPSILON &&  
       fabsf(a.ior - b.ior) < FLOAT_COMPARE_EPSILON &&  
       fabsf(a.opacity - b.opacity) < FLOAT_COMPARE_EPSILON &&  
       fabsf(a.artistic_albedo_response - b.artistic_albedo_response) < FLOAT_COMPARE_EPSILON &&  
       fabsf(a.subsurface - b.subsurface) < FLOAT_COMPARE_EPSILON &&  
       float3_esit(a.subsurface_color, b.subsurface_color);  
}  

namespace std {  
   template <>  
   struct hash<GpuMaterial> {  
       size_t operator()(const GpuMaterial& m) const {  
           size_t h = 0; // Başlangıç hash değeri  

           // Boost hash_combine prensibini uygula  
           // Her eleman için hash al ve mevcut h ile karıştır  
           // std::remove_reference_t ile referans tipini kaldırıyoruz  
           auto hash_combine = [&](size_t& seed, const auto& v) {  
               seed ^= std::hash<std::decay_t<decltype(v)>>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  
           };  

           // float3'leri eleman eleman hash'le  
           hash_combine(h, m.albedo.x);  
           hash_combine(h, m.albedo.y);  
           hash_combine(h, m.albedo.z);  

           hash_combine(h, m.opacity);  
           hash_combine(h, m.roughness);  
           hash_combine(h, m.metallic);  
           hash_combine(h, m.clearcoat);  
           hash_combine(h, m.artistic_albedo_response);  
           hash_combine(h, m.subsurface);  
           hash_combine(h, m.transmission);  
           hash_combine(h, m.ior);  

           // emission'ı eleman eleman hash'le  
           hash_combine(h, m.emission.x);  
           hash_combine(h, m.emission.y);  
           hash_combine(h, m.emission.z);  

           // subsurface_color'ı eleman eleman hash'le  
           hash_combine(h, m.subsurface_color.x);  
           hash_combine(h, m.subsurface_color.y);  
           hash_combine(h, m.subsurface_color.z);  

           return h;  
       }  
   };  
} // namespace std  

struct GpuMaterialWithTextures {  
   GpuMaterial material;  
   size_t albedoTexID = 0;  
   size_t normalTexID = 0;  
   size_t roughnessTexID = 0;  
   size_t metallicTexID = 0;  
   size_t opacityTexID = 0;  
   size_t emissionTexID = 0;  
   size_t subsurfaceTexID = 0;  

   bool operator==(const GpuMaterialWithTextures& other) const {  
       return material == other.material &&  
           albedoTexID == other.albedoTexID &&  
           normalTexID == other.normalTexID &&  
           roughnessTexID == other.roughnessTexID &&  
           metallicTexID == other.metallicTexID &&  
           opacityTexID == other.opacityTexID &&  
           emissionTexID == other.emissionTexID &&  
           subsurfaceTexID == other.subsurfaceTexID;  
   }  
};  

namespace std {  
   template <>  
   struct hash<GpuMaterialWithTextures> {  
       size_t operator()(const GpuMaterialWithTextures& x) const {  
           // Materyal hash'ini başlangıç olarak al  
           size_t h = std::hash<GpuMaterial>{}(x.material);  

           // Boost hash_combine prensibini kullanarak her bir texture ID'sini karıştır  
           // std::remove_reference_t ile referans tipini kaldırıyoruz  
           auto hash_combine = [&](size_t& seed, const auto& v) {  
               seed ^= std::hash<std::decay_t<decltype(v)>>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  
           };  

           hash_combine(h, x.albedoTexID);  
           hash_combine(h, x.normalTexID);  
           hash_combine(h, x.roughnessTexID);  
           hash_combine(h, x.metallicTexID);  
           hash_combine(h, x.opacityTexID);  
           hash_combine(h, x.emissionTexID);  
           hash_combine(h, x.subsurfaceTexID);  

           return h;  
       }  
   };  
} // namespace std