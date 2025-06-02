#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include "material_gpu.h"
#include "params.h"
#include "payload.h"
#include "vec3_utils.cuh"
#include "ray_color.cuh"
#include "ray.h"              // ← varsa Ray struct burada
#include "gpucamera.cuh"         // ← generate_camera_ray burada olabilir
#include "random_utils.cuh" // random_float() fonksiyonu burada
#include "trace_ray.cuh"   // trace_ray() fonksiyonu burada

// RAYGEN
extern "C" __global__ void __raygen__rg() {
    int index = optixGetLaunchIndex().x;
    int i = optixLaunchParams.launch_coords_x[index];
    int j = optixLaunchParams.launch_coords_y[index];
    int pixel_index = j * optixLaunchParams.image_width + i;

    // Daha iyi seed stratejisi - piksel koordinatları, frame ve global seed karıştırılır
    unsigned int seed = optixLaunchParams.frame_number * 719393 + pixel_index * 13731 + 1337;
    curandState rng;
    curand_init(seed, 0, 0, &rng);

    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);
    float3 mean = make_float3(0.0f, 0.0f, 0.0f);
    float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
    int sample_count = 0;
    const bool use_adaptive = optixLaunchParams.use_adaptive_sampling;

    // Adaptif parametreler
    const int min_samples = optixLaunchParams.min_samples; // Dışarıdan ayarlanabilir
    const int max_samples = optixLaunchParams.samples_per_pixel;
    const float base_variance_threshold = optixLaunchParams.variance_threshold; // Dışarıdan ayarlanabilir

    // Komşu piksellerin durumunu kontrol et - uzamsal tutarlılık için
    bool has_high_variance_neighbor = false;
    if (optixLaunchParams.frame_number > 1 && i > 0 && i < optixLaunchParams.image_width - 1 &&
        j > 0 && j < optixLaunchParams.image_height - 1) {
        // 3x3 komşuluk kontrolü - çok fazla hesaplama yapmamak için sadece bazı pikselleri kontrol ediyoruz
        float neighbor_variance[4];
        neighbor_variance[0] = optixLaunchParams.variance_buffer[(j - 1) * optixLaunchParams.image_width + i];
        neighbor_variance[1] = optixLaunchParams.variance_buffer[(j + 1) * optixLaunchParams.image_width + i];
        neighbor_variance[2] = optixLaunchParams.variance_buffer[j * optixLaunchParams.image_width + (i - 1)];
        neighbor_variance[3] = optixLaunchParams.variance_buffer[j * optixLaunchParams.image_width + (i + 1)];

        for (int n = 0; n < 4; n++) {
            if (neighbor_variance[n] > base_variance_threshold * 2.0f) {
                has_high_variance_neighbor = true;
                break;
            }
        }
    }

    // Ana örnekleme döngüsü
    for (int s = 0; s < max_samples; ++s) {
        // Stratified sampling - piksel içinde daha düzgün dağılmış örnekler
        float u = (i + (s + random_float(&rng)) / max_samples) / float(optixLaunchParams.image_width);
        float v = (j + (s + random_float(&rng)) / max_samples) / float(optixLaunchParams.image_height);

        // Lens apertüründe stratified sampling
        Ray ray = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);

        // Importance sampling kullanarak renk örneklemesi
        float3 sample = ray_color(ray, &rng);
        sample_count++;

        // Welford online mean ve variance güncelleme
        float3 delta = sample - mean;
        mean += delta / sample_count;
        float3 delta2 = sample - mean;
        m2 += delta * delta2;
        color_sum += sample;

        // Minimum örneklemeden sonra erken çıkış kontrolü
        if (use_adaptive && s >= min_samples - 1) {
            // Varyans hesaplama
            float3 var = m2 / (sample_count - 1);
            float luminance_var = 0.2126f * var.x + 0.7152f * var.y + 0.0722f * var.z;
            float luminance_mean = 0.2126f * mean.x + 0.7152f * mean.y + 0.0722f * mean.z;

            float rel_variance = luminance_var / (fmaxf(luminance_mean * luminance_mean, 0.01f));

            float adaptive_threshold = base_variance_threshold;

            if (luminance_mean < 0.05f) {
                adaptive_threshold *= (1.0f + (0.05f - luminance_mean) * 10.0f);
            }
            else if (luminance_mean > 0.9f) {
                adaptive_threshold *= 0.5f;
            }

            if (has_high_variance_neighbor) {
                adaptive_threshold *= 0.7f;
            }

            float confidence_factor = 1.0f / log2f(float(s) + 2.0f);
            adaptive_threshold *= fminf(confidence_factor, 2.0f);

            if (luminance_mean > 0.02f || has_high_variance_neighbor) {
                if (rel_variance < adaptive_threshold) {
                    break;
                }
            }
            else if (s >= int(min_samples * 1.5f)) {
                break;
            }
        }

    }

    // Son renk değerini hesapla
    float3 final_color = color_sum / sample_count;

    // Temporal accumulation için önceki frame ile blend etme (opsiyonel)
    if (optixLaunchParams.frame_number > 1 && optixLaunchParams.temporal_blend > 0.0f) {
        float3 prev_color = make_float3(
            optixLaunchParams.accumulation_buffer[pixel_index * 3],
            optixLaunchParams.accumulation_buffer[pixel_index * 3 + 1],
            optixLaunchParams.accumulation_buffer[pixel_index * 3 + 2]
        );

        float blend_factor = optixLaunchParams.temporal_blend;
        final_color = lerp(final_color, prev_color, blend_factor);
    }

    // Sonuçları kaydet
    optixLaunchParams.framebuffer[pixel_index] = make_color(final_color);

    // Varyans değerini bir sonraki frame için kaydet
    if (optixLaunchParams.variance_buffer) {
        float3 var = m2 / (sample_count - 1);
        float luminance_var = 0.2126f * var.x + 0.7152f * var.y + 0.0722f * var.z;
        optixLaunchParams.variance_buffer[pixel_index] = luminance_var;
    }

    // İstatistik için örnek sayısını kaydet
    if (optixLaunchParams.sample_count_buffer) {
        optixLaunchParams.sample_count_buffer[pixel_index] = sample_count;
    }
}
extern "C" __global__ void __miss__ms() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* outColor = unpackPayload<OptixHitResult>(p0, p1);   
    outColor->emission = optixLaunchParams.background_color;
    outColor->hit = 0;
}

// HIT
extern "C" __global__ void __closesthit__ch() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* payload = unpackPayload<OptixHitResult>(p0, p1);

    float t = optixGetRayTmax();
    float3 rayOrigin = optixGetWorldRayOrigin();
    float3 rayDir = optixGetWorldRayDirection();
    float3 hitPoint = rayOrigin + t * rayDir;

    float2 bary = optixGetTriangleBarycentrics();
    float u = bary.x;
    float v = bary.y;
    float w = 1.0f - u - v;
	
    const HitGroupData* hgd = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const uint3 tri = hgd->indices[primIdx];

    // Interpolated normal
    float3 normal;
    if (hgd->has_normals) {
        const float3 n0 = hgd->normals[tri.x];
        const float3 n1 = hgd->normals[tri.y];
        const float3 n2 = hgd->normals[tri.z];
        normal = normalize(w * n0 + u * n1 + v * n2);
    }
    else {
        const float3 v0 = hgd->vertices[tri.x];
        const float3 v1 = hgd->vertices[tri.y];
        const float3 v2 = hgd->vertices[tri.z];
        normal = normalize(cross(v1 - v0, v2 - v0));
    }

    // Interpolated UV
    float2 uv;
    if (hgd->has_uvs) {
        const float2 uv0 = hgd->uvs[tri.x];
        const float2 uv1 = hgd->uvs[tri.y];
        const float2 uv2 = hgd->uvs[tri.z];
        uv = w * uv0 + u * uv1 + v * uv2;
        uv.y = 1.0f - uv.y;
        // UV fract düzeltmesi
        uv.x = uv.x - floorf(uv.x);
        uv.y = uv.y - floorf(uv.y);

    }
    else {
        uv = make_float2(u, v);
       
    }
    float3 tangent = normalize(cross(normal, make_float3(0.0f, 1.0f, 0.0f)));
    if (length(tangent) < 0.1f)
        tangent = normalize(cross(normal, make_float3(1.0f, 0.0f, 0.0f)));

    // Eğer normal map varsa TBN ile dönüştür
    if (hgd->has_normal_tex) {
        // Normal interpolate et
        const float3 n0 = hgd->normals[tri.x];
        const float3 n1 = hgd->normals[tri.y];
        const float3 n2 = hgd->normals[tri.z];
        float3 interp_normal = normalize(w * n0 + u * n1 + v * n2);

        float3 final_tangent;

        if (hgd->tangents) {
            // Tangent varsa interpolate et
            const float3 t0 = hgd->tangents[tri.x];
            const float3 t1 = hgd->tangents[tri.y];
            const float3 t2 = hgd->tangents[tri.z];
            final_tangent = normalize(w * t0 + u * t1 + v * t2);
        }
        else {
            // Yoksa UV'den hesapla (senin metodun)
            const float3 p0 = hgd->vertices[tri.x];
            const float3 p1 = hgd->vertices[tri.y];
            const float3 p2 = hgd->vertices[tri.z];

            const float2 uv0 = hgd->uvs[tri.x];
            const float2 uv1 = hgd->uvs[tri.y];
            const float2 uv2 = hgd->uvs[tri.z];

            float3 edge1 = p1 - p0;
            float3 edge2 = p2 - p0;

            float2 deltaUV1 = uv1 - uv0;
            float2 deltaUV2 = uv2 - uv0;

            float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y + 1e-8f);

            final_tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            final_tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            final_tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
            final_tangent = normalize(final_tangent);

            // Gram-Schmidt
            final_tangent = normalize(final_tangent - interp_normal * dot(interp_normal, final_tangent));
        }

        // Bitangent üret
        float3 bitangent = normalize(cross(interp_normal, final_tangent));

        // Normal map oku
        float4 tex_normal4 = tex2D<float4>(hgd->normal_tex, uv.x, uv.y);
        float3 tex_normal = make_float3(tex_normal4.x, tex_normal4.y, tex_normal4.z);
        tex_normal = tex_normal * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
        // TBN matrisi ile normal dönüştür
        normal = normalize(
            tex_normal.x * final_tangent +
            tex_normal.y * bitangent +
            tex_normal.z * interp_normal
        );
    }
   
    // Payload dolduruluyor
    payload->hit = 1;
    payload->position = hitPoint;
    payload->normal = normal;    
    payload->material_id = hgd->material_id;
    payload->emission = hgd->emission;
    payload->uv = uv;
    payload->t = t;


    // Texture object'leri
    payload->albedo_tex = hgd->albedo_tex;
    payload->has_albedo_tex = hgd->has_albedo_tex;

    payload->roughness_tex = hgd->roughness_tex;
    payload->has_roughness_tex = hgd->has_roughness_tex;

    payload->normal_tex = hgd->normal_tex;
    payload->has_normal_tex = hgd->has_normal_tex;

    payload->metallic_tex = hgd->metallic_tex;
    payload->has_metallic_tex = hgd->has_metallic_tex;

    payload->transmission_tex = hgd->transmission_tex;
    payload->has_transmission_tex = hgd->has_transmission_tex;

    payload->opacity_tex = hgd->opacity_tex;
    payload->has_opacity_tex = hgd->has_opacity_tex;
	payload->emission_tex = hgd->emission_tex;
    payload->has_emission_tex = hgd->has_emission_tex;
}

extern "C" __global__ void __miss__shadow() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* shadow_payload = unpackPayload<OptixHitResult>(p0, p1);
    shadow_payload->hit = 0;
}
