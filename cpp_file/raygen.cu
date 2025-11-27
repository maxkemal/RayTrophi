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

/* launch_tile_based_progressive örneği için eski kod
* 
extern "C" __global__ void __raygen__rg() {
    // Tile-based launch için doğru koordinat hesaplama
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dim = optixGetLaunchDimensions();

    // Tile içindeki lokal koordinatlar -> Global koordinatlar
    int i = optixLaunchParams.tile_x + launch_idx.x;
    int j = optixLaunchParams.tile_y + launch_idx.y;

    // Bounds check
    if (i >= optixLaunchParams.image_width || j >= optixLaunchParams.image_height)
        return;

    int pixel_index = j * optixLaunchParams.image_width + i;

    // Seed hesaplama
    unsigned int seed = optixLaunchParams.frame_number * 719393 +
        pixel_index * 13731 +
        optixLaunchParams.current_pass * 5381 + 1337;
    curandState rng;
    curand_init(seed, 0, 0, &rng);

    const int samples_this_pass = optixLaunchParams.samples_per_pixel;
    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);

    // Örnekleme döngüsü
    for (int s = 0; s < samples_this_pass; ++s) {
        float u = (i + curand_uniform(&rng)) / float(optixLaunchParams.image_width);
        float v = (j + curand_uniform(&rng)) / float(optixLaunchParams.image_height);

        Ray ray = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);
        float3 sample = ray_color(ray, &rng);
        color_sum += sample;
    }

    float3 new_color = color_sum / samples_this_pass;

    // Accumulation (progressive refinement)
    if (optixLaunchParams.current_pass > 0) {
        float4 prev = optixLaunchParams.accumulation_buffer[pixel_index];
        float3 prev_color = make_float3(prev.x, prev.y, prev.z);
        float prev_weight = prev.w;  // Toplam sample sayısı

        float new_weight = prev_weight + samples_this_pass;
        float3 blended = (prev_color * prev_weight + new_color * samples_this_pass) / new_weight;

        optixLaunchParams.accumulation_buffer[pixel_index] = make_float4(blended.x, blended.y, blended.z, new_weight);
        new_color = blended;
    }
    else {
        optixLaunchParams.accumulation_buffer[pixel_index] = make_float4(new_color.x, new_color.y, new_color.z, samples_this_pass);
    }

    // Framebuffer'a yaz
    optixLaunchParams.framebuffer[pixel_index] = make_color(new_color);
}
*/

// RAYGEN
extern "C" __global__ void __raygen__rg() {
    int index = optixGetLaunchIndex().x;
    int i = optixLaunchParams.launch_coords_x[index];
    int j = optixLaunchParams.launch_coords_y[index];
    int pixel_index = j * optixLaunchParams.image_width + i;

    // Geliştirilmiş seed stratejisi
    unsigned int seed = optixLaunchParams.frame_number * 719393 + pixel_index * 13731 + 1337;
    curandState rng;
    curand_init(seed, 0, 0, &rng);

    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);
    float3 mean = make_float3(0.0f, 0.0f, 0.0f);
    float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
    int sample_count = 0;
    const bool use_adaptive = optixLaunchParams.use_adaptive_sampling;

    // Adaptif parametreler
    const int min_samples = optixLaunchParams.min_samples;
    const int max_samples = optixLaunchParams.max_samples;
    const float base_variance_threshold = optixLaunchParams.variance_threshold;

    // Komşu varyans analizi (5x5 çevre)
    bool has_high_variance_neighbor = false;
    float neighbor_variance_sum = 0.0f;
    int neighbor_count = 0;

    if (optixLaunchParams.frame_number > 1 && i >= 2 && i < optixLaunchParams.image_width - 2 &&
        j >= 2 && j < optixLaunchParams.image_height - 2) {

        for (int dj = -1; dj <= 1; dj++) {
            for (int di = -1; di <= 1; di++) {
                if (di == 0 && dj == 0) continue;
                float nv = optixLaunchParams.variance_buffer[(j + dj) * optixLaunchParams.image_width + (i + di)];
                float weight = 1.0f - (abs(di) + abs(dj)) * 0.25f; // Merkezi komşulara daha çok güven
                neighbor_variance_sum += nv * weight;
                neighbor_count++;
                if (nv > base_variance_threshold * 1.5f) {
                    has_high_variance_neighbor = true;
                }
            }
        }
    }

    // Dinamik min_samples ayarı
    int dynamic_min_samples = min_samples;
    if (has_high_variance_neighbor) {
        dynamic_min_samples = min(min_samples * 2, max_samples);
    }

    // Ana örnekleme döngüsü (Blue Noise benzeri dağılım)
    for (int s = 0; s < max_samples; ++s) {
        // Geliştirilmiş örnekleme
        float u = (i + (curand_uniform(&rng) + s) / max_samples) / float(optixLaunchParams.image_width);
        float v = (j + (curand_uniform(&rng) + s) / max_samples) / float(optixLaunchParams.image_height);

        Ray ray = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);
        float3 sample = ray_color(ray, &rng);
        sample_count++;

        // Welford algoritması (vectorized)
        float3 delta1 = sample - mean;
        mean += delta1 / sample_count;
        float3 delta2 = sample - mean;
        m2 += delta1 * delta2;
        color_sum += sample;

        // Adaptif erken çıkış
        if (use_adaptive && s >= dynamic_min_samples - 1) {
            float3 var = m2 / max(sample_count - 1, 1);

            // Renk kanallarına göre maksimum varyans
            float max_channel_variance = fmaxf(var.x, fmaxf(var.y, var.z));
            float luminance_mean = 0.2126f * mean.x + 0.7152f * mean.y + 0.0722f * mean.z;

            // Dinamik eşik hesaplama
            float adaptive_threshold = base_variance_threshold;

            // Parlaklığa göre ayar
            if (luminance_mean < 0.1f) {
                adaptive_threshold *= 2.0f; // Karanlık bölgeler
            }
            else if (luminance_mean > 0.9f) {
                adaptive_threshold *= 0.5f; // Parlak bölgeler
            }

            // Komşu varyans etkisi
            if (neighbor_count > 0) {
                float avg_neighbor_variance = neighbor_variance_sum / neighbor_count;
                adaptive_threshold *= fmaxf(0.5f, 1.0f - avg_neighbor_variance * 0.5f);
            }

            // Örnekleme ilerledikçe eşiği sıkılaştır
            float sample_progress = float(s - dynamic_min_samples) / float(max_samples - dynamic_min_samples);
            adaptive_threshold *= powf(1.0f - sample_progress, 1.0f);
            // Erken çıkış kontrolü
            if (max_channel_variance < adaptive_threshold) {
                break;
            }
        }
    }

    // Temporal blending (varyans temelli)
    float3 final_color = color_sum / sample_count;

    if (optixLaunchParams.frame_number > 1 && optixLaunchParams.temporal_blend > 0.0f) {
        float3 prev_color = make_float3(
            optixLaunchParams.accumulation_buffer[pixel_index * 3],
            optixLaunchParams.accumulation_buffer[pixel_index * 3 + 1],
            optixLaunchParams.accumulation_buffer[pixel_index * 3 + 2]
        );

        float blend_factor = optixLaunchParams.temporal_blend;
        float3 var = m2 / max(sample_count - 1, 1);
        float luminance_var = 0.2126f * var.x + 0.7152f * var.y + 0.0722f * var.z;

        // Gürültülü piksellerde daha az temporal blending
        if (luminance_var > base_variance_threshold) {
            blend_factor *= 0.5f;
        }

        final_color = lerp(prev_color, final_color, blend_factor);  // Daha doğal temporal blend
       
    }	
    // Sonuçları yaz
    optixLaunchParams.framebuffer[pixel_index] = make_color(final_color);

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

            float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y + 1e-6f);

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
