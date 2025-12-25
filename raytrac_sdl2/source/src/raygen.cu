#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include "material_gpu.h"
#include "params.h"
#include "payload.h"
#include "sky_model.cuh"
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

// RAYGEN - Cycles-style Accumulative Rendering
// Uses float4 accumulation buffer: RGB = accumulated color sum, W = total sample count
extern "C" __global__ void __raygen__rg() {
    // 1. Direct Indexing (Native GPU Memory Coalescing)
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();

    const int i = launch_idx.x; 
    const int j = launch_idx.y;

    // Boundary check
    if (i >= optixLaunchParams.image_width || j >= optixLaunchParams.image_height) return;

    const int pixel_index = j * optixLaunchParams.image_width + i;

    // Use frame_number (accumulated sample count) for seed variation
    // This ensures each pass gets different random samples
    unsigned int seed = optixLaunchParams.frame_number * 719393 + pixel_index * 13731 + 1337;
    curandState rng;
    curand_init(seed, 0, 0, &rng);

    // Samples per pass (usually 1 for smooth progressive refinement)
    const int samples_this_pass = optixLaunchParams.samples_per_pixel;
    
    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);

    // Sample loop - render requested number of samples for this pass
    for (int s = 0; s < samples_this_pass; ++s) {
        // Jittered sampling for anti-aliasing
        float u = (i + curand_uniform(&rng)) / float(optixLaunchParams.image_width);
        float v = (j + curand_uniform(&rng)) / float(optixLaunchParams.image_height);

        Ray ray = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);
        float3 sample = ray_color(ray, &rng);
        color_sum += sample;
    }

    // Average this pass's samples
    float3 new_color = color_sum / float(samples_this_pass);

    // Apply Camera Exposure (Must happen before accumulation if we want burned-in exposure)
    // Alternatively, apply after accumulation for post-process exposure (requires separation).
    // Current architecture resets accumulation on cam param change, so burning in is fine.
    new_color *= optixLaunchParams.camera.exposure_factor;

    // ============ ACCUMULATION LOGIC (Cycles-style) ============
    // accumulation_buffer is float4*: RGB = accumulated color * weight, W = total sample count
    float4* accum_buffer = reinterpret_cast<float4*>(optixLaunchParams.accumulation_buffer);
    
    if (accum_buffer != nullptr) {
        float4 prev = accum_buffer[pixel_index];
        float prev_samples = prev.w;  // Total samples accumulated so far
        
        if (prev_samples > 0.0f) {
            // Progressive accumulation: weighted average of old and new
            float new_total = prev_samples + samples_this_pass;
            float3 prev_color = make_float3(prev.x, prev.y, prev.z);
            
            // Weighted blend: (prev_color * prev_weight + new_color * new_weight) / total_weight
            float3 blended = (prev_color * prev_samples + new_color * samples_this_pass) / new_total;
            
            // Store accumulated result
            accum_buffer[pixel_index] = make_float4(blended.x, blended.y, blended.z, new_total);
            new_color = blended;
        }
        else {
            // First sample - just store it
            accum_buffer[pixel_index] = make_float4(new_color.x, new_color.y, new_color.z, float(samples_this_pass));
        }
    }
    
    // Write final color to display framebuffer
    optixLaunchParams.framebuffer[pixel_index] = make_color(new_color);
}
extern "C" __global__ void __miss__ms() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* outColor = unpackPayload<OptixHitResult>(p0, p1);
    
    // Default to background color (or updated by Sky)
    float3 current_bg = optixLaunchParams.background_color;

    if (optixLaunchParams.world.mode == 2) { // WORLD_MODE_NISHITA
         float3 ray_dir = optixGetWorldRayDirection();
         current_bg = calculate_nishita_sky_gpu(ray_dir, optixLaunchParams.world.nishita);
    } 
    
    outColor->emission = current_bg;

    // --- INFINITE GRID SHADER (GPU Port) ---
    float3 ray_dir = optixGetWorldRayDirection();
    float3 ray_origin = optixGetWorldRayOrigin();

    // Only if enabled AND ray is looking down
    if (optixLaunchParams.grid_enabled && ray_dir.y < -1e-4f) {
        
        // Plane Intersection (Y=0)
        float t = -ray_origin.y / ray_dir.y;
        
        if (t > 0.0f) {
             float3 p = ray_origin + ray_dir * t;
             
             // 1. Distance Fading
             float fade_start = 100.0f;
             float fade_end = optixLaunchParams.grid_fade_distance;
             if (fade_end < fade_start) fade_end = fade_start + 100.0f;
             
             float dist = t;
             float alpha_fade = 1.0f - fminf(fmaxf((dist - fade_start) / (fade_end - fade_start), 0.0f), 1.0f);
             
             if (alpha_fade > 0.0f) {
                 // 2. Grid Structure
                 float scale_primary = 10.0f;
                 float scale_secondary = 1.0f;
                 
                 float line_width_base = 0.02f;
                 float line_width = line_width_base * (1.0f + dist * 0.02f);
                 
                 // Modulo
                 float x_mod_p = fabsf(fmodf(p.x, scale_primary));
                 float z_mod_p = fabsf(fmodf(p.z, scale_primary));
                 float x_mod_s = fabsf(fmodf(p.x, scale_secondary));
                 float z_mod_s = fabsf(fmodf(p.z, scale_secondary));
                 
                 // Line Checks (Inline)
                 bool x_line_p = x_mod_p < line_width || x_mod_p > (scale_primary - line_width);
                 bool z_line_p = z_mod_p < line_width || z_mod_p > (scale_primary - line_width);
                 bool x_line_s = x_mod_s < line_width || x_mod_s > (scale_secondary - line_width);
                 bool z_line_s = z_mod_s < line_width || z_mod_s > (scale_secondary - line_width);
                 
                 // Axis
                 bool x_axis = fabsf(p.z) < line_width * 2.5f;
                 bool z_axis = fabsf(p.x) < line_width * 2.5f;
                 
                 float3 grid_col = make_float3(0.0f, 0.0f, 0.0f);
                 float grid_alpha = 0.0f;
                 
                 if (x_axis) { grid_col = make_float3(0.8f, 0.2f, 0.2f); grid_alpha = 0.9f; }
                 else if (z_axis) { grid_col = make_float3(0.2f, 0.8f, 0.2f); grid_alpha = 0.9f; }
                 else if (x_line_p || z_line_p) { grid_col = make_float3(0.40f, 0.40f, 0.40f); grid_alpha = 0.5f; }
                 else if (x_line_s || z_line_s) { grid_col = make_float3(0.25f, 0.25f, 0.25f); grid_alpha = 0.2f; }
                 
                 if (grid_alpha > 0.0f) {
                     float final_alpha = grid_alpha * alpha_fade;
                     // Blend
                     outColor->emission = current_bg * (1.0f - final_alpha) + grid_col * final_alpha;
                 }
             }
        }
    }

    outColor->hit = 0;
}

// HIT
extern "C" __global__ void __closesthit__ch() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* payload = unpackPayload<OptixHitResult>(p0, p1);

    const HitGroupData* hgd = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    
    // Restore missing t definition
    float t = optixGetRayTmax();

    float3 rayOrigin = optixGetWorldRayOrigin();
    float3 rayDir = optixGetWorldRayDirection();
    float3 hitPoint = rayOrigin + t * rayDir;

    float2 bary = optixGetTriangleBarycentrics();
    float u = bary.x;
    float v = bary.y;
    float w = 1.0f - u - v;
	
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
    
    // Volumetric material info
    payload->is_volumetric = hgd->is_volumetric;
    payload->vol_density = hgd->vol_density;
    payload->vol_absorption = hgd->vol_absorption;
    payload->vol_scattering = hgd->vol_scattering;
    payload->vol_albedo = hgd->vol_albedo;
    payload->vol_emission = hgd->vol_emission;
    payload->vol_g = hgd->vol_g;
    payload->vol_step_size = hgd->vol_step_size;
    payload->vol_max_steps = hgd->vol_max_steps;
    payload->vol_noise_scale = hgd->vol_noise_scale;
    payload->aabb_min = hgd->aabb_min;
    payload->aabb_max = hgd->aabb_max;
    
    // Multi-Scattering parameters (NEW)
    payload->vol_multi_scatter = hgd->vol_multi_scatter;
    payload->vol_g_back = hgd->vol_g_back;
    payload->vol_lobe_mix = hgd->vol_lobe_mix;
    payload->vol_light_steps = hgd->vol_light_steps;
    payload->vol_shadow_strength = hgd->vol_shadow_strength;
}

extern "C" __global__ void __anyhit__ah() {
    const HitGroupData* hgd = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    float alpha = 1.0f;

    // 1. Get scalar opacity from global material buffer
    if (hgd->material_id >= 0 && optixLaunchParams.materials) {
        alpha = optixLaunchParams.materials[hgd->material_id].opacity;
    }

    // 2. Multiply with opacity map if it exists
    if (hgd->has_opacity_tex) {
        const unsigned int primIdx = optixGetPrimitiveIndex();
        const uint3 tri = hgd->indices[primIdx];
        
        float2 bary = optixGetTriangleBarycentrics();
        float u = bary.x;
        float v = bary.y;
        float w = 1.0f - u - v;

        float2 uv;
        if (hgd->has_uvs) {
            const float2 uv0 = hgd->uvs[tri.x];
            const float2 uv1 = hgd->uvs[tri.y];
            const float2 uv2 = hgd->uvs[tri.z];
            uv = w * uv0 + u * uv1 + v * uv2;
            uv.y = 1.0f - uv.y;
             // UV fract correction
            uv.x = uv.x - floorf(uv.x);
            uv.y = uv.y - floorf(uv.y);
        } else {
             uv = make_float2(u, v);
        }

        // Opacity texture sample (single channel from float4)
        // Usually opacity is in the alpha channel or single channel (r). 
        // AssimpLoader puts it in 'opacity' float or texture.
        float4 opacity_val = tex2D<float4>(hgd->opacity_tex, uv.x, uv.y);
        alpha *= opacity_val.x; 
    }

    // Transparent Shadow Logic
    // If opacity is less than 0.9, do not block the shadow ray.
    if (alpha < 0.9f) {
        optixIgnoreIntersection();
    }
}


extern "C" __global__ void __miss__shadow() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* payload = unpackPayload<OptixHitResult>(p0, p1);
    payload->hit = 0; // Miss = No occlusion (Lit)
}
