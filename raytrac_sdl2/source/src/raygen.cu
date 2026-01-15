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
#include "water_shaders.cuh"
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
        float u = (i + pcg_float(&pcg_rng)) / float(optixLaunchParams.image_width);
        float v = (j + pcg_float(&pcg_rng)) / float(optixLaunchParams.image_height);

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
    
    // PERFORMANCE OPTIMIZATION: Use PCG for jitter, curand for ray tracing
    // PCG initialization is ~100x faster than curand_init
    PCGState pcg_rng;
    pcg_init(&pcg_rng, seed);
    
    // curand still needed for ray_color and get_ray_from_camera
    // But we use simplified init with sequence=0, offset=0 (faster)
    curandState rng;
    curand_init(seed, 0, 0, &rng);

    // Samples per pass (usually 1 for smooth progressive refinement)
    const int samples_this_pass = optixLaunchParams.samples_per_pixel;
    
    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);

    // Sample loop - render requested number of samples for this pass
    for (int s = 0; s < samples_this_pass; ++s) {
        // Jittered sampling for anti-aliasing
        float u = (i + pcg_float(&pcg_rng)) / float(optixLaunchParams.image_width);
        float v = (j + pcg_float(&pcg_rng)) / float(optixLaunchParams.image_height);

        // ═══════════════════════════════════════════════════════════════════════════
        // CHROMATIC ABERRATION (Cinema Mode)
        // Traces separate rays for R, G, B channels with radially offset UVs
        // ═══════════════════════════════════════════════════════════════════════════
        if (optixLaunchParams.camera.chromatic_aberration_enabled && 
            optixLaunchParams.camera.chromatic_aberration > 0.0001f) {
            
            float ca_amount = optixLaunchParams.camera.chromatic_aberration;
            float r_scale = optixLaunchParams.camera.chromatic_aberration_r; // > 1.0 = outward
            float b_scale = optixLaunchParams.camera.chromatic_aberration_b; // < 1.0 = inward
            
            // Red channel - slightly outward
            float2 uv_r = apply_chromatic_aberration(u, v, ca_amount, r_scale);
            Ray ray_r = get_ray_from_camera(optixLaunchParams.camera, uv_r.x, uv_r.y, &rng);
            float3 color_r = ray_color(ray_r, &rng);
            
            // Green channel - no offset
            Ray ray_g = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);
            float3 color_g = ray_color(ray_g, &rng);
            
            // Blue channel - slightly inward
            float2 uv_b = apply_chromatic_aberration(u, v, ca_amount, b_scale);
            Ray ray_b = get_ray_from_camera(optixLaunchParams.camera, uv_b.x, uv_b.y, &rng);
            float3 color_b = ray_color(ray_b, &rng);
            
            // Combine channels
            color_sum += make_float3(color_r.x, color_g.y, color_b.z);
        } else {
            // Standard rendering - single ray for all channels
            Ray ray = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);
            float3 sample = ray_color(ray, &rng);
            color_sum += sample;
        }
    }

    // Average this pass's samples
    float3 new_color = color_sum / float(samples_this_pass);

    // Apply Camera Exposure (Must happen before accumulation if we want burned-in exposure)
    // Alternatively, apply after accumulation for post-process exposure (requires separation).
    // Current architecture resets accumulation on cam param change, so burning in is fine.
    new_color *= optixLaunchParams.camera.exposure_factor;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CINEMA MODE - Vignetting (Köşe Kararması)
    // Applied after exposure to simulate lens light falloff
    // ═══════════════════════════════════════════════════════════════════════════
    if (optixLaunchParams.camera.vignetting_enabled) {
        // Calculate normalized distance from center (0 at center, 1 at corner)
        float norm_x = (float(i) / float(optixLaunchParams.image_width)) * 2.0f - 1.0f;
        float norm_y = (float(j) / float(optixLaunchParams.image_height)) * 2.0f - 1.0f;
        float dist_sq = norm_x * norm_x + norm_y * norm_y;
        
        // Vignette factor: 1 at center, decreasing towards corners
        // Using power function for controllable falloff
        float vignette_radius = 1.414f; // sqrt(2) = corner distance
        float normalized_dist = sqrtf(dist_sq) / vignette_radius;
        
        // Power-based falloff with user-controlled exponent
        float falloff = optixLaunchParams.camera.vignetting_falloff;
        float vignette_factor = 1.0f - optixLaunchParams.camera.vignetting_amount * 
                                powf(normalized_dist, falloff);
        vignette_factor = fmaxf(vignette_factor, 0.0f); // Clamp to prevent negative
        
        new_color *= vignette_factor;
    }

    // ============ ACCUMULATION LOGIC (Cycles-style) ============
    // accumulation_buffer is float4*: RGB = accumulated color sum, W = total sample count
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
// HIT
extern "C" __global__ void __closesthit__ch() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* payload = unpackPayload<OptixHitResult>(p0, p1);

    const HitGroupData* hgd = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    
    // 1. Geometric Context
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

    // 2. Interpolate UV
    float2 uv;
    if (hgd->has_uvs) {
        const float2 uv0 = hgd->uvs[tri.x];
        const float2 uv1 = hgd->uvs[tri.y];
        const float2 uv2 = hgd->uvs[tri.z];
        uv = w * uv0 + u * uv1 + v * uv2;
        uv.y = 1.0f - uv.y;
        uv.x = uv.x - floorf(uv.x);
        uv.y = uv.y - floorf(uv.y);
    } else {
        uv = make_float2(u, v);
    }
    
    // 3. Transform Matrix (needed for determinants)
    float o2w[12];
    optixGetObjectToWorldTransformMatrix(o2w);
    float m00 = o2w[0], m01 = o2w[1], m02 = o2w[2];
    float m10 = o2w[4], m11 = o2w[5], m12 = o2w[6];
    float m20 = o2w[8], m21 = o2w[9], m22 = o2w[10];

    // 4. Calculate Normals
    // a) Geometric Normal (Object Space)
    const float3 vert0 = hgd->vertices[tri.x];
    const float3 vert1 = hgd->vertices[tri.y];
    const float3 vert2 = hgd->vertices[tri.z];
    float3 obj_geo_normal = normalize(cross(vert1 - vert0, vert2 - vert0));
    float3 world_geo_normal = normalize(optixTransformNormalFromObjectToWorldSpace(obj_geo_normal));

    // b) Shading Normal (Object Space) -> World Space
    float3 world_normal;
    if (hgd->has_normals) {
        const float3 n0 = hgd->normals[tri.x];
        const float3 n1 = hgd->normals[tri.y];
        const float3 n2 = hgd->normals[tri.z];
        float3 obj_normal = normalize(w * n0 + u * n1 + v * n2);
        world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(obj_normal));
    } else {
        world_normal = world_geo_normal;
        // Also update obj_normal for tangent calc fallback
    }

    // c) Faceforward
    if (dot(world_geo_normal, rayDir) > 0.0f) {
        world_normal = -world_normal;
        world_geo_normal = -world_geo_normal; // Ensure TBN consistency
    }

    // 5. Tangent & Bitangent Setup (Common)
    // NOTE: For skinned meshes, vertex tangents are NOT updated by skinning kernel,
    // so we need to compute tangents from UV gradients to stay consistent with skinned normals.
    float3 world_tangent, world_bitangent;
    {
         float sigma = 1.0f;
         
         // Calculate Sigma (Handedness) from transform determinant
         float det_transform = m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20);
         float sigma_inst = (det_transform < 0.0f) ? -1.0f : 1.0f;
         
         // Get world-space edges for UV-based tangent calculation
         float3 edge1 = vert1 - vert0;
         float3 edge2 = vert2 - vert0;
         
         // Transform edges to world space (vertices are already in object space from SBT)
         float3 world_edge1, world_edge2;
         world_edge1.x = m00 * edge1.x + m01 * edge1.y + m02 * edge1.z;
         world_edge1.y = m10 * edge1.x + m11 * edge1.y + m12 * edge1.z;
         world_edge1.z = m20 * edge1.x + m21 * edge1.y + m22 * edge1.z;
         
         world_edge2.x = m00 * edge2.x + m01 * edge2.y + m02 * edge2.z;
         world_edge2.y = m10 * edge2.x + m11 * edge2.y + m12 * edge2.z;
         world_edge2.z = m20 * edge2.x + m21 * edge2.y + m22 * edge2.z;
         
         bool use_uv_tangent = false;
         
         if (hgd->has_uvs) {
             const float2 uv0_local = hgd->uvs[tri.x];
             const float2 uv1_local = hgd->uvs[tri.y];
             const float2 uv2_local = hgd->uvs[tri.z];
             float2 dUV1 = uv1_local - uv0_local;
             float2 dUV2 = uv2_local - uv0_local;
             float det_uv = (dUV1.x * dUV2.y - dUV2.x * dUV1.y);
             
             // UV-based tangent calculation (works correctly with skinned normals)
             if (fabsf(det_uv) > 1e-8f) {
                 float inv_det = 1.0f / det_uv;
                 
                 // T = (dUV2.y * edge1 - dUV1.y * edge2) / det
                 world_tangent.x = inv_det * (dUV2.y * world_edge1.x - dUV1.y * world_edge2.x);
                 world_tangent.y = inv_det * (dUV2.y * world_edge1.y - dUV1.y * world_edge2.y);
                 world_tangent.z = inv_det * (dUV2.y * world_edge1.z - dUV1.y * world_edge2.z);
                 world_tangent = normalize(world_tangent);
                 
                 float sigma_uv = (det_uv < 0.0f) ? -1.0f : 1.0f;
                 sigma = sigma_uv * sigma_inst;
                 use_uv_tangent = true;
             }
         }
         
         // Fallback: Use stored tangents only if UV-based failed AND we have pre-computed tangents
         // For skinned meshes, this is less reliable since tangents aren't skinned
         if (!use_uv_tangent) {
             if (hgd->has_tangents) {
                 const float3 t0 = hgd->tangents[tri.x];
                 const float3 t1 = hgd->tangents[tri.y];
                 const float3 t2 = hgd->tangents[tri.z];
                 float3 obj_tangent = normalize(w * t0 + u * t1 + v * t2);
                 
                 // Transform to world space
                 world_tangent.x = m00 * obj_tangent.x + m01 * obj_tangent.y + m02 * obj_tangent.z;
                 world_tangent.y = m10 * obj_tangent.x + m11 * obj_tangent.y + m12 * obj_tangent.z;
                 world_tangent.z = m20 * obj_tangent.x + m21 * obj_tangent.y + m22 * obj_tangent.z;
                 world_tangent = normalize(world_tangent);
             } else {
                 // Last resort: geometric tangent from first edge
                 world_tangent = normalize(world_edge1); 
             }
         }

         // Gram-Schmidt Orthogonalization against the skinned/transformed normal
         // This ensures TBN is consistent even when tangent wasn't skinned
         world_tangent = normalize(world_tangent - world_normal * dot(world_normal, world_tangent));
         
         // Bitangent from cross product (respects handedness)
         world_bitangent = normalize(cross(world_normal, world_tangent)) * sigma;
    }

    // 6. LOGIC BRANCH: TERRAIN vs STANDARD
    float3 final_normal = world_normal;
    payload->use_blended_data = 0;

    if (hgd->is_terrain && hgd->splat_map_tex) {
        // --- TERRAIN LAYER BLENDING ---
        // Note: Splat map pixel array is top-down but texture coords are bottom-up, so flip Y
        float4 mask = tex2D<float4>(hgd->splat_map_tex, uv.x, 1.0f - uv.y);
        
        float3 blended_albedo = make_float3(0.0f);
        float blended_roughness = 0.0f;
        float3 blended_ts_normal = make_float3(0.0f);
        float total_weight = 0.0f;

        for (int i = 0; i < 4; ++i) {
            float weight = (i==0) ? mask.x : (i==1) ? mask.y : (i==2) ? mask.z : mask.w;
            if (weight < 0.001f) continue;

            float2 layer_uv = uv * hgd->layer_uv_scale[i]; // Apply scale

            // Albedo
            float3 col = make_float3(1.0f);
            if (hgd->layer_albedo_tex[i]) {
                float4 c = tex2D<float4>(hgd->layer_albedo_tex[i], layer_uv.x, layer_uv.y);
                // Simple linearization (approximation)
                col = make_float3(c.x*c.x, c.y*c.y, c.z*c.z); 
            }
            
            // Roughness
            float r = 0.5f;
            if (hgd->layer_roughness_tex[i]) {
                r = tex2D<float4>(hgd->layer_roughness_tex[i], layer_uv.x, layer_uv.y).x;
            }

            // Normal
            float3 n_ts = make_float3(0,0,1);
            if (hgd->layer_normal_tex[i]) {
                float4 n = tex2D<float4>(hgd->layer_normal_tex[i], layer_uv.x, layer_uv.y);
                n_ts = make_float3(n.x, n.y, n.z) * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
            }
            
            blended_albedo += col * weight;
            blended_roughness += r * weight;
            blended_ts_normal += n_ts * weight;
            total_weight += weight;
        }

        if (total_weight > 0.0f) {
            // Normalize blended data? 
            // Splat map sum usually 1.0, but safety check:
             blended_albedo /= total_weight;
             blended_roughness /= total_weight;
             blended_ts_normal = normalize(blended_ts_normal); // Re-normalize normal vector
             
             // Apply TBN to blended tangent-space normal
             final_normal = normalize(
                 blended_ts_normal.x * world_tangent +
                 blended_ts_normal.y * world_bitangent +
                 blended_ts_normal.z * world_normal
             );
             
             payload->blended_albedo = blended_albedo;
             payload->blended_roughness = blended_roughness;
             payload->use_blended_data = 1;
        }
    } 
    else {
        // --- STANDARD MATERIAL ---
        // Normal Map
        if (hgd->has_normal_tex) {
            float4 n_val = tex2D<float4>(hgd->normal_tex, uv.x, uv.y);
            float3 ts_normal = make_float3(n_val.x, n_val.y, n_val.z) * 2.0f - make_float3(1.0f, 1.0f, 1.0f);
            
            final_normal = normalize(
                ts_normal.x * world_tangent +
                ts_normal.y * world_bitangent +
                ts_normal.z * world_normal
            );
        }
    }

    // 7. Water Wave Perturbation & Advanced Effects
    if (hgd->material_id >= 0 && optixLaunchParams.materials) {
         const GpuMaterial& mat = optixLaunchParams.materials[hgd->material_id];
         
         // Check IS_WATER flag (sheen > 0)
         if (mat.sheen > 0.0001f) {
             float time = optixLaunchParams.water_time;  // Use frozen water time for accumulation
             
             // === UNPACK WATER PARAMETERS FROM GPU MATERIAL ===
             WaterParams params;
             
             // Wave params
             params.wave_speed = mat.anisotropic;
             params.wave_strength = mat.sheen;
             params.wave_frequency = mat.sheen_tint;
             
             // Colors
             params.shallow_color = mat.emission;
             params.deep_color = mat.albedo;
             params.absorption_color = mat.subsurface_color;
             
             // Depth & Appearance
             params.depth_max = mat.subsurface * 100.0f;
             params.absorption_density = mat.subsurface_scale;
             params.clarity = fmaxf(0.1f, 1.0f - params.absorption_density);
             
             // Foam
             params.foam_level = mat.translucent;
             params.shore_foam_distance = mat.subsurface_radius.x;
             params.shore_foam_intensity = mat.clearcoat;
             
             // Caustics
             params.caustic_intensity = mat.clearcoat_roughness;
             params.caustic_scale = mat.subsurface_radius.y;
             params.caustic_speed = mat.subsurface_anisotropy;
             
             // SSS
             params.sss_intensity = mat.subsurface_radius.z;
             params.sss_color = mat.subsurface_color;
             
             // FFT Ocean
             params.use_fft_ocean = (mat.fft_height_tex != 0);
             params.fft_ocean_size = mat.fft_ocean_size;
             params.fft_choppiness = mat.fft_choppiness;
             params.fft_height_tex = mat.fft_height_tex;
             params.fft_normal_tex = mat.fft_normal_tex;
             
             // Micro Details
             params.micro_detail_strength = mat.micro_detail_strength;
             params.micro_detail_scale = mat.micro_detail_scale;
             params.foam_noise_scale = mat.foam_noise_scale;
             params.foam_threshold = mat.foam_threshold;
             
             // === WATER EVALUATION (FFT or Gerstner) ===
             WaterResult wave_res = evaluateWater(
                 hitPoint, final_normal, time, params
             );
             
             // Apply wave normal perturbation
             final_normal = wave_res.normal;
             
             // === NORMAL PERTURBATION ONLY ===
            final_normal = wave_res.normal;
            
            // === FOAM CALCULATION ===
            float shore_foam = 0.0f;
            // (Optional: Simple height-based shore foam proxy if needed, else rely on texture)
             if (params.shore_foam_intensity > 0.01f) {
                 float height_factor = fmaxf(wave_res.height + 0.5f, 0.0f);
                 shore_foam = calculateShoreFoam(
                     1.0f,  // Dummy depth
                     params.shore_foam_distance,
                     params.shore_foam_intensity,
                     hitPoint,
                     time
                 );
             }
             
             float total_foam = fminf(wave_res.foam * params.foam_level + shore_foam, 1.0f);
             
             // === BLENDED DATA FOR FOAM ===
             // We do NOT bake absorption color here. Let path tracer handle Beer's Law.
             // We only blend foam into the base albedo.
             
             float3 base_color = mat.albedo;
             if (total_foam > 0.01f) {
                 float3 foam_color = make_float3(0.92f, 0.96f, 1.0f);
                 // Mix base albedo with foam
                 base_color = lerp(base_color, foam_color, total_foam);
                 
                 // Reduce roughness where foam is
                 payload->blended_roughness = lerp(mat.roughness, 0.8f, total_foam); 
                 payload->use_blended_data = 1;
             } else {
                 // No foam, use standard material properties
                 payload->use_blended_data = 0; 
             }
             
             payload->blended_albedo = base_color;
        }
    }

    // 8. Pack Payload
    payload->hit = 1;
    payload->position = hitPoint;
    payload->normal = final_normal;
    payload->material_id = hgd->material_id;
    payload->uv = uv;  // CRITICAL: Pass UV to shader for texture sampling!
    
    // Pass standard textures (Scatter function might still check them if blended_data=0)
    payload->albedo_tex = hgd->albedo_tex;
    payload->roughness_tex = hgd->roughness_tex;
    payload->normal_tex = hgd->normal_tex;
    payload->metallic_tex = hgd->metallic_tex;
    payload->transmission_tex = hgd->transmission_tex;
    payload->opacity_tex = hgd->opacity_tex;
    payload->emission_tex = hgd->emission_tex;
    
    payload->has_albedo_tex = hgd->has_albedo_tex;
    payload->has_roughness_tex = hgd->has_roughness_tex;
    payload->has_normal_tex = hgd->has_normal_tex;
    payload->has_metallic_tex = hgd->has_metallic_tex;
    payload->has_transmission_tex = hgd->has_transmission_tex;
    payload->has_opacity_tex = hgd->has_opacity_tex;
    payload->has_emission_tex = hgd->has_emission_tex;
    
    // Volumetrics
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
    
    payload->vol_multi_scatter = hgd->vol_multi_scatter;
    payload->vol_g_back = hgd->vol_g_back;
    payload->vol_lobe_mix = hgd->vol_lobe_mix;
    payload->vol_light_steps = hgd->vol_light_steps;
    payload->vol_shadow_strength = hgd->vol_shadow_strength;
    
    // NanoVDB grid (for VDB-based volumetrics)
    payload->nanovdb_grid = hgd->nanovdb_grid;
    payload->has_nanovdb = hgd->has_nanovdb;
    
    // Mesafeyi kaydet (Beer's Law için gerekli)
    payload->t = optixGetRayTmax();
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

    // ------------------------------------------------------------------
    // STOCHASTIC TRANSPARENCY (Matches CPU Logic)
    // ------------------------------------------------------------------
    if (alpha < 1.0f) {
        // Generate pseudo-random value for this intersection
        // We use ray properties and frame index to ensure randomness per sample
        float3 ray_origin = optixGetWorldRayOrigin();
        float3 ray_dir = optixGetWorldRayDirection();
        uint3 launch_idx = optixGetLaunchIndex();
        
        // Simple distinct hash components
        float seed = ray_origin.x * 73.0f + ray_origin.y * 19.0f + ray_origin.z * 43.0f +
                     ray_dir.x * 37.0f + ray_dir.y * 11.0f + ray_dir.z * 5.0f +
                     (float)launch_idx.x * 0.1f + (float)launch_idx.y * 0.1f +
                     (float)optixLaunchParams.frame_number * 13.0f; // Enable temporal variation
                     
        // Linear Congruential Generator-like modulation
        float rnd = fmodf(fabsf(sinf(seed) * 43758.5453f), 1.0f);
        
        // Alpha Test: If random value is greater than alpha, the surface is transparent (hole)
        if (rnd > alpha) {
            optixIgnoreIntersection();
        }
    }
}


extern "C" __global__ void __miss__shadow() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* payload = unpackPayload<OptixHitResult>(p0, p1);
    payload->hit = 0; // Miss = No occlusion (Lit)
}
