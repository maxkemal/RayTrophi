#include <optix.h>
#include <optix_device.h>
#include "ray_color.cuh"
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include "material_gpu.h"
#include "params.h"
#include "vec3_utils.cuh"
#include "water_shaders.cuh"
#include "hair_bsdf.cuh"
#include "random_utils.cuh"

extern "C" __constant__ RayGenParams optixLaunchParams;

// Standard Closest Hit
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
    } else {
        uv = make_float2(u, v);
    }
    
    // 3. Transform Matrix
    float o2w[12];
    optixGetObjectToWorldTransformMatrix(o2w);
    float m00 = o2w[0], m01 = o2w[1], m02 = o2w[2];
    float m10 = o2w[4], m11 = o2w[5], m12 = o2w[6];
    float m20 = o2w[8], m21 = o2w[9], m22 = o2w[10];
    
    // 4. Calculate Normals
    const float3 vert0 = hgd->vertices[tri.x];
    const float3 vert1 = hgd->vertices[tri.y];
    const float3 vert2 = hgd->vertices[tri.z];
    float3 obj_geo_normal = normalize(cross(vert1 - vert0, vert2 - vert0));
    float3 world_geo_normal = normalize(optixTransformNormalFromObjectToWorldSpace(obj_geo_normal));

    float3 world_normal;
    if (hgd->has_normals) {
        const float3 n0 = hgd->normals[tri.x];
        const float3 n1 = hgd->normals[tri.y];
        const float3 n2 = hgd->normals[tri.z];
        float3 obj_normal = normalize(w * n0 + u * n1 + v * n2);
        world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(obj_normal));
    } else {
        world_normal = world_geo_normal;
    }

    if (dot(world_geo_normal, rayDir) > 0.0f) {
        world_normal = -world_normal;
        world_geo_normal = -world_geo_normal;
    }

    // 5. Tangent & Bitangent Setup
    float3 world_tangent, world_bitangent;
    {
         float sigma = 1.0f;
         float det_transform = m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20);
         float sigma_inst = (det_transform < 0.0f) ? -1.0f : 1.0f;
         
         float3 edge1 = vert1 - vert0;
         float3 edge2 = vert2 - vert0;
         
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
             
             if (fabsf(det_uv) > 1e-8f) {
                 float inv_det = 1.0f / det_uv;
                 world_tangent.x = inv_det * (dUV2.y * world_edge1.x - dUV1.y * world_edge2.x);
                 world_tangent.y = inv_det * (dUV2.y * world_edge1.y - dUV1.y * world_edge2.y);
                 world_tangent.z = inv_det * (dUV2.y * world_edge1.z - dUV1.y * world_edge2.z);
                 world_tangent = normalize(world_tangent);
                 
                 float sigma_uv = (det_uv < 0.0f) ? -1.0f : 1.0f;
                 sigma = sigma_uv * sigma_inst;
                 use_uv_tangent = true;
             }
         }
         
         if (!use_uv_tangent) {
             if (hgd->has_tangents) {
                 const float3 t0 = hgd->tangents[tri.x];
                 const float3 t1 = hgd->tangents[tri.y];
                 const float3 t2 = hgd->tangents[tri.z];
                 float3 obj_tangent = normalize(w * t0 + u * t1 + v * t2);
                 world_tangent.x = m00 * obj_tangent.x + m01 * obj_tangent.y + m02 * obj_tangent.z;
                 world_tangent.y = m10 * obj_tangent.x + m11 * obj_tangent.y + m12 * obj_tangent.z;
                 world_tangent.z = m20 * obj_tangent.x + m21 * obj_tangent.y + m22 * obj_tangent.z;
                 world_tangent = normalize(world_tangent);
             } else {
                 world_tangent = normalize(world_edge1); 
             }
         }
         world_tangent = normalize(world_tangent - world_normal * dot(world_normal, world_tangent));
         world_bitangent = normalize(cross(world_normal, world_tangent)) * sigma;
    }

    // 6. Terrain vs Standard
    float3 final_normal = world_normal;
    payload->use_blended_data = 0;

    if (hgd->is_terrain && hgd->splat_map_tex) {
        float4 mask = tex2D<float4>(hgd->splat_map_tex, uv.x, uv.y);
        float3 blended_albedo = make_float3(0.0f);
        float blended_roughness = 0.0f;
        float3 blended_ts_normal = make_float3(0.0f);
        float total_weight = 0.0f;

        for (int i = 0; i < 4; ++i) {
            float weight = (i==0) ? mask.x : (i==1) ? mask.y : (i==2) ? mask.z : mask.w;
            if (weight < 0.001f) continue;
            float2 layer_uv = uv * hgd->layer_uv_scale[i];
            float3 col = make_float3(1.0f);
            if (hgd->layer_albedo_tex[i]) {
                float4 c = tex2D<float4>(hgd->layer_albedo_tex[i], layer_uv.x, layer_uv.y);
                col = make_float3(c.x, c.y, c.z); 
            }
            float r = 0.5f;
            if (hgd->layer_roughness_tex[i]) {
                r = tex2D<float4>(hgd->layer_roughness_tex[i], layer_uv.x, layer_uv.y).x;
            }
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
             blended_albedo /= total_weight;
             blended_roughness /= total_weight;
             blended_ts_normal = normalize(blended_ts_normal);
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

    // 7. Water Wave Perturbation
    if (hgd->material_id >= 0 && optixLaunchParams.materials) {
         const GpuMaterial& mat = optixLaunchParams.materials[hgd->material_id];
         if (mat.sheen > 0.0001f) {
             float time = optixLaunchParams.water_time;
             WaterParams params;
             params.wave_speed = mat.anisotropic;
             params.wave_strength = mat.sheen;
             params.wave_frequency = mat.sheen_tint;
             params.shallow_color = mat.emission;
             params.deep_color = mat.albedo;
             params.absorption_color = mat.subsurface_color;
             params.depth_max = mat.subsurface * 100.0f;
             params.absorption_density = mat.subsurface_scale;
             params.clarity = fmaxf(0.1f, 1.0f - params.absorption_density);
             params.foam_level = mat.translucent;
             params.shore_foam_distance = mat.subsurface_radius.x;
             params.shore_foam_intensity = mat.clearcoat;
             params.caustic_intensity = mat.clearcoat_roughness;
             params.caustic_scale = mat.subsurface_radius.y;
             params.caustic_speed = mat.subsurface_anisotropy;
             params.sss_intensity = mat.subsurface_radius.z;
             params.sss_color = mat.subsurface_color;
             params.use_fft_ocean = (mat.fft_height_tex != 0);
             params.fft_ocean_size = mat.fft_ocean_size;
             params.fft_choppiness = mat.fft_choppiness;
             params.fft_height_tex = mat.fft_height_tex;
             params.fft_normal_tex = mat.fft_normal_tex;
             params.micro_detail_strength = mat.micro_detail_strength;
             params.micro_detail_scale = mat.micro_detail_scale;
             params.micro_anim_speed = mat.micro_anim_speed;
             params.micro_morph_speed = mat.micro_morph_speed;
             params.foam_noise_scale = mat.foam_noise_scale;
             params.foam_threshold = mat.foam_threshold;
             params.wind_direction = mat.fft_wind_direction;
             params.wind_speed = mat.fft_wind_speed;
             params.time = time;

             WaterResult wave_res = evaluateWater(hitPoint, final_normal, time, params);
             final_normal = wave_res.normal;
             
             float shore_foam = 0.0f;
             if (params.shore_foam_intensity > 0.01f) {
                 shore_foam = calculateShoreFoam(1.0f, params.shore_foam_distance, params.shore_foam_intensity, hitPoint, time);
             }
             float total_foam = fminf(wave_res.foam * params.foam_level + shore_foam, 1.0f);
             float3 base_color = mat.albedo;
             if (total_foam > 0.01f) {
                 float3 foam_color = make_float3(0.92f, 0.96f, 1.0f);
                 base_color = lerp(base_color, foam_color, total_foam);
                 payload->blended_roughness = lerp(mat.roughness, 0.8f, total_foam); 
                 payload->use_blended_data = 1;
             }
             payload->blended_albedo = base_color;
         }
    }

    // 8. Pack Payload
    payload->hit = 1;
    payload->position = hitPoint;
    payload->normal = final_normal;
    payload->material_id = hgd->material_id;
    payload->uv = uv;
    
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
    payload->opacity_has_alpha = hgd->opacity_has_alpha;
    
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
    payload->nanovdb_grid = hgd->nanovdb_grid;
    payload->has_nanovdb = hgd->has_nanovdb;
    payload->object_id = hgd->object_id;
    payload->t = t;
}

// Any Hit
extern "C" __global__ void __anyhit__ah() {
    const HitGroupData* hgd = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    float alpha = 1.0f;
    if (hgd->material_id >= 0 && optixLaunchParams.materials) {
        alpha = optixLaunchParams.materials[hgd->material_id].opacity;
    }
    if (hgd->has_opacity_tex) {
        const unsigned int primIdx = optixGetPrimitiveIndex();
        const uint3 tri = hgd->indices[primIdx];
        float2 bary = optixGetTriangleBarycentrics();
        float2 uv;
        if (hgd->has_uvs) {
            float2 uv0 = hgd->uvs[tri.x];
            float2 uv1 = hgd->uvs[tri.y];
            float2 uv2 = hgd->uvs[tri.z];
            uv = (1.0f - bary.x - bary.y) * uv0 + bary.x * uv1 + bary.y * uv2;
            uv.y = 1.0f - uv.y;
        } else {
            uv = bary;
        }
        float4 opacity_val = tex2D<float4>(hgd->opacity_tex, uv.x, uv.y);
        float mask = (hgd->opacity_has_alpha) ? opacity_val.w : opacity_val.x;
        alpha *= mask; 
    }
    if (alpha < 0.1f) alpha = 0.0f;
    if (alpha < 1.0f) {
        float3 ray_origin = optixGetWorldRayOrigin();
        float3 ray_dir = optixGetWorldRayDirection();
        uint3 launch_idx = optixGetLaunchIndex();
        float seed = ray_origin.x * 73.0f + ray_origin.y * 19.0f + ray_origin.z * 43.0f +
                     ray_dir.x * 37.0f + ray_dir.y * 11.0f + ray_dir.z * 5.0f +
                     (float)launch_idx.x * 0.1f + (float)launch_idx.y * 0.1f +
                     (float)optixLaunchParams.frame_number * 13.0f;
        float rnd = fmodf(fabsf(sinf(seed) * 43758.5453f), 1.0f);
        if (rnd > alpha) {
            optixIgnoreIntersection();
        }
    }
}

// Shadow Closest Hit
extern "C" __global__ void __closesthit__shadow() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* payload = unpackPayload<OptixHitResult>(p0, p1);
    payload->hit = 1;
}

// Hair Closest Hit
extern "C" __global__ void __closesthit__hair() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* payload = unpackPayload<OptixHitResult>(p0, p1);
    const HitGroupData* hgd = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    unsigned int prim_idx = optixGetPrimitiveIndex();
    float t = optixGetRayTmax();
    float3 ray_origin = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = ray_origin + ray_dir * t;
    float3 wo = -ray_dir;
    float3 tangent;
    if (hgd->tangents != nullptr) {
        tangent = optixTransformVectorFromObjectToWorldSpace(hgd->tangents[prim_idx]);
    } else {
        tangent = optixTransformVectorFromObjectToWorldSpace(make_float3(0.0f, 1.0f, 0.0f));
    }
    tangent = normalize(tangent);
    float3 to_hit = hit_point - ray_origin;
    float3 plane_n = normalize(cross(tangent, to_hit));
    float3 normal = normalize(cross(plane_n, tangent));
    float3 wo_perp = (wo - dot(wo, tangent) * tangent);
    float3 bitangent = cross(tangent, normalize(wo_perp));
    float h = clamp(dot(normal, bitangent), -1.0f, 1.0f);
    const GpuHairMaterial& h_mat = hgd->hair_material;
    int hair_color_mode = h_mat.colorMode;
    float3 hair_color = h_mat.color;
    float roughness = h_mat.roughness;
    float ior = h_mat.ior;
    float alpha_rad = h_mat.cuticleAngle; 
    float3 sigma_a = h_mat.sigma_a;

    int base_mat_id = (hair_color_mode == 3 && hgd->mesh_material_id >= 0) ? hgd->mesh_material_id : hgd->material_id;

    if (base_mat_id >= 0 && base_mat_id < optixLaunchParams.material_count) {
        GpuMaterial mat = optixLaunchParams.materials[base_mat_id];
        if (hair_color_mode == 3 && hair_color.x + hair_color.y + hair_color.z < 0.001f) {
            hair_color = mat.albedo;
            roughness = mat.roughness;
            ior = mat.ior;
        }
    }
    if (hair_color_mode == 3 && hgd->has_root_uvs && hgd->root_uvs != nullptr) {
        float2 root_uv = hgd->root_uvs[prim_idx];
        if (h_mat.albedo_tex != 0) {
            float4 tex_val = tex2D<float4>(h_mat.albedo_tex, root_uv.x, root_uv.y);
            hair_color = make_float3(fmaxf(0.001f, tex_val.x), fmaxf(0.001f, tex_val.y), fmaxf(0.001f, tex_val.z));
        } else if (hgd->has_albedo_tex && hgd->albedo_tex != 0) {
            float4 tex_val = tex2D<float4>(hgd->albedo_tex, root_uv.x, root_uv.y);
            hair_color = make_float3(fmaxf(0.001f, tex_val.x), fmaxf(0.001f, tex_val.y), fmaxf(0.001f, tex_val.z));
        }
        if (h_mat.roughness_tex != 0) {
            float4 rough_val = tex2D<float4>(h_mat.roughness_tex, root_uv.x, root_uv.y);
            roughness *= rough_val.x; 
        } else if (hgd->has_roughness_tex && hgd->roughness_tex != 0) {
            float4 rough_val = tex2D<float4>(hgd->roughness_tex, root_uv.x, root_uv.y);
            roughness *= rough_val.x;
        }
        float3 c_clamped = make_float3(fmaxf(0.001f, fminf(0.99f, hair_color.x)), fmaxf(0.001f, fminf(0.99f, hair_color.y)), fmaxf(0.001f, fminf(0.99f, hair_color.z)));
        sigma_a = make_float3(-logf(c_clamped.x) * 2.5f, -logf(c_clamped.y) * 2.5f, -logf(c_clamped.z) * 2.5f);
    }
    if (h_mat.tint > 0.001f) {
        sigma_a.x += -logf(fmaxf(0.001f, h_mat.tintColor.x)) * h_mat.tint;
        sigma_a.y += -logf(fmaxf(0.001f, h_mat.tintColor.y)) * h_mat.tint;
        sigma_a.z += -logf(fmaxf(0.001f, h_mat.tintColor.z)) * h_mat.tint;
    }
    if (hgd->strand_ids != nullptr) {
        uint32_t strandID = hgd->strand_ids[prim_idx];
        uint32_t h_hash = strandID * 747796405u + 2891336453u;
        h_hash = ((h_hash >> ((h_hash >> 28u) + 4u)) ^ h_hash) * 277803737u;
        float r_rand = ((h_hash >> 22u) ^ h_hash) / 4294967296.0f; 
        uint32_t h2 = strandID * 123456789u + 987654321u;
        float r2 = (h2 & 0x00FFFFFF) / 16777216.0f;
        if (h_mat.randomHue > 0.001f || h_mat.randomValue > 0.001f) {
            if (hair_color_mode == 1) {
                float m = clamp(h_mat.melanin + (r2 - 0.5f) * h_mat.randomValue, 0.0f, 1.0f);
                float red = clamp(h_mat.melaninRedness + (r_rand - 0.5f) * h_mat.randomHue, 0.0f, 1.0f);
                sigma_a = HairGPU::melanin_to_absorption(m, red);
            } else {
                float3 hsv = rgb_to_hsv(hair_color);
                hsv.x += (r_rand - 0.5f) * h_mat.randomHue;
                if (hsv.x < 0.0f) hsv.x += 1.0f;
                if (hsv.x > 1.0f) hsv.x -= 1.0f;
                hsv.z += (r2 - 0.5f) * h_mat.randomValue;
                hsv.z = fmaxf(0.001f, fminf(1.0f, hsv.z));
                hair_color = hsv_to_rgb(hsv);
                float3 c_clamped = make_float3(fmaxf(0.001f, fminf(0.99f, hair_color.x)), fmaxf(0.001f, fminf(0.99f, hair_color.y)), fmaxf(0.001f, fminf(0.99f, hair_color.z)));
                sigma_a = make_float3(-logf(c_clamped.x) * 2.5f, -logf(c_clamped.y) * 2.5f, -logf(c_clamped.z) * 2.5f);
            }
        }
    }
    HairGPU::GpuHairMaterial hair_mat_final;
    hair_mat_final.sigma_a = sigma_a;
    hair_mat_final.roughness = fmaxf(0.01f, roughness); 
    hair_mat_final.radialRoughness = fmaxf(0.01f, h_mat.radialRoughness);
    hair_mat_final.cuticleAngle = alpha_rad; 
    hair_mat_final.ior = ior;
    hair_mat_final.emission = h_mat.emission;
    hair_mat_final.emissionStrength = h_mat.emissionStrength;
    hair_mat_final.v_R = h_mat.v_R;
    hair_mat_final.v_TT = h_mat.v_TT;
    hair_mat_final.v_TRT = h_mat.v_TRT;
    hair_mat_final.s = h_mat.s;
    if (hgd->material_id >= 0 && hgd->material_id < optixLaunchParams.material_count) {
        GpuMaterial mat = optixLaunchParams.materials[hgd->material_id];
        if (mat.emission.x + mat.emission.y + mat.emission.z > 0.001f) {
            hair_mat_final.emission += mat.emission;
            hair_mat_final.emissionStrength = 1.0f; 
        }
    }
    float3 result_color = make_float3(0.0f);
    int light_count = optixLaunchParams.light_count;
    LightGPU* lights = optixLaunchParams.lights;
    for (int li = 0; li < light_count && li < 16; ++li) {
        LightGPU light = lights[li];
        float3 light_dir;
        float3 Li;
        float light_dist = 1e6f;
        if (light.type == 0) {
            float3 to_light = light.position - hit_point;
            light_dist = length(to_light);
            light_dir = to_light / light_dist;
            float atten = 1.0f / (light_dist * light_dist + 1e-4f);
            Li = light.color * light.intensity * atten;
        } else {
            light_dir = normalize(light.direction);
            Li = light.color * light.intensity;
        }
        OptixHitResult shadow_payload;
        shadow_payload.hit = 0;
        unsigned int sp0, sp1;
        packPayload(&shadow_payload, sp0, sp1);
        float3 shadow_origin = hit_point + normal * 0.001f; 
        optixTrace(optixLaunchParams.handle, shadow_origin, light_dir, 0.001f, light_dist - 0.001f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT, SHADOW_RAY_TYPE, RAY_TYPE_COUNT, SHADOW_RAY_TYPE, sp0, sp1);
        if (shadow_payload.hit == 0) {
            float3 bsdf = HairGPU::hair_bsdf_eval(wo, light_dir, tangent, hair_mat_final, h);
            result_color += bsdf * Li; 
        }
    }
    {
        float3 sky_dir = normalize(normal + make_float3(0, 1, 0));
        float3 sky_color = evaluate_background(optixLaunchParams.world, hit_point, sky_dir, nullptr);
        float3 ambient_bsdf = hair_color * (1.0f / M_PIf); 
        result_color += ambient_bsdf * sky_color * 0.15f; 
    }
    result_color += make_float3(0.01f) * hair_color;
    payload->hit = 1;
    payload->t = t;
    payload->position = hit_point;
    payload->normal = normal;
    payload->color = result_color;
    payload->emission = hair_mat_final.emission * hair_mat_final.emissionStrength;
    payload->albedo = hair_color;
    payload->roughness = roughness;
    payload->is_hair = 1;
    payload->is_volumetric = 0;
    payload->material_id = hgd->material_id;
    payload->object_id = hgd->object_id;
}
