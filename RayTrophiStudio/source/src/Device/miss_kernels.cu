#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "params.h"
#include "payload.h"
#include "sky_model.cuh"
#include "vec3_utils.cuh"

extern "C" __constant__ RayGenParams optixLaunchParams;

extern "C" __global__ void __miss__ms() {
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    OptixHitResult* outColor = unpackPayload<OptixHitResult>(p0, p1);
    
    float3 current_bg = optixLaunchParams.background_color;

    if (optixLaunchParams.world.mode == 2) { // WORLD_MODE_NISHITA
         float3 ray_dir = optixGetWorldRayDirection();
         current_bg = calculate_nishita_sky_gpu(ray_dir, optixLaunchParams.world.nishita);
    } 
    
    outColor->emission = current_bg;

    // Grid Shader
    float3 ray_dir = optixGetWorldRayDirection();
    float3 ray_origin = optixGetWorldRayOrigin();

    if (optixLaunchParams.grid_enabled && ray_dir.y < -1e-4f) {
        float t = -ray_origin.y / ray_dir.y;
        if (t > 0.0f) {
             float3 p = ray_origin + ray_dir * t;
             float fade_start = 100.0f;
             float fade_end = optixLaunchParams.grid_fade_distance;
             if (fade_end < fade_start) fade_end = fade_start + 100.0f;
             
             float dist = t;
             float alpha_fade = 1.0f - fminf(fmaxf((dist - fade_start) / (fade_end - fade_start), 0.0f), 1.0f);
             
             if (alpha_fade > 0.0f) {
                  float scale_primary = 10.0f;
                  float scale_secondary = 1.0f;
                  float line_width_base = 0.02f;
                  float line_width = line_width_base * (1.0f + dist * 0.02f);
                  
                  float x_mod_p = fabsf(fmodf(p.x, scale_primary));
                  float z_mod_p = fabsf(fmodf(p.z, scale_primary));
                  float x_mod_s = fabsf(fmodf(p.x, scale_secondary));
                  float z_mod_s = fabsf(fmodf(p.z, scale_secondary));
                  
                  bool x_line_p = x_mod_p < line_width || x_mod_p > (scale_primary - line_width);
                  bool z_line_p = z_mod_p < line_width || z_mod_p > (scale_primary - line_width);
                  bool x_line_s = x_mod_s < line_width || x_mod_s > (scale_secondary - line_width);
                  bool z_line_s = z_mod_s < line_width || z_mod_s > (scale_secondary - line_width);
                  
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
                      outColor->emission = current_bg * (1.0f - final_alpha) + grid_col * final_alpha;
                  }
             }
        }
    }
    outColor->hit = 0;
}

extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(0); // Shadow miss = not in shadow
}
