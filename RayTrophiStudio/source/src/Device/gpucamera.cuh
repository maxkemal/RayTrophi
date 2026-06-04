#pragma once
#include "ray.h"
#include "params.h"

// ═══════════════════════════════════════════════════════════════════════════════
// CHROMATIC ABERRATION HELPER
// Calculates UV offset for each color channel based on radial distance from center
// Red bends outward, Blue bends inward (typical lateral CA pattern)
// ═══════════════════════════════════════════════════════════════════════════════
__device__ inline float2 apply_chromatic_aberration(float s, float t, float ca_amount, float channel_scale) {
    // Center-relative coordinates
    float u_centered = s - 0.5f;
    float v_centered = t - 0.5f;
    
    // Radial distance squared from center
    float r2 = u_centered * u_centered + v_centered * v_centered;
    
    // CA scales with r² (stronger at edges, zero at center)
    // channel_scale: R > 1.0, G = 1.0, B < 1.0 (or vice versa)
    float offset = 1.0f + ca_amount * r2 * (channel_scale - 1.0f);
    
    // Apply offset to UV
    float new_s = u_centered * offset + 0.5f;
    float new_t = v_centered * offset + 0.5f;
    
    return make_float2(new_s, new_t);
}





__device__ float2 random_in_unit_polygon(int sides, curandState* rng) {
    if (sides < 3) return random_in_unit_disk(rng);

    // Uniform sampling within a regular polygon
    // Strategy: Sample a triangle wedge and rotate it
    float one_wedge_angle = 2.0f * M_PIf / float(sides);
    int wedge_index = curand(rng) % sides;
    
    // Sample triangle (0,0) - (1,0) - (cos(angle), sin(angle)) ?
    // Simpler: Sample in unit disk, map to polygon edge? 
    // Rejection sampling is easiest but slow.
    // Analytical mapping:
    
    // 1. Polar coordinates for polygon
    // r_max at angle theta = R0 / cos(theta - theta_center) where R0 is apothem?
    // Let's stick to the existing interpolation logic but safer
    
    float angle1 = one_wedge_angle * wedge_index;
    float angle2 = angle1 + one_wedge_angle;
    
    // Random point on the edge segment
    float t = curand_uniform(rng);
    float x_edge = cosf(angle1) * (1.0f - t) + cosf(angle2) * t;
    float y_edge = sinf(angle1) * (1.0f - t) + sinf(angle2) * t;
    
    // Scale towards center for uniform distribution
    float r_scale = sqrtf(curand_uniform(rng));
    
    return make_float2(x_edge * r_scale, y_edge * r_scale);
}

__device__ Ray get_ray_from_camera(const gpuCamera& cam, float s, float t, curandState* rng) {
    // Motion Blur: Interpolate camera vectors over shutter interval
    float time_offset = 0.0f;
    if (cam.motion_blur_enabled) {
        time_offset = curand_uniform(rng);
    }
    
    // Interpolated positions (with motion blur)
    float3 origin = cam.origin + cam.vel_origin * time_offset;
    float3 corner = cam.lower_left_corner + cam.vel_corner * time_offset;
    float3 horizontal = cam.horizontal + cam.vel_horizontal * time_offset;
    float3 vertical = cam.vertical + cam.vel_vertical * time_offset;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CAMERA SHAKE (Cinema Mode - Handheld Simulation)
    // Uses pre-calculated shake offset/rotation from CPU
    // ═══════════════════════════════════════════════════════════════════════════
    if (cam.shake_enabled) {
        // Apply position shake to origin
        origin = origin + cam.shake_offset;
        
        // Apply rotation shake to viewing direction
        float3 shake_rot = cam.shake_rotation;
        
        // Shift the image plane based on rotation shake
        float3 center = corner + horizontal * 0.5f + vertical * 0.5f;
        center = center + cam.u * shake_rot.y * 10.0f;  // Yaw
        center = center + cam.v * shake_rot.x * 10.0f;  // Pitch
        
        // Recalculate corner from shifted center
        corner = center - horizontal * 0.5f - vertical * 0.5f;
    }


    
    // ---------------- LENS DISTORTION ----------------
    // Brown-Conrady Model (Simplified Radial Distortion)
    // k < 0: Barrel Distortion (Wide Angle)
    // k > 0: Pincushion Distortion (Telephoto)
    float use_s = s;
    float use_t = t;
    
    if (fabsf(cam.distortion) > 0.001f) {
        // Calculate aspect ratio from vector lengths
        float h_len_sq = horizontal.x*horizontal.x + horizontal.y*horizontal.y + horizontal.z*horizontal.z;
        float v_len_sq = vertical.x*vertical.x + vertical.y*vertical.y + vertical.z*vertical.z;
        float aspect = sqrtf(h_len_sq) / (sqrtf(v_len_sq) + 1e-6f);
        
        // Convert to centered coords [-0.5*aspect, 0.5*aspect] x [-0.5, 0.5]
        float u_centered = (s - 0.5f) * aspect;
        float v_centered = (t - 0.5f);
        
        float r2 = u_centered * u_centered + v_centered * v_centered;
        
        // Distortion Term: (1 + k * r^2)
        // Note: For real lenses, this usually applies to undistorted->distorted.
        // Since we are tracing RAys (Camera -> Scene), we are doing the inverse mapping?
        // Actually, straightforward mapping works visually for raytracing:
        // We modify the pixel coordinate we look through.
        float factor = 1.0f + cam.distortion * r2;
        
        // Apply distortion
        u_centered *= factor;
        v_centered *= factor;
        
        // Map back to [0,1]
        use_s = (u_centered / aspect) + 0.5f;
        use_t = v_centered + 0.5f;
    }
    
    // Pinhole or DOF logic
    if (cam.lens_radius <= 0.0f) {
        float3 dir = corner + use_s * horizontal + use_t * vertical - origin;
        return Ray(origin, normalize(dir));
    }
 
    // DOF with Polygonal Bokeh
    float2 rd = cam.lens_radius * random_in_unit_polygon(cam.blade_count, rng); 
    float3 offset = cam.u * rd.x + cam.v * rd.y; // Approximation: Lens plane orientation constant
 
    float3 ray_origin = origin + offset;
    float3 ray_direction = corner + use_s * horizontal + use_t * vertical - origin - offset;
 
    return Ray(ray_origin, normalize(ray_direction));
}
