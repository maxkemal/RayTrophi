#pragma once
#include "ray.h"
#include "params.h"



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
    
    // Interpolated positions
    float3 origin = cam.origin + cam.vel_origin * time_offset;
    float3 corner = cam.lower_left_corner + cam.vel_corner * time_offset;
    float3 horizontal = cam.horizontal + cam.vel_horizontal * time_offset;
    float3 vertical = cam.vertical + cam.vel_vertical * time_offset;
    
    // Pinhole or DOF logic
    if (cam.lens_radius <= 0.0f) {
        float3 dir = corner + s * horizontal + t * vertical - origin;
        return Ray(origin, normalize(dir));
    }

    // DOF with Polygonal Bokeh
    float2 rd = cam.lens_radius * random_in_unit_polygon(cam.blade_count, rng); 
    float3 offset = cam.u * rd.x + cam.v * rd.y; // Approximation: Lens plane orientation constant

    float3 ray_origin = origin + offset;
    float3 ray_direction = corner + s * horizontal + t * vertical - origin - offset;

    return Ray(ray_origin, normalize(ray_direction));
}
