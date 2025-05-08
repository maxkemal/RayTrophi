#pragma once
#include "ray.h"
#include "params.h"
__device__ float2 random_in_unit_polygon(int sides, curandState* rng) {
    float step = 2.0f * M_PIf / float(sides);

    int edge_index = curand(rng) % sides;
    float edge_angle = step * edge_index;

    float t = curand_uniform(rng);
    float x1 = cosf(edge_angle);
    float y1 = sinf(edge_angle);

    float x2 = cosf(edge_angle + step);
    float y2 = sinf(edge_angle + step);

    float px = x1 * (1.0f - t) + x2 * t;
    float py = y1 * (1.0f - t) + y2 * t;

    float shrink = sqrtf(curand_uniform(rng)); // merkezden dolu dağılım
    px *= shrink;
    py *= shrink;

    return make_float2(px, py);
}

__device__ Ray get_ray_from_camera(const gpuCamera& cam, float s, float t, curandState* rng) {
    // Eğer aperture yoksa (f:0 gibi), direkt düz ray atalım.
    if (cam.lens_radius <= 0.0f) {
        float3 dir = cam.lower_left_corner + s * cam.horizontal + t * cam.vertical - cam.origin;
        return Ray(cam.origin, normalize(dir));
    }

    // DOF aktifse:
    float2 rd = cam.lens_radius * random_in_unit_polygon(cam.blade_count, rng); // rastgele polygon sample
    float3 offset = cam.u * rd.x + cam.v * rd.y;

    float3 ray_origin = cam.origin + offset;
    float3 ray_direction = cam.lower_left_corner + s * cam.horizontal + t * cam.vertical - cam.origin - offset;

    return Ray(ray_origin, normalize(ray_direction));
}
