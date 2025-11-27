#pragma once
#include "ray.h"
#include "params.h"

__device__ float2 random_in_unit_polygon(int sides, curandState* rng) {
    float step = 2.0f * M_PIf / float(sides);

    // Rastgele bir kenar seç (köşe indeksi)
    int edge_index = curand(rng) % sides;
    float angle1 = step * edge_index;
    float angle2 = angle1 + step;

    // Bu kenar boyunca interpolasyon için rastgele t
    float t = curand_uniform(rng);

    float x1 = cosf(angle1);
    float y1 = sinf(angle1);
    float x2 = cosf(angle2);
    float y2 = sinf(angle2);

    float px = x1 * (1.0f - t) + x2 * t;
    float py = y1 * (1.0f - t) + y2 * t;

    // Daire içine eşit doluluk için shrink faktörü (radyal içe çekme)
    float shrink = sqrtf(curand_uniform(rng)); // 0-1 arası, merkeze eşit doluluk
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
