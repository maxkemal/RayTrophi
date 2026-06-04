#pragma once

#include "Vec3SIMD.h"
#include "Ray.h"
#include <vector>
#include <array>

// 8-Wide Ray Packet for AVX2 Ray Tracing (SoA Layout)
struct RayPacket {
    // Orbit (Origin)
    Vec3SIMD orig_x;
    Vec3SIMD orig_y;
    Vec3SIMD orig_z;

    // Direction
    Vec3SIMD dir_x;
    Vec3SIMD dir_y;
    Vec3SIMD dir_z;

    // Precomputed Inverse Direction (for fast AABB intersection)
    Vec3SIMD inv_dir_x;
    Vec3SIMD inv_dir_y;
    Vec3SIMD inv_dir_z;

    // Sign bits for direction (1 = negative, 0 = positive)
    // Used for selecting AABB min/max bounds without branching
    __m256i sign_x;
    __m256i sign_y;
    __m256i sign_z;

    RayPacket() = default;

    // Gather constructor: Packs 8 separate Rays into one SIMD Packet
    // If rays.size() < 8, padding is handled (usually with dummy rays or careful internal logic)
    RayPacket(const std::vector<Ray>& rays, int start_index) {
        alignas(32) float ox[8], oy[8], oz[8];
        alignas(32) float dx[8], dy[8], dz[8];

        for (int i = 0; i < 8; ++i) {
            // Safety check for boundary
            if (start_index + i < rays.size()) {
                const Ray& r = rays[start_index + i];
                ox[i] = r.origin.x;
                oy[i] = r.origin.y;
                oz[i] = r.origin.z;
                dx[i] = r.direction.x;
                dy[i] = r.direction.y;
                dz[i] = r.direction.z;
            } else {
                // Pad with zeros (or safe values)
                ox[i] = oy[i] = oz[i] = 0.0f;
                dx[i] = dy[i] = dz[i] = 0.0f; // direction 0 might cause NaNs, handle with care? 
                // Using 0 is technically risky for inv_dir, but 
                // in practice we mask out inactive lanes anyway.
            }
        }

        orig_x = Vec3SIMD(ox);
        orig_y = Vec3SIMD(oy);
        orig_z = Vec3SIMD(oz);

        dir_x = Vec3SIMD(dx);
        dir_y = Vec3SIMD(dy);
        dir_z = Vec3SIMD(dz);

        // Precompute Inverse Direction and Signs
        // Note: Div by zero handled by IEEE 754 (Infinity), typically fine for AABB unless NaN.
        // We can use a safe small epsilon or handling.
        
        // Fast approx reciprocal or standard div? div is safer.
        inv_dir_x = Vec3SIMD(1.0f) / dir_x;
        inv_dir_y = Vec3SIMD(1.0f) / dir_y;
        inv_dir_z = Vec3SIMD(1.0f) / dir_z;

        // Sign logic: 1 if dir < 0, else 0
        // Use IEEE float sign bit extraction or comparison
        // _mm256_srai_epi32 shifts arithmetic, preserving sign.
        // But cleaner is to check < 0.
        
        // Using cast to int for sign bit check (MSB)
        sign_x = _mm256_srli_epi32(_mm256_castps_si256(inv_dir_x.data), 31);
        sign_y = _mm256_srli_epi32(_mm256_castps_si256(inv_dir_y.data), 31);
        sign_z = _mm256_srli_epi32(_mm256_castps_si256(inv_dir_z.data), 31);
    }

    // Refresh derived data (inv_dir, signs) after manual dir modification
    void update_derived_data() {
        inv_dir_x = Vec3SIMD(1.0f) / dir_x;
        inv_dir_y = Vec3SIMD(1.0f) / dir_y;
        inv_dir_z = Vec3SIMD(1.0f) / dir_z;
        sign_x = _mm256_srli_epi32(_mm256_castps_si256(inv_dir_x.data), 31);
        sign_y = _mm256_srli_epi32(_mm256_castps_si256(inv_dir_y.data), 31);
        sign_z = _mm256_srli_epi32(_mm256_castps_si256(inv_dir_z.data), 31);
    }
    
    // Position at t (component-wise)
    void point_at(const Vec3SIMD& t, Vec3SIMD& out_x, Vec3SIMD& out_y, Vec3SIMD& out_z) const {
        out_x = orig_x + t * dir_x;
        out_y = orig_y + t * dir_y;
        out_z = orig_z + t * dir_z;
    }
};
