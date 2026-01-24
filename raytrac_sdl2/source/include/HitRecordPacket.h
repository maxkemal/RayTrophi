#pragma once

#include "Vec3SIMD.h"
#include <limits>

// 8-Wide Hit Record for AVX2 Packet Tracing
struct HitRecordPacket {
    // Distance (t) - Init to Infinity
    Vec3SIMD t;
    
    // Hit Point
    Vec3SIMD p_x;
    Vec3SIMD p_y;
    Vec3SIMD p_z;

    // Normal
    Vec3SIMD normal_x;
    Vec3SIMD normal_y;
    Vec3SIMD normal_z;

    // Additional Normals
    Vec3SIMD neighbor_normal_x;
    Vec3SIMD neighbor_normal_y;
    Vec3SIMD neighbor_normal_z;
    __m256 has_neighbor_normal;

    Vec3SIMD interpolated_normal_x;
    Vec3SIMD interpolated_normal_y;
    Vec3SIMD interpolated_normal_z;

    Vec3SIMD face_normal_x;
    Vec3SIMD face_normal_y;
    Vec3SIMD face_normal_z;

    __m256 front_face;

    // Tangents
    Vec3SIMD tangent_x;
    Vec3SIMD tangent_y;
    Vec3SIMD tangent_z;
    Vec3SIMD bitangent_x;
    Vec3SIMD bitangent_y;
    Vec3SIMD bitangent_z;
    __m256 has_tangent;

    // Texture Coordinates
    Vec3SIMD u;
    Vec3SIMD v;

    // Material ID (Integer SIMD)
    __m256i mat_id;

    // Mask for active lanes (All 1s = active, All 0s = inactive)
    __m256 mask;

    // Flags
    __m256 is_instance_hit;

    HitRecordPacket() {
        t = Vec3SIMD(std::numeric_limits<float>::infinity());
        mask = _mm256_setzero_ps();
        has_neighbor_normal = _mm256_setzero_ps();
        has_tangent = _mm256_setzero_ps();
        front_face = _mm256_setzero_ps();
        is_instance_hit = _mm256_setzero_ps();
        
        for(int i=0; i<8; i++) {
            vdb_volume[i] = nullptr;
            gas_volume[i] = nullptr;
        }
    }

    // Merge a new hit if it is closer (t < current.t) AND active
    inline void merge_if_closer(const HitRecordPacket& other, const __m256& active_mask) {
        __m256 closer_mask = _mm256_cmp_ps(other.t.data, t.data, _CMP_LT_OQ);
        __m256 update_mask = _mm256_and_ps(closer_mask, active_mask);
        
        t.data = _mm256_blendv_ps(t.data, other.t.data, update_mask);
        
        p_x.data = _mm256_blendv_ps(p_x.data, other.p_x.data, update_mask);
        p_y.data = _mm256_blendv_ps(p_y.data, other.p_y.data, update_mask);
        p_z.data = _mm256_blendv_ps(p_z.data, other.p_z.data, update_mask);
        
        normal_x.data = _mm256_blendv_ps(normal_x.data, other.normal_x.data, update_mask);
        normal_y.data = _mm256_blendv_ps(normal_y.data, other.normal_y.data, update_mask);
        normal_z.data = _mm256_blendv_ps(normal_z.data, other.normal_z.data, update_mask);

        neighbor_normal_x.data = _mm256_blendv_ps(neighbor_normal_x.data, other.neighbor_normal_x.data, update_mask);
        neighbor_normal_y.data = _mm256_blendv_ps(neighbor_normal_y.data, other.neighbor_normal_y.data, update_mask);
        neighbor_normal_z.data = _mm256_blendv_ps(neighbor_normal_z.data, other.neighbor_normal_z.data, update_mask);
        has_neighbor_normal = _mm256_blendv_ps(has_neighbor_normal, other.has_neighbor_normal, update_mask);

        interpolated_normal_x.data = _mm256_blendv_ps(interpolated_normal_x.data, other.interpolated_normal_x.data, update_mask);
        interpolated_normal_y.data = _mm256_blendv_ps(interpolated_normal_y.data, other.interpolated_normal_y.data, update_mask);
        interpolated_normal_z.data = _mm256_blendv_ps(interpolated_normal_z.data, other.interpolated_normal_z.data, update_mask);

        face_normal_x.data = _mm256_blendv_ps(face_normal_x.data, other.face_normal_x.data, update_mask);
        face_normal_y.data = _mm256_blendv_ps(face_normal_y.data, other.face_normal_y.data, update_mask);
        face_normal_z.data = _mm256_blendv_ps(face_normal_z.data, other.face_normal_z.data, update_mask);
        front_face = _mm256_blendv_ps(front_face, other.front_face, update_mask);

        tangent_x.data = _mm256_blendv_ps(tangent_x.data, other.tangent_x.data, update_mask);
        tangent_y.data = _mm256_blendv_ps(tangent_y.data, other.tangent_y.data, update_mask);
        tangent_z.data = _mm256_blendv_ps(tangent_z.data, other.tangent_z.data, update_mask);
        bitangent_x.data = _mm256_blendv_ps(bitangent_x.data, other.bitangent_x.data, update_mask);
        bitangent_y.data = _mm256_blendv_ps(bitangent_y.data, other.bitangent_y.data, update_mask);
        bitangent_z.data = _mm256_blendv_ps(bitangent_z.data, other.bitangent_z.data, update_mask);
        has_tangent = _mm256_blendv_ps(has_tangent, other.has_tangent, update_mask);
        
        u.data = _mm256_blendv_ps(u.data, other.u.data, update_mask);
        v.data = _mm256_blendv_ps(v.data, other.v.data, update_mask);
        
        mat_id = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(mat_id), 
            _mm256_castsi256_ps(other.mat_id), 
            update_mask
        ));

        is_instance_hit = _mm256_blendv_ps(is_instance_hit, other.is_instance_hit, update_mask);

        // Update Pointers
        alignas(32) int update_mask_bits[8];
        _mm256_store_si256((__m256i*)update_mask_bits, _mm256_castps_si256(update_mask));
        for(int i=0; i<8; i++) {
            if (update_mask_bits[i]) {
                vdb_volume[i] = other.vdb_volume[i];
                gas_volume[i] = other.gas_volume[i];
            }
        }

        mask = _mm256_or_ps(mask, update_mask);
    }

    const class VDBVolume* vdb_volume[8] = {nullptr};
    const class GasVolume* gas_volume[8] = {nullptr};

};
