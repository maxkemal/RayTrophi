#pragma once

// Per-step cached inverse transforms and SDF/OBB bounds for CPU particle
// collision loops so Effect systems (confetti, sparkles, ghosts) do not
// recompute Matrix4x4::inverse() per particle per collider.
// Private helper for ParticleSimulation.cpp (incremental-build safe).

#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace RayTrophiSim {

struct CachedColliderTransform {
    bool valid = false;
    Matrix4x4 world_to_local = Matrix4x4::identity();
    Vec3 scale = Vec3(1.0f, 1.0f, 1.0f);
    float cooked_avg_scale = 1.0f;
    float avg_scale = 1.0f;
    float world_thick = 0.0f;
    float local_thick = 0.0f;
    Vec3 sdf_origin_scaled = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 sdf_extents_scaled = Vec3(0.0f, 0.0f, 0.0f);
    float step_x = 0.0f;
    float step_y = 0.0f;
    float step_z = 0.0f;
};

template <typename ResolvedT>
inline void prepareColliderTransforms(const std::vector<ResolvedT>& resolved,
                                      float particle_radius,
                                      std::vector<CachedColliderTransform>& out) {
    out.resize(resolved.size());
    for (std::size_t i = 0; i < resolved.size(); ++i) {
        const auto& rc = resolved[i];
        CachedColliderTransform& cache = out[i];
        const auto mode = rc.desc.source_mode;
        if (!rc.has_obb ||
            (mode != ParticleColliderSourceMode::ObjectOBB &&
             mode != ParticleColliderSourceMode::ObjectMeshSDF)) {
            cache.valid = false;
            continue;
        }
        cache.valid = true;
        cache.world_to_local = rc.obb.local_to_world.inverse();
        if (mode != ParticleColliderSourceMode::ObjectMeshSDF) {
            continue;
        }
        const Vec3 cooked_size = rc.desc.sdf_mesh_size;
        const Vec3 current_size = rc.obb.local_bounds_max - rc.obb.local_bounds_min;
        Vec3 scale(
            cooked_size.x > 1e-6f ? current_size.x / cooked_size.x : 1.0f,
            cooked_size.y > 1e-6f ? current_size.y / cooked_size.y : 1.0f,
            cooked_size.z > 1e-6f ? current_size.z / cooked_size.z : 1.0f
        );
        scale.x = std::max(1e-4f, scale.x);
        scale.y = std::max(1e-4f, scale.y);
        scale.z = std::max(1e-4f, scale.z);
        cache.scale = scale;
        cache.cooked_avg_scale = (scale.x + scale.y + scale.z) / 3.0f;

        const Matrix4x4& m = rc.obb.local_to_world;
        const float sx = Vec3(m.m[0][0], m.m[1][0], m.m[2][0]).length();
        const float sy = Vec3(m.m[0][1], m.m[1][1], m.m[2][1]).length();
        const float sz = Vec3(m.m[0][2], m.m[1][2], m.m[2][2]).length();
        float avg_scale = (sx + sy + sz) / 3.0f;
        if (avg_scale <= 1e-6f) avg_scale = 1.0f;
        cache.avg_scale = avg_scale;
        cache.world_thick = std::max(0.0f, rc.desc.thickness) + particle_radius;
        cache.local_thick = cache.world_thick / avg_scale;
        cache.sdf_origin_scaled = rc.desc.sdf_origin * scale;
        cache.sdf_extents_scaled = rc.desc.sdf_extents * scale;
        cache.step_x = cache.sdf_extents_scaled.x /
            std::max(1.0f, static_cast<float>(rc.desc.sdf_nx - 1));
        cache.step_y = cache.sdf_extents_scaled.y /
            std::max(1.0f, static_cast<float>(rc.desc.sdf_ny - 1));
        cache.step_z = cache.sdf_extents_scaled.z /
            std::max(1.0f, static_cast<float>(rc.desc.sdf_nz - 1));
    }
}

template <typename ResolveAabbFn>
inline void resolveOBBCollisionCached(const ParticleColliderDesc& collider,
                                      const ParticleColliderOBB& obb,
                                      const CachedColliderTransform& cache,
                                      Vec3& position,
                                      Vec3& velocity,
                                      ResolveAabbFn&& resolve_aabb) {
    Vec3 local_position = cache.world_to_local.transform_point(position);
    Vec3 local_velocity = cache.world_to_local.transform_vector(velocity);

    ParticleColliderDesc local_collider = collider;
    local_collider.bounds_min = obb.local_bounds_min;
    local_collider.bounds_max = obb.local_bounds_max;
    const Vec3 local_position_before = local_position;
    const Vec3 local_velocity_before = local_velocity;
    resolve_aabb(local_collider, local_position, local_velocity);

    if (local_position.x == local_position_before.x &&
        local_position.y == local_position_before.y &&
        local_position.z == local_position_before.z &&
        local_velocity.x == local_velocity_before.x &&
        local_velocity.y == local_velocity_before.y &&
        local_velocity.z == local_velocity_before.z) {
        return;
    }
    position = obb.local_to_world.transform_point(local_position);
    velocity = obb.local_to_world.transform_vector(local_velocity);
}

template <typename ResolveSweptAabbFn>
inline bool resolveSweptOBBCollisionCached(const ParticleColliderDesc& collider,
                                           const ParticleColliderOBB& obb,
                                           const CachedColliderTransform& cache,
                                           const Vec3& previous_position,
                                           Vec3& position,
                                           Vec3& velocity,
                                           ResolveSweptAabbFn&& resolve_swept_aabb) {
    const Vec3 local_previous = cache.world_to_local.transform_point(previous_position);
    Vec3 local_position = cache.world_to_local.transform_point(position);
    Vec3 local_velocity = cache.world_to_local.transform_vector(velocity);

    ParticleColliderDesc local_collider = collider;
    local_collider.bounds_min = obb.local_bounds_min;
    local_collider.bounds_max = obb.local_bounds_max;
    if (!resolve_swept_aabb(local_collider, local_previous, local_position, local_velocity)) {
        return false;
    }

    position = obb.local_to_world.transform_point(local_position);
    velocity = obb.local_to_world.transform_vector(local_velocity);
    return true;
}

template <typename ResolveVelocityFn>
inline void resolveSDFCollisionCached(const ParticleColliderDesc& collider,
                                      const ParticleColliderOBB& obb,
                                      const CachedColliderTransform& cache,
                                      Vec3& position,
                                      Vec3& velocity,
                                      ResolveVelocityFn&& resolve_vel) {
    if (!collider.sdf_grid_data || collider.sdf_grid_data->empty()) return;

    const Vec3 local_pos = cache.world_to_local.transform_point(position);
    const Vec3 sdf_local_pos = local_pos - cache.sdf_origin_scaled;
    const Vec3& ext = cache.sdf_extents_scaled;
    const float local_thick = cache.local_thick;

    if (sdf_local_pos.x < -local_thick || sdf_local_pos.x > ext.x + local_thick ||
        sdf_local_pos.y < -local_thick || sdf_local_pos.y > ext.y + local_thick ||
        sdf_local_pos.z < -local_thick || sdf_local_pos.z > ext.z + local_thick) {
        return;
    }

    const int nx = collider.sdf_nx;
    const int ny = collider.sdf_ny;
    const int nz = collider.sdf_nz;
    const auto& grid = *collider.sdf_grid_data;
    const auto sampleSDF = [&](int x, int y, int z) -> float {
        const int ix = std::clamp(x, 0, nx - 1);
        const int iy = std::clamp(y, 0, ny - 1);
        const int iz = std::clamp(z, 0, nz - 1);
        const std::size_t idx =
            static_cast<std::size_t>(iz * (nx * ny) + iy * nx + ix);
        return idx < grid.size() ? grid[idx] : 0.0f;
    };
    const auto interpSDF = [&](const Vec3& lp) -> float {
        const Vec3 slp = lp - cache.sdf_origin_scaled;
        const float px = std::clamp(ext.x > 1e-6f ? slp.x / ext.x : 0.0f, 0.0f, 1.0f);
        const float py = std::clamp(ext.y > 1e-6f ? slp.y / ext.y : 0.0f, 0.0f, 1.0f);
        const float pz = std::clamp(ext.z > 1e-6f ? slp.z / ext.z : 0.0f, 0.0f, 1.0f);
        const float fx = px * (nx - 1);
        const float fy = py * (ny - 1);
        const float fz = pz * (nz - 1);
        const int ix = std::clamp(static_cast<int>(std::floor(fx)), 0, nx - 1);
        const int iy = std::clamp(static_cast<int>(std::floor(fy)), 0, ny - 1);
        const int iz = std::clamp(static_cast<int>(std::floor(fz)), 0, nz - 1);
        const int ix1 = std::clamp(ix + 1, 0, nx - 1);
        const int iy1 = std::clamp(iy + 1, 0, ny - 1);
        const int iz1 = std::clamp(iz + 1, 0, nz - 1);
        const float dx = fx - ix;
        const float dy = fy - iy;
        const float dz = fz - iz;
        const float v00 = sampleSDF(ix, iy, iz) * (1.0f - dx) + sampleSDF(ix1, iy, iz) * dx;
        const float v10 = sampleSDF(ix, iy1, iz) * (1.0f - dx) + sampleSDF(ix1, iy1, iz) * dx;
        const float v01 = sampleSDF(ix, iy, iz1) * (1.0f - dx) + sampleSDF(ix1, iy, iz1) * dx;
        const float v11 = sampleSDF(ix, iy1, iz1) * (1.0f - dx) + sampleSDF(ix1, iy1, iz1) * dx;
        const float v0 = v00 * (1.0f - dy) + v10 * dy;
        const float v1 = v01 * (1.0f - dy) + v11 * dy;
        return (v0 * (1.0f - dz) + v1 * dz) * cache.cooked_avg_scale;
    };

    const float dist_local = interpSDF(local_pos);
    if (dist_local > local_thick) {
        return;
    }

    const float dx_p = interpSDF(local_pos + Vec3(cache.step_x, 0.0f, 0.0f));
    const float dx_m = interpSDF(local_pos - Vec3(cache.step_x, 0.0f, 0.0f));
    const float dy_p = interpSDF(local_pos + Vec3(0.0f, cache.step_y, 0.0f));
    const float dy_m = interpSDF(local_pos - Vec3(0.0f, cache.step_y, 0.0f));
    const float dz_p = interpSDF(local_pos + Vec3(0.0f, 0.0f, cache.step_z));
    const float dz_m = interpSDF(local_pos - Vec3(0.0f, 0.0f, cache.step_z));

    Vec3 local_normal(dx_p - dx_m, dy_p - dy_m, dz_p - dz_m);
    const float normal_len = local_normal.length();
    if (normal_len > 1e-6f) {
        local_normal = local_normal * (1.0f / normal_len);
    } else {
        local_normal = Vec3(0.0f, 1.0f, 0.0f);
    }

    Vec3 world_normal = obb.local_to_world.transform_vector(local_normal);
    const float world_normal_len = world_normal.length();
    if (world_normal_len > 1e-6f) {
        world_normal = world_normal * (1.0f / world_normal_len);
    } else {
        world_normal = Vec3(0.0f, 1.0f, 0.0f);
    }

    const float world_dist = dist_local * cache.avg_scale;
    const float penetration = cache.world_thick - world_dist;
    position = position + world_normal * penetration;
    resolve_vel(collider, world_normal, velocity);
}

} // namespace RayTrophiSim
