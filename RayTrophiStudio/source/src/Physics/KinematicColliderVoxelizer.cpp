#include "KinematicColliderVoxelizer.h"

#include "FluidGrid.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace RayTrophiSim {
namespace {

float lengthSquared(const Vec3& value) {
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

float capsuleDistanceSquared(const Vec3& point,
                             const Vec3& start,
                             const Vec3& end) {
    const Vec3 segment = end - start;
    const float denominator = lengthSquared(segment);
    const float t = denominator > 1.0e-12f
        ? std::clamp((point - start).dot(segment) / denominator, 0.0f, 1.0f)
        : 0.0f;
    return lengthSquared(point - (start + segment * t));
}

void worldBounds(const KinematicProxySample& sample,
                 Vec3& minimum,
                 Vec3& maximum) {
    if (sample.shape == KinematicProxyShape::Sphere) {
        const Vec3 extent(sample.radius);
        minimum = sample.center - extent;
        maximum = sample.center + extent;
        return;
    }
    if (sample.shape == KinematicProxyShape::Capsule) {
        const Vec3 extent(sample.radius);
        minimum = Vec3::min(sample.capsule_start, sample.capsule_end) - extent;
        maximum = Vec3::max(sample.capsule_start, sample.capsule_end) + extent;
        return;
    }
    minimum = Vec3(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    maximum = minimum * -1.0f;
    for (int z = -1; z <= 1; z += 2) {
        for (int y = -1; y <= 1; y += 2) {
            for (int x = -1; x <= 1; x += 2) {
                const Vec3 local(
                    sample.half_extents.x * static_cast<float>(x),
                    sample.half_extents.y * static_cast<float>(y),
                    sample.half_extents.z * static_cast<float>(z));
                const Vec3 world = sample.world_transform.transform_point(local);
                minimum = Vec3::min(minimum, world);
                maximum = Vec3::max(maximum, world);
            }
        }
    }
}

bool inside(const KinematicProxySample& sample,
            const Matrix4x4& inverse_box,
            const Vec3& point) {
    if (sample.shape == KinematicProxyShape::Sphere) {
        return lengthSquared(point - sample.center) <=
               sample.radius * sample.radius;
    }
    if (sample.shape == KinematicProxyShape::Capsule) {
        return capsuleDistanceSquared(
                   point, sample.capsule_start, sample.capsule_end) <=
               sample.radius * sample.radius;
    }
    const Vec3 local = inverse_box.transform_point(point);
    return std::fabs(local.x) <= sample.half_extents.x &&
           std::fabs(local.y) <= sample.half_extents.y &&
           std::fabs(local.z) <= sample.half_extents.z;
}

} // namespace

void prepareKinematicColliderGrid(FluidSim::FluidGrid& grid) {
    const bool had_velocity = grid.solid_vel.size() == grid.solid.size();
    const std::size_t count = grid.kinematic_touched_cells.size();
    for (std::size_t index = 0; index < count; ++index) {
        const uint32_t cell = grid.kinematic_touched_cells[index];
        if (cell >= grid.solid.size() ||
            index >= grid.kinematic_previous_solid.size()) {
            continue;
        }
        grid.solid[cell] = grid.kinematic_previous_solid[index];
        if (had_velocity && index < grid.kinematic_previous_velocity.size()) {
            grid.solid_vel[cell] = grid.kinematic_previous_velocity[index];
        }
    }
    grid.kinematic_touched_cells.clear();
    grid.kinematic_previous_solid.clear();
    grid.kinematic_previous_velocity.clear();
    if (count > 0) {
        grid.collider_voxel_valid = false;
        grid.collider_weights_init = false;
        grid.solid_cells_valid = false;
    }
}

bool voxelizeKinematicColliders(
    FluidSim::FluidGrid& grid,
    const std::vector<KinematicProxySample>& samples,
    uint32_t consumer_mask,
    std::vector<uint32_t>* stamped_cells) {
    if (stamped_cells) {
        stamped_cells->assign(samples.size(), 0u);
    }
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        grid.solid.empty() || grid.voxel_size <= 0.0f) {
        return false;
    }
    bool have_candidate = false;
    bool moving = false;
    for (const KinematicProxySample& sample : samples) {
        if (!sample.resolved || (sample.consumer_mask & consumer_mask) == 0u) {
            continue;
        }
        have_candidate = true;
        moving |= sample.velocity_valid &&
                  (lengthSquared(sample.linear_velocity) > 1.0e-12f ||
                   lengthSquared(sample.angular_velocity) > 1.0e-12f);
    }
    if (!have_candidate) {
        return false;
    }
    if (moving && grid.solid_vel.size() != grid.solid.size()) {
        grid.solid_vel.assign(grid.solid.size(), Vec3(0.0f));
    }
    const bool track_velocity = grid.solid_vel.size() == grid.solid.size();
    const float inverse_voxel = 1.0f / grid.voxel_size;
    std::unordered_set<uint32_t> touched;
    touched.reserve(256);

    for (std::size_t sample_index = 0; sample_index < samples.size();
         ++sample_index) {
        const KinematicProxySample& sample = samples[sample_index];
        if (!sample.resolved || (sample.consumer_mask & consumer_mask) == 0u) {
            continue;
        }
        uint32_t* stamped = stamped_cells
            ? &(*stamped_cells)[sample_index]
            : nullptr;
        Vec3 minimum;
        Vec3 maximum;
        worldBounds(sample, minimum, maximum);
        const Vec3 grid_min = (minimum - grid.origin) * inverse_voxel;
        const Vec3 grid_max = (maximum - grid.origin) * inverse_voxel;
        const int i0 = std::clamp(
            static_cast<int>(std::floor(grid_min.x)) - 1, 0, grid.nx - 1);
        const int j0 = std::clamp(
            static_cast<int>(std::floor(grid_min.y)) - 1, 0, grid.ny - 1);
        const int k0 = std::clamp(
            static_cast<int>(std::floor(grid_min.z)) - 1, 0, grid.nz - 1);
        const int i1 = std::clamp(
            static_cast<int>(std::ceil(grid_max.x)) + 1, 0, grid.nx - 1);
        const int j1 = std::clamp(
            static_cast<int>(std::ceil(grid_max.y)) + 1, 0, grid.ny - 1);
        const int k1 = std::clamp(
            static_cast<int>(std::ceil(grid_max.z)) + 1, 0, grid.nz - 1);
        if (i1 < i0 || j1 < j0 || k1 < k0) {
            continue;
        }
        Matrix4x4 inverse_box = Matrix4x4::identity();
        if (sample.shape == KinematicProxyShape::Box) {
            inverse_box = sample.world_transform.inverse();
        }
        for (int k = k0; k <= k1; ++k) {
            for (int j = j0; j <= j1; ++j) {
                for (int i = i0; i <= i1; ++i) {
                    const Vec3 center = grid.origin + Vec3(
                        (static_cast<float>(i) + 0.5f) * grid.voxel_size,
                        (static_cast<float>(j) + 0.5f) * grid.voxel_size,
                        (static_cast<float>(k) + 0.5f) * grid.voxel_size);
                    if (!inside(sample, inverse_box, center)) {
                        continue;
                    }
                    const uint32_t cell = static_cast<uint32_t>(
                        grid.cellIndex(i, j, k));
                    if (stamped) {
                        ++*stamped;
                    }
                    if (touched.insert(cell).second) {
                        grid.kinematic_touched_cells.push_back(cell);
                        grid.kinematic_previous_solid.push_back(grid.solid[cell]);
                        grid.kinematic_previous_velocity.push_back(
                            track_velocity ? grid.solid_vel[cell] : Vec3(0.0f));
                    }
                    if (grid.solid[cell] == 0u && grid.solid_cells_valid) {
                        grid.solid_cells.push_back(cell);
                    }
                    grid.solid[cell] = FluidSim::FluidGrid::kSolidCollider;
                    if (track_velocity) {
                        const Vec3 radius = center - sample.center;
                        grid.solid_vel[cell] = sample.velocity_valid
                            ? sample.linear_velocity +
                                  Vec3::cross(sample.angular_velocity, radius)
                            : Vec3(0.0f);
                    }
                }
            }
        }
        grid.collider_cur_lo[0] = std::min(grid.collider_cur_lo[0], i0);
        grid.collider_cur_lo[1] = std::min(grid.collider_cur_lo[1], j0);
        grid.collider_cur_lo[2] = std::min(grid.collider_cur_lo[2], k0);
        grid.collider_cur_hi[0] = std::max(grid.collider_cur_hi[0], i1);
        grid.collider_cur_hi[1] = std::max(grid.collider_cur_hi[1], j1);
        grid.collider_cur_hi[2] = std::max(grid.collider_cur_hi[2], k1);
    }
    grid.collider_weights_init = false;
    return !grid.kinematic_touched_cells.empty();
}

void appendKinematicStampRecords(
    std::vector<KinematicStampRecord>& log,
    const std::string& domain,
    uint32_t consumer_mask,
    int frame,
    const std::vector<KinematicProxySample>& samples,
    const std::vector<uint32_t>& stamped_cells) {
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const KinematicProxySample& sample = samples[index];
        if (!sample.resolved || (sample.consumer_mask & consumer_mask) == 0u) {
            continue;
        }
        KinematicStampRecord record;
        record.domain = domain;
        record.consumer_mask = consumer_mask;
        record.frame = frame;
        record.set_id = sample.set_id;
        record.proxy_id = sample.proxy_id;
        record.proxy_name = sample.proxy_name;
        record.stamped_cells =
            index < stamped_cells.size() ? stamped_cells[index] : 0u;
        record.linear_velocity = sample.linear_velocity;
        record.angular_velocity = sample.angular_velocity;
        record.velocity_valid = sample.velocity_valid;
        log.push_back(std::move(record));
    }
}

} // namespace RayTrophiSim
