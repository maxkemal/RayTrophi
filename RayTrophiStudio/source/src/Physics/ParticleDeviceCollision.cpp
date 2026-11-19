// ParticleSimulationSystem: device collision stage for resident particles
// (docs/dev/PLAYER_GAME_PAINT_ROOM.md step 2). Runs after the ballistic step.
//
// The device mirrors the CPU reference (applyColliders) collider by collider:
// the same resolved descriptors (refreshResolvedColliders), the same discrete
// formulas, the same order. What it does NOT have is the CPU's swept tests for
// sphere / capsule / AABB / OBB; a particle fast enough to cross a collider in
// one step tunnels on the device. ObjectConvexDecomp and ObjectMeshBVH keep the
// system on the CPU reference (deviceColliderIneligibility says which).
//
// Contacts: response Die kills the particle on the device only. The host keeps
// treating the slot as alive until its lifetime ends (it ages it with the same
// operations), so the host allocator never reuses a slot early; a slot record
// overwrites every device stream when it is reused, so the two never disagree
// about a live particle. downloadResidentKinematics brings device deaths home.
//
// Events (record_collision_events): a bounded append buffer + {count, overflow}
// counter on the device, kept until a consumer takes them (readCollisionEvents).

#include "ParticleSimulation.h"
#include "globals.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace RayTrophiSim {
namespace {

enum ParticleGpuColliderType : uint32_t {
    kGpuColliderPlaneY = 0,
    kGpuColliderSphere = 1,
    kGpuColliderCapsule = 2,
    kGpuColliderAABB = 3,
    kGpuColliderOBB = 4,
    kGpuColliderSDF = 5,
};

// Mirrored field for field in shaders/sim_particle_collide.comp (std430).
struct ParticleGpuCollider {
    uint32_t type = 0;
    uint32_t source_index = 0;     // event "collider" field
    uint32_t sdf_offset = 0;       // first float of this grid in collider_sdf
    int32_t sdf_nx = 0;
    int32_t sdf_ny = 0;
    int32_t sdf_nz = 0;
    float restitution = 0.0f;
    float friction = 0.0f;
    float thickness = 0.0f;        // resolved (refreshResolvedColliders) thickness
    float plane_y = 0.0f;
    float radius = 0.0f;           // sphere / capsule radius
    float _pad0 = 0.0f;
    float a[4] = {};               // sphere centre | capsule start | AABB min | OBB local min
    float b[4] = {};               // capsule end | AABB max | OBB local max
    float local_to_world[12] = {}; // rows 0..2 of the affine matrix (OBB / SDF)
    float world_to_local[12] = {};
    float sdf_origin_scaled[4] = {};   // xyz, w = avg world-per-local scale
    float sdf_extents_scaled[4] = {};  // xyz, w = mean cooked-to-local scale
};
static_assert(sizeof(ParticleGpuCollider) == 208,
              "ParticleGpuCollider must match GpuCollider in sim_particle_collide.comp");

// Mirrored field for field in sim_particle_collide.comp; kernel table: 48.
struct ParticleCollideGpuConstants {
    int particle_count = 0;
    uint32_t collider_count = 0u;
    float dt = 0.0f;
    float particle_radius = 0.0f;
    uint32_t response = 0u;        // ParticleCollisionResponse
    uint32_t record_events = 0u;
    uint32_t event_capacity = 0u;
    uint32_t plane_enabled = 0u;
    float plane_y = 0.0f;
    float plane_restitution = 0.0f;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
};
static_assert(sizeof(ParticleCollideGpuConstants) == 48,
              "sim_particle_collide push constants: shader, kernel table (48) and this "
              "struct must change together");

bool sdfCooked(const ParticleColliderDesc& c) {
    return c.sdf_grid_data && !c.sdf_grid_data->empty() && c.sdf_nx > 0 && c.sdf_ny > 0 &&
           c.sdf_nz > 0 &&
           c.sdf_grid_data->size() >= static_cast<std::size_t>(c.sdf_nx) * c.sdf_ny * c.sdf_nz;
}

void packRows(const Matrix4x4& m, float out[12]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) out[r * 4 + c] = m.m[r][c];
    }
}

std::size_t roundUpPow2(std::size_t n, std::size_t minimum) {
    std::size_t c = minimum;
    while (c < n) c *= 2;
    return c;
}

bool ensureBuffer(SimulationComputeContext& compute, ComputeBufferHandle& handle,
                  std::size_t& capacity, std::size_t needed, std::size_t element_bytes,
                  const char* name, ComputeBufferUsage usage) {
    if (handle.valid() && handle.backend == compute.backendType() && capacity >= needed) {
        return true;
    }
    if (handle.valid() && handle.backend == compute.backendType()) {
        compute.destroyBuffer(handle);
    }
    handle = {};
    capacity = 0;
    ComputeBufferDesc desc;
    desc.debug_name = name;
    desc.size_bytes = std::max<std::size_t>(needed, 1) * element_bytes;
    desc.usage = usage;
    handle = compute.createBuffer(desc);
    if (!handle.valid()) return false;
    capacity = std::max<std::size_t>(needed, 1);
    return true;
}

} // namespace

// ── Eligibility ─────────────────────────────────────────────────────────────

const char* ParticleSimulationSystem::deviceColliderIneligibility() const {
    for (const auto& collider : colliders_) {
        if (!collider.enabled) continue;
        switch (collider.source_mode) {
            case ParticleColliderSourceMode::PlaneY:
            case ParticleColliderSourceMode::Sphere:
            case ParticleColliderSourceMode::Capsule:
            case ParticleColliderSourceMode::ObjectAABB:
            case ParticleColliderSourceMode::ObjectOBB:
                break;
            case ParticleColliderSourceMode::ObjectMeshSDF:
                // The CPU reference also skips an uncooked SDF; the device does
                // the same only once a grid exists, so the step stays on the
                // CPU (with the reason) until the cook lands.
                if (!sdfCooked(collider)) return "host_consumer_collider_sdf_not_cooked";
                break;
            case ParticleColliderSourceMode::ObjectConvexDecomp:
            case ParticleColliderSourceMode::ObjectMeshBVH:
            default:
                return "host_consumer_collider_mode_cpu_only";
        }
    }
    return nullptr;
}

bool ParticleSimulationSystem::deviceCollisionStageNeeded() const {
    if (collision_plane_enabled_) return true;
    for (const auto& collider : colliders_) {
        if (collider.enabled) return true;
    }
    return false;
}

// ── The stage ───────────────────────────────────────────────────────────────

bool ParticleSimulationSystem::stepDeviceCollisions(SimulationComputeContext& compute, float dt) {
    const float particle_radius = std::max(0.0f, physics_settings_.particle_radius);
    refreshResolvedColliders(particle_radius);

    std::vector<ParticleGpuCollider> packed;
    packed.reserve(resolved_colliders_.size());
    uint64_t sdf_signature = 1469598103934665603ull;
    std::size_t sdf_floats = 0;
    const auto mix = [&](uint64_t v) { sdf_signature = (sdf_signature ^ v) * 1099511628211ull; };
    for (const ResolvedCollider& rc : resolved_colliders_) {
        const ParticleColliderDesc& d = rc.desc;
        ParticleGpuCollider g;
        g.source_index = rc.source_index;
        g.restitution = std::clamp(d.restitution, 0.0f, 1.0f);
        g.friction = std::clamp(d.friction, 0.0f, 1.0f);
        g.thickness = d.thickness;
        switch (d.source_mode) {
            case ParticleColliderSourceMode::PlaneY:
                g.type = kGpuColliderPlaneY;
                g.plane_y = d.plane_y;
                break;
            case ParticleColliderSourceMode::Sphere:
                g.type = kGpuColliderSphere;
                g.a[0] = d.sphere_center.x; g.a[1] = d.sphere_center.y; g.a[2] = d.sphere_center.z;
                g.radius = d.sphere_radius;
                break;
            case ParticleColliderSourceMode::Capsule:
                g.type = kGpuColliderCapsule;
                g.a[0] = d.capsule_start.x; g.a[1] = d.capsule_start.y; g.a[2] = d.capsule_start.z;
                g.b[0] = d.capsule_end.x; g.b[1] = d.capsule_end.y; g.b[2] = d.capsule_end.z;
                g.radius = d.capsule_radius;
                break;
            case ParticleColliderSourceMode::ObjectAABB: {
                g.type = kGpuColliderAABB;
                const Vec3 mn = Vec3::min(d.bounds_min, d.bounds_max);
                const Vec3 mx = Vec3::max(d.bounds_min, d.bounds_max);
                g.a[0] = mn.x; g.a[1] = mn.y; g.a[2] = mn.z;
                g.b[0] = mx.x; g.b[1] = mx.y; g.b[2] = mx.z;
                break;
            }
            case ParticleColliderSourceMode::ObjectOBB: {
                if (!rc.has_obb) continue;
                g.type = kGpuColliderOBB;
                const Vec3 mn = Vec3::min(rc.obb.local_bounds_min, rc.obb.local_bounds_max);
                const Vec3 mx = Vec3::max(rc.obb.local_bounds_min, rc.obb.local_bounds_max);
                g.a[0] = mn.x; g.a[1] = mn.y; g.a[2] = mn.z;
                g.b[0] = mx.x; g.b[1] = mx.y; g.b[2] = mx.z;
                packRows(rc.obb.local_to_world, g.local_to_world);
                packRows(rc.obb.local_to_world.inverse(), g.world_to_local);
                break;
            }
            case ParticleColliderSourceMode::ObjectMeshSDF: {
                if (!rc.has_obb || !sdfCooked(d)) continue;
                g.type = kGpuColliderSDF;
                packRows(rc.obb.local_to_world, g.local_to_world);
                packRows(rc.obb.local_to_world.inverse(), g.world_to_local);
                // Same derived quantities as resolveSDFCollision.
                const Vec3 cooked = d.sdf_mesh_size;
                const Vec3 current = rc.obb.local_bounds_max - rc.obb.local_bounds_min;
                Vec3 scale(cooked.x > 1e-6f ? current.x / cooked.x : 1.0f,
                           cooked.y > 1e-6f ? current.y / cooked.y : 1.0f,
                           cooked.z > 1e-6f ? current.z / cooked.z : 1.0f);
                scale.x = std::max(1e-4f, scale.x);
                scale.y = std::max(1e-4f, scale.y);
                scale.z = std::max(1e-4f, scale.z);
                const Matrix4x4& m = rc.obb.local_to_world;
                const float sx = Vec3(m.m[0][0], m.m[1][0], m.m[2][0]).length();
                const float sy = Vec3(m.m[0][1], m.m[1][1], m.m[2][1]).length();
                const float sz = Vec3(m.m[0][2], m.m[1][2], m.m[2][2]).length();
                float avg_scale = (sx + sy + sz) / 3.0f;
                if (avg_scale <= 1e-6f) avg_scale = 1.0f;
                const Vec3 origin = d.sdf_origin * scale;
                const Vec3 extents = d.sdf_extents * scale;
                g.sdf_origin_scaled[0] = origin.x; g.sdf_origin_scaled[1] = origin.y;
                g.sdf_origin_scaled[2] = origin.z; g.sdf_origin_scaled[3] = avg_scale;
                g.sdf_extents_scaled[0] = extents.x; g.sdf_extents_scaled[1] = extents.y;
                g.sdf_extents_scaled[2] = extents.z;
                g.sdf_extents_scaled[3] = (scale.x + scale.y + scale.z) / 3.0f;
                g.sdf_nx = d.sdf_nx; g.sdf_ny = d.sdf_ny; g.sdf_nz = d.sdf_nz;
                g.sdf_offset = static_cast<uint32_t>(sdf_floats);
                sdf_floats += d.sdf_grid_data->size();
                mix(reinterpret_cast<uint64_t>(d.sdf_grid_data.get()));
                mix(d.sdf_grid_data->size());
                break;
            }
            default:
                continue;  // ineligible modes never reach the device stage
        }
        packed.push_back(g);
    }
    mix(sdf_floats);

    const ComputeBufferUsage upload_usage = ComputeBufferUsage::Storage |
                                            ComputeBufferUsage::Upload |
                                            ComputeBufferUsage::ReadOnly;
    if (!ensureBuffer(compute, resident_buffers_.colliders, resident_buffers_.collider_capacity,
                      roundUpPow2(std::max<std::size_t>(packed.size(), 1), 8),
                      sizeof(ParticleGpuCollider), "ParticleGpuColliders", upload_usage)) {
        return false;
    }
    if (!packed.empty() &&
        !compute.uploadBuffer(resident_buffers_.colliders, packed.data(),
                              packed.size() * sizeof(ParticleGpuCollider))) {
        return false;
    }
    // Cooked grids move only when a collider is re-cooked: upload on change.
    if (sdf_floats > 0 && sdf_signature != resident_buffers_.sdf_signature) {
        if (!ensureBuffer(compute, resident_buffers_.collider_sdf, resident_buffers_.sdf_capacity,
                          roundUpPow2(sdf_floats, 1024), sizeof(float), "ParticleColliderSdf",
                          upload_usage)) {
            return false;
        }
        std::vector<float> concatenated;
        concatenated.reserve(sdf_floats);
        for (const ResolvedCollider& rc : resolved_colliders_) {
            if (rc.desc.source_mode == ParticleColliderSourceMode::ObjectMeshSDF && rc.has_obb &&
                sdfCooked(rc.desc)) {
                concatenated.insert(concatenated.end(), rc.desc.sdf_grid_data->begin(),
                                    rc.desc.sdf_grid_data->end());
            }
        }
        if (!compute.uploadBuffer(resident_buffers_.collider_sdf, concatenated.data(),
                                  concatenated.size() * sizeof(float))) {
            return false;
        }
        resident_buffers_.sdf_signature = sdf_signature;
    }

    const bool record = physics_settings_.record_collision_events;
    const uint32_t event_capacity =
        std::max<uint32_t>(1u, physics_settings_.collision_event_capacity);
    if (record) {
        const ComputeBufferUsage rw = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                                      ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
        const bool had_counter = resident_buffers_.collision_counter.valid() &&
                                 resident_buffers_.collision_counter.backend == compute.backendType();
        std::size_t counter_capacity = had_counter ? 2 : 0;
        if (!ensureBuffer(compute, resident_buffers_.collision_events,
                          resident_buffers_.event_capacity, event_capacity,
                          sizeof(ParticleCollisionEvent), "ParticleCollisionEvents", rw) ||
            !ensureBuffer(compute, resident_buffers_.collision_counter, counter_capacity, 2,
                          sizeof(uint32_t), "ParticleCollisionCounter", rw)) {
            return false;
        }
        if (!had_counter) {
            const uint32_t zero[2] = {0u, 0u};
            if (!compute.uploadBuffer(resident_buffers_.collision_counter, zero, sizeof(zero))) {
                return false;
            }
        }
    }

    ParticleCollideGpuConstants constants;
    constants.particle_count = static_cast<int>(
        std::min<std::size_t>(buffers_.alive.size(), std::numeric_limits<int>::max()));
    constants.collider_count = static_cast<uint32_t>(packed.size());
    constants.dt = dt;
    constants.particle_radius = particle_radius;
    constants.response = static_cast<uint32_t>(physics_settings_.collision_response);
    constants.record_events = record ? 1u : 0u;
    constants.event_capacity =
        record ? static_cast<uint32_t>(std::min<std::size_t>(event_capacity,
                                                             resident_buffers_.event_capacity))
               : 0u;
    constants.plane_enabled = collision_plane_enabled_ ? 1u : 0u;
    constants.plane_y = collision_plane_y_;
    constants.plane_restitution = collision_restitution_;
    if (constants.particle_count <= 0) return true;

    // Unused bindings still need a valid buffer; the kernel never reads them
    // when the matching count / flag is zero (same rule as the ballistic step).
    const ComputeBufferHandle placeholder = resident_buffers_.size_scale;
    const auto orPlaceholder = [&](const ComputeBufferHandle& h) {
        return h.valid() && h.backend == compute.backendType() ? h : placeholder;
    };
    ComputeBufferHandle bindings[15] = {
        resident_buffers_.position_x, resident_buffers_.position_y,
        resident_buffers_.position_z, resident_buffers_.velocity_x,
        resident_buffers_.velocity_y, resident_buffers_.velocity_z,
        resident_buffers_.alive, resident_buffers_.age_seconds,
        resident_buffers_.lifetime_seconds, resident_buffers_.appearance_profile,
        orPlaceholder(resident_buffers_.colliders),
        sdf_floats > 0 ? orPlaceholder(resident_buffers_.collider_sdf) : placeholder,
        record ? orPlaceholder(resident_buffers_.collision_events) : placeholder,
        record ? orPlaceholder(resident_buffers_.collision_counter) : placeholder,
        // The drop size goes into the event flags (bits 8..15): the wet paint
        // splat scales its paint load and radius by it.
        resident_buffers_.size_scale,
    };
    ComputeDispatch cmd;
    cmd.kernel = "sim_particle_collide";
    cmd.buffers = bindings;
    cmd.buffer_count = 15;
    cmd.constants = &constants;
    cmd.constants_size = sizeof(constants);
    cmd.groups.groups_x = (static_cast<uint32_t>(constants.particle_count) + 255u) / 256u;
    if (!compute.dispatch(cmd)) return false;
    stats_.collision_stage_on_device = true;
    stats_.gpu_colliders = constants.collider_count;
    return true;
}

// ── Events ──────────────────────────────────────────────────────────────────

bool ParticleSimulationSystem::readDeviceCollisionEvents(SimulationComputeContext& compute,
                                                         uint32_t max_events, bool consume,
                                                         ParticleCollisionEventReadout& out) {
    out = ParticleCollisionEventReadout{};
    out.device = true;
    out.capacity = static_cast<uint32_t>(resident_buffers_.event_capacity);
    const ComputeBufferHandle& counter = resident_buffers_.collision_counter;
    if (!counter.valid() || counter.backend != compute.backendType()) return false;
    uint32_t values[2] = {0u, 0u};
    if (!compute.downloadBuffer(counter, values, sizeof(values))) return false;
    out.pending = std::min<uint32_t>(values[0], out.capacity);
    out.overflow = values[1];
    const uint32_t take = std::min(out.pending, max_events);
    if (take > 0 && resident_buffers_.collision_events.valid()) {
        out.events.resize(take);
        if (!compute.downloadBuffer(resident_buffers_.collision_events, out.events.data(),
                                    take * sizeof(ParticleCollisionEvent))) {
            out.events.clear();
            return false;
        }
    }
    if (consume) {
        const uint32_t zero[2] = {0u, 0u};
        compute.uploadBuffer(counter, zero, sizeof(zero));
    }
    return true;
}

bool ParticleSimulationSystem::deviceCollisionEventView(const SimulationComputeContext& compute,
                                                        ParticleDeviceEventView& out) const {
    out = ParticleDeviceEventView{};
    // Only the context that stepped this system holds its buffers; the same
    // backend type on another context would be a different device's handle.
    if (resident_compute_ != &compute || !physics_settings_.record_collision_events) return false;
    const ComputeBufferHandle& events = resident_buffers_.collision_events;
    const ComputeBufferHandle& counter = resident_buffers_.collision_counter;
    if (!events.valid() || !counter.valid() || events.backend != compute.backendType() ||
        counter.backend != compute.backendType() || resident_buffers_.event_capacity == 0) {
        return false;
    }
    out.events = events;
    out.counter = counter;
    out.capacity = static_cast<uint32_t>(resident_buffers_.event_capacity);
    return true;
}

std::size_t ParticleSimulationSystem::takeHostCollisionEvents(std::vector<ParticleCollisionEvent>& out) {
    const std::size_t n = host_collision_events_.size();
    out.insert(out.end(), host_collision_events_.begin(), host_collision_events_.end());
    host_collision_events_.clear();
    host_collision_overflow_ = 0;
    return n;
}

void ParticleSimulationSystem::releaseDeviceCollisionBuffers(SimulationComputeContext& compute) {
    for (ComputeBufferHandle* handle :
         {&resident_buffers_.colliders, &resident_buffers_.collider_sdf,
          &resident_buffers_.collision_events, &resident_buffers_.collision_counter}) {
        if (handle->valid() && handle->backend == compute.backendType()) {
            compute.destroyBuffer(*handle);
        }
        *handle = {};
    }
    resident_buffers_.collider_capacity = 0;
    resident_buffers_.sdf_capacity = 0;
    resident_buffers_.event_capacity = 0;
    resident_buffers_.sdf_signature = 0;
}

} // namespace RayTrophiSim
