// ParticleSimulationSystem: device-resident ballistic particles (particle
// roadmap Phase 1.5 Batch B). Kept out of ParticleSimulation.cpp, which is far
// past the 2000-line limit; step() only decides which path runs.
//
// Residency contract (roadmap 7.3), per stream group:
//   kinematics  (position, velocity)            device authoritative when resident
//   lifecycle   (alive, age, lifetime, profile,
//                size scale, rotation, emitter) host authoritative, always
// The host keeps aging and killing with the same float operations the device
// uses, which is what lets it allocate spawn slots without a readback. A slot
// record overwrites every device stream of its slot, so the host allocator
// stays the authority even if the two ever disagreed.
//
// What this replaced: a "partial GPU" path that uploaded the whole SoA twice a
// step, dispatched forces, synchronized, downloaded velocity and did the rest
// on the CPU. Phase 0 measured it slower than the CPU at every count; its
// fixed ~1 ms was the mid-step sync + readback. It is gone, not kept beside
// this one: a system is either resident or runs the CPU reference end to end.

#include "ParticleSimulation.h"
#include "globals.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace RayTrophiSim {
namespace {

// Mirrored field for field in shaders/sim_particle_ballistic.comp.
struct ParticleBallisticGpuConstants {
    int particle_count = 0;
    float dt = 0.0f;
    float gravity_x = 0.0f, gravity_y = 0.0f, gravity_z = 0.0f;
    float buoyancy = 0.0f;
    float drag_factor = 1.0f;
    float time_seconds = 0.0f;
    uint32_t system_mask = 0u;
    uint32_t force_count = 0u;
};
static_assert(sizeof(ParticleBallisticGpuConstants) == 40,
              "sim_particle_ballistic push constants: shader, kernel table (40) and this "
              "struct must change together");

// shaders/sim_particle_spawn.comp declares only record_count; the kernel table
// registers 16 bytes.
struct ParticleSpawnGpuConstants {
    uint32_t record_count = 0u;
    uint32_t pad[3] = {};
};
static_assert(sizeof(ParticleSpawnGpuConstants) == 16,
              "sim_particle_spawn push constants: kernel table registers 16 bytes");

static_assert(sizeof(ParticleResidentSlotRecord) == 48,
              "ParticleResidentSlotRecord must match SlotRecord in sim_particle_spawn.comp");

constexpr std::size_t kMinResidentCapacity = 1024;
constexpr std::size_t kMinRecordCapacity = 64;

std::size_t roundUpPow2(std::size_t n, std::size_t minimum) {
    std::size_t c = minimum;
    while (c < n) c *= 2;
    return c;
}

float msSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

const ComputeBufferUsage kResidentStreamUsage =
    ComputeBufferUsage::Storage | ComputeBufferUsage::Upload | ComputeBufferUsage::Download |
    ComputeBufferUsage::ReadWrite;

} // namespace

// ── Eligibility ─────────────────────────────────────────────────────────────

const char* ParticleSimulationSystem::residentIneligibility() const {
    // Exactly the stages that read HOST positions in the middle of a step.
    // Phase 6 moves them to the device; until then they keep the system on
    // the CPU reference, and the reason is reported, not hidden.
    for (const auto& collider : colliders_) {
        if (collider.enabled) return "host_consumer_colliders";
    }
    if (physics_settings_.self_collision_enabled) return "host_consumer_self_collision";
    if (!grid_domains_.empty()) return "host_consumer_grid_domains";
    return nullptr;
}

// ── Host-side bookkeeping ───────────────────────────────────────────────────

void ParticleSimulationSystem::noteResidentSlotWrite(std::size_t index) {
    // In Host residency the next resident step uploads everything anyway.
    if (residency_ != ParticleKinematicResidency::Host) {
        resident_pending_slots_.push_back(static_cast<uint32_t>(index));
    }
}

void ParticleSimulationSystem::advanceHostLifecycle(float dt) {
    // Same operations, same order as the CPU reference loop in step() and as
    // sim_particle_ballistic.comp: age += dt, kill at age >= lifetime.
    for (std::size_t i = 0; i < buffers_.alive.size(); ++i) {
        if (buffers_.alive[i] == 0u) continue;
        buffers_.age_seconds[i] += dt;
        if (buffers_.age_seconds[i] >= buffers_.lifetime_seconds[i]) {
            buffers_.alive[i] = 0u;
            if (alive_count_ > 0) --alive_count_;
            continue;
        }
        buffers_.rotation[i] += buffers_.angular_velocity[i] * dt;
    }
}

void ParticleSimulationSystem::resetResidency() {
    residency_ = ParticleKinematicResidency::Host;
    resident_pending_slots_.clear();
}

// ── Device buffers ──────────────────────────────────────────────────────────

void ParticleSimulationSystem::forgetResidentBuffers() {
    // Handles only: the backend that owns them frees its buffers when it dies.
    resident_buffers_ = ParticleResidentBuffers{};
    resident_compute_ = nullptr;
    resetResidency();
}

void ParticleSimulationSystem::releaseResidentBuffers(SimulationComputeContext& compute) {
    ComputeBufferHandle* streams[] = {
        &resident_buffers_.position_x, &resident_buffers_.position_y,
        &resident_buffers_.position_z, &resident_buffers_.velocity_x,
        &resident_buffers_.velocity_y, &resident_buffers_.velocity_z,
        &resident_buffers_.age_seconds, &resident_buffers_.lifetime_seconds,
        &resident_buffers_.alive, &resident_buffers_.appearance_profile,
        &resident_buffers_.size_scale, &resident_buffers_.slot_records,
    };
    for (ComputeBufferHandle* handle : streams) {
        if (handle->valid() && handle->backend == compute.backendType()) {
            compute.destroyBuffer(*handle);
        }
        *handle = {};
    }
    resident_buffers_.capacity = 0;
    resident_buffers_.record_capacity = 0;
    resident_compute_ = nullptr;
    resetResidency();
}

bool ParticleSimulationSystem::residentBuffersValidFor(const SimulationComputeContext& compute) const {
    const ParticleResidentBuffers& b = resident_buffers_;
    const ComputeBufferHandle* streams[] = {
        &b.position_x, &b.position_y, &b.position_z, &b.velocity_x, &b.velocity_y,
        &b.velocity_z, &b.age_seconds, &b.lifetime_seconds, &b.alive,
        &b.appearance_profile, &b.size_scale,
    };
    for (const ComputeBufferHandle* handle : streams) {
        if (!handle->valid() || handle->backend != compute.backendType()) return false;
    }
    return b.capacity > 0;
}

bool ParticleSimulationSystem::ensureResidentBuffers(SimulationComputeContext& compute) {
    const std::size_t needed = roundUpPow2(std::max<std::size_t>(buffers_.alive.size(), 1),
                                           kMinResidentCapacity);
    if (residentBuffersValidFor(compute) && resident_buffers_.capacity >= needed) {
        return true;
    }
    // Growing reallocates: whatever only the device knows must come home first,
    // or the particles jump back to where the host last saw them.
    if (residency_ == ParticleKinematicResidency::Device && residentBuffersValidFor(compute)) {
        if (!downloadResidentKinematics(compute, "capacity_growth", /*in_step=*/false)) {
            return false;
        }
    }
    releaseResidentBuffers(compute);  // also resets residency to Host

    auto create = [&](ComputeBufferHandle& handle, const char* name, std::size_t bytes) {
        ComputeBufferDesc desc;
        desc.debug_name = name;
        desc.size_bytes = bytes;
        desc.usage = kResidentStreamUsage;
        handle = compute.createBuffer(desc);
        return handle.valid();
    };
    const std::size_t f = needed * sizeof(float);
    const std::size_t u = needed * sizeof(uint32_t);
    bool ok = create(resident_buffers_.position_x, "ParticleResidentPositionX", f) &&
              create(resident_buffers_.position_y, "ParticleResidentPositionY", f) &&
              create(resident_buffers_.position_z, "ParticleResidentPositionZ", f) &&
              create(resident_buffers_.velocity_x, "ParticleResidentVelocityX", f) &&
              create(resident_buffers_.velocity_y, "ParticleResidentVelocityY", f) &&
              create(resident_buffers_.velocity_z, "ParticleResidentVelocityZ", f) &&
              create(resident_buffers_.age_seconds, "ParticleResidentAge", f) &&
              create(resident_buffers_.lifetime_seconds, "ParticleResidentLifetime", f) &&
              create(resident_buffers_.alive, "ParticleResidentAlive", u) &&
              create(resident_buffers_.appearance_profile, "ParticleResidentProfile", u) &&
              create(resident_buffers_.size_scale, "ParticleResidentSizeScale", f);
    if (!ok) {
        releaseResidentBuffers(compute);
        return false;
    }
    resident_buffers_.capacity = needed;
    return true;
}

bool ParticleSimulationSystem::uploadResidentFullState(SimulationComputeContext& compute) {
    // Every device slot is written, padding included: device memory is not
    // zeroed, and the pull shader draws 6 x slots, so an unwritten alive word
    // would draw garbage.
    const std::size_t n = buffers_.alive.size();
    const std::size_t cap = resident_buffers_.capacity;
    auto uploadFloats = [&](ComputeBufferHandle handle, const std::vector<float>& src) {
        resident_f32_scratch_.assign(cap, 0.0f);
        std::copy_n(src.begin(), std::min(n, src.size()), resident_f32_scratch_.begin());
        return compute.uploadBuffer(handle, resident_f32_scratch_.data(), cap * sizeof(float));
    };
    bool ok = uploadFloats(resident_buffers_.position_x, buffers_.position_x) &&
              uploadFloats(resident_buffers_.position_y, buffers_.position_y) &&
              uploadFloats(resident_buffers_.position_z, buffers_.position_z) &&
              uploadFloats(resident_buffers_.velocity_x, buffers_.velocity_x) &&
              uploadFloats(resident_buffers_.velocity_y, buffers_.velocity_y) &&
              uploadFloats(resident_buffers_.velocity_z, buffers_.velocity_z) &&
              uploadFloats(resident_buffers_.age_seconds, buffers_.age_seconds) &&
              uploadFloats(resident_buffers_.lifetime_seconds, buffers_.lifetime_seconds) &&
              uploadFloats(resident_buffers_.size_scale, buffers_.size_scale);
    if (!ok) return false;

    resident_u32_scratch_.assign(cap, 0u);
    for (std::size_t i = 0; i < n; ++i) resident_u32_scratch_[i] = buffers_.alive[i] ? 1u : 0u;
    ok = compute.uploadBuffer(resident_buffers_.alive, resident_u32_scratch_.data(),
                              cap * sizeof(uint32_t));
    resident_u32_scratch_.assign(cap, 0u);
    std::copy_n(buffers_.appearance_profile.begin(),
                std::min(n, buffers_.appearance_profile.size()), resident_u32_scratch_.begin());
    ok = ok && compute.uploadBuffer(resident_buffers_.appearance_profile,
                                    resident_u32_scratch_.data(), cap * sizeof(uint32_t));
    return ok;
}

// ── Kinematics home ─────────────────────────────────────────────────────────

bool ParticleSimulationSystem::downloadResidentKinematics(SimulationComputeContext& compute,
                                                          const char* reason, bool in_step) {
    const std::size_t n = std::min(buffers_.alive.size(), resident_buffers_.capacity);
    if (n == 0) {
        residency_ = ParticleKinematicResidency::Equal;
        return true;
    }
    for (auto* scratch : {&resident_download_[0], &resident_download_[1], &resident_download_[2],
                          &resident_download_[3], &resident_download_[4],
                          &resident_download_[5]}) {
        scratch->resize(n);
    }
    const ComputeBufferHandle handles[6] = {
        resident_buffers_.position_x, resident_buffers_.position_y, resident_buffers_.position_z,
        resident_buffers_.velocity_x, resident_buffers_.velocity_y, resident_buffers_.velocity_z,
    };
    // One batch: the copies ride behind whatever this step recorded and land
    // with ONE submit + fence.
    compute.beginTransferBatch();
    bool ok = true;
    for (int s = 0; s < 6 && ok; ++s) {
        ok = compute.downloadBuffer(handles[s], resident_download_[s].data(), n * sizeof(float));
    }
    ok = compute.endTransferBatch() && ok;
    if (!ok) {
        // All-or-nothing: the host keeps its last state and says so.
        SCENE_LOG_ERROR(std::string("[Particles] device kinematics download failed (") + reason +
                        "); host keeps its previous positions");
        return false;
    }

    // Slots written on the host since the last resident step (spawns, kills)
    // are newer on the host than on the device: keep them.
    resident_slot_mask_.assign(n, 0u);
    for (uint32_t slot : resident_pending_slots_) {
        if (slot < n) resident_slot_mask_[slot] = 1u;
    }
    std::vector<float>* host[6] = {&buffers_.position_x, &buffers_.position_y,
                                   &buffers_.position_z, &buffers_.velocity_x,
                                   &buffers_.velocity_y, &buffers_.velocity_z};
    for (int s = 0; s < 6; ++s) {
        float* dst = host[s]->data();
        const float* src = resident_download_[s].data();
        for (std::size_t i = 0; i < n; ++i) {
            if (!resident_slot_mask_[i]) dst[i] = src[i];
        }
    }

    residency_ = ParticleKinematicResidency::Equal;
    host_snapshot_stats_.download_bytes += static_cast<uint64_t>(n) * 6u * sizeof(float);
    host_snapshot_stats_.last_reason = reason;
    if (in_step) {
        ++host_snapshot_stats_.mirrored_steps;
    } else {
        ++host_snapshot_stats_.synchronous_count;
    }
    return true;
}

bool ParticleSimulationSystem::syncHostState(const char* reason) {
    // Demand is recorded even when nothing has to move: a consumer that asks
    // every frame makes the next resident steps mirror inside their own fence
    // instead of paying a separate synchronisation each time.
    host_demand_serial_ = resident_step_serial_ + 1;
    if (residency_ != ParticleKinematicResidency::Device) {
        return true;
    }
    if (!resident_compute_ || !residentBuffersValidFor(*resident_compute_)) {
        SCENE_LOG_ERROR(std::string("[Particles] device-resident state lost before '") + reason +
                        "' could read it (compute backend gone); continuing from the last host "
                        "state");
        forgetResidentBuffers();
        return false;
    }
    return downloadResidentKinematics(*resident_compute_, reason, /*in_step=*/false);
}

void ParticleSimulationSystem::onComputeBackendChanging(SimulationComputeContext& compute) {
    if (residency_ == ParticleKinematicResidency::Device && residentBuffersValidFor(compute)) {
        downloadResidentKinematics(compute, "backend_switch", /*in_step=*/false);
    }
    // The handles belong to the dying backend; a new backend of the same type
    // would not know their ids.
    forgetResidentBuffers();
}

// ── The resident step ───────────────────────────────────────────────────────

bool ParticleSimulationSystem::stepDeviceResident(const SimulationContext& context,
                                                  float drag_factor) {
    const auto start = std::chrono::steady_clock::now();
    SimulationComputeContext& compute = *context.compute;
    SimulationTransferProbeScope probe(&compute, stats_.step_transfer);
    const float dt = context.dt;

    // Force fields have to be on the device, or this step would silently run
    // without them.
    const bool fields_ready =
        !context.force_snapshot || context.force_snapshot->empty() ||
        (context.force_compute_buffer && context.force_compute_buffer->valid());
    if (!fields_ready || !ensureResidentBuffers(compute)) {
        stats_.gpu_status = "buffers_not_ready";
        stats_.gpu_step_ms = msSince(start);
        return false;
    }
    resident_compute_ = &compute;
    if (residency_ == ParticleKinematicResidency::Host) {
        if (!uploadResidentFullState(compute)) {
            stats_.gpu_status = "buffers_not_ready";
            stats_.gpu_step_ms = msSince(start);
            return false;
        }
        residency_ = ParticleKinematicResidency::Equal;
        resident_pending_slots_.clear();
    }

    // Slot records, newest write per slot wins (the scatter runs in parallel,
    // so two records for one slot would race).
    const std::size_t n = buffers_.alive.size();
    resident_records_.clear();
    if (!resident_pending_slots_.empty()) {
        resident_slot_mask_.assign(n, 0u);
        for (auto it = resident_pending_slots_.rbegin(); it != resident_pending_slots_.rend();
             ++it) {
            const uint32_t s = *it;
            if (s >= n || resident_slot_mask_[s]) continue;
            resident_slot_mask_[s] = 1u;
            ParticleResidentSlotRecord r;
            r.slot = s;
            r.alive = buffers_.alive[s] ? 1u : 0u;
            r.profile = buffers_.appearance_profile[s];
            r.size_scale = buffers_.size_scale[s];
            r.position_lifetime[0] = buffers_.position_x[s];
            r.position_lifetime[1] = buffers_.position_y[s];
            r.position_lifetime[2] = buffers_.position_z[s];
            r.position_lifetime[3] = buffers_.lifetime_seconds[s];
            r.velocity_age[0] = buffers_.velocity_x[s];
            r.velocity_age[1] = buffers_.velocity_y[s];
            r.velocity_age[2] = buffers_.velocity_z[s];
            r.velocity_age[3] = buffers_.age_seconds[s];
            resident_records_.push_back(r);
        }
    }
    stats_.slot_records = static_cast<uint32_t>(resident_records_.size());

    bool dispatched = true;
    if (!resident_records_.empty()) {
        const std::size_t need = roundUpPow2(resident_records_.size(), kMinRecordCapacity);
        if (!resident_buffers_.slot_records.valid() ||
            resident_buffers_.slot_records.backend != compute.backendType() ||
            resident_buffers_.record_capacity < need) {
            if (resident_buffers_.slot_records.valid() &&
                resident_buffers_.slot_records.backend == compute.backendType()) {
                compute.destroyBuffer(resident_buffers_.slot_records);
            }
            ComputeBufferDesc desc;
            desc.debug_name = "ParticleResidentSlotRecords";
            desc.size_bytes = need * sizeof(ParticleResidentSlotRecord);
            desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                         ComputeBufferUsage::ReadOnly;
            resident_buffers_.slot_records = compute.createBuffer(desc);
            resident_buffers_.record_capacity =
                resident_buffers_.slot_records.valid() ? need : 0;
        }
        dispatched = resident_buffers_.slot_records.valid() &&
                     compute.uploadBuffer(resident_buffers_.slot_records, resident_records_.data(),
                                          resident_records_.size() *
                                              sizeof(ParticleResidentSlotRecord));
        if (dispatched) {
            ParticleSpawnGpuConstants constants;
            constants.record_count = static_cast<uint32_t>(resident_records_.size());
            ComputeBufferHandle bindings[12] = {
                resident_buffers_.slot_records, resident_buffers_.position_x,
                resident_buffers_.position_y, resident_buffers_.position_z,
                resident_buffers_.velocity_x, resident_buffers_.velocity_y,
                resident_buffers_.velocity_z, resident_buffers_.age_seconds,
                resident_buffers_.lifetime_seconds, resident_buffers_.alive,
                resident_buffers_.appearance_profile, resident_buffers_.size_scale,
            };
            ComputeDispatch cmd;
            cmd.kernel = "sim_particle_spawn";
            cmd.buffers = bindings;
            cmd.buffer_count = 12;
            cmd.constants = &constants;
            cmd.constants_size = sizeof(constants);
            cmd.groups.groups_x = (constants.record_count + 63u) / 64u;
            dispatched = compute.dispatch(cmd);
        }
    }

    if (dispatched && n > 0) {
        ParticleBallisticGpuConstants constants;
        constants.particle_count =
            static_cast<int>(std::min<std::size_t>(n, std::numeric_limits<int>::max()));
        constants.dt = dt;
        constants.gravity_x = gravity_.x * physics_settings_.gravity_scale;
        constants.gravity_y = gravity_.y * physics_settings_.gravity_scale;
        constants.gravity_z = gravity_.z * physics_settings_.gravity_scale;
        constants.buoyancy =
            physics_settings_.mode == ParticlePhysicsMode::Gas ? physics_settings_.buoyancy : 0.0f;
        constants.drag_factor = drag_factor;
        constants.time_seconds = context.time_seconds;
        constants.system_mask = toSimulationSystemMask(SimulationSystemKind::Particle);
        const bool has_fields =
            context.force_compute_buffer && context.force_compute_buffer->valid();
        constants.force_count =
            has_fields ? static_cast<uint32_t>(std::min<std::size_t>(
                             context.force_compute_buffer->count,
                             std::numeric_limits<uint32_t>::max()))
                       : 0u;
        // force_count 0 never reads binding 9; any valid buffer satisfies it.
        ComputeBufferHandle fields =
            has_fields ? context.force_compute_buffer->buffer : resident_buffers_.size_scale;
        ComputeBufferHandle bindings[10] = {
            resident_buffers_.position_x, resident_buffers_.position_y,
            resident_buffers_.position_z, resident_buffers_.velocity_x,
            resident_buffers_.velocity_y, resident_buffers_.velocity_z,
            resident_buffers_.age_seconds, resident_buffers_.lifetime_seconds,
            resident_buffers_.alive, fields,
        };
        ComputeDispatch cmd;
        cmd.kernel = "sim_particle_ballistic";
        cmd.buffers = bindings;
        cmd.buffer_count = 10;
        cmd.constants = &constants;
        cmd.constants_size = sizeof(constants);
        cmd.groups.groups_x = (static_cast<uint32_t>(constants.particle_count) + 255u) / 256u;
        dispatched = compute.dispatch(cmd);
    }
    if (!dispatched) {
        // Records not consumed stay pending; the caller pulls the device state
        // home and runs this step on the CPU (roadmap 7.4).
        stats_.gpu_status = "dispatch_failed";
        stats_.gpu_step_ms = msSince(start);
        return false;
    }

    resident_pending_slots_.clear();
    residency_ = ParticleKinematicResidency::Device;
    advanceHostLifecycle(dt);

    // A consumer asked for host state during the previous step: bring it home
    // inside this step's own submission instead of making it sync separately.
    const bool mirror = host_demand_serial_ != 0 && host_demand_serial_ >= resident_step_serial_;
    stats_.nonfinite_measured = false;
    if (mirror && downloadResidentKinematics(compute, "mirror", /*in_step=*/true)) {
        uint32_t nonfinite = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!buffers_.alive[i]) continue;
            if (!std::isfinite(buffers_.position_x[i]) || !std::isfinite(buffers_.position_y[i]) ||
                !std::isfinite(buffers_.position_z[i]) || !std::isfinite(buffers_.velocity_x[i]) ||
                !std::isfinite(buffers_.velocity_y[i]) || !std::isfinite(buffers_.velocity_z[i])) {
                ++nonfinite;
            }
        }
        stats_.nonfinite_particles = nonfinite;
        stats_.nonfinite_measured = true;
    }
    ++resident_step_serial_;
    stats_.gpu_status = "gpu_resident";
    stats_.device_resident = true;
    stats_.gpu_step_ms = msSince(start);
    return true;
}

// ── Consumers ───────────────────────────────────────────────────────────────

bool ParticleSimulationSystem::residentDrawBuffers(ParticleResidentDrawBuffers& out) const {
    if (residency_ == ParticleKinematicResidency::Host || !resident_compute_ ||
        resident_compute_->backendType() != ComputeBackendType::VulkanCompute ||
        !residentBuffersValidFor(*resident_compute_)) {
        return false;
    }
    const SimulationComputeContext& compute = *resident_compute_;
    out.device = compute.nativeDevice();
    const ComputeBufferHandle streams[8] = {
        resident_buffers_.position_x, resident_buffers_.position_y,
        resident_buffers_.position_z, resident_buffers_.age_seconds,
        resident_buffers_.lifetime_seconds, resident_buffers_.alive,
        resident_buffers_.appearance_profile, resident_buffers_.size_scale,
    };
    for (int i = 0; i < 8; ++i) {
        out.buffers[i] = compute.nativeBufferPtr(streams[i]);
        if (!out.buffers[i]) return false;
    }
    out.capacity = static_cast<uint32_t>(std::min(buffers_.alive.size(), resident_buffers_.capacity));
    out.state_version = data_version_;
    return out.device != nullptr;
}

ParticleKinematicResidency ParticleSimulationSystem::kinematicResidency() const {
    return residency_;
}

std::size_t ParticleSimulationSystem::residentCapacity() const {
    return resident_buffers_.capacity;
}

const ParticleHostSnapshotStats& ParticleSimulationSystem::hostSnapshotStats() const {
    return host_snapshot_stats_;
}

const char* particleKinematicResidencyName(ParticleKinematicResidency residency) {
    switch (residency) {
    case ParticleKinematicResidency::Host: return "host";
    case ParticleKinematicResidency::Equal: return "equal";
    case ParticleKinematicResidency::Device: return "device";
    }
    return "host";
}

} // namespace RayTrophiSim
