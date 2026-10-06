#include "Fluid/MatterGrain.h"
#include "Fluid/MatterGrainColliderBvh.h"
#include "Fluid/MatterGrainStages.h"
#include "Fluid/MatterGpuRuntime.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/FluidThermalLiquid.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>

namespace RayTrophiSim::Fluid {
namespace {

// Two banks x budget slots x {key + tangential spring, rolling spring}.
constexpr std::size_t kHistoryBytesPerGrain =
    2 * kMatterGrainContactBudget * 2 * 4 * sizeof(uint32_t);
constexpr std::size_t kDiagnosticWords = 8;

bool finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Buckets per table: load factor <= 1/4 grain so hash collisions rarely
// stack two occupied cells past kMatterGrainBucketCapacity.
uint32_t bucketsFor(std::size_t capacity) {
    uint32_t buckets = 1;
    while (buckets < capacity * 4) {
        buckets *= 2;
    }
    return buckets;
}

std::size_t bucketBytes(uint32_t buckets) {
    return std::size_t{kMatterGrainBucketTables} * buckets *
        (1 + kMatterGrainBucketCapacity) * sizeof(uint32_t);
}

// Bytes the particle-side buffers occupy at `capacity` grains.
std::size_t particleBytes(std::size_t capacity) {
    return bucketBytes(bucketsFor(capacity)) +
        2 * capacity * sizeof(float) +             // mass, ids
        9 * capacity * sizeof(float) +             // scratch bank
        capacity * kHistoryBytesPerGrain +
        2 * capacity * sizeof(uint32_t) +          // history owners
        kDiagnosticWords * sizeof(uint32_t);
}

std::size_t colliderBytes(std::size_t triangles) {
    triangles = std::max(triangles, std::size_t{1});
    return triangles * 3 * sizeof(Vec3) + triangles * 2 * sizeof(MatterGrainBvhNode) +
        triangles * sizeof(uint32_t);
}

ComputeBufferHandle makeBuffer(SimulationComputeContext& compute, const char* name,
                               std::size_t bytes) {
    ComputeBufferDesc desc;
    desc.debug_name = name;
    desc.size_bytes = bytes;
    desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite |
        ComputeBufferUsage::Upload | ComputeBufferUsage::Download;
    return compute.createBuffer(desc);
}

void destroy(SimulationComputeContext& compute, std::initializer_list<ComputeBufferHandle*> handles) {
    for (auto* h : handles) {
        if (h->valid()) {
            compute.destroyBuffer(*h);
        }
        *h = {};
    }
}

bool particleBuffersValid(const MatterGrainGpuRuntime& r) {
    return r.bucket_counts.valid() && r.bucket_slots.valid() && r.mass.valid() && r.ids.valid() &&
        r.scratch.valid() && r.history.valid() && r.history_owner.valid() &&
        r.diagnostics.valid();
}

bool colliderBuffersValid(const MatterGrainGpuRuntime& r) {
    return r.triangles.valid() && r.collider_nodes.valid() && r.collider_patches.valid();
}

// Geometric growth: a streaming emitter adds grains every frame, and every
// reallocation drops contact history (no device copy in the compute API).
std::size_t grownCapacity(std::size_t current, std::size_t count) {
    return std::max({count, current + current / 2, std::size_t{256}});
}

bool ensureParticles(SimulationComputeContext& compute, MatterGrainGpuRuntime& runtime,
                     std::size_t capacity, std::string& error) {
    MatterGrainGpuRuntime candidate;
    candidate.buckets = bucketsFor(capacity);
    candidate.bucket_counts = makeBuffer(compute, "grain_bucket_counts",
        std::size_t{kMatterGrainBucketTables} * candidate.buckets * sizeof(uint32_t));
    candidate.bucket_slots = makeBuffer(compute, "grain_bucket_slots",
        std::size_t{kMatterGrainBucketTables} * candidate.buckets * kMatterGrainBucketCapacity *
        sizeof(uint32_t));
    candidate.mass = makeBuffer(compute, "grain_mass", capacity * sizeof(float));
    candidate.ids = makeBuffer(compute, "grain_ids", capacity * sizeof(uint32_t));
    candidate.scratch = makeBuffer(compute, "grain_scratch_state", 9 * capacity * sizeof(float));
    candidate.history = makeBuffer(compute, "grain_contact_history",
        capacity * kHistoryBytesPerGrain);
    candidate.history_owner = makeBuffer(compute, "grain_history_owner",
        2 * capacity * sizeof(uint32_t));
    candidate.diagnostics = makeBuffer(compute, "grain_diagnostics",
        kDiagnosticWords * sizeof(uint32_t));
    if (!particleBuffersValid(candidate)) {
        destroy(compute, {&candidate.bucket_counts, &candidate.bucket_slots, &candidate.mass, &candidate.ids,
            &candidate.scratch, &candidate.history, &candidate.history_owner,
            &candidate.diagnostics});
        error = "grain GPU allocation failed";
        return false;
    }
    destroy(compute, {&runtime.bucket_counts, &runtime.bucket_slots, &runtime.mass, &runtime.ids,
        &runtime.scratch, &runtime.history, &runtime.history_owner, &runtime.diagnostics});
    runtime.bucket_counts = candidate.bucket_counts;
    runtime.bucket_slots = candidate.bucket_slots;
    runtime.mass = candidate.mass;
    runtime.ids = candidate.ids;
    runtime.scratch = candidate.scratch;
    runtime.history = candidate.history;
    runtime.history_owner = candidate.history_owner;
    runtime.diagnostics = candidate.diagnostics;
    runtime.buckets = candidate.buckets;
    runtime.capacity = capacity;
    runtime.history_fresh = true;
    return true;
}

bool ensureCollider(SimulationComputeContext& compute, MatterGrainGpuRuntime& runtime,
                    std::size_t triangles, std::string& error) {
    triangles = std::max(triangles, std::size_t{1});
    MatterGrainGpuRuntime candidate;
    candidate.triangles = makeBuffer(compute, "grain_flat_triangles", triangles * 3 * sizeof(Vec3));
    candidate.collider_nodes = makeBuffer(compute, "grain_collider_nodes",
        triangles * 2 * sizeof(MatterGrainBvhNode));
    candidate.collider_patches = makeBuffer(compute, "grain_collider_patches",
        triangles * sizeof(uint32_t));
    if (!colliderBuffersValid(candidate)) {
        destroy(compute, {&candidate.triangles, &candidate.collider_nodes,
            &candidate.collider_patches});
        error = "grain GPU allocation failed";
        return false;
    }
    destroy(compute, {&runtime.triangles, &runtime.collider_nodes, &runtime.collider_patches});
    runtime.triangles = candidate.triangles;
    runtime.collider_nodes = candidate.collider_nodes;
    runtime.collider_patches = candidate.collider_patches;
    runtime.triangle_capacity = triangles;
    runtime.collider_uploaded = false;
    return true;
}

} // namespace

void releaseMatterGrainGpu(SimulationComputeContext& compute, MatterGrainGpuRuntime& r) {
    destroy(compute, {&r.bucket_counts, &r.bucket_slots, &r.mass, &r.ids, &r.scratch, &r.history,
        &r.history_owner, &r.diagnostics, &r.triangles, &r.collider_nodes,
        &r.collider_patches});
    r = {};
}

bool stepMatterGrainGpu(SimulationGridDomainState& state, const MatterGrainParams& params,
    float dt, const Vec3& gravity, std::size_t budget_bytes,
    SimulationComputeContext& compute, SimulationGridDomainComputeBuffers& buffers,
    const std::vector<SurfaceMeshTriangle>& triangles, std::string& error) {
    error.clear();
    auto& p = state.particles;
    const auto count = p.size();
    auto checked = params;
    if (!patchMatterGrainParams(matterGrainParamsToJson(params), checked, error)) {
        return false;
    }
    if (!count || count > 100000 || triangles.size() > 4096 ||
        !std::isfinite(dt) || dt <= 0.0f || !finite(gravity) ||
        compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch() || buffers.fluid_particle_capacity < count ||
        buffers.fluid_uploaded_particle_count != count ||
        p.rest_mass_kg.size() != count || p.mass_fraction.size() != count ||
        p.pore_water_mass_kg.size() != count || p.affine.size() != count ||
        p.velocity.size() != count || p.constitutive_model.size() != count ||
        p.particle_id.size() != count) {
        error = "dry grain requires ready Vulkan canonical state, 1..100000 grains, <=4096 faces";
        return false;
    }
    std::vector<float> masses(count);
    std::vector<uint32_t> ids(count);
    float minimum_mass = std::numeric_limits<float>::max();
    float maximum_speed = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        masses[i] = p.rest_mass_kg[i] * p.mass_fraction[i];
        if (p.constitutive_model[i] != static_cast<uint8_t>(MatterConstitutiveModel::Granular) ||
            isFrozenParticle(p, i) || p.pore_water_mass_kg[i] != 0.0f ||
            !std::isfinite(masses[i]) || masses[i] < 1e-6f ||
            !finite(p.position[i]) || !finite(p.velocity[i]) ||
            !finite(p.affine[i].col0) || !finite(p.affine[i].col1) ||
            !finite(p.affine[i].col2)) {
            error = "dry grain candidate accepts explicit mobile dry granular carriers only";
            return false;
        }
        // History keys: 31-bit stable identity; the top bit marks supports.
        ids[i] = static_cast<uint32_t>(p.particle_id[i] & 0x7fffffffull);
        minimum_mass = std::min(minimum_mass, masses[i]);
        maximum_speed = std::max(maximum_speed, p.velocity[i].length());
    }
    // Substep size. Every grain owns at most kMatterGrainContactBudget contact
    // springs (grain, wall and mesh patch); exceeding it refuses publication,
    // so the bounds below hold for any state that is published.
    //  stability: Gershgorin bound of the coupled spring system,
    //             omega_max^2 <= 2 * Z * k_eff / m (pair sum, equal masses);
    //             half the symplectic-Euler limit 2/omega_max.
    //  accuracy:  `contact_resolution` substeps per binary collision,
    //             t_c = pi * sqrt(m_pair / k), m_pair = m/2.
    //  damping:   aggregate explicit damping rate Z*(2cn + 7cs)/m below 0.5
    //             per substep (7 = tangential 1/m + r^2/I for both grains).
    //  travel:    no grain moves more than 10% of its radius per substep.
    // The tangential spring acts through 1/m + r^2/I = 3.5/m, hence 3.5 kt.
    // The rolling spring 2.25 mu_r^2 k R^2 on I = 0.4 m r^2 is stiffest
    // against a wall (R = r): 5.625 mu_r^2 k / m, hence 2.81 mu_r^2 k.
    const double budget = kMatterGrainContactBudget;
    const double m = minimum_mass;
    const double k = params.stiffness_n_m;
    const double kt = params.tangential_stiffness_ratio * k;
    const double mu_r = params.rolling_friction;
    const double k_effective = std::max({k, 3.5 * kt, 2.8125 * mu_r * mu_r * k});
    const double stability_dt = 1.0 / std::sqrt(2.0 * budget * k_effective / m);
    const double accuracy_dt = 3.14159265358979 * std::sqrt(.5 * m / k) /
        params.contact_resolution;
    const double damping_rate = budget *
        (2.0 * params.normal_damping_n_s_m + 7.0 * params.sliding_damping_n_s_m) / m;
    const double damping_dt = damping_rate > 0.0 ? .5 / damping_rate
        : std::numeric_limits<double>::infinity();
    const double travel_dt = .1 * params.radius_m /
        std::max(double(maximum_speed) + double(gravity.length()) * dt, 1e-8);
    const std::array<std::pair<double, const char*>, 4> bounds{{
        {accuracy_dt, "accuracy"}, {stability_dt, "stability"},
        {damping_dt, "damping"}, {travel_dt, "travel"}}};
    const auto& limiting = *std::min_element(bounds.begin(), bounds.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    const double requested = std::ceil(dt / limiting.first);
    if (!std::isfinite(requested) || requested > params.max_substeps) {
        error = std::string("grain ") + limiting.second +
            " CFL exceeds max_substeps; reduce dt or stiffness";
        return false;
    }
    // Even: the ping-pong must end in the canonical (bank 0) buffers.
    auto substeps = static_cast<uint32_t>(std::max(2.0, requested));
    substeps += substeps & 1u;
    if (substeps > static_cast<uint32_t>(params.max_substeps)) {
        error = "grain CFL exceeds max_substeps after even ping-pong rounding";
        return false;
    }

    auto& runtime_owner = buffers.matter_runtime;
    const auto* previous = runtime_owner ? &runtime_owner->grain : nullptr;
    const bool grow_particles = !previous || !particleBuffersValid(*previous) ||
        previous->capacity < count;
    const bool grow_collider = !previous || !colliderBuffersValid(*previous) ||
        previous->triangle_capacity < std::max(triangles.size(), std::size_t{1});
    const std::size_t capacity = grow_particles
        ? grownCapacity(previous ? previous->capacity : 0, count) : previous->capacity;
    const std::size_t triangle_capacity = grow_collider
        ? std::max(triangles.size(), std::size_t{1}) : previous->triangle_capacity;
    std::size_t working = particleBytes(capacity) + colliderBytes(triangle_capacity);
    if (previous && (grow_particles || grow_collider)) {
        // Existing scratch remains allocated during transactional replacement.
        working += (grow_particles ? particleBytes(previous->capacity) : 0) +
            (grow_collider ? colliderBytes(previous->triangle_capacity) : 0);
    }
    const auto largest_buffer = std::max({capacity * kHistoryBytesPerGrain,
        std::size_t{kMatterGrainBucketTables} * bucketsFor(capacity) *
            kMatterGrainBucketCapacity * sizeof(uint32_t),
        triangle_capacity * 2 * sizeof(MatterGrainBvhNode),
        triangle_capacity * 3 * sizeof(Vec3)});
    if ((budget_bytes && working > budget_bytes) || state.voxel_size <= 0.0f ||
        (compute.caps().max_storage_buffer_bytes &&
         largest_buffer > compute.caps().max_storage_buffer_bytes)) {
        error = "grain buffers exceed device limits";
        return false;
    }
    const auto fingerprint = matterGrainColliderFingerprint(triangles);
    const bool refresh_collider = grow_collider || !previous->collider_uploaded ||
        previous->collider_fingerprint != fingerprint;
    MatterGrainColliderBvh collider;
    if (refresh_collider && !buildMatterGrainColliderBvh(triangles, collider, error)) {
        return false;
    }
    if (!runtime_owner) {
        runtime_owner = std::make_shared<MatterGpuRuntime>();
    }
    auto& runtime = runtime_owner->grain;
    if ((grow_particles && !ensureParticles(compute, runtime, capacity, error)) ||
        (grow_collider && !ensureCollider(compute, runtime, triangle_capacity, error))) {
        return false;
    }
    std::array<uint32_t, kDiagnosticWords> diagnostics{};
    if (!compute.uploadBuffer(runtime.diagnostics, diagnostics.data(), sizeof(diagnostics)) ||
        !compute.uploadBuffer(runtime.mass, masses.data(), masses.size() * sizeof(float)) ||
        !compute.uploadBuffer(runtime.ids, ids.data(), ids.size() * sizeof(uint32_t))) {
        error = "grain mass/identity/diagnostic upload failed";
        return false;
    }
    if (refresh_collider) {
        runtime.collider_uploaded = false;
        if ((!collider.vertices.empty() && !compute.uploadBuffer(runtime.triangles,
                collider.vertices.data(), collider.vertices.size() * sizeof(Vec3))) ||
            (!collider.nodes.empty() && !compute.uploadBuffer(runtime.collider_nodes,
                collider.nodes.data(), collider.nodes.size() * sizeof(MatterGrainBvhNode))) ||
            (!collider.surface_patches.empty() && !compute.uploadBuffer(runtime.collider_patches,
                collider.surface_patches.data(), collider.surface_patches.size() * sizeof(uint32_t)))) {
            error = "grain flat BVH upload failed";
            return false;
        }
        runtime.collider_node_count = static_cast<uint32_t>(collider.nodes.size());
        runtime.collider_fingerprint = fingerprint;
        runtime.collider_uploaded = true;
    }
    struct alignas(16) Constants {
        uint32_t count, buckets, twisting_friction_bits, collider_nodes;
        float low[4];
        float high[4];
        float contact[4];
        float rolling[4];
        uint32_t substep, reset_history, tangential_stiffness_bits, last_substep;
    } constants{};
    static_assert(sizeof(Constants) == 96);
    constants.count = static_cast<uint32_t>(count);
    constants.buckets = runtime.buckets;
    constants.collider_nodes = runtime.collider_node_count;
    std::memcpy(&constants.twisting_friction_bits, &params.twisting_friction, sizeof(float));
    const float tangential_stiffness = static_cast<float>(kt);
    std::memcpy(&constants.tangential_stiffness_bits, &tangential_stiffness, sizeof(float));
    Vec3 low, high;
    state.grid.getWorldBounds(low, high);
    for (int axis = 0; axis < 3; ++axis) {
        constants.low[axis] = axis == 0 ? low.x : axis == 1 ? low.y : low.z;
        constants.high[axis] = axis == 0 ? high.x : axis == 1 ? high.y : high.z;
    }
    constants.low[3] = params.radius_m;
    constants.high[3] = params.stiffness_n_m;
    constants.contact[0] = dt / substeps;
    constants.contact[1] = params.normal_damping_n_s_m;
    constants.contact[2] = params.sliding_damping_n_s_m;
    constants.contact[3] = params.friction;
    constants.rolling[0] = params.rolling_friction;
    constants.rolling[1] = gravity.x;
    constants.rolling[2] = gravity.y;
    constants.rolling[3] = gravity.z;
    constants.reset_history = runtime.history_fresh ? 1u : 0u;
    constants.last_substep = substeps - 1;
    const bool history_reset = runtime.history_fresh;
    // Until this frame publishes, the device history may not match the host.
    runtime.history_fresh = true;
    const ComputeBufferHandle handles[] = {buffers.fluid_positions, buffers.fluid_velocities,
        buffers.fluid_affine, runtime.mass, runtime.ids, runtime.bucket_counts, runtime.bucket_slots,
        runtime.scratch, runtime.history, runtime.history_owner, runtime.diagnostics,
        runtime.triangles, runtime.collider_nodes, runtime.collider_patches};
    static_assert(sizeof(handles) / sizeof(handles[0]) == 14);
    if (!dispatchMatterGrainStages(substeps, [&](const char* kernel, uint32_t substep) {
            ComputeDispatch command;
            command.kernel = kernel;
            command.buffers = handles;
            command.buffer_count = 14;
            constants.substep = substep;
            command.constants = &constants;
            command.constants_size = sizeof(constants);
            // Clear: all bucket tables (and both history-owner banks).
            // Step: max(count, buckets) -- each invocation also clears one
            // bucket of the table the next substep writes. Hash: grains.
            uint32_t threads = constants.count;
            if (std::strcmp(kernel, "sim_matter_grain_clear") == 0) {
                threads = kMatterGrainBucketTables * runtime.buckets;
            } else if (std::strcmp(kernel, "sim_matter_grain_step") == 0) {
                threads = std::max(constants.count, runtime.buckets);
            }
            command.groups.groups_x = (threads + 255u) / 256u;
            if (!compute.dispatch(command)) {
                buffers.fluid_uploaded_particle_count = 0;
                error = std::string("grain stage failed: ") + kernel;
                return false;
            }
            return true;
        })) {
        return false;
    }
    auto result = p;
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(buffers.fluid_positions, result.position.data(),
        count * sizeof(Vec3));
    ok = compute.downloadBuffer(buffers.fluid_velocities, result.velocity.data(),
        count * sizeof(Vec3)) && ok;
    ok = compute.downloadBuffer(buffers.fluid_affine, result.affine.data(),
        count * sizeof(AffineC)) && ok;
    ok = compute.downloadBuffer(runtime.diagnostics, diagnostics.data(), sizeof(diagnostics)) && ok;
    ok = compute.endTransferBatch() && ok;
    for (std::size_t i = 0; i < count && ok; ++i) {
        ok = finite(result.position[i]) && finite(result.velocity[i]) &&
            finite(result.affine[i].col0) && finite(result.affine[i].col1) &&
            finite(result.affine[i].col2);
    }
    buffers.fluid_uploaded_particle_count = 0;
    if (ok && diagnostics[1] != kMatterGrainShaderRevision) {
        error = "grain step shader revision mismatch; rebuild simulation shaders";
        return false;
    }
    const auto overflow = diagnostics[0];
    if (!ok || overflow) {
        error = (overflow & 4u) ? "grain neighbour bucket overflow (more than 16 grains hashed "
                "to one bucket); grains overlap far beyond contact" :
            (overflow & 2u) ? "grain collider traversal/manifold budget exceeded" :
            overflow ? "grain contact count exceeds the 24-contact history/CFL budget; "
                "reduce emitter packing or stiffness overlap" :
            "grain publication rejected nonfinite/readback state";
        return false;
    }
    result.advanceMaterialCoordinates();
    p = std::move(result);
    runtime.history_fresh = false;
    auto& stats = state.fluid_stats;
    auto& report = stats.grain_report;
    report = {};
    report.substeps = static_cast<int>(substeps);
    report.dispatches = static_cast<int>(substeps + 2);
    report.substep_dt = dt / substeps;
    report.limit = limiting.second;
    report.max_contacts = diagnostics[2];
    report.sticking_contacts = diagnostics[3];
    report.contacts = diagnostics[4];
    report.history_reset = history_reset;
    stats.mixed_model_step = true;
    stats.mixed_step_held = false;
    stats.mixed_common_substeps = static_cast<int>(substeps);
    stats.mixed_contact_pairs = diagnostics[4];
    stats.mixed_working_set_bytes = working;
    stats.pressure_on_gpu = false;
    stats.g2p_on_gpu = false;
    stats.p2g_on_gpu = false;
    stats.gpu_status = "Dry grain Vulkan DEM candidate: fused hash/contact/history/spin";
    return true;
}

} // namespace RayTrophiSim::Fluid
