#include "Fluid/MatterGrain.h"
#include "Fluid/MatterGrainColliderBvh.h"
#include "Fluid/MatterGrainStages.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/FluidThermalLiquid.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <unordered_map>
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
        18 * capacity * sizeof(float) +            // bank 0 (pos/vel/affine)
        9 * capacity * sizeof(float) +             // scratch bank
        12 * capacity * sizeof(float) +            // liquid coupling
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
    return r.positions.valid() && r.velocities.valid() && r.affines.valid() &&
        r.coupling.valid() && r.bucket_counts.valid() && r.bucket_slots.valid() && r.mass.valid() && r.ids.valid() &&
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
    candidate.positions = makeBuffer(compute, "grain_positions", capacity * sizeof(Vec3));
    candidate.velocities = makeBuffer(compute, "grain_velocities", capacity * sizeof(Vec3));
    candidate.affines = makeBuffer(compute, "grain_affines", capacity * sizeof(AffineC));
    candidate.coupling = makeBuffer(compute, "grain_liquid_coupling",
        capacity * 12 * sizeof(float));
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
        destroy(compute, {&candidate.positions, &candidate.velocities, &candidate.affines,
            &candidate.coupling, &candidate.bucket_counts, &candidate.bucket_slots, &candidate.mass, &candidate.ids,
            &candidate.scratch, &candidate.history, &candidate.history_owner,
            &candidate.diagnostics});
        error = "grain GPU allocation failed";
        return false;
    }
    destroy(compute, {&runtime.positions, &runtime.velocities, &runtime.affines,
        &runtime.coupling, &runtime.bucket_counts, &runtime.bucket_slots, &runtime.mass,
        &runtime.ids, &runtime.scratch, &runtime.history, &runtime.history_owner,
        &runtime.diagnostics});
    runtime.positions = candidate.positions;
    runtime.velocities = candidate.velocities;
    runtime.affines = candidate.affines;
    runtime.coupling = candidate.coupling;
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
    destroy(compute, {&r.positions, &r.velocities, &r.affines, &r.coupling,
        &r.bucket_counts, &r.bucket_slots, &r.mass, &r.ids, &r.scratch, &r.history,
        &r.history_owner, &r.diagnostics, &r.triangles, &r.collider_nodes,
        &r.collider_patches});
    r = {};
}

bool stepMatterGrainGpu(FluidParticles& p, const Vec3& low, const Vec3& high,
    const MatterGrainParams& params, float dt, const Vec3& gravity,
    std::size_t budget_bytes, SimulationComputeContext& compute,
    MatterGrainGpuRuntime& runtime, const std::vector<SurfaceMeshTriangle>& triangles,
    const std::vector<MatterGrainCouplingInput>* coupling,
    std::vector<MatterGrainCouplingOutput>* drag_out,
    MatterGrainStepReport& report, std::string& error) {
    error.clear();
    using Clock = std::chrono::steady_clock;
    const auto ms_since = [](Clock::time_point t) {
        return std::chrono::duration<float, std::milli>(Clock::now() - t).count();
    };
    const auto prepare_start = Clock::now();
    const auto count = p.size();
    auto checked = params;
    if (!patchMatterGrainParams(matterGrainParamsToJson(params), checked, error)) {
        return false;
    }
    if (!count || count > 100000 || triangles.size() > 4096 ||
        !std::isfinite(dt) || dt <= 0.0f || !finite(gravity) || !finite(low) || !finite(high) ||
        compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch() ||
        p.rest_mass_kg.size() != count || p.mass_fraction.size() != count ||
        p.pore_water_mass_kg.size() != count || p.affine.size() != count ||
        p.velocity.size() != count || p.constitutive_model.size() != count ||
        p.particle_id.size() != count || (coupling && coupling->size() != count)) {
        error = "dry grain requires Vulkan compute, 1..100000 grains, <=4096 faces";
        return false;
    }
    std::vector<float> masses(count);
    std::vector<uint32_t> ids(count);
    float minimum_mass = std::numeric_limits<float>::max();
    float maximum_speed = 0.0f;
    // Wet grains carry their held water (B6): it moves with the grain.
    double minimum_bridge_volume = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < count; ++i) {
        const float water = p.pore_water_mass_kg[i];
        masses[i] = p.rest_mass_kg[i] * p.mass_fraction[i] + (params.wet_grains ? water : 0.0f);
        if (water > 0.0f) {
            // Smallest bridge: this grain against a dry one, (V + 0) / 12
            // (sim_matter_grain.glsl bridge()).
            minimum_bridge_volume = std::min(minimum_bridge_volume, double(water) / 1000.0 / 12.0);
        }
        if (p.constitutive_model[i] != static_cast<uint8_t>(MatterConstitutiveModel::Granular) ||
            isFrozenParticle(p, i) || !std::isfinite(water) || water < 0.0f ||
            (!params.wet_grains && water != 0.0f) ||
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
    // Liquid coupling rows: {lump velocity, beta}, {buoyancy accel, lump mass},
    // {drag impulse accumulator}. Zero rows are an exact no-op in the shader.
    std::vector<float> coupling_rows(count * 12, 0.0f);
    if (params.wet_grains) {
        // Bridge water volume rides in the free fourth component of row 2.
        for (std::size_t i = 0; i < count; ++i) {
            coupling_rows[12 * i + 11] = p.pore_water_mass_kg[i] / 1000.0f;
        }
    }
    if (coupling) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto& in = (*coupling)[i];
            if (!finite(in.lump_velocity) || !finite(in.buoyancy_acceleration) ||
                !std::isfinite(in.drag_coefficient) || in.drag_coefficient < 0.0f ||
                !std::isfinite(in.lump_mass_kg) || in.lump_mass_kg < 0.0f) {
                error = "grain liquid coupling input is not finite";
                return false;
            }
            float* row = &coupling_rows[12 * i];
            row[0] = in.lump_velocity.x;
            row[1] = in.lump_velocity.y;
            row[2] = in.lump_velocity.z;
            row[3] = in.lump_mass_kg > 0.0f ? in.drag_coefficient : 0.0f;
            row[4] = in.buoyancy_acceleration.x;
            row[5] = in.buoyancy_acceleration.y;
            row[6] = in.buoyancy_acceleration.z;
            row[7] = in.lump_mass_kg;
            maximum_speed = std::max(maximum_speed, in.lump_velocity.length());
        }
    }
    // Substep size. Every grain owns at most kMatterGrainContactBudget contact
    // springs (grain, wall and mesh patch); exceeding it refuses publication,
    // so the bounds below hold for any state that is published.
    //  stability: Gershgorin bound of the coupled spring-dashpot system,
    //             omega_max^2 <= 2 * Z * k_eff / m (pair sum, equal masses),
    //             aggregate damping rate Z*(2cn + 7cs)/m (7 = tangential
    //             1/m + r^2/I for both grains; cn = 2 zeta sqrt(k m), the
    //             largest per-contact normal damping, a wall contact);
    //             half the damped symplectic-Euler limit. Reported as
    //             "damping" when the dashpots shorten it (aggregate zeta > .1).
    //  accuracy:  `contact_resolution` substeps per binary collision,
    //             t_c = pi * sqrt(m_pair / k), m_pair = m/2.
    //  travel:    no grain moves more than 10% of its radius per substep
    //             (the liquid lump speed counts: drag can carry a grain there).
    // Liquid drag is integrated implicitly and adds no bound.
    // The tangential spring acts through 1/m + r^2/I = 3.5/m, hence 3.5 kt.
    // The rolling spring 2.25 mu_r^2 k R^2 on I = 0.4 m r^2 is stiffest
    // against a wall (R = r): 5.625 mu_r^2 k / m, hence 2.81 mu_r^2 k.
    const double budget = kMatterGrainContactBudget;
    const double m = minimum_mass;
    const double k = params.stiffness_n_m;
    const double kt = params.tangential_stiffness_ratio * k;
    const double mu_r = params.rolling_friction;
    // Wet: liquid bridge stiffness at contact, |dF/dS| = F0 * 2.1 sqrt(R/V),
    // F0 = 2 pi gamma cos(theta) R x scale (stiffest for the smallest bridge).
    const float cohesion_scale = params.represented_grain_radius_m > 0.0f
        ? (params.radius_m / params.represented_grain_radius_m) *
          (params.radius_m / params.represented_grain_radius_m) : 1.0f;
    const double capillary = params.wet_grains ? 2.0 * 3.14159265358979 *
        params.surface_tension_n_m * std::cos(params.contact_angle_deg * 0.017453292519943295) *
        cohesion_scale : 0.0;
    const double bridge_stiffness = std::isfinite(minimum_bridge_volume)
        ? capillary * params.radius_m * 2.1 * std::sqrt(params.radius_m / minimum_bridge_volume)
        : 0.0;
    const double k_effective = std::max({k, 3.5 * kt, 2.8125 * mu_r * mu_r * k,
        bridge_stiffness});
    const double accuracy_dt = 3.14159265358979 * std::sqrt(.5 * m / k) /
        params.contact_resolution;
    const double zeta = matterGrainDampingRatio(params.restitution);
    const double normal_damping = 2.0 * zeta * std::sqrt(k * m);
    const double damping_rate = budget *
        (2.0 * normal_damping + 7.0 * params.sliding_damping_n_s_m) / m;
    // Spring and dashpot together: symplectic Euler on x'' = -w^2 x - 2 z w x'
    // is stable for w^2 h^2 + 4 z w h < 4, i.e. h < (2/w)(sqrt(1+z^2) - z).
    // Half of that, as the undamped bound always had (z = 0 gives the old
    // 1/w). The two bounds used to be applied separately, the dashpot one at
    // .5 / rate = 1/(4 z w), which is half the combined limit at large z and
    // set the substep of every damped scene (234 substeps per frame at 16k).
    const double omega = std::sqrt(2.0 * budget * k_effective / m);
    const double aggregate_zeta = damping_rate / (2.0 * omega);
    const double stability_dt =
        (std::sqrt(1.0 + aggregate_zeta * aggregate_zeta) - aggregate_zeta) / omega;
    const char* stability_name = aggregate_zeta > .1 ? "damping" : "stability";
    const double travel_dt = .1 * params.radius_m /
        std::max(double(maximum_speed) + double(gravity.length()) * dt, 1e-8);
    const std::array<std::pair<double, const char*>, 3> bounds{{
        {accuracy_dt, "accuracy"}, {stability_dt, stability_name}, {travel_dt, "travel"}}};
    const auto& limiting = *std::min_element(bounds.begin(), bounds.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    const double requested = std::ceil(dt / limiting.first - 1e-9);
    if (!std::isfinite(requested) || requested > params.max_substeps) {
        error = std::string("grain ") + limiting.second +
            " CFL exceeds max_substeps; reduce dt or stiffness";
        return false;
    }
    // Even: the ping-pong must end in the bank 0 buffers.
    auto substeps = static_cast<uint32_t>(std::max(2.0, requested));
    substeps += substeps & 1u;
    if (substeps > static_cast<uint32_t>(params.max_substeps)) {
        error = "grain CFL exceeds max_substeps after even ping-pong rounding";
        return false;
    }

    const bool grow_particles = !particleBuffersValid(runtime) || runtime.capacity < count;
    const bool grow_collider = !colliderBuffersValid(runtime) ||
        runtime.triangle_capacity < std::max(triangles.size(), std::size_t{1});
    const std::size_t capacity = grow_particles
        ? grownCapacity(runtime.capacity, count) : runtime.capacity;
    const std::size_t triangle_capacity = grow_collider
        ? std::max(triangles.size(), std::size_t{1}) : runtime.triangle_capacity;
    std::size_t working = particleBytes(capacity) + colliderBytes(triangle_capacity);
    if (grow_particles || grow_collider) {
        // Existing scratch remains allocated during transactional replacement.
        working += (grow_particles && runtime.capacity ? particleBytes(runtime.capacity) : 0) +
            (grow_collider && runtime.triangle_capacity
                ? colliderBytes(runtime.triangle_capacity) : 0);
    }
    const auto largest_buffer = std::max({capacity * kHistoryBytesPerGrain,
        std::size_t{kMatterGrainBucketTables} * bucketsFor(capacity) *
            kMatterGrainBucketCapacity * sizeof(uint32_t),
        triangle_capacity * 2 * sizeof(MatterGrainBvhNode),
        triangle_capacity * 3 * sizeof(Vec3)});
    if ((budget_bytes && working > budget_bytes) ||
        (compute.caps().max_storage_buffer_bytes &&
         largest_buffer > compute.caps().max_storage_buffer_bytes)) {
        error = "grain buffers exceed the domain resource budget or device limits (" +
            std::to_string(working / (1024 * 1024)) + " MB requested)";
        return false;
    }
    const auto fingerprint = matterGrainColliderFingerprint(triangles);
    const bool refresh_collider = grow_collider || !runtime.collider_uploaded ||
        runtime.collider_fingerprint != fingerprint;
    MatterGrainColliderBvh collider;
    if (refresh_collider && !buildMatterGrainColliderBvh(triangles, collider, error)) {
        return false;
    }
    if ((grow_particles && !ensureParticles(compute, runtime, capacity, error)) ||
        (grow_collider && !ensureCollider(compute, runtime, triangle_capacity, error))) {
        return false;
    }
    std::array<uint32_t, kDiagnosticWords> diagnostics{};
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
        float wet[4];  // hash cell size, capillary prefactor, rupture cap, -
    } constants{};
    static_assert(sizeof(Constants) == 112);
    constants.count = static_cast<uint32_t>(count);
    constants.buckets = runtime.buckets;
    constants.collider_nodes = runtime.collider_node_count;
    std::memcpy(&constants.twisting_friction_bits, &params.twisting_friction, sizeof(float));
    const float tangential_stiffness = static_cast<float>(kt);
    std::memcpy(&constants.tangential_stiffness_bits, &tangential_stiffness, sizeof(float));
    for (int axis = 0; axis < 3; ++axis) {
        constants.low[axis] = axis == 0 ? low.x : axis == 1 ? low.y : low.z;
        constants.high[axis] = axis == 0 ? high.x : axis == 1 ? high.y : high.z;
    }
    constants.low[3] = params.radius_m;
    constants.high[3] = params.stiffness_n_m;
    constants.contact[0] = dt / substeps;
    // Damping ratio; the shader scales it per contact by sqrt(k m_eff).
    constants.contact[1] = matterGrainDampingRatio(params.restitution);
    constants.contact[2] = params.sliding_damping_n_s_m;
    constants.contact[3] = params.friction;
    constants.rolling[0] = params.rolling_friction;
    constants.rolling[1] = gravity.x;
    constants.rolling[2] = gravity.y;
    constants.rolling[3] = gravity.z;
    // Map the incoming order onto the published one by identity. A moved
    // survivor means a foreign state (reset/scrub/restore/edit): drop all
    // history. Otherwise a new order is gathered on the device.
    std::string reset_reason = runtime.history_fresh
        ? (runtime.published_ids.empty() ? "first_step" : "allocation") : "";
    std::vector<uint32_t> previous_index;
    bool remap = false;
    if (!runtime.history_fresh) {
        std::unordered_map<uint64_t, uint32_t> published;
        published.reserve(runtime.published_ids.size());
        for (std::size_t k = 0; k < runtime.published_ids.size(); ++k) {
            published.emplace(runtime.published_ids[k], static_cast<uint32_t>(k));
        }
        previous_index.assign(count, 0xffffffffu);
        for (std::size_t i = 0; i < count; ++i) {
            const auto found = published.find(p.particle_id[i]);
            if (found == published.end()) {
                continue;  // birth: starts without springs
            }
            const Vec3& was = runtime.published_positions[found->second];
            if (std::memcmp(&was, &p.position[i], sizeof(Vec3)) != 0) {
                runtime.history_fresh = true;
                reset_reason = "host_state_changed";
                break;
            }
            previous_index[i] = found->second;
            remap = remap || found->second != i;
        }
        remap = remap && !runtime.history_fresh;
    }
    // B4: when the incoming grains are exactly what this runtime published
    // (same order, bits, mass), bank 0 already holds them: upload only the
    // per-frame rows. Any births, reorder, absorption or host edit re-uploads.
    const bool resident = !runtime.history_fresh && !remap &&
        runtime.published_ids.size() == count &&
        std::memcmp(runtime.published_ids.data(), p.particle_id.data(), count * sizeof(uint64_t)) == 0 &&
        std::memcmp(runtime.published_positions.data(), p.position.data(), count * sizeof(Vec3)) == 0 &&
        runtime.published_velocities.size() == count &&
        std::memcmp(runtime.published_velocities.data(), p.velocity.data(), count * sizeof(Vec3)) == 0 &&
        runtime.published_affines.size() == count &&
        std::memcmp(runtime.published_affines.data(), p.affine.data(), count * sizeof(AffineC)) == 0 &&
        runtime.published_masses == masses;
    std::size_t upload_bytes = sizeof(diagnostics) + coupling_rows.size() * sizeof(float);
    compute.beginTransferBatch();
    bool uploaded = compute.uploadBuffer(runtime.diagnostics, diagnostics.data(), sizeof(diagnostics));
    if (!resident) {
        uploaded = compute.uploadBuffer(runtime.mass, masses.data(), masses.size() * sizeof(float)) &&
            uploaded;
        uploaded = compute.uploadBuffer(runtime.ids, ids.data(), ids.size() * sizeof(uint32_t)) &&
            uploaded;
        uploaded = compute.uploadBuffer(runtime.positions, p.position.data(), count * sizeof(Vec3)) &&
            uploaded;
        uploaded = compute.uploadBuffer(runtime.velocities, p.velocity.data(), count * sizeof(Vec3)) &&
            uploaded;
        uploaded = compute.uploadBuffer(runtime.affines, p.affine.data(), count * sizeof(AffineC)) &&
            uploaded;
        upload_bytes += count * (sizeof(float) + sizeof(uint32_t) + 2 * sizeof(Vec3) + sizeof(AffineC));
    }
    uploaded = compute.uploadBuffer(runtime.coupling, coupling_rows.data(),
        coupling_rows.size() * sizeof(float)) && uploaded;
    uploaded = compute.endTransferBatch() && uploaded;
    if (!uploaded) {
        error = "grain state/mass/identity/coupling upload failed";
        return false;
    }
    // Bridges reach past contact up to the rupture distance; the hash cell
    // grows by that skin so the 27-cell search still finds every bridge.
    const float rupture_cap = params.wet_grains ? .5f * params.radius_m : 0.0f;
    constants.wet[0] = 2.0f * params.radius_m + rupture_cap;
    constants.wet[1] = static_cast<float>(capillary);
    constants.wet[2] = rupture_cap;
    constants.wet[3] = 0.0f;
    constants.reset_history = runtime.history_fresh ? 1u : 0u;
    constants.last_substep = substeps - 1;
    const bool history_reset = runtime.history_fresh;
    // Until this frame publishes, the device history may not match the host.
    runtime.history_fresh = true;
    const ComputeBufferHandle handles[] = {runtime.positions, runtime.velocities,
        runtime.affines, runtime.mass, runtime.ids, runtime.bucket_counts, runtime.bucket_slots,
        runtime.scratch, runtime.history, runtime.history_owner, runtime.diagnostics,
        runtime.triangles, runtime.collider_nodes, runtime.collider_patches, runtime.coupling};
    static_assert(sizeof(handles) / sizeof(handles[0]) == 15);
    report.host_prepare_ms = ms_since(prepare_start);
    const auto gpu_start = Clock::now();
    if (remap) {
        // bucket_slots is scratch until the clear/hash below: it carries the
        // new -> previous index map for the two gather dispatches.
        if (!compute.uploadBuffer(runtime.bucket_slots, previous_index.data(),
                count * sizeof(uint32_t))) {
            error = "grain history remap upload failed";
            return false;
        }
        for (const char* kernel : {"sim_matter_grain_permute", "sim_matter_grain_permute_copy"}) {
            ComputeDispatch command;
            command.kernel = kernel;
            command.buffers = handles;
            command.buffer_count = 15;
            constants.substep = 0;
            command.constants = &constants;
            command.constants_size = sizeof(constants);
            command.groups.groups_x = (constants.count + 255u) / 256u;
            if (!compute.dispatch(command)) {
                error = std::string("grain stage failed: ") + kernel;
                return false;
            }
        }
    }
    if (!dispatchMatterGrainStages(substeps, [&](const char* kernel, uint32_t substep) {
            ComputeDispatch command;
            command.kernel = kernel;
            command.buffers = handles;
            command.buffer_count = 15;
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
                error = std::string("grain stage failed: ") + kernel;
                return false;
            }
            return true;
        })) {
        return false;
    }
    // Only the three device-owned arrays come back; the rest of the grain
    // state is untouched, so it is not copied (a full FluidParticles copy per
    // frame was most of the publication cost).
    std::vector<Vec3> new_position(count), new_velocity(count);
    std::vector<AffineC> new_affine(count);
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(runtime.positions, new_position.data(),
        count * sizeof(Vec3));
    ok = compute.downloadBuffer(runtime.velocities, new_velocity.data(),
        count * sizeof(Vec3)) && ok;
    ok = compute.downloadBuffer(runtime.affines, new_affine.data(),
        count * sizeof(AffineC)) && ok;
    ok = compute.downloadBuffer(runtime.diagnostics, diagnostics.data(), sizeof(diagnostics)) && ok;
    if (coupling && drag_out) {
        ok = compute.downloadBuffer(runtime.coupling, coupling_rows.data(),
            coupling_rows.size() * sizeof(float)) && ok;
    }
    ok = compute.endTransferBatch() && ok;
    report.gpu_wait_ms = ms_since(gpu_start);
    const auto publish_start = Clock::now();
    for (std::size_t i = 0; i < count && ok; ++i) {
        ok = finite(new_position[i]) && finite(new_velocity[i]) &&
            finite(new_affine[i].col0) && finite(new_affine[i].col1) &&
            finite(new_affine[i].col2);
    }
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
    if (coupling && drag_out) {
        drag_out->assign(count, {});
        for (std::size_t i = 0; i < count; ++i) {
            const float* row = &coupling_rows[12 * i + 8];
            (*drag_out)[i].drag_impulse = Vec3(row[0], row[1], row[2]);
            if (!finite((*drag_out)[i].drag_impulse)) {
                error = "grain liquid drag impulse is not finite";
                return false;
            }
        }
    }
    p.position = std::move(new_position);
    p.velocity = std::move(new_velocity);
    p.affine = std::move(new_affine);
    p.advanceMaterialCoordinates();
    runtime.history_fresh = false;
    runtime.published_ids = p.particle_id;
    runtime.published_positions = p.position;
    runtime.published_velocities = p.velocity;
    runtime.published_affines = p.affine;
    runtime.published_masses = std::move(masses);
    report.substeps = static_cast<int>(substeps);
    report.dispatches = static_cast<int>(substeps + 2);
    report.substep_dt = dt / substeps;
    report.limit = limiting.second;
    report.max_contacts = diagnostics[2];
    report.sticking_contacts = diagnostics[3];
    report.contacts = diagnostics[4];
    report.liquid_bridges = diagnostics[5];
    report.history_reset = history_reset;
    report.history_reset_reason = reset_reason;
    report.history_remapped = remap;  // +2 gather dispatches, outside `dispatches`
    report.state_resident = resident;
    report.upload_bytes = upload_bytes + (remap ? count * sizeof(uint32_t) : 0) +
        (refresh_collider ? collider.vertices.size() * sizeof(Vec3) +
            collider.nodes.size() * sizeof(MatterGrainBvhNode) +
            collider.surface_patches.size() * sizeof(uint32_t) : 0);
    report.download_bytes = count * (2 * sizeof(Vec3) + sizeof(AffineC)) + sizeof(diagnostics) +
        (coupling && drag_out ? coupling_rows.size() * sizeof(float) : 0);
    report.transfer_batches = 2 + (remap ? 1 : 0) + (refresh_collider ? 1 : 0);
    report.grains = count;
    report.working_set_bytes = working;
    report.host_publish_ms = ms_since(publish_start);
    return true;
}

} // namespace RayTrophiSim::Fluid
