#include "Fluid/MatterGrain.h"
#include "Fluid/MatterGrainMpmContact.h"
#include "Fluid/MatterCommonClock.h"
#include "Fluid/MatterGrainColliderBvh.h"
#include "Fluid/MatterGrainStages.h"
#include "Fluid/MatterParticleIdentity.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/FluidThermalLiquid.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>

namespace RayTrophiSim::Fluid {
namespace {

// One block per grain, updated in place: budget slots x {key, tangential
// spring xyz, rolling spring xyz}. Was two banks of 32 B slots (1536 B), whose
// gather on every reorder needed the second bank; the host now keeps each
// grain on its block instead.
constexpr std::size_t kHistoryBytesPerGrain =
    kMatterGrainContactBudget * 7 * sizeof(uint32_t);
// The occupied-slot mask is one 32-bit word.
static_assert(kMatterGrainContactBudget <= 32);
// Grain -> block map entry bit: the block's old records are ignored this frame
// (a newborn grain, or every grain after a history reset).
constexpr uint32_t kFreshHistoryBlock = 0x80000000u;
// 0 overflow bits, 1 revision, 2 max contacts, 3-6 contact counts, 7-11 cost
// counters, 12-14 neighbour-list rebuild flags (substep % 3), 15 list builds,
// 16 sleeping grains (last substep).
constexpr std::size_t kDiagnosticWords = 17;
// Per grain {count, kMatterGrainListCapacity neighbour indices}.
constexpr std::size_t kListBytesPerGrain = (1 + kMatterGrainListCapacity) * sizeof(uint32_t);

bool finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Buckets: load factor <= 1/4 grain so hash collisions rarely stack two
// occupied cells past kMatterGrainBucketCapacity.
uint32_t bucketsFor(std::size_t capacity) {
    uint32_t buckets = 1;
    while (buckets < capacity * 4) {
        buckets *= 2;
    }
    return buckets;
}

// Vulkan only guarantees 65535 groups per axis. The kernels fold a 2-D grid
// back into one index (sim_matter_grain.glsl main), so the thread count is
// unbounded by it; a 1-D dispatch used to cap the step at ~1M grains.
constexpr uint32_t kMaxGrainGroupsX = 65535;
constexpr uint32_t kGrainGroupWidth = 256;

void setLinearGroups(ComputeDispatch& command, std::size_t threads) {
    const auto groups = static_cast<uint32_t>((threads + kGrainGroupWidth - 1) / kGrainGroupWidth);
    command.groups.groups_x = std::clamp(groups, 1u, kMaxGrainGroupsX);
    command.groups.groups_y = (groups + command.groups.groups_x - 1) / command.groups.groups_x;
}

// The one ceiling left is 32-bit index width in the shader, a correctness
// limit: the widest index is bucket_slots (tables * buckets * 16 slots), and
// 2^24 grains give 2^26 buckets. Memory runs out first in practice;
// the domain budget and the device storage-buffer limit report that below.
constexpr std::size_t kMaxGrains = std::size_t{1} << 24;
static_assert(std::size_t{kMatterGrainBucketTables} * (4 * kMaxGrains) *
              kMatterGrainBucketCapacity <= 0xffffffffull,
              "grain bucket_slots index exceeds 32 bits");

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
        capacity * kListBytesPerGrain +            // Verlet neighbour list
        3 * capacity * sizeof(float) +             // list build positions
        4 * capacity * sizeof(uint32_t) +          // history block map + owner/mask/rest
        kDiagnosticWords * sizeof(uint32_t);
}

std::size_t colliderBytes(std::size_t triangles) {
    triangles = std::max(triangles, std::size_t{1});
    // End positions, then vertex velocities of moving colliders.
    return triangles * 6 * sizeof(Vec3) + triangles * 2 * sizeof(MatterGrainBvhNode) +
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
        r.scratch.valid() && r.history.valid() && r.history_blocks.valid() &&
        r.diagnostics.valid() && r.neighbour_list.valid() && r.build_positions.valid();
}

bool colliderBuffersValid(const MatterGrainGpuRuntime& r) {
    return r.triangles.valid() && r.collider_nodes.valid() && r.collider_patches.valid();
}

// Geometric growth: a streaming emitter adds grains every frame, and every
// reallocation drops contact history (no device copy in the compute API).
std::size_t grownCapacity(std::size_t current, std::size_t count) {
    // Headroom never pushes the buffers past the dispatch cap (count is
    // already checked against it).
    return std::max(count, std::min(std::max(current + current / 2, std::size_t{256}),
                                    kMaxGrains));
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
    candidate.history_blocks = makeBuffer(compute, "grain_history_blocks",
        4 * capacity * sizeof(uint32_t));
    candidate.diagnostics = makeBuffer(compute, "grain_diagnostics",
        kDiagnosticWords * sizeof(uint32_t));
    candidate.neighbour_list = makeBuffer(compute, "grain_neighbour_list",
        capacity * kListBytesPerGrain);
    candidate.build_positions = makeBuffer(compute, "grain_list_build_positions",
        capacity * sizeof(Vec3));
    if (!particleBuffersValid(candidate)) {
        destroy(compute, {&candidate.positions, &candidate.velocities, &candidate.affines,
            &candidate.coupling, &candidate.bucket_counts, &candidate.bucket_slots, &candidate.mass, &candidate.ids,
            &candidate.scratch, &candidate.history, &candidate.history_blocks,
            &candidate.diagnostics, &candidate.neighbour_list, &candidate.build_positions});
        error = "grain GPU allocation failed";
        return false;
    }
    destroy(compute, {&runtime.positions, &runtime.velocities, &runtime.affines,
        &runtime.coupling, &runtime.bucket_counts, &runtime.bucket_slots, &runtime.mass,
        &runtime.ids, &runtime.scratch, &runtime.history, &runtime.history_blocks,
        &runtime.diagnostics, &runtime.neighbour_list, &runtime.build_positions});
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
    runtime.history_blocks = candidate.history_blocks;
    runtime.diagnostics = candidate.diagnostics;
    runtime.neighbour_list = candidate.neighbour_list;
    runtime.build_positions = candidate.build_positions;
    runtime.buckets = candidate.buckets;
    runtime.capacity = capacity;
    runtime.history_fresh = true;
    runtime.coupling_zero = false;
    return true;
}

bool ensureCollider(SimulationComputeContext& compute, MatterGrainGpuRuntime& runtime,
                    std::size_t triangles, std::string& error) {
    triangles = std::max(triangles, std::size_t{1});
    MatterGrainGpuRuntime candidate;
    candidate.triangles = makeBuffer(compute, "grain_flat_triangles", triangles * 6 * sizeof(Vec3));
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
        &r.history_blocks, &r.diagnostics, &r.triangles, &r.collider_nodes,
        &r.collider_patches, &r.neighbour_list, &r.build_positions});
    r = {};
}

bool stepMatterGrainGpu(FluidParticles& p, const Vec3& low, const Vec3& high,
    const MatterGrainParams& params, float dt, const Vec3& gravity,
    std::size_t budget_bytes, SimulationComputeContext& compute,
    MatterGrainGpuRuntime& runtime, const std::vector<SurfaceMeshTriangle>& triangles,
    const std::vector<MatterGrainCouplingInput>* coupling,
    std::vector<MatterGrainCouplingOutput>* drag_out,
    MatterGrainStepReport& report, std::string& error, const MatterGrainMotion* motion,
    MatterGrainMpmContact* mpm_contact, const MatterGrainCommonDriver* common_driver) {
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
    if (count > kMaxGrains) {
        // Its own message: the shared one below never said which bound was hit,
        // and a held step with a valid-looking scene read as a frozen solver.
        error = "dry grain step holds at " + std::to_string(count) + " grains: the GPU "
            "kernels index at most " + std::to_string(kMaxGrains) + " (32-bit bucket "
            "slots). Lower Max Particles on the domain or emit fewer grains.";
        return false;
    }
    if (!count || triangles.size() > kMatterGrainMaxColliderFaces ||
        !std::isfinite(dt) || dt <= 0.0f || !finite(gravity) || !finite(low) || !finite(high) ||
        compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch() ||
        p.rest_mass_kg.size() != count || p.mass_fraction.size() != count ||
        p.pore_water_mass_kg.size() != count || p.affine.size() != count ||
        p.velocity.size() != count || p.constitutive_model.size() != count ||
        p.particle_id.size() != count || (coupling && coupling->size() != count) ||
        (motion && !motion->external_acceleration.empty() &&
         motion->external_acceleration.size() != count) ||
        (motion && !motion->triangle_velocity.empty() &&
         motion->triangle_velocity.size() != 3 * triangles.size())) {
        error = "dry grain requires Vulkan compute, at least 1 grain and consistent "
            "per-grain/per-face arrays";
        return false;
    }
    std::vector<float> masses(count);
    std::vector<uint32_t> ids(count);
    float minimum_mass = std::numeric_limits<float>::max();
    float maximum_speed = 0.0f;
    if (mpm_contact) {
        maximum_speed = mpm_contact->maximumSpeed();
    }
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
    // {drag impulse accumulator}. Zero rows are an exact no-op in the shader,
    // and the shader writes a row only when it is coupled (beta and lump mass
    // > 0), so zeros already on the device stay zero: a dry frame after a dry
    // frame skips building and uploading 48 B per grain (62 MB at 1.3M). The
    // shared-clock path lets the GPU liquid coupling write the rows itself.
    const bool zero_rows = !params.wet_grains && !coupling && !common_driver &&
        !(motion && !motion->external_acceleration.empty());
    const bool rows_resident = zero_rows && runtime.coupling_zero;
    std::vector<float> coupling_rows(rows_resident ? 0 : count * 12, 0.0f);
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
    // Force fields ride in the acceleration row next to buoyancy (lift.xyz in
    // the shader): a constant acceleration over the frame, like gravity.
    if (motion && !motion->external_acceleration.empty()) {
        for (std::size_t i = 0; i < count; ++i) {
            const Vec3& a = motion->external_acceleration[i];
            if (!finite(a)) {
                error = "grain force-field acceleration is not finite";
                return false;
            }
            coupling_rows[12 * i + 4] += a.x;
            coupling_rows[12 * i + 5] += a.y;
            coupling_rows[12 * i + 6] += a.z;
            maximum_speed = std::max(maximum_speed, a.length() * dt);
        }
    }
    // A moving collider face approaches at its own speed: the travel bound
    // keeps it under 10% of a radius per substep relative to the grains too.
    const std::vector<Vec3>* triangle_velocity =
        motion && !motion->triangle_velocity.empty() ? &motion->triangle_velocity : nullptr;
    if (triangle_velocity) {
        for (const auto& v : *triangle_velocity) {
            maximum_speed = std::max(maximum_speed, v.length());
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
    // Contacts per grain the Gershgorin bound assumes. The normal dashpot is
    // proportional to the contact springs, so the aggregate damping ratio
    // grows with sqrt(contacts): assuming all 24 history slots full made the
    // normal dashpot set the substep (712 at 1.3M grains where 3 contacts were
    // measured). Equal spheres touch at most 12 neighbours (kissing number),
    // the suite's densest piles measured 8-10, so the bound uses last frame's
    // maximum + 4, never below 12. A frame that measures more is re-run below
    // at the full budget; that re-run starts without contact history, which
    // the +4 margin keeps rare. Coupled to the MPM/common clock the frame
    // cannot be re-run (the MPM side has already advanced), so it keeps 24.
    // The grain step always receives the MPM contact object; only an ACTIVE
    // one (MPM parcels in reach) advances state a re-run would apply twice.
    const bool adaptive_contacts = !(mpm_contact && mpm_contact->active()) && !common_driver;
    const uint32_t cfl_contacts = adaptive_contacts
        ? std::clamp(runtime.cfl_contact_hint, 12u, static_cast<uint32_t>(kMatterGrainContactBudget))
        : static_cast<uint32_t>(kMatterGrainContactBudget);
    const double budget = cfl_contacts;
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
    // Only the NORMAL dashpot is explicit. The sliding dashpot is capped in the
    // shader so the whole contact budget cannot reverse a slip within one
    // substep (sim_matter_grain.glsl contact(): min(c v, v / (BUDGET dt
    // 1/m_t))) - unconditionally stable, so it sets no bound. Counting it here
    // (7 c_t per contact) was the "damping" limit: an absolute 4 N s/m on an
    // 8 mm grain read as zeta ~4.5 and cost 2622 substeps per frame where the
    // normal dashpot alone needs ~710 (measured 2026-10-08, 1.3M grains).
    // With fewer substeps the cap may engage: slip then decays in ~BUDGET
    // substeps instead of one, still far below a frame.
    const double damping_rate = budget * 2.0 * normal_damping / m;
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
        // Say what fits. Stability and accuracy substeps scale with sqrt(k):
        // the stiffness that fits the cap is k (cap / needed)^2. A smaller
        // radius at the same N/m is a stiffer material for a lighter grain
        // (mass ~ r^3), which is the usual way to get here.
        const double ratio = std::isfinite(requested) ? params.max_substeps / requested : 0.0;
        char text[320];
        std::snprintf(text, sizeof(text),
            "grain %s bound needs %.0f substeps per frame > max_substeps %d at radius %.4f m. "
            "Lower Contact stiffness to <= %.0f N/m, or raise the substep limit.",
            limiting.second, requested, params.max_substeps, params.radius_m,
            std::strcmp(limiting.second, "travel") == 0 ? params.stiffness_n_m
                : params.stiffness_n_m * ratio * ratio);
        error = text;
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
    const auto largest_particle_buffer = [](std::size_t grains) {
        return std::max({grains * kHistoryBytesPerGrain, grains * kListBytesPerGrain,
            std::size_t{kMatterGrainBucketTables} * bucketsFor(grains) *
                kMatterGrainBucketCapacity * sizeof(uint32_t)});
    };
    std::size_t capacity = grow_particles
        ? grownCapacity(runtime.capacity, count) : runtime.capacity;
    // Growth headroom must not be what breaks the device limit: 1.09M grains
    // fit a 2 GiB storage buffer, the 1.5x headroom on top of 1M did not, and
    // the step was held with room to spare. Trim toward count; count itself
    // is still checked below and reports the limit when it truly is hit.
    const std::size_t storage_limit = compute.caps().max_storage_buffer_bytes;
    while (grow_particles && storage_limit && capacity > count &&
           largest_particle_buffer(capacity) > storage_limit) {
        capacity = std::max(count, capacity - std::max<std::size_t>(capacity / 16, 1));
    }
    const std::size_t triangle_capacity = grow_collider
        ? std::max(triangles.size(), std::size_t{1}) : runtime.triangle_capacity;
    std::size_t working = particleBytes(capacity) + colliderBytes(triangle_capacity);
    if (grow_particles || grow_collider) {
        // Existing scratch remains allocated during transactional replacement.
        working += (grow_particles && runtime.capacity ? particleBytes(runtime.capacity) : 0) +
            (grow_collider && runtime.triangle_capacity
                ? colliderBytes(runtime.triangle_capacity) : 0);
    }
    const auto largest_buffer = std::max({largest_particle_buffer(capacity),
        triangle_capacity * 2 * sizeof(MatterGrainBvhNode),
        triangle_capacity * 6 * sizeof(Vec3)});
    // Say which limit and by how much: the two used to share one message, and
    // a domain budget read as a device limit sends the user to the wrong fix.
    if (budget_bytes && working > budget_bytes) {
        char text[360];
        std::snprintf(text, sizeof(text),
            "grain buffers for %zu grains need %zu MiB%s; the domain resource budget leaves %zu MiB "
            "after the liquid lane. Raise Resource Budget (MiB) on the domain (or turn Enforce off), "
            "or emit fewer grains.",
            count, working / (1024 * 1024),
            (grow_particles || grow_collider) ? " while growing (old + new buffers)" : "",
            budget_bytes / (1024 * 1024));
        error = text;
        return false;
    }
    if (compute.caps().max_storage_buffer_bytes &&
        largest_buffer > compute.caps().max_storage_buffer_bytes) {
        error = "grain buffer of " + std::to_string(largest_buffer / (1024 * 1024)) +
            " MiB for " + std::to_string(count) + " grains exceeds the device storage-buffer "
            "limit of " + std::to_string(compute.caps().max_storage_buffer_bytes / (1024 * 1024)) +
            " MiB (contact history is " + std::to_string(kHistoryBytesPerGrain) +
            " B per grain in one buffer)";
        return false;
    }
    const auto fingerprint = matterGrainColliderFingerprint(triangles, triangle_velocity);
    const bool refresh_collider = grow_collider || !runtime.collider_uploaded ||
        runtime.collider_fingerprint != fingerprint;
    MatterGrainColliderBvh collider;
    if (refresh_collider && !buildMatterGrainColliderBvh(triangles, collider, error,
            triangle_velocity, dt)) {
        return false;
    }
    if ((grow_particles && !ensureParticles(compute, runtime, capacity, error)) ||
        (grow_collider && !ensureCollider(compute, runtime, triangle_capacity, error))) {
        return false;
    }
    std::array<uint32_t, kDiagnosticWords> diagnostics{};
    if (refresh_collider) {
        runtime.collider_uploaded = false;
        // Velocities follow the end positions in the same buffer (offset 9 N
        // floats, in the push constants); none = a static collider set.
        std::vector<Vec3> flat = collider.vertices;
        flat.insert(flat.end(), collider.velocities.begin(), collider.velocities.end());
        runtime.collider_velocity_offset = collider.velocities.empty()
            ? 0u : static_cast<uint32_t>(3 * collider.vertices.size());
        if ((!flat.empty() && !compute.uploadBuffer(runtime.triangles,
                flat.data(), flat.size() * sizeof(Vec3))) ||
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
        uint32_t substep, history_blocks, tangential_stiffness_bits, last_substep;
        float wet[4];  // hash cell size, capillary prefactor, rupture cap, velocity offset bits
        float sleep[4];  // still substeps (uint bits), sleep speed (0 = off), wake all (uint bits), -
    } constants{};
    // 128 B: the push-constant size every Vulkan device guarantees.
    static_assert(sizeof(Constants) == 128);
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
    // history. Otherwise every survivor keeps its history block.
    std::string reset_reason = runtime.history_fresh
        ? (runtime.published_ids.empty() ? "first_step" : "allocation") : "";
    std::vector<uint32_t> previous_index;
    bool remap = false;
    if (!runtime.history_fresh) {
        // A grain that kept its index needs no lookup; the index is built
        // only once one did not (cell re-sort, births, removals).
        MatterParticleIdIndex published;
        bool indexed = false;
        previous_index.assign(count, 0xffffffffu);
        for (std::size_t i = 0; i < count; ++i) {
            uint32_t found = i < runtime.published_ids.size() &&
                runtime.published_ids[i] == p.particle_id[i]
                ? static_cast<uint32_t>(i) : MatterParticleIdIndex::kMissing;
            if (found == MatterParticleIdIndex::kMissing) {
                if (!indexed) {
                    // Published ids are an identity set: the step refused otherwise.
                    published.build(runtime.published_ids);
                    indexed = true;
                }
                found = published.find(p.particle_id[i]);
            }
            if (found == MatterParticleIdIndex::kMissing) {
                continue;  // birth: starts without springs
            }
            const Vec3& was = runtime.published_positions[found];
            if (std::memcmp(&was, &p.position[i], sizeof(Vec3)) != 0) {
                runtime.history_fresh = true;
                reset_reason = "host_state_changed";
                break;
            }
            previous_index[i] = found;
            remap = remap || found != i;
        }
        remap = remap && !runtime.history_fresh;
    }
    // Grain -> history block. A reset puts grain i on block i; a newborn takes
    // a block no survivor holds (count <= capacity blocks, so one is free).
    // Either way the block starts FRESH: its old records belong to someone else.
    std::vector<uint32_t> blocks(count);
    bool blocks_changed = true;
    std::size_t survivors = 0;
    if (runtime.history_fresh) {
        for (std::size_t i = 0; i < count; ++i) {
            blocks[i] = static_cast<uint32_t>(i) | kFreshHistoryBlock;
        }
    } else {
        std::vector<uint8_t> held(runtime.capacity, 0);
        bool births = false;
        for (std::size_t i = 0; i < count; ++i) {
            if (previous_index[i] != 0xffffffffu) {
                blocks[i] = runtime.history_block[previous_index[i]];
                held[blocks[i]] = 1;
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            survivors += previous_index[i] != 0xffffffffu ? 1 : 0;
        }
        std::size_t next_free = 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (previous_index[i] == 0xffffffffu) {
                while (held[next_free]) {
                    ++next_free;
                }
                held[next_free] = 1;
                blocks[i] = static_cast<uint32_t>(next_free) | kFreshHistoryBlock;
                births = true;
            }
        }
        // Same grains in the same order: the device map is already this one.
        blocks_changed = remap || births || count != runtime.history_block.size();
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
    // Unknown until this upload lands; set again below when it does.
    runtime.coupling_zero = false;
    const auto upload_start = Clock::now();
    compute.beginTransferBatch();
    // Substep 0 always rebuilds the neighbour lists: the grains were re-ordered
    // and born/absorbed on the host since the last frame.
    diagnostics[12] = 1u;
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
    if (!rows_resident) {
        uploaded = compute.uploadBuffer(runtime.coupling, coupling_rows.data(),
            coupling_rows.size() * sizeof(float)) && uploaded;
    }
    if (blocks_changed) {
        uploaded = compute.uploadBuffer(runtime.history_blocks, blocks.data(),
            count * sizeof(uint32_t)) && uploaded;
        upload_bytes += count * sizeof(uint32_t);
    }
    uploaded = compute.endTransferBatch() && uploaded;
    if (!uploaded) {
        error = "grain state/mass/identity/coupling/history-map upload failed";
        return false;
    }
    runtime.coupling_zero = zero_rows;
    report.upload_ms = ms_since(upload_start);
    // Bridges reach past contact up to the rupture distance; the hash cell
    // grows by that skin so the 27-cell search still finds every bridge.
    const float rupture_cap = params.wet_grains ? .5f * params.radius_m : 0.0f;
    // Hash cell = Verlet list cutoff: contact (2r), bridge reach and the skin
    // that lets the list stay valid until a grain has moved half of it.
    constants.wet[0] = 2.0f * params.radius_m + rupture_cap +
        kMatterGrainListSkinRadii * params.radius_m;
    constants.wet[1] = static_cast<float>(capillary);
    constants.wet[2] = rupture_cap;
    // Float slot carrying the uint offset of the collider vertex velocities.
    std::memcpy(&constants.wet[3], &runtime.collider_velocity_offset, sizeof(uint32_t));
    // Owner/mask/rest words of the blocks follow the grain -> block map.
    constants.history_blocks = static_cast<uint32_t>(runtime.capacity);
    // Sleeping grains (docs/dev/DEM_UYUYAN_TANELER.md). Off whenever something
    // can push a still grain that the step would not see: a moving collider
    // face, MPM contact impulses written between steps, the shared clock. A
    // removed grain may have been something's support: everyone wakes.
    const bool sleep_on = params.sleep && params.sleep_speed_m_s > 0.0f &&
        runtime.collider_velocity_offset == 0 && !common_driver &&
        !(mpm_contact && mpm_contact->active());
    const uint32_t still_substeps = static_cast<uint32_t>(std::clamp(
        std::ceil(double(params.sleep_time_s) / (double(dt) / substeps)), 1.0, 1e9));
    const uint32_t wake_all = !runtime.history_fresh &&
        survivors < runtime.published_ids.size() ? 1u : 0u;
    std::memcpy(&constants.sleep[0], &still_substeps, sizeof(uint32_t));
    constants.sleep[1] = sleep_on ? params.sleep_speed_m_s : 0.0f;
    std::memcpy(&constants.sleep[2], &wake_all, sizeof(uint32_t));
    constants.last_substep = substeps - 1;
    const bool history_reset = runtime.history_fresh;
    // Until this frame publishes, the device history may not match the host.
    runtime.history_fresh = true;
    const ComputeBufferHandle handles[] = {runtime.positions, runtime.velocities,
        runtime.affines, runtime.mass, runtime.ids, runtime.bucket_counts, runtime.bucket_slots,
        runtime.scratch, runtime.history, runtime.history_blocks, runtime.diagnostics,
        runtime.triangles, runtime.collider_nodes, runtime.collider_patches, runtime.coupling,
        runtime.neighbour_list, runtime.build_positions};
    constexpr std::size_t kGrainBindings = sizeof(handles) / sizeof(handles[0]);
    static_assert(kGrainBindings == 17);
    report.host_prepare_ms = ms_since(prepare_start);
    const auto gpu_start = Clock::now();
    int dispatched = 0;
    const auto dispatch_stage = [&](const char* kernel, uint32_t substep) {
        if (mpm_contact && std::strcmp(kernel, "sim_matter_grain_step") == 0 &&
            !mpm_contact->step(runtime, substep, error)) {
            return false;
        }
        ComputeDispatch command;
        command.kernel = kernel;
        command.buffers = handles;
        command.buffer_count = kGrainBindings;
        constants.substep = substep;
        command.constants = &constants;
        command.constants_size = sizeof(constants);
        // List clear: the bucket table. Hash, list build, step: one
        // invocation per grain.
        uint32_t threads = constants.count;
        if (std::strcmp(kernel, "sim_matter_grain_list_clear") == 0) {
            threads = kMatterGrainBucketTables * runtime.buckets;
        }
        setLinearGroups(command, threads);
        if (!compute.dispatch(command)) {
            error = std::string("grain stage failed: ") + kernel;
            return false;
        }
        ++dispatched;
        return true;
    };
    const uint32_t planned_grain_substeps = substeps;
    if (common_driver) {
        report.working_set_bytes = working;
        if (!common_driver->run(substeps,
                [&](uint32_t index, uint32_t total, float step_dt, std::string&) {
                    substeps = total;
                    constants.contact[0] = step_dt;
                    constants.last_substep = total - 1;
                    return dispatchMatterGrainSubstep(index, dispatch_stage);
                }, error)) {
            return false;
        }
        report.common_clock = true;
    } else if (!dispatchMatterGrainStages(substeps, dispatch_stage)) {
        return false;
    }
    // Only the three device-owned arrays come back; the rest of the grain
    // state is untouched, so it is not copied (a full FluidParticles copy per
    // frame was most of the publication cost).
    std::vector<Vec3> new_position(count), new_velocity(count);
    std::vector<AffineC> new_affine(count);
    // The substeps are only recorded so far: submit and wait for them here, so
    // download_ms is the readback alone (one extra fence per frame).
    compute.synchronize();
    const auto download_start = Clock::now();
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
    report.download_ms = ms_since(download_start);
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
            (overflow & 2u) ? "grain collider BVH traversal stack exceeded (64)" :
            (overflow & 8u) ? "grain neighbour list full (more than 32 grains within 2r + skin); "
                "grains overlap far beyond contact" :
            overflow ? "grain contact count exceeds the 24-contact history/CFL budget "
                "(or 24 history records were held in one substep); "
                "reduce emitter packing or stiffness overlap" :
            "grain publication rejected nonfinite/readback state";
        return false;
    }
    if (adaptive_contacts && diagnostics[2] > cfl_contacts) {
        // More contacts than the bound assumed: those grains may have stepped
        // past their stability limit. Nothing is published; the host state is
        // untouched and history_fresh is already set, so the re-run uploads it
        // again and runs at the full budget (once: the hint is then 24).
        runtime.cfl_contact_hint = static_cast<uint32_t>(kMatterGrainContactBudget);
        const bool rerun = stepMatterGrainGpu(p, low, high, params, dt, gravity, budget_bytes,
            compute, runtime, triangles, coupling, drag_out, report, error, motion,
            mpm_contact, common_driver);
        report.cfl_budget_retry = true;
        if (rerun) {
            report.history_reset_reason = "cfl_contact_budget_retry";
        }
        return rerun;
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
    // Parallel to published_ids: only a published frame may move the map the
    // next frame's previous_index reads through.
    runtime.history_block.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        runtime.history_block[i] = blocks[i] & ~kFreshHistoryBlock;
    }
    runtime.published_positions = p.position;
    runtime.published_velocities = p.velocity;
    runtime.published_affines = p.affine;
    runtime.published_masses = std::move(masses);
    report.substeps = static_cast<int>(substeps);
    report.dispatches = dispatched;
    report.substep_dt = dt / substeps;
    report.limit = substeps > planned_grain_substeps ? "common CFL/travel" : limiting.second;
    report.max_contacts = diagnostics[2];
    report.cfl_contacts = cfl_contacts;
    report.cfl_budget_retry = false;
    runtime.cfl_contact_hint = std::clamp(diagnostics[2] + 4u, 12u,
        static_cast<uint32_t>(kMatterGrainContactBudget));
    report.sticking_contacts = diagnostics[3];
    report.contacts = diagnostics[4];
    report.liquid_bridges = diagnostics[5];
    report.collider_manifold_truncated = diagnostics[6];
    report.neighbour_candidates = diagnostics[7];
    report.grain_pairs = diagnostics[8];
    report.history_probes = diagnostics[9];
    report.contactless_grains = diagnostics[10];
    report.collider_nodes_visited = diagnostics[11];
    report.list_rebuilds = diagnostics[15];
    report.sleeping_grains = diagnostics[16];
    report.history_reset = history_reset;
    report.history_reset_reason = reset_reason;
    report.history_remapped = remap;  // grain -> block map re-uploaded, no history copy
    report.state_resident = resident;
    report.upload_bytes = upload_bytes +
        (refresh_collider ? collider.vertices.size() * sizeof(Vec3) +
            collider.nodes.size() * sizeof(MatterGrainBvhNode) +
            collider.surface_patches.size() * sizeof(uint32_t) : 0);
    report.download_bytes = count * (2 * sizeof(Vec3) + sizeof(AffineC)) + sizeof(diagnostics) +
        (coupling && drag_out ? coupling_rows.size() * sizeof(float) : 0);
    report.transfer_batches = 2 + (refresh_collider ? 1 : 0);
    report.grains = count;
    report.working_set_bytes = working;
    report.host_publish_ms = ms_since(publish_start);
    return true;
}

} // namespace RayTrophiSim::Fluid
