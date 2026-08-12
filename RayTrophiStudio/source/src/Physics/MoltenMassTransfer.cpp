#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace RayTrophiSim {
namespace {
uint32_t substanceTag(const std::string& text) {
    uint32_t hash = 2166136261u;
    for (unsigned char c : text) hash = (hash ^ c) * 16777619u;
    return hash;
}

bool contains(const SimulationGridDomainState& d, const Vec3& p) {
    return p.x >= d.bounds_min.x && p.y >= d.bounds_min.y && p.z >= d.bounds_min.z &&
           p.x <= d.bounds_max.x && p.y <= d.bounds_max.y && p.z <= d.bounds_max.z;
}

// Kinematic viscosity in m²/s of the melt, near its melting point. These were
// artistic 0..30 numbers on the old dial, which had no physical unit at all;
// the solver now integrates ν·dt/h², so they had to be restated as real
// quantities rather than rescaled.
//   molten plastic/wax are extremely viscous (10² … 10³ Pa·s over ~10³ kg/m³);
//   molten iron is famously RUNNIER than water in kinematic terms (~7e-7),
//   because its enormous density divides out its dynamic viscosity.
float moltenViscosity(const std::string& substance) {
    if (substance.find("Plastic") != std::string::npos) return 0.30f;
    if (substance.find("Wax") != std::string::npos) return 5.0e-3f;
    if (substance == "Iron" || substance == "Steel") return 1.0e-6f;
    if (substance == "Ice") return 0.0f;
    return 5.0e-3f;
}

Fluid::FluidChemistryPreset moltenChemistry(const std::string& substance) {
    if (substance.find("Plastic") != std::string::npos)
        return Fluid::FluidChemistryPreset::Plastic;
    if (substance.find("Wax") != std::string::npos)
        return Fluid::FluidChemistryPreset::Wax;
    if (substance == "Ice") return Fluid::FluidChemistryPreset::Water;
    return Fluid::FluidChemistryPreset::Inert;
}

void applyMoltenChemistry(SimulationGridDomainDesc& desc,
                          const std::string& substance) {
    desc.fluid_params.applyChemistryProfile(moltenChemistry(substance));
    const auto& chemistry = desc.fluid_params.fuel_profile;
    desc.fluid_flammable = chemistry.flammable;
    desc.fluid_extinguishing = chemistry.extinguishing;
    desc.fluid_auto_ignite = chemistry.flammable;
    desc.fluid_ignition_temperature = chemistry.flash_temperature;
    desc.fluid_evaporation_rate = chemistry.vaporization_rate;
    desc.fluid_cooling_power = chemistry.cooling_power;
    desc.fluid_oxygen_dilution = chemistry.oxygen_dilution;
    if (chemistry.extinguishing)
        desc.fluid_surface_cooling = std::max(
            desc.fluid_surface_cooling, chemistry.cooling_power);
}
}

void ParticleSimulationSystem::backupMoltenDomainDescriptor(
    const SimulationGridDomainDesc& desc) {
    if (desc.name.empty()) return;
    molten_domain_authored_backup_.emplace(desc.name, desc);
}

std::size_t ParticleSimulationSystem::restoreMoltenDomainDescriptors() {
    if (molten_domain_authored_backup_.empty()) return 0u;
    std::size_t restored = 0u;
    for (SimulationGridDomainDesc& desc : grid_domains_) {
        auto it = molten_domain_authored_backup_.find(desc.name);
        if (it == molten_domain_authored_backup_.end()) continue;
        desc = it->second;
        ++restored;
    }
    molten_domain_authored_backup_.clear();
    return restored;
}

void ParticleSimulationSystem::discardMoltenMassTransferState() {
    if (!molten_mass_transfer_queue_.empty()) {
        molten_mass_transfer_stats_.discarded_on_reset +=
            molten_mass_transfer_queue_.size();
        molten_mass_transfer_queue_.clear();
    }
    restoreMoltenDomainDescriptors();
    // The substance guard caches how far it scanned. After a reset the SoA it
    // was describing is gone, so the cache has to go with it or the next request
    // would trust a watermark past the end of a shorter particle array.
    for (SimulationGridDomainState& state : grid_domain_states_) {
        state.molten_scan_tag = 0u;
        state.molten_scan_count = 0u;
    }
}

uint64_t ParticleSimulationSystem::queueMoltenMassTransfer(
    const MoltenMassTransferRequest& request) {
    if (request.object_key.empty() || !(request.requested_mass > 0.0f) ||
        !(request.particles_per_kg > 0.0f)) return 0;
    MoltenMassTransferRequest queued = request;
    queued.sequence = next_molten_mass_transfer_sequence_++;
    molten_mass_transfer_queue_.push_back(queued);
    ++molten_mass_transfer_stats_.queued;
    material_state_fields_.requestReadback();
    return queued.sequence;
}

void ParticleSimulationSystem::queueAutomaticMoltenMassTransfers(float dt) {
    if (!(dt > 0.0f)) return;
    for (const ParticleColliderDesc& collider : colliders_) {
        if (!collider.enabled || !collider.msf_auto_transfer ||
            collider.source_name.empty() || collider.msf_transfer_domain.empty()) continue;
        if (std::any_of(molten_mass_transfer_queue_.begin(),
            molten_mass_transfer_queue_.end(), [&](const auto& pending) {
                return pending.object_key == collider.source_name;
            })) continue;
        const MaterialStateField* field = material_state_fields_.findField(collider.source_name);
        if (!field) continue;
        const MaterialMassBudgetSummary mass =
            MaterialStateFieldSystem::summarizeMassBudget(*field);
        if (!mass.valid || mass.molten_reservoir_mass <
                std::max(collider.msf_transfer_min_mass_kg, 0.0f)) continue;
        MoltenMassTransferRequest request;
        request.object_key = collider.source_name;
        request.preferred_domain = collider.msf_transfer_domain;
        request.requested_mass = std::min(
            mass.molten_reservoir_mass,
            std::max(collider.msf_transfer_rate_kg_s, 0.0f) * dt);
        request.particles_per_kg = std::max(
            collider.msf_transfer_particles_per_kg, 1.0f);
        request.max_batch_particles = std::max(
            collider.msf_transfer_max_batch_particles, 1u);
        request.velocity = collider.msf_transfer_velocity;
        request.configure_domain_chemistry = true;
        queueMoltenMassTransfer(request);
    }
}

void ParticleSimulationSystem::processMoltenMassTransfers(
    SimulationComputeContext& compute) {
    if (molten_mass_transfer_queue_.empty()) return;
    std::vector<MoltenMassTransferRequest> deferred;
    deferred.reserve(molten_mass_transfer_queue_.size());

    for (const MoltenMassTransferRequest& request : molten_mass_transfer_queue_) {
        const MaterialStateField* field = material_state_fields_.findField(request.object_key);
        if (!field || field->elementCount() == 0) {
            // The object lost its field (deleted, disabled, re-synced). Retrying
            // would never succeed, so this is a drop and must be counted as one.
            ++molten_mass_transfer_stats_.dropped;
            continue;
        }

        Vec3 center(0.0f);
        double total_area = 0.0;
        double weighted_temperature = 0.0;
        std::vector<std::size_t> molten_elements;
        for (std::size_t i = 0; i < field->elementCount(); ++i) {
            const float area = std::max(field->centers[i * 4u + 3u], 0.0f);
            const float melt = std::clamp(
                field->state[i * MaterialStateField::kStateStride + 5u], 0.0f, 1.0f);
            if (!(area > 0.0f) || !(melt > 1.0e-5f)) continue;
            const float weight = area * melt;
            center += Vec3(field->centers[i * 4u], field->centers[i * 4u + 1u],
                           field->centers[i * 4u + 2u]) * weight;
            weighted_temperature +=
                field->state[i * MaterialStateField::kStateStride] * weight;
            total_area += weight;
            molten_elements.push_back(i);
        }
        if (total_area <= 0.0) {
            // Nothing on the surface is molten right now. Unlike "no domain",
            // there is no spawn position to defer TO — the reservoir keeps the
            // mass and the automatic pass re-queues once melt reappears.
            ++molten_mass_transfer_stats_.dropped;
            continue;
        }
        center *= static_cast<float>(1.0 / total_area);
        const float temperature_kelvin = world_thermal_.scale().toKelvin(
            static_cast<float>(weighted_temperature / total_area));

        std::size_t domain_index = grid_domains_.size();
        for (std::size_t i = 0; i < grid_domains_.size() && i < grid_domain_states_.size(); ++i) {
            const auto& desc = grid_domains_[i];
            const auto& state = grid_domain_states_[i];
            if (!desc.enabled || desc.type != SimulationDomainType::Fluid ||
                !state.valid || !contains(state, center)) continue;
            if (!request.preferred_domain.empty() && desc.name != request.preferred_domain) continue;
            domain_index = i;
            break;
        }
        if (domain_index == grid_domains_.size() && !request.preferred_domain.empty() &&
            !request.configure_domain_chemistry) {
            for (std::size_t i = 0; i < grid_domains_.size() && i < grid_domain_states_.size(); ++i) {
                if (grid_domains_[i].enabled && grid_domains_[i].type == SimulationDomainType::Fluid &&
                    grid_domain_states_[i].valid && contains(grid_domain_states_[i], center)) {
                    domain_index = i;
                    break;
                }
            }
        }
        if (domain_index == grid_domains_.size()) {
            ++molten_mass_transfer_stats_.deferred_no_domain;
            deferred.push_back(request);
            continue;
        }

        auto& desc = grid_domains_[domain_index];
        auto& state = grid_domain_states_[domain_index];
        const uint32_t tag = substanceTag(field->substance_name);
        if (request.configure_domain_chemistry) {
            // Incremental substance guard: only particles appended since the last
            // clean scan can be carrying a different tag. A changed tag or a SoA
            // that shrank invalidates the watermark and forces a full rescan.
            const std::size_t tag_count = state.particles.substance_tag.size();
            std::size_t scan_begin = 0u;
            if (state.molten_scan_tag == tag && state.molten_scan_count <= tag_count)
                scan_begin = state.molten_scan_count;
            const bool incompatible = std::any_of(
                state.particles.substance_tag.begin() +
                    static_cast<std::ptrdiff_t>(scan_begin),
                state.particles.substance_tag.end(),
                [tag](uint32_t existing) { return existing != tag; });
            if (incompatible) {
                state.molten_scan_tag = 0u;
                state.molten_scan_count = 0u;
                ++molten_mass_transfer_stats_.deferred_no_domain;
                deferred.push_back(request);
                continue;
            }
            state.molten_scan_tag = tag;
            state.molten_scan_count = tag_count;
            // ★ From here the descriptor stops being purely authored data. Take
            // the authored copy BEFORE the first write, so a reset can undo it.
            backupMoltenDomainDescriptor(desc);
            if (state.particles.empty()) {
                if (desc.fluid_params.kinematic_viscosity <= 0.0f)
                    desc.fluid_params.kinematic_viscosity =
                        moltenViscosity(field->substance_name);
                // Sweeps track how stiff the implicit system is: ν·dt/h² grows
                // with viscosity, and an under-converged solve under-applies it.
                desc.fluid_params.viscosity_sweeps =
                    desc.fluid_params.kinematic_viscosity >= 0.1f  ? 24 :
                    desc.fluid_params.kinematic_viscosity >= 1.0e-3f ? 16 : 6;
                // A melt wets what it runs over — no-slip, not free-slip.
                desc.fluid_params.viscosity_wall_slip = 0.0f;
                desc.fluid_params.current_preset =
                    Fluid::APICSolverParams::FluidPreset::Custom;
                desc.fluid_render_mode = Fluid::FluidRenderMode::SurfaceSDF;
            }
            applyMoltenChemistry(desc, field->substance_name);
        }
        const std::size_t capacity = desc.fluid_max_particles > state.particles.size()
            ? desc.fluid_max_particles - state.particles.size() : 0u;
        if (capacity == 0u) {
            ++molten_mass_transfer_stats_.deferred_no_capacity;
            deferred.push_back(request);
            continue;
        }
        const std::size_t desired = std::max<std::size_t>(1u,
            static_cast<std::size_t>(std::ceil(request.requested_mass * request.particles_per_kg)));
        const std::size_t batch_limit = request.max_batch_particles > 0u
            ? request.max_batch_particles : std::numeric_limits<std::size_t>::max();
        const std::size_t spawn_count = std::min({desired, capacity, batch_limit});
        const float spawn_mass = request.requested_mass *
            static_cast<float>(spawn_count) / static_cast<float>(desired);
        const SubstanceProfile& profile = findSubstance(field->substance_name);
        const float combustible = profile.combustible ? 1.0f : 0.0f;
        const float h = std::max(state.voxel_size, 1.0e-3f) * 0.35f;

        // A material-transfer particle born on a closed object collider is its own
        // collider's prisoner: collision projects it to the outer skin and
        // gravity immediately presses it back onto that same skin. OBB appeared
        // to work only because its coarse/updating bounds accidentally provided
        // an escape. The molten reservoir represents material that has ALREADY
        // left the solid phase, so route it to a gravity-side outlet just beyond
        // the source collider. Other fluid still collides with the complete SDF.
        bool has_source_collider_outlet = false;
        Vec3 source_bounds_min(0.0f), source_bounds_max(0.0f);
        for (const ParticleColliderDesc& collider : colliders_) {
            const bool closed_object_collider =
                collider.source_mode == ParticleColliderSourceMode::ObjectAABB ||
                collider.source_mode == ParticleColliderSourceMode::ObjectOBB ||
                collider.source_mode == ParticleColliderSourceMode::ObjectMeshSDF ||
                collider.source_mode == ParticleColliderSourceMode::ObjectMeshBVH ||
                collider.source_mode == ParticleColliderSourceMode::ObjectConvexDecomp;
            if (!collider.enabled || collider.source_name != request.object_key ||
                !closed_object_collider)
                continue;
            if (collider_bounds_resolver_ &&
                collider_bounds_resolver_(collider, source_bounds_min, source_bounds_max)) {
                const Vec3 ordered_min = Vec3::min(source_bounds_min, source_bounds_max);
                const Vec3 ordered_max = Vec3::max(source_bounds_min, source_bounds_max);
                source_bounds_min = ordered_min;
                source_bounds_max = ordered_max;
                has_source_collider_outlet = true;
            }
            break;
        }

        Vec3 gravity_outlet = center;
        if (has_source_collider_outlet) {
            // APIC collision is cell based. Merely crossing the analytic surface
            // by half a voxel can still map the particle into the last solid
            // cell, especially after its deterministic packet jitter. Keep the
            // packet centre more than one full cell beyond the collider bounds.
            const float clearance = std::max(h * 2.5f, state.voxel_size * 1.25f);
            gravity_outlet.y = source_bounds_min.y - clearance;
            // If the fluid domain does not extend below the object, select the
            // nearest horizontal AABB exit instead of clamping the particle back
            // inside the source SDF. Outside the source AABB is necessarily
            // outside its cooked SDF, independent of mesh concavity/sign noise.
            if (gravity_outlet.y < state.bounds_min.y + h) {
                struct ExitCandidate { Vec3 p; float distance; };
                const ExitCandidate exits[] = {
                    { Vec3(source_bounds_min.x - clearance, center.y, center.z),
                      std::abs(center.x - source_bounds_min.x) },
                    { Vec3(source_bounds_max.x + clearance, center.y, center.z),
                      std::abs(source_bounds_max.x - center.x) },
                    { Vec3(center.x, center.y, source_bounds_min.z - clearance),
                      std::abs(center.z - source_bounds_min.z) },
                    { Vec3(center.x, center.y, source_bounds_max.z + clearance),
                      std::abs(source_bounds_max.z - center.z) }
                };
                float best = std::numeric_limits<float>::max();
                for (const ExitCandidate& exit : exits) {
                    if (!contains(state, exit.p) || exit.distance >= best) continue;
                    gravity_outlet = exit.p;
                    best = exit.distance;
                }
                if (best == std::numeric_limits<float>::max())
                    has_source_collider_outlet = false;
            }
        }
        const std::size_t old_size = state.particles.size();
        for (std::size_t i = 0; i < spawn_count; ++i) {
            const float x = (static_cast<float>((i * 17u) % 11u) / 10.0f - 0.5f) * h;
            const float y = (static_cast<float>((i * 29u) % 13u) / 12.0f - 0.5f) * h;
            const float z = (static_cast<float>((i * 43u) % 17u) / 16.0f - 0.5f) * h;
            // Keep each 8-particle packet spatially coherent. One particle per
            // far-apart UV texel conserves mass but never crosses the APIC/SDF
            // reconstruction threshold, while >16 in one cell is deliberately
            // trimmed by the APIC reseed cap.
            const std::size_t element = molten_elements[(i / 8u) % molten_elements.size()];
            Vec3 p(field->centers[element * 4u], field->centers[element * 4u + 1u],
                   field->centers[element * 4u + 2u]);
            if (has_source_collider_outlet) {
                // Preserve the packet's small deterministic spread but keep its
                // centre at the selected outlet. This avoids a single coincident
                // APIC clump without putting samples back inside the collider.
                p = gravity_outlet;
            }
            p += Vec3(x, y, z);
            p.x = std::clamp(p.x, state.bounds_min.x + h, state.bounds_max.x - h);
            p.y = std::clamp(p.y, state.bounds_min.y + h, state.bounds_max.y - h);
            p.z = std::clamp(p.z, state.bounds_min.z + h, state.bounds_max.z - h);
            state.particles.emit(p, request.velocity, temperature_kelvin, combustible, tag);
        }

        float consumed = 0.0f;
        if (!material_state_fields_.consumeMoltenMass(
                request.object_key, spawn_mass, compute, consumed) ||
            std::abs(consumed - spawn_mass) > std::max(1.0e-5f, spawn_mass * 1.0e-3f)) {
            while (state.particles.size() > old_size)
                state.particles.removeSwap(state.particles.size() - 1u);
            // The rollback has to reach the substance watermark too: it was moved
            // forward to cover particles that no longer exist, and leaving it
            // there would let a foreign tag slip past the next guard unscanned.
            if (state.molten_scan_count > old_size)
                state.molten_scan_count = old_size;
            // The mass is still in the reservoir and the object is still molten,
            // so this is a transient failure (a failed upload, a partial debit) —
            // retry it rather than losing the request.
            ++molten_mass_transfer_stats_.dropped;
            deferred.push_back(request);
            continue;
        }
        ++state.version;
        ++molten_mass_transfer_stats_.completed;
        molten_mass_transfer_stats_.requested_mass += request.requested_mass;
        molten_mass_transfer_stats_.transferred_mass += consumed;
        molten_mass_transfer_stats_.spawned_particles += spawn_count;
        molten_mass_transfer_stats_.last_object = request.object_key;
        molten_mass_transfer_stats_.last_domain = desc.name;
        molten_mass_transfer_stats_.last_substance = field->substance_name;
        molten_mass_transfer_stats_.last_temperature_kelvin = temperature_kelvin;
        molten_mass_transfer_stats_.last_combustible_fraction = combustible;
        if (spawn_count < desired && request.requested_mass > consumed) {
            MoltenMassTransferRequest remainder = request;
            remainder.requested_mass -= consumed;
            deferred.push_back(std::move(remainder));
        }
    }
    molten_mass_transfer_queue_ = std::move(deferred);
}
} // namespace RayTrophiSim
