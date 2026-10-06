#include "Fluid/MatterModelBatch.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace RayTrophiSim::Fluid {
namespace {

void appendSnapshot(FluidParticles& target, const FluidParticles& source, std::size_t index) {
    // emit initializes all optional sidecars; the snapshot replaces that slot
    // including identity. Temporary allocation is never committed as birth.
    target.emit(source.position[index], Vec3(0.0f));
    target.copyParticleFrom(target.size() - 1, source, index);
}

} // namespace

bool stepMatterModelBatch(FluidParticles& particles, FluidSim::FluidGrid& liquid_grid,
                         const std::array<APICSolverParams, 2>& model_params,
                         MatterConstitutiveModel legacy_model, float dt,
                         const SimulationForceFieldSnapshot* forces, float time_seconds,
                         const MatterBatchContact& contact, MatterModelBatchResult& result,
                         std::string& error) {
    error.clear();
    if (!contact) {
        error = "mixed model step requires a MAC contact service";
        return false;
    }
    if (particles.particle_id.size() != particles.size()) {
        error = "mixed model step requires canonical particle identities";
        return false;
    }
    const std::size_t count = particles.size();
    if (!std::isfinite(dt) || dt <= 0.0f || !std::isfinite(time_seconds)) {
        error = "mixed model step requires finite time and positive dt";
        return false;
    }
    if (particles.velocity.size() != count || particles.affine.size() != count ||
        particles.flags.size() != count || particles.mass_fraction.size() != count ||
        particles.rest_mass_kg.size() != count || particles.next_particle_id == 0) {
        error = "mixed model step requires complete canonical mechanical sidecars";
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        const auto& position = particles.position[index];
        const auto& velocity = particles.velocity[index];
        const float fraction = particles.mass_fraction[index];
        const float mass = particles.rest_mass_kg[index];
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
            !std::isfinite(velocity.y) || !std::isfinite(velocity.z) ||
            !std::isfinite(fraction) || fraction < 0.0f || fraction > 1.0f ||
            !std::isfinite(mass) || mass <= 0.0f ||
            particles.particle_id[index] >= particles.next_particle_id) {
            error = "mixed model step received invalid canonical particle data";
            return false;
        }
    }
    std::array<FluidParticles, 2> models;
    std::unordered_map<uint64_t, std::size_t> original_order;
    original_order.reserve(particles.size());
    for (std::size_t index = 0; index < particles.size(); ++index) {
        const uint64_t identity = particles.particle_id[index];
        if (identity == 0 || !original_order.emplace(identity, index).second) {
            error = "mixed model step requires unique nonzero identities";
            return false;
        }
        auto model = index < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[index])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Auto) {
            model = legacy_model;
        }
        if (model != MatterConstitutiveModel::Fluid && model != MatterConstitutiveModel::Granular) {
            error = "mixed CPU coordinator supports only resolved fluid/granular models";
            return false;
        }
        const std::size_t lane = model == MatterConstitutiveModel::Fluid ? 0 : 1;
        appendSnapshot(models[lane], particles, index);
        models[lane].constitutive_model.back() = static_cast<uint8_t>(model);
    }
    if (models[0].empty() || models[1].empty()) {
        error = "mixed coordinator requires both models; use the single-model fast path";
        return false;
    }
    for (auto& model : models) {
        model.uvw_step = particles.uvw_step;
        model.uvw_refresh_period = particles.uvw_refresh_period;
        // This also keeps the stage topology stamp in the domain's allocator
        // epoch; partition creation did not produce canonical new particles.
        model.next_particle_id = particles.next_particle_id;
    }
    std::array<FluidSim::FluidGrid, 2> grids{liquid_grid, liquid_grid};
    std::array<MatterModelGridStep, 2> workspaces;
    MatterModelBatchResult candidate;
    for (std::size_t lane = 0; lane < 2; ++lane) {
        candidate.before_count[lane] = models[lane].size();
        auto params = model_params[lane];
        params.granular_enabled = lane == 1;
        if (!prepareMatterModelGrid(models[lane], grids[lane], params, dt,
                                    forces, time_seconds, workspaces[lane], error)) {
            return false;
        }
    }
    if (!contact(models[0], grids[0], models[1], grids[1], error)) {
        return false;
    }
    for (std::size_t lane = 0; lane < 2; ++lane) {
        if (!finishMatterModelGrid(workspaces[lane], candidate.model_stats[lane], error)) {
            return false;
        }
        candidate.after_count[lane] = models[lane].size();
    }
    if (models[0].uvw_step != models[1].uvw_step) {
        error = "mixed models did not advance the same material-coordinate epoch";
        return false;
    }
    struct Survivor {
        std::size_t original;
        std::size_t lane;
        std::size_t index;
    };
    std::vector<Survivor> survivors;
    std::unordered_set<uint64_t> seen;
    for (std::size_t lane = 0; lane < 2; ++lane) {
        for (std::size_t index = 0; index < models[lane].size(); ++index) {
            const uint64_t identity = models[lane].particle_id[index];
            const auto original = original_order.find(identity);
            if (original == original_order.end() || !seen.insert(identity).second) {
                error = "model tail created or duplicated a canonical particle identity";
                return false;
            }
            survivors.push_back({original->second, lane, index});
        }
    }
    std::sort(survivors.begin(), survivors.end(), [](const Survivor& a, const Survivor& b) {
        return a.original < b.original;
    });
    FluidParticles merged;
    merged.reserve(survivors.size());
    for (const auto& survivor : survivors) {
        appendSnapshot(merged, models[survivor.lane], survivor.index);
    }
    merged.next_particle_id = particles.next_particle_id;
    merged.uvw_step = models[0].uvw_step;
    merged.uvw_refresh_period = particles.uvw_refresh_period;
    candidate.removed_particles = particles.size() - merged.size();
    particles = std::move(merged);
    liquid_grid = std::move(grids[0]);
    result = candidate;
    return true;
}

} // namespace RayTrophiSim::Fluid
