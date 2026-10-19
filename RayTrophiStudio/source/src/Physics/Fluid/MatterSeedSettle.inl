// Member-function fragment included by ParticleSimulation.cpp, beside
// stepGridDomains (it re-enters it). Not a standalone unit.
//
// Seed settle (2026-10-19). A seed is a voxel sub-lattice with jitter: grains
// are placed without their radius (overlaps push, gaps drop), no contact
// carries the bed's weight yet, and liquid parcels start with sub-cell density
// noise. Released at t = 0 that state slumps and splashes although the user
// asked for a bed at rest.
//
// Dynamic relaxation: the settling domains run their own steps with time,
// sources, heat, gas, foam, audio and MSF standing still (settling_), and after
// each step every carrier's velocity and APIC affine (a grain's spin lives
// there) are zeroed. The kinetic energy the seed would release is removed
// step by step instead, so what remains is the nearest rest state under
// gravity, walls, colliders and force fields. It stops when the fastest
// carrier moved slower than fluid_settle_speed_m_s in the last step (before
// zeroing) or after fluid_settle_max_steps.
//
// Known limits: animated (kinematic) colliders are not sampled while settling;
// a domain's gas lane does not step, so grains feel no wind drag change.

void ParticleSimulationSystem::settleGridDomains(const SimulationContext& context) {
    std::vector<std::size_t> settling;
    for (std::size_t i = 0; i < grid_domain_states_.size() && i < grid_domains_.size(); ++i) {
        auto& state = grid_domain_states_[i];
        if (!state.settle_stats.pending) continue;
        if (!state.valid || state.particles.empty() ||
            !simulationDomainHasLiquid(state.type)) {
            state.settle_stats.pending = false;
            continue;
        }
        state.settle_stats = SimulationSeedSettleStats{};
        state.settle_stats.pending = true;
        state.settle_stats.ran = true;
        state.settle_stats.particles = state.particles.size();
        settling.push_back(i);
    }
    if (settling.empty()) return;

    const auto settle_start = SimulationClock::now();
    SimulationContext settle_context = context;
    // The solver's own frame length, kept in the range its tuning assumes.
    settle_context.dt = std::clamp(context.dt, 1.0f / 240.0f, 1.0f / 30.0f);

    settling_ = true;
    bool any_pending = true;
    while (any_pending) {
        stepGridDomains(settle_context);
        any_pending = false;
        for (const std::size_t i : settling) {
            auto& state = grid_domain_states_[i];
            auto& stats = state.settle_stats;
            if (!stats.pending) continue;
            auto& particles = state.particles;
            float fastest_sq = 0.0f;
            for (auto& v : particles.velocity) {
                fastest_sq = std::max(fastest_sq, v.x * v.x + v.y * v.y + v.z * v.z);
                v = Vec3(0.0f, 0.0f, 0.0f);
            }
            for (auto& c : particles.affine) {
                c = Fluid::AffineC{};
            }
            const float fastest = std::sqrt(fastest_sq);
            if (stats.steps == 0) stats.first_speed_m_s = fastest;
            stats.residual_speed_m_s = fastest;
            ++stats.steps;
            const auto& domain = grid_domains_[i];
            const float target = std::max(0.0f, domain.fluid_settle_speed_m_s);
            const int limit = std::max(1, domain.fluid_settle_max_steps);
            stats.converged = fastest <= target;
            if (stats.converged || stats.steps >= limit || !state.valid ||
                particles.empty()) {
                stats.pending = false;
            } else {
                any_pending = true;
            }
        }
    }
    settling_ = false;

    const float ms = elapsedMilliseconds(settle_start, SimulationClock::now());
    for (const std::size_t i : settling) {
        auto& state = grid_domain_states_[i];
        // Whitewater is a view of motion; a relaxed bed has none to show.
        state.foam.clear();
        state.settle_stats.particles = state.particles.size();
        state.settle_stats.ms = ms;
        if (!state.settle_stats.converged) {
            SCENE_LOG_WARN("[Fluid] " + grid_domains_[i].name + ": seed settle stopped after " +
                           std::to_string(state.settle_stats.steps) + " steps at " +
                           std::to_string(state.settle_stats.residual_speed_m_s) +
                           " m/s (target " +
                           std::to_string(grid_domains_[i].fluid_settle_speed_m_s) + ")");
        }
    }
}
