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
// gravity, walls, colliders and force fields.
//
// ★ The stop rule is DISPLACEMENT, not speed. The first live run (2026-10-19)
// stopped on "fastest carrier < 1 cm/s" and never got there: one step from
// rest always returns ~0.1 m/s at a few surface/wall carriers (the solver's
// response to gravity there), so all 240 steps were paid and the bed looked
// no different. Now: the 95th percentile of per-step displacement must fall
// under fluid_settle_displacement_m (settled), or stop falling (plateau: the
// solver's own floor, more steps cannot remove it), or the step limit ends it.
//
// Known limits: animated (kinematic) colliders are not sampled while settling;
// a domain's gas lane does not step, so grains feel no wind drag change; pore
// water exchange is not paused.

namespace {

constexpr int kSettlePlateauWindow = 10;      // steps compared for "still falling"
constexpr float kSettlePlateauRatio = 0.9f;   // < 10% drop over the window = plateau
constexpr std::size_t kMotionProbeSamples = 4096;

// Positions of one population, keyed for matching after a step.
struct MotionSnapshot {
    std::unordered_map<uint64_t, Vec3> by_id;  // empty when ids are absent or repeat
    std::vector<Vec3> by_index;
};

void takeMotionSnapshot(const Fluid::FluidParticles& particles, MotionSnapshot& out) {
    out.by_id.clear();
    out.by_index.assign(particles.position.begin(), particles.position.end());
    const std::size_t n = particles.size();
    if (particles.particle_id.size() != n) return;
    out.by_id.reserve(n);
    for (std::size_t k = 0; k < n; ++k) {
        out.by_id.emplace(particles.particle_id[k], particles.position[k]);
    }
    // A fresh seed can carry no ids yet (or all zero): identity cannot match.
    if (out.by_id.size() != n) out.by_id.clear();
}

// Appends each carrier's displacement since the snapshot. A carrier with no
// partner is skipped; an empty result means "not measured".
void matchMotion(const Fluid::FluidParticles& particles, const MotionSnapshot& before,
                 std::vector<float>& moved) {
    const std::size_t n = particles.size();
    const auto distance = [](const Vec3& a, const Vec3& b) {
        const Vec3 d = a - b;
        return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    };
    if (!before.by_id.empty() && particles.particle_id.size() == n) {
        for (std::size_t k = 0; k < n; ++k) {
            const auto found = before.by_id.find(particles.particle_id[k]);
            if (found != before.by_id.end()) {
                moved.push_back(distance(particles.position[k], found->second));
            }
        }
        if (!moved.empty()) return;
    }
    if (before.by_index.size() == n) {
        for (std::size_t k = 0; k < n; ++k) {
            moved.push_back(distance(particles.position[k], before.by_index[k]));
        }
    }
}

float percentile95(std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    const std::size_t k = (values.size() * 95) / 100;
    std::nth_element(values.begin(), values.begin() + std::min(k, values.size() - 1),
                     values.end());
    return values[std::min(k, values.size() - 1)];
}

// Per-frame displacement of a sample of carriers, matched by particle id so a
// cell re-sort, a birth or a removal does not read as motion. Measures what
// the carriers DID, independent of what their velocity column says.
void measureFrameMotion(SimulationGridDomainState& state) {
    auto& probe = state.motion_probe;
    const auto& particles = state.particles;
    std::unordered_map<uint64_t, Vec3> previous;
    previous.reserve(probe.ids.size());
    for (std::size_t k = 0; k < probe.ids.size() && k < probe.positions.size(); ++k) {
        previous.emplace(probe.ids[k], probe.positions[k]);
    }
    // Without ids (or with repeats) the sample is matched by index, which is
    // only meaningful while the population size is unchanged.
    const bool by_id = particles.particle_id.size() == particles.size() &&
                       previous.size() == probe.ids.size();
    const bool same_count = probe.population == particles.size();
    probe.ids.clear();
    probe.positions.clear();
    probe.population = particles.size();
    const std::size_t count = particles.size();
    const std::size_t stride = std::max<std::size_t>(1, count / kMotionProbeSamples);
    std::vector<float> moved;
    moved.reserve(count / stride + 1);
    double sum = 0.0;
    float largest = 0.0f;
    std::size_t slot = 0;
    for (std::size_t i = 0; i < count; i += stride, ++slot) {
        const uint64_t id = i < particles.particle_id.size() ? particles.particle_id[i] : 0u;
        const Vec3& now = particles.position[i];
        const Vec3* then = nullptr;
        if (by_id) {
            const auto found = previous.find(id);
            if (found != previous.end()) then = &found->second;
        } else if (same_count && slot < probe.index_positions.size()) {
            then = &probe.index_positions[slot];
        }
        if (then) {
            const Vec3 d = now - *then;
            const float distance = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
            moved.push_back(distance);
            sum += distance;
            largest = std::max(largest, distance);
        }
        probe.ids.push_back(id);
        probe.positions.push_back(now);
    }
    probe.index_positions = probe.positions;
    probe.samples = moved.size();
    probe.mean_m = moved.empty() ? 0.0f : static_cast<float>(sum / double(moved.size()));
    probe.max_m = largest;
    probe.p95_m = percentile95(moved);
}

} // namespace

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

    // Per domain: positions before the step, and the p95 history.
    std::vector<MotionSnapshot> before(settling.size());
    std::vector<std::vector<float>> history(settling.size());
    std::vector<float> moved;

    settling_ = true;
    bool any_pending = true;
    while (any_pending) {
        for (std::size_t s = 0; s < settling.size(); ++s) {
            const auto& state = grid_domain_states_[settling[s]];
            if (!state.settle_stats.pending) continue;
            takeMotionSnapshot(state.particles, before[s]);
        }
        stepGridDomains(settle_context);
        any_pending = false;
        for (std::size_t s = 0; s < settling.size(); ++s) {
            const std::size_t i = settling[s];
            auto& state = grid_domain_states_[i];
            auto& stats = state.settle_stats;
            if (!stats.pending) continue;
            auto& particles = state.particles;
            moved.clear();
            matchMotion(particles, before[s], moved);
            for (auto& v : particles.velocity) v = Vec3(0.0f, 0.0f, 0.0f);
            for (auto& c : particles.affine) c = Fluid::AffineC{};
            // ★ No matched carrier = not measured, never "at rest" (5th build:
            // a fresh seed had no ids yet, p95 of nothing read 0 and the
            // water "converged" after one step).
            const bool measured = !moved.empty();
            const float p95 = percentile95(moved);
            if (stats.steps == 0) stats.first_p95_m = p95;
            stats.residual_p95_m = p95;
            stats.measured_steps += measured ? 1 : 0;
            ++stats.steps;
            if (measured) history[s].push_back(p95);
            const auto& domain = grid_domains_[i];
            const float tolerance = std::max(0.0f, domain.fluid_settle_displacement_m);
            const int limit = std::max(1, domain.fluid_settle_max_steps);
            stats.converged = measured && p95 <= tolerance;
            const auto& h = history[s];
            stats.plateau = !stats.converged && h.size() > kSettlePlateauWindow &&
                h.back() > kSettlePlateauRatio * h[h.size() - 1 - kSettlePlateauWindow];
            if (stats.converged || stats.plateau || stats.steps >= limit ||
                !state.valid || particles.empty()) {
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
        // The frame probe starts from the settled positions.
        state.motion_probe = SimulationMotionProbe{};
        state.settle_stats.particles = state.particles.size();
        state.settle_stats.ms = ms;
        if (!state.settle_stats.converged) {
            SCENE_LOG_INFO("[Fluid] " + grid_domains_[i].name + ": seed settle " +
                           (state.settle_stats.plateau ? "reached the solver floor" :
                                                         "hit the step limit") +
                           " after " + std::to_string(state.settle_stats.steps) +
                           " steps, p95 displacement " +
                           std::to_string(state.settle_stats.residual_p95_m) + " m/step");
        }
    }
}
