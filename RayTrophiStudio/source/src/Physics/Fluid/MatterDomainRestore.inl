void ParticleSimulationSystem::rebaseRestoredGridDomainStates() {
    const std::size_t count = std::min(grid_domain_states_.size(), grid_domains_.size());
    for (std::size_t i = 0; i < count; ++i) {
        const SimulationGridDomainDesc& domain = grid_domains_[i];
        SimulationGridDomainState& state = grid_domain_states_[i];
        if (!state.valid) continue;
        // ★★★ ManualBox ONLY, and the reason is a timing one. For ObjectBounds /
        // Adaptive the descriptor bounds are DERIVED: synchronizeGridDomains()
        // rewrites them from the resolver, and applySimSourceObjectPosesForFrame()
        // does not run until AFTER this restore. So at this instant the desc
        // still describes some other frame's pose, and rebasing onto it would
        // slide the whole bake a little on every scrub -- a drift that looks
        // like motion rather than like a bug. Those modes keep the verbatim
        // restore they always had; their bounds follow their object anyway.
        if (domain.source_mode != SimulationGridDomainSourceMode::ManualBox) continue;

        // Same expression synchronizeGridDomains() uses to derive state bounds,
        // so a domain that has NOT moved produces an exactly zero delta and
        // falls out below instead of drifting by a rounding step every scrub.
        const float padding = std::max(0.0f, domain.padding);
        const Vec3 live_min = Vec3::min(domain.bounds_min, domain.bounds_max) - Vec3(padding);
        const Vec3 live_max = Vec3::max(domain.bounds_min, domain.bounds_max) + Vec3(padding);

        const Vec3 delta = live_min - state.bounds_min;
        if (std::abs(delta.x) < 1e-6f && std::abs(delta.y) < 1e-6f && std::abs(delta.z) < 1e-6f) {
            continue;
        }

        // Pure-translation gate. A changed extent or voxel size means the grid
        // layout itself differs, so the cached cells do not correspond to the
        // live ones and shifting them would be a lie with the right shape.
        const Vec3 live_extent = live_max - live_min;
        const Vec3 cached_extent = state.bounds_max - state.bounds_min;
        const Vec3 extent_error = live_extent - cached_extent;
        if (std::abs(extent_error.x) > 1e-3f ||
            std::abs(extent_error.y) > 1e-3f ||
            std::abs(extent_error.z) > 1e-3f) {
            continue;
        }
        if (std::abs(state.grid.voxel_size - domain.voxel_size) > 1e-6f) continue;

        state.bounds_min += delta;
        state.bounds_max += delta;
        state.grid.origin += delta;

        // Fluid particles are world-space and must come along. Velocity is a
        // DIRECTION and is left alone; translating it would inject a phantom
        // drift on the first replayed step.
        for (Vec3& position : state.particles.position) position += delta;
        // ★ Material coordinates start life as a copy of the world position
        // (uvw == position for resting material) and are addressed in the same
        // world frame, so they translate with it. Leaving them behind shifts
        // every UVW-projected texture on the surface by the move distance --
        // visible as the material sliding across the liquid, not as an error.
        for (Vec3& uvw : state.particles.uvw)   uvw += delta;
        for (Vec3& uvw : state.particles.uvw_b) uvw += delta;
        for (Vec3& position : state.foam.position) position += delta;

        // The published snapshot genuinely changed; consumers key off this.
        ++state.version;
    }
}
