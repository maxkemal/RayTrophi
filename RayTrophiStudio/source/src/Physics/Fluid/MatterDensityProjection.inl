// Included after FluidGpuPressure.inl in the simulation driver's private namespace.
bool projectMatterLiquidDensity(SimulationGridDomainState& state,
    const Fluid::APICSolverParams& physical_params, float relaxation_s, float dt,
    SimulationComputeContext* compute, SimulationGridDomainComputeBuffers* buffers,
    bool gpu_requested,
    const std::function<bool(SimulationGridDomainComputeBuffers&)>& ensure) {
    if (!(relaxation_s > 0.0f) || physical_params.granular_enabled ||
        !physical_params.free_surface || state.fluid_stats.mixed_step_held) {
        return true;
    }
    auto params = physical_params;
    params.density_relaxation_s = relaxation_s;
    params.ghost_fluid_surface = false;
    const bool gpu = gpu_requested && compute && buffers && compute->supportsDispatch() &&
        params.boundary != Fluid::APICSolverParams::BoundaryMode::Periodic;
    const bool compact = buffers && buffers->sparse_mac_transfer.compact_owner;
    const bool porous = buffers && buffers->porous_solid_velocity;
    const bool model_enabled = buffers && buffers->matter_model.enabled;
    struct RestorePolicy {
        SimulationGridDomainComputeBuffers* buffers;
        bool compact;
        bool porous;
        bool model_enabled;
        ~RestorePolicy() {
            if (buffers) {
                buffers->sparse_mac_transfer.compact_owner = compact;
                buffers->porous_solid_velocity = porous;
                buffers->matter_model.enabled = model_enabled;
                // The scratch solve uploads zero solid velocity. A subsequent
                // physical solve must publish its own moving-solid data again.
                buffers->matter_model.pressure_statics_uploaded = false;
                buffers->fluid_mask_device_valid = false;
                FluidGpuParticleUpload::invalidate(*buffers);
            }
        }
    } restore_policy{gpu ? buffers : nullptr, compact, porous, model_enabled};
    if (gpu) {
        buffers->sparse_mac_transfer.compact_owner = false;
        buffers->porous_solid_velocity = false;
        buffers->fluid_mask_device_valid = false;
    }
    bool gpu_pressure_attempted = false;
    const bool projected = Fluid::projectLiquidDensityPositions(
        state.particles, state.grid, params, dt,
        [&](const std::vector<float>& mask, const std::vector<std::size_t>& mobile,
            Fluid::APICSolverStats& stats) {
            if (gpu) {
                if (!ensure(*buffers)) {
                    return false;
                }
                gpu_pressure_attempted = true;
                // The resident mixed model normally skips both the host-zero
                // velocity upload and the correction readback. This solve must
                // use the transaction's scratch grid, not physical momentum.
                buffers->matter_model.enabled = false;
                return runGpuFluidMGPCGPressure(
                    state, params, dt, compute, *buffers, mask, &stats);
            }
            std::size_t pockets = 0, sealed = 0, interior = 0;
            Fluid::FluidParticles liquid_positions;
            liquid_positions.position.reserve(mobile.size());
            for (const auto index : mobile) {
                liquid_positions.position.push_back(state.particles.position[index]);
            }
            Fluid::projectPressureFreeSurface(liquid_positions, state.grid, params, dt,
                pockets, sealed, interior, stats);
            return std::isfinite(stats.pressure_cg_final_relative_residual);
        });
    bool restored = true;
    if (gpu_pressure_attempted) {
        // The density pass uses the pressure scratch, but later phase coupling
        // must still see the physical velocity, including on a failed solve.
        compute->beginTransferBatch();
        restored = compute->uploadBuffer(buffers->vel_x, state.grid.vel_x.data(),
            state.grid.vel_x.size() * sizeof(float));
        restored = compute->uploadBuffer(buffers->vel_y, state.grid.vel_y.data(),
            state.grid.vel_y.size() * sizeof(float)) && restored;
        restored = compute->uploadBuffer(buffers->vel_z, state.grid.vel_z.data(),
            state.grid.vel_z.size() * sizeof(float)) && restored;
        restored = compute->endTransferBatch() && restored;
        buffers->sparse_mac_transfer.compact_owner = compact;
        buffers->porous_solid_velocity = porous;
        buffers->fluid_mask_device_valid = false;
        FluidGpuParticleUpload::invalidate(*buffers);
    }
    return projected && restored;
}
