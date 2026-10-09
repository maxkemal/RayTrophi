bool runGpuFluidP2G(SimulationGridDomainState& state,
                    SimulationComputeContext* compute,
                    SimulationGridDomainComputeBuffers& gpu_buffers,
                    const Fluid::APICSolverParams& fluid_params,
                    float dt,
                    bool upload_granular_state = true,
                    bool reuse_particle_inputs = false,
                    bool download_grid_velocity = true) {
    gpu_buffers.sparse_mac_transfer.used = false;
    gpu_buffers.sparse_mac_transfer.canonical = false;
    gpu_buffers.sparse_mac_transfer.snapshot_valid = false;
    gpu_buffers.sparse_mac_transfer.flip_gather_used = false;
    gpu_buffers.sparse_mac_transfer.status = "dense/reference transfer";
    auto& grid = state.grid;
    const std::size_t particle_count = state.particles.size();
    if (!compute || !compute->supportsDispatch() || particle_count == 0 ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || grid.voxel_size <= 0.0f ||
        grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty()) {
        return false;
    }
    if (particle_count > std::size_t(std::numeric_limits<int32_t>::max()) ||
        std::max({grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size()}) >
            std::size_t(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    const auto p2g_phase0 = SimulationClock::now();
    if (!ensureGpuFluidParticleBuffers(
            state, compute, gpu_buffers, false, reuse_particle_inputs)) {
        return false;
    }
    const bool granular = fluid_params.granular_enabled;
    if (granular) state.particles.ensureGranularStateSize();
    if (granular && upload_granular_state &&
        !Fluid::Granular::uploadState(*compute, state.particles,
                                      gpu_buffers.granular)) return false;
    const auto p2g_phase1 = SimulationClock::now(); // particle upload done

    ComputeBufferHandle velocity_fields[3] = {
        gpu_buffers.vel_x,
        gpu_buffers.vel_y,
        gpu_buffers.vel_z
    };
    ComputeBufferHandle weight_fields[3] = {
        gpu_buffers.temperature,
        gpu_buffers.fuel,
        gpu_buffers.scratch_scalar
    };
    FluidP2GGpuConstants constants;
    constants.nx = grid.nx;
    constants.ny = grid.ny;
    constants.nz = grid.nz;
    constants.particle_count = static_cast<int>(particle_count);
    constants.origin_x = grid.origin.x;
    constants.origin_y = grid.origin.y;
    constants.origin_z = grid.origin.z;
    constants.voxel_size = grid.voxel_size;

    constexpr uint32_t threads = 256;
    const auto groupsFor = [&](uint32_t count) {
        if (compute->backendType() == ComputeBackendType::VulkanCompute) {
            return FluidGpuDispatch::groups256(count);
        }
        return ComputeDispatchSize{(count + threads - 1u) / threads, 1u, 1u};
    };
    const auto active_window = Fluid::ActiveWindow::plan(
        state.particles.position, state.particles.velocity, grid.origin,
        grid.voxel_size, dt, grid.nx, grid.ny, grid.nz,
        !gpu_buffers.matter_model.enabled && !granular &&
        compute->backendType() == ComputeBackendType::VulkanCompute &&
        fluid_params.boundary != Fluid::APICSolverParams::BoundaryMode::Periodic);
    const bool window_normalize = active_window.bounded &&
        compute->backendType() == ComputeBackendType::VulkanCompute;
    gpu_buffers.fluid_normalize_window_cells = active_window.cells();
    gpu_buffers.fluid_normalize_window_used = window_normalize;
    bool sparse_transfer = false;
    if (grid.sparse_mode_enabled && !grid.allocate_gas_channels && !granular &&
        compute->backendType() == ComputeBackendType::VulkanCompute &&
        fluid_params.boundary != Fluid::APICSolverParams::BoundaryMode::Periodic) {
        Fluid::SparseMacTransferConstants sparse_constants;
        static_assert(sizeof(sparse_constants) == sizeof(constants));
        std::memcpy(&sparse_constants, &constants, sizeof(constants));
        std::string sparse_error;
        // A compact owner keeps the field on the pages: no dense publication.
        const bool canonical = gpu_buffers.sparse_mac_transfer.compact_owner &&
            !download_grid_velocity;
        sparse_transfer = Fluid::runSparseMacP2G(
            *compute, gpu_buffers, fluid_params, sparse_constants, canonical, sparse_error);
    }
    // The dense path (or a failed compact one) needs the dense bank. A compact
    // owner whose P2G failed has had its pages released and must reallocate it.
    for (int comp = 0; comp < 3 && !sparse_transfer; ++comp) {
        if (!velocity_fields[comp].valid() || !weight_fields[comp].valid()) {
            return false;
        }
    }
    bool ok = true;
    for (int comp = 0; comp < 3 && ok && !sparse_transfer; ++comp) {
        constants.component = comp;
        const uint32_t field_count = static_cast<uint32_t>(
            comp == 0 ? grid.vel_x.size() : (comp == 1 ? grid.vel_y.size() : grid.vel_z.size()));

        ComputeBufferHandle clear_velocity[1] = { velocity_fields[comp] };
        ComputeDispatch cmd;
        cmd.kernel = "sim_fluid_clear_float";
        cmd.buffers = clear_velocity;
        cmd.buffer_count = 1;
        cmd.constants = &constants;
        cmd.constants_size = sizeof(constants);
        cmd.groups = groupsFor(field_count);
        ok = Fluid::dispatchMatterGpuModel(*compute, cmd, gpu_buffers.matter_model);

        ComputeBufferHandle clear_weight[1] = { weight_fields[comp] };
        cmd.buffers = clear_weight;
        ok = ok && Fluid::dispatchMatterGpuModel(*compute, cmd, gpu_buffers.matter_model);

        ComputeBufferHandle scatter_buffers[5] = {
            gpu_buffers.fluid_positions,
            gpu_buffers.fluid_velocities,
            gpu_buffers.fluid_affine,
            velocity_fields[comp],
            weight_fields[comp]
        };
        cmd.kernel = "sim_fluid_p2g_scatter";
        cmd.buffers = scatter_buffers;
        cmd.buffer_count = 5;
        cmd.groups = groupsFor(static_cast<uint32_t>(constants.particle_count));
        ok = ok && Fluid::dispatchMatterGpuModel(*compute, cmd, gpu_buffers.matter_model);

        if (granular) {
            ok = ok && Fluid::Granular::dispatchStressP2G(
                *compute, gpu_buffers.granular, gpu_buffers.fluid_positions,
                velocity_fields[comp], grid.nx, grid.ny, grid.nz, comp,
                grid.origin, grid.voxel_size, dt, 1600.0f, particle_count);
        }

        cmd.kernel = "sim_fluid_p2g_normalize";
        cmd.groups = groupsFor(field_count);
        ok = ok && (window_normalize
            ? Fluid::ActiveWindow::normalize(*compute, active_window, grid.nx, grid.ny,
                                            grid.nz, comp, velocity_fields[comp],
                                            weight_fields[comp])
            : Fluid::dispatchMatterGpuModel(*compute, cmd, gpu_buffers.matter_model));
    }
    if (sparse_transfer) {
        gpu_buffers.fluid_normalize_window_used = false;
    }
    const auto p2g_phase2 = SimulationClock::now(); // dispatches recorded

    // No synchronize() here: on Vulkan the download batch records its copies
    // into the same command buffer and flushes uploads+dispatches+downloads
    // with ONE submit+fence (each extra submit costs ~0.3-1ms of WDDM latency
    // on Windows). On CUDA downloadBuffer is a blocking same-stream memcpy,
    // ordered after the kernels by the stream.
    //
    // ★ download_grid_velocity=false leaves the field DEVICE-ONLY: grid.vel_*
    // on the host is then stale, and the caller owns bringing it home before
    // any host stage reads it (see the granular resident chain in stepGridDomains).
    if (download_grid_velocity) {
        compute->beginTransferBatch();
        ok = ok &&
             compute->downloadBuffer(gpu_buffers.vel_x, grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
             compute->downloadBuffer(gpu_buffers.vel_y, grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
             compute->downloadBuffer(gpu_buffers.vel_z, grid.vel_z.data(), grid.vel_z.size() * sizeof(float));
        ok = compute->endTransferBatch() && ok;
    }

    // Phase breakdown, averaged and logged every ~240 substeps so the real
    // bottleneck (transfer vs kernel vs sync) is visible per backend without a
    // profiler. Negligible overhead; remove once the perf work settles.
    {
        static float s_up = 0.0f, s_disp = 0.0f, s_down = 0.0f;
        static int   s_n = 0;
        s_up   += elapsedMilliseconds(p2g_phase0, p2g_phase1);
        s_disp += elapsedMilliseconds(p2g_phase1, p2g_phase2);
        s_down += elapsedMilliseconds(p2g_phase2, SimulationClock::now());
        if (++s_n >= 240) {
            const float inv = 1.0f / static_cast<float>(s_n);
           /* SCENE_LOG_INFO("[FluidGPU P2G avg ms] backend=" + std::string(compute->backendName()) +
                           " particle_upload=" + std::to_string(s_up * inv) +
                           " dispatch+sync=" + std::to_string(s_disp * inv) +
                           " field_download=" + std::to_string(s_down * inv) +
                           " particles=" + std::to_string(particle_count));*/
            s_up = s_disp = s_down = 0.0f;
            s_n = 0;
        }
    }
    return ok;
}

