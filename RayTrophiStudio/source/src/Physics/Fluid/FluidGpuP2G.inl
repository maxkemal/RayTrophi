// ★ Snow's FLIP baseline (Stomakhin 2013 runs 95 % FLIP). The granular lane
// was pure PIC/APIC: the grid average smoothed away the velocity differences
// a crack opens from, and the affine modes of an elastic body never decayed.
// The baseline must be the transferred particle velocity field WITHOUT the
// stress impulse, which P2G adds into the same accumulator before the
// normalisation -- so it is copied out between the scatter and the stress
// pass, and normalised by the same weights. Matter path only (the single-model
// granular path keeps FLIP off, see FluidDomainStep.inl); both this capture
// and the G2P consumer ask this one predicate.
bool granularFlipSnapshotWanted(const Fluid::APICSolverParams& params,
                                const SimulationGridDomainComputeBuffers& buffers) {
    return params.granular_enabled &&
        (params.granular_snow_plasticity || params.elastic_solid) &&
        params.flip_blend > 0.0f && buffers.matter_model.enabled &&
        buffers.scratch_vel_x.valid() && buffers.scratch_vel_y.valid() &&
        buffers.scratch_vel_z.valid();
}

// The MPM lane's whole P2G in three dispatches (sim_matter_p2g_fused.comp):
// clear, scatter (momentum + weight + stress impulse of all three components),
// normalize (velocity and, with FLIP, the stress-free baseline). Replaces the
// per-component clear/clear/scatter/copy/stress/normalize/normalize chain.
static bool runGpuMatterFusedP2G(SimulationComputeContext& compute,
                                 SimulationGridDomainComputeBuffers& buffers,
                                 const FluidSim::FluidGrid& grid,
                                 std::size_t particle_count, float dt, bool flip_snapshot) {
    const auto& model = buffers.matter_model;
    const auto& g = buffers.granular;
    if (!model.indices.valid() || !model.counters.valid() || !model.rest_mass.valid() ||
        !model.mass_fraction.valid() || !model.dry_volume.valid() ||
        !g.stress_diag.valid() || !g.stress_shear.valid() ||
        !buffers.vel_x.valid() || !buffers.vel_y.valid() || !buffers.vel_z.valid() ||
        !buffers.temperature.valid() || !buffers.fuel.valid() || !buffers.scratch_scalar.valid()) {
        return false;
    }
    // Without FLIP the stress goes straight into the momentum and the baseline
    // slots are never touched (flag bit 0): the velocity fields stand in.
    const std::array<ComputeBufferHandle, 3> stress_slots = flip_snapshot
        ? std::array<ComputeBufferHandle, 3>{buffers.scratch_vel_x, buffers.scratch_vel_y,
                                             buffers.scratch_vel_z}
        : std::array<ComputeBufferHandle, 3>{buffers.vel_x, buffers.vel_y, buffers.vel_z};
    struct Constants {
        int nx, ny, nz, particle_count, pass;
        uint32_t flags;
        float origin_x, origin_y, origin_z, voxel_size, dt;
        uint32_t matter_lane;
    } constants{grid.nx, grid.ny, grid.nz, static_cast<int>(particle_count), 0,
                flip_snapshot ? 1u : 0u, grid.origin.x, grid.origin.y, grid.origin.z,
                grid.voxel_size, dt, model.lane};
    static_assert(sizeof(Constants) == 48);
    const ComputeBufferHandle bindings[] = {
        buffers.fluid_positions, buffers.fluid_velocities, buffers.fluid_affine,
        buffers.vel_x, buffers.vel_y, buffers.vel_z,
        buffers.temperature, buffers.fuel, buffers.scratch_scalar,
        stress_slots[0], stress_slots[1], stress_slots[2],
        g.stress_diag, g.stress_shear,
        model.indices, model.counters, model.rest_mass, model.mass_fraction, model.dry_volume};
    const uint32_t faces = static_cast<uint32_t>(
        std::max({grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size()}));
    ComputeDispatch command;
    command.kernel = "sim_matter_p2g_fused";
    command.buffers = bindings;
    command.buffer_count = 19;
    command.constants = &constants;
    command.constants_size = sizeof(constants);
    // Shared-memory block scatter (S5, docs/dev/OLCEK_MILYON_PARCACIK.md): a
    // per-substep counting sort by 4^3-cell block (a permutation; the parcel
    // order is untouched), then one workgroup per block. A device without
    // shared float atomics refuses the block kernel at its first dispatch; the
    // plain fused scatter (pass 1) then stays for the session.
    // RT_P2G_BLOCK_OFF in the environment forces the plain scatter: the A arm
    // of a parity run on the same device (the fallback must stay correct anyway).
    static bool s_block_refused = std::getenv("RT_P2G_BLOCK_OFF") != nullptr;
    const auto block_scatter = [&]() -> int {  // 1 done, 0 refused, -1 failed
        if (s_block_refused) return 0;
        constexpr int kBlock = 4;
        const int bx = (grid.nx + kBlock - 1) / kBlock;
        const int by = (grid.ny + kBlock - 1) / kBlock;
        const int bz = (grid.nz + kBlock - 1) / kBlock;
        const uint64_t blocks = uint64_t(bx) * by * bz;
        if (blocks == 0 || blocks > uint64_t(std::numeric_limits<int32_t>::max() / 4)) return 0;
        static ComputeBufferHandle s_meta{}, s_items{};
        const auto ensure = [&](ComputeBufferHandle& h, std::size_t bytes, const char* name) {
            if (h.valid() && compute.getBufferSize(h) >= bytes) return true;
            if (h.valid()) compute.destroyBuffer(h);
            ComputeBufferDesc desc;
            desc.debug_name = name;
            desc.size_bytes = bytes;
            desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite;
            h = compute.createBuffer(desc);
            return h.valid();
        };
        if (!ensure(s_meta, (3 * blocks + 1) * sizeof(uint32_t), "matter_p2g_block_meta") ||
            !ensure(s_items, std::max<std::size_t>(particle_count, 1) * sizeof(uint32_t),
                    "matter_p2g_block_items")) {
            return -1;
        }
        struct SortConstants {
            int nx, ny, nz, particle_count, pass;
            uint32_t matter_lane;
            float origin_x, origin_y, origin_z, voxel_size;
            int bx, by, bz, pad0, pad1, pad2;
        } sort{grid.nx, grid.ny, grid.nz, static_cast<int>(particle_count), 0, model.lane,
               grid.origin.x, grid.origin.y, grid.origin.z, grid.voxel_size, bx, by, bz, 0, 0, 0};
        static_assert(sizeof(SortConstants) == 64);
        const ComputeBufferHandle sort_bindings[] = {
            buffers.fluid_positions, model.indices, model.counters, s_meta, s_items};
        ComputeDispatch sort_command;
        sort_command.kernel = "sim_matter_p2g_block_sort";
        sort_command.buffers = sort_bindings;
        sort_command.buffer_count = 5;
        sort_command.constants = &sort;
        sort_command.constants_size = sizeof(sort);
        for (int pass = 0; pass < 4; ++pass) {
            sort.pass = pass;
            sort_command.groups = pass == 2 ? ComputeDispatchSize{}
                : FluidGpuDispatch::groups256(pass == 0 ? static_cast<uint32_t>(blocks)
                                                        : static_cast<uint32_t>(particle_count));
            if (!compute.dispatch(sort_command)) return -1;
        }
        struct BlockConstants {
            Constants base;
            int bx, by, bz, pad;
        } block{constants, bx, by, bz, 0};
        static_assert(sizeof(BlockConstants) == 64);
        block.base.pass = 1;
        ComputeBufferHandle block_bindings[21];
        std::copy(std::begin(bindings), std::end(bindings), block_bindings);
        block_bindings[19] = s_meta;
        block_bindings[20] = s_items;
        ComputeDispatch block_command;
        block_command.kernel = "sim_matter_p2g_block";
        block_command.buffers = block_bindings;
        block_command.buffer_count = 21;
        block_command.constants = &block;
        block_command.constants_size = sizeof(block);
        constexpr uint32_t kMaxGroups = 65535;
        block_command.groups.groups_x = static_cast<uint32_t>(std::min<uint64_t>(blocks, kMaxGroups));
        block_command.groups.groups_y = static_cast<uint32_t>((blocks + kMaxGroups - 1) / kMaxGroups);
        if (!compute.dispatch(block_command)) {
            // The kernel refuses before recording anything when the device
            // lacks shared float atomics: the sort's output is just unused.
            s_block_refused = true;
            return 0;
        }
        return 1;
    };
    for (int pass = 0; pass < 3; ++pass) {
        constants.pass = pass;
        if (pass == 1) {
            const int block = block_scatter();
            if (block < 0) return false;
            if (block > 0) continue;
        }
        command.groups = FluidGpuDispatch::groups256(
            pass == 1 ? static_cast<uint32_t>(particle_count) : faces);
        if (!compute.dispatch(command)) return false;
    }
    return true;
}

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
    // The MPM lane (matter, Vulkan, dense): one fused P2G.
    bool fused_transfer = false;
    if (!sparse_transfer && granular && gpu_buffers.matter_model.enabled &&
        !window_normalize && compute->backendType() == ComputeBackendType::VulkanCompute) {
        fused_transfer = runGpuMatterFusedP2G(*compute, gpu_buffers, grid, particle_count, dt,
            granularFlipSnapshotWanted(fluid_params, gpu_buffers));
        if (!fused_transfer) return false;
    }
    // The dense path (or a failed compact one) needs the dense bank. A compact
    // owner whose P2G failed has had its pages released and must reallocate it.
    for (int comp = 0; comp < 3 && !sparse_transfer && !fused_transfer; ++comp) {
        if (!velocity_fields[comp].valid() || !weight_fields[comp].valid()) {
            return false;
        }
    }
    bool ok = true;
    const bool flip_snapshot = granularFlipSnapshotWanted(fluid_params, gpu_buffers);
    const ComputeBufferHandle flip_fields[3] = {
        gpu_buffers.scratch_vel_x, gpu_buffers.scratch_vel_y, gpu_buffers.scratch_vel_z};
    for (int comp = 0; comp < 3 && ok && !sparse_transfer && !fused_transfer; ++comp) {
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

        if (flip_snapshot) {
            ok = ok && Fluid::copyMatterGpuFloat(*compute, velocity_fields[comp],
                                                 flip_fields[comp], field_count);
        }

        if (granular) {
            ok = ok && Fluid::Granular::dispatchStressP2G(
                *compute, gpu_buffers.granular, gpu_buffers.fluid_positions,
                velocity_fields[comp], grid.nx, grid.ny, grid.nz, comp,
                grid.origin, grid.voxel_size, dt, fluid_params.granular_density,
                particle_count);
        }

        cmd.kernel = "sim_fluid_p2g_normalize";
        cmd.groups = groupsFor(field_count);
        ok = ok && (window_normalize
            ? Fluid::ActiveWindow::normalize(*compute, active_window, grid.nx, grid.ny,
                                            grid.nz, comp, velocity_fields[comp],
                                            weight_fields[comp])
            : Fluid::dispatchMatterGpuModel(*compute, cmd, gpu_buffers.matter_model));
        if (flip_snapshot) {
            // Same normalize kernel and 5-slot layout, pointed at the baseline.
            ComputeBufferHandle baseline_buffers[5] = {
                gpu_buffers.fluid_positions, gpu_buffers.fluid_velocities,
                gpu_buffers.fluid_affine, flip_fields[comp], weight_fields[comp]};
            cmd.buffers = baseline_buffers;
            cmd.buffer_count = 5;
            ok = ok && compute->dispatch(cmd);
        }
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

