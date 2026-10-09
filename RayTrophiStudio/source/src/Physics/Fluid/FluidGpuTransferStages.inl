// Included inside ParticleSimulation.cpp's private namespace. These stages use
// its existing push-constant types; keep transfer policy out of the large driver.

bool downloadGpuGranularParticles(Fluid::FluidParticles& particles,
                                 SimulationComputeContext* compute,
                                 SimulationGridDomainComputeBuffers& buffers,
                                 bool download_transport = true) {
    if (!compute) {
        return false;
    }
    const std::size_t count = particles.size();
    bool ok = true;
    compute->beginTransferBatch();
    auto download = [&](const ComputeBufferHandle& handle, auto& values) {
        if (values.size() < count) {
            ok = false;
            return;
        }
        using Value = typename std::decay_t<decltype(values)>::value_type;
        ok = compute->downloadBuffer(handle, values.data(), count * sizeof(Value)) && ok;
    };
    if (download_transport) {
        download(buffers.fluid_positions, particles.position);
        download(buffers.fluid_velocities, particles.velocity);
        download(buffers.fluid_affine, particles.affine);
    }
    download(buffers.granular.stress_diag, particles.granular_stress_diag);
    download(buffers.granular.stress_shear, particles.granular_stress_shear);
    download(buffers.granular.plastic_volume, particles.granular_plastic_volume);
    download(buffers.granular.state_flags, particles.granular_material_flags);
    download(buffers.granular.yield_value, particles.granular_yield_value);
    download(buffers.granular.plastic_increment, particles.granular_plastic_increment);
    download(buffers.granular.damage, particles.granular_damage);
    download(buffers.granular.hardening, particles.granular_hardening);
    download(buffers.granular.fracture_history, particles.granular_fracture_history);
    download(buffers.granular.deformation_col0, particles.granular_deformation_col0);
    download(buffers.granular.deformation_col1, particles.granular_deformation_col1);
    download(buffers.granular.deformation_col2, particles.granular_deformation_col2);
    return compute->endTransferBatch() && ok;
}

bool downloadGpuGranularParticleState(SimulationGridDomainState& state,
    SimulationComputeContext* compute, SimulationGridDomainComputeBuffers& buffers) {
    return downloadGpuGranularParticles(state.particles, compute, buffers);
}

class GranularParticleResidency {
public:
    GranularParticleResidency(SimulationGridDomainState& state,
                             SimulationComputeContext* compute,
                             SimulationGridDomainComputeBuffers* buffers,
                             bool enabled)
        : state_(state), compute_(compute), buffers_(buffers),
          enabled_(enabled && compute && buffers), occupancy_(state, compute, buffers) {
    }

    bool hostStale() const {
        return host_stale_;
    }

    bool buildMask() {
        gpu_mask_ready_ = enabled_ && occupancy_.build();
        if (enabled_ && !gpu_mask_ready_) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                SCENE_LOG_WARN("[Granular] GPU occupancy unavailable; using host masks "
                               "and position readbacks. Rebuild sim_fluid_occupancy.spv.");
            }
        }
        return gpu_mask_ready_;
    }

    bool retainPositions(const Fluid::APICSolverParams& params) const {
        if (!enabled_ || !host_stale_ || !gpu_mask_ready_) {
            return false;
        }
        // Call 2 still advances the original per-substep material schedule.
        // Publish positions whenever either generation will reset to identity.
        const uint32_t period = static_cast<uint32_t>(std::max(1, params.uvw_refresh_period));
        const uint32_t half = period / 2;
        const uint32_t next = state_.particles.uvw_step + 1u;
        return next % period != 0u &&
               (half == 0u || (next + half) % period != 0u);
    }

    bool prepare() {
        if (!host_stale_ ||
            FluidGpuParticleUpload::canReuse(*buffers_, *compute_, state_.particles.size())) {
            return true;
        }
        return recover();
    }

    bool defer(int substep, int count) const {
        return enabled_ && substep + 1 < count;
    }

    bool* tailDispatchFlag() {
        tail_dispatched_ = false;
        return &tail_dispatched_;
    }

    bool finishTail(bool& succeeded) {
        if (!enabled_ || succeeded || !tail_dispatched_) {
            return true;
        }
        // A dispatched tail must not be advected again on the CPU merely
        // because its readback failed. A successful recovery drains that work
        // and publishes its output; otherwise abandon the substep.
        host_stale_ = true;
        if (!recover()) {
            return false;
        }
        succeeded = true;
        return true;
    }

    void afterG2P(bool succeeded, int substep, int count) {
        if (enabled_) {
            // Even a failed dispatch can have queued writes. Recover before a
            // host consumer or fallback touches velocity/affine/stress.
            host_stale_ = !succeeded || defer(substep, count);
        }
    }

    bool recover() {
        if (!host_stale_) {
            return true;
        }
        if (!downloadGpuGranularParticleState(state_, compute_, *buffers_)) {
            SCENE_LOG_WARN("[Granular] particle readback failed; host fallback skipped "
                           "instead of consuming stale velocity/affine/stress.");
            return false;
        }
        host_stale_ = false;
        enabled_ = false;
        FluidGpuParticleUpload::invalidate(*buffers_);
        return true;
    }

private:
    SimulationGridDomainState& state_;
    SimulationComputeContext* compute_ = nullptr;
    SimulationGridDomainComputeBuffers* buffers_ = nullptr;
    bool enabled_ = false;
    bool host_stale_ = false;
    bool tail_dispatched_ = false;
    bool gpu_mask_ready_ = false;
    FluidGpuOccupancy occupancy_;
};

bool prepareGpuFluidMask(SimulationGridDomainState& state,
                         SimulationComputeContext& compute,
                         SimulationGridDomainComputeBuffers& buffers,
                         const Fluid::APICSolverParams& params,
                         GranularParticleResidency& residency,
                         std::vector<float>& host_mask,
                         bool& solid_velocity_uploaded) {
    buffers.fluid_mask_device_valid = false;
    FluidGpuOccupancy liquid_occupancy(state, &compute, &buffers);
    const bool device_mask = params.granular_enabled ? residency.buildMask()
        : (compute.backendType() == ComputeBackendType::VulkanCompute &&
           liquid_occupancy.build());
    buffers.fluid_mask_device_valid = device_mask;
    const std::size_t cells = state.grid.getCellCount();
    if (!device_mask) {
        // A missing/failed occupancy shader must publish device positions before
        // the CPU constructs a mask. Recovery also disables deferred readbacks.
        if (!residency.recover()) {
            return false;
        }
        buildFluidMaskFromParticles(state.grid, state.particles, host_mask);
    }
    if (!params.granular_enabled) {
        return true; // Liquid pressure/viscosity reuse the device mask when available.
    }
    if (device_mask && solid_velocity_uploaded) {
        return true; // No transfer batch or host cell/particle traversal.
    }
    compute.beginTransferBatch();
    bool ok = device_mask || compute.uploadBuffer(
        buffers.fluid_mask, host_mask.data(), cells * sizeof(float));
    if (ok && !solid_velocity_uploaded) {
        static std::vector<float> svx, svy, svz;
        svx.assign(cells, 0.0f);
        svy.assign(cells, 0.0f);
        svz.assign(cells, 0.0f);
        if (state.grid.solid_vel.size() == cells) {
            for (std::size_t cell = 0; cell < cells; ++cell) {
                svx[cell] = state.grid.solid_vel[cell].x;
                svy[cell] = state.grid.solid_vel[cell].y;
                svz[cell] = state.grid.solid_vel[cell].z;
            }
        }
        ok = compute.uploadBuffer(buffers.var_svx, svx.data(), cells * sizeof(float)) && ok;
        ok = compute.uploadBuffer(buffers.var_svy, svy.data(), cells * sizeof(float)) && ok;
        ok = compute.uploadBuffer(buffers.var_svz, svz.data(), cells * sizeof(float)) && ok;
    }
    ok = compute.endTransferBatch() && ok;
    if (ok) {
        solid_velocity_uploaded = true;
    }
    return ok;
}

bool runGpuFluidG2P(SimulationGridDomainState& state,
                    const Fluid::APICSolverParams& fluid_params,
                    float dt,
                    SimulationComputeContext* compute,
                    SimulationGridDomainComputeBuffers& gpu_buffers,
                    bool has_flip_snapshot,
                    bool download_granular_state = true,
                    bool reuse_particle_velocity = false,
                    bool reuse_projected_grid_velocity = false,
                    bool defer_particle_download = false) {
    auto& grid      = state.grid;
    auto& particles = state.particles;
    const std::size_t n = particles.size();
    if (!compute || !compute->supportsDispatch() || n == 0 || dt <= 0.0f ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        return false;
    }
    if (n > std::size_t(std::numeric_limits<int32_t>::max()) ||
        std::max({grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size()}) >
            std::size_t(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    // Dense bank, or the canonical compact pages (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md).
    const auto mac = Fluid::macVelocityBinding(gpu_buffers);
    if (!gpu_buffers.fluid_positions.valid() ||
        !gpu_buffers.fluid_velocities.valid() ||
        !gpu_buffers.fluid_affine.valid()     ||
        !mac.velocity[0].valid()              ||
        !mac.velocity[1].valid()              ||
        !mac.velocity[2].valid()              ||
        !gpu_buffers.fluid_mask.valid()) {
        return false;
    }

    // The caller vouches that the device vel_* is the field G2P must sample:
    // either the Vulkan projection with no solids (a host-only solid-face clamp
    // would otherwise make the host copy the authoritative one), or the granular
    // resident chain, which clamps on the device (sim_fluid_zero_solid_faces).
    const bool grid_velocity_is_current =
        reuse_projected_grid_velocity &&
        compute->backendType() == ComputeBackendType::VulkanCompute;
    bool ok = true;

    // FLIP snapshot was prepared in scratch before pressure, either by a
    // device copy or by the host-upload fallback.
    if (has_flip_snapshot && (mac.compact
            ? gpu_buffers.sparse_mac_transfer.snapshot_valid
            : (gpu_buffers.scratch_vel_x.valid() &&
               gpu_buffers.scratch_vel_y.valid() &&
               gpu_buffers.scratch_vel_z.valid()))) {
        // No additional transfer is needed here.
    } else {
        has_flip_snapshot = false;
    }
    // A compact field lives only on the device: there is no host copy to
    // upload in its place, so the caller must vouch it is current.
    if (mac.compact && !(reuse_projected_grid_velocity &&
                         compute->backendType() == ComputeBackendType::VulkanCompute)) {
        return false;
    }

    // P2G only reads particle velocity. If its full particle upload is still
    // current, G2P can use that device copy for the FLIP v_old term.
    const bool velocity_is_current =
        reuse_particle_velocity &&
        FluidGpuParticleUpload::canReuse(gpu_buffers, *compute, n);
    if (!grid_velocity_is_current || !velocity_is_current) {
        compute->beginTransferBatch();
        if (!grid_velocity_is_current) {
            ok = compute->uploadBuffer(
                     gpu_buffers.vel_x, grid.vel_x.data(),
                     grid.vel_x.size() * sizeof(float)) &&
                 compute->uploadBuffer(
                     gpu_buffers.vel_y, grid.vel_y.data(),
                     grid.vel_y.size() * sizeof(float)) &&
                 compute->uploadBuffer(
                     gpu_buffers.vel_z, grid.vel_z.data(),
                     grid.vel_z.size() * sizeof(float));
        }
        if (!velocity_is_current) {
            ok = ok && compute->uploadBuffer(gpu_buffers.fluid_velocities,
                                             particles.velocity.data(),
                                             n * sizeof(Vec3));
        }
        ok = compute->endTransferBatch() && ok;
    }
    if (!ok) {
        return false;
    }

    FluidG2PGpuConstants c;
    c.nx                = grid.nx;
    c.ny                = grid.ny;
    c.nz                = grid.nz;
    c.particle_count = static_cast<int>(n);
    c.origin_x          = grid.origin.x;
    c.origin_y          = grid.origin.y;
    c.origin_z          = grid.origin.z;
    c.voxel_size        = grid.voxel_size;
    c.flip_blend        = std::clamp(fluid_params.flip_blend, 0.0f, 1.0f);
    c.apic_blend        = std::clamp(fluid_params.apic_blend, 0.0f, 1.0f);
    c.internal_friction = fluid_params.granular_enabled ? 0.0f
                                                         : fluid_params.internal_friction;
    c.max_velocity      = fluid_params.max_velocity;
    c.dt                = dt;
    c.has_flip_snapshot = has_flip_snapshot ? 1 : 0;
    // Same limiter on both GPU backends (the Vulkan g2p shader now carries the
    // fluid_mask binding + wall-axis damping, 1:1 with the CUDA kernel).
    c.use_solid_flip_limiter =
        (compute->backendType() == ComputeBackendType::CUDA ||
         compute->backendType() == ComputeBackendType::VulkanCompute) ? 1 : 0;
    c.affine_damping    = fluid_params.affine_damping;
    c.max_affine        = fluid_params.max_affine;

    constexpr uint32_t threads = 256;
    const bool sparse_gather = gpu_buffers.sparse_mac_transfer.used &&
        (!has_flip_snapshot || gpu_buffers.sparse_mac_transfer.snapshot_valid);
    if (sparse_gather && !Fluid::captureSparseMacPost(*compute, gpu_buffers)) {
        return false;
    }
    const auto& sparse_pool = gpu_buffers.sparse_mac_transfer.owned;
    ComputeBufferHandle bufs[11] = {
        gpu_buffers.fluid_positions,
        gpu_buffers.fluid_velocities,
        gpu_buffers.fluid_affine,
        sparse_gather ? sparse_pool[2] : gpu_buffers.vel_x,
        sparse_gather ? sparse_pool[3] : gpu_buffers.vel_y,
        sparse_gather ? sparse_pool[4] : gpu_buffers.vel_z,
        sparse_gather ? sparse_pool[8] : gpu_buffers.scratch_vel_x,
        sparse_gather ? sparse_pool[9] : gpu_buffers.scratch_vel_y,
        sparse_gather ? sparse_pool[10] : gpu_buffers.scratch_vel_z,
        gpu_buffers.fluid_mask,
        sparse_pool[0]
    };
    ComputeDispatch cmd;
    cmd.kernel = sparse_gather ? "sim_sparse_mac_g2p" : "sim_fluid_g2p";
    cmd.buffers        = bufs;
    cmd.buffer_count = sparse_gather ? 11 : 10;
    cmd.constants      = &c;
    cmd.constants_size = sizeof(c);
    cmd.groups.groups_x =
        (static_cast<uint32_t>(c.particle_count) + threads - 1u) / threads;
    if (compute->backendType() == ComputeBackendType::VulkanCompute) {
        cmd.groups = FluidGpuDispatch::groups256(static_cast<uint32_t>(c.particle_count));
    }
    ok = Fluid::dispatchMatterGpuModel(*compute, cmd, gpu_buffers.matter_model);
    gpu_buffers.sparse_mac_transfer.flip_gather_used =
        ok && sparse_gather && has_flip_snapshot && c.flip_blend > 0.0f;

    if (ok && fluid_params.granular_enabled) {
        Fluid::Granular::Parameters gp;
        constexpr float deg_to_rad = 0.017453292519943295f;
        gp.friction_angle_radians =
            fluid_params.granular_friction_angle_degrees * deg_to_rad;
        gp.cohesion = fluid_params.granular_cohesion;
        gp.dilatancy = fluid_params.granular_dilatancy_degrees * deg_to_rad;
        gp.hardening = fluid_params.granular_hardening;
        gp.compaction_hardening = fluid_params.granular_compaction_hardening;
        gp.compaction_limit = fluid_params.granular_compaction_limit;
        gp.tensile_cutoff = fluid_params.granular_tensile_cutoff;
        const auto elastic_step = Fluid::Granular::elasticStepInfo(
            fluid_params.granular_young_modulus, state.grid.voxel_size, dt);
        ok = Fluid::Granular::dispatchStressUpdate(
            *compute, gpu_buffers.granular, gpu_buffers.fluid_affine, n, dt, gp,
            elastic_step.effective_young_modulus,
            fluid_params.granular_poisson_ratio,
            fluid_params.granular_fracture_strain,
            fluid_params.granular_damage_rate,
            fluid_params.granular_healing_rate,
            fluid_params.granular_rebonding);
        if (ok) {
            ok = Fluid::Granular::dispatchSettle(
                *compute, gpu_buffers.granular,
                gpu_buffers.fluid_velocities, n, dt);
        }
    }

    if (defer_particle_download) {
        return ok;
    }

    // No synchronize(): the download batch flushes uploads+dispatch+downloads
    // in one submit (see the P2G tail note).
    compute->beginTransferBatch();
    ok = ok &&
         compute->downloadBuffer(gpu_buffers.fluid_velocities,
                                 particles.velocity.data(),
                                 n * sizeof(Vec3)) &&
         compute->downloadBuffer(gpu_buffers.fluid_affine,
                                 particles.affine.data(),
                                 n * sizeof(Fluid::AffineC));
    if (ok && fluid_params.granular_enabled && download_granular_state) {
        ok = compute->downloadBuffer(gpu_buffers.granular.stress_diag,
                                     particles.granular_stress_diag.data(), n * sizeof(Vec3)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.stress_shear,
                                     particles.granular_stress_shear.data(),
                                     n * sizeof(Vec3)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.plastic_volume,
                                     particles.granular_plastic_volume.data(),
                                     n * sizeof(float)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.state_flags,
                                     particles.granular_material_flags.data(),
                                     n * sizeof(uint32_t)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.yield_value,
                                     particles.granular_yield_value.data(),
                                     n * sizeof(float)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.plastic_increment,
                                     particles.granular_plastic_increment.data(),
                                     n * sizeof(float)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.damage,
                                     particles.granular_damage.data(), n * sizeof(float)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.hardening,
                                     particles.granular_hardening.data(), n * sizeof(float)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.fracture_history,
                                     particles.granular_fracture_history.data(),
                                     n * sizeof(float)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.deformation_col0,
                                     particles.granular_deformation_col0.data(),
                                     n * sizeof(Vec3)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.deformation_col1,
                                     particles.granular_deformation_col1.data(),
                                     n * sizeof(Vec3)) && ok;
        ok = compute->downloadBuffer(gpu_buffers.granular.deformation_col2,
                                     particles.granular_deformation_col2.data(),
                                     n * sizeof(Vec3)) && ok;
    }
    ok = compute->endTransferBatch() && ok;
    return ok;
}

bool runGpuFluidAdvectTail(SimulationGridDomainState& state,
                           const Fluid::APICSolverParams& params,
                           float dt,
                           SimulationComputeContext* compute,
                           SimulationGridDomainComputeBuffers& buffers,
                           int* executed_substeps = nullptr,
                           bool download_affine = false,
                           bool* deferred_g2p_available = nullptr,
                           bool retain_granular_velocity = false,
                           bool* dispatched_tail = nullptr,
                           bool retain_positions = false,
                           bool contact_lagrangian = false) {
    if (dispatched_tail) {
        *dispatched_tail = false;
    }
    if (executed_substeps) {
        *executed_substeps = 0;
    }
    if (deferred_g2p_available) {
        *deferred_g2p_available = !download_affine;
    }
    auto& particles = state.particles;
    auto& grid = state.grid;
    const std::size_t n = particles.size();
    if (!compute ||
        compute->backendType() != ComputeBackendType::VulkanCompute ||
        !compute->supportsDispatch() || n == 0 || dt <= 0.0f) {
        return false;
    }
    // Dense bank, or the canonical compact pages plus their map and list
    // (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md).
    const auto mac = Fluid::macVelocityBinding(buffers);
    ComputeBufferHandle bufs[11] = {
        buffers.fluid_positions, buffers.fluid_velocities,
        mac.velocity[0], mac.velocity[1], mac.velocity[2],
        buffers.fluid_mask, buffers.var_svx, buffers.var_svy, buffers.var_svz,
        mac.map, mac.list
    };
    const std::size_t buffer_count = mac.compact ? 11u : 9u;
    for (std::size_t index = 0; index < buffer_count; ++index) {
        const auto& handle = bufs[index];
        if (!handle.valid()) {
            if (download_affine && buffers.fluid_velocities.valid() &&
                buffers.fluid_affine.valid()) {
                compute->beginTransferBatch();
                bool recovered = compute->downloadBuffer(
                    buffers.fluid_velocities,
                    particles.velocity.data(), n * sizeof(Vec3));
                recovered = compute->downloadBuffer(
                    buffers.fluid_affine,
                    particles.affine.data(), n * sizeof(Fluid::AffineC)) && recovered;
                recovered = compute->endTransferBatch() && recovered;
                if (deferred_g2p_available) {
                    *deferred_g2p_available = recovered;
                }
                if (!recovered) {
                    SCENE_LOG_WARN("[SimCompute] deferred G2P readback failed after "
                                   "advect-tail buffer validation failed.");
                }
            }
            return false;
        }
    }

    FluidAdvectTailGpuConstants c;
    c.nx = grid.nx;
    c.ny = grid.ny;
    c.nz = grid.nz;
    c.particle_count = static_cast<int>(std::min<std::size_t>(
        n, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    const int boundary_mode =
        params.boundary == Fluid::APICSolverParams::BoundaryMode::Open ? 1 :
        params.boundary == Fluid::APICSolverParams::BoundaryMode::Periodic ? 2 : 0;
    // Bits 0..1 retain the boundary enum; bit 2 selects Lagrangian MPM
    // advection. Reuse the existing word so the 64-byte push ABI is unchanged.
    c.boundary = boundary_mode | ((params.granular_enabled || contact_lagrangian) ? 4 : 0);
    c.origin_x = grid.origin.x;
    c.origin_y = grid.origin.y;
    c.origin_z = grid.origin.z;
    c.voxel_size = grid.voxel_size;
    c.dt = dt;
    c.velocity_damping = std::clamp(params.velocity_damping, 0.0f, 1.0f);
    c.max_velocity = params.max_velocity;
    c.wall_damping = params.wall_damping;
    c.air_drag = params.air_drag;
    {
        const Vec3 air = Fluid::atmosphereAirVelocity(params);
        c.air_wind_x = air.x;
        c.air_wind_y = air.y;
        c.air_wind_z = air.z;
    }
    c.air_threshold = std::max(1, params.reseed_min_per_cell);
    const float safe_cfl = std::max(params.cfl, 0.05f);
    c.substeps = std::clamp(
        static_cast<int>(std::ceil(std::max(0.0f, params.max_velocity) * dt /
                                   std::max(grid.voxel_size * safe_cfl, 1.0e-6f))),
        1, std::max(1, params.max_substeps));

    ComputeDispatch cmd;
    cmd.kernel = Fluid::macKernel(mac, "sim_fluid_advect_tail", "sim_sparse_mac_advect");
    cmd.buffers = bufs;
    cmd.buffer_count = buffer_count;
    cmd.constants = &c;
    cmd.constants_size = sizeof(c);
    cmd.groups.groups_x = (static_cast<uint32_t>(c.particle_count) + 255u) / 256u;
    const bool dispatched = Fluid::dispatchMatterGpuModel(*compute, cmd, buffers.matter_model);
    if (dispatched_tail) {
        *dispatched_tail = dispatched;
    }
    if (dispatched && executed_substeps) {
        *executed_substeps = c.substeps;
    }
    if (dispatched && buffers.matter_model.enabled) {
        return true; // Mixed coordinator publishes both disjoint lanes together.
    }
    if (dispatched && retain_positions && retain_granular_velocity && !download_affine &&
        params.granular_enabled &&
        params.boundary == Fluid::APICSolverParams::BoundaryMode::Closed) {
        // The next occupancy pass reads these positions directly. Submission is
        // bounded by the backend descriptor pool, rather than a particle readback.
        return true;
    }
    compute->beginTransferBatch();
    bool ok = true;
    if (dispatched) {
        // Publish positions on material refresh steps, on the final substep,
        // and on paths that still build occupancy on the host.
        ok = compute->downloadBuffer(buffers.fluid_positions,
                                     particles.position.data(), n * sizeof(Vec3));
        if (!retain_granular_velocity) {
            ok = compute->downloadBuffer(buffers.fluid_velocities,
                                         particles.velocity.data(), n * sizeof(Vec3)) && ok;
        }
    } else if (download_affine) {
        // G2P succeeded but the device tail did not dispatch. Bring its output
        // home so Call 2 can run the correct host tail instead of consuming the
        // previous frame's velocity.
        ok = compute->downloadBuffer(buffers.fluid_velocities,
                                     particles.velocity.data(), n * sizeof(Vec3));
    }
    if (download_affine) {
        ok = compute->downloadBuffer(buffers.fluid_affine,
                                     particles.affine.data(),
                                     n * sizeof(Fluid::AffineC)) && ok;
    }
    ok = compute->endTransferBatch() && ok;
    if (deferred_g2p_available) {
        *deferred_g2p_available = ok;
    }
    if (!dispatched || !ok) {
        return false;
    }
    if (executed_substeps) {
        *executed_substeps = c.substeps;
    }

    if (params.boundary == Fluid::APICSolverParams::BoundaryMode::Open) {
        Vec3 mn, mx;
        grid.getWorldBounds(mn, mx);
        for (std::size_t pi = particles.size(); pi-- > 0;) {
            const Vec3& p = particles.position[pi];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                p.x < mn.x || p.x > mx.x || p.y < mn.y || p.y > mx.y ||
                p.z < mn.z || p.z > mx.z) {
                particles.removeSwap(pi);
            }
        }
    }
    return true;
}
