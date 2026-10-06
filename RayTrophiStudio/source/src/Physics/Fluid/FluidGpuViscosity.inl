bool runGpuFluidViscosity(SimulationGridDomainState& state,
                          const Fluid::APICSolverParams& params,
                          float dt,
                          SimulationComputeContext* compute,
                          SimulationGridDomainComputeBuffers& gpu_buffers,
                          const std::vector<float>& fluid_mask_cpu,
                          int* out_sweeps_run) {
    if (out_sweeps_run) *out_sweeps_run = 0;
    auto& grid = state.grid;
    if (!compute ||
        compute->backendType() != ComputeBackendType::VulkanCompute ||
        !compute->supportsDispatch() || dt <= 0.0f ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty()) {
        return false;
    }
    const float h = (grid.voxel_size > 1.0e-6f) ? grid.voxel_size : 1.0f;
    const float alpha = params.kinematic_viscosity * dt / (h * h);
    const uint32_t cell_count_for_nu = static_cast<uint32_t>(grid.getCellCount());
    // A per-substance field carries its own viscosities, so it is reason enough
    // to run the stage even when the domain scalar is zero. Only the pair being
    // absent means there is nothing to diffuse.
    const bool has_nu_field =
        params.substance_viscosity != nullptr &&
        params.substance_viscosity->size() == cell_count_for_nu;
    if (alpha <= 1.0e-8f && !has_nu_field) {
        return false;   // nothing to diffuse; let the host no-op it
    }

    ComputeBufferHandle bufs[11] = {
        gpu_buffers.vel_x, gpu_buffers.vel_y, gpu_buffers.vel_z,
        gpu_buffers.fluid_mask,
        // scratch2_* is the MacCormack second-pass target, which only the GAS
        // solver uses; a domain is fluid or gas, never both in one step.
        gpu_buffers.scratch2_vel_x, gpu_buffers.scratch2_vel_y, gpu_buffers.scratch2_vel_z,
        gpu_buffers.var_svx, gpu_buffers.var_svy, gpu_buffers.var_svz,
        // ★ ALWAYS BOUND, read only when has_nu. Vulkan requires every declared
        // descriptor to have a buffer behind it, so a uniform domain points this
        // at an existing cell-sized allocation the kernel then never touches.
        // Leaving it unbound would be a validation error on the common path.
        gpu_buffers.substance_viscosity.valid() ? gpu_buffers.substance_viscosity
                                                : gpu_buffers.fluid_mask
    };
    for (const auto& handle : bufs) {
        if (!handle.valid()) return false;
    }

    const uint32_t cell_count = static_cast<uint32_t>(grid.getCellCount());
    if (fluid_mask_cpu.size() < cell_count && !gpu_buffers.fluid_mask_device_valid) return false;

    // Solid velocities, deinterleaved. Zero when the grid carries no colliders —
    // with wall_slip=0 the walls then hold the fluid still, which is what a
    // static no-slip boundary means.
    static std::vector<float> svx_host, svy_host, svz_host;
    const bool cached_statics = gpu_buffers.matter_model.enabled &&
        gpu_buffers.matter_model.viscosity_statics_uploaded;
    if (!cached_statics) {
        if (svx_host.size() < cell_count) {
            svx_host.assign(cell_count, 0.0f);
            svy_host.assign(cell_count, 0.0f);
            svz_host.assign(cell_count, 0.0f);
        }
        if (grid.solid_vel.size() == cell_count) {
            for (uint32_t c = 0; c < cell_count; ++c) {
                svx_host[c] = grid.solid_vel[c].x;
                svy_host[c] = grid.solid_vel[c].y;
                svz_host[c] = grid.solid_vel[c].z;
            }
        } else {
            std::fill(svx_host.begin(), svx_host.begin() + cell_count, 0.0f);
            std::fill(svy_host.begin(), svy_host.begin() + cell_count, 0.0f);
            std::fill(svz_host.begin(), svz_host.begin() + cell_count, 0.0f);
        }
    
    }

    const std::size_t vx_bytes = grid.vel_x.size() * sizeof(float);
    const std::size_t vy_bytes = grid.vel_y.size() * sizeof(float);
    const std::size_t vz_bytes = grid.vel_z.size() * sizeof(float);

    compute->beginTransferBatch();
    bool ok =
        (gpu_buffers.matter_model.enabled ||
         compute->uploadBuffer(gpu_buffers.vel_x, grid.vel_x.data(), vx_bytes)) &&
        (gpu_buffers.matter_model.enabled ||
         compute->uploadBuffer(gpu_buffers.vel_y, grid.vel_y.data(), vy_bytes)) &&
        (gpu_buffers.matter_model.enabled ||
         compute->uploadBuffer(gpu_buffers.vel_z, grid.vel_z.data(), vz_bytes)) &&
        // The RHS is the same field: GS starts from the pre-diffusion state and
        // relaxes in place, so both copies begin identical.
        (gpu_buffers.matter_model.enabled
            ? Fluid::copyMatterGpuFloat(*compute, gpu_buffers.vel_x,
                gpu_buffers.scratch2_vel_x, static_cast<uint32_t>(grid.vel_x.size()))
            : compute->uploadBuffer(gpu_buffers.scratch2_vel_x, grid.vel_x.data(), vx_bytes)) &&
        (gpu_buffers.matter_model.enabled
            ? Fluid::copyMatterGpuFloat(*compute, gpu_buffers.vel_y,
                gpu_buffers.scratch2_vel_y, static_cast<uint32_t>(grid.vel_y.size()))
            : compute->uploadBuffer(gpu_buffers.scratch2_vel_y, grid.vel_y.data(), vy_bytes)) &&
        (gpu_buffers.matter_model.enabled
            ? Fluid::copyMatterGpuFloat(*compute, gpu_buffers.vel_z,
                gpu_buffers.scratch2_vel_z, static_cast<uint32_t>(grid.vel_z.size()))
            : compute->uploadBuffer(gpu_buffers.scratch2_vel_z, grid.vel_z.data(), vz_bytes)) &&
        // cell_count, NOT fluid_mask_cpu.size(): the caller's scratch is
        // function-static and grow-only, so a smaller domain later in the same
        // session would otherwise overflow its correctly-sized buffer.
        Fluid::ActiveWindow::uploadPreparedMask(
            *compute, gpu_buffers.fluid_mask, fluid_mask_cpu, cell_count,
            gpu_buffers.fluid_mask_device_valid) &&
        (cached_statics ||
         compute->uploadBuffer(gpu_buffers.var_svx, svx_host.data(), cell_count * sizeof(float))) &&
        (cached_statics ||
         compute->uploadBuffer(gpu_buffers.var_svy, svy_host.data(), cell_count * sizeof(float))) &&
        (cached_statics ||
         compute->uploadBuffer(gpu_buffers.var_svz, svz_host.data(), cell_count * sizeof(float)));
    // ★ Uploaded INSIDE the same batch, not after: the dispatch below reads it
    // in the first sweep, and a field one transfer late would apply the previous
    // step's substance layout to this step's velocities — a mismatch that shows
    // up as the interface lagging the flow by a frame, which reads as "a bit
    // soft" rather than as a synchronisation bug.
    bool nu_uploaded = cached_statics && has_nu_field &&
        gpu_buffers.substance_viscosity.valid();
    if (!cached_statics && has_nu_field && gpu_buffers.substance_viscosity.valid()) {
        nu_uploaded = compute->uploadBuffer(
            gpu_buffers.substance_viscosity,
            params.substance_viscosity->data(),
            cell_count * sizeof(float));
        ok = ok && nu_uploaded;
    }
    ok = compute->endTransferBatch() && ok;
    if (!ok) return false;
    if (gpu_buffers.matter_model.enabled) {
        gpu_buffers.matter_model.viscosity_statics_uploaded = true;
    }

    FluidViscosityGpuConstants c;
    c.nx = grid.nx; c.ny = grid.ny; c.nz = grid.nz;
    c.boundary = (params.boundary == Fluid::APICSolverParams::BoundaryMode::Open)     ? 0
               : (params.boundary == Fluid::APICSolverParams::BoundaryMode::Periodic) ? 2
                                                                                      : 1;
    c.voxel_size = grid.voxel_size;
    c.dt = dt;
    c.alpha = alpha;
    c.solid_weight = 1.0f - std::clamp(params.viscosity_wall_slip, 0.0f, 1.0f);
    // ★ Gated on the UPLOAD, not on the field existing. If the transfer failed
    // the buffer still holds the previous step's viscosities, and telling the
    // kernel to read it would diffuse this frame's velocities with last frame's
    // material layout — plausible-looking and untraceable. Falling back to the
    // uniform alpha is wrong by a known amount instead of wrong by an unknown
    // one.
    c.has_nu = nu_uploaded ? 1 : 0;

    const uint32_t threads = 256;
    const uint32_t max_faces = static_cast<uint32_t>(
        std::max({grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size()}));
    const uint32_t groups = (max_faces + threads - 1u) / threads;
    const int sweeps = std::clamp(params.viscosity_sweeps, 1, 64);

    ComputeDispatch cmd;
    cmd.kernel = "sim_fluid_viscosity_rbgs";
    cmd.buffers = bufs;
    cmd.buffer_count = 11;
    cmd.constants_size = sizeof(c);
    cmd.groups.groups_x = groups;
    cmd.groups.groups_y = 1;
    cmd.groups.groups_z = 1;

    for (int s = 0; s < sweeps && ok; ++s) {
        for (int parity = 0; parity < 2 && ok; ++parity) {
            c.parity = parity;
            cmd.constants = &c;
            ok = compute->dispatch(cmd);
        }
    }
    if (!ok) return false;

    if (gpu_buffers.matter_model.enabled) {
        if (out_sweeps_run) {
            *out_sweeps_run = sweeps;
        }
        return true;
    }
    compute->synchronize();
    compute->beginTransferBatch();
    ok = compute->downloadBuffer(gpu_buffers.vel_x, grid.vel_x.data(), vx_bytes) &&
         compute->downloadBuffer(gpu_buffers.vel_y, grid.vel_y.data(), vy_bytes) &&
         compute->downloadBuffer(gpu_buffers.vel_z, grid.vel_z.data(), vz_bytes);
    ok = compute->endTransferBatch() && ok;
    if (!ok) return false;

    if (out_sweeps_run) *out_sweeps_run = sweeps;
    return true;
}

