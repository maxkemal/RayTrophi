// Included in the simulation driver private namespace.
bool runGpuFluidMGPCGPressure(SimulationGridDomainState& state,
                              const Fluid::APICSolverParams& fluid_params,
                              float dt,
                              SimulationComputeContext* compute,
                              SimulationGridDomainComputeBuffers& gpu_buffers,
                              const std::vector<float>& fluid_mask_cpu,
                              Fluid::APICSolverStats* mgpcg_stats = nullptr) {
    auto& grid = state.grid;
    if (!compute || !compute->supportsDispatch() || dt <= 0.0f ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty() ||
        grid.pressure.size() != grid.getCellCount() ||
        grid.divergence.size() != grid.getCellCount()) {
        return false;
    }
    // Name the missing buffer once. At high resolution these are the allocations
    // that fail first — the CG scratch alone is five full cell-sized float fields
    // on top of the solver's own — and an invalid handle here used to return false
    // into a caller that reported "(CPU)" without explaining anything.
    auto reportMissing = [&](const char* which) {
        // Keyed, not one-shot: a second domain (or a new project in the same
        // session) failing on a different buffer must not be swallowed because an
        // earlier one already reported.
        const std::string key =
            std::string(which) + "|" + std::to_string(grid.getCellCount());
        static std::string last_key;
        if (key != last_key) {
            last_key = key;
            SCENE_LOG_WARN("[SimCompute] GPU MGPCG pressure unavailable: " +
                           std::string(which) + " buffer not allocated (cells=" +
                           std::to_string(grid.getCellCount()) +
                           "). Most likely out of VRAM at this resolution.");
        }
        return false;
    };
    // Every other way out of this function used to be a bare `return false`, which
    // is how a fallback with a known stage ("MGPCG pressure projection") still left
    // no idea WHICH step inside it gave up. Keyed like reportMissing so a later
    // domain or a new project in the same session is not swallowed.
    auto bail = [&](const char* where) {
        const std::string key =
            std::string(where) + "|" + std::to_string(grid.getCellCount());
        static std::string last_key;
        if (key != last_key) {
            last_key = key;
            SCENE_LOG_WARN("[SimCompute] GPU MGPCG gave up at: " + std::string(where) +
                           " (cells=" + std::to_string(grid.getCellCount()) + ")");
        }
        return false;
    };

    if (!gpu_buffers.vel_x.valid())      return reportMissing("vel_x");
    if (!gpu_buffers.vel_y.valid())      return reportMissing("vel_y");
    if (!gpu_buffers.vel_z.valid())      return reportMissing("vel_z");
    if (!gpu_buffers.pressure.valid())   return reportMissing("pressure");
    if (!gpu_buffers.divergence.valid()) return reportMissing("divergence");
    if (!gpu_buffers.fluid_mask.valid()) return reportMissing("fluid_mask");
    // CG scratch missing → signal fallback to the SOR / CPU PCG path.
    if (!gpu_buffers.cg_residual.valid()) return reportMissing("cg_residual");
    if (!gpu_buffers.cg_z.valid())        return reportMissing("cg_z");
    if (!gpu_buffers.cg_search.valid())   return reportMissing("cg_search");
    if (!gpu_buffers.cg_As.valid())       return reportMissing("cg_As");
    if (!gpu_buffers.cg_diag.valid())     return reportMissing("cg_diag");
    if (!gpu_buffers.cg_partials.valid()) return reportMissing("cg_partials");

    const uint32_t threads     = 256; // must match the device block size (256)
    const uint32_t cell_count  = static_cast<uint32_t>(grid.getCellCount());
    const uint32_t cell_groups = (cell_count + threads - 1u) / threads;
    const uint32_t max_faces   = static_cast<uint32_t>(
        std::max({grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size()}));

    // Upload velocity (already reflects boundary + viscosity from CPU) + mask.
    // Batched: one submission instead of four (Vulkan submit+fence per copy).
    const auto mg_upload_begin = SimulationClock::now();
    compute->beginTransferBatch();
    bool ok = (gpu_buffers.matter_model.enabled ||
             compute->uploadBuffer(gpu_buffers.vel_x, grid.vel_x.data(),
                                     grid.vel_x.size() * sizeof(float))) &&
              (gpu_buffers.matter_model.enabled ||
             compute->uploadBuffer(gpu_buffers.vel_y, grid.vel_y.data(),
                                     grid.vel_y.size() * sizeof(float))) &&
              (gpu_buffers.matter_model.enabled ||
             compute->uploadBuffer(gpu_buffers.vel_z, grid.vel_z.data(),
                                     grid.vel_z.size() * sizeof(float))) &&
              // cell_count, NOT fluid_mask_cpu.size(): the caller's scratch vector is
              // function-static and grow-only (buildFluidMaskFromParticles keeps the
              // high-water mark to avoid reallocating every step). Run a 4.6M-cell
              // domain, then open a new project with a 125k-cell one in the SAME
              // session and the vector is still 4.6M long — uploading its full size
              // into the correctly-sized 125k buffer overflows it, uploadBuffer
              // returns false, and the whole pressure+G2P path silently drops to the
              // CPU. Only the live cells are ever read by the kernels anyway.
              Fluid::ActiveWindow::uploadPreparedMask(
                  *compute, gpu_buffers.fluid_mask, fluid_mask_cpu, cell_count,
                  gpu_buffers.fluid_mask_device_valid);
    ok = compute->endTransferBatch() && ok;
    if (!ok) return bail("velocity + fluid_mask upload");
    const float mg_upload_ms = elapsedMilliseconds(mg_upload_begin, SimulationClock::now());

    const bool is_variational = fluid_params.variational_solids &&
                                (grid.u_weight.size() == grid.vel_x.size()) &&
                                (grid.v_weight.size() == grid.vel_y.size()) &&
                                (grid.w_weight.size() == grid.vel_z.size()) &&
                                gpu_buffers.var_u_weight.valid() &&
                                gpu_buffers.var_v_weight.valid() &&
                                gpu_buffers.var_w_weight.valid() &&
                                gpu_buffers.var_svx.valid() &&
                                gpu_buffers.var_svy.valid() &&
                                gpu_buffers.var_svz.valid();

    const bool is_gfm = fluid_params.ghost_fluid_surface &&
                        (grid.fluid_phi.size() == cell_count) &&
                        gpu_buffers.var_fluid_phi.valid();

    if (is_variational && (!gpu_buffers.matter_model.enabled ||
        !gpu_buffers.matter_model.pressure_statics_uploaded)) {
        // Convert and upload weights
        std::vector<float> uw_float(grid.u_weight.size());
        std::vector<float> vw_float(grid.v_weight.size());
        std::vector<float> ww_float(grid.w_weight.size());
        for (std::size_t i = 0; i < grid.u_weight.size(); ++i) uw_float[i] = FluidSim::FluidGrid::weightToFloat(grid.u_weight[i]);
        for (std::size_t i = 0; i < grid.v_weight.size(); ++i) vw_float[i] = FluidSim::FluidGrid::weightToFloat(grid.v_weight[i]);
        for (std::size_t i = 0; i < grid.w_weight.size(); ++i) ww_float[i] = FluidSim::FluidGrid::weightToFloat(grid.w_weight[i]);

        ok = ok && compute->uploadBuffer(gpu_buffers.var_u_weight, uw_float.data(), uw_float.size() * sizeof(float));
        ok = ok && compute->uploadBuffer(gpu_buffers.var_v_weight, vw_float.data(), vw_float.size() * sizeof(float));
        ok = ok && compute->uploadBuffer(gpu_buffers.var_w_weight, ww_float.data(), ww_float.size() * sizeof(float));

        // Deinterleave and upload solid velocities
        std::vector<float> svx(cell_count, 0.0f);
        std::vector<float> svy(cell_count, 0.0f);
        std::vector<float> svz(cell_count, 0.0f);
        if (grid.solid_vel.size() == cell_count) {
            for (std::size_t i = 0; i < cell_count; ++i) {
                svx[i] = grid.solid_vel[i].x;
                svy[i] = grid.solid_vel[i].y;
                svz[i] = grid.solid_vel[i].z;
            }
        }
        ok = ok && compute->uploadBuffer(gpu_buffers.var_svx, svx.data(), svx.size() * sizeof(float));
        ok = ok && compute->uploadBuffer(gpu_buffers.var_svy, svy.data(), svy.size() * sizeof(float));
        ok = ok && compute->uploadBuffer(gpu_buffers.var_svz, svz.data(), svz.size() * sizeof(float));
    }

    if (is_gfm) {
        ok = ok && compute->uploadBuffer(gpu_buffers.var_fluid_phi, grid.fluid_phi.data(), grid.fluid_phi.size() * sizeof(float));
    }
    if (!ok) return bail("variational weight / solid-velocity upload");
    if (gpu_buffers.matter_model.enabled) {
        gpu_buffers.matter_model.pressure_statics_uploaded = true;
    }

    GridProjectionGpuConstants c;
    c.nx         = grid.nx;
    c.ny         = grid.ny;
    c.nz         = grid.nz;
    // Mirror the domain wall mode onto the GPU MGPCG projection (0=open,
    // 1=closed, 2=periodic) so an Open domain treats its bounding walls as
    // p=0 outflow on the GPU exactly like the CPU free-surface solver. Without
    // this the GPU path always sealed the walls regardless of the UI setting.
    c.boundary   = (fluid_params.boundary == Fluid::APICSolverParams::BoundaryMode::Open)     ? 0
                 : (fluid_params.boundary == Fluid::APICSolverParams::BoundaryMode::Periodic) ? 2
                 : 1;
    c.voxel_size = grid.voxel_size;
    c.dt         = dt;
    c.sor_omega  = 0.0f; // reused to carry CG alpha/beta per dispatch
    c.iterations = 0;
    c.parity     = 0;
    // Bridson density-targeted projection (mask carries per-cell particle count).
    c.density_correction = fluid_params.density_correction;
    c.particles_per_cell = fluid_params.particles_per_cell;
    c.variational        = is_variational ? 1 : 0;
    c.gfm_active         = is_gfm ? 1 : 0;

    auto pressure_window = Fluid::ActiveWindow::full(grid.nx, grid.ny, grid.nz);
    if (!gpu_buffers.matter_model.enabled &&
        compute->backendType() == ComputeBackendType::VulkanCompute &&
        fluid_params.boundary != Fluid::APICSolverParams::BoundaryMode::Periodic) {
        pressure_window = gpu_buffers.fluid_mask_device_valid
            ? Fluid::ActiveWindow::plan(
                state.particles.position, state.particles.velocity, grid.origin,
                grid.voxel_size, dt, grid.nx, grid.ny, grid.nz, true)
            : Fluid::ActiveWindow::pressureMaskBounds(
                fluid_mask_cpu, grid.nx, grid.ny, grid.nz);
    }
    const Fluid::ActiveWindow::PressureDispatch window_dispatch(pressure_window);
    if (mgpcg_stats) {
        mgpcg_stats->pressure_window_used = pressure_window.bounded;
        mgpcg_stats->pressure_window_cells = pressure_window.cells();
    }

    ComputeDispatch cmd;
    cmd.constants      = &c;
    cmd.constants_size = sizeof(c);

    auto dispatch1 = [&](const char* kernel, ComputeBufferHandle* bufs, int n, uint32_t groups) -> bool {
        cmd.kernel          = kernel;
        cmd.buffers         = bufs;
        cmd.buffer_count    = n;
        cmd.constants       = &c;
        cmd.constants_size  = sizeof(c);
        cmd.groups.groups_x = groups;
        cmd.groups.groups_y = 1;
        cmd.groups.groups_z = 1;
        return window_dispatch.dispatch(*compute, cmd, c);
    };

    // divergence = (div u)/h  (positive; the residual_init kernel negates it).
    // Both GPU backends now carry the fluid-mask-aware path that mirrors the
    // CPU free-surface solver: solid neighbours contribute zero flux, and later
    // gradient subtraction updates only fluid/air faces while zeroing true
    // solid faces. Vulkan now also carries the variational (fractional face
    // weight) system through the *_var kernel variants; only GFM still routes
    // to the CPU there.
    const bool use_cuda_fluid_projection =
        compute->backendType() == ComputeBackendType::CUDA ||
        compute->backendType() == ComputeBackendType::VulkanCompute;

    // Faz 1 Vulkan MGPCG port covers only the PLAIN free-surface system. When
    // the sim actually carries variational solid weights or a GFM level set,
    // the Vulkan kernels would silently solve the wrong (binary-solid) matrix —
    // fall back to the CPU PCG, which fully supports both.
    // Variational solid weights ARE implemented on Vulkan now, through the *_var
    // kernel variants selected by varKernel() below. GFM is not: porting both
    // matrix changes in one go would make any regression impossible to attribute,
    // so ghost-fluid domains still take the CPU PCG.
    if (compute->backendType() == ComputeBackendType::VulkanCompute && is_gfm) {
        // A missing FEATURE, not a failure, and it costs the whole GPU
        // pressure+G2P path — so it says so. This used to be a bare return, which
        // is why a domain with the option merely switched on looked like the GPU
        // had broken at high resolution.
        static bool logged = false;
        if (!logged) {
            logged = true;
            SCENE_LOG_WARN(
                "[SimCompute] Vulkan MGPCG does not implement the ghost-fluid "
                "surface; pressure + G2P run on the CPU for this domain. Turn GFM "
                "off to keep the solve on the GPU, or use the CUDA backend.");
        }
        return false;
    }

    // The domain WANTS variational coupling and the grid carries the weights, but
    // the device buffers for them are missing. Solving anyway would quietly use the
    // binary-solid matrix — a wrong answer that looks like a working GPU solve,
    // which is worse than being slow. Hand it to the CPU instead.
    if (fluid_params.variational_solids &&
        grid.u_weight.size() == grid.vel_x.size() && !is_variational) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            SCENE_LOG_WARN("[SimCompute] variational solid weights requested but their "
                           "GPU buffers are unavailable; falling back to the CPU PCG "
                           "rather than solving the binary-solid system.");
        }
        return false;
    }

    // On Vulkan the variational system lives in separate kernels (the registry
    // binds a fixed buffer count per name). CUDA keeps one name and switches on
    // the dispatch's buffer_count, so it must NOT get the suffix.
    const bool vulkan_variational =
        is_variational && compute->backendType() == ComputeBackendType::VulkanCompute;
    auto varKernel = [vulkan_variational](const char* plain, const char* var) {
        return vulkan_variational ? var : plain;
    };
    ComputeBufferHandle proj_bufs[5] = {
        gpu_buffers.vel_x, gpu_buffers.vel_y, gpu_buffers.vel_z,
        gpu_buffers.pressure, gpu_buffers.divergence
    };
    ComputeBufferHandle fluid_divergence_bufs[11];
    fluid_divergence_bufs[0] = gpu_buffers.vel_x;
    fluid_divergence_bufs[1] = gpu_buffers.vel_y;
    fluid_divergence_bufs[2] = gpu_buffers.vel_z;
    fluid_divergence_bufs[3] = gpu_buffers.fluid_mask;
    fluid_divergence_bufs[4] = gpu_buffers.divergence;
    if (is_variational) {
        fluid_divergence_bufs[5] = gpu_buffers.var_u_weight;
        fluid_divergence_bufs[6] = gpu_buffers.var_v_weight;
        fluid_divergence_bufs[7] = gpu_buffers.var_w_weight;
        fluid_divergence_bufs[8] = gpu_buffers.var_svx;
        fluid_divergence_bufs[9] = gpu_buffers.var_svy;
        fluid_divergence_bufs[10] = gpu_buffers.var_svz;
    }
    const int div_buf_count = is_variational ? 11 : 5;

    const bool porous = vulkan_variational && gpu_buffers.porous_solid_velocity;
    ok = dispatch1(use_cuda_fluid_projection
                       ? (porous ? "sim_fluid_divergence_porous"
                                 : varKernel("sim_fluid_divergence", "sim_fluid_divergence_var"))
                       : "sim_grid_divergence",
                   use_cuda_fluid_projection ? fluid_divergence_bufs : proj_bufs,
                   use_cuda_fluid_projection ? div_buf_count : 5,
                   cell_groups);

    // diag = #in-bounds neighbours (fluid rows; 0 elsewhere).
    {
        ComputeBufferHandle b[6];
        b[0] = gpu_buffers.fluid_mask;
        b[1] = gpu_buffers.cg_diag;
        int diag_buf_count = 2;
        if (is_variational) {
            b[2] = gpu_buffers.var_u_weight;
            b[3] = gpu_buffers.var_v_weight;
            b[4] = gpu_buffers.var_w_weight;
            diag_buf_count = 5;
        }
        if (is_gfm) {
            b[diag_buf_count] = gpu_buffers.var_fluid_phi;
            diag_buf_count++;
        }
        ok = ok && dispatch1(varKernel("sim_fluid_cg_build_diag",
                                       "sim_fluid_cg_build_diag_var"),
                             b, diag_buf_count, cell_groups);
    }
    // r = -div*h*h/dt at fluid cells; pressure reset to 0.
    { ComputeBufferHandle b[4] = { gpu_buffers.divergence, gpu_buffers.fluid_mask,
                                   gpu_buffers.cg_residual, gpu_buffers.pressure };
      ok = ok && dispatch1("sim_fluid_cg_residual_init", b, 4, cell_groups); }
    if (!ok) return bail("divergence / build_diag / residual_init dispatch");

    const bool use_cuda_fused_reductions = compute->backendType() == ComputeBackendType::CUDA;

    // Reduction helpers: dispatch a reducing kernel, sync, download the per-block
    // double partials, sum on host. Function-static host buffer avoids per-call
    // heap churn (grows with the block count as needed).
    const uint32_t dot_blocks = window_dispatch.groups(cell_groups);
    static std::vector<double> cg_partials_host;
    if (cg_partials_host.size() < dot_blocks) cg_partials_host.assign(dot_blocks, 0.0);
    float dot_ms = 0.0f;
    int dot_count = 0;
    bool used_multigrid = false;
    auto finishReduction = [&](const auto& dot_begin, double& out) -> bool {
        compute->synchronize();
        if (!compute->downloadBuffer(gpu_buffers.cg_partials, cg_partials_host.data(),
                                     dot_blocks * sizeof(double)))
            return bail("cg_partials download");
        double sum = 0.0;
        for (uint32_t bi = 0; bi < dot_blocks; ++bi) sum += cg_partials_host[bi];
        out = sum;
        dot_ms += elapsedMilliseconds(dot_begin, SimulationClock::now());
        ++dot_count;
        return true;
    };
    auto dotProduct = [&](ComputeBufferHandle x, ComputeBufferHandle y, double& out) -> bool {
        const auto dot_begin = SimulationClock::now();
        ComputeBufferHandle b[3] = { x, y, gpu_buffers.cg_partials };
        if (!dispatch1("sim_fluid_cg_dot", b, 3, dot_blocks)) return bail("cg_dot dispatch");
        return finishReduction(dot_begin, out);
    };

    auto mgLevelValid = [](const SimulationGridDomainMGLevelBuffers& level) -> bool {
        return level.nx > 0 && level.ny > 0 && level.nz > 0 &&
               level.mask.valid() && level.rhs.valid() &&
               level.z.valid() && level.diag.valid();
    };
    auto dispatchCellKernel = [&](const char* kernel,
                                  ComputeBufferHandle* bufs,
                                  int n,
                                  GridProjectionGpuConstants& pc) -> bool {
        cmd.kernel          = kernel;
        cmd.buffers         = bufs;
        cmd.buffer_count    = n;
        cmd.constants       = &pc;
        cmd.constants_size  = sizeof(pc);
        const uint32_t groups = (static_cast<uint32_t>(pc.nx) *
                                 static_cast<uint32_t>(pc.ny) *
                                 static_cast<uint32_t>(pc.nz) + threads - 1u) / threads;
        cmd.groups.groups_x = std::max(1u, groups);
        cmd.groups.groups_y = 1;
        cmd.groups.groups_z = 1;
        return compute->dispatch(cmd);
    };
    auto zeroCells = [&](ComputeBufferHandle values, int nx, int ny, int nz) -> bool {
        GridProjectionGpuConstants pc = c;
        pc.nx = nx;
        pc.ny = ny;
        pc.nz = nz;
        ComputeBufferHandle b[1] = { values };
        return dispatchCellKernel("sim_fluid_mg_zero", b, 1, pc);
    };
    auto buildDiag = [&](ComputeBufferHandle mask, ComputeBufferHandle diag,
                         int nx, int ny, int nz) -> bool {
        GridProjectionGpuConstants pc = c;
        pc.nx = nx;
        pc.ny = ny;
        pc.nz = nz;
        ComputeBufferHandle b[2] = { mask, diag };
        return dispatchCellKernel("sim_fluid_cg_build_diag", b, 2, pc);
    };
    auto smoothLevel = [&](ComputeBufferHandle rhs, ComputeBufferHandle mask,
                           ComputeBufferHandle diag, ComputeBufferHandle z,
                           int nx, int ny, int nz, int sweeps) -> bool {
        GridProjectionGpuConstants pc = c;
        pc.nx = nx;
        pc.ny = ny;
        pc.nz = nz;
        pc.sor_omega = 0.8f;
        ComputeBufferHandle b[4] = { rhs, mask, diag, z };
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            pc.parity = 0;
            if (!dispatchCellKernel("sim_fluid_mg_rbgs", b, 4, pc)) return false;
            pc.parity = 1;
            if (!dispatchCellKernel("sim_fluid_mg_rbgs", b, 4, pc)) return false;
        }
        return true;
    };
    auto restrictLevel = [&](ComputeBufferHandle fine_rhs, ComputeBufferHandle fine_mask,
                             int fine_nx, int fine_ny, int fine_nz,
                             SimulationGridDomainMGLevelBuffers& coarse) -> bool {
        GridMGGpuConstants mgc;
        mgc.fine_nx = fine_nx;
        mgc.fine_ny = fine_ny;
        mgc.fine_nz = fine_nz;
        mgc.coarse_nx = coarse.nx;
        mgc.coarse_ny = coarse.ny;
        mgc.coarse_nz = coarse.nz;
        ComputeBufferHandle b[4] = { fine_rhs, fine_mask, coarse.rhs, coarse.mask };
        cmd.kernel = "sim_fluid_mg_restrict";
        cmd.buffers = b;
        cmd.buffer_count = 4;
        cmd.constants = &mgc;
        cmd.constants_size = sizeof(mgc);
        const uint32_t groups =
            (static_cast<uint32_t>(coarse.nx) *
             static_cast<uint32_t>(coarse.ny) *
             static_cast<uint32_t>(coarse.nz) + threads - 1u) / threads;
        cmd.groups.groups_x = std::max(1u, groups);
        cmd.groups.groups_y = 1;
        cmd.groups.groups_z = 1;
        return compute->dispatch(cmd);
    };
    auto prolongateAdd = [&](const SimulationGridDomainMGLevelBuffers& coarse,
                             ComputeBufferHandle fine_mask,
                             ComputeBufferHandle fine_z,
                             int fine_nx, int fine_ny, int fine_nz) -> bool {
        GridMGGpuConstants mgc;
        mgc.fine_nx = fine_nx;
        mgc.fine_ny = fine_ny;
        mgc.fine_nz = fine_nz;
        mgc.coarse_nx = coarse.nx;
        mgc.coarse_ny = coarse.ny;
        mgc.coarse_nz = coarse.nz;
        ComputeBufferHandle b[3] = { coarse.z, fine_mask, fine_z };
        cmd.kernel = "sim_fluid_mg_prolongate_add";
        cmd.buffers = b;
        cmd.buffer_count = 3;
        cmd.constants = &mgc;
        cmd.constants_size = sizeof(mgc);
        const uint32_t groups =
            (static_cast<uint32_t>(fine_nx) *
             static_cast<uint32_t>(fine_ny) *
             static_cast<uint32_t>(fine_nz) + threads - 1u) / threads;
        cmd.groups.groups_x = std::max(1u, groups);
        cmd.groups.groups_y = 1;
        cmd.groups.groups_z = 1;
        return compute->dispatch(cmd);
    };
    auto mgPrecondition = [&]() -> bool {
        if (is_variational || !fluid_params.pressure_multigrid_preconditioner ||
            !use_cuda_fused_reductions ||
            gpu_buffers.mg_levels.empty()) {
            return false;
        }
        for (const auto& level : gpu_buffers.mg_levels) {
            if (!mgLevelValid(level)) return false;
        }

        ComputeBufferHandle fine_rhs = gpu_buffers.cg_residual;
        ComputeBufferHandle fine_mask = gpu_buffers.fluid_mask;
        int fine_nx = grid.nx;
        int fine_ny = grid.ny;
        int fine_nz = grid.nz;
        for (auto& level : gpu_buffers.mg_levels) {
            if (!restrictLevel(fine_rhs, fine_mask, fine_nx, fine_ny, fine_nz, level)) return false;
            if (!buildDiag(level.mask, level.diag, level.nx, level.ny, level.nz)) return false;
            if (!zeroCells(level.z, level.nx, level.ny, level.nz)) return false;
            fine_rhs = level.rhs;
            fine_mask = level.mask;
            fine_nx = level.nx;
            fine_ny = level.ny;
            fine_nz = level.nz;
        }

        auto& coarsest = gpu_buffers.mg_levels.back();
        if (!smoothLevel(coarsest.rhs, coarsest.mask, coarsest.diag, coarsest.z,
                         coarsest.nx, coarsest.ny, coarsest.nz, 12)) return false;

        for (int li = static_cast<int>(gpu_buffers.mg_levels.size()) - 2; li >= 0; --li) {
            auto& coarse = gpu_buffers.mg_levels[static_cast<std::size_t>(li + 1)];
            auto& fine = gpu_buffers.mg_levels[static_cast<std::size_t>(li)];
            if (!prolongateAdd(coarse, fine.mask, fine.z, fine.nx, fine.ny, fine.nz)) return false;
            if (!smoothLevel(fine.rhs, fine.mask, fine.diag, fine.z,
                             fine.nx, fine.ny, fine.nz, 2)) return false;
        }

        auto& first_coarse = gpu_buffers.mg_levels.front();
        if (!zeroCells(gpu_buffers.cg_z, grid.nx, grid.ny, grid.nz)) return false;
        if (!prolongateAdd(first_coarse, gpu_buffers.fluid_mask, gpu_buffers.cg_z,
                           grid.nx, grid.ny, grid.nz)) return false;
        if (!smoothLevel(gpu_buffers.cg_residual, gpu_buffers.fluid_mask, gpu_buffers.cg_diag,
                         gpu_buffers.cg_z, grid.nx, grid.ny, grid.nz, 2)) return false;
        return true;
    };
    auto jacobiAndDot = [&](double& out) -> bool {
        if (mgPrecondition()) {
            used_multigrid = true;
            return dotProduct(gpu_buffers.cg_residual, gpu_buffers.cg_z, out);
        }
        if (use_cuda_fused_reductions) {
            const auto dot_begin = SimulationClock::now();
            ComputeBufferHandle b[4] = { gpu_buffers.cg_residual, gpu_buffers.cg_diag,
                                         gpu_buffers.cg_z, gpu_buffers.cg_partials };
            if (!dispatch1("sim_fluid_cg_jacobi_dot", b, 4, dot_blocks))
                return bail("cg_jacobi_dot dispatch");
            return finishReduction(dot_begin, out);
        }
        { ComputeBufferHandle b[3] = { gpu_buffers.cg_residual, gpu_buffers.cg_diag, gpu_buffers.cg_z };
          if (!dispatch1("sim_fluid_cg_jacobi", b, 3, cell_groups))
              return bail("cg_jacobi dispatch"); }
        return dotProduct(gpu_buffers.cg_residual, gpu_buffers.cg_z, out);
    };
    auto spmvAndDot = [&](double& out) -> bool {
        if (use_cuda_fused_reductions) {
            const auto dot_begin = SimulationClock::now();
            ComputeBufferHandle b[8] = { gpu_buffers.cg_search, gpu_buffers.fluid_mask,
                                         gpu_buffers.cg_diag, gpu_buffers.cg_As,
                                         gpu_buffers.cg_partials };
            int spmv_buf_count = 5;
            if (is_variational) {
                b[5] = gpu_buffers.var_u_weight;
                b[6] = gpu_buffers.var_v_weight;
                b[7] = gpu_buffers.var_w_weight;
                spmv_buf_count = 8;
            }
            if (!dispatch1("sim_fluid_cg_spmv_dot", b, spmv_buf_count, dot_blocks)) return false;
            return finishReduction(dot_begin, out);
        }
        {
            ComputeBufferHandle b[7] = { gpu_buffers.cg_search, gpu_buffers.fluid_mask,
                                         gpu_buffers.cg_diag, gpu_buffers.cg_As };
            int spmv_buf_count = 4;
            if (is_variational) {
                b[4] = gpu_buffers.var_u_weight;
                b[5] = gpu_buffers.var_v_weight;
                b[6] = gpu_buffers.var_w_weight;
                spmv_buf_count = 7;
            }
            if (!dispatch1(varKernel("sim_fluid_cg_spmv", "sim_fluid_cg_spmv_var"),
                           b, spmv_buf_count, cell_groups)) return bail("cg_spmv dispatch");
        }
        return dotProduct(gpu_buffers.cg_search, gpu_buffers.cg_As, out);
    };

    const double rel_tol_pre = std::clamp(static_cast<double>(fluid_params.pressure_relative_residual),
                                          1.0e-8, 1.0e-2);
    const double tol_pre     = rel_tol_pre * rel_tol_pre;             // relative on r.z
    const int    max_iter_pre = std::max(1, fluid_params.pressure_iterations);

    // ── GPU fast path: device-resident CG scalars ────────────────────────────
    // The generic loop below downloads a dot product TWICE per iteration —
    // a full vkQueueSubmit/vkWaitForFences on Vulkan, a blocking stream sync
    // on CUDA (the HUD's "MGPCG dot sync" measured that at ~80% of the CUDA
    // pressure solve). Here alpha/beta/sigma live in a 7-double GPU buffer
    // (sim_fluid_cg_scalar_step tree-reduces the block partials on 256
    // threads); the host only synchronizes + downloads 56 bytes every K
    // iterations for the convergence check. Fused kernels (spmv+dot,
    // jacobi+dot, paired axpy) keep it at 6 dispatches per iteration.
    // GFM stays on the generic loop, and CUDA keeps the generic loop when the
    // multigrid preconditioner is available — this branch is Jacobi-only and must
    // not silently swap the preconditioner.
    // Vulkan may now take this loop WITH variational weights: its fused spmv+dot
    // dispatch below binds them and selects sim_fluid_cg_spmv_dot_var, and every
    // other kernel here is matrix-free apart from `diag`, which is already the
    // variational diagonal. That matters a lot: on the generic loop each dot costs
    // a submit+fence, measured at 216.96 ms across 193 syncs on a 4.6M-cell domain
    // — more than the arithmetic. CUDA keeps the old exclusion because its dispatch
    // in this loop still binds the plain 5-buffer form.
    const bool device_scalar_cg =
        gpu_buffers.cg_scalars.valid() && !is_gfm &&
        (compute->backendType() == ComputeBackendType::VulkanCompute ||
         (compute->backendType() == ComputeBackendType::CUDA && !is_variational &&
          (!fluid_params.pressure_multigrid_preconditioner ||
           gpu_buffers.mg_levels.empty())));
    if (device_scalar_cg) {
        const auto mg_loop_begin = SimulationClock::now();
        c.iterations = static_cast<int>(dot_blocks); // partials count for scalar_step

        // z = M^-1 r + partials(r.z) fused ; sigma0 ; s = z
        { ComputeBufferHandle b[4] = { gpu_buffers.cg_residual, gpu_buffers.cg_diag,
                                       gpu_buffers.cg_z, gpu_buffers.cg_partials };
          ok = dispatch1("sim_fluid_cg_jacobi_dot", b, 4, dot_blocks); }
        c.parity = 0; // op: sigma = sum; sigma0 = sigma
        { ComputeBufferHandle b[2] = { gpu_buffers.cg_partials, gpu_buffers.cg_scalars };
          ok = ok && dispatch1("sim_fluid_cg_scalar_step", b, 2, 1); }
        { ComputeBufferHandle b[2] = { gpu_buffers.cg_search, gpu_buffers.cg_z };
          ok = ok && dispatch1("sim_fluid_cg_copy", b, 2, cell_groups); }

        constexpr int kCheckEvery = 8;
        double host_scalars[7] = {};
        int done_iters = 0;
        while (ok && done_iters < max_iter_pre) {
            const int batch = std::min(kCheckEvery, max_iter_pre - done_iters);
            for (int k = 0; k < batch && ok; ++k) {
                // As = A s + partials(s.As) fused ; alpha = sigma/sAs
                { ComputeBufferHandle b[8] = { gpu_buffers.cg_search, gpu_buffers.fluid_mask,
                                               gpu_buffers.cg_diag, gpu_buffers.cg_As,
                                               gpu_buffers.cg_partials };
                  int n = 5;
                  if (vulkan_variational) {
                      b[5] = gpu_buffers.var_u_weight;
                      b[6] = gpu_buffers.var_v_weight;
                      b[7] = gpu_buffers.var_w_weight;
                      n = 8;
                  }
                  ok = dispatch1(varKernel("sim_fluid_cg_spmv_dot",
                                           "sim_fluid_cg_spmv_dot_var"),
                                 b, n, dot_blocks); }
                c.parity = 1; // op: sAs + alpha
                { ComputeBufferHandle b[2] = { gpu_buffers.cg_partials, gpu_buffers.cg_scalars };
                  ok = ok && dispatch1("sim_fluid_cg_scalar_step", b, 2, 1); }
                // p += alpha s ; r -= alpha As (fused pair)
                { ComputeBufferHandle b[5] = { gpu_buffers.pressure, gpu_buffers.cg_search,
                                               gpu_buffers.cg_residual, gpu_buffers.cg_As,
                                               gpu_buffers.cg_scalars };
                  ok = ok && dispatch1("sim_fluid_cg_axpy2_dev", b, 5, cell_groups); }
                // z = M^-1 r + partials(r.z) fused ; sigma_new ; beta
                { ComputeBufferHandle b[4] = { gpu_buffers.cg_residual, gpu_buffers.cg_diag,
                                               gpu_buffers.cg_z, gpu_buffers.cg_partials };
                  ok = ok && dispatch1("sim_fluid_cg_jacobi_dot", b, 4, dot_blocks); }
                c.parity = 2; // op: sigma_new + beta (+ sigma = sigma_new)
                { ComputeBufferHandle b[2] = { gpu_buffers.cg_partials, gpu_buffers.cg_scalars };
                  ok = ok && dispatch1("sim_fluid_cg_scalar_step", b, 2, 1); }
                // s = z + beta s
                { ComputeBufferHandle b[3] = { gpu_buffers.cg_search, gpu_buffers.cg_z, gpu_buffers.cg_scalars };
                  ok = ok && dispatch1("sim_fluid_cg_zpby_dev", b, 3, cell_groups); }
            }
            if (!ok) break;
            done_iters += batch;

            // Batched download = barrier + copy recorded into the same command
            // buffer, then ONE submit+fence for the whole K-iteration block
            // (the old synchronize + immediate download cost two fences).
            const auto dot_begin = SimulationClock::now();
            compute->beginTransferBatch();
            bool check_ok = compute->downloadBuffer(gpu_buffers.cg_scalars,
                                                    host_scalars, sizeof(host_scalars));
            check_ok = compute->endTransferBatch() && check_ok;
            if (!check_ok) {
                ok = false;
                break;
            }
            dot_ms += elapsedMilliseconds(dot_begin, SimulationClock::now());
            ++dot_count;

            const double sigma0_dev    = host_scalars[1];
            const double sigma_new_dev = host_scalars[5];
            if (host_scalars[6] != 0.0) break;              // degenerate (no fluid rows)
            if (sigma0_dev <= 0.0) break;                    // nothing to solve
            if (sigma_new_dev <= tol_pre * sigma0_dev) break; // converged
        }
        if (!ok) return bail("device-scalar CG loop");

        if (mgpcg_stats) {
            mgpcg_stats->pressure_cg_iterations = done_iters;
            mgpcg_stats->pressure_cg_max_iterations = max_iter_pre;
            mgpcg_stats->pressure_cg_dot_count = dot_count;
            mgpcg_stats->pressure_cg_dot_ms = dot_ms;
            mgpcg_stats->pressure_cg_multigrid = false;
            mgpcg_stats->pressure_cg_final_relative_residual =
                (host_scalars[1] > 0.0)
                    ? std::sqrt(std::max(0.0, host_scalars[5]) / host_scalars[1])
                    : 0.0;
        }

        // Subtract pressure gradient from velocity (shared tail below expects
        // the generic loop's locals — do it here and return directly).
        ComputeBufferHandle vk_grad_bufs[11] = {
            gpu_buffers.vel_x, gpu_buffers.vel_y, gpu_buffers.vel_z,
            gpu_buffers.pressure, gpu_buffers.fluid_mask
        };
        int vk_grad_count = 5;
        if (vulkan_variational) {
            vk_grad_bufs[5]  = gpu_buffers.var_u_weight;
            vk_grad_bufs[6]  = gpu_buffers.var_v_weight;
            vk_grad_bufs[7]  = gpu_buffers.var_w_weight;
            vk_grad_bufs[8]  = gpu_buffers.var_svx;
            vk_grad_bufs[9]  = gpu_buffers.var_svy;
            vk_grad_bufs[10] = gpu_buffers.var_svz;
            vk_grad_count = 11;
        }
        cmd.kernel          = varKernel("sim_fluid_subtract_gradient",
                                        "sim_fluid_subtract_gradient_var");
        cmd.buffers         = vk_grad_bufs;
        cmd.buffer_count    = vk_grad_count;
        cmd.constants       = &c;
        cmd.constants_size  = sizeof(c);
        cmd.groups.groups_x = (max_faces + threads - 1u) / threads;
        cmd.groups.groups_y = 1;
        cmd.groups.groups_z = 1;
        const auto mg_tail_begin = SimulationClock::now();
        ok = compute->dispatch(cmd);

        // No synchronize(): the download batch flushes gradient dispatch +
        // downloads in one submit (see the P2G tail note).
        compute->beginTransferBatch();
        ok = ok &&
             (gpu_buffers.matter_model.enabled ||
             compute->downloadBuffer(gpu_buffers.vel_x, grid.vel_x.data(),
                                     grid.vel_x.size() * sizeof(float))) &&
             (gpu_buffers.matter_model.enabled ||
             compute->downloadBuffer(gpu_buffers.vel_y, grid.vel_y.data(),
                                     grid.vel_y.size() * sizeof(float))) &&
             (gpu_buffers.matter_model.enabled ||
             compute->downloadBuffer(gpu_buffers.vel_z, grid.vel_z.data(),
                                     grid.vel_z.size() * sizeof(float)));
        ok = compute->endTransferBatch() && ok;

        // Phase breakdown, averaged and logged every ~240 substeps (see the
        // matching block in runGpuFluidP2G). cg=init+iterations+syncs,
        // tail=gradient dispatch+sync+velocity download.
        {
            static float s_up = 0.0f, s_cg = 0.0f, s_tail = 0.0f;
            static float s_iters = 0.0f;
            static int   s_n = 0;
            s_up   += mg_upload_ms;
            s_cg   += elapsedMilliseconds(mg_loop_begin, mg_tail_begin);
            s_tail += elapsedMilliseconds(mg_tail_begin, SimulationClock::now());
            s_iters += static_cast<float>(done_iters);
            if (++s_n >= 240) {
                const float inv = 1.0f / static_cast<float>(s_n);
                SCENE_LOG_INFO("[FluidGPU MGPCG avg ms] backend=" + std::string(compute->backendName()) +
                               " mode=dev-scalar" +
                               " upload=" + std::to_string(s_up * inv) +
                               " cg_loop=" + std::to_string(s_cg * inv) +
                               " grad+download=" + std::to_string(s_tail * inv) +
                               " iters=" + std::to_string(s_iters * inv) +
                               " cells=" + std::to_string(cell_count));
                s_up = s_cg = s_tail = s_iters = 0.0f;
                s_n = 0;
            }
        }
        return ok;
    }

    const auto mg_loop_begin = SimulationClock::now();
    double sigma = 0.0;
    if (!jacobiAndDot(sigma)) return false;
    // s = z
    { ComputeBufferHandle b[2] = { gpu_buffers.cg_search, gpu_buffers.cg_z };
      if (!dispatch1("sim_fluid_cg_copy", b, 2, cell_groups)) return false; }

    const double sigma0   = sigma;
    const double tol      = tol_pre;                                  // relative on r.z
    const int    max_iter = max_iter_pre;
    int iterations_used = 0;

    if (sigma0 > 0.0) {
        for (int iter = 0; iter < max_iter; ++iter) {
            double sAs = 0.0;
            if (!spmvAndDot(sAs)) { ok = false; break; }
            if (std::abs(sAs) < 1e-30) break; // degenerate (no fluid rows)
            const float alpha = static_cast<float>(sigma / sAs);

            // p += alpha s
            c.sor_omega = alpha;
            { ComputeBufferHandle b[2] = { gpu_buffers.pressure, gpu_buffers.cg_search };
              if (!dispatch1("sim_fluid_cg_axpy", b, 2, cell_groups)) { ok = false; break; } }
            // r -= alpha As
            c.sor_omega = -alpha;
            { ComputeBufferHandle b[2] = { gpu_buffers.cg_residual, gpu_buffers.cg_As };
              if (!dispatch1("sim_fluid_cg_axpy", b, 2, cell_groups)) { ok = false; break; } }

            double sigma_new = 0.0;
            if (!jacobiAndDot(sigma_new)) { ok = false; break; }
            iterations_used = iter + 1;
            if (sigma_new <= tol * sigma0) { sigma = sigma_new; break; }

            const float beta = static_cast<float>(sigma_new / sigma);
            // s = z + beta s
            c.sor_omega = beta;
            { ComputeBufferHandle b[2] = { gpu_buffers.cg_search, gpu_buffers.cg_z };
              if (!dispatch1("sim_fluid_cg_zpby", b, 2, cell_groups)) { ok = false; break; } }
            sigma = sigma_new;
        }
    }
    if (!ok) return false;
    if (mgpcg_stats) {
        mgpcg_stats->pressure_cg_iterations = iterations_used;
        mgpcg_stats->pressure_cg_max_iterations = max_iter;
        mgpcg_stats->pressure_cg_dot_count = dot_count;
        mgpcg_stats->pressure_cg_dot_ms = dot_ms;
        mgpcg_stats->pressure_cg_multigrid = used_multigrid;
        mgpcg_stats->pressure_cg_final_relative_residual =
            sigma0 > 0.0 ? std::sqrt(std::max(0.0, sigma) / sigma0) : 0.0;
    }

    // Subtract pressure gradient from velocity.
    ComputeBufferHandle fluid_gradient_bufs[12];
    fluid_gradient_bufs[0] = gpu_buffers.vel_x;
    fluid_gradient_bufs[1] = gpu_buffers.vel_y;
    fluid_gradient_bufs[2] = gpu_buffers.vel_z;
    fluid_gradient_bufs[3] = gpu_buffers.pressure;
    fluid_gradient_bufs[4] = gpu_buffers.fluid_mask;
    if (is_variational) {
        fluid_gradient_bufs[5] = gpu_buffers.var_u_weight;
        fluid_gradient_bufs[6] = gpu_buffers.var_v_weight;
        fluid_gradient_bufs[7] = gpu_buffers.var_w_weight;
        fluid_gradient_bufs[8] = gpu_buffers.var_svx;
        fluid_gradient_bufs[9] = gpu_buffers.var_svy;
        fluid_gradient_bufs[10] = gpu_buffers.var_svz;
    }
    int grad_buf_count = is_variational ? 11 : 5;
    if (is_gfm) {
        fluid_gradient_bufs[grad_buf_count] = gpu_buffers.var_fluid_phi;
        grad_buf_count++;
    }

    cmd.kernel          = use_cuda_fluid_projection
        ? varKernel("sim_fluid_subtract_gradient", "sim_fluid_subtract_gradient_var")
        : "sim_grid_subtract_gradient";
    cmd.buffers         = use_cuda_fluid_projection ? fluid_gradient_bufs : proj_bufs;
    cmd.buffer_count    = use_cuda_fluid_projection ? grad_buf_count : 5;
    cmd.groups.groups_x = (max_faces + threads - 1u) / threads;
    cmd.groups.groups_y = 1;
    cmd.groups.groups_z = 1;
    const auto mg_tail_begin = SimulationClock::now();
    ok = compute->dispatch(cmd);

    // Download updated velocities for CPU boundary re-enforcement and G2P.
    // No synchronize(): the batch flushes in one submit (see the P2G note).
    compute->beginTransferBatch();
    ok = ok &&
         (gpu_buffers.matter_model.enabled ||
             compute->downloadBuffer(gpu_buffers.vel_x, grid.vel_x.data(),
                                     grid.vel_x.size() * sizeof(float))) &&
         (gpu_buffers.matter_model.enabled ||
             compute->downloadBuffer(gpu_buffers.vel_y, grid.vel_y.data(),
                                     grid.vel_y.size() * sizeof(float))) &&
         (gpu_buffers.matter_model.enabled ||
             compute->downloadBuffer(gpu_buffers.vel_z, grid.vel_z.data(),
                                     grid.vel_z.size() * sizeof(float)));
    ok = compute->endTransferBatch() && ok;

    // Phase breakdown for the generic/CUDA path — mirrors the Vulkan
    // device-scalar branch above so both backends print a comparable line.
    {
        static float s_up = 0.0f, s_cg = 0.0f, s_tail = 0.0f;
        static float s_iters = 0.0f;
        static int   s_n = 0;
        s_up   += mg_upload_ms;
        s_cg   += elapsedMilliseconds(mg_loop_begin, mg_tail_begin);
        s_tail += elapsedMilliseconds(mg_tail_begin, SimulationClock::now());
        s_iters += static_cast<float>(iterations_used);
        if (++s_n >= 240) {
            const float inv = 1.0f / static_cast<float>(s_n);
            SCENE_LOG_INFO("[FluidGPU MGPCG avg ms] backend=" + std::string(compute->backendName()) +
                           " mode=generic" +
                           " upload=" + std::to_string(s_up * inv) +
                           " cg_loop=" + std::to_string(s_cg * inv) +
                           " grad+download=" + std::to_string(s_tail * inv) +
                           " iters=" + std::to_string(s_iters * inv) +
                           " cells=" + std::to_string(cell_count));
            s_up = s_cg = s_tail = s_iters = 0.0f;
            s_n = 0;
        }
    }
    return ok;
}
