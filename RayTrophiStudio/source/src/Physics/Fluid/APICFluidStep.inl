void step(FluidParticles& particles,
          FluidSim::FluidGrid& grid,
          const APICSolverParams& params,
          float dt,
          const SimulationForceFieldSnapshot* forces,
          float time_seconds,
          APICSolverStats* stats) {
    APICSolverStats local_stats{};
    APICSolverStats& out_stats = stats ? *stats : local_stats;
    out_stats = APICSolverStats{};
    out_stats.cpu_threads = solverThreadCount(params);
    out_stats.particle_count = particles.size();
    out_stats.grid_cell_count = static_cast<size_t>(grid.nx) *
                                static_cast<size_t>(grid.ny) *
                                static_cast<size_t>(grid.nz);

    if (particles.empty() || dt <= 0.0f) return;

#ifdef APIC_DEBUG_SKIP_STEP
    // Solver disabled for bisect — sim ticks, fluid stage no-ops.
    return;
#endif

    const auto total_begin = SolverClock::now();


    auto clampVelocity = [&](Vec3& v) {
        const float speed = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        if (speed > params.max_velocity && speed > 1e-6f) {
            v = v * (params.max_velocity / speed);
        }
    };
    const int thread_count = solverThreadCount(params);
    const bool particle_parallel = shouldParallelParticles(params, particles.size());

    // ── Solid-phase parcels ─────────────────────────────────────────────────
    // Resolved ONCE per step into a per-particle mask. The tag list is short
    // (kMaxFluidSubstanceMaterials at most), but the particle stages that need
    // the answer run per substep, so a scan there would multiply by CFL.
    //
    // ★ Null mask, not an empty one, when the domain declares no solid: every
    // consumer then compiles down to exactly the code it ran before this
    // existed, which is what keeps liquid-only domains bit-identical.
    static std::vector<uint8_t> s_solid_particle;
    const uint8_t* solid_particle = nullptr;
    bool any_frozen = false;
    if (buildMatterObstacleMask(particles, params.solid_substance_tags,
                                s_solid_particle, any_frozen)) {
        solid_particle = s_solid_particle.data();
    }

    // 1. External forces on particles (gravity + force fields, incl. the wind
    // surface-drag model). Factored into applyExternalForces so the GPU pipeline
    // can run it on the CPU then upload the post-force velocities for a GPU P2G.
    auto stage_begin = SolverClock::now();
    if (!params.external_forces_preintegrated) {
        applyExternalForces(particles, grid, params, forces, time_seconds, dt);
    }
    // Frozen thermal-liquid parcels are PINNED. Gravity was just added to them
    // (here, or on the device before this call), and a solid parcel advects by
    // its own velocity — so without this a frozen layer on a vertical face
    // slides down it like sand, one g·dt² per step. Zeroed after the force
    // stage on every call, including the split-step tails, because every call
    // that advects must see the pinned value.
    if (any_frozen) {
        for (std::size_t pi = 0; pi < particles.size(); ++pi) {
            if (!(particles.flags[pi] & kParticleFlagFrozen)) continue;
            particles.velocity[pi] = Vec3(0.0f, 0.0f, 0.0f);
            if (pi < particles.affine.size()) particles.affine[pi] = AffineC{};
        }
    }
    auto stage_end = SolverClock::now();
    out_stats.forces_ms = elapsedMs(stage_begin, stage_end);

    // 2. Particle -> grid (APIC scatter)
    stage_begin = SolverClock::now();
    if (!params.p2g_precomputed) {
        particleToGrid(particles, grid, params, dt);
    }
    stage_end = SolverClock::now();
    out_stats.p2g_ms = elapsedMs(stage_begin, stage_end);
    out_stats.p2g_on_gpu = params.p2g_precomputed;

    // ── 3. FLIP snapshot ────────────────────────────────────────────────────
    // ★ ROOT CAUSE OF "THE VISCOSITY DIAL DOES NOTHING".
    // The FLIP update is  v_p = v_grid_post + flip·(v_p_old - v_snapshot).
    // This snapshot used to be taken AFTER boundary+viscosity, so writing
    // v_grid_post = v_visc + Δp and v_snapshot = v_visc gives
    //     v_p = (1-flip)·v_visc + flip·v_p_old + Δp
    // — the viscous field survives with weight (1-flip). At the water default
    // flip_blend=0.97 that is THREE PERCENT, and the FLIP fraction simply kept
    // its own old velocity, untouched by any viscosity at all. Every viscous
    // preset was therefore forced to fake thickness with per-particle friction
    // and damping, which drags the whole body instead of resisting shear.
    // Snapshotting here — right after the transfer, before any grid-side force —
    // is the standard FLIP contract: EVERY grid change (solid boundaries,
    // viscosity, pressure) becomes part of the delta the particles receive.
    static std::vector<float> vel_x_pre_buf;
    static std::vector<float> vel_y_pre_buf;
    static std::vector<float> vel_z_pre_buf;
    // ★★★★★ FLIP MUST BE OFF FOR GRANULAR, AND THIS IS WHY YOUNG MODULUS
    // LOOKED LIKE IT DID NOTHING ON THE CPU PATH.
    //
    // FLIP reconstructs v_new = v_old + (grid_post - grid_pre). The snapshot
    // above is taken right after P2G, which for granular ALREADY CONTAINS the
    // elastic stress divergence — and granular skips the pressure projection,
    // so nothing changes the grid afterwards. grid_post == grid_pre, the delta
    // is exactly zero, and the particle simply keeps its old velocity: the
    // entire elastic response is subtracted back out. At the Sand preset's
    // flip_blend = 0.92 only 8% of the stress survived, so the pile lost volume
    // and collapsed while the same parameters on Vulkan held their shape.
    //
    // ★ The device path has always gated this (`has_flip = !granular_enabled`
    // in ParticleSimulation.cpp). The CPU port reproduced the stress kernels
    // faithfully and missed the gate — which is the harder half to notice,
    // because the stress WAS being computed correctly the whole time. The two
    // backends have to agree about which stages exist, not only about the maths
    // inside each stage.
    const bool want_flip = params.flip_blend > 0.0f && params.free_surface &&
                           !params.granular_enabled;
    if (!params.pressure_g2p_precomputed && want_flip && !params.model_flip_snapshot) {
        if (vel_x_pre_buf.size() < grid.vel_x.size()) vel_x_pre_buf.resize(grid.vel_x.size());
        if (vel_y_pre_buf.size() < grid.vel_y.size()) vel_y_pre_buf.resize(grid.vel_y.size());
        if (vel_z_pre_buf.size() < grid.vel_z.size()) vel_z_pre_buf.resize(grid.vel_z.size());
        std::copy(grid.vel_x.begin(), grid.vel_x.end(), vel_x_pre_buf.begin());
        std::copy(grid.vel_y.begin(), grid.vel_y.end(), vel_y_pre_buf.begin());
        std::copy(grid.vel_z.begin(), grid.vel_z.end(), vel_z_pre_buf.begin());
        // Publish for the GPU split-step: the caller uploads THIS, not
        // grid.vel_*, which by then carries the viscous solve as well. Only on
        // the split-step path — a CPU-only step reads vel_*_pre_buf directly, so
        // copying three more face arrays for it would be pure waste.
        if (params.stop_before_pressure || params.stop_after_projection) {
            g_flip_snap_valid = true;
            g_flip_snap_x.assign(vel_x_pre_buf.begin(),
                                 vel_x_pre_buf.begin() + grid.vel_x.size());
            g_flip_snap_y.assign(vel_y_pre_buf.begin(),
                                 vel_y_pre_buf.begin() + grid.vel_y.size());
            g_flip_snap_z.assign(vel_z_pre_buf.begin(),
                                 vel_z_pre_buf.begin() + grid.vel_z.size());
        }
    } else if (!params.pressure_g2p_precomputed) {
        g_flip_snap_valid = false;
        g_flip_snap_x.clear(); g_flip_snap_y.clear(); g_flip_snap_z.clear();
    }

    // 3b. Solid boundary enforcement + viscous diffusion.
    //     Skipped in GPU split-step's second call (pressure_g2p_precomputed=true)
    //     because these were already run in the first call.
    if (!params.pressure_g2p_precomputed) {
        stage_begin = SolverClock::now();
        enforceSolidBoundaries(grid, params);
        stage_end = SolverClock::now();
        out_stats.boundary_ms = elapsedMs(stage_begin, stage_end);

        stage_begin = SolverClock::now();
        if (!params.viscosity_precomputed) {
            // The viscous stencil needs to tell fluid faces from air faces —
            // that distinction IS the stress-free surface condition. Build the
            // same cell mask the pressure solve and the GPU kernel use.
            static std::vector<float> visc_mask;
            const std::vector<float>* mask_ptr = nullptr;
            // ★ The per-substance field counts as "there is viscosity here"
            // exactly like the scalar does. Without this, an inviscid domain
            // with a thick substance would run the stencil with NO free-surface
            // mask, so it would treat air faces as walls and brake the surface
            // against nothing — a viscous skin, not a viscous liquid.
            const bool nu_field_present =
                params.substance_viscosity != nullptr &&
                params.substance_viscosity->size() == grid.getCellCount();
            if (params.free_surface &&
                (params.kinematic_viscosity > 0.0f || nu_field_present)) {
                buildViscosityFluidMask(grid, particles, visc_mask);
                mask_ptr = &visc_mask;
            }
            int sweeps_run = 0;
            applyViscosity(grid, params, dt, mask_ptr, sweeps_run);
            out_stats.viscosity_sweeps_run = sweeps_run;
            if (sweeps_run > 0) {
                enforceSolidBoundaries(grid, params);
            }
        }
        stage_end = SolverClock::now();
        out_stats.viscosity_ms = elapsedMs(stage_begin, stage_end);
    }

    // GPU split-step call 1: return after P2G + snapshot + boundary (+ CPU
    // viscosity when the device did not take it) so the caller can run the
    // device viscosity → pressure → G2P before the second call does the tail.
    if (params.stop_before_pressure) {
        out_stats.total_ms = elapsedMs(total_begin, SolverClock::now());
        return;
    }

    // 4–6. Pressure projection + G2P.
    //       When pressure_g2p_precomputed is set, the GPU path has already
    //       run these stages and downloaded updated particle velocities
    //       and affine matrices to CPU. Skip them here; advect + reseed still
    //       run on CPU below using the GPU-written velocity data.
    stage_begin = SolverClock::now();
    if (!params.pressure_g2p_precomputed) {
        // 5. Pressure projection (incompressibility).
        //
        // ★★★ GRANULAR MPM IS COMPRESSIBLE AND MUST NOT BE PROJECTED. Its
        // volumetric response already lives in the elastic stress that
        // granularStressToGrid pushed into the grid; running the liquid
        // incompressibility solve on top of it enforces volume twice and blows
        // the pressure up immediately. The Vulkan path skips this stage for the
        // same reason — the two backends have to agree about which forces exist,
        // not merely about how each one is computed.
        auto pressure_begin = SolverClock::now();
        if (params.pressure_precomputed) {
            // The independent model field was projected before contact.
        } else if (params.granular_enabled) {
            // No projection. Report the cell count so the panel does not read
            // this as "the solve found nothing to do".
            out_stats.active_fluid_cells = out_stats.grid_cell_count;
        } else if (params.free_surface) {
            out_stats.active_fluid_cells = projectPressureFreeSurface(particles,
                                                                      grid,
                                                                      params,
                                                                      dt,
                                                                      out_stats.sealed_pockets,
                                                                      out_stats.sealed_pocket_cells,
                                                                      out_stats.interior_fluid_cells);
            out_stats.sealed_pockets_measured = true;
        } else {
            GridFluid::SolverParams pp{};
            pp.pressure_iterations = params.pressure_iterations;
            pp.sor_omega           = params.sor_omega;
            pp.boundary            = GridFluid::Boundary::Closed;
            GridFluid::projectPressure(grid, pp, dt);
            out_stats.active_fluid_cells = out_stats.grid_cell_count;
        }
        out_stats.pressure_ms = elapsedMs(pressure_begin, SolverClock::now());

        if (params.stop_after_projection) {
            out_stats.total_ms = elapsedMs(total_begin, SolverClock::now());
            return;
        }

        // 6. Grid -> particle (APIC affine + PIC/FLIP linear blend + friction)
        gridToParticle(particles,
                       grid,
                       params,
                       dt,
                       want_flip ? (params.model_flip_snapshot
                           ? params.model_flip_snapshot->x.data() : vel_x_pre_buf.data()) : nullptr,
                       want_flip ? (params.model_flip_snapshot
                           ? params.model_flip_snapshot->y.data() : vel_y_pre_buf.data()) : nullptr,
                       want_flip ? (params.model_flip_snapshot
                           ? params.model_flip_snapshot->z.data() : vel_z_pre_buf.data()) : nullptr,
                       solid_particle);
    } else {
        // pressure_g2p_precomputed: GPU already handled pressure+G2P,
        // particle velocities+affine are downloaded — only tail stages remain.
    }

    // 6a. Granular constitutive update + settle (CPU path).
    //
    // ★★★ Runs only when the DEVICE did not already do it. `pressure_g2p_precomputed`
    // means the Vulkan G2P dispatch already ran the stress update and settle
    // kernels and downloaded the result; repeating them here would apply the
    // material model twice per step — which does not crash, it just makes the
    // material roughly twice as stiff and sends the whole preset table off
    // calibration. Exactly the "producer != consumer" trap this codebase keeps
    // paying for.
    if (params.granular_enabled && !params.pressure_g2p_precomputed &&
        !params.particle_tail_precomputed) {
        const auto granular_begin = SolverClock::now();
        particles.ensureGranularStateSize();
        namespace G = RayTrophiSim::Fluid::Granular;
        G::StressUpdateParams sp;
        sp.dt = dt;
        sp.young_modulus = std::max(params.granular_young_modulus, 1.0f);
        sp.poisson_ratio = params.granular_poisson_ratio;
        sp.friction_tangent = std::tan(std::clamp(
            params.granular_friction_angle_degrees * 0.017453292519943295f,
            0.0f, 1.3962634f));
        sp.cohesion = std::max(params.granular_cohesion, 0.0f);
        sp.dilatancy_tangent = std::tan(std::clamp(
            params.granular_dilatancy_degrees * 0.017453292519943295f,
            0.0f, 0.7853982f));
        sp.tensile_cutoff = std::max(params.granular_tensile_cutoff, 0.0f);
        sp.fracture_strain = std::max(params.granular_fracture_strain, 1.0e-5f);
        sp.damage_rate = std::max(params.granular_damage_rate, 0.0f);
        sp.healing_rate = std::max(params.granular_healing_rate, 0.0f);
        sp.rebonding = params.granular_rebonding;
        sp.hardening_coefficient = std::max(params.granular_hardening, 0.0f);
        sp.compaction_hardening = params.granular_compaction_hardening;
        sp.compaction_limit = params.granular_compaction_limit;
        sp.max_stored_strain = G::kGranularMaxStoredStrain;

        G::SettleParams settle;
        settle.dt = dt;

        const int64_t count = static_cast<int64_t>(particles.size());
        const int threads = solverThreadCount(params);
        const bool par = threads > 1 &&
                         shouldParallelParticles(params, particles.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(par)
#endif
        for (int64_t i = 0; i < count; ++i) {
            const size_t p = static_cast<size_t>(i);
            uint32_t flags = 0u;
            const float soft = p < particles.granular_softening.size()
                ? particles.granular_softening[p] : 1.0f;
            const float bond = p < particles.granular_bond_scale.size()
                ? particles.granular_bond_scale[p] : 1.0f;
            G::stressUpdateParticle(
                particles.affine[p], sp, soft, bond,
                particles.granular_deformation_col0[p],
                particles.granular_deformation_col1[p],
                particles.granular_deformation_col2[p],
                particles.granular_stress_diag[p],
                particles.granular_stress_shear[p],
                particles.granular_plastic_volume[p],
                particles.granular_damage[p],
                particles.granular_hardening[p],
                particles.granular_fracture_history[p],
                particles.granular_yield_value[p],
                particles.granular_plastic_increment[p],
                flags);
            // Settle runs after the stress update on the device too, and reads
            // the stress this step just wrote — keep the order.
            G::settleParticle(particles.granular_stress_diag[p], settle,
                              particles.velocity[p], flags);
            particles.granular_material_flags[p] = flags;
        }
        (void)par;

        // ★ The CPU path must report the SAME counters the device path does.
        // A backend that runs but reports nothing is indistinguishable from one
        // that never ran, and the strain/compaction rows are precisely how a
        // soft-material run is told apart from a diverging one.
        double damage_sum = 0.0, plastic_sum = 0.0, bond_sum = 0.0;
        out_stats.granular_min_softening = 1.0f;
        for (size_t p = 0; p < particles.size(); ++p) {
            const uint32_t f = particles.granular_material_flags[p];
            out_stats.granular_yielded_particles += (f & G::kFlagYielded) != 0u;
            out_stats.granular_detached_particles += (f & G::kFlagDetached) != 0u;
            out_stats.granular_invalid_particles += (f & G::kFlagInvalid) != 0u;
            out_stats.granular_sleeping_particles += (f & G::kFlagSleeping) != 0u;
            out_stats.granular_strain_limited_particles +=
                (f & G::kFlagStrainLimited) != 0u;
            out_stats.granular_compaction_capped_particles +=
                (f & G::kFlagCompactionCapped) != 0u;
            if (p < particles.granular_softening.size()) {
                const float soft_p = particles.granular_softening[p];
                if (std::isfinite(soft_p)) {
                    out_stats.granular_min_softening =
                        std::min(out_stats.granular_min_softening, soft_p);
                    out_stats.granular_softened_particles += soft_p < 0.999f;
                }
            }
            const float dmg = particles.granular_damage[p];
            if (std::isfinite(dmg)) {
                out_stats.granular_max_damage = std::max(out_stats.granular_max_damage, dmg);
                out_stats.granular_damaged_particles += dmg > 1.0e-4f;
                out_stats.granular_damage_over_10_particles += dmg >= 0.10f;
                out_stats.granular_damage_over_50_particles += dmg >= 0.50f;
                out_stats.granular_damage_over_90_particles += dmg >= 0.90f;
                damage_sum += dmg;
            }
            const float yv = particles.granular_yield_value[p];
            if (std::isfinite(yv))
                out_stats.granular_max_yield_value = std::max(out_stats.granular_max_yield_value, yv);
            const float pi = particles.granular_plastic_increment[p];
            if (std::isfinite(pi))
                out_stats.granular_max_plastic_increment =
                    std::max(out_stats.granular_max_plastic_increment, pi);
            const float ap = particles.granular_hardening[p];
            if (std::isfinite(ap)) {
                out_stats.granular_max_accumulated_plastic =
                    std::max(out_stats.granular_max_accumulated_plastic, ap);
                plastic_sum += ap;
            }
            const float bo = particles.granular_fracture_history[p];
            if (std::isfinite(bo)) {
                out_stats.granular_max_fracture_history =
                    std::max(out_stats.granular_max_fracture_history, bo);
                bond_sum += bo;
            }
        }
        if (!particles.empty()) {
            const double inv_n = 1.0 / static_cast<double>(particles.size());
            out_stats.granular_mean_damage = static_cast<float>(damage_sum * inv_n);
            out_stats.granular_mean_accumulated_plastic =
                static_cast<float>(plastic_sum * inv_n);
            out_stats.granular_mean_fracture_history =
                static_cast<float>(bond_sum * inv_n);
        }
        out_stats.granular_constitutive_ms =
            elapsedMs(granular_begin, SolverClock::now());
    }

    // 6b. Air drag for spray/droplet particles. A particle whose containing
    //     cell has fewer than `reseed_min_per_cell` neighbours is treated as
    //     "in air"; quadratic drag F = -k|v|v is integrated implicitly via
    //     v *= 1/(1 + k|v|dt) — unconditionally stable, no inner iterate.
    //     Bulk particles are skipped (their dissipation comes from
    //     internal_friction inside gridToParticle).
    if (!params.particle_tail_precomputed &&
        params.air_drag > 0.0f && dt > 0.0f) {
        const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
        const float invH = (grid.voxel_size > 1e-6f) ? (1.0f / grid.voxel_size) : 0.0f;
        const int air_threshold = std::max(1, params.reseed_min_per_cell);
        const float air_k = params.air_drag;
        const Vec3 air_wind = atmosphereAirVelocity(params);

        // Build per-cell particle counts (fresh from post-G2P positions —
        // pre-advect). Function-static to avoid heap stalls under contention.
        static std::vector<int> air_cell_count_buf;
        const std::size_t total = static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
        if (air_cell_count_buf.size() < total) air_cell_count_buf.assign(total, 0);
        else std::fill(air_cell_count_buf.begin(), air_cell_count_buf.begin() + total, 0);

        for (std::size_t pi = 0; pi < particles.size(); ++pi) {
            const Vec3 gp = (particles.position[pi] - grid.origin) * invH;
            const int i = static_cast<int>(std::floor(gp.x));
            const int j = static_cast<int>(std::floor(gp.y));
            const int k = static_cast<int>(std::floor(gp.z));
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
            ++air_cell_count_buf[grid.cellIndex(i, j, k)];
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(particle_parallel)
#endif
        for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(particles.size()); ++raw_i) {
            const std::size_t pi = static_cast<std::size_t>(raw_i);
            const Vec3 gp = (particles.position[pi] - grid.origin) * invH;
            const int i = static_cast<int>(std::floor(gp.x));
            const int j = static_cast<int>(std::floor(gp.y));
            const int k = static_cast<int>(std::floor(gp.z));
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
            const int count = air_cell_count_buf[grid.cellIndex(i, j, k)];
            if (count >= air_threshold) continue; // dense cell — certainly bulk

            // ★★ A sparse CELL is not the same thing as a detached DROPLET.
            //
            // This used to drag every particle whose own cell held fewer than
            // air_threshold particles. A thin sheet of liquid lying on the floor
            // is exactly that — one cell deep, a couple of particles per cell —
            // so the entire spread of a spill was being treated as airborne
            // spray and given quadratic drag. It stopped following a moving
            // container and visibly scattered. Reseed used to hide this by
            // pumping every sparse cell up to the bulk target, which is how the
            // liquid gained mass on impact; the two faults were propping each
            // other up.
            //
            // Spray is defined by ISOLATION, not by local thinness: sum the
            // 3x3x3 neighbourhood, and only drag a particle whose neighbourhood
            // is also nearly empty. A film has plenty of in-plane neighbours; a
            // droplet flying through air has none. Only reached by particles
            // that already failed the cheap test above, so the 27 taps cost
            // nothing on bulk fluid.
            const int isolation_threshold = std::max(4, air_threshold + 2);
            int neighbourhood = 0;
            for (int dk = -1; dk <= 1 && neighbourhood < isolation_threshold; ++dk) {
                const int nk2 = k + dk;
                if (nk2 < 0 || nk2 >= nz) continue;
                for (int dj = -1; dj <= 1 && neighbourhood < isolation_threshold; ++dj) {
                    const int nj2 = j + dj;
                    if (nj2 < 0 || nj2 >= ny) continue;
                    for (int di = -1; di <= 1; ++di) {
                        const int ni2 = i + di;
                        if (ni2 < 0 || ni2 >= nx) continue;
                        neighbourhood += air_cell_count_buf[grid.cellIndex(ni2, nj2, nk2)];
                    }
                }
            }
            if (neighbourhood >= isolation_threshold) continue; // sheet/film — bulk

            // Drag acts on the velocity RELATIVE to the air (Faz 2): with a
            // climate wind a droplet is carried downwind, not braked to rest.
            // air_wind is zero when not inheriting, which is the old formula.
            Vec3& v = particles.velocity[pi];
            const Vec3 rel = v - air_wind;
            const float speed_sq = rel.x * rel.x + rel.y * rel.y + rel.z * rel.z;
            if (speed_sq < 1e-8f) continue;
            const float speed = std::sqrt(speed_sq);
            const float decay = 1.0f / (1.0f + air_k * speed * dt);
            v = air_wind + rel * decay;
        }
    }

    if (!params.particle_tail_precomputed) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(particle_parallel)
#endif
        for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(particles.velocity.size()); ++raw_i) {
            Vec3& v = particles.velocity[static_cast<size_t>(raw_i)];
            v = v * std::clamp(params.velocity_damping, 0.0f, 1.0f);
            clampVelocity(v);
        }
    }
    stage_end = SolverClock::now();
    out_stats.g2p_ms = elapsedMs(stage_begin, stage_end); // includes air drag + damping

    // 7. Advect particle positions
    stage_begin = SolverClock::now();
    out_stats.advect_substeps = params.particle_tail_precomputed ? 0 :
        advectParticles(particles,
                        grid,
                        dt,
                        params,
                        params.max_velocity,
                        params.wall_damping,
                        params.cfl,
                        params.max_substeps,
                        solid_particle);
    stage_end = SolverClock::now();
    out_stats.advect_ms = elapsedMs(stage_begin, stage_end);

    // 8. Conservative local redistribution; no particles are added or removed.
    const uint32_t reseed_seed =
        static_cast<uint32_t>(out_stats.particle_count) * 2654435761u ^
        static_cast<uint32_t>(std::llround(time_seconds * 1000.0));
    redistributeFluidParticles(particles, grid, params, reseed_seed);
    out_stats.particle_count = particles.size();

    // 9. Advance the material-coordinate refresh schedule and reset whichever
    //    generation is due.
    //
    //    ★ LAST, and after reseed on purpose. A generation reset assigns
    //    uvw = position, so it must see the FINAL particle set of this step:
    //    reset before advection and the coordinate describes where the liquid
    //    was, not where it is; reset before reseed and the particles created
    //    this step never get the fresh generation and stay one period stale.
    //
    //    ★ The early-out at the top of this function (empty or dt <= 0) skips
    //    this too, which is correct — a step that did not advance the liquid
    //    must not age its coordinates either, or a paused sim would keep
    //    refreshing a body that is not deforming.
    particles.uvw_refresh_period = params.uvw_refresh_period;
    particles.advanceMaterialCoordinates();

    out_stats.total_ms = elapsedMs(total_begin, stage_end);
}
