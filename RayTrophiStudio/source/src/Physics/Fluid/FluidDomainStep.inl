// Statement fragment included inside ParticleSimulationSystem::stepGridDomains.
// Its locals and types come from ParticleSimulation.cpp, not a standalone unit.
#if !defined(__INTELLISENSE__) || defined(RT_FLUID_DOMAIN_STEP_CONTEXT)

            const auto fluid_step_begin = SimulationClock::now();
            SimulationTransferProbeScope fluid_transfer_probe(
                context.compute, state.fluid_transfer_stats);
            if (i < grid_domain_compute_buffers_.size()) {
                FluidGpuParticleUpload::invalidate(grid_domain_compute_buffers_[i]);
            }
            // The domain's physics come from its Default Substance, every step,
            // so an edit to the substance reaches the solver without a second
            // copy anywhere. Written on the descriptor itself: panel and IPC
            // reads see what the solver runs.
            if (i < grid_domains_.size()) {
                static std::string s_last_substance_error;
                std::string substance_error;
                if (!Fluid::resolveDomainSubstancePhysics(grid_domains_[i].fluid_params,
                        state.voxel_size, world_thermal_.scale(), substance_error)) {
                    if (substance_error != s_last_substance_error) {
                        SCENE_LOG_WARN("[Fluid] " + grid_domains_[i].name + ": " + substance_error +
                                       "; the domain keeps its last resolved physics");
                    }
                }
                s_last_substance_error = substance_error;
            }
            auto fluid_params = (i < grid_domains_.size())
                ? grid_domains_[i].fluid_params
                : Fluid::APICSolverParams{};
            // Pore-enabled dry runs must keep the same solver when water arrives
            // or wet response is toggled; saturation zero is the neutral limit.
            const bool mixed_models = matter_domain &&
                (fluid_params.grain.enabled || fluid_params.pore_exchange.enabled ||
                 fluid_params.pore_exchange.wet_response_enabled ||
                 Fluid::hasMixedMatterModels(state.particles, fluid_params.granular_enabled) ||
                 Fluid::needsMatterPoreTransport(state.particles, fluid_params.pore_exchange));
            // The grain solver owns every granular carrier; the legacy MPM
            // granular switch must not change Auto ownership, mass or radius.
            const bool mixed_legacy_granular = fluid_params.granular_enabled &&
                !fluid_params.grain.enabled;
            if (matter_domain && !mixed_models) {
                const auto model = Fluid::resolveSingleMatterModel(
                    state.particles, mixed_legacy_granular);
                if (model == Fluid::MatterConstitutiveModel::Fluid) {
                    fluid_params.granular_enabled = false;
                } else if (model == Fluid::MatterConstitutiveModel::Granular) {
                    fluid_params.granular_enabled = true;
                }
            }
            const auto mixed_ids_before = mixed_models
                ? state.particles.particle_id : std::vector<uint64_t>{};
            const auto mixed_velocity_before = mixed_models
                ? state.particles.velocity : std::vector<Vec3>{};
            if (mixed_models) {
                // The mixed coordinator owns constitutive subcycling; domain-wide
                // granular stress/settle must not run over liquid parcels too.
                fluid_params.granular_enabled = false;
                fluid_params.mixed_working_set_budget_bytes =
                    i < grid_domains_.size() && grid_domains_[i].enforce_resource_budget
                    ? static_cast<std::size_t>(grid_domains_[i].resource_budget_mb) * 1024 * 1024
                    : 0;
            }
            // Granular carriers in a grain-owned domain carry sphere mass; any
            // birth path that left it unset gets the same value as the emitter.
            if (fluid_params.grain.enabled) {
                Fluid::ensureMatterGrainRestMasses(state.particles, fluid_params.grain,
                    Fluid::domainSubstance(fluid_params), mixed_legacy_granular);
            }
            Fluid::ensureFluidParticleRestMasses(
                state.particles,
                Fluid::domainSubstance(fluid_params),
                state.voxel_size,
                fluid_params.particles_per_cell, mixed_legacy_granular);
            // Mirror the domain wall mode onto the particle solver so "Open
            // (Outflow)" actually drains instead of clamping like a sealed box.
            if (i < grid_domains_.size()) {
                switch (grid_domains_[i].boundary_mode) {
                    case SimulationGridDomainBoundaryMode::Open:
                        fluid_params.boundary = Fluid::APICSolverParams::BoundaryMode::Open;
                        break;
                    case SimulationGridDomainBoundaryMode::Periodic:
                        fluid_params.boundary = Fluid::APICSolverParams::BoundaryMode::Periodic;
                        break;
                    case SimulationGridDomainBoundaryMode::Closed:
                    default:
                        fluid_params.boundary = Fluid::APICSolverParams::BoundaryMode::Closed;
                        break;
                }
            }
            const float motion_coupling = std::clamp(fluid_params.domain_motion_coupling, 0.0f, 1.0f);
            const float motion_mag =
                std::abs(state.domain_motion_delta.x) +
                std::abs(state.domain_motion_delta.y) +
                std::abs(state.domain_motion_delta.z);
            Vec3 container_velocity_delta =
                (dt > 1e-6f && motion_coupling > 0.0f && motion_mag > 1e-7f)
                    ? state.domain_motion_delta * (motion_coupling / dt)
                    : Vec3(0.0f, 0.0f, 0.0f);
            // The delta may have accumulated over many UI syncs since the last
            // step (dragging the gizmo with the timeline parked). Dividing a
            // whole drag by one step's dt would launch the liquid, so cap the
            // impulse at the solver's own velocity ceiling — a big drag then
            // reads as a hard slosh instead of an explosion.
            {
                const float impulse = container_velocity_delta.length();
                const float ceiling = std::max(0.0f, fluid_params.max_velocity);
                if (ceiling > 0.0f && impulse > ceiling) {
                    container_velocity_delta = container_velocity_delta * (ceiling / impulse);
                }
            }
            // Consumed: clear it so the next step does not re-apply the same
            // motion. This is the ONLY place that clears it for a fluid domain.
            state.domain_motion_delta = Vec3(0.0f, 0.0f, 0.0f);
            bool gpu_integrated_forces = false;
            // True once force fields have been evaluated on the CPU (force-field
            // branch below). The forces are then already baked into the particle
            // velocities, so the downstream step() calls must NOT re-apply them and
            // the container-velocity fold must not run twice — even if the GPU
            // upload that follows fails and we drop back to a CPU P2G.
            bool cpu_forces_applied = false;
            const bool fluid_gpu_requested = (i < grid_domains_.size()) &&
                (grid_domains_[i].backend == SimulationDomainBackend::GPU_CUDA ||
                 grid_domains_[i].backend == SimulationDomainBackend::GPU_Vulkan);
            const bool fluid_gpu_compute_available =
                context.compute && context.compute->supportsDispatch();
            const bool gpu_body_force_supported =
                context.compute &&
                context.compute->backendType() == ComputeBackendType::VulkanCompute &&
                context.force_compute_buffer &&
                (context.force_snapshot == nullptr ||
                 context.force_snapshot->empty() ||
                 context.force_compute_buffer->valid());
            const bool force_fields_require_cpu =
                context.force_snapshot && !context.force_snapshot->empty() &&
                !gpu_body_force_supported;
            if (fluid_gpu_requested &&
                fluid_gpu_compute_available &&
                i < grid_domain_compute_buffers_.size()) {
                auto& gpu_buffers = grid_domain_compute_buffers_[i];
                if (ensureGridDomainComputeBuffers(*context.compute, gpu_buffers, state.grid)) {
                    if (fluid_params.grain.enabled) {
                        // The grain coordinator owns its forces: gravity inside
                        // the DEM substeps, and the liquid owner's forces on the
                        // liquid subset only (runMatterGrainStep).
                    } else if (!force_fields_require_cpu) {
                        // Gravity, container motion and shared body-force fields
                        // are evaluated in one GPU particle pass.
                        gpu_integrated_forces = runGpuFluidParticleIntegrateForces(state,
                                                                                   fluid_params,
                                                                                   container_velocity_delta,
                                                                                   dt,
                                                                                   context.time_seconds,
                                                                                   context.force_compute_buffer,
                                                                                   context.compute,
                                                                                   gpu_buffers);
                    } else {
                        // Force fields are CPU-only (noise / wind surface-drag are not
                        // ported to the device). Evaluate them on the CPU, fold in the
                        // container velocity, then UPLOAD the post-force particle state
                        // so GPU P2G / pressure / G2P still run on the device. Only the
                        // cheap force accumulation stays on the CPU — the heavy scatter
                        // and pressure solve no longer fall back just because a force
                        // field (e.g. wind) is active.
                        Fluid::applyExternalForces(state.particles, state.grid, fluid_params,
                                                   context.force_snapshot, context.time_seconds, dt);
                        cpu_forces_applied = true;
                        if (motion_mag > 1e-7f) {
                            const std::size_t pc = state.particles.velocity.size();
                            for (std::size_t pi = 0; pi < pc; ++pi)
                                state.particles.velocity[pi] =
                                    state.particles.velocity[pi] + container_velocity_delta;
                        }
                        // Upload the post-force state for GPU P2G. If this fails the
                        // forces stay correctly applied on the CPU (cpu_forces_applied);
                        // the step falls back to a CPU P2G without re-integrating.
                        gpu_integrated_forces =
                            ensureGpuFluidParticleBuffers(state, context.compute, gpu_buffers);
                    }
                }
            }
            if (!gpu_integrated_forces && !cpu_forces_applied && motion_mag > 1e-7f) {
                const std::size_t particle_count = state.particles.velocity.size();
                for (std::size_t pi = 0; pi < particle_count; ++pi) {
                    state.particles.velocity[pi] = state.particles.velocity[pi] + container_velocity_delta;
                }
            }
            // Remove LAST step's solid-phase overlay before the voxelizer runs,
            // so its cache compares a pure collider mask against a collider
            // signature. Consume-before-service: see clearSubstanceSolidOverlay.
            clearSubstanceSolidOverlay(state.grid);
            prepareKinematicColliderGrid(state.grid);
            // Stamp the active collider set into grid.solid[] every step so
            // Fluid::step's pressure projection + enforceSolidBoundaries see
            // up-to-date boundaries (works for moving/scaled colliders too).
            bool has_kinematic_solids = false;
            {
            RTPERF_FRAME_SCOPE("sim.fluid.voxelize_colliders");
            voxelizeCollidersIntoGrid(state.grid,
                                       colliders_,
                                       collider_bounds_resolver_,
                                       collider_obb_resolver_,
                                       &collider_velocities_,
                                       nullptr,
                                       nullptr,
                                       true);
            has_kinematic_solids = voxelizeKinematicColliders(
                state.grid,
                kinematic_samples,
                KinematicConsumerFluid | KinematicConsumerGranular,
                &kinematic_stamp_counts_);
            appendKinematicStampRecords(
                kinematic_stamp_log_,
                i < grid_domains_.size() ? grid_domains_[i].name : std::string(),
                KinematicConsumerFluid | KinematicConsumerGranular,
                context.frame,
                kinematic_samples,
                kinematic_stamp_counts_);
            }
            // Cached/deforming colliders can enclose particles that were valid
            // in the previous frame. Recover before P2G so neither the pressure
            // grid nor SurfaceSDF density sees hidden liquid inside solid cells.
            const std::size_t solid_recovery_count =
                fluid_params.grain.enabled ? 0 :
                    Fluid::recoverParticlesFromSolidCells(state.particles, state.grid);
            // GPU particle buffers were uploaded before collider voxelization.
            // Recovery changes positions on the host, so refresh the particle
            // buffers instead of changing physics models for one frame. The old
            // CPU fallback made granular piles alternate between MPM and liquid
            // pressure whenever a collider recovered even one parcel, observed
            // as a perpetual boiling layer at the contact surface.
            if (solid_recovery_count > 0u && fluid_gpu_requested &&
                fluid_gpu_compute_available && context.compute &&
                i < grid_domain_compute_buffers_.size()) {
                gpu_integrated_forces = ensureGpuFluidParticleBuffers(
                    state, context.compute, grid_domain_compute_buffers_[i]);
            }

            // ── Thermal liquid: cooling, then freeze/melt ────────────────────
            // HERE, between the collider stamp and the solid-phase overlay:
            // cooling must see colliders as cold surfaces but not the overlay
            // (that is the wax itself), and the overlay must see this frame's
            // frozen set. Both calls are no-ops when the chain is off, except
            // that updateThermalFreeze clears any frozen flag left behind.
            state.thermal_stats = Fluid::ThermalLiquidStats{};
            if (i < grid_domains_.size()) {
                RTPERF_FRAME_SCOPE("sim.fluid.thermal_cool_freeze");
                const float ambient_kelvin =
                    fluidDomainAmbientKelvin(grid_domains_[i], world_thermal_);
                Fluid::coolThermalLiquid(state.particles, state.grid, fluid_params,
                                         ambient_kelvin, dt, state.thermal_stats);
                const uint64_t phase_event_id =
                    (matter_exchange_step_ << 32u) |
                    (static_cast<uint64_t>(i & 0xffffu) << 16u) |
                    uint64_t{0xF000u};
                Fluid::updateThermalFreezeAndRecord(
                    grid_domains_[i], state, fluid_params, phase_event_id,
                    matter_exchange_ledger_);
            }
            const bool frozen_present = state.thermal_stats.frozen_particles > 0;
            // The freeze pass pinned velocities on the HOST. The force pass
            // already uploaded post-force velocities to the device, and GPU P2G
            // reuses that copy — so a frozen parcel would still splat gravity
            // into the grid. Refresh, exactly as the solid-recovery path above.
            if (frozen_present && gpu_integrated_forces && context.compute &&
                i < grid_domain_compute_buffers_.size()) {
                gpu_integrated_forces = ensureGpuFluidParticleBuffers(
                    state, context.compute, grid_domain_compute_buffers_[i]);
            }

            // ── Solid-phase substances → grid.solid overlay ──────────────────
            // Tags first: this list is what tells the particle stages "I AM the
            // solid" apart from "I am inside one", and the same list decides
            // whether the overlay is built at all.
            static std::vector<uint32_t> s_solid_tags;
            s_solid_tags.clear();
            // ★ The master switch is read HERE, at the producer, and it empties
            // the tag list rather than skipping the stamp further down. That way
            // the particle stages see "no solid substance" too: a half-off state
            // where parcels are exempt from advection but nothing blocks them
            // would be a third behaviour nobody asked for.
            if (i < grid_domains_.size() && grid_domains_[i].fluid_solid_phase_enabled) {
                for (const auto& b : grid_domains_[i].fluid_substance_materials) {
                    if (b.substance.empty()) continue;
                    if (b.phase != RayTrophiSim::Fluid::SubstancePhase::Solid) continue;
                    if (s_solid_tags.size() >= RayTrophiSim::Fluid::kMaxFluidSubstanceMaterials)
                        break;
                    const auto tag = RayTrophiSim::Fluid::substanceTag(b.substance);
                    bool mobile = false;
                    if (mixed_models) {
                        for (std::size_t particle = 0; particle < state.particles.size(); ++particle) {
                            if (particle < state.particles.substance_tag.size() &&
                                state.particles.substance_tag[particle] == tag) {
                                const auto model = particle < state.particles.constitutive_model.size()
                                    ? static_cast<Fluid::MatterConstitutiveModel>(
                                        state.particles.constitutive_model[particle])
                                    : Fluid::MatterConstitutiveModel::Auto;
                                mobile |= model == Fluid::MatterConstitutiveModel::Fluid ||
                                    model == Fluid::MatterConstitutiveModel::Granular ||
                                    model == Fluid::MatterConstitutiveModel::Auto;
                            }
                        }
                    }
                    if (!mobile) {
                        s_solid_tags.push_back(tag);
                    }
                }
            }
            std::size_t solid_phase_cell_count = 0;
            std::size_t solid_phase_particle_count = 0;
            // Anything that makes a parcel "the solid": a solid substance tag, or
            // a frozen thermal-liquid parcel. Every consumer below that used to
            // ask "are there solid tags?" really asks this.
            const bool solid_parcels_present = !s_solid_tags.empty() || frozen_present;
            if (solid_parcels_present) {
                RTPERF_FRAME_SCOPE("sim.fluid.solid_overlay");
                // ★ THE FILL THRESHOLD IS TIED TO THE SEED DENSITY, because that
                // is the only number in the scene that says what a FULL cell
                // means. A quarter of it: high enough that one stray parcel does
                // not dam a channel with a full h³ of wall, low enough that a
                // chunk two parcels thick still blocks. A fixed constant would
                // mean something different at every particles_per_cell setting,
                // and the difference would show up as "solid works in this scene
                // and not that one".
                const float fill_fraction = (i < grid_domains_.size())
                    ? std::max(0.01f, grid_domains_[i].fluid_solid_phase_fill)
                    : 0.25f;
                const float fill_threshold = std::max(1.0f,
                    fill_fraction * static_cast<float>(std::max(1, fluid_params.particles_per_cell)));
                static std::vector<uint32_t> s_solid_cells;
                static std::vector<Vec3>     s_solid_cell_vel;
                if (RayTrophiSim::Fluid::buildSubstanceSolidCells(
                        state.particles, state.grid,
                        s_solid_tags.data(), s_solid_tags.size(),
                        fill_threshold,
                        s_solid_cells, s_solid_cell_vel,
                        /*include_frozen=*/frozen_present)) {
                    state.grid.solid_cells_collider_count =
                        state.grid.solid_cells.size();
                    applySubstanceSolidOverlay(state.grid, s_solid_cells, s_solid_cell_vel,
                                               fluid_params.max_velocity);
                    solid_phase_cell_count = state.grid.substance_solid_cells.size();
                }
                // Counted even when no cell filled: parcels-present-with-zero-
                // cells is the reading that says "raise the resolution", and it
                // is invisible if only the cells are reported.
                for (std::size_t pi = 0; pi < state.particles.size() &&
                                         pi < state.particles.substance_tag.size(); ++pi) {
                    const uint32_t tag = state.particles.substance_tag[pi];
                    if (tag == RayTrophiSim::Fluid::kSubstanceUntagged) continue;
                    for (uint32_t st : s_solid_tags) {
                        if (st != tag) continue;
                        ++solid_phase_particle_count;
                        break;
                    }
                }
                // ★★ The overlay moves every step but does NOT invalidate the
                // collider face-weight cache any more: blockSubstanceSolidFace
                // Weights logs what it closes and clearSubstanceSolidOverlay
                // restores it, so the cache always sees pure collider weights.
                // The old invalidation forced the full mesh-collider
                // super-sample every frame (see FluidGrid::overlay_weight_restore_*).
            }
            // Variational solid coupling: fractional MAC-face open weights for
            // sub-grid-accurate boundaries + moving-collider splash. Cheap (only
            // the collider neighbourhood is super-sampled); skipped when the flag
            // is off so the binary path stays available as a fallback.
            if (fluid_params.variational_solids) {
                RTPERF_FRAME_SCOPE("sim.fluid.solid_face_weights");
                computeSolidFaceWeights(state.grid,
                                        colliders_,
                                        collider_bounds_resolver_,
                                        collider_obb_resolver_,
                                        true,
                                        has_kinematic_solids);
                // The weights above describe COLLIDERS only; close the overlay's
                // own faces or a chunk is invisible to the variational pressure
                // solve while remaining visible to the binary one.
                blockSubstanceSolidFaceWeights(state.grid);
            } else {
                // Weights aren't maintained while variational is off; mark them
                // stale so the next on-frame does a full open-init (the colliders
                // may have moved far during the gap, beyond the incremental reset).
                state.grid.collider_weights_init = false;
            }
            auto step_params = fluid_params;
            step_params.max_particles = (i < grid_domains_.size()) ? grid_domains_[i].fluid_max_particles : 100000;
            // Hand the solid tags to the solver: the mask is already stamped,
            // but the PARTICLE stages need to know which parcels are the solid
            // so they are neither advected by the flow nor ejected from their
            // own cells (see APICSolverParams::solid_substance_tags).
            step_params.solid_substance_tags =
                s_solid_tags.empty() ? nullptr : &s_solid_tags;

            // ── Per-substance viscosity field ────────────────────────────────
            // Built HERE, once per domain per step, and pointed at by the params
            // every downstream call copies — the CPU solve, the device solve and
            // both split-step halves must diffuse with the SAME field or the two
            // backends produce different liquid from the same scene.
            //
            // ★ Function-static and reused across domains: it is consumed
            // entirely within this iteration (Fluid::step and
            // runGpuFluidViscosity are both called below, before the next
            // domain), so one buffer is enough and a per-domain vector would be
            // an allocation per domain per frame.
            //
            // ★★ The gather takes NO representation/exclusion filter. Splat is
            // a RENDER routing decision; a substance drawn as spheres still has
            // mass and still resists shear. See buildSubstanceViscosityField.
            static std::vector<float> s_substance_viscosity;
            step_params.substance_viscosity = nullptr;
            if (i < grid_domains_.size() &&
                !grid_domains_[i].fluid_substance_materials.empty()) {
                RayTrophiSim::Fluid::SubstanceViscosityEntry visc_entries[
                    RayTrophiSim::Fluid::kMaxFluidSubstanceMaterials];
                std::size_t visc_entry_count = 0;
                for (const auto& b : grid_domains_[i].fluid_substance_materials) {
                    if (visc_entry_count >= RayTrophiSim::Fluid::kMaxFluidSubstanceMaterials)
                        break;
                    if (b.substance.empty()) continue;
                    visc_entries[visc_entry_count].tag =
                        RayTrophiSim::Fluid::substanceTag(b.substance);
                    visc_entries[visc_entry_count].kinematic_viscosity =
                        b.kinematic_viscosity;
                    ++visc_entry_count;
                }
                if (visc_entry_count > 0 &&
                    RayTrophiSim::Fluid::buildSubstanceViscosityField(
                        state.particles, state.grid,
                        visc_entries, visc_entry_count,
                        fluid_params.kinematic_viscosity,
                        s_substance_viscosity)) {
                    step_params.substance_viscosity = &s_substance_viscosity;
                }
            }
            // ── Thermal viscosity ν(T) ───────────────────────────────────────
            // Layered ON TOP of the substance field (its value is the hot end of
            // each cell's ramp), and handed to the solver through the same
            // pointer, so the CPU solve, the device solve and both split-step
            // halves all see one field. A separate scratch: the substance field
            // is this call's input and must not be overwritten while read.
            static std::vector<float> s_thermal_viscosity;
            {
                RTPERF_FRAME_SCOPE("sim.fluid.thermal_viscosity_field");
                if (Fluid::buildThermalViscosityField(state.particles, state.grid, fluid_params,
                                                      step_params.substance_viscosity,
                                                      s_thermal_viscosity,
                                                      &state.thermal_stats)) {
                    step_params.substance_viscosity = &s_thermal_viscosity;
                }
            }
            // Forces were evaluated on the CPU (force-field branch); make every
            // downstream step() skip its force stage so they are not applied twice.
            // This holds whether or not the GPU upload/P2G succeeded.
            if (cpu_forces_applied || (mixed_models && gpu_integrated_forces)) {
                step_params.external_forces_preintegrated = true;
            }

            // One-shot GPU MGPCG correctness self-test. Runs on a synthetic
            // isolated grid the first time any fluid domain is stepped — does
            // not perturb this domain's state. Always logs one line (the result
            // or a "no GPU dispatch" warning) so the outcome is visible without
            // needing an env var. Env var RAYTROPHI_MGPCG_SELFTEST can force it
            // too, but it is no longer required.
            // Auto-runs once per session on ANY GPU dispatch backend (CUDA and
            // Vulkan) so the PASS/FAIL line lands in the Console without launch
            // parameters — the user cannot pass env vars, and having the line
            // on both backends makes the perf/parity comparison symmetric.
            // Once the Vulkan port is long-term stable this can be demoted to
            // env-var-only again.
            static bool s_mgpcg_selftest_done = false;
            const bool gpu_selftest_wanted =
                context.compute &&
                context.compute->supportsDispatch();
            if (!s_mgpcg_selftest_done &&
                (gpu_selftest_wanted || std::getenv("RAYTROPHI_MGPCG_SELFTEST") != nullptr)) {
                s_mgpcg_selftest_done = true;
                SCENE_LOG_INFO(std::string("[MGPCG SelfTest] Running GPU MGPCG correctness self-test..."));
                validateGpuFluidMGPCG(context.compute);
            }

            // ── GPU P2G ──────────────────────────────────────────────────────
            // Elastic MPM stability depends on wave speed, not only particle
            // advection velocity. Subcycle the complete transfer/grid/gather/
            // advect path so the authored Young modulus is actually used.
            const float granular_frame_dt = dt;
            // The subcycle must satisfy the strain-rate CFL as well as the wave
            // CFL. Measuring ||C|| here (from the affine the last G2P wrote) is
            // what closes the soft-material hole: without it any E below
            // rho*(0.35h/dt)^2 silently returns a single full-frame substep.
            // ★ ONCE PER FRAME, before the subcycle and before the granular
            // state is uploaded. Deriving it per substep would make the melt
            // rate depend on the substep count.
            // Conduction FIRST, then the softening derivation reads the result
            // in the same frame. The other order costs a frame of latency per
            // link in the chain, which at 24 fps is visible as the melt front
            // lagging the flame.
            Fluid::diffuseParticleTemperature(
                state.particles, state.grid,
                fluid_params.granular_thermal_conductivity, granular_frame_dt);
            if (fluid_params.granular_enabled) {
                state.particles.ensureGranularStateSize();
                Fluid::Granular::SofteningParams softening_params;
                softening_params.softening_temperature =
                    fluid_params.granular_softening_temperature;
                softening_params.softening_range =
                    fluid_params.granular_softening_range;
                softening_params.tack_peak = fluid_params.granular_tack_peak;
                softening_params.residual_strength =
                    fluid_params.granular_residual_strength;
                Fluid::Granular::updateSoftening(state.particles, softening_params);
            }
            const auto granular_load = fluid_params.granular_enabled
                ? Fluid::Granular::measureLoad(state.particles)
                : Fluid::Granular::LoadMeasurement{};
            const float granular_strain_rate = granular_load.strain_rate;
            const float granular_overburden = granular_load.overburden_pressure;
            const auto granular_frame_elastic = Fluid::Granular::elasticStepInfo(
                fluid_params.granular_young_modulus,
                state.grid.voxel_size, granular_frame_dt,
                granular_strain_rate, granular_overburden,
                granular_load.softening_min);
            const int granular_solver_substeps = fluid_params.granular_enabled
                ? Fluid::Granular::solverSubsteps(granular_frame_elastic)
                : 1;
            // ★ A material too soft for its own overburden is not a tuning
            // choice, it is outside the corotational model's domain: holding
            // pressure p needs volumetric strain p/K, and at E far below the
            // load that strain exceeds one. It stays stable for a while, then
            // discharges. Say so once per parameter change rather than letting
            // it read as a mysterious explosion.
            if (fluid_params.granular_enabled && granular_frame_elastic.below_load) {
                static float s_last_warned_young = -1.0f;
                if (s_last_warned_young != fluid_params.granular_young_modulus) {
                    s_last_warned_young = fluid_params.granular_young_modulus;
                    SCENE_LOG_WARN(
                        "[Granular] Young modulus " +
                        std::to_string(granular_frame_elastic.requested_young_modulus) +
                        " Pa is below the " +
                        std::to_string(granular_frame_elastic.young_modulus_for_load) +
                        " Pa its own overburden needs (" +
                        std::to_string(granular_load.column_height) + " m of material = " +
                        std::to_string(granular_frame_elastic.overburden_pressure) +
                        " Pa). The pile cannot hold itself in the small-strain regime: "
                        "expect compaction, a det(F) reset, and a velocity burst. This is "
                        "a material limit, not a timestep one - substeps will not fix it.");
                }
            }
            if (fluid_params.granular_enabled) {
                fluid_params.velocity_damping = Fluid::Granular::timeScaledSubstepDamping(
                    fluid_params.velocity_damping, granular_frame_dt, granular_solver_substeps);
                fluid_params.affine_damping = Fluid::Granular::timeScaledSubstepDamping(
                    fluid_params.affine_damping, granular_frame_dt, granular_solver_substeps);
                step_params.velocity_damping = fluid_params.velocity_damping;
                step_params.affine_damping = fluid_params.affine_damping;
            }
            float granular_p2g_ms_sum = 0.0f;
            float granular_pressure_ms_sum = 0.0f;
            float granular_viscosity_ms_sum = 0.0f;
            float granular_g2p_ms_sum = 0.0f;
            float granular_advect_ms_sum = 0.0f;
            int granular_advect_substeps_sum = 0;
            bool granular_substep_failed = false;
            bool granular_state_resident = false;
            // Final substep result is also consumed by the frame-level GPU
            // status/cache section after the elastic subcycle closes.
            bool g2p_on_gpu = false;
            bool particle_tail_on_gpu = false;
            const bool granular_state_can_stay_resident =
                fluid_params.granular_enabled &&
                fluid_params.boundary != Fluid::APICSolverParams::BoundaryMode::Open;

            // ── Granular grid velocity stays on the device between substeps ──
            // ★★★ MEASURED 2026-09-25: a 114^3 granular domain holding 1000
            // particles moved 1.33 GB up and 582 MB down PER FRAME. Every elastic
            // substep (up to 32, and the count rises exactly when a collider
            // strains the pile - "some frames stall") downloaded the P2G field,
            // ran the solid-face clamp on the host, and uploaded it again for G2P,
            // plus the solid velocity deinterleaved and re-uploaded although the
            // colliders cannot change inside a frame. All of it scales with the
            // GRID, none with the particles, and all of it is single-threaded
            // memcpy - the "one core busy" the user saw.
            //
            // With the clamp on the device (sim_fluid_zero_solid_faces) nothing on
            // the host reads grid.vel_* between P2G and G2P for a granular step, so
            // the field stays DEVICE-ONLY and comes home once, after the loop.
            //
            // ★ Gated to exactly the configuration where that claim holds: every
            // host stage that could read grid.vel_* mid-chain is excluded here
            // (viscosity uploads host vel; a solid substance forces the host
            // advect tail), and the ones that can still happen by failure
            // (host tail, CPU fallback) call bring_grid_velocity_home first.
            // Not const: a missing device clamp turns it off for the rest of the frame.
            bool granular_grid_can_stay_on_device =
                fluid_params.granular_enabled &&
                gpu_integrated_forces &&
                fluid_params.free_surface &&
                fluid_params.boundary != Fluid::APICSolverParams::BoundaryMode::Periodic &&
                fluid_params.kinematic_viscosity <= 0.0f &&
                step_params.substance_viscosity == nullptr &&
                !solid_parcels_present &&
                context.compute &&
                context.compute->backendType() == ComputeBackendType::VulkanCompute &&
                context.compute->supportsDispatch() &&
                i < grid_domain_compute_buffers_.size();
            GranularParticleResidency particle_residency(
                state, context.compute,
                i < grid_domain_compute_buffers_.size()
                    ? &grid_domain_compute_buffers_[i] : nullptr,
                granular_grid_can_stay_on_device &&
                    fluid_params.boundary == Fluid::APICSolverParams::BoundaryMode::Closed);
            // True while grid.vel_* on the host is stale.
            bool grid_velocity_device_only = false;
            // Solid velocity is frame-constant (voxelized before this loop), so
            // it is deinterleaved and uploaded once, not once per substep.
            bool granular_solid_velocity_uploaded = false;
            auto bring_grid_velocity_home = [&]() -> bool {
                if (!grid_velocity_device_only) return true;
                auto& gpu_buffers = grid_domain_compute_buffers_[i];
                context.compute->beginTransferBatch();
                bool ok =
                    context.compute->downloadBuffer(gpu_buffers.vel_x, state.grid.vel_x.data(),
                                                    state.grid.vel_x.size() * sizeof(float)) &&
                    context.compute->downloadBuffer(gpu_buffers.vel_y, state.grid.vel_y.data(),
                                                    state.grid.vel_y.size() * sizeof(float)) &&
                    context.compute->downloadBuffer(gpu_buffers.vel_z, state.grid.vel_z.data(),
                                                    state.grid.vel_z.size() * sizeof(float));
                ok = context.compute->endTransferBatch() && ok;
                if (ok) grid_velocity_device_only = false;
                return ok;
            };

            for (int granular_substep = 0;
                 granular_substep < granular_solver_substeps;
                 ++granular_substep) {
                if (mixed_models) {
                    std::string mixed_error;
                    bool ok = false;
                    if (step_params.grain.enabled && i < grid_domain_compute_buffers_.size()) {
                        ok = runMatterGrainStep(state, step_params, mixed_legacy_granular,
                            granular_frame_dt, context.time_seconds, context.compute,
                            grid_domain_compute_buffers_[i],
                            [&](SimulationGridDomainComputeBuffers& buffers) {
                                return context.compute && ensureGridDomainComputeBuffers(
                                    *context.compute, buffers, state.grid);
                            }, matter_exchange_ledger_, colliders_,
                            grain_collider_mesh_resolver_, collider_velocities_, kinematic_samples,
                            context.force_snapshot, context.force_compute_buffer,
                            motion_mag > 1e-7f, mixed_error);
                    } else if (step_params.grain.enabled) {
                        mixed_error = "grain Vulkan compute buffers unavailable";
                    } else if (!fluid_gpu_requested) {
                        auto reference_params = step_params;
                        reference_params.granular_enabled = mixed_legacy_granular;
                        ok = Fluid::stepMixedMatter(state.particles, state.grid, reference_params,
                            granular_frame_dt, cpu_forces_applied ? nullptr : context.force_snapshot,
                            context.time_seconds, state.fluid_stats, mixed_error);
                    } else if (i < grid_domain_compute_buffers_.size()) {
                        ok = runMatterGpuStep(state, step_params, mixed_legacy_granular,
                            granular_frame_dt, context.compute, grid_domain_compute_buffers_[i],
                            [&](SimulationGridDomainComputeBuffers& buffers) {
                                return context.compute && ensureGridDomainComputeBuffers(
                                    *context.compute, buffers, state.grid);
                            }, matter_exchange_ledger_, mixed_error);
                    }
                    if (!ok) {
                        if (state.particles.particle_id == mixed_ids_before) {
                            state.particles.velocity = mixed_velocity_before;
                        } else {
                            std::unordered_map<uint64_t, std::size_t> original;
                            for (std::size_t particle = 0; particle < mixed_ids_before.size(); ++particle) {
                                original.emplace(mixed_ids_before[particle], particle);
                            }
                            for (std::size_t particle = 0; particle < state.particles.size(); ++particle) {
                                const auto found = original.find(state.particles.particle_id[particle]);
                                if (found != original.end() && found->second < mixed_velocity_before.size()) {
                                    state.particles.velocity[particle] = mixed_velocity_before[found->second];
                                }
                            }
                        }
                        state.fluid_stats = Fluid::APICSolverStats{};
                        if (step_params.pore_exchange.enabled) {
                            state.fluid_stats.pore_exchange.held = true;
                            state.fluid_stats.pore_exchange.status = mixed_error;
                        }
                        state.fluid_stats.gpu_fallback = true;
                        state.fluid_stats.mixed_step_held = true;
                        state.fluid_stats.gpu_status = "Mixed step held: " + mixed_error;
                        SCENE_LOG_WARN(state.fluid_stats.gpu_status);
                    }
                    g2p_on_gpu = ok && fluid_gpu_requested && !step_params.grain.enabled;
                    particle_tail_on_gpu = ok && fluid_gpu_requested;
                    granular_substep_failed = !ok;
                    granular_p2g_ms_sum = state.fluid_stats.p2g_ms;
                    granular_pressure_ms_sum = state.fluid_stats.pressure_ms;
                    granular_viscosity_ms_sum = state.fluid_stats.viscosity_ms;
                    granular_g2p_ms_sum = state.fluid_stats.g2p_ms;
                    granular_advect_ms_sum = state.fluid_stats.advect_ms;
                    break;
                }
                if (!particle_residency.prepare()) {
                    granular_substep_failed = true;
                    break;
                }
            const std::size_t granular_particle_count_before = state.particles.size();
            const float dt = granular_frame_dt /
                static_cast<float>(granular_solver_substeps);
            step_params.stop_before_pressure = false;
            step_params.pressure_g2p_precomputed = false;
            step_params.particle_tail_precomputed = false;
            step_params.viscosity_precomputed = false;
            // Every substep starts with a P2G that rewrites the whole field, on
            // the device or (if that fails) on the host. Neither flag may carry
            // over: a stale p2g_precomputed would make the host skip its own P2G
            // and step on the previous substep's grid, and a stale device-only
            // mark would later overwrite a host P2G with that old device field.
            step_params.p2g_precomputed = false;
            grid_velocity_device_only = false;

            float gpu_p2g_ms = 0.0f;
            if (gpu_integrated_forces) {
                step_params.external_forces_preintegrated = true;
                const auto gpu_p2g_begin = SimulationClock::now();
                if (context.compute && i < grid_domain_compute_buffers_.size()) {
                    auto& gpu_buffers = grid_domain_compute_buffers_[i];
                    if (ensureGridDomainComputeBuffers(*context.compute, gpu_buffers, state.grid)) {
                        step_params.p2g_precomputed = runGpuFluidP2G(state,
                                                                     context.compute,
                                                                     gpu_buffers,
                                                                     step_params,
                                                                     dt,
                                                                     !granular_state_resident,
                                                                     granular_substep == 0 ||
                                                                         particle_residency.hostStale(),
                                                                     !granular_grid_can_stay_on_device);
                        if (step_params.p2g_precomputed &&
                            granular_state_can_stay_resident)
                            granular_state_resident = true;
                        // A fresh P2G overwrote the whole field on the device; the
                        // host copy (from an earlier substep, or never) is stale.
                        grid_velocity_device_only =
                            step_params.p2g_precomputed && granular_grid_can_stay_on_device;
                    }
                }
                if (step_params.p2g_precomputed)
                    gpu_p2g_ms = elapsedMilliseconds(gpu_p2g_begin, SimulationClock::now());
            }

            // ── GPU pressure (MGPCG) + GPU G2P ───────────────────────────────
            // The pressure projection runs on the GPU via a Jacobi-preconditioned
            // CG (runGpuFluidMGPCGPressure), validated against the CPU PCG+MIC(0)
            // to float precision. Falls back to the full CPU step if any GPU
            // stage fails.
            //
            // Pattern (all-GPU pressure+G2P when compute backend dispatches):
            //   Call 1 (stop_before_pressure=true): forces(opt) + P2G(opt) +
            //     FLIP snapshot + boundary → returns; grid.vel = post-boundary,
            //     pre-viscosity field.
            //   GPU: upload the published FLIP snapshot to scratch → viscous
            //     diffusion → MGPCG pressure projection → GPU G2P → download
            //     particle vel/affine.
            //   Call 2 (pressure_g2p_precomputed=true): air_drag + damping +
            //     advect + reseed only.
            //   Any GPU step failing falls back to the full CPU step below,
            //   which re-runs P2G and therefore redoes viscosity itself — the
            //   device result is discarded rather than applied twice.
            if (!step_params.p2g_precomputed && !particle_residency.recover()) {
                granular_substep_failed = true;
                break;
            }

            float gpu_g2p_ms = 0.0f;
            float gpu_advect_ms = 0.0f;
            int gpu_advect_substeps = 0;
            float gpu_pressure_ms = 0.0f;
            float gpu_viscosity_ms = 0.0f;
            int   gpu_viscosity_sweeps = 0;
            g2p_on_gpu = false;
            particle_tail_on_gpu = false;
            Fluid::APICSolverStats gpu_mgpcg_stats;

            const bool try_gpu_g2p =
                fluid_gpu_requested &&
                fluid_params.free_surface &&
                // Periodic boundaries need the wrap-coupled CPU PCG (the GPU MGPCG
                // still treats out-of-grid as solid, i.e. behaves as Closed); fall
                // back to CPU so Periodic is correct on every device.
                fluid_params.boundary != Fluid::APICSolverParams::BoundaryMode::Periodic &&
                context.compute &&
                context.compute->supportsDispatch() &&
                i < grid_domain_compute_buffers_.size();

            if (try_gpu_g2p) {
                auto& gpu_buffers = grid_domain_compute_buffers_[i];
                if (ensureGridDomainComputeBuffers(*context.compute, gpu_buffers, state.grid)) {
                    // Call 1: P2G + FLIP snapshot + solid boundaries, STOP before
                    // pressure. Viscosity is claimed by the device below, so the
                    // host must not also run it.
                    auto call1_params = step_params;
                    call1_params.stop_before_pressure = true;
                    call1_params.viscosity_precomputed = true;
                    auto run_call1 = [&]() {
                        Fluid::step(state.particles, state.grid, call1_params, dt,
                                    gpu_integrated_forces ? nullptr : context.force_snapshot,
                                    context.time_seconds, nullptr);
                    };
                    // ★ Skipped while the field is device-only. For a granular step
                    // with GPU forces + GPU P2G, Call 1's ONLY effect is the host
                    // solid-face clamp (forces pre-integrated, P2G precomputed, FLIP
                    // off for granular, viscosity claimed below) - and that clamp
                    // runs on the device after the mask upload instead.
                    if (!grid_velocity_device_only) run_call1();

                    // ★ The FLIP snapshot is the POST-P2G field that step() just
                    // published — NOT state.grid.vel_*, which by now carries the
                    // solid boundaries and is about to carry the viscous solve.
                    // Uploading grid.vel here is what used to cancel viscosity out
                    // of the FLIP delta entirely (see the note in Fluid::step).
                    const bool has_flip =
                        !fluid_params.granular_enabled &&
                        gpu_buffers.scratch_vel_x.valid() &&
                        gpu_buffers.scratch_vel_y.valid() &&
                        gpu_buffers.scratch_vel_z.valid() &&
                        fluid_params.flip_blend > 0.0f &&
                        Fluid::hasLastFlipPreSnapshot();
                    bool upload_ok = true;
                    if (has_flip) {
                        const std::array<ComputeBufferHandle, 3> flip_sources = {
                            gpu_buffers.vel_x, gpu_buffers.vel_y, gpu_buffers.vel_z
                        };
                        const std::array<ComputeBufferHandle, 3> flip_targets = {
                            gpu_buffers.scratch_vel_x, gpu_buffers.scratch_vel_y,
                            gpu_buffers.scratch_vel_z
                        };
                        const std::array<std::size_t, 3> face_counts = {
                            state.grid.vel_x.size(), state.grid.vel_y.size(),
                            state.grid.vel_z.size()
                        };
                        const bool copied_on_gpu =
                            step_params.p2g_precomputed &&
                            FluidGpuFlipSnapshot::copyFaceFields(
                                *context.compute, flip_sources, flip_targets,
                                face_counts);
                        if (!copied_on_gpu) {
                            context.compute->beginTransferBatch();
                            upload_ok =
                                context.compute->uploadBuffer(
                                    gpu_buffers.scratch_vel_x,
                                    Fluid::getLastFlipPreSnapshotX(),
                                    face_counts[0] * sizeof(float)) &&
                                context.compute->uploadBuffer(
                                    gpu_buffers.scratch_vel_y,
                                    Fluid::getLastFlipPreSnapshotY(),
                                    face_counts[1] * sizeof(float)) &&
                                context.compute->uploadBuffer(
                                    gpu_buffers.scratch_vel_z,
                                    Fluid::getLastFlipPreSnapshotZ(),
                                    face_counts[2] * sizeof(float));
                            upload_ok = context.compute->endTransferBatch() && upload_ok;
                        }
                    }

                    // Fluid mask (cell occupancy) — the viscous stencil and the
                    // pressure matrix both classify faces with it.
                    static std::vector<float> s_fluid_mask_gpu; // function-static scratch
                    if (upload_ok) {
                        upload_ok = prepareGpuFluidMask(
                            state, *context.compute, gpu_buffers, fluid_params,
                            particle_residency, s_fluid_mask_gpu,
                            granular_solid_velocity_uploaded);
                    }

                    // Device solid-face clamp for the resident chain. Reads the
                    // fluid_mask uploaded just above. If the kernel is unavailable
                    // (missing .spv) bring the field home and let Call 1 clamp it
                    // on the host - the old path, one substep at a time.
                    if (upload_ok && grid_velocity_device_only &&
                        !runGpuFluidZeroSolidFaces(state.grid, context.compute, gpu_buffers)) {
                        static bool s_warned_zero_solid_faces = false;
                        if (!s_warned_zero_solid_faces) {
                            s_warned_zero_solid_faces = true;
                            SCENE_LOG_WARN(
                                "[SimCompute] sim_fluid_zero_solid_faces unavailable; granular "
                                "grid velocity round-trips through the host every substep. "
                                "Recompile shaders (compile_shaders.bat).");
                        }
                        granular_grid_can_stay_on_device = false;
                        upload_ok = bring_grid_velocity_home();
                        if (upload_ok) run_call1();
                    }

                    // Viscous diffusion. A requested viscosity the device cannot
                    // deliver is a FALLBACK, not something to skip: silently
                    // shipping an inviscid step for a honey preset is the exact
                    // class of failure that looks plausible and gets tuned around.
                    bool viscosity_ok = true;
                    // ★ step_params, NOT fluid_params: the per-substance field
                    // hangs off step_params, and passing the domain params here
                    // would run the DEVICE solve uniform while the host copy ran
                    // it variable — two backends, two liquids, no error.
                    // The field is also reason enough to run the stage: an
                    // inviscid domain pouring honey must still be honey.
                    if (upload_ok && (fluid_params.kinematic_viscosity > 0.0f ||
                                      step_params.substance_viscosity != nullptr)) {
                        const auto visc_begin = SimulationClock::now();
                        viscosity_ok = runGpuFluidViscosity(state, step_params, dt,
                                                            context.compute, gpu_buffers,
                                                            s_fluid_mask_gpu,
                                                            &gpu_viscosity_sweeps);
                        if (viscosity_ok) {
                            enforceGridSolidFaceBoundaries(state.grid);
                            gpu_viscosity_ms = elapsedMilliseconds(visc_begin, SimulationClock::now());
                        }
                    }

                    // GPU pressure projection (MGPCG). On success grid.vel holds
                    // the projected field and the GPU vel buffers match it.
                    bool pressure_on_gpu = false;
                    if (upload_ok && viscosity_ok && fluid_params.granular_enabled) {
                        // Granular MPM is compressible. Running the liquid
                        // incompressibility projection on top of its elastic
                        // volumetric stress double-enforces volume and causes an
                        // immediate pressure explosion.
                        pressure_on_gpu = true;
                    } else if (upload_ok && viscosity_ok) {
                        const auto gpu_pressure_begin = SimulationClock::now();
                        pressure_on_gpu = runGpuFluidMGPCGPressure(state, fluid_params, dt,
                                                                   context.compute, gpu_buffers,
                                                                   s_fluid_mask_gpu,
                                                                   &gpu_mgpcg_stats);
                        if (pressure_on_gpu) {
                            // CPU PCG zeros solid-adjacent faces during the
                            // projection update. The GPU gradient kernel uses
                            // the fluid mask matrix, then we mirror that final
                            // no-flow clamp here before G2P samples velocities.
                            enforceGridSolidFaceBoundaries(state.grid);
                            gpu_pressure_ms = elapsedMilliseconds(gpu_pressure_begin, SimulationClock::now());
                        }
                    }

                    if (pressure_on_gpu) {
                        const auto gpu_g2p_begin = SimulationClock::now();
                        // ── Solid-phase parcels are not the device's to move ──
                        // ★★★ THE DEVICE G2P KERNEL HAS NO PHASE CONCEPT, so it
                        // will hand every parcel the grid velocity — including
                        // the ones that ARE the solid. Their own cells carry
                        // solid_vel, so the value it writes is close enough to
                        // look right at rest and wrong only under flow: the
                        // chunk would slowly get carried away by the liquid it
                        // is supposed to obstruct. Snapshot before, restore
                        // after; indices are stable across P2G/pressure/G2P
                        // because nothing adds or removes parcels in between.
                        //
                        // ★ Restoring rather than gating in the shader keeps the
                        // two backends answering identically without a fifth
                        // ABI mirror. If the phase ever needs to be ON the
                        // device (it does not yet — nothing there reads it), it
                        // becomes a per-particle flag, not a tag list.
                        static std::vector<uint32_t> s_solid_idx;
                        static std::vector<Vec3>     s_solid_vel_keep;
                        static std::vector<Fluid::AffineC> s_solid_affine_keep;
                        s_solid_idx.clear();
                        s_solid_vel_keep.clear();
                        s_solid_affine_keep.clear();
                        if (solid_parcels_present) {
                            for (std::size_t pi = 0; pi < state.particles.size(); ++pi) {
                                // Frozen wax is restored too: the device G2P
                                // would hand it the grid velocity and unpin it.
                                bool solid = frozen_present &&
                                             Fluid::isFrozenParticle(state.particles, pi);
                                if (!solid && pi < state.particles.substance_tag.size()) {
                                    const uint32_t tag = state.particles.substance_tag[pi];
                                    if (tag != RayTrophiSim::Fluid::kSubstanceUntagged) {
                                        for (uint32_t st : s_solid_tags) { if (st == tag) { solid = true; break; } }
                                    }
                                }
                                if (!solid) continue;
                                s_solid_idx.push_back(static_cast<uint32_t>(pi));
                                s_solid_vel_keep.push_back(state.particles.velocity[pi]);
                                if (pi < state.particles.affine.size())
                                    s_solid_affine_keep.push_back(state.particles.affine[pi]);
                                else
                                    s_solid_affine_keep.emplace_back();
                            }
                        }
                        const bool combine_g2p_tail_readback =
                            !solid_parcels_present &&
                            !fluid_params.granular_enabled &&
                            context.compute->backendType() ==
                                ComputeBackendType::VulkanCompute;
                        g2p_on_gpu = runGpuFluidG2P(
                            state, fluid_params, dt,
                            context.compute, gpu_buffers, has_flip,
                            !granular_state_can_stay_resident ||
                                granular_substep + 1 == granular_solver_substeps,
                            step_params.p2g_precomputed,
                            (!fluid_params.granular_enabled &&
                                !state.grid.hasAnySolid()) ||
                                grid_velocity_device_only,
                            combine_g2p_tail_readback ||
                                particle_residency.defer(granular_substep,
                                                         granular_solver_substeps));
                        particle_residency.afterG2P(g2p_on_gpu, granular_substep,
                                                   granular_solver_substeps);
                        if (g2p_on_gpu) {
                            gpu_g2p_ms = elapsedMilliseconds(gpu_g2p_begin, SimulationClock::now());
                            for (std::size_t n = 0; n < s_solid_idx.size(); ++n) {
                                const std::size_t pi = s_solid_idx[n];
                                if (pi >= state.particles.velocity.size()) continue;
                                state.particles.velocity[pi] = s_solid_vel_keep[n];
                                if (pi < state.particles.affine.size())
                                    state.particles.affine[pi] = s_solid_affine_keep[n];
                            }
                            // ★★ The device advect tail is skipped outright while
                            // a solid substance is live: it moves parcels by the
                            // grid field and knows nothing about the own-cell
                            // exception, so a chunk would eject itself there
                            // exactly as it would have in the host path before
                            // that exception existed. The host tail runs instead
                            // — a measurable perf cost in these scenes, not a
                            // silent difference in result.
                            if (!solid_parcels_present) {
                                const auto gpu_advect_begin = SimulationClock::now();
                                bool deferred_g2p_available = true;
                                particle_tail_on_gpu = runGpuFluidAdvectTail(
                                    state, fluid_params, dt,
                                    context.compute, gpu_buffers,
                                    &gpu_advect_substeps,
                                    combine_g2p_tail_readback,
                                    &deferred_g2p_available,
                                    particle_residency.hostStale(),
                                    particle_residency.tailDispatchFlag(),
                                    particle_residency.retainPositions(fluid_params));
                                if (!particle_residency.finishTail(particle_tail_on_gpu)) {
                                    granular_substep_failed = true;
                                    break;
                                }
                                if (combine_g2p_tail_readback &&
                                    !particle_tail_on_gpu &&
                                    !deferred_g2p_available) {
                                    g2p_on_gpu = false;
                                }
                                if (particle_tail_on_gpu) {
                                    gpu_advect_ms = elapsedMilliseconds(
                                        gpu_advect_begin, SimulationClock::now());
                                }
                            }
                            // Granular counters are collected after Call 2 below:
                            // Fluid::step resets its output stats on entry.
                        }
                    }

                    if (!g2p_on_gpu) {
                        // GPU pressure or G2P failed — fall through to the full
                        // CPU step below (re-runs boundary+viscosity+pressure+G2P
                        // from the post-Call-1 grid; g2p_on_gpu stays false).
                        //
                        // Say WHICH stage gave up. This path used to be silent, and
                        // a silent fallback is indistinguishable from a slow GPU:
                        // a 4M-cell domain reported Pressure and G2P as "(CPU)"
                        // with no way to tell whether a buffer, the solver or the
                        // G2P kernel was the one that quit. Every other fallback in
                        // this file logs once; this one did not.
                        // Keyed on stage+size rather than a one-shot bool. A plain
                        // "log once per session" goes silent exactly when it is
                        // needed most: run a big sim, then start a new project in
                        // the SAME session and its fallback is invisible because
                        // the first one already burned the flag.
                        const char* stage = !upload_ok      ? "FLIP snapshot upload"
                                          : !viscosity_ok   ? "viscous diffusion"
                                          : !pressure_on_gpu ? "MGPCG pressure projection"
                                                             : "G2P";
                        const std::string key =
                            std::string(stage) + "|" +
                            std::to_string(state.grid.getCellCount());
                        static std::string last_fluid_fallback_key;
                        if (key != last_fluid_fallback_key) {
                            last_fluid_fallback_key = key;
                            SCENE_LOG_WARN(
                                "[SimCompute] GPU fluid step fell back to CPU at: " +
                                std::string(stage) +
                                " (cells=" + std::to_string(state.grid.getCellCount()) +
                                ", particles=" + std::to_string(state.particles.size()) +
                                "). Pressure and G2P now run on the host for this domain.");
                        }
                    }
                }
            }

            // ★ Every path below except "G2P and the advect tail both ran on the
            // device" reads grid.vel_* on the host: Call 2's host advect, or the
            // full CPU fallback step (which re-clamps itself, so a field the device
            // clamp never reached is still handled). Bring the field home first;
            // reading the stale copy would advect on the previous substep's
            // velocity - plausible, and wrong everywhere.
            if (!(g2p_on_gpu && particle_tail_on_gpu)) {
                if (!particle_residency.recover()) {
                    granular_substep_failed = true;
                    break;
                }
                granular_state_resident = false;
            }

            if (grid_velocity_device_only && !(g2p_on_gpu && particle_tail_on_gpu)) {
                if (!bring_grid_velocity_home()) {
                    SCENE_LOG_WARN("[SimCompute] granular grid velocity readback failed; "
                                   "substep abandoned instead of advecting on a stale field.");
                    granular_substep_failed = true;
                    break;
                }
            }

            if (g2p_on_gpu) {
                // Call 2: GPU G2P already done; run only tail stages.
                step_params.pressure_g2p_precomputed = true;
                step_params.particle_tail_precomputed = particle_tail_on_gpu;
                Fluid::step(state.particles, state.grid, step_params, dt,
                            nullptr, context.time_seconds, &state.fluid_stats);
                state.fluid_stats.g2p_ms       = gpu_g2p_ms;
                state.fluid_stats.g2p_on_gpu   = true;
                if (particle_tail_on_gpu) {
                    state.fluid_stats.advect_ms = gpu_advect_ms;
                    state.fluid_stats.advect_substeps = gpu_advect_substeps;
                }
                state.fluid_stats.pressure_ms     = gpu_pressure_ms;
                state.fluid_stats.pressure_on_gpu = true;
                state.fluid_stats.viscosity_ms         = gpu_viscosity_ms;
                state.fluid_stats.viscosity_on_gpu     = gpu_viscosity_sweeps > 0;
                state.fluid_stats.viscosity_sweeps_run = gpu_viscosity_sweeps;
                state.fluid_stats.pressure_cg_iterations = gpu_mgpcg_stats.pressure_cg_iterations;
                state.fluid_stats.pressure_cg_max_iterations = gpu_mgpcg_stats.pressure_cg_max_iterations;
                state.fluid_stats.pressure_cg_dot_count = gpu_mgpcg_stats.pressure_cg_dot_count;
                state.fluid_stats.pressure_cg_dot_ms = gpu_mgpcg_stats.pressure_cg_dot_ms;
                state.fluid_stats.pressure_cg_multigrid = gpu_mgpcg_stats.pressure_cg_multigrid;
                state.fluid_stats.pressure_cg_final_relative_residual =
                    gpu_mgpcg_stats.pressure_cg_final_relative_residual;
                state.fluid_stats.pressure_window_used =
                    gpu_mgpcg_stats.pressure_window_used;
                state.fluid_stats.pressure_window_cells =
                    gpu_mgpcg_stats.pressure_window_cells;
                if (fluid_params.granular_enabled &&
                    granular_substep + 1 == granular_solver_substeps) {
                    std::size_t yielded = 0, detached = 0, invalid = 0, sleeping = 0, damaged = 0;
                    std::size_t damage_over_10 = 0, damage_over_50 = 0, damage_over_90 = 0;
                    std::size_t strain_limited = 0, compaction_capped = 0;
                    for (uint32_t f : state.particles.granular_material_flags) {
                        yielded += (f & 1u) != 0u;
                        detached += (f & 2u) != 0u;
                        invalid += (f & 4u) != 0u;
                        sleeping += (f & 8u) != 0u;
                        strain_limited += (f & 16u) != 0u;
                        compaction_capped += (f & 32u) != 0u;
                    }
                    float max_yield = 0.0f, max_plastic = 0.0f;
                    float max_accumulated_plastic = 0.0f, max_damage = 0.0f;
                    float max_fracture_history = 0.0f;
                    double accumulated_plastic_sum = 0.0, damage_sum = 0.0;
                    double fracture_history_sum = 0.0;
                    std::size_t accumulated_plastic_samples = 0, damage_samples = 0;
                    std::size_t fracture_history_samples = 0;
                    for (float v : state.particles.granular_yield_value)
                        if (std::isfinite(v)) max_yield = std::max(max_yield, v);
                    for (float v : state.particles.granular_plastic_increment)
                        if (std::isfinite(v)) max_plastic = std::max(max_plastic, v);
                    for (float v : state.particles.granular_hardening) {
                        if (!std::isfinite(v)) continue;
                        max_accumulated_plastic = std::max(max_accumulated_plastic, v);
                        accumulated_plastic_sum += static_cast<double>(v);
                        ++accumulated_plastic_samples;
                    }
                    for (float v : state.particles.granular_damage) {
                        if (!std::isfinite(v)) continue;
                        max_damage = std::max(max_damage, v);
                        damaged += v > 1.0e-4f;
                        damage_over_10 += v >= 0.10f;
                        damage_over_50 += v >= 0.50f;
                        damage_over_90 += v >= 0.90f;
                        damage_sum += static_cast<double>(v);
                        ++damage_samples;
                    }
                    for (float v : state.particles.granular_fracture_history) {
                        if (!std::isfinite(v)) continue;
                        max_fracture_history = std::max(max_fracture_history, v);
                        fracture_history_sum += static_cast<double>(v);
                        ++fracture_history_samples;
                    }
                    state.fluid_stats.granular_yielded_particles = yielded;
                    state.fluid_stats.granular_detached_particles = detached;
                    state.fluid_stats.granular_invalid_particles = invalid;
                    state.fluid_stats.granular_sleeping_particles = sleeping;
                    state.fluid_stats.granular_strain_limited_particles = strain_limited;
                    state.fluid_stats.granular_compaction_capped_particles = compaction_capped;
                    state.fluid_stats.granular_damaged_particles = damaged;
                    state.fluid_stats.granular_damage_over_10_particles = damage_over_10;
                    state.fluid_stats.granular_damage_over_50_particles = damage_over_50;
                    state.fluid_stats.granular_damage_over_90_particles = damage_over_90;
                    state.fluid_stats.granular_max_yield_value = max_yield;
                    state.fluid_stats.granular_max_plastic_increment = max_plastic;
                    state.fluid_stats.granular_max_accumulated_plastic = max_accumulated_plastic;
                    state.fluid_stats.granular_mean_accumulated_plastic =
                        accumulated_plastic_samples > 0
                            ? static_cast<float>(accumulated_plastic_sum /
                                                 static_cast<double>(accumulated_plastic_samples))
                            : 0.0f;
                    state.fluid_stats.granular_max_fracture_history = max_fracture_history;
                    state.fluid_stats.granular_mean_fracture_history =
                        fracture_history_samples > 0
                            ? static_cast<float>(fracture_history_sum /
                                                 static_cast<double>(fracture_history_samples))
                            : 0.0f;
                    state.fluid_stats.granular_max_damage = max_damage;
                    state.fluid_stats.granular_mean_damage =
                        damage_samples > 0
                            ? static_cast<float>(damage_sum / static_cast<double>(damage_samples))
                            : 0.0f;
                    const auto elastic_step = Fluid::Granular::elasticStepInfo(
                        fluid_params.granular_young_modulus,
                        state.grid.voxel_size, dt,
                        granular_strain_rate, granular_overburden,
                granular_load.softening_min);
                    state.fluid_stats.granular_requested_young_modulus =
                        elastic_step.requested_young_modulus;
                    state.fluid_stats.granular_effective_young_modulus =
                        elastic_step.effective_young_modulus;
                    state.fluid_stats.granular_required_substeps =
                        elastic_step.required_substeps;
                    state.fluid_stats.granular_stiffness_capped = elastic_step.capped;
                }
            } else {
                // ★ Granular now has a real CPU path (Fluid::step runs the same
                // Drucker-Prager constitutive update, stress divergence and
                // settle as the Vulkan kernels), so this no longer falls through
                // to the incompressible liquid solver.
                //
                // That silent fallthrough was the worst outcome available: same
                // particles, same panel, a DIFFERENT material model, and a
                // resting sand pile turning into a boiling puddle with nothing
                // reporting it. Holding the step was the stopgap; a reference
                // implementation is the fix, and it is also what the roadmap's
                // CPU/Vulkan parity gate needs in order to be runnable at all.
                // Full CPU path (either GPU G2P not engaged or failed).
                Fluid::step(state.particles, state.grid, step_params, dt,
                            gpu_integrated_forces ? nullptr : context.force_snapshot,
                            context.time_seconds, &state.fluid_stats);
            }

            if (step_params.p2g_precomputed) {
                state.fluid_stats.p2g_ms     = gpu_p2g_ms;
                state.fluid_stats.p2g_on_gpu = true;
            }
            granular_p2g_ms_sum += state.fluid_stats.p2g_ms;
            granular_pressure_ms_sum += state.fluid_stats.pressure_ms;
            granular_viscosity_ms_sum += state.fluid_stats.viscosity_ms;
            granular_g2p_ms_sum += state.fluid_stats.g2p_ms;
            granular_advect_ms_sum += state.fluid_stats.advect_ms;
            granular_advect_substeps_sum += state.fluid_stats.advect_substeps;
            if (state.particles.size() != granular_particle_count_before)
                granular_state_resident = false;
            if (state.fluid_stats.gpu_fallback) {
                granular_substep_failed = true;
                break;
            }
            }

            // One readback per frame instead of one per substep: everything after
            // the elastic subcycle (cache, stats, render bridge, serialization)
            // sees the same host grid.vel_* the old per-substep round trip left.
            if (!particle_residency.recover()) {
                state.fluid_stats.gpu_fallback = true;
                continue;
            }

            if (grid_velocity_device_only && !bring_grid_velocity_home()) {
                SCENE_LOG_WARN("[SimCompute] granular grid velocity end-of-frame readback "
                               "failed; host grid velocity is one frame stale.");
            }

            state.fluid_stats.p2g_ms = granular_p2g_ms_sum;
            state.fluid_stats.pressure_ms = granular_pressure_ms_sum;
            state.fluid_stats.viscosity_ms = granular_viscosity_ms_sum;
            state.fluid_stats.g2p_ms = granular_g2p_ms_sum;
            state.fluid_stats.advect_ms = granular_advect_ms_sum;
            state.fluid_stats.advect_substeps = granular_advect_substeps_sum;
            if (fluid_params.granular_enabled && !granular_substep_failed) {
                const auto resolved_elastic = Fluid::Granular::elasticStepInfo(
                    fluid_params.granular_young_modulus,
                    state.grid.voxel_size,
                    granular_frame_dt / static_cast<float>(granular_solver_substeps),
                    granular_strain_rate, granular_overburden,
                granular_load.softening_min);
                state.fluid_stats.granular_requested_young_modulus =
                    granular_frame_elastic.requested_young_modulus;
                state.fluid_stats.granular_effective_young_modulus =
                    resolved_elastic.effective_young_modulus;
                state.fluid_stats.granular_required_substeps =
                    granular_frame_elastic.required_substeps;
                state.fluid_stats.granular_solver_substeps = granular_solver_substeps;
                state.fluid_stats.granular_stiffness_capped = resolved_elastic.capped;
                // Adaptive subcycling grants the full wave/strain request;
                // the legacy budget cannot change the authored material.
                state.fluid_stats.granular_wave_substeps =
                    granular_frame_elastic.wave_substeps;
                state.fluid_stats.granular_strain_substeps =
                    granular_frame_elastic.strain_substeps;
                state.fluid_stats.granular_strain_rate =
                    granular_frame_elastic.strain_rate;
                state.fluid_stats.granular_overburden_pressure =
                    granular_frame_elastic.overburden_pressure;
                state.fluid_stats.granular_load_measured = true;
                state.fluid_stats.granular_young_modulus_for_load =
                    granular_frame_elastic.young_modulus_for_load;
                state.fluid_stats.granular_stiffness_below_load =
                    granular_frame_elastic.below_load;
                state.fluid_stats.granular_min_softening =
                    granular_load.softening_min;
                state.fluid_stats.granular_softened_particles =
                    granular_load.softened_particles;
            }
            state.fluid_stats.recovered_solid_particles = std::max(
                state.fluid_stats.recovered_solid_particles,
                solid_recovery_count);
            // Patched in after the step because Fluid::step resets the stats
            // block: these two are MEASURED here, where the overlay was built,
            // and the pair is what distinguishes "no solid parcels" from
            // "parcels too thin for this voxel size" (see APICSolverStats).
            state.fluid_stats.solid_phase_particles = solid_phase_particle_count;
            state.fluid_stats.solid_phase_cells     = solid_phase_cell_count;

            // ── Whitewater (spray/foam/bubbles) — Ihmsen 2012 ────────────────
            // Secondary render-only particles generated from the post-step liquid
            // (relative velocity / wave crest / kinetic energy). Never fed back
            // into the pressure solve, so it cannot affect liquid mass/stability.
            // stepFoam early-outs cheaply when fluid_foam_params.enabled is false.
            if (i < grid_domains_.size()) {
                const auto& fparams = grid_domains_[i].fluid_foam_params;
                const uint32_t foam_seed =
                    static_cast<uint32_t>(context.time_seconds * 600.0f) +
                    static_cast<uint32_t>(i) * 9176u + 1u;
                // Spawn potentials on GPU when the domain is already solving there
                // (positions/velocities are resident, so this adds no transfer but
                // the readback). The emit RNG deliberately stays on the host: it
                // costs 0.13 ms and keeping it makes the GPU result directly
                // comparable with the CPU reference. Any failure falls through to
                // the full CPU path — foam can never go missing because of this.
                const float* gpu_expected = nullptr;
                const uint32_t* gpu_neigh = nullptr;
                float foam_crit_gpu_ms = 0.0f;
                float foam_neigh_gpu_ms = 0.0f;
                static thread_local std::vector<float> s_foam_expected;
                static thread_local std::vector<uint32_t> s_foam_neigh;
                const bool foam_gpu_available =
                    fluid_gpu_requested && context.compute &&
                    context.compute->supportsDispatch() &&
                    i < grid_domain_compute_buffers_.size();
                if (foam_gpu_available) {
                    // Counts for the foam about to be classified were dispatched at
                    // the end of the PREVIOUS step over this exact array. Read them
                    // BEFORE queueing anything new: a readback drains the queue, so
                    // doing it after runGpuFoamCriteria would make it wait on that
                    // call's freshly submitted bin+crit work for no reason.
                    const auto neigh_begin = SimulationClock::now();
                    if (consumeGpuFoamNeighbours(state.foam, context.compute,
                                                 grid_domain_compute_buffers_[i],
                                                 s_foam_neigh)) {
                        gpu_neigh = s_foam_neigh.data();
                    }
                    foam_neigh_gpu_ms =
                        elapsedMilliseconds(neigh_begin, SimulationClock::now());

                    const auto crit_begin = SimulationClock::now();
                    if (runGpuFoamCriteria(state, fparams, dt, context.compute,
                                           grid_domain_compute_buffers_[i],
                                           s_foam_expected)) {
                        gpu_expected = s_foam_expected.data();
                    }
                    foam_crit_gpu_ms =
                        elapsedMilliseconds(crit_begin, SimulationClock::now());
                }
                Fluid::stepFoam(state.particles, state.grid, state.foam, fparams,
                                fluid_params.gravity, dt, foam_seed, &state.foam_stats,
                                gpu_expected, gpu_neigh);
                // Foam is produced FROM the fluid and consumed by rendering; it
                // is a coupling in the same sense as the others, and a graph
                // that shows the others but hides this one would misdescribe the
                // step. Source and target are the same domain by construction.
                noteCoupling("foam_from_fluid", "fluid", "foam",
                             i < grid_domains_.size() ? grid_domains_[i].name
                                                      : std::string(),
                             i < grid_domains_.size() ? grid_domains_[i].name
                                                      : std::string());
                // Queue the next step's counts over the array stepFoam just left
                // behind — advected, culled and topped up. It stays untouched until
                // the consume above runs again, which is what makes the indices line
                // up. Silent no-op when the bins were not built this step.
                float foam_gpu_exec_ms = 0.0f;
                if (foam_gpu_available) {
                    dispatchGpuFoamNeighbours(state, fparams, context.compute,
                                              grid_domain_compute_buffers_[i]);
                    // Flush the foam pipeline here instead of letting it ride along
                    // with whatever downloads next.
                    //
                    // The Vulkan backend records dispatches and only submits when
                    // something forces a fence. None of the four foam kernels
                    // download in the step that queues them, so all of them — bins,
                    // criterion, neighbours — were being submitted by the density
                    // splat's readback, and their execution time was charged to
                    // "Density -> NanoVDB". That row read 15.6 ms with foam on and
                    // 1.5 ms with foam off: ~14 ms of foam work wearing someone
                    // else's name, while the foam block advertised 3.00 ms.
                    //
                    // One extra submit costs ~0.3-1 ms of WDDM latency. That is the
                    // price of the panel telling the truth about which phase to
                    // optimise, and it is cheap next to the misread it prevents.
                    const auto flush_begin = SimulationClock::now();
                    context.compute->synchronize();
                    foam_gpu_exec_ms =
                        elapsedMilliseconds(flush_begin, SimulationClock::now());
                }
                // stepFoam resets the stats block, so these are written after it.
                state.foam_stats.crit_on_gpu = (gpu_expected != nullptr);
                state.foam_stats.crit_gpu_ms = foam_crit_gpu_ms;
                state.foam_stats.neigh_on_gpu = (gpu_neigh != nullptr);
                state.foam_stats.neigh_gpu_ms = foam_neigh_gpu_ms;
                state.foam_stats.gpu_exec_ms = foam_gpu_exec_ms;
            }

            bool labels_on_gpu = false;
            if (fluid_gpu_requested && context.compute &&
                context.compute->backendType() == ComputeBackendType::VulkanCompute &&
                !fluid_params.granular_enabled &&
                (!step_params.solid_substance_tags ||
                 step_params.solid_substance_tags->empty()) &&
                i < grid_domain_compute_buffers_.size()) {
                auto& gpu_buffers = grid_domain_compute_buffers_[i];
                labels_on_gpu = ensureGpuFluidParticleBuffers(
                    state, context.compute, gpu_buffers, false) &&
                    Fluid::updateParticleLabelsGpu(
                        state, context.compute, gpu_buffers,
                        state.particle_label_stats);
            }
            if (!labels_on_gpu) {
                state.particle_label_stats = Fluid::updateParticleLabels(
                    state.particles, state.voxel_size, fluid_params.granular_enabled,
                    step_params.solid_substance_tags);
            }

            // Mist is a low-mass liquid parcel, so it remains in the APIC SoA
            // and keeps its substance/temperature. Inside a gas domain its
            // velocity follows the carrier field on a mass-scaled response
            // time. The next APIC step consumes this host velocity normally.
            for (std::size_t gas_index = 0;
                 gas_index < grid_domains_.size() &&
                 gas_index < grid_domain_states_.size();
                 ++gas_index) {
                if (gas_index == i) continue;
                const std::size_t carried = Fluid::applyMistGasDrag(
                    state, grid_domains_[gas_index],
                    grid_domain_states_[gas_index], dt);
                if (carried > 0) {
                    noteCoupling("mist_gas_drag", "gas", "fluid",
                                 grid_domains_[gas_index].name,
                                 grid_domains_[i].name);
                }
            }

            // ── GPU density splat ────────────────────────────────────────────
            const auto density_begin = SimulationClock::now();
            bool density_on_gpu = false;
            if (fluid_gpu_requested && context.compute &&
                context.compute->supportsDispatch() &&
                i < grid_domain_compute_buffers_.size()) {
                auto& gpu_buffers = grid_domain_compute_buffers_[i];
                if (ensureGridDomainComputeBuffers(*context.compute, gpu_buffers, state.grid)) {
                    density_on_gpu = runGpuFluidDensitySplat(state,
                                                             fluid_params,
                                                             context.compute,
                                                             gpu_buffers);
                    static bool logged_gpu_density_fallback = false;
                    if (!density_on_gpu && !logged_gpu_density_fallback) {
                        SCENE_LOG_WARN("[SimCompute] GPU APIC fluid density splat failed; falling back to CPU NanoVDB density bridge.");
                        logged_gpu_density_fallback = true;
                    }
                }
            }
            if (!density_on_gpu)
                splatFluidDensityCPU(state, fluid_params);

            const auto density_end = SimulationClock::now();
            state.fluid_stats.density_ms           = elapsedMilliseconds(density_begin, density_end);
            state.fluid_stats.density_on_gpu       = density_on_gpu;
            state.fluid_stats.active_fluid_cells   = state.active_density_cells;
            if (i < grid_domain_compute_buffers_.size()) {
                state.fluid_stats.normalize_window_cells =
                    grid_domain_compute_buffers_[i].fluid_normalize_window_cells;
                state.fluid_stats.normalize_window_used = state.fluid_stats.p2g_on_gpu &&
                    grid_domain_compute_buffers_[i].fluid_normalize_window_used;
                state.fluid_stats.occupancy_on_gpu = state.fluid_stats.g2p_on_gpu &&
                    grid_domain_compute_buffers_[i].fluid_mask_device_valid;
            }
            state.fluid_stats.forces_on_gpu        = gpu_integrated_forces && !force_fields_require_cpu;
            state.fluid_stats.gpu_requested        = fluid_gpu_requested;
            state.fluid_stats.gpu_compute_available = fluid_gpu_compute_available;
            state.fluid_stats.compute_device = fluid_gpu_compute_available && context.compute
                ? context.compute->backendName() : "CPU";
            state.fluid_stats.gpu_fallback = mixed_models
                ? !state.fluid_stats.mixed_model_step :
                fluid_gpu_requested &&
                (!gpu_integrated_forces || !step_params.p2g_precomputed || !density_on_gpu);

            // GPU status string. When g2p_on_gpu is set the all-GPU path ran,
            // which now includes the MGPCG pressure projection (Jacobi-PCG).
            // Force fields keep their evaluation on the CPU (noise / wind
            // surface-drag are not ported to the device), but the post-force
            // velocities are uploaded so P2G/pressure/G2P still run on the GPU —
            // hence "forces CPU(fields)" rather than a full CPU fallback.
            const std::string forces_loc = force_fields_require_cpu
                ? "forces CPU(fields)" : "forces GPU";
            if (mixed_models) {
                // Preserve measured mixed backend/status, including held-step reason.
            } else if (!fluid_gpu_requested) {
                state.fluid_stats.gpu_status = "CPU reference path";
            } else if (!fluid_gpu_compute_available) {
                state.fluid_stats.gpu_status = "GPU requested, but no simulation compute backend available; CPU fallback.";
            } else if (!gpu_integrated_forces) {
                state.fluid_stats.gpu_status = "GPU requested on " + state.fluid_stats.compute_device + ", but force integration failed; CPU fallback.";
            } else if (!step_params.p2g_precomputed) {
                state.fluid_stats.gpu_status = "GPU requested on " + state.fluid_stats.compute_device + ", but P2G failed; CPU P2G fallback.";
            } else if (!g2p_on_gpu) {
                state.fluid_stats.gpu_status = "GPU partial on " + state.fluid_stats.compute_device + ": P2G/density GPU, pressure+G2P CPU (PCG). " + forces_loc + ".";
            } else if (!density_on_gpu) {
                state.fluid_stats.gpu_status = "GPU on " + state.fluid_stats.compute_device + ": P2G/pressure(MGPCG)/G2P. Density CPU. " + forces_loc + ".";
            } else {
                state.fluid_stats.gpu_status = "GPU on " + state.fluid_stats.compute_device +
                    ": P2G/pressure(MGPCG)/G2P/density" +
                    std::string(particle_tail_on_gpu
                        ? "/advect+boundaries. Reseed topology CPU. "
                        : ". Advect+reseed CPU fallback. ") +
                    forces_loc + ".";
            }

            // ── Material State Field: liquid contact -> moisture (Phase 5) ────
            // Placed at the very end of the fluid branch, after the density splat
            // that this reads as liquid occupancy. Running it earlier would test
            // against the PREVIOUS frame's water, which shows up as an object
            // getting wet a frame after being submerged and — far worse — staying
            // dry for the frame a splash actually reaches it.
            //
            // Per-FLUID-domain on purpose. Wetting is a monotonic source, so two
            // overlapping tanks soaking the same plank is fine; the Phase 4
            // cooling had to leave the per-domain loop precisely because a
            // relaxation is not. Drying stays in the once-per-frame ambient pass.
            if (msf_fields_live && context.compute &&
                i < grid_domain_compute_buffers_.size()) {
                material_state_fields_.stepWetting(
                    *context.compute,
                    grid_domain_compute_buffers_[i].density,
                    state.grid.nx, state.grid.ny, state.grid.nz,
                    state.grid.origin,
                    state.grid.voxel_size,
                    dt);
                noteCoupling("fluid_to_surface_moisture", "fluid", "surface",
                             i < grid_domains_.size() ? grid_domains_[i].name
                                                      : std::string(),
                             std::string());
            }
            // Fluid::step is split into two calls on the Vulkan path. Its own
            // total_ms is therefore only the second, tiny tail call. Publish
            // the actual liquid-domain wall time after all GPU stages, labels,
            // whitewater and the density bridge have completed.
            state.fluid_stats.total_ms = elapsedMilliseconds(
                fluid_step_begin, SimulationClock::now());

#endif // IntelliSense must use the including function context.
