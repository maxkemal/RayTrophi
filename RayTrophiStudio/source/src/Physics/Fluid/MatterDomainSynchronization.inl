void ParticleSimulationSystem::synchronizeGridDomains() {
    if (grid_domain_states_.size() != grid_domains_.size()) {
        grid_domain_states_.resize(grid_domains_.size());
    }

    for (std::size_t i = 0; i < grid_domains_.size(); ++i) {
        auto& domain = grid_domains_[i];
        auto& state = grid_domain_states_[i];

        // Do not re-derive coupling flags from the material preset here.
        // These fields are editable domain overrides; doing that every sync
        // made the Water preset's non-flammable default silently uncheck
        // "Enable Flammable Surface" and prevented manual ignition tests.
        // Preset constructors may initialize them, but live UI/project values
        // must remain authoritative after synchronization.

        Vec3 bounds_min = domain.bounds_min;
        Vec3 bounds_max = domain.bounds_max;
        float current_padding = std::max(0.0f, domain.padding);

        if (domain.source_mode == SimulationGridDomainSourceMode::ObjectBounds && grid_domain_bounds_resolver_) {
            Vec3 resolved_min = bounds_min;
            Vec3 resolved_max = bounds_max;
            if (grid_domain_bounds_resolver_(domain, resolved_min, resolved_max)) {
                bounds_min = resolved_min;
                bounds_max = resolved_max;
                domain.bounds_min = resolved_min;
                domain.bounds_max = resolved_max;
            }
        } else if (domain.source_mode == SimulationGridDomainSourceMode::Adaptive) {
            current_padding = 0.0f; // Padding handled internally to preserve floor locking
            Vec3 min_p(1.0e10f, 1.0e10f, 1.0e10f);
            Vec3 max_p(-1.0e10f, -1.0e10f, -1.0e10f);
            bool has_particles = false;

            if (simulationDomainHasLiquid(domain.type)) {
                if (!state.particles.position.empty()) {
                    for (const auto& pos : state.particles.position) {
                        min_p = Vec3::min(min_p, pos);
                        max_p = Vec3::max(max_p, pos);
                    }
                    has_particles = true;
                }
            } else {
                const std::size_t num_particles = buffers_.alive.size();
                for (std::size_t p = 0; p < num_particles; ++p) {
                    if (buffers_.alive[p] != 0u) {
                        Vec3 pos(buffers_.position_x[p], buffers_.position_y[p], buffers_.position_z[p]);
                        min_p = Vec3::min(min_p, pos);
                        max_p = Vec3::max(max_p, pos);
                        has_particles = true;
                    }
                }
            }

            if (has_particles) {
                float pad_val = std::max(0.0f, domain.padding);
                if (pad_val < 1.0e-5f) {
                    pad_val = std::max(domain.voxel_size * 3.0f, 0.05f);
                }
                min_p = min_p - Vec3(pad_val, pad_val, pad_val);
                max_p = max_p + Vec3(pad_val, pad_val, pad_val);

                if (domain.adaptive_lock_floor) {
                    min_p.y = domain.adaptive_floor_y;
                }

                // Grid snapping to voxel size to completely eliminate sub-voxel jitter
                const float voxel_size = domain.voxel_size > 1.0e-6f ? domain.voxel_size : 0.1f;
                min_p.x = std::floor(min_p.x / voxel_size) * voxel_size;
                min_p.y = std::floor(min_p.y / voxel_size) * voxel_size;
                min_p.z = std::floor(min_p.z / voxel_size) * voxel_size;

                max_p.x = std::ceil(max_p.x / voxel_size) * voxel_size;
                max_p.y = std::ceil(max_p.y / voxel_size) * voxel_size;
                max_p.z = std::ceil(max_p.z / voxel_size) * voxel_size;

                bounds_min = min_p;
                bounds_max = max_p;
                domain.bounds_min = bounds_min;
                domain.bounds_max = bounds_max;
            }
        }

        const Vec3 mn = Vec3::min(bounds_min, bounds_max) - current_padding;
        const Vec3 mx = Vec3::max(bounds_min, bounds_max) + current_padding;
        const Vec3 extent = mx - mn;
        const float max_extent = std::max({ extent.x, extent.y, extent.z, 0.001f });

        const int max_auto_res = std::clamp(domain.max_auto_resolution, 32, 512);
        int res_x, res_y, res_z;
        float voxel_size;

        if (domain.preserve_voxel_size_on_resize && domain.voxel_size > 1e-6f) {
            voxel_size = domain.voxel_size;
            res_x = std::clamp(static_cast<int>(std::ceil(std::max(extent.x, 0.001f) / voxel_size)), 8, max_auto_res);
            res_y = std::clamp(static_cast<int>(std::ceil(std::max(extent.y, 0.001f) / voxel_size)), 8, max_auto_res);
            res_z = std::clamp(static_cast<int>(std::ceil(std::max(extent.z, 0.001f) / voxel_size)), 8, max_auto_res);
        } else {
            // Derive per-axis cell counts proportionally from physical extents at the
            // finest voxel_size the max_auto_res allows (max_extent / max_auto_res).
            // This prevents the classic "large flat domain" problem: when all three
            // axes start at max_auto_res and budget clamping reduces them equally, the
            // thin Y axis loses cells disproportionately because max_extent/max_auto_res
            // gives a coarse voxel for the short dimension.
            voxel_size = max_extent / static_cast<float>(max_auto_res);
            res_x = std::clamp(static_cast<int>(std::ceil(std::max(extent.x, 0.001f) / voxel_size)), 8, 1024);
            res_y = std::clamp(static_cast<int>(std::ceil(std::max(extent.y, 0.001f) / voxel_size)), 8, 1024);
            res_z = std::clamp(static_cast<int>(std::ceil(std::max(extent.z, 0.001f) / voxel_size)), 8, 1024);
            domain.voxel_size = voxel_size;
        }

        // max_auto_resolution is the single authority for grid density: the
        // per-axis clamps above already cap every axis at max_auto_res, so the
        // total can never exceed max_auto_res^3. Deriving the cell budget from
        // that same knob makes the UI honest — "Max Auto Resolution = 512" means
        // the solver may actually reach 512 per axis (voxel size permitting),
        // instead of a hidden fixed cell cap silently collapsing a 2 m cube to
        // ~80^3 (= 25 mm voxels). The y-aspect term keeps the extra headroom for
        // flat domains (thin Y) where allocated cells are dominated by empty
        // space above the slab; it only ever raises an already-non-binding
        // budget. A high absolute ceiling remains purely as an OOM guard — the
        // UI's live cell-count + memory preview is the real guardrail for the
        // user's explicit choice. (GPU backends can lift this further once the
        // MGPCG path is live-wired.)
        const std::size_t knob_budget =
            static_cast<std::size_t>(max_auto_res) *
            static_cast<std::size_t>(max_auto_res) *
            static_cast<std::size_t>(max_auto_res);
        constexpr std::size_t MAX_GRID_DOMAIN_CELLS_HARD_CAP = 134217728; // 512^3 OOM guard
        // Flat-domain headroom is capped at 4x knob^3. The old 0.01 floor let a
        // thin-Y domain inflate the budget up to 100x — with knob=512 that meant
        // sailing straight into the hard cap (134M cells), and the real memory
        // hog scales with CELLS: ~8 particles/fluid-cell x 64B (SoA pos/vel/
        // affine/flags) + CPU grid ~37B/cell + GPU mirrors ~70B/cell => tens of
        // GB. 4x keeps the thin-slab benefit without the runaway footprint.
        const float y_aspect = std::clamp(extent.y / max_extent, 0.25f, 1.0f);
        const std::size_t adaptive_budget = std::min(
            static_cast<std::size_t>(static_cast<double>(knob_budget) / static_cast<double>(y_aspect)),
            MAX_GRID_DOMAIN_CELLS_HARD_CAP);
        clampGridResolutionToCellBudget(res_x, res_y, res_z, adaptive_budget);

        // The grid is a single-voxel-size (cubic) MAC grid, so each axis spans
        // exactly res_i * voxel_size. To COVER the whole domain box on every
        // axis we must pick the COARSEST per-axis voxel (extent_i / res_i): that
        // guarantees res_i * voxel_size >= extent_i for all i. Taking the
        // minimum instead under-covers the longest axis whenever the per-axis
        // resolutions are not perfectly proportional — which happens routinely
        // because a thin axis gets clamped UP to the 8-cell floor (giving it a
        // finer voxel) or the budget clamp shrinks the largest axis. The classic
        // symptom is a tall object whose Y is silently cropped (e.g. a 10 m
        // column rendered as 8 m) — i.e. the domain looks flattened even though
        // the bounds carry the object's true width/height/depth. Using max keeps
        // the aspect ratio intact; the off-axis over-coverage is at most one
        // extra voxel of headroom (harmless padding at the walls).
        const float vs_x = extent.x / static_cast<float>(std::max(res_x, 1));
        const float vs_y = extent.y / static_cast<float>(std::max(res_y, 1));
        const float vs_z = extent.z / static_cast<float>(std::max(res_z, 1));
        voxel_size = std::max({ vs_x, vs_y, vs_z });
        if (voxel_size < 1e-6f) {
            voxel_size = max_extent / static_cast<float>(std::max({ res_x, res_y, res_z, 1 }));
        }
        domain.voxel_size = voxel_size;
        domain.resolution_x = res_x;
        domain.resolution_y = res_y;
        domain.resolution_z = res_z;
        domain.max_auto_resolution = max_auto_res;

        // ★ Combustion needs the Fuel channel, but Fuel is NOT in
        // defaultGridDomainChannels(). A gas domain with fire_enabled and no
        // Fuel channel silently drops every fuel deposit and every fuel-bearing
        // flow source — the domain looks configured for fire and simply never
        // ignites. Provision it here, in the one place every domain passes
        // through, so the requirement cannot be forgotten from the UI, from a
        // preset, or from a script. Written back to the desc so the panel and
        // the saved project agree with what the solver actually allocated.
        if (simulationDomainHasGas(domain.type) && domain.fire_enabled) {
            domain.channels |=
                static_cast<uint32_t>(SimulationGridDomainChannelFlags::Fuel) |
                static_cast<uint32_t>(SimulationGridDomainChannelFlags::Temperature);
        }
        const uint32_t channels = domain.channels;
        // Liquid domains skip the combustion channels (temperature/fuel/
        // interaction) entirely — 3 floats/cell saved, which matters at high
        // resolution. A type switch (Fluid <-> Gas) must force a re-allocate
        // even when the resolution is unchanged.
        const auto phase_layouts = Fluid::resolvePhaseLayouts(
            domain, mn, mx, res_x, res_y, res_z, voxel_size);
        Fluid::synchronizePhaseStorage(state, domain, phase_layouts);
        state.logical_bounds_min = mn;
        state.logical_bounds_max = mx;
        state.logical_bounds_valid = true;
        state.phase_config_hash = Fluid::hashPhaseSettings(0, domain);

        state.type = domain.type;
        // ★★ Do NOT clear the motion delta here. It is produced by this sync but
        // CONSUMED by the fluid step, and the two do not run in lockstep: the
        // domain gizmo calls synchronizeGridDomainsNow() on every drag frame so
        // the viewport box follows the cursor. Clearing per sync meant the drag's
        // delta was recorded and then wiped by the very next sync before any step
        // could read it — the liquid was carried rigidly with the box (below) but
        // never received the velocity impulse, so it never sloshed. Accumulate,
        // and let the consumer clear it.
        if (!simulationDomainHasLiquid(domain.type)) {
            state.domain_motion_delta = Vec3(0.0f, 0.0f, 0.0f);
        }

        // Fluid seed AABB follows the domain's translation but NOT its resize.
        // We compare the corner deltas: if both corners shifted by the same
        // vector the user translated the domain → glue the seed to it. If
        // only one corner moved the user is dragging an extent → keep the
        // seed in place (its absolute world coords stay valid).
        if (simulationDomainHasLiquid(domain.type)) {
            if (domain.fluid_seed_anchor_min.x > -9.99e9f) {
                const Vec3 delta_min = mn - domain.fluid_seed_anchor_min;
                const Vec3 delta_max = mx - domain.fluid_seed_anchor_max;
                const Vec3 diff = delta_max - delta_min;
                const float diff_mag = std::abs(diff.x) + std::abs(diff.y) + std::abs(diff.z);
                const float trans_mag = std::abs(delta_min.x) + std::abs(delta_min.y) + std::abs(delta_min.z);
                if (diff_mag < 1e-4f && trans_mag > 1e-6f) {
                    domain.fluid_seed_min = domain.fluid_seed_min + delta_min;
                    domain.fluid_seed_max = domain.fluid_seed_max + delta_min;
                    state.domain_motion_delta = state.domain_motion_delta + delta_min;
                    Fluid::translateLiquidParticles(state, delta_min);
                }
            } else {
                // First sync as a fluid domain. Drop a sensible default seed
                // AABB inside the domain (upper half, quarter-sized) so the
                // user sees the cyan box inside the bounds instead of stuck
                // at the legacy (0..1, 1..1.5, 0..1) world coords that may
                // be far outside an arbitrary domain center.
                const Vec3 domain_extent = mx - mn;
                const Vec3 seed_extent = Vec3(domain_extent.x * 0.45f,
                                              domain_extent.y * 0.35f,
                                              domain_extent.z * 0.45f);
                const Vec3 current_center = (mn + mx) * 0.5f;
                const Vec3 seed_center = Vec3(current_center.x,
                                              mn.y + domain_extent.y * 0.75f,
                                              current_center.z);
                domain.fluid_seed_min = seed_center - seed_extent * 0.5f;
                domain.fluid_seed_max = seed_center + seed_extent * 0.5f;
            }
            domain.fluid_seed_anchor_min = mn;
            domain.fluid_seed_anchor_max = mx;
        }

        state.bounds_min = state.grid.origin;
        state.bounds_max = Fluid::gridBoundsMax(state.grid);
        state.resolution_x = state.grid.nx;
        state.resolution_y = state.grid.ny;
        state.resolution_z = state.grid.nz;
        state.voxel_size = state.grid.voxel_size;
        state.channels = channels;
        state.valid = domain.enabled && res_x > 0 && res_y > 0 && res_z > 0;
        state.active_density_cells = 0;
        state.max_density = 0.0f;
        state.active_density_min[0] = state.active_density_min[1] =
            state.active_density_min[2] = 0;
        state.active_density_max[0] = state.active_density_max[1] =
            state.active_density_max[2] = -1;

        // Fluid-only: process a pending seed request from the UI. The legacy
        // FluidObject seeded directly from the panel; the unified path keeps
        // the seed deferred so it applies on the next sim tick with a freshly
        // resized grid.
        if (simulationDomainHasLiquid(domain.type) &&
            domain.fluid_pending_seed && state.valid) {
            if (domain.fluid_replace_on_seed) {
                state.particles.clear();
                state.foam.clear();   // drop stale whitewater with the old liquid
            }
            // Resolve the effective seed AABB. FillLevel mode treats the domain
            // as a resting tank: fill the whole footprint from the floor up to
            // fluid_fill_level of the domain height (skips the long emission /
            // settling transient for standing water). SeedBox uses the explicit
            // user AABB. Particles are emitted at rest (v=0) by seedBox either
            // way, so the result starts in hydrostatic balance.
            const std::size_t seed_budget =
                state.particles.size() < domain.fluid_max_particles
                    ? domain.fluid_max_particles - state.particles.size()
                    : 0u;

            // ppc is a STABILITY constant, not a budget knob: it must stay > 1 so
            // the cells carry enough samples to build real internal pressure
            // (incompressibility). At 1 ppc the liquid is under-resolved and just
            // collapses. So ppc is fixed and, when the budget can't afford the
            // full target fill, the fill HEIGHT drops instead (complete layers
            // from the floor up) — fully-resolved, stable, pile-free.
            const int ppc = std::max(1, domain.fluid_seed_particles_per_cell);

            Vec3 seed_lo = domain.fluid_seed_min;
            Vec3 seed_hi = domain.fluid_seed_max;
            if (domain.fluid_seed_mode == FluidSeedMode::FillLevel) {
                computeFluidFillSeedAABB(Fluid::liquidGrid(state).origin,
                                         Fluid::gridBoundsMax(Fluid::liquidGrid(state)),
                                         Fluid::liquidGrid(state).voxel_size,
                                         domain.fluid_fill_level,
                                         domain.fluid_fill_wall_margin,
                                         ppc, seed_budget,
                                         seed_lo, seed_hi);
            }

            // The seeded density IS the rest density: couple the solver's
            // density-correction target to the seeded ppc so a freshly filled
            // tank starts at exactly "target per cell" (over == 0). Without this,
            // a seed denser than the fixed default target makes every cell
            // over-populated and the density-targeted pressure projection
            // permanently expels particles upward (the tank "rises").
            domain.fluid_params.particles_per_cell = ppc;
            // Stable per-domain seed: index-based, so jitter patterns are
            // reproducible across runs and don't depend on heap addresses
            // (the desc vector can reallocate as domains are added).
            const std::size_t seed_begin = state.particles.size();
            Fluid::seedBox(state.particles,
                           Fluid::liquidGrid(state),
                           seed_lo,
                           seed_hi,
                           ppc,
                           /*seed=*/static_cast<uint32_t>(i + 1u) * 2654435761u,
                           seed_budget,
                           fluidDomainAmbientKelvin(domain, world_thermal_));
            const auto seed_model = domain.fluid_params.granular_enabled
                ? Fluid::MatterConstitutiveModel::Granular
                : Fluid::MatterConstitutiveModel::Fluid;
            for (std::size_t particle = seed_begin;
                 particle < state.particles.constitutive_model.size();
                 ++particle) {
                state.particles.constitutive_model[particle] =
                    static_cast<uint8_t>(seed_model);
            }
            domain.fluid_pending_seed = false;
        }
    }
}
