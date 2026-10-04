void ParticleSimulationSystem::injectFlowSourcesIntoGridDomains(
    float dt,
    float time_seconds,
    int frame,
    SimulationComputeContext* compute) {
    if (flow_sources_.empty() || grid_domain_states_.empty()) {
        return;
    }

    const float time_scale = std::max(0.0f, dt);
    // Parent motion first, unconditionally — see advanceFlowSourceMotion().
    advanceFlowSourceMotion(dt, frame);
    for (auto& source : flow_sources_) {
        const SimulationFlowSourceFrame resolved = resolveFlowSourceFrame(source, frame);
        const SimulationFlowSourceDesc::Keyframe& keyed = resolved.keyed;
        if (resolved.parent_missing) continue;
        if (!keyed.enabled ||
            source.domain_index < 0 ||
            source.domain_index >= static_cast<int>(grid_domain_states_.size())) {
            continue;
        }

        // Time Limit check (Houdini/Blender flow emitter style)
        if (source.use_time_limit) {
            if (time_seconds < source.start_time || time_seconds > source.end_time) {
                continue;
            }
        }

        auto& state = grid_domain_states_[static_cast<std::size_t>(source.domain_index)];
        if (!state.valid) {
            continue;
        }
        // Fluid (APIC liquid) flow sources spawn particles instead of
        // injecting density. Spawn rate accumulator survives across steps so
        // fractional rate*dt counts emit correctly. Capped by the domain's
        // max_particles.
        const bool liquid_source = simulationDomainHasLiquid(state.type) &&
            (!simulationDomainHasGas(state.type) ||
             source.phase == SimulationFlowSourceDesc::Phase::Liquid);
        if (liquid_source) {
            const auto& fluid_domain = grid_domains_[static_cast<std::size_t>(source.domain_index)];
            const float rate = std::max(0.0f, keyed.flow_rate);
            source.fluid_emit_accumulator += rate * std::max(0.0f, dt);
            int emit_count = static_cast<int>(source.fluid_emit_accumulator);
            if (emit_count <= 0) continue;
            source.fluid_emit_accumulator -= static_cast<float>(emit_count);

            // Hysteresis gate: reseed trims over-populated cells every step,
            // creating a small capacity gap even at max_particles. Without a
            // dead-band, the emitter fills that gap each step producing a
            // visible "trickle" of particles at full capacity.
            // Only allow emission when at least 1% capacity is available so
            // normal filling (empty → full) is unaffected but steady-state
            // at-max oscillation is suppressed.
            // ★ fluid_max_particles is a RESOURCE ceiling (memory/perf), not a
            // budget on how much material may ever be introduced. An emitter
            // authored as a continuous source is SUPPOSED to keep replacing what
            // burns away; bounding total emission is what use_particle_limit and
            // use_time_limit on the source are for.
            //
            // ★★ A previous revision subtracted state.burned_particles here so
            // burned mass could not fund new emission. It was reverted: it
            // silently repurposed the ceiling, and it would have throttled every
            // ordinary fountain draining through an open boundary too. The
            // counter is kept as telemetry only — read it, do not gate on it.
            const std::size_t max_p = fluid_domain.fluid_max_particles;
            const std::size_t cur_p = state.particles.size();
            const std::size_t dead_band = std::max<std::size_t>(1u, max_p / 100u);
            const std::size_t remaining =
                (cur_p + dead_band < max_p) ? (max_p - cur_p) : 0u;
            emit_count = std::min<int>(emit_count, static_cast<int>(remaining));
            if (emit_count <= 0) continue;

            // Particle budget limit check
            if (source.use_particle_limit) {
                int limit_rem = source.max_emitted_particles - source.total_emitted_particles;
                if (limit_rem <= 0) {
                    continue;
                }
                emit_count = std::min<int>(emit_count, limit_rem);
            }
            if (emit_count <= 0) continue;

            // Emission frame: keyframe channels + object parenting already
            // folded in. Using the resolved values (rather than the raw desc)
            // is what lets a nozzle ride a moving hose and be keyed at once.
            const Vec3  emit_origin = resolved.position;
            const float emit_radius = std::max(1e-4f, keyed.radius);

            // Resolve spawn volume for ObjectBounds; Point uses the resolved
            // origin + radius sphere; MeshSurface samples per-particle below.
            Vec3 bounds_min = emit_origin - Vec3(emit_radius);
            Vec3 bounds_max = emit_origin + Vec3(emit_radius);
            if (source.source_mode == SimulationFlowSourceMode::ObjectBounds && flow_source_bounds_resolver_) {
                Vec3 resolved_min, resolved_max;
                if (flow_source_bounds_resolver_(source, resolved_min, resolved_max)) {
                    bounds_min = Vec3::min(resolved_min, resolved_max);
                    bounds_max = Vec3::max(resolved_min, resolved_max);
                }
            }
            // Over-pack guard: a high particles/sec dumped into a small spawn
            // volume in ONE step stacks dozens of particles into a single cell.
            // The density-correction term then sees a huge overshoot and blasts
            // them outward laterally — the source "splatters" into a disc/plate.
            // Cap this step's emission at what the spawn volume can physically
            // hold at peak packing, and return the surplus to the accumulator so
            // it emits over the following steps instead of all at once. (Mesh
            // surface emission spreads over an area, so it is left uncapped.)
            if (source.source_mode != SimulationFlowSourceMode::MeshSurface) {
                const float h = std::max(1e-4f, Fluid::liquidGrid(state).voxel_size);
                float spawn_volume;
                if (source.source_mode == SimulationFlowSourceMode::ObjectBounds) {
                    const Vec3 ext = bounds_max - bounds_min;
                    spawn_volume = std::max(0.0f, ext.x) * std::max(0.0f, ext.y) * std::max(0.0f, ext.z);
                } else {
                    const float r = emit_radius;
                    spawn_volume = (4.0f / 3.0f) * 3.14159265358979f * r * r * r;
                }
                const double spawn_cells = std::max(1.0, static_cast<double>(spawn_volume) / (h * h * h));
                const int pack_ceiling = std::max({ fluid_domain.fluid_params.particles_per_cell,
                                                    fluid_domain.fluid_params.reseed_max_per_cell, 1 });
                const int cap = std::max(1, static_cast<int>(spawn_cells * static_cast<double>(pack_ceiling)));
                if (emit_count > cap) {
                    // Keep at most one locally packable batch as debt. Returning
                    // every rejected particle made the accumulator grow without
                    // bound whenever authored rate exceeded local capacity; a
                    // later radius/keyframe change then released that history as
                    // the delayed particle avalanche mistaken for a seed replay.
                    const int deferred = std::min(emit_count - cap, cap);
                    source.fluid_emit_accumulator += static_cast<float>(deferred);
                    emit_count = cap;
                }
            }

            // What this source is pouring. Empty name -> untagged, which the
            // consumers read as "use the domain's single material" — the exact
            // behaviour every existing scene has.
            const uint32_t emit_substance =
                RayTrophiSim::Fluid::substanceTag(source.fluid_substance);
            RayTrophiSim::Fluid::MatterConstitutiveModel emit_model =
                source.initial_constitutive_model;
            if (emit_model == RayTrophiSim::Fluid::MatterConstitutiveModel::Auto &&
                !source.fluid_substance.empty()) {
                const auto binding = std::find_if(
                    fluid_domain.fluid_substance_materials.begin(),
                    fluid_domain.fluid_substance_materials.end(),
                    [&](const auto& entry) {
                        return entry.substance == source.fluid_substance;
                    });
                if (binding != fluid_domain.fluid_substance_materials.end()) {
                    emit_model = binding->constitutive_model;
                }
            }
            if (emit_model == RayTrophiSim::Fluid::MatterConstitutiveModel::Auto &&
                !source.fluid_substance.empty()) {
                if (const SubstanceProfile* profile =
                        tryFindSubstance(source.fluid_substance)) {
                    emit_model = profile->default_constitutive_model;
                }
            }
            if (emit_model == RayTrophiSim::Fluid::MatterConstitutiveModel::Auto) {
                emit_model = fluid_domain.fluid_params.granular_enabled
                    ? RayTrophiSim::Fluid::MatterConstitutiveModel::Granular
                    : RayTrophiSim::Fluid::MatterConstitutiveModel::Fluid;
            }
            // Birth temperature, Kelvin. ★ This used to be a literal 0.0f —
            // every emitted parcel was born at absolute zero, contradicting the
            // FluidParticles note that emit starts at ambient. Harmless while
            // nothing read the field; the thermal-liquid chain would freeze such
            // a parcel on its first contact.
            const float emit_kelvin =
                source.fluid_temperature_override
                    ? std::max(1.0f, source.fluid_temperature_kelvin)
                    : (static_cast<std::size_t>(source.domain_index) < grid_domains_.size()
                           ? fluidDomainAmbientKelvin(
                                 grid_domains_[static_cast<std::size_t>(source.domain_index)],
                                 world_thermal_)
                           : world_thermal_.ambientKelvin());

            // Per-source-per-particle hash seed so jitter is deterministic but
            // not synchronized across sources.
            const uint32_t source_seed_base =
                static_cast<uint32_t>(reinterpret_cast<std::uintptr_t>(&source) >> 4) *
                    2654435761u;
            // Include the lifetime emission serial. Using only the per-step p
            // index respawned the exact same sample positions every frame; APIC
            // reseed then trimmed the stacked particles and a continuous hose
            // looked like a one-shot SeedBox event.
            const uint32_t emission_serial_base =
                static_cast<uint32_t>(source.fluid_emit_sample_serial);
            source.fluid_emit_sample_serial += static_cast<uint64_t>(emit_count);

            const std::size_t before_emission = state.particles.size();
            state.particles.reserve(state.particles.size() + static_cast<std::size_t>(emit_count));
            for (int p = 0; p < emit_count; ++p) {
                const uint32_t s =
                    source_seed_base ^
                    ((emission_serial_base + static_cast<uint32_t>(p)) *
                     2246822519u);
                const float u1 = hashUnitFloat(s);
                const float u2 = hashUnitFloat(s ^ 0xdeadbeefu);
                const float u3 = hashUnitFloat(s ^ 0x9e3779b9u);
                Vec3 spawn_pos;
                Vec3 spawn_normal(0.0f, 0.0f, 0.0f); // valid only for MeshSurface
                if (source.source_mode == SimulationFlowSourceMode::MeshSurface && flow_source_surface_sampler_) {
                    ParticleSurfaceSample sample;
                    if (flow_source_surface_sampler_(source, s, sample)) {
                        // Offset slightly along normal so particles spawn just
                        // off the surface, not embedded.
                        spawn_pos = sample.position + sample.normal * std::max(0.001f, emit_radius * 0.25f);
                        spawn_normal = sample.normal;
                    } else {
                        spawn_pos = emit_origin;
                    }
                } else if (source.source_mode == SimulationFlowSourceMode::ObjectBounds) {
                    spawn_pos.x = bounds_min.x + u1 * (bounds_max.x - bounds_min.x);
                    spawn_pos.y = bounds_min.y + u2 * (bounds_max.y - bounds_min.y);
                    spawn_pos.z = bounds_min.z + u3 * (bounds_max.z - bounds_min.z);
                } else {
                    // Point: rejection sample inside unit sphere then scale.
                    Vec3 d(u1 * 2.0f - 1.0f, u2 * 2.0f - 1.0f, u3 * 2.0f - 1.0f);
                    const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
                    if (r2 > 1.0f) {
                        const float inv = 1.0f / std::sqrt(r2);
                        d = d * (inv * std::cbrt(hashUnitFloat(s ^ 0x68e31da4u)));
                    }
                    spawn_pos = emit_origin + d * emit_radius;
                }
                // Break the laminar stream: an APIC liquid has nothing to
                // disperse a column of identical-velocity particles mid-air, so
                // without a per-particle perturbation the emitted mass falls as
                // a coherent sheet/plate. Add random jitter scaled by the
                // emission speed (0 spread => exact source.velocity, laminar).
                Vec3 emit_vel = resolved.velocity;
                // MeshSurface + emit-along-normal: redirect the emission speed
                // along the local surface normal so the liquid sprays off the
                // geometry instead of all moving in one global direction.
                if (source.fluid_emit_along_normal &&
                    source.source_mode == SimulationFlowSourceMode::MeshSurface) {
                    const float nlen = spawn_normal.length();
                    if (nlen > 1e-5f) {
                        // Speed comes from the resolved vector so a parented
                        // nozzle still emits at its authored rate; only the
                        // DIRECTION is taken from the surface.
                        emit_vel = spawn_normal * (resolved.velocity.length() / nlen);
                    }
                }
                if (source.fluid_velocity_spread > 0.0f) {
                    const float jitter_mag = source.fluid_velocity_spread * emit_vel.length();
                    if (jitter_mag > 1e-6f) {
                        const uint32_t vs = s ^ 0x1b56c4e9u;
                        const Vec3 jitter(
                            hashUnitFloat(vs)               * 2.0f - 1.0f,
                            hashUnitFloat(vs ^ 0x7feb352du) * 2.0f - 1.0f,
                            hashUnitFloat(vs ^ 0x846ca68bu) * 2.0f - 1.0f);
                        emit_vel = emit_vel + jitter * jitter_mag;
                    }
                }
                // ★ Hashed ONCE per source per step, not per particle: it is a
                // property of the source, and hashing a string inside the spawn
                // loop would put a per-character cost on every emitted particle
                // for a value that cannot change between them.
                if (!Fluid::gridContains(Fluid::liquidGrid(state), spawn_pos)) {
                    continue;
                }
                state.particles.emit(
                    spawn_pos, emit_vel, emit_kelvin, 0.0f, emit_substance,
                    nullptr, nullptr, 0.0f, emit_model);
            }
            source.total_emitted_particles +=
                static_cast<int>(state.particles.size() - before_emission);
            continue;
        }

        // Parented sources emit at the resolved world point. ObjectBounds /
        // MeshSurface modes overwrite this below from the bounds resolver —
        // those modes ARE an object binding, so the geometry wins over a
        // parent offset rather than the two fighting.
        Vec3 source_center = resolved.position;
        float source_radius = std::max(0.001f, keyed.radius);
        Vec3 source_min = source_center - Vec3(source_radius);
        Vec3 source_max = source_center + Vec3(source_radius);
        if ((source.source_mode == SimulationFlowSourceMode::ObjectBounds ||
            source.source_mode == SimulationFlowSourceMode::MeshSurface) &&
            flow_source_bounds_resolver_) {
            Vec3 resolved_min;
            Vec3 resolved_max;
            if (!flow_source_bounds_resolver_(source, resolved_min, resolved_max)) {
                continue;
            }
            const Vec3 mn = Vec3::min(resolved_min, resolved_max);
            const Vec3 mx = Vec3::max(resolved_min, resolved_max);
            source_center = (mn + mx) * 0.5f;
            source_min = mn;
            source_max = mx;
        }
        const Vec3 source_half_extent =
            (source.source_mode == SimulationFlowSourceMode::ObjectBounds)
                ? (source_max - source_min) * 0.5f
                : Vec3(source_radius);

        // Mesh-surface gas is represented by a deterministic set of area-weighted
        // surface samples and a shell thickness controlled by Source Radius.
        std::vector<Vec3> gas_surface_samples;
        if (source.source_mode == SimulationFlowSourceMode::MeshSurface) {
            if (!flow_source_surface_sampler_) continue;
            constexpr uint32_t kGasSurfaceSamples = 128u;
            gas_surface_samples.reserve(kGasSurfaceSamples);
            for (uint32_t sample_index = 0; sample_index < kGasSurfaceSamples; ++sample_index) {
                ParticleSurfaceSample sample;
                const uint32_t seed =
                    0x9e3779b9u ^ (sample_index * 2246822519u);
                if (flow_source_surface_sampler_(source, seed, sample)) {
                    gas_surface_samples.push_back(sample.position);
                }
            }
            if (gas_surface_samples.empty()) continue;
            source_min = source_min - Vec3(source_radius);
            source_max = source_max + Vec3(source_radius);
        }

        const Vec3 mn = state.bounds_min;
        const Vec3 mx = state.bounds_max;
        if (source_max.x < mn.x || source_min.x > mx.x ||
            source_max.y < mn.y || source_min.y > mx.y ||
            source_max.z < mn.z || source_min.z > mx.z) {
            continue;
        }

        FluidSim::FluidGrid& grid = state.grid;
        if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
            continue;
        }

        // Cell range overlapping the source sphere (grid space).
        float fi0, fj0, fk0, fi1, fj1, fk1;
        grid.worldToGrid(source_min, fi0, fj0, fk0);
        grid.worldToGrid(source_max, fi1, fj1, fk1);
        const int min_x = std::clamp(static_cast<int>(std::floor(fi0)), 0, grid.nx - 1);
        const int max_x = std::clamp(static_cast<int>(std::ceil(fi1)), 0, grid.nx - 1);
        const int min_y = std::clamp(static_cast<int>(std::floor(fj0)), 0, grid.ny - 1);
        const int max_y = std::clamp(static_cast<int>(std::ceil(fj1)), 0, grid.ny - 1);
        const int min_z = std::clamp(static_cast<int>(std::floor(fk0)), 0, grid.nz - 1);
        const int max_z = std::clamp(static_cast<int>(std::ceil(fk1)), 0, grid.nz - 1);

        const float inv_radius = 1.0f / source_radius;
        const float falloff = std::max(0.0f, keyed.falloff);
        const int range_nx = max_x - min_x + 1;
        const int range_ny = max_y - min_y + 1;
        std::vector<float> gas_surface_weights;
        if (source.source_mode == SimulationFlowSourceMode::MeshSurface) {
            const int range_nz = max_z - min_z + 1;
            gas_surface_weights.assign(
                static_cast<std::size_t>(range_nx) *
                    static_cast<std::size_t>(range_ny) *
                    static_cast<std::size_t>(range_nz),
                0.0f);
            const int cell_radius = std::max(
                1, static_cast<int>(std::ceil(source_radius /
                                              std::max(grid.voxel_size, 1e-6f))));
            for (const Vec3& sample : gas_surface_samples) {
                float sample_x, sample_y, sample_z;
                grid.worldToGrid(sample, sample_x, sample_y, sample_z);
                const int center_x = static_cast<int>(std::floor(sample_x));
                const int center_y = static_cast<int>(std::floor(sample_y));
                const int center_z = static_cast<int>(std::floor(sample_z));
                for (int z = std::max(min_z, center_z - cell_radius);
                     z <= std::min(max_z, center_z + cell_radius);
                     ++z) {
                    for (int y = std::max(min_y, center_y - cell_radius);
                         y <= std::min(max_y, center_y + cell_radius);
                         ++y) {
                        for (int x = std::max(min_x, center_x - cell_radius);
                             x <= std::min(max_x, center_x + cell_radius);
                             ++x) {
                            const float distance =
                                (grid.gridToWorld(x, y, z) - sample).length();
                            if (distance > source_radius) continue;
                            const float normalized = distance * inv_radius;
                            const float shell_weight = falloff <= 0.0f
                                ? 1.0f
                                : std::pow(std::max(0.0f, 1.0f - normalized),
                                           falloff);
                            const std::size_t local =
                                static_cast<std::size_t>(x - min_x) +
                                static_cast<std::size_t>(y - min_y) *
                                    static_cast<std::size_t>(range_nx) +
                                static_cast<std::size_t>(z - min_z) *
                                    static_cast<std::size_t>(range_nx) *
                                    static_cast<std::size_t>(range_ny);
                            gas_surface_weights[local] =
                                std::max(gas_surface_weights[local], shell_weight);
                        }
                    }
                }
            }
        }
        const bool write_density = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Density);
        const bool write_temperature = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Temperature);
        const bool write_fuel = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Fuel);
        const bool write_pressure = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Pressure);
        const bool write_velocity = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Velocity);
        const float density_amount = keyed.density * time_scale;
        const float temperature_amount = keyed.temperature * time_scale;
        const float fuel_amount = keyed.fuel * time_scale;
        const float velocity_blend =
            1.0f - std::exp(-std::max(0.0f, keyed.velocity_coupling) * time_scale);

        // Keep the small authored-source deposit on the host for now. The gas
        // solve uploads these freshly injected scalar/velocity fields below and
        // then performs advection, combustion, forces and projection on Vulkan.
        // The former direct-GPU shortcut returned here, after which the normal
        // host publication uploaded the still-empty CPU arrays over the same
        // buffers and silently erased point sources (notably the pilot flame).
        // A future fully-resident source path must carry residency through the
        // velocity and scalar upload gates; dispatching only this isolated stage
        // is incorrect.

        auto sourceWeightAt = [&](const Vec3& sample_position) {
            float normalized_distance = 0.0f;
            if (source.source_mode == SimulationFlowSourceMode::ObjectBounds) {
                const Vec3 q = sample_position - source_center;
                normalized_distance = std::max({
                    std::abs(q.x) / std::max(source_half_extent.x, 1e-6f),
                    std::abs(q.y) / std::max(source_half_extent.y, 1e-6f),
                    std::abs(q.z) / std::max(source_half_extent.z, 1e-6f)});
            } else if (source.source_mode == SimulationFlowSourceMode::MeshSurface) {
                float nearest_sq = std::numeric_limits<float>::max();
                for (const Vec3& surface_sample : gas_surface_samples) {
                    const Vec3 delta = sample_position - surface_sample;
                    nearest_sq = std::min(
                        nearest_sq,
                        delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
                }
                normalized_distance = std::sqrt(nearest_sq) * inv_radius;
            } else {
                normalized_distance =
                    (sample_position - source_center).length() * inv_radius;
            }
            if (normalized_distance > 1.0f) return 0.0f;
            return falloff <= 0.0f
                ? 1.0f
                : std::pow(std::max(0.0f, 1.0f - normalized_distance), falloff);
        };

        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = min_z; z <= max_z; ++z) {
            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    const Vec3 cell_center = grid.gridToWorld(x, y, z);
                    float weight = 0.0f;
                    if (source.source_mode == SimulationFlowSourceMode::MeshSurface) {
                        const std::size_t local =
                            static_cast<std::size_t>(x - min_x) +
                            static_cast<std::size_t>(y - min_y) *
                                static_cast<std::size_t>(range_nx) +
                            static_cast<std::size_t>(z - min_z) *
                                static_cast<std::size_t>(range_nx) *
                                static_cast<std::size_t>(range_ny);
                        const float shell_weight = gas_surface_weights[local];
                        if (shell_weight <= 0.0f) continue;
                        weight = shell_weight;
                    } else {
                        weight = sourceWeightAt(cell_center);
                    }
                    if (weight <= 0.0f) continue;
                    const std::size_t cell = grid.cellIndex(x, y, z);

                    if (write_density) {
                        grid.density[cell] += density_amount * weight;
                    }
                    if (write_temperature) {
                        grid.temperature[cell] += temperature_amount * weight;
                    }
                    if (write_fuel) {
                        grid.fuel[cell] += fuel_amount * weight;
                    }
                    if (write_pressure) {
                        grid.pressure[cell] += density_amount * 0.2f * weight;
                    }
                }
            }
        }
        if (write_velocity && velocity_blend > 0.0f) {
            // Each loop owns one MAC face. This avoids the old cell-parallel
            // overlapping writes and relaxes toward an inflow velocity instead
            // of adding kinetic energy forever.
            #pragma omp parallel for collapse(2) schedule(static)
            for (int z = min_z; z <= max_z; ++z)
                for (int y = min_y; y <= max_y; ++y)
                    for (int x = min_x; x <= std::min(grid.nx, max_x + 1); ++x) {
                        const Vec3 p = grid.origin +
                            Vec3(static_cast<float>(x),
                                 static_cast<float>(y) + 0.5f,
                                 static_cast<float>(z) + 0.5f) * grid.voxel_size;
                        const float blend =
                            std::clamp(velocity_blend * sourceWeightAt(p), 0.0f, 1.0f);
                        float& value = grid.velXAt(x, y, z);
                        value += (resolved.velocity.x - value) * blend;
                    }
            #pragma omp parallel for collapse(2) schedule(static)
            for (int z = min_z; z <= max_z; ++z)
                for (int y = min_y; y <= std::min(grid.ny, max_y + 1); ++y)
                    for (int x = min_x; x <= max_x; ++x) {
                        const Vec3 p = grid.origin +
                            Vec3(static_cast<float>(x) + 0.5f,
                                 static_cast<float>(y),
                                 static_cast<float>(z) + 0.5f) * grid.voxel_size;
                        const float blend =
                            std::clamp(velocity_blend * sourceWeightAt(p), 0.0f, 1.0f);
                        float& value = grid.velYAt(x, y, z);
                        value += (resolved.velocity.y - value) * blend;
                    }
            #pragma omp parallel for collapse(2) schedule(static)
            for (int z = min_z; z <= std::min(grid.nz, max_z + 1); ++z)
                for (int y = min_y; y <= max_y; ++y)
                    for (int x = min_x; x <= max_x; ++x) {
                        const Vec3 p = grid.origin +
                            Vec3(static_cast<float>(x) + 0.5f,
                                 static_cast<float>(y) + 0.5f,
                                 static_cast<float>(z)) * grid.voxel_size;
                        const float blend =
                            std::clamp(velocity_blend * sourceWeightAt(p), 0.0f, 1.0f);
                        float& value = grid.velZAt(x, y, z);
                        value += (resolved.velocity.z - value) * blend;
                    }
        }
    }
}
