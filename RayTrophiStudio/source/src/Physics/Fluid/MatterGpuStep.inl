// Included after the existing GPU stages in the driver's private namespace.
// Particle sidecars/identities stay canonical; model indices select disjoint
// slots in persistent GPU mirrors. There is one host publication per frame.

std::size_t matterGpuWorkingSetBytes(const FluidSim::FluidGrid& grid,
    const Fluid::APICSolverParams& params, std::size_t count) {
    const auto cells = grid.getCellCount();
    const auto faces = grid.vel_x.size() + grid.vel_y.size() + grid.vel_z.size();
    const auto pore = params.pore_exchange.enabled ? (cells + 1) * 16 + count * 2048 : 0;
    return pore + cells * 512 + faces * 256 + count * 2048 +
        count * (sizeof(Fluid::MatterWetResponse) + sizeof(float)) * 2;
}

bool runMatterGpuStep(SimulationGridDomainState& state,
    const Fluid::APICSolverParams& params, bool legacy_granular, float frame_dt,
    SimulationComputeContext* compute, SimulationGridDomainComputeBuffers& primary,
    const std::function<bool(SimulationGridDomainComputeBuffers&)>& ensure,
    MatterExchangeLedger& ledger,
    std::string& error, const Fluid::MatterCommonClockHooks* clock = nullptr) {
    error.clear();
    primary.sparse_mac_transfer.used = false;
    primary.sparse_mac_transfer.snapshot_valid = false;
    if (!compute || !compute->supportsDispatch() ||
        compute->backendType() != ComputeBackendType::VulkanCompute) {
        error = "mixed Matter requires Vulkan compute; no automatic CPU fallback";
        return false;
    }
    if (params.boundary == Fluid::APICSolverParams::BoundaryMode::Periodic) {
        error = "mixed GPU Periodic transport is not supported by the current kernels";
        return false;
    }
    if ((params.pore_exchange.enabled || params.pore_exchange.wet_response_enabled) &&
        params.boundary != Fluid::APICSolverParams::BoundaryMode::Closed) {
        error = "C5 first acceptance requires Closed Vulkan Matter";
        return false;
    }
    if (!params.external_forces_preintegrated) {
        error = "mixed GPU requires a successful canonical force-integration stage";
        return false;
    }
    auto& particles = state.particles;
    auto& grid = state.grid;
    const auto count = particles.size();
    const auto cells = grid.getCellCount();
    const std::array<std::size_t, 3> faces{
        grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size()};
    const auto lattice = static_cast<std::size_t>(grid.nx + 1) * (grid.ny + 1) * (grid.nz + 1);
    const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (*std::max_element(faces.begin(), faces.end()) >
            std::numeric_limits<uint32_t>::max() / 3u ||
        !count || count > limit || !cells || cells > limit || lattice > limit ||
        !std::isfinite(frame_dt) || frame_dt <= 0.0f ||
        particles.rest_mass_kg.size() != count || particles.mass_fraction.size() != count ||
        particles.velocity.size() != count || particles.affine.size() != count ||
        particles.particle_id.size() != count || particles.constitutive_model.size() != count ||
        particles.pore_water_mass_kg.size() != count ||
        particles.pore_water_energy_j.size() != count) {
        error = "mixed GPU layout/time/particle cardinality is invalid";
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto finite = [](const Vec3& value) {
            return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
        };
        const auto& affine = particles.affine[i];
        if (!finite(particles.position[i]) || !finite(particles.velocity[i]) ||
            !finite(affine.col0) || !finite(affine.col1) || !finite(affine.col2) ||
            particles.particle_id[i] == 0 ||
            particles.particle_id[i] >= particles.next_particle_id) {
            error = "mixed GPU received invalid canonical kinematics/identity";
            return false;
        }
        const auto model = i < particles.constitutive_model.size()
            ? static_cast<Fluid::MatterConstitutiveModel>(particles.constitutive_model[i])
            : Fluid::MatterConstitutiveModel::Auto;
        const bool granular = model == Fluid::MatterConstitutiveModel::Granular ||
            (model == Fluid::MatterConstitutiveModel::Auto && legacy_granular);
        if (!std::isfinite(particles.pore_water_mass_kg[i]) ||
            particles.pore_water_mass_kg[i] < 0.0f ||
            !std::isfinite(particles.pore_water_energy_j[i]) ||
            particles.pore_water_energy_j[i] < 0.0f ||
            (!granular && particles.pore_water_mass_kg[i] > 0.0f) ||
            (particles.pore_water_mass_kg[i] == 0.0f &&
                particles.pore_water_energy_j[i] != 0.0f)) {
            error = "mixed GPU received invalid pore mass/energy ownership";
            return false;
        }
        if (model == Fluid::MatterConstitutiveModel::Elastic ||
            Fluid::isMatterObstacle(particles, i,
                params.solid_substance_tags ? params.solid_substance_tags->data() : nullptr,
                params.solid_substance_tags ? params.solid_substance_tags->size() : 0u) ||
            !std::isfinite(particles.mass_fraction[i]) || particles.mass_fraction[i] < 0.0f ||
            particles.mass_fraction[i] > 1.0f ||
            !std::isfinite(particles.rest_mass_kg[i]) || particles.rest_mass_kg[i] <= 0.0f) {
            error = "mixed GPU requires mobile fluid/granular parcels; static blocks, "
                "frozen or elastic carriers need an accepting transport lane";
            return false;
        }
    }
    const bool has_fluid = Fluid::resolveSingleMatterModel(particles, legacy_granular)
        != Fluid::MatterConstitutiveModel::Granular;
    // The grain domain's liquid subset carries no granular parcel; it must not
    // pay the granular elastic wave/strain substep count, only the CFL.
    const bool has_granular = std::any_of(particles.constitutive_model.begin(),
        particles.constitutive_model.end(), [&](auto value) {
            const auto model = static_cast<Fluid::MatterConstitutiveModel>(value);
            return model == Fluid::MatterConstitutiveModel::Granular ||
                (model == Fluid::MatterConstitutiveModel::Auto && legacy_granular);
        });
    // Conservative bound covers both model grids, pressure scratch, gradients,
    // GPU particle mirrors and atomic host publication. No allocation precedes it.
    const std::size_t pore_working = params.pore_exchange.enabled
        ? (cells + 1) * 16 + count * 2048 : 0;
    const std::size_t working_set = matterGpuWorkingSetBytes(grid, params, count);
    if (params.mixed_working_set_budget_bytes &&
        working_set > params.mixed_working_set_budget_bytes) {
        error = "mixed GPU working set exceeds domain resource budget";
        return false;
    }
    const auto load = Fluid::measureMatterGranularLoad(particles, legacy_granular);
    const auto elastic = Fluid::Granular::elasticStepInfo(
        params.granular_young_modulus, grid.voxel_size, frame_dt, load.strain_rate,
        load.overburden_pressure, load.softening_min);
    float speed = 0.0f;
    for (const auto& velocity : particles.velocity) {
        speed = std::max(speed, velocity.length());
    }
    const double request = std::max(has_granular ? static_cast<double>(elastic.required_substeps) : 1.0,
        std::ceil(static_cast<double>(speed) * frame_dt /
            std::max(grid.voxel_size * params.cfl, 1e-6f)));
    uint32_t common_substeps = 0;
    if (!Fluid::resolveMatterCommonSubsteps(request, clock, common_substeps, error)) {
        return false;
    }
    const int substeps = static_cast<int>(common_substeps);
    const float dt = frame_dt / static_cast<float>(substeps);
    // Grid transfers obey their own CFL/elastic step, while geometry and contact
    // advance on the common micro clock. Do not solve pressure at DEM frequency.
    const int grid_stride = clock
        ? std::max(1, substeps / std::max(1, static_cast<int>(std::ceil(request)))) : 1;
    const int grid_steps = 1 + (substeps - 1) / grid_stride;
    // Sparse S1 (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md): the liquid lane keeps
    // its device MAC velocity and FLIP baseline on compact tile pages only, so
    // the allocator gives it no dense bank. Decided before allocation and kept
    // across frames; a failed compact P2G blocks it until sparse is toggled.
    auto& mac_storage = primary.sparse_mac_transfer;
    const bool compact_wanted = grid.sparse_mode_enabled && !grid.allocate_gas_channels;
    if (!compact_wanted) {
        mac_storage.compact_blocked = false;
    }
    mac_storage.compact_owner = compact_wanted && has_fluid && !mac_storage.compact_blocked;
    if (!ensure(primary)) {
        error = "mixed liquid GPU buffers could not be allocated";
        return false;
    }
    if (!primary.matter_runtime) {
        primary.matter_runtime = std::make_shared<Fluid::MatterGpuRuntime>();
    }
    auto& runtime = *primary.matter_runtime;
    if (!ensure(runtime.granular)) {
        error = "mixed granular GPU buffers could not be allocated";
        return false;
    }
    const auto legacy = legacy_granular ? Fluid::MatterConstitutiveModel::Granular
                                       : Fluid::MatterConstitutiveModel::Fluid;
    if (!Fluid::ensureMatterGpuPartition(*compute, runtime.partition, count,
            params.mixed_working_set_budget_bytes, error) ||
        !Fluid::dispatchMatterGpuPartition(*compute, runtime.partition, particles, legacy, error)) {
        return false;
    }
    const auto usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite |
        ComputeBufferUsage::Upload | ComputeBufferUsage::Download;
    auto allocation = [&](ComputeBufferHandle& handle, std::size_t bytes, const char* name) {
        if (handle.valid() && compute->getBufferSize(handle) >= bytes) {
            return true;
        }
        if (handle.valid()) {
            compute->destroyBuffer(handle);
        }
        ComputeBufferDesc desc;
        desc.debug_name = name;
        desc.size_bytes = bytes;
        desc.usage = usage;
        handle = compute->createBuffer(desc);
        return handle.valid();
    };
    if (!allocation(runtime.rest_mass, count * sizeof(float), "matter_rest_mass") ||
        !allocation(runtime.transport_fraction, count * sizeof(float), "matter_transport_fraction") ||
        !allocation(runtime.contact_pairs, sizeof(uint32_t), "matter_contact_pairs")) {
        error = "mixed GPU mass/contact allocation failed";
        return false;
    }
    std::vector<Fluid::MatterWetResponse> wet_responses;
    if (!Fluid::buildMatterWetResponses(particles, legacy_granular, grid.voxel_size,
            params.pore_exchange, wet_responses, error) ||
        !allocation(runtime.wet_response, count * sizeof(Fluid::MatterWetResponse),
            "matter_wet_response") ||
        !compute->uploadBuffer(runtime.wet_response, wet_responses.data(),
            count * sizeof(Fluid::MatterWetResponse))) {
        error = error.empty() ? "C6 wet response preparation/upload failed" : error;
        return false;
    }
    std::vector<float> dry_volumes(count);
    for (std::size_t i = 0; i < count; ++i) {
        dry_volumes[i] = Fluid::matterParticleDryVolume(particles, i);
        if (!std::isfinite(dry_volumes[i]) || dry_volumes[i] < 0.0f) {
            error = "C6 canonical dry stress volume is invalid";
            return false;
        }
    }
    if (!allocation(runtime.dry_volume, count * sizeof(float), "matter_dry_stress_volume") ||
        !compute->uploadBuffer(runtime.dry_volume, dry_volumes.data(), count * sizeof(float))) {
        error = "C6 canonical dry stress volume upload failed";
        return false;
    }
    uint32_t zero = 0;
    std::vector<float> transport_rest, transport_fraction;
    Fluid::matterPoreTransportMasses(particles, legacy_granular, transport_rest, transport_fraction);
    if (!compute->uploadBuffer(runtime.rest_mass, transport_rest.data(),
            count * sizeof(float)) ||
        !compute->uploadBuffer(runtime.transport_fraction, transport_fraction.data(), count * sizeof(float)) ||
        !compute->uploadBuffer(runtime.contact_pairs, &zero, sizeof(zero))) {
        error = "mixed GPU mass/contact upload failed";
        return false;
    }
    SimulationGridDomainComputeBuffers* lanes[] = {&primary, &runtime.granular};
    // Reset view policy even on failure; subsequent single-model steps retain
    // their original ABI and never consume a partial mixed device result.
    struct ResetViews {
        SimulationGridDomainComputeBuffers** lanes;
        ~ResetViews() {
            for (int lane = 0; lane < 2; ++lane) {
                lanes[lane]->fluid_mask_device_valid = false;
                lanes[lane]->matter_model = {};
                lanes[lane]->granular.matter_model = {};
                FluidGpuParticleUpload::invalidate(*lanes[lane]);
            }
        }
    } reset{lanes};
    std::array<Fluid::APICSolverParams, 2> models{params, params};
    models[0].granular_enabled = false;
    models[0].ghost_fluid_surface = false;
    models[1].granular_enabled = true;
    models[1].flip_blend = 0.0f;
    models[1].substance_viscosity = nullptr;
    models[1].kinematic_viscosity = 0.0f;
    Fluid::MatterGpuParticleLease shared_particles;
    for (int lane = 0; lane < 2; ++lane) {
        const float damping_dt = lane == 1 ? frame_dt
            : Fluid::Granular::kDampingReferenceDt;
        models[lane].velocity_damping = Fluid::Granular::timeScaledSubstepDamping(
            params.velocity_damping, damping_dt, grid_steps);
        models[lane].affine_damping = Fluid::Granular::timeScaledSubstepDamping(
            params.affine_damping, damping_dt, grid_steps);
        const bool transport_ready = lane == 0
            ? ensureGpuFluidParticleBuffers(state, compute, primary, false, false)
            : shared_particles.bind(*lanes[1], primary, count, error);
        if (!transport_ready || (lane == 1 &&
            !Fluid::Granular::uploadState(*compute, particles, lanes[lane]->granular))) {
            error = "mixed GPU canonical particle upload failed";
            return false;
        }
        auto& view = lanes[lane]->matter_model;
        view.enabled = true;
        view.lane = lane;
        view.boundary = params.boundary == Fluid::APICSolverParams::BoundaryMode::Open ? 0u : 1u;
        view.indices = runtime.partition.indices[lane];
        view.counters = runtime.partition.counters;
        view.rest_mass = runtime.rest_mass;
        view.mass_fraction = runtime.transport_fraction;
        view.wet_response = runtime.wet_response;
        view.dry_volume = runtime.dry_volume;
        for (int axis = 0; axis < 3; ++axis) {
            if (!allocation(runtime.gradient[lane][axis], faces[axis] * 3 * sizeof(float),
                            "matter_face_mass_gradient")) {
                error = "mixed GPU gradient allocation failed";
                return false;
            }
            view.mass_gradient[axis] = runtime.gradient[lane][axis];
        }
        lanes[lane]->granular.matter_model = view;
    }
    std::array<std::vector<float>, 3> solid_velocity;
    for (auto& axis : solid_velocity) {
        axis.assign(cells, 0.0f);
    }
    for (std::size_t i = 0; i < cells && i < grid.solid_vel.size(); ++i) {
        solid_velocity[0][i] = grid.solid_vel[i].x;
        solid_velocity[1][i] = grid.solid_vel[i].y;
        solid_velocity[2][i] = grid.solid_vel[i].z;
    }
    for (const auto lane : lanes) {
        const ComputeBufferHandle solid_fields[] = {lane->var_svx, lane->var_svy, lane->var_svz};
        for (int axis = 0; axis < 3; ++axis) {
            if (!compute->uploadBuffer(solid_fields[axis], solid_velocity[axis].data(),
                                       cells * sizeof(float))) {
                error = "mixed GPU solid velocity upload failed";
                return false;
            }
        }
    }
    FluidGpuOccupancy liquid_occupancy(state, compute, lanes[0]);
    FluidGpuOccupancy granular_occupancy(state, compute, lanes[1]);
    Fluid::APICSolverStats stats;
    std::vector<float> unused_host_mask;
    // Dense bank only (empty for a compact owner until a fallback allocates it).
    ComputeBufferHandle post[] = {primary.vel_x, primary.vel_y, primary.vel_z};
    ComputeBufferHandle pre[] = {
        primary.scratch_vel_x, primary.scratch_vel_y, primary.scratch_vel_z};
    if (!has_fluid) {
        // Preserve the empty liquid field's publication without solving an empty
        // pressure system. These persistent buffers may contain last frame's water.
        for (int axis = 0; axis < 3; ++axis) {
            const uint32_t values = static_cast<uint32_t>(faces[axis]);
            ComputeDispatch clear;
            clear.kernel = "sim_matter_clear";
            clear.buffers = &post[axis];
            clear.buffer_count = 1;
            clear.constants = &values;
            clear.constants_size = sizeof(values);
            clear.groups = FluidGpuDispatch::groups256(values);
            if (!compute->dispatch(clear)) {
                error = "mixed GPU empty fluid field clear failed";
                return false;
            }
        }
    }
    if (clock && clock->begin && !clock->begin(working_set, error)) {
        return false;
    }
    for (int substep = 0; substep < substeps; ++substep) {
        if (clock && clock->forces &&
            !clock->forces(substep, common_substeps, dt, error)) {
            return false;
        }
        if (substep % grid_stride == 0) {
            const float transfer_dt = dt * std::min(grid_stride, substeps - substep);
            if ((has_fluid && !liquid_occupancy.build()) || !granular_occupancy.build()) {
                error = "mixed GPU model occupancy failed";
                return false;
            }
            primary.fluid_mask_device_valid = has_fluid;
            lanes[1]->fluid_mask_device_valid = true;
            for (int lane = 0; lane < 2; ++lane) {
                if (lane == 0 && !has_fluid) {
                    continue;
                }
                for (int axis = 0; axis < 3; ++axis) {
                    const uint32_t values = static_cast<uint32_t>(faces[axis] * 3);
                    ComputeDispatch clear;
                    clear.kernel = "sim_matter_clear";
                    clear.buffers = &runtime.gradient[lane][axis];
                    clear.buffer_count = 1;
                    clear.constants = &values;
                    clear.constants_size = sizeof(values);
                    clear.groups = FluidGpuDispatch::groups256(values);
                    if (!compute->dispatch(clear)) {
                        error = "mixed GPU gradient clear failed";
                        return false;
                    }
                }
                bool transferred = runGpuFluidP2G(state, compute, *lanes[lane], models[lane],
                                                  transfer_dt, false, true, false);
                if (!transferred && lane == 0 && mac_storage.compact_blocked &&
                    !mac_storage.compact_owner) {
                    transferred = ensure(primary);
                    post[0] = primary.vel_x;
                    post[1] = primary.vel_y;
                    post[2] = primary.vel_z;
                    pre[0] = primary.scratch_vel_x;
                    pre[1] = primary.scratch_vel_y;
                    pre[2] = primary.scratch_vel_z;
                    transferred = transferred &&
                        runGpuFluidP2G(state, compute, *lanes[lane], models[lane],
                                       transfer_dt, false, true, false);
                }
                if (!transferred ||
                    !runGpuFluidZeroSolidFaces(grid, compute, *lanes[lane])) {
                    error = "mixed GPU indexed P2G/boundary failed";
                    return false;
                }
            }
            const bool sparse_flip_captured = has_fluid &&
                Fluid::captureSparseMacFlip(*compute, primary);
            if (has_fluid && !sparse_flip_captured && mac_storage.canonical) {
                error = "mixed GPU compact FLIP baseline capture failed";
                return false;
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (sparse_flip_captured) {
                    break;
                }
                if (!has_fluid) {
                    break;
                }
                if (!Fluid::copyMatterGpuFloat(*compute, post[axis], pre[axis],
                                              static_cast<uint32_t>(faces[axis]))) {
                    error = "mixed GPU liquid FLIP snapshot failed";
                    return false;
                }
            }
            if (has_fluid && (models[0].kinematic_viscosity > 0.0f ||
                              models[0].substance_viscosity)) {
                int sweeps = 0;
                if (!runGpuFluidViscosity(state, models[0], transfer_dt, compute, primary,
                                         unused_host_mask, &sweeps, &stats)) {
                    error = "mixed GPU liquid viscosity failed";
                    return false;
                }
                stats.viscosity_sweeps_run += sweeps;
            }
            if (has_fluid && !runGpuFluidMGPCGPressure(state, models[0], transfer_dt, compute, primary,
                                                      unused_host_mask, &stats)) {
                error = "mixed GPU liquid MGPCG projection failed";
                return false;
            }
            // Liquid lane: dense bank, or its compact velocity and mass pages
            // plus the MAC tile map and list. The granular lane stays dense.
            const auto liquid_mac = Fluid::macVelocityBinding(primary);
            const ComputeBufferHandle contact_buffers[] = {
                liquid_mac.velocity[0], liquid_mac.velocity[1], liquid_mac.velocity[2],
                lanes[1]->vel_x, lanes[1]->vel_y, lanes[1]->vel_z,
                liquid_mac.weight[0], liquid_mac.weight[1], liquid_mac.weight[2],
                lanes[1]->temperature, lanes[1]->fuel, lanes[1]->scratch_scalar,
                runtime.gradient[0][0], runtime.gradient[0][1], runtime.gradient[0][2],
                runtime.gradient[1][0], runtime.gradient[1][1], runtime.gradient[1][2],
                runtime.contact_pairs, liquid_mac.map, liquid_mac.list};
            struct ContactConstants { int nx, ny, nz; float friction; } contact_constants{
                grid.nx, grid.ny, grid.nz,
                std::tan(std::clamp(params.granular_friction_angle_degrees, 0.0f, 80.0f) *
                         0.017453292519943295f)};
            static_assert(sizeof(ContactConstants) == 16);
            ComputeDispatch contact;
            contact.kernel = Fluid::macKernel(liquid_mac, "sim_matter_contact",
                                              "sim_sparse_mac_matter_contact");
            contact.buffers = contact_buffers;
            contact.buffer_count = liquid_mac.compact ? 21 : 19;
            contact.constants = &contact_constants;
            contact.constants_size = sizeof(contact_constants);
            contact.groups = FluidGpuDispatch::groups256(static_cast<uint32_t>(lattice));
            if (has_fluid && !compute->dispatch(contact)) {
                error = "mixed GPU contact failed";
                return false;
            }
            // Every gather observes the same post-contact substep. Each lane writes
            // only its model's canonical indices; no liquid pressure on granular.
            for (int lane = 0; lane < 2; ++lane) {
                if (lane == 0 && !has_fluid) {
                    continue;
                }
                if (!runGpuFluidZeroSolidFaces(grid, compute, *lanes[lane]) ||
                    !runGpuFluidG2P(state, models[lane], transfer_dt, compute, *lanes[lane],
                        lane == 0, false, true, true, true)) {
                    error = "mixed GPU indexed gather/constitutive failed";
                    return false;
                }
            }
        }
        // All owners see current gathered velocities. Contact reactions land
        // before either continuum lane advects, and feed the next P2G directly.
        if (clock && clock->contact &&
            !clock->contact(substep, common_substeps, dt, error)) {
            return false;
        }
        for (int lane = 0; lane < 2; ++lane) {
            if (lane == 0 && !has_fluid) {
                continue;
            }
            auto advection = models[lane];
            if (clock) {
                const auto damping_dt = lane == 1 ? frame_dt : Fluid::Granular::kDampingReferenceDt;
                advection.velocity_damping = Fluid::Granular::timeScaledSubstepDamping(
                    params.velocity_damping, damping_dt, substeps);
            }
            if (!runGpuFluidAdvectTail(state, advection, dt, compute, *lanes[lane],
                    nullptr, false, nullptr, false, nullptr, false, clock != nullptr)) {
                error = "mixed GPU indexed advection after contact failed";
                return false;
            }
        }
    }
    // Publish only after the entire common subcycle succeeded. Metadata sidecars
    // are copied with the same canonical index; identity/allocator never rebirth.
    auto liquid_result = particles;
    compute->beginTransferBatch();
    // Both indexed writers used these same canonical streams. Publish once,
    // including the granular-only case; sidecars retain model-specific owners.
    bool ok = compute->downloadBuffer(primary.fluid_positions,
        liquid_result.position.data(), count * sizeof(Vec3));
    ok = compute->downloadBuffer(primary.fluid_velocities,
        liquid_result.velocity.data(), count * sizeof(Vec3)) && ok;
    ok = compute->downloadBuffer(primary.fluid_affine,
        liquid_result.affine.data(), count * sizeof(Fluid::AffineC)) && ok;
    uint32_t pairs = 0;
    ok = compute->downloadBuffer(runtime.contact_pairs, &pairs, sizeof(pairs)) && ok;
    std::array<std::vector<float>, 3> result_velocity{
        grid.vel_x, grid.vel_y, grid.vel_z};
    const bool compact_publication = has_fluid && mac_storage.used && mac_storage.canonical;
    for (int axis = 0; axis < 3 && !compact_publication; ++axis) {
        ok = compute->downloadBuffer(post[axis], result_velocity[axis].data(),
                                    faces[axis] * sizeof(float)) && ok;
    }
    ok = compute->endTransferBatch() && ok;
    if (ok && compact_publication) {
        std::string publication_error;
        ok = Fluid::publishCompactMacToHost(*compute, primary,
            {&result_velocity[0], &result_velocity[1], &result_velocity[2]},
            publication_error);
        if (!ok) {
            error = "mixed GPU " + publication_error;
            return false;
        }
    }
    auto granular_result = liquid_result;
    ok = downloadGpuGranularParticles(granular_result, compute, *lanes[1], false) && ok;
    if (!ok) {
        error = "mixed GPU atomic frame publication failed";
        return false;
    }
    for (const auto& axis : result_velocity) {
        if (!std::all_of(axis.begin(), axis.end(), [](float value) {
                return std::isfinite(value);
            })) {
            error = "mixed GPU produced nonfinite MAC velocity";
            return false;
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto model = i < particles.constitutive_model.size()
            ? static_cast<Fluid::MatterConstitutiveModel>(particles.constitutive_model[i])
            : Fluid::MatterConstitutiveModel::Auto;
        if (model == Fluid::MatterConstitutiveModel::Granular ||
            (model == Fluid::MatterConstitutiveModel::Auto && legacy_granular)) {
            liquid_result.copyParticleFrom(i, granular_result, i);
        }
        const auto& p = liquid_result.position[i];
        const auto& v = liquid_result.velocity[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            error = "mixed GPU produced nonfinite canonical particle state";
            return false;
        }
    }
    std::vector<MatterExchangeRecord> pore_events;
    if (params.pore_exchange.enabled) {
        const std::size_t remaining_budget = params.mixed_working_set_budget_bytes
            ? params.mixed_working_set_budget_bytes - working_set + pore_working : 0;
        // Equality means no free budget, not an unlimited budget.
        if ((params.mixed_working_set_budget_bytes && !remaining_budget) ||
            !Fluid::exchangeMatterPoresGpu(liquid_result, grid, params.pore_exchange, frame_dt,
                params.max_particles, remaining_budget, *compute, stats.pore_exchange,
                pore_events, error)) {
            stats.pore_exchange.held = true;
            stats.pore_exchange.status = error.empty() ? "C5 has no remaining budget" : error;
            state.fluid_stats.pore_exchange = stats.pore_exchange;
            error = stats.pore_exchange.status;
            return false;
        }
    }
    if (params.boundary == Fluid::APICSolverParams::BoundaryMode::Open) {
        Vec3 minimum, maximum;
        grid.getWorldBounds(minimum, maximum);
        for (std::size_t i = liquid_result.size(); i-- > 0;) {
            const auto& p = liquid_result.position[i];
            if (p.x < minimum.x || p.y < minimum.y || p.z < minimum.z ||
                p.x > maximum.x || p.y > maximum.y || p.z > maximum.z) {
                liquid_result.removeSwap(i);
            }
        }
    }
    liquid_result.uvw_refresh_period = params.uvw_refresh_period;
    liquid_result.advanceMaterialCoordinates();
    particles = std::move(liquid_result);
    for (auto& event : pore_events) {
        ledger.record(std::move(event));
    }
    grid.vel_x = std::move(result_velocity[0]);
    grid.vel_y = std::move(result_velocity[1]);
    grid.vel_z = std::move(result_velocity[2]);
    stats.grid_cell_count = cells;
    stats.granular_solver_substeps = grid_steps;
    stats.granular_required_substeps = elastic.required_substeps;
    stats.granular_requested_young_modulus = params.granular_young_modulus;
    stats.granular_effective_young_modulus = params.granular_young_modulus;
    stats.granular_wave_substeps = elastic.wave_substeps;
    stats.granular_strain_substeps = elastic.strain_substeps;
    stats.granular_strain_rate = elastic.strain_rate;
    stats.granular_overburden_pressure = elastic.overburden_pressure;
    stats.granular_young_modulus_for_load = elastic.young_modulus_for_load;
    stats.granular_stiffness_below_load = elastic.below_load;
    stats.granular_min_softening = load.softening_min;
    stats.granular_softened_particles = load.softened_particles;
    stats.granular_load_measured = true;
    stats.particle_count = particles.size();
    stats.mixed_model_step = true;
    stats.mixed_common_substeps = substeps;
    stats.mixed_contact_pairs = pairs;
    stats.mixed_working_set_bytes = working_set;
    stats.p2g_on_gpu = stats.g2p_on_gpu = true;
    stats.pressure_on_gpu = has_fluid;
    stats.viscosity_on_gpu = stats.viscosity_sweeps_run > 0;
    stats.gpu_status = has_fluid
        ? "Mixed Vulkan: indexed transfer/contact/common substeps"
        : "Mixed Vulkan: granular-only indexed/common substeps";
    Fluid::publishSparseMacTransferStats(primary.sparse_mac_transfer, stats);
    state.fluid_stats = stats;
    return true;
}
