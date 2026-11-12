// Included after the existing GPU stages in the driver's private namespace.
// Particle sidecars/identities stay canonical; model indices select disjoint
// slots in persistent GPU mirrors. There is one host publication per frame.

// Gravity and packed force fields integrated on every common substep, not
// once per frame. A frame's whole g*dt (0.41 m/s at 24 fps) landing in one
// kick hammers a resting MPM body into the floor every frame; the elastic
// response rings at a constant amplitude of ~0.4 g*frame_dt (2026-11-02:
// a static Dough read 0.168 / 0.077 / 0.044 m/s at frame dt 1/24, 1/48, 1/96
// with the same substep dt, while its shape held within 1 mm).
struct MatterSubstepForces {
    const SimulationForceFieldComputeBuffer* buffer = nullptr;
    float frame_start_seconds = 0.0f;
};

std::size_t matterGpuWorkingSetBytes(const FluidSim::FluidGrid& grid,
    const Fluid::APICSolverParams& params, std::size_t count) {
    const auto cells = grid.getCellCount();
    const auto faces = grid.vel_x.size() + grid.vel_y.size() + grid.vel_z.size();
    const auto pore = params.pore_exchange.enabled ? (cells + 1) * 16 + count * 2048 : 0;
    // Per-parcel term calibrated against measured VRAM growth (RTX 3060,
    // nvidia-smi, 2026-11-11): 1M snow parcels + 202k cells = 567 MB, of which
    // the cell/face terms below take ~259 MB -> ~336 B per parcel (mixed
    // snow+water ~200 B). It was 2048 B, a 4x overestimate that refused a 1M
    // snow step under the default 1 GiB budget. 384 B keeps ~15 % margin.
    // The pore exchange term is NOT calibrated yet.
    return pore + cells * 512 + faces * 256 + count * 384 +
        count * (sizeof(Fluid::MatterWetResponse) + sizeof(float)) * 2;
}

static bool runMatterGpuStepOnce(SimulationGridDomainState& state,
    const Fluid::APICSolverParams& params, bool legacy_granular, float frame_dt,
    SimulationComputeContext* compute, SimulationGridDomainComputeBuffers& primary,
    const std::function<bool(SimulationGridDomainComputeBuffers&)>& ensure,
    MatterExchangeLedger& ledger,
    std::string& error, const Fluid::MatterCommonClockHooks* clock,
    const MatterSubstepForces* substep_forces, bool snow_size_at_ceiling,
    bool& retry_at_ceiling) {
    retry_at_ceiling = false;
    error.clear();
    primary.sparse_mac_transfer.used = false;
    primary.sparse_mac_transfer.snapshot_valid = false;
    primary.sparse_mac_transfer.solid_weight_source_valid = false;
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
    Fluid::APICSolverStats stats;
    Fluid::MixedGpuStageProbe stage_probe(compute, stats.mixed_stages);
    using Stage = Fluid::MixedGpuStage;
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
    std::vector<uint8_t> held_parcels;  // frozen carriers pinned outside both lanes
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
        // ★ Frozen carriers / static blocks (2026-11-11): held in NO lane and
        // pinned where they are, like the grain domain's held ice. They used to
        // refuse the whole mixed step, so one frozen water parcel next to snow
        // stopped the simulation (honey/snow scene with thermal on). The
        // thermal pass may melt them; they rejoin their lane next frame.
        if (Fluid::isMatterObstacle(particles, i,
                params.solid_substance_tags ? params.solid_substance_tags->data() : nullptr,
                params.solid_substance_tags ? params.solid_substance_tags->size() : 0u)) {
            held_parcels.resize(count, 0u);
            held_parcels[i] = 1u;
            continue;
        }
        if (!std::isfinite(particles.mass_fraction[i]) || particles.mass_fraction[i] < 0.0f ||
            particles.mass_fraction[i] > 1.0f ||
            !std::isfinite(particles.rest_mass_kg[i]) || particles.rest_mass_kg[i] <= 0.0f) {
            error = "mixed GPU requires mobile fluid/granular/elastic parcels; static "
                "blocks and frozen carriers need an accepting transport lane";
            return false;
        }
    }
    // has_granular = an MPM continuum parcel (granular or elastic). The grain
    // domain's liquid subset carries none; it must not pay the granular
    // elastic wave/strain substep count, only the CFL.
    const auto mpm_parcel = [&](Fluid::MatterConstitutiveModel model) {
        return model == Fluid::MatterConstitutiveModel::Granular ||
            model == Fluid::MatterConstitutiveModel::Elastic ||
            (model == Fluid::MatterConstitutiveModel::Auto && legacy_granular);
    };
    const bool has_fluid = std::any_of(particles.constitutive_model.begin(),
        particles.constitutive_model.end(), [&](auto value) {
            return !mpm_parcel(static_cast<Fluid::MatterConstitutiveModel>(value));
        }) || (particles.constitutive_model.size() < particles.size() && !legacy_granular);
    const bool has_granular = std::any_of(particles.constitutive_model.begin(),
        particles.constitutive_model.end(), [&](auto value) {
            return mpm_parcel(static_cast<Fluid::MatterConstitutiveModel>(value));
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
    const bool snow = has_granular && params.granular_snow_plasticity;
    // ★ Snow sizing (2026-11-11). It was always sized for the hardening
    // CEILING (x5 -> sqrt(5) = 2.24x the substeps of virgin snow), because
    // sizing for last frame's hardest parcel was one frame late at impact and
    // over-compacted the pack (snowSizingYoungModulus). Now: size for the
    // hardest parcel MEASURED at the frame start x kSnowSizingMargin, and if
    // any parcel ends the frame harder than that (or hit the kernel's wave
    // cap), the whole frame is discarded and re-run at the ceiling. The host
    // state is untouched until publication, so the re-run starts from the same
    // state, and an impact frame gets exactly the old answer. Measured spread:
    // resting snow J_p = 1 (h = 1); water on snow h_max ~2.2; a 30 cm drop
    // reaches the ceiling. Hardening is irreversible in compression, so after
    // an impact the measured maximum stays high and frames stop retrying.
    // Not with the shared DEM clock: the DEM advances its own state inside
    // the frame and cannot be re-run.
    constexpr float kSnowSizingMargin = 1.5f;
    const auto snowHardening = [&](float plastic_volume) {
        const float jp = plastic_volume > 1.0e-6f ? 1.0f / plastic_volume : 1.0f;
        const float h = std::exp(std::clamp(params.granular_snow_hardening * (1.0f - jp), -80.0f, 80.0f));
        return std::clamp(h, params.granular_snow_min_hardening,
                          std::max(params.granular_snow_max_hardening, 1.0f));
    };
    const auto maxSnowHardening = [&](const std::vector<float>& plastic_volume) {
        const auto n = static_cast<int64_t>(std::min(plastic_volume.size(), count));
        float h_max = 1.0f;
        for (int64_t i = 0; i < n; ++i) {
            const auto model = static_cast<std::size_t>(i) < particles.constitutive_model.size()
                ? static_cast<Fluid::MatterConstitutiveModel>(particles.constitutive_model[i])
                : Fluid::MatterConstitutiveModel::Auto;
            if (!mpm_parcel(model)) continue;
            h_max = std::max(h_max, snowHardening(plastic_volume[i]));
        }
        return h_max;
    };
    const bool snow_adaptive = snow && !clock && !snow_size_at_ceiling &&
        particles.granular_plastic_volume.size() >= count;
    const float snow_ceiling = std::max(params.granular_snow_max_hardening, 1.0f);
    const float snow_sizing_hardening = snow_adaptive
        ? std::min(snow_ceiling, maxSnowHardening(particles.granular_plastic_volume) * kSnowSizingMargin)
        : snow_ceiling;
    // An elastic solid is sized at its own density, like snow, with no
    // hardening ceiling (its modulus never grows).
    const bool elastic_solid = has_granular && params.elastic_solid;
    const auto elastic = elastic_solid
        ? Fluid::Granular::elasticStepInfo(
              params.granular_young_modulus * Fluid::Granular::kElasticWaveSizingFactor,
              grid.voxel_size, frame_dt, load.strain_rate,
              load.overburden_pressure, load.softening_min, params.granular_density)
        : snow
        ? Fluid::Granular::elasticStepInfo(
              Fluid::Granular::snowSizingYoungModulus(
                  params.granular_young_modulus, snow_sizing_hardening),
              grid.voxel_size, frame_dt, load.strain_rate, load.overburden_pressure,
              load.softening_min, params.granular_density)
        : Fluid::Granular::elasticStepInfo(
              params.granular_young_modulus, grid.voxel_size, frame_dt, load.strain_rate,
              load.overburden_pressure, load.softening_min);
    float speed = 0.0f;
    for (const auto& velocity : particles.velocity) {
        speed = std::max(speed, velocity.length());
    }
    // An elastic solid's damping is explicit too: size for it, so its cap in
    // the kernel stays a backstop rather than the normal case.
    const int viscous_substeps = elastic_solid
        ? Fluid::Granular::elasticViscousSubsteps(params.elastic_viscosity_pa_s,
              grid.voxel_size, frame_dt, params.granular_density) : 1;
    const double request = std::max({has_granular
            ? static_cast<double>(elastic.required_substeps) : 1.0,
        static_cast<double>(viscous_substeps),
        std::ceil(static_cast<double>(speed) * frame_dt /
            std::max(grid.voxel_size * params.cfl, 1e-6f))});
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
    // The liquid on its own clock (docs/dev/MATTER_COKLU_ZAMAN_ADIMI.md). Snow +
    // water ran the liquid's pressure and P2G on every one of 180 elastic
    // substeps: 1.6 s of a 1.75 s frame, while the liquid's CFL asked for ~1.
    // The liquid lane transfers and projects every liquid_stride substeps (a
    // multiple of grid_stride, sized by the CFL); the MPM lane keeps every grid
    // step and exchanges momentum with the liquid's MAC field (pore drag +
    // pressure push, MATTER_SIVI_MPM_BASINC_BAGLASIMI.md); that exchange
    // reaches the liquid parcels by a pure-delta G2P at the end of each liquid
    // interval. The synchronous A/B path (contact impulse every substep) was
    // removed 2026-11-11; the reference is this path at a fine liquid step.
    const bool liquid_multirate = has_fluid && has_granular;
    int liquid_stride = grid_stride;
    if (liquid_multirate) {
        const double liquid_request = std::max(1.0, std::ceil(static_cast<double>(speed) *
            frame_dt / std::max(grid.voxel_size * params.cfl, 1e-6f)));
        const int wanted = std::max(1, static_cast<int>(substeps / liquid_request));
        liquid_stride = std::max(grid_stride, (wanted / grid_stride) * grid_stride);
    }
    const int liquid_steps = has_fluid ? 1 + (substeps - 1) / liquid_stride : 0;
    stats.mixed_stages.common_substeps = substeps;
    stats.mixed_stages.grid_steps = grid_steps;
    stats.mixed_stages.liquid_steps = liquid_steps;
    stats.mixed_stages.liquid_multirate = liquid_multirate;
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
    // Domain sleep (MatterMpmSleep.h): when every parcel rests, no substep runs.
    Fluid::MpmSleepResult mpm_sleep;
    Fluid::computeMpmSleep(particles, grid, params, legacy_granular,
        runtime.mpm_sleep, mpm_sleep);
    // Everything asleep and nothing else in the domain: no lane has work, so
    // no substep loop runs (its ~50 substeps of grid kernels cost the frame
    // with zero parcels in them). The parcels keep their state; gravity the
    // canonical force stage may have added comes back off.
    // Never under a force field (the DEM sleep rule): a skipped frame would
    // ignore it, and nothing in the rest test sees it. Known only when the
    // forces come per substep; preintegrated forces never sleep the domain.
    const bool no_force_field = substep_forces && !substep_forces->buffer;
    if (count > 0 && no_force_field && mpm_sleep.domain_asleep) {
        for (auto& v : particles.velocity) v = Vec3(0.0f);
        Fluid::APICSolverStats asleep_stats;
        // The state did not change, so its counters still hold: recount them
        // from the parcels. A blank block reported a resting pile as 0 asleep
        // and its plastic J as 1 (2026-11-04, snow_support floor_sleeps).
        // Timings stay 0: nothing ran.
        Fluid::Granular::countMaterialState(particles, asleep_stats);
        asleep_stats.particle_count = count;
        asleep_stats.mixed_model_step = true;
        asleep_stats.mixed_common_substeps = 0;
        asleep_stats.mpm_resting_particles = mpm_sleep.resting_particles;
        asleep_stats.gpu_status = "matter_asleep";
        state.fluid_stats = asleep_stats;
        return true;
    }
    if (!Fluid::ensureMatterGpuPartition(*compute, runtime.partition, count,
            params.mixed_working_set_budget_bytes, error) ||
        !Fluid::dispatchMatterGpuPartition(*compute, runtime.partition, particles, legacy, error,
            held_parcels.empty() ? nullptr : &held_parcels)) {
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
    // With the wet response off every entry is the neutral {1, 1, 0, 0}: the
    // buffer is filled once and kept while its handle and size hold (it was
    // rebuilt and re-uploaded every frame, 16 MB at 1M parcels).
    static thread_local struct { uint64_t buffer = 0; std::size_t count = 0; } wet_neutral;
    static thread_local std::vector<Fluid::MatterWetResponse> wet_responses;
    if (!allocation(runtime.wet_response, count * sizeof(Fluid::MatterWetResponse),
            "matter_wet_response")) {
        error = "C6 wet response allocation failed";
        return false;
    }
    const bool wet_neutral_valid = !params.pore_exchange.wet_response_enabled &&
        wet_neutral.buffer == runtime.wet_response.id && wet_neutral.count >= count;
    if (!wet_neutral_valid) {
        if (!Fluid::buildMatterWetResponses(particles, legacy_granular, grid.voxel_size,
                params.pore_exchange, wet_responses, error) ||
            !compute->uploadBuffer(runtime.wet_response, wet_responses.data(),
                count * sizeof(Fluid::MatterWetResponse))) {
            error = error.empty() ? "C6 wet response preparation/upload failed" : error;
            wet_neutral = {};
            return false;
        }
        wet_neutral = params.pore_exchange.wet_response_enabled
            ? decltype(wet_neutral){} : decltype(wet_neutral){runtime.wet_response.id, count};
    }
    // ★ thread_local scratch is reached in the OpenMP loops below ONLY through
    // pointers bound here: an OpenMP worker naming the thread_local gets its
    // own empty instance (null data -> access violation, 2026-11-11).
    static thread_local std::vector<float> dry_volume_scratch;
    dry_volume_scratch.resize(count);
    auto& dry_volumes = dry_volume_scratch;
    float* const dry_volume_out = dry_volumes.data();
    const auto* domain_substance = RayTrophiSim::tryFindSubstance(params.default_substance);
    {
        // Same value as matterParticleDryVolume, with the substance looked up
        // once per tag instead of once per parcel (1M lookups a frame).
        const bool tagged = particles.substance_tag.size() >= count;
        const bool sized = particles.rest_mass_kg.size() >= count &&
            particles.mass_fraction.size() >= count;
        // A domain carries a handful of substances: a linear table is enough.
        std::vector<std::pair<uint32_t, float>> density_of_tag;
        const auto find_tag = [&](uint32_t tag) -> const std::pair<uint32_t, float>* {
            for (const auto& entry : density_of_tag) {
                if (entry.first == tag) return &entry;
            }
            return nullptr;
        };
        const auto density_of = [&](const RayTrophiSim::SubstanceProfile* profile) {
            return profile && std::isfinite(profile->density) && profile->density > 0.0f
                ? profile->density : 1600.0f; // Untagged legacy stress used the Sand density.
        };
        if (tagged) {
            for (std::size_t i = 0; i < count; ++i) {
                const uint32_t tag = particles.substance_tag[i];
                if (!find_tag(tag)) {
                    const auto* profile = RayTrophiSim::tryFindSubstanceByTag(tag);
                    density_of_tag.emplace_back(tag, density_of(profile ? profile : domain_substance));
                }
            }
        }
        const float untagged_density = density_of(domain_substance);
        const auto signed_count = static_cast<int64_t>(count);
        int64_t invalid = sized ? 0 : 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : invalid) if(sized && signed_count > 65536)
#endif
        for (int64_t i = 0; i < (sized ? signed_count : 0); ++i) {
            const float density = tagged
                ? find_tag(particles.substance_tag[i])->second
                : untagged_density;
            const float mass = particles.rest_mass_kg[i] * particles.mass_fraction[i];
            const float volume = std::isfinite(mass) && mass >= 0.0f ? mass / density : -1.0f;
            dry_volume_out[i] = volume;
            if (!std::isfinite(volume) || volume < 0.0f) ++invalid;
        }
        if (invalid != 0) {
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
    // Drucker-Prager piles stay PIC/APIC (their rest state relies on it).
    // Snow takes its substance's FLIP blend; the baseline is captured inside
    // P2G before the stress impulse (granularFlipSnapshotWanted).
    // An elastic solid takes its OWN substance's blend (elastic_flip_blend):
    // the domain's flip_blend belongs to its Default Substance, often water.
    models[1].flip_blend = params.elastic_solid ? params.elastic_flip_blend
        : params.granular_snow_plasticity ? params.flip_blend : 0.0f;
    models[1].substance_viscosity = nullptr;
    models[1].kinematic_viscosity = 0.0f;
    Fluid::MatterGpuParticleLease shared_particles;
    for (int lane = 0; lane < 2; ++lane) {
        const float damping_dt = lane == 1 ? frame_dt
            : Fluid::Granular::kDampingReferenceDt;
        models[lane].velocity_damping = Fluid::Granular::timeScaledSubstepDamping(
            params.velocity_damping, damping_dt, grid_steps);
        // Affine damping is applied per G2P: the liquid lane gathers
        // liquid_steps times a frame (== grid_steps without the switch).
        models[lane].affine_damping = Fluid::Granular::timeScaledSubstepDamping(
            params.affine_damping, damping_dt,
            lane == 0 ? std::max(1, liquid_steps) : grid_steps);
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
        // The face mass gradient's only reader was the synchronous contact
        // kernel (removed 2026-11-11); the pore drag reads masses only. The P2G
        // skips its three gradient atomics per node (bit 31 of the lane).
        view.mass_gradient_wanted = false;
        // The liquid lane's collider wall separates (sim_matter_liquid_wall).
        view.liquid_wall = lane == 0;
        view.liquid_wall_density = params.granular_density;
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
    // Both lanes' wall contacts read each solid cell's collider surface: the
    // MPM lane all of it, the liquid lane its adhesion.
    if (!uploadGpuSolidContact(grid, *compute, *lanes[0]) ||
        !uploadGpuSolidContact(grid, *compute, *lanes[1])) {
        error = "mixed GPU collider contact surface upload failed";
        return false;
    }
    FluidGpuOccupancy liquid_occupancy(state, compute, lanes[0]);
    // S4: the MPM lane's grid sees only DEEP solid (phi < -h/2). A partly
    // covered cell stays open; the parcel contact below holds the true surface.
    // Cells without a distance (kinematic proxies) stay solid.
    std::vector<uint32_t> mpm_solid_cells;
    {
        const bool has_phi = grid.solid_phi.size() == cells;
        const float deep = -0.5f * grid.voxel_size;
        auto keep = [&](std::size_t cell) {
            if (!has_phi) return true;
            const float phi = grid.solid_phi[cell];
            return phi <= deep || phi >= FluidSim::FluidGrid::kSolidPhiFar;
        };
        if (grid.solid_cells_valid) {
            for (const uint32_t cell : grid.solid_cells) {
                if (keep(cell)) mpm_solid_cells.push_back(cell);
            }
        } else {
            for (std::size_t cell = 0; cell < cells && cell < grid.solid.size(); ++cell) {
                if (grid.solid[cell] && keep(cell)) {
                    mpm_solid_cells.push_back(static_cast<uint32_t>(cell));
                }
            }
        }
    }
    FluidGpuOccupancy granular_occupancy(state, compute, lanes[1], &mpm_solid_cells);
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
    // Both lanes' collider faces move and separate. MPM: the faces ARE its
    // contact (sim_matter_mpm_boundary). Liquid: after P2G they keep the
    // liquid's own velocity, the pressure solve decides held/released, and G2P
    // reads that -- not 0. Zeroing them made G2P see a rising sphere as a still
    // wall (2026-11-08: chocolate lifted ~0); sim_matter_liquid_wall.glsl.
    const bool closed_boundary =
        params.boundary == Fluid::APICSolverParams::BoundaryMode::Closed;
    const auto solid_faces = [&](int lane, float step_dt, bool wall_contact = true) {
        return lane == 1
            ? runGpuMpmSolidFaces(grid, compute, *lanes[1], closed_boundary, step_dt,
                                  params.granular_density, params, wall_contact)
            : runGpuMatterLiquidWall(grid, compute, *lanes[0], 0, step_dt);
    };
    // MPM porous coupling constants (sim_matter_mpm_porosity.glsl). Snow is made of
    // ice: the liquid fills its pores. Other MPM solids use their own density.
    constexpr float kMpmMinimumVoidage = 0.3f;
    const float mpm_constituent_density = params.granular_snow_plasticity
        ? Fluid::Granular::kIceDensity : std::max(params.granular_density, 1.0f);
    const auto* liquid_substance = Fluid::domainSubstance(params);
    const float mpm_liquid_density = liquid_substance && liquid_substance->liquid_density > 0.0f
        ? liquid_substance->liquid_density : 1000.0f;
    std::size_t mpm_porous_faces = 0;
    // Pore drag (sim_matter_liquid_drag): the liquid's dynamic viscosity and
    // the solid's grain size. No substance authors a snow grain size yet
    // (grain_real_radius_m is 0): 1 mm, mid-range of seasonal snow
    // (0.1-2 mm). Permeability goes as d^2, so this sets how fast liquid
    // percolates: the Darcy test measures it.
    constexpr float kMpmPoreGrainDiameter = 1.0e-3f;
    const float mpm_liquid_viscosity = liquid_substance
        ? std::max(liquid_substance->liquid_density, 1.0f) *
              std::max(liquid_substance->liquid_kinematic_viscosity, 1.0e-7f)
        : 1.0e-3f;
    // The last liquid projection saw the MPM lane as porous: its pressure keeps
    // pushing the MPM faces on every grid step until the next projection.
    bool mpm_reaction_live = false;
    for (int substep = 0; substep < substeps; ++substep) {
        stage_probe.select(Stage::Forces);
        if (clock && clock->forces &&
            !clock->forces(substep, common_substeps, dt, error)) {
            return false;
        }
        if (substep_forces) {
            auto force_params = params;
            force_params.granular_enabled = false;
            if (!runGpuFluidParticleIntegrateForces(state, force_params, Vec3(0.0f), dt,
                    substep_forces->frame_start_seconds + substep * dt,
                    substep_forces->buffer, compute, primary, false)) {
                error = "mixed GPU substep force integration failed";
                return false;
            }
        }
        if (substep % grid_stride == 0) {
            const float transfer_dt = dt * std::min(grid_stride, substeps - substep);
            // Liquid lane: transfer + projection only on its own steps.
            const bool liquid_step = has_fluid && substep % liquid_stride == 0;
            const float liquid_dt = dt * std::min(liquid_stride, substeps - substep);
            const int next_grid = substep + grid_stride;
            const bool liquid_interval_end = next_grid >= substeps ||
                next_grid % liquid_stride == 0;
            stage_probe.select(Stage::Occupancy);
            if ((liquid_step && !liquid_occupancy.build()) || !granular_occupancy.build()) {
                error = "mixed GPU model occupancy failed";
                return false;
            }
            if (liquid_step) {
                primary.fluid_mask_device_valid = true;
            }
            lanes[1]->fluid_mask_device_valid = true;
            for (int lane = 0; lane < 2; ++lane) {
                if (lane == 0 && !liquid_step) {
                    continue;
                }
                const float lane_dt = lane == 0 ? liquid_dt : transfer_dt;
                stage_probe.select(lane == 0 ? Stage::LiquidTransfer : Stage::MpmTransfer);
                bool transferred = runGpuFluidP2G(state, compute, *lanes[lane], models[lane],
                                                  lane_dt, false, true, false);
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
                                       lane_dt, false, true, false);
                }
                if (!transferred || !solid_faces(lane, lane_dt)) {
                    error = "mixed GPU indexed P2G/boundary failed";
                    return false;
                }
            }
            stage_probe.select(Stage::Flip);
            const bool sparse_flip_captured = liquid_step &&
                Fluid::captureSparseMacFlip(*compute, primary);
            if (liquid_step && !sparse_flip_captured && mac_storage.canonical) {
                error = "mixed GPU compact FLIP baseline capture failed";
                return false;
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (sparse_flip_captured) {
                    break;
                }
                if (!liquid_step) {
                    break;
                }
                if (!Fluid::copyMatterGpuFloat(*compute, post[axis], pre[axis],
                                              static_cast<uint32_t>(faces[axis]))) {
                    error = "mixed GPU liquid FLIP snapshot failed";
                    return false;
                }
            }
            if (liquid_step && (models[0].kinematic_viscosity > 0.0f ||
                                models[0].substance_viscosity)) {
                stage_probe.select(Stage::Viscosity);
                int sweeps = 0;
                if (!runGpuFluidViscosity(state, models[0], liquid_dt, compute, primary,
                                         unused_host_mask, &sweeps, &stats)) {
                    error = "mixed GPU liquid viscosity failed";
                    return false;
                }
                stats.viscosity_sweeps_run += sweeps;
            }
            stage_probe.select(Stage::Pressure);
            // The MPM lane as a porous moving solid in this projection, built on
            // the device (sim_matter_mpm_porosity, runMatterMpmPorosityGpu): the
            // host round trip it replaced read ~9.5 MB of MPM faces back per
            // liquid step and uploaded the result (2026-11-11: ~170 MB down and
            // ~150 MB up a frame while an emitter poured at 17 liquid steps).
            const bool mpm_porous = liquid_step && liquid_multirate &&
                compute->backendType() == ComputeBackendType::VulkanCompute;
            struct MpmPorosityRestore {
                FluidSim::FluidGrid& grid;
                SimulationGridDomainComputeBuffers& buffers;
                bool active = false;
                bool created_weights = false;
                ~MpmPorosityRestore() {
                    s_matter_porous_mobility_ratio = 0.0f;
                    s_matter_porous_fraction = ComputeBufferHandle{};
                    s_matter_porosity_gpu = MatterMpmPorosityGpu{};
                    if (!active) return;
                    buffers.porous_solid_velocity = false;
                    // var_sv* is restored on the device right after the solve
                    // (pass 2). Compact pages are repacked from the device
                    // source every solve; dense weights carry this solve's
                    // porous values, so the next one uploads the collider ones.
                    if (!Fluid::macVelocityBinding(buffers).compact) {
                        buffers.matter_model.pressure_statics_uploaded = false;
                    }
                    if (created_weights) {
                        grid.u_weight.clear();
                        grid.v_weight.clear();
                        grid.w_weight.clear();
                        buffers.sparse_mac_transfer.solid_weight_source_valid = false;
                    }
                }
            } mpm_porosity{grid, primary};
            auto pressure_model = models[0];
            if (mpm_porous) {
                // Effective solid mobility over the LIQUID step (step 2f): the
                // skeleton's stiffness K enters the solid block as M + dt^2 K,
                // locally 1 + 6 (c dt / h)^2 with c dt / h = 0.35 per elastic
                // substep (the substep count IS the elastic CFL). A free mass
                // (2c, r = rho_l / rho_c) was wrong over a long step: the
                // projection took the snow as already moving with the liquid,
                // built only eps of the hydrostatic pressure, and the slush
                // stopped rising (2.6 cm vs 8). Over 200 substeps r -> 1e-4:
                // the snow is a moving porous obstacle, its weight carried by
                // its own skeleton, the liquid's push returned by the reaction.
                const double interval_substeps = std::max(1.0, double(liquid_dt) / double(dt));
                const double wave = 0.35 * interval_substeps;
                const float mobility = static_cast<float>(
                    (double(mpm_liquid_density) / std::max(double(mpm_constituent_density), 1.0)) /
                    (1.0 + 6.0 * wave * wave));
                // The variational solve needs host-sized weight arrays; a domain
                // without colliders has none. Fully open ones for this solve.
                const std::size_t face_counts[3] = {grid.vel_x.size(), grid.vel_y.size(),
                                                    grid.vel_z.size()};
                std::vector<uint8_t>* host_weights[3] = {&grid.u_weight, &grid.v_weight,
                                                         &grid.w_weight};
                for (int axis = 0; axis < 3; ++axis) {
                    if (host_weights[axis]->size() != face_counts[axis]) {
                        host_weights[axis]->assign(face_counts[axis], 255u);
                        mpm_porosity.created_weights = true;
                    }
                }
                if (mpm_porosity.created_weights) {
                    primary.sparse_mac_transfer.solid_weight_source_valid = false;
                    primary.matter_model.pressure_statics_uploaded = false;
                }
                // The solid fraction rides the MPM lane's var_fluid_phi slot
                // (never a level set there) into sim_fluid_divergence_porous;
                // the liquid's own slot is never allocated on Vulkan.
                if (!allocation(lanes[1]->var_fluid_phi, grid.getCellCount() * sizeof(float),
                                "matter_mpm_cell_fraction")) {
                    error = "mixed GPU MPM solid fraction allocation failed";
                    return false;
                }
                s_matter_porous_fraction = lanes[1]->var_fluid_phi;
                s_matter_porous_mobility_ratio = mobility;
                s_matter_porosity_gpu.active = true;
                s_matter_porosity_gpu.mass = {lanes[1]->temperature, lanes[1]->fuel,
                                              lanes[1]->scratch_scalar};
                s_matter_porosity_gpu.velocity = {lanes[1]->vel_x, lanes[1]->vel_y,
                                                  lanes[1]->vel_z};
                s_matter_porosity_gpu.inv_full_mass = 1.0f /
                    (std::max(mpm_constituent_density, 1.0f) *
                     grid.voxel_size * grid.voxel_size * grid.voxel_size);
                s_matter_porosity_gpu.cap = 1.0f - kMpmMinimumVoidage;
                primary.porous_solid_velocity = true;
                pressure_model.variational_solids = true;
                mpm_porosity.active = true;
                mpm_porous_faces = 1;
            }
            if (liquid_step && !runGpuFluidMGPCGPressure(state, pressure_model, liquid_dt, compute,
                                                        primary, unused_host_mask, &stats)) {
                error = "mixed GPU liquid MGPCG projection failed";
                return false;
            }
            if (mpm_porosity.active && !runMatterMpmPorosityGpu(grid, compute, primary,
                    Fluid::macVelocityBinding(primary), true)) {
                error = "mixed GPU MPM porosity restore failed";
                return false;
            }
            if (liquid_step) {
                mpm_reaction_live = mpm_porous && primary.porous_solid_velocity;
            }
            // Liquid lane: dense bank, or its compact velocity and mass pages
            // plus the MAC tile map and list. The granular lane stays dense.
            stage_probe.select(Stage::Contact);
            const auto liquid_mac = Fluid::macVelocityBinding(primary);
            const ComputeBufferHandle contact_buffers[] = {
                liquid_mac.velocity[0], liquid_mac.velocity[1], liquid_mac.velocity[2],
                lanes[1]->vel_x, lanes[1]->vel_y, lanes[1]->vel_z,
                liquid_mac.weight[0], liquid_mac.weight[1], liquid_mac.weight[2],
                lanes[1]->temperature, lanes[1]->fuel, lanes[1]->scratch_scalar,
                runtime.gradient[0][0], runtime.gradient[0][1], runtime.gradient[0][2],
                runtime.gradient[1][0], runtime.gradient[1][1], runtime.gradient[1][2],
                runtime.contact_pairs, liquid_mac.map, liquid_mac.list};
            // On the liquid's own clock (step 2f) the two fast couplings run on
            // the MPM grid clock, the pressure stays on the liquid's:
            //  * the pressure's push on the solid as a FORCE: the last
            //    projection's gradient over this grid step's dt (design 3.2,
            //    "a constant force, not an impulse");
            //  * the pore drag, implicit over the grid dt, on both sides: the
            //    liquid MAC face and the MPM face exchange equal and opposite.
            //    The liquid's sum reaches its parcels by the interval-end delta
            //    gather below.
            // On the liquid step (2b/2d) the pair relaxed toward its common
            // velocity, 0.58 u_liquid for grounded snow: the snow took 42 % and
            // handed it to the floor only during the next 200 elastic substeps
            // -- the liquid kept 58 % per step, an effective damping of
            // ~14 s^-1 at 1/24 against ~110 s^-1 resolved (mean |p| 2.6 vs 0.72).
            // tau ~ 9 ms >> the grid dt: resolved here, any liquid step.
            if (liquid_multirate && mpm_reaction_live) {
                stage_probe.select(Stage::Pressure);
                struct ReactionConstants {
                    int nx, ny, nz, boundary;
                    float voxel_size, dt, density_ratio, inv_full_mass, max_fraction;
                    uint32_t lanes;
                } reaction_constants{grid.nx, grid.ny, grid.nz, closed_boundary ? 1 : 0,
                    grid.voxel_size, transfer_dt,
                    mpm_liquid_density / std::max(mpm_constituent_density, 1.0f),
                    1.0f / (std::max(mpm_constituent_density, 1.0f) *
                            grid.voxel_size * grid.voxel_size * grid.voxel_size),
                    1.0f - kMpmMinimumVoidage,
                    static_cast<uint32_t>(std::max({faces[0], faces[1], faces[2]}))};
                static_assert(sizeof(ReactionConstants) == 40);
                const ComputeBufferHandle reaction_buffers[] = {
                    primary.pressure, primary.fluid_mask,
                    lanes[1]->vel_x, lanes[1]->vel_y, lanes[1]->vel_z,
                    lanes[1]->temperature, lanes[1]->fuel, lanes[1]->scratch_scalar};
                ComputeDispatch reaction;
                reaction.kernel = "sim_matter_mpm_pressure_reaction";
                reaction.buffers = reaction_buffers;
                reaction.buffer_count = 8;
                reaction.constants = &reaction_constants;
                reaction.constants_size = sizeof(reaction_constants);
                reaction.groups = FluidGpuDispatch::groups256(reaction_constants.lanes);
                if (!compute->dispatch(reaction)) {
                    error = "mixed GPU MPM pressure reaction failed";
                    return false;
                }
                stage_probe.select(Stage::Contact);
            }
            struct DragConstants {
                int nx, ny, nz;
                float dt, drag_prefactor, inv_full_mass, max_fraction, face_volume;
                float min_liquid_mass, pad0, pad1, pad2;
            } drag_constants{grid.nx, grid.ny, grid.nz, transfer_dt,
                180.0f * mpm_liquid_viscosity /
                    (kMpmPoreGrainDiameter * kMpmPoreGrainDiameter),
                1.0f / (std::max(mpm_constituent_density, 1.0f) *
                        grid.voxel_size * grid.voxel_size * grid.voxel_size),
                1.0f - kMpmMinimumVoidage,
                grid.voxel_size * grid.voxel_size * grid.voxel_size,
                0.05f * mpm_liquid_density * grid.voxel_size * grid.voxel_size * grid.voxel_size,
                0.0f, 0.0f, 0.0f};
            static_assert(sizeof(DragConstants) == 48);
            ComputeDispatch drag;
            drag.kernel = Fluid::macKernel(liquid_mac, "sim_matter_liquid_drag",
                                           "sim_sparse_mac_matter_liquid_drag");
            drag.buffers = contact_buffers;
            drag.buffer_count = liquid_mac.compact ? 21 : 19;
            drag.constants = &drag_constants;
            drag.constants_size = sizeof(drag_constants);
            drag.groups = FluidGpuDispatch::groups256(static_cast<uint32_t>(lattice));
            if (liquid_multirate && !compute->dispatch(drag)) {
                error = "mixed GPU liquid pore drag failed";
                return false;
            }
            stage_probe.select(Stage::Gather);
            // Every gather observes the same post-contact substep. Each lane writes
            // only its model's canonical indices; no liquid pressure on granular.
            for (int lane = 0; lane < 2; ++lane) {
                if (lane == 0 && !has_fluid) {
                    continue;
                }
                if (lane == 0 && !liquid_step) {
                    // Between liquid steps the pore drag keeps adding to the
                    // liquid MAC field. Its sum reaches the
                    // parcels at the interval's end: FLIP 1 against the
                    // post-gather baseline = v_p += (u_now - u_gathered).
                    if (liquid_interval_end) {
                        auto delta = models[0];
                        delta.flip_blend = 1.0f;
                        delta.affine_damping = 1.0f;  // applied by the step's gather
                        if (!solid_faces(0, liquid_dt, false) ||
                            !runGpuFluidG2P(state, delta, liquid_dt, compute, primary,
                                true, false, true, true, true)) {
                            error = "mixed GPU liquid exchange gather failed";
                            return false;
                        }
                    }
                    continue;
                }
                const float lane_dt = lane == 0 ? liquid_dt : transfer_dt;
                if (!solid_faces(lane, lane_dt, false) ||
                    !runGpuFluidG2P(state, models[lane], lane_dt, compute, *lanes[lane],
                        lane == 0 || granularFlipSnapshotWanted(models[lane], *lanes[lane]),
                        false, true, true, true)) {
                    error = "mixed GPU indexed gather/constitutive failed";
                    return false;
                }
                if (lane == 0 && liquid_multirate && !liquid_interval_end) {
                    // Baseline for that delta: the field this gather just read.
                    stage_probe.select(Stage::Flip);
                    const bool rebased = Fluid::captureSparseMacFlip(*compute, primary);
                    for (int axis = 0; axis < 3 && !rebased; ++axis) {
                        if (mac_storage.canonical ||
                            !Fluid::copyMatterGpuFloat(*compute, post[axis], pre[axis],
                                                      static_cast<uint32_t>(faces[axis]))) {
                            error = "mixed GPU liquid exchange baseline failed";
                            return false;
                        }
                    }
                    stage_probe.select(Stage::Gather);
                }
            }
        }
        // All owners see current gathered velocities. Contact reactions land
        // before either continuum lane advects, and feed the next P2G directly.
        stage_probe.select(Stage::Contact);
        if (clock && clock->contact &&
            !clock->contact(substep, common_substeps, dt, error)) {
            return false;
        }
        stage_probe.select(Stage::Advection);
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
            if (lane == 1 && !runGpuMpmParticleContact(grid, compute, *lanes[1], dt,
                    params.granular_density, static_cast<uint32_t>(count))) {
                error = "mixed GPU MPM parcel collider contact failed";
                return false;
            }
        }
    }
    // Publish only after the entire common subcycle succeeded. Metadata sidecars
    // are copied with the same canonical index; identity/allocator never rebirth.
    stage_probe.select(Stage::Publication);
    // Scale S1 slice A (docs/dev/OLCEK_MILYON_PARCACIK.md): the frame used to
    // deep-copy the whole particle set (`auto liquid_result = particles;`, ~40
    // SoA columns) and allocate+zero 14 sidecar columns, then commit back:
    // ~450 ms of serial CPU per frame at 1M parcels. The device writes into
    // persistent staging columns instead; after validation they are SWAPPED
    // into the particles (the old columns become next frame's staging).
    // Atomicity is kept up to the swap: a failed download or a nonfinite value
    // leaves the particles untouched. The pore exchange after it is the one
    // stage that can still fail on committed transport (it holds the domain).
    struct PublicationStaging {
        std::vector<Vec3> position, velocity;
        std::vector<Fluid::AffineC> affine;
        Fluid::GranularFrameReadback granular{0};
    };
    static thread_local PublicationStaging staging_scratch;
    auto& staging = staging_scratch; // see the dry volume note: never name the thread_local in OpenMP
    staging.position.resize(count);
    staging.velocity.resize(count);
    staging.affine.resize(count);
    auto& granular_result = staging.granular;
    Fluid::forEachGranularReadbackColumn(granular_result, lanes[1]->granular,
        [&](ComputeBufferHandle, auto& values) { values.resize(count); });
    compute->beginTransferBatch();
    // Both indexed writers used these same canonical streams. Publish once,
    // including the granular-only case; sidecars retain model-specific owners.
    bool ok = compute->downloadBuffer(primary.fluid_positions,
        staging.position.data(), count * sizeof(Vec3));
    ok = compute->downloadBuffer(primary.fluid_velocities,
        staging.velocity.data(), count * sizeof(Vec3)) && ok;
    ok = compute->downloadBuffer(primary.fluid_affine,
        staging.affine.data(), count * sizeof(Fluid::AffineC)) && ok;
    uint32_t pairs = 0;
    ok = compute->downloadBuffer(runtime.contact_pairs, &pairs, sizeof(pairs)) && ok;
    std::array<std::vector<float>, 3> result_velocity{
        grid.vel_x, grid.vel_y, grid.vel_z};
    const bool compact_publication = has_fluid && mac_storage.used && mac_storage.canonical;
    for (int axis = 0; axis < 3 && !compact_publication; ++axis) {
        ok = compute->downloadBuffer(post[axis], result_velocity[axis].data(),
                                    faces[axis] * sizeof(float)) && ok;
    }
    // Sidecar destinations stay alive through the same fence as transport.
    ok = Fluid::enqueueGranularSidecars(*compute, lanes[1]->granular,
        granular_result, count) && ok;
    ok = granular_result.canCommit(particles, count) && ok;
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
    // Adaptive snow sizing was too low for this frame: discard and re-run at
    // the ceiling (nothing has been committed; see snow_adaptive above).
    if (snow_adaptive && snow_sizing_hardening < snow_ceiling) {
        bool exceeded = maxSnowHardening(granular_result.granular_plastic_volume) >
            snow_sizing_hardening * 1.0001f;
        for (std::size_t i = 0; i < count && !exceeded; ++i) {
            const auto model = i < particles.constitutive_model.size()
                ? static_cast<Fluid::MatterConstitutiveModel>(particles.constitutive_model[i])
                : Fluid::MatterConstitutiveModel::Auto;
            exceeded = mpm_parcel(model) &&
                (granular_result.granular_material_flags[i] & Fluid::Granular::kFlagHardeningCapped) != 0u;
        }
        if (exceeded) {
            // Re-run by the caller, AFTER this frame's scope (particle lease,
            // staging) has been released.
            stage_probe.finish();
            retry_at_ceiling = true;
            error = "snow sizing below the frame's hardening; re-run at the ceiling";
            return false;
        }
    }
    // Validate the staged transport before anything is committed.
    const auto signed_count = static_cast<int64_t>(count);
    int64_t nonfinite = 0;
    const Vec3* const staged_position = staging.position.data();
    const Vec3* const staged_velocity = staging.velocity.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : nonfinite) if(signed_count > 65536)
#endif
    for (int64_t i = 0; i < signed_count; ++i) {
        const auto& p = staged_position[i];
        const auto& v = staged_velocity[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            ++nonfinite;
        }
    }
    if (nonfinite != 0) {
        error = "mixed GPU produced nonfinite canonical particle state";
        return false;
    }
    // Commit: transport by swap, constitutive sidecars only for MPM owners
    // (granular AND elastic). Liquid sidecars and particle identity stay intact.
    particles.position.swap(staging.position);
    particles.velocity.swap(staging.velocity);
    particles.affine.swap(staging.affine);
    // Held parcels did not move; they carry no velocity of their own.
    for (std::size_t i = 0; i < held_parcels.size() && i < count; ++i) {
        if (held_parcels[i]) {
            particles.velocity[i] = Vec3(0.0f);
            particles.affine[i] = Fluid::AffineC();
        }
    }
    const auto model_of = [&](std::size_t i) {
        return i < particles.constitutive_model.size()
            ? static_cast<Fluid::MatterConstitutiveModel>(particles.constitutive_model[i])
            : Fluid::MatterConstitutiveModel::Auto;
    };
    int64_t non_mpm = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : non_mpm) if(signed_count > 65536)
#endif
    for (int64_t i = 0; i < signed_count; ++i) {
        if (!mpm_parcel(model_of(static_cast<std::size_t>(i)))) ++non_mpm;
    }
    if (non_mpm == 0) {
        // All-MPM domain: whole columns swap (a column may be longer than
        // count from an earlier larger frame: cut it to the particle size).
        const auto size = particles.size();
#define RT_SWAP_SIDECAR(field)         granular_result.field.resize(size); particles.field.swap(granular_result.field);
        RT_SWAP_SIDECAR(granular_stress_diag) RT_SWAP_SIDECAR(granular_stress_shear)
        RT_SWAP_SIDECAR(granular_plastic_volume) RT_SWAP_SIDECAR(granular_material_flags)
        RT_SWAP_SIDECAR(granular_yield_value) RT_SWAP_SIDECAR(granular_plastic_increment)
        RT_SWAP_SIDECAR(granular_damage) RT_SWAP_SIDECAR(granular_hardening)
        RT_SWAP_SIDECAR(granular_fracture_history) RT_SWAP_SIDECAR(granular_support_prev_vg)
        RT_SWAP_SIDECAR(granular_support_accel) RT_SWAP_SIDECAR(granular_deformation_col0)
        RT_SWAP_SIDECAR(granular_deformation_col1) RT_SWAP_SIDECAR(granular_deformation_col2)
#undef RT_SWAP_SIDECAR
    } else if (non_mpm != signed_count) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(signed_count > 65536)
#endif
        for (int64_t i = 0; i < signed_count; ++i) {
            if (mpm_parcel(model_of(static_cast<std::size_t>(i)))) {
                granular_result.commitParticle(particles, static_cast<std::size_t>(i));
            }
        }
    }
    std::vector<MatterExchangeRecord> pore_events;
    if (params.pore_exchange.enabled) {
        const std::size_t remaining_budget = params.mixed_working_set_budget_bytes
            ? params.mixed_working_set_budget_bytes - working_set + pore_working : 0;
        // Equality means no free budget, not an unlimited budget.
        if ((params.mixed_working_set_budget_bytes && !remaining_budget) ||
            !Fluid::exchangeMatterPoresGpu(particles, grid, params.pore_exchange, frame_dt,
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
        for (std::size_t i = particles.size(); i-- > 0;) {
            const auto& p = particles.position[i];
            if (p.x < minimum.x || p.y < minimum.y || p.z < minimum.z ||
                p.x > maximum.x || p.y > maximum.y || p.z > maximum.z) {
                particles.removeSwap(i);
            }
        }
    }
    particles.uvw_refresh_period = params.uvw_refresh_period;
    particles.advanceMaterialCoordinates();
    for (auto& event : pore_events) {
        ledger.record(std::move(event));
    }
    grid.vel_x = std::move(result_velocity[0]);
    grid.vel_y = std::move(result_velocity[1]);
    grid.vel_z = std::move(result_velocity[2]);
    // ★★★ The device state was published above every frame and never counted:
    // a 4 m/s Snow impact read 0 yielded / 0 damaged for 120 steps. Count the
    // granular parcels only; liquid parcels' granular sidecars are never written.
    Fluid::Granular::countMaterialState(particles, stats, [&](std::size_t i) {
        const auto model = i < particles.constitutive_model.size()
            ? static_cast<Fluid::MatterConstitutiveModel>(particles.constitutive_model[i])
            : Fluid::MatterConstitutiveModel::Auto;
        return mpm_parcel(model);
    });
    stats.mpm_law = !has_granular ? -1 : elastic_solid ? 2 : snow ? 1 : 0;
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
    // Resting share this frame (the domain was not all asleep, so it ran).
    stats.mpm_resting_particles = mpm_sleep.resting_particles;
    stats.mixed_common_substeps = substeps;
    stats.mixed_contact_pairs = pairs;
    stats.mixed_working_set_bytes = working_set;
    stats.p2g_on_gpu = stats.g2p_on_gpu = true;
    stats.pressure_on_gpu = has_fluid;
    stats.viscosity_on_gpu = stats.viscosity_sweeps_run > 0;
    stats.gpu_status = has_fluid
        ? "Mixed Vulkan: indexed transfer/contact/common substeps"
        : "Mixed Vulkan: granular-only indexed/common substeps";
    if (mpm_porous_faces > 0) {
        stats.gpu_status += "; MPM porous in the liquid projection";
    }
    Fluid::publishSparseMacTransferStats(primary.sparse_mac_transfer, stats);
    stage_probe.finish();
    state.fluid_stats = stats;
    return true;
}

// Snow frames sized below the hardening ceiling are re-run at the ceiling when
// a parcel outgrew the sizing (see snow_adaptive in runMatterGpuStepOnce).
bool runMatterGpuStep(SimulationGridDomainState& state,
    const Fluid::APICSolverParams& params, bool legacy_granular, float frame_dt,
    SimulationComputeContext* compute, SimulationGridDomainComputeBuffers& primary,
    const std::function<bool(SimulationGridDomainComputeBuffers&)>& ensure,
    MatterExchangeLedger& ledger,
    std::string& error, const Fluid::MatterCommonClockHooks* clock = nullptr,
    const MatterSubstepForces* substep_forces = nullptr) {
    bool retry = false;
    if (runMatterGpuStepOnce(state, params, legacy_granular, frame_dt, compute, primary,
            ensure, ledger, error, clock, substep_forces, false, retry)) {
        return true;
    }
    if (!retry) return false;
    bool unused = false;
    return runMatterGpuStepOnce(state, params, legacy_granular, frame_dt, compute, primary,
        ensure, ledger, error, clock, substep_forces, true, unused);
}
