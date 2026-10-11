struct FoamBinGpuConstants {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int particle_count = 0;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;
    float voxel_size = 1.0f;
    int max_per_cell = 0;
    int cell_count = 0;
    int offset_x = 0, offset_y = 0, offset_z = 0;
};

struct FoamCritGpuConstants {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int particle_count = 0;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;
    float voxel_size = 1.0f;
    float h = 0.0f;
    int reach = 1;
    int max_per_cell = 0;
    float k_ta = 0.0f;
    float k_wc = 0.0f;
    float crest_cos = 0.0f;
    float ta_min = 0.0f;
    float ta_max = 1.0f;
    float wc_min = 0.0f;
    float wc_max = 1.0f;
    float ke_min = 0.0f;
    float ke_max = 1.0f;
    float dt = 0.0f;
    int offset_x = 0, offset_y = 0, offset_z = 0;
};

// sim_foam_neigh: same grid/bucket prefix as the criterion, minus every Ihmsen
// coefficient (this kernel only counts). 14 fields x 4 = 56 bytes.
struct FoamNeighGpuConstants {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int foam_count = 0;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;
    float voxel_size = 1.0f;
    float h = 0.0f;
    int reach = 1;
    int max_per_cell = 0;
    int offset_x = 0, offset_y = 0, offset_z = 0;
};

static_assert(sizeof(FoamBinGpuConstants) == 52);
static_assert(sizeof(FoamCritGpuConstants) == 96);
static_assert(sizeof(FoamNeighGpuConstants) == 56);

// ── GPU whitewater spawn potentials (Ihmsen criteria) ───────────────────────
// Replaces stepFoam's phase 3, measured at 66.5% of the entire CPU foam cost and
// insensitive to any foam budget knob because it scales with the FLUID particle
// count. Fills `expected_out` with one fractional spawn count per fluid particle;
// the host still runs the stochastic emit, so the result stays directly
// comparable against the CPU reference (which remains the oracle).
//
// Returns false on any allocation/dispatch failure — the caller then simply runs
// the untouched CPU path, so this can never make foam disappear.
bool queueGpuFoamCriteria(SimulationGridDomainState& state,
                        const Fluid::FoamParams& fparams,
                        float dt,
                        SimulationComputeContext* compute,
                        SimulationGridDomainComputeBuffers& gpu_buffers,
                        std::vector<float>& expected_out,
                        const Fluid::APICSolverParams& source_params) {
    auto& grid = state.grid;
    const std::size_t particle_count = state.particles.size();
    if (!compute || !compute->supportsDispatch() || particle_count == 0 ||
        !fparams.enabled || fparams.max_foam == 0 || dt <= 0.0f ||
        (fparams.trapped_air_rate <= 0.0f && fparams.wave_crest_rate <= 0.0f) ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || grid.voxel_size <= 0.0f) {
        return false;
    }

    // Positions AND velocities: the criterion needs relative velocity, unlike the
    // density splat which uploads positions only.
    if (!ensureGpuFluidParticleBuffers(state, compute, gpu_buffers,
                                       /*upload_positions_only=*/false)) {
        return false;
    }

    const auto window = Fluid::whitewaterBinWindow(state, fparams.neighbor_radius_voxels,
                                                   gpu_buffers.matter_model.enabled, &source_params);
    const std::size_t cell_count = static_cast<std::size_t>(window.nx) * window.ny * window.nz;
    if (cell_count == 0 || cell_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    gpu_buffers.foam_bin_window = window;
    // Bucket capacity. A cell normally holds ~particles_per_cell particles; the
    // 4x headroom is what lets us skip a prefix scan entirely. The kernel counts
    // anything above the cap instead of silently biasing the result, and the
    // tally below turns that into a visible warning rather than a mystery.
    const int max_per_cell = 32;
    if (cell_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / max_per_cell) {
        return false;
    }
    const ComputeBufferUsage usage = ComputeBufferUsage::Storage |
                                     ComputeBufferUsage::Upload |
                                     ComputeBufferUsage::Download |
                                     ComputeBufferUsage::ReadWrite;
    const bool layout_changed = !gpu_buffers.foam_bin_counts.valid() ||
        !gpu_buffers.foam_bin_items.valid() ||
        gpu_buffers.foam_bin_allocated_cells < static_cast<int>(cell_count) ||
        gpu_buffers.foam_bin_cell_capacity != max_per_cell;
    if (layout_changed) {
        const std::size_t full_cells = grid.getCellCount();
        const std::size_t previous = static_cast<std::size_t>(
            std::max(0, gpu_buffers.foam_bin_allocated_cells));
        const std::size_t allocated = std::min(full_cells,
            std::max(cell_count, previous + previous / 2));
        if (gpu_buffers.foam_bin_counts.valid()) {
            compute->destroyBuffer(gpu_buffers.foam_bin_counts);
            gpu_buffers.foam_bin_counts = {};
        }
        if (gpu_buffers.foam_bin_items.valid()) {
            compute->destroyBuffer(gpu_buffers.foam_bin_items);
            gpu_buffers.foam_bin_items = {};
        }
        ComputeBufferDesc desc;
        desc.usage = usage;
        // +1 counter slot: the trailing overflow tally.
        desc.debug_name = "FoamBinCounts";
        desc.size_bytes = (allocated + 1) * sizeof(uint32_t);
        gpu_buffers.foam_bin_counts = compute->createBuffer(desc);
        desc.debug_name = "FoamBinItems";
        desc.size_bytes = allocated * static_cast<std::size_t>(max_per_cell) *
                          sizeof(uint32_t);
        gpu_buffers.foam_bin_items = compute->createBuffer(desc);
        gpu_buffers.foam_bin_allocated_cells = static_cast<int>(allocated);
        gpu_buffers.foam_bin_cell_capacity = max_per_cell;
    }
    gpu_buffers.foam_bin_cell_count = static_cast<int>(cell_count);
    if (!gpu_buffers.foam_bin_counts.valid() || !gpu_buffers.foam_bin_items.valid()) {
        return false;
    }

    if (!gpu_buffers.foam_expected.valid() ||
        gpu_buffers.foam_expected_capacity < particle_count) {
        if (gpu_buffers.foam_expected.valid()) {
            compute->destroyBuffer(gpu_buffers.foam_expected);
            gpu_buffers.foam_expected = {};
        }
        // Grow with the same geometric headroom the particle streams use, so
        // emitter-driven count changes do not reallocate every step.
        std::size_t capacity = std::max<std::size_t>(particle_count, 4096u);
        capacity += capacity / 2u;
        ComputeBufferDesc desc;
        desc.debug_name = "FoamExpected";
        desc.size_bytes = capacity * sizeof(float);
        desc.usage = usage;
        gpu_buffers.foam_expected = compute->createBuffer(desc);
        gpu_buffers.foam_expected_capacity =
            gpu_buffers.foam_expected.valid() ? capacity : 0;
        // The buffer the pending dispatch wrote into is gone.
        gpu_buffers.foam_expected_pending = false;
    }
    if (!gpu_buffers.foam_expected.valid()) return false;

    static thread_local std::vector<uint32_t> eligible;
    eligible.resize(particle_count);
    for (std::size_t p = 0; p < particle_count; ++p) {
        eligible[p] = Fluid::isWhitewaterFluidParcel(state.particles, p, &source_params) ? 1u : 0u;
    }
    if (!gpu_buffers.foam_eligible.valid() || gpu_buffers.foam_eligible_capacity < particle_count) {
        if (gpu_buffers.foam_eligible.valid()) compute->destroyBuffer(gpu_buffers.foam_eligible);
        const std::size_t capacity = gpu_buffers.foam_expected_capacity;
        gpu_buffers.foam_eligible = compute->createBuffer(
            {"WhitewaterEligible", capacity * sizeof(uint32_t), usage});
        gpu_buffers.foam_eligible_capacity = gpu_buffers.foam_eligible.valid() ? capacity : 0;
    }
    if (!gpu_buffers.foam_eligible.valid() || !compute->uploadBuffer(
            gpu_buffers.foam_eligible, eligible.data(), particle_count * sizeof(uint32_t))) {
        return false;
    }

    const int reach = std::max(1, static_cast<int>(
        std::ceil(fparams.neighbor_radius_voxels)));

    FoamBinGpuConstants bin_c;
    bin_c.nx = window.nx;
    bin_c.ny = window.ny;
    bin_c.nz = window.nz;
    bin_c.particle_count = static_cast<int>(std::min<std::size_t>(
        particle_count, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    bin_c.origin_x = grid.origin.x;
    bin_c.origin_y = grid.origin.y;
    bin_c.origin_z = grid.origin.z;
    bin_c.voxel_size = grid.voxel_size;
    bin_c.offset_x = window.x;
    bin_c.offset_y = window.y;
    bin_c.offset_z = window.z;
    bin_c.max_per_cell = max_per_cell;
    bin_c.cell_count = static_cast<int>(cell_count);

    gpu_buffers.foam_expected_pending = false;
    constexpr uint32_t threads = 256;
    ComputeBufferHandle clear_buffers[2] = {
        gpu_buffers.foam_bin_counts,
        gpu_buffers.foam_expected
    };
    ComputeDispatch cmd;
    cmd.kernel = "sim_foam_bin_clear";
    cmd.buffers = clear_buffers;
    cmd.buffer_count = 2;
    cmd.constants = &bin_c;
    cmd.constants_size = sizeof(bin_c);
    // Clears two ranges of different length (cells+overflow, and particles), each
    // bounded inside the kernel, so cover whichever is larger.
    cmd.groups = FluidGpuDispatch::groups256(
        static_cast<uint32_t>(std::max(cell_count + 1, particle_count)));
    bool ok = compute->dispatch(cmd);

    ComputeBufferHandle scatter_buffers[4] = {
        gpu_buffers.fluid_positions,
        gpu_buffers.foam_bin_counts,
        gpu_buffers.foam_bin_items,
        gpu_buffers.foam_eligible
    };
    cmd.kernel = "sim_foam_bin_scatter";
    cmd.buffers = scatter_buffers;
    cmd.buffer_count = 4;
    cmd.groups = FluidGpuDispatch::groups256(static_cast<uint32_t>(bin_c.particle_count));
    ok = ok && compute->dispatch(cmd);

    FoamCritGpuConstants crit_c;
    crit_c.nx = window.nx;
    crit_c.ny = window.ny;
    crit_c.nz = window.nz;
    crit_c.particle_count = bin_c.particle_count;
    crit_c.origin_x = grid.origin.x;
    crit_c.origin_y = grid.origin.y;
    crit_c.origin_z = grid.origin.z;
    crit_c.voxel_size = grid.voxel_size;
    crit_c.offset_x = window.x;
    crit_c.offset_y = window.y;
    crit_c.offset_z = window.z;
    crit_c.h = std::max(1e-4f, fparams.neighbor_radius_voxels * grid.voxel_size);
    crit_c.reach = reach;
    crit_c.max_per_cell = max_per_cell;
    crit_c.k_ta = fparams.trapped_air_rate;
    crit_c.k_wc = fparams.wave_crest_rate;
    crit_c.crest_cos = fparams.crest_cos;
    crit_c.ta_min = fparams.ta_min;
    crit_c.ta_max = fparams.ta_max;
    crit_c.wc_min = fparams.wc_min;
    crit_c.wc_max = fparams.wc_max;
    crit_c.ke_min = fparams.ke_min;
    crit_c.ke_max = fparams.ke_max;
    crit_c.dt = dt;

    ComputeBufferHandle crit_buffers[6] = {
        gpu_buffers.fluid_positions,
        gpu_buffers.fluid_velocities,
        gpu_buffers.foam_bin_counts,
        gpu_buffers.foam_bin_items,
        gpu_buffers.foam_expected,
        gpu_buffers.foam_eligible
    };
    cmd.kernel = "sim_foam_crit";
    cmd.buffers = crit_buffers;
    cmd.buffer_count = 6;
    cmd.constants = &crit_c;
    cmd.constants_size = sizeof(crit_c);
    // One thread per BUCKET SLOT, not per particle: consecutive lanes then land on
    // consecutive cells and share a neighbourhood instead of scattering across the
    // domain in emission order. Costs idle lanes in partly-filled cells — with
    // ~4 particles per cell against a cap of 32 that is most of them, which is
    // exactly what this measurement is meant to settle.
    cmd.groups = FluidGpuDispatch::groups256(
        static_cast<uint32_t>(cell_count * static_cast<std::size_t>(max_per_cell)));
    ok = ok && compute->dispatch(cmd);
    if (!ok) return false;

    // This dispatch is collected by the current-input batch before any index changes.
    gpu_buffers.foam_expected_pending = true;
    gpu_buffers.foam_expected_pending_count = particle_count;
    // Bins are current for this step's liquid: the neighbour pass may now reuse
    // them. Cleared by its dispatch so one build is never consumed twice.
    gpu_buffers.foam_bin_ready_this_step = true;

    // No previous-frame potentials are consumed.
    return true;
}

// Neighbour counts are collected before stepFoam mutates any foam indices.
void dispatchGpuFoamNeighbours(SimulationGridDomainState& state,
                               const Fluid::FoamParams& fparams,
                               SimulationComputeContext* compute,
                               SimulationGridDomainComputeBuffers& gpu_buffers) {
    const bool bins_ready = gpu_buffers.foam_bin_ready_this_step;
    gpu_buffers.foam_bin_ready_this_step = false;  // consumed either way

    auto& grid = state.grid;
    const std::size_t foam_count = state.foam.size();
    if (!bins_ready || !compute || !compute->supportsDispatch() || foam_count == 0 ||
        !fparams.enabled || grid.voxel_size <= 0.0f ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        !gpu_buffers.foam_bin_counts.valid() || !gpu_buffers.foam_bin_items.valid() ||
        !gpu_buffers.fluid_positions.valid()) {
        return;
    }

    const ComputeBufferUsage usage = ComputeBufferUsage::Storage |
                                     ComputeBufferUsage::Upload |
                                     ComputeBufferUsage::Download |
                                     ComputeBufferUsage::ReadWrite;
    // Foam is hard-capped by max_foam, so after the first few steps this stops
    // reallocating entirely; grow geometrically anyway so ramp-up does not
    // destroy/recreate on every step.
    if (!gpu_buffers.foam_positions.valid() ||
        gpu_buffers.foam_positions_capacity < foam_count) {
        if (gpu_buffers.foam_positions.valid()) {
            compute->destroyBuffer(gpu_buffers.foam_positions);
            gpu_buffers.foam_positions = {};
        }
        std::size_t capacity = std::max<std::size_t>(foam_count, 4096u);
        capacity += capacity / 2u;
        ComputeBufferDesc desc;
        desc.debug_name = "FoamPositions";
        desc.size_bytes = capacity * sizeof(Vec3);
        desc.usage = usage;
        gpu_buffers.foam_positions = compute->createBuffer(desc);
        gpu_buffers.foam_positions_capacity =
            gpu_buffers.foam_positions.valid() ? capacity : 0;
    }
    if (!gpu_buffers.foam_neigh.valid() ||
        gpu_buffers.foam_neigh_capacity < foam_count) {
        if (gpu_buffers.foam_neigh.valid()) {
            compute->destroyBuffer(gpu_buffers.foam_neigh);
            gpu_buffers.foam_neigh = {};
        }
        std::size_t capacity = std::max<std::size_t>(foam_count, 4096u);
        capacity += capacity / 2u;
        ComputeBufferDesc desc;
        desc.debug_name = "FoamNeighbourCounts";
        desc.size_bytes = capacity * sizeof(uint32_t);
        desc.usage = usage;
        gpu_buffers.foam_neigh = compute->createBuffer(desc);
        gpu_buffers.foam_neigh_capacity =
            gpu_buffers.foam_neigh.valid() ? capacity : 0;
    }
    if (!gpu_buffers.foam_positions.valid() || !gpu_buffers.foam_neigh.valid()) return;

    if (!compute->uploadBuffer(gpu_buffers.foam_positions,
                               state.foam.position.data(),
                               foam_count * sizeof(Vec3))) {
        return;
    }

    FoamNeighGpuConstants c;
    c.nx = gpu_buffers.foam_bin_window.nx;
    c.ny = gpu_buffers.foam_bin_window.ny;
    c.nz = gpu_buffers.foam_bin_window.nz;
    c.foam_count = static_cast<int>(std::min<std::size_t>(
        foam_count, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    c.origin_x = grid.origin.x;
    c.origin_y = grid.origin.y;
    c.origin_z = grid.origin.z;
    c.voxel_size = grid.voxel_size;
    c.offset_x = gpu_buffers.foam_bin_window.x;
    c.offset_y = gpu_buffers.foam_bin_window.y;
    c.offset_z = gpu_buffers.foam_bin_window.z;
    c.h = std::max(1e-4f, fparams.neighbor_radius_voxels * grid.voxel_size);
    c.reach = std::max(1, static_cast<int>(std::ceil(fparams.neighbor_radius_voxels)));
    c.max_per_cell = gpu_buffers.foam_bin_cell_capacity;

    constexpr uint32_t threads = 256;
    ComputeBufferHandle buffers[5] = {
        gpu_buffers.foam_positions,
        gpu_buffers.fluid_positions,
        gpu_buffers.foam_bin_counts,
        gpu_buffers.foam_bin_items,
        gpu_buffers.foam_neigh
    };
    ComputeDispatch cmd;
    cmd.kernel = "sim_foam_neigh";
    cmd.buffers = buffers;
    cmd.buffer_count = 5;
    cmd.constants = &c;
    cmd.constants_size = sizeof(c);
    // One thread per foam particle. Slot-dispatch was measured here and did not pay
    // — see the kernel's header comment before trying it again.
    cmd.groups = FluidGpuDispatch::groups256(static_cast<uint32_t>(c.foam_count));
    if (!compute->dispatch(cmd)) return;

    gpu_buffers.foam_neigh_pending = true;
    gpu_buffers.foam_neigh_pending_count = foam_count;
    gpu_buffers.foam_neigh_pending_generation = state.foam.topology_generation;
}


bool prepareGpuWhitewaterCurrentInputs(
    SimulationGridDomainState& state, const Fluid::FoamParams& params, float dt,
    SimulationComputeContext* compute, SimulationGridDomainComputeBuffers& buffers,
    std::vector<float>& expected, std::vector<uint32_t>& neighbours,
    bool& expected_ready, bool& neighbours_ready, bool& overflow_fallback,
    const Fluid::APICSolverParams& source_params) {
    overflow_fallback = false;
    expected_ready = false;
    neighbours_ready = false;
    buffers.foam_expected_pending = false;
    buffers.foam_neigh_pending = false;
    if (!queueGpuFoamCriteria(state, params, dt, compute, buffers, expected, source_params)) {
        return false;
    }
    dispatchGpuFoamNeighbours(state, params, compute, buffers);
    expected.resize(state.particles.size());
    neighbours.resize(state.foam.size());
    uint32_t overflow = 0;
    compute->beginTransferBatch();
    bool expected_ok = compute->downloadBuffer(
        buffers.foam_expected, expected.data(), expected.size() * sizeof(float));
    bool neighbours_ok = false;
    if (buffers.foam_neigh_pending &&
        buffers.foam_neigh_pending_count == state.foam.size() &&
        buffers.foam_neigh_pending_generation == state.foam.topology_generation) {
        neighbours_ok = compute->downloadBuffer(
            buffers.foam_neigh, neighbours.data(), neighbours.size() * sizeof(uint32_t));
    }
    const bool overflow_ok = compute->downloadBuffer(buffers.foam_bin_counts,
        &overflow, sizeof(overflow), buffers.foam_bin_cell_count * sizeof(uint32_t));
    const bool batch_ok = compute->endTransferBatch();
    buffers.foam_expected_pending = false;
    buffers.foam_neigh_pending = false;
    overflow_fallback = overflow != 0u;
    if (!batch_ok || !overflow_ok || overflow_fallback) {
        if (overflow != 0u) {
            SCENE_LOG_WARN("[Whitewater] GPU bin overflow; current inputs use exact CPU gather");
        }
        return false;
    }
    expected_ready = expected_ok;
    neighbours_ready = neighbours_ok;
    return expected_ready || neighbours_ready;
}
