// Included inside the simulation driver's private namespace before the transfer stages.
// Keep the weighted float mask ABI: air=0, solid=-1, fluid=sum(mass_fraction).
static_assert(sizeof(FluidP2GGpuConstants) == 36, "GPU occupancy push-constant ABI");
class FluidGpuOccupancy {
public:
    FluidGpuOccupancy(SimulationGridDomainState& state,
                      SimulationComputeContext* compute,
                      SimulationGridDomainComputeBuffers* buffers)
        : state_(state), compute_(compute), buffers_(buffers) {
    }

    bool build() {
        if (failed_ || !compute_ || !buffers_ ||
            !FluidGpuParticleUpload::canReuse(*buffers_, *compute_, state_.particles.size())) {
            return false;
        }
        const auto& grid = state_.grid;
        const std::size_t cells = grid.getCellCount();
        if (cells == 0 || cells > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            state_.particles.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            grid.solid.size() < cells || !buffers_->fluid_mask.valid() ||
            !(grid.voxel_size > 1.0e-6f)) {
            failed_ = true;
            return false;
        }
        if (!initialized_ && !initialize()) {
            failed_ = true;
            return false;
        }

        FluidP2GGpuConstants constants;
        constants.nx = grid.nx;
        constants.ny = grid.ny;
        constants.nz = grid.nz;
        constants.particle_count = static_cast<int>(state_.particles.size());
        constants.component = 3; // Existing scalar clear, not a MAC face clear.
        constants.origin_x = grid.origin.x;
        constants.origin_y = grid.origin.y;
        constants.origin_z = grid.origin.z;
        constants.voxel_size = grid.voxel_size;
        ComputeDispatch cmd;
        cmd.kernel = "sim_fluid_clear_float";
        cmd.buffers = &buffers_->fluid_mask;
        cmd.buffer_count = 1;
        cmd.constants = &constants;
        cmd.constants_size = sizeof(constants);
        cmd.groups.groups_x = (static_cast<uint32_t>(cells) + 255u) / 256u;
        bool ok = Fluid::dispatchMatterGpuModel(*compute_, cmd, buffers_->matter_model);

        const ComputeBufferHandle bindings[] = {
            buffers_->fluid_positions, buffers_->fluid_mass_fraction,
            buffers_->fluid_mask_solid_indices, buffers_->fluid_mask
        };
        cmd.kernel = "sim_fluid_occupancy";
        cmd.buffers = bindings;
        cmd.buffer_count = 4;
        if (solid_count_ > 0) {
            constants.component = 0;
            constants.particle_count = static_cast<int>(solid_count_);
            cmd.groups.groups_x = (solid_count_ + 255u) / 256u;
            ok = ok && Fluid::dispatchMatterGpuModel(*compute_, cmd, buffers_->matter_model);
        }
        constants.component = 1;
        constants.particle_count = static_cast<int>(state_.particles.size());
        cmd.groups.groups_x = (static_cast<uint32_t>(constants.particle_count) + 255u) / 256u;
        ok = ok && Fluid::dispatchMatterGpuModel(*compute_, cmd, buffers_->matter_model);
        failed_ = !ok;
        return ok;
    }

private:
    bool initialize() {
        const auto& grid = state_.grid;
        const std::size_t cells = grid.getCellCount();
        const std::vector<uint32_t>* solids = &grid.solid_cells;
        std::vector<uint32_t> scanned_solids;
        if (!grid.solid_cells_valid) {
            for (std::size_t cell = 0; cell < cells; ++cell) {
                if (grid.solid[cell]) {
                    scanned_solids.push_back(static_cast<uint32_t>(cell));
                }
            }
            solids = &scanned_solids;
        }
        if (solids->size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return false;
        }
        solid_count_ = static_cast<uint32_t>(solids->size());
        const std::size_t bytes = std::max<std::size_t>(1, solid_count_) * sizeof(uint32_t);
        auto& handle = buffers_->fluid_mask_solid_indices;
        if (!handle.valid() || handle.backend != compute_->backendType() ||
            compute_->getBufferSize(handle) == 0) {
            ComputeBufferDesc desc;
            desc.debug_name = "FluidMaskSolidIndices";
            desc.size_bytes = bytes;
            desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                         ComputeBufferUsage::ReadOnly;
            handle = compute_->createBuffer(desc);
        } else if (compute_->getBufferSize(handle) < bytes &&
                   !compute_->resizeBuffer(handle, bytes)) {
            return false;
        }
        if (!handle.valid()) {
            return false;
        }
        // Upload once per frame, before any occupancy dispatch. The existing
        // collider voxelization happens before the elastic substep loop.
        if (solid_count_ > 0) {
            compute_->beginTransferBatch();
            const bool uploaded = compute_->uploadBuffer(
                handle, solids->data(), solid_count_ * sizeof(uint32_t));
            if (!compute_->endTransferBatch() || !uploaded) {
                return false;
            }
        }
        initialized_ = true;
        return true;
    }

    SimulationGridDomainState& state_;
    SimulationComputeContext* compute_ = nullptr;
    SimulationGridDomainComputeBuffers* buffers_ = nullptr;
    uint32_t solid_count_ = 0;
    bool initialized_ = false;
    bool failed_ = false;
};
