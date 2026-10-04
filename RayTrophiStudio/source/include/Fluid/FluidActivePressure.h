#pragma once

#include "FluidActiveWindow.h"

#include <string_view>

namespace RayTrophiSim::Fluid::ActiveWindow {

inline Bounds pressureMaskBounds(const std::vector<float>& mask, int nx, int ny, int nz) {
    const Bounds fallback = full(nx, ny, nz);
    if (nx <= 0 || ny <= 0 || nz <= 0 || mask.size() < fallback.cells()) {
        return fallback;
    }
    Bounds result = fallback;
    result.begin[0] = nx;
    result.begin[1] = ny;
    result.begin[2] = nz;
    result.end[0] = result.end[1] = result.end[2] = 0;
    for (uint64_t index = 0; index < fallback.cells(); ++index) {
        if (!std::isfinite(mask[index])) {
            return fallback;
        }
        if (mask[index] < 0.5f) {
            continue;
        }
        const int cell[3] = {
            static_cast<int>(index % nx),
            static_cast<int>((index / nx) % ny),
            static_cast<int>(index / (uint64_t(nx) * ny))
        };
        for (int axis = 0; axis < 3; ++axis) {
            result.begin[axis] = std::min(result.begin[axis], cell[axis]);
            result.end[axis] = std::max(result.end[axis], cell[axis] + 1);
        }
    }
    const int dims[3] = {nx, ny, nz};
    for (int axis = 0; axis < 3; ++axis) {
        if (result.end[axis] <= result.begin[axis]) {
            return fallback;
        }
        result.begin[axis] = std::max(0, result.begin[axis] - 1);
        result.end[axis] = std::min(dims[axis], result.end[axis] + 1);
    }
    result.bounded = result.cells() < fallback.cells();
    return result;
}

// Keep the established 52-byte CUDA/full-grid ABI. Only window variants receive
// the appended bounds; scalar reductions and cold-start clears use the old ABI.
template <typename ProjectionConstants>
struct PressureConstants {
    ProjectionConstants projection;
    int begin_x, begin_y, begin_z;
    int extent_x, extent_y, extent_z;
};

inline const char* pressureVariant(const char* kernel) {
    struct Entry {
        const char* full;
        const char* window;
    };
    static constexpr Entry entries[] = {
        {"sim_fluid_divergence", "sim_fluid_divergence_window"},
        {"sim_fluid_divergence_var", "sim_fluid_divergence_var_window"},
        {"sim_fluid_cg_build_diag", "sim_fluid_cg_build_diag_window"},
        {"sim_fluid_cg_build_diag_var", "sim_fluid_cg_build_diag_var_window"},
        {"sim_fluid_cg_spmv", "sim_fluid_cg_spmv_window"},
        {"sim_fluid_cg_spmv_var", "sim_fluid_cg_spmv_var_window"},
        {"sim_fluid_cg_jacobi", "sim_fluid_cg_jacobi_window"},
        {"sim_fluid_cg_copy", "sim_fluid_cg_copy_window"},
        {"sim_fluid_cg_axpy", "sim_fluid_cg_axpy_window"},
        {"sim_fluid_cg_zpby", "sim_fluid_cg_zpby_window"},
        {"sim_fluid_cg_dot", "sim_fluid_cg_dot_window"},
        {"sim_fluid_cg_axpy_dev", "sim_fluid_cg_axpy_dev_window"},
        {"sim_fluid_cg_zpby_dev", "sim_fluid_cg_zpby_dev_window"},
        {"sim_fluid_cg_jacobi_dot", "sim_fluid_cg_jacobi_dot_window"},
        {"sim_fluid_cg_spmv_dot", "sim_fluid_cg_spmv_dot_window"},
        {"sim_fluid_cg_spmv_dot_var", "sim_fluid_cg_spmv_dot_var_window"},
        {"sim_fluid_cg_axpy2_dev", "sim_fluid_cg_axpy2_dev_window"}
    };
    for (const auto& entry : entries) {
        if (std::string_view(kernel) == entry.full) {
            return entry.window;
        }
    }
    return nullptr;
}

class PressureDispatch {
public:
    explicit PressureDispatch(const Bounds& bounds) : bounds_(bounds) {
    }

    uint32_t groups(uint32_t full_groups) const {
        return bounds_.bounded ? static_cast<uint32_t>((bounds_.cells() + 255) / 256)
                               : full_groups;
    }

    template <typename ProjectionConstants>
    bool dispatch(SimulationComputeContext& compute, ComputeDispatch command,
                  const ProjectionConstants& projection) const {
        static_assert(sizeof(ProjectionConstants) == 52, "Full pressure ABI");
        static_assert(sizeof(PressureConstants<ProjectionConstants>) == 76,
                      "Window pressure ABI");
        if (!command.kernel) {
            return false;
        }
        const char* variant = bounds_.bounded ? pressureVariant(command.kernel) : nullptr;
        if (!variant) {
            return compute.dispatch(command);
        }
        const PressureConstants<ProjectionConstants> constants = {
            projection, bounds_.begin[0], bounds_.begin[1], bounds_.begin[2],
            bounds_.end[0] - bounds_.begin[0], bounds_.end[1] - bounds_.begin[1],
            bounds_.end[2] - bounds_.begin[2]
        };
        command.kernel = variant;
        command.constants = &constants;
        command.constants_size = sizeof(constants);
        command.groups.groups_x = groups(command.groups.groups_x);
        return compute.dispatch(command);
    }

private:
    Bounds bounds_;
};

inline bool uploadPreparedMask(SimulationComputeContext& compute,
                               ComputeBufferHandle target,
                               const std::vector<float>& host_mask,
                               size_t cells, bool device_mask_valid) {
    if (!target.valid() || compute.getBufferSize(target) < cells * sizeof(float)) {
        return false;
    }
    if (device_mask_valid) {
        return true;
    }
    return host_mask.size() >= cells &&
        compute.uploadBuffer(target, host_mask.data(), cells * sizeof(float));
}

} // namespace RayTrophiSim::Fluid::ActiveWindow
