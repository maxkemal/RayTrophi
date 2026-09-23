#pragma once

#include "SimulationCompute.h"

#include <array>
#include <cstdint>
#include <limits>

namespace RayTrophiSim::FluidGpuFlipSnapshot {

// sim_fluid_cg_copy copies a float array. Its 52-byte pressure constants only
// use nx * ny * nz for this kernel, so a face array fits as (face_count, 1, 1).
struct CopyConstants {
    int32_t count = 0;
    int32_t height = 1;
    int32_t depth = 1;
    int32_t unused[10] = {};
};
static_assert(sizeof(CopyConstants) == 52);

inline bool copyFaceFields(
    SimulationComputeContext& compute,
    const std::array<ComputeBufferHandle, 3>& source,
    const std::array<ComputeBufferHandle, 3>& destination,
    const std::array<std::size_t, 3>& face_counts) {
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch()) {
        return false;
    }

    for (std::size_t component = 0; component < source.size(); ++component) {
        const std::size_t count = face_counts[component];
        if (count == 0 ||
            count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) ||
            !source[component].valid() || !destination[component].valid() ||
            source[component].backend != ComputeBackendType::VulkanCompute ||
            destination[component].backend != ComputeBackendType::VulkanCompute ||
            compute.getBufferSize(source[component]) < count * sizeof(float) ||
            compute.getBufferSize(destination[component]) < count * sizeof(float)) {
            return false;
        }
    }

    for (std::size_t component = 0; component < source.size(); ++component) {
        CopyConstants constants;
        constants.count = static_cast<int32_t>(face_counts[component]);
        const ComputeBufferHandle buffers[] = {
            destination[component], source[component]
        };
        ComputeDispatch command;
        command.kernel = "sim_fluid_cg_copy";
        command.buffers = buffers;
        command.buffer_count = 2;
        command.constants = &constants;
        command.constants_size = sizeof(constants);
        command.groups.groups_x =
            (static_cast<uint32_t>(constants.count) + 255u) / 256u;
        if (!compute.dispatch(command)) {
            // A previous component may already be queued. Drain it before the
            // caller's host-upload fallback replaces all scratch fields.
            compute.synchronize();
            return false;
        }
    }
    // Pressure can immediately upload CPU boundary values into the source
    // buffers. On host-visible Vulkan memory that upload is a direct memcpy,
    // so finish these copies before pressure is allowed to overwrite them.
    compute.synchronize();
    return true;
}

} // namespace RayTrophiSim::FluidGpuFlipSnapshot
