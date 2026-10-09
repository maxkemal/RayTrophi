#include "Fluid/MatterGpuModelView.h"

#include <cstring>
#include <string_view>
#include <vector>

namespace RayTrophiSim::Fluid {

bool dispatchMatterGpuModel(SimulationComputeContext& compute,
    const ComputeDispatch& command, const MatterGpuModelView& model) {
    if (!model.enabled) {
        return compute.dispatch(command);
    }
    const std::string_view name = command.kernel ? command.kernel : "";
    // The compact twin takes the same boundary word and the same rule.
    if (name == "sim_fluid_zero_solid_faces" || name == "sim_sparse_mac_zero_faces") {
        if (command.constants_size != 36) {
            return false;
        }
        std::vector<unsigned char> constants(command.constants_size);
        std::memcpy(constants.data(), command.constants, command.constants_size);
        std::memcpy(constants.data() + 16, &model.boundary, sizeof(model.boundary));
        auto boundary = command;
        boundary.kernel = name == "sim_fluid_zero_solid_faces"
            ? "sim_matter_zero_faces" : "sim_sparse_mac_matter_zero_faces";
        boundary.constants = constants.data();
        return compute.dispatch(boundary);
    }
    const char* variant = nullptr;
    if (name == "sim_sparse_mac_p2g") {
        variant = "sim_sparse_mac_matter_p2g";
    }
    else if (name == "sim_sparse_mac_g2p") {
        variant = "sim_sparse_mac_matter_g2p";
    }
    else if (name == "sim_fluid_p2g_scatter") {
        variant = "sim_matter_p2g";
    }
    else if (name == "sim_fluid_g2p") {
        variant = "sim_matter_g2p";
    }
    else if (name == "sim_fluid_granular_stress_update") {
        variant = "sim_matter_stress_update";
    }
    else if (name == "sim_fluid_granular_stress_p2g") {
        variant = "sim_matter_stress_p2g";
    }
    else if (name == "sim_fluid_granular_settle") {
        variant = "sim_matter_settle";
    }
    else if (name == "sim_fluid_advect_tail") {
        variant = "sim_matter_advect";
    }
    else if (name == "sim_sparse_mac_advect") {
        variant = "sim_sparse_mac_matter_advect";
    }
    else if (name == "sim_fluid_occupancy") {
        int component = 0;
        if (command.constants_size < 20) {
            return false;
        }
        std::memcpy(&component, static_cast<const char*>(command.constants) + 16, 4);
        if (component != 0) {
            variant = "sim_matter_occupancy";
        }
    }
    if (!variant) {
        return compute.dispatch(command);
    }
    if (!model.indices.valid() || !model.counters.valid() || model.lane > 1) {
        return false;
    }
    std::vector<ComputeBufferHandle> buffers(command.buffers,
        command.buffers + command.buffer_count);
    buffers.push_back(model.indices);
    buffers.push_back(model.counters);
    if (name == "sim_fluid_granular_stress_update") {
        if (!model.wet_response.valid()) {
            return false;
        }
        buffers.push_back(model.wet_response);
    }
    if (name == "sim_fluid_p2g_scatter" || name == "sim_sparse_mac_p2g" ||
        name == "sim_fluid_granular_stress_p2g") {
        if (!model.rest_mass.valid() || !model.mass_fraction.valid()) {
            return false;
        }
        buffers.push_back(model.rest_mass);
        buffers.push_back(model.mass_fraction);
    }
    if (name == "sim_fluid_granular_stress_p2g") {
        if (!model.dry_volume.valid()) {
            return false;
        }
        buffers.push_back(model.dry_volume);
    }
    if (name == "sim_fluid_p2g_scatter" || name == "sim_sparse_mac_p2g") {
        int component = 0;
        std::memcpy(&component, static_cast<const char*>(command.constants) + 16, 4);
        if (component < 0 || component > 2 || !model.mass_gradient[component].valid()) {
            return false;
        }
        buffers.push_back(model.mass_gradient[component]);
    }
    std::vector<unsigned char> constants(command.constants_size + sizeof(uint32_t));
    std::memcpy(constants.data(), command.constants, command.constants_size);
    std::memcpy(constants.data() + command.constants_size, &model.lane, sizeof(model.lane));
    auto indexed = command;
    indexed.kernel = variant;
    indexed.buffers = buffers.data();
    indexed.buffer_count = buffers.size();
    indexed.constants = constants.data();
    indexed.constants_size = constants.size();
    return compute.dispatch(indexed);
}

} // namespace RayTrophiSim::Fluid
