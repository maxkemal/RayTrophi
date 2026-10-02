#pragma once

#include <cstdint>
#include <memory>
#include <string>

class TriangleMesh;

namespace RayTrophiSim::Fluid {

struct GranularVirtualRepresentation;

// Creates or updates a fixed-topology, flat-SoA heightfield mesh. Empty cells
// become degenerate quads, so changing coverage never changes topology.
bool updateGranularVirtualSurface(
    const GranularVirtualRepresentation& representation,
    std::uint16_t material_id,
    const std::string& node_name,
    std::shared_ptr<TriangleMesh>& mesh,
    bool& topology_changed,
    std::string& error);

} // namespace RayTrophiSim::Fluid
