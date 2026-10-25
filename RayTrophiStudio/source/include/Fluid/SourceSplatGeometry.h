#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {

// Render authoring only. Neither geometry nor camera LOD changes collision shapes.
struct SourceSplatGeometry {
    std::string mode = "sphere"; // sphere | scene_object | mesh_grains | domain (legacy)
    std::vector<std::string> objects;
    bool source_materials = true;
    float near_distance = 2.0f;
    uint32_t mesh_budget = 200000;
};

} // namespace RayTrophiSim::Fluid
