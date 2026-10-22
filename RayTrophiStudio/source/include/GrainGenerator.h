#pragma once

#include "Fluid/GrainMeshNear.h"
#include "json.hpp"

namespace RayTrophiSim {

// Geometry-only authoring settings. The API service owns scene insertion,
// materials, layout and optional assignment to a simulation domain.
struct GrainGeneratorSettings {
    std::string type = "voronoi";
    int count = 12;
    uint32_t seed = 1337u;
    int resolution = 4;
    float roundness = 0.0f;
    float roughness = 0.25f;
    Vec3 aspect = Vec3(1.0f);
    int arms = 6;
    int branches = 3;
    float branch_length = 0.15f;
    float thickness = 0.04f;
    int crystal_sides = 6;
    float tip_height = 0.3f;
};

bool parseGrainGeneratorSettings(const nlohmann::json& input,
    GrainGeneratorSettings& settings, std::string& error);
bool generateGrainShapes(const GrainGeneratorSettings& settings,
    std::vector<Fluid::GrainExemplar>& shapes, std::string& error);
void normalizeGrainShape(Fluid::GrainExemplar& shape);

} // namespace RayTrophiSim
