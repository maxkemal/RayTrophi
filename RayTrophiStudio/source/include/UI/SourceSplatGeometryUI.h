#pragma once
#include <string>
struct SceneData;
namespace RayTrophiSim::Fluid {
void drawSubstanceSplatGeometry(SceneData& scene, const std::string& domain,
    const std::string& substance);
}
