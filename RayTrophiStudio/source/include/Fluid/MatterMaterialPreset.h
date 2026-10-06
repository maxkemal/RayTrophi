#pragma once

#include <memory>
#include <string>

class Material;

namespace RayTrophiSim::Fluid {

// Visual suggestions only. Never authors temperature, phase or solver properties.
std::shared_ptr<Material> createMatterMaterialPreset(const std::string& substance,
                                                    std::string& error);

} // namespace RayTrophiSim::Fluid
