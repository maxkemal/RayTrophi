#pragma once
#include "Vec3.h"
#include "imgui.h"
#include <functional>
namespace RayTrophiSim {
struct SimulationGridDomainDesc;
namespace Fluid {
void drawMatterSeedSourcesControls(const SimulationGridDomainDesc &domain);
void drawMatterSeedSourceBounds(
    const SimulationGridDomainDesc &domain,
    const std::function<void(const Vec3 &, const Vec3 &, ImU32, float)> &draw_box);
} // namespace Fluid
} // namespace RayTrophiSim
