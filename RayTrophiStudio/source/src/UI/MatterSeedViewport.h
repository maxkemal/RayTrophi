#pragma once
#include "Vec3.h"
#include "imgui.h"
#include <cstdint>
#include <functional>
#include <string>
struct UIContext;

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
namespace Fluid {
uint64_t selectedMatterSeed(const SimulationGridDomainDesc &domain);
void selectMatterSeed(const std::string &domain, uint64_t id);
void drawMatterSeedLabels(UIContext &ctx, const SimulationGridDomainDesc &domain,
                          const std::function<bool(const Vec3 &, ImVec2 &)> &project);
void drawMatterSeedAuthoring(
    UIContext &ctx, const std::function<bool(const Vec3 &, ImVec2 &)> &project,
    const std::function<void(const Vec3 &, const Vec3 &, ImU32, float)> &draw_box);
// Returns true while a seed owns this domain's gizmo, including a non-box
// source (the mesh object's transform remains the authoritative volume).
bool manipulateMatterSeed(const SimulationGridDomainDesc &domain, const float *view,
                          const float *projection, bool scale, bool blocked);
bool manipulateSelectedMatterSeed(UIContext &ctx, const float *view, const float *projection,
                                  bool blocked);
} // namespace Fluid
} // namespace RayTrophiSim
