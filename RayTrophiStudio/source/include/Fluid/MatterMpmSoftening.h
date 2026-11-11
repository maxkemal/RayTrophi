#pragma once

#include <cstddef>

namespace RayTrophiSim::Fluid {
class FluidParticles;
struct APICSolverParams;

// Once per frame, after heat/conduction and before CPU/GPU stress dispatch.
// Material parameters match the shared MPM law; liquids and DEM are untouched.
std::size_t updateMatterMpmSoftening(FluidParticles& particles, const APICSolverParams& params);

} // namespace RayTrophiSim::Fluid
