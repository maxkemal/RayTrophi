#pragma once

#include <cstddef>
#include <vector>

namespace RayTrophiSim { struct SurfaceMeshTriangle; }
namespace FluidSim { class FluidGrid; }
namespace RayTrophiSim::Fluid {
class FluidParticles;

// Flat exposed-cell surface of pinned frozen liquid. Appends to the collider
// face list; vertices use integer grid nodes so adjoining faces match bitwise.
std::size_t appendMatterIceColliderFaces(const FluidParticles& particles,
    const FluidSim::FluidGrid& grid, std::vector<SurfaceMeshTriangle>& faces);
}
