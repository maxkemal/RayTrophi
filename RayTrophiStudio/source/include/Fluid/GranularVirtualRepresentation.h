#pragma once

#include "../Vec3.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace FluidSim {
class FluidGrid;
}

namespace RayTrophiSim::Fluid {

class FluidParticles;

enum class GranularParticleClass : std::uint8_t {
    HiddenBulk = 0,
    SurfaceGrain = 1,
    DetachedGrain = 2
};

struct GranularVirtualSettings {
    std::uint32_t min_particles_per_cell = 2;
    std::uint32_t support_gap_cells = 1;
    std::uint32_t surface_band_cells = 1;
    std::uint32_t height_smoothing_iterations = 1;
    float surface_offset_voxels = 0.5f;
};

struct GranularVirtualStats {
    bool measured = false;
    std::size_t simulation_particles = 0;
    std::size_t supported_particles = 0;
    std::size_t hidden_bulk_particles = 0;
    std::size_t surface_grains = 0;
    std::size_t detached_grains = 0;
    std::size_t supported_cells = 0;
    std::size_t surface_columns = 0;
    double build_ms = 0.0;
};

// CPU reference representation. It deliberately consumes the canonical flat
// particle SoA and the solver grid. GPU implementations must preserve this
// classification contract instead of inventing a renderer-specific meaning of
// "active grain".
struct GranularVirtualRepresentation {
    Vec3 bounds_min = Vec3(0.0f);
    Vec3 bounds_max = Vec3(0.0f);
    int resolution_x = 0;
    int resolution_z = 0;
    float cell_size = 0.0f;

    // X-major within each Z row: x + z * resolution_x. column_valid keeps
    // empty columns distinct from a legitimate height of zero.
    std::vector<float> height;
    std::vector<std::uint8_t> column_valid;
    std::vector<GranularParticleClass> particle_class;
    std::vector<std::uint32_t> surface_grain_indices;
    std::vector<std::uint32_t> detached_grain_indices;
    GranularVirtualStats stats;

    void clear();
};

bool buildGranularVirtualRepresentation(
    const FluidParticles& particles,
    const FluidSim::FluidGrid& grid,
    const GranularVirtualSettings& settings,
    GranularVirtualRepresentation& output,
    std::string& error);

} // namespace RayTrophiSim::Fluid
