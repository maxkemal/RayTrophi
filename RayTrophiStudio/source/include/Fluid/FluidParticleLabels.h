#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainState;
namespace Fluid {

class FluidParticles;
struct FoamParticles;

// State, not substance identity or a render setting. Zero deliberately means
// unclassified: newly emitted parcels and old disk caches have no measurement.
enum class ParticleLabel : uint8_t {
    Unknown = 0,
    Body,
    Spray,
    Foam,
    Bubble,
    Mist,
    Frozen,
    Count
};

constexpr std::size_t kParticleLabelCount = static_cast<std::size_t>(ParticleLabel::Count);
using ParticleLabelCounts = std::array<uint64_t, kParticleLabelCount>;

// Reserved in FluidParticles::flags, whose existing emit/copy/compact/reset
// lifecycle already follows the particle. Do not overlap outflow or frozen.
constexpr uint32_t kParticleLabelShift = 8u;
constexpr uint32_t kParticleLabelMask = 0xFu << kParticleLabelShift;

const char* particleLabelName(ParticleLabel label);
ParticleLabel particleLabel(uint32_t flags);
ParticleLabel secondaryParticleLabel(uint8_t foam_type);
// Number of whitewater types (FoamType: spray, foam, bubble).
constexpr std::size_t kWhitewaterTypeCount = 3;
void setParticleLabel(uint32_t& flags, ParticleLabel label);

// Provisional topology criterion, independent of display and substance name.
// Counts OTHER parcels within 1.5 simulation voxels; <=2 detaches, >=6 rejoins.
// The gap retains the previous state to avoid frame-to-frame flicker.
// Granular material is not classified as liquid spray. A parcel whose remaining
// material mass falls below the mist limit is a real low-mass droplet: it is
// excluded from the liquid-neighbour topology and routed to the fog view.
constexpr float kParticleLabelRadiusVoxels = 1.5f;
constexpr unsigned kParticleLabelDetachNeighbors = 2u;
constexpr unsigned kParticleLabelRejoinNeighbors = 6u;
constexpr float kParticleMistMassFraction = 0.15f;

struct ParticleLabelStepStats {
    bool on_gpu = false;
    double milliseconds = 0.0;
    double bin_milliseconds = 0.0;
    double classify_milliseconds = 0.0;
    uint64_t occupied_bins = 0;
    uint64_t center_resolved = 0;
    uint64_t particles = 0;
    uint64_t changed = 0;
};

struct ParticleLabelReport {
    bool available = false;
    uint64_t primary_particles = 0;
    uint64_t secondary_particles = 0;
    ParticleLabelCounts primary{};
    ParticleLabelCounts secondary{};
    ParticleLabelStepStats last_step;
};

ParticleLabelStepStats updateParticleLabels(FluidParticles& particles,
                                           float voxel_size, bool granular,
                                           const std::vector<uint32_t>* solid_tags = nullptr);
ParticleLabelCounts countParticleLabels(const FluidParticles& particles);
ParticleLabelCounts countSecondaryParticleLabels(const FoamParticles& particles);
ParticleLabelReport inspectParticleLabels(const SimulationGridDomainState* state);

} // namespace Fluid
} // namespace RayTrophiSim
