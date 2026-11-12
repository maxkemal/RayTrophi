#pragma once

#include "Fluid/DomainVisualSettings.h"
#include "Fluid/DomainVisualRuntime.h"
#include "Fluid/FluidParticles.h"

#include <iosfwd>
#include <map>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;

namespace Fluid {

struct DomainVisualStream {
    uint64_t birth_source = 0;
    uint32_t tag = 0;
    uint32_t flags = 0;
    uint8_t model = 0;
    DomainFields::VisualRuntime runtime;
};

struct DomainVisualState {
    uint64_t settings_key = 0;
    uint64_t state_version = 0;
    double physical_time = 0.0;
    std::map<uint64_t, DomainVisualStream> streams;
    // Render adapter only. Never passed to the physical solver, fog or SDF.
    FluidParticles render_particles;
    std::vector<float> diameters;
    std::vector<Vec3> rotations;
    std::string status = "not published";
    bool current = false;
    bool render_ready = false;
    uint64_t sample_count = 0;
    double unrepresented_volume = 0.0;
    double represented_volume = 0.0;
    double update_ms = 0.0;
};

// Main-thread service. A published handle is immutable and is retained by RAM
// timeline snapshots. An error publishes diagnostics, never a partial new state.
void advanceDomainVisualSamples(const SimulationGridDomainDesc& domain,
    SimulationGridDomainState& state, double dt);
const FluidParticles& domainSplatParticles(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState& state);
float domainSplatDiameter(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState& state, std::size_t index, float fallback);
Vec3 domainSplatRotation(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState& state, std::size_t index, const Vec3& fallback);
nlohmann::json domainVisualStats(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState* state);
bool writeDomainVisualCache(std::ostream& output, const SimulationGridDomainState& state);
bool readDomainVisualCache(std::istream& input, SimulationGridDomainState& state);
std::size_t domainVisualAllocationBytes(const SimulationGridDomainState& state);

} // namespace Fluid
} // namespace RayTrophiSim
