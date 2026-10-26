#pragma once
#include "Fluid/GasSourceProvenance.h"
#include <iosfwd>
namespace RayTrophiSim::Fluid {
bool writeGasSourceCache(std::ostream& stream, const SimulationGridDomainState& state);
bool readGasSourceCache(std::istream& stream, SimulationGridDomainState& state);
}
