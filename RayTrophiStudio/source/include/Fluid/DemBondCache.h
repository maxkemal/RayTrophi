#pragma once
#include "DemBondNetwork.h"
#include <iosfwd>

namespace RayTrophiSim::Fluid::DemBond {
bool writeCache(std::ostream& stream, const Network& network);
bool readCache(std::istream& stream, const std::vector<uint64_t>& particle_ids,
               Network& network);
} // namespace RayTrophiSim::Fluid::DemBond
