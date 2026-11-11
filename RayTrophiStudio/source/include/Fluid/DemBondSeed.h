#pragma once
#include "DemBondNetwork.h"

namespace RayTrophiSim::Fluid::DemBond {
struct SeedSettings {
    bool enabled = false;
    Material material{1.0e6, 0.25, 1.0, 1.0e4, 1.0e4};
    double reach_ratio = 1.25;
    double section_ratio = 0.5;
};
nlohmann::json seedJson(const SeedSettings& settings);
bool parseSeed(const nlohmann::json& patch, SeedSettings& settings, std::string& error);
bool bindSeed(FluidParticles& particles, std::size_t begin, const MatterGrainParams& params,
              const SeedSettings& settings, uint64_t budget, std::string& error);
}
