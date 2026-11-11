#pragma once

#include "DemBondMechanics.h"
#include <json.hpp>
#include <map>
#include <vector>

namespace RayTrophiSim::Fluid {
class FluidParticles;
struct MatterGrainParams;
struct APICSolverParams;

namespace DemBond {
struct Node {
    uint64_t body = 0;
    Quaternion orientation{1.0, 0.0, 0.0, 0.0};
};

struct Edge {
    State state;
    Material material;
    double elastic_energy_j = 0.0;
};

// Canonical simulation state, copied with particles and serialized by SimCache.
// Indices are intentionally absent: sort/compact operations cannot change bonds.
struct Network {
    std::map<uint64_t, Node> nodes;
    std::vector<Edge> edges;
    uint64_t next_body = 1;
    double retired_energy_j = 0.0;
};
bool gpuMaterialRepresentable(const Material& material);

// Explicit runtime authoring only. Does not bond future births or adjacent bodies.
// All operations reject transactionally; IDs must select current DEM carriers.
bool bindBody(FluidParticles& particles, const MatterGrainParams& params,
              const std::vector<uint64_t>& ids, const Material& material,
              double reach_ratio, double section_ratio, uint64_t byte_budget,
              uint64_t& body, std::string& error);
bool removeBody(Network& network, uint64_t body, std::string& error);
void retain(Network& network, const std::vector<uint64_t>& live_ids);
void retireNonDem(FluidParticles& particles, const APICSolverParams& params);
nlohmann::json encode(const Network& network);
bool validate(const Network& network, std::string& error);
bool decode(const nlohmann::json& json, Network& network, std::string& error);
nlohmann::json summary(const Network& network);
} // namespace DemBond
} // namespace RayTrophiSim::Fluid
