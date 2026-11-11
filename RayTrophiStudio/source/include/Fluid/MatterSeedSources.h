#pragma once

#include "Vec3.h"
#include "json.hpp"
#include "DemBondSeed.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct SimulationFlowSourceDesc;
namespace Fluid {
class MeshVolumeTest;

struct MatterSeedSource {
    uint64_t id = 0;
    std::string name;
    bool enabled = true;
    std::string substance = "Water";
    Vec3 bounds_min = Vec3(-0.5f, 0.0f, -0.5f);
    Vec3 bounds_max = Vec3(0.5f, 0.5f, 0.5f);
    std::string mesh_object;
    Vec3 velocity = Vec3(0.0f);
    // material: domain ambient with the existing cold-solid birth rule.
    // domain: literal domain ambient. custom: the explicit Kelvin value.
    std::string temperature_mode = "material";
    float temperature_kelvin = 293.15f;
    float grain_saturation = 0.0f;
    DemBond::SeedSettings bond_body;
};

using MatterSeedMeshResolver =
    std::function<std::shared_ptr<const MeshVolumeTest>(const std::string &)>;

nlohmann::json matterSeedSourceToJson(const MatterSeedSource &source);
uint64_t matterSeedSourceId(const nlohmann::json& value);
uint64_t matterSeedSourcesHash(const std::vector<MatterSeedSource> &sources);
uint64_t matterSeedRecipeHash(const SimulationGridDomainDesc &domain);
uint64_t matterSeedAuthoringSignature(const SimulationGridDomainDesc &domain);
bool parseMatterSeedSource(const nlohmann::json &patch, MatterSeedSource &source,
                           std::string &error, bool loading = false);
bool validateMatterSeedSources(const SimulationGridDomainDesc &domain, std::string &error,
                               const std::vector<SimulationFlowSourceDesc> *flows = nullptr,
                               int domain_index = -1);
float matterSeedBirthKelvin(const MatterSeedSource &source, const SimulationGridDomainDesc &domain,
                            float ambient_kelvin);

void writeMatterSeedSources(nlohmann::json &json, const SimulationGridDomainDesc &domain);
void readMatterSeedSources(const nlohmann::json &json, SimulationGridDomainDesc &domain);
bool matterSeedSourcesUseObject(const SimulationGridDomainDesc &domain, const std::string &name);

// Transactional replacement: all enabled layers succeed or live particles stay.
// Uses the same seedBox/DEM birth, substance identity, mass and settle primitives.
bool applyMatterSeedSources(SimulationGridDomainDesc &domain, SimulationGridDomainState &state,
                            float ambient_kelvin, const MatterSeedMeshResolver &meshes,
                            const std::vector<SimulationFlowSourceDesc> *flows = nullptr,
                            int domain_index = -1);

} // namespace Fluid
} // namespace RayTrophiSim
