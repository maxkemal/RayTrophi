#pragma once

#include "Fluid/MatterFogFields.h"
#include "json.hpp"
#include <map>
#include <memory>

class VolumeShader;
class VDBVolume;

namespace RayTrophiSim { struct SimulationGridDomainDesc; }
namespace RayTrophiSim::Fluid {

struct MatterFogAppearance {
    bool visible = true;
    std::shared_ptr<VolumeShader> shader;
};

using MatterFogAppearances = std::map<FogFieldKey, MatterFogAppearance>;

struct MatterFogLiveStream {
    int vdb_id = -1;
    std::shared_ptr<VDBVolume> volume;
    bool populated = false;
    uint64_t state_version = ~uint64_t(0);
};

using MatterFogLiveStreams = std::map<FogFieldKey, MatterFogLiveStream>;

std::string fogFieldId(const FogFieldKey& key);
std::string fogFieldName(const FogFieldKey& key);
nlohmann::json writeFogAppearances(const MatterFogAppearances& appearances);
MatterFogAppearances readFogAppearances(const nlohmann::json& domain);
MatterFogAppearance& ensureFogAppearance(MatterFogAppearances& appearances,
    const FogFieldKey& key, const std::shared_ptr<VolumeShader>& legacy);
std::shared_ptr<VolumeShader>& matterGasFogShader(SimulationGridDomainDesc& domain);

} // namespace RayTrophiSim::Fluid
