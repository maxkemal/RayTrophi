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
    std::string producer_name;
    bool visible = true;
    std::shared_ptr<VolumeShader> shader;
};

using MatterFogAppearances = std::map<FogFieldKey, MatterFogAppearance>;

struct MatterFogLiveStream {
    int vdb_id = -1;
    std::shared_ptr<VDBVolume> volume;
    bool populated = false;
    bool optically_populated = true;
    std::string optical_bounds_key;
    uint64_t state_version = ~uint64_t(0);
    // Set on the one stream that carries merged gas (see MatterFogPublisher.cpp).
    // Its density is already physical, so it renders with this neutral shader.
    std::shared_ptr<VolumeShader> merged_shader;
    std::string gas_merge_key;
    std::string gas_merge_requested;
};

using MatterFogLiveStreams = std::map<FogFieldKey, MatterFogLiveStream>;

std::string fogFieldId(const FogFieldKey& key);
std::string fogFieldName(const FogFieldKey& key);
nlohmann::json writeFogAppearances(const MatterFogAppearances& appearances);
MatterFogAppearances readFogAppearances(const nlohmann::json& domain);
MatterFogAppearance& ensureFogAppearance(MatterFogAppearances& appearances,
    const FogFieldKey& key, const std::shared_ptr<VolumeShader>& legacy);
std::shared_ptr<VolumeShader>& matterGasFogShader(SimulationGridDomainDesc& domain);
// One sentence for the panel and scripting: why gas renders merged or separate.
std::string matterGasMergeStatus(const MatterFogAppearances& appearances);

} // namespace RayTrophiSim::Fluid
