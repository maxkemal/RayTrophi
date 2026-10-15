#pragma once
#include "json.hpp"
namespace RayTrophiSim {
struct SubstanceProfile;
}
namespace audio {
nlohmann::json acousticMaterialResponse(const RayTrophiSim::SubstanceProfile& profile);
}
