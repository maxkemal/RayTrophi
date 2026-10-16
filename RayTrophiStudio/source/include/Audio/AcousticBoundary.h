#pragma once
#include "json.hpp"

struct UIContext;
namespace RayTrophiSim {
struct SubstanceProfile;
}
namespace audio {
nlohmann::json acousticMediumState(const RayTrophiSim::SubstanceProfile& profile);
nlohmann::json queryAcousticBoundary(const nlohmann::json& params);
void drawAcousticBoundaryProbe(UIContext& ctx);
} // namespace audio
