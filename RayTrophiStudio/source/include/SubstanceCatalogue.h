#pragma once

#include "json.hpp"

namespace RayTrophiSim {

struct SubstanceProfile;

// Catalogue families describe authoring intent, never phase or solver ownership.
nlohmann::json substanceCatalogueInfo(const SubstanceProfile& profile);

} // namespace RayTrophiSim
