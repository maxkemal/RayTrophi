#pragma once

#include "json.hpp"
#include <vector>

namespace RayTrophiSim {

struct SubstanceProfile;

// Catalogue families describe authoring intent, never phase or solver ownership.
void appendFoundationSubstances(std::vector<SubstanceProfile>& profiles);
nlohmann::json substanceCatalogueInfo(const SubstanceProfile& profile);

} // namespace RayTrophiSim
