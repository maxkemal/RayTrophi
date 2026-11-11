#pragma once
#include "Api/RtApi.h"
#include "json.hpp"

namespace rtapi {
// Empty source: canonical ground-aligned default. Named source: current flat
// world-vertex bounds, never compatibility Triangle facades or cached selection poses.
Result createPlacedMatterDomain(const std::string &name, const std::string &source,
                                nlohmann::json &output);
std::string selectedFlatDomainSource();
} // namespace rtapi
