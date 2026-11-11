#pragma once

#include "Api/RtApi.h"
#include "json.hpp"

namespace rtapi {
Result getDomainWallContact(const std::string& domain, nlohmann::json& output);
Result setDomainWallContact(const std::string& domain, const nlohmann::json& patch,
    nlohmann::json& output);
}
