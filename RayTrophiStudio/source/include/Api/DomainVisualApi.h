#pragma once

#include "Api/RtApi.h"
#include "json.hpp"

namespace rtapi {
Result getDomainVisualSamples(const std::string& domain, nlohmann::json& output,
    uint64_t offset = 0, uint64_t limit = 0);
Result setDomainVisualSamples(const std::string& domain, const nlohmann::json& patch);
Result resetDomainVisualSamples(const std::string& domain);
}
