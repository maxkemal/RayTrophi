#pragma once

#include "Api/RtApi.h"
#include "json.hpp"

namespace rtapi {
Result validateSingleRegionSeedRequest(const std::string &domain);
// list/add/update/remove/settings/select/apply. Editing only changes the recipe; apply is the
// explicit transactional replacement of live particles and arms reset replay.
Result matterSeedSources(const std::string &domain, const std::string &operation, uint64_t id,
                         const nlohmann::json &settings, nlohmann::json &output);
} // namespace rtapi
