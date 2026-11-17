#pragma once

#include "json.hpp"
#include <string>

namespace rtapi {
struct Result;
Result setBodyPoses(const nlohmann::json &packet);
Result getBodyPoses(nlohmann::json &output);
Result clearBodyPoses();
} // namespace rtapi
