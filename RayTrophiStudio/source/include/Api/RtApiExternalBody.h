#pragma once
#include "json.hpp"
namespace rtapi {
struct Result;
Result setExternalBodyPose(const nlohmann::json &request);
Result getExternalBodyPoses(nlohmann::json &output);
Result clearExternalBodyPose(const nlohmann::json &request);
} // namespace rtapi
