#pragma once
#include "Api/RtApi.h"
#include "json.hpp"
namespace rtapi {
Result playerOperator(const std::string &operation, const nlohmann::json &config,
                      nlohmann::json &output);
}
