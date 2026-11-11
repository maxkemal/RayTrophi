#pragma once
#include <json.hpp>
#include <string>

namespace rtapi {
nlohmann::json demBodies(const std::string& domain, const nlohmann::json& request);
}
