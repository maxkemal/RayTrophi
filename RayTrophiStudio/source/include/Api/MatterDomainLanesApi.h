#pragma once
#include "json.hpp"
#include <string>

namespace rtapi {
nlohmann::json getMatterDomainLanes(const std::string& domain);
nlohmann::json setMatterDomainLanes(const std::string& domain, bool gas, bool liquid);
}
