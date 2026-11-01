#pragma once

#include "json.hpp"
#include <string>

namespace rtapi {
nlohmann::json timelineTransport();
nlohmann::json setTimelinePlaying(bool playing);
nlohmann::json setSimulationMode(const std::string& mode);
}
