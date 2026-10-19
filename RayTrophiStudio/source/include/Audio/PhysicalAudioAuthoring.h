#pragma once
#include "json.hpp"

namespace audio {
nlohmann::json physicalGeneratorCatalogue();
nlohmann::json physicalGeneratorPatch(const nlohmann::json& source,
                                       const nlohmann::json& params);
}
