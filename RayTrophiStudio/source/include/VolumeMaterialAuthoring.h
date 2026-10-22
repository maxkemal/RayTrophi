#pragma once

#include "VolumeShader.h"

#include <string>

namespace volume_material {

nlohmann::json inspect(const VolumeShader& shader);
// Atomic overlay: unknown keys, non-finite values and invalid ranges fail.
bool configure(VolumeShader& shader, const nlohmann::json& patch, std::string& error);

} // namespace volume_material
