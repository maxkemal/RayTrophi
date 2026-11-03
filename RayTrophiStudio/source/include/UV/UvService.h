// SPDX-License-Identifier: MIT
#pragma once

#include "UV/UvLayout.h"
#include "json.hpp"

namespace UV {

// Main thread only, including all read access. IPC must enqueue before calling.
bool inspect(const std::string& object, int mesh, Layout& layout, std::string& error);
uint64_t sceneRevision();
nlohmann::json request(const std::string& operation, const nlohmann::json& params);

} // namespace UV
