#pragma once

#include "json.hpp"

#include <string>

// Handles read-only mesh tool discovery. Returns true when the method belongs
// to this adapter, including validation errors encoded in out_result.
bool dispatchMeshToolMethod(const std::string& method,
                            const nlohmann::json& params,
                            nlohmann::json& out_result);
