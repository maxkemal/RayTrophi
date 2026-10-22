#pragma once

#include "Api/RtApi.h"
#include "json.hpp"

namespace rtapi {
// Creates editable, persistent flat scene meshes as one undo operation.
Result generateGrains(const nlohmann::json& settings, nlohmann::json& output);
}
