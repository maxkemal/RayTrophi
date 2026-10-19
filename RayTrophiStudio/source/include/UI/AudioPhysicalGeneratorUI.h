#pragma once
#include "json.hpp"
struct UIContext;

namespace AudioUI {
void drawPhysicalGenerator(UIContext& ctx, const nlohmann::json& source);
}
