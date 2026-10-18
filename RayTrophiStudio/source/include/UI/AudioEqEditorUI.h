#pragma once
#include "json.hpp"
struct UIContext;
namespace AudioUI {
bool drawParametricEqEditor(UIContext& ctx, nlohmann::json& effect);
}
