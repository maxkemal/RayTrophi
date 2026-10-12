#pragma once
#include "json.hpp"
struct UIContext;
namespace AudioReactionUI {
void draw(UIContext& ctx, const nlohmann::json& emitter);
}
