#pragma once
#include "json.hpp"
struct UIContext;
namespace AudioUI {
void drawMixer(UIContext& ctx);
void drawInspectorMixerLink(UIContext& ctx, const nlohmann::json& source);
}
