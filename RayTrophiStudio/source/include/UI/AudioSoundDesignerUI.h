#pragma once

#include <string>
struct UIContext;

namespace AudioUI {
void openSoundDesigner(UIContext& ctx, const std::string& assetId = {});
void drawSoundDesigner(UIContext& ctx);
void drawSoundDesignInspector(UIContext& ctx);
} // namespace AudioUI
