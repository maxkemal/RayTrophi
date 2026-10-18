#pragma once

#include <string>
struct UIContext;

namespace AudioUI {
bool isSoundDesignerOpen(UIContext& ctx);
void reuseSoundRecipe(UIContext& ctx, const std::string& assetId);
void openSoundDesigner(UIContext& ctx, const std::string& assetId = {},
                       bool equalizer = false);
void drawSoundDesigner(UIContext& ctx);
void drawSoundDesignInspector(UIContext& ctx);
} // namespace AudioUI
