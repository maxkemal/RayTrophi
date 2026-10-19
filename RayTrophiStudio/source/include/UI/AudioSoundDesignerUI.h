#pragma once

#include <string>
struct UIContext;

namespace AudioUI {
bool isSoundDesignerOpen(UIContext& ctx);
bool canSoundDesignerHistory(UIContext& ctx, bool redo);
void stepSoundDesignerHistory(UIContext& ctx, bool redo);
void reuseSoundRecipe(UIContext& ctx, const std::string& assetId);
void openSoundDesigner(UIContext& ctx, const std::string& assetId = {},
                       bool equalizer = false);
void drawSoundDesigner(UIContext& ctx);
void drawSoundDesignInspector(UIContext& ctx);
} // namespace AudioUI
