#pragma once
struct UIContext;
namespace AudioUI {
void drawAudioTrackTools(UIContext& ctx);
void drawAudioClipSelectionTools(UIContext& ctx);
bool drawSelectedAudioTrackProperties(UIContext& ctx);
} // namespace AudioUI
