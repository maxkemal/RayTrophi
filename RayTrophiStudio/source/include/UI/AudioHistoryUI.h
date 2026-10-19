#pragma once
struct UIContext;

namespace AudioUI {
void registerAudioHistoryWindow(UIContext& ctx, bool inspector = false);
bool audioHistoryOwnsFocus(UIContext& ctx);
bool drawAudioHistoryMenu(UIContext& ctx, bool blocked);
// True means the request belongs to text/Designer and must not reach scene history.
bool handleAudioHistoryShortcuts(UIContext& ctx, bool blocked);
}
