#pragma once
#include "json.hpp"
#include <string>
struct UIContext;
namespace AudioUI {
nlohmann::json workspaceCall(UIContext& ctx, const std::string& operation,
                            const nlohmann::json& params = nlohmann::json::object());
void drawWorkspace(UIContext& ctx);
void drawInspectorDock(UIContext& ctx);
void drawSourceInspector(UIContext& ctx);
void drawSynthesisControls(UIContext& ctx);
void finishAudioTimelineGesture(UIContext& ctx);
void drawAudioEditorNavigation(UIContext& ctx);
void drawAudioBottomTabs(UIContext& ctx);
void drawAudioTimeline(UIContext& ctx, float& zoom, float& pan, float legendWidth);
void drawMixerContent(UIContext& ctx);
void drawOutputControls(UIContext& ctx);
void selectClip(UIContext& ctx, const std::string& id);
void addTimelineAsset(UIContext& ctx, const std::string& assetId);
}
