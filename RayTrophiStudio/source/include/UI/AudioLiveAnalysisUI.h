#pragma once

#include "json.hpp"
#include <string>
struct UIContext;

namespace AudioUI {
void openLiveAnalysis(UIContext& ctx, const std::string& bus);
bool drawAudioAnalysisWorkspace(UIContext& ctx, const nlohmann::json& designStatus,
                                bool original, nlohmann::json& recipe);
} // namespace AudioUI
