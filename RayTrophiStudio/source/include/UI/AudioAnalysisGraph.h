#pragma once

#include "json.hpp"
struct UIContext;

namespace AudioUI {
void drawLiveAnalysisGraph(UIContext& ctx, const nlohmann::json& report,
                           const nlohmann::json& workspace);
}
