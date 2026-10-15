#pragma once

#include "json.hpp"
#include <string>
struct UIContext;

namespace AudioUI {
// layout: 0 = full rack, 1 = compact list, 2 = selected module properties.
bool drawDesignModules(UIContext& ctx, nlohmann::json& recipe, bool effects, int layout = 0);
bool drawDesignGenerationCanvas(UIContext& ctx, nlohmann::json& recipe);
bool drawDesignEffectPad(UIContext& ctx, nlohmann::json& recipe);
bool drawDesignSourceEditor(UIContext& ctx, nlohmann::json& recipe);
bool drawDesignEffectGraph(UIContext& ctx, nlohmann::json& recipe);
bool drawDesignAnalysis(UIContext& ctx, const nlohmann::json& status, bool original,
                        nlohmann::json& recipe);
const std::string& designPanelsError();
} // namespace AudioUI
