#pragma once

#include <string>
#include <vector>

class SceneUI;

namespace raytrophi::templates {

struct TemplateUiState;

struct TemplateUiApplyResult {
    bool applied = false;
    std::vector<std::string> warnings;
};

class TemplateUiStateAdapter {
public:
    static TemplateUiApplyResult apply(const TemplateUiState& state, SceneUI& ui);
};

} // namespace raytrophi::templates
