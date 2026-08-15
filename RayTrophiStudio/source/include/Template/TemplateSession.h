#pragma once

#include <string>
#include <vector>

struct UIContext;
class SceneHistory;
class SceneUI;

namespace raytrophi::templates {

struct TemplateOpenResult {
    std::string template_id;
    std::string state = "rejected";
    std::string code = "invalid_template";
    bool opened = false;
    bool ui_state_applied = false;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

class TemplateSession {
public:
    static TemplateSession& instance();

    TemplateOpenResult open(const std::string& template_id,
                            const std::string& conflict_policy,
                            UIContext& context,
                            SceneUI& ui,
                            SceneHistory* history) const;
};

} // namespace raytrophi::templates
