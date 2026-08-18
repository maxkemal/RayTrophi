#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include "json.hpp"

struct UIContext;
class SceneUI;
class SceneHistory;

namespace raytrophi::templates {

class UserTemplateManager {
public:
    static UserTemplateManager& instance();

    std::filesystem::path getUserTemplatesDir() const { return user_templates_dir_; }

    // Save current active scene as a custom user template
    bool saveCurrentSceneAsTemplate(const std::string& display_name,
                                    const std::string& description,
                                    const std::string& category,
                                    UIContext& context, SceneUI& ui, SceneHistory* history);

    // Delete a user template package from disk
    bool deleteUserTemplate(const std::string& template_id);

private:
    UserTemplateManager();
    ~UserTemplateManager() = default;

    std::filesystem::path user_templates_dir_;
};

} // namespace raytrophi::templates
