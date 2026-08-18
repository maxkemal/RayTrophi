#pragma once

#include "imgui.h"
#include <string>
#include <vector>
#include <filesystem>

struct UIContext;
class SceneUI;
class SceneHistory;

namespace raytrophi::templates {

struct RecentProjectEntry {
    std::filesystem::path path;
    std::string display_name;
    std::string last_opened_time;
    bool exists = true;
};

enum class CardScale { Compact = 0, Standard = 1, Large = 2 };

class TemplateHubUI {
public:
    static TemplateHubUI& instance();

    void show();
    void hide();
    void toggle();
    bool isVisible() const { return visible_; }

    void render(UIContext& context, SceneUI& ui, SceneHistory* history);

    // Recent project management
    void addRecentProject(const std::filesystem::path& project_path);
    void removeRecentProject(const std::filesystem::path& project_path);
    const std::vector<RecentProjectEntry>& recentProjects() const { return recent_projects_; }
    void refreshRecentProjects();
    void saveRecentProjects();
    void loadRecentProjects();
    void clearRecentProjects();

    void performAutosaveRecovery(UIContext& context, SceneUI& ui, SceneHistory* history);

private:
    TemplateHubUI();
    ~TemplateHubUI() = default;

    void renderHeader();
    void renderSidebar(UIContext& context, SceneUI& ui, SceneHistory* history);
    void renderStartupPreferencesSection();
    void renderCategoryTabs();
    void renderTemplateGrid(UIContext& context, SceneUI& ui, SceneHistory* history);
    void renderDetailPane(UIContext& context, SceneUI& ui, SceneHistory* history);
    void renderUnsavedChangesModal(UIContext& context, SceneUI& ui, SceneHistory* history);

    void launchTemplate(const std::string& template_id, const std::string& policy,
                        UIContext& context, SceneUI& ui, SceneHistory* history);

    bool visible_ = false;
    bool startup_auto_show_done_ = false;
    bool sidebar_collapsed_ = false;
    bool detail_collapsed_ = false;

    float sidebar_width_ = 230.0f;
    float detail_width_ = 310.0f;

    std::string selected_category_ = "all";
    std::string selected_template_id_ = "raytrophi.start.general_scene";
    char search_filter_[128] = "";
    CardScale card_scale_ = CardScale::Standard;

    // Unsaved changes confirmation state
    bool show_unsaved_modal_ = false;
    std::string pending_template_id_;

    std::vector<RecentProjectEntry> recent_projects_;
    std::filesystem::path recent_config_path_;
};

} // namespace raytrophi::templates
