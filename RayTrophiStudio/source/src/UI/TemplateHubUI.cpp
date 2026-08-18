#include "UI/TemplateHubUI.h"

#include "ui_modern.h"
#include "Template/TemplateRegistry.h"
#include "Template/TemplateSession.h"
#include "Template/StartupPreferences.h"
#include "Template/UserTemplateManager.h"
#include "Template/PathUtils.h"
#include "ProjectManager.h"
#include "scene_ui.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "json.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cctype>

namespace raytrophi::templates {
using pathutils::pathFromUtf8;
using pathutils::pathToUtf8;

TemplateHubUI& TemplateHubUI::instance() {
    static TemplateHubUI ui;
    return ui;
}

TemplateHubUI::TemplateHubUI() {
    recent_config_path_ = pathFromUtf8("assets/config/recent_projects.json");
    loadRecentProjects();
}

void TemplateHubUI::show() {
    visible_ = true;
    TemplateRegistry::instance().refresh();
    refreshRecentProjects();
}

void TemplateHubUI::hide() {
    visible_ = false;
    show_unsaved_modal_ = false;
}

void TemplateHubUI::toggle() {
    if (visible_) hide();
    else show();
}

void TemplateHubUI::addRecentProject(const std::filesystem::path& project_path) {
    if (project_path.empty()) return;
    std::error_code ec;
    auto abs_path = std::filesystem::absolute(project_path, ec);
    
    // Remove duplicate entry if present
    recent_projects_.erase(
        std::remove_if(recent_projects_.begin(), recent_projects_.end(),
                       [&abs_path](const RecentProjectEntry& entry) {
                           return entry.path == abs_path;
                       }),
        recent_projects_.end());

    RecentProjectEntry entry;
    entry.path = abs_path;
    entry.display_name = pathToUtf8(abs_path.filename());
    entry.last_opened_time = "Recent";
    entry.exists = std::filesystem::exists(abs_path, ec);

    recent_projects_.insert(recent_projects_.begin(), entry);

    if (recent_projects_.size() > 10) {
        recent_projects_.resize(10);
    }
    saveRecentProjects();
}

void TemplateHubUI::refreshRecentProjects() {
    std::error_code ec;
    for (auto& entry : recent_projects_) {
        if (!entry.path.empty()) {
            entry.exists = std::filesystem::exists(entry.path, ec);
        }
    }
}

void TemplateHubUI::loadRecentProjects() {
    recent_projects_.clear();
    try {
        std::error_code ec;
        if (std::filesystem::exists(recent_config_path_, ec)) {
            std::ifstream file(recent_config_path_);
            if (file) {
                nlohmann::json j;
                file >> j;
                if (j.is_array()) {
                    for (const auto& item : j) {
                        RecentProjectEntry entry;
                        std::string p_str = item.value("path", "");
                        entry.path = pathFromUtf8(p_str);
                        entry.display_name = item.value("display_name", "");
                        entry.last_opened_time = item.value("last_opened_time", "Recent");
                        if (!entry.path.empty() || !entry.display_name.empty()) {
                            entry.exists = entry.path.empty() ? true : std::filesystem::exists(entry.path, ec);
                            if (entry.display_name.empty()) {
                                entry.display_name = pathToUtf8(entry.path.filename());
                            }
                            recent_projects_.push_back(entry);
                        }
                    }
                }
            }
        }
    } catch (...) {}

    if (recent_projects_.empty()) {
        RecentProjectEntry sample1;
        sample1.display_name = "General Scene Preset";
        sample1.last_opened_time = "Built-in";
        sample1.exists = true;
        recent_projects_.push_back(sample1);
    }
}

void TemplateHubUI::saveRecentProjects() {
    try {
        std::error_code ec;
        std::filesystem::create_directories(recent_config_path_.parent_path(), ec);
        nlohmann::json j = nlohmann::json::array();
        for (const auto& entry : recent_projects_) {
            j.push_back({
                {"path", pathToUtf8(entry.path)},
                {"display_name", entry.display_name},
                {"last_opened_time", entry.last_opened_time}
            });
        }
        std::ofstream file(recent_config_path_);
        if (file) {
            file << j.dump(2);
        }
    } catch (...) {}
}

void TemplateHubUI::removeRecentProject(const std::filesystem::path& project_path) {
    if (project_path.empty()) return;
    std::error_code ec;
    auto abs_target = std::filesystem::absolute(project_path, ec);
    recent_projects_.erase(
        std::remove_if(recent_projects_.begin(), recent_projects_.end(),
                       [&project_path, &abs_target](const RecentProjectEntry& entry) {
                           return entry.path == project_path || entry.path == abs_target;
                       }),
        recent_projects_.end());
    saveRecentProjects();
}

void TemplateHubUI::performAutosaveRecovery(UIContext& context, SceneUI& ui, SceneHistory* history) {
    auto auto_path = StartupPreferencesManager::instance().getAutosavePath();
    if (std::filesystem::exists(auto_path)) {
        hide();
        g_ProjectManager.openProject(auto_path.string(), context.scene, context.render_settings, context.renderer, context.backend_ptr);
    }
}

void TemplateHubUI::clearRecentProjects() {
    recent_projects_.clear();
    saveRecentProjects();
}

void TemplateHubUI::render(UIContext& context, SceneUI& ui, SceneHistory* history) {
    if (!startup_auto_show_done_) {
        startup_auto_show_done_ = true;
        StartupMode mode = StartupPreferencesManager::instance().getStartupMode();

        if (mode == StartupMode::OpenLastProject && !recent_projects_.empty() && recent_projects_[0].exists && !recent_projects_[0].path.empty()) {
            visible_ = false;
            g_ProjectManager.openProject(recent_projects_[0].path.string(), context.scene, context.render_settings, context.renderer, context.backend_ptr);
            return;
        } else if (mode == StartupMode::StartEmpty) {
            visible_ = false;
            launchTemplate("raytrophi.start.empty", "discard", context, ui, history);
            return;
        } else if (mode == StartupMode::RestoreAutosave && StartupPreferencesManager::instance().hasValidAutosave()) {
            visible_ = false;
            performAutosaveRecovery(context, ui, history);
            return;
        } else {
            visible_ = true;
            TemplateRegistry::instance().refresh();
        }
    }

    if (!visible_) return;

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 display_size = io.DisplaySize;

    // Active System Theme Colors
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec4* sys_colors = style.Colors;

    ImVec4 theme_window_bg = sys_colors[ImGuiCol_WindowBg];
    ImVec4 theme_child_bg  = sys_colors[ImGuiCol_ChildBg];
    ImVec4 theme_border    = sys_colors[ImGuiCol_Border];
    ImVec4 theme_title_bg  = sys_colors[ImGuiCol_TitleBg];
    ImVec4 theme_title_act = sys_colors[ImGuiCol_TitleBgActive];
    ImVec4 theme_accent    = (sys_colors[ImGuiCol_HeaderActive].w > 0.1f) ? 
                             sys_colors[ImGuiCol_HeaderActive] : 
                             sys_colors[ImGuiCol_SliderGrab];

    // Dim background
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(display_size);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(theme_window_bg.x * 0.4f, theme_window_bg.y * 0.4f, theme_window_bg.z * 0.4f, 0.92f));

    ImGuiWindowFlags dim_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_NoBringToFrontOnFocus;

    if (ImGui::Begin("##TemplateHubDimBackground", nullptr, dim_flags)) {

        float default_w = std::min(display_size.x * 0.92f, 1240.0f);
        float default_h = std::min(display_size.y * 0.88f, 780.0f);
        ImVec2 default_pos = ImVec2((display_size.x - default_w) * 0.5f, (display_size.y - default_h) * 0.5f);

        ImGui::SetNextWindowPos(default_pos, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(default_w, default_h), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSizeConstraints(ImVec2(780, 500), ImVec2(display_size.x, display_size.y));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 12));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(theme_window_bg.x, theme_window_bg.y, theme_window_bg.z, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Border, theme_border);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(theme_child_bg.x, theme_child_bg.y, theme_child_bg.z, 0.75f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, theme_title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, theme_title_act);

        ImGuiWindowFlags hub_flags = ImGuiWindowFlags_NoCollapse;

        if (ImGui::Begin("RayTrophi Studio - Template Hub##MainModal", &visible_, hub_flags)) {

            // Draw bold background grid & watermark emblem using dynamic theme accent color
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 wpos = ImGui::GetWindowPos();
            ImVec2 wsize = ImGui::GetWindowSize();

            ImU32 line_col = ImGui::GetColorU32(ImVec4(theme_accent.x, theme_accent.y, theme_accent.z, 0.15f));
            for (float x = 0; x < wsize.x; x += 32.0f) {
                draw_list->AddLine(ImVec2(wpos.x + x, wpos.y), ImVec2(wpos.x + x, wpos.y + wsize.y), line_col);
            }
            for (float y = 0; y < wsize.y; y += 32.0f) {
                draw_list->AddLine(ImVec2(wpos.x, wpos.y + y), ImVec2(wpos.x + wsize.x, wpos.y + y), line_col);
            }

            // Bold Bottom Right Watermark (Positioned comfortably inside window, dynamic theme accent color)
            ImVec2 mark_pos(wpos.x + wsize.x - 270.0f, wpos.y + wsize.y - 24.0f);
            ImU32 mark_col = ImGui::GetColorU32(ImVec4(std::min(1.0f, theme_accent.x * 1.2f), 
                                                       std::min(1.0f, theme_accent.y * 1.2f), 
                                                       std::min(1.0f, theme_accent.z * 1.2f), 0.55f));
            draw_list->AddText(mark_pos, mark_col, "RAYTROPHI ENGINE v1.0 - VP-FIRST");

            // Keyboard shortcuts (Esc to close, Enter to launch)
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                hide();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) {
                if (!selected_template_id_.empty() && !show_unsaved_modal_) {
                    launchTemplate(selected_template_id_, "reject", context, ui, history);
                }
            }

            // Precise dynamic layout sizing (Aligns right detail panel perfectly a few pixels from right border)
            ImVec2 avail = ImGui::GetContentRegionAvail();

            float eff_sidebar_w = sidebar_collapsed_ ? 44.0f : sidebar_width_;
            float eff_detail_w = detail_collapsed_ ? 44.0f : detail_width_;

            float left_gap = sidebar_collapsed_ ? 4.0f : 12.0f;
            float right_gap = detail_collapsed_ ? 4.0f : 12.0f;
            float content_w = std::max(180.0f, avail.x - eff_sidebar_w - eff_detail_w - left_gap - right_gap);
            float full_h = avail.y;

            // 1. Left Sidebar (Quick Actions & Recent Projects)
            ImGui::BeginChild("##HubSidebarPane", ImVec2(eff_sidebar_w, full_h), true);
            renderSidebar(context, ui, history);
            ImGui::EndChild();

            // Seamless Invisible Left Splitter Drag Zone
            if (!sidebar_collapsed_) {
                ImGui::SameLine(0, 3.0f);
                ImGui::InvisibleButton("##SidebarSplitter", ImVec2(6.0f, full_h));
                if (ImGui::IsItemActive()) {
                    sidebar_width_ += io.MouseDelta.x;
                    sidebar_width_ = std::clamp(sidebar_width_, 150.0f, 380.0f);
                }
                if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                ImGui::SameLine(0, 3.0f);
            } else {
                ImGui::SameLine(0, 4.0f);
            }

            // 2. Center Column (Search, Tabs & Template Grid)
            ImGui::BeginChild("##HubCenterPane", ImVec2(content_w, full_h), false);
            
            // Search & Filter Bar + Grid Scale Controls
            float zoom_w = 156.0f;
            ImGui::PushItemWidth(std::max(80.0f, content_w - zoom_w - 6.0f));
            ImGui::InputTextWithHint("##TemplateSearch", "Search templates...", search_filter_, sizeof(search_filter_));
            ImGui::PopItemWidth();

            ImGui::SameLine();
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            if (UIWidgets::HorizontalTab("S", UIWidgets::IconType::ViewOverlays, card_scale_ == CardScale::Compact, 46.0f)) {
                card_scale_ = CardScale::Compact;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Compact Grid View (Small Cards)");

            ImGui::SameLine(0, 2.0f);
            if (UIWidgets::HorizontalTab("M", UIWidgets::IconType::Mesh, card_scale_ == CardScale::Standard, 48.0f)) {
                card_scale_ = CardScale::Standard;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Standard Grid View (Medium Cards)");

            ImGui::SameLine(0, 2.0f);
            if (UIWidgets::HorizontalTab("L", UIWidgets::IconType::Render, card_scale_ == CardScale::Large, 46.0f)) {
                card_scale_ = CardScale::Large;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Large Grid View (Big Cards)");
            ImGui::PopStyleVar();

            ImGui::Spacing();

            renderCategoryTabs();
            ImGui::Spacing();
            renderTemplateGrid(context, ui, history);
            ImGui::EndChild();

            // Seamless Invisible Right Splitter Drag Zone
            if (!detail_collapsed_) {
                ImGui::SameLine(0, 3.0f);
                ImGui::InvisibleButton("##DetailSplitter", ImVec2(6.0f, full_h));
                if (ImGui::IsItemActive()) {
                    detail_width_ -= io.MouseDelta.x;
                    detail_width_ = std::clamp(detail_width_, 220.0f, 480.0f);
                }
                if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                ImGui::SameLine(0, 3.0f);
            } else {
                ImGui::SameLine(0, 4.0f);
            }

            // 3. Right Inspector Pane (Selected Template Detail View)
            ImGui::BeginChild("##HubDetailPane", ImVec2(eff_detail_w, full_h), true);
            renderDetailPane(context, ui, history);
            ImGui::EndChild();

            // Unsaved Changes Confirmation Modal
            renderUnsavedChangesModal(context, ui, history);
        }
        ImGui::End();

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(2);
    }
    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void TemplateHubUI::renderSidebar(UIContext& context, SceneUI& ui, SceneHistory* history) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec4* sys_colors = style.Colors;
    ImVec4 theme_accent = (sys_colors[ImGuiCol_HeaderActive].w > 0.1f) ? 
                          sys_colors[ImGuiCol_HeaderActive] : 
                          sys_colors[ImGuiCol_SliderGrab];
    ImVec4 theme_btn_accent = ImVec4(theme_accent.x, theme_accent.y, theme_accent.z, 1.0f);
    ImVec4 theme_subtext = sys_colors[ImGuiCol_TextDisabled];
    ImVec4 theme_text = sys_colors[ImGuiCol_Text];

    if (sidebar_collapsed_) {
        if (ImGui::Button(">>##ExpandSidebar", ImVec2(30, 30))) {
            sidebar_collapsed_ = false;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Expand Sidebar");
        ImGui::Spacing();

        if (UIWidgets::IconActionButton("##NewProjBtnIcon", UIWidgets::IconType::Scene, "", false,
                                         theme_btn_accent, ImVec2(30, 30),
                                         "New Project")) {
            launchTemplate("raytrophi.start.general_scene", "reject", context, ui, history);
        }
        ImGui::Spacing();
        if (UIWidgets::IconActionButton("##OpenProjBtnIcon", UIWidgets::IconType::Assets, "", false,
                                         ImVec4(std::min(1.0f, theme_accent.x * 1.1f), 
                                                std::min(1.0f, theme_accent.y * 1.1f), 
                                                std::min(1.0f, theme_accent.z * 1.1f), 1.0f), ImVec2(30, 30),
                                         "Open Project...")) {
            hide();
            ui.performOpenProject(context);
        }
        return;
    }

    // Header with Collapse Toggle
    ImGui::TextColored(theme_btn_accent, "QUICK START");
    ImGui::SameLine(ImGui::GetWindowWidth() - 32.0f);
    if (ImGui::Button("<<##CollapseSidebar", ImVec2(24, 24))) {
        sidebar_collapsed_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Collapse Sidebar");

    ImGui::Spacing();

    // Primary Action: New General Scene
    if (UIWidgets::IconActionButton("##NewProjBtn", UIWidgets::IconType::Scene, "New Project", false,
                                     theme_btn_accent, ImVec2(-1, 34),
                                     "Create a new clean production scene")) {
        launchTemplate("raytrophi.start.general_scene", "reject", context, ui, history);
    }
    ImGui::Spacing();

    // Open Project (Native File Dialog)
    if (UIWidgets::IconActionButton("##OpenProjBtn", UIWidgets::IconType::Assets, "Open Project...", false,
                                     ImVec4(std::min(1.0f, theme_accent.x * 1.1f), 
                                            std::min(1.0f, theme_accent.y * 1.1f), 
                                            std::min(1.0f, theme_accent.z * 1.1f), 1.0f), ImVec2(-1, 34),
                                     "Open an existing RayTrophi project file (.rtp / .rts)")) {
        hide();
        ui.performOpenProject(context);
    }
    ImGui::Spacing();

    // Recover Autosave
    bool has_auto = StartupPreferencesManager::instance().hasValidAutosave();
    if (!has_auto) ImGui::BeginDisabled(true);
    if (UIWidgets::IconActionButton("##RecoverAutosaveBtn", UIWidgets::IconType::Timeline, "Recover Autosave", false,
                                     ImVec4(std::min(1.0f, theme_accent.x * 1.1f), 
                                            std::min(1.0f, theme_accent.y * 1.1f), 
                                            std::min(1.0f, theme_accent.z * 1.1f), 1.0f), ImVec2(-1, 34),
                                     has_auto ? "Recover the last autosaved scene session" : "No autosaved session found")) {
        performAutosaveRecovery(context, ui, history);
    }
    if (!has_auto) ImGui::EndDisabled();

    ImGui::Spacing();
    UIWidgets::Divider();
    ImGui::Spacing();

    ImGui::TextColored(theme_subtext, "RECENT PROJECTS");
    if (!recent_projects_.empty()) {
        ImGui::SameLine(ImGui::GetWindowWidth() - 50.0f);
        if (ImGui::SmallButton("Clear")) {
            clearRecentProjects();
        }
    }
    ImGui::Spacing();

    if (recent_projects_.empty()) {
        ImGui::TextDisabled("No recent projects");
    } else {
        for (const auto& recent : recent_projects_) {
            ImGui::PushID(recent.path.string().c_str());
            
            if (recent.exists) {
                ImGui::PushStyleColor(ImGuiCol_Text, theme_text);
                if (ImGui::Selectable(recent.display_name.c_str(), false)) {
                    hide();
                    if (!recent.path.empty() && std::filesystem::exists(recent.path)) {
                        g_ProjectManager.openProject(recent.path.string(), context.scene, context.render_settings, context.renderer, context.backend_ptr);
                    } else {
                        launchTemplate("raytrophi.start.general_scene", "discard", context, ui, history);
                    }
                }
                ImGui::PopStyleColor();
            } else {
                ImGui::TextColored(ImVec4(0.85f, 0.45f, 0.45f, 0.9f), "%s (Missing)", recent.display_name.c_str());
            }

            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", recent.path.empty() ? recent.display_name.c_str() : recent.path.string().c_str());
            }

            if (ImGui::BeginPopupContextItem("##RecentItemContext")) {
                if (recent.exists && !recent.path.empty()) {
                    if (ImGui::MenuItem("Open Project")) {
                        hide();
                        g_ProjectManager.openProject(recent.path.string(), context.scene, context.render_settings, context.renderer, context.backend_ptr);
                    }
                    if (ImGui::MenuItem("Show in Explorer")) {
                        std::string cmd = "explorer.exe /select,\"" + recent.path.string() + "\"";
                        system(cmd.c_str());
                    }
                    ImGui::Separator();
                }
                if (ImGui::MenuItem("Remove from Recent")) {
                    removeRecentProject(recent.path);
                }
                ImGui::EndPopup();
            }

            ImGui::PopID();
        }
    }

    renderStartupPreferencesSection();
}

void TemplateHubUI::renderStartupPreferencesSection() {
    ImGui::Spacing();
    UIWidgets::Divider();
    ImGui::Spacing();

    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec4* sys_colors = style.Colors;
    ImVec4 theme_subtext = sys_colors[ImGuiCol_TextDisabled];

    ImGui::TextColored(theme_subtext, "STARTUP ACTION");
    ImGui::Spacing();

    auto& pref_mgr = StartupPreferencesManager::instance();
    StartupMode current_mode = pref_mgr.getStartupMode();

    const char* mode_names[] = {
        "Always Show Template Hub",
        "Open Last Project",
        "Start Empty Canvas",
        "Restore Autosaved Session"
    };

    int current_item = static_cast<int>(current_mode);
    ImGui::PushItemWidth(-1.0f);
    if (ImGui::Combo("##StartupModeCombo", &current_item, mode_names, IM_ARRAYSIZE(mode_names))) {
        pref_mgr.setStartupMode(static_cast<StartupMode>(current_item));
    }
    ImGui::PopItemWidth();
}

void TemplateHubUI::renderCategoryTabs() {
    struct CategoryTab {
        const char* id;
        const char* label;
        UIWidgets::IconType icon;
    };

    static const CategoryTab tabs[] = {
        {"all", "All Templates", UIWidgets::IconType::Scene},
        {"production", "Production Start", UIWidgets::IconType::Render},
        {"vfx", "VFX & Simulation", UIWidgets::IconType::Volumetric},
        {"user", "User Templates", UIWidgets::IconType::Assets},
        {"learn", "Learn & Demos", UIWidgets::IconType::Help}
    };

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
    for (size_t i = 0; i < std::size(tabs); ++i) {
        if (i > 0) ImGui::SameLine();
        bool active = (selected_category_ == tabs[i].id);

        if (UIWidgets::HorizontalTab(tabs[i].label, tabs[i].icon, active, 130.0f)) {
            selected_category_ = tabs[i].id;
        }
    }
    ImGui::PopStyleVar();
}

void TemplateHubUI::renderTemplateGrid(UIContext& context, SceneUI& ui, SceneHistory* history) {
    const auto& entries = TemplateRegistry::instance().entries();

    std::string query = search_filter_;
    std::transform(query.begin(), query.end(), query.begin(), ::tolower);

    std::vector<const TemplateMetadata*> filtered;
    for (const auto& entry : entries) {
        if (!entry.valid) continue;

        // Category filter
        if (selected_category_ == "production") {
            if (entry.category != "empty" && entry.category != "general" &&
                entry.category != "lookdev" && entry.category != "hair" &&
                entry.category != "paint" && entry.category != "terrain") {
                continue;
            }
        } else if (selected_category_ == "vfx") {
            if (entry.category != "vfx" && entry.category != "volume" && entry.category != "fluid") {
                continue;
            }
        } else if (selected_category_ == "user") {
            if (entry.kind != "user" && entry.category != "user") {
                continue;
            }
        } else if (selected_category_ == "learn") {
            if (entry.kind != "learn") continue;
        }

        // Search query filter
        if (!query.empty()) {
            std::string name_lower = entry.display_name;
            std::string desc_lower = entry.description;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
            std::transform(desc_lower.begin(), desc_lower.end(), desc_lower.begin(), ::tolower);
            if (name_lower.find(query) == std::string::npos && desc_lower.find(query) == std::string::npos) {
                continue;
            }
        }

        filtered.push_back(&entry);
    }

    if (filtered.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("No templates found matching your criteria.");
        return;
    }

    float card_w = 215.0f;
    float card_h = 155.0f;
    float banner_h = 44.0f;
    float title_y = 11.0f;
    float body_y = 52.0f;
    float badge_y = card_h - 24.0f;

    if (card_scale_ == CardScale::Compact) {
        card_w = 165.0f;
        card_h = 122.0f;
        banner_h = 36.0f;
        title_y = 8.0f;
        body_y = 42.0f;
        badge_y = card_h - 20.0f;
    } else if (card_scale_ == CardScale::Large) {
        card_w = 275.0f;
        card_h = 190.0f;
        banner_h = 52.0f;
        title_y = 14.0f;
        body_y = 60.0f;
        badge_y = card_h - 28.0f;
    }

    float avail_w = ImGui::GetContentRegionAvail().x;
    int columns = std::max(1, static_cast<int>(avail_w / (card_w + 10.0f)));

    ImGui::Columns(columns, "##TemplateGridCols", false);

    for (size_t i = 0; i < filtered.size(); ++i) {
        const auto* item = filtered[i];
        ImGui::PushID(item->id.c_str());

        bool is_selected = (selected_template_id_ == item->id);
        bool is_user_template = (item->kind == "user" || item->category == "user");

        const ImGuiStyle& style = ImGui::GetStyle();
        const ImVec4* sys_colors = style.Colors;
        ImVec4 theme_accent = (sys_colors[ImGuiCol_HeaderActive].w > 0.1f) ? 
                              sys_colors[ImGuiCol_HeaderActive] : 
                              sys_colors[ImGuiCol_SliderGrab];

        if (is_selected) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(theme_accent.x * 0.25f, theme_accent.y * 0.25f, theme_accent.z * 0.25f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_Border, is_user_template ? ImVec4(0.95f, 0.72f, 0.20f, 1.0f) : theme_accent);
        } else if (is_user_template) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(sys_colors[ImGuiCol_ChildBg].x, sys_colors[ImGuiCol_ChildBg].y, sys_colors[ImGuiCol_ChildBg].z, 0.92f));
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.62f, 0.15f, 0.65f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(sys_colors[ImGuiCol_ChildBg].x, sys_colors[ImGuiCol_ChildBg].y, sys_colors[ImGuiCol_ChildBg].z, 0.90f));
            ImGui::PushStyleColor(ImGuiCol_Border, sys_colors[ImGuiCol_Border]);
        }

        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0)); // 0 padding for clean header gradient!

        if (ImGui::BeginChild("##TemplateCardItem", ImVec2(card_w, card_h), true,
                               ImGuiWindowFlags_NoScrollbar)) {

            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 cpos = ImGui::GetWindowPos();
            ImVec2 csize = ImGui::GetWindowSize();

            ImVec4 cat_color1 = ImVec4(theme_accent.x, theme_accent.y, theme_accent.z, 1.0f);
            ImVec4 cat_color2 = ImVec4(theme_accent.x * 0.45f, theme_accent.y * 0.45f, theme_accent.z * 0.45f, 1.0f);
            UIWidgets::IconType card_icon = UIWidgets::IconType::Mesh;

            if (is_user_template) {
                // Distinct Amber-Gold accent ribbon for User Custom Templates
                cat_color1 = ImVec4(0.94f, 0.68f, 0.18f, 1.0f);
                cat_color2 = ImVec4(0.52f, 0.35f, 0.08f, 1.0f);
                card_icon = UIWidgets::IconType::Assets;
            } else if (item->category == "empty" || item->id == "raytrophi.start.empty") {
                card_icon = UIWidgets::IconType::AddKey;
            } else if (item->category == "general" || item->id == "raytrophi.start.general_scene") {
                card_icon = UIWidgets::IconType::Mesh;
            } else if (item->category == "vfx" || item->category == "volume" || item->id == "raytrophi.vfx.gas_smoke") {
                card_icon = UIWidgets::IconType::Volumetric;
            } else if (item->category == "fluid" || item->id == "raytrophi.vfx.fluid_studio") {
                card_icon = UIWidgets::IconType::Water;
            } else if (item->category == "lookdev" || item->id == "raytrophi.start.product_lookdev") {
                card_icon = UIWidgets::IconType::Render;
            } else if (item->category == "paint" || item->id == "raytrophi.start.character_paint") {
                card_icon = UIWidgets::IconType::PaintTool;
            } else if (item->category == "hair" || item->id == "raytrophi.start.portrait_groom") {
                card_icon = UIWidgets::IconType::Hair;
            } else if (item->category == "terrain" || item->id == "raytrophi.start.terrain_environment") {
                card_icon = UIWidgets::IconType::Terrain;
            }

            // Draw Top Header Banner Gradient
            draw_list->AddRectFilledMultiColor(
                cpos, ImVec2(cpos.x + csize.x, cpos.y + banner_h),
                ImGui::GetColorU32(cat_color1), ImGui::GetColorU32(cat_color2),
                ImGui::GetColorU32(cat_color2), ImGui::GetColorU32(cat_color1)
            );

            // Icon Badge Circle
            float icon_circle_r = (banner_h * 0.30f);
            float icon_size = std::clamp(banner_h * 0.36f, 12.0f, 18.0f);
            draw_list->AddCircleFilled(ImVec2(cpos.x + 20.0f, cpos.y + banner_h * 0.5f), icon_circle_r, IM_COL32(0, 0, 0, 95));
            UIWidgets::DrawIcon(card_icon, ImVec2(cpos.x + 20.0f - icon_size * 0.5f, cpos.y + banner_h * 0.5f - icon_size * 0.5f), icon_size, IM_COL32(255, 255, 255, 255));

            // Banner Title
            ImGui::SetCursorPos(ImVec2(40.0f, title_y));
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "%s", item->display_name.c_str());

            // Card Body Text
            if (card_h > 130.0f) {
                ImGui::SetCursorPos(ImVec2(10.0f, body_y));
                ImGui::PushTextWrapPos(card_w - 10.0f);
                ImGui::TextColored(sys_colors[ImGuiCol_Text], "%s", item->description.c_str());
                ImGui::PopTextWrapPos();
            }

            // Bottom Badges
            ImGui::SetCursorPos(ImVec2(10.0f, badge_y));
            if (is_user_template) {
                ImGui::TextColored(ImVec4(0.95f, 0.72f, 0.20f, 1.0f), "[USER PRESET]");
            } else {
                ImGui::TextColored(theme_accent, "[%s]", item->category.c_str());
            }
            ImGui::SameLine();
            ImGui::TextDisabled("%s", item->performance_class.c_str());

            // Interaction
            if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0)) {
                selected_template_id_ = item->id;
            }
            if (ImGui::IsWindowHovered() && ImGui::IsMouseDoubleClicked(0)) {
                selected_template_id_ = item->id;
                launchTemplate(item->id, "reject", context, ui, history);
            }

            // Right-click Context Menu for User Templates
            if (item->kind == "user" || item->category == "user") {
                if (ImGui::BeginPopupContextItem("##UserTemplateContext")) {
                    if (ImGui::MenuItem("Delete User Template")) {
                        std::string target_id = item->id;
                        UserTemplateManager::instance().deleteUserTemplate(target_id);
                        if (selected_template_id_ == target_id) {
                            selected_template_id_ = "raytrophi.start.general_scene";
                        }
                    }
                    ImGui::EndPopup();
                }
            }
        }
        ImGui::EndChild();

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);

        ImGui::PopID();
        ImGui::NextColumn();
    }

    ImGui::Columns(1);
}

void TemplateHubUI::renderDetailPane(UIContext& context, SceneUI& ui, SceneHistory* history) {
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec4* sys_colors = style.Colors;
    ImVec4 theme_accent = (sys_colors[ImGuiCol_HeaderActive].w > 0.1f) ? 
                          sys_colors[ImGuiCol_HeaderActive] : 
                          sys_colors[ImGuiCol_SliderGrab];
    ImVec4 theme_btn_accent = ImVec4(theme_accent.x, theme_accent.y, theme_accent.z, 1.0f);
    ImVec4 theme_text = sys_colors[ImGuiCol_Text];

    if (detail_collapsed_) {
        if (ImGui::Button("<<##ExpandDetail", ImVec2(28, 24))) {
            detail_collapsed_ = false;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Expand Detail Inspector");
        return;
    }

    const auto* metadata = TemplateRegistry::instance().find(selected_template_id_);

    // Collapse toggle button at top right
    ImGui::TextColored(theme_btn_accent, "DETAILS");
    ImGui::SameLine(ImGui::GetWindowWidth() - 32.0f);
    if (ImGui::Button(">>##CollapseDetail", ImVec2(24, 24))) {
        detail_collapsed_ = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Collapse Detail Inspector");

    ImGui::Spacing();

    if (!metadata) {
        ImGui::TextDisabled("Select a template to view details.");
        return;
    }

    // Large Title
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextColored(theme_text, "%s", metadata->display_name.c_str());
    ImGui::PopFont();

    ImGui::TextDisabled("ID: %s", metadata->id.c_str());
    ImGui::Spacing();

    // Category & Kind Badges
    UIWidgets::StatusIndicator(metadata->kind.c_str(), UIWidgets::StatusType::Success);
    ImGui::SameLine();
    ImGui::TextColored(theme_btn_accent, "Category: %s", metadata->category.c_str());

    ImGui::Spacing();
    UIWidgets::Divider();
    ImGui::Spacing();

    // Description
    ImGui::TextWrapped("%s", metadata->description.c_str());

    ImGui::Spacing();
    UIWidgets::Divider();
    ImGui::Spacing();

    // UI Setup Contract Card
    if (UIWidgets::BeginSection("INITIAL UI SETUP", theme_btn_accent, true)) {
        ImGui::BulletText("Properties: %s", metadata->ui_state.properties_context.c_str());
        ImGui::BulletText("Shading: %s", metadata->ui_state.viewport_shading.c_str());
        ImGui::BulletText("Bottom Editor: %s", metadata->ui_state.bottom_editor.c_str());
        ImGui::BulletText("Contextual Dock: %s", metadata->ui_state.contextual_dock.c_str());
        if (!metadata->ui_state.frame_target.empty()) {
            ImGui::BulletText("Target: %s", metadata->ui_state.frame_target.c_str());
        }
        UIWidgets::EndSection();
    }

    ImGui::Spacing();

    // Performance & Renderer Card
    if (UIWidgets::BeginSection("PERFORMANCE & GPU", theme_btn_accent, true)) {
        ImGui::BulletText("Class: %s", metadata->performance_class.c_str());
        ImGui::BulletText("Est. VRAM: %d MB", metadata->estimated_vram_mb);
        ImGui::BulletText("Renderer Pref: %s", metadata->renderer_preference.c_str());
        UIWidgets::EndSection();
    }

    ImGui::Spacing();
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 28.0f);
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x > 10.0f ? ImGui::GetContentRegionAvail().x : 200.0f);
    ImGui::TextDisabled("Double-click or press Enter to launch.");
    ImGui::PopTextWrapPos();
}

void TemplateHubUI::renderUnsavedChangesModal(UIContext& context, SceneUI& ui, SceneHistory* history) {
    if (!show_unsaved_modal_) return;

    ImGui::OpenPopup("Unsaved Changes##TemplateHubModal");
    if (ImGui::BeginPopupModal("Unsaved Changes##TemplateHubModal", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("The active scene has unsaved changes.\nHow would you like to proceed?");
        ImGui::Spacing();

        if (UIWidgets::DangerButton("Discard & Open Template", ImVec2(200, 32))) {
            show_unsaved_modal_ = false;
            launchTemplate(pending_template_id_, "discard", context, ui, history);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();

        if (UIWidgets::SecondaryButton("Cancel", ImVec2(100, 32))) {
            show_unsaved_modal_ = false;
            pending_template_id_.clear();
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void TemplateHubUI::launchTemplate(const std::string& template_id, const std::string& policy,
                                   UIContext& context, SceneUI& ui, SceneHistory* history) {
    if (ProjectManager::getInstance().hasUnsavedChanges() && policy == "reject") {
        pending_template_id_ = template_id;
        show_unsaved_modal_ = true;
        return;
    }

    const auto opened = raytrophi::templates::TemplateSession::instance().open(
        template_id, policy, context, ui, history);

    if (opened.opened) {
        hide();
    } else {
        std::cerr << "[TemplateHubUI] Failed to open template " << template_id << ": "
                  << (opened.errors.empty() ? opened.code : opened.errors.front()) << std::endl;
    }
}

} // namespace raytrophi::templates
