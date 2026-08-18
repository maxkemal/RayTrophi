#include "Template/UserTemplateManager.h"
#include "Template/TemplateRegistry.h"
#include "Template/PathUtils.h"
#include "UI/TemplateHubUI.h"
#include "ProjectManager.h"
#include "scene_ui.h"
#include <fstream>
#include <iostream>

namespace raytrophi::templates {
using pathutils::pathFromUtf8;
using pathutils::pathToUtf8;

UserTemplateManager& UserTemplateManager::instance() {
    static UserTemplateManager mgr;
    return mgr;
}

UserTemplateManager::UserTemplateManager() {
    user_templates_dir_ = pathFromUtf8("assets/templates/user_templates");
    std::error_code ec;
    std::filesystem::create_directories(user_templates_dir_, ec);
}

bool UserTemplateManager::saveCurrentSceneAsTemplate(const std::string& display_name,
                                                     const std::string& description,
                                                     const std::string& category,
                                                     UIContext& context, SceneUI& ui, SceneHistory* history) {
    if (display_name.empty()) return false;

    try {
        // 1. Sync UI & scene state into ProjectManager first
        ui.updateProjectFromScene(context);

        std::string old_file_path = g_ProjectManager.getCurrentFilePath();

        // 2. Create unique safe folder ID
        std::string safe_id = "user.";
        for (char c : display_name) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                safe_id += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            } else if (c == ' ' || c == '_' || c == '-') {
                safe_id += '_';
            }
        }
        if (safe_id == "user.") safe_id += "custom_template";

        std::filesystem::path template_dir = user_templates_dir_ / safe_id;
        std::error_code ec;
        std::filesystem::create_directories(template_dir, ec);

        std::string project_filename = "scene.rtp";
        std::filesystem::path project_path = template_dir / project_filename;

        // 3. Save active scene into project package with proper UTF-8 path string
        bool saved = g_ProjectManager.saveProject(pathToUtf8(project_path), context.scene,
                                                   context.render_settings, context.renderer);

        // Restore active session project file path so session remains intact
        g_ProjectManager.getProjectData().current_file_path = old_file_path;
        
        // Remove template package from Recent Projects list (since it's a template, not a recent user project)
        TemplateHubUI::instance().removeRecentProject(project_path);

        if (!saved) {
            std::cerr << "[UserTemplateManager] Failed to save project package to " << project_path << std::endl;
            return false;
        }

        // 4. Generate fully-compliant manifest.json for user template
        nlohmann::json manifest;
        manifest["schema_version"] = "1.0";
        manifest["id"] = safe_id;
        manifest["display_name"] = display_name;
        manifest["description"] = description.empty() ? "User custom template" : description;
        manifest["kind"] = "user";
        manifest["category"] = category.empty() ? "user" : category;
        manifest["sort_order"] = 999;

        manifest["compatibility"] = {
            {"minimum_raytrophi_version", "1.0.0"}
        };

        manifest["preview"] = {
            {"image", ""},
            {"alt", display_name}
        };

        manifest["scene"] = {
            {"type", "project"},
            {"path", project_filename}
        };

        manifest["ui_state"] = {
            {"properties_context", "scene"},
            {"viewport_shading", "rendered"},
            {"bottom_editor", "timeline"},
            {"contextual_dock", "none"}
        };

        manifest["renderer"] = {
            {"preference", "any"},
            {"allow_fallback", true}
        };

        manifest["performance"] = {
            {"class", "User Custom"},
            {"estimated_vram_mb", 256}
        };

        manifest["assets"] = {
            {"required", nlohmann::json::array()},
            {"optional", nlohmann::json::array()}
        };

        {
            std::ofstream manifest_file(template_dir / "manifest.json");
            if (manifest_file) {
                manifest_file << manifest.dump(2);
                manifest_file.flush();
                manifest_file.close();
            }
        }

        // 5. Refresh TemplateRegistry so the new user template is available immediately
        TemplateRegistry::instance().refresh();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[UserTemplateManager] Error saving user template: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[UserTemplateManager] Unknown error saving user template." << std::endl;
        return false;
    }
}

bool UserTemplateManager::deleteUserTemplate(const std::string& template_id) {
    if (template_id.empty()) return false;

    const auto* metadata = TemplateRegistry::instance().find(template_id);
    if (!metadata || metadata->kind != "user") {
        std::cerr << "[UserTemplateManager] Cannot delete non-user template: " << template_id << std::endl;
        return false;
    }

    try {
        std::filesystem::remove_all(metadata->package_root);
        TemplateRegistry::instance().refresh();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[UserTemplateManager] Failed to delete user template: " << e.what() << std::endl;
        return false;
    }
}

} // namespace raytrophi::templates
