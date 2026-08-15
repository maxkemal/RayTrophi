#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace raytrophi::templates {

struct TemplateUiState {
    std::string properties_context;
    std::string bottom_editor;
    std::string contextual_dock;
    float contextual_dock_width = 0.0f;
    std::string viewport_shading;
    std::string frame_target;
    bool show_timeline = false;
};

struct TemplateMetadata {
    std::string id;
    std::string display_name;
    std::string description;
    std::string kind;
    std::string category;
    int sort_order = 0;
    std::string schema_version;
    std::string minimum_raytrophi_version;
    std::string maximum_raytrophi_version;
    std::filesystem::path package_root;
    std::filesystem::path manifest_path;
    std::filesystem::path preview_path;
    std::string preview_alt;
    std::string scene_type;
    std::filesystem::path scene_path;
    TemplateUiState ui_state;
    std::string renderer_preference = "auto";
    bool renderer_allow_fallback = true;
    std::string performance_class;
    int estimated_vram_mb = 0;
    std::vector<std::filesystem::path> required_assets;
    std::vector<std::filesystem::path> optional_assets;
    std::filesystem::path guidance_path;
    bool guidance_show_on_first_open = false;
    bool valid = false;
    std::vector<std::string> errors;
};

class TemplateRegistry {
public:
    static TemplateRegistry& instance();

    void setSearchRoots(std::vector<std::filesystem::path> roots);
    void resetSearchRoots();
    void refresh();

    const std::vector<TemplateMetadata>& entries() const;
    const TemplateMetadata* find(const std::string& id) const;
    std::vector<std::filesystem::path> searchRoots() const;

    static std::vector<std::filesystem::path> defaultSearchRoots();
    static TemplateMetadata validateManifest(const std::filesystem::path& manifest_path);

private:
    std::vector<std::filesystem::path> configured_roots_;
    std::vector<TemplateMetadata> entries_;
    bool refreshed_ = false;
};

} // namespace raytrophi::templates
