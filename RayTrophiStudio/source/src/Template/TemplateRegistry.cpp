#include "Template/TemplateRegistry.h"

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <unordered_map>

#include <SDL.h>

namespace raytrophi::templates {
namespace {

using json = nlohmann::json;

const std::set<std::string> kKinds{"start", "learn"};
const std::set<std::string> kCategories{
    "empty", "general", "lookdev", "character", "paint", "terrain", "animation", "simulation"};
const std::set<std::string> kPropertiesContexts{
    "scene", "render", "terrain", "water", "volumetric", "simulation", "world",
    "modeling", "hair", "system", "paint", "scatter", "stylize", "sculpt"};
const std::set<std::string> kBottomEditors{
    "none", "dope_sheet", "graph_editor", "console", "terrain", "anim_graph",
    "geometry", "material", "assets"};
const std::set<std::string> kContextualDocks{"none", "paint", "hair", "sculpt", "terrain"};
const std::set<std::string> kViewportShading{"solid", "material_preview", "rendered", "matcap"};
const std::set<std::string> kSceneTypes{"project", "recipe"};
const std::set<std::string> kRendererPreferences{"auto", "vulkan", "optix", "cpu"};
const std::set<std::string> kPerformanceClasses{"light", "medium", "heavy"};

void addError(TemplateMetadata& item, const std::string& message) {
    item.errors.push_back(message);
}

void rejectUnknownFields(const json& object, const std::set<std::string>& allowed,
                         const char* scope, TemplateMetadata& item) {
    if (!object.is_object()) return;
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (allowed.find(it.key()) == allowed.end())
            addError(item, std::string("Unknown field in ") + scope + ": " + it.key());
    }
}

bool isSemanticVersion(const std::string& value) {
    int dots = 0;
    int digits = 0;
    for (char c : value) {
        if (c == '.') {
            if (digits == 0 || dots == 2) return false;
            ++dots;
            digits = 0;
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            ++digits;
        } else {
            return false;
        }
    }
    return dots == 2 && digits > 0;
}

bool isTemplateId(const std::string& value) {
    if (value.size() < 3 || value.size() > 96) return false;
    bool has_separator = false;
    bool previous_separator = true;
    for (char c : value) {
        const bool separator = c == '.' || c == '_' || c == '-';
        if (separator) {
            if (previous_separator) return false;
            previous_separator = true;
            has_separator = true;
        } else {
            if (!(c >= 'a' && c <= 'z') && !std::isdigit(static_cast<unsigned char>(c))) return false;
            previous_separator = false;
        }
    }
    return has_separator && !previous_separator;
}

bool safeRelativePath(const std::filesystem::path& value) {
    if (value.empty() || value.is_absolute() || value.has_root_name() || value.has_root_directory()) return false;
    for (const auto& part : value) {
        if (part == "..") return false;
    }
    return true;
}

std::optional<std::string> stringField(const json& object, const char* key,
                                       TemplateMetadata& item, std::size_t maximum = 0) {
    if (!object.contains(key) || !object[key].is_string()) {
        addError(item, std::string("Missing or invalid string field: ") + key);
        return std::nullopt;
    }
    const std::string value = object[key].get<std::string>();
    if (value.empty() || (maximum > 0 && value.size() > maximum)) {
        addError(item, std::string("String field is empty or too long: ") + key);
        return std::nullopt;
    }
    return value;
}

std::filesystem::path packagePath(const std::filesystem::path& root, const std::string& value,
                                  const char* field, TemplateMetadata& item, bool must_exist) {
    const std::filesystem::path relative(value);
    if (!safeRelativePath(relative)) {
        addError(item, std::string("Unsafe relative path in ") + field + ": " + value);
        return {};
    }
    const std::filesystem::path result = (root / relative).lexically_normal();
    if (must_exist && !std::filesystem::is_regular_file(result)) {
        addError(item, std::string("Required file does not exist for ") + field + ": " + value);
    }
    return result;
}

template <typename Set>
void requireEnum(const std::string& value, const Set& allowed, const char* field,
                 TemplateMetadata& item) {
    if (allowed.find(value) == allowed.end()) {
        addError(item, std::string("Unsupported value for ") + field + ": " + value);
    }
}

std::vector<std::filesystem::path> readAssetList(const json& assets, const char* key,
                                                  const std::filesystem::path& root,
                                                  TemplateMetadata& item, bool required) {
    std::vector<std::filesystem::path> result;
    if (!assets.contains(key) || !assets[key].is_array()) {
        addError(item, std::string("Missing or invalid asset array: ") + key);
        return result;
    }
    std::set<std::string> unique;
    for (const auto& value : assets[key]) {
        if (!value.is_string()) {
            addError(item, std::string("Non-string entry in asset array: ") + key);
            continue;
        }
        const std::string relative = value.get<std::string>();
        if (!unique.insert(relative).second) {
            addError(item, std::string("Duplicate asset path in ") + key + ": " + relative);
            continue;
        }
        auto resolved = packagePath(root, relative, key, item, required);
        if (!resolved.empty()) result.push_back(std::move(resolved));
    }
    return result;
}

} // namespace

TemplateRegistry& TemplateRegistry::instance() {
    static TemplateRegistry registry;
    return registry;
}

void TemplateRegistry::setSearchRoots(std::vector<std::filesystem::path> roots) {
    configured_roots_ = std::move(roots);
    refreshed_ = false;
}

void TemplateRegistry::resetSearchRoots() {
    configured_roots_.clear();
    refreshed_ = false;
}

std::vector<std::filesystem::path> TemplateRegistry::defaultSearchRoots() {
    std::vector<std::filesystem::path> roots;
    if (char* base = SDL_GetBasePath()) {
        roots.emplace_back(std::filesystem::path(base) / "assets" / "templates");
        SDL_free(base);
    }
    roots.emplace_back(std::filesystem::current_path() / "assets" / "templates");
    roots.emplace_back(std::filesystem::current_path() / "RayTrophiStudio" / "assets" / "templates");
    return roots;
}

std::vector<std::filesystem::path> TemplateRegistry::searchRoots() const {
    return configured_roots_.empty() ? defaultSearchRoots() : configured_roots_;
}

void TemplateRegistry::refresh() {
    entries_.clear();
    std::set<std::filesystem::path> seen_manifests;
    for (const auto& root : searchRoots()) {
        std::error_code ec;
        if (!std::filesystem::is_directory(root, ec)) continue;
        for (std::filesystem::recursive_directory_iterator it(
                 root, std::filesystem::directory_options::skip_permission_denied, ec), end;
             it != end; it.increment(ec)) {
            if (ec) { ec.clear(); continue; }
            if (!it->is_regular_file(ec) || it->path().filename() != "manifest.json") continue;
            const auto normalized = it->path().lexically_normal();
            if (seen_manifests.insert(normalized).second) entries_.push_back(validateManifest(normalized));
        }
    }

    std::unordered_map<std::string, std::vector<std::size_t>> by_id;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (!entries_[i].id.empty()) by_id[entries_[i].id].push_back(i);
    }
    for (const auto& pair : by_id) {
        if (pair.second.size() < 2) continue;
        for (std::size_t index : pair.second) {
            addError(entries_[index], "Duplicate template id: " + pair.first);
            entries_[index].valid = false;
        }
    }

    std::sort(entries_.begin(), entries_.end(), [](const auto& a, const auto& b) {
        if (a.sort_order != b.sort_order) return a.sort_order < b.sort_order;
        if (a.display_name != b.display_name) return a.display_name < b.display_name;
        if (a.id != b.id) return a.id < b.id;
        return a.manifest_path.generic_string() < b.manifest_path.generic_string();
    });
    refreshed_ = true;
}

const std::vector<TemplateMetadata>& TemplateRegistry::entries() const {
    if (!refreshed_) const_cast<TemplateRegistry*>(this)->refresh();
    return entries_;
}

const TemplateMetadata* TemplateRegistry::find(const std::string& id) const {
    const auto& all = entries();
    const auto it = std::find_if(all.begin(), all.end(), [&](const auto& item) { return item.id == id; });
    return it == all.end() ? nullptr : &*it;
}

TemplateMetadata TemplateRegistry::validateManifest(const std::filesystem::path& manifest_path) {
    TemplateMetadata item;
    item.manifest_path = manifest_path.lexically_normal();
    item.package_root = item.manifest_path.parent_path();

    json root;
    try {
        std::ifstream stream(item.manifest_path);
        if (!stream) {
            addError(item, "Manifest cannot be opened");
            return item;
        }
        stream >> root;
    } catch (const std::exception& error) {
        addError(item, std::string("Manifest JSON parse failed: ") + error.what());
        return item;
    }
    if (!root.is_object()) {
        addError(item, "Manifest root must be an object");
        return item;
    }
    rejectUnknownFields(root,
        {"schema_version", "id", "display_name", "description", "kind", "category",
         "sort_order", "compatibility", "preview", "scene", "ui_state", "renderer",
         "performance", "assets", "guidance"}, "manifest", item);

    if (auto value = stringField(root, "schema_version", item)) item.schema_version = *value;
    if (item.schema_version != "1.0") addError(item, "Unsupported schema_version: " + item.schema_version);
    if (auto value = stringField(root, "id", item, 96)) item.id = *value;
    if (!isTemplateId(item.id)) addError(item, "Invalid template id: " + item.id);
    if (auto value = stringField(root, "display_name", item, 48)) item.display_name = *value;
    if (auto value = stringField(root, "description", item, 140)) item.description = *value;
    if (auto value = stringField(root, "kind", item)) item.kind = *value;
    requireEnum(item.kind, kKinds, "kind", item);
    if (auto value = stringField(root, "category", item)) item.category = *value;
    requireEnum(item.category, kCategories, "category", item);
    if (!root.contains("sort_order") || !root["sort_order"].is_number_integer()) {
        addError(item, "Missing or invalid integer field: sort_order");
    } else {
        item.sort_order = root["sort_order"].get<int>();
        if (item.sort_order < 0 || item.sort_order > 10000) addError(item, "sort_order is out of range");
    }

    if (!root.contains("compatibility") || !root["compatibility"].is_object()) {
        addError(item, "Missing or invalid object: compatibility");
    } else {
        const auto& compatibility = root["compatibility"];
        rejectUnknownFields(compatibility,
            {"minimum_raytrophi_version", "maximum_raytrophi_version"}, "compatibility", item);
        if (auto value = stringField(compatibility, "minimum_raytrophi_version", item))
            item.minimum_raytrophi_version = *value;
        if (!isSemanticVersion(item.minimum_raytrophi_version)) addError(item, "Invalid minimum RayTrophi version");
        if (compatibility.contains("maximum_raytrophi_version")) {
            if (compatibility["maximum_raytrophi_version"].is_string())
                item.maximum_raytrophi_version = compatibility["maximum_raytrophi_version"].get<std::string>();
            else addError(item, "Invalid maximum_raytrophi_version");
            if (!item.maximum_raytrophi_version.empty() && !isSemanticVersion(item.maximum_raytrophi_version))
                addError(item, "Invalid maximum RayTrophi version");
        }
    }

    if (!root.contains("preview") || !root["preview"].is_object()) {
        addError(item, "Missing or invalid object: preview");
    } else {
        const auto& preview = root["preview"];
        rejectUnknownFields(preview, {"image", "alt"}, "preview", item);
        if (auto value = stringField(preview, "image", item))
            item.preview_path = packagePath(item.package_root, *value, "preview.image", item, true);
        if (preview.contains("alt") && preview["alt"].is_string()) item.preview_alt = preview["alt"].get<std::string>();
    }

    if (!root.contains("scene") || !root["scene"].is_object()) {
        addError(item, "Missing or invalid object: scene");
    } else {
        const auto& scene = root["scene"];
        rejectUnknownFields(scene, {"type", "path"}, "scene", item);
        if (auto value = stringField(scene, "type", item)) item.scene_type = *value;
        requireEnum(item.scene_type, kSceneTypes, "scene.type", item);
        if (auto value = stringField(scene, "path", item))
            item.scene_path = packagePath(item.package_root, *value, "scene.path", item, true);
    }

    if (!root.contains("ui_state") || !root["ui_state"].is_object()) {
        addError(item, "Missing or invalid object: ui_state");
    } else {
        const auto& ui = root["ui_state"];
        rejectUnknownFields(ui,
            {"properties_context", "bottom_editor", "contextual_dock", "contextual_dock_width",
             "viewport_shading", "frame_target", "show_timeline"}, "ui_state", item);
        if (auto value = stringField(ui, "properties_context", item)) item.ui_state.properties_context = *value;
        requireEnum(item.ui_state.properties_context, kPropertiesContexts, "ui_state.properties_context", item);
        if (auto value = stringField(ui, "bottom_editor", item)) item.ui_state.bottom_editor = *value;
        requireEnum(item.ui_state.bottom_editor, kBottomEditors, "ui_state.bottom_editor", item);
        if (auto value = stringField(ui, "contextual_dock", item)) item.ui_state.contextual_dock = *value;
        requireEnum(item.ui_state.contextual_dock, kContextualDocks, "ui_state.contextual_dock", item);
        if (ui.contains("contextual_dock_width")) {
            if (ui["contextual_dock_width"].is_number()) {
                item.ui_state.contextual_dock_width = ui["contextual_dock_width"].get<float>();
                if (item.ui_state.contextual_dock_width < 50.0f || item.ui_state.contextual_dock_width > 400.0f)
                    addError(item, "ui_state.contextual_dock_width is out of range");
            } else addError(item, "Invalid ui_state.contextual_dock_width");
        }
        if (auto value = stringField(ui, "viewport_shading", item)) item.ui_state.viewport_shading = *value;
        requireEnum(item.ui_state.viewport_shading, kViewportShading, "ui_state.viewport_shading", item);
        if (ui.contains("frame_target") && ui["frame_target"].is_string()) item.ui_state.frame_target = ui["frame_target"].get<std::string>();
        if (ui.contains("show_timeline")) {
            if (ui["show_timeline"].is_boolean()) item.ui_state.show_timeline = ui["show_timeline"].get<bool>();
            else addError(item, "Invalid ui_state.show_timeline");
        }
    }

    if (root.contains("renderer")) {
        if (!root["renderer"].is_object()) addError(item, "Invalid object: renderer");
        else {
            const auto& renderer = root["renderer"];
            rejectUnknownFields(renderer, {"preference", "allow_fallback"}, "renderer", item);
            if (renderer.contains("preference") && renderer["preference"].is_string())
                item.renderer_preference = renderer["preference"].get<std::string>();
            requireEnum(item.renderer_preference, kRendererPreferences, "renderer.preference", item);
            if (renderer.contains("allow_fallback")) {
                if (renderer["allow_fallback"].is_boolean()) item.renderer_allow_fallback = renderer["allow_fallback"].get<bool>();
                else addError(item, "Invalid renderer.allow_fallback");
            }
        }
    }

    if (!root.contains("performance") || !root["performance"].is_object()) {
        addError(item, "Missing or invalid object: performance");
    } else {
        const auto& performance = root["performance"];
        rejectUnknownFields(performance, {"class", "estimated_vram_mb"}, "performance", item);
        if (auto value = stringField(performance, "class", item)) item.performance_class = *value;
        requireEnum(item.performance_class, kPerformanceClasses, "performance.class", item);
        if (performance.contains("estimated_vram_mb")) {
            if (performance["estimated_vram_mb"].is_number_integer()) {
                item.estimated_vram_mb = performance["estimated_vram_mb"].get<int>();
                if (item.estimated_vram_mb < 0) addError(item, "performance.estimated_vram_mb cannot be negative");
            } else addError(item, "Invalid performance.estimated_vram_mb");
        }
    }

    if (!root.contains("assets") || !root["assets"].is_object()) {
        addError(item, "Missing or invalid object: assets");
    } else {
        rejectUnknownFields(root["assets"], {"required", "optional"}, "assets", item);
        item.required_assets = readAssetList(root["assets"], "required", item.package_root, item, true);
        item.optional_assets = readAssetList(root["assets"], "optional", item.package_root, item, false);
    }

    if (root.contains("guidance")) {
        if (!root["guidance"].is_object()) addError(item, "Invalid object: guidance");
        else {
            const auto& guidance = root["guidance"];
            rejectUnknownFields(guidance, {"path", "show_on_first_open"}, "guidance", item);
            if (auto value = stringField(guidance, "path", item))
                item.guidance_path = packagePath(item.package_root, *value, "guidance.path", item, true);
            if (guidance.contains("show_on_first_open")) {
                if (guidance["show_on_first_open"].is_boolean())
                    item.guidance_show_on_first_open = guidance["show_on_first_open"].get<bool>();
                else addError(item, "Invalid guidance.show_on_first_open");
            }
        }
    }

    item.valid = item.errors.empty();
    return item;
}

} // namespace raytrophi::templates
