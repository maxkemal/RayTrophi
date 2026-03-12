// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - MAIN ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════════
// NOTE: This file has been split into multiple modules for better maintainability.
// Implementations are located in:
//   - scene_ui_camera.cpp    : Camera settings (drawCameraContent)
//   - scene_ui_materials.cpp : Material editor (drawMaterialPanel)
//   - scene_ui_hierarchy.cpp : Scene tree (drawSceneHierarchy)
//   - scene_ui_lights.cpp    : Lights panel (drawLightsContent)
//   - scene_ui_gizmos.cpp    : 3D gizmos & bounding boxes
//   - scene_ui_viewport.cpp  : Overlays (Focus/Zoom/Exposure/Dolly)
//   - scene_ui_selection.cpp : Selection logic & Marquee
//   - scene_ui_world.cpp     : World environment settings
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include <thread>
#include <filesystem>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cctype>
#include <fstream>
#include "json.hpp"
#include "ProjectManager.h"
#include "TerrainManager.h"
#include "SceneSerializer.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "Backend/IBackend.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "scene_data.h"    // Added explicit include
#include "ui_modern.h"
#include "imgui.h"
#include <SDL_image.h>
#include "stb_image.h"
#include "ImGuizmo.h"  // Transform gizmo
#include <string>
#include <memory>  // For std::make_unique
#include "KeyframeSystem.h"   // For keyframe animation
#include "scene_ui_guides.hpp" // Viewport guides (safe areas, letterbox, grids)
#include "TimelineWidget.h"   // Custom timeline widget
#include "scene_data.h"
#include "scene_ui_water.hpp"   // Water panel implementation
#include "scene_ui_river.hpp"   // River spline editor
#include "WaterSystem.h"        // Water Manager for update loop
#include "scene_ui_terrain.hpp" // Terrain panel implementation
#include "scene_ui_animgraph.hpp" // Animation Graph Editor
#include "scene_ui_gas.hpp"     // Gas Simulation panel
#include "scene_ui_forcefield.hpp" // Force Field panel
#include "ParallelBVHNode.h"
#include "Triangle.h"  // For object hierarchy
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "PrincipledBSDF.h" // For material editing
#include "Volumetric.h"     // For volumetric material
#include "VDBVolume.h"      // For VDB volume UI panel
#include "VolumetricRenderer.h"
#include "AssimpLoader.h"  // For scene rebuild after object changes
#include "SceneCommand.h"  // For undo/redo
#include "default_scene_creator.hpp"
#include "SceneSerializer.h"
#include "ProjectManager.h"  // Project system
#include "MaterialManager.h"  // For material editing
#include <map>  // For mesh grouping
#include <unordered_set>  // For fast deletion lookup
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>  // SHBrowseForFolder için
#include <chrono>  // Playback timing için
#include <filesystem>  // Frame dosyalarını kontrol için
#include <unordered_map>
#include <atomic>
#include <vector>
#include <sstream>

bool show_documentation_window = false; // Global toggle (unused now, kept for linkage if needed or remove)

namespace {
bool readBinaryFileBytes(const std::filesystem::path& path, std::vector<unsigned char>& out_bytes) {
    out_bytes.clear();

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return false;
    }

    out_bytes.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    return file.read(reinterpret_cast<char*>(out_bytes.data()), size).good();
}

bool loadPreviewTextureFromPath(SDL_Renderer* active_renderer,
                                const std::filesystem::path& preview_path,
                                SDL_Texture*& out_texture,
                                int& out_width,
                                int& out_height) {
    out_texture = nullptr;
    out_width = 0;
    out_height = 0;

    if (preview_path.empty() || !active_renderer || !std::filesystem::exists(preview_path)) {
        return false;
    }

    std::vector<unsigned char> image_bytes;
    if (!readBinaryFileBytes(preview_path, image_bytes)) {
        return false;
    }

    SDL_RWops* rw = SDL_RWFromConstMem(image_bytes.data(), static_cast<int>(image_bytes.size()));
    if (!rw) {
        return false;
    }

    SDL_Surface* preview_surface = IMG_Load_RW(rw, 1);
    if (!preview_surface) {
        int image_width = 0;
        int image_height = 0;
        int channels = 0;
        stbi_uc* stbi_pixels = stbi_load_from_memory(
            image_bytes.data(),
            static_cast<int>(image_bytes.size()),
            &image_width,
            &image_height,
            &channels,
            4);
        if (!stbi_pixels) {
            return false;
        }

        SDL_Surface* rgba_surface = SDL_CreateRGBSurfaceWithFormatFrom(
            stbi_pixels,
            image_width,
            image_height,
            32,
            image_width * 4,
            SDL_PIXELFORMAT_RGBA32);
        if (!rgba_surface) {
            stbi_image_free(stbi_pixels);
            return false;
        }

        preview_surface = SDL_ConvertSurfaceFormat(rgba_surface, SDL_PIXELFORMAT_RGBA32, 0);
        SDL_FreeSurface(rgba_surface);
        stbi_image_free(stbi_pixels);
        if (!preview_surface) {
            return false;
        }
    }

    SDL_Texture* preview_texture = SDL_CreateTextureFromSurface(active_renderer, preview_surface);
    if (!preview_texture) {
        SDL_FreeSurface(preview_surface);
        return false;
    }

    out_texture = preview_texture;
    out_width = preview_surface->w;
    out_height = preview_surface->h;
    SDL_FreeSurface(preview_surface);
    return true;
}

std::string makeUniqueAssetImportPrefix(const std::filesystem::path& asset_path) {
    static std::atomic<uint64_t> import_counter{ 1 };
    const std::string stem = asset_path.stem().string().empty() ? "asset" : asset_path.stem().string();
    const uint64_t counter = import_counter.fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    return stem + "_" + std::to_string(stamp) + "_" + std::to_string(counter);
}

bool computeAssetPreviewBounds(const std::filesystem::path& asset_path, Vec3& out_min, Vec3& out_max) {
    Assimp::Importer importer;
    unsigned int import_flags =
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality;

    const std::string ext = AssetRegistry::toLowerCopy(asset_path.extension().string());
    if (ext == ".fbx") {
        import_flags |= aiProcess_GlobalScale;
    }

    std::vector<unsigned char> file_bytes;
    const aiScene* scene = nullptr;
    if (readBinaryFileBytes(asset_path, file_bytes)) {
        scene = importer.ReadFileFromMemory(file_bytes.data(), file_bytes.size(), import_flags, ext.c_str());
    }
    if (!scene) {
        scene = importer.ReadFile(asset_path.string(), import_flags);
    }
    if (!scene || !scene->mRootNode) {
        return false;
    }

    aiVector3D bb_min(1e30f, 1e30f, 1e30f);
    aiVector3D bb_max(-1e30f, -1e30f, -1e30f);
    bool found_vertex = false;

    std::function<void(aiNode*, const aiMatrix4x4&)> walk =
        [&](aiNode* node, const aiMatrix4x4& parent_transform) {
            if (!node) {
                return;
            }

            const aiMatrix4x4 world = parent_transform * node->mTransformation;
            for (unsigned int mesh_idx = 0; mesh_idx < node->mNumMeshes; ++mesh_idx) {
                const aiMesh* mesh = scene->mMeshes[node->mMeshes[mesh_idx]];
                if (!mesh || !mesh->HasPositions()) {
                    continue;
                }

                for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
                    const aiVector3D p = world * mesh->mVertices[v];
                    bb_min.x = (std::min)(bb_min.x, p.x);
                    bb_min.y = (std::min)(bb_min.y, p.y);
                    bb_min.z = (std::min)(bb_min.z, p.z);
                    bb_max.x = (std::max)(bb_max.x, p.x);
                    bb_max.y = (std::max)(bb_max.y, p.y);
                    bb_max.z = (std::max)(bb_max.z, p.z);
                    found_vertex = true;
                }
            }

            for (unsigned int child_idx = 0; child_idx < node->mNumChildren; ++child_idx) {
                walk(node->mChildren[child_idx], world);
            }
        };

    walk(scene->mRootNode, aiMatrix4x4());

    if (!found_vertex) {
        return false;
    }

    out_min = Vec3(bb_min.x, bb_min.y, bb_min.z);
    out_max = Vec3(bb_max.x, bb_max.y, bb_max.z);
    return true;
}

bool computeVDBPreviewBounds(const AssetRecord& asset, Vec3& out_min, Vec3& out_max) {
    if (asset.entry_path.empty()) {
        return false;
    }

    VDBVolume vdb;
    const bool loaded = asset.is_sequence ? vdb.loadVDBSequence(asset.entry_path.string()) : vdb.loadVDB(asset.entry_path.string());
    if (!loaded) {
        return false;
    }

    Vec3 bmin = vdb.getLocalBoundsMin();
    Vec3 bmax = vdb.getLocalBoundsMax();
    const Vec3 size = bmax - bmin;
    float max_dim = (std::max)(size.x, (std::max)(size.y, size.z));
    float scale_factor = 1.0f;

    if (max_dim > 50.0f) {
        scale_factor = 5.0f / max_dim;
    } else if (max_dim > 0.0f && max_dim < 0.01f) {
        scale_factor = 5.0f / (std::max)(max_dim, 0.001f);
    }

    scale_factor = (std::max)(0.0001f, (std::min)(scale_factor, 1000.0f));

    const Vec3 center = (bmin + bmax) * 0.5f;
    const Vec3 pivot_offset(-center.x, -bmin.y, -center.z);
    out_min = (bmin + pivot_offset) * scale_factor;
    out_max = (bmax + pivot_offset) * scale_factor;
    vdb.unload();
    return true;
}

bool isPlaceableAssetRecord(const AssetRecord& asset) {
    return asset.asset_kind == "model" || asset.asset_kind == "vdb" || asset.asset_kind == "vdb_sequence" || asset.asset_kind == "anim_clip";
}

bool computeAssetPreviewBounds(const AssetRecord& asset, Vec3& out_min, Vec3& out_max) {
    if (asset.asset_kind == "vdb" || asset.asset_kind == "vdb_sequence") {
        return computeVDBPreviewBounds(asset, out_min, out_max);
    }
    if (asset.asset_kind == "anim_clip") {
        return false;
    }
    return computeAssetPreviewBounds(asset.entry_path, out_min, out_max);
}

std::string retargetAnimationNodeName(const std::string& node_name, const std::string& source_prefix, const std::string& target_prefix) {
    if (source_prefix.empty()) {
        return node_name;
    }

    const std::string source_token = source_prefix + "_";
    if (node_name.find(source_token) != 0) {
        return node_name;
    }

    const std::string local_name = node_name.substr(source_token.size());
    if (target_prefix.empty()) {
        return local_name;
    }
    return target_prefix + "_" + local_name;
}

template <typename TKey>
std::map<std::string, std::vector<TKey>> retargetAnimationKeyMap(
    const std::map<std::string, std::vector<TKey>>& source,
    const std::string& source_prefix,
    const std::string& target_prefix) {
    std::map<std::string, std::vector<TKey>> result;
    for (const auto& [node_name, keys] : source) {
        result[retargetAnimationNodeName(node_name, source_prefix, target_prefix)] = keys;
    }
    return result;
}

std::string makeUniqueAnimationClipName(const std::vector<std::shared_ptr<AnimationData>>& existing_clips, const std::string& base_name) {
    std::string candidate = base_name.empty() ? "Anim Clip" : base_name;
    std::unordered_set<std::string> used_names;
    for (const auto& clip : existing_clips) {
        if (clip) {
            used_names.insert(clip->name);
        }
    }

    if (used_names.find(candidate) == used_names.end()) {
        return candidate;
    }

    for (int suffix = 2; suffix < 10000; ++suffix) {
        const std::string numbered = candidate + " " + std::to_string(suffix);
        if (used_names.find(numbered) == used_names.end()) {
            return numbered;
        }
    }

    return candidate;
}

std::string findSelectedModelImportName(UIContext& ctx) {
    if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        for (const auto& model_ctx : ctx.scene.importedModelContexts) {
            for (const auto& member : model_ctx.members) {
                if (member == ctx.selection.selected.object) {
                    return model_ctx.importName;
                }
            }
        }
    }

    const std::string selected_name = ctx.selection.selected.name;
    if (!selected_name.empty()) {
        for (const auto& model_ctx : ctx.scene.importedModelContexts) {
            if (selected_name == model_ctx.importName || selected_name.find(model_ctx.importName + "_") == 0) {
                return model_ctx.importName;
            }
        }
    }

    if (ctx.scene.importedModelContexts.size() == 1) {
        return ctx.scene.importedModelContexts.front().importName;
    }

    return {};
}

std::string fitTextToWidth(const std::string& text, float max_width) {
    if (text.empty() || max_width <= 8.0f) {
        return text;
    }

    if (ImGui::CalcTextSize(text.c_str()).x <= max_width) {
        return text;
    }

    const std::string ellipsis = "...";
    const float ellipsis_width = ImGui::CalcTextSize(ellipsis.c_str()).x;
    if (ellipsis_width >= max_width) {
        return ellipsis;
    }

    std::string trimmed = text;
    while (!trimmed.empty()) {
        trimmed.pop_back();
        const std::string candidate = trimmed + ellipsis;
        if (ImGui::CalcTextSize(candidate.c_str()).x <= max_width) {
            return candidate;
        }
    }

    return ellipsis;
}

std::string trimCopy(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::vector<std::string> splitTagString(const std::string& text) {
    std::vector<std::string> tags;
    std::unordered_set<std::string> seen;
    std::string current;

    auto flush_token = [&]() {
        std::string token = AssetRegistry::toLowerCopy(trimCopy(current));
        current.clear();
        if (!token.empty() && seen.insert(token).second) {
            tags.push_back(token);
        }
    };

    for (char ch : text) {
        if (ch == ',' || ch == ';' || ch == '\n') {
            flush_token();
            continue;
        }
        current.push_back(ch);
    }
    flush_token();
    return tags;
}

std::string joinTags(const std::vector<std::string>& tags) {
    std::string joined;
    for (size_t i = 0; i < tags.size(); ++i) {
        if (i > 0) {
            joined += ", ";
        }
        joined += tags[i];
    }
    return joined;
}

std::string makeSmartFolderDefaultName(const std::string& folder_relative_dir,
                                       const std::string& search,
                                       const std::string& tag_filter,
                                       bool favorites_only) {
    if (favorites_only) {
        return "Favorites";
    }
    if (!tag_filter.empty()) {
        return "Tags: " + tag_filter;
    }
    if (!search.empty()) {
        return "Search: " + search;
    }
    if (!folder_relative_dir.empty()) {
        return folder_relative_dir;
    }
    return "Collection";
}

bool saveAssetMetadataChanges(AssetRegistry& registry, const AssetRecord& source_asset, bool favorite, const std::vector<std::string>& tags) {
    AssetRecord updated = source_asset;
    updated.favorite = favorite;
    updated.tags = tags;
    return registry.writeMetadataStub(updated);
}

bool assetMatchesFilters(const AssetRecord& asset,
                         const std::string& lowered_search,
                         const std::vector<std::string>& required_tags,
                         bool favorites_only) {
    if (favorites_only && !asset.favorite) {
        return false;
    }

    if (!required_tags.empty()) {
        std::unordered_set<std::string> asset_tags;
        for (const auto& tag : asset.tags) {
            asset_tags.insert(AssetRegistry::toLowerCopy(tag));
        }
        for (const auto& required_tag : required_tags) {
            if (asset_tags.find(required_tag) == asset_tags.end()) {
                return false;
            }
        }
    }

    if (lowered_search.empty()) {
        return true;
    }

    const std::string rel_file = AssetRegistry::toLowerCopy(asset.relative_entry_path.generic_string());
    const std::string rel_dir = AssetRegistry::toLowerCopy(asset.relative_directory.generic_string());
    const std::string display_name = AssetRegistry::toLowerCopy(asset.name);
    const std::string kind = AssetRegistry::toLowerCopy(asset.asset_kind);
    const std::string tag_blob = AssetRegistry::toLowerCopy(joinTags(asset.tags));

    return rel_file.find(lowered_search) != std::string::npos ||
           rel_dir.find(lowered_search) != std::string::npos ||
           display_name.find(lowered_search) != std::string::npos ||
           kind.find(lowered_search) != std::string::npos ||
           tag_blob.find(lowered_search) != std::string::npos;
}

std::filesystem::path normalizeAbsolutePath(const std::filesystem::path& path) {
    if (path.empty()) {
        return {};
    }

    std::error_code ec;
    const std::filesystem::path absolute_path = std::filesystem::absolute(path, ec);
    if (ec) {
        return path;
    }
    return absolute_path.lexically_normal();
}

std::filesystem::path projectDirectoryPath() {
    if (g_project.current_file_path.empty()) {
        return {};
    }

    std::error_code ec;
    const std::filesystem::path project_path = std::filesystem::absolute(g_project.current_file_path, ec);
    if (ec) {
        return {};
    }
    return project_path.parent_path();
}

nlohmann::json serializeLibraryPathEntry(const std::filesystem::path& path) {
    nlohmann::json j;
    const std::filesystem::path normalized = normalizeAbsolutePath(path);
    const std::filesystem::path project_dir = projectDirectoryPath();

    std::error_code ec;
    const std::filesystem::path relative = project_dir.empty() ? std::filesystem::path() : std::filesystem::relative(normalized, project_dir, ec);
    if (!project_dir.empty() && !ec && !relative.empty()) {
        j["path"] = relative.generic_string();
        j["relative_to_project"] = true;
    } else {
        j["path"] = normalized.string();
        j["relative_to_project"] = false;
    }
    return j;
}

std::filesystem::path deserializeLibraryPathEntry(const nlohmann::json& j) {
    if (!j.is_object()) {
        return {};
    }

    const std::string raw_path = j.value("path", std::string());
    if (raw_path.empty()) {
        return {};
    }

    std::filesystem::path path(raw_path);
    if (j.value("relative_to_project", false)) {
        const std::filesystem::path project_dir = projectDirectoryPath();
        if (!project_dir.empty()) {
            path = project_dir / path;
        }
    }

    return normalizeAbsolutePath(path);
}

void ensureDefaultAssetLibrary(std::vector<std::filesystem::path>& library_paths) {
    const std::filesystem::path default_root = normalizeAbsolutePath(AssetRegistry::resolveDefaultRoot());
    if (default_root.empty()) {
        return;
    }

    auto it = std::find_if(library_paths.begin(), library_paths.end(), [&](const std::filesystem::path& candidate) {
        return normalizeAbsolutePath(candidate) == default_root;
    });
    if (it == library_paths.end()) {
        library_paths.insert(library_paths.begin(), default_root);
    } else if (it != library_paths.begin()) {
        const std::filesystem::path existing = *it;
        library_paths.erase(it);
        library_paths.insert(library_paths.begin(), existing);
    }
}

bool refreshAssetLibrarySafely(AssetRegistry& registry, const std::filesystem::path& root_path, std::string* out_error = nullptr) {
    try {
        const std::filesystem::path normalized = normalizeAbsolutePath(root_path);
        std::error_code ec;
        if (normalized.empty()) {
            if (out_error) *out_error = "Empty path";
            return false;
        }
        if (!std::filesystem::exists(normalized, ec)) {
            if (out_error) *out_error = ec ? ("Path check failed: " + ec.message()) : "Folder does not exist";
            return false;
        }
        ec.clear();
        if (!std::filesystem::is_directory(normalized, ec)) {
            if (out_error) *out_error = ec ? ("Directory check failed: " + ec.message()) : "Selected path is not a folder";
            return false;
        }
        if (!registry.refresh(normalized)) {
            if (out_error) *out_error = "Asset scan failed";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        if (out_error) *out_error = e.what();
        return false;
    } catch (...) {
        if (out_error) *out_error = "Unknown error";
        return false;
    }
}
}



static int new_width = image_width;
static int new_height = image_height;
static int aspect_w = 16;
static int aspect_h = 9;
static bool modelLoaded = false;
static bool loadFeedback = false; // geçici hata geri bildirimi
static float feedbackTimer = 0.0f;
// show_animation_panel is now a member of SceneUI class

// Pivot Mode State: 0=Median Point (Group), 1=Individual Origins
// Pivot Mode State is now a member of SceneUI class (see scene_ui.h) 

// Not: ScaleColor ve HelpMarker artık UIWidgets namespace'inde tanımlı

struct ResolutionPreset {
    const char* name;
    int w, h;
    int bw, bh;
};

static ResolutionPreset presets[] = {
    { "Custom", 0,0,0,0 },
    { "HD 720p", 1280,720, 16,9 },
    { "Full HD 1080p", 1920,1080, 16,9 },
    { "1440p", 2560,1440, 16,9 },
    { "4K UHD", 3840,2160, 16,9 },
    { "DCI 2K", 2048,1080, 19,10 },
    { "DCI 4K", 4096,2160, 19,10 },
    { "CinemaScope 4K", 4096,1716, 239,100 },
    { "Scope HD", 1920,804, 239,100 },
    { "2.35:1 HD", 1920,817, 235,100 },
    { "Vertical 1080x1920", 1080,1920, 9,16 }
};

static int preset_index = 0;




std::string SceneUI::openFileDialogW(const wchar_t* filter, const std::string& initialDir, const std::string& defaultFilename) {
    wchar_t filename[MAX_PATH] = L"";
    wchar_t initialDirW[MAX_PATH] = L"";
    
    // Convert initial directory to wide string if provided
    if (!initialDir.empty()) {
        MultiByteToWideChar(CP_UTF8, 0, initialDir.c_str(), -1, initialDirW, MAX_PATH);
    }

    // Convert default filename to wide string if provided
    if (!defaultFilename.empty()) {
        MultiByteToWideChar(CP_UTF8, 0, defaultFilename.c_str(), -1, filename, MAX_PATH);
    }
    
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle = L"Select a file";
    ofn.hwndOwner = GetActiveWindow();
    ofn.lpstrInitialDir = initialDir.empty() ? nullptr : initialDirW;
    
    if (GetOpenFileNameW(&ofn)) {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, filename, -1, nullptr, 0, nullptr, nullptr);
        std::string utf8_path(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, utf8_path.data(), size_needed, nullptr, nullptr);
        utf8_path.resize(size_needed - 1); // null terminatörü çıkar
        return utf8_path;
    }
    return "";
}

std::string SceneUI::saveFileDialogW(const wchar_t* filter, const wchar_t* defExt) {
    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = GetActiveWindow();
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH; // Initialize buffer with 0
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
    ofn.lpstrDefExt = defExt;
    
    if (GetSaveFileNameW(&ofn)) {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, filename, -1, nullptr, 0, nullptr, nullptr);
        std::string utf8_path(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, utf8_path.data(), size_needed, nullptr, nullptr);
        utf8_path.resize(size_needed - 1); 
        return utf8_path;
    }
    return "";
}

std::string SceneUI::selectFolderDialogW(const wchar_t* title) {
    BROWSEINFOW bi = { 0 };
    bi.lpszTitle = title;
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_USENEWUI;
    bi.hwndOwner = GetActiveWindow();

    LPITEMIDLIST pidl = SHBrowseForFolderW(&bi);
    if (pidl != nullptr) {
        wchar_t path[MAX_PATH];
        if (SHGetPathFromIDListW(pidl, path)) {
            int size_needed = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
            std::string utf8_path(size_needed, 0);
            WideCharToMultiByte(CP_UTF8, 0, path, -1, utf8_path.data(), size_needed, nullptr, nullptr);
            utf8_path.resize(size_needed - 1);

            // Free pidl
            IMalloc* imalloc = nullptr;
            if (SUCCEEDED(SHGetMalloc(&imalloc))) {
                imalloc->Free(pidl);
                imalloc->Release();
            }

            return utf8_path;
        }
    }
    return "";
}

static std::string active_model_path = "No file selected yet.";

// ═══════════════════════════════════════════════════════════
// GLOBAL UI WIDGETS IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

// Initialize default theme (Retro Monochrome)
SceneUI::LCDTheme SceneUI::currentTheme = {
    IM_COL32(200, 200, 200, 255), // Lit: White-ish
    IM_COL32(40, 45, 50, 255),    // Off: Dark Gray
    IM_COL32(20, 20, 20, 255),    // Bg: Almost Black
    IM_COL32(180, 230, 255, 255), // Text: Light Cyan
    false
};

// Initialize default style
SceneUI::UISliderStyle SceneUI::globalSliderStyle = SceneUI::UISliderStyle::Modern;

bool SceneUI::DrawSmartFloat(const char* id, const char* label, float* value, float min, float max, 
                           const char* format, bool keyed, 
                           std::function<void()> onKeyframeClick, int segments) 
{
    if (globalSliderStyle == UISliderStyle::RetroLCD) {
        return DrawLCDSlider(id, label, value, min, max, format, keyed, onKeyframeClick, segments);
    }
    else {
        // Modern / Standard Style
        ImGui::PushID(id);
        bool changed = false;
        
        // Keyframe Button (if callback provided)
        if (onKeyframeClick) {
            float s = ImGui::GetFrameHeight();
            ImVec2 kf_pos = ImGui::GetCursorScreenPos();
            bool kf_clicked = ImGui::InvisibleButton("kf", ImVec2(s, s));
            
            // Standard diamond drawing
            ImU32 kf_bg = keyed ? IM_COL32(255, 200, 0, 255) : IM_COL32(40, 40, 40, 255);
            ImU32 kf_border = IM_COL32(180, 180, 180, 255);
            if (ImGui::IsItemHovered()) {
                kf_border = IM_COL32(255, 255, 255, 255);
                ImGui::SetTooltip(keyed ? "Click to REMOVE keyframe" : "Click to ADD keyframe");
            }
            ImDrawList* dl = ImGui::GetWindowDrawList();
            float cx = kf_pos.x + s * 0.5f;
            float cy = kf_pos.y + s * 0.5f;
            float r = s * 0.22f;
            ImVec2 p[4] = { ImVec2(cx, cy - r), ImVec2(cx + r, cy), ImVec2(cx, cy + r), ImVec2(cx - r, cy) };
            dl->AddQuadFilled(p[0], p[1], p[2], p[3], kf_bg);
            dl->AddQuad(p[0], p[1], p[2], p[3], kf_border, 1.0f);
            
            if (kf_clicked) onKeyframeClick();
            ImGui::SameLine();
        }

        // Standard Slider
        // Adjust width if keyframe button exists or not to align
        if (ImGui::SliderFloat(label, value, min, max, format)) {
            changed = true;
        }
        
        ImGui::PopID();
        return changed;
    }
}


bool SceneUI::DrawLCDSlider(const char* id, const char* label, float* value, float min, float max, 
                          const char* format, bool keyed, 
                          std::function<void()> onKeyframeClick, int segments) 
{
    ImGui::PushID(id);
    bool changed = false;
    float t = (*value - min) / (max - min);
    int lit = (int)(t * segments);
    
    // Keyframe diamond button (Only if callback provided)
    if (onKeyframeClick) {
        float s = ImGui::GetFrameHeight();
        ImVec2 kf_pos = ImGui::GetCursorScreenPos();
        bool kf_clicked = ImGui::InvisibleButton("kf", ImVec2(s, s));
        
        ImU32 kf_bg = keyed ? IM_COL32(100, 180, 255, 255) : IM_COL32(40, 40, 40, 255);
        if (currentTheme.isRetroGreen) {
            kf_bg = keyed ? IM_COL32(50, 255, 50, 255) : IM_COL32(20, 50, 20, 255);
        }
        
        ImU32 kf_border = ImGui::IsItemHovered() ? IM_COL32(255, 255, 255, 255) : IM_COL32(150, 150, 150, 255);
        if (ImGui::IsItemHovered()) {
            if (!currentTheme.isRetroGreen)
                kf_bg = keyed ? IM_COL32(120, 200, 255, 255) : IM_COL32(70, 70, 70, 255);
            ImGui::SetTooltip(keyed ? "%s: Click to REMOVE keyframe" : "%s: Click to ADD keyframe", label);
        }
        
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float cx = kf_pos.x + s * 0.5f;
        float cy = kf_pos.y + s * 0.5f;
        float r = s * 0.22f;
        dl->AddQuadFilled(ImVec2(cx, cy-r), ImVec2(cx+r, cy), ImVec2(cx, cy+r), ImVec2(cx-r, cy), kf_bg);
        dl->AddQuad(ImVec2(cx, cy-r), ImVec2(cx+r, cy), ImVec2(cx, cy+r), ImVec2(cx-r, cy), kf_border, 1.0f);
        
        if (kf_clicked) onKeyframeClick();
        
        ImGui::SameLine();
    } else {
        // Just add some spacing if no keyframe button
        ImGui::Dummy(ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight()));
        ImGui::SameLine();
    }
    
    // Label (fixed width)
    ImGui::Text("%-6s", label);
    ImGui::SameLine();
    
    // LCD Bar
    ImVec2 bar_pos = ImGui::GetCursorScreenPos();
    float segW = 6.0f;
    float segH = 14.0f;
    float gap = 2.0f;
    float totalW = segments * (segW + gap);
    
    ImDrawList* dl = ImGui::GetWindowDrawList();
    
    for (int i = 0; i < segments; i++) {
        float x = bar_pos.x + i * (segW + gap);
        ImU32 color;
        if (i < lit) {
            color = currentTheme.litColor;
        } else {
            color = currentTheme.offColor;
        }
        dl->AddRectFilled(ImVec2(x, bar_pos.y), ImVec2(x + segW, bar_pos.y + segH), color, 1.0f);
        dl->AddRect(ImVec2(x, bar_pos.y), ImVec2(x + segW, bar_pos.y + segH), currentTheme.bgColor, 1.0f);
    }
    
    // Invisible slider over the bar
    ImGui::SetCursorScreenPos(bar_pos);
    ImGui::InvisibleButton("bar", ImVec2(totalW, segH));
    if (ImGui::IsItemActive()) {
        float mx = ImGui::GetIO().MousePos.x - bar_pos.x;
        float newT = std::clamp(mx / totalW, 0.0f, 1.0f);
        *value = min + newT * (max - min);
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        ImGui::SetTooltip("Drag to adjust value");
    }
    
    ImGui::SameLine();
    
    // Value Input
    ImGui::PushItemWidth(60);
    char inputId[64];
    snprintf(inputId, sizeof(inputId), "##input_%s", id);
    
    // Style input specific to theme
    ImGui::PushStyleColor(ImGuiCol_Text, currentTheme.textValColor);
    if (ImGui::InputFloat(inputId, value, 0.0f, 0.0f, format)) {
        *value = std::clamp(*value, min, max); // Clamp manual input
        changed = true;
    }
    ImGui::PopStyleColor();
    ImGui::PopItemWidth();
    
    ImGui::PopID();
    return changed;
}


// Helper Methods Implementation
void SceneUI::ClampWindowToDisplay()
{
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 disp = io.DisplaySize;

    ImVec2 win_pos = ImGui::GetWindowPos();
    ImVec2 win_size = ImGui::GetWindowSize();

    // Eğer pencere invisible veya 0 boyutluysa çık
    if (win_size.x <= 0.0f || win_size.y <= 0.0f) return;

    float x = win_pos.x;
    float y = win_pos.y;

    // Sağ/bottom taşmaları düzelt
    if (x + win_size.x > disp.x) x = disp.x - win_size.x;
    if (y + win_size.y > disp.y) y = disp.y - win_size.y;

    // Negatif değerlere izin verme
    if (x < 0.0f) x = 0.0f;
    if (y < 0.0f) y = 0.0f;

    // Pozisyon değiştiyse uygula
    if (x != win_pos.x || y != win_pos.y) {
        ImGui::SetWindowPos(ImVec2(x, y), ImGuiCond_Always);
    }

    // Eğer pencere ekran boyutuna göre taşarsa, boyutu da düzelt
    bool size_changed = false;
    float new_width = win_size.x;
    float new_height = win_size.y;

    if (win_size.x > disp.x) { new_width = disp.x; size_changed = true; }
    if (win_size.y > disp.y) { new_height = disp.y; size_changed = true; }

    if (size_changed) {
        ImGui::SetWindowSize(ImVec2(new_width, new_height), ImGuiCond_Always);
    }
}

// Timeline Panel - Blender-style Custom Timeline Widget
void SceneUI::drawTimelineContent(UIContext& ctx)
{
    // Use the timeline member widget
    timeline.draw(ctx);
}


// Wrapper for compatibility (if needed) but essentially deprecated as a window creator
void SceneUI::drawTimelinePanel(UIContext& ctx, float screen_y) {
    drawTimelineContent(ctx);
}

// Eski drawAnimationSettings metodunu kaldırdık, artık kullanılmıyor
void SceneUI::drawAnimationSettings(UIContext& ctx)
{
    // Bu metod artık kullanılmıyor - timeline panel'e taşındı
}

void SceneUI::drawLogPanelEmbedded()
{
    ImFont* tinyFont = ImGui::GetIO().Fonts->Fonts.back();
    ImGui::PushFont(tinyFont);

    // Başlık reset zamanlayıcıları (global/statik olarak zaten tanımlı olmalı)
    if (titleChanged && ImGui::GetTime() > titleResetTime) {
        logTitle = "Scene Log";
        titleChanged = false;
    }

    // Başlık rengi varsa uygulayıp geri al
    if (titleChanged)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));

    // AllowItemOverlap ile header oluşturuyoruz — böylece aynı satırda başka butonlar çalışır
    ImGuiTreeNodeFlags hdrFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;
    bool open = ImGui::CollapsingHeader(logTitle.c_str(), hdrFlags);

    if (titleChanged)
        ImGui::PopStyleColor();

    // Header ile aynı satıra Copy butonunu koy
    // İtem örtüşmesine izin verdiğimiz için buton tıklanabilir kalır
    float avail = ImGui::GetContentRegionAvail().x;
    ImGui::SameLine(avail - 60.0f); // butonu sağa sabitliyoruz (60 px boşluk bırak)
    if (ImGui::SmallButton("Copy"))
    {
        std::vector<LogEntry> lines;
        g_sceneLog.getLines(lines);

        std::string total;
        total.reserve(lines.size() * 64);
        for (auto& e : lines) {
            const char* prefix =
                (e.level == LogLevel::Info) ? "INFO" :
                (e.level == LogLevel::Warning) ? "WARN" : "ERROR";
            total += "[" + std::string(prefix) + "] " + e.msg + "\n";
        }

        ImGui::SetClipboardText(total.c_str());

        // Başlığa kısa süreli bildirim ver
        logTitle = "Scene Log  (Copied)";
        titleResetTime = ImGui::GetTime() + 2.0f;
        titleChanged = true;
    }

    // Eğer header açıksa logu göster
    if (open)
    {
        // Mevcut boşluğu doldur (en az 100px)
        float avail_y = ImGui::GetContentRegionAvail().y;
        if (avail_y < 100.0f) avail_y = 150.0f; 

        ImGui::BeginChild("scroll_log", ImVec2(0, avail_y), true);

        static size_t lastCount = 0;
        std::vector<LogEntry> lines;
        g_sceneLog.getLines(lines);

        for (auto& e : lines)
        {
            ImVec4 color =
                (e.level == LogLevel::Info) ? ImVec4(1, 1, 1, 1) :
                (e.level == LogLevel::Warning) ? ImVec4(1, 1, 0, 1) :
                ImVec4(1, 0, 0, 1);

            const char* prefix =
                (e.level == LogLevel::Info) ? "INFO" :
                (e.level == LogLevel::Warning) ? "WARN" : "ERROR";

            ImGui::TextColored(color, "[%s] %s", prefix, e.msg.c_str());
        }

        if (lines.size() > lastCount)
            ImGui::SetScrollHereY(1.0f);
        lastCount = lines.size();

        ImGui::EndChild();
    }

    ImGui::PopFont();
}

void SceneUI::drawThemeSelector() {
    UIWidgets::DrawThemeSelector(panel_alpha);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ═══════════════════════════════════════════════════════════
    // LCD WIDGET THEME
    // ═══════════════════════════════════════════════════════════
    if (ImGui::CollapsingHeader("LCD Widget Theme", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        
        static bool init_theme = true;
        
        // ── SLIDER STYLE SELECTOR ──
        int styleIdx = (globalSliderStyle == UISliderStyle::Modern) ? 0 : 1;
        if (ImGui::Combo("Slider Style", &styleIdx, "Modern (Standard)\0Retro LCD\0")) {
            globalSliderStyle = (styleIdx == 0) ? UISliderStyle::Modern : UISliderStyle::RetroLCD;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        static int theme_preset = 0;
        
        // Ensure initialization on first run to prevent zero-alpha bugs
        if (init_theme) {
             theme_preset = 0;
             currentTheme.litColor = IM_COL32(200, 200, 200, 255);
             currentTheme.offColor = IM_COL32(40, 45, 50, 255);
             currentTheme.bgColor = IM_COL32(20, 20, 20, 255);
             currentTheme.textValColor = IM_COL32(180, 230, 255, 255);
             currentTheme.isRetroGreen = false;
             init_theme = false;
        }

        if (ImGui::Combo("Preset", &theme_preset, "Retro Monochrome\0Classic Green\0Amber\0Cyberpunk Blue\0Custom\0")) {
            switch (theme_preset) {
                case 0: // Mono
                    currentTheme.litColor = IM_COL32(200, 200, 200, 255);
                    currentTheme.offColor = IM_COL32(40, 45, 50, 255);
                    currentTheme.bgColor = IM_COL32(20, 20, 20, 255);
                    currentTheme.textValColor = IM_COL32(180, 230, 255, 255);
                    currentTheme.isRetroGreen = false;
                    break;
                case 1: // Green
                    currentTheme.litColor = IM_COL32(50, 255, 50, 255);
                    currentTheme.offColor = IM_COL32(20, 50, 20, 255);
                    currentTheme.bgColor = IM_COL32(10, 20, 10, 255);
                    currentTheme.textValColor = IM_COL32(100, 255, 100, 255);
                    currentTheme.isRetroGreen = true;
                    break;
                case 2: // Amber
                    currentTheme.litColor = IM_COL32(255, 180, 20, 255);
                    currentTheme.offColor = IM_COL32(60, 40, 10, 255);
                    currentTheme.bgColor = IM_COL32(20, 15, 5, 255);
                    currentTheme.textValColor = IM_COL32(255, 200, 100, 255);
                    currentTheme.isRetroGreen = false;
                    break;
                case 3: // Cyberpunk
                    currentTheme.litColor = IM_COL32(0, 255, 255, 255);
                    currentTheme.offColor = IM_COL32(0, 50, 60, 255);
                    currentTheme.bgColor = IM_COL32(5, 10, 20, 255);
                    currentTheme.textValColor = IM_COL32(0, 255, 255, 255);
                    currentTheme.isRetroGreen = false;
                    break;
                case 4: // Custom
                    // Ensure alpha is full if coming from an uninit state (just safety)
                    if ((currentTheme.litColor & 0xFF000000) == 0) currentTheme.litColor |= 0xFF000000;
                    if ((currentTheme.offColor & 0xFF000000) == 0) currentTheme.offColor |= 0xFF000000;
                    if ((currentTheme.bgColor & 0xFF000000) == 0) currentTheme.bgColor |= 0xFF000000;
                    if ((currentTheme.textValColor & 0xFF000000) == 0) currentTheme.textValColor |= 0xFF000000;
                    break;
            }
        }
        
        // Manual Color Overrides
        ImGui::Text("Custom Colors");
        ImGui::Spacing();

        bool custom_changed = false;
        ImVec4 colLit = ImGui::ColorConvertU32ToFloat4(currentTheme.litColor);
        if (ImGui::ColorEdit3("Lit Color", &colLit.x)) { 
            currentTheme.litColor = ImGui::ColorConvertFloat4ToU32(colLit); 
            theme_preset = 4; // Switch to Custom
        }

        ImVec4 colOff = ImGui::ColorConvertU32ToFloat4(currentTheme.offColor);
        if (ImGui::ColorEdit3("Off Color", &colOff.x)) { 
            currentTheme.offColor = ImGui::ColorConvertFloat4ToU32(colOff); 
            theme_preset = 4;
        }
        
        ImVec4 colBg = ImGui::ColorConvertU32ToFloat4(currentTheme.bgColor);
        if (ImGui::ColorEdit3("Background", &colBg.x)) { 
            currentTheme.bgColor = ImGui::ColorConvertFloat4ToU32(colBg); 
            theme_preset = 4;
        }

        ImVec4 colTxt = ImGui::ColorConvertU32ToFloat4(currentTheme.textValColor);
        if (ImGui::ColorEdit3("Text Color", &colTxt.x)) { 
            currentTheme.textValColor = ImGui::ColorConvertFloat4ToU32(colTxt); 
            theme_preset = 4;
        }

        ImGui::Checkbox("Retro Keyframe Style", &currentTheme.isRetroGreen);
        
        ImGui::Unindent();
    }
}
void SceneUI::drawResolutionPanel(UIContext& ctx)
{
    if (UIWidgets::BeginSection("System & Output", ImVec4(0.4f, 0.8f, 0.6f, 1.0f))) {
        

        UIWidgets::ColoredHeader("Resolution", ImVec4(0.7f, 0.9f, 0.8f, 1.0f));
        
        if (UIWidgets::BeginSection("Display Settings", ImVec4(0.5f, 0.5f, 0.6f, 1.0f))) {
            if (ImGui::Combo("Presets", &preset_index,
                [](void* data, int idx, const char** out_text) {
                    *out_text = ((ResolutionPreset*)data)[idx].name;
                    return true;
                }, presets, IM_ARRAYSIZE(presets)))
            {
                if (preset_index != 0) {
                    new_width = presets[preset_index].w;
                    new_height = presets[preset_index].h;
                    aspect_w = presets[preset_index].bw;
                    aspect_h = presets[preset_index].bh;
                }
            }

            ImGui::Spacing();
            ImGui::PushItemWidth(150);
            ImGui::InputInt("Width", &new_width);
            ImGui::InputInt("Height", &new_height);
            ImGui::PopItemWidth();
            
            ImGui::Spacing();
            ImGui::PushItemWidth(100);
            ImGui::InputInt("Aspect W", &aspect_w);
            ImGui::SameLine();
            ImGui::InputInt("Aspect H", &aspect_h);
            ImGui::PopItemWidth();

            bool resolution_changed =
                (new_width != last_applied_width) ||
                (new_height != last_applied_height) ||
                (aspect_w != last_applied_aspect_w) ||
                (aspect_h != last_applied_aspect_h);

            ImGui::Spacing();
            
            if (UIWidgets::PrimaryButton("Apply Resolution", ImVec2(150, 0), resolution_changed))
            {
                float ar = aspect_h ? float(aspect_w) / aspect_h : 1.0f;
                pending_aspect_ratio = ar;
                pending_width = new_width;
                pending_height = new_height;
                aspect_ratio = ar;
                pending_resolution_change = true;

                last_applied_width = new_width;
                last_applied_height = new_height;
                last_applied_aspect_w = aspect_w;
                last_applied_aspect_h = aspect_h;
            }
            
            UIWidgets::EndSection();
        }

        UIWidgets::EndSection();
    }
}


static void DrawRenderWindowToneMapControls(UIContext& ctx) {
    UIWidgets::ColoredHeader("Post-Processing Controls", ImVec4(1.0f, 0.65f, 0.6f, 1.0f));
    UIWidgets::Divider();

    bool changed = false;

    // -------- Main Parameters --------
    if (UIWidgets::BeginSection("Main Post-Processing", ImVec4(0.8f, 0.6f, 0.5f, 1.0f))) {
        if (UIWidgets::SliderWithHelp("Gamma", &ctx.color_processor.params.global_gamma, 
                                   0.5f, 3.0f, "Controls overall image brightness curve")) changed = true;
        if (UIWidgets::SliderWithHelp("Exposure", &ctx.color_processor.params.global_exposure, 
                                   0.1f, 5.0f, "Adjusts overall brightness level")) changed = true;
        if (UIWidgets::SliderWithHelp("Saturation", &ctx.color_processor.params.saturation, 
                                   0.0f, 2.0f, "Controls color intensity")) changed = true;
        if (UIWidgets::SliderWithHelp("Temperature (K)", &ctx.color_processor.params.color_temperature, 
                                   1000.0f, 10000.0f, "Color temperature in Kelvin", "%.0f")) changed = true;
        UIWidgets::EndSection();
    }

    // -------- Tonemapping Type --------
    if (UIWidgets::BeginSection("Tonemapping Type", ImVec4(0.6f, 0.7f, 0.9f, 1.0f))) {
        const char* tone_names[] = { "AGX", "ACES", "Uncharted", "Filmic", "None" };
        int selected_tone = static_cast<int>(ctx.color_processor.params.tone_mapping_type);
        if (ImGui::Combo("Tonemapping", &selected_tone, tone_names, IM_ARRAYSIZE(tone_names))) {
            ctx.color_processor.params.tone_mapping_type = static_cast<ToneMappingType>(selected_tone);
            changed = true;
        }
        UIWidgets::HelpMarker("AGX: Balanced look | ACES: Cinema standard | Filmic: Classic film");
        UIWidgets::EndSection();
    }

    // -------- Effects --------
    if (UIWidgets::BeginSection("Effects", ImVec4(0.7f, 0.5f, 0.8f, 1.0f))) {
        if (ImGui::Checkbox("Vignette", &ctx.color_processor.params.enable_vignette)) changed = true;
        if (ctx.color_processor.params.enable_vignette) {
            if (UIWidgets::SliderWithHelp("Vignette Strength", &ctx.color_processor.params.vignette_strength, 
                                       0.0f, 2.0f, "Darkening around image edges")) changed = true;
        }
        UIWidgets::EndSection();
    }

    // -------- Actions --------
    UIWidgets::Divider();
    
    // Checkbox controls the persistent flag
    ImGui::Checkbox("Enable Post-Processing", &ctx.render_settings.persistent_tonemap);
    UIWidgets::HelpMarker("Keep post-processing active during rendering/navigation.");

    // If enabled, any parameter change triggers a refresh
    if (ctx.render_settings.persistent_tonemap && changed) {
        ctx.apply_tonemap = true;
    }

    // Force Apply button:
    // 1. Applies effect immediately
    // 2. ENABLES persistence so it doesn't vanish on next frame
    if (UIWidgets::PrimaryButton("Force Apply", ImVec2(120, 0))) {
        ctx.apply_tonemap = true;
        ctx.render_settings.persistent_tonemap = true;
    }
        
    ImGui::SameLine();
    if (UIWidgets::SecondaryButton("Reset", ImVec2(80, 0))) 
        ctx.reset_tonemap = true;
}



// drawWorldContent moved to scene_ui_world.cpp

void SceneUI::drawRenderSettingsPanel(UIContext& ctx, float screen_y)
{
    // Dinamik yükseklik hesabı
    bool bottom_visible = show_animation_panel || show_scene_log;
    float bottom_margin = bottom_visible ? (bottom_panel_height + 24.0f) : 24.0f; // Panel + StatusBar

    float menu_height = 19.0f; 
    float target_height = screen_y - menu_height - bottom_margin;

    // Panel ayarları
    // Lock Height to target_height (MinY = MaxY), allow Width resize (300-800)
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(300, target_height),                 
        ImVec2(800, target_height) 
    );

    // LEFT SIDE DOCKING
    ImGuiIO& io = ImGui::GetIO();
    
    // Position at (0, menu_height) -> TOP LEFT
    ImGui::SetNextWindowPos(ImVec2(0, menu_height), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(side_panel_width, target_height), ImGuiCond_FirstUseEver);

    // Remove TitleBar and Resize for a seamless docked look
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;

    // Add frame styling
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

    if (ImGui::Begin("Properties", nullptr, flags))
    {
        ImDrawList* parent_dl = ImGui::GetWindowDrawList();
        ImVec2 win_pos = ImGui::GetWindowPos();
        ImVec2 win_size = ImGui::GetWindowSize();
        // Update width if user resized
        side_panel_width = ImGui::GetWindowWidth();

        // ─────────────────────────────────────────────────────────────────────────
        // MODERN VERTICAL TAB NAVIGATION
        // ─────────────────────────────────────────────────────────────────────────

        // Sync tab_to_focus with vertical tabs
        if (tab_to_focus == "Scene Edit") { active_properties_tab = 0; tab_to_focus = ""; }
        if (tab_to_focus == "Render")     { active_properties_tab = 1; tab_to_focus = ""; }
        if (tab_to_focus == "Terrain")    { active_properties_tab = 2; tab_to_focus = ""; }
        if (tab_to_focus == "Water")      { active_properties_tab = 3; tab_to_focus = ""; }
        if (tab_to_focus == "Volumetric" || tab_to_focus == "VDB" || tab_to_focus == "Gas") { active_properties_tab = 4; tab_to_focus = ""; }
        if (tab_to_focus == "Force Field"){ active_properties_tab = 5; tab_to_focus = ""; }
        if (tab_to_focus == "World")      { active_properties_tab = 6; tab_to_focus = ""; }
        if (tab_to_focus == "Modifiers")  { active_properties_tab = 7; tab_to_focus = ""; }
        if (tab_to_focus == "System")     { active_properties_tab = 8; tab_to_focus = ""; }

        float sidebar_width = 46.0f;
        
        // --- 1. SIDEBAR (Fixed Width) ---
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 8));
        
        // Sidebar background: slightly darker than main window for contrast
        ImVec4 sidebarBg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
        sidebarBg.x *= 0.85f; sidebarBg.y *= 0.85f; sidebarBg.z *= 0.85f;
        ImGui::PushStyleColor(ImGuiCol_ChildBg, sidebarBg);
        
        ImGui::BeginChild("PropSidebar", ImVec2(sidebar_width, 0), false, ImGuiWindowFlags_NoScrollbar);
        
        // Add a vertical line to separate sidebar - Use Parent DL to avoid clipping
        parent_dl->AddLine(
            ImVec2(win_pos.x + sidebar_width - 1, win_pos.y),
            ImVec2(win_pos.x + sidebar_width - 1, win_pos.y + win_size.y),
            ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.08f)) // Faint border
        );

        auto drawTabButton = [&](int index, UIWidgets::IconType icon, const char* tooltip) {
            bool is_active = (active_properties_tab == index);
            ImGui::PushID(index);
            
            ImVec2 pos = ImGui::GetCursorScreenPos();
            float size = 36.0f; // Slightly smaller buttons
            float margin = (sidebar_width - size) * 0.5f;

            ImGui::SetCursorPosX(margin);
            
            if (is_active) {
                // Connection Bridge: Use Parent DL to bleed across the child border
                parent_dl->AddRectFilled(
                    ImVec2(pos.x - margin, pos.y), 
                    ImVec2(pos.x + sidebar_width + 2, pos.y + size), 
                    ImGui::ColorConvertFloat4ToU32(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg)),
                    0.0f
                );

                // Indicator on the right edge of the sidebar
                parent_dl->AddRectFilled(
                    ImVec2(win_pos.x + sidebar_width - 3, pos.y + 4), 
                    ImVec2(win_pos.x + sidebar_width, pos.y + size - 4), 
                    ImGui::ColorConvertFloat4ToU32(ImVec4(0.05f, 0.75f, 0.65f, 1.0f)),
                    2.0f
                );

                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1,1,1, 0.05f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0)); 
            }

            if (ImGui::Button("##tab", ImVec2(size, size))) {
                active_properties_tab = index;
            }
            
            // Draw Icon
            ImU32 iconCol = is_active ? ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.9f, 0.8f, 1.0f)) : ImGui::ColorConvertFloat4ToU32(ImVec4(0.55f, 0.55f, 0.6f, 1.0f));
            UIWidgets::DrawIcon(icon, ImVec2(pos.x + (size-20)*0.5f, pos.y + (size-20)*0.5f), 20.0f, iconCol, 1.5f);

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(tooltip);
                ImGui::EndTooltip();
            }
            
            ImGui::PopStyleColor(1);
            ImGui::PopID();
        };

        ImGui::Spacing(); // Top spacing
        drawTabButton(0, UIWidgets::IconType::Scene,      "Scene / Hierarchy");
        drawTabButton(1, UIWidgets::IconType::Render,     "Render Settings");
        if (show_terrain_tab)    drawTabButton(2, UIWidgets::IconType::Terrain,    "Terrain Editor");
        if (show_water_tab)      drawTabButton(3, UIWidgets::IconType::Water,      "Water & Rivers");
        if (show_volumetric_tab) drawTabButton(4, UIWidgets::IconType::Volumetric, "Volumetrics");
        if (show_forcefield_tab) drawTabButton(5, UIWidgets::IconType::Force,      "Force Fields");
        if (show_world_tab)      drawTabButton(6, UIWidgets::IconType::World,      "World & Sky");
        if (show_hair_tab)       drawTabButton(8, UIWidgets::IconType::Scene,      "Hair & Fur");
        drawTabButton(7, UIWidgets::IconType::Sculpt, "Modifiers & Sculpt"); 
        if (show_system_tab)     drawTabButton(9, UIWidgets::IconType::System,     "System & UI");
        
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
        
        ImGui::SameLine(0, 0);
        
        // --- 2. CONTENT AREA (Inspector Style) ---
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg)); // Base background
        
        ImGui::BeginChild("PropContentArea", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar);
        

        // ── MAIN CONTENT (Flush Scroll Area) ──
        ImGui::BeginChild("PropScrollArea", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 0)); // Adding safe padding to prevent clipping
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 12));
        
        // Start Global Indent for controls (leaving headers flush)
        ImGui::Indent(8.0f); 
        ImGui::Spacing();
        ImGui::Unindent(8.0f);

        // --- CAPPED ITEM WIDTH ---
        // Prevents sliders/inputs from stretching too far on wide panels, keeping labels legible
        ImGui::PushItemWidth(UIWidgets::GetInspectorItemWidth());

        switch (active_properties_tab) {
            case 0: drawSceneHierarchy(ctx); break;
            case 1: 
                {
                    // ─────────────────────────────────────────────────────────────────────────
                    // ENGINE & BACKEND
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Render Engine", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
                        extern bool g_hasOptix;
                        extern bool g_hasVulkan;
                        
                        // Hardware support validation (essential when loading a project saved with a GPU backend)
                        if (ctx.render_settings.use_optix && !g_hasOptix) {
                            ctx.render_settings.use_optix = false;
                        }
                        if (ctx.render_settings.use_vulkan && !g_hasVulkan) {
                            ctx.render_settings.use_vulkan = false;
                        }

                        int engine_type = 0;
                        if (ctx.render_settings.use_optix) engine_type = 1;
                        if (ctx.render_settings.use_vulkan) engine_type = 2;
                        
                        const char* engines[] = { "CPU (Embree)", "NVIDIA OptiX (CUDA)", "Vulkan (Experimental)" };
                        
                        if (ImGui::BeginCombo("Engine", engines[engine_type])) {
                            for (int i = 0; i < IM_ARRAYSIZE(engines); i++) {
                                bool is_disabled = false;
                                if (i == 1 && !g_hasOptix) is_disabled = true;
                                if (i == 2 && !g_hasVulkan) is_disabled = true;

                                bool is_selected = (engine_type == i);
                                ImGuiSelectableFlags flags = is_disabled ? ImGuiSelectableFlags_Disabled : 0;
                                
                                std::string label = engines[i];
                                if (is_disabled) {
                                    label += " [Not Supported]";
                                }
                                
                                if (ImGui::Selectable(label.c_str(), is_selected, flags)) {
                                    engine_type = i;
                                    ctx.render_settings.use_optix = (engine_type == 1);
                                    ctx.render_settings.use_vulkan = (engine_type == 2);
                                    extern bool g_cpu_sync_pending; 
                                    g_cpu_sync_pending = true;
                                    
                                    ctx.render_settings.backend_changed = true;
                                    ctx.start_render = true;
                                }
                                
                                if (is_selected) {
                                    ImGui::SetItemDefaultFocus();
                                }
                            }
                            ImGui::EndCombo();
                        }
                        
                        if (!g_hasOptix && !g_hasVulkan) {
                            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "[No Compatible GPU Available]");
                        } else if (!g_hasOptix && engine_type == 1) { // Fallback print if somehow matched
                            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "[No Compatible GPU found]");
                        } else if (!g_hasVulkan && engine_type == 2) {
                            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "[Vulkan Not Supported]");
                        }

                        if (!ctx.render_settings.use_optix && !ctx.render_settings.use_vulkan) {
                            const char* bvh_items[] = { "Custom RayTrophi BVH", "Intel Embree (Recommended)" };
                            int current_bvh = ctx.render_settings.UI_use_embree ? 1 : 0;
                            if (ImGui::Combo("CPU BVH Type", &current_bvh, bvh_items, 2)) {
                                ctx.render_settings.UI_use_embree = (current_bvh == 1);
                                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                                ctx.start_render = true;
                            }
                        }
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // SAMPLING 
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Sampling", ImVec4(0.5f, 0.9f, 0.6f, 1.0f))) {
                        UIWidgets::ColoredHeader("Viewport", ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                        ImGui::Checkbox("Use Adaptive Sampling##view", &ctx.render_settings.use_adaptive_sampling);
                        if (ctx.render_settings.use_adaptive_sampling) {
                            ImGui::DragFloat("Noise Threshold", &ctx.render_settings.variance_threshold, 0.001f, 0.001f, 0.8f, "%.3f");
                            ImGui::DragInt("Min Samples##view", &ctx.render_settings.min_samples, 1, 1, 512);
                        }
                        ImGui::DragInt("Max Samples##view", &ctx.render_settings.max_samples, 1, 1, 10000);
                        
                        UIWidgets::Divider();
                        UIWidgets::ColoredHeader("Final Render (F12)", ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                        ImGui::DragInt("Samples##final", &ctx.render_settings.final_render_samples, 1, 1, 100000);
                        ImGui::Checkbox("Apply Denoiser##final", &ctx.render_settings.render_use_denoiser);
                        if (ctx.render_settings.render_use_denoiser) {
                            const char* denoiser_mode_items[] = {
                                "Fast: beauty only",
                                "Quality: beauty + albedo + normal"
                            };
                            int denoiser_mode = static_cast<int>(ctx.render_settings.denoiser_mode);
                            if (ImGui::Combo("Denoiser Mode##final", &denoiser_mode, denoiser_mode_items,
                                IM_ARRAYSIZE(denoiser_mode_items))) {
                                ctx.render_settings.denoiser_mode = static_cast<DenoiserMode>(denoiser_mode);
                            }
                        }
                        
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // LIGHT PATHS (Bounces)
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Light Paths", ImVec4(1.0f, 0.8f, 0.3f, 1.0f))) {
                        ImGui::DragInt("Total Bounces", &ctx.render_settings.max_bounces, 1, 0, 64);
                        
                        UIWidgets::HelpMarker("Higher bounces increase realism for glass and interiors but slow down rendering.");
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // DENOISING
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Denoising", ImVec4(0.8f, 0.5f, 1.0f, 1.0f))) {
                        ImGui::Checkbox("Enable Viewport Denoising", &ctx.render_settings.use_denoiser);
                        if (ctx.render_settings.use_denoiser) {
                            const char* denoiser_mode_items[] = {
                                "Fast: beauty only",
                                "Quality: beauty + albedo + normal"
                            };
                            int denoiser_mode = static_cast<int>(ctx.render_settings.denoiser_mode);
                            if (ImGui::Combo("Denoiser Mode", &denoiser_mode, denoiser_mode_items, IM_ARRAYSIZE(denoiser_mode_items))) {
                                ctx.render_settings.denoiser_mode = static_cast<DenoiserMode>(denoiser_mode);
                            }
                            ImGui::SliderFloat("Blend Factor", &ctx.render_settings.denoiser_blend_factor, 0.0f, 1.0f);
                        }
                        UIWidgets::EndSection();
                    }
 Broadway:

                    // ─────────────────────────────────────────────────────────────────────────
                    // FORMAT & OUTPUT
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Resolution & Output", ImVec4(0.9f, 0.4f, 0.5f, 1.0f))) {
                        // Presets
                        if (ImGui::Combo("Resolution Preset", &preset_index,
                            [](void* data, int idx, const char** out_text) {
                                *out_text = ((ResolutionPreset*)data)[idx].name;
                                return true;
                            }, presets, IM_ARRAYSIZE(presets)))
                        {
                            if (preset_index != 0) {
                                new_width = presets[preset_index].w;
                                new_height = presets[preset_index].h;
                                aspect_w = presets[preset_index].bw;
                                aspect_h = presets[preset_index].bh;
                            }
                        }

                        ImGui::DragInt("Width", &new_width, 1, 1, 8192);
                        ImGui::DragInt("Height", &new_height, 1, 1, 8192);
                        
                        ImGui::PushItemWidth(80);
                        ImGui::DragInt("Aspect W", &aspect_w, 1, 1, 100); ImGui::SameLine();
                        ImGui::DragInt("Aspect H", &aspect_h, 1, 1, 100);
                        ImGui::PopItemWidth();

                        bool resolution_changed = (new_width != last_applied_width) || (new_height != last_applied_height);
                        
                        if (UIWidgets::PrimaryButton("Apply Settings", ImVec2(UIWidgets::GetInspectorActionWidth(), 30), resolution_changed)) {
                            float ar = aspect_h ? float(aspect_w) / aspect_h : 1.0f;
                            pending_aspect_ratio = ar;
                            pending_width = new_width;
                            pending_height = new_height;
                            aspect_ratio = ar;
                            pending_resolution_change = true;
                            last_applied_width = new_width; last_applied_height = new_height;
                        }

                        UIWidgets::Divider();
                        if (UIWidgets::SecondaryButton("Open Dedicated Render Window", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
                            extern bool show_render_window;
                            show_render_window = true;
                        }
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // POST-PROCESSING & TONEMAPPING
                    // ─────────────────────────────────────────────────────────────────────────
                    DrawRenderWindowToneMapControls(ctx);

                    // ─────────────────────────────────────────────────────────────────────────
                    // ANIMATION RENDER (Sequence Export)
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Animation Render", ImVec4(1.0f, 0.4f, 0.7f, 1.0f))) {
                        
                        // ═══════════════════════════════════════════════════════════════════
                        // RENDERING IN PROGRESS - Show Status Panel
                        // ═══════════════════════════════════════════════════════════════════
                        if (rendering_in_progress && ctx.is_animation_mode) {
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.15f, 0.2f, 1.0f));
                            ImGui::BeginChild("AnimRenderStatus", ImVec2(0, 120), true);
                            
                            // Current Frame Info
                            int cur = ctx.render_settings.animation_current_frame;
                            int start = ctx.render_settings.animation_start_frame;
                            int end = ctx.render_settings.animation_end_frame;
                            int total = end - start + 1;
                            int done = cur - start;
                            float progress = (total > 0) ? (float)done / (float)total : 0.0f;
                            
                            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "RENDERING ANIMATION...");
                            ImGui::Spacing();
                            
                            // Big progress bar
                            char prog_text[64];
                            snprintf(prog_text, sizeof(prog_text), "Frame %d / %d  (%.0f%%)", cur, end, progress * 100.0f);
                            ImGui::ProgressBar(progress, ImVec2(-1, 24), prog_text);
                            
                            ImGui::Spacing();
                            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Samples: %d | FPS: %d", 
                                ctx.render_settings.animation_samples_per_frame,
                                ctx.render_settings.animation_fps);
                            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Output: %s", 
                                ctx.render_settings.animation_output_folder.c_str());
                            
                            ImGui::Spacing();
                            if (UIWidgets::DangerButton("STOP RENDERING", ImVec2(-1, 28))) {
                                rendering_stopped_cpu = true;
                                rendering_stopped_gpu = true;
                                SCENE_LOG_WARN("Animation render stop requested by user.");
                            }
                            
                            ImGui::EndChild();
                            ImGui::PopStyleColor();
                        }
                        else {
                            // ═══════════════════════════════════════════════════════════════════
                            // NORMAL MODE - Setup Panel
                            // ═══════════════════════════════════════════════════════════════════
                            UIWidgets::ColoredHeader("Frame Range & Speed", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
                            ImGui::PushItemWidth(80);
                            ImGui::DragInt("Start", &ctx.render_settings.animation_start_frame, 1, 0, ctx.render_settings.animation_end_frame);
                            ImGui::SameLine();
                            ImGui::DragInt("End", &ctx.render_settings.animation_end_frame, 1, ctx.render_settings.animation_start_frame, 10000);
                            ImGui::SameLine();
                            ImGui::DragInt("FPS", &ctx.render_settings.animation_fps, 1, 1, 120);
                            ImGui::PopItemWidth();
                            
                            // Auto-detect button
                            if (!ctx.scene.animationDataList.empty() && ctx.scene.animationDataList[0]) {
                                ImGui::SameLine();
                                if (ImGui::SmallButton("Auto")) {
                                    ctx.render_settings.animation_start_frame = ctx.scene.animationDataList[0]->startFrame;
                                    ctx.render_settings.animation_end_frame = ctx.scene.animationDataList[0]->endFrame;
                                    SCENE_LOG_INFO("Frame range auto-set from animation file.");
                                }
                                if (ImGui::IsItemHovered()) {
                                    ImGui::SetTooltip("Auto-detect frame range from loaded animation");
                                }
                            }
                            
                            UIWidgets::Divider();
                            UIWidgets::ColoredHeader("Quality", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
                            ImGui::DragInt("Samples Per Frame", &ctx.render_settings.animation_samples_per_frame, 1, 1, 10000);
                            
                            // Quick presets
                            ImGui::SameLine();
                            if (ImGui::SmallButton("Draft")) ctx.render_settings.animation_samples_per_frame = 16;
                            ImGui::SameLine();
                            if (ImGui::SmallButton("Medium")) ctx.render_settings.animation_samples_per_frame = 64;
                            ImGui::SameLine();
                            if (ImGui::SmallButton("High")) ctx.render_settings.animation_samples_per_frame = 256;
                            
                            UIWidgets::Divider();
                            UIWidgets::ColoredHeader("Output (PNG Sequence)", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
                            
                            // Output Path Display & Browse
                            ImGui::PushItemWidth(-50);
                            char folder_buf[512];
                            strncpy(folder_buf, ctx.render_settings.animation_output_folder.c_str(), 511);
                            if (ImGui::InputText("##outdir", folder_buf, 512)) {
                                ctx.render_settings.animation_output_folder = folder_buf;
                            }
                            ImGui::PopItemWidth();
                            ImGui::SameLine();
                            if (ImGui::Button("...##browse")) {
                                std::string path = selectFolderDialogW(L"Select Animation Output Folder");
                                if (!path.empty()) ctx.render_settings.animation_output_folder = path;
                            }
                            
                            ImGui::Spacing();
                            
                            // Summary Info Box
                            int total_frames = ctx.render_settings.animation_end_frame - ctx.render_settings.animation_start_frame + 1;
                            int samples = ctx.render_settings.animation_samples_per_frame;
                            float est_time_per_frame = (samples / 64.0f) * 2.0f; // Rough estimate: 2 sec per 64 samples
                            float est_total_minutes = (est_time_per_frame * total_frames) / 60.0f;
                            
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.1f, 0.15f, 1.0f));
                            ImGui::BeginChild("RenderSummary", ImVec2(0, 50), true);
                            ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Summary:");
                            ImGui::SameLine();
                            ImGui::Text("%d frames x %d samples = ~%.1f min (estimated)", total_frames, samples, est_total_minutes);
                            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Resolution: %dx%d", 
                                ctx.render_settings.final_render_width, 
                                ctx.render_settings.final_render_height);
                            ImGui::EndChild();
                            ImGui::PopStyleColor();
                            
                            ImGui::Spacing();
                            
                            bool can_render = !ctx.render_settings.animation_output_folder.empty();
                            bool valid_range = (ctx.render_settings.animation_end_frame >= ctx.render_settings.animation_start_frame);
                            
                            if (!can_render) {
                                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1.0f), "! Set output folder");
                            }
                            if (!valid_range) {
                                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1.0f), "! Invalid frame range");
                            }
                            
                            if (UIWidgets::PrimaryButton("RENDER ANIMATION SEQUENCE", ImVec2(UIWidgets::GetInspectorActionWidth(), 36), can_render && valid_range)) {
                                ctx.render_settings.start_animation_render = true;
                                ctx.render_settings.animation_total_frames = total_frames;
                                SCENE_LOG_INFO("Animation render triggered: " + std::to_string(total_frames) + " frames @ " + std::to_string(samples) + " samples");
                            }
                        }

                        UIWidgets::EndSection();
                    }
                }
                break;
            case 2: if (show_terrain_tab) drawTerrainPanel(ctx); break;
            case 3: if (show_water_tab) { drawWaterPanel(ctx); ImGui::Separator(); drawRiverPanel(ctx); } break;
            case 4: if (show_volumetric_tab) drawVolumetricPanel(ctx); break;
            case 5: if (show_forcefield_tab) ForceFieldUI::drawForceFieldPanel(ctx, ctx.scene); break;
            case 6: if (show_world_tab) drawWorldContent(ctx); break;
            case 7: drawModifiersPanel(ctx); break;
            case 9: drawThemeSelector(); drawResolutionPanel(ctx); break;
            case 8: if (show_hair_tab) {
                // Get selected mesh triangles for hair generation target
                static std::vector<std::shared_ptr<Triangle>> selectedMeshTriangles;
                static std::string lastSelectedMeshName;
                const std::vector<std::shared_ptr<Triangle>>* selectedTris = nullptr;
                
                // Check if we have a selected object
                bool hasValidSelection = (ctx.selection.selected.type == SelectableType::Object && 
                                         ctx.selection.selected.object != nullptr);
                
                if (hasValidSelection) {
                    // Get the nodeName of selected object
                    std::string selectedNodeName = ctx.selection.selected.object->getNodeName();
                    
                    // Only rebuild triangle list if selection changed
                    if (selectedNodeName != lastSelectedMeshName) {
                        lastSelectedMeshName = selectedNodeName;
                        selectedMeshTriangles.clear();
                        
                        for (const auto& obj : ctx.scene.world.objects) {
                            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                            if (tri && tri->getNodeName() == selectedNodeName) {
                                selectedMeshTriangles.push_back(tri);
                            }
                        }
                    }
                    
                    if (!selectedMeshTriangles.empty()) {
                        selectedTris = &selectedMeshTriangles;
                    }
                } else {
                    // Selection cleared - clear cached data
                    if (!lastSelectedMeshName.empty()) {
                        lastSelectedMeshName.clear();
                        selectedMeshTriangles.clear();
                    }
                    selectedTris = nullptr;
                }
                
                if (!hairUI.onOpenFileDialog) {
                     hairUI.onOpenFileDialog = [](const wchar_t* filter) {
                         return SceneUI::openFileDialogW(filter);
                     };
                }
                hairUI.render(ctx.renderer.getHairSystem(), selectedTris, &ctx.renderer, [&ctx, this]() {
                    // [FIX] Ensure CPU BVH is up to date before "Generate Full" so hair sits on the *current* mesh surface,
                    // not the old one (if gizmo was just used).
                    this->ensureCPUSyncForPicking(ctx);
                });
                
                // [NEW] Sync hide children preference for performance during grooming
                ctx.renderer.hideInterpolatedHair = hairUI.shouldHideChildren();
                
                // Track if material changed for render reset
                static Hair::HairMaterialParams lastMaterial;
                static bool firstFrame = true;
                Hair::HairMaterialParams currentMaterial = hairUI.getMaterial();
                
                bool materialChanged = firstFrame ||
                    (lastMaterial.colorMode != currentMaterial.colorMode) ||
                    (lastMaterial.melanin != currentMaterial.melanin) ||
                    (lastMaterial.melaninRedness != currentMaterial.melaninRedness) ||
                    (lastMaterial.roughness != currentMaterial.roughness) ||
                    (lastMaterial.radialRoughness != currentMaterial.radialRoughness) ||
                    (lastMaterial.ior != currentMaterial.ior) ||
                    (lastMaterial.cuticleAngle != currentMaterial.cuticleAngle) ||
                    (lastMaterial.coat != currentMaterial.coat) ||
                    (std::abs(lastMaterial.color.x - currentMaterial.color.x) > 0.001f) ||
                    (std::abs(lastMaterial.color.y - currentMaterial.color.y) > 0.001f) ||
                    (std::abs(lastMaterial.color.z - currentMaterial.color.z) > 0.001f) ||
                    (std::abs(lastMaterial.coatTint.x - currentMaterial.coatTint.x) > 0.001f) ||
                    (std::abs(lastMaterial.coatTint.y - currentMaterial.coatTint.y) > 0.001f) ||
                    (std::abs(lastMaterial.coatTint.z - currentMaterial.coatTint.z) > 0.001f);
                
                if (materialChanged) {
                    lastMaterial = currentMaterial;
                    firstFrame = false;
                }
                
                // [FIXED] Removed global ctx.renderer.setHairMaterial(currentMaterial) override
                // Now each hair groom uses its own material from HairSystem during intersect.

                
                // Reset render accumulation if material changed
                if (materialChanged) {
                    ctx.renderer.setHairMaterial(currentMaterial); // [UPDATED] Keep GPU in sync
                    ctx.renderer.resetCPUAccumulation();
                    
                    // Sync to GPU immediately for live feedback
                    if (ctx.backend_ptr) {
                        ctx.renderer.updateBackendMaterials(ctx.scene);
                        ctx.start_render = true; // [NEW] Trigger render pass immediately
                    }
                }
            } break;
        }
        ImGui::PopItemWidth();

        // Safety: Disable brushes if tab changed
        if (active_properties_tab != 2) terrain_brush.enabled = false;
        
        ImGui::PopStyleVar(2);  // WindowPadding, ItemSpacing
        ImGui::EndChild();      // End PropScrollArea
        ImGui::EndChild();      // End PropContentArea
        ImGui::PopStyleColor(); // ChildBg
        ImGui::PopStyleVar();   // ChildRounding
    }
    ImGui::End();
    ImGui::PopStyleColor(); // Border
    ImGui::PopStyleVar();   // BorderSize
}

// Main Menu Bar implementation moved to separate file: scene_ui_menu.hpp check end of file

#include "scene_ui_menu.hpp"


void SceneUI::draw(UIContext& ctx)
{
    // Apply project-scoped UI state after load finalization on the main thread.
    if (pending_project_ui_restore) {
        pending_project_ui_restore = false;
        if (!g_project.ui_layout_data.empty()) {
            // Disable auto-save to ini momentarily to avoid conflicts
            ImGui::GetIO().IniFilename = nullptr; 
            deserialize(g_project.ui_layout_data);
        }
    }

    // Texture Safety Cleanup
    manageTextureGraveyard();

    // Export Popup Logic
    if (SceneExporter::getInstance().drawExportPopup(ctx.scene)) {
         std::wstring filter = SceneExporter::getInstance().settings.binary_mode ? L"GLTF Binary (.glb)\0*.glb\0" : L"GLTF Text (.gltf)\0*.gltf\0";
         std::wstring defExt = SceneExporter::getInstance().settings.binary_mode ? L"glb" : L"gltf";
         
         std::string filepath = saveFileDialogW(filter.c_str(), defExt.c_str());
         
         if (!filepath.empty()) {
             // Enforce extension
             std::string ext = SceneExporter::getInstance().settings.binary_mode ? ".glb" : ".gltf";
             if (!std::string(filepath).ends_with(ext)) {
                 filepath += ext;
             }

             rendering_stopped_cpu = true;
             rendering_stopped_gpu = true;
             
             // Show "Exporting..." modal or message? 
             addViewportMessage("Exporting Scene... Check Console...", 10.0f, ImVec4(1, 1, 0, 1));
             SCENE_LOG_INFO("[Export] Thread starting for: " + filepath);
             
             // Capture SceneData via pointer
             SceneData* pScene = &ctx.scene;
             
             // Capture Selection (Convert SelectableItem to shared_ptr<Hittable>)
             std::vector<std::shared_ptr<Hittable>> selected_hittables;
             if (SceneExporter::getInstance().settings.export_selected_only) {
                 for (const auto& item : ctx.selection.multi_selection) {
                     if (item.type == SelectableType::Object && item.object) {
                         selected_hittables.push_back(item.object);
                     }
                 }
             }

             std::thread export_thread([filepath, pScene, selected_hittables]() {
                 SCENE_LOG_INFO("[Export] Thread running...");
                 ExportSettings settings = SceneExporter::getInstance().settings;
                 
                 try {
                     bool success = SceneExporter::getInstance().exportScene(filepath, *pScene, settings, selected_hittables);
                     if (success) {
                         SCENE_LOG_INFO("[Export] SUCCESS: " + filepath);
                     } else {
                         SCENE_LOG_ERROR("[Export] FAILED (Check logs)");
                     }
                 } catch (const std::exception& e) {
                     SCENE_LOG_ERROR("[Export] EXCEPTION: " + std::string(e.what()));
                 } catch (...) {
                     SCENE_LOG_ERROR("[Export] UNKNOWN EXCEPTION");
                 }

                 rendering_stopped_cpu = false;
                 rendering_stopped_gpu = false;
             });
             export_thread.detach();
         }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CENTRALIZED SCENE SYNC - Ensure selection cache is consistent
    // ═══════════════════════════════════════════════════════════════════════════
    if (ctx.scene.world.objects.size() != last_scene_obj_count) {
        if (last_scene_obj_count != 0) {
            SCENE_LOG_INFO("Scene changed (Count: " + std::to_string(last_scene_obj_count) + 
                           " -> " + std::to_string(ctx.scene.world.objects.size()) + "). Invalidating cache.");
        }
        mesh_cache_valid = false;
        last_scene_obj_count = ctx.scene.world.objects.size();
    }

    world_params_changed_this_frame = false;
    ImGuiIO& io = ImGui::GetIO();
    float screen_x = io.DisplaySize.x;
    float screen_y = io.DisplaySize.y;

    drawMainMenuBar(ctx);
    handleEditorShortcuts(ctx);

    float left_offset = 0.0f;
    drawPanels(ctx);
    left_offset = showSidePanel ? side_panel_width : 0.0f;

    float vp_width = ImGui::GetIO().DisplaySize.x;
    float vp_height = ImGui::GetIO().DisplaySize.y;
    drawStatusAndBottom(ctx, vp_width, vp_height, left_offset);

    bool gizmo_hit = drawOverlays(ctx);
    
    // --- ANIMATION UPDATE ---
    processAnimations(ctx);

    // --- HAIR FORCE FIELD UPDATE (Global - runs regardless of panel focus) ---
    {
        static float lastHairForceTime = -999.0f;
        static bool wasHairPlaying = false;
        
        bool isPlaying = timeline.isPlaying();
        float currentTime = ctx.scene.timeline.current_frame / 24.0f; // Assume 24 FPS
        bool timeChanged = (currentTime != lastHairForceTime);
        
        // Only update if: timeline is playing OR time changed (scrubbing)
        if (ctx.scene.force_field_manager.getActiveCount() > 0 && 
            ctx.renderer.getHairSystem().getGroomNames().size() > 0 &&
            (isPlaying || timeChanged)) {
            
            for (const auto& groomName : ctx.renderer.getHairSystem().getGroomNames()) {
                ctx.renderer.getHairSystem().restyleGroom(groomName, &ctx.scene.force_field_manager, currentTime);
            }
            lastHairForceTime = currentTime;
            
            // Rebuild hair BVH and upload to GPU
            ctx.renderer.getHairSystem().buildBVH(!ctx.renderer.hideInterpolatedHair);
            ctx.renderer.uploadHairToGPU();
            ctx.renderer.resetCPUAccumulation();
        }
        
        // If we just stopped playing, mark for one final update
        if (wasHairPlaying && !isPlaying && ctx.renderer.getHairSystem().getGroomNames().size() > 0) {
            ctx.renderer.getHairSystem().buildBVH(!ctx.renderer.hideInterpolatedHair);
            ctx.renderer.uploadHairToGPU();
        }
        wasHairPlaying = isPlaying;
    }

    drawSelectionGizmos(ctx);
    drawCameraGizmos(ctx);  // Draw camera frustum icons
    drawRiverGizmos(ctx, gizmo_hit);  // Draw river spline control points
    drawViewportControls(ctx);  // Blender-style viewport overlay
    
    // --- HAIR TRANSFORM SYNC (Global) ---
    // Skinned groom updates are handled in Renderer::updateAnimationWithGraph (after skinning).
    // Here we only handle:
    //   1. UI parameter changes (hairUI.isDirty)
    //   2. Rigid-body transform following (non-skinned grooms moved via gizmo)
    // We do NOT call updateAllTransforms every frame to avoid setting m_bvhDirty
    // and causing resetCPUAccumulation on every frame.
    bool needsUpdate = hairUI.isDirty();
    
    // Check if any non-skinned groom's mesh transform changed
    // (This covers gizmo drag, etc. - happens rarely, not every frame)
    if (!needsUpdate && ctx.renderer.getHairSystem().getTotalStrandCount() > 0) {
        ctx.renderer.getHairSystem().updateAllTransforms(ctx.scene.world.objects, ctx.renderer.finalBoneMatrices);
        needsUpdate = ctx.renderer.getHairSystem().isBVHDirty();
    }
    
    if (needsUpdate) {
        ctx.renderer.getHairSystem().buildBVH(!ctx.renderer.hideInterpolatedHair);
        ctx.renderer.uploadHairToGPU();
        ctx.renderer.resetCPUAccumulation();
        ctx.start_render = true;
        hairUI.clearDirty();
    }
    
    // --- BACKGROUND SAVE STATUS POLL ---
    static int last_save_state = 0;
    int save_state = bg_save_state.load();
    
    if (save_state != last_save_state) {
        if (save_state == 1) { // Saving...
            addViewportMessage("Saving...", 300.0f, ImVec4(1.0f, 0.9f, 0.2f, 1.0f));
        }
        else if (save_state == 2) { // Done
            clearViewportMessages();
            addViewportMessage("Project saved", 2.0f, ImVec4(0.2f, 1.0f, 0.4f, 1.0f));
            bg_save_state = 0; // Reset
        }
        else if (save_state == 3) { // Error
            clearViewportMessages();
            addViewportMessage("Save failed", 4.0f, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
            bg_save_state = 0; // Reset
        }
        last_save_state = save_state;
    }

    drawViewportMessages(ctx, left_offset); // Messages/HUD (e.g. Async Rebuild)

    // Hide HUD overlays if exit confirmation is open
    // Otherwise draw them (they use ForegroundDrawList so they appear on top)
    // Note: They might overlay panels like Graph, but visibility is priority.
    if (!show_exit_confirmation) {
        drawFocusIndicator(ctx);
        drawZoomRing(ctx);
        drawExposureInfo(ctx);  // Now includes lens info below the triangle
    }
    
    // Scatter Brush System
    handleScatterBrush(ctx);   // Handle brush painting input
    drawBrushPreview(ctx);     // Draw brush circle preview
    
    // Terrain Sculpting
    handleTerrainBrush(ctx);
    handleTerrainFoliageBrush(ctx);  // Foliage painting brush
    
    // Hair Brush System
    handleHairBrush(ctx);      // Hair paint brush input + preview

    handleSceneInteraction(ctx, gizmo_hit);
    processDeferredSceneUpdates(ctx);
    
     if (WaterManager::getInstance().update(ImGui::GetIO().DeltaTime)) {
         if (ctx.backend_ptr) {
             ctx.renderer.updateBackendMaterials(ctx.scene);
             // Logic for resetAccumulation should ideally be in backend or handled via renderer
             // For now, if we still need to talk to optix wrapper specifically, we go through backend
             // if (auto optix = dynamic_cast<Backend::OptixBackend*>(ctx.backend_ptr)) optix->getOptixWrapper()->resetAccumulation();
             ctx.backend_ptr->resetAccumulation();
         }
         // Also reset CPU accumulation if needed, though water FFT is mostly for GPU?
         // If CPU supports it (sampleOceanHeight), maybe reset CPU too.
         ctx.renderer.resetCPUAccumulation();
    }

    drawAuxWindows(ctx);
    
    // Global Sun Sync (Light -> Nishita)
    processSunSync(ctx);
    drawRenderWindow(ctx);
    drawExitConfirmation(ctx);
    
    // NOTE: Animation Graph Editor is now strictly in the bottom panel
        }
    
void SceneUI::handleEditorShortcuts(UIContext& ctx)
{
    ImGuiIO& io = ImGui::GetIO();
    const bool block_history_actions =
        g_scene_loading_in_progress.load() ||
        scene_loading.load() ||
        rendering_in_progress.load() ||
        ctx.render_settings.backend_changed;

    if (!io.WantCaptureKeyboard && ctx.selection.hasSelection()) {
        handleDeleteShortcut(ctx);
    }

    // F12 Render
    if (ImGui::IsKeyPressed(ImGuiKey_F12)) {
        extern bool show_render_window;
        show_render_window = !show_render_window;
        if (show_render_window) ctx.start_render = true;
    }

    // N key - Toggle Properties Panel (Blender-style sidebar toggle)
    // Use WantTextInput instead of WantCaptureKeyboard so it works even when panel has focus
    // Only block when actively typing in a text field
    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_N) && !io.KeyCtrl && !io.KeyShift && !io.KeyAlt) {
        showSidePanel = !showSidePanel;
        SCENE_LOG_INFO(showSidePanel ? "Properties panel shown (N)" : "Properties panel hidden (N)");
    }

    // Undo / Redo
    if (ImGui::IsKeyPressed(ImGuiKey_Z) && io.KeyCtrl && !io.KeyShift) {
        if (!block_history_actions && history.canUndo()) {
            history.undo(ctx);
            rebuildMeshCache(ctx.scene.world.objects);
            ctx.selection.updatePositionFromSelection();
            ctx.selection.selected.has_cached_aabb = false;
        }
    }

    if ((ImGui::IsKeyPressed(ImGuiKey_Y) && io.KeyCtrl) ||
        (ImGui::IsKeyPressed(ImGuiKey_Z) && io.KeyCtrl && io.KeyShift)) {
        if (!block_history_actions && history.canRedo()) {
            history.redo(ctx);
            rebuildMeshCache(ctx.scene.world.objects);
            ctx.selection.updatePositionFromSelection();
            ctx.selection.selected.has_cached_aabb = false;
        }
    }
}
void SceneUI::drawPanels(UIContext& ctx)
{
    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg,
        ImVec4(0.1f, 0.1f, 0.13f, panel_alpha));

    if (showSidePanel) {
        drawRenderSettingsPanel(ctx, screen_y);
    }

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}
void SceneUI::drawStatusAndBottom(UIContext& ctx,
    float screen_x,
    float screen_y,
    float left_offset)
{
    // ---------------- STATUS BAR ----------------
    float status_bar_height = 24.0f;

    // EXPORT PROGRESS HUD
    if (SceneExporter::getInstance().is_exporting) {
        ImGui::OpenPopup("Exporting...");
    }

    if (ImGui::BeginPopupModal("Exporting...", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar)) {
        if (!SceneExporter::getInstance().is_exporting) {
            ImGui::CloseCurrentPopup();
        } else {
            ImGui::Text("Exporting Scene...");
            ImGui::Separator();
            
            // Spinner or Bar
            ImGui::Text("Please wait, data is being processed.");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", SceneExporter::getInstance().current_export_status.c_str());
            
            ImGui::Separator();
        }
        ImGui::EndPopup();
    }

    ImGui::SetNextWindowPos(ImVec2(0, screen_y - status_bar_height), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_x, status_bar_height), ImGuiCond_Always);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 2));
    // Use theme color with alpha
    ImGui::SetNextWindowBgAlpha(panel_alpha);

    if (ImGui::Begin("StatusBar", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoBringToFrontOnFocus))
    {
        ImGui::SetCursorPosX(8); // Small left margin
        
        if (UIWidgets::HorizontalTab("Timeline", UIWidgets::IconType::Timeline, show_animation_panel))
        {
            show_animation_panel = !show_animation_panel;
            if (show_animation_panel) {
                show_scene_log = false;
                focus_bottom_panel_next_frame = true;
            }
        }

        if (UIWidgets::HorizontalTab("Console", UIWidgets::IconType::Console, show_scene_log))
        {
            show_scene_log = !show_scene_log;
            if (show_scene_log) { 
                show_animation_panel = false; 
                show_terrain_graph = false; 
                focus_bottom_panel_next_frame = true;
            }
        }
        
        if (UIWidgets::HorizontalTab("Graph", UIWidgets::IconType::Graph, show_terrain_graph))
        {
            show_terrain_graph = !show_terrain_graph;
            if (show_terrain_graph) {
                show_scene_log = false;
                show_animation_panel = false;
                show_anim_graph = false;
                focus_bottom_panel_next_frame = true;
            }
        }
        
        if (UIWidgets::HorizontalTab("AnimGraph", UIWidgets::IconType::AnimGraph, show_anim_graph))
        {
            show_anim_graph = !show_anim_graph;
            if (show_anim_graph) {
                show_scene_log = false;
                show_animation_panel = false;
                show_terrain_graph = false;
                show_asset_browser = false;
                focus_bottom_panel_next_frame = true;
            }
        }

        if (UIWidgets::HorizontalTab("Assets", UIWidgets::IconType::Scene, show_asset_browser))
        {
            show_asset_browser = !show_asset_browser;
            if (show_asset_browser) {
                show_scene_log = false;
                show_animation_panel = false;
                show_terrain_graph = false;
                show_anim_graph = false;
                focus_bottom_panel_next_frame = true;
            }
        }

        ImGui::SameLine();
        

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        if (ctx.scene.initialized) {
            auto drawPill = [&](const char* label, const ImVec4& color, bool highlight = false) {
                ImVec2 p = ImGui::GetCursorScreenPos();
                ImVec2 textSize = ImGui::CalcTextSize(label);
                ImVec2 pillSize = ImVec2(textSize.x + 16, 18);
                ImDrawList* dlist = ImGui::GetWindowDrawList();
                
                // Pill Background
                dlist->AddRectFilled(p, ImVec2(p.x + pillSize.x, p.y + pillSize.y), highlight ? IM_COL32(26, 230, 204, 40) : IM_COL32(255, 255, 255, 15), 9.0f);
                if (highlight) dlist->AddRect(p, ImVec2(p.x + pillSize.x, p.y + pillSize.y), IM_COL32(26, 230, 204, 80), 9.0f);

                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 8);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
                ImGui::TextColored(highlight ? ImVec4(0.1f, 0.9f, 0.8f, 1.0f) : ImVec4(0.8f, 0.8f, 0.85f, 1.0f), "%s", label);
                
                ImGui::SetCursorScreenPos(ImVec2(p.x + pillSize.x + 6, p.y));
            };

            // Scene Info Pills
            int obj_count = (int)mesh_cache.size();
            std::string obj_str = "Scene: " + std::to_string(obj_count) + " Objects";
            drawPill(obj_str.c_str(), ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            
            std::string light_str = std::to_string(ctx.scene.lights.size()) + " Lights";
            drawPill(light_str.c_str(), ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            
            // Selection Pill
            if (ctx.selection.hasSelection()) {
                std::string sel_str = "Selected: " + ctx.selection.selected.name;
                drawPill(sel_str.c_str(), ImVec4(0.4f, 0.8f, 1.0f, 1.0f), true);
            }
        }
        else {
            ImGui::Text("Ready");
        }

        // ═══════════════════════════════════════════════════════════════════════════════
        // PROFESSIONAL RENDER STATUS INDICATOR (Right side of status bar)
        // ═══════════════════════════════════════════════════════════════════════════════
        if (rendering_in_progress) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            
            // Calculate right-aligned position
            float status_width = 320.0f;  // Wider for lock indicator
            float start_x = screen_x - status_width - 12.0f;
            ImVec2 bar_pos = ImVec2(start_x, ImGui::GetCursorScreenPos().y + 2);
            
            // Show LOCKED indicator when animation render is active
            if (ctx.render_settings.animation_render_locked) {
                // Check if paused
                bool is_paused = rendering_paused.load();
                
                if (is_paused) {
                    // PAUSED pill (yellow/orange)
                    ImVec2 pause_pos = ImVec2(start_x - 70, bar_pos.y - 1);
                    dl->AddRectFilled(pause_pos, ImVec2(pause_pos.x + 65, pause_pos.y + 18), IM_COL32(200, 150, 30, 255), 4.0f);
                    dl->AddText(ImVec2(pause_pos.x + 8, pause_pos.y + 2), IM_COL32(255, 255, 255, 255), "PAUSED");
                    
                    // P hint
                    ImVec2 p_pos = ImVec2(pause_pos.x - 85, pause_pos.y);
                    dl->AddRectFilled(p_pos, ImVec2(p_pos.x + 80, p_pos.y + 18), IM_COL32(60, 80, 60, 255), 4.0f);
                    dl->AddText(ImVec2(p_pos.x + 6, p_pos.y + 2), IM_COL32(200, 255, 200, 255), "P=Resume");
                } else {
                    // Lock icon pill (red)
                    ImVec2 lock_pos = ImVec2(start_x - 65, bar_pos.y - 1);
                    dl->AddRectFilled(lock_pos, ImVec2(lock_pos.x + 60, lock_pos.y + 18), IM_COL32(180, 60, 60, 255), 4.0f);
                    dl->AddText(ImVec2(lock_pos.x + 8, lock_pos.y + 2), IM_COL32(255, 255, 255, 255), "LOCKED");
                    
                    // P hint for pause
                    ImVec2 p_pos = ImVec2(lock_pos.x - 60, lock_pos.y);
                    dl->AddRectFilled(p_pos, ImVec2(p_pos.x + 55, p_pos.y + 18), IM_COL32(80, 70, 40, 255), 4.0f);
                    dl->AddText(ImVec2(p_pos.x + 6, p_pos.y + 2), IM_COL32(200, 200, 150, 255), "P=Pause");
                    
                    // ESC hint
                    ImVec2 esc_pos = ImVec2(p_pos.x - 70, p_pos.y);
                    dl->AddRectFilled(esc_pos, ImVec2(esc_pos.x + 65, esc_pos.y + 18), IM_COL32(60, 60, 80, 255), 4.0f);
                    dl->AddText(ImVec2(esc_pos.x + 6, esc_pos.y + 2), IM_COL32(200, 200, 200, 255), "ESC=Stop");
                }
            }
            
            if (ctx.is_animation_mode) {
                // ─────────────────────────────────────────────────────────────────────
                // ANIMATION RENDER MODE
                // ─────────────────────────────────────────────────────────────────────
                int cur_frame = ctx.render_settings.animation_current_frame;
                int start_frame = ctx.render_settings.animation_start_frame;
                int end_frame = ctx.render_settings.animation_end_frame;
                int total_frames = end_frame - start_frame + 1;
                int frames_done = cur_frame - start_frame;
                
                // Clamp progress to valid range
                float progress = (total_frames > 0) ? std::clamp((float)frames_done / (float)total_frames, 0.0f, 1.0f) : 0.0f;
                
                // Background bar
                ImVec2 bar_end = ImVec2(bar_pos.x + 180, bar_pos.y + 16);
                dl->AddRectFilled(bar_pos, bar_end, IM_COL32(40, 40, 50, 255), 4.0f);
                
                // Progress fill (gradient: orange to green)
                if (progress > 0.0f) {
                    ImVec2 fill_end = ImVec2(bar_pos.x + 180 * progress, bar_pos.y + 16);
                    ImU32 col_start = IM_COL32(255, 140, 50, 255);  // Orange
                    ImU32 col_end = IM_COL32(100, 220, 100, 255);   // Green
                    dl->AddRectFilledMultiColor(bar_pos, fill_end, col_start, col_end, col_end, col_start);
                    dl->AddRect(bar_pos, fill_end, IM_COL32(255, 255, 255, 40), 4.0f);
                }
                
                // Border
                dl->AddRect(bar_pos, bar_end, IM_COL32(80, 80, 100, 255), 4.0f);
                
                // Frame text inside bar
                char frame_text[32];
                snprintf(frame_text, sizeof(frame_text), "Frame %d / %d", cur_frame, end_frame);
                ImVec2 text_size = ImGui::CalcTextSize(frame_text);
                ImVec2 text_pos = ImVec2(bar_pos.x + (180 - text_size.x) * 0.5f, bar_pos.y + 1);
                dl->AddText(text_pos, IM_COL32(255, 255, 255, 255), frame_text);
                
                // Percentage on right
                ImGui::SameLine(start_x + 188);
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f), "%.0f%%", progress * 100.0f);
                
                // Spinning indicator
                static float spin_angle = 0.0f;
                spin_angle += ImGui::GetIO().DeltaTime * 4.0f;
                ImVec2 spin_center = ImVec2(start_x - 14, bar_pos.y + 8);
                float r = 5.0f;
                for (int i = 0; i < 8; i++) {
                    float angle = spin_angle + i * (3.14159f * 2.0f / 8.0f);
                    float alpha = 0.3f + 0.7f * (1.0f - (float)i / 8.0f);
                    ImVec2 dot = ImVec2(spin_center.x + cosf(angle) * r, spin_center.y + sinf(angle) * r);
                    dl->AddCircleFilled(dot, 2.0f, IM_COL32(255, 180, 50, (int)(alpha * 255)));
                }
            }
            else {
                // ─────────────────────────────────────────────────────────────────────
                // SINGLE FRAME RENDER MODE
                // ─────────────────────────────────────────────────────────────────────
                int current_samples = ctx.render_settings.render_current_samples;
                int target_samples = ctx.render_settings.render_target_samples;
                if (target_samples <= 0) target_samples = ctx.render_settings.final_render_samples;
                if (target_samples <= 0) target_samples = 128;
                
                // Clamp progress
                float progress = std::clamp((float)current_samples / (float)target_samples, 0.0f, 1.0f);
                
                // Background bar
                ImVec2 bar_end = ImVec2(bar_pos.x + 160, bar_pos.y + 16);
                dl->AddRectFilled(bar_pos, bar_end, IM_COL32(40, 40, 50, 255), 4.0f);
                
                // Progress fill (blue gradient)
                if (progress > 0.0f) {
                    ImVec2 fill_end = ImVec2(bar_pos.x + 160 * progress, bar_pos.y + 16);
                    dl->AddRectFilled(bar_pos, fill_end, IM_COL32(80, 150, 255, 255), 4.0f);
                }
                
                // Border
                dl->AddRect(bar_pos, bar_end, IM_COL32(80, 80, 100, 255), 4.0f);
                
                // Sample text inside bar
                char sample_text[32];
                snprintf(sample_text, sizeof(sample_text), "%d / %d spp", current_samples, target_samples);
                ImVec2 text_size = ImGui::CalcTextSize(sample_text);
                ImVec2 text_pos = ImVec2(bar_pos.x + (160 - text_size.x) * 0.5f, bar_pos.y + 1);
                dl->AddText(text_pos, IM_COL32(255, 255, 255, 255), sample_text);
                
                // Percentage on right
                ImGui::SameLine(start_x + 168);
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "%.0f%%", progress * 100.0f);
            }
        }
        else if (ctx.render_settings.is_final_render_mode) {
            // Render just finished
            float w = ImGui::CalcTextSize("Render Complete").x;
            ImGui::SameLine(screen_x - w - 20);
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "Render Complete");
        }
    }
    ImGui::End();

    // ImGui::End(); // Removed redundant End
    
    // ImGui::PopStyleColor(); // Removed hardcoded color push
    ImGui::PopStyleVar(2);

    // ---------------- BOTTOM PANEL (Resizable) ----------------
    bool show_bottom = (show_animation_panel || show_scene_log || show_terrain_graph || show_anim_graph || show_asset_browser);
    if (!show_bottom) return;

    // Use class member for persistent height
    // static float bottom_height = 280.0f; <-- REMOVED
    const float min_height = 100.0f;
    const float max_height = screen_y * 0.6f;  // Max 60% of screen
    const float resize_handle_height = 6.0f;
    
    // Clamp height to valid range
    bottom_panel_height = std::clamp(bottom_panel_height, min_height, max_height);

    // Calculate panel position
    float panel_top = screen_y - bottom_panel_height - status_bar_height;
    
    // --- RESIZE HANDLE (invisible button at top edge) ---
    ImGui::SetNextWindowPos(ImVec2(0, panel_top - resize_handle_height / 2), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_x, resize_handle_height), ImGuiCond_Always);
    
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));  // Transparent
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    
    if (ImGui::Begin("##BottomPanelResizer", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings))
    {
        // Draw a subtle resize indicator line
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 p1(0, panel_top);
        ImVec2 p2(screen_x, panel_top);
        draw_list->AddLine(p1, p2, IM_COL32(100, 100, 100, 180), 2.0f);
        
        // Handle resize dragging
        ImGui::InvisibleButton("##ResizeHandle", ImVec2(screen_x, resize_handle_height));
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            bottom_panel_height -= ImGui::GetIO().MouseDelta.y;
            bottom_panel_height = std::clamp(bottom_panel_height, min_height, max_height);
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    // --- MAIN BOTTOM PANEL ---
    ImGui::SetNextWindowPos(ImVec2(0, panel_top), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_x, bottom_panel_height), ImGuiCond_Always);
    if (focus_bottom_panel_next_frame) {
        ImGui::SetNextWindowFocus();
    }

    // Use theme color with alpha
    ImGui::SetNextWindowBgAlpha(panel_alpha);
    // ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f)); // Removed hardcoded

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8)); // Added padding to prevent "stuck to left" look
    if (ImGui::Begin("BottomPanel", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse))
    {
        if (focus_bottom_panel_next_frame) {
            ImGui::SetWindowFocus();
            focus_bottom_panel_next_frame = false;
        }
        if (show_animation_panel) {
            drawTimelineContent(ctx);
        }
        else if (show_scene_log) {
            drawLogPanelEmbedded();
        }
        else if (show_terrain_graph) {
            // Terrain Node Graph Editor
            // Disable edit tool when using node graph (performance optimization)
            terrain_brush.enabled = false;
            
            TerrainObject* activeTerrain = nullptr;
            if (terrain_brush.active_terrain_id != -1) {
                activeTerrain = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
            }

            if (activeTerrain) {
                // Ensure graph exists
                if (!activeTerrain->nodeGraph) {
                    activeTerrain->nodeGraph = std::make_shared<TerrainNodesV2::TerrainNodeGraphV2>();
                }
                
                // Auto-create default graph if empty
                if (activeTerrain->nodeGraph->nodes.empty()) {
                    activeTerrain->nodeGraph->createDefaultGraph(activeTerrain);
                }
                
                // Set callbacks
                if (!terrainNodeEditorUI.onOpenFileDialog) {
                     terrainNodeEditorUI.onOpenFileDialog = [](const wchar_t* filter) -> std::string {
                         return SceneUI::openFileDialogW(filter);
                     };
                }
                if (!terrainNodeEditorUI.onSaveFileDialog) {
                     terrainNodeEditorUI.onSaveFileDialog = [](const wchar_t* filter, const wchar_t* defName) -> std::string {
                         return SceneUI::saveFileDialogW(filter, L"png");
                     };
                }
                terrainNodeEditorUI.draw(ctx, *activeTerrain->nodeGraph, activeTerrain);
            }
            else {
                 ImGui::TextColored(ImVec4(1, 1, 0, 1), "Please select a terrain to edit its node graph.");
                 // Fallback to global graph just to show the UI (optional, or just show nothing)
                 // terrainNodeEditorUI.draw(ctx, terrainNodeGraph, nullptr);
            }
        }
        else if (show_anim_graph) {
            // Animation Node Graph Editor
            drawAnimationGraphPanel(ctx);
        }
        else if (show_asset_browser) {
            drawAssetBrowser(ctx, true);
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(); // Pop WindowPadding
    // ImGui::PopStyleVar(); // Removed redundant PopStyleVar
    // ImGui::PopStyleColor(); // Removed hardcoded color push
}
bool SceneUI::drawOverlays(UIContext& ctx)
{
    // === UI SETTINGS SERIALIZATION ===
    // 1. LOAD from SceneData if a new project was loaded
    static int last_local_counter = -1;
    if (ctx.scene.load_counter != last_local_counter) {
        if (!ctx.scene.ui_settings_json_str.empty()) {
            try {
                auto j = nlohmann::json::parse(ctx.scene.ui_settings_json_str);
                
                // Pro Camera Settings
                if (j.contains("viewport_settings")) {
                    auto& vs = j["viewport_settings"];
                    viewport_settings.show_histogram = vs.value("show_histogram", false);
                    viewport_settings.histogram_mode = vs.value("histogram_mode", 0);
                    viewport_settings.histogram_opacity = vs.value("histogram_opacity", 0.5f);
                    viewport_settings.show_focus_peaking = vs.value("show_focus_peaking", false);
                    viewport_settings.focus_peaking_color = vs.value("focus_peaking_color", 0);
                    viewport_settings.focus_peaking_threshold = vs.value("focus_peaking_threshold", 0.15f);
                    viewport_settings.show_zebra = vs.value("show_zebra", false);
                    viewport_settings.zebra_threshold = vs.value("zebra_threshold", 0.95f);
                    viewport_settings.show_af_points = vs.value("show_af_points", false);
                    viewport_settings.af_mode = vs.value("af_mode", 0);
                    viewport_settings.af_selected_point = vs.value("af_selected_point", 4);
                    viewport_settings.focus_mode = vs.value("focus_mode", 1);
                }
            } catch (...) {}
        }
        last_local_counter = ctx.scene.load_counter;
    }

    // 2. SAVE to SceneData if settings changed
    static ViewportDisplaySettings last_vs_check;
    static bool first_run = true;
    if (first_run) {
        std::memset(&last_vs_check, 0, sizeof(last_vs_check));
        first_run = false;
    }
    
    if (std::memcmp(&viewport_settings, &last_vs_check, sizeof(ViewportDisplaySettings)) != 0) {
        try {
            nlohmann::json root;
            if (!ctx.scene.ui_settings_json_str.empty()) {
                try { root = nlohmann::json::parse(ctx.scene.ui_settings_json_str); } catch(...) {}
            }
            
            nlohmann::json& vs = root["viewport_settings"];
            vs["show_histogram"] = viewport_settings.show_histogram;
            vs["histogram_mode"] = viewport_settings.histogram_mode;
            vs["histogram_opacity"] = viewport_settings.histogram_opacity;
            vs["show_focus_peaking"] = viewport_settings.show_focus_peaking;
            vs["focus_peaking_color"] = viewport_settings.focus_peaking_color;
            vs["focus_peaking_threshold"] = viewport_settings.focus_peaking_threshold;
            vs["show_zebra"] = viewport_settings.show_zebra;
            vs["zebra_threshold"] = viewport_settings.zebra_threshold;
            vs["show_af_points"] = viewport_settings.show_af_points;
            vs["af_mode"] = viewport_settings.af_mode;
            vs["af_selected_point"] = viewport_settings.af_selected_point;
            vs["focus_mode"] = viewport_settings.focus_mode;
            
            ctx.scene.ui_settings_json_str = root.dump();
            std::memcpy(&last_vs_check, &viewport_settings, sizeof(ViewportDisplaySettings));
        } catch (...) {}
    }

    bool gizmo_hit = false;

    // Draw Viewport HUDs
    // Render status is now integrated into drawViewportMessages

    if (ctx.scene.camera && ctx.selection.show_gizmo) {
        drawLightGizmos(ctx, gizmo_hit);
        drawForceFieldGizmos(ctx, gizmo_hit);
    }

    // === PRO CAMERA HUD OVERLAYS ===
    // These are drawn on top of everything else
    drawHistogramOverlay(ctx);
    drawFocusPeakingOverlay(ctx);
    drawZebraOverlay(ctx);
    drawAFPointsOverlay(ctx);

    // Asset Browser -> Viewport drag & drop target
    if (ImGui::IsDragDropActive()) {
        if (const ImGuiPayload* drag_payload = ImGui::GetDragDropPayload()) {
            if (drag_payload->IsDataType("ASSET_BROWSER_ENTRY") && drag_payload->Data) {
                const char* payload_text = static_cast<const char*>(drag_payload->Data);
                if (payload_text && payload_text[0] != '\0') {
                    const AssetRecord* dragged_asset = asset_registry.findByRelativeDirectory(payload_text);
                    if (!dragged_asset || !isPlaceableAssetRecord(*dragged_asset)) {
                        dragged_asset = nullptr;
                    }
                    Vec3 ghost_hit_point, ghost_hit_normal;
                    if (dragged_asset && raycastViewportPlacement(ctx, ImGui::GetMousePos(), ghost_hit_point, ghost_hit_normal)) {
                        const std::string payload_key = dragged_asset->relative_entry_path.generic_string();
                        auto it = asset_drag_bbox_cache.find(payload_key);
                        if (it == asset_drag_bbox_cache.end()) {
                            Vec3 preview_min, preview_max;
                            if (computeAssetPreviewBounds(*dragged_asset, preview_min, preview_max)) {
                                it = asset_drag_bbox_cache.emplace(payload_key, std::make_pair(preview_min, preview_max)).first;
                            }
                        }

                        if (it != asset_drag_bbox_cache.end()) {
                            drawAssetDragGhost(
                                ctx,
                                dragged_asset->name,
                                ghost_hit_point,
                                it->second.first,
                                it->second.second);
                        }
                    }
                }
            }
        }

        const bool show_bottom =
            (show_animation_panel || show_scene_log || show_terrain_graph || show_anim_graph || show_asset_browser);
        const float menu_height = 19.0f;
        const float status_bar_height = 24.0f;
        const float left_offset = showSidePanel ? side_panel_width : 0.0f;
        const float top = menu_height;
        const float right = ImGui::GetIO().DisplaySize.x;
        const float bottom = ImGui::GetIO().DisplaySize.y -
            status_bar_height -
            (show_bottom ? bottom_panel_height : 0.0f);
        const float width = (std::max)(0.0f, right - left_offset);
        const float height = (std::max)(0.0f, bottom - top);

        if (width > 1.0f && height > 1.0f) {
            ImGui::SetNextWindowPos(ImVec2(left_offset, top), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.0f);

            const ImGuiWindowFlags drop_flags =
                ImGuiWindowFlags_NoTitleBar |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoDecoration;

            if (ImGui::Begin("##ViewportAssetDropTarget", nullptr, drop_flags)) {
                ImGui::SetCursorScreenPos(ImVec2(left_offset, top));
                ImGui::InvisibleButton("##ViewportDropSurface", ImVec2(width, height));
                if (ImGui::BeginDragDropTarget()) {
                    if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ASSET_BROWSER_ENTRY")) {
                        const char* payload_text = static_cast<const char*>(payload->Data);
                        if (payload_text && payload_text[0] != '\0') {
                            if (const AssetRecord* dropped_asset = asset_registry.findByRelativeDirectory(payload_text)) {
                                appendAssetToScene(ctx, dropped_asset->entry_path, dropped_asset->name);
                            }
                        }
                    }
                    ImGui::EndDragDropTarget();
                }
            }
            ImGui::End();
        }
    }

    // === VIEWPORT EDGE FRAME ===
    // Draw a subtle border on the right edge of the viewport to clearly delineate the area
    {
        ImGuiIO& io = ImGui::GetIO();
        ImDrawList* draw_list = ImGui::GetForegroundDrawList();
        
        // Right edge border (1px dark line)
        float border_x = io.DisplaySize.x - 1.0f;
        ImU32 border_color = IM_COL32(40, 40, 50, 200);  // Dark subtle border
        draw_list->AddLine(
            ImVec2(border_x, 0), 
            ImVec2(border_x, io.DisplaySize.y), 
            border_color, 
            1.0f
        );
        
        // Optional: Add a subtle highlight line just inside
        ImU32 highlight_color = IM_COL32(60, 60, 70, 100);  // Very subtle highlight
        draw_list->AddLine(
            ImVec2(border_x - 1.0f, 0), 
            ImVec2(border_x - 1.0f, io.DisplaySize.y), 
            highlight_color, 
            1.0f
        );
    }

    return gizmo_hit;
}



void SceneUI::handleSceneInteraction(UIContext& ctx, bool gizmo_hit)
{
    if (ctx.scene.initialized && !gizmo_hit) {
        handleMouseSelection(ctx);
        handleMarqueeSelection(ctx);
    }
}
void SceneUI::processDeferredSceneUpdates(UIContext& ctx)
{
    if (is_bvh_dirty && !ImGuizmo::IsUsing()) {
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();
        
        // OPTIMIZATION: Only update OptiX geometry when OptiX rendering is enabled
        if (ctx.render_settings.use_optix && ctx.backend_ptr) {
            ctx.backend_ptr->updateGeometry(ctx.scene.world.objects);
            ctx.backend_ptr->resetAccumulation();
        }
        is_bvh_dirty = false;
    }
}
void SceneUI::drawAuxWindows(UIContext& ctx)
{
    if (show_controls_window) {
        ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Quick Guide & Shortcuts", &show_controls_window)) {
            drawControlsContent();
        }
        ImGui::End();
    }

}

bool SceneUI::raycastViewportPlacement(UIContext& ctx, const ImVec2& screen_pos, Vec3& hit_point, Vec3& hit_normal) const
{
    if (!ctx.scene.camera || !ctx.scene.bvh) {
        return false;
    }

    const ImVec2 display = ImGui::GetIO().DisplaySize;
    if (display.x <= 1.0f || display.y <= 1.0f) {
        return false;
    }

    const float u = std::clamp(screen_pos.x / display.x, 0.0f, 1.0f);
    const float v = std::clamp(1.0f - (screen_pos.y / display.y), 0.0f, 1.0f);
    Ray ray = ctx.scene.camera->get_ray(u, v);

    HitRecord rec;
    if (ctx.scene.bvh->hit(ray, 0.001f, 1e30f, rec)) {
        hit_point = rec.point;
        hit_normal = rec.normal.length_squared() > 0.0001f ? rec.normal.normalize() : Vec3(0, 1, 0);
        return true;
    }

    const float denom = ray.direction.y;
    if (std::abs(denom) > 1e-5f) {
        const float t = -ray.origin.y / denom;
        if (t > 0.0f) {
            hit_point = ray.origin + ray.direction * t;
            hit_normal = Vec3(0, 1, 0);
            return true;
        }
    }

    return false;
}

void SceneUI::drawAssetDragGhost(UIContext& ctx, const std::string& asset_name, const Vec3& hit_point, const Vec3& bounds_min, const Vec3& bounds_max) const
{
    if (!ctx.scene.camera) {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;
    const Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    const Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    const float fov_rad = static_cast<float>(cam.vfov * 3.14159265359 / 180.0);
    const float tan_half_fov = tanf(fov_rad * 0.5f);
    const float aspect = io.DisplaySize.x / (std::max)(1.0f, io.DisplaySize.y);

    auto project_point = [&](const Vec3& point, ImVec2& out) -> bool {
        const Vec3 to_point = point - cam.lookfrom;
        const float depth = to_point.dot(cam_forward);
        if (depth <= 0.01f) {
            return false;
        }

        const float local_x = to_point.dot(cam_right);
        const float local_y = to_point.dot(cam_up);
        const float half_height = depth * tan_half_fov;
        const float half_width = half_height * aspect;
        if (std::abs(half_width) < 1e-5f || std::abs(half_height) < 1e-5f) {
            return false;
        }

        const float ndc_x = local_x / half_width;
        const float ndc_y = local_y / half_height;
        out = ImVec2(
            (ndc_x * 0.5f + 0.5f) * io.DisplaySize.x,
            (0.5f - ndc_y * 0.5f) * io.DisplaySize.y);
        return true;
    };

    const Vec3 base_center(
        (bounds_min.x + bounds_max.x) * 0.5f,
        bounds_min.y,
        (bounds_min.z + bounds_max.z) * 0.5f);
    const Vec3 delta = hit_point - base_center;
    const Vec3 placed_min = bounds_min + delta;
    const Vec3 placed_max = bounds_max + delta;

    const Vec3 corners[8] = {
        Vec3(placed_min.x, placed_min.y, placed_min.z),
        Vec3(placed_max.x, placed_min.y, placed_min.z),
        Vec3(placed_min.x, placed_max.y, placed_min.z),
        Vec3(placed_max.x, placed_max.y, placed_min.z),
        Vec3(placed_min.x, placed_min.y, placed_max.z),
        Vec3(placed_max.x, placed_min.y, placed_max.z),
        Vec3(placed_min.x, placed_max.y, placed_max.z),
        Vec3(placed_max.x, placed_max.y, placed_max.z)
    };

    ImVec2 screen_corners[8];
    bool any_projected = false;
    for (int i = 0; i < 8; ++i) {
        any_projected = project_point(corners[i], screen_corners[i]) || any_projected;
    }
    if (!any_projected) {
        return;
    }

    ImVec2 center;
    if (!project_point(hit_point, center)) {
        center = ImGui::GetMousePos();
    }

    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    const float radius = 18.0f;
    const ImU32 outer = IM_COL32(110, 255, 190, 210);
    const ImU32 inner = IM_COL32(110, 255, 190, 80);
    const ImU32 text_shadow = IM_COL32(0, 0, 0, 180);
    const ImU32 box_fill = IM_COL32(110, 255, 190, 20);

    const int edges[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0},
        {4, 5}, {5, 7}, {7, 6}, {6, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    draw_list->AddQuadFilled(screen_corners[0], screen_corners[1], screen_corners[3], screen_corners[2], box_fill);
    draw_list->AddQuadFilled(screen_corners[4], screen_corners[5], screen_corners[7], screen_corners[6], box_fill);
    for (const auto& edge : edges) {
        draw_list->AddLine(screen_corners[edge[0]], screen_corners[edge[1]], outer, 1.5f);
    }

    draw_list->AddCircle(center, radius, outer, 32, 2.0f);
    draw_list->AddCircle(center, radius * 0.45f, inner, 24, 2.0f);
    draw_list->AddLine(ImVec2(center.x - radius * 0.7f, center.y), ImVec2(center.x + radius * 0.7f, center.y), outer, 1.5f);
    draw_list->AddLine(ImVec2(center.x, center.y - radius * 0.7f), ImVec2(center.x, center.y + radius * 0.7f), outer, 1.5f);

    const std::string label = "Drop: " + asset_name;
    const ImVec2 text_pos(center.x + 22.0f, center.y - 8.0f);
    draw_list->AddText(ImVec2(text_pos.x + 1.0f, text_pos.y + 1.0f), text_shadow, label.c_str());
    draw_list->AddText(text_pos, outer, label.c_str());
}

bool SceneUI::appendAnimationClipAssetToScene(UIContext& ctx, const AssetRecord& asset, const std::string& display_name)
{
    auto loader = std::make_shared<AssimpLoader>();
    auto [loaded_triangles, loaded_animations, loaded_bone_data] = loader->loadModelToTriangles(asset.entry_path.string(), nullptr, "");
    (void)loaded_triangles;
    (void)loaded_bone_data;

    if (loaded_animations.empty()) {
        addViewportMessage("No animation clips found in asset: " + display_name, 3.5f, ImVec4(1.0f, 0.4f, 0.3f, 1.0f));
        return false;
    }

    const std::string source_prefix = loader->currentImportName;
    std::string target_import_name;
    if (asset.animation_binding != "global") {
        target_import_name = findSelectedModelImportName(ctx);
    }

    if (asset.animation_binding != "global" && target_import_name.empty() && ctx.scene.importedModelContexts.size() > 1) {
        addViewportMessage("Select a target character first; clip imported as global.", 3.5f, ImVec4(1.0f, 0.8f, 0.3f, 1.0f));
    }

    for (auto& anim : loaded_animations) {
        if (!anim) {
            continue;
        }

        anim->positionKeys = retargetAnimationKeyMap(anim->positionKeys, source_prefix, target_import_name);
        anim->rotationKeys = retargetAnimationKeyMap(anim->rotationKeys, source_prefix, target_import_name);
        anim->scalingKeys = retargetAnimationKeyMap(anim->scalingKeys, source_prefix, target_import_name);
        anim->name = makeUniqueAnimationClipName(
            ctx.scene.animationDataList,
            retargetAnimationNodeName(anim->name, source_prefix, target_import_name));
        anim->modelName = target_import_name;
        ctx.scene.animationDataList.push_back(anim);
    }

    if (!target_import_name.empty()) {
        for (auto& model_ctx : ctx.scene.importedModelContexts) {
            if (model_ctx.importName == target_import_name) {
                model_ctx.hasAnimation = true;
                break;
            }
        }
    }

    ctx.renderer.initializeAnimationSystem(ctx.scene);
    for (auto& model_ctx : ctx.scene.importedModelContexts) {
        if (!model_ctx.animator) {
            continue;
        }

        std::vector<std::shared_ptr<AnimationData>> model_clips;
        for (const auto& anim : ctx.scene.animationDataList) {
            if (!anim) {
                continue;
            }
            if (anim->modelName == model_ctx.importName || (anim->modelName.empty() && ctx.scene.importedModelContexts.size() == 1)) {
                model_clips.push_back(anim);
            }
        }
        model_ctx.animator->registerClips(model_clips);
    }

    AnimationController::getInstance().registerClips(ctx.scene.animationDataList);

    ctx.renderer.resetCPUAccumulation();
    if (ctx.backend_ptr) {
        ctx.backend_ptr->resetAccumulation();
    }

    g_ProjectManager.markModified();
    addViewportMessage(
        "Animation clips imported: " + std::to_string(loaded_animations.size()) +
        (target_import_name.empty() ? "" : " -> " + target_import_name),
        3.0f,
        ImVec4(0.4f, 1.0f, 0.6f, 1.0f));
    return true;
}

void SceneUI::appendAssetToScene(UIContext& ctx, const std::filesystem::path& asset_path, const std::string& display_name)
{
    const AssetRecord* asset_record = nullptr;
    for (const auto& asset : asset_registry.getAssets()) {
        if (asset.entry_path == asset_path) {
            asset_record = &asset;
            break;
        }
    }

    if (asset_record && asset_record->asset_kind == "anim_clip") {
        appendAnimationClipAssetToScene(ctx, *asset_record, display_name);
        return;
    }

    if (asset_record && (asset_record->asset_kind == "vdb" || asset_record->asset_kind == "vdb_sequence")) {
        Vec3 drop_hit_point(0, 0, 0);
        Vec3 drop_hit_normal(0, 1, 0);
        const bool has_drop_point = raycastViewportPlacement(ctx, ImGui::GetMousePos(), drop_hit_point, drop_hit_normal);

        auto vdb = std::make_shared<VDBVolume>();
        const bool loaded = asset_record->is_sequence
            ? vdb->loadVDBSequence(asset_record->entry_path.string())
            : vdb->loadVDB(asset_record->entry_path.string());
        if (!loaded) {
            addViewportMessage("Failed to load VDB asset: " + display_name, 3.0f, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            return;
        }

        Vec3 bmin = vdb->getLocalBoundsMin();
        Vec3 bmax = vdb->getLocalBoundsMax();
        const Vec3 size = bmax - bmin;
        float max_dim = (std::max)(size.x, (std::max)(size.y, size.z));
        float scale_factor = 1.0f;
        if (max_dim > 50.0f) {
            scale_factor = 5.0f / max_dim;
        } else if (max_dim > 0.0f && max_dim < 0.01f) {
            scale_factor = 5.0f / (std::max)(max_dim, 0.001f);
        }
        scale_factor = (std::max)(0.0001f, (std::min)(scale_factor, 1000.0f));
        vdb->setScale(Vec3(scale_factor));
        vdb->centerPivotToBottomCenter();
        vdb->setPosition(has_drop_point ? drop_hit_point : Vec3(0, 0, 0));

        if (!vdb->uploadToGPU()) {
            SCENE_LOG_WARN("VDB asset uploaded to CPU only (GPU upload failed)");
        }

        if (vdb->hasGrid("temperature")) {
            vdb->setShader(VolumeShader::createFirePreset());
        } else {
            vdb->setShader(VolumeShader::createSmokePreset());
        }

        if (asset_record->vdb_shader_preset == "fire") {
            vdb->setShader(VolumeShader::createFirePreset());
        } else if (asset_record->vdb_shader_preset == "smoke") {
            vdb->setShader(VolumeShader::createSmokePreset());
        }

        if (auto shader = vdb->getShader()) {
            shader->density.multiplier = asset_record->vdb_density_multiplier;
            shader->emission.temperature_scale = asset_record->vdb_temperature_scale;
            if (shader->emission.mode == VolumeEmissionMode::Constant) {
                shader->emission.intensity = asset_record->vdb_emission_intensity;
            } else if (shader->emission.mode == VolumeEmissionMode::Blackbody) {
                shader->emission.blackbody_intensity = (std::max)(0.0f, asset_record->vdb_emission_intensity);
            }
        }

        if (!display_name.empty()) {
            vdb->name = display_name;
        }

        ctx.scene.addVDBVolume(vdb);
        ctx.scene.world.objects.push_back(vdb);
        ctx.selection.selectVDBVolume(vdb, static_cast<int>(ctx.scene.vdb_volumes.size()) - 1, vdb->name);
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();

        if (ctx.backend_ptr) {
            if (ctx.render_settings.use_vulkan) {
                g_vulkan_rebuild_pending = true;
            } else {
                VolumetricRenderer::syncVolumetricData(ctx.scene, ctx.backend_ptr);
                ctx.backend_ptr->resetAccumulation();
            }
        }

        g_ProjectManager.markModified();
        addViewportMessage("Appended VDB asset: " + display_name, 2.5f, ImVec4(0.4f, 1.0f, 0.6f, 1.0f));
        return;
    }

    const size_t objects_before = ctx.scene.world.objects.size();
    Vec3 drop_hit_point(0, 0, 0);
    Vec3 drop_hit_normal(0, 1, 0);
    const bool has_drop_point = raycastViewportPlacement(ctx, ImGui::GetMousePos(), drop_hit_point, drop_hit_normal);
    const std::string import_prefix = makeUniqueAssetImportPrefix(asset_path);
    ctx.renderer.create_scene(
        ctx.scene,
        ctx.backend_ptr,
        asset_path.string(),
        nullptr,
        true,
        import_prefix);

    if (has_drop_point && ctx.scene.world.objects.size() > objects_before) {
        Vec3 bounds_min(1e30f, 1e30f, 1e30f);
        Vec3 bounds_max(-1e30f, -1e30f, -1e30f);
        std::vector<std::shared_ptr<Transform>> new_transforms;
        std::shared_ptr<Triangle> first_new_triangle;

        for (size_t i = objects_before; i < ctx.scene.world.objects.size(); ++i) {
            auto tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[i]);
            if (!tri) {
                continue;
            }

            if (!first_new_triangle) {
                first_new_triangle = tri;
            }

            AABB bounds;
            if (tri->bounding_box(0.0f, 0.0f, bounds)) {
                bounds_min.x = (std::min)(bounds_min.x, bounds.min.x);
                bounds_min.y = (std::min)(bounds_min.y, bounds.min.y);
                bounds_min.z = (std::min)(bounds_min.z, bounds.min.z);
                bounds_max.x = (std::max)(bounds_max.x, bounds.max.x);
                bounds_max.y = (std::max)(bounds_max.y, bounds.max.y);
                bounds_max.z = (std::max)(bounds_max.z, bounds.max.z);
            }

            if (auto th = tri->getTransformHandle()) {
                if (std::find(new_transforms.begin(), new_transforms.end(), th) == new_transforms.end()) {
                    new_transforms.push_back(th);
                }
            }
        }

        if (!new_transforms.empty() && bounds_min.x < 1e20f && bounds_max.x > -1e20f) {
            const Vec3 current_base_center(
                (bounds_min.x + bounds_max.x) * 0.5f,
                bounds_min.y,
                (bounds_min.z + bounds_max.z) * 0.5f);
            const Vec3 delta = drop_hit_point - current_base_center;
            const Matrix4x4 translation = Matrix4x4::translation(delta);

            for (const auto& transform : new_transforms) {
                if (transform) {
                    transform->setBase(translation * transform->base);
                }
            }
            for (size_t i = objects_before; i < ctx.scene.world.objects.size(); ++i) {
                if (auto tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[i])) {
                    tri->updateTransformedVertices();
                }
            }
        }

        if (first_new_triangle) {
            ctx.selection.selectObject(first_new_triangle, -1, first_new_triangle->nodeName);
        }
    }
    else if (ctx.scene.world.objects.size() > objects_before) {
        for (size_t i = objects_before; i < ctx.scene.world.objects.size(); ++i) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[i])) {
                ctx.selection.selectObject(tri, -1, tri->nodeName);
                break;
            }
        }
    }

    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();

    if (ctx.backend_ptr) {
        ctx.renderer.rebuildBackendGeometry(ctx.scene);
        ctx.backend_ptr->setLights(ctx.scene.lights);
        if (ctx.scene.camera) {
            ctx.renderer.syncCameraToBackend(*ctx.scene.camera);
        }
        ctx.backend_ptr->resetAccumulation();
    }

    ctx.start_render = true;
    addViewportMessage("Asset appended: " + display_name, 2.5f, ImVec4(0.3f, 1.0f, 0.5f, 1.0f));
}

void SceneUI::startAsyncAssetLibraryRefresh(const std::filesystem::path& root_path, const std::string& status_text)
{
    const std::filesystem::path normalized = normalizeAbsolutePath(root_path);
    if (asset_library_refresh_in_progress) {
        if (pending_asset_library_root == normalized) {
            return;
        }
        return;
    }

    asset_library_refresh_in_progress = true;
    pending_asset_library_root = normalized;
    asset_library_refresh_status = status_text;
    asset_registry_refresh_future = std::async(std::launch::async, [normalized]() -> std::pair<AssetRegistry, bool> {
        AssetRegistry refreshed_registry;
        const bool success = normalized.empty()
            ? refreshed_registry.refresh(std::filesystem::path())
            : refreshed_registry.refresh(normalized);
        return std::make_pair(std::move(refreshed_registry), success);
    });
}

void SceneUI::pollAsyncAssetLibraryRefresh()
{
    if (!asset_library_refresh_in_progress || !asset_registry_refresh_future.valid()) {
        return;
    }

    if (asset_registry_refresh_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        return;
    }

    auto refresh_result = asset_registry_refresh_future.get();
    asset_library_refresh_in_progress = false;
    releaseAssetBrowserThumbnailTextures();
    asset_registry = std::move(refresh_result.first);

    if (!refresh_result.second) {
        asset_library_refresh_status = "Asset scan failed";
    } else {
        asset_library_refresh_status.clear();
    }
}

void SceneUI::drawAssetBrowser(UIContext& ctx, bool embedded)
{
    pollAsyncAssetLibraryRefresh();

    const auto resetAssetDetailState = [&]() {
        releaseSelectedAssetPreviewTexture();
        selected_asset_tags_edit.clear();
        selected_asset_tags_edit_target.clear();
    };

    ensureDefaultAssetLibrary(asset_library_paths);
    if (asset_library_paths.empty()) {
        if (asset_registry.getRootPath() != std::filesystem::path() && !asset_library_refresh_in_progress) {
            asset_registry.refresh(std::filesystem::path());
        }
    } else {
        active_asset_library_index = (std::max)(0, (std::min)(active_asset_library_index, static_cast<int>(asset_library_paths.size()) - 1));
        const std::filesystem::path active_library_root = normalizeAbsolutePath(asset_library_paths[active_asset_library_index]);
        if (asset_registry.getRootPath() != active_library_root || !std::filesystem::exists(active_library_root)) {
            if (!asset_library_refresh_in_progress || pending_asset_library_root != active_library_root) {
                startAsyncAssetLibraryRefresh(active_library_root, "Scanning assets in background...");
                selected_asset_relative_dir.clear();
                selected_asset_folder_relative_dir.clear();
                asset_drag_bbox_cache.clear();
                resetAssetDetailState();
            }
        }
    }

    if (!embedded) {
        ImGui::SetNextWindowSize(ImVec2(1080, 720), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Asset Browser", &show_asset_browser, ImGuiWindowFlags_NoCollapse)) {
            ImGui::End();
            return;
        }
    }

    const std::filesystem::path root_path = asset_registry.getRootPath();
    std::vector<std::string> library_labels;
    library_labels.reserve(asset_library_paths.size());
    for (std::size_t i = 0; i < asset_library_paths.size(); ++i) {
        const std::filesystem::path& library_path = asset_library_paths[i];
        std::string label = library_path.filename().string();
        if (label.empty()) {
            label = library_path.string();
        }
        if (i == 0) {
            label += " (default)";
        }
        library_labels.push_back(label);
    }

    ImGui::TextWrapped("Asset Root: %s", root_path.empty() ? "<not found>" : root_path.string().c_str());
    ImGui::SameLine();
    if (ImGui::Button("Refresh")) {
        releaseAssetBrowserThumbnailTextures();
        if (!asset_library_paths.empty() &&
            active_asset_library_index >= 0 &&
            active_asset_library_index < static_cast<int>(asset_library_paths.size())) {
            startAsyncAssetLibraryRefresh(normalizeAbsolutePath(asset_library_paths[active_asset_library_index]), "Refreshing asset library...");
        } else {
            asset_registry.refresh(std::filesystem::path());
        }
    }
    if (asset_library_refresh_in_progress) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", asset_library_refresh_status.empty() ? "Scanning assets..." : asset_library_refresh_status.c_str());
    }
    if (!library_labels.empty()) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(240.0f);
        std::vector<const char*> combo_items;
        combo_items.reserve(library_labels.size());
        for (const auto& label : library_labels) {
            combo_items.push_back(label.c_str());
        }
        int selected_library = active_asset_library_index;
        if (ImGui::Combo("Library", &selected_library, combo_items.data(), static_cast<int>(combo_items.size()))) {
            if (selected_library >= 0 && selected_library < static_cast<int>(asset_library_paths.size())) {
                const std::filesystem::path target_library = normalizeAbsolutePath(asset_library_paths[selected_library]);
                active_asset_library_index = selected_library;
                selected_asset_relative_dir.clear();
                selected_asset_folder_relative_dir.clear();
                asset_drag_bbox_cache.clear();
                resetAssetDetailState();
                startAsyncAssetLibraryRefresh(target_library, "Opening asset library...");
            }
        }
        if (asset_library_paths.size() > 1 && active_asset_library_index > 0) {
            ImGui::SameLine();
            if (ImGui::Button("Remove Library")) {
                asset_library_paths.erase(asset_library_paths.begin() + active_asset_library_index);
                active_asset_library_index = 0;
                ensureDefaultAssetLibrary(asset_library_paths);
                if (!asset_library_paths.empty()) {
                    releaseAssetBrowserThumbnailTextures();
                    startAsyncAssetLibraryRefresh(normalizeAbsolutePath(asset_library_paths[active_asset_library_index]), "Opening asset library...");
                } else {
                    asset_registry.refresh(std::filesystem::path());
                }
                selected_asset_relative_dir.clear();
                selected_asset_folder_relative_dir.clear();
                asset_drag_bbox_cache.clear();
                resetAssetDetailState();
                g_ProjectManager.markModified();
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Library")) {
        const std::string folder = selectFolderDialogW(L"Select Asset Library Folder");
        if (!folder.empty()) {
            const std::filesystem::path normalized = normalizeAbsolutePath(folder);
            std::error_code path_ec;
            if (!std::filesystem::exists(normalized, path_ec) || !std::filesystem::is_directory(normalized, path_ec)) {
                const std::string reason = path_ec ? path_ec.message() : "not a directory";
                addViewportMessage("Selected folder is not a valid asset library: " + reason, 4.0f, ImVec4(1.0f, 0.5f, 0.3f, 1.0f));
            } else {
                const auto existing = std::find_if(asset_library_paths.begin(), asset_library_paths.end(), [&](const std::filesystem::path& candidate) {
                return normalizeAbsolutePath(candidate) == normalized;
                });
                if (existing != asset_library_paths.end()) {
                    addViewportMessage("Asset library already added.", 2.5f, ImVec4(1.0f, 0.8f, 0.3f, 1.0f));
                } else {
                    asset_library_paths.push_back(normalized);
                    active_asset_library_index = static_cast<int>(asset_library_paths.size()) - 1;
                    selected_asset_relative_dir.clear();
                    selected_asset_folder_relative_dir.clear();
                    asset_drag_bbox_cache.clear();
                    resetAssetDetailState();
                    g_ProjectManager.markModified();
                    addViewportMessage("Asset library added.", 2.5f, ImVec4(0.3f, 1.0f, 0.5f, 1.0f));
                    startAsyncAssetLibraryRefresh(normalized, "Scanning new asset library...");
                }
            }
        }
    }

    std::array<char, 256> search_buffer{};
    std::snprintf(search_buffer.data(), search_buffer.size(), "%s", asset_browser_search.c_str());
    ImGui::SetNextItemWidth(240.0f);
    if (ImGui::InputText("Search", search_buffer.data(), search_buffer.size())) {
        asset_browser_search = search_buffer.data();
        active_asset_smart_folder_index = -1;
    }
    ImGui::SameLine();
    std::array<char, 192> tag_filter_buffer{};
    std::snprintf(tag_filter_buffer.data(), tag_filter_buffer.size(), "%s", asset_browser_tag_filter.c_str());
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputTextWithHint("Tags", "fire, smoke", tag_filter_buffer.data(), tag_filter_buffer.size())) {
        asset_browser_tag_filter = tag_filter_buffer.data();
        active_asset_smart_folder_index = -1;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110.0f);
    ImGui::Combo("View", &asset_browser_view_mode, "Tiles\0Compact\0Details\0");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150.0f);
    ImGui::SliderFloat("Size", &asset_browser_thumbnail_size, 96.0f, 220.0f, "%.0f px");
    ImGui::SameLine();
    if (ImGui::Checkbox("Only 3D", &asset_browser_only_3d)) {
        active_asset_smart_folder_index = -1;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Favorites", &asset_browser_favorites_only)) {
        active_asset_smart_folder_index = -1;
    }
    ImGui::SameLine();
    std::array<char, 128> smart_folder_name_buffer{};
    std::snprintf(smart_folder_name_buffer.data(), smart_folder_name_buffer.size(), "%s", asset_smart_folder_name.c_str());
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputTextWithHint("Collection", "save current filter", smart_folder_name_buffer.data(), smart_folder_name_buffer.size())) {
        asset_smart_folder_name = smart_folder_name_buffer.data();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save Collection")) {
        AssetSmartFolderPreset preset;
        preset.name = trimCopy(asset_smart_folder_name);
        preset.search = asset_browser_search;
        preset.tag_filter = asset_browser_tag_filter;
        preset.folder_relative_dir = selected_asset_folder_relative_dir;
        preset.favorites_only = asset_browser_favorites_only;
        preset.only_3d = asset_browser_only_3d;
        if (preset.name.empty()) {
            preset.name = makeSmartFolderDefaultName(
                preset.folder_relative_dir,
                preset.search,
                preset.tag_filter,
                preset.favorites_only);
        }

        bool replaced = false;
        int preset_index = -1;
        for (auto& existing : asset_smart_folders) {
            ++preset_index;
            if (existing.name == preset.name) {
                existing = preset;
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            asset_smart_folders.push_back(preset);
            preset_index = static_cast<int>(asset_smart_folders.size()) - 1;
        }
        active_asset_smart_folder_index = preset_index;
        asset_smart_folder_name = preset.name;
        g_ProjectManager.getProjectData().ui_layout_data = serialize();
        g_ProjectManager.markModified();
        addViewportMessage(replaced ? "Collection updated" : "Collection saved", 1.8f, ImVec4(0.4f, 0.9f, 1.0f, 1.0f));
    }

    const AssetRecord* selected_asset = asset_registry.findByRelativeDirectory(selected_asset_relative_dir);
    if (!selected_asset && !asset_registry.getAssets().empty()) {
        selected_asset_relative_dir = asset_registry.getAssets().front().relative_entry_path.generic_string();
        selected_asset = asset_registry.findByRelativeDirectory(selected_asset_relative_dir);
        resetAssetDetailState();
    }
    if (selected_asset && selected_asset_folder_relative_dir.empty()) {
        selected_asset_folder_relative_dir = selected_asset->relative_directory.generic_string();
    }

    ImGui::Separator();

    const float avail_width_total = ImGui::GetContentRegionAvail().x;
    asset_browser_folder_width = (std::max)(180.0f, (std::min)(asset_browser_folder_width, avail_width_total * 0.6f));
    ImGui::BeginChild("AssetBrowserFolders", ImVec2(asset_browser_folder_width, 0), true);
    ImGui::TextDisabled("Collections");
    ImGui::Separator();
    if (ImGui::Selectable("All Assets", active_asset_smart_folder_index < 0)) {
        active_asset_smart_folder_index = -1;
        asset_smart_folder_name.clear();
        g_ProjectManager.getProjectData().ui_layout_data = serialize();
    }
    for (int i = 0; i < static_cast<int>(asset_smart_folders.size()); ++i) {
        const auto& preset = asset_smart_folders[i];
        if (ImGui::Selectable(preset.name.c_str(), active_asset_smart_folder_index == i)) {
            active_asset_smart_folder_index = i;
            asset_smart_folder_name = preset.name;
            asset_browser_search = preset.search;
            asset_browser_tag_filter = preset.tag_filter;
            asset_browser_favorites_only = preset.favorites_only;
            asset_browser_only_3d = preset.only_3d;
            selected_asset_folder_relative_dir = preset.folder_relative_dir;
            selected_asset_relative_dir.clear();
            resetAssetDetailState();
            g_ProjectManager.getProjectData().ui_layout_data = serialize();
        }
        if (ImGui::BeginPopupContextItem(("SmartFolderContext##" + std::to_string(i)).c_str())) {
            if (ImGui::MenuItem("Update From Current Filters")) {
                asset_smart_folders[i].search = asset_browser_search;
                asset_smart_folders[i].tag_filter = asset_browser_tag_filter;
                asset_smart_folders[i].favorites_only = asset_browser_favorites_only;
                asset_smart_folders[i].only_3d = asset_browser_only_3d;
                asset_smart_folders[i].folder_relative_dir = selected_asset_folder_relative_dir;
                asset_smart_folder_name = asset_smart_folders[i].name;
                g_ProjectManager.getProjectData().ui_layout_data = serialize();
                g_ProjectManager.markModified();
            }
            if (ImGui::MenuItem("Delete Collection")) {
                asset_smart_folders.erase(asset_smart_folders.begin() + i);
                if (active_asset_smart_folder_index == i) {
                    active_asset_smart_folder_index = -1;
                } else if (active_asset_smart_folder_index > i) {
                    --active_asset_smart_folder_index;
                }
                g_ProjectManager.getProjectData().ui_layout_data = serialize();
                g_ProjectManager.markModified();
                ImGui::EndPopup();
                break;
            }
            ImGui::EndPopup();
        }
    }
    ImGui::Spacing();
    ImGui::TextDisabled("Folders");
    ImGui::Separator();

    if (root_path.empty()) {
        ImGui::TextDisabled("Asset root not found.");
        ImGui::TextDisabled("You can still use Add Library to attach a custom asset folder.");
    } else {
        std::function<void(const std::filesystem::path&)> drawFolderNode = [&](const std::filesystem::path& directory) {
            std::error_code ec;
            std::vector<std::filesystem::path> child_dirs;
            for (std::filesystem::directory_iterator it(directory, ec), end; it != end; it.increment(ec)) {
                if (ec || !it->is_directory()) {
                    continue;
                }
                child_dirs.push_back(it->path());
            }
            std::sort(child_dirs.begin(), child_dirs.end());

            const std::filesystem::path rel_path = std::filesystem::relative(directory, root_path, ec);
            const std::string rel_string = rel_path.empty() ? std::string() : rel_path.generic_string();
            ImGuiTreeNodeFlags flags = child_dirs.empty() ? ImGuiTreeNodeFlags_Leaf : 0;
            if (selected_asset_folder_relative_dir == rel_string) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }
            if (rel_path.empty()) {
                flags |= ImGuiTreeNodeFlags_DefaultOpen;
            }

            const std::string label = rel_path.empty() ? "assets" : directory.filename().string();
            const bool opened = ImGui::TreeNodeEx((label + "##" + rel_string).c_str(), flags);
            if (ImGui::IsItemClicked()) {
                if (selected_asset_folder_relative_dir != rel_string) {
                    selected_asset_relative_dir.clear();
                    resetAssetDetailState();
                }
                selected_asset_folder_relative_dir = rel_string;
            }

            if (opened) {
                for (const auto& child_dir : child_dirs) {
                    drawFolderNode(child_dir);
                }
                ImGui::TreePop();
            }
        };

        drawFolderNode(root_path);
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::InvisibleButton("##AssetBrowserFolderSplitter", ImVec2(6.0f, ImGui::GetContentRegionAvail().y));
    if (ImGui::IsItemActive()) {
        asset_browser_folder_width += ImGui::GetIO().MouseDelta.x;
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImGui::SameLine();
    ImGui::BeginChild("AssetBrowserExplorer", ImVec2(0, 0), true);
    ImGui::TextDisabled("Folder Contents");
    ImGui::Separator();
    if (asset_library_refresh_in_progress) {
        ImGui::TextDisabled("%s", asset_library_refresh_status.empty() ? "Scanning assets..." : asset_library_refresh_status.c_str());
        ImGui::Separator();
    }

    const std::filesystem::path selected_folder_path =
        selected_asset_folder_relative_dir.empty() ? root_path : (root_path / selected_asset_folder_relative_dir);
    ImGui::TextWrapped("Selected Folder: %s",
        selected_asset_folder_relative_dir.empty() ? "assets" : selected_asset_folder_relative_dir.c_str());
    ImGui::Separator();

    const float explorer_avail_height = ImGui::GetContentRegionAvail().y;
    asset_browser_details_height = (std::max)(140.0f, (std::min)(asset_browser_details_height, explorer_avail_height - 120.0f));
    const float grid_height = (std::max)(120.0f, explorer_avail_height - asset_browser_details_height - 10.0f);
    ImGui::BeginChild("AssetBrowserGrid", ImVec2(0, grid_height), false);
    if (!root_path.empty() && std::filesystem::exists(selected_folder_path)) {
        std::vector<std::filesystem::path> directories;
        std::vector<std::filesystem::path> files;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(selected_folder_path, ec), end; it != end; it.increment(ec)) {
            if (ec) {
                continue;
            }
            if (it->is_directory()) {
                directories.push_back(it->path());
                continue;
            }
            if (it->is_regular_file()) {
                files.push_back(it->path());
            }
        }
        std::sort(directories.begin(), directories.end());
        std::sort(files.begin(), files.end());

        const bool compact_view = (asset_browser_view_mode == 1);
        const bool details_view = (asset_browser_view_mode == 2);
        const float card_width = details_view ? ImGui::GetContentRegionAvail().x - 4.0f : asset_browser_thumbnail_size;
        const float card_height = details_view ? 30.0f : (compact_view ? 76.0f : 110.0f);
        const float avail_width = ImGui::GetContentRegionAvail().x;
        const int columns = details_view ? 1 : (std::max)(1, static_cast<int>(avail_width / (card_width + 12.0f)));
        int index = 0;
        const std::string lowered_search = AssetRegistry::toLowerCopy(asset_browser_search);
        const std::vector<std::string> required_tags = splitTagString(asset_browser_tag_filter);

        for (const auto& directory : directories) {
            const std::string relative_dir = std::filesystem::relative(directory, root_path).generic_string();
            bool folder_matches = false;
            for (const auto& asset : asset_registry.getAssets()) {
                if (asset.relative_directory.generic_string() != relative_dir) {
                    continue;
                }
                if (asset_browser_only_3d && !isPlaceableAssetRecord(asset)) {
                    continue;
                }
                if (assetMatchesFilters(asset, lowered_search, required_tags, asset_browser_favorites_only)) {
                    folder_matches = true;
                    break;
                }
            }
            if (!folder_matches) {
                const std::string lowered_name = AssetRegistry::toLowerCopy(relative_dir);
                if (asset_browser_only_3d || !required_tags.empty() || asset_browser_favorites_only ||
                    (!lowered_search.empty() && lowered_name.find(lowered_search) == std::string::npos)) {
                    continue;
                }
            }

            ImGui::PushID(relative_dir.c_str());
            if (index > 0 && (index % columns) != 0) {
                ImGui::SameLine();
            }

            ImVec2 card_pos = ImGui::GetCursorScreenPos();
            ImGui::BeginGroup();
            ImGui::InvisibleButton("##AssetDirectoryCard", ImVec2(card_width, card_height));
            const bool hovered = ImGui::IsItemHovered();
            const bool selected = (selected_asset_folder_relative_dir == relative_dir);

            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImU32 bg = selected ? IM_COL32(100, 95, 35, 180) : (hovered ? IM_COL32(88, 78, 38, 175) : IM_COL32(60, 55, 36, 160));
            draw_list->AddRectFilled(card_pos, ImVec2(card_pos.x + card_width, card_pos.y + card_height), bg, 8.0f);
            draw_list->AddRect(card_pos, ImVec2(card_pos.x + card_width, card_pos.y + card_height), IM_COL32(255, 230, 160, 42), 8.0f);

            int child_count = 0;
            std::error_code child_ec;
            for (std::filesystem::directory_iterator child_it(directory, child_ec), child_end; child_it != child_end; child_it.increment(child_ec)) {
                if (!child_ec) {
                    ++child_count;
                }
            }
            const std::string count_label = std::to_string(child_count) + " items";
            const std::string directory_name_fit = fitTextToWidth(directory.filename().string(), card_width - 20.0f);
            const std::string relative_dir_fit = fitTextToWidth(relative_dir, card_width - 20.0f);
            if (details_view) {
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 7), IM_COL32(255, 225, 120, 255), "[DIR]");
                draw_list->AddText(ImVec2(card_pos.x + 64, card_pos.y + 7), IM_COL32(255, 255, 255, 255), directory_name_fit.c_str());
                draw_list->AddText(ImVec2(card_pos.x + card_width - 120, card_pos.y + 7), IM_COL32(235, 215, 160, 255), count_label.c_str());
            } else if (compact_view) {
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 10), IM_COL32(255, 225, 120, 255), "FOLDER");
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 30), IM_COL32(255, 255, 255, 255), directory_name_fit.c_str());
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 50), IM_COL32(235, 215, 160, 255), count_label.c_str());
            } else {
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 10), IM_COL32(255, 225, 120, 255), "FOLDER");
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 36), IM_COL32(255, 255, 255, 255), directory_name_fit.c_str());
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 66), IM_COL32(235, 215, 160, 255), count_label.c_str());
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 84), IM_COL32(210, 210, 220, 220), relative_dir_fit.c_str());
            }

            if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                if (selected_asset_folder_relative_dir != relative_dir) {
                    selected_asset_relative_dir.clear();
                    resetAssetDetailState();
                }
                selected_asset_folder_relative_dir = relative_dir;
            }
            if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                if (selected_asset_folder_relative_dir != relative_dir) {
                    selected_asset_relative_dir.clear();
                    resetAssetDetailState();
                }
                selected_asset_folder_relative_dir = relative_dir;
            }

            ImGui::EndGroup();
            ImGui::PopID();
            ++index;
        }

        for (const auto& file : files) {
            const std::string relative_file = std::filesystem::relative(file, root_path).generic_string();
            const std::string ext = AssetRegistry::toLowerCopy(file.extension().string());
            const AssetRecord* file_asset = nullptr;
            for (const auto& asset : asset_registry.getAssets()) {
                if (asset.entry_path == file || asset.preview_path == file || asset.metadata_path == file) {
                    file_asset = &asset;
                    break;
                }
            }

            if (file_asset && file_asset->entry_path != file) {
                continue;
            }

            const bool is_placeable_asset = file_asset && isPlaceableAssetRecord(*file_asset);
            if (asset_browser_only_3d && !is_placeable_asset) {
                continue;
            }
            if (file_asset && !assetMatchesFilters(*file_asset, lowered_search, required_tags, asset_browser_favorites_only)) {
                continue;
            }
            if (!file_asset) {
                const std::string lowered_name = AssetRegistry::toLowerCopy(relative_file);
                if (asset_browser_favorites_only || !required_tags.empty() ||
                    (!lowered_search.empty() && lowered_name.find(lowered_search) == std::string::npos)) {
                    continue;
                }
            }

            ImGui::PushID(relative_file.c_str());
            if (index > 0 && (index % columns) != 0) {
                ImGui::SameLine();
            }

            ImVec2 card_pos = ImGui::GetCursorScreenPos();
            ImGui::BeginGroup();
            ImGui::InvisibleButton("##AssetFileCard", ImVec2(card_width, card_height));
            const bool hovered = ImGui::IsItemHovered();
            const bool selected = file_asset && (selected_asset_relative_dir == file_asset->relative_entry_path.generic_string());

            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImU32 bg = selected ? IM_COL32(30, 120, 110, 180) : (hovered ? IM_COL32(70, 70, 80, 180) : IM_COL32(45, 45, 52, 170));
            draw_list->AddRectFilled(card_pos, ImVec2(card_pos.x + card_width, card_pos.y + card_height), bg, 8.0f);
            draw_list->AddRect(card_pos, ImVec2(card_pos.x + card_width, card_pos.y + card_height), IM_COL32(255, 255, 255, 28), 8.0f);
            const std::string asset_name_fit = fitTextToWidth(file_asset ? file_asset->name : "No asset metadata", card_width - 20.0f);
            const std::string relative_file_fit = fitTextToWidth(relative_file, card_width - 20.0f);
            const char* type_label =
                file_asset && file_asset->asset_kind == "anim_clip" ? "ANIM" :
                file_asset && file_asset->asset_kind == "vdb_sequence" ? "VDB SEQ" :
                file_asset && file_asset->asset_kind == "vdb" ? "VDB" :
                file_asset && file_asset->asset_kind == "model" ? "MODEL" :
                (ext == ".json" ? "META" : "FILE");
            const std::string title_text = fitTextToWidth(
                file.filename().string() + (file_asset && file_asset->favorite ? " *" : ""),
                card_width - (details_view ? 92.0f : 20.0f));
            SDL_Texture* thumb_texture = nullptr;
            int thumb_w = 0;
            int thumb_h = 0;
            const bool has_thumb = file_asset && file_asset->has_preview &&
                ensureAssetBrowserThumbnailTexture(ctx, file_asset->preview_path, thumb_texture, thumb_w, thumb_h);
            (void)thumb_w;
            (void)thumb_h;

            if (details_view) {
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 7), IM_COL32(220, 220, 230, 255), type_label);
                draw_list->AddText(ImVec2(card_pos.x + 72, card_pos.y + 7), IM_COL32(255, 255, 255, 255), title_text.c_str());
                if (file_asset) {
                    const std::string details_name_fit = fitTextToWidth(file_asset->name, 220.0f);
                    draw_list->AddText(ImVec2(card_pos.x + card_width - 240, card_pos.y + 7), IM_COL32(120, 230, 210, 255), details_name_fit.c_str());
                }
            } else if (compact_view) {
                if (has_thumb && thumb_texture) {
                    const float thumb_size = 36.0f;
                    const ImVec2 thumb_min(card_pos.x + card_width - thumb_size - 8.0f, card_pos.y + 8.0f);
                    const ImVec2 thumb_max(card_pos.x + card_width - 8.0f, card_pos.y + 8.0f + thumb_size);
                    draw_list->AddImage((ImTextureID)thumb_texture, thumb_min, thumb_max);
                    draw_list->AddRect(thumb_min, thumb_max, IM_COL32(255, 255, 255, 45), 4.0f);
                }
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 10), IM_COL32(220, 220, 230, 255), type_label);
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 30), IM_COL32(255, 255, 255, 255), title_text.c_str());
                draw_list->AddText(
                    ImVec2(card_pos.x + 10, card_pos.y + 50),
                    file_asset ? IM_COL32(120, 230, 210, 255) : IM_COL32(180, 180, 190, 255),
                    asset_name_fit.c_str());
            } else {
                if (has_thumb && thumb_texture) {
                    const float preview_height = 48.0f;
                    const ImVec2 thumb_min(card_pos.x + card_width - 58.0f, card_pos.y + 10.0f);
                    const ImVec2 thumb_max(card_pos.x + card_width - 10.0f, card_pos.y + 10.0f + preview_height);
                    draw_list->AddImage((ImTextureID)thumb_texture, thumb_min, thumb_max);
                    draw_list->AddRect(thumb_min, thumb_max, IM_COL32(255, 255, 255, 45), 4.0f);
                }
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 10), IM_COL32(220, 220, 230, 255), type_label);
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 34), IM_COL32(255, 255, 255, 255), title_text.c_str());

                if (file_asset) {
                    draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 62), IM_COL32(120, 230, 210, 255), asset_name_fit.c_str());
                } else {
                    draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 62), IM_COL32(180, 180, 190, 255), "No asset metadata");
                }
                draw_list->AddText(ImVec2(card_pos.x + 10, card_pos.y + 84), IM_COL32(180, 180, 190, 220), relative_file_fit.c_str());
            }

            if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && file_asset && is_placeable_asset) {
                const std::string new_selection = file_asset->relative_entry_path.generic_string();
                if (selected_asset_relative_dir != new_selection) {
                    resetAssetDetailState();
                }
                selected_asset_relative_dir = new_selection;
                selected_asset = file_asset;
            } else if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && file_asset) {
                const std::string new_selection = file_asset->relative_entry_path.generic_string();
                if (selected_asset_relative_dir != new_selection) {
                    resetAssetDetailState();
                }
                selected_asset_relative_dir = new_selection;
                selected_asset = file_asset;
            } else if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !file_asset) {
                if (!selected_asset_relative_dir.empty()) {
                    selected_asset_relative_dir.clear();
                    selected_asset = nullptr;
                    resetAssetDetailState();
                }
            }

            if (file_asset && is_placeable_asset && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                const std::string payload_key = file_asset->relative_entry_path.generic_string();
                if (asset_drag_bbox_cache.find(payload_key) == asset_drag_bbox_cache.end()) {
                    Vec3 preview_min, preview_max;
                    if (computeAssetPreviewBounds(*file_asset, preview_min, preview_max)) {
                        asset_drag_bbox_cache.emplace(payload_key, std::make_pair(preview_min, preview_max));
                    }
                }
                ImGui::SetDragDropPayload("ASSET_BROWSER_ENTRY", payload_key.c_str(), payload_key.size() + 1);
                ImGui::TextUnformatted(file_asset->name.c_str());
                ImGui::TextDisabled("%s", file_asset->entry_path.filename().string().c_str());
                ImGui::EndDragDropSource();
            }
            if (file_asset && ImGui::BeginPopupContextItem("AssetCardContext")) {
                if (ImGui::MenuItem(file_asset->favorite ? "Unfavorite" : "Add Favorite")) {
                    if (saveAssetMetadataChanges(asset_registry, *file_asset, !file_asset->favorite, file_asset->tags)) {
                        asset_registry.refresh();
                        selected_asset = asset_registry.findByRelativeDirectory(selected_asset_relative_dir);
                        g_ProjectManager.markModified();
                    }
                }
                if (ImGui::MenuItem("Copy Asset Path")) {
                    SDL_SetClipboardText(file_asset->entry_path.string().c_str());
                    addViewportMessage("Asset path copied", 1.8f, ImVec4(0.5f, 1.0f, 0.6f, 1.0f));
                }
                if (ImGui::MenuItem("Regenerate Metadata")) {
                    if (saveAssetMetadataChanges(asset_registry, *file_asset, file_asset->favorite, file_asset->tags)) {
                        asset_registry.refresh();
                        selected_asset = asset_registry.findByRelativeDirectory(selected_asset_relative_dir);
                        g_ProjectManager.markModified();
                        addViewportMessage("Metadata regenerated", 1.8f, ImVec4(0.4f, 0.9f, 1.0f, 1.0f));
                    }
                }
                if (ImGui::MenuItem("Clear Preview Cache")) {
                    releaseSelectedAssetPreviewTexture();
                    releaseAssetBrowserThumbnailTextures();
                }
                ImGui::EndPopup();
            }
            ImGui::EndGroup();
            ImGui::PopID();
            ++index;
        }

        if (index == 0) {
            ImGui::TextDisabled("This folder has no matching files.");
        }
    } else {
        ImGui::TextDisabled("Select a folder from the left tree.");
    }
    ImGui::EndChild();

    ImGui::InvisibleButton("##AssetBrowserDetailSplitter", ImVec2(ImGui::GetContentRegionAvail().x, 6.0f));
    if (ImGui::IsItemActive()) {
        asset_browser_details_height -= ImGui::GetIO().MouseDelta.y;
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    }

    ImGui::Separator();
    ImGui::TextDisabled("Selected Asset Details");
    ImGui::Separator();
    ImGui::BeginChild("AssetBrowserDetails", ImVec2(0, 0), false);

    if (!selected_asset) {
        releaseSelectedAssetPreviewTexture();
        ImGui::TextDisabled("Pick a placeable asset in the current folder to inspect its metadata.");
    } else {
        if (selected_asset_tags_edit_target != selected_asset->relative_entry_path.generic_string()) {
            selected_asset_tags_edit_target = selected_asset->relative_entry_path.generic_string();
            selected_asset_tags_edit = joinTags(selected_asset->tags);
        }

        nlohmann::json asset_metadata_json;
        bool has_asset_metadata_json = false;
        if (selected_asset->has_metadata && !selected_asset->metadata_path.empty()) {
            try {
                std::ifstream metadata_file(selected_asset->metadata_path);
                if (metadata_file.is_open()) {
                    metadata_file >> asset_metadata_json;
                    has_asset_metadata_json = true;
                }
            } catch (...) {
                has_asset_metadata_json = false;
            }
        }

        ImGui::Text("%s", selected_asset->name.c_str());
        ImGui::TextDisabled("%s", selected_asset->id.c_str());
        if (selected_asset->favorite) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.35f, 1.0f), "Favorite");
        }
        if (selected_asset->has_preview) {
            int preview_width = 0;
            int preview_height = 0;
            if (ensureSelectedAssetPreviewTexture(ctx, selected_asset->preview_path, preview_width, preview_height)) {
                ImGui::Spacing();
                ImGui::TextDisabled("Preview");

                float draw_width = (std::min)(ImGui::GetContentRegionAvail().x, 320.0f);
                float draw_height = draw_width;
                if (preview_width > 0 && preview_height > 0) {
                    const float aspect = static_cast<float>(preview_width) / static_cast<float>(preview_height);
                    draw_height = draw_width / (aspect > 0.001f ? aspect : 1.0f);
                    if (draw_height > 220.0f) {
                        const float scale = 220.0f / draw_height;
                        draw_width *= scale;
                        draw_height *= scale;
                    }
                }

                ImGui::Image((ImTextureID)selected_asset_preview_texture, ImVec2(draw_width, draw_height));
            } else {
                ImGui::Spacing();
                ImGui::TextDisabled("Preview");
                ImGui::TextDisabled("Preview image could not be loaded.");
            }
        } else {
            releaseSelectedAssetPreviewTexture();
        }

        ImGui::BulletText("Folder: %s", selected_asset->relative_directory.generic_string().c_str());
        ImGui::BulletText("Entry: %s", selected_asset->entry_path.filename().string().c_str());
        ImGui::BulletText("Kind: %s", selected_asset->asset_kind.empty() ? "asset" : selected_asset->asset_kind.c_str());
        ImGui::BulletText("Preview: %s", selected_asset->has_preview ? selected_asset->preview_path.filename().string().c_str() : "missing");
        ImGui::BulletText("Metadata: %s", selected_asset->has_metadata ? selected_asset->metadata_path.filename().string().c_str() : "auto-derived");
        ImGui::BulletText("Tags: %s", selected_asset->tags.empty() ? "none" : joinTags(selected_asset->tags).c_str());
        if (selected_asset->is_sequence) {
            ImGui::BulletText("Sequence: %d - %d", selected_asset->sequence_start_frame, selected_asset->sequence_end_frame);
        }
        if (selected_asset->animation_clip_count > 0) {
            ImGui::BulletText("Animation Clips: %llu", static_cast<unsigned long long>(selected_asset->animation_clip_count));
        }
        if (selected_asset->asset_kind == "anim_clip") {
            ImGui::BulletText("Clip Mode: %s", selected_asset->clip_mode.empty() ? "skeletal" : selected_asset->clip_mode.c_str());
            ImGui::BulletText("Binding: %s", selected_asset->animation_binding.empty() ? "selected-model" : selected_asset->animation_binding.c_str());
        }
        if (selected_asset->asset_kind == "vdb" || selected_asset->asset_kind == "vdb_sequence") {
            const std::string grid_list = selected_asset->vdb_grids.empty() ? "none" : [&]() {
                std::string joined;
                for (size_t i = 0; i < selected_asset->vdb_grids.size(); ++i) {
                    if (i > 0) {
                        joined += ", ";
                    }
                    joined += selected_asset->vdb_grids[i];
                }
                return joined;
            }();
            ImGui::BulletText("Grids: %s", grid_list.c_str());
            ImGui::BulletText("Primary Grid: %s", selected_asset->vdb_primary_grid.empty() ? "density" : selected_asset->vdb_primary_grid.c_str());
            ImGui::BulletText("Voxel Size: %.5f", selected_asset->vdb_voxel_size);
            ImGui::BulletText("Preset: %s", selected_asset->vdb_shader_preset.empty() ? "auto" : selected_asset->vdb_shader_preset.c_str());
        }

        if (has_asset_metadata_json && asset_metadata_json.contains("metrics") && asset_metadata_json["metrics"].is_object()) {
            const auto& metrics = asset_metadata_json["metrics"];
            ImGui::Spacing();
            ImGui::TextDisabled("Cost / Metrics");
            ImGui::BulletText("Triangles: %llu", static_cast<unsigned long long>(metrics.value("triangleCount", 0ull)));
            ImGui::BulletText("Meshes: %llu", static_cast<unsigned long long>(metrics.value("meshCount", 0ull)));
            ImGui::BulletText("Materials: %llu", static_cast<unsigned long long>(metrics.value("materialCount", 0ull)));
            ImGui::BulletText("Texture Refs: %llu", static_cast<unsigned long long>(metrics.value("textureReferenceCount", 0ull)));
            ImGui::BulletText("Anim Clips: %llu", static_cast<unsigned long long>(metrics.value("animationClipCount", 0ull)));
        }

        if (has_asset_metadata_json && asset_metadata_json.contains("dimensions") && asset_metadata_json["dimensions"].is_object()) {
            const auto& dimensions = asset_metadata_json["dimensions"];
            ImGui::Spacing();
            ImGui::TextDisabled("Dimensions");
            ImGui::BulletText(
                "Size: %.2f x %.2f x %.2f",
                dimensions.value("width", 0.0),
                dimensions.value("height", 0.0),
                dimensions.value("depth", 0.0));
        }

        if (selected_asset->asset_kind == "vdb" || selected_asset->asset_kind == "vdb_sequence") {
            ImGui::Spacing();
            ImGui::TextDisabled("Volume");
            if (selected_asset->vdb_has_bounds) {
                ImGui::BulletText(
                    "Bounds Min: %.2f, %.2f, %.2f",
                    selected_asset->vdb_bounds_min_x,
                    selected_asset->vdb_bounds_min_y,
                    selected_asset->vdb_bounds_min_z);
                ImGui::BulletText(
                    "Bounds Max: %.2f, %.2f, %.2f",
                    selected_asset->vdb_bounds_max_x,
                    selected_asset->vdb_bounds_max_y,
                    selected_asset->vdb_bounds_max_z);
            }
            ImGui::BulletText("Hint Fire: %s", selected_asset->vdb_is_fire ? "yes" : "no");
            ImGui::BulletText("Hint Smoke: %s", selected_asset->vdb_is_smoke ? "yes" : "no");
            ImGui::BulletText("Velocity Grid: %s", selected_asset->vdb_has_velocity ? "yes" : "no");
            ImGui::BulletText("Density Mult: %.2f", selected_asset->vdb_density_multiplier);
            ImGui::BulletText("Temp Scale: %.2f", selected_asset->vdb_temperature_scale);
            ImGui::BulletText("Emission Intensity: %.2f", selected_asset->vdb_emission_intensity);
            if (selected_asset->is_sequence) {
                ImGui::BulletText("FPS: %.2f", selected_asset->sequence_fps);
            }
        }

        if (has_asset_metadata_json && asset_metadata_json.contains("placementPivot") && asset_metadata_json["placementPivot"].is_object()) {
            const auto& pivot = asset_metadata_json["placementPivot"];
            ImGui::Spacing();
            ImGui::TextDisabled("Placement");
            ImGui::BulletText(
                "Pivot: %.2f, %.2f, %.2f",
                pivot.value("x", 0.0),
                pivot.value("y", 0.0),
                pivot.value("z", 0.0));
            const std::string pivot_mode = pivot.value("mode", std::string());
            if (!pivot_mode.empty()) {
                ImGui::BulletText("Pivot Mode: %s", pivot_mode.c_str());
            }
        }

        if (has_asset_metadata_json) {
            const std::string source_tool = asset_metadata_json.value("sourceTool", std::string());
            if (!source_tool.empty()) {
                ImGui::Spacing();
                ImGui::TextDisabled("Pipeline");
                ImGui::BulletText("Source Tool: %s", source_tool.c_str());
            }
        }

        if (!selected_asset->description.empty()) {
            ImGui::Spacing();
            ImGui::TextWrapped("%s", selected_asset->description.c_str());
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextDisabled("Library");
        bool favorite_value = selected_asset->favorite;
        if (ImGui::Checkbox("Favorite Asset", &favorite_value)) {
            if (saveAssetMetadataChanges(asset_registry, *selected_asset, favorite_value, splitTagString(selected_asset_tags_edit))) {
                asset_registry.refresh();
                selected_asset = asset_registry.findByRelativeDirectory(selected_asset_relative_dir);
                g_ProjectManager.markModified();
            }
        }
        std::array<char, 512> tags_buffer{};
        std::snprintf(tags_buffer.data(), tags_buffer.size(), "%s", selected_asset_tags_edit.c_str());
        if (ImGui::InputTextWithHint("Asset Tags", "comma separated tags", tags_buffer.data(), tags_buffer.size())) {
            selected_asset_tags_edit = tags_buffer.data();
        }
        if (ImGui::BeginPopupContextItem("SelectedAssetDetailsContext")) {
            if (ImGui::MenuItem("Copy Asset Path")) {
                SDL_SetClipboardText(selected_asset->entry_path.string().c_str());
                addViewportMessage("Asset path copied", 1.8f, ImVec4(0.5f, 1.0f, 0.6f, 1.0f));
            }
            if (ImGui::MenuItem("Regenerate Metadata")) {
                if (saveAssetMetadataChanges(asset_registry, *selected_asset, favorite_value, splitTagString(selected_asset_tags_edit))) {
                    asset_registry.refresh();
                    selected_asset = asset_registry.findByRelativeDirectory(selected_asset_relative_dir);
                    g_ProjectManager.markModified();
                }
            }
            ImGui::EndPopup();
        }
        if (ImGui::Button("Save Asset Metadata", ImVec2(190, 0))) {
            if (saveAssetMetadataChanges(asset_registry, *selected_asset, favorite_value, splitTagString(selected_asset_tags_edit))) {
                asset_registry.refresh();
                selected_asset = asset_registry.findByRelativeDirectory(selected_asset_relative_dir);
                if (selected_asset) {
                    selected_asset_tags_edit_target = selected_asset->relative_entry_path.generic_string();
                    selected_asset_tags_edit = joinTags(selected_asset->tags);
                }
                g_ProjectManager.markModified();
                addViewportMessage("Asset metadata saved", 1.8f, ImVec4(0.4f, 0.9f, 1.0f, 1.0f));
            } else {
                addViewportMessage("Asset metadata save failed", 2.4f, ImVec4(1.0f, 0.5f, 0.35f, 1.0f));
            }
        }

        ImGui::Spacing();
        const char* append_button_label = selected_asset->asset_kind == "anim_clip" ? "Import Clip" : "Append To Scene";
        if (ImGui::Button(append_button_label, ImVec2(160, 0)) && isPlaceableAssetRecord(*selected_asset)) {
            appendAssetToScene(ctx, selected_asset->entry_path, selected_asset->name);
        }
        ImGui::SameLine();
        if (ImGui::Button("Generate Metadata", ImVec2(190, 0))) {
            int generated_count = 0;
            for (const auto& asset : asset_registry.getAssets()) {
                if (asset.directory_path == selected_folder_path) {
                    if (asset_registry.writeMetadataStub(asset)) {
                        ++generated_count;
                    }
                }
            }

            releaseAssetBrowserThumbnailTextures();
            asset_registry.refresh();
            selected_asset = asset_registry.findByRelativeDirectory(selected_asset_relative_dir);

            if (generated_count > 0) {
                addViewportMessage(
                    std::to_string(generated_count) + " metadata file(s) created in folder",
                    2.5f,
                    ImVec4(0.3f, 0.8f, 1.0f, 1.0f));
            } else {
                addViewportMessage("No metadata files were created for this folder", 3.0f, ImVec4(1.0f, 0.7f, 0.2f, 1.0f));
            }
        }
    }

    ImGui::EndChild();
    ImGui::EndChild();
    if (!embedded) {
        ImGui::End();
    }
}

SceneUI::~SceneUI() {
    releaseSelectedAssetPreviewTexture();
}

bool SceneUI::ensureSelectedAssetPreviewTexture(UIContext& ctx, const std::filesystem::path& preview_path, int& width, int& height)
{
    extern SDL_Renderer* renderer;

    width = 0;
    height = 0;

    SDL_Renderer* active_renderer = ctx.renderer.sdlRenderer ? ctx.renderer.sdlRenderer : renderer;

    if (preview_path.empty() || !std::filesystem::exists(preview_path) || !active_renderer) {
        releaseSelectedAssetPreviewTexture();
        return false;
    }

    if (selected_asset_preview_texture &&
        selected_asset_preview_texture_path == preview_path) {
        width = selected_asset_preview_texture_width;
        height = selected_asset_preview_texture_height;
        return true;
    }

    releaseSelectedAssetPreviewTexture();
    if (!loadPreviewTextureFromPath(active_renderer, preview_path, selected_asset_preview_texture, width, height)) {
        return false;
    }

    selected_asset_preview_texture_path = preview_path;
    selected_asset_preview_texture_width = width;
    selected_asset_preview_texture_height = height;
    return true;
}

bool SceneUI::ensureAssetBrowserThumbnailTexture(UIContext& ctx, const std::filesystem::path& preview_path, SDL_Texture*& out_texture, int& width, int& height)
{
    extern SDL_Renderer* renderer;

    out_texture = nullptr;
    width = 0;
    height = 0;

    SDL_Renderer* active_renderer = ctx.renderer.sdlRenderer ? ctx.renderer.sdlRenderer : renderer;
    if (preview_path.empty() || !active_renderer || !std::filesystem::exists(preview_path)) {
        return false;
    }

    const std::string cache_key = normalizeAbsolutePath(preview_path).generic_string();
    auto found = asset_browser_thumbnail_cache.find(cache_key);
    if (found != asset_browser_thumbnail_cache.end() && found->second.texture) {
        found->second.last_used = ++asset_browser_thumbnail_use_counter;
        out_texture = found->second.texture;
        width = found->second.width;
        height = found->second.height;
        return true;
    }

    SDL_Texture* preview_texture = nullptr;
    if (!loadPreviewTextureFromPath(active_renderer, preview_path, preview_texture, width, height)) {
        return false;
    }

    AssetThumbnailCacheEntry entry;
    entry.texture = preview_texture;
    entry.width = width;
    entry.height = height;
    entry.last_used = ++asset_browser_thumbnail_use_counter;
    asset_browser_thumbnail_cache[cache_key] = entry;

    constexpr size_t kMaxThumbnailCacheEntries = 48;
    if (asset_browser_thumbnail_cache.size() > kMaxThumbnailCacheEntries) {
        auto lru_it = asset_browser_thumbnail_cache.end();
        for (auto it = asset_browser_thumbnail_cache.begin(); it != asset_browser_thumbnail_cache.end(); ++it) {
            if (lru_it == asset_browser_thumbnail_cache.end() || it->second.last_used < lru_it->second.last_used) {
                lru_it = it;
            }
        }
        if (lru_it != asset_browser_thumbnail_cache.end() && lru_it->second.texture) {
            SDL_DestroyTexture(lru_it->second.texture);
            asset_browser_thumbnail_cache.erase(lru_it);
        }
    }

    out_texture = preview_texture;
    return true;
}

void SceneUI::releaseSelectedAssetPreviewTexture()
{
    if (selected_asset_preview_texture) {
        SDL_DestroyTexture(selected_asset_preview_texture);
        selected_asset_preview_texture = nullptr;
    }

    selected_asset_preview_texture_path.clear();
    selected_asset_preview_texture_width = 0;
    selected_asset_preview_texture_height = 0;
}

void SceneUI::releaseAssetBrowserThumbnailTextures()
{
    for (auto& [key, entry] : asset_browser_thumbnail_cache) {
        if (entry.texture) {
            SDL_DestroyTexture(entry.texture);
            entry.texture = nullptr;
        }
    }
    asset_browser_thumbnail_cache.clear();
    asset_browser_thumbnail_use_counter = 0;
}

void SceneUI::drawControlsContent()
{
     UIWidgets::ColoredHeader("Viewport & Camera Navigation", ImVec4(1.0f, 0.9f, 0.4f, 1.0f));
     UIWidgets::Divider();
     
     ImGui::BulletText("Rotate: Middle Mouse Drag");
     ImGui::BulletText("Pan: Shift + Middle Mouse Drag");
     ImGui::BulletText("Zoom: Mouse Wheel OR Ctrl + Middle Mouse Drag");
     ImGui::BulletText("Move Forward/Back: Arrow Up / Arrow Down");
     ImGui::BulletText("Move Left/Right: Arrow Left / Arrow Right");
     ImGui::BulletText("Move Up/Down: PageUp / PageDown");
     
     ImGui::Spacing();
     UIWidgets::ColoredHeader("General Shortcuts & Undo System", ImVec4(0.6f, 0.8f, 1.0f, 1.0f));
     UIWidgets::Divider();
     ImGui::BulletText("Toggle Properties/Panels: N");
     ImGui::BulletText("Toggle Render Window: F12");
     ImGui::BulletText("Toggle Help Window (This): F1");
     ImGui::BulletText("Save Project: Ctrl + S"); 
     ImGui::BulletText("Undo (Object/Light transform & delete): Ctrl + Z");
     ImGui::BulletText("Redo: Ctrl + Y (or Ctrl + Shift + Z)");
     ImGui::BulletText("Delete Selected (O(1) perf): Delete or X");
     ImGui::BulletText("Duplicate Selected: Shift + D");
     
     ImGui::Spacing();
     UIWidgets::ColoredHeader("Advanced Selection", ImVec4(0.8f, 0.4f, 1.0f, 1.0f));
     UIWidgets::Divider();
     ImGui::BulletText("Select Object / Light: Left Click in viewport");
     ImGui::BulletText("Multi-Select (Add/Remove): Ctrl + Left Click");
     ImGui::BulletText("Box Selection: Right Mouse Drag (Selects multiple)");
     ImGui::TextDisabled("  * You can select both lights and physical objects simultaneously.");

     ImGui::Spacing();
     UIWidgets::ColoredHeader("Transform Gizmo & Idle Preview", ImVec4(1.0f, 0.6f, 0.4f, 1.0f));
     UIWidgets::Divider();
     ImGui::BulletText("Move Mode: G (Translate)");
     ImGui::BulletText("Rotate Mode: R");
     ImGui::BulletText("Scale Mode: S");
     ImGui::BulletText("Switch Mode: W (Cycle through modes)");
     ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Idle Preview Feature:");
     ImGui::TextDisabled("  * While dragging a gizmo handle, pause your mouse for 0.3s");
     ImGui::TextDisabled("  * The engine will render a quick preview of your new position.");
     ImGui::TextDisabled("  * Release the mouse to finalize and update the BVH.");

     ImGui::Spacing();
     UIWidgets::ColoredHeader("Interface Guide", ImVec4(0.4f, 1.0f, 0.6f, 1.0f));
     UIWidgets::Divider();
     
     if (ImGui::CollapsingHeader("Render Settings")) {
         ImGui::BulletText("Quality Preset: Quick setup for Preview (Fast) vs Cinematic (High Quality).");
         ImGui::BulletText("Use OptiX: Enable NVIDIA GPU acceleration (Requires RTX card).");
         ImGui::BulletText("Use Denoiser: Clean up noise using AI (OIDN).");
         ImGui::BulletText("Start/Stop: Control the rendering process.");
         ImGui::TextDisabled("  * Pausing allows you to resume later.");
         ImGui::TextDisabled("  * Stopping resets the progress.");
     }
     
     if (ImGui::CollapsingHeader("Sampling")) {
         ImGui::BulletText("Adaptive Sampling: Focuses rays on noisy areas for efficiency.");
         ImGui::BulletText("Max Samples: The target quality level (higher = less noise).");
         ImGui::BulletText("Max Bounces: Light reflection depth (higher = more realistic light).");
     }
     
     if (ImGui::CollapsingHeader("Timeline & Animation")) {
         ImGui::BulletText("Play/Pause: Preview animation movement.");
         ImGui::BulletText("Scrubbing: Drag the timeline handle to jump to frames.");
         ImGui::BulletText("Render Animation: Renders the full sequence to the output folder.");
     }
     
     if (ImGui::CollapsingHeader("Physical Camera System")) {
         ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.7f, 1.0f), "Camera Body Selection:");
         ImGui::BulletText("Select real camera bodies (Canon, Sony, Pentax, RED, ARRI...)");
         ImGui::BulletText("Each body has a specific sensor size and crop factor.");
         ImGui::BulletText("APS-C sensors have 1.5x-1.6x crop, narrowing FOV.");
         ImGui::BulletText("Medium Format has 0.79x crop, widening FOV.");
         ImGui::BulletText("Custom mode: Manual crop factor slider (0.5x - 2.5x).");
         
         ImGui::Separator();
         ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.5f, 1.0f), "Lens Selection:");
         ImGui::BulletText("Professional lenses from Canon, Sony, Sigma, Zeiss, Pentax...");
         ImGui::BulletText("Each lens provides: Focal length, Max aperture, Blade count.");
         ImGui::BulletText("Blade count affects bokeh shape (more = rounder).");
         ImGui::BulletText("Effective focal = Base focal x Crop factor.");
         
         ImGui::Separator();
         ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.9f, 1.0f), "Depth of Field:");
         ImGui::BulletText("F-Stop presets: f/1.4 (blur) to f/16 (sharp).");
         ImGui::BulletText("Focus Distance: Distance to the sharpest plane.");
         ImGui::BulletText("Focus to Selection: Auto-set focus to selected object.");
         
         ImGui::Separator();
         ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Keyframing:");
         ImGui::BulletText("Click [K] buttons to insert keyframes for properties.");
         ImGui::BulletText("Key All: Insert keyframe for all camera properties.");
     }
     
     if (ImGui::CollapsingHeader("Lights Panel")) {
         ImGui::BulletText("Lists all light sources in the scene.");
         ImGui::BulletText("Allows adjusting Intensity (Brightness) and Color.");
         ImGui::BulletText("Supports Point, Directional, Area, and Spot lights.");
     }
     
     if (ImGui::CollapsingHeader("World Panel")) {
         ImGui::BulletText("Sky Model: Switch between Solid Color, HDRI, or Raytrophi Spectral Sky.");
         ImGui::BulletText("Raytrophi Spectral Sky: Realistic day/night cycle with Sun & Moon controls.");
         ImGui::BulletText("Atmosphere: Adjust Air, Dust, and Ozone density for realistic scattering.");
         ImGui::BulletText("Volumetric Clouds: Enable 3D clouds with various weather presets.");
         ImGui::BulletText("HDRI: Load external HDR enviromaps for realistic reflections.");
         ImGui::BulletText("Light Sync: Sync Sun position with the main directional light.");
     }
     
     if (ImGui::CollapsingHeader("Post-FX Panel")) {
         ImGui::BulletText("Main: Adjust Gamma, Exposure, Saturation, and Color Temperature.");
         ImGui::BulletText("Tonemapping: Choose from AGX, ACES, Filmic, etc.");
         ImGui::BulletText("Effects: Add Vignette (dark corners) to frame the image.");
         ImGui::BulletText("Apply/Reset: Post-processing is applied AFTER rendering.");
     }

     if (ImGui::CollapsingHeader("System Panel")) {
         ImGui::BulletText("Theme: Switch between Dark/Light/Classic themes.");
         ImGui::BulletText("Resolution: Set render resolution (Presets like 720p, 1080p, 4K).");
         ImGui::BulletText("Animation Panel: Toggle visibility of the timeline.");
     }
}


void SceneUI::rebuildMeshCache(const std::vector<std::shared_ptr<Hittable>>& objects) {
    SCENE_LOG_INFO("Rebuilding selection cache for " + std::to_string(objects.size()) + " objects...");
    mesh_cache.clear();
    mesh_ui_cache.clear();
    this->tri_to_index.clear(); // Clear the lookup map
    bbox_cache.clear();  
    material_slots_cache.clear();
    
    // Hint for potential large scenes (1.2M objects!)
    tri_to_index.reserve(objects.size());

    for (size_t i = 0; i < objects.size(); ++i) {
        auto tri = std::dynamic_pointer_cast<Triangle>(objects[i]);
        if (tri) {
            std::string name = tri->nodeName.empty() ? "Unnamed" : tri->nodeName;
            mesh_cache[name].push_back({(int)i, tri});
            this->tri_to_index[tri.get()] = (int)i; // Store const pointer to index mapping
        }
    }
    
    // Transfer to sequential vector for ImGui Clipper AND calculate bounding boxes AND material slots
    mesh_ui_cache.reserve(mesh_cache.size());
    for (auto& kv : mesh_cache) {
        mesh_ui_cache.push_back(kv);
        
        // Calculate LOCAL bounding box from ORIGINAL vertices (not transformed!)
        // This allows us to properly apply the transform matrix when drawing
        Vec3 bb_min(1e10f, 1e10f, 1e10f);
        Vec3 bb_max(-1e10f, -1e10f, -1e10f);
        
        // Collect unique material IDs for this object
        std::vector<uint16_t> mat_ids;
        
        for (auto& pair : kv.second) {
            auto& tri = pair.second;
            // Use ORIGINAL vertices (local space) - not getV0() which returns transformed!
            Vec3 v0 = tri->getOriginalVertexPosition(0);
            Vec3 v1 = tri->getOriginalVertexPosition(1);
            Vec3 v2 = tri->getOriginalVertexPosition(2);
            
            bb_min.x = fminf(bb_min.x, fminf(v0.x, fminf(v1.x, v2.x)));
            bb_min.y = fminf(bb_min.y, fminf(v0.y, fminf(v1.y, v2.y)));
            bb_min.z = fminf(bb_min.z, fminf(v0.z, fminf(v1.z, v2.z)));
            bb_max.x = fmaxf(bb_max.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
            bb_max.y = fmaxf(bb_max.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
            bb_max.z = fmaxf(bb_max.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
            
            // Collect material ID (check for duplicates - usually few materials)
            uint16_t mid = tri->getMaterialID();
            bool found = false;
            for (uint16_t existing : mat_ids) {
                if (existing == mid) { found = true; break; }
            }
            if (!found) mat_ids.push_back(mid);
        }
        
        bbox_cache[kv.first] = {bb_min, bb_max};
        material_slots_cache[kv.first] = std::move(mat_ids);
    }
    
    mesh_cache_valid = true;
    last_scene_obj_count = objects.size();
}

void SceneUI::invalidateCache() { 
    mesh_cache_valid = false; 
    mesh_cache.clear();
    mesh_ui_cache.clear();
    bbox_cache.clear();
    material_slots_cache.clear();
    SCENE_LOG_INFO("Selection cache fully cleared and invalidated");
}

// Update bounding box for a specific object (after transform)
// NOTE: Since we now store LOCAL bbox (from original vertices), this may not need
// to be called unless the mesh geometry itself changes. Transform is applied at draw time.
void SceneUI::updateBBoxCache(const std::string& objectName) {
    auto it = mesh_cache.find(objectName);
    if (it == mesh_cache.end()) return;
    
    Vec3 bb_min(1e10f, 1e10f, 1e10f);
    Vec3 bb_max(-1e10f, -1e10f, -1e10f);
    
    for (auto& pair : it->second) {
        auto& tri = pair.second;
        // Use ORIGINAL vertices (local space) for consistency with rebuildMeshCache
        Vec3 v0 = tri->getOriginalVertexPosition(0);
        Vec3 v1 = tri->getOriginalVertexPosition(1);
        Vec3 v2 = tri->getOriginalVertexPosition(2);
        
        bb_min.x = fminf(bb_min.x, fminf(v0.x, fminf(v1.x, v2.x)));
        bb_min.y = fminf(bb_min.y, fminf(v0.y, fminf(v1.y, v2.y)));
        bb_min.z = fminf(bb_min.z, fminf(v0.z, fminf(v1.z, v2.z)));
        bb_max.x = fmaxf(bb_max.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
        bb_max.y = fmaxf(bb_max.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
        bb_max.z = fmaxf(bb_max.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
    }
    
    bbox_cache[objectName] = {bb_min, bb_max};
}

// Lazy CPU Sync - called before mouse picking to ensure vertices are up to date
// This is much more efficient than updating on gizmo release because:
// 1. Gizmo release is instant (no freeze)
// 2. Sync only happens when user actually tries to pick something
// 3. If user moves object multiple times without picking, we only sync once
void SceneUI::ensureCPUSyncForPicking(UIContext& ctx) {
    if (objects_needing_cpu_sync.empty()) return;
    
    size_t synced_count = 0;
    
    // Update all pending objects - apply current transforms to vertices
    for (const auto& name : objects_needing_cpu_sync) {
        auto it = mesh_cache.find(name);
        if (it != mesh_cache.end() && !it->second.empty()) {
            for (auto& pair : it->second) {
                pair.second->updateTransformedVertices();
                synced_count++;
            }
        }
    }
    
    size_t count = objects_needing_cpu_sync.size();
    objects_needing_cpu_sync.clear();
    
    if (synced_count > 0) {
        // [FIX] Force rebuild BVH so picking works with new vertex positions
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
       // SCENE_LOG_INFO("Lazy CPU sync: updated " + std::to_string(synced_count) + " triangles for " + std::to_string(count) + " objects");
    }
}

// Global flag for Render Window visibility
bool show_render_window = false;

void SceneUI::drawRenderWindow(UIContext& ctx) {
    if (!show_render_window) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    
    // Auto-Stop Logic
    // Auto-Stop Logic (ONLY for Single Frame Render)
    // For animation, the render thread manages the loop and frame progression.
    int current_samples = ctx.renderer.getCPUAccumulatedSamples();
    int target_samples = ctx.render_settings.final_render_samples;

    if (!ctx.is_animation_mode && ctx.render_settings.is_final_render_mode && current_samples >= target_samples) {
        ctx.render_settings.is_final_render_mode = false; // Finish
        extern std::atomic<bool> rendering_stopped_cpu;
        rendering_stopped_cpu = true; 
    }

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.08f, 1.0f)); // Opaque Background
    
    // Enable Resize and Collapse/Maximize (removed NoCollapse)
    ImGuiWindowFlags win_flags = ImGuiWindowFlags_None; 
    
    if (ImGui::Begin("Render Result", &show_render_window, win_flags)) {
        
        static bool show_sidebar = true;

        // Progress Info
        float progress = (float)current_samples / (float)target_samples;
        if (progress > 1.0f) progress = 1.0f;

        // Header and Toolbar
        // ---------------------------------------------------------
        if (ctx.is_animation_mode) {            
             ImGui::Text("Animation:");
             ImGui::SameLine();
             
             if (rendering_in_progress) ImGui::TextColored(ImVec4(1, 1, 0, 1), "[RENDERING...]");
             else ImGui::TextColored(ImVec4(0, 1, 0, 1), "[FINISHED / STOPPED]");
             
             ImGui::SameLine();
             int cur_frame = ctx.render_settings.animation_current_frame;
             int end_frame = ctx.render_settings.animation_end_frame;
             int start_frame = ctx.render_settings.animation_start_frame;
             int total = end_frame - start_frame + 1;
             int current_idx = cur_frame - start_frame + 1;
             if (current_idx < 0) current_idx = 0;
             
             float progress = (total > 0) ? (float)current_idx / (float)total : 0.0f;
             char buf[64];
             sprintf(buf, "Frame: %d / %d", cur_frame, end_frame);
             ImGui::ProgressBar(progress, ImVec2(200, 0), buf);
        }
        else {
            ImGui::Text("Status:");
            ImGui::SameLine();
            if (current_samples >= target_samples) {
                ImGui::TextColored(ImVec4(0,1,0,1), "[FINISHED]");
            } else if (ctx.render_settings.is_final_render_mode) {
                 ImGui::TextColored(ImVec4(1,1,0,1), "[RENDERING...]");
            } else {
                 ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "[IDLE]");
            }

            // Progress Bar
            ImGui::SameLine();
            char buf[32];
            sprintf(buf, "%d / %d", current_samples, target_samples);
            ImGui::ProgressBar(progress, ImVec2(200, 0), buf);
        }

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        // Toolbar Buttons
        if (ImGui::Button("Save Image")) {
             std::string filename = "Render_" + std::to_string(time(0)) + ".png";
             ctx.render_settings.save_image_requested = true;
        }
        
        ImGui::SameLine();
        if (ctx.is_animation_mode) {
            
             if (rendering_in_progress) {
                 if (UIWidgets::DangerButton("Stop Anim", ImVec2(80, 0))) {
                     extern std::atomic<bool> rendering_stopped_cpu;
                     extern std::atomic<bool> rendering_stopped_gpu;
                     rendering_stopped_cpu = true;
                     rendering_stopped_gpu = true;
                 }
             } else {
                 if (ImGui::Button("Close")) {
                     ctx.is_animation_mode = false;
                     show_render_window = false;
                 }
             }
        }
        else if (ctx.render_settings.is_final_render_mode) {
            if (UIWidgets::DangerButton("Stop", ImVec2(60, 0))) {
                ctx.render_settings.is_final_render_mode = false;
                extern std::atomic<bool> rendering_stopped_cpu; // Also stop loop!
                rendering_stopped_cpu = true;
            }
        } else {
            if (UIWidgets::PrimaryButton("Render", ImVec2(60, 0))) {
                ctx.renderer.resetCPUAccumulation();
                if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
                ctx.render_settings.is_final_render_mode = true;
                ctx.start_render = true; 
            }
        }

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        // Zoom & Fit
        static float zoom = 1.0f;
        if (ImGui::Button("Fit")) {
             extern int image_width, image_height;
             ImVec2 avail = ImGui::GetContentRegionAvail();
             // Account for sidebar if visible
             float avail_w = avail.x - (show_sidebar ? 305.0f : 0.0f);
             
             if (image_width > 0 && image_height > 0) {
                 float rX = avail_w / image_width;
                 float rY = avail.y / image_height;
                 zoom = (rX < rY) ? rX : rY;
             }
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if (SceneUI::DrawSmartFloat("zoom_res", "Zoom", &zoom, 0.1f, 5.0f, "%.1fx", false, nullptr, 16)) {}

        // Sidebar Toggle
        ImGui::SameLine();
        float avail_width_right = ImGui::GetContentRegionAvail().x;
        // Align to right
        ImGui::SameLine(ImGui::GetWindowWidth() - 110.0f);
        if (UIWidgets::SecondaryButton(show_sidebar ? "Hide Panel >>" : "<< Options", ImVec2(90, 0))) {
            show_sidebar = !show_sidebar;
        }

        ImGui::Separator();

        // Layout: Left (Image) | Right (Settings)
        // ---------------------------------------------------------
        float sidebar_width = 300.0f;
        float content_w = ImGui::GetContentRegionAvail().x;
        float image_view_w = show_sidebar ? (content_w - sidebar_width - 8.0f) : content_w;

        // 1. Image Viewer (Left)
        ImGui::BeginChild("RenderView", ImVec2(image_view_w, 0), true, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoMove);
        {
            extern SDL_Texture* raytrace_texture; 
            extern int image_width, image_height;
            
            if (raytrace_texture && image_width > 0 && image_height > 0) {
                 ImGuiIO& io = ImGui::GetIO();
                 
                 // Handle Zoom/Pan if hovered
                 if (ImGui::IsWindowHovered()) {
                     if (io.MouseWheel != 0.0f) {
                         zoom += io.MouseWheel * 0.1f * zoom;
                         if (zoom < 0.1f) zoom = 0.1f;
                         if (zoom > 10.0f) zoom = 10.0f;
                     }
                     if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                         ImVec2 delta = io.MouseDelta;
                         ImGui::SetScrollX(ImGui::GetScrollX() - delta.x);
                         ImGui::SetScrollY(ImGui::GetScrollY() - delta.y);
                     }
                 }

                 float w = (float)image_width * zoom;
                 float h = (float)image_height * zoom;
                 
                 // Center logic
                 ImVec2 avail = ImGui::GetContentRegionAvail();
                 float offX = (avail.x > w) ? (avail.x - w) * 0.5f : 0.0f;
                 float offY = (avail.y > h) ? (avail.y - h) * 0.5f : 0.0f;
                 
                 if (offX > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offX);
                 if (offY > 0) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + offY);

                 SDL_Texture* display_tex = raytrace_texture;
                 if (ctx.is_animation_mode && ctx.animation_preview_texture) {
                     display_tex = ctx.animation_preview_texture;
                     // Use Animation Preview Dimensions
                     w = (float)ctx.animation_preview_width * zoom;
                     h = (float)ctx.animation_preview_height * zoom;
                 }
                 
                 ImGui::Image((ImTextureID)display_tex, ImVec2(w, h));
                 
                 // HUD calls removed from here (moved back to main draw function)
                 
                 if (ImGui::IsItemHovered()) {
                     ImGui::SetTooltip("Res: %dx%d | Zoom: %.1f%%", image_width, image_height, zoom * 100.0f);
                 }
            } else {
                 ImGui::TextColored(ImVec4(1,0,0,1), "No Render Output Available");
            }
        }
        ImGui::EndChild();

        // 2. Sidebar (Right)
        if (show_sidebar) {
            ImGui::SameLine();
            ImGui::BeginChild("RenderSettingsSidebar", ImVec2(sidebar_width, 0), true);
            
            // Post-Processing moved to main window only
            // DrawRenderWindowToneMapControls(ctx); // Removed as requested
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextDisabled("Stats:");
            ImGui::Text("Samples: %d", current_samples);
            ImGui::Text("Resolution: %dx%d", ctx.render_settings.final_render_width, ctx.render_settings.final_render_height);
            
            ImGui::EndChild();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    
    // Safety: If window is closed, ensure we exit final render mode
    if (!show_render_window && ctx.render_settings.is_final_render_mode) {
        ctx.render_settings.is_final_render_mode = false;
    }
    
    // Sync F12 trigger from main loop
    if (show_render_window && ctx.start_render && !ctx.render_settings.is_final_render_mode && current_samples < 5) {
         // Detect if F12 just opened this
         ctx.render_settings.is_final_render_mode = true;
    }
}

void SceneUI::tryExit() {
    if (ProjectManager::getInstance().hasUnsavedChanges()) {
        pending_action = PendingAction::Exit;
        show_exit_confirmation = true;
    } else {
        extern bool quit;
        quit = true;
    }
}

void SceneUI::tryNew(UIContext& ctx) {
    if (ProjectManager::getInstance().hasUnsavedChanges()) {
        pending_action = PendingAction::NewProject;
        show_exit_confirmation = true;
    } else {
        performNewProject(ctx);
    }
}

void SceneUI::tryOpen(UIContext& ctx) {
    if (ProjectManager::getInstance().hasUnsavedChanges()) {
        pending_action = PendingAction::OpenProject;
        show_exit_confirmation = true;
    } else {
        performOpenProject(ctx);
    }
}

void SceneUI::setSceneLoadingStage(const std::string& stage) {
    std::lock_guard<std::mutex> lock(scene_loading_stage_mutex);
    scene_loading_stage = stage;
}

std::string SceneUI::getSceneLoadingStage() const {
    std::lock_guard<std::mutex> lock(scene_loading_stage_mutex);
    return scene_loading_stage;
}

void SceneUI::drawExitConfirmation(UIContext& ctx) {
    if (!show_exit_confirmation) return;

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    ImGui::OpenPopup("Unsaved Changes?");

    if (ImGui::BeginPopupModal("Unsaved Changes?", &show_exit_confirmation, ImGuiWindowFlags_AlwaysAutoResize)) {
        
        std::string actionName = "exiting";
        if (pending_action == PendingAction::NewProject) actionName = "creating a new project";
        else if (pending_action == PendingAction::OpenProject) actionName = "opening a project";
        else if (pending_action == PendingAction::Exit) actionName = "exiting";

        ImGui::Text("You have unsaved changes.");
        ImGui::Text("Do you want to save them before %s?", actionName.c_str());
        ImGui::Separator();

        // Save & Continue
        if (ImGui::Button("Save & Continue", ImVec2(140, 0))) {
            std::string path = ProjectManager::getInstance().getCurrentFilePath();
            if (path.empty()) {
                 path = saveFileDialogW(L"RayTrophi Project (.rtp)\0*.rtp\0", L"rtp");
            }

            if (!path.empty()) {
                updateProjectFromScene(ctx);
                rendering_stopped_cpu = true;
                bool success = ProjectManager::getInstance().saveProject(path, ctx.scene, ctx.render_settings, ctx.renderer);
                
                try {
                    std::string auxPath = path + ".aux.json";
                    nlohmann::json rootJson;
                    rootJson["terrain_graph"] = terrainNodeGraph.toJson();
                     rootJson["viewport_settings"] = {
                        {"shading_mode", viewport_settings.shading_mode},
                        {"show_gizmos", viewport_settings.show_gizmos},
                        {"show_camera_hud", viewport_settings.show_camera_hud},
                        {"show_focus_ring", viewport_settings.show_focus_ring},
                        {"show_zoom_ring", viewport_settings.show_zoom_ring}
                    };
                    rootJson["guide_settings"] = {
                        {"show_safe_areas", guide_settings.show_safe_areas},
                        {"safe_area_type", guide_settings.safe_area_type},
                        {"show_letterbox", guide_settings.show_letterbox},
                        {"aspect_ratio_index", guide_settings.aspect_ratio_index},
                        {"show_grid", guide_settings.show_grid},
                        {"grid_type", guide_settings.grid_type},
                        {"show_center", guide_settings.show_center}
                    };
                    rootJson["sync_sun_with_light"] = sync_sun_with_light;
                    std::ofstream auxFile(auxPath);
                    if (auxFile.is_open()) {
                        auxFile << rootJson.dump(2);
                        auxFile.close();
                    }
                } catch (...) {}
                
                rendering_stopped_cpu = false;

                if (success) {
                    ImGui::CloseCurrentPopup();
                    show_exit_confirmation = false;
                    
                    if (pending_action == PendingAction::Exit) {
                         extern bool quit; quit = true;
                    } else if (pending_action == PendingAction::NewProject) {
                         performNewProject(ctx);
                    } else if (pending_action == PendingAction::OpenProject) {
                         performOpenProject(ctx);
                    }
                    pending_action = PendingAction::None;
                }
            }
        }

        ImGui::SameLine();

        // Discard & Continue
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.0f, 0.6f, 0.6f));
        if (ImGui::Button("Discard & Continue", ImVec2(140, 0))) {
            ImGui::CloseCurrentPopup();
            show_exit_confirmation = false;
            
            if (pending_action == PendingAction::Exit) {
                extern bool quit; quit = true;
            } else if (pending_action == PendingAction::NewProject) {
                 performNewProject(ctx);
            } else if (pending_action == PendingAction::OpenProject) {
                 performOpenProject(ctx);
            }
            pending_action = PendingAction::None;
        }
        ImGui::PopStyleColor();

        ImGui::SameLine();

        // Cancel
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
            show_exit_confirmation = false;
            pending_action = PendingAction::None;
        }
        
        ImGui::EndPopup();
    }
}




void SceneUI::performNewProject(UIContext& ctx) {
     // Clear selection to remove references to objects about to be deleted
     ctx.selection.clearSelection();

     // Reset Foliage Brush to prevent crashes (referencing deleted terrain)
     foliage_brush.enabled = false;
     foliage_brush.active_group_id = -1;

     // Stop rendering while resetting
     rendering_stopped_cpu = true;
     rendering_stopped_gpu = true;

     // Ensure no in-flight render/update is still touching scene/backend.
     int wait_count = 0;
     while (rendering_in_progress.load() && wait_count < 200) {
         std::this_thread::sleep_for(std::chrono::milliseconds(10));
         ++wait_count;
     }
     if (ctx.backend_ptr) {
         ctx.backend_ptr->waitForCompletion();
     }
     
     // 1. Reset Global Project System
     g_ProjectManager.newProject(ctx.scene, ctx.renderer);
     
     // 2. Reset UI-Side Persistent Data (Node Graphs, History, Cache)
     terrainNodeGraph.clear();
     terrainNodeEditorUI.reset(); // Reset editor pan/zoom/selection
     show_terrain_graph = false;  // Hide graph panel
     show_anim_graph = false;     // Hide animation graph panel
     ForceFieldUI::selected_force_field = nullptr; // Clear force field selection
     resetMaterialUI();           // Reset material editor state
     hairUI.clear();              // Clear hair UI state

     
     history.clear();
     timeline.reset();
     active_messages.clear();
     invalidateCache();
     
     // 3. Reset Viewport & Guide Settings
     viewport_settings = ViewportDisplaySettings();
     guide_settings = GuideSettings();
     sync_sun_with_light = true;
     is_picking_focus = false;
     
     // 4. Reset Rendering State
     ctx.sample_count = 0;
     ctx.renderer.resetCPUAccumulation();
     if (ctx.backend_ptr) {
         ctx.backend_ptr->resetAccumulation();
         // Explicitly clear GPU VDB buffer
         syncVDBVolumesToGPU(ctx); // Sends empty list since scene.vdb_volumes is cleared
     }
     
     // 5. Create Default Scene (Camera, Ground Plane, default light)
     createDefaultScene(ctx.scene, ctx.renderer, ctx.backend_ptr);
     
     ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
     if (ctx.backend_ptr) {
         ctx.renderer.rebuildBackendGeometry(ctx.scene);
         ctx.renderer.updateBackendMaterials(ctx.scene);
         ctx.renderer.updateBackendGasVolumes(ctx.scene);
         ctx.backend_ptr->setLights(ctx.scene.lights);
         auto wd = ctx.renderer.world.getGPUData();
         ctx.backend_ptr->setWorldData(&wd);
         if (ctx.scene.camera) {
             ctx.renderer.syncCameraToBackend(*ctx.scene.camera);
         }
         ctx.backend_ptr->resetAccumulation();
     }
     if(ctx.scene.camera) ctx.scene.camera->update_camera_vectors();

     extern bool g_camera_dirty;
     extern bool g_lights_dirty;
     extern bool g_world_dirty;
     extern std::atomic<bool> g_needs_optix_sync;
     g_camera_dirty = true;
     g_lights_dirty = true;
     g_world_dirty = true;
     g_needs_optix_sync.store(true);
     
     active_model_path = "Untitled";
     ctx.active_model_path = "Untitled";
     ctx.start_render = true;
     
     SCENE_LOG_INFO("New project created.");
     addViewportMessage("New Project Created");
     
     g_ProjectManager.getProjectData().is_modified = false;

     pending_action = PendingAction::None;
     show_exit_confirmation = false;
     
     // Reset Animation Graph UI
     g_animGraphUI = AnimGraphUIState();
}

void SceneUI::performOpenProject(UIContext& ctx) {
    if (g_scene_loading_in_progress.load()) {
        SCENE_LOG_WARN("Already loading a project. Please wait...");
        return;
    }
    
    std::string filepath = openFileDialogW(L"RayTrophi Project (.rtp;.rts)\0*.rtp;*.rts\0All Files\0*.*\0");
    if (!filepath.empty()) {
        // Clear selection to remove references to old objects (Fixes ghost camera issue)
        ctx.selection.clearSelection();

        // Reset Foliage Brush
        foliage_brush.enabled = false;
        foliage_brush.active_group_id = -1;

        // 1. Reset UI-Side Persistent Data before loading new project
        terrainNodeGraph.clear();
        terrainNodeEditorUI.reset(); // Reset editor pan/zoom/selection
        show_terrain_graph = false;  // Hide graph panel
        show_anim_graph = false;     // Hide animation graph panel
        ForceFieldUI::selected_force_field = nullptr; // Clear force field selection
        resetMaterialUI();           // Reset material editor state
        hairUI.clear();              // Clear hair UI state

        
        history.clear();
        timeline.reset();
        active_messages.clear();
        active_messages.clear();
        invalidateCache();
        
        // Reset Animation Graph UI
        g_animGraphUI = AnimGraphUIState();
        
        // Reset Viewport & Guide Settings to default before loading
        viewport_settings = ViewportDisplaySettings();
        guide_settings = GuideSettings();
        sync_sun_with_light = true;
        is_picking_focus = false;


        g_scene_loading_in_progress = true;
        rendering_stopped_cpu = true;
        rendering_stopped_gpu = true;
        
        scene_loading = true;
        scene_loading_done = false;
        pending_project_ui_restore = false;
        scene_loading_progress = 0;
        ctx.sample_count = 0;
        setSceneLoadingStage("Opening project...");
        
        std::thread loader_thread([this, filepath, &ctx]() {
            try {
                scene_loading_progress = 1;
                setSceneLoadingStage("Preparing loader...");

                // Ensure no in-flight render/update is touching scene/backend before load.
                int wait_count = 0;
                while (rendering_in_progress.load() && wait_count < 200) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    ++wait_count;
                }
                if (ctx.backend_ptr) {
                    ctx.backend_ptr->waitForCompletion();
                }

                std::string ext;
                {
                    auto dot_pos = filepath.find_last_of('.');
                    if (dot_pos != std::string::npos) ext = filepath.substr(dot_pos);
                }
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".rtp") {
                    g_ProjectManager.openProject(filepath, ctx.scene, ctx.render_settings, ctx.renderer, ctx.backend_ptr,
                        [this](int p, const std::string& s) {
                            scene_loading_progress = p;
                            setSceneLoadingStage(s);
                        });

                    {
                        std::string auxPath = filepath + ".aux.json";
                        std::string oldGraphPath = filepath + ".nodegraph.json";
                        bool loaded = false;

                        if (std::filesystem::exists(auxPath)) {
                            try {
                                std::ifstream file(auxPath);
                                if (file.is_open()) {
                                    nlohmann::json rootJson;
                                    file >> rootJson;
                                    file.close();

                                    TerrainObject* terrain = nullptr;
                                    auto& terrains = TerrainManager::getInstance().getTerrains();
                                    if (!terrains.empty()) terrain = &terrains[0];

                                    if (rootJson.contains("terrain_graph")) {
                                        terrainNodeGraph.fromJson(rootJson["terrain_graph"], terrain);
                                    }

                                    if (rootJson.contains("viewport_settings")) {
                                        auto& vs = rootJson["viewport_settings"];
                                        viewport_settings.shading_mode = vs.value("shading_mode", 1);
                                        viewport_settings.show_gizmos = vs.value("show_gizmos", true);
                                        viewport_settings.show_camera_hud = vs.value("show_camera_hud", true);
                                        viewport_settings.show_focus_ring = vs.value("show_focus_ring", true);
                                        viewport_settings.show_zoom_ring = vs.value("show_zoom_ring", true);
                                        viewport_settings.focus_mode = vs.value("focus_mode", 1); // Reset to AF-S if missing
                                    }

                                    if (rootJson.contains("guide_settings")) {
                                        auto& gs = rootJson["guide_settings"];
                                        guide_settings.show_safe_areas = gs.value("show_safe_areas", false);
                                        guide_settings.safe_area_type = gs.value("safe_area_type", 0);
                                        guide_settings.show_letterbox = gs.value("show_letterbox", false);
                                        guide_settings.aspect_ratio_index = gs.value("aspect_ratio_index", 0);
                                        guide_settings.show_grid = gs.value("show_grid", false);
                                        guide_settings.grid_type = gs.value("grid_type", 0);
                                        guide_settings.show_center = gs.value("show_center", false);
                                    }

                                    if (rootJson.contains("sync_sun_with_light")) {
                                        sync_sun_with_light = rootJson["sync_sun_with_light"].get<bool>();
                                    }

                                    SCENE_LOG_INFO("[Load] Auxiliary settings loaded.");
                                    loaded = true;
                                }
                            } catch (...) {}
                        }

                        if (!loaded && std::filesystem::exists(oldGraphPath)) {
                            try {
                                std::ifstream ngFile(oldGraphPath);
                                if (ngFile.is_open()) {
                                    nlohmann::json graphJson;
                                    ngFile >> graphJson;
                                    ngFile.close();
                                    TerrainObject* terrain = nullptr;
                                    auto& terrains = TerrainManager::getInstance().getTerrains();
                                    if (!terrains.empty()) terrain = &terrains[0];
                                    terrainNodeGraph.fromJson(graphJson, terrain);
                                }
                            } catch (...) {}
                        }
                    }
                } else {
                    SceneSerializer::Deserialize(ctx.scene, ctx.render_settings, ctx.renderer, ctx.backend_ptr, filepath);
                }

                invalidateCache();
                active_model_path = g_ProjectManager.getProjectName();

                if (ctx.backend_ptr) {
                    ctx.backend_ptr->waitForCompletion();
                }

                if (!ctx.scene.animationDataList.empty()) {
                    auto& animCtrl = AnimationController::getInstance();
                    animCtrl.registerClips(ctx.scene.animationDataList);
                    const auto& clips = animCtrl.getAllClips();
                    if (!clips.empty()) {
                        animCtrl.play(clips[0].name, 0.0f);
                        SCENE_LOG_INFO("[SceneUI] Auto-playing animation: " + clips[0].name);
                    }
                    SCENE_LOG_INFO("[SceneUI] Registered " + std::to_string(ctx.scene.animationDataList.size()) + " animation clips after project load.");
                }

                g_ProjectManager.getProjectData().is_modified = false;

                TerrainObject* terrain = nullptr;
                auto& terrains = TerrainManager::getInstance().getTerrains();
                if (!terrains.empty()) terrain = &terrains[0];
                if (terrain) {
                    terrainNodeGraph.evaluateTerrain(terrain, ctx.scene);
                    SCENE_LOG_INFO("[Load] Terrain graph evaluated.");
                }

                hairUI.clear();
                SceneUI::syncVDBVolumesToGPU(ctx);

                ctx.active_model_path = filepath; // Update project name for window title
            } catch (const std::exception& e) {
                SCENE_LOG_ERROR(std::string("[Open] Exception: ") + e.what());
            } catch (...) {
                SCENE_LOG_ERROR("[Open] Unknown exception.");
            }

            scene_loading = false;
            scene_loading_done = true;
            pending_project_ui_restore = true;
            g_scene_loading_in_progress = false;
            rendering_stopped_cpu = false;
            rendering_stopped_gpu = false;
        });
        loader_thread.detach();
    }
    
    pending_action = PendingAction::None;
    show_exit_confirmation = false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED VOLUMETRIC UI (VDB & Gas Simulation)
// ═══════════════════════════════════════════════════════════════════════════════

#include "VDBVolume.h"
#include "GasVolume.h"

bool SceneUI::drawVolumeShaderUI(UIContext& ctx, std::shared_ptr<VolumeShader> shader, VDBVolume* vdb, GasVolume* gas) {
    if (!shader) return false;
    
    bool changed = false;
    ImGui::PushID(shader.get());
    
    // ─────────────────────────────────────────────────────────────────────────
    // DENSITY
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // DENSITY
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Density", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
        ImGui::Indent();
        // Channel Selection
        std::vector<std::string> grids;
        if (vdb) grids = vdb->getAvailableGrids();
        else if (gas) grids = {"density", "fuel", "temperature", "interaction"}; // Standard Gas grids
        
        if (!grids.empty() && ImGui::BeginCombo("Channel", shader->density.channel.c_str())) {
            for (const auto& g : grids) {
                if (ImGui::Selectable(g.c_str(), shader->density.channel == g)) {
                    shader->density.channel = g;
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        
        if (ImGui::SliderFloat("Multiplier", &shader->density.multiplier, 0.0f, 100.0f)) changed = true;
        if (ImGui::DragFloatRange2("Remap", &shader->density.remap_low, &shader->density.remap_high, 0.01f, 0.0f, 1.0f)) changed = true;
        if (ImGui::SliderFloat("Edge Falloff", &shader->density.edge_falloff, 0.0f, 2.0f)) changed = true;
        
        ImGui::Unindent();
        UIWidgets::EndSection();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SCATTERING & ABSORPTION
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // SCATTERING & ABSORPTION
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Scattering & Absorption", ImVec4(0.8f, 0.5f, 1.0f, 1.0f))) {
        ImGui::Indent();
        float col[3] = { shader->scattering.color.x, shader->scattering.color.y, shader->scattering.color.z };
        if (ImGui::ColorEdit3("Scattering Color", col)) {
            shader->scattering.color = Vec3(col[0], col[1], col[2]);
            changed = true;
        }
        if (ImGui::DragFloat("Scattering Strength", &shader->scattering.coefficient, 0.1f, 0.0f, 100.0f)) changed = true;
        if (ImGui::SliderFloat("Anisotropy (G)", &shader->scattering.anisotropy, -0.99f, 0.99f)) changed = true;
        
        ImGui::Separator();
        float abs_col[3] = { shader->absorption.color.x, shader->absorption.color.y, shader->absorption.color.z };
        if (ImGui::ColorEdit3("Absorption Color", abs_col)) {
            shader->absorption.color = Vec3(abs_col[0], abs_col[1], abs_col[2]);
            changed = true;
        }
        if (ImGui::DragFloat("Absorption Coeff", &shader->absorption.coefficient, 0.1f, 0.0f, 100.0f)) changed = true;
        
        if (ImGui::TreeNode("Advanced Scattering")) {
            if (ImGui::SliderFloat("Back Scatter G", &shader->scattering.anisotropy_back, -0.99f, 0.0f)) changed = true;
            if (ImGui::SliderFloat("Lobe Mix", &shader->scattering.lobe_mix, 0.0f, 1.0f)) changed = true;
            if (ImGui::SliderFloat("Multi-Scatter", &shader->scattering.multi_scatter, 0.0f, 1.0f)) changed = true;
            ImGui::TreePop();
        }
        ImGui::Unindent();
        UIWidgets::EndSection();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EMISSION (FIRE CONTROLS)
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // EMISSION (FIRE CONTROLS)
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Emission / Fire", ImVec4(1.0f, 0.5f, 0.2f, 1.0f))) {
        ImGui::Indent();
        const char* modes[] = { "None", "Constant", "Blackbody", "Channel" };
        int mode = static_cast<int>(shader->emission.mode);
        if (ImGui::Combo("Emission Mode", &mode, modes, 4)) {
            shader->emission.mode = static_cast<VolumeEmissionMode>(mode);
            changed = true;
        }

        if (shader->emission.mode == VolumeEmissionMode::Constant) {
            float ecol[3] = { shader->emission.color.x, shader->emission.color.y, shader->emission.color.z };
            if (ImGui::ColorEdit3("Color", ecol)) {
                shader->emission.color = Vec3(ecol[0], ecol[1], ecol[2]);
                changed = true;
            }
            if (ImGui::DragFloat("Intensity", &shader->emission.intensity, 0.1f, 0.0f, 1000.0f)) changed = true;
        }
        else if (shader->emission.mode == VolumeEmissionMode::Blackbody) {
            std::vector<std::string> grids;
            if (vdb) grids = vdb->getAvailableGrids();
            else if (gas) grids = {"temperature", "fuel", "density"}; 

            if (!grids.empty() && ImGui::BeginCombo("Temp Channel", shader->emission.temperature_channel.c_str())) {
                for (const auto& g : grids) {
                    if (ImGui::Selectable(g.c_str(), shader->emission.temperature_channel == g)) {
                         shader->emission.temperature_channel = g;
                         changed = true;
                    }
                }
                ImGui::EndCombo();
            }
            
            if (ImGui::SliderFloat("Temp Scale", &shader->emission.temperature_scale, 0.1f, 10.0f)) changed = true;
            if (ImGui::SliderFloat("Blackbody Intensity", &shader->emission.blackbody_intensity, 0.0f, 100.0f)) changed = true;
            
            // Temperature range for color mapping
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Temperature Range (K above ambient)");
            if (ImGui::DragFloat("Temp Min", &shader->emission.temperature_min, 10.0f, 0.0f, 2000.0f, "%.0f K")) changed = true;
            if (ImGui::DragFloat("Temp Max", &shader->emission.temperature_max, 50.0f, 100.0f, 5000.0f, "%.0f K")) changed = true;
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Fire typically ranges 500-1500K above ambient.\nExplosions can reach 2000-3000K.");
            }
            ImGui::Separator();
            
            // ═══════════════════════════════════════════════════════════════
            // INTERACTIVE COLOR RAMP EDITOR (Now Shared!)
            // ═══════════════════════════════════════════════════════════════
            ImGui::Spacing();
            if (ImGui::Checkbox("Use Interactive Color Ramp", &shader->emission.color_ramp.enabled)) changed = true;
            
            if (shader->emission.color_ramp.enabled) {
                auto& ramp = shader->emission.color_ramp;
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 p = ImGui::GetCursorScreenPos();
                float width = (std::max)(100.0f, ImGui::GetContentRegionAvail().x);
                float height = 24.0f;
                float marker_size = 6.0f;
                
                static int selected_stop = -1;
                static int dragging_stop = -1;

                ImGui::InvisibleButton("gradient_bar", ImVec2(width, height + marker_size * 2));
                bool is_clicked = ImGui::IsItemClicked(0);
                ImVec2 mouse_pos = ImGui::GetIO().MousePos;
                float mouse_t = (std::max)(0.0f, (std::min)(1.0f, (mouse_pos.x - p.x) / width));

                if (is_clicked) {
                    bool hit = false;
                    for(int i=0; i<(int)ramp.stops.size(); ++i) {
                        if(fabs(p.x + ramp.stops[i].position * width - mouse_pos.x) < 8) {
                            selected_stop = i; dragging_stop = i; hit = true; break;
                        }
                    }
                    if(!hit) {
                        ColorRampStop stop; stop.position = mouse_t; stop.color = ramp.sample(mouse_t);
                        ramp.stops.push_back(stop);
                        std::sort(ramp.stops.begin(), ramp.stops.end(), [](auto& a, auto& b){ return a.position < b.position; });
                        changed = true;
                    }
                }
                
                if (dragging_stop != -1 && ImGui::IsMouseDown(0)) {
                    ramp.stops[dragging_stop].position = mouse_t;
                    std::sort(ramp.stops.begin(), ramp.stops.end(), [](auto& a, auto& b){ return a.position < b.position; });
                    for(int i=0; i<(int)ramp.stops.size(); ++i) if(ramp.stops[i].position == mouse_t) dragging_stop = i;
                    changed = true;
                } else dragging_stop = -1;

                // Draw Ramp
                for(int i=0; i<(int)width; ++i) {
                    Vec3 c = ramp.sample((float)i/width);
                    draw_list->AddRectFilled(ImVec2(p.x+i, p.y), ImVec2(p.x+i+1, p.y+height), IM_COL32(c.x*255, c.y*255, c.z*255, 255));
                }
                // Draw Markers
                for(int i=0; i<(int)ramp.stops.size(); ++i) {
                    float x = p.x + ramp.stops[i].position * width;
                    draw_list->AddTriangleFilled(ImVec2(x, p.y+height+marker_size*2), ImVec2(x-marker_size, p.y+height), ImVec2(x+marker_size, p.y+height), (i==selected_stop)?IM_COL32(255,255,0,255):IM_COL32(255,255,255,255));
                }
                ImGui::Dummy(ImVec2(width, marker_size * 2 + 5));

                if (selected_stop >= 0 && selected_stop < (int)ramp.stops.size()) {
                    if (ImGui::SliderFloat("Stop Pos", &ramp.stops[selected_stop].position, 0, 1)) changed = true;
                    float c[3] = { ramp.stops[selected_stop].color.x, ramp.stops[selected_stop].color.y, ramp.stops[selected_stop].color.z };
                    if (ImGui::ColorEdit3("Stop Color", c)) { ramp.stops[selected_stop].color = Vec3(c[0],c[1],c[2]); changed = true; }
                    if (ramp.stops.size() > 2 && ImGui::Button("Delete Stop")) { ramp.stops.erase(ramp.stops.begin()+selected_stop); selected_stop = -1; changed = true; }
                }
            }
        }
        ImGui::Unindent();
        UIWidgets::EndSection();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // RAY MARCHING QUALITY
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // RAY MARCHING QUALITY
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Ray Marching Quality", ImVec4(0.7f, 0.7f, 0.7f, 1.0f), false)) { // Closed by default
        ImGui::Indent();
        enum VolumeQualityPresetUI { VolQualityFast = 0, VolQualityBalanced = 1, VolQualityExact = 2, VolQualityCustom = 3 };
        const char* qualityPresetNames[] = { "Fast", "Balanced", "Exact", "Custom" };
        int qualityPreset = shader->quality.quality_preset;
        if (qualityPreset < VolQualityFast || qualityPreset > VolQualityCustom) {
            qualityPreset = VolQualityBalanced;
            shader->quality.quality_preset = VolQualityBalanced;
            changed = true;
        }
        if (ImGui::Combo("Quality Preset", &qualityPreset, qualityPresetNames, IM_ARRAYSIZE(qualityPresetNames))) {
            if (qualityPreset == VolQualityFast) {
                shader->quality.step_size = 0.20f;
                shader->quality.max_steps = 256;
                shader->quality.shadow_steps = 8;
                shader->quality.quality_preset = VolQualityFast;
                changed = true;
            } else if (qualityPreset == VolQualityBalanced) {
                shader->quality.step_size = 0.08f;
                shader->quality.max_steps = 512;
                shader->quality.shadow_steps = 12;
                shader->quality.quality_preset = VolQualityBalanced;
                changed = true;
            } else if (qualityPreset == VolQualityExact) {
                shader->quality.step_size = 0.04f;
                shader->quality.max_steps = 1024;
                shader->quality.shadow_steps = 20;
                shader->quality.quality_preset = VolQualityExact;
                changed = true;
            } else {
                shader->quality.quality_preset = VolQualityCustom;
                changed = true;
            }
        }
        UIWidgets::HelpMarker("Fast: hizli preview, Balanced: varsayilan, Exact: parity/quality.");

        if (ImGui::SliderFloat("Step Size", &shader->quality.step_size, 0.005f, 0.5f, "%.3f")) {
            shader->quality.quality_preset = VolQualityCustom;
            changed = true;
        }
        if (ImGui::SliderInt("Max Steps", &shader->quality.max_steps, 16, 1024)) {
            shader->quality.quality_preset = VolQualityCustom;
            changed = true;
        }
        if (ImGui::SliderInt("Shadow Steps", &shader->quality.shadow_steps, 0, 48)) {
            shader->quality.quality_preset = VolQualityCustom;
            changed = true;
        }
        if (ImGui::SliderFloat("Shadow Strength", &shader->quality.shadow_strength, 0.0f, 1.0f)) changed = true;

        // Hard safety clamp for loaded scenes / manual edits beyond UI limits.
        shader->quality.step_size = (std::max)(0.005f, (std::min)(shader->quality.step_size, 0.5f));
        shader->quality.max_steps = (std::max)(16, (std::min)(shader->quality.max_steps, 1024));
        shader->quality.shadow_steps = (std::max)(0, (std::min)(shader->quality.shadow_steps, 48));
        ImGui::Unindent();
        UIWidgets::EndSection();
    }

    // NOTE: Shader presets moved to Gas Simulation panel (scene_ui_gas.hpp)
    // The "Quick Presets" there configure BOTH simulation AND shader for best results

    ImGui::PopID();

    if (changed) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            syncVDBVolumesToGPU(ctx); 
            ctx.backend_ptr->resetAccumulation();
        }
    }

    return changed;
}

// ============================================================================
// Hair Brush Handling
// ============================================================================

void SceneUI::handleHairBrush(UIContext& ctx) {
    // Check if hair paint mode is active
    if (!hairUI.isPainting()) return;
    
    // [FIX] Ensure CPU BVH is in sync with GPU transforms for accurate painting
    // Gizmo updates only GPU transforms in OptiX mode, leaving CPU vertices stale.
    ensureCPUSyncForPicking(ctx);
    
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return; // UI interaction
    
    int x, y;
    Uint32 buttons = SDL_GetMouseState(&x, &y);
    bool is_left_down = (buttons & SDL_BUTTON(SDL_BUTTON_LEFT));
    
    float win_w = (std::max)(1.0f, io.DisplaySize.x);
    float win_h = (std::max)(1.0f, io.DisplaySize.y);
    float u = (float)x / win_w;
    float v = (float)(win_h - y) / win_h;
    
    if (!ctx.scene.camera || !std::isfinite(u) || !std::isfinite(v)) return;
    Ray r = ctx.scene.camera->get_ray(u, v);
    
    // Safety check for ray
    if (!std::isfinite(r.direction.x)) return;

    // Raycast against scene (Meshes)
    HitRecord rec;
    bool sceneHit = (ctx.scene.bvh && std::isfinite(r.origin.x)) ? ctx.scene.bvh->hit(r, 0.001f, 1e6f, rec, false) : false;

    // Raycast against hair (Curves) - MODERN: Use Volumetric intersection for styling
    Hair::HairHitInfo hRec;
    float searchRadius = hairUI.getBrushSettings().radius;
    bool hairHit = false;
    Hair::HairPaintMode mode = hairUI.getPaintMode();
    bool preferHair = (mode == Hair::HairPaintMode::COMB || 
                      mode == Hair::HairPaintMode::CUT || 
                      mode == Hair::HairPaintMode::REMOVE || 
                      mode == Hair::HairPaintMode::LENGTH || 
                      mode == Hair::HairPaintMode::PUFF || 
                      mode == Hair::HairPaintMode::CLUMP ||
                      mode == Hair::HairPaintMode::WAVE ||
                      mode == Hair::HairPaintMode::FRIZZ ||
                      mode == Hair::HairPaintMode::SMOOTH ||
                      mode == Hair::HairPaintMode::PINCH ||
                      mode == Hair::HairPaintMode::SPREAD ||
                      mode == Hair::HairPaintMode::BRAID);
    // -------------------------------------------------------------------------
    // MODERN PICKING: Always prioritize the Scalp (Emitter) surface.
    // This prevents the brush from "jumping" in depth when passing over hairs.
    // -------------------------------------------------------------------------
    bool hit = false;
    Vec3 hitPoint, hitNormal;
    std::string hitGroomName = "";

    // 1. Try hitting the scene (Scalp Mesh) first
    bool hitScalp = false;
    if (sceneHit && rec.triangle) {
        std::string meshName = rec.triangle->getNodeName();
        if (hairUI.isSurfaceValid(ctx.renderer.getHairSystem(), meshName)) {
            hitScalp = true;
            hit = true;
            hitPoint = rec.point;
            hitNormal = rec.normal;
            hitGroomName = hairUI.getSelectedGroom(ctx.renderer.getHairSystem()) ? hairUI.getSelectedGroom(ctx.renderer.getHairSystem())->name : "";
        }
    }

    // 2. Fallback to Hair Hit if no scalp or if mode specifically requires hair (like CUT/REMOVE)
    if (!hit && mode != Hair::HairPaintMode::ADD && mode != Hair::HairPaintMode::DENSITY) {
        // Only use Volumetric for fallback or specific modes
        bool useVolumetric = (mode != Hair::HairPaintMode::ADD && mode != Hair::HairPaintMode::DENSITY);
        if (useVolumetric) {
            hairHit = ctx.renderer.getHairSystem().intersectVolumetric(r.origin, r.direction, 0.001f, 1e6f, searchRadius, hRec);
        } else {
            hairHit = ctx.renderer.getHairSystem().intersect(r.origin, r.direction, 0.001f, 1e6f, hRec);
        }

        if (hairHit && hairUI.isGroomValid(hRec.groomName)) {
            hit = true;
            hitPoint = hRec.position;
            hitNormal = hRec.normal;
            hitGroomName = hRec.groomName;
        }
    }

    // 3. Last fallback: Any scene hit
    if (!hit && sceneHit) {
        hit = true;
        hitPoint = rec.point;
        hitNormal = rec.normal;
    }
    
    
    if (hit && std::isfinite(hitPoint.x) && std::isfinite(hitNormal.x)) {
        // [FIX] Ensure preview doesn't crash on invalid normals
        if (hitNormal.length_squared() < 0.0001f) hitNormal = Vec3(0, 1, 0);
        
        drawHairBrushPreview(ctx, hitPoint, hitNormal);
        
        // Apply brush on mouse down
        static Vec3 lastHitPos = Vec3(0,0,0);
        static bool wasMouseDown = false;

        bool strokeJustEnded = false;
        if (is_left_down) {
            float deltaTime = io.DeltaTime;
            
            // --- UNDO: Begin stroke on first mouse-down ---
            if (!wasMouseDown) {
                hairUI.beginStroke(ctx.renderer.getHairSystem());
            }
            
            // Calculate dynamic comb direction based on mouse movement
            Vec3 dragDir(0,0,0);
            if (wasMouseDown) {
                Vec3 rawDrag = hitPoint - lastHitPos;
                if (rawDrag.length() > 0.001f) {
                    dragDir = rawDrag.normalize();
                }
            }
            lastHitPos = hitPoint;
            wasMouseDown = true;

            // [FIX] Mirror Surface Projector (Ensures mirrored brush snaps to valid surface only)
            auto surfaceProjector = [&](Vec3& pos, Vec3& norm) -> bool {
                if (!ctx.scene.bvh) return false;
                
                // Increased tolerance for marginal hits
                Ray probe(pos + norm * 1.0f, -norm);
                HitRecord pRec;
                
                if (ctx.scene.bvh->hit(probe, 0.001f, 2.0f, pRec, false)) {
                    if (pRec.triangle) {
                        pos = pRec.point;
                        norm = pRec.normal;
                        return true;
                    }
                }
                return false;
            };

            // [FIX] Sync Groom Transform for RIGID objects only
            // This ensures new projects (Rigid Emitters) follow the object while painting,
            // but Skinned Emitters rely on updateSkinnedGroom (avoiding conflicts).
            if (rec.triangle && !rec.triangle->hasSkinData()) {
                  ctx.renderer.getHairSystem().updateFromMeshTransform(rec.triangle->getNodeName(), rec.triangle->getTransformMatrix());
            }

            hairUI.setSurfaceProjector(surfaceProjector);
            hairUI.applyBrush(ctx.renderer.getHairSystem(), hitPoint, hitNormal, deltaTime, dragDir);
            hairUI.setSurfaceProjector(nullptr);
            
        } else {
            // --- UNDO: End stroke on mouse-up ---
            if (wasMouseDown) {
                hairUI.endStroke(ctx.renderer.getHairSystem());
                strokeJustEnded = true;
            }
            wasMouseDown = false;
            // Handle brush release
            hairUI.applyBrush(ctx.renderer.getHairSystem(), Vec3(0,0,0), Vec3(0,1,0), 0.0f);
        }

        if (hairUI.isDirty()) {
            static uint32_t s_lastHairUploadMs = 0;
            const uint32_t nowMs = SDL_GetTicks();
            const uint32_t kHairUploadIntervalMs = 66; // ~15Hz during brush drag
            const bool uploadNow = strokeJustEnded || (nowMs - s_lastHairUploadMs >= kHairUploadIntervalMs);

            if (uploadNow) {
                ctx.renderer.uploadHairToGPU();
                ctx.renderer.resetCPUAccumulation();
                hairUI.clearDirty();
                s_lastHairUploadMs = nowMs;
            }
        }
    }
    
    // --- UNDO/REDO Keyboard Shortcuts (Ctrl+Z / Ctrl+Y) ---
    if (hairUI.isPainting() && !io.WantCaptureKeyboard) {
        bool ctrlHeld = io.KeyCtrl;
        
        // Ctrl+Z = Undo
        if (ctrlHeld && ImGui::IsKeyPressed(ImGuiKey_Z) && !io.KeyShift) {
            if (hairUI.undo(ctx.renderer.getHairSystem())) {
                ctx.renderer.uploadHairToGPU();
                ctx.renderer.resetCPUAccumulation();
                hairUI.clearDirty();
            }
        }
        
        // Ctrl+Y or Ctrl+Shift+Z = Redo
        if (ctrlHeld && (ImGui::IsKeyPressed(ImGuiKey_Y) || 
            (ImGui::IsKeyPressed(ImGuiKey_Z) && io.KeyShift))) {
            if (hairUI.redo(ctx.renderer.getHairSystem())) {
                ctx.renderer.uploadHairToGPU();
                ctx.renderer.resetCPUAccumulation();
                hairUI.clearDirty();
            }
        }
    }
}

void SceneUI::drawHairBrushPreview(UIContext& ctx, const Vec3& hitPoint, const Vec3& hitNormal) {
    if (!hairUI.isPainting()) return;

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    
    float win_w = io.DisplaySize.x;
    float win_h = io.DisplaySize.y;
    
    if (!ctx.scene.camera) return;
    
    const Hair::HairBrushSettings& brush = hairUI.getBrushSettings();
    
    // Get brush color based on mode
    ImVec4 brushColor;
    switch (hairUI.getPaintMode()) {
        case Hair::HairPaintMode::ADD:
            brushColor = ImVec4(0.2f, 1.0f, 0.2f, 0.8f);  // Green
            break;
        case Hair::HairPaintMode::REMOVE:
            brushColor = ImVec4(1.0f, 0.2f, 0.2f, 0.8f);  // Red
            break;
        case Hair::HairPaintMode::CUT:
            brushColor = ImVec4(1.0f, 0.6f, 0.2f, 0.8f);  // Orange
            break;
        case Hair::HairPaintMode::COMB:
            brushColor = ImVec4(0.2f, 0.6f, 1.0f, 0.8f);  // Blue
            break;
        case Hair::HairPaintMode::LENGTH:
            brushColor = ImVec4(1.0f, 1.0f, 0.2f, 0.8f);  // Yellow
            break;
        case Hair::HairPaintMode::WAVE:
        case Hair::HairPaintMode::FRIZZ:
            brushColor = ImVec4(1.0f, 0.4f, 1.0f, 0.8f);  // Magenta/Pink
            break;
        case Hair::HairPaintMode::SMOOTH:
            brushColor = ImVec4(0.4f, 1.0f, 1.0f, 0.8f);  // Cyan
            break;
        case Hair::HairPaintMode::PINCH:
        case Hair::HairPaintMode::SPREAD:
            brushColor = ImVec4(1.0f, 1.0f, 1.0f, 0.8f);  // White
            break;
        case Hair::HairPaintMode::BRAID:
            brushColor = ImVec4(1.0f, 0.8f, 0.3f, 0.8f);  // Gold/Amber
            break;
        default:
            brushColor = ImVec4(0.8f, 0.8f, 0.8f, 0.6f);  // Gray
            break;
    }
    
    ImU32 col = ImGui::ColorConvertFloat4ToU32(brushColor);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    
    // Project 3D circle to screen
    Camera& cam = *ctx.scene.camera;
    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = Vec3::cross(cam_forward, cam.vup).normalize();
    Vec3 cam_up = Vec3::cross(cam_right, cam_forward).normalize();
    
    auto Project = [&](Vec3 p) -> ImVec2 {
        Vec3 d = p - cam.lookfrom;
        float z = d.dot(cam_forward);
        if (z < 0.1f) z = 0.1f;
        
        float fov_rad = cam.vfov * 3.14159f / 180.0f;
        float h = 2.0f * z * tanf(std::clamp(fov_rad * 0.5f, 0.01f, 1.5f));
        float w = (std::max)(0.001f, h * cam.aspect_ratio);
        
        float sx = d.dot(cam_right) / (w * 0.5f);
        float sy = d.dot(cam_up) / (h * 0.5f);
        
        // Safety check for projection
        if (!std::isfinite(sx) || !std::isfinite(sy)) return ImVec2(-10000, -10000);
        
        return ImVec2(
            (0.5f + sx * 0.5f) * win_w,
            (0.5f - sy * 0.5f) * win_h
        );
    };
    
    auto DrawBrushCircle = [&](const Vec3& center, const Vec3& n, bool is_mirror) {
        ImU32 c = col;
        if (is_mirror) {
             // 50% opacity for mirror ghost
             int alpha = (col >> 24) & 0xFF;
             alpha /= 2;
             c = (col & 0x00FFFFFF) | (alpha << 24);
        }

        int segments = 32;
        Vec3 tangent = Vec3::cross(n, Vec3(0, 1, 0)).normalize();
        if (tangent.length() < 0.1f) {
            tangent = Vec3::cross(n, Vec3(1, 0, 0)).normalize();
        }
        Vec3 bitangent = Vec3::cross(n, tangent);
        
        for (int i = 0; i < segments; i++) {
            float theta1 = (float)i / segments * 6.28318f;
            float theta2 = (float)(i + 1) / segments * 6.28318f;
            
            Vec3 p1 = center + (tangent * cosf(theta1) + bitangent * sinf(theta1)) * brush.radius;
            Vec3 p2 = center + (tangent * cosf(theta2) + bitangent * sinf(theta2)) * brush.radius;
            
            // Offset slightly above surface
            p1 = p1 + n * 0.01f;
            p2 = p2 + n * 0.01f;
            
            ImVec2 sp1 = Project(p1);
            ImVec2 sp2 = Project(p2);
            
            dl->AddLine(sp1, sp2, c, is_mirror ? 1.0f : 2.0f);
        }
        
        // Draw center dot
        ImVec2 center_scr = Project(center + n * 0.01f);
        dl->AddCircleFilled(center_scr, is_mirror ? 3.0f : 4.0f, c);
    };

    // Draw Volumetric Gizmo (3D Wireframe Sphere)
    // 3 Perpendicular circles to give a 3D volume feel
    auto DrawVolumetricBrush = [&](const Vec3& center, bool is_mirror) {
        DrawBrushCircle(center, Vec3(1, 0, 0), is_mirror); // X plane
        DrawBrushCircle(center, Vec3(0, 1, 0), is_mirror); // Y plane
        DrawBrushCircle(center, Vec3(0, 0, 1), is_mirror); // Z plane
    };

    DrawVolumetricBrush(hitPoint, false);

    // [MODERN] Visual Highlight: Color affected hair strands
    auto HighlightStrands = [&](const Vec3& center, bool is_mirror) {
        Hair::HairGroom* groom = hairUI.getSelectedGroom(ctx.renderer.getHairSystem());
        if (!groom) return;
        
        Matrix4x4 localToWorld = groom->transform;
        float brushRadSq = brush.radius * brush.radius;
        
        for (const auto& strand : groom->guides) {
            // Safety: Skip empty or corrupt strands
            if (strand.groomedPositions.empty()) continue;

            // Quick culling
            Vec3 rootWorld = localToWorld.transform_point(strand.baseRootPos);
            float distToRootSq = (rootWorld - center).length_squared();
            if (distToRootSq > (brush.radius + strand.baseLength) * (brush.radius + strand.baseLength) * 2.0f) continue;

            float minDistSq = 1e30f;
            for (const auto& p : strand.groomedPositions) {
                Vec3 pW = localToWorld.transform_point(p);
                float d2 = (pW - center).length_squared();
                if (d2 < minDistSq) minDistSq = d2;
                if (minDistSq < brushRadSq) break; // Optimization
            }

            if (minDistSq < brushRadSq) {
                float dist = std::sqrt(minDistSq);
                float falloff = (std::max)(0.0f, 1.0f - dist / brush.radius);
                
                ImVec4 heatColor = (falloff > 0.7f) ? ImVec4(1.0f, 0.2f, 0.2f, 0.8f) :
                                   (falloff > 0.3f) ? ImVec4(1.0f, 0.8f, 0.0f, 0.6f) :
                                                      ImVec4(0.2f, 1.0f, 0.2f, 0.4f);
                
                if (is_mirror) heatColor.w *= 0.5f;
                ImU32 highlightCol = ImGui::ColorConvertFloat4ToU32(heatColor);
                
                if (strand.groomedPositions.size() > 1) {
                    for (size_t i = 0; i < strand.groomedPositions.size() - 1; ++i) {
                        Vec3 p1w = localToWorld.transform_point(strand.groomedPositions[i]);
                        Vec3 p2w = localToWorld.transform_point(strand.groomedPositions[i+1]);
                        
                        if (!std::isfinite(p1w.x) || !std::isfinite(p2w.x)) continue;

                        ImVec2 p1s = Project(p1w);
                        ImVec2 p2s = Project(p2w);
                        
                        if (p1s.x > -5000 && p2s.x > -5000) {
                            dl->AddLine(p1s, p2s, highlightCol, is_mirror ? 1.2f : 2.0f);
                        }
                    }
                }
            }
        }
    };

    DrawVolumetricBrush(hitPoint, false);
    HighlightStrands(hitPoint, false);

    // Draw Mirror Brushes (Ghost)
    if (brush.mirrorX || brush.mirrorY || brush.mirrorZ) {
        if (Hair::HairGroom* groom = hairUI.getSelectedGroom(ctx.renderer.getHairSystem())) {
            Matrix4x4 localToWorld = groom->transform;
            float det = localToWorld.determinant();
            if (std::abs(det) > 1e-12f) {
                Matrix4x4 worldToLocal = localToWorld.inverse();
                
                Vec3 lp = worldToLocal.transform_point(hitPoint);
                Vec3 ln = worldToLocal.transform_vector(hitNormal).normalize();
                
                for (int i = 1; i < 8; ++i) {
                    bool mx = (i & 1) && brush.mirrorX;
                    bool my = (i & 2) && brush.mirrorY;
                    bool mz = (i & 4) && brush.mirrorZ;
                    
                    if ((i & 1) && !brush.mirrorX) continue;
                    if ((i & 2) && !brush.mirrorY) continue;
                    if ((i & 4) && !brush.mirrorZ) continue;
                    
                    Vec3 mP = lp;
                    Vec3 mN = ln;
                    if (mx) { mP.x = -mP.x; mN.x = -mN.x; }
                    if (my) { mP.y = -mP.y; mN.y = -mN.y; }
                    if (mz) { mP.z = -mP.z; mN.z = -mN.z; }
                    
                    Vec3 wP = localToWorld.transform_point(mP);
                    Vec3 wN = localToWorld.transform_vector(mN).normalize();
                    
                    if (std::isfinite(wP.x) && std::isfinite(wN.x)) {
                        DrawVolumetricBrush(wP, true);
                        HighlightStrands(wP, true);
                    }
                }
            }
        }
    }
}

// ============================================================================
// UI SERIALIZAION
// ============================================================================

std::string SceneUI::serialize() {
    nlohmann::json j;
    // Save state variables
    j["show_animation_panel"] = show_animation_panel;
    j["show_foliage_tab"] = show_foliage_tab;
    j["show_water_tab"] = show_water_tab;
    j["show_terrain_tab"] = show_terrain_tab;
    j["show_system_tab"] = show_system_tab;
    j["show_terrain_graph"] = show_terrain_graph;
    j["show_anim_graph"] = show_anim_graph;
    j["show_volumetric_tab"] = show_volumetric_tab;
    j["show_forcefield_tab"] = show_forcefield_tab;
    j["show_world_tab"] = show_world_tab;
    j["show_hair_tab"] = show_hair_tab;
    j["show_modifiers_tab"] = show_modifiers_tab;
    j["pivot_mode"] = pivot_mode;
    j["active_properties_tab"] = active_properties_tab;
    j["showSidePanel"] = showSidePanel;
    
    // extern bool show_controls_window;
    j["show_controls_window"] = show_controls_window; 
    j["show_asset_browser"] = show_asset_browser;
    j["asset_browser_search"] = asset_browser_search;
    j["asset_browser_tag_filter"] = asset_browser_tag_filter;
    j["asset_browser_view_mode"] = asset_browser_view_mode;
    j["asset_browser_thumbnail_size"] = asset_browser_thumbnail_size;
    j["asset_browser_only_3d"] = asset_browser_only_3d;
    j["asset_browser_favorites_only"] = asset_browser_favorites_only;
    j["active_asset_smart_folder_index"] = active_asset_smart_folder_index;
    j["asset_smart_folders"] = nlohmann::json::array();
    for (const auto& preset : asset_smart_folders) {
        j["asset_smart_folders"].push_back({
            { "name", preset.name },
            { "search", preset.search },
            { "tag_filter", preset.tag_filter },
            { "folder_relative_dir", preset.folder_relative_dir },
            { "favorites_only", preset.favorites_only },
            { "only_3d", preset.only_3d }
        });
    }
    j["asset_browser_folder_width"] = asset_browser_folder_width;
    j["asset_browser_details_height"] = asset_browser_details_height;
    j["active_asset_library_index"] = active_asset_library_index;
    j["asset_libraries"] = nlohmann::json::array();
    ensureDefaultAssetLibrary(asset_library_paths);
    for (const auto& library_path : asset_library_paths) {
        j["asset_libraries"].push_back(serializeLibraryPathEntry(library_path));
    }

    j["show_scene_log"] = show_scene_log;

    // Save panel dimensions
    j["side_panel_width"] = side_panel_width;
    j["bottom_panel_height"] = bottom_panel_height;
    j["hierarchy_panel_height"] = hierarchy_panel_height;

    // Save ImGui Settings (Window Layout)
    size_t size = 0;
    const char* ini_data = ImGui::SaveIniSettingsToMemory(&size);
    if (ini_data && size > 0) {
        j["imgui_ini"] = std::string(ini_data, size);
    }

    return j.dump();
}

void SceneUI::deserialize(const std::string& data) {
    if (data.empty()) return;
    try {
        nlohmann::json j = nlohmann::json::parse(data);
        
        // Restore state variables with checks
        if (j.contains("show_animation_panel")) show_animation_panel = j["show_animation_panel"];
        if (j.contains("show_foliage_tab")) show_foliage_tab = j["show_foliage_tab"];
        if (j.contains("show_water_tab")) show_water_tab = j["show_water_tab"];
        if (j.contains("show_terrain_tab")) show_terrain_tab = j["show_terrain_tab"];
        if (j.contains("show_system_tab")) show_system_tab = j["show_system_tab"];
        if (j.contains("show_terrain_graph")) show_terrain_graph = j["show_terrain_graph"];
        if (j.contains("show_anim_graph")) show_anim_graph = j["show_anim_graph"];
        if (j.contains("show_volumetric_tab")) show_volumetric_tab = j["show_volumetric_tab"];
        if (j.contains("show_forcefield_tab")) show_forcefield_tab = j["show_forcefield_tab"];
        if (j.contains("show_world_tab")) show_world_tab = j["show_world_tab"];
        if (j.contains("show_hair_tab")) show_hair_tab = j["show_hair_tab"];
        if (j.contains("show_modifiers_tab")) show_modifiers_tab = j["show_modifiers_tab"];
        
        if (j.contains("pivot_mode")) pivot_mode = j["pivot_mode"];
        if (j.contains("active_properties_tab")) active_properties_tab = j["active_properties_tab"];
        if (j.contains("showSidePanel")) showSidePanel = j["showSidePanel"];
        
        // extern bool show_controls_window;
        if (j.contains("show_controls_window")) show_controls_window = j["show_controls_window"];
        if (j.contains("show_asset_browser")) show_asset_browser = j["show_asset_browser"];
        if (j.contains("asset_browser_search")) asset_browser_search = j["asset_browser_search"];
        if (j.contains("asset_browser_tag_filter")) asset_browser_tag_filter = j["asset_browser_tag_filter"];
        if (j.contains("asset_browser_view_mode")) asset_browser_view_mode = j["asset_browser_view_mode"];
        if (j.contains("asset_browser_thumbnail_size")) asset_browser_thumbnail_size = j["asset_browser_thumbnail_size"];
        if (j.contains("asset_browser_only_3d")) asset_browser_only_3d = j["asset_browser_only_3d"];
        if (j.contains("asset_browser_favorites_only")) asset_browser_favorites_only = j["asset_browser_favorites_only"];
        asset_smart_folders.clear();
        if (j.contains("asset_smart_folders") && j["asset_smart_folders"].is_array()) {
            for (const auto& entry : j["asset_smart_folders"]) {
                AssetSmartFolderPreset preset;
                preset.name = entry.value("name", std::string());
                preset.search = entry.value("search", std::string());
                preset.tag_filter = entry.value("tag_filter", std::string());
                preset.folder_relative_dir = entry.value("folder_relative_dir", std::string());
                preset.favorites_only = entry.value("favorites_only", false);
                preset.only_3d = entry.value("only_3d", true);
                if (!preset.name.empty()) {
                    asset_smart_folders.push_back(preset);
                }
            }
        }
        active_asset_smart_folder_index = j.value("active_asset_smart_folder_index", -1);
        if (active_asset_smart_folder_index < -1 || active_asset_smart_folder_index >= static_cast<int>(asset_smart_folders.size())) {
            active_asset_smart_folder_index = -1;
        }
        if (j.contains("asset_browser_folder_width")) asset_browser_folder_width = j["asset_browser_folder_width"];
        if (j.contains("asset_browser_details_height")) asset_browser_details_height = j["asset_browser_details_height"];
        asset_library_paths.clear();
        if (j.contains("asset_libraries") && j["asset_libraries"].is_array()) {
            for (const auto& library_entry : j["asset_libraries"]) {
                const std::filesystem::path library_path = deserializeLibraryPathEntry(library_entry);
                if (!library_path.empty()) {
                    asset_library_paths.push_back(library_path);
                }
            }
        }
        ensureDefaultAssetLibrary(asset_library_paths);
        active_asset_library_index = j.value("active_asset_library_index", 0);
        if (!asset_library_paths.empty()) {
            active_asset_library_index = (std::max)(0, (std::min)(active_asset_library_index, static_cast<int>(asset_library_paths.size()) - 1));
            refreshAssetLibrarySafely(asset_registry, normalizeAbsolutePath(asset_library_paths[active_asset_library_index]));
        }
        
        if (j.contains("show_scene_log")) show_scene_log = j["show_scene_log"];

        if (j.contains("side_panel_width")) side_panel_width = j["side_panel_width"];
        if (j.contains("bottom_panel_height")) bottom_panel_height = j["bottom_panel_height"];
        if (j.contains("hierarchy_panel_height")) hierarchy_panel_height = j["hierarchy_panel_height"];

        // Restore ImGui Settings
        if (j.contains("imgui_ini")) {
             std::string ini_str = j["imgui_ini"];
             if (!ini_str.empty()) {
                 ImGui::LoadIniSettingsFromMemory(ini_str.c_str(), ini_str.size());
                 SCENE_LOG_INFO("Restored ImGui layout (Size: " + std::to_string(ini_str.size()) + ")");
             } else {
                 SCENE_LOG_WARN("ImGui layout string is empty in project data.");
             }
        } else {
             SCENE_LOG_WARN("No imgui_ini found in project data.");
        }
        SCENE_LOG_INFO("UI layout restored from project file.");
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("Failed to deserialize UI layout: " + std::string(e.what()));
    }
}
