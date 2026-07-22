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
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "scene_data.h"    // Added explicit include
#include "ui_modern.h"
#include "imgui.h"
#include "imgui_internal.h"   // DockBuilder* API + ImGuiDockNode for the dockable layout
#include <SDL_image.h>
#include "stb_image.h"
#include "ImGuizmo.h"  // Transform gizmo
#include <string>
#include <memory>  // For std::make_unique
#include "KeyframeSystem.h"   // For keyframe animation
#include "scene_ui_guides.hpp" // Viewport guides (safe areas, letterbox, grids)
#include "TimelineWidget.h"   // Custom timeline widget
#include "scene_data.h"
#include "scene_ui_water_v2.hpp"   // Vulkan-first Water V2 panel
#include "scene_ui_river.hpp"   // River spline editor
#include "WaterSystem.h"        // Water Manager for update loop
#include "scene_ui_terrain.hpp" // Terrain panel implementation
#include "scene_ui_animgraph.hpp" // Animation Graph Editor
#include "scene_ui_gas.hpp"     // Gas Simulation panel
#include "scene_ui_forcefield.hpp" // Force Field panel
#include "ParallelBVHNode.h"
#include "Triangle.h"  // For object hierarchy
#include "HittableInstance.h"
#include "InstanceManager.h"
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
#include "Paint/MeshPaintAdapter.h"
#include "default_scene_creator.hpp"
#include "SceneSerializer.h"
#include "ProjectManager.h"  // Project system
#include "Api/RtPython.h"
#include "Api/RtIpcPanel.h"
#include "Api/RtApi.h"
#include "MaterialManager.h"  // For material editing
#include <map>  // For mesh grouping
#include <unordered_set>  // For fast deletion lookup
#include <windows.h>
#include <commdlg.h>

// System RAM query for the "RAM sim cache is large, bake to disk" nudge. Declared in
// scene_ui_forcefield.hpp (which is included before <windows.h>), defined here where
// the Win32 API is available.
namespace ForceFieldUI {
    std::uint64_t queryTotalPhysicalRamBytes() {
        MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) return static_cast<std::uint64_t>(ms.ullTotalPhys);
        return 0;
    }
}

#include <shlobj.h>  // SHBrowseForFolder için
#include <shobjidl.h>
#include <chrono>  // Playback timing için
#include <filesystem>  // Frame dosyalarını kontrol için
#include <unordered_map>
#include <atomic>
#include <vector>
#include <sstream>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

float g_main_menu_reserved_height = 30.0f;

float getMainMenuReservedHeight() {
    return (std::max)(28.0f, g_main_menu_reserved_height);
}

namespace {
Backend::IBackend* getSceneUiRenderBackend(UIContext& ctx) {
    if (g_backend) {
        return g_backend.get();
    }
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr) {
        return ctx.backend_ptr;
    }
    return nullptr;
}

Backend::IViewportBackend* getSceneUiViewportBackend(UIContext& ctx) {
    if (g_viewport_backend) {
        return g_viewport_backend.get();
    }
    return dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
}

void resetSceneUiSamplingAccumulation(UIContext& ctx) {
    Backend::IBackend* renderBackend = getSceneUiRenderBackend(ctx);
    Backend::IViewportBackend* viewportBackend = getSceneUiViewportBackend(ctx);
    if (renderBackend) {
        renderBackend->resetAccumulation();
    }
    if (viewportBackend && viewportBackend != renderBackend) {
        viewportBackend->resetAccumulation();
    }
}

bool sceneUiRenderBackendIsVulkan(UIContext& ctx) {
    return dynamic_cast<Backend::VulkanBackendAdapter*>(getSceneUiRenderBackend(ctx)) != nullptr;
}

void applySceneUiPendingDeleteVisibility(UIContext& ctx, Backend::IBackend* backend) {
    if (!backend || ctx.scene.editor_pending_delete_object_names.empty()) {
        return;
    }

    for (const auto& nodeName : ctx.scene.editor_pending_delete_object_names) {
        if (!nodeName.empty()) {
            backend->setVisibilityByNodeName(nodeName, false);
        }
    }
}

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

bool selectionContainsLight(const SceneData& scene, const std::shared_ptr<Light>& light) {
    return light && std::find(scene.lights.begin(), scene.lights.end(), light) != scene.lights.end();
}

bool selectionContainsCamera(const SceneData& scene, const std::shared_ptr<Camera>& camera) {
    return camera && std::find(scene.cameras.begin(), scene.cameras.end(), camera) != scene.cameras.end();
}

bool selectionContainsVDB(const SceneData& scene, const std::shared_ptr<VDBVolume>& vdb) {
    return vdb && std::find(scene.vdb_volumes.begin(), scene.vdb_volumes.end(), vdb) != scene.vdb_volumes.end();
}

bool selectionContainsGas(const SceneData& scene, const std::shared_ptr<GasVolume>& gas) {
    return gas && std::find(scene.gas_volumes.begin(), scene.gas_volumes.end(), gas) != scene.gas_volumes.end();
}

bool selectionContainsForceField(const SceneData& scene, const std::shared_ptr<Physics::ForceField>& field) {
    const auto& fields = scene.force_field_manager.force_fields;
    return field && std::find(fields.begin(), fields.end(), field) != fields.end();
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
    auto wideToUtf8 = [](const wchar_t* path) -> std::string {
        if (!path || path[0] == L'\0') return {};
        const int size_needed = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
        if (size_needed <= 1) return {};
        std::string utf8_path(static_cast<std::size_t>(size_needed), '\0');
        WideCharToMultiByte(CP_UTF8, 0, path, -1, utf8_path.data(), size_needed, nullptr, nullptr);
        utf8_path.resize(static_cast<std::size_t>(size_needed - 1));
        return utf8_path;
    };

    const HRESULT co_init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    const bool should_uninit = SUCCEEDED(co_init);

    IFileDialog* dialog = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog));
    if (SUCCEEDED(hr) && dialog) {
        DWORD options = 0;
        if (SUCCEEDED(dialog->GetOptions(&options))) {
            dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR);
        }
        if (title) {
            dialog->SetTitle(title);
        }

        hr = dialog->Show(GetActiveWindow());
        if (SUCCEEDED(hr)) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dialog->GetResult(&item)) && item) {
                PWSTR path = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                    std::string result = wideToUtf8(path);
                    CoTaskMemFree(path);
                    item->Release();
                    dialog->Release();
                    if (should_uninit) CoUninitialize();
                    return result;
                }
                item->Release();
            }
        }
        dialog->Release();
    }

    if (should_uninit) CoUninitialize();
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
    ImGuiTreeNodeFlags hdrFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowOverlap;
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
    if (ImGui::CollapsingHeader("LCD Widget Theme", 0)) {
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
                [](void* data, int idx) -> const char* {
                    return ((ResolutionPreset*)data)[idx].name;
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

void SceneUI::drawRenderInspectorContent(UIContext& ctx)
{
    extern bool g_hasOptix;
    extern bool g_hasVulkan;
    extern bool g_viewport_raster_rebuild_pending;
    static int deferred_engine_type = -1; // 0=CPU,1=OptiX,2=Vulkan

    UIWidgets::PushControlSurfaceStyle(ImVec4(0.68f, 0.78f, 1.0f, 1.0f));
    float child_round = 14.0f;
    float frame_round = 14.0f;
    float grab_round = 14.0f;
    float popup_round = 14.0f;
    ImVec2 item_spacing = ImVec2(8.0f, 6.0f);
    
    ImVec4 child_bg = ImVec4(0.10f, 0.115f, 0.14f, 0.94f);
    ImVec4 frame_bg = ImVec4(0.13f, 0.145f, 0.17f, 0.98f);
    ImVec4 frame_bg_hovered = ImVec4(0.16f, 0.18f, 0.215f, 0.99f);
    ImVec4 frame_bg_active = ImVec4(0.19f, 0.215f, 0.25f, 1.0f);
    
    ImVec4 header = ImVec4(0.14f, 0.17f, 0.21f, 0.96f);
    ImVec4 header_hovered = ImVec4(0.18f, 0.21f, 0.26f, 0.98f);
    ImVec4 header_active = ImVec4(0.22f, 0.25f, 0.30f, 1.0f);
    
    ImVec4 inspectorSliderGrab = ImVec4(0.72f, 0.82f, 1.0f, 0.95f);
    ImVec4 inspectorSliderGrabActive = ImVec4(0.86f, 0.91f, 1.0f, 1.0f);
    
    if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
        const auto& curTheme = ThemeManager::instance().current();
        child_round = curTheme.style.windowRounding;
        frame_round = curTheme.style.frameRounding;
        grab_round = curTheme.style.grabRounding;
        popup_round = curTheme.style.popupRounding;
        
        child_bg = ImVec4(curTheme.colors.surface.x, curTheme.colors.surface.y, curTheme.colors.surface.z, 0.94f);
        frame_bg = curTheme.colors.surface;
        frame_bg_hovered = UIWidgets::ScaleColor(curTheme.colors.surface, 1.3f);
        frame_bg_active = UIWidgets::ScaleColor(curTheme.colors.surface, 1.5f);
        
        header = ImVec4(curTheme.colors.accent.x, curTheme.colors.accent.y, curTheme.colors.accent.z, 0.22f);
        header_hovered = ImVec4(curTheme.colors.accent.x, curTheme.colors.accent.y, curTheme.colors.accent.z, 0.48f);
        header_active = ImVec4(curTheme.colors.accent.x, curTheme.colors.accent.y, curTheme.colors.accent.z, 0.70f);
        
        inspectorSliderGrab = ImVec4(curTheme.colors.accent.x, curTheme.colors.accent.y, curTheme.colors.accent.z, 0.92f);
        inspectorSliderGrabActive = ImVec4((std::min)(1.0f, curTheme.colors.accent.x + 0.10f), (std::min)(1.0f, curTheme.colors.accent.y + 0.10f), (std::min)(1.0f, curTheme.colors.accent.z + 0.10f), 1.0f);
    }
    
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, child_round);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, frame_round);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, grab_round);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, popup_round);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 14.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, item_spacing);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, child_bg);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, frame_bg);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, frame_bg_hovered);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, frame_bg_active);
    ImGui::PushStyleColor(ImGuiCol_Header, header);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, header_hovered);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, header_active);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, inspectorSliderGrab);
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, inspectorSliderGrabActive);

    const bool rendered_viewport_active = (viewport_settings.shading_mode == 2);
    const bool raster_quality_active = (viewport_settings.shading_mode == 0 ||
                                        viewport_settings.shading_mode == 1 ||
                                        viewport_settings.shading_mode == 3);

    if (ctx.render_settings.use_optix && !g_hasOptix) {
        ctx.render_settings.use_optix = false;
    }
    if (ctx.render_settings.use_vulkan && !g_hasVulkan) {
        ctx.render_settings.use_vulkan = false;
    }

    int engine_type = 0;
    if (ctx.render_settings.use_optix) engine_type = 1;
    if (ctx.render_settings.use_vulkan) engine_type = 2;

    if (rendered_viewport_active && deferred_engine_type >= 0) {
        const int clamped_deferred = std::clamp(deferred_engine_type, 0, 2);
        const bool next_use_optix = (clamped_deferred == 1);
        const bool next_use_vulkan = (clamped_deferred == 2);
        if (ctx.render_settings.use_optix != next_use_optix ||
            ctx.render_settings.use_vulkan != next_use_vulkan) {
            ctx.render_settings.use_optix = next_use_optix;
            ctx.render_settings.use_vulkan = next_use_vulkan;
            extern bool g_cpu_sync_pending;
            g_cpu_sync_pending = true;
            ctx.render_settings.backend_changed = true;
            ctx.start_render = true;
        }
        deferred_engine_type = -1;
    }
    if (!rendered_viewport_active && deferred_engine_type >= 0) {
        engine_type = std::clamp(deferred_engine_type, 0, 2);
    }

    if (UIWidgets::BeginSection("Render Engine & Backend", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
        UIWidgets::ColoredHeader("Execution", ImVec4(0.82f, 0.88f, 1.0f, 1.0f));
        const char* engines[] = { "CPU (Reference)", "NVIDIA OptiX (CUDA)", "Vulkan RT (Recommended)" };
        if (ImGui::BeginCombo("Engine", engines[engine_type])) {
            for (int i = 0; i < IM_ARRAYSIZE(engines); i++) {
                bool is_disabled = false;
                if (i == 1 && !g_hasOptix) is_disabled = true;
                // Disable Vulkan engine selection when Vulkan isn't present OR when
                // Vulkan is available but hardware ray-tracing is not supported.
                if (i == 2 && (!g_hasVulkan || !g_hasVulkanRT)) is_disabled = true;

                bool is_selected = (engine_type == i);
                ImGuiSelectableFlags combo_flags = is_disabled ? ImGuiSelectableFlags_Disabled : 0;
                std::string label = engines[i];
                if (is_disabled) {
                    if (i == 2 && g_hasVulkan && !g_hasVulkanRT) {
                        label += " [Raster Only]";
                    } else {
                        label += " [Not Supported]";
                    }
                }

                if (ImGui::Selectable(label.c_str(), is_selected, combo_flags)) {
                    engine_type = i;
                    if (rendered_viewport_active) {
                        ctx.render_settings.use_optix = (engine_type == 1);
                        ctx.render_settings.use_vulkan = (engine_type == 2);
                        extern bool g_cpu_sync_pending;
                        g_cpu_sync_pending = true;
                        ctx.render_settings.backend_changed = true;
                        ctx.start_render = true;
                        // Surface a HUD message immediately so the user sees status text
                        // on the LAST rendered frame before the backend switch (which can
                        // freeze the UI thread for 1-3 min on first OptiX JIT compile).
                        // ImGui finishes drawing the current frame after this combo handler
                        // returns, so the message renders before the freeze hits.
                        if (engine_type == 1) {
                            addViewportMessage(
                                "Preparing OptiX backend. Renderer keeps running.",
                                5.0f, ImVec4(1.0f, 0.85f, 0.3f, 1.0f));
                        } else if (engine_type == 2) {
                            addViewportMessage(
                                "Switching to Vulkan RT...",
                                10.0f, ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
                        } else {
                            addViewportMessage(
                                "Switching to CPU renderer...",
                                10.0f, ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
                        }
                    } else {
                        deferred_engine_type = engine_type;
                        addViewportMessage("Engine preference saved. It will apply in Rendered mode.",
                            2.0f, ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
                    }
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (!g_hasOptix && !g_hasVulkan) {
            UIWidgets::StatusIndicator("No compatible GPU backend detected", UIWidgets::StatusType::Error);
        } else if (!g_hasOptix && engine_type == 1) {
            UIWidgets::StatusIndicator("Selected OptiX backend is not supported on this machine", UIWidgets::StatusType::Error);
        } else if (!g_hasVulkan && engine_type == 2) {
            UIWidgets::StatusIndicator("Selected Vulkan backend is not supported on this machine", UIWidgets::StatusType::Error);
        }

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("CPU Fallback", ImVec4(0.72f, 0.90f, 0.84f, 1.0f));
        if (!ctx.render_settings.use_optix && !ctx.render_settings.use_vulkan) {
            const char* bvh_items[] = { "Custom RayTrophi BVH", "Intel Embree (Recommended)" };
            int current_bvh = ctx.render_settings.UI_use_embree ? 1 : 0;
            if (ImGui::Combo("CPU BVH Type", &current_bvh, bvh_items, 2)) {
                ctx.render_settings.UI_use_embree = (current_bvh == 1);
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.start_render = true;
            }
        } else {
            ImGui::TextDisabled("CPU BVH selection is only used when the CPU render backend is active.");
        }
        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("Sampling & Quality", ImVec4(0.5f, 0.9f, 0.6f, 1.0f))) {
        UIWidgets::ColoredHeader("Viewport (Interactive)", ImVec4(0.84f, 0.94f, 0.86f, 1.0f));
        bool sampling_changed = false;
        sampling_changed |= ImGui::Checkbox("Use Adaptive Sampling##view", &ctx.render_settings.use_adaptive_sampling);
        if (ctx.render_settings.use_adaptive_sampling) {
            sampling_changed |= ImGui::DragFloat("Noise Threshold", &ctx.render_settings.variance_threshold, 0.001f, 0.001f, 0.8f, "%.3f");
            sampling_changed |= ImGui::DragInt("Min Samples##view", &ctx.render_settings.min_samples, 1, 1, 512);
        }
        sampling_changed |= ImGui::DragInt("Max Samples##view", &ctx.render_settings.max_samples, 1, 1, 10000);
        if (sampling_changed) {
            ctx.start_render = true;
            resetSceneUiSamplingAccumulation(ctx);
        }

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("Viewport HUD", ImVec4(0.70f, 0.86f, 1.0f, 1.0f));
        ImGui::Checkbox("Show Scene Stats HUD", &ctx.render_settings.show_scene_stats_hud);

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("Raster Viewport Quality", ImVec4(0.96f, 0.84f, 0.58f, 1.0f));
        const char* raster_quality_items[] = { "Auto", "Performance", "Balanced", "Quality" };
        int raster_quality = static_cast<int>(ctx.render_settings.raster_viewport_quality_preset);
        if (!raster_quality_active) ImGui::BeginDisabled();
        if (ImGui::Combo("Raster / Preview Quality", &raster_quality, raster_quality_items, IM_ARRAYSIZE(raster_quality_items))) {
            ctx.render_settings.raster_viewport_quality_preset = static_cast<RasterViewportQualityPreset>(raster_quality);
            if (raster_quality_active) {
                g_viewport_raster_rebuild_pending = true;
            }
            ctx.start_render = true;
        }
        if (!raster_quality_active) ImGui::EndDisabled();
        if (raster_quality_active) {
            ImGui::TextDisabled("Solid/Matcap: viewport proxy aggressiveness. Material Preview: specular BRDF quality.");
        } else {
            ImGui::TextDisabled("Active only in Solid, Material Preview, or Matcap viewport mode.");
        }

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("Viewport Grid", ImVec4(0.72f, 0.84f, 0.97f, 1.0f));
        if (!raster_quality_active) ImGui::BeginDisabled();
        ImGui::SliderFloat("Grid Fade Distance", &ctx.render_settings.grid_fade_distance, 0.25f, 3.0f, "%.2fx");
        ImGui::SliderFloat("Grid Opacity", &ctx.render_settings.grid_opacity, 0.0f, 1.0f, "%.2f");
        if (!raster_quality_active) ImGui::EndDisabled();
        if (raster_quality_active) {
            ImGui::TextDisabled("Fade scales the fog horizon where grid lines dissolve. Opacity 0 hides the grid.");
        } else {
            ImGui::TextDisabled("Grid is drawn only in Solid, Material Preview, or Matcap viewport mode.");
        }

        const bool material_preview_active = (viewport_settings.shading_mode == 1);
        const char* preview_lighting_items[] = { "Classic 3-Point", "Studio", "Outdoor" };
        int preview_lighting = static_cast<int>(ctx.render_settings.material_preview_lighting_preset);
        if (!material_preview_active) ImGui::BeginDisabled();
        if (ImGui::Combo("Preview Lighting", &preview_lighting, preview_lighting_items, IM_ARRAYSIZE(preview_lighting_items))) {
            ctx.render_settings.material_preview_lighting_preset = static_cast<MaterialPreviewLightingPreset>(preview_lighting);
            ctx.start_render = true;
            if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
        }
        if (!material_preview_active) ImGui::EndDisabled();
        if (material_preview_active) {
            ImGui::TextDisabled("Classic keeps the old key/fill/rim look. Studio and Outdoor use stable environment-style preview lighting.");
        } else {
            ImGui::TextDisabled("Preview lighting presets are only active in Material Preview mode.");
        }

        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("Light Transport & Denoise", ImVec4(0.95f, 0.68f, 0.34f, 1.0f))) {
        UIWidgets::ColoredHeader("Light Paths", ImVec4(1.0f, 0.88f, 0.54f, 1.0f));
        ImGui::DragInt("Total Bounces", &ctx.render_settings.max_bounces, 1, 1, 64);
        ImGui::DragInt("Diffuse Bounces", &ctx.render_settings.diffuse_bounces, 1, 1, 64);
        ImGui::DragInt("Transmission Bounces", &ctx.render_settings.transmission_bounces, 1, 1, 64);
        ctx.render_settings.diffuse_bounces = std::clamp(ctx.render_settings.diffuse_bounces, 1, ctx.render_settings.max_bounces);
        ctx.render_settings.transmission_bounces = std::clamp(ctx.render_settings.transmission_bounces, 1, ctx.render_settings.max_bounces);
        UIWidgets::HelpMarker("Total is the global path limit. Diffuse limits indirect diffuse/SSS/translucent paths; Transmission limits glass and water continuation.");

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("Caustics (Photon, Vulkan RT)", ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
        ImGui::Checkbox("Enable Caustics##caustics", &ctx.render_settings.caustics_enabled);
        UIWidgets::HelpMarker("Light-traced photon caustics for glass/water (Vulkan RT + CPU).\n"
                              "Photons refract through transmissive surfaces and deposit into a\n"
                              "world-space grid read back during shading.");
        if (ctx.render_settings.caustics_enabled && ctx.render_settings.use_optix) {
            UIWidgets::StatusIndicator("Caustics are not supported on the OptiX backend yet (Vulkan RT / CPU only)",
                                       UIWidgets::StatusType::Warning);
        }
        if (ctx.render_settings.caustics_enabled) {
            ImGui::Indent();
            ImGui::DragInt("Photons / frame##causticsn", &ctx.render_settings.caustics_photons, 1024, 8192, 4194304);
            ImGui::DragFloat("Grid Cell Size##causticscs", &ctx.render_settings.caustics_cell_size, 0.005f, 0.005f, 2.0f, "%.3f");
            ImGui::DragFloat("Energy##causticse", &ctx.render_settings.caustics_energy, 0.05f, 0.0f, 100.0f, "%.2f");
            UIWidgets::HelpMarker("Global photon power multiplier (1 = physical). Also scales the\n"
                                  "volumetric shafts — leave at 1 and use Scatter Strength for those.");
            ImGui::Unindent();
        }

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("Volumetric Light Shafts (Photon, Vulkan RT)", ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
        ImGui::Checkbox("Volumetric (light shafts)##causticsvol", &ctx.render_settings.caustics_volumetric);
        UIWidgets::HelpMarker("Photons deposit energy along their flight paths, making light\n"
                              "visible as shafts — no fog or volume object needed. Independent\n"
                              "of surface caustics; shares the photon budget above.");
        if (ctx.render_settings.caustics_volumetric) {
            ImGui::Indent();
            ImGui::DragFloat("Scatter Strength##causticsvols", &ctx.render_settings.caustics_vol_strength, 0.05f, 0.0f, 100.0f, "%.2f");
            ImGui::Checkbox("Direct light shafts##causticsvoldir", &ctx.render_settings.caustics_vol_direct);
            UIWidgets::HelpMarker("Also deposit the light-to-target leg of the beam, so the full\n"
                                  "light path glows. Point lights emit half their photons\n"
                                  "omnidirectionally in this mode (true light distribution).");
            ImGui::SliderFloat("Shaft Noise##causticsvolnoise", &ctx.render_settings.caustics_vol_noise, 0.0f, 1.0f, "%.2f");
            UIWidgets::HelpMarker("Heterogeneous dust: modulates shaft density with a static 3D\n"
                                  "turbulence field for a wispy, volumetric look. 0 = uniform.");
            ImGui::Unindent();
        }

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("Debug Visualizer (Vulkan RT)", ImVec4(1.0f, 0.62f, 0.42f, 1.0f));
        {
            const char* dv_items[] = {
                "Off", "Photon Grid (volume)", "Light Shaft Density",
                "Photon Energy (surface)", "Caustic Cells", "Photon Directions",
                "Bounce Count", "Transmission", "Absorption", "Medium Density (interior)",
                "Normal (first hit)", "Albedo (first hit)", "Depth", "Material ID",
                "Sample Heatmap"
            };
            const char* dv_tips[] = {
                "Normal beauty render — debug visualizer off.",
                "Marches the primary ray through the VOLUME photon grid and shows the raw "
                "deposited energy — verifies shaft & photon placement in space.\n"
                "Arms the photon pass automatically (even with caustics off).",
                "The volumetric in-scatter integral on its own: the light shafts without "
                "the scene. Arms the photon pass automatically.",
                "SURFACE photon-grid irradiance at the first hit — bright areas are where "
                "caustic energy lands. Arms the photon pass automatically.",
                "Same surface-grid data, but every hash cell gets a stable colour — shows "
                "the grid tiling. Use it to tune Grid Cell Size.",
                "Average photon FLOW direction per volume cell, RGB = XYZ (e.g. green = "
                "photons travelling up). Brightness follows the deposited energy. Arms "
                "the photon pass + direction grid automatically.",
                "Viridis heatmap of path depth per pixel. Yellow = paths that reach the "
                "bounce limit (glass/GI hotspots); purple = terminated early.",
                "Surviving path throughput — the colour a white backlight would keep after "
                "crossing this pixel's path. Amber glass reads amber; dense dust darkens.",
                "What the path LOST on the way: 1 - throughput, per channel.",
                "Interior Volume dust-coverage integral along the (refracted) view ray; "
                "opaque specks flash white. Materials without an interior read ~0 "
                "(dark purple).",
                "First-hit shading normal in world space (RGB = XYZ). Winding, normal-map "
                "and sculpt shading bugs show instantly.",
                "First-hit base colour after texturing — exactly the denoiser's albedo "
                "feed. UV / texture-pipeline check.",
                "Distance to the first hit as an exponential ramp. Exposure tunes the "
                "range: higher = shorter falloff. Near = dark, far = yellow.",
                "Stable hash colour per material — multi-material import & assignment "
                "check. Only 'no hit' is black (material ID 0 is valid).",
                "Per-pixel ACCUMULATED SAMPLE COUNT, read at display time — the beauty "
                "render keeps accumulating underneath and toggling never resets it. Best "
                "with Adaptive Sampling on; Exposure scales the ramp."
            };
            int dv = std::clamp(ctx.render_settings.debug_view, 0, 14);
            if (ImGui::BeginCombo("View Mode##dbgview", dv_items[dv])) {
                for (int i = 0; i < 15; ++i) {
                    const bool unavailable = false; // all 15 views live
                    if (unavailable) ImGui::BeginDisabled();
                    if (ImGui::Selectable(dv_items[i], dv == i)) {
                        ctx.render_settings.debug_view = i;
                        // Kick one render-loop pass even when accumulation is
                        // complete — the change tracker only runs inside the
                        // render block, and the tonemap-only refresh (heatmap)
                        // adds no sample, so this is what makes the toggle
                        // show up without camera motion.
                        ctx.start_render = true;
                    }
                    if (unavailable) ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                        ImGui::SetTooltip("%s", dv_tips[i]);
                    }
                }
                ImGui::EndCombo();
            }
            if (dv != 0) {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextDisabled("%s", dv_tips[dv]);
                ImGui::PopTextWrapPos();
            }
            if (ctx.render_settings.debug_view != 0) {
                ImGui::Indent();
                if (ImGui::DragFloat("Exposure##dbgexp", &ctx.render_settings.debug_exposure, 0.05f, 0.0f, 1000.0f, "%.2f"))
                    ctx.start_render = true;
                if (ImGui::SliderFloat("Overlay Beauty##dbgovl", &ctx.render_settings.debug_overlay, 0.0f, 1.0f, "%.2f"))
                    ctx.start_render = true;
                UIWidgets::HelpMarker("0 = pure debug view, 1 = pure beauty — mix to see the\n"
                                      "debug data ghosted over the rendered image.");
                if (ctx.render_settings.use_optix) {
                    UIWidgets::StatusIndicator("Debug views run on the Vulkan RT backend only",
                                               UIWidgets::StatusType::Warning);
                }
                ImGui::Unindent();
            }
        }

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("Denoiser (Viewport + Final + Sequence)", ImVec4(0.90f, 0.76f, 1.0f, 1.0f));
        if (ImGui::Checkbox("Enable Denoiser", &ctx.render_settings.use_denoiser)) {
            // Mirror to legacy 'render_use_denoiser' so the final single-frame
            // render path (Main.cpp ~4049 / 4382) stays in sync without
            // exposing a separate checkbox.
            ctx.render_settings.render_use_denoiser = ctx.render_settings.use_denoiser;
        }
        UIWidgets::HelpMarker("Applies to interactive viewport, single-frame final render, "
                              "AND sequence render output. Sequence frames are denoised "
                              "before being saved to disk.");
        if (ctx.render_settings.use_denoiser) {
            const char* denoiser_mode_items[] = {
                "Beauty only",
                "Beauty + Albedo + Normal"
            };
            int denoiser_mode = static_cast<int>(ctx.render_settings.denoiser_mode);
            if (ImGui::Combo("Input Buffers", &denoiser_mode, denoiser_mode_items, IM_ARRAYSIZE(denoiser_mode_items))) {
                ctx.render_settings.denoiser_mode = static_cast<DenoiserMode>(denoiser_mode);
            }
            const char* denoiser_quality_items[] = {
                "Fast (viewport)",
                "Balanced",
                "High (final)"
            };
            int denoiser_quality = static_cast<int>(ctx.render_settings.denoiser_quality);
            if (ImGui::Combo("Denoiser Quality", &denoiser_quality, denoiser_quality_items, IM_ARRAYSIZE(denoiser_quality_items))) {
                ctx.render_settings.denoiser_quality = static_cast<DenoiserQuality>(denoiser_quality);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("OIDN model tier for the viewport. Fast is the cheapest\n(dominant cost on the GPU). Final renders always use High.");
            }
            ImGui::SliderFloat("Blend Factor", &ctx.render_settings.denoiser_blend_factor, 0.0f, 1.0f);
        } else {
            ImGui::TextDisabled("Denoiser is disabled.");
        }
        UIWidgets::EndSection();
    }

    // Resolution & Output controls moved to the System panel (single source of truth).

    DrawRenderWindowToneMapControls(ctx);

    if (UIWidgets::BeginSection("Animation Render & Export", ImVec4(1.0f, 0.4f, 0.7f, 1.0f))) {
        if (rendering_in_progress && ctx.is_animation_mode) {
            ImVec4 status_bg = ImVec4(0.15f, 0.15f, 0.2f, 1.0f);
            float status_round = 10.0f;
            if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
                const auto& curTheme = ThemeManager::instance().current();
                status_bg = ImVec4(curTheme.colors.surface.x, curTheme.colors.surface.y, curTheme.colors.surface.z, 1.0f);
                status_round = curTheme.style.windowRounding;
            }
            ImGui::PushStyleColor(ImGuiCol_ChildBg, status_bg);
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, status_round);
            ImGui::BeginChild("AnimRenderStatus", ImVec2(0, 126), true);

            const int cur = ctx.render_settings.animation_current_frame;
            const int start = ctx.render_settings.animation_start_frame;
            const int end = ctx.render_settings.animation_end_frame;
            const int total = end - start + 1;
            const int done = cur - start;
            const float progress = (total > 0) ? (float)done / (float)total : 0.0f;

            UIWidgets::StatusIndicator("Animation sequence rendering in progress", UIWidgets::StatusType::Warning);
            ImGui::Spacing();
            char prog_text[64];
            std::snprintf(prog_text, sizeof(prog_text), "Frame %d / %d  (%.0f%%)", cur, end, progress * 100.0f);
            UIWidgets::ProgressBarEx(progress, ImVec2(-1, 24), prog_text, ImVec4(1.0f, 0.65f, 0.20f, 1.0f));
            ImGui::Spacing();
            ImGui::Text("Samples per frame: %d", ctx.render_settings.animation_samples_per_frame);
            ImGui::Text("Playback FPS: %d", ctx.render_settings.animation_fps);
            ImGui::TextWrapped("Output Folder: %s", ctx.render_settings.animation_output_folder.c_str());
            UIWidgets::PushControlSurfaceStyle(ImVec4(1.0f, 0.42f, 0.38f, 1.0f));
            if (UIWidgets::IconActionButton("StopAnimationRender", UIWidgets::IconType::Stop, "Stop Rendering",
                false, ImVec4(1.0f, 0.42f, 0.38f, 1.0f), ImVec2(148.0f, 42.0f),
                "Stop the active animation sequence render.")) {
                rendering_stopped_cpu = true;
                rendering_stopped_gpu = true;
                SCENE_LOG_WARN("Animation render stop requested by user.");
            }
            UIWidgets::PopControlSurfaceStyle();

            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
        } else {
            UIWidgets::ColoredHeader("Sequence Range", ImVec4(1.0f, 0.80f, 0.92f, 1.0f));
            ImGui::PushItemWidth(80.0f);
            ImGui::DragInt("Start", &ctx.render_settings.animation_start_frame, 1, 0, ctx.render_settings.animation_end_frame);
            ImGui::SameLine();
            ImGui::DragInt("End", &ctx.render_settings.animation_end_frame, 1, ctx.render_settings.animation_start_frame, 10000);
            ImGui::SameLine();
            ImGui::DragInt("FPS", &ctx.render_settings.animation_fps, 1, 1, 120);
            ImGui::PopItemWidth();
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
            UIWidgets::ColoredHeader("Sequence Quality", ImVec4(1.0f, 0.74f, 0.86f, 1.0f));
            // Sequence render now reuses the main "Samples" render-quality control
            // (final_render_samples). Show a read-only mirror here so users see
            // which quality their sequence will use without duplicating the
            // setting in two places.
            ImGui::TextDisabled("Uses main Samples setting: %d / frame",
                ctx.render_settings.final_render_samples);
            UIWidgets::HelpMarker("Sequence render uses the same samples-per-frame value "
                                  "as the main Render Settings 'Samples' field. Change it "
                                  "there to adjust sequence quality.");

            UIWidgets::Divider();
            UIWidgets::ColoredHeader("Sequence Output", ImVec4(1.0f, 0.78f, 0.82f, 1.0f));
            char folder_buf[512] = {};
            std::snprintf(folder_buf, sizeof(folder_buf), "%s", ctx.render_settings.animation_output_folder.c_str());
            ImGui::PushItemWidth(-56.0f);
            if (ImGui::InputText("##outdir", folder_buf, sizeof(folder_buf))) {
                ctx.render_settings.animation_output_folder = folder_buf;
            }
            ImGui::PopItemWidth();
            UIWidgets::PushControlSurfaceStyle(ImVec4(1.0f, 0.60f, 0.56f, 1.0f));
            if (UIWidgets::IconActionButton("BrowseAnimOutput", UIWidgets::IconType::Assets, "Browse",
                false, ImVec4(1.0f, 0.60f, 0.56f, 1.0f), ImVec2(104.0f, 38.0f),
                "Choose the output folder for PNG sequence export.")) {
                std::string path = selectFolderDialogW(L"Select Animation Output Folder");
                if (!path.empty()) {
                    ctx.render_settings.animation_output_folder = path;
                }
            }
            UIWidgets::PopControlSurfaceStyle();

            const int total_frames = ctx.render_settings.animation_end_frame - ctx.render_settings.animation_start_frame + 1;
            const int samples = ctx.render_settings.animation_samples_per_frame;
            const float est_time_per_frame = (samples / 64.0f) * 2.0f;
            const float est_total_minutes = (est_time_per_frame * total_frames) / 60.0f;
            const bool can_render = !ctx.render_settings.animation_output_folder.empty();
            const bool valid_range = (ctx.render_settings.animation_end_frame >= ctx.render_settings.animation_start_frame);

            ImVec4 summary_bg = ImVec4(0.10f, 0.11f, 0.15f, 0.95f);
            float summary_round = 10.0f;
            if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
                const auto& curTheme = ThemeManager::instance().current();
                summary_bg = ImVec4(curTheme.colors.surface.x, curTheme.colors.surface.y, curTheme.colors.surface.z, 0.95f);
                summary_round = curTheme.style.windowRounding;
            }
            ImGui::PushStyleColor(ImGuiCol_ChildBg, summary_bg);
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, summary_round);
            ImGui::BeginChild("AnimRenderSummary", ImVec2(0, 72), true);
            UIWidgets::StatusIndicator(
                can_render && valid_range ? "Sequence is ready to render" : "Sequence needs attention before rendering",
                can_render && valid_range ? UIWidgets::StatusType::Success : UIWidgets::StatusType::Warning);
            ImGui::Text("%d frames x %d spp  |  est. %.1f min", total_frames, samples, est_total_minutes);
            // Sequence render uses the active viewport resolution.
            ImGui::Text("Resolution: %dx%d  (viewport)", image_width, image_height);
            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();

            if (!can_render) {
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "Set an output folder before starting the sequence render.");
            }
            if (!valid_range) {
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "Frame range is invalid. End frame must be greater than or equal to start frame.");
            }

            UIWidgets::PushControlSurfaceStyle(ImVec4(1.0f, 0.40f, 0.66f, 1.0f));
            if (UIWidgets::IconActionButton("RenderAnimationSequence", UIWidgets::IconType::Render, "Render Sequence",
                false, ImVec4(1.0f, 0.40f, 0.66f, 1.0f), ImVec2(164.0f, 44.0f),
                "Start rendering the animation as a PNG image sequence.", can_render && valid_range)) {
                ctx.render_settings.start_animation_render = true;
                ctx.render_settings.animation_total_frames = total_frames;
                SCENE_LOG_INFO("Animation render triggered: " + std::to_string(total_frames) + " frames @ " + std::to_string(samples) + " samples");
            }
            UIWidgets::PopControlSurfaceStyle();
        }

        UIWidgets::EndSection();
    }

    ImGui::PopStyleColor(9);
    ImGui::PopStyleVar(7);
    UIWidgets::PopControlSurfaceStyle();
}



// drawWorldContent moved to scene_ui_world.cpp

// ─────────────────────────────────────────────────────────────────────────────
// DETACHABLE (TEAR-OFF) PROPERTIES SUB-TABS
// Any sidebar tab can be dragged out of the rail into its own dockable/closable
// window so several panels are usable at once. Content renders in exactly ONE place
// (main panel XOR popped window), so side effects never double-fire.
// ─────────────────────────────────────────────────────────────────────────────
static bool isPoppablePropertyTab(int tab)
{
    return tab >= 0 && tab <= 13; // every sidebar tab index
}

static const char* poppablePropertyTabName(int tab)
{
    switch (tab) {
        case 0:  return "Scene";
        case 1:  return "Render Settings";
        case 2:  return "Terrain";
        case 3:  return "Water";
        case 4:  return "Volumetric";
        case 5:  return "Simulation";
        case 6:  return "World";
        case 7:  return "Modeling";
        case 8:  return "Hair & Fur";
        case 9:  return "System";
        case 10: return "Paint";
        case 11: return "Scatter";
        case 12: return "Stylize";
        case 13: return "Sculpting";
        default: return "Panel";
    }
}

// Renders a single tab's content. This is the single source of truth shared by the
// main Properties switch and the torn-off windows (kept in lockstep with that switch).
void SceneUI::drawPoppedTabContent(UIContext& ctx, int tab)
{
    switch (tab) {
        case 0:  drawSceneHierarchy(ctx); break;
        case 1:  drawRenderInspectorContent(ctx); break;
        case 2:  if (show_terrain_tab) drawTerrainPanel(ctx); break;
        case 3:
            if (show_water_tab) {
                if (ImGui::Button("Water##WaterSubtabPop")) active_water_subtab = 0;
                ImGui::SameLine();
                if (ImGui::Button("River##WaterSubtabPop")) active_water_subtab = 1;
                ImGui::Separator();
                if (active_water_subtab == 0) drawWaterPanel(ctx);
                else                          drawRiverPanel(ctx);
            }
            break;
        case 4:  if (show_volumetric_tab) drawVolumetricPanel(ctx); break;
        case 5:  if (show_forcefield_tab) ForceFieldUI::drawForceFieldPanel(*this, ctx, ctx.scene, &timeline); break;
        case 6:  if (show_world_tab) drawWorldContent(ctx); break;
        case 7:  drawModifiersPanel(ctx); break;
        case 8:  if (show_hair_tab) drawHairTabContent(ctx); break;
        case 9:  drawThemeSelector(); drawResolutionPanel(ctx); break;
        case 10: if (show_paint_tab) drawPaintPanel(ctx); break;
        case 11: if (show_scatter_tab) drawScatterBrushPanel(ctx); break;
        case 12: if (show_stylize_tab) drawStylizePanel(ctx); break;
        case 13: drawSculptPanel(ctx); break;
        default: break;
    }
}

// Hosts every currently popped tab as its own dockable/closable window. Closing a
// window pops the tab back into the main Properties panel.
void SceneUI::drawPoppedPropertyWindows(UIContext& ctx)
{
    // Works with or without docking: with docking these windows are dockable,
    // otherwise they simply float.
    for (int tab = 0; tab < 16; ++tab) {
        if (!properties_tab_popped_[tab] || !isPoppablePropertyTab(tab))
            continue;

        bool open = true;
        ImGui::SetNextWindowSize(ImVec2(380, 540), ImGuiCond_FirstUseEver);
        // Fresh button pop -> appear at the cursor. Serialized restore -> keep the
        // position imgui.ini already holds for this window.
        if (properties_pop_spawn_pending_[tab]) {
            ImGui::SetNextWindowPos(properties_pop_spawn_pos_[tab], ImGuiCond_Always);
            properties_pop_spawn_pending_[tab] = false;
        }
        std::string title = std::string(poppablePropertyTabName(tab)) + "###prop_pop_" + std::to_string(tab);
        if (ImGui::Begin(title.c_str(), &open, ImGuiWindowFlags_NoCollapse)) {
            ImGui::PushItemWidth(UIWidgets::GetInspectorItemWidth());
            UIWidgets::PushControlSurfaceStyle(ImVec4(0.62f, 0.74f, 0.98f, 1.0f)); // same modern surface as the docked panel
            drawPoppedTabContent(ctx, tab);
            UIWidgets::PopControlSurfaceStyle();
            ImGui::PopItemWidth();
        }
        ImGui::End();

        if (!open)
            properties_tab_popped_[tab] = false; // closed -> back into the main panel
    }
}

// Hair & Fur tab body. Extracted from the main Properties switch so the popped
// window can reuse it verbatim (function-static caches persist as before).
void SceneUI::drawHairTabContent(UIContext& ctx)
{
    // Get selected mesh triangles for hair generation target
    static std::vector<std::shared_ptr<Triangle>> selectedMeshTriangles;
    static std::string lastSelectedMeshName;
    const std::vector<std::shared_ptr<Triangle>>* selectedTris = nullptr;

    // Check if we have a selected object
    bool hasValidSelection = (ctx.selection.selected.type == SelectableType::Object &&
                             ctx.selection.selected.object != nullptr);

    if (hasValidSelection) {
        std::string selectedNodeName = ctx.selection.selected.object->getNodeName();
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
        this->ensureCPUSyncForPicking(ctx);
    });

    ctx.renderer.hideInterpolatedHair = hairUI.shouldHideChildren();

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

    if (materialChanged) {
        ctx.renderer.setHairMaterial(currentMaterial);
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterials(ctx.scene);
            ctx.start_render = true;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MODERN DOCKABLE LAYOUT
// A full-viewport host window owns an ImGui DockSpace with a pass-through central
// node (so the SDL-rendered 3D viewport shows through). Panels (Properties,
// BottomPanel, and any floating tool window) dock into it and become freely
// movable / tabbable / closable.
//
// The 3D viewport is NOT an ImGui window: picking / gizmos / overlays derive their
// region from side_panel_width / bottom_panel_height / menu_height. To keep those
// aligned no matter where panels are docked, we read the *central node* rect every
// frame and feed it back into those legacy members.
// ─────────────────────────────────────────────────────────────────────────────
void SceneUI::drawDockSpaceHost(UIContext& ctx)
{
    if (!docking_enabled) return;

    ImGuiIO& io = ImGui::GetIO();
    const float menu_height = getMainMenuReservedHeight();
    const float status_bar_height = 24.0f;

    const ImVec2 host_pos(0.0f, menu_height);
    const ImVec2 host_size(io.DisplaySize.x,
                           (std::max)(0.0f, io.DisplaySize.y - menu_height - status_bar_height));

    ImGui::SetNextWindowPos(host_pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(host_size, ImGuiCond_Always);

    const ImGuiWindowFlags host_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("##RayTrophiDockHost", nullptr, host_flags);
    ImGui::PopStyleVar(3);

    // NOTE: bump this string whenever the default layout changes — a new id has no
    // persisted node in imgui.ini, so the correct default rebuilds and any stale
    // user layout (e.g. an old full-width bottom panel) is discarded automatically.
    const ImGuiID dockspace_id = ImGui::GetID("RayTrophiDockSpace_v3");
    this->dockspace_id = dockspace_id;

    // (Re)build the default layout once, or when the user requests a reset.
    if (docking_layout_dirty || ImGui::DockBuilderGetNode(dockspace_id) == nullptr) {
        docking_layout_dirty = false;
        ImGui::DockBuilderRemoveNode(dockspace_id);
        ImGui::DockBuilderAddNode(dockspace_id,
            ImGuiDockNodeFlags_DockSpace | ImGuiDockNodeFlags_PassthruCentralNode);
        ImGui::DockBuilderSetNodeSize(dockspace_id, host_size);

        const float left_ratio = (host_size.x > 1.0f)
            ? std::clamp(side_panel_width / host_size.x, 0.12f, 0.45f) : 0.22f;
        const float bottom_ratio = (host_size.y > 1.0f)
            ? std::clamp(bottom_panel_height / host_size.y, 0.12f, 0.50f) : 0.26f;

        ImGuiID dock_main = dockspace_id;
        ImGuiID dock_left = ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Left, left_ratio, nullptr, &dock_main);
        ImGuiID dock_bottom = ImGui::DockBuilderSplitNode(dock_main, ImGuiDir_Down, bottom_ratio, nullptr, &dock_main);

        ImGui::DockBuilderDockWindow("Properties", dock_left);
        ImGui::DockBuilderDockWindow("Timeline", dock_bottom);
        ImGui::DockBuilderDockWindow("Console", dock_bottom);
        ImGui::DockBuilderDockWindow("Terrain Graph", dock_bottom);
        ImGui::DockBuilderDockWindow("Geometry Graph", dock_bottom);
        ImGui::DockBuilderDockWindow("AnimGraph", dock_bottom);
        ImGui::DockBuilderDockWindow("Asset Browser", dock_bottom);
        ImGui::DockBuilderFinish(dockspace_id);
        dock_bottom_id = dock_bottom;
    } else {
        // Fallback: scan active bottom windows to retrieve dock_bottom_id when loaded from ini
        for (const char* name : {"Timeline", "Console", "Terrain Graph", "Geometry Graph", "AnimGraph", "Asset Browser"}) {
            ImGuiWindow* win = ImGui::FindWindowByName(name);
            if (win && win->DockNode) {
                ImGuiDockNode* node = win->DockNode;
                while (node->ParentNode && node->ParentNode->ID != dockspace_id) {
                    node = node->ParentNode;
                }
                if (node->ParentNode && node->ParentNode->ID == dockspace_id) {
                    dock_bottom_id = node->ID;
                    break;
                }
            }
        }
    }

    if (dock_bottom_id != 0) {
        if (ImGuiDockNode* node = ImGui::DockBuilderGetNode(dock_bottom_id)) {
            node->LocalFlags |= ImGuiDockNodeFlags_AutoHideTabBar;
        }
    }

    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

    // Keep viewport picking / gizmos / overlays aligned: the central (empty) node IS
    // the 3D viewport region, so mirror its rect into the legacy offset members.
    if (ImGuiDockNode* central = ImGui::DockBuilderGetCentralNode(dockspace_id)) {
        const float left_inset = (std::max)(0.0f, central->Pos.x);
        const float central_bottom = central->Pos.y + central->Size.y;
        const float host_bottom = host_pos.y + host_size.y;

        if (showSidePanel)
            side_panel_width = left_inset;

        const float derived_bottom = (std::max)(0.0f, host_bottom - central_bottom);
        if (derived_bottom > 1.0f) {
            bottom_panel_height = derived_bottom;
            preferred_bottom_panel_height = derived_bottom;
        }
    }

    ImGui::End();
}
void SceneUI::dockToBottom(const char* window_name)
{
    if (docking_enabled && dock_bottom_id != 0) {
        ImGuiWindow* win = ImGui::FindWindowByName(window_name);
        bool already_docked = false;
        if (win && win->DockNode) {
            ImGuiDockNode* node = win->DockNode;
            while (node->ParentNode && node->ParentNode->ID != dockspace_id) {
                node = node->ParentNode;
            }
            if (node->ID == dock_bottom_id) {
                already_docked = true;
            }
        }
        if (!already_docked) {
            ImGui::DockBuilderDockWindow(window_name, dock_bottom_id);
        }
    }
}

void SceneUI::drawRenderSettingsPanel(UIContext& ctx, float screen_y)
{
    const float menu_height = getMainMenuReservedHeight();
    float status_bar_height = 24.0f;
    // Keep the properties panel full-height so it can overlap the bottom panel when focused.
    float target_height = screen_y - menu_height - status_bar_height;

    // Panel ayarları
    // Lock Height to target_height (MinY = MaxY), allow Width resize (300-800).
    // When docking is enabled the dock node owns the geometry, so we skip the lock.
    if (!docking_enabled) {
        ImGui::SetNextWindowSizeConstraints(
            ImVec2(300, target_height),
            ImVec2(800, target_height)
        );
    }

    // LEFT SIDE DOCKING
    ImGuiIO& io = ImGui::GetIO();
    
    // Legacy pinned layout: hard-anchor at (0, menu_height) with no title bar.
    // Docking layout: let the dock node place/size the window; keep the title bar
    // as the drag/tab handle.
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar;
    if (!docking_enabled) {
        ImGui::SetNextWindowPos(ImVec2(0, menu_height), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(side_panel_width, target_height), ImGuiCond_FirstUseEver);
        flags |= ImGuiWindowFlags_NoMove;
    }
    if (focus_properties_panel_next_frame) {
        ImGui::SetNextWindowFocus();
    }

    // Panel shell styling
    float win_rounding = 0.0f;
    ImVec4 win_bg = ImVec4(0.085f, 0.09f, 0.105f, panel_alpha);
    ImVec4 win_border = ImVec4(0.46f, 0.54f, 0.64f, 0.18f);

    if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
        const auto& curTheme = ThemeManager::instance().current();
        win_rounding = curTheme.style.windowRounding;
        win_bg = ImVec4(curTheme.colors.background.x, curTheme.colors.background.y, curTheme.colors.background.z, panel_alpha);
        win_border = curTheme.colors.border;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, win_rounding);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, win_bg);
    ImGui::PushStyleColor(ImGuiCol_Border, win_border);

    if (ImGui::Begin("Properties", nullptr, flags))
    {
        if (focus_properties_panel_next_frame) {
            ImGui::SetWindowFocus();
            focus_properties_panel_next_frame = false;
        }
        ImDrawList* parent_dl = ImGui::GetWindowDrawList();
        ImVec2 win_pos = ImGui::GetWindowPos();
        ImVec2 win_size = ImGui::GetWindowSize();
        // Update width if user resized. Under docking the viewport left inset is
        // derived from the dock central node (drawDockSpaceHost), so this window's
        // width is NOT authoritative — the panel may be docked right/floating/closed.
        if (!docking_enabled)
            side_panel_width = ImGui::GetWindowWidth();

        // ─────────────────────────────────────────────────────────────────────────
        // MODERN VERTICAL TAB NAVIGATION
        // ─────────────────────────────────────────────────────────────────────────

        // Sync tab_to_focus with vertical tabs
        if (tab_to_focus == "Scene Edit") { active_properties_tab = 0; tab_to_focus = ""; }
        if (tab_to_focus == "Render")     { active_properties_tab = 1; tab_to_focus = ""; }
        if (tab_to_focus == "Terrain")    { active_properties_tab = 2; tab_to_focus = ""; }
        if (tab_to_focus == "Scatter")    { active_properties_tab = 11; tab_to_focus = ""; }
        if (tab_to_focus == "Water")      { active_properties_tab = 3; tab_to_focus = ""; }
        if (tab_to_focus == "Volumetric" || tab_to_focus == "VDB" || tab_to_focus == "Gas") { active_properties_tab = 4; tab_to_focus = ""; }
        if (tab_to_focus == "Force Field" || tab_to_focus == "Simulation"){ active_properties_tab = 5; tab_to_focus = ""; }
        if (tab_to_focus == "World")      { active_properties_tab = 6; tab_to_focus = ""; }
        if (tab_to_focus == "Modifiers" || tab_to_focus == "Modeling")  { active_properties_tab = 7; tab_to_focus = ""; }
        if (tab_to_focus == "Paint")      { active_properties_tab = 10; tab_to_focus = ""; }
        if (tab_to_focus == "System")     { active_properties_tab = 9; tab_to_focus = ""; }

        static float s_sidebar_width = 44.0f;
        float sidebar_width = s_sidebar_width;
        
        // --- 1. SIDEBAR (Fixed Width) ---
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 4));
        
        // Sidebar background
        ImVec4 sidebarBg = ThemeManager::instance().current().colors.secondary;
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
            static float hover_anim[16] = {};
            ImGui::PushID(index);
            
            const float size = (std::max)(28.0f, sidebar_width - 8.0f);
            float margin = (sidebar_width - size) * 0.5f;

            ImGui::SetCursorPosX(margin);
            ImVec2 pos = ImGui::GetCursorScreenPos();

            auto getHoverTint = [&](UIWidgets::IconType iconType) -> ImVec4 {
                switch (iconType) {
                    case UIWidgets::IconType::Scene:      return ImVec4(0.82f, 0.88f, 1.00f, 1.0f);
                    case UIWidgets::IconType::Render:     return ImVec4(1.00f, 0.84f, 0.42f, 1.0f);
                    case UIWidgets::IconType::Terrain:    return ImVec4(0.56f, 0.90f, 0.47f, 1.0f);
                    case UIWidgets::IconType::Water:      return ImVec4(0.38f, 0.78f, 1.00f, 1.0f);
                    case UIWidgets::IconType::Volumetric: return ImVec4(0.82f, 0.92f, 1.00f, 1.0f);
                    case UIWidgets::IconType::Force:      return ImVec4(1.00f, 0.66f, 0.34f, 1.0f);
                    case UIWidgets::IconType::World:      return ImVec4(0.44f, 0.88f, 0.92f, 1.0f);
                    case UIWidgets::IconType::Brush:      return ImVec4(0.88f, 0.70f, 0.42f, 1.0f);
                    case UIWidgets::IconType::Hair:       return ImVec4(1.00f, 0.73f, 0.42f, 1.0f);
                    case UIWidgets::IconType::SprayTool:  return ImVec4(0.58f, 0.96f, 0.52f, 1.0f);
                    case UIWidgets::IconType::Sculpt:     return ImVec4(1.00f, 0.56f, 0.38f, 1.0f);
                    case UIWidgets::IconType::PaintTool:  return ImVec4(1.00f, 0.48f, 0.50f, 1.0f);
                    case UIWidgets::IconType::System:     return ImVec4(0.84f, 0.86f, 0.94f, 1.0f);
                    default:                              return ImVec4(0.50f, 0.92f, 0.72f, 1.0f);
                }
            };

            const ImVec4 accent = getHoverTint(icon);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            float child_left = ImGui::GetWindowPos().x;

            // Sleek left-side vertical accent line for active tab (Blender style)
            if (is_active) {
                dl->AddRectFilled(
                    ImVec2(child_left + 2.0f, pos.y + 3.0f),
                    ImVec2(child_left + 6.0f, pos.y + size - 3.0f),
                    ImGui::ColorConvertFloat4ToU32(accent),
                    2.0f
                );
            }

            // Flat invisible button for interaction
            static float hold_timers[16] = {};
            static float flash_timers[16] = {};

            if (flash_timers[index] > 0.0f) {
                flash_timers[index] -= ImGui::GetIO().DeltaTime * 2.0f;
                if (flash_timers[index] < 0.0f) flash_timers[index] = 0.0f;
            }

            if (ImGui::InvisibleButton("##tab", ImVec2(size, size))) {
                if (hold_timers[index] < 2.0f) {
                    active_properties_tab = index;
                    focus_properties_panel_next_frame = true;
                }
            }

            const bool is_hovered = ImGui::IsItemHovered();
            const bool is_held = ImGui::IsItemActive();

            // 2-second hold-to-pop logic
            if (is_held && isPoppablePropertyTab(index) && !properties_tab_popped_[index]) {
                hold_timers[index] += ImGui::GetIO().DeltaTime;
                if (hold_timers[index] >= 2.0f) {
                    properties_tab_popped_[index] = true;
                    properties_pop_spawn_pos_[index] = ImGui::GetIO().MousePos;
                    properties_pop_spawn_pending_[index] = true;
                    flash_timers[index] = 1.0f;
                    hold_timers[index] = 0.0f;
                }
            } else {
                if (hold_timers[index] > 0.0f) {
                    hold_timers[index] -= ImGui::GetIO().DeltaTime * 4.0f;
                    if (hold_timers[index] < 0.0f) hold_timers[index] = 0.0f;
                }
            }

            // Draw a circular progress arc as the tab is held
            if (hold_timers[index] > 0.01f && !properties_tab_popped_[index]) {
                float progress = hold_timers[index] / 2.0f;
                float endAngle = -1.5707f + progress * 6.2831f;
                dl->PathClear();
                dl->PathArcTo(ImVec2(pos.x + size*0.5f, pos.y + size*0.5f), size * 0.44f, -1.5707f, endAngle, 24);
                dl->PathStroke(IM_COL32(0, 160, 255, 230), false, 2.5f);
            }

            // Flash the properties panel border when triggered
            if (flash_timers[index] > 0.0f) {
                int alpha = (int)(255.0f * flash_timers[index]);
                parent_dl->AddRect(
                    win_pos,
                    ImVec2(win_pos.x + win_size.x, win_pos.y + win_size.y),
                    IM_COL32(0, 160, 255, alpha),
                    0.0f,
                    0,
                    3.0f
                );
            }

            // Right-click and Double-click popping logic
            if (is_hovered && isPoppablePropertyTab(index)) {
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    properties_tab_popped_[index] = !properties_tab_popped_[index];
                    if (properties_tab_popped_[index]) {
                        properties_pop_spawn_pos_[index] = ImGui::GetIO().MousePos;
                        properties_pop_spawn_pending_[index] = true;
                    }
                }
                
                if (ImGui::BeginPopupContextItem("##tab_context")) {
                    const bool popped = properties_tab_popped_[index];
                    if (popped) {
                        if (ImGui::MenuItem("Dock Back to Panel")) {
                            properties_tab_popped_[index] = false;
                        }
                    } else {
                        if (ImGui::MenuItem("Open in Separate Window")) {
                            properties_tab_popped_[index] = true;
                            properties_pop_spawn_pos_[index] = ImGui::GetIO().MousePos;
                            properties_pop_spawn_pending_[index] = true;
                        }
                    }
                    ImGui::EndPopup();
                }
            }

            const float target_hover = (is_hovered || is_active) ? 1.0f : 0.0f;
            const float anim_speed = 12.0f * ImGui::GetIO().DeltaTime;
            hover_anim[index] += (target_hover - hover_anim[index]) * ImClamp(anim_speed, 0.0f, 1.0f);

            // Beautiful background highlight using tab's accent color for active, or soft white for hover
            if (hover_anim[index] > 0.01f) {
                ImVec4 highlightBg = is_active 
                    ? ImVec4(accent.x, accent.y, accent.z, 0.24f) 
                    : ImVec4(1.0f, 1.0f, 1.0f, 0.06f * hover_anim[index]);
                dl->AddRectFilled(
                    pos,
                    ImVec2(pos.x + size, pos.y + size),
                    ImGui::ColorConvertFloat4ToU32(highlightBg),
                    6.0f
                );
            }

            // Accent-colored subtle border around active tab to guarantee visibility
            if (is_active) {
                dl->AddRect(
                    pos,
                    ImVec2(pos.x + size, pos.y + size),
                    ImGui::ColorConvertFloat4ToU32(ImVec4(accent.x, accent.y, accent.z, 0.35f)),
                    6.0f,
                    0,
                    1.2f
                );
            }

            // Shaded 3D vector icons (scaled proportionally with sidebar size)
            const float base_icon_size = size * 0.95f;
            const float iconSize = base_icon_size + (size * 0.04f) * hover_anim[index];
            ImVec4 idleTint(0.42f, 0.44f, 0.48f, 0.60f); // Dimmed inactive tabs for high contrast
            ImVec4 activeTint = accent;
            ImVec4 iconTint = is_active
                ? activeTint
                : ImLerp(idleTint, accent, hover_anim[index]);

            UIWidgets::DrawIcon(
                icon,
                ImVec2(
                    pos.x + (size - iconSize) * 0.5f,
                    pos.y + (size - iconSize) * 0.5f),
                iconSize,
                ImGui::ColorConvertFloat4ToU32(iconTint),
                1.35f + 0.15f * hover_anim[index]
            );

            if (is_hovered) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.07f, 0.08f, 0.10f, 0.84f));
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(iconTint.x, iconTint.y, iconTint.z, 0.22f));
                float tooltip_round = 10.0f;
                if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
                    tooltip_round = ThemeManager::instance().current().style.popupRounding;
                }
                ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, tooltip_round);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 22.0f);
                ImGui::TextUnformatted(tooltip);
                if (isPoppablePropertyTab(index)) {
                    ImGui::Spacing();
                    if (properties_tab_popped_[index]) {
                        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.20f, 1.0f), "(Separate Window - Hold 2s, Double-click, or Right-click to dock back)");
                    } else {
                        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 0.85f), "(Hold 2s, Double-click, or Right-click to open in separate window)");
                    }
                }
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(2);
            }
            
            ImGui::PopID();
        };

        ImGui::Dummy(ImVec2(0.0f, 10.0f));
        // 1. Scene Setup & Config
        drawTabButton(0, UIWidgets::IconType::Scene,      "Scene / Hierarchy");
        if (show_world_tab)      drawTabButton(6, UIWidgets::IconType::World,      "World & Sky");
        drawTabButton(1, UIWidgets::IconType::Render,     "Render Settings");
        
        ImGui::Dummy(ImVec2(0.0f, 4.0f)); // Subtle grouping spacing
        
        // 2. Geometry Creation & Modeling / Sculpting / Hair
        drawTabButton(7, UIWidgets::IconType::Mesh, "Modeling");
        drawTabButton(13, UIWidgets::IconType::Sculpt, "Sculpting");
        if (show_paint_tab)      drawTabButton(10, UIWidgets::IconType::PaintTool,  "Paint Mode");
        if (show_hair_tab)       drawTabButton(8, UIWidgets::IconType::Hair,       "Hair & Fur");
        
        ImGui::Dummy(ImVec2(0.0f, 4.0f));
        
        // 3. Environment & Landscape
        if (show_terrain_tab)    drawTabButton(2, UIWidgets::IconType::Terrain,    "Terrain Editor");
        if (show_water_tab)      drawTabButton(3, UIWidgets::IconType::Water,      active_water_subtab == 0 ? "Water" : "River Spline");
        if (show_scatter_tab)    drawTabButton(11, UIWidgets::IconType::SprayTool, "Scatter (Foliage & Mesh)");
        
        ImGui::Dummy(ImVec2(0.0f, 4.0f));
        
        // 4. Dynamics, Effects & Styling
        if (show_forcefield_tab) drawTabButton(5, UIWidgets::IconType::Force,      "Physics");
        if (show_volumetric_tab) drawTabButton(4, UIWidgets::IconType::Volumetric, "Volumetrics");
        if (show_stylize_tab)    drawTabButton(12, UIWidgets::IconType::Brush,     "Stylize Mode");
        
        ImGui::Dummy(ImVec2(0.0f, 4.0f));
        
        // 5. System
        if (show_system_tab)     drawTabButton(9, UIWidgets::IconType::System,     "System & UI");
        
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);

        // Sidebar Splitter / Resizer (Invisible drag line on the right edge of sidebar)
        ImGui::SameLine(0, 0);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
        ImGui::Button("##sidebar_splitter", ImVec2(4.0f, win_size.y));
        ImGui::PopStyleColor(3);

        if (ImGui::IsItemActive()) {
            s_sidebar_width += ImGui::GetIO().MouseDelta.x;
            if (s_sidebar_width < 36.0f) s_sidebar_width = 36.0f;
            if (s_sidebar_width > 80.0f) s_sidebar_width = 80.0f;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        }
        
        ImGui::SameLine(0, 0);
        
        // --- 2. CONTENT AREA (Inspector Style) ---
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ThemeManager::instance().current().colors.background);
        
        ImGui::BeginChild("PropContentArea", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar);
        

        // ── MAIN CONTENT (Flush Scroll Area) ──
        ImGui::BeginChild("PropScrollArea", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);
        // Reset scroll-Y when the active properties tab changes so each tab
        // opens at the top instead of inheriting the previous tab's scroll.
        {
            static int s_last_properties_tab = -1;
            if (s_last_properties_tab != active_properties_tab) {
                ImGui::SetScrollY(0.0f);
                s_last_properties_tab = active_properties_tab;
            }
        }
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 6));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 14));
        
        // Start Global Indent for controls (leaving headers flush)
        ImGui::Indent(10.0f); 
        ImGui::Spacing();
        ImGui::Unindent(10.0f);

        // --- CAPPED ITEM WIDTH ---
        // Prevents sliders/inputs from stretching too far on wide panels, keeping labels legible
        ImGui::PushItemWidth(UIWidgets::GetInspectorItemWidth());

        // Unified modern/soft control surface for EVERY tab (matches the Render
        // Settings look). The theme still drives window/text colors; this only
        // softens buttons/frames/sliders inside the content area. Render's own
        // inner Push/Pop simply nests on top (balanced).
        UIWidgets::PushControlSurfaceStyle(ImVec4(0.62f, 0.74f, 0.98f, 1.0f));

        // --- Detachable tab control (every editor tab) ---
        const bool tab_is_poppable = isPoppablePropertyTab(active_properties_tab);
        const bool tab_is_popped = tab_is_poppable && properties_tab_popped_[active_properties_tab];
        if (tab_is_popped) {
            ImGui::Spacing();
            ImGui::Indent(10.0f);
            ImGui::TextDisabled("This tab is active in a separate window.");
            ImGui::TextDisabled("Double-click or Right-click the tab icon to dock back.");
            ImGui::Unindent(10.0f);
            ImGui::Spacing();
        }

        // When the active tab is popped out its content lives in a separate window,
        // so skip the in-panel switch (avoids double-rendering side effects).
        if (!tab_is_popped)
        switch (active_properties_tab) {
            case 0: drawSceneHierarchy(ctx); break;
            case 1:
                {
                    drawRenderInspectorContent(ctx);
                }
                break;
#if 0
                {
                    // ─────────────────────────────────────────────────────────────────────────
                    // ENGINE & BACKEND  (always visible - render device selection should be
                    // accessible regardless of the active viewport shading mode)
                    // ─────────────────────────────────────────────────────────────────────────
                    // Viewport shading mode: 0=Solid, 1=MaterialPreview, 2=Rendered, 3=Matcap
                    // When Solid/Matcap is active, the Vulkan raster pipeline is used
                    // automatically. The engine selection below sets the path-tracing
                    // backend used when the viewport switches to Rendered mode.
                    {
                        if (UIWidgets::BeginSection("Render Engine", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
                            extern bool g_hasOptix;
                            extern bool g_hasVulkan;
                            static int deferred_engine_type = -1; // 0=CPU,1=OptiX,2=Vulkan

                            // Info text when in Solid/Matcap mode
                            const bool is_raster_viewport = (viewport_settings.shading_mode == 0 || viewport_settings.shading_mode == 3);
                            if (is_raster_viewport) {
                                const bool interactive_backend_active =
                                    (dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) != nullptr) ||
                                    (g_viewport_backend != nullptr);
                                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f), "[i] Solid / Matcap Experimental");
                                ImGui::TextWrapped(interactive_backend_active
                                    ? "Solid/Matcap is currently running on the interactive viewport path. Engine selection below still applies to Rendered mode."
                                    : "Solid/Matcap is experimental and the interactive viewport path is unavailable right now. Engine selection below applies to Rendered mode.");
                                ImGui::Spacing();
                            }

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

                            // Apply deferred choice only when viewport is in Rendered mode.
                            const bool rendered_viewport_active = (viewport_settings.shading_mode == 2);
                            if (rendered_viewport_active && deferred_engine_type >= 0) {
                                const int clamped_deferred = std::clamp(deferred_engine_type, 0, 2);
                                const bool next_use_optix = (clamped_deferred == 1);
                                const bool next_use_vulkan = (clamped_deferred == 2);
                                if (ctx.render_settings.use_optix != next_use_optix ||
                                    ctx.render_settings.use_vulkan != next_use_vulkan) {
                                    ctx.render_settings.use_optix = next_use_optix;
                                    ctx.render_settings.use_vulkan = next_use_vulkan;
                                    extern bool g_cpu_sync_pending;
                                    g_cpu_sync_pending = true;
                                    ctx.render_settings.backend_changed = true;
                                    ctx.start_render = true;
                                }
                                deferred_engine_type = -1;
                            }
                            if (!rendered_viewport_active && deferred_engine_type >= 0) {
                                engine_type = std::clamp(deferred_engine_type, 0, 2);
                            }

                            const char* engines[] = { "CPU (Reference)", "NVIDIA OptiX (CUDA)", "Vulkan RT (Recommended)" };

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
                                        if (rendered_viewport_active) {
                                            ctx.render_settings.use_optix = (engine_type == 1);
                                            ctx.render_settings.use_vulkan = (engine_type == 2);
                                            extern bool g_cpu_sync_pending;
                                            g_cpu_sync_pending = true;

                                            ctx.render_settings.backend_changed = true;
                                            ctx.start_render = true;
                                            // See sibling combo above — surface HUD status on the last
                                            // frame before the (possibly long) backend switch begins.
                                            if (engine_type == 1) {
                                                addViewportMessage(
                                                    "Preparing OptiX backend. Renderer keeps running.",
                                                    5.0f, ImVec4(1.0f, 0.85f, 0.3f, 1.0f));
                                            } else if (engine_type == 2) {
                                                addViewportMessage(
                                                    "Switching to Vulkan RT...",
                                                    10.0f, ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
                                            } else {
                                                addViewportMessage(
                                                    "Switching to CPU renderer...",
                                                    10.0f, ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
                                            }
                                        } else {
                                            // In Solid/Matcap/MaterialPreview this is just a preference.
                                            deferred_engine_type = engine_type;
                                            addViewportMessage("Engine preference saved. It will apply in Rendered mode.",
                                                2.0f, ImVec4(0.55f, 0.85f, 1.0f, 1.0f));
                                        }
                                    }

                                    if (is_selected) {
                                        ImGui::SetItemDefaultFocus();
                                    }
                                }
                                ImGui::EndCombo();
                            }

                            if (!g_hasOptix && !g_hasVulkan) {
                                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "[No Compatible GPU Available]");
                            } else if (!g_hasOptix && engine_type == 1) {
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
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // SAMPLING 
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Sampling", ImVec4(0.5f, 0.9f, 0.6f, 1.0f))) {
                        UIWidgets::ColoredHeader("Viewport", ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                        bool sampling_changed = false;
                        sampling_changed |= ImGui::Checkbox("Use Adaptive Sampling##view", &ctx.render_settings.use_adaptive_sampling);
                        if (ctx.render_settings.use_adaptive_sampling) {
                            sampling_changed |= ImGui::DragFloat("Noise Threshold", &ctx.render_settings.variance_threshold, 0.001f, 0.001f, 0.8f, "%.3f");
                            sampling_changed |= ImGui::DragInt("Min Samples##view", &ctx.render_settings.min_samples, 1, 1, 512);
                        }
                        sampling_changed |= ImGui::DragInt("Max Samples##view", &ctx.render_settings.max_samples, 1, 1, 10000);
                        if (sampling_changed) {
                            ctx.start_render = true;
                            resetSceneUiSamplingAccumulation(ctx);
                        }

                        UIWidgets::Divider();
                        ImGui::Checkbox("Show Scene Stats HUD", &ctx.render_settings.show_scene_stats_hud);
                        
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // LIGHT PATHS (Bounces)
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Light Paths", ImVec4(1.0f, 0.8f, 0.3f, 1.0f))) {
                        ImGui::DragInt("Total Bounces", &ctx.render_settings.max_bounces, 1, 1, 64);
                        ImGui::DragInt("Diffuse Bounces", &ctx.render_settings.diffuse_bounces, 1, 1, 64);
                        ImGui::DragInt("Transmission Bounces", &ctx.render_settings.transmission_bounces, 1, 1, 64);
                        ctx.render_settings.diffuse_bounces = std::clamp(ctx.render_settings.diffuse_bounces, 1, ctx.render_settings.max_bounces);
                        ctx.render_settings.transmission_bounces = std::clamp(ctx.render_settings.transmission_bounces, 1, ctx.render_settings.max_bounces);

                        UIWidgets::HelpMarker("Total is the global path limit. Diffuse limits indirect diffuse/SSS/translucent paths; Transmission limits glass and water continuation.");

                        UIWidgets::Divider();
                        ImGui::Checkbox("Photon Caustics (Vulkan RT)##caustics2", &ctx.render_settings.caustics_enabled);
                        if (ctx.render_settings.caustics_enabled && ctx.render_settings.use_optix) {
                            UIWidgets::StatusIndicator("Caustics are not supported on the OptiX backend yet (Vulkan RT / CPU only)",
                                                       UIWidgets::StatusType::Warning);
                        }
                        if (ctx.render_settings.caustics_enabled) {
                            ImGui::Indent();
                            ImGui::DragInt("Photons / frame##causticsn2", &ctx.render_settings.caustics_photons, 1024, 8192, 4194304);
                            ImGui::DragFloat("Grid Cell Size##causticscs2", &ctx.render_settings.caustics_cell_size, 0.005f, 0.005f, 2.0f, "%.3f");
                            ImGui::DragFloat("Energy##causticse2", &ctx.render_settings.caustics_energy, 0.05f, 0.0f, 100.0f, "%.2f");
                            ImGui::Unindent();
                        }
                        ImGui::Checkbox("Volumetric (light shafts)##causticsvol2", &ctx.render_settings.caustics_volumetric);
                        if (ctx.render_settings.caustics_volumetric) {
                            ImGui::Indent();
                            ImGui::DragFloat("Scatter Strength##causticsvols2", &ctx.render_settings.caustics_vol_strength, 0.05f, 0.0f, 100.0f, "%.2f");
                            ImGui::Checkbox("Direct light shafts##causticsvoldir2", &ctx.render_settings.caustics_vol_direct);
                            ImGui::SliderFloat("Shaft Noise##causticsvolnoise2", &ctx.render_settings.caustics_vol_noise, 0.0f, 1.0f, "%.2f");
                            ImGui::Unindent();
                        }
                        UIWidgets::Divider();
                        {
                            const char* dv_items2[] = {
                                "Off", "Photon Grid (volume)", "Light Shaft Density",
                                "Photon Energy (surface)", "Caustic Cells", "Photon Directions",
                                "Bounce Count", "Transmission", "Absorption", "Medium Density (interior)",
                                "Normal (first hit)", "Albedo (first hit)", "Depth", "Material ID",
                                "Sample Heatmap"
                            };
                            const char* dv_tips2[] = {
                                "Normal beauty render — debug visualizer off.",
                                "Raw VOLUME photon-grid energy along the primary ray — verifies "
                                "shaft & photon placement. Arms the photon pass automatically.",
                                "The volumetric in-scatter integral alone: shafts without the "
                                "scene. Arms the photon pass automatically.",
                                "SURFACE photon-grid irradiance at the first hit — where caustic "
                                "energy lands. Arms the photon pass automatically.",
                                "Surface grid with a stable colour per hash cell — shows the "
                                "tiling; use to tune Grid Cell Size.",
                                "Average photon flow direction per volume cell, RGB = XYZ; "
                                "brightness = deposited energy. Arms the photon pass.",
                                "Viridis heatmap of path depth. Yellow = bounce-limit paths, "
                                "purple = terminated early.",
                                "Surviving path throughput — the colour a white backlight keeps "
                                "after crossing this pixel's path.",
                                "What the path lost: 1 - throughput, per channel.",
                                "Interior Volume dust integral along the view ray; specks flash "
                                "white; non-interior materials read ~0.",
                                "First-hit world-space shading normal (RGB = XYZ).",
                                "First-hit base colour after texturing (denoiser albedo feed).",
                                "Distance to first hit, exponential ramp; Exposure tunes range.",
                                "Stable hash colour per material; only 'no hit' is black "
                                "(material ID 0 is valid).",
                                "Per-pixel accumulated sample count at display time — beauty "
                                "keeps accumulating; toggling never resets. Best with Adaptive "
                                "Sampling on."
                            };
                            int dv = std::clamp(ctx.render_settings.debug_view, 0, 14);
                            if (ImGui::BeginCombo("Debug View##dbgview2", dv_items2[dv])) {
                                for (int i = 0; i < 15; ++i) {
                                    const bool unavailable = false; // all 15 views live
                                    if (unavailable) ImGui::BeginDisabled();
                                    if (ImGui::Selectable(dv_items2[i], dv == i)) {
                                        ctx.render_settings.debug_view = i;
                                        ctx.start_render = true; // change tracker runs only inside the render block
                                    }
                                    if (unavailable) ImGui::EndDisabled();
                                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                                        ImGui::SetTooltip("%s", dv_tips2[i]);
                                    }
                                }
                                ImGui::EndCombo();
                            }
                            if (dv != 0) {
                                ImGui::PushTextWrapPos(0.0f);
                                ImGui::TextDisabled("%s", dv_tips2[dv]);
                                ImGui::PopTextWrapPos();
                            }
                            if (ctx.render_settings.debug_view != 0) {
                                ImGui::Indent();
                                if (ImGui::DragFloat("Exposure##dbgexp2", &ctx.render_settings.debug_exposure, 0.05f, 0.0f, 1000.0f, "%.2f"))
                                    ctx.start_render = true;
                                if (ImGui::SliderFloat("Overlay Beauty##dbgovl2", &ctx.render_settings.debug_overlay, 0.0f, 1.0f, "%.2f"))
                                    ctx.start_render = true;
                                ImGui::Unindent();
                            }
                        }
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // DENOISING
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Denoising", ImVec4(0.8f, 0.5f, 1.0f, 1.0f))) {
                        if (ImGui::Checkbox("Enable Denoiser (Viewport + Final + Sequence)", &ctx.render_settings.use_denoiser)) {
                            // Mirror to legacy 'render_use_denoiser' so the single-frame final render
                            // path stays in sync without a separate checkbox.
                            ctx.render_settings.render_use_denoiser = ctx.render_settings.use_denoiser;
                        }
                        UIWidgets::HelpMarker("Applies to interactive viewport, single-frame final render, "
                                              "AND sequence render output.");
                        if (ctx.render_settings.use_denoiser) {
                            const char* denoiser_mode_items[] = {
                                "Beauty only",
                                "Beauty + Albedo + Normal"
                            };
                            int denoiser_mode = static_cast<int>(ctx.render_settings.denoiser_mode);
                            if (ImGui::Combo("Input Buffers", &denoiser_mode, denoiser_mode_items, IM_ARRAYSIZE(denoiser_mode_items))) {
                                ctx.render_settings.denoiser_mode = static_cast<DenoiserMode>(denoiser_mode);
                            }
                            const char* denoiser_quality_items[] = {
                                "Fast (viewport)",
                                "Balanced",
                                "High (final)"
                            };
                            int denoiser_quality = static_cast<int>(ctx.render_settings.denoiser_quality);
                            if (ImGui::Combo("Denoiser Quality", &denoiser_quality, denoiser_quality_items, IM_ARRAYSIZE(denoiser_quality_items))) {
                                ctx.render_settings.denoiser_quality = static_cast<DenoiserQuality>(denoiser_quality);
                            }
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("OIDN model tier for the viewport. Fast is the cheapest\n(dominant cost on the GPU). Final renders always use High.");
                            }
                            ImGui::SliderFloat("Blend Factor", &ctx.render_settings.denoiser_blend_factor, 0.0f, 1.0f);
                        }
                        UIWidgets::EndSection();
                    }


                    // Resolution & Output controls moved to the System panel.

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
                            // Sequence render shares the main render-quality setting.
                            ImGui::TextDisabled("Uses main Samples setting: %d / frame",
                                ctx.render_settings.final_render_samples);
                            UIWidgets::HelpMarker("Sequence render uses the same samples-per-frame value "
                                                  "as the main Render Settings 'Samples' field.");
                            
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
                            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Resolution: %dx%d (viewport)",
                                image_width, image_height);
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
#endif
            case 2: if (show_terrain_tab) drawTerrainPanel(ctx); break;
            case 3:
                if (show_water_tab) {
                    if (ImGui::Button("Water##WaterSubtab")) active_water_subtab = 0;
                    ImGui::SameLine();
                    if (ImGui::Button("River##WaterSubtab")) active_water_subtab = 1;
                    ImGui::Separator();
                    if (active_water_subtab == 0) {
                        drawWaterPanel(ctx);
                    } else {
                        drawRiverPanel(ctx);
                    }
                }
                break;
            case 4: if (show_volumetric_tab) drawVolumetricPanel(ctx); break;
            case 5: if (show_forcefield_tab) ForceFieldUI::drawForceFieldPanel(*this, ctx, ctx.scene, &timeline); break;
            case 6: if (show_world_tab) drawWorldContent(ctx); break;
            case 12: if (show_stylize_tab) drawStylizePanel(ctx); break;
            case 7: drawModifiersPanel(ctx); break;
            case 13: drawSculptPanel(ctx); break;
            case 11: if (show_scatter_tab) drawScatterBrushPanel(ctx); break;
            case 9: drawThemeSelector(); drawResolutionPanel(ctx); break;
            case 10: if (show_paint_tab) drawPaintPanel(ctx); break;
            case 8: if (show_hair_tab) drawHairTabContent(ctx); break;
        }
        UIWidgets::PopControlSurfaceStyle();
        ImGui::PopItemWidth();

        // Safety: Disable terrain brush when leaving terrain tools, unless the new paint mode
        // is intentionally driving the same optimized terrain paint backend from the modifiers tab.
        const bool allow_paint_bridge =
            (active_properties_tab == 10) &&
            paint_mode_state.enabled &&
            paint_mode_state.hasValidTarget();
        if (active_properties_tab != 2 && !properties_tab_popped_[2] && !allow_paint_bridge) {
            terrain_brush.enabled = false;
        }

        // If the user switched sidebar tabs, ensure any panel-local paint/brush modes are
        // deactivated so they don't continue painting when their panel is no longer visible.
        static int s_last_active_tab = -1;
        if (s_last_active_tab == -1) s_last_active_tab = active_properties_tab;
        if (s_last_active_tab != active_properties_tab) {
            // Leaving Terrain tab -> disable foliage brush (unless it is popped out & still visible)
            if (s_last_active_tab == 2 && !properties_tab_popped_[2]) {
                foliage_brush.enabled = false;
                foliage_brush.active_group_id = -1;
                foliage_brush.pending_instances.clear();
                foliage_brush.pending_group_id = -1;
            }
            // Leaving Scatter (legacy) tab -> disable scatter brush (unless popped out)
            if (s_last_active_tab == 11 && !properties_tab_popped_[11]) {
                scatter_brush.enabled = false;
                scatter_brush.active_group_id = -1;
                scatter_brush.pending_instances.clear();
                scatter_brush.pending_group_id = -1;
            }
            // Leaving Paint tab -> fully exit paint mode (unless popped out & still visible)
            if (s_last_active_tab == 10 && !properties_tab_popped_[10]) {
                paint_mode_state.enabled = false;
                paint_mode_state.clearAdapter();
            }
            // Leaving Modifiers (Modeling) tab -> fully exit sculpt/edit modes (unless popped out or moving to Sculpting)
            if (s_last_active_tab == 7 && !properties_tab_popped_[7] && active_properties_tab != 13) {
                resetMeshEditState(ctx);
            }
            // Leaving Sculpting tab -> fully exit sculpt/edit modes (unless popped out or moving to Modeling)
            if (s_last_active_tab == 13 && !properties_tab_popped_[13] && active_properties_tab != 7) {
                resetMeshEditState(ctx);
            }
            // Leaving Physics tab -> fully exit edit/sculpt modes if active (unless popped out or moving to Modeling/Sculpting)
            if (s_last_active_tab == 5 && !properties_tab_popped_[5] && active_properties_tab != 7 && active_properties_tab != 13) {
                resetMeshEditState(ctx);
            }
            // Leaving Hair & Fur tab -> fully exit hair paint mode (unless popped out)
            if (s_last_active_tab == 8 && !properties_tab_popped_[8]) {
                hairUI.setPaintMode(Hair::HairPaintMode::NONE);
            }

            // Entering hooks
            if (active_properties_tab == 13) {
                activateSculptWorkspace(ctx);
            }

            s_last_active_tab = active_properties_tab;
        }
        
        ImGui::PopStyleVar(2);  // WindowPadding, ItemSpacing
        ImGui::EndChild();      // End PropScrollArea
        ImGui::EndChild();      // End PropContentArea
        ImGui::PopStyleColor(); // ChildBg
        ImGui::PopStyleVar();   // ChildRounding
    }
    ImGui::End();
    ImGui::PopStyleColor(2); // WindowBg, Border
    ImGui::PopStyleVar(3);   // WindowPadding, WindowRounding, BorderSize
}

// Main Menu Bar implementation moved to separate file: scene_ui_menu.hpp check end of file

#include "scene_ui_menu.hpp"
#include <omp.h>


void SceneUI::validateSelectionAgainstScene(UIContext& ctx) {
    if (ctx.selection.multi_selection.empty() && !ctx.selection.selected.is_valid()) {
        return;
    }

    std::unordered_map<std::string, std::pair<int, std::shared_ptr<Triangle>>> live_objects_by_name;
    std::unordered_map<std::string, std::shared_ptr<TriangleMesh>> live_meshes_by_name;
    live_objects_by_name.reserve(ctx.scene.world.objects.size());
    live_meshes_by_name.reserve(ctx.scene.world.objects.size());

    for (size_t i = 0; i < ctx.scene.world.objects.size(); ++i) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[i])) {
            std::string node_name = tri->getNodeName();
            if (node_name.empty()) continue;
            live_objects_by_name.emplace(node_name, std::make_pair(static_cast<int>(i), tri));
            if (tri->parentMesh) live_meshes_by_name.emplace(node_name, tri->parentMesh);
        } else if (auto tmesh = std::dynamic_pointer_cast<TriangleMesh>(ctx.scene.world.objects[i])) {
            std::string node_name = tmesh->nodeName;
            if (node_name.empty()) continue;
            live_meshes_by_name.emplace(node_name, tmesh);
            std::shared_ptr<Triangle> rep = nullptr;
            auto rep_it = direct_mesh_rep_by_ptr.find(tmesh.get());
            if (rep_it != direct_mesh_rep_by_ptr.end()) {
                rep = rep_it->second;
            } else {
                auto d_it = direct_mesh_nodes.find(node_name);
                if (d_it != direct_mesh_nodes.end()) {
                    rep = d_it->second.rep;
                }
            }
            if (rep) {
                live_objects_by_name.emplace(node_name, std::make_pair(static_cast<int>(i), rep));
            }
        }
    }

    auto rebind_item = [&](SelectableItem& item) -> bool {
        switch (item.type) {
            case SelectableType::Object: {
                std::string node_name = item.name;
                if (node_name.empty() && item.object) {
                    node_name = item.object->getNodeName();
                }
                if (node_name.empty()) return false;

                auto it = live_objects_by_name.find(node_name);
                if (it == live_objects_by_name.end() || !it->second.second) {
                    return false;
                }

                item.object = it->second.second;
                auto mesh_it = live_meshes_by_name.find(node_name);
                item.mesh_object = mesh_it != live_meshes_by_name.end()
                    ? mesh_it->second
                    : item.object->parentMesh;
                if (item.object && item.object->parentMesh == item.mesh_object) {
                    item.mesh_face_index = item.object->faceIndex;
                }
                item.object_index = it->second.first;
                item.name = node_name;
                return true;
            }
            case SelectableType::Light:
                return selectionContainsLight(ctx.scene, item.light);
            case SelectableType::Camera:
            case SelectableType::CameraTarget:
                return selectionContainsCamera(ctx.scene, item.camera);
            case SelectableType::VDBVolume:
                return selectionContainsVDB(ctx.scene, item.vdb_volume);
            case SelectableType::GasVolume:
                return selectionContainsGas(ctx.scene, item.gas_volume);
            case SelectableType::ForceField:
                return selectionContainsForceField(ctx.scene, item.force_field);
            case SelectableType::ParticleSystem:
                return item.particle_system_index >= 0 &&
                       item.particle_system_index < static_cast<int>(ctx.scene.particle_systems.size());
            case SelectableType::SimulationDomain:
                if (item.particle_system_index < 0 ||
                    item.particle_system_index >= static_cast<int>(ctx.scene.particle_systems.size())) {
                    return false;
                }
                if (!ctx.scene.particle_systems[static_cast<std::size_t>(item.particle_system_index)].runtime) {
                    return false;
                }
                return item.simulation_domain_index >= 0 &&
                       item.simulation_domain_index < static_cast<int>(
                           ctx.scene.particle_systems[static_cast<std::size_t>(item.particle_system_index)]
                               .runtime->gridDomains().size());
            case SelectableType::World:
                return true;
            case SelectableType::None:
            default:
                return false;
        }
    };

    bool changed = false;
    auto& multi = ctx.selection.multi_selection;
    for (auto it = multi.begin(); it != multi.end();) {
        if (!rebind_item(*it)) {
            it = multi.erase(it);
            changed = true;
        } else {
            ++it;
        }
    }

    if (!ctx.selection.selected.is_valid()) {
        if (!multi.empty()) {
            ctx.selection.syncPrimarySelection();
            changed = true;
        }
    } else {
        SelectableItem rebound_primary = ctx.selection.selected;
        if (!rebind_item(rebound_primary)) {
            ctx.selection.syncPrimarySelection();
            changed = true;
        } else if (rebound_primary.type != ctx.selection.selected.type ||
                   rebound_primary.object != ctx.selection.selected.object ||
                   rebound_primary.mesh_object != ctx.selection.selected.mesh_object ||
                   rebound_primary.mesh_face_index != ctx.selection.selected.mesh_face_index ||
                   rebound_primary.light != ctx.selection.selected.light ||
                   rebound_primary.camera != ctx.selection.selected.camera ||
                   rebound_primary.vdb_volume != ctx.selection.selected.vdb_volume ||
                   rebound_primary.gas_volume != ctx.selection.selected.gas_volume ||
                   rebound_primary.force_field != ctx.selection.selected.force_field ||
                   rebound_primary.object_index != ctx.selection.selected.object_index ||
                   rebound_primary.particle_system_index != ctx.selection.selected.particle_system_index ||
                   rebound_primary.simulation_domain_index != ctx.selection.selected.simulation_domain_index ||
                   rebound_primary.name != ctx.selection.selected.name) {
            ctx.selection.selected = rebound_primary;
            ctx.selection.updatePositionFromSelection();
            changed = true;
        }
    }

    if (changed && ctx.selection.multi_selection.empty()) {
        ctx.selection.clearSelection();
    }
}

// Publish the current Edit-Mesh vertex selection (world space) into the
// ForceFieldUI global so the (free-function) Bodies panel can offer
// "Pin Selected Vertices" for cloth/soft bodies without touching SceneUI's
// internals. Cheap: only fills the buffer while Edit > Vertex mode is active.
void SceneUI::publishEditPinSelection(UIContext& ctx)
{
    auto& snap = ForceFieldUI::g_edit_pin_selection;
    const bool active =
        mesh_overlay_settings.enabled && mesh_overlay_settings.edit_mode &&
        mesh_workspace_mode == MeshWorkspaceMode::Edit &&
        ctx.selection.mesh_element_mode == MeshElementSelectMode::Vertex &&
        !active_mesh_edit_object_name.empty() &&
        editable_mesh_cache.object_name == active_mesh_edit_object_name &&
        !editable_mesh_cache.selection.vertex_ids.empty();

    if (!active) {
        if (snap.active) {
            snap.active = false;
            snap.object_name.clear();
            snap.world_positions.clear();
        }
        return;
    }

    snap.active = true;
    snap.object_name = active_mesh_edit_object_name;
    snap.world_positions.clear();
    snap.world_positions.reserve(editable_mesh_cache.selection.vertex_ids.size());
    const Matrix4x4& xf = editable_mesh_cache.source_object_transform;
    for (int vid : editable_mesh_cache.selection.vertex_ids) {
        if (vid < 0 || vid >= (int)editable_mesh_cache.vertices.size()) continue;
        snap.world_positions.push_back(
            xf.transform_point(editable_mesh_cache.vertices[vid].local_position));
    }
}

void SceneUI::selectManagedMesh(UIContext& ctx, const std::shared_ptr<TriangleMesh>& mesh) {
    if (!mesh) return;
    int objectIndex = -1;
    for (size_t i = 0; i < ctx.scene.world.objects.size(); ++i) {
        if (ctx.scene.world.objects[i].get() == mesh.get()) {
            objectIndex = static_cast<int>(i);
            break;
        }
    }
    ctx.selection.selectObject(mesh, objectIndex,
        mesh->nodeName.empty() ? std::string("Managed Surface") : mesh->nodeName);
    // Apply the reverse mapping immediately so another managed panel drawn in
    // this same frame reflects the click without a one-frame selection delay.
    syncManagedObjectSelection(ctx);
}

void SceneUI::syncManagedObjectSelection(UIContext& ctx) {
    if (ctx.selection.selected.type != SelectableType::Object) return;
    std::shared_ptr<TriangleMesh> selectedMesh = ctx.selection.selected.mesh_object;
    if (!selectedMesh && ctx.selection.selected.object) {
        selectedMesh = ctx.selection.selected.object->parentMesh;
    }
    if (!selectedMesh) return;

    for (auto& terrain : TerrainManager::getInstance().getTerrains()) {
        if (terrain.flatMesh.get() == selectedMesh.get()) {
            terrain_brush.active_terrain_id = terrain.id;
            break;
        }
    }

    for (auto& surface : WaterManager::getInstance().getWaterSurfaces()) {
        if (surface.flatMesh.get() == selectedMesh.get()) {
            selected_water_surface_id = surface.id;
            break;
        }
    }

    auto& riverManager = RiverManager::getInstance();
    for (auto& river : riverManager.getRivers()) {
        if (river.flatMesh.get() == selectedMesh.get()) {
            riverManager.editingRiverId = river.id;
            riverManager.selectedControlPoint = -1;
            if (river.waterSurfaceId >= 0) selected_water_surface_id = river.waterSurfaceId;
            break;
        }
    }
}

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
    syncMeshEditState(ctx);
    publishEditPinSelection(ctx);
    tryRestoreSerializedMeshEditLayer(ctx);
    processPendingMeshEditGpuSync(ctx);

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
                     if (item.type == SelectableType::Object) {
                         if (item.mesh_object) {
                             selected_hittables.push_back(item.mesh_object);
                         } else if (item.object) {
                             selected_hittables.push_back(item.object);
                         }
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
    bool scene_membership_changed = false;
    if (ctx.scene.world.objects.size() != last_scene_obj_count) {
        if (last_scene_obj_count != 0) {
            SCENE_LOG_INFO("Scene changed (Count: " + std::to_string(last_scene_obj_count) + 
                           " -> " + std::to_string(ctx.scene.world.objects.size()) + "). Invalidating cache.");
        }
        mesh_cache_valid = false;
        last_scene_obj_count = ctx.scene.world.objects.size();
        scene_membership_changed = true;
    }
    // Data-side ops can change an object's geometry without changing the object
    // count (e.g. "Apply at Frame" freezes a body's deformed mesh). Honor their
    // one-shot request to rebuild the mesh/bbox caches so the selection outline
    // tracks the new shape.
    if (ctx.scene.consumeUiMeshCacheRebuild()) {
        mesh_cache_valid = false;
    }

    if (ctx.scene.lights.size() != last_scene_light_count) {
        last_scene_light_count = ctx.scene.lights.size();
        scene_membership_changed = true;
    }
    if (ctx.scene.cameras.size() != last_scene_camera_count) {
        last_scene_camera_count = ctx.scene.cameras.size();
        scene_membership_changed = true;
    }
    if (ctx.scene.vdb_volumes.size() != last_scene_vdb_count) {
        last_scene_vdb_count = ctx.scene.vdb_volumes.size();
        scene_membership_changed = true;
    }
    if (ctx.scene.gas_volumes.size() != last_scene_gas_count) {
        last_scene_gas_count = ctx.scene.gas_volumes.size();
        scene_membership_changed = true;
    }
    if (ctx.scene.force_field_manager.force_fields.size() != last_scene_forcefield_count) {
        last_scene_forcefield_count = ctx.scene.force_field_manager.force_fields.size();
        scene_membership_changed = true;
    }

    if (scene_membership_changed) {
        selection_validation_pending = true;
    }

    if (selection_validation_pending) {
        validateSelectionAgainstScene(ctx);
        selection_validation_pending = false;
    }
    syncManagedObjectSelection(ctx);

    world_params_changed_this_frame = false;
    ImGuiIO& io = ImGui::GetIO();
    float screen_x = io.DisplaySize.x;
    float screen_y = io.DisplaySize.y;

    drawMainMenuBar(ctx);
    rtpython::drawConsole(&show_python_console);
    rtipc_panel::draw(&show_remote_ipc_panel);
    rtpython::drawAddonPanels();  // Faz 4b: addon-registered rt.ui panels
    handleEditorShortcuts(ctx);

    // Host the dockable layout (no-op when docking_enabled is false).
    drawDockSpaceHost(ctx);

    float left_offset = 0.0f;
    drawPanels(ctx);
    left_offset = showSidePanel ? side_panel_width : 0.0f;
    drawPoppedPropertyWindows(ctx); // detachable Properties sub-tabs (floating/dockable)
    drawPaintBrushDock(ctx);

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
        ctx.scene.refreshSimulationForceFieldSnapshot();
        const auto& forceSnapshot = ctx.scene.getSimulationWorld().getForceFieldSnapshot();

        if (!forceSnapshot.empty() &&
            ctx.renderer.getHairSystem().getGroomNames().size() > 0 &&
            (isPlaying || timeChanged)) {
            
            for (const auto& groomName : ctx.renderer.getHairSystem().getGroomNames()) {
                ctx.renderer.getHairSystem().restyleGroom(groomName, &forceSnapshot, currentTime);
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


    // [SEQUENCE-RENDER OWNERSHIP] While a sequence render is active the worker
    // thread (Renderer::render_Animation) is the SOLE driver of the sim timeline
    // + the particle/foam render bridge — it bakes each frame deterministically
    // and rebuilds the backend AS itself (the main loop's rebuild handlers are
    // gated by skip_backend_for_anim). If the UI thread ALSO stepped the sim or
    // re-wrote the InstanceManager bridge groups here, the two threads would
    // race on sim state + the shared TLAS/InstanceManager (silent NVIDIA hang,
    // same hazard as render_owns_timeline in TimelineWidget). Skip both writes
    // while the worker is authoritative.
    extern std::atomic<bool> rendering_in_progress;
    // A viewport-driven sequence save keeps both anim flags set for the UI but the
    // UI MUST keep driving the per-frame sim scrub (the save reads what the viewport
    // renders), so it is NOT a worker-owned render.
    extern bool g_seq_save_active;
    const bool render_owns_sim =
        ctx.is_animation_mode && rendering_in_progress.load() && !g_seq_save_active;

    if (ctx.scene.isSimulationBaking()) {
        // A cooperative disk bake owns the sim timeline while it runs. Advance it
        // a time-budgeted slice per UI tick (the progress bar + Cancel live in the
        // sim panel) and keep the loop awake so the next slice fires and the
        // progress redraws. Skip the normal live/timeline drive entirely — the
        // bake is the sole stepper, so the two cannot race on sim state.
        ctx.scene.tickSimulationDiskBake(20.0 /* ms budget per UI frame */);
        ctx.start_render = true;
    } else if (!render_owns_sim && ctx.scene.anySimulationRuntimeEnabled()) {
        const float rt_dt = std::clamp(io.DeltaTime, 0.0f, 1.0f / 30.0f);
        const bool live_mode = !g_sim_timeline_mode;
        // Timeline mode (default): play bakes into the cache, scrub restores, a
        // stopped timeline stays frozen. Live mode: continuous free-run preview.
        // ui_editing: while a widget is held (slider drag, etc.) the sim-config
        // signature changes every tick. Pass this so the cache invalidation is
        // deferred until the edit settles (no per-tick re-bake-from-0 thrash).
        const bool ui_editing = ImGui::IsAnyItemActive();
        ctx.scene.updateSimulationTimeline(timeline.getCurrentFrame(), timeline.isPlaying(), rt_dt, 24.0f, live_mode, ui_editing);
        // A fluid-affecting edit rewinds the sim to frame 0 instead of auto-resimming
        // up to a high parked frame; move the playhead to start so the cost of the
        // re-bake is opt-in (the user plays forward when ready). Flash a transient
        // HUD toast at the moment it happens — far more noticeable than a static
        // panel note, and it tells the user why the playhead just jumped.
        if (ctx.scene.consumeSimRewindRequest()) {
            timeline.setCurrentFrame(timeline.getStartFrame());
            addViewportMessage("Sim changed - cache reset, rewound to start. Press Play to re-bake.",
                               4.0f, ImVec4(1.00f, 0.62f, 0.10f, 1.00f));
        }
        // Keep rendering / reset accumulation only while the gas is actually
        // changing (live mode, or a baked/scrubbed frame). A frozen timeline lets
        // the render converge and the loop go idle — the cheap default.
        if (live_mode || ctx.scene.simulation_render_updated) {
            ctx.start_render = true;
            resetSceneUiSamplingAccumulation(ctx);
        }
    }
    // Feed camera-facing billboards to the Vulkan viewport every frame so particles
    // render (depth-tested) in Solid mode and re-face the camera as it moves.
    // GATED during a sequence render: this reads the live particle/foam SoA on the
    // MAIN thread while the render WORKER is stepping + mutating that exact SoA —
    // a data race plus competing viewport-backend GPU work ("two jobs at once").
    // As foam grows the main-thread billboard rebuild gets heavier and starves the
    // worker, which showed up as a ~10x slowdown ~10-15 frames in. The viewport
    // isn't interactively shown during the render anyway, so skip it entirely.
    if (!render_owns_sim) {
        uploadParticleBillboards(ctx);
    }

    // Mirror discrete particles into the ray-traced render every frame, driven
    // independently of the timeline sim driver (just like the billboard upload
    // above). It reads the live SoA and self-flags a cheap TLAS refit on motion;
    // a settled/frozen SoA is skipped (hash). This decouples particle RT render
    // from the timeline bake/scrub/cache, which only manages grid domains.
    // Debug display mode (1) uses the lightweight ImGui overlay, so the instanced
    // geometry is suppressed there (Solid/Render show it).
    // Skipped while the sequence-render worker owns the bridge (see above).
    if (!render_owns_sim) {
        ctx.scene.syncParticleRenderInstances(ctx.particle_display_mode != 1);
    }

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
    handleMeshSculpt(ctx);
    stepWetClayField(ctx);   // dynamic wet-clay: settle + dry the active wet region each frame
    handleMeshPaint(ctx);
    
    // Hair Brush System
    handleHairBrush(ctx);      // Hair paint brush input + preview

    handleSceneInteraction(ctx, gizmo_hit);
    processDeferredSceneUpdates(ctx);
    
    
    drawAuxWindows(ctx);
    
    // Global Sun Sync (Light -> Nishita)
    processSunSync(ctx);
    drawRenderWindow(ctx);
    drawExitConfirmation(ctx);
    
    // NOTE: Animation Graph Editor is now strictly in the bottom panel
        }
    
void SceneUI::handleEditorShortcuts(UIContext& ctx)
{
    if (rtpython::wantsInputCapture() || rtapi::renderOutputPending()) return;
    ImGuiIO& io = ImGui::GetIO();
    const bool block_history_actions =
        g_scene_loading_in_progress.load() ||
        scene_loading.load() ||
        rendering_in_progress.load() ||
        ctx.render_settings.backend_changed;

    // Block only while typing in a text field, not whenever a panel has focus, so the
    // delete shortcut works whenever the app is focused (matches the N/Tab handlers below).
    // EXCEPTION: while the Geometry Graph node editor is focused, Delete/Backspace is claimed
    // by NodeEditorUIV2 for node/link deletion — without this, deleting a node in the graph
    // also deleted the scene object the graph belongs to (same Delete keypress, two listeners).
    if (!io.WantTextInput && ctx.selection.hasSelection() &&
        !terrain_graph_focused && !geometry_graph_focused && !material_graph_focused) {
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

    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Tab) && !io.KeyCtrl && !io.KeyShift && !io.KeyAlt) {
        const bool hasSelectedObject =
            ctx.selection.selected.type == SelectableType::Object &&
            ctx.selection.selected.object != nullptr;

        if (hasSelectedObject) {
            if (mesh_overlay_settings.edit_mode) {
                resetMeshEditState(ctx);
            } else {
                activateEditWorkspace(ctx);
            }
        }
    }

    // Selection Shortcuts (A=All, Alt+A=None, Ctrl+I=Invert)
    if (!io.WantTextInput) {
        const bool edit_mode_active = mesh_overlay_settings.enabled &&
                                      mesh_overlay_settings.edit_mode &&
                                      ctx.selection.mesh_element_mode != MeshElementSelectMode::Object;

        if (ImGui::IsKeyPressed(ImGuiKey_A) && !io.KeyCtrl && !io.KeyShift && !io.KeyAlt) {
            if (edit_mode_active) {
                selectAllMeshElements(ctx);
            } else {
                selectAllObjects(ctx);
            }
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_A) && !io.KeyCtrl && !io.KeyShift && io.KeyAlt) {
            if (edit_mode_active) {
                clearEditableMeshSelection();
            } else {
                ctx.selection.clearSelection();
            }
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_I) && io.KeyCtrl && !io.KeyShift && !io.KeyAlt) {
            if (edit_mode_active) {
                invertMeshSelection(ctx);
            } else {
                invertObjectSelection(ctx);
            }
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_B) && !io.KeyCtrl && !io.KeyShift && !io.KeyAlt) {
            mesh_overlay_settings.selection_tool = 0;
            addViewportMessage("Selection Tool: Box Select", 2.0f, ImVec4(0.38f, 0.82f, 1.0f, 1.0f));
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_L) && !io.KeyCtrl && !io.KeyShift && !io.KeyAlt) {
            mesh_overlay_settings.selection_tool = 1;
            addViewportMessage("Selection Tool: Lasso Select", 2.0f, ImVec4(0.38f, 0.82f, 1.0f, 1.0f));
        }
    }

    // Undo / Redo
    if (ImGui::IsKeyPressed(ImGuiKey_Z) && io.KeyCtrl && !io.KeyShift) {
        if (!block_history_actions && history.canUndo()) {
            history.undo(ctx);
            rebuildMeshCache(ctx.scene.world.objects);
            mesh_overlay_cache = MeshOverlayCache{};
            editable_mesh_cache = EditableMeshCache{};
            ctx.selection.updatePositionFromSelection();
            ctx.selection.selected.has_cached_aabb = false;
        }
    }

    if ((ImGui::IsKeyPressed(ImGuiKey_Y) && io.KeyCtrl) ||
        (ImGui::IsKeyPressed(ImGuiKey_Z) && io.KeyCtrl && io.KeyShift)) {
        if (!block_history_actions && history.canRedo()) {
            history.redo(ctx);
            rebuildMeshCache(ctx.scene.world.objects);
            mesh_overlay_cache = MeshOverlayCache{};
            editable_mesh_cache = EditableMeshCache{};
            ctx.selection.updatePositionFromSelection();
            ctx.selection.selected.has_cached_aabb = false;
        }
    }
}
void SceneUI::drawPanels(UIContext& ctx)
{
    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;

    float win_rounding = 4.0f;
    ImVec4 win_bg = ImVec4(0.1f, 0.1f, 0.13f, panel_alpha);

    if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
        const auto& curTheme = ThemeManager::instance().current();
        win_rounding = curTheme.style.windowRounding;
        win_bg = ImVec4(curTheme.colors.background.x, curTheme.colors.background.y, curTheme.colors.background.z, panel_alpha);
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, win_rounding);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, win_bg);

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
    // Sized for the highest tab_index passed to handleBottomTabEvents below (7 =
    // Material Graph) + 1. The old sizes (6/7) were already overrun by the Asset
    // Browser's tab_index 6 on the hold-timer array.
    static float bottom_hold_timers[8] = { 0.0f };
    static float bottom_flash_timers[8] = { 0.0f };

    auto isBottomPanelFloating = [&](const char* window_name) -> bool {
        if (!docking_enabled) return false;
        ImGuiWindow* win = ImGui::FindWindowByName(window_name);
        if (!win) return false;
        if (!win->DockNode) return true;
        ImGuiDockNode* node = win->DockNode;
        while (node->ParentNode && node->ParentNode->ID != dockspace_id) {
            node = node->ParentNode;
        }
        if (node->ParentNode && node->ParentNode->ID == dockspace_id) {
            if (dock_bottom_id == 0) {
                dock_bottom_id = node->ID;
            }
            return (node->ID != dock_bottom_id);
        }
        return true;
    };

    auto handleBottomTabEvents = [&](int tab_index, const char* window_name, bool& show_var, auto on_activate) {
        bool hovered = ImGui::IsItemHovered();
        bool active = ImGui::IsItemActive();

        // Right-click context menu
        if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
            ImGui::OpenPopup(window_name);
        }

        if (ImGui::BeginPopup(window_name)) {
            bool is_floating = isBottomPanelFloating(window_name);
            if (is_floating) {
                if (ImGui::MenuItem("Dock Back to Bottom")) {
                    dockToBottom(window_name);
                    bottom_flash_timers[tab_index] = 1.0f;
                }
            } else {
                if (ImGui::MenuItem("Undock (Float)")) {
                    ImGui::DockBuilderDockWindow(window_name, 0);
                    bottom_flash_timers[tab_index] = 1.0f;
                }
            }
            ImGui::EndPopup();
        }

        // Double-click to toggle float
        if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            bool is_floating = isBottomPanelFloating(window_name);
            if (is_floating) {
                dockToBottom(window_name);
            } else {
                show_var = true;
                on_activate();
                ImGui::DockBuilderDockWindow(window_name, 0);
            }
            bottom_flash_timers[tab_index] = 1.0f;
            bottom_hold_timers[tab_index] = 0.0f;
            return;
        }

        // Hold-to-pop (2 seconds)
        if (active && hovered) {
            bottom_hold_timers[tab_index] += ImGui::GetIO().DeltaTime;
            if (bottom_hold_timers[tab_index] >= 2.0f) {
                bool is_floating = isBottomPanelFloating(window_name);
                if (is_floating) {
                    dockToBottom(window_name);
                } else {
                    show_var = true;
                    on_activate();
                    ImGui::DockBuilderDockWindow(window_name, 0);
                }
                bottom_flash_timers[tab_index] = 1.0f;
                bottom_hold_timers[tab_index] = 0.0f;
                ImGui::ClearActiveID();
            }
        } else {
            bottom_hold_timers[tab_index] = (std::max)(0.0f, bottom_hold_timers[tab_index] - ImGui::GetIO().DeltaTime * 2.0f);
        }

        // Draw progress circle if held
        if (bottom_hold_timers[tab_index] > 0.0f) {
            float progress = bottom_hold_timers[tab_index] / 2.0f;
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 min_p = ImGui::GetItemRectMin();
            ImVec2 max_p = ImGui::GetItemRectMax();
            ImVec2 center = ImVec2((min_p.x + max_p.x) * 0.5f, (min_p.y + max_p.y) * 0.5f);
            float radius = (std::min)(max_p.x - min_p.x, max_p.y - min_p.y) * 0.35f;
            
            int num_segments = 32;
            float start_angle = -1.57079f;
            float end_angle = start_angle + progress * 6.28318f;
            
            draw_list->AddCircle(center, radius, IM_COL32(255, 255, 255, 40), num_segments, 1.5f);
            draw_list->PathArcTo(center, radius, start_angle, end_angle, num_segments);
            draw_list->PathStroke(IM_COL32(0, 160, 255, 220), 0, 2.5f);
        }
    };

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

        auto closeOtherBottomPanels = [&](int keep_index) {
            show_animation_panel = (keep_index == 0) ? show_animation_panel : false;
            show_scene_log      = (keep_index == 1) ? show_scene_log : false;
            show_terrain_graph  = (keep_index == 2) ? show_terrain_graph : false;
            show_anim_graph     = (keep_index == 3) ? show_anim_graph : false;
            show_asset_browser  = (keep_index == 4) ? show_asset_browser : false;
            show_geometry_graph = (keep_index == 5) ? show_geometry_graph : false;
            show_material_graph = (keep_index == 6) ? show_material_graph : false;
        };
        
        bool active_dope = show_animation_panel && (timeline.getEditorMode() == TimelineEditorMode::DopeSheet);
        if (UIWidgets::HorizontalTab("Dope Sheet", UIWidgets::IconType::Timeline, active_dope))
        {
            if (active_dope) {
                show_animation_panel = false;
                if (docking_enabled) dockToBottom("Timeline");
            } else {
                show_animation_panel = true;
                timeline.setEditorMode(TimelineEditorMode::DopeSheet);
                closeOtherBottomPanels(0);
                focus_bottom_panel_next_frame = true;
            }
        }
        handleBottomTabEvents(0, "Timeline", show_animation_panel, [&]() {
            show_animation_panel = true;
            timeline.setEditorMode(TimelineEditorMode::DopeSheet);
            closeOtherBottomPanels(0);
            focus_bottom_panel_next_frame = true;
        });

        bool active_graph = show_animation_panel && (timeline.getEditorMode() == TimelineEditorMode::GraphEditor);
        if (UIWidgets::HorizontalTab("Graph Editor", UIWidgets::IconType::Graph, active_graph))
        {
            if (active_graph) {
                show_animation_panel = false;
                if (docking_enabled) dockToBottom("Timeline");
            } else {
                show_animation_panel = true;
                timeline.setEditorMode(TimelineEditorMode::GraphEditor);
                closeOtherBottomPanels(0);
                focus_bottom_panel_next_frame = true;
            }
        }
        handleBottomTabEvents(1, "Timeline", show_animation_panel, [&]() {
            show_animation_panel = true;
            timeline.setEditorMode(TimelineEditorMode::GraphEditor);
            closeOtherBottomPanels(0);
            focus_bottom_panel_next_frame = true;
        });

        if (UIWidgets::HorizontalTab("Console", UIWidgets::IconType::Console, show_scene_log))
        {
            show_scene_log = !show_scene_log;
            if (show_scene_log) { 
                closeOtherBottomPanels(1);
                focus_bottom_panel_next_frame = true;
            } else {
                if (docking_enabled) dockToBottom("Console");
            }
        }
        handleBottomTabEvents(2, "Console", show_scene_log, [&]() {
            show_scene_log = true;
            closeOtherBottomPanels(1);
            focus_bottom_panel_next_frame = true;
        });

        if (UIWidgets::HorizontalTab("Python", UIWidgets::IconType::Console, show_python_console)) {
            show_python_console = !show_python_console;
        }
        
        if (UIWidgets::HorizontalTab("Graph", UIWidgets::IconType::Graph, show_terrain_graph))
        {
            show_terrain_graph = !show_terrain_graph;
            if (show_terrain_graph) {
                closeOtherBottomPanels(2);
                focus_bottom_panel_next_frame = true;
            } else {
                if (docking_enabled) dockToBottom("Terrain Graph");
            }
        }
        handleBottomTabEvents(3, "Terrain Graph", show_terrain_graph, [&]() {
            show_terrain_graph = true;
            closeOtherBottomPanels(2);
            focus_bottom_panel_next_frame = true;
        });
        
        if (UIWidgets::HorizontalTab("AnimGraph", UIWidgets::IconType::AnimGraph, show_anim_graph))
        {
            show_anim_graph = !show_anim_graph;
            if (show_anim_graph) {
                closeOtherBottomPanels(3);
                focus_bottom_panel_next_frame = true;
            } else {
                if (docking_enabled) dockToBottom("AnimGraph");
            }
        }
        handleBottomTabEvents(4, "AnimGraph", show_anim_graph, [&]() {
            show_anim_graph = true;
            closeOtherBottomPanels(3);
            focus_bottom_panel_next_frame = true;
        });

        if (UIWidgets::HorizontalTab("Geometry", UIWidgets::IconType::Mesh, show_geometry_graph))
        {
            show_geometry_graph = !show_geometry_graph;
            if (show_geometry_graph) {
                closeOtherBottomPanels(5);
                focus_bottom_panel_next_frame = true;
            } else {
                if (docking_enabled) dockToBottom("Geometry Graph");
            }
        }
        handleBottomTabEvents(5, "Geometry Graph", show_geometry_graph, [&]() {
            show_geometry_graph = true;
            closeOtherBottomPanels(5);
            focus_bottom_panel_next_frame = true;
        });

        if (UIWidgets::HorizontalTab("Material", UIWidgets::IconType::Graph, show_material_graph))
        {
            show_material_graph = !show_material_graph;
            if (show_material_graph) {
                closeOtherBottomPanels(6);
                focus_bottom_panel_next_frame = true;
            } else {
                if (docking_enabled) dockToBottom("Material Graph");
            }
        }
        handleBottomTabEvents(7, "Material Graph", show_material_graph, [&]() {
            show_material_graph = true;
            closeOtherBottomPanels(6);
            focus_bottom_panel_next_frame = true;
        });

        if (UIWidgets::HorizontalTab("Assets", UIWidgets::IconType::Assets, show_asset_browser))
        {
            show_asset_browser = !show_asset_browser;
            if (show_asset_browser) {
                closeOtherBottomPanels(4);
                focus_bottom_panel_next_frame = true;
            } else {
                if (docking_enabled) dockToBottom("Asset Browser");
            }
        }
        handleBottomTabEvents(6, "Asset Browser", show_asset_browser, [&]() {
            show_asset_browser = true;
            closeOtherBottomPanels(4);
            focus_bottom_panel_next_frame = true;
        });

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
            // drawPill moves the cursor via SetCursorScreenPos without submitting an
            // item; anchor it so ImGui 1.92 doesn't warn about extending boundaries.
            ImGui::Dummy(ImVec2(0.0f, 0.0f));
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
    bool show_bottom = (show_animation_panel || show_scene_log || show_terrain_graph || show_geometry_graph || show_material_graph || show_anim_graph || show_asset_browser);
    if (!show_bottom) return;

    // Use class member for persistent height
    // static float bottom_height = 280.0f; <-- REMOVED
    const float min_height = 100.0f;
    const float max_height = screen_y * 0.6f;  // Max 60% of screen
    const float resize_handle_height = 14.0f;
    const bool viewport_is_stable = screen_y > (min_height + status_bar_height + 80.0f);

    if (preferred_bottom_panel_height < min_height) {
        preferred_bottom_panel_height = bottom_panel_height;
    }

    float effective_bottom_panel_height = bottom_panel_height;
    if (viewport_is_stable) {
        preferred_bottom_panel_height = (std::max)(preferred_bottom_panel_height, min_height);
        effective_bottom_panel_height = std::clamp(preferred_bottom_panel_height, min_height, max_height);
        bottom_panel_height = effective_bottom_panel_height;
    } else {
        effective_bottom_panel_height = (std::min)(bottom_panel_height, (std::max)(0.0f, screen_y - status_bar_height));
    }

    // Calculate panel position
    float panel_top = screen_y - effective_bottom_panel_height - status_bar_height;

    // --- RESIZE HANDLE (invisible button at top edge) ---
    // Docking provides its own splitter between dock nodes, so skip the custom handle.
    if (!docking_enabled) {
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
        const bool resize_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
        draw_list->AddLine(
            p1,
            p2,
            resize_hovered ? IM_COL32(140, 140, 140, 220) : IM_COL32(100, 100, 100, 180),
            resize_hovered ? 3.0f : 2.0f);
        
        // Handle resize dragging
        ImGui::InvisibleButton("##ResizeHandle", ImVec2(screen_x, resize_handle_height));
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            preferred_bottom_panel_height -= ImGui::GetIO().MouseDelta.y;
            preferred_bottom_panel_height = std::clamp(preferred_bottom_panel_height, min_height, max_height);
            bottom_panel_height = preferred_bottom_panel_height;
            effective_bottom_panel_height = bottom_panel_height;
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    } // end if (!docking_enabled) custom resize handle

    // --- MAIN BOTTOM PANEL ---
    // Legacy: pin to the bottom strip. Docking: the dock node owns geometry.
    if (docking_enabled) {
        if (show_animation_panel) {
            bool open = true;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
            ImGui::SetNextWindowSize(ImVec2(900, 300), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2((screen_x - 900) * 0.5f, screen_y - 300 - 50), ImGuiCond_FirstUseEver);
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;
            if (!isBottomPanelFloating("Timeline")) {
                flags |= ImGuiWindowFlags_NoTitleBar;
            }
            if (focus_bottom_panel_next_frame) {
                ImGui::SetNextWindowFocus();
            }
            if (ImGui::Begin("Timeline", &open, flags)) {
                if (focus_bottom_panel_next_frame) {
                    ImGui::SetWindowFocus();
                }
                drawTimelineContent(ctx);
            }
            ImGui::End();
            ImGui::PopStyleVar();
            if (!open) {
                show_animation_panel = false;
                dockToBottom("Timeline");
            }
        }

        if (show_scene_log) {
            bool open = true;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
            ImGui::SetNextWindowSize(ImVec2(800, 400), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2((screen_x - 800) * 0.5f, (screen_y - 400) * 0.5f), ImGuiCond_FirstUseEver);
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;
            if (!isBottomPanelFloating("Console")) {
                flags |= ImGuiWindowFlags_NoTitleBar;
            }
            if (focus_bottom_panel_next_frame) {
                ImGui::SetNextWindowFocus();
            }
            if (ImGui::Begin("Console", &open, flags)) {
                if (focus_bottom_panel_next_frame) {
                    ImGui::SetWindowFocus();
                }
                drawLogPanelEmbedded();
            }
            ImGui::End();
            ImGui::PopStyleVar();
            if (!open) {
                show_scene_log = false;
                dockToBottom("Console");
            }
        }

        if (show_terrain_graph) {
            bool open = true;
            terrain_graph_focused = false;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
            ImGui::SetNextWindowSize(ImVec2(950, 450), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2((screen_x - 950) * 0.5f, (screen_y - 450) * 0.5f), ImGuiCond_FirstUseEver);
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;
            if (!isBottomPanelFloating("Terrain Graph")) {
                flags |= ImGuiWindowFlags_NoTitleBar;
            }
            if (focus_bottom_panel_next_frame) {
                ImGui::SetNextWindowFocus();
            }
            if (ImGui::Begin("Terrain Graph", &open, flags)) {
                if (focus_bottom_panel_next_frame) {
                    ImGui::SetWindowFocus();
                }
                terrain_graph_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
                terrain_brush.enabled = false;
                TerrainObject* activeTerrain = nullptr;
                if (terrain_brush.active_terrain_id != -1) {
                    activeTerrain = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
                }
                if (activeTerrain) {
                    if (!activeTerrain->nodeGraph) {
                        activeTerrain->nodeGraph = std::make_shared<TerrainNodesV2::TerrainNodeGraphV2>();
                    }
                    if (activeTerrain->nodeGraph->nodes.empty()) {
                        activeTerrain->nodeGraph->createDefaultGraph(activeTerrain);
                    }
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
                    terrainNodeEditorUI.onFoliageThumbnail = [this, &ctx](
                        const std::string& relativePath, int& width, int& height) -> ImTextureID {
                        for (const auto& asset : FoliageAssets::catalog(false).getAssets()) {
                            if (asset.relative_entry_path.generic_string() != relativePath ||
                                !asset.has_preview) continue;
                            SDL_Texture* texture = nullptr;
                            if (ensureAssetBrowserThumbnailTexture(
                                    ctx, asset.preview_path, texture, width, height) && texture) {
                                return (ImTextureID)texture;
                            }
                            break;
                        }
                        width = 0;
                        height = 0;
                        return ImTextureID{};
                    };
                    terrainNodeEditorUI.onFoliageScattered = [&ctx](
                        TerrainObject* terrain, const std::vector<int>& groupIds) {
                        SceneUI::syncNodeFoliageToScene(ctx, terrain, groupIds);
                    };
                    terrainNodeEditorUI.draw(ctx, *activeTerrain->nodeGraph, activeTerrain);
                } else {
                     ImGui::TextColored(ImVec4(1, 1, 0, 1), "Please select a terrain to edit its node graph.");
                }
            }
            ImGui::End();
            ImGui::PopStyleVar();
            if (!open) {
                show_terrain_graph = false;
                terrain_graph_focused = false;
                dockToBottom("Terrain Graph");
            }
        } else {
            terrain_graph_focused = false;
        }

        if (show_geometry_graph) {
            bool open = true;
            geometry_graph_focused = false;  // re-armed below only if the window is actually focused this frame
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
            ImGui::SetNextWindowSize(ImVec2(950, 450), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2((screen_x - 950) * 0.5f, (screen_y - 450) * 0.5f), ImGuiCond_FirstUseEver);
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;
            if (!isBottomPanelFloating("Geometry Graph")) {
                flags |= ImGuiWindowFlags_NoTitleBar;
            }
            if (focus_bottom_panel_next_frame) {
                ImGui::SetNextWindowFocus();
            }
            if (ImGui::Begin("Geometry Graph", &open, flags)) {
                if (focus_bottom_panel_next_frame) {
                    ImGui::SetWindowFocus();
                }
                geometry_graph_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

                // Follow the current scene selection: switching to a different mesh object
                // shows THAT object's own graph (auto Base Mesh if it doesn't have one yet)
                // instead of silently continuing to operate on whatever was bound before.
                // Anything OTHER than a mesh object being selected — nothing selected at all
                // (e.g. the bound object was just deleted), or a non-mesh selection like a
                // light/camera — drops the stale binding, rather than leaving the panel showing
                // (and Evaluate silently operating on) whatever mesh was bound before.
                if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
                    const std::string curSel = ctx.selection.selected.object->getNodeName();
                    if (!curSel.empty() && curSel != geometry_graph_active_object_name) {
                        geometry_graph_active_object_name = curSel;
                    }
                } else {
                    geometry_graph_active_object_name.clear();
                }

                const std::string& objName = geometry_graph_active_object_name;
                if (objName.empty()) {
                    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Select a mesh object and click 'Edit Geometry Graph' to open its graph.");
                } else {
                    auto& graphPtr = ctx.scene.geometry_node_graphs[objName];
                    geometryNodeEditorUI.config.showMinimap = true;
                    if (!graphPtr) {
                        graphPtr = std::make_shared<GeometryNodesV2::GeometryNodeGraphV2>();
                    }
                    if (graphPtr->nodes.empty()) {
                        // Default graph: Base Mesh wired straight into Output, ready to insert
                        // nodes into the middle of. Output makes the result deterministic even
                        // once several branches exist (see evaluateGeometryGraph) instead of
                        // leaving it to whichever terminal node happens to be found first.
                        auto* baseMeshNode = graphPtr->addGeometryNode(GeometryNodesV2::NodeType::BaseMesh, 60, 120);
                        auto* outputNode = graphPtr->addGeometryNode(GeometryNodesV2::NodeType::Output, 340, 120);
                        if (baseMeshNode && outputNode && !baseMeshNode->outputs.empty() && !outputNode->inputs.empty()) {
                            graphPtr->addLink(baseMeshNode->outputs[0].id, outputNode->inputs[0].id);
                        }
                    }
                    // Keep the Base Mesh node's title showing which object it's bound to —
                    // it implicitly reads GeometryContext::baseMesh (== this object), and with
                    // only one allowed per graph (enforced in drawGeometryGraphToolbar) there's
                    // no ambiguity, but the label makes that explicit instead of silent.
                    for (auto& n : graphPtr->nodes) {
                        if (n->getTypeId() == "GeoV2.BaseMesh") {
                            n->metadata.displayName = "Base Mesh: " + objName;
                        }
                    }
                    drawGeometryGraphToolbar(ctx, objName, *graphPtr);
                    ImGui::Separator();

                    // Right-click on a NODE now opens NodeEditorUIV2's own built-in
                    // "LocalNodeContextPopup" (Delete Node + group management) automatically —
                    // that popup body already existed but nothing used to call OpenPopup for it,
                    // fixed at the source (NodeEditorUIV2.h) so every node graph in the app
                    // benefits, not just this one.
                    // Right-click on EMPTY CANVAS opens the library's "LocalGraphContextPopup"
                    // (group tools); we append our own "Add Base Mesh"/"Add Subdivide" items into
                    // that SAME popup via the onDrawBackgroundMenu hook (also previously declared
                    // but unused) instead of racing it with a second competing popup. Reassigned
                    // every frame since it captures this frame's `graphPtr` by reference and is
                    // only ever invoked synchronously inside `draw()` below, never stored.
                    geometryNodeEditorUI.onDrawBackgroundMenu = [this, &graphPtr]() {
                        bool hasBaseMesh = false, hasOutput = false;
                        for (auto& n : graphPtr->nodes) {
                            if (n->getTypeId() == "GeoV2.BaseMesh") hasBaseMesh = true;
                            else if (n->getTypeId() == "GeoV2.Output") hasOutput = true;
                        }
                        
                        ImVec2 spawnPos = geometryNodeEditorUI.mousePosOnRightClick;
                        auto addNodeHelper = [&](GeometryNodesV2::NodeType type) {
                            auto* n = graphPtr->addGeometryNode(type, spawnPos.x, spawnPos.y);
                            geometryNodeEditorUI.onNodeAdded(*graphPtr, n);
                        };

                        if (ImGui::BeginMenu("Input / Output")) {
                            if (ImGui::MenuItem("Base Mesh", nullptr, false, !hasBaseMesh)) {
                                addNodeHelper(GeometryNodesV2::NodeType::BaseMesh);
                            }
                            if (ImGui::MenuItem("Object Source")) {
                                addNodeHelper(GeometryNodesV2::NodeType::ObjectSource);
                            }
                            if (ImGui::MenuItem("Output", nullptr, false, !hasOutput)) {
                                addNodeHelper(GeometryNodesV2::NodeType::Output);
                            }
                            ImGui::EndMenu();
                        }
                        if (ImGui::BeginMenu("Mesh Modifiers")) {
                            if (ImGui::MenuItem("Subdivide")) {
                                addNodeHelper(GeometryNodesV2::NodeType::SubdivideCC);
                            }
                            if (ImGui::MenuItem("Transform")) {
                                addNodeHelper(GeometryNodesV2::NodeType::Transform);
                            }
                            if (ImGui::MenuItem("Mirror")) {
                                addNodeHelper(GeometryNodesV2::NodeType::Mirror);
                            }
                            if (ImGui::MenuItem("Array")) {
                                addNodeHelper(GeometryNodesV2::NodeType::Array);
                            }
                            if (ImGui::MenuItem("Extrude")) {
                                addNodeHelper(GeometryNodesV2::NodeType::Extrude);
                            }
                            if (ImGui::MenuItem("Inset")) {
                                addNodeHelper(GeometryNodesV2::NodeType::Inset);
                            }
                            if (ImGui::MenuItem("Bevel")) {
                                addNodeHelper(GeometryNodesV2::NodeType::Bevel);
                            }
                            if (ImGui::MenuItem("Remesh")) {
                                addNodeHelper(GeometryNodesV2::NodeType::Remesh);
                            }
                            if (ImGui::MenuItem("Noise Displace")) {
                                addNodeHelper(GeometryNodesV2::NodeType::NoiseDisplace);
                            }
                            ImGui::EndMenu();
                        }
                        if (ImGui::BeginMenu("Topology / Combine")) {
                            if (ImGui::MenuItem("Merge (Join)")) {
                                addNodeHelper(GeometryNodesV2::NodeType::Merge);
                            }
                            if (ImGui::MenuItem("Weld")) {
                                addNodeHelper(GeometryNodesV2::NodeType::Weld);
                            }
                            ImGui::EndMenu();
                        }
                        if (ImGui::BeginMenu("Masking")) {
                            if (ImGui::MenuItem("Mask by Height")) {
                                addNodeHelper(GeometryNodesV2::NodeType::MaskByHeight);
                            }
                            if (ImGui::MenuItem("Mask by Slope")) {
                                addNodeHelper(GeometryNodesV2::NodeType::MaskBySlope);
                            }
                            if (ImGui::MenuItem("Mask by Noise")) {
                                addNodeHelper(GeometryNodesV2::NodeType::MaskNoise);
                            }
                            if (ImGui::MenuItem("Mask Remap")) {
                                addNodeHelper(GeometryNodesV2::NodeType::MaskRemap);
                            }
                            if (ImGui::MenuItem("Mask Math")) {
                                addNodeHelper(GeometryNodesV2::NodeType::MaskMath);
                            }
                            ImGui::EndMenu();
                        }
                        if (ImGui::BeginMenu("Scattering")) {
                            if (ImGui::MenuItem("Scatter Instances")) {
                                addNodeHelper(GeometryNodesV2::NodeType::ScatterInstances);
                            }
                            ImGui::EndMenu();
                        }
                    };

                    // Object Source node's picker combo. Reassigned every frame (captures
                    // this frame's ctx by reference — same lifetime contract as
                    // onDrawBackgroundMenu above, only invoked synchronously inside this
                    // frame's properties-panel drawContent() below). The list is built
                    // lazily, only while the combo popup is open, from direct_mesh_nodes —
                    // so a freshly imported/added object shows up the next time the picker
                    // is opened, with zero per-frame tracking cost.
                    GeometryNodesV2::g_sceneObjectListProvider = [this, &ctx]() {
                        if (!mesh_cache_valid) {
                            rebuildMeshCache(ctx.scene.world.objects);
                        }
                        std::vector<std::string> names;
                        names.reserve(direct_mesh_nodes.size());
                        for (const auto& kv : direct_mesh_nodes) {
                            if (kv.second.mesh) names.push_back(kv.first);
                        }
                        std::sort(names.begin(), names.end());
                        return names;
                    };

                    if (geometry_graph_show_properties) {
                        float availWidth = ImGui::GetContentRegionAvail().x;
                        float propsWidth = geometry_graph_properties_width;
                        float canvasWidth = availWidth - propsWidth - 8.0f;
                        if (canvasWidth < 100.0f) {
                            canvasWidth = 100.0f;
                            propsWidth = availWidth - canvasWidth - 8.0f;
                        }

                        ImGui::BeginChild("GeoDagCanvas", ImVec2(canvasWidth, 0), false);
                        geometryNodeEditorUI.draw(*graphPtr);
                        ImGui::EndChild();

                        ImGui::SameLine();

                        // Splitter Bar
                        ImGui::Button("##GeoDagSplitter", ImVec2(4.0f, -1.0f));
                        if (ImGui::IsItemActive()) {
                            propsWidth -= ImGui::GetIO().MouseDelta.x;
                            if (propsWidth < 150.0f) propsWidth = 150.0f;
                            if (propsWidth > availWidth - 150.0f) propsWidth = availWidth - 150.0f;
                            geometry_graph_properties_width = propsWidth;
                        }
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                        }

                        ImGui::SameLine();
                        ImGui::BeginChild("GeoDagProperties", ImVec2(0, 0), true);
                        NodeSystem::NodeBase* selected = graphPtr->getNode(geometryNodeEditorUI.selectedNodeId);
                        if (!selected) {
                            ImGui::TextDisabled("Select a node to edit its parameters.");
                        } else {
                            const std::string title = selected->metadata.displayName.empty() ? selected->name : selected->metadata.displayName;
                            ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f), "%s", title.c_str());
                            if (!selected->metadata.description.empty()) {
                                ImGui::TextWrapped("%s", selected->metadata.description.c_str());
                            }
                            ImGui::Separator();
                            ImGui::Spacing();
                            ImGui::PushItemWidth(-1.0f);
                            selected->drawContent();
                            ImGui::PopItemWidth();
                        }
                        ImGui::EndChild();
                    } else {
                        ImGui::BeginChild("GeoDagCanvas", ImVec2(0, 0), false);
                        geometryNodeEditorUI.draw(*graphPtr);
                        ImGui::EndChild();
                    }
                }
            }
            ImGui::End();
            ImGui::PopStyleVar();
            if (!open) {
                show_geometry_graph = false;
                geometry_graph_focused = false;
                geometry_graph_active_object_name.clear();
                dockToBottom("Geometry Graph");
                // The provider captured this frame's ctx by reference — clear it so a
                // stale copy can never outlive the panel (nothing else invokes it, but
                // a dangling std::function is not worth the risk).
                GeometryNodesV2::g_sceneObjectListProvider = nullptr;
            }
        } else {
            geometry_graph_focused = false;
        }

        if (show_material_graph) {
            bool open = true;
            material_graph_focused = false;  // re-armed below only if the window is actually focused this frame
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
            ImGui::SetNextWindowSize(ImVec2(950, 450), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2((screen_x - 950) * 0.5f, (screen_y - 450) * 0.5f), ImGuiCond_FirstUseEver);
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;
            if (!isBottomPanelFloating("Material Graph")) {
                flags |= ImGuiWindowFlags_NoTitleBar;
            }
            if (focus_bottom_panel_next_frame) {
                ImGui::SetNextWindowFocus();
            }
            if (ImGui::Begin("Material Graph", &open, flags)) {
                if (focus_bottom_panel_next_frame) {
                    ImGui::SetWindowFocus();
                }
                material_graph_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

                if (!materialNodeEditorUI.onOpenFileDialog) {
                    materialNodeEditorUI.onOpenFileDialog = [](const wchar_t* filter) -> std::string {
                        return SceneUI::openFileDialogW(filter);
                    };
                }
                materialNodeEditorUI.draw(ctx, ctx.scene.material_node_graphs);
            }
            ImGui::End();
            ImGui::PopStyleVar();
            if (!open) {
                show_material_graph = false;
                material_graph_focused = false;
                dockToBottom("Material Graph");
            }
        } else {
            material_graph_focused = false;
        }

        if (show_anim_graph) {
            bool open = true;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
            ImGui::SetNextWindowSize(ImVec2(950, 450), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2((screen_x - 950) * 0.5f, (screen_y - 450) * 0.5f), ImGuiCond_FirstUseEver);
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;
            if (!isBottomPanelFloating("AnimGraph")) {
                flags |= ImGuiWindowFlags_NoTitleBar;
            }
            if (focus_bottom_panel_next_frame) {
                ImGui::SetNextWindowFocus();
            }
            if (ImGui::Begin("AnimGraph", &open, flags)) {
                if (focus_bottom_panel_next_frame) {
                    ImGui::SetWindowFocus();
                }
                drawAnimationGraphPanel(ctx);
            }
            ImGui::End();
            ImGui::PopStyleVar();
            if (!open) {
                show_anim_graph = false;
                dockToBottom("AnimGraph");
            }
        }

        if (show_asset_browser) {
            bool open = true;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
            ImGui::SetNextWindowSize(ImVec2(900, 400), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2((screen_x - 900) * 0.5f, (screen_y - 400) * 0.5f), ImGuiCond_FirstUseEver);
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;
            if (!isBottomPanelFloating("Asset Browser")) {
                flags |= ImGuiWindowFlags_NoTitleBar;
            }
            if (focus_bottom_panel_next_frame) {
                ImGui::SetNextWindowFocus();
            }
            if (ImGui::Begin("Asset Browser", &open, flags)) {
                if (focus_bottom_panel_next_frame) {
                    ImGui::SetWindowFocus();
                }
                drawAssetBrowser(ctx, true);
            }
            ImGui::End();
            ImGui::PopStyleVar();
            if (!open) {
                show_asset_browser = false;
                dockToBottom("Asset Browser");
            }
        }
        
        if (focus_bottom_panel_next_frame) {
            focus_bottom_panel_next_frame = false;
        }
    } else {
        // Legacy single BottomPanel
        ImGui::SetNextWindowPos(ImVec2(0, panel_top), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(screen_x, effective_bottom_panel_height), ImGuiCond_Always);
        if (focus_bottom_panel_next_frame) {
            ImGui::SetNextWindowFocus();
        }
        ImGui::SetNextWindowBgAlpha(panel_alpha);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
        ImGuiWindowFlags bottom_flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        if (ImGui::Begin("BottomPanel", nullptr, bottom_flags))
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
                terrain_brush.enabled = false;
                TerrainObject* activeTerrain = nullptr;
                if (terrain_brush.active_terrain_id != -1) {
                    activeTerrain = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
                }
                if (activeTerrain) {
                    if (!activeTerrain->nodeGraph) {
                        activeTerrain->nodeGraph = std::make_shared<TerrainNodesV2::TerrainNodeGraphV2>();
                    }
                    if (activeTerrain->nodeGraph->nodes.empty()) {
                        activeTerrain->nodeGraph->createDefaultGraph(activeTerrain);
                    }
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
                    terrainNodeEditorUI.onFoliageThumbnail = [this, &ctx](
                        const std::string& relativePath, int& width, int& height) -> ImTextureID {
                        for (const auto& asset : FoliageAssets::catalog(false).getAssets()) {
                            if (asset.relative_entry_path.generic_string() != relativePath ||
                                !asset.has_preview) continue;
                            SDL_Texture* texture = nullptr;
                            if (ensureAssetBrowserThumbnailTexture(
                                    ctx, asset.preview_path, texture, width, height) && texture) {
                                return (ImTextureID)texture;
                            }
                            break;
                        }
                        width = 0;
                        height = 0;
                        return ImTextureID{};
                    };
                    terrainNodeEditorUI.onFoliageScattered = [&ctx](
                        TerrainObject* terrain, const std::vector<int>& groupIds) {
                        SceneUI::syncNodeFoliageToScene(ctx, terrain, groupIds);
                    };
                    terrainNodeEditorUI.draw(ctx, *activeTerrain->nodeGraph, activeTerrain);
                }
                else {
                     ImGui::TextColored(ImVec4(1, 1, 0, 1), "Please select a terrain to edit its node graph.");
                }
            }
            else if (show_anim_graph) {
                drawAnimationGraphPanel(ctx);
            }
            else if (show_asset_browser) {
                drawAssetBrowser(ctx, true);
            }
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }
    // ImGui::PopStyleVar(); // Removed redundant PopStyleVar
    // ImGui::PopStyleColor(); // Removed hardcoded color push

    // Draw flash borders
    for (int i = 0; i < 7; i++) {
        if (bottom_flash_timers[i] > 0.0f) {
            bottom_flash_timers[i] -= ImGui::GetIO().DeltaTime;
            const char* name = nullptr;
            if (i == 0 || i == 1) name = "Timeline";
            else if (i == 2) name = "Console";
            else if (i == 3) name = "Terrain Graph";
            else if (i == 4) name = "AnimGraph";
            else if (i == 5) name = "Geometry Graph";
            else if (i == 6) name = "Asset Browser";

            if (name) {
                ImGuiWindow* win = ImGui::FindWindowByName(name);
                if (win && win->Active && !win->Hidden) {
                    ImVec2 min_p = win->Pos;
                    ImVec2 max_p = ImVec2(win->Pos.x + win->Size.x, win->Pos.y + win->Size.y);
                    ImDrawList* fg_list = ImGui::GetForegroundDrawList();
                    
                    float alpha = bottom_flash_timers[i];
                    for (int g = 1; g <= 4; g++) {
                        float offset = (float)g;
                        fg_list->AddRect(
                            ImVec2(min_p.x - offset, min_p.y - offset),
                            ImVec2(max_p.x + offset, max_p.y + offset),
                            IM_COL32(0, 160, 255, (int)(alpha * 255 / g)),
                            4.0f,
                            0,
                            1.5f
                        );
                    }
                }
            }
        }
    }
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
    drawParticleDebugOverlay(ctx);

    // === PRO CAMERA HUD OVERLAYS ===
    // These are drawn on top of everything else
    drawHistogramOverlay(ctx);
    drawFocusPeakingOverlay(ctx);
    drawZebraOverlay(ctx);
    drawAFPointsOverlay(ctx);

    // Asset Browser -> Viewport drag & drop target
    // Keep this path fully dormant unless the asset browser is actually visible.
    if (show_asset_browser && ImGui::IsDragDropActive()) {
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
            (show_animation_panel || show_scene_log || show_terrain_graph || show_geometry_graph || show_material_graph || show_anim_graph || show_asset_browser);
        const float menu_height = getMainMenuReservedHeight();
        const float status_bar_height = 24.0f;
        const float left_offset = showSidePanel ? side_panel_width : 0.0f;
        const float top = menu_height;
        const float right = ImGui::GetIO().DisplaySize.x;

        bool bottom_docked = false;
        if (show_bottom) {
            if (!docking_enabled) {
                bottom_docked = true;
            } else {
                ImGuiID dockspace_id = this->dockspace_id;
                for (const char* name : {"Timeline", "Console", "Terrain Graph", "Geometry Graph", "AnimGraph", "Asset Browser"}) {
                    ImGuiWindow* win = ImGui::FindWindowByName(name);
                    if (win && win->Active && win->DockNode) {
                        ImGuiDockNode* node = win->DockNode;
                        while (node->ParentNode) {
                            node = node->ParentNode;
                        }
                        if (node->ID == dockspace_id) {
                            bottom_docked = true;
                            break;
                        }
                    }
                }
            }
        }

        const float bottom = ImGui::GetIO().DisplaySize.y -
            status_bar_height -
            (bottom_docked ? bottom_panel_height : 0.0f);
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
    if (rtpython::wantsInputCapture() || rtapi::renderOutputPending()) return;
    bool mesh_paint_locked = false;
    if (paint_mode_state.enabled && paint_mode_state.hasValidTarget()) {
        if (auto mesh_adapter = std::dynamic_pointer_cast<Paint::MeshPaintAdapter>(paint_mode_state.getAdapter())) {
            if (auto* texture_set = mesh_adapter->getTextureSet()) {
                mesh_paint_locked = texture_set->initialized;
            }
        }
    }

    if ((paint_mode_state.enabled && paint_mode_state.stroke.active) || mesh_paint_locked) {
        return;
    }

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
        
        if (Backend::IBackend* renderBackend = getSceneUiRenderBackend(ctx)) {
            renderBackend->updateGeometry(ctx.scene.world.objects);
            renderBackend->resetAccumulation();
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
    
    if (show_procedural_generator) {
        drawProceduralGeneratorWindow(ctx);
    }
}

bool SceneUI::raycastViewportPlacement(UIContext& ctx, const ImVec2& screen_pos, Vec3& hit_point, Vec3& hit_normal) const
{
    HitRecord rec;
    if (raycastViewportHit(ctx, screen_pos, rec)) {
        hit_point = rec.point;
        hit_normal = rec.normal.length_squared() > 0.0001f ? rec.normal.normalize() : Vec3(0, 1, 0);
        return true;
    }

    if (!ctx.scene.camera) {
        return false;
    }

    const ImVec2 display = ImGui::GetIO().DisplaySize;
    if (display.x <= 1.0f || display.y <= 1.0f) {
        return false;
    }

    const float u = std::clamp(screen_pos.x / display.x, 0.0f, 1.0f);
    const float v = std::clamp(1.0f - (screen_pos.y / display.y), 0.0f, 1.0f);
    Ray ray = ctx.scene.camera->get_ray(u, v);

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

bool SceneUI::raycastViewportHit(UIContext& ctx, const ImVec2& screen_pos, HitRecord& hit_record) const
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
    return ctx.scene.bvh->hit(ray, 0.001f, 1e30f, hit_record);
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
    auto [loaded_triangles, loaded_animations, loaded_bone_data] =
        loader->loadModelToTriangles(asset.entry_path.string(), nullptr, "", false);
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
        applyVDBImportOrientation(*vdb, vdb_import_orientation_preset, asset_record->source + " " + asset_record->entry_path.string());

        if (!display_name.empty()) {
            vdb->name = display_name;
        }

        ctx.scene.addVDBVolume(vdb);
        ctx.scene.world.objects.push_back(vdb);
        ctx.selection.selectVDBVolume(vdb, static_cast<int>(ctx.scene.vdb_volumes.size()) - 1, vdb->name);
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();

        if (ctx.backend_ptr) {
            const bool has_vulkan_viewport_path =
                (dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) != nullptr) ||
                (g_viewport_backend != nullptr);
            if (has_vulkan_viewport_path) {
                extern bool g_viewport_raster_rebuild_pending;
                g_viewport_raster_rebuild_pending = true;
                if (sceneUiRenderBackendIsVulkan(ctx)) {
                    g_vulkan_rebuild_pending = true;
                } else if (Backend::IBackend* renderBackend = getSceneUiRenderBackend(ctx)) {
                    VolumetricRenderer::syncVolumetricData(ctx.scene, renderBackend);
                    renderBackend->resetAccumulation();
                }
            } else {
                if (Backend::IBackend* renderBackend = getSceneUiRenderBackend(ctx)) {
                    VolumetricRenderer::syncVolumetricData(ctx.scene, renderBackend);
                    renderBackend->resetAccumulation();
                }
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
        g_backend.get(),
        asset_path.string(),
        nullptr,
        true,
        import_prefix);

    if (has_drop_point && ctx.scene.world.objects.size() > objects_before) {
        Vec3 bounds_min(1e30f, 1e30f, 1e30f);
        Vec3 bounds_max(-1e30f, -1e30f, -1e30f);
        std::vector<std::shared_ptr<Transform>> new_transforms;
        std::unordered_set<Transform*> unique_transforms_set;
        std::shared_ptr<Triangle> first_new_triangle;

        for (size_t i = objects_before; i < ctx.scene.world.objects.size(); ++i) {
            // Imports collapse into a flat TriangleMesh directly in world.objects (the
            // "import-flat" pass in create_scene) rather than per-face Triangle facades, so this
            // must handle both shapes — a dynamic_pointer_cast<Triangle>-only loop found nothing
            // for flat imports, new_transforms stayed empty, and the drop-to-raycast-point
            // reposition below silently never ran (dropped assets always landed at the imported
            // file's own origin instead of the drop point).
            if (auto tmesh = std::dynamic_pointer_cast<TriangleMesh>(ctx.scene.world.objects[i])) {
                AABB bounds;
                if (tmesh->bounding_box(0.0f, 0.0f, bounds)) {
                    bounds_min.x = (std::min)(bounds_min.x, bounds.min.x);
                    bounds_min.y = (std::min)(bounds_min.y, bounds.min.y);
                    bounds_min.z = (std::min)(bounds_min.z, bounds.min.z);
                    bounds_max.x = (std::max)(bounds_max.x, bounds.max.x);
                    bounds_max.y = (std::max)(bounds_max.y, bounds.max.y);
                    bounds_max.z = (std::max)(bounds_max.z, bounds.max.z);
                }
                if (tmesh->transform && unique_transforms_set.insert(tmesh->transform.get()).second) {
                    new_transforms.push_back(tmesh->transform);
                }
                if (!first_new_triangle && tmesh->geometry && !tmesh->geometry->indices.empty()) {
                    first_new_triangle = std::make_shared<Triangle>(tmesh, 0);
                }
                continue;
            }

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
                if (unique_transforms_set.insert(th.get()).second) {
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
            // Re-bake transformed vertices after the drop reposition. Facade triangles share
            // one TriangleMesh SoA, so bake each unique mesh ONCE (the old per-face
            // updateTransformedVertices() re-transformed every shared vertex ~6× and ran a
            // dynamic_pointer_cast per face — the asset-drop cost import never paid).
            std::unordered_set<TriangleMesh*> rebaked_meshes;
            auto rebakeMesh = [&](TriangleMesh* mesh) {
                if (!mesh || !mesh->geometry) return;
                if (!rebaked_meshes.insert(mesh).second) return; // already baked this mesh
                auto& geom = *mesh->geometry;
                Vec3* P = geom.get_positions_mut();
                Vec3* N = geom.get_normals_mut();
                const Vec3* Po = geom.get_positions_orig();
                const Vec3* No = geom.get_normals_orig();
                const int vc = static_cast<int>(geom.get_vertex_count());
                Matrix4x4 fT = Matrix4x4::identity(), nT = Matrix4x4::identity();
                if (mesh->transform) { fT = mesh->transform->getFinal(); nT = mesh->transform->getNormalTransform(); }
                if (P && Po) {
                    #pragma omp parallel for schedule(static) if(vc >= 4096)
                    for (int v = 0; v < vc; ++v) {
                        P[v] = fT.transform_point(Po[v]);
                        if (N && No) N[v] = nT.transform_vector(No[v]).normalize();
                    }
                }
            };
            for (size_t i = objects_before; i < ctx.scene.world.objects.size(); ++i) {
                if (auto tmesh = std::dynamic_pointer_cast<TriangleMesh>(ctx.scene.world.objects[i])) {
                    rebakeMesh(tmesh.get());
                    continue;
                }
                auto tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[i]);
                if (!tri) continue;
                if (tri->parentMesh) {
                    rebakeMesh(tri->parentMesh.get());
                } else {
                    tri->updateTransformedVertices(); // standalone triangle
                }
            }
        }

        if (first_new_triangle) {
            ctx.selection.selectObject(first_new_triangle, -1, first_new_triangle->getNodeName());
        }
    }
    else if (ctx.scene.world.objects.size() > objects_before) {
        for (size_t i = objects_before; i < ctx.scene.world.objects.size(); ++i) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[i])) {
                ctx.selection.selectObject(tri, -1, tri->getNodeName());
                break;
            }
        }
    }

    // Restore the "foliage instances live at the TAIL of world.objects" invariant.
    // create_scene appended the new objects at the very END — i.e. AFTER any existing
    // instance tail (scatter brush or Geo-DAG Scatter node). Every CPU-side scan that
    // computes `selectable = size - getTotalInstanceCount()` (rebuildMeshCache,
    // hierarchy, picking, sync) would then treat the freshly imported object as
    // foliage and silently hide it — the GPU paths iterate ALL objects and kept
    // showing it, which made the bug look like "GPU has it, CPU caches don't".
    // Rotate [instance tail | new imports] -> [new imports | instance tail]. The tail
    // length is measured directly (walk back over HittableInstance) instead of
    // trusting getTotalInstanceCount(), which also counts transient particle-bridge
    // groups that are never expanded into world.objects.
    {
        auto& objs = ctx.scene.world.objects;
        if (objs.size() > objects_before && objects_before > 0) {
            size_t tail = 0;
            while (tail < objects_before &&
                   dynamic_cast<HittableInstance*>(objs[objects_before - 1 - tail].get()) != nullptr) {
                ++tail;
            }
            if (tail > 0) {
                std::rotate(objs.begin() + static_cast<std::ptrdiff_t>(objects_before - tail),
                            objs.begin() + static_cast<std::ptrdiff_t>(objects_before),
                            objs.end());
            }
        }
    }

    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();

    if (g_backend) {
        if (sceneUiRenderBackendIsVulkan(ctx)) {
            extern bool g_vulkan_geometry_append_pending;
            g_vulkan_geometry_append_pending = true;
        } else {
            ctx.renderer.rebuildBackendGeometry(ctx.scene);
        }
        applySceneUiPendingDeleteVisibility(ctx, g_backend.get());
        g_backend->setLights(ctx.scene.lights);
        if (ctx.scene.camera) {
            g_backend->syncCamera(*ctx.scene.camera);
        }
        g_backend->resetAccumulation();
    }

    // Always signal raster viewport rebuild so new objects appear in Solid/Matcap mode
    // (rebuildBackendGeometry only updates the render backend, not the raster viewport)
    extern bool g_vulkan_rebuild_pending;
    extern bool g_vulkan_geometry_append_pending;
    extern bool g_viewport_raster_rebuild_pending;
    g_viewport_raster_rebuild_pending = true;
    // Increment geometry generation so raster viewport rebuilds mesh list
    extern bool g_geometry_dirty;
    extern bool g_materials_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    g_geometry_dirty = true;
    g_materials_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    if (sceneUiRenderBackendIsVulkan(ctx) && !g_vulkan_geometry_append_pending) {
        g_vulkan_rebuild_pending = true;
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
    UIWidgets::PushControlSurfaceStyle(ImVec4(0.98f, 0.78f, 0.40f, 1.0f));
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
            UIWidgets::PopControlSurfaceStyle();
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
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.10f, 0.13f, 0.82f));
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
    ImGui::PopStyleColor();

    ImGui::SameLine();
    ImGui::InvisibleButton("##AssetBrowserFolderSplitter", ImVec2(6.0f, ImGui::GetContentRegionAvail().y));
    if (ImGui::IsItemActive()) {
        asset_browser_folder_width += ImGui::GetIO().MouseDelta.x;
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.10f, 0.13f, 0.82f));
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
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.07f, 0.09f, 0.12f, 0.68f));
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
        if (UIWidgets::PrimaryButton("Save Asset Metadata", ImVec2(190, 0))) {
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
        if (UIWidgets::PrimaryButton(append_button_label, ImVec2(160, 0)) && isPlaceableAssetRecord(*selected_asset)) {
            appendAssetToScene(ctx, selected_asset->entry_path, selected_asset->name);
        }
        ImGui::SameLine();
        if (UIWidgets::SecondaryButton("Generate Metadata", ImVec2(190, 0))) {
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
    ImGui::PopStyleColor();
    ImGui::EndChild();
    ImGui::PopStyleColor();
    if (!embedded) {
        ImGui::End();
    }
    UIWidgets::PopControlSurfaceStyle();
}

SceneUI::~SceneUI() {
    releaseSelectedAssetPreviewTexture();
    //releaseLayerThumbnails();
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
    // Foliage instances are always appended at the END of the objects vector
    // by InstanceManager::rebuildSceneObjects. Skip them entirely — they are
    // not individually selectable and iterating 2M+ of them with RTTI is expensive.
    size_t foliage_count = InstanceManager::getInstance().getTotalInstanceCount();
    size_t selectable_count = (foliage_count <= objects.size())
                                  ? (objects.size() - foliage_count)
                                  : objects.size();

   /* SCENE_LOG_INFO("Rebuilding selection cache: " + std::to_string(selectable_count) +
                   " selectable / " + std::to_string(objects.size()) + " total objects (" +
                   std::to_string(foliage_count) + " foliage skipped)");*/
    mesh_cache.clear();
    mesh_ui_cache.clear();
    last_synced_transforms.clear();
    mesh_overlay_cache = MeshOverlayCache{};
    // Preserve the editable cache (and its sub-element selection) for the object that is
    // actively being edited. rebuildMeshCache historically wiped editable_mesh_cache
    // unconditionally, assuming a geometry change invalidates it — but the editable cache OWNS
    // its own source_triangles (independent of world.objects / mesh_cache) and self-validates
    // in ensureEditableMeshCache (needsRebuild on object/count/transform change). The blanket
    // wipe meant any of the ~30 `if (!mesh_cache_valid) rebuildMeshCache()` call sites (and the
    // modifier-stack re-eval) firing mid-edit silently dropped the selection — e.g. dragging the
    // live Catmull-Clark Crease slider creased once then lost the edge selection ("unusable").
    // Keep it for the active edit object; ensureEditableMeshCache still rebuilds it on demand if
    // the cage topology actually changed, and topology operators wipe it explicitly themselves.
    const bool preserveEditable =
        !active_mesh_edit_object_name.empty() &&
        editable_mesh_cache.object_name == active_mesh_edit_object_name &&
        !editable_mesh_cache.vertices.empty();
    EditableMeshCache preservedEditable;
    if (preserveEditable) {
        preservedEditable = std::move(editable_mesh_cache);
    }
    editable_mesh_cache = EditableMeshCache{};
    this->tri_to_index.clear(); // Clear the lookup map
    direct_mesh_nodes.clear();  // Flat/proxy: rebuilt below for facade-less TriangleMesh nodes
    direct_mesh_rep_by_ptr.clear();
    cached_triangle_count_by_object.clear();
    cached_scene_triangle_count = 0;
    bbox_cache.clear();
    hull_candidate_cache.clear();
    selection_skin_pose_hash.clear();
    selection_outline_frame_cache.clear();
    material_slots_cache.clear();
    
    tri_to_index.reserve(selectable_count);

    std::string lastName = "";
    std::vector<std::pair<int, std::shared_ptr<Triangle>>>* lastVector = nullptr;

    for (size_t i = 0; i < selectable_count; ++i) {
        if (objects[i]->isTriangle()) {
            auto tri = std::static_pointer_cast<Triangle>(objects[i]);
            const std::string& name = tri->getNodeName().empty() ? "Unnamed" : tri->getNodeName();
            
            if (name != lastName || !lastVector) {
                lastName = name;
                lastVector = &mesh_cache[name];
            }
            
            lastVector->push_back({(int)i, tri});
            this->tri_to_index[tri.get()] = (int)i; // Store const pointer to index mapping
            continue;
        }

        // Flat/proxy migration: a dense mesh placed directly in world.objects as a single
        // TriangleMesh (no per-face facades). We keep a SINGLE representative facade (face 0) as
        // the UI handle and push it into mesh_cache exactly like a normal mesh entry — the bbox /
        // material-slot / transform-sync fast-paths all key off parentMesh->geometry (the full SoA)
        // and only need one facade as the handle, so the entire facade-based UI (hierarchy listing,
        // click selection, gizmo, transform-bake) works unchanged without materializing 12.6M
        // facades. object_index points at the TriangleMesh's slot in world.objects.
        if (auto tmesh = std::dynamic_pointer_cast<TriangleMesh>(objects[i])) {
            const std::string& name = tmesh->nodeName.empty() ? "Unnamed" : tmesh->nodeName;
            // Pick a NON-DEGENERATE representative face. Face 0 of some procedural meshes (e.g. a UV
            // sphere's first row) is a zero-area pole triangle; its per-face material can also diverge
            // from the body after edits, so using it as the UI/paint handle made the paint adapter
            // target a phantom material that no body face carries (paint silently did nothing — the
            // diagnosed Sphere "target=0, rec.matID=body vs adapter=pole" case). The rep facade's
            // faceIndex only feeds material/UV/identity lookups, so any valid face works; bbox /
            // transform read the full SoA. Scan a bounded prefix for the first face with area.
            uint32_t repFace = 0;
            if (tmesh->geometry) {
                const auto& idx = tmesh->geometry->indices;
                const Vec3* P = tmesh->geometry->get_positions_orig();
                if (!P) P = tmesh->geometry->get_positions();
                const size_t nF = idx.size() / 3;
                const size_t scanLimit = (nF < 1024) ? nF : 1024;
                if (P) {
                    for (size_t f = 0; f < scanLimit; ++f) {
                        const Vec3& pa = P[idx[f * 3 + 0]];
                        const Vec3& pb = P[idx[f * 3 + 1]];
                        const Vec3& pc = P[idx[f * 3 + 2]];
                        if ((pb - pa).cross(pc - pa).length_squared() > 1e-12f) {
                            repFace = static_cast<uint32_t>(f);
                            break;
                        }
                    }
                }
            }
            auto rep = std::make_shared<Triangle>(tmesh, repFace);
            mesh_cache[name].push_back({ (int)i, rep });
            tri_to_index[rep.get()] = (int)i;
            direct_mesh_nodes[name] = DirectMeshNode{ tmesh, (int)i, rep };
            direct_mesh_rep_by_ptr[tmesh.get()] = rep;
            lastName = name;
            lastVector = &mesh_cache[name];
            continue;
        }

        auto inst = std::dynamic_pointer_cast<HittableInstance>(objects[i]);
        if (inst && inst->source_triangles) {
            std::string inst_name = inst->node_name.empty() ? "Unnamed" : inst->node_name;
            
            // Only add the source triangles of this unique instanced mesh to mesh_cache ONCE.
            // But still update tri_to_index so picking works (it will map to the last instance).
            bool already_in_cache = false;
            if (!inst->source_triangles->empty() && inst->source_triangles->front()) {
                const std::string& first_tri_name = inst->source_triangles->front()->getNodeName().empty() ? inst_name : inst->source_triangles->front()->getNodeName();
                if (mesh_cache.find(first_tri_name) != mesh_cache.end()) {
                    already_in_cache = true;
                }
            }

            for (const auto& srcTri : *inst->source_triangles) {
                if (!srcTri) continue;
                this->tri_to_index[srcTri.get()] = (int)i;
                
                if (!already_in_cache) {
                    const std::string& tri_name = srcTri->getNodeName().empty() ? inst_name : srcTri->getNodeName();
                    if (tri_name != lastName || !lastVector) {
                        lastName = tri_name;
                        lastVector = &mesh_cache[tri_name];
                    }
                    lastVector->push_back({(int)i, srcTri});
                }
            }
        }
    }
    
    // Transfer to sequential vector for ImGui Clipper AND calculate bounding boxes AND material slots
    mesh_ui_cache.reserve(mesh_cache.size());
    for (auto& kv : mesh_cache) {
        mesh_ui_cache.push_back(kv);
        cached_triangle_count_by_object[kv.first] = kv.second.size();
        cached_scene_triangle_count += kv.second.size();
        
        // Calculate LOCAL bounding box from ORIGINAL vertices (not transformed!)
        // This allows us to properly apply the transform matrix when drawing
        Vec3 bb_min(1e10f, 1e10f, 1e10f);
        Vec3 bb_max(-1e10f, -1e10f, -1e10f);
        
        const auto& trisVector = kv.second;
        std::vector<uint16_t> mat_ids;

        if (!trisVector.empty() && trisVector[0].second->parentMesh && trisVector[0].second->parentMesh->geometry) {
            // Optimized flat geometry fast-path. A multi-material import shares one nodeName
            // across SEVERAL sibling TriangleMesh objects (one per material) — trisVector holds
            // one rep per sibling. Iterate every DISTINCT parentMesh here (not just trisVector[0])
            // so bbox and material_slots_cache cover ALL materials of the object, not just the
            // first one (previously the Materials panel only ever showed one merged material for
            // multi-material imports).
            std::unordered_set<TriangleMesh*> seenMeshes;
            for (const auto& pair : trisVector) {
                TriangleMesh* pm = pair.second->parentMesh.get();
                if (!pm || !pm->geometry || !seenMeshes.insert(pm).second) continue;

                size_t vCount = pm->geometry->get_vertex_count();
                const Vec3* origP = pm->geometry->get_positions_orig();
                if (!origP) origP = pm->geometry->get_positions();
                // Clamp to the buffer's REAL length: a regenerating sim surface can leave a
                // stale, shorter P_orig behind while vertex_count already advanced, so iterating
                // to vCount would read past the end (the observed access violation).
                const size_t posCount = (std::min)(vCount, pm->geometry->get_positions_orig_count());
                if (origP) {
                    for (size_t v = 0; v < posCount; ++v) {
                        Vec3 p = origP[v];
                        bb_min.x = fminf(bb_min.x, p.x);
                        bb_min.y = fminf(bb_min.y, p.y);
                        bb_min.z = fminf(bb_min.z, p.z);
                        bb_max.x = fmaxf(bb_max.x, p.x);
                        bb_max.y = fmaxf(bb_max.y, p.y);
                        bb_max.z = fmaxf(bb_max.z, p.z);
                    }
                }

                const uint16_t* matIDs = pm->geometry->get_material_ids();
                if (matIDs) {
                    // Build the material-slot list per FACE, skipping zero-area (degenerate) faces.
                    // A UV sphere's pole triangles are degenerate; their per-face material can diverge
                    // from the body after edits (the diagnosed "adapter mat=2 (pole) vs rec.matID=1
                    // (body)" paint failure). Including such a stray material here surfaced a phantom
                    // material slot that the paint adapter targeted but no visible face carries, so the
                    // brush matched nothing. Degenerate faces contribute no pixels — drop their material.
                    const auto& idx = pm->geometry->indices;
                    const size_t nF = idx.size() / 3;
                    uint16_t lastMid = 0xFFFF;
                    bool anyFace = false;
                    for (size_t f = 0; f < nF; ++f) {
                        const uint32_t a = idx[f * 3 + 0], b = idx[f * 3 + 1], c = idx[f * 3 + 2];
                        // Stale/short P_orig or matID buffer (regenerating surface) must not be
                        // indexed past its real length.
                        if (a >= posCount || b >= posCount || c >= posCount) continue;
                        if (origP &&
                            (origP[b] - origP[a]).cross(origP[c] - origP[a]).length_squared() <= 1e-12f) {
                            continue; // degenerate face — no visible surface, ignore its material
                        }
                        anyFace = true;
                        const uint16_t mid = matIDs[a];
                        if (mid == lastMid) continue;
                        lastMid = mid;

                        bool found = false;
                        for (uint16_t existing : mat_ids) {
                            if (existing == mid) { found = true; break; }
                        }
                        if (!found) mat_ids.push_back(mid);
                    }
                    // Safety: a fully-degenerate scan (or missing positions) must still yield a slot.
                    if (!anyFace && vCount > 0) {
                        uint16_t mid = matIDs[0];
                        bool found = false;
                        for (uint16_t existing : mat_ids) {
                            if (existing == mid) { found = true; break; }
                        }
                        if (!found) mat_ids.push_back(mid);
                    }
                }
            }
        } else {
            // Fallback for standalone/legacy triangles
            for (const auto& pair : trisVector) {
                const auto& tri = pair.second;
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

            uint16_t lastMid = 0xFFFF;
            for (const auto& pair : trisVector) {
                const auto& tri = pair.second;
                uint16_t mid = tri->getMaterialID();
                if (mid == lastMid) continue;
                lastMid = mid;
                
                bool found = false;
                for (uint16_t existing : mat_ids) {
                    if (existing == mid) { found = true; break; }
                }
                if (!found) mat_ids.push_back(mid);
            }
        }
        
        bbox_cache[kv.first] = {bb_min, bb_max};
        material_slots_cache[kv.first] = std::move(mat_ids);
    }
    
    mesh_cache_valid = true;
    last_scene_obj_count = objects.size();

    // Restore the active edit object's editable cache (see preserve note above). Only if the
    // object still exists in the rebuilt cache — otherwise it was genuinely removed and the
    // empty cache is correct.
    if (preserveEditable && mesh_cache.find(active_mesh_edit_object_name) != mesh_cache.end()) {
        editable_mesh_cache = std::move(preservedEditable);
    }

   /* SCENE_LOG_INFO("Selection cache built: " + std::to_string(mesh_cache.size()) +
                   " mesh groups, " + std::to_string(cached_scene_triangle_count) +
                   " triangles cached");*/
}

void SceneUI::syncAllTransformedVertices(struct SceneData& scene) {
    if (!mesh_cache_valid) {
        rebuildMeshCache(scene.world.objects);
    }

    // Iterate through all mesh groups in mesh_cache
    for (auto& [name, tris] : mesh_cache) {
        if (tris.empty()) continue;

        // Sync instance's transform first if it is a HittableInstance
        const int object_index = tris[0].first;
        if (object_index >= 0 && static_cast<size_t>(object_index) < scene.world.objects.size()) {
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(scene.world.objects[object_index])) {
                inst->syncTransformFromSourceTriangles();
            }
        }

        // Only sync vertices if transform has changed
        if (auto t_ptr = tris[0].second->getTransformPtr()) {
            Matrix4x4 current_xform = t_ptr->getFinal();
            auto cache_it = last_synced_transforms.find(name);
            if (cache_it == last_synced_transforms.end() || !(cache_it->second == current_xform)) {
                
                if (tris[0].second->parentMesh && tris[0].second->parentMesh->geometry) {
                    TriangleMesh* pm = tris[0].second->parentMesh.get();
                    size_t vCount = pm->geometry->get_vertex_count();
                    
                    const Vec3* origP = pm->geometry->get_positions_orig();
                    if (!origP) {
                        tris[0].second->getOriginalVertexPosition(0);
                        origP = pm->geometry->get_positions_orig();
                    }
                    const Vec3* origN = pm->geometry->get_normals_orig();
                    if (!origN) {
                        tris[0].second->getOriginalVertexNormal(0);
                        origN = pm->geometry->get_normals_orig();
                    }
                    
                    Vec3* positions = pm->geometry->get_attribute_data_mut<Vec3>("P");
                    Vec3* normals = pm->geometry->get_attribute_data_mut<Vec3>("N");
                    
                    if (origP && origN && positions && normals) {
                        Matrix4x4 normal_xform = t_ptr->getNormalTransform();
                        #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static)
                        for (int v = 0; v < (int)vCount; ++v) {
                            positions[v] = current_xform.transform_point(origP[v]);
                            normals[v] = normal_xform.transform_vector(origN[v]).normalize();
                        }
                        
                        for (auto& pair : tris) {
                            pair.second->vertexPositionsDirty = true;
                        }
                    }
                } else {
                    // Fallback for standalone/legacy triangles
                    for (auto& pair : tris) {
                        if (!pair.second->hasAnySkinWeights()) {
                            pair.second->updateTransformedVertices();
                        }
                    }
                }
                
                last_synced_transforms[name] = current_xform;
            }
        }
    }
}

void SceneUI::rebuildTriToIndex(const std::vector<std::shared_ptr<Hittable>>& objects) {
    // Lightweight rebuild: only the tri_to_index hashmap (O(N) pointer inserts).
    // Skips bbox, material slot, and mesh_ui_cache work that rebuildMeshCache does.
    // Called immediately after deletion so picking keeps working with stale BVH.
    // Skip foliage tail (always at end of vector) — not individually selectable.
    size_t foliage_count = InstanceManager::getInstance().getTotalInstanceCount();
    size_t selectable_count = (foliage_count <= objects.size())
                                  ? (objects.size() - foliage_count)
                                  : objects.size();
    tri_to_index.clear();
    tri_to_index.reserve(selectable_count);
    for (size_t i = 0; i < selectable_count; ++i) {
        if (objects[i]->isTriangle()) {
            auto tri = std::static_pointer_cast<Triangle>(objects[i]);
            tri_to_index[tri.get()] = (int)i;
            continue;
        }

        auto inst = std::dynamic_pointer_cast<HittableInstance>(objects[i]);
        if (inst && inst->source_triangles) {
            for (const auto& srcTri : *inst->source_triangles) {
                if (srcTri) {
                    tri_to_index[srcTri.get()] = (int)i;
                }
            }
        }
    }
    SCENE_LOG_INFO("rebuildTriToIndex: indexed " + std::to_string(tri_to_index.size()) + " triangles");
}

void SceneUI::invalidateCache() {
    mesh_cache_valid = false; 
    mesh_cache.clear();
    mesh_ui_cache.clear();
    last_synced_transforms.clear();
    mesh_overlay_cache = MeshOverlayCache{};
    editable_mesh_cache = EditableMeshCache{};
    cached_triangle_count_by_object.clear();
    cached_scene_triangle_count = 0;
    bbox_cache.clear();
    hull_candidate_cache.clear();
    selection_skin_pose_hash.clear();
    selection_outline_frame_cache.clear();
    material_slots_cache.clear();
    picking_vertices_synced = false;
}

// Update bounding box for a specific object (after transform)
// NOTE: Since we now store LOCAL bbox (from original vertices), this may not need
// to be called unless the mesh geometry itself changes. Transform is applied at draw time.
void SceneUI::updateBBoxCache(const std::string& objectName) {
    Vec3 bb_min(1e10f, 1e10f, 1e10f);
    Vec3 bb_max(-1e10f, -1e10f, -1e10f);

    // Flat node: read the authoritative TriangleMesh SoA from direct_mesh_nodes, NOT the rep
    // facade's parentMesh. The rep facade's parentMesh link can go stale after some edits
    // (sculpt / re-emit), and the per-facade fallback below then collapses the gizmo bbox to a
    // single triangle (the reported "sculpt → gizmo selects one triangle"). direct_mesh_nodes is
    // rebuilt straight from world.objects so it always points at the live flat mesh.
    {
        auto dmIt = direct_mesh_nodes.find(objectName);
        if (dmIt != direct_mesh_nodes.end() && dmIt->second.mesh && dmIt->second.mesh->geometry) {
            TriangleMesh* pm = dmIt->second.mesh.get();
            size_t vCount = pm->geometry->get_vertex_count();
            const Vec3* origP = pm->geometry->get_positions_orig();
            if (!origP) origP = pm->geometry->get_positions();
            if (origP) {
                for (size_t v = 0; v < vCount; ++v) {
                    Vec3 p = origP[v];
                    bb_min.x = fminf(bb_min.x, p.x); bb_min.y = fminf(bb_min.y, p.y); bb_min.z = fminf(bb_min.z, p.z);
                    bb_max.x = fmaxf(bb_max.x, p.x); bb_max.y = fmaxf(bb_max.y, p.y); bb_max.z = fmaxf(bb_max.z, p.z);
                }
                bbox_cache[objectName] = {bb_min, bb_max};
                hull_candidate_cache.erase(objectName);
                return;
            }
        }
    }

    auto it = mesh_cache.find(objectName);
    if (it == mesh_cache.end()) return;

    const auto& trisVector = it->second;
    if (!trisVector.empty() && trisVector[0].second->parentMesh && trisVector[0].second->parentMesh->geometry) {
        TriangleMesh* pm = trisVector[0].second->parentMesh.get();
        size_t vCount = pm->geometry->get_vertex_count();
        const Vec3* origP = pm->geometry->get_positions_orig();
        if (!origP) origP = pm->geometry->get_positions();
        if (origP) {
            for (size_t v = 0; v < vCount; ++v) {
                Vec3 p = origP[v];
                bb_min.x = fminf(bb_min.x, p.x);
                bb_min.y = fminf(bb_min.y, p.y);
                bb_min.z = fminf(bb_min.z, p.z);
                bb_max.x = fmaxf(bb_max.x, p.x);
                bb_max.y = fmaxf(bb_max.y, p.y);
                bb_max.z = fmaxf(bb_max.z, p.z);
            }
        }
    } else {
        for (auto& pair : trisVector) {
            auto& tri = pair.second;
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
    }
    
    bbox_cache[objectName] = {bb_min, bb_max};
    hull_candidate_cache.erase(objectName); // Force re-extraction on next selection
}

// Lazy CPU Sync - called before mouse picking to ensure vertices are up to date
// This is much more efficient than updating on gizmo release because:
// 1. Gizmo release is instant (no freeze)
// 2. Sync only happens when user actually tries to pick something
// 3. If user moves object multiple times without picking, we only sync once
void SceneUI::ensureCPUSyncForPicking(UIContext& ctx) {
    if (objects_needing_cpu_sync.empty()) return;
    
    size_t synced_count = 0;

    if (!mesh_cache_valid) {
        rebuildMeshCache(ctx.scene.world.objects);
    }
    
    // Update all pending objects - apply current transforms to vertices
    for (const auto& name : objects_needing_cpu_sync) {
        auto it = mesh_cache.find(name);
        if (it != mesh_cache.end() && !it->second.empty()) {
            auto& tris = it->second;
            const int object_index = tris[0].first;
            if (object_index >= 0 && static_cast<size_t>(object_index) < ctx.scene.world.objects.size()) {
                if (auto inst = std::dynamic_pointer_cast<HittableInstance>(ctx.scene.world.objects[object_index])) {
                    if (inst->syncTransformFromSourceTriangles()) {
                        synced_count += tris.size();
                        if (auto t_ptr = tris[0].second->getTransformPtr()) {
                            last_synced_transforms[name] = t_ptr->getFinal();
                        }
                        continue;
                    }
                }
            }

            if (tris[0].second->parentMesh && tris[0].second->parentMesh->geometry) {
                TriangleMesh* pm = tris[0].second->parentMesh.get();
                size_t vCount = pm->geometry->get_vertex_count();
                const Vec3* origP = pm->geometry->get_positions_orig();
                if (!origP) {
                    tris[0].second->getOriginalVertexPosition(0);
                    origP = pm->geometry->get_positions_orig();
                }
                const Vec3* origN = pm->geometry->get_normals_orig();
                if (!origN) {
                    tris[0].second->getOriginalVertexNormal(0);
                    origN = pm->geometry->get_normals_orig();
                }
                
                Vec3* positions = pm->geometry->get_attribute_data_mut<Vec3>("P");
                Vec3* normals = pm->geometry->get_attribute_data_mut<Vec3>("N");
                
                if (origP && origN && positions && normals) {
                    if (auto t_ptr = tris[0].second->getTransformPtr()) {
                        Matrix4x4 current_xform = t_ptr->getFinal();
                        Matrix4x4 normal_xform = t_ptr->getNormalTransform();
                        #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static)
                        for (int v = 0; v < (int)vCount; ++v) {
                            positions[v] = current_xform.transform_point(origP[v]);
                            normals[v] = normal_xform.transform_vector(origN[v]).normalize();
                        }
                        last_synced_transforms[name] = current_xform;
                    }
                    
                    for (auto& pair : tris) {
                        pair.second->vertexPositionsDirty = true;
                    }
                    synced_count += tris.size();
                }
            } else {
                for (auto& pair : tris) {
                    pair.second->updateTransformedVertices();
                    ++synced_count;
                }
                if (auto t_ptr = tris[0].second->getTransformPtr()) {
                    last_synced_transforms[name] = t_ptr->getFinal();
                }
            }
        }
    }
    
    size_t count = objects_needing_cpu_sync.size();
    objects_needing_cpu_sync.clear();
    
    if (synced_count > 0) {
        // Queue async BVH rebuild instead of blocking synchronous rebuild on click.
        // Selection can still use updated triangle hits/fallback while BVH refreshes at frame-end.
        extern bool g_bvh_rebuild_pending;
        g_bvh_rebuild_pending = true;
        ctx.renderer.resetCPUAccumulation();
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
                if (rtapi::renderOutputPending()) {
                    (void)rtapi::cancelRender();
                } else {
                    ctx.render_settings.is_final_render_mode = false;
                    extern std::atomic<bool> rendering_stopped_cpu; // Also stop loop!
                    rendering_stopped_cpu = true;
                }
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
            ImGui::Text("Resolution: %dx%d", image_width, image_height);

            ImGui::EndChild();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    
    // Safety: If window is closed, ensure we exit final render mode
    if (!show_render_window && ctx.render_settings.is_final_render_mode) {
        if (rtapi::renderOutputPending()) {
            (void)rtapi::cancelRender();
        } else {
            ctx.render_settings.is_final_render_mode = false;
        }
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
                        {"show_gizmos", viewport_settings.show_gizmos},
                        {"show_selection_outline", viewport_settings.show_selection_outline},
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
     resetMeshEditState(ctx);
     ctx.scene.base_mesh_cache.clear();
     ctx.scene.mesh_modifiers.clear();

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
     paint_mode_state = Paint::PaintModeState();
     
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
     createDefaultScene(ctx.scene, ctx.renderer, getSceneUiRenderBackend(ctx));
     
     ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
     if (Backend::IBackend* renderBackend = getSceneUiRenderBackend(ctx)) {
         ctx.renderer.rebuildBackendGeometry(ctx.scene);
         ctx.renderer.updateBackendMaterials(ctx.scene);
         ctx.renderer.updateBackendGasVolumes(ctx.scene);
         applySceneUiPendingDeleteVisibility(ctx, renderBackend);
         renderBackend->setLights(ctx.scene.lights);
         auto wd = ctx.renderer.world.getGPUData();
         renderBackend->setWorldData(&wd);
         if (ctx.scene.camera) {
             ctx.renderer.syncCameraToBackend(*ctx.scene.camera);
         }
         renderBackend->resetAccumulation();
     }
     if (Backend::IViewportBackend* viewportBackend = getSceneUiViewportBackend(ctx)) {
         viewportBackend->setLights(ctx.scene.lights);
         viewportBackend->buildRasterGeometry(ctx.scene.world.objects);
         viewportBackend->resetAccumulation();
         ctx.renderer.uploadHairToGPU();
     }
     // Signal raster viewport rebuild for Solid/Matcap mode
     extern bool g_vulkan_rebuild_pending;
     extern bool g_viewport_raster_rebuild_pending;
     g_viewport_raster_rebuild_pending = true;
     if (sceneUiRenderBackendIsVulkan(ctx)) {
         g_vulkan_rebuild_pending = true;
     }
     if(ctx.scene.camera) ctx.scene.camera->update_camera_vectors();

     extern bool g_camera_dirty;
     extern bool g_lights_dirty;
     extern bool g_world_dirty;
     extern bool g_geometry_dirty;
     extern bool g_materials_dirty;
     extern bool g_gas_volumes_dirty;
     extern std::atomic<uint64_t> g_scene_geometry_generation;
     extern std::atomic<bool> g_needs_optix_sync;
     g_camera_dirty = true;
     g_lights_dirty = true;
     g_world_dirty = true;
     g_geometry_dirty = true;
     g_materials_dirty = true;
     g_gas_volumes_dirty = true;
     g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
     g_needs_optix_sync.store(true);

     // Route New Project through the canonical backend switch path for Vulkan RT,
     // matching the project-open path and forcing a complete render-backend sync.
     if (sceneUiRenderBackendIsVulkan(ctx)) {
         ctx.render_settings.backend_changed = true;
     }

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
        resetMeshEditState(ctx);
        ctx.scene.base_mesh_cache.clear();
        ctx.scene.mesh_modifiers.clear();

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
        invalidateCache();
        paint_mode_state = Paint::PaintModeState();
        
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
                                        viewport_settings.show_gizmos = vs.value("show_gizmos", true);
                                        viewport_settings.show_selection_outline = vs.value("show_selection_outline", true);
                                        viewport_settings.show_camera_hud = vs.value("show_camera_hud", true);
                                        viewport_settings.show_focus_ring = vs.value("show_focus_ring", true);
                                        viewport_settings.show_zoom_ring = vs.value("show_zoom_ring", true);
                                        viewport_settings.focus_mode = vs.value("focus_mode", 1); // Reset to AF-S if missing
                                    }

                                    // Opened projects now start in Solid mode so the lightweight
                                    // raster viewport is available immediately after load.
                                    viewport_settings.shading_mode = g_hasVulkan ? 0 : 2;

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

                scene_loading_progress = 92;
                setSceneLoadingStage("Waiting for backend...");
                if (ctx.backend_ptr) {
                    ctx.backend_ptr->waitForCompletion();
                }

                scene_loading_progress = 93;
                setSceneLoadingStage("Registering animations...");
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
                    scene_loading_progress = 95;
                    setSceneLoadingStage("Evaluating terrain graph...");
                    auto t_start = std::chrono::steady_clock::now();
                    terrainNodeGraph.evaluateTerrain(terrain, ctx.scene);
                    auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t_start).count();
                    SCENE_LOG_INFO("[Load] Terrain graph evaluated in " + std::to_string(t_ms) + " ms");
                }

                scene_loading_progress = 98;
                setSceneLoadingStage("Syncing volumes...");
                hairUI.clear();
                SceneUI::syncVDBVolumesToGPU(ctx);

                scene_loading_progress = 100;
                setSceneLoadingStage("Finalizing...");
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
        if (ImGui::SliderFloat("Edge Cutoff", &shader->density.cutoff_threshold, 0.0f, 0.5f)) changed = true;
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
            // Preset value rationale (post Aşama-2/3 + accessor cache + Aşama A shadow_steps cut):
            //   - OptiX hardcoded fallback uses step=0.1, max=100, shadow=4.
            //   - Old presets were 5-10× OptiX's values — preview burning GPU cycles for
            //     imperceptible quality past shadow=4 / max=192 on diffuse media.
            //   - New presets give clear tiers: Fast = fluid editing, Balanced ≈ OptiX
            //     parity, Exact = converged stills / blackbody fire detail.
            if (qualityPreset == VolQualityFast) {
                shader->quality.step_size = 0.15f;
                shader->quality.max_steps = 128;
                shader->quality.shadow_steps = 2;
                shader->quality.quality_preset = VolQualityFast;
                changed = true;
            } else if (qualityPreset == VolQualityBalanced) {
                shader->quality.step_size = 0.08f;
                shader->quality.max_steps = 192;
                shader->quality.shadow_steps = 4;
                shader->quality.quality_preset = VolQualityBalanced;
                changed = true;
            } else if (qualityPreset == VolQualityExact) {
                shader->quality.step_size = 0.04f;
                shader->quality.max_steps = 512;
                shader->quality.shadow_steps = 8;
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

        // ESC = Exit Paint Mode
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            hairUI.setPaintMode(Hair::HairPaintMode::NONE);
            ctx.renderer.resetCPUAccumulation();
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
    j["show_geometry_graph"] = show_geometry_graph;
    j["show_material_graph"] = show_material_graph;
    j["show_anim_graph"] = show_anim_graph;
    j["show_volumetric_tab"] = show_volumetric_tab;
    j["show_forcefield_tab"] = show_forcefield_tab;
    j["show_world_tab"] = show_world_tab;
    j["show_stylize_tab"] = show_stylize_tab;
    j["show_hair_tab"] = show_hair_tab;
    j["show_modifiers_tab"] = show_modifiers_tab;
    j["show_scatter_tab"] = show_scatter_tab;
    j["show_paint_tab"] = show_paint_tab;
    j["vdb_import_orientation_preset"] = vdb_import_orientation_preset;
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
    j["show_animation_panel"] = show_animation_panel;
    j["show_terrain_graph"] = show_terrain_graph;
    j["show_geometry_graph"] = show_geometry_graph;
    j["show_material_graph"] = show_material_graph;
    j["show_anim_graph"] = show_anim_graph;
    j["show_asset_browser"] = show_asset_browser;

    {
        nlohmann::json floating_panels = nlohmann::json::object();
        for (const char* name : {"Timeline", "Console", "Terrain Graph", "Geometry Graph", "AnimGraph", "Asset Browser"}) {
            ImGuiWindow* win = ImGui::FindWindowByName(name);
            bool is_floating = false; // Default to docked
            if (win) {
                if (win->DockNode) {
                    ImGuiDockNode* node = win->DockNode;
                    while (node->ParentNode && node->ParentNode->ID != dockspace_id) {
                        node = node->ParentNode;
                    }
                    if (node->ID != dock_bottom_id) {
                        is_floating = true;
                    }
                } else {
                    is_floating = true;
                }
            } else {
                // If the window is currently null (closed/hidden or not yet submitted), check its settings
                ImGuiWindowSettings* settings = ImGui::FindWindowSettingsByID(ImHashStr(name));
                if (settings) {
                    is_floating = (settings->DockId == 0);
                } else {
                    is_floating = false; // Default to docked
                }
            }
            floating_panels[name] = is_floating;
        }
        j["bottom_panels_floating"] = floating_panels;
    }

    // Persist terrain subsection open states
    j["terrain_layer_open"] = nlohmann::json::array();
    for (int i = 0; i < 4; ++i) j["terrain_layer_open"].push_back(terrain_layer_open[i]);
    j["foliage_section_open"] = foliage_section_open;

    // Save panel dimensions
    j["side_panel_width"] = side_panel_width;
    j["bottom_panel_height"] = bottom_panel_height;
    j["preferred_bottom_panel_height"] = preferred_bottom_panel_height;
    j["hierarchy_panel_height"] = hierarchy_panel_height;
    j["docking_enabled"] = docking_enabled;
    // Which Properties sub-tabs are currently torn off into their own windows.
    {
        nlohmann::json popped = nlohmann::json::array();
        for (int t = 0; t <= 13; ++t)
            if (properties_tab_popped_[t]) popped.push_back(t);
        j["properties_popped_tabs"] = popped;
    }

    if (mesh_edit_layer.active && !mesh_edit_layer.object_name.empty()) {
        nlohmann::json layer;
        layer["enabled"] = mesh_edit_layer.enabled;
        layer["object_name"] = mesh_edit_layer.object_name;
        layer["base_positions"] = nlohmann::json::array();
        layer["edited_positions"] = nlohmann::json::array();
        for (const auto& state : mesh_edit_layer.base_states) {
            layer["base_positions"].push_back({
                { state.positions[0].x, state.positions[0].y, state.positions[0].z },
                { state.positions[1].x, state.positions[1].y, state.positions[1].z },
                { state.positions[2].x, state.positions[2].y, state.positions[2].z }
            });
        }
        for (const auto& state : mesh_edit_layer.edited_states) {
            layer["edited_positions"].push_back({
                { state.positions[0].x, state.positions[0].y, state.positions[0].z },
                { state.positions[1].x, state.positions[1].y, state.positions[1].z },
                { state.positions[2].x, state.positions[2].y, state.positions[2].z }
            });
        }
        j["mesh_edit_layer"] = layer;
    }

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
        if (j.contains("show_geometry_graph")) show_geometry_graph = j["show_geometry_graph"];
        if (j.contains("show_material_graph")) show_material_graph = j["show_material_graph"];
        if (j.contains("show_anim_graph")) show_anim_graph = j["show_anim_graph"];
        if (j.contains("show_volumetric_tab")) show_volumetric_tab = j["show_volumetric_tab"];
        if (j.contains("show_forcefield_tab")) show_forcefield_tab = j["show_forcefield_tab"];
        if (j.contains("show_world_tab")) show_world_tab = j["show_world_tab"];
        if (j.contains("show_stylize_tab")) show_stylize_tab = j["show_stylize_tab"];
        if (j.contains("show_hair_tab")) show_hair_tab = j["show_hair_tab"];
        if (j.contains("show_modifiers_tab")) show_modifiers_tab = j["show_modifiers_tab"];
        if (j.contains("show_scatter_tab")) show_scatter_tab = j["show_scatter_tab"];
        if (j.contains("show_paint_tab")) show_paint_tab = j["show_paint_tab"];
        if (j.contains("vdb_import_orientation_preset")) vdb_import_orientation_preset = j["vdb_import_orientation_preset"];
        
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
        asset_library_refresh_in_progress = false;
        asset_library_refresh_status.clear();
        pending_asset_library_root.clear();

        if (!asset_library_paths.empty()) {
            active_asset_library_index = (std::max)(0, (std::min)(active_asset_library_index, static_cast<int>(asset_library_paths.size()) - 1));

            // Asset library scan is expensive (filesystem walk + optional metadata/VDB inspection).
            // Do not block project/UI restore for a panel that may stay closed; defer loading until
            // the asset browser is actually opened, where we already use the async refresh path.
            if (show_asset_browser) {
                startAsyncAssetLibraryRefresh(
                    normalizeAbsolutePath(asset_library_paths[active_asset_library_index]),
                    "Restoring asset library...");
            } else {
                asset_registry = AssetRegistry();
            }
        } else {
            asset_registry = AssetRegistry();
        }
        
        if (j.contains("show_scene_log")) show_scene_log = j["show_scene_log"];
        if (j.contains("show_animation_panel")) show_animation_panel = j["show_animation_panel"];
        if (j.contains("show_terrain_graph")) show_terrain_graph = j["show_terrain_graph"];
        if (j.contains("show_geometry_graph")) show_geometry_graph = j["show_geometry_graph"];
        if (j.contains("show_material_graph")) show_material_graph = j["show_material_graph"];
        if (j.contains("show_anim_graph")) show_anim_graph = j["show_anim_graph"];
        if (j.contains("show_asset_browser")) show_asset_browser = j["show_asset_browser"];

        if (j.contains("bottom_panels_floating") && j["bottom_panels_floating"].is_object()) {
            auto floating_panels = j["bottom_panels_floating"];
            for (const char* name : {"Timeline", "Console", "Terrain Graph", "Geometry Graph", "AnimGraph", "Asset Browser"}) {
                if (floating_panels.contains(name)) {
                    bool is_floating = floating_panels[name].get<bool>();
                    if (is_floating) {
                        ImGui::DockBuilderDockWindow(name, 0);
                    } else {
                        dockToBottom(name);
                    }
                }
            }
        }

        if (j.contains("side_panel_width")) side_panel_width = j["side_panel_width"];
        if (j.contains("bottom_panel_height")) bottom_panel_height = j["bottom_panel_height"];
        preferred_bottom_panel_height = bottom_panel_height;
        if (j.contains("preferred_bottom_panel_height")) preferred_bottom_panel_height = j["preferred_bottom_panel_height"];
        if (j.contains("hierarchy_panel_height")) hierarchy_panel_height = j["hierarchy_panel_height"];
        if (j.contains("docking_enabled")) {
            bool old_val = docking_enabled;
            docking_enabled = j["docking_enabled"];
            if (docking_enabled != old_val) {
                docking_layout_dirty = true; // rebuild default layout to match restored preference
            }
        }
        // Restore torn-off Properties sub-tabs (positions come from imgui.ini, so no spawn override).
        for (int t = 0; t < 16; ++t) { properties_tab_popped_[t] = false; properties_pop_spawn_pending_[t] = false; }
        if (j.contains("properties_popped_tabs") && j["properties_popped_tabs"].is_array()) {
            for (const auto& t : j["properties_popped_tabs"]) {
                int idx = t.get<int>();
                if (isPoppablePropertyTab(idx)) properties_tab_popped_[idx] = true;
            }
        }

        pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
        if (j.contains("mesh_edit_layer") && j["mesh_edit_layer"].is_object()) {
            const auto& layer = j["mesh_edit_layer"];
            pending_serialized_mesh_edit_layer.has_data = true;
            pending_serialized_mesh_edit_layer.enabled = layer.value("enabled", true);
            pending_serialized_mesh_edit_layer.object_name = layer.value("object_name", std::string{});

            auto readPositionList = [](const nlohmann::json& src, std::vector<std::array<Vec3, 3>>& dst) {
                dst.clear();
                if (!src.is_array()) return;
                for (const auto& tri : src) {
                    if (!tri.is_array() || tri.size() != 3) continue;
                    std::array<Vec3, 3> points{};
                    for (size_t i = 0; i < 3; ++i) {
                        if (tri[i].is_array() && tri[i].size() == 3) {
                            points[i] = Vec3(
                                tri[i][0].get<float>(),
                                tri[i][1].get<float>(),
                                tri[i][2].get<float>());
                        }
                    }
                    dst.push_back(points);
                }
            };

            readPositionList(layer.value("base_positions", nlohmann::json::array()),
                             pending_serialized_mesh_edit_layer.base_positions);
            readPositionList(layer.value("edited_positions", nlohmann::json::array()),
                             pending_serialized_mesh_edit_layer.edited_positions);
            if (pending_serialized_mesh_edit_layer.object_name.empty() ||
                pending_serialized_mesh_edit_layer.base_positions.empty() ||
                pending_serialized_mesh_edit_layer.base_positions.size() != pending_serialized_mesh_edit_layer.edited_positions.size()) {
                pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
            }
        }

        // Restore terrain subsection open states
        if (j.contains("terrain_layer_open") && j["terrain_layer_open"].is_array()) {
            for (int i = 0; i < 4 && i < (int)j["terrain_layer_open"].size(); ++i) {
                terrain_layer_open[i] = j["terrain_layer_open"][i];
            }
        }
        if (j.contains("foliage_section_open")) foliage_section_open = j["foliage_section_open"];

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
