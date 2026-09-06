#include "FoliageAssetLibrary.h"

#include "Import/ModelImport.h"
#include "Triangle.h"   // CachedGeometry stores facades; ModelImport.h only forward-declares
#include "MaterialManager.h"

#include <algorithm>
#include <cstdint>
#include <cctype>
#include <system_error>
#include <unordered_map>

namespace FoliageAssets {
namespace {

struct CachedGeometry {
    std::filesystem::file_time_type timestamp{};
    // The material registry generation these triangles were loaded under. A
    // Triangle stores a material ID, not a material; MaterialManager::clear()
    // (New Project, and every project open via deserialize) retires every ID
    // while this cache survives, so a later cache hit would hand out geometry
    // pointing at whatever now occupies those slots -- terrain layers, usually.
    // Cheap to reload, impossible to notice when wrong.
    uint64_t material_generation = 0;
    std::vector<std::shared_ptr<Triangle>> triangles;
};

AssetRegistry g_catalog;
std::filesystem::path g_root;
bool g_catalogReady = false;
std::unordered_map<std::string, CachedGeometry> g_geometryCache;

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool containsToken(const AssetRecord& asset, const std::vector<std::string>& tokens) {
    std::string haystack = lowerCopy(asset.name + " " + asset.id + " " +
        asset.category + " " + asset.subcategory + " " +
        asset.relative_entry_path.generic_string());
    for (const auto& tag : asset.tags) haystack += " " + lowerCopy(tag);
    for (const auto& token : tokens) {
        if (!token.empty() && haystack.find(token) != std::string::npos) return true;
    }
    return false;
}

// One import namespace PER ASSET.
//
// AssimpLoader derives two global keys from the import name: the material
// registry key (currentImportName + "_" + material name) and the embedded
// texture key ("embedded_" + currentImportName + "_" + <texture> + "_" + type).
// The second one is consulted against the process-wide TextureCache, whose hit
// path SKIPS THE DECODE and adopts the cached entry. So a shared import name
// makes two different assets' embedded textures collide, and the second tree
// silently wears the first tree's pixels.
//
// The whole library used to import under the single literal "foliage_asset",
// which disarmed exactly the guard AssimpLoader documents as critical. Keyed by
// the relative entry path: unique per asset and stable across sessions, so
// material names survive a project round-trip.
std::string importPrefixForAsset(const std::string& relativeEntryPath) {
    std::string prefix = "foliage_";
    prefix.reserve(prefix.size() + relativeEntryPath.size());
    for (const char raw : relativeEntryPath) {
        const unsigned char c = static_cast<unsigned char>(raw);
        prefix += std::isalnum(c) ? static_cast<char>(std::tolower(c)) : '_';
    }
    return prefix;
}

std::vector<std::string> typeTokens(const std::string& layerType) {
    const std::string type = lowerCopy(layerType);
    if (type.find("forest") != std::string::npos) return {"tree", "trees", "forest", "bush", "shrub",
                                                          "pine", "fir", "spruce", "conifer", "larch"};
    if (type.find("grass") != std::string::npos) return {"grass", "meadow", "flower", "plant"};
    if (type.find("rock") != std::string::npos) return {"rock", "rocks", "stone", "cliff"};
    // Alpine must admit mountain conifers: with no tree-ish token here the
    // hard type filter below made trees impossible to place on alpine layers.
    if (type.find("alpine") != std::string::npos) return {"alpine", "mountain", "grass", "shrub", "rock",
                                                          "pine", "fir", "spruce", "conifer", "larch", "tree"};
    return {};
}

std::vector<std::string> biomeTokens(const std::string& biome) {
    const std::string value = lowerCopy(biome);
    if (value.empty() || value == "auto") return {};
    if (value.find("temperate") != std::string::npos) return {"temperate", "mixed"};
    if (value.find("lush") != std::string::npos) return {"lush", "wet", "tropical", "valley"};
    if (value.find("alpine") != std::string::npos) return {"alpine", "tundra", "mountain"};
    if (value.find("arid") != std::string::npos) return {"arid", "dry", "desert", "highland"};
    if (value.find("boreal") != std::string::npos) return {"boreal", "pine", "fir", "conifer", "mountain"};
    return {value};
}

} // namespace

const AssetRegistry& catalog(bool forceRefresh) {
    if (!g_catalogReady || forceRefresh) {
        g_root = AssetRegistry::resolveDefaultRoot();
        g_catalog.refresh(g_root);
        g_catalogReady = true;
    }
    return g_catalog;
}

const std::filesystem::path& libraryRoot() {
    catalog(false);
    return g_root;
}

std::vector<const AssetRecord*> recommendedAssets(
    const std::string& layerType,
    const std::string& biome,
    const std::string& searchText) {
    const auto& registry = catalog(false);
    const auto types = typeTokens(layerType);
    const auto climates = biomeTokens(biome);
    const std::string search = lowerCopy(searchText);
    std::vector<std::pair<int, const AssetRecord*>> ranked;

    for (const auto& asset : registry.getAssets()) {
        if (asset.asset_kind != "model") continue;
        const std::string category = lowerCopy(asset.category);
        if (category != "vegetation" && category != "rocks") continue;

        const std::string searchable = lowerCopy(asset.name + " " + asset.id + " " +
            asset.relative_entry_path.generic_string());
        if (!search.empty() && searchable.find(search) == std::string::npos &&
            !containsToken(asset, {search})) continue;

        int score = 1;
        if (!types.empty()) {
            if (!containsToken(asset, types)) continue;
            score += 20;
        }
        if (!climates.empty() && containsToken(asset, climates)) score += 10;
        if (asset.has_preview) score += 2;
        if (asset.favorite) score += 3;
        ranked.emplace_back(score, &asset);
    }

    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second->name < b.second->name;
    });
    std::vector<const AssetRecord*> result;
    result.reserve(ranked.size());
    for (const auto& item : ranked) result.push_back(item.second);
    return result;
}

bool loadScatterSource(const std::string& relativeEntryPath,
                       const std::string& displayName,
                       float weight,
                       ScatterSource& outSource,
                       std::string* errorMessage) {
    if (relativeEntryPath.empty()) {
        if (errorMessage) *errorMessage = "Asset path is empty";
        return false;
    }
    const std::filesystem::path fullPath = libraryRoot() / std::filesystem::path(relativeEntryPath);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(fullPath, ec)) {
        if (errorMessage) *errorMessage = "Missing foliage asset: " + fullPath.string();
        return false;
    }

    const std::string key = fullPath.lexically_normal().generic_string();
    // ★ A `fbx_reader` field used to take part in this cache key: while both
    // Assimp and ufbx could read FBX, switching readers had to invalidate the
    // cached geometry. There is one reader per format now, so the key would be
    // constant — removed rather than left as a field that can never differ
    // (CLAUDE.md rule 5).
    const auto timestamp = std::filesystem::last_write_time(fullPath, ec);
    const uint64_t materialGeneration = MaterialManager::getInstance().generation();
    auto cached = g_geometryCache.find(key);
    if (cached == g_geometryCache.end() ||
        (!ec && cached->second.timestamp != timestamp) ||
        cached->second.material_generation != materialGeneration) {
        // ★★★ THE WHOLE VEGETATION LIBRARY IS .glb, AND IT USED TO GO THROUGH
        // ASSIMP. Faz 2 removed the Assimp fallback in Renderer::create_scene and
        // the batch was written up as "glTF never reaches Assimp" — but this call
        // site was never checked, so every planted tree kept the OLD convention:
        // AssimpLoader sets no aiProcess_FlipUVs, so raw glTF UVs (v down) met a
        // sampler that already flips — i.e. every scattered tree sampled its atlas
        // upside down while the SAME file dragged in as a model came out correct.
        rtimport::ImportOptions opts;
        opts.importPrefix   = importPrefixForAsset(relativeEntryPath);
        opts.loadGeometry   = true;
        opts.loadAnimations = false;   // foliage is static; these were discarded anyway
        opts.loadSkinning   = false;
        opts.loadCameras    = false;
        opts.loadLights     = false;
        // ★★ PER-FACE FACADES ARE REQUIRED HERE, not representative ones.
        // A library prototype is NOT a scene world object, so it has no entry in
        // the backend mesh registry — and that registry is where
        // ScatterSource::flat_meshes gets its BLAS from (VulkanBackend.cpp: no
        // registry hit -> `continue` -> an invisible forest, which is exactly the
        // EXT_mesh_gpu_instancing failure this batch already paid for once).
        // Library foliage therefore uses the facade shape, and InstanceManager
        // bakes its centred copies by walking source.triangles ONE AT A TIME.
        opts.emitSingleFacadePerMesh = false;

        rtimport::ImportedModel imported;
        std::string importError;
        if (!rtimport::loadModel(fullPath.string(), opts, imported, importError)) {
            if (errorMessage) *errorMessage = "Failed to load foliage asset: " + importError;
            return false;
        }
        if (imported.objects.empty()) {
            if (errorMessage) *errorMessage = "No renderable triangles in foliage asset: " + fullPath.string();
            return false;
        }
        CachedGeometry geometry;
        geometry.timestamp = timestamp;
        geometry.material_generation = materialGeneration;
        geometry.triangles = std::move(imported.objects);
        cached = g_geometryCache.insert_or_assign(key, std::move(geometry)).first;
    }

    outSource = ScatterSource(displayName.empty() ? fullPath.stem().string() : displayName,
                              cached->second.triangles);
    // Asset-library foliage is planted from its base, not its centroid. Keep
    // legacy scene-triangle sources unchanged while giving imported trees and
    // plants a predictable terrain contact point.
    if (outSource.has_local_bbox) outSource.mesh_center.y = outSource.local_bbox.min.y;
    outSource.weight = (std::max)(0.0f, weight);
    outSource.asset_relative_path = std::filesystem::path(relativeEntryPath).generic_string();
    outSource.asset_id = fullPath.stem().string();
    return true;
}

float defaultTargetHeight(const std::string& layerType) {
    const std::string type = lowerCopy(layerType);
    if (type.find("forest") != std::string::npos || type.find("tree") != std::string::npos) return 8.0f;
    if (type.find("grass") != std::string::npos || type.find("meadow") != std::string::npos) return 0.35f;
    if (type.find("rock") != std::string::npos || type.find("stone") != std::string::npos) return 1.5f;
    if (type.find("alpine") != std::string::npos) return 0.6f;
    return 1.0f;
}

void configurePlacement(ScatterSource& source,
                        float targetHeight,
                        float heightVariation,
                        bool alignToNormal,
                        float normalInfluence,
                        float yOffsetMin,
                        float yOffsetMax) {
    const float variation = (std::max)(0.0f, (std::min)(heightVariation, 0.95f));
    if (targetHeight > 0.0f && source.has_local_bbox) {
        const float sourceHeight = source.local_bbox.max.y - source.local_bbox.min.y;
        if (sourceHeight > 1e-5f) {
            const float baseScale = targetHeight / sourceHeight;
            source.settings.scale_min = (std::max)(0.0001f, baseScale * (1.0f - variation));
            source.settings.scale_max = (std::max)(source.settings.scale_min,
                baseScale * (1.0f + variation));
        }
    }
    source.settings.align_to_normal = alignToNormal;
    source.settings.normal_influence = alignToNormal
        ? (std::max)(0.0f, (std::min)(normalInfluence, 1.0f)) : 0.0f;
    source.settings.rotation_random_xz = 0.0f;
    source.settings.y_offset_min = (std::min)(yOffsetMin, yOffsetMax);
    source.settings.y_offset_max = (std::max)(yOffsetMin, yOffsetMax);
}

} // namespace FoliageAssets
