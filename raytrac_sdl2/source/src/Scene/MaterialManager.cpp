#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "material_gpu.h"
#include "globals.h"
#include "ProjectManager.h"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <deque>
#include <thread>
#include <algorithm>
#include <unordered_map>

using json = nlohmann::json;

namespace {
struct DeserializeTextureStats {
    size_t embedded_hits = 0;
    size_t disk_hits = 0;
    size_t cache_hits = 0;
    size_t embedded_construct_ms = 0;
    size_t disk_construct_ms = 0;
};

static std::string makeTextureCacheKey(const std::string& texPath, TextureType texType) {
    return texPath + "#" + std::to_string(static_cast<int>(texType));
}

struct TextureLoadRequest {
    std::string cache_key;
    std::string texture_path;
    std::string full_path;
    TextureType type = TextureType::Albedo;
    bool is_embedded = false;
    size_t estimated_cost = 0;
};

struct TextureLoadResult {
    TextureLoadRequest request;
    std::shared_ptr<Texture> texture;
    size_t elapsed_ms = 0;
};

static void collectTextureRequest(const json& propertyJson,
                                  const std::string& sceneDir,
                                  TextureType texType,
                                  std::unordered_map<std::string, TextureLoadRequest>& requests) {
    if (!propertyJson.is_object() || !propertyJson.contains("texture")) {
        return;
    }

    const std::string texPath = propertyJson["texture"].get<std::string>();
    if (texPath.empty()) {
        return;
    }

    const std::string cacheKey = makeTextureCacheKey(texPath, texType);
    if (requests.find(cacheKey) != requests.end()) {
        return;
    }

    TextureLoadRequest req;
    req.cache_key = cacheKey;
    req.texture_path = texPath;
    req.type = texType;

    auto& pm = ProjectManager::getInstance();
    if (const auto* embedded = pm.getEmbeddedTexture(texPath)) {
        req.is_embedded = true;
        req.estimated_cost = embedded->data.size();
    }

    std::string fullPath = texPath;
    if (!req.is_embedded && !std::filesystem::exists(fullPath) && !sceneDir.empty()) {
        fullPath = sceneDir + "/" + texPath;
    }
    req.full_path = fullPath;
    if (!req.is_embedded) {
        std::error_code ec;
        const auto file_size = std::filesystem::file_size(req.full_path, ec);
        req.estimated_cost = ec ? 0 : static_cast<size_t>(file_size);
    }

    if (req.is_embedded || std::filesystem::exists(req.full_path)) {
        requests.emplace(cacheKey, std::move(req));
    }
}

static void finalizeTextureFuture(std::future<TextureLoadResult>& future,
                                  std::unordered_map<std::string, std::shared_ptr<Texture>>& textureCache,
                                  DeserializeTextureStats& stats) {
    TextureLoadResult result = future.get();
    if (result.texture) {
        textureCache.emplace(result.request.cache_key, result.texture);
    }

    if (result.request.is_embedded) {
        ++stats.embedded_hits;
        stats.embedded_construct_ms += result.elapsed_ms;
    } else {
        ++stats.disk_hits;
        stats.disk_construct_ms += result.elapsed_ms;
    }
}

static void preloadTexturesParallel(const std::unordered_map<std::string, TextureLoadRequest>& requests,
                                    std::unordered_map<std::string, std::shared_ptr<Texture>>& textureCache,
                                    DeserializeTextureStats& stats) {
    if (requests.empty()) {
        return;
    }

    std::vector<TextureLoadRequest> ordered_requests;
    ordered_requests.reserve(requests.size());
    size_t total_estimated_cost = 0;
    size_t large_texture_count = 0;
    for (const auto& [_, request] : requests) {
        ordered_requests.push_back(request);
        total_estimated_cost += request.estimated_cost;
        if (request.estimated_cost >= (8u * 1024u * 1024u)) {
            ++large_texture_count;
        }
    }
    std::sort(ordered_requests.begin(), ordered_requests.end(),
              [](const TextureLoadRequest& a, const TextureLoadRequest& b) {
                  return a.estimated_cost > b.estimated_cost;
              });

    const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t worker_cap = std::max<size_t>(1, hw_threads / 2);
    size_t max_parallel = 2;
    if (total_estimated_cost >= (64u * 1024u * 1024u) || large_texture_count >= 4) {
        max_parallel = 4;
    } else if (total_estimated_cost >= (24u * 1024u * 1024u) || large_texture_count >= 2 || ordered_requests.size() >= 16) {
        max_parallel = 3;
    }
    max_parallel = std::min(worker_cap, max_parallel);
    max_parallel = std::max<size_t>(1, max_parallel);

    std::deque<std::future<TextureLoadResult>> active_jobs;

    for (const auto& request : ordered_requests) {
        active_jobs.emplace_back(std::async(std::launch::async, [request]() -> TextureLoadResult {
            TextureLoadResult result;
            result.request = request;

            const auto start = std::chrono::steady_clock::now();
            if (request.is_embedded) {
                auto& pm = ProjectManager::getInstance();
                if (const auto* embedded = pm.getEmbeddedTexture(request.texture_path)) {
                    result.texture = std::make_shared<Texture>(embedded->data, request.type, request.texture_path);
                }
            } else if (!request.full_path.empty() && std::filesystem::exists(request.full_path)) {
                result.texture = std::make_shared<Texture>(request.full_path, request.type);
            }
            result.elapsed_ms = static_cast<size_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count());
            return result;
        }));

        if (active_jobs.size() >= max_parallel) {
            finalizeTextureFuture(active_jobs.front(), textureCache, stats);
            active_jobs.pop_front();
        }
    }

    while (!active_jobs.empty()) {
        finalizeTextureFuture(active_jobs.front(), textureCache, stats);
        active_jobs.pop_front();
    }
}
}

// Helper: Vec3 to JSON
static json vec3ToJson(const Vec3& v) {
    return { v.x, v.y, v.z };
}

// Helper: JSON to Vec3 (with defaults)
static Vec3 jsonToVec3(const json& j, const Vec3& defaultVal = Vec3(0,0,0)) {
    if (j.is_array() && j.size() >= 3)
        return Vec3(j[0].get<double>(), j[1].get<double>(), j[2].get<double>());
    return defaultVal;
}

// Helper: Serialize MaterialProperty
static json serializeProperty(const MaterialProperty& prop, const std::string& sceneDir) {
    json j;
    j["color"] = vec3ToJson(prop.color);
    j["intensity"] = prop.intensity;
    j["alpha"] = prop.alpha;
    
    // Texture path (relative to scene dir if possible)
    if (prop.texture && !prop.texture->name.empty()) {
        std::string texPath = prop.texture->name;
        // Sanitize path for JSON: remove non-printable characters that may come from 
        // embedded GLB textures (e.g., binary references like *0, *1 with binary data)
        std::string sanitized;
        sanitized.reserve(texPath.size());
        for (unsigned char c : texPath) {
            // Keep only printable ASCII and valid UTF-8 continuation bytes
            if (c >= 32 && c < 127) {
                sanitized.push_back(c);
            } else if (c >= 0xC0) {
                // Start of multi-byte UTF-8 sequence - keep it
                sanitized.push_back(c);
            } else if (c >= 0x80 && c < 0xC0 && !sanitized.empty()) {
                // UTF-8 continuation byte - only keep if we're in a sequence
                sanitized.push_back(c);
            }
            // Skip control characters and invalid bytes
        }
        j["texture"] = sanitized.empty() ? texPath : sanitized;
    }
    
    return j;
}

// Helper: Deserialize MaterialProperty
// CRITICAL: TextureType must match the property type to ensure correct texture processing
// Normal maps require TextureType::Normal to avoid sRGB conversion and enable proper normal decoding
static void deserializeProperty(MaterialProperty& prop, const json& j, const std::string& sceneDir, 
                                 TextureType texType = TextureType::Albedo,
                                 std::unordered_map<std::string, std::shared_ptr<Texture>>* textureCache = nullptr,
                                 DeserializeTextureStats* stats = nullptr) {
    prop.color = jsonToVec3(j.value("color", json::array({1,1,1})), Vec3(1,1,1));
    prop.intensity = j.value("intensity", 1.0f);
    prop.alpha = j.value("alpha", 1.0f);
    
    // Load texture if specified
    if (j.contains("texture") && !j["texture"].get<std::string>().empty()) {
        std::string texPath = j["texture"].get<std::string>();
        const std::string cacheKey = makeTextureCacheKey(texPath, texType);

        if (textureCache) {
            auto cacheIt = textureCache->find(cacheKey);
            if (cacheIt != textureCache->end()) {
                prop.texture = cacheIt->second;
                if (stats) {
                    ++stats->cache_hits;
                }
                return;
            }
        }
        
        // FIRST: Check embedded texture cache (memory-based, no disk I/O)
        auto& pm = ProjectManager::getInstance();
        auto* embedded = pm.getEmbeddedTexture(texPath);
        if (embedded) {
            // Load directly from memory buffer - fast and no temp files!
            const auto start = std::chrono::steady_clock::now();
            prop.texture = std::make_shared<Texture>(embedded->data, texType, texPath);
            if (stats) {
                ++stats->embedded_hits;
                stats->embedded_construct_ms += static_cast<size_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count());
            }
            if (textureCache && prop.texture) {
                textureCache->emplace(cacheKey, prop.texture);
            }
            // SCENE_LOG_INFO("[MATERIAL] Loaded embedded texture from memory: " + texPath);
            return;
        }
        
        // FALLBACK: Try loading from disk (for external textures)
        std::string fullPath = texPath;
        if (!std::filesystem::exists(fullPath) && !sceneDir.empty()) {
            fullPath = sceneDir + "/" + texPath;
        }
        if (std::filesystem::exists(fullPath)) {
            const auto start = std::chrono::steady_clock::now();
            prop.texture = std::make_shared<Texture>(fullPath, texType);
            if (stats) {
                ++stats->disk_hits;
                stats->disk_construct_ms += static_cast<size_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count());
            }
            if (textureCache && prop.texture) {
                textureCache->emplace(cacheKey, prop.texture);
            }
        }
    }
}

uint16_t MaterialManager::addMaterial(const std::string& name, std::shared_ptr<Material> mat) {
    std::lock_guard<std::mutex> lock(mutex);
    
    // Check if already exists
    auto it = nameToID.find(name);
    if (it != nameToID.end()) {
        return it->second;
    }

    // Check capacity
    if (materials.size() >= MAX_MATERIALS) {
       // std::cerr << "[MaterialManager] Maximum material count reached!" << std::endl;
        return INVALID_MATERIAL_ID;
    }

    uint16_t id = static_cast<uint16_t>(materials.size());
    materials.push_back(mat);
    nameToID[name] = id;

    return id;
}

uint16_t MaterialManager::getMaterialID(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = nameToID.find(name);
    return (it != nameToID.end()) ? it->second : INVALID_MATERIAL_ID;
}

Material* MaterialManager::getMaterial(uint16_t id) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (id >= materials.size() || id == INVALID_MATERIAL_ID) {
        return nullptr;
    }
    return materials[id].get();
}

std::shared_ptr<Material> MaterialManager::getMaterialShared(uint16_t id) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (id >= materials.size() || id == INVALID_MATERIAL_ID) {
        return nullptr;
    }
    return materials[id];
}

uint16_t MaterialManager::getOrCreateMaterialID(const std::string& name, std::shared_ptr<Material> mat) {
    std::lock_guard<std::mutex> lock(mutex);
    
    auto it = nameToID.find(name);
    if (it != nameToID.end()) {
        // Material exists - but we need to UPDATE texture references if the new material has them!
        // This fixes the bug where re-importing a model loses texture assignments
        uint16_t existingId = it->second;
        
        if (mat && mat->type() == MaterialType::PrincipledBSDF && existingId < materials.size()) {
            auto existingMat = materials[existingId];
            if (existingMat && existingMat->type() == MaterialType::PrincipledBSDF) {
                PrincipledBSDF* newPbsdf = dynamic_cast<PrincipledBSDF*>(mat.get());
                PrincipledBSDF* existingPbsdf = dynamic_cast<PrincipledBSDF*>(existingMat.get());
                
                if (newPbsdf && existingPbsdf) {
                    // Update texture references if new material has them and existing doesn't
                    bool updated = false;
                    if (newPbsdf->albedoProperty.texture && !existingPbsdf->albedoProperty.texture) {
                        existingPbsdf->albedoProperty.texture = newPbsdf->albedoProperty.texture;
                        updated = true;
                    }
                    if (newPbsdf->normalProperty.texture && !existingPbsdf->normalProperty.texture) {
                        existingPbsdf->normalProperty.texture = newPbsdf->normalProperty.texture;
                        updated = true;
                    }
                    if (newPbsdf->roughnessProperty.texture && !existingPbsdf->roughnessProperty.texture) {
                        existingPbsdf->roughnessProperty.texture = newPbsdf->roughnessProperty.texture;
                        updated = true;
                    }
                    if (newPbsdf->metallicProperty.texture && !existingPbsdf->metallicProperty.texture) {
                        existingPbsdf->metallicProperty.texture = newPbsdf->metallicProperty.texture;
                        updated = true;
                    }
                    if (newPbsdf->emissionProperty.texture && !existingPbsdf->emissionProperty.texture) {
                        existingPbsdf->emissionProperty.texture = newPbsdf->emissionProperty.texture;
                        updated = true;
                    }
                    if (newPbsdf->opacityProperty.texture && !existingPbsdf->opacityProperty.texture) {
                        existingPbsdf->opacityProperty.texture = newPbsdf->opacityProperty.texture;
                        updated = true;
                    }
                    if (newPbsdf->transmissionProperty.texture && !existingPbsdf->transmissionProperty.texture) {
                        existingPbsdf->transmissionProperty.texture = newPbsdf->transmissionProperty.texture;
                        updated = true;
                    }
                    
                    if (updated) {
                       // SCENE_LOG_INFO("[MaterialManager] Updated texture refs for existing material: " + name);
                    }
                }
            }
        }
        
        return existingId;
    }

    if (materials.size() >= MAX_MATERIALS) {
       // std::cerr << "[MaterialManager] Maximum material count reached!" << std::endl;
        return INVALID_MATERIAL_ID;
    }

    uint16_t id = static_cast<uint16_t>(materials.size());
    materials.push_back(mat);
    nameToID[name] = id;
    
    // DEBUG: Log when creating new material with texture info
    std::string texInfo = "NO TEXTURE";
    if (mat->type() == MaterialType::PrincipledBSDF) {
        PrincipledBSDF* pbsdf = dynamic_cast<PrincipledBSDF*>(mat.get());
        if (pbsdf && pbsdf->albedoProperty.texture) {
            texInfo = "HAS ALBEDO TEX: " + pbsdf->albedoProperty.texture->name;
        }
    }
   // SCENE_LOG_INFO("[MaterialManager] getOrCreateMaterialID NEW: " + name + " -> ID=" + std::to_string(id) + " | " + texInfo);

    return id;
}

bool MaterialManager::hasMaterial(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex);
    return nameToID.find(name) != nameToID.end();
}

void MaterialManager::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    materials.clear();
    nameToID.clear();
}

std::string MaterialManager::getMaterialName(uint16_t id) const {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& [name, mat_id] : nameToID) {
        if (mat_id == id) return name;
    }
    return "";
}

// ============================================================================
// SERIALIZATION
// ============================================================================

json MaterialManager::serialize(const std::string& sceneDir) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    json root;
    root["version"] = MATERIAL_SERIALIZATION_VERSION;
    root["material_count"] = materials.size();
    
    json materialsArray = json::array();
    
    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& mat = materials[i];
        json matJson;
        
        // Find name for this material
        std::string matName = "";
        for (const auto& [name, id] : nameToID) {
            if (id == i) { matName = name; break; }
        }
        
        matJson["id"] = i;
        matJson["name"] = matName;
        matJson["type"] = static_cast<int>(mat->type());
        
        // Common properties
        matJson["albedo"] = vec3ToJson(mat->albedo);
        matJson["ior"] = mat->ior;
        
        // PrincipledBSDF specific
        if (mat->type() == MaterialType::PrincipledBSDF) {
            PrincipledBSDF* pbsdf = dynamic_cast<PrincipledBSDF*>(mat.get());
            if (pbsdf) {
                // Properties with textures
                matJson["albedoProperty"] = serializeProperty(pbsdf->albedoProperty, sceneDir);
                matJson["roughnessProperty"] = serializeProperty(pbsdf->roughnessProperty, sceneDir);
                matJson["metallicProperty"] = serializeProperty(pbsdf->metallicProperty, sceneDir);
                matJson["normalProperty"] = serializeProperty(pbsdf->normalProperty, sceneDir);
                matJson["opacityProperty"] = serializeProperty(pbsdf->opacityProperty, sceneDir);
                matJson["emissionProperty"] = serializeProperty(pbsdf->emissionProperty, sceneDir);
                matJson["transmissionProperty"] = serializeProperty(pbsdf->transmissionProperty, sceneDir);
                
                // Scalar values
                matJson["transmission"] = pbsdf->transmission;
                matJson["clearcoat"] = pbsdf->clearcoat;
                matJson["clearcoatRoughness"] = pbsdf->clearcoatRoughness;
                matJson["anisotropic"] = pbsdf->anisotropic;
                matJson["normalStrength"] = pbsdf->get_normal_strength();
                
                // SSS
                matJson["subsurface"] = pbsdf->subsurface;
                matJson["subsurfaceColor"] = vec3ToJson(pbsdf->subsurfaceColor);
                matJson["subsurfaceRadius"] = vec3ToJson(pbsdf->subsurfaceRadius);
                matJson["subsurfaceScale"] = pbsdf->subsurfaceScale;
                matJson["subsurfaceAnisotropy"] = pbsdf->subsurfaceAnisotropy;
                matJson["subsurfaceIOR"] = pbsdf->subsurfaceIOR;
                
                // Translucent
                matJson["translucent"] = pbsdf->translucent;
                
                // Tiling
                matJson["tilingFactor"] = { pbsdf->tilingFactor.x, pbsdf->tilingFactor.y };
            }
        }
        
        materialsArray.push_back(matJson);
    }
    
    root["materials"] = materialsArray;
    
   // SCENE_LOG_INFO("[MaterialManager] Serialized " + std::to_string(materials.size()) + " materials");
    return root;
}

void MaterialManager::deserialize(const json& data, const std::string& sceneDir) {
    const auto total_start = std::chrono::steady_clock::now();

    // Clear existing materials first
    clear();
    
    std::lock_guard<std::mutex> lock(mutex);
    std::unordered_map<std::string, std::shared_ptr<Texture>> textureCache;
    DeserializeTextureStats textureStats;
    
    // Version check
    int version = data.value("version", 1);
    if (version > MATERIAL_SERIALIZATION_VERSION) {
        SCENE_LOG_WARN("[MaterialManager] Scene uses newer material format (v" + 
                       std::to_string(version) + "), some features may not load correctly");
    }
    
    if (!data.contains("materials") || !data["materials"].is_array()) {
        SCENE_LOG_WARN("[MaterialManager] No materials found in scene data");
        return;
    }

    const auto& materialsJson = data["materials"];
    materials.reserve(materialsJson.size());
    nameToID.reserve(materialsJson.size());

    std::unordered_map<std::string, TextureLoadRequest> preloadRequests;
    preloadRequests.reserve(materialsJson.size() * 4);
    for (const auto& matJson : materialsJson) {
        if (matJson.contains("albedoProperty"))
            collectTextureRequest(matJson["albedoProperty"], sceneDir, TextureType::Albedo, preloadRequests);
        if (matJson.contains("roughnessProperty"))
            collectTextureRequest(matJson["roughnessProperty"], sceneDir, TextureType::Roughness, preloadRequests);
        if (matJson.contains("metallicProperty"))
            collectTextureRequest(matJson["metallicProperty"], sceneDir, TextureType::Metallic, preloadRequests);
        if (matJson.contains("normalProperty"))
            collectTextureRequest(matJson["normalProperty"], sceneDir, TextureType::Normal, preloadRequests);
        if (matJson.contains("opacityProperty"))
            collectTextureRequest(matJson["opacityProperty"], sceneDir, TextureType::Opacity, preloadRequests);
        if (matJson.contains("emissionProperty"))
            collectTextureRequest(matJson["emissionProperty"], sceneDir, TextureType::Emission, preloadRequests);
        if (matJson.contains("transmissionProperty"))
            collectTextureRequest(matJson["transmissionProperty"], sceneDir, TextureType::Transmission, preloadRequests);
    }
    preloadTexturesParallel(preloadRequests, textureCache, textureStats);
    
    for (const auto& matJson : materialsJson) {
        std::string name = matJson.value("name", "Unnamed_" + std::to_string(materials.size()));
        int typeInt = matJson.value("type", 0);
        MaterialType type = static_cast<MaterialType>(typeInt);
        
        std::shared_ptr<Material> mat = nullptr;
        
        if (type == MaterialType::PrincipledBSDF) {
            auto pbsdf = std::make_shared<PrincipledBSDF>();
            
            // Albedo
            pbsdf->albedo = jsonToVec3(matJson.value("albedo", json::array({1,1,1})), Vec3(1,1,1));
            pbsdf->ior = matJson.value("ior", 1.45f);
            
            // Load properties (with backward compatibility - use value() with defaults)
            // Load properties with correct texture types for proper GPU processing
            if (matJson.contains("albedoProperty"))
                deserializeProperty(pbsdf->albedoProperty, matJson["albedoProperty"], sceneDir, TextureType::Albedo, &textureCache, &textureStats);
            if (matJson.contains("roughnessProperty"))
                deserializeProperty(pbsdf->roughnessProperty, matJson["roughnessProperty"], sceneDir, TextureType::Roughness, &textureCache, &textureStats);
            if (matJson.contains("metallicProperty"))
                deserializeProperty(pbsdf->metallicProperty, matJson["metallicProperty"], sceneDir, TextureType::Metallic, &textureCache, &textureStats);
            if (matJson.contains("normalProperty"))
                deserializeProperty(pbsdf->normalProperty, matJson["normalProperty"], sceneDir, TextureType::Normal, &textureCache, &textureStats);
            if (matJson.contains("opacityProperty"))
                deserializeProperty(pbsdf->opacityProperty, matJson["opacityProperty"], sceneDir, TextureType::Opacity, &textureCache, &textureStats);
            if (matJson.contains("emissionProperty"))
                deserializeProperty(pbsdf->emissionProperty, matJson["emissionProperty"], sceneDir, TextureType::Emission, &textureCache, &textureStats);
            if (matJson.contains("transmissionProperty"))
                deserializeProperty(pbsdf->transmissionProperty, matJson["transmissionProperty"], sceneDir, TextureType::Transmission, &textureCache, &textureStats);
            
            // Scalar values (with defaults for backward compatibility)
            pbsdf->transmission = matJson.value("transmission", 0.0f);
            pbsdf->clearcoat = matJson.value("clearcoat", 0.0f);
            pbsdf->clearcoatRoughness = matJson.value("clearcoatRoughness", 0.03f);
            pbsdf->anisotropic = matJson.value("anisotropic", 0.0f);
            pbsdf->normalStrength = matJson.value("normalStrength", 1.0f);
            
            // SSS
            pbsdf->subsurface = matJson.value("subsurface", 0.0f);
            if (matJson.contains("subsurfaceColor"))
                pbsdf->subsurfaceColor = jsonToVec3(matJson["subsurfaceColor"], Vec3(1.0f, 0.8f, 0.6f));
            if (matJson.contains("subsurfaceRadius"))
                pbsdf->subsurfaceRadius = jsonToVec3(matJson["subsurfaceRadius"], Vec3(1.0f, 0.2f, 0.1f));
            pbsdf->subsurfaceScale = matJson.value("subsurfaceScale", 0.05f);
            pbsdf->subsurfaceAnisotropy = matJson.value("subsurfaceAnisotropy", 0.0f);
            pbsdf->subsurfaceIOR = matJson.value("subsurfaceIOR", 1.4f);
            
            // Translucent
            pbsdf->translucent = matJson.value("translucent", 0.0f);
            
            // Tiling
            if (matJson.contains("tilingFactor") && matJson["tilingFactor"].is_array()) {
                pbsdf->tilingFactor = Vec2(matJson["tilingFactor"][0], matJson["tilingFactor"][1]);
            }
            
            mat = pbsdf;
        }
        // TODO: Add other material types (Metal, Dielectric, Volumetric) as needed
        else {
            // Default to PrincipledBSDF for unknown types
            mat = std::make_shared<PrincipledBSDF>();
            mat->albedo = jsonToVec3(matJson.value("albedo", json::array({1,1,1})), Vec3(1,1,1));
        }
        
        if (mat) {
            mat->materialName = name;
            uint16_t id = static_cast<uint16_t>(materials.size());
            materials.push_back(mat);
            nameToID[name] = id;
        }
    }
    
    SCENE_LOG_INFO("[MaterialManager] Deserialized " + std::to_string(materials.size()) + " materials (format v" + std::to_string(version) + ")");
    
    // CRITICAL: Sync gpuMaterial pointers after deserialize
    // This ensures GPU rendering works correctly without manual UI trigger
    // NOTE: Call internal version since we already hold the mutex
    syncAllGpuMaterials_internal();

    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - total_start).count();
    SCENE_LOG_INFO("[Perf] MaterialManager::deserialize textures - cache hits: " + std::to_string(textureStats.cache_hits) +
                   ", embedded loads: " + std::to_string(textureStats.embedded_hits) +
                   " (" + std::to_string(textureStats.embedded_construct_ms) + " ms), disk loads: " +
                   std::to_string(textureStats.disk_hits) + " (" + std::to_string(textureStats.disk_construct_ms) +
                   " ms), unique textures: " + std::to_string(textureCache.size()) +
                   ", total: " + std::to_string(total_ms) + " ms");
}

// Internal version - does NOT acquire lock (caller must hold mutex)
void MaterialManager::syncAllGpuMaterials_internal() {
    int synced_count = 0;
    
    for (auto& mat : materials) {
        if (!mat) continue;
        
        if (mat->type() == MaterialType::PrincipledBSDF) {
            PrincipledBSDF* pbsdf = dynamic_cast<PrincipledBSDF*>(mat.get());
            if (!pbsdf) continue;
            
            // Create or update gpuMaterial
            if (!mat->gpuMaterial) {
                mat->gpuMaterial = std::make_shared<GpuMaterial>();
            }
            
            // Sync all properties from PrincipledBSDF to GpuMaterial
            Vec3 alb = pbsdf->albedoProperty.color;
            mat->gpuMaterial->albedo = make_float3((float)alb.x, (float)alb.y, (float)alb.z);
            mat->gpuMaterial->roughness = (float)pbsdf->roughnessProperty.color.x;
            mat->gpuMaterial->metallic = (float)pbsdf->metallicProperty.intensity;
            
            Vec3 em = pbsdf->emissionProperty.color;
            float emStr = pbsdf->emissionProperty.intensity;
            mat->gpuMaterial->emission = make_float3((float)em.x * emStr, (float)em.y * emStr, (float)em.z * emStr);
            
            mat->gpuMaterial->ior = pbsdf->ior;
            mat->gpuMaterial->transmission = pbsdf->transmission;
            // IMPORTANT: Opacity property alpha is what we use for transparency
            mat->gpuMaterial->opacity = pbsdf->opacityProperty.alpha;
            
            // SSS
            mat->gpuMaterial->subsurface = pbsdf->subsurface;
            Vec3 sssColor = pbsdf->subsurfaceColor;
            mat->gpuMaterial->subsurface_color = make_float3((float)sssColor.x, (float)sssColor.y, (float)sssColor.z);
            Vec3 sssRadius = pbsdf->subsurfaceRadius;
            mat->gpuMaterial->subsurface_radius = make_float3((float)sssRadius.x, (float)sssRadius.y, (float)sssRadius.z);
            mat->gpuMaterial->subsurface_scale = pbsdf->subsurfaceScale;
            mat->gpuMaterial->subsurface_anisotropy = pbsdf->subsurfaceAnisotropy;
            mat->gpuMaterial->subsurface_ior = pbsdf->subsurfaceIOR;
            mat->gpuMaterial->sss_use_random_walk = pbsdf->useRandomWalkSSS ? 1 : 0;
            mat->gpuMaterial->sss_max_steps = pbsdf->sssMaxSteps;
            
            // Clear Coat
            mat->gpuMaterial->clearcoat = pbsdf->clearcoat;
            mat->gpuMaterial->clearcoat_roughness = pbsdf->clearcoatRoughness;
            
            // Translucent & Anisotropic
            mat->gpuMaterial->translucent = pbsdf->translucent;
            mat->gpuMaterial->anisotropic = pbsdf->anisotropic;
            
            synced_count++;
        }
    }
    
    SCENE_LOG_INFO("[MaterialManager] Synced " + std::to_string(synced_count) + " GPU materials");
}

// Public version - acquires lock for thread safety
void MaterialManager::syncAllGpuMaterials() {
    std::lock_guard<std::mutex> lock(mutex);
    syncAllGpuMaterials_internal();
}
