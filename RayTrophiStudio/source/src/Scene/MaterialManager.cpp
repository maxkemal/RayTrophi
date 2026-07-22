#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Volumetric.h"
#include "PBRMaterialSnapshot.h"
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

    // Full hardware width: stb decode is pure CPU work and this pool runs only during
    // project load (nothing else competes). The old cap of 8 left half a 16-thread
    // machine idle and made texture decode the single largest block of openProject
    // (931 ms of the 1336 ms bedroom load — 3.8 s of decode squeezed through 8 threads).
    const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t max_parallel = std::max<size_t>(1, static_cast<size_t>(hw_threads));

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

static json vec2ToJson(const Vec2& v) {
    return { v.u, v.v };
}

static Vec2 jsonToVec2(const json& j, const Vec2& defaultVal = Vec2(0, 0)) {
    if (j.is_array() && j.size() >= 2) {
        return Vec2(j[0].get<double>(), j[1].get<double>());
    }
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
            prop.texture = std::make_shared<Texture>(embedded->data, embedded->type, texPath);
            if (stats) {
                ++stats->embedded_hits;
                stats->embedded_construct_ms += static_cast<size_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count());
            }
            if (textureCache && prop.texture) {
                textureCache->emplace(cacheKey, prop.texture);
            }
            return;
        }

        // FALLBACK: Try loading from disk (for external textures)
        std::string fullPath = texPath;
        // Check if there's a remap for this name (e.g. "embedded_0" -> project-local copy path)
        if (const std::string* remapped = pm.getTexturePathRemap(texPath)) {
            fullPath = *remapped;
        } else if (!std::filesystem::exists(fullPath) && !sceneDir.empty()) {
            fullPath = sceneDir + "/" + texPath;
        }
        if (std::filesystem::exists(fullPath)) {
            const auto start = std::chrono::steady_clock::now();
            if (std::filesystem::path(fullPath).extension() == ".bin") {
                std::ifstream in(fullPath, std::ios::binary);
                if (in.is_open()) {
                    std::vector<char> buffer((std::istreambuf_iterator<char>(in)),
                                             std::istreambuf_iterator<char>());
                    if (!buffer.empty()) {
                        prop.texture = std::make_shared<Texture>(buffer, texType, fullPath);
                    }
                }
            }
            if (!prop.texture || !prop.texture->is_loaded()) {
                prop.texture = std::make_shared<Texture>(fullPath, texType);
            }
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
                    if (newPbsdf->heightProperty.texture && !existingPbsdf->heightProperty.texture) {
                        existingPbsdf->heightProperty.texture = newPbsdf->heightProperty.texture;
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
                matJson["specularProperty"] = serializeProperty(pbsdf->specularProperty, sceneDir);
                matJson["normalProperty"] = serializeProperty(pbsdf->normalProperty, sceneDir);
                matJson["heightProperty"] = serializeProperty(pbsdf->heightProperty, sceneDir);
                matJson["opacityProperty"] = serializeProperty(pbsdf->opacityProperty, sceneDir);
                matJson["emissionProperty"] = serializeProperty(pbsdf->emissionProperty, sceneDir);
                matJson["transmissionProperty"] = serializeProperty(pbsdf->transmissionProperty, sceneDir);
                
                // Scalar values
                matJson["transmission"] = pbsdf->transmission;
                matJson["dispersion"] = pbsdf->dispersion;
                matJson["metallicTexChannel"] = pbsdf->metallic_tex_channel;
                matJson["roughnessTexChannel"] = pbsdf->roughness_tex_channel;
                matJson["clearcoat"] = pbsdf->getClearcoat();
                matJson["clearcoatRoughness"] = pbsdf->getClearcoatRoughness();
                matJson["anisotropic"] = pbsdf->anisotropic;
                matJson["normalStrength"] = pbsdf->get_normal_strength();
                
                // SSS
                matJson["subsurface"] = pbsdf->getSubsurface();
                matJson["subsurfaceColor"] = vec3ToJson(pbsdf->getSubsurfaceColor());
                matJson["subsurfaceRadius"] = vec3ToJson(pbsdf->getSubsurfaceRadius());
                matJson["subsurfaceScale"] = pbsdf->getSubsurfaceScale();
                matJson["subsurfaceAnisotropy"] = pbsdf->getSubsurfaceAnisotropy();
                matJson["subsurfaceIOR"] = pbsdf->getSubsurfaceIOR();
                
                // Translucent
                matJson["translucent"] = pbsdf->translucent;

                // Procedural surface detail
                matJson["microDetailStrength"] = pbsdf->micro_detail_strength;
                matJson["microDetailScale"]    = pbsdf->micro_detail_scale;
                matJson["tileBreakStrength"]   = pbsdf->tile_break_strength;

                // Thin-shell bubble (champagne / soda / soap close-up)
                matJson["isBubble"]   = pbsdf->getIsBubble();
                matJson["bubbleIor"]  = pbsdf->getBubbleIor();
                matJson["bubbleFilm"] = pbsdf->getBubbleFilm();
                matJson["clearcoatIridescence"]   = pbsdf->getClearcoatIridescence();
                matJson["clearcoatFilmThickness"] = pbsdf->getClearcoatFilmThickness();
                matJson["transmissionDensity"] = pbsdf->getTransmissionDensity();
                matJson["resinColor"] = vec3ToJson(pbsdf->getResinColor());
                matJson["resinRoughness"] = pbsdf->getResinRoughness();
                matJson["resinInclusion"] = pbsdf->getResinInclusion();
                matJson["resinDirt"] = pbsdf->getResinDirt();
                matJson["resinInclusionScale"] = pbsdf->getResinInclusionScale();
                matJson["resinDirtColor"] = vec3ToJson(pbsdf->getResinDirtColor());
                matJson["resinShard"] = pbsdf->getResinShard();
                matJson["resinShardHue"] = pbsdf->getResinShardHue();
                matJson["resinObjectSpace"] = pbsdf->getResinObjectSpace();
                matJson["dustStyle"] = pbsdf->getDustStyle();
                matJson["dustColorA"] = vec3ToJson(pbsdf->getDustColorA());
                matJson["dustColorB"] = vec3ToJson(pbsdf->getDustColorB());
                matJson["shardShape"] = pbsdf->getShardShape();
                matJson["glassMarbleVolume"] = pbsdf->getGlassMarbleVolume();

                // Legacy tiling fallback for older scene compatibility
                matJson["tilingFactor"] = { pbsdf->tilingFactor.x, pbsdf->tilingFactor.y };
                matJson["selectedUvSet"] = pbsdf->selected_uv_set;

                json uvTransformJson;
                uvTransformJson["scale"] = vec2ToJson(pbsdf->textureTransform.scale);
                uvTransformJson["offset"] = vec2ToJson(pbsdf->textureTransform.translation);
                uvTransformJson["rotationDegrees"] = pbsdf->textureTransform.rotation_degrees;
                uvTransformJson["tiling"] = vec2ToJson(pbsdf->textureTransform.tilingFactor);
                uvTransformJson["wrapMode"] = static_cast<int>(pbsdf->textureTransform.wrapMode);
                matJson["uvTransform"] = uvTransformJson;
            }
        }
        else if (mat->type() == MaterialType::Volumetric) {
            Volumetric* vol = dynamic_cast<Volumetric*>(mat.get());
            if (vol) {
                matJson["albedo"] = vec3ToJson(vol->getAlbedo());
                matJson["density"] = vol->getDensity();
                matJson["absorption"] = vol->getAbsorption();
                matJson["scattering"] = vol->getScattering();
                matJson["emissionColor"] = vec3ToJson(vol->getEmissionColor());
                matJson["g"] = vol->getG();
                matJson["stepSize"] = vol->getStepSize();
                matJson["maxSteps"] = vol->getMaxSteps();
                matJson["noiseScale"] = vol->getNoiseScale();
                matJson["voidThreshold"] = vol->getVoidThreshold();
                matJson["multiScatter"] = vol->getMultiScatter();
                matJson["gBack"] = vol->getGBack();
                matJson["lobeMix"] = vol->getLobeMix();
                matJson["lightSteps"] = vol->getLightSteps();
                matJson["shadowStrength"] = vol->getShadowStrength();
                matJson["vdbVolumeId"] = vol->getVDBVolumeID();
                matJson["densitySource"] = vol->getDensitySource();
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
        if (matJson.contains("heightProperty"))
            collectTextureRequest(matJson["heightProperty"], sceneDir, TextureType::Unknown, preloadRequests);
        if (matJson.contains("opacityProperty"))
            collectTextureRequest(matJson["opacityProperty"], sceneDir, TextureType::Opacity, preloadRequests);
        if (matJson.contains("emissionProperty"))
            collectTextureRequest(matJson["emissionProperty"], sceneDir, TextureType::Emission, preloadRequests);
        if (matJson.contains("transmissionProperty"))
            collectTextureRequest(matJson["transmissionProperty"], sceneDir, TextureType::Transmission, preloadRequests);
        if (matJson.contains("specularProperty"))
            collectTextureRequest(matJson["specularProperty"], sceneDir, TextureType::Specular, preloadRequests);
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
            if (matJson.contains("specularProperty"))
                deserializeProperty(pbsdf->specularProperty, matJson["specularProperty"], sceneDir, TextureType::Specular, &textureCache, &textureStats);
            if (matJson.contains("normalProperty"))
                deserializeProperty(pbsdf->normalProperty, matJson["normalProperty"], sceneDir, TextureType::Normal, &textureCache, &textureStats);
            if (matJson.contains("heightProperty"))
                deserializeProperty(pbsdf->heightProperty, matJson["heightProperty"], sceneDir, TextureType::Unknown, &textureCache, &textureStats);
            if (matJson.contains("opacityProperty"))
                deserializeProperty(pbsdf->opacityProperty, matJson["opacityProperty"], sceneDir, TextureType::Opacity, &textureCache, &textureStats);
            if (matJson.contains("emissionProperty"))
                deserializeProperty(pbsdf->emissionProperty, matJson["emissionProperty"], sceneDir, TextureType::Emission, &textureCache, &textureStats);
            if (matJson.contains("transmissionProperty"))
                deserializeProperty(pbsdf->transmissionProperty, matJson["transmissionProperty"], sceneDir, TextureType::Transmission, &textureCache, &textureStats);
            
            // Scalar values (with defaults for backward compatibility)
            pbsdf->transmission = matJson.value("transmission", 0.0f);
            pbsdf->dispersion = matJson.value("dispersion", 0.0f);
            pbsdf->metallic_tex_channel = matJson.value("metallicTexChannel", 0);
            pbsdf->roughness_tex_channel = matJson.value("roughnessTexChannel", 0);
            pbsdf->setClearcoat(matJson.value("clearcoat", 0.0f), matJson.value("clearcoatRoughness", 0.03f));
            pbsdf->anisotropic = matJson.value("anisotropic", 0.0f);
            pbsdf->normalStrength = matJson.value("normalStrength", 1.0f);
            
            // SSS
            pbsdf->setSubsurface(matJson.value("subsurface", 0.0f));
            if (matJson.contains("subsurfaceColor"))
                pbsdf->setSubsurfaceColor(jsonToVec3(matJson["subsurfaceColor"], Vec3(1.0f, 0.8f, 0.6f)));
            if (matJson.contains("subsurfaceRadius"))
                pbsdf->setSubsurfaceRadius(jsonToVec3(matJson["subsurfaceRadius"], Vec3(1.0f, 0.2f, 0.1f)));
            pbsdf->setSubsurfaceScale(matJson.value("subsurfaceScale", 0.05f));
            pbsdf->setSubsurfaceAnisotropy(matJson.value("subsurfaceAnisotropy", 0.0f));
            pbsdf->setSubsurfaceIOR(matJson.value("subsurfaceIOR", 1.4f));
            pbsdf->selected_uv_set = std::max(0, matJson.value("selectedUvSet", 0));
            
            // Translucent
            pbsdf->translucent = matJson.value("translucent", 0.0f);

            // Procedural surface detail (defaults = off for old scenes)
            pbsdf->micro_detail_strength = matJson.value("microDetailStrength", 0.0f);
            pbsdf->micro_detail_scale    = matJson.value("microDetailScale",    2.0f);
            pbsdf->tile_break_strength   = matJson.value("tileBreakStrength",   0.0f);

            // Thin-shell bubble (champagne / soda / soap close-up)
            pbsdf->setIsBubble(matJson.value("isBubble",   false));
            pbsdf->setBubbleIor(matJson.value("bubbleIor",  1.33f));
            pbsdf->setBubbleFilm(matJson.value("bubbleFilm", 0.0f));
            pbsdf->setClearcoatIridescence(matJson.value("clearcoatIridescence",   0.0f));
            pbsdf->setClearcoatFilmThickness(matJson.value("clearcoatFilmThickness", 0.55f));
            pbsdf->setTransmissionDensity(matJson.value("transmissionDensity", 0.0f));
            if (matJson.contains("resinColor"))
                pbsdf->setResinColor(jsonToVec3(matJson["resinColor"], Vec3(1.0f, 1.0f, 1.0f)));
            pbsdf->setResinRoughness(matJson.value("resinRoughness", 0.1f));
            pbsdf->setResinInclusion(matJson.value("resinInclusion", 0.0f));
            pbsdf->setResinDirt(matJson.value("resinDirt", 0.0f));
            pbsdf->setResinInclusionScale(matJson.value("resinInclusionScale", 8.0f));
            if (matJson.contains("resinDirtColor"))
                pbsdf->setResinDirtColor(jsonToVec3(matJson["resinDirtColor"], Vec3(0.18f, 0.14f, 0.10f)));
            pbsdf->setResinShard(matJson.value("resinShard", 0.0f));
            pbsdf->setResinShardHue(matJson.value("resinShardHue", -1.0f));
            pbsdf->setResinObjectSpace(matJson.value("resinObjectSpace", true));
            pbsdf->setDustStyle(matJson.value("dustStyle", 0));
            if (matJson.contains("dustColorA"))
                pbsdf->setDustColorA(jsonToVec3(matJson["dustColorA"], Vec3(1.0f, 1.0f, 1.0f)));
            if (matJson.contains("dustColorB"))
                pbsdf->setDustColorB(jsonToVec3(matJson["dustColorB"], Vec3(1.0f, 1.0f, 1.0f)));
            pbsdf->setShardShape(matJson.value("shardShape", 0));
            pbsdf->setGlassMarbleVolume(matJson.value("glassMarbleVolume", false));

            // Legacy tiling fallback
            if (matJson.contains("tilingFactor") && matJson["tilingFactor"].is_array()) {
                pbsdf->tilingFactor = Vec2(matJson["tilingFactor"][0], matJson["tilingFactor"][1]);
            }

            if (matJson.contains("uvTransform") && matJson["uvTransform"].is_object()) {
                const auto& uvTransformJson = matJson["uvTransform"];
                pbsdf->textureTransform.scale =
                    jsonToVec2(uvTransformJson.value("scale", json::array({1.0, 1.0})), Vec2(1.0, 1.0));
                pbsdf->textureTransform.translation =
                    jsonToVec2(uvTransformJson.value("offset", json::array({0.0, 0.0})), Vec2(0.0, 0.0));
                pbsdf->textureTransform.rotation_degrees = uvTransformJson.value("rotationDegrees", 0.0f);
                pbsdf->textureTransform.tilingFactor =
                    jsonToVec2(uvTransformJson.value("tiling", json::array({1.0, 1.0})), Vec2(1.0, 1.0));
                pbsdf->textureTransform.wrapMode =
                    static_cast<WrapMode>(uvTransformJson.value("wrapMode", static_cast<int>(WrapMode::Repeat)));
            } else {
                pbsdf->textureTransform.scale = Vec2(1.0, 1.0);
                pbsdf->textureTransform.translation = Vec2(0.0, 0.0);
                pbsdf->textureTransform.rotation_degrees = 0.0f;
                pbsdf->textureTransform.tilingFactor = pbsdf->tilingFactor;
                pbsdf->textureTransform.wrapMode = WrapMode::Repeat;
            }

            // Keep legacy/base tiling field in sync with the new transform payload.
            pbsdf->tilingFactor = pbsdf->textureTransform.tilingFactor;
            
            mat = pbsdf;
        }
        else if (type == MaterialType::Volumetric) {
            auto perlin = std::make_shared<Perlin>();
            
            Vec3 albedo = jsonToVec3(matJson.value("albedo", json::array({0.8f, 0.8f, 0.8f})), Vec3(0.8f));
            float density = matJson.value("density", 1.0f);
            float absorption = matJson.value("absorption", 0.1f);
            float scattering = matJson.value("scattering", 0.5f);
            Vec3 emission = jsonToVec3(matJson.value("emissionColor", json::array({0.0f, 0.0f, 0.0f})), Vec3(0.0f));
            
            auto vol = std::make_shared<Volumetric>(albedo, density, absorption, scattering, emission, perlin);
            
            // set remaining fields
            vol->setG(matJson.value("g", 0.0f));
            vol->setStepSize(matJson.value("stepSize", 0.05f));
            vol->setMaxSteps(matJson.value("maxSteps", 128));
            vol->setNoiseScale(matJson.value("noiseScale", 1.0f));
            vol->setVoidThreshold(matJson.value("voidThreshold", 0.0f));
            vol->setMultiScatter(matJson.value("multiScatter", 0.3f));
            vol->setGBack(matJson.value("gBack", -0.3f));
            vol->setLobeMix(matJson.value("lobeMix", 0.7f));
            vol->setLightSteps(matJson.value("lightSteps", 4));
            vol->setShadowStrength(matJson.value("shadowStrength", 0.8f));
            vol->setVDBVolumeID(matJson.value("vdbVolumeId", -1));
            vol->setDensitySource(matJson.value("densitySource", 0));
            
            mat = vol;
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
            
            const PBRMaterialSnapshot snapshot = capturePBRMaterialSnapshot(*pbsdf);
            applyPBRMaterialSnapshotToGpuMaterial(snapshot, *mat->gpuMaterial);

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
