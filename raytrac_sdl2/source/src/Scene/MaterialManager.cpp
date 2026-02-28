#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "material_gpu.h"
#include "globals.h"
#include "ProjectManager.h"
#include <iostream>
#include <filesystem>
#include <fstream>

using json = nlohmann::json;

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
                                 TextureType texType = TextureType::Albedo) {
    prop.color = jsonToVec3(j.value("color", json::array({1,1,1})), Vec3(1,1,1));
    prop.intensity = j.value("intensity", 1.0f);
    prop.alpha = j.value("alpha", 1.0f);
    
    // Load texture if specified
    if (j.contains("texture") && !j["texture"].get<std::string>().empty()) {
        std::string texPath = j["texture"].get<std::string>();
        
        // FIRST: Check embedded texture cache (memory-based, no disk I/O)
        auto& pm = ProjectManager::getInstance();
        auto* embedded = pm.getEmbeddedTexture(texPath);
        if (embedded) {
            // Load directly from memory buffer - fast and no temp files!
            prop.texture = std::make_shared<Texture>(embedded->data, texType, texPath);
            // SCENE_LOG_INFO("[MATERIAL] Loaded embedded texture from memory: " + texPath);
            return;
        }
        
        // FALLBACK: Try loading from disk (for external textures)
        std::string fullPath = texPath;
        if (!std::filesystem::exists(fullPath) && !sceneDir.empty()) {
            fullPath = sceneDir + "/" + texPath;
        }
        if (std::filesystem::exists(fullPath)) {
            prop.texture = std::make_shared<Texture>(fullPath, texType);
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
    // Clear existing materials first
    clear();
    
    std::lock_guard<std::mutex> lock(mutex);
    
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
    
    for (const auto& matJson : data["materials"]) {
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
                deserializeProperty(pbsdf->albedoProperty, matJson["albedoProperty"], sceneDir, TextureType::Albedo);
            if (matJson.contains("roughnessProperty"))
                deserializeProperty(pbsdf->roughnessProperty, matJson["roughnessProperty"], sceneDir, TextureType::Roughness);
            if (matJson.contains("metallicProperty"))
                deserializeProperty(pbsdf->metallicProperty, matJson["metallicProperty"], sceneDir, TextureType::Metallic);
            if (matJson.contains("normalProperty"))
                deserializeProperty(pbsdf->normalProperty, matJson["normalProperty"], sceneDir, TextureType::Normal);
            if (matJson.contains("opacityProperty"))
                deserializeProperty(pbsdf->opacityProperty, matJson["opacityProperty"], sceneDir, TextureType::Opacity);
            if (matJson.contains("emissionProperty"))
                deserializeProperty(pbsdf->emissionProperty, matJson["emissionProperty"], sceneDir, TextureType::Emission);
            if (matJson.contains("transmissionProperty"))
                deserializeProperty(pbsdf->transmissionProperty, matJson["transmissionProperty"], sceneDir, TextureType::Transmission);
            
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
