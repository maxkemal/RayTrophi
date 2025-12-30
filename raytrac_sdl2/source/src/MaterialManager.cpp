#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "globals.h"
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
        // Store as-is for now, could make relative later
        j["texture"] = texPath;
    }
    
    return j;
}

// Helper: Deserialize MaterialProperty
static void deserializeProperty(MaterialProperty& prop, const json& j, const std::string& sceneDir) {
    prop.color = jsonToVec3(j.value("color", json::array({1,1,1})), Vec3(1,1,1));
    prop.intensity = j.value("intensity", 1.0f);
    prop.alpha = j.value("alpha", 1.0f);
    
    // Load texture if specified
    if (j.contains("texture") && !j["texture"].get<std::string>().empty()) {
        std::string texPath = j["texture"].get<std::string>();
        // Try loading texture
        // For now, use absolute path or relative to sceneDir
        std::string fullPath = texPath;
        if (!std::filesystem::exists(fullPath) && !sceneDir.empty()) {
            fullPath = sceneDir + "/" + texPath;
        }
        if (std::filesystem::exists(fullPath)) {
            prop.texture = std::make_shared<Texture>(fullPath, TextureType::Albedo);
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
        std::cerr << "[MaterialManager] Maximum material count reached!" << std::endl;
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
        return it->second;
    }

    if (materials.size() >= MAX_MATERIALS) {
        std::cerr << "[MaterialManager] Maximum material count reached!" << std::endl;
        return INVALID_MATERIAL_ID;
    }

    uint16_t id = static_cast<uint16_t>(materials.size());
    materials.push_back(mat);
    nameToID[name] = id;

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
                matJson["normalStrength"] = pbsdf->normalStrength;
                matJson["subsurfaceColor"] = vec3ToJson(pbsdf->subsurfaceColor);
                matJson["subsurfaceRadius"] = vec3ToJson(pbsdf->subsurfaceRadius);
                
                // Tiling
                matJson["tilingFactor"] = { pbsdf->tilingFactor.x, pbsdf->tilingFactor.y };
            }
        }
        
        materialsArray.push_back(matJson);
    }
    
    root["materials"] = materialsArray;
    
    SCENE_LOG_INFO("[MaterialManager] Serialized " + std::to_string(materials.size()) + " materials");
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
            if (matJson.contains("albedoProperty"))
                deserializeProperty(pbsdf->albedoProperty, matJson["albedoProperty"], sceneDir);
            if (matJson.contains("roughnessProperty"))
                deserializeProperty(pbsdf->roughnessProperty, matJson["roughnessProperty"], sceneDir);
            if (matJson.contains("metallicProperty"))
                deserializeProperty(pbsdf->metallicProperty, matJson["metallicProperty"], sceneDir);
            if (matJson.contains("normalProperty"))
                deserializeProperty(pbsdf->normalProperty, matJson["normalProperty"], sceneDir);
            if (matJson.contains("opacityProperty"))
                deserializeProperty(pbsdf->opacityProperty, matJson["opacityProperty"], sceneDir);
            if (matJson.contains("emissionProperty"))
                deserializeProperty(pbsdf->emissionProperty, matJson["emissionProperty"], sceneDir);
            if (matJson.contains("transmissionProperty"))
                deserializeProperty(pbsdf->transmissionProperty, matJson["transmissionProperty"], sceneDir);
            
            // Scalar values (with defaults for backward compatibility)
            pbsdf->transmission = matJson.value("transmission", 0.0f);
            pbsdf->clearcoat = matJson.value("clearcoat", 0.0f);
            pbsdf->clearcoatRoughness = matJson.value("clearcoatRoughness", 0.0f);
            pbsdf->anisotropic = matJson.value("anisotropic", 0.0f);
            pbsdf->normalStrength = matJson.value("normalStrength", 1.0f);
            
            if (matJson.contains("subsurfaceColor"))
                pbsdf->subsurfaceColor = jsonToVec3(matJson["subsurfaceColor"]);
            if (matJson.contains("subsurfaceRadius"))
                pbsdf->subsurfaceRadius = jsonToVec3(matJson["subsurfaceRadius"]);
            
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
}
