#include "MaterialManager.h"
#include <iostream>

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
