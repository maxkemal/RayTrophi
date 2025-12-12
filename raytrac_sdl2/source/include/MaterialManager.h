#ifndef MATERIAL_MANAGER_H
#define MATERIAL_MANAGER_H

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include "Material.h"

/**
 * @brief Centralized material management for memory optimization.
 * 
 * This class replaces per-triangle material pointers with material IDs,
 * reducing memory usage from ~72 bytes (shared_ptr + GpuMaterial ptr + string)
 * to 2 bytes (uint16_t ID) per triangle.
 */
class MaterialManager {
public:
    static constexpr uint16_t INVALID_MATERIAL_ID = 0xFFFF;
    static constexpr size_t MAX_MATERIALS = 65535;

    static MaterialManager& getInstance() {
        static MaterialManager instance;
        return instance;
    }

    // Prevent copying
    MaterialManager(const MaterialManager&) = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;

    /**
     * @brief Register a material and get its ID
     * @param name Material name (for lookup)
     * @param mat Material shared pointer
     * @return Material ID (0-65534), or INVALID_MATERIAL_ID if full
     */
    uint16_t addMaterial(const std::string& name, std::shared_ptr<Material> mat);

    /**
     * @brief Get material ID by name
     * @param name Material name
     * @return Material ID, or INVALID_MATERIAL_ID if not found
     */
    uint16_t getMaterialID(const std::string& name) const;

    /**
     * @brief Get material by ID
     * @param id Material ID
     * @return Raw pointer to material (not owning), nullptr if invalid
     */
    Material* getMaterial(uint16_t id) const;

    /**
     * @brief Get shared_ptr to material by ID
     * @param id Material ID
     * @return Shared pointer to material
     */
    std::shared_ptr<Material> getMaterialShared(uint16_t id) const;

    /**
     * @brief Get or create material ID
     * @param name Material name
     * @param mat Material to add if not exists
     * @return Material ID
     */
    uint16_t getOrCreateMaterialID(const std::string& name, std::shared_ptr<Material> mat);

    /**
     * @brief Check if material exists
     * @param name Material name
     * @return true if exists
     */
    bool hasMaterial(const std::string& name) const;

    /**
     * @brief Get total material count
     */
    size_t getMaterialCount() const { return materials.size(); }

    /**
     * @brief Clear all materials (use with caution)
     */
    void clear();

    /**
     * @brief Get all materials (for serialization/debugging)
     */
    const std::vector<std::shared_ptr<Material>>& getAllMaterials() const { return materials; }

private:
    MaterialManager() = default;

    std::vector<std::shared_ptr<Material>> materials;
    std::unordered_map<std::string, uint16_t> nameToID;
    mutable std::mutex mutex;
};

#endif // MATERIAL_MANAGER_H
