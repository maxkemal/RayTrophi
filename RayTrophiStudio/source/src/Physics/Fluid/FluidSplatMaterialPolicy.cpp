#include "Fluid/FluidSplatMaterialPolicy.h"

#include "MaterialManager.h"

#include <cstddef>

namespace RayTrophiSim::Fluid {

bool isExistingSplatMaterial(int material_id) {
    if (material_id < 0 ||
        material_id >= static_cast<int>(MaterialManager::INVALID_MATERIAL_ID)) {
        return false;
    }
    return MaterialManager::getInstance().getMaterialShared(
               static_cast<uint16_t>(material_id)) != nullptr;
}

uint16_t resolveExistingSplatMaterial(int authored_material_id) {
    if (isExistingSplatMaterial(authored_material_id)) {
        return static_cast<uint16_t>(authored_material_id);
    }

    const auto& materials = MaterialManager::getInstance().getAllMaterials();
    for (std::size_t i = 0;
         i < materials.size() && i < MaterialManager::INVALID_MATERIAL_ID; ++i) {
        if (materials[i]) return static_cast<uint16_t>(i);
    }

    // A scene normally owns material 0. Keeping the triangle ID representable
    // is safer than creating hidden fluid assets when an empty registry is
    // temporarily observed during project load.
    return 0u;
}

} // namespace RayTrophiSim::Fluid
