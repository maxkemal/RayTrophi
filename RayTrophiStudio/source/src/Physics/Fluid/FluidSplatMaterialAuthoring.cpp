#include "Fluid/FluidSplatMaterialAuthoring.h"

#include "MaterialManager.h"
#include "ParticleSimulation.h"

#include <cstddef>

namespace RayTrophiSim::Fluid {

namespace {

int findMaterialByName(const std::string& name) {
    const auto& materials = MaterialManager::getInstance().getAllMaterials();
    for (std::size_t i = 0; i < materials.size(); ++i) {
        if (materials[i] && materials[i]->materialName == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

} // namespace

SplatMaterialAuthoringResult setFluidSplatMaterial(
    ParticleSimulationSystem& simulation,
    const std::string& domain_name,
    const std::string& material_name) {
    for (auto& domain : simulation.gridDomains()) {
        if (domain.name != domain_name ||
            domain.type != SimulationDomainType::Fluid) {
            continue;
        }

        int material_id = -1;
        if (!material_name.empty()) {
            material_id = findMaterialByName(material_name);
            if (material_id < 0) {
                return {SplatMaterialAuthoringStatus::MaterialNotFound,
                        false, -1};
            }
        }

        const bool changed = domain.fluid_particle_material_id != material_id;
        domain.fluid_particle_material_id = material_id;
        return {SplatMaterialAuthoringStatus::Applied, changed, material_id};
    }

    return {SplatMaterialAuthoringStatus::DomainNotFound, false, -1};
}

std::string fluidSplatMaterialName(const SimulationGridDomainDesc& domain) {
    if (domain.fluid_particle_material_id < 0 ||
        domain.fluid_particle_material_id >=
            static_cast<int>(MaterialManager::INVALID_MATERIAL_ID)) {
        return {};
    }
    const auto material = MaterialManager::getInstance().getMaterialShared(
        static_cast<uint16_t>(domain.fluid_particle_material_id));
    return material ? material->materialName : std::string{};
}

} // namespace RayTrophiSim::Fluid
