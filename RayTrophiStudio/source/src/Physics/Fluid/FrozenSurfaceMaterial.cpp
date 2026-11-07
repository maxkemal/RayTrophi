#include "Fluid/FrozenSurfaceMaterial.h"
#include "Fluid/MatterMaterialPreset.h"
#include "MaterialManager.h"
#include "ParticleSimulation.h"
#include "globals.h"

#include <utility>

namespace RayTrophiSim::Fluid {

uint64_t frozenSurfaceMaterialRegistrySignature() {
    const auto& manager = MaterialManager::getInstance();
    return (manager.generation() << 32u) ^ manager.getMaterialID("Ice (Automatic)");
}

std::size_t collectPhaseSurfaceMaterialEntries(const SimulationGridDomainDesc& domain,
    const FluidParticles& particles, SubstanceMaterialEntry* entries, bool createDefault) {
    std::size_t count = collectSubstanceMaterialEntries(domain.fluid_substance_materials,
        domain.fluid_params.default_substance, entries);
    if (domain.fluid_surface_material_id >= 0) {
        return count;
    }
    const auto plan = resolveFluidViews(domain, {});
    bool needsIce = false;
    for (std::size_t i = 0; i < particles.size() && i < particles.flags.size(); ++i) {
        if ((particles.flags[i] & kParticleFlagFrozen) != 0u &&
            plan.viewForParticle(particles, i) == FluidView::Surface &&
            surfaceMaterialForParticle(particles, i, entries, count, -1) < 0) {
            needsIce = true;
            break;
        }
    }
    if (!needsIce) {
        return count;
    }
    int material = -1;
    for (const auto& binding : domain.fluid_substance_materials) {
        if (binding.substance == "Ice" && binding.material_id >= 0) {
            material = binding.material_id;
            break;
        }
    }
    if (material < 0) {
        auto& manager = MaterialManager::getInstance();
        constexpr const char* name = "Ice (Automatic)";
        auto id = manager.getMaterialID(name);
        if (id == MaterialManager::INVALID_MATERIAL_ID && createDefault) {
            std::string error;
            auto preset = createMatterMaterialPreset("Ice", error);
            if (preset) {
                id = manager.addUniqueMaterial(name, std::move(preset));
                if (id != MaterialManager::INVALID_MATERIAL_ID) {
                    manager.syncAllGpuMaterials();
                    g_materials_dirty = true;
                }
            }
        }
        if (id != MaterialManager::INVALID_MATERIAL_ID) {
            material = static_cast<int>(id);
        }
    }
    if (material >= 0 && count < kSubstanceMaterialEntryCapacity) {
        entries[count++] = {0u, material, 0.15f, true};
    }
    return count;
}

} // namespace RayTrophiSim::Fluid
