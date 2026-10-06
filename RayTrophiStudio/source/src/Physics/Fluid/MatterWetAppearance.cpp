#include "Fluid/MatterWetAppearance.h"
#include "Fluid/MatterWetResponse.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/FluidSplatMaterialPolicy.h"
#include "Fluid/SubstanceTag.h"
#include "ParticleSimulation.h"
#include "MaterialStateField.h"
#include "MaterialManager.h"
#include "PBRMaterialSnapshot.h"
#include "PrincipledBSDF.h"
#include "globals.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace RayTrophiSim::Fluid {
namespace {
struct VariantState {
    uint64_t generation = 0;
    uint64_t fingerprint = 0;
    uint16_t material = MaterialManager::INVALID_MATERIAL_ID;
};
std::unordered_map<std::string, VariantState> variants;
uint64_t variants_generation = 0;

uint64_t fingerprint(const PrincipledBSDF& dry, const MatterPoreParams& params) {
    GpuMaterial gpu{};
    applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(dry), gpu);
    uint64_t hash = matterPoreSettingsHash(params) ^
        static_cast<uint64_t>(std::hash<GpuMaterial>{}(gpu));
    // Include texture ownership so replacing a texture with identical scalars
    // still refreshes all variants; clone keeps the authored texture channels.
    const MaterialProperty* properties[] = {&dry.albedoProperty, &dry.roughnessProperty,
        &dry.metallicProperty, &dry.normalProperty, &dry.opacityProperty,
        &dry.specularProperty, &dry.transmissionProperty, &dry.emissionProperty,
        &dry.heightProperty};
    for (const auto* property : properties) {
        hash = (hash ^ reinterpret_cast<uintptr_t>(property->texture.get())) * 1099511628211ull;
        const float values[] = {property->color.x, property->color.y, property->color.z,
            property->intensity, property->alpha};
        for (const float value : values) {
            hash = (hash ^ std::hash<float>{}(value)) * 1099511628211ull;
        }
    }
    return hash;
}
}

MatterWetPalette prepareMatterWetPalette(const SimulationGridDomainDesc& domain,
                                        const FluidParticles& particles) {
    MatterWetPalette palette;
    const auto& params = domain.fluid_params.pore_exchange;
    if (domain.type != SimulationDomainType::Matter || !params.wet_appearance_enabled) {
        return palette;
    }
    palette.legacy_granular = domain.fluid_params.granular_enabled;
    palette.appearance_full_saturation = params.wet_appearance_full_saturation;
    auto bindings = domain.fluid_substance_materials;
    std::unordered_set<uint32_t> seen;
    for (const auto& binding : bindings) {
        seen.insert(substanceTag(binding.substance));
    }
    for (const auto tag : particles.substance_tag) {
        if (!seen.insert(tag).second) {
            continue;
        }
        const auto* profile = tryFindSubstanceByTag(tag);
        if (profile && profile->default_constitutive_model == MatterConstitutiveModel::Granular) {
            SimulationGridDomainDesc::SubstanceMaterial binding;
            binding.substance = profile->name;
            bindings.push_back(binding);
        }
    }
    auto& manager = MaterialManager::getInstance();
    const auto generation = manager.generation();
    if (variants_generation != generation) {
        variants.clear();
        variants_generation = generation;
    }
    for (const auto& binding : bindings) {
        const auto* profile = tryFindSubstance(binding.substance);
        if (!profile || profile->default_constitutive_model != MatterConstitutiveModel::Granular) {
            continue;
        }
        const auto dry_id = isExistingSplatMaterial(binding.material_id)
            ? static_cast<uint16_t>(binding.material_id) : resolveExistingSplatMaterial(
                domain.fluid_particle_material_id);
        const auto dry = std::dynamic_pointer_cast<PrincipledBSDF>(
            manager.getMaterialShared(dry_id));
        if (!dry) {
            continue;
        }
        MatterWetPaletteEntry entry;
        entry.substance_tag = substanceTag(binding.substance);
        entry.dry_material = dry_id;
        entry.materials.fill(dry_id);
        const auto hash = fingerprint(*dry, params);
        for (int bin = 1; bin < 8; ++bin) {
            const std::string name = "[MatterWet] " + domain.name + " " + binding.substance +
                " " + std::to_string(bin);
            auto& cached = variants[name];
            auto variant = cached.generation == generation
                ? std::dynamic_pointer_cast<PrincipledBSDF>(
                    manager.getMaterialShared(cached.material))
                : nullptr;
            const bool fresh = !variant || variant->materialName != name;
            if (fresh) {
                const auto existing = manager.getMaterialID(name);
                variant = std::dynamic_pointer_cast<PrincipledBSDF>(
                    manager.getMaterialShared(existing));
                if (!variant) {
                    if (existing != MaterialManager::INVALID_MATERIAL_ID) {
                        continue;
                    }
                    variant = std::make_shared<PrincipledBSDF>(*dry);
                }
                cached.material = existing;
            }
            if (fresh || cached.fingerprint != hash) {
                *variant = *dry;
                variant->materialName = name;
                const float saturation = static_cast<float>(bin) / 7.0f;
                const float color_scale = 1.0f + (params.wet_color_scale - 1.0f) * saturation;
                const float roughness_scale = 1.0f +
                    (params.wet_roughness_scale - 1.0f) * saturation;
                variant->albedoProperty.color = dry->albedoProperty.color * color_scale;
                const float roughness = std::clamp(dry->getScalarRoughness() *
                    roughness_scale, 0.02f, 1.0f);
                variant->roughnessProperty.color = Vec3(roughness);
                variant->roughnessProperty.intensity = 1.0f;
                variant->setRoughness(roughness);
                // Texture evaluation uses intensity; preserve its map and scale it.
                if (variant->roughnessProperty.texture) {
                    variant->roughnessProperty.intensity =
                        dry->roughnessProperty.intensity * roughness_scale;
                }
                variant->gpuMaterial = std::make_shared<GpuMaterial>();
                applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(*variant),
                    *variant->gpuMaterial);
                if (fresh && cached.material == MaterialManager::INVALID_MATERIAL_ID) {
                    cached.material = manager.addMaterial(name, variant);
                }
                cached.generation = generation;
                cached.fingerprint = hash;
                ::g_materials_dirty = true;
            }
            if (cached.material != MaterialManager::INVALID_MATERIAL_ID) {
                entry.materials[bin] = cached.material;
            }
        }
        palette.entries.push_back(entry);
    }
    return palette;
}

void MatterWetPalette::appendMaterialKeys(std::vector<int>& keys) const {
    for (const auto& entry : entries) {
        for (const int material : entry.materials) {
            if (std::find(keys.begin(), keys.end(), material) == keys.end()) {
                keys.push_back(material);
            }
        }
    }
}

std::size_t MatterWetPalette::sourceIndex(const FluidParticles& particles, std::size_t particle,
                                        const std::vector<int>& material_keys,
                                        std::size_t dry_source) const {
    if (particle >= particles.substance_tag.size()) {
        return dry_source;
    }
    auto model = particle < particles.constitutive_model.size()
        ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[particle])
        : MatterConstitutiveModel::Auto;
    if (model == MatterConstitutiveModel::Auto) {
        model = legacy_granular ? MatterConstitutiveModel::Granular
            : MatterConstitutiveModel::Fluid;
    }
    if (model != MatterConstitutiveModel::Granular) {
        return dry_source;
    }
    const auto tag = particles.substance_tag[particle];
    for (const auto& entry : entries) {
        if (entry.substance_tag != tag) {
            continue;
        }
        const float saturation = matterParticleSaturation(particles, particle);
        const int bin = matterWetAppearanceBand(saturation, appearance_full_saturation);
        const auto found = std::find(
            material_keys.begin(), material_keys.end(), entry.materials[bin]);
        return found != material_keys.end()
            ? static_cast<std::size_t>(found - material_keys.begin()) : dry_source;
    }
    return dry_source;
}
} // namespace RayTrophiSim::Fluid
