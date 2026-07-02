#include "HitMaterialResolver.h"

#include "MaterialManager.h"
#include "TerrainManager.h"
#include "PrincipledBSDF.h"
#include <algorithm>

namespace HitMaterialResolver {

namespace {

float sampleThicknessValue(const PrincipledBSDF& mat, const Vec2& uv) {
    float thickness = 0.0f;
    if (mat.heightProperty.texture) {
        thickness = mat.heightProperty.texture->sampleIntensity(uv.u, uv.v);
    } else {
        thickness = static_cast<float>(mat.heightProperty.color.x * mat.heightProperty.intensity);
    }

    thickness = std::max(0.0f, thickness);
    thickness *= mat.surface_deposition.thickness_scale;
    thickness = std::min(thickness, mat.surface_deposition.max_thickness);
    return thickness;
}

void applySurfaceDepositionIfNeeded(HitRecord& rec) {
    if (rec.terrain_id != -1 || !rec.materialPtr) {
        return;
    }

    auto* pbsdf = dynamic_cast<PrincipledBSDF*>(rec.materialPtr);
    if (!pbsdf || !pbsdf->surface_deposition.enabled) {
        return;
    }

    const float thickness = sampleThicknessValue(*pbsdf, rec.uv);
    if (thickness <= 1e-6f) {
        return;
    }

    const float world_offset = thickness * std::max(0.0f, pbsdf->surface_deposition.hit_offset_scale) * 0.1f;
    if (world_offset <= 1e-6f) {
        return;
    }

    rec.surface_override.deposited_thickness = thickness;
    rec.surface_override.hit_offset = world_offset;
}

} // namespace

void resolveMaterialPointers(HitRecord& rec) {
    if (!rec.materialPtr && rec.materialID != MaterialManager::INVALID_MATERIAL_ID) {
        rec.materialPtr = MaterialManager::getInstance().getMaterial(rec.materialID);
    } else if (rec.materialID == MaterialManager::INVALID_MATERIAL_ID) {
        rec.materialPtr = nullptr;
    }
}

void applyTerrainBlendIfNeeded(HitRecord& rec) {
    rec.surface_override = {};

    if (rec.terrain_id == -1) {
        return;
    }

    TerrainObject* terrain = TerrainManager::getInstance().getTerrain(rec.terrain_id);
    if (!terrain || !terrain->splatMap || terrain->layers.empty()) {
        return;
    }

    Vec3 rgb = terrain->splatMap->get_color_bilinear(rec.u, rec.v);
    float a = terrain->splatMap->get_alpha_bilinear(rec.u, rec.v);
    float weights[4] = { rgb.x, rgb.y, rgb.z, a };

    Vec3 blended_albedo(0.0f);
    float blended_roughness = 0.0f;
    float blended_metallic = 0.0f;
    float blended_clearcoat = 0.0f;
    float blended_clearcoat_roughness = 0.0f;
    float blended_subsurface = 0.0f;
    Vec3 blended_subsurface_color(0.0f);
    float blended_transmission = 0.0f;
    float blended_ior = 0.0f;
    float total_weight = 0.0f;

    for (size_t i = 0; i < 4 && i < terrain->layers.size(); ++i) {
        float weight = weights[i];
        if (weight <= 0.001f) {
            continue;
        }

        auto mat = dynamic_cast<PrincipledBSDF*>(terrain->layers[i].get());
        if (!mat) {
            continue;
        }

        float scale = (i < terrain->layer_uv_scales.size()) ? terrain->layer_uv_scales[i] : 1.0f;
        Vec2 layer_uv = rec.uv * scale;

        blended_albedo = blended_albedo + mat->getPropertyValue(mat->albedoProperty, layer_uv) * weight;
        blended_roughness += mat->getPropertyValue(mat->roughnessProperty, layer_uv).y * weight;
        blended_metallic += mat->getPropertyValue(mat->metallicProperty, layer_uv).z * weight;
        blended_clearcoat += mat->getClearcoat() * weight;
        blended_clearcoat_roughness += mat->getClearcoatRoughness() * weight;
        blended_subsurface += mat->getSubsurface() * weight;
        blended_subsurface_color = blended_subsurface_color + mat->getSubsurfaceColor() * weight;
        blended_transmission += mat->transmission * weight;
        blended_ior += mat->getIndexOfRefraction() * weight;
        total_weight += weight;
    }

    if (total_weight <= 0.001f) {
        return;
    }

    float inv_weight = 1.0f / total_weight;
    rec.surface_override.valid = true;
    rec.surface_override.albedo = blended_albedo * inv_weight;
    rec.surface_override.roughness = blended_roughness * inv_weight;
    rec.surface_override.metallic = blended_metallic * inv_weight;
    rec.surface_override.clearcoat = blended_clearcoat * inv_weight;
    rec.surface_override.clearcoat_roughness = blended_clearcoat_roughness * inv_weight;
    rec.surface_override.subsurface = blended_subsurface * inv_weight;
    rec.surface_override.subsurface_color = blended_subsurface_color * inv_weight;
    rec.surface_override.transmission = blended_transmission * inv_weight;
    rec.surface_override.ior = blended_ior * inv_weight;
}

void resolveSurfaceData(HitRecord& rec) {
    resolveMaterialPointers(rec);
    applyTerrainBlendIfNeeded(rec);
    applySurfaceDepositionIfNeeded(rec);
}

}
