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
    Vec3 resolvedAlbedo = blended_albedo * inv_weight;
    float resolvedRoughness = blended_roughness * inv_weight;
    float resolvedMetallic = blended_metallic * inv_weight;
    float resolvedClearcoat = blended_clearcoat * inv_weight;
    float resolvedClearcoatRoughness = blended_clearcoat_roughness * inv_weight;
    float resolvedSubsurface = blended_subsurface * inv_weight;
    Vec3 resolvedSubsurfaceColor = blended_subsurface_color * inv_weight;
    float resolvedTransmission = blended_transmission * inv_weight;
    float resolvedIor = blended_ior * inv_weight;

    if (terrain->surfaceSemanticMap && terrain->surfaceSemanticMap->is_loaded()) {
        const Vec3 semanticRgb = terrain->surfaceSemanticMap->get_color_bilinear(rec.u, rec.v);
        const float hardness = terrain->surfaceSemanticMap->get_alpha_bilinear(rec.u, rec.v);
        const float flow = std::clamp(static_cast<float>(semanticRgb.x), 0.0f, 1.0f);
        const float wet = std::clamp(static_cast<float>(semanticRgb.y), 0.0f, 1.0f);
        const float ice = std::clamp(static_cast<float>(semanticRgb.z), 0.0f, 1.0f);
        const float semanticWeight[TerrainObject::kSemanticLayerSlots] = {flow, wet, ice, hardness};

        // Overlay slots composite OVER the normalized splat blend, each by
        // its own weight. They are not part of that normalization: a
        // semantic weight states how strongly a condition holds here, not
        // what share of the pixel it owns, and a river bed covers the
        // substrate rather than competing with it for area.
        // A semantic value is a MEASUREMENT; an overlay's coverage is a
        // VISIBILITY decision. Flow reads 0.9 under two metres of snow - the
        // measurement is right and erosion must keep reading it, but painting
        // it there drew a river across a snow-filled valley. Overlays
        // composite UNDER the snow the splat map placed.
        const float splatTotal = (std::max)(
            weights[0] + weights[1] + weights[2] + weights[3], 0.001f);
        const float snowCover = std::clamp(weights[2] / splatTotal, 0.0f, 1.0f);
        const float exposed = 1.0f - snowCover;
        // Hardness is not a surface condition - it is a SUBSTRATE property.
        // Hardness 0.8 under two metres of valley soil is still 0.8, so a
        // material bound to it painted granite across meadows. Gated by rock
        // exposure it becomes a bedrock VARIANT instead: hard outcrops against
        // soft ones, visible only where rock shows. The unmasked field still
        // drives erosion resistance and soil capacity.
        const float rockExposure = std::clamp(weights[1] / splatTotal, 0.0f, 1.0f);

        // Bottom to top: Hardness (bedrock), Flow, Wetness, then Ice as cover.
        // The old loop ran 0,1,2,3, putting bedrock over both water and ice.
        static constexpr int kOverlayOrder[TerrainObject::kSemanticLayerSlots] = {3, 0, 1, 2};
        const auto overlayBound = [terrain](int channel) {
            const size_t slot = static_cast<size_t>(TerrainObject::kSplatLayerSlots + channel);
            return slot < terrain->layers.size() && terrain->layers[slot] != nullptr;
        };
        const bool overlayCoveredWetness = overlayBound(0) || overlayBound(1);
        const bool overlayCoveredIce = overlayBound(2);
        for (const int s : kOverlayOrder) {
            const size_t slot = static_cast<size_t>(TerrainObject::kSplatLayerSlots + s);
            if (slot >= terrain->layers.size() || !terrain->layers[slot]) continue;
            auto* overlay = dynamic_cast<PrincipledBSDF*>(terrain->layers[slot].get());
            if (!overlay) continue;

            const float strength = slot < terrain->layer_overlay_strength.size()
                ? terrain->layer_overlay_strength[slot] : 1.0f;
            float coverage = std::clamp(semanticWeight[s] * strength, 0.0f, 1.0f);
            // Ice is a cover in its own right and is never buried. Hardness is
            // gated by exposure, which is already net of snow. The rest are
            // buried by snow, unless the author opted the slot out.
            const bool ignoresCover = slot < terrain->layer_overlay_ignore_cover.size() &&
                terrain->layer_overlay_ignore_cover[slot] != 0;
            if (!ignoresCover) {
                if (s == 3)      coverage *= rockExposure;
                else if (s != 2) coverage *= exposed;
            }
            if (coverage <= 0.001f) continue;

            const float scale = (slot < terrain->layer_uv_scales.size())
                ? terrain->layer_uv_scales[slot] : 1.0f;
            const Vec2 overlay_uv = rec.uv * scale;

            resolvedAlbedo = resolvedAlbedo * (1.0f - coverage) +
                overlay->getPropertyValue(overlay->albedoProperty, overlay_uv) * coverage;
            resolvedRoughness = resolvedRoughness * (1.0f - coverage) +
                overlay->getPropertyValue(overlay->roughnessProperty, overlay_uv).y * coverage;
            resolvedMetallic = resolvedMetallic * (1.0f - coverage) +
                overlay->getPropertyValue(overlay->metallicProperty, overlay_uv).z * coverage;
            resolvedClearcoat = resolvedClearcoat * (1.0f - coverage) +
                overlay->getClearcoat() * coverage;
            resolvedClearcoatRoughness = resolvedClearcoatRoughness * (1.0f - coverage) +
                overlay->getClearcoatRoughness() * coverage;
            resolvedSubsurface = resolvedSubsurface * (1.0f - coverage) +
                overlay->getSubsurface() * coverage;
            resolvedSubsurfaceColor = resolvedSubsurfaceColor * (1.0f - coverage) +
                overlay->getSubsurfaceColor() * coverage;
            resolvedTransmission = resolvedTransmission * (1.0f - coverage) +
                overlay->transmission * coverage;
            resolvedIor = resolvedIor * (1.0f - coverage) +
                overlay->getIndexOfRefraction() * coverage;

        }

        // Built-in shading remains for every channel without a material, so
        // a terrain authored before overlays existed renders identically. It
        // is buried by snow for the same reason the overlays are.
        if (!overlayCoveredWetness) {
            const float wetness = (std::max)(flow, wet) * exposed;
            resolvedAlbedo = resolvedAlbedo * (1.0f - wetness * 0.28f);
            resolvedRoughness = resolvedRoughness * (1.0f - wetness * 0.65f) + 0.16f * wetness * 0.65f;
        }
        if (!overlayCoveredIce) {
            const float iceLuma = resolvedAlbedo.x * 0.2126f + resolvedAlbedo.y * 0.7152f +
                resolvedAlbedo.z * 0.0722f;
            const Vec3 iceColor = Vec3(0.70f, 0.82f, 0.88f) * (std::max)(iceLuma, 0.35f);
            resolvedAlbedo = resolvedAlbedo * (1.0f - ice * 0.55f) + iceColor * (ice * 0.55f);
            resolvedRoughness = resolvedRoughness * (1.0f - ice * 0.65f) + 0.12f * ice * 0.65f;
        }
        const size_t hardnessSlot = static_cast<size_t>(TerrainObject::kSplatLayerSlots + 3);
        const bool hardnessOverlay = hardnessSlot < terrain->layers.size() &&
            terrain->layers[hardnessSlot] != nullptr;
        if (!hardnessOverlay) {
            // Same gate as the overlay: hard bedrock only roughens the surface
            // where bedrock IS the surface.
            resolvedRoughness = std::clamp(resolvedRoughness + hardness * 0.035f * rockExposure,
                                           0.0f, 1.0f);
        }
    }
    rec.surface_override.valid = true;
    rec.surface_override.albedo = resolvedAlbedo;
    rec.surface_override.roughness = std::clamp(resolvedRoughness, 0.0f, 1.0f);
    rec.surface_override.metallic = resolvedMetallic;
    rec.surface_override.clearcoat = resolvedClearcoat;
    rec.surface_override.clearcoat_roughness = resolvedClearcoatRoughness;
    rec.surface_override.subsurface = resolvedSubsurface;
    rec.surface_override.subsurface_color = resolvedSubsurfaceColor;
    rec.surface_override.transmission = resolvedTransmission;
    rec.surface_override.ior = resolvedIor;
}

void resolveSurfaceData(HitRecord& rec) {
    resolveMaterialPointers(rec);
    applyTerrainBlendIfNeeded(rec);
    applySurfaceDepositionIfNeeded(rec);
}

}
