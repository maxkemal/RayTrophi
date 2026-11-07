#pragma once

#include "UI/MaterialInspectorControls.h"
#include "PrincipledBSDF.h"

#include <cstdint>
#include <functional>

namespace MaterialSurfaceBasicsUI {

// Reuses the caller's existing keyframe operations and material write path.
// No new authoring operation or independent material state is introduced.
using IsKeyed = std::function<bool(uint16_t, bool, bool, bool, bool, bool, bool, bool, bool)>;
using InsertKey =
    std::function<void(PrincipledBSDF*, uint16_t, bool, bool, bool, bool, bool, bool, bool, bool)>;

inline bool draw(PrincipledBSDF* pbsdf, uint16_t mat_id, const IsKeyed& isMatKeyed,
                 const InsertKey& insertMatPropKey) {
    bool changed = false;
    float albedo_arr[3] = {static_cast<float>(pbsdf->albedoProperty.color.x),
                           static_cast<float>(pbsdf->albedoProperty.color.y),
                           static_cast<float>(pbsdf->albedoProperty.color.z)};
    const bool keyed = isMatKeyed(mat_id, true, false, false, false, false, false, false, false);
    if (MaterialInspectorControls::color("Base Color", albedo_arr, keyed, [&]() {
            insertMatPropKey(pbsdf, mat_id, true, false, false, false, false, false, false, false);
        })) {
        pbsdf->albedoProperty.color = Vec3(albedo_arr[0], albedo_arr[1], albedo_arr[2]);
        changed = true;
    }

    float metallic = pbsdf->metallicProperty.intensity;
    bool metKeyed = isMatKeyed(mat_id, false, false, false, true, false, false, false, false);
    if (PropertyLayout::slider("met", "Metallic", &metallic, 0.0f, 1.0f, "%.3f", metKeyed, [&]() {
            insertMatPropKey(pbsdf, mat_id, false, false, false, true, false, false, false, false);
        })) {
        pbsdf->metallicProperty.intensity = metallic;
        changed = true;
    }

    float roughness = static_cast<float>(pbsdf->roughnessProperty.color.x);
    bool rghKeyed = isMatKeyed(mat_id, false, false, true, false, false, false, false, false);
    if (PropertyLayout::slider("rgh", "Roughness", &roughness, 0.0f, 1.0f, "%.3f", rghKeyed, [&]() {
            insertMatPropKey(pbsdf, mat_id, false, false, true, false, false, false, false, false);
        })) {
        pbsdf->roughnessProperty.color = Vec3(roughness);
        changed = true;
    }

    return changed;
}

} // namespace MaterialSurfaceBasicsUI
