#pragma once

#include "UI/MaterialSurfaceBasicsUI.h"
#include "UI/MaterialCoverageUI.h"
#include "UI/MaterialInspectorControls.h"
#include "ui_modern.h"

namespace MaterialInspectorSections {

inline bool drawEmission(PrincipledBSDF* pbsdf, uint16_t mat_id,
                         const MaterialSurfaceBasicsUI::IsKeyed& isMatKeyed,
                         const MaterialSurfaceBasicsUI::InsertKey& insertMatPropKey) {
    bool changed = false;
    Vec3 emission = pbsdf->emissionProperty.color;
    float emission_arr[3] = {(float)emission.x, (float)emission.y, (float)emission.z};
    const bool keyed = isMatKeyed(mat_id, false, false, false, false, true, false, false, false);
    if (MaterialInspectorControls::color("Emission", emission_arr, keyed, [&]() {
            insertMatPropKey(pbsdf, mat_id, false, false, false, false, true, false, false, false);
        })) {
        pbsdf->emissionProperty.color = Vec3(emission_arr[0], emission_arr[1], emission_arr[2]);
        changed = true;
    }

    float emissionStr = pbsdf->emissionProperty.intensity;
    if (MaterialInspectorControls::smartFloat("mems", "EmStr", &emissionStr, 0.0f, 1000.0f, "%.1f",
                                              false, nullptr, 12)) {
        pbsdf->emissionProperty.intensity = emissionStr;
        changed = true;
    }
    return changed;
}

} // namespace MaterialInspectorSections
