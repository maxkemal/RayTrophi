#pragma once

#include "UI/MaterialSurfaceBasicsUI.h"
#include "UI/MaterialCoverageUI.h"
#include "UI/MaterialInspectorControls.h"
#include "ui_modern.h"

namespace MaterialInspectorSections {

inline bool drawSurface(PrincipledBSDF* pbsdf, uint16_t mat_id,
                        const MaterialSurfaceBasicsUI::IsKeyed& isMatKeyed,
                        const MaterialSurfaceBasicsUI::InsertKey& insertMatPropKey) {
    bool changed = false;

    changed |= MaterialSurfaceBasicsUI::draw(pbsdf, mat_id, isMatKeyed, insertMatPropKey);
    float specular = pbsdf->specularProperty.intensity;
    bool specKeyed = isMatKeyed(mat_id, false, false, false, false, false, false, false, true);
    if (MaterialInspectorControls::smartFloat(
            "spec", "Spec", &specular, 0.0f, 1.0f, "%.3f", specKeyed,
            [&]() {
                insertMatPropKey(pbsdf, mat_id, false, false, false, false, false, false, false,
                                 true);
            },
            16)) {
        pbsdf->specularProperty.intensity = specular;
        changed = true;
    }

    return changed;
}

} // namespace MaterialInspectorSections
