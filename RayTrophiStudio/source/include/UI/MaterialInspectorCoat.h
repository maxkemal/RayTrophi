#pragma once

#include "UI/MaterialSurfaceBasicsUI.h"
#include "UI/MaterialCoverageUI.h"
#include "UI/MaterialInspectorControls.h"
#include "ui_modern.h"

namespace MaterialInspectorSections {

inline bool drawCoat(PrincipledBSDF* pbsdf, uint16_t mat_id,
                     const MaterialSurfaceBasicsUI::IsKeyed& isMatKeyed,
                     const MaterialSurfaceBasicsUI::InsertKey& insertMatPropKey) {
    bool changed = false;

    float cc_amount = pbsdf->getClearcoat();
    if (MaterialInspectorControls::smartFloat("cc", "ClearCt", &cc_amount, 0.0f, 1.0f, "%.3f",
                                              false, nullptr, 12)) {
        pbsdf->setClearcoat(cc_amount, pbsdf->getClearcoatRoughness());
        changed = true;
    }

    float cc_roughness = pbsdf->getClearcoatRoughness();
    if (MaterialInspectorControls::smartFloat("ccr", "CCRough", &cc_roughness, 0.0f, 1.0f, "%.3f",
                                              false, nullptr, 12)) {
        pbsdf->setClearcoat(pbsdf->getClearcoat(), cc_roughness);
        changed = true;
    }

    // Iridescent clearcoat (thin-film tint on the clearcoat lobe): oil-slick, beetle
    // shell, candy paint. 0 = plain white clearcoat. Thickness drives the hue cycle.
    float cc_irid = pbsdf->getClearcoatIridescence();
    if (MaterialInspectorControls::smartFloat("cci", "Irides", &cc_irid, 0.0f, 1.0f, "%.3f", false,
                                              nullptr, 12)) {
        pbsdf->setClearcoatIridescence(cc_irid);
        changed = true;
    }
    if (pbsdf->getClearcoatIridescence() > 0.001f) {
        float cc_thick = pbsdf->getClearcoatFilmThickness();
        if (MaterialInspectorControls::smartFloat("ccft", "FilmThk", &cc_thick, 0.1f, 2.0f, "%.3f",
                                                  false, nullptr, 12)) {
            pbsdf->setClearcoatFilmThickness(cc_thick);
            changed = true;
        }
    }

    ImGui::Separator();
    if (ImGui::SmallButton("Car Paint")) {
        pbsdf->setClearcoat(1.0f, 0.03f);
        changed = true;
    }
    MaterialInspectorControls::nextPreset();
    if (ImGui::SmallButton("Oil Slick")) {
        pbsdf->setClearcoat(1.0f, 0.04f);
        pbsdf->setClearcoatIridescence(0.85f);
        pbsdf->setClearcoatFilmThickness(0.55f);
        changed = true;
    }
    return changed;
}

} // namespace MaterialInspectorSections
