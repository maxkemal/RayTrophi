#pragma once

#include "UI/MaterialSurfaceBasicsUI.h"
#include "UI/MaterialCoverageUI.h"
#include "UI/MaterialInspectorControls.h"
#include "ui_modern.h"

namespace MaterialInspectorSections {

inline bool drawTransparency(PrincipledBSDF* pbsdf, uint16_t mat_id,
                             const MaterialSurfaceBasicsUI::IsKeyed& isMatKeyed,
                             const MaterialSurfaceBasicsUI::InsertKey& insertMatPropKey) {
    bool changed = false;

    float ior = pbsdf->ior;
    bool iorKeyed = isMatKeyed(mat_id, false, false, false, false, false, false, true, false);
    if (PropertyLayout::slider("ior", "IOR", &ior, 1.0f, 3.0f, "%.3f", iorKeyed, [&]() {
            insertMatPropKey(pbsdf, mat_id, false, false, false, false, false, false, true, false);
        })) {
        pbsdf->ior = ior;
        changed = true;
    }

    float transmission = pbsdf->transmission;
    bool trnKeyed = isMatKeyed(mat_id, false, false, false, false, false, true, false, false);
    if (PropertyLayout::slider("trans", "Transmission", &transmission, 0.0f, 1.0f, "%.3f", trnKeyed,
                               [&]() {
                                   insertMatPropKey(pbsdf, mat_id, false, false, false, false,
                                                    false, true, false, false);
                               })) {
        pbsdf->setTransmission(transmission, pbsdf->ior);
        changed = true;
    }

    // Spectral dispersion: per-path stochastic wavelength-channel refraction.
    // Only meaningful with Transmission > 0 (prism / diamond colour fringing).
    float dispersion = pbsdf->dispersion;
    if (MaterialInspectorControls::sliderFloat("Dispersion##disp", &dispersion, 0.0f, 1.0f,
                                               "%.3f")) {
        pbsdf->dispersion = dispersion;
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Spectral dispersion (chromatic refraction): red/green/blue bend differently,\n"
            "producing rainbow fringing in glass (prism, diamond). Needs Transmission > 0.\n"
            "Adds colour noise that converges with more samples. 0 = off.");
    }

    // Thin-shell BUBBLE (champagne / soda / soap-foam close-up). Renders the
    // surface as a thin dielectric film (Fresnel rim + straight pass-through) —
    // assign this material to foam BUBBLES (foam panel) for the production look.
    bool is_bubble = pbsdf->getIsBubble();
    if (MaterialInspectorControls::checkbox("Bubble (thin shell)", &is_bubble)) {
        pbsdf->setIsBubble(is_bubble);
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Thin dielectric bubble film: bright Fresnel rim + transparent centre.\n"
            "For champagne/soda/soap close-ups. Assign to foam Bubbles in the foam panel.");
    }
    if (pbsdf->getIsBubble()) {

        float bubble_ior = pbsdf->getBubbleIor();
        if (MaterialInspectorControls::sliderFloat("Rim IOR##bubble", &bubble_ior, 1.0f, 2.0f,
                                                   "%.3f")) {
            pbsdf->setBubbleIor(bubble_ior);
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Fresnel strength of the bubble rim. ~1.33 = air/water interface.");
        }
        float bubble_film = pbsdf->getBubbleFilm();
        if (MaterialInspectorControls::sliderFloat("Thin-Film##bubble", &bubble_film, 0.0f, 30.0f,
                                                   "%.1f")) {
            pbsdf->setBubbleFilm(bubble_film);
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Soap-bubble iridescence: thin-film interference tints the rim by view angle.\n"
                "0 = clean silver rim. Higher = rainbow shimmer (thicker film / more cycles).");
        }
        // Thin-shell is a TYPE branch, not a lobe: both shaders dispatch it
        // and return before the resin coat and the transmission lobes are
        // ever considered. Silently winning over settings the user can still
        // see and drag reads as "resin does nothing" — say which one is
        // actually driving the surface.
        if (pbsdf->getTransmissionDensity() > 1e-4f || pbsdf->transmission > 0.01f) {
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                               "Thin shell OVERRIDES resin + transmission.");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("A bubble is shaded as a thin film and returns before the resin\n"
                                  "coat and the glass lobe run — Interior Depth and Trans have no\n"
                                  "effect while this is on. Uncheck it to author a resin coat.");
            }
        }
    }

    bool cutout = pbsdf->coverage.alphaCutout;
    if (MaterialInspectorControls::checkbox("Alpha Cutout", &cutout)) {
        const auto result =
            rtapi::setMaterialParamByName(MaterialManager::getInstance().getMaterialName(mat_id),
                                          "alpha_cutout", cutout ? 1.0f : 0.0f);
        if (!result.ok) {
            ImGui::TextWrapped("%s", result.error.c_str());
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Use alpha as a coverage mask (threshold 0.5). "
                          "Transmission remains independent.");
    }
    float opacity = pbsdf->opacityProperty.alpha;
    bool opKeyed = isMatKeyed(mat_id, false, true, false, false, false, false, false, false);
    if (MaterialInspectorControls::smartFloat(
            "opac", "Alpha", &opacity, 0.0f, 1.0f, "%.3f", opKeyed,
            [&]() {
                insertMatPropKey(pbsdf, mat_id, false, true, false, false, false, false, false,
                                 false);
            },
            16)) {
        pbsdf->opacityProperty.alpha = opacity;
        changed = true;
    }

    float trans = pbsdf->translucent;
    if (MaterialInspectorControls::smartFloat("trns", "Transl", &trans, 0.0f, 1.0f, "%.3f", false,
                                              nullptr, 12)) {
        pbsdf->translucent = trans;
        changed = true;
    }
    return changed;
}

} // namespace MaterialInspectorSections
