#pragma once

#include "UI/MaterialSurfaceBasicsUI.h"
#include "UI/MaterialCoverageUI.h"
#include "UI/MaterialInspectorControls.h"
#include "ui_modern.h"

namespace MaterialInspectorSections {

inline bool drawSubsurface(PrincipledBSDF* pbsdf, uint16_t mat_id,
                           const MaterialSurfaceBasicsUI::IsKeyed& isMatKeyed,
                           const MaterialSurfaceBasicsUI::InsertKey& insertMatPropKey) {
    bool changed = false;

    float sss_amount = pbsdf->getSubsurface();
    if (MaterialInspectorControls::smartFloat("sss", "Subsurf", &sss_amount, 0.0f, 1.0f, "%.3f",
                                              false, nullptr, 12)) {
        pbsdf->setSubsurface(sss_amount);
        changed = true;
    }
    UIWidgets::HelpMarker("Amount of subsurface scattering (0=off, 1=full)");

    Vec3 sss_color = pbsdf->getSubsurfaceColor();
    float sss_color_arr[3] = {(float)sss_color.x, (float)sss_color.y, (float)sss_color.z};
    if (MaterialInspectorControls::colorEdit3("SSS Color", sss_color_arr)) {
        pbsdf->setSubsurfaceColor(Vec3(sss_color_arr[0], sss_color_arr[1], sss_color_arr[2]));
        changed = true;
    }

    Vec3 sss_radius = pbsdf->getSubsurfaceRadius();
    float sss_radius_arr[3] = {(float)sss_radius.x, (float)sss_radius.y, (float)sss_radius.z};
    if (MaterialInspectorControls::dragFloat3("Radius (RGB)", sss_radius_arr, 0.01f, 0.001f, 10.0f,
                                              "%.3f")) {
        pbsdf->setSubsurfaceRadius(Vec3(sss_radius_arr[0], sss_radius_arr[1], sss_radius_arr[2]));
        changed = true;
    }

    float sss_scale = pbsdf->getSubsurfaceScale();
    if (MaterialInspectorControls::smartFloat("sscl", "Scale", &sss_scale, 0.001f, 2.0f, "%.4f",
                                              false, nullptr, 12)) {
        pbsdf->setSubsurfaceScale(sss_scale);
        changed = true;
    }

    float sss_aniso = pbsdf->getSubsurfaceAnisotropy();
    if (MaterialInspectorControls::smartFloat("ssani", "Aniso", &sss_aniso, -0.9f, 0.9f, "%.2f",
                                              false, nullptr, 12)) {
        pbsdf->setSubsurfaceAnisotropy(sss_aniso);
        changed = true;
    }

    // SSS Index of Refraction (controls internal refraction/exit behavior)
    float sss_ior = pbsdf->getSubsurfaceIOR();
    if (MaterialInspectorControls::smartFloat("ssior", "SSS IOR", &sss_ior, 1.0f, 3.0f, "%.3f",
                                              false, nullptr, 12)) {
        pbsdf->setSubsurfaceIOR(sss_ior);
        changed = true;
    }

    // Method + walk cap. Both reach every backend (Vulkan RT via the
    // material ext record, OptiX via its legacy fields).
    int sss_method = pbsdf->getSssMethod();
    const char* sss_methods[] = {"Random Walk", "Fast"};
    if (MaterialInspectorControls::combo("SSS Method", &sss_method, sss_methods, 2)) {
        pbsdf->setSssMethod(sss_method);
        changed = true;
    }
    UIWidgets::HelpMarker(
        "Random Walk: geometry-aware, light bleeds through thin parts. "
        "Fast: same colour, no bleed, near-zero cost (use on heavy scenes / TDR).");
    ImGui::BeginDisabled(sss_method != 0);
    int sss_steps = pbsdf->getSssWalkMaxSteps();
    if (MaterialInspectorControls::sliderInt("Walk Max Steps", &sss_steps, 8, 256)) {
        pbsdf->setSssWalkMaxSteps(sss_steps);
        changed = true;
    }
    ImGui::EndDisabled();
    UIWidgets::HelpMarker("Hard cap on scatter events per walk. Too low DARKENS the result "
                          "(cut walks lose their energy); 64 is enough for most materials.");

    ImGui::Separator();
    ImGui::Text("Presets:");
    // Calibrated for the random-walk remap (2026-09-24): SSS Color is the
    // colour the surface SETTLES to (multi-scatter albedo), no longer a
    // per-event albedo that compounds darker with every scatter. Radius is
    // the per-channel mean free path in scene units, times Scale.
    if (ImGui::SmallButton("Skin")) {
        pbsdf->setSubsurface(1.0f);
        pbsdf->setSubsurfaceColor(Vec3(0.80f, 0.52f, 0.42f));
        pbsdf->setSubsurfaceRadius(Vec3(1.0f, 0.2f, 0.1f));
        pbsdf->setSubsurfaceScale(0.05f);
        pbsdf->setSubsurfaceAnisotropy(0.0f);
        pbsdf->setSubsurfaceIOR(1.4f);
        changed = true;
    }
    MaterialInspectorControls::nextPreset();
    if (ImGui::SmallButton("Wax")) {
        pbsdf->setSubsurface(1.0f);
        pbsdf->setSubsurfaceColor(Vec3(0.85f, 0.72f, 0.45f));
        pbsdf->setSubsurfaceRadius(Vec3(1.0f, 0.8f, 0.5f));
        pbsdf->setSubsurfaceScale(0.03f);
        pbsdf->setSubsurfaceAnisotropy(0.0f);
        pbsdf->setSubsurfaceIOR(1.45f);
        changed = true;
    }
    MaterialInspectorControls::nextPreset();
    if (ImGui::SmallButton("Milk")) {
        pbsdf->setSubsurface(1.0f);
        pbsdf->setSubsurfaceColor(Vec3(0.92f, 0.90f, 0.85f));
        pbsdf->setSubsurfaceRadius(Vec3(1.0f, 1.0f, 0.9f));
        pbsdf->setSubsurfaceScale(0.1f);
        pbsdf->setSubsurfaceAnisotropy(0.3f);
        pbsdf->setSubsurfaceIOR(1.35f);
        changed = true;
    }
    MaterialInspectorControls::nextPreset();
    if (ImGui::SmallButton("Jade")) {
        pbsdf->setSubsurface(1.0f);
        pbsdf->setSubsurfaceColor(Vec3(0.35f, 0.65f, 0.40f));
        pbsdf->setSubsurfaceRadius(Vec3(0.3f, 1.0f, 0.4f));
        pbsdf->setSubsurfaceScale(0.02f);
        pbsdf->setSubsurfaceAnisotropy(0.3f);
        pbsdf->setSubsurfaceIOR(1.6f);
        changed = true;
    }

    return changed;
}

} // namespace MaterialInspectorSections
