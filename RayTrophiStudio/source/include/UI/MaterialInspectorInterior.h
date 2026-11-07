#pragma once

#include "UI/MaterialSurfaceBasicsUI.h"
#include "UI/MaterialCoverageUI.h"
#include "UI/MaterialInspectorControls.h"
#include "ui_modern.h"

namespace MaterialInspectorSections {

inline bool drawInterior(PrincipledBSDF* pbsdf, uint16_t mat_id,
                         const MaterialSurfaceBasicsUI::IsKeyed& isMatKeyed,
                         const MaterialSurfaceBasicsUI::InsertKey& insertMatPropKey) {
    bool changed = false;

    const float transmission = pbsdf->transmission;
    // Thick-resin / glass-marble depth. Self-contained: when > 0 the material
    // becomes a transmissive resin body (no need to raise Transmission too) and
    // absorbs colour over the REAL distance light travels inside it — deep centre,
    // clear thin edges. Albedo = the absorption tint. 0 = off (plain material).
    // INTERIOR VOLUME — the material's embedded medium: depth absorption
    // (resin/thick glass), nebula dust, dirt specks, bubbles and coloured
    // glass shards. "Resin" survives as the opaque-base coat mode's name;
    // the structure itself outgrew it.
    float trans_density = pbsdf->getTransmissionDensity();
    if (MaterialInspectorControls::sliderFloat("Interior Depth##transdens", &trans_density, 0.0f,
                                               8.0f, "%.2f")) {
        pbsdf->setTransmissionDensity(trans_density);
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Interior Volume: depth absorption. With Transmission <= 0.5 this is a resin\n"
            "body over an OPAQUE base. With Transmission > 0.5 it becomes a TRANSLUCENT\n"
            "STONE (amber/jade): see-through, colour deepens with the real distance light\n"
            "travels inside (thick centre, clear edges) and the material KEEPS casting\n"
            "caustics. Interior Tint drives which wavelengths survive the depth.");
    }
    if (pbsdf->getTransmissionDensity() > 0.001f) {

        float resinCol[3] = {(float)pbsdf->getResinColor().x, (float)pbsdf->getResinColor().y,
                             (float)pbsdf->getResinColor().z};
        if (MaterialInspectorControls::colorEdit3("Interior Tint##resincol", resinCol)) {
            pbsdf->setResinColor(Vec3(resinCol[0], resinCol[1], resinCol[2]));
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("The colour the interior builds with depth (separate from albedo).\n"
                              "White = clear (lensing only). Saturated = stronger, richer depth.");
        }
        float resin_roughness = pbsdf->getResinRoughness();
        if (MaterialInspectorControls::sliderFloat("Coat Gloss##resinrough", &resin_roughness, 0.0f,
                                                   1.0f, "%.3f")) {
            pbsdf->setResinRoughness(resin_roughness);
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Roughness of the coat's reflection (the glossy top layer),\n"
                              "independent of the base material roughness underneath.\n"
                              "0 = mirror-sharp clear coat, 1 = matte/satin coat.");
        }
    }

    // Internal inclusions — a shared procedural march used by TWO modes:
    //   - Resin (Resin Depth > 0): opaque base under the coat.
    //   - Glass Marble (base Transmission high, no Resin Depth): real see-through
    //     glass with volumetric dust/dirt inside (independent of the resin coat).
    // Mode split mirrors the shader: Interior Depth + Trans > 0.5 =
    // TRANSLUCENT STONE (amber/jade: see-through + real-distance depth
    // absorption, stays a caustic caster); Trans <= 0.5 = resin coat over
    // an opaque base; Trans > 0.5 without depth = clear glass marble.
    bool stoneMode = pbsdf->getTransmissionDensity() > 0.001f && transmission > 0.5f;
    bool resinMode = pbsdf->getTransmissionDensity() > 0.001f && transmission <= 0.5f;
    bool glassMarbleMode = (transmission > 0.5f) && !stoneMode;
    if (resinMode || glassMarbleMode || stoneMode) {

        ImGui::Separator();
        ImGui::TextDisabled(
            stoneMode
                ? "Interior Volume (Translucent Stone \xE2\x80\x94 amber/jade, casts caustics)"
            : glassMarbleMode ? "Interior Volume (Glass Marble \xE2\x80\x94 see-through)"
                              : "Interior Volume (resin \xE2\x80\x94 opaque base)");
        // Curated interior presets — one click sets the whole inclusion
        // stack; every value remains hand-editable afterwards.
        {
            struct ResinPreset {
                const char* name;
                float dust, dirt, shard, shardHue, scale;
                float dirtCol[3];
                int dustStyle, shardShape;
                float dustA[3], dustB[3];
            };
            static const ResinPreset kResinPresets[] = {
                {"Clear",
                 0.00f,
                 0.00f,
                 0.00f,
                 -1.0f,
                 8.0f,
                 {0.18f, 0.14f, 0.10f},
                 0,
                 0,
                 {1, 1, 1},
                 {1, 1, 1}},
                {"Dusty Amber",
                 0.45f,
                 0.25f,
                 0.00f,
                 -1.0f,
                 6.0f,
                 {0.30f, 0.20f, 0.10f},
                 0,
                 0,
                 {1, 1, 1},
                 {1, 1, 1}},
                {"Nebula",
                 0.70f,
                 0.10f,
                 0.15f,
                 0.65f,
                 3.5f,
                 {0.10f, 0.08f, 0.14f},
                 0,
                 0,
                 {1, 1, 1},
                 {1, 1, 1}},
                {"Galaxy Marble",
                 0.35f,
                 0.20f,
                 0.55f,
                 -1.0f,
                 5.0f,
                 {0.05f, 0.05f, 0.08f},
                 1,
                 0,
                 {0.15f, 0.10f, 0.45f},
                 {0.95f, 0.55f, 0.25f}},
                {"Cat's Eye",
                 0.15f,
                 0.05f,
                 0.65f,
                 0.33f,
                 2.5f,
                 {0.10f, 0.10f, 0.10f},
                 0,
                 0,
                 {1, 1, 1},
                 {1, 1, 1}},
                {"Crystal Geode",
                 0.20f,
                 0.10f,
                 0.80f,
                 0.72f,
                 3.0f,
                 {0.10f, 0.08f, 0.14f},
                 1,
                 1,
                 {0.30f, 0.20f, 0.50f},
                 {0.85f, 0.80f, 0.95f}},
                {"Ink in Water",
                 0.85f,
                 0.00f,
                 0.00f,
                 -1.0f,
                 2.2f,
                 {0.10f, 0.10f, 0.10f},
                 3,
                 0,
                 {0.05f, 0.15f, 0.55f},
                 {0.92f, 0.92f, 0.98f}},
                {"Riverbed Epoxy",
                 0.30f,
                 0.55f,
                 0.10f,
                 0.55f,
                 10.0f,
                 {0.16f, 0.12f, 0.08f},
                 2,
                 0,
                 {0.45f, 0.38f, 0.25f},
                 {0.20f, 0.28f, 0.18f}},
                {"Champagne",
                 0.10f,
                 0.35f,
                 0.00f,
                 -1.0f,
                 14.0f,
                 {0.85f, 0.80f, 0.65f},
                 0,
                 0,
                 {1, 1, 1},
                 {1, 1, 1}},
            };
            int presetIdx = -1;
            const char* names[IM_ARRAYSIZE(kResinPresets)];
            for (int i = 0; i < IM_ARRAYSIZE(kResinPresets); ++i) {
                names[i] = kResinPresets[i].name;
            }
            if (MaterialInspectorControls::combo("Interior Preset##resinpreset", &presetIdx, names,
                                                 IM_ARRAYSIZE(kResinPresets))) {
                const ResinPreset& preset = kResinPresets[presetIdx];
                pbsdf->setResinInclusion(preset.dust);
                pbsdf->setResinDirt(preset.dirt);
                pbsdf->setResinShard(preset.shard);
                pbsdf->setResinShardHue(preset.shardHue);
                pbsdf->setResinInclusionScale(preset.scale);
                pbsdf->setResinDirtColor(
                    Vec3(preset.dirtCol[0], preset.dirtCol[1], preset.dirtCol[2]));
                pbsdf->setDustStyle(preset.dustStyle);
                pbsdf->setShardShape(preset.shardShape);
                pbsdf->setDustColorA(Vec3(preset.dustA[0], preset.dustA[1], preset.dustA[2]));
                pbsdf->setDustColorB(Vec3(preset.dustB[0], preset.dustB[1], preset.dustB[2]));
                changed = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Curated interior looks (dust + specks + shards).\n"
                                  "Applies once; all sliders stay editable.");
            }
        }
        float resin_inclusion = pbsdf->getResinInclusion();
        if (MaterialInspectorControls::sliderFloat("Dust##resindust", &resin_inclusion, 0.0f, 1.0f,
                                                   "%.3f")) {
            pbsdf->setResinInclusion(resin_inclusion);
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Cloudy dust suspended inside — heterogeneous absorption that\n"
                              "thickens at depth (amorphous, varies in 3D). 0 = perfectly clear.");
        }
        if (pbsdf->getResinInclusion() > 0.001f) {

            int dustStyle = pbsdf->getDustStyle();
            const char* kDustStyles[] = {"Nebula (auto tint)", "Billow 2-Color", "Wispy Streaks",
                                         "Paint Swirl"};

            if (MaterialInspectorControls::combo("Dust Style##duststyle", &dustStyle, kDustStyles,
                                                 4)) {
                pbsdf->setDustStyle(dustStyle);
                changed = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Nebula: legacy auto tint derived from the interior colour.\n"
                    "Billow: puffy clouds mixing your two colours.\n"
                    "Wispy: stretched horizontal filaments.\n"
                    "Paint Swirl: domain-warped ink-in-water curls \xE2\x80\x94 the two\n"
                    "pigments fold into each other like stirred paint.");
            }
            if (pbsdf->getDustStyle() > 0) {
                float dA[3] = {(float)pbsdf->getDustColorA().x, (float)pbsdf->getDustColorA().y,
                               (float)pbsdf->getDustColorA().z};
                if (MaterialInspectorControls::colorEdit3("Dust Color A##dustcola", dA)) {
                    pbsdf->setDustColorA(Vec3(dA[0], dA[1], dA[2]));
                    changed = true;
                }
                float dB[3] = {(float)pbsdf->getDustColorB().x, (float)pbsdf->getDustColorB().y,
                               (float)pbsdf->getDustColorB().z};
                if (MaterialInspectorControls::colorEdit3("Dust Color B##dustcolb", dB)) {
                    pbsdf->setDustColorB(Vec3(dB[0], dB[1], dB[2]));
                    changed = true;
                }
            }
        }
        float resin_dirt = pbsdf->getResinDirt();
        if (MaterialInspectorControls::sliderFloat("Dirt Specks##resindirt", &resin_dirt, 0.0f,
                                                   1.0f, "%.3f")) {
            pbsdf->setResinDirt(resin_dirt);
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Opaque specks (dirt/debris) embedded at random depths. A ray that\n"
                "hits one terminates early — its colour shows through the medium crossed.");
        }
        if (pbsdf->getResinDirt() > 0.001f) {

            float dirtCol[3] = {(float)pbsdf->getResinDirtColor().x,
                                (float)pbsdf->getResinDirtColor().y,
                                (float)pbsdf->getResinDirtColor().z};
            if (MaterialInspectorControls::colorEdit3("Dirt Color##resindirtcol", dirtCol)) {
                pbsdf->setResinDirtColor(Vec3(dirtCol[0], dirtCol[1], dirtCol[2]));
                changed = true;
            }
        }
        float resin_shard = pbsdf->getResinShard();
        if (MaterialInspectorControls::sliderFloat("Glass Shards##resinshard", &resin_shard, 0.0f,
                                                   1.0f, "%.3f")) {
            pbsdf->setResinShard(resin_shard);
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Colored translucent glass chips suspended inside (faceted, gem-like).\n"
                "They tint light passing through like stained glass — the classic\n"
                "geometric colours of a toy marble.");
        }
        if (pbsdf->getResinShard() > 0.001f) {

            int shardShape = pbsdf->getShardShape();
            const char* kShardShapes[] = {"Round Chips", "Faceted Crystals"};

            if (MaterialInspectorControls::combo("Shard Shape##shardshape", &shardShape,
                                                 kShardShapes, 2)) {
                pbsdf->setShardShape(shardShape);
                changed = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Round Chips: soft translucent colour blobs.\n"
                                  "Faceted Crystals: elongated shards with flat cut faces that\n"
                                  "flash as the light or the object turns.");
            }
            float shardHue = pbsdf->getResinShardHue();
            bool rainbow = shardHue < 0.0f;
            if (MaterialInspectorControls::checkbox("Rainbow##resinshardrainbow", &rainbow)) {
                pbsdf->setResinShardHue(rainbow ? -1.0f : 0.33f);
                changed = true;
            }
            if (!rainbow) {
                shardHue = pbsdf->getResinShardHue();
                if (MaterialInspectorControls::sliderFloat("Shard Hue##resinshardhue", &shardHue,
                                                           0.0f, 1.0f, "%.2f")) {
                    pbsdf->setResinShardHue(shardHue);
                    changed = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(
                        "Base hue of the shards (0 = red, 0.33 = green, 0.66 = blue);\n"
                        "each shard varies slightly around it. Rainbow = full palette.");
                }
            }
        }
        if (pbsdf->getResinInclusion() > 0.001f || pbsdf->getResinDirt() > 0.001f ||
            pbsdf->getResinShard() > 0.001f) {
            float resin_inclusion_scale = pbsdf->getResinInclusionScale();
            if (MaterialInspectorControls::sliderFloat("Inclusion Scale##resininclscale",
                                                       &resin_inclusion_scale, 1.0f, 64.0f,
                                                       "%.1f")) {
                pbsdf->setResinInclusionScale(resin_inclusion_scale);
                changed = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Feature size of the procedural dust/dirt field.\n"
                                  "Low = large blobs/clouds, high = fine grain.");
            }
            int anchorIdx = pbsdf->getResinObjectSpace() ? 0 : 1;
            const char* kAnchors[] = {"Object (moves with it)", "World (fixed in space)"};

            if (MaterialInspectorControls::combo("Anchor##resinanchor", &anchorIdx, kAnchors, 2)) {
                pbsdf->setResinObjectSpace(anchorIdx == 0);
                changed = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Object: the interior pattern travels/rotates with the mesh (usual).\n"
                    "World: the pattern stays fixed in space \xE2\x80\x94 a moving object appears\n"
                    "to pass THROUGH a frozen medium (deliberate effect).");
            }
        }
        if (glassMarbleMode || stoneMode) {
            // Full Volume (real-interior medium march) was retired — too camera-angle
            // dependent and the interior dust/dirt never read as intended. Inclusion glass
            // uses the front-shell approximation only. Keep the flag dormant/off.
            pbsdf->setGlassMarbleVolume(false);
            ImGui::TextDisabled(
                stoneMode
                    ? "Translucent Stone: see-through + depth absorption; remains a caustic caster."
                    : "Glass Marble: light passes through (see-through) + shell inclusions.");
        }
    }

    return changed;
}

} // namespace MaterialInspectorSections
