#pragma once

// ═══════════════════════════════════════════════════════════════════════════════
// ROAD ASSIGNMENT PANEL - contextual dock section for per-spline road authoring
// ═══════════════════════════════════════════════════════════════════════════════
// Surfaces every field that script/IPC can write, closing the panel debt:
//   assign/clear profile, crossing mode, enabled, carve override, effective readback.
// All writes go through rtapi::* — the SAME core logic script and IPC use.
// Validation is the SAME validateRoadCarveSettings the solver runs.

#include "Api/RtApi.h"
#include "TerrainRoadNetwork.h"
#include "TerrainRoadProfile.h"
#include "ProjectManager.h"
#include "ui_modern.h"
#include "imgui.h"
#include "scene_ui.h"     // UIContext

#include <string>

namespace {

// ─── helpers ────────────────────────────────────────────────────────────────

inline void roadPanelError(const char* msg) {
    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", msg);
}

inline void roadPanelReadonlyFloat(const char* label, float value, const char* unit) {
    ImGui::TextDisabled("  %s: %.2f %s", label, value, unit);
}

// ─── main entry point ───────────────────────────────────────────────────────

inline void drawRoadAssignmentPanel(UIContext& ctx) {
    const auto& splineObj = ctx.selection.selected.spline_object;
    if (!splineObj) return;

    const std::string& splineName = splineObj->nodeName;
    if (splineName.empty()) return;

    UIWidgets::Divider();

    if (!UIWidgets::BeginSection("Road Assignment",
                                  ImVec4(0.95f, 0.70f, 0.30f, 1.0f), true)) {
        return;
    }

    // ── look up current assignment ──────────────────────────────────────────
    const auto& registry = TerrainNodesV2::RoadNetworkRegistry::getInstance();
    const TerrainNodesV2::RoadAssignment* assignment = registry.find(splineName);

    // ── no assignment: offer "Assign Profile" ───────────────────────────────
    if (!assignment) {
        ImGui::TextDisabled("No road profile assigned.");

        // Profile picker
        static int s_profileChoice = 1; // default to dirt_road
        const auto profiles = TerrainNodesV2::builtinRoadProfiles();
        if (ImGui::BeginCombo("Profile##RoadAssign", profiles[s_profileChoice].displayName.c_str())) {
            for (int i = 0; i < static_cast<int>(profiles.size()); ++i) {
                const bool selected = (i == s_profileChoice);
                if (ImGui::Selectable(profiles[i].displayName.c_str(), selected))
                    s_profileChoice = i;
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (UIWidgets::PrimaryButton("Assign Profile", ImVec2(-1, 0))) {
            auto result = rtapi::assignRoadProfile(splineName, profiles[s_profileChoice].id);
            if (!result) {
                // Will be shown on next frame via the error path below
            }
        }

        UIWidgets::EndSection();
        return;
    }

    // ── assignment exists: show controls ────────────────────────────────────
    // --- Profile combo ---
    {
        const auto profiles = TerrainNodesV2::builtinRoadProfiles();
        int currentIdx = -1;
        for (int i = 0; i < static_cast<int>(profiles.size()); ++i) {
            if (profiles[i].id == assignment->profileId) { currentIdx = i; break; }
        }
        const char* preview = (currentIdx >= 0)
            ? profiles[currentIdx].displayName.c_str()
            : assignment->profileId.c_str();

        if (ImGui::BeginCombo("Profile##RoadEdit", preview)) {
            for (int i = 0; i < static_cast<int>(profiles.size()); ++i) {
                const bool selected = (i == currentIdx);
                if (ImGui::Selectable(profiles[i].displayName.c_str(), selected)) {
                    rtapi::assignRoadProfile(splineName, profiles[i].id);
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    // --- Crossing mode combo ---
    {
        static const char* modeNames[] = { "Auto", "Terrain", "Bridge", "Ford", "Tunnel" };
        static const char* modeKeys[]  = { "auto", "terrain", "bridge", "ford", "tunnel" };
        int currentMode = static_cast<int>(assignment->crossingMode);
        if (currentMode < 0 || currentMode > 4) currentMode = 0;

        if (ImGui::BeginCombo("Crossing Mode##Road", modeNames[currentMode])) {
            for (int i = 0; i < 5; ++i) {
                const bool selected = (i == currentMode);
                if (ImGui::Selectable(modeNames[i], selected)) {
                    rtapi::setRoadCrossingMode(splineName, modeKeys[i]);
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    // --- Enabled checkbox ---
    {
        bool enabled = assignment->enabled;
        if (ImGui::Checkbox("Enabled##Road", &enabled)) {
            rtapi::setRoadEnabled(splineName, enabled);
        }
    }

    ImGui::Spacing();

    // --- Carve Settings ---
    {
        static std::string s_lastSpline;
        static TerrainNodesV2::RoadCarveSettings s_overrideEdit;
        static std::string s_errorMsg;

        if (s_lastSpline != splineName) {
            s_lastSpline = splineName;
            s_overrideEdit = assignment->effectiveCarve();
            s_errorMsg.clear();
        }

        if (ImGui::TreeNodeEx("Carve Settings##Road", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed)) {
            
            bool isOverride = assignment->hasOverride;
            if (ImGui::Checkbox("Override Profile Defaults", &isOverride)) {
                if (isOverride) {
                    s_overrideEdit = assignment->effectiveCarve();
                    rtapi::RoadCarveValues apiValues;
                    apiValues.road_width       = s_overrideEdit.roadWidthMeters;
                    apiValues.shoulder_width   = s_overrideEdit.shoulderWidthMeters;
                    apiValues.grading_falloff  = s_overrideEdit.gradingFalloffMeters;
                    apiValues.foliage_margin   = s_overrideEdit.foliageExclusionMarginMeters;
                    apiValues.max_grade_percent = s_overrideEdit.maxGradePercent;
                    apiValues.elevation_offset = s_overrideEdit.elevationOffsetMeters;
                    apiValues.max_cut_meters   = s_overrideEdit.maxCutMeters;
                    apiValues.max_fill_meters  = s_overrideEdit.maxFillMeters;
                    apiValues.crown_meters     = s_overrideEdit.crownMeters;
                    apiValues.ditch_width      = s_overrideEdit.ditchWidthMeters;
                    apiValues.ditch_depth      = s_overrideEdit.ditchDepthMeters;
                    apiValues.use_point_width  = s_overrideEdit.usePointWidth;
                    auto result = rtapi::setRoadCarveOverride(splineName, apiValues);
                    if (!result) s_errorMsg = result.error;
                    else s_errorMsg.clear();
                } else {
                    auto result = rtapi::clearRoadCarveOverride(splineName);
                    if (!result) s_errorMsg = result.error;
                    else {
                        s_errorMsg.clear();
                        s_overrideEdit = assignment->effectiveCarve();
                    }
                }
            }

            ImGui::Spacing();

            if (!isOverride) {
                ImGui::BeginDisabled();
                // Keep the local edit state strictly synchronized with the profile
                // when not overriding, so the sliders reflect upstream profile edits.
                s_overrideEdit = assignment->effectiveCarve();
            }

            bool needsApply = false;
            
            auto sliderFloat = [&](const char* label, float* v, float v_min, float v_max, const char* format) {
                ImGui::SliderFloat(label, v, v_min, v_max, format);
                if (ImGui::IsItemDeactivatedAfterEdit()) needsApply = true;
            };
            auto dragFloat = [&](const char* label, float* v, float speed, float v_min, float v_max, const char* format) {
                ImGui::DragFloat(label, v, speed, v_min, v_max, format);
                if (ImGui::IsItemDeactivatedAfterEdit()) needsApply = true;
            };

            sliderFloat("Road Width (m)",      &s_overrideEdit.roadWidthMeters, 0.5f, 20.0f, "%.2f");
            sliderFloat("Shoulder (m)",        &s_overrideEdit.shoulderWidthMeters, 0.0f, 10.0f, "%.2f");
            sliderFloat("Grading Falloff (m)", &s_overrideEdit.gradingFalloffMeters, 0.0f, 20.0f, "%.2f");
            sliderFloat("Foliage Margin (m)",  &s_overrideEdit.foliageExclusionMarginMeters, 0.0f, 10.0f, "%.2f");
            sliderFloat("Max Grade (%)",       &s_overrideEdit.maxGradePercent, 1.0f, 50.0f, "%.1f");
            dragFloat("Elev. Offset (m)",      &s_overrideEdit.elevationOffsetMeters, 0.1f, -50.0f, 50.0f, "%.2f");
            sliderFloat("Max Cut (m)",         &s_overrideEdit.maxCutMeters, 0.0f, 30.0f, "%.1f");
            sliderFloat("Max Fill (m)",        &s_overrideEdit.maxFillMeters, 0.0f, 30.0f, "%.1f");
            // Crown and ditch are the cross-section that keeps a carved road
            // from reading as a river bed: the surface sheds, the ditch carries.
            sliderFloat("Crown (m)",           &s_overrideEdit.crownMeters, 0.0f, 0.6f, "%.3f");
            sliderFloat("Ditch Width (m)",     &s_overrideEdit.ditchWidthMeters, 0.0f, 6.0f, "%.2f");
            sliderFloat("Ditch Depth (m)",     &s_overrideEdit.ditchDepthMeters, 0.0f, 3.0f, "%.2f");
            if (ImGui::Checkbox("Use Point Width", &s_overrideEdit.usePointWidth)) {
                needsApply = true;
            }

            if (!isOverride) {
                ImGui::EndDisabled();
            }

            if (needsApply && isOverride) {
                rtapi::RoadCarveValues apiValues;
                apiValues.road_width       = s_overrideEdit.roadWidthMeters;
                apiValues.shoulder_width   = s_overrideEdit.shoulderWidthMeters;
                apiValues.grading_falloff  = s_overrideEdit.gradingFalloffMeters;
                apiValues.foliage_margin   = s_overrideEdit.foliageExclusionMarginMeters;
                apiValues.max_grade_percent = s_overrideEdit.maxGradePercent;
                apiValues.elevation_offset = s_overrideEdit.elevationOffsetMeters;
                apiValues.max_cut_meters   = s_overrideEdit.maxCutMeters;
                apiValues.max_fill_meters  = s_overrideEdit.maxFillMeters;
                apiValues.crown_meters     = s_overrideEdit.crownMeters;
                apiValues.ditch_width      = s_overrideEdit.ditchWidthMeters;
                apiValues.ditch_depth      = s_overrideEdit.ditchDepthMeters;
                apiValues.use_point_width  = s_overrideEdit.usePointWidth;

                auto result = rtapi::setRoadCarveOverride(splineName, apiValues);
                if (!result) s_errorMsg = result.error;
                else s_errorMsg.clear();
            }

            if (!s_errorMsg.empty()) {
                roadPanelError(s_errorMsg.c_str());
            }

            ImGui::TreePop();
        }
    }

    ImGui::Spacing();

    // --- Road surface mesh (optional, terrain-only road stays valid) ---
    {
        static std::string s_meshStatus;
        static std::string s_meshStatusSpline;
        if (s_meshStatusSpline != splineName) {
            s_meshStatusSpline = splineName;
            s_meshStatus.clear();
        }

        if (ImGui::TreeNodeEx("Surface Mesh##Road", ImGuiTreeNodeFlags_Framed)) {
            static bool s_includeShoulder = true;
            static float s_surfaceOffset = 0.05f;
            static float s_uvTile = 4.0f;
            ImGui::Checkbox("Include Shoulder##RoadMesh", &s_includeShoulder);
            ImGui::SliderFloat("Surface Offset (m)##RoadMesh", &s_surfaceOffset, 0.0f, 0.5f, "%.3f");
            ImGui::SliderFloat("UV Metres/Tile##RoadMesh", &s_uvTile, 0.5f, 40.0f, "%.1f");

            if (assignment->meshObject.empty()) {
                ImGui::TextDisabled("No generated surface");
            } else {
                ImGui::TextDisabled("Owns: %s", assignment->meshObject.c_str());
            }

            if (UIWidgets::PrimaryButton(assignment->meshObject.empty()
                                             ? "Build Road Surface" : "Rebuild Road Surface",
                                         ImVec2(-1, 0))) {
                rtapi::RoadMeshOptions options;
                options.include_shoulder = s_includeShoulder;
                options.surface_offset = s_surfaceOffset;
                options.uv_meters_per_tile = s_uvTile;
                rtapi::RoadMeshInfo info;
                const auto result = rtapi::buildRoadMesh(splineName, options, info);
                s_meshStatus = result
                    ? info.object_name + ": " + std::to_string(info.triangle_count) +
                      " tris, " + std::to_string(info.span_count) + " span(s)"
                    : result.error;
            }
            if (!assignment->meshObject.empty() &&
                UIWidgets::DangerButton("Delete Road Surface", ImVec2(-1, 0))) {
                const auto result = rtapi::clearRoadMesh(splineName);
                s_meshStatus = result ? std::string("surface removed") : result.error;
            }

            // The route measurement lives beside the button that consumes it: a
            // bridge span that exists in the solve but not in the mesh is the
            // difference these two numbers make visible.
            rtapi::RoadRouteInfo route;
            if (rtapi::getRoadRoute(splineName, 0, route)) {
                ImGui::TextDisabled("Route %.1f m, %d sample(s)",
                                    route.length_meters, route.sample_count);
                if (route.bridge_samples || route.ford_samples || route.tunnel_samples) {
                    ImGui::TextDisabled("Crossings: %d bridge / %d ford / %d tunnel",
                                        route.bridge_samples, route.ford_samples,
                                        route.tunnel_samples);
                }
                if (!route.crossing_diagnostic.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.30f, 1.0f), "%s",
                                       route.crossing_diagnostic.c_str());
                }
            } else {
                ImGui::TextDisabled("Not solved yet (needs a Road Network node)");
            }

            if (!s_meshStatus.empty()) ImGui::TextWrapped("%s", s_meshStatus.c_str());
            ImGui::TreePop();
        }
    }

    ImGui::Spacing();

    // --- Remove Assignment (danger) ---
    if (UIWidgets::DangerButton("Remove Road Assignment", ImVec2(-1, 0))) {
        rtapi::clearRoadProfile(splineName);
    }

    UIWidgets::EndSection();
}

} // anonymous namespace
