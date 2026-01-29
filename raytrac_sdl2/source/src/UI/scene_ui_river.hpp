/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_river.hpp
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - RIVER PANEL
// ═══════════════════════════════════════════════════════════════════════════════
// UI for creating and editing river splines
// ═══════════════════════════════════════════════════════════════════════════════
#ifndef SCENE_UI_RIVER_HPP
#define SCENE_UI_RIVER_HPP

#include "scene_ui.h"
#include "imgui.h"
#include "RiverSpline.h"
#include "TerrainManager.h"
#include "ProjectManager.h"
#include "WaterSystem.h"

// ═══════════════════════════════════════════════════════════════════════════════
// RIVER PANEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════
inline void SceneUI::drawRiverPanel(UIContext& ctx) {
    auto& riverMgr = RiverManager::getInstance();
    
    // Track if river panel is currently visible/active
    static bool s_riverPanelActive = false;
    s_riverPanelActive = true;
    riverMgr.lastActiveFrame = ImGui::GetFrameCount();
    
    // ════════════════════════════════════════════════════════════════════════════════
    // PROCEDURAL OCEAN GENERATOR (High-Quality Terrain-Like Water)
    // ════════════════════════════════════════════════════════════════════════════════



    // Redundant Ocean Generator removed.
    

    
    // ════════════════════════════════════════════════════════════════════════════════
    // RIVER MANAGER
    // ════════════════════════════════════════════════════════════════════════════════
    if (UIWidgets::BeginSection("Manage Rivers", ImVec4(0.5f, 0.7f, 0.9f, 1.0f))) {
        
        // Option to show gizmos even when panel not active
        ImGui::Checkbox("Always Show Gizmos", &riverMgr.showGizmosWhenInactive);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Show river splines in viewport\neven when River panel is not focused");
        }
        
        // Create new river button
        if (ImGui::Button("+ New River", ImVec2(-1, 0))) {
            RiverSpline* newRiver = riverMgr.createRiver("River");
            riverMgr.editingRiverId = newRiver->id;
            riverMgr.isEditing = true;
            riverMgr.selectedControlPoint = -1;
            ProjectManager::getInstance().markModified();
        }
        
        ImGui::Separator();
        
        // List existing rivers
        auto& rivers = riverMgr.getRivers();
        for (auto& river : rivers) {
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_SpanAvailWidth;
            if (riverMgr.editingRiverId == river.id) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }
            
            bool open = ImGui::TreeNodeEx(
                (void*)(intptr_t)river.id,
                flags,
                "%s (%d pts)", river.name.c_str(), (int)river.controlPointCount()
            );
            
            if (ImGui::IsItemClicked()) {
                riverMgr.editingRiverId = river.id;
                riverMgr.isEditing = true;
                riverMgr.selectedControlPoint = -1;
            }
            
            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Rename")) {
                    // TODO: Rename dialog
                }
                if (ImGui::MenuItem("Delete")) {
                    int deleteId = river.id;
                    
                    // Reset editing state BEFORE deletion
                    if (riverMgr.editingRiverId == deleteId) {
                        riverMgr.editingRiverId = -1;
                        riverMgr.isEditing = false;
                        riverMgr.selectedControlPoint = -1;
                    }
                    
                    riverMgr.removeRiver(ctx.scene, deleteId);
                    
                    // Trigger rebuild
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    
                    ProjectManager::getInstance().markModified();
                    ImGui::EndPopup();
                    if (open) ImGui::TreePop();
                    break; // Exit loop after deletion
                }
                ImGui::EndPopup();
            }
            
            if (open) ImGui::TreePop();
        }
        UIWidgets::EndSection();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // SELECTED RIVER PROPERTIES
    // ─────────────────────────────────────────────────────────────────────────
    RiverSpline* selectedRiver = riverMgr.getRiver(riverMgr.editingRiverId);
    
    if (selectedRiver) {
        ImGui::Separator();
        
        if (UIWidgets::BeginSection("River Properties", ImVec4(0.5f, 0.7f, 0.9f, 1.0f))) {
            // Name
            char nameBuf[128];
            strncpy(nameBuf, selectedRiver->name.c_str(), sizeof(nameBuf) - 1);
            nameBuf[sizeof(nameBuf) - 1] = 0;
            if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) {
                selectedRiver->name = nameBuf;
                ProjectManager::getInstance().markModified();
            }
            
            ImGui::Separator();
            ImGui::Text("Mesh Generation");
            
            if (ImGui::SliderInt("Length Subdivs", &selectedRiver->lengthSubdivisions, 4, 128)) {
                selectedRiver->needsRebuild = true;
            }
            if (ImGui::SliderInt("Width Segments", &selectedRiver->widthSegments, 1, 16)) {
                selectedRiver->needsRebuild = true;
            }
            if (SceneUI::DrawSmartFloat("rbnk", "Bank Height", &selectedRiver->bankHeight, -0.5f, 1.0f, "%.3f", false, nullptr, 16)) {
                selectedRiver->needsRebuild = true;
            }
            if (ImGui::Checkbox("Follow Terrain", &selectedRiver->followTerrain)) {
                selectedRiver->needsRebuild = true;
            }
            
            ImGui::Separator();
            ImGui::Text("Default Values (for new points)");
            SceneUI::DrawSmartFloat("rdw", "Default Width", &riverMgr.defaultWidth, 0.5f, 20.0f, "%.1f", false, nullptr, 16);
            SceneUI::DrawSmartFloat("rdd", "Default Depth", &riverMgr.defaultDepth, 0.1f, 5.0f, "%.1f", false, nullptr, 16);
            UIWidgets::EndSection();
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // FLOW PHYSICS & DYNAMICS
        // ─────────────────────────────────────────────────────────────────────
        if (UIWidgets::BeginSection("Flow Physics & Dynamics", ImVec4(0.4f, 0.9f, 1.0f, 1.0f), false)) {
            RiverSpline::PhysicsParams& pp = selectedRiver->physics;
            bool changed = false;
            
            ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Geometric Deformation");
            
            if (ImGui::Checkbox("Enable Rapids/Turbulence", &pp.enableTurbulence)) changed = true;
            if (pp.enableTurbulence) {
                ImGui::Indent();
                if (SceneUI::DrawSmartFloat("rts", "Turbulence Strength", &pp.turbulenceStrength, 0.0f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                if (SceneUI::DrawSmartFloat("rrt", "Rapids Threshold", &pp.turbulenceThreshold, 0.01f, 0.2f, "%.2f", false, nullptr, 16)) changed = true;
                if (SceneUI::DrawSmartFloat("rns", "Noise Scale", &pp.noiseScale, 0.1f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::Unindent();
            }
            
            if (ImGui::Checkbox("Enable Banking (Curves)", &pp.enableBanking)) changed = true;
            if (pp.enableBanking) {
                ImGui::Indent();
                if (SceneUI::DrawSmartFloat("rbs", "Banking Strength", &pp.bankingStrength, 0.0f, 3.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::Unindent();
            }
            
            if (ImGui::Checkbox("Enable Flow Bulge", &pp.enableFlowBulge)) changed = true;
            if (pp.enableFlowBulge) {
                ImGui::Indent();
                if (SceneUI::DrawSmartFloat("rfb", "Flow Bulge", &pp.flowBulgeStrength, 0.0f, 2.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::Unindent();
            }
            
            if (changed) {
                selectedRiver->needsRebuild = true;
                // Auto-rebuild for quick feedback
                if (!selectedRiver->meshTriangles.empty()) {
                    riverMgr.generateMesh(selectedRiver, ctx.scene);
                    
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    ctx.renderer.resetCPUAccumulation();
                }
                ProjectManager::getInstance().markModified();
            }
            UIWidgets::EndSection();
        }
        
        // ─────────────────────────────────────────────────────────────────────
        // CONTROL POINTS
        // ─────────────────────────────────────────────────────────────────────
        if (UIWidgets::BeginSection("Control Points", ImVec4(1.0f, 0.6f, 0.6f, 1.0f))) {
            
            // Edit mode toggle
            ImVec4 editColor = riverMgr.isEditing ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Button, editColor);
            if (ImGui::Button(riverMgr.isEditing ? "Stop Editing (Click terrain to add points)" : "Start Editing", ImVec2(-1, 0))) {
                riverMgr.isEditing = !riverMgr.isEditing;
            }
            ImGui::PopStyleColor();
            
            if (riverMgr.isEditing) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Click on terrain to add control points");
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Right-click point to delete");
            }
            
            ImGui::Separator();
            
            // List control points
            for (int i = 0; i < (int)selectedRiver->controlPointCount(); ++i) {
                BezierControlPoint* pt = selectedRiver->getControlPoint(i);
                if (!pt) continue;
                
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf;
                if (riverMgr.selectedControlPoint == i) {
                    flags |= ImGuiTreeNodeFlags_Selected;
                }
                
                char label[64];
                snprintf(label, sizeof(label), "Point %d (W: %.1f)", i, pt->userData1);
                
                bool open = ImGui::TreeNodeEx((void*)(intptr_t)(i + 1000), flags, "%s", label);
                
                if (ImGui::IsItemClicked()) {
                    riverMgr.selectedControlPoint = i;
                }
                
                if (open) ImGui::TreePop();
            }
            UIWidgets::EndSection();
        }
            
            // Selected point properties
            if (riverMgr.selectedControlPoint >= 0 && 
                riverMgr.selectedControlPoint < (int)selectedRiver->controlPointCount()) {
                
                BezierControlPoint* pt = selectedRiver->getControlPoint(riverMgr.selectedControlPoint);
                if (pt) {
                    ImGui::Separator();
                    
                    // Show if last point (for extrude hint)
                    bool isLastPoint = (riverMgr.selectedControlPoint == (int)selectedRiver->controlPointCount() - 1);
                    if (isLastPoint) {
                        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.8f, 1.0f), 
                            "Point %d (Last - Click terrain to extend)", riverMgr.selectedControlPoint);
                    } else {
                        ImGui::Text("Selected Point %d", riverMgr.selectedControlPoint);
                    }
                    
                    bool changed = false;
                    
                    float pos[3] = {pt->position.x, pt->position.y, pt->position.z};
                    if (ImGui::DragFloat3("Position", pos, 0.1f)) {
                        pt->position = Vec3(pos[0], pos[1], pos[2]);
                        changed = true;
                    }
                    
                    if (SceneUI::DrawSmartFloat("cpw", "Width", &pt->userData1, 0.1f, 50.0f, "%.1f", false, nullptr, 16)) {
                        changed = true;
                    }
                    
                    if (SceneUI::DrawSmartFloat("cpd", "Depth", &pt->userData2, 0.0f, 10.0f, "%.2f", false, nullptr, 16)) {
                        changed = true;
                    }
                    
                    if (ImGui::Checkbox("Auto Tangent", &pt->autoTangent)) {
                        if (pt->autoTangent) {
                            selectedRiver->spline.calculateAutoTangents();
                        }
                        changed = true;
                    }
                    
                    if (!pt->autoTangent) {
                        float tin[3] = {pt->tangentIn.x, pt->tangentIn.y, pt->tangentIn.z};
                        float tout[3] = {pt->tangentOut.x, pt->tangentOut.y, pt->tangentOut.z};
                        
                        if (ImGui::DragFloat3("Tangent In", tin, 0.1f)) {
                            pt->tangentIn = Vec3(tin[0], tin[1], tin[2]);
                            changed = true;
                        }
                        if (ImGui::DragFloat3("Tangent Out", tout, 0.1f)) {
                            pt->tangentOut = Vec3(tout[0], tout[1], tout[2]);
                            changed = true;
                        }
                    }
                    
                    // Delete button + DEL key hint
                    if (ImGui::Button("Delete Point [DEL]")) {
                        selectedRiver->removeControlPoint(riverMgr.selectedControlPoint);
                        riverMgr.selectedControlPoint = -1;
                        changed = true;
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(Press DEL key)");
                    
                    // Handle DEL key press
                    if (ImGui::IsKeyPressed(ImGuiKey_Delete) && !ImGui::GetIO().WantTextInput) {
                        selectedRiver->removeControlPoint(riverMgr.selectedControlPoint);
                        // Select previous point or -1 if no points left
                        if (selectedRiver->controlPointCount() > 0) {
                            riverMgr.selectedControlPoint = (std::min)(
                                riverMgr.selectedControlPoint, 
                                (int)selectedRiver->controlPointCount() - 1);
                        } else {
                            riverMgr.selectedControlPoint = -1;
                        }
                        changed = true;
                    }
                    
                    // Auto-rebuild on changes (quick feedback)
                    if (changed) {
                        selectedRiver->needsRebuild = true;
                        
                        // Auto-rebuild if mesh exists (for quick iteration)
                        if (!selectedRiver->meshTriangles.empty()) {
                            riverMgr.generateMesh(selectedRiver, ctx.scene);
                            
                            extern bool g_bvh_rebuild_pending;
                            extern bool g_optix_rebuild_pending;
                            g_bvh_rebuild_pending = true;
                            g_optix_rebuild_pending = true;
                            ctx.renderer.resetCPUAccumulation();
                        }
                        
                        ProjectManager::getInstance().markModified();
                    }
                }
            }
        }
        
        if (selectedRiver) {
            // ─────────────────────────────────────────────────────────────────────
            // WATER MATERIAL INFO (Parameters are managed via Water panel)
            // ─────────────────────────────────────────────────────────────────────
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Water Material");
            if (selectedRiver->waterSurfaceId >= 0) {
                ImGui::BulletText("Registered as WaterSurface (ID: %d)", selectedRiver->waterSurfaceId);
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), 
                    "Edit waves, colors, and effects in the Water panel");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), 
                    "Build mesh to create water surface");
            }
            
            // ─────────────────────────────────────────────────────────────────────
            // ACTIONS
            // ─────────────────────────────────────────────────────────────────────
            ImGui::Separator();
            
            if (selectedRiver->needsRebuild) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Mesh needs rebuild!");
            }
            
            if (ImGui::Button("Rebuild Mesh", ImVec2(-1, 0))) {
                riverMgr.generateMesh(selectedRiver, ctx.scene);
                
                // Trigger scene rebuild
                extern bool g_bvh_rebuild_pending;
                extern bool g_optix_rebuild_pending;
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                ctx.renderer.resetCPUAccumulation();
            }
        }
        
        // Carve River Bed into terrain
        if (ImGui::CollapsingHeader("Carve Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Check for terrain availability only for interactivity/info
            if (!TerrainManager::getInstance().hasActiveTerrain()) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "No active terrain found! Carve disabled.");
            }
            
            SceneUI::DrawSmartFloat("cdm", "Depth Multiplier", &riverMgr.carveDepthMult, 0.1f, 3.0f, "%.1f", false, nullptr, 16);
                SceneUI::DrawSmartFloat("csm", "Smoothness", &riverMgr.carveSmoothness, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                ImGui::Checkbox("Apply Post-Erosion (Smooth Edges)", &riverMgr.carveAutoPostErosion);
                if (riverMgr.carveAutoPostErosion) {
                    ImGui::SliderInt("Erosion Iterations", &riverMgr.carveErosionIterations, 5, 30);
                }


                
                // ═══════════════════════════════════════════════════════════════
                // NATURAL CAVE SETTINGS (Doğal Nehir Yatağı)
                // ═══════════════════════════════════════════════════════════════
                ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.7f, 1.0f), "Natural Riverbed Features");
                
                // Noise-based edge irregularity
                ImGui::Checkbox("Edge Noise", &riverMgr.carveEnableNoise);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add natural irregularity to river edges");
                }
                if (riverMgr.carveEnableNoise) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cns", "Noise Scale", &riverMgr.carveNoiseScale, 0.05f, 0.5f, "%.2f", false, nullptr, 16);
                    SceneUI::DrawSmartFloat("cnst", "Noise Strength", &riverMgr.carveNoiseStrength, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                // Deep Pools
                ImGui::Checkbox("Deep Pools", &riverMgr.carveEnableDeepPools);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add random deep pools along the river");
                }
                if (riverMgr.carveEnableDeepPools) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cpf", "Pool Frequency", &riverMgr.carvePoolFrequency, 0.0f, 0.5f, "%.2f", false, nullptr, 16);
                    SceneUI::DrawSmartFloat("cpdm", "Pool Depth Mult", &riverMgr.carvePoolDepthMult, 1.0f, 3.0f, "%.1f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                // Riffles (shallow zones)
                ImGui::Checkbox("Riffles (Shallow Zones)", &riverMgr.carveEnableRiffles);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add shallow riffle zones (rapids-like areas)");
                }
                if (riverMgr.carveEnableRiffles) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("crf", "Riffle Frequency", &riverMgr.carveRiffleFrequency, 0.0f, 0.5f, "%.2f", false, nullptr, 16);
                    SceneUI::DrawSmartFloat("crdm", "Riffle Depth Mult", &riverMgr.carveRiffleDepthMult, 0.1f, 0.8f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                // Asymmetric Banks (meander physics)
                ImGui::Checkbox("Asymmetric Banks", &riverMgr.carveEnableAsymmetry);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Make outer bends deeper, inner bends shallower (realistic meander physics)");
                }
                if (riverMgr.carveEnableAsymmetry) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cas", "Asymmetry Strength", &riverMgr.carveAsymmetryStrength, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                // Point Bars
                ImGui::Checkbox("Point Bars", &riverMgr.carveEnablePointBars);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add sediment deposits on inner bends (Point Bar formation)");
                }
                if (riverMgr.carveEnablePointBars) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cpbs", "Point Bar Strength", &riverMgr.carvePointBarStrength, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }
                
                ImGui::Separator();
                
                // Auto-Carve Option
                ImGui::Checkbox("Auto-Carve on Move", &riverMgr.autoCarveOnMove);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Automatically carves terrain when points are moved.\nWARNING: Destructive (leaves old channel).");
                }
                
                ImGui::Separator();
                
                // ═══════════════════════════════════════════════════════════════
                // CARVE BUTTONS
                // ═══════════════════════════════════════════════════════════════
                
                // Standard Carve Button
                if (ImGui::Button("Carve River Bed (Simple)", ImVec2(-1, 0))) {
                    if (selectedRiver->spline.pointCount() >= 2) {
                        // Backup before manual carve
                        auto& tm = TerrainManager::getInstance();
                        if (!riverMgr.hasTerrainBackup && !tm.getTerrains().empty()) {
                            auto& t = tm.getTerrains()[0];
                            riverMgr.terrainBackupData = t.heightmap.data;
                            riverMgr.terrainBackupWidth = t.heightmap.width;
                            riverMgr.terrainBackupHeight = t.heightmap.height;
                            riverMgr.hasTerrainBackup = true;
                        }

                        // Sample many points along spline
                        std::vector<Vec3> samplePoints;
                        std::vector<float> sampleWidths;
                        std::vector<float> sampleDepths;
                        
                        int numSamples = selectedRiver->lengthSubdivisions * 3;
                        for (int i = 0; i <= numSamples; ++i) {
                            float t = (float)i / (float)numSamples;
                            samplePoints.push_back(selectedRiver->spline.samplePosition(t));
                            sampleWidths.push_back(selectedRiver->spline.sampleUserData1(t));
                            sampleDepths.push_back(selectedRiver->spline.sampleUserData2(t) * riverMgr.carveDepthMult);
                        }
                        
                        // Carve the river bed
                        TerrainManager::getInstance().carveRiverBed(
                            -1,
                            samplePoints,
                            sampleWidths,
                            sampleDepths,
                            riverMgr.carveSmoothness,
                            ctx.scene
                        );
                        
                        // Apply post-erosion
                        if (riverMgr.carveAutoPostErosion) {
                            auto& terrains = TerrainManager::getInstance().getTerrains();
                            if (!terrains.empty()) {
                                ThermalErosionParams ep;
                                ep.iterations = riverMgr.carveErosionIterations;
                                ep.talusAngle = 0.3f;
                                ep.erosionAmount = 0.4f;
                                TerrainManager::getInstance().thermalErosion(&terrains[0], ep);
                            }
                        }
                        
                        // Rebuild river mesh
                        selectedRiver->needsRebuild = true;
                        riverMgr.generateMesh(selectedRiver, ctx.scene);
                        
                        extern bool g_bvh_rebuild_pending;
                        extern bool g_optix_rebuild_pending;
                        g_bvh_rebuild_pending = true;
                        g_optix_rebuild_pending = true;
                        ctx.renderer.resetCPUAccumulation();
                        
                        ProjectManager::getInstance().markModified();
                    }
                }
                
                // NATURAL Carve Button (Main Feature)
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.4f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.5f, 1.0f));
                if (ImGui::Button("Carve Natural Riverbed", ImVec2(-1, 0))) {
                    if (selectedRiver->spline.pointCount() >= 2) {
                         // Backup before manual carve
                        auto& tm = TerrainManager::getInstance();
                        if (!riverMgr.hasTerrainBackup && !tm.getTerrains().empty()) {
                            auto& t = tm.getTerrains()[0];
                            riverMgr.terrainBackupData = t.heightmap.data;
                            riverMgr.terrainBackupWidth = t.heightmap.width;
                            riverMgr.terrainBackupHeight = t.heightmap.height;
                            riverMgr.hasTerrainBackup = true;
                        }

                        // Sample many points along spline
                        std::vector<Vec3> samplePoints;
                        std::vector<float> sampleWidths;
                        std::vector<float> sampleDepths;
                        
                        int numSamples = selectedRiver->lengthSubdivisions * 3;
                        for (int i = 0; i <= numSamples; ++i) {
                            float t = (float)i / (float)numSamples;
                            samplePoints.push_back(selectedRiver->spline.samplePosition(t));
                            sampleWidths.push_back(selectedRiver->spline.sampleUserData1(t));
                            sampleDepths.push_back(selectedRiver->spline.sampleUserData2(t) * riverMgr.carveDepthMult);
                        }
                        
                        // Build NaturalCarveParams from UI settings
                        TerrainManager::NaturalCarveParams np;
                        np.enableNoise = riverMgr.carveEnableNoise;
                        np.noiseScale = riverMgr.carveNoiseScale;
                        np.noiseStrength = riverMgr.carveNoiseStrength;
                        np.enableDeepPools = riverMgr.carveEnableDeepPools;
                        np.poolFrequency = riverMgr.carvePoolFrequency;
                        np.poolDepthMult = riverMgr.carvePoolDepthMult;
                        np.enableRiffles = riverMgr.carveEnableRiffles;
                        np.riffleFrequency = riverMgr.carveRiffleFrequency;
                        np.riffleDepthMult = riverMgr.carveRiffleDepthMult;
                        np.enableAsymmetry = riverMgr.carveEnableAsymmetry;
                        np.asymmetryStrength = riverMgr.carveAsymmetryStrength;
                        np.enablePointBars = riverMgr.carveEnablePointBars;
                        np.pointBarStrength = riverMgr.carvePointBarStrength;
                        
                        // Carve the NATURAL river bed
                        TerrainManager::getInstance().carveRiverBedNatural(
                            -1,
                            samplePoints,
                            sampleWidths,
                            sampleDepths,
                            riverMgr.carveSmoothness,
                            np,
                            ctx.scene
                        );
                        
                        // Apply post-erosion
                        if (riverMgr.carveAutoPostErosion) {
                            auto& terrains = TerrainManager::getInstance().getTerrains();
                            if (!terrains.empty()) {
                                ThermalErosionParams ep;
                                ep.iterations = riverMgr.carveErosionIterations;
                                ep.talusAngle = 0.3f;
                                ep.erosionAmount = 0.4f;
                                TerrainManager::getInstance().thermalErosion(&terrains[0], ep);
                            }
                        }
                        
                        // Rebuild river mesh
                        selectedRiver->needsRebuild = true;
                        riverMgr.generateMesh(selectedRiver, ctx.scene);
                        
                        extern bool g_bvh_rebuild_pending;
                        extern bool g_optix_rebuild_pending;
                        g_bvh_rebuild_pending = true;
                        g_optix_rebuild_pending = true;
                        ctx.renderer.resetCPUAccumulation();
                        
                        ProjectManager::getInstance().markModified();
                    }
                }
                ImGui::PopStyleColor(2);
                
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                    "Natural: Pools, Riffles, Asymmetry, Point Bars");
                // BACKUP / RESTORE CONTROLS
                ImGui::Separator();
                if (riverMgr.hasTerrainBackup) {
                    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Backup Available");
                    
                    if (ImGui::Button("Reset / Undo Carve", ImVec2(140, 0))) {
                        // Restore from backup
                        auto& tm = TerrainManager::getInstance();
                        if (!tm.getTerrains().empty()) {
                            auto& t = tm.getTerrains()[0]; // Assume first terrain for now
                            if (t.heightmap.data.size() == riverMgr.terrainBackupData.size()) {
                                t.heightmap.data = riverMgr.terrainBackupData;
                                tm.updateTerrainMesh(&t);
                                
                                extern bool g_bvh_rebuild_pending;
                                extern bool g_optix_rebuild_pending;
                                g_bvh_rebuild_pending = true;
                                g_optix_rebuild_pending = true;
                                ctx.renderer.resetCPUAccumulation();
                                ProjectManager::getInstance().markModified();
                                SCENE_LOG_INFO("Terrain restored from backup.");
                            }
                        }
                    }
                    
                    ImGui::SameLine();
                    if (ImGui::Button("Commit", ImVec2(80, 0))) {
                        riverMgr.hasTerrainBackup = false;
                        riverMgr.terrainBackupData.clear();
                        SCENE_LOG_INFO("Carve changes committed.");
                    }
                    ImGui::SameLine(); 
                    ImGui::TextDisabled("(?)");
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Commit: Clears backup, making current state the new baseline.\nReset: Reverts to original state before first carve.");
                } else {
                     ImGui::TextDisabled("Backup logic active (Backup created on first Carve)");
                }
                
                // UIWidgets::EndSection();
            }
        
        if (ImGui::Button("Clear All Points", ImVec2(-1, 0))) {
            // Remove existing mesh from scene
            for (auto& tri : selectedRiver->meshTriangles) {
                auto it = std::find(ctx.scene.world.objects.begin(), ctx.scene.world.objects.end(),
                                   std::static_pointer_cast<Hittable>(tri));
                if (it != ctx.scene.world.objects.end()) {
                    ctx.scene.world.objects.erase(it);
                }
            }
            selectedRiver->meshTriangles.clear();
            selectedRiver->spline.clear();
            selectedRiver->needsRebuild = true;
            riverMgr.selectedControlPoint = -1;
            
            extern bool g_bvh_rebuild_pending;
            extern bool g_optix_rebuild_pending;
            g_bvh_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            
            ProjectManager::getInstance().markModified();
        }
    
}

// ═══════════════════════════════════════════════════════════════════════════════
// RIVER SPLINE VISUALIZATION (Call from draw loop)
// ═══════════════════════════════════════════════════════════════════════════════
inline void SceneUI::drawRiverGizmos(UIContext& ctx, bool& gizmo_hit) {
    auto& riverMgr = RiverManager::getInstance();
    
    // Check if panel is active (drawn this frame or last frame)
    bool isPanelActive = (riverMgr.lastActiveFrame >= ImGui::GetFrameCount() - 1);
    
    // Hide gizmos if not editing active, panel is not focused, and "Always Show" is off
    if (!riverMgr.showGizmosWhenInactive && !isPanelActive) {
        return;
    }
    
    if (!ctx.scene.camera) return;
    
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;
    
    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    float aspect = io.DisplaySize.x / io.DisplaySize.y;
    
        // Calculate Viewport offsets and sizes
        float left_off = showSidePanel ? side_panel_width : 0.0f;
        float top_off = 19.0f; // Menu bar height estimate
        bool show_bottom = (show_animation_panel || show_scene_log || show_terrain_graph || show_anim_graph);
        float bottom_off = show_bottom ? (bottom_panel_height + 24.0f) : 24.0f;

        float v_width = io.DisplaySize.x - left_off;
        float v_height = io.DisplaySize.y - top_off - bottom_off;

        auto Project = [&](const Vec3& p) -> ImVec2 {
            Vec3 dir = (p - cam.lookfrom);
            float dist = dir.length();
            dir = dir.normalize();

            Vec3 forward = (cam.lookat - cam.lookfrom).normalize();
            float depth = Vec3::dot(p - cam.lookfrom, forward);

            if (depth <= 0.01f) return ImVec2(-1, -1);

            float local_y = Vec3::dot(p - cam.lookfrom, cam.v);
            float local_x = Vec3::dot(p - cam.lookfrom, cam.u);

            float half_h = depth * tan_half_fov;
            float half_w = half_h * aspect;

            return ImVec2(
                left_off + ((local_x / half_w) * 0.5f + 0.5f) * v_width,
                top_off + (0.5f - (local_y / half_h) * 0.5f) * v_height
            );
        };
    
    auto IsOnScreen = [](const ImVec2& v) { return v.x > -5000; };
    
    // Draw all rivers
    for (auto& river : riverMgr.getRivers()) {
        bool isSelected = (riverMgr.editingRiverId == river.id);
        
        if (!river.showSpline) continue;
        if (river.spline.pointCount() < 2) {
            // Just draw single point if only one exists
            if (river.spline.pointCount() == 1) {
                ImVec2 pt = Project(river.spline.points[0].position);
                if (IsOnScreen(pt)) {
                    draw_list->AddCircleFilled(pt, 8.0f, IM_COL32(100, 150, 255, 200));
                    draw_list->AddCircle(pt, 8.0f, IM_COL32(255, 255, 255, 255), 0, 2.0f);
                }
            }
            continue;
        }
        
        // Draw spline curve
        ImU32 splineColor = isSelected ? IM_COL32(100, 200, 255, 200) : IM_COL32(50, 100, 200, 150);
        
        Vec3 prevPos = river.spline.samplePosition(0);
        for (int i = 1; i <= 50; ++i) {
            float t = (float)i / 50.0f;
            Vec3 pos = river.spline.samplePosition(t);
            
            ImVec2 p1 = Project(prevPos);
            ImVec2 p2 = Project(pos);
            
            if (IsOnScreen(p1) && IsOnScreen(p2)) {
                draw_list->AddLine(p1, p2, splineColor, 2.0f);
            }
            
            prevPos = pos;
        }
        
        // Draw width visualization (dashed lines on sides)
        if (isSelected) {
            for (int i = 0; i <= 20; ++i) {
                float t = (float)i / 20.0f;
                Vec3 center = river.spline.samplePosition(t);
                Vec3 right = river.spline.sampleRight(t);
                float width = river.spline.sampleUserData1(t);
                
                Vec3 left3d = center - right * (width * 0.5f);
                Vec3 right3d = center + right * (width * 0.5f);
                
                ImVec2 leftPt = Project(left3d);
                ImVec2 rightPt = Project(right3d);
                
                if (i % 2 == 0) { // Dashed effect
                    if (IsOnScreen(leftPt)) {
                        draw_list->AddCircleFilled(leftPt, 2.0f, IM_COL32(100, 200, 255, 100));
                    }
                    if (IsOnScreen(rightPt)) {
                        draw_list->AddCircleFilled(rightPt, 2.0f, IM_COL32(100, 200, 255, 100));
                    }
                }
            }
        }
        
        // Draw control points
        if (river.showControlPoints || isSelected) {
            for (int i = 0; i < (int)river.spline.pointCount(); ++i) {
                auto& pt = river.spline.points[i];
                ImVec2 screenPt = Project(pt.position);
                
                if (!IsOnScreen(screenPt)) continue;
                
                bool isPointSelected = (isSelected && riverMgr.selectedControlPoint == i);
                
                // Point appearance
                float radius = isPointSelected ? 10.0f : 7.0f;
                ImU32 fillColor = isPointSelected ? IM_COL32(255, 200, 50, 255) : IM_COL32(100, 150, 255, 200);
                ImU32 outlineColor = IM_COL32(255, 255, 255, 255);
                
                draw_list->AddCircleFilled(screenPt, radius, fillColor);
                draw_list->AddCircle(screenPt, radius, outlineColor, 0, 2.0f);
                
                // Point index label
                char label[16];
                snprintf(label, sizeof(label), "%d", i);
                draw_list->AddText(ImVec2(screenPt.x + 12, screenPt.y - 6), IM_COL32(255, 255, 255, 200), label);
                
                // Draw tangent handles for selected point
                if (isPointSelected && !pt.autoTangent) {
                    Vec3 tangentInWorld = pt.position + pt.tangentIn;
                    Vec3 tangentOutWorld = pt.position + pt.tangentOut;
                    
                    ImVec2 tin = Project(tangentInWorld);
                    ImVec2 tout = Project(tangentOutWorld);
                    
                    if (IsOnScreen(tin)) {
                        draw_list->AddLine(screenPt, tin, IM_COL32(255, 100, 100, 200), 1.5f);
                        draw_list->AddCircleFilled(tin, 5.0f, IM_COL32(255, 100, 100, 255));
                    }
                    if (IsOnScreen(tout)) {
                        draw_list->AddLine(screenPt, tout, IM_COL32(100, 255, 100, 200), 1.5f);
                        draw_list->AddCircleFilled(tout, 5.0f, IM_COL32(100, 255, 100, 255));
                    }
                }
                
                // Click detection for point selection
                if (isSelected && !ImGuizmo::IsOver()) {
                    float dx = io.MousePos.x - screenPt.x;
                    float dy = io.MousePos.y - screenPt.y;
                    float dist = sqrtf(dx * dx + dy * dy);
                    
                    if (dist < 15.0f) {
                        // Left click - select
                        if (ImGui::IsMouseClicked(0)) {
                            riverMgr.selectedControlPoint = i;
                            gizmo_hit = true;
                        }
                        // Right click - delete
                        else if (ImGui::IsMouseClicked(1)) {
                            river.removeControlPoint(i);
                            river.needsRebuild = true;
                            riverMgr.selectedControlPoint = -1;
                            ProjectManager::getInstance().markModified();
                            gizmo_hit = true;
                            break;
                        }
                    }
                }
                
                // ─────────────────────────────────────────────────────────────
                // DRAG TO MOVE selected control point
                // ─────────────────────────────────────────────────────────────
                if (isPointSelected && ImGui::IsMouseDragging(0) && !ImGui::GetIO().WantCaptureMouse) {
                    // Project mouse delta to world movement
                    Vec3 forward = cam_forward;
                    Vec3 right = cam_right;
                    Vec3 up = cam_up;
                    
                    // Calculate depth of point from camera
                    Vec3 toPoint = pt.position - cam.lookfrom;
                    float depth = toPoint.dot(forward);
                    
                    if (depth > 0.1f) {
                        // Convert pixel delta to world delta
                        float half_h = depth * tan_half_fov;
                        float pixelsPerUnit = io.DisplaySize.y / (2.0f * half_h);
                        
                        ImVec2 delta = io.MouseDelta;
                        float worldDeltaX = delta.x / pixelsPerUnit;
                        float worldDeltaY = -delta.y / pixelsPerUnit;
                        
                        // Move in camera plane (right + up)
                        Vec3 movement = right * worldDeltaX + up * worldDeltaY;
                        pt.position = pt.position + movement;
                        
                        // Optionally snap Y to terrain
                        if (river.followTerrain && TerrainManager::getInstance().hasActiveTerrain()) {
                            pt.position.y = TerrainManager::getInstance().sampleHeight(pt.position.x, pt.position.z);
                        }
                        
                        // Recalculate tangents if auto
                        if (pt.autoTangent) {
                            river.spline.calculateAutoTangents();
                        }
                        
                        river.needsRebuild = true;
                        riverMgr.isDraggingPoint = true;  // Track drag state
                        gizmo_hit = true;
                    }
                }
                
                // Rebuild mesh when drag ends
                if (isPointSelected && riverMgr.isDraggingPoint && ImGui::IsMouseReleased(0)) {
                    riverMgr.isDraggingPoint = false;
                    
                    // AUTO-CARVE ON MOVE
                    if (riverMgr.autoCarveOnMove && TerrainManager::getInstance().hasActiveTerrain()) {
                        // Backup before auto-carve if not exists
                        auto& tm = TerrainManager::getInstance();
                        if (!riverMgr.hasTerrainBackup && !tm.getTerrains().empty()) {
                            auto& t = tm.getTerrains()[0];
                            riverMgr.terrainBackupData = t.heightmap.data;
                            riverMgr.terrainBackupWidth = t.heightmap.width;
                            riverMgr.terrainBackupHeight = t.heightmap.height;
                            riverMgr.hasTerrainBackup = true;
                        }

                        // Sample many points along spline
                        std::vector<Vec3> samplePoints;
                        std::vector<float> sampleWidths;
                        std::vector<float> sampleDepths;
                        
                        int numSamples = river.lengthSubdivisions * 3;
                        for (int k = 0; k <= numSamples; ++k) {
                            float t = (float)k / (float)numSamples;
                            samplePoints.push_back(river.spline.samplePosition(t));
                            sampleWidths.push_back(river.spline.sampleUserData1(t));
                            sampleDepths.push_back(river.spline.sampleUserData2(t) * riverMgr.carveDepthMult);
                        }
                        
                        // Carve
                        TerrainManager::getInstance().carveRiverBed(
                            -1,
                            samplePoints,
                            sampleWidths,
                            sampleDepths,
                            riverMgr.carveSmoothness,
                            ctx.scene
                        );
                        
                        // Post-Erosion
                        if (riverMgr.carveAutoPostErosion) {
                            auto& terrains = TerrainManager::getInstance().getTerrains();
                            if (!terrains.empty()) {
                                ThermalErosionParams ep;
                                ep.iterations = riverMgr.carveErosionIterations;
                                ep.talusAngle = 0.3f;
                                ep.erosionAmount = 0.4f;
                                TerrainManager::getInstance().thermalErosion(&terrains[0], ep);
                            }
                        }
                    }
                    
                    // Rebuild the mesh after dragging

                    if (river.needsRebuild && !river.meshTriangles.empty()) {
                        riverMgr.generateMesh(&river, ctx.scene);
                        
                        extern bool g_bvh_rebuild_pending;
                        extern bool g_optix_rebuild_pending;
                        g_bvh_rebuild_pending = true;
                        g_optix_rebuild_pending = true;
                        ctx.renderer.resetCPUAccumulation();
                    }
                    
                    ProjectManager::getInstance().markModified();
                }
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // ADD NEW POINT ON TERRAIN CLICK (when editing)
    // ─────────────────────────────────────────────────────────────────────────
    RiverSpline* editingRiver = riverMgr.getRiver(riverMgr.editingRiverId);
    
    if (editingRiver && riverMgr.isEditing && !gizmo_hit && !ImGuizmo::IsOver()) {
        if (ImGui::IsMouseClicked(0) && !ImGui::GetIO().WantCaptureMouse) {
            // Raycast to terrain
            float mx = io.MousePos.x;
            float my = io.MousePos.y;

            // Calculate Viewport offsets and sizes
            float left_off = showSidePanel ? side_panel_width : 0.0f;
            float top_off = 19.0f; // Menu bar height estimate
            bool show_bottom = (show_animation_panel || show_scene_log || show_terrain_graph || show_anim_graph);
            float bottom_off = show_bottom ? (bottom_panel_height + 24.0f) : 24.0f;

            float v_width = io.DisplaySize.x - left_off;
            float v_height = io.DisplaySize.y - top_off - bottom_off;

            // Normalize mouse position relative to viewport
            float u = (mx - left_off) / v_width;
            float v = 1.0f - ((my - top_off) / v_height);

            // Use Camera's own ray generation for perfect consistency
            Ray cameraRay = cam.get_ray(u, v);
            Vec3 rayDir = cameraRay.direction;
            Vec3 rayOrigin = cameraRay.origin;
            
            // Perform accurate terrain raycast
            if (TerrainManager::getInstance().hasActiveTerrain()) {
                float closest_t = 1e20f;
                Vec3 hitPoint;
                bool found_terrain = false;
                
                auto& terrains = TerrainManager::getInstance().getTerrains();
                for (auto& terrain : terrains) {
                    float t_out;
                    Vec3 n_out;
                    if (TerrainManager::getInstance().intersectRay(&terrain, cameraRay, t_out, n_out)) {
                        if (t_out < closest_t) {
                            closest_t = t_out;
                            hitPoint = cameraRay.origin + cameraRay.direction * t_out;
                            found_terrain = true;
                        }
                    }
                }
                
                if (found_terrain) {
                    // Add control point
                    editingRiver->addControlPoint(hitPoint, riverMgr.defaultWidth, riverMgr.defaultDepth);
                    riverMgr.selectedControlPoint = (int)editingRiver->controlPointCount() - 1;
                    ProjectManager::getInstance().markModified();
                    gizmo_hit = true;
                }
            } else {
                // No terrain - intersect with Y=0 plane
                if (fabsf(rayDir.y) > 0.01f) {
                    float t = -rayOrigin.y / rayDir.y;
                    if (t > 0) {
                        Vec3 hitPoint = rayOrigin + rayDir * t;
                        editingRiver->addControlPoint(hitPoint, riverMgr.defaultWidth, riverMgr.defaultDepth);
                        riverMgr.selectedControlPoint = (int)editingRiver->controlPointCount() - 1;
                        ProjectManager::getInstance().markModified();
                        gizmo_hit = true;
                    }
                }
            }
        }
    }
}

#endif // SCENE_UI_RIVER_HPP

