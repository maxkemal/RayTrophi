/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_animgraph.hpp
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file scene_ui_animgraph.hpp
 * @brief Animation Graph Editor UI
 * 
 * Node-based animation editor integrated with the existing scene UI system.
 * Uses NodeEditorUIV2 for consistent look and feel.
 */

#include "scene_ui.h"
#include "AnimationNodes.h"
#include "AnimationController.h"
#include "AnimatedObject.h"
#include "Triangle.h" // Added for dynamic_pointer_cast<Triangle>
#include "imgui.h"
#include <string>
#include <vector>
#include <SceneSelection.h>

// ============================================================================
// ANIMATION GRAPH UI STATE
// ============================================================================

struct AnimGraphUIState {
    // Graph instance per skeleton/character
    std::unordered_map<std::string, std::unique_ptr<AnimationGraph::AnimationNodeGraph>> graphs;
    
    // Current selected character for editing
    std::string activeCharacter;
    
    // Editor state
    bool showNodeEditor = false;
    bool showParameterPanel = true;
    bool showPreviewPanel = true;
    
    // Node creation popup
    bool showAddNodePopup = false;
    ImVec2 addNodePopupPos;
    std::string nodeSearchFilter;
    
    // Selection
    std::vector<uint32_t> selectedNodeIds;
    std::vector<uint32_t> selectedLinkIds;
    
    // Pan/Zoom
    ImVec2 canvasOffset = ImVec2(0, 0);
    float canvasZoom = 1.0f;
    
    // Link Creation State (for connecting pins by dragging)
    bool isCreatingLink = false;
    uint32_t linkStartPinId = 0;
    bool linkStartIsOutput = true;  // true if dragging from output pin
    
    // Pin positions cache (filled during node draw, used for link drawing)
    std::unordered_map<uint32_t, ImVec2> pinScreenPositions;
    
    // Debug
    bool showDebugInfo = false;
    int debugBoneIndex = 0;
};

// Global state
inline AnimGraphUIState g_animGraphUI;

// ============================================================================
// ANIMATION PARAMETERS PANEL
// ============================================================================

inline void drawAnimationParametersPanel(UIContext& ctx, AnimationGraph::AnimationNodeGraph* graph) {
    if (!graph) return;
    
    ImGui::BeginChild("AnimParams", ImVec2(0, 150), true);
    ImGui::Text("Animation Parameters");
    ImGui::Separator();
    
    auto& evalCtx = graph->evalContext;
    
    // Float parameters
    if (!evalCtx.floatParams.empty()) {
        if (ImGui::CollapsingHeader("Float Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto& [name, value] : evalCtx.floatParams) {
                SceneUI::DrawSmartFloat(name.c_str(), name.c_str(), &value, 0.0f, 1.0f, "%.3f", false, nullptr, 16);
            }
        }
    }
    
    // Bool parameters
    if (!evalCtx.boolParams.empty()) {
        if (ImGui::CollapsingHeader("Bool Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto& [name, value] : evalCtx.boolParams) {
                ImGui::Checkbox(name.c_str(), &value);
            }
        }
    }
    
    // Add parameter buttons
    ImGui::Separator();
    static char newParamName[64] = "";
    static int paramTypeIdx = 0;
    const char* paramTypes[] = { "Float", "Bool", "Int", "Trigger" };
    
    ImGui::InputText("Name##NewParam", newParamName, sizeof(newParamName));
    ImGui::Combo("Type##NewParam", &paramTypeIdx, paramTypes, IM_ARRAYSIZE(paramTypes));
    
    if (ImGui::Button("Add Parameter") && strlen(newParamName) > 0) {
        switch (paramTypeIdx) {
            case 0: evalCtx.floatParams[newParamName] = 0.0f; break;
            case 1: evalCtx.boolParams[newParamName] = false; break;
            case 2: evalCtx.intParams[newParamName] = 0; break;
            case 3: graph->triggerParam(newParamName); break;
        }
        newParamName[0] = '\0';
    }
    
    ImGui::EndChild();
}

// ============================================================================
// ANIMATION CLIPS PANEL
// ============================================================================

inline void drawAnimationClipsPanel(UIContext& ctx) {
    // Determine which animator to use
    AnimationController* animCtrl = nullptr;
    if (!g_animGraphUI.activeCharacter.empty()) {
        for (auto& mctx : ctx.scene.importedModelContexts) {
            if (mctx.importName == g_animGraphUI.activeCharacter) {
                animCtrl = mctx.animator.get();
                break;
            }
        }
        
        // Fallback: If not an imported model context, maybe it's a specific animated object
        if (!animCtrl) {
            for (auto& obj : ctx.scene.world.objects) {
                auto animObj = std::dynamic_pointer_cast<AnimatedObject>(obj);
                if (animObj) {
                    // AnimatedObject doesn't store a single name, it's a hierarchy.
                    // We check if the active character name corresponds to a node in its hierarchy.
                    if (animObj->m_nodeHierarchy.count(g_animGraphUI.activeCharacter)) {
                         // Found a match in this object's hierarchy
                         // However, AnimatedObject doesn't expose a controller directly.
                         // This path is incomplete, but we avoid the build error.
                         break;
                    }
                }
            }
        }
    }
    
    // Fallback to singleton for non-model animations (e.g. camera/light)
    if (!animCtrl) animCtrl = &AnimationController::getInstance();
    
    const auto& clips = animCtrl->getAllClips();
    
    ImGui::BeginChild("AnimClips", ImVec2(0, 140), true);
    ImGui::Text("Animations (%zu) - %s", clips.size(), g_animGraphUI.activeCharacter.empty() ? "Global" : g_animGraphUI.activeCharacter.c_str());
    
    // Show scene animation count as well
    if (ctx.scene.animationDataList.size() != clips.size()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(Scene: %zu)", ctx.scene.animationDataList.size());
    }
    
    ImGui::Separator();
    
    if (clips.empty()) {
        ImGui::TextDisabled("No animations loaded.");
        ImGui::TextDisabled("Load a model with animations.");
    } else {
        for (size_t i = 0; i < clips.size(); ++i) {
            const auto& clip = clips[i];
            
            bool isPlaying = (animCtrl->getCurrentClipName() == clip.name);
            
            ImGui::PushID(static_cast<int>(i));
            
            if (isPlaying) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
            }
            
            // Show clip name with duration
            ImGui::BulletText("%s", clip.name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("(%.1fs)", clip.getDurationInSeconds());
            
            if (isPlaying) {
                ImGui::PopStyleColor();
            }
            
            ImGui::SameLine();
            
            // Play button
            if (ImGui::SmallButton(isPlaying ? "Stop" : "Play")) {
                if (isPlaying) {
                    animCtrl->stopAll();
                } else {
                    animCtrl->play(clip.name, 0.2f);
                }
            }
            
            ImGui::PopID();
        }
    }
    
    ImGui::EndChild();
}

// ============================================================================
// ANIMATION PLAYBACK CONTROLS
// ============================================================================

inline void drawAnimationPlaybackControls(UIContext& ctx) {
    AnimationController* animCtrl = nullptr;
    
    // 1. Resolve Active Animator
    if (!g_animGraphUI.activeCharacter.empty()) {
        // Check imported models
        for (auto& mctx : ctx.scene.importedModelContexts) {
            if (mctx.importName == g_animGraphUI.activeCharacter) {
                animCtrl = mctx.animator.get();
                break;
            }
        }
        // Fallback: Check static objects if not found
        if (!animCtrl) {
             for (auto& obj : ctx.scene.world.objects) {
                 auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                 if (tri && tri->getNodeName() == g_animGraphUI.activeCharacter) {
                     // TODO: Static objects might need their own controller or use global
                     break;
                 }
             }
        }
    }
    
    // Default to global instance if no specific character selected
    if (!animCtrl) animCtrl = &AnimationController::getInstance();
    
    ImGui::BeginChild("AnimPlayback", ImVec2(0, 80), true);
    
    // Current state info
    std::string currentClip = animCtrl->getCurrentClipName();
    float normalizedTime = animCtrl->getNormalizedTime();
    bool isPlaying = animCtrl->isPlaying();
    bool isPaused = animCtrl->isPaused();
    
    ImGui::Text("Current: %s", currentClip.empty() ? "(Stopped)" : currentClip.c_str());
    
    // Progress bar
    ImGui::ProgressBar(normalizedTime, ImVec2(-1, 0), 
        (std::to_string((int)(normalizedTime * 100)) + "%").c_str());
    
    // Playback buttons
    // Play/Pause Toggle
    if (ImGui::Button(isPaused ? "Resume" : (isPlaying ? "Pause" : "Play"), ImVec2(60, 22))) {
        if (isPlaying || isPaused) {
            animCtrl->setPaused(!isPaused);
        } else {
            // If stopped and user hits Play, try to play the first available clip or continue last
            if (!currentClip.empty()) animCtrl->play(currentClip, 0.2f);
            else {
                 auto clips = animCtrl->getAllClips();
                 if (!clips.empty()) animCtrl->play(clips[0].name, 0.2f);
            }
        }
    }
    
    ImGui::SameLine();
    
    // Stop Button
    if (ImGui::Button("Stop", ImVec2(50, 22))) {
        animCtrl->stopAll(); 
    }
    
    ImGui::SameLine();
    
    // Rewind
    if (ImGui::Button("<<", ImVec2(30, 22))) {
        animCtrl->setTime(0.0f);
    }

    // Status Indicators
    if (animCtrl->isBlending()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "[Blending]");
    }
    
    ImGui::EndChild();
}

// ============================================================================
// ANIMATION STATE MACHINE PANEL
// ============================================================================

inline void drawStateMachinePanel(UIContext& ctx, AnimationGraph::AnimationNodeGraph* graph) {
    if (!graph) return;
    
    ImGui::BeginChild("StateMachine", ImVec2(0, 150), true);
    ImGui::Text("State Machine");
    ImGui::Separator();
    
    // Find state machine nodes in graph
    for (auto& node : graph->nodes) {
        auto* smNode = dynamic_cast<AnimationGraph::StateMachineNode*>(node.get());
        if (!smNode) continue;
        
        ImGui::Text("Current State: %s", smNode->currentStateName.c_str());
        
        if (smNode->isTransitioning) {
            ImGui::ProgressBar(smNode->transitionProgress, ImVec2(-1, 0),
                ("-> " + smNode->targetStateName).c_str());
        }
        
        ImGui::Separator();
        ImGui::Text("States:");
        
        for (const auto& state : smNode->states) {
            bool isCurrent = (state.name == smNode->currentStateName);
            
            if (isCurrent) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
            }
            
            if (ImGui::Selectable(state.name.c_str(), isCurrent)) {
                smNode->forceState(state.name);
            }
            
            if (isCurrent) {
                ImGui::PopStyleColor();
            }
        }
        
        break; // Only show first state machine for now
    }
    
    ImGui::EndChild();
}
// ============================================================================
// NODE CANVAS DRAWING (with pin connection support)
// ============================================================================

inline void drawNodeCanvas(UIContext& ctx, AnimationGraph::AnimationNodeGraph* graph) {
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    float zoom = g_animGraphUI.canvasZoom;

    // Clear pin position cache
    g_animGraphUI.pinScreenPositions.clear();

    // Background
    drawList->AddRectFilled(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(30, 30, 35, 255));

    // Grid
    float gridSize = 32.0f * zoom;
    for (float x = fmodf(g_animGraphUI.canvasOffset.x, gridSize); x < canvasSize.x; x += gridSize) {
        drawList->AddLine(
            ImVec2(canvasPos.x + x, canvasPos.y),
            ImVec2(canvasPos.x + x, canvasPos.y + canvasSize.y),
            IM_COL32(50, 50, 55, 255));
    }
    for (float y = fmodf(g_animGraphUI.canvasOffset.y, gridSize); y < canvasSize.y; y += gridSize) {
        drawList->AddLine(
            ImVec2(canvasPos.x, canvasPos.y + y),
            ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + y),
            IM_COL32(50, 50, 55, 255));
    }

    // ========== DRAW NODES & COLLECT PIN POSITIONS ==========
    for (auto& node : graph->nodes) {
        ImVec2 nodePos = ImVec2(
            canvasPos.x + g_animGraphUI.canvasOffset.x + node->x * zoom,
            canvasPos.y + g_animGraphUI.canvasOffset.y + node->y * zoom);

        float nodeWidth = 150.0f * zoom;
        float nodeHeight = (60.0f + node->inputs.size() * 20.0f + node->outputs.size() * 20.0f) * zoom;
        float headerHeight = 25.0f * zoom;
        float pinSpacing = 20.0f * zoom;
        float pinRadius = 6.0f * zoom;
        float cornerRadius = 4.0f * zoom;

        ImVec2 nodeMax = ImVec2(nodePos.x + nodeWidth, nodePos.y + nodeHeight);

        // Node background
        drawList->AddRectFilled(nodePos, nodeMax, IM_COL32(50, 50, 55, 240), cornerRadius);

        // Header
        drawList->AddRectFilled(nodePos,
            ImVec2(nodeMax.x, nodePos.y + headerHeight),
            node->metadata.headerColor, cornerRadius, ImDrawFlags_RoundCornersTop);

        // Title
        drawList->AddText(ImVec2(nodePos.x + 5 * zoom, nodePos.y + 5 * zoom),
            IM_COL32(255, 255, 255, 255), node->metadata.displayName.c_str());

        // Border
        bool isSelected = std::find(g_animGraphUI.selectedNodeIds.begin(),
            g_animGraphUI.selectedNodeIds.end(), node->id) != g_animGraphUI.selectedNodeIds.end();

        drawList->AddRect(nodePos, nodeMax,
            isSelected ? IM_COL32(255, 180, 50, 255) : IM_COL32(80, 80, 85, 255),
            cornerRadius, 0, isSelected ? 2.0f : 1.0f);

        // Input pins
        float pinY = nodePos.y + 35 * zoom;
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            auto& pin = node->inputs[i];
            ImVec2 pinPos = ImVec2(nodePos.x, pinY);
            
            // Cache pin screen position
            g_animGraphUI.pinScreenPositions[pin.id] = pinPos;
            
            // Draw pin circle
            ImU32 pinColor = pin.cachedColor != 0 ? pin.cachedColor : IM_COL32(100, 180, 255, 255);
            drawList->AddCircleFilled(pinPos, pinRadius, pinColor);
            drawList->AddCircle(pinPos, pinRadius, IM_COL32(255, 255, 255, 150), 0, 1.0f);
            
            // Pin label
            drawList->AddText(ImVec2(pinPos.x + 10 * zoom, pinY - 8 * zoom),
                IM_COL32(200, 200, 200, 255), pin.name.c_str());

            pinY += pinSpacing;
        }

        // Output pins
        for (size_t i = 0; i < node->outputs.size(); ++i) {
            auto& pin = node->outputs[i];
            ImVec2 pinPos = ImVec2(nodePos.x + nodeWidth, pinY);
            
            // Cache pin screen position
            g_animGraphUI.pinScreenPositions[pin.id] = pinPos;
            
            // Draw pin circle
            ImU32 pinColor = pin.cachedColor != 0 ? pin.cachedColor : IM_COL32(255, 200, 100, 255);
            drawList->AddCircleFilled(pinPos, pinRadius, pinColor);
            drawList->AddCircle(pinPos, pinRadius, IM_COL32(255, 255, 255, 150), 0, 1.0f);

            // Pin label
            ImVec2 textSize = ImGui::CalcTextSize(pin.name.c_str());
            drawList->AddText(ImVec2(pinPos.x - textSize.x - 10 * zoom, pinY - 8 * zoom),
                IM_COL32(200, 200, 200, 255), pin.name.c_str());

            pinY += pinSpacing;
        }
    }

    // ========== DRAW EXISTING LINKS ==========
    for (const auto& link : graph->links) {
        auto itStart = g_animGraphUI.pinScreenPositions.find(link.startPinId);
        auto itEnd = g_animGraphUI.pinScreenPositions.find(link.endPinId);
        
        if (itStart == g_animGraphUI.pinScreenPositions.end() || 
            itEnd == g_animGraphUI.pinScreenPositions.end()) continue;
        
        ImVec2 p1 = itStart->second;
        ImVec2 p2 = itEnd->second;

        // Bezier curve control points
        float dist = std::abs(p1.x - p2.x);
        float cpOffset = std::max(dist * 0.5f, 50.0f * zoom);
        ImVec2 cp1 = ImVec2(p1.x + cpOffset, p1.y);
        ImVec2 cp2 = ImVec2(p2.x - cpOffset, p2.y);

        // Check Selection
        bool isSelected = std::find(g_animGraphUI.selectedLinkIds.begin(),
            g_animGraphUI.selectedLinkIds.end(), link.id) != g_animGraphUI.selectedLinkIds.end();

        ImU32 linkColor = isSelected ? IM_COL32(255, 200, 50, 255) : IM_COL32(200, 200, 100, 255);
        float thickness = (isSelected ? 4.0f : 2.5f) * zoom;
        
        drawList->AddBezierCubic(p1, cp1, cp2, p2, linkColor, thickness);

        // Link Hit Testing & Selection
        if (ImGui::IsWindowHovered() && !g_animGraphUI.isCreatingLink) {
            ImVec2 mousePos = ImGui::GetMousePos();
            bool isHovered = false;
            ImVec2 prevP = p1;
            
            // Subdivide curve into segments for hit testing
            const int segs = 20;
            for (int i = 1; i <= segs; ++i) {
                float t = i / (float)segs;
                float u = 1.0f - t;
                float w1 = u*u*u; 
                float w2 = 3*u*u*t; 
                float w3 = 3*u*t*t; 
                float w4 = t*t*t;
                
                ImVec2 p = ImVec2(
                    w1*p1.x + w2*cp1.x + w3*cp2.x + w4*p2.x,
                    w1*p1.y + w2*cp1.y + w3*cp2.y + w4*p2.y);
                
                // Distance Point-to-Segment
                float l2 = (p.x - prevP.x)*(p.x - prevP.x) + (p.y - prevP.y)*(p.y - prevP.y);
                if (l2 > 0.001f) {
                    float t_seg = ((mousePos.x - prevP.x)*(p.x - prevP.x) + (mousePos.y - prevP.y)*(p.y - prevP.y)) / l2;
                    if (t_seg < 0.0f) t_seg = 0.0f;
                    if (t_seg > 1.0f) t_seg = 1.0f;
                    
                    ImVec2 proj = ImVec2(prevP.x + t_seg*(p.x - prevP.x), prevP.y + t_seg*(p.y - prevP.y));
                    float d2 = (mousePos.x - proj.x)*(mousePos.x - proj.x) + (mousePos.y - proj.y)*(mousePos.y - proj.y);
                    
                    if (d2 < 25.0f * zoom * zoom) { // ~5px radius tolerance
                        isHovered = true;
                        break;
                    }
                }
                prevP = p;
            }

            if (isHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                g_animGraphUI.selectedLinkIds.clear();
                // If Ctrl not held, clear node selection too for exclusive selection
                if (!ImGui::GetIO().KeyCtrl) g_animGraphUI.selectedNodeIds.clear();
                g_animGraphUI.selectedLinkIds.push_back(link.id);
            }
        }
    }

    // ========== DRAW CREATING LINK (if dragging) ==========
    if (g_animGraphUI.isCreatingLink && g_animGraphUI.linkStartPinId != 0) {
        auto itStart = g_animGraphUI.pinScreenPositions.find(g_animGraphUI.linkStartPinId);
        if (itStart != g_animGraphUI.pinScreenPositions.end()) {
            ImVec2 p1 = itStart->second;
            ImVec2 p2 = ImGui::GetMousePos();
            
            float dist = std::abs(p1.x - p2.x);
            float cpOffset = std::max(dist * 0.5f, 50.0f * zoom);
            ImVec2 cp1 = ImVec2(p1.x + cpOffset, p1.y);
            ImVec2 cp2 = ImVec2(p2.x - cpOffset, p2.y);
            
            drawList->AddBezierCubic(p1, cp1, cp2, p2, IM_COL32(255, 200, 100, 200), 3.0f * zoom);
        }
    }

    // ========== PIN INTERACTION (hover/click/drag) ==========
    float pinHitRadius = 12.0f * zoom;
    ImVec2 mousePos = ImGui::GetMousePos();
    
    for (auto& node : graph->nodes) {
        // Check input pins
        for (auto& pin : node->inputs) {
            auto it = g_animGraphUI.pinScreenPositions.find(pin.id);
            if (it == g_animGraphUI.pinScreenPositions.end()) continue;
            
            ImVec2 pinPos = it->second;
            float dist = std::hypot(mousePos.x - pinPos.x, mousePos.y - pinPos.y);
            
            if (dist < pinHitRadius) {
                // Hover highlight
                drawList->AddCircle(pinPos, pinHitRadius, IM_COL32(255, 255, 255, 100), 0, 2.0f);
                
                // Complete link on mouse release
                if (g_animGraphUI.isCreatingLink && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    if (g_animGraphUI.linkStartPinId != pin.id && g_animGraphUI.linkStartIsOutput) {
                        // Connect: output -> input
                        graph->connect(g_animGraphUI.linkStartPinId, pin.id);
                    }
                    g_animGraphUI.isCreatingLink = false;
                    g_animGraphUI.linkStartPinId = 0;
                }
                
                // Start drag from input
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    g_animGraphUI.isCreatingLink = true;
                    g_animGraphUI.linkStartPinId = pin.id;
                    g_animGraphUI.linkStartIsOutput = false;
                }
            }
        }
        
        // Check output pins
        for (auto& pin : node->outputs) {
            auto it = g_animGraphUI.pinScreenPositions.find(pin.id);
            if (it == g_animGraphUI.pinScreenPositions.end()) continue;
            
            ImVec2 pinPos = it->second;
            float dist = std::hypot(mousePos.x - pinPos.x, mousePos.y - pinPos.y);
            
            if (dist < pinHitRadius) {
                // Hover highlight
                drawList->AddCircle(pinPos, pinHitRadius, IM_COL32(255, 255, 255, 100), 0, 2.0f);
                
                // Complete link on mouse release
                if (g_animGraphUI.isCreatingLink && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    if (g_animGraphUI.linkStartPinId != pin.id && !g_animGraphUI.linkStartIsOutput) {
                        // Connect: output -> input (reverse direction)
                        graph->connect(pin.id, g_animGraphUI.linkStartPinId);
                    }
                    g_animGraphUI.isCreatingLink = false;
                    g_animGraphUI.linkStartPinId = 0;
                }
                
                // Start drag from output
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    g_animGraphUI.isCreatingLink = true;
                    g_animGraphUI.linkStartPinId = pin.id;
                    g_animGraphUI.linkStartIsOutput = true;
                }
            }
        }
    }

    // ========== CANCEL LINK CREATION ==========
    if (g_animGraphUI.isCreatingLink) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            g_animGraphUI.isCreatingLink = false;
            g_animGraphUI.linkStartPinId = 0;
        }
        
        // Complete on release (if not over a valid pin, cancel)
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            g_animGraphUI.isCreatingLink = false;
            g_animGraphUI.linkStartPinId = 0;
        }
    }

    // ========== PANNING (middle mouse) ==========
    if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        g_animGraphUI.canvasOffset.x += ImGui::GetIO().MouseDelta.x;
        g_animGraphUI.canvasOffset.y += ImGui::GetIO().MouseDelta.y;
    }
    
    // ========== ZOOM (mouse wheel) ==========
    if (ImGui::IsWindowHovered()) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            ImVec2 mouseCanvasPos = ImVec2(
                mousePos.x - canvasPos.x - g_animGraphUI.canvasOffset.x,
                mousePos.y - canvasPos.y - g_animGraphUI.canvasOffset.y);
            
            float oldZoom = g_animGraphUI.canvasZoom;
            g_animGraphUI.canvasZoom += wheel * 0.1f;
            g_animGraphUI.canvasZoom = std::clamp(g_animGraphUI.canvasZoom, 0.25f, 3.0f);
            
            float zoomRatio = g_animGraphUI.canvasZoom / oldZoom;
            g_animGraphUI.canvasOffset.x = mousePos.x - canvasPos.x - mouseCanvasPos.x * zoomRatio;
            g_animGraphUI.canvasOffset.y = mousePos.y - canvasPos.y - mouseCanvasPos.y * zoomRatio;
        }
    }

    // ========== NODE DRAGGING (only if not creating link) ==========
    if (!g_animGraphUI.isCreatingLink && ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        for (uint32_t nodeId : g_animGraphUI.selectedNodeIds) {
            auto* node = graph->findNodeById(nodeId);
            if (node) {
                node->x += ImGui::GetIO().MouseDelta.x / zoom;
                node->y += ImGui::GetIO().MouseDelta.y / zoom;
            }
        }
    }

    // ========== NODE SELECTION ==========
    if (!g_animGraphUI.isCreatingLink && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        bool clickedOnNode = false;
        bool clickedOnPin = false;
        
        // Check if clicked on a pin first
        for (auto& [pinId, pinPos] : g_animGraphUI.pinScreenPositions) {
            if (std::hypot(mousePos.x - pinPos.x, mousePos.y - pinPos.y) < pinHitRadius) {
                clickedOnPin = true;
                break;
            }
        }
        
        if (!clickedOnPin) {
            for (auto& node : graph->nodes) {
                ImVec2 nodePos = ImVec2(
                    canvasPos.x + g_animGraphUI.canvasOffset.x + node->x * zoom,
                    canvasPos.y + g_animGraphUI.canvasOffset.y + node->y * zoom);

                float nodeWidth = 150.0f * zoom;
                float nodeHeight = (60.0f + node->inputs.size() * 20.0f + node->outputs.size() * 20.0f) * zoom;

                if (mousePos.x >= nodePos.x && mousePos.x <= nodePos.x + nodeWidth &&
                    mousePos.y >= nodePos.y && mousePos.y <= nodePos.y + nodeHeight) {

                    if (!ImGui::GetIO().KeyCtrl) {
                        g_animGraphUI.selectedNodeIds.clear();
                    }

                    auto it = std::find(g_animGraphUI.selectedNodeIds.begin(),
                        g_animGraphUI.selectedNodeIds.end(), node->id);
                    if (it == g_animGraphUI.selectedNodeIds.end()) {
                        g_animGraphUI.selectedNodeIds.push_back(node->id);
                    }

                    clickedOnNode = true;
                    break;
                }
            }

            if (!clickedOnNode && !ImGui::GetIO().KeyCtrl) {
                g_animGraphUI.selectedNodeIds.clear();
            }
        }
    }

    // ========== CONTEXT MENU (right-click) ==========
    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !g_animGraphUI.isCreatingLink) {
        g_animGraphUI.showAddNodePopup = true;
        g_animGraphUI.addNodePopupPos = ImGui::GetMousePos();
    }

    if (g_animGraphUI.showAddNodePopup) {
        ImGui::OpenPopup("AddNodePopup");
        g_animGraphUI.showAddNodePopup = false;
    }

    if (ImGui::BeginPopup("AddNodePopup")) {
        ImGui::Text("Add Node");
        ImGui::Separator();

        auto nodeTypes = AnimationGraph::AnimationNodeGraph::getAvailableNodeTypes();

        for (const auto& [typeId, displayName] : nodeTypes) {
            if (ImGui::MenuItem(displayName.c_str())) {
                auto newNode = AnimationGraph::AnimationNodeGraph::createNodeByType(typeId);
                if (newNode) {
                    newNode->x = (g_animGraphUI.addNodePopupPos.x - canvasPos.x - g_animGraphUI.canvasOffset.x) / zoom;
                    newNode->y = (g_animGraphUI.addNodePopupPos.y - canvasPos.y - g_animGraphUI.canvasOffset.y) / zoom;

                    newNode->id = graph->nextNodeId++;
                    for (auto& pin : newNode->inputs) {
                        pin.id = graph->nextPinId++;
                        pin.nodeId = newNode->id;
                    }
                    for (auto& pin : newNode->outputs) {
                        pin.id = graph->nextPinId++;
                        pin.nodeId = newNode->id;
                    }

                    graph->nodes.push_back(std::move(newNode));
                }
            }
        }

        ImGui::EndPopup();
    }
    
    // ========== DELETE KEY - Remove Selected Nodes & Links ==========
    if (ImGui::IsWindowHovered() && (!g_animGraphUI.selectedNodeIds.empty() || !g_animGraphUI.selectedLinkIds.empty())) {
        if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X)) {
            
            // 1. Delete Explicitly Selected Links
            if (!g_animGraphUI.selectedLinkIds.empty()) {
                graph->links.erase(
                    std::remove_if(graph->links.begin(), graph->links.end(),
                        [&](const auto& link) {
                            return std::find(g_animGraphUI.selectedLinkIds.begin(),
                                           g_animGraphUI.selectedLinkIds.end(),
                                           link.id) != g_animGraphUI.selectedLinkIds.end();
                        }),
                    graph->links.end());
                g_animGraphUI.selectedLinkIds.clear();
            }

            // 2. Delete Selected Nodes
            if (!g_animGraphUI.selectedNodeIds.empty()) {
                for (uint32_t nodeId : g_animGraphUI.selectedNodeIds) {
                    // Remove links connected to this node
                    graph->links.erase(
                        std::remove_if(graph->links.begin(), graph->links.end(),
                            [&](const auto& link) {
                                auto* startNode = graph->findNodeByPinId(link.startPinId);
                                auto* endNode = graph->findNodeByPinId(link.endPinId);
                                return (startNode && startNode->id == nodeId) || 
                                       (endNode && endNode->id == nodeId);
                            }),
                        graph->links.end());
                    
                    // Remove node
                    graph->nodes.erase(
                        std::remove_if(graph->nodes.begin(), graph->nodes.end(),
                            [nodeId](const auto& n) { return n->id == nodeId; }),
                        graph->nodes.end());
                }
                g_animGraphUI.selectedNodeIds.clear();
            }
        }
    }
}

// ============================================================================
// ANIMATION GRAPH PANEL (Embedded in Bottom Panel - like Terrain Graph)
// ============================================================================

// External reference for AnimClipNodes to see all scene animations
inline std::vector<std::shared_ptr<AnimationData>>* g_uiClipsRef = nullptr;

inline void drawAnimationGraphPanel(UIContext& ctx) {
    // Set global reference for AnimClipNodes to list all available clips
    g_uiClipsRef = &ctx.scene.animationDataList;

    // 1. BIDIRECTIONAL SELECTION SYNC: Viewport -> UI
    if (ctx.selection.hasSelection() && ctx.selection.selected.type == SelectableType::Object) {
        std::string selName = ctx.selection.selected.name;
        
        // If it's a member of an imported model, use model name instead
        for (auto& mctx : ctx.scene.importedModelContexts) {
            for (auto& member : mctx.members) {
                auto tri = std::dynamic_pointer_cast<Triangle>(member);
                if (tri && tri->getNodeName() == selName) {
                    selName = mctx.importName;
                    break;
                }
            }
        }

        // Only auto-switch if we have a graph or it's a known character
        if (g_animGraphUI.activeCharacter != selName) {
            bool hasGraph = g_animGraphUI.graphs.count(selName);
            bool isModel = false;
            for (auto& mctx : ctx.scene.importedModelContexts) { if (mctx.importName == selName) { isModel = true; break; } }
            
            if (hasGraph || isModel) {
                g_animGraphUI.activeCharacter = selName;
            }
        }
    }

    // Character Selector
    ImGui::Text("Active Character:"); ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##CharSelect", g_animGraphUI.activeCharacter.empty() ? "(None Selected)" : g_animGraphUI.activeCharacter.c_str())) {
        
        // 1. List Imported Models (Characters)
        if (!ctx.scene.importedModelContexts.empty()) {
            ImGui::SeparatorText("Characters");
            for (auto& mctx : ctx.scene.importedModelContexts) {
                bool isSelected = (g_animGraphUI.activeCharacter == mctx.importName);
                if (ImGui::Selectable(mctx.importName.c_str(), isSelected)) {
                    g_animGraphUI.activeCharacter = mctx.importName;
                    
                    // 2. BIDIRECTIONAL SELECTION SYNC: UI -> Viewport
                    if (!mctx.members.empty() && mctx.members[0]) {
                        auto tri = std::dynamic_pointer_cast<Triangle>(mctx.members[0]);
                        if(tri) {
                            ctx.selection.selectObject(tri, -1, mctx.importName);
                        }
                    }
                }
            }
        }

        // 2. List Static/Other Objects (that might have anim graphs)
        ImGui::SeparatorText("Static Objects");
        std::set<std::string> uniqueStaticNames;
        for (auto& obj : ctx.scene.world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri) {
                std::string name = tri->getNodeName();
                if (name.empty()) continue;
                
                // Skip if it's already in a model context
                bool isModelMember = false;
                for (auto& mctx : ctx.scene.importedModelContexts) {
                    for (auto& member : mctx.members) {
                        auto memTri = std::dynamic_pointer_cast<Triangle>(member);
                        if (memTri && memTri->getNodeName() == name) { isModelMember = true; break; }
                    }
                    if (isModelMember) break;
                }
                
                if (!isModelMember) uniqueStaticNames.insert(name);
            }
        }

        for (const auto& name : uniqueStaticNames) {
            bool isSelected = (g_animGraphUI.activeCharacter == name);
            if (ImGui::Selectable(name.c_str(), isSelected)) {
                g_animGraphUI.activeCharacter = name;
                
                // Select in viewport
                for (auto& obj : ctx.scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (tri && tri->getNodeName() == name) {
                        ctx.selection.selectObject(tri, -1, name);
                        break;
                    }
                }
            }
        }
        
        ImGui::Separator();
        if (ImGui::Selectable("Global / Legacy", g_animGraphUI.activeCharacter.empty())) {
            g_animGraphUI.activeCharacter = "";
            ctx.selection.clearSelection();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    
    // Model Settings (Visibility, Root Motion)
    if (!g_animGraphUI.activeCharacter.empty()) {
        for (auto& mctx : ctx.scene.importedModelContexts) {
            if (mctx.importName == g_animGraphUI.activeCharacter) {
                ImGui::Checkbox("Use Root Motion", &mctx.useRootMotion);
                ImGui::SameLine();
                ImGui::Checkbox("Use Anim Graph", &mctx.useAnimGraph);
                ImGui::SameLine();
                ImGui::Checkbox("Visible", &mctx.visible);
                break;
            }
        }
    }

    // Get or create graph for current character
    AnimationGraph::AnimationNodeGraph* currentGraph = nullptr;
    if (!g_animGraphUI.activeCharacter.empty()) {
        auto it = g_animGraphUI.graphs.find(g_animGraphUI.activeCharacter);
        if (it != g_animGraphUI.graphs.end()) {
            currentGraph = it->second.get();
        }
    }
    
    // ========== TOOLBAR AT TOP ==========
    // Toolbar (horizontal layout at top)
    {
        // Quick create button (only if no graph)
        if (!currentGraph) {
            if (ImGui::Button("Create Graph", ImVec2(100, 22))) {
                auto graph = std::make_unique<AnimationGraph::AnimationNodeGraph>();

                auto* outputNode = graph->addNode<AnimationGraph::FinalPoseNode>();
                outputNode->x = 400.0f;
                outputNode->y = 100.0f;

                auto* clipNode = graph->addNode<AnimationGraph::AnimClipNode>();
                clipNode->x = 100.0f;
                clipNode->y = 100.0f;

                if (!clipNode->outputs.empty() && !outputNode->inputs.empty()) {
                    graph->connect(clipNode->outputs[0].id, outputNode->inputs[0].id);
                }

                g_animGraphUI.activeCharacter = "Default";
                g_animGraphUI.graphs["Default"] = std::move(graph);
            }
            ImGui::SameLine();
        }

        // Add node button
        if (ImGui::Button("Add Node", ImVec2(80, 22))) {
            ImGui::OpenPopup("AddAnimNodePopupToolbar");
        }

        ImGui::SameLine();

        // Sync Scene button
        if (ImGui::Button("Sync Scene", ImVec2(90, 22))) {
            auto& animCtrl = AnimationController::getInstance();
            animCtrl.registerClips(ctx.scene.animationDataList);

            for (const auto& anim : ctx.scene.animationDataList) {
                if (!anim) continue;
                if (!anim->name.empty() && g_animGraphUI.graphs.find(anim->name) == g_animGraphUI.graphs.end()) {
                    auto graph = std::make_unique<AnimationGraph::AnimationNodeGraph>();

                    auto* outputNode = graph->addNode<AnimationGraph::FinalPoseNode>();
                    outputNode->x = 400.0f;
                    outputNode->y = 100.0f;

                    auto* clipNode = graph->addNode<AnimationGraph::AnimClipNode>();
                    clipNode->clipName = anim->name;
                    clipNode->x = 100.0f;
                    clipNode->y = 100.0f;

                    if (!clipNode->outputs.empty() && !outputNode->inputs.empty()) {
                        graph->connect(clipNode->outputs[0].id, outputNode->inputs[0].id);
                    }

                    g_animGraphUI.graphs[anim->name] = std::move(graph);

                    if (g_animGraphUI.activeCharacter.empty()) {
                        g_animGraphUI.activeCharacter = anim->name;
                    }
                }
            }
        }

        ImGui::SameLine();

        // Reset View button
        if (ImGui::Button("Reset View", ImVec2(80, 22))) {
            g_animGraphUI.canvasOffset = ImVec2(0, 0);
            g_animGraphUI.canvasZoom = 1.0f;
        }

        ImGui::SameLine();
        ImGui::Text("Zoom: %.0f%%", g_animGraphUI.canvasZoom * 100.0f);

        // Debug info
        if (currentGraph) {
            ImGui::SameLine();
            ImGui::TextDisabled("| Nodes: %zu Links: %zu", currentGraph->nodes.size(), currentGraph->links.size());
        }

        // Add Node popup (from toolbar)
        if (ImGui::BeginPopup("AddAnimNodePopupToolbar")) {
            ImGui::Text("Select Node Type");
            ImGui::Separator();

            auto nodeTypes = AnimationGraph::AnimationNodeGraph::getAvailableNodeTypes();
            for (const auto& [typeId, displayName] : nodeTypes) {
                if (ImGui::MenuItem(displayName.c_str())) {
                    if (g_animGraphUI.activeCharacter.empty()) {
                        g_animGraphUI.activeCharacter = "Default";
                        g_animGraphUI.graphs["Default"] = std::make_unique<AnimationGraph::AnimationNodeGraph>();
                    }

                    auto& graph = g_animGraphUI.graphs[g_animGraphUI.activeCharacter];
                    auto node = AnimationGraph::AnimationNodeGraph::createNodeByType(typeId);
                    if (node) {
                        node->x = 100.0f + graph->nodes.size() * 60.0f;
                        node->y = 100.0f;
                        node->id = graph->nextNodeId++;
                        for (auto& pin : node->inputs) {
                            pin.id = graph->nextPinId++;
                            pin.nodeId = node->id;
                        }
                        for (auto& pin : node->outputs) {
                            pin.id = graph->nextPinId++;
                            pin.nodeId = node->id;
                        }
                        graph->nodes.push_back(std::move(node));
                    }
                }
            }
            ImGui::EndPopup();
        }



        // Auto-create graph for active character if missing
        if (!g_animGraphUI.activeCharacter.empty() &&
            g_animGraphUI.graphs.find(g_animGraphUI.activeCharacter) == g_animGraphUI.graphs.end()) {

            g_animGraphUI.graphs[g_animGraphUI.activeCharacter] = std::make_unique<AnimationGraph::AnimationNodeGraph>();
            auto& graph = g_animGraphUI.graphs[g_animGraphUI.activeCharacter];

            // Add default Final Pose node
            auto finalNode = std::make_unique<AnimationGraph::FinalPoseNode>();
            finalNode->x = 600; finalNode->y = 300;
            finalNode->id = graph->nextNodeId++;
            finalNode->inputs[0].id = graph->nextPinId++;
            finalNode->inputs[0].nodeId = finalNode->id;

            graph->outputNode = finalNode.get();
            graph->nodes.push_back(std::move(finalNode));
        }

        ImGui::Separator();

        // ========== MAIN CONTENT: Left Panel | Resize Handle | Node Canvas ==========
        static float leftPanelWidth = 250.0f;
        const float minPanelWidth = 150.0f;
        const float maxPanelWidth = 400.0f;

        float availHeight = ImGui::GetContentRegionAvail().y;

        // Left panel
        ImGui::BeginChild("AnimLeftPanel", ImVec2(leftPanelWidth, availHeight), true);

        // Character selector
        ImGui::Text("Character:");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##AnimCharSelect", g_animGraphUI.activeCharacter.c_str())) {
            for (auto& [name, graph] : g_animGraphUI.graphs) {
                bool isSelected = (name == g_animGraphUI.activeCharacter);
                if (ImGui::Selectable(name.c_str(), isSelected)) {
                    g_animGraphUI.activeCharacter = name;
                }
            }
            ImGui::EndCombo();
        }

        ImGui::Separator();

        // Clips panel
        drawAnimationClipsPanel(ctx);

        // Playback controls
        drawAnimationPlaybackControls(ctx);

        // Parameters panel
        if (g_animGraphUI.showParameterPanel && currentGraph) {
            drawAnimationParametersPanel(ctx, currentGraph);
        }

        ImGui::EndChild();

        ImGui::SameLine();

        // Resize handle
        ImGui::InvisibleButton("##AnimPanelResize", ImVec2(6.0f, availHeight));
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        }
        if (ImGui::IsItemActive()) {
            leftPanelWidth += ImGui::GetIO().MouseDelta.x;
            leftPanelWidth = std::clamp(leftPanelWidth, minPanelWidth, maxPanelWidth);
        }

        // Draw resize handle indicator
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 handleMin = ImGui::GetItemRectMin();
        ImVec2 handleMax = ImGui::GetItemRectMax();
        drawList->AddRectFilled(handleMin, handleMax,
            ImGui::IsItemHovered() ? IM_COL32(100, 100, 100, 255) : IM_COL32(60, 60, 60, 255));

        ImGui::SameLine();

        // Node canvas
        ImGui::BeginChild("AnimNodeCanvas", ImVec2(0, availHeight), true, ImGuiWindowFlags_NoScrollbar);

        if (currentGraph) {
            drawNodeCanvas(ctx, currentGraph);
        }
        else {
            ImGui::Text("No animation graph loaded.");
            ImGui::Text("Use 'Sync Scene' or 'Create Graph' from toolbar.");
        }

        ImGui::EndChild();
    }
}

// ============================================================================
// MAIN ANIMATION EDITOR PANEL
// ============================================================================

inline void drawAnimationEditorPanel(UIContext& ctx) {
    // 1. BIDIRECTIONAL SELECTION SYNC: Viewport -> UI
    if (ctx.selection.hasSelection() && ctx.selection.selected.type == SelectableType::Object) {
        std::string selName = ctx.selection.selected.name;
        
        // If it's a member of an imported model, use model name instead
        for (auto& mctx : ctx.scene.importedModelContexts) {
            for (auto& member : mctx.members) {
                auto tri = std::dynamic_pointer_cast<Triangle>(member);
                if (tri && tri->getNodeName() == selName) {
                    selName = mctx.importName;
                    break;
                }
            }
        }

        // Only auto-switch if we have a graph or it's a known character
        if (g_animGraphUI.activeCharacter != selName) {
            bool hasGraph = g_animGraphUI.graphs.count(selName);
            bool isModel = false;
            for (auto& mctx : ctx.scene.importedModelContexts) { if (mctx.importName == selName) { isModel = true; break; } }
            
            if (hasGraph || isModel) {
                g_animGraphUI.activeCharacter = selName;
            }
        }
    }

    if (!ImGui::Begin("Animation Editor", nullptr, ImGuiWindowFlags_MenuBar)) {
        ImGui::End();
        return;
    }
    
    // Menu bar
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Graph")) {
                // Create new graph for active character
                // TODO
            }
            if (ImGui::MenuItem("Save Graph")) {
                // Save current graph
                // TODO
            }
            if (ImGui::MenuItem("Load Graph")) {
                // Load graph
                // TODO
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Parameters", nullptr, &g_animGraphUI.showParameterPanel);
            ImGui::MenuItem("Preview", nullptr, &g_animGraphUI.showPreviewPanel);
            ImGui::MenuItem("Debug Info", nullptr, &g_animGraphUI.showDebugInfo);
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Add Node")) {
            auto nodeTypes = AnimationGraph::AnimationNodeGraph::getAvailableNodeTypes();
            
            for (const auto& [typeId, displayName] : nodeTypes) {
                if (ImGui::MenuItem(displayName.c_str())) {
                    // Add node to current graph
                    AnimationGraph::AnimationNodeGraph* graph = nullptr;
                    if (!g_animGraphUI.activeCharacter.empty()) {
                        auto it = g_animGraphUI.graphs.find(g_animGraphUI.activeCharacter);
                        if (it != g_animGraphUI.graphs.end()) {
                            graph = it->second.get();
                        }
                    }
                    
                    if (graph) {
                        auto node = AnimationGraph::AnimationNodeGraph::createNodeByType(typeId);
                        if (node) {
                            node->x = 100.0f;
                            node->y = 100.0f;
                            // Assign IDs and add to graph
                            node->id = graph->nextNodeId++;
                            for (auto& pin : node->inputs) {
                                pin.id = graph->nextPinId++;
                                pin.nodeId = node->id;
                            }
                            for (auto& pin : node->outputs) {
                                pin.id = graph->nextPinId++;
                                pin.nodeId = node->id;
                            }
                            graph->nodes.push_back(std::move(node));
                        }
                    }
                }
            }
            ImGui::EndMenu();
        }
        
        ImGui::EndMenuBar();
    }
    
    // Get or create graph for current character
    AnimationGraph::AnimationNodeGraph* currentGraph = nullptr;
    if (!g_animGraphUI.activeCharacter.empty()) {
        auto it = g_animGraphUI.graphs.find(g_animGraphUI.activeCharacter);
        if (it != g_animGraphUI.graphs.end()) {
            currentGraph = it->second.get();
        }
    }
    
    // Layout: Left panel (params/clips) | Node canvas
    float panelWidth = 250.0f;
    
    // Left panel
    ImGui::BeginChild("LeftPanel", ImVec2(panelWidth, 0), true);
    
    // Character selector
    ImGui::Text("Character:");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##CharSelect", g_animGraphUI.activeCharacter.empty() ? "Select Object..." : g_animGraphUI.activeCharacter.c_str())) {
        // List from graphs (logical names)
        if (!g_animGraphUI.graphs.empty()) {
            ImGui::SeparatorText("Active Graphs");
            for (auto& [name, graph] : g_animGraphUI.graphs) {
                bool isSelected = (name == g_animGraphUI.activeCharacter);
                if (ImGui::Selectable(name.c_str(), isSelected)) {
                    g_animGraphUI.activeCharacter = name;
                    // Auto-selection removed as per request
                }
            }
        }
        
        // Quick list of all characters in scene
        if (!ctx.scene.importedModelContexts.empty()) {
            ImGui::SeparatorText("Scene Characters");
            for (auto& mctx : ctx.scene.importedModelContexts) {
                if (g_animGraphUI.graphs.count(mctx.importName)) continue; // Already listed
                if (ImGui::Selectable(mctx.importName.c_str())) {
                    g_animGraphUI.activeCharacter = mctx.importName;
                    // Auto-selection removed as per request
                }
            }
        }

        ImGui::EndCombo();
    }
    
    ImGui::Separator();
    
    // Clips panel
    drawAnimationClipsPanel(ctx);
    
    // Playback controls
    drawAnimationPlaybackControls(ctx);
    
    // Parameters panel
    if (g_animGraphUI.showParameterPanel && currentGraph) {
        drawAnimationParametersPanel(ctx, currentGraph);
    }
    
    // State machine panel
    if (currentGraph) {
        drawStateMachinePanel(ctx, currentGraph);
    }
    
    ImGui::EndChild();
    
    ImGui::SameLine();
    
    // Node canvas
    ImGui::BeginChild("NodeCanvas", ImVec2(0, 0), true, ImGuiWindowFlags_NoScrollbar);
    
    if (currentGraph) {
        drawNodeCanvas(ctx, currentGraph);
    } else {
        ImGui::Text("No animation graph loaded.");
        ImGui::Text("Select a skinned character to create a graph.");
        
        // Quick create button
        if (ImGui::Button("Create Default Graph")) {
            // Create a simple default graph
            auto graph = std::make_unique<AnimationGraph::AnimationNodeGraph>();
            
            // Add output node
            auto* outputNode = graph->addNode<AnimationGraph::FinalPoseNode>();
            outputNode->x = 400.0f;
            outputNode->y = 200.0f;
            
            // Add a clip node
            auto* clipNode = graph->addNode<AnimationGraph::AnimClipNode>();
            clipNode->x = 100.0f;
            clipNode->y = 200.0f;
            
            // Connect them
            if (!clipNode->outputs.empty() && !outputNode->inputs.empty()) {
                graph->connect(clipNode->outputs[0].id, outputNode->inputs[0].id);
            }
            
            g_animGraphUI.activeCharacter = "Default";
            g_animGraphUI.graphs["Default"] = std::move(graph);
        }
    }
    
    ImGui::EndChild();
    
    // Debug info
    if (g_animGraphUI.showDebugInfo && currentGraph) {
        ImGui::Separator();
        ImGui::Text("Nodes: %zu | Links: %zu", 
            currentGraph->nodes.size(), currentGraph->links.size());
        ImGui::Text("Global Time: %.2fs", currentGraph->evalContext.globalTime);
    }
    
    ImGui::End();
}


// ============================================================================
// INTEGRATION: Animation Tab for Properties Panel
// ============================================================================

inline void drawAnimationPropertiesTab(UIContext& ctx) {
    // Quick controls without full editor
    ImGui::Text("Animation");
    ImGui::Separator();
    
    auto& animCtrl = AnimationController::getInstance();
    
    // Current playback state
    std::string currentClip = animCtrl.getCurrentClipName();
    ImGui::Text("Playing: %s", currentClip.empty() ? "(none)" : currentClip.c_str());
    
    float progress = animCtrl.getNormalizedTime();
    ImGui::ProgressBar(progress, ImVec2(-1, 0));
    
    // Clips list
    const auto& clips = animCtrl.getAllClips();
    
    static int selectedClipIdx = 0;
    if (!clips.empty()) {
        std::vector<const char*> clipNames;
        for (const auto& c : clips) {
            clipNames.push_back(c.name.c_str());
        }
        
        if (ImGui::Combo("Clip", &selectedClipIdx, clipNames.data(), (int)clipNames.size())) {
            animCtrl.play(clips[selectedClipIdx].name, 0.3f);
        }
    }
    
    // Blend parameters
    ImGui::Separator();
    ImGui::Text("Blend Parameters");
    
    auto& evalCtx = AnimationController::getInstance();
    // Add common parameters here
    // e.g., Speed, Direction, etc.
    
    // Open full editor button
    ImGui::Separator();
    if (ImGui::Button("Open Animation Editor", ImVec2(-1, 0))) {
        g_animGraphUI.showNodeEditor = true;
    }
}

