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
#include "imgui.h"
#include <string>
#include <vector>

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
                ImGui::SliderFloat(name.c_str(), &value, 0.0f, 1.0f);
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
    auto& animCtrl = AnimationController::getInstance();
    
    // Auto-sync animations from scene if controller is empty
    if (animCtrl.getAllClips().empty() && !ctx.scene.animationDataList.empty()) {
        animCtrl.registerClips(ctx.scene.animationDataList);
    }
    
    const auto& clips = animCtrl.getAllClips();
    
    ImGui::BeginChild("AnimClips", ImVec2(0, 140), true);
    ImGui::Text("Animations (%zu)", clips.size());
    
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
            
            bool isPlaying = (animCtrl.getCurrentClipName() == clip.name);
            
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
                    animCtrl.stopAll();
                } else {
                    animCtrl.play(clip.name, 0.2f);
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
    auto& animCtrl = AnimationController::getInstance();
    
    ImGui::BeginChild("AnimPlayback", ImVec2(0, 80), true);
    
    // Current state
    std::string currentClip = animCtrl.getCurrentClipName();
    float currentTime = animCtrl.getCurrentTime();
    float normalizedTime = animCtrl.getNormalizedTime();
    
    ImGui::Text("Current: %s", currentClip.empty() ? "(none)" : currentClip.c_str());
    
    // Progress bar
    ImGui::ProgressBar(normalizedTime, ImVec2(-1, 0), 
        (std::to_string((int)(normalizedTime * 100)) + "%").c_str());
    
    // Playback controls
    static bool isPaused = false;
    
    if (ImGui::Button(isPaused ? ">" : "||", ImVec2(30, 25))) {
        isPaused = !isPaused;
        animCtrl.setPaused(isPaused);
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button("<<", ImVec2(30, 25))) {
        animCtrl.setTime(0.0f);
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button("Stop", ImVec2(50, 25))) {
        animCtrl.stopAll();
        isPaused = false;
    }
    
    ImGui::SameLine();
    
    // Blend state indicator
    if (animCtrl.isBlending()) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), " [Blending]");
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

        // Bezier curve
        float dist = std::abs(p1.x - p2.x);
        float cpOffset = std::max(dist * 0.5f, 50.0f * zoom);
        ImVec2 cp1 = ImVec2(p1.x + cpOffset, p1.y);
        ImVec2 cp2 = ImVec2(p2.x - cpOffset, p2.y);

        drawList->AddBezierCubic(p1, cp1, cp2, p2, IM_COL32(200, 200, 100, 255), 2.5f * zoom);
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
    
    // ========== DELETE KEY - Remove Selected Nodes ==========
    if (ImGui::IsWindowHovered() && !g_animGraphUI.selectedNodeIds.empty()) {
        if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X)) {
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

// ============================================================================
// ANIMATION GRAPH PANEL (Embedded in Bottom Panel - like Terrain Graph)
// ============================================================================

inline void drawAnimationGraphPanel(UIContext& ctx) {
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
                if (!anim.name.empty() && g_animGraphUI.graphs.find(anim.name) == g_animGraphUI.graphs.end()) {
                    auto graph = std::make_unique<AnimationGraph::AnimationNodeGraph>();
                    
                    auto* outputNode = graph->addNode<AnimationGraph::FinalPoseNode>();
                    outputNode->x = 400.0f;
                    outputNode->y = 100.0f;
                    
                    auto* clipNode = graph->addNode<AnimationGraph::AnimClipNode>();
                    clipNode->clipName = anim.name;
                    clipNode->x = 100.0f;
                    clipNode->y = 100.0f;
                    
                    if (!clipNode->outputs.empty() && !outputNode->inputs.empty()) {
                        graph->connect(clipNode->outputs[0].id, outputNode->inputs[0].id);
                    }
                    
                    g_animGraphUI.graphs[anim.name] = std::move(graph);
                    
                    if (g_animGraphUI.activeCharacter.empty()) {
                        g_animGraphUI.activeCharacter = anim.name;
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
    } else {
        ImGui::Text("No animation graph loaded.");
        ImGui::Text("Use 'Sync Scene' or 'Create Graph' from toolbar.");
    }
    
    ImGui::EndChild();
}

// ============================================================================
// MAIN ANIMATION EDITOR PANEL
// ============================================================================

inline void drawAnimationEditorPanel(UIContext& ctx) {
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
    if (ImGui::BeginCombo("##CharSelect", g_animGraphUI.activeCharacter.c_str())) {
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
