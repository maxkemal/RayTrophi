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
#include "NodeSystem/NodeEditorChrome.h"

#include <cstdio>
#include <set>
#include "Triangle.h" // Added for dynamic_pointer_cast<Triangle>
#include "imgui.h"
#include <string>
#include <unordered_set>
#include <vector>
#include <SceneSelection.h>

// ============================================================================
// ANIMATION GRAPH UI STATE
// ============================================================================

struct AnimGraphUIState {
    // Graph instance per skeleton/character
    std::unordered_map<std::string, std::shared_ptr<AnimationGraph::AnimationNodeGraph>> graphs;
    
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
    uint32_t resizingNodeId = 0;
    
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
    std::unordered_set<std::string> runtimeStaleAssetKeys;
};

// Global state
inline AnimGraphUIState g_animGraphUI;

inline SceneData::ImportedModelContext* findImportedModelContext(SceneData& scene, const std::string& name) {
    for (auto& mctx : scene.importedModelContexts) {
        if (mctx.importName == name) return &mctx;
    }
    return nullptr;
}

inline std::string getAnimGraphAssetKeyForCharacter(SceneData& scene, const std::string& characterName) {
    if (auto* mctx = findImportedModelContext(scene, characterName)) {
        if (mctx->animGraphAssetKey.empty()) {
            mctx->animGraphAssetKey = mctx->importName.empty() ? characterName : mctx->importName;
        }
        return mctx->animGraphAssetKey;
    }
    return characterName;
}

inline AnimationGraph::AnimationNodeGraph* getRuntimeGraphForCharacter(SceneData& scene, const std::string& characterName) {
    if (auto* mctx = findImportedModelContext(scene, characterName)) {
        if (mctx->runtimeGraph) return mctx->runtimeGraph.get();
        if (mctx->graph) return mctx->graph.get();
    }
    return nullptr;
}

inline void markAnimGraphRuntimeStale(SceneData& scene, const std::string& characterName) {
    if (characterName.empty()) return;
    g_animGraphUI.runtimeStaleAssetKeys.insert(getAnimGraphAssetKeyForCharacter(scene, characterName));
}

inline bool isAnimGraphRuntimeStale(SceneData& scene, const std::string& characterName) {
    if (characterName.empty()) return false;
    return g_animGraphUI.runtimeStaleAssetKeys.find(getAnimGraphAssetKeyForCharacter(scene, characterName)) !=
        g_animGraphUI.runtimeStaleAssetKeys.end();
}

inline void syncRuntimeGraphFromAsset(SceneData& scene, const std::string& characterName) {
    auto* mctx = findImportedModelContext(scene, characterName);
    if (!mctx) return;

    const std::string assetKey = getAnimGraphAssetKeyForCharacter(scene, characterName);
    auto it = g_animGraphUI.graphs.find(assetKey);
    if (it == g_animGraphUI.graphs.end() || !it->second) return;

    mctx->runtimeGraph = it->second->clone();
    mctx->graph = mctx->runtimeGraph;
    g_animGraphUI.runtimeStaleAssetKeys.erase(assetKey);
}

inline float animGraphCanvasMetric(float base, float zoom, float minValue, float maxValue) {
    return std::clamp(base * zoom, minValue, maxValue);
}

inline std::string fitAnimGraphCanvasText(const std::string& text, float maxWidth) {
    if (text.empty() || maxWidth <= 8.0f) return {};
    if (ImGui::CalcTextSize(text.c_str()).x <= maxWidth) return text;

    static const char* kEllipsis = "...";
    std::string result = text;
    while (!result.empty()) {
        result.pop_back();
        std::string candidate = result + kEllipsis;
        if (ImGui::CalcTextSize(candidate.c_str()).x <= maxWidth) {
            return candidate;
        }
    }
    return {};
}

inline void drawAnimGraphCanvasTextClipped(ImDrawList* drawList, const ImVec2& pos, const ImVec2& clipMin,
    const ImVec2& clipMax, ImU32 color, const std::string& text) {
    if (text.empty()) return;
    drawList->PushClipRect(clipMin, clipMax, true);
    drawList->AddText(pos, color, text.c_str());
    drawList->PopClipRect();
}

inline std::vector<std::shared_ptr<AnimationData>> getCharacterAnimationClips(SceneData& scene, const std::string& characterName) {
    std::vector<std::shared_ptr<AnimationData>> result;
    if (characterName.empty()) return result;

    for (const auto& anim : scene.animationDataList) {
        if (!anim) continue;
        if (anim->modelName == characterName || anim->modelName.empty()) {
            result.push_back(anim);
        }
    }

    if (result.empty() && !scene.animationDataList.empty()) {
        result = scene.animationDataList;
    }
    return result;
}

inline void captureAnimClipOverrides(AnimationGraph::AnimationNodeGraph* graph, AnimGraphKeyframe& outKey) {
    if (!graph) return;
    for (const auto& node : graph->nodes) {
        auto* clipNode = dynamic_cast<AnimationGraph::AnimClipNode*>(node.get());
        if (!clipNode) continue;
        if (!clipNode->clipName.empty()) {
            outKey.clip_overrides[clipNode->id] = clipNode->clipName;
        }
        outKey.clip_speed_overrides[clipNode->id] = clipNode->playbackSpeed;
    }
}

inline void drawAnimGraphQuickGuide(bool hasActiveCharacter, bool hasGraph, bool useAnimGraph, bool followTimeline) {
    if (!ImGui::CollapsingHeader("Quick Guide")) return;

    ImGui::TextWrapped("AnimGraph now works in three layers: Asset Graph is the editor definition, Runtime Graph is the live per-character instance, and Timeline drives runtime parameters and events.");
    ImGui::Spacing();

    ImGui::BulletText("1. Select a character in the viewport. Each character owns its own runtime graph.");
    ImGui::BulletText("2. Build the asset graph with Create Graph or Build Demo Rig.");
    ImGui::BulletText("3. Push To Runtime to update the live character instance.");
    ImGui::BulletText("4. Enable Use Anim Graph, then test with playback or Timeline.");
    ImGui::BulletText("5. Add AnimGraph keyframes to Timeline to drive parameters, triggers, or states.");

    ImGui::Spacing();
    ImGui::TextDisabled("Status");
    ImGui::BulletText("Character: %s", hasActiveCharacter ? "Selected" : "None");
    ImGui::BulletText("Asset Graph: %s", hasGraph ? "Ready" : "Missing");
    ImGui::BulletText("Runtime Mode: %s", useAnimGraph ? "AnimGraph enabled" : "Legacy controller");
    ImGui::BulletText("Playback Source: %s", followTimeline ? "Following Timeline" : "Viewport autoplay");
}

inline void showAnimGraphButtonTooltip(const char* text) {
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", text);
    }
}

inline void drawAnimGraphSectionLabel(const char* title, const char* subtitle = nullptr) {
    ImGui::TextColored(ImVec4(0.92f, 0.82f, 0.45f, 1.0f), "%s", title);
    if (subtitle && subtitle[0] != '\0') {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", subtitle);
    }
}

inline void drawAnimGraphToolbarLabel(const char* title) {
    ImGui::TextDisabled("%s", title);
    ImGui::SameLine();
}

inline void drawAnimGraphStatusCard(SceneData& scene, const std::string& characterName, AnimationGraph::AnimationNodeGraph* assetGraph) {
    ImGui::BeginChild("AnimGraphStatusCard", ImVec2(0, 78), true);
    drawAnimGraphSectionLabel("Session", "Asset / Runtime summary");
    ImGui::Separator();

    const bool runtimeStale = isAnimGraphRuntimeStale(scene, characterName);
    const auto* runtimeGraph = getRuntimeGraphForCharacter(scene, characterName);
    ImGui::Text("Character: %s", characterName.empty() ? "(Select in viewport)" : characterName.c_str());
    ImGui::Text("Asset Nodes: %d", assetGraph ? (int)assetGraph->nodes.size() : 0);
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::Text("Runtime: %s", runtimeGraph ? "Bound" : "Missing");
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::TextColored(runtimeStale ? ImVec4(1.0f, 0.55f, 0.35f, 1.0f) : ImVec4(0.45f, 0.9f, 0.55f, 1.0f),
        "%s", runtimeStale ? "Pending Push" : "Live");
    ImGui::EndChild();
}

inline void createAnimGraphDemoRig(SceneData& scene, const std::string& characterName) {
    if (characterName.empty()) return;

    const std::string assetKey = getAnimGraphAssetKeyForCharacter(scene, characterName);
    auto clips = getCharacterAnimationClips(scene, characterName);
    if (clips.empty()) return;

    auto graph = std::make_shared<AnimationGraph::AnimationNodeGraph>();

    auto* finalNode = graph->addNode<AnimationGraph::FinalPoseNode>();
    finalNode->x = 740.0f;
    finalNode->y = 180.0f;

    auto* smNode = graph->addNode<AnimationGraph::StateMachineNode>();
    smNode->x = 470.0f;
    smNode->y = 170.0f;

    auto clipNameFor = [&](size_t index) -> std::string {
        if (clips.empty()) return "";
        if (index < clips.size() && clips[index]) return clips[index]->name;
        return clips.front() ? clips.front()->name : "";
    };

    auto* idleNode = graph->addNode<AnimationGraph::AnimClipNode>();
    idleNode->clipName = clipNameFor(0);
    idleNode->playbackSpeed = 0.85f;
    idleNode->x = 120.0f;
    idleNode->y = 40.0f;

    auto* moveNode = graph->addNode<AnimationGraph::AnimClipNode>();
    moveNode->clipName = clipNameFor(clips.size() > 1 ? 1 : 0);
    moveNode->playbackSpeed = (clips.size() > 1) ? 1.0f : 1.35f;
    moveNode->x = 120.0f;
    moveNode->y = 180.0f;

    auto* attackNode = graph->addNode<AnimationGraph::AnimClipNode>();
    attackNode->clipName = clipNameFor(clips.size() > 2 ? 2 : 0);
    attackNode->playbackSpeed = (clips.size() > 2) ? 1.0f : 1.8f;
    attackNode->loop = clips.size() <= 2;
    attackNode->x = 120.0f;
    attackNode->y = 320.0f;

    smNode->addState("Idle", idleNode->id, true);
    smNode->addState("Move", moveNode->id, false);
    smNode->addState("Action", attackNode->id, false);

    if (!idleNode->outputs.empty() && smNode->inputs.size() >= 1) graph->connect(idleNode->outputs[0].id, smNode->inputs[0].id);
    if (!moveNode->outputs.empty() && smNode->inputs.size() >= 2) graph->connect(moveNode->outputs[0].id, smNode->inputs[1].id);
    if (!attackNode->outputs.empty() && smNode->inputs.size() >= 3) graph->connect(attackNode->outputs[0].id, smNode->inputs[2].id);
    if (!smNode->outputs.empty() && !finalNode->inputs.empty()) graph->connect(smNode->outputs[0].id, finalNode->inputs[0].id);

    AnimationGraph::StateMachineNode::Transition tIdleToMove;
    tIdleToMove.fromState = "Idle";
    tIdleToMove.toState = "Move";
    tIdleToMove.parameterName = "speed";
    tIdleToMove.conditionType = AnimationGraph::StateMachineNode::Transition::ConditionType::FloatGreater;
    tIdleToMove.compareValue = 0.15f;
    tIdleToMove.hasExitTime = false;
    tIdleToMove.blendTime = 0.2f;
    smNode->addTransition(tIdleToMove);

    AnimationGraph::StateMachineNode::Transition tMoveToIdle;
    tMoveToIdle.fromState = "Move";
    tMoveToIdle.toState = "Idle";
    tMoveToIdle.parameterName = "speed";
    tMoveToIdle.conditionType = AnimationGraph::StateMachineNode::Transition::ConditionType::FloatLess;
    tMoveToIdle.compareValue = 0.1f;
    tMoveToIdle.hasExitTime = false;
    tMoveToIdle.blendTime = 0.25f;
    smNode->addTransition(tMoveToIdle);

    AnimationGraph::StateMachineNode::Transition tIdleToAction;
    tIdleToAction.fromState = "Idle";
    tIdleToAction.toState = "Action";
    tIdleToAction.parameterName = "attack";
    tIdleToAction.conditionType = AnimationGraph::StateMachineNode::Transition::ConditionType::Trigger;
    tIdleToAction.hasExitTime = false;
    tIdleToAction.blendTime = 0.12f;
    smNode->addTransition(tIdleToAction);

    AnimationGraph::StateMachineNode::Transition tMoveToAction = tIdleToAction;
    tMoveToAction.fromState = "Move";
    smNode->addTransition(tMoveToAction);

    AnimationGraph::StateMachineNode::Transition tActionToIdle;
    tActionToIdle.fromState = "Action";
    tActionToIdle.toState = "Idle";
    tActionToIdle.parameterName = "speed";
    tActionToIdle.conditionType = AnimationGraph::StateMachineNode::Transition::ConditionType::FloatLess;
    tActionToIdle.compareValue = 0.1f;
    tActionToIdle.hasExitTime = true;
    tActionToIdle.exitTime = 0.85f;
    tActionToIdle.blendTime = 0.18f;
    smNode->addTransition(tActionToIdle);

    AnimationGraph::StateMachineNode::Transition tActionToMove = tActionToIdle;
    tActionToMove.toState = "Move";
    tActionToMove.conditionType = AnimationGraph::StateMachineNode::Transition::ConditionType::FloatGreater;
    tActionToMove.compareValue = 0.1f;
    smNode->addTransition(tActionToMove);

    graph->evalContext.floatParams["speed"] = 0.0f;
    graph->evalContext.boolParams["isCinematic"] = false;
    graph->evalContext.intParams["variant"] = 0;
    graph->evalContext.triggerParams["attack"] = "attack";
    graph->evalContext.triggerParams.clear();

    g_animGraphUI.graphs[assetKey] = graph;
    syncRuntimeGraphFromAsset(scene, characterName);
}

inline void createAnimGraphDemoTimeline(SceneData& scene, const std::string& characterName) {
    if (characterName.empty()) return;

    if (auto* mctx = findImportedModelContext(scene, characterName)) {
        mctx->useAnimGraph = true;
        mctx->animGraphFollowTimeline = true;
    }

    AnimationGraph::AnimationNodeGraph* sourceGraph = getRuntimeGraphForCharacter(scene, characterName);
    if (!sourceGraph) {
        const std::string assetKey = getAnimGraphAssetKeyForCharacter(scene, characterName);
        auto it = g_animGraphUI.graphs.find(assetKey);
        if (it != g_animGraphUI.graphs.end()) sourceGraph = it->second.get();
    }

    const int start = scene.timeline.current_frame;

    Keyframe k0(start + 0);
    k0.has_anim_graph = true;
    k0.anim_graph.float_params["speed"] = 0.0f;
    captureAnimClipOverrides(sourceGraph, k0.anim_graph);
    scene.timeline.insertKeyframe(characterName, k0);

    Keyframe k1(start + 24);
    k1.has_anim_graph = true;
    k1.anim_graph.float_params["speed"] = 1.0f;
    captureAnimClipOverrides(sourceGraph, k1.anim_graph);
    scene.timeline.insertKeyframe(characterName, k1);

    Keyframe k2(start + 48);
    k2.has_anim_graph = true;
    k2.anim_graph.float_params["speed"] = 1.0f;
    k2.anim_graph.triggers.push_back("attack");
    captureAnimClipOverrides(sourceGraph, k2.anim_graph);
    scene.timeline.insertKeyframe(characterName, k2);

    Keyframe k3(start + 72);
    k3.has_anim_graph = true;
    k3.anim_graph.float_params["speed"] = 0.0f;
    captureAnimClipOverrides(sourceGraph, k3.anim_graph);
    scene.timeline.insertKeyframe(characterName, k3);
}

// ============================================================================
// ANIMATION PARAMETERS PANEL
// ============================================================================

inline void drawAnimationParametersPanel(UIContext& ctx,
    AnimationGraph::AnimationNodeGraph* assetGraph,
    AnimationGraph::AnimationNodeGraph* runtimeGraph,
    const std::string& characterName) {
    if (!assetGraph && !runtimeGraph) return;
    
    ImGui::BeginChild("AnimParams", ImVec2(0, 240), true);
    ImGui::Text("Animation Parameters");
    ImGui::Separator();
    
    auto* primaryGraph = runtimeGraph ? runtimeGraph : assetGraph;
    auto& evalCtx = primaryGraph->evalContext;
    
    // Float parameters
    if (!evalCtx.floatParams.empty()) {
        if (ImGui::CollapsingHeader("Float Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto& [name, value] : evalCtx.floatParams) {
                SceneUI::DrawSmartFloat(name.c_str(), name.c_str(), &value, -10.0f, 10.0f, "%.3f", false, nullptr, 16);
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

    if (!evalCtx.intParams.empty()) {
        if (ImGui::CollapsingHeader("Int Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto& [name, value] : evalCtx.intParams) {
                ImGui::InputInt(name.c_str(), &value);
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
            case 3: break;
        }
        newParamName[0] = '\0';
    }
    
    if (ImGui::CollapsingHeader("Triggers", ImGuiTreeNodeFlags_DefaultOpen)) {
        static char triggerName[64] = "";
        ImGui::InputText("Trigger Name", triggerName, sizeof(triggerName));
        if (ImGui::Button("Fire Trigger") && strlen(triggerName) > 0) {
            evalCtx.triggerParams[triggerName] = triggerName;
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear Triggers")) {
            evalCtx.triggerParams.clear();
        }
        if (evalCtx.triggerParams.empty()) {
            ImGui::TextDisabled("No live triggers queued.");
        } else {
            for (auto it = evalCtx.triggerParams.begin(); it != evalCtx.triggerParams.end(); ) {
                ImGui::PushID(it->first.c_str());
                ImGui::BulletText("%s", it->first.c_str());
                ImGui::SameLine();
                if (ImGui::SmallButton("X")) {
                    it = evalCtx.triggerParams.erase(it);
                } else {
                    ++it;
                }
                ImGui::PopID();
            }
        }
    }

    if (!characterName.empty()) {
        ImGui::Separator();
        if (ImGui::Button("Keyframe Params To Timeline")) {
            Keyframe kf(ctx.scene.timeline.current_frame);
            kf.has_anim_graph = true;
            kf.anim_graph.float_params = evalCtx.floatParams;
            kf.anim_graph.bool_params = evalCtx.boolParams;
            kf.anim_graph.int_params = evalCtx.intParams;
            captureAnimClipOverrides(primaryGraph, kf.anim_graph);
            for (const auto& [name, _] : evalCtx.triggerParams) {
                kf.anim_graph.triggers.push_back(name);
            }
            ctx.scene.timeline.insertKeyframe(characterName, kf);
            if (auto* mctx = findImportedModelContext(ctx.scene, characterName)) {
                mctx->useAnimGraph = true;
                mctx->animGraphFollowTimeline = true;
            }
        }
    }

    if (runtimeGraph && assetGraph && runtimeGraph != assetGraph) {
        assetGraph->evalContext.floatParams = runtimeGraph->evalContext.floatParams;
        assetGraph->evalContext.boolParams = runtimeGraph->evalContext.boolParams;
        assetGraph->evalContext.intParams = runtimeGraph->evalContext.intParams;
    }

    ImGui::EndChild();
}

inline void syncRuntimeNodeFromAsset(AnimationGraph::AnimationNodeGraph* assetGraph,
    AnimationGraph::AnimationNodeGraph* runtimeGraph,
    uint32_t nodeId) {
    if (!assetGraph || !runtimeGraph || assetGraph == runtimeGraph) return;

    auto* assetNode = assetGraph->findNodeById(nodeId);
    auto* runtimeNode = runtimeGraph->findNodeById(nodeId);
    if (!assetNode || !runtimeNode) return;
    if (assetNode->getTypeId() != runtimeNode->getTypeId()) return;

    nlohmann::json nodeJson;
    assetNode->onSave(nodeJson);
    runtimeNode->onLoad(nodeJson);
}

// ============================================================================
// NODE PROPERTIES PANEL
// ============================================================================

inline void drawNodePropertiesPanel(UIContext& ctx,
    AnimationGraph::AnimationNodeGraph* graph,
    AnimationGraph::AnimationNodeGraph* runtimeGraph) {
    if (!graph) return;
    
    ImGui::BeginChild("NodeProps", ImVec2(0, 0), true);
    ImGui::Text("Node Properties");
    ImGui::Separator();
    
    if (g_animGraphUI.selectedNodeIds.empty()) {
        ImGui::TextDisabled("Select a node to edit.");
    } else {
        uint32_t nodeId = g_animGraphUI.selectedNodeIds[0];
        auto* node = graph->findNodeById(nodeId);
        if (node) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "[%s]", node->metadata.displayName.c_str());
            ImGui::Separator();
            ImGui::PushID(node->id);
            node->drawContent();
            ImGui::PopID();
            syncRuntimeNodeFromAsset(graph, runtimeGraph, node->id);
            
            // Special UI explicitly for State Machine addition just to make sure we assign correct Pin IDs
            if (node->getTypeId() == "StateMachine") {
                auto* smNode = static_cast<AnimationGraph::StateMachineNode*>(node);
                ImGui::Separator();
                
                ImGui::Text("States Management");
                static char newStateName[64] = "";
                ImGui::InputText("New State", newStateName, sizeof(newStateName));
                
                if (ImGui::Button("Add State") && strlen(newStateName) > 0) {
                    smNode->addState(newStateName, 0, smNode->states.empty());
                    
                    // The pin just added by addState needs an ID!
                    if (!smNode->inputs.empty()) {
                        auto& newPin = smNode->inputs.back();
                        if (newPin.id == 0) {
                            newPin.id = graph->nextPinId++;
                            newPin.nodeId = smNode->id;
                            graph->needsRebuild = true;
                        }
                    }
                    newStateName[0] = '\0';
                }

                ImGui::Separator();
                ImGui::Text("Transitions Management");
                
                // Show existing transitions
                for (size_t t = 0; t < smNode->transitions.size(); ++t) {
                    auto& trans = smNode->transitions[t];
                    ImGui::PushID(static_cast<int>(t));
                    
                    if (ImGui::TreeNodeEx((trans.fromState + " -> " + trans.toState).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::DragFloat("Blend Time", &trans.blendTime, 0.01f, 0.0f, 5.0f, "%.2fs");
                        ImGui::Checkbox("Has Exit Time", &trans.hasExitTime);
                        if (trans.hasExitTime) {
                            ImGui::SliderFloat("Exit Time", &trans.exitTime, 0.0f, 1.0f, "%.2f");
                        }
                        
                        // Gather all available parameters for dropdown
                        std::vector<const char*> availableParams;
                        for (const auto& [name, _] : graph->evalContext.floatParams) availableParams.push_back(name.c_str());
                        for (const auto& [name, _] : graph->evalContext.boolParams) availableParams.push_back(name.c_str());
                        for (const auto& [name, _] : graph->evalContext.triggerParams) availableParams.push_back(name.c_str());
                        availableParams.push_back("(Type Manual)"); // Fallback option
                        
                        int currentParamIdx = availableParams.size() - 1; // Default to Type Manual
                        for (int i = 0; i < (int)availableParams.size() - 1; ++i) {
                            if (trans.parameterName == availableParams[i]) {
                                currentParamIdx = i;
                                break;
                            }
                        }

                        if (ImGui::Combo("Condition Param", &currentParamIdx, availableParams.data(), availableParams.size())) {
                            if (currentParamIdx < (int)availableParams.size() - 1) {
                                trans.parameterName = availableParams[currentParamIdx];
                            }
                        }

                        // Also allow manual typing just in case they haven't created it yet
                        if (currentParamIdx == (int)availableParams.size() - 1) {
                            char paramBuf[64];
                            strncpy(paramBuf, trans.parameterName.c_str(), sizeof(paramBuf));
                            if (ImGui::InputText("Condition Param (Manual)", paramBuf, sizeof(paramBuf))) {
                                trans.parameterName = paramBuf;
                            }
                        }

                        const char* conds[] = {"Bool", "Float >", "Float <", "Trigger"};
                        int cType = (int)trans.conditionType;
                        if (ImGui::Combo("Condition Type", &cType, conds, 4)) trans.conditionType = (AnimationGraph::StateMachineNode::Transition::ConditionType)cType;
                        
                        if (trans.conditionType == AnimationGraph::StateMachineNode::Transition::ConditionType::FloatGreater || 
                            trans.conditionType == AnimationGraph::StateMachineNode::Transition::ConditionType::FloatLess) {
                            ImGui::DragFloat("Compare Val", &trans.compareValue, 0.1f);
                        }
                        
                        if (ImGui::Button("Delete Transition")) {
                            smNode->transitions.erase(smNode->transitions.begin() + t);
                        }
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                    // If we deleted, break the loop to avoid invalid access (will refresh next frame)
                    if (t >= smNode->transitions.size()) break; 
                }
                
                // Add new transition
                ImGui::Separator();
                static int fromStateIdx = 0;
                static int toStateIdx = 0;
                
                if (smNode->states.size() >= 2) {
                    std::vector<const char*> stateNames;
                    for (const auto& s : smNode->states) stateNames.push_back(s.name.c_str());
                    
                    fromStateIdx = std::clamp(fromStateIdx, 0, (int)stateNames.size() - 1);
                    toStateIdx = std::clamp(toStateIdx, 0, (int)stateNames.size() - 1);

                    ImGui::Combo("From State", &fromStateIdx, stateNames.data(), stateNames.size());
                    ImGui::Combo("To State", &toStateIdx, stateNames.data(), stateNames.size());
                    
                    if (ImGui::Button("Add Transition") && fromStateIdx != toStateIdx) {
                        AnimationGraph::StateMachineNode::Transition newTrans;
                        newTrans.fromState = smNode->states[fromStateIdx].name;
                        newTrans.toState = smNode->states[toStateIdx].name;
                        newTrans.blendTime = 0.3f;
                        newTrans.hasExitTime = true;
                        newTrans.exitTime = 0.9f;
                        smNode->addTransition(newTrans);
                    }
                }
            }
        }
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
        
        // Fallback: Check static objects if not found - REMOVED for performance
        // if (!animCtrl) { ... }
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
    AnimationGraph::AnimationNodeGraph* runtimeGraph = nullptr;
    SceneData::ImportedModelContext* activeModelCtx = nullptr;
    bool useGraphPlayback = false;
    
    // 1. Resolve Active Animator
    if (!g_animGraphUI.activeCharacter.empty()) {
        // Check imported models
        for (auto& mctx : ctx.scene.importedModelContexts) {
            if (mctx.importName == g_animGraphUI.activeCharacter) {
                animCtrl = mctx.animator.get();
                runtimeGraph = mctx.runtimeGraph ? mctx.runtimeGraph.get() : mctx.graph.get();
                activeModelCtx = &mctx;
                useGraphPlayback = mctx.useAnimGraph && runtimeGraph;
                break;
            }
        }
        // Fallback: Check static objects - REMOVED for performance
        // if (!animCtrl) { ... }
    }
    
    // Default to global instance if no specific character selected
    if (!animCtrl) animCtrl = &AnimationController::getInstance();
    
    ImGui::BeginChild("AnimPlayback", ImVec2(0, 88), true);
    ImGui::TextDisabled("Playback");
    ImGui::SameLine();
    UIWidgets::HelpMarker("If Use Anim Graph is enabled, these buttons control runtime graph clip playback. Otherwise the legacy controller is used.");
    
    // Current state info
    std::string currentClip;
    float normalizedTime = 0.0f;
    bool isPlaying = false;
    bool isPaused = false;

    if (useGraphPlayback && runtimeGraph) {
        auto status = runtimeGraph->getPlaybackStatus();
        currentClip = status.clipName;
        normalizedTime = status.normalizedTime;
        isPlaying = status.isPlaying;
        isPaused = status.isPaused;
    } else {
        currentClip = animCtrl->getCurrentClipName();
        normalizedTime = animCtrl->getNormalizedTime();
        isPlaying = animCtrl->isPlaying();
        isPaused = animCtrl->isPaused();
    }
    
    ImGui::Text("Current: %s", currentClip.empty() ? "(Stopped)" : currentClip.c_str());
    
    // Progress bar
    ImGui::ProgressBar(normalizedTime, ImVec2(-1, 0), 
        (std::to_string((int)(normalizedTime * 100)) + "%").c_str());
    
    // Playback buttons
    // Play/Pause Toggle
    if (ImGui::Button(isPaused ? "Resume" : (isPlaying ? "Pause" : "Play"), ImVec2(60, 22))) {
        if (useGraphPlayback && runtimeGraph) {
            if (activeModelCtx) {
                activeModelCtx->animGraphFollowTimeline = false;
            }
            runtimeGraph->setPlaybackPaused(!isPaused);
        } else if (isPlaying || isPaused) {
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
    showAnimGraphButtonTooltip("Play or pause local preview for the selected character. In AnimGraph mode this also switches playback to viewport preview instead of Timeline follow.");
    
    ImGui::SameLine();
    
    // Stop Button
    if (ImGui::Button("Stop", ImVec2(50, 22))) {
        if (useGraphPlayback && runtimeGraph) {
            runtimeGraph->stopPlayback();
        } else {
            animCtrl->stopAll();
        }
    }
    showAnimGraphButtonTooltip("Stop the active clip playback. Runtime graph playback resets clip time to its start.");
    
    ImGui::SameLine();
    
    // Rewind
    if (ImGui::Button("<<", ImVec2(30, 22))) {
        if (useGraphPlayback && runtimeGraph) {
            runtimeGraph->resetPlayback();
        } else {
            animCtrl->setTime(0.0f);
        }
    }
    showAnimGraphButtonTooltip("Rewind clip time to the beginning. State machine selection stays intact.");

    // Status Indicators
    if (!useGraphPlayback && animCtrl->isBlending()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "[Blending]");
    } else if (useGraphPlayback) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "[Graph]");
    }
    
    ImGui::EndChild();
}

// ============================================================================
// ANIMATION STATE MACHINE PANEL
// ============================================================================

inline void drawStateMachinePanel(UIContext& ctx, AnimationGraph::AnimationNodeGraph* graph, const std::string& characterName) {
    if (!graph) return;
    
    ImGui::BeginChild("StateMachine", ImVec2(0, 220), true);
    ImGui::Text("State Machine");
    ImGui::SameLine();
    UIWidgets::HelpMarker("The state list reflects the asset setup. The Flow Map shows which state and transition are active in the runtime instance.");
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

        if (!characterName.empty() && ImGui::Button("Keyframe Current State")) {
            Keyframe kf(ctx.scene.timeline.current_frame);
            kf.has_anim_graph = true;
            kf.anim_graph.force_state = smNode->currentStateName;
            ctx.scene.timeline.insertKeyframe(characterName, kf);
        }
        showAnimGraphButtonTooltip("Write a force-state key to Timeline. During playback, the runtime graph will force that state on this frame.");

        ImGui::Separator();
        ImGui::Text("Flow Map");
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImVec2(ImGui::GetContentRegionAvail().x, 90.0f);
        drawList->AddRectFilled(origin, ImVec2(origin.x + canvasSize.x, origin.y + canvasSize.y), IM_COL32(28, 28, 32, 255), 6.0f);

        std::unordered_map<std::string, ImVec2> stateCenters;
        const float nodeW = 90.0f;
        const float nodeH = 34.0f;
        const float spacing = smNode->states.empty() ? 0.0f : canvasSize.x / (float)std::max<size_t>(1, smNode->states.size());
        for (size_t i = 0; i < smNode->states.size(); ++i) {
            float x = origin.x + 10.0f + i * spacing;
            float y = origin.y + 28.0f;
            ImVec2 minP(x, y);
            ImVec2 maxP(std::min(x + nodeW, origin.x + canvasSize.x - 10.0f), y + nodeH);
            bool isCurrent = smNode->states[i].name == smNode->currentStateName;
            bool isTarget = smNode->states[i].name == smNode->targetStateName && smNode->isTransitioning;
            ImU32 fill = isCurrent ? IM_COL32(60, 140, 80, 255) : (isTarget ? IM_COL32(160, 120, 40, 255) : IM_COL32(60, 60, 70, 255));
            drawList->AddRectFilled(minP, maxP, fill, 6.0f);
            drawList->AddRect(minP, maxP, IM_COL32(180, 180, 190, 180), 6.0f, 0, 1.5f);
            drawList->AddText(ImVec2(minP.x + 8.0f, minP.y + 9.0f), IM_COL32_WHITE, smNode->states[i].name.c_str());
            stateCenters[smNode->states[i].name] = ImVec2((minP.x + maxP.x) * 0.5f, (minP.y + maxP.y) * 0.5f);
        }

        for (const auto& trans : smNode->transitions) {
            auto fromIt = stateCenters.find(trans.fromState);
            auto toIt = stateCenters.find(trans.toState);
            if (fromIt == stateCenters.end() || toIt == stateCenters.end()) continue;
            bool active = smNode->isTransitioning && trans.fromState == smNode->currentStateName && trans.toState == smNode->targetStateName;
            ImU32 col = active ? IM_COL32(255, 210, 90, 255) : IM_COL32(140, 140, 150, 160);
            drawList->AddLine(fromIt->second, toIt->second, col, active ? 3.0f : 1.5f);
        }

        ImGui::Dummy(canvasSize);

        if (!graph->debugTrace.eventLog.empty()) {
            ImGui::Separator();
            ImGui::Text("Recent Events");
            for (auto it = graph->debugTrace.eventLog.rbegin(); it != graph->debugTrace.eventLog.rend(); ++it) {
                ImGui::BulletText("%s", it->c_str());
            }
        }
        
        break; // Only show first state machine for now
    }
    
    ImGui::EndChild();
}

inline void drawRuntimeFlowPanel(AnimationGraph::AnimationNodeGraph* graph) {
    if (!graph) return;

    ImGui::BeginChild("AnimRuntimeFlow", ImVec2(0, 0), true);
    ImGui::Text("Runtime Flow");
    ImGui::SameLine();
    UIWidgets::HelpMarker("This panel shows the live runtime instance. The asset graph may change in the editor, but only evaluated runtime results appear here.");
    ImGui::Separator();

    const auto& trace = graph->debugTrace;
    ImGui::Text("Eval Pass: %llu", static_cast<unsigned long long>(trace.evaluationSerial));
    ImGui::Text("Evaluated Nodes: %zu", trace.evaluatedNodeOrder.size());
    ImGui::Text("Active Links: %zu", trace.activeLinkIds.size());

    if (ImGui::CollapsingHeader("Execution Order", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < trace.evaluatedNodeOrder.size(); ++i) {
            auto* node = graph->findNodeById(trace.evaluatedNodeOrder[i]);
            if (!node) continue;
            int evalCount = 1;
            auto countIt = trace.nodeEvalCounts.find(node->id);
            if (countIt != trace.nodeEvalCounts.end()) evalCount = countIt->second;
            ImGui::BulletText("%zu. %s x%d", i + 1, node->metadata.displayName.c_str(), evalCount);
        }
    }

    if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (const auto& [name, value] : trace.floatParamsSnapshot) {
            ImGui::BulletText("%s = %.3f", name.c_str(), value);
        }
        for (const auto& [name, value] : trace.boolParamsSnapshot) {
            ImGui::BulletText("%s = %s", name.c_str(), value ? "true" : "false");
        }
        for (const auto& [name, value] : trace.intParamsSnapshot) {
            ImGui::BulletText("%s = %d", name.c_str(), value);
        }
        for (const auto& name : trace.triggerParamsSnapshot) {
            ImGui::BulletText("%s [trigger]", name.c_str());
        }
        if (trace.floatParamsSnapshot.empty() && trace.boolParamsSnapshot.empty() &&
            trace.intParamsSnapshot.empty() && trace.triggerParamsSnapshot.empty()) {
            ImGui::TextDisabled("No runtime parameters.");
        }
    }

    if (ImGui::CollapsingHeader("State Machines", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (trace.stateMachines.empty()) {
            ImGui::TextDisabled("No state machine activity.");
        } else {
            for (const auto& sm : trace.stateMachines) {
                ImGui::PushID(static_cast<int>(sm.nodeId));
                ImGui::Text("Node %u", sm.nodeId);
                ImGui::BulletText("Current: %s", sm.currentState.empty() ? "(none)" : sm.currentState.c_str());
                if (sm.isTransitioning) {
                    ImGui::BulletText("Target: %s", sm.targetState.c_str());
                    ImGui::ProgressBar(sm.transitionProgress, ImVec2(-1, 0));
                }
                if (!sm.lastTriggeredTransition.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "%s", sm.lastTriggeredTransition.c_str());
                }
                ImGui::Separator();
                ImGui::PopID();
            }
        }
    }

    if (ImGui::CollapsingHeader("Event Log", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (trace.eventLog.empty()) {
            ImGui::TextDisabled("No runtime events.");
        } else {
            for (auto it = trace.eventLog.rbegin(); it != trace.eventLog.rend(); ++it) {
                ImGui::BulletText("%s", it->c_str());
            }
        }
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
    const bool showTitle = zoom >= 0.34f;
    const bool showPinLabels = zoom >= 0.72f;
    bool interactingWithNodeChrome = false;

    // Clear pin position cache
    g_animGraphUI.pinScreenPositions.clear();

    // Background
    drawList->AddRectFilled(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(22, 24, 30, 255));

    // Grid
    float gridSize = 32.0f * zoom;
    for (float x = fmodf(g_animGraphUI.canvasOffset.x, gridSize); x < canvasSize.x; x += gridSize) {
        drawList->AddLine(
            ImVec2(canvasPos.x + x, canvasPos.y),
            ImVec2(canvasPos.x + x, canvasPos.y + canvasSize.y),
            IM_COL32(43, 47, 58, 160));
    }
    for (float y = fmodf(g_animGraphUI.canvasOffset.y, gridSize); y < canvasSize.y; y += gridSize) {
        drawList->AddLine(
            ImVec2(canvasPos.x, canvasPos.y + y),
            ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + y),
            IM_COL32(43, 47, 58, 160));
    }

    // ========== DRAW NODES & COLLECT PIN POSITIONS ==========
    for (auto& node : graph->nodes) {
        ImVec2 nodePos = ImVec2(
            canvasPos.x + g_animGraphUI.canvasOffset.x + node->x * zoom,
            canvasPos.y + g_animGraphUI.canvasOffset.y + node->y * zoom);

        float baseWidthScreen = animGraphCanvasMetric(156.0f, zoom, 112.0f, 240.0f);
        float titleWidth = ImGui::CalcTextSize(node->metadata.displayName.c_str()).x +
            animGraphCanvasMetric(10.0f, zoom, 7.0f, 14.0f) * 2.0f;
        NodeSystem::NodeChromeLayout chrome = NodeSystem::buildNodeChromeLayout(
            *node, zoom, baseWidthScreen, node->inputs.size(), node->outputs.size(), titleWidth);

        float nodeWidth = chrome.width;
        float headerHeight = chrome.headerHeight;
        float pinRadius = chrome.pinRadius;
        float cornerRadius = chrome.cornerRadius;
        float bodyPadding = chrome.bodyPadding;
        float shadowOffset = chrome.shadowOffset;
        bool isCollapsed = chrome.collapsed;
        float nodeHeight = chrome.height;
        float labelWidth = chrome.labelWidth;

        ImVec2 nodeMax = ImVec2(nodePos.x + nodeWidth, nodePos.y + nodeHeight);

        drawList->AddRectFilled(
            ImVec2(nodePos.x + shadowOffset, nodePos.y + shadowOffset),
            ImVec2(nodeMax.x + shadowOffset, nodeMax.y + shadowOffset),
            IM_COL32(0, 0, 0, 55), cornerRadius);

        // Node background
        drawList->AddRectFilled(nodePos, nodeMax, IM_COL32(34, 37, 45, 245), cornerRadius);

        // Header
        drawList->AddRectFilled(nodePos,
            ImVec2(nodeMax.x, nodePos.y + headerHeight),
            node->metadata.headerColor, cornerRadius, ImDrawFlags_RoundCornersTop);

        drawList->AddLine(
            ImVec2(nodePos.x + 1.0f, nodePos.y + headerHeight),
            ImVec2(nodeMax.x - 1.0f, nodePos.y + headerHeight),
            IM_COL32(255, 255, 255, 22), 1.0f);

        // Title
        if (showTitle) {
            const std::string title = fitAnimGraphCanvasText(node->metadata.displayName, nodeWidth - bodyPadding * 2.0f);
            drawAnimGraphCanvasTextClipped(
                drawList,
                ImVec2(nodePos.x + bodyPadding, nodePos.y + (headerHeight - ImGui::GetTextLineHeight()) * 0.5f - 1.0f),
                ImVec2(nodePos.x + bodyPadding, nodePos.y + 2.0f),
                ImVec2(nodeMax.x - bodyPadding, nodePos.y + headerHeight - 2.0f),
                IM_COL32(245, 247, 250, 255),
                title);
        }

        // Border
        bool isSelected = std::find(g_animGraphUI.selectedNodeIds.begin(),
            g_animGraphUI.selectedNodeIds.end(), node->id) != g_animGraphUI.selectedNodeIds.end();
        bool isRuntimeActive = graph->debugTrace.nodeEvalCounts.find(node->id) != graph->debugTrace.nodeEvalCounts.end();

        drawList->AddRect(nodePos, nodeMax,
            isSelected ? IM_COL32(255, 180, 50, 255) : IM_COL32(66, 74, 92, 210),
            cornerRadius, 0, isSelected ? 2.0f : 1.0f);
        if (isRuntimeActive) {
            drawList->AddRect(nodePos, nodeMax, IM_COL32(80, 255, 140, 220), cornerRadius, 0, 3.0f);
        }

        float toggleSize = chrome.toggleSize;
        ImVec2 toggleMin(
            nodeMax.x - toggleSize - bodyPadding * 0.6f,
            nodePos.y + (headerHeight - toggleSize) * 0.5f);
        ImVec2 toggleMax(toggleMin.x + toggleSize, toggleMin.y + toggleSize);
        ImGui::SetCursorScreenPos(toggleMin);
        ImGui::PushID((int)node->id + 2000000);
        ImGui::InvisibleButton("AnimNodeCollapseToggle", ImVec2(toggleSize, toggleSize));
        bool toggleHovered = ImGui::IsItemHovered();
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
            node->collapsed = !node->collapsed;
        }
        if (toggleHovered) {
            ImGui::SetTooltip("%s", node->collapsed ? "Expand node" : "Collapse node");
        }
        ImGui::PopID();
        drawList->AddRectFilled(toggleMin, toggleMax,
            toggleHovered ? IM_COL32(255, 255, 255, 28) : IM_COL32(0, 0, 0, 26),
            toggleSize * 0.28f);
        drawList->AddLine(
            ImVec2(toggleMin.x + 4.0f, toggleMin.y + toggleSize * 0.5f),
            ImVec2(toggleMax.x - 4.0f, toggleMin.y + toggleSize * 0.5f),
            IM_COL32(245, 247, 250, 220), 1.4f);
        if (node->collapsed) {
            drawList->AddLine(
                ImVec2(toggleMin.x + toggleSize * 0.5f, toggleMin.y + 4.0f),
                ImVec2(toggleMin.x + toggleSize * 0.5f, toggleMax.y - 4.0f),
                IM_COL32(245, 247, 250, 220), 1.4f);
        }

        float resizeHandleWidth = std::max(chrome.resizeHandleWidth, 12.0f);
        ImVec2 resizeMin(nodePos.x + nodeWidth - resizeHandleWidth, nodePos.y + headerHeight);
        ImVec2 resizeMax(nodePos.x + nodeWidth + resizeHandleWidth, nodePos.y + nodeHeight);
        ImGui::SetCursorScreenPos(resizeMin);
        ImGui::PushID((int)node->id + 2200000);
        ImGui::InvisibleButton("AnimNodeResizeHandle", ImVec2(resizeMax.x - resizeMin.x, resizeMax.y - resizeMin.y));
        bool resizeHovered = ImGui::IsItemHovered();
        if (resizeHovered || g_animGraphUI.resizingNodeId == node->id) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        }
        if (ImGui::IsItemActivated() || (ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Left))) {
            g_animGraphUI.resizingNodeId = node->id;
        }
        if (g_animGraphUI.resizingNodeId == node->id && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            node->uiWidth += ImGui::GetIO().MouseDelta.x / zoom;
            node->uiWidth = std::clamp(node->uiWidth, 110.0f, 360.0f);
        }
        if (resizeHovered) {
            ImGui::SetTooltip("Drag to resize node");
        }
        if (resizeHovered || g_animGraphUI.resizingNodeId == node->id || toggleHovered) {
            interactingWithNodeChrome = true;
        }
        ImGui::PopID();
        drawList->AddLine(
            ImVec2(nodePos.x + nodeWidth - 1.0f, nodePos.y + headerHeight + 4.0f),
            ImVec2(nodePos.x + nodeWidth - 1.0f, nodePos.y + nodeHeight - 4.0f),
            resizeHovered || g_animGraphUI.resizingNodeId == node->id ? IM_COL32(255, 255, 255, 90) : IM_COL32(255, 255, 255, 34),
            2.0f);

        // Input pins
        float inputPinStart = NodeSystem::getNodePinStartY(chrome, nodePos.y, node->inputs.size());
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            auto& pin = node->inputs[i];
            float pinY = inputPinStart + i * (isCollapsed ? chrome.collapsedPinSpacing : chrome.pinSpacing);
            ImVec2 pinPos = ImVec2(nodePos.x, pinY);
            
            // Cache pin screen position
            g_animGraphUI.pinScreenPositions[pin.id] = pinPos;
            
            // Draw pin circle
            ImU32 pinColor = pin.cachedColor != 0 ? pin.cachedColor : IM_COL32(100, 180, 255, 255);
            drawList->AddCircleFilled(pinPos, pinRadius, pinColor);
            drawList->AddCircle(pinPos, pinRadius, IM_COL32(255, 255, 255, 150), 0, 1.0f);
            
            // Pin label
            if (!isCollapsed && showPinLabels) {
                const std::string label = fitAnimGraphCanvasText(pin.name, labelWidth);
                drawAnimGraphCanvasTextClipped(
                    drawList,
                    ImVec2(pinPos.x + pinRadius + 6.0f * zoom, pinY - ImGui::GetTextLineHeight() * 0.5f),
                    ImVec2(pinPos.x + pinRadius + 4.0f * zoom, pinY - ImGui::GetTextLineHeight()),
                    ImVec2(pinPos.x + pinRadius + 4.0f * zoom + labelWidth, pinY + ImGui::GetTextLineHeight()),
                    IM_COL32(220, 224, 230, 255),
                    label);
            }
        }

        // Output pins
        float outputPinStart = NodeSystem::getNodePinStartY(chrome, nodePos.y, node->outputs.size());
        for (size_t i = 0; i < node->outputs.size(); ++i) {
            auto& pin = node->outputs[i];
            float pinY = outputPinStart + i * (isCollapsed ? chrome.collapsedPinSpacing : chrome.pinSpacing);
            ImVec2 pinPos = ImVec2(nodePos.x + nodeWidth, pinY);
            
            // Cache pin screen position
            g_animGraphUI.pinScreenPositions[pin.id] = pinPos;
            
            // Draw pin circle
            ImU32 pinColor = pin.cachedColor != 0 ? pin.cachedColor : IM_COL32(255, 200, 100, 255);
            drawList->AddCircleFilled(pinPos, pinRadius, pinColor);
            drawList->AddCircle(pinPos, pinRadius, IM_COL32(255, 255, 255, 150), 0, 1.0f);

            // Pin label
            if (!isCollapsed && showPinLabels) {
                const std::string label = fitAnimGraphCanvasText(pin.name, labelWidth);
                ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
                drawAnimGraphCanvasTextClipped(
                    drawList,
                    ImVec2(pinPos.x - textSize.x - pinRadius - 6.0f * zoom, pinY - ImGui::GetTextLineHeight() * 0.5f),
                    ImVec2(pinPos.x - labelWidth - pinRadius - 8.0f * zoom, pinY - ImGui::GetTextLineHeight()),
                    ImVec2(pinPos.x - pinRadius - 4.0f * zoom, pinY + ImGui::GetTextLineHeight()),
                    IM_COL32(220, 224, 230, 255),
                    label);
            }
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
        bool isRuntimeActive = std::find(graph->debugTrace.activeLinkIds.begin(),
            graph->debugTrace.activeLinkIds.end(), link.id) != graph->debugTrace.activeLinkIds.end();

        ImU32 linkColor = isSelected ? IM_COL32(255, 200, 50, 255) :
            (isRuntimeActive ? IM_COL32(80, 255, 160, 255) : IM_COL32(200, 200, 100, 255));
        float thickness = std::max(1.5f, (isSelected ? 4.0f : (isRuntimeActive ? 4.0f : 2.5f)) * zoom);
        
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
            
            drawList->AddBezierCubic(p1, cp1, cp2, p2, IM_COL32(255, 200, 100, 200), std::max(1.5f, 3.0f * zoom));
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
                        markAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
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
                        markAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
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
        if (g_animGraphUI.resizingNodeId != 0) {
            auto* node = graph->findNodeById(g_animGraphUI.resizingNodeId);
            if (node) {
                node->uiWidth += ImGui::GetIO().MouseDelta.x / zoom;
                node->uiWidth = std::clamp(node->uiWidth, 110.0f, 360.0f);
            }
        } else if (!interactingWithNodeChrome) {
            for (uint32_t nodeId : g_animGraphUI.selectedNodeIds) {
                auto* node = graph->findNodeById(nodeId);
                if (node) {
                    node->x += ImGui::GetIO().MouseDelta.x / zoom;
                    node->y += ImGui::GetIO().MouseDelta.y / zoom;
                }
            }
        }
    }

    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        g_animGraphUI.resizingNodeId = 0;
    }

    // ========== NODE SELECTION ==========
    if (!g_animGraphUI.isCreatingLink && g_animGraphUI.resizingNodeId == 0 &&
        !interactingWithNodeChrome && ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
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

                float nodeWidth = node->uiWidth > 0.0f
                    ? node->uiWidth * zoom
                    : animGraphCanvasMetric(156.0f, zoom, 112.0f, 240.0f);
                float titleWidth = ImGui::CalcTextSize(node->metadata.displayName.c_str()).x +
                    animGraphCanvasMetric(10.0f, zoom, 7.0f, 14.0f) * 2.0f;
                NodeSystem::NodeChromeLayout hitChrome = NodeSystem::buildNodeChromeLayout(
                    *node, zoom, nodeWidth, node->inputs.size(), node->outputs.size(), titleWidth);
                float headerHeight = hitChrome.headerHeight;
                float nodeHeight = hitChrome.height;
                nodeWidth = hitChrome.width;

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
                    if (typeId == "FinalPose") {
                        graph->outputNode = static_cast<AnimationGraph::FinalPoseNode*>(newNode.get());
                    }
                    graph->nodes.push_back(std::move(newNode));
                    markAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
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
                for (uint32_t linkId : g_animGraphUI.selectedLinkIds) {
                    graph->disconnect(linkId);
                }
                markAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
                g_animGraphUI.selectedLinkIds.clear();
            }

            // 2. Delete Selected Nodes
            if (!g_animGraphUI.selectedNodeIds.empty()) {
                for (uint32_t nodeId : g_animGraphUI.selectedNodeIds) {
                    graph->removeNode(nodeId);
                }
                markAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
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
                g_animGraphUI.selectedNodeIds.clear();
                g_animGraphUI.selectedLinkIds.clear();
            }
        }
    }

    ImGui::Text("Active Character: %s", g_animGraphUI.activeCharacter.empty() ? "(Select in viewport)" : g_animGraphUI.activeCharacter.c_str());
    
    // Model Settings (Visibility, Root Motion)
    if (!g_animGraphUI.activeCharacter.empty()) {
        for (auto& mctx : ctx.scene.importedModelContexts) {
            if (mctx.importName == g_animGraphUI.activeCharacter) {
                ImGui::Checkbox("Use Root Motion", &mctx.useRootMotion);
                UIWidgets::HelpMarker("Apply motion from the animation to the character transform. For cinematic testing you may prefer to keep this disabled.");
                if (mctx.useRootMotion) {
                    std::vector<std::string> rootMotionCandidates;
                    rootMotionCandidates.emplace_back("");
                    if (mctx.animator) {
                        std::set<std::string> uniqueNames;
                        for (const auto& clip : mctx.animator->getAllClips()) {
                            if (!clip.sourceData) {
                                continue;
                            }
                            for (const auto& [name, _] : clip.sourceData->positionKeys) {
                                uniqueNames.insert(name);
                            }
                        }
                        rootMotionCandidates.insert(rootMotionCandidates.end(), uniqueNames.begin(), uniqueNames.end());
                    }

                    int selectedRootMotionIndex = 0;
                    for (int i = 1; i < static_cast<int>(rootMotionCandidates.size()); ++i) {
                        if (rootMotionCandidates[i] == mctx.rootMotionBone) {
                            selectedRootMotionIndex = i;
                            break;
                        }
                    }

                    auto comboGetter = [](void* data, int idx, const char** out_text) -> bool {
                        auto* items = static_cast<std::vector<std::string>*>(data);
                        if (!items || idx < 0 || idx >= static_cast<int>(items->size())) {
                            return false;
                        }
                        *out_text = (*items)[idx].empty() ? "Auto Detect" : (*items)[idx].c_str();
                        return true;
                    };

                    ImGui::SetNextItemWidth(180.0f);
                    if (ImGui::Combo("Root Motion Bone", &selectedRootMotionIndex, comboGetter, &rootMotionCandidates, static_cast<int>(rootMotionCandidates.size()))) {
                        mctx.rootMotionBone = rootMotionCandidates[selectedRootMotionIndex];
                    }
                    UIWidgets::HelpMarker("Auto Detect uses the built-in heuristic. The list is populated from animated position channels in the imported clips, so renamed scene bone prefixes do not break it.");
                }
                ImGui::SameLine();
                const char* runtimeItems[] = { "Legacy", "Ozz" };
                int runtimeMode = mctx.preferOzzRuntime ? 1 : 0;
                ImGui::SetNextItemWidth(92.0f);
                if (ImGui::Combo("Animation Runtime", &runtimeMode, runtimeItems, IM_ARRAYSIZE(runtimeItems))) {
                    mctx.preferOzzRuntime = (runtimeMode == 1);
                    mctx.loggedOzzRuntimeUsage = false;
                }
                UIWidgets::HelpMarker("Legacy uses the existing controller sampling path. Ozz uses Ozz runtime sampling while keeping the same playback state.");
                ImGui::SameLine();
                ImGui::Checkbox("Use Anim Graph", &mctx.useAnimGraph);
                UIWidgets::HelpMarker("When enabled, the runtime graph is evaluated. When disabled, the legacy AnimationController drives playback.");
                ImGui::SameLine();
                ImGui::Checkbox("Follow Timeline", &mctx.animGraphFollowTimeline);
                UIWidgets::HelpMarker("When enabled, graph playback follows Timeline play and pause state. When disabled, the graph can keep running in the viewport.");
                ImGui::SameLine();
                ImGui::Checkbox("Visible", &mctx.visible);
                break;
            }
        }
    }

    // Get or create graph for current character
    std::shared_ptr<AnimationGraph::AnimationNodeGraph> currentGraph = nullptr;
    if (!g_animGraphUI.activeCharacter.empty()) {
        std::string assetKey = getAnimGraphAssetKeyForCharacter(ctx.scene, g_animGraphUI.activeCharacter);
        auto it = g_animGraphUI.graphs.find(assetKey);
        if (it != g_animGraphUI.graphs.end()) {
            currentGraph = it->second;
        }
    }
    // ========== TOOLBAR AT TOP ==========
    // Toolbar (horizontal layout at top)
    {
        bool useAnimGraph = false;
        bool followTimeline = true;
        if (!g_animGraphUI.activeCharacter.empty()) {
            if (auto* mctx = findImportedModelContext(ctx.scene, g_animGraphUI.activeCharacter)) {
                useAnimGraph = mctx->useAnimGraph;
                followTimeline = mctx->animGraphFollowTimeline;
            }
        }

        drawAnimGraphQuickGuide(!g_animGraphUI.activeCharacter.empty(), currentGraph != nullptr, useAnimGraph, followTimeline);
        ImGui::Separator();

        if (!g_animGraphUI.activeCharacter.empty()) {
            const bool runtimeStale = isAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
            ImGui::TextDisabled("Runtime:");
            ImGui::SameLine();
            ImGui::TextColored(followTimeline ? ImVec4(0.95f, 0.75f, 0.35f, 1.0f) : ImVec4(0.45f, 0.85f, 1.0f, 1.0f),
                "%s", followTimeline ? "Timeline" : "Viewport");
            ImGui::SameLine();
            ImGui::TextDisabled("| Sync:");
            ImGui::SameLine();
            ImGui::TextColored(runtimeStale ? ImVec4(1.0f, 0.55f, 0.35f, 1.0f) : ImVec4(0.45f, 0.9f, 0.55f, 1.0f),
                "%s", runtimeStale ? "Pending Push" : "Live");
            ImGui::SameLine();
            UIWidgets::HelpMarker("Timeline means playback is driven by Timeline frames. Viewport means local preview. Pending Push means the asset graph has structural edits that are not yet cloned into the runtime instance.");
            ImGui::Separator();
        }

        drawAnimGraphToolbarLabel("Authoring");

        // Quick create button (only if no graph)
        if (!currentGraph) {
            if (ImGui::Button("Create Graph", ImVec2(100, 22))) {
                auto graph = std::make_shared<AnimationGraph::AnimationNodeGraph>();

                auto* outputNode = graph->addNode<AnimationGraph::FinalPoseNode>();
                outputNode->x = 400.0f;
                outputNode->y = 100.0f;

                auto* clipNode = graph->addNode<AnimationGraph::AnimClipNode>();
                clipNode->x = 100.0f;
                clipNode->y = 100.0f;

                if (!clipNode->outputs.empty() && !outputNode->inputs.empty()) {
                    graph->connect(clipNode->outputs[0].id, outputNode->inputs[0].id);
                }

                std::string graphKey = g_animGraphUI.activeCharacter.empty() ? "Default" :
                    getAnimGraphAssetKeyForCharacter(ctx.scene, g_animGraphUI.activeCharacter);
                g_animGraphUI.graphs[graphKey] = graph;
                markAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
                syncRuntimeGraphFromAsset(ctx.scene, g_animGraphUI.activeCharacter);
            }
            showAnimGraphButtonTooltip("Create a minimal graph for the selected character with one clip node connected to Final Pose.");
            ImGui::SameLine();
        }

        // Add node button
        if (ImGui::Button("Add Node", ImVec2(80, 22))) {
            ImGui::OpenPopup("AddAnimNodePopupToolbar");
        }
        showAnimGraphButtonTooltip("Add a new node to the asset graph. You can reposition it later on the canvas.");

        ImGui::SameLine();

        // Sync Scene button
        if (ImGui::Button("Sync Scene", ImVec2(90, 22))) {
            auto& animCtrl = AnimationController::getInstance();
            animCtrl.registerClips(ctx.scene.animationDataList);

            for (const auto& anim : ctx.scene.animationDataList) {
                if (!anim) continue;
                if (!anim->name.empty() && g_animGraphUI.graphs.find(anim->name) == g_animGraphUI.graphs.end()) {
                    auto graph = std::make_shared<AnimationGraph::AnimationNodeGraph>();

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

                    g_animGraphUI.graphs[anim->name] = graph;

                    if (g_animGraphUI.activeCharacter.empty()) {
                        g_animGraphUI.activeCharacter = anim->name;
                    }
                }
            }
        }
        showAnimGraphButtonTooltip("Build quick starter graphs from scene animation clips and register clip lists with the controller.");

        if (!g_animGraphUI.activeCharacter.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("|");
            ImGui::SameLine();
            drawAnimGraphToolbarLabel("Runtime");
            if (ImGui::Button("Push To Runtime", ImVec2(110, 22))) {
                syncRuntimeGraphFromAsset(ctx.scene, g_animGraphUI.activeCharacter);
            }
            showAnimGraphButtonTooltip("Clone the editor asset graph into the selected character's runtime instance. Use this after editing to see changes in the scene.");
            ImGui::SameLine();

            if (ImGui::Button("Build Demo Rig", ImVec2(110, 22))) {
                createAnimGraphDemoRig(ctx.scene, g_animGraphUI.activeCharacter);
                markAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
            }
            showAnimGraphButtonTooltip("Build an Idle/Move/Action demo rig so you can test the state machine even with single-clip characters.");
            ImGui::SameLine();

            ImGui::TextDisabled("|");
            ImGui::SameLine();
            drawAnimGraphToolbarLabel("Timeline");
            if (ImGui::Button("Demo Timeline", ImVec2(105, 22))) {
                createAnimGraphDemoTimeline(ctx.scene, g_animGraphUI.activeCharacter);
            }
            showAnimGraphButtonTooltip("Write sample speed and trigger keys to Timeline for fast playback testing.");
            ImGui::SameLine();
        }

        ImGui::TextDisabled("|");
        ImGui::SameLine();
        drawAnimGraphToolbarLabel("View");

        // Reset View button
        if (ImGui::Button("Reset View", ImVec2(80, 22))) {
            g_animGraphUI.canvasOffset = ImVec2(0, 0);
            g_animGraphUI.canvasZoom = 1.0f;
        }
        showAnimGraphButtonTooltip("Reset canvas pan and zoom to their default values.");

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
                        if (g_animGraphUI.graphs.find("Default") == g_animGraphUI.graphs.end()) {
                            g_animGraphUI.graphs["Default"] = std::make_shared<AnimationGraph::AnimationNodeGraph>();
                        }
                    }

                    std::string assetKey = getAnimGraphAssetKeyForCharacter(ctx.scene, g_animGraphUI.activeCharacter);
                    auto& graph = g_animGraphUI.graphs[assetKey];
                    if (!graph) {
                        graph = std::make_shared<AnimationGraph::AnimationNodeGraph>();
                    }
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
                        if (typeId == "FinalPose") {
                            graph->outputNode = static_cast<AnimationGraph::FinalPoseNode*>(node.get());
                        }
                        graph->nodes.push_back(std::move(node));
                        markAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
                        syncRuntimeGraphFromAsset(ctx.scene, g_animGraphUI.activeCharacter);
                    }
                }
            }
            ImGui::EndPopup();
        }



        // Auto-create graph for active character if missing
        if (!g_animGraphUI.activeCharacter.empty() &&
            g_animGraphUI.graphs.find(getAnimGraphAssetKeyForCharacter(ctx.scene, g_animGraphUI.activeCharacter)) == g_animGraphUI.graphs.end()) {

            std::string assetKey = getAnimGraphAssetKeyForCharacter(ctx.scene, g_animGraphUI.activeCharacter);
            g_animGraphUI.graphs[assetKey] = std::make_shared<AnimationGraph::AnimationNodeGraph>();
            auto& graph = g_animGraphUI.graphs[assetKey];

            // Add default Final Pose node
            auto finalNode = std::make_unique<AnimationGraph::FinalPoseNode>();
            finalNode->x = 600; finalNode->y = 300;
            finalNode->id = graph->nextNodeId++;
            finalNode->inputs[0].id = graph->nextPinId++;
            finalNode->inputs[0].nodeId = finalNode->id;

            graph->outputNode = finalNode.get();
            graph->nodes.push_back(std::move(finalNode));
            markAnimGraphRuntimeStale(ctx.scene, g_animGraphUI.activeCharacter);
            syncRuntimeGraphFromAsset(ctx.scene, g_animGraphUI.activeCharacter);
        }

        ImGui::Separator();

        if (!g_animGraphUI.activeCharacter.empty()) {
            std::string refreshedAssetKey = getAnimGraphAssetKeyForCharacter(ctx.scene, g_animGraphUI.activeCharacter);
            auto refreshedIt = g_animGraphUI.graphs.find(refreshedAssetKey);
            currentGraph = (refreshedIt != g_animGraphUI.graphs.end()) ? refreshedIt->second : nullptr;
        } else {
            currentGraph = nullptr;
        }

        // ========== MAIN CONTENT: Left Panel | Resize Handle | Node Canvas ==========
        static float leftPanelWidth = 250.0f;
        static float rightPanelWidth = 300.0f;
        static float rightPanelSplitY = 300.0f;
        const float minPanelWidth = 150.0f;
        const float maxPanelWidth = 500.0f;

        float availHeight = ImGui::GetContentRegionAvail().y;
        float availWidth = ImGui::GetContentRegionAvail().x;
        AnimationGraph::AnimationNodeGraph* runtimeGraphForActiveCharacter =
            getRuntimeGraphForCharacter(ctx.scene, g_animGraphUI.activeCharacter);

        // Left panel
        ImGui::BeginChild("AnimLeftPanel", ImVec2(leftPanelWidth, availHeight), true);
        drawAnimGraphStatusCard(ctx.scene, g_animGraphUI.activeCharacter, currentGraph.get());
        ImGui::Spacing();
        
        // --- ONLY SHOW RELEVANT STUFF FOR THE ACTIVE GRAPH ---
        if (currentGraph) {
            drawAnimGraphSectionLabel("Playback", "Local preview controls");
            drawAnimationPlaybackControls(ctx);
            ImGui::Spacing();

            // Clips panel (useful as a reference list)
            drawAnimGraphSectionLabel("Clip Library", "Reference list for the selected character");
            if (ImGui::CollapsingHeader("Available Clips", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawAnimationClipsPanel(ctx);
            }
            ImGui::Spacing();
            
            // Parameters panel
            if (g_animGraphUI.showParameterPanel) {
                drawAnimGraphSectionLabel("Parameters", "Runtime values and Timeline key source");
                drawAnimationParametersPanel(ctx, currentGraph.get(), runtimeGraphForActiveCharacter, g_animGraphUI.activeCharacter);
            }
        } else {
            ImGui::TextDisabled("Select a character in the viewport to start editing.");
            ImGui::Spacing();
            ImGui::TextWrapped("AnimGraph editing is driven by viewport selection. Once a character is selected you can create a graph, preview it locally, and push changes to runtime.");
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
        // Calculate the maximum space we can take taking the right panel into account
        float canvasWidth = availWidth - leftPanelWidth - rightPanelWidth - 16.0f;
        if (canvasWidth < 200.0f) canvasWidth = 200.0f; // Limit to min width

        ImGui::BeginChild("AnimNodeCanvas", ImVec2(canvasWidth, availHeight), true, ImGuiWindowFlags_NoScrollbar);

        if (currentGraph) {
            // --- CRITICAL LINK: UI Graph -> Renderer Model Context ---
            if (!g_animGraphUI.activeCharacter.empty()) {
                for (auto& mctx : ctx.scene.importedModelContexts) {
                    if (mctx.importName == g_animGraphUI.activeCharacter) {
                        mctx.animGraphAssetKey = getAnimGraphAssetKeyForCharacter(ctx.scene, g_animGraphUI.activeCharacter);
                        mctx.graph = currentGraph;
                        if (!mctx.runtimeGraph) {
                            mctx.runtimeGraph = currentGraph ? currentGraph->clone() : nullptr;
                        }
                        break;
                    }
                }
            }
            drawNodeCanvas(ctx, currentGraph.get());
        }
        else {
            ImGui::Text("No animation graph loaded.");
            ImGui::Text("Use 'Sync Scene' or 'Create Graph' from toolbar.");
        }

        ImGui::EndChild();
        
        ImGui::SameLine();
        
        // Right Resize handle
        ImGui::InvisibleButton("##AnimRightPanelResize", ImVec2(6.0f, availHeight));
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        }
        if (ImGui::IsItemActive()) {
            rightPanelWidth -= ImGui::GetIO().MouseDelta.x;
            rightPanelWidth = std::clamp(rightPanelWidth, minPanelWidth, maxPanelWidth);
        }

        // Draw right resize handle indicator
        ImVec2 handleMinRight = ImGui::GetItemRectMin();
        ImVec2 handleMaxRight = ImGui::GetItemRectMax();
        drawList->AddRectFilled(handleMinRight, handleMaxRight,
            ImGui::IsItemHovered() ? IM_COL32(100, 100, 100, 255) : IM_COL32(60, 60, 60, 255));

        ImGui::SameLine();
        
        // Right panel
        ImGui::BeginChild("AnimRightPanel", ImVec2(rightPanelWidth, availHeight), true);
        
        if (currentGraph) {
            // Upper Area: Node Properties
            ImGui::BeginChild("AnimRightTop", ImVec2(0, rightPanelSplitY), false);
            drawAnimGraphSectionLabel("Inspector", "Selected node properties");
            ImGui::Separator();
            if (!g_animGraphUI.selectedNodeIds.empty()) {
                drawNodePropertiesPanel(ctx, currentGraph.get(), runtimeGraphForActiveCharacter);
            } else {
                ImGui::TextDisabled("Select a node on the canvas to inspect and edit its properties.");
            }
            ImGui::EndChild();

            // Horizontal Splitter
            ImGui::InvisibleButton("##RightPanelSplitter", ImVec2(-1, 6.0f));
            if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            if (ImGui::IsItemActive()) {
                rightPanelSplitY += ImGui::GetIO().MouseDelta.y;
                rightPanelSplitY = std::clamp(rightPanelSplitY, 100.0f, availHeight - 100.0f);
            }

            // Lower Area: State Machine
            ImGui::BeginChild("AnimRightBottom", ImVec2(0, 0), false);
            drawAnimGraphSectionLabel("Diagnostics", "Runtime state machine and execution");
            ImGui::Separator();
            
            // State machine panel for quick overview
            AnimationGraph::StateMachineNode* smNodeToDisplay = nullptr;
            if (!g_animGraphUI.selectedNodeIds.empty()) {
                auto* selNode = currentGraph->findNodeById(g_animGraphUI.selectedNodeIds[0]);
                if (selNode && selNode->getTypeId() == "StateMachine") {
                    smNodeToDisplay = static_cast<AnimationGraph::StateMachineNode*>(selNode);
                }
            }
            
            if (!smNodeToDisplay) {
                for (auto& node : currentGraph->nodes) {
                    if (node->getTypeId() == "StateMachine") {
                        smNodeToDisplay = static_cast<AnimationGraph::StateMachineNode*>(node.get());
                        break;
                    }
                }
            }
            
            if (smNodeToDisplay) {
                drawStateMachinePanel(ctx, runtimeGraphForActiveCharacter ? runtimeGraphForActiveCharacter : currentGraph.get(), g_animGraphUI.activeCharacter);
            } else {
                ImGui::TextDisabled("No state machine node found in this graph.");
            }

            drawRuntimeFlowPanel(runtimeGraphForActiveCharacter ? runtimeGraphForActiveCharacter : currentGraph.get());
            
            ImGui::EndChild();
        } else {
            ImGui::TextDisabled("No active graph.");
            ImGui::TextWrapped("Select a character in the viewport and create an AnimGraph to unlock the inspector and diagnostics panels.");
        }
        
        ImGui::EndChild();
    }
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

