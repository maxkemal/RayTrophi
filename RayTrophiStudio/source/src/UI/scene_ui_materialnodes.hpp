/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_materialnodes.hpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file scene_ui_materialnodes.hpp
 * @brief Material node editor panel (Faz 1) — see MaterialNodesV2.h for the
 * no-bake design rationale.
 *
 * Layout mirrors the terrain node editor (library | canvas | properties).
 * Apply folds the graph into the SELECTED material through the exact same
 * update chain the material panel's sliders use (PBRMaterialSnapshot ->
 * gpuMaterial, triangle texture bundles, updateBackendMaterial, accumulation
 * resets) — no new GPU paths, no descriptor churn beyond a normal material edit.
 */

#include "imgui.h"
#include "MaterialNodesV2.h"
#include "PBRMaterialSnapshot.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "SceneSelection.h"
#include "ProjectManager.h"
#include "NodeSystem/NodeEditorUIV2.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace MaterialNodesV2 {

class MaterialNodeEditorUI {
public:
    /// File dialog callback (set from scene_ui.cpp, same as the terrain editor)
    std::function<std::string(const wchar_t*)> onOpenFileDialog;

    /// Which material's graph the panel is showing
    std::string activeMaterialName;

    /// Last Apply diagnostics (shown under the toolbar)
    std::vector<std::string> lastWarnings;
    std::vector<std::string> lastErrors;
    bool lastApplyOk = false;
    bool hasApplied = false;

    // Panel layout (user-resizable / collapsible via the strip on its own edge).
    // There is no properties panel: nodes carry their parameters.
    float libraryWidth = 160.0f;
    bool showLibrary = true;

    // Live mode: every graph edit (link/node change, parameter drag, texture
    // pick) applies to the material the same frame — same per-edit cost as the
    // material panel's sliders. The Apply button remains for Live-off use.
    bool liveApply = true;

    // Scene-selection follow: which object's materials the picker lists
    std::string selObjectName;
    std::vector<std::pair<uint16_t, std::string>> selMaterials;  // id -> name

    MaterialNodeEditorUI() {
        editor.config.gridSizeMinor = 32.0f;
        editor.config.showMinimap = true;
        editor.onGraphModified = [this]() {
            ProjectManager::getInstance().markModified();
            graphChangedThisFrame_ = true;
        };
        categories = {
            { "Input", ImVec4(0.25f, 0.62f, 0.35f, 1.0f), {
                { NodeType::Value, "Value" },
                { NodeType::Color, "Color" },
                { NodeType::TextureCoordinate, "Texture Coordinate" },
                { NodeType::Geometry, "Geometry" },
                { NodeType::ObjectInfo, "Object Info" },
                { NodeType::Attribute, "Attribute" },
                { NodeType::Fresnel, "Fresnel" },
                { NodeType::LayerWeight, "Layer Weight" },
                { NodeType::AmbientOcclusion, "Ambient Occlusion" },
                { NodeType::MaterialRef, "Material (Ref)" }
            } },
            { "Texture", ImVec4(0.75f, 0.55f, 0.25f, 1.0f), {
                { NodeType::ImageTexture, "Image Texture" },
                // Voronoi/Checker live inside Noise Texture's Type combo now.
                { NodeType::Noise, "Noise Texture" },
                { NodeType::Wave, "Wave Texture" },
                { NodeType::Gradient, "Gradient Texture" }
            } },
            { "Color", ImVec4(0.7f, 0.6f, 0.3f, 1.0f), {
                { NodeType::MixColor, "Mix Color" },
                { NodeType::ColorRamp, "Color Ramp" },
                { NodeType::Invert, "Invert" },
                { NodeType::Gamma, "Gamma" },
                { NodeType::BrightContrast, "Bright/Contrast" },
                { NodeType::HueSaturation, "Hue/Saturation" },
                { NodeType::RGBCurves, "RGB Curves" }
            } },
            { "Convert", ImVec4(0.35f, 0.55f, 0.75f, 1.0f), {
                { NodeType::Math, "Math" },
                { NodeType::Clamp, "Clamp" },
                { NodeType::MapRange, "Map Range" },
                { NodeType::FloatCurve, "Float Curve" },
                { NodeType::SeparateColor, "Separate RGB" },
                { NodeType::CombineColor, "Combine RGB" }
            } },
            { "Vector", ImVec4(0.45f, 0.4f, 0.75f, 1.0f), {
                { NodeType::Mapping, "Mapping" },
                { NodeType::Bump, "Bump" },
                { NodeType::Bevel, "Bevel" },
                { NodeType::VectorMath, "Vector Math" }
            } },
            { "Shader", ImVec4(0.8f, 0.35f, 0.5f, 1.0f), {
                { NodeType::MixMaterial, "Mix Material" },
                { NodeType::Output, "Material Output" }
            } }
        };
    }

    void reset() {
        editor.reset();
        activeMaterialName.clear();
        lastWarnings.clear();
        lastErrors.clear();
        hasApplied = false;
    }

    template<typename ContextT>
    void draw(ContextT& ctx,
              std::unordered_map<std::string, std::shared_ptr<MaterialNodeGraphV2>>& graphs) {
        auto& mm = MaterialManager::getInstance();

        // ------------------------------------------------------------------
        // Follow the scene selection: the selected object's material comes up
        // automatically; multi-material objects expose their full list in the
        // toolbar picker. Rescanned only when the selected object changes.
        // ------------------------------------------------------------------
        std::string curSel;
        if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
            curSel = ctx.selection.selected.object->getNodeName();
        }
        if (curSel != selObjectName) {
            selObjectName = curSel;
            selMaterials.clear();
            if (!selObjectName.empty()) {
                // Hittable has no node-name accessor — resolve it per concrete
                // type (TriangleMesh carries a plain nodeName member, Triangle
                // facades expose getNodeName()).
                std::unordered_set<uint16_t> seen;
                for (const auto& obj : ctx.scene.world.objects) {
                    if (!obj) continue;
                    if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                        if (tm->nodeName != selObjectName || !tm->geometry) continue;
                        const uint16_t* mats = tm->geometry->get_material_ids();
                        const size_t vc = tm->geometry->get_vertex_count();
                        if (mats) {
                            for (size_t i = 0; i < vc; ++i) {
                                if (mats[i] != MaterialManager::INVALID_MATERIAL_ID) seen.insert(mats[i]);
                            }
                        }
                    } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                        if (tri->getNodeName() != selObjectName) continue;
                        const uint16_t id = tri->getMaterialID();
                        if (id != MaterialManager::INVALID_MATERIAL_ID) seen.insert(id);
                    }
                }
                for (uint16_t id : seen) {
                    // Only PrincipledBSDF materials are graph-editable
                    if (dynamic_cast<PrincipledBSDF*>(mm.getMaterial(id))) {
                        selMaterials.emplace_back(id, mm.getMaterialName(id));
                    }
                }
                std::sort(selMaterials.begin(), selMaterials.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });
            }
            // Auto-bind: keep the current material if the new object also uses it,
            // otherwise jump to the object's first material.
            if (!selMaterials.empty()) {
                const bool currentInList = std::any_of(selMaterials.begin(), selMaterials.end(),
                    [this](const auto& p) { return p.second == activeMaterialName; });
                if (!currentInList) {
                    activeMaterialName = selMaterials.front().second;
                    lastWarnings.clear();
                    lastErrors.clear();
                    hasApplied = false;
                }
            }
        }

        // ------------------------------------------------------------------
        // Active material resolution
        // ------------------------------------------------------------------
        PrincipledBSDF* pbsdf = nullptr;
        uint16_t matId = MaterialManager::INVALID_MATERIAL_ID;
        if (!activeMaterialName.empty()) {
            matId = mm.getMaterialID(activeMaterialName);
            if (matId != MaterialManager::INVALID_MATERIAL_ID) {
                pbsdf = dynamic_cast<PrincipledBSDF*>(mm.getMaterial(matId));
            }
            if (!pbsdf) activeMaterialName.clear();  // deleted/renamed — drop stale binding
        }

        std::shared_ptr<MaterialNodeGraphV2> graphPtr;
        if (pbsdf) {
            auto& slot = graphs[activeMaterialName];
            if (!slot) slot = std::make_shared<MaterialNodeGraphV2>();
            if (slot->nodes.empty()) {
                // First open: the graph VISIBLY represents the material — bound
                // textures come in as wired Image Texture nodes, albedo color as
                // a Color node, everything else as Output defaults.
                materializeGraphFromMaterial(*slot, *pbsdf);
                // (The one-shot backend sync is driven by MaterialNodeGraphV2::
                // needsInitialApply, which a fresh graph and a disk-loaded graph both carry.)
            }
            if (auto* out = slot->findOutputNode()) {
                // Saved graphs from before a field-group existed (resin/bubble/
                // extended): fill those groups from the material's live values
                // instead of constructor defaults (no-op otherwise).
                out->seedMissingGroupsFromMaterial(*pbsdf);
                // Textures live ONLY in visible Image Texture nodes — any texture
                // still in the Output defaults (old saves, Pull) becomes a wired
                // node here. Otherwise deleting a texture node/link couldn't
                // unbind it: defaults would silently re-apply it.
                if (migrateDefaultTexturesToNodes(*slot, *out)) {
                    ProjectManager::getInstance().markModified();
                }
                // Socket groups: collapsed groups hide their unconnected pins.
                out->syncPinVisibility(*slot);
                // Auto-sync FROM the live material (replaces the manual Pull for
                // the live workflow): keeps regular Properties-panel edits and
                // texture swaps from being reverted by the graph's next Apply.
                // Idempotent for graph-side edits (see pullMaterialStateIntoGraph).
                if (liveApply) {
                    pullMaterialStateIntoGraph(*slot, *out, *pbsdf);
                }
            }
            graphPtr = slot;
        }

        // Right-click background menu (rebound each frame — captures this frame's graph)
        if (graphPtr) {
            editor.onDrawBackgroundMenu = [this, &graphPtr]() {
                bool hasOutput = (graphPtr->findOutputNode() != nullptr);
                const ImVec2 spawnPos = editor.mousePosOnRightClick;
                for (const auto& cat : categories) {
                    ImGui::PushStyleColor(ImGuiCol_Text, cat.color);
                    const bool open = ImGui::BeginMenu(cat.name);
                    ImGui::PopStyleColor();
                    if (!open) continue;
                    for (const auto& [type, label] : cat.nodes) {
                        const bool enabled = !(type == NodeType::Output && hasOutput);
                        if (ImGui::MenuItem(label, nullptr, false, enabled)) {
                            auto* n = graphPtr->addMaterialNode(type, spawnPos.x, spawnPos.y);
                            editor.onNodeAdded(*graphPtr, n);
                            ProjectManager::getInstance().markModified();
                            graphChangedThisFrame_ = true;
                        }
                    }
                    ImGui::EndMenu();
                }
            };
        }

        // ------------------------------------------------------------------
        // Layout: [library | splitter | strip] canvas
        //
        // There is no properties panel any more. Every node carries its own parameters on
        // its body, which is not only fewer places to look: the panel drew the SAME
        // drawContent() as the node body, twice per frame, against the same member state —
        // and that is what made the curve editor's drag jump to the bottom of the widget
        // (two live copies fighting over one drag, each mapping the mouse into its own rect).
        // With one drawer, that entire class of bug cannot happen again.
        // ------------------------------------------------------------------
        auto verticalSplitter = [](const char* id, float& width, bool leftPanel,
                                   float minW, float maxW) {
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::InvisibleButton(id, ImVec2(6.0f, ImGui::GetContentRegionAvail().y));
            if (ImGui::IsItemActive()) {
                width += leftPanel ? ImGui::GetIO().MouseDelta.x : -ImGui::GetIO().MouseDelta.x;
                width = std::clamp(width, minW, maxW);
            }
            if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            }
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImU32 col = (ImGui::IsItemHovered() || ImGui::IsItemActive())
                ? IM_COL32(100, 150, 255, 200) : IM_COL32(80, 80, 90, 120);
            dl->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), col);
            ImGui::SameLine(0.0f, 0.0f);
        };

        // Feed the Attribute node's channel picker. The node cannot reach for the scene
        // itself (MaterialNodesV2 sits below SceneData in the include graph), and this panel
        // is the only place that both draws it and holds a SceneData — same arrangement as
        // the session texture list the Image Texture node picks from.
        {
            auto& names = MaterialNodesV2::availableVertexAttributeNames();
            names.clear();
            std::unordered_set<const TriangleMesh*> seen;
            auto collect = [&](const TriangleMesh* tm) {
                if (!tm || !tm->geometry || !seen.insert(tm).second) return;
                for (auto& n : tm->geometry->listCustomAttributeNames()) {
                    if (std::find(names.begin(), names.end(), n) == names.end()) names.push_back(n);
                }
            };
            for (const auto& obj : ctx.scene.world.objects) {
                if (!obj) continue;
                if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) collect(tm.get());
                else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) collect(tri->parentMesh.get());
            }
            std::sort(names.begin(), names.end());
        }

        if (showLibrary) {
            ImGui::BeginChild("MatNodeLibrary", ImVec2(libraryWidth, 0), true);
            drawNodeLibrary();
            ImGui::EndChild();
            verticalSplitter("##MatLibSplitter", libraryWidth, true, 110.0f, 360.0f);
        }

        // The panel's own collapse handle, on its edge — the same place the main docked
        // panels put theirs. A toolbar button that hides a panel on the far side of the
        // window is a guessing game; a chevron ON the seam says what it does.
        {
            const float stripW = 13.0f;
            ImGui::BeginChild("##MatLibStrip", ImVec2(stripW, 0), false, ImGuiWindowFlags_NoScrollbar);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const ImVec2 sz = ImGui::GetContentRegionAvail();
            ImGui::InvisibleButton("##libToggle", ImVec2(stripW, std::max(24.0f, sz.y)));
            const bool hov = ImGui::IsItemHovered();
            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) showLibrary = !showLibrary;
            if (hov) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::SetTooltip(showLibrary ? "Hide node library" : "Show node library");
            }
            ImDrawList* dl = ImGui::GetWindowDrawList();
            if (hov) {
                dl->AddRectFilled(p, ImVec2(p.x + stripW, p.y + sz.y), IM_COL32(90, 130, 200, 60));
            }
            // Chevron points the way the panel will move.
            const float cx = p.x + stripW * 0.5f;
            const float cy = p.y + sz.y * 0.5f;
            const float a = 3.5f;
            const ImU32 col = hov ? IM_COL32(220, 230, 245, 255) : IM_COL32(130, 135, 145, 255);
            if (showLibrary) {
                dl->AddTriangleFilled(ImVec2(cx + a * 0.7f, cy - a), ImVec2(cx + a * 0.7f, cy + a),
                                      ImVec2(cx - a * 0.9f, cy), col);   // '<' : collapse
            } else {
                dl->AddTriangleFilled(ImVec2(cx - a * 0.7f, cy - a), ImVec2(cx - a * 0.7f, cy + a),
                                      ImVec2(cx + a * 0.9f, cy), col);   // '>' : expand
            }
            ImGui::EndChild();
            ImGui::SameLine(0.0f, 0.0f);
        }

        // ------------------------------------------------------------------
        // Canvas (toolbar + graph) — takes everything that is left
        // ------------------------------------------------------------------
        ImGui::BeginChild("MatNodeCanvas", ImVec2(0, 0), true, ImGuiWindowFlags_NoScrollbar);

        drawToolbar(ctx, graphPtr, pbsdf, matId);

        if (graphPtr) {
            const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
            const ImVec2 canvasSize = ImGui::GetContentRegionAvail();

            // Inline node-body widgets edit parameters during editor.draw() —
            // detect their dirty TRANSITIONS (same rule as the properties
            // panel) so live apply picks them up.
            preDirty_.clear();
            for (const auto& n : graphPtr->nodes) preDirty_.emplace_back(n->id, n->dirty);

            editor.draw(*graphPtr, canvasSize);

            for (const auto& n : graphPtr->nodes) {
                if (!n->dirty) continue;
                const auto it = std::find_if(preDirty_.begin(), preDirty_.end(),
                    [&](const auto& p) { return p.first == n->id; });
                if (it == preDirty_.end() || !it->second) {
                    ProjectManager::getInstance().markModified();
                    graphChangedThisFrame_ = true;
                    break;
                }
            }

            // Drag-drop from the library
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MATERIAL_NODE_TYPE")) {
                    const NodeType droppedType = *static_cast<const NodeType*>(payload->Data);
                    if (!(droppedType == NodeType::Output && graphPtr->findOutputNode())) {
                        const ImVec2 mousePos = ImGui::GetMousePos();
                        const float spawnX = (mousePos.x - canvasPos.x - editor.scrollX) / editor.zoom;
                        const float spawnY = (mousePos.y - canvasPos.y - editor.scrollY) / editor.zoom;
                        auto* n = graphPtr->addMaterialNode(droppedType, spawnX, spawnY);
                        editor.onNodeAdded(*graphPtr, n);
                        ProjectManager::getInstance().markModified();
                        graphChangedThisFrame_ = true;
                    }
                }
                ImGui::EndDragDropTarget();
            }

            // Image Texture "Browse...": the button lives on the NODE, so the host-side file
            // dialog has to be driven from here now (it used to hang off the properties panel,
            // which only ever drew the SELECTED node — meaning Browse on any other node was
            // silently dead). Scanning every node fixes that as a side effect.
            for (const auto& n : graphPtr->nodes) {
                auto* img = dynamic_cast<ImageTextureNode*>(n.get());
                if (!img || !img->browseRequested) continue;
                img->browseRequested = false;
                if (!onOpenFileDialog) continue;
                const std::string path = onOpenFileDialog(
                    L"Image Files\0*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.hdr;*.exr\0");
                if (path.empty()) continue;
                auto tex = std::make_shared<Texture>(path, TextureType::Albedo);
                if (tex->is_loaded()) {
                    img->setTexture(tex);
                    ProjectManager::getInstance().markModified();
                    graphChangedThisFrame_ = true;
                }
            }
        } else {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Pick a material in the toolbar to edit its node graph.");
        }
        ImGui::EndChild();

        // ------------------------------------------------------------------
        // Live apply: any edit this frame (links, nodes, parameters, texture
        // picks — flagged above and by editor.onGraphModified) folds into the
        // material immediately. Same per-edit cost as a material-panel slider.
        // ------------------------------------------------------------------
        if (graphChangedThisFrame_) {
            if (liveApply && graphPtr && pbsdf) {
                applyGraph(ctx, *graphPtr, pbsdf, matId);
            }
            graphChangedThisFrame_ = false;
        } else if (graphPtr && pbsdf && liveApply && graphPtr->needsInitialApply) {
            // One-shot sync for a graph this session has not applied yet. It does two jobs,
            // and the second one is why an OPENED PROJECT used to be dead to node edits:
            //   1. pushes the folded material + per-pixel program to the backend, so Vulkan
            //      RT shows the saved graph without waiting for a first edit;
            //   2. clears the dirty flag every node is BORN with, which is what live apply's
            //      false->true transition detection needs in order to ever fire.
            // This used to be gated on "the graph was just MATERIALIZED from the material"
            // (slot->nodes.empty()), so a project whose graph came off DISK — nodes already
            // populated — never got it, and stayed deaf until Live was toggled off and on.
            // markProjectModified=false: this is semantically what is already on disk.
            applyGraph(ctx, *graphPtr, pbsdf, matId, /*markProjectModified=*/false);
        }
    }

private:
    struct NodeCategory {
        const char* name;
        ImVec4 color;
        std::vector<std::pair<NodeType, const char*>> nodes;
    };

    std::vector<NodeCategory> categories;
    char searchBuffer[128] = "";
    NodeSystem::NodeEditorUIV2 editor;
    bool graphChangedThisFrame_ = false;  ///< any edit this frame -> live apply at end of draw()
    std::vector<std::pair<uint32_t, bool>> preDirty_;  ///< per-node dirty snapshot around editor.draw()

    void drawNodeLibrary() {
        ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.8f, 1.0f), "Material Nodes");
        ImGui::Separator();
        ImGui::PushItemWidth(-1);
        ImGui::InputTextWithHint("##MatSearch", "Search...", searchBuffer, sizeof(searchBuffer));
        ImGui::PopItemWidth();
        ImGui::Spacing();

        std::string searchStr = searchBuffer;
        std::transform(searchStr.begin(), searchStr.end(), searchStr.begin(), ::tolower);

        int catIdx = 0;
        for (auto& category : categories) {
            // Category header and node buttons can share a label (the "Color"
            // category contains a "Color" node) — scope every category and item
            // with PushID so labels never collide in the ImGui ID stack.
            ImGui::PushID(catIdx++);
            ImGui::PushStyleColor(ImGuiCol_Header, category.color);
            const bool categoryOpen = ImGui::CollapsingHeader(category.name, ImGuiTreeNodeFlags_DefaultOpen);
            ImGui::PopStyleColor();
            if (!categoryOpen) { ImGui::PopID(); continue; }

            ImGui::Indent(8);
            for (auto& [type, label] : category.nodes) {
                std::string nameLower = label;
                std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                if (!searchStr.empty() && nameLower.find(searchStr) == std::string::npos) continue;

                ImGui::PushID(static_cast<int>(type));
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
                ImGui::Button(label, ImVec2(-1, 0));
                if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                    ImGui::SetDragDropPayload("MATERIAL_NODE_TYPE", &type, sizeof(NodeType));
                    ImGui::Text("+ %s", label);
                    ImGui::EndDragDropSource();
                }
                if (ImGui::IsItemHovered() && !ImGui::IsMouseDragging(0)) {
                    ImGui::SetTooltip("Drag to canvas");
                }
                ImGui::PopStyleColor();
                ImGui::PopID();
            }
            ImGui::Unindent(8);
            ImGui::PopID();
        }
    }

    template<typename ContextT>
    void drawToolbar(ContextT& ctx, std::shared_ptr<MaterialNodeGraphV2>& graphPtr,
                     PrincipledBSDF* pbsdf, uint16_t matId) {
        auto& mm = MaterialManager::getInstance();

        auto selectMaterial = [this](const std::string& name) {
            activeMaterialName = name;
            lastWarnings.clear();
            lastErrors.clear();
            hasApplied = false;
        };

        // Material picker: the selected OBJECT's material(s) when one is
        // selected (multi-material objects list all of theirs), otherwise the
        // full session material list.
        ImGui::SetNextItemWidth(180);
        if (!selMaterials.empty()) {
            std::string preview = activeMaterialName;
            if (selMaterials.size() > 1) preview += " (" + std::to_string(selMaterials.size()) + ")";
            if (ImGui::BeginCombo("##MatGraphTarget", preview.c_str())) {
                for (const auto& [id, name] : selMaterials) {
                    if (ImGui::Selectable(name.c_str(), name == activeMaterialName)) selectMaterial(name);
                }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Materials of \"%s\"", selObjectName.c_str());
            }
        } else {
            const char* preview = activeMaterialName.empty() ? "<select material>" : activeMaterialName.c_str();
            if (ImGui::BeginCombo("##MatGraphTarget", preview)) {
                for (const auto& mat : mm.getAllMaterials()) {
                    if (!mat || mat->type() != MaterialType::PrincipledBSDF) continue;
                    if (ImGui::Selectable(mat->materialName.c_str(), mat->materialName == activeMaterialName)) {
                        selectMaterial(mat->materialName);
                    }
                }
                ImGui::EndCombo();
            }
        }

        ImGui::SameLine();
        ImGui::BeginDisabled(!graphPtr || !pbsdf);
        if (ImGui::Checkbox("Live", &liveApply) && liveApply && graphPtr && pbsdf) {
            // Re-entering live mode: sync the material to the graph's current state.
            applyGraph(ctx, *graphPtr, pbsdf, matId);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Apply every graph edit immediately");
        if (!liveApply) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.3f, 1.0f));
            if (ImGui::Button("Apply")) {
                applyGraph(ctx, *graphPtr, pbsdf, matId);
            }
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Apply the graph to the material (all backends)");
        }
        // Pull is only needed in manual mode — with Live on, the graph auto-syncs
        // FROM the material every frame (pullMaterialStateIntoGraph), so Properties-
        // panel edits and texture swaps are never reverted by Apply.
        if (!liveApply) {
            ImGui::SameLine();
            if (ImGui::Button("Pull")) {
                if (auto* out = graphPtr->findOutputNode()) {
                    out->initDefaultsFromMaterial(*pbsdf);
                    ProjectManager::getInstance().markModified();
                    graphChangedThisFrame_ = true;
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Re-seed the Output node's fallback values from the material's current state");
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Rebuild")) {
            ImGui::OpenPopup("MatGraphRebuildConfirm");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Rebuild the graph from the material's current textures/colors\n(REPLACES the whole graph)");
        }
        if (ImGui::BeginPopup("MatGraphRebuildConfirm")) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Replace the graph with the material's current state?");
            if (ImGui::Button("Rebuild##confirm")) {
                materializeGraphFromMaterial(*graphPtr, *pbsdf);
                lastWarnings.clear();
                lastErrors.clear();
                hasApplied = false;
                ProjectManager::getInstance().markModified();
                graphChangedThisFrame_ = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel##confirm")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
        ImGui::EndDisabled();

        // Status chip, right-aligned. The diagnostics used to sit in a permanent scroll block
        // under the toolbar, eating canvas height to tell you things were FINE. Now the
        // toolbar states the result in one word and only opens up when you ask it to.
        {
            const int nErr = static_cast<int>(lastErrors.size());
            const int nWarn = static_cast<int>(lastWarnings.size());

            char chip[64];
            ImVec4 chipCol;
            if (nErr > 0) {
                snprintf(chip, sizeof(chip), "%d error%s", nErr, nErr == 1 ? "" : "s");
                chipCol = ImVec4(0.75f, 0.22f, 0.22f, 1.0f);
            } else if (nWarn > 0) {
                snprintf(chip, sizeof(chip), "%d warning%s", nWarn, nWarn == 1 ? "" : "s");
                chipCol = ImVec4(0.62f, 0.48f, 0.15f, 1.0f);
            } else if (hasApplied && lastApplyOk) {
                snprintf(chip, sizeof(chip), "OK");
                chipCol = ImVec4(0.20f, 0.42f, 0.28f, 1.0f);
            } else {
                chip[0] = '\0';
            }

            char stats[48] = "";
            if (graphPtr) {
                snprintf(stats, sizeof(stats), "%d nodes  %d links",
                         static_cast<int>(graphPtr->nodeCount()), static_cast<int>(graphPtr->linkCount()));
            }

            float need = 0.0f;
            if (stats[0]) need += ImGui::CalcTextSize(stats).x + 12.0f;
            if (chip[0])  need += ImGui::CalcTextSize(chip).x + 18.0f;
            ImGui::SameLine();
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                                 std::max(0.0f, ImGui::GetContentRegionAvail().x - need));

            if (stats[0]) {
                ImGui::TextDisabled("%s", stats);
                if (chip[0]) ImGui::SameLine();
            }
            if (chip[0]) {
                ImGui::PushStyleColor(ImGuiCol_Button, chipCol);
                if (ImGui::SmallButton(chip)) ImGui::OpenPopup("##matDiagPopup");
                ImGui::PopStyleColor();
                if ((nErr || nWarn) && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Click for details");
                }
            }
            if (ImGui::BeginPopup("##matDiagPopup")) {
                for (const auto& e : lastErrors) {
                    ImGui::TextColored(ImVec4(1.0f, 0.40f, 0.40f, 1.0f), "Error: %s", e.c_str());
                }
                for (const auto& w : lastWarnings) {
                    ImGui::TextColored(ImVec4(1.0f, 0.80f, 0.30f, 1.0f), "Warning: %s", w.c_str());
                }
                if (lastErrors.empty() && lastWarnings.empty()) {
                    ImGui::TextDisabled("Graph applied cleanly.");
                }
                ImGui::EndPopup();
            }
        }
        ImGui::Separator();
    }

    /**
     * @brief Fold the graph into the material and push it through the SAME
     * update chain a material-panel slider edit uses. No bake, no new GPU path.
     */
    template<typename ContextT>
    void applyGraph(ContextT& ctx, MaterialNodeGraphV2& graph, PrincipledBSDF* pbsdf, uint16_t matId,
                    bool markProjectModified = true) {
        MaterialGraphResult res = evaluateMaterialGraph(graph, pbsdf);
        lastWarnings = res.warnings;
        lastErrors = res.errors;
        lastApplyOk = res.ok;
        hasApplied = true;
        if (!res.ok || !pbsdf) return;

        const bool texChanged = applyShadeStateToMaterial(res.state, *pbsdf);

        // CPU -> GPU struct (the canonical sync, see MaterialManager::syncAllGpuMaterials)
        if (!pbsdf->gpuMaterial) pbsdf->gpuMaterial = std::make_shared<GpuMaterial>();
        const PBRMaterialSnapshot snapshot = capturePBRMaterialSnapshot(*pbsdf);
        applyPBRMaterialSnapshotToGpuMaterial(snapshot, *pbsdf->gpuMaterial);

        // Compile the spatially-varying chains into a per-pixel program. Both the
        // CPU render (Faz 2a) and Vulkan RT (Faz 2b, closesthit VM) consume it, so
        // these slots shade per-pixel on both. active=false (all-constant / direct
        // texture bind) clears it -> zero per-pixel cost, unchanged behaviour.
        const bool hadPointiness = pbsdf->proceduralProgram && pbsdf->proceduralProgram->usesPointiness;
        const bool hadAttributes = pbsdf->proceduralProgram && pbsdf->proceduralProgram->usesAttributes;
        const size_t attrSlotsBefore = MaterialNodesV2::materialAttributeSlots().size();
        {
            MaterialProgram prog = compileMaterialProgram(graph, pbsdf);
            const int drivenCount = prog.active ?
                [&] { int c = 0; for (uint32_t s = 0; s < static_cast<uint32_t>(MatSlot::Count); ++s)
                                     if (prog.drivenSlots & (1u << s)) ++c; return c; }() : 0;
            if (prog.active) pbsdf->proceduralProgram = std::make_shared<MaterialProgram>(std::move(prog));
            else pbsdf->proceduralProgram.reset();
            if (drivenCount > 0) {
                lastWarnings.push_back(std::to_string(drivenCount) +
                    " slot(s) shade PER-PIXEL on CPU + Vulkan RT; the frozen OptiX backend uses the folded average");
            }
        }

        // Pointiness is the one Geometry output that isn't free at the shading point: it
        // needs a per-vertex precompute (CPU caches, built in rebuildBVH) and a per-vertex
        // GPU block (uploaded with the BLAS). Both are lazy, so the FIRST graph to read it
        // has to force one geometry pass. Strictly on the transition (or when the geometry
        // on the GPU predates it) — otherwise every slider tweak would rebuild the scene.
        // The project-load sync (markProjectModified == false) rebuilds geometry itself.
        const bool usesPointiness = pbsdf->proceduralProgram && pbsdf->proceduralProgram->usesPointiness;
        // The Attribute node needs the exact same one-time geometry pass, plus one extra
        // trigger the pointiness gate has no equivalent of: picking a DIFFERENT attribute
        // name interns a NEW slot, and the per-vertex blocks already on the GPU carry only
        // the old slots. Without the slot-count check the newly chosen channel would read 0
        // everywhere until something else happened to rebuild the geometry.
        const bool usesAttributes = pbsdf->proceduralProgram && pbsdf->proceduralProgram->usesAttributes;
        const bool attrSlotsGrew  = MaterialNodesV2::materialAttributeSlots().size() != attrSlotsBefore;
        const bool needAttribPass = usesAttributes &&
            (!hadAttributes || attrSlotsGrew ||
             (ctx.backend_ptr && !ctx.backend_ptr->geometryHasAttributes()));
        const bool needPointinessPass = usesPointiness &&
            (!hadPointiness || (ctx.backend_ptr && !ctx.backend_ptr->geometryHasPointiness()));

        if (markProjectModified && (needPointinessPass || needAttribPass)) {
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            if (ctx.backend_ptr) ctx.renderer.rebuildBackendGeometry(ctx.scene);
        }

        if (usesAttributes) {
            // An Attribute node whose name is not interned compiled to a constant 0 — say so,
            // or the user just sees a black material and has no idea the budget ran out.
            for (const auto& n : graph.nodes) {
                auto* an = dynamic_cast<MaterialNodesV2::AttributeNode*>(n.get());
                if (!an) continue;
                if (an->attributeName.empty()) {
                    lastWarnings.push_back("Attribute node has no channel selected - reads 0");
                } else if (MaterialNodesV2::findMaterialAttributeSlot(an->attributeName) < 0) {
                    lastWarnings.push_back("Attribute '" + an->attributeName + "': all " +
                        std::to_string(MaterialNodesV2::kMatAttribSlots) +
                        " attribute slots are in use - reads 0");
                }
            }
        }

        if (texChanged) {
            // OptiX per-triangle texture bundles — same loop as the material
            // panel's texture_changed path (scene_ui_materials.cpp).
            for (auto& obj : ctx.scene.world.objects) {
                auto t = std::dynamic_pointer_cast<Triangle>(obj);
                if (!t || t->getMaterialID() != matId) continue;
                OptixGeometryData::TextureBundle bundle = {};
                auto SetupTex = [](std::shared_ptr<Texture>& tex, cudaTextureObject_t& outTex, int& outHas) {
                    if (tex && tex->is_loaded()) {
                        tex->upload_to_gpu();
                        outTex = tex->get_cuda_texture();
                        outHas = 1;
                    } else {
                        outTex = 0;
                        outHas = 0;
                    }
                };
                SetupTex(pbsdf->albedoProperty.texture, bundle.albedo_tex, bundle.has_albedo_tex);
                SetupTex(pbsdf->normalProperty.texture, bundle.normal_tex, bundle.has_normal_tex);
                SetupTex(pbsdf->roughnessProperty.texture, bundle.roughness_tex, bundle.has_roughness_tex);
                SetupTex(pbsdf->metallicProperty.texture, bundle.metallic_tex, bundle.has_metallic_tex);
                SetupTex(pbsdf->emissionProperty.texture, bundle.emission_tex, bundle.has_emission_tex);
                SetupTex(pbsdf->opacityProperty.texture, bundle.opacity_tex, bundle.has_opacity_tex);
                t->setTextureBundle(bundle);
            }
        }

        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterial(ctx.scene, matId);
            // Faz 2b: re-upload the per-pixel program buffer so Vulkan RT reflects
            // Noise/ColorRamp edits live (updateBackendMaterial only refreshes the
            // folded VkGpuMaterial, not the program stream). No-op off Vulkan.
            ctx.renderer.syncMaterialProgramsToBackend(ctx.backend_ptr);
            ctx.backend_ptr->resetAccumulation();
        }
        // First-open sync (markProjectModified == false) must NOT dirty the project:
        // it only pushes the freshly-materialized graph's folded material + per-pixel
        // program to the backend, which is semantically identical to what's on disk.
        if (markProjectModified) ProjectManager::getInstance().markModified();

        // Clear every node's dirty flag so next frame's edit detection (a false->true
        // transition) fires reliably. evaluate only clears nodes it actually FOLDS;
        // nodes on binding-only chains (e.g. a Bump on the Normal Map slot) are never
        // evaluated, so markAllDirty leaves them stuck dirty — which would swallow all
        // their later param edits (strength/distance wouldn't re-apply live).
        //
        // This is ALSO what arms live apply for a graph loaded from disk: its nodes are born
        // dirty (NodeBase::dirty = true), and until they are cleared once no edit can produce
        // the false->true transition live apply looks for. See MaterialNodeGraphV2::
        // needsInitialApply — this function is the only place that clears it.
        for (auto& n : graph.nodes) n->dirty = false;
        graph.needsInitialApply = false;
    }
};

} // namespace MaterialNodesV2
