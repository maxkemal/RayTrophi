#include "scene_ui.h"
#include "ui_modern.h"
#include "MeshModifiers.h"
#include "ProjectManager.h"
#include "globals.h"
#include "Triangle.h"
#include "Renderer.h"
#include "scene_data.h"
#include <SceneSelection.h>

void SceneUI::drawModifiersPanel(UIContext& ctx) {
    if (UIWidgets::BeginSection("Modifiers & Sculpting", ImVec4(0.8f, 0.4f, 0.9f, 1.0f))) {
        bool hasSelection = (ctx.selection.selected.type == SelectableType::Object &&
                             ctx.selection.selected.object != nullptr);

        if (!hasSelection) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.6f, 1.0f), "Please select a mesh object.");
            UIWidgets::EndSection();
            return;
        }

        std::string selectedNodeName = ctx.selection.selected.object->getNodeName();
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Selected: %s", selectedNodeName.c_str());
        UIWidgets::Divider();

        // 1. Lazy initialize base mesh cache if not present
        if (ctx.scene.base_mesh_cache.find(selectedNodeName) == ctx.scene.base_mesh_cache.end()) {
            std::vector<std::shared_ptr<Triangle>> baseTriangles;
            for (const auto& obj : ctx.scene.world.objects) {
                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                if (tri && tri->getNodeName() == selectedNodeName) {
                    baseTriangles.push_back(tri);
                }
            }
            if (!baseTriangles.empty()) {
                ctx.scene.base_mesh_cache[selectedNodeName] = baseTriangles;
            }
        }

        // Get reference to stack
        auto& modifierStack = ctx.scene.mesh_modifiers[selectedNodeName];

        bool stackChanged = false;

        // 2. Draw existing modifiers
        for (size_t i = 0; i < modifierStack.modifiers.size(); ++i) {
            auto& mod = modifierStack.modifiers[i];
            
            ImGui::PushID(static_cast<int>(i));
            if (UIWidgets::BeginSection(mod.name.c_str(), ImVec4(0.4f, 0.6f, 0.9f, 1.0f))) {
                
                if (ImGui::Checkbox("Enabled", &mod.enabled)) stackChanged = true;
                
                ImGui::SameLine();
                if (ImGui::Button("Delete")) {
                    modifierStack.modifiers.erase(modifierStack.modifiers.begin() + i);
                    stackChanged = true;
                    UIWidgets::EndSection();
                    ImGui::PopID();
                    break; // iterator invalidated
                }

                if (mod.type == MeshModifiers::ModifierType::FlatSubdivision || mod.type == MeshModifiers::ModifierType::SmoothSubdivision) {
                    if (ImGui::SliderInt("Levels", &mod.levels, 1, 4)) stackChanged = true;
                }
                
                if (mod.type == MeshModifiers::ModifierType::SmoothSubdivision) {
                    if (ImGui::SliderFloat("Smooth Weight", &mod.smoothAngle, 0.0f, 1.0f)) stackChanged = true;
                }

                UIWidgets::EndSection();
            }
            ImGui::PopID();
        }

        UIWidgets::Divider();

        // 3. Add new modifier
        UIWidgets::ColoredHeader("Add Modifier", ImVec4(0.5f, 0.8f, 1.0f, 1.0f));
        if (ImGui::Button("Flat Subdivision", ImVec2(-1, 0))) {
            MeshModifiers::ModifierData newMod;
            newMod.name = "Flat Subdivision";
            newMod.type = MeshModifiers::ModifierType::FlatSubdivision;
            modifierStack.modifiers.push_back(newMod);
            stackChanged = true;
        }
        if (ImGui::Button("Smooth Subdivision", ImVec2(-1, 0))) {
            MeshModifiers::ModifierData newMod;
            newMod.name = "Smooth Subdivision";
            newMod.type = MeshModifiers::ModifierType::SmoothSubdivision;
            modifierStack.modifiers.push_back(newMod);
            stackChanged = true;
        }

        // 4. Evaluate stack if changed
        if (stackChanged) {
            SCENE_LOG_INFO("Evaluating Modifier Stack for '" + selectedNodeName + "'...");
            
            // 4a. Get base geometry
            const auto& baseMesh = ctx.scene.base_mesh_cache[selectedNodeName];
            
            // 4b. Evaluate
            auto newMesh = modifierStack.evaluate(baseMesh);
            
            // 4c. Replace in scene
            std::vector<std::shared_ptr<Hittable>> remainingObjects;
            for (const auto& obj : ctx.scene.world.objects) {
                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                if (!tri || tri->getNodeName() != selectedNodeName) {
                    remainingObjects.push_back(obj);
                }
            }
            
            for (const auto& tri : newMesh) {
                remainingObjects.push_back(tri);
            }
            
            ctx.scene.world.objects = remainingObjects;

            // Make sure selection is valid
            if (!newMesh.empty()) {
                ctx.selection.selected.object = newMesh[0];
            } else if (!baseMesh.empty()) {
                ctx.selection.selected.object = baseMesh[0];
            } // else handling deleting all?
            
            SCENE_LOG_INFO("Evaluated mesh '" + selectedNodeName + "': " + std::to_string(baseMesh.size()) + " -> " + std::to_string(newMesh.size()) + " triangles.");

            // 4d. Rebuild acceleration structures
            rebuildMeshCache(ctx.scene.world.objects);
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);

            addViewportMessage("Modifiers Updated");
            g_ProjectManager.markModified();
        }

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("WIP Features:", ImVec4(0.8f, 0.4f, 0.9f, 1.0f));
        ImGui::BulletText("Mesh Sculpting Brush");

        UIWidgets::EndSection();
    }
}
