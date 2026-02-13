/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_forcefield.hpp
* Author:        Kemal Demirtaş
* Date:          January 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
/**
 * @file scene_ui_forcefield.hpp
 * @brief UI Panel for Force Fields
 * 
 * Provides controls for:
 * - Force field list management
 * - Force field type and shape selection
 * - Strength, falloff, and noise parameters
 * - Per-system affect masks
 */

#pragma once

#include "scene_ui.h"
#include "ForceField.h"
#include <memory>
#include <string>

namespace ForceFieldUI {

// Currently selected force field for UI
inline std::shared_ptr<Physics::ForceField> selected_force_field = nullptr;

/**
 * @brief Draw the Force Field panel content
 */
inline void drawForceFieldPanel(UIContext& ui_ctx, SceneData& scene) {
    auto& manager = scene.force_field_manager;
    
    // ═══════════════════════════════════════════════════════════════════════
    // FORCE FIELD LIST
    // ═══════════════════════════════════════════════════════════════════════
    
    ImGui::Text("Force Fields (%d)", static_cast<int>(manager.force_fields.size()));
    ImGui::Separator();
    
    // Add new force field dropdown
    if (ImGui::Button("+ Add Force Field", ImVec2(-1, 0))) {
        ImGui::OpenPopup("AddForceFieldPopup");
    }
    
    if (ImGui::BeginPopup("AddForceFieldPopup")) {
        const char* types[] = { 
            "Wind", "Gravity", "Attractor", "Repeller", 
            "Vortex", "Turbulence", "Curl Noise", "Drag", "Magnetic"
        };
        
        for (int i = 0; i < 9; ++i) {
            if (ImGui::MenuItem(types[i])) {
                auto field = std::make_shared<Physics::ForceField>();
                field->name = std::string(types[i]) + " Field " + std::to_string(manager.force_fields.size() + 1);
                field->type = static_cast<Physics::ForceFieldType>(i);
                
                // Set defaults based on type
                switch (field->type) {
                    case Physics::ForceFieldType::Turbulence:
                    case Physics::ForceFieldType::CurlNoise:
                        field->use_noise = true;
                        field->shape = Physics::ForceFieldShape::Sphere;
                        break;
                    case Physics::ForceFieldType::Vortex:
                        field->shape = Physics::ForceFieldShape::Cylinder;
                        field->inward_force = 0.5f;
                        field->upward_force = 0.2f;
                        break;
                    case Physics::ForceFieldType::Drag:
                        field->shape = Physics::ForceFieldShape::Sphere;
                        field->linear_drag = 0.5f;
                        break;
                    default:
                        break;
                }
                
                manager.addForceField(field);
                ui_ctx.selection.selectForceField(field, -1, field->name);
                selected_force_field = field;
            }
        }
        ImGui::EndPopup();
    }
    
    // List existing force fields
    ImGui::BeginChild("ForceFieldList", ImVec2(-1, 150), true);
    for (size_t i = 0; i < manager.force_fields.size(); ++i) {
        auto& field = manager.force_fields[i];
        if (!field) continue;
        
        bool is_selected = (selected_force_field == field);
        
        // Professional geometric icons
        UIWidgets::IconType icon_type = UIWidgets::IconType::Force;
        switch (field->type) {
            case Physics::ForceFieldType::Wind:      icon_type = UIWidgets::IconType::Wind; break;
            case Physics::ForceFieldType::Gravity:   icon_type = UIWidgets::IconType::Gravity; break;
            case Physics::ForceFieldType::Vortex:    icon_type = UIWidgets::IconType::Vortex; break;
            case Physics::ForceFieldType::Turbulence:
            case Physics::ForceFieldType::CurlNoise: icon_type = UIWidgets::IconType::Noise; break;
            case Physics::ForceFieldType::Magnetic:  icon_type = UIWidgets::IconType::Magnet; break;
            case Physics::ForceFieldType::Attractor:
            case Physics::ForceFieldType::Repeller:
            case Physics::ForceFieldType::Drag:      icon_type = UIWidgets::IconType::Physics; break;
            default: break;
        }
        
        ImVec2 pos = ImGui::GetCursorScreenPos();
        UIWidgets::DrawIcon(icon_type, ImVec2(pos.x, pos.y), 16, 
            is_selected ? ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.9f, 0.8f, 1.0f)) : ImGui::ColorConvertFloat4ToU32(ImVec4(0.7f, 0.7f, 0.7f, 1.0f)), 1.0f);
        
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 20);
        std::string label = field->name + "##ff" + std::to_string(i);
        
        // Dim if disabled
        if (!field->enabled) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
        }
        
        if (ImGui::Selectable(label.c_str(), is_selected)) {
            ui_ctx.selection.selectForceField(field, -1, field->name);
            selected_force_field = field;
        }
        
        if (!field->enabled) {
            ImGui::PopStyleColor();
        }
        
        // Right-click context menu
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Delete")) {
                manager.removeForceField(field);
                if (selected_force_field == field) {
                    selected_force_field = nullptr;
                }
            }
            if (ImGui::MenuItem("Duplicate")) {
                auto copy = std::make_shared<Physics::ForceField>(*field);
                copy->name += " Copy";
                manager.addForceField(copy);
            }
            ImGui::Separator();
            if (ImGui::MenuItem(field->enabled ? "Disable" : "Enable")) {
                field->enabled = !field->enabled;
            }
            ImGui::EndPopup();
        }
    }
    ImGui::EndChild();
    
    ImGui::Separator();
    
    // ═══════════════════════════════════════════════════════════════════════
    // SELECTED FORCE FIELD PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════
    
    if (!selected_force_field) {
        // Sync from unified selection if local is null
        if (ui_ctx.selection.selected.type == SelectableType::ForceField) {
            selected_force_field = ui_ctx.selection.selected.force_field;
        }
    }
    
    if (!selected_force_field) {
        ImGui::TextDisabled("Select a force field to edit properties");
        return;
    }
    
    auto& field = selected_force_field;
    
    // Name
    char name_buf[128];
    strncpy_s(name_buf, field->name.c_str(), sizeof(name_buf) - 1);
    if (ImGui::InputText("Name", name_buf, sizeof(name_buf))) {
        field->name = name_buf;
    }
    
    ImGui::Checkbox("Enabled", &field->enabled);
    ImGui::SameLine();
    ImGui::Checkbox("Visible", &field->visible);
    
    ImGui::Spacing();
    
    // ─────────────────────────────────────────────────────────────────────────
    // TYPE & SHAPE
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // TYPE & SHAPE
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Type & Shape", ImVec4(0.5f, 0.9f, 0.5f, 1.0f))) {
        const char* types[] = { 
            "Wind", "Gravity", "Attractor", "Repeller", 
            "Vortex", "Turbulence", "Curl Noise", "Drag", "Magnetic", "Directional Noise"
        };
        int current_type = static_cast<int>(field->type);
        if (ImGui::Combo("Type", &current_type, types, 10)) {
            field->type = static_cast<Physics::ForceFieldType>(current_type);
        }
        
        const char* shapes[] = { "Infinite", "Sphere", "Box", "Cylinder", "Cone" };
        int current_shape = static_cast<int>(field->shape);
        if (ImGui::Combo("Shape", &current_shape, shapes, 5)) {
            field->shape = static_cast<Physics::ForceFieldShape>(current_shape);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // TRANSFORM
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        float pos[3] = { field->position.x, field->position.y, field->position.z };
        if (ImGui::DragFloat3("Position", pos, 0.1f)) {
            field->position = Vec3(pos[0], pos[1], pos[2]);
        }
        
        float rot[3] = { field->rotation.x, field->rotation.y, field->rotation.z };
        if (ImGui::DragFloat3("Rotation", rot, 1.0f, -180.0f, 180.0f)) {
            field->rotation = Vec3(rot[0], rot[1], rot[2]);
        }
        
        float scale[3] = { field->scale.x, field->scale.y, field->scale.z };
        if (ImGui::DragFloat3("Scale", scale, 0.1f, 0.1f, 10.0f)) {
            field->scale = Vec3(scale[0], scale[1], scale[2]);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // FORCE PARAMETERS
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Force Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat("Strength", &field->strength, 0.1f, -100.0f, 100.0f);
        
        // Direction for Wind/Gravity
        if (field->type == Physics::ForceFieldType::Wind || 
            field->type == Physics::ForceFieldType::Gravity ||
            field->type == Physics::ForceFieldType::Magnetic) {
            float dir[3] = { field->direction.x, field->direction.y, field->direction.z };
            if (ImGui::DragFloat3("Direction", dir, 0.1f, -1.0f, 1.0f)) {
                field->direction = Vec3(dir[0], dir[1], dir[2]);
                // Normalize
                float len = field->direction.length();
                if (len > 0.001f) field->direction = field->direction / len;
            }
        }
        
        // Vortex-specific
        if (field->type == Physics::ForceFieldType::Vortex) {
            ImGui::Separator();
            ImGui::Text("Vortex Settings");
            
            float axis[3] = { field->axis.x, field->axis.y, field->axis.z };
            if (ImGui::DragFloat3("Axis", axis, 0.1f, -1.0f, 1.0f)) {
                field->axis = Vec3(axis[0], axis[1], axis[2]);
                float len = field->axis.length();
                if (len > 0.001f) field->axis = field->axis / len;
            }
            
            ImGui::DragFloat("Inward Force", &field->inward_force, 0.1f, 0.0f, 10.0f);
            ImGui::DragFloat("Upward Force", &field->upward_force, 0.1f, -10.0f, 10.0f);
        }
        
        // Drag-specific
        if (field->type == Physics::ForceFieldType::Drag) {
            ImGui::Separator();
            ImGui::Text("Drag Settings");
            ImGui::DragFloat("Linear Drag", &field->linear_drag, 0.01f, 0.0f, 2.0f);
            ImGui::DragFloat("Quadratic Drag", &field->quadratic_drag, 0.001f, 0.0f, 1.0f);
        }

        // Falloff
        if (field->shape != Physics::ForceFieldShape::Infinite) {
            ImGui::Separator();
            ImGui::Text("Falloff Settings");
            const char* falloff_types[] = { 
                "None", "Linear", "Smooth", "Sphere", "Inverse Square", "Exponential", "Custom" 
            };
            int current_falloff = static_cast<int>(field->falloff_type);
            if (ImGui::Combo("Falloff Type", &current_falloff, falloff_types, 7)) {
                field->falloff_type = static_cast<Physics::FalloffType>(current_falloff);
            }
            
            ImGui::DragFloat("Inner Radius", &field->inner_radius, 0.1f, 0.0f, field->falloff_radius);
            ImGui::DragFloat("Falloff Radius", &field->falloff_radius, 0.1f, field->inner_radius, 100.0f);
        }
        UIWidgets::EndSection();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // NOISE & TURBULENCE
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Noise & Turbulence", ImVec4(1.0f, 0.9f, 0.5f, 1.0f))) {
        // NOISE (for Turbulence/CurlNoise/Wind)
        if (field->type == Physics::ForceFieldType::Turbulence || 
            field->type == Physics::ForceFieldType::CurlNoise ||
            field->type == Physics::ForceFieldType::Wind) {
            ImGui::Checkbox("Use Noise", &field->use_noise);
            
            if (field->use_noise) {
                ImGui::DragFloat("Frequency", &field->noise.frequency, 0.01f, 0.01f, 10.0f);
                ImGui::DragFloat("Amplitude", &field->noise.amplitude, 0.1f, 0.0f, 10.0f);
                ImGui::DragInt("Octaves", &field->noise.octaves, 1, 1, 8);
                ImGui::DragFloat("Lacunarity", &field->noise.lacunarity, 0.1f, 1.0f, 4.0f);
                ImGui::DragFloat("Persistence", &field->noise.persistence, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Speed", &field->noise.speed, 0.01f, 0.0f, 2.0f);
                ImGui::DragInt("Seed", &field->noise.seed, 1, 0, 99999);
            }
        }
        UIWidgets::EndSection();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // TIME
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Time")) {
        ImGui::DragFloat("Start Frame", &field->start_frame, 1.0f, 0.0f, 10000.0f);
        ImGui::DragFloat("End Frame", &field->end_frame, 1.0f, -1.0f, 10000.0f);
        UIWidgets::HelpMarker("Set End Frame to -1 for infinite duration");
        ImGui::DragFloat("Phase", &field->phase, 0.01f, 0.0f, 1.0f);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // AFFECT MASKS
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Affects")) {
        ImGui::TextDisabled("Which systems this field affects:");
        ImGui::Checkbox("Gas/Smoke", &field->affects_gas);
        ImGui::Checkbox("Particles", &field->affects_particles);
        ImGui::Checkbox("Cloth", &field->affects_cloth);
        ImGui::Checkbox("Rigid Body", &field->affects_rigidbody);
    }
}

} // namespace ForceFieldUI
