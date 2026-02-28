// ===============================================================================
// SCENE UI - LIGHTS PANEL
// ===============================================================================
// This file handles the Lights properties panel.
// ===============================================================================

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "imgui.h"
#include "scene_data.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "ProjectManager.h"
#include <Backend/VulkanBackend.h>


// =============================================================================
void SceneUI::drawLightsContent(UIContext& ctx)
{
    ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1), "Scene Lights");
    bool changed = false;

    for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
        auto light = ctx.scene.lights[i];
        if (!light) continue;

        std::string label = "Light #" + std::to_string(i);

        if (ImGui::TreeNode(label.c_str())) {
            const char* names[] = { "Point", "Directional", "Spot", "Area" };
            int current_type = (int)light->type();
            int new_type = current_type;

            if (ImGui::Combo("Type", &new_type, names, IM_ARRAYSIZE(names))) {
                if (new_type != current_type) {
                    // Prepare common data
                    Vec3 pos = light->position;
                    Vec3 col = light->color;
                    float inten = light->intensity;
                    std::string name = light->nodeName;
                    Vec3 dir = light->direction;
                    // Default direction if switching from Point (which might have 0,0,0 dir)
                    if (dir.length_squared() < 0.001f) dir = Vec3(0, -1, 0);

                    std::shared_ptr<Light> new_light = nullptr;

                    switch (new_type) {
                    case 0: // Point
                        new_light = std::make_shared<PointLight>(pos, col, inten);
                        new_light->radius = light->radius;
                        break;
                    case 1: // Directional
                        new_light = std::make_shared<DirectionalLight>(dir, col, inten);
                        new_light->radius = light->radius; // Preserves soft shadow size if applicable
                        break;
                    case 2: // Spot
                        // Use reasonable defaults for Angle/Falloff
                        new_light = std::make_shared<SpotLight>(pos, dir, col, 45.0f, 10.0f);
                        new_light->intensity = inten;
                        break;
                    case 3: // Area
                        // Default 2x2 area facing +Z initially, will be overridden by logic potentially
                        new_light = std::make_shared<AreaLight>(pos, Vec3(2, 0, 0), Vec3(0, 0, 2), 2.0f, 2.0f, col);
                        // AreaLight constructor sig: pos, u, v, w, h, color
                        // Logic inside AreaLight usually sets internal u/v based on w/h.
                        new_light->intensity = inten;
                        break;
                    }

                    if (new_light) {
                        new_light->nodeName = name;

                        // Update Scene List
                        ctx.scene.lights[i] = new_light;

                        // Update Selection if necessary
                        if (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light == light) {
                            ctx.selection.selected.light = new_light;
                            // No need to reset index, it's the same i
                        }

                        light = new_light; // Update local pointer for rest of this frame's UI
                        changed = true;
                    }
                }
            }

            // Helper lambda to insert single-property keyframe for light
            auto insertLightPropertyKey = [&](const std::string& prop_name,
                bool key_pos, bool key_color, bool key_intensity, bool key_dir) {
                    int current_frame = ctx.render_settings.animation_current_frame;
                    std::string light_name = light->nodeName;
                    if (light_name.empty()) {
                        light_name = "Light_" + std::to_string(i);
                        light->nodeName = light_name;
                    }

                    Keyframe kf(current_frame);
                    kf.has_light = true;
                    kf.light.has_position = key_pos;
                    kf.light.has_color = key_color;
                    kf.light.has_intensity = key_intensity;
                    kf.light.has_direction = key_dir;
                    kf.light.position = light->position;
                    kf.light.color = light->color;
                    kf.light.intensity = light->intensity;
                    kf.light.direction = light->direction;

                    auto& track = ctx.scene.timeline.tracks[light_name];
                    bool found = false;
                    for (auto& existing : track.keyframes) {
                        if (existing.frame == current_frame) {
                            existing.has_light = true;
                            // Merge flags - only set new ones, don't overwrite
                            if (key_pos) { existing.light.has_position = true; existing.light.position = kf.light.position; }
                            if (key_color) { existing.light.has_color = true; existing.light.color = kf.light.color; }
                            if (key_intensity) { existing.light.has_intensity = true; existing.light.intensity = kf.light.intensity; }
                            if (key_dir) { existing.light.has_direction = true; existing.light.direction = kf.light.direction; }
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        track.keyframes.push_back(kf);
                        std::sort(track.keyframes.begin(), track.keyframes.end(),
                            [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
                    }
                    SCENE_LOG_INFO("Light " + prop_name + " keyframe @ frame " + std::to_string(current_frame));
                };

            // Position with ? key button
            if (ImGui::DragFloat3("Position", &light->position.x, 0.1f)) changed = true;
            ImGui::SameLine();
            if (ImGui::SmallButton(("##KPos" + std::to_string(i)).c_str())) {
                insertLightPropertyKey("Position", true, false, false, false);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Key Position");

            // Direction with ? key button (only for Directional/Spot)
            if (light->type() == LightType::Directional || light->type() == LightType::Spot) {
                if (ImGui::DragFloat3("Direction", &light->direction.x, 0.1f)) changed = true;
                ImGui::SameLine();
                if (ImGui::SmallButton(("##KDir" + std::to_string(i)).c_str())) {
                    insertLightPropertyKey("Direction", false, false, false, true);
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Key Direction");
            }

            // Color with ? key button
            if (ImGui::ColorEdit3("Color", &light->color.x)) changed = true;
            ImGui::SameLine();
            if (ImGui::SmallButton(("##KCol" + std::to_string(i)).c_str())) {
                insertLightPropertyKey("Color", false, true, false, false);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Key Color");

            // Intensity with ? key button
            if (ImGui::DragFloat("Intensity", &light->intensity, 0.1f, 0, 1000.0f)) changed = true;
            ImGui::SameLine();
            if (ImGui::SmallButton(("##KInt" + std::to_string(i)).c_str())) {
                insertLightPropertyKey("Intensity", false, false, true, false);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Key Intensity");

            if (light->type() == LightType::Point ||
                light->type() == LightType::Area ||
                light->type() == LightType::Directional)
                if (ImGui::DragFloat("Radius", &light->radius, 0.01f, 0.01f, 100.0f)) changed = true;

            if (auto sl = std::dynamic_pointer_cast<SpotLight>(light)) {
                float angle = sl->getAngleDegrees();
                if (ImGui::DragFloat("Cone Angle", &angle, 0.5f, 1.0f, 89.0f)) {
                    sl->setAngleDegrees(angle);
                    changed = true;
                }
                float falloff = sl->getFalloff();
                if (ImGui::SliderFloat("Falloff", &falloff, 0.0f, 1.0f)) {
                    sl->setFalloff(falloff);
                    changed = true;
                }
            }
            else if (auto al = std::dynamic_pointer_cast<AreaLight>(light)) {
                if (ImGui::DragFloat("Width", &al->width, 0.05f, 0.01f, 100.0f)) {
                    al->u = al->u.normalize() * al->width;
                    changed = true;
                }
                if (ImGui::DragFloat("Height", &al->height, 0.05f, 0.01f, 100.0f)) {
                    al->v = al->v.normalize() * al->height;
                    changed = true;
                }
            }

            // �����������������������������������������������������������������
            // LIGHT KEYFRAME BUTTON
            // �����������������������������������������������������������������
            ImGui::Separator();
            if (ImGui::SmallButton("Key Light")) {
                // Get current frame
                int current_frame = ctx.render_settings.animation_current_frame;

                // Get light name (use nodeName or generate one)
                std::string light_name = light->nodeName;
                if (light_name.empty()) {
                    light_name = "Light_" + std::to_string(i);
                    light->nodeName = light_name;  // Assign for future use
                }

                // Create keyframe with light data
                Keyframe kf(current_frame);
                kf.has_light = true;

                // Set per-property flags (Key Light keys ALL properties)
                kf.light.has_position = true;
                kf.light.has_color = true;
                kf.light.has_intensity = true;
                kf.light.has_direction = true;

                // Set property values
                kf.light.position = light->position;
                kf.light.color = light->color;
                kf.light.intensity = light->intensity;
                kf.light.direction = light->direction;

                // Add to timeline
                auto& track = ctx.scene.timeline.tracks[light_name];

                // Check if keyframe exists at this frame
                bool found = false;
                for (auto& existing : track.keyframes) {
                    if (existing.frame == current_frame) {
                        // Update existing
                        existing.has_light = true;
                        existing.light = kf.light;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    track.keyframes.push_back(kf);
                    // Sort keyframes by frame
                    std::sort(track.keyframes.begin(), track.keyframes.end(),
                        [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
                }

                SCENE_LOG_INFO("Light keyframe inserted at frame " + std::to_string(current_frame) + " for " + light_name);
            }
            ImGui::SameLine();
            UIWidgets::HelpMarker("Insert keyframe for light properties (position, color, intensity, direction)");

            ImGui::TreePop();
        }
    }

    if (changed) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.backend_ptr->setLights(ctx.scene.lights);
            ctx.backend_ptr->resetAccumulation();
        }
        // If a directional light was changed in the UI, sync the World sun parameters
        // so miss/sky shaders reflect color/intensity updates immediately.
        for (const auto& l : ctx.scene.lights) {
            if (!l) continue;
            if (l->type() == LightType::Directional) {
                // World expects Nishita sun direction = -direction (scene convention)
                ctx.renderer.world.setSunDirection(-l->direction);
                ctx.renderer.world.setSunIntensity(l->intensity);
                if (ctx.backend_ptr) {
                    auto worldGPU = ctx.renderer.world.getGPUData();
                    ctx.backend_ptr->setWorldData(&worldGPU);
                    // Upload LUTs for Vulkan backend if available
                    auto* vulkanBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr);
                    if (vulkanBackend) {
                        auto* al = ctx.renderer.world.getLUT();
                        if (al && al->is_initialized()) vulkanBackend->uploadAtmosphereLUT(al);
                    }
                    ctx.backend_ptr->resetAccumulation();
                }
                break; // Sync to first directional light only
            }
        }
        ProjectManager::getInstance().markModified();
    }
}
bool SceneUI::deleteSelectedLight(UIContext& ctx)
{
    auto light = ctx.selection.selected.light;
    if (!light) return false;

    auto& lights = ctx.scene.lights;
    auto it = std::find(lights.begin(), lights.end(), light);
    if (it == lights.end()) return false;

    history.record(std::make_unique<DeleteLightCommand>(light));
    lights.erase(it);

    // GPU + CPU reset
            if (ctx.backend_ptr) {
                ctx.backend_ptr->setLights(ctx.scene.lights);
                // Ensure LUT upload after world sync
                auto worldGPU = ctx.renderer.world.getGPUData();
                ctx.backend_ptr->setWorldData(&worldGPU);
                auto* vulkanBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr);
                if (vulkanBackend) {
                    auto* al = ctx.renderer.world.getLUT();
                    if (al && al->is_initialized()) vulkanBackend->uploadAtmosphereLUT(al);
                }
                ctx.backend_ptr->resetAccumulation();
            }
    ctx.renderer.resetCPUAccumulation();

    // If deleted light was directional, sync World sun (clear or update to next directional)
    bool foundDir = false;
    for (const auto& l : ctx.scene.lights) {
        if (l && l->type() == LightType::Directional) { foundDir = true;
            ctx.renderer.world.setSunDirection(-l->direction);
            ctx.renderer.world.setSunIntensity(l->intensity);
            if (ctx.backend_ptr) {
                auto worldGPU = ctx.renderer.world.getGPUData();
                ctx.backend_ptr->setWorldData(&worldGPU);
                auto* vulkanBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr);
                if (vulkanBackend) {
                    auto* al = ctx.renderer.world.getLUT();
                    if (al && al->is_initialized()) vulkanBackend->uploadAtmosphereLUT(al);
                }
                ctx.backend_ptr->resetAccumulation();
            }
            break;
        }
    }
    if (!foundDir) {
        // No directional light remains — clear sun intensity so sky disk disappears
        ctx.renderer.world.setSunIntensity(0.0f);
        if (ctx.backend_ptr) {
            auto worldGPU = ctx.renderer.world.getGPUData();
            ctx.backend_ptr->setWorldData(&worldGPU);
            ctx.backend_ptr->resetAccumulation();
        }
    }

    SCENE_LOG_INFO("Deleted Light");
    ProjectManager::getInstance().markModified();
    return true;
}
