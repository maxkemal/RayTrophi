// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - CAMERA PANEL
// ═══════════════════════════════════════════════════════════════════════════════
// This file handles the Camera settings panel content.
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "imgui.h"
#include "CameraPresets.h"
#include "scene_data.h"
#include "ProjectManager.h"

// ═══════════════════════════════════════════════════════════════════════════
// CAMERA SETTINGS PANEL CONTENT NOT USING. MOVE THE PRO CAMERA FEATURES 
// ═══════════════════════════════════════════════════════════════════════════

/*void SceneUI::drawCameraContent(UIContext& ctx)
{
    // Camera selector for multi-camera support
    if (ctx.scene.cameras.size() > 1) {
        std::string current_label = "Camera #" + std::to_string(ctx.scene.active_camera_index);
        ImGui::PushItemWidth(-1);
        if (ImGui::BeginCombo("##ActiveCam", current_label.c_str())) {
            for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
                bool is_selected = (i == ctx.scene.active_camera_index);
                std::string label = "Camera #" + std::to_string(i);
                if (ImGui::Selectable(label.c_str(), is_selected)) {
                    ctx.scene.setActiveCamera(i);
                    if (ctx.optix_gpu_ptr && ctx.scene.camera) {
                        ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                    SCENE_LOG_INFO("Switched to Camera #" + std::to_string(i));
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();
        ImGui::TextDisabled("%zu cameras in scene", ctx.scene.cameras.size());
        ImGui::Separator();
    }

    if (!ctx.scene.camera) {
        UIWidgets::StatusIndicator("No Camera Available", UIWidgets::StatusType::Warning);
        return;
    }

    Vec3 pos = ctx.scene.camera->lookfrom;
    Vec3 target = ctx.scene.camera->lookat;
    float fov = (float)ctx.scene.camera->vfov;
    float& aperture = ctx.scene.camera->aperture;
    float& focus_dist = ctx.scene.camera->focus_dist;

    // Helper lambda to insert single-property keyframe for camera
    auto insertCameraPropertyKey = [&](const std::string& prop_name,
        bool key_pos, bool key_target, bool key_fov, bool key_focus, bool key_aperture) {
            int current_frame = ctx.render_settings.animation_current_frame;
            std::string cam_name = ctx.scene.camera->nodeName;
            if (cam_name.empty()) {
                cam_name = "Camera";
                ctx.scene.camera->nodeName = cam_name;
            }

            Keyframe kf(current_frame);
            kf.has_camera = true;
            kf.camera.has_position = key_pos;
            kf.camera.has_target = key_target;
            kf.camera.has_fov = key_fov;
            kf.camera.has_focus = key_focus;
            kf.camera.has_aperture = key_aperture;
            kf.camera.position = ctx.scene.camera->lookfrom;
            kf.camera.target = ctx.scene.camera->lookat;
            kf.camera.fov = (float)ctx.scene.camera->vfov;
            kf.camera.focus_distance = (float)ctx.scene.camera->focus_dist;
            kf.camera.lens_radius = (float)ctx.scene.camera->lens_radius;

            auto& track = ctx.scene.timeline.tracks[cam_name];
            bool found = false;
            for (auto& existing : track.keyframes) {
                if (existing.frame == current_frame) {
                    existing.has_camera = true;
                    if (key_pos) { existing.camera.has_position = true; existing.camera.position = kf.camera.position; }
                    if (key_target) { existing.camera.has_target = true; existing.camera.target = kf.camera.target; }
                    if (key_fov) { existing.camera.has_fov = true; existing.camera.fov = kf.camera.fov; }
                    if (key_focus) { existing.camera.has_focus = true; existing.camera.focus_distance = kf.camera.focus_distance; }
                    if (key_aperture) { existing.camera.has_aperture = true; existing.camera.lens_radius = kf.camera.lens_radius; }
                    found = true;
                    break;
                }
            }
            if (!found) {
                track.keyframes.push_back(kf);
                std::sort(track.keyframes.begin(), track.keyframes.end(),
                    [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
            }
            SCENE_LOG_INFO("Camera " + prop_name + " keyframe @ frame " + std::to_string(current_frame));
        };

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 1: TRANSFORM (Position & Target)
    // ═══════════════════════════════════════════════════════════════════════════
    static bool targetLock = true;
    if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8.0f);

        // Position
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Position");
        ImGui::SameLine(80);
        ImGui::PushItemWidth(-40);
        bool pos_changed = ImGui::DragFloat3("##CamPos", &pos.x, 0.01f);
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::SmallButton("K##KCamPos")) {
            insertCameraPropertyKey("Position", true, false, false, false, false);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Insert Position Keyframe");

        // Target
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Target");
        ImGui::SameLine(80);
        ImGui::PushItemWidth(-40);
        bool target_changed = false;
        if (!targetLock) {
            target_changed = ImGui::DragFloat3("##CamTarget", &target.x, 0.01f);
        }
        else {
            ImGui::BeginDisabled();
            ImGui::DragFloat3("##CamTarget", &target.x, 0.1f);
            ImGui::EndDisabled();
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (!targetLock) {
            if (ImGui::SmallButton("K##KCamTarget")) {
                insertCameraPropertyKey("Target", false, true, false, false, false);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Insert Target Keyframe");
        }
        else {
            ImGui::TextDisabled(" ");
        }

        // Lock checkbox inline
        ImGui::Checkbox("Lock Target", &targetLock);
        ImGui::SameLine();
        UIWidgets::HelpMarker("Lock: camera orbits around target\nUnlock: edit target freely");

        if (pos_changed) {
            if (targetLock)
                ctx.scene.camera->moveToTargetLocked(pos);
            else {
                ctx.scene.camera->lookfrom = pos;
                ctx.scene.camera->origin = pos;
                ctx.scene.camera->update_camera_vectors();
            }
            ProjectManager::getInstance().markModified();
        }
        if (target_changed) {
            ctx.scene.camera->lookat = target;
            ctx.scene.camera->update_camera_vectors();
            ProjectManager::getInstance().markModified();
        }

        ImGui::Unindent(8.0f);
        ImGui::Spacing();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 2: PHYSICAL CAMERA (Body + Lens)
    // ═══════════════════════════════════════════════════════════════════════════
    if (ImGui::CollapsingHeader("Physical Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8.0f);

        // State for physical camera selections
        static int selected_body = 1;  // Default: Generic Full Frame
        static int selected_lens = 4;  // Default: 50mm Normal
        static float current_crop = 1.0f;

        // ─────────────────────────────────────────────────────────────────────
        // CAMERA BODY (Sensor + Crop Factor)
        // ─────────────────────────────────────────────────────────────────────
        ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Camera Body");

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Body");
        ImGui::SameLine(80);
        ImGui::PushItemWidth(-1);
        if (ImGui::BeginCombo("##CameraBody", CameraPresets::CAMERA_BODIES[selected_body].name)) {
            for (size_t i = 0; i < CameraPresets::CAMERA_BODY_COUNT; ++i) {
                bool is_selected = (selected_body == (int)i);

                // Format: "Name - Sensor (CropFactor)"
                char label[128];
                if (i == 0) {
                    snprintf(label, sizeof(label), "%s", CameraPresets::CAMERA_BODIES[i].name);
                }
                else {
                    snprintf(label, sizeof(label), "%s - %s (%.2fx)",
                        CameraPresets::CAMERA_BODIES[i].name,
                        CameraPresets::getSensorTypeName(CameraPresets::CAMERA_BODIES[i].sensor),
                        CameraPresets::CAMERA_BODIES[i].crop_factor);
                }

                if (ImGui::Selectable(label, is_selected)) {
                    selected_body = (int)i;
                    current_crop = CameraPresets::CAMERA_BODIES[i].crop_factor;

                    // Recalculate FOV with new crop factor
                    if (selected_lens > 0) {
                        float base_fov = CameraPresets::LENS_PRESETS[selected_lens].fov_deg;
                        fov = CameraPresets::getEffectiveFOV(base_fov, current_crop);
                        ctx.scene.camera->vfov = fov;
                        ctx.scene.camera->fov = fov;
                        ctx.scene.camera->update_camera_vectors();

                        if (ctx.optix_gpu_ptr) {
                            ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                    }
                }
                if (is_selected) ImGui::SetItemDefaultFocus();

                // Tooltip with description
                if (ImGui::IsItemHovered() && i > 0) {
                    ImGui::SetTooltip("%s\n%s | %d MP",
                        CameraPresets::CAMERA_BODIES[i].description,
                        CameraPresets::CAMERA_BODIES[i].brand,
                        CameraPresets::CAMERA_BODIES[i].resolution_mp);
                }
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        // Show current sensor info OR manual crop for Custom
        if (selected_body == 0) {
            // Custom mode - manual crop factor input
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Crop");
            ImGui::SameLine(80);
            ImGui::PushItemWidth(-1);
            if (ImGui::SliderFloat("##ManualCrop", &current_crop, 0.5f, 2.5f, "%.2fx")) {
                // Recalculate FOV with manual crop
                if (selected_lens > 0) {
                    float base_fov = CameraPresets::LENS_PRESETS[selected_lens].fov_deg;
                    fov = CameraPresets::getEffectiveFOV(base_fov, current_crop);
                    ctx.scene.camera->vfov = fov;
                    ctx.scene.camera->fov = fov;
                    ctx.scene.camera->update_camera_vectors();

                    if (ctx.optix_gpu_ptr) {
                        ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                }
            }
            ImGui::PopItemWidth();
            ImGui::TextDisabled("Manual sensor simulation");
        }
        else {
            // Preset body - show sensor info
            ImGui::TextDisabled("%s | Crop: %.2fx",
                CameraPresets::getSensorTypeName(CameraPresets::CAMERA_BODIES[selected_body].sensor),
                CameraPresets::CAMERA_BODIES[selected_body].crop_factor);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ─────────────────────────────────────────────────────────────────────
        // LENS SELECTION
        // ─────────────────────────────────────────────────────────────────────
        ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.5f, 1.0f), "Lens");

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Lens");
        ImGui::SameLine(80);
        ImGui::PushItemWidth(-1);

        // Build lens label
        const char* current_lens_label = "Custom";
        char lens_label_buf[128];
        if (selected_lens > 0) {
            snprintf(lens_label_buf, sizeof(lens_label_buf), "%s (%.0fmm f/%.1f)",
                CameraPresets::LENS_PRESETS[selected_lens].name,
                CameraPresets::LENS_PRESETS[selected_lens].focal_mm,
                CameraPresets::LENS_PRESETS[selected_lens].max_aperture);
            current_lens_label = lens_label_buf;
        }

        if (ImGui::BeginCombo("##LensSelect", current_lens_label)) {
            // Group lenses by category
            CameraPresets::LensCategory current_cat = CameraPresets::LensCategory::UltraWide;
            bool first_in_group = true;

            for (size_t i = 0; i < CameraPresets::LENS_PRESET_COUNT; ++i) {
                // Category separator
                if (i > 0 && CameraPresets::LENS_PRESETS[i].category != current_cat) {
                    current_cat = CameraPresets::LENS_PRESETS[i].category;
                    ImGui::Separator();
                    ImGui::TextDisabled("── %s ──", CameraPresets::getLensCategoryName(current_cat));
                }

                bool is_selected = (selected_lens == (int)i);

                // Format: "Name - focal f/aperture"
                char label[128];
                if (i == 0) {
                    snprintf(label, sizeof(label), "Custom (Manual FOV)");
                }
                else {
                    snprintf(label, sizeof(label), "%s - %.0fmm f/%.1f",
                        CameraPresets::LENS_PRESETS[i].name,
                        CameraPresets::LENS_PRESETS[i].focal_mm,
                        CameraPresets::LENS_PRESETS[i].max_aperture);
                }

                if (ImGui::Selectable(label, is_selected)) {
                    selected_lens = (int)i;

                    if (selected_lens > 0) {
                        // Apply lens settings
                        float base_fov = CameraPresets::LENS_PRESETS[i].fov_deg;
                        fov = CameraPresets::getEffectiveFOV(base_fov, current_crop);
                        ctx.scene.camera->vfov = fov;
                        ctx.scene.camera->fov = fov;

                        // Set blade count from lens
                        ctx.scene.camera->blade_count = CameraPresets::LENS_PRESETS[i].blade_count;

                        ctx.scene.camera->update_camera_vectors();

                        if (ctx.optix_gpu_ptr) {
                            ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                    }
                }
                if (is_selected) ImGui::SetItemDefaultFocus();

                // Tooltip
                if (ImGui::IsItemHovered() && i > 0) {
                    float effective_focal = CameraPresets::LENS_PRESETS[i].focal_mm * current_crop;
                    ImGui::SetTooltip("%s\nEffective: %.0fmm (with crop)\n%d aperture blades",
                        CameraPresets::LENS_PRESETS[i].description,
                        effective_focal,
                        CameraPresets::LENS_PRESETS[i].blade_count);
                }
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        // Show effective focal length and blade info
        if (selected_lens > 0) {
            float effective_focal = CameraPresets::LENS_PRESETS[selected_lens].focal_mm * current_crop;
            ImGui::TextDisabled("Effective: %.0fmm | %d blades | Max f/%.1f",
                effective_focal,
                CameraPresets::LENS_PRESETS[selected_lens].blade_count,
                CameraPresets::LENS_PRESETS[selected_lens].max_aperture);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ─────────────────────────────────────────────────────────────────────
        // FOV (Manual override or display)
        // ─────────────────────────────────────────────────────────────────────
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("FOV");
        ImGui::SameLine(80);
        ImGui::PushItemWidth(-40);
        bool fov_changed = ImGui::SliderFloat("##CamFOV", &fov, 10.0f, 120.0f, "%.1f°");
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::SmallButton("K##KCamFOV")) {
            insertCameraPropertyKey("FOV", false, false, true, false, false);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Insert FOV Keyframe");

        if (fov_changed) {
            ctx.scene.camera->vfov = fov;
            ctx.scene.camera->fov = fov;
            ctx.scene.camera->update_camera_vectors();
            selected_lens = 0; // Switch to Custom when manually adjusted

            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();
        }

        // ─────────────────────────────────────────────────────────────────────
        // LENS DISTORTION
        // ─────────────────────────────────────────────────────────────────────
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Distortion");
        ImGui::SameLine(80);
        ImGui::PushItemWidth(-40);
        float dist = ctx.scene.camera->distortion;
        if (ImGui::SliderFloat("##CamDist", &dist, -0.5f, 0.5f, "%.3f")) {
             ctx.scene.camera->distortion = dist;
             if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();
        }
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Lens Distortion:\nNegative (-): Barrel (Wide Angle)\nPositive (+): Pincushion (Telephoto)");

        ImGui::Unindent(8.0f);
        ImGui::Spacing();
   
    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 2.5: CAMERA MODE (Auto / Pro / Cinema)
    // ═══════════════════════════════════════════════════════════════════════════
    if (ImGui::CollapsingHeader("Camera Mode", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8.0f);
        
        // Mode Selection
        const char* mode_names[] = { "Auto", "Pro", "Cinema" };
        int current_mode = static_cast<int>(ctx.scene.camera->camera_mode);
        
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.8f, 1.0f), "Camera Mode");
        ImGui::SameLine();
        UIWidgets::HelpMarker(
            "AUTO: Point & shoot - automatic settings\n"
            "PRO: Manual exposure, clean render\n"
            "CINEMA: Full physical simulation with lens imperfections"
        );
        
        ImGui::PushItemWidth(-1);
        if (ImGui::Combo("##CameraMode", &current_mode, mode_names, IM_ARRAYSIZE(mode_names))) {
            ctx.scene.camera->camera_mode = static_cast<CameraMode>(current_mode);
            
            // Apply mode presets
            if (ctx.scene.camera->camera_mode == CameraMode::Auto) {
                ctx.scene.camera->auto_exposure = true;
                ctx.scene.camera->enable_chromatic_aberration = false;
                ctx.scene.camera->enable_vignetting = false;
                ctx.scene.camera->enable_camera_shake = false;
            } else if (ctx.scene.camera->camera_mode == CameraMode::Pro) {
                ctx.scene.camera->auto_exposure = false;
                ctx.scene.camera->enable_chromatic_aberration = false;
                ctx.scene.camera->enable_vignetting = false;
                ctx.scene.camera->enable_camera_shake = false;
            } else { // Cinema
                ctx.scene.camera->auto_exposure = false;
                // Enable cinema features with reasonable defaults
                ctx.scene.camera->enable_vignetting = true;
                ctx.scene.camera->vignetting_amount = 0.3f;
            }
            
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();
        }
        ImGui::PopItemWidth();
        
        // Mode description
        const char* mode_desc[] = {
            "Automatic exposure, minimal controls, clean output",
            "Manual control, professional quality, optional effects",
            "Full physical simulation with lens imperfections"
        };
        ImGui::TextDisabled("%s", mode_desc[current_mode]);
        
        // ─────────────────────────────────────────────────────────────────────
        // CINEMA MODE EFFECTS (Only visible in Cinema mode)
        // ─────────────────────────────────────────────────────────────────────
        if (ctx.scene.camera->camera_mode == CameraMode::Cinema) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Lens Imperfections");
            
            // Chromatic Aberration
            bool ca_changed = false;
            ImGui::Checkbox("Chromatic Aberration##CAEnable", &ctx.scene.camera->enable_chromatic_aberration);
            if (ctx.scene.camera->enable_chromatic_aberration) {
                ImGui::Indent(16.0f);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Amount");
                ImGui::SameLine(80);
                ImGui::PushItemWidth(-1);
                ca_changed |= ImGui::SliderFloat("##CAAmount", &ctx.scene.camera->chromatic_aberration, 0.0f, 0.05f, "%.3f");
                ImGui::PopItemWidth();
                ImGui::TextDisabled("Red/Blue channel separation");
                ImGui::Unindent(16.0f);
            }
            
            // Vignetting
            bool vig_changed = false;
            ImGui::Checkbox("Vignetting##VigEnable", &ctx.scene.camera->enable_vignetting);
            if (ctx.scene.camera->enable_vignetting) {
                ImGui::Indent(16.0f);
                
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Amount");
                ImGui::SameLine(80);
                ImGui::PushItemWidth(-1);
                vig_changed |= ImGui::SliderFloat("##VigAmount", &ctx.scene.camera->vignetting_amount, 0.0f, 1.0f, "%.2f");
                ImGui::PopItemWidth();
                
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Falloff");
                ImGui::SameLine(80);
                ImGui::PushItemWidth(-1);
                vig_changed |= ImGui::SliderFloat("##VigFalloff", &ctx.scene.camera->vignetting_falloff, 1.0f, 4.0f, "%.1f");
                ImGui::PopItemWidth();
                ImGui::TextDisabled("Higher = sharper edge transition");
                
                ImGui::Unindent(16.0f);
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Camera Motion");
            
            // Camera Shake
            bool shake_changed = false;
            ImGui::Checkbox("Camera Shake (Handheld)##ShakeEnable", &ctx.scene.camera->enable_camera_shake);
            if (ctx.scene.camera->enable_camera_shake) {
                ImGui::Indent(16.0f);
                
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Intensity");
                ImGui::SameLine(80);
                ImGui::PushItemWidth(-1);
                shake_changed |= ImGui::SliderFloat("##ShakeInt", &ctx.scene.camera->shake_intensity, 0.0f, 1.0f, "%.2f");
                ImGui::PopItemWidth();
                
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Frequency");
                ImGui::SameLine(80);
                ImGui::PushItemWidth(-1);
                shake_changed |= ImGui::SliderFloat("##ShakeFreq", &ctx.scene.camera->shake_frequency, 2.0f, 15.0f, "%.1f Hz");
                ImGui::PopItemWidth();
                
                // Operator Skill
                const char* skill_names[] = { "Amateur", "Intermediate", "Professional", "Expert" };
                int skill = static_cast<int>(ctx.scene.camera->operator_skill);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Operator");
                ImGui::SameLine(80);
                ImGui::PushItemWidth(-1);
                if (ImGui::Combo("##OpSkill", &skill, skill_names, IM_ARRAYSIZE(skill_names))) {
                    ctx.scene.camera->operator_skill = static_cast<Camera::OperatorSkill>(skill);
                    shake_changed = true;
                }
                ImGui::PopItemWidth();
                
                // IBIS
                ImGui::Checkbox("IBIS (Stabilization)", &ctx.scene.camera->ibis_enabled);
                if (ctx.scene.camera->ibis_enabled) {
                    ImGui::SameLine();
                    ImGui::PushItemWidth(60);
                    shake_changed |= ImGui::DragFloat("##IBISStops", &ctx.scene.camera->ibis_effectiveness, 0.5f, 1.0f, 8.0f, "%.1f stops");
                    ImGui::PopItemWidth();
                }
                
                ImGui::Unindent(16.0f);
            }
            
            // Update GPU if any cinema parameter changed
            if (ca_changed || vig_changed || shake_changed) {
                if (ctx.optix_gpu_ptr) {
                    ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
                ctx.renderer.resetCPUAccumulation();
                ProjectManager::getInstance().markModified();
            }
        }
        
        ImGui::Unindent(8.0f);
        ImGui::Spacing();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 3: DEPTH OF FIELD
    // ═══════════════════════════════════════════════════════════════════════════
    if (ImGui::CollapsingHeader("Depth of Field", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8.0f);

        // DOF Enable
        static bool dof_enabled = (aperture > 0.001f);
        dof_enabled = (aperture > 0.001f);

        if (ImGui::Checkbox("Enable DOF", &dof_enabled)) {
            if (!dof_enabled) {
                aperture = 0.0f;
                ctx.scene.camera->aperture = 0.0f;
                ctx.scene.camera->lens_radius = 0.0f;
            }
            else {
                if (aperture < 0.01f) {
                    aperture = 0.1f;
                    ctx.scene.camera->aperture = 0.1f;
                    ctx.scene.camera->lens_radius = 0.05f;
                }
            }
            ctx.scene.camera->update_camera_vectors();
        }

        if (dof_enabled) {
            ImGui::Spacing();

            // F-Stop Presets
            struct FStopPreset { const char* name; float aperture_value; };
            static const FStopPreset fstop_presets[] = {
                { "Custom", 0.0f },
                { "f/1.4 (Max Blur)", 2.5f },
                { "f/1.8", 2.0f },
                { "f/2.8", 1.2f },
                { "f/4.0", 0.7f },
                { "f/5.6", 0.4f },
                { "f/8.0", 0.2f },
                { "f/11", 0.1f },
                { "f/16 (Sharp)", 0.05f },
            };
            static int selected_fstop = 0;

            selected_fstop = 0;
            for (int i = 1; i < IM_ARRAYSIZE(fstop_presets); i++) {
                if (std::abs(aperture - fstop_presets[i].aperture_value) < 0.05f) {
                    selected_fstop = i;
                    break;
                }
            }

            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("F-Stop");
            ImGui::SameLine(80);
            ImGui::PushItemWidth(-1);
            if (ImGui::Combo("##FStop", &selected_fstop,
                [](void* data, int idx, const char** out_text) {
                    *out_text = ((FStopPreset*)data)[idx].name;
                    return true;
                }, (void*)fstop_presets, IM_ARRAYSIZE(fstop_presets)))
            {
                if (selected_fstop > 0) {
                    aperture = fstop_presets[selected_fstop].aperture_value;
                    ctx.scene.camera->aperture = aperture;
                    ctx.scene.camera->lens_radius = aperture * 0.5f;
                    ctx.scene.camera->update_camera_vectors();

                    // Refresh viewport
                    if (ctx.optix_gpu_ptr) {
                        ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                }
            }
            ImGui::PopItemWidth();

            // Aperture Slider with Keyframe
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Aperture");
            ImGui::SameLine(80);
            ImGui::PushItemWidth(-40);
            bool aperture_changed = ImGui::SliderFloat("##CamAperture", &aperture, 0.01f, 5.0f, "%.2f");
            ImGui::PopItemWidth();
            ImGui::SameLine();
            if (ImGui::SmallButton("K##KCamAperture")) {
                insertCameraPropertyKey("Aperture", false, false, false, false, true);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Insert Aperture Keyframe");

            // Focus Distance with Keyframe
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Focus");
            ImGui::SameLine(80);
            ImGui::PushItemWidth(-40);
            bool focus_changed = ImGui::DragFloat("##CamFocus", &focus_dist, 0.05f, 0.01f, 100.0f, "%.2f");
            ImGui::PopItemWidth();
            ImGui::SameLine();
            if (ImGui::SmallButton("K##KCamFocus")) {
                insertCameraPropertyKey("Focus", false, false, false, true, false);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Insert Focus Keyframe");

            // Focus buttons in row
            bool has_selection = ctx.selection.hasSelection();

            if (!has_selection) ImGui::BeginDisabled();
            if (ImGui::Button("Focus to Selection", ImVec2(-1, 0))) {
                Vec3 selection_pos = ctx.selection.selected.position;
                Vec3 cam_pos = ctx.scene.camera->lookfrom;
                float distance = (selection_pos - cam_pos).length();

                if (distance > 0.01f) {
                    ctx.scene.camera->focus_dist = distance;
                    focus_dist = distance;
                    ctx.scene.camera->update_camera_vectors();
                    SCENE_LOG_INFO("Focus distance set to: " + std::to_string(distance));

                    if (ctx.optix_gpu_ptr) {
                        ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                }
            }
            if (!has_selection) ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Select an object first");
            }

            if (aperture_changed || focus_changed) {
                ctx.scene.camera->lens_radius = aperture * 0.5f;
                ctx.scene.camera->update_camera_vectors();

                // Update GPU and reset accumulation for viewport refresh
                if (ctx.optix_gpu_ptr) {
                    ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
                }
                ctx.renderer.resetCPUAccumulation();
                ProjectManager::getInstance().markModified();
            }

            // Bokeh Shape (collapsible sub-section)
            ImGui::Spacing();
            if (ImGui::TreeNode("Bokeh Shape")) {
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Blades");
                ImGui::SameLine(80);
                ImGui::PushItemWidth(-1);
                ImGui::SliderInt("##BladeCount", &ctx.scene.camera->blade_count, 3, 12);
                ImGui::PopItemWidth();
                ImGui::TextDisabled("More blades = rounder bokeh");
                ImGui::TreePop();
            }

            // Focus Ring toggle
            ImGui::Spacing();
            ImGui::Checkbox("Show Focus Ring", &viewport_settings.show_focus_ring);
            ImGui::SameLine();
            UIWidgets::HelpMarker("Shows split-prism focus aid in viewport center.\nDrag the ring to adjust focus distance.");
        }
        else {
            ImGui::TextDisabled("Enable DOF to access settings");
        }

        ImGui::Unindent(8.0f);
        ImGui::Spacing();
    
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 4: CONTROLS & ACTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8.0f);

        // Mouse Control
        ImGui::Checkbox("Mouse Look", &ctx.mouse_control_enabled);
        ImGui::SameLine();
        UIWidgets::HelpMarker("Use mouse to rotate camera view");

        if (ctx.mouse_control_enabled) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Sensitivity");
            ImGui::SameLine(80);
            ImGui::PushItemWidth(-1);
            ImGui::SliderFloat("##MouseSens", &ctx.render_settings.mouse_sensitivity, 0.01f, 5.0f, "%.2f");
            ImGui::PopItemWidth();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Action Buttons
        float btn_width = (ImGui::GetContentRegionAvail().x - 8) / 2.0f;

        if (ImGui::Button("Key All", ImVec2(btn_width, 0))) {
            int current_frame = ctx.render_settings.animation_current_frame;
            std::string cam_name = ctx.scene.camera->nodeName;
            if (cam_name.empty()) {
                cam_name = "Camera";
                ctx.scene.camera->nodeName = cam_name;
            }

            Keyframe kf(current_frame);
            kf.has_camera = true;
            kf.camera.has_position = true;
            kf.camera.has_target = true;
            kf.camera.has_fov = true;
            kf.camera.has_focus = true;
            kf.camera.has_aperture = true;
            kf.camera.position = ctx.scene.camera->lookfrom;
            kf.camera.target = ctx.scene.camera->lookat;
            kf.camera.fov = (float)ctx.scene.camera->vfov;
            kf.camera.focus_distance = (float)ctx.scene.camera->focus_dist;
            kf.camera.lens_radius = (float)ctx.scene.camera->lens_radius;

            auto& track = ctx.scene.timeline.tracks[cam_name];
            bool found = false;
            for (auto& existing : track.keyframes) {
                if (existing.frame == current_frame) {
                    existing.has_camera = true;
                    existing.camera = kf.camera;
                    found = true;
                    break;
                }
            }
            if (!found) {
                track.keyframes.push_back(kf);
                std::sort(track.keyframes.begin(), track.keyframes.end(),
                    [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
            }
            SCENE_LOG_INFO("Camera keyframe inserted at frame " + std::to_string(current_frame));
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Insert keyframe for ALL camera properties");

        ImGui::SameLine();

        if (ImGui::Button("Reset", ImVec2(btn_width, 0))) {
            ctx.scene.camera->reset();
            ctx.start_render = true;
            SCENE_LOG_INFO("Camera reset to initial state.");
            ProjectManager::getInstance().markModified();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset camera to initial position");

        ImGui::Unindent(8.0f);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SECTION 5: CAMERA HUD (Viewport Overlay Controls)
    // ═══════════════════════════════════════════════════════════════════════════
    if (ImGui::CollapsingHeader("Camera HUD")) {
        ImGui::Indent(8.0f);

        // Master toggle
        ImGui::Checkbox("Enable Camera HUD", &viewport_settings.show_camera_hud);
        ImGui::SameLine();
        UIWidgets::HelpMarker("Show camera controls overlay in viewport.\nIncludes focus and zoom rings.");

        if (viewport_settings.show_camera_hud) {
            ImGui::Spacing();

            // Sub-toggles with indent
            ImGui::Indent(16.0f);

            ImGui::Checkbox("Focus Ring", &viewport_settings.show_focus_ring);
            ImGui::SameLine();
            UIWidgets::HelpMarker("Split-prism focus indicator (requires DOF enabled).\nDrag ring to adjust focus distance.");

            ImGui::Checkbox("Zoom Ring", &viewport_settings.show_zoom_ring);
            ImGui::SameLine();
            UIWidgets::HelpMarker("FOV zoom control bar on left side.\nDrag to adjust field of view.");

            ImGui::Unindent(16.0f);
        }
        else {
            ImGui::TextDisabled("Enable HUD to access controls");
        }

        ImGui::Unindent(8.0f);
    }
}*/

