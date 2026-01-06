// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - SCATTER BRUSH MODULE
// ═══════════════════════════════════════════════════════════════════════════════
// Brush-based instance painting for foliage, rocks, debris etc.
// Part of the SceneUI modular system.
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include "InstanceManager.h"
#include "InstanceGroup.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "SceneSelection.h"
#include "scene_data.h"
#include "Triangle.h"
#include "imgui.h"
#include "globals.h"
#include <cmath>
#include <algorithm>

// Forward declaration
static void syncInstancesToScene(UIContext& ctx, InstanceGroup& group, bool clear_only);

// ═══════════════════════════════════════════════════════════════════════════════
// SCATTER BRUSH PANEL UI
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::drawScatterBrushPanel(UIContext& ctx) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
    
    InstanceManager& im = InstanceManager::getInstance();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 1: SOURCE MESH SETUP
    // ═══════════════════════════════════════════════════════════════════════════
    
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "STEP 1: Source Mesh");
    ImGui::Separator();
    
    // Show currently selected object
    if (ctx.selection.hasSelection()) {
        ImGui::Text("Selected: ");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", ctx.selection.selected.name.c_str());
        
        // Count triangles
        int tri_count = 0;
        for (auto& obj : ctx.scene.world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == ctx.selection.selected.name) {
                tri_count++;
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(%d tris)", tri_count);
        
        // Get or create group
        InstanceGroup* active_group = scatter_brush.active_group_id >= 0 ? 
            im.getGroup(scatter_brush.active_group_id) : nullptr;
        
        if (active_group) {
            // Add to existing group's sources
            if (ImGui::Button("Add to Sources", ImVec2(-1, 24))) {
                std::string node_name = ctx.selection.selected.name;
                std::vector<std::shared_ptr<Triangle>> selected_tris;
                
                for (auto& obj : ctx.scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (tri && tri->getNodeName() == node_name) {
                        selected_tris.push_back(tri);
                    }
                }
                
                if (!selected_tris.empty()) {
                    // Check if source already exists
                    bool exists = false;
                    for (auto& src : active_group->sources) {
                        if (src.name == node_name) {
                            exists = true;
                            break;
                        }
                    }
                    
                    if (!exists) {
                        ScatterSource new_source(node_name, selected_tris);
                        active_group->sources.push_back(new_source);
                        
                        // If first source, also set legacy fields
                        if (active_group->sources.size() == 1) {
                            active_group->source_node_name = node_name;
                            active_group->source_triangles = selected_tris;
                        }
                        
                        SCENE_LOG_INFO("[Scatter] Added source '" + node_name + "' to group");
                    }
                }
            }
        } else {
            // Create new group
            if (ImGui::Button("Create Scatter Group", ImVec2(-1, 24))) {
                std::string node_name = ctx.selection.selected.name;
                std::vector<std::shared_ptr<Triangle>> selected_tris;
                
                for (auto& obj : ctx.scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (tri && tri->getNodeName() == node_name) {
                        selected_tris.push_back(tri);
                    }
                }
                
                if (!selected_tris.empty()) {
                    std::string group_name = node_name + "_scatter";
                    int new_id = im.createGroup(group_name, node_name, selected_tris);
                    scatter_brush.active_group_id = new_id;
                    
                    // Add as first source
                    InstanceGroup* new_group = im.getGroup(new_id);
                    if (new_group) {
                        ScatterSource src(node_name, selected_tris);
                        new_group->sources.push_back(src);
                    }
                    
                    SCENE_LOG_INFO("[Scatter] Created scatter group '" + group_name + "'");
                }
            }
        }
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Select an object in the viewport first");
    }
    
    // Show existing groups
    auto& groups = im.getGroups();
    if (!groups.empty()) {
        ImGui::Spacing();
        ImGui::Text("Scatter Groups:");
        
        const char* current_name = "-- Select --";
        InstanceGroup* active_group = im.getGroup(scatter_brush.active_group_id);
        if (active_group) {
            current_name = active_group->name.c_str();
        }
        
        ImGui::PushItemWidth(-1);
        if (ImGui::BeginCombo("##ScatterGroup", current_name)) {
            for (const auto& g : groups) {
                ImGui::PushID(g.id);
                bool is_selected = (g.id == scatter_brush.active_group_id);
                char label[128];
                snprintf(label, sizeof(label), "%s  [%zu instances]", g.name.c_str(), g.instances.size());
                
                if (ImGui::Selectable(label, is_selected)) {
                    scatter_brush.active_group_id = g.id;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();
    }
    
    ImGui::Spacing();
    ImGui::Spacing();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SCATTER SOURCES LIST (Multi-Source)
    // ═══════════════════════════════════════════════════════════════════════════
    
    InstanceGroup* current_group = im.getGroup(scatter_brush.active_group_id);
    if (current_group && !current_group->sources.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.8f, 1.0f), "Sources (%zu):", current_group->sources.size());
        ImGui::Separator();
        
        int remove_idx = -1;
        for (size_t i = 0; i < current_group->sources.size(); i++) {
            auto& src = current_group->sources[i];
            
            ImGui::PushID(static_cast<int>(i));
            
            // Source name
            ImGui::BulletText("%s", src.name.c_str());
            ImGui::SameLine(200);
            
            // Weight slider
            ImGui::SetNextItemWidth(100);
            ImGui::SliderFloat("##weight", &src.weight, 0.0f, 1.0f, "%.2f");
            
            // Remove button
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) {
                remove_idx = static_cast<int>(i);
            }
            
            ImGui::PopID();
        }
        
        // Remove source if requested
        if (remove_idx >= 0) {
            current_group->sources.erase(current_group->sources.begin() + remove_idx);
            // Update legacy fields if needed
            if (!current_group->sources.empty()) {
                current_group->source_node_name = current_group->sources[0].name;
                current_group->source_triangles = current_group->sources[0].triangles;
            } else {
                current_group->source_node_name.clear();
                current_group->source_triangles.clear();
            }
            
            // Clear instances and rebuild scene (source changed)
            current_group->clearInstances();
            syncInstancesToScene(ctx, *current_group, true);
            
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
            }
            
            SCENE_LOG_INFO("[Scatter] Source removed, instances cleared");
        }
        
        ImGui::Spacing();
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 2: PAINTING
    // ═══════════════════════════════════════════════════════════════════════════
    
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "STEP 2: Paint on Surface");
    ImGui::Separator();
    
    InstanceGroup* active_group = im.getGroup(scatter_brush.active_group_id);
    
    if (!active_group) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Create or select a scatter group first");
    } else {
        // Enable brush toggle
        ImGui::Checkbox("Enable Brush", &scatter_brush.enabled);
        
        if (scatter_brush.enabled) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "ACTIVE - Click on surfaces to paint!");
        }
        
        ImGui::Spacing();
        
        // Brush Mode
        ImGui::Text("Mode:");
        ImGui::SameLine();
        ImGui::RadioButton("Add", &scatter_brush.brush_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Remove", &scatter_brush.brush_mode, 1);
        
        // Brush Settings
        ImGui::SliderFloat("Brush Size", &scatter_brush.brush_radius, 0.5f, 20.0f, "%.1f m");
        ImGui::SliderFloat("Density", &active_group->brush_settings.density, 0.01f, 20.0f, "%.2f /sq.m");
        
        ImGui::Checkbox("Preview Circle", &scatter_brush.show_brush_preview);
        
        ImGui::Spacing();
        ImGui::Separator();
        
        // TARGET SURFACE SELECTION
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Target Surface:");
        if (scatter_brush.target_surface_name.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "(Any surface)");
        } else {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", scatter_brush.target_surface_name.c_str());
        }
        
        if (ctx.selection.hasSelection() && ctx.selection.selected.name != active_group->source_node_name) {
            if (ImGui::Button("Set Target from Selection")) {
                scatter_brush.target_surface_name = ctx.selection.selected.name;
                SCENE_LOG_INFO("[Scatter] Target surface set to: " + scatter_brush.target_surface_name);
            }
        }
        ImGui::SameLine();
        if (!scatter_brush.target_surface_name.empty()) {
            if (ImGui::Button("Clear Target")) {
                scatter_brush.target_surface_name.clear();
            }
        }
        
        ImGui::Spacing();
        
        // Randomization
        if (ImGui::CollapsingHeader("Randomization")) {
            auto& bs = active_group->brush_settings;
            
            ImGui::Text("Scale:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            ImGui::DragFloat("##smin", &bs.scale_min, 0.01f, 0.1f, bs.scale_max, "%.2f");
            ImGui::SameLine();
            ImGui::Text("-");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            ImGui::DragFloat("##smax", &bs.scale_max, 0.01f, bs.scale_min, 3.0f, "%.2f");
            
            ImGui::SliderFloat("Y Rotation", &bs.rotation_random_y, 0.0f, 360.0f, "%.0f°");
            ImGui::SliderFloat("Tilt", &bs.rotation_random_xz, 0.0f, 30.0f, "%.0f°");
            ImGui::Checkbox("Align to Surface Normal", &bs.align_to_normal);
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        
        // Statistics
        ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1.0f), 
            "Instances: %zu | Triangles: %zu", 
            active_group->instances.size(), 
            active_group->getTriangleCount());
        
        // Actions
        if (ImGui::Button("Clear All Instances")) {
            active_group->clearInstances();
            // Remove from scene
            syncInstancesToScene(ctx, *active_group, true);
            
            // Rebuild BVH and OptiX
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
            }
        }
        
        ImGui::SameLine();
        
        if (ImGui::Button("Delete Group")) {
            // Remove instances from scene first
            syncInstancesToScene(ctx, *active_group, true);
            im.deleteGroup(scatter_brush.active_group_id);
            scatter_brush.active_group_id = -1;
            
            // Rebuild BVH and OptiX
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
            }
        }
    }
    
    ImGui::PopStyleVar();
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYNC INSTANCES TO SCENE (Make them renderable)
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::syncInstancesToScene(UIContext& ctx, InstanceGroup& group, bool clear_only) {
    // Remove existing instances from scene (they have special name prefix)
    std::string instance_prefix = "_inst_" + group.name + "_";
    
    auto& objects = ctx.scene.world.objects;
    objects.erase(
        std::remove_if(objects.begin(), objects.end(),
            [&instance_prefix](const std::shared_ptr<Hittable>& obj) {
                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                if (tri) {
                    return tri->getNodeName().find(instance_prefix) == 0;
                }
                return false;
            }),
        objects.end()
    );
    
    if (clear_only || group.instances.empty()) {
        return;
    }
    
    // Compute source mesh center (average of all vertices)
    Vec3 mesh_center(0, 0, 0);
    int vertex_count = 0;
    for (const auto& src_tri : group.source_triangles) {
        mesh_center = mesh_center + src_tri->getV0();
        mesh_center = mesh_center + src_tri->getV1();
        mesh_center = mesh_center + src_tri->getV2();
        vertex_count += 3;
    }
    if (vertex_count > 0) {
        mesh_center = mesh_center / (float)vertex_count;
    }
    
    // Add new instances
    for (size_t i = 0; i < group.instances.size(); i++) {
        const auto& inst = group.instances[i];
        Matrix4x4 transform = inst.toMatrix();
        
        // Clone each source triangle with new transform
        for (const auto& src_tri : group.source_triangles) {
            // Move vertex to local space (centered), apply transform, then place at instance position
            Vec3 local_v0 = src_tri->getV0() - mesh_center;
            Vec3 local_v1 = src_tri->getV1() - mesh_center;
            Vec3 local_v2 = src_tri->getV2() - mesh_center;
            
            // Apply instance transform (scale, rotation, translation)
            Vec3 v0 = transform.transform_point(local_v0);
            Vec3 v1 = transform.transform_point(local_v1);
            Vec3 v2 = transform.transform_point(local_v2);
            
            // Compute face normal from transformed vertices
            Vec3 edge1 = v1 - v0;
            Vec3 edge2 = v2 - v0;
            Vec3 face_normal = edge1.cross(edge2).normalize();
            
            auto new_tri = std::make_shared<Triangle>(
                v0, v1, v2, 
                face_normal, face_normal, face_normal,  // Use face normal
                src_tri->t0, src_tri->t1, src_tri->t2,  // UV coordinates
                src_tri->getMaterial()
            );
            
            // Set instance name for identification
            char name[64];
            snprintf(name, sizeof(name), "%s%zu", instance_prefix.c_str(), i);
            new_tri->setNodeName(name);
            
            objects.push_back(new_tri);
        }
    }
    
    SCENE_LOG_INFO("[Scatter] Synced " + std::to_string(group.instances.size()) + 
                   " instances (" + std::to_string(group.getTriangleCount()) + " triangles) to scene");
}

// ═══════════════════════════════════════════════════════════════════════════════
// VIEWPORT BRUSH INTERACTION
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::handleScatterBrush(UIContext& ctx) {
    if (!scatter_brush.enabled) return;
    if (scatter_brush.active_group_id < 0) return;
    
    InstanceManager& im = InstanceManager::getInstance();
    InstanceGroup* group = im.getGroup(scatter_brush.active_group_id);
    if (!group) return;
    
    ImGuiIO& io = ImGui::GetIO();
    
    // Don't process if over UI
    if (io.WantCaptureMouse) return;
    
    // Get mouse position
    int mx, my;
    SDL_GetMouseState(&mx, &my);
    
    float u = (float)mx / io.DisplaySize.x;
    float v = 1.0f - ((float)my / io.DisplaySize.y);
    
    // Raycast to find surface hit
    if (!ctx.scene.camera || !ctx.scene.bvh) return;
    
    Ray ray = ctx.scene.camera->get_ray(u, v);
    HitRecord hit;
    
    bool did_hit = ctx.scene.bvh->hit(ray, 0.001f, 1e10f, hit);
    
    // Filter out scattered instances from hits (they start with "_inst_")
    if (did_hit && hit.triangle) {
        std::string hit_node = hit.triangle->getNodeName();
        
        // Skip if hit an instance triangle
        if (hit_node.find("_inst_") == 0) {
            did_hit = false;  // Ignore this hit, don't paint on instances
        }
        
        // Check target surface filter
        if (did_hit && !scatter_brush.target_surface_name.empty()) {
            if (hit_node != scatter_brush.target_surface_name) {
                did_hit = false;  // Not on target surface
            }
        }
    }
    
    if (did_hit && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        static float last_paint_time = 0.0f;
        float current_time = ImGui::GetTime();
        
        // Throttle painting to avoid too many instances per frame (300ms interval)
        if (current_time - last_paint_time < 0.3f) return;
        last_paint_time = current_time;
        
        if (scatter_brush.brush_mode == 0) {
            // ADD mode
            int added = im.paintInstances(
                scatter_brush.active_group_id,
                hit.point,
                hit.normal,
                scatter_brush.brush_radius,
                1.0f
            );
            
            if (added > 0) {
                // Sync to scene for rendering
                syncInstancesToScene(ctx, *group, false);
                
                // Rebuild BVH immediately
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.resetCPUAccumulation();
                
                // Full OptiX rebuild (includes new geometry and SBT records)
                if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                }
            }
        }
        else if (scatter_brush.brush_mode == 1) {
            // REMOVE mode
            size_t before = group->instances.size();
            group->removeInstancesInRadius(hit.point, scatter_brush.brush_radius);
            
            if (group->instances.size() < before) {
                // Sync to scene
                syncInstancesToScene(ctx, *group, false);
                
                // Rebuild BVH immediately  
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.resetCPUAccumulation();
                
                // Full OptiX rebuild
                if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                }
            }
        }
        
        // Capture mouse to prevent selection
        hud_captured_mouse = true;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BRUSH PREVIEW (Viewport Circle)
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::drawBrushPreview(UIContext& ctx) {
    // Check if any brush is active
    bool is_scatter = scatter_brush.enabled && scatter_brush.show_brush_preview;
    bool is_foliage = foliage_brush.enabled && foliage_brush.show_preview;
    
    if (!is_scatter && !is_foliage) return;
    
    if (!ctx.scene.camera || !ctx.scene.bvh) return;
    
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    
    // Get mouse position
    int mx, my;
    SDL_GetMouseState(&mx, &my);
    
    // Use active brush parameters
    float radius = is_scatter ? scatter_brush.brush_radius : foliage_brush.radius;
    int mode = is_scatter ? scatter_brush.brush_mode : foliage_brush.mode;
    
    float u = (float)mx / io.DisplaySize.x;
    float v = 1.0f - ((float)my / io.DisplaySize.y);
    
    // Raycast
    Ray ray = ctx.scene.camera->get_ray(u, v);
    HitRecord hit;
    
    if (!ctx.scene.bvh->hit(ray, 0.001f, 1e10f, hit)) return;
    
    // Draw circle at hit point
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    Camera& cam = *ctx.scene.camera;
    
    // Project to screen
    auto projectToScreen = [&](const Vec3& world_pos) -> ImVec2 {
        Vec3 dir = world_pos - cam.lookfrom;
        float dist = dir.length();
        dir = dir.normalize();
        
        Vec3 forward = (cam.lookat - cam.lookfrom).normalize();
        Vec3 right = cam.u;
        Vec3 up = cam.v;
        
        float z = Vec3::dot(dir, forward) * dist;
        if (z <= 0.01f) return ImVec2(-1, -1);
        
        float x = Vec3::dot(world_pos - cam.lookfrom, right);
        float y = Vec3::dot(world_pos - cam.lookfrom, up);
        
        float fov_scale = tanf(cam.vfov * 0.5f * 3.14159f / 180.0f);
        float aspect = io.DisplaySize.x / io.DisplaySize.y;
        
        float sx = (x / (z * fov_scale * aspect)) * 0.5f + 0.5f;
        float sy = (y / (z * fov_scale)) * 0.5f + 0.5f;
        
        return ImVec2(sx * io.DisplaySize.x, (1.0f - sy) * io.DisplaySize.y);
    };
    
    ImVec2 center = projectToScreen(hit.point);
    if (center.x < 0) return;
    
    // Project edge point
    Vec3 tangent = hit.normal.cross(Vec3(0, 1, 0));
    if (tangent.length() < 0.01f) tangent = hit.normal.cross(Vec3(1, 0, 0));
    tangent = tangent.normalize();
    
    Vec3 edge_point = hit.point + tangent * radius;
    ImVec2 edge = projectToScreen(edge_point);
    
    float screen_radius = sqrtf((edge.x - center.x) * (edge.x - center.x) + 
                                 (edge.y - center.y) * (edge.y - center.y));
    screen_radius = std::max(10.0f, std::min(screen_radius, 500.0f));
    
    // Color based on mode
    ImU32 color;
    if (is_foliage) {
         // Cyan for foliage add, orange for remove
         color = (mode == 0) 
            ? IM_COL32(0, 255, 255, 180) 
            : IM_COL32(255, 128, 0, 180);
    } else {
        // Green for scatter add, red for remove
        color = (mode == 0) 
            ? IM_COL32(100, 255, 100, 180)
            : IM_COL32(255, 100, 100, 180);
    }
    
    // Draw visual
    draw_list->AddCircle(center, screen_radius, color, 32, 3.0f);
    draw_list->AddCircleFilled(center, 4.0f, color);
    
    // Mode text
    const char* mode_text = (mode == 0) ? "ADD" : "REMOVE";
    ImVec2 text_size = ImGui::CalcTextSize(mode_text);
    draw_list->AddText(ImVec2(center.x - text_size.x * 0.5f, center.y + screen_radius + 5), color, mode_text);
    
    // Extra info for foliage (density)
    if (is_foliage) {
        std::string info = "Density: " + std::to_string(foliage_brush.density);
        ImVec2 info_size = ImGui::CalcTextSize(info.c_str());
        draw_list->AddText(ImVec2(center.x - info_size.x * 0.5f, center.y + screen_radius + 20), color, info.c_str());
    }
}
