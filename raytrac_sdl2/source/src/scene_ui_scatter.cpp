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
#include <future>
#include <thread>
#include <atomic>
#include "HittableInstance.h"
#include "EmbreeBVH.h"

// Forward declaration
// Deprecated separate scatter module - functionality moved to Terrain Tab

// ═══════════════════════════════════════════════════════════════════════════════
// SCATTER BRUSH PANEL UI
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::drawScatterBrushPanel(UIContext& ctx) {
    ImGui::Text("Foliage tools moved to Terrain Tab"); return;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));

    InstanceManager& im = InstanceManager::getInstance();

    // ═══════════════════════════════════════════════════════════════════════════
    // FOLIAGE LAYER MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════════

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "FOLIAGE LAYERS");
    ImGui::Separator();
    
    // Group Selector (Layer Selector)
    auto& groups = im.getGroups();
    
    // Layer Actions
    if (ImGui::Button("New Layer")) {
        std::string name = "Foliage_Layer_" + std::to_string(groups.size() + 1);
        int new_id = im.createGroup(name, "", {}); // Empty group
        scatter_brush.active_group_id = new_id;
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Delete Layer") && scatter_brush.active_group_id >= 0) {
        im.deleteGroup(scatter_brush.active_group_id);
        scatter_brush.active_group_id = -1;
    }

    // Layer Combo
    const char* current_name = (scatter_brush.active_group_id < 0) ? "-- No Selection --" : "Unknown";
    InstanceGroup* active_group = im.getGroup(scatter_brush.active_group_id);
    if (active_group) current_name = active_group->name.c_str();

    if (ImGui::BeginCombo("Active Layer", current_name)) {
        for (const auto& g : groups) {
            bool is_selected = (g.id == scatter_brush.active_group_id);
            if (ImGui::Selectable(g.name.c_str(), is_selected)) {
                scatter_brush.active_group_id = g.id;
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    
    if (active_group) {
        // Edit Layer Name
        char name_buf[128];
        strncpy(name_buf, active_group->name.c_str(), sizeof(name_buf));
        if (ImGui::InputText("Layer Name", name_buf, sizeof(name_buf))) {
            active_group->name = name_buf;
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "PLACEMENT RULES");
        ImGui::Checkbox("Use Global Settings Override", &active_group->brush_settings.use_global_settings);
        
        // Placement Rules (Splat Map, Slope, etc.)
        const char* channels[] = { "Red (0)", "Green (1)", "Blue (2)", "Alpha (3)" };
        int current_ch = active_group->brush_settings.splat_map_channel;
        const char* combo_preview = (current_ch >= 0 && current_ch < 4) ? channels[current_ch] : "No Mask (Paint Anywhere)";
        
        if (ImGui::BeginCombo("Splat Map Mask", combo_preview)) {
            if (ImGui::Selectable("No Mask (Paint Anywhere)", current_ch == -1)) active_group->brush_settings.splat_map_channel = -1;
            for (int i = 0; i < 4; i++) {
                if (ImGui::Selectable(channels[i], current_ch == i)) active_group->brush_settings.splat_map_channel = i;
            }
            ImGui::EndCombo();
        }
        
        ImGui::SliderFloat("Max Slope", &active_group->brush_settings.slope_max, 0.0f, 90.0f, "%.1f deg");
        
        ImGui::Spacing();
        ImGui::Separator();
        
        // ═══════════════════════════════════════════════════════════════════════════
        // SOURCE MESHES LIST
        // ═══════════════════════════════════════════════════════════════════════════
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "SOURCE MESHES (%zu)", active_group->sources.size());
        
        // "Add Selected" Button
        bool has_selection = ctx.selection.hasSelection();
        if (ImGui::Button("Add Selected Object", ImVec2(-1, 0))) {
            if (has_selection) {
                std::string node_name = ctx.selection.selected.name;
                // Find triangles
                std::vector<std::shared_ptr<Triangle>> selected_tris;
                for (auto& obj : ctx.scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (tri && tri->getNodeName() == node_name) selected_tris.push_back(tri);
                }
                
                if (!selected_tris.empty()) {
                    active_group->sources.emplace_back(node_name, selected_tris);
                    SCENE_LOG_INFO("[Scatter] Added " + node_name + " to layer " + active_group->name);
                }
            }
        }
        if (!has_selection) ImGui::TextDisabled("(Select an object to add)");

        ImGui::Spacing();

        if (ImGui::BeginTable("SourcesTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
            ImGui::TableSetupColumn("Mesh", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Weight", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            ImGui::TableSetupColumn("Scale", ImGuiTableColumnFlags_WidthFixed, 100.0f);
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 30.0f);
            ImGui::TableHeadersRow();

            int remove_idx = -1;
            for (size_t i = 0; i < active_group->sources.size(); i++) {
                auto& src = active_group->sources[i];
                ImGui::PushID((int)i);
                
                ImGui::TableNextRow();
                
                // Column 1: Name & Expander for Details
                ImGui::TableSetColumnIndex(0);
                bool open = ImGui::TreeNodeEx(src.name.c_str(), ImGuiTreeNodeFlags_SpanFullWidth);
                
                // Column 2: Weight
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                ImGui::DragFloat("##w", &src.weight, 0.05f, 0.0f, 10.0f, "%.1f");
                
                // Column 3: Scale Preview
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%.1f - %.1f", src.settings.scale_min, src.settings.scale_max);
                
                // Column 4: Remove
                ImGui::TableSetColumnIndex(3);
                if (ImGui::Button("X")) remove_idx = (int)i;

                // Details Row
                if (open) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(ImGuiCol_FrameBg));
                    // Checkbox to enable per-source editing (or just show them if global is off)
                    if (active_group->brush_settings.use_global_settings) {
                        ImGui::TextDisabled("Global settings override active.");
                    } else {
                        ImGui::Indent();
                        ImGui::DragFloatRange2("Scale Range", &src.settings.scale_min, &src.settings.scale_max, 0.01f, 0.1f, 5.0f);
                        ImGui::DragFloat("Rand Rot Y", &src.settings.rotation_random_y, 1.0f, 0.0f, 360.0f);
                        ImGui::DragFloat("Rand Rot XZ", &src.settings.rotation_random_xz, 1.0f, 0.0f, 45.0f);
                        ImGui::DragFloatRange2("Y Offset", &src.settings.y_offset_min, &src.settings.y_offset_max, 0.01f, -5.0f, 5.0f);
                        ImGui::Checkbox("Align Normal", &src.settings.align_to_normal);
                        if(src.settings.align_to_normal) ImGui::SliderFloat("Influence", &src.settings.normal_influence, 0.0f, 1.0f);
                        ImGui::Unindent();
                    }
                    ImGui::TreePop(); 
                }
                
                ImGui::PopID();
            }
            ImGui::EndTable();
            
             // Handle Removal
            if (remove_idx >= 0) {
                active_group->sources.erase(active_group->sources.begin() + remove_idx);
                // Clear and rebuild
                active_group->clearInstances();
                syncInstancesToScene(ctx, *active_group, true);
                
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                     ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 2: PAINTING
    // ═══════════════════════════════════════════════════════════════════════════

    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "STEP 2: Paint on Surface");
    ImGui::Separator();


    if (!active_group) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Create or select a scatter group first");
    }
    else {
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
        }
        else {
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

            ImGui::SliderFloat("Y Rotation", &bs.rotation_random_y, 0.0f, 360.0f, "%.0fdeg");
            ImGui::SliderFloat("Tilt", &bs.rotation_random_xz, 0.0f, 30.0f, "%.0fdeg");
            ImGui::Checkbox("Align to Surface Normal", &bs.align_to_normal);
            
            ImGui::Spacing();
            ImGui::Text("Pivot Adjustment:");
            // Using DragFloatRange2 for min/max offset
            ImGui::DragFloatRange2("Y Offset", &bs.y_offset_min, &bs.y_offset_max, 0.05f, -5.0f, 5.0f, "Min: %.2f", "Max: %.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Offset the scattered instances vertically (e.g. to fix pivot issues)");
            }
        }

        ImGui::Spacing();
        ImGui::Separator();

        // Post-Scatter Tools
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Offset Tools:");
        static float global_y_offset = 0.0f;
        ImGui::DragFloat("Y Offset", &global_y_offset, 0.05f, -10.0f, 10.0f);
        if (ImGui::Button("Apply Offset to All")) {
            if (active_group && !active_group->instances.empty()) {
                for (auto& inst : active_group->instances) {
                    inst.position.y += global_y_offset;
                }

                // Sync group to scene
                syncInstancesToScene(ctx, *active_group, false);

                // Rebuild BVH and OptiX
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                }
                SCENE_LOG_INFO("[Scatter] Applied Y-Offset of " + std::to_string(global_y_offset) + " to " + std::to_string(active_group->instances.size()) + " instances.");
            }
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


        ImGui::PopStyleVar();
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // SYNC INSTANCES TO SCENE (Make them renderable)
    // ═══════════════════════════════════════════════════════════════════════════════
}
void SceneUI::syncInstancesToScene(UIContext& ctx, InstanceGroup& group, bool clear_only) {
    // 1. Compute Prefix for identification
    // 1. Compute Prefix for identification
    // NEW: Use Group ID to guarantee uniqueness regardless of name overlaps or renames
    std::string instance_prefix = "_inst_gid" + std::to_string(group.id) + "_";
    
    // Legacy Prefix (to clean up old instances during transition)
    std::string legacy_prefix = "_inst_" + group.name + "_";

    // Clear active hittables as we are about to rebuild/clear them
    group.active_hittables.clear();
    
    // 2. Remove existing instances (Single-threaded removal for safety)
    auto& objects = ctx.scene.world.objects;
    
    // Optimized removal: Partition first, then erase
    auto it = std::remove_if(objects.begin(), objects.end(),
        [&](const std::shared_ptr<Hittable>& obj) {
            auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
            if (inst) {
                // Check for NEW ID-based prefix
                if (inst->node_name.find(instance_prefix) == 0) return true;
                // Check for OLD Name-based prefix (cleanup legacy)
                if (inst->node_name.find(legacy_prefix) == 0) return true;
            }
            return false;
        });
    objects.erase(it, objects.end());
    
    if (clear_only || group.instances.empty()) {
        return;
    }

    // 3. Ensure Source Geometry is Ready (For ALL sources)
    // Legacy support: If definitions exist in legacy fields but not in sources, migrate them?
    // Or just treat legacy fields as "Source 0" if sources empty.
    if (group.sources.empty() && !group.source_triangles.empty()) {
        // Create a temporary source wrapper for legacy data
        group.sources.emplace_back(group.source_node_name, group.source_triangles);
        // We'll use this new source structure going forward
    }

    for (auto& source : group.sources) {
        if (!source.bvh || !source.centered_triangles_ptr) {
             if (source.triangles.empty()) continue;

             // Calculate Center (Pivot)
             Vec3 mesh_bbox_min(1e9, 1e9, 1e9);
             Vec3 mesh_bbox_max(-1e9, -1e9, -1e9);
            for (const auto& src_tri : source.triangles) {
                 Matrix4x4 transform = src_tri->getTransformMatrix();
                 
                 // Bake Transform: World Space Vertices
                 // Use Original Vertices to avoid Double Transformation
                 Vec3 v0 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(0), 1.0f)).xyz();
                 Vec3 v1 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(1), 1.0f)).xyz();
                 Vec3 v2 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(2), 1.0f)).xyz();
                 
                 Vec3 v[3] = { v0, v1, v2 };
                 for(int k=0; k<3; k++) {
                     mesh_bbox_min = Vec3::min(mesh_bbox_min, v[k]);
                     mesh_bbox_max = Vec3::max(mesh_bbox_max, v[k]);
                 }
             }
             Vec3 mesh_center = (mesh_bbox_min + mesh_bbox_max) * 0.5f;
             mesh_center.y = mesh_bbox_min.y; // Pivot at Bottom-Center

             // Create Centered Copies
             auto centered_tris = std::make_shared<std::vector<std::shared_ptr<Triangle>>>();
             centered_tris->reserve(source.triangles.size());
             
             std::vector<std::shared_ptr<Hittable>> hittables_for_bvh;
             hittables_for_bvh.reserve(source.triangles.size());

             bool first_tri_logged = false;
             for (const auto& src_tri : source.triangles) {
                 Matrix4x4 transform = src_tri->getTransformMatrix();
                 
                 // DEBUG: Log the transform and vertices to diagnose "lying down" issue
                 if (!first_tri_logged) {
                     Vec3 o0 = src_tri->getOriginalVertexPosition(0);
                     Vec3 v0_world = transform.multiplyVector(Vec4(o0, 1.0f)).xyz();
                     
                     SCENE_LOG_INFO("[Scatter DEBUG] Source: " + source.name + 
                         " | Original V0: " + std::to_string(o0.x) + "," + std::to_string(o0.y) + "," + std::to_string(o0.z) +
                         " | World V0: " + std::to_string(v0_world.x) + "," + std::to_string(v0_world.y) + "," + std::to_string(v0_world.z));
                         
                     // Log Transform Basis (rudimentary)
                     Vec3 right = transform.multiplyVector(Vec4(1,0,0,0)).xyz();
                     Vec3 up = transform.multiplyVector(Vec4(0,1,0,0)).xyz();
                     Vec3 fwd = transform.multiplyVector(Vec4(0,0,1,0)).xyz();
                     SCENE_LOG_INFO("[Scatter DEBUG] Transform Basis X: " + std::to_string(right.x) + "," + std::to_string(right.y) + "," + std::to_string(right.z));
                     SCENE_LOG_INFO("[Scatter DEBUG] Transform Basis Y: " + std::to_string(up.x) + "," + std::to_string(up.y) + "," + std::to_string(up.z));
                     
                     first_tri_logged = true;
                 }
                 
                 // Convert World (Baked) -> Local (Centered)
                 // Use Original Vertices
                 Vec3 v0 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(0), 1.0f)).xyz() - mesh_center;
                 Vec3 v1 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(1), 1.0f)).xyz() - mesh_center;
                 Vec3 v2 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexPosition(2), 1.0f)).xyz() - mesh_center;
                 
                 // Transform Normals (Rotation only)
                 Vec3 n0 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexNormal(0), 0.0f)).xyz().normalize();
                 Vec3 n1 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexNormal(1), 0.0f)).xyz().normalize();
                 Vec3 n2 = transform.multiplyVector(Vec4(src_tri->getOriginalVertexNormal(2), 0.0f)).xyz().normalize();
                 
                 auto new_tri = std::make_shared<Triangle>(
                     v0, v1, v2, 
                     n0, n1, n2,
                     src_tri->t0, src_tri->t1, src_tri->t2,
                     src_tri->getMaterial()
                 );
                  // Only used for Optix BLAS identification.
                  // Append _BAKED to ensure it doesn't reuse the source mesh BLAS.
                  new_tri->setNodeName(source.name + "_BAKED"); 
                  
                  centered_tris->push_back(new_tri);
                 hittables_for_bvh.push_back(new_tri);
             }
             
             source.centered_triangles_ptr = centered_tris;

             // Build BVH for Source
             auto bvh = std::make_shared<EmbreeBVH>();
             bvh->build(hittables_for_bvh);
             source.bvh = bvh;
             SCENE_LOG_INFO("[Scatter] Built Source BVH for source: " + source.name);
        }
    }
    
    // Sync Legacy Pointers for backward compatibility (if needed by other code)
    if (!group.sources.empty()) {
        group.source_bvh = group.sources[0].bvh;
        group.source_triangles_ptr = group.sources[0].centered_triangles_ptr;
    }

    // 4. Memory Reservation
    size_t num_instances = group.instances.size();
    size_t start_offset = objects.size();
    objects.resize(start_offset + num_instances);
    group.active_hittables.resize(num_instances); // Resize to match instances for 1:1 mapping
    
    // 5. Create Instances
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    size_t chunk_size = (num_instances + num_threads - 1) / num_threads;
    
    std::vector<std::future<void>> futures;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, num_instances);
        if (start >= end) continue;

        futures.push_back(std::async(std::launch::async, 
            [&, start, end, start_offset]() {
                for (size_t i = start; i < end; ++i) {
                    const auto& inst_transform = group.instances[i];
                    
                    // Validate source index
                    int src_idx = inst_transform.source_index;
                    if (src_idx < 0 || src_idx >= group.sources.size()) src_idx = 0;
                    if (group.sources.empty()) continue; // Should not happen due to check above

                    auto& source = group.sources[src_idx];
                    if (!source.bvh) continue;

                    Matrix4x4 mat = inst_transform.toMatrix();
                    
                    char name_buf[64];
                    snprintf(name_buf, sizeof(name_buf), "%s%zu", instance_prefix.c_str(), i);
                    
                    auto hit_inst = std::make_shared<HittableInstance>(
                        source.bvh, 
                        source.centered_triangles_ptr, 
                        mat, 
                        std::string(name_buf)
                    );
                    // FIXED: Do NOT overwrite node_name with source name. 
                    // We must keep name_buf (_inst_Group_ID) for deletion logic to work!
                    // hit_inst->node_name = source.name + "_" + std::to_string(i); 
                    
                    objects[start_offset + i] = hit_inst;
                    group.active_hittables[i] = hit_inst; // Link for wind animation
                }
            }));
    }
    
    for (auto& f : futures) f.get();
    
    // Remove any failed instances (nullptrs) to prevent crashes
    auto& objs_ref = ctx.scene.world.objects;
    objs_ref.erase(std::remove(objs_ref.begin(), objs_ref.end(), nullptr), objs_ref.end());

    SCENE_LOG_INFO("[Scatter] Instanced " + std::to_string(num_instances) + " objects (Multi-Source).");
}

void SceneUI::appendInstancesToScene(UIContext& ctx, InstanceGroup& group, size_t start_index) {
    if (start_index >= group.instances.size()) return;
    
    std::string instance_prefix = "_inst_gid" + std::to_string(group.id) + "_";
    auto& objects = ctx.scene.world.objects;
 
    // Ensure sources are available
    // Ensure sources are available and BVHs are built
    bool sources_ready = !group.sources.empty();
    if (sources_ready) {
        for(const auto& s : group.sources) {
            if (!s.bvh) { sources_ready = false; break; }
        }
    }

    if (!sources_ready) {
         syncInstancesToScene(ctx, group, false);
         return; // Sync handles adding everything, so we return to avoid duplicates.
    }

    size_t count_to_add = group.instances.size() - start_index;
    size_t start_offset = objects.size();
    objects.resize(start_offset + count_to_add);
    group.active_hittables.resize(group.instances.size()); // Grow to hold new instances
    
    // Parallel Append
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    size_t chunk_size = (count_to_add + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t inst_start = start_index + t * chunk_size;
        size_t inst_end = std::min(inst_start + chunk_size, group.instances.size());
        
        if (inst_start >= inst_end) continue;
        
        // Calculate where to write in objects array
        size_t write_start = start_offset + (inst_start - start_index);

        futures.push_back(std::async(std::launch::async, 
            [&, inst_start, inst_end, write_start]() {
                for (size_t i = inst_start; i < inst_end; ++i) {
                    const auto& inst_transform = group.instances[i];
                    
                    // Validate source index
                    int src_idx = inst_transform.source_index;
                    if (src_idx < 0 || src_idx >= group.sources.size()) src_idx = 0;
                    
                    // Safety check
                    if (group.sources.empty()) continue;
                    auto& source = group.sources[src_idx];
                    if (!source.bvh) continue;

                    Matrix4x4 transform = inst_transform.toMatrix();
                    
                    char name_buf[64];
                    snprintf(name_buf, sizeof(name_buf), "%s%zu", instance_prefix.c_str(), i);
                    
                    auto hit_inst = std::make_shared<HittableInstance>(
                        source.bvh, 
                        source.centered_triangles_ptr, 
                        transform, 
                        std::string(name_buf)
                    );
                    
                    size_t write_idx = write_start + (i - inst_start);
                    objects[write_idx] = hit_inst;
                    // Map to the correct index in active_hittables (same as instances index 'i')
                    group.active_hittables[i] = hit_inst;
                }
            }));
    }
    
    for (auto& f : futures) f.get();
}


// ═══════════════════════════════════════════════════════════════════════════════
// VIEWPORT BRUSH INTERACTION
// ═══════════════════════════════════════════════════════════════════════════════

#include "TerrainManager.h"
#include <random>

// ═══════════════════════════════════════════════════════════════════════════════
// FOLIAGE BRUSH INTERACTION (Terrain Specific)
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::handleTerrainFoliageBrush(UIContext& ctx) {
    if (!foliage_brush.enabled) return;
    if (foliage_brush.active_group_id < 0) return;

    InstanceManager& im = InstanceManager::getInstance();
    InstanceGroup* group = im.getGroup(foliage_brush.active_group_id);
    if (!group) return;

    // Check for Mouse Release (Commit changes Lazy Update)
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && foliage_brush.pending_group_id != -1) {
        if (foliage_brush.pending_group_id == group->id) {
             // Determine start index for fast append (Add mode only)
             size_t total = group->instances.size();
             size_t added = foliage_brush.pending_instances.size();
             size_t start_idx = (total >= added) ? (total - added) : 0;
             
             if (foliage_brush.mode == 0) { // ADD
                 appendInstancesToScene(ctx, *group, start_idx);
             } else { // REMOVE (Full sync required)
                 syncInstancesToScene(ctx, *group, false);
             }

             // Rebuild BVH and OptiX (Once per stroke)
             ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
             ctx.renderer.resetCPUAccumulation();
             if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                 ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
             }
             SCENE_LOG_INFO("[Foliage] Committed stroke.");
        }
        foliage_brush.pending_group_id = -1;
        foliage_brush.pending_instances.clear();
    }

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    // Get mouse position
    int mx, my;
    SDL_GetMouseState(&mx, &my);
    float u = (float)mx / io.DisplaySize.x;
    float v = 1.0f - ((float)my / io.DisplaySize.y);

    // Raycast
    // Raycast (Terrain Only)
    Ray ray = ctx.scene.camera->get_ray(u, v);
    
    // Find closest terrain hit
    float closest_t = 1e20f;
    Vec3 hit_point;
    Vec3 hit_normal;
    bool found_terrain = false;
    
    auto& terrains = TerrainManager::getInstance().getTerrains();
    for (auto& terrain : terrains) {
        float t_out;
        Vec3 n_out;
        if (TerrainManager::getInstance().intersectRay(&terrain, ray, t_out, n_out)) {
            if (t_out < closest_t) {
                closest_t = t_out;
                hit_point = ray.origin + ray.direction * t_out;
                hit_normal = n_out;
                found_terrain = true;
            }
        }
    }
    
    HitRecord hit;
    bool did_hit = found_terrain;
    
    if (did_hit) {
        hit.t = closest_t;
        hit.point = hit_point;
        hit.normal = hit_normal;
        // hit.triangle is null, but we don't need it for basic painting
    }

    if (did_hit && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        static float last_paint_time = 0.0f;
        float current_time = ImGui::GetTime();
        if (current_time - last_paint_time < 0.1f) return; // Faster throttle for foliage
        last_paint_time = current_time;
        
        bool modified = false;

        if (foliage_brush.mode == 0) { // ADD
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            // Random point in circle
            for (int i = 0; i < foliage_brush.density; ++i) {
                float r = foliage_brush.radius * sqrt(dist(rng));
                float theta = dist(rng) * 2 * 3.14159f;
                
                // Construct tangent space for circle (simple projection)
                Vec3 tangent, bitangent;
                if (std::abs(hit.normal.y) > 0.9f) tangent = Vec3(1, 0, 0);
                else tangent = Vec3::cross(Vec3(0, 1, 0), hit.normal).normalize();
                bitangent = Vec3::cross(hit.normal, tangent).normalize();
                
                Vec3 offset = tangent * (r * cos(theta)) + bitangent * (r * sin(theta));
                Vec3 targetPos = hit.point + offset;
                
                // Re-project simple height? No, for full 3D painting we should ideally raycast down.
                // But for simple terrain painting, using the plane defined by hit.point/normal is okay for small radius.
                // Better: If we have terrain heightmap, use it. If not, stick to tangent plane.
                // User requirement: "layerdeki objeler brush ile de ekilmeli".
                // We'll stick to a simple tangent plane offset for now or just targetPos.
                
                // Use InstanceGroup's setting-aware generator
                // Note: generateRandomTransform applies scale/rotation/slope rules.
                InstanceTransform inst = group->generateRandomTransform(targetPos, hit.normal);
                
                // Basic slope check provided by generateRandomTransform logic if implemented? 
                // Currently generateRandomTransform logic in InstanceGroup uses normal for alignment.
                
                group->addInstance(inst);
                
                if (foliage_brush.lazy_update) {
                    // LAZY: Add to pending for visualization
                    foliage_brush.pending_instances.push_back(inst);
                    foliage_brush.pending_group_id = group->id;
                } 
                else {
                    // REAL-TIME: Immediate Append
                    size_t idx = group->instances.size() - 1;
                    appendInstancesToScene(ctx, *group, idx);
                    modified = true;
                }
            }
        }
        else { // REMOVE
            size_t before = group->instances.size();
            group->removeInstancesInRadius(hit.point, foliage_brush.radius);
            
            if (group->instances.size() < before) {
                 if (foliage_brush.lazy_update) {
                     foliage_brush.pending_group_id = group->id; // Mark for lazy update
                 } else {
                     // REAL-TIME: Full sync required for removal
                     syncInstancesToScene(ctx, *group, false);
                     modified = true;
                 }
            }
        }

        // Trigger Rebuild if modified (Real-time mode)
        if (modified && !foliage_brush.lazy_update) {
             ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
             ctx.renderer.resetCPUAccumulation();
             
             if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                 // Fast instance update
                 ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
             }
        }
        
        hud_captured_mouse = true;
    }
}

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
    
    // Filter out scattered instances from hits
    if (did_hit) {
        // Robust instance check
        if (hit.is_instance_hit) did_hit = false;
        
        // Skip if hit an instance triangle (Legacy fallback)
        if (did_hit && hit.triangle && hit.triangle->getNodeName().find("_inst_") == 0) {
            did_hit = false; 
        }
        
        // Check target surface filter
        if (did_hit && !scatter_brush.target_surface_name.empty() && hit.triangle) {
            if (hit.triangle->getNodeName() != scatter_brush.target_surface_name) {
                did_hit = false;  // Not on target surface
            }
        }
    }
    
    // Check for Mouse Release (Commit changes)
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && scatter_brush.pending_group_id != -1) {
        // Commit the stroke
        if (scatter_brush.pending_group_id == scatter_brush.active_group_id && group) {
             // Only add the NEW instances to the scene (Lazy Update)
            size_t start_idx = scatter_brush.pending_instances.size(); // We re-purposed this field? No.
            // We need to know where we started.
            // Let's rely on the pending_instances vector to append to scene? 
            // Actually, group->instances has ALL of them now.
            // We just need to sync the ones added during this stroke.
            
            // To be safe and simple: just Sync all (or use a static start index).
            // Let's use the static start index approach.
            static size_t stroke_start_index = 0; // Defines start of current stroke
            
            // Wait, static variable inside if block is bad if logic is split.
            // But we can just use appendInstancesToScene with a calculated index?
            // "group->instances.size() - count_added_during_stroke".
            // We can track "count_added" in the struct? 'pending_instances' vector size IS the count added!
            
            size_t total_count = group->instances.size();
            size_t added_count = scatter_brush.pending_instances.size();
            start_idx = (total_count >= added_count) ? (total_count - added_count) : 0;
            
            if (scatter_brush.brush_mode == 0) { // Add Mode
                 appendInstancesToScene(ctx, *group, start_idx);
            } else {
                 // Remove mode needs full sync usually
                 syncInstancesToScene(ctx, *group, false);
            }
            
            // Rebuild BVH & OptiX (Only once per stroke!)
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
            }
            
            SCENE_LOG_INFO("[Scatter] Committed stroke: " + std::to_string(added_count) + " instances.");
        }
        
        scatter_brush.pending_group_id = -1;
        scatter_brush.pending_instances.clear();
    }

    if (did_hit && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        static float last_paint_time = 0.0f;
        float current_time = ImGui::GetTime();
        
        // Start Stroke if not started
        if (scatter_brush.pending_group_id == -1) {
            scatter_brush.pending_group_id = scatter_brush.active_group_id;
            scatter_brush.pending_instances.clear();
        }
        
        // Throttle (faster now since we don't rebuild)
        if (current_time - last_paint_time < 0.03f) return; // 30ms = ~30fps painting
        last_paint_time = current_time;
        
        if (scatter_brush.brush_mode == 0) {
            // ADD mode (Lazy)
            size_t before_count = group->instances.size();
            
            int added = im.paintInstances(
                scatter_brush.active_group_id,
                hit.point,
                hit.normal,
                scatter_brush.brush_radius,
                1.0f
            );
            
            if (added > 0) {
                // Determine which ones were added and store in pending_instances for visualization
                size_t after_count = group->instances.size();
                for(size_t i=before_count; i<after_count; i++) {
                    scatter_brush.pending_instances.push_back(group->instances[i]);
                }
                // DO NOT REBUILD HERE
            }
        }
        else if (scatter_brush.brush_mode == 1) {
            // REMOVE mode (Still somewhat lazy - delay rebuild)
            size_t before = group->instances.size();
            group->removeInstancesInRadius(hit.point, scatter_brush.brush_radius);
            // We don't visualize removal easily, just let it happen in data
            // Visual feedback will lag until release, maybe unacceptable?
            // For removal, maybe we DO want to rebuild? 
            // User asked for "Lazy update". Fast scrubbing removal is fine if it pops later.
            // But user might not know what they erased.
            // Let's force a rebuild for removal? Or maybe a cheap wireframe update?
            // Let's keep removal lazy too for consistency with "Speed".
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
    bool did_hit = false;

    if (is_foliage) {
        // Terrain-only raycast for foliage brush preview
        float closest_t = 1e20f;
        auto& terrains = TerrainManager::getInstance().getTerrains();
        for (auto& terrain : terrains) {
            float t_out;
            Vec3 n_out;
            if (TerrainManager::getInstance().intersectRay(&terrain, ray, t_out, n_out)) {
                if (t_out < closest_t) {
                    closest_t = t_out;
                    hit.t = closest_t;
                    hit.point = ray.origin + ray.direction * t_out;
                    hit.normal = n_out;
                    did_hit = true;
                }
            }
        }
    } else {
        // Standard raycast for generic scatter brush
        did_hit = ctx.scene.bvh->hit(ray, 0.001f, 1e10f, hit);
    }
    
    if (!did_hit) return;
    
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
    // Draw pending instances (feedback for lazy update)
    if (is_scatter && !scatter_brush.pending_instances.empty()) {
        for (const auto& inst : scatter_brush.pending_instances) {
            ImVec2 pt = projectToScreen(inst.position);
            if (pt.x >= 0) {
                draw_list->AddCircleFilled(pt, 3.0f, IM_COL32(255, 255, 0, 200));
            }
        }
    }
    // Draw pending foliage instances
    if (is_foliage && !foliage_brush.pending_instances.empty()) {
        for (const auto& inst : foliage_brush.pending_instances) {
            ImVec2 pt = projectToScreen(inst.position);
            if (pt.x >= 0) {
                draw_list->AddCircleFilled(pt, 3.0f, IM_COL32(0, 255, 255, 200)); // Cyan for foliage
            }
        }
    }
    
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
