// ===============================================================================
// SCENE UI - SELECTION & INTERACTION
// ===============================================================================
// This file handles Mouse picking, Marquee selection, and Delete operations.
// ===============================================================================

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "globals.h"
#include "Backend/VulkanBackend.h"
#include "Backend/OptixBackend.h"
#include "Backend/IViewportBackend.h"
#include "HittableInstance.h"
#include "InstanceManager.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "scene_data.h"
#include <unordered_set>
#include <ProjectManager.h>
#include "WaterSystem.h"
#include "TerrainManager.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <future>
#include <thread>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;
bool g_timeline_selection_sync_pending = false;

namespace {
bool projectSelectionPointToScreen(const Camera& cam, const ImVec2& displaySize, const Vec3& point, ImVec2& out) {
    if (displaySize.x <= 1.0f || displaySize.y <= 1.0f) {
        return false;
    }

    const Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    const Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    const Vec3 to_point = point - cam.lookfrom;
    const float depth = to_point.dot(cam_forward);
    if (depth <= 0.01f) {
        return false;
    }

    const float aspect = displaySize.x / displaySize.y;
    const float tan_half_fov = tanf(cam.vfov * 3.14159265359f / 180.0f * 0.5f);
    if (fabsf(aspect) <= 1e-6f || fabsf(tan_half_fov) <= 1e-6f) {
        return false;
    }

    const float local_x = to_point.dot(cam_right);
    const float local_y = to_point.dot(cam_up);
    const float half_h = depth * tan_half_fov;
    const float half_w = half_h * aspect;
    if (fabsf(half_w) <= 1e-6f || fabsf(half_h) <= 1e-6f) {
        return false;
    }

    out.x = ((local_x / half_w) * 0.5f + 0.5f) * displaySize.x;
    out.y = (0.5f - (local_y / half_h) * 0.5f) * displaySize.y;
    return true;
}

bool useInteractiveViewportSelectionFallback(const UIContext& ctx, const SceneUI& ui) {
    const bool hasInteractiveViewportBackend =
        (dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) != nullptr) ||
        (g_viewport_backend != nullptr);
    return hasInteractiveViewportBackend && ui.viewport_settings.shading_mode != 2;
}

bool selectionNodeMatches(const std::string& candidate, const std::string& target) {
    if (candidate.empty() || target.empty()) return false;
    return candidate == target || candidate.rfind(target + "_mat_", 0) == 0;
}

Backend::IBackend* getSelectionRenderBackend(UIContext& ctx) {
    if (g_backend) {
        return g_backend.get();
    }
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr) {
        return ctx.backend_ptr;
    }
    return nullptr;
}

Backend::IViewportBackend* getSelectionViewportBackend(UIContext& ctx) {
    if (g_viewport_backend) {
        return g_viewport_backend.get();
    }
    return dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
}

void rebuildInteractiveViewportAfterTopologyChange(UIContext& ctx) {
    if (Backend::IViewportBackend* viewportBackend = getSelectionViewportBackend(ctx)) {
        viewportBackend->buildRasterGeometry(ctx.scene.world.objects);
        viewportBackend->resetAccumulation();
    }
}

void setSelectionObjectVisibility(UIContext& ctx, const std::string& nodeName, bool visible) {
    if (nodeName.empty()) return;
    if (ctx.scene.isEditorPendingDeleteObjectName(nodeName)) {
        visible = false;
    }

    if (Backend::IViewportBackend* viewportBackend = getSelectionViewportBackend(ctx)) {
        viewportBackend->setVisibilityByNodeName(nodeName, visible);
    }

    if (Backend::IBackend* renderBackend = getSelectionRenderBackend(ctx)) {
        if (renderBackend != getSelectionViewportBackend(ctx)) {
            renderBackend->setVisibilityByNodeName(nodeName, visible);
        }
    }
}

void setSelectionSceneNodeLocalVisibility(UIContext& ctx, const std::string& nodeName, bool visible) {
    if (nodeName.empty()) return;
    if (ctx.scene.isEditorPendingDeleteObjectName(nodeName)) {
        visible = false;
    }

    for (const auto& world_obj : ctx.scene.world.objects) {
        if (!world_obj) continue;

        if (auto tri = std::dynamic_pointer_cast<Triangle>(world_obj)) {
            if (selectionNodeMatches(tri->getNodeName(), nodeName)) {
                tri->visible = visible;
            }
            continue;
        }

        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(world_obj)) {
            if (selectionNodeMatches(inst->node_name, nodeName)) {
                inst->visible = visible;
            }
        }
    }
}

bool selectionHasGpuRenderBackend(UIContext& ctx) {
    Backend::IBackend* renderBackend = getSelectionRenderBackend(ctx);
    if (!renderBackend) return false;
    // Don't treat viewport-only backends as GPU selection targets
    if (dynamic_cast<Backend::IViewportBackend*>(renderBackend) != nullptr) return false;

    // If Vulkan backend is present but ray-tracing pipeline / TLAS isn't ready,
    // fall back to CPU selection (we only want GPU selection when path-tracing is available).
    if (auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(renderBackend)) {
        auto* dev = vk->getVulkanDevice();
        if (!dev) return false;
        if (!dev->isRTReady() || !dev->hasTLAS()) return false;
        return true;
    }

    if (dynamic_cast<Backend::OptixBackend*>(renderBackend) != nullptr) return true;

    return false;
}

void syncSelectionSceneState(UIContext& ctx, bool syncLights, bool syncCamera) {
    Backend::IBackend* renderBackend = getSelectionRenderBackend(ctx);
    Backend::IViewportBackend* viewportBackend = getSelectionViewportBackend(ctx);

    if (syncLights) {
        if (renderBackend) {
            renderBackend->setLights(ctx.scene.lights);
            renderBackend->resetAccumulation();
        }
        if (viewportBackend && viewportBackend != renderBackend) {
            viewportBackend->setLights(ctx.scene.lights);
            viewportBackend->resetAccumulation();
        }
    }

    if (syncCamera && ctx.scene.camera) {
        if (renderBackend) {
            renderBackend->syncCamera(*ctx.scene.camera);
            renderBackend->resetAccumulation();
        }
        if (viewportBackend && viewportBackend != renderBackend) {
            viewportBackend->syncCamera(*ctx.scene.camera);
            viewportBackend->resetAccumulation();
        }
    }
}

}



void SceneUI::handleMarqueeSelection(UIContext& ctx) {
    ImGuiIO& io = ImGui::GetIO();

    // Only handle when not interacting with UI windows and not using gizmo
    // WantCaptureMouse is true when mouse is over an interactive UI element (button, slider, etc.)
    // This is less restrictive than IsAnyItemHovered which blocks even when hovering inactive areas
    if (io.WantCaptureMouse || ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
        return;
    }

    // Start marquee on right mouse button down (or B key + left click for Blender style)
    bool start_marquee = ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !io.KeyCtrl && !io.KeyShift;

    if (start_marquee && !is_marquee_selecting) {
        is_marquee_selecting = true;
        marquee_start = io.MousePos;
        marquee_end = io.MousePos;
    }

    // Update marquee while dragging
    if (is_marquee_selecting && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        marquee_end = io.MousePos;
    }

    // Complete marquee on mouse release
    if (is_marquee_selecting && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        is_marquee_selecting = false;

        // Normalize rectangle
        float x1 = fminf(marquee_start.x, marquee_end.x);
        float y1 = fminf(marquee_start.y, marquee_end.y);
        float x2 = fmaxf(marquee_start.x, marquee_end.x);
        float y2 = fmaxf(marquee_start.y, marquee_end.y);

        // Minimum size to prevent accidental selections
        if ((x2 - x1) < 10 || (y2 - y1) < 10) {
            return;
        }

        // Clear current selection if Ctrl is not held
        if (!io.KeyCtrl) {
            ctx.selection.clearSelection();
        }

        if (!ctx.scene.camera) return;

        Camera& cam = *ctx.scene.camera;
        float screen_w = io.DisplaySize.x;
        float screen_h = io.DisplaySize.y;

        // Camera basis vectors for projection
        Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
        Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
        Vec3 cam_up = cam_right.cross(cam_forward).normalize();
        float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
        float tan_half_fov = tanf(fov_rad * 0.5f);

        // FIX: Use actual screen aspect ratio, not the global render aspect_ratio.
        // The global aspect_ratio may be locked to e.g. 16:9 for final render but
        // the viewport can have any size. Mismatch causes horizontal misalignment.
        const float viewport_aspect = (screen_h > 1.0f) ? (screen_w / screen_h) : 1.0f;

        // Lambda to project 3D world-space point to screen coordinates.
        // Returns {-10000,-10000} when the point is behind the camera.
        auto ProjectToScreen = [&](const Vec3& p) -> ImVec2 {
            Vec3 to_point = p - cam.lookfrom;
            float depth = to_point.dot(cam_forward);
            if (depth <= 0.01f) return ImVec2(-10000, -10000);

            float local_x = to_point.dot(cam_right);
            float local_y = to_point.dot(cam_up);

            float half_height = depth * tan_half_fov;
            float half_width = half_height * viewport_aspect;

            float ndc_x = local_x / half_width;
            float ndc_y = local_y / half_height;

            return ImVec2(
                (ndc_x * 0.5f + 0.5f) * screen_w,
                (0.5f - ndc_y * 0.5f) * screen_h
            );
            };

        // Check which objects are inside the marquee
        if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

        int skipped_procedural = 0;

        for (auto& [name, triangles] : mesh_cache) {
            if (triangles.empty()) continue;

            // IMPORTANT: Check if all triangles share the same TransformHandle
            // Procedural objects may have separate transforms per triangle
            Transform* firstHandle = triangles[0].second->getTransformPtr();
            bool all_same_transform = true;

            // Performance: for objects with many triangles, only sample a few
            const size_t check_limit = std::min(triangles.size(), (size_t)16);
            for (size_t i = 1; i < check_limit && all_same_transform; ++i) {
                Transform* handle = triangles[i].second->getTransformPtr();
                if (handle != firstHandle) {
                    all_same_transform = false;
                }
            }

            if (!all_same_transform) {
                skipped_procedural++;
                continue;
            }

            // Get the object's transform matrix (for converting local AABB to world space)
            Matrix4x4 obj_transform = Matrix4x4::identity();
            bool has_transform = false;
            if (firstHandle) {
                obj_transform = firstHandle->getMatrix();
                has_transform = true;
            }

            // Get AABB - cached is in LOCAL space, fallback computes in WORLD space
            Vec3 bb_min, bb_max;
            bool bbox_is_local = false;
            auto bbox_it = bbox_cache.find(name);
            if (bbox_it != bbox_cache.end()) {
                bb_min = bbox_it->second.first;
                bb_max = bbox_it->second.second;
                bbox_is_local = true; // cached bbox uses getOriginalVertexPosition (local space)
            } else {
                // Fallback: calculate from transformed vertices (already world space)
                bb_min = Vec3(1e10f, 1e10f, 1e10f);
                bb_max = Vec3(-1e10f, -1e10f, -1e10f);
                for (auto& pair : triangles) {
                    auto& tri = pair.second;
                    Vec3 v0 = tri->getV0();
                    Vec3 v1 = tri->getV1();
                    Vec3 v2 = tri->getV2();
                    bb_min.x = fminf(bb_min.x, fminf(v0.x, fminf(v1.x, v2.x)));
                    bb_min.y = fminf(bb_min.y, fminf(v0.y, fminf(v1.y, v2.y)));
                    bb_min.z = fminf(bb_min.z, fminf(v0.z, fminf(v1.z, v2.z)));
                    bb_max.x = fmaxf(bb_max.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
                    bb_max.y = fmaxf(bb_max.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
                    bb_max.z = fmaxf(bb_max.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
                }
            }

            // Build 8 corners of the AABB
            Vec3 local_corners[8] = {
                Vec3(bb_min.x, bb_min.y, bb_min.z), Vec3(bb_max.x, bb_min.y, bb_min.z),
                Vec3(bb_min.x, bb_max.y, bb_min.z), Vec3(bb_max.x, bb_max.y, bb_min.z),
                Vec3(bb_min.x, bb_min.y, bb_max.z), Vec3(bb_max.x, bb_min.y, bb_max.z),
                Vec3(bb_min.x, bb_max.y, bb_max.z), Vec3(bb_max.x, bb_max.y, bb_max.z)
            };

            // FIX: Transform local-space corners to world space when using cached bbox.
            // bbox_cache stores LOCAL (original) vertex positions. Without this transform,
            // moved/rotated/scaled objects would be tested at their untransformed position,
            // causing "select all" or "select none" bugs in crowded scenes.
            Vec3 world_corners[8];
            if (bbox_is_local && has_transform) {
                for (int ci = 0; ci < 8; ++ci) {
                    world_corners[ci] = obj_transform.transform_point(local_corners[ci]);
                }
            } else {
                for (int ci = 0; ci < 8; ++ci) {
                    world_corners[ci] = local_corners[ci];
                }
            }

            // Early rejection: check if the object center is entirely behind the camera
            // (all 8 corners behind camera → skip entirely)
            int behind_count = 0;
            for (int ci = 0; ci < 8; ++ci) {
                Vec3 to_corner = world_corners[ci] - cam.lookfrom;
                if (to_corner.dot(cam_forward) <= 0.01f) {
                    behind_count++;
                }
            }
            if (behind_count == 8) {
                continue; // entire object behind camera
            }

            float proj_min_x = 1e10f, proj_min_y = 1e10f;
            float proj_max_x = -1e10f, proj_max_y = -1e10f;
            bool has_projected_point = false;
            bool has_behind_corner = (behind_count > 0);

            for (int ci = 0; ci < 8; ++ci) {
                ImVec2 sp = ProjectToScreen(world_corners[ci]);
                if (sp.x <= -9999.0f || sp.y <= -9999.0f) {
                    continue;
                }

                has_projected_point = true;
                proj_min_x = fminf(proj_min_x, sp.x);
                proj_min_y = fminf(proj_min_y, sp.y);
                proj_max_x = fmaxf(proj_max_x, sp.x);
                proj_max_y = fmaxf(proj_max_y, sp.y);
            }

            // FIX: When some corners are behind the camera, the projected AABB can be
            // too small (missing corners shrink it). Expand to full screen extent because
            // the object straddles the camera plane and could fill much of the viewport.
            if (has_projected_point && has_behind_corner) {
                proj_min_x = fminf(proj_min_x, 0.0f);
                proj_min_y = fminf(proj_min_y, 0.0f);
                proj_max_x = fmaxf(proj_max_x, screen_w);
                proj_max_y = fmaxf(proj_max_y, screen_h);
            }

            if (!has_projected_point) {
                // All corners behind camera but center might be in front (shouldn't happen
                // after the behind_count==8 early-out, but kept as safety)
                Vec3 center = (world_corners[0] + world_corners[7]) * 0.5f;
                ImVec2 center_screen = ProjectToScreen(center);
                if (center_screen.x > -9999.0f && center_screen.y > -9999.0f) {
                    has_projected_point = true;
                    proj_min_x = proj_max_x = center_screen.x;
                    proj_min_y = proj_max_y = center_screen.y;
                }
            }

            bool overlaps_marquee = has_projected_point &&
                proj_max_x >= x1 && proj_min_x <= x2 &&
                proj_max_y >= y1 && proj_min_y <= y2;

            if (overlaps_marquee) {
                SelectableItem item;
                item.type = SelectableType::Object;
                item.object = triangles[0].second;
                item.object_index = triangles[0].first;
                item.name = name;

                if (!ctx.selection.isSelected(item)) {
                    ctx.selection.addToSelection(item);
                }
            }
        }

        if (skipped_procedural > 0) {
            SCENE_LOG_WARN("Skipped " + std::to_string(skipped_procedural) + " objects with mixed transforms (use Ctrl+Click)");
            addViewportMessage("Skipped " + std::to_string(skipped_procedural) + " objects (Mixed Transforms)", 3.0f, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
        }

        // Also check lights
        for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
            auto& light = ctx.scene.lights[i];
            if (!light) continue;

            ImVec2 screenPos = ProjectToScreen(light->position);

            if (screenPos.x >= x1 && screenPos.x <= x2 && screenPos.y >= y1 && screenPos.y <= y2) {
                SelectableItem item;
                item.type = SelectableType::Light;
                item.light = light;
                item.light_index = (int)i;
                item.name = "Light_" + std::to_string(i);

                if (!ctx.selection.isSelected(item)) {
                    ctx.selection.addToSelection(item);
                }
            }
        }

        if (ctx.selection.multi_selection.size() > 0) {
            // [VERBOSE] SCENE_LOG_INFO("Marquee selected " + std::to_string(ctx.selection.multi_selection.size()) + " items");
        }
    }

    // Draw the marquee rectangle while selecting
    drawMarqueeRect();
}

void SceneUI::handleDeleteShortcut(UIContext& ctx)
{
    if (!ImGui::IsKeyPressed(ImGuiKey_Delete) &&
        !ImGui::IsKeyPressed(ImGuiKey_X)) return;

    // Use the optimized multi-selection capable triggerDelete
    if (ctx.selection.hasSelection()) {
        triggerDelete(ctx);
    }
}

void SceneUI::handleMouseSelection(UIContext& ctx) {
    // Only select if not interacting with UI or Gizmo
    if (ImGui::IsMouseClicked(0)) {
        bool capture = ImGui::GetIO().WantCaptureMouse;
        bool gizmo_over = ImGuizmo::IsOver();
        if (gizmo_over && !ctx.selection.hasSelection()) {
            gizmo_over = false;
        }
        
        //// Detailed logging for diagnostics
        //SCENE_LOG_INFO("Viewport click. Cache: " + std::string(mesh_cache_valid ? "OK" : "NO") +
        //               ", ObjCount: " + std::to_string(ctx.scene.world.objects.size()) +
        //               ", TriToIdx: " + std::to_string(tri_to_index.size()) +
        //               ", Capture=" + std::to_string(capture) +
        //               ", GizmoOver=" + std::to_string(gizmo_over) +
        //               ", HUD=" + std::to_string(hud_captured_mouse) +
        //               ", Dragging=" + std::to_string(is_dragging) +
        //               ", BVH=" + std::string(ctx.scene.bvh ? "yes" : "no"));
                       
        if (!is_dragging) {
            // Ignore click if over UI elements (Window/Panel), Gizmo, or HUD overlay
            if (capture || gizmo_over || hud_captured_mouse) {
                // Reset HUD flag for next frame
                hud_captured_mouse = false;
                return;
            }
        }

        // ===========================================================================
        // SELECTION CACHE SYNC (Now centralized in update(), but kept here as safety)
        // ===========================================================================
        if (!mesh_cache_valid) {
            rebuildMeshCache(ctx.scene.world.objects);
        }

        // ===========================================================================
        // CPU VERTEX SYNC FOR PICKING: Triangle::hit() uses vertices[i].position
        // which must be in world space. In Solid/Matcap viewport the Vulkan raster
        // backend uploads getOriginalVertexPosition() + GPU transform, so CPU .position
        // may never have been synced from the TransformHandle. This causes complete
        // selection failure: the ray is in world space but triangles are tested at
        // their local-space positions.
        //
        // Sync when:
        //  - g_bvh_rebuild_pending (transforms changed via gizmo/animation/load)
        //  - !picking_vertices_synced (first click after scene load / cache clear)
        //
        // CRITICAL: Runs in ALL viewport modes (Solid, Matcap, Rendered).
        // ===========================================================================
        extern bool g_bvh_rebuild_pending;
        const bool interactive_selection_fallback = useInteractiveViewportSelectionFallback(ctx, *this);

        // Process lazy sync queue first (objects moved with gizmo).
        ensureCPUSyncForPicking(ctx);

        const bool pending_timeline_selection_sync = g_timeline_selection_sync_pending;

        if (g_bvh_rebuild_pending || !picking_vertices_synced || pending_timeline_selection_sync) {
            int synced_objects = 0;
            if (mesh_cache_valid) {
                // First-click vertex sync used to walk every triangle of every
                // node single-threaded with a per-triangle matrix fetch + 4x4
                // inverse — seconds of stall on dense scenes. Restructured:
                //  - instance sync stays sequential (clones can share source
                //    triangle arrays — not thread-safe),
                //  - per node the final/normal matrices are computed ONCE on
                //    this thread (Transform::getFinal() lazily writes its
                //    cache, so workers must never touch the handle),
                //  - the flat triangle work list is then chunked across
                //    threads with updateTransformedVerticesWith (pure
                //    per-triangle writes, no shared state).
                //  - skinned nodes are EXCLUDED here: their world positions
                //    come from apply_skinning below, and rigid-transforming
                //    them first would just be overwritten (or worse, clobber
                //    a still-valid skinned pose when the skin pass skips).
                std::vector<std::pair<Matrix4x4, Matrix4x4>> sync_xforms;
                std::vector<std::pair<Triangle*, uint32_t>> sync_work;
                for (auto& [obj_name, tris] : mesh_cache) {
                    if (tris.empty()) continue;
                    if (!tris[0].second->getTransformPtr()) continue;
                    const bool skinned_bucket = tris[0].second->hasAnySkinWeights();
                    const Transform* lastHandle = nullptr;
                    for (auto& pair : tris) {
                        // Triangles with skin data get their world positions
                        // from the apply_skinning pass below; rigid-syncing
                        // them here would clobber a still-valid skinned pose.
                        // Skinned nodes can still carry skinless triangles —
                        // those keep the rigid path.
                        if (skinned_bucket && pair.second->hasSkinData()) continue;
                        const int object_index = pair.first;
                        if (object_index >= 0 && static_cast<size_t>(object_index) < ctx.scene.world.objects.size()) {
                            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(ctx.scene.world.objects[object_index])) {
                                if (inst->syncTransformFromSourceTriangles()) {
                                    continue;
                                }
                            }
                        }
                        Transform* h = pair.second->getTransformPtr();
                        if (h != lastHandle) {
                            Matrix4x4 finalT = pair.second->getTransformMatrix();
                            sync_xforms.emplace_back(finalT, finalT.inverse().transpose());
                            lastHandle = h;
                        }
                        sync_work.emplace_back(pair.second.get(),
                                               (uint32_t)(sync_xforms.size() - 1));
                    }
                    ++synced_objects;
                }

                auto runSyncRange = [&](size_t s, size_t e) {
                    for (size_t i = s; i < e; ++i) {
                        const auto& xf = sync_xforms[sync_work[i].second];
                        sync_work[i].first->updateTransformedVerticesWith(xf.first, xf.second);
                    }
                };
                const size_t kParallelSyncThreshold = 16384;
                unsigned syncThreads = std::thread::hardware_concurrency();
                if (syncThreads == 0) syncThreads = 4;
                if (sync_work.size() < kParallelSyncThreshold || syncThreads < 2) {
                    runSyncRange(0, sync_work.size());
                } else {
                    const size_t chunk = (sync_work.size() + syncThreads - 1) / syncThreads;
                    std::vector<std::future<void>> futures;
                    futures.reserve(syncThreads);
                    for (unsigned t = 0; t < syncThreads; ++t) {
                        const size_t s = t * chunk;
                        const size_t e = (std::min)(s + chunk, sync_work.size());
                        if (s >= e) break;
                        futures.push_back(std::async(std::launch::async, runSyncRange, s, e));
                    }
                    for (auto& f : futures) f.get();
                }
            } else {
                // Skip foliage tail: InstanceManager always appends at end
                size_t foliage_count = InstanceManager::getInstance().getTotalInstanceCount();
                size_t selectable_count = (foliage_count <= ctx.scene.world.objects.size())
                                              ? (ctx.scene.world.objects.size() - foliage_count)
                                              : ctx.scene.world.objects.size();
                for (size_t si = 0; si < selectable_count; ++si) {
                    const auto& obj = ctx.scene.world.objects[si];
                    if (!obj) continue;
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (tri && tri->getTransformPtr()) {
                        tri->updateTransformedVertices();
                        ++synced_objects;
                        continue;
                    }
                    auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
                    if (inst && inst->syncTransformFromSourceTriangles()) {
                        ++synced_objects;
                        continue;
                    }
                    if (inst && inst->source_triangles) {
                        for (auto& srcTri : *inst->source_triangles) {
                            if (srcTri && srcTri->getTransformPtr()) {
                                srcTri->updateTransformedVertices();
                            }
                        }
                        ++synced_objects;
                    }
                }
            }
            picking_vertices_synced = true;
            g_timeline_selection_sync_pending = false;
            // Don't clear g_bvh_rebuild_pending here. Main.cpp's async BVH builder
            // needs it to reconstruct the acceleration structure with updated positions.
            // We only sync vertices for the linear scan / hit tests on this click.
        }

        // ===========================================================================
        // SKINNED MESH FIX: When using GPU rendering with animations,
        // CPU vertices may be out of sync. Force a sync for picking accuracy.
        // Re-skin ONLY when the pose (or a skinned node's root transform)
        // actually changed since the last click-sync — repeated clicks on a
        // paused scene used to re-skin the entire cast every time. The old
        // loop also RTTI-cast every world object including the 2M+ foliage
        // tail; the selection cache buckets already exclude foliage.
        // ===========================================================================
        if (!ctx.scene.animationDataList.empty() &&
            !ctx.renderer.finalBoneMatrices.empty() && mesh_cache_valid) {
            uint64_t skin_hash = 1469598103934665603ull;
            auto mixSkin = [&](uint64_t v) { skin_hash ^= v; skin_hash *= 1099511628211ull; };
            auto mixSkinF = [&](float f) { uint32_t b; std::memcpy(&b, &f, 4); mixSkin(b); };
            const auto& pick_bones = ctx.renderer.finalBoneMatrices;
            mixSkin(pick_bones.size());
            {
                const size_t step = (pick_bones.size() > 16) ? (pick_bones.size() / 16) : 1;
                for (size_t bi = 0; bi < pick_bones.size(); bi += step) {
                    const float* mm = &pick_bones[bi].m[0][0];
                    mixSkinF(mm[3]); mixSkinF(mm[7]); mixSkinF(mm[11]); mixSkinF(mm[0]);
                }
            }

            std::vector<Triangle*> skin_work;
            for (auto& [obj_name, tris] : mesh_cache) {
                if (tris.empty() || !tris[0].second->hasAnySkinWeights()) continue;
                // Root transform joins the hash (apply_skinning output depends
                // on it) and the handle's lazy matrix cache is warmed HERE so
                // the parallel workers below never write to it.
                Matrix4x4 m = tris[0].second->getTransformMatrix();
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 4; ++c) mixSkinF(m.m[r][c]);
                for (auto& pair : tris) {
                    if (pair.second && pair.second->hasSkinData()) {
                        skin_work.push_back(pair.second.get());
                    }
                }
            }

            static uint64_t s_last_pick_skin_hash = 0;
            if (!skin_work.empty() && skin_hash != s_last_pick_skin_hash) {
                auto runSkinRange = [&](size_t s, size_t e) {
                    for (size_t i = s; i < e; ++i) {
                        skin_work[i]->apply_skinning(pick_bones);
                    }
                };
                const size_t kParallelSkinThreshold = 8192;
                unsigned skinThreads = std::thread::hardware_concurrency();
                if (skinThreads == 0) skinThreads = 4;
                if (skin_work.size() < kParallelSkinThreshold || skinThreads < 2) {
                    runSkinRange(0, skin_work.size());
                } else {
                    const size_t chunk = (skin_work.size() + skinThreads - 1) / skinThreads;
                    std::vector<std::future<void>> futures;
                    futures.reserve(skinThreads);
                    for (unsigned t = 0; t < skinThreads; ++t) {
                        const size_t s = t * chunk;
                        const size_t e = (std::min)(s + chunk, skin_work.size());
                        if (s >= e) break;
                        futures.push_back(std::async(std::launch::async, runSkinRange, s, e));
                    }
                    for (auto& f : futures) f.get();
                }
                s_last_pick_skin_hash = skin_hash;
                SCENE_LOG_INFO("Synced skinned mesh vertices for viewport selection (" +
                               std::to_string(skin_work.size()) + " tris)");
            }
        }

        // Check if Ctrl is held for multi-selection
        bool ctrl_held = ImGui::GetIO().KeyCtrl;

        int x, y;
        SDL_GetMouseState(&x, &y);

        float win_w = ImGui::GetIO().DisplaySize.x;
        float win_h = ImGui::GetIO().DisplaySize.y;

        float u = (float)x / win_w;
        float v = (float)y / win_h;
        v = 1.0f - v;

        if (ctx.scene.camera) {

            Ray r = ctx.scene.camera->get_ray(u, v);

            const bool edit_mode_locked =
                mesh_overlay_settings.enabled &&
                mesh_overlay_settings.edit_mode &&
                ctx.selection.mesh_element_mode != MeshElementSelectMode::Object &&
                !active_mesh_edit_object_name.empty();

            const bool sculpt_mode_locked =
                sculpt_mode_state.enabled &&
                mesh_workspace_mode == MeshWorkspaceMode::Sculpt &&
                mesh_overlay_settings.edit_mode &&
                (terrain_sculpt_proxy_active || !sculpt_mode_state.active_target_name.empty());

            if (edit_mode_locked) {
                if (ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
                    return;
                }

                handleMeshElementSelection(ctx, ImVec2(static_cast<float>(x), static_cast<float>(y)));
                return;
            }

            if (sculpt_mode_locked) {
                if (ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
                    return;
                }
                return;
            }

            // Check for Light Selection first (Bounding Sphere Intersection)
            std::shared_ptr<Light> closest_light = nullptr;
            float closest_t = 1e9f;
            int closest_light_index = -1;

            for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
                auto& light = ctx.scene.lights[i];
                if (!light) continue;

                // Proxy Sphere at light position - smaller radius for precise selection
                Vec3 oc = r.origin - light->position;
                float radius = 0.2f;  // Reduced from 0.5 to not block nearby objects
                float a = r.direction.dot(r.direction);
                float half_b = oc.dot(r.direction);
                float c = oc.dot(oc) - radius * radius;
                float discriminant = half_b * half_b - a * c;

                if (discriminant > 0) {
                    float root = sqrt(discriminant);
                    float temp = (-half_b - root) / a;
                    if (temp < closest_t && temp > 0.001f) {
                        closest_t = temp;
                        closest_light = light;
                        closest_light_index = (int)i;
                    }
                    temp = (-half_b + root) / a;
                    if (temp < closest_t && temp > 0.001f) {
                        closest_t = temp;
                        closest_light = light;
                        closest_light_index = (int)i;
                    }
                }
            }

            // Check for Camera Selection (non-active cameras only)
            std::shared_ptr<Camera> closest_camera = nullptr;
            float closest_camera_t = closest_t;  // Must be closer than light

            for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
                auto& cam = ctx.scene.cameras[i];
                if (!cam || cam == ctx.scene.camera) continue;

                Vec3 oc = r.origin - cam->lookfrom;
                float radius = 0.6f;
                float a = r.direction.dot(r.direction);
                float half_b = oc.dot(r.direction);
                float c = oc.dot(oc) - radius * radius;
                float discriminant = half_b * half_b - a * c;

                if (discriminant > 0) {
                    float root = sqrt(discriminant);
                    float temp = (-half_b - root) / a;
                    if (temp < closest_camera_t && temp > 0.001f) {
                        closest_camera_t = temp;
                        closest_camera = cam;
                    }
                }
            }

            // Check for Force Field Selection (Bounding Sphere Intersection for icons)
            std::shared_ptr<Physics::ForceField> closest_force_field = nullptr;
            float closest_force_field_t = closest_camera_t; // Must be closer than lights/cameras

            for (size_t i = 0; i < ctx.scene.force_field_manager.force_fields.size(); ++i) {
                auto& field = ctx.scene.force_field_manager.force_fields[i];
                if (!field || !field->visible) continue;

                Vec3 oc = r.origin - field->position;
                float radius = 0.4f; // Interface icon size
                float a = r.direction.dot(r.direction);
                float half_b = oc.dot(r.direction);
                float c = oc.dot(oc) - radius * radius;
                float discriminant = half_b * half_b - a * c;

                if (discriminant > 0) {
                    float root = sqrt(discriminant);
                    float temp = (-half_b - root) / a;
                    if (temp < closest_force_field_t && temp > 0.001f) {
                        closest_force_field_t = temp;
                        closest_force_field = field;
                    }
                }
            }

            int closest_domain_system_index = -1;
            int closest_domain_index = -1;
            float closest_domain_t = closest_force_field_t;

            auto isNearProjectedDomainEdge = [&](const Vec3& in_min, const Vec3& in_max) -> bool {
                const Vec3 mn = Vec3::min(in_min, in_max);
                const Vec3 mx = Vec3::max(in_min, in_max);
                const Vec3 corners[8] = {
                    Vec3(mn.x, mn.y, mn.z), Vec3(mx.x, mn.y, mn.z),
                    Vec3(mx.x, mx.y, mn.z), Vec3(mn.x, mx.y, mn.z),
                    Vec3(mn.x, mn.y, mx.z), Vec3(mx.x, mn.y, mx.z),
                    Vec3(mx.x, mx.y, mx.z), Vec3(mn.x, mx.y, mx.z)
                };
                const int edges[12][2] = {
                    {0, 1}, {1, 2}, {2, 3}, {3, 0},
                    {4, 5}, {5, 6}, {6, 7}, {7, 4},
                    {0, 4}, {1, 5}, {2, 6}, {3, 7}
                };

                const ImVec2 mouse_pos(static_cast<float>(x), static_cast<float>(y));
                const ImVec2 display_size(win_w, win_h);
                auto distanceToSegment = [](const ImVec2& p, const ImVec2& a, const ImVec2& b) {
                    const float abx = b.x - a.x;
                    const float aby = b.y - a.y;
                    const float apx = p.x - a.x;
                    const float apy = p.y - a.y;
                    const float ab_len_sq = abx * abx + aby * aby;
                    const float t = ab_len_sq > 1e-5f
                        ? std::clamp((apx * abx + apy * aby) / ab_len_sq, 0.0f, 1.0f)
                        : 0.0f;
                    const float cx = a.x + abx * t;
                    const float cy = a.y + aby * t;
                    const float dx = p.x - cx;
                    const float dy = p.y - cy;
                    return std::sqrt(dx * dx + dy * dy);
                };

                float closest_screen_distance = 1e9f;
                for (const auto& edge : edges) {
                    ImVec2 a, b;
                    if (!projectSelectionPointToScreen(*ctx.scene.camera, display_size, corners[edge[0]], a) ||
                        !projectSelectionPointToScreen(*ctx.scene.camera, display_size, corners[edge[1]], b)) {
                        continue;
                    }
                    closest_screen_distance = std::min(closest_screen_distance, distanceToSegment(mouse_pos, a, b));
                }
                return closest_screen_distance <= 10.0f;
            };

            auto intersectDomainAABB = [&](const Vec3& in_min, const Vec3& in_max, float& out_t) -> bool {
                const Vec3 mn = Vec3::min(in_min, in_max);
                const Vec3 mx = Vec3::max(in_min, in_max);
                float tmin = 0.001f;
                float tmax = 1e9f;

                auto testAxis = [&](float origin, float direction, float min_bound, float max_bound) -> bool {
                    if (std::abs(direction) < 1e-8f) {
                        return origin >= min_bound && origin <= max_bound;
                    }
                    float t1 = (min_bound - origin) / direction;
                    float t2 = (max_bound - origin) / direction;
                    if (t1 > t2) std::swap(t1, t2);
                    tmin = std::max(tmin, t1);
                    tmax = std::min(tmax, t2);
                    return tmin <= tmax;
                };

                if (!testAxis(r.origin.x, r.direction.x, mn.x, mx.x)) return false;
                if (!testAxis(r.origin.y, r.direction.y, mn.y, mx.y)) return false;
                if (!testAxis(r.origin.z, r.direction.z, mn.z, mx.z)) return false;
                out_t = tmin;
                return out_t >= 0.001f;
            };

            for (int system_i = 0; system_i < static_cast<int>(ctx.scene.particle_systems.size()); ++system_i) {
                auto& system = ctx.scene.particle_systems[static_cast<std::size_t>(system_i)];
                if (!system.visible || !system.runtime) continue;

                auto& domains = system.runtime->gridDomains();
                for (int domain_i = 0; domain_i < static_cast<int>(domains.size()); ++domain_i) {
                    auto& domain = domains[static_cast<std::size_t>(domain_i)];
                    if (!domain.enabled) continue;

                    Vec3 min_bound = domain.bounds_min;
                    Vec3 max_bound = domain.bounds_max;
                    if (domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::ObjectBounds &&
                        !ctx.scene.resolveObjectBoundsForSimulation(domain.source_name, min_bound, max_bound)) {
                        continue;
                    }
                    const Vec3 pad(std::max(0.0f, domain.padding));
                    float domain_t = 0.0f;
                    const Vec3 pick_min = min_bound - pad;
                    const Vec3 pick_max = max_bound + pad;
                    if (isNearProjectedDomainEdge(pick_min, pick_max) &&
                        intersectDomainAABB(pick_min, pick_max, domain_t) &&
                        domain_t < closest_domain_t) {
                        closest_domain_t = domain_t;
                        closest_domain_system_index = system_i;
                        closest_domain_index = domain_i;
                    }
                }
            }

            // ===========================================================================
            // SMART SELECTION: GPU picking (O(1)) with CPU fallback
            // GPU mode: Try pick buffer first, fall back to CPU linear scan
            // CPU mode: Linear scan through mesh_cache with updated vertices
            // ===========================================================================

            HitRecord rec;
            bool hit = false;
            float closest_so_far = 1e9f;
            HitRecord temp_rec;

            // GPU PICKING PATH: Use pick buffer for O(1) object selection
            // Skip GPU pick if rebuild is pending (pick buffer is stale after object delete/add)
            extern bool g_optix_rebuild_pending;
            extern bool g_vulkan_rebuild_pending;
            bool gpu_pick_success = false;
            std::string gpu_picked_name;
            
            bool use_gpu = ctx.render_settings.use_optix;
            bool has_ptr = (ctx.backend_ptr != nullptr);
            bool rebuild_pending = g_optix_rebuild_pending || g_vulkan_rebuild_pending;
            
            // Temporary safety fallback: GPU pick name lookup occasionally returns an unsafe path
            // into the selection cache. Keep selection stable by forcing CPU/BVH picking here.
            if (false && use_gpu && has_ptr && !rebuild_pending) {
                // Pass viewport dimensions for coordinate scaling
                int vp_w = static_cast<int>(win_w);
                int vp_h = static_cast<int>(win_h);
                int object_id = ctx.backend_ptr->getPickedObjectId(x, y, vp_w, vp_h);
                if (object_id >= 0) {
                    gpu_picked_name = ctx.backend_ptr->getPickedObjectName(x, y, vp_w, vp_h);
                    // Only mark as success if name found AND exists in mesh_cache
                    if (!gpu_picked_name.empty() && mesh_cache.find(gpu_picked_name) != mesh_cache.end()) {
                        // [FIX] Ignore ForceField visualization meshes
                        if (gpu_picked_name.find("ForceField") == std::string::npos && 
                            gpu_picked_name.find("Force Field") == std::string::npos) {
                            gpu_pick_success = true;
                        }
                       // SCENE_LOG_INFO("GPU Pick: " + gpu_picked_name);
                    }
                }
            }

            // =======================================================================
            // CPU BVH PICKING: Faster fallback for large scenes (e.g. 1.2M triangles)
            // =======================================================================
            if (!gpu_pick_success) {
                // Use BVH in ALL viewport modes. After the vertex sync above,
                // Triangle::hit() uses world-space positions so the BVH's internal
                // AABB pruning may be slightly stale (local-space) but can still
                // return valid hits. We try BVH first for O(log N) performance,
                // then fall back to linear scan if BVH misses or is unavailable.
                if (ctx.scene.bvh && !pending_timeline_selection_sync) {
                    if (ctx.scene.bvh->hit(r, 0.001f, closest_so_far, temp_rec)) {
                        hit = true;
                        closest_so_far = temp_rec.t;
                        rec = temp_rec;
                    }
                }
                
                // Fallback to Linear Scan if BVH missed or missing.
                // Linear scan always works because we synced vertices above.
                if (!hit && (pending_timeline_selection_sync || ctx.scene.world.objects.size() < 1000 || !ctx.scene.bvh || interactive_selection_fallback)) {
                    for (const auto& obj : ctx.scene.world.objects) {
                        if (!obj) continue;
                        if (obj->hit(r, 0.001f, closest_so_far, temp_rec)) {
                            // Filter ForceField helper meshes by resolved hit triangle name.
                            if (temp_rec.triangle) {
                                const std::string& name = temp_rec.triangle->getNodeName();
                                if (name.find("ForceField") != std::string::npos ||
                                    name.find("Force Field") != std::string::npos) {
                                    continue;
                                }
                            }
                            hit = true;
                            closest_so_far = temp_rec.t;
                            rec = temp_rec;
                        }
                    }
                }
            }
            
            // Check Gas Volumes (not in main BVH, separate list)
            for (const auto& gas : ctx.scene.gas_volumes) {
                if (gas->hit(r, 0.001f, closest_so_far, temp_rec)) {
                    hit = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            // Check VDB Volumes (not in main BVH, separate list)
            for (const auto& vdb : ctx.scene.vdb_volumes) {
                if (vdb->hit(r, 0.001f, closest_so_far, temp_rec)) {
                    hit = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            // ===========================================================
            // APPLY PICK FOCUS (Using Found Hits)
            // ===========================================================
            if (is_picking_focus) {
                float min_dist = 1e9f;
                bool found_hit = false;

                // Check Object Hit
                if (hit && rec.t < min_dist) {
                    min_dist = rec.t;
                    found_hit = true;
                }
                // Check Light Hit
                if (closest_light && closest_t < min_dist) {
                    min_dist = closest_t;
                    found_hit = true;
                }
                // Check Camera Hit
                if (closest_camera && closest_camera_t < min_dist) {
                    min_dist = closest_camera_t;
                    found_hit = true;
                }
                // Check Force Field Hit
                if (closest_force_field && closest_force_field_t < min_dist) {
                    min_dist = closest_force_field_t;
                    found_hit = true;
                }
                // Check Simulation Domain Hit
                if (closest_domain_index >= 0 && closest_domain_t < min_dist) {
                    min_dist = closest_domain_t;
                    found_hit = true;
                }

                if (found_hit) {
                    min_dist = std::max(min_dist, 0.05f);
                    ctx.scene.camera->focus_dist = min_dist;
                    ctx.scene.camera->update_camera_vectors();
                    if (ctx.backend_ptr) {
                        ctx.backend_ptr->syncCamera(*ctx.scene.camera);
                        ctx.backend_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                   // SCENE_LOG_INFO(std::string("Pick Focus set to: ") + std::to_string(min_dist) + "m");
                }

                is_picking_focus = false;
                ctx.start_render = true;
                return;
            }

            // Priority Selection: ForceField > Camera > Light > Object (by distance)
            if (closest_force_field && closest_force_field_t < closest_so_far && closest_force_field_t < closest_camera_t && closest_force_field_t < closest_t) {
                if (ctrl_held) {
                    SelectableItem item;
                    item.type = SelectableType::ForceField;
                    item.force_field = closest_force_field;
                    item.name = closest_force_field->name;
                    if (ctx.selection.isSelected(item)) ctx.selection.removeFromSelection(item);
                    else ctx.selection.addToSelection(item);
                } else {
                    ctx.selection.selectForceField(closest_force_field);
                }
                return;
            }

            if (closest_domain_index >= 0 &&
                closest_domain_t < closest_so_far &&
                closest_domain_t < closest_camera_t &&
                closest_domain_t < closest_t &&
                closest_domain_t < closest_force_field_t) {
                auto& system = ctx.scene.particle_systems[static_cast<std::size_t>(closest_domain_system_index)];
                auto& domain = system.runtime->gridDomains()[static_cast<std::size_t>(closest_domain_index)];
                const Vec3 mn = Vec3::min(domain.bounds_min, domain.bounds_max);
                const Vec3 mx = Vec3::max(domain.bounds_min, domain.bounds_max);

                SelectableItem item;
                item.type = SelectableType::SimulationDomain;
                item.particle_system_index = closest_domain_system_index;
                item.simulation_domain_index = closest_domain_index;
                item.name = domain.name;
                item.position = (mn + mx) * 0.5f;
                item.scale = mx - mn;

                ctx.scene.setActiveParticleSystemObject(static_cast<std::size_t>(closest_domain_system_index));
                if (ctrl_held) {
                    if (ctx.selection.isSelected(item)) ctx.selection.removeFromSelection(item);
                    else ctx.selection.addToSelection(item);
                } else {
                    ctx.selection.selectSimulationDomain(closest_domain_system_index, closest_domain_index, domain.name);
                    ctx.selection.selected.position = item.position;
                    ctx.selection.selected.scale = item.scale;
                }
                show_forcefield_tab = true;
                tab_to_focus = "Simulation";
                return;
            }

            if (closest_camera && closest_camera_t < closest_so_far && closest_camera_t < closest_t) {
                // Camera is the closest thing clicked
                if (ctrl_held) {
                    SelectableItem item;
                    item.type = SelectableType::Camera;
                    item.camera = closest_camera;
                    item.name = closest_camera->nodeName.empty() ? "Camera" : closest_camera->nodeName;

                    if (ctx.selection.isSelected(item)) {
                        ctx.selection.removeFromSelection(item);
                    } else {
                        ctx.selection.addToSelection(item);
                    }
                } else {
                    ctx.selection.selectCamera(closest_camera);
                }
                return; // Camera selected, done
            }

            // ===========================================================================
            // GPU PICK SUCCESS PATH: Direct mesh selection from pick buffer result
            // ===========================================================================
            if (gpu_pick_success && !gpu_picked_name.empty()) {
                // Find the object in mesh_cache using GPU-provided name
                auto cache_it = mesh_cache.find(gpu_picked_name);
                if (cache_it != mesh_cache.end() && !cache_it->second.empty()) {
                    auto& first_tri = cache_it->second[0].second;
                    int index = cache_it->second[0].first;
                    
                    if (ctrl_held) {
                        SelectableItem item;
                        item.type = SelectableType::Object;
                        item.object = first_tri;
                        item.object_index = index;
                        item.name = first_tri->nodeName;
                        
                        if (ctx.selection.isSelected(item)) {
                            ctx.selection.removeFromSelection(item);
                        } else {
                            ctx.selection.addToSelection(item);
                        }
                    } else {
                        ctx.selection.selectObject(first_tri, index, first_tri->nodeName);
                        
                        // TERRAIN CONNECTION: Check if this is a terrain chunk
                        std::string tName = first_tri->nodeName;
                        if (tName.find("Terrain_") == 0) {
                            size_t chunkPos = tName.find("_Chunk");
                            if (chunkPos != std::string::npos) {
                                tName = tName.substr(0, chunkPos);
                            }
                            auto terrain = TerrainManager::getInstance().getTerrainByName(tName);
                            if (terrain) {
                                terrain_brush.active_terrain_id = terrain->id;
                                show_terrain_tab = true;
                                //SCENE_LOG_INFO("Terrain selected via GPU pick: " + tName);
                            }
                        }
                    }
                    return; // GPU pick selection done
                }
                // Note: If gpu_pick_success is true but cache lookup failed, we already
                // handled that above by setting gpu_pick_success = false before CPU scan
            }

            if (hit && (rec.t < closest_t)) {

                // --- VDB VOLUME SELECTION ---
                if (rec.vdb_volume) {
                    std::shared_ptr<VDBVolume> found_vdb = nullptr;
                    int index = -1;

                    // Find shared_ptr for this VDB
                    for (size_t i = 0; i < ctx.scene.vdb_volumes.size(); ++i) {
                        if (ctx.scene.vdb_volumes[i].get() == rec.vdb_volume) {
                            found_vdb = ctx.scene.vdb_volumes[i];
                            index = (int)i;
                            break;
                        }
                    }

                    if (found_vdb) {
                        if (ctrl_held) {
                            // Multi-selection
                            SelectableItem item;
                            item.type = SelectableType::VDBVolume;
                            item.vdb_volume = found_vdb;
                            item.vdb_index = index;
                            item.name = found_vdb->name;

                            if (ctx.selection.isSelected(item)) {
                                ctx.selection.removeFromSelection(item);
                            }
                            else {
                                ctx.selection.addToSelection(item);
                            }
                        }
                        else {
                            // Single selection
                            ctx.selection.selectVDBVolume(found_vdb, index, found_vdb->name);
                           // SCENE_LOG_INFO("Selected VDB Volume via viewport: " + found_vdb->name);
                        }
                    }
                }
                
                // --- GAS VOLUME SELECTION ---
                else if (rec.gas_volume) {
                    std::shared_ptr<GasVolume> found_gas = nullptr;
                    int index = -1;

                    // Find shared_ptr for this Gas Volume
                    for (size_t i = 0; i < ctx.scene.gas_volumes.size(); ++i) {
                        if (ctx.scene.gas_volumes[i].get() == rec.gas_volume) {
                            found_gas = ctx.scene.gas_volumes[i];
                            index = (int)i;
                            break;
                        }
                    }

                    if (found_gas) {
                        if (ctrl_held) {
                            // Multi-selection
                            SelectableItem item;
                            item.type = SelectableType::GasVolume;
                            item.gas_volume = found_gas;
                            item.vdb_index = index; // Reuse vdb_index for gas index
                            item.name = found_gas->name;

                            if (ctx.selection.isSelected(item)) {
                                ctx.selection.removeFromSelection(item);
                            }
                            else {
                                ctx.selection.addToSelection(item);
                            }
                        }
                        else {
                            // Single selection
                            ctx.selection.selectGasVolume(found_gas, index, found_gas->name);
                           // SCENE_LOG_INFO("Selected Gas Volume via viewport: " + found_gas->name);
                        }
                    }
                }

                // --- TRIANGLE SELECTION ---
                else if (rec.triangle) {
                    std::shared_ptr<Triangle> found_tri = nullptr;
                    int index = -1;

                    if (!mesh_cache_valid) {
                        rebuildMeshCache(ctx.scene.world.objects);
                    }

                    // Try fast O(1) lookup first with const-correctness
                    auto it = tri_to_index.find(rec.triangle);
                    if (it != tri_to_index.end()) {
                        index = it->second;
                        if (index >= 0 && index < (int)ctx.scene.world.objects.size()) {
                            found_tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[index]);
                            if (!found_tri) {
                                for (const auto& kv : mesh_cache) {
                                    for (const auto& pair : kv.second) {
                                        if (pair.second && pair.second.get() == rec.triangle) {
                                            found_tri = pair.second;
                                            break;
                                        }
                                    }
                                    if (found_tri) break;
                                }
                            }
                        }
                    }

                    if (found_tri) {
                        if (ctrl_held) {
                            SelectableItem item;
                            item.type = SelectableType::Object;
                            item.object = found_tri;
                            item.object_index = index;
                            item.name = found_tri->nodeName;

                            if (ctx.selection.isSelected(item)) {
                                ctx.selection.removeFromSelection(item);
                            } else {
                                ctx.selection.addToSelection(item);
                            }
                        } else {
                            ctx.selection.selectObject(found_tri, index, found_tri->nodeName);
                           // SCENE_LOG_INFO("Selected via CPU Viewport: " + found_tri->nodeName);

                            // TERRAIN CONNECTION: Check if this is a terrain chunk
                            std::string tName = found_tri->nodeName;
                            if (tName.find("Terrain_") == 0) {
                                size_t chunkPos = tName.find("_Chunk");
                                if (chunkPos != std::string::npos) {
                                    tName = tName.substr(0, chunkPos);
                                }
                                auto terrain = TerrainManager::getInstance().getTerrainByName(tName);
                                if (terrain) {
                                    terrain_brush.active_terrain_id = terrain->id;
                                    show_terrain_tab = true;
                                }
                            }
                        }
                    } else {
                        SCENE_LOG_INFO("Selection: BVH hit stale triangle - falling back to linear scan");
                        if (!mesh_cache_valid) {
                            rebuildMeshCache(ctx.scene.world.objects);
                        }
                        HitRecord retry_rec;
                        float retry_closest = 1e9f;
                        bool retry_hit = false;
                        for (const auto& obj : ctx.scene.world.objects) {
                            if (!obj) continue;
                            if (obj->hit(r, 0.001f, retry_closest, retry_rec)) {
                                if (retry_rec.triangle) {
                                    const std::string& retry_name = retry_rec.triangle->getNodeName();
                                    if (retry_name.find("ForceField") != std::string::npos ||
                                        retry_name.find("Force Field") != std::string::npos) {
                                        continue;
                                    }
                                }
                                retry_hit = true;
                                retry_closest = retry_rec.t;
                            }
                        }
                        if (retry_hit && retry_rec.triangle) {
                            auto it2 = tri_to_index.find(retry_rec.triangle);
                            if (it2 != tri_to_index.end()) {
                                int idx2 = it2->second;
                                if (idx2 >= 0 && idx2 < (int)ctx.scene.world.objects.size()) {
                                    auto retry_tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[idx2]);
                                    if (!retry_tri) {
                                        for (const auto& kv : mesh_cache) {
                                            for (const auto& pair : kv.second) {
                                                if (pair.second && pair.second.get() == retry_rec.triangle) {
                                                    retry_tri = pair.second;
                                                    break;
                                                }
                                            }
                                            if (retry_tri) break;
                                        }
                                    }
                                    if (retry_tri) {
                                        ctx.selection.selectObject(retry_tri, idx2, retry_tri->nodeName);
                                       // SCENE_LOG_INFO("Selected via linear fallback: " + retry_tri->nodeName);
                                    }
                                }
                            }
                        }
                    }
                }
                else if (closest_camera && closest_camera_t < closest_t) {
                    // Camera is closer than light
                    if (ctrl_held) {
                        SelectableItem item;
                        item.type = SelectableType::Camera;
                        item.camera = closest_camera;
                        item.name = "Camera";

                        if (ctx.selection.isSelected(item)) {
                            ctx.selection.removeFromSelection(item);
                        }
                        else {
                            ctx.selection.addToSelection(item);
                        }
                    }
                    else {
                        ctx.selection.selectCamera(closest_camera);
                    }
                }
                else if (closest_light) {
                    if (ctrl_held) {
                        SelectableItem item;
                        item.type = SelectableType::Light;
                        item.light = closest_light;
                        item.light_index = closest_light_index;
                        item.name = "Light";

                        if (ctx.selection.isSelected(item)) {
                            ctx.selection.removeFromSelection(item);
                        }
                        else {
                            ctx.selection.addToSelection(item);
                        }
                    }
                    else {
                        ctx.selection.selectLight(closest_light);
                    }
                }
                else {
                    // Clicked on empty space - clear selection only if Ctrl is not held
                    if (!ctrl_held) {
                        ctx.selection.clearSelection();
                    }
                }
            }
        }
    }
}

// ============================================================================
// Delete Operation (Shared by Menu and Key Shortcut)
// ============================================================================
// OPTIMIZED VERSION - O(n) instead of O(n�)
void SceneUI::triggerDelete(UIContext& ctx) {
    if (!ctx.selection.hasSelection()) return;

    // Collect all items to delete (supports multi-selection)
    std::vector<SelectableItem> items_to_delete = ctx.selection.multi_selection;

    // OPTIMIZATION: Collect ALL hittable pointers to delete into a single set for O(1) lookup
    // Using Hittable* (base class) for type safety
    std::unordered_set<Hittable*> objects_to_delete;
    std::vector<std::string> deleted_names;
    std::vector<std::pair<std::string, std::vector<std::shared_ptr<Triangle>>>> undo_data;

    // ===========================================================================
    // VDB VOLUME DELETION
    // ===========================================================================
    bool vdb_deleted = false;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::VDBVolume && item.vdb_volume) {
            // Unload GPU resources
            item.vdb_volume->unload();

            // Remove from Scene VDB list
            if (ctx.scene.removeVDBVolume(item.vdb_volume)) {
                vdb_deleted = true;
                SCENE_LOG_INFO("Deleted VDB Volume: " + item.vdb_volume->name);
            }

            // Also ensure it is removed from world.objects (Hittable list)
            // This is handled generically below if we add it to objects_to_delete, 
            // OR we can explicitly remove it here.
            // Since VDBVolume is a Hittable, forcing it into objects_to_delete is safest.
            objects_to_delete.insert(item.vdb_volume.get());
        }
    }

    if (vdb_deleted) {
        ctx.renderer.resetCPUAccumulation();
    }

    // ===========================================================================
    // GAS VOLUME DELETION
    // ===========================================================================
    bool gas_deleted = false;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::GasVolume && item.gas_volume) {
            // Unload GPU resources
            item.gas_volume->freeGPUResources();

            // Remove from Scene Gas list
            if (ctx.scene.removeGasVolume(item.gas_volume)) {
                gas_deleted = true;
                SCENE_LOG_INFO("Deleted Gas Volume: " + item.gas_volume->name);
            }

            // GasVolume is a Hittable, must be removed from world.objects
            objects_to_delete.insert(item.gas_volume.get());
        }
    }

    if (gas_deleted) {
        ctx.renderer.resetCPUAccumulation();
    }

    // ===========================================================================
    // FORCE FIELD DELETION
    // ===========================================================================
    int ff_deleted_count = 0;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::ForceField && item.force_field) {
            if (ctx.scene.removeForceField(item.force_field)) {
                ff_deleted_count++;
                SCENE_LOG_INFO("Deleted Force Field: " + item.force_field->name);
            }
        }
    }

    if (ff_deleted_count > 0) {
        // Reset accumulation if needed (force fields might affect visual simulation)
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
    }

    std::vector<int> particle_system_indices_to_delete;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::ParticleSystem && item.particle_system_index >= 0) {
            particle_system_indices_to_delete.push_back(item.particle_system_index);
        }
    }
    std::sort(particle_system_indices_to_delete.begin(), particle_system_indices_to_delete.end(), std::greater<int>());
    particle_system_indices_to_delete.erase(
        std::unique(particle_system_indices_to_delete.begin(), particle_system_indices_to_delete.end()),
        particle_system_indices_to_delete.end());

    int particle_system_deleted_count = 0;
    for (int index : particle_system_indices_to_delete) {
        if (ctx.scene.removeParticleSystemObject(static_cast<std::size_t>(index))) {
            ++particle_system_deleted_count;
            SCENE_LOG_INFO("Deleted Particle System #" + std::to_string(index));
        }
    }

    if (particle_system_deleted_count > 0) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
    }

    std::vector<std::pair<int, int>> simulation_domains_to_delete;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::SimulationDomain &&
            item.particle_system_index >= 0 &&
            item.simulation_domain_index >= 0) {
            simulation_domains_to_delete.emplace_back(item.particle_system_index, item.simulation_domain_index);
        }
    }
    std::sort(simulation_domains_to_delete.begin(), simulation_domains_to_delete.end(),
        [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second > b.second;
        });
    simulation_domains_to_delete.erase(
        std::unique(simulation_domains_to_delete.begin(), simulation_domains_to_delete.end()),
        simulation_domains_to_delete.end());

    int simulation_domain_deleted_count = 0;
    for (const auto& [system_index, domain_index] : simulation_domains_to_delete) {
        if (system_index < 0 || system_index >= static_cast<int>(ctx.scene.particle_systems.size())) {
            continue;
        }
        auto& system = ctx.scene.particle_systems[static_cast<std::size_t>(system_index)];
        if (!system.runtime ||
            domain_index < 0 ||
            domain_index >= static_cast<int>(system.runtime->gridDomains().size())) {
            continue;
        }
        const std::string domain_name = system.runtime->gridDomains()[static_cast<std::size_t>(domain_index)].name;
        system.runtime->removeGridDomain(static_cast<std::size_t>(domain_index));
        ++simulation_domain_deleted_count;
        SCENE_LOG_INFO("Deleted Simulation Domain: " + domain_name);
    }

    if (simulation_domain_deleted_count > 0) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
    }

    // Build mesh cache once if needed
    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

    // First pass: Collect all triangles to delete
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::Object && item.object) {
            std::string deleted_name = item.name;

            auto cache_it = mesh_cache.find(deleted_name);
            if (cache_it != mesh_cache.end()) {
                std::vector<std::shared_ptr<Triangle>> tris_for_undo;
                for (auto& pair : cache_it->second) {
                    pair.second->visible = false;
                    tris_for_undo.push_back(pair.second);
                }

                if (!tris_for_undo.empty()) {
                    ctx.scene.markObjectPendingDelete(deleted_name);
                    deleted_names.push_back(deleted_name);
                    undo_data.push_back({ deleted_name, std::move(tris_for_undo) });
                }
            }
            setSelectionSceneNodeLocalVisibility(ctx, deleted_name, false);
        }
    }

    // OPTIMIZATION: Single remove_if pass for ALL objects - O(n) instead of O(n�)
    // CRITICAL: Using raw pointer .get() instead of dynamic_pointer_cast for massive speedup
    // dynamic_pointer_cast does RTTI check on every call = very slow on 4M objects
    // We already know exact pointers from mesh_cache, so just compare raw pointers
    if (!objects_to_delete.empty()) {
        auto& objs = ctx.scene.world.objects;
        objs.erase(
            std::remove_if(objs.begin(), objs.end(), [&](const std::shared_ptr<Hittable>& h) {
                // Fast O(1) lookup - no RTTI, just pointer comparison with base class
                return objects_to_delete.count(h.get()) > 0;
                }),
            objs.end()
        );
    }

    if (vdb_deleted || gas_deleted) {
        ctx.selection.clearSelection();
        const bool activeVulkanRT =
            (g_backend && dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr) ||
            (ctx.backend_ptr && dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr) != nullptr);

        // Now that world.objects no longer contains the deleted volume, sync the
        // volume SSBO and force Vulkan TLAS rebuild. Syncing before the erase left
        // the stale AABB instance alive until a backend switch rebuilt everything.
        if (ctx.backend_ptr) {
            syncVDBVolumesToGPU(ctx);
            if (gas_deleted) {
                ctx.renderer.updateBackendGasVolumes(ctx.scene);
            }
            ctx.backend_ptr->resetAccumulation();
        }
        if (activeVulkanRT) {
            extern bool g_vulkan_rebuild_pending;
            g_vulkan_rebuild_pending = true;
        }
        ctx.renderer.resetCPUAccumulation();
        ctx.start_render = true;
        g_ProjectManager.markModified();
    }

    // Track deletions in ProjectManager (batch update)
    auto& proj_data = g_ProjectManager.getProjectData();
    for (const auto& deleted_name : deleted_names) {
        // Check Imported Models
        bool found = false;
        for (auto& model : proj_data.imported_models) {
            std::string prefix = std::to_string(model.id) + "_";
            if (deleted_name.find(prefix) == 0) {
                model.deleted_objects.push_back(deleted_name);
                found = true;
                break;
            }
            for (const auto& obj_inst : model.objects) {
                if (obj_inst.node_name == deleted_name) {
                    model.deleted_objects.push_back(deleted_name);
                    found = true;
                    break;
                }
            }
            if (found) break;
        }

        // Check Procedural Objects
        if (!found) {
            auto it = std::remove_if(proj_data.procedural_objects.begin(), proj_data.procedural_objects.end(),
                [&](const ProceduralObjectData& p) { return p.display_name == deleted_name; });
            proj_data.procedural_objects.erase(it, proj_data.procedural_objects.end());
        }

        // Check Water Surfaces - remove from WaterManager if name matches
        auto& water_surfaces = WaterManager::getInstance().getWaterSurfaces();
        for (auto& surf : water_surfaces) {
            if (surf.name == deleted_name) {
                // Don't call removeWaterSurface here - triangles already removed above
                // Just mark for removal from water_surfaces vector
                surf.id = -1; // Mark for removal
                break;
            }
        }
    }

    // Remove marked water surfaces
    auto& water_surfaces = WaterManager::getInstance().getWaterSurfaces();
    water_surfaces.erase(
        std::remove_if(water_surfaces.begin(), water_surfaces.end(),
            [](const WaterSurface& s) { return s.id == -1; }),
        water_surfaces.end()
    );

    // Record undo commands
    for (auto& [name, tris] : undo_data) {
        history.record(std::make_unique<DeleteObjectCommand>(name, tris));
    }

    // Handle light deletions
    int deleted_lights = 0;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::Light && item.light) {
            auto& lights = ctx.scene.lights;
            auto it = std::find(lights.begin(), lights.end(), item.light);
            if (it != lights.end()) {
                history.record(std::make_unique<DeleteLightCommand>(item.light));
                lights.erase(it);
                deleted_lights++;
            }
        }
    }

    // ===========================================================================
    // Handle camera deletions
    // ===========================================================================
    int deleted_cameras = 0;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::Camera && item.camera) {
            // Safety checks are in removeCamera method
            if (ctx.scene.cameras.size() <= 1) {
                SCENE_LOG_WARN("Cannot delete the last camera in scene");
                continue;
            }

            // Check if this is the active camera
            if (item.camera == ctx.scene.camera) {
                SCENE_LOG_WARN("Cannot delete the active camera. Switch to another camera first.");
                continue;
            }

            // Remove camera
            if (ctx.scene.removeCamera(item.camera)) {
                deleted_cameras++;
                SCENE_LOG_INFO("Camera deleted successfully");
            }
        }
    }

    // Only rebuild once after all deletions are done
    int deleted_objects = static_cast<int>(deleted_names.size());
    if (deleted_objects > 0 || deleted_lights > 0 || deleted_cameras > 0 ||
        particle_system_deleted_count > 0 || simulation_domain_deleted_count > 0) {
        ctx.selection.clearSelection();
        g_ProjectManager.markModified();

        // Keep UI caches warm; object tombstones are hidden by name instead of forcing
        // a full mesh cache rebuild for large scenes.
        for (const auto& name : deleted_names) {
            auto countIt = cached_triangle_count_by_object.find(name);
            if (countIt != cached_triangle_count_by_object.end()) {
                if (cached_scene_triangle_count >= countIt->second) {
                    cached_scene_triangle_count -= countIt->second;
                }
                cached_triangle_count_by_object.erase(countIt);
            }

            mesh_cache.erase(name);
            bbox_cache.erase(name);
            material_slots_cache.erase(name);
        }
        mesh_ui_cache.erase(
            std::remove_if(mesh_ui_cache.begin(), mesh_ui_cache.end(),
                [&](const auto& entry) {
                    return std::find(deleted_names.begin(), deleted_names.end(), entry.first) != deleted_names.end();
                }),
            mesh_ui_cache.end());
        last_scene_obj_count = ctx.scene.world.objects.size();

        // Immediately rebuild the lightweight tri_to_index so viewport picking
        // keeps working with the stale BVH (no expensive full rebuild needed).
        rebuildTriToIndex(ctx.scene.world.objects);

        if (deleted_objects > 0) {
            extern bool g_cpu_sync_pending;
            for (const auto& name : deleted_names) {
                setSelectionObjectVisibility(ctx, name, false);
            }
            ctx.renderer.resetCPUAccumulation();
            if (selectionHasGpuRenderBackend(ctx)) {
                g_cpu_sync_pending = true;
            }
            ctx.start_render = true;
        }

        // Update lights if any were deleted
        if (deleted_lights > 0) {
            syncSelectionSceneState(ctx, true, false);
        }

        // Update GPU camera if cameras changed
        if (deleted_cameras > 0 && ctx.scene.camera) {
            syncSelectionSceneState(ctx, false, true);
        }

        // ===========================================================================
        // AUTO-CLEANUP TIMELINE TRACKS FOR DELETED ENTITIES
        // ===========================================================================
        // Remove timeline tracks for deleted objects/lights/cameras
        // This prevents orphan keyframes from cluttering the timeline
        for (const auto& deleted_name : deleted_names) {
            auto it = ctx.scene.timeline.tracks.find(deleted_name);
            if (it != ctx.scene.timeline.tracks.end()) {
                ctx.scene.timeline.tracks.erase(it);
            }
        }

        // Also clean up deleted lights and cameras from timeline
        for (const auto& item : items_to_delete) {
            if (item.type == SelectableType::Light && item.light) {
                auto it = ctx.scene.timeline.tracks.find(item.light->nodeName);
                if (it != ctx.scene.timeline.tracks.end()) {
                    ctx.scene.timeline.tracks.erase(it);
                }
            }
            else if (item.type == SelectableType::Camera && item.camera) {
                auto it = ctx.scene.timeline.tracks.find(item.camera->nodeName);
                if (it != ctx.scene.timeline.tracks.end()) {
                    ctx.scene.timeline.tracks.erase(it);
                }
            }
        }

        ctx.start_render = true;
    }
}

// ============================================================================
// MARQUEE (BOX) SELECTION
// ============================================================================
void SceneUI::drawMarqueeRect() {
    if (!is_marquee_selecting) return;

    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    // Normalize rectangle (handle dragging in any direction)
    float x1 = fminf(marquee_start.x, marquee_end.x);
    float y1 = fminf(marquee_start.y, marquee_end.y);
    float x2 = fmaxf(marquee_start.x, marquee_end.x);
    float y2 = fmaxf(marquee_start.y, marquee_end.y);

    // Draw filled rect with transparency
    draw_list->AddRectFilled(ImVec2(x1, y1), ImVec2(x2, y2), IM_COL32(100, 150, 255, 40));
    // Draw border
    draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), IM_COL32(100, 150, 255, 200), 0.0f, 0, 2.0f);
}

void SceneUI::triggerDuplicate(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    if (!sel.hasSelection()) return;

    struct PendingDuplicateCacheEntry {
        std::string source_name;
        std::string new_name;
        Matrix4x4 transform;
        std::vector<std::pair<int, std::shared_ptr<Triangle>>> cache_entries;
    };

    // Build a list of objects to duplicate
    // If multi-selection exists, use it. Otherwise use the single active selection.
    std::vector<SelectableItem> itemsToDuplicate;
    if (sel.multi_selection.size() > 0) {
        itemsToDuplicate = sel.multi_selection;
    } else {
        itemsToDuplicate.push_back(sel.selected);
    }

    std::vector<std::shared_ptr<Hittable>> allNewTriangles;
    std::vector<SelectableItem> newSelectionList;
    std::vector<PendingDuplicateCacheEntry> pendingCacheEntries;
    
    // Temporary map for name uniqueness check
    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
    
    // Perform duplication for each item
    bool anyDuplicated = false;
    std::unordered_set<std::string> alreadyDuplicated; // Track duplicated source names
    std::unordered_set<std::string> assignedNames;   // Track names assigned to clones
    
    for (const auto& item : itemsToDuplicate) {
        if (item.type == SelectableType::Object && item.object) {
            
            // Use item.name as primary, fallback to nodeName
            std::string targetName = item.name;
            if (targetName.empty()) {
                targetName = item.object->nodeName;
            }
            if (targetName.empty()) targetName = "Unnamed";
            
            // Skip if this object name was already duplicated
            if (alreadyDuplicated.count(targetName) > 0) continue;
            alreadyDuplicated.insert(targetName);

            // Unique name generation
            std::string baseName = targetName;
            size_t lastUnderscore = baseName.rfind('_');
            if (lastUnderscore != std::string::npos) {
                std::string suffix = baseName.substr(lastUnderscore + 1);
                if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
                    baseName = baseName.substr(0, lastUnderscore);
                }
            }
            int copyNum = 1;
            std::string newName;
            do { 
                newName = baseName + "_" + std::to_string(copyNum++); 
            } while (mesh_cache.count(newName) > 0 || assignedNames.count(newName) > 0);
            
            assignedNames.insert(newName);

            // Create Unique Transform
            std::shared_ptr<Transform> newTransform = std::make_shared<Transform>();
            if (Transform* th = item.object->getTransformPtr()) {
                *newTransform = *th;
            }

            // Duplicate Triangles - search by targetName
            std::shared_ptr<Triangle> firstNewTri = nullptr;
            auto it = mesh_cache.find(targetName);
            if (it != mesh_cache.end()) {
                const int baseIndex = static_cast<int>(ctx.scene.world.objects.size() + allNewTriangles.size());
                PendingDuplicateCacheEntry pendingCacheEntry;
                pendingCacheEntry.source_name = targetName;
                pendingCacheEntry.new_name = newName;
                pendingCacheEntry.cache_entries.reserve(it->second.size());

                int localIndex = 0;
                for (auto& pair : it->second) {
                    auto& oldTri = pair.second;
                    auto newTri = std::make_shared<Triangle>(*oldTri);
                    newTri->setTransformHandle(newTransform);
                    newTri->setNodeName(newName);
                    
                    allNewTriangles.push_back(newTri);
                    pendingCacheEntry.cache_entries.push_back({baseIndex + localIndex, newTri});
                    ++localIndex;
                    if (!firstNewTri) firstNewTri = newTri;
                }

                if (!pendingCacheEntry.cache_entries.empty()) {
                    if (firstNewTri) {
                        pendingCacheEntry.transform = firstNewTri->getTransformMatrix();
                    }
                    pendingCacheEntries.push_back(std::move(pendingCacheEntry));
                }
            }
            
            if (firstNewTri) {
                SelectableItem newItem;
                newItem.type = SelectableType::Object;
                newItem.object = firstNewTri;
                newItem.object_index = (int)ctx.scene.world.objects.size() + (int)allNewTriangles.size() - 1;
                newItem.name = newName;
                
                if (Transform* th = firstNewTri->getTransformPtr()) {
                    Matrix4x4 pivotMat = th->getPivotMatrix();
                    newItem.position = Vec3(pivotMat.m[0][3], pivotMat.m[1][3], pivotMat.m[2][3]);
                } else {
                    newItem.position = Vec3(0, 0, 0);
                }
                newItem.rotation = Vec3(0, 0, 0); // Decomposed on next update if needed
                newItem.scale = Vec3(1, 1, 1);
                
                newSelectionList.push_back(newItem);
                anyDuplicated = true;
            }
        }
    }
    
    // VDB Duplication
    for (const auto& item : itemsToDuplicate) {
        if (item.type == SelectableType::VDBVolume && item.vdb_volume) {
            auto oldVDB = item.vdb_volume;
            auto newVDB = std::make_shared<VDBVolume>();
            
            newVDB->loadVDB(oldVDB->getFilePath());
            VDBVolumeManager::getInstance().uploadToGPU(newVDB->getVDBVolumeID());
            newVDB->setTransform(oldVDB->getTransform());
            
            auto oldShader = oldVDB->getShader();
            if (oldShader) {
                auto newShader = std::make_shared<VolumeShader>(*oldShader);
                newShader->name = oldShader->name + " (Copy)";
                newVDB->setShader(newShader);
            }
            
            newVDB->name = oldVDB->name + "_Copy";
            ctx.scene.addVDBVolume(newVDB);
            ctx.scene.world.add(newVDB);
            
            SelectableItem newSel;
            newSel.type = SelectableType::VDBVolume;
            newSel.vdb_volume = newVDB;
            newSel.vdb_index = (int)ctx.scene.vdb_volumes.size() - 1;
            newSel.name = newVDB->name;
            newSel.position = newVDB->getPosition();
            newSel.rotation = newVDB->getRotation();
            newSel.scale = newVDB->getScale();
            
            newSelectionList.push_back(newSel);
            anyDuplicated = true;
        }
    }

    // Light Duplication
    for (const auto& item : itemsToDuplicate) {
        if (item.type == SelectableType::Light && item.light) {
            std::shared_ptr<Light> newLight = nullptr;
            auto l = item.light;
            
            if (auto pl = std::dynamic_pointer_cast<PointLight>(l)) newLight = std::make_shared<PointLight>(*pl);
            else if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(l)) newLight = std::make_shared<DirectionalLight>(*dl);
            else if (auto sl = std::dynamic_pointer_cast<SpotLight>(l)) newLight = std::make_shared<SpotLight>(*sl);
            else if (auto al = std::dynamic_pointer_cast<AreaLight>(l)) newLight = std::make_shared<AreaLight>(*al);

            if (newLight) {
                std::string newName = l->nodeName + "_Copy";
                newLight->nodeName = newName;
                
                ctx.scene.lights.push_back(newLight);
                history.record(std::make_unique<AddLightCommand>(newLight));
                
                SelectableItem newSel;
                newSel.type = SelectableType::Light;
                newSel.light = newLight;
                newSel.light_index = (int)ctx.scene.lights.size() - 1;
                newSel.name = newName;
                newSel.position = newLight->position;
                newSel.rotation = Vec3(0, 0, 0);
                newSel.scale = Vec3(1, 1, 1);
                
                newSelectionList.push_back(newSel);
                anyDuplicated = true;
            }
        }
    }

    if (anyDuplicated) {
        if (!allNewTriangles.empty()) {
            ctx.scene.world.objects.insert(ctx.scene.world.objects.end(), allNewTriangles.begin(), allNewTriangles.end());

            // Incrementally extend selection caches instead of rescanning the full scene.
            for (auto& pending : pendingCacheEntries) {
                mesh_cache[pending.new_name] = pending.cache_entries;
                mesh_ui_cache.push_back({pending.new_name, mesh_cache[pending.new_name]});

                auto bbox_it = bbox_cache.find(pending.source_name);
                if (bbox_it != bbox_cache.end()) {
                    bbox_cache[pending.new_name] = bbox_it->second;
                }

                auto mats_it = material_slots_cache.find(pending.source_name);
                if (mats_it != material_slots_cache.end()) {
                    material_slots_cache[pending.new_name] = mats_it->second;
                }
            }
        }
        
        sel.clearSelection();
        for (const auto& newItem : newSelectionList) {
            sel.addToSelection(newItem);
        }
        
        bool rasterCloneSucceeded = false;
        if (Backend::IViewportBackend* viewportBackend = getSelectionViewportBackend(ctx)) {
            rasterCloneSucceeded = !pendingCacheEntries.empty();
            for (const auto& pending : pendingCacheEntries) {
                if (!viewportBackend->cloneRasterObjectByNodeName(pending.source_name, pending.new_name, pending.transform)) {
                    rasterCloneSucceeded = false;
                    break;
                }
            }
        }
        g_viewport_raster_rebuild_pending = !rasterCloneSucceeded;
        extern bool g_geometry_dirty;
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        g_geometry_dirty = true;
        g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
        extern bool g_vulkan_rebuild_pending;
        Backend::IBackend* renderBackend = getSelectionRenderBackend(ctx);
        bool optixCloneSucceeded = false;
        if (auto* optixBackend = dynamic_cast<Backend::OptixBackend*>(renderBackend)) {
            for (const auto& pending : pendingCacheEntries) {
                auto* optix = optixBackend->getOptixWrapper();
                if (!optix) {
                    optixCloneSucceeded = false;
                    break;
                }
                const auto newIds = optix->cloneInstancesByNodeName(pending.source_name, pending.new_name);
                if (newIds.empty()) {
                    optixCloneSucceeded = false;
                    break;
                }
                optixBackend->updateObjectTransform(pending.new_name, pending.transform);
                optixCloneSucceeded = true;
            }
            if (optixCloneSucceeded) {
                optixBackend->rebuildAccelerationStructure();
                optixBackend->resetAccumulation();
            }
        }
        bool vulkanCloneSucceeded = false;
        if (auto* vkBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(renderBackend)) {
            vulkanCloneSucceeded = !pendingCacheEntries.empty();
            for (const auto& pending : pendingCacheEntries) {
                std::shared_ptr<Hittable> representative;
                if (!pending.cache_entries.empty()) {
                    representative = pending.cache_entries.front().second;
                }
                if (!vkBackend->cloneRtObjectByNodeName(
                        pending.source_name, pending.new_name, representative, pending.transform)) {
                    vulkanCloneSucceeded = false;
                    break;
                }
            }
            g_vulkan_rebuild_pending = !vulkanCloneSucceeded;
        } else if (renderBackend && !optixCloneSucceeded) {
            g_optix_rebuild_pending = true;
        }
        extern bool g_bvh_rebuild_pending;
        extern int g_bvh_rebuild_deferred_frames;
        if (renderBackend && dynamic_cast<Backend::VulkanBackendAdapter*>(renderBackend) != nullptr) {
            g_bvh_rebuild_deferred_frames = std::max(g_bvh_rebuild_deferred_frames, 30);
            g_bvh_rebuild_pending = false;
        } else if (renderBackend && dynamic_cast<Backend::OptixBackend*>(renderBackend) != nullptr) {
            g_bvh_rebuild_deferred_frames = std::max(g_bvh_rebuild_deferred_frames, 30);
            g_bvh_rebuild_pending = false;
        } else {
            g_bvh_rebuild_pending = true;
        }
        
        ctx.renderer.resetCPUAccumulation();
        syncSelectionSceneState(ctx, true, false);
        
        ProjectManager::getInstance().markModified();
    }
}
