// ===============================================================================
// SCENE UI - GIZMOS & TRANSFORM
// ===============================================================================
// This file handles 3D Gizmos (Move/Rotate/Scale), Bounding Boxes, and overlays.
// ===============================================================================

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "scene_data.h"
#include "ParticleSimulation.h"
#include "ProjectManager.h"
#include "VDBVolumeManager.h"
#include "GasVolume.h"  // For gas simulation gizmos
#include "Api/RtPython.h"
#include "Api/RtApi.h"
#include "scene_ui_gas.hpp"  // For GasUI::selected_gas_volume
#include "scene_ui_forcefield.hpp"
#include "Backend/IViewportBackend.h"
#include <Backend/VulkanBackend.h>
#include <Backend/OptixBackend.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <omp.h>
#include <cstdint>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;

namespace {
Backend::IBackend* getGizmoRenderBackend(UIContext& ctx) {
    if (g_backend) {
        return g_backend.get();
    }
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr) {
        return ctx.backend_ptr;
    }
    return nullptr;
}

Backend::IViewportBackend* getGizmoViewportBackend(UIContext& ctx) {
    if (g_viewport_backend) {
        return g_viewport_backend.get();
    }
    return dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
}

// World->screen projection for the viewport gizmo overlays (lights, selection box, camera,
// force fields). Ortho-aware so these track the orthographic standard views instead of drifting
// to the wrong place/size — perspective math is bit-identical to the previous inline code.
// Returns false when the point should be skipped (behind a perspective camera, or degenerate).
inline bool projectGizmoWorldPoint(const Camera& cam, bool useOrtho, float aspect_ratio,
                                   float screen_w, float screen_h, const Vec3& point, ImVec2& out) {
    const Vec3 fwd = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 right = fwd.cross(cam.vup).normalize();
    const Vec3 up = right.cross(fwd).normalize();
    const Vec3 to_pt = point - cam.lookfrom;
    const float depth = to_pt.dot(fwd);
    if (!useOrtho && depth <= 0.01f) return false;
    const float local_x = to_pt.dot(right);
    const float local_y = to_pt.dot(up);
    const float half_h = useOrtho
        ? (((cam.ortho_height > 1e-4f) ? cam.ortho_height : 10.0f) * 0.5f)
        : (depth * tanf(cam.vfov * 0.5f * 3.14159265359f / 180.0f));
    const float half_w = half_h * aspect_ratio;
    if (fabsf(half_w) <= 1e-6f || fabsf(half_h) <= 1e-6f) return false;
    out.x = ((local_x / half_w) * 0.5f + 0.5f) * screen_w;
    out.y = (0.5f - (local_y / half_h) * 0.5f) * screen_h;
    return true;
}

void updateGizmoObjectTransformOnActiveBackends(UIContext& ctx, const std::string& objectName, const Matrix4x4& transform) {
    if (objectName.empty()) return;

    if (Backend::IViewportBackend* viewportBackend = getGizmoViewportBackend(ctx)) {
        viewportBackend->updateObjectTransform(objectName, transform);
        viewportBackend->resetAccumulation();
    }

    if (Backend::IBackend* renderBackend = getGizmoRenderBackend(ctx)) {
        if (renderBackend != getGizmoViewportBackend(ctx) && renderBackend->isUsingTLAS()) {
            renderBackend->updateObjectTransform(objectName, transform);
            renderBackend->resetAccumulation();
        }
    }
}
}

void SceneUI::drawParticleDebugOverlay(UIContext& ctx) {
    auto particles = ctx.scene.getParticleSimulationSystem();
    if (!ctx.scene.camera || !viewport_settings.show_gizmos) {
        return;
    }
    if (particles) {
        ctx.scene.pruneInvalidParticleObjectBindings();
    }

    // Keep sim-source object poses (and thus their collider/emitter/domain gizmos)
    // synced to the timeline playhead while editing. The sim loop re-poses during
    // playback/bake, but idle frame changes — and keyframe edits AT the current
    // frame (which don't change the frame number) — do not, so the surface mesh
    // cache the gizmos read for bounds stays stale and the boxes lag behind the
    // object. applySimSourceObjectPosesForFrame is now a no-op when the evaluated
    // pose is unchanged (it tracks the last pushed matrix), so calling it every
    // idle frame is cheap and catches both scrubbing and same-frame key edits.
    if (particles && !timeline.isPlaying()) {
        ctx.scene.applySimSourceObjectPosesForFrame(timeline.getCurrentFrame());
    }

    const Camera& cam = *ctx.scene.camera;
    ImGuiIO& io = ImGui::GetIO();
    const float screen_w = io.DisplaySize.x;
    const float screen_h = io.DisplaySize.y;
    const float aspect_ratio = (image_height > 0)
        ? (static_cast<float>(image_width) / static_cast<float>(image_height))
        : (screen_w / std::max(1.0f, screen_h));

    const Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    const Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    const float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    const float tan_half_fov = tanf(fov_rad * 0.5f);
    const bool gizmoOrtho = cam.orthographic && viewport_settings.shading_mode != 2;

    auto projectPoint = [&](const Vec3& point, ImVec2& out, float& depth) -> bool {
        const Vec3 to_pt = point - cam.lookfrom;
        depth = to_pt.dot(cam_forward);
        if (!gizmoOrtho && depth <= 0.01f) {
            return false;
        }

        const float local_x = to_pt.dot(cam_right);
        const float local_y = to_pt.dot(cam_up);
        const float half_h = gizmoOrtho ? (((cam.ortho_height > 1e-4f) ? cam.ortho_height : 10.0f) * 0.5f)
                                        : (depth * tan_half_fov);
        const float half_w = half_h * aspect_ratio;
        if (fabsf(half_w) <= 1e-6f || fabsf(half_h) <= 1e-6f) {
            return false;
        }

        out.x = ((local_x / half_w) * 0.5f + 0.5f) * screen_w;
        out.y = (0.5f - (local_y / half_h) * 0.5f) * screen_h;
        return out.x >= -32.0f && out.x <= screen_w + 32.0f &&
               out.y >= -32.0f && out.y <= screen_h + 32.0f;
    };

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();

    auto selectedSourceName = [&]() -> std::string {
        if (ctx.selection.selected.type == SelectableType::Object &&
            ctx.selection.selected.object &&
            !ctx.selection.selected.object->getNodeName().empty()) {
            return ctx.selection.selected.object->getNodeName();
        }
        if (ctx.selection.selected.type == SelectableType::ForceField &&
            ctx.selection.selected.force_field) {
            return ctx.selection.selected.force_field->name;
        }
        return std::string();
    };

    const std::string selected_source_name = selectedSourceName();
    // The collider/bounds gizmos resolve their box from the per-epoch surface-mesh
    // cache. A manual gizmo drag moves the object but may not bump the geometry
    // generation until release, so that memo can stay stale and the box lags the
    // object. Drop the selected object's memo each idle frame so its gizmo always
    // rebuilds from the live verts and tracks the drag. (Sim writeback already
    // drops the memo via the pivot setter; this covers manual editing.) Cosmetic.
    if (!timeline.isPlaying() && !selected_source_name.empty() &&
        ctx.scene.isObjectUsedAsSimSource(selected_source_name)) {
        ctx.scene.refreshSimSourceGizmoBounds(selected_source_name);
    }
    const int selected_domain_index =
        (ctx.selection.selected.type == SelectableType::SimulationDomain &&
         ctx.selection.selected.particle_system_index == ctx.scene.active_particle_system_index)
            ? ctx.selection.selected.simulation_domain_index
            : -1;

    auto drawAABB = [&](const Vec3& min_bound, const Vec3& max_bound, ImU32 color, float thickness) {
        const Vec3 mn = Vec3::min(min_bound, max_bound);
        const Vec3 mx = Vec3::max(min_bound, max_bound);
        const Vec3 c[8] = {
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
        for (const auto& edge : edges) {
            ImVec2 a, b;
            float da = 0.0f;
            float db = 0.0f;
            if (projectPoint(c[edge[0]], a, da) && projectPoint(c[edge[1]], b, db)) {
                draw_list->AddLine(a, b, color, thickness);
            }
        }
    };

    auto resolveObjectOBBCorners = [&](const std::string& object_name, float padding, Vec3 corners[8]) -> bool {
        if (object_name.empty()) {
            return false;
        }
        RayTrophiSim::ParticleColliderOBB obb;
        if (!ctx.scene.resolveObjectOBBForSimulation(object_name, obb)) {
            return false;
        }

        const Vec3 pad(std::max(0.0f, padding));
        const Vec3 local_min = obb.local_bounds_min - pad;
        const Vec3 local_max = obb.local_bounds_max + pad;
        const Vec3 local_corners[8] = {
            Vec3(local_min.x, local_min.y, local_min.z),
            Vec3(local_max.x, local_min.y, local_min.z),
            Vec3(local_max.x, local_max.y, local_min.z),
            Vec3(local_min.x, local_max.y, local_min.z),
            Vec3(local_min.x, local_min.y, local_max.z),
            Vec3(local_max.x, local_min.y, local_max.z),
            Vec3(local_max.x, local_max.y, local_max.z),
            Vec3(local_min.x, local_max.y, local_max.z)
        };

        for (int i = 0; i < 8; ++i) {
            corners[i] = obb.local_to_world.transform_point(local_corners[i]);
        }
        return true;
    };

    auto drawOBB = [&](const Vec3 corners[8], ImU32 color, float thickness) {
        const int edges[12][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            {0, 4}, {1, 5}, {2, 6}, {3, 7}
        };
        for (const auto& edge : edges) {
            ImVec2 a, b;
            float da = 0.0f;
            float db = 0.0f;
            if (projectPoint(corners[edge[0]], a, da) && projectPoint(corners[edge[1]], b, db)) {
                draw_list->AddLine(a, b, color, thickness);
            }
        }
    };

    auto drawPlaneY = [&](float y, ImU32 color, float thickness) {
        const Vec3 center = ctx.scene.camera ? ctx.scene.camera->lookat : Vec3(0.0f);
        const float size = 8.0f;
        const Vec3 min_bound(center.x - size, y, center.z - size);
        const Vec3 max_bound(center.x + size, y, center.z + size);
        const Vec3 corners[4] = {
            Vec3(min_bound.x, y, min_bound.z),
            Vec3(max_bound.x, y, min_bound.z),
            Vec3(max_bound.x, y, max_bound.z),
            Vec3(min_bound.x, y, max_bound.z)
        };
        for (int i = 0; i < 4; ++i) {
            ImVec2 a, b;
            float da = 0.0f;
            float db = 0.0f;
            if (projectPoint(corners[i], a, da) && projectPoint(corners[(i + 1) % 4], b, db)) {
                draw_list->AddLine(a, b, color, thickness);
            }
        }
    };

    auto drawSphere = [&](const Vec3& center, float radius, ImU32 color, float thickness) {
        const float r = std::max(0.001f, radius);
        constexpr int segments = 48;
        auto drawCircle = [&](const Vec3& axis_a, const Vec3& axis_b) {
            ImVec2 previous_screen;
            float previous_depth = 0.0f;
            bool has_previous = false;
            for (int i = 0; i <= segments; ++i) {
                const float t = (static_cast<float>(i) / static_cast<float>(segments)) * 6.28318530718f;
                const Vec3 p = center + axis_a * (std::cos(t) * r) + axis_b * (std::sin(t) * r);
                ImVec2 screen;
                float depth = 0.0f;
                if (projectPoint(p, screen, depth)) {
                    if (has_previous) {
                        draw_list->AddLine(previous_screen, screen, color, thickness);
                    }
                    previous_screen = screen;
                    previous_depth = depth;
                    has_previous = true;
                } else {
                    has_previous = false;
                }
            }
            (void)previous_depth;
        };
        drawCircle(Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f));
        drawCircle(Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));
        drawCircle(Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f));
    };

    auto drawCapsule = [&](const Vec3& a, const Vec3& b, float radius, ImU32 color, float thickness) {
        const float r = std::max(0.001f, radius);
        Vec3 axis = b - a;
        const float axis_length = axis.length();
        if (axis_length <= 1e-6f) {
            drawSphere(a, r, color, thickness);
            return;
        }
        axis = axis * (1.0f / axis_length);
        Vec3 side = Vec3::cross(axis, Vec3(0.0f, 1.0f, 0.0f));
        if (side.length() <= 1e-6f) {
            side = Vec3::cross(axis, Vec3(1.0f, 0.0f, 0.0f));
        }
        side = side * (1.0f / std::max(side.length(), 1e-6f));
        const Vec3 side_b = Vec3::cross(axis, side);

        drawSphere(a, r, color, thickness);
        drawSphere(b, r, color, thickness);
        const Vec3 offsets[4] = { side * r, side * -r, side_b * r, side_b * -r };
        for (const auto& offset : offsets) {
            ImVec2 pa, pb;
            float da = 0.0f;
            float db = 0.0f;
            if (projectPoint(a + offset, pa, da) && projectPoint(b + offset, pb, db)) {
                draw_list->AddLine(pa, pb, color, thickness);
            }
        }
    };

    // Legacy FluidObject overlay is suppressed once the user has any new
    // grid-domain Fluid in the active particle system — they've migrated and
    // the leftover legacy gizmos (plus their particles ticked by the old
    // FluidSimulationSystem) just create confusing duplicates. Faz 2 will
    // remove the FluidObject path entirely; this is the bridge.
    bool legacy_fluid_overlay_suppressed = false;
    if (particles) {
        for (const auto& gd : particles->gridDomains()) {
            if (gd.enabled && gd.type == RayTrophiSim::SimulationDomainType::Fluid) {
                legacy_fluid_overlay_suppressed = true;
                break;
            }
        }
        if (!legacy_fluid_overlay_suppressed) {
            for (const auto& gd_state : particles->gridDomainStates()) {
                if (gd_state.type == RayTrophiSim::SimulationDomainType::Fluid) {
                    legacy_fluid_overlay_suppressed = true;
                    break;
                }
            }
        }
    }
    for (const auto& fluid : ctx.scene.fluid_objects) {
        if (!fluid.visible || legacy_fluid_overlay_suppressed) {
            continue;
        }

        const ImU32 box_color = fluid.enabled ? IM_COL32(70, 190, 255, 160) : IM_COL32(70, 120, 150, 80);
        drawAABB(fluid.domain_min, fluid.domain_max, box_color, fluid.enabled ? 1.6f : 1.0f);
        drawAABB(fluid.seed_min, fluid.seed_max, IM_COL32(80, 235, 255, 120), 1.1f);

        if (fluid.particles.empty()) {
            continue;
        }

        ImDrawList* fluid_draw_list = ImGui::GetForegroundDrawList();
        const std::size_t count = fluid.particles.size();
        const std::size_t stride = count > 6000 ? (count / 6000 + 1) : 1;
        const float focal_px = screen_h / (2.0f * std::max(tan_half_fov, 1e-4f));
        const float world_radius = std::max(0.006f, fluid.voxel_size * 0.22f);

        for (std::size_t i = 0; i < count; i += stride) {
            ImVec2 screen_pos;
            float depth = 0.0f;
            if (!projectPoint(fluid.particles.position[i], screen_pos, depth)) {
                continue;
            }
            const float radius = std::clamp(world_radius * focal_px / std::max(depth, 0.05f), 1.1f, 18.0f);
            fluid_draw_list->AddCircleFilled(screen_pos, radius * 1.8f, IM_COL32(40, 170, 255, 70), 12);
            fluid_draw_list->AddCircleFilled(screen_pos, radius, IM_COL32(95, 225, 255, 210), 12);
        }
    }

    if (!particles) {
        return;
    }

    const auto& grid_domains = particles->gridDomains();
    const auto& grid_domain_states = particles->gridDomainStates();
    for (std::size_t domain_index = 0; domain_index < grid_domains.size(); ++domain_index) {
        const auto& domain = grid_domains[domain_index];
        if (!domain.enabled) {
            continue;
        }
        const bool selected =
            static_cast<int>(domain_index) == selected_domain_index ||
            (!domain.source_name.empty() && domain.source_name == selected_source_name);
        Vec3 min_bound = domain.bounds_min;
        Vec3 max_bound = domain.bounds_max;
        bool using_runtime_bounds = false;
        if (domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::ManualBox &&
            !selected &&
            domain_index < grid_domain_states.size() &&
            grid_domain_states[domain_index].valid) {
            min_bound = grid_domain_states[domain_index].bounds_min;
            max_bound = grid_domain_states[domain_index].bounds_max;
            using_runtime_bounds = true;
        }
        const Vec3 pad(using_runtime_bounds ? 0.0f : std::max(0.0f, domain.padding));
        if (domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::ObjectBounds &&
            !domain.source_name.empty()) {
            if (!ctx.scene.resolveObjectBoundsForSimulation(domain.source_name, min_bound, max_bound)) {
                continue;
            }
        }
        const Vec3 draw_min = Vec3::min(min_bound, max_bound);
        const Vec3 draw_max = Vec3::max(min_bound, max_bound);
        const Vec3 draw_extent = draw_max - draw_min;
        if (!std::isfinite(draw_extent.x) || !std::isfinite(draw_extent.y) || !std::isfinite(draw_extent.z) ||
            draw_extent.length() > 100000.0f) {
            continue;
        }
        const ImU32 color = selected ? IM_COL32(130, 220, 255, 245) : IM_COL32(105, 190, 255, 170);
        const float thickness = selected ? 2.2f : 1.35f;
        if (domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::ObjectBounds &&
            !domain.source_name.empty()) {
            Vec3 corners[8];
            if (resolveObjectOBBCorners(domain.source_name, domain.padding, corners)) {
                Vec3 corner_min(std::numeric_limits<float>::max());
                Vec3 corner_max(-std::numeric_limits<float>::max());
                bool valid_corners = true;
                for (const auto& corner : corners) {
                    valid_corners = valid_corners &&
                        std::isfinite(corner.x) &&
                        std::isfinite(corner.y) &&
                        std::isfinite(corner.z);
                    corner_min = Vec3::min(corner_min, corner);
                    corner_max = Vec3::max(corner_max, corner);
                }
                if (valid_corners && (corner_max - corner_min).length() <= 100000.0f) {
                    drawOBB(corners, color, thickness);
                }
            } else {
                drawAABB(Vec3::min(min_bound, max_bound) - pad, Vec3::max(min_bound, max_bound) + pad, color, thickness);
            }
        } else {
            drawAABB(Vec3::min(min_bound, max_bound) - pad, Vec3::max(min_bound, max_bound) + pad, color, thickness);
        }
    }

    // Flow-source authoring overlay: spawn region (Point sphere) + an arrow
    // showing the emission velocity direction so the user is not setting the
    // launch vector blind. Per-point normal emission (MeshSurface) has no single
    // direction, so its arrow is omitted.
    {
        const auto& flow_sources = particles->flowSources();
        for (const auto& src : flow_sources) {
            if (!src.enabled) continue;
            if (src.domain_index < 0 || src.domain_index >= static_cast<int>(grid_domains.size())) continue;
            if (!grid_domains[static_cast<std::size_t>(src.domain_index)].enabled) continue;

            const bool src_selected = (src.domain_index == selected_domain_index);
            const ImU32 src_col = src_selected ? IM_COL32(255, 180, 60, 245) : IM_COL32(255, 160, 40, 165);
            const float src_thick = src_selected ? 2.2f : 1.4f;

            if (src.source_mode == RayTrophiSim::SimulationFlowSourceMode::Point) {
                drawSphere(src.position, src.radius, src_col, src_selected ? 1.5f : 1.0f);
            }

            const float speed = src.velocity.length();
            const bool per_point = src.fluid_emit_along_normal &&
                src.source_mode == RayTrophiSim::SimulationFlowSourceMode::MeshSurface;
            if (speed > 1e-4f && !per_point) {
                const Vec3 dir = src.velocity * (1.0f / speed);
                const float len = std::clamp(speed * 0.25f, 0.3f, 5.0f);
                ImVec2 a, b;
                float da = 0.0f, db = 0.0f;
                if (projectPoint(src.position, a, da) &&
                    projectPoint(src.position + dir * len, b, db)) {
                    draw_list->AddLine(a, b, src_col, src_thick);
                    ImVec2 d(b.x - a.x, b.y - a.y);
                    const float dl = std::sqrt(d.x * d.x + d.y * d.y);
                    if (dl > 1e-3f) {
                        d.x /= dl; d.y /= dl;
                        const float hs = 11.0f; // arrowhead length (px)
                        const ImVec2 n(-d.y, d.x);
                        const ImVec2 h1(b.x - d.x * hs + n.x * hs * 0.5f, b.y - d.y * hs + n.y * hs * 0.5f);
                        const ImVec2 h2(b.x - d.x * hs - n.x * hs * 0.5f, b.y - d.y * hs - n.y * hs * 0.5f);
                        draw_list->AddLine(b, h1, src_col, src_thick);
                        draw_list->AddLine(b, h2, src_col, src_thick);
                    }
                }
            }
        }
    }

    // Fluid-type grid-domain particles + seed AABB as ImGui overlay. The dot
    // splat is gated by the per-domain `fluid_debug_overlay` toggle (default
    // OFF — the Particles render mode draws real RT instances now, so the
    // overlay would double-paint on top of them). The Seed AABB outline is
    // separate and always drawn for authoring feedback.
    for (std::size_t domain_index = 0; domain_index < grid_domain_states.size(); ++domain_index) {
        const auto& state = grid_domain_states[domain_index];
        if (!state.valid || state.type != RayTrophiSim::SimulationDomainType::Fluid) {
            continue;
        }
        bool draw_particle_dots = false;
        if (domain_index < grid_domains.size()) {
            const auto& fluid_domain = grid_domains[domain_index];
            // Seed AABB — cyan tint, distinguishes it from the domain box. Lets
            // the user see where Seed Fluid will deposit particles before they
            // press the button. FillLevel mirrors the resting-tank region the
            // seeder actually produces, including the budget-capped (effective)
            // fill height, so the cyan box matches the real result.
            Vec3 seed_lo = fluid_domain.fluid_seed_min;
            Vec3 seed_hi = fluid_domain.fluid_seed_max;
            if (fluid_domain.fluid_seed_mode == RayTrophiSim::FluidSeedMode::FillLevel) {
                RayTrophiSim::computeFluidFillSeedAABB(
                    state.bounds_min, state.bounds_max, state.voxel_size,
                    fluid_domain.fluid_fill_level, fluid_domain.fluid_fill_wall_margin,
                    std::max(1, fluid_domain.fluid_seed_particles_per_cell),
                    fluid_domain.fluid_max_particles,
                    seed_lo, seed_hi);
            }
            drawAABB(seed_lo,
                     seed_hi,
                     IM_COL32(80, 235, 255, 130),
                     1.1f);
            draw_particle_dots = fluid_domain.fluid_debug_overlay;
        }
        if (!draw_particle_dots || state.particles.empty()) {
            continue;
        }
        ImDrawList* fluid_draw_list = ImGui::GetForegroundDrawList();
        const std::size_t count = state.particles.size();
        const std::size_t stride = count > 6000 ? (count / 6000 + 1) : 1;
        const float focal_px = screen_h / (2.0f * std::max(tan_half_fov, 1e-4f));
        const float world_radius = std::max(0.006f, state.voxel_size * 0.22f);
        for (std::size_t i = 0; i < count; i += stride) {
            ImVec2 screen_pos;
            float depth = 0.0f;
            if (!projectPoint(state.particles.position[i], screen_pos, depth)) {
                continue;
            }
            const float radius = std::clamp(world_radius * focal_px / std::max(depth, 0.05f), 1.1f, 18.0f);
            fluid_draw_list->AddCircleFilled(screen_pos, radius * 1.8f, IM_COL32(40, 170, 255, 70), 12);
            fluid_draw_list->AddCircleFilled(screen_pos, radius,       IM_COL32(95, 225, 255, 210), 12);
        }
    }

    for (const auto& emitter : particles->emitters()) {
        if ((emitter.spawn_mode != RayTrophiSim::ParticleEmitterSpawnMode::ObjectAABBSurface &&
             emitter.spawn_mode != RayTrophiSim::ParticleEmitterSpawnMode::MeshSurface) ||
            emitter.source_name.empty()) {
            continue;
        }
        Vec3 min_bound;
        Vec3 max_bound;
        if (!ctx.scene.resolveObjectBoundsForSimulation(emitter.source_name, min_bound, max_bound)) {
            continue;
        }
        const bool selected = emitter.source_name == selected_source_name;
        const ImU32 color = emitter.enabled
            ? (selected ? IM_COL32(80, 240, 255, 245) : IM_COL32(80, 220, 255, 150))
            : IM_COL32(80, 150, 170, 70);
        Vec3 corners[8];
        if (resolveObjectOBBCorners(emitter.source_name, 0.0f, corners)) {
            drawOBB(corners, color, selected ? 2.2f : 1.3f);
        } else {
            drawAABB(min_bound, max_bound, color, selected ? 2.2f : 1.3f);
        }
    }

    for (const auto& collider : particles->colliders()) {
        const bool selected = collider.source_name == selected_source_name;
        const ImU32 color = collider.enabled
            ? (selected ? IM_COL32(255, 185, 70, 250) : IM_COL32(255, 150, 60, 150))
            : IM_COL32(150, 105, 70, 70);
        const float thickness = selected ? 2.4f : 1.4f;
        if (collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectAABB ||
            collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectOBB) {
            Vec3 min_bound;
            Vec3 max_bound;
            if (ctx.scene.resolveObjectBoundsForSimulation(collider.source_name, min_bound, max_bound)) {
                const Vec3 pad(std::max(0.0f, collider.thickness));
                if (collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectOBB) {
                    Vec3 corners[8];
                    if (resolveObjectOBBCorners(collider.source_name, collider.thickness, corners)) {
                        drawOBB(corners, color, thickness);
                    } else {
                        drawAABB(min_bound - pad, max_bound + pad, color, thickness);
                    }
                } else {
                    drawAABB(min_bound - pad, max_bound + pad, color, thickness);
                }
            }
        } else if (collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::PlaneY) {
            drawPlaneY(collider.plane_y, color, thickness);
        } else if (collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::Sphere) {
            Vec3 center = collider.sphere_center;
            float radius = collider.sphere_radius;
            if (!collider.source_name.empty()) {
                Vec3 min_bound;
                Vec3 max_bound;
                if (ctx.scene.resolveObjectBoundsForSimulation(collider.source_name, min_bound, max_bound)) {
                    const Vec3 mn = Vec3::min(min_bound, max_bound);
                    const Vec3 mx = Vec3::max(min_bound, max_bound);
                    center = (mn + mx) * 0.5f;
                    radius = (mx - mn).length() * 0.5f;
                }
            }
            drawSphere(center, radius + std::max(0.0f, collider.thickness), color, thickness);
        } else if (collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::Capsule) {
            Vec3 start = collider.capsule_start;
            Vec3 end = collider.capsule_end;
            float radius = collider.capsule_radius;
            if (!collider.source_name.empty()) {
                Vec3 min_bound;
                Vec3 max_bound;
                if (ctx.scene.resolveObjectBoundsForSimulation(collider.source_name, min_bound, max_bound)) {
                    const Vec3 mn = Vec3::min(min_bound, max_bound);
                    const Vec3 mx = Vec3::max(min_bound, max_bound);
                    const Vec3 center = (mn + mx) * 0.5f;
                    const Vec3 extent = mx - mn;
                    const float min_side = std::min({ extent.x, extent.y, extent.z });
                    radius = std::max(0.001f, min_side * 0.5f);
                    if (extent.x >= extent.y && extent.x >= extent.z) {
                        start = Vec3(mn.x, center.y, center.z);
                        end = Vec3(mx.x, center.y, center.z);
                    } else if (extent.y >= extent.x && extent.y >= extent.z) {
                        start = Vec3(center.x, mn.y, center.z);
                        end = Vec3(center.x, mx.y, center.z);
                    } else {
                        start = Vec3(center.x, center.y, mn.z);
                        end = Vec3(center.x, center.y, mx.z);
                    }
                }
            }
            drawCapsule(start, end, radius + std::max(0.0f, collider.thickness), color, thickness);
        } else if (collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF) {
            RayTrophiSim::ParticleColliderOBB obb;
            if (!ctx.scene.resolveObjectOBBForSimulation(collider.source_name, obb)) {
                // Static AABB fallback if no OBB is resolved
                const float thick = std::max(0.0f, collider.thickness);
                drawAABB(collider.sdf_origin, collider.sdf_origin + collider.sdf_extents, color, thickness);
                if (thick > 0.0f) {
                    drawAABB(collider.sdf_origin - Vec3(thick), collider.sdf_origin + collider.sdf_extents + Vec3(thick), IM_COL32(255, 185, 70, 90), 1.0f);
                }
                continue;
            }

            const float thick = std::max(0.0f, collider.thickness);
            
            // Draw OBB bounding box of the compiled SDF
            const Vec3 local_mn = collider.sdf_origin;
            const Vec3 local_mx = collider.sdf_origin + collider.sdf_extents;
            const Vec3 local_corners[8] = {
                Vec3(local_mn.x, local_mn.y, local_mn.z),
                Vec3(local_mx.x, local_mn.y, local_mn.z),
                Vec3(local_mx.x, local_mx.y, local_mn.z),
                Vec3(local_mn.x, local_mx.y, local_mn.z),
                Vec3(local_mn.x, local_mn.y, local_mx.z),
                Vec3(local_mx.x, local_mn.y, local_mx.z),
                Vec3(local_mx.x, local_mx.y, local_mx.z),
                Vec3(local_mn.x, local_mx.y, local_mx.z)
            };
            Vec3 world_corners[8];
            for (int i = 0; i < 8; ++i) {
                world_corners[i] = obb.local_to_world.transform_point(local_corners[i]);
            }
            drawOBB(world_corners, color, thickness);

            // Draw thickness contour OBB
            if (thick > 0.0f) {
                const Vec3 local_mn_thick = local_mn - Vec3(thick);
                const Vec3 local_mx_thick = local_mx + Vec3(thick);
                const Vec3 local_corners_thick[8] = {
                    Vec3(local_mn_thick.x, local_mn_thick.y, local_mn_thick.z),
                    Vec3(local_mx_thick.x, local_mn_thick.y, local_mn_thick.z),
                    Vec3(local_mx_thick.x, local_mx_thick.y, local_mn_thick.z),
                    Vec3(local_mn_thick.x, local_mx_thick.y, local_mn_thick.z),
                    Vec3(local_mn_thick.x, local_mn_thick.y, local_mx_thick.z),
                    Vec3(local_mx_thick.x, local_mn_thick.y, local_mx_thick.z),
                    Vec3(local_mx_thick.x, local_mx_thick.y, local_mx_thick.z),
                    Vec3(local_mn_thick.x, local_mx_thick.y, local_mx_thick.z)
                };
                Vec3 world_corners_thick[8];
                for (int i = 0; i < 8; ++i) {
                    world_corners_thick[i] = obb.local_to_world.transform_point(local_corners_thick[i]);
                }
                drawOBB(world_corners_thick, IM_COL32(255, 185, 70, 90), 1.0f);
            }

            // Draw wireframe isosurface zero-crossings in world coordinates (mapped from local space)
            if (collider.draw_wireframe && collider.sdf_grid_data && !collider.sdf_grid_data->empty()) {
                int nx = collider.sdf_nx;
                int ny = collider.sdf_ny;
                int nz = collider.sdf_nz;
                int stride = nx > 32 ? (nx / 32) : 1;
                
                const auto sampleSDF = [&](int x, int y, int z) -> float {
                    std::size_t idx = static_cast<std::size_t>(z * (nx * ny) + y * nx + x);
                    return idx < collider.sdf_grid_data->size() ? (*collider.sdf_grid_data)[idx] : 1.0f;
                };
                
                auto getCellPos = [&](int x, int y, int z) {
                    return collider.sdf_origin + Vec3(
                        (x + 0.5f) * (collider.sdf_extents.x / nx),
                        (y + 0.5f) * (collider.sdf_extents.y / ny),
                        (z + 0.5f) * (collider.sdf_extents.z / nz)
                    );
                };

                int count = 0;
                for (int k = 0; k < nz && count < 2000; k += stride)
                for (int j = 0; j < ny && count < 2000; j += stride)
                for (int i = 0; i < nx && count < 2000; i += stride) {
                    float v0 = sampleSDF(i, j, k);
                    
                    if (i + 1 < nx) {
                        float v1 = sampleSDF(i + 1, j, k);
                        if ((v0 <= 0.0f && v1 > 0.0f) || (v0 > 0.0f && v1 <= 0.0f)) {
                            ImVec2 a, b;
                            float da = 0.0f, db = 0.0f;
                            Vec3 wp0 = obb.local_to_world.transform_point(getCellPos(i, j, k));
                            Vec3 wp1 = obb.local_to_world.transform_point(getCellPos(i + 1, j, k));
                            if (projectPoint(wp0, a, da) && projectPoint(wp1, b, db)) {
                                draw_list->AddLine(a, b, color, 1.0f);
                                count++;
                            }
                        }
                    }
                    if (j + 1 < ny) {
                        float v1 = sampleSDF(i, j + 1, k);
                        if ((v0 <= 0.0f && v1 > 0.0f) || (v0 > 0.0f && v1 <= 0.0f)) {
                            ImVec2 a, b;
                            float da = 0.0f, db = 0.0f;
                            Vec3 wp0 = obb.local_to_world.transform_point(getCellPos(i, j, k));
                            Vec3 wp1 = obb.local_to_world.transform_point(getCellPos(i, j + 1, k));
                            if (projectPoint(wp0, a, da) && projectPoint(wp1, b, db)) {
                                draw_list->AddLine(a, b, color, 1.0f);
                                count++;
                            }
                        }
                    }
                    if (k + 1 < nz) {
                        float v1 = sampleSDF(i, j, k + 1);
                        if ((v0 <= 0.0f && v1 > 0.0f) || (v0 > 0.0f && v1 <= 0.0f)) {
                            ImVec2 a, b;
                            float da = 0.0f, db = 0.0f;
                            Vec3 wp0 = obb.local_to_world.transform_point(getCellPos(i, j, k));
                            Vec3 wp1 = obb.local_to_world.transform_point(getCellPos(i, j, k + 1));
                            if (projectPoint(wp0, a, da) && projectPoint(wp1, b, db)) {
                                draw_list->AddLine(a, b, color, 1.0f);
                                count++;
                            }
                        }
                    }
                }
            }

            // Draw 2D Kesit Grid Slice Preview
            if (collider.draw_slice_preview && collider.sdf_grid_data && !collider.sdf_grid_data->empty()) {
                int nx = collider.sdf_nx;
                int ny = collider.sdf_ny;
                int nz = collider.sdf_nz;
                
                const auto sampleSDF = [&](int x, int y, int z) -> float {
                    std::size_t idx = static_cast<std::size_t>(z * (nx * ny) + y * nx + x);
                    return idx < collider.sdf_grid_data->size() ? (*collider.sdf_grid_data)[idx] : 1.0f;
                };
                
                auto getCellPos = [&](int x, int y, int z) {
                    return collider.sdf_origin + Vec3(
                        (x + 0.5f) * (collider.sdf_extents.x / nx),
                        (y + 0.5f) * (collider.sdf_extents.y / ny),
                        (z + 0.5f) * (collider.sdf_extents.z / nz)
                    );
                };

                int axis = collider.slice_axis;
                float t_slice = collider.slice_plane_distance;
                
                if (axis == 1) { // Y Axis
                    int j = std::clamp(static_cast<int>(t_slice * (ny - 1)), 0, ny - 1);
                    for (int k = 0; k < nz; k += 2)
                    for (int i = 0; i < nx; i += 2) {
                        float dist = sampleSDF(i, j, k);
                        if (dist <= std::max(0.0f, collider.thickness)) {
                            Vec3 cp = getCellPos(i, j, k);
                            Vec3 wcp = obb.local_to_world.transform_point(cp);
                            ImVec2 scr;
                            float depth = 0.0f;
                            if (projectPoint(wcp, scr, depth)) {
                                draw_list->AddCircleFilled(scr, 2.5f, IM_COL32(255, 120, 30, 180));
                            }
                        }
                    }
                } else if (axis == 0) { // X Axis
                    int i = std::clamp(static_cast<int>(t_slice * (nx - 1)), 0, nx - 1);
                    for (int k = 0; k < nz; k += 2)
                    for (int j = 0; j < ny; j += 2) {
                        float dist = sampleSDF(i, j, k);
                        if (dist <= std::max(0.0f, collider.thickness)) {
                            Vec3 cp = getCellPos(i, j, k);
                            Vec3 wcp = obb.local_to_world.transform_point(cp);
                            ImVec2 scr;
                            float depth = 0.0f;
                            if (projectPoint(wcp, scr, depth)) {
                                draw_list->AddCircleFilled(scr, 2.5f, IM_COL32(255, 120, 30, 180));
                            }
                        }
                    }
                } else { // Z Axis
                    int k = std::clamp(static_cast<int>(t_slice * (nz - 1)), 0, nz - 1);
                    for (int j = 0; j < ny; j += 2)
                    for (int i = 0; i < nx; i += 2) {
                        float dist = sampleSDF(i, j, k);
                        if (dist <= std::max(0.0f, collider.thickness)) {
                            Vec3 cp = getCellPos(i, j, k);
                            Vec3 wcp = obb.local_to_world.transform_point(cp);
                            ImVec2 scr;
                            float depth = 0.0f;
                            if (projectPoint(wcp, scr, depth)) {
                                draw_list->AddCircleFilled(scr, 2.5f, IM_COL32(255, 120, 30, 180));
                            }
                        }
                    }
                }
            }
        } else if (collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp ||
                   collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH) {
            Vec3 min_bound;
            Vec3 max_bound;
            if (ctx.scene.resolveObjectBoundsForSimulation(collider.source_name, min_bound, max_bound)) {
                const Vec3 pad(std::max(0.0f, collider.thickness));
                Vec3 corners[8];
                if (resolveObjectOBBCorners(collider.source_name, collider.thickness, corners)) {
                    drawOBB(corners, color, thickness);
                    if (collider.draw_wireframe) {
                        drawOBB(corners, IM_COL32(255, 185, 70, 80), 1.0f);
                    }
                } else {
                    drawAABB(min_bound - pad, max_bound + pad, color, thickness);
                }
            }
        }
    }

    // Particle dots only in Debug display mode. Solid/Render render the particles
    // for real via the Vulkan billboard pass (uploadParticleBillboards). These
    // ImGui foreground dots draw on top of everything (including UI panels), so
    // they are an opt-in debug aid, not the default.
    if (ctx.particle_display_mode != 1) {
        return;
    }
    if (particles->aliveCount() == 0) {
        return;
    }
    draw_list = ImGui::GetForegroundDrawList();

    const auto& buffers = particles->buffers();
    const std::size_t capacity = particles->capacity();
    const std::size_t stride = capacity > 4000 ? (capacity / 4000 + 1) : 1;
    const float focal_px = screen_h / (2.0f * std::max(tan_half_fov, 1e-4f));

    const auto hasAttr = [](const std::vector<float>& v, std::size_t i) {
        return i < v.size();
    };
    const auto channel = [](float c) {
        return static_cast<int>(std::clamp(c, 0.0f, 1.0f) * 255.0f + 0.5f);
    };

    for (std::size_t i = 0; i < capacity; i += stride) {
        if (i >= buffers.alive.size() || buffers.alive[i] == 0u) {
            continue;
        }
        const Vec3 pos(buffers.position_x[i], buffers.position_y[i], buffers.position_z[i]);
        ImVec2 screen_pos;
        float depth = 0.0f;
        if (!projectPoint(pos, screen_pos, depth)) {
            continue;
        }
        const float opacity = hasAttr(buffers.opacity, i) ? std::clamp(buffers.opacity[i], 0.0f, 1.0f) : 1.0f;
        if (opacity <= 0.002f) {
            continue;
        }
        const float world_size = hasAttr(buffers.size, i) ? buffers.size[i] : 0.05f;
        const float radius = std::clamp((world_size * 0.5f) * focal_px / std::max(depth, 0.05f), 1.0f, 320.0f);
        const int cr = hasAttr(buffers.color_r, i) ? channel(buffers.color_r[i]) : 255;
        const int cg = hasAttr(buffers.color_g, i) ? channel(buffers.color_g[i]) : 255;
        const int cb = hasAttr(buffers.color_b, i) ? channel(buffers.color_b[i]) : 255;
        const ImU32 halo_col = IM_COL32(cr, cg, cb, static_cast<int>(opacity * 70.0f));
        const ImU32 core_col = IM_COL32(cr, cg, cb, static_cast<int>(opacity * 205.0f));
        draw_list->AddCircleFilled(screen_pos, radius * 1.9f, halo_col, 14);
        draw_list->AddCircleFilled(screen_pos, radius, core_col, 14);
    }
}

void SceneUI::uploadParticleBillboards(UIContext& ctx) {
    auto* vpb = dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get());
    if (!vpb) {
        return;
    }
    // Debug display mode uses the ImGui overlay instead; clear any billboards so we
    // don't render both. Solid (0) and Render (2) draw real billboards.
    if (ctx.particle_display_mode == 1 || !ctx.scene.camera) {
        vpb->uploadParticleBillboards({}, 0, {}, 0);
        return;
    }

    const Camera& cam = *ctx.scene.camera;
    // Hand-rolled normalize: Vec3::normalize() zeroes sub-mm vectors.
    auto unit = [](const Vec3& v, const Vec3& fb) {
        const float l = v.length();
        return l > 1e-8f ? v * (1.0f / l) : fb;
    };
    const Vec3 fwd   = unit(cam.lookat - cam.lookfrom, Vec3(0.0f, 0.0f, -1.0f));
    const Vec3 right = unit(Vec3::cross(fwd, cam.vup), Vec3(1.0f, 0.0f, 0.0f));
    const Vec3 up    = unit(Vec3::cross(right, fwd), Vec3(0.0f, 1.0f, 0.0f));

    std::vector<float> addData;
    std::vector<float> alphaData;
    constexpr std::size_t kMaxBillboards = 60000; // safety cap across all systems
    std::size_t drawn = 0;

    auto pushVertex = [](std::vector<float>& out, const Vec3& p, float u, float v,
                         float r, float g, float b, float a) {
        out.push_back(p.x); out.push_back(p.y); out.push_back(p.z);
        out.push_back(u);   out.push_back(v);
        out.push_back(r);   out.push_back(g); out.push_back(b); out.push_back(a);
    };

    for (const auto& system : ctx.scene.particle_systems) {
        if (!system.visible || !system.runtime) {
            continue;
        }
        std::vector<float>& out = (system.blend_mode == SceneData::ParticleBlendMode::Alpha)
            ? alphaData : addData;

        const auto& buf = system.runtime->buffers();
        const std::size_t cap = buf.alive.size();
        for (std::size_t i = 0; i < cap && drawn < kMaxBillboards; ++i) {
            if (buf.alive[i] == 0u) continue;

            const float a = (i < buf.opacity.size()) ? buf.opacity[i] : 1.0f;
            if (a <= 0.002f) continue;
            const float sz = (i < buf.size.size()) ? buf.size[i] : 0.05f;
            const float h = sz * 0.5f;
            if (h <= 1e-5f) continue;

            const Vec3 c(buf.position_x[i], buf.position_y[i], buf.position_z[i]);
            const float r = (i < buf.color_r.size()) ? buf.color_r[i] : 1.0f;
            const float g = (i < buf.color_g.size()) ? buf.color_g[i] : 1.0f;
            const float bcol = (i < buf.color_b.size()) ? buf.color_b[i] : 1.0f;

            const Vec3 rh = right * h;
            const Vec3 uh = up * h;
            const Vec3 c00 = c - rh - uh;
            const Vec3 c10 = c + rh - uh;
            const Vec3 c11 = c + rh + uh;
            const Vec3 c01 = c - rh + uh;

            pushVertex(out, c00, -1.f, -1.f, r, g, bcol, a);
            pushVertex(out, c10,  1.f, -1.f, r, g, bcol, a);
            pushVertex(out, c11,  1.f,  1.f, r, g, bcol, a);
            pushVertex(out, c00, -1.f, -1.f, r, g, bcol, a);
            pushVertex(out, c11,  1.f,  1.f, r, g, bcol, a);
            pushVertex(out, c01, -1.f,  1.f, r, g, bcol, a);
            ++drawn;
        }
        if (drawn >= kMaxBillboards) break;
    }

    const uint32_t addCount   = static_cast<uint32_t>(addData.size() / 9);
    const uint32_t alphaCount = static_cast<uint32_t>(alphaData.size() / 9);
    vpb->uploadParticleBillboards(addData, addCount, alphaData, alphaCount);
}

void SceneUI::moveObjectPivot(UIContext& ctx, const std::string& objectName, const Vec3& worldDelta) {
    if (objectName.empty()) return;
    if (worldDelta.length_squared() < 1e-12f) return;
    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

    auto cache_it = mesh_cache.find(objectName);
    if (cache_it == mesh_cache.end() || cache_it->second.empty()) return;

    Transform* transform = cache_it->second[0].second->getTransformPtr();
    if (!transform) return;

    const Vec3 newPivotWorld = transform->position + worldDelta;
    const Vec3 newPivotLocal = transform->base.inverse().transform_point(newPivotWorld);
    transform->setPivotOffset(newPivotLocal, true);

    updateBBoxCache(objectName);
    objects_needing_cpu_sync.erase(objectName);

    if (ctx.selection.selected.type == SelectableType::Object &&
        ctx.selection.selected.object &&
        ctx.selection.selected.object->getNodeName() == objectName) {
        Matrix4x4 pivotMat = transform->getPivotMatrix();
        Vec3 p, r, s;
        pivotMat.decompose(p, r, s);
        ctx.selection.selected.position = p;
        ctx.selection.selected.rotation = r;
        ctx.selection.selected.scale = s;
        ctx.selection.selected.has_cached_aabb = false;
    }

    const bool render_backend_has_tlas =
        (getGizmoRenderBackend(ctx) && getGizmoRenderBackend(ctx)->isUsingTLAS());
    // Always push to active backends (raster viewport + render backend)
    updateGizmoObjectTransformOnActiveBackends(ctx, objectName, transform->base);
    if (!render_backend_has_tlas) {
        // CPU mode: update vertices for BVH/picking
        for (auto& pair : cache_it->second) {
            pair.second->updateTransformedVertices();
        }
        if (ctx.backend_ptr && !dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr)) {
            ctx.backend_ptr->updateGeometry(ctx.scene.world.objects);
            ctx.backend_ptr->resetAccumulation();
        } else {
            extern bool g_cpu_bvh_refit_pending;
            g_cpu_bvh_refit_pending = true;
        }
    }
    ctx.renderer.resetCPUAccumulation();
}

void SceneUI::recenterObjectPivotToBoundsCenter(UIContext& ctx, const std::string& objectName) {
    if (objectName.empty()) return;
    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

    auto bbox_it = bbox_cache.find(objectName);
    if (bbox_it == bbox_cache.end()) return;

    const Vec3 localCenter = (bbox_it->second.first + bbox_it->second.second) * 0.5f;
    if (localCenter.length_squared() < 1e-12f) return;

    auto cache_it = mesh_cache.find(objectName);
    if (cache_it == mesh_cache.end() || cache_it->second.empty()) return;

    Transform* transform = cache_it->second[0].second->getTransformPtr();
    if (!transform) return;

    const Vec3 targetPivotWorld = transform->base.transform_point(localCenter);
    moveObjectPivot(ctx, objectName, targetPivotWorld - transform->position);
}

// =============================================================================
// ===============================================================================
// SELECTION BOUNDING BOX DRAWING (Multi-selection support)
// ===============================================================================
bool SceneUI::bodyWorldAABB(UIContext& ctx, const std::string& node, Vec3& out_min, Vec3& out_max) {
    const uint64_t ver = ctx.scene.bodyGeomVersion();
    // Memo hit: stopped body (version unchanged) and not being dragged → O(1).
    auto it = body_aabb_memo_.find(node);
    if (!is_dragging && it != body_aabb_memo_.end() && std::get<0>(it->second) == ver) {
        out_min = std::get<1>(it->second);
        out_max = std::get<2>(it->second);
        return true;
    }
    // Recompute from the body's OWN triangles (O(1) node lookup, NOT a scene scan).
    auto mcit = mesh_cache.find(node);
    if (mcit == mesh_cache.end()) return false;
    Vec3 mn(1e30f, 1e30f, 1e30f), mx(-1e30f, -1e30f, -1e30f);
    bool any = false;
    for (const auto& entry : mcit->second) {
        const auto& tri = entry.second;
        if (!tri) continue;
        for (int i = 0; i < 3; ++i) {
            const Vec3 p = tri->getVertexPosition(i);
            mn = Vec3::min(mn, p); mx = Vec3::max(mx, p); any = true;
        }
    }
    if (!any) return false;
    out_min = mn; out_max = mx;
    body_aabb_memo_[node] = std::make_tuple(ver, mn, mx);
    return true;
}

void SceneUI::drawSelectionBoundingBox(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    Camera& cam = *ctx.scene.camera;
    ImGuiIO& io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;
    // Use render resolution for aspect ratio to match GPU projection matrix.
    float aspect_ratio = (image_height > 0)
        ? (static_cast<float>(image_width) / static_cast<float>(image_height))
        : (screen_w / screen_h);

    // Camera basis vectors
    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();

    // FOV calculations
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    // Orthographic standard views (raster preview only — Rendered path-traces in perspective).
    const bool gizmoOrtho = cam.orthographic && viewport_settings.shading_mode != 2;
    // Helper lambda to draw an oriented box from 8 explicit corners with granular occlusion.
    // Corner order matches axis-aligned convention:
    //   0:(-,-,-) 1:(+,-,-) 2:(+,+,-) 3:(-,+,-) 4:(-,-,+) 5:(+,-,+) 6:(+,+,+) 7:(-,+,+)
    auto DrawOrientedBox = [&](const Vec3 corners[8], ImU32 color, float thickness) {
        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        auto ProjectPoint = [&](const Vec3& point, ImVec2& out) -> bool {
            return projectGizmoWorldPoint(cam, gizmoOrtho, aspect_ratio, screen_w, screen_h, point, out);
        };

        float proj_min_x = screen_w;
        float proj_min_y = screen_h;
        float proj_max_x = 0.0f;
        float proj_max_y = 0.0f;
        bool has_projected_corner = false;
        for (int i = 0; i < 8; ++i) {
            ImVec2 projected;
            if (!ProjectPoint(corners[i], projected)) continue;
            has_projected_corner = true;
            proj_min_x = fminf(proj_min_x, projected.x);
            proj_min_y = fminf(proj_min_y, projected.y);
            proj_max_x = fmaxf(proj_max_x, projected.x);
            proj_max_y = fmaxf(proj_max_y, projected.y);
        }

        // Derive world AABB of the (possibly oriented) corners for occlusion heuristics only.
        Vec3 wmin(1e10f, 1e10f, 1e10f), wmax(-1e10f, -1e10f, -1e10f);
        for (int i = 0; i < 8; ++i) {
            const Vec3& c = corners[i];
            wmin.x = fminf(wmin.x, c.x); wmin.y = fminf(wmin.y, c.y); wmin.z = fminf(wmin.z, c.z);
            wmax.x = fmaxf(wmax.x, c.x); wmax.y = fmaxf(wmax.y, c.y); wmax.z = fmaxf(wmax.z, c.z);
        }
        const Vec3 bb_extent = wmax - wmin;
        const float bbox_diagonal = bb_extent.length();
        const Vec3 bb_center = (wmin + wmax) * 0.5f;
        const float camera_distance = (bb_center - cam.lookfrom).length();
        const float projected_width = has_projected_corner ? (proj_max_x - proj_min_x) : 0.0f;
        const float projected_height = has_projected_corner ? (proj_max_y - proj_min_y) : 0.0f;
        const float screen_area = (std::max)(1.0f, screen_w * screen_h);
        const float projected_area_ratio = (projected_width * projected_height) / screen_area;
        const float screen_diagonal = sqrtf(screen_w * screen_w + screen_h * screen_h);
        const float projected_diagonal = sqrtf(projected_width * projected_width + projected_height * projected_height);
        const float apparent_scale = camera_distance > 0.001f ? (bbox_diagonal / camera_distance) : bbox_diagonal;

        const bool use_occlusion =
            has_projected_corner &&
            projected_area_ratio < 0.10f &&
            projected_diagonal < screen_diagonal * 0.40f &&
            apparent_scale < 1.2f;
        
        // Helper to draw a line with subdivision and occlusion check
        auto DrawSegmentedLine = [&](const Vec3& p_start, const Vec3& p_end) {
            if (!use_occlusion) {
                ImVec2 start_scr, end_scr;
                if (ProjectPoint(p_start, start_scr) && ProjectPoint(p_end, end_scr)) {
                    draw_list->AddLine(start_scr, end_scr, color, thickness);
                }
                return;
            }

            const int segments = 8; // Subdivision level for accurate occlusion
            Vec3 prev_p = p_start;
            ImVec2 prev_scr;
            
            // Project start point
            bool prev_vis = ProjectPoint(prev_p, prev_scr);

            for (int i = 1; i <= segments; ++i) {
                float t = (float)i / (float)segments;
                Vec3 curr_p = p_start * (1.0f - t) + p_end * t;
                
                ImVec2 curr_scr;
                bool curr_vis = ProjectPoint(curr_p, curr_scr);

                if (prev_vis && curr_vis) {
                    // Check visibility of the segment midpoint
                    Vec3 mid_p = (prev_p + curr_p) * 0.5f;
                    ImU32 segment_color = color;
                    
                    // [RACE FIX] Animation worker thread mutates HittableInstance
                    // transforms (and thus bounds / inv_transform) per frame via
                    // updateAnimationState. CPU BVH hit traverses those same
                    // hittables and reads transform / inv_transform / bbox during
                    // ray-object intersection — Embree call backs into user
                    // geometry. Concurrent read/write on Matrix4x4 produces torn
                    // matrices that the Embree internals then dereference,
                    // crashing inside embree4.dll. Skip the occlusion probe while
                    // the render worker owns the scene.
                    extern bool g_bvh_rebuild_pending;
                    if (ctx.scene.bvh && !g_bvh_rebuild_pending && !ctx.is_animation_mode) {
                        Vec3 to_mid = mid_p - cam.lookfrom;
                        float dist = to_mid.length();
                        if (dist > 0.1f) {
                            Ray r(cam.lookfrom, to_mid / dist);
                            HitRecord rec;
                            // Check occlusion: if hit anything closer than the segment
                            if (ctx.scene.bvh->hit(r, 0.001f, dist - 0.05f, rec, true)) {
                                // Occluded → desaturate toward neutral gray (not just
                                // fade) so alignment against the occluder reads
                                // clearly instead of disappearing.
                                int rr = (color      ) & 0xFF;
                                int gg = (color >>  8) & 0xFF;
                                int bb = (color >> 16) & 0xFF;
                                rr = (rr + 160) >> 1;
                                gg = (gg + 160) >> 1;
                                bb = (bb + 160) >> 1;
                                segment_color = IM_COL32(rr, gg, bb, 120);
                            }
                        }
                    }
                    
                    draw_list->AddLine(prev_scr, curr_scr, segment_color, thickness);
                }

                prev_p = curr_p;
                prev_scr = curr_scr;
                prev_vis = curr_vis;
            }
        };

        // Draw 12 edges of the box
        DrawSegmentedLine(corners[0], corners[1]);
        DrawSegmentedLine(corners[1], corners[2]);
        DrawSegmentedLine(corners[2], corners[3]);
        DrawSegmentedLine(corners[3], corners[0]);

        DrawSegmentedLine(corners[4], corners[5]);
        DrawSegmentedLine(corners[5], corners[6]);
        DrawSegmentedLine(corners[6], corners[7]);
        DrawSegmentedLine(corners[7], corners[4]);

        DrawSegmentedLine(corners[0], corners[4]);
        DrawSegmentedLine(corners[1], corners[5]);
        DrawSegmentedLine(corners[2], corners[6]);
        DrawSegmentedLine(corners[3], corners[7]);
    };

    // Axis-aligned wrapper for non-mesh selectables (VDB, camera, force field, etc.)
    auto DrawBoundingBox = [&](Vec3 bb_min, Vec3 bb_max, ImU32 color, float thickness) {
        Vec3 corners[8] = {
            Vec3(bb_min.x, bb_min.y, bb_min.z),
            Vec3(bb_max.x, bb_min.y, bb_min.z),
            Vec3(bb_max.x, bb_max.y, bb_min.z),
            Vec3(bb_min.x, bb_max.y, bb_min.z),
            Vec3(bb_min.x, bb_min.y, bb_max.z),
            Vec3(bb_max.x, bb_min.y, bb_max.z),
            Vec3(bb_max.x, bb_max.y, bb_max.z),
            Vec3(bb_min.x, bb_max.y, bb_max.z),
        };
        DrawOrientedBox(corners, color, thickness);
    };

    // Projection for hull + per-vertex raster outline (ortho-aware). Inlined with the
    // precomputed camera basis — this is called per mesh vertex on dense selections, so it
    // must not re-normalize the basis every call.
    auto ProjectWorldPoint = [&](const Vec3& point, ImVec2& out) -> bool {
        Vec3 to_pt = point - cam.lookfrom;
        float depth = to_pt.dot(cam_forward);
        if (!gizmoOrtho && depth <= 0.01f) return false;
        float local_x = to_pt.dot(cam_right);
        float local_y = to_pt.dot(cam_up);
        float half_h = gizmoOrtho ? (((cam.ortho_height > 1e-4f) ? cam.ortho_height : 10.0f) * 0.5f)
                                  : (depth * tan_half_fov);
        float half_w = half_h * aspect_ratio;
        if (fabsf(half_w) <= 1e-6f || fabsf(half_h) <= 1e-6f) return false;
        out.x = ((local_x / half_w) * 0.5f + 0.5f) * screen_w;
        out.y = (0.5f - (local_y / half_h) * 0.5f) * screen_h;
        return true;
    };

    // Screen-space convex hull of an object's transformed mesh vertices.
    // Returns true if hull was drawn; false → caller falls back to OBB.
    // ─── CPU rasterized outline ──────────────────────────────
    // Per-frame: project mesh, back-face cull, scanline-rasterize into a screen-space
    // mask covering only the object's projected bbox, then draw the mask boundary.
    // No cache — selection moves and rotates freely with the object.
    auto DrawSelectionRaster = [&](const std::string& name, ImU32 color, float thickness) -> bool {
       
        auto mit = mesh_cache.find(name);
        if (mit == mesh_cache.end() || mit->second.empty()) return false;
        const auto& tris = mit->second;

        // Cost cap. Above this the per-frame projection cost is too high for
        // a full per-triangle raster, and triangle subsampling (stride) cannot
        // robustly preserve the silhouette — kept triangles are not spatially
        // adjacent in mesh order, so screen-space gaps leak through the outer
        // boundary at close zoom and confuse the flood-fill at far zoom.
        // Above the cap we fall back to the cached convex hull. A proper fix
        // for very dense meshes would be a GPU outline pass or a one-time
        // decimated proxy cached per selection — out of scope here.
        TriangleMesh* pm = (!tris.empty() && tris[0].second->parentMesh) ? tris[0].second->parentMesh.get() : nullptr;
        const int64_t tri_count = pm ? static_cast<int64_t>(pm->geometry->indices.size() / 3) : static_cast<int64_t>(tris.size());
        constexpr size_t MAX_TRIS_FOR_RASTER = 1000000;
        if (static_cast<size_t>(tri_count) > MAX_TRIS_FOR_RASTER) return false;
        constexpr size_t tri_stride = 1;

        // pose_hash is hoisted to function scope so it can feed the per-frame
        // drawcall cache below regardless of whether this mesh is skinned.
        uint64_t pose_hash = 0;

        // Use the fully-composed transform (base * current * pivot) — same matrix the
        // renderer feeds to updateTransformedVertices. transform->base alone misses
        // gizmo-driven moves which write to the pivot component.
        Matrix4x4 m = tris[0].second->getTransformMatrix();

        // "Truly skinned" = has skin data AND at least one non-empty bone weight.
        // Assimp sometimes attaches empty SkinnedTriangleData to rigid meshes; those
        // would falsely take the skinned path. The skinned path reads
        // vertices[i].position which is only refreshed by apply_skinning() —
        // without that, gizmo drag doesn't update it and the outline freezes.
        // Treat such "spurious-skin" meshes as rigid (original + M) so gizmo
        // tracking works without depending on the animation system running.
        bool mesh_is_skinned = tris[0].second->hasAnySkinWeights();
        if (mesh_is_skinned && !ctx.renderer.finalBoneMatrices.empty()) {
            // Pose-hash cache: a CPU skinning pass over a high-poly skinned mesh
            // costs more than the rest of the outline path combined. Triangle
            // vertices[i].position persists between frames, so if the bone buffer
            // hasn't changed since the last raster we can reuse last frame's
            // skinned positions verbatim. The hash mixes size + a sparse sample of
            // matrix entries (collisions don't matter — only a held-pose match
            // matters, and a real pose change shifts dozens of floats).
            const auto& bones =
                static_cast<const std::vector<Matrix4x4>&>(ctx.renderer.finalBoneMatrices);
            pose_hash = 1469598103934665603ull ^ bones.size();
            const size_t step = (bones.size() > 16) ? (bones.size() / 16) : 1;
            for (size_t bi = 0; bi < bones.size(); bi += step) {
                const float* mm = &bones[bi].m[0][0];
                // Sample translation + a couple of rotation entries — cheap and
                // changes on any meaningful pose update.
                uint32_t a, b, c, d;
                std::memcpy(&a, mm + 3, 4);
                std::memcpy(&b, mm + 7, 4);
                std::memcpy(&c, mm + 11, 4);
                std::memcpy(&d, mm + 0, 4);
                pose_hash ^= (uint64_t(a) << 32) | b;
                pose_hash *= 1099511628211ull;
                pose_hash ^= (uint64_t(c) << 32) | d;
                pose_hash *= 1099511628211ull;
            }

            auto pit = selection_skin_pose_hash.find(name);
            const bool pose_unchanged =
                (pit != selection_skin_pose_hash.end()) && (pit->second == pose_hash);
            if (!pose_unchanged) {
                // Skin only the triangles the projection pass will actually visit
                // (matches tri_stride below). Triangles in between keep their
                // previous-frame skinned position — fine for the outline, since
                // the rasterizer doesn't see them either and the closing pass
                // below stitches the silhouette back together.
                for (size_t i = 0; i < tris.size(); i += tri_stride) {
                    if (tris[i].second->hasSkinData()) {
                        tris[i].second->apply_skinning(bones);
                    }
                }
                selection_skin_pose_hash[name] = pose_hash;
            }
        }
        // For rigid meshes back-face cull happens in local space (cam→local once).
        // For skinned meshes vertices are already world-space, so we cull in world.
        const Vec3 cull_origin = (!mesh_is_skinned)
            ? m.inverse().transform_point(cam.lookfrom)
            : cam.lookfrom;

        if (!std::isfinite(screen_w) || !std::isfinite(screen_h) || screen_w < 2.0f || screen_h < 2.0f) {
            return false;
        }
        const int sw = static_cast<int>(screen_w);
        const int sh = static_cast<int>(screen_h);

        // Per-frame drawcall cache: when camera + transform + pose + screen
        // + style are all unchanged from last frame, every pixel we'd compute
        // below is identical. Skip projection, raster, flood-fill, and per-
        // boundary BVH occlusion entirely — replay the stored pixel list.
        // Held-camera frames cost a vector walk instead of an O(tris + mask)
        // CPU pass that was starving the GPU command queue.
        extern bool g_bvh_rebuild_pending;
        // Disable CPU BVH occlusion probe during sequence render — the worker
        // thread mutates HittableInstance transforms / bounds per animation
        // frame, and Embree's intersection callbacks would otherwise race
        // those reads and crash inside embree4.dll.
        const bool can_occlude = (ctx.scene.bvh != nullptr) && !g_bvh_rebuild_pending && !ctx.is_animation_mode;

        uint64_t frame_hash = 1469598103934665603ull;
        auto mix64 = [&](uint64_t v) { frame_hash ^= v; frame_hash *= 1099511628211ull; };
        auto mixf  = [&](float f)    { uint32_t x; std::memcpy(&x, &f, 4); mix64(x); };
        for (int rr = 0; rr < 3; ++rr) for (int cc = 0; cc < 4; ++cc) mixf(m.m[rr][cc]);
        mixf(cam.lookfrom.x); mixf(cam.lookfrom.y); mixf(cam.lookfrom.z);
        mixf(cam_forward.x);  mixf(cam_forward.y);  mixf(cam_forward.z);
        mixf(cam_right.x);    mixf(cam_right.y);    mixf(cam_right.z);
        mixf(cam_up.x);       mixf(cam_up.y);       mixf(cam_up.z);
        mixf(tan_half_fov); mixf(aspect_ratio);
        // Ortho zoom changes ONLY ortho_height (lookfrom/basis/fov stay put), so it must be
        // part of the hash or scroll-zoom replays a stale outline at the old screen size.
        mix64(gizmoOrtho ? 1ull : 0ull);
        mixf(gizmoOrtho ? cam.ortho_height : 0.0f);
        mix64(static_cast<uint64_t>(sw)); mix64(static_cast<uint64_t>(sh));
        mixf(thickness); mix64(static_cast<uint64_t>(color));
        mix64(pose_hash);
        mix64(can_occlude ? 1ull : 0ull);

        auto cit = selection_outline_frame_cache.find(name);
        if (cit != selection_outline_frame_cache.end() &&
            cit->second.hash == frame_hash &&
            !cit->second.runs.empty()) {
            ImDrawList* dl = ImGui::GetBackgroundDrawList();
            const float pix = static_cast<float>(cit->second.scale);
            const float half = cit->second.thickness * 0.5f;
            for (const auto& run : cit->second.runs) {
                const float run_w = static_cast<float>(run.len) * pix;
                dl->AddRectFilled(
                    ImVec2(run.sx - half + 0.5f, run.sy - half + 0.5f),
                    ImVec2(run.sx + half + 0.5f + run_w, run.sy + half + 0.5f + pix),
                    run.col);
            }
            return true;
        }

        // Cache miss — recompute below and refill.
        SelectionOutlineFrameCache& cache_entry = selection_outline_frame_cache[name];
        cache_entry.runs.clear();
        cache_entry.hash = frame_hash;

        struct ProjTri { float x[3], y[3]; float mean_depth; bool valid; };
        std::vector<ProjTri> projected(tri_count);

        int min_x = sw, min_y = sh, max_x = -1, max_y = -1;
        bool has_any = false;

        // Parallel projection: each iteration only touches its own projected[i]
        // slot (read-only mesh/camera captures otherwise), so this is safe to
        // fan out across cores. Legacy raw-Triangle (non-flat/SoA)
        // meshes push every real triangle into `tris`, unlike flat/proxy
        // TriangleMesh objects which only ever expose a single facade here —
        // for those dense raw meshes this per-triangle projection was the
        // single-threaded cost that pegged one core while orbiting the camera
        // with the object selected. The bbox/has_any reduction happens in a
        // cheap serial pass below instead of inside the parallel region.
        #pragma omp parallel for schedule(dynamic, 4096) if(tri_count > 8192)
        for (int64_t i = 0; i < tri_count; i += static_cast<int64_t>(tri_stride)) {
            ProjTri& pt = projected[i];
            pt.valid = false;
            
            Vec3 a, b, c;
            if (pm) {
                const auto& idx = pm->geometry->indices;
                const Vec3* P = nullptr;
                if (mesh_is_skinned) {
                    P = pm->geometry->get_positions();
                } else {
                    P = pm->geometry->get_positions_orig();
                    if (!P) P = pm->geometry->get_positions();
                }
                a = P[idx[i * 3 + 0]];
                b = P[idx[i * 3 + 1]];
                c = P[idx[i * 3 + 2]];
            } else {
                const Triangle& T = *tris[i].second;
                a = mesh_is_skinned ? T.getVertexPosition(0) : T.getOriginalVertexPosition(0);
                b = mesh_is_skinned ? T.getVertexPosition(1) : T.getOriginalVertexPosition(1);
                c = mesh_is_skinned ? T.getVertexPosition(2) : T.getOriginalVertexPosition(2);
            }

            // Back-face cull (local frame for rigid, world frame for skinned).
            Vec3 e1(b.x - a.x, b.y - a.y, b.z - a.z);
            Vec3 e2(c.x - a.x, c.y - a.y, c.z - a.z);
            Vec3 n(
                e1.y * e2.z - e1.z * e2.y,
                e1.z * e2.x - e1.x * e2.z,
                e1.x * e2.y - e1.y * e2.x
            );
            Vec3 view(
                (a.x + b.x + c.x) * (1.0f / 3.0f) - cull_origin.x,
                (a.y + b.y + c.y) * (1.0f / 3.0f) - cull_origin.y,
                (a.z + b.z + c.z) * (1.0f / 3.0f) - cull_origin.z
            );
            if (view.x * n.x + view.y * n.y + view.z * n.z >= 0.0f) {
                continue; // back-facing
            }

            ImVec2 p0, p1, p2;
            const Vec3* lv[3] = { &a, &b, &c };
            bool all_ok = true;
            ImVec2* outp[3] = { &p0, &p1, &p2 };
            float depth_sum = 0.0f;
            for (int v = 0; v < 3; ++v) {
                const Vec3& vp = *lv[v];
                Vec3 wp;
                if (mesh_is_skinned) {
                    wp = vp; // already world space
                } else {
                    wp = Vec3(
                        m.m[0][0] * vp.x + m.m[0][1] * vp.y + m.m[0][2] * vp.z + m.m[0][3],
                        m.m[1][0] * vp.x + m.m[1][1] * vp.y + m.m[1][2] * vp.z + m.m[1][3],
                        m.m[2][0] * vp.x + m.m[2][1] * vp.y + m.m[2][2] * vp.z + m.m[2][3]
                    );
                }
                if (!ProjectWorldPoint(wp, *outp[v])) { all_ok = false; break; }
                Vec3 to_pt = wp - cam.lookfrom;
                depth_sum += to_pt.dot(cam_forward);
            }
            if (!all_ok) continue;

            pt.x[0] = p0.x; pt.y[0] = p0.y;
            pt.x[1] = p1.x; pt.y[1] = p1.y;
            pt.x[2] = p2.x; pt.y[2] = p2.y;
            pt.mean_depth = depth_sum * (1.0f / 3.0f);
            pt.valid = true;
        }

        // Serial bbox/has_any reduction — cheap (just int compares) next to the
        // projection math above, so it doesn't need its own parallel reduction.
        for (int64_t i = 0; i < tri_count; i += static_cast<int64_t>(tri_stride)) {
            const ProjTri& pt = projected[i];
            if (!pt.valid) continue;
            has_any = true;
            for (int v = 0; v < 3; ++v) {
                int xi = static_cast<int>(floorf(pt.x[v]));
                int yi = static_cast<int>(floorf(pt.y[v]));
                if (xi < min_x) min_x = xi;
                if (yi < min_y) min_y = yi;
                if (xi > max_x) max_x = xi;
                if (yi > max_y) max_y = yi;
            }
        }
        if (!has_any) return false;

        min_x = std::max(0, min_x - 2);
        min_y = std::max(0, min_y - 2);
        max_x = std::min(sw - 1, max_x + 2);
        max_y = std::min(sh - 1, max_y + 2);
        int mw = max_x - min_x + 1;
        int mh = max_y - min_y + 1;
        if (mw <= 2 || mh <= 2) return false;

        // Cost cap on mask area. Very large projected selections fall back to
        // hull/bbox instead of trying to allocate a huge per-pixel mask.
        int scale = 1;
        constexpr size_t MAX_SELECTION_MASK_CELLS = 2000000;
        size_t mask_area = static_cast<size_t>(mw) * static_cast<size_t>(mh);
        while (mask_area > MAX_SELECTION_MASK_CELLS && scale < 8) {
            scale *= 2;
            mw = (mw + 1) / 2;
            mh = (mh + 1) / 2;
            mask_area = static_cast<size_t>(mw) * static_cast<size_t>(mh);
        }
        if (mask_area > MAX_SELECTION_MASK_CELLS) {
            return false;
        }

        // depth_mask: per-pixel min camera-space depth of the rasterized object.
        // 1e30f → empty. Used both as a fill bit (via < 1e29f) and as the depth
        // reference for the per-boundary occlusion test below.
        constexpr float EMPTY_DEPTH = 1e30f;
        std::vector<float> depth_mask(static_cast<size_t>(mw) * static_cast<size_t>(mh), EMPTY_DEPTH);
        const float inv_scale = 1.0f / static_cast<float>(scale);
        const float ox = static_cast<float>(min_x);
        const float oy = static_cast<float>(min_y);

        // Rasterize each surviving triangle into the mask. Flat/SoA nodes keep
        // only ONE representative facade in `tris`; their real faces were
        // projected from parentMesh above and therefore must be consumed using
        // `tri_count`. Iterating tris.size() here reduced every flat selection
        // to the representative face whenever the GPU outline was unavailable.
        for (int64_t i = 0; i < tri_count; i += static_cast<int64_t>(tri_stride)) {
            const ProjTri& pt = projected[i];
            if (!pt.valid) continue;
            // Map to mask space.
            float x0 = (pt.x[0] - ox) * inv_scale;
            float y0 = (pt.y[0] - oy) * inv_scale;
            float x1 = (pt.x[1] - ox) * inv_scale;
            float y1 = (pt.y[1] - oy) * inv_scale;
            float x2 = (pt.x[2] - ox) * inv_scale;
            float y2 = (pt.y[2] - oy) * inv_scale;

            float fminx = std::min(std::min(x0, x1), x2);
            float fminy = std::min(std::min(y0, y1), y2);
            float fmaxx = std::max(std::max(x0, x1), x2);
            float fmaxy = std::max(std::max(y0, y1), y2);
            int bx0 = std::max(0, static_cast<int>(floorf(fminx)));
            int by0 = std::max(0, static_cast<int>(floorf(fminy)));
            int bx1 = std::min(mw - 1, static_cast<int>(ceilf(fmaxx)));
            int by1 = std::min(mh - 1, static_cast<int>(ceilf(fmaxy)));
            if (bx1 < bx0 || by1 < by0) continue;

            float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
            if (fabsf(area) < 1e-6f) continue;
            const float sign = (area < 0.0f) ? -1.0f : 1.0f;

            const float tri_depth = pt.mean_depth;
            for (int y = by0; y <= by1; ++y) {
                const float py = static_cast<float>(y) + 0.5f;
                float* row = depth_mask.data() + static_cast<size_t>(y) * mw;
                for (int x = bx0; x <= bx1; ++x) {
                    const float px = static_cast<float>(x) + 0.5f;
                    float w0 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
                    float w1 = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2);
                    float w2 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0);
                    if ((w0 * sign) >= 0.0f && (w1 * sign) >= 0.0f && (w2 * sign) >= 0.0f) {
                        // Keep the closest triangle covering this pixel — depth is
                        // what the occlusion test below compares against.
                        if (tri_depth < row[x]) row[x] = tri_depth;
                    }
                }
            }
        }

        // Flood-fill from the mask border to classify each empty pixel as either
        // "outside" (reachable from the border through empty space) or "interior
        // hole" (an unfilled pocket between sampled triangles). The silhouette
        // is then only filled pixels adjacent to outside-empty — internal stride
        // holes never bleed through no matter how large they grow at close zoom.
        // Cost is O(mask_area) regardless of triangle count or hole width.
        std::vector<uint8_t> outside(static_cast<size_t>(mw) * static_cast<size_t>(mh), 0);
        {
            std::vector<int> stack;
            stack.reserve(4096);
            auto try_push = [&](int x, int y) {
                if (x < 0 || x >= mw || y < 0 || y >= mh) return;
                size_t idx = static_cast<size_t>(y) * mw + x;
                if (outside[idx]) return;
                if (depth_mask[idx] < EMPTY_DEPTH * 0.5f) return; // filled
                outside[idx] = 1;
                stack.push_back(static_cast<int>(idx));
            };
            for (int x = 0; x < mw; ++x) { try_push(x, 0); try_push(x, mh - 1); }
            for (int y = 0; y < mh; ++y) { try_push(0, y); try_push(mw - 1, y); }
            while (!stack.empty()) {
                int idx = stack.back(); stack.pop_back();
                int y = idx / mw;
                int x = idx - y * mw;
                try_push(x - 1, y);
                try_push(x + 1, y);
                try_push(x, y - 1);
                try_push(x, y + 1);
            }
        }

        // 4-neighbor boundary: a filled pixel with at least one unfilled neighbor.
        // For each boundary pixel we also test scene-BVH occlusion: cast a ray
        // through the pixel and if something else is closer than the stored
        // object depth, draw that pixel in a desaturated gray-tint so the user
        // sees exactly which portion of the silhouette is behind another object.
        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        const float pix = static_cast<float>(scale);
        const float half = thickness * 0.5f;

        const ImU32 visible_col = color;
        const int oc_r = ((( color       ) & 0xFF) + 160) >> 1;
        const int oc_g = ((( color >>  8) & 0xFF) + 160) >> 1;
        const int oc_b = ((( color >> 16) & 0xFF) + 160) >> 1;
        const ImU32 occluded_col = IM_COL32(oc_r, oc_g, oc_b, 120);

        // can_occlude hoisted above for the frame-cache key.

        // Camera basis for ray reconstruction at a screen pixel.
        const float view_h = tan_half_fov;
        const float view_w = view_h * aspect_ratio;
        const float inv_sw = 1.0f / static_cast<float>(sw);
        const float inv_sh = 1.0f / static_cast<float>(sh);

        // Occlusion sample stride: BVH cast is the only meaningful per-pixel
        // cost and was the dominant gizmo bottleneck on dense meshes — it
        // serializes the main thread and starves the GPU command queue.
        // Bucket by triangle count; neighbours inherit the result so the
        // perceived occlusion edge stays within a few pixels of truth.
        int oc_stride;
        if      (tri_count > 200000) oc_stride = 16;
        else if (tri_count >  30000) oc_stride = 10;
        else                           oc_stride = 6;
        bool last_occluded = false;
        int  last_oc_x = -9999, last_oc_y = -9999;

        size_t drawn = 0;
        for (int y = 0; y < mh; ++y) {
            const float*   row    = depth_mask.data() + static_cast<size_t>(y) * mw;
            const uint8_t* o_row  = outside.data()    + static_cast<size_t>(y) * mw;
            const uint8_t* o_up   = (y > 0)      ? o_row - mw : nullptr;
            const uint8_t* o_down = (y < mh - 1) ? o_row + mw : nullptr;

            // RLE state for this row: collect consecutive same-colour boundary
            // pixels into a single wide rect at flush time.
            int  run_start = -1;
            ImU32 run_col  = 0;
            const float sy_row = oy + static_cast<float>(y) * pix;

            auto flush_run = [&](int run_end_exclusive) {
                if (run_start < 0) return;
                const float sx0   = ox + static_cast<float>(run_start) * pix;
                const float run_w = static_cast<float>(run_end_exclusive - run_start) * pix;
                draw_list->AddRectFilled(
                    ImVec2(sx0 - half + 0.5f,         sy_row - half + 0.5f),
                    ImVec2(sx0 + half + 0.5f + run_w, sy_row + half + 0.5f + pix),
                    run_col);
                const uint16_t len = static_cast<uint16_t>(
                    std::min<int>(run_end_exclusive - run_start, 0xFFFF));
                cache_entry.runs.push_back({ sx0, sy_row, len, static_cast<uint32_t>(run_col) });
                ++drawn;
                run_start = -1;
            };

            for (int x = 0; x < mw; ++x) {
                const float d_here = row[x];
                const bool  is_filled = (d_here < EMPTY_DEPTH * 0.5f);
                const bool  boundary  = is_filled && (
                    (x == 0) || (x == mw - 1) || (y == 0) || (y == mh - 1) ||
                    o_row[x - 1] || o_row[x + 1] ||
                    o_up[x] || o_down[x]);
                if (!boundary) { flush_run(x); continue; }

                const float sx = ox + static_cast<float>(x) * pix;

                ImU32 px_col = visible_col;
                if (can_occlude) {
                    bool occluded;
                    // Reuse the previous BVH result if we're inside the stride
                    // window — boundaries change occlusion state slowly.
                    int dx = x - last_oc_x;
                    int dy = y - last_oc_y;
                    if (dx * dx + dy * dy < oc_stride * oc_stride) {
                        occluded = last_occluded;
                    } else {
                        float ndc_x = (sx * inv_sw) * 2.0f - 1.0f;
                        float ndc_y = 1.0f - (sy_row * inv_sh) * 2.0f;
                        Vec3 dir = (cam_right * (ndc_x * view_w)
                                  + cam_up    * (ndc_y * view_h)
                                  + cam_forward).normalize();
                        Ray r(cam.lookfrom, dir);
                        HitRecord rec;
                        // Convert object depth (along forward) to ray-t along dir.
                        const float t_obj = d_here / dir.dot(cam_forward);
                        occluded = ctx.scene.bvh->hit(r, 0.001f, t_obj - 0.05f, rec, true);
                        last_occluded = occluded;
                        last_oc_x = x;
                        last_oc_y = y;
                    }
                    if (occluded) px_col = occluded_col;
                }

                if (run_start < 0) {
                    run_start = x; run_col = px_col;
                } else if (px_col != run_col) {
                    flush_run(x);
                    run_start = x; run_col = px_col;
                }
            }
            flush_run(mw);
        }
        cache_entry.thickness = thickness;
        cache_entry.scale = scale;
        return drawn > 0;
    };


    // Build a small set of local-space extremal points for a mesh by argmax-ing
    // along K well-distributed directions. Result approximates the 3D convex hull
    // vertex set tightly enough for a screen-space 2D hull to be visually identical.
    auto ExtractHullCandidates = [&](const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& tris,
                                     std::vector<Vec3>& out) {
        // Fibonacci-sphere directions: uniform spherical sampling for tight 3D convex
        // hull approximation. K=128 gives smooth silhouette on organic shapes with
        // ~50ms one-time cost on a 150k-vertex mesh.
        constexpr int K = 128;
        static std::array<Vec3, K> dirs = [] {
            std::array<Vec3, K> d{};
            const float golden = 3.14159265359f * (3.0f - sqrtf(5.0f));
            for (int i = 0; i < K; ++i) {
                float y = 1.0f - (float(i) / float(K - 1)) * 2.0f; // [1, -1]
                float r = sqrtf(fmaxf(0.0f, 1.0f - y * y));
                float theta = golden * float(i);
                d[i] = Vec3(cosf(theta) * r, y, sinf(theta) * r);
            }
            return d;
        }();

        std::array<float, K> bestDot;
        std::array<Vec3, K> bestPt;
        for (int k = 0; k < K; ++k) bestDot[k] = -1e30f;

        TriangleMesh* pm = (!tris.empty() && tris[0].second->parentMesh) ? tris[0].second->parentMesh.get() : nullptr;
        if (pm) {
            const Vec3* origP = pm->geometry->get_positions_orig();
            if (!origP) origP = pm->geometry->get_positions();
            if (origP) {
                size_t vCount = pm->geometry->get_vertex_count();
                for (size_t v = 0; v < vCount; ++v) {
                    const Vec3& p = origP[v];
                    for (int k = 0; k < K; ++k) {
                        float d = p.x * dirs[k].x + p.y * dirs[k].y + p.z * dirs[k].z;
                        if (d > bestDot[k]) { bestDot[k] = d; bestPt[k] = p; }
                    }
                }
            }
        } else {
            for (const auto& tp : tris) {
                const Triangle& T = *tp.second;
                for (int v = 0; v < 3; ++v) {
                    const Vec3& p = T.getOriginalVertexPosition(v);
                    for (int k = 0; k < K; ++k) {
                        float d = p.x * dirs[k].x + p.y * dirs[k].y + p.z * dirs[k].z;
                        if (d > bestDot[k]) { bestDot[k] = d; bestPt[k] = p; }
                    }
                }
            }
        }

        // Dedupe (different dirs often pick the same vertex on flat regions).
        out.clear();
        out.reserve(K);
        for (int k = 0; k < K; ++k) {
            if (bestDot[k] <= -1e29f) continue;
            bool dup = false;
            for (const Vec3& q : out) {
                if (fabsf(q.x - bestPt[k].x) < 1e-5f &&
                    fabsf(q.y - bestPt[k].y) < 1e-5f &&
                    fabsf(q.z - bestPt[k].z) < 1e-5f) { dup = true; break; }
            }
            if (!dup) out.push_back(bestPt[k]);
        }
    };

    auto DrawSelectionHull = [&](const std::string& name, ImU32 color, float thickness) -> bool {
        auto it = mesh_cache.find(name);
        if (it == mesh_cache.end() || it->second.empty()) return false;

        // Fully composed transform (includes gizmo-driven pivot, not just base).
        Matrix4x4 m = it->second[0].second->getTransformMatrix();

        // Lazy candidate extraction (one-time per mesh; ~ms even for 500k tris).
        auto cand_it = hull_candidate_cache.find(name);
        if (cand_it == hull_candidate_cache.end()) {
            std::vector<Vec3> candidates;
            ExtractHullCandidates(it->second, candidates);
            cand_it = hull_candidate_cache.emplace(name, std::move(candidates)).first;
        }
        const std::vector<Vec3>& local_pts = cand_it->second;
        if (local_pts.size() < 3) return false;

        std::vector<ImVec2> pts;
        pts.reserve(local_pts.size());
        for (const Vec3& lp : local_pts) {
            Vec3 wp(
                m.m[0][0] * lp.x + m.m[0][1] * lp.y + m.m[0][2] * lp.z + m.m[0][3],
                m.m[1][0] * lp.x + m.m[1][1] * lp.y + m.m[1][2] * lp.z + m.m[1][3],
                m.m[2][0] * lp.x + m.m[2][1] * lp.y + m.m[2][2] * lp.z + m.m[2][3]
            );
            ImVec2 sp;
            if (ProjectWorldPoint(wp, sp)) pts.push_back(sp);
        }
        if (pts.size() < 3) return false;

        // Andrew monotone chain convex hull.
        std::sort(pts.begin(), pts.end(), [](const ImVec2& a, const ImVec2& b) {
            return a.x < b.x || (a.x == b.x && a.y < b.y);
        });
        auto cross2d = [](const ImVec2& O, const ImVec2& A, const ImVec2& B) {
            return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
        };
        const int n = static_cast<int>(pts.size());
        std::vector<ImVec2> hull(2 * n);
        int k = 0;
        for (int i = 0; i < n; ++i) {
            while (k >= 2 && cross2d(hull[k - 2], hull[k - 1], pts[i]) <= 0.0f) --k;
            hull[k++] = pts[i];
        }
        for (int i = n - 2, t = k + 1; i >= 0; --i) {
            while (k >= t && cross2d(hull[k - 2], hull[k - 1], pts[i]) <= 0.0f) --k;
            hull[k++] = pts[i];
        }
        if (k <= 1) return false;
        hull.resize(k - 1); // drop duplicated closing vertex
        if (hull.size() < 3) return false;

        // Reject degenerate hulls (sub-pixel): caller falls back to OBB.
        float hmin_x = hull[0].x, hmax_x = hull[0].x, hmin_y = hull[0].y, hmax_y = hull[0].y;
        for (const auto& p : hull) {
            hmin_x = fminf(hmin_x, p.x); hmax_x = fmaxf(hmax_x, p.x);
            hmin_y = fminf(hmin_y, p.y); hmax_y = fmaxf(hmax_y, p.y);
        }
        if ((hmax_x - hmin_x) < 2.0f && (hmax_y - hmin_y) < 2.0f) return false;

        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        const size_t H = hull.size();
        for (size_t i = 0; i < H; ++i) {
            draw_list->AddLine(hull[i], hull[(i + 1) % H], color, thickness);
        }
        return true;
    };

    // �������������������������������������������������������������������������
    // DRAW VDB GHOST BOUNDS (Always visible)
    // �������������������������������������������������������������������������
    for (const auto& vdb : ctx.scene.vdb_volumes) {
        // Skip if this VDB is currently selected (will be drawn with highlight)
        if (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume == vdb) continue;
        
        AABB bounds = vdb->getWorldBounds();
        // Ghost outline - visible but not obtrusive (Alpha 100)
        DrawBoundingBox(bounds.min, bounds.max, IM_COL32(180, 180, 180, 100), 1.0f);
    }

    // �������������������������������������������������������������������������
    // DRAW GAS VOLUME BOUNDS (Smoke/Fire Simulation) - With Picking Support
    // �������������������������������������������������������������������������
    for (const auto& gas : ctx.scene.gas_volumes) {
        if (!gas || !gas->visible) continue;
        
        Vec3 bounds_min, bounds_max;
        gas->getWorldBounds(bounds_min, bounds_max);
        
        // Ray-AABB Intersection for Picking
        // ----------------------------------------------------
        // 1. Generate Ray from Camera through Mouse position
        float mouse_x = io.MousePos.x;
        float mouse_y = io.MousePos.y;
        
        // Convert screen space to NDC (-1 to 1)
        float ndc_x = (mouse_x / screen_w) * 2.0f - 1.0f;
        float ndc_y = 1.0f - (mouse_y / screen_h) * 2.0f; // Flip Y
        
        // Ray direction in camera space
        float view_h = tan_half_fov;
        float view_w = view_h * aspect_ratio;
        
        Vec3 ray_dir_cam(ndc_x * view_w, ndc_y * view_h, -1.0f);
        ray_dir_cam = ray_dir_cam.normalize();
        
        // Transform ray direction to world space
        // cam_right, cam_up, -cam_forward are the basis vectors
        Vec3 ray_dir_world = (cam_right * ray_dir_cam.x + cam_up * ray_dir_cam.y + cam_forward * ray_dir_cam.z).normalize();
        
        // 2. Perform Ray-AABB Intersection
        Vec3 inv_dir(1.0f / ray_dir_world.x, 1.0f / ray_dir_world.y, 1.0f / ray_dir_world.z);
        
        float t1 = (bounds_min.x - cam.lookfrom.x) * inv_dir.x;
        float t2 = (bounds_max.x - cam.lookfrom.x) * inv_dir.x;
        float t3 = (bounds_min.y - cam.lookfrom.y) * inv_dir.y;
        float t4 = (bounds_max.y - cam.lookfrom.y) * inv_dir.y;
        float t5 = (bounds_min.z - cam.lookfrom.z) * inv_dir.z;
        float t6 = (bounds_max.z - cam.lookfrom.z) * inv_dir.z;
        
        float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
        float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));
        
        // If tmax < 0, ray (line) is intersecting AABB, but whole AABB is behind us
        // If tmin > tmax, ray doesn't intersect AABB
        if (tmax >= 0 && tmin <= tmax) {
             // Intersection found!
             if (ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver()) {
                // Use SceneSelection!
                ctx.selection.selectGasVolume(gas, -1, gas->name);
                GasUI::selected_gas_volume = gas; // Keep sync for now if needed, but rely on ctx
            }
        }
        
        // Check if selected using Scene Selection
        bool is_selected = (
            ctx.selection.selected.type == SelectableType::GasVolume && 
            ctx.selection.selected.gas_volume == gas
        );
        
        if (is_selected) {
            // Bright cyan for selected gas volume
            DrawBoundingBox(bounds_min, bounds_max, IM_COL32(0, 200, 255, 255), 2.0f);
            
            // Draw emitter wireframes for selected gas volume
            const auto& emitters = gas->getEmitters();
            for (const auto& e : emitters) {
                if (!e.enabled) continue;
                
                Vec3 epos = bounds_min + e.position;  // Emitter pos relative to grid origin
                
                if (e.shape == FluidSim::EmitterShape::Sphere) {
                    // Draw sphere as circle gizmo (8 segments)
                    float r = e.radius;
                    ImU32 emit_col = IM_COL32(255, 100, 50, 200);
                    
                    // Draw in 3 planes
                    for (int plane = 0; plane < 3; ++plane) {
                        for (int i = 0; i < 8; ++i) {
                            float a0 = i * (6.28318f / 8.0f);
                            float a1 = (i + 1) * (6.28318f / 8.0f);
                            Vec3 p0, p1;
                            
                            if (plane == 0) { // XY
                                p0 = epos + Vec3(cosf(a0) * r, sinf(a0) * r, 0);
                                p1 = epos + Vec3(cosf(a1) * r, sinf(a1) * r, 0);
                            } else if (plane == 1) { // XZ
                                p0 = epos + Vec3(cosf(a0) * r, 0, sinf(a0) * r);
                                p1 = epos + Vec3(cosf(a1) * r, 0, sinf(a1) * r);
                            } else { // YZ
                                p0 = epos + Vec3(0, cosf(a0) * r, sinf(a0) * r);
                                p1 = epos + Vec3(0, cosf(a1) * r, sinf(a1) * r);
                            }
                            
                            DrawBoundingBox(p0 - Vec3(0.02f), p0 + Vec3(0.02f), emit_col, 1.0f);
                        }
                    }
                } else if (e.shape == FluidSim::EmitterShape::Box) {
                    // Draw box emitter
                    Vec3 half = e.size * 0.5f;
                    Vec3 emin = epos - half;
                    Vec3 emax = epos + half;
                    DrawBoundingBox(emin, emax, IM_COL32(255, 150, 50, 200), 1.5f);
                } else {
                    // Point emitter - small box
                    DrawBoundingBox(epos - Vec3(0.1f), epos + Vec3(0.1f), IM_COL32(255, 50, 50, 255), 1.5f);
                }
            }
        } else {
            // Ghost outline for unselected gas volumes - purple/magenta tint
            DrawBoundingBox(bounds_min, bounds_max, IM_COL32(180, 100, 200, 100), 1.0f);
        }
    }

    // �������������������������������������������������������������������������
    // DRAW SELECTION HIGHLIGHTS
    // �������������������������������������������������������������������������
    // GPU outline gate: in raster viewport modes the Vulkan overlay pass draws
    // the mesh silhouette (selected-instance mask + screen-space edge
    // composite) instead of the per-frame CPU raster below. Rendered mode
    // (shading_mode == 2) and CPU-only builds keep the ImGui fallback. Names
    // are collected in loop order so the primary (last) selection keeps its
    // brighter tier on the GPU side too.
    auto* gpu_outline_backend =
        dynamic_cast<Backend::VulkanBackendAdapter*>(getGizmoViewportBackend(ctx));
    // hasGpuSelectionOutline goes false when the SPIR-V is missing or the
    // pipelines/targets failed to build — the CPU outline below then keeps
    // working instead of the selection silently losing its highlight.
    const bool gpu_outline_eligible =
        gpu_outline_backend != nullptr && viewport_settings.shading_mode != 2 &&
        gpu_outline_backend->hasGpuSelectionOutline();
    Backend::SelectionOutlineParams gpu_outline_params;

    // Rendered-mode outline drawer: the Vulkan composite can't blend over the
    // CPU-presented path-traced image, so the backend renders the selection
    // mask standalone (scene depth prepass + selected silhouette), we read the
    // R8G8 mask back, scan its boundary into RLE runs and draw them as an
    // ImGui overlay. CPU cost scales with resolution, not polygon count, and
    // the run cache replays for free while the path tracer accumulates with a
    // held camera.
    auto DrawGpuMaskOutlineRuns = [&](const std::vector<std::pair<std::string, Matrix4x4>>& nodes,
                                      bool has_skinned,
                                      uint64_t state_hash, uint64_t xform_hash) -> bool {
        const int sw = (int)screen_w, sh = (int)screen_h;
        if (sw < 4 || sh < 4) return false;
        static const std::string kCacheKey = "\x01gpu_mask_outline";

        auto replayRuns = [&](const SelectionOutlineFrameCache& ce) {
            ImDrawList* dl = ImGui::GetBackgroundDrawList();
            const float pix = (float)ce.scale;
            const float half = ce.thickness * 0.5f;
            for (const auto& run : ce.runs) {
                dl->AddRectFilled(
                    ImVec2(run.sx - half + 0.5f, run.sy - half + 0.5f),
                    ImVec2(run.sx + half + 0.5f + (float)run.len * pix, run.sy + half + 0.5f + pix),
                    run.col);
            }
        };
        static uint64_t s_last_failed_hash = 0;
        // Adaptive resolution: while the state changes every frame
        // (drag/orbit) the mask renders at half res for speed; on the first
        // frame the state holds still, the cached coarse result is refined
        // once at full res so the resting outline is crisp.
        static uint64_t s_prev_state_hash = 0;
        const bool settled = (state_hash == s_prev_state_hash);
        s_prev_state_hash = state_hash;

        auto cit = selection_outline_frame_cache.find(kCacheKey);
        const bool cache_hit =
            cit != selection_outline_frame_cache.end() && cit->second.hash == state_hash;
        if (cache_hit) {
            const bool want_refine =
                settled && cit->second.scale > 1 && state_hash != s_last_failed_hash;
            if (!want_refine) {
                replayRuns(cit->second);
                return true;
            }
        } else if (state_hash == s_last_failed_hash) {
            // Failure memo: when the GPU path failed for this exact state
            // (e.g. missing SPIR-V), don't re-run the sync + mask attempt
            // every frame — the CPU outline handles it until something changes.
            return false;
        }
        // Churn throttle: while the state changes EVERY frame (skinned
        // animation playback, drag), recompute at most every other frame and
        // replay the one-frame-old runs in between — halves the per-frame
        // sync + mask + readback load next to a path-trace that's restarting
        // anyway.
        static int s_churn_cooldown = 0;
        if (!cache_hit && !settled && s_churn_cooldown > 0 &&
            cit != selection_outline_frame_cache.end() && !cit->second.runs.empty()) {
            --s_churn_cooldown;
            replayRuns(cit->second);
            return true;
        }
        const int use_scale = cache_hit ? 1 : 2; // refine pass : interactive pass
        const int mw = (std::max)(2, sw / use_scale);
        const int mh = (std::max)(2, sh / use_scale);

        // Rendered mode skips the per-frame raster sync that Solid mode does,
        // but only the SELECTED nodes' transforms matter for the mask — push
        // just those (O(selection), not a whole-scene walk: dragging a dense
        // object was paying a full syncRasterInstanceTransforms every frame).
        static uint64_t s_last_synced_xform_hash = 0;
        if (xform_hash != s_last_synced_xform_hash) {
            gpu_outline_backend->setRasterInstanceTransformsForNodes(nodes);
            if (has_skinned && !ctx.renderer.finalBoneMatrices.empty()) {
                // Virtual dispatch lands in VulkanViewportBackend's override:
                // GPU compute skinning of the raster vertex buffers (a cheap
                // dispatch — NOT the CPU base impl). User-verified this is
                // what keeps the skinned outline tracking in Rendered mode;
                // do NOT replace it with a CPU apply_bone_to_vertex pass
                // (tried once — skinned the whole mesh on the UI thread and
                // tanked performance).
                gpu_outline_backend->syncRasterSkinnedVertices(ctx.scene.world.objects,
                                                               ctx.renderer.finalBoneMatrices);
            }
            s_last_synced_xform_hash = xform_hash;
        }
        std::vector<std::string> names;
        names.reserve(nodes.size());
        for (const auto& node : nodes) names.push_back(node.first);

        static std::vector<uint8_t> mask_rg; // UI-thread scratch, reused across frames
        if (!gpu_outline_backend->renderSelectionOutlineMaskReadback(
                names, cam.lookfrom, cam.lookat, cam.vup, cam.vfov, aspect_ratio,
                sw, sh, mw, mh, mask_rg) ||
            mask_rg.size() < (size_t)mw * (size_t)mh * 2) {
            s_last_failed_hash = state_hash;
            // A failed refine keeps the coarse cached runs on screen instead
            // of dropping to the CPU path mid-interaction.
            if (cache_hit) {
                replayRuns(cit->second);
                return true;
            }
            return false;
        }

        SelectionOutlineFrameCache& ce = selection_outline_frame_cache[kCacheKey];
        ce.runs.clear();
        ce.hash = state_hash;
        ce.scale = use_scale;
        // Thinner + muted green, matching the raster-mode composite defaults
        // (SelectionOutlineParams) so mode switches don't change the style.
        ce.thickness = 1.0f;
        const ImU32 col_primary   = IM_COL32(72, 191, 125, 217);
        const ImU32 col_secondary = IM_COL32(51, 140, 94, 158);
        const ImU32 col_occluded  = IM_COL32(160, 160, 160, 130);
        for (int y = 0; y < mh; ++y) {
            const uint8_t* row  = mask_rg.data() + (size_t)y * mw * 2;
            const uint8_t* rowU = (y > 0)      ? row - (size_t)mw * 2 : nullptr;
            const uint8_t* rowD = (y < mh - 1) ? row + (size_t)mw * 2 : nullptr;
            int run_start = -1;
            ImU32 run_col = 0;
            auto flushRun = [&](int x_end) {
                if (run_start < 0) return;
                int len = x_end - run_start;
                int sx = run_start;
                while (len > 0) {
                    const int chunk = (std::min)(len, 65535);
                    // Run coordinates in SCREEN pixels; len in mask pixels
                    // (replay multiplies by ce.scale).
                    ce.runs.push_back({ (float)(sx * use_scale), (float)(y * use_scale),
                                        (uint16_t)chunk, run_col });
                    sx += chunk;
                    len -= chunk;
                }
                run_start = -1;
            };
            for (int x = 0; x < mw; ++x) {
                const uint8_t g = row[x * 2 + 1];
                ImU32 col = 0;
                bool boundary = false;
                if (g) {
                    // Silhouette pixel with an empty 4-neighbor (or at the
                    // image border) = outline pixel.
                    boundary = (x == 0 || x == mw - 1 || !rowU || !rowD ||
                                row[(x - 1) * 2 + 1] == 0 || row[(x + 1) * 2 + 1] == 0 ||
                                rowU[x * 2 + 1] == 0 || rowD[x * 2 + 1] == 0);
                    if (boundary) {
                        // R channel = depth-tested visible coverage; none in
                        // the neighborhood -> this part is hidden behind other
                        // geometry -> desaturated occluded style.
                        const bool visible =
                            row[x * 2] > 0 ||
                            (x > 0 && row[(x - 1) * 2] > 0) ||
                            (x < mw - 1 && row[(x + 1) * 2] > 0) ||
                            (rowU && rowU[x * 2] > 0) ||
                            (rowD && rowD[x * 2] > 0);
                        col = !visible ? col_occluded
                                       : ((g >= 192) ? col_primary : col_secondary);
                    }
                }
                if (boundary && run_start >= 0 && col == run_col) continue;
                flushRun(x);
                if (boundary) { run_start = x; run_col = col; }
            }
            flushRun(mw);
        }
        // Empty runs (selection fully off-screen) still count as handled; the
        // hash match replays the empty set for free until something changes.
        // Re-arm the churn throttle only while motion continues.
        s_churn_cooldown = settled ? 0 : 1;
        replayRuns(ce);
        return true;
    };

    // Pre-pass: in Rendered mode try the GPU mask path for every eligible
    // mesh item at once; the loop below then skips their CPU raster. Falls
    // through to the CPU outline when the raster cache isn't built yet
    // (fresh load straight into Rendered) or the mask render fails.
    bool gpu_outline_readback_drawn = false;
    if (gpu_outline_backend && viewport_settings.shading_mode == 2 &&
        viewport_settings.show_selection_outline && sel.hasSelection()) {
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        const uint64_t scene_gen = g_scene_geometry_generation.load(std::memory_order_acquire);
        if (gpu_outline_backend->hasValidRasterCache(scene_gen)) {
            if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
            std::vector<std::pair<std::string, Matrix4x4>> readback_nodes;
            bool readback_has_skinned = false;
            // xform_hash covers everything that requires re-SYNCING raster
            // object state (names, transforms, bones, geometry generation);
            // state_hash additionally folds in the camera/screen so the run
            // cache invalidates on view changes WITHOUT re-syncing objects.
            uint64_t xform_hash = 1469598103934665603ull;
            auto mixh  = [&](uint64_t v) { xform_hash ^= v; xform_hash *= 1099511628211ull; };
            auto mixhf = [&](float v) { uint32_t b; std::memcpy(&b, &v, 4); mixh(b); };
            for (const auto& item : sel.multi_selection) {
                if (item.type != SelectableType::Object || !item.object) continue;
                const std::string& nm = item.object->getNodeName();
                if (nm.empty()) continue;
                bool is_body = false;
                for (const auto& rb : ctx.scene.rigid_bodies) {
                    if (rb.enabled && rb.source_name == nm) { is_body = true; break; }
                }
                if (is_body) continue; // bodies keep their cheap AABB in the loop
                Matrix4x4 m = item.object->getTransformMatrix();
                readback_nodes.emplace_back(nm, m);
                for (const char c : nm) mixh((uint64_t)(unsigned char)c);
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 4; ++c) mixhf(m.m[r][c]);
                auto mcit = mesh_cache.find(nm);
                if (mcit != mesh_cache.end() && !mcit->second.empty() &&
                    mcit->second[0].second && mcit->second[0].second->hasAnySkinWeights()) {
                    readback_has_skinned = true;
                }
            }
            if (!readback_nodes.empty()) {
                // Bone-buffer sample joins the hash ONLY when a selected mesh
                // is actually skinned — otherwise an animated (unrelated)
                // character would invalidate a rigid selection's mask every
                // frame and force constant recomputes.
                const auto& bones = ctx.renderer.finalBoneMatrices;
                if (readback_has_skinned) {
                    mixh(bones.size());
                    if (!bones.empty()) {
                        const size_t bstep = (bones.size() > 16) ? (bones.size() / 16) : 1;
                        for (size_t bi = 0; bi < bones.size(); bi += bstep) {
                            const float* mm = &bones[bi].m[0][0];
                            mixhf(mm[3]); mixhf(mm[7]); mixhf(mm[11]); mixhf(mm[0]);
                        }
                    }
                }
                mixh(scene_gen);

                uint64_t state_hash = xform_hash;
                auto mixs  = [&](uint64_t v) { state_hash ^= v; state_hash *= 1099511628211ull; };
                auto mixsf = [&](float v) { uint32_t b; std::memcpy(&b, &v, 4); mixs(b); };
                mixsf(cam.lookfrom.x); mixsf(cam.lookfrom.y); mixsf(cam.lookfrom.z);
                mixsf(cam.lookat.x);   mixsf(cam.lookat.y);   mixsf(cam.lookat.z);
                mixsf(cam.vup.x);      mixsf(cam.vup.y);      mixsf(cam.vup.z);
                mixsf(cam.vfov);
                mixs((uint64_t)(int)screen_w);
                mixs((uint64_t)(int)screen_h);
                // Render resolution drives the projection aspect on the
                // backend side — a render-size change must re-render the mask.
                mixs((uint64_t)image_width);
                mixs((uint64_t)image_height);
                gpu_outline_readback_drawn = DrawGpuMaskOutlineRuns(
                    readback_nodes, readback_has_skinned, state_hash, xform_hash);
            }
        }
    }

    if (sel.hasSelection()) {
        // Draw bounding box for each selected item (multi-selection support)
        for (size_t idx = 0; idx < sel.multi_selection.size(); ++idx) {
            auto& item = sel.multi_selection[idx];

            // Primary selection (last one) gets a brighter color
            bool is_primary = (idx == sel.multi_selection.size() - 1);
            ImU32 color = is_primary ? IM_COL32(0, 255, 128, 255) : IM_COL32(0, 200, 100, 180);
            float thickness = is_primary ? 2.0f : 1.5f;

            Vec3 bb_min, bb_max;
            bool has_bounds = false;
            // For mesh objects we draw an oriented box (tight to local AABB after transform).
            Vec3 obb_corners[8];
            bool has_obb = false;

            if (item.type == SelectableType::Object && item.object) {
                std::string selectedName = item.object->getNodeName();
                if (selectedName.empty()) selectedName = "Unnamed";

                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                // USE CACHED BOUNDING BOX (O(1) lookup instead of O(N) triangle scan!)
                auto bbox_it = bbox_cache.find(selectedName);
                if (bbox_it != bbox_cache.end()) {
                    Vec3 cached_min = bbox_it->second.first;
                    Vec3 cached_max = bbox_it->second.second;

                    // Build local-space corners.
                    Vec3 local_corners[8] = {
                        Vec3(cached_min.x, cached_min.y, cached_min.z),
                        Vec3(cached_max.x, cached_min.y, cached_min.z),
                        Vec3(cached_max.x, cached_max.y, cached_min.z),
                        Vec3(cached_min.x, cached_max.y, cached_min.z),
                        Vec3(cached_min.x, cached_min.y, cached_max.z),
                        Vec3(cached_max.x, cached_min.y, cached_max.z),
                        Vec3(cached_max.x, cached_max.y, cached_max.z),
                        Vec3(cached_min.x, cached_max.y, cached_max.z),
                    };

                    // Fully composed transform — picks up gizmo-driven moves which
                    // write to the pivot component (transform->base alone is stale).
                    Matrix4x4 m = item.object->getTransformMatrix();
                    for (int c = 0; c < 8; ++c) {
                        const Vec3& p = local_corners[c];
                        obb_corners[c] = Vec3(
                            m.m[0][0] * p.x + m.m[0][1] * p.y + m.m[0][2] * p.z + m.m[0][3],
                            m.m[1][0] * p.x + m.m[1][1] * p.y + m.m[1][2] * p.z + m.m[1][3],
                            m.m[2][0] * p.x + m.m[2][1] * p.y + m.m[2][2] * p.z + m.m[2][3]
                        );
                    }
                    has_obb = true;
                }
            }
            else if (item.type == SelectableType::Light && item.light) {
                // No bbox helper for lights — drawLightGizmos already renders
                // a type-specific icon (circle/sun/cone/rectangle) that highlights
                // when selected, so an extra wireframe cube is redundant.
                has_bounds = false;
            }
            else if (item.type == SelectableType::Camera && item.camera) {
                Vec3 camPos = item.camera->lookfrom;
                float boxSize = 0.5f;
                bb_min = Vec3(camPos.x - boxSize, camPos.y - boxSize, camPos.z - boxSize);
                bb_max = Vec3(camPos.x + boxSize, camPos.y + boxSize, camPos.z + boxSize);
                has_bounds = true;
                color = is_primary ? IM_COL32(100, 200, 255, 255) : IM_COL32(80, 160, 200, 180);
            }
            else if (item.type == SelectableType::VDBVolume && item.vdb_volume) {
                AABB bounds = item.vdb_volume->getWorldBounds();
                bb_min = bounds.min;
                bb_max = bounds.max;
                has_bounds = true;
                // Orange for VDB
                color = is_primary ? IM_COL32(255, 128, 0, 255) : IM_COL32(200, 100, 0, 180);
            }
            else if (item.type == SelectableType::ForceField && item.force_field) {
                has_bounds = false; // Disable generic bbox drawing for force fields
            }

            bool drew_outline = false;
            if (item.type == SelectableType::Object && item.object) {
                // DrawSelectionRaster rasterizes the WHOLE mesh into a screen-space
                // mask (plus per-edge BVH occlusion ray casts) EVERY frame — it runs
                // even when the scene is static, which is the dominant idle CPU cost a
                // selected body pays (10x slower with gizmos on). For a physics body we
                // don't need that pixel-tight silhouette: draw a cheap world-AABB from
                // the body's OWN verts instead. It tracks the body when it moves and
                // costs effectively nothing when static. Applies whether playing or not.
                bool is_body = false;
                {
                    const std::string& bn = item.object->getNodeName();
                    for (const auto& rb : ctx.scene.rigid_bodies) {
                        if (rb.enabled && rb.source_name == bn) { is_body = true; break; }
                    }
                }
                if (is_body) {
                    if (viewport_settings.show_selection_outline) {
                        Vec3 mn, mx;
                        // Memoized world-AABB: a STOPPED body recomputes only when its
                        // geometry version changes, so it costs O(1)/frame at idle (the
                        // per-frame triangle walk was the ~6% idle CPU the user saw).
                        if (bodyWorldAABB(ctx, item.object->getNodeName(), mn, mx)) {
                            // Plain, occlusion-FREE border. DrawBoundingBox/DrawOrientedBox
                            // run per-edge BVH occlusion ray casts (96/frame for a small
                            // box) even when static — exactly the constant CPU we're
                            // killing. 12 straight AddLine calls cost nothing.
                            const Vec3 cs[8] = {
                                Vec3(mn.x, mn.y, mn.z), Vec3(mx.x, mn.y, mn.z),
                                Vec3(mx.x, mx.y, mn.z), Vec3(mn.x, mx.y, mn.z),
                                Vec3(mn.x, mn.y, mx.z), Vec3(mx.x, mn.y, mx.z),
                                Vec3(mx.x, mx.y, mx.z), Vec3(mn.x, mx.y, mx.z)
                            };
                            const int eds[12][2] = {
                                {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}
                            };
                            ImDrawList* dl = ImGui::GetBackgroundDrawList();
                            for (const auto& ed : eds) {
                                ImVec2 a, b;
                                if (ProjectWorldPoint(cs[ed[0]], a) && ProjectWorldPoint(cs[ed[1]], b))
                                    dl->AddLine(a, b, color, thickness);
                            }
                        }
                    }
                    drew_outline = true;  // cheap bbox (or outline off) — skip raster/hull
                } else if (viewport_settings.show_selection_outline) {
                    std::string outlineName = item.object->getNodeName();
                    if (gpu_outline_readback_drawn && !outlineName.empty()) {
                        // Already drawn by the Rendered-mode GPU mask
                        // pre-pass above (same eligibility criteria).
                        drew_outline = true;
                    } else if (gpu_outline_eligible && !outlineName.empty()) {
                        // GPU silhouette pass draws this node — collect the
                        // name, push the whole set to the backend after the
                        // loop. Skips the CPU raster/hull entirely.
                        gpu_outline_params.nodeNames.push_back(outlineName);
                        drew_outline = true;
                    } else {
                        if (outlineName.empty()) outlineName = "Unnamed";
                        // Prefer  rasterized outline (concave-tight, occlusion
                        // free of internal geometry). Falls back to convex hull when the mesh
                        // is too dense for the raster path's cost cap.
                        drew_outline = DrawSelectionRaster(outlineName, color, thickness);
                        if (!drew_outline) {
                            drew_outline = DrawSelectionHull(outlineName, color, thickness);
                        }
                    }
                } else {
                    // Toggle off → suppress raster/hull and bbox fallback alike
                    // so the user gets a fully clean viewport for the perf win.
                    drew_outline = true;
                }
            }
            if (!drew_outline) {
                if (has_obb) {
                    DrawOrientedBox(obb_corners, color, thickness);
                }
                else if (has_bounds) {
                    DrawBoundingBox(bb_min, bb_max, color, thickness);
                }
            }
        }
    }
    // Push or clear the GPU outline once per frame; the backend only
    // re-renders when the set actually changed. Cleared in Rendered mode and
    // when nothing eligible is selected (the names list stays empty then).
    if (gpu_outline_backend) {
        if (!gpu_outline_params.nodeNames.empty()) {
            gpu_outline_params.enabled = true;
            gpu_outline_backend->setSelectionOutlineParams(gpu_outline_params);
        }
        else {
            gpu_outline_backend->clearSelectionOutline();
        }
    }
}

void SceneUI::drawLightGizmos(UIContext& ctx, bool& gizmo_hit)
{
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;

    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();

    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    float aspect = (image_height > 0)
        ? (static_cast<float>(image_width) / static_cast<float>(image_height))
        : (io.DisplaySize.x / io.DisplaySize.y);
    const bool gizmoOrtho = cam.orthographic && viewport_settings.shading_mode != 2;

    auto Project = [&](const Vec3& p) -> ImVec2 {
        ImVec2 out;
        if (!projectGizmoWorldPoint(cam, gizmoOrtho, aspect, io.DisplaySize.x, io.DisplaySize.y, p, out))
            return ImVec2(-10000, -10000);
        return out;
        };

    auto IsOnScreen = [](const ImVec2& v) { return v.x > -5000; };

    for (auto& light : ctx.scene.lights) {
        if (!light->visible) continue;

        bool selected =
            (ctx.selection.selected.type == SelectableType::Light &&
                ctx.selection.selected.light == light);

        ImU32 col = selected
            ? IM_COL32(255, 100, 50, 255)
            : IM_COL32(255, 255, 100, 180);

        Vec3 pos = light->position;

        // [FIX] Depth/Occlusion Check for Light Gizmos
        //
        // [RACE FIX] Skip during animation render: the worker thread is
        // mutating HittableInstance transforms / bounds per frame and
        // Embree's intersection callback reads those concurrently — torn
        // matrices crash inside embree4.dll. Without this guard the
        // sequence render reliably crashes mid-render whenever there's a
        // visible light gizmo in the viewport (stack: drawLightGizmos →
        // EmbreeBVH::hit → embree4.dll AV).
        extern bool g_bvh_rebuild_pending;
        if (ctx.scene.bvh && !g_bvh_rebuild_pending && !ctx.is_animation_mode) {
            Vec3 to_pos = pos - cam.lookfrom;
            float dist = to_pos.length();
             if (dist > 0.1f) {
                 Ray r(cam.lookfrom, to_pos / dist);
                 HitRecord rec;
                 if (ctx.scene.bvh->hit(r, 0.001f, dist - 0.1f, rec, true)) {
                     // Occluded: Fade out
                     int alpha = (col >> 24) & 0xFF;
                     alpha = alpha / 5;
                     if (alpha < 20) alpha = 20;
                     col = (col & 0x00FFFFFF) | (alpha << 24);
                 }
            }
        }

        ImVec2 center = Project(pos);
        bool visible = IsOnScreen(center);

        if (!visible) continue;

        // -------- PICKING --------
        float dx = io.MousePos.x - center.x;
        float dy = io.MousePos.y - center.y;
        float d = sqrtf(dx * dx + dy * dy);

        // Mirror the viewport-raycast lock in handleObjectSelection: while
        // sculpt/edit-mesh tools own the click, icon picks must not steal
        // selection, otherwise the brush stays armed on an off-mesh target
        // and subsequent clicks hit the lock and go nowhere.
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

        if (d < 20.0f && ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver() &&
            !edit_mode_locked && !sculpt_mode_locked) {
            ctx.selection.selectLight(light);
            gizmo_hit = true;
        }

        if (selected) {
            std::string label = light->nodeName.empty() ? "Light" : light->nodeName;
            draw_list->AddText(ImVec2(center.x + 12, center.y - 12), col, label.c_str());
        }

        // ================== DRAW BY TYPE ==================

        // ---- POINT (Orta Daire + D\u0131\u015f Halka) ----
        if (light->type() == LightType::Point) {
            // Parlak merkez daire
            draw_list->AddCircleFilled(center, 5.0f, IM_COL32(255, 240, 180, 255));
            // D\u0131\u015f halka
            draw_list->AddCircle(center, 12.0f, col, 0, 2.0f);
            // \u0130nce i\u00e7 halka (derinlik etkisi i\u00e7in)
            draw_list->AddCircle(center, 8.0f, IM_COL32(255, 220, 100, 120), 0, 1.0f);

            // World-space radius ring (soft-shadow sphere visualization)
            if (light->getRadius() > 0.001f) {
                // Project a screen-aligned circle centered on the light position
                Vec3 rOffset = cam_right * light->getRadius();
                ImVec2 rEdge = Project(pos + rOffset);
                if (IsOnScreen(rEdge)) {
                    float px = sqrtf((rEdge.x - center.x) * (rEdge.x - center.x) +
                                     (rEdge.y - center.y) * (rEdge.y - center.y));
                    if (px > 4.0f) {
                        ImU32 ringCol = selected
                            ? IM_COL32(255, 200, 80, 200)
                            : IM_COL32(255, 220, 100, 110);
                        draw_list->AddCircle(center, px, ringCol, 48, 1.5f);
                    }
                }
            }
        }

        // ---- DIRECTIONAL (Sun + Arrow) ----
        else if (light->type() == LightType::Directional) {
            draw_list->AddCircle(center, 8.0f, col, 0, 2.0f);

            for (int i = 0; i < 8; ++i) {
                float a = i * (6.28f / 8.0f);
                ImVec2 dir(cosf(a), sinf(a));
                draw_list->AddLine(
                    ImVec2(center.x + dir.x * 12, center.y + dir.y * 12),
                    ImVec2(center.x + dir.x * 18, center.y + dir.y * 18),
                    col);
            }

            auto dl = std::dynamic_pointer_cast<DirectionalLight>(light);
            if (dl) {
                Vec3 end3d = pos + dl->direction.normalize() * 3.0f;
                ImVec2 end = Project(end3d);
                if (IsOnScreen(end)) {
                    draw_list->AddLine(center, end, col, 2.0f);
                    draw_list->AddCircleFilled(end, 3.0f, col);
                }
            }
        }

        // ---- AREA (Rectangle) ----
        else if (light->type() == LightType::Area) {
            auto al = std::dynamic_pointer_cast<AreaLight>(light);
            if (!al) continue;

            // Normalle�tirilmi� u ve v vekt�rleri
            Vec3 u = al->getU();
            Vec3 v = al->getV();
            float halfW = al->getWidth() * 0.5f;
            float halfH = al->getHeight() * 0.5f;

            // pos merkez noktas�, k��eleri merkezden hesapla
            Vec3 corner1 = pos - u * halfW - v * halfH;  // Sol-Alt
            Vec3 corner2 = pos + u * halfW - v * halfH;  // Sa�-Alt  
            Vec3 corner3 = pos + u * halfW + v * halfH;  // Sa�-�st
            Vec3 corner4 = pos - u * halfW + v * halfH;  // Sol-�st

            ImVec2 c1 = Project(corner1);
            ImVec2 c2 = Project(corner2);
            ImVec2 c3 = Project(corner3);
            ImVec2 c4 = Project(corner4);

            draw_list->AddLine(c1, c2, col);
            draw_list->AddLine(c2, c3, col);
            draw_list->AddLine(c3, c4, col);
            draw_list->AddLine(c4, c1, col);
            // X �izgisi: merkez art�k ger�ek merkez
            draw_list->AddLine(c1, c3, col, 1.0f);
            draw_list->AddLine(c2, c4, col, 1.0f);
        }

        // ---- SPOT (Cone) ----
        else if (light->type() == LightType::Spot) {
            auto sl = std::dynamic_pointer_cast<SpotLight>(light);
            if (!sl) continue;

            Vec3 dir = sl->direction.normalize();
            float len = 3.0f;
            float radius = len * tanf(sl->getAngleDegrees() * 3.14159f / 360.0f);

            Vec3 base = pos + dir * len;
            Vec3 right = (fabs(dir.y) > 0.9f) ? Vec3(1, 0, 0)
                : dir.cross(Vec3(0, 1, 0)).normalize();
            Vec3 up = right.cross(dir).normalize();

            const int segs = 12;
            ImVec2 last;
            for (int i = 0; i <= segs; ++i) {
                float a = i * (6.28f / segs);
                Vec3 p = base + right * (cosf(a) * radius)
                    + up * (sinf(a) * radius);

                ImVec2 sp = Project(p);
                if (i > 0 && IsOnScreen(sp) && IsOnScreen(last))
                    draw_list->AddLine(last, sp, col);
                if (i < segs && IsOnScreen(sp))
                    draw_list->AddLine(center, sp, col);

                last = sp;
            }
        }
    }
}


// ===============================================================================
// IMGUIZMO TRANSFORM GIZMO
// ===============================================================================
void SceneUI::drawTransformGizmo(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    ImGuiIO& io = ImGui::GetIO();

    // Sleek and modern ImGuizmo styling
    ImGuizmo::Style& style = ImGuizmo::GetStyle();
    style.TranslationLineThickness   = 1.8f;
    style.TranslationLineArrowSize   = 5.0f;
    style.RotationLineThickness      = 1.6f;
    style.RotationOuterLineThickness = 2.0f;
    style.ScaleLineThickness         = 1.8f;
    style.ScaleLineCircleSize        = 5.0f;
    style.CenterCircleSize           = 4.5f;

    // Vibrant modern colors
    style.Colors[ImGuizmo::DIRECTION_X]           = ImVec4(0.96f, 0.22f, 0.33f, 1.00f); // Sleek Red (X)
    style.Colors[ImGuizmo::DIRECTION_Y]           = ImVec4(0.33f, 0.84f, 0.12f, 1.00f); // Sleek Green (Y)
    style.Colors[ImGuizmo::DIRECTION_Z]           = ImVec4(0.18f, 0.52f, 0.98f, 1.00f); // Sleek Blue (Z)
    style.Colors[ImGuizmo::PLANE_X]               = ImVec4(0.96f, 0.22f, 0.33f, 0.22f);
    style.Colors[ImGuizmo::PLANE_Y]               = ImVec4(0.33f, 0.84f, 0.12f, 0.22f);
    style.Colors[ImGuizmo::PLANE_Z]               = ImVec4(0.18f, 0.52f, 0.98f, 0.22f);
    style.Colors[ImGuizmo::SELECTION]             = ImVec4(1.00f, 0.65f, 0.00f, 0.90f); // Sleek Orange (Selected)
    style.Colors[ImGuizmo::INACTIVE]              = ImVec4(0.55f, 0.57f, 0.60f, 0.45f);
    style.Colors[ImGuizmo::TRANSLATION_LINE]      = ImVec4(0.60f, 0.60f, 0.60f, 0.25f);
    style.Colors[ImGuizmo::SCALE_LINE]            = ImVec4(0.60f, 0.60f, 0.60f, 0.25f);
    style.Colors[ImGuizmo::ROTATION_USING_BORDER] = ImVec4(1.00f, 0.65f, 0.00f, 1.00f);
    style.Colors[ImGuizmo::ROTATION_USING_FILL]   = ImVec4(1.00f, 0.65f, 0.00f, 0.15f);

    // IMPORTANT: Keep ImGuizmo's per-frame state alive even when we early-out.
    // We hit a bug where deleting an object could clear selection before this
    // function reached Manipulate(), leaving ImGuizmo::IsOver() latched from the
    // previous frame. Viewport clicks were then blocked as if a gizmo was still
    // under the mouse, and selection only "woke up" again after clicking an item
    // in the hierarchy. Calling BeginFrame/SetRect before any early return resets
    // that stale hover state and keeps viewport picking responsive after deletes.
    ImGuizmo::BeginFrame();
    // Match the viewport projection so the gizmo overlays the geometry. Orthographic only applies
    // to the raster preview modes (Solid/Matcap/Preview); the Rendered viewport (shading_mode==2)
    // path-traces in PERSPECTIVE even under a standard view, so the gizmo must stay perspective there.
    const bool gizmoUseOrtho = ctx.scene.camera && ctx.scene.camera->orthographic &&
                               viewport_settings.shading_mode != 2;
    ImGuizmo::SetOrthographic(gizmoUseOrtho);
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

    if (!sel.hasSelection() || !sel.show_gizmo || !ctx.scene.camera) return;

    // Check visibility of selected item
    bool is_visible = true;
    if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        is_visible = sel.selected.object->visible;
    }
    else if (sel.selected.type == SelectableType::Light && sel.selected.light) is_visible = sel.selected.light->visible;
    else if (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) is_visible = sel.selected.vdb_volume->visible;
    else if (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume) is_visible = sel.selected.gas_volume->visible;
    
    if (!is_visible) return;

    Camera& cam = *ctx.scene.camera;

    // �������������������������������������������������������������������������
    // Build View Matrix (LookAt)
    // �������������������������������������������������������������������������
    Vec3 eye = cam.lookfrom;
    Vec3 target = cam.lookat;
    Vec3 up = cam.vup;

    Vec3 f = (target - eye).normalize();  // Forward
    Vec3 r = f.cross(up).normalize();     // Right
    Vec3 u = r.cross(f);                   // Up

    float viewMatrix[16] = {
        r.x,  u.x, -f.x, 0.0f,
        r.y,  u.y, -f.y, 0.0f,
        r.z,  u.z, -f.z, 0.0f,
        -r.dot(eye), -u.dot(eye), f.dot(eye), 1.0f
    };

    // �������������������������������������������������������������������������
    // Build Projection Matrix (Perspective)
    // �������������������������������������������������������������������������
    float near_plane = 0.1f;
    float far_plane = 10000.0f;
    // Kept at function scope: also used by the drag-speed (pixel->world) math below.
    float tan_half_fov = tanf(cam.vfov * 3.14159265359f / 180.0f * 0.5f);

    float projMatrix[16] = { 0 };
    if (gizmoUseOrtho) {
        // GL-style orthographic (parallel) projection so the gizmo matches the ortho viewport.
        const float orthoH = (cam.ortho_height > 1e-4f) ? cam.ortho_height : 10.0f;
        const float orthoW = orthoH * aspect_ratio;
        projMatrix[0]  = 2.0f / orthoW;
        projMatrix[5]  = 2.0f / orthoH;
        projMatrix[10] = -2.0f / (far_plane - near_plane);
        projMatrix[14] = -(far_plane + near_plane) / (far_plane - near_plane);
        projMatrix[15] = 1.0f;
    } else {
        projMatrix[0] = 1.0f / (aspect_ratio * tan_half_fov);
        projMatrix[5] = 1.0f / tan_half_fov;
        projMatrix[10] = -(far_plane + near_plane) / (far_plane - near_plane);
        projMatrix[11] = -1.0f;
        projMatrix[14] = -(2.0f * far_plane * near_plane) / (far_plane - near_plane);
    }

    // Gizmo sizing. The gizmo is a FIXED world size, identical for every selection type, so it
    // scales with zoom (ortho zoom changes ortho_height, perspective dolly changes distance) but
    // NOT with the selected object: AABB-derived sizing pegged the gizmo huge on terrain/ship
    // selections and tiny on gizmo-only objects (lights, force fields, domains) whose AABB is
    // small or absent. The screen-fraction clamps keep it usable at zoom extremes.
    {
        constexpr float kGizmoWorldSize = 0.85f; // world units the gizmo axes should span (reduced from 1.0f)

        // mGizmoSizeClipSpace = worldSize * (NDC-x units per world unit at the gizmo) so
        // mScreenFactor (= clipSpace / rightLength) resolves to a constant world size.
        float ndcPerWorldUnit;
        if (gizmoUseOrtho) {
            const float orthoW = ((cam.ortho_height > 1e-4f) ? cam.ortho_height : 10.0f) * aspect_ratio;
            ndcPerWorldUnit = 2.0f / orthoW;
        } else {
            // Perspective: 1 world unit at view depth d spans projMatrix[0]/d NDC-x units.
            const Vec3 gizmoAnchor = sel.selected.has_cached_aabb
                ? (sel.selected.cached_aabb.min + sel.selected.cached_aabb.max) * 0.5f
                : sel.selected.position;
            float depth = (gizmoAnchor - eye).dot(f);
            if (depth < near_plane) depth = near_plane;
            ndcPerWorldUnit = 1.0f / (aspect_ratio * tan_half_fov * depth);
        }
        // Floor near ImGuizmo's default (0.1) so the gizmo never shrinks into uselessness on
        // far/zoomed-out views; modest ceiling so close-ups don't turn it into a billboard.
        // Clamped to [0.055f, 0.135f] to make the gizmo smaller and less obtrusive.
        const float gizmoClip = std::clamp(kGizmoWorldSize * ndcPerWorldUnit, 0.055f, 0.135f);
        ImGuizmo::SetGizmoSizeClipSpace(gizmoClip);
    }
    auto Project = [&](Vec3 p) -> ImVec2 {
        float x = p.x, y = p.y, z = p.z;
        float vx = viewMatrix[0] * x + viewMatrix[4] * y + viewMatrix[8] * z + viewMatrix[12];
        float vy = viewMatrix[1] * x + viewMatrix[5] * y + viewMatrix[9] * z + viewMatrix[13];
        float vz = viewMatrix[2] * x + viewMatrix[6] * y + viewMatrix[10] * z + viewMatrix[14];
        float vw = viewMatrix[3] * x + viewMatrix[7] * y + viewMatrix[11] * z + viewMatrix[15];
        float cx = projMatrix[0] * vx + projMatrix[4] * vy + projMatrix[8] * vz + projMatrix[12] * vw;
        float cy = projMatrix[1] * vx + projMatrix[5] * vy + projMatrix[9] * vz + projMatrix[13] * vw;
        float cw = projMatrix[3] * vx + projMatrix[7] * vy + projMatrix[11] * vz + projMatrix[15] * vw;
        if (cw < 0.1f) return ImVec2(-10000, -10000);
        return ImVec2(((cx / cw) * 0.5f + 0.5f) * io.DisplaySize.x, (1.0f - ((cy / cw) * 0.5f + 0.5f)) * io.DisplaySize.y);
        };

    // �������������������������������������������������������������������������
    // Get Object Matrix
    // �������������������������������������������������������������������������
    const bool edit_mode_active =
        mesh_overlay_settings.enabled &&
        mesh_overlay_settings.edit_mode &&
        sel.multi_selection.size() == 1 &&
        sel.selected.type == SelectableType::Object &&
        sel.selected.object &&
        sel.mesh_element_mode != MeshElementSelectMode::Object;

    if (edit_mode_active) {
        static bool edit_gizmo_active = false;
        static bool edit_was_using_gizmo = false;
        static bool edit_drag_changed = false;
        static std::string edit_drag_object_name;
        static std::vector<MeshEditTriangleState> edit_drag_start_states;
        static float editGizmoMatrix[16] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };

        ImGuizmo::OPERATION editOperation = ImGuizmo::TRANSLATE;
        switch (sel.transform_mode) {
        case TransformMode::Translate: editOperation = ImGuizmo::TRANSLATE; break;
        case TransformMode::Rotate: editOperation = ImGuizmo::ROTATE; break;
        case TransformMode::Scale: editOperation = ImGuizmo::SCALE; break;
        }
        ImGuizmo::MODE editMode =
            (sel.transform_space == TransformSpace::Local) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;

        bool has_editable_element = false;
        const Vec3 editable_gizmo_pos = getSelectedMeshElementWorldPosition(ctx, &has_editable_element);
        const bool gizmo_using = ImGuizmo::IsUsing();
        if (!has_editable_element && !edit_gizmo_active && !gizmo_using) {
            edit_gizmo_active = false;
            return;
        }

        if (!edit_gizmo_active && !gizmo_using) {
            editGizmoMatrix[0] = 1.0f; editGizmoMatrix[1] = 0.0f; editGizmoMatrix[2] = 0.0f; editGizmoMatrix[3] = 0.0f;
            editGizmoMatrix[4] = 0.0f; editGizmoMatrix[5] = 1.0f; editGizmoMatrix[6] = 0.0f; editGizmoMatrix[7] = 0.0f;
            editGizmoMatrix[8] = 0.0f; editGizmoMatrix[9] = 0.0f; editGizmoMatrix[10] = 1.0f; editGizmoMatrix[11] = 0.0f;
            editGizmoMatrix[12] = editable_gizmo_pos.x;
            editGizmoMatrix[13] = editable_gizmo_pos.y;
            editGizmoMatrix[14] = editable_gizmo_pos.z;
            editGizmoMatrix[15] = 1.0f;
        }

        const Vec3 oldEditPos(editGizmoMatrix[12], editGizmoMatrix[13], editGizmoMatrix[14]);
        Matrix4x4 oldEditMatrix;
        oldEditMatrix.m[0][0] = editGizmoMatrix[0]; oldEditMatrix.m[1][0] = editGizmoMatrix[1]; oldEditMatrix.m[2][0] = editGizmoMatrix[2]; oldEditMatrix.m[3][0] = editGizmoMatrix[3];
        oldEditMatrix.m[0][1] = editGizmoMatrix[4]; oldEditMatrix.m[1][1] = editGizmoMatrix[5]; oldEditMatrix.m[2][1] = editGizmoMatrix[6]; oldEditMatrix.m[3][1] = editGizmoMatrix[7];
        oldEditMatrix.m[0][2] = editGizmoMatrix[8]; oldEditMatrix.m[1][2] = editGizmoMatrix[9]; oldEditMatrix.m[2][2] = editGizmoMatrix[10]; oldEditMatrix.m[3][2] = editGizmoMatrix[11];
        oldEditMatrix.m[0][3] = editGizmoMatrix[12]; oldEditMatrix.m[1][3] = editGizmoMatrix[13]; oldEditMatrix.m[2][3] = editGizmoMatrix[14]; oldEditMatrix.m[3][3] = editGizmoMatrix[15];

        const bool manipulated = !rtpython::wantsInputCapture() && !rtapi::renderOutputPending() && ImGuizmo::Manipulate(
            viewMatrix, projMatrix, editOperation, editMode, editGizmoMatrix);
        const bool gizmo_is_using_now = ImGuizmo::IsUsing();

        if (gizmo_is_using_now && !edit_was_using_gizmo) {
            edit_drag_object_name = sel.selected.object ? sel.selected.object->getNodeName() : std::string{};
            edit_drag_start_states.clear();
            edit_drag_changed = false;
            if (!edit_drag_object_name.empty()) {
                beginInteractiveSubdivisionPreview(edit_drag_object_name);
                captureMeshEditLayerState(ctx, edit_drag_object_name, edit_drag_start_states);
            }
        }

        if (manipulated) {
            if (!edit_gizmo_active && !gizmo_is_using_now) {
                edit_drag_object_name = sel.selected.object ? sel.selected.object->getNodeName() : std::string{};
                edit_drag_start_states.clear();
                if (!edit_drag_object_name.empty()) {
                    captureMeshEditLayerState(ctx, edit_drag_object_name, edit_drag_start_states);
                }
            }
            const Vec3 newPos(editGizmoMatrix[12], editGizmoMatrix[13], editGizmoMatrix[14]);
            Matrix4x4 newEditMatrix;
            newEditMatrix.m[0][0] = editGizmoMatrix[0]; newEditMatrix.m[1][0] = editGizmoMatrix[1]; newEditMatrix.m[2][0] = editGizmoMatrix[2]; newEditMatrix.m[3][0] = editGizmoMatrix[3];
            newEditMatrix.m[0][1] = editGizmoMatrix[4]; newEditMatrix.m[1][1] = editGizmoMatrix[5]; newEditMatrix.m[2][1] = editGizmoMatrix[6]; newEditMatrix.m[3][1] = editGizmoMatrix[7];
            newEditMatrix.m[0][2] = editGizmoMatrix[8]; newEditMatrix.m[1][2] = editGizmoMatrix[9]; newEditMatrix.m[2][2] = editGizmoMatrix[10]; newEditMatrix.m[3][2] = editGizmoMatrix[11];
            newEditMatrix.m[0][3] = editGizmoMatrix[12]; newEditMatrix.m[1][3] = editGizmoMatrix[13]; newEditMatrix.m[2][3] = editGizmoMatrix[14]; newEditMatrix.m[3][3] = editGizmoMatrix[15];

            bool applied = false;
            if (editOperation == ImGuizmo::TRANSLATE) {
                applied = applySelectedMeshElementTranslation(ctx, newPos - oldEditPos);
            } else {
                const Matrix4x4 deltaMatrix = newEditMatrix * oldEditMatrix.inverse();
                applied = applySelectedMeshElementTransform(ctx, deltaMatrix);
            }

            if (applied) {
                edit_drag_changed = true;
                ProjectManager::getInstance().markModified();
                if (ctx.backend_ptr && !edit_drag_object_name.empty()) {
                    queueMeshEditGpuSync(edit_drag_object_name);
                }
            }
        }

        if (!gizmo_using && !manipulated && has_editable_element) {
            editGizmoMatrix[0] = 1.0f; editGizmoMatrix[1] = 0.0f; editGizmoMatrix[2] = 0.0f; editGizmoMatrix[3] = 0.0f;
            editGizmoMatrix[4] = 0.0f; editGizmoMatrix[5] = 1.0f; editGizmoMatrix[6] = 0.0f; editGizmoMatrix[7] = 0.0f;
            editGizmoMatrix[8] = 0.0f; editGizmoMatrix[9] = 0.0f; editGizmoMatrix[10] = 1.0f; editGizmoMatrix[11] = 0.0f;
            editGizmoMatrix[12] = editable_gizmo_pos.x;
            editGizmoMatrix[13] = editable_gizmo_pos.y;
            editGizmoMatrix[14] = editable_gizmo_pos.z;
            editGizmoMatrix[15] = 1.0f;
        }

        if (edit_was_using_gizmo && !gizmo_is_using_now && edit_drag_changed &&
            !edit_drag_start_states.empty() && !edit_drag_object_name.empty()) {
            endInteractiveSubdivisionPreview(ctx, edit_drag_object_name, true);
            refreshMeshEditLayerEditedState(ctx);

            std::vector<MeshEditTriangleState> edit_drag_end_states;
            captureMeshEditLayerState(ctx, edit_drag_object_name, edit_drag_end_states);

            bool mesh_edit_changed = edit_drag_end_states.size() == edit_drag_start_states.size() && !edit_drag_end_states.empty();
            if (mesh_edit_changed) {
                for (size_t i = 0; i < edit_drag_end_states.size(); ++i) {
                    for (int corner = 0; corner < 3; ++corner) {
                        if ((edit_drag_end_states[i].positions[corner] - edit_drag_start_states[i].positions[corner]).length_squared() > 1e-12f) {
                            goto mesh_edit_changed_confirmed;
                        }
                    }
                }
                mesh_edit_changed = false;
            }

mesh_edit_changed_confirmed:
            if (mesh_edit_changed) {
                history.record(std::make_unique<MeshEditCommand>(
                    edit_drag_object_name,
                    edit_drag_start_states,
                    std::move(edit_drag_end_states)));
                ProjectManager::getInstance().markModified();
            }

            edit_drag_start_states.clear();
            edit_drag_object_name.clear();
            edit_drag_changed = false;
        } else if (edit_was_using_gizmo && !gizmo_is_using_now && !edit_drag_object_name.empty()) {
            endInteractiveSubdivisionPreview(ctx, edit_drag_object_name, false);
            edit_drag_start_states.clear();
            edit_drag_object_name.clear();
            edit_drag_changed = false;
        }

        edit_gizmo_active = gizmo_is_using_now;
        edit_was_using_gizmo = gizmo_is_using_now;
        is_dragging = edit_gizmo_active;
        return;
    }

    auto* selected_domain = static_cast<RayTrophiSim::SimulationGridDomainDesc*>(nullptr);
    if (sel.selected.type == SelectableType::SimulationDomain &&
        sel.selected.particle_system_index >= 0 &&
        sel.selected.particle_system_index < static_cast<int>(ctx.scene.particle_systems.size())) {
        auto& system = ctx.scene.particle_systems[static_cast<std::size_t>(sel.selected.particle_system_index)];
        if (system.runtime &&
            sel.selected.simulation_domain_index >= 0 &&
            sel.selected.simulation_domain_index < static_cast<int>(system.runtime->gridDomains().size())) {
            selected_domain = &system.runtime->gridDomains()[static_cast<std::size_t>(sel.selected.simulation_domain_index)];
            const Vec3 mn = Vec3::min(selected_domain->bounds_min, selected_domain->bounds_max);
            const Vec3 mx = Vec3::max(selected_domain->bounds_min, selected_domain->bounds_max);
            sel.selected.position = (mn + mx) * 0.5f;
            sel.selected.scale = mx - mn;
        }
    }

    float objectMatrix[16];
    // While a physics body is sim-driven (timeline playing) we suppress the
    // interactive gizmo entirely — repositioning it on the moving geometry cost a
    // whole-scene vertex walk every frame. drawSelectionBoundingBox draws a cheap
    // live bbox instead. Set in the Object branch below, consumed at Manipulate().
    bool skip_body_gizmo = false;
    Vec3 pos = sel.selected.position;

    // AreaLight: position zaten merkez noktas\u0131, ek offset gerekli de\u011fil

    // Initialize as identity with position
    Matrix4x4 startMat = Matrix4x4::identity();
    startMat.m[0][3] = pos.x;
    startMat.m[1][3] = pos.y;
    startMat.m[2][3] = pos.z;
    if (selected_domain) {
        const Vec3 extent = Vec3::max(sel.selected.scale, Vec3(0.001f));
        startMat.m[0][0] = extent.x;
        startMat.m[1][1] = extent.y;
        startMat.m[2][2] = extent.z;
    }

    // Handle Light Rotation (Directional/Spot)
    if (sel.selected.type == SelectableType::Light && sel.selected.light) {
        Vec3 dir(0, 0, 0);
        bool hasDir = false;

        if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(sel.selected.light)) {
            // DirectionalLight: getDirection returns vector TO Light (inverse of direction)
            // We want the light direction for visualization
            dir = dl->getDirection(Vec3(0)).normalize() * -1.0f;
            hasDir = true;
        }
        else if (auto sl = std::dynamic_pointer_cast<SpotLight>(sel.selected.light)) {
            // Manual Matrix for SpotLight to include Angle as Scale
            hasDir = false;
            float angle = sl->getAngleDegrees();
            if (angle < 1.0f) angle = 1.0f;

            // Direction Basis
            Vec3 dirVec = sl->direction.normalize();
            Vec3 Z = -dirVec;
            Vec3 Y_temp(0, 1, 0);
            if (abs(Vec3::dot(Z, Y_temp)) > 0.99f) Y_temp = Vec3(1, 0, 0);

            Vec3 X = Vec3::cross(Y_temp, Z).normalize();
            Vec3 Y = Vec3::cross(Z, X).normalize();

            // Scale X and Y by Angle for Gizmo Interaction
            Vec3 X_scaled = X * angle;
            Vec3 Y_scaled = Y * angle;

            // Scale Z by (1 + Falloff) for Falloff interaction
            float zScale = 1.0f + sl->getFalloff();
            Vec3 Z_scaled = Z * zScale;

            startMat.m[0][0] = X_scaled.x; startMat.m[0][1] = Y_scaled.x; startMat.m[0][2] = Z_scaled.x;
            startMat.m[1][0] = X_scaled.y; startMat.m[1][1] = Y_scaled.y; startMat.m[1][2] = Z_scaled.y;
            startMat.m[2][0] = X_scaled.z; startMat.m[2][1] = Y_scaled.z; startMat.m[2][2] = Z_scaled.z;

            // -----------------------------------------------------
            // Visual Helper: Cone Draw
            // -----------------------------------------------------
            ImDrawList* dl = ImGui::GetBackgroundDrawList();
            Vec3 pos = sl->position;
            float h = 5.0f; // Visual height
            float r = tanf(angle * 3.14159f / 180.0f * 0.5f) * h;

            ImVec2 pTip = Project(pos);
            Vec3 centerBase = pos + dirVec * h;

            ImU32 col = IM_COL32(255, 255, 0, 180);
            Vec3 prevP;
            bool first = true;

            for (int i = 0; i <= 24; ++i) {
                float t = (float)i / 24.0f * 6.28318f;
                Vec3 pBase = centerBase + (X * cosf(t) + Y * sinf(t)) * r;
                ImVec2 pScreen = Project(pBase);

                if (!first) dl->AddLine(Project(prevP), pScreen, col, 2.0f);

                if (i % 6 == 0) dl->AddLine(pTip, pScreen, col, 1.0f);

                prevP = pBase;
                first = false;
            }

            // Inner Cone (Falloff)
            float falloff = sl->getFalloff();
            if (falloff > 0.05f) {
                float innerAngle = angle * (1.0f - falloff);
                float rInner = tanf(innerAngle * 3.14159f / 180.0f * 0.5f) * h;
                ImU32 colInner = IM_COL32(255, 160, 20, 120);
                Vec3 prevIn;
                bool firstIn = true;
                for (int i = 0; i <= 24; i++) {
                    float t = (float)i / 24.0f * 6.28318f;
                    Vec3 pBase = centerBase + (X * cosf(t) + Y * sinf(t)) * rInner;
                    if (!firstIn && i % 2 == 0) dl->AddLine(Project(prevIn), Project(pBase), colInner, 1.0f); // Dashed-ish effect
                    prevIn = pBase;
                    firstIn = false;
                }
            }
        }
        else if (auto al = std::dynamic_pointer_cast<AreaLight>(sel.selected.light)) {
            hasDir = false;
            // Normalize vekt\u00f6rleri width/height ile \u00f6l\u00e7eklendirerek ger\u00e7ek boyutu yans\u0131t
            Vec3 X = al->getU() * al->getWidth();   // u normalize, width ile \u00f6l\u00e7ekle
            Vec3 Z = al->getV() * al->getHeight();  // v normalize, height ile \u00f6l\u00e7ekle
            // Normalized Y (Normal)
            Vec3 Y = Vec3::cross(al->getU(), al->getV()).normalize();

            startMat.m[0][0] = X.x; startMat.m[0][1] = Y.x; startMat.m[0][2] = Z.x;
            startMat.m[1][0] = X.y; startMat.m[1][1] = Y.y; startMat.m[1][2] = Z.y;
            startMat.m[2][0] = X.z; startMat.m[2][1] = Y.z; startMat.m[2][2] = Z.z;

            // Visualization: Direction Arrow (\u00d6l\u00e7e\u011fe g\u00f6re k\u0131salt)
            ImDrawList* dl = ImGui::GetBackgroundDrawList();
            Vec3 center = pos;
            Vec3 normal = Y.normalize();
            float len = std::min(al->getWidth(), al->getHeight()) * 0.5f; // Daha k\u0131sa, orant\u0131l\u0131 ok
            if (len < 0.3f) len = 0.3f;
            Vec3 pTip = center + normal * len;
            dl->AddLine(Project(center), Project(pTip), IM_COL32(255, 255, 0, 200), 2.0f);
            dl->AddCircleFilled(Project(pTip), 4.0f, IM_COL32(255, 255, 0, 255));
        }

        if (hasDir) {
            // Align Gizmo -Z with Light Direction
            Vec3 Z = -dir;
            Vec3 Y(0, 1, 0);
            if (abs(Vec3::dot(Z, Y)) > 0.99f) Y = Vec3(1, 0, 0); // Lock prevention
            Vec3 X = Vec3::cross(Y, Z).normalize();
            Y = Vec3::cross(Z, X).normalize();

            startMat.m[0][0] = X.x; startMat.m[0][1] = Y.x; startMat.m[0][2] = Z.x;
            startMat.m[1][0] = X.y; startMat.m[1][1] = Y.y; startMat.m[1][2] = Z.y;
            startMat.m[2][0] = X.z; startMat.m[2][1] = Y.z; startMat.m[2][2] = Z.z;
        }
    }

    objectMatrix[0] = startMat.m[0][0]; objectMatrix[1] = startMat.m[1][0]; objectMatrix[2] = startMat.m[2][0]; objectMatrix[3] = startMat.m[3][0];
    objectMatrix[4] = startMat.m[0][1]; objectMatrix[5] = startMat.m[1][1]; objectMatrix[6] = startMat.m[2][1]; objectMatrix[7] = startMat.m[3][1];
    objectMatrix[8] = startMat.m[0][2]; objectMatrix[9] = startMat.m[1][2]; objectMatrix[10] = startMat.m[2][2]; objectMatrix[11] = startMat.m[3][2];
    objectMatrix[12] = startMat.m[0][3]; objectMatrix[13] = startMat.m[1][3]; objectMatrix[14] = startMat.m[2][3]; objectMatrix[15] = startMat.m[3][3];

    // If object has transform, use it
    // If object has transform, use it - BUT ONLY if not a Mixed Group
    // Mixed groups use the Centroid pivot (startMat) to avoid fighting/resets
    bool is_mixed_group = false;
    if (sel.multi_selection.size() > 1) {
        is_mixed_group = true;
    }
    else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        std::string name = sel.selected.object->getNodeName();
        if (name.empty()) name = "Unnamed";
        auto it = mesh_cache.find(name);
        if (it != mesh_cache.end()) {
            // OPTIMIZED: Only check first 100 triangles, not all 2M!
            Transform* firstT = nullptr;
            const size_t MAX_CHECK = std::min((size_t)100, it->second.size());
            for (size_t i = 0; i < MAX_CHECK; ++i) {
                auto th = it->second[i].second->getTransformHandle().get();
                if (!firstT) firstT = th;
                else if (th != firstT) {
                    is_mixed_group = true;
                    break;
                }
            }
        }
    }

    static Matrix4x4 mixed_gizmo_matrix;
    static bool was_using_mixed = false;

    // Check global drag state
    bool is_using_gizmo_now = ImGuizmo::IsUsing();
    // auto transform = ... (Removed unsafe access)
    if (is_mixed_group) {
        // PERISISTENT GIZMO STATE for Mixed Groups
        // Prevents Gizmo from snapping back to Identity Rotation every frame which causes explosion

        if (!is_using_gizmo_now) {
            // Not dragging: Reset to Centroid Position + Identity Rotation
            // (Or we could average rotations, but Identity is safer for a group pivot)
            mixed_gizmo_matrix = Matrix4x4::identity();
            mixed_gizmo_matrix.m[0][3] = sel.selected.position.x;
            mixed_gizmo_matrix.m[1][3] = sel.selected.position.y;
            mixed_gizmo_matrix.m[2][3] = sel.selected.position.z;
        }

        // Use the persistent matrix
        objectMatrix[0] = mixed_gizmo_matrix.m[0][0]; objectMatrix[1] = mixed_gizmo_matrix.m[1][0]; objectMatrix[2] = mixed_gizmo_matrix.m[2][0]; objectMatrix[3] = mixed_gizmo_matrix.m[3][0];
        objectMatrix[4] = mixed_gizmo_matrix.m[0][1]; objectMatrix[5] = mixed_gizmo_matrix.m[1][1]; objectMatrix[6] = mixed_gizmo_matrix.m[2][1]; objectMatrix[7] = mixed_gizmo_matrix.m[3][1];
        objectMatrix[8] = mixed_gizmo_matrix.m[0][2]; objectMatrix[9] = mixed_gizmo_matrix.m[1][2]; objectMatrix[10] = mixed_gizmo_matrix.m[2][2]; objectMatrix[11] = mixed_gizmo_matrix.m[3][2];
        objectMatrix[12] = mixed_gizmo_matrix.m[0][3]; objectMatrix[13] = mixed_gizmo_matrix.m[1][3]; objectMatrix[14] = mixed_gizmo_matrix.m[2][3]; objectMatrix[15] = mixed_gizmo_matrix.m[3][3];
    }
    
    else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        // Single Object (or Homogeneous Group) - Lock to Object Transform
        auto transform = sel.selected.object->getTransformHandle();
        if (transform) {
            Matrix4x4 mat = transform->getPivotMatrix();

            // Body-simulated objects move their GEOMETRY (vertex baking) but leave
            // the transform handle at the spawn pose, so getPivotMatrix would leave
            // the gizmo behind the moving object. Re-center the gizmo on the object's
            // LIVE world-vertex AABB (translation only, keeping the spawn orientation)
            // so it sits on the object regardless of where the authored pivot point
            // is — using the rigid pivot pose instead would chase the pivot POINT,
            // which on imported meshes is often far from the visible geometry.
            const std::string node = sel.selected.object->getNodeName();
            if (!node.empty()) {
                bool is_body = false;
                for (const auto& rb : ctx.scene.rigid_bodies) {
                    if (rb.enabled && rb.source_name == node) { is_body = true; break; }
                }
                if (is_body) {
                    if (timeline.isPlaying()) {
                        // Sim-driven: skip the interactive gizmo (and its per-frame
                        // vertex walk). The cheap live bbox in drawSelectionBoundingBox
                        // is the selection indicator while playing.
                        skip_body_gizmo = true;
                    } else {
                        // Idle: re-center on the body's live world-AABB so the gizmo sits
                        // on the object regardless of where the authored pivot is. The
                        // memoized AABB (recomputed only when the body's geometry version
                        // changes) keeps a STOPPED body at O(1)/frame — the per-frame
                        // triangle walk here was the residual idle CPU cost.
                        Vec3 mn, mx;
                        if (bodyWorldAABB(ctx, node, mn, mx)) {
                            const Vec3 c = (mn + mx) * 0.5f;
                            mat.m[0][3] = c.x; mat.m[1][3] = c.y; mat.m[2][3] = c.z;
                        }
                    }
                }
            }

            objectMatrix[0] = mat.m[0][0]; objectMatrix[1] = mat.m[1][0]; objectMatrix[2] = mat.m[2][0]; objectMatrix[3] = mat.m[3][0];
            objectMatrix[4] = mat.m[0][1]; objectMatrix[5] = mat.m[1][1]; objectMatrix[6] = mat.m[2][1]; objectMatrix[7] = mat.m[3][1];
            objectMatrix[8] = mat.m[0][2]; objectMatrix[9] = mat.m[1][2]; objectMatrix[10] = mat.m[2][2]; objectMatrix[11] = mat.m[3][2];
            objectMatrix[12] = mat.m[0][3]; objectMatrix[13] = mat.m[1][3]; objectMatrix[14] = mat.m[2][3]; objectMatrix[15] = mat.m[3][3];
        }
    }
    else if (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume) {
        Matrix4x4 mat = sel.selected.gas_volume->getPivotMatrix();
        objectMatrix[0] = mat.m[0][0]; objectMatrix[1] = mat.m[1][0]; objectMatrix[2] = mat.m[2][0]; objectMatrix[3] = mat.m[3][0];
        objectMatrix[4] = mat.m[0][1]; objectMatrix[5] = mat.m[1][1]; objectMatrix[6] = mat.m[2][1]; objectMatrix[7] = mat.m[3][1];
        objectMatrix[8] = mat.m[0][2]; objectMatrix[9] = mat.m[1][2]; objectMatrix[10] = mat.m[2][2]; objectMatrix[11] = mat.m[3][2];
        objectMatrix[12] = mat.m[0][3]; objectMatrix[13] = mat.m[1][3]; objectMatrix[14] = mat.m[2][3]; objectMatrix[15] = mat.m[3][3];
    }
    else if (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) {
        Matrix4x4 mat = sel.selected.vdb_volume->getPivotMatrix();
        objectMatrix[0] = mat.m[0][0]; objectMatrix[1] = mat.m[1][0]; objectMatrix[2] = mat.m[2][0]; objectMatrix[3] = mat.m[3][0];
        objectMatrix[4] = mat.m[0][1]; objectMatrix[5] = mat.m[1][1]; objectMatrix[6] = mat.m[2][1]; objectMatrix[7] = mat.m[3][1];
        objectMatrix[8] = mat.m[0][2]; objectMatrix[9] = mat.m[1][2]; objectMatrix[10] = mat.m[2][2]; objectMatrix[11] = mat.m[3][2];
        objectMatrix[12] = mat.m[0][3]; objectMatrix[13] = mat.m[1][3]; objectMatrix[14] = mat.m[2][3]; objectMatrix[15] = mat.m[3][3];
    }
    else if (sel.selected.type == SelectableType::ForceField && sel.selected.force_field) {
        Matrix4x4 mat = Matrix4x4::fromTRS(sel.selected.force_field->position, sel.selected.force_field->rotation, sel.selected.force_field->scale);

        objectMatrix[0] = mat.m[0][0]; objectMatrix[1] = mat.m[1][0]; objectMatrix[2] = mat.m[2][0]; objectMatrix[3] = mat.m[3][0];
        objectMatrix[4] = mat.m[0][1]; objectMatrix[5] = mat.m[1][1]; objectMatrix[6] = mat.m[2][1]; objectMatrix[7] = mat.m[3][1];
        objectMatrix[8] = mat.m[0][2]; objectMatrix[9] = mat.m[1][2]; objectMatrix[10] = mat.m[2][2]; objectMatrix[11] = mat.m[3][2];
        objectMatrix[12] = mat.m[0][3]; objectMatrix[13] = mat.m[1][3]; objectMatrix[14] = mat.m[2][3]; objectMatrix[15] = mat.m[3][3];
    }
    else if (selected_domain) {
        objectMatrix[0] = startMat.m[0][0]; objectMatrix[1] = startMat.m[1][0]; objectMatrix[2] = startMat.m[2][0]; objectMatrix[3] = startMat.m[3][0];
        objectMatrix[4] = startMat.m[0][1]; objectMatrix[5] = startMat.m[1][1]; objectMatrix[6] = startMat.m[2][1]; objectMatrix[7] = startMat.m[3][1];
        objectMatrix[8] = startMat.m[0][2]; objectMatrix[9] = startMat.m[1][2]; objectMatrix[10] = startMat.m[2][2]; objectMatrix[11] = startMat.m[3][2];
        objectMatrix[12] = startMat.m[0][3]; objectMatrix[13] = startMat.m[1][3]; objectMatrix[14] = startMat.m[2][3]; objectMatrix[15] = startMat.m[3][3];
    }

    // �������������������������������������������������������������������������
    const bool can_edit_single_pivot =
        sel.multi_selection.size() == 1 &&
        ((sel.selected.type == SelectableType::Object && sel.selected.object) ||
         (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) ||
         (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume));

    // Keyboard Shortcuts for Transform Mode
    // �������������������������������������������������������������������������
    // Engage whenever the app is focused; only block while typing in a text field
    // (WantTextInput), not whenever any UI panel has focus (WantCaptureKeyboard).
    if (sel.hasSelection() && !ImGui::GetIO().WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_G)) {
            sel.transform_mode = TransformMode::Translate;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
            sel.transform_mode = TransformMode::Rotate;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_S) && !ImGui::GetIO().KeyShift) {
            // S alone = Scale, Shift+S would trigger duplication so check
            sel.transform_mode = TransformMode::Scale;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_W)) {
            // Cycle through modes
            switch (sel.transform_mode) {
            case TransformMode::Translate: sel.transform_mode = TransformMode::Rotate; break;
            case TransformMode::Rotate: sel.transform_mode = TransformMode::Scale; break;
            case TransformMode::Scale: sel.transform_mode = TransformMode::Translate; break;
            }
        }
        else if (can_edit_single_pivot && ImGui::IsKeyPressed(ImGuiKey_P)) {
            pivot_edit_mode = !pivot_edit_mode;
        }

        // Shift + D = Duplicate Object
        if (ImGui::IsKeyPressed(ImGuiKey_D) && ImGui::GetIO().KeyShift) {
            if (sel.selected.type == SelectableType::Object && sel.selected.object) {
                std::string targetName = sel.selected.object->getNodeName();
                if (targetName.empty()) targetName = "Unnamed";

                // Ensure mesh cache is valid
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                // Unique name generation - USE MESH_CACHE instead of scanning all triangles!
                std::string baseName = targetName;
                size_t lastUnderscore = baseName.rfind('_');
                if (lastUnderscore != std::string::npos) {
                    std::string suffix = baseName.substr(lastUnderscore + 1);
                    if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
                        baseName = baseName.substr(0, lastUnderscore);
                    }
                }

                int counter = 1;
                std::string newName;
                bool nameExists = true;
                while (nameExists) {
                    newName = baseName + "_" + std::to_string(counter);
                    // O(log N) lookup in mesh_cache instead of O(N) triangle scan!
                    nameExists = mesh_cache.find(newName) != mesh_cache.end();
                    counter++;
                }

                // Find source triangles
                auto it = mesh_cache.find(targetName);
                if (it != mesh_cache.end() && !it->second.empty()) {
                    // Pre-allocate for performance
                    size_t numTris = it->second.size();
                    std::vector<std::shared_ptr<Hittable>> newTriangles;
                    newTriangles.reserve(numTris);
                    
                    std::vector<std::pair<int, std::shared_ptr<Triangle>>> newCacheEntries;
                    newCacheEntries.reserve(numTris);

                    std::shared_ptr<Triangle> firstNewTri = nullptr;
                    int baseIndex = (int)ctx.scene.world.objects.size();

                    // Create duplicates
                    for (size_t i = 0; i < numTris; ++i) {
                        auto& oldTri = it->second[i].second;
                        auto newTri = std::make_shared<Triangle>(*oldTri);
                        newTri->setNodeName(newName);
                        newTriangles.push_back(newTri);
                        newCacheEntries.push_back({baseIndex + (int)i, newTri});
                        if (!firstNewTri) firstNewTri = newTri;
                    }

                    // Add to scene
                    ctx.scene.world.objects.insert(ctx.scene.world.objects.end(), newTriangles.begin(), newTriangles.end());
                    
                    // INCREMENTAL CACHE UPDATE (instead of full rebuildMeshCache!)
                    mesh_cache[newName] = std::move(newCacheEntries);
                    mesh_ui_cache.push_back({newName, mesh_cache[newName]});
                    
                    // Calculate bbox for new object (from original since it's a copy)
                    auto orig_bbox = bbox_cache.find(targetName);
                    if (orig_bbox != bbox_cache.end()) {
                        bbox_cache[newName] = orig_bbox->second;
                    }
                    
                    // Copy material slots cache
                    auto orig_mats = material_slots_cache.find(targetName);
                    if (orig_mats != material_slots_cache.end()) {
                        material_slots_cache[newName] = orig_mats->second;
                    }
                    
                    sel.selectObject(firstNewTri, -1, newName);

                    // Record undo command
                    std::vector<std::shared_ptr<Triangle>> new_tri_vec;
                    new_tri_vec.reserve(numTris);
                    for (auto& ht : newTriangles) {
                        auto tri = std::dynamic_pointer_cast<Triangle>(ht);
                        if (tri) new_tri_vec.push_back(tri);
                    }
                    auto command = std::make_unique<DuplicateObjectCommand>(targetName, newName, new_tri_vec);
                    history.record(std::move(command));

                    // ===========================================================
                    // DEFERRED FULL REBUILD (Reliable - async in Main.cpp)
                    // ===========================================================
                    extern bool g_viewport_raster_rebuild_pending;
                    extern bool g_geometry_dirty;
                    extern std::atomic<uint64_t> g_scene_geometry_generation;
                    bool rasterCloneSucceeded = false;
                    if (Backend::IViewportBackend* viewportBackend = dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr)) {
                        rasterCloneSucceeded = viewportBackend->cloneRasterObjectByNodeName(
                            targetName, newName, firstNewTri ? firstNewTri->getTransformMatrix() : Matrix4x4());
                    } else if (g_viewport_backend) {
                        rasterCloneSucceeded = g_viewport_backend->cloneRasterObjectByNodeName(
                            targetName, newName, firstNewTri ? firstNewTri->getTransformMatrix() : Matrix4x4());
                    }
                    g_viewport_raster_rebuild_pending = !rasterCloneSucceeded;
                    g_geometry_dirty = true;
                    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);

                    extern bool g_optix_rebuild_pending;
                    extern bool g_vulkan_rebuild_pending;
                    bool vulkanCloneSucceeded = false;
                    if (Backend::IBackend* renderBackend = getGizmoRenderBackend(ctx)) {
                        if (auto* vkBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(renderBackend)) {
                            vulkanCloneSucceeded = vkBackend->cloneRtObjectByNodeName(
                                targetName, newName, firstNewTri, firstNewTri ? firstNewTri->getTransformMatrix() : Matrix4x4());
                            g_vulkan_rebuild_pending = !vulkanCloneSucceeded;
                        }
                    }
                    bool optixCloneSucceeded = false;
                    if (Backend::IBackend* renderBackend = getGizmoRenderBackend(ctx)) {
                        if (auto* optixBackend = dynamic_cast<Backend::OptixBackend*>(renderBackend)) {
                            auto* optix = optixBackend->getOptixWrapper();
                            if (optix) {
                                const auto newIds = optix->cloneInstancesByNodeName(targetName, newName);
                                if (!newIds.empty()) {
                                    optixBackend->updateObjectTransform(newName, firstNewTri ? firstNewTri->getTransformMatrix() : Matrix4x4());
                                    optixBackend->rebuildAccelerationStructure();
                                    optixBackend->resetAccumulation();
                                    optixCloneSucceeded = true;
                                }
                            }
                        }
                    }
                    g_optix_rebuild_pending = !optixCloneSucceeded;
                    
                    extern bool g_bvh_rebuild_pending;
                    extern int g_bvh_rebuild_deferred_frames;
                    Backend::IBackend* renderBackend = getGizmoRenderBackend(ctx);
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
                    
                    is_bvh_dirty = false;
                    SCENE_LOG_INFO("Duplicated object: " + targetName + " -> " + newName + " (" + std::to_string(numTris) + " triangles)");
                }
            }
        }
    }

    // �������������������������������������������������������������������������
    // Determine Gizmo Operation
    // �������������������������������������������������������������������������
    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
    switch (sel.transform_mode) {
    case TransformMode::Translate: operation = ImGuizmo::TRANSLATE; break;
    case TransformMode::Rotate: operation = ImGuizmo::ROTATE; break;
    case TransformMode::Scale: operation = ImGuizmo::SCALE; break;
    }

    if (!can_edit_single_pivot) {
        pivot_edit_mode = false;
    }
    if (pivot_edit_mode) {
        operation = ImGuizmo::TRANSLATE;
    }

    // Restriction Removed: Fallback logic now handles Rot/Scale for mixed groups


    ImGuizmo::MODE mode = (sel.transform_space == TransformSpace::Local) ?
        ImGuizmo::LOCAL : ImGuizmo::WORLD;
    if (pivot_edit_mode) {
        mode = ImGuizmo::WORLD;
    }
    // �������������������������������������������������������������������������
    // Shift + Drag Duplication Logic + IDLE PREVIEW
    // �������������������������������������������������������������������������
    static bool was_using_gizmo = false;
    static LightState drag_start_light_state;
    static std::shared_ptr<Light> drag_light = nullptr;
    bool is_using = ImGuizmo::IsUsing();
    is_dragging = is_using; // Sync class member

    // IDLE PREVIEW: Track when mouse stops moving during drag
    // NOTE: In TLAS mode, transforms are already updated in real-time via instance matrices.
    // The heavy updateTLASGeometry and rebuildBVH calls here were causing major freezes!
    static ImVec2 last_mouse_pos = ImVec2(0, 0);
    static float idle_time = 0.0f;
    static bool preview_updated = false;
    const float IDLE_THRESHOLD = 0.3f;  // 0.3 seconds before preview update


    if (is_using && is_bvh_dirty) {
        ImVec2 current_mouse = io.MousePos;
        float mouse_delta = sqrtf(powf(current_mouse.x - last_mouse_pos.x, 2) +
            powf(current_mouse.y - last_mouse_pos.y, 2));

        if (mouse_delta < 1.0f) {  // Mouse essentially stationary
            idle_time += io.DeltaTime;

            // If idle for threshold and not yet updated, do preview update
            if (idle_time >= IDLE_THRESHOLD && !preview_updated) {
                // SCENE_LOG_INFO("[GIZMO] Idle preview - updating geometry");
                if (ctx.backend_ptr) {
                    if (ctx.backend_ptr->isUsingTLAS()) {
                        // TLAS MODE: Transforms are ALREADY updated via instance matrices!
                        // Just reset accumulation to show the updated render, NO heavy rebuild.
                        ctx.backend_ptr->resetAccumulation();
                        // Skip rebuildBVH too - picking uses linear search, not BVH.
                    } else {
                        // GAS MODE: Use fast vertex update (legacy)
                        ctx.backend_ptr->updateGeometry(ctx.scene.world.objects);
                        ctx.backend_ptr->setLights(ctx.scene.lights);
                        ctx.backend_ptr->resetAccumulation();
                        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                    }
                } else {
                    // No OptiX: CPU mode still needs BVH
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                }
                ctx.renderer.resetCPUAccumulation();
                preview_updated = true;
                // Note: Don't set is_bvh_dirty = false, so final update still happens
            }
        }
        else {
            // Mouse moved - reset idle tracking
            idle_time = 0.0f;
            preview_updated = false;  // Allow another preview after next pause
        }
        last_mouse_pos = current_mouse;
    }
    else {
        // Not using gizmo - reset tracking
        idle_time = 0.0f;
        preview_updated = false;
        last_mouse_pos = io.MousePos;
    }

    if (is_using && !was_using_gizmo) {
        // ===========================================================================
        // MANIPULATION START
        // ===========================================================================
        if (io.KeyShift && sel.hasSelection()) {
            triggerDuplicate(ctx);
            // Capture start state for the newly created clone
            if (sel.selected.type == SelectableType::Object && sel.selected.object) {
                if (!pivot_edit_mode) {
                    auto transform = sel.selected.object->getTransformHandle();
                    if (transform) {
                        drag_start_state.matrix = transform->base;
                        drag_object_name = sel.selected.object->getNodeName();
                    }
                }
            }
        }
        else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
            // START LIGHT TRANSFORM RECORDING
            drag_light = sel.selected.light;
            drag_start_light_state = LightState::capture(*drag_light);
        }
        else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            // START TRANSFORM RECORDING (Normal drag without Shift)
            if (!pivot_edit_mode) {
                auto transform = sel.selected.object->getTransformHandle();
                if (transform) {
                    drag_start_state.matrix = transform->base;
                    drag_object_name = sel.selected.object->getNodeName();
                }
            }
        }
    }

    // END DRAG (Release)
    if (!is_using && was_using_gizmo) {
        if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            // END TRANSFORM RECORDING
            if (pivot_edit_mode) {
                ProjectManager::getInstance().markModified();
            } else {
                auto t = sel.selected.object->getTransformHandle();
                if (t) {
                    TransformState final_state;
                    final_state.matrix = t->base;

                    // Check delta
                    bool changed = false;
                    for (int i = 0; i < 4; ++i)
                        for (int j = 0; j < 4; ++j)
                            if (std::abs(final_state.matrix.m[i][j] - drag_start_state.matrix.m[i][j]) > 0.0001f)
                                changed = true;

                    if (changed) {
                        history.record(std::make_unique<TransformCommand>(drag_object_name, drag_start_state, final_state));
                        ProjectManager::getInstance().markModified();
                    }
                }
            }
        }
        else if (pivot_edit_mode &&
                 ((sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) ||
                  (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume))) {
            ProjectManager::getInstance().markModified();
        }
        else if (sel.selected.type == SelectableType::Light && drag_light) {
            LightState final_light_state = LightState::capture(*drag_light);

            // Check if position or other properties changed
            bool changed = (final_light_state.position - drag_start_light_state.position).length() > 0.0001f ||
                (final_light_state.direction - drag_start_light_state.direction).length() > 0.0001f ||
                std::abs(final_light_state.angle - drag_start_light_state.angle) > 0.0001f;

            if (changed) {
                history.record(std::make_unique<TransformLightCommand>(drag_light, drag_start_light_state, final_light_state));
                ProjectManager::getInstance().markModified();
            }
            drag_light = nullptr;
        }
    }

    // NOTE: was_using_gizmo update moved to END of function (after is_bvh_dirty is set)

    // �������������������������������������������������������������������������
    // Render and Manipulate Gizmo
    // �������������������������������������������������������������������������
    // Save old position BEFORE manipulation for delta calculation (multi-selection)
    // Save old position & MATRIX BEFORE manipulation for delta calculation
    Vec3 oldGizmoPos(objectMatrix[12], objectMatrix[13], objectMatrix[14]);

    Matrix4x4 oldMat;
    oldMat.m[0][0] = objectMatrix[0]; oldMat.m[1][0] = objectMatrix[1]; oldMat.m[2][0] = objectMatrix[2]; oldMat.m[3][0] = objectMatrix[3];
    oldMat.m[0][1] = objectMatrix[4]; oldMat.m[1][1] = objectMatrix[5]; oldMat.m[2][1] = objectMatrix[6]; oldMat.m[3][1] = objectMatrix[7];
    oldMat.m[0][2] = objectMatrix[8]; oldMat.m[1][2] = objectMatrix[9]; oldMat.m[2][2] = objectMatrix[10]; oldMat.m[3][2] = objectMatrix[11];
    oldMat.m[0][3] = objectMatrix[12]; oldMat.m[1][3] = objectMatrix[13]; oldMat.m[2][3] = objectMatrix[14]; oldMat.m[3][3] = objectMatrix[15];

    // Sim-driven bodies suppress the interactive gizmo (no per-frame vertex walk);
    // the live bbox in drawSelectionBoundingBox stands in for it while playing.
    bool manipulated = skip_body_gizmo || rtpython::wantsInputCapture() || rtapi::renderOutputPending()
        ? false
        : ImGuizmo::Manipulate(viewMatrix, projMatrix, operation, mode, objectMatrix);

    if (manipulated && is_mixed_group) {
        // Update persistent matrix for next frame interaction
        mixed_gizmo_matrix.m[0][0] = objectMatrix[0]; mixed_gizmo_matrix.m[1][0] = objectMatrix[1]; mixed_gizmo_matrix.m[2][0] = objectMatrix[2]; mixed_gizmo_matrix.m[3][0] = objectMatrix[3];
        mixed_gizmo_matrix.m[0][1] = objectMatrix[4]; mixed_gizmo_matrix.m[1][1] = objectMatrix[5]; mixed_gizmo_matrix.m[2][1] = objectMatrix[6]; mixed_gizmo_matrix.m[3][1] = objectMatrix[7];
        mixed_gizmo_matrix.m[0][2] = objectMatrix[8]; mixed_gizmo_matrix.m[1][2] = objectMatrix[9]; mixed_gizmo_matrix.m[2][2] = objectMatrix[10]; mixed_gizmo_matrix.m[3][2] = objectMatrix[11];
        mixed_gizmo_matrix.m[0][3] = objectMatrix[12]; mixed_gizmo_matrix.m[1][3] = objectMatrix[13]; mixed_gizmo_matrix.m[2][3] = objectMatrix[14]; mixed_gizmo_matrix.m[3][3] = objectMatrix[15];
    }

    if (manipulated) {
        // Mark objects for lazy CPU sync (re-enable picking accuracy later)
        if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            objects_needing_cpu_sync.insert(sel.selected.object->getNodeName());
        }
        for (const auto& item : sel.multi_selection) {
            if (item.type == SelectableType::Object && item.object) {
                objects_needing_cpu_sync.insert(item.object->getNodeName());
            }
        }

        Vec3 newPos(objectMatrix[12], objectMatrix[13], objectMatrix[14]);

        // -------------------------------------------------------------------------
        // SINGULARITY FIX: Clamp extreme movements (Axis parallel to Camera View)
        // -------------------------------------------------------------------------
        if (operation == ImGuizmo::TRANSLATE) {
            // -------------------------------------------------------------------------
            // STABILITY FIX: Zero-Drift Check (If mouse is still, object stays still)
            // -------------------------------------------------------------------------
            float mouse_delta_sq = io.MouseDelta.x * io.MouseDelta.x + io.MouseDelta.y * io.MouseDelta.y;
            if (mouse_delta_sq < 0.01f) {
                // Determine if we should force-reset the position
                // If the user's hand is steady, we reject ANY jitter from the gizmo projection math.
                newPos = oldGizmoPos;
                objectMatrix[12] = oldGizmoPos.x;
                objectMatrix[13] = oldGizmoPos.y;
                objectMatrix[14] = oldGizmoPos.z;
            } else {   

            float dist_to_cam = (oldGizmoPos - cam.lookfrom).length();
            // Estimate world size of 1 pixel at object depth. Under orthographic projection the
            // pixel size is constant (set by ortho_height), independent of depth.
            float pixel_world_size = gizmoUseOrtho
                ? (((cam.ortho_height > 1e-4f) ? cam.ortho_height : 10.0f) / io.DisplaySize.y)
                : (dist_to_cam * (2.0f * tan_half_fov) / io.DisplaySize.y);

            float mouse_move_len = sqrtf(io.MouseDelta.x * io.MouseDelta.x + io.MouseDelta.y * io.MouseDelta.y);
            if (mouse_move_len < 1.0f) mouse_move_len = 1.0f;

            // -------------------------------------------------------------------------
            // CONSISTENT SPEED LIMIT (Fixes "Acceleration" feel)
            // -------------------------------------------------------------------------
            // Instead of variable damping, we enforce a strict "max world-units per mouse-pixel" limit.
            // This ensures the object moves linearly with the hand, destroying the singularity.

            Vec3 move_vector = newPos - oldGizmoPos;
            
            // -------------------------------------------------------------------------
            // DIRECTION CORRECTION (Fixes "Stuck" wrong-way movement)
            // -------------------------------------------------------------------------
            // In singularities, 3D projection can flip sign (moving mouse Right goes -Z instead of +Z).
            // We project the 3D move back to 2D and compare with Mouse Delta.
            ImVec2 s1 = Project(oldGizmoPos);
            ImVec2 s2 = Project(newPos);
            
            if (s1.x > -5000 && s2.x > -5000) { // Valid projection
                float dx = s2.x - s1.x;
                float dy = s2.y - s1.y;
                float dot = dx * io.MouseDelta.x + dy * io.MouseDelta.y;
                
                // If dot < 0, the visual movement opposes the mouse movement!
                if (dot < -0.01f) {
                    move_vector = move_vector * -1.0f; // FLIP DIRECTION
                    newPos = oldGizmoPos + move_vector; // Update target
                }
            }
            
            float world_move_dist = move_vector.length();

            // Calculate "World Units per Mouse Pixel" ratio
            // If this ratio is huge (e.g. 100.0), it means 1 pixel movement caused 100 units jump (Singularity).
            // Normal 1:1 interaction is roughly ratio ~ 1.0 (relative to pixel_world_size).
            
            float safe_ratio = 4.0f; // Base speed multiplier

            // DYNAMIC RATIO ADJUSTMENT:
            // If movement is parallel to camera view (Singularity Case), strictly reduce the ratio.
            // This prevents "runaway sensitivity" when dragging objects far away along the Z-axis.
            if (world_move_dist > 0.0001f) {
                Vec3 move_dir_norm = move_vector / world_move_dist;
                Vec3 cam_dir = (oldGizmoPos - cam.lookfrom).normalize();
                float dot = fabsf(move_dir_norm.dot(cam_dir)); // 0.0 = Perpendicular, 1.0 = Parallel

                if (dot > 0.7f) {
                    // Linearly reduce ratio from 4.0 to 1.0 as angle becomes parallel
                    // 0.7 -> 4.0
                    // 1.0 -> 1.0
                    float t = (dot - 0.7f) / 0.3f; // 0..1
                    safe_ratio = 4.0f * (1.0f - t) + 1.0f * t;
                }
            }
            
            float max_allowed_dist = mouse_move_len * pixel_world_size * safe_ratio;

            if (world_move_dist > max_allowed_dist) {
                // The projection wants to move too fast. Clamp it to the speed limit.
                // We preserve direction but limit magnitude.
                Vec3 dir = move_vector.normalize();
                newPos = oldGizmoPos + dir * max_allowed_dist;

                // Sync matrix
                objectMatrix[12] = newPos.x;
                objectMatrix[13] = newPos.y;
                objectMatrix[14] = newPos.z;
            }
            } // End of stationary check
        }
        // -------------------------------------------------------------------------
        Vec3 deltaPos = newPos - oldGizmoPos;  // Calculate delta from BEFORE manipulation
        sel.selected.position = newPos; // Update gizmo/bbox center

        // CRITICAL FIX: Update rotation and scale from object's transform matrix
        // This ensures keyframes capture correct rotation/scale values
        if (sel.selected.type == SelectableType::Object && sel.selected.object) {
                auto transformHandle = sel.selected.object->getTransformHandle();
                if (transformHandle) {
                Matrix4x4 objTransform = transformHandle->getPivotMatrix();

                // Extract rotation (Euler angles in degrees)
                // Assuming rotation order: Z * Y * X
                float sy = sqrtf(objTransform.m[0][0] * objTransform.m[0][0] + objTransform.m[1][0] * objTransform.m[1][0]);
                bool singular = sy < 1e-6f;

                if (!singular) {
                    sel.selected.rotation.x = atan2f(objTransform.m[2][1], objTransform.m[2][2]) * (180.0f / 3.14159265f);
                    sel.selected.rotation.y = atan2f(-objTransform.m[2][0], sy) * (180.0f / 3.14159265f);
                    sel.selected.rotation.z = atan2f(objTransform.m[1][0], objTransform.m[0][0]) * (180.0f / 3.14159265f);
                }
                else {
                    sel.selected.rotation.x = atan2f(-objTransform.m[1][2], objTransform.m[1][1]) * (180.0f / 3.14159265f);
                    sel.selected.rotation.y = atan2f(-objTransform.m[2][0], sy) * (180.0f / 3.14159265f);
                    sel.selected.rotation.z = 0.0f;
                }

                // Extract scale
                sel.selected.scale.x = sqrtf(objTransform.m[0][0] * objTransform.m[0][0] +
                    objTransform.m[1][0] * objTransform.m[1][0] +
                    objTransform.m[2][0] * objTransform.m[2][0]);
                sel.selected.scale.y = sqrtf(objTransform.m[0][1] * objTransform.m[0][1] +
                    objTransform.m[1][1] * objTransform.m[1][1] +
                    objTransform.m[2][1] * objTransform.m[2][1]);
                sel.selected.scale.z = sqrtf(objTransform.m[0][2] * objTransform.m[0][2] +
                    objTransform.m[1][2] * objTransform.m[1][2] +
                    objTransform.m[2][2] * objTransform.m[2][2]);
            }
        }

        // Check if this is multi-selection (handle mixed types: lights + objects together)
        bool is_multi_select = sel.multi_selection.size() > 1;

        if (is_multi_select) {
            // MULTI-SELECTION: Apply delta to ALL selected items (mixed types)
            float deltaMagnitude = sqrtf(deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y + deltaPos.z * deltaPos.z);

            // For Rotation/Scale, deltaPos might be zero, so we check operation too
            if (deltaMagnitude >= 0.0001f || operation != ImGuizmo::TRANSLATE) {
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                // Calculate Delta Matrix for Rotation/Scale
                Matrix4x4 newMat;
                newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
                newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
                newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
                newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

                Matrix4x4 deltaMat = newMat * oldMat.inverse();

                // Decompose
                Vec3 deltaTranslation(deltaMat.m[0][3], deltaMat.m[1][3], deltaMat.m[2][3]);
                Matrix4x4 deltaRotScale = deltaMat;
                deltaRotScale.m[0][3] = 0; deltaRotScale.m[1][3] = 0; deltaRotScale.m[2][3] = 0;

                for (auto& item : sel.multi_selection) {
                    if (item.type == SelectableType::Object && item.object) {
                        std::string targetName = item.object->getNodeName();
                        if (targetName.empty()) targetName = "Unnamed";

                        auto th = item.object->getTransformHandle();
                        if (th) {
                            Matrix4x4 pivotMat = th->getPivotMatrix();
                            // Apply transform to the shared handle
                            if (pivot_mode == 1) {
                                // Individual Origins
                                Vec3 pos(pivotMat.m[0][3], pivotMat.m[1][3], pivotMat.m[2][3]);
                                pivotMat.m[0][3] = 0; pivotMat.m[1][3] = 0; pivotMat.m[2][3] = 0;
                                Matrix4x4 updated = deltaRotScale * pivotMat;
                                updated.m[0][3] = pos.x + deltaTranslation.x;
                                updated.m[1][3] = pos.y + deltaTranslation.y;
                                updated.m[2][3] = pos.z + deltaTranslation.z;
                                th->setPivotMatrix(updated);
                            }
                            else {
                                // Median Point
                                th->setPivotMatrix(deltaMat * pivotMat);
                            }
                            
                            // GPU INSTANCE UPDATE (TLAS + Raster)
                            updateGizmoObjectTransformOnActiveBackends(ctx, targetName, th->base);
                            // CPU mode handled on release
                        }
                        item.has_cached_aabb = false;
                    }
                    else if (item.type == SelectableType::Light && item.light) {
                        item.light->position = item.light->position + deltaPos;
                    }
                    else if (item.type == SelectableType::Camera && item.camera) {
                        // Skip active camera - moving it would affect viewport
                        if (item.camera != ctx.scene.camera) {
                            item.camera->lookfrom = item.camera->lookfrom + deltaPos;
                            item.camera->lookat = item.camera->lookat + deltaPos;
                            item.camera->update_camera_vectors();
                        }
                    }
                    else if (item.type == SelectableType::GasVolume && item.gas_volume) {
                        item.gas_volume->setPosition(item.gas_volume->getPosition() + deltaPos);
                    }
                    else if (item.type == SelectableType::ForceField && item.force_field) {
                        item.force_field->position = item.force_field->position + deltaPos;
                        
                        if (operation == ImGuizmo::ROTATE) {
                            // Apply rotation delta to direction vector (manual matrix multiply)
                            Vec3 d = item.force_field->direction;
                            Vec3 newDir(
                                deltaRotScale.m[0][0]*d.x + deltaRotScale.m[0][1]*d.y + deltaRotScale.m[0][2]*d.z,
                                deltaRotScale.m[1][0]*d.x + deltaRotScale.m[1][1]*d.y + deltaRotScale.m[1][2]*d.z,
                                deltaRotScale.m[2][0]*d.x + deltaRotScale.m[2][1]*d.y + deltaRotScale.m[2][2]*d.z
                            );
                            item.force_field->direction = newDir.normalize();
                            
                            // Also update vortex axis if applicable
                            if (item.force_field->type == Physics::ForceFieldType::Vortex) {
                                Vec3 a = item.force_field->axis;
                                Vec3 newAxis(
                                    deltaRotScale.m[0][0]*a.x + deltaRotScale.m[0][1]*a.y + deltaRotScale.m[0][2]*a.z,
                                    deltaRotScale.m[1][0]*a.x + deltaRotScale.m[1][1]*a.y + deltaRotScale.m[1][2]*a.z,
                                    deltaRotScale.m[2][0]*a.x + deltaRotScale.m[2][1]*a.y + deltaRotScale.m[2][2]*a.z
                                );
                                item.force_field->axis = newAxis.normalize();
                            }
                        }
                    }
                } // End of multi_selection loop

                // Trigger accumulation reset after processing all objects
                if (Backend::IViewportBackend* viewportBackend = getGizmoViewportBackend(ctx)) {
                    viewportBackend->resetAccumulation();
                }
                if (Backend::IBackend* renderBackend = getGizmoRenderBackend(ctx)) {
                    if (renderBackend->isUsingTLAS()) {
                        renderBackend->resetAccumulation();
                    }
                }

                sel.selected.has_cached_aabb = false;

                // DEFERRED UPDATE: Only mark dirty during drag (for CPU mode)
                bool using_gpu_tlas = ctx.backend_ptr && ctx.backend_ptr->isUsingTLAS();
                if (!using_gpu_tlas) {
                    is_bvh_dirty = true;
                }
            }
        }
        else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
            sel.selected.light->position = newPos;
            Vec3 zAxis(objectMatrix[8], objectMatrix[9], objectMatrix[10]);
            Vec3 newDir = -zAxis.normalize(); // Gizmo -Z aligned

            if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(sel.selected.light)) dl->setDirection(newDir);
            else if (auto sl = std::dynamic_pointer_cast<SpotLight>(sel.selected.light)) {
                sl->direction = newDir;

                // Update Angle from Gizmo Scale
                Vec3 right(objectMatrix[0], objectMatrix[1], objectMatrix[2]);
                float angle = right.length();

                if (angle < 0.1f) angle = 0.1f;
                if (angle > 179.0f) angle = 179.0f;

                sl->setAngleDegrees(angle);

                // Falloff Update (Z Scale represents 1.0 + Falloff)
                Vec3 forward(objectMatrix[8], objectMatrix[9], objectMatrix[10]);
                float sz = forward.length();
                float newF = sz - 1.0f;
                if (newF < 0.0f) newF = 0.0f;
                if (newF > 1.0f) newF = 1.0f;
                sl->setFalloff(newF);
            }
            else if (auto al = std::dynamic_pointer_cast<AreaLight>(sel.selected.light)) {
                Vec3 right(objectMatrix[0], objectMatrix[1], objectMatrix[2]);
                Vec3 forward(objectMatrix[8], objectMatrix[9], objectMatrix[10]);

                // Scale bilgisini vekt�rlerden ��kar
                float sx = right.length();
                float sz = forward.length();

                // Width ve Height g�ncelle
                if (sx > 0.001f) al->setWidth(sx);
                if (sz > 0.001f) al->setHeight(sz);

                // u ve v HER ZAMAN normalize tutulmal�!
                Vec3 new_u = al->getU();
                Vec3 new_v = al->getV();
                if (sx > 0.001f) new_u = (right / sx).normalize();
                if (sz > 0.001f) new_v = (forward / sz).normalize();
                al->setUVVectors(new_u, new_v);

                // Position do�rudan gizmo merkezinden al�nmal� (art�k position = merkez)
                al->position = newPos;
            }
            if (ctx.backend_ptr) {
                ctx.backend_ptr->setLights(ctx.scene.lights);
                ctx.backend_ptr->resetAccumulation();
            }
        }
        else if (sel.selected.type == SelectableType::ForceField && sel.selected.force_field) {
            // Apply matrix to Force Field
            Matrix4x4 newMat;
            newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
            newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
            newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
            newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

            Vec3 p, r, s;
            newMat.decompose(p, r, s);
            
            sel.selected.force_field->position = p;
            sel.selected.force_field->rotation = r;
            sel.selected.force_field->scale = s;
            
            // Update selection cache
            sel.selected.position = p;
            sel.selected.rotation = r;
            sel.selected.scale = s;
            
            // Removed direction override here, because ForceField direction is a LOCAL property
            // and should only be explicitly set in the UI, not implicitly by gizmo rotation
            // (The Gizmo rotation already rotates the final world force).

            if (ctx.backend_ptr) {
                ctx.backend_ptr->resetAccumulation();
            }
        }
        else if (selected_domain) {
            const Vec3 extent(
                std::max(0.001f, Vec3(objectMatrix[0], objectMatrix[1], objectMatrix[2]).length()),
                std::max(0.001f, Vec3(objectMatrix[4], objectMatrix[5], objectMatrix[6]).length()),
                std::max(0.001f, Vec3(objectMatrix[8], objectMatrix[9], objectMatrix[10]).length()));
            const Vec3 half_extent = extent * 0.5f;
            selected_domain->source_mode = RayTrophiSim::SimulationGridDomainSourceMode::ManualBox;
            selected_domain->source_name.clear();
            selected_domain->bounds_min = newPos - half_extent;
            selected_domain->bounds_max = newPos + half_extent;
            sel.selected.position = newPos;
            sel.selected.rotation = Vec3(0.0f);
            sel.selected.scale = extent;

            if (ctx.backend_ptr) {
                ctx.backend_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();
        }
        else if (sel.selected.type == SelectableType::Camera && sel.selected.camera) {
            // Skip active camera - moving it would affect viewport directly
            if (sel.selected.camera != ctx.scene.camera) {
                Vec3 delta = newPos - sel.selected.camera->lookfrom;
                sel.selected.camera->lookfrom = newPos;
                sel.selected.camera->lookat = sel.selected.camera->lookat + delta;
                sel.selected.camera->update_camera_vectors();
                // Note: Don't call setCameraParams for non-active cameras
            }
        }
        else if (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) {
             Matrix4x4 newMat;
             newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
             newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
             newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
             newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

             Vec3 p, r, s;
             if (pivot_edit_mode) {
                 Matrix4x4 renderMat = sel.selected.vdb_volume->getTransform();
                 Vec3 newPivotLocal = renderMat.inverse().transform_point(newPos);
                 sel.selected.vdb_volume->setPivotOffset(newPivotLocal);
                 Matrix4x4 pivotMat = sel.selected.vdb_volume->getPivotMatrix();
                 pivotMat.decompose(p, r, s);
             } else {
                 sel.selected.vdb_volume->setPivotMatrix(newMat);
                 newMat.decompose(p, r, s);
             }
             sel.selected.position = p;
             sel.selected.rotation = r;
             sel.selected.scale = s;
             
             // VDB is a BOX: a transform move only SHIFTS its AABB — topology is
             // stable — so a cheap CPU-BVH REFIT (Embree RTC_BUILD_QUALITY_REFIT)
             // suffices for picking. The old full rebuild (g_bvh_rebuild_pending)
             // rebuilt the WHOLE scene BVH every drag frame, which stalled dense
             // (foam/sim) scenes. The GPU side only needs the volume-table transform
             // refreshed (syncVDBVolumesToGPU → updateVDBVolumeBuffer = a small device
             // memcpy; NO grid re-upload, NO acceleration-structure rebuild). This
             // mirrors the mesh-object move fast path.
             extern bool g_cpu_bvh_refit_pending;
             g_cpu_bvh_refit_pending = true;

             ctx.renderer.resetCPUAccumulation();

             if (ctx.backend_ptr) {
                 SceneUI::syncVDBVolumesToGPU(ctx);
                 ctx.backend_ptr->resetAccumulation();
             }
        }
        else if (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume) {
             Matrix4x4 newMat;
             newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
             newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
             newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
             newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

             Vec3 p, r, s;
             if (pivot_edit_mode) {
                 Matrix4x4 renderMat = sel.selected.gas_volume->getTransform();
                 Vec3 newPivotLocal = renderMat.inverse().transform_point(newPos);
                 sel.selected.gas_volume->setPivotOffset(newPivotLocal);
                 Matrix4x4 pivotMat = sel.selected.gas_volume->getPivotMatrix();
                 pivotMat.decompose(p, r, s);
             } else {
                 sel.selected.gas_volume->setPivotMatrix(newMat);
                 newMat.decompose(p, r, s);
             }
             
             // Update selection struct to match new transform
             sel.selected.position = p;
             sel.selected.rotation = r;
             sel.selected.scale = s;
             
             ctx.renderer.resetCPUAccumulation();
             
             if (ctx.backend_ptr) {
                 if (sel.selected.gas_volume->render_path == GasVolume::VolumeRenderPath::VDBUnified) {
                     SceneUI::syncVDBVolumesToGPU(ctx);
                 } else {
                     ctx.renderer.updateBackendGasVolumes(ctx.scene);
                 }
                 ctx.backend_ptr->resetAccumulation();
             }
        }
        else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            // SINGLE SELECTION or Rotate/Scale operations
            // (Multi-select TRANSLATE is handled above)

            Matrix4x4 newMat;
            newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
            newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
            newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
            newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

            float deltaMagnitude = sqrtf(deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y + deltaPos.z * deltaPos.z);

            // Only apply transform if there's significant movement or it's rotate/scale
            if (deltaMagnitude >= 0.0001f || operation != ImGuizmo::TRANSLATE) {
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                std::string targetName = sel.selected.object->getNodeName();
                if (targetName.empty()) targetName = "Unnamed";

                if (pivot_edit_mode) {
                    moveObjectPivot(ctx, targetName, deltaPos);
                    sel.selected.position = newPos;
                    sel.selected.has_cached_aabb = false;
                } else {
                    auto t_handle = sel.selected.object->getTransformHandle();
                    if (t_handle) {
                        // Apply full matrix from gizmo (supports translate, rotate, scale)
                        t_handle->setPivotMatrix(newMat);

                        // GPU INSTANCE UPDATE (TLAS + Raster)
                        updateGizmoObjectTransformOnActiveBackends(ctx, targetName, t_handle->base);

                        if (!getGizmoRenderBackend(ctx) || !getGizmoRenderBackend(ctx)->isUsingTLAS()) {
                            // CPU/GAS MODE: Update CPU vertices for triangles sharing this transform handle
                            auto it = mesh_cache.find(targetName);
                            if (it != mesh_cache.end()) {
                                for (auto& pair : it->second) {
                                    if (pair.second && pair.second->getTransformHandle().get() == t_handle.get()) {
                                        pair.second->updateTransformedVertices();
                                    }
                                }
                            }

                            is_bvh_dirty = true;

                            // Trigger Fast Refit during interaction (CPU Mode only)
                            extern bool g_cpu_bvh_refit_pending;
                            g_cpu_bvh_refit_pending = true;
                        }
                    }
                }

                sel.selected.has_cached_aabb = false;

                // DEFERRED UPDATE: Mark dirty when NOT using GPU ray-tracing backend
                // Viewport backend (raster preview) does NOT count as GPU ray-tracing
                bool using_gpu_raytracing =
                    (getGizmoRenderBackend(ctx) && getGizmoRenderBackend(ctx)->isUsingTLAS());
                if (!using_gpu_raytracing) {
                    is_bvh_dirty = true;
                    extern bool g_cpu_bvh_refit_pending;
                    g_cpu_bvh_refit_pending = true;
                    // Reset CPU accumulation so the render loop re-enters
                    ctx.renderer.resetCPUAccumulation();
                }
            }
        }
    }

    // DEFERRED BVH UPDATE: Rebuild when gizmo drag ends (not during)
    // This check is at the END so is_bvh_dirty has been set above
    if (!is_using && was_using_gizmo && is_bvh_dirty) {
        // SCENE_LOG_INFO("[GIZMO] Released - Triggering deferred geometry update");
        Backend::IViewportBackend* viewportBackend = getGizmoViewportBackend(ctx);
        Backend::IBackend* renderBackend = getGizmoRenderBackend(ctx);
        const bool using_gpu = (viewportBackend != nullptr) || (renderBackend != nullptr);
        const bool active_interactive_viewport_backend = (viewportBackend != nullptr);
        const bool active_vulkan_render_backend =
            (renderBackend && dynamic_cast<Backend::VulkanBackendAdapter*>(renderBackend) != nullptr);

        // Handle viewport backend (raster preview) independently
        if (active_interactive_viewport_backend) {
            viewportBackend->resetAccumulation();
        }

        const bool render_backend_has_tlas = (renderBackend && renderBackend->isUsingTLAS());
        if (render_backend_has_tlas) {
            if (active_interactive_viewport_backend) {
                // Solid viewport already received live transform updates.
                if (active_vulkan_render_backend) {
                    // Object move is topology-stable and the per-frame
                    // updateObjectTransform during drag already committed the new
                    // transform to the TLAS (m_device->updateTLAS = instance refit
                    // on an allowUpdate TLAS). A full geometry rebuild here is
                    // redundant and stalls dense scenes (~2s on 4.2M tris). Skip it.
                } else if (renderBackend) {
                    extern bool g_optix_rebuild_pending;
                    g_optix_rebuild_pending = true;
                }
            } else {
                if (active_vulkan_render_backend) {
                    // Pure Rendered (Vulkan RT): live TLAS refit during drag already
                    // committed the move; no full rebuild needed (topology-stable).
                } else {
                    // OptiX / other TLAS mode: commit transform changes
                    ctx.backend_ptr->rebuildAccelerationStructure();
                }
            }
        } else if (renderBackend) {
            // GAS MODE: Defer update to Main loop to avoid UI freeze
            extern bool g_gpu_refit_pending;
            g_gpu_refit_pending = true;

            // Object move is topology-stable → a fast CPU BVH refit keeps picking
            // accurate without the full async rebuild (+ "Rebuilding BVH..." HUD).
            extern bool g_cpu_bvh_refit_pending;
            g_cpu_bvh_refit_pending = true;
        } else {
            // CPU-only mode: object move is topology-stable → refit (not full rebuild).
            extern bool g_cpu_bvh_refit_pending;
            g_cpu_bvh_refit_pending = true;
        }

        is_bvh_dirty = false;
    }
    
    // LAZY CPU SYNC: Mark objects for later sync instead of updating now
    // This makes gizmo release INSTANT - sync happens when user tries to pick something
    if (!is_using && was_using_gizmo) {
        bool using_gpu_tlas =
            (getGizmoRenderBackend(ctx) && getGizmoRenderBackend(ctx)->isUsingTLAS());
        
        if (sel.multi_selection.size() > 0) {
            bool any_object_moved = false;
            for (auto& item : sel.multi_selection) {
                if (item.type == SelectableType::Object && item.object) {
                    std::string name = item.object->getNodeName();
                    if (name.empty()) name = "Unnamed";

                    if (using_gpu_tlas) {
                        // TLAS mode: Just mark for lazy sync (instant release!)
                        objects_needing_cpu_sync.insert(name);
                    } else {
                        // CPU mode: Need immediate update for rendering
                        auto cache_it = mesh_cache.find(name);
                        if (cache_it != mesh_cache.end()) {
                            for (auto& pair : cache_it->second) {
                                pair.second->updateTransformedVertices();
                            }
                        }
                    }
                    any_object_moved = true;
                }
            }
            // BVH update needed only when geometry (objects) actually moved — lights/cameras don't affect BVH.
            // CPU mode only: verts were just updated above, so refit the picking/CPU-render BVH now.
            // TLAS mode (Vulkan RT / OptiX): verts are deferred to lazy pick-sync, so refitting here
            // would scan the WHOLE scene (O(all tris)) against STALE verts — wasted work + a multi-second
            // CPU spike on dense scenes. ensureCPUSyncForPicking() rebuilds the BVH on first pick instead.
            if (any_object_moved && !using_gpu_tlas) {
                extern bool g_cpu_bvh_refit_pending;
                g_cpu_bvh_refit_pending = true;
            }
        } else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            std::string name = sel.selected.object->getNodeName();
            if (name.empty()) name = "Unnamed";
            
            if (using_gpu_tlas) {
                // TLAS mode: Just mark for lazy sync (instant release!)
                objects_needing_cpu_sync.insert(name);
                SCENE_LOG_INFO("Marked for lazy sync: " + name);
            } else {
                // CPU mode: Need immediate update
                auto cache_it = mesh_cache.find(name);
                if (cache_it != mesh_cache.end()) {
                    for (auto& pair : cache_it->second) {
                        pair.second->updateTransformedVertices();
                    }
                }
            }
            // CPU mode only: verts updated above → refit the picking/CPU-render BVH now.
            // TLAS mode defers verts to lazy pick-sync; an immediate full-scene refit here
            // would run against stale verts and spike the CPU for seconds on dense scenes.
            if (!using_gpu_tlas) {
                extern bool g_cpu_bvh_refit_pending;
                g_cpu_bvh_refit_pending = true;
            }
        }
    }

    // Update gizmo state tracking at the END of the function
    was_using_gizmo = is_using;
}

// ===============================================================================
// CAMERA GIZMOS - Draw camera icons in viewport
// ===============================================================================
void SceneUI::drawCameraGizmos(UIContext& ctx) {
    if (!ctx.scene.camera || ctx.scene.cameras.size() <= 1) return;

    Camera& activeCam = *ctx.scene.camera;
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();  // Changed from ForegroundDrawList to render behind UI panels
    ImGuiIO& io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;

    // Camera basis vectors for projection
    Vec3 cam_forward = (activeCam.lookat - activeCam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(activeCam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    float tan_half_fov = tan(activeCam.vfov * 0.5f * M_PI / 180.0f);
    float aspect = (image_height > 0)
        ? (static_cast<float>(image_width) / static_cast<float>(image_height))
        : (screen_w / screen_h);

    const bool gizmoOrtho = activeCam.orthographic && viewport_settings.shading_mode != 2;
    // Lambda to project 3D point to screen (ortho-aware)
    auto Project = [&](const Vec3& world_pos, ImVec2& screen_pos) -> bool {
        return projectGizmoWorldPoint(activeCam, gizmoOrtho, aspect, screen_w, screen_h, world_pos, screen_pos);
        };

    // Draw each non-active camera with 3D frustum
    for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
        if (i == ctx.scene.active_camera_index) continue;  // Skip active camera

        auto& cam = ctx.scene.cameras[i];
        if (!cam) continue;

        // Check if this camera is selected
        bool is_selected = (ctx.selection.hasSelection() &&
            ctx.selection.selected.type == SelectableType::Camera &&
            ctx.selection.selected.camera == cam);

        // Frustum colors
        ImU32 frustum_color = is_selected ? IM_COL32(255, 200, 50, 220) : IM_COL32(100, 200, 255, 180);
        ImU32 body_color = is_selected ? IM_COL32(255, 200, 50, 255) : IM_COL32(80, 150, 255, 255);
        float line_thickness = is_selected ? 2.5f : 1.5f;

        // Calculate frustum vertices in world space
        Vec3 cam_pos = cam->lookfrom;
        Vec3 look_dir = (cam->lookat - cam->lookfrom).normalize();
        Vec3 cam_right = look_dir.cross(cam->vup).normalize();
        Vec3 cam_up = cam_right.cross(look_dir).normalize();

        // Frustum dimensions at near and far planes
        float frustum_length = 1.5f;  // Length of frustum visualization
        float near_dist = 0.2f;
        float far_dist = frustum_length;
        float cam_fov_rad = cam->vfov * M_PI / 180.0f;
        float cam_aspect = (image_height > 0)
            ? (static_cast<float>(image_width) / static_cast<float>(image_height))
            : (screen_w / screen_h);

        float near_height = near_dist * tan(cam_fov_rad * 0.5f);
        float near_width = near_height * cam_aspect;
        float far_height = far_dist * tan(cam_fov_rad * 0.5f);
        float far_width = far_height * cam_aspect;

        // Near plane corners
        Vec3 near_center = cam_pos + look_dir * near_dist;
        Vec3 near_tl = near_center + cam_up * near_height - cam_right * near_width;
        Vec3 near_tr = near_center + cam_up * near_height + cam_right * near_width;
        Vec3 near_bl = near_center - cam_up * near_height - cam_right * near_width;
        Vec3 near_br = near_center - cam_up * near_height + cam_right * near_width;

        // Far plane corners
        Vec3 far_center = cam_pos + look_dir * far_dist;
        Vec3 far_tl = far_center + cam_up * far_height - cam_right * far_width;
        Vec3 far_tr = far_center + cam_up * far_height + cam_right * far_width;
        Vec3 far_bl = far_center - cam_up * far_height - cam_right * far_width;
        Vec3 far_br = far_center - cam_up * far_height + cam_right * far_width;

        // Project all points
        ImVec2 p_cam, p_near_tl, p_near_tr, p_near_bl, p_near_br;
        ImVec2 p_far_tl, p_far_tr, p_far_bl, p_far_br;

        bool visible = Project(cam_pos, p_cam);
        visible &= Project(near_tl, p_near_tl) && Project(near_tr, p_near_tr);
        visible &= Project(near_bl, p_near_bl) && Project(near_br, p_near_br);
        visible &= Project(far_tl, p_far_tl) && Project(far_tr, p_far_tr);
        visible &= Project(far_bl, p_far_bl) && Project(far_br, p_far_br);

        if (!visible) continue;  // Skip if frustum is behind camera or off-screen

        // Draw frustum lines from camera to far plane
        draw_list->AddLine(p_cam, p_far_tl, frustum_color, line_thickness);
        draw_list->AddLine(p_cam, p_far_tr, frustum_color, line_thickness);
        draw_list->AddLine(p_cam, p_far_bl, frustum_color, line_thickness);
        draw_list->AddLine(p_cam, p_far_br, frustum_color, line_thickness);

        // Draw near plane rectangle
        draw_list->AddLine(p_near_tl, p_near_tr, frustum_color, line_thickness);
        draw_list->AddLine(p_near_tr, p_near_br, frustum_color, line_thickness);
        draw_list->AddLine(p_near_br, p_near_bl, frustum_color, line_thickness);
        draw_list->AddLine(p_near_bl, p_near_tl, frustum_color, line_thickness);

        // Draw far plane rectangle (thicker to show viewing direction endpoint)
        draw_list->AddLine(p_far_tl, p_far_tr, frustum_color, line_thickness * 1.2f);
        draw_list->AddLine(p_far_tr, p_far_br, frustum_color, line_thickness * 1.2f);
        draw_list->AddLine(p_far_br, p_far_bl, frustum_color, line_thickness * 1.2f);
        draw_list->AddLine(p_far_bl, p_far_tl, frustum_color, line_thickness * 1.2f);

        // Draw camera body at position (small filled circle)
        float body_size = 6.0f;
        draw_list->AddCircleFilled(p_cam, body_size, body_color);
        draw_list->AddCircle(p_cam, body_size, IM_COL32(255, 255, 255, 255), 0, 1.5f);

        // Camera label
        std::string label = cam->nodeName.empty() ? "Cam " + std::to_string(i) : cam->nodeName;
        ImVec2 text_pos(p_cam.x + body_size + 5, p_cam.y - 8);
        draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 220), label.c_str());
    }
}

void SceneUI::drawForceFieldGizmos(UIContext& ctx, bool& gizmo_hit) {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;
    SceneSelection& sel = ctx.selection;

    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    float aspect = (image_height > 0)
        ? (static_cast<float>(image_width) / static_cast<float>(image_height))
        : (io.DisplaySize.x / io.DisplaySize.y);
    const bool gizmoOrtho = cam.orthographic && viewport_settings.shading_mode != 2;

    auto Project = [&](const Vec3& p) -> ImVec2 {
        ImVec2 out;
        if (!projectGizmoWorldPoint(cam, gizmoOrtho, aspect, io.DisplaySize.x, io.DisplaySize.y, p, out))
            return ImVec2(-10000, -10000);
        return out;
    };

    for (const auto& ff : ctx.scene.force_field_manager.force_fields) {
        if (!ff->enabled || !ff->visible) continue;

        ImVec2 screen_pos = Project(ff->position);
        if (screen_pos.x < -5000) continue;

        bool is_selected = (sel.selected.type == SelectableType::ForceField && sel.selected.force_field == ff);
        ImU32 color = is_selected ? IM_COL32(255, 100, 255, 255) : IM_COL32(200, 100, 200, 180);

        // Picking check for Icon
        float mouse_dist = sqrtf(powf(io.MousePos.x - screen_pos.x, 2) + powf(io.MousePos.y - screen_pos.y, 2));
        // Same lock as drawLightGizmos: don't let icon picks bypass sculpt/edit-mesh selection lock.
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
        if (mouse_dist < 15.0f && ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver() &&
            !edit_mode_locked && !sculpt_mode_locked) {
            sel.selectForceField(ff, -1, ff->name);
            ForceFieldUI::selected_force_field = ff;
            gizmo_hit = true;
        }

        // Draw Icon
        float size = 12.0f;
        draw_list->AddCircleFilled(screen_pos, size, IM_COL32(30, 30, 30, 200));
        draw_list->AddCircle(screen_pos, size, color, 0, 2.0f);
        
        // Symbol based on type
        const char* symbol = "?";
        bool has_direction = false;
        switch (ff->type) {
            case Physics::ForceFieldType::Wind:      symbol = "W"; has_direction = true; break;
            case Physics::ForceFieldType::Vortex:    symbol = "@"; break;
            case Physics::ForceFieldType::Gravity:   symbol = "G"; has_direction = true; break;
            case Physics::ForceFieldType::Magnetic:  symbol = "M"; has_direction = true; break;
            case Physics::ForceFieldType::Turbulence: symbol = "~"; break;
            case Physics::ForceFieldType::Attractor:  symbol = "+"; break;
            case Physics::ForceFieldType::Repeller:   symbol = "-"; break;
            default: break;
        }

        ImVec2 text_size = ImGui::GetFont()->CalcTextSizeA(ImGui::GetFontSize(), 100.0f, 0.0f, symbol);
        draw_list->AddText(ImVec2(screen_pos.x - text_size.x * 0.5f, screen_pos.y - text_size.y * 0.5f), color, symbol);

        // Draw Direction Arrow
        if (has_direction) {
            Matrix4x4 rot = Matrix4x4::rotationX(ff->rotation.x * 0.0174533f) * 
                            Matrix4x4::rotationY(ff->rotation.y * 0.0174533f) * 
                            Matrix4x4::rotationZ(ff->rotation.z * 0.0174533f);
            
            Vec3 local_dir = ff->direction;
            if (local_dir.length() < 0.001f) {
                // If it's a directional force but direction is 0,0,0, assume -Y (down) default
                local_dir = Vec3(0, -1, 0);
            }
            
            Vec3 world_dir = rot.transform_vector(local_dir).normalize();
            
            // Scale arrow length based on strength to visually indicate force magnitude
            float arrow_len = 1.5f + std::abs(ff->strength) * 0.1f;
            if (arrow_len > 15.0f) arrow_len = 15.0f; // Cap max length
            
            Vec3 arrow_end = ff->position + world_dir * arrow_len;
            ImVec2 screen_end = Project(arrow_end);
            if (screen_end.x > -5000) {
                draw_list->AddLine(screen_pos, screen_end, color, 2.0f);
                // Simple arrow head
                Vec3 right = world_dir.cross(cam_up).normalize() * 0.2f;
                ImVec2 h1 = Project(arrow_end - world_dir * 0.3f + right);
                ImVec2 h2 = Project(arrow_end - world_dir * 0.3f - right);
                if (h1.x > -5000 && h2.x > -5000) {
                    draw_list->AddLine(screen_end, h1, color, 2.0f);
                    draw_list->AddLine(screen_end, h2, color, 2.0f);
                }
            }
        }

        // Show name and strength for ALL force fields in the viewport
        char label[256];
        snprintf(label, sizeof(label), "%s (Str: %.1f)", ff->name.c_str(), ff->strength);
        draw_list->AddText(ImVec2(screen_pos.x + size + 5, screen_pos.y - 8), color, label);

        if (is_selected) {
            // Draw volumetric boundaries based on shape
            if (ff->shape != Physics::ForceFieldShape::Infinite) {
                // Construct full TRS matrix for the force field
                Matrix4x4 local_to_world = Matrix4x4::translation(ff->position) * 
                                           Matrix4x4::rotationX(ff->rotation.x * 0.0174533f) * 
                                           Matrix4x4::rotationY(ff->rotation.y * 0.0174533f) * 
                                           Matrix4x4::rotationZ(ff->rotation.z * 0.0174533f) * 
                                           Matrix4x4::scaling(ff->scale);

                auto TransformAndProject = [&](const Vec3& local_pos) -> ImVec2 {
                    Vec3 world_pos = local_to_world.transform_point(local_pos);
                    return Project(world_pos);
                };

                const ImU32 boundary_color = IM_COL32(255, 0, 255, 120);
                const float thickness = 1.5f;

                if (ff->shape == Physics::ForceFieldShape::Sphere) {
                    float r = ff->falloff_radius;
                    for (int plane = 0; plane < 3; ++plane) {
                        const int segs = 32;
                        ImVec2 last_p;
                        for (int i = 0; i <= segs; ++i) {
                            float a = i * (6.2831853f / segs);
                            Vec3 local_p;
                            if (plane == 0)      local_p = Vec3(cosf(a)*r, sinf(a)*r, 0.0f);
                            else if (plane == 1) local_p = Vec3(cosf(a)*r, 0.0f, sinf(a)*r);
                            else                 local_p = Vec3(0.0f, cosf(a)*r, sinf(a)*r);
                            
                            ImVec2 p_screen = TransformAndProject(local_p);
                            if (i > 0 && p_screen.x > -5000 && last_p.x > -5000) {
                                draw_list->AddLine(last_p, p_screen, boundary_color, thickness);
                            }
                            last_p = p_screen;
                        }
                    }
                }
                else if (ff->shape == Physics::ForceFieldShape::Box) {
                    float r = ff->falloff_radius;
                    Vec3 local_corners[8] = {
                        Vec3(-r, -r, -r), Vec3( r, -r, -r),
                        Vec3( r,  r, -r), Vec3(-r,  r, -r),
                        Vec3(-r, -r,  r), Vec3( r, -r,  r),
                        Vec3( r,  r,  r), Vec3(-r,  r,  r)
                    };
                    const int edges[12][2] = {
                        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // back face
                        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // front face
                        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // connections
                    };
                    for (const auto& edge : edges) {
                        ImVec2 a = TransformAndProject(local_corners[edge[0]]);
                        ImVec2 b = TransformAndProject(local_corners[edge[1]]);
                        if (a.x > -5000 && b.x > -5000) {
                            draw_list->AddLine(a, b, boundary_color, thickness);
                        }
                    }
                }
                else if (ff->shape == Physics::ForceFieldShape::Cylinder) {
                    float r = ff->falloff_radius;
                    const int segs = 32;
                    ImVec2 last_p_top, last_p_bottom;
                    for (int i = 0; i <= segs; ++i) {
                        float a = i * (6.2831853f / segs);
                        Vec3 local_top(cosf(a)*r, r, sinf(a)*r);
                        Vec3 local_bottom(cosf(a)*r, -r, sinf(a)*r);
                        
                        ImVec2 p_top = TransformAndProject(local_top);
                        ImVec2 p_bottom = TransformAndProject(local_bottom);
                        
                        if (i > 0) {
                            if (p_top.x > -5000 && last_p_top.x > -5000) {
                                draw_list->AddLine(last_p_top, p_top, boundary_color, thickness);
                            }
                            if (p_bottom.x > -5000 && last_p_bottom.x > -5000) {
                                draw_list->AddLine(last_p_bottom, p_bottom, boundary_color, thickness);
                            }
                        }
                        // Draw 4 connecting lines
                        if (i % (segs / 4) == 0) {
                            if (p_top.x > -5000 && p_bottom.x > -5000) {
                                draw_list->AddLine(p_top, p_bottom, boundary_color, thickness);
                            }
                        }
                        last_p_top = p_top;
                        last_p_bottom = p_bottom;
                    }
                }
                else if (ff->shape == Physics::ForceFieldShape::Cone) {
                    float r = ff->falloff_radius;
                    const int segs = 32;
                    ImVec2 last_p_base;
                    ImVec2 tip = TransformAndProject(Vec3(0.0f, 0.0f, 0.0f));
                    
                    for (int i = 0; i <= segs; ++i) {
                        float a = i * (6.2831853f / segs);
                        Vec3 local_base(cosf(a)*r, r, sinf(a)*r);
                        ImVec2 p_base = TransformAndProject(local_base);
                        
                        if (i > 0 && p_base.x > -5000 && last_p_base.x > -5000) {
                            draw_list->AddLine(last_p_base, p_base, boundary_color, thickness);
                        }
                        
                        // Draw 4 connecting lines from tip to base
                        if (i % (segs / 4) == 0) {
                            if (tip.x > -5000 && p_base.x > -5000) {
                                draw_list->AddLine(tip, p_base, boundary_color, thickness);
                            }
                        }
                        last_p_base = p_base;
                    }
                }
            }
        }
    }

    // Cloth/soft pin regions of the SELECTED body: small cyan wire spheres so the
    // user sees what is held fixed (authored from Edit-Mesh vertex selection).
    if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        const std::string& sel_node = sel.selected.object->getNodeName();
        for (const auto& rb : ctx.scene.rigid_bodies) {
            if (rb.kind == RayTrophiSim::BodyKind::Rigid) continue;
            if (rb.source_name != sel_node || rb.getSoftPins().empty()) continue;
            for (const auto& pin : rb.getSoftPins()) {
                if (!pin.enabled) continue;
                const float r = pin.radius;
                const ImU32 pin_col = IM_COL32(80, 230, 255, 220);
                ImVec2 c = Project(pin.center);
                if (c.x > -5000) draw_list->AddCircleFilled(c, 3.0f, pin_col);
                for (int plane = 0; plane < 3; ++plane) {
                    const int segs = 20;
                    ImVec2 last_p;
                    for (int i = 0; i <= segs; ++i) {
                        float a = i * (6.28318f / segs);
                        Vec3 p3d;
                        if (plane == 0)      p3d = pin.center + Vec3(cosf(a)*r, sinf(a)*r, 0);
                        else if (plane == 1) p3d = pin.center + Vec3(cosf(a)*r, 0, sinf(a)*r);
                        else                 p3d = pin.center + Vec3(0, cosf(a)*r, sinf(a)*r);
                        ImVec2 p_screen = Project(p3d);
                        if (i > 0 && p_screen.x > -5000 && last_p.x > -5000)
                            draw_list->AddLine(last_p, p_screen, pin_col, 1.0f);
                        last_p = p_screen;
                    }
                }
            }
        }
    }
}
void SceneUI::drawSelectionGizmos(UIContext& ctx)
{
    gpu_edit_overlay_sync.drawn_this_frame = false;
    // Sculpt protection mask tint: backend-independent ImGui overlay so it shows
    // in both Solid and Rendered viewport modes (the GPU edit overlay is Edit +
    // Solid only). Self-guards on sculpt mode / mask presence.
    drawSculptMaskViewportOverlay(ctx);
    // While a sculpt session owns the selected object, suppress the selection bbox /
    // outline / transform gizmo: the brush draws its own cursor + mask overlay, and the
    // selection outline kept re-tracing the brush-deformed dab edges (distracting + it
    // steals clicks). The GPU outline is cleared below so nothing lingers.
    const bool sculpt_session_active =
        sculpt_mode_state.enabled &&
        mesh_workspace_mode == MeshWorkspaceMode::Sculpt &&
        mesh_overlay_settings.edit_mode &&
        !sculpt_mode_state.active_target_name.empty();
    if (!sculpt_session_active &&
        ctx.selection.hasSelection() && ctx.selection.show_gizmo && ctx.scene.camera && viewport_settings.show_gizmos) {
        drawSelectionBoundingBox(ctx);
        if (mesh_overlay_settings.enabled && mesh_workspace_mode == MeshWorkspaceMode::Edit) {
            drawEditableMeshOverlay(ctx);
        }
        drawTransformGizmo(ctx);
    } else if (auto* outline_vpb =
                   dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get())) {
        // Selection/gizmos gone this frame — drop the GPU outline too
        // (drawSelectionBoundingBox, which normally manages it, didn't run).
        outline_vpb->clearSelectionOutline();
    }
    // GPU edit overlay didn't sync this frame (deselected, edit mode exited,
    // overlay disabled, sculpt stroke) -> clear it from the viewport backend.
    if (gpu_edit_overlay_sync.active && !gpu_edit_overlay_sync.drawn_this_frame) {
        releaseGpuEditMeshOverlay();
    }
}

// Overlay grid removed — use depth-tested raster grid in Vulkan backends instead.
