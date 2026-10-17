#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// Gizmo occlusion: scene gizmos that sit correctly in perspective.
// ═══════════════════════════════════════════════════════════════════════════
// Scene gizmos (simulation domains, colliders, seed boxes, ...) are ImGui
// draw-list lines: projected to 2D, they carry no depth and draw in front of
// everything, so a domain box behind a wall reads as floating in front of it
// and placing anything needs the top/front views.
//
// drawSegment splits a world segment into short screen pieces and casts one
// ray from the eye to each piece through the scene's CPU BVH (the same one
// click-selection uses, so it works on every backend). A piece something
// solid stands in front of is drawn faint (hidden_alpha) - Blender's hidden
// line look - so the gizmo still shows where it goes. Volumes are ignored:
// fog and liquid do not hide a gizmo.
//
// Results are cached per segment while the camera, the scene geometry
// generation and the timeline frame stay the same. A per-frame ray budget
// keeps a gizmo-heavy view interactive: past it, segments draw unoccluded
// (the old look) and stats().budget_exhausted says so.
//
// Transform handles (ImGuizmo) deliberately stay on top: a handle hidden
// behind its own object could not be grabbed.

#include "imgui.h"
#include "Vec3.h"

#include <cstdint>
#include <functional>
#include <string>

struct SceneData;

namespace GizmoOcclusion {

using ProjectFn = std::function<bool(const Vec3& world, ImVec2& screen)>;

struct Stats {
    bool active = false;
    std::string inactive_reason;
    uint64_t segments = 0;          // drawn through drawSegment, last frame
    uint64_t rays = 0;              // BVH rays cast, last frame (cache misses)
    uint64_t hidden_pieces = 0;     // pieces drawn faint, last frame
    uint64_t cache_entries = 0;
    bool budget_exhausted = false;  // last frame ran out of rays
};

// Once per UI frame, before any drawSegment. Returns whether occlusion is on
// for this frame; when off, drawSegment draws plain lines.
bool beginFrame(const SceneData& scene, bool enabled, float hidden_alpha, bool orthographic,
                int timeline_frame, bool scene_busy = false);

void drawSegment(ImDrawList* draw_list, const Vec3& a, const Vec3& b, ImU32 color,
                 float thickness, const ProjectFn& project);

// Point markers/labels share the segment cache, ray budget and hidden opacity.
float pointAlpha(const Vec3& point);

const Stats& stats();

} // namespace GizmoOcclusion
