#pragma once

// ═══════════════════════════════════════════════════════════════════════════════
// SURFACE AUTHORING - one raycast policy for "place a point where the cursor aims"
// ═══════════════════════════════════════════════════════════════════════════════
// This used to exist twice: an anonymous-namespace helper inside
// ProfileSplineOverlay.cpp (mesh BVH + terrain + ground plane, unreachable from
// anywhere else) and an inline copy in scene_ui_river.hpp that only ever saw the
// terrain. Two placement policies meant a river point and a spline point could
// land in different places for the same click, and neither could be driven from
// script.
//
// The result is a VALUE, not just a position: a caller that only learns "where"
// has to guess "what", and a guessed answer is the failure this repository keeps
// paying for. `kind` and `object_name` say what was actually under the cursor.
//
// The core takes a SceneData and a Ray, so the same policy serves the viewport
// tools and the scene.raycast IPC method without an ImGui dependency.

#include "Vec3.h"
#include "Ray.h"

#include <string>

struct SceneData;
struct UIContext;
struct ImVec2;

namespace MeshEdit {

// What a caller is willing to snap to. This is deliberately explicit: River's
// old copy could not see meshes, so unifying the code without a filter would
// silently START snapping river points onto rocks and bridges. That reads as a
// plausible result rather than a bug, so the choice has to be stated.
enum class SurfaceFilter : uint8_t {
    MeshAndTerrain,  // Default: nearest of scene geometry and terrain.
    TerrainOnly,     // Terrain (then ground plane). River keeps its semantics.
    GroundPlaneOnly  // Y=0 only; ignores everything in the scene.
};

enum class SurfaceHitKind : uint8_t {
    None,
    Mesh,
    Terrain,
    GroundPlane
};

struct SurfaceSnapResult {
    Vec3 position;
    Vec3 normal = Vec3(0.0f, 1.0f, 0.0f);
    float distance = 0.0f;
    SurfaceHitKind kind = SurfaceHitKind::None;
    // Empty for terrain and the ground plane, and for meshes that carry no name.
    std::string object_name;

    bool hit() const { return kind != SurfaceHitKind::None; }
};

// Core policy. Returns false only in the genuinely degenerate case: nothing was
// hit and the ray is near-parallel to the ground plane (looking at the horizon
// with nothing behind it).
bool raycastSurface(SceneData& scene,
                    const Ray& ray,
                    SurfaceFilter filter,
                    SurfaceSnapResult& out);

// Viewport wrapper: builds the ray from a mouse position through the scene
// camera, using the same full-window normalization the selection tool uses.
bool snapToSurface(UIContext& ctx,
                   const ImVec2& mousePos,
                   SurfaceFilter filter,
                   SurfaceSnapResult& out);

} // namespace MeshEdit
