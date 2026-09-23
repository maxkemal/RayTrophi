/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiCameraNav.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * Viewport camera NAVIGATION: orbit, pan, dolly, Frame Selected and the pivot
 * lock they all turn around.
 *
 * ★★★ Why this file exists at all (2026-09-23). Every one of these gestures was
 * mouse- and numpad-only. `camera.*` over IPC could set a position, a target,
 * an fov and a focus distance -- it could not orbit, it could not frame the
 * selection, and above all it could not ANSWER WHERE THE PIVOT WAS. CLAUDE.md
 * rule 1 calls that untestable, and it was: the bug that prompted this file
 * (the pivot silently collapsing onto the lens focal plane) is invisible to a
 * screenshot and has no log line. It is only visible as a NUMBER, and until
 * `camera.get_pivot` existed there was no number to read.
 *
 * ★★★★★ The one contract in here worth reading before changing anything:
 *
 *   IN SELECTION MODE THE PIVOT IS A DERIVED VALUE, NOT STORED STATE.
 *
 * It is recomputed from the current selection at the start of every navigation
 * gesture (refreshPivotFromSelection, called by the mouse/numpad handlers in
 * Main.cpp and by every entry point here). That is deliberate and it is the
 * whole point: a pivot that is stored goes stale, and a stale pivot is exactly
 * the failure this work removed -- Frame Selected wrote the selection centre,
 * then the next middle-click raycast overwrote it with whatever surface the
 * cursor happened to be over, with no symptom beyond "the camera feels wrong".
 * A derived pivot cannot go stale. It also follows an animated object for free.
 *
 * ★★ And the pivot is NOT `focus_dist`. See the contract block in Camera.h;
 * nothing in this file writes the focal plane. Autofocus owns it alone.
 */

#include "Api/RtApiInternal.h"
#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "Camera.h"
#include "SceneSelection.h"
#include "TriangleMesh.h"   // flat SoA bounds: SceneSelection only forward-declares it
#include "Triangle.h"       // facade fallback for the legacy selection path
#include "globals.h"

#include <algorithm>
#include <cmath>

// The interactive raster viewport, owned by the UI layer (same file-local
// extern the other RtApi translation units use).
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

namespace rtapi {

namespace {

Camera* navCamera() {
    return (g_ctx && g_ctx->scene.camera) ? g_ctx->scene.camera.get() : nullptr;
}

// ★★★ Post-navigation sync. This is NOT the same body as RtApi.cpp's
//   cameraChanged(): that one arms `g_camera_dirty` and leaves the raster
//   viewport to notice on its own, which races (see the comment on
//   refreshNumpadView in Main.cpp). A navigation call whose effect only lands
//   one or two frames later is unusable for exactly the caller this surface
//   exists for -- an agent that orbits, screenshots, and compares.
void navChanged(Camera& cam) {
    cam.update_camera_vectors();
    cam.markDirty();
    g_camera_dirty = true;
    if (g_viewport_backend) {
        g_viewport_backend->syncCamera(cam);
        g_viewport_backend->resetAccumulation();
    }
    if (g_ctx) {
        if (g_ctx->backend_ptr) {
            g_ctx->backend_ptr->syncCamera(cam);
            g_ctx->backend_ptr->resetAccumulation();
        }
        g_ctx->renderer.resetCPUAccumulation();
        g_ctx->start_render = true;
    }
}

// World bounds of the current selection. Returns false when nothing is selected
// or the selection has no extent we can measure.
//
// ★ A selection with no bounds is not an error and must not be reported as a
//   zero-size box at the origin: the caller falls back to the item's cached
//   position with a unit radius, which frames SOMETHING the user can see. A
//   zero radius would divide the framing distance down to the near plane.
bool selectionExtent(Vec3& center, float& radius) {
    if (!g_ctx) return false;
    const SelectableItem& item = g_ctx->selection.selected;
    if (!item.is_valid()) return false;

    AABB bounds;
    bool hasBounds = false;
    if (item.type == SelectableType::Object) {
        // Flat SoA first: `mesh_object` is the canonical identity and
        // `object` is only a representative facade triangle. Asking the facade
        // for bounds frames ONE TRIANGLE of a million-triangle mesh.
        if (item.mesh_object) {
            hasBounds = item.mesh_object->bounding_box(0.0f, 0.0f, bounds);
        } else if (item.object) {
            hasBounds = item.object->bounding_box(0.0f, 0.0f, bounds);
        }
    } else if (item.type == SelectableType::VDBVolume && item.vdb_volume) {
        bounds = item.vdb_volume->getWorldBounds();
        hasBounds = true;
    } else if (item.type == SelectableType::GasVolume && item.gas_volume) {
        Vec3 bmin, bmax;
        item.gas_volume->getWorldBounds(bmin, bmax);
        bounds = AABB(bmin, bmax);
        hasBounds = true;
    }

    center = item.position;
    radius = 1.0f;
    if (hasBounds) {
        center = (bounds.min + bounds.max) * 0.5f;
        radius = std::max(0.001f, (bounds.max - bounds.min).length() * 0.5f);
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Pivot
// ---------------------------------------------------------------------------

Result listCameras(CameraListState& out) {
    if (!g_ctx) return notBound();
    const SceneData& scene = g_ctx->scene;

    out.count = static_cast<int>(scene.cameras.size());
    out.active_index = static_cast<int>(scene.active_camera_index);
    out.has_active_camera = (scene.camera != nullptr);
    out.active_index_out_of_range =
        (scene.active_camera_index >= scene.cameras.size());

    bool found_active = false;
    out.cameras.reserve(scene.cameras.size());
    for (size_t i = 0; i < scene.cameras.size(); ++i) {
        const std::shared_ptr<Camera>& cam = scene.cameras[i];
        if (!cam) continue;
        CameraInfo info;
        info.index = static_cast<int>(i);
        info.name = cam->nodeName;
        info.position = cam->lookfrom;
        info.target = cam->lookat;
        info.fov = cam->vfov;
        // ★ POINTER identity, not index and not a value comparison. The whole
        //   question is whether the object the scene is driving is the same
        //   OBJECT the registry holds; two cameras can carry identical numbers
        //   and still be different objects, and that is precisely the failure
        //   this field exists to catch.
        info.active = (cam == scene.camera);
        if (info.active) found_active = true;
        out.cameras.push_back(std::move(info));
    }

    // An active camera that is not in the registry. Reported as its own fact
    // rather than inferred by the caller from count/index, because "the list is
    // empty" and "the list does not contain this one" lead to different fixes.
    out.active_is_orphan = out.has_active_camera && !found_active;
    return Result::success();
}

Result refreshPivotFromSelection() {
    Camera* cam = navCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    if (cam->pivot_mode != Camera::PivotMode::Selection) return Result::success();

    Vec3 center;
    float radius = 1.0f;
    if (!selectionExtent(center, radius)) {
        // Locked to the selection but nothing is selected. Disarm rather than
        // keep the last object's centre: a lock pointing at a deleted or
        // deselected object is the stale pivot this design exists to prevent.
        // ★ The MODE is kept, so re-selecting re-arms without a second toggle.
        cam->clearOrbitPivot();
        return Result::success();
    }
    cam->setOrbitPivot(center);
    return Result::success();
}

Result setCameraPivotMode(const std::string& mode) {
    if (!g_ctx) return notBound();
    Camera* cam = navCamera();
    if (!cam) return Result::fail("no active camera in the scene");

    if (mode == "free") {
        cam->pivot_mode = Camera::PivotMode::Free;
        cam->clearOrbitPivot();
        return Result::success();
    }
    if (mode == "selection") {
        cam->pivot_mode = Camera::PivotMode::Selection;
        return refreshPivotFromSelection();
    }
    return Result::fail("pivot mode must be 'free' or 'selection'");
}

Result getCameraPivot(CameraPivotState& out) {
    if (!g_ctx) return notBound();
    Camera* cam = navCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    refreshPivotFromSelection();

    out.mode = (cam->pivot_mode == Camera::PivotMode::Selection) ? "selection" : "free";
    out.locked = cam->pivot_valid;
    out.pivot = cam->effectivePivot();
    out.nav_distance = cam->navDistance();
    // ★★★ The whole reason this getter exists: `nav_distance` and
    //   `focus_distance` were ONE number until 2026-09-23, and their being one
    //   number is what made panning unusable. Returning both side by side is
    //   the measurement that proves they are apart -- pull the focus ring to
    //   0.5 m and `nav_distance` must NOT follow.
    out.focus_distance = cam->focus_dist;
    out.orthographic = cam->orthographic;
    out.ortho_height = cam->ortho_height;
    if (g_ctx->selection.selected.is_valid())
        out.selection_name = g_ctx->selection.selected.name;
    return Result::success();
}

// ---------------------------------------------------------------------------
// Gestures
// ---------------------------------------------------------------------------

Result orbitCamera(float yaw_degrees, float pitch_degrees) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    Camera* cam = navCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    refreshPivotFromSelection();
    cam->orbitAroundPivot(yaw_degrees, pitch_degrees);
    navChanged(*cam);
    return Result::success();
}

Result dollyCamera(float factor) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!(factor > 0.0f) || !std::isfinite(factor))
        return Result::fail("factor must be a positive multiplier (0.5 = halve the distance)");
    Camera* cam = navCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    refreshPivotFromSelection();
    // The API takes a DISTANCE MULTIPLIER, not the internal exponent: "half as
    // far" is a thing a caller can reason about, an exponent is not.
    cam->dollyToPivot(std::log(factor));
    navChanged(*cam);
    return Result::success();
}

Result panCamera(float right, float up) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    Camera* cam = navCamera();
    if (!cam) return Result::fail("no active camera in the scene");
    refreshPivotFromSelection();
    // World units along the camera's own right/up axes. Screen-correct pixel
    // panning lives in the mouse handler; over IPC a caller wants metres.
    cam->panWorld(cam->u * right + cam->v * up);
    navChanged(*cam);
    return Result::success();
}

Result frameSelected(bool lock_pivot) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    Camera* cam = navCamera();
    if (!cam) return Result::fail("no active camera in the scene");

    Vec3 center;
    float radius = 1.0f;
    if (!selectionExtent(center, radius))
        return Result::fail("nothing is selected");

    Vec3 forward = cam->lookat - cam->lookfrom;
    if (forward.length_squared() < 1e-8f) forward = Vec3(0.0f, 0.0f, -1.0f);
    forward = forward.normalize();

    // Fit the bounding sphere to the LIMITING half-angle. Using the vertical
    // half-fov alone under-frames a wide object on a wide viewport.
    const float halfV = std::max(1.0f, cam->vfov) * 0.5f * 3.14159265f / 180.0f;
    const float halfH = std::atan(std::tan(halfV) * std::max(cam->aspect_ratio, 0.01f));
    const float limitingHalfFov = std::max(0.01f, std::min(halfV, halfH));
    const float framedDistance = std::clamp(
        (radius * 1.25f) / std::sin(limitingHalfFov), 0.01f, 10000000.0f);

    cam->lookat = center;
    cam->lookfrom = center - forward * framedDistance;
    // ★ focus_dist is NOT set here any more. Framing an object is a navigation
    //   act; refocusing the lens on it is a separate decision that belongs to
    //   autofocus or to camera.set_focus_distance. Writing both from one
    //   gesture is how the two got welded together in the first place.
    if (cam->orthographic)
        cam->ortho_height = std::max(0.01f, radius * 2.5f);

    if (lock_pivot) cam->pivot_mode = Camera::PivotMode::Selection;
    cam->setOrbitPivot(center);
    navChanged(*cam);
    return Result::success();
}

} // namespace rtapi
