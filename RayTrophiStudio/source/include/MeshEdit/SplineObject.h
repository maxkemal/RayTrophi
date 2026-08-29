#pragma once

#include "BezierSpline.h"
#include "Hittable.h"
#include "Transform.h"
#include <memory>
#include <cstddef>
#include <string>
#include <vector>

namespace DNA { class GeometryDetail; }

namespace MeshEdit {

// Engine convention: +Y is up. XY is the front profile plane (X=lateral,
// Y=height), XZ is top, and YZ is the right/left profile plane. Free is
// unconstrained 3D authoring for consumers that are not planar profiles
// (road/path guides, force-field direction curves, hair guides, surface
// boundary curves) - point edits are not flattened to any single axis.
enum class SplinePlane : uint8_t { XY, XZ, YZ, Free };

enum class SplineEditTool : uint8_t {
    Select,
    InsertPoint,
    Subdivide,
    Extrude
};

// Persistent, non-destructive viewport/render display for an authoring spline.
// The linked host owns evaluated flat geometry while this object remains the
// editable source. Converting detaches the host as an ordinary TriangleMesh.
struct SplineSkinDisplaySettings {
    bool enabled = false;
    std::string host_name;
    std::string custom_profile;
    float radius = 0.1f;
    int path_samples = 48;
    int radial_segments = 12;
    bool cap_start = true;
    bool cap_end = true;
    bool use_point_radius = true;
    float taper_start = 1.0f;
    float taper_end = 1.0f;
    float taper_falloff = 1.0f;
    float twist_start_degrees = 0.0f;
    float twist_end_degrees = 0.0f;
    float wave_amplitude = 0.0f;
    float wave_cycles = 1.0f;
    float wave_phase_degrees = 0.0f;
    float wave_noise = 0.0f;
    int wave_seed = 0;
    int wave_axis = 1;
};

// Non-mesh scene authoring object. Its spline is the source for profile/curve
// modifiers and Geometry Nodes; no generated triangles are stored here.
class SplineObject final : public Hittable {
public:
    std::string nodeName;
    BezierSpline spline;
    SplinePlane plane = SplinePlane::XY;
    std::shared_ptr<Transform> transform = std::make_shared<Transform>();
    int selected_point = -1;
    std::vector<int> selected_points;
    bool edit_controls = true;
    bool edit_mode = false;
    bool point_drag_dirty = false;
    SplineEditTool edit_tool = SplineEditTool::Select;
    int subdivide_cuts = 1;
    Vec3 insert_preview_position;
    bool has_insert_preview = false;
    // Right-drag box selection state. Kept on the authoring object so the
    // overlay remains deterministic when several spline objects are present.
    bool selection_box_active = false;
    Vec2 selection_box_start{};
    Vec2 selection_box_current{};

    SplineSkinDisplaySettings skin_display;
    std::string skin_display_status;

    // Transient non-destructive profile-operation preview. It is intentionally
    // excluded from project serialization; Apply publishes a separate mesh.
    std::shared_ptr<DNA::GeometryDetail> profile_preview_geometry;
    Matrix4x4 profile_preview_transform = Matrix4x4::identity();
    std::string profile_preview_operation;
    std::string profile_preview_counterpart;
    std::string profile_preview_status;
    size_t profile_preview_signature = 0;
    std::string animation_status;

    bool hit(const Ray&, float, float, HitRecord&, bool = false) const override { return false; }
    bool bounding_box(float, float, AABB& output_box) const override;
    bool isSplineObject() const { return true; }
};

} // namespace MeshEdit
