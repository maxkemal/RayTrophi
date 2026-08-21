#pragma once

#include "BezierSpline.h"
#include "Hittable.h"
#include "Transform.h"
#include <memory>
#include <string>
#include <vector>

namespace MeshEdit {

// Engine convention: +Y is up. XY is therefore the default vertical profile
// plane (X=lateral, Z=depth); XY and YZ are explicit alternate planes.
enum class SplinePlane : uint8_t { XY, XZ, YZ };

enum class SplineEditTool : uint8_t {
    Select,
    InsertPoint,
    Subdivide,
    Extrude
};

// Non-mesh scene authoring object. Its spline is the source for profile/curve
// modifiers and Geometry Nodes; no generated triangles are stored here.
class SplineObject final : public Hittable {
public:
    std::string nodeName;
    BezierSpline spline;
    SplinePlane plane = SplinePlane::XZ;
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

    bool hit(const Ray&, float, float, HitRecord&, bool = false) const override { return false; }
    bool bounding_box(float, float, AABB& output_box) const override;
    bool isSplineObject() const { return true; }
};

} // namespace MeshEdit
