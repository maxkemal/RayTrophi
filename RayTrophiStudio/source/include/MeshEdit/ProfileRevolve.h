#pragma once

#include "BezierSpline.h"
#include "MeshEdit/MeshTool.h"
#include "DNA/GeometryDetail.h"
#include <memory>
#include <string>

namespace MeshEdit {

enum class ProfileRevolveAxis : uint8_t { X, Y, Z };

struct ProfileRevolveSettings {
    int angle_segments = 32;
    int profile_samples = 24;
    float start_angle = 0.0f;
    float end_angle = 2.0f * M_PI;
    float radius_offset = 0.0f;
    Vec3 axis_pivot = Vec3(0.0f);
    ProfileRevolveAxis axis = ProfileRevolveAxis::Y;
};

struct ProfileRevolveResult {
    std::shared_ptr<DNA::GeometryDetail> geometry;
    MeshOperationReport report;
    uint32_t angle_ring_count = 0;
    uint32_t profile_ring_count = 0;
};

// Revolves an open or closed 2D profile in the (radius,height) plane around the
// selected world axis through axis_pivot. Partial angle ranges keep their
// start/end seams open.
// The spline remains the editable authoring source; this only emits flat preview data.
ProfileRevolveResult buildProfileRevolve(const BezierSpline& profile,
                                         const ProfileRevolveSettings& settings = {});

BezierSpline makeCupProfile();
BezierSpline makeBottleProfile();
bool runProfileRevolveSelfTest(std::string* details = nullptr);

} // namespace MeshEdit
