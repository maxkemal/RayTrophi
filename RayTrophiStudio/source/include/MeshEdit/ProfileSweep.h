#pragma once

#include "BezierSpline.h"
#include "MeshEdit/MeshTool.h"
#include "DNA/GeometryDetail.h"
#include <memory>
#include <string>

namespace MeshEdit {

// First production slice of the profile/spline pipeline. The profile is sampled in
// its local XY plane; the path is sampled in 3D and receives a transported frame.
struct ProfileSweepSettings {
    int path_samples = 32;
    int profile_samples = 16;
    bool cap_start = true;
    bool cap_end = true;
    float profile_scale = 1.0f;
    Vec3 up = Vec3(0.0f, 1.0f, 0.0f);
};

struct ProfileSweepResult {
    std::shared_ptr<DNA::GeometryDetail> geometry;
    MeshOperationReport report;
    uint32_t path_ring_count = 0;
    uint32_t profile_ring_count = 0;
};

// Builds canonical flat SoA data directly. No Triangle facade is created or consulted.
ProfileSweepResult buildProfileSweep(const BezierSpline& profile,
                                     const BezierSpline& path,
                                     const ProfileSweepSettings& settings = {});

// Deterministic geometry-only smoke test used by the future Python/IPC service tests.
bool runProfileSweepSelfTest(std::string* details = nullptr);

} // namespace MeshEdit
