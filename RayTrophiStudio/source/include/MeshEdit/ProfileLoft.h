#pragma once

#include "BezierSpline.h"
#include "MeshEdit/MeshTool.h"
#include "DNA/GeometryDetail.h"
#include <memory>
#include <string>
#include <vector>

namespace MeshEdit {

struct ProfileLoftSettings {
    int samples_per_section = 24;
    bool cap_start = true;
    bool cap_end = true;
};

struct ProfileLoftResult {
    std::shared_ptr<DNA::GeometryDetail> geometry;
    MeshOperationReport report;
    uint32_t section_count = 0;
    uint32_t ring_size = 0;
};

// Connects multiple closed spline profiles into one canonical flat SoA mesh.
// Section order is preserved; the source splines remain untouched.
ProfileLoftResult buildProfileLoft(const std::vector<const BezierSpline*>& sections,
                                   const ProfileLoftSettings& settings = {});

bool runProfileLoftSelfTest(std::string* details = nullptr);

} // namespace MeshEdit
