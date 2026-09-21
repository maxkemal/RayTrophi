#pragma once
#include "../shaders/surface_coverage.h"

// Authored independently from optical transmission. Default preserves old files.
struct MaterialCoverage {
    bool alphaCutout = false;
    float opacity(float value) const {
        return SurfaceCoverage::materialCoverageOpacity(value, alphaCutout);
    }
    static bool validCutoutValue(float value) { return value == 0.0f || value == 1.0f; }
    void setCutout(float value) { alphaCutout = value == 1.0f; }
};
