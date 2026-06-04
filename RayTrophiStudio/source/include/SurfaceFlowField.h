#pragma once

#include "Vec3.h"

class Triangle;

namespace RayTrophiSim {

struct SurfaceFlowSample {
    float flow_x = 0.0f;
    float flow_y = 0.0f;
    float flow_u = 0.0f;
    float flow_v = 0.0f;
    float flow_length = 0.0f;
    float slope = 0.0f;

    bool valid() const { return flow_length > 0.0f && slope > 0.0f; }
};

class SurfaceFlowField {
public:
    static bool computeTrianglePixelFlow(const Triangle& tri,
                                         int uv_set,
                                         int width,
                                         int height,
                                         SurfaceFlowSample& out_sample,
                                         const Vec3& driving_acceleration = Vec3(0.0f, -1.0f, 0.0f));
};

} // namespace RayTrophiSim
