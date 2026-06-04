#include "SurfaceFlowField.h"

#include "Triangle.h"

#include <cstddef>
#include <cmath>
#include <tuple>

namespace RayTrophiSim {

bool SurfaceFlowField::computeTrianglePixelFlow(const Triangle& tri,
                                                int uv_set,
                                                int width,
                                                int height,
                                                SurfaceFlowSample& out_sample,
                                                const Vec3& driving_acceleration) {
    out_sample = SurfaceFlowSample{};
    if (width <= 1 || height <= 1) {
        return false;
    }

    Vec2 uv0, uv1, uv2;
    if (uv_set > 0 && static_cast<size_t>(uv_set) < tri.getUVSetCount()) {
        std::tie(uv0, uv1, uv2) = tri.getUVSetCoordinates(static_cast<size_t>(uv_set));
    } else {
        std::tie(uv0, uv1, uv2) = tri.getUVCoordinates();
    }

    const Vec3 p0 = tri.getVertexPosition(0);
    const Vec3 e1 = tri.getVertexPosition(1) - p0;
    const Vec3 e2 = tri.getVertexPosition(2) - p0;
    const Vec3 raw_normal = e1.cross(e2);
    const float raw_len_sq = raw_normal.length_squared();
    if (raw_len_sq <= 1e-20f) {
        return false;
    }

    const float raw_len = std::sqrt(raw_len_sq);
    const Vec3 geom_normal(raw_normal.x / raw_len, raw_normal.y / raw_len, raw_normal.z / raw_len);

    const float du1 = uv1.u - uv0.u;
    const float dv1 = uv1.v - uv0.v;
    const float du2 = uv2.u - uv0.u;
    const float dv2 = uv2.v - uv0.v;
    const float det = du1 * dv2 - du2 * dv1;
    if (std::abs(det) < 1e-10f) {
        return false;
    }

    const float inv = 1.0f / det;
    const Vec3 dPdU = (e1 * dv2 - e2 * dv1) * inv;
    const Vec3 dPdV = (e2 * du1 - e1 * du2) * inv;

    const Vec3 tangent_drive = driving_acceleration - geom_normal * driving_acceleration.dot(geom_normal);
    const float tangent_len = tangent_drive.length();
    if (tangent_len <= 1e-5f) {
        return false;
    }

    const Vec3 flow_dir = tangent_drive / tangent_len;
    out_sample.slope = tangent_len;

    const float a = dPdU.dot(dPdU);
    const float b = dPdU.dot(dPdV);
    const float c = dPdV.dot(dPdV);
    const float rhs_u = dPdU.dot(flow_dir);
    const float rhs_v = dPdV.dot(flow_dir);
    const float metric_det = a * c - b * b;
    if (std::abs(metric_det) < 1e-10f) {
        return false;
    }

    float flow_u = (c * rhs_u - b * rhs_v) / metric_det;
    float flow_v = (a * rhs_v - b * rhs_u) / metric_det;
    const float flow_uv_len = std::sqrt(flow_u * flow_u + flow_v * flow_v);
    if (flow_uv_len <= 1e-8f) {
        return false;
    }
    flow_u /= flow_uv_len;
    flow_v /= flow_uv_len;

    const float pixel_flow_x = flow_u * static_cast<float>(width - 1);
    const float pixel_flow_y = -flow_v * static_cast<float>(height - 1);
    const float pixel_flow_len = std::sqrt(pixel_flow_x * pixel_flow_x + pixel_flow_y * pixel_flow_y);
    if (pixel_flow_len <= 1e-6f) {
        return false;
    }

    out_sample.flow_x = pixel_flow_x;
    out_sample.flow_y = pixel_flow_y;
    out_sample.flow_u = flow_u;
    out_sample.flow_v = flow_v;
    out_sample.flow_length = pixel_flow_len;

    return std::isfinite(out_sample.flow_x) &&
           std::isfinite(out_sample.flow_y) &&
           std::isfinite(out_sample.slope);
}

} // namespace RayTrophiSim
