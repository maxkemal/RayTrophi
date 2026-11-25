#include "Paint/WetPaintGeometryBake.h"

#include "Paint/PaintHeightNormalMath.h"
#include "SurfaceFlowField.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace Paint {

namespace {

// Octahedral normal, packSnorm2x16 into a float's bits (octDecode in the shader).
float packOctNormal(Vec3 n) {
    const float len = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    if (len <= 1e-12f) {
        n = Vec3(0.0f, 1.0f, 0.0f);
    } else {
        n = Vec3(n.x / len, n.y / len, n.z / len);
    }
    float ex = n.x;
    float ey = n.y;
    if (n.z < 0.0f) {
        ex = (1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
        ey = (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
    }
    const auto snorm = [](float v) -> uint32_t {
        const int i = static_cast<int>(std::round(std::clamp(v, -1.0f, 1.0f) * 32767.0f));
        return static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(i)));
    };
    const uint32_t bits = snorm(ex) | (snorm(ey) << 16);
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

} // namespace

bool bakeWetPaintGeometry(const std::vector<WetPaintBakeFace>& faces,
                          uint32_t width,
                          uint32_t height,
                          uint32_t tile_size,
                          int gutter_texels,
                          const std::string& object_name,
                          int uv_set,
                          WetPaintBakeResult& out,
                          std::string& error) {
    const size_t texels = static_cast<size_t>(width) * height;
    out.pos_valid.assign(texels * 4, 0.0f);
    out.flow_normal.assign(texels * 4, 0.0f);
    out.normal_spans.assign(texels, Vec2(1.0f, 1.0f));
    out.valid_texels = 0;
    out.gutter_texels = 0;
    out.triangles = 0;

    const float W1 = static_cast<float>(width - 1);
    const float H1 = static_cast<float>(height - 1);
    const Vec3 gravity(0.0f, -1.0f, 0.0f);

    for (const auto& tri : faces) {
        const auto& uv0 = tri.uvs[0];
        const auto& uv1 = tri.uvs[1];
        const auto& uv2 = tri.uvs[2];
        const float x[3] = {uv0.u * W1, uv1.u * W1, uv2.u * W1};
        const float y[3] = {(1.0f - uv0.v) * H1, (1.0f - uv1.v) * H1, (1.0f - uv2.v) * H1};
        const float area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
        if (std::abs(area) < 1e-12f) {
            continue;
        }
        ++out.triangles;
        const auto& p = tri.positions;
        Vec3 geom_n = (p[1] - p[0]).cross(p[2] - p[0]);
        const float gl = geom_n.length();
        geom_n = gl > 1e-20f ? geom_n / gl : Vec3(0.0f, 1.0f, 0.0f);
        const auto& n = tri.normals;
        const Vec2 normal_span = HeightNormalMath::twoTexelWorldSpan(
            p[0], p[1], p[2], uv0, uv1, uv2, width, height);
        RayTrophiSim::SurfaceFlowSample flow;
        const bool has_flow = RayTrophiSim::SurfaceFlowField::computePixelFlow(
            p[0], p[1], p[2], uv0, uv1, uv2, static_cast<int>(width),
            static_cast<int>(height), flow, gravity) && flow.valid();
        const float fx_flow = has_flow ? flow.flow_x : 0.0f;
        const float fy_flow = has_flow ? flow.flow_y : 0.0f;
        const float f_slope = has_flow ? flow.slope : 0.0f;

        const int x0 = std::max(0, static_cast<int>(std::floor(std::min({x[0], x[1], x[2]}))));
        const int x1 = std::min(static_cast<int>(width) - 1,
                                static_cast<int>(std::ceil(std::max({x[0], x[1], x[2]}))));
        const int y0 = std::max(0, static_cast<int>(std::floor(std::min({y[0], y[1], y[2]}))));
        const int y1 = std::min(static_cast<int>(height) - 1,
                                static_cast<int>(std::ceil(std::max({y[0], y[1], y[2]}))));
        const float inv_area = 1.0f / area;

        const int rows = y1 - y0 + 1;
        const int cols = x1 - x0 + 1;
        const bool parallel_tri = rows >= 32 && static_cast<int64_t>(rows) * cols >= 4096;
        uint32_t tri_valid = 0;

        #pragma omp parallel for schedule(static) reduction(+:tri_valid) if(parallel_tri)
        for (int py = y0; py <= y1; ++py) {
            const float fy = static_cast<float>(py);
            const size_t row_base = static_cast<size_t>(py) * width;
            for (int px = x0; px <= x1; ++px) {
                const float fx = static_cast<float>(px);
                const float w0 = ((x[1] - fx) * (y[2] - fy) - (x[2] - fx) * (y[1] - fy)) * inv_area;
                const float w1 = ((x[2] - fx) * (y[0] - fy) - (x[0] - fx) * (y[2] - fy)) * inv_area;
                const float w2 = 1.0f - w0 - w1;
                if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) {
                    continue;
                }
                const size_t ti = row_base + static_cast<size_t>(px);
                if (out.pos_valid[ti * 4 + 3] > 0.5f) {
                    continue;   // first triangle wins
                }
                const Vec3 wp = p[0] * w0 + p[1] * w1 + p[2] * w2;
                Vec3 wn = n[0] * w0 + n[1] * w1 + n[2] * w2;
                const float nl = wn.length();
                wn = nl > 1e-6f ? wn / nl : geom_n;
                if (wn.dot(geom_n) < 0.0f && nl <= 1e-6f) {
                    wn = geom_n;
                }
                out.pos_valid[ti * 4 + 0] = wp.x;
                out.pos_valid[ti * 4 + 1] = wp.y;
                out.pos_valid[ti * 4 + 2] = wp.z;
                out.pos_valid[ti * 4 + 3] = 1.0f;
                out.flow_normal[ti * 4 + 0] = fx_flow;
                out.flow_normal[ti * 4 + 1] = fy_flow;
                out.flow_normal[ti * 4 + 2] = f_slope;
                out.flow_normal[ti * 4 + 3] = packOctNormal(wn);
                out.normal_spans[ti] = normal_span;
                ++tri_valid;
            }
        }
        out.valid_texels += tri_valid;
    }

    if (out.valid_texels == 0) {
        error = "no texel of " + object_name + " is covered by its UVs (uv_set " +
                std::to_string(uv_set) + ")";
        return false;
    }

    // Gutter: skip completely when every texel is already covered (common for
    // full-UV walls/floors: saves 2 x 40 MB copies + full scans per surface).
    if (gutter_texels > 0 && static_cast<size_t>(out.valid_texels) < texels) {
        std::vector<float> pv = out.pos_valid;
        std::vector<float> fn = out.flow_normal;
        std::vector<Vec2> spans = out.normal_spans;
        const int w_i = static_cast<int>(width);
        const int h_i = static_cast<int>(height);

        for (int pass = 0; pass < gutter_texels; ++pass) {
            if (pass > 0) {
                pv = out.pos_valid;
                fn = out.flow_normal;
                spans = out.normal_spans;
            }
            uint32_t pass_added = 0;
            #pragma omp parallel for schedule(static) reduction(+:pass_added)
            for (int py = 0; py < h_i; ++py) {
                static constexpr int kN[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                const size_t row_base = static_cast<size_t>(py) * width;
                for (int px = 0; px < w_i; ++px) {
                    const size_t ti = row_base + static_cast<size_t>(px);
                    if (out.pos_valid[ti * 4 + 3] > 0.5f) {
                        continue;
                    }
                    for (const auto& o : kN) {
                        const int nx = px + o[0];
                        const int ny = py + o[1];
                        if (nx < 0 || ny < 0 || nx >= w_i || ny >= h_i) {
                            continue;
                        }
                        const size_t ni = static_cast<size_t>(ny) * width + static_cast<size_t>(nx);
                        if (out.pos_valid[ni * 4 + 3] < 0.5f) {
                            continue;
                        }
                        for (int c = 0; c < 4; ++c) {
                            pv[ti * 4 + c] = out.pos_valid[ni * 4 + c];
                            fn[ti * 4 + c] = out.flow_normal[ni * 4 + c];
                        }
                        spans[ti] = out.normal_spans[ni];
                        pv[ti * 4 + 3] = 2.0f;
                        ++pass_added;
                        break;
                    }
                }
            }
            out.pos_valid.swap(pv);
            out.flow_normal.swap(fn);
            out.normal_spans.swap(spans);
            out.gutter_texels += pass_added;
            if (pass_added == 0 ||
                static_cast<size_t>(out.valid_texels) + out.gutter_texels >= texels) {
                break;
            }
        }
    }

    // Tiles: world AABB of their texels (island + gutter), parallelized per tile row.
    out.tiles_x = (width + tile_size - 1) / tile_size;
    out.tiles_y = (height + tile_size - 1) / tile_size;
    out.tiles.assign(static_cast<size_t>(out.tiles_x) * out.tiles_y, WetPaintBakeTile{});

    const int ty_count = static_cast<int>(out.tiles_y);
    #pragma omp parallel for schedule(static)
    for (int ty_i = 0; ty_i < ty_count; ++ty_i) {
        const uint32_t ty = static_cast<uint32_t>(ty_i);
        for (uint32_t tx = 0; tx < out.tiles_x; ++tx) {
            WetPaintBakeTile& t = out.tiles[static_cast<size_t>(ty) * out.tiles_x + tx];
            t.tx = tx;
            t.ty = ty;
            float mn[3] = {std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()};
            float mx[3] = {-std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max()};
            for (uint32_t ly = 0; ly < tile_size; ++ly) {
                const uint32_t py = ty * tile_size + ly;
                if (py >= height) {
                    break;
                }
                const size_t row_base = static_cast<size_t>(py) * width;
                for (uint32_t lx = 0; lx < tile_size; ++lx) {
                    const uint32_t px = tx * tile_size + lx;
                    if (px >= width) {
                        break;
                    }
                    const size_t ti = row_base + px;
                    if (out.pos_valid[ti * 4 + 3] < 0.5f) {
                        continue;
                    }
                    ++t.valid_texels;
                    for (int c = 0; c < 3; ++c) {
                        mn[c] = std::min(mn[c], out.pos_valid[ti * 4 + c]);
                        mx[c] = std::max(mx[c], out.pos_valid[ti * 4 + c]);
                    }
                }
            }
            if (t.valid_texels > 0) {
                for (int c = 0; c < 3; ++c) {
                    t.bmin[c] = mn[c];
                    t.bmax[c] = mx[c];
                }
            }
        }
    }

    return true;
}

} // namespace Paint
