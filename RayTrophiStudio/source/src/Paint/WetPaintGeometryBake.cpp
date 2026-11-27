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

    struct PreparedFace {
        float x[3] = {0.0f, 0.0f, 0.0f};
        float y[3] = {0.0f, 0.0f, 0.0f};
        float inv_area = 0.0f;
        float dw0_dx = 0.0f;
        float dw1_dx = 0.0f;
        int x0 = 0;
        int x1 = -1;
        int y0 = 0;
        int y1 = -1;
        Vec3 p[3];
        Vec3 n[3];
        Vec3 geom_n;
        Vec2 normal_span;
        float fx_flow = 0.0f;
        float fy_flow = 0.0f;
        float f_slope = 0.0f;
        float packed_flat_normal = 0.0f;
        bool flat_normal = false;
        bool valid = false;
    };

    const auto prepareFace = [&](const WetPaintBakeFace& tri) -> PreparedFace {
        PreparedFace pf;
        const auto& uv0 = tri.uvs[0];
        const auto& uv1 = tri.uvs[1];
        const auto& uv2 = tri.uvs[2];
        pf.x[0] = uv0.u * W1;
        pf.x[1] = uv1.u * W1;
        pf.x[2] = uv2.u * W1;
        pf.y[0] = (1.0f - uv0.v) * H1;
        pf.y[1] = (1.0f - uv1.v) * H1;
        pf.y[2] = (1.0f - uv2.v) * H1;
        const float area =
            (pf.x[1] - pf.x[0]) * (pf.y[2] - pf.y[0]) -
            (pf.x[2] - pf.x[0]) * (pf.y[1] - pf.y[0]);
        if (std::abs(area) < 1e-12f) {
            return pf;
        }
        pf.valid = true;
        pf.inv_area = 1.0f / area;
        pf.dw0_dx = (pf.y[1] - pf.y[2]) * pf.inv_area;
        pf.dw1_dx = (pf.y[2] - pf.y[0]) * pf.inv_area;
        pf.p[0] = tri.positions[0];
        pf.p[1] = tri.positions[1];
        pf.p[2] = tri.positions[2];
        Vec3 geom_n = (pf.p[1] - pf.p[0]).cross(pf.p[2] - pf.p[0]);
        const float gl = geom_n.length();
        pf.geom_n = gl > 1e-20f ? geom_n / gl : Vec3(0.0f, 1.0f, 0.0f);
        pf.n[0] = tri.normals[0];
        pf.n[1] = tri.normals[1];
        pf.n[2] = tri.normals[2];
        const bool same_n01 =
            (pf.n[0].x == pf.n[1].x && pf.n[0].y == pf.n[1].y && pf.n[0].z == pf.n[1].z);
        const bool same_n12 =
            (pf.n[1].x == pf.n[2].x && pf.n[1].y == pf.n[2].y && pf.n[1].z == pf.n[2].z);
        if (same_n01 && same_n12) {
            pf.flat_normal = true;
            const float nl = pf.n[0].length();
            const Vec3 wn = nl > 1e-6f ? pf.n[0] / nl : pf.geom_n;
            pf.packed_flat_normal = packOctNormal(wn);
        }
        pf.normal_span = HeightNormalMath::twoTexelWorldSpan(
            pf.p[0], pf.p[1], pf.p[2], uv0, uv1, uv2, width, height);
        RayTrophiSim::SurfaceFlowSample flow;
        const bool has_flow = RayTrophiSim::SurfaceFlowField::computePixelFlow(
            pf.p[0], pf.p[1], pf.p[2], uv0, uv1, uv2, static_cast<int>(width),
            static_cast<int>(height), flow, gravity) && flow.valid();
        pf.fx_flow = has_flow ? flow.flow_x : 0.0f;
        pf.fy_flow = has_flow ? flow.flow_y : 0.0f;
        pf.f_slope = has_flow ? flow.slope : 0.0f;

        pf.x0 = std::max(0, static_cast<int>(std::floor(std::min({pf.x[0], pf.x[1], pf.x[2]}))));
        pf.x1 = std::min(static_cast<int>(width) - 1,
                         static_cast<int>(std::ceil(std::max({pf.x[0], pf.x[1], pf.x[2]}))));
        pf.y0 = std::max(0, static_cast<int>(std::floor(std::min({pf.y[0], pf.y[1], pf.y[2]}))));
        pf.y1 = std::min(static_cast<int>(height) - 1,
                         static_cast<int>(std::ceil(std::max({pf.y[0], pf.y[1], pf.y[2]}))));
        return pf;
    };

    const auto rasterizeRow = [&](const PreparedFace& pf, int py) -> uint32_t {
        const float fy = static_cast<float>(py);
        const float fx0 = static_cast<float>(pf.x0);
        const float w0_base =
            ((pf.x[1] - fx0) * (pf.y[2] - fy) - (pf.x[2] - fx0) * (pf.y[1] - fy)) * pf.inv_area;
        const float w1_base =
            ((pf.x[2] - fx0) * (pf.y[0] - fy) - (pf.x[0] - fx0) * (pf.y[2] - fy)) * pf.inv_area;
        const size_t row_base = static_cast<size_t>(py) * width;
        uint32_t row_valid = 0;
        for (int px = pf.x0; px <= pf.x1; ++px) {
            const float dx = static_cast<float>(px - pf.x0);
            const float w0 = w0_base + dx * pf.dw0_dx;
            const float w1 = w1_base + dx * pf.dw1_dx;
            const float w2 = 1.0f - w0 - w1;
            if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) {
                continue;
            }
            const size_t ti = row_base + static_cast<size_t>(px);
            if (out.pos_valid[ti * 4 + 3] > 0.5f) {
                continue;   // first triangle wins
            }
            const Vec3 wp = pf.p[0] * w0 + pf.p[1] * w1 + pf.p[2] * w2;
            float packed_n = pf.packed_flat_normal;
            if (!pf.flat_normal) {
                Vec3 wn = pf.n[0] * w0 + pf.n[1] * w1 + pf.n[2] * w2;
                const float nl = wn.length();
                wn = nl > 1e-6f ? wn / nl : pf.geom_n;
                packed_n = packOctNormal(wn);
            }
            out.pos_valid[ti * 4 + 0] = wp.x;
            out.pos_valid[ti * 4 + 1] = wp.y;
            out.pos_valid[ti * 4 + 2] = wp.z;
            out.pos_valid[ti * 4 + 3] = 1.0f;
            out.flow_normal[ti * 4 + 0] = pf.fx_flow;
            out.flow_normal[ti * 4 + 1] = pf.fy_flow;
            out.flow_normal[ti * 4 + 2] = pf.f_slope;
            out.flow_normal[ti * 4 + 3] = packed_n;
            out.normal_spans[ti] = pf.normal_span;
            ++row_valid;
        }
        return row_valid;
    };

    // High-poly meshes (many small triangles) parallelize poorly per-triangle.
    // Bin faces into disjoint horizontal row strips in ascending face order so
    // all CPU cores rasterize simultaneously with exact first-triangle-wins.
    if (faces.size() >= 64) {
        const size_t face_count = faces.size();
        std::vector<PreparedFace> prepared(face_count);
        const int64_t face_count_i = static_cast<int64_t>(face_count);
        uint32_t tri_count = 0;
        #pragma omp parallel for schedule(static) reduction(+:tri_count)
        for (int64_t fi = 0; fi < face_count_i; ++fi) {
            prepared[static_cast<size_t>(fi)] = prepareFace(faces[static_cast<size_t>(fi)]);
            if (prepared[static_cast<size_t>(fi)].valid) {
                ++tri_count;
            }
        }
        out.triangles = tri_count;

        constexpr int kStripRows = 16;
        const int num_strips = (static_cast<int>(height) + kStripRows - 1) / kStripRows;
        std::vector<std::vector<uint32_t>> strip_faces(static_cast<size_t>(num_strips));
        for (size_t fi = 0; fi < face_count; ++fi) {
            const PreparedFace& pf = prepared[fi];
            if (!pf.valid || pf.x0 > pf.x1 || pf.y0 > pf.y1) {
                continue;
            }
            const int s0 = pf.y0 / kStripRows;
            const int s1 = pf.y1 / kStripRows;
            for (int s = s0; s <= s1; ++s) {
                strip_faces[static_cast<size_t>(s)].push_back(static_cast<uint32_t>(fi));
            }
        }

        uint32_t total_valid = 0;
        #pragma omp parallel for schedule(dynamic, 1) reduction(+:total_valid)
        for (int s = 0; s < num_strips; ++s) {
            const int sy0 = s * kStripRows;
            const int sy1 = std::min(static_cast<int>(height) - 1, sy0 + kStripRows - 1);
            uint32_t strip_valid = 0;
            for (uint32_t fi : strip_faces[static_cast<size_t>(s)]) {
                const PreparedFace& pf = prepared[fi];
                const int ry0 = std::max(pf.y0, sy0);
                const int ry1 = std::min(pf.y1, sy1);
                for (int py = ry0; py <= ry1; ++py) {
                    strip_valid += rasterizeRow(pf, py);
                }
            }
            total_valid += strip_valid;
        }
        out.valid_texels = total_valid;
    } else {
        for (const auto& tri : faces) {
            const PreparedFace pf = prepareFace(tri);
            if (!pf.valid) {
                continue;
            }
            ++out.triangles;
            if (pf.x0 > pf.x1 || pf.y0 > pf.y1) {
                continue;
            }
            const int rows = pf.y1 - pf.y0 + 1;
            const int cols = pf.x1 - pf.x0 + 1;
            const bool parallel_tri = rows >= 32 && static_cast<int64_t>(rows) * cols >= 4096;
            uint32_t tri_valid = 0;
            #pragma omp parallel for schedule(static) reduction(+:tri_valid) if(parallel_tri)
            for (int py = pf.y0; py <= pf.y1; ++py) {
                tri_valid += rasterizeRow(pf, py);
            }
            out.valid_texels += tri_valid;
        }
    }

    if (out.valid_texels == 0) {
        error = "no texel of " + object_name + " is covered by its UVs (uv_set " +
                std::to_string(uv_set) + ")";
        return false;
    }

    // Gutter: skip completely when every texel is already covered (common for
    // full-UV walls/floors: saves 2 x 40 MB copies + full scans per surface).
    if (gutter_texels > 0 && static_cast<size_t>(out.valid_texels) < texels) {
        std::vector<float> pv(texels * 4);
        std::vector<float> fn(texels * 4);
        std::vector<Vec2> spans(texels);
        const int w_i = static_cast<int>(width);
        const int h_i = static_cast<int>(height);

        for (int pass = 0; pass < gutter_texels; ++pass) {
            uint32_t pass_added = 0;
            #pragma omp parallel for schedule(static) reduction(+:pass_added)
            for (int py = 0; py < h_i; ++py) {
                static constexpr int kN[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                const size_t row_base = static_cast<size_t>(py) * width;
                for (int px = 0; px < w_i; ++px) {
                    const size_t ti = row_base + static_cast<size_t>(px);
                    const size_t t4 = ti * 4;
                    if (out.pos_valid[t4 + 3] > 0.5f) {
                        for (int c = 0; c < 4; ++c) {
                            pv[t4 + c] = out.pos_valid[t4 + c];
                            fn[t4 + c] = out.flow_normal[t4 + c];
                        }
                        spans[ti] = out.normal_spans[ti];
                        continue;
                    }
                    bool filled = false;
                    for (const auto& o : kN) {
                        const int nx = px + o[0];
                        const int ny = py + o[1];
                        if (nx < 0 || ny < 0 || nx >= w_i || ny >= h_i) {
                            continue;
                        }
                        const size_t ni = static_cast<size_t>(ny) * width + static_cast<size_t>(nx);
                        const size_t n4 = ni * 4;
                        if (out.pos_valid[n4 + 3] < 0.5f) {
                            continue;
                        }
                        for (int c = 0; c < 4; ++c) {
                            pv[t4 + c] = out.pos_valid[n4 + c];
                            fn[t4 + c] = out.flow_normal[n4 + c];
                        }
                        spans[ti] = out.normal_spans[ni];
                        pv[t4 + 3] = 2.0f;
                        ++pass_added;
                        filled = true;
                        break;
                    }
                    if (!filled) {
                        for (int c = 0; c < 4; ++c) {
                            pv[t4 + c] = 0.0f;
                            fn[t4 + c] = 0.0f;
                        }
                        spans[ti] = Vec2(1.0f, 1.0f);
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
