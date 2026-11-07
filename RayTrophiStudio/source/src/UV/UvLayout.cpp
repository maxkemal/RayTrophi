// SPDX-License-Identifier: MIT
#include "UV/UvLayout.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace UV {

bool validate(const Layout& layout, std::string& error) {
    if (layout.coordinates.empty() || layout.indices.empty()) {
        error = "uv_empty_layout: mesh needs UVs and triangles";
        return false;
    }
    if (layout.positions.size() != layout.coordinates.size() ||
        layout.indices.size() % 3 != 0) {
        error = "uv_invalid_layout: inconsistent flat buffers";
        return false;
    }
    for (const Vec2& uv : layout.coordinates) {
        if (!std::isfinite(uv.u) || !std::isfinite(uv.v)) {
            error = "uv_non_finite: coordinates must be finite";
            return false;
        }
    }
    for (const Vec3& p : layout.positions) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            error = "uv_non_finite: positions must be finite";
            return false;
        }
    }
    for (uint32_t index : layout.indices) {
        if (index >= layout.coordinates.size()) {
            error = "uv_invalid_index: triangle index outside vertex buffer";
            return false;
        }
    }
    return true;
}

void buildIslands(Layout& layout) {
    const uint32_t faces = static_cast<uint32_t>(layout.indices.size() / 3);
    std::vector<uint32_t> parent(faces);
    std::iota(parent.begin(), parent.end(), 0u);
    auto root = [&](uint32_t face) {
        while (parent[face] != face) {
            parent[face] = parent[parent[face]];
            face = parent[face];
        }
        return face;
    };
    struct Edge {
        uint32_t face, a, b, count;
    };
    std::unordered_map<uint64_t, Edge> edges;
    for (uint32_t face = 0; face < faces; ++face) {
        for (uint32_t k = 0; k < 3; ++k) {
            const uint32_t a = layout.indices[face * 3 + k];
            const uint32_t b = layout.indices[face * 3 + (k + 1) % 3];
            const uint64_t key = (uint64_t((std::min)(a, b)) << 32) |
                                 (std::max)(a, b);
            auto [it, inserted] = edges.emplace(key, Edge{face, a, b, 0});
            ++it->second.count;
        }
    }
    // Non-manifold, same-direction and collapsed edges never join islands.
    for (uint32_t face = 0; face < faces; ++face) {
        for (uint32_t k = 0; k < 3; ++k) {
            const uint32_t a = layout.indices[face * 3 + k];
            const uint32_t b = layout.indices[face * 3 + (k + 1) % 3];
            const uint64_t key = (uint64_t((std::min)(a, b)) << 32) |
                                 (std::max)(a, b);
            const Edge& edge = edges.at(key);
            if (a != b && edge.count == 2 && edge.a == b && edge.b == a) {
                parent[root(face)] = root(edge.face);
            }
        }
    }
    std::unordered_map<uint32_t, uint32_t> ids;
    layout.islands.resize(faces);
    for (uint32_t face = 0; face < faces; ++face) {
        const uint32_t r = root(face);
        auto [it, inserted] = ids.emplace(r, static_cast<uint32_t>(ids.size()));
        layout.islands[face] = it->second;
    }
    layout.island_count = static_cast<uint32_t>(ids.size());
}

std::string fingerprint(const Layout& layout) {
    uint64_t hash = 14695981039346656037ull;
    auto word = [&](uint32_t bits) {
        for (int k = 0; k < 4; ++k) {
            hash ^= (bits >> (k * 8)) & 255u;
            hash *= 1099511628211ull;
        }
    };
    auto number = [&](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        word(bits);
    };
    word(static_cast<uint32_t>(layout.coordinates.size()));
    word(static_cast<uint32_t>(layout.indices.size()));
    for (uint32_t index : layout.indices) {
        word(index);
    }
    for (const Vec2& uv : layout.coordinates) {
        number(uv.u);
        number(uv.v);
    }
    for (const Vec3& p : layout.positions) {
        number(p.x);
        number(p.y);
        number(p.z);
    }
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

bool transform(Layout& layout, const std::vector<uint32_t>& islands,
               Vec2 offset, Vec2 scale, Vec2 pivot, float degrees,
               bool normalize, std::string& error, bool swap_axes,
               const std::vector<uint32_t>* vertices) {
    if (!validate(layout, error)) {
        return false;
    }
    for (float value : {offset.u, offset.v, scale.u, scale.v, pivot.u, pivot.v, degrees}) {
        if (!std::isfinite(value)) {
            error = "uv_non_finite: transform must be finite";
            return false;
        }
    }
    buildIslands(layout);
    std::unordered_set<uint32_t> selected(islands.begin(), islands.end());
    for (uint32_t id : selected) {
        if (id >= layout.island_count) {
            error = "uv_invalid_island";
            return false;
        }
    }
    std::vector<uint8_t> mask(layout.coordinates.size(), 0);
    if (vertices) {
        if (!islands.empty() || vertices->empty()) {
            error = "uv_invalid_selection: vertices must be nonempty and exclude islands";
            return false;
        }
        for (uint32_t vertex : *vertices) {
            if (vertex >= mask.size()) {
                error = "uv_invalid_vertex";
                return false;
            }
            mask[vertex] = 1;
        }
    }
    for (size_t face = 0; !vertices && face < layout.islands.size(); ++face) {
        const bool active = selected.empty() || selected.count(layout.islands[face]) != 0;
        for (size_t k = 0; k < 3; ++k) {
            mask[layout.indices[face * 3 + k]] |= active ? 1 : 2;
        }
    }
    if (std::find(mask.begin(), mask.end(), uint8_t{3}) != mask.end()) {
        error = "uv_shared_vertex: selected islands share a vertex with unselected islands";
        return false;
    }
    Vec2 lo((std::numeric_limits<float>::max)(), (std::numeric_limits<float>::max)());
    Vec2 hi(-(std::numeric_limits<float>::max)(), -(std::numeric_limits<float>::max)());
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] != 1) {
            continue;
        }
        lo.u = (std::min)(lo.u, layout.coordinates[i].u);
        lo.v = (std::min)(lo.v, layout.coordinates[i].v);
        hi.u = (std::max)(hi.u, layout.coordinates[i].u);
        hi.v = (std::max)(hi.v, layout.coordinates[i].v);
    }
    const float span = (std::max)(hi.u - lo.u, hi.v - lo.v);
    if (normalize && (!std::isfinite(span) || span <= 1.0e-12f)) {
        error = "uv_degenerate: cannot normalize a zero-area layout";
        return false;
    }
    const double angle = double(degrees) * 3.141592653589793 / 180.0;
    const double c = std::cos(angle), s = std::sin(angle);
    std::vector<Vec2> result = layout.coordinates;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] != 1) {
            continue;
        }
        const Vec2& uv = layout.coordinates[i];
        if (normalize) {
            result[i] = Vec2((uv.u - lo.u) / span, (uv.v - lo.v) / span);
        } else {
            const double x = (double(swap_axes ? uv.v : uv.u) - pivot.u) * scale.u;
            const double y = (double(swap_axes ? uv.u : uv.v) - pivot.v) * scale.v;
            result[i] = Vec2(float(c * x - s * y + pivot.u + offset.u),
                             float(s * x + c * y + pivot.v + offset.v));
        }
        if (!std::isfinite(result[i].u) || !std::isfinite(result[i].v)) {
            error = "uv_overflow: transform exceeds finite coordinate range";
            return false;
        }
    }
    layout.coordinates = std::move(result);
    return true;
}

} // namespace UV
