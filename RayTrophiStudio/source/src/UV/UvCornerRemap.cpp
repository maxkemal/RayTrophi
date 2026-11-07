// SPDX-License-Identifier: MIT
#include "UV/UvLayout.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace UV {
namespace {

struct CornerKey {
    uint32_t vertex, u, v;
    bool operator==(const CornerKey& other) const {
        return vertex == other.vertex && u == other.u && v == other.v;
    }
};

struct CornerHash {
    size_t operator()(const CornerKey& key) const {
        size_t value = key.vertex;
        value ^= size_t(key.u) + 0x9e3779b9u + (value << 6) + (value >> 2);
        value ^= size_t(key.v) + 0x9e3779b9u + (value << 6) + (value >> 2);
        return value;
    }
};

uint32_t bits(float value) {
    if (value == 0.0f) {
        return 0;
    }
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

} // namespace

bool applyCornerCoordinates(Layout& layout, const std::vector<Vec2>& corners,
                            std::string& error) {
    if (!validate(layout, error)) {
        return false;
    }
    if (corners.size() != layout.indices.size()) {
        error = "uv_corner_count_mismatch";
        return false;
    }
    for (const Vec2& uv : corners) {
        if (!std::isfinite(uv.u) || !std::isfinite(uv.v)) {
            error = "uv_non_finite: corner coordinates must be finite";
            return false;
        }
    }
    if (corners.size() > (std::numeric_limits<uint32_t>::max)() ||
        layout.positions.size() > (std::numeric_limits<uint32_t>::max)() - corners.size()) {
        error = "uv_remap_too_large";
        return false;
    }
    // Keep all existing slots, including unreferenced vertices. Only append seam copies.
    Layout result = layout;
    if (result.source_vertices.empty()) {
        result.source_vertices.resize(layout.positions.size());
        std::iota(result.source_vertices.begin(), result.source_vertices.end(), 0u);
    } else if (result.source_vertices.size() != layout.positions.size()) {
        error = "uv_invalid_existing_remap";
        return false;
    }
    std::vector<uint8_t> assigned(layout.positions.size(), 0);
    std::unordered_map<CornerKey, uint32_t, CornerHash> lookup;
    lookup.reserve(corners.size());
    for (size_t corner = 0; corner < corners.size(); ++corner) {
        const uint32_t source = layout.indices[corner];
        const Vec2 uv = corners[corner];
        const CornerKey key{source, bits(uv.u), bits(uv.v)};
        auto found = lookup.find(key);
        uint32_t output;
        if (found != lookup.end()) {
            output = found->second;
        } else {
            if (!assigned[source]) {
                output = source;
                result.coordinates[output] = uv;
                assigned[source] = 1;
            } else {
                output = static_cast<uint32_t>(result.coordinates.size());
                result.coordinates.push_back(uv);
                result.positions.push_back(layout.positions[source]);
                result.source_vertices.push_back(result.source_vertices[source]);
            }
            lookup.emplace(key, output);
        }
        result.indices[corner] = output;
    }
    buildIslands(result);
    layout = std::move(result);
    return true;
}

} // namespace UV
