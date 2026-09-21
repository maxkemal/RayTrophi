#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include "../../shaders/raster_material_policy.h"
#include "../../shaders/surface_coverage.h"

namespace Backend {

// Only membership is cached. Material values and program membership are read
// live, so opaque -> glass edits cannot leave a cached negative classification.
class RasterMaterialUsage {
public:
    void invalidate() { valid_ = false; }
    const std::vector<uint32_t>& ids(const std::vector<uint32_t>& stream) const {
        if (!valid_) {
            std::unordered_set<uint32_t> unique;
            for (uint32_t id : stream) unique.insert(id);
            ids_.assign(unique.begin(), unique.end());
            std::sort(ids_.begin(), ids_.end());
            valid_ = true;
        }
        return ids_;
    }
private:
    mutable bool valid_ = false;
    mutable std::vector<uint32_t> ids_;
};

class RasterMaterialPrograms {
public:
    void update(const std::vector<uint32_t>& words) {
        active_.clear();
        unknown_ = false;
        if (words.empty()) return;
        const std::size_t count = words[0];
        if (count >= words.size()) { unknown_ = true; return; }
        active_.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            active_.push_back(words[i + 1] != UINT32_MAX);
    }
    bool hasProgram(uint32_t id) const {
        return unknown_ || (id < active_.size() && active_[id]);
    }
private:
    bool unknown_ = false;
    std::vector<bool> active_;
};

template<class Materials>
bool rasterMeshMayTransmit(const RasterMaterialUsage& usage,
                          const std::vector<uint32_t>& stream,
                          uint32_t vertexCount, const Materials& materials,
                          uint32_t boundCount, const RasterMaterialPrograms& programs,
                          bool externalMaterials) {
    if (!vertexCount) return false;
    // Missing CPU provenance must retain the draw, never silently erase glass.
    if (externalMaterials || !boundCount || materials.size() < boundCount ||
        stream.size() < vertexCount) return true;
    for (uint32_t encoded : usage.ids(stream)) {
        if ((encoded & 0x80000000u) != 0u) continue; // shader impostors are opaque
        const uint32_t id = (std::min)(encoded & 0x7fffffffu, boundCount - 1u);
        const auto& m = materials[id];
        if (RasterMaterialPolicy::rasterMaterialMayTransmit(
                m.transmission, m.opacity, m.transmission_tex != 0u,
                m.opacity_tex != 0u, programs.hasProgram(id),
                (m.flags & (1u << 19u)) != 0u,
                (m.flags & MATERIAL_FLAGS_PREVIEW_CUTOUT) != 0u)) return true;
    }
    return false;
}

// EQUAL shading is allowed only when the camera prepass proves exactly the
// same coverage. One unsupported member keeps a mixed mesh on the legacy path.
template<class Materials>
bool rasterMeshHasExactCoverage(const RasterMaterialUsage& usage,
                               const std::vector<uint32_t>& stream,
                               uint32_t vertexCount, const Materials& materials,
                               uint32_t boundCount, const RasterMaterialPrograms& programs,
                               bool externalMaterials) {
    if (!vertexCount || externalMaterials || !boundCount ||
        materials.size() < boundCount || stream.size() < vertexCount) return false;
    for (uint32_t encoded : usage.ids(stream)) {
        if ((encoded & 0x80000000u) != 0u) continue;
        const uint32_t id = (std::min)(encoded & 0x7fffffffu, boundCount - 1u);
        const auto& m = materials[id];
        if (programs.hasProgram(id) || !(m.tile_break_strength <= 0.0f) ||
            !(m.transmission <= 0.001f) || m.transmission_tex != 0u ||
            (m.flags & ((1u << 17u) | (1u << 19u) | (1u << 24u))) != 0u) return false;
        if ((m.flags & MATERIAL_FLAGS_PREVIEW_CUTOUT) == 0u &&
            (!(m.opacity >= 0.999f) || m.opacity_tex != 0u)) return false;
    }
    return true;
}
} // namespace Backend
