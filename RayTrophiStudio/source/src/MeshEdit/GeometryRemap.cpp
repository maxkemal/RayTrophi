// SPDX-License-Identifier: MIT
#include "DNA/GeometryRemap.h"
#include "DNA/GeometryDetail.h"

#include <limits>
#include <utility>

namespace DNA {

struct GeometryRemapAccess {
    static std::shared_ptr<GeometryDetail> run(
        const GeometryDetail& source, const std::vector<uint32_t>& vertices,
        const std::vector<uint32_t>& indices, std::string& error) {
        if (vertices.size() > (std::numeric_limits<uint32_t>::max)() ||
            indices.size() > (std::numeric_limits<uint32_t>::max)() ||
            indices.size() % 3 != 0) {
            error = "uv_invalid_remap_size";
            return {};
        }
        for (uint32_t vertex : vertices) {
            if (vertex >= source.vertex_count) {
                error = "uv_invalid_remap_source";
                return {};
            }
        }
        for (uint32_t index : indices) {
            if (index >= vertices.size()) {
                error = "uv_invalid_remap_index";
                return {};
            }
        }
        if (!source.skin_weights.empty() && source.skin_weights.size() != source.vertex_count) {
            error = "uv_attribute_size_mismatch: skin_weights";
            return {};
        }
        source.evaluate_active_state();
        std::lock_guard<std::mutex> lock(source.active_state_mutex);
        const auto& core = source.delta_stack.empty() ? source.core_base : source.core_active;
        const auto& custom = source.delta_stack.empty() ? source.custom_base : source.custom_active;
        auto result = std::make_shared<GeometryDetail>();
        result->vertex_count = vertices.size();
        result->indices.assign(indices.begin(), indices.end());
        auto gather = [&](const AttributeBuffer& buffer, const std::string& name) {
            if (buffer.element_count() != source.vertex_count) {
                error = "uv_attribute_size_mismatch: " + name;
                return std::unique_ptr<AttributeBuffer>{};
            }
            auto out = buffer.remap(vertices);
            if (!out || out->element_count() != vertices.size()) {
                error = "uv_attribute_remap_unsupported: " + name;
                return std::unique_ptr<AttributeBuffer>{};
            }
            return out;
        };
        for (size_t slot = 0; slot < GeometryDetail::kCoreCount; ++slot) {
            if (!core[slot]) {
                continue;
            }
            result->core_base[slot] = gather(*core[slot], "core " + std::to_string(slot));
            if (!result->core_base[slot]) {
                return {};
            }
        }
        for (const auto& [name, buffer] : custom) {
            if (!buffer) {
                continue;
            }
            auto remapped = gather(*buffer, name);
            if (!remapped) {
                return {};
            }
            result->custom_base.emplace(name, std::move(remapped));
        }
        if (!source.skin_weights.empty()) {
            result->skin_weights.reserve(vertices.size());
            for (uint32_t vertex : vertices) {
                result->skin_weights.push_back(source.skin_weights[vertex]);
            }
        }
        // The new topology must be skinned on the next update, even at the same pose.
        result->last_skinned_pose_hash = 0;
        return result;
    }
};

std::shared_ptr<GeometryDetail> remapGeometry(
    const GeometryDetail& source, const std::vector<uint32_t>& source_vertices,
    const std::vector<uint32_t>& new_indices, std::string& error) {
    return GeometryRemapAccess::run(source, source_vertices, new_indices, error);
}

} // namespace DNA
