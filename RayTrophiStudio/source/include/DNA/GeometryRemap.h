// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace DNA {
class GeometryDetail;

// Main-thread, detached snapshot: copies the evaluated attribute state with type-safe
// per-element gathers. Authored source arrays and the original delta stack stay unchanged.
std::shared_ptr<GeometryDetail> remapGeometry(
    const GeometryDetail& source, const std::vector<uint32_t>& source_vertices,
    const std::vector<uint32_t>& new_indices, std::string& error);
} // namespace DNA
