#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Triangle;

namespace HierarchyUI {

using MeshCacheEntry = std::pair<
    std::string,
    std::vector<std::pair<int, std::shared_ptr<Triangle>>>>;

using NamePredicate = std::function<bool(const std::string&)>;

// Header-only by design: this tiny UI view must not depend on Visual Studio
// discovering an additional translation unit when the hierarchy file changes.
// Callers provide canonical scene-liveness and optional grouping predicates.
inline std::vector<std::size_t> buildLiveObjectView(
    const std::vector<MeshCacheEntry>& cache,
    const NamePredicate& is_live,
    const NamePredicate& passes_filter,
    const NamePredicate& is_grouped = {}) {
    std::vector<std::size_t> indices;
    indices.reserve(cache.size());

    for (std::size_t i = 0; i < cache.size(); ++i) {
        const auto& entry = cache[i];
        if (entry.first.empty() || entry.second.empty() || !entry.second.front().second)
            continue;
        if (is_live && !is_live(entry.first)) continue;
        if (passes_filter && !passes_filter(entry.first)) continue;
        if (is_grouped && is_grouped(entry.first)) continue;
        indices.push_back(i);
    }
    return indices;
}

} // namespace HierarchyUI
