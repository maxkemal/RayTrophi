#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <utility>
namespace RigAuthoring {
using VertexInfluences = std::vector<std::pair<int, float>>;
// Producer contract: merge duplicate IDs, discard invalid/nonpositive entries,
// retain strongest four, normalize. Empty stays empty; never invent a bone.
inline void canonicalizeInfluences(VertexInfluences& weights) {
    std::map<int, double> merged;
    for (const auto& w : weights)
        if (w.first >= 0 && std::isfinite(w.second) && w.second > 0) merged[w.first] += w.second;
    std::vector<std::pair<int, double>> sorted(merged.begin(), merged.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    if (sorted.size() > 4) sorted.resize(4);
    double sum = 0;
    for (const auto& w : sorted) sum += w.second;
    weights.clear();
    if (sum > 0) for (const auto& w : sorted) {
        const float normalized = static_cast<float>(w.second / sum);
        if (normalized > 0) weights.emplace_back(w.first, normalized);
    }
}
inline void canonicalizeSkinWeights(std::vector<VertexInfluences>& weights) {
    for (auto& vertex : weights) canonicalizeInfluences(vertex);
}
}
