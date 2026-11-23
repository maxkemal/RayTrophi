#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace Backend::MaterialTexturePages {

constexpr uint32_t kPageBudget = 32;
constexpr uint32_t kNoTexture = 0xffffffffu;

struct Key {
    // Identifies a compatible Vulkan format, sampler and image usage.
    uint32_t compatibility = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 0;

    auto fields() const {
        return std::tie(compatibility, width, height, mipLevels);
    }

    bool operator==(const Key& other) const {
        return fields() == other.fields();
    }
};

struct Texture {
    Key key;
    uint32_t references = 1;
};

struct Page {
    Key key;
    std::vector<std::size_t> textures;
};

struct MergedKey {
    Key original;
    Key target;
    std::size_t textures = 0;
    uint64_t extraPayloadBytes = 0;
};

enum class PlanError {
    None,
    IncompatiblePageBudget
};

struct RejectedTexture {
    std::size_t texture;
    Key key;
};

struct Plan {
    PlanError error = PlanError::None;
    std::vector<Page> pages;
    std::vector<uint32_t> refs;
    std::vector<MergedKey> mergedKeys;
    std::vector<RejectedTexture> rejectedTextures;
};

struct Group {
    Key key;
    std::vector<std::size_t> textures;
    uint64_t references = 0;
    uint32_t layerLimit = 0;

    std::size_t pageCount() const {
        return 1 + (textures.size() - 1) / layerLimit;
    }
};

inline uint32_t layerLimit(uint32_t deviceLayers, uint32_t formatLayers) {
    // Reserve the all-ones packed texture reference for missing textures.
    return (std::min)({deviceLayers, formatLayers, 65535u});
}

inline uint32_t packRef(uint32_t page, uint32_t layer) {
    return (page << 16) | layer;
}

// queryLayers returns zero for unsupported extents, formats or mip counts.
// payloadBytes returns the actual per-layer format-block payload for the key.
template <typename QueryLayers, typename PayloadBytes>
Plan makePlan(const std::vector<Texture>& textures,
              uint32_t deviceLayerLimit,
              QueryLayers queryLayers,
              PayloadBytes payloadBytes) {
    Plan plan;
    plan.refs.resize(textures.size(), kNoTexture);
    std::vector<Group> groups;
    for (std::size_t i = 0; i < textures.size(); ++i) {
        const Texture& texture = textures[i];
        const Key& key = texture.key;
        if (!key.width || !key.height || !key.mipLevels) {
            plan.rejectedTextures.push_back({i, key});
            continue;
        }
        auto group = std::find_if(groups.begin(), groups.end(), [&](const Group& item) {
            return item.key == key;
        });
        if (group == groups.end()) {
            const uint32_t layers = layerLimit(deviceLayerLimit, queryLayers(key));
            if (!layers) {
                plan.rejectedTextures.push_back({i, key});
                continue;
            }
            groups.push_back({key, {}, 0, layers});
            group = groups.end() - 1;
        }
        group->textures.push_back(i);
        group->references += texture.references;
    }

    std::size_t pageCount = 0;
    for (const Group& group : groups) {
        pageCount += group.pageCount();
    }

    while (pageCount > kPageBudget) {
        std::size_t source = groups.size();
        std::size_t destination = groups.size();
        Key target{};
        uint32_t targetLayers = 0;
        std::size_t pagesSaved = 0;
        auto best = std::make_tuple(true, std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const Group& a = groups[i];
            for (std::size_t j = 0; j < groups.size(); ++j) {
                const Group& b = groups[j];
                if (i == j || a.key.compatibility != b.key.compatibility) {
                    continue;
                }
                const Key upper{
                    a.key.compatibility,
                    (std::max)(a.key.width, b.key.width),
                    (std::max)(a.key.height, b.key.height),
                    (std::max)(a.key.mipLevels, b.key.mipLevels)
                };
                const uint32_t layers = layerLimit(deviceLayerLimit, queryLayers(upper));
                if (!layers) {
                    continue;
                }
                const std::size_t oldPages = a.pageCount() + b.pageCount();
                const std::size_t newPages =
                    1 + (a.textures.size() + b.textures.size() - 1) / layers;
                if (newPages > oldPages) {
                    continue;
                }
                const uint64_t upperBytes = payloadBytes(upper);
                const uint64_t extra = (upperBytes - payloadBytes(a.key)) * a.textures.size() +
                                       (upperBytes - payloadBytes(b.key)) * b.textures.size();
                const auto rank = std::make_tuple(newPages == oldPages, a.references, extra);
                if (source == groups.size() || rank < best) {
                    source = i;
                    destination = j;
                    target = upper;
                    targetLayers = layers;
                    pagesSaved = oldPages - newPages;
                    best = rank;
                }
            }
        }
        if (source == groups.size()) {
            plan.error = PlanError::IncompatiblePageBudget;
            plan.refs.clear();
            return plan;
        }
        const Group& removed = groups[source];
        Group& retained = groups[destination];
        retained.key = target;
        retained.layerLimit = targetLayers;
        retained.references += removed.references;
        retained.textures.insert(retained.textures.end(),
                                 removed.textures.begin(), removed.textures.end());
        groups.erase(groups.begin() + source);
        pageCount -= pagesSaved;
    }

    plan.pages.reserve(pageCount);
    for (const Group& group : groups) {
        for (std::size_t first = 0; first < group.textures.size(); first += group.layerLimit) {
            const std::size_t end =
                (std::min)(group.textures.size(), first + group.layerLimit);
            const uint32_t pageIndex = static_cast<uint32_t>(plan.pages.size());
            Page& page = plan.pages.emplace_back(Page{group.key, {}});
            page.textures.reserve(end - first);
            for (std::size_t i = first; i < end; ++i) {
                const std::size_t texture = group.textures[i];
                const uint32_t layer = static_cast<uint32_t>(i - first);
                page.textures.push_back(texture);
                plan.refs[texture] = packRef(pageIndex, layer);
                const Key& original = textures[texture].key;
                if (original == group.key) {
                    continue;
                }
                auto report = std::find_if(
                    plan.mergedKeys.begin(), plan.mergedKeys.end(), [&](const MergedKey& key) {
                        return key.original == original && key.target == group.key;
                    });
                if (report == plan.mergedKeys.end()) {
                    plan.mergedKeys.push_back({original, group.key});
                    report = plan.mergedKeys.end() - 1;
                }
                ++report->textures;
                report->extraPayloadBytes += payloadBytes(group.key) - payloadBytes(original);
            }
        }
    }
    return plan;
}

} // namespace Backend::MaterialTexturePages
