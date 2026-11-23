#pragma once

#include "VulkanMaterialTextureArrayImage.h"

#include <array>
#include <optional>

namespace Backend::MaterialTexturePages {

enum class LayerState : uint8_t {
    Unloaded,
    Resident,
    Retired
};

struct PublishedPage {
    const ArrayImage* image = nullptr;
    VkSampler sampler = VK_NULL_HANDLE;
    // Missing entries and reserved uploads are unloaded until ready to publish.
    std::vector<LayerState> layers;
};

using PublishedPages = std::array<PublishedPage, kPageBudget>;
using DescriptorImages = std::array<VkDescriptorImageInfo, kPageBudget>;

enum class DescriptorRejection {
    None,
    InvalidDummy,
    InvalidPage,
    LayerStateOverCapacity,
    PageOutOfRange,
    MissingPage,
    LayerOutOfRange,
    LayerUnloaded,
    LayerRetired
};

struct DescriptorPreparation {
    DescriptorRejection reason = DescriptorRejection::None;
    std::optional<uint32_t> page;
    std::optional<uint32_t> layer;
    std::optional<std::size_t> referenceIndex;
    std::optional<uint32_t> reference;

    explicit operator bool() const {
        return reason == DescriptorRejection::None;
    }
};

inline DescriptorRejection checkPublishedPage(const PublishedPage& page) {
    if (!page.image) {
        return !page.sampler && page.layers.empty() ? DescriptorRejection::None
                                                    : DescriptorRejection::InvalidPage;
    }
    if (!page.image->view || !page.sampler || !page.image->layers) {
        return DescriptorRejection::InvalidPage;
    }
    return page.layers.size() <= page.image->layers ? DescriptorRejection::None
                                                   : DescriptorRejection::LayerStateOverCapacity;
}

inline DescriptorRejection checkPublishedTexture(const PublishedPages& pages, uint32_t ref) {
    if (ref == kNoTexture) {
        return DescriptorRejection::None;
    }
    const uint32_t pageIndex = ref >> 16;
    const uint32_t layer = ref & 0xffffu;
    if (pageIndex >= pages.size()) {
        return DescriptorRejection::PageOutOfRange;
    }
    const PublishedPage& page = pages[pageIndex];
    if (!page.image) {
        return DescriptorRejection::MissingPage;
    }
    if (layer >= page.image->layers) {
        return DescriptorRejection::LayerOutOfRange;
    }
    if (layer >= page.layers.size() || page.layers[layer] == LayerState::Unloaded) {
        return DescriptorRejection::LayerUnloaded;
    }
    return page.layers[layer] == LayerState::Resident ? DescriptorRejection::None
                                                     : DescriptorRejection::LayerRetired;
}

// The caller keeps all views and samplers alive for the descriptor set's frame
// snapshot and has ordered their shader-read transitions before that frame.
// On rejection, publish neither the new descriptor set nor the material buffer.
// The previous pair can only be rendered while its complete snapshot is retained.
inline DescriptorPreparation prepareDescriptorImages(const PublishedPages& pages,
                                                      const std::vector<uint32_t>& references,
                                                      const ArrayImage& dummy,
                                                      VkSampler dummySampler,
                                                      DescriptorImages& output) {
    if (!dummy.view || !dummySampler || dummy.layers != 1 || dummy.key.width != 1 ||
        dummy.key.height != 1 || dummy.key.mipLevels != 1) {
        return {DescriptorRejection::InvalidDummy, {}, {}, {}, {}};
    }
    DescriptorImages images;
    images.fill({dummySampler, dummy.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    for (std::size_t i = 0; i < pages.size(); ++i) {
        const PublishedPage& page = pages[i];
        const DescriptorRejection reason = checkPublishedPage(page);
        if (reason != DescriptorRejection::None) {
            return {reason, static_cast<uint32_t>(i), {}, {}, {}};
        }
        if (page.image) {
            images[i] = {page.sampler, page.image->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
    }
    for (std::size_t i = 0; i < references.size(); ++i) {
        const uint32_t ref = references[i];
        const DescriptorRejection reason = checkPublishedTexture(pages, ref);
        if (reason != DescriptorRejection::None) {
            return {reason, ref >> 16, ref & 0xffffu, i, ref};
        }
    }
    output = images;
    return {};
}

// A single write fills the entire binding. The caller owns descriptor-set
// synchronization and must only update a set that is safe to rewrite.
inline VkWriteDescriptorSet descriptorWrite(VkDescriptorSet set, uint32_t binding,
                                            const DescriptorImages& images) {
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = static_cast<uint32_t>(images.size());
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = images.data();
    return write;
}

} // namespace Backend::MaterialTexturePages
