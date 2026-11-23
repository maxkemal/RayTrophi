#include "Backend/VulkanMaterialTextureStorage.h"
#include "Backend/VulkanBackend.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace Backend::MaterialTextureStorage {

namespace {

namespace MTP = Backend::MaterialTexturePages;

struct GroupKey {
    VkFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t mipLevels;
    bool operator<(const GroupKey& o) const {
        return std::tie(format, width, height, mipLevels) <
               std::tie(o.format, o.width, o.height, o.mipLevels);
    }
};

struct Placement {
    int64_t id;
    uint32_t page;    // index into the new page list
    uint32_t layer;
};

constexpr VkImageUsageFlags kPageUsage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
// The ray-tracing stage is only valid when the RT pipeline extension is enabled
// (not on Android/Mali), so it is added per device.
VkPipelineStageFlags samplingStages(VulkanRT::VulkanDevice& device) {
    VkPipelineStageFlags stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (device.hasHardwareRT()) stages |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    return stages;
}

bool packable(const VulkanRT::ImageHandle& h) {
    return h.image && h.view && h.sampler && h.width && h.height && h.mipLevels &&
           h.format != VK_FORMAT_UNDEFINED;
}

void layerBarrier(VkCommandBuffer cmd, VkImage image, uint32_t layer, uint32_t mips,
                  VkImageLayout from, VkImageLayout to, VkAccessFlags srcAccess,
                  VkAccessFlags dstAccess, VkPipelineStageFlags srcStage,
                  VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, layer, 1};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

VkImageView createLayerView(VkDevice device, VkImage image, VkFormat format,
                            uint32_t layer, uint32_t mips) {
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = image;
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D;   // consumers bind it as a plain sampler2D
    ci.format = format;
    ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, layer, 1};
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(device, &ci, nullptr, &view) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return view;
}

constexpr uint64_t kPageByteBudget = 256ull * 1024ull * 1024ull;

// Bytes one texture occupies (all mips). Standalone: the image's own requirement;
// already paged: its page's share.
uint64_t estimateLayerBytes(VkDevice device, const VulkanRT::ImageHandle& h,
                            const std::vector<Page>& pages) {
    if (h.sharedImage) {
        for (const Page& page : pages) {
            if (page.image.image == h.image && page.image.layers) {
                return page.image.allocatedBytes / page.image.layers;
            }
        }
        return 0;
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, h.image, &req);
    return req.size;
}

void destroyPages(VulkanRT::VulkanDevice& device, std::vector<Page>& pages) {
    for (Page& page : pages) {
        if (page.sampler) {
            vkDestroySampler(device.getDevice(), page.sampler, nullptr);
        }
        MTP::destroyArrayImage(device, page.image);
    }
    pages.clear();
}

} // namespace

CompactionReport Storage::compact(VulkanRT::VulkanDevice& device,
                                  std::unordered_map<int64_t, VulkanRT::ImageHandle>& images,
                                  const SamplerFactory& makeSampler,
                                  const std::function<bool(int64_t)>& keepStandalone) {
    CompactionReport report;
    const VkDevice vk = device.getDevice();
    const VkPipelineStageFlags kSamplingStages = samplingStages(device);

    // 1. Group every packable 2D texture by exact key. Order ids for a stable layout.
    std::map<GroupKey, std::vector<int64_t>> groups;
    for (const auto& [id, h] : images) {
        if (!packable(h) || (keepStandalone && keepStandalone(id))) {
            ++report.texturesSkipped;
            continue;
        }
        groups[{h.format, h.width, h.height, h.mipLevels}].push_back(id);
    }
    for (auto& [key, ids] : groups) {
        std::sort(ids.begin(), ids.end());
    }

    // 2. Query per-format limits in the order the planner's compatibility ids use.
    std::vector<MTP::ImageProfile> profiles;
    std::vector<GroupKey> keys;
    for (const auto& [key, ids] : groups) {
        profiles.push_back({key.format, kPageUsage, 0});
        keys.push_back(key);
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device.getPhysicalDevice(), &props);
    const MTP::ImageLimits limits =
        MTP::queryImageLimits(device.getPhysicalDevice(), props.limits, profiles);

    // Largest groups first so the most textures land inside the viewport budget.
    std::vector<uint32_t> order(keys.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return groups[keys[a]].size() > groups[keys[b]].size();
    });

    VkCommandBuffer cmd = device.beginSingleTimeCommands();
    if (!cmd) {
        report.result = VK_ERROR_INITIALIZATION_FAILED;
        m_last = report;
        return report;
    }

    std::vector<Page> fresh;
    std::vector<Placement> placements;
    bool failed = false;
    for (uint32_t gi : order) {
        if (failed) break;
        const GroupKey& key = keys[gi];
        const std::vector<int64_t>& ids = groups[key];
        const MTP::Key pageKey{gi, key.width, key.height, key.mipLevels};
        uint32_t perPage = limits.layersFor(pageKey);
        // Keep each page allocation moderate: one huge allocation can exceed
        // maxMemoryAllocationSize / maxResourceSize and fail the whole compaction.
        const uint64_t layerBytes = estimateLayerBytes(vk, images.at(ids.front()), m_pages);
        if (perPage && layerBytes) {
            const uint64_t cap = (std::max)(uint64_t(1), kPageByteBudget / layerBytes);
            perPage = static_cast<uint32_t>((std::min)(uint64_t(perPage), cap));
        }
        if (!perPage || key.width > limits.maxDimension2D || key.height > limits.maxDimension2D) {
            report.texturesSkipped += static_cast<uint32_t>(ids.size());
            continue;
        }
        for (size_t first = 0; first < ids.size() && !failed; first += perPage) {
            const uint32_t layers =
                static_cast<uint32_t>((std::min)(static_cast<size_t>(perPage), ids.size() - first));
            Page page;
            const VkResult r = MTP::allocateArrayImage(device, limits, pageKey, layers,
                                                       VulkanRT::VramCategory::Texture, cmd,
                                                       kSamplingStages, page.image);
            if (r != VK_SUCCESS) {
                report.result = r;
                failed = true;
                break;
            }
            page.sampler = makeSampler ? makeSampler(key.mipLevels) : VK_NULL_HANDLE;
            if (!page.sampler) {
                MTP::destroyArrayImage(device, page.image);
                report.result = VK_ERROR_INITIALIZATION_FAILED;
                failed = true;
                break;
            }
            page.usedLayers = layers;
            const uint32_t pageIndex = static_cast<uint32_t>(fresh.size());
            for (uint32_t l = 0; l < layers; ++l) {
                const int64_t id = ids[first + l];
                const VulkanRT::ImageHandle& src = images.at(id);
                // Source at rest is SHADER_READ_ONLY (standalone or previous page layer).
                layerBarrier(cmd, src.image, src.arrayLayer, src.mipLevels,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 0,
                             VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT);
                layerBarrier(cmd, page.image.image, l, key.mipLevels,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                             VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT);
                std::vector<VkImageCopy> regions(key.mipLevels);
                uint32_t w = key.width, h = key.height;
                for (uint32_t m = 0; m < key.mipLevels; ++m) {
                    VkImageCopy& c = regions[m];
                    c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, src.arrayLayer, 1};
                    c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, l, 1};
                    c.extent = {w, h, 1};
                    w = (std::max)(1u, w / 2);
                    h = (std::max)(1u, h / 2);
                }
                vkCmdCopyImage(cmd, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               page.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<uint32_t>(regions.size()), regions.data());
                layerBarrier(cmd, src.image, src.arrayLayer, src.mipLevels,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
                             VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             kSamplingStages);
                layerBarrier(cmd, page.image.image, l, key.mipLevels,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, kSamplingStages);
                placements.push_back({id, pageIndex, l});
            }
            report.allocatedBytes += page.image.allocatedBytes;
            fresh.push_back(page);
        }
    }

    // Submit and wait even on failure: recorded barriers touch live images.
    device.endSingleTimeCommands(cmd);
    if (failed) {
        destroyPages(device, fresh);
        m_last = report;
        return report;
    }

    // 3. Per-layer 2D views for the new placement. Abort cleanly if any fails.
    std::vector<VkImageView> views(placements.size(), VK_NULL_HANDLE);
    for (size_t i = 0; i < placements.size(); ++i) {
        const Placement& p = placements[i];
        const VulkanRT::ImageHandle& src = images.at(p.id);
        views[i] = createLayerView(vk, fresh[p.page].image.image, src.format, p.layer,
                                   src.mipLevels);
        if (!views[i]) {
            for (VkImageView v : views) {
                if (v) vkDestroyImageView(vk, v, nullptr);
            }
            destroyPages(device, fresh);
            report.result = VK_ERROR_INITIALIZATION_FAILED;
            m_last = report;
            return report;
        }
    }

    // 4. Frames in flight may still sample the old views/images.
    device.waitIdle();

    // 5. Swap handles in place. Old view (+ standalone image/memory) is released;
    //    the per-texture sampler is kept.
    for (size_t i = 0; i < placements.size(); ++i) {
        const Placement& p = placements[i];
        VulkanRT::ImageHandle& h = images.at(p.id);
        vkDestroyImageView(vk, h.view, nullptr);
        if (!h.sharedImage) {
            vkDestroyImage(vk, h.image, nullptr);
            if (h.memory) device.freeTrackedMemory(h.memory);
        }
        h.image = fresh[p.page].image.image;
        h.view = views[i];
        h.memory = VK_NULL_HANDLE;
        h.arrayLayer = p.layer;
        h.sharedImage = true;
    }
    destroyPages(device, m_pages);
    m_pages = std::move(fresh);
    ++m_generation;

    report.texturesPacked = static_cast<uint32_t>(placements.size());
    report.pages = static_cast<uint32_t>(m_pages.size());
    report.pagesOverViewportBudget =
        report.pages > MTP::kPageBudget ? report.pages - MTP::kPageBudget : 0;
    m_last = report;
    return report;
}

void Storage::destroyAll(VulkanRT::VulkanDevice& device) {
    destroyPages(device, m_pages);
    destroyDescriptorResources(device);
    ++m_generation;
}

bool Storage::ensureDescriptorResources(VulkanRT::VulkanDevice& device) {
    if (descriptorResourcesReady()) return true;
    const VkDevice vk = device.getDevice();
    if (!m_dummy.view) {
        const std::vector<MTP::ImageProfile> profiles{
            {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0}};
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(device.getPhysicalDevice(), &props);
        const MTP::ImageLimits limits =
            MTP::queryImageLimits(device.getPhysicalDevice(), props.limits, profiles);
        VkCommandBuffer cmd = device.beginSingleTimeCommands();
        if (!cmd) return false;
        // Never sampled (unused page slots only carry invalid refs); it only has to
        // be a valid sampler2DArray descriptor.
        const VkResult r = MTP::allocateArrayImage(device, limits, MTP::Key{0, 1, 1, 1}, 1,
                                                   VulkanRT::VramCategory::Texture, cmd,
                                                   samplingStages(device), m_dummy);
        device.endSingleTimeCommands(cmd);
        if (r != VK_SUCCESS) return false;
    }
    if (!m_dummySampler) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxAnisotropy = 1.0f;
        if (vkCreateSampler(vk, &sci, nullptr, &m_dummySampler) != VK_SUCCESS) return false;
    }
    if (!m_refTable) {
        rebuildRefTable(device, {});
    }
    return descriptorResourcesReady();
}

void Storage::rebuildRefTable(VulkanRT::VulkanDevice& device,
                              const std::unordered_map<int64_t, VulkanRT::ImageHandle>& images) {
    int64_t maxId = 0;
    for (const auto& [id, h] : images) maxId = (std::max)(maxId, id);
    // Slot 0 is "no texture"; keep at least one entry so the binding is never empty.
    std::vector<uint32_t> table(static_cast<size_t>(maxId) + 1, MTP::kNoTexture);
    for (const auto& [id, h] : images) {
        if (id > 0) table[static_cast<size_t>(id)] = refFor(h);
    }
    VulkanRT::BufferCreateInfo ci{};
    ci.size = table.size() * sizeof(uint32_t);
    ci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST;
    ci.location = VulkanRT::MemoryLocation::GPU_ONLY;
    ci.category = VulkanRT::VramCategory::Texture;
    // createBuffer ignores initialData for GPU_ONLY memory: upload explicitly,
    // otherwise every slot reads garbage (all refs 0 -> page 0 layer 0).
    VulkanRT::BufferHandle fresh = device.createBuffer(ci);
    if (!fresh.buffer) return;   // keep the previous table rather than none
    device.uploadBuffer(fresh, table.data(), ci.size);
    destroyRefTable(device);
    m_refTable = fresh.buffer;
    m_refTableMemory = fresh.memory;
    m_refTableBytes = fresh.size;
}

void Storage::pageDescriptorImages(std::array<VkDescriptorImageInfo, 32>& out) const {
    static_assert(MTP::kPageBudget == 32, "viewport ABI is 32 pages");
    out.fill({m_dummySampler, m_dummy.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    const size_t live = (std::min)(m_pages.size(), out.size());
    for (size_t i = 0; i < live; ++i) {
        out[i] = {m_pages[i].sampler, m_pages[i].image.view,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }
}

VkDescriptorBufferInfo Storage::refTableDescriptor() const {
    return {m_refTable, 0, VK_WHOLE_SIZE};
}

void Storage::destroyRefTable(VulkanRT::VulkanDevice& device) {
    if (!m_refTable) return;
    VulkanRT::BufferHandle h{};
    h.buffer = m_refTable;
    h.memory = m_refTableMemory;
    h.size = m_refTableBytes;
    device.destroyBuffer(h);
    m_refTable = VK_NULL_HANDLE;
    m_refTableMemory = VK_NULL_HANDLE;
    m_refTableBytes = 0;
}

void Storage::destroyDescriptorResources(VulkanRT::VulkanDevice& device) {
    destroyRefTable(device);
    if (m_dummySampler) {
        vkDestroySampler(device.getDevice(), m_dummySampler, nullptr);
        m_dummySampler = VK_NULL_HANDLE;
    }
    MTP::destroyArrayImage(device, m_dummy);
}

uint32_t Storage::refFor(const VulkanRT::ImageHandle& handle) const {
    if (!handle.sharedImage) {
        return MTP::kNoTexture;
    }
    const uint32_t budget = (std::min)(static_cast<uint32_t>(m_pages.size()), MTP::kPageBudget);
    for (uint32_t i = 0; i < budget; ++i) {
        if (m_pages[i].image.image == handle.image) {
            return MTP::packRef(i, handle.arrayLayer);
        }
    }
    return MTP::kNoTexture;
}

} // namespace Backend::MaterialTextureStorage
