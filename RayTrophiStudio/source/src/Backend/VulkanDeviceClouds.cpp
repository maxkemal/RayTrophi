// Cloud field resources on one VkDevice (docs/dev/ATMOSPHERE_CLOUDS.md, Faz 3a):
// generated noise textures, the weather map, and the measurement dispatch
// behind world.sample_clouds. Each Vulkan device builds its own copy with the
// same shader and inputs -- two VkDevices cannot share a texture.

#include "Backend/VulkanBackend.h"
#include "Backend/CloudParams.h"
#include "Backend/VulkanFailureReporting.h"
#include "Atmosphere/AtmosphereLocalWeather.h"
#include "SimFrameCompress.h"

#include <algorithm>
#include <cstring>

namespace VulkanRT {

namespace {

VkSampler makeRepeatSampler(VkDevice device) {
    VkSamplerCreateInfo s{};
    s.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    s.magFilter = VK_FILTER_LINEAR;
    s.minFilter = VK_FILTER_LINEAR;
    s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // REPEAT: every cloud texture tiles by construction (cloud_noise.comp).
    s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.maxLod = 0.0f;
    VkSampler out = VK_NULL_HANDLE;
    vkCreateSampler(device, &s, nullptr, &out);
    return out;
}

VkSampler makeClampSampler(VkDevice device) {
    VkSamplerCreateInfo s{};
    s.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    s.magFilter = VK_FILTER_LINEAR;
    s.minFilter = VK_FILTER_LINEAR;
    s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    s.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.maxLod = 0.0f;
    VkSampler out = VK_NULL_HANDLE;
    vkCreateSampler(device, &s, nullptr, &out);
    return out;
}

// FNV-1a over the shape inputs of a volume slot (b.xyz, c, d, e, f): equal key =
// same generated SDF, so the slot is not repainted.
uint64_t hashCloudVolumeSlot(const Backend::CloudVolumeSlotGPU& s) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const float* v, int count) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(v);
        for (size_t i = 0; i < size_t(count) * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    };
    mix(s.b, 3);
    mix(s.c, 4);
    mix(s.d, 4);
    mix(s.e, 4);
    mix(s.f, 4);
    return h | 1ull;   // never 0: 0 means "not painted"
}

bool makeComputePipeline(VkDevice device, const std::vector<uint32_t>& spv, VkPipelineLayout layout,
                         VkPipeline& out) {
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = spv.size() * sizeof(uint32_t);
    sm.pCode = spv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &sm, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.layout = layout;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = module;
    cp.stage.pName = "main";
    const bool ok = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &out) == VK_SUCCESS;
    vkDestroyShaderModule(device, module, nullptr);
    return ok;
}

} // namespace

ImageHandle VulkanDevice::createImage3D(uint32_t w, uint32_t h, uint32_t d, VkFormat format,
                                        VkImageUsageFlags usage) {
    ImageHandle handle{};
    if (w == 0 || h == 0 || d == 0) return {};
    handle.width = w;
    handle.height = h;
    handle.format = format;

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_3D;
    ii.format = format;
    ii.extent = {w, h, d};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = usage;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(m_device, &ii, nullptr, &handle.image) != VK_SUCCESS) return {};

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_device, handle.image, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(m_device, &ai, nullptr, &handle.memory) != VK_SUCCESS) {
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }
    noteMemoryAllocated(handle.memory, req.size, ai.memoryTypeIndex, VramCategory::RenderTarget);
    if (vkBindImageMemory(m_device, handle.image, handle.memory, 0) != VK_SUCCESS) {
        freeTrackedMemory(handle.memory);
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = handle.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vi.format = format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(m_device, &vi, nullptr, &handle.view) != VK_SUCCESS) {
        vkDestroyImage(m_device, handle.image, nullptr);
        freeTrackedMemory(handle.memory);
        return {};
    }
    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) { destroyImage(handle); return {}; }
    transitionImageLayout(cmd, handle.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    endSingleTimeCommands(cmd);
    return handle;
}

void VulkanDevice::destroyCloudResources() {
    destroyCloudRaster();
    destroyCloudPipelines();
    destroyCloudUploadRing();
    for (ImageHandle* img : {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, &m_cloudMajorant,
                             &m_cloudCellMap, &m_cloudVolumeAtlas}) {
        if (img->image || img->view) destroyImage(*img);
        *img = {};
    }
    for (BufferHandle* b : {&m_cloudSampleParamsBuffer, &m_cloudQueryBuffer, &m_cloudResultBuffer,
                            &m_cloudRtParamsBuffer, &m_cloudCellsBuffer}) {
        if (b->buffer) destroyBuffer(*b);
        *b = {};
    }
    m_cloudVolumeSlots = 0;
    for (uint64_t& key : m_cloudVolumeKeys) key = 0;
    m_cloudQueryCapacity = 0;
    m_cloudNoiseReady = false;
    m_cloudWeatherHash = 0;
}

bool VulkanDevice::ensureCloudTextures() {
    if (m_cloudBase.view && m_cloudMajorant.view && m_cloudRtParamsBuffer.buffer &&
        m_cloudCellMap.view && m_cloudCellsBuffer.buffer && m_cloudVolumeAtlas.view) return true;
    const VkImageUsageFlags use = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VramCategoryScope vram(VramCategory::RenderTarget);
    // createImage2D / createImage3D leave the images in GENERAL: the storage
    // layout for generation and a legal sampled layout for the readers.
    m_cloudBase = createImage3D(Backend::kCloudBaseNoiseSize, Backend::kCloudBaseNoiseSize,
                                Backend::kCloudBaseNoiseSize, VK_FORMAT_R8G8B8A8_UNORM, use);
    m_cloudDetail = createImage3D(Backend::kCloudDetailNoiseSize, Backend::kCloudDetailNoiseSize,
                                  Backend::kCloudDetailNoiseSize, VK_FORMAT_R8G8B8A8_UNORM, use);
    m_cloudCurl = createImage2D(Backend::kCloudCurlNoiseSize, Backend::kCloudCurlNoiseSize,
                                VK_FORMAT_R8G8B8A8_UNORM, use);
    m_cloudWeather = createImage2D(Backend::kCloudWeatherMapSize, Backend::kCloudWeatherMapSize,
                                   VK_FORMAT_R8G8B8A8_UNORM, use);
    m_cloudMajorant = createImage2D(Backend::kCloudMajorantSize, Backend::kCloudMajorantSize,
                                    VK_FORMAT_R8G8B8A8_UNORM, use);
    m_cloudCellMap = createImage2D(Backend::kCloudCellMapSize, Backend::kCloudCellMapRows,
                                   VK_FORMAT_R16G16B16A16_SFLOAT, use | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    bool ok = m_cloudBase.view && m_cloudDetail.view && m_cloudCurl.view && m_cloudWeather.view &&
              m_cloudMajorant.view && m_cloudCellMap.view;
    if (ok) {
        for (ImageHandle* img : {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather}) {
            img->sampler = makeRepeatSampler(m_device);
            ok = ok && img->sampler != VK_NULL_HANDLE;
        }
        // The majorant is read with texelFetch; any sampler will do.
        m_cloudMajorant.sampler = makeRepeatSampler(m_device);
        ok = ok && m_cloudMajorant.sampler != VK_NULL_HANDLE;
        // The cell map does NOT tile: linear, clamped (cloudStormCellAt also
        // returns zero outside it).
        VkSamplerCreateInfo cs{};
        cs.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        cs.magFilter = VK_FILTER_LINEAR;
        cs.minFilter = VK_FILTER_LINEAR;
        cs.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        cs.addressModeU = cs.addressModeV = cs.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        cs.maxLod = 0.0f;
        ok = ok && vkCreateSampler(m_device, &cs, nullptr, &m_cloudCellMap.sampler) == VK_SUCCESS;
    }
    if (ok) {
        // No storm until the first cells dispatch: the map must read zero,
        // not uninitialised memory.
        VkCommandBuffer cmd = beginSingleTimeCommands();
        ok = cmd != VK_NULL_HANDLE;
        if (ok) {
            const VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_cloudCellMap.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
            endSingleTimeCommands(cmd);
        }
    }
    // Volume atlas placeholder: bound everywhere from the first descriptor
    // write; holds no slot (cloudVolumeSdf skips every entry).
    if (ok && !m_cloudVolumeAtlas.view) {
        ensureCloudVolumeAtlas(0);
        ok = m_cloudVolumeAtlas.view != VK_NULL_HANDLE;
    }
    if (ok) {
        BufferCreateInfo ci{};
        ci.size = sizeof(Backend::CloudCellsGPU);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_cloudCellsBuffer = createBuffer(ci);
        ok = m_cloudCellsBuffer.buffer != VK_NULL_HANDLE;
    }
    if (ok) {
        BufferCreateInfo ci{};
        ci.size = sizeof(Backend::CloudParamsGPU);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
        ci.location = MemoryLocation::GPU_ONLY;
        m_cloudRtParamsBuffer = createBuffer(ci);
        ok = m_cloudRtParamsBuffer.buffer != VK_NULL_HANDLE;
    }
    if (ok) {
        // Zero = every layer off and the render flag clear: RT draws no cloud
        // from uninitialised texture memory.
        const Backend::CloudParamsGPU zero{};
        uploadBuffer(m_cloudRtParamsBuffer, &zero, sizeof(zero));
        return true;
    }
    for (ImageHandle* img : {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, &m_cloudMajorant,
                             &m_cloudCellMap, &m_cloudVolumeAtlas}) {
        if (img->image || img->view) destroyImage(*img);
        *img = {};
    }
    m_cloudVolumeSlots = 0;
    for (BufferHandle* b : {&m_cloudRtParamsBuffer, &m_cloudCellsBuffer}) {
        if (b->buffer) destroyBuffer(*b);
        *b = {};
    }
    return false;
}

// RT set bindings 30-37: base, detail, curl, weather, params, majorant, cell
// map, volume atlas.
void VulkanDevice::writeRtCloudDescriptors() {
    if (m_rtDescriptorSet == VK_NULL_HANDLE || !ensureCloudTextures()) return;
    const ImageHandle* imgs[7] = {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, &m_cloudMajorant,
                                  &m_cloudCellMap, &m_cloudVolumeAtlas};
    const uint32_t bindings[7] = {30, 31, 32, 33, 35, 36, 37};
    VkDescriptorImageInfo ii[7]{};
    VkWriteDescriptorSet w[8]{};
    for (uint32_t i = 0; i < 7; ++i) {
        ii[i].sampler = imgs[i]->sampler;
        ii[i].imageView = imgs[i]->view;
        ii[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = m_rtDescriptorSet;
        w[i].dstBinding = bindings[i];
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[i].pImageInfo = &ii[i];
    }
    VkDescriptorBufferInfo bi{m_cloudRtParamsBuffer.buffer, 0, sizeof(Backend::CloudParamsGPU)};
    w[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[7].dstSet = m_rtDescriptorSet;
    w[7].dstBinding = 34;
    w[7].descriptorCount = 1;
    w[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[7].pBufferInfo = &bi;
    vkUpdateDescriptorSets(m_device, 8, w, 0, nullptr);
}

// ── Storm cell volumes (ATMOSPHERE_CLOUD_VOLUMES.md §3.3) ──────────────────

// Atlas of `slots` 128^3 RGBA16F slots stacked along z; 0 = a 2x2x1
// placeholder that holds no slot (z / x = 0). Reallocation waits for the
// device: frames in flight read the old image through every cloud set.
bool VulkanDevice::ensureCloudVolumeAtlas(uint32_t slots) {
    slots = (std::min)(slots, static_cast<uint32_t>(atmosphere::kMaxCloudVolumeSlots));
    if (m_cloudVolumeAtlas.view && slots == m_cloudVolumeSlots) return false;
    waitIdle();
    if (m_cloudVolumeAtlas.image || m_cloudVolumeAtlas.view) destroyImage(m_cloudVolumeAtlas);
    m_cloudVolumeAtlas = {};
    const uint32_t n = static_cast<uint32_t>(atmosphere::kCloudVolumeVoxels);
    {
        VramCategoryScope vram(VramCategory::RenderTarget);
        m_cloudVolumeAtlas = slots == 0
            ? createImage3D(2, 2, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            : createImage3D(n, n, n * slots, VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    }
    if (!m_cloudVolumeAtlas.view && slots > 0) {
        // Out of memory for the requested budget: keep a placeholder, the
        // cells stay in the storm-column shell (world.cloud_stats shows 0 slots).
        m_cloudVolumeAtlas = createImage3D(2, 2, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        slots = 0;
    }
    if (!m_cloudVolumeAtlas.view) {
        m_cloudVolumeSlots = 0;
        return false;
    }
    m_cloudVolumeAtlas.sampler = makeClampSampler(m_device);
    {
        // Far outside every surface until a slot is painted.
        VkCommandBuffer cmd = beginSingleTimeCommands();
        if (cmd != VK_NULL_HANDLE) {
            VkClearColorValue far{};
            far.float32[0] = 2000.0f;
            far.float32[3] = 1.0f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_cloudVolumeAtlas.image, VK_IMAGE_LAYOUT_GENERAL, &far, 1, &range);
            endSingleTimeCommands(cmd);
        }
    }
    m_cloudVolumeSlots = slots;
    for (uint64_t& key : m_cloudVolumeKeys) key = 0;
    ++m_cloudVolumeAllocations;
    writeCloudVolumeDescriptors();
    return true;
}

// The atlas in every set that reads the cloud field, plus the generator's
// storage binding. Only called after a device idle (ensureCloudVolumeAtlas)
// or from a set's own creation, so no in-flight frame holds these sets.
void VulkanDevice::writeCloudVolumeDescriptors() {
    if (!m_cloudVolumeAtlas.view) return;
    VkDescriptorImageInfo sampled{m_cloudVolumeAtlas.sampler, m_cloudVolumeAtlas.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo storage{VK_NULL_HANDLE, m_cloudVolumeAtlas.view, VK_IMAGE_LAYOUT_GENERAL};
    std::vector<VkWriteDescriptorSet> ws;
    auto add = [&](VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const VkDescriptorImageInfo* ii) {
        if (set == VK_NULL_HANDLE) return;
        VkWriteDescriptorSet x{};
        x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        x.dstSet = set;
        x.dstBinding = binding;
        x.descriptorCount = 1;
        x.descriptorType = type;
        x.pImageInfo = ii;
        ws.push_back(x);
    };
    add(m_rtDescriptorSet, 37, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampled);
    add(m_cloudRasterSets[0], 17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampled);
    add(m_cloudRasterSets[1], 17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampled);
    add(m_cloudSceneSet, 17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampled);
    add(m_cloudSampleDescSet, 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampled);
    add(m_cloudNearDescSet, 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sampled);
    add(m_cloudVolumeSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &storage);
    if (!ws.empty()) vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
}

bool VulkanDevice::createCloudVolumePipeline(const std::vector<uint32_t>& spv) {
    if (m_cloudVolumePipeline) return true;
    if (spv.empty() || !ensureCloudTextures()) return false;
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 1; li.pBindings = &b;
    bool ok = vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_cloudVolumeDescLayout) == VK_SUCCESS;
    if (ok) {
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = 1; pi.pPoolSizes = &size; pi.maxSets = 1;
        ok = vkCreateDescriptorPool(m_device, &pi, nullptr, &m_cloudVolumePool) == VK_SUCCESS;
    }
    if (ok) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = m_cloudVolumePool; ai.descriptorSetCount = 1;
        ai.pSetLayouts = &m_cloudVolumeDescLayout;
        ok = vkAllocateDescriptorSets(m_device, &ai, &m_cloudVolumeSet) == VK_SUCCESS;
    }
    if (ok) {
        // The slot entry (CloudVolumeSlotGPU, 96 B) + atlas slot / voxels.
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 112};
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &m_cloudVolumeDescLayout;
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &range;
        ok = vkCreatePipelineLayout(m_device, &pl, nullptr, &m_cloudVolumeLayout) == VK_SUCCESS;
    }
    ok = ok && makeComputePipeline(m_device, spv, m_cloudVolumeLayout, m_cloudVolumePipeline);
    if (ok) {
        VkDescriptorImageInfo storage{VK_NULL_HANDLE, m_cloudVolumeAtlas.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = m_cloudVolumeSet;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &storage;
        vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
        for (uint64_t& key : m_cloudVolumeKeys) key = 0;   // paint every live slot
        return true;
    }
    if (m_cloudVolumePipeline) vkDestroyPipeline(m_device, m_cloudVolumePipeline, nullptr);
    if (m_cloudVolumeLayout) vkDestroyPipelineLayout(m_device, m_cloudVolumeLayout, nullptr);
    if (m_cloudVolumePool) vkDestroyDescriptorPool(m_device, m_cloudVolumePool, nullptr);
    if (m_cloudVolumeDescLayout) vkDestroyDescriptorSetLayout(m_device, m_cloudVolumeDescLayout, nullptr);
    m_cloudVolumePipeline = VK_NULL_HANDLE;
    m_cloudVolumeLayout = VK_NULL_HANDLE;
    m_cloudVolumePool = VK_NULL_HANDLE;
    m_cloudVolumeDescLayout = VK_NULL_HANDLE;
    m_cloudVolumeSet = VK_NULL_HANDLE;
    return false;
}

void VulkanDevice::destroyCloudUploadRing() {
    for (CloudUploadSlot& slot : m_cloudUploadSlots) {
        if (slot.fence) {
            vkWaitForFences(m_device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
            vkDestroyFence(m_device, slot.fence, nullptr);
        }
        if (slot.cmd) vkFreeCommandBuffers(m_device, m_commandPool, 1, &slot.cmd);
        if (slot.staging.buffer) destroyBuffer(slot.staging);
        slot = {};
    }
    m_cloudUploadNext = 0;
}

// Params + storm cells + the cell map, in queue order and WITHOUT a CPU wait.
// Every frame is submitted to this same queue, so a frame submitted after
// this sees the new block; the leading barrier keeps this from overwriting
// what an earlier, still-running frame reads. The CPU waits only when all
// ring slots are still in flight (four updates, ~two frames, ahead).
void VulkanDevice::updateCloudRtParams(const Backend::CloudParamsGPU& params,
                                       const Backend::CloudCellsGPU& cells) {
    m_lensWind[0] = params.cirrusDirection[3];
    m_lensWind[1] = params.emissionSource[3];
    m_lensTime = params.cirrusDirection[2];
    if (!ensureCloudTextures()) return;
    // Volume atlas follows the requested budget (0 with volumes off: the
    // 128 MB atlas is released). Rare; waits for the device when it changes.
    ensureCloudVolumeAtlas(static_cast<uint32_t>(params.volume[3] + 0.5f));
    // Volumes are drawn only where this device can paint them: without the
    // generator or an atlas the flag is cleared and the shell keeps the cells
    // (an unpainted atlas reads "far" -- the cells would vanish otherwise).
    const bool volumesLive = m_cloudVolumePipeline != VK_NULL_HANDLE && m_cloudVolumeSlots > 0;
    constexpr VkDeviceSize kCellsOffset = (sizeof(Backend::CloudParamsGPU) + 255u) & ~VkDeviceSize(255u);
    // Local fog envelope: one RGBA16F cell-map sheet, uploaded only when the
    // published map changed (its placement is in `params`, same snapshot).
    constexpr VkDeviceSize kFogOffset =
        (kCellsOffset + sizeof(Backend::CloudCellsGPU) + 255u) & ~VkDeviceSize(255u);
    constexpr VkDeviceSize kFogBytes =
        VkDeviceSize(Backend::kCloudCellMapSize) * Backend::kCloudCellMapSize * 4u * sizeof(uint16_t);
    constexpr VkDeviceSize kStagingBytes = kFogOffset + kFogBytes;
    const std::shared_ptr<const atmosphere::LocalFogMap> fog = atmosphere::localFogMapSnapshot();
    const uint64_t fogRevision = fog && fog->active ? fog->revision : 0;
    const bool uploadFog = fogRevision != 0 && fogRevision != m_cloudFogRevision &&
        fog->texels.size() == size_t(Backend::kCloudCellMapSize) * Backend::kCloudCellMapSize;
    CloudUploadSlot& slot = m_cloudUploadSlots[m_cloudUploadNext];
    m_cloudUploadNext = (m_cloudUploadNext + 1u) % kCloudUploadSlots;
    if (!slot.fence) {
        BufferCreateInfo ci{};
        ci.size = kStagingBytes;
        ci.usage = BufferUsage::TRANSFER_SRC;
        ci.location = MemoryLocation::CPU_TO_GPU;
        slot.staging = createBuffer(ci);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = m_commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (!slot.staging.buffer ||
            vkAllocateCommandBuffers(m_device, &ai, &slot.cmd) != VK_SUCCESS ||
            vkCreateFence(m_device, &fi, nullptr, &slot.fence) != VK_SUCCESS) {
            // Never lose an update: fall back to the blocking path (no volume
            // repaint on it, so the volumes are off for this update).
            Backend::CloudParamsGPU gated = params;
            gated.volume[0] = 0.0f;
            uploadBuffer(m_cloudRtParamsBuffer, &gated, sizeof(gated));
            uploadBuffer(m_cloudCellsBuffer, &cells, sizeof(cells));
            if (slot.cmd) vkFreeCommandBuffers(m_device, m_commandPool, 1, &slot.cmd);
            if (slot.staging.buffer) destroyBuffer(slot.staging);
            slot = {};
            return;
        }
    }
    vkWaitForFences(m_device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(m_device, 1, &slot.fence);
    void* mapped = nullptr;
    if (vkMapMemory(m_device, slot.staging.memory, 0, kStagingBytes, 0, &mapped) != VK_SUCCESS) return;
    std::memcpy(mapped, &params, sizeof(params));
    if (!volumesLive) {
        static_cast<Backend::CloudParamsGPU*>(mapped)->volume[0] = 0.0f;
    }
    std::memcpy(static_cast<char*>(mapped) + kCellsOffset, &cells, sizeof(cells));
    if (uploadFog) {
        uint16_t* halfs = reinterpret_cast<uint16_t*>(static_cast<char*>(mapped) + kFogOffset);
        for (size_t i = 0; i < fog->texels.size(); ++i) {
            const atmosphere::LocalFogTexel& t = fog->texels[i];
            halfs[i * 4 + 0] = RayTrophiSim::SimFrameCompress::floatToHalf(t.fog);
            halfs[i * 4 + 1] = RayTrophiSim::SimFrameCompress::floatToHalf(t.fog_top);
            halfs[i * 4 + 2] = RayTrophiSim::SimFrameCompress::floatToHalf(t.mist);
            halfs[i * 4 + 3] = RayTrophiSim::SimFrameCompress::floatToHalf(t.ground);
        }
    }
    vkUnmapMemory(m_device, slot.staging.memory);

    vkResetCommandBuffer(slot.cmd, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(slot.cmd, &bi);
    VkMemoryBarrier before{};
    before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(slot.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &before, 0, nullptr, 0, nullptr);
    const VkBufferCopy paramsCopy{0, 0, sizeof(params)};
    vkCmdCopyBuffer(slot.cmd, slot.staging.buffer, m_cloudRtParamsBuffer.buffer, 1, &paramsCopy);
    const VkBufferCopy cellsCopy{kCellsOffset, 0, sizeof(cells)};
    vkCmdCopyBuffer(slot.cmd, slot.staging.buffer, m_cloudCellsBuffer.buffer, 1, &cellsCopy);
    if (uploadFog) {
        VkBufferImageCopy region{};
        region.bufferOffset = kFogOffset;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, static_cast<int32_t>(Backend::kCloudFogSheet * Backend::kCloudCellMapSize), 0};
        region.imageExtent = {Backend::kCloudCellMapSize, Backend::kCloudCellMapSize, 1};
        vkCmdCopyBufferToImage(slot.cmd, slot.staging.buffer, m_cloudCellMap.image,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        m_cloudFogRevision = fogRevision;
        ++m_cloudFogUploads;
    }
    VkMemoryBarrier copied{};
    copied.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(slot.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &copied, 0, nullptr, 0, nullptr);
    if (m_cloudCellsPipeline && m_cloudCellsSet) {
        vkCmdBindPipeline(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudCellsPipeline);
        vkCmdBindDescriptorSets(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudCellsLayout, 0, 1,
                                &m_cloudCellsSet, 0, nullptr);
        const uint32_t groups = (Backend::kCloudCellMapSize + 7u) / 8u;
        vkCmdDispatch(slot.cmd, groups, groups, 1);
        VkMemoryBarrier painted{};
        painted.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        painted.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        painted.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(slot.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &painted, 0, nullptr, 0, nullptr);
    }
    // Storm cell volumes: repaint the slots whose shape inputs changed. The
    // leading barrier of this buffer already ordered the write after every
    // earlier frame's reads; the trailing one publishes it to later frames.
    m_cloudVolumeLastRepaints = 0;
    if (volumesLive && m_cloudVolumeSet && params.volume[0] > 0.5f) {
        bool bound = false;
        for (int i = 0; i < atmosphere::kMaxCloudVolumeSlots; ++i) {
            const Backend::CloudVolumeSlotGPU& v = params.volumeSlots[i];
            if (v.a[3] < 0.5f) break;
            const uint32_t atlasSlot = static_cast<uint32_t>(v.b[3] + 0.5f);
            if (atlasSlot >= m_cloudVolumeSlots) continue;
            const uint64_t key = hashCloudVolumeSlot(v);
            if (key == m_cloudVolumeKeys[atlasSlot]) continue;
            if (!bound) {
                vkCmdBindPipeline(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudVolumePipeline);
                vkCmdBindDescriptorSets(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudVolumeLayout, 0, 1,
                                        &m_cloudVolumeSet, 0, nullptr);
                bound = true;
            }
            struct VolumeGenPush {
                Backend::CloudVolumeSlotGPU slot;
                uint32_t info[4];
            } push{v, {atlasSlot, static_cast<uint32_t>(atmosphere::kCloudVolumeVoxels), 0u, 0u}};
            static_assert(sizeof(VolumeGenPush) == 112, "cloud_volume_gen.comp push ABI");
            vkCmdPushConstants(slot.cmd, m_cloudVolumeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            const uint32_t groups = (static_cast<uint32_t>(atmosphere::kCloudVolumeVoxels) + 7u) / 8u;
            vkCmdDispatch(slot.cmd, groups, groups, groups);
            m_cloudVolumeKeys[atlasSlot] = key;
            ++m_cloudVolumeLastRepaints;
            ++m_cloudVolumeGenerations;
        }
        if (bound) {
            VkMemoryBarrier painted{};
            painted.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            painted.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            painted.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(slot.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &painted, 0, nullptr, 0, nullptr);
        }
    }
    // Near-camera drops (ATMOSPHERE_NEAR_PRECIP.md): their camera state --
    // rate, form mix, cloud shadow -- ONCE here, written into this params
    // buffer, instead of in every pixel of RT and RayFusion. After the cell
    // map and volumes it reads; skipped (state stays zero = no drops) when
    // near drops are off or the field is not generated yet.
    if ((params.precip3[3] > 0.0f || m_lensWeatherCamera.settings.enabled) && m_cloudSamplePipeline && m_cloudNearDescSet &&
        m_cloudNoiseReady && m_cloudWeatherHash != 0) {
        vkCmdBindPipeline(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudSamplePipeline);
        vkCmdBindDescriptorSets(slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudSamplePipelineLayout,
                                0, 1, &m_cloudNearDescSet, 0, nullptr);
        const uint32_t pc[4] = {7u, 1u, 1u, Backend::kCloudNearStateFloatOffset};
        vkCmdPushConstants(slot.cmd, m_cloudSamplePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(slot.cmd, 1, 1, 1);
        VkMemoryBarrier nearDone{};
        nearDone.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        nearDone.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        nearDone.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(slot.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &nearDone, 0, nullptr, 0, nullptr);
        ++m_cloudNearStateDispatches;
    }
    vkEndCommandBuffer(slot.cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &slot.cmd;
    reportVulkanDeviceFailure(vkQueueSubmit(m_computeQueue, 1, &si, slot.fence),
                              "updateCloudRtParams/vkQueueSubmit");
}

bool VulkanDevice::createCloudCellsPipeline(const std::vector<uint32_t>& spv) {
    if (m_cloudCellsPipeline) return true;
    if (spv.empty() || !ensureCloudTextures()) return false;
    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding = 0;
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[0].descriptorCount = 1;
    b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[1].descriptorCount = 1;
    b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 2; li.pBindings = b;
    bool ok = vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_cloudCellsDescLayout) == VK_SUCCESS;
    if (ok) {
        VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = 2; pi.pPoolSizes = sizes; pi.maxSets = 1;
        ok = vkCreateDescriptorPool(m_device, &pi, nullptr, &m_cloudCellsPool) == VK_SUCCESS;
    }
    if (ok) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = m_cloudCellsPool; ai.descriptorSetCount = 1;
        ai.pSetLayouts = &m_cloudCellsDescLayout;
        ok = vkAllocateDescriptorSets(m_device, &ai, &m_cloudCellsSet) == VK_SUCCESS;
    }
    if (ok) {
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &m_cloudCellsDescLayout;
        ok = vkCreatePipelineLayout(m_device, &pl, nullptr, &m_cloudCellsLayout) == VK_SUCCESS;
    }
    ok = ok && makeComputePipeline(m_device, spv, m_cloudCellsLayout, m_cloudCellsPipeline);
    if (ok) {
        VkDescriptorBufferInfo buffer{m_cloudCellsBuffer.buffer, 0, sizeof(Backend::CloudCellsGPU)};
        VkDescriptorImageInfo image{VK_NULL_HANDLE, m_cloudCellMap.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = m_cloudCellsSet;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
        }
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[0].pBufferInfo = &buffer;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo = &image;
        vkUpdateDescriptorSets(m_device, 2, w, 0, nullptr);
        return true;
    }
    if (m_cloudCellsPipeline) vkDestroyPipeline(m_device, m_cloudCellsPipeline, nullptr);
    if (m_cloudCellsLayout) vkDestroyPipelineLayout(m_device, m_cloudCellsLayout, nullptr);
    if (m_cloudCellsPool) vkDestroyDescriptorPool(m_device, m_cloudCellsPool, nullptr);
    if (m_cloudCellsDescLayout) vkDestroyDescriptorSetLayout(m_device, m_cloudCellsDescLayout, nullptr);
    m_cloudCellsPipeline = VK_NULL_HANDLE;
    m_cloudCellsLayout = VK_NULL_HANDLE;
    m_cloudCellsPool = VK_NULL_HANDLE;
    m_cloudCellsDescLayout = VK_NULL_HANDLE;
    m_cloudCellsSet = VK_NULL_HANDLE;
    return false;
}

void VulkanDevice::destroyCloudPipelines() {
    auto kill = [&](VkPipeline& p) { if (p) { vkDestroyPipeline(m_device, p, nullptr); p = VK_NULL_HANDLE; } };
    auto killL = [&](VkPipelineLayout& p) { if (p) { vkDestroyPipelineLayout(m_device, p, nullptr); p = VK_NULL_HANDLE; } };
    auto killD = [&](VkDescriptorSetLayout& p) { if (p) { vkDestroyDescriptorSetLayout(m_device, p, nullptr); p = VK_NULL_HANDLE; } };
    kill(m_cloudNoisePipeline);
    kill(m_cloudSamplePipeline);
    killL(m_cloudNoisePipelineLayout);
    killL(m_cloudSamplePipelineLayout);
    killD(m_cloudNoiseDescLayout);
    killD(m_cloudSampleDescLayout);
    if (m_cloudDescPool) { vkDestroyDescriptorPool(m_device, m_cloudDescPool, nullptr); m_cloudDescPool = VK_NULL_HANDLE; }
    m_cloudNoiseDescSet = VK_NULL_HANDLE;
    m_cloudSampleDescSet = VK_NULL_HANDLE;
    m_cloudNearDescSet = VK_NULL_HANDLE;
    kill(m_cloudCellsPipeline);
    killL(m_cloudCellsLayout);
    killD(m_cloudCellsDescLayout);
    if (m_cloudCellsPool) { vkDestroyDescriptorPool(m_device, m_cloudCellsPool, nullptr); m_cloudCellsPool = VK_NULL_HANDLE; }
    m_cloudCellsSet = VK_NULL_HANDLE;
    kill(m_cloudVolumePipeline);
    killL(m_cloudVolumeLayout);
    killD(m_cloudVolumeDescLayout);
    if (m_cloudVolumePool) { vkDestroyDescriptorPool(m_device, m_cloudVolumePool, nullptr); m_cloudVolumePool = VK_NULL_HANDLE; }
    m_cloudVolumeSet = VK_NULL_HANDLE;
}

bool VulkanDevice::createCloudPipelines(const std::vector<uint32_t>& noiseSPV,
                                        const std::vector<uint32_t>& sampleSPV) {
    if (noiseSPV.empty() || sampleSPV.empty()) return false;
    // Pipelines only: the textures may already be bound to the RT set.
    destroyCloudPipelines();
    if (!ensureCloudTextures()) return false;

    // Noise set: 0,1 = 3D storage, 2,3 = 2D storage, 4 = majorant (2D storage).
    VkDescriptorSetLayoutBinding nb[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        nb[i].binding = i;
        nb[i].descriptorCount = 1;
        nb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        nb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // Sample set: 0-3 sampled textures, 4 params, 5 queries, 6 results,
    // 7 storm cell map, 8 storm cell volume atlas.
    VkDescriptorSetLayoutBinding sb[9]{};
    for (uint32_t i = 0; i < 9; ++i) {
        sb[i].binding = i;
        sb[i].descriptorCount = 1;
        sb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        sb[i].descriptorType = (i < 4 || i >= 7) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                 : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 5; li.pBindings = nb;
    bool ok = vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_cloudNoiseDescLayout) == VK_SUCCESS;
    li.bindingCount = 9; li.pBindings = sb;
    ok = ok && vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_cloudSampleDescLayout) == VK_SUCCESS;

    if (ok) {
        // Noise set + sample set + near-state set (the sample layout twice).
        VkDescriptorPoolSize sizes[3] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6}};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = 3; pi.pPoolSizes = sizes; pi.maxSets = 3;
        ok = vkCreateDescriptorPool(m_device, &pi, nullptr, &m_cloudDescPool) == VK_SUCCESS;
    }
    if (ok) {
        VkDescriptorSetLayout layouts[3] = {m_cloudNoiseDescLayout, m_cloudSampleDescLayout,
                                            m_cloudSampleDescLayout};
        VkDescriptorSet sets[3] = {};
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = m_cloudDescPool; ai.descriptorSetCount = 3; ai.pSetLayouts = layouts;
        ok = vkAllocateDescriptorSets(m_device, &ai, sets) == VK_SUCCESS;
        m_cloudNoiseDescSet = sets[0];
        m_cloudSampleDescSet = sets[1];
        m_cloudNearDescSet = sets[2];
    }
    auto makeLayout = [&](VkDescriptorSetLayout set, uint32_t pushBytes, VkPipelineLayout& out) {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &set;
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &range;
        return vkCreatePipelineLayout(m_device, &pl, nullptr, &out) == VK_SUCCESS;
    };
    ok = ok && makeLayout(m_cloudNoiseDescLayout, 16, m_cloudNoisePipelineLayout);
    ok = ok && makeLayout(m_cloudSampleDescLayout, 16, m_cloudSamplePipelineLayout);
    ok = ok && makeComputePipeline(m_device, noiseSPV, m_cloudNoisePipelineLayout, m_cloudNoisePipeline);
    ok = ok && makeComputePipeline(m_device, sampleSPV, m_cloudSamplePipelineLayout, m_cloudSamplePipeline);

    if (ok && !m_cloudSampleParamsBuffer.buffer) {
        BufferCreateInfo ci{};
        ci.size = sizeof(Backend::CloudParamsGPU);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::CPU_TO_GPU;
        m_cloudSampleParamsBuffer = createBuffer(ci);
        ok = m_cloudSampleParamsBuffer.buffer != VK_NULL_HANDLE;
    }
    if (!ok) {
        destroyCloudPipelines();
        return false;
    }

    // Noise set: fixed.
    VkDescriptorImageInfo storage[5]{};
    const ImageHandle* imgs[5] = {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, &m_cloudMajorant};
    VkWriteDescriptorSet w[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        storage[i].imageView = imgs[i]->view;
        storage[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = m_cloudNoiseDescSet;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &storage[i];
    }
    vkUpdateDescriptorSets(m_device, 5, w, 0, nullptr);

    // Sample set: textures + params fixed; queries/results written per call.
    VkDescriptorImageInfo sampled[4]{};
    VkWriteDescriptorSet sw[5]{};
    for (uint32_t i = 0; i < 4; ++i) {
        sampled[i].sampler = imgs[i]->sampler;
        sampled[i].imageView = imgs[i]->view;
        sampled[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sw[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw[i].dstSet = m_cloudSampleDescSet;
        sw[i].dstBinding = i;
        sw[i].descriptorCount = 1;
        sw[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw[i].pImageInfo = &sampled[i];
    }
    VkDescriptorBufferInfo pinfo{m_cloudSampleParamsBuffer.buffer, 0, sizeof(Backend::CloudParamsGPU)};
    sw[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    sw[4].dstSet = m_cloudSampleDescSet;
    sw[4].dstBinding = 4;
    sw[4].descriptorCount = 1;
    sw[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sw[4].pBufferInfo = &pinfo;
    VkDescriptorImageInfo cellInfo{m_cloudCellMap.sampler, m_cloudCellMap.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet cellWrite{};
    cellWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    cellWrite.dstSet = m_cloudSampleDescSet;
    cellWrite.dstBinding = 7;
    cellWrite.descriptorCount = 1;
    cellWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    cellWrite.pImageInfo = &cellInfo;
    vkUpdateDescriptorSets(m_device, 5, sw, 0, nullptr);
    vkUpdateDescriptorSets(m_device, 1, &cellWrite, 0, nullptr);
    VkDescriptorImageInfo volumeInfo{m_cloudVolumeAtlas.sampler, m_cloudVolumeAtlas.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet volumeWrite = cellWrite;
    volumeWrite.dstBinding = 8;
    volumeWrite.pImageInfo = &volumeInfo;
    vkUpdateDescriptorSets(m_device, 1, &volumeWrite, 0, nullptr);

    // Near-state set: the same textures; params (4), queries (5, unused by
    // mode 7) and results (6) all the RT params buffer. Mode 7 reads the
    // camera / sun and writes nearState.state -- disjoint bytes of one buffer.
    {
        VkWriteDescriptorSet nw[9]{};
        VkDescriptorBufferInfo rtParams{m_cloudRtParamsBuffer.buffer, 0, sizeof(Backend::CloudParamsGPU)};
        for (uint32_t i = 0; i < 4; ++i) {
            nw[i] = sw[i];
            nw[i].dstSet = m_cloudNearDescSet;
        }
        for (uint32_t b = 4; b < 7; ++b) {
            nw[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            nw[b].dstSet = m_cloudNearDescSet;
            nw[b].dstBinding = b;
            nw[b].descriptorCount = 1;
            nw[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            nw[b].pBufferInfo = &rtParams;
        }
        nw[7] = cellWrite;
        nw[7].dstSet = m_cloudNearDescSet;
        nw[8] = volumeWrite;
        nw[8].dstSet = m_cloudNearDescSet;
        vkUpdateDescriptorSets(m_device, 9, nw, 0, nullptr);
    }
    return true;
}

bool VulkanDevice::dispatchCloudNoise(uint32_t mode, uint32_t size, uint32_t seed, uint32_t cells) {
    if (!m_cloudNoisePipeline || !m_cloudNoiseDescSet) return false;
    // Frames of THIS device may be sampling the textures being rewritten. A
    // regeneration is rare (startup, seed/extent change), so a full idle is
    // the simple correct choice.
    waitIdle();
    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudNoisePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudNoisePipelineLayout,
                            0, 1, &m_cloudNoiseDescSet, 0, nullptr);
    const uint32_t pc[4] = {mode, size, seed, cells};
    vkCmdPushConstants(cmd, m_cloudNoisePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    const uint32_t g = (size + 7u) / 8u;
    const uint32_t gz = (mode <= 1u) ? size : 1u;   // 3D modes: one Z slice per group
    vkCmdDispatch(cmd, g, g, gz);
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    endSingleTimeCommands(cmd);
    return true;
}

bool VulkanDevice::generateCloudNoise() {
    if (m_cloudNoiseReady) return true;
    const bool ok =
        dispatchCloudNoise(0u, Backend::kCloudBaseNoiseSize, 0u, 0u) &&
        dispatchCloudNoise(1u, Backend::kCloudDetailNoiseSize, 0u, 0u) &&
        dispatchCloudNoise(2u, Backend::kCloudCurlNoiseSize, 0u, 0u);
    if (ok) {
        m_cloudNoiseReady = true;
        ++m_cloudNoiseGenerations;
    }
    return ok;
}

bool VulkanDevice::generateCloudWeather(const Backend::CloudWeatherGenParams& w) {
    const uint64_t h = Backend::hashCloudWeatherGen(w);
    if (h == m_cloudWeatherHash) return true;
    // The majorant is derived from the map, so it is rebuilt with it.
    if (!dispatchCloudNoise(3u, Backend::kCloudWeatherMapSize, w.seed, w.feature_cells) ||
        !dispatchCloudNoise(4u, Backend::kCloudMajorantSize, 0u, 0u)) return false;
    m_cloudWeatherHash = h;
    ++m_cloudWeatherGenerations;
    return true;
}

bool VulkanDevice::sampleClouds(const Backend::CloudParamsGPU& params, uint32_t mode, uint32_t steps,
                                const std::vector<Backend::CloudQueryGPU>& queries,
                                std::vector<float>& out) {
    out.clear();
    if (!m_cloudSamplePipeline || !m_cloudNoiseReady || m_cloudWeatherHash == 0 || queries.empty())
        return false;
    const uint32_t n = static_cast<uint32_t>(queries.size());
    if (n > m_cloudQueryCapacity) {
        if (m_cloudQueryBuffer.buffer) destroyBuffer(m_cloudQueryBuffer);
        if (m_cloudResultBuffer.buffer) destroyBuffer(m_cloudResultBuffer);
        const uint32_t cap = std::max<uint32_t>(64u, n);
        BufferCreateInfo q{};
        q.size = uint64_t(cap) * sizeof(Backend::CloudQueryGPU);
        q.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        q.location = MemoryLocation::CPU_TO_GPU;
        m_cloudQueryBuffer = createBuffer(q);
        BufferCreateInfo r{};
        r.size = uint64_t(cap) * sizeof(float);
        r.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC;
        r.location = MemoryLocation::GPU_TO_CPU;
        m_cloudResultBuffer = createBuffer(r);
        if (!m_cloudQueryBuffer.buffer || !m_cloudResultBuffer.buffer) {
            m_cloudQueryCapacity = 0;
            return false;
        }
        m_cloudQueryCapacity = cap;
        VkDescriptorBufferInfo qi{m_cloudQueryBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo ri{m_cloudResultBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[2]{};
        for (int i = 0; i < 2; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = m_cloudSampleDescSet;
            w[i].dstBinding = 5u + uint32_t(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = i == 0 ? &qi : &ri;
        }
        // The sample set is used only by this synchronous call, so rewriting
        // it cannot race a frame.
        vkUpdateDescriptorSets(m_device, 2, w, 0, nullptr);
    }
    {
        // Same volume gate as updateCloudRtParams: measure what is drawn.
        Backend::CloudParamsGPU gated = params;
        if (!m_cloudVolumePipeline || m_cloudVolumeSlots == 0) gated.volume[0] = 0.0f;
        uploadBuffer(m_cloudSampleParamsBuffer, &gated, sizeof(gated));
    }
    uploadBuffer(m_cloudQueryBuffer, queries.data(), uint64_t(n) * sizeof(Backend::CloudQueryGPU));

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudSamplePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudSamplePipelineLayout,
                            0, 1, &m_cloudSampleDescSet, 0, nullptr);
    const uint32_t pc[4] = {mode, n, std::max(1u, steps), 0u};
    vkCmdPushConstants(cmd, m_cloudSamplePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cmd, (n + 63u) / 64u, 1, 1);
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    endSingleTimeCommands(cmd);

    out.resize(n);
    downloadBuffer(m_cloudResultBuffer, out.data(), uint64_t(n) * sizeof(float));
    ++m_cloudSampleDispatches;
    return true;
}

// ── RayFusion cloud layer (Faz 3c) ──────────────────────────────────────────
// Set: 0-3 cloud textures, 4 CloudParams (the RT params buffer: one block for
// both renderers), 5 majorant, 6 atmosphere LUTs [transmittance, sky-view],
// 7 frame block, 8 previous history (sampled), 9 current history, 10 output.

void VulkanDevice::destroyCloudRaster() {
    if (m_cloudScenePipeline) vkDestroyPipeline(m_device, m_cloudScenePipeline, nullptr);
    m_cloudScenePipeline = VK_NULL_HANDLE;
    m_cloudSceneSet = VK_NULL_HANDLE;
    m_cloudSceneHdr = m_cloudSceneDepth = VK_NULL_HANDLE;
    m_cloudSceneSampler = VK_NULL_HANDLE;
    for (ImageHandle* img : {&m_cloudHist[0], &m_cloudHist[1], &m_cloudRasterOut, &m_cloudShadowMap,
                             &m_cloudFresh, &m_cloudFreshAux}) {
        if (img->image || img->view) destroyImage(*img);
        *img = {};
    }
    m_cloudFreshW = m_cloudFreshH = 0;
    if (m_cloudResolvePipeline) vkDestroyPipeline(m_device, m_cloudResolvePipeline, nullptr);
    m_cloudResolvePipeline = VK_NULL_HANDLE;
    if (m_cloudShadowPipeline) vkDestroyPipeline(m_device, m_cloudShadowPipeline, nullptr);
    m_cloudShadowPipeline = VK_NULL_HANDLE;
    if (m_cloudRasterFrameBuf.buffer) destroyBuffer(m_cloudRasterFrameBuf);
    m_cloudRasterFrameBuf = {};
    if (m_cloudRasterPipeline) vkDestroyPipeline(m_device, m_cloudRasterPipeline, nullptr);
    if (m_cloudRasterLayout) vkDestroyPipelineLayout(m_device, m_cloudRasterLayout, nullptr);
    if (m_cloudRasterDescLayout) vkDestroyDescriptorSetLayout(m_device, m_cloudRasterDescLayout, nullptr);
    if (m_cloudRasterPool) vkDestroyDescriptorPool(m_device, m_cloudRasterPool, nullptr);
    m_cloudRasterPipeline = VK_NULL_HANDLE;
    m_cloudRasterLayout = VK_NULL_HANDLE;
    m_cloudRasterDescLayout = VK_NULL_HANDLE;
    m_cloudRasterPool = VK_NULL_HANDLE;
    m_cloudRasterSets[0] = m_cloudRasterSets[1] = VK_NULL_HANDLE;
    m_cloudRasterW = m_cloudRasterH = 0;
}

bool VulkanDevice::createCloudRasterPipeline(const std::vector<uint32_t>& spv) {
    if (spv.empty() || !ensureCloudTextures()) return false;
    destroyCloudRaster();
    // 14/15: the interleaved march's fresh grid (cloud_raster.comp writes,
    // cloud_resolve.comp reads).
    // 16: storm cell map (cloud_common.glsl, ATMOSPHERE_STORMS.md).
    // 17: storm cell volume atlas (ATMOSPHERE_CLOUD_VOLUMES.md).
    VkDescriptorSetLayoutBinding b[18]{};
    for (uint32_t i = 0; i < 18; ++i) {
        b[i].binding = i;
        b[i].descriptorCount = (i == 6) ? 2u : 1u;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].descriptorType = (i == 4 || i == 7) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                            : ((i >= 9 && i <= 11) || (i >= 13 && i <= 15)) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 18; li.pBindings = b;
    bool ok = vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_cloudRasterDescLayout) == VK_SUCCESS;
    if (ok) {
        VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * 12},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 2},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 * 6}};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = 3; pi.pPoolSizes = sizes; pi.maxSets = 3;
        ok = vkCreateDescriptorPool(m_device, &pi, nullptr, &m_cloudRasterPool) == VK_SUCCESS;
    }
    if (ok) {
        VkDescriptorSetLayout layouts[2] = {m_cloudRasterDescLayout, m_cloudRasterDescLayout};
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = m_cloudRasterPool; ai.descriptorSetCount = 2; ai.pSetLayouts = layouts;
        ok = vkAllocateDescriptorSets(m_device, &ai, m_cloudRasterSets) == VK_SUCCESS;
    }
    if (ok) {
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &m_cloudRasterDescLayout;
        VkPushConstantRange scenePush{VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &scenePush;
        ok = vkCreatePipelineLayout(m_device, &pl, nullptr, &m_cloudRasterLayout) == VK_SUCCESS;
    }
    ok = ok && makeComputePipeline(m_device, spv, m_cloudRasterLayout, m_cloudRasterPipeline);
    if (ok) {
        BufferCreateInfo ci{};
        ci.size = sizeof(Backend::CloudRasterFrameGPU);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_cloudRasterFrameBuf = createBuffer(ci);
        ok = m_cloudRasterFrameBuf.buffer != VK_NULL_HANDLE;
    }
    if (!ok) destroyCloudRaster();
    return ok;
}

bool VulkanDevice::createCloudScenePipeline(const std::vector<uint32_t>& spv) {
    if (m_cloudScenePipeline) {
        return true;
    }
    if (spv.empty() || !m_cloudRasterLayout) {
        return false;
    }
    if (!makeComputePipeline(m_device, spv, m_cloudRasterLayout, m_cloudScenePipeline)) {
        return false;
    }
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = m_cloudRasterPool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &m_cloudRasterDescLayout;
    if (vkAllocateDescriptorSets(m_device, &allocate, &m_cloudSceneSet) != VK_SUCCESS) {
        vkDestroyPipeline(m_device, m_cloudScenePipeline, nullptr);
        m_cloudScenePipeline = VK_NULL_HANDLE;
        m_cloudSceneSet = VK_NULL_HANDLE;
        return false;
    }
    ImageHandle* textures[6] = {&m_cloudBase, &m_cloudDetail, &m_cloudCurl,
                                &m_cloudWeather, nullptr, &m_cloudMajorant};
    VkDescriptorImageInfo images[6]{};
    VkDescriptorBufferInfo params{m_cloudRtParamsBuffer.buffer, 0, sizeof(Backend::CloudParamsGPU)};
    VkDescriptorBufferInfo frame{m_cloudRasterFrameBuf.buffer, 0, sizeof(Backend::CloudRasterFrameGPU)};
    VkDescriptorImageInfo cellMap{m_cloudCellMap.sampler, m_cloudCellMap.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo volumeAtlas{m_cloudVolumeAtlas.sampler, m_cloudVolumeAtlas.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet writes[9]{};
    uint32_t count = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        if (i == 6) {
            continue;
        }
        auto& write = writes[count++];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_cloudSceneSet;
        write.dstBinding = i;
        write.descriptorCount = 1;
        if (i == 4 || i == 7) {
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = i == 4 ? &params : &frame;
        } else {
            images[i] = {textures[i]->sampler, textures[i]->view, VK_IMAGE_LAYOUT_GENERAL};
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &images[i];
        }
    }
    {
        auto& write = writes[count++];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_cloudSceneSet;
        write.dstBinding = 16;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &cellMap;
    }
    {
        auto& write = writes[count++];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_cloudSceneSet;
        write.dstBinding = 17;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &volumeAtlas;
    }
    vkUpdateDescriptorSets(m_device, count, writes, 0, nullptr);
    writeCloudRasterLutDescriptors();
    return true;
}

void VulkanDevice::invalidateCloudSceneDescriptors() {
    m_cloudSceneHdr = m_cloudSceneDepth = VK_NULL_HANDLE;
    m_cloudSceneSampler = VK_NULL_HANDLE;
}

void VulkanDevice::recordCloudScene(VkCommandBuffer cmd, VkImageView hdr, VkImageView depth,
                                   VkSampler sampler, uint32_t width, uint32_t height,
                                   float depthA, float depthB, float depthC, uint32_t steps) {
    if (!cmd || !m_cloudScenePipeline || !hdr || !depth || !sampler ||
        width == 0 || height == 0) {
        return;
    }
    // Target recreation already waits for viewport work. Cache descriptors
    // so steady-state frames never rewrite a set used by an in-flight frame.
    if (hdr != m_cloudSceneHdr || depth != m_cloudSceneDepth || sampler != m_cloudSceneSampler) {
        VkDescriptorImageInfo images[2] = {
            {sampler, depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
            {VK_NULL_HANDLE, hdr, VK_IMAGE_LAYOUT_GENERAL}
        };
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            auto& write = writes[i];
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = m_cloudSceneSet;
            write.dstBinding = 12 + i;
            write.descriptorCount = 1;
            write.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
                                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pImageInfo = &images[i];
        }
        vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);
        m_cloudSceneHdr = hdr;
        m_cloudSceneDepth = depth;
        m_cloudSceneSampler = sampler;
    }
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
    struct ScenePush {
        uint32_t width, height;
        float depthA, depthB, depthC;
        uint32_t steps;
    } push{width, height, depthA, depthB, depthC, steps};
    static_assert(sizeof(ScenePush) == 24, "cloud_scene push ABI");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudScenePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudRasterLayout,
                            0, 1, &m_cloudSceneSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_cloudRasterLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    vkCmdDispatch(cmd, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
}

// Resolve pass of the interleaved RayFusion clouds (cloud_resolve.comp). Same
// layout and sets as the march; without it the march has no consumer, so
// ensureCloudRasterTargets() refuses to build targets and no clouds draw.
bool VulkanDevice::createCloudResolvePipeline(const std::vector<uint32_t>& spv) {
    if (m_cloudResolvePipeline) return true;
    if (spv.empty() || !m_cloudRasterLayout) return false;
    return makeComputePipeline(m_device, spv, m_cloudRasterLayout, m_cloudResolvePipeline);
}

// Shadow map (binding 11 of the same set): created with its pipeline, fixed
// size, so the binding is written once for both sets.
bool VulkanDevice::createCloudShadowPipeline(const std::vector<uint32_t>& spv) {
    if (spv.empty() || !m_cloudRasterLayout || !m_cloudRasterSets[0]) return false;
    if (!makeComputePipeline(m_device, spv, m_cloudRasterLayout, m_cloudShadowPipeline)) return false;
    {
        VramCategoryScope vram(VramCategory::RenderTarget);
        m_cloudShadowMap = createImage2D(Backend::kCloudShadowMapSize, Backend::kCloudShadowMapSize,
                                         VK_FORMAT_R16G16B16A16_SFLOAT,
                                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    }
    if (m_cloudShadowMap.view) m_cloudShadowMap.sampler = makeClampSampler(m_device);
    if (!m_cloudShadowMap.sampler) {
        vkDestroyPipeline(m_device, m_cloudShadowPipeline, nullptr);
        m_cloudShadowPipeline = VK_NULL_HANDLE;
        if (m_cloudShadowMap.image || m_cloudShadowMap.view) destroyImage(m_cloudShadowMap);
        m_cloudShadowMap = {};
        return false;
    }
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, m_cloudShadowMap.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2]{};
    for (uint32_t s = 0; s < 2; ++s) {
        w[s].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[s].dstSet = m_cloudRasterSets[s];
        w[s].dstBinding = 11;
        w[s].descriptorCount = 1;
        w[s].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[s].pImageInfo = &ii;
    }
    vkUpdateDescriptorSets(m_device, 2, w, 0, nullptr);
    return true;
}

// LUT slots (binding 6). Called with the targets and again whenever the LUT
// images are rebuilt (writeRtAtmosphereDescriptors runs after a device idle,
// so no in-flight frame holds these sets). Before the LUTs exist the weather
// map stands in; the frame block's lutReady = 0 keeps the shader off them.
void VulkanDevice::writeCloudRasterLutDescriptors() {
    if (!m_cloudRasterSets[0] || !m_cloudWeather.view) return;
    VkDescriptorImageInfo ii[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        const ImageHandle& lut = m_lutImages[i == 0 ? 0 : 1];
        const bool have = lut.view && lut.sampler;
        ii[i].sampler = have ? lut.sampler : m_cloudWeather.sampler;
        ii[i].imageView = have ? lut.view : m_cloudWeather.view;
        ii[i].imageLayout = have ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    }
    VkDescriptorSet sets[3] = {m_cloudRasterSets[0], m_cloudRasterSets[1], m_cloudSceneSet};
    VkWriteDescriptorSet w[3]{};
    uint32_t count = 0;
    for (VkDescriptorSet set : sets) {
        if (!set) continue;
        auto& write = w[count++];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = 6;
        write.descriptorCount = 2;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = ii;
    }
    vkUpdateDescriptorSets(m_device, count, w, 0, nullptr);
}

bool VulkanDevice::ensureCloudRasterTargets(uint32_t w, uint32_t h, uint32_t interleave) {
    if (!m_cloudRasterPipeline || !m_cloudResolvePipeline || w == 0 || h == 0) return false;
    interleave = (std::clamp)(interleave, 1u, 4u);
    if (w == m_cloudRasterW && h == m_cloudRasterH && interleave == m_cloudInterleave &&
        m_cloudRasterOut.view && m_cloudFresh.view) return false;
    waitIdle();
    for (ImageHandle* img : {&m_cloudHist[0], &m_cloudHist[1], &m_cloudRasterOut,
                             &m_cloudFresh, &m_cloudFreshAux}) {
        if (img->image || img->view) destroyImage(*img);
        *img = {};
    }
    const uint32_t fw = (w + interleave - 1u) / interleave;
    const uint32_t fh = (h + interleave - 1u) / interleave;
    const VkImageUsageFlags use = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    {
        VramCategoryScope vram(VramCategory::RenderTarget);
        for (ImageHandle* img : {&m_cloudHist[0], &m_cloudHist[1], &m_cloudRasterOut}) {
            *img = createImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT, use);
            if (img->view) img->sampler = makeClampSampler(m_device);
        }
        for (ImageHandle* img : {&m_cloudFresh, &m_cloudFreshAux}) {
            *img = createImage2D(fw, fh, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
        }
    }
    if (!m_cloudHist[0].sampler || !m_cloudHist[1].sampler || !m_cloudRasterOut.sampler ||
        !m_cloudFresh.view || !m_cloudFreshAux.view) {
        m_cloudRasterW = m_cloudRasterH = 0;
        return false;
    }
    m_cloudRasterW = w;
    m_cloudRasterH = h;
    m_cloudInterleave = interleave;
    m_cloudFreshW = fw;
    m_cloudFreshH = fh;
    m_cloudRasterParity = 0;

    const ImageHandle* tex[6] = {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, nullptr, &m_cloudMajorant};
    for (uint32_t s = 0; s < 2; ++s) {
        VkDescriptorImageInfo ti[6]{};
        VkDescriptorImageInfo prev{}, cur{}, out{}, fresh{}, freshAux{}, cellMap{}, volumeAtlas{};
        VkDescriptorBufferInfo params{m_cloudRtParamsBuffer.buffer, 0, sizeof(Backend::CloudParamsGPU)};
        VkDescriptorBufferInfo frame{m_cloudRasterFrameBuf.buffer, 0, sizeof(Backend::CloudRasterFrameGPU)};
        std::vector<VkWriteDescriptorSet> ws;
        auto add = [&](uint32_t binding, VkDescriptorType type, const VkDescriptorImageInfo* ii,
                       const VkDescriptorBufferInfo* bi) {
            VkWriteDescriptorSet x{};
            x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            x.dstSet = m_cloudRasterSets[s];
            x.dstBinding = binding;
            x.descriptorCount = 1;
            x.descriptorType = type;
            x.pImageInfo = ii;
            x.pBufferInfo = bi;
            ws.push_back(x);
        };
        for (uint32_t i = 0; i < 6; ++i) {
            if (i == 4) continue;
            ti[i] = {tex[i]->sampler, tex[i]->view, VK_IMAGE_LAYOUT_GENERAL};
            add(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ti[i], nullptr);
        }
        add(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &params);
        add(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &frame);
        // Set s writes history s and reads the other one.
        prev = {m_cloudHist[1 - s].sampler, m_cloudHist[1 - s].view, VK_IMAGE_LAYOUT_GENERAL};
        cur = {VK_NULL_HANDLE, m_cloudHist[s].view, VK_IMAGE_LAYOUT_GENERAL};
        out = {VK_NULL_HANDLE, m_cloudRasterOut.view, VK_IMAGE_LAYOUT_GENERAL};
        add(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prev, nullptr);
        add(9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &cur, nullptr);
        add(10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &out, nullptr);
        fresh = {VK_NULL_HANDLE, m_cloudFresh.view, VK_IMAGE_LAYOUT_GENERAL};
        freshAux = {VK_NULL_HANDLE, m_cloudFreshAux.view, VK_IMAGE_LAYOUT_GENERAL};
        add(14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fresh, nullptr);
        add(15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &freshAux, nullptr);
        cellMap = {m_cloudCellMap.sampler, m_cloudCellMap.view, VK_IMAGE_LAYOUT_GENERAL};
        add(16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cellMap, nullptr);
        volumeAtlas = {m_cloudVolumeAtlas.sampler, m_cloudVolumeAtlas.view, VK_IMAGE_LAYOUT_GENERAL};
        add(17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &volumeAtlas, nullptr);
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
    }
    writeCloudRasterLutDescriptors();
    return true;
}

void VulkanDevice::recordCloudRaster(VkCommandBuffer cmd, const Backend::CloudRasterFrameGPU& frame) {
    if (!m_cloudRasterPipeline || !m_cloudRasterOut.view || cmd == VK_NULL_HANDLE) return;
    // Last frame's sky pass read the output and last frame's dispatch wrote
    // the histories: both must finish before this frame overwrites them.
    VkMemoryBarrier pre{};
    pre.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    pre.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &pre, 0, nullptr, 0, nullptr);
    vkCmdUpdateBuffer(cmd, m_cloudRasterFrameBuf.buffer, 0, sizeof(frame), &frame);
    VkMemoryBarrier up{};
    up.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    up.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    up.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &up, 0, nullptr, 0, nullptr);
    const uint32_t s = m_cloudRasterParity;
    // Interleave phase: a bijection over the N x N cell per N^2 frames that
    // alternates diagonally (N = 2: (0,0) (1,1) (0,1) (1,0)).
    const uint32_t n = m_cloudInterleave;
    const uint32_t k = m_cloudInterleaveFrame++ % (n * n);
    struct InterleavePush {
        uint32_t n, phaseX, phaseY, freshW, freshH;
    } push{n, k % n, (k / n + k % n) % n, m_cloudFreshW, m_cloudFreshH};
    static_assert(sizeof(InterleavePush) == 20, "cloud_interleave.glsl push ABI");
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudRasterLayout, 0, 1,
                            &m_cloudRasterSets[s], 0, nullptr);
    vkCmdPushConstants(cmd, m_cloudRasterLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    // March: one thread per marched pixel (compacted; see cloud_raster.comp).
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudRasterPipeline);
    vkCmdDispatch(cmd, (m_cloudFreshW + 7u) / 8u, (m_cloudFreshH + 7u) / 8u, 1);
    VkMemoryBarrier marched{};
    marched.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    marched.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    marched.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &marched, 0, nullptr, 0, nullptr);
    // Resolve: every pixel of the target, full-resolution history.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudResolvePipeline);
    vkCmdDispatch(cmd, (m_cloudRasterW + 7u) / 8u, (m_cloudRasterH + 7u) / 8u, 1);
    if (m_cloudShadowPipeline) {
        // Same set (binding 11 + the frame block); independent of the march.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudShadowPipeline);
        vkCmdDispatch(cmd, (Backend::kCloudShadowMapSize + 7u) / 8u,
                      (Backend::kCloudShadowMapSize + 7u) / 8u, 1);
    }
    VkMemoryBarrier post{};
    post.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &post, 0, nullptr, 0, nullptr);
    m_cloudRasterParity = 1u - s;
}

} // namespace VulkanRT
