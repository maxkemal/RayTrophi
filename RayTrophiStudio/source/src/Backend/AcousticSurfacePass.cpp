#include "Backend/AcousticSurfacePass.h"
#include "Backend/VulkanBackend.h"
#include <cstring>
#include <fstream>

namespace VulkanRT {
struct AcousticSurfacePass::Impl {
    struct Slot {
        BufferHandle instances, sources;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::shared_ptr<const std::vector<audio::AcousticSurfaceInstance>> uploaded;
    };
    static_assert(VulkanDevice::kFrameSlotCount == 2, "Surface upload frame slots");
    VulkanDevice& owner;
    std::array<Slot, 2> slots{};
    audio::AcousticSurfaceBatch batch;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::string failure;
    bool enabled = false;
    bool ready = false;
    explicit Impl(VulkanDevice& device) : owner(device) {}
    ~Impl() {
        for (auto& slot : slots) {
            for (auto* buffer : {&slot.instances, &slot.sources}) {
                if (buffer->mappedPtr) {
                    owner.unmapBuffer(*buffer);
                }
                owner.destroyBuffer(*buffer);
            }
        }
        const auto device = owner.getDevice();
        if (pool) {
            vkDestroyDescriptorPool(device, pool, nullptr);
        }
        if (pipeline) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        if (pipelineLayout) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        }
        if (layout) {
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
        }
    }
    bool buffer(BufferHandle& target, std::size_t size) {
        BufferCreateInfo info;
        info.size = size;
        info.usage = BufferUsage::STORAGE;
        info.location = MemoryLocation::CPU_TO_GPU;
        target = owner.createBuffer(info);
        if (!target.buffer) {
            return false;
        }
        target.mappedPtr = owner.mapBuffer(target);
        return target.mappedPtr != nullptr;
    }
    bool initialize(const std::string& directory) {
        if (ready) {
            return true;
        }
        if (!failure.empty()) {
            return false;
        }
        failure = "Acoustic surface pipeline unavailable; rebuild acoustic_surface.spv";
        std::ifstream file(directory + "/acoustic_surface.spv", std::ios::binary | std::ios::ate);
        const auto size = file ? static_cast<std::streamsize>(file.tellg()) : 0;
        if (size <= 0 || size % 4 != 0) {
            return false;
        }
        std::vector<std::uint32_t> code(static_cast<std::size_t>(size) / 4);
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(code.data()), size)) {
            return false;
        }
        const auto device = owner.getDevice();
        VkDescriptorSetLayoutBinding bindings[6]{};
        for (unsigned i = 0; i < 6; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR :
                i < 3 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        }
        VkDescriptorSetLayoutCreateInfo descriptors{};
        descriptors.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptors.bindingCount = 6;
        descriptors.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &descriptors, nullptr, &layout) != VK_SUCCESS) {
            return false;
        }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &layout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            return false;
        }
        VkShaderModule module = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo shader{};
        shader.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader.codeSize = static_cast<std::size_t>(size);
        shader.pCode = code.data();
        if (vkCreateShaderModule(device, &shader, nullptr, &module) != VK_SUCCESS) {
            return false;
        }
        VkComputePipelineCreateInfo compute{};
        compute.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        compute.layout = pipelineLayout;
        compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        compute.stage.module = module;
        compute.stage.pName = "main";
        const auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
            &compute, nullptr, &pipeline);
        vkDestroyShaderModule(device, module, nullptr);
        if (result != VK_SUCCESS) {
            return false;
        }
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6}};
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
            return false;
        }
        for (auto& slot : slots) {
            VkDescriptorSetAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocation.descriptorPool = pool;
            allocation.descriptorSetCount = 1;
            allocation.pSetLayouts = &layout;
            if (vkAllocateDescriptorSets(device, &allocation, &slot.set) != VK_SUCCESS ||
                !buffer(slot.instances, audio::kAcousticGpuMaxInstances *
                    sizeof(audio::AcousticSurfaceInstance)) ||
                !buffer(slot.sources, 4 * 16 * sizeof(float))) return false;
        }
        failure.clear();
        ready = true;
        return true;
    }
};

AcousticSurfacePass::AcousticSurfacePass(VulkanDevice& device)
    : impl_(std::make_unique<Impl>(device)) {}
AcousticSurfacePass::~AcousticSurfacePass() = default;

bool AcousticSurfacePass::update(audio::AcousticSurfaceBatch batch, const std::string& directory) {
    auto& state = *impl_;
    if (!state.initialize(directory)) {
        return false;
    }
    const bool changed = !state.enabled || state.batch.stamp != batch.stamp;
    state.batch = std::move(batch);
    state.enabled = true;
    return changed;
}
void AcousticSurfacePass::disable() {
    impl_->enabled = false;
}
const std::string& AcousticSurfacePass::error() const {
    return impl_->batch.error.empty() ? impl_->failure : impl_->batch.error;
}

bool AcousticSurfacePass::record(VkCommandBuffer cmd, unsigned index, VkImageView position,
    VkImageView normal, VkImageView output, unsigned width, unsigned height,
    float exposure, float overlay) {
    auto& state = *impl_;
    if (!state.enabled || !state.pipeline || index >= state.slots.size() ||
        !position || !normal || !output || !state.owner.hasTLAS()) return false;
    auto& slot = state.slots[index];
    const auto& batch = state.batch;
    const bool valid = batch.error.empty() && batch.instances &&
        !batch.instances->empty() && batch.instances->size() <= audio::kAcousticGpuMaxInstances &&
        batch.sourceCount <= 4 && batch.stillCurrent && batch.stillCurrent();
    state.failure = valid ? "" : batch.error.empty() ? "Surface metadata pending/stale" : batch.error;
    if (valid && slot.uploaded != batch.instances) {
        std::memcpy(slot.instances.mappedPtr, batch.instances->data(),
            batch.instances->size() * sizeof(audio::AcousticSurfaceInstance));
        slot.uploaded = batch.instances;
    }
    float sources[4][16]{};
    for (unsigned i = 0; i < batch.sourceCount && i < 4; ++i) {
        std::memcpy(sources[i], batch.sources[i].data(), sizeof(audio::AcousticRenderSource));
    }
    std::memcpy(slot.sources.mappedPtr, sources, sizeof(sources));
    const auto tlas = state.owner.getTLASHandle();
    VkWriteDescriptorSetAccelerationStructureKHR as{};
    as.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as.accelerationStructureCount = 1;
    as.pAccelerationStructures = &tlas;
    VkDescriptorBufferInfo buffers[] = {{slot.instances.buffer, 0, slot.instances.size},
        {slot.sources.buffer, 0, slot.sources.size}};
    VkDescriptorImageInfo images[3]{};
    const VkImageView views[] = {position, normal, output};
    for (unsigned i = 0; i < 3; ++i) {
        images[i].imageView = views[i];
        images[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkWriteDescriptorSet writes[6]{};
    for (unsigned i = 0; i < 6; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = slot.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR :
            i < 3 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (i == 0) {
            writes[i].pNext = &as;
        }
        else if (i < 3) writes[i].pBufferInfo = &buffers[i - 1];
        else writes[i].pImageInfo = &images[i - 3];
    }
    vkUpdateDescriptorSets(state.owner.getDevice(), 6, writes, 0, nullptr);
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT |
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_HOST_BIT |
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    struct Push {
        unsigned width, height, sources, instances;
        int band;
        unsigned reflections;
        float exposure, overlay;
    } push{width, height, valid ? batch.sourceCount : 0,
        valid ? static_cast<unsigned>(batch.instances->size()) : 0,
        batch.band, batch.reflectionSamples, exposure, overlay};
    static_assert(sizeof(Push) == 32, "Acoustic surface push ABI");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipelineLayout,
        0, 1, &slot.set, 0, nullptr);
    vkCmdPushConstants(cmd, state.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, &push);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
    return true;
}
} // namespace VulkanRT
