#include "Backend/AcousticGpuPass.h"
#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <cstring>
#include <fstream>
#include <optional>
#include <utility>

namespace VulkanRT {
struct AcousticGpuPass::Impl {
    struct Slot {
        BufferHandle instances, rays, results;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool recorded = false;
        bool inFlight = false;
        audio::AcousticGpuBatch batch;
    };
    VulkanDevice& owner;
    std::array<Slot, 3> slots{};
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::optional<audio::AcousticGpuBatch> pending;
    std::optional<audio::AcousticGpuCompletion> completed;
    std::string failure;
    bool tracedRecord = false;
    bool tracedCompletion = false;

    explicit Impl(VulkanDevice& device) : owner(device) {}

    ~Impl() {
        // The owning device drains its render submissions before releasing this pass.
        const auto device = owner.getDevice();
        for (auto& slot : slots) {
            if (slot.instances.mappedPtr) {
                owner.unmapBuffer(slot.instances);
            }
            if (slot.rays.mappedPtr) {
                owner.unmapBuffer(slot.rays);
            }
            if (slot.results.mappedPtr) {
                owner.unmapBuffer(slot.results);
            }
            owner.destroyBuffer(slot.instances);
            owner.destroyBuffer(slot.rays);
            owner.destroyBuffer(slot.results);
        }
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

    bool buffer(BufferHandle& target, std::size_t bytes) {
        BufferCreateInfo info;
        info.size = bytes;
        info.usage = BufferUsage::STORAGE;
        // Small coherent mappings avoid staging submissions and invalidate/flush waits.
        info.location = MemoryLocation::CPU_TO_GPU;
        target = owner.createBuffer(info);
        if (!target.buffer) {
            return false;
        }
        target.mappedPtr = owner.mapBuffer(target);
        return target.mappedPtr != nullptr;
    }

    bool initialize(const std::string& directory) {
        if (!failure.empty()) {
            return false;
        }
        if (pipeline) {
            return true;
        }
        failure = "Acoustic GPU pipeline initialization failed";
        SCENE_LOG_INFO("[AcousticGPU] initializing compute pipeline");
        std::ifstream file(directory + "/acoustic_segment.spv", std::ios::binary | std::ios::ate);
        const auto bytes = file ? static_cast<std::streamsize>(file.tellg()) : 0;
        if (bytes <= 0 || bytes % 4 != 0) {
            failure = "acoustic_segment.spv missing or invalid; compile shaders";
            return false;
        }
        std::vector<std::uint32_t> code(static_cast<std::size_t>(bytes) / 4);
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(code.data()), bytes)) {
            return false;
        }
        const auto device = owner.getDevice();
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (unsigned i = 0; i < 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = i == 0 ?
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo descriptorInfo{};
        descriptorInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorInfo.bindingCount = 4;
        descriptorInfo.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &descriptorInfo, nullptr, &layout) != VK_SUCCESS) {
            return false;
        }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
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
        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = static_cast<std::size_t>(bytes);
        shaderInfo.pCode = code.data();
        if (vkCreateShaderModule(device, &shaderInfo, nullptr, &module) != VK_SUCCESS) {
            return false;
        }
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = "main";
        SCENE_LOG_INFO("[AcousticGPU] vkCreateComputePipelines begin");
        const auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
            &pipelineInfo, nullptr, &pipeline);
        SCENE_LOG_INFO("[AcousticGPU] vkCreateComputePipelines result=" +
            std::to_string(static_cast<int>(result)));
        vkDestroyShaderModule(device, module, nullptr);
        if (result != VK_SUCCESS) {
            return false;
        }
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 3},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9}
        };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 3;
        poolInfo.poolSizeCount = 2;
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
                !buffer(slot.instances,
                    audio::kAcousticGpuMaxInstances * sizeof(audio::AcousticGpuInstance)) ||
                !buffer(slot.rays, 4 * sizeof(audio::AcousticGpuRay)) ||
                !buffer(slot.results, 4 * sizeof(audio::AcousticGpuSummary))) {
                return false;
            }
        }
        failure.clear();
        SCENE_LOG_INFO("[AcousticGPU] pipeline and three mapped slots ready");
        return true;
    }
};

AcousticGpuPass::AcousticGpuPass(VulkanDevice& device) : impl_(std::make_unique<Impl>(device)) {}
AcousticGpuPass::~AcousticGpuPass() = default;

bool AcousticGpuPass::enqueue(audio::AcousticGpuBatch batch, const std::string& directory) {
    if (batch.rays.empty() || batch.rays.size() > 4 || batch.ids.size() != batch.rays.size() ||
        batch.instances.empty() || batch.instances.size() > audio::kAcousticGpuMaxInstances) {
        impl_->failure = "Acoustic GPU batch exceeds the source/instance budget";
        return false;
    }
    if (!impl_->initialize(directory)) {
        return false;
    }
    impl_->pending = std::move(batch);
    return true;
}

void AcousticGpuPass::collect(VkFence fence) {
    auto& state = *impl_;
    for (auto& slot : state.slots) {
        if (!slot.inFlight || (fence && slot.fence != fence)) {
            continue;
        }
        const auto result = vkGetFenceStatus(state.owner.getDevice(), slot.fence);
        if (result == VK_NOT_READY) {
            continue;
        }
        if (result == VK_SUCCESS && !slot.batch.ids.empty()) {
            if (!state.tracedCompletion) {
                SCENE_LOG_INFO("[AcousticGPU] first fence completion; copying summaries");
                state.tracedCompletion = true;
            }
            audio::AcousticGpuCompletion output;
            output.ids = std::move(slot.batch.ids);
            output.stamp = std::move(slot.batch.stamp);
            output.summaries.resize(output.ids.size());
            std::memcpy(output.summaries.data(), slot.results.mappedPtr,
                output.summaries.size() * sizeof(audio::AcousticGpuSummary));
            // Only one completed CPU result is retained; stamps control publication.
            state.completed = std::move(output);
        } else if (result != VK_SUCCESS) {
            state.failure = "Acoustic render submission fence failed";
        }
        slot.inFlight = false;
        slot.fence = VK_NULL_HANDLE;
        slot.batch = {};
    }
}

bool AcousticGpuPass::take(audio::AcousticGpuCompletion& result) {
    collect();
    if (!impl_->completed) {
        return false;
    }
    result = std::move(*impl_->completed);
    impl_->completed.reset();
    return true;
}

void AcousticGpuPass::cancel() {
    impl_->pending.reset();
    impl_->completed.reset();
    for (auto& slot : impl_->slots) {
        // GPU storage stays owned until fence completion, even for cancelled work.
        slot.batch.ids.clear();
        slot.batch.stamp.clear();
    }
}

bool AcousticGpuPass::record(VkCommandBuffer cmd, VkFence fence) {
    auto& state = *impl_;
    collect();
    if (!state.pending || !state.pipeline || !state.failure.empty() || !fence ||
        !state.owner.hasTLAS()) {
        return false;
    }
    if (!state.pending->stillCurrent || !state.pending->stillCurrent()) {
        state.pending.reset();
        return false;
    }
    const bool trace = !state.tracedRecord;
    if (trace) {
        SCENE_LOG_INFO("[AcousticGPU] first record: instances=" +
            std::to_string(state.pending->instances.size()) + " sources=" +
            std::to_string(state.pending->rays.size()));
    }
    Impl::Slot* target = nullptr;
    for (auto& slot : state.slots) {
        if (!slot.inFlight && !slot.recorded) {
            target = &slot;
            break;
        }
    }
    if (!target) {
        return false;
    } // Render never waits for acoustic capacity.
    auto& slot = *target;
    slot.batch = std::move(*state.pending);
    state.pending.reset();
    std::memcpy(slot.instances.mappedPtr, slot.batch.instances.data(),
        slot.batch.instances.size() * sizeof(audio::AcousticGpuInstance));
    std::memcpy(slot.rays.mappedPtr, slot.batch.rays.data(),
        slot.batch.rays.size() * sizeof(audio::AcousticGpuRay));
    const auto tlas = state.owner.getTLASHandle();
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlas;
    VkDescriptorBufferInfo buffers[] = {
        {slot.instances.buffer, 0, slot.instances.size},
        {slot.rays.buffer, 0, slot.rays.size},
        {slot.results.buffer, 0, slot.results.size}
    };
    VkWriteDescriptorSet writes[4]{};
    for (unsigned i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = slot.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR :
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (i == 0) {
            writes[i].pNext = &asInfo;
        }
        else writes[i].pBufferInfo = &buffers[i - 1];
    }
    vkUpdateDescriptorSets(state.owner.getDevice(), 4, writes, 0, nullptr);
    if (trace) {
        SCENE_LOG_INFO("[AcousticGPU] descriptors updated; recording barriers/dispatch");
    }
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
        VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipelineLayout,
        0, 1, &slot.set, 0, nullptr);
    const std::uint32_t push[] = {static_cast<std::uint32_t>(slot.batch.rays.size()),
        static_cast<std::uint32_t>(slot.batch.instances.size())};
    vkCmdPushConstants(cmd, state.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
    vkCmdDispatch(cmd, push[0], 1, 1);
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    slot.fence = fence;
    slot.recorded = true;
    if (trace) {
        SCENE_LOG_INFO("[AcousticGPU] first dispatch recorded");
        state.tracedRecord = true;
    }
    return true;
}

bool AcousticGpuPass::hasPending() const {
    return impl_->pending.has_value();
}

void AcousticGpuPass::submitted(VkFence fence, bool success) {
    for (auto& slot : impl_->slots) {
        if (!slot.recorded || slot.fence != fence) {
            continue;
        }
        slot.recorded = false;
        slot.inFlight = success;
        if (!impl_->tracedCompletion) {
            SCENE_LOG_INFO(std::string("[AcousticGPU] submission notified: ") +
                (success ? "success" : "failure"));
        }
        if (!success) {
            slot.batch = {};
            slot.fence = VK_NULL_HANDLE;
            impl_->failure = "Acoustic render submission failed";
        }
    }
}

void AcousticGpuPass::retireFence(VkFence fence) {
    collect(fence);
    for (auto& slot : impl_->slots) {
        if (slot.fence == fence) {
            slot.batch = {};
            slot.fence = VK_NULL_HANDLE;
            slot.inFlight = false;
            slot.recorded = false;
        }
    }
}

const std::string& AcousticGpuPass::error() const {
    return impl_->failure;
}
} // namespace VulkanRT
