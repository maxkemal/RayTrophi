#pragma once

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include "Backend/VulkanBackend.h"
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>
#ifdef _WIN32
#include <windows.h>
// This header is transitively included by every Player viewport translation unit.
// Also remove macros from any Windows header included earlier by the caller.
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif
#include "Paint/HiddenGameObjects.h"

namespace Backend {

// Player presentation input only: no scene objects, materials or Triangle facades.
// Buffer slots follow the raster command buffers, whose reuse fence has already
// been waited by beginFrame. Synchronous submission uses one additional slot.
class PlayerDepthCard {
public:
    struct Push {
        float viewProj[16];
        float right[4];
        float up[4];
        float anchor[4];
        float image[4];
    };
    static_assert(sizeof(Push) == 128);

    void reset() {
        if (device_) {
            for (auto& slot : slots_) {
                if (slot.second.buffer.mappedPtr) device_->unmapBuffer(slot.second.buffer);
                device_->destroyBuffer(slot.second.buffer);
            }
            if (pipeline_) vkDestroyPipeline(device_->getDevice(), pipeline_, nullptr);
            if (layout_) vkDestroyPipelineLayout(device_->getDevice(), layout_, nullptr);
        }
        slots_.clear();
        pipeline_ = VK_NULL_HANDLE;
        layout_ = VK_NULL_HANDLE;
        buffer_ = VK_NULL_HANDLE;
        device_ = nullptr;
        data_.clear();
        shown_ = {};
        ready_ = false;
        closeMapping();
    }

    bool prepare(VulkanRT::VulkanDevice& device, VkRenderPass pass,
                 const std::string& shaderDir, VkCommandBuffer slot, bool enabled) {
        ready_ = false;
#ifdef _WIN32
        if (!enabled) {
            closeMapping();
            shown_ = {};
            data_.clear();
            return false;
        }
        const auto openNow = GetTickCount64();
        if (!pixels_ && openNow >= next_open_ms_) {
            next_open_ms_ = openNow + kOpenRetryMs;
            openMapping();
        }
        if (!pixels_) {
            return false;
        }
        const auto locked = WaitForSingleObject(lock_, 0);
        if (locked == WAIT_OBJECT_0 || locked == WAIT_ABANDONED) {
            Header header{};
            if (locked == WAIT_OBJECT_0) {
                std::memcpy(&header, pixels_, sizeof(header));
            }
            const auto now = GetTickCount64();
            const bool valid = isValidHeader(header, now);
            if (valid && shown_.serial != header.serial) {
                const auto bytes = static_cast<size_t>(header.width) * header.height * 8;
                data_.resize(bytes);
                std::memcpy(data_.data(), pixels_ + sizeof(Header), bytes);
                shown_ = header;
            }
            ReleaseMutex(lock_);
            if (!valid) {
                shown_ = {};
                data_.clear();
                return false;
            }
        } else if (locked == WAIT_TIMEOUT) {
            const auto now = GetTickCount64();
            if (!isValidHeader(shown_, now) || data_.empty()) {
                shown_ = {};
                data_.clear();
                return false;
            }
        } else {
            return false;
        }
        device_ = &device;
        if (!pipeline_ && !createPipeline(pass, shaderDir)) {
            return false;
        }
        auto& entry = slots_[slot];
        auto& buffer = entry.buffer;
        if (!buffer.buffer) {
            VulkanRT::BufferCreateInfo info{};
            info.size = 640 * 640 * 8;
            info.usage = VulkanRT::BufferUsage::VERTEX;
            info.location = VulkanRT::MemoryLocation::CPU_TO_GPU;
            buffer = device.createBuffer(info);
        }
        if (!buffer.buffer) {
            return false;
        }
        // CPU_TO_GPU is HOST_VISIBLE | HOST_COHERENT; no transfer queue or wait.
        if (!buffer.mappedPtr) {
            buffer.mappedPtr = device.mapBuffer(buffer);
        }
        void* mapped = buffer.mappedPtr;
        if (!mapped) {
            return false;
        }
        if (entry.serial != shown_.serial) {
            std::memcpy(mapped, data_.data(), data_.size());
            entry.serial = shown_.serial;
        }
        buffer_ = buffer.buffer;
        ready_ = true;
#endif
        return ready_;
    }

    void draw(VkCommandBuffer cmd, const float* viewProj, const float* view) const {
        if (!ready_) {
            return;
        }
        const Vec3 anchor = Paint::constrainHiddenPlayerAnchor(
            "depth_card", Vec3(shown_.x, 0.0f, shown_.z), 0.25f, shown_.physicalHeight);
        Push push{};
        std::memcpy(push.viewProj, viewProj, sizeof(push.viewProj));
        for (int i = 0; i < 3; ++i) {
            push.right[i] = view[i * 4];
            push.up[i] = view[i * 4 + 1];
        }
        push.anchor[0] = anchor.x;
        push.anchor[1] = anchor.y;
        push.anchor[2] = anchor.z;
        push.anchor[3] = shown_.physicalHeight;
        push.image[0] = static_cast<float>(shown_.width);
        push.image[1] = static_cast<float>(shown_.height);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &buffer_, &offset);
        vkCmdDraw(cmd, 6, shown_.width * shown_.height, 0, 0);
    }

private:
    struct Header {
        uint64_t serial, received;
        float x, z, physicalHeight;
        uint32_t width, height, magic;
    };
    static_assert(sizeof(Header) == 40);
    static constexpr uint32_t kMagic = 0x31445452; // RTD1
    static constexpr size_t kBytes = sizeof(Header) + 640 * 640 * 8;
    static constexpr uint64_t kExpiryMs = 300;
    static constexpr uint64_t kOpenRetryMs = 200;

    static bool isValidHeader(const Header& header, uint64_t now) {
        return header.magic == kMagic && header.received <= now &&
            now - header.received <= kExpiryMs && header.width > 0 && header.height > 0 &&
            header.width <= 640 && header.height <= 640 &&
            std::isfinite(header.x) && std::abs(header.x) <= 10000.0f &&
            std::isfinite(header.z) && std::abs(header.z) <= 10000.0f &&
            header.physicalHeight >= 0.3f && header.physicalHeight <= 3.2f;
    }

    bool createPipeline(VkRenderPass pass, const std::string& directory) {
        const VkDevice device = device_->getDevice();
        auto load = [&](const char* name) {
            std::ifstream file(directory + "/" + name, std::ios::binary | std::ios::ate);
            if (!file) return VkShaderModule(VK_NULL_HANDLE);
            const std::streamsize size = file.tellg();
            if (size < 20 || size % 4 != 0) return VkShaderModule(VK_NULL_HANDLE);
            std::vector<uint32_t> bytes(static_cast<size_t>(size) / 4);
            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
                return VkShaderModule(VK_NULL_HANDLE);
            }
            VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            info.codeSize = static_cast<size_t>(size);
            info.pCode = bytes.data();
            VkShaderModule module = VK_NULL_HANDLE;
            vkCreateShaderModule(device, &info, nullptr, &module);
            return module;
        };
        VkShaderModule vert = load("player_depth_card.spv");
        VkShaderModule frag = load("player_depth_card_frag.spv");
        if (!vert || !frag) {
            if (vert) vkDestroyShaderModule(device, vert, nullptr);
            if (frag) vkDestroyShaderModule(device, frag, nullptr);
            return false;
        }
        VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo l{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        l.pushConstantRangeCount = 1;
        l.pPushConstantRanges = &range;
        if (!layout_) vkCreatePipelineLayout(device, &l, nullptr, &layout_);
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        for (int i = 0; i < 2; ++i) {
            stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[i].stage = i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[i].module = i == 0 ? vert : frag;
            stages[i].pName = "main";
        }
        VkVertexInputBindingDescription binding{0, 8, VK_VERTEX_INPUT_RATE_INSTANCE};
        VkVertexInputAttributeDescription attributes[2] = {
            {0, 0, VK_FORMAT_R32_UINT, 0}, {1, 0, VK_FORMAT_R32_SFLOAT, 4}};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = attributes;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendAttachmentState attachment{};
        attachment.colorWriteMask = 15;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &attachment;
        VkDynamicState states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = states;
        VkGraphicsPipelineCreateInfo p{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        p.stageCount = 2;
        p.pStages = stages.data();
        p.pVertexInputState = &vi;
        p.pInputAssemblyState = &ia;
        p.pViewportState = &vp;
        p.pRasterizationState = &rs;
        p.pMultisampleState = &ms;
        p.pDepthStencilState = &ds;
        p.pColorBlendState = &blend;
        p.pDynamicState = &dynamic;
        p.layout = layout_;
        p.renderPass = pass;
        const auto result = layout_ ? vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
            &p, nullptr, &pipeline_) : VK_ERROR_INITIALIZATION_FAILED;
        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
        return result == VK_SUCCESS;
    }

    void openMapping() {
#ifdef _WIN32
        lock_ = OpenMutexW(SYNCHRONIZE | MUTEX_MODIFY_STATE, FALSE,
                          L"Local\\RayTrophiDepthCard.v1.lock");
        if (!lock_) return;
        mapping_ = OpenFileMappingW(FILE_MAP_READ, FALSE, L"Local\\RayTrophiDepthCard.v1");
        if (mapping_) {
            pixels_ = static_cast<const unsigned char*>(
                MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, kBytes));
        }
        if (!pixels_) closeMapping();
#endif
    }

    void closeMapping() {
#ifdef _WIN32
        if (pixels_) UnmapViewOfFile(pixels_);
        if (mapping_) CloseHandle(mapping_);
        if (lock_) CloseHandle(lock_);
        pixels_ = nullptr;
        mapping_ = lock_ = nullptr;
#endif
    }

    VulkanRT::VulkanDevice* device_ = nullptr;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    struct Slot {
        VulkanRT::BufferHandle buffer;
        uint64_t serial = 0;
    };
    std::map<VkCommandBuffer, Slot> slots_;
    std::vector<unsigned char> data_;
    Header shown_{};
    bool ready_ = false;
#ifdef _WIN32
    HANDLE mapping_ = nullptr, lock_ = nullptr;
    const unsigned char* pixels_ = nullptr;
    uint64_t next_open_ms_ = 0;
#endif
};

} // namespace Backend
