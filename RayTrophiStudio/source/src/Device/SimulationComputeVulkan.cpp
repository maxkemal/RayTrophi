/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          SimulationComputeVulkan.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Vulkan compute backend for the simulation pipeline.
 * Implements ISimulationComputeBackend using Vulkan compute shaders (GLSL
 * compiled to SPIR-V). Serves systems without CUDA (AMD / Intel / ARM GPUs).
 *
 * Buffer management:  VkBuffer + VkDeviceMemory (manual, no VMA).
 *   Device-local buffers for simulation data.
 *   HOST_VISIBLE staging buffers created on demand for upload/download.
 *
 * Dispatch model: Commands are recorded into a single command buffer.
 *   dispatch() records; synchronize() submits + waits then resets.
 *
 * Descriptor strategy: One descriptor set layout per buffer count (1..9).
 *   A large descriptor pool is reset each synchronize() call.
 *   Each dispatch() allocates one set from the pool, writes the buffers,
 *   then binds + dispatches.
 *
 * Float atomics (P2G scatter / density splat): Requires
 *   VK_EXT_shader_atomic_float with shaderBufferFloat32AtomicAdd.
 *   Checked at construction; has_float_atomics_ flag set accordingly.
 *   Kernels that need atomics return false (CPU fallback) when unavailable.
 */

#include "SimulationCompute.h"
#include "SimulationComputeVulkanContext.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace RayTrophiSim {

// ── SPIR-V loading ────────────────────────────────────────────────────────────

static std::vector<uint32_t> loadSPV(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return {};
    std::streamsize size = f.tellg();
    if (size <= 0 || (size % 4) != 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> spv(static_cast<std::size_t>(size) / 4);
    f.read(reinterpret_cast<char*>(spv.data()), size);
    return spv;
}

static std::string findShaderDir() {
    for (const char* dir : {"shaders", "source/shaders", "../shaders",
                             "RayTrophiStudio/source/shaders"}) {
        if (std::filesystem::exists(std::string(dir) + "/skinning.spv"))
            return dir;
    }
    return "shaders";
}

// ── Backend ───────────────────────────────────────────────────────────────────

class VulkanSimulationComputeBackend final : public ISimulationComputeBackend {
public:
    VulkanSimulationComputeBackend(VkDevice device,
                                   VkPhysicalDevice physDevice,
                                   VkQueue computeQueue,
                                   uint32_t queueFamily,
                                   bool atomicFloatEnabled)
        : m_device(device)
        , m_physDevice(physDevice)
        , m_queue(computeQueue)
        , m_queueFamily(queueFamily)
        , m_has_float_atomics(atomicFloatEnabled) {
        if (!m_device || !m_physDevice || !m_queue) return;
        if (!initCommandPool())     return;
        if (!initDescriptorPool())  return;
        if (!initDescLayouts())     return;
        loadPipelines();
        m_ready = true;
    }

    ~VulkanSimulationComputeBackend() override {
        if (!m_device) return;
        if (m_queue) vkQueueWaitIdle(m_queue);

        for (auto& kv : m_pipelines) {
            vkDestroyPipeline(m_device, kv.second.pipeline, nullptr);
            vkDestroyPipelineLayout(m_device, kv.second.layout, nullptr);
        }
        for (auto& dl : m_descLayouts)
            if (dl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_device, dl, nullptr);
        if (m_descPool)  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
        if (m_cmdBuf)    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_cmdBuf);
        if (m_cmdPool)   vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
        if (m_fence)     vkDestroyFence(m_device, m_fence, nullptr);

        for (auto& kv : m_buffers) {
            if (kv.second.buffer) vkDestroyBuffer(m_device, kv.second.buffer, nullptr);
            if (kv.second.memory) vkFreeMemory(m_device, kv.second.memory, nullptr);
        }
    }

    // ── Identity ─────────────────────────────────────────────────────────────

    ComputeBackendType type() const override { return ComputeBackendType::VulkanCompute; }
    const char* name() const override        { return "Vulkan Simulation Compute"; }

    ComputeBackendCaps caps() const override {
        ComputeBackendCaps c;
        c.available                      = m_ready;
        c.supports_async                 = false;
        c.supports_shared_graphics_interop = false;
        c.max_storage_buffer_bytes       = static_cast<std::size_t>(2) * 1024 * 1024 * 1024;
        c.max_threads_per_group          = 256;
        return c;
    }

    bool supportsDispatch() const override { return m_ready; }

    // ── Buffer management ─────────────────────────────────────────────────────

    ComputeBufferHandle createBuffer(const ComputeBufferDesc& desc) override {
        if (!m_device || desc.size_bytes == 0) return {};

        // Phase 3d: an AccelInput buffer doubles as RT BLAS build input, so it needs the
        // accel-input + vertex + device-address usages and an address-capable allocation.
        const bool accelInput = hasComputeBufferUsage(desc.usage, ComputeBufferUsage::AccelInput);

        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = desc.size_bytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (accelInput) {
            bci.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        }
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buf = VK_NULL_HANDLE;
        if (vkCreateBuffer(m_device, &bci, nullptr, &buf) != VK_SUCCESS) return {};

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(m_device, buf, &req);

        uint32_t memType = findMemType(req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memType == UINT32_MAX) {
            // Fallback: host-visible (integrated GPU / APU)
            memType = findMemType(req.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        if (memType == UINT32_MAX) {
            vkDestroyBuffer(m_device, buf, nullptr);
            return {};
        }

        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = memType;

        // Address-capable allocation so vkGetBufferDeviceAddress works for BLAS interop.
        VkMemoryAllocateFlagsInfo allocFlags{};
        if (accelInput) {
            allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            ai.pNext = &allocFlags;
        }

        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (vkAllocateMemory(m_device, &ai, nullptr, &mem) != VK_SUCCESS) {
            vkDestroyBuffer(m_device, buf, nullptr);
            return {};
        }
        vkBindBufferMemory(m_device, buf, mem, 0);

        uint64_t id = m_nextId++;
        m_buffers[id] = { buf, mem, desc.size_bytes };
        ComputeBufferHandle h;
        h.id = id;
        return h;
    }

    bool destroyBuffer(ComputeBufferHandle h) override {
        auto it = m_buffers.find(h.id);
        if (it == m_buffers.end()) return false;
        if (it->second.buffer) vkDestroyBuffer(m_device, it->second.buffer, nullptr);
        if (it->second.memory) vkFreeMemory(m_device, it->second.memory, nullptr);
        m_buffers.erase(it);
        return true;
    }

    bool resizeBuffer(ComputeBufferHandle h, std::size_t new_size) override {
        auto it = m_buffers.find(h.id);
        if (it == m_buffers.end() || it->second.size >= new_size) return it != m_buffers.end();
        if (it->second.buffer) vkDestroyBuffer(m_device, it->second.buffer, nullptr);
        if (it->second.memory) vkFreeMemory(m_device, it->second.memory, nullptr);
        it->second = {};

        ComputeBufferDesc d; d.size_bytes = new_size;
        ComputeBufferHandle nh = createBuffer(d);
        if (!nh.valid()) return false;
        auto src = m_buffers.find(nh.id);
        it->second = src->second;
        m_buffers.erase(src);
        return true;
    }

    std::size_t getBufferSize(ComputeBufferHandle h) const override {
        auto it = m_buffers.find(h.id);
        return (it != m_buffers.end()) ? it->second.size : 0;
    }

    void* nativeBufferPtr(ComputeBufferHandle h) const override {
        auto it = m_buffers.find(h.id);
        if (it == m_buffers.end()) return nullptr;
        return reinterpret_cast<void*>(it->second.buffer);
    }

    uint64_t bufferDeviceAddress(ComputeBufferHandle h) const override {
        auto it = m_buffers.find(h.id);
        if (it == m_buffers.end() || !it->second.buffer) return 0;
        VkBufferDeviceAddressInfo info{};
        info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.buffer = it->second.buffer;
        return static_cast<uint64_t>(vkGetBufferDeviceAddress(m_device, &info));
    }

    // ── Upload / Download ─────────────────────────────────────────────────────

    bool uploadBuffer(ComputeBufferHandle h,
                      const void* data,
                      std::size_t size_bytes,
                      std::size_t dst_offset = 0) override {
        auto it = m_buffers.find(h.id);
        if (it == m_buffers.end() || !data || size_bytes == 0) return false;
        if (dst_offset + size_bytes > it->second.size) return false;

        // Create staging buffer, map, copy, submit transfer, destroy staging.
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        if (!createStagingBuffer(size_bytes, staging, stagingMem)) return false;

        void* mapped = nullptr;
        vkMapMemory(m_device, stagingMem, 0, size_bytes, 0, &mapped);
        std::memcpy(mapped, data, size_bytes);
        vkUnmapMemory(m_device, stagingMem);

        submitCopyImmediate(staging, it->second.buffer, size_bytes, 0, dst_offset);

        vkDestroyBuffer(m_device, staging, nullptr);
        vkFreeMemory(m_device, stagingMem, nullptr);
        return true;
    }

    bool downloadBuffer(ComputeBufferHandle h,
                        void* data,
                        std::size_t size_bytes,
                        std::size_t src_offset = 0) const override {
        auto it = m_buffers.find(h.id);
        if (it == m_buffers.end() || !data || size_bytes == 0) return false;
        if (src_offset + size_bytes > it->second.size) return false;

        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        if (!const_cast<VulkanSimulationComputeBackend*>(this)
                ->createStagingBuffer(size_bytes, staging, stagingMem))
            return false;

        const_cast<VulkanSimulationComputeBackend*>(this)
            ->submitCopyImmediate(it->second.buffer, staging, size_bytes, src_offset, 0);

        void* mapped = nullptr;
        vkMapMemory(m_device, stagingMem, 0, size_bytes, 0, &mapped);
        std::memcpy(data, mapped, size_bytes);
        vkUnmapMemory(m_device, stagingMem);

        vkDestroyBuffer(m_device, staging, nullptr);
        vkFreeMemory(m_device, stagingMem, nullptr);
        return true;
    }

    // ── Frame lifecycle ───────────────────────────────────────────────────────

    void beginFrame(uint64_t /*frame_index*/) override {}
    void endFrame() override {}

    void synchronize() override {
        if (!m_recording) return;
        vkEndCommandBuffer(m_cmdBuf);

        VkSubmitInfo si{};
        si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount   = 1;
        si.pCommandBuffers      = &m_cmdBuf;
        vkQueueSubmit(m_queue, 1, &si, m_fence);
        vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(m_device, 1, &m_fence);

        vkResetCommandBuffer(m_cmdBuf, 0);
        vkResetDescriptorPool(m_device, m_descPool, 0);
        m_recording = false;
    }

    // ── Dispatch ──────────────────────────────────────────────────────────────

    bool dispatch(const ComputeDispatch& cmd) override {
        if (!m_ready || !cmd.constants || cmd.constants_size == 0) return false;
        if (cmd.buffer_count > MAX_BINDINGS) return false;

        // Kernels using float atomics (P2G scatter, density splat) need hardware support.
        const std::string kernel(cmd.kernel ? cmd.kernel : "");
        const bool needs_atomics = (kernel == "sim_fluid_p2g_scatter" ||
                                    kernel == "sim_fluid_density_splat" ||
                                    kernel == "terrain_thermal" ||
                                    kernel == "terrain_thermal_hardness" ||
                                    kernel == "terrain_stream_power" ||
                                    kernel == "terrain_wind" ||
                                    kernel == "terrain_hydraulic_droplet");
        if (needs_atomics && !m_has_float_atomics) return false;

        auto pit = m_pipelines.find(kernel);
        if (pit == m_pipelines.end()) return false;
        const PipelineEntry& pe = pit->second;

        // Ensure command buffer is recording.
        if (!ensureRecording()) return false;

        // Allocate and write descriptor set.
        VkDescriptorSetLayout layout = m_descLayouts[cmd.buffer_count];
        if (layout == VK_NULL_HANDLE) return false;

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = m_descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &layout;

        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(m_device, &dsai, &ds) != VK_SUCCESS) return false;

        std::array<VkDescriptorBufferInfo, MAX_BINDINGS> bufInfos{};
        std::array<VkWriteDescriptorSet,   MAX_BINDINGS> writes{};

        for (uint32_t b = 0; b < cmd.buffer_count; ++b) {
            auto bit = m_buffers.find(cmd.buffers[b].id);
            if (bit == m_buffers.end() || !bit->second.buffer) return false;
            bufInfos[b].buffer = bit->second.buffer;
            bufInfos[b].offset = 0;
            bufInfos[b].range  = VK_WHOLE_SIZE;

            writes[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet          = ds;
            writes[b].dstBinding      = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo     = &bufInfos[b];
        }
        vkUpdateDescriptorSets(m_device, cmd.buffer_count, writes.data(), 0, nullptr);

        // Record pipeline bind + descriptor bind + push constants + dispatch.
        vkCmdBindPipeline(m_cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pe.pipeline);
        vkCmdBindDescriptorSets(m_cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pe.layout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(m_cmdBuf, pe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, static_cast<uint32_t>(cmd.constants_size), cmd.constants);

        uint32_t gx = cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1;
        uint32_t gy = cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1;
        uint32_t gz = cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1;
        vkCmdDispatch(m_cmdBuf, gx, gy, gz);

        // Pipeline barrier: SHADER_WRITE → SHADER_READ so the next kernel
        // sees the updated data.
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(m_cmdBuf,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        return true;
    }

private:
    // ── Internal types ────────────────────────────────────────────────────────

    static constexpr uint32_t MAX_BINDINGS      = 9;
    static constexpr uint32_t MAX_DESC_SETS     = 512;
    static constexpr uint32_t MAX_PUSH_CONSTANT = 128;

    struct Buf {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::size_t    size   = 0;
    };

    struct PipelineEntry {
        VkPipeline       pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout   = VK_NULL_HANDLE;
    };

    VkDevice         m_device     = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VkQueue          m_queue      = VK_NULL_HANDLE;
    uint32_t         m_queueFamily = 0;

    VkCommandPool    m_cmdPool  = VK_NULL_HANDLE;
    VkCommandBuffer  m_cmdBuf   = VK_NULL_HANDLE;
    VkFence          m_fence    = VK_NULL_HANDLE;
    VkDescriptorPool m_descPool = VK_NULL_HANDLE;

    // One desc layout per buffer count (index = count, 0 unused).
    std::array<VkDescriptorSetLayout, MAX_BINDINGS + 1> m_descLayouts{};

    std::unordered_map<std::string, PipelineEntry> m_pipelines;
    std::unordered_map<uint64_t, Buf>              m_buffers;

    uint64_t m_nextId         = 1;
    bool     m_ready          = false;
    bool     m_recording      = false;
    bool     m_has_float_atomics = false;

    // ── Initialization ────────────────────────────────────────────────────────

    bool initCommandPool() {
        VkCommandPoolCreateInfo pi{};
        pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.queueFamilyIndex = m_queueFamily;
        pi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(m_device, &pi, nullptr, &m_cmdPool) != VK_SUCCESS)
            return false;

        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = m_cmdPool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_device, &ai, &m_cmdBuf) != VK_SUCCESS)
            return false;

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        return vkCreateFence(m_device, &fi, nullptr, &m_fence) == VK_SUCCESS;
    }

    bool initDescriptorPool() {
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = MAX_DESC_SETS * MAX_BINDINGS;

        VkDescriptorPoolCreateInfo pi{};
        pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets       = MAX_DESC_SETS;
        pi.poolSizeCount = 1;
        pi.pPoolSizes    = &ps;
        pi.flags         = 0; // will be reset via vkResetDescriptorPool
        return vkCreateDescriptorPool(m_device, &pi, nullptr, &m_descPool) == VK_SUCCESS;
    }

    bool initDescLayouts() {
        m_descLayouts[0] = VK_NULL_HANDLE;
        for (uint32_t n = 1; n <= MAX_BINDINGS; ++n) {
            std::vector<VkDescriptorSetLayoutBinding> bindings(n);
            for (uint32_t b = 0; b < n; ++b) {
                bindings[b].binding         = b;
                bindings[b].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[b].descriptorCount = 1;
                bindings[b].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = n;
            ci.pBindings    = bindings.data();
            if (vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_descLayouts[n])
                    != VK_SUCCESS) {
                m_descLayouts[n] = VK_NULL_HANDLE;
                return false;
            }
        }
        return true;
    }

    // ── Pipeline creation ─────────────────────────────────────────────────────

    bool createPipeline(const std::string& kernelName,
                        const std::string& spvPath,
                        uint32_t bufferCount,
                        uint32_t pushConstantSize) {
        auto spv = loadSPV(spvPath);
        if (spv.empty()) return false;
        if (bufferCount == 0 || bufferCount > MAX_BINDINGS) return false;

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size() * sizeof(uint32_t);
        smci.pCode    = spv.data();
        VkShaderModule mod = VK_NULL_HANDLE;
        if (vkCreateShaderModule(m_device, &smci, nullptr, &mod) != VK_SUCCESS)
            return false;

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = pushConstantSize;

        VkDescriptorSetLayout dl = m_descLayouts[bufferCount];
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(m_device, &plci, nullptr, &pipelineLayout) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, mod, nullptr);
            return false;
        }

        VkComputePipelineCreateInfo cpci{};
        cpci.sType          = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.layout         = pipelineLayout;
        cpci.stage.sType    = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage    = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module   = mod;
        cpci.stage.pName    = "main";

        VkPipeline pipeline = VK_NULL_HANDLE;
        bool ok = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpci,
                                           nullptr, &pipeline) == VK_SUCCESS;
        vkDestroyShaderModule(m_device, mod, nullptr);
        if (!ok) {
            vkDestroyPipelineLayout(m_device, pipelineLayout, nullptr);
            return false;
        }

        m_pipelines[kernelName] = { pipeline, pipelineLayout };
        return true;
    }

    void loadPipelines() {
        const std::string sd = findShaderDir();
        // {kernel_name, spv_file, buffer_count, push_constant_size_bytes}
        // push_constant sizes must match CUDA-side struct sizes exactly.
        struct KernelDef { const char* name; const char* spv; uint32_t bufs; uint32_t pc; };
        static const KernelDef defs[] = {
            // name                             spv file                                    bufs  pc_bytes
            // name                                  spv                                bufs  pc_bytes
            // pc_bytes must match the host-side GpuConstants struct sizes exactly.
            // FluidP2GGpuConstants             = 9 fields × 4 bytes = 36
            // FluidG2PGpuConstants             = 16 fields × 4 bytes = 64
            //   (+affine_damping, +max_affine — APIC clamp parity with CPU)
            // GridProjectionGpuConstants       = 9 fields × 4 bytes = 36
            // FluidDensitySplatGpuConstants    = 9 fields × 4 bytes = 36
            // FluidParticleIntegrateGpuConstants = 9 fields × 4 bytes = 36
            // GridScalarAdvectionGpuConstants  = 5 fields × 4 bytes = 20
            // GridVelocityDissipationGpuConstants = 5 fields × 4 bytes = 20
            // FluidP2GGpuConstants = 36 bytes (nx,ny,nz,particle_count,component,orig_x,orig_y,orig_z,voxel_size)
            // FluidDensitySplatGpuConstants = 36 bytes (nx,ny,nz,particle_count,orig_x,orig_y,orig_z,voxel_size,particle_density)
            // FluidParticleIntegrateGpuConstants = 36 bytes (9 fields × 4)
            { "sim_fluid_clear_float",              "sim_fluid_clear_float.spv",           1, 36 },
            { "sim_fluid_particle_integrate_forces","sim_fluid_particle_forces.spv",       1, 36 },
            { "sim_fluid_p2g_scatter",              "sim_fluid_p2g_scatter.spv",           5, 36 },
            { "sim_fluid_p2g_normalize",            "sim_fluid_p2g_normalize.spv",         2, 36 },
            { "sim_fluid_density_clear",            "sim_fluid_density_clear.spv",         1, 36 },
            { "sim_fluid_density_splat",            "sim_fluid_density_splat.spv",         2, 36 },
            { "sim_fluid_g2p",                      "sim_fluid_g2p.spv",                   9, 64 },
            { "sim_fluid_free_surface_sor",         "sim_fluid_free_surface_sor.spv",      3, 36 },
            { "sim_grid_divergence",                "sim_grid_divergence.spv",             5, 36 },
            { "sim_grid_sor",                       "sim_grid_sor.spv",                    5, 36 },
            { "sim_grid_subtract_gradient",         "sim_grid_subtract_gradient.spv",      5, 36 },
            { "sim_grid_advect_scalar",             "sim_grid_advect_scalar.spv",          5, 20 },
            { "sim_grid_advect_velocity",           "sim_grid_advect_velocity.spv",        6, 20 },
            { "sim_grid_velocity_dissipate_clamp",  "sim_grid_velocity_dissipate.spv",     3, 20 },
            // Mesh subdivision (linear 1->4). 2 SSBO (in/out tris) + 16B push-const.
            { "subdivide_linear",                   "subdivide_linear.spv",                2, 16 },
            // Catmull-Clark GPU refine (Phase 3b): per-level sparse stencil apply (5 SSBO:
            // inPos/outPos/off/idx/w) then Newell face normals (4 SSBO) gathered per
            // vertex atomic-free (4 SSBO). All take a 16B push-const (one uint count).
            { "cc_stencil_apply",                   "cc_stencil_apply.spv",                5, 16 },
            { "cc_face_normals",                    "cc_face_normals.spv",                 4, 16 },
            { "cc_vertex_normals",                  "cc_vertex_normals.spv",               4, 16 },
            // CC device-resident expand (Phase 3d): indexed refine -> non-indexed combined
            // BLAS-layout buffer (6 SSBO: pos/nrm/triIdx/triUV/triMat/out, 16B push-const).
            { "cc_expand_blas",                     "cc_expand_blas.spv",                  6, 16 },
            // Terrain erosion (Vulkan port of erosion_kernels.cu). Push-constant sizes
            // must match TerrainPhysics::*ParamsGPU (minus pointer fields, which become
            // SSBO bindings instead). 1:1 numerical port; see
            // project_terrain_erosion_gpu_migration memory for follow-up physics fixes.
            // ThermalErosionParamsGPU  (no ptrs) = 6 fields x 4 bytes = 24
            // PostProcessParamsGPU     (no ptrs) = 7 fields x 4 bytes = 28
            // StreamPowerParamsGPU     (no ptrs) = 9 fields x 4 bytes = 36
            { "terrain_thermal",                    "terrain_thermal.spv",                 1, 24 },
            { "terrain_thermal_hardness",           "terrain_thermal_hardness.spv",        2, 24 },
            { "terrain_stream_power",               "terrain_stream_power.spv",            5, 36 },
            { "terrain_apply_stream_power",         "terrain_apply_stream_power.spv",      2, 36 },
            { "terrain_pit_fill",                   "terrain_pit_fill.spv",                1, 28 },
            { "terrain_spike_removal",              "terrain_spike_removal.spv",           1, 28 },
            { "terrain_edge_preservation",          "terrain_edge_preservation.spv",       2, 28 },
            { "terrain_smooth",                     "terrain_smooth.spv",                  1, 8  },
            // Shallow-water "virtual pipes" hydraulic model (revived dead CUDA code —
            // never had a working host caller — replaces the droplet Monte-Carlo model).
            // FluvialErosionParamsGPU (no ptrs) = 11 fields x 4 bytes = 44.
            { "terrain_pipe_rain",                  "terrain_pipe_rain.spv",               1, 16 },
            { "terrain_pipe_flux",                  "terrain_pipe_flux.spv",               3, 44 },
            { "terrain_pipe_water",                 "terrain_pipe_water.spv",              3, 44 },
            { "terrain_pipe_erosion",               "terrain_pipe_erosion.spv",             4, 44 },
            // WindErosionParamsGPU (no ptrs, +iterationSeed) = 10 fields x 4 bytes = 40.
            { "terrain_wind",                       "terrain_wind.spv",                    1, 40 },
            // Monte-Carlo droplet hydraulic erosion (replaces the pipe-model attempt for
            // the "Hydraulic" node — user found droplet's organic channel character
            // clearly better). 18 fields x 4 bytes = 72.
            { "terrain_hydraulic_droplet",          "terrain_hydraulic_droplet.spv",        3, 72 },
            // GPU flow-accumulation for Fluvial (iterative relaxation approximation of
            // the CPU priority-flood + MFD accumulation — see terrain_flow_*.comp
            // headers for the algorithm notes).
            { "terrain_flow_fill",                  "terrain_flow_fill.spv",               3, 16 },
            { "terrain_flow_weights",                "terrain_flow_weights.spv",             2, 12 },
            { "terrain_flow_accumulate",             "terrain_flow_accumulate.spv",          3, 8  },
        };
        for (const auto& d : defs)
            createPipeline(d.name, sd + "/" + d.spv, d.bufs, d.pc);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    uint32_t findMemType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(m_physDevice, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        return UINT32_MAX;
    }

    bool createStagingBuffer(std::size_t size,
                              VkBuffer& outBuf, VkDeviceMemory& outMem) {
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = size;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(m_device, &bci, nullptr, &outBuf) != VK_SUCCESS) return false;

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(m_device, outBuf, &req);
        uint32_t mt = findMemType(req.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (mt == UINT32_MAX) { vkDestroyBuffer(m_device, outBuf, nullptr); return false; }

        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = mt;
        if (vkAllocateMemory(m_device, &ai, nullptr, &outMem) != VK_SUCCESS) {
            vkDestroyBuffer(m_device, outBuf, nullptr);
            return false;
        }
        vkBindBufferMemory(m_device, outBuf, outMem, 0);
        return true;
    }

    void submitCopyImmediate(VkBuffer src, VkBuffer dst, std::size_t size,
                              VkDeviceSize srcOff, VkDeviceSize dstOff) {
        // Use a separate one-shot command buffer so uploads/downloads don't
        // interleave with the main compute recording.
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = m_cmdPool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(m_device, &ai, &cb);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);

        VkBufferCopy region{ srcOff, dstOff, static_cast<VkDeviceSize>(size) };
        vkCmdCopyBuffer(cb, src, dst, 1, &region);
        vkEndCommandBuffer(cb);

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vkQueueSubmit(m_queue, 1, &si, m_fence);
        vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(m_device, 1, &m_fence);
        vkFreeCommandBuffers(m_device, m_cmdPool, 1, &cb);
    }

    bool ensureRecording() {
        if (m_recording) return true;
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_cmdBuf, &bi) != VK_SUCCESS) return false;
        m_recording = true;
        return true;
    }
};

// ── Factory ───────────────────────────────────────────────────────────────────

std::unique_ptr<ISimulationComputeBackend>
createVulkanSimulationComputeBackend(const SimulationComputeVulkanContext& ctx) {
    if (!ctx.device || !ctx.physical_device || !ctx.compute_queue) return nullptr;
    auto backend = std::make_unique<VulkanSimulationComputeBackend>(
        static_cast<VkDevice>(ctx.device),
        static_cast<VkPhysicalDevice>(ctx.physical_device),
        static_cast<VkQueue>(ctx.compute_queue),
        ctx.queue_family_index,
        ctx.shader_atomic_float_enabled);
    if (!backend->supportsDispatch()) return nullptr;
    return backend;
}

} // namespace RayTrophiSim

// The Vulkan compute context filled in by VulkanBackend init (globals.h). Declared
// here directly to avoid pulling the heavy globals.h into this device TU.
extern RayTrophiSim::SimulationComputeVulkanContext g_vulkan_sim_compute_ctx;

namespace RayTrophiSim {

// Lazily-created, process-wide compute backend shared by non-simulation GPU work
// (mesh subdivision, etc.), independent of whether a GPU sim domain is active.
// Returns nullptr when no Vulkan compute device is available (caller falls back to
// CPU). The backend is created once on first use and reused; it is NOT torn down
// (matches the lifetime of the Vulkan device, which outlives all mesh ops).
static std::unique_ptr<ISimulationComputeBackend> s_sharedMeshBackend;
static void* s_sharedMeshBackendDevice = nullptr;

ISimulationComputeBackend* acquireSharedMeshComputeBackend() {
    void* curDev = g_vulkan_sim_compute_ctx.device;
    // Backend switch (e.g. Vulkan RT -> OptiX -> Vulkan RT): the VkDevice our cached
    // backend was built on was destroyed and a new one created. releaseSharedMeshCompute-
    // Backend() should have run during the old device's teardown (s_backend then null); if
    // some path skipped it, LEAK the stale backend — its VkDevice is gone, so running its
    // destructor would dereference freed handles (the createBuffer crash) — and rebuild.
    if (s_sharedMeshBackend && curDev && curDev != s_sharedMeshBackendDevice) {
        s_sharedMeshBackend.release();   // intentional: stale device, cannot safely destruct
        s_sharedMeshBackendDevice = nullptr;
    }
    // Retry creation until the Vulkan device exists (subdivision may run before the
    // render backend finishes filling g_vulkan_sim_compute_ctx); once created, reuse.
    if (!s_sharedMeshBackend && curDev) {
        s_sharedMeshBackend = createVulkanSimulationComputeBackend(g_vulkan_sim_compute_ctx);
        s_sharedMeshBackendDevice = curDev;
    }
    return (s_sharedMeshBackend && s_sharedMeshBackend->supportsDispatch()) ? s_sharedMeshBackend.get() : nullptr;
}

// Tear down the shared mesh compute backend. MUST be called while its VkDevice is still
// valid (i.e. before vkDestroyDevice) so its buffers/pipelines are freed cleanly; a later
// acquire then rebuilds against whatever device is current.
void releaseSharedMeshComputeBackend() {
    s_sharedMeshBackend.reset();
    s_sharedMeshBackendDevice = nullptr;
}

} // namespace RayTrophiSim
