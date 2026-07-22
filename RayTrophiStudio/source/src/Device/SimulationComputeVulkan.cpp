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
                                   bool atomicFloatEnabled,
                                   bool shaderFloat64Enabled)
        : m_device(device)
        , m_physDevice(physDevice)
        , m_queue(computeQueue)
        , m_queueFamily(queueFamily)
        , m_has_float_atomics(atomicFloatEnabled)
        , m_has_shader_float64(shaderFloat64Enabled) {
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
        destroyStaging(m_stagingUp);
        destroyStaging(m_stagingDown);
        for (auto& s : m_retiredStaging) destroyStaging(s);
        if (m_copyCmdBuf) vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_copyCmdBuf);
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

        VkMemoryAllocateInfo ai{};
        ai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;

        // Address-capable allocation so vkGetBufferDeviceAddress works for BLAS interop.
        VkMemoryAllocateFlagsInfo allocFlags{};
        if (accelInput) {
            allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            ai.pNext = &allocFlags;
        }

        // Try ReBAR first (device-local + host-visible: shader speed of VRAM,
        // uploads become one memcpy). VRAM pressure can make this allocation
        // fail even when the heap is large — fall back silently.
        VkDeviceMemory mem    = VK_NULL_HANDLE;
        void*          mapped = nullptr;
        const uint32_t barType = findBarMemType(req.memoryTypeBits);
        if (barType != UINT32_MAX) {
            ai.memoryTypeIndex = barType;
            if (vkAllocateMemory(m_device, &ai, nullptr, &mem) == VK_SUCCESS) {
                if (vkMapMemory(m_device, mem, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
                    vkFreeMemory(m_device, mem, nullptr);
                    mem    = VK_NULL_HANDLE;
                    mapped = nullptr;
                }
            }
        }

        if (mem == VK_NULL_HANDLE) {
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
            ai.memoryTypeIndex = memType;
            if (vkAllocateMemory(m_device, &ai, nullptr, &mem) != VK_SUCCESS) {
                vkDestroyBuffer(m_device, buf, nullptr);
                return {};
            }
        }
        vkBindBufferMemory(m_device, buf, mem, 0);

        // One-time diagnostic so the Console shows whether the ReBAR path
        // engaged (BIOS ReBAR off => 256MB BAR heap => rejected by the >=1GB
        // heap check, and uploads silently keep using staging copies).
        static bool s_bar_logged = false;
        if (!s_bar_logged) {
            s_bar_logged = true;
            logSimulationComputeInfo(mapped
                ? "[SimComputeVk] ReBAR active: sim buffers are device-local host-visible (uploads = direct memcpy)."
                : "[SimComputeVk] ReBAR unavailable; uploads use staging copies.");
        }

        uint64_t id = m_nextId++;
        m_buffers[id] = { buf, mem, desc.size_bytes, mapped };
        ComputeBufferHandle h;
        h.id = id;
        return h;
    }

    bool destroyBuffer(ComputeBufferHandle h) override {
        auto it = m_buffers.find(h.id);
        if (it == m_buffers.end()) return false;
        // The recorded-but-unsubmitted command buffer may still reference this
        // VkBuffer (recorded transfers/dispatches can outlive a phase that
        // bailed out before its synchronize). Freeing it now would make the
        // next submit touch a destroyed handle — adaptive domains hit exactly
        // this by resizing buffers between steps. Flush first.
        if (m_recording) synchronize();
        if (it->second.buffer) vkDestroyBuffer(m_device, it->second.buffer, nullptr);
        if (it->second.memory) vkFreeMemory(m_device, it->second.memory, nullptr);
        m_buffers.erase(it);
        return true;
    }

    bool resizeBuffer(ComputeBufferHandle h, std::size_t new_size) override {
        auto it = m_buffers.find(h.id);
        if (it == m_buffers.end() || it->second.size >= new_size) return it != m_buffers.end();
        if (m_recording) synchronize(); // see destroyBuffer — same stale-handle window
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

        // ReBAR buffer: the destination is host-visible VRAM — write it
        // directly, no staging copy / submit / fence. Safe even mid-batch or
        // while recording: every GPU submission in this backend is fence-waited
        // before returning, so there is never in-flight work reading the
        // buffer, and coherent host writes are visible to any later submit.
        if (it->second.mapped) {
            std::memcpy(static_cast<uint8_t*>(it->second.mapped) + dst_offset,
                        data, size_bytes);
            return true;
        }

        // Recorded path: the copy goes INTO the compute command buffer instead
        // of its own submit+fence round-trip. On Windows every
        // vkQueueSubmit+vkWaitForFences costs ~0.3-1ms wall regardless of size
        // (WDDM scheduling latency) — with 3 submits per sim phase that tax,
        // not bandwidth, dominated the transfer timings. Taken whenever a
        // batch is active or recorded work already exists (call order with the
        // recorded stream must be preserved). Host data is captured into
        // staging immediately, so the caller may reuse its buffer right away;
        // the GPU copy lands with the next synchronize().
        if (m_batchActive || m_recording || m_stagingUpCursor > 0) {
            const std::size_t off =
                stagingReserve(m_stagingUp, m_stagingUpCursor, size_bytes, /*for_download=*/false);
            if (off == SIZE_MAX || !ensureRecording()) return false;
            std::memcpy(static_cast<uint8_t*>(m_stagingUp.mapped) + off, data, size_bytes);
            VkBufferCopy region{ off, dst_offset, static_cast<VkDeviceSize>(size_bytes) };
            vkCmdCopyBuffer(m_cmdBuf, m_stagingUp.buf, it->second.buffer, 1, &region);
            if (m_batchActive) m_batchHasUploads = true; // one barrier at endTransferBatch
            else               recordUploadBarrier();
            return true;
        }

        // Cold path (no recorded work pending): immediate copy + fence. The
        // reserved staging region is rolled back after the fence — the wait
        // guarantees it is reusable.
        const std::size_t off =
            stagingReserve(m_stagingUp, m_stagingUpCursor, size_bytes, /*for_download=*/false);
        if (off == SIZE_MAX) return false;
        std::memcpy(static_cast<uint8_t*>(m_stagingUp.mapped) + off, data, size_bytes);
        submitCopyImmediate(m_stagingUp.buf, it->second.buffer, size_bytes, off, dst_offset);
        m_stagingUpCursor = off;
        return true;
    }

    bool downloadBuffer(ComputeBufferHandle h,
                        void* data,
                        std::size_t size_bytes,
                        std::size_t src_offset = 0) const override {
        auto it = m_buffers.find(h.id);
        if (it == m_buffers.end() || !data || size_bytes == 0) return false;
        if (src_offset + size_bytes > it->second.size) return false;

        auto* self = const_cast<VulkanSimulationComputeBackend*>(this);
        if (m_batchActive) {
            // Recorded: barrier (once per batch) + copy go into the compute
            // command buffer, AFTER the already-recorded dispatches; the host
            // memcpy runs after the fence in synchronize(), which
            // endTransferBatch() triggers. `data` is valid when
            // endTransferBatch() returns.
            const std::size_t off =
                self->stagingReserve(self->m_stagingDown, self->m_stagingDownCursor,
                                     size_bytes, /*for_download=*/true);
            if (off == SIZE_MAX || !self->ensureRecording()) return false;
            if (!self->m_batchDownBarrierDone) {
                self->recordDownloadBarrier();
                self->m_batchDownBarrierDone = true;
            }
            VkBufferCopy region{ src_offset, off, static_cast<VkDeviceSize>(size_bytes) };
            vkCmdCopyBuffer(self->m_cmdBuf, it->second.buffer, self->m_stagingDown.buf, 1, &region);
            // Capture the mapping NOW: if a later reserve grows/retires this
            // staging buffer, the retired mapping stays valid until the fence.
            self->m_pendingDownloads.push_back(
                { data, static_cast<const uint8_t*>(self->m_stagingDown.mapped) + off, size_bytes });
            return true;
        }
        const std::size_t off =
            self->stagingReserve(self->m_stagingDown, self->m_stagingDownCursor,
                                 size_bytes, /*for_download=*/true);
        if (off == SIZE_MAX) return false;
        self->submitCopyImmediate(it->second.buffer, m_stagingDown.buf, size_bytes, src_offset, off);
        std::memcpy(data, static_cast<const uint8_t*>(m_stagingDown.mapped) + off, size_bytes);
        self->m_stagingDownCursor = off;
        return true;
    }

    // Transfer batches record their copies into the compute command buffer.
    // An upload-only batch produces NO submission — the copies (and one
    // TRANSFER→COMPUTE barrier) simply precede the dispatches recorded after
    // it, and everything lands in the phase's single synchronize(). A batch
    // containing downloads ends with one submit+fence covering uploads,
    // dispatches AND downloads; the downloaded host pointers are valid when
    // endTransferBatch() returns.
    void beginTransferBatch() override {
        m_batchActive = true;
        m_batchHasUploads = false;
        m_batchDownBarrierDone = false;
    }

    bool endTransferBatch() override {
        m_batchActive = false;
        if (m_batchHasUploads) {
            recordUploadBarrier();
            m_batchHasUploads = false;
        }
        if (!m_pendingDownloads.empty()) {
            synchronize(); // one submit+fence; flushes m_pendingDownloads
        }
        m_batchDownBarrierDone = false;
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

        // Recorded transfers are now complete: flush deferred download
        // memcpys, release retired staging buffers, rewind the cursors.
        for (const auto& pd : m_pendingDownloads)
            std::memcpy(pd.host, pd.src, pd.size);
        m_pendingDownloads.clear();
        for (auto& s : m_retiredStaging) destroyStaging(s);
        m_retiredStaging.clear();
        m_stagingUpCursor = 0;
        m_stagingDownCursor = 0;
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
                                    kernel == "terrain_hydraulic_droplet" ||
                                    kernel == "terrain_fluvial_runoff" ||
                                    kernel == "terrain_fluvial_talus");
        if (needs_atomics && !m_has_float_atomics) return false;

        // The MGPCG dot / device-scalar kernels work in double — running them
        // without the shaderFloat64 device feature is UB. Failing here sends
        // the host cleanly to the CPU PCG fallback.
        if (!m_has_shader_float64 &&
            (kernel == "sim_fluid_cg_dot" ||
             kernel == "sim_fluid_cg_scalar_step" ||
             kernel == "sim_fluid_cg_axpy_dev" ||
             kernel == "sim_fluid_cg_zpby_dev" ||
             kernel == "sim_fluid_cg_jacobi_dot" ||
             kernel == "sim_fluid_cg_spmv_dot" ||
             kernel == "sim_fluid_cg_axpy2_dev")) return false;

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

    // 10 bindings needed by sim_fluid_g2p (9 particle/velocity buffers +
    // fluid_mask for the solid FLIP limiter). Raising this only grows the
    // per-count descriptor layout table.
    // Coupled terrain solvers keep several scalar fields resident at once.
    // Snow uses 14 bindings (base, climate, ping-pong mass/ice/water and trace
    // fields); Vulkan RT-class devices targeted by the renderer expose well
    // above the core-minimum storage-buffer count.
    static constexpr uint32_t MAX_BINDINGS      = 16;
    static constexpr uint32_t MAX_DESC_SETS     = 512;
    static constexpr uint32_t MAX_PUSH_CONSTANT = 128;

    struct Buf {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::size_t    size   = 0;
        void*          mapped = nullptr; // non-null: ReBAR (device-local host-visible) — uploads are direct memcpy
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
    bool     m_has_shader_float64 = false;

    // Persistent staging + copy command buffer (see stagingReserve /
    // submitCopyImmediate). Direction-split: the upload staging prefers
    // write-combined memory (fast streaming host writes, snoop-free GPU DMA
    // reads), the download staging prefers host-cached (readback from WC
    // memory is ~0.3GB/s — the 13x-slower-downloads lesson).
    struct Staging {
        VkBuffer       buf    = VK_NULL_HANDLE;
        VkDeviceMemory mem    = VK_NULL_HANDLE;
        std::size_t    cap    = 0;
        void*          mapped = nullptr;
    };
    Staging         m_stagingUp;
    Staging         m_stagingDown;
    VkCommandBuffer m_copyCmdBuf    = VK_NULL_HANDLE;

    // Transfer batch state (beginTransferBatch/endTransferBatch).
    bool m_batchActive          = false;
    bool m_batchHasUploads      = false; // batch records copies; one barrier at end
    bool m_batchDownBarrierDone = false; // COMPUTE→TRANSFER barrier once per batch
    // Staging cursors persist across batches until synchronize(): recorded
    // copies reference their staging regions until the fence. Reset in
    // synchronize().
    std::size_t m_stagingUpCursor = 0;
    std::size_t m_stagingDownCursor = 0;
    // Downloads recorded into the command buffer; the host memcpy happens
    // after the fence in synchronize(). `src` points into a staging mapping
    // (possibly of a retired buffer — kept alive until that same fence).
    struct PendingDownload {
        void*       host;
        const void* src;
        std::size_t size;
    };
    std::vector<PendingDownload> m_pendingDownloads;
    // Staging buffers replaced (grown) while recorded copies still reference
    // them — destroyed after the next fence.
    std::vector<Staging> m_retiredStaging;

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
            // Normalize shares the scatter pass's 5-buffer dispatch (vel/weight
            // at bindings 3/4 — CUDA index parity). Registering it as 2 bound a
            // 5-binding set on a 2-binding layout (UB): bindings 0/1 hit the
            // particle position/velocity buffers → positions corrupted per step.
            { "sim_fluid_p2g_normalize",            "sim_fluid_p2g_normalize.spv",         5, 36 },
            { "sim_fluid_density_clear",            "sim_fluid_density_clear.spv",         1, 36 },
            { "sim_fluid_density_splat",            "sim_fluid_density_splat.spv",         2, 36 },
            // FluidG2PGpuConstants = 17 fields x 4 = 68 (has use_solid_flip_limiter);
            // buffer 10 = fluid_mask (solid FLIP limiter parity with CUDA).
            { "sim_fluid_g2p",                      "sim_fluid_g2p.spv",                  10, 68 },
            // GridProjectionGpuConstants = 13 fields x 4 = 52. The shaders may
            // declare only the leading fields; the pipeline range must cover the
            // full struct the host pushes (old 36 made vkCmdPushConstants exceed
            // the range → validation error / UB on the gas + SOR paths).
            { "sim_fluid_free_surface_sor",         "sim_fluid_free_surface_sor.spv",      3, 52 },
            { "sim_grid_divergence",                "sim_grid_divergence.spv",             5, 52 },
            { "sim_grid_sor",                       "sim_grid_sor.spv",                    5, 52 },
            { "sim_grid_subtract_gradient",         "sim_grid_subtract_gradient.spv",      5, 52 },
            // MGPCG free-surface pressure solve (Faz 1 Vulkan port of the CUDA
            // sim_fluid_cg_* family — plain path only: non-variational, non-GFM,
            // non-fused reductions, no multigrid; the host's generic path covers
            // exactly this subset on non-CUDA backends).
            { "sim_fluid_divergence",               "sim_fluid_divergence.spv",            5, 52 },
            { "sim_fluid_subtract_gradient",        "sim_fluid_subtract_gradient.spv",     5, 52 },
            { "sim_fluid_cg_build_diag",            "sim_fluid_cg_build_diag.spv",         2, 52 },
            { "sim_fluid_cg_residual_init",         "sim_fluid_cg_residual_init.spv",      4, 52 },
            { "sim_fluid_cg_spmv",                  "sim_fluid_cg_spmv.spv",               4, 52 },
            { "sim_fluid_cg_jacobi",                "sim_fluid_cg_jacobi.spv",             3, 52 },
            { "sim_fluid_cg_copy",                  "sim_fluid_cg_copy.spv",               2, 52 },
            { "sim_fluid_cg_axpy",                  "sim_fluid_cg_axpy.spv",               2, 52 },
            { "sim_fluid_cg_zpby",                  "sim_fluid_cg_zpby.spv",               2, 52 },
            // Double-precision block partials → requires shaderFloat64 (gated in
            // dispatch(); pipeline creation simply fails without the feature).
            { "sim_fluid_cg_dot",                   "sim_fluid_cg_dot.spv",                3, 52 },
            // Device-resident CG scalars (alpha/beta/sigma live on the GPU) —
            // collapses per-dot submit+fence round-trips into one small download
            // every K iterations. All three read/write double scalars →
            // shaderFloat64-gated like sim_fluid_cg_dot.
            { "sim_fluid_cg_scalar_step",           "sim_fluid_cg_scalar_step.spv",        2, 52 },
            { "sim_fluid_cg_axpy_dev",              "sim_fluid_cg_axpy_dev.spv",           3, 52 },
            { "sim_fluid_cg_zpby_dev",              "sim_fluid_cg_zpby_dev.spv",           3, 52 },
            // Fused CG kernels (CUDA name/buffer-order parity, plain branch only;
            // + axpy2_dev, Vulkan-only fused p/r pair update). Cut the device-
            // scalar loop from 9 to 6 dispatches+barriers per iteration.
            { "sim_fluid_cg_jacobi_dot",            "sim_fluid_cg_jacobi_dot.spv",         4, 52 },
            { "sim_fluid_cg_spmv_dot",              "sim_fluid_cg_spmv_dot.spv",           5, 52 },
            { "sim_fluid_cg_axpy2_dev",             "sim_fluid_cg_axpy2_dev.spv",          5, 52 },
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
            // Device-resident particle runoff: height/material plus persistent scalar
            // discharge and XY channel-direction memory.  All share 112-byte constants.
            { "terrain_fluvial_runoff",             "terrain_fluvial_runoff.spv",           9, 112 },
            { "terrain_fluvial_apply",              "terrain_fluvial_apply.spv",            6, 112 },
            { "terrain_fluvial_talus",              "terrain_fluvial_talus.spv",            2, 112 },
            // GPU flow-accumulation for Fluvial (iterative relaxation approximation of
            // the CPU priority-flood + MFD accumulation — see terrain_flow_*.comp
            // headers for the algorithm notes).
            { "terrain_flow_fill",                  "terrain_flow_fill.spv",               3, 16 },
            { "terrain_flow_weights",                "terrain_flow_weights.spv",             2, 12 },
            { "terrain_flow_accumulate",             "terrain_flow_accumulate.spv",          3, 8  },
            // Snow Layer's coupled Jacobi passes. Fourteen persistent scalar
            // fields keep the entire settle/melt/runoff/geometry solve on the
            // device; 88-byte constants select the current pass and physical
            // metre-scale parameters. Gather writes need no float atomics.
            { "terrain_snow_solver",                 "terrain_snow_solver.spv",             14, 88 },
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

    // ReBAR memory type: DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT, but only
    // when its heap is >=1GB — without resizable BAR the device-local
    // host-visible window is just 256MB and it is shared with the graphics
    // backend (this app renders through Vulkan RT), so we must not squat on it.
    // Buffers landing here run at full VRAM speed for shaders while host
    // uploads become a plain memcpy over PCIe (no staging copy, no submit, no
    // fence). Host READS from this memory are write-combined-slow — downloads
    // must keep going through the cached staging path.
    uint32_t findBarMemType(uint32_t typeBits) const {
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(m_physDevice, &memProps);
        constexpr VkMemoryPropertyFlags kBar =
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & kBar) == kBar &&
                memProps.memoryHeaps[memProps.memoryTypes[i].heapIndex].size >=
                    (VkDeviceSize(1) << 30))
                return i;
        }
        return UINT32_MAX;
    }

    void destroyStaging(Staging& s) {
        if (s.buf == VK_NULL_HANDLE) return;
        vkUnmapMemory(m_device, s.mem);
        vkDestroyBuffer(m_device, s.buf, nullptr);
        vkFreeMemory(m_device, s.mem, nullptr);
        s = {};
    }

    bool createStagingBuffer(std::size_t size, bool for_download,
                              VkBuffer& outBuf, VkDeviceMemory& outMem) {
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = size;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(m_device, &bci, nullptr, &outBuf) != VK_SUCCESS) return false;

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(m_device, outBuf, &req);
        // Download staging MUST prefer HOST_CACHED: reading back from
        // write-combined host memory is ~0.3GB/s (measured 13x slower field
        // downloads vs CUDA). Upload staging prefers plain coherent (WC):
        // streaming host writes are fastest there and the GPU's DMA read
        // doesn't have to snoop the CPU cache. Both fall back to whatever
        // coherent host-visible type the driver offers.
        constexpr VkMemoryPropertyFlags kBase =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t mt = UINT32_MAX;
        if (for_download) {
            mt = findMemType(req.memoryTypeBits, kBase | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        } else {
            VkPhysicalDeviceMemoryProperties memProps{};
            vkGetPhysicalDeviceMemoryProperties(m_physDevice, &memProps);
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                const VkMemoryPropertyFlags f = memProps.memoryTypes[i].propertyFlags;
                if ((req.memoryTypeBits & (1u << i)) && (f & kBase) == kBase &&
                    (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) == 0) { mt = i; break; }
            }
        }
        if (mt == UINT32_MAX) mt = findMemType(req.memoryTypeBits, kBase);
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

    // Reserve `size` bytes in the (grow-only, persistently mapped) staging
    // buffer at its cursor; returns the region's offset or SIZE_MAX. If a
    // grow is needed while recorded copies still reference the current buffer
    // (cursor > 0), the old buffer is RETIRED — kept alive until the next
    // fence — instead of destroyed.
    std::size_t stagingReserve(Staging& s, std::size_t& cursor,
                               std::size_t size, bool for_download) {
        const std::size_t aligned = (size + 7u) & ~std::size_t(7);
        if (s.buf == VK_NULL_HANDLE || cursor + aligned > s.cap) {
            const std::size_t cap = std::max<std::size_t>({ aligned, s.cap * 2, 1u << 20 });
            if (s.buf != VK_NULL_HANDLE && cursor > 0) {
                m_retiredStaging.push_back(s);
                s = {};
            } else {
                destroyStaging(s);
            }
            cursor = 0;
            if (!createStagingBuffer(cap, for_download, s.buf, s.mem)) return SIZE_MAX;
            if (vkMapMemory(m_device, s.mem, 0, cap, 0, &s.mapped) != VK_SUCCESS) {
                vkDestroyBuffer(m_device, s.buf, nullptr);
                vkFreeMemory(m_device, s.mem, nullptr);
                s = {};
                return SIZE_MAX;
            }
            s.cap = cap;
        }
        const std::size_t base = cursor;
        cursor += aligned;
        return base;
    }

    // Barriers pairing recorded transfer copies with the surrounding compute
    // dispatches inside the shared command buffer.
    void recordUploadBarrier() {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(m_cmdBuf,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void recordDownloadBarrier() {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(m_cmdBuf,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void submitCopyImmediate(VkBuffer src, VkBuffer dst, std::size_t size,
                              VkDeviceSize srcOff, VkDeviceSize dstOff) {
        // Persistent copy command buffer (separate from the main compute
        // recording), reset per use — the pool is created resettable (the main
        // cmdbuf is already vkResetCommandBuffer'd every synchronize()).
        if (m_copyCmdBuf == VK_NULL_HANDLE) {
            VkCommandBufferAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool        = m_cmdPool;
            ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(m_device, &ai, &m_copyCmdBuf) != VK_SUCCESS) return;
        }
        vkResetCommandBuffer(m_copyCmdBuf, 0);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(m_copyCmdBuf, &bi);

        VkBufferCopy region{ srcOff, dstOff, static_cast<VkDeviceSize>(size) };
        vkCmdCopyBuffer(m_copyCmdBuf, src, dst, 1, &region);
        vkEndCommandBuffer(m_copyCmdBuf);

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &m_copyCmdBuf;
        vkQueueSubmit(m_queue, 1, &si, m_fence);
        vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(m_device, 1, &m_fence);
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
        ctx.shader_atomic_float_enabled,
        ctx.shader_float64_enabled);
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
static std::recursive_mutex s_sharedMeshComputeMutex;

std::recursive_mutex& sharedMeshComputeMutex() {
    return s_sharedMeshComputeMutex;
}

ISimulationComputeBackend* acquireSharedMeshComputeBackend() {
    std::lock_guard<std::recursive_mutex> lock(s_sharedMeshComputeMutex);
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
    std::lock_guard<std::recursive_mutex> lock(s_sharedMeshComputeMutex);
    s_sharedMeshBackend.reset();
    s_sharedMeshBackendDevice = nullptr;
}

} // namespace RayTrophiSim
