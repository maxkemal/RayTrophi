#pragma once

// Global raster instance buffer — single CPU_TO_GPU buffer for all instance
// matrices across all raster meshes.
//
// The old raster path maintained per-mesh instance buffers, each requiring
// drainInteractiveViewportInFlight() + create/destroy cycles whenever the
// visibility set changed. This module replaces that with the proven RT pattern:
//
//   1. Single buffer, geometric growth (1.5×), CPU_TO_GPU persistently mapped
//   2. Meshes get contiguous ranges via firstInstance offset
//   3. Transform updates write directly to mapped memory (no staging)
//   4. Layout rebuild only on topology change (instance add/remove)
//
// ★ Synchronisation rule: the caller must ensure no GPU reader overlaps the
// written range.  For the raster frame ring this means either:
//   - writing between waitAll() and beginFrame() (build/topology change), or
//   - writing BEFORE beginFrame() records draws referencing that range.
// A static camera / static scene needs ZERO writes and ZERO drain.

#include <cstdint>
#include <memory>

namespace VulkanRT { class VulkanDevice; }

namespace Backend {

class RasterGlobalInstanceBuffer final {
public:
    static constexpr uint32_t kInstanceStride = 64; // sizeof(float[16])

    RasterGlobalInstanceBuffer();
    ~RasterGlobalInstanceBuffer();

    RasterGlobalInstanceBuffer(const RasterGlobalInstanceBuffer&) = delete;
    RasterGlobalInstanceBuffer& operator=(const RasterGlobalInstanceBuffer&) = delete;

    // Ensure capacity for at least `instanceCount` instances.  Buffer grows
    // geometrically (1.5×) and never shrinks — a subsequent smaller ensure is a
    // no-op.  Returns false if the Vulkan allocation fails (caller should fall
    // back to per-mesh buffers).
    //
    // ★ When the buffer grows, existing mapped data is INVALID.  The caller
    // must re-upload all instance matrices after a grow.
    bool ensure(VulkanRT::VulkanDevice& device, uint32_t instanceCount);

    // Direct memcpy into the persistently mapped buffer.  No Vulkan API call,
    // no drain, no staging.  `offset` and `count` are in INSTANCES, not bytes.
    // Matrices are float[16] column-major (matches existing RasterInstanceGPU).
    void write(uint32_t instanceOffset, const float* matrices4x4, uint32_t count);

    // Zero-fill a range.  Used for removed/hidden instances whose slot is
    // retained to keep layout stable.
    void zero(uint32_t instanceOffset, uint32_t count);

    // The underlying VkBuffer (as void* to avoid vulkan.h include here).
    // Cast to VkBuffer at the call site.
    void* vkBuffer() const;
    uint32_t capacity() const;
    bool isReady() const;

    // Release the Vulkan buffer.  Must be called before device teardown.
    void destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Backend
