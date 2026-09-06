#include "Viewport/RasterInstanceUpload.h"

#include "Backend/VulkanBackend.h"

#include <algorithm>
#include <cstring>

namespace Backend {

struct RasterGlobalInstanceBuffer::Impl {
    VulkanRT::VulkanDevice* device = nullptr;
    VulkanRT::BufferHandle  buffer;
    uint32_t                capacity = 0;   // in instances
    void*                   mapped   = nullptr;
};

RasterGlobalInstanceBuffer::RasterGlobalInstanceBuffer()
    : impl_(std::make_unique<Impl>()) {}

RasterGlobalInstanceBuffer::~RasterGlobalInstanceBuffer() {
    destroy();
}

bool RasterGlobalInstanceBuffer::ensure(VulkanRT::VulkanDevice& device,
                                        uint32_t instanceCount) {
    if (instanceCount == 0) return true;

    // Already big enough — no-op.
    if (impl_->device == &device &&
        impl_->buffer.buffer &&
        impl_->capacity >= instanceCount) {
        return true;
    }

    // Geometric growth: max(requested, oldCapacity * 1.5) — same pattern as RT
    // instance source buffers. Never shrink.
    const uint32_t newCapacity = (std::max)(
        instanceCount,
        static_cast<uint32_t>(impl_->capacity + impl_->capacity / 2u));

    const VkDeviceSize requiredBytes =
        static_cast<VkDeviceSize>(newCapacity) * kInstanceStride;

    // Tear down old buffer first.
    if (impl_->buffer.buffer) {
        if (impl_->mapped) {
            device.unmapBuffer(impl_->buffer);
            impl_->mapped = nullptr;
        }
        device.destroyBuffer(impl_->buffer);
        impl_->buffer = VulkanRT::BufferHandle{};
    }

    VulkanRT::BufferCreateInfo ci{};
    ci.size        = requiredBytes;
    // ★ STORAGE: GPU culling compute shader'i (raster_cull.comp) bu buffer'i
    //   kaynak matris dizisi olarak OKUR. Bu biti dusurursen descriptor
    //   yazimi gecersiz olur ve culling sessizce kurulamaz.
    ci.usage       = VulkanRT::BufferUsage::VERTEX |
                     VulkanRT::BufferUsage::STORAGE |
                     VulkanRT::BufferUsage::TRANSFER_DST;
    // ★ CPU_TO_GPU = HOST_VISIBLE | HOST_COHERENT: the whole point is to write
    // directly through a persistently mapped pointer and avoid staging +
    // blocking-fence transfers that the old per-mesh path paid for.
    ci.location    = VulkanRT::MemoryLocation::CPU_TO_GPU;
    ci.initialData = nullptr;

    impl_->buffer = device.createBuffer(ci);
    if (!impl_->buffer.buffer) {
        impl_->capacity = 0;
        impl_->device   = nullptr;
        return false;
    }

    impl_->mapped = device.mapBuffer(impl_->buffer);
    if (!impl_->mapped) {
        device.destroyBuffer(impl_->buffer);
        impl_->buffer   = VulkanRT::BufferHandle{};
        impl_->capacity = 0;
        impl_->device   = nullptr;
        return false;
    }

    // Zero-fill so unused slots render degenerate (zero-scale) geometry.
    std::memset(impl_->mapped, 0, requiredBytes);

    impl_->capacity = newCapacity;
    impl_->device   = &device;
    return true;
}

void RasterGlobalInstanceBuffer::write(uint32_t instanceOffset,
                                       const float* matrices4x4,
                                       uint32_t count) {
    if (!impl_->mapped || count == 0) return;
    if (instanceOffset + count > impl_->capacity) return;

    void* dst = static_cast<uint8_t*>(impl_->mapped) +
                static_cast<size_t>(instanceOffset) * kInstanceStride;
    std::memcpy(dst, matrices4x4,
                static_cast<size_t>(count) * kInstanceStride);
}

void RasterGlobalInstanceBuffer::zero(uint32_t instanceOffset,
                                      uint32_t count) {
    if (!impl_->mapped || count == 0) return;
    if (instanceOffset + count > impl_->capacity) return;

    void* dst = static_cast<uint8_t*>(impl_->mapped) +
                static_cast<size_t>(instanceOffset) * kInstanceStride;
    std::memset(dst, 0, static_cast<size_t>(count) * kInstanceStride);
}

void* RasterGlobalInstanceBuffer::vkBuffer() const {
    return impl_->buffer.buffer;
}

uint32_t RasterGlobalInstanceBuffer::capacity() const {
    return impl_->capacity;
}

bool RasterGlobalInstanceBuffer::isReady() const {
    return impl_->mapped != nullptr && impl_->buffer.buffer != nullptr;
}

void RasterGlobalInstanceBuffer::destroy() {
    if (!impl_) return;
    if (impl_->buffer.buffer && impl_->device) {
        if (impl_->mapped) {
            impl_->device->unmapBuffer(impl_->buffer);
            impl_->mapped = nullptr;
        }
        impl_->device->destroyBuffer(impl_->buffer);
        impl_->buffer = VulkanRT::BufferHandle{};
    }
    impl_->capacity = 0;
    impl_->device   = nullptr;
}

} // namespace Backend
