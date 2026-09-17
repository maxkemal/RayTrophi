#include "Api/RtApi.h"
#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"

#include <memory>

// Same file-scope rule as RtApi.cpp: declared OUTSIDE rtapi, or a second,
// never-defined symbol is created and the mistake surfaces at link time.
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;

namespace rtapi {
namespace {

GpuMemoryDevice describeBackend(Backend::IBackend& backend, const char* role) {
    GpuMemoryDevice out;
    out.role = role;
    auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(&backend);
    if (!vk) {
        // OptiX/CPU: nothing allocated through VulkanDevice. Its bytes show up
        // only in the process-wide untracked remainder.
        out.backend = backend.getInfo().name;
        return out;
    }
    out.backend = "vulkan";
    VulkanRT::VulkanDevice* dev = vk->getVulkanDevice();
    if (!dev || !dev->isInitialized()) return out;
    out.tracked = true;
    const VulkanRT::VramAllocationReport r = dev->vramAllocationReport();
    out.device_local_bytes = r.deviceLocalBytes;
    out.host_bytes = r.hostBytes;
    out.allocations = r.allocations;
    for (size_t i = 0; i < static_cast<size_t>(VulkanRT::VramCategory::Count); ++i) {
        GpuMemoryCategory c;
        c.name = VulkanRT::vramCategoryName(static_cast<VulkanRT::VramCategory>(i));
        c.device_local_bytes = r.categories[i].deviceLocalBytes;
        c.host_bytes = r.categories[i].hostBytes;
        c.allocations = r.categories[i].allocations;
        out.categories.push_back(std::move(c));
    }
    const VulkanRT::BlasCompactionStats cs = dev->blasCompactionStats();
    out.compaction_supported = cs.supported;
    out.compacted = cs.compacted;
    out.compaction_skipped_skinned = cs.skippedSkinned;
    out.compaction_failures = cs.failures;
    out.compaction_bytes_before = cs.bytesBefore;
    out.compaction_bytes_after = cs.bytesAfter;
    return out;
}

} // namespace

GpuMemoryReport gpuMemoryReport() {
    GpuMemoryReport out;
    out.blas_compaction_enabled = VulkanRT::VulkanDevice::blasCompactionEnabled();
    Backend::IBackend* render = g_backend.get();
    Backend::IBackend* viewport = g_viewport_backend.get();
    if (render) out.devices.push_back(describeBackend(*render, "render"));
    // The render backend doubles as the viewport when no dedicated one exists;
    // listing it twice would count its bytes twice.
    if (viewport && viewport != render) out.devices.push_back(describeBackend(*viewport, "viewport"));

    for (const GpuMemoryDevice& d : out.devices) out.tracked_device_local_bytes += d.device_local_bytes;

    for (Backend::IBackend* b : { render, viewport }) {
        auto* vk = b ? dynamic_cast<Backend::VulkanBackendAdapter*>(b) : nullptr;
        VulkanRT::VulkanDevice* dev = vk ? vk->getVulkanDevice() : nullptr;
        if (dev && dev->queryDeviceLocalMemory(out.vram_usage_bytes, out.vram_budget_bytes)) {
            out.vram_measured = true;
            break;
        }
    }
    if (out.vram_measured) {
        out.untracked_bytes = static_cast<int64_t>(out.vram_usage_bytes) -
                              static_cast<int64_t>(out.tracked_device_local_bytes);
    }
    return out;
}

Result setBlasCompaction(bool enabled) {
    VulkanRT::VulkanDevice::setBlasCompactionEnabled(enabled);
    return Result::success();
}

} // namespace rtapi
