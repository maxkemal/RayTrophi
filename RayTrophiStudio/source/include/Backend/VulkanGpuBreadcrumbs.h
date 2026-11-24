#pragma once

// GPU breadcrumbs: "how far did the GPU get before the device was lost".
//
// ★★★★ WHY THIS EXISTS. 2026-11-23, Omix X700 (Mali-G76, Vulkan 1.1): the
//   Player showed VK_ERROR_DEVICE_LOST "observed at RasterViewportFrameRing.cpp"
//   -- the fence wait that NOTICED the loss, never the work that caused it.
//   VK_EXT_device_fault is the tool that names the cause, and Mali does not
//   expose it. The phone belongs to someone else: no adb, no logcat, one APK
//   per round trip. An APK that only says "lost" costs a round trip per guess.
//
// HOW. Each trail owns one HOST_VISIBLE|HOST_COHERENT uint32. At every mark
//   the command buffer gets a full barrier (all earlier commands on the queue
//   must finish) followed by vkCmdFillBuffer writing (submit serial, mark
//   index). After a loss the mapped word still holds the last boundary the GPU
//   retired, so the hung or faulting work is the FIRST mark after it.
//
// ★★ What it can and cannot say:
//   - Marks need to be OUTSIDE a render pass (vkCmdFillBuffer is a transfer
//     command). Work inside one render pass is a single region.
//   - A word whose serial is older than the newest submission means the newer
//     submission never started: the GPU was stuck on something else (another
//     trail, or another queue).
//   - The barrier serialises the frame. Enabled only where nothing better
//     exists: on a device without VK_EXT_device_fault, or when forced with
//     RAYTROPHI_GPU_BREADCRUMBS=1 (=0 disables). Never a silent cost on a
//     device that can report its own faults.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#define RT_BREADCRUMB_CALLER() _ReturnAddress()
#else
#define RT_BREADCRUMB_CALLER() __builtin_return_address(0)
#endif

namespace VulkanRT {

// True when this device was created with VK_EXT_device_fault (defined in
// VulkanBackend.cpp next to the fault registry).
bool vulkanDeviceHasFaultReporting(VkDevice device);

// Logs every live trail's last retired boundary. Called once from
// logVulkanDeviceFaultInfo(), i.e. at the first observed loss.
void logGpuBreadcrumbs();

// Frees every trail's buffer on `device`. Call right before vkDestroyDevice.
void releaseGpuBreadcrumbs(VkDevice device);

// Qualified function name for a code address (Android: dladdr + demangle;
// elsewhere the raw address). Labels one-shot submissions by their caller so
// ~160 call sites need no edits.
std::string describeCodeAddress(void* address);

// Suffix of the mark recorded as the LAST command of a command buffer. A trail
// whose newest retired mark carries it says "that submission finished".
constexpr const char* kEndOfCommandBuffer = "<end>";

class GpuBreadcrumbTrail;
// The raster viewport frame trail: marks are recorded by the stage
// instrumentation, the closing <end> mark and the commit by the frame ring
// right before its vkQueueSubmit (after the readback copy).
GpuBreadcrumbTrail& rasterFrameBreadcrumbs();

class GpuBreadcrumbTrail {
public:
    // keptSubmissions: how many submitted mark lists stay resolvable. One-shot
    // trails see hundreds of submissions at startup and need a deeper history.
    explicit GpuBreadcrumbTrail(const char* name, std::size_t keptSubmissions = 6);
    ~GpuBreadcrumbTrail();
    GpuBreadcrumbTrail(const GpuBreadcrumbTrail&) = delete;
    GpuBreadcrumbTrail& operator=(const GpuBreadcrumbTrail&) = delete;

    // Creates the buffer on first use for `device`. A different device (the
    // viewport is rebuilt after a loss) starts a fresh trail; the old device
    // released this one through releaseGpuBreadcrumbs() before it died.
    // Returns whether marks will be recorded.
    bool ensure(VkPhysicalDevice physical, VkDevice device);
    bool enabled() const { return m_mapped != nullptr; }

    // Starts a new submission's mark list. Call right after vkBeginCommandBuffer.
    void beginSubmit();
    // Must be recorded OUTSIDE a render pass.
    void mark(VkCommandBuffer cmd, const char* label);
    void mark(VkCommandBuffer cmd, const std::string& label) { mark(cmd, label.c_str()); }
    // Call right before vkQueueSubmit of the command buffer the marks went to.
    void commitSubmit();
    // Same, when the recorded buffer is thrown away instead of submitted.
    void abandonSubmit();
    // A whole one-shot submission with a single end-of-buffer mark; thread-safe.
    // Call right before vkEndCommandBuffer.
    void markOneShot(VkCommandBuffer cmd, const std::string& label);

    std::string describe() const;
    VkDevice device() const { return m_device; }
    void release();

private:
    struct Submission {
        std::uint32_t serial = 0;
        std::vector<std::string> labels;
    };
    const char* m_name;
    std::size_t m_keptSubmissions;
    std::mutex m_oneShotMutex;
    VkDevice m_device = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    volatile std::uint32_t* m_mapped = nullptr;
    bool m_disabledForDevice = false;
    std::uint32_t m_serial = 0;      // last serial handed to beginSubmit
    bool m_open = false;
    Submission m_pending;
    mutable std::mutex m_submittedMutex;  // describe() runs on whichever thread saw the loss
    std::vector<Submission> m_submitted;  // newest last
};

} // namespace VulkanRT
