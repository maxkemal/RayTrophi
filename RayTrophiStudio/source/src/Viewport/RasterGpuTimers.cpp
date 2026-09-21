#include "Viewport/RasterGpuTimers.h"
#include "Backend/VulkanBackend.h"

#include <algorithm>
#include <vector>

namespace Backend {

const char* rasterStageName(RasterStage stage) {
    switch (stage) {
        case RasterStage::GpuCull:      return "gpu_cull";
        case RasterStage::ShadowAtlas:  return "shadow_atlas";
        case RasterStage::TransmissionPrep: return "transmission_prep";
        case RasterStage::Sky:          return "sky";
        case RasterStage::DepthPrepass: return "depth_prepass";
        case RasterStage::RtShadow:     return "rt_shadow";
        case RasterStage::ScreenGiTrace: return "screen_gi_trace";
        case RasterStage::ScreenGiFilter: return "screen_gi_filter";
        case RasterStage::Reflection: return "reflection";
        case RasterStage::ReflectionFilter: return "reflection_filter";
        case RasterStage::MainPass:     return "main_pass";
        case RasterStage::VolumeSdf:    return "volume_sdf";
        case RasterStage::Transmission: return "transmission";
        case RasterStage::Taa:          return "taa";
        case RasterStage::Post:         return "post";
        case RasterStage::Overlay:      return "overlay";
        default:                        return "unknown";
    }
}

bool RasterGpuTimers::ensure(VulkanRT::VulkanDevice& device) {
    if (m_pool != VK_NULL_HANDLE) return m_supported;
    m_supported = false;

    VkDevice vk = device.getDevice();
    VkPhysicalDevice physical = device.getPhysicalDevice();
    if (vk == VK_NULL_HANDLE || physical == VK_NULL_HANDLE) {
        m_reason = "no device";
        return false;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical, &props);
    if (props.limits.timestampPeriod <= 0.0f) {
        m_reason = "device reports no timestamp period";
        return false;
    }

    // ★★ timestampValidBits is PER QUEUE FAMILY, and a device that advertises a
    //   timestamp period can still have a queue that cannot write one. This app
    //   submits raster work on the same queue it uses for compute, so that is
    //   the family that has to be asked -- checking the device limit alone
    //   would produce a pool whose results are always garbage.
    std::uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
    const std::uint32_t family = device.getComputeQueueFamily();
    if (family >= familyCount || families[family].timestampValidBits == 0u) {
        m_reason = "the submit queue family cannot write timestamps";
        return false;
    }

    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = kSlotCount * kMarksPerSlot;
    if (vkCreateQueryPool(vk, &info, nullptr, &m_pool) != VK_SUCCESS) {
        m_pool = VK_NULL_HANDLE;
        m_reason = "timestamp query pool creation failed";
        return false;
    }

    m_periodNs = static_cast<double>(props.limits.timestampPeriod);
    m_slotMarks.fill(0u);
    m_slotSerial.fill(0u);
    m_slotStageBits.fill(0u);
    m_supported = true;
    m_reason = "";
    return true;
}

void RasterGpuTimers::destroy(VulkanRT::VulkanDevice& device) {
    if (m_pool != VK_NULL_HANDLE && device.getDevice() != VK_NULL_HANDLE)
        vkDestroyQueryPool(device.getDevice(), m_pool, nullptr);
    m_pool = VK_NULL_HANDLE;
    m_supported = false;
    m_open = false;
    m_reason = "not initialised";
    m_slotMarks.fill(0u);
    m_slotStageBits.fill(0u);
}

void RasterGpuTimers::beginFrame(VkCommandBuffer cmd) {
    if (!m_supported || m_pool == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE) return;
    m_slot = static_cast<std::uint32_t>(m_frameSerial % kSlotCount);
    // Invalidated BEFORE the reset, not after: if this command buffer is never
    // submitted the slot must not stay collectable with last frame's numbers.
    m_slotMarks[m_slot] = 0u;
    m_slotStageBits[m_slot] = 0u;
    m_writtenMarks = 0u;
    m_stageBits = 0u;
    const std::uint32_t base = m_slot * kMarksPerSlot;
    vkCmdResetQueryPool(cmd, m_pool, base, kMarksPerSlot);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_pool, base);
    m_writtenMarks = 1u;
    m_open = true;
}

void RasterGpuTimers::mark(VkCommandBuffer cmd, RasterStage stage, bool ran) {
    if (!m_open || cmd == VK_NULL_HANDLE) return;
    const std::uint32_t index = static_cast<std::uint32_t>(stage) + 1u;
    // ★★★ Closes every stage from wherever we are THROUGH `stage`. The frame has
    //   many conditional bodies (no HDR pass, no instances, no depth prepass,
    //   no transmission) and requiring one mark call per stage on every path
    //   would mean a path that forgets one silently loses the WHOLE frame's
    //   timings. Filling the gap here makes the skipped stages report ~0 ms
    //   with ran=false, which is the truth about them.
    if (index < m_writtenMarks) {
        // Backwards means the call sites and the enum have drifted. Publishing
        // deltas between mismatched boundaries would look plausible and be
        // wrong, which is exactly what this file exists to prevent.
        m_open = false;
        m_writtenMarks = 0u;
        return;
    }
    while (m_writtenMarks <= index) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_pool,
                            m_slot * kMarksPerSlot + m_writtenMarks);
        ++m_writtenMarks;
    }
    if (ran) m_stageBits |= (1u << static_cast<std::uint32_t>(stage));
}

void RasterGpuTimers::endFrame(VkCommandBuffer cmd) {
    if (!m_open || cmd == VK_NULL_HANDLE) return;
    // Close out whatever the frame did not reach, so a frame that returned
    // early is still collectable instead of vanishing from the window.
    while (m_writtenMarks < kMarksPerSlot) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_pool,
                            m_slot * kMarksPerSlot + m_writtenMarks);
        ++m_writtenMarks;
    }
    m_open = false;
    m_slotMarks[m_slot] = m_writtenMarks;
    m_slotStageBits[m_slot] = m_stageBits;
    m_slotSerial[m_slot] = ++m_frameSerial;
}

RasterGpuFrameTimings RasterGpuTimers::collect(VulkanRT::VulkanDevice& device) {
    RasterGpuFrameTimings out{};
    if (!m_supported || m_pool == VK_NULL_HANDLE) return out;
    VkDevice vk = device.getDevice();
    if (vk == VK_NULL_HANDLE) return out;

    // Newest complete slot first. Without WAIT this returns VK_NOT_READY for a
    // frame still in flight, which is the answer we want: skip it, do not stall
    // the pipeline to read a number about the pipeline.
    std::uint32_t order[kSlotCount];
    for (std::uint32_t i = 0; i < kSlotCount; ++i) order[i] = i;
    std::sort(std::begin(order), std::end(order),
              [this](std::uint32_t a, std::uint32_t b) {
                  return m_slotSerial[a] > m_slotSerial[b];
              });

    std::uint64_t stamps[kMarksPerSlot]{};
    for (std::uint32_t i = 0; i < kSlotCount; ++i) {
        const std::uint32_t slot = order[i];
        if (m_slotMarks[slot] != kMarksPerSlot) continue;
        const VkResult result = vkGetQueryPoolResults(
            vk, m_pool, slot * kMarksPerSlot, kMarksPerSlot,
            sizeof(stamps), stamps, sizeof(std::uint64_t),
            VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS) continue;

        double total = 0.0;
        for (std::uint32_t s = 0; s < static_cast<std::uint32_t>(RasterStage::Count); ++s) {
            // Ticks are unsigned and the counter can wrap. A negative delta is
            // a wrap or a driver quirk, not a fast pass; clamp it to zero and
            // let the total stay honest rather than reporting a negative stage.
            const std::uint64_t a = stamps[s];
            const std::uint64_t b = stamps[s + 1u];
            const double ms = (b > a)
                ? (static_cast<double>(b - a) * m_periodNs) / 1.0e6
                : 0.0;
            out.stageMs[s] = ms;
            out.stageRan[s] = (m_slotStageBits[slot] & (1u << s)) != 0u;
            total += ms;
        }
        out.totalMs = total;
        out.frameSerial = m_slotSerial[slot];
        out.available = true;
        return out;
    }
    return out;
}

} // namespace Backend
