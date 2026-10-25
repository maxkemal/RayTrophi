#include "Backend/VulkanRayGpuTiming.h"
#include "Backend/VulkanBackend.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace VulkanRT {
namespace {
uint64_t nowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
}

bool VulkanRayGpuTiming::configure(VkDevice device, VkPhysicalDevice physical,
    uint32_t queueFamily, bool enabled, std::string& error) {
    if (!enabled) {
        m_enabled = false;
        return true;
    }
    if (m_recorded != 0) {
        collect();
        for (uint32_t i = 0; i < m_recorded; ++i) {
            if (!m_slots[i].consumed) {
                error = "GPU timing results are pending; refresh before rearming";
                return false;
            }
        }
    }
    if (device == VK_NULL_HANDLE || physical == VK_NULL_HANDLE) {
        error = "Vulkan device is unavailable";
        return false;
    }
    if (m_pool == VK_NULL_HANDLE) {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical, &properties);
        if (queueFamily >= count || queues[queueFamily].timestampValidBits == 0 ||
            properties.limits.timestampPeriod <= 0.0f) {
            error = "RT queue does not support GPU timestamp queries";
            return false;
        }
        VkQueryPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = capacity * 2;
        VkQueryPool pool = VK_NULL_HANDLE;
        if (vkCreateQueryPool(device, &info, nullptr, &pool) != VK_SUCCESS) {
            error = "cannot allocate GPU timestamp query pool";
            return false;
        }
        m_device = device;
        m_pool = pool;
        m_validBits = queues[queueFamily].timestampValidBits;
        m_periodNs = properties.limits.timestampPeriod;
    }
    m_supported = true;
    m_enabled = true;
    m_recorded = 0;
    m_failed = 0;
    m_hostWaits = 0;
    m_hostWaitMs = 0.0;
    m_maxHostWaitMs = 0.0;
    m_hostCaptureFinished = false;
    m_asyncSlotWaits = 0;
    m_asyncSlotWaitMs = 0.0;
    m_maxAsyncSlotWaitMs = 0.0;
    m_error.clear();
    m_slots = {};
    m_samples.clear();
    return true;
}

void VulkanRayGpuTiming::destroy() {
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device, m_pool, nullptr);
    }
    m_pool = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_enabled = false;
    m_supported = false;
    m_recorded = 0;
    m_slots = {};
    m_samples.clear();
}

void VulkanRayGpuTiming::resetQueries(VkCommandBuffer command) {
    vkCmdResetQueryPool(command, m_pool, 0, capacity * 2);
}

uint32_t VulkanRayGpuTiming::begin(VkCommandBuffer command, uint32_t width,
    uint32_t height, uint32_t sampleIndex) {
    if (!m_enabled || m_recorded >= capacity || command == VK_NULL_HANDLE) {
        return invalid;
    }
    const uint32_t token = m_recorded++;
    m_slots[token].command = command;
    m_slots[token].sample.captureIndex = token;
    m_slots[token].sample.sampleIndex = sampleIndex;
    m_slots[token].sample.width = width;
    m_slots[token].sample.height = height;
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_pool, token * 2);
    return token;
}

void VulkanRayGpuTiming::end(VkCommandBuffer command, uint32_t token) {
    if (token >= m_recorded || token == invalid) {
        return;
    }
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        m_pool, token * 2 + 1);
    m_slots[token].ended = true;
}

void VulkanRayGpuTiming::cancel(VkCommandBuffer command) {
    for (uint32_t i = 0; i < m_recorded; ++i) {
        auto& slot = m_slots[i];
        if (slot.command == command && !slot.consumed) {
            slot.consumed = true;
            ++m_failed;
            m_error = "RT command submission failed; capture is incomplete";
        }
    }
}

void VulkanRayGpuTiming::collect() {
    if (m_pool == VK_NULL_HANDLE) {
        return;
    }
    for (uint32_t i = 0; i < m_recorded; ++i) {
        auto& slot = m_slots[i];
        if (!slot.ended || slot.consumed) {
            continue;
        }
        // value, availability for each of two queries. No wait, no GPU idle.
        uint64_t values[4]{};
        const VkResult result = vkGetQueryPoolResults(m_device, m_pool, i * 2, 2,
            sizeof(values), values, sizeof(uint64_t) * 2,
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result != VK_SUCCESS && result != VK_NOT_READY) {
            slot.consumed = true;
            ++m_failed;
            m_error = "GPU timestamp read failed: " + std::to_string(result);
            continue;
        }
        if (result == VK_NOT_READY || values[1] == 0 || values[3] == 0) {
            continue;
        }
        slot.consumed = true;
        const uint64_t mask = m_validBits >= 64 ? UINT64_MAX :
            ((uint64_t(1) << m_validBits) - 1);
        const uint64_t elapsed = (values[2] - values[0]) & mask;
        slot.sample.gpuMs = static_cast<double>(elapsed) * m_periodNs / 1000000.0;
        m_samples.push_back(slot.sample);
    }
}

RayGpuTimingSnapshot VulkanRayGpuTiming::snapshot() {
    // Explicit caller request only; disabled capture can still retrieve its
    // final pending results, without per-frame automatic polling.
    collect();
    RayGpuTimingSnapshot result;
    result.enabled = m_enabled;
    result.capturing = m_enabled && m_recorded < capacity;
    result.supported = m_supported;
    result.recorded = m_recorded;
    result.failed = m_failed;
    result.hostWaits = m_hostWaits;
    result.hostWaitMs = m_hostWaitMs;
    result.maxHostWaitMs = m_maxHostWaitMs;
    result.asyncSlotWaits = m_asyncSlotWaits;
    result.asyncSlotWaitMs = m_asyncSlotWaitMs;
    result.maxAsyncSlotWaitMs = m_maxAsyncSlotWaitMs;
    result.error = m_error;
    result.samples = m_samples;
    std::sort(result.samples.begin(), result.samples.end(), [](const auto& a, const auto& b) {
        return a.captureIndex < b.captureIndex;
    });
    for (uint32_t i = 0; i < m_recorded; ++i) {
        if (!m_slots[i].consumed) {
            ++result.pending;
        }
    }
    return result;
}

uint64_t VulkanRayGpuTiming::beginHostWait() const {
    return m_enabled && !m_hostCaptureFinished && m_recorded > 0 ? nowNs() : 0;
}

void VulkanRayGpuTiming::endHostWait(uint64_t started, bool asyncSlot) {
    if (started == 0) {
        return;
    }
    const double elapsed = static_cast<double>(nowNs() - started) / 1000000.0;
    if (asyncSlot) {
        ++m_asyncSlotWaits;
        m_asyncSlotWaitMs += elapsed;
        m_maxAsyncSlotWaitMs = std::max(m_maxAsyncSlotWaitMs, elapsed);
    } else {
        ++m_hostWaits;
        m_hostWaitMs += elapsed;
        m_maxHostWaitMs = std::max(m_maxHostWaitMs, elapsed);
    }
    if (m_recorded >= capacity) {
        m_hostCaptureFinished = true;
    }
}

bool VulkanDevice::setRayGpuTiming(bool enabled, std::string& error) {
    if (!m_rayGpuTiming) {
        if (!enabled) {
            return true;
        }
        m_rayGpuTiming = std::make_unique<VulkanRayGpuTiming>();
    }
    if (!m_rayGpuTiming->configure(m_device, m_physicalDevice,
            m_computeQueueFamily, enabled, error)) {
        return false;
    }
    if (enabled) {
        // Explicit capture setup only. Drain the reset once so an asynchronous
        // refresh cannot mistake old query availability for the new capture.
        VkCommandBuffer command = beginSingleTimeCommands();
        if (command == VK_NULL_HANDLE) {
            m_rayGpuTiming->configure(m_device, m_physicalDevice,
                m_computeQueueFamily, false, error);
            error = "cannot initialize GPU timing capture";
            return false;
        }
        m_rayGpuTiming->resetQueries(command);
        endSingleTimeCommands(command);
    }
    return true;
}

RayGpuTimingSnapshot VulkanDevice::rayGpuTimingStats() {
    return m_rayGpuTiming ? m_rayGpuTiming->snapshot() : RayGpuTimingSnapshot{};
}

uint32_t VulkanDevice::beginRayGpuTiming(VkCommandBuffer command, uint32_t w, uint32_t h) {
    if (!m_rayGpuTiming || !m_rayGpuTiming->enabled()) {
        return VulkanRayGpuTiming::invalid;
    }
    uint32_t sample = 0;
    if (m_pushConstantData.size() >= 68) {
        std::memcpy(&sample, m_pushConstantData.data() + 64, sizeof(sample));
    }
    return m_rayGpuTiming->begin(command, w, h, sample);
}

void VulkanDevice::endRayGpuTiming(VkCommandBuffer command, uint32_t token) {
    if (m_rayGpuTiming && token != VulkanRayGpuTiming::invalid) {
        m_rayGpuTiming->end(command, token);
    }
}

} // namespace VulkanRT

namespace Backend {
bool VulkanBackendAdapter::setRayGpuTimingCapture(bool enabled, std::string& error) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) {
        error = "Vulkan render device is unavailable";
        return false;
    }
    return m_device->setRayGpuTiming(enabled, error);
}

VulkanRT::RayGpuTimingSnapshot VulkanBackendAdapter::rayGpuTimingCaptureStats() {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_device ? m_device->rayGpuTimingStats() : VulkanRT::RayGpuTimingSnapshot{};
}

bool VulkanBackendAdapter::rayGpuTimingCaptureEnabled() {
    // UI/main-thread state peek only, matching getVulkanDevice(). Do not make
    // the disabled performance panel wait behind a render-worker mutex.
    return m_device && m_device->rayGpuTimingEnabled();
}
} // namespace Backend
