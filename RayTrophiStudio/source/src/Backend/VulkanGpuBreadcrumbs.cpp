// GPU breadcrumbs -- see Backend/VulkanGpuBreadcrumbs.h for what the numbers mean.

#include "Backend/VulkanGpuBreadcrumbs.h"
#include "globals.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__ANDROID__) || defined(__linux__)
#include <cxxabi.h>
#include <dlfcn.h>
#endif

namespace VulkanRT {
namespace {

// Function-local: trails can be file-scope statics in other translation units,
// and their constructors must not run before these exist.
std::mutex& trailsMutex() { static std::mutex m; return m; }
std::vector<GpuBreadcrumbTrail*>& trails() { static std::vector<GpuBreadcrumbTrail*> v; return v; }

// -1 = follow the device (on without VK_EXT_device_fault), 0 = off, 1 = on.
int breadcrumbOverride() {
    const char* value = std::getenv("RAYTROPHI_GPU_BREADCRUMBS");
    if (!value || !*value) return -1;
    return value[0] == '0' ? 0 : 1;
}

constexpr std::uint32_t packWord(std::uint32_t serial, std::uint32_t markIndex) {
    return ((serial & 0xFFFFu) << 16) | ((markIndex + 1u) & 0xFFFFu);
}

} // namespace

GpuBreadcrumbTrail::GpuBreadcrumbTrail(const char* name, std::size_t keptSubmissions)
    : m_name(name), m_keptSubmissions(keptSubmissions < 2 ? 2 : keptSubmissions) {
    std::lock_guard<std::mutex> lock(trailsMutex());
    trails().push_back(this);
}

GpuBreadcrumbTrail::~GpuBreadcrumbTrail() {
    {
        std::lock_guard<std::mutex> lock(trailsMutex());
        auto& list = trails();
        list.erase(std::remove(list.begin(), list.end(), this), list.end());
    }
    release();
}

void GpuBreadcrumbTrail::release() {
    // Destroying objects of a LOST device is valid Vulkan; only a destroyed
    // device is not, and releaseGpuBreadcrumbs() runs before vkDestroyDevice.
    if (m_device != VK_NULL_HANDLE) {
        if (m_mapped) vkUnmapMemory(m_device, m_memory);
        if (m_buffer != VK_NULL_HANDLE) vkDestroyBuffer(m_device, m_buffer, nullptr);
        if (m_memory != VK_NULL_HANDLE) vkFreeMemory(m_device, m_memory, nullptr);
    }
    m_device = VK_NULL_HANDLE;
    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_mapped = nullptr;
    m_open = false;
}

void releaseGpuBreadcrumbs(VkDevice device) {
    std::lock_guard<std::mutex> lock(trailsMutex());
    for (GpuBreadcrumbTrail* trail : trails())
        if (trail->device() == device) trail->release();
}

bool GpuBreadcrumbTrail::ensure(VkPhysicalDevice physical, VkDevice device) {
    if (device == VK_NULL_HANDLE || physical == VK_NULL_HANDLE) return false;
    if (device == m_device) return m_mapped != nullptr;

    // New device: the old one released this trail before it was destroyed
    // (releaseGpuBreadcrumbs); if not, its handles are dropped, never called.
    m_device = device;
    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_mapped = nullptr;
    m_open = false;
    m_pending = Submission{};
    {
        std::lock_guard<std::mutex> lock(m_submittedMutex);
        m_submitted.clear();
    }

    const int forced = breadcrumbOverride();
#ifdef RT_PLAYER
    // In Player, breadcrumbs insert full ALL_COMMANDS_BIT pipeline barriers on
    // every stage boundary (severe on Mali/TBDR). Keep opt-in via
    // RAYTROPHI_GPU_BREADCRUMBS=1 for device-lost diagnostics.
    const bool want = forced == 1;
#else
    const bool want = forced == 1 || (forced == -1 && !vulkanDeviceHasFaultReporting(device));
#endif
    m_disabledForDevice = !want;
    if (!want) return false;

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 64;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bi, nullptr, &m_buffer) != VK_SUCCESS) {
        m_buffer = VK_NULL_HANDLE;
        SCENE_LOG_WARN(std::string("[GpuBreadcrumb] ") + m_name + ": buffer creation failed; trail off");
        return false;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, m_buffer, &req);
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    const VkMemoryPropertyFlags wanted =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    std::uint32_t typeIndex = UINT32_MAX;
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & wanted) == wanted) {
            typeIndex = i;
            break;
        }
    }
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = typeIndex;
    void* mapped = nullptr;
    if (typeIndex == UINT32_MAX ||
        vkAllocateMemory(device, &ai, nullptr, &m_memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, m_buffer, m_memory, 0) != VK_SUCCESS ||
        vkMapMemory(device, m_memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
        if (m_memory != VK_NULL_HANDLE) vkFreeMemory(device, m_memory, nullptr);
        vkDestroyBuffer(device, m_buffer, nullptr);
        m_buffer = VK_NULL_HANDLE;
        m_memory = VK_NULL_HANDLE;
        SCENE_LOG_WARN(std::string("[GpuBreadcrumb] ") + m_name +
                       ": no host-coherent memory; trail off");
        return false;
    }
    m_mapped = static_cast<volatile std::uint32_t*>(mapped);
    m_mapped[0] = 0u;
    SCENE_LOG_INFO(std::string("[GpuBreadcrumb] ") + m_name + " trail on (" +
                   (forced == 1 ? "forced by RAYTROPHI_GPU_BREADCRUMBS"
                                : "device has no VK_EXT_device_fault") + ")");
    return true;
}

void GpuBreadcrumbTrail::beginSubmit() {
    if (!m_mapped) return;
    ++m_serial;
    m_pending.serial = m_serial;
    m_pending.labels.clear();
    m_open = true;
}

void GpuBreadcrumbTrail::mark(VkCommandBuffer cmd, const char* label) {
    if (!m_mapped || !m_open || cmd == VK_NULL_HANDLE) return;
    const std::uint32_t index = static_cast<std::uint32_t>(m_pending.labels.size());
    if (index >= 0xFFFEu) return;
    m_pending.labels.emplace_back(label ? label : "?");

    // First scope: every earlier command on this queue, any stage. The fill
    // therefore lands only once all of it retired -- that is the whole claim.
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    vkCmdFillBuffer(cmd, m_buffer, 0, 4, packWord(m_pending.serial, index));
}

void GpuBreadcrumbTrail::commitSubmit() {
    if (!m_mapped || !m_open) return;
    m_open = false;
    std::lock_guard<std::mutex> lock(m_submittedMutex);
    m_submitted.push_back(std::move(m_pending));
    m_pending = Submission{};
    if (m_submitted.size() > m_keptSubmissions)
        m_submitted.erase(m_submitted.begin());
}

void GpuBreadcrumbTrail::markOneShot(VkCommandBuffer cmd, const std::string& label) {
    if (!m_mapped || cmd == VK_NULL_HANDLE) return;
    // One-shot submissions come from any thread; the whole begin/mark/commit
    // sequence is one critical section so serials and labels stay paired.
    std::lock_guard<std::mutex> lock(m_oneShotMutex);
    beginSubmit();
    mark(cmd, label + " " + kEndOfCommandBuffer);
    commitSubmit();
}

void GpuBreadcrumbTrail::abandonSubmit() {
    m_open = false;
    m_pending = Submission{};
}

std::string GpuBreadcrumbTrail::describe() const {
    const std::string tag = std::string("[GpuBreadcrumb] ") + m_name + ": ";
    if (m_disabledForDevice) return tag + "off (device reports its own faults)";
    if (!m_mapped) return tag + "not created";

    const std::uint32_t word = m_mapped[0];
    std::lock_guard<std::mutex> lock(m_submittedMutex);
    if (m_submitted.empty()) return tag + "nothing submitted yet";
    const Submission& newest = m_submitted.back();
    if (word == 0u) {
        return tag + "no mark ever retired; first submission " +
               std::to_string(m_submitted.front().serial) +
               " never got past its first mark '" +
               (m_submitted.front().labels.empty() ? std::string("?")
                                                   : m_submitted.front().labels.front()) + "'";
    }
    const std::uint32_t serial16 = word >> 16;
    const std::uint32_t index = (word & 0xFFFFu) - 1u;
    const Submission* hit = nullptr;
    for (const Submission& s : m_submitted)
        if ((s.serial & 0xFFFFu) == serial16) hit = &s;
    if (!hit) {
        return tag + "retired word serial " + std::to_string(serial16) +
               " is older than the kept submissions (newest " +
               std::to_string(newest.serial) + "): the GPU stopped long before the loss was seen";
    }
    const std::size_t count = hit->labels.size();
    const std::string last = index < count ? hit->labels[index] : std::string("?");
    std::string out = tag + "submission " + std::to_string(hit->serial) + " retired mark " +
                      std::to_string(index + 1u) + "/" + std::to_string(count) + " '" + last + "'";
    if (index + 1u < count) {
        out += "; STUCK IN the work between it and '" + hit->labels[index + 1u] + "'";
    } else if (last.find(kEndOfCommandBuffer) != std::string::npos) {
        out += "; that submission FINISHED";
    } else {
        out += "; work recorded after the last mark is not covered";
    }
    // The first submission after the retired one is where the GPU stopped on
    // this trail (or it never started because another queue/trail held it).
    for (const Submission& s : m_submitted) {
        if (s.serial > hit->serial) {
            out += "; FIRST UNRETIRED submission " + std::to_string(s.serial) + " '" +
                   (s.labels.empty() ? std::string("?") : s.labels.front()) + "'";
            if (s.serial != newest.serial)
                out += " (newest " + std::to_string(newest.serial) + ")";
            break;
        }
    }
    return out;
}

std::string describeCodeAddress(void* address) {
    if (!address) return "?";
#if defined(__ANDROID__) || defined(__linux__)
    Dl_info info{};
    if (dladdr(address, &info) && info.dli_sname) {
        int status = 0;
        char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
        std::string name = (status == 0 && demangled) ? demangled : info.dli_sname;
        std::free(demangled);
        // Keep the qualified function name, drop the parameter list.
        const std::size_t paren = name.find('(');
        if (paren != std::string::npos) name.resize(paren);
        return name;
    }
    if (dladdr(address, &info) && info.dli_fbase) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "+0x%zx",
                      static_cast<std::size_t>(static_cast<char*>(address) -
                                               static_cast<char*>(info.dli_fbase)));
        return buf;
    }
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%p", address);
    return buf;
}

GpuBreadcrumbTrail& rasterFrameBreadcrumbs() {
    static GpuBreadcrumbTrail trail("raster_frame");
    return trail;
}

void logGpuBreadcrumbs() {
    std::vector<GpuBreadcrumbTrail*> list;
    {
        std::lock_guard<std::mutex> lock(trailsMutex());
        list = trails();
    }
    if (list.empty()) {
        SCENE_LOG_WARN("[GpuBreadcrumb] no trail registered");
        return;
    }
    for (const GpuBreadcrumbTrail* trail : list) {
        SCENE_LOG_ERROR(trail->describe());
    }
}

} // namespace VulkanRT
