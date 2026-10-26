#include "Backend/VulkanMotionHistory.h"

#include "Backend/VulkanBackend.h"

#include <cmath>
#include <cstring>

namespace VulkanRT {

namespace {

constexpr uint32_t kParamSlots = 4;   // > frames in flight, so a slot is never rewritten while read

float halton(uint32_t index, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

} // namespace

struct VulkanMotionHistory::Impl {
    ImageHandle motion;
    ImageHandle history[2];
    BufferHandle params;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t readIndex = 0;
    bool historyValid = false;
    bool prevCameraValid = false;
    float pendingCamera[4][4] = {};
    float prevCamera[4][4] = {};
    uint64_t dispatch = 0;
};

VulkanMotionHistory::VulkanMotionHistory()
    : m_impl(std::make_unique<Impl>()) {}

VulkanMotionHistory::~VulkanMotionHistory() = default;

bool VulkanMotionHistory::ensure(VulkanDevice& device, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return false;
    if (m_impl->width == width && m_impl->height == height && m_impl->motion.image &&
        m_impl->history[0].image && m_impl->history[1].image && m_impl->params.buffer) {
        return true;
    }
    destroy(device);

    constexpr VkImageUsageFlags usage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    m_impl->motion = device.createImage2D(width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
    for (auto& h : m_impl->history)
        h = device.createImage2D(width, height, VK_FORMAT_R32G32B32A32_UINT, usage);

    BufferCreateInfo pci;
    pci.size = sizeof(MotionParamsGPU) * kParamSlots;
    pci.usage = BufferUsage::STORAGE;
    pci.location = MemoryLocation::CPU_TO_GPU;
    pci.initialData = nullptr;
    m_impl->params = device.createBuffer(pci);

    if (!m_impl->motion.image || !m_impl->history[0].image || !m_impl->history[1].image ||
        !m_impl->params.buffer || m_impl->params.deviceAddress == 0) {
        destroy(device);
        return false;
    }
    m_impl->width = width;
    m_impl->height = height;
    device.clearImages({
        { &m_impl->motion, 0.0f, 0.0f, 0.0f, 1.0f },
        { &m_impl->history[0], 0.0f, 0.0f, 0.0f, 0.0f },
        { &m_impl->history[1], 0.0f, 0.0f, 0.0f, 0.0f }
    });
    invalidate();
    return true;
}

void VulkanMotionHistory::destroy(VulkanDevice& device) {
    if (m_impl->motion.image) device.destroyImage(m_impl->motion);
    for (auto& h : m_impl->history)
        if (h.image) device.destroyImage(h);
    if (m_impl->params.buffer) device.destroyBuffer(m_impl->params);
    m_impl->motion = {};
    m_impl->history[0] = {};
    m_impl->history[1] = {};
    m_impl->params = {};
    m_impl->width = 0;
    m_impl->height = 0;
    invalidate();
}

void VulkanMotionHistory::setEnabled(bool enabled) {
    // History written before the AOV was switched off is not the previous dispatch.
    if (enabled && !m_enabled) invalidate();
    m_enabled = enabled;
}

void VulkanMotionHistory::invalidate() {
    m_impl->historyValid = false;
    m_impl->prevCameraValid = false;
    ++m_counters.history_resets;
}

uint64_t VulkanMotionHistory::prepareDispatch(VulkanDevice& device, const float camera[4][4],
                                              uint64_t curInstances, uint64_t prevInstances,
                                              uint32_t instanceCount, bool objectMotion) {
    std::memcpy(m_impl->pendingCamera, camera, sizeof(m_impl->pendingCamera));
    if (!m_enabled || !m_impl->params.buffer) return 0;

    MotionParamsGPU rec{};
    std::memcpy(rec.prevOrigin, m_impl->prevCamera[0], sizeof(rec.prevOrigin));
    std::memcpy(rec.prevLowerLeft, m_impl->prevCamera[1], sizeof(rec.prevLowerLeft));
    std::memcpy(rec.prevHorizontal, m_impl->prevCamera[2], sizeof(rec.prevHorizontal));
    std::memcpy(rec.prevVertical, m_impl->prevCamera[3], sizeof(rec.prevVertical));

    // Halton(2,3) over 8 phases, centred on the pixel; 0 in random-AA mode.
    if (m_halton) {
        const uint32_t k = static_cast<uint32_t>(m_impl->dispatch % 8u) + 1u;
        m_jitter[0] = halton(k, 2) - 0.5f;
        m_jitter[1] = halton(k, 3) - 0.5f;
    } else {
        m_jitter[0] = m_jitter[1] = 0.0f;
    }
    rec.jitter[0] = m_jitter[0];
    rec.jitter[1] = m_jitter[1];

    const bool objects = objectMotion && curInstances != 0 && prevInstances != 0 && instanceCount > 0;
    rec.instAddr[0] = static_cast<uint32_t>(curInstances & 0xffffffffull);
    rec.instAddr[1] = static_cast<uint32_t>(curInstances >> 32);
    rec.instAddr[2] = static_cast<uint32_t>(prevInstances & 0xffffffffull);
    rec.instAddr[3] = static_cast<uint32_t>(prevInstances >> 32);
    rec.info[0] = objects ? instanceCount : 0u;
    uint32_t flags = 0;
    if (m_impl->historyValid) flags |= kMotionHistoryValid;
    if (objects) flags |= kMotionObjects;
    if (m_halton) flags |= kMotionHalton;
    if (m_impl->prevCameraValid) flags |= kMotionPrevCamera;
    rec.info[1] = flags;
    rec.info[2] = static_cast<uint32_t>(m_impl->dispatch);

    const uint32_t slot = static_cast<uint32_t>(m_impl->dispatch % kParamSlots);
    const size_t offset = size_t(slot) * sizeof(MotionParamsGPU);
    if (void* mapped = device.mapBuffer(m_impl->params)) {
        std::memcpy(static_cast<char*>(mapped) + offset, &rec, sizeof(rec));
        device.unmapBuffer(m_impl->params);
    } else {
        return 0;
    }
    if (objects) ++m_counters.object_motion_dispatches;
    return m_impl->params.deviceAddress + offset;
}

void VulkanMotionHistory::advance() {
    std::memcpy(m_impl->prevCamera, m_impl->pendingCamera, sizeof(m_impl->prevCamera));
    m_impl->prevCameraValid = true;
    if (m_enabled && m_impl->motion.image) {
        m_impl->readIndex ^= 1u;
        m_impl->historyValid = true;
    }
    ++m_impl->dispatch;
    ++m_counters.dispatches;
}

uint32_t VulkanMotionHistory::writeIndex() const { return m_impl->readIndex ^ 1u; }

void VulkanMotionHistory::cameraFootprint(float out[4]) const {
    const float (*cam)[4] = m_impl->pendingCamera;   // origin, lowerLeft, horizontal, vertical
    out[0] = cam[0][0];
    out[1] = cam[0][1];
    out[2] = cam[0][2];
    // Image-plane centre distance and vertical extent give the pixel pitch at
    // that distance; divided by the distance it is per unit of depth.
    float centre[3], dist2 = 0.0f, vlen2 = 0.0f;
    for (int i = 0; i < 3; ++i) {
        centre[i] = cam[1][i] + 0.5f * cam[2][i] + 0.5f * cam[3][i] - cam[0][i];
        dist2 += centre[i] * centre[i];
        vlen2 += cam[3][i] * cam[3][i];
    }
    const float dist = std::sqrt(dist2);
    out[3] = (dist > 0.0f && m_impl->height > 0)
        ? std::sqrt(vlen2) / (dist * static_cast<float>(m_impl->height)) : 0.0f;
}
bool VulkanMotionHistory::hasHistory() const { return m_impl->historyValid; }
const ImageHandle* VulkanMotionHistory::motionImage() const { return &m_impl->motion; }
const ImageHandle* VulkanMotionHistory::historySlot(uint32_t index) const { return &m_impl->history[index & 1u]; }
uint32_t VulkanMotionHistory::width() const { return m_impl->width; }
uint32_t VulkanMotionHistory::height() const { return m_impl->height; }

} // namespace VulkanRT
