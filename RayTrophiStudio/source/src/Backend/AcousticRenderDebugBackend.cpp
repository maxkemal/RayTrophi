#include "Audio/AcousticSurface.h"
#include "Backend/VulkanBackend.h"
#include <SDL.h>
#include <filesystem>
#include <mutex>

namespace Backend {
bool VulkanBackendAdapter::setAcousticSurface(audio::AcousticSurfaceBatch batch) {
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock() || !m_device || !m_device->isInitialized()) {
        return false;
    }
    const auto& caps = m_device->getCapabilities();
    if (!caps.supportsRayQuery || !caps.supportsBufferDeviceAddress || !caps.supportsShaderInt64) {
        m_acousticSurfaceFailure = "Surface field requires GPU ray query and buffer addresses";
        return false;
    }
    const bool creating = !m_device->acousticSurfacePass;
    if (creating) {
        m_device->acousticSurfacePass = std::make_shared<VulkanRT::AcousticSurfacePass>(*m_device);
    }
    std::string directory;
    if (creating) {
        directory = m_rayFusionShaderDir;
        if (directory.empty()) {
            for (const auto* candidate : {"shaders", "source/shaders", "../shaders"}) {
                if (std::filesystem::exists(std::filesystem::path(candidate) / "acoustic_surface.spv")) {
                    directory = candidate;
                    break;
                }
            }
            if (directory.empty()) {
                if (char* base = SDL_GetBasePath()) {
                    directory = (std::filesystem::path(base) / "shaders").string();
                    SDL_free(base);
                }
            }
        }
    }
    const bool changed = m_device->acousticSurfacePass->update(std::move(batch), directory);
    m_acousticSurfaceFailure.clear();
    if (changed) {
        m_tonemapRefreshPending = true;
    }
    return changed;
}
void VulkanBackendAdapter::disableAcousticSurface() {
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (lock.owns_lock() && m_device && m_device->acousticSurfacePass) {
        m_device->acousticSurfacePass->disable();
    }
}
std::string VulkanBackendAdapter::acousticSurfaceError() const {
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return "Renderer busy";
    }
    if (!m_acousticSurfaceFailure.empty()) {
        return m_acousticSurfaceFailure;
    }
    return m_device && m_device->acousticSurfacePass ?
        m_device->acousticSurfacePass->error() : std::string();
}
} // namespace Backend

namespace VulkanRT {
void VulkanDevice::recordAcousticSurface(VkCommandBuffer cmd, unsigned slot,
    unsigned width, unsigned height, VkImageView output) {
    if (acousticSurfacePass) {
        acousticSurfacePass->record(cmd, slot, acousticSurfacePosition,
            acousticSurfaceNormal, output, width, height, m_tmExposure, m_tmOverlay);
    }
}
} // namespace VulkanRT
