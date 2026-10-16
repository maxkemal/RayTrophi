#include "Backend/VulkanBackend.h"
#include "TriangleMesh.h"
#include "HittableInstance.h"
#include "Triangle.h"
#include "globals.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <SDL.h>

namespace Backend {
audio::AcousticGpuGeometry VulkanBackendAdapter::acousticGpuGeometry() const {
    audio::AcousticGpuGeometry result;
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        result.error = "Active renderer is busy; acoustic query deferred";
        return result;
    }
    if (!m_device || !m_device->isInitialized() ||
        !m_device->getCapabilities().supportsRayQuery ||
        !m_device->getCapabilities().supportsBufferDeviceAddress ||
        !m_device->getCapabilities().supportsShaderInt64 || !m_device->hasTLAS()) {
        result.error = "Active backend has no compatible current GPU ray-query AS";
        return result;
    }
    std::ostringstream revision;
    revision << m_device->getTLASHandle() << ':' << g_scene_geometry_generation.load() << ':';
    if (shouldUseInteractiveViewport()) {
        if (!rayFusionAcousticGeometry(result)) {
            return result;
        }
        if (m_device->tlasInstanceCount() != result.instances.size()) {
            result.error = "Raster AS instance table no longer matches the active TLAS";
            return result;
        }
        revision << result.revision;
        result.revision = revision.str();
        return result;
    }
    if (m_vkInstances.size() > audio::kAcousticGpuMaxInstances) {
        result.error = "RT acoustic instance budget exceeded: " +
            std::to_string(m_vkInstances.size()) + " > " +
            std::to_string(audio::kAcousticGpuMaxInstances);
        return result;
    }
    if (m_vkInstances.empty() || !m_hairVkInstances.empty() ||
        m_vkInstances.size() != m_instanceSources.size() ||
        m_device->tlasInstanceCount() != m_vkInstances.size() || m_gpuInstanceUpdatePending) {
        result.error = "RT AS instance coverage exceeds acoustic limits or has unsupported entries";
        return result;
    }
    const auto maximum = m_device->getCapabilities().maxTlasInstances;
    if (maximum && m_vkInstances.size() > maximum) {
        result.error = "RT AS was truncated by the device instance limit";
        return result;
    }
    for (const auto& instance : m_vkInstances) {
        if (instance.blasIndex >= m_device->m_blasList.size()) {
            result.error = "RT AS has hidden or unsupported acoustic geometry";
            return result;
        }
    }
    result.instances.resize(m_vkInstances.size());
    result.names.resize(m_vkInstances.size());
    for (std::size_t i = 0; i < m_vkInstances.size(); ++i) {
        const auto& instance = m_vkInstances[i];
        // uint8_t streams as a character: 0xFF would corrupt the UTF-8 JSON stamp.
        revision << instance.customIndex << ':' <<
            static_cast<std::uint32_t>(instance.mask) << ':';
        if (instance.mask == 0) {
            continue;
        }
        const auto& blas = m_device->m_blasList[instance.blasIndex];
        std::string name;
        if (const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(m_instanceSources[i])) {
            name = mesh->nodeName;
        } else if (const auto object =
            std::dynamic_pointer_cast<HittableInstance>(m_instanceSources[i])) {
            name = object->node_name;
        } else if (const auto representative =
            std::dynamic_pointer_cast<Triangle>(m_instanceSources[i])) {
            // Legacy renderer identity only; geometry comes exclusively from
            // the current GPU buffers and canonical flat-object coverage check.
            name = representative->getNodeName();
        }
        if (name.empty() || blas.aabbCount || !blas.vertexBuffer.deviceAddress ||
            blas.vertexCount < 3 || (blas.indexCount && !blas.indexBuffer.deviceAddress)) {
            result.error = "RT AS requires flat triangle geometry with unique object mapping";
            return result;
        }
        auto& item = result.instances[i];
        item.positions = blas.vertexBuffer.deviceAddress;
        item.indices = blas.indexCount ? blas.indexBuffer.deviceAddress : 0;
        item.vertexCount = blas.vertexCount;
        item.triangleCount = blas.indexCount ? blas.indexCount / 3 : blas.vertexCount / 3;
        result.names[i] = name;
        revision << instance.customIndex << ':' << name.size() << ':' << name << ':' <<
            instance.blasIndex << ':' << item.positions << ':' << item.indices << ':' <<
            item.vertexCount << ':' << item.triangleCount << ':' << blas.refitCount << ':';
        // Bit-exact matrix identity avoids textual float precision collisions.
        const auto* bytes = reinterpret_cast<const unsigned char*>(&instance.transform);
        for (std::size_t byte = 0; byte < sizeof(instance.transform); ++byte) {
            revision << static_cast<unsigned>(bytes[byte]) << ',';
        }
    }
    result.revision = revision.str();
    return result;
}

bool VulkanBackendAdapter::queueAcousticGpu(audio::AcousticGpuBatch batch,
    const std::string& revision) {
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    if (!m_device || !m_device->isInitialized()) {
        return false;
    }
    if (!m_device->acousticPass) {
        m_device->acousticPass = std::make_shared<VulkanRT::AcousticGpuPass>(*m_device);
    }
    batch.stillCurrent = [this, revision] {
        const auto current = acousticGpuGeometry();
        return current.error.empty() && current.revision == revision;
    };
    std::string directory = m_rayFusionShaderDir;
    if (directory.empty()) {
        for (const auto* candidate : {"shaders", "source/shaders", "../shaders"}) {
            if (std::filesystem::exists(std::filesystem::path(candidate) / "acoustic_segment.spv")) {
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
    const bool queued = m_device->acousticPass->enqueue(std::move(batch), directory);
    if (queued && !shouldUseInteractiveViewport() && m_currentSamples >= m_targetSamples &&
        !m_tonemapRefreshPending) {
        // The main loop stops requesting progressive frames after convergence.
        // Reuse a ready frame slot directly, without waking a full image render.
        serviceAcousticGpuIdle();
    }
    return queued;
}

bool VulkanBackendAdapter::pollAcousticGpu(audio::AcousticGpuCompletion& result) {
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    return m_device && m_device->acousticPass && m_device->acousticPass->take(result);
}

void VulkanBackendAdapter::cancelAcousticGpu() {
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return;
    }
    if (m_device && m_device->acousticPass) {
        m_device->acousticPass->cancel();
    }
}

std::string VulkanBackendAdapter::acousticGpuError() const {
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return "Active renderer is busy; acoustic query deferred";
    }
    return m_device && m_device->acousticPass ? m_device->acousticPass->error() :
        "Acoustic GPU pass is unavailable";
}

bool VulkanBackendAdapter::acousticGpuPending() const {
    std::unique_lock<std::recursive_mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    return m_device && m_device->acousticPass && m_device->acousticPass->hasPending();
}
} // namespace Backend
