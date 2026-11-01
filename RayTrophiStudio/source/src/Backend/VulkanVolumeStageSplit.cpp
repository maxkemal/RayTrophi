#include "Backend/VulkanVolumeStageSplit.h"
#include "Backend/VulkanBackend.h"
#include "VDBVolume.h"
#include "GasVolume.h"
#include "globals.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <array>
#include <sstream>
#include <utility>

namespace {

std::vector<std::uint32_t> readSplitShader(const std::string& path) {
    std::ifstream input(path, std::ios::ate | std::ios::binary);
    if (!input) {
        return {};
    }
    const std::streamoff length = input.tellg();
    if (length < 20 || length % 4 != 0) {
        return {};
    }
    std::vector<std::uint32_t> words(static_cast<std::size_t>(length) / 4);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(words.data()), length) ||
        words.front() != 0x07230203u) {
        return {};
    }
    return words;
}

std::uint32_t splitShaderStamp(const std::vector<std::uint32_t>& words) {
    std::uint32_t hash = 2166136261u;
    for (const auto word : words) {
        hash = (hash ^ word) * 16777619u;
    }
    return hash;
}

bool splitManifestMatches(const std::string& directory,
    const std::vector<std::uint32_t>& raygen, const std::vector<std::uint32_t>& fog,
    const std::vector<std::uint32_t>& surface) {
    std::ifstream manifest(directory + "/volume_split_stage_set.txt");
    std::string version;
    if (!(manifest >> version) || version != "volume-split-v1") {
        return false;
    }
    const std::array<const char*, 3> names = {
        "raygen.spv", "volume_fog_closesthit.spv", "volume_surface_closesthit.spv"
    };
    const std::array<const std::vector<std::uint32_t>*, 3> modules = { &raygen, &fog, &surface };
    for (std::size_t index = 0; index < modules.size(); ++index) {
        std::string name, digest;
        std::uint64_t bytes = 0;
        if (!(manifest >> name >> bytes >> digest) || name != names[index] || digest.size() != 8 ||
            bytes != modules[index]->size() * sizeof(std::uint32_t)) {
            return false;
        }
        std::uint32_t expected = 0;
        std::istringstream parser(digest);
        if (!(parser >> std::hex >> expected) || !parser.eof() ||
            expected != splitShaderStamp(*modules[index])) {
            return false;
        }
    }
    std::string extra;
    return !(manifest >> extra);
}

} // namespace

namespace VulkanRT {

bool configureSplitVolumeShaders(const std::string& shaderDirectory,
    const std::vector<std::uint32_t>& raygen,
    std::vector<std::uint32_t>& fog, std::vector<std::uint32_t>& surface) {
    static const bool requested = [] {
        const char* value = std::getenv("RT_VK_SPLIT_VOLUME");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!requested) {
        return false;
    }
    auto fogStage = readSplitShader(shaderDirectory + "/volume_fog_closesthit.spv");
    auto surfaceStage = readSplitShader(shaderDirectory + "/volume_surface_closesthit.spv");
    if (raygen.empty() || fog.empty() || fogStage.empty() || surfaceStage.empty() ||
        !splitManifestMatches(shaderDirectory, raygen, fogStage, surfaceStage)) {
        SCENE_LOG_WARN("[Vulkan] Split volume stage set missing/mixed; keeping legacy shader.");
        return false;
    }
    fog = std::move(fogStage);
    surface = std::move(surfaceStage);
    SCENE_LOG_INFO("[Vulkan] Split volume stages selected (fog + SurfaceSDF).");
    return true;
}

std::uint32_t VulkanDevice::getVolumeRoutedSbtOffset(
    std::uint32_t mask, std::uint32_t original) const {
    return routedVolumeSbtOffset(m_hasSurfaceVolumeShader, m_hasVolumeShaders,
        m_hasHairShaders, m_hasSphereShaders, mask, original);
}

} // namespace VulkanRT

namespace Backend {

bool VulkanBackendAdapter::refreshVolumeStageMask(std::size_t instanceIndex) {
    if (!m_device || instanceIndex >= m_instanceSources.size() ||
        instanceIndex >= m_vkInstances.size()) {
        return false;
    }
    bool surface = false;
    if (const auto volume = std::dynamic_pointer_cast<VDBVolume>(m_instanceSources[instanceIndex])) {
        surface = volume->render_as_isosurface;
    } else if (!std::dynamic_pointer_cast<GasVolume>(m_instanceSources[instanceIndex])) {
        return false;
    }
    auto& instance = m_vkInstances[instanceIndex];
    const std::uint32_t mask = instance.mask == 0 ? 0u : (surface ? 0x08u : 0x02u);
    if (instance.mask == mask) {
        return false;
    }
    instance.mask = mask;
    if (instanceIndex < m_pendingGpuInstanceSources.size()) {
        auto& source = m_pendingGpuInstanceSources[instanceIndex];
        source.metadata[1] = mask;
        source.metadata[2] = m_device->getVolumeRoutedSbtOffset(mask, instance.sbtRecordOffset);
    }
    return true;
}

void VulkanBackendAdapter::refreshVolumeStageRouting() {
    if (!m_device || m_orderedVDBInstances.empty()) {
        return;
    }
    for (std::size_t index = 0; index < m_vkInstances.size(); ++index) {
        refreshVolumeStageMask(index);
    }
    // The CPU and GPU upload paths normalize offsets from the reserved domain
    // mask. Invalidate no transform caches; patch already-prepared GPU metadata.
    for (auto& source : m_pendingGpuInstanceSources) {
        source.metadata[2] = m_device->getVolumeRoutedSbtOffset(
            source.metadata[1], source.metadata[2]);
    }
    auto merged = m_vkInstances;
    merged.insert(merged.end(), m_hairVkInstances.begin(), m_hairVkInstances.end());
    m_device->waitIdle();
    m_device->updateTLAS(merged);
}

} // namespace Backend
