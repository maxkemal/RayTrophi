// Texture-page compaction hook of VulkanBackendAdapter (doku dizisi C modeli).
// Moves every 2D material texture into a layer of a shared 2D-array page and
// rewrites every descriptor that named the old standalone image view.
// Design: docs/dev/MATERIAL_TEXTURE_ARRAYS.md, "C MODELI — KESIN TASARIM".

#include "Backend/VulkanBackend.h"
#include "Backend/VulkanMaterialTextureBinding.h"
#include "Backend/VulkanMaterialTextureStorage.h"

#include <algorithm>
#include <string>

namespace Backend {

void VulkanBackendAdapter::compactMaterialTexturesLocked() {
    if (!m_device || !m_device->isInitialized() || m_uploadedImages.empty()) return;
    if (!m_textureStorage) {
        m_textureStorage = std::make_unique<Backend::MaterialTextureStorage::Storage>();
    }

    // The matcap is held by value without its id; per-texture samplers survive
    // compaction unchanged, so the sampler identifies it afterwards.
    const VkSampler matcapSampler = m_interactiveViewport.matcapImage.sampler;

    const auto& caps = m_device->getCapabilities();
    const VkDevice vkDevice = m_device->getDevice();
    // Same policy as the per-texture samplers in uploadTexture2D.
    auto makeSampler = [&](uint32_t mipLevels) -> VkSampler {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        if (caps.supportsSamplerAnisotropy && mipLevels > 1) {
            sci.anisotropyEnable = VK_TRUE;
            sci.maxAnisotropy = std::clamp(caps.maxSamplerAnisotropy, 1.0f, 8.0f);
        } else {
            sci.anisotropyEnable = VK_FALSE;
            sci.maxAnisotropy = 1.0f;
        }
        sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.minLod = 0.0f;
        sci.maxLod = static_cast<float>(mipLevels - 1);
        VkSampler sampler = VK_NULL_HANDLE;
        vkCreateSampler(vkDevice, &sci, nullptr, &sampler);
        return sampler;
    };

    // Descriptors queued before the RT set existed still name the old views.
    m_device->clearPendingRTTextureDescriptors();

    const Backend::MaterialTextureStorage::CompactionReport report =
        m_textureStorage->compact(*m_device, m_uploadedImages, makeSampler,
                                  [this](int64_t id) {
                                      // The world environment is bound to RT/viewport env
                                      // slots outside the material table; keep it standalone.
                                      return id == m_envTexID;
                                  });

    SCENE_LOG_INFO("[TexturePages] compaction result=" + std::to_string(report.result) +
                   " packed=" + std::to_string(report.texturesPacked) +
                   " skipped=" + std::to_string(report.texturesSkipped) +
                   " pages=" + std::to_string(report.pages) +
                   " over_viewport_budget=" + std::to_string(report.pagesOverViewportBudget) +
                   " allocated_mb=" + std::to_string(report.allocatedBytes >> 20));

    // Rewrite every slot even on failure: pending descriptors were cleared above
    // and the handles are either unchanged (failure) or swapped (success).
    for (const auto& [id, img] : m_uploadedImages) {
        if (id <= 0 || !img.view || !img.sampler) continue;
        const uint32_t slot = static_cast<uint32_t>(id);
        if (m_device->hasHardwareRT()) {
            m_device->updateRTTextureDescriptor(slot, img);
        }
    }
    // Material preview set: 32 pages + slot->ref table, rewritten as a whole
    // (the device is idle). The RayFusion passes follow the generation stamp.
    // Viewport slot->ref table follows the (possibly unchanged) placement. The
    // device is idle here (compact waited), so the old table can go.
    m_textureStorage->ensureDescriptorResources(*m_device);
    m_textureStorage->rebuildRefTable(*m_device, m_uploadedImages);
    bool previewSetRewritten = false;
    if (m_interactiveViewport.materialPreviewDescSet != VK_NULL_HANDLE) {
        previewSetRewritten = Backend::MaterialTextureBinding::writeAll(*m_device, m_textureStorage,
                                                  m_interactiveViewport.materialPreviewDescSet,
                                                  1, 29);
    }
    ++m_textureCacheGeneration;
    // The preview set was rewritten above against this generation: stamp it so
    // the stale-set tripwire (renderInteractiveViewportImpl) does not rebuild it
    // again or report a purge that did not happen.
    if (previewSetRewritten) {
        m_interactiveViewport.materialPreviewDescSetTextureGeneration = m_textureCacheGeneration;
    }
    m_interactiveViewport.dirty = true;
    if (report.result != VK_SUCCESS || report.texturesPacked == 0) return;

    if (matcapSampler) {
        for (const auto& [id, img] : m_uploadedImages) {
            if (img.sampler != matcapSampler) continue;
            m_interactiveViewport.matcapImage = img;
            if (m_interactiveViewport.matcapDescSet != VK_NULL_HANDLE) {
                VkDescriptorImageInfo di{img.sampler, img.view,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                wds.dstSet = m_interactiveViewport.matcapDescSet;
                wds.dstBinding = 0;
                wds.descriptorCount = 1;
                wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                wds.pImageInfo = &di;
                vkUpdateDescriptorSets(vkDevice, 1, &wds, 0, nullptr);
            }
            break;
        }
    }

    // Every other viewport consumer (material preview set, env slots, RayFusion
    // passes, IBL) re-resolves from m_uploadedImages when the generation stamp
    // (bumped above) moves.
}

} // namespace Backend
