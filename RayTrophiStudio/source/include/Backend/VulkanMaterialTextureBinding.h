#pragma once

// Single home of the VIEWPORT backend's material-texture descriptor ABI.
// Every viewport pass that samples material textures declares:
//   - a binding of 32 COMBINED_IMAGE_SAMPLER descriptors, one sampler2DArray
//     page each (shaders: materialTexturePages[32], MATERIAL_TEXTURE_BINDING);
//   - a STORAGE_BUFFER slot->ref table (shaders: MATERIAL_TEXTURE_REF_BINDING)
//     that maps the RT material slot id stored in the shared material buffer to
//     a packed (page << 16 | layer) reference.
// No descriptor indexing: all 32 descriptors are always written (dummy page for
// unused ones) and shaders select a page with a constant-index switch.
// The RT render backend (VulkanBackend.cpp, binding 6) stays bindless and is
// intentionally NOT routed through here. Shader side: material_texture_arrays.glsl.

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <memory>

namespace VulkanRT { class VulkanDevice; }
namespace Backend::MaterialTextureStorage { class Storage; }

namespace Backend::MaterialTextureBinding {

constexpr uint32_t kPageCount = 32;

// Core features of the RayFusion device query (independentBlend etc. are
// checked by the caller). The texture ABI itself needs no optional feature.
void queryCoreFeatures(VkPhysicalDevice device, VkPhysicalDeviceFeatures& core);

VkDescriptorSetLayoutBinding pagesBinding(uint32_t binding, VkShaderStageFlags stages);
VkDescriptorSetLayoutBinding refTableBinding(uint32_t binding, VkShaderStageFlags stages);

// Combined-sampler pool contribution: the 32 pages plus `extraSamplers` others.
// The ref table adds one STORAGE_BUFFER descriptor to the caller's buffer count.
VkDescriptorPoolSize pagesPoolSize(uint32_t extraSamplers);

struct Descriptors {
    std::array<VkDescriptorImageInfo, kPageCount> pages{};
    VkDescriptorBufferInfo refTable{};
};

// Fills the descriptors from the adapter's texture storage, creating the storage
// and its dummy page / default ref table on first use. False when the device
// cannot provide them (the caller must not draw with the material set then).
bool collect(VulkanRT::VulkanDevice& device,
             std::unique_ptr<MaterialTextureStorage::Storage>& storage,
             Descriptors& out);

// Two writes (pages, ref table); `d` must outlive vkUpdateDescriptorSets.
void makeWrites(VkDescriptorSet set, uint32_t pagesBindingIndex, uint32_t refBindingIndex,
                const Descriptors& d, VkWriteDescriptorSet out[2]);

// collect + makeWrites + vkUpdateDescriptorSets. Returns false (and writes
// nothing) when collect fails.
bool writeAll(VulkanRT::VulkanDevice& device,
              std::unique_ptr<MaterialTextureStorage::Storage>& storage,
              VkDescriptorSet set, uint32_t pagesBindingIndex, uint32_t refBindingIndex);

} // namespace Backend::MaterialTextureBinding
