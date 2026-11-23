#include "Backend/VulkanMaterialTextureBinding.h"
#include "Backend/VulkanBackend.h"
#include "Backend/VulkanMaterialTextureStorage.h"

namespace Backend::MaterialTextureBinding {

void queryCoreFeatures(VkPhysicalDevice device, VkPhysicalDeviceFeatures& core) {
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    vkGetPhysicalDeviceFeatures2(device, &features);
    core = features.features;
}

VkDescriptorSetLayoutBinding pagesBinding(uint32_t binding, VkShaderStageFlags stages) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = kPageCount;
    b.stageFlags = stages;
    return b;
}

VkDescriptorSetLayoutBinding refTableBinding(uint32_t binding, VkShaderStageFlags stages) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = stages;
    return b;
}

VkDescriptorPoolSize pagesPoolSize(uint32_t extraSamplers) {
    return {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kPageCount + extraSamplers};
}

bool collect(VulkanRT::VulkanDevice& device,
             std::unique_ptr<MaterialTextureStorage::Storage>& storage,
             Descriptors& out) {
    if (!storage) {
        storage = std::make_unique<MaterialTextureStorage::Storage>();
    }
    if (!storage->ensureDescriptorResources(device)) {
        return false;
    }
    storage->pageDescriptorImages(out.pages);
    out.refTable = storage->refTableDescriptor();
    return true;
}

void makeWrites(VkDescriptorSet set, uint32_t pagesBindingIndex, uint32_t refBindingIndex,
                const Descriptors& d, VkWriteDescriptorSet out[2]) {
    out[0] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    out[0].dstSet = set;
    out[0].dstBinding = pagesBindingIndex;
    out[0].descriptorCount = kPageCount;
    out[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    out[0].pImageInfo = d.pages.data();
    out[1] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    out[1].dstSet = set;
    out[1].dstBinding = refBindingIndex;
    out[1].descriptorCount = 1;
    out[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    out[1].pBufferInfo = &d.refTable;
}

bool writeAll(VulkanRT::VulkanDevice& device,
              std::unique_ptr<MaterialTextureStorage::Storage>& storage,
              VkDescriptorSet set, uint32_t pagesBindingIndex, uint32_t refBindingIndex) {
    Descriptors d;
    if (!set || !collect(device, storage, d)) return false;
    VkWriteDescriptorSet w[2];
    makeWrites(set, pagesBindingIndex, refBindingIndex, d, w);
    vkUpdateDescriptorSets(device.getDevice(), 2, w, 0, nullptr);
    return true;
}

} // namespace Backend::MaterialTextureBinding
