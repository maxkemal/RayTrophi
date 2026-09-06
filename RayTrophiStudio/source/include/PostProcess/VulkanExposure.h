#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
namespace rtpost {
// Slots are owned by VulkanDevice's existing fences. Never read a mapped slot
// until its fence completes; never rewrite its descriptor while in flight.
class VulkanExposure {
public:
    bool initialize(VkDevice,VkPhysicalDevice,const std::vector<uint32_t>&);
    void shutdown(); // owner has drained all frame fences
    void record(VkCommandBuffer,VkImageView,unsigned slot);
    void consume(unsigned slot); // owner guarantees fence completion
private:
    VkDevice device=VK_NULL_HANDLE;
    VkDescriptorPool pool=VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout=VK_NULL_HANDLE;
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkPipeline pipeline=VK_NULL_HANDLE;
    struct Slot {
        VkBuffer buffer=VK_NULL_HANDLE;
        VkDeviceMemory memory=VK_NULL_HANDLE;
        VkDescriptorSet set=VK_NULL_HANDLE;
        void* mapped=nullptr;
        uint64_t generation=0;
        bool pending=false;
    } slots[2];
};
}
