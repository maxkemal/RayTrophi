#pragma once

#include <string>
#include <vulkan/vulkan.h>
#include <ffx_api/ffx_api.h>

namespace VulkanRT {

// Stack-owned until vkCreateDevice returns. Optional core 1.2 features mirror
// capabilities queried by AMD's Vulkan backend; physical support alone is not
// permission to emit shaders using a feature on our logical device.
struct FsrDeviceFeatures {
    VkPhysicalDeviceShaderFloat16Int8Features arithmetic{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures storage{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures subgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES};
    void append(VkPhysicalDevice physical, void**& nextLink, bool legacy);
};

bool enableFsrImageFeatures(const VkPhysicalDeviceFeatures& supported,
                           VkPhysicalDeviceFeatures& enabled, bool legacy);

class VulkanFsrLoader {
public:
    ~VulkanFsrLoader();
    bool open(std::string& reason);
    bool loaded() const {
        return m_module != nullptr;
    }
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
private:
    void* m_module = nullptr;
};
} // namespace VulkanRT
