#pragma once

#include <vulkan/vulkan.h>
#include <string>

namespace Backend {

inline constexpr uint32_t kRayFusionFrameUniformBinding = 28;

struct RayFusionDrawPush {
    float viewProj[16];
    float cameraPos[4];
    float lightDir0[4];
    float lightDir1[4];
    uint32_t materialMeta[4];
};
static_assert(sizeof(RayFusionDrawPush) == 128);

std::string resolveRayFusionShaderDirectory();

VkResult selectRayFusionInstanceVersion(PFN_vkGetInstanceProcAddr getProc, uint32_t& version);
const char* rayFusionDeviceRejection(
    VkPhysicalDevice device, const VkPhysicalDeviceProperties& properties);
// "<adapter>: <reason>" for every adapter skipped by the last device selection, joined by
// "; ". Empty when nothing was rejected. Lets the Player say WHY it has no viewport on screen
// instead of only in the log (Android A6a: no silent black screen on an unsupported GPU).
void recordRayFusionDeviceRejection(const char* adapter, const char* reason);
#if defined(__ANDROID__) && defined(RT_PLAYER)
void recordAndroidRayFusionDeviceRejection(const VkPhysicalDeviceProperties& props,
                                          const char* reason);
#endif
void clearRayFusionDeviceRejections();
std::string rayFusionDeviceRejections();
void logRayFusionDeviceLimits(
    VkPhysicalDevice device, const VkPhysicalDeviceLimits& limits);

uint32_t rayFusionMaterialTextureCapacity(
    const VkPhysicalDeviceLimits& limits,
    const VkDescriptorSetLayoutBinding* bindings,
    uint32_t bindingCount,
    uint32_t requestedCapacity);

inline constexpr VkFormat kRfGbufferNormalFormat = VK_FORMAT_R16G16_SFLOAT;
inline constexpr VkFormat kRfGbufferSpecularFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

bool ensureRayFusionViewportPasses(
    VkDevice vkDevice,
    VkRenderPass& renderPass,
    VkRenderPass& renderPassLoad,
    VkRenderPass& hdrRenderPass,
    VkRenderPass& hdrRenderPassLoad,
    bool storeReflectionGbuffer = true);

} // namespace Backend
