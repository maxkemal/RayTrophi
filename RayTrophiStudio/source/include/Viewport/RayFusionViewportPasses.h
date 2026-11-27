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
inline constexpr VkFormat kRfObjectMotionFormat = VK_FORMAT_R16G16_SFLOAT;

void recordRayFusionFrameUniform(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    const float viewColMajor[16],
    const float curViewProjColMajor[16],
    const float prevViewProjColMajor[16],
    bool hasPrevViewProj);

struct RasterPostFastResources {
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

bool ensureRasterPostFastResources(
    VkDevice device,
    const std::string& shaderDir,
    uint32_t pushConstantSize,
    RasterPostFastResources& out);

void updateRasterPostFastDescriptor(
    VkDevice device,
    const RasterPostFastResources& res,
    VkSampler sampler,
    VkImageView hdrView);

void destroyRasterPostFastResources(
    VkDevice device,
    RasterPostFastResources& res);

bool ensureRayFusionViewportPasses(
    VkDevice vkDevice,
    VkRenderPass& renderPass,
    VkRenderPass& renderPassLoad,
    VkRenderPass& hdrRenderPass,
    VkRenderPass& hdrRenderPassLoad,
    bool storeReflectionGbuffer = true,
    bool storePreviousClip = true);

VkRenderPass createHdrNoMotionRenderPass(
    VkDevice device,
    bool storeReflectionGbuffer,
    bool loadColor);

} // namespace Backend
