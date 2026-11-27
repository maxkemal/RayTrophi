#include "Viewport/PlayerSwapchainPresenter.h"

#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))

#include "Backend/VulkanBackend.h"
#include "Backend/VulkanFailureReporting.h"
#include "Backend/VulkanGpuBreadcrumbs.h"
#include "PerfProfile.h"
#include "imgui/imgui_impl_vulkan.h"
#include "imgui/imgui_internal.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>

namespace AppRuntime {
namespace {

bool check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) {
        return true;
    }
    VulkanRT::reportVulkanDeviceFailure(result, operation);
    SDL_LogError(SDL_LOG_CATEGORY_RENDER, "%s: VkResult %d", operation, result);
    return false;
}

void imageBarrier(VkCommandBuffer command, VkImage image,
                  VkImageLayout before, VkImageLayout after,
                  VkAccessFlags sourceAccess, VkAccessFlags destinationAccess,
                  VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = sourceAccess;
    barrier.dstAccessMask = destinationAccess;
    barrier.oldLayout = before;
    barrier.newLayout = after;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, sourceStage, destinationStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace

struct PlayerSwapchainPresenter::State {
    struct Frame {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkSemaphore acquired = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
    };

    struct Target {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        // Reacquiring this image guarantees presentation consumed this semaphore.
        VkSemaphore rendered = VK_NULL_HANDLE;
    };

    SDL_Window* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkRenderPass renderPass = VK_NULL_HANDLE;
    // The UI render pass and the ImGui Vulkan backend outlive swapchain
    // rebuilds; they are recreated only when the surface format changes.
    VkFormat renderPassFormat = VK_FORMAT_UNDEFINED;
    VkCommandPool pool = VK_NULL_HANDLE;
    std::array<Frame, 2> frames{};
    std::vector<Target> targets;
    std::vector<std::unique_ptr<ImTextureData>> uiTextures;
    size_t nextFrame = 0;
    bool retiredUiTextures = false;
    VkExtent2D lastDrawable{};
    uint64_t frameSerial = 0;
    VkImage source = VK_NULL_HANDLE;
    VkExtent2D sourceExtent{};
    bool imguiReady = false;
    bool rebuild = true;
    // preTransform deliberately differs from currentTransform (see
    // createSwapchain): VK_SUBOPTIMAL_KHR then reports that, not a resize.
    bool transformMismatch = false;
    bool suspended = false;
    bool surfaceLost = false;
    bool failed = false;

    void clearSwapchain();

    void releaseUiRenderer();
    void retireUiTextures();
    bool createSwapchain(VkExtent2D drawable);
    void record(VkCommandBuffer command, const Target& target);
};

PlayerSwapchainPresenter& PlayerSwapchainPresenter::instance() {
    static PlayerSwapchainPresenter presenter;
    return presenter;
}

PlayerSwapchainPresenter::State& PlayerSwapchainPresenter::state() {
    static State data;
    return data;
}

bool PlayerSwapchainPresenter::appendInstanceExtensions(
    SDL_Window* window, std::vector<const char*>& extensions) {
    unsigned count = 0;
    if (!window || !SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr)) {
        return false;
    }
    const size_t offset = extensions.size();
    extensions.resize(offset + count);
    if (!SDL_Vulkan_GetInstanceExtensions(window, &count, extensions.data() + offset)) {
        return false;
    }
    extensions.resize(offset + count);
    state().window = window;
    return true;
}

bool PlayerSwapchainPresenter::owns(VkInstance instance) const {
    return instance && state().instance == instance;
}

bool PlayerSwapchainPresenter::createSurface(VkInstance instance) {
    auto& s = state();
    s.instance = instance;
    return SDL_Vulkan_CreateSurface(s.window, instance, &s.surface) == SDL_TRUE;
}

bool PlayerSwapchainPresenter::supportsDevice(VkPhysicalDevice physical) const {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr) !=
        VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()) !=
        VK_SUCCESS || std::none_of(extensions.begin(), extensions.end(), [](const auto& ext) {
            return std::string_view(ext.extensionName) == VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        })) {
        return false;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    constexpr auto required = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueCount && (families[i].queueFlags & required) == required &&
            supportsPresent(physical, i)) {
            return true;
        }
    }
    return false;
}

bool PlayerSwapchainPresenter::supportsPresent(VkPhysicalDevice physical, uint32_t family) const {
    VkBool32 supported = VK_FALSE;
    const auto surface = state().surface;
    return surface && vkGetPhysicalDeviceSurfaceSupportKHR(
        physical, family, surface, &supported) == VK_SUCCESS && supported;
}

bool PlayerSwapchainPresenter::attach(VulkanRT::VulkanDevice& engine) {
    auto& s = state();
    s.physical = engine.getPhysicalDevice();
    s.device = engine.getDevice();
    s.queue = engine.getComputeQueue();
    s.family = engine.getComputeQueueFamily();

    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = s.family;
    if (!check(vkCreateCommandPool(s.device, &pool, nullptr, &s.pool), "Player command pool")) {
        return false;
    }
    for (auto& frame : s.frames) {
        VkCommandBufferAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocation.commandPool = s.pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkSemaphoreCreateInfo semaphore{};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (!check(vkAllocateCommandBuffers(s.device, &allocation, &frame.command),
                   "Player command buffer") ||
            !check(vkCreateSemaphore(s.device, &semaphore, nullptr, &frame.acquired),
                   "Player acquire semaphore") ||
            !check(vkCreateFence(s.device, &fence, nullptr, &frame.fence), "Player fence")) {
            return false;
        }
    }
    int width = 0;
    int height = 0;
    SDL_Vulkan_GetDrawableSize(s.window, &width, &height);
    return width <= 0 || height <= 0 || s.createSwapchain(
        {static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
}

void PlayerSwapchainPresenter::State::clearSwapchain() {
    // ★★★ Swapchain objects only. The ImGui backend stays initialised: in this
    //   ImGui (docking) ImGui_ImplVulkan_Shutdown() -> DestroyPlatformWindows()
    //   NULLs the main viewport's PlatformHandle, and imgui_impl_sdl2 drops every
    //   mouse/keyboard event whose window ID no longer maps to a viewport.
    //   2026-11-23: the first resize / fullscreen toggle left the F2 console
    //   visible but deaf to all input, in windowed mode too, for good.
    for (const auto& target : targets) {
        vkDestroyFramebuffer(device, target.framebuffer, nullptr);
        vkDestroyImageView(device, target.view, nullptr);
        vkDestroySemaphore(device, target.rendered, nullptr);
    }
    targets.clear();
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
}

void PlayerSwapchainPresenter::State::releaseUiRenderer() {
    if (imguiReady) {
        // Shutdown destroys the platform windows and with them the main
        // viewport's window-ID handle (see clearSwapchain). Put it back so a
        // later re-init (format change, device-loss rebuild) still gets input.
        ImGuiViewport* main = ImGui::GetMainViewport();
        void* handle = main->PlatformHandle;
        void* handleRaw = main->PlatformHandleRaw;
        ImGui_ImplVulkan_Shutdown();
        main->PlatformHandle = handle;
        main->PlatformHandleRaw = handleRaw;
        imguiReady = false;
    }
    if (renderPass != VK_NULL_HANDLE) vkDestroyRenderPass(device, renderPass, nullptr);
    renderPass = VK_NULL_HANDLE;
    renderPassFormat = VK_FORMAT_UNDEFINED;
}

void PlayerSwapchainPresenter::release(VkInstance instance) {
    auto& s = state();
    if (s.instance != instance || !instance) {
        return;
    }
    if (s.device) {
        if (!g_vulkan_device_lost) {
            check(vkQueueWaitIdle(s.queue), "Player presentation shutdown");
        }
        s.clearSwapchain();
        s.releaseUiRenderer();
        for (const auto& frame : s.frames) {
            vkDestroyFence(s.device, frame.fence, nullptr);
            vkDestroySemaphore(s.device, frame.acquired, nullptr);
        }
        vkDestroyCommandPool(s.device, s.pool, nullptr);
        for (const auto& texture : s.uiTextures) {
            ImGui::GetPlatformIO().Textures.find_erase(texture.get());
            ImGui::UnregisterUserTexture(texture.get());
        }
    }
    vkDestroySurfaceKHR(instance, s.surface, nullptr);
    s = State{};
}

ImTextureData* PlayerSwapchainPresenter::createUiTexture(int width, int height) {
    auto texture = std::make_unique<ImTextureData>();
    texture->Create(ImTextureFormat_RGBA32, width, height);
    std::memset(texture->Pixels, 0, texture->GetSizeInBytes());
    auto* result = texture.get();
    ImGui::RegisterUserTexture(result);
    state().uiTextures.push_back(std::move(texture));
    return result;
}

void PlayerSwapchainPresenter::updateUiTexture(
    ImTextureData& texture, const void* rgba, int width, int height, int pitch) {
    const auto* source = static_cast<const unsigned char*>(rgba);
    for (int y = 0; y < height; ++y) {
        std::memcpy(texture.GetPixelsAt(0, y), source + y * pitch, width * 4);
    }
    const ImTextureRect rect{0, 0, static_cast<unsigned short>(width),
                             static_cast<unsigned short>(height)};
    if (texture.Status == ImTextureStatus_OK || texture.Status == ImTextureStatus_WantUpdates) {
        texture.Updates.resize(0);
        texture.Updates.push_back(rect);
        texture.UpdateRect = rect;
        texture.SetStatus(ImTextureStatus_WantUpdates);
    }
}

void PlayerSwapchainPresenter::destroyUiTexture(ImTextureData* texture) {
    if (!texture) {
        return;
    }
    // Retain the ImTextureData until ImGui has retired any draw commands using it.
    texture->WantDestroyNextFrame = true;
    state().retiredUiTextures = true;
}

bool PlayerSwapchainPresenter::active() const {
    const auto& s = state();
    return s.device && s.pool;
}

bool PlayerSwapchainPresenter::failed() const {
    return state().failed;
}

bool PlayerSwapchainPresenter::needsRedraw() const {
    const auto& s = state();
    return active() && !s.failed && !g_vulkan_device_lost && s.rebuild;
}

uint64_t PlayerSwapchainPresenter::frameSerial() const {
    return state().frameSerial;
}

void PlayerSwapchainPresenter::publish(const VulkanRT::ImageHandle& image) {
    auto& s = state();
    ++s.frameSerial;
    s.source = image.image;
    s.sourceExtent = {image.width, image.height};
}

void PlayerSwapchainPresenter::discard(VkImage image) {
    auto& s = state();
    if (!image || s.source != image) {
        return;
    }
    s.source = VK_NULL_HANDLE;
    if (!g_vulkan_device_lost) {
        check(vkQueueWaitIdle(s.queue), "Player viewport image retirement");
    }
}

void PlayerSwapchainPresenter::State::retireUiTextures() {
    if (!retiredUiTextures || !imguiReady || g_vulkan_device_lost) {
        return;
    }
    if (!check(vkQueueWaitIdle(queue), "Player UI texture retirement")) {
        return;
    }
    auto retired = std::remove_if(uiTextures.begin(), uiTextures.end(), [&](const auto& tex) {
        if (!tex->WantDestroyNextFrame) {
            return false;
        }
        tex->UnusedFrames = static_cast<int>(targets.size());
        tex->SetStatus(ImTextureStatus_WantDestroy);
        ImGui_ImplVulkan_UpdateTexture(tex.get());
        ImGui::GetPlatformIO().Textures.find_erase(tex.get());
        ImGui::UnregisterUserTexture(tex.get());
        return true;
    });
    uiTextures.erase(retired, uiTextures.end());
    retiredUiTextures = false;
}

bool PlayerSwapchainPresenter::prepareUiFrame() {
    auto& s = state();
    if (!active() || s.failed || s.suspended || g_vulkan_device_lost) {
        return false;
    }
    // attach() may have deferred the swapchain when Android's native window
    // had no drawable extent. present() takes its rebuild branch in that case.
    if (!s.imguiReady) {
        present();
    }
    return s.imguiReady && !s.failed && !g_vulkan_device_lost;
}

void PlayerSwapchainPresenter::newFrame() {
    auto& s = state();
    s.retireUiTextures();
    if (s.imguiReady) {
        ImGui_ImplVulkan_NewFrame();
    }
}

bool PlayerSwapchainPresenter::State::createSwapchain(VkExtent2D drawable) {
    VkSurfaceCapabilitiesKHR caps{};
    const VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps);
    if (result == VK_ERROR_SURFACE_LOST_KHR) {
        surfaceLost = true;
        return true;
    }
    if (!check(result, "Player surface capabilities")) {
        return false;
    }
    lastDrawable = drawable;
    extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width = std::clamp(drawable.width, caps.minImageExtent.width,
                                 caps.maxImageExtent.width);
        extent.height = std::clamp(drawable.height, caps.minImageExtent.height,
                                  caps.maxImageExtent.height);
    }
    if (!extent.width || !extent.height) {
        return true;
    }
    constexpr VkImageUsageFlags usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((caps.supportedUsageFlags & usage) != usage) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Player swapchain lacks transfer/color usage");
        return false;
    }

    uint32_t count = 0;
    if (!check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, nullptr),
               "Player surface format count") || !count) {
        return false;
    }
    std::vector<VkSurfaceFormatKHR> formats(count);
    if (!check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, formats.data()),
               "Player surface formats")) {
        return false;
    }

    // The raster output already contains display-encoded values. A UNORM
    // destination preserves them while the blit handles RGBA/BGRA ordering.
    VkSurfaceFormatKHR format{};
    const auto supported = [&](VkFormat desired) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physical, desired, &props);
        return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
    };
    for (const auto& candidate : formats) {
        if (candidate.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            continue;
        }
        const VkFormat desired = candidate.format == VK_FORMAT_UNDEFINED
            ? VK_FORMAT_B8G8R8A8_UNORM : candidate.format;
        if ((desired == VK_FORMAT_B8G8R8A8_UNORM || desired == VK_FORMAT_R8G8B8A8_UNORM)
            && supported(desired)) {
            format = {desired, candidate.colorSpace};
            break;
        }
    }
    VkFormatProperties sourceProps{};
    vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_R8G8B8A8_UNORM, &sourceProps);
    if (format.format == VK_FORMAT_UNDEFINED ||
        !(sourceProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Player needs a blit-capable UNORM swapchain");
        return false;
    }

    const uint32_t minimum = (std::max)(2u, caps.minImageCount);
    if (caps.maxImageCount && caps.maxImageCount < minimum) {
        return false;
    }
    VkSwapchainCreateInfoKHR create{};
    create.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create.surface = surface;
    create.minImageCount = minimum;
    create.imageFormat = format.format;
    create.imageColorSpace = format.colorSpace;
    create.imageExtent = extent;
    create.imageArrayLayers = 1;
    create.imageUsage = usage;
    create.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // ★ Landscape-locked game on phones whose natural orientation is portrait:
    //   currentTransform is ROTATE_90/270 and currentExtent the rotated window
    //   size. Matching it would mean rendering pre-rotated, which the blit and
    //   ImGui cannot do; IDENTITY lets the compositor rotate. Android then
    //   reports VK_SUBOPTIMAL_KHR, which must not trigger a rebuild per frame.
    create.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
        ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;
    transformMismatch = create.preTransform != caps.currentTransform;
    create.compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(
        caps.supportedCompositeAlpha & (0u - caps.supportedCompositeAlpha));
    create.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    uint32_t modeCount = 0;
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &modeCount, nullptr) ==
            VK_SUCCESS && modeCount) {
        std::vector<VkPresentModeKHR> modes(modeCount);
        if (vkGetPhysicalDeviceSurfacePresentModesKHR(
                physical, surface, &modeCount, modes.data()) == VK_SUCCESS) {
#ifdef __ANDROID__
            // Phones: vsync FIFO (battery, thermals); the Player paces itself.
            modes.clear();
#endif
            for (const auto preferred : {VK_PRESENT_MODE_MAILBOX_KHR,
                                         VK_PRESENT_MODE_IMMEDIATE_KHR}) {
                if (std::find(modes.begin(), modes.end(), preferred) != modes.end()) {
                    create.presentMode = preferred;
                    break;
                }
            }
        }
    }
    create.clipped = VK_TRUE;
    const VkResult created = vkCreateSwapchainKHR(device, &create, nullptr, &swapchain);
    if (created == VK_ERROR_OUT_OF_DATE_KHR || created == VK_ERROR_SURFACE_LOST_KHR) {
        surfaceLost = created == VK_ERROR_SURFACE_LOST_KHR;
        return true;
    }
    if (!check(created, "Player swapchain") ||
        !check(vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr),
               "Player swapchain image count")) {
        return false;
    }
    std::vector<VkImage> images(count);
    if (!check(vkGetSwapchainImagesKHR(device, swapchain, &count, images.data()),
               "Player swapchain images")) {
        return false;
    }

    if (renderPass != VK_NULL_HANDLE && renderPassFormat != format.format) {
        releaseUiRenderer();  // a new surface format needs a new pass and UI pipeline
    }
    if (renderPass == VK_NULL_HANDLE) {
        VkAttachmentDescription color{};
        color.format = format.format;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &reference;
        VkRenderPassCreateInfo pass{};
        pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        pass.attachmentCount = 1;
        pass.pAttachments = &color;
        pass.subpassCount = 1;
        pass.pSubpasses = &subpass;
        if (!check(vkCreateRenderPass(device, &pass, nullptr, &renderPass), "Player UI render pass")) {
            return false;
        }
        renderPassFormat = format.format;
    }
    targets.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        auto& target = targets[i];
        target.image = images[i];
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = target.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!check(vkCreateImageView(device, &view, nullptr, &target.view),
                   "Player swapchain image view")) {
            return false;
        }
        VkFramebufferCreateInfo framebuffer{};
        framebuffer.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer.renderPass = renderPass;
        framebuffer.attachmentCount = 1;
        framebuffer.pAttachments = &target.view;
        framebuffer.width = extent.width;
        framebuffer.height = extent.height;
        framebuffer.layers = 1;
        VkSemaphoreCreateInfo semaphore{};
        semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (!check(vkCreateFramebuffer(device, &framebuffer, nullptr, &target.framebuffer),
                   "Player framebuffer") ||
            !check(vkCreateSemaphore(device, &semaphore, nullptr, &target.rendered),
                   "Player present semaphore")) {
            return false;
        }
    }

    if (imguiReady) {
        rebuild = false;
        return true;
    }
    ImGui_ImplVulkan_InitInfo ui{};
    ui.ApiVersion = VK_API_VERSION_1_2;
    ui.Instance = instance;
    ui.PhysicalDevice = physical;
    ui.Device = device;
    ui.QueueFamily = family;
    ui.Queue = queue;
    ui.DescriptorPoolSize = 128;
    ui.MinImageCount = minimum;
    ui.ImageCount = count;
    ui.PipelineInfoMain.RenderPass = renderPass;
    ui.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ui.CheckVkResultFn = [](VkResult error) { check(error, "ImGui Vulkan"); };
    imguiReady = ImGui_ImplVulkan_Init(&ui);
    rebuild = false;
    return imguiReady;
}

void PlayerSwapchainPresenter::State::record(VkCommandBuffer command, const Target& target) {
    constexpr auto transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
    constexpr auto all = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    constexpr auto color = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    imageBarrier(command, target.image, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                 transfer, transfer);
    if (source) {
        imageBarrier(command, source, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT, all, transfer);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstSubresource = blit.srcSubresource;
        blit.srcOffsets[1] = {static_cast<int32_t>(sourceExtent.width),
                              static_cast<int32_t>(sourceExtent.height), 1};
        blit.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                              static_cast<int32_t>(extent.height), 1};
        vkCmdBlitImage(command, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                       VK_FILTER_NEAREST);
        imageBarrier(command, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, transfer, all);
    } else {
        VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(command, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &black, 1, &range);
    }
    imageBarrier(command, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                 transfer, color);
    VkRenderPassBeginInfo pass{};
    pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass.renderPass = renderPass;
    pass.framebuffer = target.framebuffer;
    pass.renderArea.extent = extent;
    vkCmdBeginRenderPass(command, &pass, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command);
    vkCmdEndRenderPass(command);
    imageBarrier(command, target.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                 0, color, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

bool PlayerSwapchainPresenter::present() {
    auto& s = state();
    if (!active() || s.failed || s.suspended || g_vulkan_device_lost) {
        return false;
    }
    int width = 0;
    int height = 0;
    SDL_Vulkan_GetDrawableSize(s.window, &width, &height);
    if (width <= 0 || height <= 0) {
        return false;
    }
    const VkExtent2D drawable{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    if (!s.swapchain || s.rebuild || drawable.width != s.lastDrawable.width ||
        drawable.height != s.lastDrawable.height) {
        if (!check(vkQueueWaitIdle(s.queue), "Player swapchain retirement")) {
            return false;
        }
        s.clearSwapchain();
        if (s.surfaceLost) {
            vkDestroySurfaceKHR(s.instance, s.surface, nullptr);
            s.surface = VK_NULL_HANDLE;
            s.surfaceLost = false;
            if (!createSurface(s.instance) || !supportsPresent(s.physical, s.family)) {
                s.failed = true;
                return false;
            }
        }
        s.failed = !s.createSwapchain(drawable);
        // The draw data belongs to the previous ImGui renderer. Let the next
        // UI frame build fresh draw data for the new renderer and extent.
        return false;
    }

    auto& frame = s.frames[s.nextFrame];
    VkResult fenceResult;
    {
        RTPERF_FRAME_SCOPE("player.swapchain.fence_wait");
        fenceResult = VulkanRT::waitFenceBounded(s.device, frame.fence, "Player wait for frame");
    }
    if (fenceResult != VK_SUCCESS) {
        s.failed = true;
        return false;
    }
    uint32_t index = 0;
    constexpr uint64_t acquireTimeoutNs = 16'000'000;
    VkResult acquired;
    {
        RTPERF_FRAME_SCOPE("player.swapchain.acquire");
        acquired = vkAcquireNextImageKHR(s.device, s.swapchain, acquireTimeoutNs,
                                         frame.acquired, VK_NULL_HANDLE, &index);
    }
    if (acquired == VK_TIMEOUT || acquired == VK_NOT_READY) {
        return false;
    }
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR || acquired == VK_ERROR_SURFACE_LOST_KHR) {
        s.rebuild = true;
        s.surfaceLost = acquired == VK_ERROR_SURFACE_LOST_KHR;
        return false;
    }
    if (acquired != VK_SUBOPTIMAL_KHR && !check(acquired, "Player acquire image")) {
        s.failed = true;
        return false;
    }
    s.rebuild = acquired == VK_SUBOPTIMAL_KHR && !s.transformMismatch;
    const auto& target = s.targets[index];

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!check(vkResetCommandBuffer(frame.command, 0), "Player reset commands") ||
        !check(vkBeginCommandBuffer(frame.command, &begin), "Player begin commands")) {
        s.failed = true;
        return false;
    }
    auto& crumbs = VulkanRT::rasterFrameBreadcrumbs();
    crumbs.beginSubmit();
    crumbs.mark(frame.command, "player_swapchain");
    s.record(frame.command, target);
    crumbs.mark(frame.command, std::string("player_swapchain ") + VulkanRT::kEndOfCommandBuffer);
    crumbs.commitSubmit();
    if (!check(vkEndCommandBuffer(frame.command), "Player end commands") ||
        !check(vkResetFences(s.device, 1, &frame.fence), "Player reset frame fence")) {
        s.failed = true;
        return false;
    }
    const VkPipelineStageFlags wait = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &frame.acquired;
    submit.pWaitDstStageMask = &wait;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &target.rendered;
    VkResult submitResult;
    {
        RTPERF_FRAME_SCOPE("player.swapchain.queue_submit");
        submitResult = vkQueueSubmit(s.queue, 1, &submit, frame.fence);
    }
    if (!check(submitResult, "Player submit presentation")) {
        s.failed = true;
        return false;
    }
    s.nextFrame = (s.nextFrame + 1) % s.frames.size();

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &target.rendered;
    present.swapchainCount = 1;
    present.pSwapchains = &s.swapchain;
    present.pImageIndices = &index;
    VkResult result;
    {
        RTPERF_FRAME_SCOPE("player.swapchain.queue_present");
        result = vkQueuePresentKHR(s.queue, &present);
    }
    if (result == VK_SUBOPTIMAL_KHR && s.transformMismatch) {
        return true;  // compositor rotation, not a stale swapchain
    }
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
        result == VK_ERROR_SURFACE_LOST_KHR) {
        s.rebuild = true;
        s.surfaceLost = result == VK_ERROR_SURFACE_LOST_KHR;
        return result == VK_SUBOPTIMAL_KHR;
    }
    s.failed = !check(result, "Player queue present");
    return !s.failed;
}

void PlayerSwapchainPresenter::onBackground() {
    auto& s = state();
    s.suspended = true;
    if (!s.device || s.swapchain == VK_NULL_HANDLE) {
        return;
    }
    // The window may already be gone; the swapchain objects are still ours to
    // free. The UI renderer and its render pass survive (see clearSwapchain).
    if (!g_vulkan_device_lost) {
        check(vkQueueWaitIdle(s.queue), "Player background idle");
    }
    s.clearSwapchain();
    s.rebuild = true;
}

void PlayerSwapchainPresenter::onForeground() {
    auto& s = state();
    if (!s.suspended) {
        return;
    }
    s.suspended = false;
    // A new ANativeWindow: the old VkSurface points at a destroyed window.
    s.surfaceLost = true;
    s.rebuild = true;
}

} // namespace AppRuntime

#endif
