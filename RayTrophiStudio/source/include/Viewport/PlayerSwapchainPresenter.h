#pragma once

#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))

#include <vulkan/vulkan.h>

#include <vector>

struct SDL_Window;
struct ImTextureData;

namespace VulkanRT {
class VulkanDevice;
struct ImageHandle;
}

namespace AppRuntime {

// The Player uses the engine's graphics/compute queue for both raster and WSI.
class PlayerSwapchainPresenter {
public:
    static PlayerSwapchainPresenter& instance();

    bool appendInstanceExtensions(SDL_Window* window, std::vector<const char*>& extensions);
    bool owns(VkInstance instance) const;
    bool createSurface(VkInstance instance);
    bool supportsDevice(VkPhysicalDevice physical) const;
    bool supportsPresent(VkPhysicalDevice physical, uint32_t family) const;
    bool attach(VulkanRT::VulkanDevice& engine);
    void release(VkInstance instance);
    void publish(const VulkanRT::ImageHandle& image);
    void discard(VkImage image);
    bool prepareUiFrame();
    void newFrame();
    bool present();
    // Android: the activity's native window is destroyed while paused and a
    // NEW one exists after resume, so the VkSurface must be recreated.
    void onBackground();
    void onForeground();
    bool active() const;
    bool failed() const;
    bool needsRedraw() const;
    uint64_t frameSerial() const;
    ImTextureData* createUiTexture(int width, int height);
    void updateUiTexture(ImTextureData& texture, const void* rgba,
                         int width, int height, int pitch);
    void destroyUiTexture(ImTextureData* texture);

private:
    struct State;
    static State& state();
};

} // namespace AppRuntime

#endif
