#pragma once

// Shared context layout. Runtime/API code can use scene state without pulling
// in the editor panels. Keep member order stable for existing aggregate callers.
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct SceneData;
class Renderer;
class OptixWrapper;
namespace Backend {
class IBackend;
}
class ColorProcessor;
struct RenderSettings;
class SceneSelection;
class SceneUI;
struct SDL_Texture;

struct UIContext {
    SceneData& scene;
    Renderer& renderer;
    OptixWrapper* optix_gpu_ptr;
    Backend::IBackend* backend_ptr;
    ColorProcessor& color_processor;
    RenderSettings& render_settings;
    SceneSelection& selection;  // Scene selection manager
    SceneUI* scene_ui_ptr = nullptr; // Optional pointer back to UI host

    int& sample_count;                // dynamic counter (not in settings)
    bool& start_render;              // trigger flag
    std::string& active_model_path;

    bool& apply_tonemap;
    bool& reset_tonemap;

    bool& mouse_control_enabled;

    // Animation Preview Buffer (Thread-Safe)
    // Moved to end to preserve initializer list compatibility
    std::vector<uint32_t> animation_preview_buffer;
    std::atomic<bool> animation_preview_ready{false};
    std::mutex animation_preview_mutex;
    int animation_preview_width = 0;
    int animation_preview_height = 0;
    SDL_Texture* animation_preview_texture = nullptr;
    bool show_animation_preview = false;
    // Particle viewport display mode: 0=Solid (Vulkan billboards), 1=Debug (ImGui
    // dots clipped to the viewport background layer), 2=Render (path-traced;
    // previews as Solid for now).
    int particle_display_mode = 0;
    bool is_animation_mode = false; // NEW: For Unified Render Window
    SDL_Texture* ray_texture;

};
