#include <SDL_main.h> 
#include <fstream>
#include <locale>
#include <chrono>
#include <vector>
#include <atomic>
#include <thread>
#include <future>
#include <SDL_image.h>
#include "Renderer.h"
#include "CPUInfo.h"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"  // Değiştirildi: sdlrenderer2
#include <scene_ui.h>

#include "EmbreeBVH.h"
#include "ParallelBVHNode.h"
#include "scene_ui_guides.hpp"  // Viewport guides (safe areas, letterbox, grids)
#include "default_scene_creator.hpp"
#include "ColorProcessingParams.h"
#include "scene_data.h"       // Added explicit include
#include "OptixWrapper.h"     // Added explicit include
#include "SceneSelection.h"   // Added explicit include
#include "Triangle.h"         // Added for CPU sync
#include "WaterSystem.h"      // Added for WaterManager
#include "CloudManager.h"     // Added for CloudManager

#include <filesystem>
#include <windows.h>
#include <commdlg.h>
#include "SplashScreen.h"  // Splash screen support
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1  // Use stb's zlib if available
// Fix Windows.h min/max macro conflicts
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#define NOMINMAX
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#include <InstanceManager.h>

bool SaveSurface(SDL_Surface* surface, const char* file_path) {
    SDL_Surface* surface_to_save = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);
    if (!surface_to_save) {
        SCENE_LOG_ERROR("Couldn't convert surface: " + std::string(SDL_GetError()));
        return false;
    }

    int result = IMG_SavePNG(surface_to_save, file_path);
    SDL_FreeSurface(surface_to_save);

    if (result != 0) {
        SCENE_LOG_ERROR("Failed to save image: " + std::string(IMG_GetError()));
        return false;
    }

    SCENE_LOG_INFO("Image saved to: " + std::string(file_path));
    return true;
}

std::string saveFileDialogW(const wchar_t* filter = L"PNG Files\0*.png\0") {
    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW ofn = { 0 };
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_OVERWRITEPROMPT;
    ofn.lpstrTitle = L"Save Image";
    ofn.hwndOwner = GetActiveWindow();

    if (GetSaveFileNameW(&ofn)) {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, filename, -1, nullptr, 0, nullptr, nullptr);
        std::string utf8_path(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, utf8_path.data(), size_needed, nullptr, nullptr);

        if (!utf8_path.empty() && utf8_path.back() == '\0')
            utf8_path.pop_back();

        std::string lower = utf8_path;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find(".png") == std::string::npos)
            utf8_path += ".png";

        SCENE_LOG_INFO("Save dialog returned: " + utf8_path);
        return utf8_path;
    }

    SCENE_LOG_ERROR("Save dialog canceled or failed.");
    return "";
}



constexpr float to_radians(float degrees) { return degrees * 3.1415926535f / 180.0f; }
int sample_per_pass = 1;
float animation_duration = 1.0f;
float animation_fps = 24;
bool start_render = false;
bool quit = false;
bool apply_tonemap = false;
bool reset_tonemap = false;
bool mouse_control_enabled = true;
float mouse_sensitivity = 0.4f;
bool camera_moved = false;
bool use_denoiser = false;
bool camera_moved_recently = false;
auto last_camera_move_time = std::chrono::steady_clock::now();
float denoiser_blend_factor = 0.9f;
bool use_optix = true;
bool UI_use_embree = true;
int sample_count = 1;
float yaw = -90.0f;
float pitch = 0.0f;
bool dragging = false;
int last_mouse_x = 0;
int last_mouse_y = 0;
float move_speed = 0.5f;
HitRecord hit_record;
Ray ray;
int mx = 0;
int my = 0;
float u;
float v;
bool rayhit = false;

// ===========================================================================
// DIRTY FLAGS - Prevents unnecessary GPU buffer updates when data unchanged
// ===========================================================================
bool g_camera_dirty = true;
bool g_lights_dirty = true;
bool g_world_dirty = true;

// ===========================================================================
// DEFERRED REBUILD FLAGS - For optimized batched rebuilds
// Set these instead of calling rebuild immediately, Main loop handles them
// ===========================================================================
bool g_bvh_rebuild_pending = false;      // CPU BVH needs rebuild
bool g_gpu_refit_pending = false;        // GPU Geometry needs update (Deferred)
bool g_optix_rebuild_pending = false;
bool g_optix_rebuild_in_progress = false; // True while TLAS rebuild is happening    // GPU OptiX geometry needs rebuild
bool g_mesh_cache_dirty = false;         // UI mesh cache needs rebuild
bool g_cpu_sync_pending = false;         // CPU data needs sync after TLAS mode changes
bool g_cpu_bvh_refit_pending = false;    // CPU BVH fast refit (Embree only)

// ===========================================================================
// SCENE LOADING FLAGS - Thread safety for project load/save operations
// ===========================================================================
std::atomic<bool> g_scene_loading_in_progress{false};  // Prevents concurrent load operations
bool g_needs_geometry_rebuild = false;   // Set by loader thread, main loop does actual rebuild
bool g_needs_optix_sync = false;         // Set by loader thread, main loop syncs OptiX buffers


Vec3 applyVignette(const Vec3& color, int x, int y, int width, int height, float strength = 1.0f) {
    float u = (x / (float)width - 0.5f) * 2.0f;
    float v = (y / (float)height - 0.5f) * 2.0f;
    float dist = u * u + v * v;
    float falloff = std::clamp(1.0f - strength * dist, 0.0f, 1.0f);
    return color * falloff;
}
// GLOBAL
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;
SDL_Surface* surface = nullptr;
SDL_Surface* original_surface = nullptr;
SDL_Texture* raytrace_texture = nullptr;
int saved_viewport_width = -1;
int saved_viewport_height = -1;
bool saved_window_maximized = false;
std::mutex surface_mutex;  // Surface erişimi için mutex
SceneUI ui;
SceneData scene;
Renderer ray_renderer(image_width, image_height, 1, 1);
OptixWrapper optix_gpu;
ColorProcessor color_processor(image_width, image_height);
std::string active_model_path;
SceneSelection scene_selection;  // Scene selection manager
UIContext ui_ctx{
   scene,
   ray_renderer,
   &optix_gpu,
   color_processor,
   render_settings,
   scene_selection,  // Add selection reference
   sample_count,
   start_render,
   active_model_path,
   apply_tonemap,
   reset_tonemap,
   mouse_control_enabled,
   mouse_sensitivity
};
void applyToneMappingToSurface(SDL_Surface* surface, SDL_Surface* original, ColorProcessor& processor) {
    Uint32* pixels = (Uint32*)surface->pixels;
    Uint32* src = (Uint32*)original->pixels;
    int width = surface->w;
    int height = surface->h;
    SDL_PixelFormat* fmt = surface->format;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            Uint8 r, g, b;
            SDL_GetRGB(src[j * width + i], fmt, &r, &g, &b);

            Vec3 raw_color(r / 255.0f, g / 255.0f, b / 255.0f);
            Vec3 final_color = processor.processColor(raw_color, i, j);
            if (processor.params.enable_vignette)
                final_color = applyVignette(final_color, i, j, width, height, processor.params.vignette_strength);
            r = static_cast<Uint8>(255.0f * std::clamp(final_color.x, 0.0f, 1.0f));
            g = static_cast<Uint8>(255.0f * std::clamp(final_color.y, 0.0f, 1.0f));
            b = static_cast<Uint8>(255.0f * std::clamp(final_color.z, 0.0f, 1.0f));

            pixels[j * width + i] = SDL_MapRGB(fmt, r, g, b);
        }
    }
}
void reset_render_resolution(int w, int h)
{
    // ------------------------------------------------------------------
    // 1. SDL Pencere Boyutunu Güncelle
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // 1. SDL Pencere Boyutunu Güncelle - IPTAL (Kullanıcı pencere boyutu değişsin istemiyor)
    // ------------------------------------------------------------------
    // SDL_SetWindowSize(window, w, h);
    
    // ------------------------------------------------------------------
    // 2. SDL kaynaklarını sıfırla (DESTROY)
    // ------------------------------------------------------------------
    if (raytrace_texture) SDL_DestroyTexture(raytrace_texture);
    if (surface) SDL_FreeSurface(surface);
    if (original_surface) SDL_FreeSurface(original_surface);

    // ------------------------------------------------------------------
    // 3. Yeni kaynakları oluştur (CREATE)
    // ------------------------------------------------------------------
    surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    original_surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    raytrace_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING, w, h);

    if (!surface || !original_surface || !raytrace_texture) {
        SCENE_LOG_ERROR("Failed to create SDL surfaces or texture!");
        return;
    }

    // Siyah ekranla başla
    SDL_FillRect(surface, nullptr, SDL_MapRGBA(surface->format, 0, 0, 0, 255));
    SDL_FillRect(original_surface, nullptr, SDL_MapRGBA(original_surface->format, 0, 0, 0, 255));
    
    // Texture'ı güncelle
    SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
    
    // Ekranı güncelle
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);

    // ------------------------------------------------------------------
    // 4. Aspect ratio ve global değişkenleri güncelle
    // ------------------------------------------------------------------
    aspect_ratio = (float)w / h;
    
    // ------------------------------------------------------------------
    // 5. Diğer Bileşenleri Güncelle
    // ------------------------------------------------------------------
    ray_renderer.resetResolution(w, h);
    // Always resize OptiX buffers if available
    if (g_hasOptix) { optix_gpu.resetBuffers(w, h); }
    color_processor.resize(w, h);

    // CRITICAL: Ensure Camera aspect ratio matches new resolution for correct ray generation (AF, Picking)
    if (scene.camera) {
        scene.camera->aspect_ratio = aspect_ratio;
        scene.camera->update_camera_vectors();
        if (g_hasOptix) { optix_gpu.setCameraParams(*scene.camera); }
    }

    SCENE_LOG_INFO("Render resolution updated: " + std::to_string(w) + "x" + std::to_string(h));
}
// Check if GPU has RT Cores (hardware ray tracing)
static bool hasRTCores(int major, int minor)
{
    // RTX donanımı SM 7.5 ile başladı.
    if (major > 7) return true;            // SM 8.x, 9.x, 10.x › yeni RTX mimarileri
    if (major == 7 && minor >= 5) return true; // SM 7.5 › Turing
    return false;
}

// Check if GPU can run OptiX (SM 5.0+ required)
// Non-RTX GPUs will use compute-based BVH traversal (slower but works)
static bool isOptixCapable(int major, int minor)
{
    // OptiX 7.x requires SM 5.0 minimum (Maxwell and newer)
    // SM 5.0+ = GTX 9xx, GTX 10xx, GTX 16xx, RTX 20xx, RTX 30xx, RTX 40xx
    return major >= 5;
}

// Global GPU info for HUD display
std::string g_gpu_name = "";
bool g_has_rt_cores = false;

void detectOptixHardware()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    // NVIDIA kart yok / CUDA yok / sürücü yok › OptiX yok.
    if (err != cudaSuccess || deviceCount == 0) {
        g_hasOptix = false;
        g_gpu_name = "CPU Only";
        SCENE_LOG_WARN("No CUDA-capable devices found. OptiX will be disabled.");
        return;
    }

    bool foundOptix = false;
    bool hasRT = false;

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int major = prop.major;
        int minor = prop.minor;

        if (isOptixCapable(major, minor)) {
            foundOptix = true;
            hasRT = hasRTCores(major, minor);
            g_gpu_name = prop.name;
            g_has_rt_cores = hasRT;
            
            if (hasRT) {
                SCENE_LOG_INFO("Found RTX device: " + std::string(prop.name) +
                    " (SM " + std::to_string(major) + "." + std::to_string(minor) + ") - Hardware RT Cores");
            } else {
                SCENE_LOG_INFO("Found OptiX-compatible device: " + std::string(prop.name) +
                    " (SM " + std::to_string(major) + "." + std::to_string(minor) + ") - Compute Mode (no RT cores)");
            }
            break;
        }
    }

    g_hasOptix = foundOptix;
}
std::string WStringToString(const std::wstring& wstr) {
    if (wstr.empty()) return {};
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), nullptr, 0, nullptr, nullptr);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), strTo.data(), size_needed, nullptr, nullptr);
    return strTo;
}
// Tek merkezden kontrol edilen OptiX init
bool initializeOptixIfAvailable(OptixWrapper& optix_gpu) {
    if (!g_hasOptix) return false; // Donanım yoksa direkt false döner

    try {
        optix_gpu.initialize();

        // PTX dosyası her zaman yüklenmeye çalışılır
        std::filesystem::path ptx_path = L"raygen.ptx";
        std::ifstream file(ptx_path, std::ios::binary);
        if (!file.is_open()) {
            SCENE_LOG_ERROR("Failed to open PTX file: " + WStringToString(ptx_path));
            g_hasOptix = false;
            return false;
        }

        std::string ptx((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());
        optix_gpu.setupPipeline(ptx.c_str());
    }
    catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("OptiX initialization failed: ") + e.what());
        g_hasOptix = false;
        return false;
    }

    return true;
}



void init_RayTrophi_Pro_Dark_Thema()
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* c = style.Colors;

    // Modern yumuşaklık
    style.FrameRounding = 6.0f;
    style.WindowRounding = 5.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.TabRounding = 5.0f;
    style.PopupRounding = 4.0f;

    // Daha düz ve net border
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;

    // Accent Color (marka rengi)
    ImVec4 accent = ImVec4(0.05f, 0.75f, 0.65f, 1.0f);

    // Arkaplanlar
    c[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.11f, ui.panel_alpha);
    c[ImGuiCol_ChildBg] = ImVec4(0.10f, 0.10f, 0.11f, 0.95f);
    c[ImGuiCol_PopupBg] = ImVec4(0.11f, 0.11f, 0.13f, 1.0f);

    // Yazılar
    c[ImGuiCol_Text] = ImVec4(0.95f, 0.97f, 0.99f, 1.0f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.45f, 0.45f, 0.48f, 1.0f);

    // Çerçeveler
    c[ImGuiCol_FrameBg] = ImVec4(0.17f, 0.17f, 0.19f, 1);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.23f, 0.23f, 0.26f, 1);
    c[ImGuiCol_FrameBgActive] = accent;

    // Butonlar
    c[ImGuiCol_Button] = ImVec4(0.21f, 0.21f, 0.23f, 1);
    c[ImGuiCol_ButtonHovered] = accent;
    c[ImGuiCol_ButtonActive] = ImVec4(0.03f, 0.63f, 0.55f, 1.0f);

    // Header / seçili elemanlar
    c[ImGuiCol_Header] = ImVec4(0.19f, 0.19f, 0.21f, 1);
    c[ImGuiCol_HeaderHovered] = accent;
    c[ImGuiCol_HeaderActive] = ImVec4(0.03f, 0.63f, 0.55f, 1);

    // Slider
    c[ImGuiCol_SliderGrab] = accent;
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.03f, 0.63f, 0.55f, 1);

    // Border
    c[ImGuiCol_Border] = ImVec4(0.0f, 0.0f, 0.0f, 0.60f);
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);

    // Separator'lar (daha belirgin, modern çizgi)
    style.SeparatorTextPadding = ImVec2(10.0f, 8.0f);
    style.SeparatorTextAlign = ImVec2(0.0f, 0.5f); // sola hizalı başlıklar çok daha modern
    style.SeparatorTextBorderSize = 2.0f;  // default 1.0f
	// Accent renkli separator
    c[ImGuiCol_Separator] = ImVec4(0.25f, 0.75f, 0.70f, 0.60f); // accent soft
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.25f, 0.80f, 0.75f, 1.00f);
    c[ImGuiCol_SeparatorActive] = ImVec4(0.20f, 0.90f, 0.85f, 1.00f);
   
    style.ItemSpacing = ImVec2(8, 6);   // aralıklar açılır
    style.ItemInnerSpacing = ImVec2(6, 6);
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Turkish");
#ifdef _WIN32

    // Konsolu tamamen kapat
    FreeConsole();

#endif
   // SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
 // Önce SDL açılacak
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        SCENE_LOG_ERROR(std::string("SDL_Init Error: ") + SDL_GetError());
        return 1;
    }
    
    // ===========================================================================
    // SPLASH SCREEN - Frameless startup screen with loading status
    // ===========================================================================
    SplashScreen splash;
    bool splashOk = splash.init("RayTrophi_image.png", 900, 700);
    if (splashOk) {
        splash.setStatus("Initializing...");
        splash.render();
    }
    
    // Detect GPU Hardware
    if (splashOk) { splash.setStatus("Detecting CUDA/OptiX hardware..."); splash.render(); }
    g_sceneLog.clear();
    detectOptixHardware();

     // Create main window
    if (splashOk) { splash.setStatus("Creating main window..."); splash.render(); }
     window = SDL_CreateWindow("RayTrophi Studio",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        image_width,
        image_height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN);

     renderer =
        SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

     surface = SDL_GetWindowSurface(window);

     raytrace_texture =
        SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
            SDL_TEXTUREACCESS_STREAMING, image_width, image_height);

     original_surface =
        SDL_ConvertSurface(surface, surface->format, 0);

    SDL_SetTextureBlendMode(raytrace_texture, SDL_BLENDMODE_NONE);
    
    // Initialize ImGui
    if (splashOk) { splash.setStatus("Initializing ImGui..."); splash.render(); }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);

    // Initialize Theme using ThemeManager (prevents style overriding issues)
    ThemeManager::instance().setTheme("RayTrophi Pro Dark");
    ThemeManager::instance().applyCurrentTheme(ui.panel_alpha); 
	
    // Initialize OptiX
    if (splashOk) { splash.setStatus("Initializing OptiX pipeline..."); splash.render(); }
    if (initializeOptixIfAvailable(optix_gpu)) {
        SCENE_LOG_INFO("OptiX is ready!");
        
        // Link OptixAccelManager logs to HUD
        optix_gpu.setAccelManagerStatusCallback([&](const std::string& msg, int type) {
            ImVec4 color = ImVec4(1, 1, 1, 1); // Info = White
            if (type == 1) color = ImVec4(1, 1, 0, 1); // Warn = Yellow
            if (type == 2) color = ImVec4(1, 0.2f, 0.2f, 1); // Error = Red
            ui.addViewportMessage(msg, 4.0f, color); // Slightly longer duration for backend msgs
        });
        
        render_settings.use_optix = true;
        ui_ctx.render_settings.use_optix = true;
    }
    else {
        SCENE_LOG_WARN("Falling back to CPU rendering.");
    }

    // Create Default Scene
    if (splashOk) { splash.setStatus("Creating default scene..."); splash.render(); }
    createDefaultScene(scene, ray_renderer, g_hasOptix ? &optix_gpu : nullptr);
    ui.invalidateCache(); // Ensure procedural objects are listed/selectable
    
    // Build initial BVH and OptiX structures
    if (splashOk) { splash.setStatus("Building BVH structures..."); splash.render(); }
    ray_renderer.rebuildBVH(scene, UI_use_embree);
    if (g_hasOptix) {
        ray_renderer.rebuildOptiXGeometry(scene, &optix_gpu);
        // CRITICAL: Immediately sync GPU buffers so sun direction is correct at startup
        optix_gpu.setWorld(ray_renderer.world.getGPUData());
        optix_gpu.setLightParams(scene.lights);
        if (scene.camera) optix_gpu.setCameraParams(*scene.camera);
    }
    // Update initial camera vectors
    if (scene.camera) scene.camera->update_camera_vectors();

    // ===========================================================================
    // SPLASH COMPLETE - Auto-close with fade out (no more waiting for click)
    // ===========================================================================
    if (splashOk) {
        splash.setStatus("Ready!");
        splash.render();
        SDL_Delay(300);  // Brief pause to show "Ready!" status
        splash.fadeOut(400);  // Quick fade out animation
        splash.close();
    }
    
    // Show main window now that loading is complete
    SDL_ShowWindow(window);
    SDL_MaximizeWindow(window);
    
    // ===========================================================================
    // GPU MODE HUD NOTIFICATION - Inform user about rendering backend
    // ===========================================================================
    if (g_hasOptix) {
        if (g_has_rt_cores) {
            ui.addViewportMessage("GPU: " + g_gpu_name + " (RTX - Hardware Ray Tracing)", 5.0f, ImVec4(0.3f, 1.0f, 0.5f, 1.0f));
        } else {
            ui.addViewportMessage("GPU: " + g_gpu_name + " (Compute Mode - No RT Cores)", 6.0f, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
            ui.addViewportMessage("Performance may be slower than RTX cards", 6.0f, ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
        }
    } else {
        ui.addViewportMessage("CPU Rendering Mode (No compatible GPU found)", 6.0f, ImVec4(1.0f, 0.5f, 0.3f, 1.0f));
    }

    SDL_Event e;
    while (!quit) {

        // --- AUTO RESIZE FOR FINAL RENDER ---
        bool is_final_mode = ui_ctx.render_settings.is_final_render_mode;
        
        if (is_final_mode) {
             int target_w = ui_ctx.render_settings.final_render_width;
             int target_h = ui_ctx.render_settings.final_render_height;
             
             // Check if resize needed for final render
             if (image_width != target_w || image_height != target_h) {
                 if (saved_viewport_width == -1 && !pending_resolution_change) {
                     // Save current (viewport) state
                     saved_viewport_width = image_width;
                     saved_viewport_height = image_height;
                     saved_window_maximized = (SDL_GetWindowFlags(window) & SDL_WINDOW_MAXIMIZED);
                     
                     // Trigger resize
                     pending_width = target_w;
                     pending_height = target_h;
                     pending_resolution_change = true;
                     SCENE_LOG_INFO("Switching to Final Render Resolution.");
                 }
             }
        } 
        else {
             // If not final mode, but we have a saved state, it means we must restore
             if (saved_viewport_width != -1) {
                  if ((image_width != saved_viewport_width || image_height != saved_viewport_height) && !pending_resolution_change) {
                      pending_width = saved_viewport_width;
                      pending_height = saved_viewport_height;
                      pending_resolution_change = true;
                      SCENE_LOG_INFO("Restoring Viewport Resolution.");
                  } else if (image_width == saved_viewport_width && image_height == saved_viewport_height) {
                      saved_viewport_width = -1; // Restoration done
                  }
             }
        }
        // ------------------------------------

        camera_moved = false;

        // Update Texture if rendering is happening in background (for visualization)
        if (rendering_in_progress && scene.initialized) {
            static auto last_tex_update = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            // 10 FPS visual update for animation
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_tex_update).count() > 100) { 
                SDL_LockSurface(surface);
                SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
                SDL_UnlockSurface(surface);
                last_tex_update = now;
            }
            
            // --- ANIMATION PREVIEW UPDATE ---
            if (ui_ctx.animation_preview_ready) {
                std::lock_guard<std::mutex> lock(ui_ctx.animation_preview_mutex);
                int w = ui_ctx.animation_preview_width;
                int h = ui_ctx.animation_preview_height;
                
                if (w > 0 && h > 0) {
                    // Check if texture needs (re)creation
                    int tw = 0, th = 0;
                    if (ui_ctx.animation_preview_texture) {
                        SDL_QueryTexture(ui_ctx.animation_preview_texture, nullptr, nullptr, &tw, &th);
                    }
                    
                    if (!ui_ctx.animation_preview_texture || tw != w || th != h) {
                        if (ui_ctx.animation_preview_texture) SDL_DestroyTexture(ui_ctx.animation_preview_texture);
                        ui_ctx.animation_preview_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, w, h);
                    }
                    
                    if (ui_ctx.animation_preview_texture) {
                        SDL_UpdateTexture(ui_ctx.animation_preview_texture, nullptr, ui_ctx.animation_preview_buffer.data(), w * sizeof(uint32_t));
                    }
                }
                ui_ctx.animation_preview_ready = false;
            }
        }
        // =========================================================================
        // TIME & ANIMATION UPDATE
        // =========================================================================
        static auto last_sim_time = std::chrono::steady_clock::now();
        auto current_sim_time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(current_sim_time - last_sim_time).count();
        last_sim_time = current_sim_time;
        
        bool timeline_playing = ui.timeline.isPlaying();
        
        if (timeline_playing) {
             // Update Water Manager
             WaterManager::getInstance().update(dt);
             
             // Update OptiX Time
                  // Use frame count from timeline (24 FPS assumption or settings)
                  float time_seconds = ui.timeline.getCurrentFrame() / 24.0f;

                  // Calculate wind transforms on CPU
                  InstanceManager::getInstance().updateWind(time_seconds, scene);

                  // Update OptiX Time
                  optix_gpu.setTime(time_seconds, time_seconds);
                  
                  // Efficiently update instance transforms on GPU (no full rebuild)
                  optix_gpu.updateTLASMatricesOnly(scene.world.objects);
                  
                  // VDB update moved outside of timeline_playing block (see below)

                  // Force redraw (break accumulation)
                  start_render = true;
                  g_needs_optix_sync = true; // Signal that params changed
             
        } else {
             // Static update (for editor changes)
             WaterManager::getInstance().update(0.0f);
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // VDB SEQUENCE UPDATE (Always check, even when not playing)
        // ═══════════════════════════════════════════════════════════════════
        static int last_vdb_frame = -1;
        int current_vdb_frame = ui.timeline.getCurrentFrame();
        
        if (current_vdb_frame != last_vdb_frame) {
            bool vdb_needs_sync = false;
            for (auto& vdb : scene.vdb_volumes) {
                if (vdb->isAnimated() && vdb->isLinkedToTimeline()) {
                    vdb->updateFromTimeline(current_vdb_frame);
                    vdb_needs_sync = true;
                }
            }
            
            if (vdb_needs_sync) {
                ui.syncVDBVolumesToGPU(ui_ctx);
                if (use_optix) {
                    optix_gpu.resetAccumulation();
                }
                ray_renderer.resetCPUAccumulation();
            }
            last_vdb_frame = current_vdb_frame;
        }

        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        ui_ctx.ray_texture = raytrace_texture;
        
        // ===================================================================
        // SCENE LOADING POPUP
        // ===================================================================
        if (ui.scene_loading.load()) {
            if (!ImGui::IsPopupOpen("Loading Scene...")) {
                 ImGui::OpenPopup("Loading Scene...");
            }
            
            ImVec2 center = ImGui::GetMainViewport()->GetCenter();
            ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(420, 160));
            
            if (ImGui::BeginPopupModal("Loading Scene...", nullptr, 
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
                
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Loading / Saving...");
                ImGui::Spacing();
                
                // Real progress from loading thread
                int progress = ui.scene_loading_progress.load();
                float progress_f = progress / 100.0f;
                
                // Progress bar with percentage
                char overlay[32];
                snprintf(overlay, sizeof(overlay), "%d%%", progress);
                ImGui::ProgressBar(progress_f, ImVec2(-1, 22), overlay);
                
                ImGui::Spacing();
                
                // Current stage text
                ImGui::TextColored(ImVec4(0.7f, 0.9f, 0.7f, 1.0f), "%s", ui.scene_loading_stage.c_str());
                
                ImGui::EndPopup();
            }
        }
        // Close scene loading popup when done
        else if (ui.scene_loading_done.load()) {
            ui.scene_loading_done = false;
            // Scene loaded, popup will auto-close
            
            // Refresh UI cache and start render
            ui.invalidateCache();
            
            // =========================================================================
            // REBUILD ON MAIN THREAD (Thread-safe OptiX handling)
            // ProjectManager sets g_needs_geometry_rebuild, we do actual GPU work here
            // =========================================================================
            if (g_needs_geometry_rebuild) {
                ui_ctx.renderer.rebuildBVH(ui_ctx.scene, ui_ctx.render_settings.UI_use_embree);
                g_needs_geometry_rebuild = false;
            }
            
            ui_ctx.renderer.resetCPUAccumulation();
            
            if (ui_ctx.optix_gpu_ptr && g_needs_optix_sync) {
                ui_ctx.renderer.rebuildOptiXGeometry(ui_ctx.scene, ui_ctx.optix_gpu_ptr);
                
                // CRITICAL: Update materials AFTER geometry rebuild
                ui_ctx.renderer.updateOptiXMaterialsOnly(ui_ctx.scene, ui_ctx.optix_gpu_ptr);
                
                ui_ctx.optix_gpu_ptr->setLightParams(ui_ctx.scene.lights);
                if (ui_ctx.scene.camera) ui_ctx.optix_gpu_ptr->setCameraParams(*ui_ctx.scene.camera);
                ui_ctx.optix_gpu_ptr->resetAccumulation();
                g_needs_optix_sync = false;
                SCENE_LOG_INFO("GPU rebuild complete on main thread.");
            }
            
            // Mark all buffers dirty for fresh scene
            g_camera_dirty = true;
            g_lights_dirty = true;
            g_world_dirty = true;
            
            ui_ctx.start_render = true;
        }
        
        ui.draw(ui_ctx);
        
        // ===========================================================================
        // CENTRALIZED CAMERA UPDATE (PHASE 1)
        // Checks if UI changed camera or if continuous effects (Shake/AF-C) are active
        // ===========================================================================
        if (scene.camera) {
            bool is_dirty = scene.camera->checkDirty();
            
            // Continuous effects require per-frame updates
            bool is_shaking = scene.camera->enable_camera_shake;
            bool is_af_c = (ui.viewport_settings.focus_mode == 2);
            
            if (is_dirty || is_shaking || is_af_c) {
                 // Ensure vectors are up to date
                 scene.camera->update_camera_vectors();
                 
                 if (g_hasOptix) {
                    optix_gpu.setCameraParams(*scene.camera);
                    optix_gpu.resetAccumulation();
                 }
                 ray_renderer.resetCPUAccumulation();
                 
                 start_render = true;
            }
        }
        // NOTE: handleMouseSelection is now called inside ui.draw() - removed duplicate call here
        
        // Scene Hierarchy panel now called inside ui.draw()
        
        // ===========================================================================
        // VIEWPORT GUIDES - Draw safe areas, letterbox, grids on viewport
        // ===========================================================================
        {
            // Get viewport bounds (full window for now)
            int win_w, win_h;
            SDL_GetWindowSize(window, &win_w, &win_h);
            
            ImVec2 viewport_min(0, 0);
            ImVec2 viewport_max((float)win_w, (float)win_h);
            
            // Convert GuideSettings to ViewportGuides format
            ViewportGuides::GuideSettings vg_settings;
            vg_settings.show_safe_areas = ui.guide_settings.show_safe_areas;
            vg_settings.safe_area_type = ui.guide_settings.safe_area_type;
            vg_settings.title_safe_percent = ui.guide_settings.title_safe_percent;
            vg_settings.action_safe_percent = ui.guide_settings.action_safe_percent;
            vg_settings.show_letterbox = ui.guide_settings.show_letterbox;
            vg_settings.aspect_ratio_index = ui.guide_settings.aspect_ratio_index;
            vg_settings.letterbox_opacity = ui.guide_settings.letterbox_opacity;
            vg_settings.show_grid = ui.guide_settings.show_grid;
            vg_settings.grid_type = ui.guide_settings.grid_type;
            vg_settings.show_center = ui.guide_settings.show_center;
            
            // Draw guides on background draw list (behind UI, on top of viewport texture)
            ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
            ViewportGuides::drawAllGuides(draw_list, viewport_min, viewport_max, vg_settings);
        }
        
        // Handle resolution change - SKIP RENDERING THIS FRAME
        if (pending_resolution_change) {
            pending_resolution_change = false;
            
            // CRITICAL: Prevent any rendering during resolution change
            rendering_stopped_cpu = true;
            rendering_stopped_gpu = true;
            
            // Reset accumulation buffers BEFORE resolution change
            ray_renderer.resetCPUAccumulation();
            if (g_hasOptix) {
                optix_gpu.resetAccumulation();
            }
            
            // Reset render state
            render_settings.is_rendering_active = false;
            render_settings.render_current_samples = 0;
            render_settings.render_progress = 0.0f;
            render_settings.render_elapsed_seconds = 0.0f;
            render_settings.render_estimated_remaining = 0.0f;
            render_settings.avg_sample_time_ms = 0.0f;
            
            // Change resolution
            image_width = pending_width;
            image_height = pending_height;
             reset_render_resolution(image_width, image_height);
             
             // Restore Maximize State if returning to saved viewport size
             if (saved_viewport_width != -1 && image_width == saved_viewport_width && image_height == saved_viewport_height) {
                 if (saved_window_maximized) {
                     SDL_MaximizeWindow(window);
                 }
             }

             // Reset stop flags and allow rendering
            rendering_stopped_cpu = false;
            rendering_stopped_gpu = false;
            render_settings.is_render_paused = false;
            start_render = true;  // Trigger fresh render at new resolution
            
            SCENE_LOG_INFO("Resolution changed to " + std::to_string(image_width) + "x" + std::to_string(image_height));
            
            // Skip to next loop iteration - render starts next frame
            ImGui::Render();
            ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
            SDL_RenderPresent(renderer);
            continue;  // Skip rest of loop
        }
        while (SDL_PollEvent(&e)) {
            // surface = SDL_GetWindowSurface(window);
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_s && (e.key.keysym.mod & KMOD_CTRL)) {
                     ui_ctx.render_settings.save_image_requested = true;
                }
                
                // DELETE OBJECT SHORTCUT
                if (e.key.keysym.sym == SDLK_DELETE) {
                    // Only invoke if not typing in an Input field (ImGui check)
                    if (!ImGui::GetIO().WantCaptureKeyboard) {
                        ui.triggerDelete(ui_ctx);
                    }
                }
            }

            if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
            {
               // image_width = pending_width;
               // image_height = pending_height;
               // reset_render_resolution(image_width, image_height);
            }

            ImGui_ImplSDL2_ProcessEvent(&e);
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_MIDDLE) {
                dragging = true;
                last_mouse_x = e.button.x;
                last_mouse_y = e.button.y;
            }
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                if (!ImGui::GetIO().WantCaptureMouse) {
                    int win_w, win_h;
                    SDL_GetWindowSize(window, &win_w, &win_h);

                    rayhit = true;
                    mx = e.button.x;
                    // my'yi window height'a göre ters çevir (global my değişkenini kullanarak)
                    my = win_h - e.button.y; 

                    u = (mx + 0.5f) / (float)win_w;
                    v = (my + 0.5f) / (float)win_h;
                }
            }
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_MIDDLE) {
                dragging = false;
                if (mouse_control_enabled)
                    start_render = true;
            }

            if (e.type == SDL_MOUSEMOTION && dragging && scene.camera && mouse_control_enabled) {
                // Mouse ImGui widgetlerinin üzerindeyse kamera dönmesin
                if (!ImGui::GetIO().WantCaptureMouse) { // Not on UI
                    int dx = e.motion.x - last_mouse_x;
                    int dy = e.motion.y - last_mouse_y;

                    const Uint8* state = SDL_GetKeyboardState(NULL);
                    bool is_shift_pressed = state[SDL_SCANCODE_LSHIFT] || state[SDL_SCANCODE_RSHIFT];

                    if (is_shift_pressed) {
                        // PANNING (Shift + Middle Mouse)
                        float pan_speed = mouse_sensitivity * 0.1f * scene.camera->focus_dist; 
                        
                        Vec3 right = scene.camera->u; 
                        Vec3 up = scene.camera->v;    
                        
                        Vec3 offset = right * -(float)dx * pan_speed + up * (float)dy * pan_speed;
                        
                        scene.camera->lookfrom += offset;
                        scene.camera->lookat += offset;
                        
                        scene.camera->update_camera_vectors();
                    }
                    else {
                        // Check for Control Key (Zoom)
                        bool is_ctrl_pressed = state[SDL_SCANCODE_LCTRL] || state[SDL_SCANCODE_RCTRL];
                        
                        if (is_ctrl_pressed) {
                            // ZOOM (Ctrl + Middle Mouse Drag Up/Down)
                            // Dragging mouse Up (negative dy) = Zoom In
                            // Dragging mouse Down (positive dy) = Zoom Out
                            
                            float zoom_speed = 0.05f * scene.camera->focus_dist;
                            float zoom_amount = -(float)dy * zoom_speed * 0.1f; 

                            Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                            scene.camera->lookfrom += forward * zoom_amount;
                            scene.camera->lookat += forward * zoom_amount; // pan-zoom vs dolly-zoom? 
                            // Blender "Zoom" usually acts like Dolly (moves camera), keeping LookAt distance relative?
                            // Actually pure dolly moves camera but maintains focus distance for DOF?
                            // Let's emulate scroll wheel logic: move both.
                            
                            scene.camera->update_camera_vectors();
                        }
                        else {
                            // ORBIT / ROTATION (Middle Mouse only)
                            // Rotation speed reduced by factor of 0.2 for better control
                            yaw += dx * mouse_sensitivity * 0.2f;
                            pitch -= dy * mouse_sensitivity * 0.2f;
                            pitch = std::clamp(pitch, -89.9f, 89.9f);

                            float rad_yaw = yaw * 3.14159265f / 180.0f;
                            float rad_pitch = pitch * 3.14159265f / 180.0f;

                            Vec3 direction;
                            direction.x = cosf(rad_yaw) * cosf(rad_pitch);
                            direction.y = sinf(rad_pitch);
                            direction.z = sinf(rad_yaw) * cosf(rad_pitch);

                            direction = direction.normalize();
                            
                            // Prevent unwanted roll by locking Up vector
                            scene.camera->vup = Vec3(0.0f, 1.0f, 0.0f);
                            
                            scene.camera->setLookDirection(direction);
                        }
                    }

                    last_mouse_x = e.motion.x;
                    last_mouse_y = e.motion.y;
                    last_camera_move_time = std::chrono::steady_clock::now();
                    start_render = true;
                    g_camera_dirty = true;  // Mark camera buffer for GPU update
                }
            }

            if (e.type == SDL_MOUSEWHEEL && mouse_control_enabled && scene.camera) {
                // Eğer mouse ImGui üzerinde değilse
                if (!ImGui::GetIO().WantCaptureMouse) {
                    float scroll_amount = e.wheel.y;  // yukarı: pozitif, aşağı: negatif
                    float move_speed = 1.5f * mouse_sensitivity; // Apply Mouse Sensitivity to Zoom
                    Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                    scene.camera->lookfrom += forward * scroll_amount * move_speed;
                    scene.camera->lookat = scene.camera->lookfrom + forward * scene.camera->focus_dist;

                    scene.camera->update_camera_vectors();
                    
                    // Update Autofocus (AF-C) if enabled
                    ui.updateAutofocus(ui_ctx);
                    
                    last_camera_move_time = std::chrono::steady_clock::now();
                    camera_moved = true;
                    start_render = true;
                    g_camera_dirty = true;  // Mark camera buffer for GPU update
                }
            }


            if (e.type == SDL_QUIT) ui.tryExit();
        }
        const Uint8* key_state = SDL_GetKeyboardState(NULL);

        if (mouse_control_enabled && scene.camera) {

            Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
            Vec3 right = Vec3::cross(forward, scene.camera->vup).normalize();
            Vec3 up = scene.camera->vup;

            // Arrow Keys for Camera Movement (avoids conflict with G/R/S gizmo shortcuts)
            // Apply sensitivity to base movement speed
            float effective_speed = move_speed * (mouse_sensitivity * 5.0f); // Scale up a bit as sensitivity is usually low (0.1)

            if (key_state[SDL_SCANCODE_UP]) {
                scene.camera->lookfrom += forward * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_DOWN]) {
                scene.camera->lookfrom -= forward * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_LEFT]) {
                scene.camera->lookfrom -= right * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_RIGHT]) {
                scene.camera->lookfrom += right * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_PAGEUP]) {
                scene.camera->lookfrom += up * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_PAGEDOWN]) {
                scene.camera->lookfrom -= up * effective_speed;
                camera_moved = true;
            }

            if (camera_moved) {
                Vec3 dir = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                scene.camera->lookat = scene.camera->lookfrom + dir * scene.camera->focus_dist;
                scene.camera->update_camera_vectors();
                last_camera_move_time = std::chrono::steady_clock::now();
                start_render = true;
                g_camera_dirty = true;  // Mark camera buffer for GPU update
            }
            // hareket durmuşsa foveation seviyesi büyüt
            if (camera_moved_recently &&
                std::chrono::steady_clock::now() - last_camera_move_time > std::chrono::milliseconds(50)) {
                camera_moved_recently = false;

            }

        }

        bool wind_active = false; // Declared here for wider scope
        auto now = std::chrono::steady_clock::now();
        auto delta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_camera_move_time).count();
        camera_moved_recently = (delta_ms < 50);
        /*if (rayhit && scene.initialized) {
            rayhit = false;
            start_render = true;
            ray = scene.camera->get_ray(u, v);
            if (scene.bvh->hit(ray, 0.00001f, 1e10f, hit_record)) {
                scene.camera->focus_dist = hit_record.t;

            }

            else {
                SCENE_LOG_INFO("No hit detected.");
            }
        }*/
        // New Render Logic
        if (scene.initialized) {

            // 1. Handle Animation Render Request
            // Prioritize this over interactive loop
            if (ui_ctx.render_settings.start_animation_render) {
                
                if (rendering_in_progress) {
                     // Already running something heavy, ignore new request
                     SCENE_LOG_WARN("Cannot start animation: Rendering already in progress.");
                     ui_ctx.render_settings.start_animation_render = false;
                     render_settings.start_animation_render = false;
                }
                else {
                    // Start Animation
                    ui_ctx.render_settings.start_animation_render = false;
                    render_settings.start_animation_render = false;
                    rendering_in_progress = true; // Set block flag immediately
                    
                    // Reset stop flags for new render
                    rendering_stopped_cpu = false;
                    rendering_stopped_gpu = false;
                    
                    start_render = false; // Cancel any pending interactive render
                    
                    SCENE_LOG_INFO("Starting animation render...");
                    std::string output_folder = ui_ctx.render_settings.animation_output_folder;
                    if (output_folder.empty()) output_folder = "render_animation";
                    SCENE_LOG_INFO("Output folder set to: " + output_folder);

                    // Capture local copies of settings to avoid thread race
                    int anim_sample_count = ui_ctx.render_settings.final_render_samples; // Use intended target, NOT viewport accum count
                    int anim_sample_per_pass = sample_per_pass;
                    int anim_fps = ui_ctx.render_settings.animation_fps;
                    float anim_duration = ui_ctx.render_settings.animation_duration;
                    bool anim_use_denoiser = ui_ctx.render_settings.use_denoiser;
                    float anim_denoiser_blend = ui_ctx.render_settings.denoiser_blend_factor;
                    bool anim_use_optix = ui_ctx.render_settings.use_optix;
                    int anim_start_frame = ui_ctx.render_settings.animation_start_frame;
                    int anim_end_frame = ui_ctx.render_settings.animation_end_frame;
                    
                    SCENE_LOG_INFO("DEBUG: Main starting animation thread. UI frames: " + std::to_string(anim_start_frame) + " - " + std::to_string(anim_end_frame));

                    // Detach thread
                    std::thread anim_thread([=]() {
                        ray_renderer.render_Animation(surface, window, raytrace_texture, renderer,
                            anim_sample_count, anim_sample_per_pass,
                            anim_fps, anim_duration, anim_start_frame, anim_end_frame,
                            scene,
                            output_folder,
                            anim_use_denoiser,
                            anim_denoiser_blend,
                            &optix_gpu,
                            anim_use_optix,
                            &ui_ctx);
                            
                         SCENE_LOG_INFO("Animation render completed.");
                    });
                    anim_thread.detach();
                }
            }

           
            // 2. Handle Interactive Render (One Frame / Progressive)
            // ONLY if no background rendering is happening
            if (start_render) {
                 if (rendering_in_progress) {
                     // Block interactive render if animation is running
                     start_render = false;
                 }
                 else {
                     // Safe to start new render
                     start_render = false;

                     // Reset stop flags for new render
                     rendering_stopped_cpu = false;
                     rendering_stopped_gpu = false;// --- Animation State Update ---
                    float fps = ui_ctx.render_settings.animation_fps;
                    if (fps <= 0.0f) fps = 24.0f;
                    int start_f = ui_ctx.render_settings.animation_start_frame;
                    int current_f = ui_ctx.render_settings.animation_playback_frame;
                    float time = (current_f - start_f) / fps;

                    if (ui_ctx.render_settings.use_optix && g_hasOptix) {
                        // ============ SYNCHRONOUS OPTIX RENDER (No Thread) ============
                        // Each pass is ~10-50ms for 1 sample, fast enough for UI
                        
                        // OPTIMIZATION: Only update animation state when timeline frame changed
                        // Skip during camera-only movement to avoid expensive geometry updates
                        // ALSO: Skip for manual keyframes - TimelineWidget::draw() handles those
                        static int last_anim_frame = -1;
                        bool geometry_updated = false;
                        bool has_file_animations = !scene.animationDataList.empty();
                        
                        if (current_f != last_anim_frame && has_file_animations) {
                            geometry_updated = ray_renderer.updateAnimationState(scene, time);
                            last_anim_frame = current_f;
                        }

                        // WIND ANIMATION (Independent of FBX animations)
                        // Checks if any InstanceGroup has wind enabled and updates matrices
                        static float last_wind_time = -1.0f;
                        

                        if (std::abs(time - last_wind_time) > 0.001f) {
                            ray_renderer.updateWind(scene, time);
                            
                            // Check if wind actually modified anything
                            for(const auto& group : InstanceManager::getInstance().getGroups()) {
                                if(group.gpu_dirty) { wind_active = true; break; }
                            }
                            
                            if (wind_active) {
                                // geometry_updated = true; // REMOVED: Managed by updateWind (Direct Instance Update + Refit)
                                // This prevents expensive updateTLASGeometry (BLAS Rebuild/Skinning) for simple wind sway
                            }
                            last_wind_time = time;
                        }
                        
                        if (geometry_updated) {
                            // SKINNING FIX: File-based animations may include skinning which deforms vertices.
                            // updateGeometry() in TLAS mode only updates matrices (transforms).
                            // For skinning, we need updateTLASGeometry() which rebuilds BLAS with new vertex data.
                            
                            // Pass Calculated Bone Matrices for GPU Skinning
                            optix_gpu.updateTLASGeometry(scene.world.objects, ray_renderer.finalBoneMatrices);
                            
                            // Geometry change implies all buffers need refresh
                            g_camera_dirty = true;
                            g_lights_dirty = true;
                            g_world_dirty = true;
                        }
                        
                        // OPTIMIZATION: Only update GPU buffers when data has changed
                        if (g_camera_dirty && scene.camera) {
                            optix_gpu.setCameraParams(*scene.camera);
                            Vec3 dir = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                            yaw = atan2f(dir.z, dir.x) * 180.0f / 3.14159265f;
                            pitch = asinf(dir.y) * 180.0f / 3.14159265f;
                            
                            // Set camera Y for volumetric cloud parallax
                            ray_renderer.world.setCameraY(scene.camera->lookfrom.y);
                            g_camera_dirty = false;
                        }
                        if (g_lights_dirty && !scene.lights.empty()) {
                            optix_gpu.setLightParams(scene.lights);
                            g_lights_dirty = false;
                        }
                        // Update Cloud FFT map binding dynamically
                        // 1. Update Cloud Manager (independent FFT) if needed
                        float dt_cloud = ImGui::GetIO().DeltaTime;
                        if (ui.timeline.isPlaying()) {
                           // Use animation time step if playing
                           // dt_cloud is already calculated above as dt?
                        }

                        CloudManager::getInstance().update(ImGui::GetIO().DeltaTime, ray_renderer.world.getNishitaParams());

                        NishitaSkyParams nishita = ray_renderer.world.getNishitaParams();
                        cudaTextureObject_t waterFFT = WaterManager::getInstance().getFirstFFTHeightMap();
                        cudaTextureObject_t cloudFFT = CloudManager::getInstance().getCloudFFTTexture();
                        
                        // Priority: Ocean > CloudFFT (Standalone)
                        // If we have an ocean, sync clouds to it. If not, use independent cloud FFT.
                        cudaTextureObject_t activeFFT = waterFFT ? waterFFT : cloudFFT;
                        
                        if (nishita.cloud_fft_map != activeFFT) {
                             nishita.cloud_fft_map = activeFFT;
                             // Only set 'use_fft' flag if User requested it in settings, 
                             // not just because we have a texture. But nishita.cloud_use_fft comes from UI.
                             // Actually, we should force it enabled if texture exists AND user checked it?
                             // No, user checks 'cloud_use_fft' in UI. We just provide map.
                             // But render_cloud_layer checks both.
                             
                             // Update texture handle
                             nishita.cloud_fft_map = activeFFT;
                             ray_renderer.world.setNishitaParams(nishita);
                             g_world_dirty = true;
                        }

                        if (g_world_dirty) {
                            optix_gpu.setWorld(ray_renderer.world.getGPUData());
                            g_world_dirty = false;
                        }

                        static std::vector<uchar4> framebuffer;
                        if (framebuffer.size() != image_width * image_height) {
                            framebuffer.resize(image_width * image_height);
                        }
                        
                        // Single pass render (1 sample) - fast, no UI blocking
                        auto sample_start = std::chrono::high_resolution_clock::now();
                       if (g_hasOptix && render_settings.use_optix) {
                     // Check if we are playing animation
                     int loop_count = 1;
                     if (timeline_playing) {
                         loop_count = std::max(1, render_settings.animation_samples_per_frame);
                     }
                     
                     for (int i = 0; i < loop_count; ++i) {
                         optix_gpu.launch_random_pixel_mode_progressive(
                             surface, window, renderer, 
                             image_width, image_height, 
                             framebuffer, raytrace_texture
                         );
                     }
                     // Debug Log (Temporary)
                     // if (timeline_playing) SCENE_LOG_INFO("Preview Loop: " + std::to_string(loop_count));
                } else {
                        optix_gpu.launch_random_pixel_mode_progressive(
                            surface, window, renderer, 
                            image_width, image_height, 
                            framebuffer, raytrace_texture
                        );
                }
                        
                        auto sample_end = std::chrono::high_resolution_clock::now();
                        float sample_time_ms = std::chrono::duration<float, std::milli>(sample_end - sample_start).count();
                        
                        // Update progress for UI
                        int prev_samples = render_settings.render_current_samples;
                        render_settings.render_current_samples = optix_gpu.getAccumulatedSamples();
                        
                        int effective_max_samples = render_settings.is_final_render_mode ? render_settings.final_render_samples : render_settings.max_samples;
                        render_settings.render_target_samples = effective_max_samples > 0 ? effective_max_samples : 100;
                        
                        render_settings.render_progress = (float)render_settings.render_current_samples / render_settings.render_target_samples;
                        render_settings.is_rendering_active = !optix_gpu.isAccumulationComplete();
                        
                        // Update time estimation
                        if (render_settings.render_current_samples > prev_samples) {
                            // Moving average for sample time
                            render_settings.avg_sample_time_ms = render_settings.avg_sample_time_ms * 0.8f + sample_time_ms * 0.2f;
                            render_settings.render_elapsed_seconds += sample_time_ms / 1000.0f;
                            
                            int remaining_samples = render_settings.render_target_samples - render_settings.render_current_samples;
                            render_settings.render_estimated_remaining = (remaining_samples * render_settings.avg_sample_time_ms) / 1000.0f;
                        }
                        
                        // ===== DENOISE LOGIC =====
                        bool effective_denoiser = render_settings.is_final_render_mode ? render_settings.render_use_denoiser : render_settings.use_denoiser;
                        if (effective_denoiser && optix_gpu.getAccumulatedSamples() > 0) {
                                ray_renderer.applyOIDNDenoising(surface, 0, true, ui_ctx.render_settings.denoiser_blend_factor);
                            }
                        }
                    else {
                        // ============ SYNCHRONOUS CPU RENDER (Like OptiX) ============
                        // Each pass is 1 sample per pixel, accumulates progressively
                        
                        // ===============================================================
                        // GPU -> CPU MODE SYNC: Update CPU vertices from GPU state
                        // ===============================================================
                        if (g_cpu_sync_pending) {
                            ui.addViewportMessage("Syncing CPU data...", 5.0f);
                            
                            // Update all CPU vertices from their Transform handles
                            for (auto& obj : scene.world.objects) {
                                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                if (tri) {
                                    tri->updateTransformedVertices();
                                }
                            }
                            
                            // Trigger CPU BVH rebuild (SYNCHRONOUS for safety)
                            // Async rebuild causes crashes if objects were deleted in GPU mode
                            extern bool use_embree;
                            ray_renderer.rebuildBVH(scene, use_embree);
                            
                            g_bvh_rebuild_pending = false; // Cancel any pending async rebuilds
                            ray_renderer.resetCPUAccumulation();
                            g_cpu_sync_pending = false;
                            
                            SCENE_LOG_INFO("[CPU Sync] Updated vertices and performed SYNCHRONOUS BVH rebuild");
                        }
                        
                        // OPTIMIZATION: Only update animation state when timeline frame changed
                        // AND when we have file-based animations (not manual keyframes)
                        static int last_cpu_anim_frame = -1;
                        bool has_file_animations = !scene.animationDataList.empty();
                        if (current_f != last_cpu_anim_frame && has_file_animations) {
                             ray_renderer.updateAnimationState(scene, time);
                             last_cpu_anim_frame = current_f;
                        }
                        
                        // Set camera Y for volumetric cloud parallax
                        if (scene.camera) {
                            ray_renderer.world.setCameraY(scene.camera->lookfrom.y);
                        }
                        
                        // Single pass render (1 sample) - uses accumulation internally
                        auto sample_start = std::chrono::high_resolution_clock::now();
                        
                        ray_renderer.render_progressive_pass(surface, window, scene, 1);
                        
                        auto sample_end = std::chrono::high_resolution_clock::now();
                        float sample_time_ms = std::chrono::duration<float, std::milli>(sample_end - sample_start).count();
                        
                        // Update progress for UI
                        int prev_samples = render_settings.render_current_samples;
                        render_settings.render_current_samples = ray_renderer.getCPUAccumulatedSamples();
                        
                        int effective_max_samples = render_settings.is_final_render_mode ? render_settings.final_render_samples : render_settings.max_samples;
                        render_settings.render_target_samples = effective_max_samples > 0 ? effective_max_samples : 100;
                        
                        render_settings.render_progress = (float)render_settings.render_current_samples / render_settings.render_target_samples;
                        render_settings.is_rendering_active = !ray_renderer.isCPUAccumulationComplete();
                        
                        // Update time estimation
                        if (render_settings.render_current_samples > prev_samples) {
                            // Moving average for sample time
                            render_settings.avg_sample_time_ms = render_settings.avg_sample_time_ms * 0.8f + sample_time_ms * 0.2f;
                            render_settings.render_elapsed_seconds += sample_time_ms / 1000.0f;
                            
                            int remaining_samples = render_settings.render_target_samples - render_settings.render_current_samples;
                            render_settings.render_estimated_remaining = (remaining_samples * render_settings.avg_sample_time_ms) / 1000.0f;
                        }
                        
                        // ===== DENOISE LOGIC =====
                        bool effective_denoiser = render_settings.is_final_render_mode ? render_settings.render_use_denoiser : render_settings.use_denoiser;
                        if (effective_denoiser && ray_renderer.getCPUAccumulatedSamples() > 0) {
                                 ray_renderer.applyOIDNDenoising(surface, 0, true,
                                     ui_ctx.render_settings.denoiser_blend_factor);
                             }
                        }
                    }
                 }
            }

        // Tonemap reset ve apply
        if (!start_render) {
            if (reset_tonemap) {
                SDL_BlitSurface(original_surface, nullptr, surface, nullptr);               
                reset_tonemap = false;
                SCENE_LOG_INFO("Tonemap reset applied.");
            }

            if (apply_tonemap) {
                applyToneMappingToSurface(surface, original_surface, color_processor);               
                apply_tonemap = false;
                SCENE_LOG_INFO("Tonemap applied.");
            }
        }


        // Image save
        if (ui_ctx.render_settings.save_image_requested && original_surface) {
            ui_ctx.render_settings.save_image_requested = false;

            std::string path = saveFileDialogW(L"PNG Dosyaları\0*.png\0Tüm Dosyalar\0*.*\0");
            if (!path.empty()) {
                if (SaveSurface(surface, path.c_str())) {
                    SCENE_LOG_INFO("Image saved to: " + path);
                }
                else {
                    SCENE_LOG_ERROR("Image save failed!");
                }
            }
        }
        
        // ---- Animation Playback Preview ----

        static int last_playback_frame = -1;

        // Check for frame changes (Scrubbing or Playback)
        int current_playback_frame = ui_ctx.render_settings.animation_playback_frame;
        if (current_playback_frame != last_playback_frame) {
            
            // Update 3D Scene State for Live Preview
            if (!rendering_in_progress && scene.initialized) {
                float fps = ui_ctx.render_settings.animation_fps;
                if (fps <= 0.0f) fps = 24.0f;
                
                int start_frame = ui_ctx.render_settings.animation_start_frame;
                float time = (current_playback_frame - start_frame) / fps;
                
                // PERFORMANCE OPTIMIZATION: 
                // Check if ANY animation data exists (file-based OR manual keyframes)
                // If no animation at all, SKIP all expensive updates - nothing to animate!
                bool has_file_animations = !scene.animationDataList.empty();
                bool has_timeline_tracks = !scene.timeline.tracks.empty();
                
                // Count actual keyframes (not just tracks) - CACHE THIS to avoid repeated iteration
                static size_t cached_total_keyframes = 0;
                static bool keyframe_cache_valid = false;
                
                if (!keyframe_cache_valid || has_timeline_tracks) {
                    cached_total_keyframes = 0;
                    for (const auto& [name, track] : scene.timeline.tracks) {
                        cached_total_keyframes += track.keyframes.size();
                    }
                    keyframe_cache_valid = true;
                }
                bool has_manual_keyframes = cached_total_keyframes > 0;
                
                // DEBUG: Uncomment to trace animation state
                // SCENE_LOG_INFO("[ANIM DEBUG] Frame " + std::to_string(current_playback_frame) + 
                //                " | FileAnims=" + std::to_string(has_file_animations) +
                //                " | Tracks=" + std::to_string(scene.timeline.tracks.size()) +
                //                " | Keyframes=" + std::to_string(cached_total_keyframes));
                
                // SKIP EVERYTHING if no animations exist at all
                if (!has_file_animations && !has_manual_keyframes) {
                    // No animation data - just reset accumulation and skip geometry updates
                    if (ui_ctx.render_settings.use_optix && g_hasOptix) {
                        optix_gpu.resetAccumulation();
                    } else {
                        ray_renderer.resetCPUAccumulation();
                    }
                    start_render = true;
                    last_playback_frame = current_playback_frame;
                    // FAST PATH: Skip all expensive work below
                }
                else {
                    // We have some animation data - process accordingly
                    
                    if (has_file_animations) {
                        // File-based animations present - need full update (Assimp skinning, node hierarchy)
                        ray_renderer.updateAnimationState(scene, time);
                    }
                    // else: TimelineWidget::draw() handles manual keyframes with O(1) object lookup - FAST!
                    
                    // Mark all buffers dirty for animation frame change
                    g_camera_dirty = true;
                    g_lights_dirty = true;
                    g_world_dirty = true;
                    
                    // Update OptiX if needed
                    if (ui_ctx.render_settings.use_optix && g_hasOptix) {
                        // PERFORMANCE: Only update geometry if file-based animations modified it
                        if (has_file_animations) {
                            optix_gpu.updateGeometry(scene.world.objects);
                        } else if (wind_active) {
                            // FAST PATH: Wind only updates matrices, not vertex geometry.
                            optix_gpu.updateTLASMatricesOnly(scene.world.objects);
                        } else if (has_manual_keyframes) {
                            // PERFORMANCE CRITICAL: 
                            // updateTLASMatricesOnly calls syncInstanceTransforms which iterates ALL 2M objects!
                            // This was causing 5 second delays per frame.
                            //
                            // TimelineWidget::draw() already updated the transforms via th->setBase().
                            // We DON'T need to sync transforms from CPU Triangle objects to GPU instances.
                            // Instead, we just need to trigger a TLAS rebuild with the EXISTING instance data.
                            //
                            // The GPU instances already have correct transforms from the last build.
                            // Manual keyframe transforms are applied to CPU Triangle->TransformHandle,
                            // which OptiX instances already reference.
                            //
                            // SKIP: optix_gpu.updateTLASMatricesOnly(scene.world.objects);
                            // SKIP: g_gpu_refit_pending = true; // This also calls updateTLASMatricesOnly!
                            // 
                            // TimelineWidget::draw() already updated instance transforms and called rebuildTLAS()
                            // We don't need to do anything here - just let the updated TLAS be used.
                            //
                            // REMOVED: optix_gpu.rebuildTLAS(); // Already done in TimelineWidget
                        }
                        // If !has_file_animations && !has_manual_keyframes, we don't reach here
                        
                        if (scene.camera) optix_gpu.setCameraParams(*scene.camera);
                        optix_gpu.setLightParams(scene.lights);
                        
                        // Update GPU materials for material keyframe animation
                        ray_renderer.updateOptiXMaterialsOnly(scene, &optix_gpu);
                        
                        // Reset accumulation for new frame
                        optix_gpu.resetAccumulation();
                        
                        // Clear dirty flags since we just updated manually
                        g_camera_dirty = false;
                        g_lights_dirty = false;
                        g_world_dirty = false;
                    } else {
                        // Reset CPU accumulation for new frame
                        ray_renderer.resetCPUAccumulation();
                    }
                    
                    // Trigger preview render
                    start_render = true;
                }
            }
            
            last_playback_frame = current_playback_frame;
        }
        
        // ============ CYCLES-STYLE AUTO-PROGRESSIVE ACCUMULATION ============
        // When camera is stationary, keep accumulating samples until max is reached
        // SKIP during animation playback or when paused
        bool is_playing = ui_ctx.render_settings.animation_is_playing;
        bool is_paused = ui_ctx.render_settings.is_render_paused;
        
        if (scene.initialized && 
            !camera_moved_recently &&
            !start_render &&
            !is_playing &&
            !is_paused) {  // Don't accumulate when paused
            
            bool accumulation_complete = false;
            
            if (ui_ctx.render_settings.use_optix && g_hasOptix) {
                accumulation_complete = optix_gpu.isAccumulationComplete();
            } else {
                accumulation_complete = ray_renderer.isCPUAccumulationComplete();
            }
            
            if (!accumulation_complete) {
                // Automatically trigger next sample pass
                start_render = true;
            }
        }

        // ===========================================================================
        // TEXTURE UPDATE OPTIMIZATION
        // Only update texture when there's actual rendering happening
        // ===========================================================================
        bool accumulation_done_for_display = false;
        if (ui_ctx.render_settings.use_optix && g_hasOptix) {
            accumulation_done_for_display = optix_gpu.isAccumulationComplete();
        } else {
            accumulation_done_for_display = ray_renderer.isCPUAccumulationComplete();
        }
        
        // Only update texture if rendering is active or UI needs it
        static bool last_texture_updated = false;
        bool needs_texture_update = !accumulation_done_for_display || !last_texture_updated || ImGui::GetIO().WantCaptureMouse;
        
        if (needs_texture_update) {
            SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
            last_texture_updated = !accumulation_done_for_display;
        }
        
        // SDL_FreeSurface(original_surface);
        // original_surface = SDL_ConvertSurface(surface, surface->format, 0);
        ImGui::Render();       
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);

        // ===========================================================================
        // IDLE OPTIMIZATION: Longer sleep when render is complete
        // ===========================================================================
        if (accumulation_done_for_display && !camera_moved && !dragging && !start_render && !ImGui::GetIO().WantCaptureMouse) {
            // Render complete and no user interaction - aggressive sleep
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } else if (!ui_ctx.render_settings.is_rendering_active && !camera_moved) {
            // Not rendering but may have UI interaction - light sleep
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        
        // ===========================================================================
        // DEFERRED REBUILD PROCESSING - Batched at frame end for faster UI response
        // ===========================================================================
        if (g_mesh_cache_dirty) {
            ui.rebuildMeshCache(scene.world.objects);
            g_mesh_cache_dirty = false;
        }
        
        // ===========================================================================
        // ASYNC BVH REBUILD (Non-Blocking)
        // ===========================================================================
        if (g_gpu_refit_pending) {
            // ui.addViewportMessage("Updating GPU...", 2.0f); // Removed to prevent spam
            if (g_hasOptix) {
                // LIGHTWEIGHT UPDATE: Transforms only (fast)
                // If actual geometry (vertices) changed, use g_optix_rebuild_pending instead.
                optix_gpu.updateTLASMatricesOnly(scene.world.objects);
                
                optix_gpu.setLightParams(scene.lights);
                if (scene.camera) optix_gpu.setCameraParams(*scene.camera);
                optix_gpu.resetAccumulation();
            }
            g_gpu_refit_pending = false;
        }

        // CPU BVH Fast Refit (Embree only)
        // CPU BVH Fast Refit (Embree only)
        if (g_cpu_bvh_refit_pending && !g_bvh_rebuild_pending) {
            // Only update if not fully rebuilding
            bool use_embree = ui_ctx.render_settings.UI_use_embree;
            // FORCE REBUILD for correctness during drag (User Request: "CPU Sync")
            ray_renderer.rebuildBVH(scene, use_embree);
            ray_renderer.resetCPUAccumulation();
            g_cpu_bvh_refit_pending = false;
        }

        static std::future<std::shared_ptr<Hittable>> g_bvh_future;
        
        if (g_bvh_rebuild_pending) {
            // OPTIMIZATION: In GPU mode, skip CPU BVH rebuild entirely!
            // - GPU raytracing uses OptiX, not CPU BVH
            // - Picking (mouse selection) uses linear search, not BVH
            // - Copying 4M shared_ptr for async rebuild takes ~3 seconds due to atomic ref count increments
            // NOTE: Only skip if actually using GPU AND not switching to CPU mode
            // g_cpu_sync_pending means user switched from GPU to CPU - must rebuild!
            if (g_hasOptix && ui_ctx.render_settings.use_optix && !g_cpu_sync_pending) {
                // GPU mode active: Skip expensive CPU BVH rebuild
                g_bvh_rebuild_pending = false;
            }
            // Only trigger if not already rebuilding
            else if (!g_bvh_future.valid()) {
                // Determine if we actually need it (if using OptiX, maybe unnecessary?)
                // But picking (selection) uses CPU BVH. So we DO need it eventually.
                // Async allows it to happen without freezing.
                
                // Copy list of objects for thread safety (prevents crashes if objects deleted from vector)
                std::vector<std::shared_ptr<Hittable>> objects_copy = scene.world.objects;
                bool use_embree = ui_ctx.render_settings.UI_use_embree;
                
                ui.addViewportMessage("Rebuilding BVH...", 10.0f); // Show status
                
                // Capture by MOVE to avoid second copy (Save 1x copy of millions of shared_ptrs)
                g_bvh_future = std::async(std::launch::async, [objs = std::move(objects_copy), use_embree]() -> std::shared_ptr<Hittable> {
                     // Empty scene guard - prevents freeze when last object is deleted
                     if (objs.empty()) {
                         return nullptr;
                     }
                     
                     if (use_embree) {
                         auto bvh = std::make_shared<EmbreeBVH>();
                         bvh->build(objs);
                         return std::dynamic_pointer_cast<Hittable>(bvh);
                     } else {
                         // Fallback to ParallelBVH (CPU)
                         // Note: 0.0, 1.0 are time parameters for motion blur (shutter open/close)
                         return std::make_shared<ParallelBVHNode>(objs, 0, objs.size(), 0.0, 1.0, 0);
                     } 
                });
            }
            g_bvh_rebuild_pending = false; 
        }
        
        // Check Async Result
        if (g_bvh_future.valid()) {
            if (g_bvh_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    auto new_bvh = g_bvh_future.get();
                    if (new_bvh) {
                        // CRITICAL OPTIMIZATION: Destroy OLD BVH on a background thread!
                        // Replacing 'scene.bvh' triggers the destructor of the old BVH.
                        // For large scenes, recursively destroying millions of nodes on the Main Thread 
                        // causes a massive freeze (blocking GPU commands/UI).
                        auto old_bvh = scene.bvh;
                        scene.bvh = new_bvh;
                        
                        std::thread([old_bvh]() {
                            // old_bvh goes out of scope here and is destroyed in background
                        }).detach();
                        
                        ray_renderer.resetCPUAccumulation();
                        ui.clearViewportMessages(); 
                        ui.addViewportMessage("Async BVH Rebuild Complete", 2.0f);
                    }
                } catch (const std::exception& e) {
                   SCENE_LOG_WARN(std::string("BVH Rebuild Failed: ") + e.what());
                }
            }
        }
        
        // -----------------------------------------------------------------
        // ASYNC OPTIX REBUILD (Non-blocking)
        // -----------------------------------------------------------------
        static std::future<void> g_optix_future;
        static bool g_optix_rebuilding = false;
        
        if (g_optix_rebuild_pending && ui_ctx.render_settings.use_optix && g_hasOptix) {
            // Only start if not already rebuilding
            if (!g_optix_rebuilding) {
                ui.addViewportMessage("Rebuilding OptiX Geometry...", 10.0f);
                
                // Capture references for lambda
                auto& scene_ref = scene;
                auto optix_ptr = &optix_gpu;
                auto& renderer_ref = ray_renderer;
                
                // CRITICAL FIX: Set global flag to pause rendering in OptixWrapper
                g_optix_rebuild_in_progress = true;

                g_optix_future = std::async(std::launch::async, [&scene_ref, optix_ptr, &renderer_ref]() {
                    renderer_ref.rebuildOptiXGeometry(scene_ref, optix_ptr);
                    
                    // REAPPLY FOLIAGE persistence (since rebuild wipes instances)
                    // We must do this inside adding or after it finishes?
                    // Actually, TerrainManager manages its own instances in AccelManager.
                    // If rebuildOptiXGeometry wipes AccelManager, we must re-add them.
                    TerrainManager::getInstance().reapplyAllFoliage(optix_ptr);
                });
                
                g_optix_rebuilding = true;
            }
            g_optix_rebuild_pending = false;
        }
        
        // Check if async OptiX rebuild is complete
        if (g_optix_rebuilding && g_optix_future.valid()) {
            if (g_optix_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    g_optix_future.get();
                    optix_gpu.resetAccumulation();
                    ui.clearViewportMessages();
                    ui.addViewportMessage("OptiX Rebuild Complete", 2.0f);
                    start_render = true;
                } catch (const std::exception& e) {
                    SCENE_LOG_WARN(std::string("OptiX Rebuild Failed: ") + e.what());
                }
                g_optix_rebuilding = false;
                // CRITICAL FIX: Resume rendering
                g_optix_rebuild_in_progress = false;
            }
        }
        
        SDL_Delay(16); // ~60 FPS cap

    }
    
    SDL_DestroyTexture(raytrace_texture);
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_FreeSurface(surface);
    SDL_FreeSurface(original_surface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    g_sceneLog.closeLogFile();
    return 0;
}


