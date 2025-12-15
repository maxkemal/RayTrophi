#include <SDL_main.h> 
#include <fstream>
#include <locale>
#include <SDL_image.h>
#include "Renderer.h"
#include "CPUInfo.h"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"  // Değiştirildi: sdlrenderer2
#include <scene_ui.h>
#include "ColorProcessingParams.h"
#include <filesystem>
#include <windows.h>
#include <commdlg.h>

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
float mouse_sensitivity = 0.1f;
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
std::mutex surface_mutex;  // Surface erişimi için mutex
SceneUI ui;
SceneData scene;
Renderer ray_renderer(image_width, image_height, 1, 1);
OptixWrapper optix_gpu;
ColorProcessor color_processor(image_width, image_height);
std::string active_model_path;
UIContext ui_ctx{
   scene,
   ray_renderer,
   &optix_gpu,
   color_processor,
   render_settings,
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
    SDL_SetWindowSize(window, w, h);
    
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

    SCENE_LOG_INFO("Render resolution updated: " + std::to_string(w) + "x" + std::to_string(h));
}
static bool isRTX(int major, int minor)
{
    // RTX donanımı SM 7.5 ile başladı.
    if (major > 7) return true;            // SM 8.x, 9.x, 10.x → yeni RTX mimarileri
    if (major == 7 && minor >= 5) return true; // SM 7.5 → Turing
    return false;
}

void detectOptixHardware()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    // NVIDIA kart yok / CUDA yok / sürücü yok → OptiX yok.
    if (err != cudaSuccess || deviceCount == 0) {
        g_hasOptix = false;
        SCENE_LOG_WARN("No CUDA-capable devices found. OptiX will be disabled.");
        return;
    }

    bool foundRTX = false;

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int major = prop.major;
        int minor = prop.minor;

        if (isRTX(major, minor)) {
            foundRTX = true;
            SCENE_LOG_INFO("Found RTX-capable device: " + std::string(prop.name) +
                " (SM " + std::to_string(major) + "." + std::to_string(minor) + ")");

            break;
        }
    }

    g_hasOptix = foundRTX;
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

     window = SDL_CreateWindow("RayTrophi",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        image_width,
        image_height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

     renderer =
        SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

     surface = SDL_GetWindowSurface(window);

     raytrace_texture =
        SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
            SDL_TEXTUREACCESS_STREAMING, image_width, image_height);

     original_surface =
        SDL_ConvertSurface(surface, surface->format, 0);

    // Artık pencere var, maximize edebilirsin
    SDL_MaximizeWindow(window);
    g_sceneLog.clear();
    detectOptixHardware();
    // SDL başlatıldıktan hemen sonra, render döngüsünden önce ekle
    SDL_Surface* splash = IMG_Load("RayTrophi_image.png");
    if (splash) {
        SDL_BlitScaled(splash, nullptr, surface, nullptr); // splash'ı surface'a bas
        SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch); // texture güncelle
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        SDL_FreeSurface(splash); // işimiz bitti
    }
    else {
        SCENE_LOG_ERROR(std::string("Splash görseli yüklenemedi: ") + IMG_GetError());
    }

    SDL_SetTextureBlendMode(raytrace_texture, SDL_BLENDMODE_NONE);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);
   // ImGui::StyleColorsDark();

	init_RayTrophi_Pro_Dark_Thema();// Özel tema uygula 

    if (initializeOptixIfAvailable(optix_gpu)) {
        SCENE_LOG_INFO("OptiX is ready!");
        render_settings.use_optix = true;
        ui_ctx.render_settings.use_optix = true;
    }
    else {
        SCENE_LOG_WARN("Falling back to CPU rendering.");
    }

    
    SDL_Event e;
    while (!quit) {
        camera_moved = false;
       // start_render = false;
       
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
        }
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        ui.draw(ui_ctx);
        
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
                    rayhit = true;
                    mx = e.button.x;
                    my = e.button.y;
                    my = image_height - my;
                    u = (mx + 0.5f) / image_width;
                    v = (my + 0.5f) / image_height;

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
                            yaw += dx * mouse_sensitivity;
                            pitch -= dy * mouse_sensitivity;
                            pitch = std::clamp(pitch, -89.9f, 89.9f);

                            float rad_yaw = yaw * 3.14159265f / 180.0f;
                            float rad_pitch = pitch * 3.14159265f / 180.0f;

                            Vec3 direction;
                            direction.x = cosf(rad_yaw) * cosf(rad_pitch);
                            direction.y = sinf(rad_pitch);
                            direction.z = sinf(rad_yaw) * cosf(rad_pitch);

                            direction = direction.normalize();
                            scene.camera->setLookDirection(direction);
                        }
                    }

                    last_mouse_x = e.motion.x;
                    last_mouse_y = e.motion.y;
                    last_camera_move_time = std::chrono::steady_clock::now();
                    start_render = true;
                }
            }

            if (e.type == SDL_MOUSEWHEEL && mouse_control_enabled && scene.camera) {
                // Eğer mouse ImGui üzerinde değilse
                if (!ImGui::GetIO().WantCaptureMouse) {
                    float scroll_amount = e.wheel.y;  // yukarı: pozitif, aşağı: negatif
                    float move_speed = 1.5f; // İstersen ayarlanabilir yap
                    Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                    scene.camera->lookfrom += forward * scroll_amount * move_speed;
                    scene.camera->lookat = scene.camera->lookfrom + forward * scene.camera->focus_dist;

                    scene.camera->update_camera_vectors();
                    last_camera_move_time = std::chrono::steady_clock::now();
                    camera_moved = true;
                    start_render = true;
                }
            }


            if (e.type == SDL_QUIT) quit = true;
        }
        const Uint8* key_state = SDL_GetKeyboardState(NULL);

        if (mouse_control_enabled && scene.camera) {

            Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
            Vec3 right = Vec3::cross(forward, scene.camera->vup).normalize();
            Vec3 up = scene.camera->vup;

            if (key_state[SDL_SCANCODE_W]) {
                scene.camera->lookfrom += forward * move_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_S]) {
                scene.camera->lookfrom -= forward * move_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_A]) {
                scene.camera->lookfrom -= right * move_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_D]) {
                scene.camera->lookfrom += right * move_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_E]) {
                scene.camera->lookfrom += up * move_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_Q]) {
                scene.camera->lookfrom -= up * move_speed;
                camera_moved = true;
            }

            if (camera_moved) {
                Vec3 dir = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                scene.camera->lookat = scene.camera->lookfrom + dir * scene.camera->focus_dist;
                scene.camera->update_camera_vectors();
                last_camera_move_time = std::chrono::steady_clock::now();
                start_render = true;

            }
            // hareket durmuşsa foveation seviyesi büyüt
            if (camera_moved_recently &&
                std::chrono::steady_clock::now() - last_camera_move_time > std::chrono::milliseconds(50)) {
                camera_moved_recently = false;

            }

        }

             
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
                    
                    start_render = false; // Cancel any pending interactive render
                    
                    SCENE_LOG_INFO("Starting animation render...");
                    std::string output_folder = "render_animation";
                    SCENE_LOG_INFO("Output folder set to: " + output_folder);

                    // Capture local copies of settings to avoid thread race
                    int anim_sample_count = sample_count;
                    int anim_sample_per_pass = sample_per_pass;
                    int anim_fps = ui_ctx.render_settings.animation_fps;
                    float anim_duration = ui_ctx.render_settings.animation_duration;
                    bool anim_use_denoiser = ui_ctx.render_settings.use_denoiser;
                    float anim_denoiser_blend = ui_ctx.render_settings.denoiser_blend_factor;
                    bool anim_use_optix = ui_ctx.render_settings.use_optix;
                    
                    // Detach thread
                    std::thread anim_thread([=]() {
                        ray_renderer.render_Animation(surface, window, raytrace_texture, renderer,
                            anim_sample_count, anim_sample_per_pass,
                            anim_fps, anim_duration,
                            scene,
                            output_folder,
                            anim_use_denoiser,
                            anim_denoiser_blend,
                            &optix_gpu,
                            anim_use_optix);
                            
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
                     rendering_stopped_gpu = false;

                    // --- Animation State Update ---
                    float fps = ui_ctx.render_settings.animation_fps;
                    if (fps <= 0.0f) fps = 24.0f;
                    int start_f = ui_ctx.render_settings.animation_start_frame;
                    int current_f = ui_ctx.render_settings.animation_playback_frame;
                    float time = (current_f - start_f) / fps;

                    if (ui_ctx.render_settings.use_optix && g_hasOptix) {
                        // ============ SYNCHRONOUS OPTIX RENDER (No Thread) ============
                        // Each pass is ~10-50ms for 1 sample, fast enough for UI
                        
                        bool geometry_updated = ray_renderer.updateAnimationState(scene, time);
                        
                        if (geometry_updated) {
                            optix_gpu.updateGeometry(scene.world.objects);
                        }
                        
                        if (scene.camera) {
                            optix_gpu.setCameraParams(*scene.camera);
                            Vec3 dir = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                            yaw = atan2f(dir.z, dir.x) * 180.0f / 3.14159265f;
                            pitch = asinf(dir.y) * 180.0f / 3.14159265f;
                        }
                        if (!scene.lights.empty())
                            optix_gpu.setLightParams(scene.lights);
                        optix_gpu.setBackgroundColor(scene.background_color);

                        std::vector<uchar4> framebuffer(image_width * image_height);
                        
                        // Single pass render (1 sample) - fast, no UI blocking
                        auto sample_start = std::chrono::high_resolution_clock::now();
                        
                        optix_gpu.launch_random_pixel_mode_progressive(
                            surface, window, renderer, 
                            image_width, image_height, 
                            framebuffer, raytrace_texture
                        );
                        
                        auto sample_end = std::chrono::high_resolution_clock::now();
                        float sample_time_ms = std::chrono::duration<float, std::milli>(sample_end - sample_start).count();
                        
                        // Update progress for UI
                        int prev_samples = render_settings.render_current_samples;
                        render_settings.render_current_samples = optix_gpu.getAccumulatedSamples();
                        render_settings.render_target_samples = render_settings.max_samples > 0 ? render_settings.max_samples : 100;
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
                        
                        // ===== ALWAYS DENOISE WHEN ENABLED =====
                        // Denoise after every sample - user never sees noisy image
                        if (ui_ctx.render_settings.use_denoiser && optix_gpu.getAccumulatedSamples() > 0) {
                            ray_renderer.applyOIDNDenoising(surface, 0, true, ui_ctx.render_settings.denoiser_blend_factor);
                        }
                    }
                    else {
                        // ============ SYNCHRONOUS CPU RENDER (Like OptiX) ============
                        // Each pass is 1 sample per pixel, accumulates progressively
                        
                        ray_renderer.updateAnimationState(scene, time);
                        
                        // Single pass render (1 sample) - uses accumulation internally
                        auto sample_start = std::chrono::high_resolution_clock::now();
                        
                        ray_renderer.render_progressive_pass(surface, window, scene, 1);
                        
                        auto sample_end = std::chrono::high_resolution_clock::now();
                        float sample_time_ms = std::chrono::duration<float, std::milli>(sample_end - sample_start).count();
                        
                        // Update progress for UI
                        int prev_samples = render_settings.render_current_samples;
                        render_settings.render_current_samples = ray_renderer.getCPUAccumulatedSamples();
                        render_settings.render_target_samples = render_settings.max_samples > 0 ? render_settings.max_samples : 100;
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
                        
                        // ===== ALWAYS DENOISE WHEN ENABLED =====
                        // Denoise after every sample - user never sees noisy image
                        if (ui_ctx.render_settings.use_denoiser && ray_renderer.getCPUAccumulatedSamples() > 0) {
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
                
                // Update animation state immediately
                ray_renderer.updateAnimationState(scene, time);
                
                // Update OptiX if needed
                if (ui_ctx.render_settings.use_optix && g_hasOptix) {
                    optix_gpu.updateGeometry(scene.world.objects);
                    if (scene.camera) optix_gpu.setCameraParams(*scene.camera);
                    optix_gpu.setLightParams(scene.lights);
                    // Reset accumulation for new frame
                    optix_gpu.resetAccumulation();
                } else {
                    // Reset CPU accumulation for new frame
                    ray_renderer.resetCPUAccumulation();
                }
                
                // Trigger preview render
                start_render = true;
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

        SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
        SDL_FreeSurface(original_surface);
        original_surface = SDL_ConvertSurface(surface, surface->format, 0);
        ImGui::Render();       
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
        SDL_Delay(16); // ~60 FPS

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
