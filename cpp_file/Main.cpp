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
    if (surface_to_save == NULL) {
        SDL_Log("Couldn't convert surface: %s", SDL_GetError());
        return false;
    }

    int result = IMG_SavePNG(surface_to_save, file_path);
    SDL_FreeSurface(surface_to_save);

    if (result != 0) {
        SDL_Log("Failed to save image: %s", IMG_GetError());
        return false;
    }

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
        // UTF-16 → UTF-8 dönüşümü
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, filename, -1, nullptr, 0, nullptr, nullptr);
        std::string utf8_path(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, utf8_path.data(), size_needed, nullptr, nullptr);

        // Boş terminatörü kaldır
        if (!utf8_path.empty() && utf8_path.back() == '\0')
            utf8_path.pop_back();

        // Eğer kullanıcı uzantıyı yazmadıysa otomatik ekle
        std::string lower = utf8_path;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find(".png") == std::string::npos)
            utf8_path += ".png";

        return utf8_path;
    }
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

SceneUI ui;
SceneData scene;
Renderer ray_renderer(image_width, image_height, 1, 1);
OptixWrapper optix_gpu;
ColorProcessor color_processor(image_width, image_height);
std::string active_model_path;
SDL_Window* window = SDL_CreateWindow("RayTrophi", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
    image_width, image_height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
SDL_Surface* surface = SDL_GetWindowSurface(window);

SDL_Texture* raytrace_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
    SDL_TEXTUREACCESS_STREAMING, image_width, image_height);
SDL_Surface* original_surface = SDL_ConvertSurface(surface, surface->format, 0);
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
    // 1. SDL surface ve texture'ları sıfırla
    if (raytrace_texture) SDL_DestroyTexture(raytrace_texture);
    if (surface) SDL_FreeSurface(surface);
    if (original_surface) SDL_FreeSurface(original_surface);

    surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    original_surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    raytrace_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING, w, h);

    if (!surface || !original_surface || !raytrace_texture) {
        std::cerr << "Failed to create SDL surfaces or texture!" << std::endl;
        return;
    }

    // 2. CPU renderer'ı güncelle
    ray_renderer.resetResolution(w, h);

    // 3. OptiX GPU bufferlarını yalnızca GPU aktifse resetle
    if (use_optix && g_hasOptix)
    {
        optix_gpu.resetBuffers(w, h);
    }

    // 4. Color processor (CPU-side tone mapping)
    color_processor.resize(w, h);

    std::cout << "Render resolution updated: " << w << "x" << h << std::endl;
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
		std::cout << "No CUDA-capable devices found. OptiX will be disabled." << std::endl;
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
			std::cout << "Found RTX-capable device: " << prop.name << " (SM " << major << "." << minor << ")" << std::endl;
            break;
        }
    }

    g_hasOptix = foundRTX;
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
            std::wcerr << L"Failed to open PTX file: " << ptx_path << std::endl;
            g_hasOptix = false;
            return false;
        }

        std::string ptx((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());
        optix_gpu.setupPipeline(ptx.c_str());
    }
    catch (const std::exception& e) {
        std::cerr << "OptiX initialization failed: " << e.what() << std::endl;
        g_hasOptix = false;
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Turkish");
  
		std::cout << "RayTrophi Render Engine Launched" << std::endl;
        detectOptixHardware();
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }
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
        SDL_Log("Splash görseli yüklenemedi: %s", IMG_GetError());
    }

    SDL_SetTextureBlendMode(raytrace_texture, SDL_BLENDMODE_NONE);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);
    ImGui::StyleColorsDark();
   
    if (initializeOptixIfAvailable(optix_gpu)) {
        std::cout << "OptiX is ready!" << std::endl;
    }
    else {
        std::cout << "Falling back to CPU rendering." << std::endl;
    }


    SDL_Event e;
    while (!quit) {
        camera_moved = false;
		start_render = false;
        if (pending_resolution_change) {
            pending_resolution_change = false;
            image_width = pending_width;
            image_height = pending_height;
            reset_render_resolution(image_width, image_height);

        }
        while (SDL_PollEvent(&e)) {
           
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
                if (!ImGui::GetIO().WantCaptureMouse) {
                    int dx = e.motion.x - last_mouse_x;
                    int dy = e.motion.y - last_mouse_y;

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
       
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ui.draw(ui_ctx);

        auto now = std::chrono::steady_clock::now();
        auto delta_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_camera_move_time).count();
        camera_moved_recently = (delta_ms < 50);
        if (rayhit && scene.initialized) {
            rayhit = false;
            start_render = true;
            ray = scene.camera->get_ray(u, v);
            if (scene.bvh->hit(ray, 0.00001f, 1e10f, hit_record)) {						
                scene.camera->focus_dist = hit_record.t;
               
            }

            else {
                std::cout << "No hit detected." << std::endl;
            }           
        }
        if (start_render && scene.initialized) {
            start_render = false;
            auto start_time = std::chrono::high_resolution_clock::now();  // Zaman başlat
           
            if (ui_ctx.render_settings.use_optix&& g_hasOptix) {
                //std::cout << "Starting Optix Render... " << std::endl;
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
                optix_gpu.resetBuffers(image_width, image_height);

                // Bu fonksiyon artık kendi içinde progressive update yapacak
                optix_gpu.launch_random_pixel_mode_progressive(surface, window, image_width, image_height, framebuffer, raytrace_texture);

                if (ui_ctx.render_settings.use_denoiser) {
                    //std::cout << "Applying OIDN Denoising..." << std::endl;
                    optix_gpu.applyOIDNDenoising(surface, true, ui_ctx.render_settings.denoiser_blend_factor);
                    // Denoising sonrası da texture'ı güncelle
                    
                }
                SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
                SDL_FreeSurface(original_surface);
                original_surface = SDL_ConvertSurface(surface, surface->format, 0);
            }
            else {
               // std::cout << "Starting CPU Render... " << std::endl;
                Vec3 dir = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                yaw = atan2f(dir.z, dir.x) * 180.0f / 3.14159265f;
                pitch = asinf(dir.y) * 180.0f / 3.14159265f;
               
                ray_renderer.render_image(surface, window, sample_count, sample_per_pass, scene);
                if (ui_ctx.render_settings.use_denoiser) {
                   // std::cout << "Applying OIDN Denoising..." << std::endl;
                    optix_gpu.applyOIDNDenoising(surface, true, ui_ctx.render_settings.denoiser_blend_factor);
                }
                SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
                SDL_FreeSurface(original_surface);
                original_surface = SDL_ConvertSurface(surface, surface->format, 0);
            }
			if (ui_ctx.render_settings.start_animation_render&& scene.initialized) {
				//std::cout << "Starting Animation Render... " << std::endl;
                Vec3 dir = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                yaw = atan2f(dir.z, dir.x) * 180.0f / 3.14159265f;
                pitch = asinf(dir.y) * 180.0f / 3.14159265f;
				// Animation render işlemi				
				ray_renderer.render_Animation(surface, window, sample_count, sample_per_pass, ui_ctx.render_settings.animation_fps, ui_ctx.render_settings.animation_duration, scene);
                if (render_settings.use_denoiser) {
                  //  std::cout << "Applying OIDN Denoising..." << std::endl;
                    optix_gpu.applyOIDNDenoising(surface, true, denoiser_blend_factor);
                }
                SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
                SDL_FreeSurface(original_surface);
                original_surface = SDL_ConvertSurface(surface, surface->format, 0);
                render_settings.start_animation_render = false;  // Animasyon render işlemi tamamlandıktan sonra false yap
			}
            auto end_time = std::chrono::high_resolution_clock::now();  // Zaman bitir
            float render_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count()/1000.0f;
            //std::cout << "Rendering Completed. Render time: " << render_ms << " seconds\n";
			
            last_render_time_ms = render_ms;  // ImGui’de göstermek istersen
           
        }

        if (!start_render) {
            if (reset_tonemap) {
                SDL_BlitSurface(original_surface, nullptr, surface, nullptr);
                SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
                reset_tonemap = false;
            }
			
           
            if (apply_tonemap) {
                applyToneMappingToSurface(surface, original_surface, color_processor);
                SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
                apply_tonemap = false;
            }
        }
        if (ui_ctx.render_settings.save_image_requested && original_surface) {
            ui_ctx.render_settings.save_image_requested = false;

            std::string path = saveFileDialogW(L"PNG Dosyaları\0*.png\0Tüm Dosyalar\0*.*\0");
            if (!path.empty()) {
                if (SaveSurface(surface, path.c_str())) {
                    SDL_Log("Image saved to: %s", path.c_str());
                }
                else {
                    SDL_Log("Image save failed!");
                }
            }
        }


        ImGui::Render();
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
        std::this_thread::sleep_for(std::chrono::milliseconds(32));  // C++17 alternatifi

       //SDL_Delay(32);
    }
   

    SDL_DestroyTexture(raytrace_texture);
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_FreeSurface(surface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
