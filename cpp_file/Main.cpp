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

std::string saveFileDialog(const char* filter = "PNG File\0*.png\0") {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));

    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL; // SDL window handle verilebilir
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_OVERWRITEPROMPT; //  önemli: üzerine yaz uyarısı verir
    ofn.lpstrTitle = "Save Render Image";

    if (GetSaveFileNameA(&ofn)) {
        return std::string(filename);
    }
    return ""; // iptal edildi
}


bool SaveSurface(SDL_Surface* surface, const char* file_path) {
    SDL_Surface* surface_to_save = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);
    /* int imgFlags = IMG_INIT_PNG;
     if (!(IMG_Init(imgFlags) & imgFlags)) {
         SDL_Log("SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError());
         SDL_Quit();
         return 1;
     }*/

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
int sample_count = 1;
float yaw = -90.0f;
float pitch = 0.0f;
bool dragging = false;
int last_mouse_x = 0;
int last_mouse_y = 0;
float move_speed = 0.5f;
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
void reset_render_resolution(int w, int h) {
    // 1. SDL Surfaceleri sıfırla
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
	
    // 2. CPU Renderer’e yeni boyutu bildir
    ray_renderer.resetResolution(w, h);

    // 3. OptiX GPU buffer’larını yeniden başlat
    optix_gpu.resetBuffers(w, h);
   
    // 4. Tonemapping ve post-process ayarlarını yeniden yapılandır (gerekirse)
    color_processor.resize(w, h);

    std::cout << "Render resolution updated: " << w << "x" << h << std::endl;
}
int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Turkish");
   
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }
   
    SDL_SetTextureBlendMode(raytrace_texture, SDL_BLENDMODE_NONE);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);
    ImGui::StyleColorsDark();
   
    if (use_optix) {
        optix_gpu.initialize();
        std::ifstream file("E:/visual studio proje c++/raytracing_Proje_Moduler/raytrac_sdl2/source/cpp_file/raygen.ptx");
        if (file.is_open()) {
            std::string ptx((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            optix_gpu.setupPipeline(ptx.c_str());
        }
        else {
            std::cerr << "Failed to open PTX file!" << std::endl;
            use_optix = false;
        }
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
        if (start_render && scene.initialized) {
            start_render = false;
            auto start_time = std::chrono::high_resolution_clock::now();  // Zaman başlat
           
            if (ui_ctx.render_settings.use_optix) {
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

                optix_gpu.launch_random_pixel_mode(surface, window, image_width, image_height, framebuffer);
				if (ui_ctx.render_settings.use_denoiser) {
					//std::cout << "Applying OIDN Denoising..." << std::endl;
                    ray_renderer.applyOIDNDenoising(surface, 0, true, ui_ctx.render_settings.denoiser_blend_factor);
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
                    ray_renderer.applyOIDNDenoising(surface, 0, true, ui_ctx.render_settings.denoiser_blend_factor);
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
                    ray_renderer.applyOIDNDenoising(surface, 0, true, denoiser_blend_factor);
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
            std::string path = saveFileDialog("PNG Files\0*.png\0");
            if (!path.empty()) {
                SaveSurface(surface, path.c_str());
                SDL_Log("Image saved to: %s", path.c_str());
            }
        }

        ImGui::Render();
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
        std::this_thread::sleep_for(std::chrono::milliseconds(16));  // C++17 alternatifi

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
