#include <SDL_main.h> 
#include <fstream>
#include <locale>
#include <chrono>
#include <vector>
#include <atomic>
#include <thread>
#include <future>
#include <algorithm>
#include <execution>
#include <numeric>
#include <exception>
#include <csignal>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <string_view>
#include <SDL_image.h>
#include "Renderer.h"
#include "Backend/IBackend.h"
#include "Backend/OptixBackend.h"
#include "Backend/VulkanBackend.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanViewportBackend.h"
#include "Core/RenderStateManager.h"
#include "CPUInfo.h"

#include "imgui.h"
#include "ImGuizmo.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"  // De�i�tirildi: sdlrenderer2
#include <scene_ui.h>

#include "EmbreeBVH.h"
#include "ParallelBVHNode.h"
#include "scene_ui_guides.hpp"  // Viewport guides (safe areas, letterbox, grids)
#include "default_scene_creator.hpp"
#include "ColorProcessingParams.h"
#include "Stylize/StylizePostProcess.h"
#include "Stylize/StylizeKernel.h"
#include "CameraPresets.h"
#include "Jolt/JoltSmokeTest.h"  // Faz 0: --jolt-selftest link/runtime proof
#include "scene_data.h"       // Added explicit include
#include "AnimationNodes.h"
#include "OptixWrapper.h"     // Added explicit include
#include "SceneSelection.h"   // Added explicit include
#include "Triangle.h"         // Added for CPU sync
#include "HittableInstance.h" // Added for instance CPU sync
#include "MaterialManager.h"
#include "WaterSystem.h"      // Added for WaterManager
#include "VDBVolumeManager.h"

#include <filesystem>
#include <windows.h>
#include <commdlg.h>
#include <malloc.h>
#include "SplashScreen.h"  // Splash screen support
#include <cstdlib>
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
#include "DllLoadPolicy.h"

namespace {
bool g_startupDiagVerbose = false;
constexpr const char* kStartupDiagVersion = "startup_diag_v3_2026_03_01";

constexpr DWORD kMsCppException = 0xE06D7363;
constexpr DWORD kDbgPrintExceptionC = 0x40010006;
constexpr DWORD kDbgPrintExceptionWideC = 0x4001000A;

bool isBenignStatusException(DWORD code) {
    return code == kDbgPrintExceptionC ||
           code == kDbgPrintExceptionWideC ||
           code == EXCEPTION_BREAKPOINT ||
           code == EXCEPTION_SINGLE_STEP;
}

bool isFatalExceptionCode(DWORD code) {
    return code == EXCEPTION_ACCESS_VIOLATION ||
           code == EXCEPTION_STACK_OVERFLOW ||
           code == EXCEPTION_ILLEGAL_INSTRUCTION ||
           code == EXCEPTION_IN_PAGE_ERROR ||
           code == EXCEPTION_ARRAY_BOUNDS_EXCEEDED ||
           code == EXCEPTION_DATATYPE_MISALIGNMENT ||
           code == EXCEPTION_INT_DIVIDE_BY_ZERO ||
           code == EXCEPTION_PRIV_INSTRUCTION ||
           code == 0xC0000374; // STATUS_HEAP_CORRUPTION
}

const char* exceptionSeverityTag(DWORD code) {
    if (isFatalExceptionCode(code)) return "FATAL";
    if (code == kMsCppException) return "WARN";  // often first-chance, may be handled safely
    return "WARN";
}

bool shouldLogExceptionOnce(DWORD code, const void* address) {
    static std::mutex seenLock;
    static std::unordered_set<std::string> seen;

    std::ostringstream key;
    key << std::hex << code << "@" << address;
    const std::string keyStr = key.str();

    std::lock_guard<std::mutex> guard(seenLock);
    auto [it, inserted] = seen.insert(keyStr);
    return inserted;
}

bool hasArg(int argc, char* argv[], std::string_view arg) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] && std::string_view(argv[i]) == arg) {
            return true;
        }
    }
    return false;
}
}

extern std::unique_ptr<Backend::IBackend> g_backend;

namespace {
Backend::ViewportMode viewportModeFromShadingMode(int shadingMode) {
    switch (shadingMode) {
        case 0: return Backend::ViewportMode::Solid;
        case 1: return Backend::ViewportMode::MaterialPreview;
        case 3: return Backend::ViewportMode::Matcap;
        case 2:
        default:
            return Backend::ViewportMode::Rendered;
    }
}

bool isInteractiveViewportShadingMode(int shadingMode) {
    return shadingMode != 2;
}

bool isVulkanInteractiveViewportActive(bool viewportBackendActive, int shadingMode) {
    // Interactive viewport shading modes use a dedicated viewport backend that is
    // intentionally separate from the selected render device for Rendered mode.
    return viewportBackendActive && isInteractiveViewportShadingMode(shadingMode);
}

bool isActiveRenderBackendVulkan() {
    return dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr;
}

bool isActiveRenderBackendOptix() {
    return dynamic_cast<Backend::OptixBackend*>(g_backend.get()) != nullptr;
}

bool isActiveRenderBackendGpu() {
    return isActiveRenderBackendVulkan() || isActiveRenderBackendOptix();
}

void prepareCpuPickingState(SceneData& scene, SceneUI& ui) {
    const size_t foliage_count = InstanceManager::getInstance().getTotalInstanceCount();
    const size_t selectable_count = (foliage_count <= scene.world.objects.size())
        ? (scene.world.objects.size() - foliage_count)
        : scene.world.objects.size();

    for (size_t obj_index = 0; obj_index < selectable_count; ++obj_index) {
        auto& obj = scene.world.objects[obj_index];
        if (!obj) continue;

        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            tri->updateTransformedVertices();
            continue;
        }

        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            inst->syncTransformFromSourceTriangles();

            if (inst->source_triangles) {
                for (auto& srcTri : *inst->source_triangles) {
                    if (srcTri) {
                        srcTri->updateTransformedVertices();
                    }
                }
            }
        }
    }

    ui.rebuildMeshCache(scene.world.objects);
    ui.picking_vertices_synced = true;
}

#ifdef _WIN32
bool canLoadRuntimeDll(const char* dllName) {
    if (!dllName || !dllName[0]) return false;

    HMODULE h = LoadLibraryExA(dllName, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!h) {
        h = LoadLibraryExA(dllName, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }
    if (!h) {
        h = LoadLibraryA(dllName);
    }
    if (!h) {
        return false;
    }

    FreeLibrary(h);
    return true;
}

bool g_vcRuntimeMissing = false;
std::string g_vcRuntimeMissingList;

void checkVcRuntimeRedistributable() {
    static const char* kRequiredDlls[] = {
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll"
    };

    std::vector<std::string> missing;
    for (const char* dll : kRequiredDlls) {
        if (!canLoadRuntimeDll(dll)) {
            missing.emplace_back(dll);
        }
    }

    if (!missing.empty()) {
        g_vcRuntimeMissing = true;
        g_vcRuntimeMissingList.clear();
        for (size_t i = 0; i < missing.size(); ++i) {
            if (i > 0) g_vcRuntimeMissingList += ", ";
            g_vcRuntimeMissingList += missing[i];
        }
    }
}

void notifyIfVcRuntimeMissing() {
    if (!g_vcRuntimeMissing) return;

    const std::string warn =
        "Visual C++ Redistributable may be missing or corrupted. Missing DLL(s): " +
        g_vcRuntimeMissingList +
        ". Please install Microsoft Visual C++ 2015-2022 Redistributable (x64).";

    SCENE_LOG_WARN(warn);

    std::string msg =
        "Visual C++ Redistributable may be missing or corrupted.\n\n"
        "Missing DLL(s): " + g_vcRuntimeMissingList +
        "\n\nInstall Microsoft Visual C++ 2015-2022 Redistributable (x64).";

    MessageBoxA(nullptr, msg.c_str(), "RayTrophi - Dependency Warning", MB_OK | MB_ICONWARNING | MB_TOPMOST);
}
#endif

std::string startupCrashLogPath() {
    char exePath[MAX_PATH] = {};
    DWORD len = GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        try {
            std::filesystem::path p(exePath);
            return (p.parent_path() / "StartupCrash.log").string();
        } catch (...) {
        }
    }
    return "StartupCrash.log";
}

void removeStartupCrashLogIfExists() {
    try {
        std::string p = startupCrashLogPath();
        if (!p.empty() && std::filesystem::exists(p)) {
            std::error_code ec;
            std::filesystem::remove(p, ec);
            if (!ec) {
                SCENE_LOG_INFO("StartupCrash.log removed on clean shutdown.");
            } else {
                SCENE_LOG_WARN(std::string("Failed to remove StartupCrash.log: ") + ec.message());
            }
        }
    } catch (...) {
        // best-effort, never throw from cleanup
    }
}

void emergencyStartupLog(const std::string& message) {
    try {
        std::ofstream file(startupCrashLogPath(), std::ios::out | std::ios::app);
        if (file.is_open()) {
            SYSTEMTIME st{};
            GetLocalTime(&st);
            file << "["
                 << st.wYear << "-"
                 << st.wMonth << "-"
                 << st.wDay << " "
                 << st.wHour << ":"
                 << st.wMinute << ":"
                 << st.wSecond
                 << "] " << message << std::endl;
            file.flush();
        }
    } catch (...) {
    }
}

void startupDiagLog(const std::string& message) {
    if (g_startupDiagVerbose) {
        emergencyStartupLog(message);
    }
}

LONG WINAPI topLevelExceptionHandler(EXCEPTION_POINTERS* exceptionInfo) {
    DWORD code = exceptionInfo && exceptionInfo->ExceptionRecord
        ? exceptionInfo->ExceptionRecord->ExceptionCode
        : 0;
    const void* address = (exceptionInfo && exceptionInfo->ExceptionRecord)
        ? exceptionInfo->ExceptionRecord->ExceptionAddress
        : nullptr;

    std::ostringstream ss;
    ss << "[" << exceptionSeverityTag(code) << "] Unhandled SEH exception at startup. code=0x"
       << std::hex << code << " address=0x" << address;
    emergencyStartupLog(ss.str());
    return EXCEPTION_EXECUTE_HANDLER;
}

LONG CALLBACK vectoredExceptionHandler(PEXCEPTION_POINTERS exceptionInfo) {
    if (!exceptionInfo || !exceptionInfo->ExceptionRecord) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    const auto* rec = exceptionInfo->ExceptionRecord;

    // Ignore common non-fatal debug/status exceptions.
    if (isBenignStatusException(rec->ExceptionCode)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // Most C++ exceptions are first-chance and eventually handled; only log them in verbose mode.
    if (rec->ExceptionCode == kMsCppException && !g_startupDiagVerbose) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // Re-entrancy guard: if logging itself triggers another exception, do not recurse.
    static thread_local bool handlingException = false;
    if (handlingException) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    handlingException = true;

    if (!shouldLogExceptionOnce(rec->ExceptionCode, rec->ExceptionAddress)) {
        handlingException = false;
        return EXCEPTION_CONTINUE_SEARCH;
    }

    std::ostringstream ss;
    ss << "[" << exceptionSeverityTag(rec->ExceptionCode) << "] Vectored exception. code=0x"
       << std::hex << rec->ExceptionCode
       << " address=0x" << rec->ExceptionAddress;

    if (rec->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && rec->NumberParameters >= 2) {
        const auto accessType = static_cast<unsigned long long>(rec->ExceptionInformation[0]);
        const auto faultAddr = static_cast<unsigned long long>(rec->ExceptionInformation[1]);
        ss << " av_type=" << std::dec << accessType << " fault_addr=0x" << std::hex << faultAddr;
    }

    emergencyStartupLog(ss.str());
    handlingException = false;
    return EXCEPTION_CONTINUE_SEARCH;
}

[[noreturn]] void terminateHandler() {
    try {
        auto current = std::current_exception();
        if (current) {
            try {
                std::rethrow_exception(current);
            } catch (const std::exception& e) {
                emergencyStartupLog(std::string("[FATAL] std::terminate: ") + e.what());
            } catch (...) {
                emergencyStartupLog("[FATAL] std::terminate: unknown exception");
            }
        } else {
            emergencyStartupLog("[FATAL] std::terminate called without active exception");
        }
    } catch (...) {
        emergencyStartupLog("[FATAL] std::terminate logging failed");
    }
    std::_Exit(EXIT_FAILURE);
}

void signalHandler(int sig) {
    emergencyStartupLog("[FATAL] Signal received: " + std::to_string(sig));
    std::_Exit(EXIT_FAILURE);
}

void installEarlyCrashHandlers() {
    AddVectoredExceptionHandler(1, vectoredExceptionHandler);
    SetUnhandledExceptionFilter(topLevelExceptionHandler);
    std::set_terminate(terminateHandler);
    std::signal(SIGABRT, signalHandler);
    std::signal(SIGSEGV, signalHandler);
    std::signal(SIGFPE, signalHandler);
    std::signal(SIGILL, signalHandler);
    std::signal(SIGTERM, signalHandler);
}
}

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
// Request a display rebuild + stylize re-apply WITHOUT re-rendering and WITHOUT forcing
// tonemap on — honors the current persistent_tonemap setting. Set by Stylize param changes.
bool stylize_redisplay = false;
bool mouse_control_enabled = true;
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
float current_nav_dist = 10.0f; // NEW: Dynamic distance for sensitivity scaling
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
bool g_original_surface_needs_sync = true;

// Granular scene-sync dirty flags
bool g_geometry_dirty = true;        // Mesh/triangle data changed
bool g_materials_dirty = true;       // Material properties changed
bool g_gas_volumes_dirty = true;     // Gas/VDB volumes changed
bool g_texture_pool_dirty = false;   // Evict SceneTextureManager pool inside uploadMaterials (after waitIdle)

// Scene geometry generation counter
std::atomic<uint64_t> g_scene_geometry_generation{1};

// ===========================================================================
// DEFERRED REBUILD FLAGS - For optimized batched rebuilds
// Set these instead of calling rebuild immediately, Main loop handles them
// ===========================================================================
bool g_bvh_rebuild_pending = false;      // CPU BVH needs rebuild
bool g_gpu_refit_pending = false;        // GPU Geometry needs update (Deferred)
bool g_vulkan_rebuild_pending = false;    // GPU Vulkan geometry needs rebuild
bool g_vulkan_geometry_append_pending = false; // Additive-only mutation (scatter etc.) — try incremental TLAS refit first
bool g_vulkan_geometry_deform_pending = false; // Physics body deformed its mesh — refit only those BLAS in place (vs full rebuild)
bool g_viewport_raster_rebuild_pending = false; // Interactive raster viewport needs rebuild
bool g_optix_rebuild_pending = false;
bool g_particle_cpu_geometry_dirty = false; // Particle bridge changed; CPU-active path re-expands particles into the BVH

// ── Viewport-driven sequence save ───────────────────────────────────────────
// Renders an animation to disk through the SAME fast interactive viewport path
// (scrub each frame → converge to the interactive quality → save) instead of the
// separate render_Animation worker, which ran slower than and diverged from the
// viewport (foam/TLAS/rebuild differences). Quality + denoiser come from the
// interactive render-panel settings. Driven by a state machine in the main loop.
bool        g_seq_save_active = false;
int         g_seq_save_frame = 0;
int         g_seq_save_end = 0;
std::string g_seq_save_dir;
bool        g_seq_save_denoise = false;

// True while the active viewport is an interactive shading mode (Solid/Matcap),
// i.e. NOT Rendered. The fluid particle bridge reads this to render a cheap splat-
// sphere proxy in Solid mode even when the fluid's render_mode is SurfaceSDF (the
// raster viewport cannot draw the NanoVDB surface). Updated once per frame.
bool g_solid_viewport_active = false;
bool g_sim_timeline_mode = true;  // true = timeline-driven (bake/scrub, idle when stopped); false = live free-run preview
bool g_sim_use_gpu_solver = false; // experimental GPU simulation compute (CUDA today): grid solver + APIC fluid + NanoVDB bridge
RayTrophiSim::SimulationComputeVulkanContext g_vulkan_sim_compute_ctx{};
std::atomic<bool> g_optix_rebuild_in_progress{false}; // True while TLAS rebuild is happening // GPU OptiX geometry needs rebuild
std::atomic<bool> g_viewport_rebuild_in_progress{false}; // True while viewport backend resources are being torn down/rebuilt
std::atomic<bool> g_anim_backend_needs_full_rebuild{false}; // See globals.h — set by aborted-anim cleanup and backend-swap paths
bool g_mesh_cache_dirty = false;         // UI mesh cache needs rebuild
bool g_cpu_sync_pending = false;         // CPU data needs sync after TLAS mode changes
bool g_cpu_bvh_refit_pending = false;    // CPU BVH fast refit (Embree only)
int g_bvh_rebuild_deferred_frames = 0;
uint64_t g_cpu_bvh_requested_generation = 0;
uint64_t g_cpu_bvh_future_generation = 0;
bool g_viewport_hovered = false;         // True when mouse is over the main RenderView
int g_backend_switch_cooldown_frames = 0; // Skip a few GPU animation/render ticks right after backend switch

// Helper to ensure original_surface matches main surface dimensions
// This prevents tonemap accumulation issues by giving us a clean "Raw" buffer
void EnsureOriginalSurface(SDL_Surface* reference) {
    if (!reference) return;
    extern SDL_Surface* original_surface;
    
    if (!original_surface || original_surface->w != reference->w || original_surface->h != reference->h) {
        if (original_surface) SDL_FreeSurface(original_surface);
        original_surface = SDL_CreateRGBSurfaceWithFormat(0, reference->w, reference->h, 32, reference->format->format);
        // [FIX] Initialize pixels to black — prevents random black/white/garbage
        // when VulkanRT pipeline isn't ready yet and renderProgressiveImpl early-returns
        if (original_surface) SDL_FillRect(original_surface, nullptr, SDL_MapRGBA(original_surface->format, 0, 0, 0, 255));
    }
}

// ===========================================================================
// SCENE LOADING FLAGS - Thread safety for project load/save operations
// ===========================================================================
std::atomic<bool> g_scene_loading_in_progress{false};  // Prevents concurrent load operations
std::atomic<bool> g_needs_geometry_rebuild{false};   // Set by loader thread, main loop does actual rebuild
std::atomic<bool> g_needs_optix_sync{false};         // Set by loader thread, main loop syncs backend buffers
std::atomic<bool> g_deferred_render_backend_prepare_pending{false};
int g_deferred_render_backend_prepare_delay_frames = 0;

// Async OptiX rebuild handle/state. Kept at file scope so backend switch can safely wait.
std::future<void> g_optix_future;
bool g_optix_rebuilding = false;


Vec3 applyVignette(const Vec3& color, int x, int y, int width, int height, float strength = 1.0f) {
    float u = (x / (float)width - 0.5f) * 2.0f;
    float v = (y / (float)height - 0.5f) * 2.0f;
    float dist = u * u + v * v;
    float falloff = std::clamp(1.0f - strength * dist, 0.0f, 1.0f);
    return color * falloff;
}

bool hasNoOpColorProcessing(const ColorProcessor& processor) {
    const auto nearlyEqual = [](float a, float b, float eps = 1e-5f) {
        return std::fabs(a - b) <= eps;
    };

    const auto& params = processor.params;
    const bool vignette_noop = !params.enable_vignette || nearlyEqual(params.vignette_strength, 0.0f);
    return nearlyEqual(params.global_exposure, 1.0f) &&
           nearlyEqual(params.global_gamma, 1.0f) &&
           nearlyEqual(params.saturation, 1.0f) &&
           nearlyEqual(params.color_temperature, 6500.0f) &&
           params.tone_mapping_type == ToneMappingType::None &&
           vignette_noop;
}

void copySurfacePixelsOrBlit(SDL_Surface* dst, SDL_Surface* src) {
    if (!dst || !src || !dst->pixels || !src->pixels) return;

    const bool canMemcpy =
        dst->w == src->w &&
        dst->h == src->h &&
        dst->format &&
        src->format &&
        dst->format->format == src->format->format &&
        dst->pitch == src->pitch;

    if (canMemcpy) {
        std::memcpy(dst->pixels, src->pixels, (size_t)src->pitch * (size_t)src->h);
        return;
    }

    SDL_BlitSurface(src, nullptr, dst, nullptr);
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
std::mutex surface_mutex;  // Surface eri�imi i�in mutex
SceneUI ui;
SceneData scene;
Renderer ray_renderer(image_width, image_height, 1, 1);
std::unique_ptr<Backend::IBackend> g_backend;
std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
ColorProcessor color_processor(image_width, image_height);

// Sync material data to the raster viewport backend. Walks Texture::vulkan_dirty
// so paint strokes refresh the material preview descriptor in-place. When the
// render device is Vulkan RT there is no dedicated g_viewport_backend — the same
// VulkanBackendAdapter serves both RT render and raster viewport, so fall back
// to g_backend in that case. Without this fallback, material preview mode on
// Vulkan RT never tazelenmez and paint strokes appear frozen.
static void syncMaterialBufferToViewportBackend(SceneData& scene, Renderer& renderer) {
    if (g_viewport_backend) {
        renderer.updateBackendMaterials(scene, g_viewport_backend.get());
        return;
    }
    if (auto* vkAdapter = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
        renderer.updateBackendMaterials(scene, vkAdapter);
    }
}

static void shutdownAndResetBackendSafe(const char* reason) {
    if (!g_backend) return;
    try {
        scene.simulation_world.compute().synchronize();
    } catch (const std::exception& e) {
        SCENE_LOG_WARN(std::string("[Backend] simulation compute synchronize failed during reset (") +
                       (reason ? reason : "unknown") + "): " + e.what());
    } catch (...) {
        SCENE_LOG_WARN(std::string("[Backend] simulation compute synchronize failed during reset (") +
                       (reason ? reason : "unknown") + ").");
    }

    try {
        g_backend->waitForCompletion();
    } catch (const std::exception& e) {
        SCENE_LOG_WARN(std::string("[Backend] waitForCompletion failed during reset (") +
                       (reason ? reason : "unknown") + "): " + e.what());
    } catch (...) {
        SCENE_LOG_WARN(std::string("[Backend] waitForCompletion failed during reset (") +
                       (reason ? reason : "unknown") + ").");
    }

    try {
        g_backend->shutdown();
    } catch (const std::exception& e) {
        SCENE_LOG_WARN(std::string("[Backend] shutdown failed during reset (") +
                       (reason ? reason : "unknown") + "): " + e.what());
    } catch (...) {
        SCENE_LOG_WARN(std::string("[Backend] shutdown failed during reset (") +
                       (reason ? reason : "unknown") + ").");
    }

    g_backend.reset();

    // Release all VDB CUDA memory immediately when tearing down the backend.
    // This prevents VRAM double allocation leaks when switching to Vulkan or CPU rendering.
    if (g_hasCUDA) {
        VDBVolumeManager::getInstance().freeAllGPU();
    }

#ifdef _WIN32
    // Heavy Vulkan scene rebuild/switch patterns can leave large freed pages in
    // process working set and CRT heap caches. Trim once after backend teardown
    // so repeated Vulkan<->OptiX switches do not show runaway RAM growth.
    (void)_heapmin();
    HANDLE hProc = GetCurrentProcess();
    if (!SetProcessWorkingSetSize(hProc, (SIZE_T)-1, (SIZE_T)-1)) {
        SCENE_LOG_WARN("[Backend] SetProcessWorkingSetSize trim request failed.");
    }
#endif
}

static bool initializeViewportBackendIfAvailable() {
    if (g_viewport_backend) return true;
    if (!g_hasVulkan) return false;
    try {
        auto backend = std::make_unique<Backend::VulkanViewportBackend>();
        if (!backend->initialize()) {
            SCENE_LOG_WARN("Viewport Vulkan backend initialization returned false.");
            return false;
        }
        g_viewport_backend = std::move(backend);
        return true;
    } catch (const std::exception& e) {
        SCENE_LOG_WARN(std::string("Viewport Vulkan initialization failed: ") + e.what());
        return false;
    } catch (...) {
        SCENE_LOG_WARN("Viewport Vulkan initialization failed with unknown exception.");
        return false;
    }
}

static Backend::IViewportBackend* getRasterViewportBackend() {
    if (g_viewport_backend) {
        return g_viewport_backend.get();
    }
    if (auto* viewportBackend = dynamic_cast<Backend::IViewportBackend*>(g_backend.get())) {
        return viewportBackend;
    }
    return nullptr;
}

static Backend::IBackend* getActiveViewportBackendForShading(int shadingMode) {
    if (isInteractiveViewportShadingMode(shadingMode)) {
        if (auto* vk = getRasterViewportBackend()) return vk;
    }
    return g_backend.get();
}

static void applyPendingDeleteVisibilityToBackend(SceneData& scene, Backend::IBackend* backend) {
    if (!backend || scene.editor_pending_delete_object_names.empty()) {
        return;
    }

    for (const auto& nodeName : scene.editor_pending_delete_object_names) {
        if (!nodeName.empty()) {
            backend->setVisibilityByNodeName(nodeName, false);
        }
    }
}

static std::vector<std::shared_ptr<Hittable>> collectVisibleSceneObjects(SceneData& scene) {
    std::vector<std::shared_ptr<Hittable>> visibleObjects;
    visibleObjects.reserve(scene.world.objects.size());

    for (const auto& obj : scene.world.objects) {
        if (!obj || !obj->visible) {
            continue;
        }
        visibleObjects.push_back(obj);
    }

    return visibleObjects;
}

std::string active_model_path;
SceneSelection scene_selection;  // Scene selection manager
UIContext ui_ctx{
   scene,
   ray_renderer,
    nullptr, // optix_gpu_ptr will be set later during initialization
   nullptr, // backend_ptr (assigned later)
   color_processor,
   render_settings,
   scene_selection,  // Add selection reference
   sample_count,
   start_render,
   active_model_path,
   apply_tonemap,
   reset_tonemap,
   mouse_control_enabled
};
void applyToneMappingToSurfaceWithCamera(SDL_Surface* surface, SDL_Surface* original, ColorProcessor& processor, Renderer* renderer, const Camera* camera) {
    if (!surface || !surface->pixels) return;
    Uint32* pixels = (Uint32*)surface->pixels;
    int width = surface->w;
    int height = surface->h;
    SDL_PixelFormat* fmt = surface->format;

    const bool use_float_buffer = (renderer != nullptr) &&
                                  renderer->cpu_accumulation_valid &&
                                  (renderer->cpu_accumulation_buffer.size() == (size_t)(width * height));

    Uint32* src = (original && original->pixels) ? (Uint32*)original->pixels : nullptr;
    if (!use_float_buffer && !src) return;

    // Capture format masks/shifts once — avoid SDL_MapRGB/SDL_GetRGB per-pixel dispatch.
    const Uint32 rMask = fmt->Rmask, gMask = fmt->Gmask, bMask = fmt->Bmask, aMask = fmt->Amask;
    const Uint8  rShift = fmt->Rshift, gShift = fmt->Gshift, bShift = fmt->Bshift;
    const float  inv255 = 1.0f / 255.0f;

    // Precompute linear→sRGB→uint8 LUT so the per-pixel hot path avoids 3× std::pow.
    // Only needed for use_float_buffer branch; non-float branch consumes pre-sRGB pixels.
    constexpr int LUT_SIZE = 4096;
    constexpr float LUT_MAX = float(LUT_SIZE - 1);
    alignas(64) uint8_t srgbLut[LUT_SIZE];
    if (use_float_buffer) {
        for (int i = 0; i < LUT_SIZE; ++i) {
            float x = float(i) / LUT_MAX;
            float s = (x <= 0.0031308f) ? 12.92f * x
                                        : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
            if (s < 0.0f) s = 0.0f;
            if (s > 1.0f) s = 1.0f;
            srgbLut[i] = static_cast<uint8_t>(s * 255.0f + 0.5f);
        }
    }

    const bool reinhard_none = (processor.params.tone_mapping_type == ToneMappingType::None);
    const bool use_denoised  = use_float_buffer && renderer->hasCPUDenoisedBuffer();
    const bool vignette_on   = processor.params.enable_vignette;
    const float vignette_strength = processor.params.vignette_strength;
    const Stylize::StylizeModeState* stylize_state = renderer ? &renderer->stylizeMode : nullptr;
    const int stylize_frame = renderer ? renderer->world.getGPUData().frame_count : 0;
    const WorldData stylize_world = renderer ? renderer->world.getGPUData() : WorldData{};
    const bool use_cpu_stylize_aov =
        stylize_state &&
        stylize_state->enabled &&
        use_float_buffer &&
        renderer->cpu_albedo_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer->cpu_normal_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer->cpu_world_position_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer->cpu_depth_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer->cpu_material_id_buffer.size() == static_cast<size_t>(width * height);

    auto makeStylizeAOV = [=](int sx, int sy) -> Stylize::StylizeAOVSample {
        Stylize::StylizeAOVSample aov;
        if (!use_cpu_stylize_aov ||
            sx < 0 || sy < 0 || sx >= width || sy >= height) {
            return aov;
        }
        aov.valid = true;
        aov.screen_u = (static_cast<float>(sx) + 0.5f) / std::max(1.0f, static_cast<float>(width));
        aov.screen_v = (static_cast<float>(sy) + 0.5f) / std::max(1.0f, static_cast<float>(height));
        aov.sun_dir = Vec3(stylize_world.nishita.sun_direction.x, stylize_world.nishita.sun_direction.y, stylize_world.nishita.sun_direction.z);
        if (aov.sun_dir.length_squared() <= 1e-8f) {
            aov.sun_dir = Vec3(0.32f, 0.82f, 0.46f);
        } else {
            aov.sun_dir = aov.sun_dir.normalize();
        }
        aov.sun_size_degrees = std::max(0.01f, stylize_world.nishita.sun_size);
        aov.sun_elevation_degrees = stylize_world.nishita.sun_elevation;
        aov.nishita_clouds_enabled = stylize_world.nishita.clouds_enabled != 0;
        aov.nishita_cloud_coverage = std::clamp(stylize_world.nishita.cloud_coverage, 0.0f, 1.0f);
        aov.nishita_cloud_density = std::max(0.0f, stylize_world.nishita.cloud_density);
        aov.nishita_cloud_scale = std::max(0.05f, stylize_world.nishita.cloud_scale);
        aov.nishita_cloud_offset_x = stylize_world.nishita.cloud_offset_x;
        aov.nishita_cloud_offset_z = stylize_world.nishita.cloud_offset_z;
        aov.nishita_cloud_seed = stylize_world.nishita.cloud_seed;
        if (camera) {
            Vec3 view_dir = camera->lower_left_corner
                + aov.screen_u * camera->horizontal
                + aov.screen_v * camera->vertical
                - camera->origin;
            aov.view_dir = view_dir.length_squared() > 1e-8f ? view_dir.normalize() : Vec3(0.0f, 0.0f, -1.0f);
        }
        const size_t idx = static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx);
        const Renderer::Vec4& albedo = renderer->cpu_albedo_accumulation_buffer[idx];
        const Renderer::Vec4& normal = renderer->cpu_normal_accumulation_buffer[idx];
        const Renderer::Vec4& world_position = renderer->cpu_world_position_accumulation_buffer[idx];
        aov.hit = albedo.w > 0.0f && renderer->cpu_depth_accumulation_buffer[idx] > 0.0f;
        aov.albedo = Vec3(albedo.x, albedo.y, albedo.z);
        aov.normal = Vec3(normal.x, normal.y, normal.z);
        aov.world_position = Vec3(world_position.x, world_position.y, world_position.z);
        aov.depth = renderer->cpu_depth_accumulation_buffer[idx];
        aov.material_id = renderer->cpu_material_id_buffer[idx];
        return aov;
    };

    auto makeStylizeAOVWithEdges = [=](int sx, int sy) -> Stylize::StylizeAOVSample {
        Stylize::StylizeAOVSample aov = makeStylizeAOV(sx, sy);
        if (!aov.hit) {
            return aov;
        }
        const Stylize::StylizeAOVSample right = makeStylizeAOV(sx + 1, sy);
        const Stylize::StylizeAOVSample down = makeStylizeAOV(sx, sy + 1);
        float edge = 0.0f;
        auto accumulateEdge = [&](const Stylize::StylizeAOVSample& n) {
            if (!n.hit) {
                edge += 1.0f;
                return;
            }
            const float depth_scale = std::max(0.025f, aov.depth * 0.015f);
            edge += std::min(1.0f, std::abs(aov.depth - n.depth) / depth_scale);
            edge += std::min(1.0f, (aov.normal - n.normal).length() * 0.75f);
            if (aov.material_id != n.material_id) {
                edge += 0.45f;
            }
        };
        accumulateEdge(right);
        accumulateEdge(down);
        aov.edge = std::clamp(edge * 0.55f, 0.0f, 1.0f);
        return aov;
    };

    const float* denoised_ptr = use_denoised ? renderer->cpu_denoised_buffer.data() : nullptr;
    const Renderer::Vec4* accum_ptr = (use_float_buffer && !use_denoised)
                                      ? renderer->cpu_accumulation_buffer.data() : nullptr;

    // std::for_each_n over row indices → inner loop vectorizable; avoids std::async thread spawn per call.
    std::vector<int> rowIndices(height);
    std::iota(rowIndices.begin(), rowIndices.end(), 0);

    std::for_each_n(std::execution::par_unseq, rowIndices.data(), (size_t)height,
        [=, &processor](int j) {
            const int buffer_y = height - 1 - j;
            Uint32* __restrict rowDst = pixels + (size_t)j * (size_t)width;
            const Uint32* __restrict rowSrc = src ? (src + (size_t)j * (size_t)width) : nullptr;

            for (int i = 0; i < width; ++i) {
                Vec3 raw_color;

                if (use_float_buffer) {
                    if (use_denoised) {
                        const size_t idx = ((size_t)j * (size_t)width + (size_t)i) * 3;
                        raw_color = Vec3(denoised_ptr[idx], denoised_ptr[idx + 1], denoised_ptr[idx + 2]);
                    } else {
                        const Renderer::Vec4& p = accum_ptr[(size_t)buffer_y * (size_t)width + (size_t)i];
                        raw_color = Vec3(p.x, p.y, p.z);
                    }
                    if (reinhard_none) {
                        raw_color.x = raw_color.x / (raw_color.x + 1.0f);
                        raw_color.y = raw_color.y / (raw_color.y + 1.0f);
                        raw_color.z = raw_color.z / (raw_color.z + 1.0f);
                    }
                } else {
                    Uint32 px = rowSrc[i];
                    float r = float((px & rMask) >> rShift) * inv255;
                    float g = float((px & gMask) >> gShift) * inv255;
                    float b = float((px & bMask) >> bShift) * inv255;
                    raw_color = Vec3(r, g, b);
                }

                Vec3 final_color = processor.processColor(raw_color, i, j);

                if (vignette_on)
                    final_color = applyVignette(final_color, i, j, width, height, vignette_strength);

                if (stylize_state && stylize_state->enabled) {
                    if (use_cpu_stylize_aov) {
                        final_color = Stylize::applyPostProcess(
                            final_color,
                            makeStylizeAOVWithEdges(i, buffer_y),
                            i, j, stylize_frame, *stylize_state);
                    } else {
                        final_color = Stylize::applyPostProcess(final_color, i, j, stylize_frame, *stylize_state);
                    }
                }

                Uint8 ri, gi, bi;
                if (use_float_buffer) {
                    float fx = final_color.x; if (fx < 0.0f) fx = 0.0f; else if (fx > 1.0f) fx = 1.0f;
                    float fy = final_color.y; if (fy < 0.0f) fy = 0.0f; else if (fy > 1.0f) fy = 1.0f;
                    float fz = final_color.z; if (fz < 0.0f) fz = 0.0f; else if (fz > 1.0f) fz = 1.0f;
                    ri = srgbLut[int(fx * LUT_MAX)];
                    gi = srgbLut[int(fy * LUT_MAX)];
                    bi = srgbLut[int(fz * LUT_MAX)];
                } else {
                    float fx = final_color.x; if (fx < 0.0f) fx = 0.0f; else if (fx > 1.0f) fx = 1.0f;
                    float fy = final_color.y; if (fy < 0.0f) fy = 0.0f; else if (fy > 1.0f) fy = 1.0f;
                    float fz = final_color.z; if (fz < 0.0f) fz = 0.0f; else if (fz > 1.0f) fz = 1.0f;
                    ri = uint8_t(fx * 255.0f);
                    gi = uint8_t(fy * 255.0f);
                    bi = uint8_t(fz * 255.0f);
                }

                Uint32 alpha = rowSrc ? (rowSrc[i] & aMask) : aMask;
                rowDst[i] = alpha
                          | ((Uint32)ri << rShift)
                          | ((Uint32)gi << gShift)
                          | ((Uint32)bi << bShift);
            }
        });
}

void applyToneMappingToSurface(SDL_Surface* surface, SDL_Surface* original, ColorProcessor& processor, Renderer* renderer) {
    applyToneMappingToSurfaceWithCamera(surface, original, processor, renderer, nullptr);
}

void applyStylizeToSurfaceWithCamera(SDL_Surface* surface, Renderer& renderer, bool use_cpu_aov, const Camera* camera) {
    if (!surface || !surface->pixels || !renderer.stylizeMode.enabled) return;

    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    SDL_PixelFormat* fmt = surface->format;
    const int width = surface->w;
    const int height = surface->h;

    const Uint32 rMask = fmt->Rmask, gMask = fmt->Gmask, bMask = fmt->Bmask, aMask = fmt->Amask;
    const Uint8 rShift = fmt->Rshift, gShift = fmt->Gshift, bShift = fmt->Bshift;
    const float inv255 = 1.0f / 255.0f;
    const int stylize_frame = renderer.world.getGPUData().frame_count;
    const WorldData stylize_world = renderer.world.getGPUData();
    const bool use_cpu_stylize_aov =
        use_cpu_aov &&
        renderer.cpu_albedo_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer.cpu_normal_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer.cpu_world_position_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer.cpu_depth_accumulation_buffer.size() == static_cast<size_t>(width * height) &&
        renderer.cpu_material_id_buffer.size() == static_cast<size_t>(width * height);

    auto makeStylizeAOV = [&](int sx, int sy) -> Stylize::StylizeAOVSample {
        Stylize::StylizeAOVSample aov;
        if (!use_cpu_stylize_aov ||
            sx < 0 || sy < 0 || sx >= width || sy >= height) {
            return aov;
        }
        aov.valid = true;
        aov.screen_u = (static_cast<float>(sx) + 0.5f) / std::max(1.0f, static_cast<float>(width));
        aov.screen_v = (static_cast<float>(sy) + 0.5f) / std::max(1.0f, static_cast<float>(height));
        aov.sun_dir = Vec3(stylize_world.nishita.sun_direction.x, stylize_world.nishita.sun_direction.y, stylize_world.nishita.sun_direction.z);
        if (aov.sun_dir.length_squared() <= 1e-8f) {
            aov.sun_dir = Vec3(0.32f, 0.82f, 0.46f);
        } else {
            aov.sun_dir = aov.sun_dir.normalize();
        }
        aov.sun_size_degrees = std::max(0.01f, stylize_world.nishita.sun_size);
        aov.sun_elevation_degrees = stylize_world.nishita.sun_elevation;
        aov.nishita_clouds_enabled = stylize_world.nishita.clouds_enabled != 0;
        aov.nishita_cloud_coverage = std::clamp(stylize_world.nishita.cloud_coverage, 0.0f, 1.0f);
        aov.nishita_cloud_density = std::max(0.0f, stylize_world.nishita.cloud_density);
        aov.nishita_cloud_scale = std::max(0.05f, stylize_world.nishita.cloud_scale);
        aov.nishita_cloud_offset_x = stylize_world.nishita.cloud_offset_x;
        aov.nishita_cloud_offset_z = stylize_world.nishita.cloud_offset_z;
        aov.nishita_cloud_seed = stylize_world.nishita.cloud_seed;
        if (camera) {
            Vec3 view_dir = camera->lower_left_corner
                + aov.screen_u * camera->horizontal
                + aov.screen_v * camera->vertical
                - camera->origin;
            aov.view_dir = view_dir.length_squared() > 1e-8f ? view_dir.normalize() : Vec3(0.0f, 0.0f, -1.0f);
        }
        const size_t idx = static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx);
        const Renderer::Vec4& albedo = renderer.cpu_albedo_accumulation_buffer[idx];
        const Renderer::Vec4& normal = renderer.cpu_normal_accumulation_buffer[idx];
        const Renderer::Vec4& world_position = renderer.cpu_world_position_accumulation_buffer[idx];
        aov.hit = albedo.w > 0.0f && renderer.cpu_depth_accumulation_buffer[idx] > 0.0f;
        aov.albedo = Vec3(albedo.x, albedo.y, albedo.z);
        aov.normal = Vec3(normal.x, normal.y, normal.z);
        aov.world_position = Vec3(world_position.x, world_position.y, world_position.z);
        aov.depth = renderer.cpu_depth_accumulation_buffer[idx];
        aov.material_id = renderer.cpu_material_id_buffer[idx];
        return aov;
    };

    auto makeStylizeAOVWithEdges = [&](int sx, int sy) -> Stylize::StylizeAOVSample {
        Stylize::StylizeAOVSample aov = makeStylizeAOV(sx, sy);
        if (!aov.hit) return aov;
        const Stylize::StylizeAOVSample right = makeStylizeAOV(sx + 1, sy);
        const Stylize::StylizeAOVSample down = makeStylizeAOV(sx, sy + 1);
        float edge = 0.0f;
        auto accumulateEdge = [&](const Stylize::StylizeAOVSample& n) {
            if (!n.hit) {
                edge += 1.0f;
                return;
            }
            const float depth_scale = std::max(0.025f, aov.depth * 0.015f);
            edge += std::min(1.0f, std::abs(aov.depth - n.depth) / depth_scale);
            edge += std::min(1.0f, (aov.normal - n.normal).length() * 0.75f);
            if (aov.material_id != n.material_id) edge += 0.45f;
        };
        accumulateEdge(right);
        accumulateEdge(down);
        aov.edge = std::clamp(edge * 0.55f, 0.0f, 1.0f);
        return aov;
    };

    std::vector<int> rowIndices(height);
    std::iota(rowIndices.begin(), rowIndices.end(), 0);

    std::for_each_n(std::execution::par_unseq, rowIndices.data(), static_cast<size_t>(height),
        [=, &renderer](int y) {
            Uint32* row = pixels + static_cast<size_t>(y) * static_cast<size_t>(width);
            for (int x = 0; x < width; ++x) {
                const Uint32 px = row[x];
                Vec3 color(
                    static_cast<float>((px & rMask) >> rShift) * inv255,
                    static_cast<float>((px & gMask) >> gShift) * inv255,
                    static_cast<float>((px & bMask) >> bShift) * inv255
                );
                if (use_cpu_stylize_aov) {
                    const int buffer_y = height - 1 - y;
                    color = Stylize::applyPostProcess(
                        color,
                        makeStylizeAOVWithEdges(x, buffer_y),
                        x, y, stylize_frame, renderer.stylizeMode);
                } else {
                    color = Stylize::applyPostProcess(color, x, y, stylize_frame, renderer.stylizeMode);
                }

                const Uint8 ri = static_cast<Uint8>(std::clamp(color.x, 0.0f, 1.0f) * 255.0f);
                const Uint8 gi = static_cast<Uint8>(std::clamp(color.y, 0.0f, 1.0f) * 255.0f);
                const Uint8 bi = static_cast<Uint8>(std::clamp(color.z, 0.0f, 1.0f) * 255.0f);
                row[x] = (px & aMask)
                       | (static_cast<Uint32>(ri) << rShift)
                       | (static_cast<Uint32>(gi) << gShift)
                       | (static_cast<Uint32>(bi) << bShift);
            }
        });
}

void applyStylizeToSurface(SDL_Surface* surface, Renderer& renderer, bool use_cpu_aov) {
    applyStylizeToSurfaceWithCamera(surface, renderer, use_cpu_aov, nullptr);
}

void applyCPUDenoisedPreviewToSurface(SDL_Surface* surface, Renderer& renderer, const Camera* camera) {
    if (!surface || !surface->pixels || !renderer.hasCPUDenoisedBuffer()) return;

    Uint32* pixels = (Uint32*)surface->pixels;
    SDL_PixelFormat* fmt = surface->format;
    const int width = surface->w;
    const int height = surface->h;

    const Uint32 rMask = fmt->Rmask, gMask = fmt->Gmask, bMask = fmt->Bmask, aMask = fmt->Amask;
    const Uint8  rShift = fmt->Rshift, gShift = fmt->Gshift, bShift = fmt->Bshift;

    float exposure_factor = 1.0f;
    if (camera) {
        if (camera->auto_exposure) {
            exposure_factor = std::pow(2.0f, camera->ev_compensation);
        } else if (camera->use_physical_exposure) {
            float iso_mult = (camera->iso_preset_index >= 0 && camera->iso_preset_index < (int)CameraPresets::ISO_PRESET_COUNT) ?
                CameraPresets::ISO_PRESETS[camera->iso_preset_index].exposure_multiplier : 1.0f;
            float shutter_time = (camera->shutter_preset_index >= 0 && camera->shutter_preset_index < (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT) ?
                CameraPresets::SHUTTER_SPEED_PRESETS[camera->shutter_preset_index].speed_seconds : 0.004f;

            float f_number = 16.0f;
            if (camera->fstop_preset_index > 0 && camera->fstop_preset_index < (int)CameraPresets::FSTOP_PRESET_COUNT) {
                f_number = CameraPresets::FSTOP_PRESETS[camera->fstop_preset_index].f_number;
            } else if (camera->aperture > 0.001f) {
                f_number = 0.8f / camera->aperture;
            }

            float aperture_sq = f_number * f_number;
            float ev_comp = std::pow(2.0f, camera->ev_compensation);
            float current_val = (iso_mult * shutter_time) / (aperture_sq + 1e-6f);
            float baseline_val = 0.00003125f;
            exposure_factor = (current_val / baseline_val) * ev_comp * 2.0f;
        } else {
            exposure_factor = std::pow(2.0f, camera->ev_compensation);
        }
    }

    // LUT: Reinhard + linear→sRGB + *255 → uint8, indexed on clamp01(raw*exposure).
    // Eliminates 3× std::pow per pixel.
    constexpr int LUT_SIZE = 4096;
    constexpr float LUT_MAX = float(LUT_SIZE - 1);
    alignas(64) uint8_t outLut[LUT_SIZE];
    for (int i = 0; i < LUT_SIZE; ++i) {
        float x = float(i) / LUT_MAX;          // x = raw * exposure in [0,1] after clamp
        float r = x / (1.0f + x);              // Reinhard
        float s = (r <= 0.0031308f) ? 12.92f * r
                                    : 1.055f * std::pow(r, 1.0f / 2.4f) - 0.055f;
        if (s < 0.0f) s = 0.0f;
        if (s > 1.0f) s = 1.0f;
        outLut[i] = static_cast<uint8_t>(s * 255.0f + 0.5f);
    }

    const float* denoised = renderer.cpu_denoised_buffer.data();
    const float expF = exposure_factor;

    std::vector<int> rowIndices(height);
    std::iota(rowIndices.begin(), rowIndices.end(), 0);

    std::for_each_n(std::execution::par_unseq, rowIndices.data(), (size_t)height,
        [=](int j) {
            Uint32* __restrict rowDst = pixels + (size_t)j * (size_t)width;
            for (int i = 0; i < width; ++i) {
                const size_t idx = ((size_t)j * (size_t)width + (size_t)i) * 3;
                float r = denoised[idx]     * expF; if (r < 0.0f) r = 0.0f;
                float g = denoised[idx + 1] * expF; if (g < 0.0f) g = 0.0f;
                float b = denoised[idx + 2] * expF; if (b < 0.0f) b = 0.0f;

                // Clamp raw*exposure to [0,1] for LUT indexing.
                // Reinhard handles >1 natively, but LUT represents the post-Reinhard
                // compressed range; raw*exposure >1 just saturates to LUT_MAX.
                float xr = r / (1.0f + r);
                float xg = g / (1.0f + g);
                float xb = b / (1.0f + b);
                // xr/xg/xb already in [0,1) — but LUT keyed on pre-Reinhard x.
                // Cheaper: key LUT on clamp01(raw*exposure) directly; rebuild LUT.
                (void)xr; (void)xg; (void)xb;

                if (r > 1.0f) r = 1.0f;
                if (g > 1.0f) g = 1.0f;
                if (b > 1.0f) b = 1.0f;

                const uint8_t ri = outLut[int(r * LUT_MAX)];
                const uint8_t gi = outLut[int(g * LUT_MAX)];
                const uint8_t bi = outLut[int(b * LUT_MAX)];

                rowDst[i] = aMask
                          | ((Uint32)ri << rShift)
                          | ((Uint32)gi << gShift)
                          | ((Uint32)bi << bShift);
            }
        });
}
void reset_render_resolution(int w, int h)
{
    // ------------------------------------------------------------------
    // 1. SDL Pencere Boyutunu G�ncelle
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // 1. SDL Pencere Boyutunu G�ncelle - IPTAL (Kullan�c� pencere boyutu de�i�sin istemiyor)
    // ------------------------------------------------------------------
    // SDL_SetWindowSize(window, w, h);
    
    // ------------------------------------------------------------------
    // 2. SDL kaynaklar�n� s�f�rla (DESTROY)
    // ------------------------------------------------------------------
    if (raytrace_texture) SDL_DestroyTexture(raytrace_texture);
    if (surface) SDL_FreeSurface(surface);
    if (original_surface) SDL_FreeSurface(original_surface);

    // ------------------------------------------------------------------
    // 3. Yeni kaynaklar� olu�tur (CREATE)
    // ------------------------------------------------------------------
    surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    original_surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    raytrace_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING, w, h);

    if (!surface || !original_surface || !raytrace_texture) {
        SCENE_LOG_ERROR("Failed to create SDL surfaces or texture!");
        return;
    }

    // Siyah ekranla ba�la
    SDL_FillRect(surface, nullptr, SDL_MapRGBA(surface->format, 0, 0, 0, 255));
    SDL_FillRect(original_surface, nullptr, SDL_MapRGBA(original_surface->format, 0, 0, 0, 255));
    g_original_surface_needs_sync = true;
    
    // Texture'� g�ncelle
    SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
    
    // Ekran� g�ncelle
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);

    // ------------------------------------------------------------------
    // 4. Aspect ratio ve global de�i�kenleri g�ncelle
    // ------------------------------------------------------------------
    aspect_ratio = (float)w / h;
    
    // ------------------------------------------------------------------
    // 5. Di�er Bile�enleri G�ncelle
    // ------------------------------------------------------------------
    ray_renderer.resetResolution(w, h);
    // Always resize OptiX buffers if available
    if (g_backend) {
        Backend::RenderParams rp = {};
        rp.imageWidth = w;
        rp.imageHeight = h;
        rp.samplesPerPixel = render_settings.samples_per_pixel;
        rp.minSamples = render_settings.min_samples;
        rp.maxBounces = std::max(1, render_settings.max_bounces); // min 1: sıfır = primary ray only
        rp.diffuseBounces = std::clamp(render_settings.diffuse_bounces, 1, rp.maxBounces);
        rp.transmissionBounces = std::clamp(render_settings.transmission_bounces, 1, rp.maxBounces);
        rp.useAdaptiveSampling = render_settings.use_adaptive_sampling;
        rp.adaptiveThreshold = render_settings.variance_threshold;
        g_backend->setRenderParams(rp);
    }
    color_processor.resize(w, h);

    // CRITICAL: Ensure Camera aspect ratio matches new resolution for correct ray generation (AF, Picking)
    if (scene.camera) {
        scene.camera->aspect_ratio = aspect_ratio;
        scene.camera->update_camera_vectors();
        if (g_backend) {
            ray_renderer.syncCameraToBackend(*scene.camera);
        }
    }

    SCENE_LOG_INFO("Render resolution updated: " + std::to_string(w) + "x" + std::to_string(h));
}
// Check if GPU has RT Cores (hardware ray tracing)
static bool hasRTCores(int major, int minor)
{
    // RTX donan�m� SM 7.5 ile ba�lad�.
    if (major > 7) return true;            // SM 8.x, 9.x, 10.x � yeni RTX mimarileri
    if (major == 7 && minor >= 5) return true; // SM 7.5 � Turing
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

// ---------------------------------------------------------------------------
// Async OptiX backend init state (Blender-style non-blocking compile).
// When user switches to OptiX from the combo, we DON'T call initializeOptixIfAvailable
// synchronously on the UI thread (it blocks for 1-3 min on cold-cache first launches).
// Instead a worker thread builds the new OptiX backend while the current backend keeps
// rendering. Main thread polls the future each frame and hot-swaps when ready.
// ---------------------------------------------------------------------------
static std::atomic<bool> g_optix_async_in_progress{false};
static std::future<std::unique_ptr<Backend::IBackend>> g_optix_async_future;
static std::chrono::steady_clock::time_point g_optix_async_start_time;
static bool g_optix_async_cache_likely_warm = false;
// Pending result stash: once future.get() consumes the worker's output we hold the
// new backend here until rendering_in_progress is false so we can swap without
// tearing down GPU resources mid-animation/render frame.
static std::unique_ptr<Backend::IBackend> g_optix_async_built_pending;

static bool isOptixDiskCacheLikelyWarm()
{
#ifdef _WIN32
    wchar_t exeBufW[MAX_PATH] = {};
    const DWORD exeLen = GetModuleFileNameW(nullptr, exeBufW, MAX_PATH);
    if (exeLen == 0 || exeLen >= MAX_PATH) return false;

    std::error_code ec;
    const std::filesystem::path cacheDir = std::filesystem::path(exeBufW).parent_path() / "optix_cache";
    if (!std::filesystem::exists(cacheDir, ec) || !std::filesystem::is_directory(cacheDir, ec)) {
        return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(cacheDir, ec)) {
        if (ec) return false;
        if (entry.is_regular_file(ec) && entry.file_size(ec) > 0) {
            return true;
        }
    }
#endif
    return false;
}

static const char* optixAsyncHudMessage()
{
    return g_optix_async_cache_likely_warm
        ? "Preparing OptiX backend from shader cache. Renderer keeps running."
        : "OptiX shaders compiling on GPU (~1-3 min on first launch). Renderer keeps running.";
}

static float optixAsyncHudDuration(bool stillWantsOptix)
{
    if (!stillWantsOptix) return 0.05f;
    return g_optix_async_cache_likely_warm ? 5.0f : 999.0f;
}

void detectOptixHardware()
{
    SCENE_LOG_INFO("--- Hardware Detection ---");
#ifdef _WIN32
    // Prefer using the CUDA Driver API dynamically via nvcuda.dll so we never
    // rely on delay-load imports. This avoids SEH crashes when the driver is
    // missing. We'll query device count and compute capability from the driver.
    HMODULE hDriver = Platform::Dll::loadModuleWithPolicy("nvcuda.dll", Platform::Dll::DllCategory::Driver, false);
    if (!hDriver) {
        g_hasOptix = false;
        g_hasCUDA = false;
        g_gpu_name = "CPU Only";
        SCENE_LOG_WARN("NVIDIA Driver (nvcuda.dll) not found. GPU Rendering disabled.");
        return;
    }

    using PFN_cuInit = int(*)(unsigned int);
    using PFN_cuDeviceGetCount = int(*)(int*);
    using PFN_cuDeviceGet = int(*)(int*, int);
    using PFN_cuDeviceGetName = int(*)(char*, int, int);
    using PFN_cuDeviceComputeCapability = int(*)(int*, int*, int);

    PFN_cuInit pcuInit = (PFN_cuInit)GetProcAddress(hDriver, "cuInit");
    PFN_cuDeviceGetCount pcuDeviceGetCount = (PFN_cuDeviceGetCount)GetProcAddress(hDriver, "cuDeviceGetCount");
    PFN_cuDeviceGet pcuDeviceGet = (PFN_cuDeviceGet)GetProcAddress(hDriver, "cuDeviceGet");
    PFN_cuDeviceGetName pcuDeviceGetName = (PFN_cuDeviceGetName)GetProcAddress(hDriver, "cuDeviceGetName");
    PFN_cuDeviceComputeCapability pcuDevComputeCap = (PFN_cuDeviceComputeCapability)GetProcAddress(hDriver, "cuDeviceComputeCapability");

    if (!pcuInit || !pcuDeviceGetCount) {
        // Driver lacks expected symbols
        g_hasOptix = false;
        g_hasCUDA = false;
        g_gpu_name = "CPU Only";
        SCENE_LOG_WARN("CUDA Driver API symbols not found in nvcuda.dll. GPU Rendering disabled.");
        return;
    }

    int cuRes = pcuInit(0);
    if (cuRes != 0) {
        g_hasOptix = false;
        g_hasCUDA = false;
        g_gpu_name = "CPU Only";
        SCENE_LOG_WARN("cuInit failed. GPU Rendering disabled.");
        return;
    }

    int deviceCount = 0;
    cuRes = pcuDeviceGetCount(&deviceCount);
    if (cuRes != 0 || deviceCount == 0) {
        g_hasOptix = false;
        g_hasCUDA = false;
        g_gpu_name = "CPU Only";
        SCENE_LOG_WARN("CUDA Detection Failed: no driver devices found. Falling back to CPU mode.");
        return;
    }

    g_hasCUDA = true;
    SCENE_LOG_INFO("Found " + std::to_string(deviceCount) + " CUDA device(s).");

    bool foundOptix = false;
    for (int i = 0; i < deviceCount; ++i) {
        int dev = 0;
        if (pcuDeviceGet) pcuDeviceGet(&dev, i);

        int major = 0, minor = 0;
        if (pcuDevComputeCap) pcuDevComputeCap(&major, &minor, dev);

        char nameBuf[128] = "";
        if (pcuDeviceGetName) pcuDeviceGetName(nameBuf, sizeof(nameBuf), dev);

        bool capable = isOptixCapable(major, minor);
        std::string devName = nameBuf[0] ? std::string(nameBuf) : ("CUDA Device " + std::to_string(i));

        SCENE_LOG_INFO("Device [" + std::to_string(i) + "]: " + devName +
                       " (SM " + std::to_string(major) + "." + std::to_string(minor) + ") - " +
                       (capable ? "OptiX Capable" : "OptiX NOT Capable"));

        if (capable && !foundOptix) {
            g_gpu_name = devName;
            foundOptix = true;
            g_has_rt_cores = hasRTCores(major, minor);
            if (g_has_rt_cores) {
                SCENE_LOG_INFO("Selected: " + devName + " (Hardware RT enabled)");
            } else {
                SCENE_LOG_INFO("Selected: " + devName + " (OptiX Compute mode)");
            }

            // OptiX 9.0.0 SDK officially supports up to Ada (SM 8.9).
            // Blackwell (SM 10.0+) may work but JIT can take minutes on first run,
            // or initialization may fail outright. The user will get a CPU fallback.
            if (major >= 10) {
                SCENE_LOG_WARN("[GPU Compat] " + devName + " is SM " +
                    std::to_string(major) + "." + std::to_string(minor) +
                    " (Blackwell or newer). OptiX 9.0.0 SDK may not officially support this "
                    "architecture. First-time JIT compilation may take several minutes "
                    "— the splash screen WILL appear frozen during this time, that is normal. "
                    "If initialization ultimately fails, the app will fall back to CPU rendering. "
                    "Upgrade to OptiX 9.1+ for full Blackwell support.");
            }
        }
    }

    g_hasOptix = foundOptix;
    if (!g_hasOptix) {
        SCENE_LOG_WARN("No OptiX-compatible GPU (Maxwell SM 5.0+) found. GPU Rendering will be disabled.");
    }

    SCENE_LOG_INFO("--------------------------");
#else
    // Non-Windows platforms rely on standard CUDA runtime checks elsewhere
    g_hasCUDA = false;
    g_hasOptix = false;
    g_gpu_name = "CPU Only";
#endif
}
std::string WStringToString(const std::wstring& wstr) {
    if (wstr.empty()) return {};
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), nullptr, 0, nullptr, nullptr);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), strTo.data(), size_needed, nullptr, nullptr);
    return strTo;
}
bool initializeOptixIfAvailable() {
    if (!g_hasOptix) return false;

    try {
        // Create an OWNING backend (it will create its own OptixWrapper internally)
        auto backend = std::make_unique<Backend::OptixBackend>();
        g_backend = std::move(backend);
        
        // Initialize the backend
        if (!g_backend->initialize()) {
            SCENE_LOG_ERROR("OptiX backend initialize() returned false.");
            shutdownAndResetBackendSafe("optix_initialize_failed");
            // Initialization failed - mark OptiX as unavailable so UI/backends
            // won't continue to present it as a selectable runtime.
            g_hasOptix = false;
            return false;
        }
        
        // Set backend to renderer and UI
        ray_renderer.setBackend(g_backend.get());
        ui_ctx.backend_ptr = g_backend.get();
        if (auto* optixBackend = dynamic_cast<Backend::OptixBackend*>(g_backend.get())) {
            ui_ctx.optix_gpu_ptr = optixBackend->getOptixWrapper();
        } else {
            ui_ctx.optix_gpu_ptr = nullptr;
        }

        auto load_ptx = [](const std::wstring& filename) -> std::string {
            std::filesystem::path ptx_path = std::filesystem::path(L"ptx") / filename;
            if (!std::filesystem::exists(ptx_path)) ptx_path = filename; // legacy: PTX directly in exe dir
            std::ifstream file(ptx_path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open PTX file: " + WStringToString(ptx_path));
            }
            return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        };

        Backend::ShaderProgramData shader_data;
        std::string raygen_ptx = load_ptx(L"raygen.ptx");
        std::string miss_ptx = load_ptx(L"miss_kernels.ptx");
        std::string hitgroup_ptx = load_ptx(L"hitgroup_kernels.ptx");

        shader_data.raygen = raygen_ptx;
        shader_data.miss = miss_ptx;
        shader_data.hitgroup = hitgroup_ptx;

        g_backend->loadShaders(shader_data);
    }
    catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("OptiX initialization failed: ") + e.what());
        shutdownAndResetBackendSafe("optix_initialize_exception");
        g_hasOptix = false;
        return false;
    }
    catch (...) {
        SCENE_LOG_ERROR("OptiX initialization failed with unknown exception.");
        shutdownAndResetBackendSafe("optix_initialize_unknown_exception");
        g_hasOptix = false;
        return false;
    }

    return true;
}

// Worker-thread-safe OptiX backend builder.
// Returns a fully-initialized OptixBackend with shaders loaded, or nullptr on failure.
// Does NOT touch g_backend, ray_renderer, ui_ctx, or any other shared globals —
// installation is the main thread's job (see installPrebuiltOptixBackend below).
// CUDA/OptiX context creation and module compile on a worker thread is supported
// by the API; the primary CUDA context is shared across threads in the same process,
// and OptixDeviceContext is documented as thread-safe.
std::unique_ptr<Backend::IBackend> buildOptixBackendWorker() {
    if (!g_hasOptix) return nullptr;
    try {
        auto backend = std::make_unique<Backend::OptixBackend>();
        if (!backend->initialize()) {
            SCENE_LOG_ERROR("[OptiX async] backend->initialize() returned false on worker thread.");
            return nullptr;
        }

        auto load_ptx = [](const std::wstring& filename) -> std::string {
            std::filesystem::path ptx_path = std::filesystem::path(L"ptx") / filename;
            if (!std::filesystem::exists(ptx_path)) ptx_path = filename; // legacy: PTX directly in exe dir
            std::ifstream file(ptx_path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open PTX file: " + WStringToString(ptx_path));
            }
            return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        };

        Backend::ShaderProgramData shader_data;
        shader_data.raygen = load_ptx(L"raygen.ptx");
        shader_data.miss = load_ptx(L"miss_kernels.ptx");
        shader_data.hitgroup = load_ptx(L"hitgroup_kernels.ptx");
        backend->loadShaders(shader_data);
        return backend;
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("[OptiX async] worker exception: ") + e.what());
        return nullptr;
    } catch (...) {
        SCENE_LOG_ERROR("[OptiX async] worker unknown exception.");
        return nullptr;
    }
}

// Main-thread installer for a pre-built OptiX backend. Tears down the currently
// active backend (whatever it is) and swaps in the new one, then wires up the
// renderer + UI pointers. Must run on the UI/main thread — touches globals.
// IMPORTANT: caller must guarantee rendering_in_progress is false and detach the
// renderer before calling. We do the detach defensively here too.
bool installPrebuiltOptixBackend(std::unique_ptr<Backend::IBackend> new_backend) {
    if (!new_backend) {
        g_hasOptix = false;
        return false;
    }
    // Detach the renderer/UI from the live backend before destroying it. The async
    // path's caller has typically already done this, but redo it as a safety net so
    // the renderer can't dereference a dangling pointer between shutdown and swap.
    ray_renderer.setBackend(nullptr);
    ui_ctx.backend_ptr = nullptr;
    ui_ctx.optix_gpu_ptr = nullptr;

    shutdownAndResetBackendSafe("optix_async_swap");
    g_backend = std::move(new_backend);
    ray_renderer.setBackend(g_backend.get());
    ui_ctx.backend_ptr = g_backend.get();
    if (auto* optixBackend = dynamic_cast<Backend::OptixBackend*>(g_backend.get())) {
        ui_ctx.optix_gpu_ptr = optixBackend->getOptixWrapper();
    } else {
        ui_ctx.optix_gpu_ptr = nullptr;
    }
    return true;
}

bool initializeVulkanIfAvailable() {
#ifdef _WIN32
    HMODULE vulkanDll = Platform::Dll::loadModuleWithPolicy("vulkan-1.dll", Platform::Dll::DllCategory::Driver, false);
    if (!vulkanDll) {
        SCENE_LOG_WARN("Vulkan Driver (vulkan-1.dll) not found. Vulkan Backend disabled.");
        return false;
    }
#endif
    try {
        auto backend = std::make_unique<Backend::VulkanBackendAdapter>();
        if (!backend->initialize()) {
            SCENE_LOG_ERROR("Vulkan backend initialization returned false.");
            return false;
        }
        
        g_backend = std::move(backend);
        ray_renderer.setBackend(g_backend.get());
        ui_ctx.backend_ptr = g_backend.get();
        ui_ctx.optix_gpu_ptr = nullptr;
        return true;
    }
    catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("Vulkan initialization failed: ") + e.what());
        return false;
    }
    catch (...) {
        SCENE_LOG_ERROR("Vulkan initialization failed with unknown exception.");
        return false;
    }
}



void init_RayTrophi_Pro_Dark_Thema()
{
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* c = style.Colors;

    // Modern yumu�akl�k
    style.FrameRounding = 6.0f;
    style.WindowRounding = 5.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.TabRounding = 5.0f;
    style.PopupRounding = 4.0f;

    // Daha d�z ve net border
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;

    // Accent Color (marka rengi)
    ImVec4 accent = ImVec4(0.05f, 0.75f, 0.65f, 1.0f);

    // Arkaplanlar
    c[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.11f, ui.panel_alpha);
    c[ImGuiCol_ChildBg] = ImVec4(0.10f, 0.10f, 0.11f, 0.95f);
    c[ImGuiCol_PopupBg] = ImVec4(0.11f, 0.11f, 0.13f, 1.0f);

    // Yaz�lar
    c[ImGuiCol_Text] = ImVec4(0.95f, 0.97f, 0.99f, 1.0f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.45f, 0.45f, 0.48f, 1.0f);

    // �er�eveler
    c[ImGuiCol_FrameBg] = ImVec4(0.17f, 0.17f, 0.19f, 1);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.23f, 0.23f, 0.26f, 1);
    c[ImGuiCol_FrameBgActive] = accent;

    // Butonlar
    c[ImGuiCol_Button] = ImVec4(0.21f, 0.21f, 0.23f, 1);
    c[ImGuiCol_ButtonHovered] = accent;
    c[ImGuiCol_ButtonActive] = ImVec4(0.03f, 0.63f, 0.55f, 1.0f);

    // Header / se�ili elemanlar
    c[ImGuiCol_Header] = ImVec4(0.19f, 0.19f, 0.21f, 1);
    c[ImGuiCol_HeaderHovered] = accent;
    c[ImGuiCol_HeaderActive] = ImVec4(0.03f, 0.63f, 0.55f, 1);

    // Slider
    c[ImGuiCol_SliderGrab] = accent;
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.03f, 0.63f, 0.55f, 1);

    // Border
    c[ImGuiCol_Border] = ImVec4(0.0f, 0.0f, 0.0f, 0.60f);
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);

    // Separator'lar (daha belirgin, modern �izgi)
    style.SeparatorTextPadding = ImVec2(10.0f, 8.0f);
    style.SeparatorTextAlign = ImVec2(0.0f, 0.5f); // 
    style.SeparatorTextBorderSize = 2.0f;  // default 1.0f
	// Accent renkli separator
    c[ImGuiCol_Separator] = ImVec4(0.25f, 0.75f, 0.70f, 0.60f); // accent soft
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.25f, 0.80f, 0.75f, 1.00f);
    c[ImGuiCol_SeparatorActive] = ImVec4(0.20f, 0.90f, 0.85f, 1.00f);
   
    style.ItemSpacing = ImVec2(8, 6);   // 
    style.ItemInnerSpacing = ImVec2(6, 6);
}

int main(int argc, char* argv[]) try {
    installEarlyCrashHandlers();

    // Optional diagnostics:
    //   --startup-diagnostics : verbose startup breadcrumb logs
    //   --no-startup-diagnostics : suppress breadcrumb logs (default behavior)
    g_startupDiagVerbose = hasArg(argc, argv, "--startup-diagnostics") &&
                           !hasArg(argc, argv, "--no-startup-diagnostics");

    emergencyStartupLog(std::string("[Startup] main() entered ver=") +
                        kStartupDiagVersion +
                        " pid=" + std::to_string(GetCurrentProcessId()) +
                        " verbose=" + (g_startupDiagVerbose ? "1" : "0"));

    // Faz 0 Jolt Physics integration self-test. Runs a tiny standalone Jolt
    // world (sphere dropped onto a floor) and exits — proves Jolt links and
    // runs before any adapter/runtime wiring exists. Normal launches skip this.
    // Result goes to SceneLog.txt so it is visible even with the console closed.
    if (hasArg(argc, argv, "--jolt-selftest")) {
        auto logResult = [](const char* tag, const RayTrophiSim::JoltIntegration::SmokeTestResult& r) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "[Jolt][%s] init=%d stepped=%d steps=%d start_y=%.3f final_y=%.3f (expected ~0.5)",
                          tag, r.initialized ? 1 : 0, r.stepped ? 1 : 0, r.steps, r.start_y, r.final_y);
            const bool ok = r.initialized && r.stepped && r.final_y > 0.2f && r.final_y < 0.8f;
            if (ok) SCENE_LOG_INFO(std::string(buf) + " -> PASS");
            else    SCENE_LOG_ERROR(std::string(buf) + " -> FAIL");
        };
        // Raw-Jolt link/runtime proof (Faz 0) + JoltWorld wrapper proof (Faz 1).
        logResult("SmokeTest", RayTrophiSim::JoltIntegration::runSmokeTest());
        logResult("WorldTest", RayTrophiSim::JoltIntegration::runWorldTest());
        return 0;
    }

    // Register clean-shutdown cleanup: remove StartupCrash.log when the
    // application exits normally. Crashes will not run this handler.
    std::atexit(removeStartupCrashLogIfExists);

#ifdef _WIN32
    checkVcRuntimeRedistributable();
#endif

    setlocale(LC_ALL, "Turkish");
    startupDiagLog("[Startup] locale set");
    
    // Save project path passed via command line
    std::string project_to_load = "";
    if (argc > 1) {
        project_to_load = std::filesystem::absolute(argv[1]).string();
    }
    startupDiagLog("[Startup] argv parsed");
    
#ifdef _WIN32
    // Konsolu tamamen kapat
    FreeConsole();
    startupDiagLog("[Startup] FreeConsole done");
    
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    std::filesystem::current_path(std::filesystem::path(exePath).parent_path());
    startupDiagLog("[Startup] current_path set (win32)");
#else
    if (argc > 0) {
        std::filesystem::current_path(std::filesystem::path(argv[0]).parent_path());
    }
    startupDiagLog("[Startup] current_path set (non-win32)");
#endif

    // Ensure logger is firmly bound to the application directory (prevent wandering logs)
    g_sceneLog.initLogLocation();
    startupDiagLog("[Startup] g_sceneLog.initLogLocation done");

#ifdef _WIN32
    notifyIfVcRuntimeMissing();
#endif

   // SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
 // �nce SDL a��lacak
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        SCENE_LOG_ERROR(std::string("SDL_Init Error: ") + SDL_GetError());
        return 1;
    }
    startupDiagLog("[Startup] SDL_Init done");
    SDL_EnableScreenSaver();
    startupDiagLog("[Startup] SDL_EnableScreenSaver done");
    // ===========================================================================
    // SPLASH SCREEN - Frameless startup screen with loading status
    // ===========================================================================
    SplashScreen splash;
    startupDiagLog("[Startup] SplashScreen constructed");
    bool splashOk = splash.init("RayTrophi_image.png", 900, 700);
    startupDiagLog(std::string("[Startup] splash.init done, ok=") + (splashOk ? "1" : "0"));
    if (splashOk) {
        splash.setStatus("Initializing...");
        splash.render();
        startupDiagLog("[Startup] splash first render done");
    }
    
    // Detect GPU Hardware
    if (splashOk) { splash.setStatus("Detecting CUDA/OptiX hardware..."); splash.render(); }
        startupDiagLog("[Startup] before g_sceneLog.clear");
    g_sceneLog.clear();
        startupDiagLog("[Startup] g_sceneLog.clear done");
    detectOptixHardware();
        startupDiagLog("[Startup] detectOptixHardware done");

    // Enable IME UI (must be set before creating window) so SDL can show native IME when needed
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");

    // Create main window
    if (splashOk) { splash.setStatus("Creating main window..."); splash.render(); }
    startupDiagLog("[Startup] before SDL_CreateWindow");
      window = SDL_CreateWindow("RayTrophi Studio",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        image_width,
        image_height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN);
      startupDiagLog(std::string("[Startup] SDL_CreateWindow done, ptr=") + (window ? "valid" : "null"));

     renderer =
        SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
      startupDiagLog(std::string("[Startup] SDL_CreateRenderer done, ptr=") + (renderer ? "valid" : "null"));

     surface = SDL_CreateRGBSurfaceWithFormat(0, image_width, image_height, 32, SDL_PIXELFORMAT_RGBA32);
     original_surface = SDL_CreateRGBSurfaceWithFormat(0, image_width, image_height, 32, SDL_PIXELFORMAT_RGBA32);
     // [FIX] Prevent uninitialized pixel data causing random black/white on startup
     if (surface) SDL_FillRect(surface, nullptr, SDL_MapRGBA(surface->format, 0, 0, 0, 255));
     if (original_surface) SDL_FillRect(original_surface, nullptr, SDL_MapRGBA(original_surface->format, 0, 0, 0, 255));
    startupDiagLog("[Startup] SDL_CreateRGBSurfaceWithFormat done");

     raytrace_texture =
        SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
            SDL_TEXTUREACCESS_STREAMING, image_width, image_height);
     startupDiagLog(std::string("[Startup] SDL_CreateTexture done, ptr=") + (raytrace_texture ? "valid" : "null"));

    SDL_SetTextureBlendMode(raytrace_texture, SDL_BLENDMODE_NONE);
     startupDiagLog("[Startup] SDL_SetTextureBlendMode done");
    // Ensure SDL_TEXTINPUT events are enabled so ImGui_ImplSDL2 can receive UTF-8 characters
    SDL_StartTextInput();
    
    // Initialize ImGui
    if (splashOk) { splash.setStatus("Initializing ImGui..."); splash.render(); }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    startupDiagLog("[Startup] ImGui::CreateContext done");
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // Modern dockable panel layout. Internal docking only — the SDLRenderer2 backend
    // does not support multi-viewport (OS-detached) windows, so we do NOT set
    // ImGuiConfigFlags_ViewportsEnable here.
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);
    startupDiagLog("[Startup] ImGui SDL init done");

    // Initialize Theme using ThemeManager (prevents style overriding issues)
    ThemeManager::instance().setTheme("RayTrophi Pro Dark");
    ThemeManager::instance().applyCurrentTheme(ui.panel_alpha); 
    startupDiagLog("[Startup] ThemeManager init done");

    auto attachActiveBackendStatusCallback = [&]() {
        if (!g_backend) {
            return;
        }
        g_backend->setStatusCallback([&](const std::string& msg, int type) {
            ImVec4 color = ImVec4(1, 1, 1, 1);
            float  duration = 4.0f;
            if (type == 1) { color = ImVec4(1, 1, 0, 1);         duration = 6.0f; }
            if (type == 2) { color = ImVec4(1, 0.2f, 0.2f, 1);   duration = 8.0f; }
            ui.addViewportMessage(msg, duration, color);
        });
    };

    // -----------------------------------------------------------------------
    // syncActiveRenderBackendScene — GRANULAR version
    // Only re-uploads subsystems whose dirty flag is set. Falls back to full
    // sync when forceFullSync is true (e.g. first sync after backend create).
    // -----------------------------------------------------------------------
    auto syncVulkanWorldWithAtmosphere = [&](Backend::VulkanBackendAdapter* vulkanBackend, WorldData& wd) {
        if (!vulkanBackend) {
            return;
        }

        if (wd.mode == WORLD_MODE_NISHITA) {
            if (vulkanBackend->generateAtmosphereLUTGPU(&wd)) {
                ray_renderer.world.clearLUTDirty();
                wd = ray_renderer.world.getGPUData();
                return;
            }

            if (ray_renderer.world.needsLUTUpdate() && ray_renderer.world.flushLUT()) {
                wd = ray_renderer.world.getGPUData();
            }
            if (auto* al = ray_renderer.world.getLUT()) {
                if (al->is_initialized()) {
                    vulkanBackend->uploadAtmosphereLUT(al);
                    wd = ray_renderer.world.getGPUData();
                }
            }
        }

        vulkanBackend->setWorldData(&wd);
    };

    auto syncWorldDataToBackend = [&](Backend::IBackend* backend) {
        if (!backend) {
            return;
        }

        std::unique_ptr<ScopedCudaTextureUpload> allowCudaWorldTextureUpload;
        if (g_hasCUDA && g_hasOptix && dynamic_cast<Backend::OptixBackend*>(backend) != nullptr) {
            allowCudaWorldTextureUpload = std::make_unique<ScopedCudaTextureUpload>();
        }

        WorldData wd = ray_renderer.world.getGPUData();
        if (auto* vulkanBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(backend)) {
            struct VulkanEnvCache {
                const Texture* texture = nullptr;
                int64_t handle = 0;
                uint64_t textureCacheGeneration = 0;
            };
            static std::unordered_map<Backend::IBackend*, VulkanEnvCache> s_vulkanEnvCache;

            auto uploadVulkanEnvironmentTexture = [&](const Texture* texture) -> int64_t {
                if (!texture || !texture->is_loaded() || texture->width <= 0 || texture->height <= 0) {
                    s_vulkanEnvCache.erase(backend);
                    return 0;
                }

                auto& cache = s_vulkanEnvCache[backend];
                VulkanRT::ImageHandle existingEnvImage{};
                const VkFormat expectedFormat = (texture->is_hdr && !texture->float_pixels.empty())
                    ? VK_FORMAT_R32G32B32A32_SFLOAT
                    : VK_FORMAT_R8G8B8A8_UNORM;
                const bool cacheValid =
                    cache.texture == texture &&
                    cache.handle != 0 &&
                    cache.textureCacheGeneration == vulkanBackend->textureCacheGeneration() &&
                    vulkanBackend->tryGetUploadedImageHandle(cache.handle, existingEnvImage) &&
                    existingEnvImage.width == static_cast<uint32_t>(texture->width) &&
                    existingEnvImage.height == static_cast<uint32_t>(texture->height) &&
                    existingEnvImage.format == expectedFormat;
                if (cacheValid) {
                    return cache.handle;
                }

                int64_t envHandle = 0;
                if (texture->is_hdr && !texture->float_pixels.empty()) {
                    envHandle = vulkanBackend->uploadTexture2D(
                        texture->float_pixels.data(),
                        static_cast<uint32_t>(texture->width),
                        static_cast<uint32_t>(texture->height),
                        4,
                        false,
                        true);
                } else if (!texture->pixels.empty()) {
                    envHandle = vulkanBackend->uploadTexture2D(
                        texture->pixels.data(),
                        static_cast<uint32_t>(texture->width),
                        static_cast<uint32_t>(texture->height),
                        4,
                        false,
                        false);
                }
                cache.texture = texture;
                cache.handle = envHandle;
                cache.textureCacheGeneration = vulkanBackend->textureCacheGeneration();
                return envHandle;
            };

            if (wd.mode == WORLD_MODE_HDRI) {
                int64_t envHandle = uploadVulkanEnvironmentTexture(ray_renderer.world.getHDRITexture());
                if (envHandle != 0) {
                    vulkanBackend->setEnvironmentMap(envHandle);
                } else {
                    vulkanBackend->setEnvironmentMap(0);
                }
            } else if (wd.mode == WORLD_MODE_NISHITA && wd.advanced.env_overlay_enabled != 0) {
                int64_t envHandle = uploadVulkanEnvironmentTexture(ray_renderer.world.getNishitaEnvOverlayTexture());
                if (envHandle != 0) {
                    vulkanBackend->setEnvironmentMap(envHandle);
                } else {
                    vulkanBackend->setEnvironmentMap(0);
                }
            } else {
                s_vulkanEnvCache.erase(backend);
                vulkanBackend->setEnvironmentMap(0);
            }
            syncVulkanWorldWithAtmosphere(vulkanBackend, wd);
        } else {
            backend->setWorldData(&wd);
        }
    };

    auto syncActiveRenderBackendScene = [&](bool forceFullSync = false) -> bool {
        if (!g_backend) {
            return false;
        }

        bool did_geometry = false;
        if (forceFullSync || g_geometry_dirty) {
            ray_renderer.rebuildBackendGeometry(scene);
            applyPendingDeleteVisibilityToBackend(scene, g_backend.get());
            g_geometry_dirty = false;
            did_geometry = true;
        }

        if (forceFullSync || g_materials_dirty || g_texture_pool_dirty || did_geometry) {
            ray_renderer.updateBackendMaterials(scene);
            syncMaterialBufferToViewportBackend(scene, ray_renderer);
            g_materials_dirty = false;
            // Clear after both backends have evicted their pools (each backend checks
            // g_texture_pool_dirty inside uploadMaterials after vkDeviceWaitIdle).
            g_texture_pool_dirty = false;
        }

        if (forceFullSync || g_gas_volumes_dirty) {
            ray_renderer.updateBackendGasVolumes(scene);
            g_gas_volumes_dirty = false;
        }

        if (forceFullSync || g_world_dirty) {
            syncWorldDataToBackend(g_backend.get());
            g_world_dirty = false;
        }

        if (forceFullSync || g_lights_dirty) {
            g_backend->setLights(scene.lights);
            g_lights_dirty = false;
        }

        if (forceFullSync || g_camera_dirty) {
            if (scene.camera) {
                ray_renderer.syncCameraToBackend(*scene.camera);
            }
            g_camera_dirty = false;
        }

        g_backend->resetAccumulation();
        if (forceFullSync || did_geometry) {
            // A full geometry sync has already rebuilt/refreshed the active
            // backend. Drop stale rebuild/refit requests that may have been
            // raised by live fluid/foam while a backend switch was in flight.
            g_optix_rebuild_pending = false;
            // EXCEPTION — Vulkan: rebuildBackendGeometry() above does NOT build for
            // Vulkan; it only RAISES g_vulkan_rebuild_pending so the heavy
            // updateGeometry()/createTLAS runs later in the deferred pending-block.
            // Clearing it here (correct for the SYNCHRONOUS OptiX path) silently
            // cancels that rebuild — after switching back to Vulkan RT the TLAS is
            // never built, so geometry is invisible and accumulation is stuck at
            // pass 0 (the reported bug; only Vulkan RT, because OptiX/CPU rebuild
            // synchronously). Keep the full-rebuild flag for Vulkan; the lighter
            // append/refit requests below are subsumed by it and safe to drop.
            if (!render_settings.use_vulkan) {
                g_vulkan_rebuild_pending = false;
            }
            g_vulkan_geometry_append_pending = false;
            g_gpu_refit_pending = false;
        }
        g_needs_optix_sync.store(false, std::memory_order_release);
        return true;
    };

    auto resolveRequestedRenderBackend = [&](bool allowAutoSelect, bool showHudWarnings) {
        bool requestedOptix = render_settings.use_optix;
        bool requestedVulkan = render_settings.use_vulkan;
        std::string fallbackMessage;

        if (requestedOptix && requestedVulkan) {
            // Vulkan RT is the recommended primary backend when hardware RT is
            // available — it outperforms OptiX in interactive viewport scenarios
            // across mesh / hair / volume after the async ping-pong + GPU tonemap +
            // LSS + persistent NanoVDB accessor refactor. Fall back to OptiX only
            // when Vulkan RT hardware support is absent.
            if (g_hasVulkanRT) {
                requestedOptix = false;
                SCENE_LOG_INFO("Both OptiX and Vulkan requested. Preferring Vulkan RT (recommended primary backend).");
            } else {
                requestedVulkan = false;
                SCENE_LOG_WARN("Both OptiX and Vulkan requested but no hardware Vulkan RT. Preferring OptiX.");
            }
        }

        if (requestedOptix && !g_hasOptix) {
            requestedOptix = false;
            if (g_hasVulkan) {
                requestedVulkan = true;
                fallbackMessage = "OptiX unsupported on this machine. Falling back to Vulkan.";
            } else {
                requestedVulkan = false;
                fallbackMessage = "OptiX unsupported on this machine. Falling back to CPU rendering.";
            }
        }

        if (requestedVulkan && !g_hasVulkan) {
            requestedVulkan = false;
            if (g_hasOptix) {
                requestedOptix = true;
                fallbackMessage = "Vulkan unsupported on this machine. Falling back to OptiX.";
            } else {
                fallbackMessage = "Vulkan unsupported on this machine. Falling back to CPU rendering.";
            }
        }

        if (!requestedOptix && !requestedVulkan && allowAutoSelect) {
            // Auto-select priority: Vulkan RT (hardware) → OptiX (CUDA) → CPU (Reference).
            // RTX / RT-capable cards naturally support Vulkan RT too, so when both are
            // available Vulkan wins the default slot; OptiX remains available but is
            // no longer auto-selected. Vulkan without hardware RT is NOT auto-selected
            // for rendering — it falls through to OptiX or CPU.
            if (g_hasVulkanRT) {
                requestedVulkan = true;
            } else if (g_hasOptix) {
                requestedOptix = true;
            }
        }

        render_settings.use_optix = requestedOptix;
        render_settings.use_vulkan = requestedVulkan;
        ui_ctx.render_settings.use_optix = requestedOptix;
        ui_ctx.render_settings.use_vulkan = requestedVulkan;

        if (!fallbackMessage.empty()) {
            SCENE_LOG_WARN(fallbackMessage);
            if (showHudWarnings) {
                ui.addViewportMessage(fallbackMessage, 7.0f, ImVec4(1.0f, 0.5f, 0.2f, 1.0f));
            }
        }
    };

    auto runSplashBusyTask = [&](const std::string& status, auto&& task) {
        if (!splashOk) {
            task();
            return;
        }

        splash.beginBusyStatus(status);
        auto future = std::async(std::launch::async, [&task]() {
            task();
        });

        while (future.wait_for(std::chrono::milliseconds(16)) != std::future_status::ready) {
            splash.tick();
            SDL_Delay(16);
        }

        future.get();
        splash.stopBusyStatus();
    };

    // ===========================================================================
    // VULKAN BACKEND TEST & CAPABILITY CHECK
    // ===========================================================================
    if (splashOk) { splash.setStatus("Checking Vulkan capabilities..."); splash.render(); }
    
    g_hasVulkan = false;
#ifdef _WIN32
    HMODULE testVulkan = Platform::Dll::loadModuleWithPolicy("vulkan-1.dll", Platform::Dll::DllCategory::Driver, false);
    if (testVulkan) {
        try {
            auto vulkanTest = std::make_unique<Backend::VulkanBackendAdapter>();
            if (vulkanTest->initialize()) {
                g_hasVulkan = true;
                // Query whether the device supports hardware ray-tracing
                if (auto* dev = vulkanTest->getVulkanDevice()) {
                    g_hasVulkanRT = dev->hasHardwareRT();
                } else {
                    g_hasVulkanRT = false;
                }
                SCENE_LOG_INFO(std::string("Vulkan capabilities checked and supported! Hardware RT: ") + (g_hasVulkanRT ? "yes" : "no"));
                vulkanTest->shutdown();
            }
        } catch (...) {
            SCENE_LOG_WARN("Vulkan test threw an exception, disabled.");
        }
    } else {
        SCENE_LOG_WARN("vulkan-1.dll not found. Vulkan capabilities disabled.");
    }
#else
    auto vulkanTest = std::make_unique<Backend::VulkanBackendAdapter>();
    if (vulkanTest->initialize()) {
        g_hasVulkan = true; // 
        SCENE_LOG_INFO("Vulkan capabilities checked and supported!");
        vulkanTest->shutdown();
    } else {
        g_hasVulkan = false; // 
    }
#endif

    if (g_hasVulkan) {
        (void)initializeViewportBackendIfAvailable();
    }

    // Load Project or Create Default Scene
    if (!project_to_load.empty() && std::filesystem::exists(project_to_load)) {
        if (splashOk) { splash.setStatus("Loading Project..."); splash.render(); }
        ProjectManager::getInstance().openProject(project_to_load, scene, render_settings, ray_renderer, g_backend.get(),
            [&](int progress, const std::string& msg) {
                if (splashOk) {
                    splash.setStatus(msg);
                    splash.render();
                }
            });
        ui.viewport_settings.shading_mode = g_hasVulkan ? 0 : 2;
    } else {
        if (splashOk) { splash.setStatus("Creating default scene..."); splash.render(); }
        createDefaultScene(scene, ray_renderer, g_backend.get());
        ui.viewport_settings.shading_mode = g_hasVulkan ? 0 : 2;
    }
    resolveRequestedRenderBackend(true, false);
    ui_ctx.render_settings.use_optix = render_settings.use_optix;
    ui_ctx.render_settings.use_vulkan = render_settings.use_vulkan;
    ui_ctx.render_settings.backend_changed = render_settings.backend_changed;
    ui.invalidateCache(); // Ensure procedural objects are listed/selectable
    prepareCpuPickingState(scene, ui);
    
    // Build initial BVH and OptiX structures
    if (splashOk) { splash.setStatus("Building BVH structures..."); splash.render(); }
    ray_renderer.world.initializeLUT(); // Essential for Nishita Sky performance
    ray_renderer.rebuildBVH(scene, UI_use_embree);
    if (g_viewport_backend) {
        WorldData wd = ray_renderer.world.getGPUData();
        if (auto* vkViewport = dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get())) {
            syncVulkanWorldWithAtmosphere(vkViewport, wd);
        } else {
            g_viewport_backend->setWorldData(&wd);
        }
        g_viewport_backend->setLights(scene.lights);
        if (scene.camera) {
            g_viewport_backend->syncCamera(*scene.camera);
        }
        if (ui.viewport_settings.shading_mode != 2) {
            g_viewport_raster_rebuild_pending = true;
        }
    }
    if (render_settings.use_optix || render_settings.use_vulkan) {
        // Always allow deferred prepare so the render backend gets created.
        // The granular syncActiveRenderBackendScene and generation-based
        // buildRasterGeometry will prevent redundant heavy work.
        g_deferred_render_backend_prepare_pending.store(true, std::memory_order_release);
        g_deferred_render_backend_prepare_delay_frames = 2;
        g_needs_optix_sync.store(true, std::memory_order_release);
    } else {
        g_deferred_render_backend_prepare_pending.store(false, std::memory_order_release);
        g_deferred_render_backend_prepare_delay_frames = 0;
        g_needs_optix_sync.store(false, std::memory_order_release);
    }
    g_needs_geometry_rebuild.store(false, std::memory_order_release);
    // Update initial camera vectors
    if (scene.camera) scene.camera->update_camera_vectors();

    // First-run PTX JIT / backend creation can block for a long time on a new
    // machine. Do the initial backend prepare before the main window is shown
    // so the app waits behind the splash instead of looking frozen in Solid mode.
    if (g_deferred_render_backend_prepare_pending.load(std::memory_order_acquire) &&
        !render_settings.backend_changed &&
        !ui_ctx.render_settings.backend_changed &&
        !rendering_in_progress.load()) {
        resolveRequestedRenderBackend(false, false);

        const bool currentIsVulkan = (dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr);
        const bool currentIsOptix = (dynamic_cast<Backend::OptixBackend*>(g_backend.get()) != nullptr);
        const bool requestedVulkan = render_settings.use_vulkan;
        const bool requestedOptix = render_settings.use_optix;
        const bool matchingBackendReady =
            (requestedVulkan && currentIsVulkan) ||
            (requestedOptix && currentIsOptix);

        if ((requestedVulkan || requestedOptix) && !matchingBackendReady) {
            bool success = true;
            if (requestedVulkan) {
                const std::string status =
                    "Vulkan JIT compile on " + g_gpu_name +
                    " (first launch, ~1-3 min)";
                runSplashBusyTask(status, [&]() {
                    success = initializeVulkanIfAvailable();
                    if (!success) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(180));
                        success = initializeVulkanIfAvailable();
                    }
                    if (!success) {
                        SCENE_LOG_ERROR("Startup fallback to CPU. Vulkan failed.");
                        render_settings.use_vulkan = false;
                        ui_ctx.render_settings.use_vulkan = false;
                    }
                });
            } else if (requestedOptix) {
                const std::string status =
                    "OptiX JIT compile on " + g_gpu_name +
                    " (first launch, ~1-3 min)";
                runSplashBusyTask(status, [&]() {
                    success = initializeOptixIfAvailable();
                    if (!success) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(180));
                        success = initializeOptixIfAvailable();
                    }
                    if (!success) {
                        SCENE_LOG_ERROR("Startup fallback to CPU. OptiX failed.");
                        render_settings.use_optix = false;
                        ui_ctx.render_settings.use_optix = false;
                    }
                });
            }

            if (success && g_backend) {
                attachActiveBackendStatusCallback();
                const std::string uploadStatus =
                    std::string("Uploading render scene to ") +
                    (requestedOptix ? "OptiX" : "Vulkan") +
                    " on GPU: " + g_gpu_name;
                runSplashBusyTask(uploadStatus, [&]() {
                    (void)syncActiveRenderBackendScene(true);
                });
            } else {
                extern bool g_cpu_sync_pending;
                g_cpu_sync_pending = true;
            }

            ui_ctx.render_settings.use_optix = render_settings.use_optix;
            ui_ctx.render_settings.use_vulkan = render_settings.use_vulkan;
            render_settings.backend_changed = false;
            ui_ctx.render_settings.backend_changed = false;
            g_deferred_render_backend_prepare_pending.store(false, std::memory_order_release);
            g_deferred_render_backend_prepare_delay_frames = 0;
        } else if (matchingBackendReady && g_needs_optix_sync.load(std::memory_order_acquire)) {
            const std::string uploadStatus =
                std::string("Uploading render scene to active GPU backend: ") + g_gpu_name;
            runSplashBusyTask(uploadStatus, [&]() {
                (void)syncActiveRenderBackendScene(true);
            });
            g_deferred_render_backend_prepare_pending.store(false, std::memory_order_release);
            g_deferred_render_backend_prepare_delay_frames = 0;
        }
    }

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
    // Prefer render_settings flags (they may be set before backend object exists)
    bool activeOptix = render_settings.use_optix || ui_ctx.render_settings.use_optix ||
                       (dynamic_cast<Backend::OptixBackend*>(g_backend.get()) != nullptr);
    bool activeVulkan = render_settings.use_vulkan || ui_ctx.render_settings.use_vulkan ||
                        (dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr);

    if (activeOptix) {
        if (g_has_rt_cores) {
            ui.addViewportMessage("GPU: " + g_gpu_name + " (RTX - Hardware Ray Tracing)", 5.0f, ImVec4(0.3f, 1.0f, 0.5f, 1.0f));
        } else {
            ui.addViewportMessage("GPU: " + g_gpu_name + " (Compute Mode - No RT Cores)", 6.0f, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
            ui.addViewportMessage("Performance may be slower than RTX cards", 6.0f, ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
        }
    } else if (activeVulkan) {
        ui.addViewportMessage("GPU: " + g_gpu_name + " (Vulkan Path Tracing)", 5.0f, ImVec4(0.45f, 0.8f, 1.0f, 1.0f));
    } else {
        ui.addViewportMessage("CPU Rendering Mode (No compatible GPU found)", 6.0f, ImVec4(1.0f, 0.5f, 0.3f, 1.0f));
    }

    // ===========================================================================
    // FORCE INITIAL RESOLUTION APPLY
    // Ensures rendering bounds, surfaces, and CPU buffers perfectly align 
    // down to the exact pixel, bypassing Windows DPI/Window-Frame issues.
    // ===========================================================================
    pending_width = image_width;
    pending_height = image_height;
    pending_resolution_change = true;

    SDL_Event e;
    while (!quit) {
        // If Vulkan reported memory pressure (typically OOM/shared-memory exhaustion),
        // schedule a safe backend recreate using the already hardened switch path.
        if (isActiveRenderBackendVulkan() &&
            g_vulkan_trim_recreate_requested.exchange(false, std::memory_order_acq_rel)) {
            if (!ui.scene_loading.load() && !g_scene_loading_in_progress.load()) {
                SCENE_LOG_WARN("Vulkan memory pressure detected. Scheduling safe Vulkan backend recreate.");
                ui.addViewportMessage("Vulkan memory pressure detected. Scheduling safe Vulkan backend recreate.", 5.0f, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
                render_settings.backend_changed = true;
                ui_ctx.render_settings.backend_changed = true;
                // Force a fresh accumulation after device recreate.
                start_render = true;
            } else {
                // Loader is active; defer to next safe loop iteration.
                g_vulkan_trim_recreate_requested.store(true, std::memory_order_release);
            }
        }

        if (g_deferred_render_backend_prepare_pending.load(std::memory_order_acquire) &&
            !render_settings.backend_changed &&
            !ui_ctx.render_settings.backend_changed &&
            !ui.scene_loading.load() &&
            !g_scene_loading_in_progress.load() &&
            !rendering_in_progress.load() &&
            !g_viewport_raster_rebuild_pending) {
            if (g_deferred_render_backend_prepare_delay_frames > 0) {
                --g_deferred_render_backend_prepare_delay_frames;
            } else {
                resolveRequestedRenderBackend(false, false);

                const bool currentIsVulkan = (dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr);
                const bool currentIsOptix = (dynamic_cast<Backend::OptixBackend*>(g_backend.get()) != nullptr);
                const bool requestedVulkan = render_settings.use_vulkan;
                const bool requestedOptix = render_settings.use_optix;
                const bool matchingBackendReady =
                    (requestedVulkan && currentIsVulkan) ||
                    (requestedOptix && currentIsOptix);

                if ((requestedVulkan || requestedOptix) && !matchingBackendReady) {
                    render_settings.backend_changed = true;
                    ui_ctx.render_settings.backend_changed = true;
                } else if (matchingBackendReady && g_needs_optix_sync.load(std::memory_order_acquire)) {
                    (void)syncActiveRenderBackendScene();
                    g_deferred_render_backend_prepare_pending.store(false, std::memory_order_release);
                } else {
                    g_deferred_render_backend_prepare_pending.store(false, std::memory_order_release);
                }
            }
        }

        if (render_settings.backend_changed) {
            // Avoid backend destruction/recreation while a scene load thread is active.
            // The switch will be processed immediately after loading completes.
            if (ui.scene_loading.load() || g_scene_loading_in_progress.load()) {
                SDL_Delay(1);
            } else {
            resolveRequestedRenderBackend(false, true);
            // If user re-selects the already active backend, skip costly teardown/recreate.
            const bool currentIsVulkan = (dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr);
            const bool currentIsOptix  = (dynamic_cast<Backend::OptixBackend*>(g_backend.get()) != nullptr);
            const bool requestedVulkan = render_settings.use_vulkan;
            const bool requestedOptix  = render_settings.use_optix;
            const bool sameBackendRequested =
                (requestedVulkan && currentIsVulkan) ||
                (requestedOptix && currentIsOptix) ||
                ((!requestedVulkan && !requestedOptix) && !g_backend);

            if (sameBackendRequested) {
                if (g_backend) {
                    attachActiveBackendStatusCallback();
                    // Re-selecting the active Vulkan backend may still need a scene
                    // sync because New Project/initialization paths can mark the
                    // backend dirty without a teardown/recreate. But gate it on an
                    // ACTUAL dirty signal — an accidental re-select of the already
                    // active backend with nothing dirty must be a no-op, NOT a forced
                    // full geometry rebuild (the heavy deferred Vulkan TLAS build for
                    // nothing). OptiX already behaves this way; this brings Vulkan in
                    // line. The g_*_dirty set below is exactly what syncActiveRender-
                    // BackendScene consumes, so it's complete.
                    const bool sceneDirty =
                        g_geometry_dirty || g_materials_dirty || g_texture_pool_dirty ||
                        g_gas_volumes_dirty || g_world_dirty || g_lights_dirty ||
                        g_camera_dirty || g_vulkan_rebuild_pending;
                    const bool forceFull =
                        (currentIsVulkan && sceneDirty) ||
                        g_needs_optix_sync.load(std::memory_order_acquire);
                    if (forceFull) {
                        (void)syncActiveRenderBackendScene(currentIsVulkan);
                    }
                }
                render_settings.backend_changed = false;
                ui_ctx.render_settings.backend_changed = false;
                ui_ctx.render_settings.use_optix = render_settings.use_optix;
                ui_ctx.render_settings.use_vulkan = render_settings.use_vulkan;
                if (g_backend) g_backend->resetAccumulation();
                g_deferred_render_backend_prepare_pending.store(false, std::memory_order_release);
                g_deferred_render_backend_prepare_delay_frames = 0;
                continue;
            }

            // ┌─────────────────────────────────────────────────────────────────────┐
            // │ Async path: user switching TO OptiX from a different live backend.  │
            // │ Don't destroy the current backend here — it must keep rendering     │
            // │ during the 1-3 min JIT compile. Kick off a worker thread to build a │
            // │ new OptixBackend, then fall through to the end of the switch block. │
            // │ The polling block below this `if (backend_changed)` hot-swaps when  │
            // │ the future is ready (typically <5s post-cache, up to 3min on cold). │
            // └─────────────────────────────────────────────────────────────────────┘
            bool asyncOptixHandled = false;
            if (requestedOptix && !currentIsOptix) {
                if (!g_optix_async_in_progress.load(std::memory_order_acquire)) {
                    g_optix_async_cache_likely_warm = isOptixDiskCacheLikelyWarm();
                    g_optix_async_in_progress.store(true, std::memory_order_release);
                    g_optix_async_start_time = std::chrono::steady_clock::now();
                    g_optix_async_future = std::async(std::launch::async, &buildOptixBackendWorker);
                    SCENE_LOG_INFO("[OptiX async] worker launched; current backend keeps rendering.");
                }
                ui.addViewportMessage(
                    optixAsyncHudMessage(),
                    optixAsyncHudDuration(true), ImVec4(1.0f, 0.85f, 0.3f, 1.0f));
                render_settings.backend_changed = false;
                ui_ctx.render_settings.backend_changed = false;
                g_deferred_render_backend_prepare_pending.store(false, std::memory_order_release);
                g_deferred_render_backend_prepare_delay_frames = 0;
                asyncOptixHandled = true;
            }

            if (!asyncOptixHandled) {

            bool success = true;

            // If an async OptiX rebuild is still running, wait for it before backend teardown.
            if (g_optix_rebuilding && g_optix_future.valid()) {
                try {
                    g_optix_future.get();
                } catch (const std::exception& e) {
                    SCENE_LOG_WARN(std::string("OptiX async rebuild join failed during backend switch: ") + e.what());
                } catch (...) {
                    SCENE_LOG_WARN("OptiX async rebuild join failed during backend switch (unknown exception).");
                }
                g_optix_rebuilding = false;
                g_optix_rebuild_in_progress.store(false, std::memory_order_release);
            }

            // Detach renderer from current backend first to avoid renderer using a destroyed backend
            ray_renderer.setBackend(nullptr);
            ui_ctx.backend_ptr = nullptr;
            ui_ctx.optix_gpu_ptr = nullptr;

            // Never switch backend while a render/animation pass is still active.
            // The previous timeout-based behavior could tear down GPU objects while
            // animation thread was still touching backend resources (race/crash).
            if (rendering_in_progress.load()) {
                ui.addViewportMessage("Backend degisimi render/animasyon bitince uygulanacak.", 3.0f, ImVec4(1.0f, 0.85f, 0.3f, 1.0f));
                SDL_Delay(1);
                continue;
            }

            // Now safe to destroy existing backend
            shutdownAndResetBackendSafe("backend_switch");

            if (render_settings.use_vulkan) {
                if (g_hasCUDA) {
                    cudaDeviceSynchronize();
                    cudaGetLastError();
                }
                success = initializeVulkanIfAvailable();
                if (!success) {
                    // After OptiX shutdown, WDDM/driver residency updates may lag briefly.
                    // Retry once after a short settle delay before CPU fallback.
                    if (g_hasCUDA) {
                        cudaDeviceSynchronize();
                        cudaGetLastError();
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(350));
                    success = initializeVulkanIfAvailable();
                }
                if (!success) {
                    SCENE_LOG_ERROR("Fallback to CPU. Vulkan failed.");
                    render_settings.use_vulkan = false;
                    ui_ctx.render_settings.use_vulkan = false;
                    ui.addViewportMessage("Vulkan not available on this machine. Switched to CPU rendering.", 7.0f, ImVec4(1.0f, 0.5f, 0.2f, 1.0f));
                    // Keep capability flag intact so user can retry Vulkan later
                    // without restarting the application.
                }
            } else if (render_settings.use_optix) {
                // Async OptiX path is handled by the early-skip block above this big
                // if/else chain; if we somehow reach here it's a logic error — fall
                // through with success=true so we don't accidentally drop into CPU.
                SCENE_LOG_WARN("[OptiX] Sync init path reached unexpectedly; async should have handled the switch.");
                success = true;
            }

            // If Vulkan reported a runtime device loss, ensure the user sees a HUD message
            // and force a CPU fallback immediately.
            if (g_vulkan_device_lost) {
                try {
                    SCENE_LOG_ERROR(std::string("Vulkan device lost detected: ") + g_vulkan_device_lost_msg);
                } catch (...) {}
                ui.addViewportMessage("Vulkan device lost — switched to CPU rendering.", 8.0f, ImVec4(1.0f, 0.4f, 0.2f, 1.0f));
                render_settings.use_vulkan = false;
                ui_ctx.render_settings.use_vulkan = false;
                success = false;
                // clear the flag so we don't spam messages
                g_vulkan_device_lost = false;
                g_vulkan_device_lost_msg.clear();
            }

            if (!success && (!render_settings.use_optix && !render_settings.use_vulkan)) {
                // Now falling back to CPU
                extern bool g_cpu_sync_pending;
                g_cpu_sync_pending = true;
                ui_ctx.render_settings.use_optix = false;
                ui_ctx.render_settings.use_vulkan = false;
            }

            // Also handle explicit GPU -> CPU switch (successful path with no backend requested).
            // Without this, CPU may render with stale geometry/BVH after leaving GPU mode.
            if (!render_settings.use_optix && !render_settings.use_vulkan && !g_backend) {
                extern bool g_cpu_sync_pending;
                g_cpu_sync_pending = true;
                g_bvh_rebuild_pending = true;
                start_render = true;
            }

            if (g_backend) {
                // Reattach renderer to the new backend
                ray_renderer.setBackend(g_backend.get());
                ui_ctx.backend_ptr = g_backend.get();
                if (auto* optixBackend = dynamic_cast<Backend::OptixBackend*>(g_backend.get())) {
                    ui_ctx.optix_gpu_ptr = optixBackend->getOptixWrapper();
                } else {
                    ui_ctx.optix_gpu_ptr = nullptr;
                }
                attachActiveBackendStatusCallback();
                (void)syncActiveRenderBackendScene(true); // New backend — full sync required
            }

            if (g_hasVulkan) {
                (void)initializeViewportBackendIfAvailable();
            }
            if (g_viewport_backend) {
                auto wdViewport = ray_renderer.world.getGPUData();
                if (auto* vkViewport = dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get())) {
                    syncVulkanWorldWithAtmosphere(vkViewport, wdViewport);
                } else {
                    g_viewport_backend->setWorldData(&wdViewport);
                }
                g_viewport_backend->setLights(scene.lights);
                if (scene.camera) {
                    g_viewport_backend->syncCamera(*scene.camera);
                }
                if (ui.viewport_settings.shading_mode != 2) {
                    g_viewport_raster_rebuild_pending = true;
                }
                applyPendingDeleteVisibilityToBackend(scene, g_viewport_backend.get());
            }

            // Let backend/device settle for a few frames before animation-driven GPU updates
            // (dispatchSkinning/updateSceneGeometry) to avoid switch-edge races.
            g_backend_switch_cooldown_frames = 3;

            ui_ctx.render_settings.use_optix = render_settings.use_optix;
            ui_ctx.render_settings.use_vulkan = render_settings.use_vulkan;
            render_settings.backend_changed = false;
            ui_ctx.render_settings.backend_changed = false;
            g_deferred_render_backend_prepare_pending.store(false, std::memory_order_release);
            g_deferred_render_backend_prepare_delay_frames = 0;

            // [ANIM-SWAP FIX] Backend was just torn down and rebuilt. Next animation
            // start must do a full RT rebuild — the granular sync above with
            // forceFullSync=true at line 2211 already covered current viewport
            // needs, but an animation triggered immediately after swap can still
            // race with the cooldown window. Mark explicitly so the animation
            // start path doesn't trust stale granular dirty flags.
            g_anim_backend_needs_full_rebuild.store(true);

            } // end of if (!asyncOptixHandled)
            }
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Async OptiX init poll. Runs every frame regardless of backend_changed.
        // When the worker thread finishes building the new OptixBackend, we install
        // it here on the main thread (tear down current backend + assign new).
        // ─────────────────────────────────────────────────────────────────────────
        if (g_optix_async_in_progress.load(std::memory_order_acquire)) {
            const bool stillWantsOptix = render_settings.use_optix;
            const float hudDuration = optixAsyncHudDuration(stillWantsOptix);
            ui.addViewportMessage(
                optixAsyncHudMessage(),
                hudDuration, ImVec4(1.0f, 0.85f, 0.3f, 1.0f));

            // Step 1: drain the worker future once it's ready, stash result.
            if (g_optix_async_future.valid() &&
                g_optix_async_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                try {
                    g_optix_async_built_pending = g_optix_async_future.get();
                } catch (const std::exception& e) {
                    SCENE_LOG_ERROR(std::string("[OptiX async] future.get() threw: ") + e.what());
                    g_optix_async_built_pending.reset();
                } catch (...) {
                    SCENE_LOG_ERROR("[OptiX async] future.get() threw unknown exception.");
                    g_optix_async_built_pending.reset();
                }
            }

            // Step 2: install (or discard) once the worker has been drained AND no
            // other render pass is using GPU resources. Otherwise wait next frame.
            const bool drainedFuture = !g_optix_async_future.valid();

            if (drainedFuture && !rendering_in_progress.load()) {
                const float elapsed = std::chrono::duration<float>(
                    std::chrono::steady_clock::now() - g_optix_async_start_time).count();

                // Expire the "compiling" HUD line on next render regardless of branch.
                ui.addViewportMessage(
                    optixAsyncHudMessage(),
                    0.05f, ImVec4(1.0f, 0.85f, 0.3f, 1.0f));

                if (!stillWantsOptix) {
                    g_optix_async_built_pending.reset();
                    SCENE_LOG_INFO("[OptiX async] result discarded (user switched away during compile).");
                    ui.addViewportMessage(
                        "OptiX compile cancelled (user switched backend).",
                        5.0f, ImVec4(0.6f, 0.85f, 1.0f, 1.0f));
                } else if (g_optix_async_built_pending && installPrebuiltOptixBackend(std::move(g_optix_async_built_pending))) {
                    SCENE_LOG_INFO("[OptiX async] swap complete (" + std::to_string(elapsed) + "s).");
                    ui.addViewportMessage(
                        "OptiX ready (" + std::to_string((int)std::round(elapsed)) + "s).",
                        5.0f, ImVec4(0.45f, 0.95f, 0.45f, 1.0f));
                    ui_ctx.render_settings.use_optix = true;
                    ui_ctx.render_settings.use_vulkan = false;
                    render_settings.use_optix = true;
                    render_settings.use_vulkan = false;
                    attachActiveBackendStatusCallback();
                    (void)syncActiveRenderBackendScene(true);
                    g_backend_switch_cooldown_frames = 3;
                    g_anim_backend_needs_full_rebuild.store(true);
                } else {
                    SCENE_LOG_ERROR("[OptiX async] build failed; staying on previous backend.");
                    ui.addViewportMessage(
                        "OptiX init failed — staying on previous backend.",
                        7.0f, ImVec4(1.0f, 0.5f, 0.2f, 1.0f));
                    render_settings.use_optix = false;
                    ui_ctx.render_settings.use_optix = false;
                }
                g_optix_async_built_pending.reset();
                g_optix_async_in_progress.store(false, std::memory_order_release);
            }
        }

        bool did_render_this_frame = false;
        bool post_processing_happened = false;
        // =========================================================================
        // POWER MANAGEMENT - Prevent sleep during render but allow screen off
        // =========================================================================
        static bool last_sleep_state = false;
        bool busy_rendering = rendering_in_progress.load() || (ui_ctx.render_settings.is_rendering_active && ui_ctx.render_settings.is_final_render_mode);
        if (busy_rendering != last_sleep_state) {
            if (busy_rendering) {
                SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
            } else {
                SetThreadExecutionState(ES_CONTINUOUS);
            }
            last_sleep_state = busy_rendering;
        }

        // =========================================================================
        // INPUT & EVENT HANDLING (CRITICAL: MUST BE AT TOP FOR RESPONSIVENESS)
        // =========================================================================
        // ANIMATION RENDER LOCK: Block camera/viewport input during animation render
        bool input_locked = ui_ctx.render_settings.animation_render_locked || 
                           (rendering_in_progress && ui_ctx.is_animation_mode);

        // [FIX] Master guard: Skip ALL GPU backend calls when animation thread
        // is actively using the Vulkan/OptiX backend.  VulkanBackendAdapter
        // methods acquire m_mutex — if the main thread calls any of them while
        // renderProgressiveImpl holds the mutex, the main thread blocks for the
        // entire GPU trace duration, freezing the UI.
        // !g_seq_save_active: a viewport-driven sequence save keeps both flags set
        // for the UI, but it RENDERS through the live backend, so it must NOT skip it.
        const bool skip_backend_for_anim =
            rendering_in_progress.load() && ui_ctx.is_animation_mode && !g_seq_save_active;

        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if (e.type == SDL_QUIT) ui.tryExit();

            if (e.type == SDL_KEYDOWN) {
                // ===================================================================
                // ANIMATION RENDER SHORTCUTS - Always active, even during render!
                // ===================================================================
                if (rendering_in_progress && ui_ctx.is_animation_mode) {
                    if (e.key.keysym.sym == SDLK_ESCAPE) {
                        rendering_stopped_cpu = true;
                        rendering_stopped_gpu = true;
                        SCENE_LOG_WARN("Animation render stopped by ESC key.");
                        ui.addViewportMessage("Render Stopped (ESC)", 3.0f, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
                    }
                    if (e.key.keysym.sym == SDLK_p || e.key.keysym.sym == SDLK_SPACE) {
                        rendering_paused = !rendering_paused.load();
                        if (rendering_paused.load()) {
                             SCENE_LOG_INFO("Animation render PAUSED (press P or Space to resume)");
                             ui.addViewportMessage("PAUSED (P to resume)", 0.0f, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
                        } else {
                             SCENE_LOG_INFO("Animation render RESUMED");
                             ui.addViewportMessage("Resumed", 2.0f, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
                        }
                    }
                }
                
                if (e.key.keysym.sym == SDLK_s && (e.key.keysym.mod & KMOD_CTRL)) {
                     ui_ctx.render_settings.save_image_requested = true;
                }
                
                if (e.key.keysym.sym == SDLK_DELETE) {
                    if (!ImGui::GetIO().WantCaptureKeyboard) {
                        ui.triggerDelete(ui_ctx);
                    }
                }

                if (e.key.keysym.sym == SDLK_h) {
                    if (!ImGui::GetIO().WantCaptureKeyboard) {
                        for (auto& obj : scene.world.objects) {
                            if (!obj) continue;

                            bool visible = true;
                            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                                visible = !scene.isEditorPendingDeleteObjectName(tri->getNodeName());
                            } else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                                visible = !scene.isEditorPendingDeleteObjectName(inst->node_name);
                            }

                            obj->visible = visible;
                        }
                        for (auto& light : scene.lights) light->visible = true;
                        for (auto& vdb : scene.vdb_volumes) vdb->visible = true;
                        for (auto& gas : scene.gas_volumes) gas->visible = true;
                        if (g_hasOptix) {
                        if (g_backend) {
                            g_backend->showAllInstances();
                            applyPendingDeleteVisibilityToBackend(scene, g_backend.get());
                            g_backend->setLights(scene.lights);
                        }
                        }
                        if (g_viewport_backend) {
                            applyPendingDeleteVisibilityToBackend(scene, g_viewport_backend.get());
                            g_viewport_backend->resetAccumulation();
                        }
                        ray_renderer.resetCPUAccumulation();
                        SCENE_LOG_INFO("All objects and lights are now visible.");
                        ui.addViewportMessage("All objects visible", 2.0f, ImVec4(0.4f, 1.0f, 0.6f, 1.0f));
                    }
                }
            }

            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_MIDDLE && !input_locked) {
                dragging = true;
                last_mouse_x = e.button.x;
                last_mouse_y = e.button.y;

                // --- Navigation Distance Calibration ---
                // We raycast on middle-click to determine what the user is interacting with.
                // This updates 'current_nav_dist' to provide perfect panning/rotation speed.
                if (scene.initialized && scene.camera && scene.bvh &&
                    (!ImGui::GetIO().WantCaptureMouse || (ImGuizmo::IsOver() && !ImGuizmo::IsUsing()))) {
                    int win_w, win_h;
                    SDL_GetWindowSize(window, &win_w, &win_h);
                    
                    // Convert screen pixels to NDC-like coordinates [0, 1] for get_ray
                    float u = (float)e.button.x / (float)win_w;
                    float v = (float)(win_h - e.button.y) / (float)win_h;
                    
                    Ray r = scene.camera->get_ray(u, v);
                    HitRecord rec;

                    // FIX: Always sync current yaw/pitch from camera direction to prevent "one-time snap" 
                    // if the camera was moved by presets or other logic since last drag.
                    Vec3 dir = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                    pitch = asinf(std::clamp(dir.y, -0.999f, 0.999f)) * 180.0f / 3.1415926535f;
                    yaw = atan2f(dir.z, dir.x) * 180.0f / 3.1415926535f;

                    if (scene.bvh->hit(r, 0.001f, 10000.0f, rec)) {
                        // Cap the nav distance to something sane for panning/sensitivity (prevents "extreme movement")
                        const float safe_focus_dist = std::max(rec.t, 0.05f);
                        current_nav_dist = std::min(safe_focus_dist, 500.0f);
                        
                        // If not in Manual Focus mode (mode 0), sync pivot and focus_dist
                        if (ui.viewport_settings.focus_mode != 0) {
                            scene.camera->focus_dist = safe_focus_dist;
                            // FIXED: Use current forward direction instead of ray 'r' to prevent camera "turn snap"
                            scene.camera->lookat = scene.camera->lookfrom + dir * safe_focus_dist;
                            scene.camera->update_camera_vectors();
                            g_camera_dirty = true;
                        }
                    } else {
                        // Fallback to existing focus distance if hitting background
                        current_nav_dist = std::min(scene.camera->focus_dist, 500.0f);
                    }
                }
            }
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT && !input_locked) {
                if (!ImGui::GetIO().WantCaptureMouse) {
                    int win_w, win_h;
                    SDL_GetWindowSize(window, &win_w, &win_h);
                    rayhit = true;
                    mx = e.button.x;
                    my = win_h - e.button.y; 
                    u = (mx + 0.5f) / (float)win_w;
                    v = (my + 0.5f) / (float)win_h;
                }
            }
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_MIDDLE) {
                dragging = false;
                if (mouse_control_enabled && !input_locked)
                    start_render = true;
            }

            if (e.type == SDL_MOUSEMOTION && dragging && scene.camera && mouse_control_enabled && !input_locked) {
                if (!ImGui::GetIO().WantCaptureMouse || (ImGuizmo::IsOver() && !ImGuizmo::IsUsing())) {
                    int dx = e.motion.x - last_mouse_x;
                    int dy = e.motion.y - last_mouse_y;
                    const Uint8* state = SDL_GetKeyboardState(NULL);
                    bool is_shift_pressed = state[SDL_SCANCODE_LSHIFT] || state[SDL_SCANCODE_RSHIFT];

                    if (is_shift_pressed) {
                        // Very strict clamp on panning logic
                        // Distances above 100 units get heavily log-scaled
                        float dist_factor = current_nav_dist;
                        if (current_nav_dist > 100.0f) {
                            dist_factor = 100.0f + std::log10(current_nav_dist - 99.0f) * 10.0f;
                        }
                        
                        // Extremely small base speeds for panning
                        float pan_speed = (0.0005f + render_settings.mouse_sensitivity * 0.001f) * dist_factor;
                        
                        // Enforce a hard cap of 0.5 units per mouse pixel moved, regardless of distance
                        pan_speed = std::min(pan_speed, 0.5f);
                        
                        // Additional clamp: if dx or dy are somehow massive (e.g. framerate drops), clamp their values
                        int safe_dx = std::clamp(dx, -50, 50);
                        int safe_dy = std::clamp(dy, -50, 50);
                        
                        Vec3 offset = scene.camera->u * -(float)safe_dx * pan_speed + scene.camera->v * (float)safe_dy * pan_speed;
                        scene.camera->lookfrom += offset;
                        scene.camera->lookat += offset;
                        scene.camera->update_camera_vectors();
                    }
                    else {
                        bool is_ctrl_pressed = state[SDL_SCANCODE_LCTRL] || state[SDL_SCANCODE_RCTRL];
                        if (is_ctrl_pressed) {
                            float zoom_speed = 0.04f * current_nav_dist;
                            float zoom_amount = -(float)dy * zoom_speed * 0.1f; 
                            Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                            scene.camera->lookfrom += forward * zoom_amount;
                            scene.camera->lookat += forward * zoom_amount;
                            scene.camera->update_camera_vectors();
                        }
                        else {
                            // Rotation: Pan/Zoom can be fast, but rotation speed is strictly dampened
                            // so that high sensitivity doesn't lead to uncontrollable spinning.
                            float rot_speed = 0.1f + (render_settings.mouse_sensitivity * 0.05f);
                            yaw += dx * rot_speed;
                            pitch -= dy * rot_speed;
                            pitch = std::clamp(pitch, -89.9f, 89.9f);
                            float rad_yaw = yaw * 3.14159265f / 180.0f;
                            float rad_pitch = pitch * 3.14159265f / 180.0f;
                            Vec3 direction;
                            direction.x = cosf(rad_yaw) * cosf(rad_pitch);
                            direction.y = sinf(rad_pitch);
                            direction.z = sinf(rad_yaw) * cosf(rad_pitch);
                            scene.camera->vup = Vec3(0.0f, 1.0f, 0.0f);
                            // Free rotation leaves an aligned orthographic standard view and
                            // returns to a normal perspective camera.
                            scene.camera->orthographic = false;
                            scene.camera->standard_view = Camera::StandardView::Perspective;
                            scene.camera->setLookDirection(direction.normalize());
                        }
                    }
                    last_mouse_x = e.motion.x;
                    last_mouse_y = e.motion.y;
                    last_camera_move_time = std::chrono::steady_clock::now();
                    start_render = true;
                    g_camera_dirty = true;
                }
            }

            if (e.type == SDL_MOUSEWHEEL && mouse_control_enabled && scene.camera && !input_locked) {
                if (!ImGui::GetIO().WantCaptureMouse || (ImGuizmo::IsOver() && !ImGuizmo::IsUsing())) {
                    float scroll_amount = e.wheel.y;
                    const Uint8* k_state = SDL_GetKeyboardState(NULL);
                    bool is_shift = k_state[SDL_SCANCODE_LSHIFT] || k_state[SDL_SCANCODE_RSHIFT];
                    float wheel_boost = is_shift ? (3.0f + render_settings.mouse_sensitivity * 2.0f) : 1.0f;
                    float move_v = 1.5f * render_settings.mouse_sensitivity * wheel_boost;
                    if (scene.camera->orthographic) {
                        // Dollying does nothing under parallel projection — zoom by scaling the
                        // visible extent instead so the grid/geometry actually grow/shrink.
                        float zoom_factor = std::pow(0.9f, scroll_amount * wheel_boost);
                        scene.camera->ortho_height = std::clamp(
                            scene.camera->ortho_height * zoom_factor, 0.01f, 100000.0f);
                        scene.camera->update_camera_vectors();
                    } else {
                        Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                        scene.camera->lookfrom += forward * scroll_amount * move_v;
                        scene.camera->lookat = scene.camera->lookfrom + forward * scene.camera->focus_dist;
                        scene.camera->update_camera_vectors();
                    }
                    ui.updateAutofocus(ui_ctx);
                    last_camera_move_time = std::chrono::steady_clock::now();
                    camera_moved = true;
                    start_render = true;
                    g_camera_dirty = true;
                }
            }
        }

        // --- AUTO RESIZE FOR FINAL RENDER ---
        // Final render (F12) now uses the current viewport resolution — the
        // separate final_render_width/height fields and their auto-resize
        // dance have been retired. Keep the field synced for backward
        // compatibility with save files / panels that still display it.
        ui_ctx.render_settings.final_render_width = image_width;
        ui_ctx.render_settings.final_render_height = image_height;
        render_settings.final_render_width = image_width;
        render_settings.final_render_height = image_height;
        bool is_final_mode = ui_ctx.render_settings.is_final_render_mode;
        (void)is_final_mode; // no longer triggers a resize
        // If a previous version of this code switched to a separate final
        // render resolution, restore the saved viewport size on exit.
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
        bool timeline_has_manual_keyframes = false;
        bool timeline_has_water_keyframes = false;
        bool timeline_has_transform_keyframes = false;
        bool timeline_has_material_keyframes = false;
        bool timeline_has_light_keyframes = false;
        bool timeline_has_camera_keyframes = false;
        bool timeline_has_world_keyframes = false;
        static size_t cached_timeline_track_count = std::numeric_limits<size_t>::max();
        static size_t cached_timeline_keyframe_count = std::numeric_limits<size_t>::max();
        static bool cached_has_manual_keyframes = false;
        static bool cached_has_water_keyframes = false;
        static bool cached_has_transform_keyframes = false;
        static bool cached_has_material_keyframes = false;
        static bool cached_has_light_keyframes = false;
        static bool cached_has_camera_keyframes = false;
        static bool cached_has_world_keyframes = false;

        size_t current_timeline_keyframe_count = 0;
        for (const auto& [track_name, track] : scene.timeline.tracks) {
            (void)track_name;
            current_timeline_keyframe_count += track.keyframes.size();
        }

        if (!timeline_playing ||
            cached_timeline_track_count != scene.timeline.tracks.size() ||
            cached_timeline_keyframe_count != current_timeline_keyframe_count) {
            cached_timeline_track_count = scene.timeline.tracks.size();
            cached_timeline_keyframe_count = current_timeline_keyframe_count;
            cached_has_manual_keyframes = false;
            cached_has_water_keyframes = false;
            cached_has_transform_keyframes = false;
            cached_has_material_keyframes = false;
            cached_has_light_keyframes = false;
            cached_has_camera_keyframes = false;
            cached_has_world_keyframes = false;

            for (const auto& [track_name, track] : scene.timeline.tracks) {
                (void)track_name;
                if (track.keyframes.empty()) continue;
                cached_has_manual_keyframes = true;
                for (const auto& kf : track.keyframes) {
                    cached_has_transform_keyframes |= kf.has_transform;
                    cached_has_material_keyframes |= kf.has_material;
                    cached_has_light_keyframes |= kf.has_light;
                    cached_has_camera_keyframes |= kf.has_camera;
                    cached_has_world_keyframes |= kf.has_world;
                    cached_has_water_keyframes |= kf.has_water;
                }
            }
        }

        timeline_has_manual_keyframes = cached_has_manual_keyframes;
        timeline_has_water_keyframes = cached_has_water_keyframes;
        timeline_has_transform_keyframes = cached_has_transform_keyframes;
        timeline_has_material_keyframes = cached_has_material_keyframes;
        timeline_has_light_keyframes = cached_has_light_keyframes;
        timeline_has_camera_keyframes = cached_has_camera_keyframes;
        timeline_has_world_keyframes = cached_has_world_keyframes;

        bool autonomous_anim_graph_playing = false;
        for (const auto& modelCtx : scene.importedModelContexts) {
            auto activeGraph = modelCtx.runtimeGraph ? modelCtx.runtimeGraph : modelCtx.graph;
            if (!modelCtx.useAnimGraph || !activeGraph || modelCtx.animGraphFollowTimeline) {
                continue;
            }
            const auto playback = activeGraph->getPlaybackStatus();
            if (playback.isPlaying) {
                autonomous_anim_graph_playing = true;
                break;
            }
        }
        if (autonomous_anim_graph_playing) {
            start_render = true;
        }
        
        // ===================================================================
        // VOLUMETRIC TIMELINE UPDATE (Gas & VDB)
        // ===================================================================
        static int last_volume_frame = -1;
        int current_volume_frame = ui.timeline.getCurrentFrame();
        bool timeline_drives_volumes = false;
        for (const auto& gas : scene.gas_volumes) {
            if (gas && gas->isLinkedToTimeline()) {
                timeline_drives_volumes = true;
                break;
            }
        }
        if (!timeline_drives_volumes) {
            for (const auto& vdb : scene.vdb_volumes) {
                if (vdb && vdb->isAnimated() && vdb->isLinkedToTimeline()) {
                    timeline_drives_volumes = true;
                    break;
                }
            }
        }

        bool timeline_drives_water = timeline_has_water_keyframes;
        if (!timeline_drives_water) {
            for (const auto& surf : WaterManager::getInstance().getWaterSurfaces()) {
                const bool geometricMeshAnim = surf.params.use_geometric_waves && surf.animate_mesh;
                const bool fftMeshAnim = surf.params.use_fft_ocean && surf.params.use_fft_mesh_displacement;
                if (geometricMeshAnim || fftMeshAnim) {
                    timeline_drives_water = true;
                    break;
                }
            }
        }

        bool timeline_drives_wind = false;
        for (const auto& group : InstanceManager::getInstance().getGroups()) {
            if (group.wind_settings.enabled && !group.instances.empty()) {
                timeline_drives_wind = true;
                break;
            }
        }

        const bool timeline_has_runtime_work =
            timeline_has_manual_keyframes ||
            timeline_drives_volumes ||
            timeline_drives_water ||
            timeline_drives_wind ||
            !scene.animationDataList.empty();
        
        // =========================================================================
        // EARLY SCENE LOADING GUARD
        // Prevent any scene/backend mutation while loader thread is active.
        // =========================================================================
        if (ui.scene_loading.load()) {
            ImGui_ImplSDLRenderer2_NewFrame();
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();

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

                int progress = ui.scene_loading_progress.load();
                float progress_f = progress / 100.0f;
                char overlay[32];
                snprintf(overlay, sizeof(overlay), "%d%%", progress);
                ImGui::ProgressBar(progress_f, ImVec2(-1, 22), overlay);

                ImGui::Spacing();
                const std::string loading_stage = ui.getSceneLoadingStage();
                ImGui::TextColored(ImVec4(0.7f, 0.9f, 0.7f, 1.0f), "%s", loading_stage.c_str());
                ImGui::EndPopup();
            }

            SDL_SetRenderDrawColor(renderer, 20, 20, 25, 255);
            SDL_RenderClear(renderer);
            if (raytrace_texture) {
                SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
            }

            ImGui::Render();
            ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
            SDL_RenderPresent(renderer);

            SDL_Delay(10);
            continue;
        }

        if (current_volume_frame != last_volume_frame && !skip_backend_for_anim && timeline_drives_volumes) {
            bool volume_changed = false;
            
            // Update Gas Volumes
            for (auto& gas : scene.gas_volumes) {
                if (gas->isLinkedToTimeline()) {
                    void* stream = (g_hasOptix && g_backend) ? g_backend->getNativeCommandQueue() : nullptr;
                    gas->updateFromTimeline(current_volume_frame, stream);
                    volume_changed = true;
                }
            }
            
            // Update VDB Volumes
            for (auto& vdb : scene.vdb_volumes) {
                if (vdb->isAnimated() && vdb->isLinkedToTimeline()) {
                    void* stream = (g_hasOptix && g_backend) ? g_backend->getNativeCommandQueue() : nullptr;
                    vdb->updateFromTimeline(current_volume_frame, stream);
                    volume_changed = true;
                }
            }
            
            if (volume_changed) {
                // Sync to GPU
                ui.syncVDBVolumesToGPU(ui_ctx);
                ui_ctx.renderer.updateBackendGasVolumes(scene);
                
                // Break accumulation for new frame (OptiX or Vulkan)
                if (g_backend) g_backend->resetAccumulation();
                ray_renderer.resetCPUAccumulation();
                start_render = true;
            }
            last_volume_frame = current_volume_frame;
        }

        // Real-time Gas Simulation Step (sequential sim logic)
        bool gas_stepped = false;
        if (!skip_backend_for_anim) {
           for (auto& gas : scene.gas_volumes) {
               if (gas->getSettings().mode == FluidSim::SimulationMode::RealTime) {
                 if (gas->isPlaying() || (timeline_playing && gas->linked_to_timeline)) {
                     // Note: updateFromTimeline already handles catching up if linked.
                     // But if NOT linked, it only steps if isPlaying().
                     if (!gas->linked_to_timeline && gas->isPlaying()) {
                        void* stream = (g_hasOptix && g_backend) ? g_backend->getNativeCommandQueue() : nullptr;
                        gas->stepFrame(dt, stream);
                        gas_stepped = true;
                     }
                 }
             }
        }
        
        if (gas_stepped) {
            // CRITICAL: Sync both render paths (Legacy and Unified VDB)
            ui.syncVDBVolumesToGPU(ui_ctx); // Logic for Unified VDB path
            ui_ctx.renderer.updateBackendGasVolumes(scene); // Logic for Legacy path
            
            if (g_backend) g_backend->resetAccumulation(); // OptiX or Vulkan
            ray_renderer.resetCPUAccumulation();
            start_render = true;
        }

        if (timeline_playing && timeline_has_runtime_work) {
             // Use frame count from timeline (24 FPS assumption or settings)
             const float timeline_fps = static_cast<float>(std::max(1, render_settings.animation_fps));
             float time_seconds = current_volume_frame / timeline_fps;

             // Calculate wind transforms on CPU
            FoliageWindUpdateStats wind_stats;
            if (timeline_drives_wind) {
                wind_stats = InstanceManager::getInstance().updateWind(time_seconds, scene, g_backend.get());
            }

             // Update backend time only when something actually consumes timeline time.
             if (g_backend && (timeline_drives_water || timeline_drives_wind || !scene.animationDataList.empty())) {
                 g_backend->setTime(time_seconds, time_seconds);
             }

             if (timeline_drives_water) {
                 WaterUpdateResult water_update = WaterManager::getInstance().update(time_seconds);
                 if (water_update.material_changed || water_update.mesh_changed) {
                     for (auto& water : WaterManager::getInstance().getWaterSurfaces()) {
                         if (water.material_id != MaterialManager::INVALID_MATERIAL_ID) {
                             WaterManager::getInstance().syncSurfaceMaterial(&water);
                             ray_renderer.updateBackendMaterial(scene, water.material_id);
                         }
                     }
                 }
                 if (water_update.mesh_changed) {
                     const bool interactiveViewportActive =
                         isVulkanInteractiveViewportActive(
                             getRasterViewportBackend() != nullptr,
                             ui.viewport_settings.shading_mode);

                     if (g_backend && !interactiveViewportActive) {
                         if (auto* vulkanBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
                             bool partialUpdated = false;
                             bool needsFullRefresh = false;
                             for (const auto& water : WaterManager::getInstance().getWaterSurfaces()) {
                                 if (water.name.empty() || water.mesh_triangles.empty()) continue;
                                 const bool updated = vulkanBackend->updateMeshBLASPartial(
                                     water.name,
                                     water.mesh_triangles);
                                 partialUpdated = partialUpdated || updated;
                                 needsFullRefresh = needsFullRefresh || !updated;
                             }
                             if (needsFullRefresh || !partialUpdated) {
                                 vulkanBackend->rebuildAccelerationStructure();
                                 vulkanBackend->updateGeometry(scene.world.objects);
                             }
                         } else {
                             g_backend->updateSceneGeometry(scene.world.objects, ray_renderer.finalBoneMatrices);
                         }
                         g_backend->resetAccumulation();
                     }

                     g_cpu_bvh_refit_pending = true;
                     g_mesh_cache_dirty = true;
                     ray_renderer.resetCPUAccumulation();
                 }
             }
             
             // Efficiently update instance transforms on GPU (no full rebuild)
             if (timeline_drives_wind && g_backend && (wind_stats.any_cpu_update || wind_stats.gpu_deform_applied)) {
                 g_backend->updateInstanceTransforms(scene.world.objects);
             }
             
             // Force redraw only when playback has scene work to show.
             start_render = true;
             if (timeline_drives_water || timeline_drives_wind || !scene.animationDataList.empty()) {
                 g_needs_optix_sync.store(true, std::memory_order_release); // Signal that params changed
             }
             
        } else if (!timeline_playing) {
             // Static update (for editor changes)
             WaterManager::getInstance().update(0.0f);
        }
        } // end skip_backend_for_anim guard (gas + timeline)

        // =========================================================================
        // SCENE LOADING STATE (PROTECT MAIN THREAD)
        // =========================================================================
        if (ui.scene_loading.load()) {
            // 1. DRAIN EVENTS (Done above)
            
            // 2. START NEW FRAME
            ImGui_ImplSDLRenderer2_NewFrame();
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();

            // 3. DRAW LOADING POPUP
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
                
                int progress = ui.scene_loading_progress.load();
                float progress_f = progress / 100.0f;
                char overlay[32];
                snprintf(overlay, sizeof(overlay), "%d%%", progress);
                ImGui::ProgressBar(progress_f, ImVec2(-1, 22), overlay);
                
                ImGui::Spacing();
                const std::string loading_stage = ui.getSceneLoadingStage();
                ImGui::TextColored(ImVec4(0.7f, 0.9f, 0.7f, 1.0f), "%s", loading_stage.c_str());
                ImGui::EndPopup();
            }

            // 4. RENDER & PRESENT
            SDL_SetRenderDrawColor(renderer, 20, 20, 25, 255); 
            SDL_RenderClear(renderer);
            
            // Draw last rendered frame behind the loading progress
            if (raytrace_texture) {
                SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
            }
            
            ImGui::Render();
            ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
            SDL_RenderPresent(renderer);

            // 5. DO NOT HOG CPU - Give cycles to the loader thread
            SDL_Delay(10); 
            continue; // Skip the rest of the loop (Camera, Scene, UI Drawing)
        }
        // =========================================================================
        // SCENE LOADING COMPLETE (Finalize on Main Thread)
        // =========================================================================
        else if (ui.scene_loading_done.load()) {
            ui.scene_loading_done = false;
            
            // Refresh UI cache and start render
            ui.invalidateCache();

            // Prepare CPU-side picking data immediately after scene load so the
            // first Solid/CPU selection works without waiting for a transform edit.
            prepareCpuPickingState(scene, ui);
            
            // CPU BVH: defer to async path instead of blocking main thread.
            // The async BVH builder at g_bvh_rebuild_pending handles large scenes
            // without freezing the UI (copies objects, builds on a background thread).
            if (g_needs_geometry_rebuild.exchange(false, std::memory_order_acq_rel)) {
                g_bvh_rebuild_pending = true;
            }
            
            ui_ctx.renderer.resetCPUAccumulation();
            const bool deferRenderBackendWarmup = isInteractiveViewportShadingMode(ui.viewport_settings.shading_mode);
            // GPU build was deferred from create_scene — force a full backend sync here.
            // forceFullSync=true bypasses dirty-flag guards (flags are set below, after
            // this call), ensuring geometry/materials/lights/world are all uploaded.
            if (g_backend && !skip_backend_for_anim &&
                !ui_ctx.render_settings.backend_changed &&
                !deferRenderBackendWarmup) {
                g_needs_optix_sync.store(true, std::memory_order_release);
                (void)syncActiveRenderBackendScene(true);
            }
            
            // Raster viewport rebuild only needed when in Solid/Matcap mode.
            // In Rendered mode (OptiX/Vulkan RT), defer raster rebuild until user
            // actually switches to Solid/Matcap — avoids a synchronous
            // buildRasterGeometry stall on 500K+ foliage instances at load time.
            if (isInteractiveViewportShadingMode(ui.viewport_settings.shading_mode)) {
                g_viewport_raster_rebuild_pending = true;
                // In interactive viewport mode the render backend is not visible.
                // Defer its heavy sync until the user switches to Rendered mode.
                // The granular dirty flags (g_geometry_dirty etc.) are already set
                // below, so the first Solid→Rendered transition will pick them up.
                if (render_settings.use_optix || render_settings.use_vulkan) {
                    g_needs_optix_sync.store(true, std::memory_order_release);
                }
                // Do NOT set g_deferred_render_backend_prepare_pending here.
                // That path can cascade into backend_changed→full teardown/recreate
                // which is wasted work while the user is in Solid mode.
            } else if (deferRenderBackendWarmup && (render_settings.use_optix || render_settings.use_vulkan || g_backend)) {
                g_deferred_render_backend_prepare_pending.store(true, std::memory_order_release);
                g_deferred_render_backend_prepare_delay_frames = 2;
                if (render_settings.use_optix || render_settings.use_vulkan) {
                    g_needs_optix_sync.store(true, std::memory_order_release);
                }
            } else if (dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr) {
                g_vulkan_rebuild_pending = true;
            }

            // Mark all buffers dirty for fresh scene
            g_camera_dirty = true;
            g_lights_dirty = true;
            g_world_dirty = true;
            g_geometry_dirty = true;
            g_materials_dirty = true;
            g_gas_volumes_dirty = true;
            g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
            ui_ctx.start_render = true;
            SCENE_LOG_INFO("Project visualization ready.");
        }

        // =========================================================================
        // NORMAL UI & RENDER LOOP (Starts here)
        // =========================================================================
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        ui_ctx.ray_texture = raytrace_texture;
        
        ui_ctx.backend_ptr = getActiveViewportBackendForShading(ui.viewport_settings.shading_mode);
        // Solid/Matcap (not Rendered=2): fluid bridge renders a splat-sphere proxy
        // even for SurfaceSDF fluids, since the raster viewport can't draw the volume.
        g_solid_viewport_active = isInteractiveViewportShadingMode(ui.viewport_settings.shading_mode);
        ui.draw(ui_ctx);
        render_settings.stylize_enabled =
            ray_renderer.stylizeMode.enabled && ui.viewport_settings.shading_mode == 2;
        const bool viewport_transform_dragging = ui.is_dragging;

        if (viewport_transform_dragging) {
            start_render = true;
        }
        
        // ===========================================================================
        // CENTRALIZED CAMERA UPDATE (PHASE 1)
        // Checks if UI changed camera or if continuous effects (Shake/AF-C) are active
        //
        // [RENDER-LOCK RACE FIX] During an active sequence render the worker
        // thread owns scene.camera — Renderer::updateAnimationState writes
        // lookfrom/lookat/vfov/focus/lens_radius and calls update_camera_vectors
        // per animation frame (Renderer.cpp ~line 2336), then syncCameraToBackend
        // uploads the result. If the main thread also calls update_camera_vectors
        // here (e.g. AF-C mode keeps is_af_c=true and forces the block every
        // frame), the two threads mutate the same Camera u/v/w basis vectors
        // concurrently — the GPU reads a torn camera and traceRays hangs inside
        // the driver. The skip_backend_for_anim gate on syncCameraToBackend was
        // not enough; the mutation at update_camera_vectors() above the gate is
        // the actual race source.
        // ===========================================================================
        const bool anim_owns_camera = rendering_in_progress.load() && ui_ctx.is_animation_mode;
        if (scene.camera && !anim_owns_camera) {
            bool is_dirty = scene.camera->checkDirty();

            // Continuous effects require per-frame updates
            bool is_shaking = scene.camera->enable_camera_shake;
            bool is_af_c = (ui.viewport_settings.focus_mode == 2);

            if (is_dirty || is_shaking || is_af_c) {
                 // Ensure vectors are up to date
                 scene.camera->update_camera_vectors();
                 ray_renderer.world.setCameraY(scene.camera->lookfrom.y);

                    if (ui_ctx.backend_ptr && !skip_backend_for_anim) {
                        ui_ctx.renderer.syncCameraToBackend(*scene.camera);
                    }
                 ray_renderer.resetCPUAccumulation();
                 g_camera_dirty = false;

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
            if (g_hasOptix && !skip_backend_for_anim) {
                if (g_backend) g_backend->resetAccumulation();
            }
            
            // Reset render state
            render_settings.is_rendering_active = false;
            render_settings.render_current_samples = 0;
            render_settings.render_progress = 0.0f;
            render_settings.render_elapsed_seconds = 0.0f;
            render_settings.render_estimated_remaining = 0.0f;
            render_settings.avg_sample_time_ms = 0.0f;
            render_settings.avg_total_frame_time_ms = 0.0f;
            render_settings.avg_total_frame_fps = 0.0f;
            
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

            // Notify viewport raster backend about the new camera aspect ratio and
            // trigger a raster rebuild so Solid/Matcap mode updates without needing
            // a camera move first.
            g_camera_dirty = true;
            g_viewport_raster_rebuild_pending = true;

            SCENE_LOG_INFO("Resolution changed to " + std::to_string(image_width) + "x" + std::to_string(image_height));
            
            // Skip to next loop iteration - render starts next frame
            ImGui::Render();
            ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
            SDL_RenderPresent(renderer);
            continue;  // Skip rest of loop
        }
        const Uint8* key_state = SDL_GetKeyboardState(NULL);
        
        // Re-check input lock for keyboard controls (input_locked was inside event loop scope)
        bool keyboard_locked = ui_ctx.render_settings.animation_render_locked || 
                              (rendering_in_progress && ui_ctx.is_animation_mode);

        if (mouse_control_enabled && scene.camera && !keyboard_locked && !ImGui::GetIO().WantCaptureKeyboard) {

            Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
            Vec3 right = Vec3::cross(forward, scene.camera->vup).normalize();
            Vec3 up = scene.camera->vup;

            // Arrow Keys + WASDQE for Camera Movement
            const Uint8* s = SDL_GetKeyboardState(NULL);
            bool is_shift = s[SDL_SCANCODE_LSHIFT] || s[SDL_SCANCODE_RSHIFT];
            float boost = is_shift ? (4.0f + render_settings.mouse_sensitivity * 6.0f) : 1.0f;
            float effective_speed = move_speed * (render_settings.mouse_sensitivity * 5.0f) * boost;

            if (key_state[SDL_SCANCODE_UP] || key_state[SDL_SCANCODE_W]) {
                scene.camera->lookfrom += forward * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_DOWN] || key_state[SDL_SCANCODE_S]) {
                scene.camera->lookfrom -= forward * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_LEFT] || key_state[SDL_SCANCODE_A]) {
                scene.camera->lookfrom -= right * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_RIGHT] || key_state[SDL_SCANCODE_D]) {
                scene.camera->lookfrom += right * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_PAGEUP] || key_state[SDL_SCANCODE_E]) {
                scene.camera->lookfrom += up * effective_speed;
                camera_moved = true;
            }
            if (key_state[SDL_SCANCODE_PAGEDOWN] || key_state[SDL_SCANCODE_Q]) {
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
            // hareket durmu�sa foveation seviyesi b�y�t
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
                else if (([&]() -> bool {
                    // [SOLID-MODE FIX] Switch to Rendered viewport mode FIRST and
                    // defer the actual animation launch by a few frames. The
                    // solid→rendered transition block (~line 3760 below) needs to
                    // run rebuildAccelerationStructure + updateGeometry +
                    // uploadHairToGPU + materials/world sync on the RT backend
                    // before the animation worker can safely call
                    // updateSceneGeometry — otherwise BLASes don't exist yet and
                    // createTLAS crashes inside the NVIDIA driver.
                    //
                    // [FRAME-0 POSE FIX] Also pre-pin the timeline to the
                    // sequence's start frame here. SceneUI::processAnimations
                    // reads scene.timeline.current_frame each UI frame and
                    // applies keyframes via mesh_cache — which correctly
                    // handles HittableInstance wrappers (calls
                    // cached.instance->setTransform after setting the
                    // underlying triangle's TransformHandle). The animation
                    // worker's updateAnimationState only iterates raw
                    // Triangle objects in scene.world.objects, so
                    // HittableInstance wrappers (default scene cube et al)
                    // are NOT updated by the worker on frame 0 — they keep
                    // whatever transform they had when the user clicked
                    // Render. By pinning the timeline to the start frame
                    // BEFORE the worker spawns, processAnimations evaluates
                    // and applies the start-frame pose to every object
                    // (HittableInstance included) during the countdown
                    // window. By the time the worker takes over, the scene
                    // is already at the correct pose and Vulkan's frame 0
                    // matches OptiX.
                    static int s_anim_mode_switch_countdown = 0;
                    static bool s_anim_timeline_pinned = false;
                    const int pin_frame = ui_ctx.render_settings.animation_start_frame;
                    auto pinTimelineToStart = [&]() {
                        // Pin BOTH scene.timeline AND TimelineWidget's internal
                        // current_frame. The widget's draw() writes its own
                        // current_frame back into scene.timeline.current_frame
                        // every UI iteration (line ~348), which would otherwise
                        // immediately overwrite our pin with the user's pre-render
                        // scrub position.
                        scene.timeline.current_frame = pin_frame;
                        ui.timeline.setCurrentFrame(pin_frame);
                    };
                    if (ui.viewport_settings.shading_mode != 2) {
                        SCENE_LOG_INFO("[Anim] Auto-switching viewport from shading_mode=" +
                                       std::to_string(ui.viewport_settings.shading_mode) +
                                       " to Rendered (2); deferring sequence launch until RT rebuild settles.");
                        ui.viewport_settings.shading_mode = 2;
                        s_anim_mode_switch_countdown = 4; // wait ~4 frames for transition block + rebuild
                        pinTimelineToStart();
                        s_anim_timeline_pinned = true;
                        return true; // hold off launch this frame
                    }
                    // Already in Rendered mode — still need to pin the timeline
                    // to the sequence start frame and give processAnimations one
                    // UI frame to apply the start-frame pose before the worker
                    // takes over. Otherwise frame 0's TLAS uses whatever pose the
                    // scene happened to be in (last user scrub / playback frame),
                    // because the worker's updateAnimationState misses
                    // HittableInstance wrappers.
                    if (!s_anim_timeline_pinned &&
                        scene.timeline.current_frame != pin_frame) {
                        SCENE_LOG_INFO("[Anim] Pinning timeline to start frame " +
                                       std::to_string(pin_frame) +
                                       " before launch; waiting one frame for processAnimations to settle.");
                        pinTimelineToStart();
                        s_anim_timeline_pinned = true;
                        s_anim_mode_switch_countdown = std::max(s_anim_mode_switch_countdown, 2);
                        return true; // wait for processAnimations to apply
                    }
                    // Re-pin every wait iteration so TimelineWidget's per-frame
                    // sync-back can't drift us back to the user's scrub position.
                    if (s_anim_mode_switch_countdown > 0) {
                        pinTimelineToStart();
                    }
                    if (s_anim_mode_switch_countdown > 0) {
                        s_anim_mode_switch_countdown--;
                        if (s_anim_mode_switch_countdown == 0) {
                            SCENE_LOG_INFO("[Anim] Mode-switch settle complete, sequence will launch next frame.");
                        }
                        return true; // still waiting
                    }
                    // Reset pin flag for next sequence run.
                    s_anim_timeline_pinned = false;
                    return false; // good to launch
                })()) {
                    // Hold start_animation_render set; re-enter next frame.
                }
                else {
                    // Start Animation
                    ui_ctx.render_settings.start_animation_render = false;
                    render_settings.start_animation_render = false;
                    rendering_in_progress = true; // Set block flag immediately
                    ui_ctx.is_animation_mode = true; // Enable animation mode for UI status bar
                    
                    // Reset stop flags for new render
                    rendering_stopped_cpu = false;
                    rendering_stopped_gpu = false;
                    
                    start_render = false; // Cancel any pending interactive render
                    
                    SCENE_LOG_INFO("Starting animation render...");
                    std::string output_folder = ui_ctx.render_settings.animation_output_folder;
                    if (output_folder.empty()) output_folder = "render_animation";
                    SCENE_LOG_INFO("Output folder set to: " + output_folder);

                    // Capture local copies of settings to avoid thread race
                    // Sequence render uses the SAME sampling controls as the
                    // interactive viewport: max_samples as the per-frame cap,
                    // plus the adaptive sampling threshold + min_samples. The
                    // old "final_render_samples" / "animation_samples_per_frame"
                    // duplicates have been retired — one set of sliders drives
                    // viewport, single-frame final, and sequence.
                    int anim_sample_count = ui_ctx.render_settings.max_samples;
                    if (anim_sample_count <= 0) anim_sample_count = 128; // Fallback
                    // Keep legacy fields in sync so any save-file / progress UI
                    // that still references them shows the active value.
                    ui_ctx.render_settings.animation_samples_per_frame = anim_sample_count;
                    ui_ctx.render_settings.final_render_samples = anim_sample_count;
                    int anim_sample_per_pass = sample_per_pass;
                    int anim_fps = ui_ctx.render_settings.animation_fps;
                    float anim_duration = ui_ctx.render_settings.animation_duration;
                    bool anim_use_denoiser = ui_ctx.render_settings.use_denoiser;
                    float anim_denoiser_blend = ui_ctx.render_settings.denoiser_blend_factor;
                    bool anim_use_gpu = isActiveRenderBackendGpu();
                    int anim_start_frame = ui_ctx.render_settings.animation_start_frame;
                    int anim_end_frame = ui_ctx.render_settings.animation_end_frame;
                    
                    // VALIDATION: Ensure valid frame range
                    if (anim_end_frame < anim_start_frame) {
                        anim_end_frame = anim_start_frame; // At least 1 frame
                    }
                    if (anim_end_frame == 0 && anim_start_frame == 0) {
                        // Both zero - check if scene has animation data
                        if (!scene.animationDataList.empty() && scene.animationDataList[0]) {
                            anim_start_frame = scene.animationDataList[0]->startFrame;
                            anim_end_frame = scene.animationDataList[0]->endFrame;
                            SCENE_LOG_INFO("Frame range auto-detected from animation file: " + std::to_string(anim_start_frame) + " - " + std::to_string(anim_end_frame));
                        } else {
                            anim_end_frame = 100; // Default fallback
                            SCENE_LOG_WARN("No frame range set, using default 0-100");
                        }
                    }
                    
                    // Update UI with validated values for progress display
                    ui_ctx.render_settings.animation_start_frame = anim_start_frame;
                    ui_ctx.render_settings.animation_end_frame = anim_end_frame;
                    ui_ctx.render_settings.animation_total_frames = anim_end_frame - anim_start_frame + 1;
                    
                    SCENE_LOG_INFO("Animation render: Frames " + std::to_string(anim_start_frame) + " - " + std::to_string(anim_end_frame) + " (" + std::to_string(ui_ctx.render_settings.animation_total_frames) + " total) @ " + std::to_string(anim_sample_count) + " samples/frame");

                    // [SIM PRE-BAKE] If the scene has a live simulation, bake it to
                    // the sequence START frame on the MAIN thread, BEFORE the backend
                    // sync below, and force that sync to be a FULL rebuild. The
                    // worker would otherwise discover a structural mismatch on its
                    // very first frame (the viewport was parked on a DIFFERENT frame,
                    // so the InstanceManager particle/foam pools + SurfaceSDF volume
                    // were sized for that frame) and fire a destructive Vulkan
                    // rebuildAccelerationStructure — which tears down every BLAS and
                    // sets m_rtPipelineReady=false — at the exact moment the viewport→
                    // render transition is still settling, so the RT geometry briefly
                    // (sometimes persistently) vanishes. Doing the structural rebuild
                    // here, where the main thread is the sole backend owner, removes
                    // the race; the worker then finds start_frame already cached
                    // (restoreSimFrame hit) and its first frame is a no-op refit.
                    bool sim_prebaked = false;
                    if (anim_use_gpu && g_backend && scene.anySimulationRuntimeEnabled()) {
                        scene.bakeSimulationForRenderFrame(
                            anim_start_frame,
                            static_cast<float>(std::max(1, anim_fps)),
                            /*enable_rt_geometry*/ true);
                        sim_prebaked = true;
                    }

                    // [BACKEND-STATE FIX] Force a FULL rebuild only when we have a
                    // concrete reason — either an aborted previous animation left
                    // partial TLAS/BLAS state, or a backend swap left dirty flags
                    // cleared. Forcing a full rebuild unconditionally was destroying
                    // a healthy already-built RT backend (Vulkan's rebuild is a
                    // pending operation, and the worker would launch before it
                    // finished → stuck at 1% with vanished geometry).
                    //
                    // For all other cases (user simply pressed Render Sequence
                    // while in Rendered mode), the granular sync path below
                    // handles the normal dirty-flag-driven updates without
                    // touching healthy state.
                    if (anim_use_gpu && g_backend) {
                        const bool needs_full =
                            sim_prebaked || g_anim_backend_needs_full_rebuild.exchange(false);
                        if (needs_full) {
                            SCENE_LOG_INFO("[Anim] Forcing full RT backend rebuild (sim pre-bake / previous animation aborted or backend was swapped).");
                            (void)syncActiveRenderBackendScene(/*forceFullSync=*/true);
                        } else {
                            (void)syncActiveRenderBackendScene(/*forceFullSync=*/false);
                        }
                        // The pre-bake + full sync above already pushed the start-frame
                        // sim geometry to the backend. Clear the bridge's rebuild-pending
                        // flags so the worker's first frame doesn't redundantly tear the
                        // freshly-built AS down again.
                        if (sim_prebaked) {
                            extern bool g_optix_rebuild_pending;
                            extern bool g_vulkan_rebuild_pending;
                            extern bool g_gpu_refit_pending;
                            g_optix_rebuild_pending = false;
                            g_vulkan_rebuild_pending = false;
                            g_gpu_refit_pending = false;
                        }
                    }

                    // [VIEWPORT-DRIVEN SEQUENCE SAVE] Instead of the separate
                    // render_Animation worker (which ran slower than and diverged
                    // from the interactive viewport — foam/TLAS/rebuild differences),
                    // drive the sequence through the SAME fast interactive viewport
                    // render. The main-loop state machine (g_seq_save_*) scrubs each
                    // frame, lets it converge to the interactive render-panel quality
                    // (max_samples + adaptive noise threshold), saves it, then
                    // advances. Quality + denoiser come from the interactive settings.
                    g_seq_save_active = true;
                    g_seq_save_frame = anim_start_frame;
                    g_seq_save_end = anim_end_frame;
                    g_seq_save_dir = output_folder;
                    g_seq_save_denoise = anim_use_denoiser;

                    // Keep BOTH rendering_in_progress and is_animation_mode true so
                    // the existing UI shows the render status + the "Stop Anim"
                    // button (no new UI needed). The worker-path gates that normally
                    // key off (rendering_in_progress && is_animation_mode) —
                    // skip_backend_for_anim, render_owns_sim, render_owns_timeline,
                    // and the interactive-render block — are each additionally guarded
                    // with `&& !g_seq_save_active`, so during a viewport-driven save
                    // the backend stays live and the UI keeps driving the per-frame
                    // sim/animation scrub.
                    rendering_in_progress = true;
                    ui_ctx.is_animation_mode = true;
                    ui_ctx.render_settings.animation_render_locked = true;
                    rendering_stopped_gpu = false;
                    rendering_stopped_cpu = false;

                    // Timeline is already pinned to the start frame (pinTimelineToStart
                    // above). Mirror it into the progress fields and reset accumulation
                    // so frame one renders clean.
                    ui.timeline.setCurrentFrame(anim_start_frame);
                    scene.timeline.current_frame = anim_start_frame;
                    render_settings.animation_current_frame = anim_start_frame;
                    ui_ctx.render_settings.animation_current_frame = anim_start_frame;
                    if (g_backend) g_backend->resetAccumulation();
                    start_render = true;
                    SCENE_LOG_INFO("[SeqSave] Viewport-driven sequence save started: frames " +
                                   std::to_string(anim_start_frame) + " - " +
                                   std::to_string(anim_end_frame) + " -> " + output_folder);
                }
            }

           
            // 2. Handle Interactive Render (One Frame / Progressive)
            // ONLY if no background rendering is happening
            if (start_render) {
                 // CRITICAL INVARIANT:
                 // rendering_in_progress may remain true in some non-animation GPU flows.
                 // Only block interactive rendering while a WORKER animation render is
                 // active. A viewport-driven sequence save (g_seq_save_active) renders
                 // THROUGH this exact path, so it must NOT be blocked here.
                 if (rendering_in_progress && ui_ctx.is_animation_mode && !g_seq_save_active) {
                     // Block interactive render only while animation render thread is active
                     start_render = false;
                 }
                 else {
                     // Safe to start new render
                     did_render_this_frame = true;
                     start_render = false;

                     // Reset stop flags for new render — but NOT during a viewport
                     // driven sequence save: its "Stop Anim" sets these and the
                     // sequence state machine consumes them to cancel; resetting here
                     // every render iteration would swallow the cancel.
                     if (!g_seq_save_active) {
                         rendering_stopped_cpu = false;
                         rendering_stopped_gpu = false;
                     }// --- Animation State Update ---
                    float fps = ui_ctx.render_settings.animation_fps;
                    if (fps <= 0.0f) fps = 24.0f;
                    int start_f = ui_ctx.render_settings.animation_start_frame;
                    int current_f = ui_ctx.render_settings.animation_playback_frame;
                    float time = (current_f - start_f) / fps;

                    static bool last_vulkan_state = false;
                    if (ui_ctx.render_settings.use_vulkan != last_vulkan_state) {
                        SCENE_LOG_INFO(std::string("[DEBUG] Backend Viewport: use_vulkan=") + (ui_ctx.render_settings.use_vulkan ? "TRUE" : "FALSE") + 
                                       " use_optix=" + (ui_ctx.render_settings.use_optix ? "TRUE" : "FALSE") + 
                                       " hasOptix=" + (g_hasOptix ? "TRUE" : "FALSE"));
                        last_vulkan_state = ui_ctx.render_settings.use_vulkan;
                    }

                    // Allow GPU render path when:
                    // 1. A GPU backend is actually active, OR
                    // 2. Viewport is in Solid/Matcap mode and Vulkan raster backend is active
                    Backend::IBackend* activeViewportBackend = getActiveViewportBackendForShading(ui.viewport_settings.shading_mode);
                    const bool backendIsOptix = (dynamic_cast<Backend::OptixBackend*>(g_backend.get()) != nullptr);
                    const bool backendIsVulkan = (dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr);
                    auto* activeViewportRasterBackend = dynamic_cast<Backend::IViewportBackend*>(activeViewportBackend);
                    const bool activeViewportHasRasterBackend = (activeViewportRasterBackend != nullptr);
                    const Backend::ViewportMode requestedViewportMode =
                        viewportModeFromShadingMode(ui.viewport_settings.shading_mode);
                    const bool backendSupportsRequestedViewport =
                        activeViewportBackend && activeViewportBackend->supportsViewportMode(requestedViewportMode);
                    const bool vulkanRasterActive = isVulkanInteractiveViewportActive(
                        activeViewportHasRasterBackend,
                        ui.viewport_settings.shading_mode);
                    const bool pendingBackendSceneSync =
                        g_geometry_dirty ||
                        g_materials_dirty ||
                        g_texture_pool_dirty ||
                        g_gas_volumes_dirty ||
                        g_world_dirty ||
                        g_lights_dirty ||
                        g_camera_dirty;
                    // g_needs_optix_sync is also used by lightweight runtime updates
                    // (timeline/water/wind). Only upgrade it to a full backend scene
                    // sync when real scene dirty flags are still pending.
                    if (g_backend &&
                        g_needs_optix_sync.load(std::memory_order_acquire) &&
                        pendingBackendSceneSync &&
                        !render_settings.backend_changed &&
                        !ui_ctx.render_settings.backend_changed &&
                        (backendIsOptix || backendIsVulkan) &&
                        activeViewportBackend == g_backend.get()) {
                        (void)syncActiveRenderBackendScene();
                    }
                    if (ui.viewport_settings.shading_mode != 2 &&
                        !vulkanRasterActive &&
                        !backendSupportsRequestedViewport) {
                        static uint32_t lastUnsupportedViewportWarnMs = 0;
                        const uint32_t nowMs = SDL_GetTicks();
                        ui.viewport_settings.shading_mode = 2;
                        if (nowMs - lastUnsupportedViewportWarnMs > 3000) {
                            ui.addViewportMessage(
                                "Selected viewport mode is not supported by the active backend. Switched to Rendered mode.",
                                3.0f,
                                ImVec4(1.0f, 0.75f, 0.25f, 1.0f));
                            lastUnsupportedViewportWarnMs = nowMs;
                        }
                    }
                    if (backendIsOptix || backendIsVulkan || vulkanRasterActive) {
                        if (render_settings.backend_changed || ui_ctx.render_settings.backend_changed) {
                            SDL_Delay(1);
                            continue;
                        }
                        if (g_backend_switch_cooldown_frames > 0) {
                            --g_backend_switch_cooldown_frames;
                            SDL_Delay(1);
                            continue;
                        }
                        // ============ SYNCHRONOUS GPU RENDER (No Thread) ============
                        // Each pass is ~10-50ms for 1 sample, fast enough for UI
                        
                        // [FIX] Skip ALL GPU state updates and rendering when animation render
                        // thread is using the same backend — concurrent Vulkan/GPU access causes
                        // contention, freezes, and T-pose artifacts.
                        // !g_seq_save_active: a viewport-driven sequence save keeps
                        // both anim flags set for the UI but renders THROUGH this very
                        // viewport path — it must NOT be treated as a worker owning
                        // the backend (that skipped the render → CPU spin, no output).
                        const bool anim_owns_backend = rendering_in_progress && ui_ctx.is_animation_mode
                            && !g_seq_save_active
                            && g_backend && (activeViewportBackend == g_backend.get());
                        if (anim_owns_backend) {
                            // Animation thread owns the GPU backend — skip viewport render this frame.
                            // UI stays responsive, animation preview is updated via animation_preview_buffer.
                            SDL_Delay(1);
                        }
                        else {

                        bool geometry_updated = false;
                        const bool has_file_animations = !scene.animationDataList.empty();
                        bool force_bind_pose = (ui.show_hair_tab && ui.active_properties_tab == 8);
                        const bool should_update_animation = has_file_animations ||
                            autonomous_anim_graph_playing ||
                            force_bind_pose;
                        if (should_update_animation) {
                            geometry_updated = ray_renderer.updateAnimationState(scene, time, false, force_bind_pose);
                        }

                        // WIND ANIMATION (Independent of FBX animations)
                        // Checks if any InstanceGroup has wind enabled and updates matrices
                        static float last_wind_time = -1.0f;
                        

                        if ((should_update_animation || (timeline_playing && timeline_drives_wind)) &&
                            std::abs(time - last_wind_time) > 0.001f) {
                            FoliageWindUpdateStats wind_stats = ray_renderer.updateWind(scene, time);
                            wind_active = wind_stats.any_cpu_update || wind_stats.gpu_deform_applied;
                            
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
                            if (activeViewportBackend && activeViewportBackend == g_backend.get()) {
                                // Pass Calculated Bone Matrices for GPU Skinning
                                activeViewportBackend->updateSceneGeometry(scene.world.objects, ray_renderer.finalBoneMatrices);
                            }
                            // Solid/Matcap mode: update raster viewport skinning directly
                            // (don't use g_vulkan_rebuild_pending — that calls buildRasterGeometry
                            //  which uploads base-pose vertices and destroys skinning state)
                            if (activeViewportHasRasterBackend && activeViewportBackend != g_backend.get()) {
                                if (activeViewportRasterBackend) {
                                    activeViewportRasterBackend->syncRasterInstanceTransforms(scene.world.objects);
                                    if (!ray_renderer.finalBoneMatrices.empty()) {
                                        activeViewportRasterBackend->syncRasterSkinnedVertices(
                                            scene.world.objects, ray_renderer.finalBoneMatrices);
                                    }
                                }
                            }

                            
                            // Geometry change implies all buffers need refresh
                            g_camera_dirty = true;
                            g_lights_dirty = true;
                            g_world_dirty = true;
                        }

                        // OPTIMIZATION: Only update GPU buffers when data has changed
                        // [CAMERA SYNC FIX] Force sync every frame if camera shake is enabled to allow real-time animation
                        bool needs_camera_sync = g_camera_dirty || (scene.camera && (scene.camera->enable_camera_shake));
                        if (needs_camera_sync && scene.camera) {
                            if (activeViewportBackend) {
                                activeViewportBackend->syncCamera(*scene.camera);
                                // Vulkan already resets in setCamera(), but OptiX needs explicit reset
                                // This ensures GPU accumulation buffer is cleared when camera/exposure changes
                                activeViewportBackend->resetAccumulation();
                            }
                            Vec3 dir = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                            yaw = atan2f(dir.z, dir.x) * 180.0f / 3.14159265f;
                            pitch = asinf(dir.y) * 180.0f / 3.14159265f;
                            
                            // Set camera Y for volumetric cloud parallax
                            ray_renderer.world.setCameraY(scene.camera->lookfrom.y);
                            g_camera_dirty = false;
                        }
                        // Always call setLights when lights are dirty, even if the list
                        // is empty. This allows backends to clear GPU-side light state
                        // (e.g., the Nishita sun stored in WorldData) when the user
                        // deletes the last directional light.
                        if (g_lights_dirty) {
                            if (activeViewportBackend) {
                                activeViewportBackend->setLights(scene.lights);
                            }
                            g_lights_dirty = false;
                        }

                        if (g_gas_volumes_dirty) {
                            if (activeViewportBackend && activeViewportBackend == g_backend.get()) {
                                ray_renderer.updateBackendGasVolumes(scene);
                                activeViewportBackend->resetAccumulation();
                            }
                            g_gas_volumes_dirty = false;
                        }

                        static int last_timeline_lut_update_frame = std::numeric_limits<int>::min();
                        const int timeline_lut_interval_frames =
                            std::max(1, std::max(1, render_settings.animation_fps) / 6);
                        const bool nishitaWorldActive =
                            ray_renderer.world.getMode() == WORLD_MODE_NISHITA;
                        const bool allowTimelineLUTUpdate =
                            nishitaWorldActive &&
                            timeline_playing &&
                            ray_renderer.world.needsLUTUpdate() &&
                            std::abs(ui.timeline.getCurrentFrame() - last_timeline_lut_update_frame) >= timeline_lut_interval_frames;
                        const bool allowImmediateVulkanLUTUpdate =
                            nishitaWorldActive &&
                            timeline_playing &&
                            ray_renderer.world.needsLUTUpdate() &&
                            dynamic_cast<Backend::VulkanBackendAdapter*>(activeViewportBackend) != nullptr;

                        if (g_world_dirty ||
                            (nishitaWorldActive && !timeline_playing && ray_renderer.world.needsLUTUpdate()) ||
                            allowTimelineLUTUpdate ||
                            allowImmediateVulkanLUTUpdate) {
                            if (activeViewportBackend) {
                                const bool hadPendingLUT = ray_renderer.world.needsLUTUpdate();
                                syncWorldDataToBackend(activeViewportBackend);
                                if (hadPendingLUT && !ray_renderer.world.needsLUTUpdate() && timeline_playing) {
                                    last_timeline_lut_update_frame = ui.timeline.getCurrentFrame();
                                }
                            }
                            g_world_dirty = false;
                        }


                        static std::vector<uint32_t> framebuffer;
                        if (framebuffer.size() != (size_t)image_width * image_height) {
                            framebuffer.resize((size_t)image_width * image_height);
                        }
                        
                        // Single pass render (1 sample) - fast, no UI blocking
                        auto sample_start = std::chrono::high_resolution_clock::now();
                        if (activeViewportBackend) {
                            static Backend::IBackend* last_backend = nullptr;
                            static int last_w = -1, last_h = -1, last_max = -1, last_min = -1, last_bounces = -1, last_diffuse_bounces = -1, last_transmission_bounces = -1;
                            static bool last_adaptive = false;
                            static float last_threshold = -1.0f;
                            int current_max = render_settings.is_final_render_mode ? render_settings.final_render_samples : render_settings.max_samples;

                            if (activeViewportBackend != last_backend ||
                                image_width != last_w || image_height != last_h || 
                                current_max != last_max || 
                                render_settings.min_samples != last_min ||
                                render_settings.max_bounces != last_bounces ||
                                render_settings.diffuse_bounces != last_diffuse_bounces ||
                                render_settings.transmission_bounces != last_transmission_bounces ||
                                render_settings.use_adaptive_sampling != last_adaptive ||
                                std::abs(render_settings.variance_threshold - last_threshold) > 0.0001f) 
                            {
                                Backend::RenderParams rp = {};
                                rp.imageWidth = image_width;
                                rp.imageHeight = image_height;
                                rp.samplesPerPixel = current_max;
                                rp.minSamples = render_settings.min_samples;
                                rp.maxBounces = std::max(1, render_settings.max_bounces);
                                rp.diffuseBounces = std::clamp(render_settings.diffuse_bounces, 1, rp.maxBounces);
                                rp.transmissionBounces = std::clamp(render_settings.transmission_bounces, 1, rp.maxBounces);
                                rp.useAdaptiveSampling = render_settings.use_adaptive_sampling;
                                rp.adaptiveThreshold = render_settings.variance_threshold;
                                activeViewportBackend->setRenderParams(rp);

                                last_backend = activeViewportBackend;
                                last_w = image_width;
                                last_h = image_height;
                                last_max = current_max;
                                last_min = render_settings.min_samples;
                                last_bounces = render_settings.max_bounces;
                                last_diffuse_bounces = render_settings.diffuse_bounces;
                                last_transmission_bounces = render_settings.transmission_bounces;
                                last_adaptive = render_settings.use_adaptive_sampling;
                                last_threshold = render_settings.variance_threshold;
                            }
                        }
                        // Enter GPU render block when selected engine is GPU OR when
                        // Vulkan raster pipeline is active for Solid/Matcap viewport
                        if (backendIsOptix || backendIsVulkan || vulkanRasterActive) {
                             static bool logged_gpu = false;
                             if (!logged_gpu) { SCENE_LOG_INFO("[DEBUG] Entering GPU Render block"); logged_gpu = true; }
                            // Check if we are playing animation
                            int loop_count = 1;
                            const Backend::ViewportMode viewportMode =
                                viewportModeFromShadingMode(ui.viewport_settings.shading_mode);
                            // Track viewport mode globally so solid→rendered transition is
                            // detected even when activeViewportBackend changes between modes
                            // (e.g. Vulkan raster in Solid → OptiX in Rendered).
                            static Backend::ViewportMode s_prevGlobalViewportMode = Backend::ViewportMode::Rendered;
                            // The active backend runs its own side effects (destroyInteractive,
                            // topology_dirty, resetAccumulation). Its setViewportMode body also
                            // propagates the value to Core::RenderStateManager so inactive
                            // backends can observe the authoritative mode passively.
                            if (activeViewportBackend) {
                                activeViewportBackend->setViewportMode(viewportMode);
                            }
                            // Switching solid/matcap → rendered: trigger rebuild for the
                            // active render backend so updated transforms are synced.
                            if (s_prevGlobalViewportMode != Backend::ViewportMode::Rendered &&
                                viewportMode == Backend::ViewportMode::Rendered) {
                                // [FIX] Clear shared framebuffer so Solid-mode grey pixels
                                // don't leak into the first Rendered frames while RT pipeline
                                // is still being built (lazy-init window).
                                if (!framebuffer.empty()) {
                                    std::fill(framebuffer.begin(), framebuffer.end(), 0u);
                                }
                                if (original_surface && original_surface->pixels) {
                                    SDL_FillRect(original_surface, nullptr, 0);
                                }
                                // Track whether scene geometry actually changed while in Solid mode.
                                // If unchanged, skip the expensive full rebuild — just sync lightweight state.
                                static uint64_t s_lastRenderedGeometryGen = 0;
                                const uint64_t currentGen = g_scene_geometry_generation.load(std::memory_order_acquire);
                                const bool geometryChangedSinceSolid = (currentGen != s_lastRenderedGeometryGen)
                                    || g_geometry_dirty || g_materials_dirty || g_texture_pool_dirty || g_gas_volumes_dirty;

                                auto* vkRenderBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get());
                                if (vkRenderBackend != nullptr) {
                                    if (geometryChangedSinceSolid) {
                                        // Geometry changed while in Solid mode — full sync needed
                                        vkRenderBackend->rebuildAccelerationStructure();
                                        vkRenderBackend->updateGeometry(scene.world.objects);
                                        ui.syncVDBVolumesToGPU(ui_ctx);
                                        ray_renderer.updateBackendGasVolumes(scene);
                                        ray_renderer.uploadHairToGPU();
                                        ray_renderer.updateBackendMaterials(scene);
                                        syncMaterialBufferToViewportBackend(scene, ray_renderer);
                                        g_geometry_dirty = false;
                                        g_materials_dirty = false;
                                        g_gas_volumes_dirty = false;
                                        g_texture_pool_dirty = false;
                                    }
                                    // Always sync lightweight state (camera/lights/world may have changed)
                                    if (scene.camera) {
                                        ray_renderer.syncCameraToBackend(*scene.camera);
                                    }
                                    vkRenderBackend->setLights(scene.lights);
                                    auto wd = ray_renderer.world.getGPUData();
                                    syncVulkanWorldWithAtmosphere(vkRenderBackend, wd);
                                    vkRenderBackend->resetAccumulation();
                                    g_vulkan_rebuild_pending = false;
                                    g_camera_dirty = true;
                                    g_lights_dirty = true;
                                    g_world_dirty = true;
                                } else if (dynamic_cast<Backend::OptixBackend*>(g_backend.get()) != nullptr) {
                                    if (geometryChangedSinceSolid) {
                                        g_optix_rebuild_pending = true;
                                        g_geometry_dirty = false;
                                        g_gas_volumes_dirty = false;
                                        // Preserve pending material uploads so the post-rebuild
                                        // g_needs_optix_sync pass refreshes OptiX texture handles.
                                    } else if (g_needs_optix_sync.load(std::memory_order_acquire)) {
                                        (void)syncActiveRenderBackendScene();
                                    } else {
                                        // Only lightweight sync needed
                                        g_backend->resetAccumulation();
                                        g_camera_dirty = true;
                                        g_lights_dirty = true;
                                        g_world_dirty = true;
                                    }
                                } else {
                                    // CPU mode (no GPU render backend): synchronous vertex sync
                                    // so the first CPU frame renders with up-to-date geometry.
                                    extern bool g_cpu_sync_pending;
                                    g_cpu_sync_pending = true;
                                }
                                // CPU BVH only needs rebuild if geometry actually changed
                                if (geometryChangedSinceSolid) {
                                    g_bvh_rebuild_pending = true;
                                }
                                s_lastRenderedGeometryGen = currentGen;
                            }
                            // Switching rendered → solid/matcap: only rebuild raster geometry
                            // if it doesn't exist yet or scene geometry has changed since last build.
                            if (s_prevGlobalViewportMode == Backend::ViewportMode::Rendered &&
                                viewportMode != Backend::ViewportMode::Rendered) {
                                auto* rasterBk = getRasterViewportBackend();
                                const uint64_t curGen = g_scene_geometry_generation.load(std::memory_order_acquire);
                                const bool rasterCacheValid = rasterBk && rasterBk->hasValidRasterCache(curGen);
                                if (!rasterCacheValid) {
                                    g_viewport_raster_rebuild_pending = true;
                                }
                            }
                            s_prevGlobalViewportMode = viewportMode;
                            // Ensure we have a valid Raw Buffer (original_surface) for Backend output
                            EnsureOriginalSurface(surface);
                            
                            // [FIX] did_render_this_frame is set AFTER the render call below,
                            // only if the backend actually produced output. Setting it before
                            // caused uninitialized original_surface pixels to be blitted to
                            // the display when the RT pipeline wasn't ready yet.
      
                            if (timeline_playing) {
                                loop_count = std::max(1, render_settings.animation_samples_per_frame);
                            }
                            
                            for (int i = 0; i < loop_count; ++i) {
                                 // [CRITICAL FIX] Check if camera changed during render and reset accumulation
                                 // This prevents bright-while-moving artifacts by ensuring GPU accumulation
                                 // is cleared when camera parameters change mid-render
                                 if (g_camera_dirty && scene.camera && activeViewportBackend) {
                                     activeViewportBackend->syncCamera(*scene.camera);
                                     activeViewportBackend->resetAccumulation();
                                     g_camera_dirty = false;
                                 }
                                 
                                 // Render using the backend's generic progressive interface
                                 if (activeViewportBackend) {
                                     if (activeViewportBackend->isAccumulationComplete()) break;
                                     activeViewportBackend->renderProgressive(
                                         original_surface, nullptr, renderer,
                                         image_width, image_height, &framebuffer, raytrace_texture);
                                     // Backend always writes valid pixels to original_surface
                                     // (RT trace, background fallback, or cached re-present).
                                     // Flag as rendered so display pipeline blits to screen.
                                     did_render_this_frame = true;
                                 }
                            }
                        } else {
                            // CPU Rendering (Fallback)
                            const bool active_gpu_backend = backendIsOptix || backendIsVulkan || vulkanRasterActive;
                            // Check if camera changed during render
                            if (g_camera_dirty && scene.camera) {
                                ray_renderer.syncCameraToBackend(*scene.camera);
                                if (active_gpu_backend && activeViewportBackend) activeViewportBackend->resetAccumulation();
                                ray_renderer.resetCPUAccumulation();
                                g_camera_dirty = false;
                            }
                            
                            if (active_gpu_backend && activeViewportBackend) {
                                if (activeViewportBackend->isAccumulationComplete()) {
                                    // Done — but surface still has valid data from last render
                                    did_render_this_frame = true;
                                } else {
                                    activeViewportBackend->renderProgressive(
                                        original_surface, nullptr, renderer,
                                        image_width, image_height, &framebuffer, raytrace_texture);
                                    did_render_this_frame = true;
                                }
                            } else {
                                // Default legacy CPU Path
                                ray_renderer.render_progressive_pass(original_surface, window, scene, 1);
                                did_render_this_frame = true;
                            }
                        }
                        
                        auto sample_end = std::chrono::high_resolution_clock::now();
                        float sample_time_ms = std::chrono::duration<float, std::milli>(sample_end - sample_start).count();
                        
                        // Update progress for UI
                        const bool active_gpu_backend_for_stats = backendIsOptix || backendIsVulkan;
                        int prev_samples = render_settings.render_current_samples;
                        if (active_gpu_backend_for_stats && g_backend) {
                            render_settings.render_current_samples = g_backend->getCurrentSampleCount();
                        }
                        
                        int effective_max_samples = render_settings.is_final_render_mode ? render_settings.final_render_samples : render_settings.max_samples;
                        render_settings.render_target_samples = effective_max_samples > 0 ? effective_max_samples : 100;
                        
                        render_settings.render_progress = (float)render_settings.render_current_samples / render_settings.render_target_samples;
                        if (active_gpu_backend_for_stats && g_backend) {
                            render_settings.is_rendering_active = !g_backend->isAccumulationComplete();
                        }
                        
                        // Update progress for UI from GPU Backend
                        if (active_gpu_backend_for_stats && g_backend) {
                            render_settings.render_current_samples = g_backend->getCurrentSampleCount();
                            if (render_settings.render_target_samples > 0) {
                                render_settings.render_progress = (float)render_settings.render_current_samples / render_settings.render_target_samples;
                            }
                            render_settings.is_rendering_active = !g_backend->isAccumulationComplete();
                        }
                        
                        // ===== DENOISE LOGIC =====
                        // Priority: Final Render > Timeline Playback > Viewport
                        bool effective_denoiser = false;
                        if (render_settings.is_final_render_mode) {
                            effective_denoiser = render_settings.render_use_denoiser;
                        } else if (timeline_playing) {
                            effective_denoiser = render_settings.timeline_use_denoiser;
                        } else {
                            effective_denoiser = render_settings.use_denoiser;
                        }
                        const bool use_denoiser_aux = (ui_ctx.render_settings.denoiser_mode == DenoiserMode::Quality);
                        const int denoiser_sample_count = (g_backend ? g_backend->getCurrentSampleCount() : 0);
                        const bool allow_vulkan_viewport_denoiser = true;
                        
                        if (effective_denoiser &&
                            g_backend &&
                            denoiser_sample_count > 0 &&
                            allow_vulkan_viewport_denoiser &&
                            !g_viewport_rebuild_in_progress.load(std::memory_order_acquire) &&
                            !g_optix_rebuild_in_progress.load(std::memory_order_acquire)) {
                            bool gpuPathSucceeded = false;
                            bool cpuPathSucceeded = false;
                            bool anyFrameProvided = false;

                            // Exposure factor — hoisted so both the fused-GPU tonemap kernel
                            // and the CPU fallback loop share the same value.
                            float exposure_factor = 1.0f;
                            if (scene.camera) {
                                if (scene.camera->auto_exposure) {
                                    exposure_factor = std::pow(2.0f, scene.camera->ev_compensation);
                                } else if (scene.camera->use_physical_exposure) {
                                    float iso_mult = 1.0f;
                                    if (scene.camera->iso_preset_index >= 0 &&
                                        scene.camera->iso_preset_index < (int)CameraPresets::ISO_PRESET_COUNT) {
                                        iso_mult = CameraPresets::ISO_PRESETS[scene.camera->iso_preset_index].exposure_multiplier;
                                    }

                                    float shutter_time = 0.004f;
                                    if (scene.camera->shutter_preset_index >= 0 &&
                                        scene.camera->shutter_preset_index < (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT) {
                                        shutter_time = CameraPresets::SHUTTER_SPEED_PRESETS[scene.camera->shutter_preset_index].speed_seconds;
                                    }

                                    float f_number = 16.0f;
                                    if (scene.camera->fstop_preset_index > 0 &&
                                        scene.camera->fstop_preset_index < (int)CameraPresets::FSTOP_PRESET_COUNT) {
                                        f_number = CameraPresets::FSTOP_PRESETS[scene.camera->fstop_preset_index].f_number;
                                    } else if (scene.camera->aperture > 0.001f) {
                                        f_number = 0.8f / scene.camera->aperture;
                                    }

                                    float aperture_sq = f_number * f_number;
                                    float ev_comp = std::pow(2.0f, scene.camera->ev_compensation);
                                    float current_val = (iso_mult * shutter_time) / (aperture_sq + 1e-6f);
                                    float baseline_val = 0.00003125f;
                                    exposure_factor = (current_val / baseline_val) * ev_comp * 2.0f;
                                } else {
                                    exposure_factor = std::pow(2.0f, scene.camera->ev_compensation);
                                }
                            }

                            SDL_PixelFormat* fmt = original_surface ? original_surface->format : nullptr;
                            const Uint32 aMask = fmt ? fmt->Amask : 0u;
                            const Uint8 rShift = fmt ? fmt->Rshift : 0;
                            const Uint8 gShift = fmt ? fmt->Gshift : 8;
                            const Uint8 bShift = fmt ? fmt->Bshift : 16;

                            // GPU path — fused on-device tonemap, single 4 B/px D2H copy.
                            {
                                Backend::DenoiserFrameDataGPU gpuFrame;
                                std::vector<uint32_t> packedDenoised;
                                if (g_backend->getDenoiserFrameGPU(gpuFrame, use_denoiser_aux)) {
                                    anyFrameProvided = true;
                                    Renderer::OIDNTonemapParams tm;
                                    tm.exposure = exposure_factor;
                                    tm.aMaskOr  = aMask;
                                    tm.rShift   = rShift;
                                    tm.gShift   = gShift;
                                    tm.bShift   = bShift;
                                    tm.flipY    = true;
                                    // OIDN model tier: user setting in the viewport,
                                    // forced to High for final renders.
                                    if (render_settings.is_final_render_mode) {
                                        tm.quality = OIDN_QUALITY_HIGH;
                                    } else {
                                        switch (ui_ctx.render_settings.denoiser_quality) {
                                            case DenoiserQuality::Balanced: tm.quality = OIDN_QUALITY_BALANCED; break;
                                            case DenoiserQuality::High:     tm.quality = OIDN_QUALITY_HIGH; break;
                                            case DenoiserQuality::Fast:
                                            default:                        tm.quality = OIDN_QUALITY_FAST; break;
                                        }
                                    }
                                    if (ray_renderer.applyOIDNDenoisingGPU(gpuFrame,
                                                                           ui_ctx.render_settings.denoiser_blend_factor,
                                                                           tm, packedDenoised)) {
                                        const int denoisedW = gpuFrame.width;
                                        const int denoisedH = gpuFrame.height;
                                        if (original_surface && original_surface->pixels &&
                                            original_surface->w == denoisedW && original_surface->h == denoisedH) {
                                            const int row_stride = original_surface->pitch / 4;
                                            Uint32* dst = static_cast<Uint32*>(original_surface->pixels);
                                            const uint32_t* src = packedDenoised.data();
                                            if (row_stride == denoisedW) {
                                                std::memcpy(dst, src,
                                                            static_cast<size_t>(denoisedW) *
                                                                static_cast<size_t>(denoisedH) *
                                                                sizeof(uint32_t));
                                            } else {
                                                for (int y = 0; y < denoisedH; ++y) {
                                                    std::memcpy(dst + static_cast<size_t>(y) * row_stride,
                                                                src + static_cast<size_t>(y) * denoisedW,
                                                                static_cast<size_t>(denoisedW) * sizeof(uint32_t));
                                                }
                                            }
                                        }
                                        gpuPathSucceeded = true;
                                    }
                                }
                            }

                            // CPU fallback — only if the GPU path didn't land (binding failed,
                            // shared memory not available, etc.). Keeps the original per-pixel
                            // tonemap loop for parity.
                            if (!gpuPathSucceeded) {
                                std::vector<float> denoised;
                                Backend::DenoiserFrameData denoiserFrame;
                                if (g_backend->getDenoiserFrame(denoiserFrame, use_denoiser_aux)) {
                                    anyFrameProvided = true;
                                    Renderer::OIDNFrameData frame;
                                    frame.width = denoiserFrame.width;
                                    frame.height = denoiserFrame.height;
                                    frame.color = denoiserFrame.color;
                                    frame.albedo = use_denoiser_aux ? denoiserFrame.albedo : nullptr;
                                    frame.normal = use_denoiser_aux ? denoiserFrame.normal : nullptr;
                                    // Same tier mapping as the GPU path so AMD/Intel (host
                                    // fallback) honor the viewport quality setting too.
                                    int host_quality = OIDN_QUALITY_FAST;
                                    if (render_settings.is_final_render_mode) {
                                        host_quality = OIDN_QUALITY_HIGH;
                                    } else {
                                        switch (ui_ctx.render_settings.denoiser_quality) {
                                            case DenoiserQuality::Balanced: host_quality = OIDN_QUALITY_BALANCED; break;
                                            case DenoiserQuality::High:     host_quality = OIDN_QUALITY_HIGH; break;
                                            case DenoiserQuality::Fast:
                                            default:                        host_quality = OIDN_QUALITY_FAST; break;
                                        }
                                    }
                                    if (ray_renderer.applyOIDNDenoising(frame, ui_ctx.render_settings.denoiser_blend_factor, denoised, host_quality)) {
                                        const int denoisedW = frame.width;
                                        const int denoisedH = frame.height;
                                        if (original_surface && original_surface->pixels &&
                                            original_surface->w == denoisedW && original_surface->h == denoisedH) {
                                            Uint32* pixels = static_cast<Uint32*>(original_surface->pixels);
                                            const int row_stride = original_surface->pitch / 4;
                                            const size_t pixelCount = (size_t)denoisedW * (size_t)denoisedH;
                                            Uint32* __restrict pxBase = pixels;
                                            const float* __restrict denoisedBase = denoised.data();
                                            std::for_each_n(std::execution::par_unseq,
                                                pxBase, pixelCount,
                                                [=](Uint32& px) {
                                                    const size_t i = static_cast<size_t>(&px - pxBase);
                                                    const size_t idx = i * 3;
                                                    const int x = static_cast<int>(i % (size_t)denoisedW);
                                                    const int y = static_cast<int>(i / (size_t)denoisedW);
                                                    const int screen_y = denoisedH - 1 - y;
                                                    const size_t screen_index = (size_t)screen_y * (size_t)row_stride + (size_t)x;

                                                    float r = std::max(denoisedBase[idx] * exposure_factor, 0.0f);
                                                    float g = std::max(denoisedBase[idx + 1] * exposure_factor, 0.0f);
                                                    float b = std::max(denoisedBase[idx + 2] * exposure_factor, 0.0f);

                                                    r = r / (1.0f + r);
                                                    g = g / (1.0f + g);
                                                    b = b / (1.0f + b);

                                                    r = std::pow(r, 1.0f / 2.2f);
                                                    g = std::pow(g, 1.0f / 2.2f);
                                                    b = std::pow(b, 1.0f / 2.2f);

                                                    const Uint8 ri = static_cast<Uint8>(std::min(r, 1.0f) * 255.0f + 0.5f);
                                                    const Uint8 gi = static_cast<Uint8>(std::min(g, 1.0f) * 255.0f + 0.5f);
                                                    const Uint8 bi = static_cast<Uint8>(std::min(b, 1.0f) * 255.0f + 0.5f);
                                                    const Uint32 alpha = pxBase[screen_index] & aMask;
                                                    pxBase[screen_index] = alpha
                                                        | ((Uint32)ri << rShift)
                                                        | ((Uint32)gi << gShift)
                                                        | ((Uint32)bi << bShift);
                                                });
                                        }
                                        cpuPathSucceeded = true;
                                    }
                                }
                            }

                            if (!gpuPathSucceeded && !cpuPathSucceeded && !anyFrameProvided) {
                                // Backend supplied no denoiser frame at all — fall back to SDL surface OIDN.
                                ray_renderer.applyOIDNDenoising(original_surface, 0, true, ui_ctx.render_settings.denoiser_blend_factor);
                            }
                        }

                        if (render_settings.render_current_samples > prev_samples) {
                            const auto total_frame_end = std::chrono::high_resolution_clock::now();
                            const float total_frame_time_ms = std::chrono::duration<float, std::milli>(total_frame_end - sample_start).count();
                            const float total_frame_fps = total_frame_time_ms > 0.0f ? (1000.0f / total_frame_time_ms) : 0.0f;

                            render_settings.avg_sample_time_ms = render_settings.avg_sample_time_ms * 0.8f + total_frame_time_ms * 0.2f;
                            render_settings.avg_total_frame_time_ms = render_settings.avg_total_frame_time_ms * 0.8f + total_frame_time_ms * 0.2f;
                            render_settings.avg_total_frame_fps = render_settings.avg_total_frame_fps * 0.8f + total_frame_fps * 0.2f;
                            render_settings.render_elapsed_seconds += total_frame_time_ms / 1000.0f;

                            const int remaining_samples = render_settings.render_target_samples - render_settings.render_current_samples;
                            render_settings.render_estimated_remaining = (remaining_samples * render_settings.avg_sample_time_ms) / 1000.0f;

                            if (window && active_gpu_backend_for_stats) {
                                std::string projectName = active_model_path;
                                if (projectName.empty() || projectName == "Untitled") {
                                    projectName = "Untitled Project";
                                } else {
                                    size_t lastSlash = projectName.find_last_of("\\/");
                                    if (lastSlash != std::string::npos) projectName = projectName.substr(lastSlash + 1);
                                }
                                size_t dot = projectName.find_last_of(".");
                                if (dot != std::string::npos) projectName = projectName.substr(0, dot);

                                float progress_pct = render_settings.render_progress * 100.0f;
                                std::string backend_name = isActiveRenderBackendVulkan() ? "Vulkan" :
                                    (isActiveRenderBackendOptix() ? "OptiX" : "CPU");
                                const bool denoiser_enabled_now = effective_denoiser && denoiser_sample_count > 0;
                                std::string title = "RayTrophi Studio [" + projectName + "] - " + backend_name + " - Sample " +
                                    std::to_string(render_settings.render_current_samples) + "/" +
                                    std::to_string(render_settings.render_target_samples) +
                                    " (" + std::to_string(int(progress_pct)) + "%) - " +
                                    std::to_string(int(total_frame_time_ms)) + "ms/frame - " +
                                    std::to_string(total_frame_fps).substr(0, 4) + " fps";
                                if (denoiser_enabled_now) title += " - Denoised";
                                SDL_SetWindowTitle(window, title.c_str());
                            }
                        }
                        } // end anim_owns_backend else
                        }
                    else {
                        // ============ SYNCHRONOUS CPU RENDER (Like OptiX) ============
                        static bool logged_cpu = false;
                        if (!logged_cpu) { SCENE_LOG_INFO("[DEBUG] Entering CPU Render block"); logged_cpu = true; }
                        // Each pass is 1 sample per pixel, accumulates progressively
                        
                        // ===============================================================
                        // GPU -> CPU MODE SYNC: Update CPU vertices from GPU state
                        // ===============================================================
                        if (g_cpu_sync_pending) {
                            ui.addViewportMessage("Syncing CPU data...", 5.0f);

                            // Update all CPU vertices from their Transform handles.
                            // Per-object work is independent (each writes only its own
                            // transform/AABB) so par_unseq is safe; foliage scenes with
                            // millions of HittableInstances were single-thread bound here.
                            std::for_each(std::execution::par_unseq,
                                scene.world.objects.begin(), scene.world.objects.end(),
                                [](std::shared_ptr<Hittable>& obj) {
                                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                    if (tri) {
                                        tri->updateTransformedVertices();
                                        return;
                                    }

                                    auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
                                    if (inst && inst->syncTransformFromSourceTriangles()) {
                                        return;
                                    }

                                    if (inst && inst->source_triangles) {
                                        for (auto& srcTri : *inst->source_triangles) {
                                            if (srcTri) {
                                                srcTri->updateTransformedVertices();
                                            }
                                        }
                                    }
                                });

                            // Trigger CPU BVH rebuild (SYNCHRONOUS for safety)
                            // Async rebuild causes crashes if objects were deleted in GPU mode.
                            // skip_sync=true: we already did the per-object sync above,
                            // don't repeat the same serial-equivalent work inside rebuildBVH.
                            extern bool use_embree;
                            ray_renderer.rebuildBVH(scene, use_embree, /*skip_sync=*/true);
                            
                            g_bvh_rebuild_pending = false; // Cancel any pending async rebuilds
                            ray_renderer.resetCPUAccumulation();
                            g_cpu_sync_pending = false;
                            
                            SCENE_LOG_INFO("[CPU Sync] Updated vertices and performed SYNCHRONOUS BVH rebuild");
                        }
                        
                        // OPTIMIZATION: Only update animation state when timeline frame changed
                        // AND when we have file-based animations (not manual keyframes)
                        static int last_cpu_anim_frame = -1;
                        bool has_file_animations = !scene.animationDataList.empty();
                        if (has_file_animations) {
                             // CPU mode: apply CPU vertex skinning for ray-triangle intersection
                             bool force_bind_pose = (ui.show_hair_tab && ui.active_properties_tab == 8);
                             if (ray_renderer.updateAnimationState(scene, time, true, force_bind_pose)) {
                                 // Geometry changed (skinning applied), trigger BVH update
                                 g_cpu_bvh_refit_pending = true;
                                 // Note: resetCPUAccumulation already called inside updateAnimationWithGraph
                             }
                             last_cpu_anim_frame = current_f;
                        }
                        
                        // Set camera Y for volumetric cloud parallax
                        if (scene.camera) {
                            ray_renderer.world.setCameraY(scene.camera->lookfrom.y);
                        }

                        // Discrete sim particles moved this frame (bridge flagged it in
                        // ui.draw). Rebuild the particle-only BVH HERE — immediately before
                        // the CPU pass — so the render uses the CURRENT frame's particles
                        // with zero lag. Doing it in the post-render trigger instead left
                        // every pass one frame stale (and a converged/early-returned pass
                        // could skip the update entirely), so particles looked frozen until
                        // a backend round-trip forced a full rebuild. Cleared here; the
                        // post-render trigger only kicks start_render so this block runs.
                        if (g_particle_cpu_geometry_dirty) {
                            scene.rebuildParticleBVH(ui_ctx.render_settings.UI_use_embree);
                            ray_renderer.resetCPUAccumulation();
                            g_particle_cpu_geometry_dirty = false;
                        }

                        // Single pass render - uses accumulation internally
                        // Apply quality preset samples when timeline is playing
                        auto sample_start = std::chrono::high_resolution_clock::now();
                        
                        int loop_count = 1;
                        if (timeline_playing) {
                            loop_count = std::max(1, render_settings.animation_samples_per_frame);
                        }
                        
                        for (int i = 0; i < loop_count; ++i) {
                            if (!isActiveRenderBackendOptix()) {
                                // Forced to scalar for stability and brightness fix
                                // Ensure we have a valid Raw Buffer (original_surface) to render into
                                EnsureOriginalSurface(surface);
                                
                                // Mark frame as rendered so display update logic works
                                did_render_this_frame = true;
                                
                                // Render to original_surface (Raw LDR) instead of surface (Display/Tonemapped)
                                // This prevents "Accumulating Tonemap" corruption.
                                ray_renderer.render_progressive_pass(original_surface, window, scene, 1);
                                /*
                                if (ui_ctx.render_settings.use_vectorized_renderer) {
                                    ray_renderer.render_progressive_pass_packet(surface, window, scene, 1);
                                } else {
                                    ray_renderer.render_progressive_pass(surface, window, scene, 1);
                                }
                                */
                            }
                        }
                        
                        auto sample_end = std::chrono::high_resolution_clock::now();
                        float sample_time_ms = std::chrono::duration<float, std::milli>(sample_end - sample_start).count();
                        
                        
                        // Update progress for UI
                        int prev_samples = render_settings.render_current_samples;
                        render_settings.render_current_samples = ray_renderer.getCPUAccumulatedSamples();
                        
                        int effective_max_samples = render_settings.is_final_render_mode ? render_settings.final_render_samples : render_settings.max_samples;
                        render_settings.render_target_samples = effective_max_samples > 0 ? effective_max_samples : 100;
                        
                        render_settings.render_progress = (float)render_settings.render_current_samples / render_settings.render_target_samples;
                        render_settings.is_rendering_active = !ray_renderer.isCPUAccumulationComplete();
                        
                        // ===== DENOISE LOGIC =====
                        // Priority: Final Render > Timeline Playback > Viewport
                        bool effective_denoiser = false;
                        if (render_settings.is_final_render_mode) {
                            effective_denoiser = render_settings.render_use_denoiser;
                        } else if (timeline_playing) {
                            effective_denoiser = render_settings.timeline_use_denoiser;
                        } else {
                            effective_denoiser = render_settings.use_denoiser;
                        }
                        
                        if (effective_denoiser &&
                            ray_renderer.getCPUAccumulatedSamples() > 0) {
                            ray_renderer.applyOIDNDenoisingToCPUAccumulation(
                                ui_ctx.render_settings.denoiser_blend_factor,
                                ui_ctx.render_settings.denoiser_mode == DenoiserMode::Quality);
                        } else {
                            ray_renderer.invalidateCPUDenoisedBuffer();
                        }

                        if (render_settings.render_current_samples > prev_samples) {
                            const auto total_frame_end = std::chrono::high_resolution_clock::now();
                            const float total_frame_time_ms = std::chrono::duration<float, std::milli>(total_frame_end - sample_start).count();
                            const float total_frame_fps = total_frame_time_ms > 0.0f ? (1000.0f / total_frame_time_ms) : 0.0f;

                            render_settings.avg_sample_time_ms = render_settings.avg_sample_time_ms * 0.8f + total_frame_time_ms * 0.2f;
                            render_settings.avg_total_frame_time_ms = render_settings.avg_total_frame_time_ms * 0.8f + total_frame_time_ms * 0.2f;
                            render_settings.avg_total_frame_fps = render_settings.avg_total_frame_fps * 0.8f + total_frame_fps * 0.2f;
                            render_settings.render_elapsed_seconds += total_frame_time_ms / 1000.0f;

                            const int remaining_samples = render_settings.render_target_samples - render_settings.render_current_samples;
                            render_settings.render_estimated_remaining = (remaining_samples * render_settings.avg_sample_time_ms) / 1000.0f;

                            if (window) {
                                std::string projectName = active_model_path;
                                if (projectName.empty() || projectName == "Untitled") {
                                    projectName = "Untitled Project";
                                } else {
                                    size_t lastSlash = projectName.find_last_of("\\/");
                                    if (lastSlash != std::string::npos) projectName = projectName.substr(lastSlash + 1);
                                }
                                size_t dot = projectName.find_last_of(".");
                                if (dot != std::string::npos) projectName = projectName.substr(0, dot);

                                float progress_pct = render_settings.render_progress * 100.0f;
                                const bool denoiser_enabled_now = effective_denoiser && ray_renderer.getCPUAccumulatedSamples() > 0;
                                std::string title = "RayTrophi Studio [" + projectName + "] - CPU - Sample " +
                                    std::to_string(render_settings.render_current_samples) + "/" +
                                    std::to_string(render_settings.render_target_samples) +
                                    " (" + std::to_string(int(progress_pct)) + "%) - " +
                                    std::to_string(int(total_frame_time_ms)) + "ms/frame - " +
                                    std::to_string(total_frame_fps).substr(0, 4) + " fps";
                                if (denoiser_enabled_now) title += " - Denoised";
                                SDL_SetWindowTitle(window, title.c_str());
                            }
                        }
                        }
                    }
                 }
            }

        // ===========================================================================
        // POST-PROCESSING: Tonemap & Color Grading
        // ===========================================================================
        
        // 1. Sync RAW copy (original_surface) logic REMOVED.
        // We now render directly to 'original_surface', preventing the feedback loop.
        // if (did_render_this_frame ...) { SDL_BlitSurface... } -> Removed.
        
        // Ensure original_surface exists for fallback logic
        if (!original_surface && surface) EnsureOriginalSurface(surface);

        // 2. Handle Tonemap Reset (F11/UI)
        if (reset_tonemap) {
            if (original_surface && surface) {
                copySurfacePixelsOrBlit(surface, original_surface);
            }
            reset_tonemap = false;
            post_processing_happened = true;
            SCENE_LOG_INFO("Tonemap reset applied.");
        }

        bool stylize_applied_by_tonemap = false;
        // True when the display surface was (re)built this frame — by a tonemap apply or a
        // fresh render. Stylize must re-run on any rebuild (e.g. a Stylize param change
        // requests a redisplay with no new render), not only when a render happened.
        bool surface_rebuilt = false;
        const bool stylize_post_active =
            ray_renderer.stylizeMode.enabled && ui.viewport_settings.shading_mode == 2;
        // A Stylize param change wants the post pass re-run on the existing render. Treat it
        // like a render for display purposes so the surface rebuild HONORS the tonemap
        // on/off setting (don't force tonemap), then the stylize block re-applies on top.
        const bool needs_redisplay = did_render_this_frame || stylize_redisplay;

        // 3. Handle Tonemap Apply OR Display Update
        if (apply_tonemap || (ui_ctx.render_settings.persistent_tonemap && needs_redisplay)) {
            if (original_surface && surface) {
                const bool gpu_noop_post =
                    isActiveRenderBackendGpu() &&
                    ui_ctx.render_settings.persistent_tonemap &&
                    hasNoOpColorProcessing(color_processor) &&
                    !stylize_post_active;

                if (gpu_noop_post) {
                    copySurfacePixelsOrBlit(surface, original_surface);
                } else {
                    // Pass renderer to use float buffer if available (prevents quantization artifacts)
                    // Pass nullptr if using OptiX/Vulkan (GPU path already provides display-ready pixels)
                    applyToneMappingToSurfaceWithCamera(surface, original_surface, color_processor,
                        isActiveRenderBackendGpu() ? nullptr : &ray_renderer,
                        scene.camera.get());
                    stylize_applied_by_tonemap = stylize_post_active && !isActiveRenderBackendGpu();
                }
            }
            if (apply_tonemap) {
                apply_tonemap = false;
                SCENE_LOG_INFO("Tonemap applied.");
            }
            post_processing_happened = true;
            surface_rebuilt = true;
        }
        else if (needs_redisplay && original_surface && surface) {
            // If CPU denoiser produced a float buffer, display it even when persistent tonemap is off.
            // Otherwise denoiser appears to do nothing because original_surface still contains raw pre-denoise pixels.
            if (!isActiveRenderBackendGpu() &&
                ray_renderer.hasCPUDenoisedBuffer()) {
                applyCPUDenoisedPreviewToSurface(surface, ray_renderer, scene.camera.get());
            } else {
                // If tonemapping is disabled, we must still copy the Raw Render (original_surface)
                // to the Display Surface (surface) so the user sees the output!
                copySurfacePixelsOrBlit(surface, original_surface);
            }
            surface_rebuilt = true;
        }

        if (surface_rebuilt &&
            stylize_post_active &&
            !stylize_applied_by_tonemap &&
            surface) {
            bool stylized_on_gpu = false;

            // GPU-direct path (OptiX): run the stylize on-device using the resident
            // AOV buffers (no readback) + the already-graded surface, exactly the
            // same StylizeCore math as the CPU path. Falls back to CPU on any
            // failure (size mismatch, missing AOVs, rebuild in progress, sample 0).
            if (isActiveRenderBackendGpu() && g_backend && scene.camera &&
                g_backend->getCurrentSampleCount() >= 1) {
                const Camera& cam = *scene.camera;
                const WorldData gw = ray_renderer.world.getGPUData();

                StylizeGPU::KernelParams kp;
                kp.frame_index = gw.frame_count;
                kp.cam_lower_left = float3{ cam.lower_left_corner.x, cam.lower_left_corner.y, cam.lower_left_corner.z };
                kp.cam_horizontal = float3{ cam.horizontal.x, cam.horizontal.y, cam.horizontal.z };
                kp.cam_vertical   = float3{ cam.vertical.x, cam.vertical.y, cam.vertical.z };
                kp.cam_origin     = float3{ cam.origin.x, cam.origin.y, cam.origin.z };
                kp.ray_origin     = float3{ cam.lookfrom.x, cam.lookfrom.y, cam.lookfrom.z };
                // Raw nishita values — the kernel applies the same clamps as makeStylizeAOV.
                kp.sun_direction  = float3{ gw.nishita.sun_direction.x, gw.nishita.sun_direction.y, gw.nishita.sun_direction.z };
                kp.sun_size       = gw.nishita.sun_size;
                kp.sun_elevation  = gw.nishita.sun_elevation;
                kp.clouds_enabled = gw.nishita.clouds_enabled != 0 ? 1 : 0;
                kp.cloud_coverage = gw.nishita.cloud_coverage;
                kp.cloud_density  = gw.nishita.cloud_density;
                kp.cloud_scale    = gw.nishita.cloud_scale;
                kp.cloud_offset_x = gw.nishita.cloud_offset_x;
                kp.cloud_offset_z = gw.nishita.cloud_offset_z;
                kp.cloud_seed     = gw.nishita.cloud_seed;

                const StylizeCore::StyleProfileCore profile =
                    Stylize::makeCoreProfile(ray_renderer.stylizeMode.profile);
                stylized_on_gpu = g_backend->applyStylizeGPU(surface, kp, profile);
            }

            if (!stylized_on_gpu) {
                // CPU fallback: the CPU AOV accumulation buffers are empty on the GPU
                // render path, so pull the primary-hit AOVs to the host first, then run
                // the full surface-locked stylize (CPU path already fills them).
                if (isActiveRenderBackendGpu() && g_backend && scene.camera) {
                    ray_renderer.fillStylizeAOVFromBackend(g_backend.get(), *scene.camera);
                }
                applyStylizeToSurfaceWithCamera(surface, ray_renderer, true, scene.camera.get());
            }
            // Log only when the active stylize path changes, so it confirms which
            // backend ran without spamming every frame.
            static int s_lastStylizePath = -1;   // -1 unknown, 0 CPU, 1 GPU
            const int curStylizePath = stylized_on_gpu ? 1 : 0;
            if (curStylizePath != s_lastStylizePath) {
                s_lastStylizePath = curStylizePath;
                SCENE_LOG_INFO(std::string("Stylize path: ") +
                    (stylized_on_gpu ? "GPU (CUDA/OptiX)" : "CPU"));
            }
            post_processing_happened = true;
        }
        stylize_redisplay = false;   // one-shot redisplay request consumed

        // [DIAG] Log display pipeline state — separate counter for Rendered mode
        {
            static int s_diagDisplayFrame = 0;
            static int s_diagDisplayRendered = 0;
            ++s_diagDisplayFrame;
            const bool inRenderedMode = (ui.viewport_settings.shading_mode == 2);
            if (inRenderedMode) ++s_diagDisplayRendered;
            const bool shouldLog = (s_diagDisplayFrame <= 5) || 
                                   (inRenderedMode && s_diagDisplayRendered <= 15);
            if (shouldLog) {
                auto hexPixel = [](Uint32 v) {
                    char buf[16]; snprintf(buf, sizeof(buf), "%08X", v); return std::string(buf);
                };
                Uint32 origP0 = (original_surface && original_surface->pixels) 
                    ? static_cast<Uint32*>(original_surface->pixels)[0] : 0xDEADCAFE;
                Uint32 surfP0 = (surface && surface->pixels) 
                    ? static_cast<Uint32*>(surface->pixels)[0] : 0xDEADCAFE;
                
            }
        }

        // Image save
        if (ui_ctx.render_settings.save_image_requested && original_surface) {
            ui_ctx.render_settings.save_image_requested = false;

            std::string path = saveFileDialogW(L"PNG Files\0*.png\0All Files\0*.*\0");
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
                const bool has_active_render_gpu_backend = (g_backend != nullptr);
                
                // DEBUG: Uncomment to trace animation state
                // SCENE_LOG_INFO("[ANIM DEBUG] Frame " + std::to_string(current_playback_frame) + 
                //                " | FileAnims=" + std::to_string(has_file_animations) +
                //                " | Tracks=" + std::to_string(scene.timeline.tracks.size()) +
                //                " | Keyframes=" + std::to_string(cached_total_keyframes));
                
                // SKIP EVERYTHING if no animations exist at all
                if (!has_file_animations && !has_manual_keyframes) {
                    // No animation data: playhead can move without invalidating the scene.
                    last_playback_frame = current_playback_frame;
                    // FAST PATH: Skip all expensive work below
                }
                else {
                    // We have some animation data - process accordingly
                    
                    if (has_file_animations) {
                        // File-based animations present - need full update (Assimp skinning, node hierarchy)
                        bool geometry_modified = ray_renderer.updateAnimationState(scene, time, 
                           !has_active_render_gpu_backend);
                        
                        // CRITICAL: Update CPU BVH so viewport 'sees' the new pose (Fixes BBox/Selection)
                        if (geometry_modified && !has_active_render_gpu_backend) {
                            ray_renderer.updateBVH(scene, ui_ctx.render_settings.UI_use_embree);
                        }
                    }
                        // else: TimelineWidget::draw() handles manual keyframes with O(1) object lookup - FAST!
                    
                    if (has_file_animations || timeline_has_camera_keyframes) g_camera_dirty = true;
                    if (has_file_animations || timeline_has_light_keyframes) g_lights_dirty = true;
                    if (has_file_animations) g_world_dirty = true;
                    
                    // Update Backend if needed
                        if (has_active_render_gpu_backend) {
                        // PERFORMANCE: Only update geometry if file-based animations modified it
                        if (has_file_animations) {
                            if (g_backend) g_backend->updateGeometry(scene.world.objects);
                        } else if (wind_active) {
                            if (g_backend) g_backend->updateInstanceTransforms(scene.world.objects);
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
                        
                        if ((has_file_animations || timeline_has_camera_keyframes) && scene.camera) {
                            ray_renderer.syncCameraToBackend(*scene.camera);
                        }

                        // Timeline material keys already use per-slot updates.
                        // Keep the expensive full material sync only for explicit
                        // dirty-state rebuilds that still need one catch-up upload.
                        if (g_materials_dirty || g_texture_pool_dirty) {
                            ray_renderer.updateBackendMaterials(scene);
                            syncMaterialBufferToViewportBackend(scene, ray_renderer);
                            g_materials_dirty = false;
                            g_texture_pool_dirty = false;
                        }

                        if (g_gas_volumes_dirty) {
                            ray_renderer.updateBackendGasVolumes(scene);
                            g_gas_volumes_dirty = false;
                        }

                        // Reset accumulation for new frame
                        g_backend->resetAccumulation();
                        
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
        // SKIP during animation playback, when paused, or in Solid/Matcap mode
        // (raster viewport modes don't use sample accumulation)
        bool is_playing = ui_ctx.render_settings.animation_is_playing;
        bool is_paused = ui_ctx.render_settings.is_render_paused;
        const bool in_rendered_mode = (ui.viewport_settings.shading_mode == 2);
        const WeatherParams preview_weather = ui_ctx.renderer.world.getWeatherParams();
        const bool realtime_weather_preview_active =
            ui_ctx.render_settings.realtime_weather_preview &&
            in_rendered_mode &&
            preview_weather.enabled != 0 &&
            preview_weather.type != WEATHER_NONE &&
            preview_weather.visual_mode != WEATHER_VISUAL_SURFACE_ONLY;

        if (scene.initialized &&
            !camera_moved_recently &&
            !start_render &&
            !is_playing &&
            !is_paused &&
            !skip_backend_for_anim) {  // Don't accumulate when animation render owns backend

            if (in_rendered_mode) {
                if (realtime_weather_preview_active) {
                    const float preview_time = static_cast<float>(SDL_GetTicks()) / 1000.0f;
                    if (g_backend) {
                        g_backend->setTime(preview_time, preview_time);
                        g_backend->resetAccumulation();
                    }
                    Backend::IBackend* activeViewportBackend =
                        getActiveViewportBackendForShading(ui.viewport_settings.shading_mode);
                    if (activeViewportBackend && activeViewportBackend != g_backend.get()) {
                        activeViewportBackend->resetAccumulation();
                    }
                    ray_renderer.resetCPUAccumulation();
                    start_render = true;
                } else {
                    bool accumulation_complete = false;

                    if (isActiveRenderBackendGpu()) {
                        accumulation_complete = g_backend ? g_backend->isAccumulationComplete() : false;
                    } else {
                        accumulation_complete = ray_renderer.isCPUAccumulationComplete();
                    }

                    if (!accumulation_complete) {
                        // Automatically trigger next sample pass
                        start_render = true;
                    }
                }
            } else {
                Backend::IBackend* viewportBackend =
                    getActiveViewportBackendForShading(ui.viewport_settings.shading_mode);
                if (viewportBackend && viewportBackend->needsViewportRender()) {
                    // Interactive raster modes are event-driven rather than sample-driven.
                    // If the backend marked itself dirty (material edit, mesh paint,
                    // texture upload, project restore), wake the render loop even while
                    // the UI is otherwise idle.
                    start_render = true;
                }
            }
        }

        // ===========================================================================
        // TEXTURE UPDATE OPTIMIZATION
        // Only update texture when there's actual rendering happening
        // ===========================================================================
        bool accumulation_done_for_display = false;
        if (!in_rendered_mode) {
            // Solid/Matcap: raster viewport has no progressive accumulation.
            // Consider "done" whenever we didn't actively render this frame.
            // Also force is_rendering_active off so idle detection works.
            accumulation_done_for_display = !did_render_this_frame;
            if (!did_render_this_frame) {
                render_settings.is_rendering_active = false;
            }
        } else if (isActiveRenderBackendGpu()) {
            accumulation_done_for_display = skip_backend_for_anim ? true :
                (g_backend ? g_backend->isAccumulationComplete() : false);
            if (accumulation_done_for_display) {
                render_settings.is_rendering_active = false;
            }
        } else {
            accumulation_done_for_display = ray_renderer.isCPUAccumulationComplete();
            if (accumulation_done_for_display) {
                render_settings.is_rendering_active = false;
            }
        }

        // ── VIEWPORT-DRIVEN SEQUENCE SAVE STATE MACHINE ──────────────────────
        // Renders an animation to disk through the interactive viewport itself.
        // Each frame accumulates to the interactive quality (max_samples / adaptive
        // noise); when it converges we save EXACTLY what the viewport shows — the
        // display `surface` already carries the viewport's tonemap + stylize +
        // denoise (produced just above at the display/post block) — then advance the
        // timeline one frame and let the UI's normal per-frame sim/animation scrub +
        // render drive the next one. No separate render path, so the sequence output
        // is identical to (and as fast as) what the user sees while scrubbing.
        if (g_seq_save_active) {
            if (rendering_stopped_gpu.load() || rendering_stopped_cpu.load()) {
                g_seq_save_active = false;
                rendering_in_progress = false;
                ui_ctx.is_animation_mode = false;
                ui_ctx.render_settings.animation_render_locked = false;
                g_camera_dirty = g_lights_dirty = g_world_dirty = true;
                SCENE_LOG_INFO("[SeqSave] Cancelled at frame " + std::to_string(g_seq_save_frame));
            } else if (in_rendered_mode && isActiveRenderBackendGpu() && g_backend) {
                if (accumulation_done_for_display) {
                    if (!g_seq_save_dir.empty() && surface) {
                        char fn[512];
                        std::snprintf(fn, sizeof(fn), "%s/frame_%04d.png",
                                      g_seq_save_dir.c_str(), g_seq_save_frame);
                        if (SaveSurface(surface, fn))
                            SCENE_LOG_INFO("[SeqSave] Saved " + std::string(fn));
                        else
                            SCENE_LOG_ERROR("[SeqSave] Failed to save " + std::string(fn));

                        // Mirror to the animation preview panel.
                        {
                            std::lock_guard<std::mutex> lock(ui_ctx.animation_preview_mutex);
                            const size_t pc = (size_t)surface->w * surface->h;
                            if (ui_ctx.animation_preview_buffer.size() != pc) {
                                ui_ctx.animation_preview_buffer.resize(pc);
                                ui_ctx.animation_preview_width = surface->w;
                                ui_ctx.animation_preview_height = surface->h;
                            }
                            std::memcpy(ui_ctx.animation_preview_buffer.data(),
                                        surface->pixels, pc * sizeof(uint32_t));
                            ui_ctx.animation_preview_ready = true;
                        }
                    }

                    if (g_seq_save_frame >= g_seq_save_end) {
                        g_seq_save_active = false;
                        rendering_in_progress = false;
                        ui_ctx.is_animation_mode = false;
                        ui_ctx.render_settings.animation_render_locked = false;
                        g_camera_dirty = g_lights_dirty = g_world_dirty = true;
                        SCENE_LOG_INFO("[SeqSave] Sequence complete (last frame " +
                                       std::to_string(g_seq_save_frame) + ").");
                    } else {
                        ++g_seq_save_frame;
                        ui.timeline.setCurrentFrame(g_seq_save_frame);
                        scene.timeline.current_frame = g_seq_save_frame;
                        render_settings.animation_current_frame = g_seq_save_frame;
                        ui_ctx.render_settings.animation_current_frame = g_seq_save_frame;
                        if (g_backend) g_backend->resetAccumulation();
                        start_render = true;   // render the next frame
                    }
                } else {
                    start_render = true;       // keep accumulating this frame
                }
            }
        }

        // Only update texture if rendering is active, if we just applied a tonemap, or if UI needs it
        static bool last_texture_updated = false;
        
        // Force update if accumulation count changed, if we are in final render, 
        // if explicitly requested, OR if post-processing just happened.
        bool needs_texture_update = !accumulation_done_for_display || !last_texture_updated ||
                                     did_render_this_frame || post_processing_happened;
        
        // ===========================================================================
        // IDLE TIER SYSTEM
        //
        // Tier 0 – ACTIVE     : Rendering / camera / painting / drag  → full speed
        // Tier 1 – INTERACT   : Brush hover, UI focus, mouse on vp    → ~60 FPS
        // Tier 2 – LIGHT      : Foreground but nothing happening      → ~20 FPS
        // Tier 3 – DORMANT    : Foreground, fully still               → event-wait
        // Tier 4 – BACKGROUND : Window not focused                    → event-wait (long)
        //
        // GPU cost: T0 > T1 > T2 > T3 ≈ 0 > T4 ≈ 0
        // ===========================================================================

        // --- Window state ---
        const Uint32 win_flags = SDL_GetWindowFlags(window);
        const bool window_focused  = (win_flags & SDL_WINDOW_INPUT_FOCUS) != 0;
        const bool window_visible  = (win_flags & SDL_WINDOW_MINIMIZED) == 0;

        // --- Input state snapshot ---
        const auto& io = ImGui::GetIO();
        const bool ui_interacting = io.WantCaptureMouse || io.WantCaptureKeyboard;
        const Uint32 mouse_buttons = SDL_GetMouseState(nullptr, nullptr);
        const bool left_mouse_down = (mouse_buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;

        // --- Tool modes that need viewport alive ---
        // Single source of truth — add new tools in SceneUI::isAnyViewportToolActive()
        const bool any_brush_enabled = ui.isAnyViewportToolActive();

        const bool brush_painting = left_mouse_down && any_brush_enabled;

        // Mouse-over-viewport detection: mouse moved and NOT over ImGui panels
        static int prev_mouse_x = 0, prev_mouse_y = 0;
        int cur_mouse_x, cur_mouse_y;
        SDL_GetMouseState(&cur_mouse_x, &cur_mouse_y);
        const bool mouse_moved_on_viewport =
            window_focused && !io.WantCaptureMouse &&
            (cur_mouse_x != prev_mouse_x || cur_mouse_y != prev_mouse_y);
        prev_mouse_x = cur_mouse_x;
        prev_mouse_y = cur_mouse_y;

        // --- Classify tier ---
        // Mode-independent: direct frame signals only.
        const bool scene_load_active =
            ui.scene_loading.load() ||
            ui.scene_loading_done.load();
        const bool pending_scene_refresh =
            scene_load_active ||
            g_viewport_raster_rebuild_pending ||
            g_vulkan_rebuild_pending ||
            g_vulkan_geometry_append_pending ||
            g_optix_rebuild_pending ||
            g_bvh_rebuild_pending;
        const bool rendering_active = did_render_this_frame || start_render ||
                           autonomous_anim_graph_playing || pending_scene_refresh ||
                                       ui_ctx.render_settings.is_rendering_active;

        const bool tier0_active = window_focused &&
                                   (rendering_active || camera_moved ||
                                    brush_painting || dragging ||
                                    viewport_transform_dragging);

        const bool tier1_interact = !tier0_active && window_focused &&
                                     (any_brush_enabled || ui_interacting ||
                                      mouse_moved_on_viewport);

        const bool tier2_light = !tier0_active && !tier1_interact && window_focused;

        const bool tier3_dormant = tier2_light && !ui_interacting &&
                                    !mouse_moved_on_viewport && !any_brush_enabled;

        // Background: window lost focus (alt-tab, another app on top).
        // Final render keeps running but we skip ALL presentation.
        const bool tier4_background = !window_focused;

        // Track consecutive dormant/background frames
        static int idle_frame_count = 0;
        if (tier3_dormant || tier4_background) {
            idle_frame_count++;
        } else {
            idle_frame_count = 0;
        }
        const bool skip_present = tier4_background ||
                                   (!window_visible) ||
                       ((idle_frame_count > 3) && !autonomous_anim_graph_playing);

        // --- Present / skip ---
        if (!skip_present) {
            if (needs_texture_update) {
                SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
                last_texture_updated = !accumulation_done_for_display;
            }

            ImGui::Render();
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
            ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
            SDL_RenderPresent(renderer);
        } else {
            // Idle — consume ImGui draw data without GPU work.
            ImGui::Render();
            ImGui::GetDrawData();

            if (tier4_background || !window_visible) {
                // Background / minimized: long sleep, wake only on events.
                // GPU usage drops to ~0%. Final render thread (if any) keeps going.
                const bool keep_pumping_scene_load = scene_load_active || pending_scene_refresh || start_render;
                SDL_WaitEventTimeout(nullptr, keep_pumping_scene_load ? 16 : 500);
            } else {
                // Foreground dormant: moderate wait, responsive to mouse/keyboard.
                SDL_WaitEventTimeout(nullptr, 200);
            }
        }

        // --- Throttle based on tier ---
        if (!skip_present) {
            if (tier2_light) {
                // Foreground idle → ~20 FPS
                std::this_thread::sleep_for(std::chrono::milliseconds(48));
            } else if (tier1_interact) {
                // Brush hover / UI interaction → ~60 FPS
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
            // tier0: full speed, no sleep
        }
        
        // ===========================================================================
        // DEFERRED REBUILD PROCESSING - Batched at frame end for faster UI response
        // ===========================================================================
        auto* rasterViewportBackend = getRasterViewportBackend();
        const bool interactive_viewport_active =
            isVulkanInteractiveViewportActive(
                rasterViewportBackend != nullptr,
                ui.viewport_settings.shading_mode);
        const bool active_optix_backend =
            (dynamic_cast<Backend::OptixBackend*>(g_backend.get()) != nullptr);
        const bool active_vulkan_render_backend =
            (dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr);
        const bool active_vulkan_raster_backend =
            (rasterViewportBackend != nullptr);

        // Tell the physics write-back which refresh path to use for deforming bodies.
        // Vulkan RT (rendered mode) → cheap per-mesh BLAS refit; anything else → the
        // full-rebuild path. Backend type is stable across a play session, so the
        // one-frame lag relative to the sim step (which runs earlier, in the UI pass)
        // is harmless.
        scene.setDeformRefitViaVulkan(active_vulkan_render_backend && g_hasVulkan &&
                                      !interactive_viewport_active && !skip_backend_for_anim);

        // Drop stale pending flags that belong to an inactive backend to avoid
        // repeated heavy rebuild attempts after backend/mode transitions.
        if (!active_optix_backend && g_optix_rebuild_pending) {
            g_optix_rebuild_pending = false;
        }
        if (!active_vulkan_render_backend && g_vulkan_rebuild_pending) {
            g_vulkan_rebuild_pending = false;
        }
        if (!active_vulkan_raster_backend && g_viewport_raster_rebuild_pending) {
            g_viewport_raster_rebuild_pending = false;
        }

        if (g_mesh_cache_dirty) {
            ui.rebuildMeshCache(scene.world.objects);
            g_mesh_cache_dirty = false;
        }
        
        // ===========================================================================
        // ASYNC BVH REBUILD (Non-Blocking)
        // ===========================================================================
        if (g_gpu_refit_pending && !skip_backend_for_anim) {
            if ((g_backend || rasterViewportBackend) &&
                (isActiveRenderBackendGpu() ||
                 interactive_viewport_active)) {
                if (interactive_viewport_active && !g_viewport_raster_rebuild_pending) {
                    // Solid mode: only sync raster instance transforms (no TLAS cost)
                    if (auto* vkBackend = rasterViewportBackend) {
                        vkBackend->syncRasterInstanceTransforms(scene.world.objects);
                    }
                } else if (!interactive_viewport_active) {
                    // Rendered mode: full TLAS refit
                    g_backend->updateInstanceTransforms(scene.world.objects);
                    g_backend->setLights(scene.lights);
                    if (scene.camera) {
                        g_backend->syncCamera(*scene.camera);
                    }
                    g_backend->resetAccumulation();
                }
                start_render = true;
            }
            g_gpu_refit_pending = false;
        }

        // CPU BVH Fast Refit (Embree only)
        if (g_cpu_bvh_refit_pending && !g_bvh_rebuild_pending) {
            bool use_embree = ui_ctx.render_settings.UI_use_embree;
            ray_renderer.rebuildBVH(scene, use_embree);
            ray_renderer.resetCPUAccumulation();
            g_cpu_bvh_refit_pending = false;
        }

        if (g_bvh_rebuild_deferred_frames > 0 &&
            !g_bvh_rebuild_pending &&
            !g_vulkan_rebuild_pending &&
            !g_optix_rebuild_pending &&
            !g_viewport_raster_rebuild_pending) {
            --g_bvh_rebuild_deferred_frames;
            if (g_bvh_rebuild_deferred_frames <= 0) {
                g_bvh_rebuild_pending = true;
            }
        }

        // Particle bridge changed its live instance set this frame. The actual
        // particle_bvh rebuild happens in the CPU render block immediately before the
        // pass (zero-lag, current-frame particles). Here we only KICK a render so that
        // block is entered — gated on CPU being the active backend so a moving sim
        // costs nothing during GPU/viewport sessions (GPU refits via g_gpu_refit_pending).
        // The flag is left set for the render block to consume + clear.
        if (g_particle_cpu_geometry_dirty && !use_optix && !render_settings.use_vulkan) {
            start_render = true;
        }

        static std::future<std::shared_ptr<Hittable>> g_bvh_future;

        if (g_bvh_rebuild_pending) {
            ++g_cpu_bvh_requested_generation;

            // Always keep CPU BVH up to date (async) even while GPU backends are active.
            // Solid/Vulkan sessions otherwise leave CPU BVH stale and CPU render appears empty.
            if (!g_bvh_future.valid()) {
                // Determine if we actually need it (if using OptiX, maybe unnecessary?)
                // But picking (selection) uses CPU BVH. So we DO need it eventually.
                // Async allows it to happen without freezing.

                // Sync transform-handle based geometry to CPU-space before snapshot.
                // Runs on the main thread before the async BVH build is dispatched, so
                // a serial loop here stalls the UI for foliage-heavy scenes.
                std::for_each(std::execution::par_unseq,
                    scene.world.objects.begin(), scene.world.objects.end(),
                    [](std::shared_ptr<Hittable>& obj) {
                        if (!obj) return;

                        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                            if (tri->getTransformPtr() && !tri->hasAnySkinWeights()) {
                                tri->updateTransformedVertices();
                            }
                            return;
                        }

                        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                            if (inst->syncTransformFromSourceTriangles()) {
                                return;
                            }

                            if (inst->source_triangles) {
                                for (auto& srcTri : *inst->source_triangles) {
                                    if (srcTri && srcTri->getTransformPtr() && !srcTri->hasAnySkinWeights()) {
                                        srcTri->updateTransformedVertices();
                                    }
                                }
                            }
                        }
                    });
                // Vertices are now in world-space; mark UI flag so the first
                // mouse-click selection won't redundantly re-sync them.
                ui.picking_vertices_synced = true;
                
                // Copy list of objects for thread safety (prevents crashes if objects deleted from vector)
                std::vector<std::shared_ptr<Hittable>> objects_copy = scene.world.objects;
                bool use_embree = ui_ctx.render_settings.UI_use_embree;
                
                if (!interactive_viewport_active) {
                    ui.addViewportMessage("Rebuilding BVH...", 10.0f);
                }
                g_cpu_bvh_future_generation = g_cpu_bvh_requested_generation;
                
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
                    const uint64_t completed_generation = g_cpu_bvh_future_generation;
                    auto new_bvh = g_bvh_future.get();
                    g_cpu_bvh_future_generation = 0;

                    if (completed_generation != g_cpu_bvh_requested_generation) {
                        // A newer topology request arrived while this async build was in flight.
                        // Discard the stale result and queue a fresh rebuild from current scene state.
                        g_bvh_rebuild_pending = true;
                    }
                    else {
                        // CRITICAL OPTIMIZATION: Destroy OLD BVH on a background thread!
                        // Replacing 'scene.bvh' triggers the destructor of the old BVH.
                        // For large scenes, recursively destroying millions of nodes on the Main Thread 
                        // causes a massive freeze (blocking GPU commands/UI).
                        auto old_bvh = scene.bvh;
                        scene.bvh = new_bvh;
                        
                        if (old_bvh) {
                            std::thread([old_bvh]() {
                                // old_bvh goes out of scope here and is destroyed in background
                            }).detach();
                        }
                        
                        ray_renderer.resetCPUAccumulation();
                        start_render = true;
                        render_settings.is_rendering_active = true;
                        if (!interactive_viewport_active) {
                            ui.clearViewportMessages();
                            if (new_bvh) {
                                ui.addViewportMessage("Async BVH Rebuild Complete", 2.0f);
                            } else {
                                ui.addViewportMessage("BVH cleared for empty scene", 2.0f);
                            }
                        }
                    }
                } catch (const std::exception& e) {
                   SCENE_LOG_WARN(std::string("BVH Rebuild Failed: ") + e.what());
                }
            }
        }
        
        // -----------------------------------------------------------------
        // ASYNC OPTIX REBUILD (Non-blocking)
        // -----------------------------------------------------------------
        if (g_optix_rebuild_pending && active_optix_backend && g_hasOptix && !interactive_viewport_active && !skip_backend_for_anim) {
            // Only start if not already rebuilding
            if (!g_optix_rebuilding) {
                ui.addViewportMessage("Rebuilding OptiX Geometry...", 10.0f);
                
                // CRITICAL FIX: Ensure GPU is completely idle before freeing/modifying geometry
                if (g_hasCUDA) cudaDeviceSynchronize();

                auto& renderer_ref = ray_renderer;
                
                // CRITICAL FIX: Set global flag to pause rendering in OptixWrapper
                g_optix_rebuild_in_progress.store(true, std::memory_order_release);

                // CRITICAL FIX: COPY the objects list to avoid race condition with Main Thread modifications 
                // such as syncInstancesToScene or object deletion.
                std::vector<std::shared_ptr<Hittable>> objects_snapshot = collectVisibleSceneObjects(scene);

                g_optix_future = std::async(std::launch::async, [objects_snapshot, &renderer_ref]() {
                    // Use the snapshot instead of direct reference to scene.world.objects
                    renderer_ref.rebuildBackendGeometryWithList(objects_snapshot);
                });
                
                g_optix_rebuilding = true;
                g_optix_rebuild_pending = false; // Only clear when we actually start the rebuild
            }
            // Else: Keep g_optix_rebuild_pending = true so it triggers after current rebuild finishes
        }
        
        // Check if async OptiX rebuild is complete
        if (g_optix_rebuilding && g_optix_future.valid() && !skip_backend_for_anim) {
            if (g_optix_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    g_optix_future.get();

                    applyPendingDeleteVisibilityToBackend(scene, g_backend.get());
                    if (g_backend) g_backend->resetAccumulation();
                    ui.clearViewportMessages();
                    ui.addViewportMessage("OptiX Rebuild Complete", 2.0f);
                    
                    // CRITICAL FIX: Mark all scene data dirty after rebuild to force re-sync
                    g_camera_dirty = true;
                    g_lights_dirty = true;
                    g_world_dirty = true;
                    g_needs_optix_sync.store(true, std::memory_order_release);
                    
                    start_render = true;
                } catch (const std::exception& e) {
                    SCENE_LOG_WARN(std::string("OptiX Rebuild Failed: ") + e.what());
                }
                g_optix_rebuilding = false;
                // CRITICAL FIX: Resume rendering
                g_optix_rebuild_in_progress.store(false, std::memory_order_release);
            }
        }
        
        
        // -----------------------------------------------------------------
        // VULKAN REBUILD (Synchronous for now)
        // -----------------------------------------------------------------
        if (g_optix_rebuild_pending && active_vulkan_render_backend && !skip_backend_for_anim) {
            g_vulkan_rebuild_pending = true;
            g_optix_rebuild_pending = false; // Redirected
        }

        if (g_viewport_raster_rebuild_pending && active_vulkan_raster_backend && g_hasVulkan && !skip_backend_for_anim) {
            if (auto* vkBackend = rasterViewportBackend) {
                ui.addViewportMessage("Updating Solid View...", 1.0f);
                vkBackend->buildRasterGeometry(scene.world.objects);
                applyPendingDeleteVisibilityToBackend(scene, vkBackend);
                syncMaterialBufferToViewportBackend(scene, ray_renderer);
                // Hair viewport lines must be refreshed here: Renderer::uploadHairToGPU
                // feeds the viewport backend's line buffer via uploadHairViewportLines,
                // and no other pending-block covers Solid/Matcap when the render backend
                // sync is deferred (Solid mode skips syncActiveRenderBackendScene).
                // Without this call, hair stays invisible in raster until the user
                // toggles to a different backend which re-triggers uploadHairToGPU.
                ray_renderer.uploadHairToGPU();
                g_viewport_raster_rebuild_pending = false;
                start_render = true;
                g_camera_dirty = true;
            }
        }

        // ── Incremental append fast path (scatter / instance-add) ───────────────────
        // Try the cheap path first: if the only change is appending new HittableInstances
        // (sharing already-uploaded source BLASes), refit the TLAS in-place. Falling back
        // to the full destroy+rebuild block below would re-upload every BLAS and texture
        // in the scene — pure waste when a single asset got scattered.
        if (g_vulkan_geometry_append_pending && active_vulkan_render_backend && g_hasVulkan && !interactive_viewport_active && !skip_backend_for_anim) {
            if (g_backend) {
                auto* vkBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get());
                bool applied = false;
                if (vkBackend) {
                    applied = vkBackend->tryAppendGeometryIncremental(scene.world.objects);
                }
                if (applied) {
                    ray_renderer.updateBackendMaterials(scene);
                    syncMaterialBufferToViewportBackend(scene, ray_renderer);
                    g_materials_dirty = false;
                    g_texture_pool_dirty = false;
                    g_geometry_dirty = false;
                    g_backend->resetAccumulation();
                    start_render = true;
                    g_camera_dirty = true;
                } else {
                    // Incremental path declined — promote to full rebuild.
                    g_vulkan_rebuild_pending = true;
                }
                g_vulkan_geometry_append_pending = false;
            }
        }

        // ── Deforming-body refit fast path (rigid / soft / cloth sim) ───────────────
        // A simulated body bakes new verts into its source mesh every frame. The full
        // rebuild below would destroy + recreate EVERY BLAS in the scene per frame —
        // unusable in a crowded scene. Refit ONLY the changed bodies' BLAS in place
        // ([World-Solo] meshes are created with allowUpdate) and refresh the TLAS;
        // promote to a full rebuild if any mesh can't be refit (topology mismatch) or
        // a full rebuild is already queued this frame.
        if (g_vulkan_geometry_deform_pending && active_vulkan_render_backend && g_hasVulkan &&
            !interactive_viewport_active && !skip_backend_for_anim) {
            g_vulkan_geometry_deform_pending = false;
            if (g_backend) {
                auto* vkBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get());
                bool all_ok = (vkBackend != nullptr) && !g_vulkan_rebuild_pending;
                if (all_ok) {
                    for (const auto& node : scene.takePendingDeformNodes()) {
                        auto tris = scene.collectNodeTriangles(node);
                        if (tris.empty() || !vkBackend->updateInteractiveMesh(node, tris)) {
                            all_ok = false;
                            break;
                        }
                    }
                }
                if (all_ok) {
                    g_backend->resetAccumulation();
                    start_render = true;
                    g_camera_dirty = true;
                } else {
                    // Could not refit in place (or a full rebuild is already queued) —
                    // fall back to the proven full-scene rebuild below.
                    scene.clearPendingDeformNodes();
                    g_vulkan_rebuild_pending = true;
                }
            }
        }

        if (g_vulkan_rebuild_pending && active_vulkan_render_backend && g_hasVulkan && !interactive_viewport_active && !skip_backend_for_anim) {
            if (g_backend) {
                ui.addViewportMessage("Rebuilding Vulkan Geometry...", 2.0f);

                // [VULKAN FIX] Clear mesh registry to ensure dynamic meshes (terrain) are re-uploaded
                g_backend->rebuildAccelerationStructure();

                g_backend->updateGeometry(scene.world.objects);
                // Re-sync VDB SSBO after TLAS rebuild so SSBO order matches TLAS customIndex.
                ui.syncVDBVolumesToGPU(ui_ctx);
                // Re-upload hair after full BLAS rebuild
                ray_renderer.uploadHairToGPU();
                // Upload material SSBO after geometry rebuild
                ray_renderer.updateBackendMaterials(scene);
                syncMaterialBufferToViewportBackend(scene, ray_renderer);
                syncWorldDataToBackend(g_backend.get());
                applyPendingDeleteVisibilityToBackend(scene, g_backend.get());
                g_backend->resetAccumulation();
                g_vulkan_rebuild_pending = false;
                start_render = true;

                g_camera_dirty = true;
                g_lights_dirty = true;
                g_world_dirty = false;
            }
        }

    }
    
    if (g_viewport_backend) {
        try { g_viewport_backend->waitForCompletion(); } catch (...) {}
        try { g_viewport_backend->shutdown(); } catch (...) {}
        g_viewport_backend.reset();
    }
    SDL_DestroyTexture(raytrace_texture);
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_FreeSurface(surface);
    SDL_FreeSurface(original_surface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    // Perform clean-shutdown cleanup: remove StartupCrash.log since shutdown is clean.
    removeStartupCrashLogIfExists();
    SDL_Quit();
    g_sceneLog.closeLogFile();
    return 0;
}

catch (const std::exception& e) {
    emergencyStartupLog(std::string("[FATAL] Unhandled std::exception in main: ") + e.what());
    return 1;
}
catch (...) {
    emergencyStartupLog("[FATAL] Unhandled unknown exception in main");
    return 1;
}


