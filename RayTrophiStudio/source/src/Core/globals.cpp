#include "globals.h"
#include <mutex>
#include <string>
#include <unordered_map>

// Include new node system to verify compilation
#include "NodeSystemV2.h"
#include "Backend/SceneTextureManager.h"
// Old TerrainNodes folder removed - now using TerrainNodesV2 directly

std::atomic<int> completed_pixels(0);

 float aspect_ratio = 16.0f / 9.0f; // Sabit olarak float türünde tanımlıyoruz
 int image_width = 1680;
 int image_height = static_cast<int>(image_width / aspect_ratio);
constexpr float EPSILON = 1e-7f;
std::atomic<int> next_row(0);
//constexpr double infinity = std::numeric_limits<double>::max();
std::string baseDirectory="";

constexpr float gamma= 1.0f;
constexpr float exposure= 1.0f;
constexpr float saturation=1.0f;
constexpr float aperture = 0.0f;
constexpr float focusdistance = 1.573f;
float light_radius = 0.1f; // Işık kaynağı için yarıçap
int hitcount=0;
bool is_normal_map = false;
bool globalreflectance = false;
bool use_embree = true;
bool g_hasOptix = false;
bool g_hasVulkan = false;
bool g_hasVulkanRT = false;
bool g_hasCUDA = false;
bool g_hasVulkanComputeSim = false;
std::atomic<int> g_cuda_texture_upload_scope_depth{0};
namespace {
std::mutex g_sharedSceneTextureManagerMutex;
std::shared_ptr<Backend::SceneTextureManager> g_sharedSceneTextureManager;
}

Backend::RenderBackendCapabilities captureRuntimeRenderCapabilities() {
    Backend::RenderBackendCapabilities caps;
    caps.hasVulkan = g_hasVulkan;
    caps.hasMaterialPreview = g_hasVulkan;
    caps.hasVulkanRT = g_hasVulkan && g_hasVulkanRT;
    caps.hasOptix = g_hasOptix;
    caps.hasCUDA = g_hasCUDA;
    return caps;
}
std::shared_ptr<Backend::SceneTextureManager> getSharedSceneTextureManager() {
    std::lock_guard<std::mutex> lock(g_sharedSceneTextureManagerMutex);
    if (!g_sharedSceneTextureManager) {
        g_sharedSceneTextureManager = std::make_shared<Backend::SceneTextureManager>();
    }
    return g_sharedSceneTextureManager;
}

void notifyOptixTextureDestroyed(int64_t textureId) {
    if (textureId == 0) return;
    // Use the existing instance only — never create a manager during a destroy notification
    // (would be wasteful and risks reviving a singleton during shutdown).
    std::shared_ptr<Backend::SceneTextureManager> mgr;
    {
        std::lock_guard<std::mutex> lock(g_sharedSceneTextureManagerMutex);
        mgr = g_sharedSceneTextureManager;
    }
    if (mgr) {
        mgr->clearOptixTextureId(textureId);
    }
}
float last_render_time_ms = 0.0f;  // Render süresi buraya yazılacak
int pending_width = 1280;
int pending_height = 720;
float pending_aspect_ratio = 16.0f / 9.0f;
bool pending_resolution_change=false;
bool render_finished = false;   
std::atomic<bool> rendering_in_progress = false;
std::atomic<bool> rendering_stopped_gpu = false;
std::atomic<bool> rendering_stopped_cpu = false;
std::atomic<bool> rendering_paused = false;  // Pause animation render

// Vulkan runtime device-loss indicator
bool g_vulkan_device_lost = false;
std::string g_vulkan_device_lost_msg;
std::atomic<bool> g_vulkan_trim_recreate_requested = false;

// Macros are defined in globals.h

RenderSettings render_settings;  // Uses default values from header
UILogger g_sceneLog; // global logger’ın tanımı burada

namespace {
std::mutex& sceneLogOnChangeMutex() {
    static std::mutex m;
    return m;
}
std::unordered_map<std::string, long long>& sceneLogOnChangeStates() {
    static std::unordered_map<std::string, long long> s;
    return s;
}
}

void sceneLogOnChangeReset() {
    // ★ The recorded states MUST be dropped when the scene is replaced.
    //
    // Two runs of the same script in one session (new project in between) were
    // compared to find the black-band trigger, and the second run's gate lines
    // were simply MISSING — not because the gates did not fire, but because the
    // state carried over from the first run and nothing had "changed". An
    // edge-triggered diagnostic silently reports nothing across a scene reset,
    // which is exactly the ambiguity this whole mechanism exists to remove.
    std::lock_guard<std::mutex> lock(sceneLogOnChangeMutex());
    sceneLogOnChangeStates().clear();
}

void sceneLogOnChange(const std::string& key, long long state, const std::string& msg) {
    std::mutex& s_mutex = sceneLogOnChangeMutex();
    std::unordered_map<std::string, long long>& s_states = sceneLogOnChangeStates();
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        auto it = s_states.find(key);
        if (it != s_states.end() && it->second == state) return;
        s_states[key] = state;
    }
    // Deliberately goes to SceneLog.txt only. A separate append-only file was
    // used briefly while chasing the disappearing fluid surface, because
    // SceneLog.txt is truncated on every launch and relaunching after a repro
    // destroyed the evidence. That is a capture workflow, not a permanent
    // need: an unrotated file nobody reads only grows. If a future
    // investigation needs cross-session capture, copy SceneLog.txt before
    // relaunching.
    SCENE_LOG_WARN(msg);
}




