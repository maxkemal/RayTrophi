#include "globals.h"

// Include new node system to verify compilation
#include "NodeSystemV2.h"
// Old TerrainNodes folder removed - now using TerrainNodesV2 directly

std::atomic<int> completed_pixels(0);

 float aspect_ratio = 16.0f / 9.0f; // Sabit olarak float türünde tanımlıyoruz
 int image_width = 1280*1;
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
bool g_hasCUDA = false;
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




