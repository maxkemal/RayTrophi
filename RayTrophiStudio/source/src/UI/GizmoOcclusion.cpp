// Gizmo occlusion against the scene BVH. See UI/GizmoOcclusion.h.
#include "UI/GizmoOcclusion.h"

#include "Camera.h"
#include "Hittable.h"
#include "Ray.h"
#include "scene_data.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

extern std::atomic<uint64_t> g_scene_geometry_generation;

namespace GizmoOcclusion {
namespace {

constexpr float kPiecePixels = 6.0f;      // screen length of one tested piece
constexpr int kMaxPieces = 64;            // per segment
constexpr uint64_t kRayBudget = 6000;     // per frame (~1000 for two boxes + a field); cached pieces are free
// A gizmo edge often lies ON a surface (a collider box on its object, a seed
// box on the floor). The ray stops this far short of the point - relative to
// the distance, with an absolute floor - so the surface under the line does
// not hide the line.
constexpr float kRelativeTolerance = 2.0e-3f;
constexpr float kAbsoluteTolerance = 1.0e-3f;

struct FrameState {
    const Hittable* bvh = nullptr;
    Vec3 eye;
    Vec3 forward;
    bool orthographic = false;
    bool enabled = false;
    float hidden_alpha = 0.25f;
    uint64_t rays = 0;
};

FrameState g_frame;
Stats g_stats;
Stats g_building;
uint64_t g_cache_key = 0;
// Segment key -> one byte per piece (1 = visible).
std::unordered_map<uint64_t, std::vector<uint8_t>> g_cache;

uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}
uint64_t mixFloat(uint64_t h, float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return mix(h, bits);
}
uint64_t mixVec(uint64_t h, const Vec3& v) {
    return mixFloat(mixFloat(mixFloat(h, v.x), v.y), v.z);
}

bool pointVisible(const Vec3& p) {
    Vec3 origin = g_frame.eye;
    Vec3 direction = p - origin;
    float distance = 0.0f;
    if (g_frame.orthographic) {
        // Parallel rays: start on the camera plane above the point.
        const float along = (p - g_frame.eye).dot(g_frame.forward);
        if (along <= 0.0f) {
            return true;
        }
        origin = p - g_frame.forward * along;
        direction = g_frame.forward;
        distance = along;
    } else {
        distance = direction.length();
        if (!(distance > 1.0e-5f)) {
            return true;
        }
        direction = direction * (1.0f / distance);
    }
    const float t_max = distance - std::max(kAbsoluteTolerance, distance * kRelativeTolerance);
    if (!(t_max > 1.0e-3f)) {
        return true;
    }
    ++g_frame.rays;
    HitRecord record;
    return !g_frame.bvh->hit(Ray(origin, direction), 1.0e-3f, t_max, record,
                             /*ignore_volumes=*/true);
}

ImU32 faded(ImU32 color, float alpha_scale) {
    const uint32_t a = (color >> IM_COL32_A_SHIFT) & 0xFFu;
    const uint32_t scaled = static_cast<uint32_t>(std::lround(a * std::clamp(alpha_scale, 0.0f, 1.0f)));
    return (color & ~IM_COL32_A_MASK) | (scaled << IM_COL32_A_SHIFT);
}

} // namespace

bool beginFrame(const SceneData& scene, bool enabled, float hidden_alpha, bool orthographic,
                int timeline_frame) {
    // Publish the frame that just ended, start counting the new one.
    g_building.rays = g_frame.rays;
    g_building.cache_entries = g_cache.size();
    g_stats = g_building;
    g_building = Stats{};
    g_frame = FrameState{};
    g_frame.hidden_alpha = hidden_alpha;

    if (!enabled) {
        g_building.inactive_reason = "disabled by setting (x-ray gizmos)";
    } else if (!scene.camera) {
        g_building.inactive_reason = "no active camera";
    } else if (!scene.bvh) {
        g_building.inactive_reason = "scene has no BVH (nothing to occlude against)";
    }
    if (!g_building.inactive_reason.empty()) {
        g_cache.clear();
        g_cache_key = 0;
        return false;
    }

    const Camera& cam = *scene.camera;
    g_frame.bvh = scene.bvh.get();
    g_frame.eye = cam.lookfrom;
    g_frame.forward = (cam.lookat - cam.lookfrom).normalize();
    g_frame.orthographic = orthographic;
    g_frame.enabled = true;
    g_building.active = true;

    uint64_t key = 0x5bd1e995ull;
    key = mixVec(key, cam.lookfrom);
    key = mixVec(key, cam.lookat);
    key = mixVec(key, cam.vup);
    key = mixFloat(key, cam.vfov);
    key = mix(key, orthographic ? 1u : 0u);
    key = mix(key, g_scene_geometry_generation.load(std::memory_order_acquire));
    key = mix(key, reinterpret_cast<uintptr_t>(g_frame.bvh));
    key = mix(key, static_cast<uint64_t>(static_cast<int64_t>(timeline_frame)));
    if (key != g_cache_key) {
        g_cache.clear();
        g_cache_key = key;
    }
    return true;
}

void drawSegment(ImDrawList* draw_list, const Vec3& a, const Vec3& b, ImU32 color,
                 float thickness, const ProjectFn& project) {
    if (!draw_list) {
        return;
    }
    ImVec2 sa, sb;
    const bool pa = project(a, sa);
    const bool pb = project(b, sb);
    ++g_building.segments;
    if (!g_frame.enabled) {
        if (pa && pb) {
            draw_list->AddLine(sa, sb, color, thickness);
        }
        return;
    }

    // Piece count from the projected length; a clipped end falls back to a
    // fixed subdivision so the visible part is still split and tested.
    int pieces = kMaxPieces / 2;
    if (pa && pb) {
        const float dx = sb.x - sa.x, dy = sb.y - sa.y;
        pieces = static_cast<int>(std::ceil(std::sqrt(dx * dx + dy * dy) / kPiecePixels));
    }
    pieces = std::clamp(pieces, 1, kMaxPieces);

    uint64_t key = mixVec(mixVec(0x2545f4914f6cdd1dull, a), b);
    key = mix(key, static_cast<uint64_t>(pieces));
    std::vector<uint8_t>* visibility = nullptr;
    auto found = g_cache.find(key);
    if (found != g_cache.end()) {
        visibility = &found->second;
    } else if (g_frame.rays + static_cast<uint64_t>(pieces) <= kRayBudget) {
        std::vector<uint8_t> fresh(static_cast<std::size_t>(pieces));
        for (int i = 0; i < pieces; ++i) {
            const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(pieces);
            fresh[static_cast<std::size_t>(i)] = pointVisible(a + (b - a) * t) ? 1u : 0u;
        }
        visibility = &(g_cache[key] = std::move(fresh));
    } else {
        g_building.budget_exhausted = true;
    }

    const ImU32 hidden_color = faded(color, g_frame.hidden_alpha);
    const float hidden_thickness = std::max(1.0f, thickness * 0.75f);
    int run_start = 0;
    while (run_start < pieces) {
        const bool visible = !visibility || (*visibility)[static_cast<std::size_t>(run_start)] != 0;
        int run_end = run_start + 1;
        while (run_end < pieces && visibility &&
               ((*visibility)[static_cast<std::size_t>(run_end)] != 0) == visible) {
            ++run_end;
        }
        if (!visibility) {
            run_end = pieces;
        }
        ImVec2 s0, s1;
        const Vec3 w0 = a + (b - a) * (static_cast<float>(run_start) / pieces);
        const Vec3 w1 = a + (b - a) * (static_cast<float>(run_end) / pieces);
        if (project(w0, s0) && project(w1, s1)) {
            if (visible) {
                draw_list->AddLine(s0, s1, color, thickness);
            } else {
                draw_list->AddLine(s0, s1, hidden_color, hidden_thickness);
                g_building.hidden_pieces += static_cast<uint64_t>(run_end - run_start);
            }
        }
        run_start = run_end;
    }
}

const Stats& stats() {
    return g_stats;
}

} // namespace GizmoOcclusion
