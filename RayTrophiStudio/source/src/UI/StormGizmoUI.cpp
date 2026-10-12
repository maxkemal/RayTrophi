#include "UI/StormGizmoUI.h"

#include "Api/RtApi.h"
#include "Api/RtPython.h"
#include "Atmosphere/AtmosphereClouds.h"
#include "Camera.h"
#include "Renderer.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace StormGizmoUI {

bool visible = true;

namespace {

constexpr int kNone = -2;
constexpr int kDomain = -1;
constexpr int kAutoBase = -100;   // kAutoBase - i: automatic cell i (click converts)
int selected = kNone;   // kDomain, an authored cell index, or kNone
std::string error;

ImU32 phaseColour(float life, bool permanent, int alpha) {
    if (permanent) return IM_COL32(255, 185, 70, alpha);
    if (life < 0.25f) return IM_COL32(110, 200, 255, alpha);   // growing
    if (life < 0.7f) return IM_COL32(255, 120, 70, alpha);     // mature
    return IM_COL32(170, 170, 185, alpha);                     // dissipating
}

const char* phaseName(float life) {
    return life < 0.25f ? "growing" : (life < 0.7f ? "mature" : "dissipating");
}

void apply(const atmosphere::CloudState& clouds) {
    nlohmann::json json;
    atmosphere::cloudsToJson(clouds, json);
    const rtapi::Result result = rtapi::setCloudsJson(nlohmann::json{{"storms", json["storms"]}}.dump());
    error = result.ok ? std::string{} : result.error;
}

} // namespace

void drawViewport(UIContext& ctx, bool ortho, bool showGizmos, bool& hit) {
    if (!visible || !showGizmos || !ctx.scene.camera) return;
    if (rtpython::wantsInputCapture() || rtapi::renderOutputPending()) return;
    const World& world = ctx.renderer.world;
    const atmosphere::CloudState& clouds = world.getClouds();
    const atmosphere::CloudStorms& storms = clouds.storms;
    if (!storms.enabled) {
        selected = kNone;
        return;
    }
    if (selected >= static_cast<int>(storms.cells.size())) selected = kNone;

    const auto& cam = *ctx.scene.camera;
    const ImVec2 size = ImGui::GetIO().DisplaySize;
    if (size.x <= 0 || size.y <= 0) return;
    const Vec3 f = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 r = f.cross(cam.vup).normalize();
    const Vec3 u = r.cross(f);
    const float aspect = size.x / size.y;
    const float tangent = std::tan(cam.vfov * 0.00872664626f);
    auto project = [&](const Vec3& p, ImVec2& screen) {
        const Vec3 delta = p - cam.lookfrom;
        const float depth = delta.dot(f);
        if (!ortho && depth <= 0.1f) return false;
        const float half = ortho ? cam.ortho_height * 0.5f : depth * tangent;
        if (half <= 1e-6f) return false;
        screen = ImVec2((0.5f + delta.dot(r) / (2 * half * aspect)) * size.x,
                        (0.5f - delta.dot(u) / (2 * half)) * size.y);
        return true;
    };
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    auto line = [&](const Vec3& a, const Vec3& b, ImU32 colour, float width) {
        ImVec2 sa, sb;
        if (project(a, sa) && project(b, sb)) draw->AddLine(sa, sb, colour, width);
    };
    auto ring = [&](float x, float y, float z, float radius, ImU32 colour, float width) {
        ImVec2 last;
        bool valid = false;
        for (int s = 0; s <= 64; ++s) {
            const float a = s * 0.09817477042f;
            ImVec2 point;
            const bool current = project(Vec3(x + std::cos(a) * radius, y, z + std::sin(a) * radius), point);
            if (current && valid) draw->AddLine(last, point, colour, width);
            last = point;
            valid = current;
        }
    };

    // Scene height of an altitude above sea level (curvature is a few tens of
    // metres over a storm domain: irrelevant for a handle).
    const float altitudeOffset = world.getNishitaParams().altitude;
    const float groundY = clouds.lightning.ground_altitude_m - altitudeOffset;
    float baseY = groundY + 1500.0f;
    for (const auto& layer : clouds.layers) {
        if (layer.enabled) {
            baseY = layer.base_altitude_m - altitudeOffset;
            break;
        }
    }

    const auto* hovered = ImGui::GetCurrentContext()->HoveredWindow;
    const bool viewportInput = !ImGui::GetIO().WantCaptureMouse ||
        (hovered && (std::strcmp(hovered->Name, "##ViewportAssetDropTarget") == 0 ||
                     std::strcmp(hovered->Name, "##RayTrophiDockHost") == 0));
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    bool overHandle = false;
    auto handle = [&](const Vec3& p, int id, ImU32 colour, const char* label) {
        ImVec2 s;
        if (!project(p, s)) return;
        const bool isSelected = selected == id;
        draw->AddCircleFilled(s, isSelected ? 8.0f : 6.0f, isSelected ? IM_COL32(255, 230, 120, 255) : colour);
        draw->AddText(ImVec2(s.x + 10, s.y - 8), colour, label);
        const float dx = mouse.x - s.x, dy = mouse.y - s.y;
        if (dx * dx + dy * dy < 144.0f && viewportInput && !ImGuizmo::IsUsing()) {
            hit = true;
            overHandle = true;
            if (ImGui::IsMouseClicked(0)) selected = id;
        }
    };

    // Storm domain: where automatic cells are born.
    {
        const float h = storms.extent_m * 0.5f;
        const float cx = storms.center_m.x, cz = storms.center_m.z;
        const ImU32 c = IM_COL32(120, 160, 255, 110);
        line(Vec3(cx - h, groundY, cz - h), Vec3(cx + h, groundY, cz - h), c, 1.0f);
        line(Vec3(cx + h, groundY, cz - h), Vec3(cx + h, groundY, cz + h), c, 1.0f);
        line(Vec3(cx + h, groundY, cz + h), Vec3(cx - h, groundY, cz + h), c, 1.0f);
        line(Vec3(cx - h, groundY, cz + h), Vec3(cx - h, groundY, cz - h), c, 1.0f);
        handle(Vec3(cx, groundY, cz), kDomain, IM_COL32(120, 160, 255, 230), "Storm domain");
    }

    // Live cells: a column from the ground to the cloud base.
    const Vec3 wind = world.cloudWindVelocity();
    atmosphere::StormCellState cells[atmosphere::kMaxStormCells];
    const int n = atmosphere::evaluateStormCells(clouds, world.getCloudTime(), wind, cells);
    for (int i = 0; i < n; ++i) {
        const auto& c = cells[i];
        const bool permanent = c.manual && storms.cells[c.manual_index].lifetime_s <= 0.0f;
        const bool pick = c.manual_index >= 0 && c.manual_index == selected;
        const ImU32 colour = phaseColour(c.life, permanent, pick ? 255 : 170);
        const float width = pick ? 2.5f : (c.manual ? 1.8f : 1.0f);
        ring(c.x, groundY, c.z, c.radius_m, colour, width);
        ring(c.x, baseY, c.z, c.radius_m, colour, width);
        for (int k = 0; k < 4; ++k) {
            const float a = k * 1.57079632679f;
            const float x = c.x + std::cos(a) * c.radius_m, z = c.z + std::sin(a) * c.radius_m;
            line(Vec3(x, groundY, z), Vec3(x, baseY, z), colour, 1.0f);
        }
        char label[64];
        if (c.manual) {
            std::snprintf(label, sizeof(label), "Cell %d (%s)", c.manual_index + 1,
                          permanent ? "permanent" : phaseName(c.life));
            handle(Vec3(c.x, groundY, c.z), c.manual_index, colour, label);
        } else {
            std::snprintf(label, sizeof(label), "auto cell (%s)", phaseName(c.life));
            handle(Vec3(c.x, groundY, c.z), kAutoBase - i, colour, label);
        }
    }

    // Clicking an automatic cell converts the live automatic cells into
    // authored ones (atmosphere::freezeStormCells: the sky stays the same) and
    // selects the clicked one, so it can be dragged, edited and deleted.
    if (selected <= kAutoBase) {
        const int autoIndex = kAutoBase - selected;
        atmosphere::CloudState edited = clouds;
        const int first = static_cast<int>(edited.storms.cells.size());
        const int added = atmosphere::freezeStormCells(edited, world.getCloudTime(), wind, autoIndex);
        selected = kNone;
        if (added > 0) {
            apply(edited);
            if (error.empty()) selected = first;
        } else {
            error = "authored cell limit reached (8)";
        }
        return;
    }

    if (viewportInput && ImGui::IsMouseClicked(0) && !overHandle &&
        !ImGuizmo::IsOver() && !ImGuizmo::IsUsing()) {
        selected = kNone;
    }
    if (selected == kNone) return;

    // The drag handle sits where the item is drawn NOW; a drifting authored
    // cell stores its birth position, so the drift is taken back off.
    float drift = 0.0f;
    Vec3 at(storms.center_m.x, groundY, storms.center_m.z);
    if (selected >= 0) {
        const atmosphere::StormCell& cell = storms.cells[selected];
        bool alive = false;
        for (int i = 0; i < n; ++i) {
            if (cells[i].manual_index == selected) {
                at = Vec3(cells[i].x, groundY, cells[i].z);
                alive = true;
            }
        }
        if (!alive) at = Vec3(cell.position_m.x, groundY, cell.position_m.z);
        drift = alive && cell.lifetime_s > 0.0f ? world.getCloudTime() - cell.birth_s : 0.0f;
    }
    float view[16] = {r.x, u.x, -f.x, 0, r.y, u.y, -f.y, 0, r.z, u.z, -f.z, 0,
                      -r.dot(cam.lookfrom), -u.dot(cam.lookfrom), f.dot(cam.lookfrom), 1};
    float projection[16]{};
    const float zFar = 1.0e6f, zNear = 0.1f;
    if (ortho) {
        projection[0] = 2 / (cam.ortho_height * aspect);
        projection[5] = 2 / cam.ortho_height;
        projection[10] = -2 / (zFar - zNear);
        projection[14] = -(zFar + zNear) / (zFar - zNear);
        projection[15] = 1;
    } else {
        projection[0] = 1 / (aspect * tangent);
        projection[5] = 1 / tangent;
        projection[10] = -(zFar + zNear) / (zFar - zNear);
        projection[11] = -1;
        projection[14] = -2 * zFar * zNear / (zFar - zNear);
    }
    float matrix[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, at.x, at.y, at.z, 1};
    ImGuizmo::PushID(8272);
    ImGuizmo::SetDrawlist(draw);
    ImGuizmo::SetOrthographic(ortho);
    ImGuizmo::SetRect(0, 0, size.x, size.y);
    const bool changed = ImGuizmo::Manipulate(view, projection,
                                              ImGuizmo::TRANSLATE_X | ImGuizmo::TRANSLATE_Z,
                                              ImGuizmo::WORLD, matrix);
    if (ImGuizmo::IsOver() || ImGuizmo::IsUsing()) hit = true;
    ImGuizmo::PopID();
    if (changed) {
        atmosphere::CloudState edited = clouds;
        if (selected == kDomain) {
            edited.storms.center_m.x = matrix[12];
            edited.storms.center_m.z = matrix[14];
        } else {
            atmosphere::StormCell& cell = edited.storms.cells[selected];
            cell.position_m.x = matrix[12] - wind.x * drift;
            cell.position_m.z = matrix[14] - wind.z * drift;
        }
        apply(edited);
    }
    if (!error.empty()) {
        ImVec2 s;
        if (project(at, s)) draw->AddText(ImVec2(s.x + 10, s.y + 8), IM_COL32(255, 100, 80, 255), error.c_str());
    }
}

} // namespace StormGizmoUI
