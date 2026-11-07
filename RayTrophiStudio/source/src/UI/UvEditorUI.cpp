// SPDX-License-Identifier: MIT
#include "UI/UvEditorUI.h"
#include "UI/UvTexturePreview.h"

#include "UV/UvService.h"
#include "Texture.h"
#include "imgui.h"
#include "ui_modern.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace UvEditorUI {
namespace {

// Use the established public UIWidgets API. The UV commit is independent of
// the other ongoing inspector refactor and its unpublished helper headers.
namespace Controls {

bool button(const char* label, ImVec2 size = ImVec2(0, 0)) {
    return UIWidgets::SecondaryButton(label, size);
}

float actionWidth(int columns) {
    return (std::max)(1.0f, (ImGui::GetContentRegionAvail().x -
        ImGui::GetStyle().ItemSpacing.x * (columns - 1)) / columns);
}

void heading(const char* label) {
    ImGui::Spacing();
    ImGui::TextDisabled("%s", label);
}

bool beginGroup(const char* label, bool default_open = true) {
    return ImGui::CollapsingHeader(label, default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0);
}

void endGroup() {
    ImGui::Spacing();
}

class Row {
public:
    Row(const char* id, const char* label) {
        ImGui::PushID(id);
        const float available = ImGui::GetContentRegionAvail().x;
        const float start = ImGui::GetCursorPosX();
        if (available > ImGui::GetFontSize() * 24.0f) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(label);
            ImGui::SameLine(start + available * 0.4f);
        } else {
            ImGui::TextUnformatted(label);
        }
        ImGui::SetNextItemWidth(-1);
    }
    ~Row() {
        ImGui::PopID();
    }
    Row(const Row&) = delete;
    Row& operator=(const Row&) = delete;
};

} // namespace Controls

struct State {
    bool visible = false;
    bool dock_next = false;
    bool refresh = true;
    bool checker = true;
    bool show_tools = true;
    bool frame_next = true;
    std::string object;
    int mesh = 0;
    UV::Layout layout;
    std::set<uint32_t> selected;
    std::string error;
    std::string status;
    ImVec2 center{0.5f, 0.5f};
    float zoom = 240.0f;
    float offset[2] = {0, 0};
    float scale[2] = {1, 1};
    float pivot[2] = {0.5f, 0.5f};
    float degrees = 0;
    float tools_width_em = 22.0f;
    float tools_height_em = 16.0f;
    int resolution = 2048;
    int padding = 8;
    uint64_t scene_revision = 0;
    int corner_face = 0;
    float corner_uv[3][2] = {};
    int tool = 1; // Select, move, rotate, scale.
    bool gesture = false;
    bool box = false;
    bool subtract = false;
    bool extend = false;
    ImVec2 start_mouse;
    UV::Layout gesture_layout;
    std::set<uint32_t> gesture_selection;
    float gesture_offset[2] = {};
    float gesture_scale = 1;
    float gesture_degrees = 0;
    Vec2 gesture_pivot;
    std::string background;
    float background_opacity = 0.8f;
    std::vector<UV::PreviewTexture> textures;
    bool texture_refresh = true;
    std::string preview_error;
    int selection_mode = 0; // Island, UV vertex, UV edge.
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    std::set<uint32_t> used_vertices;
    std::vector<uint32_t> gesture_vertices;
    double next_texture_poll = 0;
    bool background_auto = true;
};

State state;

void refreshTextureInventory() {
    // Material slot assignment does not necessarily change geometry generation.
    // Poll only metadata; never rebuild UV selection or cancel an active gesture.
    const double now = ImGui::GetTime();
    if (now < state.next_texture_poll) {
        return;
    }
    state.next_texture_poll = now + 0.25;
    state.textures = UV::previewTextures(state.object, state.mesh);
    const bool present = std::any_of(state.textures.begin(), state.textures.end(),
        [](const auto& item) { return item.label == state.background; });
    if (!present) {
        state.background.clear();
        if (state.background_auto && !state.textures.empty()) {
            state.background = state.textures.front().label;
        }
    }
}

const char* selectionName() {
    return state.selection_mode == 0 ? "islands" :
        (state.selection_mode == 1 ? "vertices" : "edges");
}

std::vector<uint32_t> selectedVertices() {
    std::set<uint32_t> result;
    if (state.selection_mode == 0) {
        for (size_t face = 0; face < state.layout.islands.size(); ++face) {
            if (state.selected.count(state.layout.islands[face])) {
                for (size_t k = 0; k < 3; ++k) {
                    result.insert(state.layout.indices[face * 3 + k]);
                }
            }
        }
    } else if (state.selection_mode == 1) {
        result = state.selected;
    } else {
        for (uint32_t edge : state.selected) {
            if (edge < state.edges.size()) {
                result.insert(state.edges[edge].first);
                result.insert(state.edges[edge].second);
            }
        }
    }
    return {result.begin(), result.end()};
}

void selectAll() {
    state.selected.clear();
    if (state.selection_mode == 1) {
        state.selected = state.used_vertices;
        return;
    }
    const size_t count = state.selection_mode == 0 ? state.layout.island_count :
        state.edges.size();
    for (uint32_t id = 0; id < count; ++id) {
        state.selected.insert(id);
    }
}

void cancelGesture() {
    if (state.gesture && !state.box) {
        state.layout = std::move(state.gesture_layout);
    } else if (state.gesture) {
        state.selected = state.gesture_selection;
    }
    state.gesture = false;
}

void startGesture(ImVec2 mouse, bool box) {
    state.gesture = true;
    state.box = box;
    state.start_mouse = mouse;
    state.gesture_layout = state.layout;
    state.gesture_selection = state.selected;
    state.gesture_vertices = selectedVertices();
    state.extend = ImGui::GetIO().KeyShift;
    state.subtract = ImGui::GetIO().KeyCtrl;
    state.gesture_offset[0] = state.gesture_offset[1] = 0;
    state.gesture_scale = 1;
    state.gesture_degrees = 0;
    Vec2 lo(1.0e30f, 1.0e30f), hi(-1.0e30f, -1.0e30f);
    for (uint32_t vertex : state.gesture_vertices) {
        const Vec2 uv = state.layout.coordinates[vertex];
        lo.u = (std::min)(lo.u, uv.u);
        lo.v = (std::min)(lo.v, uv.v);
        hi.u = (std::max)(hi.u, uv.u);
        hi.v = (std::max)(hi.v, uv.v);
    }
    state.gesture_pivot = Vec2((lo.u + hi.u) * 0.5f, (lo.v + hi.v) * 0.5f);
}

void commitGesture() {
    state.status.clear();
    nlohmann::json params{
        {"object", state.object}, {"mesh", state.mesh},
        {"revision", state.gesture_layout.revision},
        {"allow_texture_change", true},
        {"offset", {state.gesture_offset[0], state.gesture_offset[1]}},
        {"scale", {state.gesture_scale, state.gesture_scale}},
        {"pivot", {state.gesture_pivot.u, state.gesture_pivot.v}},
        {"degrees", state.gesture_degrees}};
    if (state.selection_mode == 0) {
        params["islands"] = std::vector<uint32_t>(state.gesture_selection.begin(),
                                                 state.gesture_selection.end());
    } else {
        params["vertices"] = state.gesture_vertices;
    }
    const auto result = UV::request("transform", params);
    cancelGesture();
    if (result.contains("__error")) {
        state.error = result.at("__error").get<std::string>();
    }
    state.refresh = true;
}

void textureBackground(ImVec2 origin, ImVec2 size) {
    if (state.background.empty()) {
        state.preview_error.clear();
        UvTexturePreview::release();
        return;
    }
    const auto found = std::find_if(state.textures.begin(), state.textures.end(),
        [](const auto& item) { return item.label == state.background; });
    if (found == state.textures.end()) {
        UvTexturePreview::release();
        state.preview_error = "Background channel changed. Refresh and choose a texture.";
        return;
    }
    auto point = [&](float u, float v) {
        return ImVec2(origin.x + size.x * 0.5f + (u - state.center.x) * state.zoom,
                      origin.y + size.y * 0.5f - (v - state.center.y) * state.zoom);
    };
    UvTexturePreview::draw(found->texture, point(0, 1), point(1, 0),
        state.background_opacity, state.texture_refresh, state.preview_error);
    state.texture_refresh = false;
}

void readFaceCorners() {
    if (state.layout.indices.empty()) {
        return;
    }
    state.corner_face = std::clamp(state.corner_face, 0,
        static_cast<int>(state.layout.indices.size() / 3) - 1);
    for (size_t corner = 0; corner < 3; ++corner) {
        const Vec2 uv = state.layout.coordinates[
            state.layout.indices[size_t(state.corner_face) * 3 + corner]];
        state.corner_uv[corner][0] = uv.u;
        state.corner_uv[corner][1] = uv.v;
    }
}

void refresh() {
    cancelGesture();
    UV::Layout layout;
    if (UV::inspect(state.object, state.mesh, layout, state.error)) {
        const bool reload_corners = layout.revision != state.layout.revision;
        if (layout.indices != state.layout.indices || layout.islands != state.layout.islands) {
            state.selected.clear();
        }
        state.layout = std::move(layout);
        state.used_vertices = std::set<uint32_t>(state.layout.indices.begin(),
                                                state.layout.indices.end());
        std::set<std::pair<uint32_t, uint32_t>> edges;
        for (size_t face = 0; face < state.layout.indices.size(); face += 3) {
            for (size_t k = 0; k < 3; ++k) {
                const uint32_t a = state.layout.indices[face + k];
                const uint32_t b = state.layout.indices[face + (k + 1) % 3];
                if (a != b) {
                    edges.emplace((std::min)(a, b), (std::max)(a, b));
                }
            }
        }
        state.edges.assign(edges.begin(), edges.end());
        state.next_texture_poll = 0;
        if (reload_corners) {
            readFaceCorners();
        }
        state.error.clear();
    } else {
        state.layout = {};
        state.selected.clear();
        state.textures.clear();
        state.edges.clear();
        state.used_vertices.clear();
    }
    state.refresh = false;
}

void apply(const char* operation) {
    nlohmann::json params{{"object", state.object}, {"mesh", state.mesh},
        {"revision", state.layout.revision},
        {"allow_texture_change", true}};
    if (std::string(operation) == "set_corners") {
        if (state.layout.indices.size() > 60000 || state.corner_face < 0 ||
            size_t(state.corner_face) >= state.layout.indices.size() / 3) {
            state.error = "Choose a valid triangle in a mesh of at most 20000 triangles.";
            return;
        }
        auto corners = nlohmann::json::array();
        for (uint32_t vertex : state.layout.indices) {
            const Vec2 uv = state.layout.coordinates[vertex];
            corners.push_back({uv.u, uv.v});
        }
        for (size_t corner = 0; corner < 3; ++corner) {
            corners[size_t(state.corner_face) * 3 + corner] =
                {state.corner_uv[corner][0], state.corner_uv[corner][1]};
        }
        params["corners"] = std::move(corners);
    }
    if (std::string(operation) == "transform" || std::string(operation) == "normalize" ||
        std::string(operation) == "flip_u" || std::string(operation) == "flip_v" ||
        std::string(operation) == "swap_uv") {
        if (state.selection_mode == 0) {
            params["islands"] = std::vector<uint32_t>(state.selected.begin(), state.selected.end());
        } else {
            params["vertices"] = selectedVertices();
        }
        params["offset"] = {state.offset[0], state.offset[1]};
        params["scale"] = {state.scale[0], state.scale[1]};
        params["pivot"] = {state.pivot[0], state.pivot[1]};
        params["degrees"] = state.degrees;
    } else if (std::string(operation) == "pack" || std::string(operation) == "unwrap") {
        params["resolution"] = state.resolution;
        params["padding"] = state.padding;
    }
    const auto result = UV::request(operation, params);
    state.status.clear();
    if (result.contains("__error")) {
        state.error = result.at("__error").get<std::string>();
    } else {
        refresh();
        const uint32_t preserved = result.value("atlas_preserved_faces", 0u);
        if (preserved > 0) {
            state.status = "Atlas complete: " + std::to_string(preserved) +
                " degenerate triangles kept their previous UVs.";
        }
    }
}

void frame(const ImVec2& size) {
    if (state.layout.coordinates.empty()) {
        return;
    }
    float min_u = 0, min_v = 0, max_u = 1, max_v = 1;
    for (const Vec2& uv : state.layout.coordinates) {
        min_u = (std::min)(min_u, uv.u);
        min_v = (std::min)(min_v, uv.v);
        max_u = (std::max)(max_u, uv.u);
        max_v = (std::max)(max_v, uv.v);
    }
    state.center = ImVec2((min_u + max_u) * 0.5f, (min_v + max_v) * 0.5f);
    state.zoom = (std::max)(1.0f, (std::min)(size.x / (max_u - min_u),
                                          size.y / (max_v - min_v)) * 0.85f);
}

bool inside(ImVec2 p, ImVec2 a, ImVec2 b, ImVec2 c) {
    auto cross = [](ImVec2 x, ImVec2 y, ImVec2 z) {
        return (y.x - x.x) * (z.y - x.y) - (y.y - x.y) * (z.x - x.x);
    };
    if (std::abs(cross(a, b, c)) < 1.0e-5f) {
        return false;
    }
    const float x = cross(a, b, p), y = cross(b, c, p), z = cross(c, a, p);
    return !((x < 0 || y < 0 || z < 0) && (x > 0 || y > 0 || z > 0));
}

void canvas() {
    ImVec2 size = ImGui::GetContentRegionAvail();
    size.x = (std::max)(size.x, 1.0f);
    size.y = (std::max)(size.y, 1.0f);
    if (state.frame_next) {
        frame(size);
        state.frame_next = false;
    }
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 end(origin.x + size.x, origin.y + size.y);
    ImGui::InvisibleButton("##uv_canvas", size,
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    ImGuiIO& io = ImGui::GetIO();
    if (!state.gesture && hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        state.center.x -= io.MouseDelta.x / state.zoom;
        state.center.y += io.MouseDelta.y / state.zoom;
    }
    if (!state.gesture && hovered && io.MouseWheel != 0) {
        const float old = state.zoom;
        state.zoom = std::clamp(old * std::pow(1.2f, io.MouseWheel), 1.0f, 100000.0f);
        const float x = io.MousePos.x - origin.x - size.x * 0.5f;
        const float y = io.MousePos.y - origin.y - size.y * 0.5f;
        state.center.x += x / old - x / state.zoom;
        state.center.y -= y / old - y / state.zoom;
    }
    if (!state.gesture && !io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_F)) {
        frame(size);
    }
    auto point = [&](const Vec2& uv) {
        return ImVec2(origin.x + size.x * 0.5f + (uv.u - state.center.x) * state.zoom,
                      origin.y + size.y * 0.5f - (uv.v - state.center.y) * state.zoom);
    };
    if (state.gesture && ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        cancelGesture();
    }
    if (!state.gesture && !io.WantTextInput) {
        int shortcut = -1;
        if (ImGui::IsKeyPressed(ImGuiKey_G)) {
            shortcut = 1;
        } else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
            shortcut = 2;
        } else if (ImGui::IsKeyPressed(ImGuiKey_S)) {
            shortcut = 3;
        }
        if (shortcut >= 0 && !io.KeyCtrl && !io.KeyAlt) {
            state.tool = shortcut;
        }
    }
    if (!state.gesture && !io.WantTextInput && !io.KeyCtrl &&
        ImGui::IsKeyPressed(ImGuiKey_A)) {
        if (io.KeyAlt) {
            state.selected.clear();
        } else {
            selectAll();
        }
    }
    if (state.gesture && !state.box) {
        const float dx = io.MousePos.x - state.start_mouse.x;
        const float dy = io.MousePos.y - state.start_mouse.y;
        state.gesture_offset[0] = state.tool == 1 ? dx / state.zoom : 0;
        state.gesture_offset[1] = state.tool == 1 ? -dy / state.zoom : 0;
        const ImVec2 pivot = point(state.gesture_pivot);
        if (state.tool == 2) {
            const float a = std::atan2(pivot.y - state.start_mouse.y,
                                      state.start_mouse.x - pivot.x);
            const float b = std::atan2(pivot.y - io.MousePos.y, io.MousePos.x - pivot.x);
            state.gesture_degrees = (b - a) * 180.0f / 3.141592654f;
        }
        if (state.tool == 3) {
            state.gesture_scale = std::exp(std::clamp((dx - dy) / 150.0f, -8.0f, 8.0f));
        }
        state.layout = state.gesture_layout;
        std::string error;
        const auto islands = state.selection_mode == 0
            ? std::vector<uint32_t>(state.gesture_selection.begin(), state.gesture_selection.end())
            : std::vector<uint32_t>{};
        if (!UV::transform(state.layout,
                islands,
                Vec2(state.gesture_offset[0], state.gesture_offset[1]),
                Vec2(state.gesture_scale, state.gesture_scale), state.gesture_pivot,
                state.gesture_degrees, false, error, false,
                state.selection_mode == 0 ? nullptr : &state.gesture_vertices)) {
            cancelGesture();
            state.error = error;
        } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            commitGesture();
        }
    }
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->PushClipRect(origin, end, true);
    draw->AddRectFilled(origin, end, ImGui::GetColorU32(ImGuiCol_WindowBg));
    if (state.checker) {
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 16; ++x) {
                const ImVec2 a = point(Vec2(float(x) / 16, float(y + 1) / 16));
                const ImVec2 b = point(Vec2(float(x + 1) / 16, float(y) / 16));
                draw->AddRectFilled(a, b, (x + y) % 2 ? IM_COL32(51, 54, 59, 255)
                                                                    : IM_COL32(34, 37, 42, 255));
            }
        }
    }
    textureBackground(origin, size);
    draw->AddRect(point(Vec2(0, 1)), point(Vec2(1, 0)),
                  ImGui::GetColorU32(ImGuiCol_Border), 0, 0, 2.0f);
    const bool clicked = !state.gesture && hovered &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    uint32_t hit = UINT32_MAX;
    int hit_face = -1;
    for (size_t face = 0; face < state.layout.islands.size(); ++face) {
        ImVec2 p[3];
        for (size_t k = 0; k < 3; ++k) {
            p[k] = point(state.layout.coordinates[state.layout.indices[face * 3 + k]]);
        }
        if ((std::max)({p[0].x, p[1].x, p[2].x}) < origin.x ||
            (std::min)({p[0].x, p[1].x, p[2].x}) > end.x ||
            (std::max)({p[0].y, p[1].y, p[2].y}) < origin.y ||
            (std::min)({p[0].y, p[1].y, p[2].y}) > end.y) {
            continue;
        }
        const uint32_t island = state.layout.islands[face];
        const bool selected = state.selection_mode == 0 && state.selected.count(island) != 0;
        float r, g, b;
        ImGui::ColorConvertHSVtoRGB(std::fmod(float(island) * 0.618034f, 1.0f),
                                   0.45f, 0.85f, r, g, b);
        const ImU32 color = selected ? ImGui::GetColorU32(ImGuiCol_PlotHistogram)
            : ImGui::GetColorU32(ImVec4(r, g, b, 0.75f));
        if (selected) {
            ImVec4 fill = ImGui::GetStyleColorVec4(ImGuiCol_Header);
            fill.w = 0.16f;
            draw->AddTriangleFilled(p[0], p[1], p[2], ImGui::GetColorU32(fill));
        }
        draw->AddTriangle(p[0], p[1], p[2], color, selected ? 1.5f : 1.0f);
        if (clicked && inside(io.MousePos, p[0], p[1], p[2])) {
            if (state.selection_mode == 0) {
                hit = island;
            }
            hit_face = static_cast<int>(face);
        }
    }
    float nearest = 8.0f * 8.0f;
    if (state.selection_mode == 1) {
        for (uint32_t vertex : state.used_vertices) {
            const ImVec2 p = point(state.layout.coordinates[vertex]);
            const bool selected = state.selected.count(vertex) != 0;
            draw->AddCircleFilled(p, selected ? 4.0f : 2.5f,
                ImGui::GetColorU32(selected ? ImGuiCol_PlotHistogram : ImGuiCol_Text));
            const float dx = io.MousePos.x - p.x, dy = io.MousePos.y - p.y;
            if (clicked && dx * dx + dy * dy < nearest) {
                nearest = dx * dx + dy * dy;
                hit = vertex;
            }
        }
    } else if (state.selection_mode == 2) {
        for (uint32_t edge = 0; edge < state.edges.size(); ++edge) {
            const auto endpoints = state.edges[edge];
            const ImVec2 a = point(state.layout.coordinates[endpoints.first]);
            const ImVec2 b = point(state.layout.coordinates[endpoints.second]);
            const bool selected = state.selected.count(edge) != 0;
            if (selected) {
                draw->AddLine(a, b, ImGui::GetColorU32(ImGuiCol_PlotHistogram), 3.0f);
            }
            const float dx = b.x - a.x, dy = b.y - a.y;
            const float length = dx * dx + dy * dy;
            const float t = length > 0 ? std::clamp(((io.MousePos.x - a.x) * dx +
                (io.MousePos.y - a.y) * dy) / length, 0.0f, 1.0f) : 0;
            const float x = io.MousePos.x - a.x - t * dx;
            const float y = io.MousePos.y - a.y - t * dy;
            if (clicked && x * x + y * y < nearest) {
                nearest = x * x + y * y;
                hit = edge;
            }
        }
    }
    if (clicked) {
        const bool manipulate = state.tool != 0 && hit != UINT32_MAX &&
            !io.KeyShift && !io.KeyCtrl;
        if (manipulate && !state.selected.count(hit)) {
            state.selected.clear();
            state.selected.insert(hit);
        }
        if (manipulate) {
            startGesture(io.MousePos, false);
        } else if (state.tool == 0 || hit == UINT32_MAX) {
            startGesture(io.MousePos, true);
        }
        if (hit_face >= 0) {
            state.corner_face = hit_face;
            readFaceCorners();
        }
        if (!manipulate && !io.KeyShift && !io.KeyCtrl) {
            state.selected.clear();
        }
        if (hit != UINT32_MAX) {
            if (io.KeyCtrl) {
                state.selected.erase(hit);
            }
            else state.selected.insert(hit);
        }
    }
    if (state.gesture && state.box) {
        const ImVec2 a((std::min)(state.start_mouse.x, io.MousePos.x),
                       (std::min)(state.start_mouse.y, io.MousePos.y));
        const ImVec2 b((std::max)(state.start_mouse.x, io.MousePos.x),
                       (std::max)(state.start_mouse.y, io.MousePos.y));
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            draw->AddRectFilled(a, b, IM_COL32(80, 150, 255, 30));
            draw->AddRect(a, b, IM_COL32(100, 180, 255, 255));
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            if (std::abs(b.x - a.x) + std::abs(b.y - a.y) > 6) {
                state.selected = (state.extend || state.subtract)
                    ? state.gesture_selection : std::set<uint32_t>{};
                auto enclosedPoint = [&](uint32_t vertex) {
                    const ImVec2 p = point(state.layout.coordinates[vertex]);
                    return p.x >= a.x && p.x <= b.x && p.y >= a.y && p.y <= b.y;
                };
                auto choose = [&](uint32_t id) {
                    if (state.subtract) {
                        state.selected.erase(id);
                    } else {
                        state.selected.insert(id);
                    }
                };
                if (state.selection_mode == 1) {
                    for (uint32_t vertex : state.used_vertices) {
                        if (enclosedPoint(vertex)) {
                            choose(vertex);
                        }
                    }
                } else if (state.selection_mode == 2) {
                    for (uint32_t edge = 0; edge < state.edges.size(); ++edge) {
                        if (enclosedPoint(state.edges[edge].first) &&
                            enclosedPoint(state.edges[edge].second)) {
                            choose(edge);
                        }
                    }
                }
                for (size_t face = 0; state.selection_mode == 0 &&
                     face < state.layout.islands.size(); ++face) {
                    bool enclosed = false;
                    for (size_t k = 0; k < 3; ++k) {
                        const ImVec2 p = point(state.layout.coordinates[
                            state.layout.indices[face * 3 + k]]);
                        enclosed |= p.x >= a.x && p.x <= b.x && p.y >= a.y && p.y <= b.y;
                    }
                    if (enclosed) {
                        choose(state.layout.islands[face]);
                    }
                }
            }
            state.gesture = false;
        }
    }
    draw->PopClipRect();
}

void tools() {
    ImGui::BeginDisabled(state.gesture);
    if (ImGui::Combo("Selection", &state.selection_mode, "Island\0Vertex\0Edge\0")) {
        state.selected.clear();
    }
    ImGui::Combo("Mouse tool", &state.tool, "Select / box\0Move (G)\0Rotate (R)\0Scale (S)\0");
    if (ImGui::BeginCombo("Background",
            state.background.empty() ? "Checker only" : state.background.c_str())) {
        if (ImGui::Selectable("Checker only", state.background.empty())) {
            state.background.clear();
            state.background_auto = false;
        }
        for (const auto& item : state.textures) {
            if (ImGui::Selectable(item.label.c_str(), state.background == item.label)) {
                state.background = item.label;
                state.background_auto = false;
            }
        }
        ImGui::EndCombo();
    }
    if (!state.background.empty()) {
        ImGui::SliderFloat("Texture opacity", &state.background_opacity, 0, 1);
        ImGui::TextDisabled("Native resolution / raw UV 0 tile");
        if (Controls::button("Refresh texture", ImVec2(-1, 0))) {
            state.texture_refresh = true;
        }
    }
    Controls::heading("Selection tools");
    const float selection_width = Controls::actionWidth(2);
    if (Controls::button("Select all", ImVec2(selection_width, 0))) {
        selectAll();
    }
    ImGui::SameLine();
    if (Controls::button("Clear", ImVec2(selection_width, 0))) {
        state.selected.clear();
    }
    if (state.selected.empty()) {
        ImGui::TextDisabled(state.selection_mode == 0 ? "Scope: all islands" : "Select UV elements");
    } else {
        ImGui::TextDisabled("Scope: %zu selected %s", state.selected.size(), selectionName());
    }
    ImGui::TextWrapped("UV edits change texture placement; painted images are not rebaked.");

    if (Controls::beginGroup("Transform")) {
        {
            Controls::Row row("uv_offset", "Offset U / V");
            ImGui::InputFloat2("##value", state.offset, "%.4f");
        }
        {
            Controls::Row row("uv_scale", "Scale U / V");
            ImGui::InputFloat2("##value", state.scale, "%.4f");
        }
        {
            Controls::Row row("uv_pivot", "Pivot U / V");
            ImGui::InputFloat2("##value", state.pivot, "%.4f");
        }
        {
            Controls::Row row("uv_rotation", "Rotation (deg)");
            ImGui::InputFloat("##value", &state.degrees, 0, 0, "%.2f");
        }
        ImGui::BeginDisabled(state.layout.coordinates.empty() ||
            (state.selection_mode != 0 && state.selected.empty()));
        if (Controls::button("Apply transform", ImVec2(-1, 0))) {
            apply("transform");
        }
        if (Controls::button("Normalize uniformly", ImVec2(-1, 0))) {
            apply("normalize");
        }
        if (Controls::button("Flip U", ImVec2(-1, 0))) {
            apply("flip_u");
        }
        if (Controls::button("Flip V", ImVec2(-1, 0))) {
            apply("flip_v");
        }
        if (Controls::button("Swap U / V", ImVec2(-1, 0))) {
            apply("swap_uv");
        }
        ImGui::EndDisabled();
        if (Controls::button("Reset transform fields", ImVec2(-1, 0))) {
            state.offset[0] = state.offset[1] = 0;
            state.scale[0] = state.scale[1] = 1;
            state.pivot[0] = state.pivot[1] = 0.5f;
            state.degrees = 0;
        }
        Controls::endGroup();
    }

    if (Controls::beginGroup("Triangle corners", false)) {
        ImGui::TextWrapped("Click a UV triangle or enter its index. Only these three corners change.");
        {
            Controls::Row row("uv_corner_face", "Triangle index");
            if (ImGui::InputInt("##value", &state.corner_face)) {
                readFaceCorners();
            }
        }
        const char* labels[] = {"Corner 0 U / V", "Corner 1 U / V", "Corner 2 U / V"};
        for (int corner = 0; corner < 3; ++corner) {
            Controls::Row row(labels[corner], labels[corner]);
            ImGui::InputFloat2("##value", state.corner_uv[corner], "%.5f");
        }
        ImGui::BeginDisabled(state.layout.indices.empty() ||
            state.layout.indices.size() > 60000);
        if (Controls::button("Apply triangle UVs", ImVec2(-1, 0))) {
            apply("set_corners");
        }
        ImGui::EndDisabled();
        ImGui::TextWrapped("Shared vertices split when necessary. The 3D surface stays in place.");
        Controls::endGroup();
    }

    if (Controls::beginGroup("Projection")) {
        ImGui::TextDisabled("Scope: entire mesh");
        ImGui::BeginDisabled(state.layout.coordinates.empty());
        if (Controls::button("Project from View", ImVec2(-1, 0))) {
            apply("project_view");
        }
        ImGui::EndDisabled();
        Controls::endGroup();
    }

    if (Controls::beginGroup("Atlas")) {
        ImGui::TextDisabled("Scope: entire mesh");
        {
            Controls::Row row("uv_resolution", "Resolution (px)");
            ImGui::InputInt("##value", &state.resolution, 128, 1024);
        }
        {
            Controls::Row row("uv_padding", "Padding (px)");
            ImGui::InputInt("##value", &state.padding);
        }
        ImGui::BeginDisabled(state.layout.coordinates.empty() ||
            !UV::xatlasAvailable());
        if (Controls::button("Auto unwrap", ImVec2(-1, 0))) {
            apply("unwrap");
        }
        if (Controls::button("Pack existing UVs", ImVec2(-1, 0))) {
            apply("pack");
        }
        ImGui::EndDisabled();
        if (!UV::xatlasAvailable()) {
            ImGui::TextWrapped("The UV atlas backend is unavailable in this build.");
        }
        Controls::endGroup();
    }
    ImGui::EndDisabled();
}

float resizeTools(const ImVec2& start, const ImVec2& space, bool horizontal,
                  float extent, float handle, float minimum, float maximum) {
    const float font = ImGui::GetFontSize();
    ImGui::SetCursorPos(horizontal ? ImVec2(start.x + extent, start.y)
                                   : ImVec2(start.x, start.y + extent));
    ImGui::InvisibleButton("##uv_tools_splitter",
        horizontal ? ImVec2(handle, space.y) : ImVec2(space.x, handle));
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    if (hovered || active) {
        ImGui::SetMouseCursor(horizontal ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
    }
    float& desired = horizontal ? state.tools_width_em : state.tools_height_em;
    if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
        const ImVec2 delta = ImGui::GetIO().MouseDelta;
        extent = std::clamp(extent + (horizontal ? delta.x : delta.y), minimum, maximum);
        desired = extent / font;
    }
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        desired = horizontal ? 22.0f : 16.0f;
        extent = std::clamp(desired * font, minimum, maximum);
    }
    if (hovered && !active) {
        ImGui::SetTooltip("Drag to resize tools. Double-click to reset.");
    }
    const ImVec2 a = ImGui::GetItemRectMin();
    const ImVec2 b = ImGui::GetItemRectMax();
    const ImU32 color = ImGui::GetColorU32(active ? ImGuiCol_SeparatorActive :
        (hovered ? ImGuiCol_SeparatorHovered : ImGuiCol_Separator));
    const ImVec2 middle((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    if (horizontal) {
        draw->AddLine(ImVec2(middle.x, a.y), ImVec2(middle.x, b.y), color);
    } else {
        draw->AddLine(ImVec2(a.x, middle.y), ImVec2(b.x, middle.y), color);
    }
    for (int dot = -1; dot <= 1; ++dot) {
        const float shift = dot * font * 0.4f;
        draw->AddCircleFilled(horizontal ? ImVec2(middle.x, middle.y + shift)
                                         : ImVec2(middle.x + shift, middle.y),
                              font * 0.08f, color);
    }
    return extent;
}

void panes() {
    // Splitter items consume layout height. Both children use the saved origin,
    // never a SameLine chain after the full-height splitter (layout guardrails).
    const ImVec2 start = ImGui::GetCursorPos();
    const ImVec2 space = ImGui::GetContentRegionAvail();
    const float font = ImGui::GetFontSize();
    const float handle = (std::max)(6.0f, font * 0.55f);
    if (space.x <= 1 || space.y <= 1) {
        return;
    }
    const bool horizontal = space.x >= font * 28.0f + handle;
    ImVec2 view_start = start;
    ImVec2 view_size = space;
    if (state.show_tools && space.y > handle + 2.0f) {
        const float available = horizontal ? space.x : space.y;
        const float minimum = horizontal ? font * 16.0f :
            (std::min)(font * 6.0f, (std::max)(1.0f, available - handle) * 0.4f);
        const float canvas_minimum = horizontal ? font * 12.0f : minimum;
        const float maximum = (std::max)(minimum, available - handle - canvas_minimum);
        const float desired = horizontal ? state.tools_width_em : state.tools_height_em;
        const float extent = resizeTools(start, space, horizontal,
            std::clamp(desired * font, minimum, maximum), handle, minimum, maximum);
        ImGui::SetCursorPos(start);
        ImGui::BeginChild("##uv_tools",
            horizontal ? ImVec2(extent, space.y) : ImVec2(space.x, extent), true);
        tools();
        ImGui::EndChild();
        if (horizontal) {
            view_start.x += extent + handle;
            view_size.x -= extent + handle;
        } else {
            view_start.y += extent + handle;
            view_size.y -= extent + handle;
        }
    }
    ImGui::SetCursorPos(view_start);
    ImGui::BeginChild("##uv_view",
        ImVec2((std::max)(1.0f, view_size.x), (std::max)(1.0f, view_size.y)), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    canvas();
    ImGui::EndChild();
}

void nextToolbarItem(float width_em) {
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    const float next = ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x;
    if (next + width_em * ImGui::GetFontSize() <= right) {
        ImGui::SameLine();
    }
}

} // namespace

bool capturesShortcuts() {
    return state.visible;
}

void open(const std::string& object) {
    cancelGesture();
    state.visible = true;
    state.dock_next = true;
    state.refresh = true;
    state.status.clear();
    if (!object.empty()) {
        if (state.object != object) {
            state.selected.clear();
            state.background.clear();
            state.background_auto = true;
        }
        state.object = object;
        state.mesh = 0;
        state.frame_next = true;
    }
}

void draw(UIContext&, const std::function<void(const char*)>& dock_to_bottom) {
    if (!state.visible) {
        cancelGesture();
        UvTexturePreview::release();
        return;
    }
    const uint64_t revision = UV::sceneRevision();
    if (revision != state.scene_revision) {
        state.refresh = true;
        state.scene_revision = revision;
    }
    if (state.dock_next && dock_to_bottom) {
        dock_to_bottom("UV Editor");
        state.dock_next = false;
    }
    ImGui::SetNextWindowSize(ImVec2(900, 420), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("UV Editor", &state.visible)) {
        ImGui::End();
        return;
    }
    const std::string label = state.object.empty() ? "Choose flat mesh" :
        state.object + " / " + std::to_string(state.mesh);
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::BeginCombo("##uv_target", label.c_str())) {
        const auto targets = UV::request("targets", nlohmann::json::object());
        if (targets.is_array()) {
            for (const auto& target : targets) {
                const std::string object = target.at("object").get<std::string>();
                const int mesh = target.at("mesh").get<int>();
                const std::string name = object + " / " + std::to_string(mesh);
                if (ImGui::Selectable(name.c_str(), state.object == object && state.mesh == mesh)) {
                    state.selected.clear();
                    state.status.clear();
                    if (state.object != object || state.mesh != mesh) {
                        state.background.clear();
                        state.background_auto = true;
                    }
                    state.object = object;
                    state.mesh = mesh;
                    state.refresh = true;
                    state.frame_next = true;
                }
            }
        }
        ImGui::EndCombo();
    }
    if (Controls::button("Refresh")) {
        state.refresh = true;
        state.texture_refresh = true;
    }
    nextToolbarItem(4.0f);
    if (Controls::button("Frame")) {
        state.frame_next = true;
    }
    nextToolbarItem(6.0f);
    ImGui::Checkbox("Checker", &state.checker);
    nextToolbarItem(4.5f);
    ImGui::Checkbox("Tools", &state.show_tools);
    if (state.refresh && !state.object.empty()) {
        refresh();
    }
    refreshTextureInventory();
    ImGui::TextWrapped("UV 0 | %zu faces | %u islands | %zu selected %s",
        state.layout.indices.size() / 3, state.layout.island_count, state.selected.size(),
        selectionName());
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("Click / box select; A all; Alt+A clear; Shift add; Ctrl remove; G/R/S tool; "
        "drag selection; Esc cancel; MMB pan; wheel zoom; F frame");
    ImGui::PopStyleColor();
    if (!state.error.empty()) {
        ImGui::TextWrapped("%s", state.error.c_str());
    }
    if (!state.status.empty()) {
        ImGui::TextWrapped("%s", state.status.c_str());
    }
    if (!state.preview_error.empty()) {
        ImGui::TextWrapped("%s", state.preview_error.c_str());
    }
    panes();
    ImGui::End();
}

} // namespace UvEditorUI
