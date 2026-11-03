// SPDX-License-Identifier: MIT
#include "UI/UvEditorUI.h"

#include "UV/UvService.h"
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

bool beginGroup(const char* label) {
    return ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen);
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
    bool allow_texture_change = false;
    std::string object;
    int mesh = 0;
    UV::Layout layout;
    std::set<uint32_t> selected;
    std::string error;
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
};

State state;

void refresh() {
    UV::Layout layout;
    if (UV::inspect(state.object, state.mesh, layout, state.error)) {
        if (layout.indices != state.layout.indices || layout.islands != state.layout.islands) {
            state.selected.clear();
        }
        state.layout = std::move(layout);
        state.error.clear();
    } else {
        state.layout = {};
        state.selected.clear();
    }
    state.refresh = false;
}

void apply(const char* operation) {
    nlohmann::json params{{"object", state.object}, {"mesh", state.mesh},
        {"revision", state.layout.revision},
        {"allow_texture_change", state.allow_texture_change}};
    if (std::string(operation) == "transform" || std::string(operation) == "normalize" ||
        std::string(operation) == "flip_u" || std::string(operation) == "flip_v" ||
        std::string(operation) == "swap_uv") {
        params["islands"] = std::vector<uint32_t>(state.selected.begin(), state.selected.end());
        params["offset"] = {state.offset[0], state.offset[1]};
        params["scale"] = {state.scale[0], state.scale[1]};
        params["pivot"] = {state.pivot[0], state.pivot[1]};
        params["degrees"] = state.degrees;
    } else {
        params["resolution"] = state.resolution;
        params["padding"] = state.padding;
    }
    const auto result = UV::request(operation, params);
    if (result.contains("__error")) {
        state.error = result.at("__error").get<std::string>();
    } else {
        refresh();
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
    if (hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        state.center.x -= io.MouseDelta.x / state.zoom;
        state.center.y += io.MouseDelta.y / state.zoom;
    }
    if (hovered && io.MouseWheel != 0) {
        const float old = state.zoom;
        state.zoom = std::clamp(old * std::pow(1.2f, io.MouseWheel), 1.0f, 100000.0f);
        const float x = io.MousePos.x - origin.x - size.x * 0.5f;
        const float y = io.MousePos.y - origin.y - size.y * 0.5f;
        state.center.x += x / old - x / state.zoom;
        state.center.y -= y / old - y / state.zoom;
    }
    if (hovered && ImGui::IsKeyPressed(ImGuiKey_F)) {
        frame(size);
    }
    auto point = [&](const Vec2& uv) {
        return ImVec2(origin.x + size.x * 0.5f + (uv.u - state.center.x) * state.zoom,
                      origin.y + size.y * 0.5f - (uv.v - state.center.y) * state.zoom);
    };
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
    draw->AddRect(point(Vec2(0, 1)), point(Vec2(1, 0)),
                  ImGui::GetColorU32(ImGuiCol_Border), 0, 0, 2.0f);
    const bool clicked = hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    uint32_t hit = UINT32_MAX;
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
        const bool selected = state.selected.count(island) != 0;
        float r, g, b;
        ImGui::ColorConvertHSVtoRGB(std::fmod(float(island) * 0.618034f, 1.0f),
                                   0.45f, 0.85f, r, g, b);
        const ImU32 color = selected ? ImGui::GetColorU32(ImGuiCol_PlotHistogram)
            : ImGui::GetColorU32(ImVec4(r, g, b, 0.75f));
        if (selected) draw->AddTriangleFilled(p[0], p[1], p[2],
            ImGui::GetColorU32(ImGuiCol_Header));
        draw->AddTriangle(p[0], p[1], p[2], color, selected ? 1.5f : 1.0f);
        if (clicked && inside(io.MousePos, p[0], p[1], p[2])) {
            hit = island;
        }
    }
    if (clicked) {
        if (!io.KeyShift && !io.KeyCtrl) {
            state.selected.clear();
        }
        if (hit != UINT32_MAX) {
            if (io.KeyCtrl) {
                state.selected.erase(hit);
            }
            else state.selected.insert(hit);
        }
    }
    draw->PopClipRect();
}

void tools() {
    Controls::heading("Island tools");
    const float selection_width = Controls::actionWidth(2);
    if (Controls::button("Select all", ImVec2(selection_width, 0))) {
        for (uint32_t island = 0; island < state.layout.island_count; ++island) {
            state.selected.insert(island);
        }
    }
    ImGui::SameLine();
    if (Controls::button("Clear", ImVec2(selection_width, 0))) {
        state.selected.clear();
    }
    if (state.selected.empty()) {
        ImGui::TextDisabled("Scope: all islands");
    } else {
        ImGui::TextDisabled("Scope: %zu selected islands", state.selected.size());
    }
    ImGui::Checkbox("Allow texture changes", &state.allow_texture_change);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("UV edits change texture placement. Painted textures are not rebaked.");
    }

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
        ImGui::BeginDisabled(state.layout.coordinates.empty() || !state.allow_texture_change);
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

    if (Controls::beginGroup("Projection")) {
        ImGui::TextDisabled("Scope: entire mesh");
        ImGui::BeginDisabled(state.layout.coordinates.empty() || !state.allow_texture_change);
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
            !state.allow_texture_change || !UV::xatlasAvailable());
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

void open(const std::string& object) {
    state.visible = true;
    state.dock_next = true;
    state.refresh = true;
    if (!object.empty()) {
        if (state.object != object) {
            state.selected.clear();
        }
        state.object = object;
        state.mesh = 0;
        state.frame_next = true;
    }
}

void drawBottomTab(UIContext&) {
    if (UIWidgets::HorizontalTab("UV Editor", UIWidgets::IconType::Mesh, state.visible)) {
        if (state.visible) {
            state.visible = false;
        }
        else open();
    }
}

void draw(UIContext&, const std::function<void(const char*)>& dock_to_bottom) {
    if (!state.visible) {
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
    ImGui::TextWrapped("UV 0 | %zu faces | %u islands | %zu selected",
        state.layout.indices.size() / 3, state.layout.island_count, state.selected.size());
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("Click island; Shift add; Ctrl remove; MMB pan; wheel zoom; F frame");
    ImGui::PopStyleColor();
    if (!state.error.empty()) {
        ImGui::TextWrapped("%s", state.error.c_str());
    }
    panes();
    ImGui::End();
}

} // namespace UvEditorUI
