/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtUi.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* rt.ui — addon panels, named mount points ("regions") and the immediate-mode
* widget surface. Moved here out of RtPython.cpp (which had grown past the
* 2000-line working limit); this is the only part of the Python layer that
* links against ImGui, so it isolates cleanly.
*
* Threading: every draw entry point runs on the main thread. Registry mutation
* comes from Python (GIL held). The lock order is always GIL -> g_mutex, never
* the reverse, so the two can never deadlock against each other. The Python
* draw callback is copied out from under g_mutex before it is invoked, because
* a callback is free to register/unregister panels while it runs.
*/

#include "Api/RtUi.h"
#include "RtUiBindings.h"

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "imgui.h"
#include "Api/RtPython.h"

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
struct AddonPanel {
    std::string title;
    std::string region;   // empty => floating window
    py::function draw;
    bool visible = true;
};

std::mutex g_mutex;
std::map<int, AddonPanel> g_panels;
// region id -> panel ids in registration order. Lets drawRegion() answer
// "is anything mounted here?" without walking every panel, which matters
// because the host calls it from ~15 places every frame.
std::map<std::string, std::vector<int>> g_region_index;
int g_next_panel_id = 1;

// Widget guard: rt.ui.* immediate-mode calls are only meaningful while a draw
// callback is running. Saved/restored rather than blindly cleared so a nested
// draw (a region reached from inside a floating panel) cannot switch it off
// for the outer callback.
bool g_drawing = false;

// Scopes opened by the CURRENT callback, in push order. ImGui state is global:
// an addon that opens a table or an id scope and forgets to close it would
// corrupt every widget drawn after it, including the host's own panels. We keep
// a LIFO stack — not per-kind counters — because an addon is free to interleave
// kinds (push_id, begin_group, pop_id, ...) and only LIFO unwinds that correctly.
enum class Scope {
    Id, Indent, Disabled, Group, Tree,
    Child, Table, Popup, TabBar, TabItem, ListBox,
    StyleColor, StyleVar
};
std::vector<Scope> g_scopes;

void requireDrawContext() {
    if (!g_drawing)
        throw std::runtime_error("rt.ui.* widgets are only valid inside a panel draw callback");
}

// ImGui's End* rules differ per construct: EndChild runs unconditionally, while
// EndTable/EndPopup/EndTabBar/EndTabItem/EndListBox must run ONLY when the
// matching Begin returned true. We therefore push a scope only when the caller
// is actually obliged to close it, so closing here is always correct.
void closeScope(Scope scope) {
    switch (scope) {
        case Scope::Id:         ImGui::PopID();         break;
        case Scope::Indent:     ImGui::Unindent();      break;
        case Scope::Disabled:   ImGui::EndDisabled();   break;
        case Scope::Group:      ImGui::EndGroup();      break;
        case Scope::Tree:       ImGui::TreePop();       break;
        case Scope::Child:      ImGui::EndChild();      break;
        case Scope::Table:      ImGui::EndTable();      break;
        case Scope::Popup:      ImGui::EndPopup();      break;
        case Scope::TabBar:     ImGui::EndTabBar();     break;
        case Scope::TabItem:    ImGui::EndTabItem();    break;
        case Scope::ListBox:    ImGui::EndListBox();    break;
        case Scope::StyleColor: ImGui::PopStyleColor(); break;
        case Scope::StyleVar:   ImGui::PopStyleVar();   break;
    }
}

void pushScope(Scope scope) { g_scopes.push_back(scope); }

// An explicit end_* call closes only when it matches the innermost open scope.
// A mismatched call (end_table with a group still open) is ignored rather than
// obeyed — obeying it would unbalance ImGui and take the host's UI down with it.
void popScope(Scope scope) {
    if (g_scopes.empty() || g_scopes.back() != scope) return;
    closeScope(scope);
    g_scopes.pop_back();
}

void unwindScopesTo(size_t base) {
    while (g_scopes.size() > base) {
        closeScope(g_scopes.back());
        g_scopes.pop_back();
    }
}

int addPanel(std::string title, std::string region, py::function draw) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const int id = g_next_panel_id++;
    g_region_index[region].push_back(id);
    g_panels.emplace(id, AddonPanel{ std::move(title), std::move(region), std::move(draw), true });
    return id;
}

void erasePanel(int panel_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_panels.find(panel_id);
    if (it == g_panels.end()) return;
    auto region_it = g_region_index.find(it->second.region);
    if (region_it != g_region_index.end()) {
        auto& ids = region_it->second;
        ids.erase(std::remove(ids.begin(), ids.end(), panel_id), ids.end());
        if (ids.empty()) g_region_index.erase(region_it);
    }
    g_panels.erase(it);   // GIL is held by every caller: py::function may release here
}

std::vector<int> snapshotRegion(const std::string& region) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_region_index.find(region);
    if (it == g_region_index.end()) return {};
    return it->second;
}

// Copies out what drawing needs so the lock is released before Python runs.
bool fetchPanel(int panel_id, std::string& out_title, py::function& out_draw, bool& out_visible) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_panels.find(panel_id);
    if (it == g_panels.end()) return false;
    out_title = it->second.title;
    out_visible = it->second.visible;
    out_draw = it->second.draw;   // GIL held by caller
    return true;
}

// Runs one callback with the widget guard armed and leftover ImGui scopes
// unwound. A Python exception is reported in place and logged; it must never
// escape into the host's frame.
void invokeDraw(int panel_id, const py::function& draw) {
    const bool prev_drawing = g_drawing;
    const size_t scope_base = g_scopes.size();   // nested draws unwind only their own
    g_drawing = true;

    ImGui::PushID(panel_id);   // two addons may use the same widget label in one region
    try {
        draw();
    } catch (const py::error_already_set& e) {
        // The exception may have escaped mid-scope, so close what it left open
        // BEFORE drawing the error line — otherwise the text lands inside a
        // half-built table and ImGui asserts.
        unwindScopesTo(scope_base);
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "panel draw error (see console)");
        rtpython::appendConsoleText(std::string(e.what()) + "\n");
    } catch (const std::exception& e) {
        unwindScopesTo(scope_base);
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "panel draw error (see console)");
        rtpython::appendConsoleText(std::string("rt.ui: ") + e.what() + "\n");
    }
    unwindScopesTo(scope_base);
    ImGui::PopID();

    g_drawing = prev_drawing;
}

void drawRegionImpl(const char* region_id, bool as_section) {
    if (!region_id || !*region_id) return;
    if (!rtpython::isInitialized()) return;
    if (!rtui::hasRegion(region_id)) return;   // cheap gate: no GIL when nothing is mounted

    py::gil_scoped_acquire gil;
    const std::vector<int> ids = snapshotRegion(region_id);
    for (int id : ids) {
        std::string title;
        py::function draw;
        bool visible = true;
        if (!fetchPanel(id, title, draw, visible)) continue;  // unregistered mid-loop
        if (!visible) continue;

        if (as_section && !title.empty()) {
            ImGui::PushID(id);
            const bool open = ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
            ImGui::PopID();
            if (!open) continue;
        }
        invokeDraw(id, draw);
    }
}

// ---------------------------------------------------------------------------
// Widget helpers
// ---------------------------------------------------------------------------
std::vector<float> floatsFromPython(const py::handle& value, size_t expected, const char* what) {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(seq) != expected)
        throw py::value_error(std::string(what) + " expects " + std::to_string(expected) + " components");
    std::vector<float> out(expected);
    for (size_t i = 0; i < expected; ++i) out[i] = seq[i].cast<float>();
    return out;
}

py::list floatsToPython(const float* data, size_t count) {
    py::list out;
    for (size_t i = 0; i < count; ++i) out.append(data[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Style / key name tables. Snake_case names rather than raw ImGui enum ints, so
// an addon reads like the rest of the rt API and stays source-compatible if the
// enum values shift underneath.
// ---------------------------------------------------------------------------
struct NamedEnum { const char* name; int value; };

const NamedEnum kStyleColors[] = {
    { "text",              ImGuiCol_Text },
    { "text_disabled",     ImGuiCol_TextDisabled },
    { "window_bg",         ImGuiCol_WindowBg },
    { "child_bg",          ImGuiCol_ChildBg },
    { "popup_bg",          ImGuiCol_PopupBg },
    { "border",            ImGuiCol_Border },
    { "frame_bg",          ImGuiCol_FrameBg },
    { "frame_bg_hovered",  ImGuiCol_FrameBgHovered },
    { "frame_bg_active",   ImGuiCol_FrameBgActive },
    { "title_bg",          ImGuiCol_TitleBg },
    { "title_bg_active",   ImGuiCol_TitleBgActive },
    { "menubar_bg",        ImGuiCol_MenuBarBg },
    { "scrollbar_bg",      ImGuiCol_ScrollbarBg },
    { "scrollbar_grab",    ImGuiCol_ScrollbarGrab },
    { "check_mark",        ImGuiCol_CheckMark },
    { "slider_grab",       ImGuiCol_SliderGrab },
    { "slider_grab_active",ImGuiCol_SliderGrabActive },
    { "button",            ImGuiCol_Button },
    { "button_hovered",    ImGuiCol_ButtonHovered },
    { "button_active",     ImGuiCol_ButtonActive },
    { "header",            ImGuiCol_Header },
    { "header_hovered",    ImGuiCol_HeaderHovered },
    { "header_active",     ImGuiCol_HeaderActive },
    { "separator",         ImGuiCol_Separator },
    { "tab",               ImGuiCol_Tab },
    { "tab_hovered",       ImGuiCol_TabHovered },
    { "tab_selected",      ImGuiCol_TabSelected },
    { "plot_lines",        ImGuiCol_PlotLines },
    { "plot_histogram",    ImGuiCol_PlotHistogram },
    { "table_header_bg",   ImGuiCol_TableHeaderBg },
    { "table_row_bg",      ImGuiCol_TableRowBg },
    { "table_row_bg_alt",  ImGuiCol_TableRowBgAlt },
    { "text_selected_bg",  ImGuiCol_TextSelectedBg },
};

// `vec2` marks the vars ImGui stores as ImVec2 — pushing a float into one of
// those is an ImGui assert, so the binding has to know which is which.
struct NamedStyleVar { const char* name; int value; bool vec2; };

const NamedStyleVar kStyleVars[] = {
    { "alpha",              ImGuiStyleVar_Alpha,             false },
    { "disabled_alpha",     ImGuiStyleVar_DisabledAlpha,     false },
    { "window_padding",     ImGuiStyleVar_WindowPadding,     true  },
    { "window_rounding",    ImGuiStyleVar_WindowRounding,    false },
    { "window_border_size", ImGuiStyleVar_WindowBorderSize,  false },
    { "child_rounding",     ImGuiStyleVar_ChildRounding,     false },
    { "child_border_size",  ImGuiStyleVar_ChildBorderSize,   false },
    { "popup_rounding",     ImGuiStyleVar_PopupRounding,     false },
    { "frame_padding",      ImGuiStyleVar_FramePadding,      true  },
    { "frame_rounding",     ImGuiStyleVar_FrameRounding,     false },
    { "frame_border_size",  ImGuiStyleVar_FrameBorderSize,   false },
    { "item_spacing",       ImGuiStyleVar_ItemSpacing,       true  },
    { "item_inner_spacing", ImGuiStyleVar_ItemInnerSpacing,  true  },
    { "indent_spacing",     ImGuiStyleVar_IndentSpacing,     false },
    { "cell_padding",       ImGuiStyleVar_CellPadding,       true  },
    { "scrollbar_size",     ImGuiStyleVar_ScrollbarSize,     false },
    { "grab_min_size",      ImGuiStyleVar_GrabMinSize,       false },
    { "grab_rounding",      ImGuiStyleVar_GrabRounding,      false },
    { "tab_rounding",       ImGuiStyleVar_TabRounding,       false },
};

const NamedEnum kKeys[] = {
    { "enter", ImGuiKey_Enter },   { "escape", ImGuiKey_Escape },
    { "space", ImGuiKey_Space },   { "tab",    ImGuiKey_Tab },
    { "delete", ImGuiKey_Delete }, { "backspace", ImGuiKey_Backspace },
    { "left",  ImGuiKey_LeftArrow },  { "right", ImGuiKey_RightArrow },
    { "up",    ImGuiKey_UpArrow },    { "down",  ImGuiKey_DownArrow },
};

int styleColorIndex(const std::string& name) {
    for (const NamedEnum& e : kStyleColors) if (name == e.name) return e.value;
    return -1;
}

int styleVarIndex(const std::string& name, bool& out_vec2) {
    for (const NamedStyleVar& e : kStyleVars)
        if (name == e.name) { out_vec2 = e.vec2; return e.value; }
    return -1;
}

int keyIndex(const std::string& name) {
    for (const NamedEnum& e : kKeys) if (name == e.name) return e.value;
    if (name.size() == 1) {
        const char c = name[0];
        if (c >= 'a' && c <= 'z') return ImGuiKey_A + (c - 'a');
        if (c >= 'A' && c <= 'Z') return ImGuiKey_A + (c - 'A');
        if (c >= '0' && c <= '9') return ImGuiKey_0 + (c - '0');
    }
    if (name.size() >= 2 && (name[0] == 'f' || name[0] == 'F')) {
        const int n = std::atoi(name.c_str() + 1);
        if (n >= 1 && n <= 12) return ImGuiKey_F1 + (n - 1);
    }
    return -1;
}

std::vector<std::string> styleColorNames() {
    std::vector<std::string> out;
    for (const NamedEnum& e : kStyleColors) out.emplace_back(e.name);
    return out;
}

std::vector<std::string> styleVarNames() {
    std::vector<std::string> out;
    for (const NamedStyleVar& e : kStyleVars) out.emplace_back(e.name);
    return out;
}

} // namespace

namespace rtui {

std::vector<std::string> knownRegions() {
    return {
        // Properties tabs — drawn from the shared tab dispatch, so these show up
        // both in the docked Properties panel and in a torn-off tab window.
        "properties.scene",
        "properties.render",
        "properties.terrain",
        "properties.water",
        "properties.volumetric",
        "properties.simulation",
        "properties.world",
        "properties.modeling",
        "properties.hair",
        "properties.system",
        "properties.paint",
        "properties.scatter",
        "properties.stylize",
        "properties.sculpt",
        // Main menu bar.
        "menu.file",
        "menu.edit",
        "menu.render",
        "menu.view",
        "menu.help",
    };
}

bool hasRegion(const char* region_id) {
    if (!region_id || !*region_id) return false;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_region_index.find(region_id);
    return it != g_region_index.end() && !it->second.empty();
}

void drawRegion(const char* region_id)     { drawRegionImpl(region_id, true); }
void drawMenuRegion(const char* region_id) { drawRegionImpl(region_id, false); }

void drawAddonPanels() {
    if (!rtpython::isInitialized()) return;
    // Floating panels live under the empty region id, which hasRegion() rejects
    // by contract, so probe the index directly to keep the no-GIL fast path.
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_region_index.find(std::string{});
        if (it == g_region_index.end() || it->second.empty()) return;
    }

    py::gil_scoped_acquire gil;
    const std::vector<int> ids = snapshotRegion(std::string{});
    for (int id : ids) {
        std::string title;
        py::function draw;
        bool visible = true;
        if (!fetchPanel(id, title, draw, visible)) continue;
        if (!visible) continue;

        bool open = true;
        if (ImGui::Begin(title.c_str(), &open)) {
            invokeDraw(id, draw);
        }
        ImGui::End();

        // Closing the window hides the panel; it stays registered so the user can
        // bring it back from View > Addon Panels without reloading the addon.
        if (!open) setPanelVisible(id, false);
    }
}

std::vector<PanelInfo> listPanels() {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::vector<PanelInfo> out;
    out.reserve(g_panels.size());
    for (const auto& [id, panel] : g_panels)
        out.push_back(PanelInfo{ id, panel.title, panel.region, panel.visible });
    return out;
}

void setPanelVisible(int panel_id, bool visible) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_panels.find(panel_id);
    if (it != g_panels.end()) it->second.visible = visible;
}

void clearPanels() noexcept {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_panels.clear();          // caller holds the GIL (rtpython::shutdown)
        g_region_index.clear();
    } catch (...) {
        // Shutdown must never throw into renderer teardown.
    }
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------
void registerBindings(py::module_& module) {
    py::module_ ui = module.def_submodule(
        "ui", "Addon panels, panel regions & immediate-mode widgets (Faz 4b)");

    // --- registration ---------------------------------------------------
    ui.def("register_panel", [](const std::string& title, py::function draw) {
        return addPanel(title, std::string{}, std::move(draw));
    }, py::arg("title"), py::arg("draw_callback"),
       "Register a floating panel window. Returns a panel id.");

    ui.def("register_region", [](const std::string& region, const std::string& title,
                                 py::function draw) {
        return addPanel(title, region, std::move(draw));
    }, py::arg("region"), py::arg("title"), py::arg("draw_callback"),
       "Mount a draw callback inside an existing editor panel. See rt.ui.regions().");

    ui.def("unregister_panel", [](int panel_id) { erasePanel(panel_id); }, py::arg("panel_id"));
    ui.def("regions", &knownRegions, "Region ids this build draws.");

    ui.def("set_panel_visible", [](int panel_id, bool visible) {
        setPanelVisible(panel_id, visible);
    }, py::arg("panel_id"), py::arg("visible"));

    ui.def("list_panels", [] {
        py::list out;
        for (const PanelInfo& p : listPanels()) {
            py::dict d;
            d["id"] = p.id;
            d["title"] = p.title;
            d["region"] = p.region;
            d["visible"] = p.visible;
            out.append(std::move(d));
        }
        return out;
    });

    // --- text & layout --------------------------------------------------
    ui.def("text", [](const std::string& s) {
        requireDrawContext(); ImGui::TextUnformatted(s.c_str());
    }, py::arg("text"));
    ui.def("text_colored", [](const std::string& s, const py::object& rgba) {
        requireDrawContext();
        const std::vector<float> c = floatsFromPython(rgba, 4, "text_colored color");
        ImGui::TextColored(ImVec4(c[0], c[1], c[2], c[3]), "%s", s.c_str());
    }, py::arg("text"), py::arg("color"));
    ui.def("text_disabled", [](const std::string& s) {
        requireDrawContext(); ImGui::TextDisabled("%s", s.c_str());
    }, py::arg("text"));
    ui.def("text_wrapped", [](const std::string& s) {
        requireDrawContext(); ImGui::TextWrapped("%s", s.c_str());
    }, py::arg("text"));
    ui.def("bullet_text", [](const std::string& s) {
        requireDrawContext(); ImGui::BulletText("%s", s.c_str());
    }, py::arg("text"));

    ui.def("separator", [] { requireDrawContext(); ImGui::Separator(); });
    ui.def("separator_text", [](const std::string& s) {
        requireDrawContext(); ImGui::SeparatorText(s.c_str());
    }, py::arg("text"));
    ui.def("spacing", [] { requireDrawContext(); ImGui::Spacing(); });
    ui.def("same_line", [](float offset, float spacing) {
        requireDrawContext(); ImGui::SameLine(offset, spacing);
    }, py::arg("offset") = 0.0f, py::arg("spacing") = -1.0f);
    ui.def("dummy", [](float w, float h) {
        requireDrawContext(); ImGui::Dummy(ImVec2(w, h));
    }, py::arg("width"), py::arg("height"));

    ui.def("indent", [] { requireDrawContext(); ImGui::Indent(); pushScope(Scope::Indent); });
    ui.def("unindent", [] { requireDrawContext(); popScope(Scope::Indent); });
    ui.def("push_id", [](const std::string& id) {
        requireDrawContext(); ImGui::PushID(id.c_str()); pushScope(Scope::Id);
    }, py::arg("id"));
    ui.def("pop_id", [] { requireDrawContext(); popScope(Scope::Id); });
    ui.def("begin_group", [] { requireDrawContext(); ImGui::BeginGroup(); pushScope(Scope::Group); });
    ui.def("end_group", [] { requireDrawContext(); popScope(Scope::Group); });
    ui.def("begin_disabled", [](bool disabled) {
        requireDrawContext(); ImGui::BeginDisabled(disabled); pushScope(Scope::Disabled);
    }, py::arg("disabled") = true);
    ui.def("end_disabled", [] { requireDrawContext(); popScope(Scope::Disabled); });

    ui.def("collapsing_header", [](const std::string& label, bool default_open) {
        requireDrawContext();
        return ImGui::CollapsingHeader(label.c_str(),
                                       default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0);
    }, py::arg("label"), py::arg("default_open") = true);
    ui.def("tree_node", [](const std::string& label) {
        requireDrawContext();
        const bool open = ImGui::TreeNode(label.c_str());
        if (open) pushScope(Scope::Tree);
        return open;
    }, py::arg("label"));
    ui.def("tree_pop", [] { requireDrawContext(); popScope(Scope::Tree); });

    // --- inputs ---------------------------------------------------------
    ui.def("button", [](const std::string& label, float width, float height) {
        requireDrawContext(); return ImGui::Button(label.c_str(), ImVec2(width, height));
    }, py::arg("label"), py::arg("width") = 0.0f, py::arg("height") = 0.0f);
    ui.def("small_button", [](const std::string& label) {
        requireDrawContext(); return ImGui::SmallButton(label.c_str());
    }, py::arg("label"));
    ui.def("checkbox", [](const std::string& label, bool value) {
        requireDrawContext(); ImGui::Checkbox(label.c_str(), &value); return value;
    }, py::arg("label"), py::arg("value"));
    ui.def("radio_button", [](const std::string& label, bool active) {
        requireDrawContext(); return ImGui::RadioButton(label.c_str(), active);
    }, py::arg("label"), py::arg("active"));

    ui.def("slider_float", [](const std::string& label, float value, float v_min, float v_max) {
        requireDrawContext(); ImGui::SliderFloat(label.c_str(), &value, v_min, v_max); return value;
    }, py::arg("label"), py::arg("value"), py::arg("v_min"), py::arg("v_max"));
    ui.def("slider_int", [](const std::string& label, int value, int v_min, int v_max) {
        requireDrawContext(); ImGui::SliderInt(label.c_str(), &value, v_min, v_max); return value;
    }, py::arg("label"), py::arg("value"), py::arg("v_min"), py::arg("v_max"));
    ui.def("drag_float", [](const std::string& label, float value, float speed,
                            float v_min, float v_max) {
        requireDrawContext();
        ImGui::DragFloat(label.c_str(), &value, speed, v_min, v_max);
        return value;
    }, py::arg("label"), py::arg("value"), py::arg("speed") = 0.01f,
       py::arg("v_min") = 0.0f, py::arg("v_max") = 0.0f);
    ui.def("drag_int", [](const std::string& label, int value, float speed, int v_min, int v_max) {
        requireDrawContext();
        ImGui::DragInt(label.c_str(), &value, speed, v_min, v_max);
        return value;
    }, py::arg("label"), py::arg("value"), py::arg("speed") = 1.0f,
       py::arg("v_min") = 0, py::arg("v_max") = 0);
    ui.def("input_float", [](const std::string& label, float value, float step) {
        requireDrawContext(); ImGui::InputFloat(label.c_str(), &value, step); return value;
    }, py::arg("label"), py::arg("value"), py::arg("step") = 0.0f);
    ui.def("input_int", [](const std::string& label, int value, int step) {
        requireDrawContext(); ImGui::InputInt(label.c_str(), &value, step); return value;
    }, py::arg("label"), py::arg("value"), py::arg("step") = 1);

    ui.def("input_text", [](const std::string& label, const std::string& value, size_t max_length) {
        requireDrawContext();
        // Grows with the incoming value instead of silently truncating at a fixed
        // 1 KiB, which the original binding did.
        const size_t capacity = (std::max)(max_length, value.size()) + 1;
        std::vector<char> buffer(capacity, '\0');
        std::copy(value.begin(), value.end(), buffer.begin());
        ImGui::InputText(label.c_str(), buffer.data(), buffer.size());
        return std::string(buffer.data());
    }, py::arg("label"), py::arg("value"), py::arg("max_length") = size_t{ 1024 });

    ui.def("combo", [](const std::string& label, int current, const std::vector<std::string>& items) {
        requireDrawContext();
        if (items.empty()) return current;
        const int count = static_cast<int>(items.size());
        int index = (current < 0 || current >= count) ? 0 : current;
        if (ImGui::BeginCombo(label.c_str(), items[index].c_str())) {
            for (int i = 0; i < count; ++i) {
                const bool selected = (index == i);
                if (ImGui::Selectable(items[i].c_str(), selected)) index = i;
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        return index;
    }, py::arg("label"), py::arg("current_index"), py::arg("items"));

    ui.def("color_edit3", [](const std::string& label, const py::object& color) {
        requireDrawContext();
        std::vector<float> c = floatsFromPython(color, 3, "color_edit3 color");
        ImGui::ColorEdit3(label.c_str(), c.data());
        return floatsToPython(c.data(), 3);
    }, py::arg("label"), py::arg("color"));
    ui.def("color_edit4", [](const std::string& label, const py::object& color) {
        requireDrawContext();
        std::vector<float> c = floatsFromPython(color, 4, "color_edit4 color");
        ImGui::ColorEdit4(label.c_str(), c.data());
        return floatsToPython(c.data(), 4);
    }, py::arg("label"), py::arg("color"));

    // --- menus & feedback -----------------------------------------------
    ui.def("menu_item", [](const std::string& label, const std::string& shortcut,
                           bool selected, bool enabled) {
        requireDrawContext();
        return ImGui::MenuItem(label.c_str(), shortcut.empty() ? nullptr : shortcut.c_str(),
                               selected, enabled);
    }, py::arg("label"), py::arg("shortcut") = std::string{},
       py::arg("selected") = false, py::arg("enabled") = true);

    ui.def("tooltip", [](const std::string& text) {
        requireDrawContext();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", text.c_str());
    }, py::arg("text"), "Tooltip for the widget drawn immediately before this call.");
    ui.def("is_item_hovered", [] { requireDrawContext(); return ImGui::IsItemHovered(); });
    ui.def("progress_bar", [](float fraction, const std::string& overlay) {
        requireDrawContext();
        ImGui::ProgressBar(fraction, ImVec2(-1.0f, 0.0f),
                           overlay.empty() ? nullptr : overlay.c_str());
    }, py::arg("fraction"), py::arg("overlay") = std::string{});

    // --- containers (Faz 5.5d) ------------------------------------------
    // EndChild runs unconditionally in ImGui, so the scope is pushed whatever
    // BeginChild returned; every other Begin* below pushes only on true.
    ui.def("begin_child", [](const std::string& id, float width, float height, bool border) {
        requireDrawContext();
        const bool visible = ImGui::BeginChild(id.c_str(), ImVec2(width, height),
                                               border ? ImGuiChildFlags_Borders : 0);
        pushScope(Scope::Child);
        return visible;
    }, py::arg("id"), py::arg("width") = 0.0f, py::arg("height") = 0.0f,
       py::arg("border") = false,
       "Scrollable sub-region. Always pair with end_child(), even when it returns False.");
    ui.def("end_child", [] { requireDrawContext(); popScope(Scope::Child); });

    ui.def("begin_table", [](const std::string& id, int columns, bool borders,
                             bool row_background, bool resizable, float height) {
        requireDrawContext();
        if (columns < 1) throw py::value_error("begin_table needs at least one column");
        ImGuiTableFlags flags = 0;
        if (borders)        flags |= ImGuiTableFlags_Borders;
        if (row_background) flags |= ImGuiTableFlags_RowBg;
        if (resizable)      flags |= ImGuiTableFlags_Resizable;
        const bool open = ImGui::BeginTable(id.c_str(), columns, flags, ImVec2(0.0f, height));
        if (open) pushScope(Scope::Table);
        return open;
    }, py::arg("id"), py::arg("columns"), py::arg("borders") = true,
       py::arg("row_background") = true, py::arg("resizable") = true, py::arg("height") = 0.0f,
       "Call end_table() only when this returns True.");
    ui.def("end_table", [] { requireDrawContext(); popScope(Scope::Table); });
    ui.def("table_setup_column", [](const std::string& label) {
        requireDrawContext(); ImGui::TableSetupColumn(label.c_str());
    }, py::arg("label"));
    ui.def("table_headers_row", [] { requireDrawContext(); ImGui::TableHeadersRow(); });
    ui.def("table_next_row", [] { requireDrawContext(); ImGui::TableNextRow(); });
    ui.def("table_next_column", [] { requireDrawContext(); return ImGui::TableNextColumn(); });
    ui.def("table_set_column_index", [](int index) {
        requireDrawContext(); return ImGui::TableSetColumnIndex(index);
    }, py::arg("index"));

    ui.def("begin_tab_bar", [](const std::string& id) {
        requireDrawContext();
        const bool open = ImGui::BeginTabBar(id.c_str());
        if (open) pushScope(Scope::TabBar);
        return open;
    }, py::arg("id"));
    ui.def("end_tab_bar", [] { requireDrawContext(); popScope(Scope::TabBar); });
    ui.def("begin_tab_item", [](const std::string& label) {
        requireDrawContext();
        const bool selected = ImGui::BeginTabItem(label.c_str());
        if (selected) pushScope(Scope::TabItem);
        return selected;
    }, py::arg("label"));
    ui.def("end_tab_item", [] { requireDrawContext(); popScope(Scope::TabItem); });

    ui.def("begin_list_box", [](const std::string& label, float width, float height) {
        requireDrawContext();
        const bool open = ImGui::BeginListBox(label.c_str(), ImVec2(width, height));
        if (open) pushScope(Scope::ListBox);
        return open;
    }, py::arg("label"), py::arg("width") = 0.0f, py::arg("height") = 0.0f);
    ui.def("end_list_box", [] { requireDrawContext(); popScope(Scope::ListBox); });
    ui.def("selectable", [](const std::string& label, bool selected) {
        requireDrawContext(); return ImGui::Selectable(label.c_str(), selected);
    }, py::arg("label"), py::arg("selected") = false);

    // Popups: open_popup() only arms it; the begin_* call must run every frame.
    ui.def("open_popup", [](const std::string& id) {
        requireDrawContext(); ImGui::OpenPopup(id.c_str());
    }, py::arg("id"));
    ui.def("begin_popup", [](const std::string& id) {
        requireDrawContext();
        const bool open = ImGui::BeginPopup(id.c_str());
        if (open) pushScope(Scope::Popup);
        return open;
    }, py::arg("id"));
    ui.def("begin_popup_modal", [](const std::string& name) {
        requireDrawContext();
        const bool open = ImGui::BeginPopupModal(name.c_str());
        if (open) pushScope(Scope::Popup);
        return open;
    }, py::arg("name"));
    ui.def("end_popup", [] { requireDrawContext(); popScope(Scope::Popup); });
    ui.def("close_current_popup", [] { requireDrawContext(); ImGui::CloseCurrentPopup(); });

    // --- styling ---------------------------------------------------------
    // Without these an addon panel cannot follow the app theme, which is what
    // made injected sections read as foreign next to the host's own controls.
    ui.def("push_style_color", [](const std::string& name, const py::object& color) {
        requireDrawContext();
        const int idx = styleColorIndex(name);
        if (idx < 0) throw py::value_error("unknown style color: " + name);
        const std::vector<float> c = floatsFromPython(color, 4, "push_style_color color");
        ImGui::PushStyleColor(static_cast<ImGuiCol>(idx), ImVec4(c[0], c[1], c[2], c[3]));
        pushScope(Scope::StyleColor);
    }, py::arg("name"), py::arg("color"));
    ui.def("pop_style_color", [] { requireDrawContext(); popScope(Scope::StyleColor); });
    ui.def("style_colors", &styleColorNames, "Names accepted by push_style_color().");

    ui.def("push_style_var", [](const std::string& name, const py::object& value) {
        requireDrawContext();
        bool wants_vec2 = false;
        const int idx = styleVarIndex(name, wants_vec2);
        if (idx < 0) throw py::value_error("unknown style var: " + name);
        if (wants_vec2) {
            const std::vector<float> v = floatsFromPython(value, 2, name.c_str());
            ImGui::PushStyleVar(static_cast<ImGuiStyleVar>(idx), ImVec2(v[0], v[1]));
        } else {
            ImGui::PushStyleVar(static_cast<ImGuiStyleVar>(idx), py::cast<float>(value));
        }
        pushScope(Scope::StyleVar);
    }, py::arg("name"), py::arg("value"),
       "Scalar vars take a float; padding/spacing vars take a (x, y) pair.");
    ui.def("pop_style_var", [] { requireDrawContext(); popScope(Scope::StyleVar); });
    ui.def("style_vars", &styleVarNames, "Names accepted by push_style_var().");

    // --- remaining widgets ----------------------------------------------
    ui.def("input_text_multiline", [](const std::string& label, const std::string& value,
                                      float width, float height, size_t max_length) {
        requireDrawContext();
        const size_t capacity = (std::max)(max_length, value.size()) + 1;
        std::vector<char> buffer(capacity, '\0');
        std::copy(value.begin(), value.end(), buffer.begin());
        ImGui::InputTextMultiline(label.c_str(), buffer.data(), buffer.size(),
                                  ImVec2(width, height));
        return std::string(buffer.data());
    }, py::arg("label"), py::arg("value"), py::arg("width") = 0.0f, py::arg("height") = 0.0f,
       py::arg("max_length") = size_t{ 4096 });

    ui.def("plot_lines", [](const std::string& label, const std::vector<float>& values,
                            float height, const std::string& overlay) {
        requireDrawContext();
        if (values.empty()) return;
        ImGui::PlotLines(label.c_str(), values.data(), static_cast<int>(values.size()), 0,
                         overlay.empty() ? nullptr : overlay.c_str(),
                         FLT_MAX, FLT_MAX, ImVec2(0.0f, height));
    }, py::arg("label"), py::arg("values"), py::arg("height") = 60.0f,
       py::arg("overlay") = std::string{});
    ui.def("plot_histogram", [](const std::string& label, const std::vector<float>& values,
                                float height, const std::string& overlay) {
        requireDrawContext();
        if (values.empty()) return;
        ImGui::PlotHistogram(label.c_str(), values.data(), static_cast<int>(values.size()), 0,
                             overlay.empty() ? nullptr : overlay.c_str(),
                             FLT_MAX, FLT_MAX, ImVec2(0.0f, height));
    }, py::arg("label"), py::arg("values"), py::arg("height") = 60.0f,
       py::arg("overlay") = std::string{});

    ui.def("is_key_pressed", [](const std::string& key) {
        requireDrawContext();
        const int code = keyIndex(key);
        if (code < 0) throw py::value_error("unknown key: " + key);
        return ImGui::IsKeyPressed(static_cast<ImGuiKey>(code), false);
    }, py::arg("key"), "Keys: a-z, 0-9, f1-f12, enter, escape, space, tab, delete, arrows.");
    ui.def("content_region_avail", [] {
        requireDrawContext();
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        return py::make_tuple(avail.x, avail.y);
    });
}

} // namespace rtui
