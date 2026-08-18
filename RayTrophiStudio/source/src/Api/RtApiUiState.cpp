/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiUiState.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * `rt.editor` — the application's own editor view state, as VALUES.
 *
 * ★★★ Read the long note on EditorState in RtApi.h before extending this file.
 * The short version: `rt.ui` draws panels and stays in-process; `rt.editor`
 * reports and moves which editor is showing. The line between them is not
 * "UI vs not-UI", it is "draw call vs value".
 *
 * ★★ Nothing here stores anything. Every reading is taken from the live SceneUI
 * flags each call. A cached copy would be a second authority for a fact the UI
 * already owns, which is the exact defect this surface exists to expose.
 */

#include "Api/RtApi.h"
#include "RtApiInternal.h"

#include <string>

namespace rtapi {

namespace {

// One table, both directions. The names are the SAME vocabulary the template
// system uses for `ui_state.bottom_editor`, deliberately: two spellings for one
// concept is how a name silently works in one place and silently fails in
// another, and that has already cost this repo a debugging session.
struct BottomEditorEntry {
    const char* name;
    bool SceneUI::* flag;      // nullptr for the timeline modes, handled apart
};

const BottomEditorEntry kBottomEditors[] = {
    {"console",    &SceneUI::show_scene_log},
    {"assets",     &SceneUI::show_asset_browser},
    {"simulation", &SceneUI::show_node_editor},
    {"geometry",   &SceneUI::show_geometry_graph},
    {"material",   &SceneUI::show_material_graph},
    {"terrain",    &SceneUI::show_terrain_graph},
    {"anim_graph", &SceneUI::show_anim_graph},
};

const char* kNodeDomains[] = { "simulation", "geometry", "material", "terrain", "animation" };

// Which node-editor domain a bottom editor name corresponds to, or -1 when the
// name is not a node editor at all. ★ Selecting "geometry" must move the Nodes
// selector too: the selector naming one graph while another is on screen is the
// shape of a lying panel, and it would be worse coming from a script because
// nobody is looking at it.
int nodeDomainForBottomEditor(const std::string& name) {
    for (int i = 0; i < 5; ++i) {
        if (name == kNodeDomains[i]) return i;
    }
    if (name == "anim_graph") return 4;   // template vocabulary -> "animation"
    return -1;
}

} // namespace

EditorState editorState() {
    EditorState out;

    // ★ Order matters here and it is not arbitrary. The timeline panel hosts TWO
    // bottom editors that differ only by its editor mode, so asking "is the
    // timeline showing" is not enough to name what the user is looking at.
    if (ui.show_animation_panel) {
        out.open_editors.push_back(
            ui.timeline.getEditorMode() == TimelineEditorMode::GraphEditor
                ? "graph_editor" : "dope_sheet");
    }
    // No `break`: the loop collects all of them. The panels are supposed to be
    // mutually exclusive, and a reader that stops at the first one would report
    // a healthy-looking answer precisely when they are not.
    for (const auto& entry : kBottomEditors) {
        if (ui.*(entry.flag)) out.open_editors.push_back(entry.name);
    }
    out.bottom_editor = out.open_editors.empty() ? "none" : out.open_editors.front();

    const int domain = static_cast<int>(ui.node_editor_domain);
    out.node_editor_domain =
        (domain >= 0 && domain < 5) ? kNodeDomains[domain] : "simulation";
    out.node_editor_open = ui.show_node_editor;
    // Which scoped simulation graph the canvas is on. Reported even when the
    // Nodes window is closed: it is the selection, not a property of the window.
    out.sim_graph_scope = ui.sim_graph_scope;
    out.sim_graph_owner = ui.sim_graph_owner;
    return out;
}

Result setSimGraphScope(const std::string& scope, const std::string& owner) {
    // *** VALIDATE BEFORE MUTATING, for the same reason setBottomEditor does:
    // a refused call that had already moved the selection would leave the canvas
    // somewhere the caller was told it had not gone.
    NodeSystem::Sim::GraphScope parsed;
    if (!NodeSystem::Sim::parseScope(scope, parsed))
        return Result::fail("unknown graph scope '" + scope +
                            "' (expected 'object', 'domain' or 'world')");
    // ** Selecting a scope does NOT require a graph to exist there. The panel
    // draws an explicit "no graph for this owner -- create one" state, which is
    // how a user reaches creation in the first place. Refusing here would make
    // the empty case unreachable from the UI and from a script alike.
    ui.sim_graph_scope = scope;
    ui.sim_graph_owner = (parsed == NodeSystem::Sim::GraphScope::World)
                         ? std::string() : owner;
    return Result::success();
}

Result setBottomEditor(const std::string& name) {
    // ★★★ VALIDATE BEFORE MUTATING. The first cut closed every panel and only
    // then discovered it did not recognise the name — so a rejected call left
    // the screen blank, and the caller got an error with no hint that anything
    // had changed. A failure with a side effect is worse than either a clean
    // failure or a success: it is the one outcome nobody checks for.
    const bool known_timeline = (name == "dope_sheet" || name == "graph_editor");
    bool known_panel = false;
    for (const auto& entry : kBottomEditors)
        if (name == entry.name) { known_panel = true; break; }
    if (name != "none" && !known_timeline && !known_panel)
        return Result::fail("unknown bottom editor: " + name);

    // Everything off first, then exactly one on. Toggling the requested one and
    // trusting the panels to be exclusive would leave a second panel showing
    // whenever a caller changed editors twice — "close the others" is a rule the
    // UI enforces on click, and a script must not be able to route around it.
    ui.show_animation_panel = false;
    ui.show_scene_log = false;
    ui.show_asset_browser = false;
    ui.show_node_editor = false;
    ui.show_geometry_graph = false;
    ui.show_material_graph = false;
    ui.show_terrain_graph = false;
    ui.show_anim_graph = false;

    if (name == "none") return Result::success();

    if (name == "dope_sheet" || name == "graph_editor") {
        ui.show_animation_panel = true;
        ui.timeline.setEditorMode(name == "graph_editor" ? TimelineEditorMode::GraphEditor
                                                         : TimelineEditorMode::DopeSheet);
        ui.focus_bottom_panel_next_frame = true;
        return Result::success();
    }

    for (const auto& entry : kBottomEditors) {
        if (name != entry.name) continue;
        ui.*(entry.flag) = true;
        const int domain = nodeDomainForBottomEditor(name);
        if (domain >= 0)
            ui.node_editor_domain = static_cast<SceneUI::NodeEditorDomain>(domain);
        ui.focus_bottom_panel_next_frame = true;
        return Result::success();
    }

    // Unreachable: the name was validated above, before anything was touched.
    return Result::fail("unknown bottom editor: " + name);
}

Result setNodeEditorDomain(const std::string& name) {
    for (int i = 0; i < 5; ++i) {
        if (name != kNodeDomains[i]) continue;
        ui.node_editor_domain = static_cast<SceneUI::NodeEditorDomain>(i);
        // If a node editor is already showing, MOVE it, the same way the panel's
        // own selector does. Setting the domain while a different graph stays on
        // screen would make `editor.get_state` report a domain the user is not
        // looking at.
        const bool any_open = ui.show_node_editor || ui.show_geometry_graph ||
            ui.show_material_graph || ui.show_terrain_graph || ui.show_anim_graph;
        if (any_open) {
            static const char* kBottomNameForDomain[] = {
                "simulation", "geometry", "material", "terrain", "anim_graph"
            };
            return setBottomEditor(kBottomNameForDomain[i]);
        }
        return Result::success();
    }
    return Result::fail("unknown node editor domain: " + name);
}

} // namespace rtapi
