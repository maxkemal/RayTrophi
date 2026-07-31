/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtUi.h
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* rt.ui host surface — addon-authored ImGui panels (Faz 4b) and the named
* mount points ("regions") that let an addon inject widgets INTO an existing
* editor panel instead of only opening its own floating window.
*
* Split out of RtPython.cpp: that translation unit had grown past the 2000-line
* working limit and the UI surface is a self-contained concern (it is the only
* part of the Python layer that touches ImGui).
*
* Rules:
*  - Every entry point here runs on the main thread, inside the host's
*    ImGui NewFrame/Render pair.
*  - The Python draw callback never opens or closes an ImGui window itself:
*    drawAddonPanels() owns Begin/End for floating panels, and a region draws
*    inline into whatever window the host already has open.
*  - Keep this header pybind-free so UI translation units can include it
*    without pulling in the embedded interpreter. The binding registration
*    lives in the internal src/Api/RtUiBindings.h.
*/
#pragma once

#include <string>
#include <vector>

namespace rtui {

// ---------------------------------------------------------------------------
// Regions — stable string ids an addon passes to rt.ui.register_region().
// A region is a mount point the host draws at a fixed spot in its own UI.
//
//   properties.<tab>   inline section at the bottom of a Properties tab; it
//                      shows in both the docked panel and the torn-off window
//                      because the host draws it from the shared tab dispatch.
//   menu.<name>        entries appended to the end of a main-menu-bar menu.
//
// knownRegions() is the authoritative list (surfaced as rt.ui.regions()).
// Registering an unknown id is not an error — nothing draws it, which is the
// intended behavior for an addon written against a newer build.
// ---------------------------------------------------------------------------
std::vector<std::string> knownRegions();

// ---------------------------------------------------------------------------
// Host draw entry points.
// ---------------------------------------------------------------------------

// Floating panels (registered with an empty region). Call once per UI frame.
void drawAddonPanels();

// Inline mount point. Each panel is wrapped in its own ImGui id scope and, when
// it has a title, a collapsing header. Safe (and cheap) to call unconditionally:
// it returns before touching the GIL when nothing is registered for the region.
void drawRegion(const char* region_id);

// Same, without the collapsing-header wrapper — for menu-bar regions where the
// addon draws rt.ui.menu_item() entries directly.
void drawMenuRegion(const char* region_id);

// Cheap registry probe. No GIL, no Python.
bool hasRegion(const char* region_id);

// ---------------------------------------------------------------------------
// Panel registry inspection — drives the View menu's addon panel list.
// ---------------------------------------------------------------------------
struct PanelInfo {
    int id = 0;
    std::string title;
    std::string region;     ///< empty => floating window
    bool visible = true;
};

std::vector<PanelInfo> listPanels();
void setPanelVisible(int panel_id, bool visible);

// Shutdown: drop every py::function while the interpreter and GIL are alive.
void clearPanels() noexcept;

} // namespace rtui
