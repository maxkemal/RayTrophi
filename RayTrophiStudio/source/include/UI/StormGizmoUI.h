#pragma once
struct UIContext;
// Storm cells and the storm domain drawn in the viewport
// (docs/dev/ATMOSPHERE_STORMS.md §7): every live cell as a column from the
// ground to the cloud base, coloured by phase; authored cells and the domain
// centre can be picked and dragged on the ground plane. Edits go through
// rtapi::setCloudsJson, the same path as the panel and IPC.
namespace StormGizmoUI {
// Viewport display preference only (no storm state lives here).
extern bool visible;
void drawViewport(UIContext& ctx, bool orthographic, bool showGizmos, bool& hit);
}
