"""RayTrophi example addon.

Demonstrates the Faz 4a addon contract: a folder under scripts/addons/ with an
__init__.py that exposes register() and unregister(). Enable it from the Python
console with:

    rt.addons.enable("example_addon")

The enabled state persists to addon_state.json next to the executable, so the
addon re-loads automatically on the next launch until you disable it with:

    rt.addons.disable("example_addon")

It also shows the two ways an addon can put widgets on screen:

  * rt.ui.register_panel(title, draw)          -> its own floating window
  * rt.ui.register_region(region, title, draw) -> a section INSIDE an existing
                                                  editor panel or menu

Call rt.ui.regions() to list every mount point this build draws.
"""

import rt

# Optional Blender-style metadata. rt.addons.list() surfaces these fields.
bl_info = {
    "name": "Example Addon",
    "description": "Frame-change logging, a floating panel, and World/View mount points.",
    "version": (1, 1, 0),
}

_frame_cb_id = None
_panel_ids = []

# Panel widget state lives on the addon side (immediate-mode: values go in, new
# values come back each frame).
_state = {
    "log_frames": True,
    "clicks": 0,
    "amount": 0.5,
    "tint": [0.9, 0.5, 0.2],
    "preset": 0,
    "note": "hello",
}

_PRESETS = ["Dawn", "Noon", "Dusk", "Night"]
_SIGNAL = [0.0, 0.4, 0.7, 1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1.0, -0.7, -0.4]


def _on_frame_change(frame):
    if _state["log_frames"]:
        print(f"[example_addon] frame changed -> {frame}")


def _draw_panel():
    """Runs every frame while the panel is open. Only rt.ui.* calls are valid here."""
    rt.ui.text("Example addon panel (Faz 4b)")
    rt.ui.separator()

    if rt.ui.button("Click me"):
        _state["clicks"] += 1
        print(f"[example_addon] button clicked x{_state['clicks']}")
    rt.ui.tooltip("Increments a counter and logs to the Python console.")
    rt.ui.same_line()
    rt.ui.text(f"clicks: {_state['clicks']}")

    _state["log_frames"] = rt.ui.checkbox("Log frame changes", _state["log_frames"])
    _state["amount"] = rt.ui.slider_float("Amount", _state["amount"], 0.0, 1.0)
    _state["note"] = rt.ui.input_text("Note", _state["note"])

    if rt.ui.collapsing_header("More widgets", default_open=False):
        _state["preset"] = rt.ui.combo("Preset", _state["preset"], _PRESETS)
        _state["tint"] = rt.ui.color_edit3("Tint", _state["tint"])
        rt.ui.progress_bar(_state["amount"], f"{_state['amount'] * 100:.0f}%")

    if rt.ui.collapsing_header("Containers & style", default_open=False):
        # Tabs. begin_tab_item returns True only for the selected tab, and
        # end_tab_item runs only in that case — same rule as ImGui itself.
        if rt.ui.begin_tab_bar("example_tabs"):
            if rt.ui.begin_tab_item("Table"):
                # A table: end_table only when begin_table returned True.
                if rt.ui.begin_table("scene_table", 2, height=90.0):
                    rt.ui.table_setup_column("Object")
                    rt.ui.table_setup_column("Triangles")
                    rt.ui.table_headers_row()
                    for obj in rt.scene.objects()[:5]:
                        rt.ui.table_next_row()
                        rt.ui.table_next_column()
                        rt.ui.text(obj["name"])
                        rt.ui.table_next_column()
                        rt.ui.text(str(obj["triangles"]))
                    rt.ui.end_table()
                rt.ui.end_tab_item()
            if rt.ui.begin_tab_item("Styled"):
                # Styling lets an injected section match the host theme.
                rt.ui.push_style_color("button", [0.25, 0.55, 0.35, 1.0])
                rt.ui.push_style_var("frame_rounding", 8.0)
                if rt.ui.button("Themed button"):
                    print("[example_addon] themed button clicked")
                rt.ui.pop_style_var()
                rt.ui.pop_style_color()
                rt.ui.plot_lines("Signal", _SIGNAL, height=50.0)
                rt.ui.end_tab_item()
            rt.ui.end_tab_bar()

        # A child region scrolls independently. end_child ALWAYS runs.
        rt.ui.begin_child("notes", 0.0, 60.0, border=True)
        rt.ui.text_wrapped("Child regions scroll on their own, so a long list "
                           "does not stretch the whole panel.")
        rt.ui.end_child()

        if rt.ui.button("Open modal"):
            rt.ui.open_popup("example_modal")
        if rt.ui.begin_popup_modal("example_modal"):
            rt.ui.text("Modals work from a script too.")
            if rt.ui.button("Close"):
                rt.ui.close_current_popup()
            rt.ui.end_popup()


def _draw_world_section():
    """Mounted into the World tab — appears docked AND in a torn-off World window."""
    rt.ui.text_disabled("Injected by example_addon into properties.world")
    if rt.ui.button("Sun to noon"):
        rt.world.set(sun_elevation=90.0)
        print("[example_addon] sun elevation -> 90")
    rt.ui.same_line()
    if rt.ui.button("Sun to dusk"):
        rt.world.set(sun_elevation=3.0)
        print("[example_addon] sun elevation -> 3")


def _draw_view_menu():
    """Mounted into the View menu. Use rt.ui.menu_item() here, not buttons."""
    if rt.ui.menu_item("Example Addon: reset counter"):
        _state["clicks"] = 0
        print("[example_addon] counter reset")


def register():
    """Called when the addon is enabled (and on startup if it stays enabled)."""
    global _frame_cb_id
    _frame_cb_id = rt.on_frame_change(_on_frame_change)
    _panel_ids.append(rt.ui.register_panel("Example Addon", _draw_panel))
    _panel_ids.append(
        rt.ui.register_region("properties.world", "Example Addon", _draw_world_section))
    _panel_ids.append(rt.ui.register_region("menu.view", "", _draw_view_menu))
    print("[example_addon] registered")


def unregister():
    """Called when the addon is disabled or on shutdown. Must undo register()."""
    global _frame_cb_id
    if _frame_cb_id is not None:
        rt.remove_frame_change_callback(_frame_cb_id)
        _frame_cb_id = None
    for panel_id in _panel_ids:
        rt.ui.unregister_panel(panel_id)
    _panel_ids.clear()
    print("[example_addon] unregistered")
