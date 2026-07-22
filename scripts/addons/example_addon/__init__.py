"""RayTrophi example addon.

Demonstrates the Faz 4a addon contract: a folder under scripts/addons/ with an
__init__.py that exposes register() and unregister(). Enable it from the Python
console with:

    rt.addons.enable("example_addon")

The enabled state persists to addon_state.json next to the executable, so the
addon re-loads automatically on the next launch until you disable it with:

    rt.addons.disable("example_addon")
"""

import rt

# Optional Blender-style metadata. rt.addons.list() surfaces these fields.
bl_info = {
    "name": "Example Addon",
    "description": "Logs a line on every timeline frame change.",
    "version": (1, 0, 0),
}

_frame_cb_id = None
_panel_id = None

# Panel widget state lives on the addon side (immediate-mode: values go in, new
# values come back each frame).
_state = {"log_frames": True, "clicks": 0, "amount": 0.5}


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
    rt.ui.same_line()
    rt.ui.text(f"clicks: {_state['clicks']}")
    _state["log_frames"] = rt.ui.checkbox("Log frame changes", _state["log_frames"])
    _state["amount"] = rt.ui.slider_float("Amount", _state["amount"], 0.0, 1.0)


def register():
    """Called when the addon is enabled (and on startup if it stays enabled)."""
    global _frame_cb_id, _panel_id
    _frame_cb_id = rt.on_frame_change(_on_frame_change)
    _panel_id = rt.ui.register_panel("Example Addon", _draw_panel)
    print("[example_addon] registered")


def unregister():
    """Called when the addon is disabled or on shutdown. Must undo register()."""
    global _frame_cb_id, _panel_id
    if _frame_cb_id is not None:
        rt.remove_frame_change_callback(_frame_cb_id)
        _frame_cb_id = None
    if _panel_id is not None:
        rt.ui.unregister_panel(_panel_id)
        _panel_id = None
    print("[example_addon] unregistered")
