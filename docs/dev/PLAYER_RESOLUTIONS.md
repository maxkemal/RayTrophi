# Player render resolution

F2 opens a scrollable supported-resolution list. Selection changes the render
target, not the presentation window. Manual window resizing and F11 fullscreen
continue to scale presentation only and keep the selected render resolution.

At startup, 1920x1080 is selected/recommended if advertised by the active monitor.
Otherwise its desktop resolution is selected/recommended. Choices come from SDL
display modes, deduplicated across refresh rates, up to the highest OS-advertised
mode (with the existing 16384-pixel safety limit per axis). OS query failure preserves
the initial render target and offers no
invented modes. Moving to another monitor updates the available choices, without
silently changing the current target. There is no saved machine-specific override.

## Shared controls

IPC `player.display` reports `resolutions` (arrays of [width, height]) and
`recommended_resolution`. `render_base` is the selected resolution; `render` is
the actual target after `render_scale`, updated on the next main-loop iteration.

IPC `player.set_display` accepts `render_width` and `render_height` together.
They must be integral and match a currently listed supported mode. An invalid
pair rejects the entire patch before mutation. Selecting a resolution resets
`render_scale` to 1 unless a scale is explicitly supplied in the same patch.
Existing `width`/`height` remain window dimensions only. `render_scale` remains
available for performance probes (0.25..1 of the selected base).

Python uses the same display host:

```python
rt.player.display()
rt.player.set_display({"render_width": 1920, "render_height": 1080})
```

Studio reports that these controls require the Player runtime.

## Verification after user build

1. Build the Player C++ target (Codex does not run C++ builds).
2. Open Player; on a Full HD-capable monitor check startup render is 1920x1080.
3. F2: select another mode, close/reopen with F2 and Escape; scroll a long list.
4. Resize the window and toggle F11: the HUD render size stays selected.
5. Run externally: `python scripts/test/rt_player_resolution_ipc.py`.
6. On a smaller monitor verify its desktop resolution is recommended; on a
   higher-resolution monitor verify Full HD is recommended and higher modes
   remain selectable. Monitor movement must not change the current render size.

No shader changes are involved. Live checks require the newly built executable.
