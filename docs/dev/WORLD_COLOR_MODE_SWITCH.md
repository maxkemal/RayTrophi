# World color mode transitions

`World::setMode` preserves the flat background color separately from the spectral
sky sun tint. UI, Python `world.set_mode`, and IPC `world.set_mode` use this core
transition. Existing mode validation and errors remain unchanged.

World serialization stores `solid_color` and `sky_tint` alongside the active
legacy `color`. Older projects retain their active color; inactive flat color
defaults to (0.13, 0.14, 0.16).

Manual verification after a user build:

1. In Solid viewport, choose Solid Color atmosphere and a custom dark color.
2. Switch to Spectral Sky, then Solid Color: the custom color must return.
3. Repeat through HDRI, including a black custom color.
4. Repeat mode changes through Python and external IPC (`scripts/test/rt_ipc.py`).
5. Save while Spectral Sky is selected, reload, and switch to Solid Color:
   the custom color must return. Repeat saving in Solid Color.
6. Check RT and RayFusion still display the selected atmosphere and sky tint.

No build or live application verification was performed by the agent.
