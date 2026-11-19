# Curved-wall paint acceptance — 17 November 2026

Open Player: `pastel-play-room-simple.rtp`. The ghost-volume fix was accepted by
the user; remaining gaps were confined to curved wall strips.

Measured wall state was mesh_sdf, ready in all four pools, resolution mode 1
(64^3). Paint splat radius was .09m and facing_min was 0. The room spans over 8m,
so the coarse distance lattice can displace a curved contact beyond this footprint.

Only the live wall collider was recooked at mode 2 (128^3), through the existing
external `paint.wet_demo operation=set_collider` service. No paint reset, camera,
splat, facing filter, balloon count, droplet count or other collider edit.
All four systems reported ready. The user explicitly confirmed that curved
strips then received paint. This is a successful live A/B observation; it is not
a claim of exact geometric distance error or performance measurement.

Before/after snapshots: `PLAYER_CURVED_WALL_BEFORE_128_2026_11_17.json` and
`PLAYER_CURVED_PAINT_AFTER_128_2026_11_17.json`.

## Save/load correction

ProjectManager's particle collider serialization omitted sdf_resolution_mode,
although the separate SceneSerializer path already handled it. Thus the saved
room reopened at the default 64^3. Added only the missing write/read wiring in
`ProjectManager.cpp`: optional integer field `sdf_resolution_mode`, bounded 0..4,
with old documents retaining the existing default. No foundational header,
struct layout, shader or public method change.

Focused diff check PASS. No C++ build/application launch. Save/reopen acceptance
of the new field awaits user incremental Studio/Player builds for ProjectManager.
After rebuild: set the wall to 128^3 in Studio, save, open in Player, and confirm
mode 2 and ready in all four pools before repeating the curved-wall test.
The current Player session is already corrected; an old binary can still reopen
the old saved project at 64^3 until this save/load path is rebuilt and resaved.
