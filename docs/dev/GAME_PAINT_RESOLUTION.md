# Game paint texture resolution

Game Design > Wet paint surfaces has one Paint texture size selection (512/1024/2048)
and Apply to all surfaces. Current surfaces displays their actual size or mixed sizes.
Changing the selection alone does not resize textures. Apply uses the same
MeshPaintAdapter::resizeTextureSet service as Paint Edit, rebuilds all registered
wet paint surfaces, and preserves each surface's paint settings and contact settings.
Transient wet paint is cleared when a size changes. An already matching size is a no-op.
New surface registration uses the selection; existing textures still need Apply.
Save the project after applying. game_design stores each actual surface resolution;
load resizes existing Paint Edit textures to the saved size before atlas baking.

Python: rt.paint.wet_resize_surfaces(1024)
IPC: paint.wet_resize_surfaces with {"resolution": 1024}; success {"ok": true}.
Both use the same core as UI. Supported range 32..4096. Rejects unavailable flat
meshes/textures, no registered surfaces, pending project load and active render jobs.
Preflight failures leave state intact. A failure during resize/rebake reports objects;
partial results can remain (no transactional rollback).

## Verification after user C++ build

Build Studio and Player incrementally; no shader changes are required.
In a disposable Studio scene with at least two surfaces of different sizes:

1. Select 1024 and Apply: every surface detail must read 1024; settings unchanged.
2. Save, reopen, check all sizes; open that saved project in Player and query
   paint.wet_status: every surface must still read 1024.
3. Check Paint Edit shows the same actual texture size. Try 512 then 2048.
4. External automated Studio roundtrip (writes a new test project, clears paint):
   python scripts/test/rt_game_paint_resolution_ipc.py --roundtrip E:/tmp/paint-test.rtp
5. Python binding parity: rt.paint.wet_resize_surfaces(1024); invalid 31 raises an error.

Codex performs source/descriptor checks only; live acceptance awaits the user's build.
