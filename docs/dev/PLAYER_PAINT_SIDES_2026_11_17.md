# Player room paint and ghost wall — 17 November 2026

## Scene work completed live

Original Player project had 113 objects but only 8 paint surfaces. The second
import duplicated the floor, ceiling and furniture in the same positions. Its
unpainted geometry could obscure the painted first set. After user approval:

- Removed 44 redundant, previously hidden objects in a separate project copy.
- Preserved 24 game balloons and the remaining lamp geometry/emission behavior.
- Per user's subsequent choice, changed the 35 visible environment props to
  independent, neutral default Principled materials, one material slot per prop.
- Packed sibling mesh UVs into separate tiles before merging material slots.
  Each prop now has one paint surface; preserved the existing large-surface
  resolutions and used 512px for newly registered props. Paint uses color-only
  defaults. Collider assignments were refreshed after geometry edits.
- Saved `assets/scenes/pastel-play-room/pastel-play-room-simple.rtp` and loaded it
  into the already open Player. No app was launched and no C++ build was run.
  An earlier `pastel-play-room-paint-ready.rtp` copy retains the multi-slot setup.

The Player dab probe measured positive covered-texel deltas at all six points:
left/right wall 886/111, left/right floor 5033/4803, left/right ceiling 2244/2210.
This verifies the atlas path, not droplet contact. These were session-only test
marks; no Player paint was saved. Report: `PLAYER_SIMPLE_PAINT_PROBE_2026_11_17.json`.

## Ghost contact isolated

User observed invisible particle contact in front of the left chair/sofa group.
Removing only the left sofa collider did not help (user confirmation). It was
restored. A left armchair comparison was interrupted by the user's project reload;
after reload its collider was confirmed present again.

The geometry exported from Studio's live flat scene showed closed/manifold
armchairs, rug and sofas, but the replacement wall had **1890 boundary edges**
after position welding. Its 3414-triangle mesh is not a closed volume.
`GHOST_MESH_TOPOLOGY_2026_11_17.json` records the counts.

Removing only `2_Room_Wall PaintCollider` in Player removed the ghost barrier.
The user explicitly confirmed drops then reached the left side, and supplied
before/after images. Other furniture collider assignments were left enabled.
The currently running Player has the wall collider temporarily removed. This was
not saved: the project copy retains its authored wall collider for reload.

## Permanent source correction (build pending)

`ColliderGpuQuery.cpp` now checks whether a signed-cook triangle soup encloses a
closed, nonzero-volume region. Open surfaces use the existing unsigned-distance
shader mode; closed solids retain signed distance. This avoids ray-parity votes
on an open wall inventing a solid region inside the room. The small local helper
`Fluid/ColliderMeshClosure.h` welds position seams and checks edge use/volume.
The check runs only for one-off signed cooks; per-frame unsigned queries skip it.
No shader source, binding, push constant, foundational header or shared struct
layout changed. Existing scripting/IPC collider operations use the same cook.

Scope: this correction is in the GPU query/cook path. The older CPU fallback cook
was not changed or verified for open meshes; if a GPU cook falls back, do not
claim open-shell correctness. Verify `sdf_cooked_on_gpu` in Studio's `collider.get`.

Separately, `BalloonPaintDemo.cpp` deduplicates implicit surface object names when
an object has multiple painted material slots. Explicit duplicate user config
entries remain errors. The simple room uses one slot per prop and already works
with the running binary. The multi-slot fix is pending build too.

## Checks and handoff

- Live simple room: 35 paint surfaces, 24 bound balloons, prepared=true.
- Six atlas probes PASS; ghost source isolated by user-confirmed wall-only test.
- Python syntax and focused diff whitespace checks PASS.
- Added `tests/collider_mesh_closure_test.cpp`: closed tetrahedron, missing face,
  nonmanifold duplicate, zero-volume sheet, welded seam, empty/nonfinite input.
  Native test has not been compiled/run; user performs C++ builds.

User: incremental **Player** build (affected `ColliderGpuQuery.cpp` and
`BalloonPaintDemo.cpp`); Studio can follow when its preview fix is wanted. No
shader compilation is required. Then reload `pastel-play-room-simple.rtp` (or
recreate the wall SDF). Confirm the wall collider is present and ready, ghost
contact remains absent, and droplets paint both wall sides, floor and props.
This post-build acceptance is still pending. Do not treat a disabled collider as
the completed permanent repair.

Particle controls: Game Design / Balloon demo / Drops per balloon, currently 1800;
actual drops scale by balloon size squared. For 24 balloons and size_max 1.2,
3000–3600 fits the existing total budget. Particle contacts / Splat radius (m)
controls footprint (currently .09); Splat amount controls deposited intensity
(currently .9, max 1). These values were explained, not changed by Codex.
