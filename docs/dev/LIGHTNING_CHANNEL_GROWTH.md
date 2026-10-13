# Progressive lightning channel trees

2026-10-13: source ready; user shader/Studio build and visual/live acceptance pending.

The previous bolt had 16 main pieces and a 40-piece budget. Every piece grew over
the same 1/16 interval, irrespective of length; branch birth was estimated from a
root index, so a branch could extend before its parent endpoint arrived. Long
terminal branches were clamped to late birth times with too little remaining growth.

`Backend/CloudBoltGeometry.h` now builds one deterministic tree per event. The main
channel has 32 pieces with multi-scale displacement. Up to 80 total pieces include
three-dimensional forks, tapering twigs and a third hierarchy level. Fork directions
follow the local parent tangent rather than a single common sideways vector.
Changing time reveals the same topology; it never re-randomizes channel shape.
Cloud-to-ground, upward, vertical in-cloud and horizontal in-cloud events reuse the
same builder and their existing endpoint/cell selection. Ground branches are bounded
by the authored endpoint floor; this is not terrain/mesh intersection.

Each segment records birth, completion, parent and depth. Birth is at or after the
parent's completion, and the start is exactly its endpoint. Main-channel timing uses
arc length. Branch timing uses path travel with a bounded late completion; terminal
cloud twigs may develop after the main channel connects but finish by progress 1.35.
The shader holds then extends each tip, rather than uniformly wiping the whole tree.
Branches remain illuminated in the return stroke. Existing flash/re-stroke envelope,
cloud diffusion, scene illumination and channel occlusion consume the same packet.

This is a bounded procedural visual approximation. It does not solve electric-field
breakdown, leader plasma dynamics, charge conservation or streamer attachment to scene
geometry. Real leader/return-stroke development is very fast; the timing control is
a cinematic scale. See [NOAA stepped-leader and return-stroke description](https://www.noaa.gov/jetstream/lightning/how-lightning-is-created/jetstream-max-lightning-process-keeping-in-step).

## Shared authoring/API

`lightning.leader_fraction` defaults to 0.3 and accepts finite numbers in 0.05..0.65.
It determines the leader portion of `duration_s`, before the bright stroke. The
range leaves time for terminal growth and decay. Existing scenes without this field
use the new default. The shared CloudState validator, serialization, scene loading
and interpolation all retain it. The existing cloud UI exposes Leader formation
fraction; Python and IPC use the existing authority:

```python
rt.world.set_clouds(lightning={"leader_fraction": 0.5})
print(rt.world.get_clouds()["lightning"]["leader_fraction"])
```

IPC: `world.set_clouds` with `params.lightning.leader_fraction`. Invalid values and
types fail before scene mutation through the same parser/validator. `duration_s`,
`branching`, radius and brightness controls remain available. Existing undo/history
and cloud revision behavior is reused; no additional scene geometry is created.

`world.cloud_stats` / `rt.world.cloud_stats()` retains the legacy `segments` rows
and adds `segment_growth` metadata and `geometry_revision=2` per event. Main depth
is zero, forks/twigs depths 1..3, and root parent is -1. Event `growth` may now reach
1.4 (main connection is 1.0). Consumers must not interpret >1 as invalid.

## Thunder alignment

Automatic thunder observes the same lane/seed identity, storm-cell producer and
channel endpoints. It now submits at `leader onset + duration * leader_fraction`,
the bright discharge time, before adding distance / sound speed. It no longer
starts the close crack during a deliberately slowed faint leader. Cloud geometry
and thunder use the same duration variation; distance, deduplication and transport
still come from their existing shared services. This is still a trunk-based distance
approximation, not a distributed acoustic simulation of every branch.

## ABI, cost and user checks

CloudBoltSegmentGPU / CloudBoltSegG are now 48 bytes (three vec4 blocks); the event
budget is 80 and the packet carries 320 pieces. CloudParamsGPU is **16512 bytes**,
up from 6272. It is an SSBO, not a uniform buffer. All shaders consuming
cloud_common/lightning includes must be rebuilt with the C++ host. Shader layout
audit now covers the segment mirror. Per-event geometric loops can visit twice
the prior segment budget; no extra density taps, particles, meshes or BLAS/TLAS
updates are added. Rendering cost and visible quality have not been measured.

After the user builds:

1. Run the full shader build (`RayTrophiStudio/compile_shaders.bat`) and the normal
   Studio build. Ensure the application uses the newly built shared cloud shaders.
2. In the storm scene start with duration_s=0.8, leader_fraction=0.5, branching=1
   and shutter_s=0.03. Play: a faint origin grows stepwise, forks remain attached,
   smaller twigs emerge later, then the channel flashes and decays. Check ground,
   upward, horizontal and vertical cloud types in both Vulkan RT and RayFusion.
3. Run externally while Play is active:
   `python scripts/test/rt_test_lightning_growth_ipc.py --seconds 15`.
   This validates actual published trees, parent continuity, causal intervals,
   bounds and sampled leader activity; it does not judge rendered appearance.
4. Verify world.set_clouds/get_clouds round-trip, save/reopen and undo of the new
   field. Reject -1, 0.9, strings and nonfinite inputs atomically. Old scenes open.
5. Listen to a close event: the crack follows the bright discharge, then its existing
   long rumble. Check distant propagation delay, Pause/Stop/loop and event deduplication.
6. Compare GPU frame cost to the old storm scene and confirm channel rendering does
   not dominate the cloud march. Same-frame redraw and repeated sampling must not
   change geometry; wind translation must move the complete tree together.

The agent performed source/layout checks and probe-fixture tests only. It did not
build shaders/C++, launch Studio or execute the new geometry in the running old build.

## Fine structure and visible branches (2026-10-13, written, not built)

Live view: an upward ground strike 8-10 km long was 32 equal main pieces, so a
piece was a ~300 m straight run; below the cloud base only 4-5 kinks remained
and no branch was visible (fork roots were spread over points 3..26, mostly
inside the cloud).

- View rendering (`cloudBoltRayFine`, used by Vulkan RT raygen and RayFusion
  `cloud_scene.comp`): a piece the view sees larger than ~0.006 rad per
  sub-piece is split into up to 32 sub-pieces by midpoint displacement with a
  constant knot angle per level (amplitude halves with spacing; level-1 offset
  0.15 x piece length). Deterministic per segment (seed from its birth and
  completion progress, not its wind-drifting position), so the shape is stable
  and a growing tip reveals the same fine path. The wide faint glow stays on the
  coarse piece; core and tight sheath follow the fine path. Pixels farther than
  the displacement envelope from the coarse piece take the old single-piece path.
- Geometry (`buildCloudBolt`, `clearAirBelowY`): for ground strikes three in four
  fork roots are drawn from the main-channel points below the cloud base at the
  strike (scene y via `lightningSceneHeight`).
- Unchanged: the CPU tree, timing, the in-cloud flash source and the scene flash
  lights use the coarse segments.

## Scene attachment (2026-10-14, written, not built)

A ground strike used to end at `lightning.ground_altitude_m` (y = 0): on a
mountain the channel ran on into the rock and its branches, clamped to that
floor, piled up around one spot.

- `lightning.attach_to_scene` (default true; JSON, panel, IPC). The ground end is
  the scene point a sphere of the striking distance, descending on the leader's
  vertical, touches first (rolling sphere): max over candidates of
  `y + sqrt(d^2 - r^2)`. Striking distance 10 I^0.65 m with I 10..60 kA per event
  (45-140 m).
- Candidates (`Atmosphere/LightningStrikeResolver.cpp`): first surface down 25
  verticals (centre + 3 rings x 8): meshes through the scene BVH ONLY, terrains
  through TerrainManager. Not `MeshEdit::raycastSurface`: built for picking, it
  sweeps every object's triangles linearly when the BVH misses; most strike
  columns miss a 1 km, 2 M-triangle mountain mesh under a 30 km storm, and each
  new bolt stalled the frame for seconds (live, 2026-10-14). Plus scattered
  instance tops from `InstanceManager` (base pivot: `local_bbox.max.y -
  mesh_center.y` scaled), which the CPU BVH does not hold on Vulkan.
- Resolved once per event (cache keyed by the geometry seed and a coarse scene
  generation), fixed at onset: the ground end does not drift with the wind.
- Branches of a ground strike stop where they cross a floor 4 % of the drop
  (>= 15 m) above the strike point instead of being flattened onto it.
- Stats: `world.cloud_stats` lightning lanes carry `strike` (null = no
  attachment; the fallback altitude applied). `geometry_revision` 3.

