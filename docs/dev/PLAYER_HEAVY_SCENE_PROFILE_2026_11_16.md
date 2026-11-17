# Heavy-scene Player measurement and fixed render resolution

2026-11-16, external IPC against user-opened applications. Scene:
`E:/rt_k/oyuncak.rtp`, 1627 draw calls/flat meshes, about 3.72 million visible
triangles, 11 scene lights and 8 shadowed lights. No application was launched,
no C++ build run, no shader changed, no project saved, no screenshot saved.

## Player, continuous uncapped drawing

Capture disabled to preserve the normal asynchronous presentation path.
One warmup and about four seconds of measurements at each scale. Display,
capture and pacing settings were restored afterward.

| Render target | Drawn FPS | Render + upload cost | GPU total | Shadow atlas GPU |
|---|---:|---:|---:|---:|
| 1680×945 | 11.38 | 85.71 ms | 59.98 ms | 33.10 ms |
| 1260×709 | 12.71 | 77.06 ms | 51.51 ms | ~33 ms |
| 840×473 | 13.90 | 70.45 ms | 43.88 ms | ~33 ms |

At full resolution: upload about 1.00 ms, runtime advance about 0.80 ms,
main pass GPU 11.69 ms, transmission GPU 8.21 ms, RT shadow GPU 5.51 ms.
CPU shadow recording measured 77.73 ms uncapped. CPU/GPU stage times overlap
and must not be added. A large CPU recording interval can include driver waits.
Reducing pixels does not remove the fixed shadow-atlas work.

## Studio comparison and pacing control

Same project, camera position/target/FOV, 1680×945 target, scene lighting,
depth prepass, GPU culling, 1627 draws, 8 shadowed lights, reflection/GI disabled.
Studio camera was restored exactly to its prior position/target/FOV afterward.
The Studio camera had a slightly different target initially; initial mixed/cold
samples are not the matched comparison.

Studio: 41 warmed frames driven by zero-angle redraw requests, CPU frame mean
27.20 ms, GPU 60.49 ms, shadow recording CPU 24.32 ms / GPU 33.12 ms.
These are frame costs, **not sustained Studio FPS**: the timing report explicitly
warns that the frames are individually driven.

Player control at 10 Hz pacing: 39 frames, CPU frame mean 26.91 ms, GPU 60.54 ms,
shadow recording CPU 24.38 ms / GPU 33.18 ms. The measured drawn rate 9.45 FPS is
artificially capped and is not a throughput result. Previous display settings
were restored.

Conclusion: the large uncapped Player CPU interval does not establish an
intrinsically slower shadow implementation: the same Player reaches the Studio
per-frame CPU cost at lower request pressure, while GPU work remains effectively
identical. GPU/driver backpressure or recording/submission overlap is a candidate,
not a proven source-level root cause. No performance optimization is claimed.
The main measured GPU cost is the shared shadow atlas. Removing the editor UI
alone does not remove that work. A fair sustained Studio-vs-Player FPS comparison
is still outstanding.

Raw measurements (ignored local files):
`tools/perception_bridge/out/player-heavy-scene-profile.json`,
`studio-heavy-scene-matched-profile.json`,
`player-heavy-scene-paced-profile.json`.
Earlier Studio files preserve the cold and unmatched-camera windows separately.

## Fixed render extent source change

Previously window resize/fullscreen marked the render targets dirty and rebuilt
them at `SDL output × render_scale`, changing scene resolution with the window.
`Core/PlayerRenderExtent.h` now keeps the initial render surface dimensions as
the immutable render base. F2 and the existing `player.set_display.render_scale`
operation scale that base. Window resize, IPC width/height and fullscreen only
scale presentation; they do not recreate render targets or change projection.
The HUD reports render and output separately. `player.display.render_base` is
read-only; width/height retain their window-size meaning. The descriptor notes
were updated. No new renderer, shader, public struct layout or IPC command.

The PlayerMain edit is confined to display target sizing/status/resize redraw;
the kiosk/operator-key branch was not changed. The coordination board records
this boundary for the agent owning kiosk work.

Validation: Player include boundary 665/665 clean; IPC descriptor/capability
audits, Python acceptance CLI/parse and focused diff checks PASS. User manual fixed-extent acceptance is now PASS: the user tested the new
resolution behavior and confirmed it works. The automated display acceptance
script has not been run against that binary.

With rebuilt Player open, from an external terminal:

```powershell
python scripts/test/rt_player_display_ipc.py --project E:/rt_k/oyuncak.rtp --seconds 3
```

The test reloads this supplied project, verifies scale against render_base,
resizes the window to 960×540 and 1200×800 without changing render resolution,
checks fullscreen preserves the target, and restores window/display/capture
settings. Also manually drag-resize the Player: output changes, render stays
fixed. Press F2: render changes while the window stays fixed. No shader compile
is needed. The user subsequently confirmed the fixed-resolution behavior works; this is
manual acceptance, not a claim that the automated script was executed.
