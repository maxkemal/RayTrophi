# Player clip transitions — 2026-11-16

Player exposes the existing shared animation services for `anim.clips`, `play`,
`stop`, `status`, `set_paused`, `set_time`, `set_speed`, `graph_status`,
`set_graph_param`, `trigger_graph_param`, `state_machines` and `force_state`.
The existing Python `rt.anim` bindings and IPC call the same rtapi functions.
No new animation evaluator, geometry path, shader or transport was added.

`anim.play(character, clip, blend=0.3, layer=0)` now explicitly selects that
character's existing controller and Animated pose view. Previously a graph-driven
character accepted this command but the renderer continued evaluating its graph.
Base-layer play resumes a paused controller. The graph objects are preserved;
reopening the authored project restores its saved playback source. This runtime
operation is not an undoable authoring command. Saving later in Studio can persist
the selected controller mode through the existing project serialization.

For graph handover, the first reported graph clip's phase seeds the controller's
existing crossfade. This does not reproduce the complete pose of a composite
blend/IK/state-machine graph. Game graphs should use their authored parameters,
triggers and state-machine commands for graph-to-graph transitions. `anim.status`
reports the controller; use `anim.graph_status` when `anim.character.uses_graph`
is true. A non-base-layer handover returns
`animation_graph_requires_base_layer_play_first`. Time/speed/stop commands also
reject graph ownership with that error rather than claiming an invisible edit.

Play rejects invalid clips/layers, nonfinite or negative blend duration, active
external tracking, Rig Edit/Pose, and final-render scene lock before changing
playback. External tracking must first be cleared explicitly. Time and speed
must be finite and nonnegative. Validation is shared by scripting and IPC.
The existing controller stops immediately; its blend-out parameter does not
implement a fade to bind pose. The acceptance test requests `blend_out=0`.
`anim.set_loop` remains outside the Player allowlist because the existing
controller setter is a stub. No kiosk capabilities or security roles are widened.

FBX import and undoable same-rig binding remain Studio authoring operations.
The Player consumes clips stored in a prepared `.rtp` project. Source FBX files
and the original project are never overwritten by the preparation script.

## Build and live acceptance

Source changes require **incremental Studio and Player builds by the user**.
No C++ build or application launch was performed by Codex. No shader compilation
is needed. Source checks alone are not live acceptance.

Save any current Studio work before preparation: this script replaces Studio's
open scene, imports the additional FBX as a temporary character, validates and
clones its clip onto character `1`, deletes the temporary character using the
shared lifecycle service, saves a new output, then reopens it to check persistence.
It refuses an existing output file and does not fall back to a different mapping.
If mapping fails, its error is retained and no output project is saved.

From an external terminal, with the rebuilt Studio open:

```powershell
python scripts/test/rt_prepare_player_clips_ipc.py --project tools/perception_bridge/out/wiezzorek-balloon.rtp --fbx "E:/3d model_arsiv/mixamo_arsiv/wiezzorek/Block With Rifle.fbx" --output tools/perception_bridge/out/wiezzorek-two-clips.rtp
```

The new project does not receive a `.balloons.json` sidecar: balloon startup is
independent of this animation acceptance. Load the new project in rebuilt Player,
keep its window foreground, then run:

```powershell
python scripts/test/rt_player_clip_transitions_ipc.py --character 1 --output tools/perception_bridge/out/player-clip-transitions.json
```

Checks: invalid play requests preserve character settings; A→B→A reports actual
crossfades and settles on the requested clips; canonical bone snapshots move;
pause holds pose and time; resume moves again; animation/skin counters advance;
no device loss or runtime/simulation error; immediate stop clears playback.
The test never saves or changes capture settings. After any mutation it reopens
the same disk project, restoring authored playback and resetting simulation.
Visible mesh deformation and the quality of the crossfade still need viewport
observation. Imported FBX compatibility and live acceptance remain NOT_RUN until
the user builds and these tests run against the new binaries.

## Live acceptance and zero-weight crossfade fix

2026-11-16, after user builds: Studio import, same-rig binding, temporary source
character deletion and save/reopen PASS. Separate project:
`tools/perception_bridge/out/wiezzorek-two-clips.rtp`; character `1`, 79 bones,
`1_mixamo.com` (1.4167 s) and `1_mixamo.com_2` (1.6333 s, Block With Rifle).
Original project and FBX files were not overwritten.

Player loads both clips. Immediate cuts in both directions PASS, with measured
joint excursions 1.0888 m and 1.6669 m. Pause holds pose exactly (0 m movement),
resume moves it again (1.6602 m), immediate stop clears playback. No GPU loss or
runtime/simulation errors. The test reopened the authored project afterward.

The actual crossfade test FAILED: `play` schedules clip B at weight zero, but
`AnimationBlendState::isBlending()` excluded zero, preventing `updateLayer()`
from ever advancing B. Source fix accepts zero weight when both A and B exist;
idle/completed states remain excluded. This corrects the shared controller,
including queued transitions, without adding a Player-specific evaluator.
Only an inline predicate changed; no struct layout or shader changes. The user
must rebuild Player (and Studio to share the fix); dependent files will recompile.
Crossfade acceptance remains pending that rebuild and a rerun of
`rt_player_clip_transitions_ipc.py`. Existing reports preserve the failed test
and the independent successful immediate-cut probe:
`tools/perception_bridge/out/player-clip-transitions.json`,
`tools/perception_bridge/out/player-clip-cuts.json`.

## Manual Space-key acceptance bridge

`rt_player_clip_keys_ipc.py` is an external test client, not a shipped Player
input mapping. It reads only Space/F8 while the foreground window belongs to
the connected pipe server process; other apps cannot trigger clip commands.
It calls the same `anim.play` service, handles one switch per press, records
the result, and leaves the chosen clip playing without saving the scene.
F8 ends the bridge; the timeout bounds its lifetime. Run in a separate terminal:

```powershell
python -u scripts/test/rt_player_clip_keys_ipc.py --character 1 --duration 600 --blend 0 --output tools/perception_bridge/out/player-clip-keys.json
```

Space alternates the two clips in the prepared project. `--blend 0` verifies
immediate selection only. After the corrected Player binary passes the automatic
crossfade test, use `--blend 0.65` to observe and verify soft transitions. Positive
blend mode requires an observed crossfade and a settled target clip; it never
silently falls back to cuts. Latest live retry still stalled on the first clip
with positive blend. This manual cut test cannot close crossfade acceptance.

### Visual skin regression blocks acceptance

The manual bridge was stopped after the user observed severely stretched skin,
before any Space events were recorded. Merely selecting the original walking
clip through `anim.play` reproduces the distortion in controller mode. Reopening
the exact same two-clip project restores a visibly correct graph-driven walk.
No weights, source FBX or project files were edited during this keyboard test.
This is not evidence of retarget damage: the original clip also fails.

At approximately matched clip phase, graph/controller joint matrices were close
(largest component difference about 0.00042), despite the visible skin failure.
The joint probe does not observe the actual raster skin matrices. Controller
skin matrix production/transfer and raster consumption still need isolation.
Previous joint/counter PASS results do not close visible deformation acceptance.
The Player was returned to graph playback; the keyboard process was terminated.
Reports: `player-controller-graph-comparison.json` and `player-clip-keys.json`
under `tools/perception_bridge/out`. No skin source fix is claimed yet.

## Shared palette publication and Fire/Push source package

2026-11-16: source changes now route graph and controller hierarchy globals
through `Animation/CharacterSkinPalette.h` before raster/CPU skin consumption.
This removes the separate copy of the controller's cached skin palette at that
boundary. Publication uses the context character, its canonical global bone
slots, per-model inverse and bind offsets, preserving other characters and sparse
indices. Missing weighted globals/offsets reject partial publication. No scene
weights, retarget data, geometry, shaders or public struct layouts were changed.
The legacy controller's own cache remains available to its other existing callers.

This is a corrective publication change based on the observed agreement of
joint globals; the exact pre-build discrepancy in the consumed raster palette
has not been measured. It must not be declared a confirmed skin fix until the
same first clip passes visible controller-vs-graph acceptance in the new binary.

Both UI Fire buttons now call the same `rtapi::triggerAnimGraphParam` used by
Python and IPC. They no longer queue an additional trigger on the editor asset.
That extra editor trigger could remain after runtime consumption and be cloned
again on Push. Push now explicitly clears transient triggers on its new runtime
clone. The shared trigger service also wakes drawing with `start_render=true`.
UI service errors are displayed instead of leaving a dormant asset-only trigger.
Push remains a topology publication operation, not a transition command.

Validation delivered: native `tests/character_skin_palette_test.cpp` checks a
centimetre bind vertex following its joint, per-character inverse selection,
sparse global slots, other-character isolation and rejection without partial
publication. Native fixture has not been compiled or run (user builds C++).
External trigger fixture sends exactly one trigger for each of three attempts,
without Push, and restores the initial runtime state:

```powershell
python scripts/test/rt_anim_graph_trigger_ipc.py --character 1 --from-state q --to-state w --trigger 1 --cycles 3 --output tools/perception_bridge/out/studio-trigger-acceptance.json
```

Build checklist: incremental Studio + Player, including the previously pending
zero-weight crossfade predicate. No shader compilation or application launch by
Codex. No project reload is necessary for these header-only helper modules.
After user rebuilds/reopens, test Studio UI Fire repeatedly from q (return to q
between attempts), verify Push alone does not replay Fire, then run the trigger
fixture. In Player reopen the two-clip fixture, verify graph mesh first, then
run `rt_player_clip_transitions_ipc.py` while watching the original walking mesh
and second clip. Finally run the Space bridge with `--blend 0.65` to observe
both directions. Script kinematics/counters alone cannot close visible skin
acceptance. Both reported defects remain live-verification pending this package.


## Latest user acceptance — 2026-11-16

Supersedes the graph-pending status above: user confirmed Studio Fire works,
then configured both directions with None/Exit Time and observed the two clips
alternating. After loading the correct saved project in Player, the user confirmed
both automatic transitions and a visibly correct mesh. Graph/state-machine path
is user-accepted. An earlier Player attempt had loaded the old one-clip
wiezzorek-balloon.rtp and had no state machine; it was a different project.

This does not close direct anim.play controller skin, controller crossfade or
external Space-key acceptance. Those require separate tests against the new
binary; do not infer them from graph success. No new scene save was performed
by Codex for this acceptance update.
