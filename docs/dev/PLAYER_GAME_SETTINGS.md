# Player / Game settings

2026-11-19: source ready; C++ build and live acceptance pending. No C++ build,
application launch, scene edit or package refresh performed by Codex.

The operator console has a Game settings tab in the same visual style as the
balloon game cards. Apply / Save / Play remain in the fixed footer on this tab.
Only fields actually edited are applied. Mixed paint values are labelled; editing
one applies it to every registered wet-paint surface without replacing the other
surface-specific paint properties. Pending drafts can be discarded; the Play button
is disabled until they are applied, saved or discarded. Escape can still close the
console and leaves the current live settings unchanged.

## Personnel controls

| Control | Meaning / limits |
|---|---|
| Players pop / Automatic demo | Existing `autostart` choice. Manual preserves mouse/body input; automatic also schedules random pops. Explicit Apply changes the mode; presets never change it. |
| Balloon count | 3 up to the prepared mesh capacity and existing particle budget, at most 24. No synthetic mesh cloning. |
| Return after pop | New canonical demo config `respawn_seconds`, 0.25..60 seconds, default 3.5. |
| Inflate / playful birth | New `birth_seconds`, 0..3 seconds, default 0.75; 0 restores instant appearance. Applies to flat mesh balloon visuals. |
| Automatic pop min/max | Existing interval settings, 2 <= min <= max <= 30 seconds. |
| Drying time | `lifetime_seconds`, API 0.05..600 seconds; personnel slider up to 120. Drying does not erase dry colour. |
| Paint spread / flow | Existing diffusion/runoff, 0..2; all registered surfaces. |
| Paint thickness | `paint_thickness`, 0..1 relief limit for new paint on all registered surfaces. Each deposit fills part of the remaining relief. A positive value enables the height/normal channel. A value of 0 stops new height deposition. Lowering the value preserves existing ridges. Colour remains mixable at the relief limit. |
| Texture resolution | 512/1024/2048 through the existing bulk Paint Edit resize service. |
| Reset after inactivity | Existing `idle_reset_seconds`: 0 disables, otherwise 10..3600 seconds without a player pop, once the room is dirty. |
| Reset after finale | Existing `finale_reset_seconds`: 0 disables, otherwise 3..120 seconds after the room-goal finale. |
| Room progress / finale | Existing `room_goal`: 0 disables; otherwise 0.05..1 of paintable coverage. Finale reset has no effect when the goal is disabled. |

Presets are draft patches, obtained from the shared service. They do not mutate the
game until Apply or Save, do not select automatic mode, and do not change resolution:

| Preset | Balloons (capped) | Return | Auto interval | Drying | Spread / flow |
|---|---:|---:|---|---:|---|
| Calm | 6 | 6 s | 6..9 s | 20 s | 0.12 / 0.15 |
| Balanced | 12 | 3.5 s | 2.5..5 s | 12 s | 0.24 / 0.35 |
| Lively | 18 | 2 s | 2..3 s | 8 s | 0.35 / 0.55 |

Changing demo controls calls the existing validated demo configure service, preserving
unexposed authored configuration. This renews balloons and clears in-flight effects.
Registered surfaces, their existing paint and untouched paint settings are retained;
the config deliberately omits the sidecar `surfaces/resolution` rebuild path. Changing
texture resolution clears transient wet paint as the existing resize operation does.
Applying paint-only properties does not reconfigure the demo.

## Quick actions and shortcuts

- **Clear paint / F8:** calls existing queued `resetWetPaint`. Keeps game and balloons.
  Droplets still in flight can paint again as they land; this is not a simulation reset.
- **Renew popped balloons:** canonical demo `reset_balloons` operation, reinflates only
  popped balloons, preserves paint and running mode, cancels queued finale cascade/reset.
  Already inflated visual particles are not duplicated.
- **Reset room / F9:** existing room reset; clears paint, renews popped balloons and
  resets room progress. Its existing delayed second clear handles remaining droplets.

F8 is new personnel input. Kiosk/attendant key gates must restrict it alongside F9;
this change adds no new roles or PIN implementation and does not alter existing gates.
F8 uses the same operator service as the panel, Python and IPC. No gameplay-specific
logic is implemented in PlayerMain.

## Session versus saved defaults

Apply affects the current scene session only. Save applies the patch and saves current
uniform personnel values under the canonical scene path in **player_game_settings.json**
beside the executable. Mixed surface fields are omitted from the stored override.
The original `.rtp` and companion geometry are never saved or modified by this feature.
Other games have independent profiles. The normal package refresh preserves this local
file because it copies generated payload without deleting unknown local settings.

On project load, the shared Player tick waits for the runtime scene and queued Game
Design to finish loading, then applies the saved override once per scene revision.
The baseline authored demo supplies mesh capacity even after reducing the live count.
Same-path reloads are identified by runtime `scene_revision`, not only the filename.
Profile application failures are shown and are not retried every frame.

**Restore the authored game** removes only this game's saved override, then schedules
the existing project loader between UI frames. It reloads all original authored state,
including per-surface mixed resolutions, and loses current session paint/changes.
The UI makes that reload explicit. Camera input must be stopped first. This avoids
pretending a partial field reset can reconstruct an arbitrary authored scene.

## Shared scripting / IPC contract

All operations use `rtapi::playerOperator` -> `PlayerGameSettings`; they are available
through existing `player.operator`, registered `rt.player.operator`, and external
`PlayerOperator.command`. The existing operator endpoint's capabilities remain
Admin|SceneWrite|FilesRead|FilesWrite. No shader, binding layout or ABI change.

| operation | config / result |
|---|---|
| `game.settings.status` | Empty. ready/project/revision/values/max_balloons/saved/error/storage; null values mean mixed surfaces. |
| `game.settings.preset` | `{name: calm|balanced|lively}` -> `{patch: ...}`; no mutation. |
| `game.settings.apply` | Optional supported personnel fields from the table; applies edited fields only. |
| `game.settings.save` | Same patch, or empty to save the current live personnel settings. |
| `game.settings.restore` | Empty -> scheduled/reload_project; camera must be stopped. |
| `game.settings.clear_paint` | Empty -> queued paint reset. |
| `game.settings.renew_balloons` | Empty -> existing demo status after renewal. |

Mesh balloon games expose `wind_strength` (0–1, default 0.25), `jolt_motion`
and `impact_soft_body`. Wind updates the live simulation without renewing
balloons or resetting paint. The physics toggles rebuild the balloon demo and
are also saved in the local per-scene Player profile.

`player.operator(status)` also includes `game_settings`. Direct
`paint.wet_demo(configure)` accepts `respawn_seconds`; direct
`paint.wet_demo(reset_balloons)` and `rt.paint.wet_demo("reset_balloons")` call the same
core used by the personnel action. Legacy project configs without respawn_seconds
retain the 3.5-second default and Game Design saves the canonical config on Studio save.

Types, unknown fields, limits, interval ordering, capacity/budget and paint settings
are validated before mutation. Existing scene/device/configure/resize failures can
still leave partial live results; no transactional GPU rollback is promised. Save
publication errors are reported. Invalid saved profiles leave authored state active
and show the error. Successful apply reports balloons_reprepared/wet_paint_cleared.

## Validation after user build

Source checks: new source registrations and /EHsc in both targets, Python regression
syntax, descriptor/capability audit, focused whitespace checks. No live acceptance
or C++ compilation has been performed for this slice.

1. Incrementally build Studio + Player. No shader compilation is needed.
2. Open a disposable prepared paint game. Inspect Game settings at 1920x1080 and
   800x600; presets, mixed labels, scroll/footer Apply/Save/Play and discard behavior.
3. Read/validation/preset contract, externally:
   `python scripts/test/rt_player_game_settings_ipc.py`
4. Full mutation test on a disposable game without a saved Player profile/camera:
   `python scripts/test/rt_player_game_settings_ipc.py --exercise-disposable`
   This tests apply-all surfaces, manual mode, respawn/reset parameters, quick actions,
   save/reload persistence, removal/reload cleanup and unchanged source `.rtp` SHA-256.
5. Physically time a popped balloon's return; test min/max automatic intervals and
   verify presets do not turn on automatic popping. Play still respects manual mode.
6. Time idle room reset (after a player pop) and finale reset with an enabled room goal;
   test 0/off independently. Check F8 clears only paint, F9 resets the room, and renewal
   does not duplicate visuals or erase paint.
7. Test saved game profiles independently for two scenes, same-path reload, restart,
   missing/invalid profile data, mixed authored paint settings and explicit restore.
8. Build-dependent standard Player repack is a later step; current dist is unchanged.


## Elastic balloon birth / source ready

Flat balloon meshes now appear at 8% target size, inflate with a gentle back-ease
overshoot and damped squash/stretch, then settle exactly to the authored target scale.
The default is 0.75 game seconds; `birth_seconds` is shared by demo configure, Game
Design serialization, Player game settings, scripting and IPC. The original size still
controls drop quantity, sound and paint budget. Screen picking follows the growing
visual envelope instead of retaining a full-sized invisible target.

Animation uses the existing 30 Hz absolute instance-transform push from the authored
base. It never compounds the previous frame's scale and adds no per-frame CPU vertex
rebake or readback. Even with float_amplitude=drift_speed=0 it runs through the final
identity pose, then stops pushing transforms. Initial prepare, popped-balloon respawn
and manual renewal share the same birth path. The legacy fixed-particle fallback is
unchanged and remains instant; this is a flat mesh balloon animation.

Read-only status: each balloon adds `birth_active` (until final transform delivery)
and `visual_scale` (local XYZ multipliers relative to its target size). Actual shader,
geometry layout and descriptor bindings are unchanged.

After user build, run `python scripts/test/rt_balloon_birth_ipc.py --exercise-disposable`
in a disposable mesh-balloon game. It changes transient demo config only and restores
it finally; verifies tiny start, overshoot, exact finish, zero-motion idle, respawn,
unchanged target size and the instant option. `tests/balloon_birth_animation_test.cpp`
covers curve bounds/endpoints, positive scales, damping and invalid inputs; it is
provided for user-run C++ testing and was not compiled by Codex. Visually inspect
normal/gold balloons, click picking while tiny, repeated respawns and the birth slider.
