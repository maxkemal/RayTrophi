# Scoped transport and acoustic comparison

2026-10-18 source delivery; user build and live acceptance pending. No build,
Studio launch or IPC run was performed. This continues the main audio plan.

The Audio workspace adds a collapsible Source scope / acoustic A-B section.
Selected source, transient checkbox group and whole-scene scope use the same
AudioService operations exposed dynamically to Python and IPC. Group membership
is a temporary UI target list, not a persistent project group system. Switching
service or removing a source clears/prunes that list.

## Common scope

`scope` is selected (default), group or scene. Selected requires exactly one
explicit source ID in `ids`; group requires 1..32 unique existing IDs. Scene
omits IDs and resolves all emitters at invocation. Empty scenes are valid no-ops.
Unknown IDs, duplicates, bad types and invalid scopes reject before mutation.
Neither API implicitly depends on current UI selection. No source names, gain,
loop, binding, telemetry or simulation configuration are changed.

## Transport

`audio.transport_batch(scope=..., ids=[...], action="play|pause|stop")` is also
`rt.audio.transport_batch(...)`. Action is required. Play reuses play_all's
whole-target asset/pose/resident-budget validation, sound preparation rollback and
shared scheduled start. Scene play skips unbound sources; explicit selected/group
play rejects unbound sources. Pause retains voices/cursors; stop releases scoped
voices. Existing automatic suspension, timeline hold and resume semantics apply.
Stop scope affects only sources in scope; the existing global Stop all also stops
other playback categories. Asset audition remains exclusive.

Input/budget/preparation failures start no scoped voices. A native start/device
failure after preparation may occur after another voice starts; no device-level
transaction is claimed. Sound preparation can reclaim completed voices. Returned
play IDs list bound targets; pause/stop IDs list all resolved targets.

`source_runtime` now reports automatic_suspended and playback_state:
playing, paused, stopped, timeline_held or automatic_suspended. Held/playing take
precedence over suspension. These are transient shared runtime diagnostics.

## Acoustic batch and A/B

`audio.acoustics_batch(scope=..., ids=[...], action=..., config=...)` is also
`rt.audio.acoustics_batch(...)`:

- configure requires config.enabled (boolean); remaining options are the existing
  spectral/reflections/diffuse/meters_per_unit/boundary_mode controls and defaults.
  Disable removes preview entries. Configure replaces the recipe, as single-source
  acoustics_preview does; enable returns the source to processed state.
- dry/processed omit config and require every target to have an existing preview.
  Dry bypasses direct filtering and reflected output without deleting settings,
  completed estimates or revisions. Processed restores the same configured model.
  Queries continue during dry comparison, keeping moving-source estimates current.

The whole configuration is staged before swapping live entries. Source eligibility,
diffuse omnidirectionality and the shared 32-preview budget apply to the complete
batch. Any validation failure preserves published entries and revisions. Internal
revision sequence numbers can have gaps from discarded staging. Results include
ids, action and entries; status exposes bypassed (absent means false). A/B does
not reset playback cursors or regenerate a recipe. Existing coefficient smoothing
still applies, and processed output may initially be pending/bypassed by solver
policy. Session preview settings remain transient, outside undo/project state.

The UI Enable action explicitly uses spectral=true, meters_per_unit=1 and
object_barrier, plus displayed reflection/diffuse toggles. It reports the defaults
before applying them; callers needing different calibration can configure through
the existing inspector or the same script/IPC service. Scene acoustic enable is
strict: nonspatial sources reject the batch rather than being silently skipped.

Both methods accept SceneWrite; transport additionally requires FilesRead, matching
its possible asset preparation. No new file-write capability is introduced.

## Consolidated verification

Existing rt_test_acoustic_playback_ipc.py now covers batch configuration, A/B recipe
revision preservation, invalid config/ID/duplicate rejection and pause/play.
It was extended but not run, following the user's request for fewer test rounds.
Run its existing disposable-scene acceptance at the next consolidated build gate:

```
python scripts/test/rt_test_acoustic_playback_ipc.py --report PATH_OUTSIDE_REPO.json
```

User builds Studio C++; no shader change in this slice. At that gate also check
selected/group/scene UI targets, mixed eligibility atomic failure, Stop all versus
scoped stop, automatic suspension/resume, timeline hold, moving A/B and source
removal. Compare per-source gains/bindings before/after; listen for smooth A/B.
Long-run, 16/32-source performance and export parity remain separate open gates.
