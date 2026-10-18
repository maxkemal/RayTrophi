# Authored physical history and measured-contact texture

Source checkpoint, 2026-10-18. UI remains deferred. No build, Studio launch or live
test was performed; unified-domain solver and tests are untouched. This continues
the pre-UI core work without claiming that every original audio milestone is done.

## Source history as project data

The optional emitter generator_history field now stores version=1,
duration_seconds, controls and events. Its row/range/pause/pitch validation is the
same canonical PhysicalAudioTimeline validation used by native replay and bake.
Missing histories in existing projects normalize to null. Unknown versions/keys,
invalid time ranges/order/types, unsupported source kind/pitch and incompatible
recipes reject before changing authored source data. Set generator_history=null
to clear it. Continuous histories require frame-zero activity controls; impact
histories support discrete energy events.

Existing source create/update/get/list/duplicate operations expose this field in
both rt.audio and audio.* IPC, through one emitter schema. Save/restore and undo
use the existing audio snapshot path; no private UI copy or external history file
is required. Duplicates receive their own source ID and a copy of authored history.
Compatible recipe/gain edits change future rendering intentionally; changing to
an incompatible recipe requires replacing or clearing history in the same update.
Authored gain is not duplicated in the stored history; it remains source authority.

Emitter normalization and parameter discovery were moved to focused
AudioEmitterSchema.cpp/.h so source create/update/load share one property registry
and AudioService.cpp stays below the workspace size limit. The same vector validator
also serves asset-append positioning. This is source/schema refactoring, not a
UI reorganization.

```python
rt.audio.update(id=source_id, generator_history={
    "version": 1, "duration_seconds": 2,
    "events": [{"frame": 0, "sequence": 1, "energy_j": 0.1}]})
# Omit duration/controls/events to use the authored history:
rt.audio.physical_bake_start(id=source_id, path="E:/AudioOutput/stored.wav")
# Or, after stopping the source, prepare replay with that same history:
rt.audio.physical_replay_start(id=source_id, start_frame=12000)
```

An explicit duration_seconds, controls or events argument selects an entirely
explicit history with normal defaults. There is no partial implicit merge with
stored rows, so old events cannot leak into a new explicit request. id/path and
replay start_frame alone do not override the stored history.

physical_history_finish(store=True) stores the finished sampled take through the
same source update. Its default remains store=False for compatibility. The operation
is now potentially undoable authoring; existing render-job authoring locks apply.
Adding path can also start bake from the take. Store type and output path validate
before source changes; if later job creation fails, the completed/stored take is
preserved and the result returns bake_error instead of falsely reporting a queued
job. Explicit bake apply still controls final file publication/asset import.
Histories remain sampled control reconstruction, not substep-complete solver data
or exact capture of device filter/resampler phase.

Per-source history is bounded to 4096 controls and 4096 events/60 seconds. Large
authored histories increase project/snapshot cost; no performance acceptance or
unlimited recorder is claimed. Baking to an ordinary audio asset remains available
when long or many simultaneous takes are needed.

## Normal contact activity without invented impact events

New generator model contact_activity accepts existing domain_contact telemetry on
dem_contact or mixed_contact, with input_metric=power_w; explicit signal driving
also remains available for authored control/fixtures. It excites the same six-mode
resonator bank continuously with deterministic noise scaled by normalized activity.
wood/metal/stone/glass, size_m, decay_seconds and brightness use the existing authored
timbre controls. reference_energy_j is reserved for the discrete impact model and
rejects for contact_activity.

```python
rt.audio.create(name="Measured contact texture", loop=True, gain=0.15,
    generator={"model": "contact_activity", "resonator": "stone", "size_m": 0.08},
    reaction={"driver": "domain_contact", "system_id": 1, "domain_name": "Actual domain",
              "domain_channel": "dem_contact", "input_metric": "power_w",
              "input_min": 0, "input_max": 1, "automatic": True,
              "follow_position": True})
```

Use real IDs/domain names/calibration. Creation never enables telemetry/simulation.
The existing measured damping work/duration provides power; the existing weighted
center provides pose. Neither the solver, shader ABI nor publication format changes.
The result is aggregate activity sonification, not timed collision events or a
calibrated elastic/material radiation model. Individual contact identity, contact
timestamps/material pairs, measured splash/fracture events and exact solver cache
are still open. The explicit impact operation remains separate and rejects this
continuous model.

Oscillator radius is bounded on excitation; tiny state is zeroed. Callback memory
and work remain bounded and allocation/lock free. Inactivity uses a drain allowance
of decay_seconds+1.25 seconds for this live resonant texture, allowing its authored
modal decay before the current one-second acoustic history is released. Native
scheduled replay still obeys its explicit end frame and ordinary 1.25 s acoustic
drain. The shared DSP/schedule also renders this model through history/bake.

## Deferred combined verification

Existing reactive acceptance was extended but not run: contact texture PCM/control,
stored-history type/version atomicity, duplicate preservation, implicit stored bake
duration and sampled finish/store/bake. Later user-build acceptance additionally
checks project save/load/undo/redo, clear/incompatible source edits, authored versus
explicit history selection, actual measured DEM/mixed contacts, resonance listening,
drain/mute/gain edits, snapshot cost and callback/32-voice budgets. No new script or
live test round is introduced here.
