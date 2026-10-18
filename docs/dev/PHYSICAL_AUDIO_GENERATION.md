# Continuous physical activity sound generation

2026-10-18 source checkpoint. No C++ build, Studio launch or live test performed.
UI reorganization is deferred until the physical audio core is delivered, per
the user's correction. This is the first runtime procedural source slice, not
completion of physical sound, splash/contact events, replay or export parity.

## What produces PCM

PhysicalAudioSource.cpp implements a mono 48 kHz miniaudio data source. It generates
PCM continuously without an imported recording, disk file or pre-rendered loop.
The existing emitter/reaction service publishes normalized activity through one
lock-free control value. The callback uses seeded noise and bounded filters:

| model | domain_channel | Approximate texture |
|---|---|---|
| liquid_flow | fluid_flow | Band-limited, amplitude-modulated flow noise |
| gas_flow | gas_flow | Low-pass airflow noise |
| friction | dem_friction | Broad friction/scraping noise |
| rolling | dem_rolling | Low-frequency rolling rumble |
| contact_activity | dem_contact / mixed_contact | Continuous resonant contact texture |
| deformation | mpm_deformation | Smoothed noise exciting an authored resonator |
| fracture_activity | mpm_fracture | Seeded synthetic crackle exciting an authored resonator |

These models sonify measured activity; they do not solve acoustic radiation,
material resonances, fluid bubbles or individual impacts. Liquid transport activity
has W dimensions but is not acoustic watts. DEM friction/rolling damping telemetry
likewise controls authored sound, not calibrated SPL. Aggregate data supplies no
material-pair classification. No impact or splash is inferred from a scalar delta.

## Shared API, persistence and controls

Existing create/update/get/list/duplicate, transport, source_runtime and acoustics
methods support the optional emitter `generator` field through the same service
in Python and IPC. UI forms are intentionally deferred; no UI-local generator
model was added. Existing generic source selection, transform, transport and
acoustic controls operate on the source. Bind using existing authoring methods:

```python
# Replace system/domain and input_max with an actual measured domain and calibration.
# This does not enable telemetry or simulation.
source = rt.audio.create(
    name="Faucet flow", generator={"version": 1, "model": "liquid_flow",
                                  "seed": 1, "brightness": 0.5},
    loop=True, spatial=True, gain=0.15,
    reaction={"driver": "domain_contact", "system_id": 1, "domain_name": "Faucet",
              "domain_channel": "fluid_flow", "input_metric": "power_w",
              "input_min": 0, "input_max": 10, "response_seconds": 0.08,
              "automatic": True, "follow_position": True})
rt.audio.acoustics_preview(id=source["id"], enabled=True, spectral=True,
                          reflections=True, diffuse=True)
```

External IPC uses identical `audio.create`/`audio.update` JSON parameters. Domain
existence is checked by AudioAuthoring. `signal` driver is also supported for
explicit authored control and deterministic control fixtures, not evidence of
solver measurement. Manual/weather drivers are rejected for these models.

Recipe fields: version integer 1, model from the table, uint32 seed (including
zero), finite brightness 0..1. Defaults: version=1, liquid_flow, seed=1,
brightness=0.5. Unknown fields/versions/models and invalid ranges reject before
scene mutation. Generator sources require loop=true and empty asset_id; domain
drivers must match the model's channel and use power_w, except fracture_activity,
which requires input_metric=value. Existing scenes normalize
missing generator to null. Setting generator=null returns to asset semantics;
provide a compatible asset/reaction in the same update when needed. Recipe edits
release the existing voice and reset its DSP on next play; no live timbre morph
is claimed. Recipes/seed persist through existing project/undo/duplicate paths.
`source_runtime.input_kind` identifies procedural versus asset input.

Activity is clamped to 0..1 and smoothed per sample with a 20 ms time constant.
It scales excitation; existing reaction gain/pitch additionally retain their
authored mapping. This combined response is artistic and not a physical power
conversion. The callback allocates nothing, locks nothing, performs no JSON,
geometry, file or telemetry work, and publishes only an atomic cursor per block.
PRNG/filter state belongs to the callback. Format is fixed mono float32/48 kHz;
the existing sound engine owns pitch, spatialization and mixing. Per-source state
is bounded; generators share the existing 32-resident-voice budget and existing
acoustic node/history cost. Asset decode accounting skips file-free generators.

Automatic control follows existing thresholds, signal expiry, Stop All suspension,
explicit resume and pose following. Native sounds are uninitialized before their
data sources are released on stop/remove/update/scene change/rollback. Asset and
procedural voices can play concurrently. The same spectral/specular/diffuse node
processes generated PCM; no second propagation model is introduced.

## Explicit remaining boundaries

### MPM activity continuation

The deformation driver consumes the existing end-state positive stress-power
estimate in W. It is not an integrated elastic-work history. The fracture driver
consumes the existing mass-weighted damage-growth rate in 1/s through `value`;
`input_min`/`input_max` must therefore be authored in 1/s, not joules or watts.
Both models accept the same wood/metal/stone/glass `resonator`, `size_m` and
`decay_seconds` controls as contact_activity; `reference_energy_j` remains
exclusive to explicit impact. Signal controls are also allowed.

Fracture crackle is a seeded texture controlled by normalized activity, not a
sequence of measured fracture events. No event identities, substep timestamps,
fracture energy or material classification are reconstructed. These models use
the shared DSP for native playback, stored history replay and dry bake. Their
resonant tails receive the same bounded retirement allowance as contact_activity.
The existing IPC acceptance fixture includes both models with injected signals;
actual MPM telemetry routing and audible output still require the user's build
and consolidated live acceptance pass. No solver code was changed for this slice.

Subsequent source continuation: [stored history and normal-contact texture](PHYSICAL_AUDIO_STORED_HISTORY.md)
adds source-owned validated histories to project/undo/duplicate, implicit bake/replay
inputs and contact_activity sonification from existing measured damping power.
This does not create individual solver collision events or calibrated radiation.

2026-10-18 dry-output continuation: [shared DSP, sampled takes and deterministic
bake](PHYSICAL_AUDIO_BAKE.md) now connect explicit/sample histories to ordinary
assets through a bounded worker. This supersedes the missing dry procedural bake
path below, not acoustic parity, substep solver history or exact live PCM capture.

### Bounded retirement continuation

2026-10-18 subsequent source update, isolated from unified-domain solver/testing:
automatic continuous generators now drain instead of immediately releasing their
native voice at the inactivity threshold. Input excitation fades to zero using
the existing sample smoother; the last reaction gain is held while authored
gain, bus gain and mute remain live. Fresh activity cancels retirement. A 1.25 s
transport-delta allowance covers the current <=1 s acoustic history and filter
decay. Delta is bounded by AudioTransport; stalled UI updates can extend wall time.
Explicit stop/remove/scene reset and invalid managed poses retain immediate stop.

Impact sources publish a lock-free block peak. After an accepted event, an empty
queue and peak below 1e-5 begin the same 1.25 s allowance. New accepted excitation
cancels it. Retirement only advances for native-playing, nonsuspended sources
while transport is unpaused. Timing depends on consumed audio and authored pitch;
this is not a calibrated perceptual-silence criterion. Timeline holds/explicit
pause retain the source state; full acoustic-node pause parity remains unverified.

Automatic retirement destroys sound/DSP state but preserves the last accepted
impact sequence in a main-thread per-source map. Repeating that sequence after
retirement returns accepted=false without allocating a voice or resuming playback.
A higher sequence creates a new voice. Explicit stop, recipe replacement, remove,
scene/seek reset and Stop All clear the dedup epoch. Preparation rollback preserves
the sequence record. The retained map stores one identity per authored impact
source, not an unbounded event history. This supersedes the earlier resident-only
dedup and indefinitely-resident impact descriptions below.

source_runtime.generator_runtime reports resident/draining/not_resident,
drain_seconds, queued_impacts, block_peak and held_reaction_gain. During drain the
existing effective_gain is the target control value; held_reaction_gain explains
the maintained multiplier. These diagnostics share the existing UI/script/IPC
service; no UI authoring or unified-domain changes were added. Existing reactive
acceptance now includes automatic impact release and duplicate rejection after
release, but was not run. Build and combined listening/lifecycle acceptance remain
pending. Future multibounce/late-reverb support must revisit the fixed allowance.

### Explicit impact excitation source checkpoint

The subsequent source slice adds model=impact with a six-mode damped resonator.
Create it with loop=true and the default manual reaction. The data source stays
resident and silent until excited; stop/remove releases it. It is a bounded
resonator node, not a new voice per impact. Recipe controls are:

- resonator: wood (default), metal, stone or glass; authored timbre profiles,
  not measured Substance coefficients or automatic material classification.
- size_m: 0.005..10, default 0.1; inverse-size modal-frequency heuristic,
  clamped to 30..10000 Hz. No shape or elastic eigenmode solution is claimed.
- decay_seconds: 0.005..10, default 0.4; base-mode time to -60 dB amplitude.
  Higher modes damp faster according to brightness.
- reference_energy_j: 1e-9..1e9, default 1; authored excitation reference.
  Event amplitude is sqrt(min(energy/reference, 1)), not radiated acoustic power.

All controls are finite and validated by the same source create/update service.
The new shared operation is available as rt.audio.physical_impact in Python and
audio.physical_impact over IPC:

```python
source = rt.audio.create(name="Metal impact", loop=True, gain=0.15,
    generator={"model": "impact", "resonator": "metal", "size_m": 0.1,
               "decay_seconds": 0.8})
rt.audio.physical_impact(id=source["id"], sequence=1, energy_j=0.05)
```

sequence is required, an integer in 0..2^53-1, strictly increasing per resident
source. energy_j is required, finite and in (0,1e9]. Wrong source kind, unavailable
pose, invalid sequence/energy and unknown keys reject. Duplicate/out-of-order
events return accepted=false without resuming a paused source. Validation occurs
before preparing a voice. Accepted events start/resume the existing source using
shared transport; a paused timeline holds consumption. Permission is SceneWrite;
this operation never decodes a file. A failed native start is not transactional:
the excitation may remain queued; retry playback explicitly rather than assuming
the same sequence will be reaccepted.

The single-main-thread-producer/single-callback-consumer ring holds 32 excitations.
Full queues reject without advancing the accepted sequence. The callback consumes
at most the 32 entries present at a block boundary; zero-length reads consume none.
Block arrivals are coalesced, summed excitation is capped at 1, and oscillator
state radius is bounded on excitation. This intentionally limits repeated-impact
amplitude and does not conserve physical energy. Modal computation uses fixed
arrays and no allocation/locking/scene access. Small residual state is zeroed to
bound denormals. Sound gain/pitch, spatialization and acoustic processing are shared.

The supplied sequence/energy is an explicit authoring or future solver-event
input. This slice does NOT connect aggregate DEM damping work to invented contact
events. Per-contact identity, solver timestamp, measured energy provenance, world
position/material pairs, splash/fracture event generation and automatic event
publication remain open. Events are applied at the next consumed audio block,
not scheduled at a scene sample timestamp. Stop, source recipe replacement,
scene/seek reset and removal discard queue/dedup state and start a new epoch.
Caller-owned sequences alone do not provide replay/history correctness.

The existing reactive-acoustics acceptance script additionally covers the four
resonator profiles, nonzero native PCM, duplicate/invalid rejection, sequence
preservation and paused-source retry behavior. It was extended but not run.
Listening realism, concurrent-event saturation, queue overflow, deterministic
history/export and automatic tail release remain later consolidated acceptance.

Generator time advances while its data source is consumed. Pause preserves DSP;
stop or timeline discontinuity discards it. Same seed and constant control produce
the same sequence, but UI-sampled varying controls are not sample-exact simulation
history. Rewind/cache replay reproduction, bounded excitation/echo tail drain,
phase-aware sources, calibrated material response and event scheduling remain open.
Automatic threshold stop can still truncate tails.

Existing capture builds clips from asset-backed control envelopes. capture_start
rejects scenes containing procedural sources; introducing one during capture also
rejects. This prevents a missing-asset failure or silently incomplete capture.
Offline clip export remains asset-only and does not include live procedural
sources or live acoustic filtering. Shared DSP/control-history capture/export is
the next core integration, not an accepted capability of this slice.

## Consolidated acceptance

The existing rt_test_reactive_acoustics_ipc.py was extended, without a new script
or live run, to cover file-free PCM for all four models, recipe rejection without
mutation, automatic stop/resume, no asset creation and no telemetry auto-enable.
It uses injected signals, not real solver measurements. At the next combined
gate, the user builds Studio C++ (no shader change), then runs the existing
external probe in its required disposable empty scene. Also verify real measured
fluid/gas/DEM activity, save/load/undo/duplicate, recipe/asset switching, mixed
asset/procedural voice budgets, capture rejection, pause/seek and device teardown.
Listen and measure callback/voice cost separately; syntax/static checks do not
establish water/friction realism or performance acceptance.
