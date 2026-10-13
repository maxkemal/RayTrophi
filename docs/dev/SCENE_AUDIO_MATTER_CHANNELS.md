# Matter audio channels and Vulkan DEM telemetry

2026-10-13: source implementation; user build and live/cache acceptance pending.

A Matter domain can contain fluid, gas, MPM and DEM simultaneously. A domain ID
does not identify a sound behavior. Each source now selects `reaction.domain_channel`
through the same validated audio service in UI, Python and IPC:

| Channel | Producer / current status |
|---|---|
| `mixed_contact` | Existing CPU mixed-contact energy; legacy default |
| `dem_contact` | Vulkan DEM normal-contact damping work, full committed frame |
| `fluid_flow` | Reserved, unavailable until a fluid flow/splash producer exists |
| `gas_flow` | Reserved, unavailable until a gas flow producer exists |
| `mpm_deformation` | Reserved, unavailable until an MPM deformation/fracture producer exists |

`audio.domains` / `rt.audio.domains()` returns one row per domain **and channel**.
Consumers must select both `domain_name` and `domain_channel`; five rows do not mean
five domains. Several ordinary loop sources can independently bind the same domain.
An unavailable channel never substitutes another channel's measurement. The legacy
mixed channel retains its signal key; new channels have independent keys under the
reserved `@domain/<channel>/<system_id>/<domain_name>` namespace.
Unknown channels fail validation before mutation. Existing scenes default to
`mixed_contact`; new channel configuration uses existing save/load/undo snapshots.
Material-pair filters remain rejected; these measurements do not classify substances.
Reserved channels can be authored but remain visibly unavailable and inactive.

## DEM measurement scope

For each normal contact, the shader records
`max(0, (elastic_normal_force - applied_normal_force) * normal_relative_speed) * dt`.
This is normal dashpot work **after** the unilateral force clamp. It is mechanical
dissipation telemetry, not acoustic power or a full kinetic-energy budget. Grain
pairs contribute half from each endpoint; static/moving wall and mesh contacts
contribute once per grain. All solver substeps accumulate, including shared-clock
steps. Elastic spring work, tangential/rolling plasticity, twist, wet bridge energy,
liquid drag and MPM/DEM coupling impulses are excluded. Restitution=1 can therefore
produce a valid zero report even when friction remains active.

`contact_energy_j`, duration and `contact_power_w = energy / duration` drive the
existing `energy_j` / `power_w` evaluator. Positive energy gives an energy-weighted
world contact centroid; zero energy uses the domain center. `contact_pairs` for DEM
retains the last-substep contact count (grain pairs are counted from both ends),
not the number of impacts accumulated across the frame.

Four float-bit words per grain are appended to the existing diagnostics buffer.
Each invocation owns its row, so no float atomics or fixed-point scale is needed.
The existing upload/readback batches transfer this tail: **16 extra bytes per grain
each direction per frame**, 16 bytes per capacity slot on device, plus transient
host storage and a host reduction. No additional submission, fence or audio-side
particle scan is introduced. GPU accumulation is float32; host reduction is double.
Nonfinite, negative or out-of-range telemetry stays unavailable without changing
forces or rejecting otherwise valid physics. Physics/revision/readback failures
still reject frame publication through the existing path. Shader revision is **25**;
old C++/SPIR-V combinations cannot publish a matching frame.

## RAM cache and transport

The measurement lives in `fluid_stats.grain_report`. RAM compression moves the
complete state to `CachedGridDomain.meta`; decompression copies it back. No disk
bake format change is made. Older disk frames do not gain measurements.
Cache translation rebases the measured DEM centroid with the particles and leaves
energy/duration unchanged. Existing bridge time+version rules apply separately to
each channel: forward RAM frames publish without new solver substeps; same-frame
redraw/re-restore does not refresh TTL; backward seek/load/undo invalidates signals.
Pause/Stop, automatic thresholds, voice budgets and mixer routing remain shared.

```python
reaction = dict(rt.audio.get(id=dem_source)["reaction"])
reaction.update(driver="domain_contact", system_id=system_id, domain_name=domain_name,
                domain_channel="dem_contact", input_metric="power_w",
                input_min=0, input_max=10, automatic=True, follow_position=True)
rt.audio.update(id=dem_source, loop=True, reaction=reaction)
```

IPC uses `audio.update` with the identical reaction object. Reactive Audio exposes
Physics input alongside Domain binding and Contact magnitude. Use a mono loop for
positional sound and tune input bounds in watts to the scene's measured values.

## User build and acceptance

1. Recompile simulation shaders with `source/shaders/compile_sim_shaders.bat`, then
   perform the normal Studio build. The shared grain include affects list/hash/step
   shaders; ensure runtime SPIR-V matches revision 25. The agent performs no builds.
2. Simulate a small dry Sand pour onto a closed floor. Bind a mono loop to
   `dem_contact`, automatic=true, power_w and gain_min=0. Check measured positive
   duration, nonnegative energy, power=energy/duration and a centroid near contacts.
   Check Effects/Master PCM and listen. A settled, motionless pile should approach zero.
3. Run from a separate terminal while Studio is open:
   `python scripts/test/rt_test_audio_domain_channels_ipc.py --domain NAME --require-dem`.
   This read-only probe checks channel completeness/isolation and measured semantics;
   it does not establish audible acceptance or control simulation.
4. Populate RAM frames, rewind and Play again. Run the same probe during forward
   replay and compare power/position for matching frames with the live pass. Check
   source reaction input and PCM on both passes; same-frame waiting must expire
   the signal rather than indefinitely repeat contact activity.
5. In a domain with DEM, MPM and fluid/gas owners, bind separate sources to the
   different channels. DEM activity must not activate the reserved flow/deformation
   sources. Keep `mixed_contact` separate: it describes the CPU mixed interaction,
   not an individual owner's sound. Independent channel producers remain future work.
6. Test Pause/Stop, backward seek/loop, disabled/deleted domains, scene reload and
   undo/redo of channel edits. Translate a cached domain and verify its centroid
   translates while energy/power stay unchanged. Reject an unknown channel atomically.

Static/non-build checks and numeric contract checks do not execute C++ or shaders.
Build, audible playback, resource cost and live RAM replay acceptance remain pending.

## Short live IPC acceptance — 2026-10-13

On the user's new build and initially empty scene, external named-pipe IPC created
`DEM_Audio_Replay`: a closed Vulkan Matter domain with Sand DEM, radius 0.025 m,
stiffness_scale=0.1, sleep=false and a capped 120-grain pour. The bundled CC0
`sand_hiss_mono.wav` was bound to `dem_contact` on an automatic loop.

Thirty-five direct solver steps all reported positive energy, positive duration and
valid energy-weighted position; peak sampled power was 40.0557 W. Channel-contract
checks passed; fluid/gas/MPM reserved inputs remained unmeasured. The device became
ready and an automatic voice advanced without automatic errors.

The scene accumulated 56 RAM frames (0..55). A 24-frame forward sample and its
subsequent rewind/forward replay matched energy, power, duration and position on
all 24 frames. During replay in continuous-preview mode, source reaction input
matched DEM power and followed the reported position; Effects/Master PCM was
nonzero. For example frame 10 reported 0.964240 J over 0.0416667 s = 23.1418 W,
with Effects RMS approximately 0.01239. Repeated same-frame restores left signals
empty and voices released. The initial signal sample for that last check was
already empty, so it does **not** prove the transition from an active signal to
expiry. No exact substeps/cache-hit counter was available in this probe.

The fixture remains in the scene, unsaved, at frame zero with timeline audio,
automation re-armed and the audio source selected. Timeline Play/Pause/Stop,
audible quality, save/reopen, translated-domain replay and mixed-owner acceptance
are still pending. This test does not establish any flow/deformation sound producer.
Evidence: `dem_audio_replay_acceptance_20261013.json` contains per-frame reports and
the final runtime state. No build or Studio launch was performed by the agent.
