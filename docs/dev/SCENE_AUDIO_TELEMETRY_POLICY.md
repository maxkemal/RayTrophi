# Manual physics audio telemetry policy

2026-10-14: source implementation; user Studio/shader build and live acceptance pending.

Physics audio telemetry is OFF by default for new scenes and legacy scenes without
this policy. A scene audio binding, capture, opening Audio, querying audio.domains,
waveform generation or playback does not enable it. Ordinary assets, sources,
thunder, waveform, mixer, clips and generated WAV assets work independently.

Enable manually in Audio > Scene > Collect physics audio telemetry, or the same
scene-wide checkbox in a source's domain-contact reaction inspector. This first
policy is scene-wide and affects all domains. It is not a per-domain/per-channel
budget or a claim that the enabled path is scalable to millions of particles.

UI, `rt.audio.telemetry_get()/telemetry_update(enabled=True|False)` and IPC
`audio.telemetry_get/telemetry_update` use the same AudioService policy. Only a
boolean enabled is accepted; unknown fields/nonboolean values reject atomically.
An omitted enabled is a no-op. Saved audio JSON stores `telemetry: {enabled:false}`.
Undo/redo restores the policy and synchronizes transient domain solver gates.
Malformed stored policy is rejected before replacing the audio scene.

## Off-path behavior

- MatterAudioFrame clears published audio stats and exits before any particle scan
  or MPM previous-damage ID map. Liquid/MPM publishing and gas CPU/GPU measurement
  return immediately. No gas audio reduction, scratch allocation, synchronization
  or 64-byte audio-result download is submitted.
- DEM diagnostics have only the required 17-word solver header (68 bytes).
  The optional 12-float/tane audio tail is not allocated/uploaded/downloaded or
  CPU-reduced. Shader work recording/energy calculations are gated by the bound
  diagnostics layout; physical forces, contact history and sleep remain active.
- Policy changes resize only the DEM diagnostics buffer at the next valid solver
  step, preserving other particle buffers/contact history. Paused domains keep an
  already-allocated tail until they next step or the runtime is destroyed. Old+new
  allocation peak participates in the existing resource budget.
- Domain reports return telemetry_enabled=false, measured=false, zero published
  audio quantities and reason=telemetry_disabled. This is unavailable data, not a
  measured physical zero. Previously recorded RAM frames cannot bypass this gate.
- Existing solver diagnostics and simulation-required physics transfers still run.
  Mixed-contact metrics already needed by the solver remain; this policy suppresses
  their audio publication rather than removing physical collision diagnostics.

The saved policy is authoritative; the MatterGrainParams transient flag is neither
an authored grain/material property nor separately serialized. Audio scene sync
propagates it to current/new domains. No other agent's simulation transport is replaced.
Enabling still uses the current O(N) DEM tail, liquid/MPM scans, MPM ID map and gas
reduction. GPU aggregation and selective domain/channel requests are future work.

## Build and user acceptance

Build Studio and the simulation shaders including all grain variants sharing
sim_matter_grain.glsl. Host/shader revision is now 27; an old shader fails revision
validation. Codex did not compile shaders, build Studio or launch the application.

Run externally on the new build:

```powershell
python scripts/test/rt_test_audio_telemetry_ipc.py
```

With existing domains, this probe does not enable collection unless explicitly
passed `--exercise-enable`; use that option only on a small acceptance fixture.
It checks strict invalid input rejection, policy/state parity and disabled report
semantics, and restores the prior policy.

Manual checks: new/legacy scene default off; Save/Reopen and undo/redo preserve the
explicit setting; connect a domain source while off and confirm no implicit enable;
small DEM/liquid/gas/MPM fixture off/on/off, live and forward RAM replay; compare
reported simulation upload/download bytes and frame timings (grain diagnostics
expose audio_readback_bytes and diagnostics_readback_bytes separately); confirm disabling only
resizes diagnostics and does not reset DEM contacts/sleep. Existing channel probe
with --require-dem now requires explicitly enabling telemetry and publishing a frame.
Inspect shader/storage buffers: off is 68 diagnostic bytes; enabled adds 48*N bytes
per published DEM frame. No performance or audible acceptance has yet been claimed.

## Next architecture

1. Explicit domain/channel selection; collect only requested measurements.
2. GPU block reductions with bounded domain/spatial-region results; reuse solver
   work and avoid CPU per-particle ID maps for sound-only needs.
3. Separate bounded active sound regions/voices from carrier count; smooth controls
   at a capped update frequency while integrating impact statistics correctly.
4. Optional geometry acoustics: camera listener and active emitters, flat
   TriangleMesh/DNA SoA/BVH visibility queries, capped ray/source/time budgets,
   absorption/transmission/filtering, then early reflections and room reverberation.
   Optical materials must not silently stand in for acoustic coefficients.
5. The same policies and evaluators must serve UI/script/IPC, live playback,
   captured recipes and deterministic offline output.

These later stages are not implemented by the manual telemetry switch.
