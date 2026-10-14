# Audio sound design and preview

2026-10-14 — approved direction; first source delivery implemented. Builds and listening
acceptance belong to the user. This document extends the scene audio runtime plan.

## Product direction

Use the existing Audio bottom editor and contextual inspector. Keep the viewport
primary; do not add a permanent shelf. Synthetic and imported sounds share one
sound design workflow: choose, shape, preview, compare, then create an asset.

- Preview in memory before writing a file or adding scene assets. Play/Stop/Loop,
  waveform, level readout, envelope and filter feedback are first priorities.
- Synthetic sound families: tone, impact/resonance, noise/flow. Expose meaningful
  controls and an advanced section for oscillator, envelope, noise and filtering.
- Imported mono/stereo assets: range selection, fades, filters, original/processed
  audition; later add independent pitch/time processing and loop crossfades.
- Reuse miniaudio for playback. Evaluate pinned MIT DaisySP modules for synthesis
  and filtering; Signalsmith Stretch for later independent pitch/time processing.
  Record exact dependency revisions and notices. Do not silently include the
  separate DaisySP-LGPL library. JUCE is an alternative, not the selected framework.

## Saving without typing paths

The normal flow asks for a sound name, not an absolute WAV path. Resolve the folder
from the current project location, falling back to the active asset-library root
for unsaved scenes. Show the destination and allow a folder picker. Use a generated
audio subfolder, safe filenames and automatic numeric suffixes for name collisions.
Do not overwrite an existing sound. Preview needs no output folder or file.

Create/import the WAV only on Save asset. Optional timeline insertion uses the
existing playhead operation. Preserve recipe metadata and existing portable project
packaging. Undo removes the asset/clip, leaving the authored file as documented.
Explicit paths remain available for automation and advanced users.

## Delivery order and shared contracts

1. Shared destination resolver and safe name-based save; in-memory synthesis and
   imported-asset processing; bounded preview voice; visual designer and controls.
2. Richer authored presets and listening refinements, interactive envelope/filter
   curves, spectral analysis and live parameter updates without transport interruption.
3. Independent pitch/time processing, loop crossfade and reusable effect recipes.
4. Physics-driven synthesis using bounded domain/region summaries, then geometry
   acoustics. These remain separate from asset synthesis.

UI, rt.audio and audio.* IPC must call the same AudioService/core implementation.
Validate unknown keys, types, finite bounds and output targets before mutation.
Preview must not mutate assets, enable physics telemetry or participate in timeline
capture. Stop, scene replacement and service destruction must release preview PCM
after the native voice stops. Offline use of saved assets must match preview PCM.

## Acceptance

Static checks: IPC capability/schema parity, Python probe syntax, dependency notices
and project source registration. No builds or Studio launch by Codex.
User build: Studio only for this feature, no additional shaders.
External IPC probe: deterministic generation, no-file preview, asset processing,
invalid requests preserving state, safe filenames/collisions and PCM/save parity.
Manual: hear control changes, original/processed comparison, loop/stop, narrow/high-DPI
layout, Save asset without path entry, timeline insertion, undo and save/reopen.

## Source delivery, 2026-10-14

[Sound designer v2](AUDIO_SOUND_DESIGN.md) implements the first workflow plus
independent pitch/time and loop crossfade: pinned DaisySP/Signalsmith dependencies,
in-memory audition, imported-asset range processing, waveform, draggable envelope,
filter response, sampled spectrum and name-based saving. Unsaved scenes reuse the
active asset-library folder, with runtime-library fallback; saved scenes use their
project folder. UI, Python and IPC share six design service operations.

Static capability audit, binding/source registration checks, enabled dependency
include closure, notice parity, new-source delimiter/whitespace checks and Python
probe syntax passed. These checks do not confirm C++ compilation or audible quality.
Studio build and external probe/manual acceptance remain pending. Further authored
presets, uninterrupted live DSP/background jobs and physics-driven synthesis remain
future stages; they are not marked complete by this asset designer delivery.

## Build checkpoint, 2026-10-14

User confirmed successful Studio compilation after C7692 and M_E compatibility
fixes. New Designer listening/UI/external IPC results have not yet been supplied.
The source/build gate is recorded separately from audible and interaction acceptance.
