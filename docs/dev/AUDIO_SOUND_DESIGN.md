# Sound designer v2

2026-10-14: implemented in source; user confirmed the Studio build after the MSVC
JSON-comparison and DaisySP math-constant fixes. New external IPC, listening and UI
acceptance remain pending. Extends [the approved plan](AUDIO_SOUND_DESIGN_PLAN_2026_10_14.md).
Legacy synth_generate/preset contracts remain available.

## Workflow

Audio > Assets > Design sound opens Designer inside the existing Audio bottom
editor. An asset's Design / process button selects that imported sound. Timeline,
Mixer and Designer share one bottom Audio entry. Controls / Save opens the existing
Audio Inspector. No permanent shelf or separate application is introduced.

Render + listen builds transient PCM and auditions it; no WAV, asset or physics
telemetry is created. Listen original auditions the selected input before effects;
Listen processed auditions the designed result. Loop preview and Stop are independent
of timeline pause. Audition isolates the sound by stopping existing scene voices and
suspending automatic playback; Timeline Play or Resume automatic audio re-arms it.
The existing preview mute and Effects/Master bus gain/mute/solo also affect audition.
Changing controls previews on release when enabled. This first implementation renders
synchronously on the editor thread and restarts audition after each accepted edit;
uninterrupted realtime DSP/background jobs remain future work.

Waveform, peak/RMS, amplitude envelope, filter response and a sampled spectrum show
what was rendered. Attack/Sustain/Release handles also edit the synthetic envelope.
Pending controls are marked until successfully rendered. Spectrum is a 2048-frame
Hann-window analysis of the loudest of sixteen sampled windows, with 64 logarithmic
frequency probes; it is not a continuous analyzer or a full-resolution FFT display.
Waveform combines channel extrema; stereo channels remain separate in playback/PCM.

Save asset asks for a sound name. Default destination is the current project's
assets/audio/generated folder; unsaved scenes use the active asset-library
root/audio/generated (runtime library fallback). The folder is shown and can be
changed with a folder picker.
No absolute path typing is needed. Folders are created only when saving. Existing
names receive _2, _3, ... suffixes with exclusive publication. Explicit new WAV paths
remain supported by script/IPC. Invalid filenames/Windows reserved names are rejected.
Saving imports the actual preview PCM as float32 48 kHz WAV and records its recipe;
processed assets preserve original license/author and source provenance. Optional
timeline insertion reuses clip_create at the playhead (a separate undo action).
Undo removes authoring state; the authored WAV remains on disk. Existing project
packaging copies these WAV assets through its normal portable audio path.

## Shared operations

UI, rt.audio.<operation>(**kwargs) and audio.<operation> IPC use AudioAuthoring ->
AudioService -> AudioSoundDesign. Errors use existing audio_error / Python ValueError
semantics. Unknown top-level/recipe fields and invalid types/ranges are rejected.
Rendering failures preserve the previous design and audition. Save failures preserve
scene state; failed asset import removes only the newly authored WAV.

| Operation | Parameters | Capability / history |
|---|---|---|
| design_render | required recipe object below | SceneWrite + FilesRead; transient, no undo |
| design_preview | original=false, loop=false, strict booleans | SceneWrite; transient |
| design_status | none | Read; PCM summary, recipe, waveform, envelope, filter, spectrum, cursor |
| design_stop | none | SceneWrite; releases voice/data source, retains design |
| design_destination | optional absolute folder | Read + FilesRead; resolves folder, creates nothing |
| design_save | name="Sound", optional folder OR path | SceneWrite + FilesRead + FilesWrite; asset undo |
| workspace_update | designer_open boolean among existing workspace fields | SceneWrite; workspace only |

Render/save/preview are locked while a render output job owns authoring. Scene
replacement, undo/redo and service destruction clear transient designs/voices.
No preview PCM is inserted into timeline capture. Stop All also stops Designer.
At most one designer voice exists; it borrows owned PCM until sound uninitialization
completes, and its data source is destroyed before the PCM can be replaced.

## Recipe

Mode defaults to synth. Each request is a complete recipe, not a partial patch.

| Common field | Default / bounds |
|---|---|
| gain | synth 0.5 / asset 1; 0..2; samples limited to [-1,1] |
| highpass_hz / lowpass_hz | 20 / 20000; 20..20000; highpass strictly below lowpass |
| drive | 0; 0..1 soft saturation |
| fade_in_seconds / fade_out_seconds | 0.003 / 0.02; 0..10; multiplicative edge fades |
| pitch_semitones | 0; -24..24, duration-independent |
| stretch_ratio | 1; 0.5..2, pitch-independent; max 30 s result |
| loop_crossfade_seconds | 0; 0..1, strictly shorter than half the result |

Filter endpoints mean bypass. Loop crossfade joins tail/head and shortens the final
sound by the overlap. It does not implicitly enable preview/clip looping.
Very short sources can fail Signalsmith's pre-roll requirement when pitch/time
processing is requested; neutral pitch/time preserves those short sounds.

| Synth-only field | Default / bounds |
|---|---|
| preset | metal; tone, wind, metal, wood |
| duration_seconds | 2; 0.05..10 |
| seed | 1; integer uint32 |
| frequency_hz | wood 180, tone 440, wind 220, metal 330; 20..4000 |
| decay_seconds | wood 0.15, others 0.8; 0.01..10 |
| brightness | 0.5; 0..1; noise filter/modal damping |
| noise_mix | wind 1, others 0.15; 0..1 |
| attack_seconds / sustain / release_seconds | 0.003 / 0.7 / 0.1; 0.001..5 / 0..1 / 0.001..5 |
| wave | sine; sine, triangle, saw, square (tone/wind oscillator) |
| detune_cents | 0; -100..100 (second tone oscillator) |
| modulation_hz / modulation_depth | 0.7 / 0; 0.05..20 / 0..1 amplitude modulation |

Release gate uses at most half the input duration. ADSR is rendered with DaisySP;
metal/wood use authored four-mode damped resonators plus filtered noise. These are
sound design presets, not calibrated material acoustics. Synthetic input is peak
normalized before common effects/gain; original audition is therefore unity peak.
Some oscillator controls do not apply to modal impact presets.

| Asset-only field | Contract |
|---|---|
| asset_id | required existing audio asset |
| source_in_seconds | default 0 |
| source_out_seconds | default min(asset duration, start+10) |

Selected range must exist, be >=0.05 s and <=30 s. Existing decoder bounds remain
600 s / 128 MiB for the full imported file; range decode currently reads that entire
file before extracting the range. Stereo is preserved. No source file is overwritten.
Same recipe/input reproduces PCM within this implementation/platform; dependency
versions are pinned and Signalsmith uses a fixed local seed. Cross-platform bitwise
identity is not promised. FNV-1a PCM hashes are verification fingerprints.

## Dependencies and delivery

- DaisySP selected MIT oscillator, ADSR, one-pole, DSP utility modules:
  2c72eaf9eac5fc0dca1919d65d606da907832618. No DaisySP-LGPL modules.
- Signalsmith Stretch: a670068d9aeb64913331d5cc29337b19a457a7df.
- Signalsmith Linear: 146b26f91b48835d665fa0498d64d9617818d27a.

Original notices and revision files are included in external/ and copied to
assets/licenses/audio_dsp by both project packaging paths. Miniaudio remains the
device/decoder; ImGui remains the UI. VS sources and CMake discovery are wired.

Build Studio only for this feature. No new shaders. Run externally on that build:

```powershell
python scripts/test/rt_test_audio_sound_design_ipc.py --output-dir E:\YOUR_EXISTING_TEST_FOLDER
```

The probe does not play audio unless --audition is explicitly passed. It creates
only its own temporary assets/files, validates PCM parity/determinism, transformed
duration, preserved origin metadata, strict errors and collision/overwrite behavior.
Transient design is cleared by scene reload/undo; the probe leaves its final design
available for inspection and removes its test assets/files.

Manual: preview without choosing a folder, adjust controls/graph handles, compare
original/processed, loop/stop while timeline is paused; check narrow/high-DPI layout;
save by name twice, inspect actual destinations; insert a clip, undo and save/reopen.
Listening and build acceptance have not been claimed by source/static checks.
