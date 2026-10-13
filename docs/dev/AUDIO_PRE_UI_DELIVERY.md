# SAS before UI: source delivery

Scope: independent DEM normal/sliding/rolling channels; liquid and physical gas
kinetic transport activity; MPM stress power and damage growth; waveform; persistent
timeline clips; scene-clock capture; offline WAV. UI layout is deferred.

All operations share AudioService through scripting rt.audio and IPC audio methods.
waveform(id, points=512) returns min/max/RMS bins with exact source-frame coverage.
clip_list/create/update/remove/duplicate/trim/split support source range, phase,
looping, gain/pitch/pan curves and fades. Trim/split preserve source travel and
envelopes. Validation uses a candidate scene to keep rejected edits atomic.

capture_start(name, max_seconds=120), capture_status/finish/cancel require timeline
mode. Start capture, play the scene, pause and finish. Played sources and thunder
become clips. Capture samples controls at 30 Hz plus lifecycle changes; existing
clips are excluded. Backwards seek fails capture. Limits: 512 tracks, 4096 knots,
600 seconds. Capture bakes distance/cone/balance, not HRTF or speaker PCM.

export_start(path, start_frame, end_frame_exclusive, fps_num=24, fps_den=1,
overwrite=false), export_status/cancel snapshot clips/mixer and write stereo
float32 48 kHz WAV in blocks on editor ticks. Half-open rational frame boundaries,
atomic publication and temporary-file cleanup preserve existing targets on cancel.
Preview mute is independent. Reports include peak/RMS/overload; samples are not
clipped. Export evaluates authored/captured clips; it does not resimulate physics
or generate new lightning. Save the scene to preserve clips.

DEM revision 26 records three energy/centroid channels, 12 float words per grain.
These are mechanical diagnostics, not calibrated acoustic Joules. Fluid/gas report
sum(0.5*m*speed^3/voxel) in W; deformation reports positive end-state stress power.
These samples have energy_measured=false, not invented step-integrated Joules.
Fracture is mass-weighted damage growth in 1/s: select input_metric=value.
Flow/deformation select power_w; DEM selects energy_j or power_w. RAM metadata
retains measurements/centroids; restored positions follow domain translation.
GPU gas reduces physical inventory with FP64 and downloads 64 bytes; unavailable
device metrics do not substitute stale host data.

## User build and acceptance

Codex did not build or launch Studio. Build Studio and simulation shaders including
sim_audio_gas_flow.comp and DEM revision 26. Progressive lightning changes also
require cloud shaders. Static shader mirror layout audit passed.

Run the external scripts/test/rt_test_audio_domain_channels_ipc.py during live
DEM/liquid/gas/MPM and forward RAM replay. Check independent channels and unavailable
versus measured-zero semantics. Check rt.audio and IPC expose the same operations.

Check waveform bins, clip trim/split and pitch curves; pause/resume/backward seek;
Run python scripts/test/rt_test_audio_clips_ipc.py --asset-id EXISTING_ID
--output NEW_ABSOLUTE_WAV_PATH from an external terminal for automatic acceptance.
save/reload and invalid asset relink rejection. Capture weather, physics and thunder
during forward playback, finish paused, export its frame range and listen.
Check invalid mutation leaves the scene unchanged. Export one second at 24 fps:
WAV should be 56 + 48000*8 bytes with 48000 stereo float frames. Cancel long export:
no completed output or temporary file; an existing target survives cancellation.
New C++/shader execution and audible acceptance remain pending the user build.
