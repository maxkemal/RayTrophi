# Audio workspace UI

Status: implemented in source; user build and visual acceptance pending.

Entry points:
- Audio waveform icon in the existing main context rail.
- Audio / Mixer in the bottom editor bar, sharing Timeline's transport.
- Scene hierarchy sources and Asset Browser's Add timeline clip action.
- Audio > Assets can create timeline clips or scene sources.

Main Audio context has Sources, Assets, Scene and Output sections. Selection
can be edited in an optional movable/resizable/closable inspector: source details/reactive binding
or clip range/phase/gain/pitch/pan/fades/automation. Diagnostics are collapsed.
The Audio context occupies tab 15; existing tabs retain their identity.

Timeline uses shared scene FPS, range, playhead, zoom/pan and Play/Pause.
Bus rows pack overlapping clips into additional lanes without changing clip data.
The timeline's Add clip selector lists existing scene assets and source bindings.
Source binding insertion copies the sound and base gain/pitch/bus, not its reactive
behavior. Capture is the path for recording actual weather/physics/thunder reactions.
Clip body drag moves; 7px edge regions trim inward; release performs one canonical
mutation/undo record. Split at playhead, duplicate/delete and frame snap are exposed.
Ctrl+wheel zooms; wheel pans; ruler scrubs through the canonical timeline setter.
Waveforms are bounded min/max envelopes, at most one new asset decode per UI frame.
Refresh waveforms explicitly refreshes the UI envelope cache.

Mixer has dB faders (-60=off, max +12.04), numeric gain, balance knobs, mute/solo,
post-fader peak/RMS/hold meters and overload indicators. Drag applies live gain/pan
and records one undo entry when released. Stereo balance is canonical mixer pan
(-1..1); legacy scenes default to zero. Native mixer and offline renderer apply
child and Master balance once each. Direct Master clips receive Master once.

Output section exposes capture start/finish/cancel and offline export using scene
range, explicit path and overwrite option. Active capture/export indicators remain
in the bottom bar. Capture produces clips, not speaker PCM. Offline output uses
authored/captured clips; it does not resimulate source physics.

Script/IPC parity:
- editor.set_bottom_editor(name="audio"|"audio_mixer"); editor.get_state.
- audio.workspace_get / workspace_update(selected_clip, snap, waveforms, inspector_open).
  Invalid selected IDs, unknown parameters and nonboolean toggles are rejected
  atomically. Selection is ephemeral; clips/mixer settings persist with the scene.
- audio.mixer_update now accepts pan; same validator, native and offline routing.
- Existing clip/waveform/capture/export operations perform all authoring.

## User build and checks

Build Studio only for these UI/knob/pan changes; no additional shader was introduced.
First compile includes vendor imgui-knobs.cpp and focused AudioWorkspaceState,
AudioWorkspaceUI, AudioTimelineUI and AudioMixerStripUI modules.

Run external scripts/test/rt_test_audio_workspace_ipc.py with an existing mono asset
and a new output directory. Then manually verify:
1. Open Audio from rail; select hierarchy source; right dock edits that source.
2. Add clip from Assets; timeline and inspector select the same clip.
3. Move/trim with snap on/off, split, duplicate and delete; undo/redo once per drag.
4. Pan knob, fader and numeric dB entry; left/right balance and preview mute.
5. Capture, finish, export/progress/cancel and file conflict display.
6. Narrow window/high DPI, scrolling, overlapping clips and existing paint/hair/
   sculpt dock switching. Viewport picking stays within the reduced viewport.

Static checks do not establish successful application compilation or visual QA.

2026-10-14 live probe on the user's initial UI build: editor modes, atomic invalid
selection rejection, shared clip selection, hard-right offline bus balance and
clip selection cleanup PASS. Test clips removed; mixer/editor restored.
The subsequent optional-inspector and direct timeline insertion changes await
the next user build. The inspector no longer reserves a fixed 340px viewport strip.
