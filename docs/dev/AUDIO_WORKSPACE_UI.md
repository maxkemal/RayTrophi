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

## 2026-10-14 editor refinement (source; new build pending)

The bottom bar now has one Audio entry for both audio modes. Timeline/Mixer switches
live inside that editor; script/IPC mode names remain `audio` and `audio_mixer`.
Timeline rows are existing mixer buses, not independent custom audio tracks.
Overlaps still receive additional display lanes. Dragging vertically reroutes to the
bus under the pointer and highlights that destination. Changes are applied through
canonical clip_update during the gesture, before rendering the canvas. Release
records one undo entry. Selection, trim, gain and fades use the same service as IPC
and scripting. When switching audio modes, the gesture is finalized.

- Body: live time/bus movement. Edge: inward trim, reversible within the gesture.
- Horizontal gain line: vertical drag (0.25 dB/pixel, -60 dB off, max +24.08 dB).
- Square fade handles: fade-in/out drag. Preserved envelope offsets after trim/split
  are respected; edits use existing linear envelope semantics.
- A selects; B activates Razor, then click strictly inside a clip to split at the
  snapped cursor time. Split at playhead, duplicate/delete remain available.
- Waveform amplitude reflects evaluated gain/fades. Hover explains each affordance.
- Mixer: adaptive 136..200 px strips, clear bus-to-Master routing, Mute/Solo buttons,
  dB ticks, segmented RMS energy, transient peak/hold and overload indicators.
  Meters are existing aggregate bus measurements, not invented independent L/R values.

Clip authoring no longer clears all runtime voices. The runtime compares each
voice's asset/path/source range/bus/loop configuration; gain/fade/time edits reuse
voices, structural changes rebuild only the affected voice. Start time edits seek
according to the existing scene-clock evaluator. No decoder work is added to the
audio callback.

Generate sound creates a normal WAV asset and optionally a clip without requiring
an existing audio import: see [Procedural audio](SCENE_AUDIO_SYNTHESIS.md).

Build Studio only, then check:
1. One Audio bottom entry; Timeline/Mixer switch internally, other editors still open.
2. Generate a metal/wood/tone/wind clip in an empty scene; listen with timeline Play.
3. Keep mouse held while moving between Effects/Ambience/Music; waveform follows
   continuously, bus highlight and inspector agree, release then undo once.
4. Drag gain line while playing; unrelated clips continue; gain/fade edits do not
   re-decode all voices. Drag fade squares, trim, reverse trim before release.
5. B + click cuts at cursor; A resumes selection; split/trim preserve envelope/source
   travel. Reopen a saved scene; gain/fades/routing remain.
6. Short/narrow/high-DPI layouts, horizontal/vertical scrolling, overlapping clips,
   meter readability and one undo per mixer fader/knob gesture.

Run `rt_test_audio_synthesis_ipc.py` externally for new core contracts and the
existing `rt_test_audio_workspace_ipc.py` for mixer/export regression. Static audits
cannot confirm build, mouse interaction or sound quality.

## 2026-10-14 user acceptance and right-edge extension

User build/listening confirmation: synthetic sounds are correct; timeline clip fades
work; held-mouse movement updates live without issues. These confirmations do not
establish telemetry performance or geometry acoustics acceptance.

New source changes after that build: inspector labels sit above full-width inputs;
clip/source/reaction settings no longer reserve a fixed label gutter on the right.
Automation time/value fields stack in narrow layouts. Minimum inspector size is
180x180; horizontal scroll is reset so a former overflow cannot hide left content.

Right-edge drag and inspector Duration call canonical clip_resize(id, end_seconds,
auto_loop=true). First reveal available source; once pitch-adjusted source travel
exceeds source_out-source_in, repeat the selected source range and show LOOP on
clip. This is repetition, not time stretching. Existing phase, automation knots,
envelope offset and fade durations remain; extending beyond the envelope horizon
moves its fade-out to the extended horizon. Left-edge trim remains inward only.
Dragging back within the same gesture reconstructs from the original clip, so an
automatically introduced loop is removed if the original clip no longer needs it.
Once committed, loop remains an explicit clip setting until disabled.

Script rt.audio.clip_resize and IPC audio.clip_resize share the same validator;
auto_loop defaults false to reject unintended repetition. Unknown/nonboolean
parameters, invalid end, or a nonloop source overrun fail atomically. The existing
synthesis IPC probe now includes source recovery, explicit looping, invalid loop
flags, preserved gain/fades and split after extension. No shader change is needed
for these latest UI/resize changes. Build Studio, then manually check inspector
widths and source-aware extension while playing, single undo, save/reopen and WAV.

## Sound designer v2

[Designer contracts and acceptance](AUDIO_SOUND_DESIGN.md). Audio bottom navigation
now adds Designer; existing Timeline/Mixer names remain unchanged. Asset Design /
process opens that asset in the designer, with controls in the existing Audio
Inspector. Preview is in memory; Save asset uses a name and resolved project/library
folder with a picker and collision suffixes. Source-only delivery, build and listening
acceptance pending.

## Sound designer v2

[Designer contracts and acceptance](AUDIO_SOUND_DESIGN.md). Audio bottom navigation
now adds Designer; existing Timeline/Mixer names remain unchanged. Asset Design /
process opens that asset in the designer, with controls in the existing Audio
Inspector. Preview is in memory; Save asset uses a name and resolved project/library
folder with a picker and collision suffixes. Source-only delivery, build and listening
acceptance pending.

2026-10-14 build checkpoint: user confirmed successful Studio compilation after
Designer MSVC fixes. New listening/UI/external IPC acceptance is pending the user's
test results; earlier accepted synthesis/fade/live movement stays recorded above.
