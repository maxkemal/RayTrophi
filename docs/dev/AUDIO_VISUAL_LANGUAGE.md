# Audio workspace visual language

## Screenshot feedback correction

User-provided test1.jpg, inspectorpanel.jpg and solpanel.jpg showed that the
fixed 280px Acoustics child wasted the tall right dock, keyboard-only property
fields felt like forms, and New/Listen/View menus concealed primary operations.
Those images are user artifacts, not agent verification captures.

Source correction: remove the Acoustics child and redundant outer tree indent;
normal inspector scrolling now owns the content. Show Enable scene acoustics
and audible-effect toggles first, with scene scale/geometry and technical reports
as secondary disclosures. Source/asset list height follows available left-panel
space instead of a fixed row count. Add/import/play/stop and inspector visibility
are direct controls again; Timeline/Mixer/Designer remain in the bottom navigation.
The left panel adds a real Master peak meter, explicitly labeled as the combined
scene signal before output mute, not a selected-source meter or speaker monitor.

Volume/pitch use logarithmic sliders; position/direction support drag editing.
Distance, falloff and cone controls use sliders with readable labels. Distance
and cone pairs constrain each other's interactive ranges; canonical service
validation still decides whether an authored value is valid. Ctrl+click preserves
precise keyboard input. Local drafts are scoped by source/widget identity and
commit through the existing authoring API on release, so a drag is one undoable
edit. They are not live audio automation: audible changes apply on release.

This correction has static source checks only and needs the user's next build.
Check drag/release, Ctrl+click, undo, switching source/tab during a gesture,
zero/large values, paired cone/distance boundaries, short/tall docks and the
existing bottom navigation. It supersedes conflicting menu/fixed-area descriptions
below. No screenshot of the changed build or live UI acceptance is claimed.

2026-10-18 source delivery. The user requested fewer buttons, readable active
states, shorter panels and a coherent modern presentation. No application build,
launch, screenshot capture or live visual verification was performed for this UI
change. Earlier 20-case core acceptance is not acceptance of the new layout.

## Information hierarchy

Keep the existing viewport, context rail, right inspector and bottom editor.
The scene workspace starts with New / Listen / View menus, rather than a stack
of equal-weight action buttons. Output enabled/muted has an explicit text label,
theme-derived active styling and a tooltip explaining its preview-only scope.
Subsequent user feedback: this output button made mute semantics unclear. Live
IPC diagnosis found preview_muted=true while the device and mixer buses were
ready; unmuting through the canonical service restored output_muted=false.
Source now uses an explicit `Mute output` checkbox and explains that it silences
both scene sounds and asset preview. The source inspector also offers an Unmute
output action when muted. These new UI controls await the next user build.
Timeline/Mixer/Designer navigation retains its existing selected-view state.

Source and asset lists use bounded scroll areas. Assets add search; their rows
expose Preview and Actions, with design, recipe reuse, clip/source creation,
relink and remove in the row menu. Destructive operations are separated by a
divider. No service operations were removed.

The source inspector shows resolved playback status and input kind first,
followed by name and a Play/Pause action, Stop and an Actions menu. Play/Pause
reflects actual source_runtime playback state, not a cached last-click flag.
Generated sources are playable without an asset and no longer show the missing
file warning. Status distinguishes invalid position, timeline hold, suspended
automation, active playback, fading generator tail, pause and waiting/stopped.
Playback state is not proof of audible speaker output. Zero effective source
gain has its own explanation.

Properties use Sound / Space / Response / Acoustics tabs. Response mapping is
collapsed under Advanced response mapping. Acoustic controls remain in Acoustics;
trace reports and implementation details are collapsed under Technical diagnostics
inside a bounded area. Acoustic queries are not triggered just by opening tabs.

Scoped listening keeps the canonical selected/group/scene operations. Acoustic
configuration uses one popup with explicit Apply, reducing four competing action
buttons to a configuration action and Dry/Processed comparison. Status derives
from actual configured entries; mixed and partially configured targets are
reported without falsely highlighting a uniform state. A/B is disabled until the
whole target is configured. Detailed per-target diagnostics are collapsed and
bounded. Configuration popup checkboxes represent proposed settings to apply,
not the current heterogeneous scene state, and are labeled accordingly.

Selecting a file for a generated source uses one canonical update to replace
generator and stored history with the selected asset. The Sound tab explains
this replacement before the file chooser. Undo uses the existing authoring path.
All source/property/transport/acoustic mutations call the existing shared
AudioAuthoring/AudioService operations, already available to Python and IPC.
No UI-only audio business logic or new persistence schema was added.

## User build and visual acceptance

Build Studio; no shader change is needed for this UI slice. Check:

- Narrow and wide right docks: menus/tabs remain reachable; list and diagnostic
  scroll stays local; source selection and full names remain identifiable.
- New/import, asset Preview/Actions, duplicate/delete and undo retain their actions.
- File and generated sources: source name, asset replacement/undo, Play/Pause,
  Stop, timeline hold, automatic waiting/suspension and fading-tail labels agree
  with actual source_runtime state. Output mute must be visible without opening a menu.
- Selected/group/scene: empty and partial targets, uniform Dry/Processed and mixed
  bypass states; Apply uses current scope and service errors remain visible.
- Sound/Space/Response/Acoustics: edits survive tab changes, do not share a local
  configuration between unrelated sources, and technical diagnostics start closed.

Static checks covered lexical delimiter balance, VS XML registration, installed
ImGui API availability and diff whitespace only. They are not compile or layout
acceptance. Full generator/history authoring forms and further Designer/Timeline
presentation refinement remain subsequent UI work.
