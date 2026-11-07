# Output material assignment on a paused SDF frame

User report: Output material assignment did not repaint the current SDF in RT
or RayFusion, even after render reset; changing timeline frame made it appear.

The material indices live in the composition field. Volume-property refresh
and accumulation reset alone do not replace that field. The source already
hashes material IDs and bumps composition/UVW versions on publication, but the
UI deferred publication to the timeline resync path. The scripting/IPC setter
only refreshed volume properties and reset accumulation, without requesting
composition publication. Timeline authoring gates and cached-frame restoration
are inappropriate dependencies for this visual operation.

`SceneData::refreshFluidSurfaceMaterialBindings` is the shared commit for UI
assignment/clear/new-material and `fluid.set_substance_material`. It updates
volume properties, dirties material tables for both backends and forces render
publication directly from the current live state. It does not restore cache,
advance the playhead, step the solver or edit particle state. Existing guards
still defer publication while backend rebuilds are in progress. Cheap IOR and
shader-property edits retain the original refresh-only path.

UI binding changes also reset accumulation immediately, like the API setter.
The full SDF/composition rebuild can cost time on a large grid; this fix is for
correct current-frame publication, not a composition-only performance rewrite.

Build: C++ only; no shader change. Source/whitespace checks only by agent.
No C++ test files added, per user preference.

IPC/manual acceptance after user compilation:

1. Pause on an existing cached/nonzero SDF frame and record timeline frame,
   particle count/state digest and existing volume table versions.
2. In Output, assign two visibly different materials in turn, clear the
   assignment, and create/assign a new material. The current image must change
   without Play or timeline navigation, in both RT and RayFusion.
3. Repeat with `fluid.set_substance_material` through the external named-pipe
   client. Read back `fluid.get` and `render.volume_slots`; check current material
   indices/composition version and unchanged playhead/particle state.
4. Check domain Surface Material assignment as well as per-substance assignment,
   a single-material/mixed-material surface, and ordinary IOR edits.
5. Repeat while Keyframe Edit Mode is enabled: appearance commit must not need a
   physics step or cache restoration. Check existing explicit authoring locks.

The user subsequently compiled and confirmed current-frame material assignment
works and the reported problems are resolved.
