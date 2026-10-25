# Shared Matter phase output

Output chooses representation by current phase across all substances in one
Matter domain. No Snow-specific settings, virtual emitters, new particles,
retagging, solver handoff or physical-state edits are involved.

## Authoring

Matter Output starts with Phase output (all substances):

- Solid phase: Follow substance / labels, SDF, Splat, Fog, Hidden.
- Liquid phase: the same choices.
- Gas / vapor phase: Fog or Hidden. Gas is an Eulerian field, not splat carriers.

For the user's melting scene: Solid=Fog, Liquid=SDF, Gas=Fog. This also applies
to Sand, Ice, other named substances and untagged seeded carriers. Existing
per-substance output/material/geometry bindings remain below the common table.

Explicit solid/liquid phase selection overrides parcel-label routes, including
spray and mist. Follow preserves existing label and substance resolution. Massless
secondary whitewater retains its label routing: it has no physical carrier phase.
Unknown physical state retains legacy routing. Canonical granular/elastic models
and the frozen flag select Solid; canonical Fluid selects Liquid. A mushy carrier
uses its current canonical model, not a guessed temperature or new phase split.
Particle substance identity is retained through melting; representation changes
never fabricate a Water identity, emitter, mass source or new thermal material.

The common FluidViewPlan controls SDF, Splat and carrier Fog selection. Live keys
now include physical phase, so same-tag/same-label solid and liquid carriers can
coexist in different views. SDF signatures include phase routing and live carrier
phase, including paused-frame edits and phase changes without position changes.
Authored routes keep resources allocated across empty frames. Source mesh banks
remain available when phase output selects Splat over a non-Splat base binding.

Gas Hidden suppresses the gas-grid density and gas temperature contribution;
carrier Fog remains selectable in the same volume. Gas Fog continues to use the
existing gas shader. This change does not create separate shaders for solid Fog
and vapor, and does not convert gas to SDF/particles. Grid condensate remains part
of the existing grid Fog path.

## API and persistence

Script and external IPC share the same render-only undoable service:

```python
rt.fluid.get_phase_output("Physics Domain 1")
rt.fluid.set_phase_output("Physics Domain 1", solid="fog", liquid="sdf", gas="fog")
```

IPC: `fluid.get_phase_output` and `fluid.set_phase_output`, with `domain` and
optional `solid`, `liquid`, `gas` string fields. Omitted fields retain values.
Invalid routes, nonstrings and unknown fields reject the whole patch. Gas accepts
only follow/fog/hidden. Non-Matter domains are rejected. Defaults are all follow.
Both project serialization paths save `fluid_phase_output`; older files keep
their existing render behavior. No simulation reset or bake invalidation occurs.

## Live ceiling diagnosis (read-only, before user build)

The open scene was paused at frame 58. Domain `Physics Domain 1` reported:

- Gas boundary `closed` (fluid domain also `closed`).
- World and effective ambient both 293 K; environment override disabled.
- Effective ambient stratification 0.
- 1,406 Fluid-model carriers and no remaining granular-model carriers in that
  snapshot; their untagged Snow identity did not independently select a melt view.

The closed boundary is a concrete explanation for retained vapor at the domain
ceiling. The inspected gas solver directly maps this descriptor to Closed. No
atmospheric temperature mismatch was established. This inspection neither steps
nor changes the scene. Output visibility and physical boundary conditions are
independent; the shared physical boundary was not changed. Open outflow is not
automatically a transfer to a global atmosphere simulation.

## User build / acceptance checklist

Codex did not build, compile shaders or launch Studio. C++ application rebuild
is needed for the new UI/API; no shader source was changed for phase output.

1. Open the existing melting scene, pause, choose Solid=Fog, Liquid=SDF, Gas=Fog.
   Confirm surviving solid and melted carriers have separate representation.
2. Repeat with another substance and an untagged seed: no Snow-only behavior.
3. Rewind/play and verify phase changes reroute automatically; routing itself
   never duplicates/removes carriers or resets physics.
4. Gas=Hidden must hide grid Fog while keeping solid/carrier Fog visible.
5. Undo/redo and save/reopen must retain the phase table and all material bindings.
6. After build, external probe while paused:
   `python scripts/test/rt_phase_output_ipc.py --domain "Physics Domain 1"`.
   It checks round trips, patch retention, transactional refusal and unchanged
   physical state; restores original phase output. It does not reset or step.

Visual runtime acceptance of the new routing and gas visibility remains pending
the user's build. Sharing a Fog shader/volume between solid Fog and vapor remains
an existing visual limitation, not a new per-phase material system.

Static verification: IPC capability/discovery audit passes (688 methods, 669
documented), descriptor generation is current, Python probes parse, JSON/project
XML parse, and `git diff --check` passes. These are not C++ compilation or live
acceptance results for the new feature.
