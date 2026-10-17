# Shared object physical materials

2026-10-15 source delivery, not built or live-accepted. The user requested that
test scripts stay at the end of structural work; no new test script or live probe
was run for this change. Codex did not build or launch Studio.

## Authority

`SceneData::object_substances` stores the authored physical material of a named
flat mesh node. This is scene metadata, independent of domain membership, collider
creation, render shader IDs and per-face Triangle facades. Multi-material flat
meshes sharing a node name share this physical binding. Names follow the existing
scene node identity convention; no new stable node-ID system is claimed.

A binding contains a SubstanceLibrary name and the existing SubstanceOverride
fields. Project substances keep their existing built-in -> derived profile
inheritance. The binding never copies density, thermal data or acoustic fields
out of the library. Runtime temperature, wetness, burn state and damage remain
simulation state, not authored identity. Per-face physical assignments, layered
construction and generic bindings for lights/cameras/volumes/instances are future
extensions; this authoring surface currently targets canonical flat mesh nodes.

Collider MSF substance/override fields become compatibility projections when an
explicit object binding exists. Every existing particle system consumes that
binding during scene/runtime synchronization. A changed projection clears the
corresponding MSF state and material damage history. Assignment invalidates sim
frame caches, requests timeline resync and resets render accumulation. It does
not create a particle system, collider, domain or physics telemetry subscription.
Shape, collision enablement, contact friction/restitution and solver quality are
not silently changed by selecting a substance.

Without an explicit object binding, existing colliders retain their values. A
new flat object is Unassigned. The legacy collider descriptor still defaults to
Wood (Oak); creating such a collider can therefore produce legacy_collider
readback without creating an explicit central object assignment. In the UI, select
the mesh and use Physical material -> Substance above Material Properties.

A read reports `legacy_collider` only when all matching object-backed colliders agree
on both substance and overrides. Disagreement reports `source=conflict`, a null
effective binding and the per-system collider rows. No first-wins migration is
performed. `adopt_object` explicitly promotes a valid unambiguous legacy binding.

## UI, Python and IPC

The selected mesh's existing properties panel has a **Physical material** section.
It is also available beside an object-backed collider's material controls. Both
use the shared `invokeObjectSubstance` service. The existing substance editor
derives/edits profiles through the existing substance API. When object authority
exists, legacy collider material/override controls are disabled; collider API
updates that conflict with object authority are rejected. Collider creation reads
the object's material automatically. Clearing object authority leaves the last
projected values on existing colliders for compatibility.

Python uses keyword arguments; external IPC has identical method names prefixed
by `substance.`. Operations return their value directly, without an `ok` wrapper.

| Python `rt.substance` / IPC suffix | Parameters | Result |
|---|---|---|
| list_objects | none | Flat mesh nodes plus saved unresolved bindings |
| get_object | object | exists, source, binding, resolved, collider rows, conflict |
| set_object | object, binding | Updated resolved object state |
| clear_object | object | State after clearing central authority |
| adopt_object | object | State after explicit legacy promotion |

```python
rt.substance.set_object(
    object="Wall",
    binding={"substance": "Wood (Oak)", "overrides": {"burn_rate_scale": 0.7}},
)
info = rt.substance.get_object(object="Wall")
# IPC: substance.set_object with the same object/binding JSON.
```

`source` is object, legacy_collider, conflict or unassigned. A missing mesh can
retain an inspectable binding (`exists=false`); clear_object permits removing it.
`profile_fields` is the resolved LIBRARY profile. The separately returned binding
overrides are still applied by the existing MSF conversion path; profile_fields
does not pretend those overrides were already applied to the profile.

Get/list require Read. Set/clear/adopt require SceneWrite and participate in scene
undo/redo. They are blocked by project loading or a final render job. Unknown
operation/parameter/binding/override keys, wrong types, missing/unknown substance
names and invalid numbers are rejected before changing authored state. Assignment
and adoption require an existing non-transient flat mesh node.

Names contain 1..1024 characters; at most 65536 explicit bindings. Overrides are
replacement values, not a JSON merge patch: omitted values use defaults.
`override_ignition` is boolean; ignition_kelvin is finite 0..100000 K; burn/fuel
capacity scales are finite 0..1000000. Values are normalized to the float precision
of SubstanceOverride so synchronization does not repeatedly detect double/float
rounding as a material edit. Null is not a field-reset instruction here; omit the
field to use its default, or derive a library substance for other differences.

## History and persistence

Both .rtp ProjectManager and the fallback SceneSerializer save `object_substances`
after the substance library is available. Restore validates the complete candidate
before replacement. A missing section restores an empty registry and retains old
collider behavior; scene clear removes all bindings. Unknown substance references
in a new binding section are rejected instead of becoming the default substance.

Set/clear/adopt history captures both the authored binding and matching collider
material values, so undoing an initial assignment restores the original legacy
values, including previously conflicting domain assignments. Ordinary object
delete and flat duplicate commands preserve explicit binding metadata across
undo/redo. Duplicates get independent assignments; no geometry/facade scan is used
as physical authority. A geometry duplicate does not implicitly adopt legacy
collider metadata. Profile deletion checks all scene systems and object bindings.
Object-material undo commands and deleted/duplicated bindings retain library
references until those history entries are discarded, preventing dangling undo.

## Acoustic material contract

SubstanceProfile and its existing editor/API now expose `acoustic_defined`,
`acoustic_scattering` and absorption/transmission fields at 125, 250, 500, 1000,
2000 and 4000 Hz. Example keys: acoustic_absorption_1000 and
acoustic_transmission_1000. Each coefficient is an ENERGY fraction in 0..1.
Absorption + transmission must not exceed 1 in any resolved profile; invalid
patches, including inherited descendant conflicts, do not publish the candidate
library. Scattering is the fraction of reflected energy routed to diffuse paths.

Object readback includes `acoustics`. The shared AcousticMaterial response computes
reflection = 1 - absorption - transmission, then splits it into diffuse/specular
energy. Unconfigured profiles return `defined=false` without invented band values.
Built-in materials deliberately remain unconfigured until supported response data
is authored. These coefficients describe an authored boundary response; thickness,
layering, openings and incidence-dependent transmission need the later geometry /
construction model. Density or shader transparency is not treated as a measured
acoustic response. No propagation, audible occlusion or echo playback is claimed.

## Final build and acceptance backlog

2026-10-18 UI follow-up: the shared editor should use logical tabs and a bounded
internal body across objects, domain defaults, flow sources and collider contexts.
This is broader than audio. See [shared Substance UI direction](SHARED_SUBSTANCE_UI_ROADMAP.md)
for grouping, context/assignment semantics, pending edits and acceptance. It is a
recorded UI task, not part of the current acoustics-core implementation.

User build: Studio C++ only for this change, no new shader. Source checks cover
new-source delimiter/size/whitespace, VS XML/registration, all 14 acoustic schema
fields and five IPC methods including their parameter/capability descriptors.

At the final script/manual pass: assign without any domain and confirm no runtime
creation; script/IPC/UI parity; malformed input atomicity; legacy agreement and
conflict/adoption; multi-system collider projection; rejection of conflicting
collider edits; no repeated MSF reset for float values; assignment undo/redo restoring
legacy values; duplicate/delete undo; project save/reopen and older projects;
library deletion with live/history references; profile inheritance and acoustic
energy-budget validation; unconfigured acoustic readback; narrow-panel interaction.
