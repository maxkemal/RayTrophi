# Domain creation placement audit — 2026-11-09

Status: placement correction implemented; application build and live verification pending.

## Implemented contract

Add Domain always uses SimulationGridDomainDesc's canonical ground box:
[-2.5,0,-2.5] to [2.5,5,2.5], independent of selection or other scene geometry.
From Select reads canonical flat selection identity, aggregates every TriangleMesh
part with the same node name, and measures DNA SoA world vertices from P_orig x
getFinal(), or P when no local transform is available. It does not read per-face
Triangle facades, selection-position caches or the legacy facade-first resolver.
The fitted result is a manual box; later source movement does not bind/move it.

UI, `rt.fluid.create_placed_domain(name="", source_object="")`, and IPC
`fluid.create_placed_domain` call createPlacedMatterDomain. Empty source means
ground default; a named source explicitly requests fitting. Existing explicit-
coordinate fluid.create_domain keeps its semantics. Missing/deleted/empty,
nonfinite or zero-volume source geometry rejects before domain addition; no
silent fallback to another location. Explicit duplicate names reject; omitted
names get a unique Physics Domain N. Successful creation selects the domain and
invalidates simulation cache. Final-render locks reject creation.

Non-build check: scripts/test/check_domain_placement_contracts.py.
After user rebuild, external disposable-scene test:
python scripts/test/rt_test_domain_placement_ipc.py.
Manual: compare Add Domain with/without a translated mesh selected; both must
rest at y=0. From Select must fit a translated/rotated/scaled flat mesh. Check
multi-material flat parts and missing/empty/planar source error handling.
No C++ tests or builds are run by the agent.

## Original panel behavior (before correction)

`scene_ui_simulation_domains.cpp` initializes the Add Domain center to (0,1,0).
When the selected item is Object, has a non-null legacy `object` pointer and a
nonempty node name, the current world-space bounds resolver supplies the center.
The object pivot is not the center used here: it is the world AABB midpoint.
The new domain is a manual 5x5x5 m box centered there.

Without a qualifying selection, objects elsewhere in the scene are not consulted.
The resulting bounds are (-2.5,-1.5,-2.5) to (2.5,3.5,2.5). This is a deterministic
fallback, not random placement based on an unselected scene object. If the selected
bounds cannot be resolved, the same fallback is used without an explicit message.

The selection gate uses the compatibility `object` facade, rather than canonical
`mesh_object` / object identity. A mesh selection without that compatibility pointer
can therefore take the no-selection branch. Most current selection constructors
create a compatibility facade, so this is a structural inconsistency, not proof
that it is the cause in the user's open scene.

The descriptor itself has different defaults: (-2.5,0,-2.5) to (2.5,5,2.5). Thus
panel Add Domain's unselected default extends below y=0, while an untouched
SimulationGridDomainDesc rests on y=0. These two default definitions should be
unified when the placement policy is chosen.

From Select explicitly fits the object's bounds and then stores a manual box;
it does not keep the domain bound to the source object's later movement.
The API createFluidDomain receives explicit bounds and does not infer selection.
ParticleSimulationSystem::addGridDomain stores the descriptor without relocating it.

A predictable UX policy to consider: ordinary Add Domain uses the canonical
fixed default, and From Select explicitly places/fits around selected geometry.
Alternatively preserve selection-based centering but expose its fallback and use
canonical flat selection identity. Do not change these semantics implicitly.

## Seed-layer work paused by the user

User reports that after adding a second seed layer, selecting/dragging a layer can
lose the owning domain selection/focus. Earlier label transparency/corner placement
changes did not establish correct end-to-end behavior. Treat viewport layer
selection and dragging as unresolved; no claim of live verification.

User also requests keeping box layers inside the domain. Current authored boxes
can extend outside; generation clips to the effective liquid grid. Strict authored
containment / constrained dragging has not been implemented. This remains open
and must use common service semantics for UI, scripting and IPC when resumed.

Do not resume these seed fixes until requested; the user asked to record them and
switch to inspecting domain creation placement.
