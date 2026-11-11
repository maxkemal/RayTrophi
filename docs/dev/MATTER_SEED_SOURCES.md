# Initial Matter layers

A Matter domain can hold multiple one-shot box or mesh-volume recipes. Each layer
has a substance, initial velocity and a material/domain/custom temperature policy.
The domain can expose temperature and Heat Exchange without any emitters.

## Authoring

In Initial Matter Sources, add layers and select their substances and volumes.
Generate particles from all layers replaces the entire initial particle state,
rebases the scene timeline to zero, and arms one shared optional settle pass.
Draft edits do not regenerate live particles. Continuous emitters remain separate.
Settling is shared, before physical time/sources/heat advance; its finite stopping
criteria do not promise perfect equilibrium.

Automatic particle capacity is on by default. The planner calculates a conservative
particle upper bound from the effective liquid grid and current recipe, then adds
15 percent plus 64 slots. Successful Apply/reset adopts this capacity. It is not a
requested particle count and does not change voxel size, DEM radius or material
physics. A small previous manual limit therefore no longer blocks automatic mode.
For emission beyond the initial layers, choose a larger manual capacity in Advanced
seed settings; automatic sizing plans initial layers, not an emitter's lifetime.

The preview reports an upper bound, not an exact count. Continuum bounds include
boundary cells; solid cells and mesh holes reduce actual births. DEM bounds count
the FCC lattice candidates before collider, mesh and birth filters. The shared
birth-temperature resolver selects DEM versus continuum, including hot births.
There is no per-layer particle-count knob.

## Viewport

Enabled layers show compact translucent `name | substance` labels outside the
projected upper-right box corner. The selected box label passes mouse input
through so move/scale handles remain usable. Labels also pass input through
while ImGuizmo is hovered or being dragged. Click a label to select the
layer and its domain. Box sources have a highlighted outline and use G (Move) and
S (Scale) with the existing gizmo. Boxes stay axis aligned; rotation is not authored.
The panel's Edit in viewport button uses the same selection operation. Escape or
Edit domain instead of layer restores domain manipulation. Viewport bounds edits
call the same validated update service as scripts and IPC, and leave live particles
unchanged until Generate.

For a mesh-volume layer, geometry remains the existing flat scene object. Use Edit
source mesh in viewport in the panel, or the label's right-click menu, to select
that object for its normal transform tools. Labels follow the resolved object
bounds. Mesh-volume layers do not create a second transform on top of the object.
Labels follow the existing Show Gizmos visibility setting. Box bounds are world
coordinates and do not automatically follow later domain movement.

## Memory planning

The preview exposes **planning allowances**, not a calibrated peak allocation:
2048 GPU bytes and 4096 host bytes per planned particle, plus 512 GPU bytes and
1024 host bytes per effective liquid-grid cell. These cover anticipated copies,
state/contact data and scratch conservatively; actual demand depends on solver,
contacts, rendering and allocator behavior. They are not a proof of allocation
success and do not include all other scene resources.

The default memory guard compares the allowance to 70 percent of currently
available system RAM and, for a Vulkan domain, the simulation GPU's driver budget
via VK_EXT_memory_budget. Measurements are cached for up to one second. Driver
budget support missing means unmeasured, never zero available bytes. CPU/CUDA
paths have the RAM check; CUDA VRAM is not measured by this planner. Grid resource
budgets already enforced by the domain remain in force independently.

When the guard would be exceeded, the panel shows allowance, available memory and
suggested changes before generation. Apply rejects without replacing existing
particles or changing capacity. Advanced memory_guard=false makes that warning
advisory; it does not bypass index-width or allocator failures. No quality setting
is silently reduced. Real allocation errors still propagate from the existing
allocation paths; the planner cannot guarantee recovery from OS/device exhaustion.

## Shared scripting and IPC

All authoring calls use `rtapi::matterSeedSources`. Python uses `rt.fluid` methods;
external IPC uses the same names prefixed by `fluid.`:

```python
rt.fluid.set_seed_source_settings(domain, {
    "auto_capacity": True, "memory_guard": True,
    "particles_per_cell": 4, "settle": True,
})
recipe = rt.fluid.add_seed_source(domain, {
    "name": "Snow layer", "substance": "Snow",
    "bounds_min": [-1, 0, -1], "bounds_max": [1, .3, 1],
    "temperature_mode": "material",
})
layer_id = recipe["sources"][-1]["id"]
rt.fluid.select_seed_source(domain, layer_id)
rt.fluid.update_seed_source(domain, layer_id, {
    "bounds_min": [-1, .1, -1], "bounds_max": [1, .4, 1],
})
rt.fluid.apply_seed_sources(domain)
```

Other methods: get_seed_sources(domain), remove_seed_source(domain,id).
select_seed_source(domain,0) selects the domain gizmo. Selection is transient;
unknown, negative or fractional IDs reject without changing it. Sources have
stable positive IDs. Names are unique and 1..63 bytes. Mutable fields are name,
enabled, substance, bounds_min/max, mesh_object, velocity, temperature_mode,
temperature_kelvin (1..3000 K), grain_saturation (0..1, wet-capable DEM only).
IDs are read-only. Unknown keys, nonfinite/invalid bounds and values reject.
Shared settings also support settle_max_steps (1..4000) and settle_tolerance
(0..1 m). auto_capacity and memory_guard accept booleans only.

get_seed_sources returns sources, resolved birth temperatures, settings,
selected_source_id, pending_recipe_changes and plan. The plan includes per-layer
particle_upper_bound, total particle_upper_bound, suggested_capacity, heuristic
host/GPU planning bytes, measurement availability and memory_check_passed/message.
A plan error is reported as plan.error; a listing does not mutate particles.

## Limits and persistence

Only one named DEM material and one named MPM material can own a domain, including
potential emitter materials. Water + DEM Sand + MPM Snow is supported. Two distinct
MPM materials such as Snow and Mud reject instead of sharing wrong parameters.
Overlapping clipped layer AABBs reject at Apply, including conservative mesh AABB
intersections. Missing/empty mesh, empty births and insufficient manual capacity
also reject. Generation stages all particles before replacing live data; there is
no partial layer truncation. The existing single-region seed command rejects while
a layered recipe is active; clear particles and recipe to return to that workflow.

Both project serialization paths save the recipes, automatic-capacity and memory-
guard settings. Selection and pending commands are not saved. Successful Apply
arms reset replay; removing the last layer leaves an explicitly empty layered
recipe. Saving drafts saves the desired recipe, so Apply before save when exact
validated initial replay is intended. Live particle/cache persistence is separate.

## Verification

No C++ tests or builds are run by the agent. Non-build wiring checks:
`python scripts/test/check_matter_seed_sources_contracts.py`.
After the user rebuilds, run externally in a disposable empty scene:
`python scripts/test/rt_test_matter_seed_sources_ipc.py`.
It creates/removes its own domain, rebases the scene timeline and checks mixed
births, temperatures, automatic capacity from a one-particle manual limit,
selection, deterministic replay, invalid settings and preservation on failure.
Manual viewport and save/reopen checks are in NEXT_BUILD_CHECKS.md.
