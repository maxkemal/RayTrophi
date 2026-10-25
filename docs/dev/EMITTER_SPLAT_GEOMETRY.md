# Substance Splat geometry in Output

The canonical authoring path is **Output → substance → Splat → Splat Geometry**.
Geometry is stored per domain/substance, independently of emitters. Two emitters
pouring Sand share Sand's output geometry; Water can retain its SDF surface.
Emitter ordering, renaming and deletion do not control this output binding.

**Represented by** offers Sphere or **Scene Objects (multiple selection)**. The
scene-object list extends the previous single-object selector to multiple objects.
Each particle uses a stable choice from the selected library. Scene Objects uses
the selected geometries at all distances by default. Optional **Use spheres beyond
distance** exposes this substance's Mesh Distance and Mesh Budget. Zero budget in
that optional mode keeps spheres. Face materials/UVs can be preserved, or existing
substance/domain/wet materials used. Apply Geometry commits one undoable change;
Revert discards a draft. SDF/Fog substances do not show the geometry selector.

The emitter panel has no geometry authoring controls. The shared preview section
contains sphere/material preview defaults, not another geometry-object selector.

## Independent generator

**Add / Procedural Generator** creates editable scene objects for any purpose.
It does not know about output bindings, substances or emitters and does not assign
its result. Select those objects separately in Output if they should represent a
Splat substance. The generator UI sends no target_source/domain settings, fixing
the previous unknown-generator-setting error. Unsupported assignment inputs are
rejected. Generation retains its single undo step for scene-object creation only.

## Shared scripting and IPC

UI, Python and IPC call the same get/set services. Geometry is read from canonical
flat TriangleMesh/DNA data, with no per-face scene facade authority.

```python
rt.scene.generate_grains(name="Grain", type="rock", count=5)
rt.fluid.set_substance_splat_geometry(
    "Physics Domain 1", "Sand", mode="scene_object",
    objects=["Grain_1", "Grain_2", "Grain_3"], source_materials=True)
rt.fluid.get_substance_splat_geometry("Physics Domain 1", "Sand")
```

IPC methods `fluid.set_substance_splat_geometry` and
`fluid.get_substance_splat_geometry` take `domain` and `substance`. The setter
accepts overlay fields `mode`, `objects`, `source_materials`, `near_distance`,
`mesh_budget`. Modes are `sphere`, `scene_object` (multiple objects, all distances)
and `mesh_grains` (optional distance/budget fallback). Omitted fields retain values.
Unknown fields, invalid modes/numbers, duplicate/empty/missing meshes and render
locks reject the live mutation atomically. Physics is unchanged.

`fluid_substance_splat_geometry` is saved by both scene serializers. Rendering
selects the geometry bank from the particle's substance tag before any legacy
emitter routing. Existing disk caches retaining substance tags therefore do not
need new birth-source attribution for this path. Initial fills carrying that tag
use the same geometry. The old source-geometry API remains compatibility only;
substance Output bindings/configurations take precedence. New output bindings
without a geometry configuration default to Sphere.

## Verification

Codex did not compile or launch Studio for this change. Source checks include IPC
capability/discovery schemas, Python syntax, project registration and diff checks.
After the user build:

1. In Output, set Water to SDF and Sand to Splat. Only Sand should show geometry.
2. Generate independent objects through Add / Procedural Generator. Creation
   must succeed and must not change any output binding.
3. Select multiple objects for Sand and Apply Geometry. All Sand particles from
   different emitters and initial fills should share that library, while Water's
   SDF remains unchanged. Renaming/deleting an emitter must not change Sand output.
4. Leave distance fallback disabled and move the camera away: selected geometry
   must remain. Enable fallback, change Sand's distance/budget and verify only
   Sand's Splat representation changes. Test undo/redo and save/reopen.
5. With the app open, run externally on a paused disposable scene:

   ```powershell
   python scripts/test/rt_substance_splat_geometry_ipc.py --domain "Physics Domain 1" --substance Sand --other-substance Water
   ```

   This verifies independent generation, multi-object assignment, isolation,
   overlay semantics, rejected mutations and undo/redo. It restores appearance
   settings and deletes its own generated objects.
6. The standalone C++ `source_splat_geometry_test.cpp` includes substance-bank
   precedence, shared geometry across emitter identities and cached particles
   without source IDs, and SDF suspension. It was not compiled/executed by Codex.
