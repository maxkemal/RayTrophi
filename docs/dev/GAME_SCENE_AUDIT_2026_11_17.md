# Pastel play room: live inspection, 17 November 2026

Scene: `assets/scenes/pastel-play-room/pastel-play-room.rtp`, already open in
Studio. All probes ran from an external process through `scripts/test/rt_ipc.py`.
No C++ build, application launch or project save was performed.

## Measured findings

- 113 object names. All pass the available flat-mesh geometry validation.
- Originally 24 visible balloon meshes but only `1_Balloon_00..11` were bound to
  the game. The twelve `2_Balloon_*` meshes were visible static decorations.
  Screen picking iterates the demo's balloons, not scene colliders: an SDF cannot
  make an unbound decorative mesh respond to pop commands.
- Per user request, the live config now binds both sets: 24 resolved game meshes.
  Running/stopped state was preserved. This is binding/state verification, not
  a claim that all 24 screen positions have been visually clicked.
- The room's eight registered paint surfaces have nonzero valid UV atlas texels.
  Their UVs have no nonfinite coordinates, zero-area faces or out-of-unit faces;
  a 256x256 interior-sample overlap check found no overlapping samples. This
  sampling does not prove absence of subpixel overlap or prove droplet contact.
- Full environment inspection covers 93 sibling meshes belonging to 69 props
  after excluding game balloons and the 20 lamp/ceiling-light objects. It found
  **46 completely collapsed UV meshes on 34 second-import objects**. Existing
  registered-surface UV validity does not imply the entire room is paintable.
- The original eight-surface registration excludes many props. A collider alone
  does not make an object paintable. A surface alone does not guarantee droplets
  will collide with a thin shape at the chosen SDF resolution.

## SDF failure and current workaround

`2_Room_Wall Collider` (disabled duplicate) and
`2_Room_Wall PaintCollider` referenced the same flat node in one particle runtime.
`SceneData::publishCompletedSdfBakes` matched only node name, then moved the result
into the first matching collider and broke. The managed collider could therefore
remain uncooked, even after its own job finished. Increasing detail or waiting
would not fix the recipient mismatch.

The source condition now also matches the existing `sdf_bake_serial` shared token
to the bake's token. Jobs can publish only to their actual request owner, with
the existing request-version check still guarding stale cooks. No shader/layout
change. The oversized scene header received only this binding condition.

In the running old binary, the disabled duplicate was removed through the shared
collider API. The managed wall collider then cooked successfully at **128^3** in
all four droplet systems; demo configure succeeded with 24 balloons. The source
fix itself still requires a rebuild and its duplicate-collider regression.

## All-props authoring handoff

User requested all props paintable except lamps; balloons remain game visuals
because the core rejects a balloon visual also being a paint surface.
Plan: 69 props, 85 additional material surfaces, 55 shared slots requiring isolation.
Existing surfaces/settings remain. Additional slots use 512px color-only defaults.
This substantially expands paint/collider storage and needs live memory/timing
acceptance; no claim of zero additional paint cost is made.

Two optional extensions now share the existing material service in C++, Python
and IPC, without adding Player authoring permissions:

- `material.create(type, name, clone_from="")`: when supplied, clones a supported
  Principled/Volumetric source; type must match. Preserves material properties and
  texture bindings, creates an independent GPU-material record, returns a unique
  asset name. Missing/unsupported/mismatched sources fail before allocation.
- `material.assign(object_name, material_name, source_material_name="")`: a
  supplied source must exist on the object; remaps only that slot's flat SoA IDs.
  Omitting it preserves whole-object assignment. Existing geometry/backend sync
  and validation are reused. These bulk authoring operations remain non-undoable.

Commands after user rebuild and UV repair acceptance:

```powershell
python scripts/test/rt_room_authoring_regression.py --object 1_Table
python scripts/scenes/prepare_paint_room.py
python scripts/scenes/prepare_paint_room.py --apply
```

Use a disposable room for the regression; its clone material remains in the
registry. The setup command refuses old binaries before mutation, preserves
multi-material slots, checks collapsed UVs before edits, isolates shared slots,
adds missing surfaces and missing managed SDF colliders, waits for readiness and
reconfigures without starting or saving. Existing good colliders are preserved.
The fixture and full setup have not run against rebuilt binaries.

UV repair command is `scripts/scenes/repair_room_uvs.py`. Automatic approval review
initially rejected the broad live UV rewrite. The user then explicitly approved
re-unwrapping all 46 meshes. **46/46 repairs subsequently completed successfully**
using the existing UV service. Original layouts are retained in
`GAME_ROOM_UV_REPAIR_2026_11_17.json`; the script refuses to overwrite that backup.
No project save was performed.

Post-repair verification: all 93 environment sibling meshes have zero degenerate
UV faces, zero nonfinite UV coordinates, zero out-of-unit UV faces and zero
overlap samples in the 256x256 interior test. All 113 objects still pass the
available flat-mesh geometry validation. This does not prove subpixel overlap
absence or successful live paint contact on every prop.

The post-edit demo was unprepared and two armchair collider object bindings were
empty during the final inspection. Their original modes/source bindings were
restored and the 24-balloon config was prepared again. Final measured state:
prepared=true, 24/24 visuals resolved, error empty; wall collider ready in all four
systems at 128^3. `GAME_ROOM_AFTER_UV_2026_11_17.json` records that final state.
The full room's additional paint registrations still require the new build and
the all-props setup command above.

Validation so far: IPC capability/descriptor audit PASS; Python scripts parse;
focused whitespace check PASS. Rebuild Studio/Player broadly due shared headers.
Then validate duplicate collider identity, material-slot isolation, UV repair,
paint contact on thin details, screen popping, save/reopen and GPU memory/timing.
