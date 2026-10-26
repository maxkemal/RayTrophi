# Matter gas-source volume streams

Matter gas flow sources now carry passive density and fuel ownership. Each
source uses its authored `timeline_uid`, exposed read-only as `producer_uid`
by both `rt.flow_source.list/get` and IPC `flow_source.list/get`. Names and
source-list indices are not identities. Legacy standalone Gas domains retain
their existing rendering path.

The live VDB publisher creates a separate named volume for every populated gas
source. The original `Vapor / Gas` stream becomes `Vapor / Unattributed Gas`
once source tracking starts. It contains initial/background density, vapor
from phase exchange, dust, and other producers without flow-source provenance.
An empty source retains its authoring; it acquires a VDB when it first has
positive render density. No source identity is inferred from old mixed gas.

## Transport and ownership

- Source deposition uses the existing resolved point, object-bounds or mesh
  surface weights, including animation and parenting. It adds no physical mass,
  energy, velocity, or temperature.
- Density and fuel contributions use the existing gas scalar advection kernels
  on Vulkan, with the same advected MAC velocity, boundary mode, solid mask,
  timestep and selected MacCormack correction. CPU uses the shared passive
  scalar reference. CPU sparse Matter gas with tracked sources uses the dense
  operator ordering so source transport and chemistry have the same stages.
- Transport limiter differences are normalized per cell against the physical
  density/fuel fields. Depletion and dissipation reduce contributions
  proportionally. Unattributed additions go to UID 0. Combustion smoke is
  distributed according to the mixed pre-burn fuel shares, including the
  unattributed fuel share. This is passive render provenance, not chemical
  species tracking or separate source-specific reaction/temperature solvers.
- Every published gas density partitions the authoritative gas field; the
  aggregate is not copied into each source. All source volumes read the same
  physical Kelvin field. Volume materials control optical appearance.
- Current contribution fields stay resident on Vulkan. Explicit VDB publication,
  diagnostic and cache consumers download snapshots. Source staging uploads
  wait before reusing staging buffers. Failed transport/snapshots reject rather
  than claiming a stale host image is current.

At most eight source UIDs plus the residual stream are retained per domain,
including deleted sources with surviving history. A conservative 256 MiB
provenance allocation budget is checked before source deposition. Fog field
preparation and its diagnostic snapshots retain their separate 128 MiB budgets.
The existing sixteen-volume render capacity still applies scene-wide.

## Shared UI, scripting and IPC operations

Use the existing VDB material editor, or the shared material service:

| Operation | Scripting | IPC |
| --- | --- | --- |
| Inspect streams | `rt.fluid.inspect_fog_fields(domain)` | `fluid.inspect_fog_fields` |
| List targets | `rt.volume.list_materials()` | `volume.list_materials` |
| Read material | `rt.volume.get_material(target)` | `volume.get_material` |
| Edit material/visibility | `rt.volume.set_material(target, **settings)` | `volume.set_material` with `target`, `settings` |

Canonical targets are `matter:<domain>:gas_source:<producer_uid>` and
`matter:<domain>:gas` for the residual. Inspection includes `producer_uid`,
`producer_name`, `material_target`, `vdb_id`, `published`, `visible`, and
`gas_max_abs_density_error`. Material validation and atomic error semantics
remain those of the existing volume material service; appearance edits do not
reset physical simulation. Legacy `gas:<domain>` material aliases address only
the residual field.

Source appearance and visibility serialize with the stable stream key in both
project paths. In-memory snapshots detach their live device view. Disk cache
v18 stores source UID/name and density/fuel images; readers validate counts,
dimensions, finite nonnegative values, duplicate UIDs, and the required residual
stream. Old caches still play as aggregate gas; resimulate from the beginning
to recover per-source history. Backend changes materialize before releasing
device storage. Reset/layout changes clear runtime provenance.

## Validation and user build checklist

Non-build checks: `scripts/test/check_gas_source_provenance_contracts.py`
checks randomized partition/nonnegativity/order independence, vapor fallback,
mixed-fuel smoke attribution, transport ordering, lifecycle/cache wiring,
script/IPC parity, shader bindings and project registration. These checks do
not execute C++ or GPU code.

The user performs C++ and simulation shader compilation; include the new
`sim_gas_source_partition.comp`. No application launch or compilation was
performed by Codex.

After building and reopening Studio:

1. Reset/resimulate the current Matter scene from its beginning. Confirm
   `Point Flow Source 4` and `Vapor / Unattributed Gas` appear independently in
   the VDB list when populated. Check each populated diagnostic field reports
   `published=true` and its own VDB ID. Check partition error near float epsilon.
2. Assign contrasting scattering/emission materials to two crossing gas
   sources. Compare the shared physical temperature in an overlap; density
   must not double. Repeat with fuel-only sources that generate smoke on burn.
3. Include Snow Fog, vapor and two gas sources, plus a liquid SDF and opaque
   collider. Verify all contributions, independent emission, shadowing and
   existing multiple-scattering behavior on supported render backends.
4. Hide/show and edit one canonical material target using UI, Python and IPC;
   read back through the other surfaces. Invalid settings must reject atomically.
5. Rewind/replay, save/reopen, bake/play v18, rename/disable/delete/recreate a
   source, remove the domain, and switch CPU/Vulkan. Surviving history must keep
   its source UID; a recreated source receives a different UID. Appearance must
   persist independently of empty/reborn volume instances.
6. Check isolated imported VDB/cloud/legacy Gas domains retain their old path.

Run live probes externally with `scripts/test/rt_ipc.py`, never through the
embedded script workspace. Live visual acceptance and performance remain
pending the user's build; no speedup or compiled-runtime correctness is claimed.

Read-only publication probe for the current scene, after resimulation:

```powershell
python -B scripts/test/rt_probe_gas_source_streams_ipc.py --domain "Physics Domain 1" --expect-source "Point Flow Source 4"
```

The general kernel-binding audit reports the same 53 existing findings with
both HEAD's table and the working table; the new kernel adds none. Descriptor
generation `--check` passes. `verify_descriptor_claims.py` cannot finish because
its existing enum parser calls `.split()` on an enum list; this verifier was
not changed as part of gas-source work.
